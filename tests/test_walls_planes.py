"""Tests for the Walls_Step's second cross-frame stages: splatpipe/walls/planes.py (walls spec,
Req 13, 15.1 to 15.4, 16.2, 16.3, 16.6, 22.1, A10, A12, C11).

The unit tests (sub-task 8.2):
- robust_plane runs on a few hundred hand-made points on a slanted plane.
- world_planes runs on the box room of tests/camera_files.py, all ten frames as Trusted_Frames, at
  its camera's own 128 x 96, the size its table of Facet-sized surfaces is worked out for. Each
  frame's Facets come from cut_facets on its true normals and inverse depths, and its Fused_Depth
  is its true depth.
- projection_fallback and plane_registration run on one box-room frame at a Prior_Size of 64 x 48,
  half its camera's, whose Start_Depth is its true depth through a Bas_Relief_Change written here.
- flatten_frame runs on one box-room frame at 64 x 48, its Facets made here from box_surface, with
  some pixels moved a little off their wall and some moved further.

Every oracle is written here and never calls the function it checks. The settings come from the
real config.toml, read by path, through read_settings on its `[walls]` and `[walls.default]`, and
dataclasses.replace changes the one a test sets. Everything runs on the CPU on a few thousand
numbers (Req 22.1).
"""

from __future__ import annotations

import dataclasses
import functools
import math
import tomllib
from pathlib import Path

import numpy as np
import pytest

from splatpipe.train.colmap import Camera
from splatpipe.walls import planes
from splatpipe.walls.facets import Facet, cut_facets
from splatpipe.walls.planes import (FallbackDepth, WorldPlane, fallback_medians,
                                    flatten_frame, plane_registration, projection_fallback,
                                    register_tolerances, robust_fit, robust_plane, world_planes)
from splatpipe.walls.settings import WallsSettings, read_settings

from camera_files import (BOX_BACK, BOX_FLOOR, BOX_FRAMES, BOX_PLANES, BOX_SIDE,
                          BOX_SMALL_PATCHES, box_camera, box_covered, box_depth, box_normal,
                          box_pose, box_surface)

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_WALLS = tomllib.loads((REPO_ROOT / "config.toml").read_text(encoding="utf-8"))["walls"]


def real_settings(preset: str = "default") -> WallsSettings:
    """The real `[walls]` settings with the Preset table `preset`, as the step reads them."""
    shared = {key: value for key, value in CONFIG_WALLS.items() if not isinstance(value, dict)}
    return read_settings({**shared, **CONFIG_WALLS[preset]}, "walls", preset)


SETTINGS = real_settings()
FULL_SIZE = (128, 96)   # (w, h), the box room's camera, for the Facet cut
PRIOR_SIZE = (64, 48)   # (w, h), half of it, for the fallbacks and the flatten
# The fallbacks' frame: frame 7 sees the back wall and the floor, and at 64 x 48 a few pixels
# past the floor's edge that aren't Covered_Pixels.
FALLBACK_FRAME = 7
# The flatten's frame: frame 5 sees the back wall and the floor, each over about half the frame.
FLATTEN_FRAME = 5
SURFACES = (BOX_BACK, BOX_FLOOR, BOX_SIDE)


# ---------------------------------------------------------------------------------------------
# Oracles: the geometry written out here.


def oracle_coords(camera: Camera, size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Each pixel's Normalised_Coordinates at `size` (w, h): its centre in the camera's own pixels,
    (i + 0.5)·W / w across and (j + 0.5)·H / h down, less (cx, cy), over (fx, fy)."""
    width, height = size
    columns, rows = np.meshgrid(np.arange(width), np.arange(height))
    u = (columns + 0.5) * camera.width / width
    v = (rows + 0.5) * camera.height / height
    return (u - camera.cx) / camera.fx, (v - camera.cy) / camera.fy


def oracle_change(inverse: np.ndarray, change: tuple[float, ...], m: float, x: np.ndarray,
                  y: np.ndarray) -> np.ndarray:
    """The inverse depth a·inverse + m·(b·x + c·y + d) of the Bas_Relief_Change (a, b, c, d)."""
    a, b, c, d = change
    return a * inverse + m * (b * x + c * y + d)


def oracle_changed_normal(normal: np.ndarray, z: np.ndarray, x: np.ndarray, y: np.ndarray,
                          change: tuple[float, ...], m: float) -> np.ndarray:
    """The Changed_Normal a·n + m·z·(n·r)·(b, c, d), unit length and facing the camera."""
    a, b, c, d = change
    along = m * z * (normal[..., 0] * x + normal[..., 1] * y + normal[..., 2])
    raw = a * normal + along[..., np.newaxis] * np.array([b, c, d])
    unit = raw / np.linalg.norm(raw, axis=-1, keepdims=True)
    facing = np.where(unit[..., 0] * x + unit[..., 1] * y + unit[..., 2] > 0, -1.0, 1.0)
    return unit * facing[..., np.newaxis]


def camera_normal(world_normal: np.ndarray, index: int) -> np.ndarray:
    """A world normal in box-room frame `index`'s camera, R·n."""
    return box_pose(index)[:3, :3] @ np.asarray(world_normal, dtype=np.float64)


def angle_degrees(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """The angle between two unit normals, (..., 3), in degrees, whichever way they face (C11)."""
    dot = np.abs(np.sum(first * second, axis=-1))
    return np.degrees(np.arccos(np.clip(dot, 0.0, 1.0)))


def turned(normal: np.ndarray, axis: tuple[float, float, float], degrees: float) -> np.ndarray:
    """`normal` turned by `degrees` about the unit `axis`, by Rodrigues' formula."""
    k = np.asarray(axis, dtype=np.float64)
    k = k / np.linalg.norm(k)
    v = np.asarray(normal, dtype=np.float64)
    theta = math.radians(degrees)
    return (v * math.cos(theta) + np.cross(k, v) * math.sin(theta)
            + k * float(k @ v) * (1 - math.cos(theta)))


def true_planes() -> list[WorldPlane]:
    """The box room's World_Planes as they truly are, from BOX_PLANES, in its surfaces' order."""
    return [WorldPlane(normal=np.array(normal, dtype=np.float64), offset=float(offset),
                       frames=frozenset(), pixels=0) for normal, offset in BOX_PLANES]


def matches_surface(plane: WorldPlane, surface: int) -> bool:
    """Whether `plane` is box-room surface `surface`'s true plane, to 10⁻⁹, either way it faces
    (C11): its normal along the surface's, and its offset, with the normals facing the same way,
    the surface's."""
    normal, offset = BOX_PLANES[surface]
    dot = float(np.dot(plane.normal, normal))
    return abs(dot) >= 1 - 1e-12 and abs(math.copysign(1.0, dot) * plane.offset - offset) <= 1e-9


def surface_of(plane: WorldPlane) -> int:
    """The box-room surface whose true plane `plane` is, and fail when it is none of them."""
    found = [surface for surface in SURFACES if matches_surface(plane, surface)]
    assert len(found) == 1, f"plane {plane.normal}, {plane.offset} is no box-room surface"
    return found[0]


# ---------------------------------------------------------------------------------------------
# The box room as Trusted_Frames.


def frame_name(index: int) -> str:
    return f"{index:06d}"


@functools.cache
def box_facets() -> dict[str, list[Facet]]:
    """Each box-room frame's Facets at its camera's own 128 x 96, cut from its true normals and
    inverse depths with the real settings."""
    return {frame_name(index): cut_facets(box_normal(index, FULL_SIZE),
                                          1.0 / box_depth(index, FULL_SIZE),
                                          box_covered(index, FULL_SIZE), SETTINGS)
            for index in range(BOX_FRAMES)}


@functools.cache
def box_inputs() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Each box-room frame's Fused_Depth, its true depth at 128 x 96, and its Image_Pose."""
    depths = {frame_name(index): box_depth(index, FULL_SIZE) for index in range(BOX_FRAMES)}
    poses = {frame_name(index): box_pose(index) for index in range(BOX_FRAMES)}
    return depths, poses


def box_world_planes(settings: WallsSettings = SETTINGS) -> list[WorldPlane]:
    depths, poses = box_inputs()
    return world_planes(box_facets(), depths, poses, box_camera(), settings)


@functools.cache
def facet_surfaces() -> dict[int, tuple[frozenset[str], int]]:
    """The oracle of the merge: for each surface, the frames whose Facets lie on it and the sum
    of those Facets' pixels, from box_surface at each Facet's pixels."""
    found: dict[int, tuple[set[str], int]] = {surface: (set(), 0) for surface in SURFACES}
    for index in range(BOX_FRAMES):
        surface_map = box_surface(index, FULL_SIZE)
        for facet in box_facets()[frame_name(index)]:
            on = np.unique(surface_map[facet.pixels[:, 0], facet.pixels[:, 1]])
            assert on.size == 1, f"a Facet of frame {index} spans surfaces {on}"
            frames, pixels = found[int(on[0])]
            frames.add(frame_name(index))
            found[int(on[0])] = (frames, pixels + facet.count)
    return {surface: (frozenset(frames), pixels) for surface, (frames, pixels) in found.items()}


# ---------------------------------------------------------------------------------------------
# robust_plane (Req 13.2).


def slanted_points(count: int, seed: int) -> tuple[np.ndarray, np.ndarray, float]:
    """`count` points spread at random over a 4 x 4 square of the slanted plane n·X = δ, as
    (points (count, 3), n, δ)."""
    rng = np.random.default_rng(seed)
    normal = np.array([0.3, -0.5, 0.81])
    normal = normal / np.linalg.norm(normal)
    across = np.cross(normal, [1.0, 0.0, 0.0])
    across = across / np.linalg.norm(across)
    down = np.cross(normal, across)
    origin = np.array([1.5, -2.0, 6.0])
    a, b = rng.uniform(-2.0, 2.0, size=(2, count))
    points = origin + a[:, np.newaxis] * across + b[:, np.newaxis] * down
    return points, normal, float(normal @ origin)


def assert_plane(normal: np.ndarray, offset: float, true_normal: np.ndarray, true_offset: float,
                 tolerance: float) -> None:
    """Assert that (normal, offset) is the plane (true_normal, true_offset) within `tolerance`,
    whichever way it faces (C11)."""
    dot = float(normal @ true_normal)
    assert math.isclose(float(np.linalg.norm(normal)), 1.0, abs_tol=1e-12)
    assert 1 - abs(dot) <= tolerance
    assert abs(math.copysign(1.0, dot) * offset - true_offset) <= tolerance


def test_robust_plane_fits_an_exact_plane() -> None:
    points, normal, offset = slanted_points(200, seed=1)
    fitted, fitted_offset = robust_plane(points, SETTINGS.tukey_constant,
                                         SETTINGS.tukey_iterations)
    assert_plane(fitted, fitted_offset, normal, offset, 1e-12)
    assert np.max(np.abs(points @ fitted - fitted_offset)) <= 1e-12


def test_robust_plane_keeps_every_point_of_a_perfect_plane() -> None:
    points, _, _ = slanted_points(200, seed=2)
    _, _, weights = robust_fit(points, SETTINGS.tukey_constant, SETTINGS.tukey_iterations)
    assert np.array_equal(weights, np.ones(200))


def test_robust_plane_still_fits_with_one_point_far_off() -> None:
    # The first fit weights every point 1, so the moved point must not become the unweighted
    # covariance's smallest direction's rival. 20 units off, its share of the variance along the
    # normal is about 20² / 500 = 0.8, below the cloud's in-plane variance 4² / 12 ≈ 1.33, so the
    # first fit's normal stays close to the plane's and the reweighting then drops the point. On
    # this cloud, one 30 units off (30² / 500 = 1.8, above 1.33) already turns the first fit away,
    # and the reweighting doesn't bring it back.
    points, normal, offset = slanted_points(500, seed=3)
    points[17] = points[17] + 20.0 * normal
    fitted, fitted_offset, weights = robust_fit(points, SETTINGS.tukey_constant,
                                                SETTINGS.tukey_iterations)
    assert_plane(fitted, fitted_offset, normal, offset, 1e-6)
    assert weights[17] == 0.0
    plane_normal, plane_offset = robust_plane(points, SETTINGS.tukey_constant,
                                              SETTINGS.tukey_iterations)
    assert np.array_equal(plane_normal, fitted) and plane_offset == fitted_offset


def test_robust_plane_refuses_fewer_than_three_points() -> None:
    with pytest.raises(ValueError):
        robust_plane(np.zeros((2, 3)), SETTINGS.tukey_constant, SETTINGS.tukey_iterations)


# ---------------------------------------------------------------------------------------------
# world_planes (Req 13.1, 13.3, 13.4, A10, C11).


def test_the_box_room_facets_are_as_its_table_says() -> None:
    # The precondition of the merge tests: each surface has Facets in the frames its table names,
    # and the small patches are no Facet.
    found = facet_surfaces()
    assert found[BOX_BACK][0] == {frame_name(i) for i in (4, 5, 6, 7, 8, 9)}
    assert found[BOX_FLOOR][0] == {frame_name(i) for i in (0, 1, 2, 3, 4, 5, 6, 7, 9)}
    assert found[BOX_SIDE][0] == {frame_name(i) for i in (0, 1, 2, 3)}
    for index, surface in BOX_SMALL_PATCHES.items():
        assert frame_name(index) not in found[surface][0]


def test_world_planes_keeps_the_box_rooms_three_planes() -> None:
    kept = box_world_planes()
    assert len(kept) == 3
    found = facet_surfaces()
    surfaces = [surface_of(plane) for plane in kept]
    assert sorted(surfaces) == sorted(SURFACES)
    for plane, surface in zip(kept, surfaces):
        # Each surface's Facets, from every frame that has one there, merge into this one plane.
        assert plane.frames == found[surface][0]
        assert plane.pixels == found[surface][1]
        assert math.isclose(float(np.linalg.norm(plane.normal)), 1.0, abs_tol=1e-12)


def test_world_planes_doesnt_change_with_the_frames_order() -> None:
    depths, poses = box_inputs()
    backwards = dict(reversed(list(box_facets().items())))
    again = world_planes(backwards, depths, poses, box_camera(), SETTINGS)
    for first, second in zip(box_world_planes(), again, strict=True):
        assert np.array_equal(first.normal, second.normal)
        assert first.offset == second.offset
        assert (first.frames, first.pixels) == (second.frames, second.pixels)


def test_world_planes_drops_a_plane_from_too_few_frames() -> None:
    # The side wall has Facets in 4 frames, the back wall in 6 and the floor in 9.
    side_frames = len(facet_surfaces()[BOX_SIDE][0])
    kept = box_world_planes(dataclasses.replace(SETTINGS, plane_min_frames=side_frames))
    assert sorted(surface_of(plane) for plane in kept) == sorted(SURFACES)
    kept = box_world_planes(dataclasses.replace(SETTINGS, plane_min_frames=side_frames + 1))
    assert sorted(surface_of(plane) for plane in kept) == sorted([BOX_BACK, BOX_FLOOR])


def test_world_planes_drops_a_plane_below_the_min_share() -> None:
    # The share is of the largest plane with Facets from enough frames, worked out here.
    found = facet_surfaces()
    largest = max(pixels for _, pixels in found.values())
    side_share = found[BOX_SIDE][1] / largest
    assert side_share < min(found[BOX_BACK][1], found[BOX_FLOOR][1]) / largest
    kept = box_world_planes(dataclasses.replace(SETTINGS, plane_min_share=side_share * 0.999))
    assert sorted(surface_of(plane) for plane in kept) == sorted(SURFACES)
    kept = box_world_planes(dataclasses.replace(SETTINGS, plane_min_share=side_share * 1.001))
    assert sorted(surface_of(plane) for plane in kept) == sorted([BOX_BACK, BOX_FLOOR])


def test_world_planes_merges_facets_whose_normals_point_opposite_ways(
        monkeypatch: pytest.MonkeyPatch) -> None:
    # A plane normal's sign is arbitrary (C11). This stand-in for robust_plane gives the real fit
    # turned to face one way on its even calls and the other way on its odd ones, measured against
    # a direction no box-room normal is square to, so neighbouring Facets on one surface get
    # opposite normals, and each refit after a join comes back facing either way too.
    real = robust_plane
    reference = np.array([0.3, 0.5, 0.8])
    fits: list[np.ndarray] = []

    def flipping(points: np.ndarray, constant: float, iterations: int) -> tuple[np.ndarray, float]:
        normal, offset = real(points, constant, iterations)
        want = 1.0 if len(fits) % 2 == 0 else -1.0
        if math.copysign(1.0, float(normal @ reference)) != want:
            normal, offset = -normal, -offset
        fits.append(normal)
        return normal, offset

    monkeypatch.setattr(planes, "robust_plane", flipping)
    kept = box_world_planes()

    facet_count = sum(len(facets) for facets in box_facets().values())
    facet_fits = fits[:facet_count]   # the Facet_Planes come first, before any merge
    for surface in SURFACES:
        signs = {math.copysign(1.0, float(fit @ BOX_PLANES[surface][0]))
                 for fit in facet_fits if 1 - abs(float(fit @ BOX_PLANES[surface][0])) < 1e-9}
        if len(facet_surfaces()[surface][0]) > 1:
            assert signs == {-1.0, 1.0}, f"surface {surface}'s Facets all faced one way"

    found = facet_surfaces()
    assert sorted(surface_of(plane) for plane in kept) == sorted(SURFACES)
    for plane in kept:
        assert (plane.frames, plane.pixels) == found[surface_of(plane)]


# ---------------------------------------------------------------------------------------------
# The Projection_Fallback (Req 15.1, 15.2, A12).

# The change the medians are of the Start_Depth: a scale with a tilt across and down.
MEDIANS_CHANGE = (1.1, 0.04, -0.03, 0.05)


def fallback_frame() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The fallbacks' frame at 64 x 48: its true depth, its true normals, its Covered_Pixels and
    its Normalised_Coordinates (x, y)."""
    depth = box_depth(FALLBACK_FRAME, PRIOR_SIZE)
    normal = box_normal(FALLBACK_FRAME, PRIOR_SIZE)
    covered = box_covered(FALLBACK_FRAME, PRIOR_SIZE)
    x, y = oracle_coords(box_camera(), PRIOR_SIZE)
    return depth, normal, covered, x, y


def test_the_fallback_frame_has_pixels_that_arent_covered() -> None:
    _, _, covered, _, _ = fallback_frame()
    assert 0 < np.count_nonzero(~covered) < covered.size


def test_projection_fallback_gets_the_plane_back_from_agreeing_medians() -> None:
    start, normal, covered, x, y = fallback_frame()
    m = float(np.median(1.0 / start[covered]))
    truth = 1.0 / oracle_change(1.0 / start, MEDIANS_CHANGE, m, x, y)
    # Every other Covered_Pixel, a checkerboard spread over both surfaces, gets three Projected
    # Values: two agreeing 1% either side of the truth and a stray 50% further off.
    rows, columns = np.indices(start.shape)
    with_median = covered & ((rows + columns) % 2 == 0)
    projected = np.full(start.shape + (3,), np.nan)
    projected[with_median] = np.stack([truth[with_median] * 0.99, truth[with_median] * 1.01,
                                       truth[with_median] * 1.5], axis=-1)
    medians = fallback_medians(projected, SETTINGS)
    assert np.array_equal(np.isfinite(medians), with_median)
    assert np.allclose(medians[with_median], truth[with_median], rtol=1e-12, atol=0)
    share = np.count_nonzero(with_median) / np.count_nonzero(covered)
    assert share >= SETTINGS.projection_min_coverage

    result = projection_fallback(medians, start, normal, covered, m, box_camera(), SETTINGS)
    assert isinstance(result, FallbackDepth)
    assert result.m == m
    others = covered & ~with_median
    assert np.array_equal(result.depth[with_median], medians[with_median])
    assert np.allclose(result.depth[others], truth[others], rtol=1e-9, atol=0)
    assert np.all(np.isnan(result.depth[~covered]))
    want = oracle_changed_normal(normal, start, x, y, MEDIANS_CHANGE, m)
    assert result.normal.dtype == np.float32
    assert np.allclose(result.normal[covered], want[covered], rtol=0, atol=1e-6)
    assert np.all(np.isnan(result.normal[~covered]))
    # The fitted change gives the truth's inverse depths at every Covered_Pixel.
    fitted = oracle_change(1.0 / start, result.change, m, x, y)
    assert np.allclose(fitted[covered], 1.0 / truth[covered], rtol=1e-9, atol=0)


def test_projection_fallback_gives_none_below_the_min_coverage() -> None:
    start, normal, covered, x, y = fallback_frame()
    m = float(np.median(1.0 / start[covered]))
    truth = 1.0 / oracle_change(1.0 / start, MEDIANS_CHANGE, m, x, y)
    total = np.count_nonzero(covered)
    enough = next(k for k in range(total + 1) if k / total >= SETTINGS.projection_min_coverage)
    order = np.flatnonzero(covered.ravel())   # the Covered_Pixels in row-major order

    def medians_on(count: int) -> np.ndarray:
        medians = np.full(start.size, np.nan)
        medians[order[:count]] = truth.ravel()[order[:count]]
        return medians.reshape(start.shape)

    assert projection_fallback(medians_on(enough - 1), start, normal, covered, m, box_camera(),
                               SETTINGS) is None
    assert projection_fallback(medians_on(enough), start, normal, covered, m, box_camera(),
                               SETTINGS) is not None


# ---------------------------------------------------------------------------------------------
# The Plane_Registration (Req 15.3, 15.4, A12).

# The change that takes the truth to the Start_Depth: a scale of about 1 / 0.88 in depth, so the
# Start_Depth is about 14% too far, with a tilt across and down.
START_CHANGE = (0.88, 0.03, -0.02, 0.0)


def registration_start(change: tuple[float, ...] = START_CHANGE
                       ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                                  np.ndarray, float]:
    """The fallbacks' frame with its truth taken through `change` as its Start_Depth: (truth,
    true normals, Start_Depth, its normals in float64, Covered_Pixels, its normals in float32 as
    the Depth_Prior holds them, m), m the Start_Depth's Median_Inverse_Depth."""
    truth, true_normal, covered, x, y = fallback_frame()
    m_truth = float(np.median(1.0 / truth[covered]))
    start = 1.0 / oracle_change(1.0 / truth, change, m_truth, x, y)
    start_normal = oracle_changed_normal(true_normal, truth, x, y, change, m_truth)
    start = np.where(covered, start, np.nan)
    start_normal = np.where(covered[..., np.newaxis], start_normal, np.nan)
    m = float(np.median(1.0 / start[covered]))
    return truth, true_normal, start, start_normal, covered, start_normal.astype(np.float32), m


def oracle_tolerances() -> list[float]:
    """The Plane_Registration's tolerance in each pass, falling geometrically from
    register_start_tolerance to register_end_tolerance: each pass's is the one before's times the
    same ratio, (end / start)^(1 / (passes − 1))."""
    start, end = SETTINGS.register_start_tolerance, SETTINGS.register_end_tolerance
    passes = SETTINGS.register_passes
    ratio = (end / start) ** (1.0 / (passes - 1))
    tolerances = [start]
    for _ in range(passes - 1):
        tolerances.append(tolerances[-1] * ratio)
    return tolerances


def test_register_tolerances_fall_geometrically() -> None:
    want = oracle_tolerances()
    got = register_tolerances(SETTINGS.register_start_tolerance,
                              SETTINGS.register_end_tolerance, SETTINGS.register_passes)
    assert got == pytest.approx(want, rel=1e-12)
    assert got[0] == SETTINGS.register_start_tolerance
    assert got[-1] == pytest.approx(SETTINGS.register_end_tolerance, rel=1e-12)
    assert register_tolerances(0.5, 0.03, 1) == [0.5]


def test_plane_registration_registers_a_start_off_by_a_scale_and_a_tilt() -> None:
    truth, true_normal, start, start_normal, covered, start_normal32, m = registration_start()
    _, _, _, x, y = fallback_frame()
    tolerances = oracle_tolerances()
    # The first pass's tolerance takes in every Covered_Pixel's own wall, and the last pass's
    # would take in none of them: the passes have the whole way to go.
    error = np.abs(truth[covered] / start[covered] - 1.0)
    assert np.max(error) <= tolerances[0]
    assert np.min(error) > tolerances[-1]
    # Each start normal is within register_max_degrees of its own wall's and past it from every
    # other wall's, so only its own wall matches.
    surface = box_surface(FALLBACK_FRAME, PRIOR_SIZE)
    for number in SURFACES:
        wall = camera_normal(BOX_PLANES[number][0], FALLBACK_FRAME)
        angle = angle_degrees(start_normal, wall)
        assert np.all(angle[covered & (surface == number)] < SETTINGS.register_max_degrees)
        assert np.all(angle[covered & (surface != number)] > SETTINGS.register_max_degrees)

    result = plane_registration(true_planes(), start, start_normal32, covered, m,
                                box_pose(FALLBACK_FRAME), box_camera(), SETTINGS)
    assert isinstance(result, FallbackDepth)
    assert result.m == m
    assert np.allclose(result.depth[covered], truth[covered], rtol=1e-6, atol=0)
    assert np.all(np.isnan(result.depth[~covered]))
    assert result.normal.dtype == np.float32
    assert np.allclose(result.normal[covered], true_normal[covered], rtol=0, atol=1e-5)
    fitted = 1.0 / oracle_change(1.0 / start, result.change, m, x, y)
    assert np.allclose(result.depth[covered], fitted[covered], rtol=1e-9, atol=0)


def test_plane_registration_gives_none_with_no_world_plane() -> None:
    _, _, start, _, covered, start_normal32, m = registration_start()
    assert plane_registration([], start, start_normal32, covered, m, box_pose(FALLBACK_FRAME),
                              box_camera(), SETTINGS) is None


def test_plane_registration_gives_none_with_too_few_pixels_matched() -> None:
    _, _, start, _, covered, start_normal32, m = registration_start()
    # In the first pass every Covered_Pixel matches its own wall, as the registration test shows,
    # so register_min_pixels one above their number leaves the pass short.
    matched = int(np.count_nonzero(covered))
    for minimum, registers in ((matched, True), (matched + 1, False)):
        settings = dataclasses.replace(SETTINGS, register_min_pixels=minimum)
        result = plane_registration(true_planes(), start, start_normal32, covered, m,
                                    box_pose(FALLBACK_FRAME), box_camera(), settings)
        assert (result is not None) == registers
    # A Start_Depth 1 / 0.4 times too far has |z_plane / z − 1| = 0.6 at every pixel, past the
    # first pass's tolerance, so no pixel matches.
    far_change = (0.4, 0.0, 0.0, 0.0)
    assert 1.0 - far_change[0] > oracle_tolerances()[0]
    _, _, far, _, _, far_normal32, far_m = registration_start(far_change)
    assert plane_registration(true_planes(), far, far_normal32, covered, far_m,
                              box_pose(FALLBACK_FRAME), box_camera(), SETTINGS) is None


# ---------------------------------------------------------------------------------------------
# The flatten (Req 16.2, 16.3, 16.6, C11).

# Of each 10 pixels along the pattern (row + 3·column) mod 10, those at 0 and 1 are moved 1% off
# their wall, farther and nearer, inside flatten_tolerance, and those at 2 and 3 6%, past it. Six
# in ten stay on the wall, and the moves go both ways, so the first, unweighted fit of the robust
# normal isn't pulled off the wall, and the Tukey reweighting then drops every moved pixel.
NEAR_OFF, FAR_OFF = 1.01, 1.06
PATTERN = 10
NEAR_PIXELS, FAR_PIXELS = (0, 1), (2, 3)
# Every pixel's normal going in: one direction, unlike every wall's, so a pixel that moves shows.
GIVEN_NORMAL = np.array([0.1, 0.2, -0.97]) / np.linalg.norm([0.1, 0.2, -0.97])


def flatten_pattern(shape: tuple[int, int]) -> np.ndarray:
    """Each pixel's place in the pattern, (row + 3·column) mod PATTERN."""
    rows, columns = np.indices(shape)
    return (rows + 3 * columns) % PATTERN


def flatten_inputs() -> tuple[list[Facet], np.ndarray, np.ndarray, np.ndarray]:
    """The flatten's frame at 64 x 48: its back-wall and floor Facets, made from box_surface, its
    depth with some pixels moved off their wall, its normals going in and its surfaces."""
    surface = box_surface(FLATTEN_FRAME, PRIOR_SIZE)
    truth = box_depth(FLATTEN_FRAME, PRIOR_SIZE)
    facets = []
    for number in (BOX_BACK, BOX_FLOOR):
        pixels = np.argwhere(surface == number)
        facets.append(Facet(pixels=pixels, normal=camera_normal(BOX_PLANES[number][0],
                                                                FLATTEN_FRAME),
                            m=float(np.median(1.0 / truth[surface == number]))))
    pattern = flatten_pattern(truth.shape)
    factor = np.ones(truth.shape)
    factor[pattern == 0], factor[pattern == 1] = NEAR_OFF, 1.0 / NEAR_OFF
    factor[pattern == 2], factor[pattern == 3] = FAR_OFF, 1.0 / FAR_OFF
    depth = truth * factor
    normal = np.broadcast_to(GIVEN_NORMAL, truth.shape + (3,)).astype(np.float32)
    return facets, depth, normal, surface


def back_plane_turned(degrees: float) -> WorldPlane:
    """The back wall's World_Plane with its normal turned by `degrees` about a slanted axis."""
    normal = turned(np.array(BOX_PLANES[BOX_BACK][0]), (1.0, 0.4, 0.0), degrees)
    return WorldPlane(normal=normal, offset=float(normal @ [-0.75, -1.0, 4.0]),
                      frames=frozenset({frame_name(4), frame_name(5), frame_name(6)}),
                      pixels=10_000)


def oracle_flatten(facet: Facet, depth: np.ndarray, normal: np.ndarray, plane_normal: np.ndarray,
                   x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The flatten of one Facet onto the camera-frame normal `plane_normal`: the offset is the
    median of n·X over its pixels, each pixel's plane depth δ / (n·r) and its share off it,
    |z / z_plane − 1|. Gives (which pixels move, (count,) bool, the plane depths, and the normal
    facing the camera at each pixel, (count, 3))."""
    rows, columns = facet.pixels[:, 0], facet.pixels[:, 1]
    z, px, py = depth[rows, columns], x[rows, columns], y[rows, columns]
    along = plane_normal[0] * px + plane_normal[1] * py + plane_normal[2]
    offset = float(np.median(z * along))
    z_plane = offset / along
    off = np.abs(z / z_plane - 1.0)
    # Every pixel is well away from the tolerance's edge, so rounding can't change what moves.
    assert np.min(np.abs(off - SETTINGS.flatten_tolerance)) > 1e-6
    facing = np.where(along[:, np.newaxis] > 0, -plane_normal, plane_normal)
    return off <= SETTINGS.flatten_tolerance, z_plane, facing


def assert_flattened(facet: Facet, plane_normal: np.ndarray, depth: np.ndarray,
                     normal: np.ndarray, new_depth: np.ndarray, new_normal: np.ndarray) -> None:
    """Assert that `facet`'s pixels were flattened onto `plane_normal`: each one within the
    tolerance takes its plane depth and the normal facing the camera, and each other keeps its
    depth and its normal, bit for bit. Also that some pixels move and some don't."""
    x, y = oracle_coords(box_camera(), PRIOR_SIZE)
    moves, z_plane, facing = oracle_flatten(facet, depth, normal, plane_normal, x, y)
    assert 0 < np.count_nonzero(moves) < moves.size
    rows, columns = facet.pixels[:, 0], facet.pixels[:, 1]
    got_depth, got_normal = new_depth[rows, columns], new_normal[rows, columns]
    assert np.allclose(got_depth[moves], z_plane[moves], rtol=1e-9, atol=0)
    assert np.allclose(got_normal[moves], facing[moves], rtol=0, atol=1e-6)
    assert np.array_equal(got_depth[~moves], depth[rows, columns][~moves])
    assert np.array_equal(got_normal[~moves], normal[rows, columns][~moves])


def test_flatten_takes_a_world_planes_normal_within_flatten_degrees() -> None:
    facets, depth, normal, surface = flatten_inputs()
    # The back wall's World_Plane turned 3°, within flatten_degrees (5°) of the Facet's own robust
    # normal, the wall's true one. The floor's Facet is 90° from it and keeps its own.
    tilt = 3.0
    assert tilt < SETTINGS.flatten_degrees
    plane = back_plane_turned(tilt)
    new_depth, new_normal = flatten_frame(facets, [plane], depth, normal,
                                          box_pose(FLATTEN_FRAME), box_camera(), SETTINGS)
    assert new_normal.dtype == np.float32
    back, floor = facets
    assert_flattened(back, camera_normal(plane.normal, FLATTEN_FRAME), depth, normal,
                     new_depth, new_normal)
    assert_flattened(floor, camera_normal(BOX_PLANES[BOX_FLOOR][0], FLATTEN_FRAME), depth,
                     normal, new_depth, new_normal)
    # The back wall's pixels that took the World_Plane's normal are 3° off their true one.
    moved = back.mask(depth.shape) & (new_depth != depth)
    true_back = camera_normal(BOX_PLANES[BOX_BACK][0], FLATTEN_FRAME)
    assert np.allclose(angle_degrees(new_normal[moved].astype(np.float64), true_back), tilt,
                       atol=1e-3)


def test_flatten_takes_the_facets_own_normal_past_flatten_degrees() -> None:
    facets, depth, normal, surface = flatten_inputs()
    # The back wall's World_Plane turned 8°, past flatten_degrees, so the back wall's Facet takes
    # its own robust normal, the wall's true one, which the moved pixels don't sway.
    tilt = 8.0
    assert tilt > SETTINGS.flatten_degrees
    new_depth, new_normal = flatten_frame(facets, [back_plane_turned(tilt)], depth, normal,
                                          box_pose(FLATTEN_FRAME), box_camera(), SETTINGS)
    for facet, number in zip(facets, (BOX_BACK, BOX_FLOOR)):
        assert_flattened(facet, camera_normal(BOX_PLANES[number][0], FLATTEN_FRAME), depth,
                         normal, new_depth, new_normal)
    # On its own wall, the flatten puts the 1% pixels back and leaves the 6% ones.
    truth = box_depth(FLATTEN_FRAME, PRIOR_SIZE)
    pattern = flatten_pattern(depth.shape)
    near, far = np.isin(pattern, NEAR_PIXELS), np.isin(pattern, FAR_PIXELS)
    assert np.allclose(new_depth[near], truth[near], rtol=1e-9, atol=0)
    assert np.array_equal(new_depth[far], depth[far])
    assert np.array_equal(new_normal[far], normal[far])


def test_flatten_leaves_pixels_off_every_facet_as_they_were() -> None:
    facets, depth, normal, surface = flatten_inputs()
    keep = np.ones(depth.shape, dtype=bool)
    keep[facets[0].pixels[:, 0], facets[0].pixels[:, 1]] = False   # only the back wall's Facet
    new_depth, new_normal = flatten_frame(facets[:1], [], depth, normal,
                                          box_pose(FLATTEN_FRAME), box_camera(), SETTINGS)
    assert np.array_equal(new_depth[keep], depth[keep])
    assert np.array_equal(new_normal[keep], normal[keep])
    assert np.any(new_depth[~keep] != depth[~keep])
