"""Unit tests for the Walls_Step's Facets, splatpipe/walls/facets.py (walls spec, sub-task 6.4,
Req 12.1–12.6, 22.1).

The Jump_Pixels and the cut run on the box room of tests/camera_files.py at its Prior_Size,
128 x 96, with its true normals and inverse depths as the Depth_Prior. The refit runs on the
textured plane of sub-task 5.3, because the box room gives too few Source_Frames for a solve: one
real refit on a Solved_Depth tilted off the true plane shows a FacetChange that passes the three
gates, and the gates' numbers are then handed in for halves_difference and tile_scores, as the
Self_Check tests did, so each gate sits exactly where a test puts it. Every oracle is worked out
here and calls nothing it checks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pytest
from scipy import ndimage

from splatpipe.walls import facets
from splatpipe.walls.facets import (Facet, FacetChange, apply_facet_change, cut_facets,
                                    facet_windows, jump_pixels, refit_facet)
from splatpipe.walls.geometry import normalised_coords
from splatpipe.walls.passes import (PassDepth, PriorScore, ThirdPass, band_coherences,
                                    frame_band_weights, third_pass)
from splatpipe.walls.scoring import ScoreSource, Windows

from camera_files import (BOX_BACK, BOX_CAMERA, BOX_END_X, BOX_FLOOR, BOX_FLOOR_Y, BOX_FRAMES,
                          BOX_JUMP_FRAMES, BOX_NONE, BOX_SMALL_PATCHES, BOX_TOP_Y, BOX_BACK_Z,
                          box_camera, box_depth, box_normal, box_pose, box_surface,
                          textured_depth, textured_normal)
from test_walls_passes import (PASS_FRAME, SETTINGS, TEXTURED_SIZE, above, below,
                               textured_prior, textured_sources, textured_windows)

BOX_SIZE = (BOX_CAMERA.width, BOX_CAMERA.height)   # the box room's Prior_Size, (w, h)
FOUR_CONNECTED = ndimage.generate_binary_structure(2, 1)
NOISE_SEED = 12
NOISE = 0.03          # the spread of each normal part's noise, about 2.4° in angle


@dataclass(frozen=True)
class BoxPrior:
    """One box-room frame as a Depth_Prior at the Prior_Size."""
    surface: np.ndarray   # (h, w) the surface each pixel sees, or BOX_NONE
    covered: np.ndarray   # (h, w) bool
    depth: np.ndarray     # (h, w) true depth, NaN off the Covered mask
    inverse: np.ndarray   # (h, w) 1 / depth, NaN off the Covered mask
    normal: np.ndarray    # (h, w, 3)


def box_prior(index: int, noise: float = 0.0) -> BoxPrior:
    """Box-room frame `index` with its true normals, each part plus Gaussian noise of spread
    `noise`, drawn with a seed of the frame's own."""
    surface = box_surface(index, BOX_SIZE)
    covered = surface != BOX_NONE
    depth = box_depth(index, BOX_SIZE)
    normal = box_normal(index, BOX_SIZE)
    if noise:
        normal = normal + np.random.default_rng(NOISE_SEED + index).normal(0.0, noise,
                                                                           normal.shape)
    inverse = np.where(covered, 1.0 / depth, np.nan)
    return BoxPrior(surface, covered, depth, inverse, normal)


def box_jumps(prior: BoxPrior) -> np.ndarray:
    """The box-room frame's Jump_Pixels with the real settings."""
    return jump_pixels(prior.inverse, prior.covered, SETTINGS.facet_jump,
                       SETTINGS.facet_jump_distance)


# ---------------------------------------------------------------------------------------------
# Jump_Pixels (Req 12.2)
# ---------------------------------------------------------------------------------------------

ALL = (True,) * 6


@pytest.mark.parametrize("inverse, covered, wanted", [
    # Pairs (1, 3) and (2, 4) differ by 0.02, so all four of their pixels are marked.
    ((1.0, 1.0, 1.0, 1.02, 1.02, 1.02), ALL, (False, True, True, True, True, False)),
    # 0.0101 is above 0.01 × 1.0, the smaller, though not above 0.01 × 1.0101, the larger.
    ((1.0, 1.0, 1.0101, 1.0101, 1.0, 1.0), ALL, ALL),
    # 0.0099 is no jump.
    ((1.0, 1.0, 1.0099, 1.0099, 1.0, 1.0), ALL, (False,) * 6),
    # With pixel 3 uncovered, only the pair (2, 4) is left to jump.
    ((1.0, 1.0, 1.0, 1.02, 1.02, 1.02), (True, True, True, False, True, True),
     (False, False, True, False, True, False)),
], ids=["step", "smaller-of-the-two", "under-the-jump", "uncovered-pixel"])
@pytest.mark.parametrize("down", [False, True], ids=["across", "down"])
def test_jump_pixels_mark_both_pixels_of_each_pair_that_differs(inverse: tuple, covered: tuple,
                                                                wanted: tuple,
                                                                down: bool) -> None:
    """With `facet_jump` 0.01 and a distance of 2, a pair of Covered_Pixels 2 apart across or
    down is a jump when its inverse depths differ by more than 0.01 of the smaller one, and both
    its pixels are marked (Req 12.2)."""
    image, mask, expected = (np.array([values]) for values in (inverse, covered, wanted))
    if down:
        image, mask, expected = image.T, mask.T, expected.T
    assert np.array_equal(jump_pixels(image, mask, 0.01, 2), expected)


def end_edge(index: int) -> tuple[np.ndarray, np.ndarray]:
    """The back wall's end edge, x = BOX_END_X on the back wall from its top to the floor,
    projected into box-room frame `index`, as its two ends' (u, v) in pixels."""
    pose = box_pose(index)
    ends = np.array([[BOX_END_X, BOX_TOP_Y, BOX_BACK_Z, 1.0],
                     [BOX_END_X, BOX_FLOOR_Y, BOX_BACK_Z, 1.0]])
    camera = (pose @ ends.T).T[:, :3]
    u = BOX_CAMERA.fx * camera[:, 0] / camera[:, 2] + BOX_CAMERA.cx
    v = BOX_CAMERA.fy * camera[:, 1] / camera[:, 2] + BOX_CAMERA.cy
    return np.array([u[0], v[0]]), np.array([u[1], v[1]])


def segment_distance(points: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    """Each (u, v) point's distance from the segment start–end, in pixels."""
    along = end - start
    t = np.clip(np.sum((points - start) * along, axis=1) / float(np.sum(along * along)), 0, 1)
    return np.hypot(*(points - (start + t[:, np.newaxis] * along)).T)


@pytest.mark.parametrize("index", range(BOX_FRAMES))
def test_jump_pixels_mark_the_back_walls_end_edge_in_front_of_the_floor(index: int) -> None:
    """In the frames that see the back wall end in front of the floor, the Jump_Pixels lie along
    that edge, on both its sides: each within `facet_jump_distance` pixels of the edge, since its
    pair's other pixel is that far away across the edge, on the back wall and on the floor, and in
    each row where the edge has two back-wall pixels on its left and two floor pixels on its right
    with the floor clearly further, all four are marked. The creases, where the surfaces meet at
    one depth, have none, and the other frames have none at all (Req 12.2)."""
    prior = box_prior(index)
    jumps = box_jumps(prior)
    assert not np.any(jumps & ~prior.covered)
    if index not in BOX_JUMP_FRAMES:
        assert not jumps.any()
        return

    rows, columns = np.nonzero(jumps)
    start, end = end_edge(index)
    centres = np.stack([columns + 0.5, rows + 0.5], axis=1)
    assert segment_distance(centres, start, end).max() <= SETTINGS.facet_jump_distance
    on = prior.surface[jumps]
    assert set(on.tolist()) == {BOX_BACK, BOX_FLOOR}

    checked = 0
    height, width = prior.covered.shape
    for row in range(height):
        v = row + 0.5
        if not min(start[1], end[1]) <= v <= max(start[1], end[1]):
            continue
        u = start[0] + (v - start[1]) / (end[1] - start[1]) * (end[0] - start[0])
        left = math.floor(u - 0.5)    # the last pixel whose centre is left of the edge
        if left < 1 or left + 2 >= width:
            continue
        sides = prior.surface[row, left - 1:left + 3]
        if not np.array_equal(sides, [BOX_BACK, BOX_BACK, BOX_FLOOR, BOX_FLOOR]):
            continue
        depths = prior.depth[row, left - 1:left + 3]
        if not depths[2:].min() > (1 + 2 * SETTINGS.facet_jump) * depths[:2].max():
            continue
        assert jumps[row, left - 1:left + 3].all(), row
        checked += 1
    assert checked >= 20


# ---------------------------------------------------------------------------------------------
# Cutting the Facets (Req 12.1, 12.3)
# ---------------------------------------------------------------------------------------------

def unit_normals(normal: np.ndarray) -> np.ndarray:
    """Each normal at unit length, worked out here."""
    return normal / np.sqrt(np.sum(normal * normal, axis=-1, keepdims=True))


def check_facet_rules(prior: BoxPrior, cut: list[Facet]) -> None:
    """Every rule of Req 12.3 on the Facets of one frame, with oracles worked out here."""
    shape = prior.covered.shape
    jumps = box_jumps(prior)
    unit = unit_normals(np.where(prior.covered[..., np.newaxis], prior.normal, 1.0))
    taken = np.zeros(shape, dtype=np.int64)
    limit = SETTINGS.facet_normal_degrees
    counts = [facet.count for facet in cut]
    assert counts == sorted(counts, reverse=True)
    for facet in cut:
        mask = facet.mask(shape)
        assert facet.count == int(mask.sum()) >= SETTINGS.facet_min_pixels
        flat = facet.pixels[:, 0] * shape[1] + facet.pixels[:, 1]
        assert np.all(np.diff(flat) > 0)             # row-major order, no pixel twice
        assert prior.covered[mask].all()
        assert not jumps[mask].any()
        _, pieces = ndimage.label(mask, structure=FOUR_CONNECTED)
        assert pieces == 1
        total = unit[mask].sum(axis=0)
        mean = total / math.sqrt(float(total @ total))
        np.testing.assert_allclose(facet.normal, mean, rtol=0, atol=1e-12)
        angles = np.degrees(np.arccos(np.clip(unit[mask] @ mean, -1.0, 1.0)))
        assert angles.max() <= limit + 1e-9
        assert facet.m == pytest.approx(float(np.median(prior.inverse[mask])), rel=1e-15)
        taken += mask
    assert taken.max(initial=0) <= 1                  # no pixel in two Facets


def big_surfaces(prior: BoxPrior, jumps: np.ndarray) -> list[int]:
    """The surfaces the frame sees with at least `facet_min_pixels` pixels that aren't
    Jump_Pixels."""
    return [surface for surface in (BOX_BACK, BOX_FLOOR, 2)
            if np.count_nonzero((prior.surface == surface) & ~jumps)
            >= SETTINGS.facet_min_pixels]


@pytest.mark.parametrize("index", range(BOX_FRAMES))
def test_cut_facets_gives_one_facet_per_surface_seen(index: int) -> None:
    """With the true normals, each surface the frame sees with at least `facet_min_pixels`
    pixels is exactly one Facet, all its pixels but the Jump_Pixels, with the surface's own
    normal, and a smaller patch is no Facet (Req 12.1, 12.3)."""
    prior = box_prior(index)
    cut = cut_facets(prior.normal, prior.inverse, prior.covered, SETTINGS)
    check_facet_rules(prior, cut)
    jumps = box_jumps(prior)
    wanted = big_surfaces(prior, jumps)
    assert len(cut) == len(wanted)
    shape = prior.covered.shape
    for facet in cut:
        on = set(prior.surface[facet.mask(shape)].tolist())
        assert len(on) == 1
        surface = on.pop()
        assert surface in wanted
        assert np.array_equal(facet.mask(shape), (prior.surface == surface) & ~jumps)
        surface_normal = prior.normal[prior.surface == surface][0]
        np.testing.assert_allclose(facet.normal, surface_normal, rtol=0, atol=1e-12)
    if index in BOX_SMALL_PATCHES:
        small = prior.surface == BOX_SMALL_PATCHES[index]
        assert 0 < np.count_nonzero(small) < SETTINGS.facet_min_pixels
        assert not any(facet.mask(shape)[small].any() for facet in cut)


@pytest.mark.parametrize("index", range(BOX_FRAMES))
def test_cut_facets_keeps_every_rule_with_noisy_normals(index: int) -> None:
    """With noise of about 2.4° on every normal, so the angle and the clean-up have work to do,
    each Facet still keeps every rule of Req 12.3, lies on one surface and covers most of it,
    and the surfaces with a Facet are still the ones seen with at least `facet_min_pixels`
    pixels (Req 12.1, 12.3)."""
    prior = box_prior(index, NOISE)
    cut = cut_facets(prior.normal, prior.inverse, prior.covered, SETTINGS)
    check_facet_rules(prior, cut)
    jumps = box_jumps(prior)
    shape = prior.covered.shape
    surfaces = []
    for facet in cut:
        on = set(prior.surface[facet.mask(shape)].tolist())
        assert len(on) == 1
        surface = on.pop()
        surfaces.append(surface)
        assert facet.count >= 0.95 * np.count_nonzero((prior.surface == surface) & ~jumps)
    assert sorted(surfaces) == big_surfaces(prior, jumps)


# ---------------------------------------------------------------------------------------------
# Refitting a Facet (Req 12.4, 12.5)
# ---------------------------------------------------------------------------------------------

TILT = (1.0, 0.05, 0.0, 0.0)   # the Solved_Depth's change off the true depth, a slope across


@dataclass(frozen=True)
class Plane:
    """PASS_FRAME of the textured plane, its Start_Depth the true depth and its Solved_Depth the
    true depth through TILT, with its one Facet and the Third_Pass around that Solved_Depth."""
    prior: PriorScore
    windows: Windows
    sources: list[ScoreSource]
    true_depth: np.ndarray
    solved: PassDepth
    third: ThirdPass
    facet: Facet


def through(depth: np.ndarray, change: tuple[float, float, float, float], m: float,
            x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """`depth` through the change with the m, 1 / (a / z + m·(b·x + c·y + d)), worked out here."""
    a, b, c, d = change
    return 1.0 / (a / depth + m * (b * x + c * y + d))


@pytest.fixture(scope="module")
def plane() -> Plane:
    """The textured plane's one Facet, cut from its true normals and inverse depth, and the
    Third_Pass of its tilted Solved_Depth on the real Band_Weights."""
    prior = textured_prior(PASS_FRAME, 1.0)
    windows = textured_windows(prior)
    sources = textured_sources(PASS_FRAME)
    true_depth = textured_depth(PASS_FRAME, TEXTURED_SIZE)
    m = prior.m
    solved = PassDepth(score_depth=through(prior.start_depth, TILT, m, prior.frame.x,
                                           prior.frame.y),
                       prior_depth=through(prior.prior_start_depth, TILT, m, prior.prior_x,
                                           prior.prior_y),
                       m=float(np.median(1.0 / through(prior.prior_start_depth, TILT, m,
                                                        prior.prior_x, prior.prior_y))),
                       change=TILT)
    weights = frame_band_weights(prior, band_coherences(prior, windows, sources, solved,
                                                        SETTINGS), SETTINGS)
    third = third_pass(prior, windows, sources, weights, solved, SETTINGS)
    cut = cut_facets(textured_normal(PASS_FRAME, TEXTURED_SIZE), 1.0 / true_depth,
                     prior.covered, SETTINGS)
    assert len(cut) == 1 and cut[0].count == prior.covered.sum()
    return Plane(prior, windows, sources, true_depth, solved, third, cut[0])


def refit(plane: Plane) -> FacetChange | None:
    """refit_facet on the plane's Facet, index 0."""
    return refit_facet(plane.facet, plane.windows, plane.sources, plane.solved, SETTINGS,
                       prior=plane.prior, third=plane.third, index=0)


def facet_error(depth: np.ndarray, plane: Plane) -> float:
    """The median over the Facet's pixels of |z / z_true − 1|."""
    mask = plane.facet.mask(plane.true_depth.shape)
    return float(np.median(np.abs(depth[mask] / plane.true_depth[mask] - 1.0)))


def change_size(plane: Plane, change: FacetChange) -> float:
    """Gate 2's change, the median over the Facet's pixels of |z_facet / z − 1|, worked out
    here."""
    mask = plane.facet.mask(plane.true_depth.shape)
    z = plane.solved.prior_depth[mask]
    moved = through(z, change.change, change.m, plane.prior.prior_x[mask],
                    plane.prior.prior_y[mask])
    return float(np.median(np.abs(moved / z - 1.0)))


@pytest.fixture(scope="module")
def real_change(plane: Plane) -> FacetChange:
    """The real refit of the tilted plane, with nothing handed in."""
    change = refit(plane)
    assert change is not None
    return change


def test_refit_facet_finds_the_tilt_off_the_true_plane(plane: Plane,
                                                       real_change: FacetChange) -> None:
    """On the textured plane with its Solved_Depth tilted off the true plane, the real refit
    gives a FacetChange whose three gates pass on their own numbers, whose m is the Solved_Depth's
    Median_Inverse_Depth over the Facet, and which takes the Facet most of the way back to the
    true plane (Req 12.4, 12.5)."""
    change = real_change
    mask = plane.facet.mask(plane.true_depth.shape)
    assert change.facet == 0
    assert change.m == pytest.approx(float(np.median(1.0 / plane.solved.prior_depth[mask])),
                                     rel=1e-15)
    assert change.halves_difference <= SETTINGS.facet_halves_tolerance
    assert change_size(plane, change) > SETTINGS.facet_min_change_ratio * change.halves_difference
    assert change.tile_low >= SETTINGS.facet_min_z and change.tile_high >= SETTINGS.facet_min_z

    before = facet_error(plane.solved.prior_depth, plane)
    moved = through(plane.solved.prior_depth, change.change, change.m, plane.prior.prior_x,
                    plane.prior.prior_y)
    after = facet_error(moved, plane)
    assert before > 0.005
    assert after < before / 10


@pytest.mark.parametrize("numbers, passed, tiles_made", [
    ({}, True, True),
    ({"difference": "above_tolerance"}, False, False),
    ({"difference": "just_below_ratio"}, True, True),
    ({"difference": "at_ratio"}, False, False),
    ({"low": "below_min_z"}, False, True),
    ({"high": "below_min_z"}, False, True),
    ({"low": math.nan}, False, True),
    ({"high": math.nan}, False, True),
], ids=["passes", "halves-above-tolerance", "change-just-above-ratio", "change-not-above-ratio",
        "low-z-below-min", "high-z-below-min", "no-low-z", "no-high-z"])
def test_refit_facet_fails_on_each_gate_on_its_own(monkeypatch, plane: Plane,
                                                   real_change: FacetChange, numbers: dict,
                                                   passed: bool, tiles_made: bool) -> None:
    """refit_facet gives a FacetChange with a Halves_Difference at most `facet_halves_tolerance`,
    a change above `facet_min_change_ratio` × it, and both Tile_Scores at least `facet_min_z`,
    each passing case at or just past its limit, and None when any one of them fails with the
    others passing, or a Tile_Score is NaN, no Z. The Tile_Scores are made only when the first
    two gates pass, on the Facet's own Score_Windows at the answer, with `occlusion_tolerance`
    (Req 12.4, 12.5). The Halves_Difference and the Tile_Scores are handed in for
    halves_difference and tile_scores; the fits are the real ones, so the change is the real
    refit's."""
    size = change_size(plane, real_change)
    ratio = SETTINGS.facet_min_change_ratio
    differences = {"passing": min(size / ratio / 2, SETTINGS.facet_halves_tolerance),
                   "above_tolerance": above(SETTINGS.facet_halves_tolerance),
                   "just_below_ratio": size / ratio * (1 - 1e-9),
                   "at_ratio": size / ratio * (1 + 1e-9)}
    assert differences["at_ratio"] <= SETTINGS.facet_halves_tolerance
    difference = differences[numbers.get("difference", "passing")]
    tile = {"low": numbers.get("low", SETTINGS.facet_min_z),
            "high": numbers.get("high", SETTINGS.facet_min_z)}
    for side in tile:
        if tile[side] == "below_min_z":
            tile[side] = below(SETTINGS.facet_min_z)
    mask = plane.facet.mask(plane.true_depth.shape)
    own = facet_windows(plane.facet, plane.windows, TEXTURED_SIZE, TEXTURED_SIZE)
    seen: dict = {}

    def hand_difference(prior, current, change_a, change_b):
        assert np.array_equal(prior.covered, mask)
        assert current.m == real_change.m
        return difference

    def hand_tiles(frame, windows, sources, answer_depth, prior_image, *, weights, **options):
        seen["tiles"] = (windows, answer_depth, prior_image, weights, options)
        return tile["low"], tile["high"], 3, 4

    monkeypatch.setattr(facets, "halves_difference", hand_difference)
    monkeypatch.setattr(facets, "tile_scores", hand_tiles)
    result = refit(plane)

    assert ("tiles" in seen) is tiles_made
    if tiles_made:
        windows, answer, prior_image, weights, options = seen["tiles"]
        assert np.array_equal(windows.centres_x, own.centres_x)
        assert np.array_equal(windows.centres_y, own.centres_y)
        assert prior_image is plane.third.prior_image and weights is plane.third.scored_weights
        assert options["occlusion_tolerance"] == SETTINGS.occlusion_tolerance
        np.testing.assert_allclose(answer, through(plane.solved.score_depth, real_change.change,
                                                   real_change.m, plane.prior.frame.x,
                                                   plane.prior.frame.y), rtol=1e-12)
    if not passed:
        assert result is None
        return
    assert result is not None
    assert result.change == real_change.change and result.m == real_change.m
    assert result.halves_difference == difference
    assert (result.tile_low, result.tile_high) == (tile["low"], tile["high"])


# ---------------------------------------------------------------------------------------------
# Applying a Facet_Change (Req 12.6)
# ---------------------------------------------------------------------------------------------

def test_apply_facet_change_changes_only_the_facets_pixels() -> None:
    """On box-room frame 6, a Facet_Change on its floor Facet gives the Facet's pixels the
    change's depth and Changed_Normal, worked out here, and leaves every other pixel's depth and
    normal as it was, bit for bit, without touching the inputs (Req 12.6)."""
    prior = box_prior(6)
    cut = cut_facets(prior.normal, prior.inverse, prior.covered, SETTINGS)
    facet = cut[0]
    shape = prior.covered.shape
    mask = facet.mask(shape)
    assert set(prior.surface[mask].tolist()) == {BOX_FLOOR}
    x, y = normalised_coords(box_camera(), *BOX_SIZE)
    change = FacetChange(facet=0, change=(1.04, 0.03, -0.02, 0.01), m=facet.m,
                         halves_difference=0.001, tile_low=5.0, tile_high=6.0)
    depth, normal = prior.depth.copy(), prior.normal.copy()

    new_depth, new_normal = apply_facet_change(depth, normal, facet, change, x, y)

    assert np.array_equal(depth, prior.depth, equal_nan=True)
    assert np.array_equal(normal, prior.normal)
    assert new_depth.dtype == depth.dtype and new_normal.dtype == normal.dtype
    assert np.array_equal(new_depth[~mask], depth[~mask], equal_nan=True)
    assert np.array_equal(new_normal[~mask], normal[~mask])

    a, b, c, d = change.change
    z, n = depth[mask], normal[mask]
    np.testing.assert_allclose(new_depth[mask], through(z, change.change, change.m, x[mask],
                                                        y[mask]), rtol=1e-12)
    ray = np.stack([x[mask], y[mask], np.ones(z.shape)], axis=1)
    raw = a * n + (change.m * z * np.sum(n * ray, axis=1))[:, np.newaxis] * np.array([b, c, d])
    wanted = unit_normals(raw)
    wanted = np.where((np.sum(wanted * ray, axis=1) > 0)[:, np.newaxis], -wanted, wanted)
    np.testing.assert_allclose(new_normal[mask], wanted, rtol=0, atol=1e-12)
    assert not np.allclose(new_depth[mask], z)
