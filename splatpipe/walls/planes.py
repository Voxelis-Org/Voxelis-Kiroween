"""The Walls_Step's second cross-frame stages: the World_Planes, the two Fallback_Depths and the
flatten, as pure numpy functions in float64 (walls spec, Req 13, 15, 16.2, 16.3, 16.5, 16.6, D1,
D4, D14, D15, A10, A12, A13, C11, C14).

Every function takes plain arrays, poses, Facets and settings and gives back new arrays, plain
values or frozen dataclasses of them, so a stage's worker task can pickle what it needs, and none
reads or writes a file. Depths are (h, w) float64 at the Prior_Size, NaN where there's none.
Normals are (h, w, 3), in the camera frame, unit length and facing the camera (n·r < 0), and
come back as float32, the Prior_Format's type, NaN where there's none.

The robust plane fit (Req 13.2, the design's decisions table).
- robust_plane: Tukey-reweighted principal components. The first fit weights every point 1; each
  of `tukey_iterations` rounds then reweights the points and fits again. A fit is the weighted
  mean μ and the 3 × 3 weighted covariance C of the points, both built with np.einsum (C14), and
  the normal n is C's eigenvector of the smallest eigenvalue, from np.linalg.eigh. The reweight
  takes the signed distances r = (X − μ)·n, their unscaled median absolute deviation
  σ = median(|r − median(r)|), and Tukey's weights (1 − (r / (`tukey_constant`·σ))²)² where
  |r| < `tukey_constant`·σ, else 0. With σ = 0 the weights stay as they were, so a perfectly
  planar set keeps every point. A σ at or below _ROUNDING_DEVIATION × the largest |coordinate|
  is float64 rounding of a perfect plane, not noise, so it counts as 0, as in fit.tukey_change;
  and when no point would keep a weight above 0 the weights stay as they were too. The plane is
  (n, δ = n·μ), so n·X = δ. A plane normal's sign is arbitrary (C11). The first fit weights
  every point 1, so one point far enough off to become the unweighted covariance's largest
  direction turns that first fit away from the plane, and the reweighting doesn't bring it back.
  On 500 points spread over 4 × 4 units, one point 30 units off still fits to float64 rounding,
  and one 40 units off doesn't.

The World_Planes (Req 13.1, 13.3, 13.4, D15, A10, C11).
- world_planes: every Facet of every Trusted_Frame, its pixels with a Fused_Depth finite and above
  0 back-projected at that depth with the Prior_Size Normalised_Coordinates and taken to the world
  with the frame's pose, elementwise (C14). Each Facet's Facet_Plane is robust_plane through its
  points, and its camera distance is |n·C − δ|, C its frame's Camera_Centre. The Facets are then
  merged largest first, by their number of points, ties to the lower frame name and then the
  lower Facet index, so the order the frames are given in changes nothing. A Facet joins the
  first World_Plane, in the order they were started, whose normal n_w has |n_f·n_w| at least
  cos(`plane_merge_degrees`) and whose offset differs from the Facet's, with the Facet's normal
  turned to face the same way as n_w, by at most `plane_merge_offset` × the Facet's camera
  distance. Otherwise it starts a new World_Plane. Each join refits the World_Plane with
  robust_plane through all its Facets' points, in the order they joined, and the refitted normal
  is turned, with its offset, to face the same way as the one before, so a plane's sign never
  flips. A Facet with fewer than 3 points is left out. The World_Planes kept, in the order they
  were started, are those with Facets from at least `plane_min_frames` Prior_Frames and at least
  `plane_min_share` of the points of the largest World_Plane with Facets from that many frames.
- camera_plane: a World_Plane in a camera, n_c = R·n_w and δ_c = δ_w + n_c·t for the
  world-to-camera pose [[R, t], [0, 0, 0, 1]], so n_c·X_c = δ_c. plane_depth gives its depth along
  each Ray r, δ_c / (n_c·r), NaN where that isn't finite and above 0.

The Fallback_Depths of a Failed_Frame (Req 15.1–15.4, D1, D14, A12).
- fallback_medians gives each pixel's median of at least `agree_min_frames` Agreeing_Values of its
  Projected_Values from the Neighbour_Frames' Fused_Depths, with crossframe.agreeing_median, and
  median_coverage the share of the Covered_Pixels with one (Req 15.1, 15.5).
- projection_fallback (Req 15.2): when that share is at least `projection_min_coverage`, one
  fit.tukey_change of the Start_Depth to the medians in inverse depth, with `tukey_constant` and
  `tukey_iterations` and no fit_bound (A12). Each Covered_Pixel with a median takes the median,
  each other Covered_Pixel the fitted change's depth, and every Covered_Pixel the fitted change's
  Changed_Normal of its Start_Depth's normal (the Depth_Prior's). None below the share.
- plane_registration (Req 15.3, 15.4): `register_passes` passes, pass i of P with the tolerance
  start × (end / start)^(i / (P − 1)), start and end being `register_start_tolerance` and
  `register_end_tolerance`, so it falls geometrically from the first to the last, and start
  when P is 1 (register_tolerances). Each pass matches every Covered_Pixel to the World_Plane,
  in its camera, whose plane depth z_p along the pixel's Ray is finite and above 0, whose normal has
  |n_pixel·n_c| at least cos(`register_max_degrees`), n_pixel its current normal, and with
  |z_p / z − 1| at most the pass's tolerance, z the pixel's current depth; among those the nearest
  its current depth, the smallest |z_p − z|, ties to the World_Plane given first. Then one
  fit.tukey_change of the Start_Depth to the matched pixels' plane depths, and the pixels' current
  depth and normal become the Start_Depth and its normal through that change. The first pass
  starts from the Start_Depth and the Depth_Prior's normal. The Fallback_Depth is the Start_Depth
  and its normals through the last pass's change. None when there is no World_Plane, or when a
  pass matches fewer than `register_min_pixels` pixels (Req 15.4).
  Each fallback gives a FallbackDepth: the depth, the normals and the fitted change (a, b, c, d)
  of the Start_Depth, with its m.

The flatten (Req 16.2, 16.3, 16.5, 16.6, D4, C11).
- flatten_frame: for each Facet of a Prior_Frame with a Walls_Depth, its pixels with a depth
  finite and above 0 back-projected in the camera frame, and its robust normal n_f by
  robust_plane through them. The normal n is that of the World_Plane, in this camera, nearest
  n_f in angle, the largest |n_f·n_c|, ties to the World_Plane given first, when that angle is
  within `flatten_degrees`, and n_f otherwise. The offset δ is the median over those pixels of
  n·X. A pixel with depth z moves when its plane depth z_plane = δ / (n·r) is finite and above 0
  and |z / z_plane − 1| is at most `flatten_tolerance`, a share of the plane depth, in depth and
  not in log depth, as the settings table says: "within this share of the Facet's plane". It
  takes the depth z_plane and the normal n turned to face the camera. Every other pixel keeps its
  depth and its normal, bit for bit. Every Facet reads the depths from before the stage and the
  Facets hold no pixel in common, so no pixel's move moves another's (Req 16.5). A Facet with
  fewer than 3 points with a depth is left as it is.

Every sum over many points is elementwise numpy, np.sum, np.median or np.einsum with its default
optimize=False, which calls no BLAS, every pose is applied elementwise, and LAPACK only sees 3 × 3
covariances and fit.tukey_change's 4 × 4 normal equations, so the bits don't depend on the thread
count (C14). Every number comes from [walls] through the settings the caller gives.

This module imports only the standard library, numpy and the walls pure modules.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from splatpipe.walls.crossframe import agreeing_median
from splatpipe.walls.facets import Facet
from splatpipe.walls.fit import FourNumbers, tukey_change
from splatpipe.walls.geometry import (Camera, apply_change, changed_normal, normalised_coords,
                                      to_world)

# A median absolute deviation at or below this many float64 roundings of the largest |coordinate|
# is the rounding of a perfect plane, not noise, so it counts as 0 and the weights stay as they
# were, as in fit.tukey_change. Without it, a perfect plane's distances are a few roundings and
# Tukey's weights would drop points at random.
_ROUNDING_DEVIATION = 1024 * float(np.finfo(np.float64).eps)
# The fewest points that pin a plane down.
_PLANE_POINTS = 3


class PlaneFitSettings(Protocol):
    """The [walls] settings robust_plane's callers read. WallsSettings has them all."""

    tukey_constant: float
    tukey_iterations: int


class WorldPlaneSettings(PlaneFitSettings, Protocol):
    """The [walls] settings world_planes reads."""

    plane_merge_degrees: float
    plane_merge_offset: float
    plane_min_frames: int
    plane_min_share: float


class FallbackSettings(PlaneFitSettings, Protocol):
    """The [walls] settings the two fallbacks read."""

    agree_tolerance: float
    agree_min_frames: int
    projection_min_coverage: float
    register_max_degrees: float
    register_start_tolerance: float
    register_end_tolerance: float
    register_passes: int
    register_min_pixels: int


class FlattenSettings(PlaneFitSettings, Protocol):
    """The [walls] settings the flatten reads."""

    flatten_degrees: float
    flatten_tolerance: float


@dataclass(frozen=True)
class WorldPlane:
    """A plane of the room that Facet_Planes of several Trusted_Frames make together (Req 13)."""

    normal: np.ndarray        # (3,) float64, unit length, in the world; its sign is arbitrary
    offset: float             # normal·X = offset
    frames: frozenset[str]    # the Prior_Frames its Facets come from
    pixels: int               # its Facets' points, one per Prior_Size pixel


@dataclass(frozen=True)
class FallbackDepth:
    """A Failed_Frame's Fallback_Depth (Req 15.2, 15.3)."""

    depth: np.ndarray         # (h, w) float64, NaN off the Covered_Pixels and where there's none
    normal: np.ndarray        # (h, w, 3) float32, unit, facing the camera, NaN where there's none
    change: FourNumbers       # the fitted (a, b, c, d) of the Start_Depth
    m: float                  # the Start_Depth's Median_Inverse_Depth the change is relative to


def robust_fit(points: np.ndarray, constant: float,
               iterations: int) -> tuple[np.ndarray, float, np.ndarray]:
    """robust_plane's plane through `points`, (n, 3), with the points' last weights, as
    (normal (3,), offset, weights (n,)) (Req 13.2). Refuses fewer than 3 points, or any that
    isn't finite."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if points.shape[0] < _PLANE_POINTS or not np.all(np.isfinite(points)):
        raise ValueError(f"robust_plane: wants at least {_PLANE_POINTS} finite points, got "
                         f"{points.shape[0]}")
    rounding = _ROUNDING_DEVIATION * float(np.max(np.abs(points)))
    weights = np.ones(points.shape[0])
    normal, centre = _weighted_plane(points, weights)
    for _ in range(iterations):
        distance = _distances(points, centre, normal)
        weights = _tukey_weights(distance, weights, constant, rounding)
        normal, centre = _weighted_plane(points, weights)
    offset = float(normal[0] * centre[0] + normal[1] * centre[1] + normal[2] * centre[2])
    return normal, offset, weights


def robust_plane(points: np.ndarray, constant: float, iterations: int) -> tuple[np.ndarray, float]:
    """The Facet_Plane through `points`, (n, 3), as (unit normal (3,), offset) with
    normal·X = offset: Tukey-reweighted principal components at `constant` (tukey_constant)
    median absolute deviations, for `iterations` (tukey_iterations) rounds (Req 13.2)."""
    normal, offset, _ = robust_fit(points, constant, iterations)
    return normal, offset


def _weighted_plane(points: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The weighted principal-components plane: its unit normal, the smallest eigenvector of the
    weighted covariance, and the weighted mean, both (3,), built with np.einsum (C14)."""
    total = float(np.sum(weights))
    centre = np.einsum("n,ni->i", weights, points) / total
    offsets = points - centre
    covariance = np.einsum("n,ni,nj->ij", weights, offsets, offsets) / total
    _, vectors = np.linalg.eigh(covariance)
    normal = np.array(vectors[:, 0], dtype=np.float64)
    return normal / math.sqrt(float(np.sum(normal * normal))), centre


def _distances(points: np.ndarray, centre: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """Each point's signed distance (X − μ)·n, elementwise."""
    return ((points[:, 0] - centre[0]) * normal[0] + (points[:, 1] - centre[1]) * normal[1]
            + (points[:, 2] - centre[2]) * normal[2])


def _tukey_weights(distance: np.ndarray, weights: np.ndarray, constant: float,
                   rounding: float) -> np.ndarray:
    """Tukey's weights at `constant` unscaled median absolute deviations of `distance`, or
    `weights` as they were when the deviation is at most `rounding` or no point would keep a
    weight above 0."""
    deviation = float(np.median(np.abs(distance - np.median(distance))))
    if not deviation > rounding:
        return weights
    ratio = distance / (constant * deviation)
    new = np.where(np.abs(ratio) < 1.0, (1.0 - ratio * ratio) ** 2, 0.0)
    if not np.any(new > 0):
        return weights
    return new


def _dot(vector: np.ndarray, other: np.ndarray) -> float:
    """The dot product of two (3,) vectors, elementwise."""
    return float(vector[0] * other[0] + vector[1] * other[1] + vector[2] * other[2])


def camera_centre(world_to_camera: np.ndarray) -> np.ndarray:
    """The Camera_Centre of a world-to-camera pose, −Rᵀt, as (3,) float64."""
    zero = np.zeros(1)
    return np.array([part[0] for part in to_world(world_to_camera, zero, zero, zero)])


def camera_plane(plane: WorldPlane, world_to_camera: np.ndarray) -> tuple[np.ndarray, float]:
    """The World_Plane `plane` in the camera of `world_to_camera`, [[R, t], [0, 0, 0, 1]], as
    (n_c, δ_c) with n_c = R·n_w and δ_c = δ_w + n_c·t, so n_c·X_c = δ_c."""
    pose = np.asarray(world_to_camera, dtype=np.float64)
    r, t = pose[:3, :3], pose[:3, 3]
    n = np.asarray(plane.normal, dtype=np.float64)
    normal = np.array([r[i, 0] * n[0] + r[i, 1] * n[1] + r[i, 2] * n[2] for i in range(3)])
    return normal, float(plane.offset) + _dot(normal, t)


def plane_depth(normal: np.ndarray, offset: float, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """The depth along each Ray r = (x, y, 1) of the camera-frame plane normal·X = offset,
    δ / (n·r), as float64 in x's shape, NaN where that isn't finite and above 0."""
    n = np.asarray(normal, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        depth = float(offset) / (n[0] * x + n[1] * y + n[2])
    return np.where(np.isfinite(depth) & (depth > 0), depth, np.nan)


def _has_depth(depth: np.ndarray) -> np.ndarray:
    """Where `depth` is finite and above 0."""
    with np.errstate(invalid="ignore"):
        return np.isfinite(depth) & (depth > 0)


def _facet_points(facet: Facet, depth: np.ndarray, x: np.ndarray,
                  y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The Facet's pixels with a depth finite and above 0, as (rows, columns) and their camera
    points z·(x, y, 1), (n, 3)."""
    rows, columns = facet.pixels[:, 0], facet.pixels[:, 1]
    z = np.asarray(depth, dtype=np.float64)[rows, columns]
    good = _has_depth(z)
    rows, columns, z = rows[good], columns[good], z[good]
    points = np.stack([z * x[rows, columns], z * y[rows, columns], z], axis=-1)
    return rows, columns, points


class _Building:
    """A World_Plane while the Facets merge: its plane, its Facets' points in the order they
    joined, and the frames they come from."""

    def __init__(self, normal: np.ndarray, offset: float, points: np.ndarray, frame: str) -> None:
        self.normal = normal
        self.offset = offset
        self.points = [points]
        self.frames = {frame}

    @property
    def count(self) -> int:
        return int(sum(part.shape[0] for part in self.points))

    def join(self, points: np.ndarray, frame: str, settings: PlaneFitSettings) -> None:
        """Add a Facet's points and refit through all of them, keeping the normal's sign."""
        self.points.append(points)
        self.frames.add(frame)
        normal, offset = robust_plane(np.concatenate(self.points, axis=0),
                                      settings.tukey_constant, settings.tukey_iterations)
        if _dot(normal, self.normal) < 0:
            normal, offset = -normal, -offset
        self.normal, self.offset = normal, offset


def world_planes(facets: Mapping[str, Sequence[Facet]], fused_depths: Mapping[str, np.ndarray],
                 poses: Mapping[str, np.ndarray], camera: Camera,
                 settings: WorldPlaneSettings) -> list[WorldPlane]:
    """The World_Planes kept, from every Facet of every Trusted_Frame (Req 13.1–13.4, D15, A10,
    C11).

    `facets` maps each Trusted_Frame's name to its Facets, `fused_depths` to its (h, w)
    Fused_Depth at the Prior_Size and `poses` to its world-to-camera Image_Pose. `camera` is the
    Shared_Camera.
    """
    pieces = []
    for name in sorted(facets):
        depth = np.asarray(fused_depths[name], dtype=np.float64)
        height, width = depth.shape
        x, y = normalised_coords(camera, width, height)
        pose = np.asarray(poses[name], dtype=np.float64)
        centre = camera_centre(pose)
        for index, facet in enumerate(facets[name]):
            _, _, points = _facet_points(facet, depth, x, y)
            if points.shape[0] < _PLANE_POINTS:
                continue
            world = np.stack(to_world(pose, points[:, 0], points[:, 1], points[:, 2]), axis=-1)
            normal, offset = robust_plane(world, settings.tukey_constant,
                                          settings.tukey_iterations)
            distance = abs(_dot(normal, centre) - offset)
            pieces.append((-world.shape[0], name, index, world, normal, offset, distance))
    pieces.sort(key=lambda piece: piece[:3])

    cos_limit = math.cos(math.radians(settings.plane_merge_degrees))
    building: list[_Building] = []
    for _, name, _, world, normal, offset, distance in pieces:
        for plane in building:
            dot = _dot(normal, plane.normal)
            if abs(dot) < cos_limit:
                continue
            facing = -1.0 if dot < 0 else 1.0
            if abs(facing * offset - plane.offset) <= settings.plane_merge_offset * distance:
                plane.join(world, name, settings)
                break
        else:
            building.append(_Building(normal, offset, world, name))

    enough = [plane for plane in building if len(plane.frames) >= settings.plane_min_frames]
    if not enough:
        return []
    largest = max(plane.count for plane in enough)
    return [WorldPlane(normal=np.array(plane.normal, dtype=np.float64), offset=float(plane.offset),
                       frames=frozenset(plane.frames), pixels=plane.count)
            for plane in enough if plane.count >= settings.plane_min_share * largest]


def fallback_medians(projected: np.ndarray, settings: FallbackSettings) -> np.ndarray:
    """Each pixel's median of its Agreeing_Values among the (h, w, k) Projected_Values
    `projected`, from the Neighbour_Frames' Fused_Depths, where at least agree_min_frames agree,
    and NaN elsewhere, (h, w) float64 (Req 15.1), by crossframe.agreeing_median."""
    return agreeing_median(projected, settings.agree_tolerance, settings.agree_min_frames)


def median_coverage(medians: np.ndarray, covered: np.ndarray) -> float:
    """The share of the Covered_Pixels with a median, finite and above 0 (Req 15.2, 15.5, D14).
    0 when there's no Covered_Pixel."""
    covered = np.asarray(covered, dtype=bool)
    total = int(np.count_nonzero(covered))
    if total == 0:
        return 0.0
    return int(np.count_nonzero(covered & _has_depth(np.asarray(medians)))) / total


def _changed_depth(start_depth: np.ndarray, change: FourNumbers, m: float, x: np.ndarray,
                   y: np.ndarray) -> np.ndarray:
    """`start_depth` through the Bas_Relief_Change `change` with the m `m`, NaN where it gives
    none."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return 1.0 / apply_change(1.0 / np.asarray(start_depth, dtype=np.float64), change, m, x, y)


def _fallback(start_depth: np.ndarray, start_normal: np.ndarray, covered: np.ndarray,
              change: FourNumbers, m: float, x: np.ndarray,
              y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The Start_Depth and its normals through `change` on the Covered_Pixels, NaN elsewhere."""
    depth = np.where(covered, _changed_depth(start_depth, change, m, x, y), np.nan)
    normal = changed_normal(np.asarray(start_normal, dtype=np.float64), start_depth, x, y,
                            change, m)
    normal = np.where(covered[..., np.newaxis], normal, np.nan)
    return depth, normal


def projection_fallback(medians: np.ndarray, start_depth: np.ndarray, start_normal: np.ndarray,
                        covered: np.ndarray, m: float, camera: Camera,
                        settings: FallbackSettings) -> FallbackDepth | None:
    """A Failed_Frame's Fallback_Depth by the Projection_Fallback, or None when fewer than
    projection_min_coverage of its Covered_Pixels have a median (Req 15.2, D1, D14, A12).

    `medians` is fallback_medians' (h, w) answer, `start_depth` and `start_normal` the frame's
    Start_Depth and the Depth_Prior's normal at the Prior_Size, `covered` its Covered_Pixels and
    `m` the Start_Depth's Median_Inverse_Depth.
    """
    covered = np.asarray(covered, dtype=bool)
    start = np.asarray(start_depth, dtype=np.float64)
    medians = np.asarray(medians, dtype=np.float64)
    if median_coverage(medians, covered) < settings.projection_min_coverage:
        return None
    height, width = start.shape
    x, y = normalised_coords(camera, width, height)
    fit = covered & _has_depth(medians) & _has_depth(start)
    if not fit.any():
        return None
    change = tukey_change(1.0 / start[fit], 1.0 / medians[fit], m, x[fit], y[fit],
                          settings.tukey_constant, settings.tukey_iterations)
    depth, normal = _fallback(start, start_normal, covered, change, m, x, y)
    depth = np.where(covered & _has_depth(medians), medians, depth)
    return FallbackDepth(depth=depth, normal=normal.astype(np.float32), change=change,
                         m=float(m))


def register_tolerances(start: float, end: float, passes: int) -> list[float]:
    """The Plane_Registration's depth tolerance in each of its `passes` passes, falling
    geometrically from `start` (register_start_tolerance) to `end` (register_end_tolerance):
    start × (end / start)^(i / (passes − 1)) for pass i, and [start] for one pass (A12)."""
    if passes == 1:
        return [float(start)]
    return [float(start) * (float(end) / float(start)) ** (i / (passes - 1))
            for i in range(passes)]


def _match_planes(depth: np.ndarray, normal: np.ndarray, plane_depths: Sequence[np.ndarray],
                  plane_normals: Sequence[np.ndarray], cos_limit: float,
                  tolerance: float) -> np.ndarray:
    """Each pixel's matched plane depth, NaN where none matches: among the planes whose depth is
    finite and above 0, whose normal has |n_pixel·n_c| at least `cos_limit` and whose depth is
    within `tolerance` of the pixel's, the one nearest its depth, ties to the first."""
    best = np.full(depth.shape, np.nan)
    gap = np.full(depth.shape, np.inf)
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        for z_plane, n_c in zip(plane_depths, plane_normals):
            along = np.abs(normal[..., 0] * n_c[0] + normal[..., 1] * n_c[1]
                           + normal[..., 2] * n_c[2])
            distance = np.abs(z_plane - depth)
            match = (_has_depth(z_plane) & (along >= cos_limit)
                     & (np.abs(z_plane / depth - 1.0) <= tolerance) & (distance < gap))
            best = np.where(match, z_plane, best)
            gap = np.where(match, distance, gap)
    return best


def plane_registration(planes: Sequence[WorldPlane], start_depth: np.ndarray,
                       start_normal: np.ndarray, covered: np.ndarray, m: float,
                       world_to_camera: np.ndarray, camera: Camera,
                       settings: FallbackSettings) -> FallbackDepth | None:
    """A Failed_Frame's Fallback_Depth by the Plane_Registration, or None when there's no
    World_Plane or a pass matches fewer than register_min_pixels pixels (Req 15.3, 15.4, A12).

    `start_depth` and `start_normal` are the frame's Start_Depth and the Depth_Prior's normal at
    the Prior_Size, `covered` its Covered_Pixels, `m` the Start_Depth's Median_Inverse_Depth and
    `world_to_camera` its Image_Pose.
    """
    if not planes:
        return None
    covered = np.asarray(covered, dtype=bool)
    start = np.asarray(start_depth, dtype=np.float64)
    first_normal = np.asarray(start_normal, dtype=np.float64)
    height, width = start.shape
    x, y = normalised_coords(camera, width, height)
    in_camera = [camera_plane(plane, world_to_camera) for plane in planes]
    plane_normals = [n_c for n_c, _ in in_camera]
    plane_depths = [plane_depth(n_c, d_c, x, y) for n_c, d_c in in_camera]
    cos_limit = math.cos(math.radians(settings.register_max_degrees))
    usable = covered & _has_depth(start)
    depth = np.where(usable, start, np.nan)
    normal = first_normal
    result = None
    for tolerance in register_tolerances(settings.register_start_tolerance,
                                         settings.register_end_tolerance,
                                         settings.register_passes):
        matched = _match_planes(depth, normal, plane_depths, plane_normals, cos_limit, tolerance)
        matches = usable & _has_depth(matched)
        if int(np.count_nonzero(matches)) < settings.register_min_pixels:
            return None
        change = tukey_change(1.0 / start[matches], 1.0 / matched[matches], m, x[matches],
                              y[matches], settings.tukey_constant, settings.tukey_iterations)
        depth, normal = _fallback(start, first_normal, covered, change, m, x, y)
        result = change
    return FallbackDepth(depth=depth, normal=normal.astype(np.float32), change=result,
                         m=float(m))


def flatten_frame(facets: Sequence[Facet], planes: Sequence[WorldPlane], depth: np.ndarray,
                  normal: np.ndarray, world_to_camera: np.ndarray, camera: Camera,
                  settings: FlattenSettings) -> tuple[np.ndarray, np.ndarray]:
    """A Prior_Frame's Walls_Depth and normals after the flatten, (h, w) float64 and (h, w, 3)
    in `normal`'s dtype (Req 16.2, 16.3, 16.5, 16.6, D4, C11).

    `facets` are the frame's Facets, `planes` the World_Planes, `depth` and `normal` its
    Walls_Depth and normals from before the stage at the Prior_Size, `world_to_camera` its
    Image_Pose and `camera` the Shared_Camera. Only a moved pixel changes, in depth and normal.
    """
    before = np.asarray(depth, dtype=np.float64)
    new_depth = before.copy()
    new_normal = np.array(normal, copy=True)
    height, width = before.shape
    x, y = normalised_coords(camera, width, height)
    plane_normals = [camera_plane(plane, world_to_camera)[0] for plane in planes]
    cos_limit = math.cos(math.radians(settings.flatten_degrees))
    for facet in facets:
        rows, columns, points = _facet_points(facet, before, x, y)
        if points.shape[0] < _PLANE_POINTS:
            continue
        own, _ = robust_plane(points, settings.tukey_constant, settings.tukey_iterations)
        chosen, nearest = own, -1.0
        for n_c in plane_normals:
            along = abs(_dot(own, n_c))
            if along >= cos_limit and along > nearest:
                chosen, nearest = n_c, along
        offset = float(np.median(points[:, 0] * chosen[0] + points[:, 1] * chosen[1]
                                 + points[:, 2] * chosen[2]))
        pixel_x, pixel_y = x[rows, columns], y[rows, columns]
        z_plane = plane_depth(chosen, offset, pixel_x, pixel_y)
        z = points[:, 2]
        with np.errstate(invalid="ignore", divide="ignore"):
            moves = _has_depth(z_plane) & (np.abs(z / z_plane - 1.0)
                                           <= settings.flatten_tolerance)
        facing = np.where(chosen[0] * pixel_x + chosen[1] * pixel_y + chosen[2] > 0, -1.0, 1.0)
        new_depth[rows[moves], columns[moves]] = z_plane[moves]
        turned = facing[moves, np.newaxis] * chosen[np.newaxis, :]
        new_normal[rows[moves], columns[moves]] = turned.astype(new_normal.dtype)
    return new_depth, new_normal
