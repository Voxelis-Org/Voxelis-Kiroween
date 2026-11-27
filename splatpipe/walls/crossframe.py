"""The Walls_Step's first cross-frame stages and its per-frame choice, as pure numpy functions in
float64 (walls spec, Req 14 to 17).

Every function here takes plain arrays, poses and settings and gives back new arrays or plain
values, so a stage's worker task can pickle what it needs, and none reads or writes a file.

The Neighbour_Frames and the Projected_Values (Req 14.1, 14.2, D14, A11).
- neighbours: the Trusted_Frames other than the frame whose Pick_Numbers are within
  neighbour_window of its own, the nearest Pick_Number first, the lower Pick_Number on a tie, at
  most neighbour_frames of them.
- projected_values: each Neighbour_Frame's pixels with a finite depth above 0, at the Prior_Size,
  are back-projected at that depth with their own Normalised_Coordinates, go to the world with
  the Neighbour_Frame's pose and into the frame's camera with the frame's pose, both applied
  elementwise with geometry.to_world and geometry.to_camera (C14), and are z-buffered into the
  frame's Prior_Size grid with geometry.zbuffer_project, given the Shared_Camera's full-size K and
  its size, so each lands by the Depth_Step's pixel rule, in front of the camera, and each
  Neighbour_Frame gives a pixel at most one value, its nearest. It gives an (h, w, k) array, one
  layer per Neighbour_Frame in the order given, NaN where a Neighbour_Frame gives nothing and at
  every pixel that isn't a Covered_Pixel of the frame.
- agreeing (C2): a pixel's Projected_Values v with |v / v_near − 1| ≤ agree_tolerance, v_near the
  smallest Projected_Value at the pixel, the nearest to the camera. The others become NaN.
- agreeing_median (Req 15.1): the median of a pixel's Agreeing_Values where there are at least
  agree_min_frames of them, and NaN elsewhere.

The stages. Each takes the depths from before the stage and gives a new array, so no pixel's
change within a stage moves another's (Req 16.5), and the caller gives every frame of a stage the
depths the stage before left, so the order the frames go through it changes nothing (Req 14.4,
A11). They change depths only, never normals (Req 16.6, A13). A pixel works on only when its own
depth is finite and above 0, and the Projected_Values are NaN off the Covered_Pixels, so the fuse
and the correction change Covered_Pixels only.
- fuse_frame (Req 14.3): the pixel's own Solved_Depth z and the Projected_Values v from its
  Neighbour_Frames' Solved_Depths with |v / z − 1| ≤ fuse_tolerance. When those are at least
  fuse_min_values values, the own one included, the pixel takes their median, and otherwise
  keeps z.
- correct_frame (Req 16.1): with M the agreeing_median of the Projected_Values from the
  Neighbour_Frames' Fused_Depths, a pixel whose own depth z has |ln(z / M)| above
  correction_tolerance takes M.
- despeckle_frame (Req 16.4): for each Covered_Pixel, the despeckle_window × despeckle_window
  window centred on it, cut at the image's border, has as valid pixels its Covered_Pixels with a
  finite depth above 0, the pixel itself included. When there are at least despeckle_min_valid
  and the pixel's depth z has |z / M − 1| above despeckle_tolerance, M the valid pixels' median,
  the pixel takes M. The window is gathered one offset at a time, each offset vectorised over
  every pixel, and the median is exact, so a pixel left alone keeps its bits (Property 6).

Every median is exact: the values sorted, NaN last, and the middle one, or the two middle ones'
(lo + hi) / 2 for an even count, as np.median works it out.

The per-frame choice (Req 17.1–17.5, D1, A14, C5).
- sparse_error: the median over the Scale_Points of |ln(z / z_point)|, z the Walls_Depth at the
  point's pixel and z_point the point's camera-space z, over the points whose Walls_Depth and
  z_point are finite and above 0. A point where the Walls_Depth has no depth gets its
  Depth_Prior's value in the Walls_Prior (D3), so it doesn't speak for the Walls_Depth. NaN when
  no point is left.
- frame_choice: a Timed_Out_Frame keeps its Depth_Prior (`depth_prior`, `timed_out`); a
  Trusted_Frame gets a Walls_Prior (`passed`); a Failed_Frame with no Fallback_Depth keeps its
  Depth_Prior (`depth_prior`, `no_fallback`); one with a Fallback_Depth and fewer than
  scale_min_points Scale_Points, or a Sparse_Error of at most fallback_max_sparse_error, gets a
  Walls_Prior (`fallback`); and every other one keeps its Depth_Prior (`depth_prior`,
  `sparse_error`). A Sparse_Error that is None or NaN, with enough Scale_Points, isn't at most
  the limit, so the frame keeps its Depth_Prior.

The only reductions over many pixels are np.sort, np.min and np.count_nonzero, numpy's own loops,
and every pose is applied elementwise, so no BLAS call sees the pixels and the bits don't depend
on the thread count (C14). Every number comes from [walls] through the settings the caller gives.

This module imports only the standard library, numpy and splatpipe.walls.geometry.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Protocol

import numpy as np

from splatpipe.walls.geometry import Camera, normalised_coords, to_camera, to_world, zbuffer_project

PASSED = "passed"
FALLBACK = "fallback"
DEPTH_PRIOR = "depth_prior"
TIMED_OUT = "timed_out"
NO_FALLBACK = "no_fallback"
SPARSE_ERROR = "sparse_error"


class FuseSettings(Protocol):
    """The [walls] settings the fuse reads. WallsSettings has them all."""

    fuse_tolerance: float
    fuse_min_values: int


class CorrectSettings(Protocol):
    """The [walls] settings the correction reads."""

    agree_tolerance: float
    agree_min_frames: int
    correction_tolerance: float


class ChoiceSettings(Protocol):
    """The [walls] settings the per-frame choice reads."""

    scale_min_points: int
    fallback_max_sparse_error: float


def neighbours(frame: str, pick_number: int, trusted: Mapping[str, int], window: int,
               count: int) -> list[str]:
    """The Neighbour_Frames of the Prior_Frame `frame`, whose Pick_Number is `pick_number`
    (Req 14.1, D14, A11).

    `trusted` maps each Trusted_Frame's name to its Pick_Number. The ones other than `frame`
    within `window` (neighbour_window) Pick_Numbers, the nearest first and the lower Pick_Number
    on a tie, at most `count` (neighbour_frames) of them.
    """
    near = [(abs(number - pick_number), number, name) for name, number in trusted.items()
            if name != frame and abs(number - pick_number) <= window]
    near.sort(key=lambda item: (item[0], item[1]))
    return [name for _, _, name in near[:count]]


def full_size_k(camera: Camera) -> np.ndarray:
    """The Shared_Camera's K at its own size, (3, 3) float64."""
    return np.array([[float(camera.fx), 0.0, float(camera.cx)],
                     [0.0, float(camera.fy), float(camera.cy)],
                     [0.0, 0.0, 1.0]])


def projected_values(frame_pose: np.ndarray, frame_covered: np.ndarray,
                     neighbour_depths: Sequence[np.ndarray],
                     neighbour_poses: Sequence[np.ndarray], camera: Camera) -> np.ndarray:
    """The Projected_Values of a Prior_Frame, (h, w, k) float64, one layer per Neighbour_Frame in
    the order given, NaN where that Neighbour_Frame gives none (Req 14.2, A11).

    `frame_pose` is the frame's world-to-camera Image_Pose and `frame_covered` its (h, w) bool
    Covered_Pixels at the Prior_Size. Each of `neighbour_depths` is a Neighbour_Frame's current
    depth at the Prior_Size, whose pixels with a finite depth above 0 are its points, and each of
    `neighbour_poses` its Image_Pose. `camera` is the Shared_Camera.
    """
    if len(neighbour_depths) != len(neighbour_poses):
        raise ValueError("projected_values: one pose per neighbour depth")
    covered = np.asarray(frame_covered, dtype=bool)
    height, width = covered.shape
    values = np.full((height, width, len(neighbour_depths)), np.nan)
    k = full_size_k(camera)
    size = (int(camera.width), int(camera.height))
    for layer, (depth, pose) in enumerate(zip(neighbour_depths, neighbour_poses)):
        points = _points_in(frame_pose, np.asarray(depth, dtype=np.float64), pose, camera)
        nearest, filled = zbuffer_project(points, k, width, height, camera_size=size)
        values[..., layer] = np.where(filled & covered, nearest, np.nan)
    return values


def _points_in(frame_pose: np.ndarray, depth: np.ndarray, pose: np.ndarray,
               camera: Camera) -> np.ndarray:
    """A Neighbour_Frame's points, its pixels with a finite depth above 0 back-projected with its
    own Normalised_Coordinates, in the frame's camera, as (n, 3) float64."""
    height, width = depth.shape
    x, y = normalised_coords(camera, width, height)
    with np.errstate(invalid="ignore"):
        has_depth = np.isfinite(depth) & (depth > 0)
    z = depth[has_depth]
    world = to_world(pose, z * x[has_depth], z * y[has_depth], z)
    return np.stack(to_camera(frame_pose, *world), axis=-1)


def agreeing(values: np.ndarray, tolerance: float) -> np.ndarray:
    """The Agreeing_Values of each pixel's Projected_Values `values`, (..., k), with NaN for none:
    the values v with |v / v_near − 1| ≤ `tolerance` (agree_tolerance), v_near the pixel's
    smallest value, the nearest to the camera (C2). The others become NaN."""
    values = np.asarray(values, dtype=np.float64)
    if values.shape[-1] == 0:
        return values.copy()
    nearest = np.where(np.isnan(values), np.inf, values).min(axis=-1, keepdims=True)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        keep = np.abs(values / nearest - 1.0) <= tolerance
    return np.where(keep, values, np.nan)


def agreeing_median(values: np.ndarray, tolerance: float, min_frames: int) -> np.ndarray:
    """The median of each pixel's Agreeing_Values where it has at least `min_frames`
    (agree_min_frames) of them, and NaN elsewhere, (...) float64 (Req 15.1)."""
    median, count = _median_last(agreeing(values, tolerance))
    return np.where(count >= min_frames, median, np.nan)


def fuse_frame(own_depth: np.ndarray, projected: np.ndarray,
               settings: FuseSettings) -> np.ndarray:
    """A Trusted_Frame's Fused_Depth, (h, w) float64 (Req 14.3).

    `own_depth` is its Solved_Depth and `projected` the (h, w, k) Projected_Values from its
    Neighbour_Frames' Solved_Depths. At a pixel with an own depth z, finite and above 0, the
    values v with |v / z − 1| ≤ fuse_tolerance and z itself, when they are at least
    fuse_min_values, give their median; every other pixel keeps its own depth.
    """
    own = np.asarray(own_depth, dtype=np.float64)
    values = np.asarray(projected, dtype=np.float64)
    has_own = _has_depth(own)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        close = np.abs(values / own[..., np.newaxis] - 1.0) <= settings.fuse_tolerance
    pool = np.concatenate([own[..., np.newaxis], np.where(close, values, np.nan)], axis=-1)
    median, count = _median_last(pool)
    return np.where(has_own & (count >= settings.fuse_min_values), median, own)


def correct_frame(own_depth: np.ndarray, projected: np.ndarray,
                  settings: CorrectSettings) -> np.ndarray:
    """A Prior_Frame's Walls_Depth after the correction, (h, w) float64 (Req 16.1).

    `own_depth` is its Walls_Depth before the correction and `projected` the (h, w, k)
    Projected_Values from its Neighbour_Frames' Fused_Depths. A pixel with an own depth z, finite
    and above 0, and an agreeing_median M of at least agree_min_frames Agreeing_Values takes M
    when |ln(z / M)| is above correction_tolerance; every other pixel keeps its depth.
    """
    own = np.asarray(own_depth, dtype=np.float64)
    target = agreeing_median(projected, settings.agree_tolerance, settings.agree_min_frames)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        off = np.abs(np.log(own / target)) > settings.correction_tolerance
    return np.where(_has_depth(own) & np.isfinite(target) & off, target, own)


def despeckle_frame(depth: np.ndarray, covered: np.ndarray, window: int, min_valid: int,
                    tolerance: float) -> np.ndarray:
    """A Prior_Frame's Walls_Depth after the despeckle, (h, w) float64 (Req 16.4).

    For each Covered_Pixel of `covered`, the `window` × `window` (despeckle_window) window centred
    on it, cut at the border, has as valid pixels its Covered_Pixels with a finite depth above 0,
    the pixel itself included. When there are at least `min_valid` (despeckle_min_valid) and the
    pixel's depth z has |z / M − 1| above `tolerance` (despeckle_tolerance), M the valid pixels'
    exact median, the pixel takes M. Every other pixel keeps its depth, bit for bit.
    """
    depth = np.asarray(depth, dtype=np.float64)
    covered = np.asarray(covered, dtype=bool)
    if window < 1 or window % 2 == 0:
        raise ValueError("despeckle_frame: the window must be an odd whole number")
    median, count = _median_last(_window_stack(np.where(covered & _has_depth(depth), depth,
                                                        np.nan), window))
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        off = np.abs(depth / median - 1.0) > tolerance
    return np.where(covered & (count >= min_valid) & off, median, depth)


def _window_stack(valid_depth: np.ndarray, window: int) -> np.ndarray:
    """Each pixel's `window` × `window` window of `valid_depth`, NaN off the valid pixels and
    past the border, as (h, w, window²), gathered one offset at a time over every pixel."""
    height, width = valid_depth.shape
    half = window // 2
    padded = np.full((height + 2 * half, width + 2 * half), np.nan)
    padded[half:half + height, half:half + width] = valid_depth
    stack = np.empty((height, width, window * window))
    for row in range(window):
        for column in range(window):
            stack[..., row * window + column] = padded[row:row + height, column:column + width]
    return stack


def _has_depth(depth: np.ndarray) -> np.ndarray:
    """Where `depth` is finite and above 0."""
    with np.errstate(invalid="ignore"):
        return np.isfinite(depth) & (depth > 0)


def _median_last(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The exact median of the values that aren't NaN along the last axis, and their count, as
    (...) float64 and (...) int64. NaN where there are none.

    The values are sorted, NaN last, and the median is the middle one for an odd count and
    (lo + hi) / 2 of the two middle ones for an even count, as np.median works it out.
    """
    values = np.asarray(values, dtype=np.float64)
    count = np.count_nonzero(~np.isnan(values), axis=-1)
    if values.shape[-1] == 0:
        return np.full(values.shape[:-1], np.nan), count
    ordered = np.sort(values, axis=-1)
    low = np.take_along_axis(ordered, np.maximum((count - 1) // 2, 0)[..., np.newaxis],
                             axis=-1)[..., 0]
    high = np.take_along_axis(ordered, (count // 2)[..., np.newaxis], axis=-1)[..., 0]
    with np.errstate(over="ignore", invalid="ignore"):
        median = np.where(count % 2 == 1, low, (low + high) / 2)
    return np.where(count > 0, median, np.nan), count


def sparse_error(scale_points_z: np.ndarray, walls_depth_at_points: np.ndarray) -> float:
    """The Sparse_Error, the median over the Scale_Points of |ln(z / z_point)|, with z the
    Walls_Depth at each point's pixel, `walls_depth_at_points`, and z_point its camera-space z,
    `scale_points_z`, both (n,) (A14, C5).

    Only the points whose z and z_point are finite and above 0 count. NaN when none does. The
    caller gives a frame with fewer than scale_min_points Scale_Points no Sparse_Error.
    """
    point_z = np.asarray(scale_points_z, dtype=np.float64).reshape(-1)
    walls_z = np.asarray(walls_depth_at_points, dtype=np.float64).reshape(-1)
    if point_z.shape != walls_z.shape:
        raise ValueError("sparse_error: one Walls_Depth per Scale_Point")
    counts = _has_depth(point_z) & _has_depth(walls_z)
    if not counts.any():
        return math.nan
    return float(np.median(np.abs(np.log(walls_z[counts] / point_z[counts]))))


def frame_choice(trusted: bool, timed_out: bool, has_fallback: bool,
                 sparse_error: float | None, scale_points: int,
                 settings: ChoiceSettings) -> tuple[str, str | None]:
    """A Prior_Frame's outcome and, for `depth_prior`, its reason (Req 17.2–17.5, D1, A14, C5).

    `trusted` is whether its Frame_Solve passed its Self_Check, `timed_out` whether it is a
    Timed_Out_Frame, `has_fallback` whether it got a Fallback_Depth, `sparse_error` its
    Sparse_Error or None, and `scale_points` its number of Scale_Points. Gives (`passed`, None),
    (`fallback`, None) or (`depth_prior`, `timed_out` | `no_fallback` | `sparse_error`).
    """
    if trusted and timed_out:
        raise ValueError("frame_choice: a Timed_Out_Frame can't be a Trusted_Frame")
    if timed_out:
        return DEPTH_PRIOR, TIMED_OUT
    if trusted:
        return PASSED, None
    if not has_fallback:
        return DEPTH_PRIOR, NO_FALLBACK
    if scale_points < settings.scale_min_points:
        return FALLBACK, None
    if sparse_error is not None and sparse_error <= settings.fallback_max_sparse_error:
        return FALLBACK, None
    return DEPTH_PRIOR, SPARSE_ERROR
