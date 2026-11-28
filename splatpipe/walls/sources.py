"""The Walls_Step's Scene_Scale, Start_Depths and Source_Frames, as pure numpy functions in
float64 (walls spec, Req 7 and 8).

The Scene_Scale (Req 7.1–7.3, A7).
- scale_points: a Prior_Frame's Scale_Points. Each Observed_Point, found with observed_rows, goes
  into the camera with the Image_Pose from world_to_camera, applied elementwise (C14), and goes on
  only when its z is above 0. Its projection with the Shared_Camera at its full size W × H,
  u = fx·X/Z + cx and v = fy·Y/Z + cy in COLMAP's convention, must lie on [0, W) × [0, H). Its
  Prior_Size pixel is the Depth_Step's rule, (min(⌊u·w/W⌋, w − 1), min(⌊v·h/H⌋, h − 1)), and the
  point is a Scale_Point when that pixel is a Covered_Pixel. This is the geometry of
  splatpipe.depth.fit.fit_points, written again here, because no walls file imports
  splatpipe.depth.
- scene_scale: with r = ln(z / the Prior_Depth at each Scale_Point's pixel), r̃ = median(r) and
  D = median(|r − r̃|), the median absolute deviation as it is, not scaled to a standard
  deviation, k = exp(median of the r with |r − r̃| ≤ scale_mad_gate × D). With D = 0 that keeps
  only the r at the median. With fewer than scale_min_points Scale_Points, k = 1.

The Start_Depths (Req 7.4, 7.5).
- start_depth: at the Prior_Size, k × the Prior_Depth, k × (1 / inverse_depth), on the
  Covered_Pixels and NaN elsewhere. At the Score_Size, the Depth_Prior's inverse_depth scaled up
  with the Train_Step's own rule, prior_views.upscale with prior_taps (built on upscale_taps), and
  k / that, on the Depth_Pixels from prior_pixel_mask(mask, None, taps) and NaN elsewhere. The
  keep mask is None there because the Score_Mask is a test of its own (Req 7.5). With k = 1 the
  Prior_Size map is the Prior_Depth itself, which scale_points takes.

The Source_Frames (Req 8.1–8.6, A8, D8).
- Candidate: another Prior_Frame as source choice sees it: its name, Pick_Number, Frame_Time and
  world-to-camera Image_Pose.
- overlap: the share of the Prior_Frame's Covered_Pixels whose point at the Start_Depth lands in
  front of the candidate's camera and inside its image, at the Score_Size.
- image_motion: over the Covered_Pixels whose point lands inside the candidate, the median of
  the distance, in the candidate's Score_Size pixels, between the projections of the point at
  the Start_Depth and at (1 + motion_depth_change) × the Start_Depth. The second point lies on
  the same ray from the Prior_Frame's camera centre c, at c + (1 + δ)·(p − c). A second point
  that isn't in front of the candidate moves by an infinite distance. With no point inside, the
  Image_Motion is 0.
- source_rank: Overlap × min(Image_Motion, source_motion_cap).
- covered_points_world: each Covered_Pixel's point at its Prior_Size Start_Depth, in the world,
  the points overlap and image_motion take.
- choose_sources: the other Prior_Frames whose Frame_Time differs from the frame's by at least
  source_min_seconds are the candidates. Those below source_min_overlap or source_min_motion are
  dropped, the rest ranked by Source_Rank, highest first, ties to the lower Pick_Number, and taken
  in rank order when their Pick_Number differs by at least source_min_gap from every one already
  chosen, up to max_sources. With fewer than min_sources it chooses again with
  source_fallback_seconds. It gives the SourceChoices in rank order and the time it used. A frame
  that still has fewer than min_sources fails its Self_Check (Req 8.7), which the caller decides.

Every pose is applied elementwise with geometry.to_world and geometry.to_camera, and the only
reductions over many pixels are np.median and np.count_nonzero, so no BLAS call sees the pixels
and the bits don't depend on the thread count (C14). Every number comes from [walls] through the
settings the caller gives.

This module imports only the standard library, numpy, splatpipe.walls.geometry and the
Train_Step's colmap, scene and prior_views modules.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from splatpipe.train.colmap import Camera, ModelImage, SparseModel, observed_rows
from splatpipe.train.prior_views import PriorTaps, prior_pixel_mask, upscale
from splatpipe.train.scene import scaled_intrinsics, world_to_camera
from splatpipe.walls.geometry import back_project, normalised_coords, project, to_camera, to_world


class SourceSettings(Protocol):
    """The [walls] settings source choice reads. WallsSettings has them all."""

    max_sources: int
    source_min_seconds: float
    source_fallback_seconds: float
    min_sources: int
    source_min_gap: int
    source_min_overlap: float
    motion_depth_change: float
    source_min_motion: float
    source_motion_cap: float


@dataclass(frozen=True)
class Candidate:
    """A Prior_Frame as source choice sees it (Req 8, D8)."""

    name: str  # its name in images.bin
    pick_number: int  # its Pick's number in frames.json
    frame_time: float  # its Pick's seconds
    world_to_camera: np.ndarray  # (4, 4) float64, its Image_Pose


@dataclass(frozen=True)
class SourceChoice:
    """One chosen Source_Frame, with the numbers that ranked it (Req 8.2–8.4)."""

    candidate: Candidate
    overlap: float
    image_motion: float
    rank: float  # its Source_Rank


def scale_points(model: SparseModel, image: ModelImage, camera: Camera, covered: np.ndarray,
                 prior_depth: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The Scale_Points of `image`, a Prior_Frame of `model`, as (pixels, z, prior_depth_at)
    (Req 7.1, A7).

    `covered` is the Prior_Frame's (h, w) bool Covered_Pixels at the Prior_Size and `prior_depth`
    its (h, w) Prior_Depth, read only on them. `camera` is the Shared_Camera. pixels is (n, 2)
    int64, each Scale_Point's Prior_Size (column, row); z its camera-space z, (n,) float64; and
    prior_depth_at the Prior_Depth at its pixel, (n,) float64, in the order of the image's
    Observed_Points.
    """
    covered = np.asarray(covered, dtype=bool)
    height, width = covered.shape
    pose = world_to_camera(image.quaternion, image.translation)
    points = np.asarray(model.points, dtype=np.float64)[observed_rows(model, image)]
    in_camera = np.stack(to_camera(pose, points[:, 0], points[:, 1], points[:, 2]), axis=-1)
    u, v, z = project(in_camera, scaled_intrinsics(camera, camera.width, camera.height))
    with np.errstate(invalid="ignore"):
        inside = (np.isfinite(z) & (z > 0)
                  & (u >= 0) & (u < camera.width) & (v >= 0) & (v < camera.height))
    u, v, z = u[inside], v[inside], z[inside]
    column = np.minimum(np.floor(u * width / camera.width).astype(np.int64), width - 1)
    row = np.minimum(np.floor(v * height / camera.height).astype(np.int64), height - 1)
    on_covered = covered[row, column]
    column, row = column[on_covered], row[on_covered]
    depth_at = np.asarray(prior_depth, dtype=np.float64)[row, column]
    return np.stack([column, row], axis=-1), z[on_covered], depth_at


def scene_scale(scale_points_z: np.ndarray, prior_depth_at_points: np.ndarray, mad_gate: float,
                min_points: int) -> tuple[float, int]:
    """The Scene_Scale k and the Scale_Point count (Req 7.2, 7.3, A7).

    r = ln(z / Prior_Depth), r̃ = median(r) and D = median(|r − r̃|), unscaled. k = exp(median of
    the r with |r − r̃| ≤ `mad_gate` × D), so a D of 0 keeps only the r at the median. With fewer
    than `min_points` Scale_Points, k = 1. When the gate keeps none, which only a `mad_gate`
    below 1 can do, the r nearest r̃ stay.
    """
    z = np.asarray(scale_points_z, dtype=np.float64)
    depth = np.asarray(prior_depth_at_points, dtype=np.float64)
    count = int(z.size)
    if count < min_points or count == 0:
        return 1.0, count
    r = np.log(z / depth)
    deviation = np.abs(r - np.median(r))
    kept = deviation <= mad_gate * np.median(deviation)
    if not kept.any():
        kept = deviation == deviation.min()
    return float(np.exp(np.median(r[kept]))), count


def start_depth(prior_inverse_depth: np.ndarray, mask: np.ndarray, taps: PriorTaps,
                k: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A Prior_Frame's Start_Depths, as (prior_size, score_size, depth_pixels) (Req 7.4, 7.5).

    `prior_inverse_depth` and `mask` are its Depth_Prior's (h, w) arrays, and `taps` the
    prior_taps from the Prior_Size to the Score_Size. prior_size is k × (1 / inverse_depth) on
    the Covered_Pixels and NaN elsewhere, (h, w) float64. score_size is k / the inverse_depth
    scaled up with upscale, (H, W) float64, on depth_pixels, the (H, W) bool
    prior_pixel_mask(mask, None, taps), and NaN elsewhere.
    """
    inverse = np.asarray(prior_inverse_depth, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    k = float(k)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        prior_size = np.where(mask, k * (1.0 / inverse), np.nan)
        depth_pixels = np.asarray(prior_pixel_mask(mask, None, taps), dtype=bool)
        score_size = np.where(depth_pixels, k / upscale(inverse, taps), np.nan)
    return prior_size, score_size, depth_pixels


def _lands(points_world: np.ndarray, source_pose: np.ndarray, source_k: np.ndarray,
           source_w: int, source_h: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Where each (n, 3) world point lands in the source at the Score_Size, as (u, v, inside):
    inside is true where its z is finite and above 0 and (u, v) lies on [0, W) × [0, H)."""
    points = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
    with np.errstate(over="ignore", invalid="ignore"):
        in_source = np.stack(to_camera(source_pose, points[:, 0], points[:, 1], points[:, 2]),
                             axis=-1)
        u, v, z = project(in_source, source_k)
        inside = (np.isfinite(z) & (z > 0)
                  & (u >= 0) & (u < source_w) & (v >= 0) & (v < source_h))
    return u, v, inside


def overlap(covered_points_world: np.ndarray, source_pose: np.ndarray, source_k: np.ndarray,
            source_w: int, source_h: int) -> float:
    """The Overlap: the share of the (n, 3) world points of a Prior_Frame's Covered_Pixels at
    the Start_Depth that land in front of the source's camera and inside its `source_w` ×
    `source_h` image, with `source_k` the Shared_Camera at that size (Req 8.2, A8). 0 with no
    point."""
    _, _, inside = _lands(covered_points_world, source_pose, source_k, source_w, source_h)
    if inside.size == 0:
        return 0.0
    return float(np.count_nonzero(inside) / inside.size)


def image_motion(covered_points_world: np.ndarray, centre: np.ndarray, source_pose: np.ndarray,
                 source_k: np.ndarray, source_w: int, source_h: int,
                 motion_depth_change: float) -> float:
    """The Image_Motion, in the source's Score_Size pixels (Req 8.3, A8).

    Each (n, 3) world point p of a Covered_Pixel at the Start_Depth, with `centre` the
    Prior_Frame's camera centre c, moves to c + (1 + `motion_depth_change`)·(p − c), the same
    pixel's point at that much more depth. Over the points p that land inside the source, this
    is the median distance between the two projections, infinite where the moved point isn't in
    front of the source. 0 when no point lands inside.
    """
    points = np.asarray(covered_points_world, dtype=np.float64).reshape(-1, 3)
    c = np.asarray(centre, dtype=np.float64)
    grow = 1.0 + float(motion_depth_change)
    moved = np.stack([c[axis] + grow * (points[:, axis] - c[axis]) for axis in range(3)],
                     axis=-1)
    u, v, inside = _lands(points, source_pose, source_k, source_w, source_h)
    if not inside.any():
        return 0.0
    moved_points = moved[inside]
    with np.errstate(over="ignore", invalid="ignore"):
        moved_source = np.stack(to_camera(source_pose, moved_points[:, 0], moved_points[:, 1],
                                          moved_points[:, 2]), axis=-1)
        moved_u, moved_v, moved_z = project(moved_source, source_k)
        distance = np.hypot(moved_u - u[inside], moved_v - v[inside])
        in_front = np.isfinite(moved_z) & (moved_z > 0) & np.isfinite(distance)
    return float(np.median(np.where(in_front, distance, np.inf)))


def source_rank(overlap: float, motion: float, motion_cap: float) -> float:
    """The Source_Rank, Overlap × min(Image_Motion, `motion_cap`) (Req 8.4)."""
    return float(overlap) * min(float(motion), float(motion_cap))


def covered_points_world(start: np.ndarray, covered: np.ndarray, camera: Camera,
                         pose: np.ndarray) -> np.ndarray:
    """The world point of each Covered_Pixel at its Prior_Size Start_Depth `start`, (n, 3)
    float64 in row-major pixel order, with Normalised_Coordinates by the Prior_Format's rule and
    the world-to-camera `pose` applied elementwise."""
    covered = np.asarray(covered, dtype=bool)
    height, width = covered.shape
    x, y = normalised_coords(camera, width, height)
    in_camera = back_project(np.asarray(start, dtype=np.float64)[covered], x[covered], y[covered])
    return np.stack(to_world(pose, in_camera[:, 0], in_camera[:, 1], in_camera[:, 2]), axis=-1)


def choose_sources(frame: Candidate, start: np.ndarray, covered: np.ndarray, camera: Camera,
                   score_size: tuple[int, int], candidates: Sequence[Candidate],
                   settings: SourceSettings) -> tuple[list[SourceChoice], float]:
    """A Prior_Frame's Source_Frames in rank order, and the time cut they were chosen with,
    source_min_seconds or source_fallback_seconds (Req 8.1–8.6, A8, D8).

    `frame` is the Prior_Frame as a Candidate, `start` its Prior_Size Start_Depth, `covered` its
    Covered_Pixels, `camera` the Shared_Camera and `score_size` the Score_Size (W, H) every
    candidate is measured at. `candidates` are the other Prior_Frames. One with `frame`'s name is
    never a candidate. Each candidate's Overlap and Image_Motion are worked out once, for every
    candidate the looser of the two time cuts lets through.
    """
    width, height = score_size
    k = scaled_intrinsics(camera, width, height)
    points = covered_points_world(start, covered, camera, frame.world_to_camera)
    centre = np.array(to_world(frame.world_to_camera, np.zeros(1), np.zeros(1), np.zeros(1)),
                      dtype=np.float64).reshape(3)
    loosest = min(settings.source_min_seconds, settings.source_fallback_seconds)
    scored = []
    for candidate in candidates:
        if candidate.name == frame.name or _apart(frame, candidate) < loosest:
            continue
        share = overlap(points, candidate.world_to_camera, k, width, height)
        if share < settings.source_min_overlap:
            continue
        motion = image_motion(points, centre, candidate.world_to_camera, k, width, height,
                              settings.motion_depth_change)
        if motion < settings.source_min_motion:
            continue
        rank = source_rank(share, motion, settings.source_motion_cap)
        scored.append(SourceChoice(candidate, share, motion, rank))
    scored.sort(key=lambda choice: (-choice.rank, choice.candidate.pick_number))
    chosen = _pick(frame, scored, settings.source_min_seconds, settings)
    if len(chosen) >= settings.min_sources:
        return chosen, float(settings.source_min_seconds)
    return (_pick(frame, scored, settings.source_fallback_seconds, settings),
            float(settings.source_fallback_seconds))


def _apart(frame: Candidate, candidate: Candidate) -> float:
    """How far apart in time the two Prior_Frames are, in seconds."""
    return abs(float(candidate.frame_time) - float(frame.frame_time))


def _pick(frame: Candidate, ranked: Sequence[SourceChoice], seconds: float,
          settings: SourceSettings) -> list[SourceChoice]:
    """The ranked choices at least `seconds` from `frame`, taken greedily in rank order when
    their Pick_Number is at least source_min_gap from every one already taken, up to
    max_sources (Req 8.5)."""
    chosen: list[SourceChoice] = []
    for choice in ranked:
        if len(chosen) >= settings.max_sources:
            break
        if _apart(frame, choice.candidate) < seconds:
            continue
        number = choice.candidate.pick_number
        if all(abs(number - taken.candidate.pick_number) >= settings.source_min_gap
               for taken in chosen):
            chosen.append(choice)
    return chosen
