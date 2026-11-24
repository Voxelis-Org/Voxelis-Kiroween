"""The Back_Projected_Splats: extra Start_Splats on the plain surfaces of the Prior_Views, worked
out with pure numpy functions (depth-priors spec, Req 12, A7, O5).

- back_project_budget (A7): the Back_Project_Budget, min(⌊backproject_share × max_splats⌋,
  backproject_max).
- landed_pixels (Req 12.1, O5): the training pixels a Frame's Observed_Points land on under the
  view's scaled camera. Each point in front of the camera rules out the one pixel it lands on.
- candidate_pixels (Req 12.1): a Prior_View's Candidate_Pixels, its Prior_Pixels that are
  Plain_Pixels and that no Observed_Point lands on.
- draw_candidates (Req 12.2): which Candidate_Pixels, numbered over every Prior_View, the
  Back_Projected_Splats start from, drawn without replacement by a generator of their own,
  seeded with seed.
- world_points (Req 12.3): the world point of each chosen pixel's centre at the camera-space z
  of 1 / its scaled-up inverse_depth, in float64.
- start_points (Req 12.2–12.4, 12.6, 12.8): the points and 8-bit colours the Start_Splats start
  from. With no Prior_Set they're exactly choose_points' Sparse_Points. With one, they're the
  Sparse_Points that max_splats − B leaves room for, then the B Back_Projected_Splats in view and
  pixel order.

The step sizes every Start_Splat together afterwards, with the same 3-nearest-neighbour rule, and
gives them all the same opacity, rotation and f_rest (Req 12.4, 12.5).

This module imports only the standard library, numpy and splatpipe, and no torch (Req 12.9).
From splatpipe it imports colmap.py, prior_views.py and splats.py, and scene.py and settings.py
for their types.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from splatpipe.train.colmap import SparseModel, observed_rows
from splatpipe.train.prior_views import PriorTaps, plain_pixels, prior_pixel_mask, upscale
from splatpipe.train.scene import Scene, TrainingView
from splatpipe.train.settings import TrainSettings
from splatpipe.train.splats import choose_points

# Pixel (x, y) covers [x, x + 1) × [y, y + 1), so its centre is at (x + 0.5, y + 0.5), as COLMAP
# places pixel centres.
_PIXEL_CENTRE = 0.5


@dataclass(frozen=True)
class StartPoints:
    """The points the Start_Splats start from, with their colours."""

    points: np.ndarray  # (N, 3) float64: the Sparse_Points used, then the Back_Projected_Splats
    colours: np.ndarray  # (N, 3) uint8, RGB
    sparse: int  # Sparse_Points used
    back_projected: int  # Back_Projected_Splats


def back_project_budget(settings: TrainSettings) -> int:
    """The Back_Project_Budget (A7): min(⌊backproject_share × max_splats⌋, backproject_max).

    333,333 for fast's 1,000,000 splats, and 500,000 for default and quality.
    """
    share = math.floor(settings.backproject_share * settings.max_splats)
    return min(share, settings.backproject_max)


def landed_pixels(points: np.ndarray, view: TrainingView, width: int, height: int) -> np.ndarray:
    """The (height, width) bool training pixels that the (K, 3) world `points`, a Frame's
    Observed_Points, land on (Req 12.1, O5).

    With the view's scaled K and world-to-camera matrix, a point lands when its camera-space z is
    above 0 and (⌊u⌋, ⌊v⌋), with u = fx·x / z + cx and v = fy·y / z + cy, is inside the
    Training_Size. Each point rules out only that one pixel. A point so close to the camera's
    plane that u or v overflows to an infinity lands on no pixel.
    """
    landed = np.zeros((height, width), dtype=bool)
    rotation = view.world_to_camera[:3, :3]
    translation = view.world_to_camera[:3, 3]
    camera = points @ rotation.T + translation
    front = camera[camera[:, 2] > 0]
    k = view.intrinsics
    with np.errstate(over="ignore", invalid="ignore"):
        columns = np.floor(k[0, 0] * front[:, 0] / front[:, 2] + k[0, 2])
        rows = np.floor(k[1, 1] * front[:, 1] / front[:, 2] + k[1, 2])
    inside = (columns >= 0) & (columns < width) & (rows >= 0) & (rows < height)
    landed[rows[inside].astype(np.int64), columns[inside].astype(np.int64)] = True
    return landed


def candidate_pixels(view: TrainingView, points: np.ndarray, taps: PriorTaps,
                     settings: TrainSettings, width: int, height: int) -> np.ndarray:
    """The (height, width) bool Candidate_Pixels of `view`, whose Observed_Points are the (K, 3)
    world `points` (Req 12.1).

    They're the view's Prior_Pixels, through the taps from the Prior_Size, that are Plain_Pixels
    and that no Observed_Point lands on. A view with no Prior has no Prior_Pixel, so none.
    """
    if view.prior is None:
        return np.zeros((height, width), dtype=bool)
    return (prior_pixel_mask(view.prior.prior.mask, view.keep, taps)
            & plain_pixels(view.prior.gradient, settings)
            & ~landed_pixels(points, view, width, height))


def draw_candidates(counts: Sequence[int], budget: int, seed: int) -> np.ndarray:
    """The Candidate_Pixels the Back_Projected_Splats start from, as sorted int64 numbers from 0
    to C − 1, with the Candidate_Pixels numbered over every Prior_View and `counts` each view's
    number of them (Req 12.2).

    With C = Σ counts, it draws B = min(budget, C) distinct numbers, with an equal chance for
    each, by np.random.default_rng(seed).choice(C, B, replace=False). B = 0 makes no generator.
    """
    total = sum(int(count) for count in counts)
    drawn = min(budget, total)
    if drawn == 0:
        return np.empty(0, dtype=np.int64)
    chosen = np.random.default_rng(seed).choice(total, drawn, replace=False)
    return np.sort(chosen).astype(np.int64, copy=False)


def world_points(columns: np.ndarray, rows: np.ndarray, inverse_depth: np.ndarray,
                 intrinsics: np.ndarray, world_to_camera: np.ndarray) -> np.ndarray:
    """The (B, 3) float64 world points of the training pixels (`columns`, `rows`), each at the
    camera-space z of 1 / its scaled-up `inverse_depth` (Req 12.3).

    With K the view's scaled `intrinsics`, the camera point of pixel (x, y) is
    ((x + 0.5 − cx) / fx · z, (y + 0.5 − cy) / fy · z, z), and its world point is Rᵀ(c − t), with
    R and t from `world_to_camera`. Every step is float64.
    """
    z = 1.0 / np.asarray(inverse_depth, dtype=np.float64)
    across = (np.asarray(columns, dtype=np.float64) + _PIXEL_CENTRE - intrinsics[0, 2])
    down = (np.asarray(rows, dtype=np.float64) + _PIXEL_CENTRE - intrinsics[1, 2])
    camera = np.stack([across / intrinsics[0, 0] * z, down / intrinsics[1, 1] * z, z], axis=1)
    rotation = world_to_camera[:3, :3]
    translation = world_to_camera[:3, 3]
    return (camera - translation) @ rotation


def start_points(model: SparseModel, scene: Scene, settings: TrainSettings) -> StartPoints:
    """The points and colours the Start_Splats start from (Req 12.2–12.4, 12.6, 12.8).

    With no Prior_Set, scene.taps is None and they're exactly choose_points' Sparse_Points for
    max_splats, as the splat-training spec gives them (Req 12.8). With one, each Prior_View's
    Candidate_Pixels are worked out once, numbered in view order and then row-major order, and B
    of them drawn (Req 12.1, 12.2). Each drawn pixel becomes a world point at its Prior's depth,
    with its 8-bit colour (Req 12.3, 12.4). The Sparse_Points come first, from choose_points for
    max_splats − B, then the Back_Projected_Splats in view and pixel order (Req 12.6).
    """
    if scene.taps is None:
        points, colours = choose_points(model.points, model.colours, settings.max_splats,
                                        settings.seed)
        return StartPoints(points, colours, len(points), 0)
    budget = back_project_budget(settings)
    masks = _candidate_masks(model, scene, settings) if budget else [None] * len(scene.views)
    counts = [0 if mask is None else int(np.count_nonzero(mask)) for mask in masks]
    chosen = draw_candidates(counts, budget, settings.seed)
    new_points, new_colours = _back_projected(scene, masks, counts, chosen)
    points, colours = choose_points(model.points, model.colours,
                                    settings.max_splats - len(chosen), settings.seed)
    return StartPoints(np.concatenate([points, new_points]),
                       np.concatenate([colours, new_colours]), len(points), len(chosen))


def _candidate_masks(model: SparseModel, scene: Scene,
                     settings: TrainSettings) -> list[np.ndarray | None]:
    """Each view's Candidate_Pixels, in scene order, worked out once: None for a view with no
    Prior. Each Prior_View's Observed_Points are its image's, as observed_rows maps them."""
    masks: list[np.ndarray | None] = []
    for image, view in zip(model.images, scene.views, strict=True):
        if view.prior is None or scene.taps is None:
            masks.append(None)
            continue
        observed = model.points[observed_rows(model, image)]
        masks.append(candidate_pixels(view, observed, scene.taps, settings, scene.width,
                                      scene.height))
    return masks


def _back_projected(scene: Scene, masks: Sequence[np.ndarray | None], counts: Sequence[int],
                    chosen: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The world points, (B, 3) float64, and 8-bit RGB colours, (B, 3) uint8, of the `chosen`
    Candidate_Pixels, in view and pixel order.

    A view's numbers run from the sum of the counts before it, so the chosen numbers of view v
    are the sorted ones from starts[v] up to ends[v], and number n there is the
    (n − starts[v])-th true pixel of its mask in row-major order. The Prior's inverse_depth is
    scaled up one view at a time, and only for views with a chosen pixel.
    """
    ends = np.cumsum(np.asarray(counts, dtype=np.int64))
    starts = ends - np.asarray(counts, dtype=np.int64)
    points = [np.empty((0, 3), dtype=np.float64)]
    colours = [np.empty((0, 3), dtype=np.uint8)]
    for view, mask, start, end in zip(scene.views, masks, starts, ends, strict=True):
        first, last = np.searchsorted(chosen, [start, end])
        if first == last or mask is None or view.prior is None or scene.taps is None:
            continue
        flat = np.flatnonzero(mask)[chosen[first:last] - start]
        rows, columns = np.divmod(flat, scene.width)
        inverse_depth = upscale(view.prior.prior.inverse_depth, scene.taps)[rows, columns]
        points.append(world_points(columns, rows, inverse_depth, view.intrinsics,
                                   view.world_to_camera))
        colours.append(view.pixels[rows, columns])
    return np.concatenate(points), np.concatenate(colours)
