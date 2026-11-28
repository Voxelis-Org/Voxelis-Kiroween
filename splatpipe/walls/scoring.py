"""The Walls_Step's Score_Windows, Valid_Pairs, Score_Tables, Guess_Scores and Tile_Scores, as pure
numpy functions on the score images of images.py and the warp of geometry.py (walls spec, Req 9,
11.4–11.6, 11.8).

What the scoring works on:
- ScoreFrame: the Prior_Frame's side of the warp at the Score_Size, its world-to-camera Image_Pose,
  the Shared_Camera scaled to the Score_Size and each pixel's Normalised_Coordinates.
- ScoreSource: what one Source_Frame brings, its High_Pass, Bands and Score_Mask at the
  Score_Size, its Start_Depth (k × its Prior_Depth, NaN off its Covered_Pixels) and its Covered
  mask at the Prior_Size, and its Source_Half (D24).
- Windows: the Prior_Frame's Score_Windows, their centres, Tiles and centre Normalised_Coordinates.

The rules:
- score_windows (Req 9.2, A2): the windows on the grid every `window_step` pixels from
  ⌊`window_size` / 2⌋ in, inside the image, whose pixels are all Depth_Pixels and all kept by the
  Score_Mask, and whose High_Pass variance is at least `flat_variance`.
- occluded (Req 9.6, D24, A1): a window is occluded for a Source_Frame when its centre, warped at
  the Current_Depth, lands on no Covered_Pixel of the Source_Frame, or the depth the warp predicts
  there differs from the Source_Frame's Start_Depth under it by more than `occlusion_tolerance` ×
  that Start_Depth. The Prior_Size pixel under a centre that lands at (u, v) of the Source_Frame's
  W × H Score_Size, in COLMAP's convention, is the Depth_Step's pixel rule,
  (min(⌊u·w/W⌋, w − 1), min(⌊v·h/H⌋, h − 1)) for a w × h Prior_Size: the same pixel as the rule on
  the Shared_Camera's full pixels, because u·w/W is the full-size u times w over the full width.
  The test is made once per Score_Table, at the Current_Depth, and holds at every Table_Scale.
- valid_pairs (Req 9.5, A2): a window and a Source_Frame at a Depth_Factor are a Valid_Pair when
  every window pixel warps in front of the Source_Frame and inside it, no bilinear tap with a
  weight above 0 reads a masked Source pixel, the warped High_Pass's variance is at least
  `flat_variance` (always the High_Pass, in every pass), and the window isn't occluded.
- table_scales (Req 9.7, A3): ⌈ln(max / min) / step⌉ + 1 scales, evenly spaced in ln(s), both ends
  included.
- score_table (Req 9.8): C(p, s) as an (S, P) array, row j for the Table_Scale j, the sum of the
  Window_Scores of window p's Valid_Pairs at the Current_Depth × s, 0 where p has none. It is
  always summed as Half_A's table + Half_B's, each over its own Source_Frames in their order, so
  the table has the same bits whether the Half_Tables are asked for or not, and they add up to it.
- guess_score (Req 9.9, 9.10, D11, A3): Σ_p C(p, s(p)), looked up by linear interpolation in ln(s)
  between the two neighbouring Table_Scales, or −∞ for a rejected guess.
- tile_scores (Req 11.4–11.6, 11.8, A6): the two Tile_Scores, against the answer ÷ and ×
  `z_scale`. NaN stands for no Z: fewer than 2 counting Tiles, or a standard error of 0 with
  mean_A − mean_B not above 0. NaN is never above `min_z`, so either fails the Self_Check.

Every sum over many pixels is elementwise numpy, np.sum, np.bincount or images.py's window sums,
and the poses are applied elementwise by geometry.py, so no BLAS call sees them and the bits don't
depend on the thread count (C14). This module imports only the standard library, numpy,
splatpipe.walls.geometry and splatpipe.walls.images.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, overload

import numpy as np

from splatpipe.walls.geometry import Change, depth_factor, warp_maps
from splatpipe.walls.images import (band_mix, warp_image, window_centres, window_ncc,
                                    window_sums, window_variance)

HALVES = ("A", "B")
# How far ln(max / min) / step may be above a whole number from rounding alone and still count as
# that number, so a range that is a whole number of steps doesn't get one scale too many.
_STEPS_ROUNDING = 1e-9


@dataclass(frozen=True)
class Windows:
    """A Prior_Frame's Score_Windows, P of them, in row-major order of their centres."""
    centres_x: np.ndarray  # (P,) int64, window-centre columns in Score_Size pixels
    centres_y: np.ndarray  # (P,) int64, window-centre rows
    tile: np.ndarray       # (P,) int64, each window's Tile, row-major from the top left
    cx: np.ndarray         # (P,) float64, the Normalised_Coordinates of each centre
    cy: np.ndarray

    @property
    def centres(self) -> tuple[np.ndarray, np.ndarray]:
        """(centres_x, centres_y), as images.py's window functions take them."""
        return self.centres_x, self.centres_y

    @property
    def count(self) -> int:
        """P, the number of Score_Windows."""
        return int(self.centres_x.shape[0])


@dataclass(frozen=True)
class ScoreFrame:
    """The Prior_Frame's side of the warp, at the Score_Size."""
    world_to_camera: np.ndarray  # (4, 4) Image_Pose
    k: np.ndarray                # (3, 3) the Shared_Camera at the Score_Size, scaled_intrinsics
    x: np.ndarray                # (h, w) Normalised_Coordinates at the Score_Size
    y: np.ndarray


@dataclass(frozen=True)
class ScoreSource:
    """What one Source_Frame brings to the scoring (D24)."""
    name: str
    world_to_camera: np.ndarray  # (4, 4) Image_Pose
    high_pass: np.ndarray        # (h, w) float64 at the Score_Size
    bands: np.ndarray            # (B, h, w) float64 at the Score_Size
    score_mask: np.ndarray       # (h, w) bool at the Score_Size, true where kept
    start_depth: np.ndarray      # (ph, pw) float64, Prior_Size, k × Prior_Depth, NaN off covered
    covered: np.ndarray          # (ph, pw) bool at the Prior_Size, its Covered_Pixels
    half: str                    # its Source_Half, "A" or "B"

    def __post_init__(self) -> None:
        if self.half not in HALVES:
            raise ValueError(f"Source_Frame {self.name}: its Source_Half must be 'A' or 'B', "
                             f"got {self.half!r}")
        score_shape = np.shape(self.high_pass)
        if (np.shape(self.score_mask) != score_shape or np.ndim(self.bands) != 3
                or np.shape(self.bands)[1:] != score_shape):
            raise ValueError(f"Source_Frame {self.name}: its High_Pass, Bands and Score_Mask must "
                             f"all be at one Score_Size")
        if np.shape(self.covered) != np.shape(self.start_depth):
            raise ValueError(f"Source_Frame {self.name}: its Start_Depth and Covered mask must "
                             f"both be at the Prior_Size")

    @property
    def score_size(self) -> tuple[int, int]:
        """Its Score_Size as (width, height)."""
        height, width = np.shape(self.high_pass)
        return int(width), int(height)


def score_windows(high_pass: np.ndarray, score_mask: np.ndarray, depth_pixel: np.ndarray,
                  x: np.ndarray, y: np.ndarray, *, window_size: int, window_step: int,
                  tile_size: int, flat_variance: float) -> Windows:
    """The Prior_Frame's Score_Windows (Req 9.2, A2).

    The windows on window_centres's grid, so inside the image, with no pixel that isn't a
    Depth_Pixel (`depth_pixel` false) or that the Score_Mask masks (`score_mask` false), and a
    High_Pass variance per pixel of at least `flat_variance`. Each keeps its Tile, the
    `tile_size` × `tile_size` square of the Score_Size its centre lies in, and the
    Normalised_Coordinates `x`, `y` of its centre.
    """
    high_pass = np.asarray(high_pass, dtype=np.float64)
    height, width = high_pass.shape
    for name, array in (("score_mask", score_mask), ("depth_pixel", depth_pixel), ("x", x),
                        ("y", y)):
        if np.shape(array) != high_pass.shape:
            raise ValueError(f"score_windows: {name} must be the High_Pass's shape "
                             f"{high_pass.shape}, got {np.shape(array)}")
    centres = window_centres(width, height, window_size, window_step)
    keep = ((window_sums(~np.asarray(depth_pixel, dtype=bool), centres, window_size) == 0)
            & (window_sums(~np.asarray(score_mask, dtype=bool), centres, window_size) == 0))
    keep &= window_variance(high_pass, centres, window_size) >= flat_variance
    xs, ys = centres[0][keep], centres[1][keep]
    tiles_across = -(-width // tile_size)
    tile = (ys // tile_size) * tiles_across + xs // tile_size
    return Windows(centres_x=xs, centres_y=ys, tile=tile.astype(np.int64),
                   cx=np.asarray(x, dtype=np.float64)[ys, xs],
                   cy=np.asarray(y, dtype=np.float64)[ys, xs])


def occluded(source_z: np.ndarray, centre_u: np.ndarray, centre_v: np.ndarray,
             inside: np.ndarray, start_depth: np.ndarray, covered: np.ndarray,
             score_size: tuple[int, int], tolerance: float) -> np.ndarray:
    """Which windows are occluded for a Source_Frame, as (P,) bool (Req 9.6, D24, A1).

    `source_z` is the depth the warp predicts in the Source_Frame at each window's centre, which
    lands at (`centre_u`, `centre_v`) of its `score_size` (W, H) in COLMAP's convention, and
    `inside` is where it lands in front of it and inside it. The Prior_Size pixel under the
    centre is (min(⌊u·w/W⌋, w − 1), min(⌊v·h/H⌋, h − 1)) for the w × h of `start_depth`. A window
    is occluded when its centre isn't inside, that pixel isn't `covered` or has no finite
    Start_Depth above 0, or |source_z − Start_Depth| > `tolerance` × Start_Depth.
    """
    start_depth = np.asarray(start_depth, dtype=np.float64)
    covered = np.asarray(covered, dtype=bool)
    inside = np.asarray(inside, dtype=bool)
    width, height = score_size
    prior_height, prior_width = start_depth.shape
    with np.errstate(invalid="ignore"):
        u = np.where(inside, np.asarray(centre_u, dtype=np.float64), 0.0)
        v = np.where(inside, np.asarray(centre_v, dtype=np.float64), 0.0)
        column = np.minimum(np.floor(u * prior_width / width).astype(np.int64), prior_width - 1)
        row = np.minimum(np.floor(v * prior_height / height).astype(np.int64), prior_height - 1)
    column = np.maximum(column, 0)
    row = np.maximum(row, 0)
    start = start_depth[row, column]
    source_z = np.asarray(source_z, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        has_start = inside & covered[row, column] & np.isfinite(start) & (start > 0)
        agrees = np.abs(source_z - start) <= float(tolerance) * start
    return ~(has_start & agrees)


def occluded_windows(frame: ScoreFrame, windows: Windows, source: ScoreSource,
                     current_depth: np.ndarray, tolerance: float) -> np.ndarray:
    """occluded for each window of `windows` and `source`, its centre warped at the Score_Size
    `current_depth` itself, the test made once per Score_Table (A1)."""
    centre_depth = np.asarray(current_depth, dtype=np.float64)[windows.centres_y,
                                                               windows.centres_x]
    map_x, map_y, source_z, inside = warp_maps(
        centre_depth, 1.0, frame.world_to_camera, source.world_to_camera, frame.k, windows.cx,
        windows.cy, source_size=source.score_size)
    return occluded(source_z, map_x + 0.5, map_y + 0.5, inside, source.start_depth,
                    source.covered, source.score_size, tolerance)


def valid_pairs(windows: Windows, map_x: np.ndarray, map_y: np.ndarray, inside: np.ndarray,
                source: ScoreSource, occluded_window: np.ndarray, *, window_size: int,
                flat_variance: float) -> np.ndarray:
    """Which windows make a Valid_Pair with `source` at one warp, as (P,) bool (Req 9.5, A2).

    `map_x`, `map_y` and `inside` are geometry.warp_maps's for the whole Score_Size image, and
    `occluded_window` is occluded_windows's at the Current_Depth. A window counts when every pixel
    is `inside`, its warp reads no masked Source pixel (the Score_Mask's masked pixels warped as
    1, which is exactly 0 where every tap is kept), the warped High_Pass's variance per pixel is
    at least `flat_variance`, and the window isn't occluded.
    """
    centres = windows.centres
    inside = np.asarray(inside, dtype=bool)
    valid = (window_sums(~inside, centres, window_size) == 0) & ~np.asarray(occluded_window,
                                                                             dtype=bool)
    masked = ~np.asarray(source.score_mask, dtype=bool)
    if masked.any():
        touched = warp_image(masked.astype(np.float64), map_x, map_y, inside) > 0
        valid &= window_sums(touched, centres, window_size) == 0
    warped = warp_image(source.high_pass, map_x, map_y, inside)
    valid &= window_variance(warped, centres, window_size) >= flat_variance
    return valid


def source_image(source: ScoreSource, weights: np.ndarray | None) -> np.ndarray:
    """The image a Source_Frame is scored on: its High_Pass when `weights` is None, else its
    Band_Mix with the Prior_Frame's own Band_Weights `weights` (Req 10.6)."""
    if weights is None:
        return np.asarray(source.high_pass, dtype=np.float64)
    return band_mix(source.bands, weights)


def pair_scores(frame: ScoreFrame, windows: Windows, source: ScoreSource,
                current_depth: np.ndarray, scale: float, prior_image: np.ndarray,
                scored_image: np.ndarray, occluded_window: np.ndarray, *, window_size: int,
                flat_variance: float) -> tuple[np.ndarray, np.ndarray]:
    """Each window's Window_Score with `source` at the Current_Depth × `scale`, and whether it is
    a Valid_Pair, as (P,) float64 and (P,) bool, the score 0 where it isn't.

    `prior_image` is the Prior_Frame's score image and `scored_image` the Source_Frame's, the
    same kind, from source_image. window_ncc gets a flat_variance of 0, because the flat test is
    valid_pairs's, on the High_Pass (A2).
    """
    map_x, map_y, _, inside = warp_maps(current_depth, float(scale), frame.world_to_camera,
                                        source.world_to_camera, frame.k, frame.x, frame.y,
                                        source_size=source.score_size)
    valid = valid_pairs(windows, map_x, map_y, inside, source, occluded_window,
                        window_size=window_size, flat_variance=flat_variance)
    scores = window_ncc(prior_image, scored_image, map_x, map_y, inside, windows.centres,
                        window_size, 0.0)
    return np.where(valid, scores, 0.0), valid


def table_scales(min_scale: float, max_scale: float, step: float) -> np.ndarray:
    """A pass's Table_Scales, (S,) float64 (Req 9.7, A3): S = ⌈ln(max / min) / step⌉ + 1 scales
    from `min_scale` to `max_scale`, both exactly, evenly spaced in ln(s), so neighbouring scales
    are at most `step` apart in ln(s)."""
    if not (min_scale > 0 and max_scale >= min_scale and step > 0):
        raise ValueError(f"table_scales: wants 0 < min ≤ max and a step above 0, got "
                         f"{min_scale}, {max_scale} and {step}")
    steps = math.log(max_scale / min_scale) / step
    count = math.ceil(steps - _STEPS_ROUNDING * max(steps, 1.0)) + 1
    scales = np.exp(np.linspace(math.log(min_scale), math.log(max_scale), count))
    scales[0] = min_scale
    scales[-1] = max_scale
    return scales


@overload
def score_table(frame: ScoreFrame, windows: Windows, sources: Sequence[ScoreSource],
                current_depth: np.ndarray, scales: np.ndarray, prior_image: np.ndarray, *,
                weights: np.ndarray | None, window_size: int, flat_variance: float,
                occlusion_tolerance: float, halves: Literal[False] = False) -> np.ndarray: ...


@overload
def score_table(frame: ScoreFrame, windows: Windows, sources: Sequence[ScoreSource],
                current_depth: np.ndarray, scales: np.ndarray, prior_image: np.ndarray, *,
                weights: np.ndarray | None, window_size: int, flat_variance: float,
                occlusion_tolerance: float, halves: Literal[True]
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...


def score_table(frame: ScoreFrame, windows: Windows, sources: Sequence[ScoreSource],
                current_depth: np.ndarray, scales: np.ndarray, prior_image: np.ndarray, *,
                weights: np.ndarray | None, window_size: int, flat_variance: float,
                occlusion_tolerance: float, halves: bool = False
                ) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The Score_Table C(p, s) around the Score_Size `current_depth`, (S, P) float64 with row j
    for `scales[j]` (Req 9.8), and with `halves` the Half_Tables of Half_A and Half_B from the
    same warps, as (table, half_a, half_b).

    Each Source_Frame is warped at each Table_Scale. The Prior_Frame's `prior_image` is scored
    against each Source_Frame's High_Pass, with `weights` None, or its Band_Mix with the
    Prior_Frame's own Band_Weights `weights`; the caller gives the Prior_Frame's matching image.
    The occlusion test is made once per Source_Frame at `current_depth` with
    `occlusion_tolerance` and holds at every scale (A1). The table is half_a + half_b.
    """
    scales = np.asarray(scales, dtype=np.float64)
    half_tables = {half: np.zeros((scales.shape[0], windows.count)) for half in HALVES}
    for source in sources:
        hidden = occluded_windows(frame, windows, source, current_depth, occlusion_tolerance)
        scored = source_image(source, weights)
        accumulated = half_tables[source.half]
        for row, scale in enumerate(scales):
            scores, _ = pair_scores(frame, windows, source, current_depth, scale, prior_image,
                                    scored, hidden, window_size=window_size,
                                    flat_variance=flat_variance)
            accumulated[row] = accumulated[row] + scores
    table = half_tables["A"] + half_tables["B"]
    if halves:
        return table, half_tables["A"], half_tables["B"]
    return table


def guess_score(table: np.ndarray, scales: np.ndarray, windows: Windows, z: np.ndarray,
                change: Change, m: float, fit_bound: float) -> float:
    """A guess's Guess_Score F = Σ_p C(p, s(p)) (Req 9.9, 9.10, D11, A3).

    `z` is the Current_Depth, the (h, w) Score_Size image or its (P,) values at the window
    centres, and s(p) the Depth_Factor of `change` on it at p's centre, with `m` its
    Median_Inverse_Depth. C(p, s) is interpolated linearly in ln(s) between the neighbouring
    Table_Scales, so a factor on a Table_Scale reads that scale's value exactly. The guess is
    rejected, −∞, when a number of `change` isn't finite, |b|, |c| or |d| is above `fit_bound`,
    or any s(p) isn't finite or lies outside the Table_Range.
    """
    numbers = np.asarray(change, dtype=np.float64)
    if not np.all(np.isfinite(numbers)) or np.any(np.abs(numbers[1:]) > fit_bound):
        return -math.inf
    z = np.asarray(z, dtype=np.float64)
    centre_depth = z[windows.centres_y, windows.centres_x] if z.ndim == 2 else z
    factor = depth_factor(centre_depth, change, m, windows.cx, windows.cy)
    scales = np.asarray(scales, dtype=np.float64)
    if (not np.all(np.isfinite(factor)) or np.any(factor < scales[0])
            or np.any(factor > scales[-1])):
        return -math.inf
    table = np.asarray(table, dtype=np.float64)
    window = np.arange(windows.count)
    if scales.shape[0] == 1:
        return float(np.sum(table[0, window]))
    logs = np.log(scales)
    log_factor = np.log(factor)
    lower = np.clip(np.searchsorted(logs, log_factor, side="right") - 1, 0, scales.shape[0] - 2)
    fraction = (log_factor - logs[lower]) / (logs[lower + 1] - logs[lower])
    values = table[lower, window] * (1.0 - fraction) + table[lower + 1, window] * fraction
    return float(np.sum(values))


def _shared_sums(frame: ScoreFrame, windows: Windows, sources: Sequence[ScoreSource],
                 answer: np.ndarray, factors: Sequence[float], prior_image: np.ndarray,
                 weights: np.ndarray | None, window_size: int, flat_variance: float,
                 occlusion_tolerance: float) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """For each of `factors`, each window's count of Valid_Pairs at both the answer and the
    answer × factor, and the sums of their Window_Scores at each, as (P,) arrays (A6)."""
    sums = [(np.zeros(windows.count, dtype=np.int64), np.zeros(windows.count),
             np.zeros(windows.count)) for _ in factors]
    settings = {"window_size": window_size, "flat_variance": flat_variance}
    for source in sources:
        hidden = occluded_windows(frame, windows, source, answer, occlusion_tolerance)
        scored = source_image(source, weights)
        at_answer, valid_answer = pair_scores(frame, windows, source, answer, 1.0, prior_image,
                                              scored, hidden, **settings)
        for (count, sum_answer, sum_other), factor in zip(sums, factors):
            at_other, valid_other = pair_scores(frame, windows, source, answer, factor,
                                                prior_image, scored, hidden, **settings)
            shared = valid_answer & valid_other
            count += shared
            sum_answer += np.where(shared, at_answer, 0.0)
            sum_other += np.where(shared, at_other, 0.0)
    return sums


def _z(windows: Windows, count: np.ndarray, sum_answer: np.ndarray, sum_other: np.ndarray,
       z_error_factor: float, tile_min_windows: int) -> tuple[float, int]:
    """One Tile_Score and the number of Tiles it counted (Req 11.6, 11.8, A6). NaN is no Z."""
    tiles = int(windows.tile.max()) + 1 if windows.count else 0
    windows_with_pair = np.bincount(windows.tile, weights=(count > 0).astype(np.float64),
                                    minlength=tiles)
    pairs = np.bincount(windows.tile, weights=count.astype(np.float64), minlength=tiles)
    counting = (windows_with_pair >= tile_min_windows) & (pairs > 0)
    n = int(np.count_nonzero(counting))
    if n < 2:
        return math.nan, n
    mean_a = np.bincount(windows.tile, weights=sum_answer, minlength=tiles)[counting] / pairs[
        counting]
    mean_b = np.bincount(windows.tile, weights=sum_other, minlength=tiles)[counting] / pairs[
        counting]
    difference = float(np.mean(mean_a) - np.mean(mean_b))
    error = z_error_factor * math.sqrt(float(np.var(mean_a, ddof=1)) / n
                                       + float(np.var(mean_b, ddof=1)) / n)
    if error == 0:
        return (math.inf if difference > 0 else math.nan), n
    return difference / error, n


def tile_scores(frame: ScoreFrame, windows: Windows, sources: Sequence[ScoreSource],
                answer_depth: np.ndarray, prior_image: np.ndarray, *,
                weights: np.ndarray | None, z_scale: float, z_error_factor: float,
                tile_min_windows: int, window_size: int, flat_variance: float,
                occlusion_tolerance: float) -> tuple[float, float, int, int]:
    """The two Tile_Scores at the Score_Size `answer_depth`, as (against ÷ z_scale, against
    × z_scale, the Tiles the first counted, the Tiles the second counted) (Req 11.4–11.6, 11.8,
    A6).

    For each other depth, the answer × or ÷ `z_scale`, a Tile counts when at least
    `tile_min_windows` of its windows have a Valid_Pair at both depths, and its two means are over
    those shared Valid_Pairs. Z = (mean_A − mean_B) / (`z_error_factor` × √(var_A / n + var_B / n))
    over the n counting Tiles, with sample variances. It is NaN, no Z, with fewer than 2 Tiles, and
    with a standard error of 0 it is +∞ when mean_A − mean_B is above 0 and NaN otherwise. The
    occlusion test is made at the answer, against each Source_Frame's Start_Depth (D24). The images
    are as score_table's.
    """
    low, high = _shared_sums(frame, windows, sources, answer_depth, (1.0 / z_scale, z_scale),
                             prior_image, weights, window_size, flat_variance,
                             occlusion_tolerance)
    tile_low, low_count = _z(windows, *low, z_error_factor, tile_min_windows)
    tile_high, high_count = _z(windows, *high, z_error_factor, tile_min_windows)
    return tile_low, tile_high, low_count, high_count
