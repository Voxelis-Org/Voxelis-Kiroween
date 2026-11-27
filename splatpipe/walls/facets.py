"""The Walls_Step's Facets: cutting a Prior_Frame's Depth_Prior into planar pieces, refitting each
on its own Score_Windows and applying the changes that pass, as pure functions on arrays (walls
spec, Req 12.1–12.6, D9, D13, D24, A5, A9).

- jump_pixels (Req 12.2): a Covered_Pixel is a Jump_Pixel when its `inverse_depth` and that of a
  Covered_Pixel `facet_jump_distance` pixels away across or down differ by more than `facet_jump`
  of the smaller of the two. Both pixels of such a pair are Jump_Pixels, so a depth jump is a
  band on both of its sides.
- cut_facets (Req 12.1, 12.3, A9): the seed-and-grow of the design's decisions table, at the
  Prior_Size, on the Depth_Prior's `normal`. The candidates are the Covered_Pixels that aren't
  Jump_Pixels and have a finite normal of length above 0, each normal set to unit length. The
  next seed is always the first candidate in row-major order that is free and hasn't seeded yet,
  so a pixel freed by an earlier region's clean-up may seed later. A seed grows a 4-connected
  region by breadth-first search over the free candidates, a neighbour joining when the angle
  between its normal and the region's running mean normal, the sum of the region's normals so
  far, is at most `facet_normal_degrees`. The clean-up then repeats until nothing changes: drop
  every pixel more than that angle from the region's mean normal, then keep only the region's
  largest 4-connected piece, the first in row-major order on a tie. Dropped pixels are free again.
  A region of at least `facet_min_pixels` pixels is a Facet; a smaller one is no Facet, and its
  pixels are taken, so they join no later region. The Facets come back largest first, ties in the
  order they were cut, each with its pixels in row-major order, its normal (the normalised mean
  of its pixels' unit normals) and m, the median of the Depth_Prior's `inverse_depth` over them.
- refit_facet (Req 12.4, 12.5, D13, D24, A5): inside a Trusted_Frame's Frame_Solve, the Facet's
  Score_Table and two Half_Tables on the image the Third_Pass scored (the Band_Mix, or the
  High_Pass when every Band_Weight is 0), around the Solved_Depth before any Facet_Change, with
  the facet_* Table_Scales, over the Score_Windows whose centre's Prior_Size pixel lies in the
  Facet, by scene.keep_mask's nearest-pixel rule: Score_Size centre (X, Y) of a W × H Score_Size
  is Prior_Size pixel ((2X + 1)·w // (2W), (2Y + 1)·h // (2H)) of a w × h Prior_Size. The
  tables' occlusion test is made at the Solved_Depth with `facet_occlusion_tolerance`. The
  Change_Fit runs on each table with the Facet_Change's m, the Median_Inverse_Depth of the
  Solved_Depth over the Facet's pixels, the depth map the change applies to (Req 6.1); the
  Facet's own m is the Depth_Prior's, which the planes use. The three gates of Req 12.5:
  1. the Halves_Difference over the Facet's pixels, passes.halves_difference's rule, is at most
     `facet_halves_tolerance`;
  2. the change, the median over the Facet's pixels of |z_facet / z − 1|, with z the Solved_Depth
     and z_facet the Solved_Depth through the Facet_Change, is above `facet_min_change_ratio` ×
     that Halves_Difference. A Facet pixel the change gives no depth makes the median NaN, which
     is never above it;
  3. both Tile_Scores at the Solved_Depth through the Facet_Change, over the Facet's own
     Score_Windows and so the Tiles their centres lie in, are at least `facet_min_z`. Their
     occlusion test is Req 11.5's, at that answer with `occlusion_tolerance`, because Req 12.4
     names the looser tolerance for the refit's tables only. A NaN Tile_Score, no Z (fewer than 2
     Tiles with `tile_min_windows` windows), is never at least `facet_min_z`.
  The gates are checked in that order, and the Tile_Scores' warps are made only when the first
  two pass. A Facet with no Score_Window, or with a pixel whose Solved_Depth isn't finite and
  above 0, gets no FacetChange.
- apply_facet_change (Req 12.6): the Facet's pixels get the Facet_Change's depth and its
  Changed_Normal, NaN where the change gives none (geometry.py's rule), and every other pixel
  keeps its depth and normal as they were, bit for bit.

The growth runs in plain Python floats over the pixels in a fixed order, and every other sum over
many pixels is elementwise numpy, np.sum or np.median, so no BLAS call sees them and the bits don't
depend on the thread count (C14). This module imports only the standard library, numpy,
scipy.ndimage and the walls pure modules.
"""

from __future__ import annotations

import heapq
import math
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Protocol

import numpy as np
from scipy import ndimage

from splatpipe.walls.fit import FourNumbers, change_fit
from splatpipe.walls.geometry import apply_change, changed_normal, median_inverse_depth
from splatpipe.walls.passes import (PassDepth, PassSettings, PriorScore, ThirdPass,
                                    halves_difference)
from splatpipe.walls.scoring import ScoreSource, Windows, score_table, table_scales, tile_scores

# A pixel's state while the Facets are cut.
_TAKEN = 0      # not a candidate, or taken by a region
_FREE = 1       # a candidate no region holds
_GROWING = 2    # in the region being grown


class FacetCutSettings(Protocol):
    """The `[walls]` settings cut_facets and jump_pixels read, as WallsSettings holds them."""
    facet_normal_degrees: float
    facet_jump: float
    facet_jump_distance: int
    facet_min_pixels: int


class FacetSettings(FacetCutSettings, PassSettings, Protocol):
    """The `[walls]` settings refit_facet reads, as WallsSettings holds them."""
    facet_min_scale: float
    facet_max_scale: float
    facet_scale_step: float
    facet_occlusion_tolerance: float
    facet_halves_tolerance: float
    facet_min_change_ratio: float
    facet_min_z: float


@dataclass(frozen=True)
class Facet:
    """A planar piece of a Prior_Frame's Depth_Prior at the Prior_Size (Req 12.3)."""
    pixels: np.ndarray  # (n, 2) int64, (row, col) at the Prior_Size, in row-major order
    normal: np.ndarray  # (3,) float64, the normalised mean of its pixels' unit normals
    m: float            # the median of the Depth_Prior's inverse_depth over its pixels

    @property
    def count(self) -> int:
        """Its number of pixels."""
        return int(self.pixels.shape[0])

    def mask(self, shape: tuple[int, int]) -> np.ndarray:
        """Its pixels as a bool image of the Prior_Size `shape` (height, width)."""
        mask = np.zeros(shape, dtype=bool)
        mask[self.pixels[:, 0], self.pixels[:, 1]] = True
        return mask


@dataclass(frozen=True)
class FacetChange:
    """A Facet_Change that passed the three gates of Req 12.5, relative to the Solved_Depth."""
    facet: int                 # the Facet's index in cut_facets's list
    change: FourNumbers        # (a, b, c, d)
    m: float                   # the Solved_Depth's Median_Inverse_Depth over the Facet's pixels
    halves_difference: float
    tile_low: float            # Z against ÷ z_scale
    tile_high: float           # Z against × z_scale


def jump_pixels(inverse_depth: np.ndarray, covered: np.ndarray, jump: float,
                distance: int) -> np.ndarray:
    """The Jump_Pixels, as an (h, w) bool image (Req 12.2).

    For each pair of Covered_Pixels `distance` pixels apart across or down, both are Jump_Pixels
    when |ι₁ − ι₂| > `jump` × min(ι₁, ι₂), with ι the `inverse_depth`.
    """
    inverse = np.asarray(inverse_depth, dtype=np.float64)
    covered = np.asarray(covered, dtype=bool)
    if inverse.shape != covered.shape or inverse.ndim != 2:
        raise ValueError(f"jump_pixels: inverse_depth and covered must be one (h, w) shape, got "
                         f"{inverse.shape} and {covered.shape}")
    if distance < 1:
        raise ValueError(f"jump_pixels: the distance must be at least 1, got {distance}")
    jumps = np.zeros(inverse.shape, dtype=bool)
    for axis in (1, 0):
        if inverse.shape[axis] <= distance:
            continue
        near = [slice(None), slice(None)]
        far = [slice(None), slice(None)]
        near[axis] = slice(None, -distance)
        far[axis] = slice(distance, None)
        first, second = inverse[tuple(near)], inverse[tuple(far)]
        with np.errstate(invalid="ignore"):
            differ = (covered[tuple(near)] & covered[tuple(far)]
                      & (np.abs(first - second) > jump * np.minimum(first, second)))
        jumps[tuple(near)] |= differ
        jumps[tuple(far)] |= differ
    return jumps


def _unit_normals(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Each pixel's normal set to unit length, (h, w, 3) float64, and where it has one: finite,
    with a length above 0."""
    normal = np.asarray(normal, dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        length = np.hypot(np.hypot(normal[..., 0], normal[..., 1]), normal[..., 2])
        good = np.all(np.isfinite(normal), axis=-1) & np.isfinite(length) & (length > 0)
        unit = normal / np.where(good, length, 1.0)[..., np.newaxis]
    return np.where(good[..., np.newaxis], unit, 0.0), good


def _grow(seed: int, state: bytearray, normals: tuple[list, list, list], width: int,
          height: int, cos_limit: float) -> list[int]:
    """The region a seed grows by breadth-first search over the free candidates, each neighbour
    joining when its normal is within the angle of the region's running mean normal. The
    region's pixels are left _GROWING in `state`."""
    nx, ny, nz = normals
    sum_x, sum_y, sum_z = nx[seed], ny[seed], nz[seed]
    state[seed] = _GROWING
    region = [seed]
    queue = deque((seed,))
    while queue:
        pixel = queue.popleft()
        row, column = divmod(pixel, width)
        for neighbour, inside in ((pixel - width, row > 0), (pixel + width, row < height - 1),
                                  (pixel - 1, column > 0), (pixel + 1, column < width - 1)):
            if not inside or state[neighbour] != _FREE:
                continue
            length = math.sqrt(sum_x * sum_x + sum_y * sum_y + sum_z * sum_z)
            dot = nx[neighbour] * sum_x + ny[neighbour] * sum_y + nz[neighbour] * sum_z
            if dot >= cos_limit * length:
                state[neighbour] = _GROWING
                sum_x += nx[neighbour]
                sum_y += ny[neighbour]
                sum_z += nz[neighbour]
                region.append(neighbour)
                queue.append(neighbour)
    return region


def _mean_normal(unit: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """The normalised mean of the unit normals at the flat `indices`, (3,) float64."""
    total = np.sum(unit[indices], axis=0)
    return total / math.sqrt(float(np.sum(total * total)))


def _largest_piece(indices: np.ndarray, width: int) -> np.ndarray:
    """The largest 4-connected piece of the pixels at the sorted flat `indices`, the first in
    row-major order on a tie, as sorted flat indices."""
    rows, columns = np.divmod(indices, width)
    top, left = int(rows.min()), int(columns.min())
    box = np.zeros((int(rows.max()) - top + 1, int(columns.max()) - left + 1), dtype=bool)
    box[rows - top, columns - left] = True
    labels, count = ndimage.label(box)
    if count <= 1:
        return indices
    sizes = np.bincount(labels.ravel(), minlength=count + 1)
    sizes[0] = 0
    keep = labels[rows - top, columns - left] == int(np.argmax(sizes))
    return indices[keep]


def _clean(region: np.ndarray, unit: np.ndarray, width: int, cos_limit: float) -> np.ndarray:
    """The clean-up: drop every pixel more than the angle from the mean normal, then keep the
    largest 4-connected piece, until nothing changes. Sorted flat indices, maybe empty."""
    while region.size:
        mean = _mean_normal(unit, region)
        dots = (unit[region, 0] * mean[0] + unit[region, 1] * mean[1]
                + unit[region, 2] * mean[2])
        kept = region[dots >= cos_limit]
        if kept.size:
            kept = _largest_piece(kept, width)
        if kept.size == region.size:
            return region
        region = kept
    return region


class _Seeds:
    """The next seed: the first free candidate in row-major order that hasn't seeded yet."""

    def __init__(self, candidates: np.ndarray, state: bytearray) -> None:
        self._order = np.flatnonzero(candidates).tolist()
        self._pointer = 0
        self._freed: list[int] = []
        self._seeded = bytearray(len(state))
        self._state = state

    def _usable(self, pixel: int) -> bool:
        return self._state[pixel] == _FREE and not self._seeded[pixel]

    def free(self, pixels: Sequence[int]) -> None:
        """Pixels a clean-up freed, which may seed later."""
        for pixel in pixels:
            heapq.heappush(self._freed, pixel)

    def next(self) -> int | None:
        """The next seed, marked as seeded, or None when there's none."""
        while self._freed and not self._usable(self._freed[0]):
            heapq.heappop(self._freed)
        while self._pointer < len(self._order) and not self._usable(self._order[self._pointer]):
            self._pointer += 1
        ahead = self._order[self._pointer] if self._pointer < len(self._order) else None
        behind = self._freed[0] if self._freed else None
        if ahead is None and behind is None:
            return None
        if behind is not None and (ahead is None or behind < ahead):
            seed = heapq.heappop(self._freed)
        else:
            seed = ahead
            self._pointer += 1
        self._seeded[seed] = 1
        return seed


def cut_facets(normal: np.ndarray, inverse_depth: np.ndarray, covered: np.ndarray,
               settings: FacetCutSettings) -> list[Facet]:
    """A Prior_Frame's Facets, from its Depth_Prior's (h, w, 3) `normal`, (h, w) `inverse_depth`
    and (h, w) `covered` mask at the Prior_Size, largest first (Req 12.1, 12.3, A9)."""
    inverse = np.asarray(inverse_depth, dtype=np.float64)
    covered = np.asarray(covered, dtype=bool)
    if (inverse.ndim != 2 or covered.shape != inverse.shape
            or np.shape(normal) != inverse.shape + (3,)):
        raise ValueError(f"cut_facets: wants an (h, w, 3) normal and (h, w) inverse_depth and "
                         f"covered, got {np.shape(normal)}, {inverse.shape} and {covered.shape}")
    height, width = inverse.shape
    unit, has_normal = _unit_normals(normal)
    jumps = jump_pixels(inverse, covered, settings.facet_jump, settings.facet_jump_distance)
    candidates = (covered & ~jumps & has_normal).ravel()
    state = bytearray(np.where(candidates, _FREE, _TAKEN).astype(np.uint8).tobytes())
    flat_unit = unit.reshape(-1, 3)
    python_normals = (flat_unit[:, 0].tolist(), flat_unit[:, 1].tolist(),
                      flat_unit[:, 2].tolist())
    cos_limit = math.cos(math.radians(settings.facet_normal_degrees))
    seeds = _Seeds(candidates, state)
    flat_inverse = inverse.ravel()
    facets: list[Facet] = []
    while (seed := seeds.next()) is not None:
        grown = np.sort(np.asarray(_grow(seed, state, python_normals, width, height, cos_limit),
                                   dtype=np.int64))
        region = _clean(grown, flat_unit, width, cos_limit)
        freed = np.setdiff1d(grown, region, assume_unique=True)
        for pixel in freed.tolist():
            state[pixel] = _FREE
        for pixel in region.tolist():
            state[pixel] = _TAKEN
        seeds.free(freed.tolist())
        if region.size >= settings.facet_min_pixels:
            rows, columns = np.divmod(region, width)
            facets.append(Facet(pixels=np.stack([rows, columns], axis=1),
                                normal=_mean_normal(flat_unit, region),
                                m=float(np.median(flat_inverse[region]))))
    return sorted(facets, key=lambda facet: -facet.count)


def facet_windows(facet: Facet, windows: Windows, score_size: tuple[int, int],
                  prior_size: tuple[int, int]) -> Windows:
    """The Score_Windows whose centre's Prior_Size pixel lies in the Facet, by scene.keep_mask's
    nearest-pixel rule: centre (X, Y) of the (W, H) `score_size` is pixel
    ((2X + 1)·w // (2W), (2Y + 1)·h // (2H)) of the (w, h) `prior_size`. Each keeps its Tile."""
    score_width, score_height = score_size
    prior_width, prior_height = prior_size
    columns = (2 * windows.centres_x + 1) * prior_width // (2 * score_width)
    rows = (2 * windows.centres_y + 1) * prior_height // (2 * score_height)
    inside = facet.mask((prior_height, prior_width))[rows, columns]
    return Windows(centres_x=windows.centres_x[inside], centres_y=windows.centres_y[inside],
                   tile=windows.tile[inside], cx=windows.cx[inside], cy=windows.cy[inside])


def _changed_depth(depth: np.ndarray, change: FourNumbers, m: float, x: np.ndarray,
                   y: np.ndarray) -> np.ndarray:
    """`depth` through the Bas_Relief_Change `change` with the m `m`, NaN where it gives none."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return 1.0 / apply_change(1.0 / np.asarray(depth, dtype=np.float64), change, m, x, y)


def refit_facet(facet: Facet, windows: Windows, sources: Sequence[ScoreSource],
                solved: PassDepth, settings: FacetSettings, *, prior: PriorScore,
                third: ThirdPass, index: int) -> FacetChange | None:
    """The Facet_Change of the Facet at `index` of a Trusted_Frame, or None when a gate of
    Req 12.5 fails (Req 12.4, 12.5, D13, D24, A5).

    `windows` are the Prior_Frame's Score_Windows, `solved` its Solved_Depth before any
    Facet_Change at both sizes, `prior` its side of the passes and `third` its Third_Pass, whose
    scored image and Band_Weights the Facet's tables score.
    """
    prior_height, prior_width = np.shape(prior.covered)
    score_height, score_width = np.shape(prior.high_pass)
    own = facet_windows(facet, windows, (score_width, score_height), (prior_width, prior_height))
    mask = facet.mask((prior_height, prior_width))
    rows, columns = facet.pixels[:, 0], facet.pixels[:, 1]
    solved_on_facet = np.asarray(solved.prior_depth, dtype=np.float64)[rows, columns]
    if own.count == 0 or not np.all(np.isfinite(solved_on_facet) & (solved_on_facet > 0)):
        return None
    m = median_inverse_depth(solved.prior_depth, mask)
    scales = table_scales(settings.facet_min_scale, settings.facet_max_scale,
                          settings.facet_scale_step)
    table, half_a, half_b = score_table(
        prior.frame, own, sources, solved.score_depth, scales, third.prior_image,
        weights=third.scored_weights, window_size=settings.window_size,
        flat_variance=settings.flat_variance,
        occlusion_tolerance=settings.facet_occlusion_tolerance, halves=True)
    change, _ = change_fit(table, scales, own, solved.score_depth, m, settings)
    change_a, _ = change_fit(half_a, scales, own, solved.score_depth, m, settings)
    change_b, _ = change_fit(half_b, scales, own, solved.score_depth, m, settings)
    difference = halves_difference(replace(prior, covered=mask), replace(solved, m=m),
                                   change_a, change_b)
    if not difference <= settings.facet_halves_tolerance:
        return None
    changed = _changed_depth(solved_on_facet, change, m, prior.prior_x[rows, columns],
                             prior.prior_y[rows, columns])
    size = float(np.median(np.abs(changed / solved_on_facet - 1.0)))
    if not size > settings.facet_min_change_ratio * difference:
        return None
    answer = _changed_depth(solved.score_depth, change, m, prior.frame.x, prior.frame.y)
    tile_low, tile_high, _, _ = tile_scores(
        prior.frame, own, sources, answer, third.prior_image, weights=third.scored_weights,
        z_scale=settings.z_scale, z_error_factor=settings.z_error_factor,
        tile_min_windows=settings.tile_min_windows, window_size=settings.window_size,
        flat_variance=settings.flat_variance, occlusion_tolerance=settings.occlusion_tolerance)
    if not (tile_low >= settings.facet_min_z and tile_high >= settings.facet_min_z):
        return None
    return FacetChange(facet=int(index), change=change, m=float(m),
                       halves_difference=float(difference), tile_low=float(tile_low),
                       tile_high=float(tile_high))


def apply_facet_change(depth: np.ndarray, normal: np.ndarray, facet: Facet,
                       change: FacetChange, x: np.ndarray,
                       y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The (h, w) depth and (h, w, 3) normal with the Facet_Change on the Facet's pixels (Req 12.6).

    The Facet's pixels get the change's depth and Changed_Normal, from their depth and normal in
    `depth` and `normal`, with `x` and `y` the Prior_Size Normalised_Coordinates, NaN where the
    change gives none. Every other pixel is as it was. New arrays come back, in the input's dtypes.
    """
    rows, columns = facet.pixels[:, 0], facet.pixels[:, 1]
    new_depth = np.array(depth, copy=True)
    new_normal = np.array(normal, copy=True)
    old = np.asarray(depth, dtype=np.float64)[rows, columns]
    pixel_x = np.asarray(x, dtype=np.float64)[rows, columns]
    pixel_y = np.asarray(y, dtype=np.float64)[rows, columns]
    new_depth[rows, columns] = _changed_depth(old, change.change, change.m, pixel_x, pixel_y)
    new_normal[rows, columns] = changed_normal(np.asarray(normal, dtype=np.float64)[rows, columns],
                                               old, pixel_x, pixel_y, change.change, change.m)
    return new_depth, new_normal
