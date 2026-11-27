"""The Walls_Step's three passes and its Self_Check, as pure functions on arrays (walls spec,
Req 10.1, 10.2, 10.4–10.8, 11.1–11.5, 11.7, 11.8, D11–D14, D24, A1, A4–A6).

What the passes work on:
- PriorScore: the Prior_Frame's side, its ScoreFrame and score images at the Score_Size, its
  Score_Size Start_Depth and Depth_Pixels, its Prior_Size Start_Depth, Covered_Pixels and
  Normalised_Coordinates, and m, the Prior_Size Start_Depth's Median_Inverse_Depth.
- PassDepth: one pass's depth, at the Score_Size, which the next pass scores around, and at the
  Prior_Size, whose Median_Inverse_Depth over the Covered_Pixels is the m of the next pass's
  change. It also holds the change so far, from the Start_Depth, relative to the Start_Depth's m.

The rules:
- first_pass (Req 10.1): a Score_Table on the High_Pass around the Start_Depth with the first_*
  Table_Scales, and the scale s with the highest Σ_p C(p, s), the one nearest 1 in ln(s) on a tie
  (the smaller s when s and 1/s tie). Its change is (1/s, 0, 0, 0) and its depth s × the
  Start_Depth.
- second_pass (Req 10.2): a Score_Table on the High_Pass around the First_Pass's depth with the
  second_* scales, and the Change_Fit on it, of the First_Pass's depth with its own m.
- band_coherences (Req 10.4): each Band's mean Window_Score over the Prior_Frame's Valid_Pairs at
  the Second_Pass's depth. band_variances and frame_band_weights give the Band_Weights from them
  (Req 10.5, A4), each Band's variance taken over the Depth_Pixels the Score_Mask keeps.
- third_pass (Req 10.6, 10.7, A4, A5): a Score_Table and the two Half_Tables on the Band_Mix (the
  High_Pass when every Band_Weight is 0) around the Second_Pass's depth with the third_* scales,
  and the Change_Fit on the Score_Table, of the Second_Pass's depth with its own m.
- The Solved_Change (Req 10.8): the three passes' changes composed with geometry.compose, each
  relative to its own pass's Current_Depth and that depth's m, into one change of the Start_Depth
  relative to the Start_Depth's m. So k stays a number of its own.
- self_check (Req 11.2–11.5, 11.7, 11.8, D12, D13, D14, A6): each Source_Half's Change_Fit on its
  Half_Table, the Halves_Difference over the Prior_Size Covered_Pixels, the two Tile_Scores at the
  answer (the Second_Pass's depth through the Third_Pass's change), and the verdict.
- source_halves (Req 11.1, D12): the Source_Half of each ranked Source_Frame, "A" for the 1st,
  3rd, … and "B" for the 2nd, 4th, …, which the caller gives each ScoreSource.

A Prior_Frame with fewer than `min_sources` Source_Frames fails its Self_Check with no pass run
(Req 8.7): the caller doesn't call these functions for it. self_check still refuses a source count
below `min_sources` that it is handed (Req 11.7).

Every sum over many pixels is elementwise numpy, np.sum, np.median or np.var, and the poses are
applied elementwise by geometry.py, so no BLAS call sees them and the bits don't depend on the
thread count (C14). This module imports only the standard library, numpy and the walls pure
modules.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from splatpipe.walls.fit import FitSettings, FourNumbers, change_fit
from splatpipe.walls.geometry import apply_change, compose, median_inverse_depth, warp_maps
from splatpipe.walls.images import band_mix, band_weights, window_ncc
from splatpipe.walls.scoring import (HALVES, ScoreFrame, ScoreSource, Windows, occluded_windows,
                                     score_table, table_scales, tile_scores, valid_pairs)


class PassSettings(FitSettings, Protocol):
    """The `[walls]` settings the passes and the Self_Check read, as WallsSettings holds them."""
    window_size: int
    flat_variance: float
    occlusion_tolerance: float
    first_min_scale: float
    first_max_scale: float
    first_scale_step: float
    second_min_scale: float
    second_max_scale: float
    second_scale_step: float
    third_min_scale: float
    third_max_scale: float
    third_scale_step: float
    band_max_coherence: float
    min_sources: int
    halves_tolerance: float
    tile_min_windows: int
    z_scale: float
    min_z: float
    z_error_factor: float


@dataclass(frozen=True)
class PriorScore:
    """The Prior_Frame's side of the passes."""
    frame: ScoreFrame               # its Image_Pose, the Score_Size K and Normalised_Coordinates
    high_pass: np.ndarray           # (h, w) float64 at the Score_Size
    bands: np.ndarray               # (B, h, w) float64 at the Score_Size
    score_mask: np.ndarray          # (h, w) bool, true where kept
    depth_pixel: np.ndarray         # (h, w) bool, its Depth_Pixels
    start_depth: np.ndarray         # (h, w) float64, the Score_Size Start_Depth, NaN elsewhere
    prior_start_depth: np.ndarray   # (ph, pw) float64, the Prior_Size Start_Depth, k × Prior_Depth
    covered: np.ndarray             # (ph, pw) bool, its Covered_Pixels
    prior_x: np.ndarray             # (ph, pw) float64, Normalised_Coordinates at the Prior_Size
    prior_y: np.ndarray
    m: float                        # the Prior_Size Start_Depth's Median_Inverse_Depth


@dataclass(frozen=True)
class PassDepth:
    """One pass's depth and the change that made it from the Start_Depth."""
    score_depth: np.ndarray   # (h, w) at the Score_Size, NaN where there's none
    prior_depth: np.ndarray   # (ph, pw) at the Prior_Size, NaN where there's none
    m: float                  # its Median_Inverse_Depth over the Covered_Pixels
    change: FourNumbers       # from the Start_Depth so far, relative to the Start_Depth's m


@dataclass(frozen=True)
class ThirdPass:
    """The Third_Pass: its change, its tables and what the Self_Check needs from it."""
    change: FourNumbers            # its own, of the Second_Pass's depth with that depth's m
    score: float                   # its Guess_Score on the Score_Table
    current: PassDepth             # the Second_Pass's depth, its Current_Depth
    answer: PassDepth              # the Second_Pass's depth through its change
    band_weights: np.ndarray       # (B,) the Prior_Frame's Band_Weights
    scored_weights: np.ndarray | None  # the weights it scored with, None for the High_Pass
    prior_image: np.ndarray        # the Prior_Frame's image it scored, Band_Mix or High_Pass
    scales: np.ndarray             # (S,) its Table_Scales
    table: np.ndarray              # (S, P) its Score_Table
    half_a: np.ndarray             # (S, P) Half_A's Half_Table
    half_b: np.ndarray             # (S, P) Half_B's Half_Table

    @property
    def solved_change(self) -> FourNumbers:
        """The Solved_Change, the three passes' changes composed, of the Start_Depth (Req 10.8)."""
        return self.answer.change


@dataclass(frozen=True)
class SolveResult:
    """One Prior_Frame's numbers from its passes and its Self_Check (Req 10.9, 11.9)."""
    k: float
    scale_points: int
    source_count: int
    seconds_cut: float
    first_scale: float
    change: FourNumbers            # the Solved_Change, relative to the Start_Depth
    m: float                       # the Start_Depth's Median_Inverse_Depth
    band_weights: np.ndarray       # (B,)
    halves_difference: float
    tile_low: float                # Z against ÷ z_scale, NaN for no Z
    tile_high: float               # Z against × z_scale, NaN for no Z
    tile_low_count: int
    tile_high_count: int
    passed: bool


def source_halves(count: int) -> tuple[str, ...]:
    """The Source_Half of each of `count` Source_Frames in rank order: "A" for the 1st, 3rd, …
    and "B" for the 2nd, 4th, … (Req 11.1, D12)."""
    return tuple(HALVES[index % 2] for index in range(count))


def _through(depth: np.ndarray, change: FourNumbers, m: float, x: np.ndarray,
             y: np.ndarray) -> np.ndarray:
    """`depth` through the Bas_Relief_Change `change` with the m `m`, NaN where it gives no
    depth."""
    with np.errstate(divide="ignore", invalid="ignore"):
        inverse = apply_change(1.0 / np.asarray(depth, dtype=np.float64), change, m, x, y)
        return 1.0 / inverse


def start(prior: PriorScore) -> PassDepth:
    """The Start_Depth as a PassDepth, with the Identity_Change."""
    return PassDepth(score_depth=np.asarray(prior.start_depth, dtype=np.float64),
                     prior_depth=np.asarray(prior.prior_start_depth, dtype=np.float64),
                     m=float(prior.m), change=(1.0, 0.0, 0.0, 0.0))


def next_depth(prior: PriorScore, current: PassDepth, change: FourNumbers) -> PassDepth:
    """`current` through a pass's `change`, which is relative to `current` and its m, at both
    sizes, with its own m and the change from the Start_Depth composed with compose (Req 10.8)."""
    frame = prior.frame
    score_depth = _through(current.score_depth, change, current.m, frame.x, frame.y)
    prior_depth = _through(current.prior_depth, change, current.m, prior.prior_x, prior.prior_y)
    m = median_inverse_depth(prior_depth, np.asarray(prior.covered, dtype=bool)
                             & np.isfinite(prior_depth))
    return PassDepth(score_depth=score_depth, prior_depth=prior_depth, m=m,
                     change=compose(current.change, change, prior.m, current.m))


def _table(prior: PriorScore, windows: Windows, sources: Sequence[ScoreSource],
           current: np.ndarray, scales: np.ndarray, prior_image: np.ndarray,
           weights: np.ndarray | None, settings: PassSettings, halves: bool = False):
    """score_table with the settings' window size, flat variance and occlusion tolerance."""
    return score_table(prior.frame, windows, sources, current, scales, prior_image,
                       weights=weights, window_size=settings.window_size,
                       flat_variance=settings.flat_variance,
                       occlusion_tolerance=settings.occlusion_tolerance, halves=halves)


def best_scale(table: np.ndarray, scales: np.ndarray) -> float:
    """The Table_Scale s with the highest Σ_p C(p, s), the one nearest 1 in ln(s) on a tie, and
    the smaller of s and 1/s when both are on the table and tie (Req 10.1)."""
    scales = np.asarray(scales, dtype=np.float64)
    sums = np.sum(np.asarray(table, dtype=np.float64), axis=1)
    tied = np.flatnonzero(sums == np.max(sums))
    return float(min(scales[tied], key=lambda s: (abs(math.log(s)), s)))


def first_pass(prior: PriorScore, windows: Windows, sources: Sequence[ScoreSource],
               settings: PassSettings) -> tuple[float, PassDepth]:
    """The First_Pass (Req 10.1): its scale s and its depth, s × the Start_Depth, whose change
    from the Start_Depth is (1/s, 0, 0, 0)."""
    begin = start(prior)
    scales = table_scales(settings.first_min_scale, settings.first_max_scale,
                          settings.first_scale_step)
    table = _table(prior, windows, sources, begin.score_depth, scales, prior.high_pass, None,
                   settings)
    s = best_scale(table, scales)
    prior_depth = s * begin.prior_depth
    m = median_inverse_depth(prior_depth, np.asarray(prior.covered, dtype=bool)
                             & np.isfinite(prior_depth))
    return s, PassDepth(score_depth=s * begin.score_depth, prior_depth=prior_depth, m=m,
                        change=(1.0 / s, 0.0, 0.0, 0.0))


def second_pass(prior: PriorScore, windows: Windows, sources: Sequence[ScoreSource],
                first: PassDepth, settings: PassSettings) -> tuple[FourNumbers, PassDepth]:
    """The Second_Pass (Req 10.2): the Change_Fit on a High_Pass Score_Table around the
    First_Pass's depth, as its own change of that depth with that depth's m, and its depth."""
    scales = table_scales(settings.second_min_scale, settings.second_max_scale,
                          settings.second_scale_step)
    table = _table(prior, windows, sources, first.score_depth, scales, prior.high_pass, None,
                   settings)
    change, _ = change_fit(table, scales, windows, first.score_depth, first.m, settings)
    return change, next_depth(prior, first, change)


def band_coherences(prior: PriorScore, windows: Windows, sources: Sequence[ScoreSource],
                    second: PassDepth, settings: PassSettings) -> np.ndarray:
    """Each Band's Coherence ρ, (B,) float64: its mean Window_Score over the Prior_Frame's
    Valid_Pairs at the Second_Pass's depth (Req 10.4), the occlusion test made there (A1). The
    Valid_Pairs are the same for every Band, because the flat test reads the High_Pass (A2). With
    no Valid_Pair every ρ is 0, so every Band_Weight is 0."""
    frame = prior.frame
    band_count = int(np.shape(prior.bands)[0])
    sums = np.zeros(band_count)
    pairs = 0
    for source in sources:
        hidden = occluded_windows(frame, windows, source, second.score_depth,
                                  settings.occlusion_tolerance)
        map_x, map_y, _, inside = warp_maps(second.score_depth, 1.0, frame.world_to_camera,
                                            source.world_to_camera, frame.k, frame.x, frame.y,
                                            source_size=source.score_size)
        valid = valid_pairs(windows, map_x, map_y, inside, source, hidden,
                            window_size=settings.window_size,
                            flat_variance=settings.flat_variance)
        pairs += int(np.count_nonzero(valid))
        for band in range(band_count):
            scores = window_ncc(prior.bands[band], source.bands[band], map_x, map_y, inside,
                                windows.centres, settings.window_size, 0.0)
            sums[band] += float(np.sum(scores[valid]))
    if pairs == 0:
        return np.zeros(band_count)
    return sums / pairs


def band_variances(prior: PriorScore) -> np.ndarray:
    """Each Band's variance over the Prior_Frame's Depth_Pixels the Score_Mask keeps, (B,)
    float64, NaN with no such pixel (A4)."""
    kept = np.asarray(prior.depth_pixel, dtype=bool) & np.asarray(prior.score_mask, dtype=bool)
    if not kept.any():
        return np.full(int(np.shape(prior.bands)[0]), np.nan)
    return np.array([float(np.var(np.asarray(band, dtype=np.float64)[kept]))
                     for band in prior.bands])


def frame_band_weights(prior: PriorScore, coherences: np.ndarray,
                       settings: PassSettings) -> np.ndarray:
    """The Prior_Frame's Band_Weights from its Coherences (Req 10.5, A4), with
    images.band_weights, its Band variances and `band_max_coherence`."""
    return band_weights(coherences, band_variances(prior), settings.band_max_coherence)


def third_pass(prior: PriorScore, windows: Windows, sources: Sequence[ScoreSource],
               weights: np.ndarray, second: PassDepth, settings: PassSettings) -> ThirdPass:
    """The Third_Pass (Req 10.6, 10.7, A4, A5): the Score_Table and the two Half_Tables around the
    Second_Pass's depth, on the Band_Mix of the Prior_Frame and of each Source_Frame with the
    Prior_Frame's Band_Weights `weights`, or on the High_Pass when every weight is 0, and the
    Change_Fit on the Score_Table, of the Second_Pass's depth with its m."""
    weights = np.asarray(weights, dtype=np.float64)
    scored = weights if np.any(weights != 0) else None
    prior_image = (np.asarray(prior.high_pass, dtype=np.float64) if scored is None
                   else band_mix(prior.bands, scored))
    scales = table_scales(settings.third_min_scale, settings.third_max_scale,
                          settings.third_scale_step)
    table, half_a, half_b = _table(prior, windows, sources, second.score_depth, scales,
                                   prior_image, scored, settings, halves=True)
    change, score = change_fit(table, scales, windows, second.score_depth, second.m, settings)
    return ThirdPass(change=change, score=score, current=second,
                     answer=next_depth(prior, second, change), band_weights=weights,
                     scored_weights=scored, prior_image=prior_image, scales=scales, table=table,
                     half_a=half_a, half_b=half_b)


def halves_difference(prior: PriorScore, current: PassDepth, change_a: FourNumbers,
                      change_b: FourNumbers) -> float:
    """The Halves_Difference (Req 11.3, D12): the median over the Prior_Frame's Covered_Pixels, at
    the Prior_Size, of |z_A / z_B − 1|, with z_A and z_B the Second_Pass's depth `current` through
    each Source_Half's change with its m. A Covered_Pixel where either half gives no depth counts
    as an infinite difference."""
    covered = np.asarray(prior.covered, dtype=bool)
    z_a = _through(current.prior_depth, change_a, current.m, prior.prior_x, prior.prior_y)
    z_b = _through(current.prior_depth, change_b, current.m, prior.prior_x, prior.prior_y)
    with np.errstate(divide="ignore", invalid="ignore"):
        difference = np.abs(z_a[covered] / z_b[covered] - 1.0)
    return float(np.median(np.where(np.isfinite(difference), difference, np.inf)))


def self_check(prior: PriorScore, third: ThirdPass, windows: Windows,
               sources: Sequence[ScoreSource], settings: PassSettings, *, first_scale: float,
               k: float, scale_points: int, source_count: int,
               seconds_cut: float) -> SolveResult:
    """The Self_Check (Req 11.2–11.5, 11.7, 11.8, D12, D13, D14, A6) and the frame's SolveResult.

    Each Source_Half's Change_Fit on its Half_Table of `third`, of the Second_Pass's depth; the
    Halves_Difference of the two; and the two Tile_Scores at the answer, the Second_Pass's depth
    through the Third_Pass's change, on the image the Third_Pass scored, with the occlusion test
    made at the answer (D24). It passes only when `source_count` is at least `min_sources`, the
    Halves_Difference is below `halves_tolerance` and both Tile_Scores are above `min_z`; a NaN
    Tile_Score, no Z, is never above it. `first_scale`, `k`, `scale_points`, `source_count` and
    `seconds_cut` are the caller's, kept in the result as they are.
    """
    current = third.current
    change_a, _ = change_fit(third.half_a, third.scales, windows, current.score_depth, current.m,
                             settings)
    change_b, _ = change_fit(third.half_b, third.scales, windows, current.score_depth, current.m,
                             settings)
    difference = halves_difference(prior, current, change_a, change_b)
    tile_low, tile_high, low_count, high_count = tile_scores(
        prior.frame, windows, sources, third.answer.score_depth, third.prior_image,
        weights=third.scored_weights, z_scale=settings.z_scale,
        z_error_factor=settings.z_error_factor, tile_min_windows=settings.tile_min_windows,
        window_size=settings.window_size, flat_variance=settings.flat_variance,
        occlusion_tolerance=settings.occlusion_tolerance)
    passed = (source_count >= settings.min_sources and difference < settings.halves_tolerance
              and tile_low > settings.min_z and tile_high > settings.min_z)
    return SolveResult(k=float(k), scale_points=int(scale_points), source_count=int(source_count),
                       seconds_cut=float(seconds_cut), first_scale=float(first_scale),
                       change=third.solved_change, m=float(prior.m),
                       band_weights=third.band_weights, halves_difference=difference,
                       tile_low=float(tile_low), tile_high=float(tile_high),
                       tile_low_count=int(low_count), tile_high_count=int(high_count),
                       passed=bool(passed))
