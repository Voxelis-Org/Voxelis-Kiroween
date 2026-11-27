"""The Walls_Step's two fits of the four numbers (a, b, c, d) of a Bas_Relief_Change, as pure
functions (walls spec, Req 10.3, 15.2, 15.3, A5, A12, C10, C14).

- change_fit (Req 10.3, D11, A5, C10): the Change_Fit, scipy's Nelder-Mead on −Guess_Score of one
  Score_Table from `fit_starts` starts, with `fit_max_iterations`, `fit_x_tolerance` and
  `fit_f_tolerance`. The starts are a fixed pattern, because `[walls]` holds no seed and Req 5.9
  wants the same bits whatever the workers: the Identity_Change, then start j, for j from 1,
  moves number (j − 1) mod 4 of (a, b, c, d) by +`fit_start_spread` in the first round of four,
  by −`fit_start_spread` in the next, and so on, the sign turning each round. Each start's
  simplex is the start and the start moved by `fit_start_spread` along each of the four numbers.
  A rejected guess scores −∞, worse than every other. It gives back the best guess it evaluated
  over every start, the first one evaluated on a tie, with its Guess_Score, so the score is at
  least the Identity_Change's, the first guess it evaluates.
- tukey_change (Req 15.2, 15.3, A12, C14): the four numbers enter the inverse depth linearly,
  inv′ = a·inv0 + m·(b·x + c·y + d), so the fit is iteratively reweighted linear least squares
  on the columns (inv0, m·x, m·y, m) against the target inverse depths, with Tukey's weights at
  `tukey_constant` median absolute deviations (unscaled) of the residuals, for `tukey_iterations`
  rounds, and no `fit_bound`, because it fits geometry rather than searches a table. A deviation
  of 0, or one that is only float64 rounding of the targets, leaves the weights as they were.

Every sum over many pixels is elementwise numpy or np.einsum with its default optimize=False,
which calls no BLAS, and LAPACK only sees the 4 × 4 normal equations, so the bits don't depend on
the thread count (C14). This module imports only the standard library, numpy, scipy.optimize and
splatpipe.walls.scoring.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Protocol

import numpy as np
from scipy.optimize import minimize

from splatpipe.walls.scoring import Windows, guess_score

FourNumbers = tuple[float, float, float, float]

IDENTITY_CHANGE: FourNumbers = (1.0, 0.0, 0.0, 0.0)
# The four numbers (a, b, c, d) of a Bas_Relief_Change.
_NUMBERS = 4
# What Nelder-Mead minimises for a rejected guess in place of −(−∞) = +∞. scipy copes with +∞ but
# works out ∞ − ∞ in its convergence test when two of the simplex's guesses are rejected, which
# warns and gives NaN. This finite value is above −F for every real Guess_Score, a sum of window
# NCCs, so a rejected guess still ranks worse than every other, and change_fit keeps the true −∞.
_REJECTED_VALUE = 1e300
# A median absolute deviation at or below this many float64 roundings of the largest target is
# rounding, not noise, so it counts as 0 and the Tukey weights stay as they were. Without it, once
# a fit is exact the deviation is a few roundings and Tukey's weights drop pixels at random, which
# can leave too few to pin the change down.
_ROUNDING_DEVIATION = 1024 * float(np.finfo(np.float64).eps)


class FitSettings(Protocol):
    """The `[walls]` settings change_fit reads, as WallsSettings holds them."""
    fit_starts: int
    fit_bound: float
    fit_start_spread: float
    fit_max_iterations: int
    fit_x_tolerance: float
    fit_f_tolerance: float


def fit_starts(count: int, spread: float) -> list[FourNumbers]:
    """The Change_Fit's `count` starts (C10): the Identity_Change, then start j, for j from 1,
    with number (j − 1) mod 4 moved by +`spread` in the first round of four, −`spread` in the
    next, and so on."""
    starts = [IDENTITY_CHANGE]
    for j in range(1, count):
        number = (j - 1) % _NUMBERS
        sign = 1.0 if ((j - 1) // _NUMBERS) % 2 == 0 else -1.0
        start = list(IDENTITY_CHANGE)
        start[number] += sign * spread
        starts.append((start[0], start[1], start[2], start[3]))
    return starts


def initial_simplex(start: Sequence[float], spread: float) -> np.ndarray:
    """A start's Nelder-Mead simplex, (5, 4) float64: the start, then the start moved by `spread`
    along each of the four numbers in turn."""
    simplex = np.tile(np.asarray(start, dtype=np.float64), (_NUMBERS + 1, 1))
    for number in range(_NUMBERS):
        simplex[number + 1, number] += spread
    return simplex


def change_fit(table: np.ndarray, scales: np.ndarray, windows: Windows, z: np.ndarray, m: float,
               settings: FitSettings) -> tuple[FourNumbers, float]:
    """The Change_Fit on one Score_Table (Req 10.3, D11, A5, C10): the best guess of (a, b, c, d)
    over every guess Nelder-Mead evaluated from every start, and its Guess_Score.

    `table`, `scales`, `windows` and `z` are as scoring.guess_score takes them, and `m` is the
    Median_Inverse_Depth of the depth map the change applies to (the Facet's for a Facet's fit).
    A rejected guess scores −∞. The Identity_Change is the first guess evaluated, and a later
    guess replaces the best only when its score is strictly higher, so a tie goes to the guess
    evaluated first and the answer's score is at least the Identity_Change's. Nelder-Mead's own
    final point is never used in place of a better guess it evaluated on the way.
    """
    best: list = [IDENTITY_CHANGE, guess_score(table, scales, windows, z, IDENTITY_CHANGE, m,
                                               settings.fit_bound)]

    def negative_score(numbers: np.ndarray) -> float:
        guess = (float(numbers[0]), float(numbers[1]), float(numbers[2]), float(numbers[3]))
        score = guess_score(table, scales, windows, z, guess, m, settings.fit_bound)
        if score > best[1]:
            best[0], best[1] = guess, score
        return -score if math.isfinite(score) else _REJECTED_VALUE

    for start in fit_starts(settings.fit_starts, settings.fit_start_spread):
        minimize(negative_score, np.asarray(start, dtype=np.float64), method="Nelder-Mead",
                 options={"maxiter": settings.fit_max_iterations,
                          "xatol": settings.fit_x_tolerance,
                          "fatol": settings.fit_f_tolerance,
                          "initial_simplex": initial_simplex(start, settings.fit_start_spread)})
    return best[0], float(best[1])


def _tukey_weights(residual: np.ndarray, weights: np.ndarray, constant: float,
                   rounding: float) -> np.ndarray:
    """Tukey's weights (1 − (r / (`constant`·σ))²)² where |r| < `constant`·σ, else 0, with σ the
    unscaled median absolute deviation of the residuals `residual`. With σ at most `rounding`,
    which is 0 as far as float64 can tell, or when no row would keep a weight above 0, the weights
    stay as they were."""
    deviation = float(np.median(np.abs(residual - np.median(residual))))
    if not deviation > rounding:
        return weights
    ratio = residual / (constant * deviation)
    new = np.where(np.abs(ratio) < 1.0, (1.0 - ratio * ratio) ** 2, 0.0)
    if not np.any(new > 0):
        return weights
    return new


def _weighted_solve(columns: np.ndarray, target: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """The weighted least-squares (a, b, c, d): the 4 × 4 normal equations Σ w·Xᵀ·X β = Σ w·Xᵀ·t,
    built with np.einsum (no BLAS) and solved with np.linalg.lstsq, which gives the
    smallest-norm answer when they're rank-deficient (C14)."""
    normal = np.einsum("ni,n,nj->ij", columns, weights, columns)
    right = np.einsum("ni,n,n->i", columns, weights, target)
    solution, _, _, _ = np.linalg.lstsq(normal, right, rcond=None)
    return solution


def tukey_change(inverse0: np.ndarray, target: np.ndarray, m: float, x: np.ndarray,
                 y: np.ndarray, constant: float, iterations: int) -> FourNumbers:
    """The Bas_Relief_Change (a, b, c, d) that takes the inverse depths `inverse0` closest to the
    `target` inverse depths, a·inv0 + m·(b·x + c·y + d) ≈ target, fitted robustly (Req 15.2,
    15.3, A12).

    `x` and `y` are the pixels' Normalised_Coordinates, every array the same shape. Only the
    pixels where all five are finite take part. The first fit weights every pixel 1; each of the
    `iterations` rounds then reweights the pixels with _tukey_weights at `constant` median absolute
    deviations of the last fit's residuals and fits again, keeping the weights as they were when
    the deviation is at most `_ROUNDING_DEVIATION` × the largest |target|. No `fit_bound` (A12).
    On any plane the inverse depth inv0 is itself affine in (x, y), so a trades off against
    (b, c, d), the normal equations are rank-deficient and the answer is their smallest-norm
    solution, whose depths still match.
    """
    arrays = [np.asarray(array, dtype=np.float64) for array in (inverse0, target, x, y)]
    if any(array.shape != arrays[0].shape for array in arrays):
        raise ValueError(f"tukey_change: inverse0, target, x and y must have one shape, got "
                         f"{[array.shape for array in arrays]}")
    inverse0, target, x, y = (array.ravel() for array in arrays)
    usable = np.isfinite(inverse0) & np.isfinite(target) & np.isfinite(x) & np.isfinite(y)
    if not np.any(usable) or not math.isfinite(m):
        raise ValueError("tukey_change: no pixel with a finite inverse depth, target and "
                         "coordinates, or m isn't finite")
    inverse0, target, x, y = inverse0[usable], target[usable], x[usable], y[usable]
    m = float(m)
    columns = np.stack([inverse0, m * x, m * y, np.full_like(inverse0, m)], axis=1)
    weights = np.ones_like(inverse0)
    rounding = _ROUNDING_DEVIATION * float(np.max(np.abs(target)))
    solution = _weighted_solve(columns, target, weights)
    for _ in range(iterations):
        a, b, c, d = (float(value) for value in solution)
        residual = target - (a * inverse0 + m * (b * x + c * y + d))
        weights = _tukey_weights(residual, weights, constant, rounding)
        solution = _weighted_solve(columns, target, weights)
    a, b, c, d = (float(value) for value in solution)
    return a, b, c, d
