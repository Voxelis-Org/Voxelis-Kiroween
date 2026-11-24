"""The Start_Splats, the Exposure_Bake and the drop of non-finite splats: pure numpy functions.

They import no torch, so the Test_Suite checks every rule on a laptop (Req 3.13, 6.6, 8.9). A
Splats holds N splats as six arrays, in the Trainer's conventions: means (N, 3), log scales
(N, 3), quaternions (N, 4) w first, opacity logits (N,), sh0 (N, 3), the band-0 coefficient of
each colour channel, and sh_rest (N, 15, 3), coefficient first and channel last, as gsplat's
shN. The Trainer renders exp() of the log scales and sigmoid() of the logits, and a splat's
colour is 0.5 + SH_C0 · sh0 plus the higher bands, before the clamp at 0.

- choose_points (Req 3.11): every Sparse_Point when they fit the Splat_Budget, else `budget` of
  them, drawn by a generator seeded with `seed` and kept in the file's order.
- need_points (Req 1.18): refuses fewer than start_scale_neighbours + 1 points, so each
  Start_Splat has that many other points to take its size from.
- log_start_distances and start_log_scales (Req 3.12, 3.14, 3.15, 3.16): the log of each
  Start_Splat's scale, start_scale_factor times the root mean square of its distances to its
  nearest other points, worked out as a sum of logs so that it stays finite for any distance
  above 0. A splat whose nearest points all sit on it takes the smallest Start_Distance above 0
  (A6).
- start_splats (Req 3.12): the Start_Splats, as float32.
- mean_exposure and bake_exposure (Req 6.4, 6.5): the mean of the Exposure_Fixes, baked into the
  colours in float64, so a viewer that shows the splat with no Exposure_Fix shows the average
  frame's exposure.
- drop_non_finite (Req 8.9, D6): leaves out every splat with a NaN or an infinity in any of its
  59 values, judging each value as the float32 that the PLY_Writer writes for it.

From splatpipe, this module imports only splatpipe.errors and splatpipe.train.settings.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields, replace

import numpy as np

from splatpipe.errors import ExpectedFailure
from splatpipe.train.settings import TrainSettings

SH_C0 = 0.28209479177387814  # Fixed_Constant: 1 / (2√π), the band-0 basis value
SH_REST = 15  # Fixed_Constant: coefficients per channel above band 0 at SH_Degree 3

_ALL_ON_TOP = ("The camera solve's points all sit on top of each other, so the step can't size "
               "the splats. Run the camera solve step on this job again.")  # Req 3.15


@dataclass(frozen=True)
class Splats:
    """N splats: the Start_Splats, or the Trainer's trained splats.

    float32 from start_splats and the Trainer. After bake_exposure, sh0 and sh_rest are float64.
    """

    means: np.ndarray  # (N, 3)
    log_scales: np.ndarray  # (N, 3)
    quats: np.ndarray  # (N, 4), w first, as the Trainer holds them
    opacity_logits: np.ndarray  # (N,)
    sh0: np.ndarray  # (N, 3): band 0 per channel, the f_dc fields
    sh_rest: np.ndarray  # (N, 15, 3): coefficient first, channel last, as gsplat's shN

    def __len__(self) -> int:
        """N, the number of splats."""
        return self.means.shape[0]


def choose_points(points: np.ndarray, colours: np.ndarray, budget: int,
                  seed: int) -> tuple[np.ndarray, np.ndarray]:
    """The Sparse_Points and their colours that the Start_Splats start from (Req 3.11).

    Every point when there are at most `budget` of them. Otherwise `budget` points drawn without
    replacement by np.random.default_rng(seed), then sorted, so the subset keeps the file's
    order. The step warns with both counts when it gets a subset.
    """
    count = len(points)
    if count <= budget:
        return points, colours
    chosen = np.sort(np.random.default_rng(seed).choice(count, budget, replace=False))
    return points[chosen], colours[chosen]


def need_points(count: int, neighbours: int) -> None:
    """Refuse fewer than neighbours + 1 points, giving the count and the fewest (Req 1.18).

    `neighbours` is start_scale_neighbours: each Start_Splat takes its size from that many other
    points, so the step needs one more point than that.
    """
    fewest = neighbours + 1
    if count < fewest:
        raise ExpectedFailure(
            f"The camera solve found too few points to start the splats from: {count}, and the "
            f"train step needs at least {fewest}, since each splat takes its starting size from "
            "the points nearest to it. Film more of the place, or run the camera solve step on "
            "this job again.")


def log_start_distances(distances: np.ndarray) -> np.ndarray:
    """The log of each splat's Start_Distance, the root mean square of its row (Req 3.14).

    `distances` is (N, k): each splat's distances, all 0 or more, to its k nearest other points.
    The root mean square is never formed, because it can round to 0. With m the row's largest
    distance, the row's log is log m + ½ · log(mean((d / m)²)). With m above 0, every d / m is
    from 0 to 1 and one of them is exactly 1, so the mean is from 1/k to 1 and both logs are
    finite, even for m = 5e-324. A row with m = 0, a Start_Distance of 0, takes the smallest log
    of the other rows, the log of the smallest Start_Distance above 0 (A6). When every row has
    m = 0, the step can't size the splats, and it's refused (Req 3.15).
    """
    distances = np.asarray(distances, dtype=np.float64)
    largest = distances.max(axis=1)
    sized = largest > 0
    if not sized.any():
        raise ExpectedFailure(_ALL_ON_TOP)
    m = largest[sized]
    ratios = distances[sized] / m[:, np.newaxis]
    logs = np.empty(len(distances))
    logs[sized] = np.log(m) + 0.5 * np.log(np.mean(ratios * ratios, axis=1))
    logs[~sized] = logs[sized].min()
    return logs


def start_log_scales(distances: np.ndarray, factor: float) -> np.ndarray:
    """Each splat's log scale, log(factor × Start_Distance), in three equal columns (Req 3.12).

    It's log(factor) + log_start_distances(distances), a sum of two finite logs, so a tiny
    Start_Distance can't round factor × d to 0 (Req 3.16). The result is (N, 3), in float64.
    """
    logs = math.log(factor) + log_start_distances(distances)
    return np.repeat(logs[:, np.newaxis], 3, axis=1)


def start_splats(points: np.ndarray, colours: np.ndarray, distances: np.ndarray,
                 settings: TrainSettings) -> Splats:
    """The Start_Splats, as float32 arrays (Req 3.12, D2, A6).

    Each takes its point's position as its mean; an sh0 of (colour / 255 − 0.5) / SH_C0, so it
    shows the point's 8-bit colour; a zero sh_rest; the logit log(p / (1 − p)) of start_opacity;
    the identity rotation (1, 0, 0, 0); and start_log_scales of its row of `distances` with
    start_scale_factor. `points` is (N, 3), `colours` (N, 3) uint8, and `distances` (N, k).
    """
    count = len(points)
    opacity = settings.start_opacity
    quats = np.zeros((count, 4), dtype=np.float32)
    quats[:, 0] = 1
    return Splats(
        means=points.astype(np.float32),
        log_scales=start_log_scales(distances, settings.start_scale_factor).astype(np.float32),
        quats=quats,
        opacity_logits=np.full(count, math.log(opacity / (1 - opacity)), dtype=np.float32),
        sh0=((colours.astype(np.float64) / 255 - 0.5) / SH_C0).astype(np.float32),
        sh_rest=np.zeros((count, SH_REST, 3), dtype=np.float32),
    )


def mean_exposure(fixes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The Mean_Exposure, Ā (3, 3) and b̄ (3,), in float64 (Req 6.4).

    `fixes` is (V, 3, 4), each Training_Image's Exposure_Fix [A | b]. Their element-wise mean is
    split into its first three columns, Ā, and its last, b̄.
    """
    mean = np.mean(fixes, axis=0, dtype=np.float64)
    return mean[:, :3], mean[:, 3]


def bake_exposure(splats: Splats, a: np.ndarray, b: np.ndarray) -> Splats:
    """The splats with the Mean_Exposure Ā, b̄ baked into their colours, in float64 (Req 6.5).

    Each higher-band coefficient's RGB vector v becomes Ā·v, which is sh_rest @ Āᵀ, and sh0
    becomes Ā·sh0 + (Ā·0.5·1 + b̄ − 0.5·1) / SH_C0, with 1 = (1, 1, 1). A colour c, 0.5 + SH_C0 ·
    sh0 plus the higher bands, so becomes Ā·c + b̄ (Req 6.8). Only the colours change: the other
    arrays are kept as they are. The PLY_Writer rounds the result to float32.
    """
    a = np.asarray(a, dtype=np.float64)
    half = np.full(3, 0.5)
    shift = (a @ half + np.asarray(b, dtype=np.float64) - half) / SH_C0
    return replace(splats,
                   sh0=np.asarray(splats.sh0, dtype=np.float64) @ a.T + shift,
                   sh_rest=np.asarray(splats.sh_rest, dtype=np.float64) @ a.T)


def drop_non_finite(splats: Splats) -> tuple[Splats, int]:
    """The splats whose 59 values are all finite, in their order, and the number dropped.

    A value counts as finite when the float32 that the PLY_Writer writes for it is finite
    (Req 8.9, D6). So each of the six arrays is rounded to float32 inside
    np.errstate(over="ignore"), as splat_records rounds it, and a baked float64 value too big
    for float32 becomes an infinity that drops its splat, instead of failing the PLY_Check.
    np.isfinite is reduced over every axis but the first into one `keep` of length N. The splats
    kept are each array's [keep], with the input's own values and dtypes. When nothing is
    dropped, the Splats given come back with no copy, so 0 splats give 0 splats and a count of 0.
    """
    arrays = {field.name: getattr(splats, field.name) for field in fields(splats)}
    keep = np.ones(len(splats), dtype=bool)
    with np.errstate(over="ignore"):
        for array in arrays.values():
            written = np.asarray(array, dtype=np.float32)
            keep &= np.isfinite(written).all(axis=tuple(range(1, written.ndim)))
    dropped = int(np.count_nonzero(~keep))
    if dropped == 0:
        return splats, 0
    return Splats(**{name: array[keep] for name, array in arrays.items()}), dropped
