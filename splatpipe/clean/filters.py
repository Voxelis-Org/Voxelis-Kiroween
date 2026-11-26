"""The Filter_Order: the four filters, each run on the set the one before it kept.

`run_filters` runs the Broken_Filter on every Gaussian of the Splat_Data, then the Unseen_Filter
on what it kept, then the Faint_Filter on what that kept, then the Isolated_Filter. The
Kept_Splats are the Gaussians that pass all four (Req 2.1). The Unseen_Filter comes before the
Faint_Filter, so the fog it drops doesn't stretch the faint filter's trimmed box.

Each set goes on as an increasing array of row numbers into the Splat_Data, never as a copy of
its rows. Indexing such an array with a filter's mask keeps its order, so each filter gives back
a subset of the set it was given, in the original order, with no Gaussian added or repeated.
Nothing here or in the filters writes to the Splat_Data, so every kept Gaussian keeps its values
bit for bit (Req 2.18).

How many Gaussians each filter dropped goes into the Clean_Report through add_count, as an int,
as soon as that filter is done. So a failure later in the step leaves the counts worked out
before it, and no others (Req 2.17, 7.2, 7.3).

From splatpipe, this module imports only splatpipe.clean.settings, splatpipe.clean.neighbours,
splatpipe.clean.broken, splatpipe.clean.unseen, splatpipe.clean.faint and
splatpipe.clean.isolated.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from splatpipe.clean.broken import broken_keep
from splatpipe.clean.faint import faint_keep
from splatpipe.clean.isolated import isolated_keep
from splatpipe.clean.neighbours import positions
from splatpipe.clean.settings import CleanSettings
from splatpipe.clean.unseen import Views, unseen_keep


@dataclass(frozen=True)
class Kept:
    """The Gaussians that passed every filter, and how many each filter dropped."""

    rows: np.ndarray  # indices into the Splat_Data, increasing
    dropped_broken: int
    dropped_faint: int
    dropped_isolated: int
    dropped_unseen: int = 0  # 0 when no Views were given


def run_filters(data: np.ndarray, s: CleanSettings, add_count: Callable[[str, int], None], *,
                views: Views | None = None) -> Kept:
    """The Gaussians of the Splat_Data `data` that pass every filter, in Filter_Order.

    `add_count` is step.add_count. Each filter gets the set the one before it kept, as row
    numbers into `data`, and its counts are added as soon as it's done (Req 2.1, 2.17, 7.2):

    1. The Broken_Filter runs on every Gaussian, and dropped_broken is added.
    2. With `views`, the training frames' cameras, the Unseen_Filter runs on what the
       Broken_Filter kept and drops the Gaussians fewer than min_seen_frames frames see. Then
       dropped_unseen is added. The step always gives `views`; the tests of the other filters
       give none, and then this filter doesn't run.
    3. The Faint_Filter runs on what the filters before it kept, with their centres, their stored
       opacities and the sums of their two largest stored scales, each in float64. Its inputs
       are freed once it has answered, and only the centres of the Gaussians it kept go on.
       Then dropped_faint is added.
    4. The Isolated_Filter runs on what the Faint_Filter kept, with max(1, max_workers) worker
       threads for its tree's query, because scipy refuses 0 workers. A max_workers of 0 means
       inline encoding, which only splat-transform has. Then dropped_isolated and kept_splats
       are added.

    So nothing of the faint stage is still held when the Isolated_Filter builds the step's one
    cKDTree, which is the step's peak in memory. `data` is never written to.
    """
    rows1 = np.flatnonzero(broken_keep(data))
    dropped_broken = int(len(data) - len(rows1))
    add_count("dropped_broken", dropped_broken)
    points1 = positions(data, rows1)
    dropped_unseen = 0
    if views is not None:
        keep = unseen_keep(points1, views, s.min_seen_frames)
        dropped_unseen = int(len(rows1) - np.count_nonzero(keep))
        rows1, points1 = rows1[keep], points1[keep]
        del keep
        add_count("dropped_unseen", dropped_unseen)
    keep = faint_keep(points1, _float64_at(data, "opacity", rows1), _top_two(data, rows1), s)
    rows2, points2 = rows1[keep], points1[keep]
    dropped_faint = int(len(rows1) - len(rows2))
    del rows1, points1, keep
    add_count("dropped_faint", dropped_faint)
    keep = isolated_keep(points2, s, max(1, s.max_workers))
    rows3 = rows2[keep]
    dropped_isolated = int(len(rows2) - len(rows3))
    del points2, rows2, keep
    add_count("dropped_isolated", dropped_isolated)
    add_count("kept_splats", int(len(rows3)))
    return Kept(rows3, dropped_broken, dropped_faint, dropped_isolated, dropped_unseen)


def _float64_at(data: np.ndarray, name: str, rows: np.ndarray) -> np.ndarray:
    """The property `name` of the Gaussians `rows` of `data`, as a new float64 array.

    Gathering at `rows` already makes a new array, so copy=False never hands back a view of
    `data`, and a property that's already float64 isn't copied twice.
    """
    return data[name][rows].astype(np.float64, copy=False)


def _top_two(data: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """The sum of the two largest stored scales of each Gaussian `rows` of `data`, in float64.

    A stored scale is the log of its Real_Scale, and exp only grows, so these are the logs of
    the two largest Real_Scales. The larger of scale_0 and scale_1 is one of the two, and the
    larger of the smaller one and scale_2 is the other. A float32 value converts to float64
    exactly, and the sum of two of them is at most about 6.8e38, so it can't overflow
    (Req 2.11).
    """
    s0, s1, s2 = (_float64_at(data, f"scale_{axis}", rows) for axis in range(3))
    return np.maximum(s0, s1) + np.maximum(np.minimum(s0, s1), s2)
