"""The Isolated_Filter: it drops the floaters, the Gaussians far from the spacing around them.

A Gaussian's Nearest_Neighbours are the `neighbours` other Gaussians nearest to it, and its
Neighbour_Distance is its mean Centre_Distance to them. Its Neighbour_Median is the median of its
Nearest_Neighbours' own Neighbour_Distances. The filter drops a Gaussian whose Neighbour_Distance
is more than isolation_ratio times its Neighbour_Median (Req 2.12). The test is local, because
each Gaussian is measured against the spacing around it, so a sparse but even region, like a far
wall or the sky, survives where a global distance cutoff would eat it (A5).

Every Neighbour_Distance and Neighbour_Median comes from the whole set the filter was given,
before it drops any Gaussian (Req 2.13). This is the only filter that searches for neighbours,
with the step's one cKDTree, and a Gaussian is never its own neighbour (Req 2.16). Among other
Gaussians at exactly the same distance, the tree's own order decides which are the nearest
(A10). That can change a Neighbour_Median but never a Neighbour_Distance.

From splatpipe, this module imports only splatpipe.clean.neighbours and splatpipe.clean.settings.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from splatpipe.clean.neighbours import nearest_others
from splatpipe.clean.settings import CleanSettings


def isolated_keep(points: np.ndarray, s: CleanSettings, workers: int) -> np.ndarray:
    """Which Gaussians the Isolated_Filter keeps, as a bool array with one per row of `points`.

    `points` is the set the filter was given, as from positions: an (n, 3) C-ordered float64
    array of finite centres, which the tree uses as it is, with no copy. It isn't changed.
    `workers` is the number of worker threads for the tree's query. Everything comes from the
    whole set, before any Gaussian is dropped (Req 2.13):

    1. A set of `neighbours` Gaussians or fewer is kept whole, with no tree (Req 2.14).
    2. Otherwise the step's one cKDTree is built on `points`, and nearest_others gives each
       Gaussian's `neighbours` nearest others, never itself, with their Centre_Distances
       (Req 2.16).
    3. Each Gaussian's Neighbour_Distance is the mean of those Centre_Distances.
    4. Each Gaussian's Neighbour_Median is the median of its Nearest_Neighbours'
       Neighbour_Distances. With an even `neighbours`, like 8, that's the mean of the middle two.
    5. A Gaussian is kept when its Neighbour_Median is 0 (Req 2.15), or when its
       Neighbour_Distance divided by its Neighbour_Median is at most isolation_ratio (Req 2.12).
       The division is taken only where the median is above 0, so nothing is divided by 0. It
       can't overflow either, whatever isolation_ratio is. Two different float32 values are at
       least 2**-149 apart, so a median above 0 is at least 2**-149 / (2 x neighbours), and a
       distance between finite float32 centres is at most about 1.2e39. So the ratio stays
       below about 1.7e84 x neighbours, and `neighbours` is below the set's size here.

    Each array is freed once it's done with: the tree once it has answered, the Centre_Distances
    once the means are made, the neighbours' indices once their Neighbour_Distances are gathered,
    and the gather once its median, which partitions it in place, is taken.
    """
    n = len(points)
    if n <= s.neighbours:
        return np.ones(n, dtype=bool)
    tree = cKDTree(points)
    dist, idx = nearest_others(tree, np.arange(n), s.neighbours, workers)
    del tree
    nd = dist.mean(axis=1)  # each Gaussian's Neighbour_Distance
    del dist
    around = nd[idx]  # each row, the Neighbour_Distances of one Gaussian's Nearest_Neighbours
    del idx
    med = np.median(around, axis=1, overwrite_input=True)  # each Gaussian's Neighbour_Median
    del around
    # nd / med, in nd's own array and only where med is above 0. Where med is 0, nd keeps its
    # Neighbour_Distance, and the Gaussian is kept whatever that is.
    ratio = np.divide(nd, med, out=nd, where=med > 0)
    return (med == 0) | (ratio <= s.isolation_ratio)
