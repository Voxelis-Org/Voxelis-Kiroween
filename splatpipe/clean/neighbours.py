"""The neighbour search: the Gaussians' centres in float64, and each one's nearest others.

`positions` gives the centres of chosen Gaussians as one float64 array. The Faint_Filter puts
that array in Cubes, and the Isolated_Filter builds its cKDTree on it. `nearest_others` asks the
tree for each point's nearest points and drops the point itself from its own row, so a Gaussian
is never its own neighbour, even where centres coincide (Req 2.16).

The distance between two Gaussians is their Centre_Distance, worked out in float64 from the
stored centres.

This module imports only numpy and scipy.spatial, and nothing from splatpipe.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def positions(data: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """The centres of the Gaussians `rows` of `data`, as a new (len(rows), 3) float64 array.

    The array is in C order, and its columns are filled in turn from x, y and z at `rows`. A
    float32 value converts to float64 exactly, so every Centre_Distance is worked out in float64
    from the stored centres (Req 2.16). A cKDTree keeps C-ordered float64 points as they are, so
    a tree built on this array makes no copy of it.
    """
    points = np.empty((len(rows), 3), dtype=np.float64)
    for column, name in enumerate(("x", "y", "z")):
        points[:, column] = data[name][rows]
    return points


def nearest_others(tree: cKDTree, rows: np.ndarray, k: int,
                   workers: int) -> tuple[np.ndarray, np.ndarray]:
    """The Centre_Distances and the tree indices of the `k` nearest others of the points `rows`.

    `rows` are indices of the tree's points. Both arrays are (len(rows), k), nearest first, and
    the caller makes sure the tree holds more than `k` points. The tree is asked for each
    point's `k` + 1 nearest points, with `workers` worker threads, and the point is dropped from
    its own row (Req 2.16):

    - It's usually first, at distance 0, but another point on the same centre can come ahead of
      it, so it's found by its index wherever it is in the row.
    - With more than `k` + 1 centres on one spot, it can be missing from its row. Every entry of
      such a row is at distance 0, so the row's last column goes instead, which still leaves
      `k` nearest others.

    Among other points at exactly the same distance, the tree's own order decides which are the
    `k` nearest (A10). The raw answers are freed one at a time as they're trimmed, so the most
    this holds at once is both raw arrays, the mask and the trimmed distances.
    """
    n = len(rows)
    every = n == tree.n and np.array_equal(rows, np.arange(n))
    points = tree.data if every else tree.data[rows]
    # The ranks go as a list, because an int k of 1 would make scipy drop the second axis.
    dist, idx = tree.query(points, k=list(range(1, k + 2)), workers=workers)
    is_self = idx == rows[:, None]
    is_self[~is_self.any(axis=1), -1] = True  # a row that doesn't hold its point drops its last
    others = np.logical_not(is_self, out=is_self)  # the same array, now True at the k others
    near_dist = dist[others].reshape(n, k)
    del dist
    near_idx = idx[others].reshape(n, k)
    del idx
    return near_dist, near_idx
