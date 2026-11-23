"""The Faint_Filter: it judges every faint Gaussian together with the rest of its Cube.

A Faint_Gaussian is one whose Real_Opacity is below min_opacity. A Gaussian that isn't faint is
always kept, and no Gaussian is dropped for its opacity alone (Req 2.4, 2.5). Space is cut into
Cubes whose side, the Cube_Edge, is the Trimmed_Box's longest side divided by
faint_cube_divisions. A Cube keeps its Faint_Gaussians when it holds at least faint_min_per_cube
of them and their Paint is more than faint_min_paint_share of the whole Cube's, and drops them
otherwise (Req 2.9). A Gaussian's Paint is its Real_Opacity times pi times its two largest
Real_Scales. So a sheet of faint Gaussians that together paint a visible surface stays, and
scattered faint specks go (A3).

Everything is worked out once, from the whole set the filter was given, before it drops any
Gaussian (Req 2.6). The filter builds no KD-tree and searches for no neighbours.

No finite stored value can overflow it (Req 2.11). Real_Opacity and Paint are only ever worked
with as logarithms, so no exp of a stored scale is taken, and each Cube's Paints are divided by
its largest one before any exp. A Cube coordinate stays a float64 and is never cast to an
integer. Only the underflow each step allows, which float64 gets right, runs inside
np.errstate(under="ignore").

From splatpipe, this module imports only splatpipe.clean.settings.
"""

from __future__ import annotations

import math

import numpy as np

from splatpipe.clean.settings import CleanSettings

# Format_Constants: the pi in Paint, and float32's smallest gap, which bounds how small a Cube
# matters. They describe the formats, not our choices, so they stay in code (Req 6.5).
LOG_PI = math.log(math.pi)
FLOAT32_HALF_STEP = 2.0 ** -150  # half the smallest gap between two float32 values, 2**-149
# The Extent is below 2**129, so this many divisions or more make an edge below the half step.
HUGE_DIVISIONS = 2 ** 279


def log_opacity(stored: np.ndarray) -> np.ndarray:
    """log Real_Opacity of the stored opacities `stored`, as a new float64 array, each at most 0.

    Real_Opacity is the logistic function of the stored opacity, so its log is
    -logaddexp(0, -stored), which numpy works out as log1p(exp(...)) of a value at or below 0.
    It's finite for every finite stored opacity: -3.4e38 gives -3.4e38, 0 gives -log 2 and
    3.4e38 gives 0. Only underflow can happen inside it, and it's right, because the exp of a
    very negative number really is 0 to float64 (Req 2.11).
    """
    with np.errstate(under="ignore"):
        return -np.logaddexp(0.0, -stored, dtype=np.float64)


def log_paint(log_alpha: np.ndarray, top_two: np.ndarray) -> np.ndarray:
    """log Paint, worked in place from log Real_Opacity `log_alpha` and the two largest scales.

    Paint is Real_Opacity x pi x the two largest Real_Scales. A stored scale is already the log
    of its Real_Scale, so log Paint is log_alpha + LOG_PI + top_two, where `top_two` is the sum
    of the two largest stored scales, in float64. `log_alpha` becomes the log Paint and is
    returned, so no second array is made. Every term is finite, so the sum is finite, from about
    -1.0e39 to 6.8e38, and no exp of a stored scale is ever taken (Req 2.11).
    """
    log_alpha += LOG_PI
    log_alpha += top_two
    return log_alpha


def cube_groups(points: np.ndarray, s: CleanSettings) -> tuple[np.ndarray, np.ndarray]:
    """The rows of `points` sorted by Cube, and a bool array, True at each Cube's first one.

    `points` is the set's centres, an (n, 3) float64 array of finite values with n above 0. The
    Cubes come from the whole set, before anything is dropped (Req 2.6):

    1. The Trimmed_Box runs on each axis from the extent_trim_percent percentile of the centres
       to the (100 - extent_trim_percent) percentile, each interpolated linearly, as
       np.percentile does by default.
    2. The Extent is its longest side, and never below 0, so a box that rounding made a hair
       inside out can't give a negative one.
    3. When the Extent is 0, every Gaussian is in one Cube (Req 2.8, A12).
    4. The Cube_Edge is the Extent divided by faint_cube_divisions.
    5. Each axis's Cube coordinate is floor((p - low) / edge), so a Cube holds its low face and
       not its high face, and a centre outside the Trimmed_Box is in a Cube of the same size
       outside it (Req 2.7). When the edge is at most FLOAT32_HALF_STEP, the coordinate is the
       centre itself.
    6. The rows are sorted by their three coordinates, and a Cube starts wherever one changes.
    """
    low, high = _trimmed_box(points, s.extent_trim_percent)
    extent = max(0.0, max(high[axis] - low[axis] for axis in range(3)))
    if extent == 0.0:
        starts = np.zeros(len(points), dtype=bool)
        starts[0] = True
        return np.arange(len(points)), starts
    return _groups(_cube_cells(points, low, extent, s.faint_cube_divisions))


def faint_keep(points: np.ndarray, stored_opacity: np.ndarray, top_two: np.ndarray,
               s: CleanSettings) -> np.ndarray:
    """Which Gaussians the Faint_Filter keeps, as a bool array with one per row of `points`.

    `points` is the set the filter was given, as from positions, so every centre is finite.
    `stored_opacity` is that set's opacity as float64, and `top_two` the sum of each Gaussian's
    two largest stored scales, in float64. None of the three is changed. Everything comes from
    the whole set, before any Gaussian is dropped (Req 2.6):

    1. An empty set gives an empty answer, because it has no Trimmed_Box (Req 2.10).
    2. Each Gaussian's log Real_Opacity says whether it's a Faint_Gaussian. Every Gaussian that
       isn't one is kept (Req 2.4).
    3. With no Faint_Gaussian, every Gaussian is kept at once, and no Cubes are made.
    4. The log Real_Opacity becomes the log Paint, in place, and the Gaussians go in Cubes.
    5. In Cube order, each row gets its Cube's number, from 0 up.
    6. Each Cube's Paints are divided by its largest one.
    7. Each Cube's Cube_Paint, Faint_Paint and count of Faint_Gaussians are summed.
    8. Each Cube that holds a Faint_Gaussian decides on all of them by the cube rule (Req 2.5,
       2.9).
    9. A Gaussian is kept when it isn't a Faint_Gaussian or its Cube keeps its Faint_Gaussians,
       and the answers go back in the set's order.

    Each array is freed once it's done with. Besides `order`, at most four arrays in Cube order,
    each of one float64 or int64 a Gaussian, are alive at once.
    """
    n = len(points)
    if n == 0:
        return np.zeros(0, dtype=bool)
    la = log_opacity(stored_opacity)
    faint = _faint_gaussians(la, s.min_opacity)
    if not faint.any():
        return np.ones(n, dtype=bool)
    lp = log_paint(la, top_two)  # la's own array, now the log Paint
    order, starts = cube_groups(points, s)
    g = np.cumsum(starts)  # each sorted row's Cube, from 0 to m - 1
    g -= 1
    bounds = np.flatnonzero(starts)  # each Cube's first sorted row
    del starts
    paint = lp[order]
    del la, lp
    faint_s = faint[order]
    del faint
    _shift_to_paint(paint, g, bounds)
    cube_keeps = _cube_keeps(paint, g, faint_s, len(bounds), s)
    del paint, bounds
    keep = np.empty(n, dtype=bool)
    keep[order] = ~faint_s | cube_keeps[g]
    return keep


# The Cubes

def _trimmed_box(points: np.ndarray, t: float) -> tuple[list[float], list[float]]:
    """The Trimmed_Box's low and high corners, each as three Python floats, one per axis.

    Each corner comes from np.percentile, one axis at a time, so each call copies one column.
    Its interpolation between two tiny values can underflow, which is right, so it runs inside
    np.errstate(under="ignore"). The corners are Python floats, so Python works out the Extent
    and the Cube_Edge. A tiny Extent's division then underflows without an error, which is right
    too, because such an edge is at most FLOAT32_HALF_STEP and the centres are the Cubes.
    """
    low: list[float] = []
    high: list[float] = []
    with np.errstate(under="ignore"):
        for axis in range(3):
            low.append(float(np.percentile(points[:, axis], t)))
            high.append(float(np.percentile(points[:, axis], 100 - t)))
    return low, high


def _cube_cells(points: np.ndarray, low: list[float], extent: float,
                divisions: int) -> np.ndarray:
    """Each Gaussian's Cube coordinates, as a (3, n) float64 array with one row per axis.

    The Cube_Edge is extent / divisions. A count of HUGE_DIVISIONS or more skips the division,
    because Python can't turn an int above about 1.8e308 into a float, and the Extent is below
    2**129, so such a count gives an edge below FLOAT32_HALF_STEP anyway.

    Above FLOAT32_HALF_STEP, each row is made in place as p - low and then floor-divided by the
    edge in place, so no temporary array is made. numpy's floor division takes the exact
    remainder first, as Python's // does. The coordinate stays a float64 and is never cast to an
    integer: p - low is at most 6.8e38 in size, so the quotient is below 2**279, far inside
    float64 (Req 2.7, 2.11). A p - low far smaller than the edge can underflow in the division,
    which is right, so it runs inside np.errstate(under="ignore").

    At or below FLOAT32_HALF_STEP, the coordinate is the centre itself. Two different float32
    values are at least 2**-149 apart, two such edges or more, so they're in different Cubes on
    that axis, and equal ones are in the same Cube: grouping by the centres is exactly grouping
    by the Cubes.
    """
    cells = np.empty((3, len(points)), dtype=np.float64)
    edge = 0.0 if divisions >= HUGE_DIVISIONS else extent / divisions
    if edge <= FLOAT32_HALF_STEP:
        np.copyto(cells, points.T)
        return cells
    for axis in range(3):
        np.subtract(points[:, axis], low[axis], out=cells[axis])
    with np.errstate(under="ignore"):
        np.floor_divide(cells, edge, out=cells)
    return cells


def _groups(cells: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The rows sorted by their Cube coordinates `cells`, and True where each Cube starts.

    np.lexsort makes three stable typed sorts of float64 keys, each a contiguous row that it
    sorts where it is, with no copy. Then one axis at a time, the coordinates are gathered in
    sorted order and each is compared with the one before it, so only one gathered axis and its
    comparison are alive at once. != on float64 treats -0.0 and 0.0 as equal, so they're in one
    Cube.
    """
    order = np.lexsort((cells[2], cells[1], cells[0]))
    starts = np.zeros(len(order), dtype=bool)
    starts[0] = True
    for axis_cells in cells:
        in_order = axis_cells[order]
        starts[1:] |= in_order[1:] != in_order[:-1]
        del in_order
    return order, starts


# The cube rule

def _faint_gaussians(la: np.ndarray, min_opacity: float) -> np.ndarray:
    """Which Gaussians are Faint_Gaussians, from their log Real_Opacity `la` (Req 2.4).

    A Gaussian is one when its Real_Opacity is below min_opacity, so when la is below
    log(min_opacity). At a min_opacity of 0 none is, because no Real_Opacity is below 0. At 1
    every one is, because every Real_Opacity of a finite stored opacity is below 1, and la < 0
    can't be trusted there: log_opacity's underflow makes la 0 for a stored opacity above about
    745.
    """
    if min_opacity == 0:
        return np.zeros(len(la), dtype=bool)
    if min_opacity == 1:
        return np.ones(len(la), dtype=bool)
    return la < math.log(min_opacity)


def _shift_to_paint(paint: np.ndarray, g: np.ndarray, bounds: np.ndarray) -> None:
    """Turn the log Paints `paint`, in Cube order, into each Paint over its Cube's largest.

    `g` is each row's Cube and `bounds` each Cube's first row. Each Cube's largest log Paint is
    subtracted from its rows, in place, which can't overflow, because every log Paint is from
    about -1.0e39 to 6.8e38. Every value is then at most 0, so every exp is of a value at or
    below 0: nothing overflows, and each result is from 0 to 1. Only underflow can happen in the
    exp, and it's right, so it runs inside np.errstate(under="ignore") (Req 2.11).
    """
    top = np.maximum.reduceat(paint, bounds)  # each Cube's largest log Paint
    paint -= top[g]
    with np.errstate(under="ignore"):
        np.exp(paint, out=paint)


def _cube_keeps(paint: np.ndarray, g: np.ndarray, faint_s: np.ndarray, m: int,
                s: CleanSettings) -> np.ndarray:
    """Which of the `m` Cubes keep their Faint_Gaussians, as a bool array with one per Cube.

    `paint` is each sorted row's Paint over its Cube's largest, `g` its Cube and `faint_s`
    whether it's a Faint_Gaussian. Cube_Paint sums every row of a Cube, faint or not, and
    Faint_Paint only its Faint_Gaussians' rows, both with np.bincount. Both are the real sums
    divided by the same positive number, the Cube's largest Paint, so the share test gives the
    real rule's answer (Req 2.9).

    A Cube keeps its Faint_Gaussians when it holds at least faint_min_per_cube of them and its
    Faint_Paint is above faint_min_paint_share times its Cube_Paint, a product with no division.
    The min with n + 1 changes no answer, because no Cube holds more than the set's n
    Gaussians, and keeps a very large setting from meeting an int64 array. With the share at 0
    the paint condition holds for every Cube with a Faint_Gaussian, because every real Paint is
    above 0, so it's taken as met rather than trust a sum that could have underflowed. Only the
    Cubes that hold a Faint_Gaussian decide anything.
    """
    cube_paint = np.bincount(g, weights=paint, minlength=m)
    g_faint = g[faint_s]
    faint_paint = np.bincount(g_faint, weights=paint[faint_s], minlength=m)
    faint_count = np.bincount(g_faint, minlength=m)
    del g_faint
    keeps = faint_count >= min(s.faint_min_per_cube, len(g) + 1)
    if s.faint_min_paint_share > 0:
        keeps &= faint_paint > s.faint_min_paint_share * cube_paint
    return keeps
