"""Tests for the clean step's filters, in splatpipe/clean/, on Splat_Data and centres that the
tests build as arrays themselves (Req 10.1).

Each filter has a part of its own below, and so do the neighbour search that the
Isolated_Filter uses and the Filter_Order that runs the three filters. A part holds its unit
tests first and its property tests after them:

- The Broken_Filter, in broken.py: a NaN, +infinity or -infinity in each Required_Property in
  turn drops a Gaussian, and so does the zero-length quaternion (0, -0.0, 0, 0). A NaN in a colour
  alone keeps it, and so does a quaternion that's the smallest subnormal float32 in one component
  and 0 in the rest (Req 2.2, 2.3). Quaternion components of 3e38 and -3e38 raise nothing
  (Req 2.11). Property 2 checks broken_keep against an oracle on Splat_Data from splats, NaNs
  and infinities in any property included, and that it keeps all it kept when run again
  (Req 2.2, 2.3, 2.19).
- The neighbour search, in neighbours.py: positions gives the stored float32 centres exactly, as
  float64. nearest_others gives every point k others, none of them itself, both with more than
  k + 1 centres on one spot and with another point at distance 0 ahead of a point in its own row
  (Req 2.16). Property 3 checks nearest_others on sets from centre_sets, coincident pairs
  included: each Gaussian gets k others, none of them itself and none twice, which are exactly
  its k nearest others found by brute force, at their brute-force Centre_Distances (Req 2.16).
- The Faint_Filter, in faint.py: log_opacity is finite and at most 0 for stored opacities from
  -3.4e38 to 3.4e38, and log_paint is finite for stored scales of 3.4e38 and -3.4e38 (Req 2.11).
  A Cube of faint_min_per_cube Faint_Gaussians that paint just over faint_min_paint_share of it
  keeps them all, one at a stored opacity of -1000 included, and drops them all with one fewer or
  a little less Faint_Paint, while a Gaussian at or above min_opacity beside them is kept
  (Req 2.4, 2.5, 2.9). A min_opacity of 0 makes no Faint_Gaussian and one of 1 makes every
  Gaussian one, and a faint_min_paint_share of 0 keeps a Cube whose Faint_Paint underflows. A
  centre on a Cube face goes to the Cube above it, a far outlier sets the Cube_Edge only when
  nothing is trimmed, and a centre at 3e38 raises nothing (Req 2.6, 2.7, 2.11). Coincident centres
  and a set of one give one Cube, an empty set gives an empty answer, and HUGE_DIVISIONS divisions
  or subnormal centres give each distinct centre a Cube of its own (Req 2.8, 2.10). Property 4
  checks faint_keep against cube_rule, a reference in plain Python that takes no logarithm and
  shifts no sum, on Splat_Data from reference_splats, the empty set included, and settings from
  valid_settings (Req 2.4 to 2.7, 2.9, 2.10, 2.20). Property 8 checks that faint_keep raises no
  floating-point error, and log_opacity and log_paint give finite values, on Splat_Data from
  extreme_splats, whose centres, stored opacities and stored scales are any finite float32
  values. It also checks, on draws from rescaled_splats, that adding c to every stored scale and
  multiplying every centre by 2**m changes no answer (Req 2.11).
- The Isolated_Filter, in isolated.py: a set of neighbours Gaussians, however far apart, is kept
  whole, and a spy on cKDTree shows that no tree is built (Req 2.14). Coincident centres give a
  Neighbour_Median of 0 and are kept, with no division by 0 (Req 2.15). One far Gaussian next to
  a tight cluster is dropped (Req 2.12). Two even grids far apart, the sparse one's spacing 100
  times the dense one's, keep every Gaussian, where the test shows that a global distance cutoff
  would drop the whole sparse grid (A5). Property 5 checks isolated_keep against isolation_rule,
  a brute-force reference in plain Python, on centres from reference_centres and settings from
  valid_settings (Req 2.12 to 2.14, 2.21).
- The Filter_Order, in filters.py: run_filters on Splat_Data of two broken Gaussians, a tight
  cluster, a floater and the floater's seven faint companions. Given the companions, the
  Isolated_Filter would keep the floater, but the Faint_Filter drops them first, so the
  Isolated_Filter drops it, which shows the Filter_Order and that each filter gets the set the
  one before it kept (Req 2.1, 2.13). The counts reach add_count in order, each as an int, and
  when isolated_keep raises, only dropped_broken and dropped_faint were added (Req 2.17, 7.2,
  7.3). A max_workers of 0 runs, with 1 worker for the Isolated_Filter's tree query.
  Property 1 checks run_filters on Splat_Data from splats and settings from valid_settings: it
  gives back increasing rows, none repeated, exactly the rows the test gets by calling
  broken_keep, faint_keep and isolated_keep one after the other, it leaves the Splat_Data's bytes
  as they were, and each dropped_* count is the size of the set its filter was given less the
  size of the set it kept (Req 2.1, 2.17, 2.18). Property 6 runs run_filters with the real
  config.toml's [clean] settings on Splat_Data from grids, evenly spaced grids of more than
  neighbours opaque Gaussians, full of ties: it keeps every Gaussian, and each dropped_* count is
  0, however the tree breaks the ties (Req 2.22).

The autouse fixture float_errors_raise runs every test here inside np.errstate(all="raise"), so
an overflow, an invalid value, a division by 0 or an underflow anywhere in a filter fails the
test. Only the underflow a filter allows on purpose, inside its own np.errstate(under="ignore"),
passes (Req 2.11).
"""

import functools
import math
import statistics
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import NoReturn

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from scipy.spatial import cKDTree

from splatpipe.clean.broken import broken_keep
from splatpipe.clean.faint import (FLOAT32_HALF_STEP, HUGE_DIVISIONS, cube_groups, faint_keep,
                                   log_opacity, log_paint)
from splatpipe.clean.filters import run_filters
from splatpipe.clean.isolated import isolated_keep
from splatpipe.clean.neighbours import nearest_others, positions
from splatpipe.clean.settings import CleanSettings, check_settings
from splatpipe.config import REPO_ROOT, load_config

from splats import (Rescaled, centre_distances, centre_sets, clean_settings, extreme_splats,
                    gaussians, grids, reference_centres, reference_splats, rescaled_splats,
                    splats, valid_settings)

# The smallest float32 above 0, 2**-149, a subnormal.
SMALLEST_SUBNORMAL = np.finfo(np.float32).smallest_subnormal


@pytest.fixture(autouse=True)
def float_errors_raise() -> Iterator[None]:
    """Run the test with every one of numpy's floating-point errors set to raise."""
    with np.errstate(all="raise"):
        yield


def test_every_test_here_runs_with_numpys_floating_point_errors_set_to_raise() -> None:
    assert np.geterr() == dict.fromkeys(("divide", "over", "under", "invalid"), "raise")


# The Broken_Filter

# The eleven Required_Properties, written out here rather than taken from ply_read.py's REQUIRED,
# and the four that hold the rotation quaternion.
REQUIRED_PROPERTIES = ("x", "y", "z", "opacity", "scale_0", "scale_1", "scale_2",
                       "rot_0", "rot_1", "rot_2", "rot_3")
ROTATION = ("rot_0", "rot_1", "rot_2", "rot_3")


def kept(data: np.ndarray) -> list[bool]:
    """broken_keep's answer for `data`, which must be a bool array with one entry per Gaussian,
    as a list."""
    keep = broken_keep(data)
    assert keep.dtype == np.bool_ and keep.shape == (len(data),)
    return keep.tolist()


def with_quaternion(*components: float) -> np.ndarray:
    """Three plain Gaussians from gaussians, the middle one with the quaternion `components`."""
    data = gaussians(3)
    for name, value in zip(ROTATION, components, strict=True):
        data[name][1] = value
    return data


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf], ids=["NaN", "+infinity", "-infinity"])
@pytest.mark.parametrize("name", REQUIRED_PROPERTIES)
def test_a_nan_or_an_infinity_in_any_required_property_drops_the_gaussian(name: str,
                                                                          value: float) -> None:
    data = gaussians(3)
    data[name][1] = value
    assert kept(data) == [True, False, True]


def test_the_zero_length_quaternion_0_minus_0_0_0_drops_the_gaussian() -> None:
    data = with_quaternion(0.0, -0.0, 0.0, 0.0)
    assert np.signbit(data["rot_1"][1])  # stored as -0.0, whose bits aren't those of 0.0
    assert kept(data) == [True, False, True]


def test_a_nan_in_a_colour_alone_leaves_the_gaussian_kept() -> None:
    data = gaussians(3)
    data["f_dc_0"][1] = np.nan
    assert kept(data) == [True, True, True]


@pytest.mark.parametrize("name", ROTATION)
def test_the_smallest_subnormal_in_one_quaternion_component_and_0_in_the_rest_is_kept(
        name: str) -> None:
    # Its square is 0 in float32, so a test of the squared length would call this quaternion
    # zero length, and would raise for the underflow here.
    data = with_quaternion(*(SMALLEST_SUBNORMAL if component == name else 0.0
                             for component in ROTATION))
    assert kept(data) == [True, True, True]


def test_quaternion_components_of_3e38_and_minus_3e38_raise_nothing() -> None:
    # Each one's square overflows float32, which would raise here.
    data = with_quaternion(3e38, -3e38, 3e38, -3e38)
    assert kept(data) == [True, True, True]


def can_be_drawn(gaussian: np.void) -> bool:
    """The oracle for Property 2, which shares no code with broken_keep: whether `gaussian`, one
    row of Splat_Data, has a finite value in every Required_Property and a quaternion that isn't
    four zeros, looked at one value at a time with math.isfinite and ==."""
    return (all(math.isfinite(float(gaussian[name])) for name in REQUIRED_PROPERTIES)
            and not all(float(gaussian[name]) == 0 for name in ROTATION))


@settings(deadline=None)
@given(splats())
def test_the_broken_filter_keeps_exactly_the_gaussians_that_can_be_drawn(
        data: np.ndarray) -> None:
    """Feature: clean, Property 2: The Broken_Filter keeps exactly the Gaussians that can be drawn

    **Validates: Requirements 2.2, 2.3, 2.19**
    """
    # The oracle looks at no Carried_Property, so a NaN or an infinity in one never changes the
    # answer.
    assert kept(data) == [can_be_drawn(gaussian) for gaussian in data]
    survivors = data[broken_keep(data)]
    assert kept(survivors) == [True] * len(survivors)


# The neighbour search

def test_positions_gives_the_float32_centres_exactly_as_float64() -> None:
    data = gaussians(5)  # centres on a jittered lattice, which no short decimal gives exactly
    data["x"][1], data["y"][1], data["z"][1] = 3e38, -SMALLEST_SUBNORMAL, -3e38  # and the extremes
    rows = np.array([1, 2, 4])
    points = positions(data, rows)
    assert points.dtype == np.float64 and points.shape == (3, 3) and points.flags.c_contiguous
    assert points.tolist() == [[float(data[axis][row]) for axis in ("x", "y", "z")]
                               for row in rows]


def centre_distance(a: np.ndarray, b: np.ndarray) -> float:
    """The Centre_Distance between the float64 centres `a` and `b`, as sqrt(dx*dx + dy*dy + dz*dz)
    summed in that order, which is how the tree works it out, so the two agree to within 4 ulps:
    scipy on arm64 uses fused multiply-add, so each of the three squares in the tree's sum skips
    one rounding, and its distance can differ from this one by up to about 3 ulps.
    math.dist can differ from both by one unit in the last place."""
    dx, dy, dz = (float(p) - float(q) for p, q in zip(a, b, strict=True))
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def check_within_4_ulps(distances: Sequence[float], expected: Sequence[float]) -> None:
    """Check that the tree's `distances` and the `expected` ones from centre_distance() have the
    same length and that each distance is within 4 * math.ulp of its counterpart, in order.
    scipy on arm64 uses fused multiply-add, so each of the three squares in the tree's sum skips
    one rounding, and the sum under the square root can differ from centre_distance()'s by about
    3 ulps. After the square root and its own rounding, the distance can differ by up to about
    2 ulps, and by 3 near a power of two, so 4 covers it."""
    assert len(distances) == len(expected), (distances, expected)
    for distance, counterpart in zip(distances, expected, strict=True):
        assert abs(distance - counterpart) <= 4 * math.ulp(counterpart), (distances, expected)


def raw_rows(tree: cKDTree, k: int) -> list[list[int]]:
    """The tree's own rows for its points, as nearest_others asks for them: the indices of each
    point's k + 1 nearest points, before the point itself is dropped."""
    return tree.query(tree.data, k=list(range(1, k + 2)))[1].tolist()


def check_nearest_others(tree: cKDTree, k: int) -> None:
    """Check nearest_others on every point of `tree`. It gives each one k others, none of them
    itself and none twice, with the Centre_Distance to each. Their distances, nearest first, are
    those of the point's k nearest others found by brute force, so they're as near as any k
    others can be, whichever others at the same distance the tree picked (A10)."""
    dist, idx = nearest_others(tree, np.arange(tree.n), k, workers=1)
    assert dist.shape == idx.shape == (tree.n, k)
    for point, (distances, others) in enumerate(zip(dist.tolist(), idx.tolist())):
        assert point not in others and len(set(others)) == k
        centre = tree.data[point]
        check_within_4_ulps(distances, [centre_distance(centre, tree.data[other])
                                        for other in others])
        check_within_4_ulps(distances, sorted(centre_distance(centre, tree.data[other])
                                              for other in range(tree.n) if other != point)[:k])


def test_more_than_k_plus_1_centres_on_one_spot_each_get_k_others_but_never_themselves() -> None:
    # k = 2: five Gaussians on one centre, more than k + 1, and three on centres of their own.
    tree = cKDTree(np.array([(0, 0, 0)] * 5 + [(3, 0, 0), (0, 4, 0), (0, 0, 12)], np.float64))
    # The tree gives all five the same three of them, so two are missing from their own rows.
    assert any(point not in row for point, row in enumerate(raw_rows(tree, 2)))
    check_nearest_others(tree, 2)


def test_a_point_with_another_at_distance_0_ahead_of_it_gets_k_others_but_never_itself() -> None:
    # k = 2: two Gaussians on one centre, and two on centres of their own.
    tree = cKDTree(np.array([(0, 0, 0)] * 2 + [(3, 0, 0), (0, 4, 0)], np.float64))
    # The tree gives the two the same row, so each is in its own row, and one of them second.
    rows = raw_rows(tree, 2)
    assert all(point in row for point, row in enumerate(rows))
    assert any(row[0] != point for point, row in enumerate(rows))
    check_nearest_others(tree, 2)


@settings(deadline=None)
@given(centre_sets())
def test_no_gaussian_is_its_own_neighbour(centres_and_k: tuple[np.ndarray, int]) -> None:
    """Feature: clean, Property 3: No Gaussian is its own neighbour

    **Validates: Requirements 2.16**
    """
    centres, k = centres_and_k
    tree = cKDTree(centres)
    dist, idx = nearest_others(tree, np.arange(tree.n), k, workers=1)
    assert dist.shape == idx.shape == (tree.n, k)
    for point, (distances, others) in enumerate(zip(dist.tolist(), idx.tolist())):
        assert point not in others and len(set(others)) == k
        # No Gaussian of a set from centre_sets has its k-th and (k + 1)-th nearest others at one
        # distance, so its k nearest others are one set, however the ties nearer in are broken.
        by_distance = sorted((centre_distance(centres[point], centres[other]), other)
                             for other in range(tree.n) if other != point)
        assert set(others) == {other for _, other in by_distance[:k]}
        check_within_4_ulps(distances, [centre_distance(centres[point], centres[other])
                                        for other in others])


# The Faint_Filter

# The test settings' faint_min_per_cube and faint_min_paint_share, 8 and 0.10.
PER_CUBE = clean_settings().faint_min_per_cube
SHARE = clean_settings().faint_min_paint_share
# Stored opacities from one end of float32's range to the other.
STORED_OPACITIES = (0.0, 1.0, -1.0, 40.0, -40.0, 1000.0, -1000.0, 3.4e38, -3.4e38)


@dataclass(frozen=True)
class Stored:
    """One Gaussian as the Faint_Filter reads it: its stored centre, opacity and three scales."""

    centre: tuple[float, float, float]
    opacity: float
    scales: tuple[float, float, float] = (0.0, 0.0, 0.0)  # so each Real_Scale is 1


def log_real_opacity(stored: float) -> float:
    """log Real_Opacity for the stored opacity `stored`, in plain Python: log(1 / (1 + e^-o)),
    written as min(o, 0) - log(1 + e^-|o|) so that no exp can overflow."""
    return min(stored, 0.0) - math.log1p(math.exp(-abs(stored)))


def logit(real_opacity: float) -> float:
    """The stored opacity whose Real_Opacity is `real_opacity`."""
    return math.log(real_opacity / (1 - real_opacity))


def top_two(scales: Sequence[float]) -> float:
    """The sum of the two largest of a Gaussian's three stored scales."""
    return sum(sorted(scales)[1:])


def paint(gaussian: Stored) -> float:
    """The Gaussian's Paint by its rule: Real_Opacity x pi x the two largest Real_Scales."""
    smaller, larger = sorted(gaussian.scales)[1:]
    return (math.exp(log_real_opacity(gaussian.opacity)) * math.pi
            * math.exp(smaller) * math.exp(larger))


def faint_answer(members: Sequence[Stored], **settings: object) -> list[bool]:
    """faint_keep's answer for the Gaussians `members`, as a list, with the test settings changed
    as `settings` says, like faint_answer(members, min_opacity=0).

    faint_keep gets the three float64 arrays filters.py gives it: the centres, one row each, the
    stored opacities, and the sum of each Gaussian's two largest stored scales. Its answer must
    be a bool array with one entry per Gaussian.
    """
    points = np.array([g.centre for g in members], dtype=np.float64).reshape(-1, 3)
    opacity = np.array([g.opacity for g in members], dtype=np.float64)
    sums = np.array([top_two(g.scales) for g in members], dtype=np.float64)
    keep = faint_keep(points, opacity, sums, clean_settings(**settings))
    assert keep.dtype == np.bool_ and keep.shape == (len(members),)
    return keep.tolist()


def cubes(points: Sequence[Sequence[float]], **settings: object) -> set[frozenset[int]]:
    """cube_groups's Cubes for the centres `points`, each as the set of its rows, with the test
    settings changed as `settings` says. Every row must be in exactly one Cube."""
    order, starts = cube_groups(np.array(points, dtype=np.float64), clean_settings(**settings))
    assert sorted(order.tolist()) == list(range(len(points)))
    assert starts.dtype == np.bool_ and starts.shape == (len(points),) and starts[0]
    bounds = [*np.flatnonzero(starts).tolist(), len(points)]
    return {frozenset(order[a:b].tolist()) for a, b in zip(bounds, bounds[1:])}


def test_log_opacity_is_finite_and_at_most_0_for_stored_opacities_from_minus_to_plus_3_4e38(
        ) -> None:
    la = log_opacity(np.array(STORED_OPACITIES))
    assert la.dtype == np.float64
    assert np.isfinite(la).all() and (la <= 0).all()
    # So -3.4e38 gives -3.4e38, 0 gives -log 2, and 1000 and 3.4e38 give 0.
    assert la.tolist() == pytest.approx([log_real_opacity(o) for o in STORED_OPACITIES])


def test_log_paint_is_finite_for_stored_scales_of_3_4e38_and_minus_3_4e38() -> None:
    # A stored opacity of 3.4e38 or -3.4e38 with each set of stored scales, so log Paint runs
    # from about -1.0e39 to 6.8e38.
    pairs = [(o, scales) for o in (3.4e38, -3.4e38)
             for scales in ((3.4e38,) * 3, (-3.4e38,) * 3, (3.4e38, -3.4e38, 0.0))]
    lp = log_paint(log_opacity(np.array([o for o, _ in pairs])),
                   np.array([top_two(scales) for _, scales in pairs]))
    assert np.isfinite(lp).all()
    assert lp.tolist() == pytest.approx([log_real_opacity(o) + math.log(math.pi) + top_two(scales)
                                         for o, scales in pairs])


# Two bright Gaussians that make the Trimmed_Box run from (0, 0, 0) to (128, 128, 128) when
# extent_trim_percent is 0, so at the test settings' 128 faint_cube_divisions the Cube_Edge is 1.
FRAME = (Stored((0.0, 0.0, 0.0), 3.0), Stored((128.0, 128.0, 128.0), 3.0))
# The bright Gaussian of the cube rule's Cube has a Real_Opacity of 0.5, which is above the test
# settings' min_opacity of 0.005 and exactly at a min_opacity of 0.5.
AT_OR_ABOVE = pytest.mark.parametrize("min_opacity", [0.005, 0.5], ids=["above", "at"])


def in_cube(k: int) -> tuple[float, float, float]:
    """The k-th of 16 centres in the Cube from (5, 5, 5) up to (6, 6, 6) when the Cube_Edge is 1."""
    return (5 + k / 16, 5.5, 5.5)


def faint_cube(faint_opacity: float, *, minus_1000: bool = True) -> tuple[list[Stored], Stored]:
    """The cube rule's Cube, as its Faint_Gaussians and its bright Gaussian, which comes last.

    faint_min_per_cube - 1 Faint_Gaussians have the Real_Opacity `faint_opacity`, and one more has
    a stored opacity of -1000 unless `minus_1000` is false. Each of their Real_Scales is 1, so
    each paints faint_opacity x pi, and the one at -1000 next to nothing. The bright Gaussian's
    Real_Opacity is 0.5 and its two largest Real_Scales are 1 and 0.5, so it paints 0.25 pi.
    """
    opacities = [logit(faint_opacity)] * (PER_CUBE - 1) + ([-1000.0] if minus_1000 else [])
    faint = [Stored(in_cube(k), o) for k, o in enumerate(opacities)]
    half = -math.log(2)
    return faint, Stored(in_cube(len(faint)), 0.0, (0.0, half, half))


def faint_share(faint: Sequence[Stored], bright: Stored) -> float:
    """The Faint_Paint of the Cube that holds `faint` and `bright` over its Cube_Paint, with each
    Paint worked out by its rule."""
    faint_paint = math.fsum(paint(g) for g in faint)
    return faint_paint / (faint_paint + paint(bright))


@AT_OR_ABOVE
def test_a_cube_of_faint_min_per_cube_faint_gaussians_over_the_paint_share_keeps_them_all(
        min_opacity: float) -> None:
    faint, bright = faint_cube(0.004)
    # 7 x 0.004 pi = 0.028 pi of a Cube_Paint of 0.278 pi, a share of 0.1007.
    assert len(faint) == PER_CUBE and faint_share(faint, bright) > 1.005 * SHARE
    answer = faint_answer([*FRAME, *faint, bright], extent_trim_percent=0, min_opacity=min_opacity)
    assert answer == [True] * (len(faint) + 3)


@AT_OR_ABOVE
def test_the_same_cube_with_one_faint_gaussian_fewer_drops_them_all(min_opacity: float) -> None:
    # Leaving out the one at -1000, which paints next to nothing, leaves the share at 0.1007.
    faint, bright = faint_cube(0.004, minus_1000=False)
    assert len(faint) == PER_CUBE - 1 and faint_share(faint, bright) > 1.005 * SHARE
    answer = faint_answer([*FRAME, *faint, bright], extent_trim_percent=0, min_opacity=min_opacity)
    assert answer == [True, True, *[False] * len(faint), True]


@AT_OR_ABOVE
def test_the_same_cube_with_a_little_less_faint_paint_drops_them_all(min_opacity: float) -> None:
    faint, bright = faint_cube(0.0039)
    # 7 x 0.0039 pi = 0.0273 pi of a Cube_Paint of 0.2773 pi, a share of 0.0984.
    assert len(faint) == PER_CUBE and faint_share(faint, bright) < 0.99 * SHARE
    answer = faint_answer([*FRAME, *faint, bright], extent_trim_percent=0, min_opacity=min_opacity)
    assert answer == [True, True, *[False] * len(faint), True]


def in_a_row(*stored_opacities: float) -> list[Stored]:
    """Gaussians with the stored opacities given, at (0, 0, 0), (1, 0, 0) and so on, so that at
    the test settings each is alone in its Cube."""
    return [Stored((float(k), 0.0, 0.0), o) for k, o in enumerate(stored_opacities)]


def test_a_min_opacity_of_0_makes_no_gaussian_faint() -> None:
    # At the test settings' min_opacity each is a Faint_Gaussian alone in its Cube, so it's dropped.
    specks = in_a_row(-3.4e38, -1000.0, -40.0, -6.0)
    assert faint_answer(specks) == [False] * 4
    assert faint_answer(specks, min_opacity=0) == [True] * 4


def test_a_min_opacity_of_1_makes_every_gaussian_faint_a_stored_opacity_of_3_4e38_included(
        ) -> None:
    # At the test settings' min_opacity none is a Faint_Gaussian, so each is kept. At 1 each is,
    # alone in its Cube, so it's dropped, though log_opacity gives 0, the log of 1, at 1000 and
    # 3.4e38.
    bright = in_a_row(3.4e38, 1000.0, 40.0, 0.0)
    assert faint_answer(bright) == [True] * 4
    assert faint_answer(bright, min_opacity=1) == [False] * 4


def test_a_paint_share_of_0_keeps_a_cube_whose_faint_paint_underflows_next_to_a_bright_one(
        ) -> None:
    # Over the bright Gaussian's Paint, each Faint_Gaussian's is about e^-1000, which is 0 in
    # float64, so their Faint_Paint underflows to 0.
    faint = [Stored(in_cube(k), -1000.0) for k in range(PER_CUBE)]
    members = [*FRAME, *faint, Stored(in_cube(PER_CUBE), 0.0)]
    assert faint_answer(members, extent_trim_percent=0) == [True, True, *[False] * PER_CUBE, True]
    assert faint_answer(members, extent_trim_percent=0,
                        faint_min_paint_share=0) == [True] * (PER_CUBE + 3)


@pytest.mark.parametrize("axis", [0, 1, 2], ids=["x", "y", "z"])
def test_a_centre_on_a_cube_face_goes_to_the_cube_above_and_one_a_step_below_to_the_cube_below(
        axis: int) -> None:
    # With extent_trim_percent at 0, the corners at 0 and 32 make the Extent 32, so at the test
    # settings' 128 faint_cube_divisions the Cube_Edge is 0.25, and there's a Cube face at 8.
    def at(value: float) -> tuple[float, ...]:
        centre = [16.125] * 3  # in the middle of a Cube on the other two axes
        centre[axis] = value
        return tuple(centre)

    points = [(0.0, 0.0, 0.0), (32.0, 32.0, 32.0),
              at(7.875), at(8.125), at(8.0), at(math.nextafter(8.0, 0.0))]
    assert cubes(points, extent_trim_percent=0) == {
        frozenset({0}), frozenset({1}), frozenset({2, 5}), frozenset({3, 4})}


@pytest.mark.parametrize("far", [1e6, 3e38])
def test_a_far_outlier_sets_the_cube_edge_at_a_trim_of_0_but_not_at_a_trim_of_1(far: float) -> None:
    # 199 Gaussians on whole-number points from (0, 0, 0) to (5, 5, 5), and one far outlier.
    lattice = [(float(x), float(y), float(z))
               for z in range(6) for y in range(6) for x in range(6)][:199]
    points = [*lattice, (far, 2.0, 2.0)]
    # At 0 the outlier makes the Extent `far`, and a Cube_Edge of far / 128 holds the lattice.
    assert cubes(points, extent_trim_percent=0) == {frozenset(range(199)), frozenset({199})}
    # At 1 the Trimmed_Box leaves the outlier out and runs from (0, 0, 0) to (5, 5, 5), so a
    # Cube_Edge of 5 / 128 gives every lattice point a Cube of its own. The outlier lands alone
    # in a far Cube of that size. Put in the box's last Cube, it would share the Cube of (5, 2, 2).
    assert cubes(points, extent_trim_percent=1) == {frozenset({k}) for k in range(200)}


def test_every_centre_the_same_gives_one_cube_and_so_does_a_set_of_one() -> None:
    assert cubes([(1.5, -2.25, 3.0)] * 5) == {frozenset(range(5))}
    assert cubes([(1.5, -2.25, 3.0)]) == {frozenset({0})}


def test_an_empty_set_gives_an_empty_answer() -> None:
    assert faint_answer([]) == []


@pytest.mark.parametrize("divisions", [HUGE_DIVISIONS, HUGE_DIVISIONS ** 4],
                         ids=["HUGE_DIVISIONS", "too big for a float"])
def test_huge_divisions_take_the_centre_path_and_give_each_distinct_centre_a_cube(
        divisions: int) -> None:
    # Two centres one float32 step apart on x, and two on one spot.
    points = [(0.0, 0.0, 0.0), (1.0, 2.0, 3.0), (1.0, 2.0, 3.0), (1.0 + 2.0 ** -23, 2.0, 3.0)]
    assert cubes(points, faint_cube_divisions=divisions) == {
        frozenset({0}), frozenset({1, 2}), frozenset({3})}


def test_subnormal_centres_with_a_tiny_extent_take_the_centre_path_and_give_each_a_cube() -> None:
    # Centres a few times the smallest float32 step from 0, two of them on one spot, so the
    # Extent is about 2.7e-45 and the Cube_Edge under FLOAT32_HALF_STEP.
    tiny = float(SMALLEST_SUBNORMAL)
    points = [(0.0, 0.0, 0.0), (tiny, 0.0, 0.0), (tiny, 0.0, 0.0), (0.0, tiny, 0.0),
              (0.0, 0.0, 2 * tiny), (-tiny, 0.0, 0.0)]
    assert cubes(points) == {frozenset({0}), frozenset({1, 2}), frozenset({3}), frozenset({4}),
                             frozenset({5})}


# How near a threshold a draw of Property 4 may come, as a relative distance. Nearer than that,
# faint_keep's path through logarithms and shifted sums and the reference's plain one can round
# to different answers, so the draw is dropped.
NEAR = 1e-9


def as_stored(data: np.ndarray) -> list[Stored]:
    """The Gaussians of the Splat_Data `data` as the Faint_Filter reads them, one Stored each,
    with every float32 value as the Python float that holds it exactly."""
    return [Stored((float(g["x"]), float(g["y"]), float(g["z"])), float(g["opacity"]),
                   (float(g["scale_0"]), float(g["scale_1"]), float(g["scale_2"])))
            for g in data]


def real_opacity(stored: float) -> float:
    """The Real_Opacity for the stored opacity `stored`, the logistic function, with no log."""
    return 1 / (1 + math.exp(-stored))


def reference_paint(alpha: float, scales: Sequence[float]) -> float:
    """The Paint of a Gaussian with the Real_Opacity `alpha` and the stored scales `scales`:
    alpha x pi x the product of its two largest Real_Scales, each from math.exp."""
    smaller, larger = sorted(math.exp(scale) for scale in scales)[1:]
    return alpha * math.pi * smaller * larger


def reference_cubes(members: Sequence[Stored], s: CleanSettings) -> list[tuple[float, ...]]:
    """The Cube of each of `members`, a set of at least one Gaussian, as a key for a dict.

    The Trimmed_Box's corners come from np.percentile on each axis, as the glossary defines
    them, and the Extent and the Cube_Edge follow. A Gaussian's key is the tuple of
    (p - low) // edge on the three axes, with Python's // on the float64 values faint_keep gets.
    When the Extent is 0, every Gaussian gets the same key (Req 2.8).

    A draw whose Cube_Edge is above 0 but at most FLOAT32_HALF_STEP is dropped with assume,
    because faint_keep groups those by their centres, which a unit test covers.
    """
    columns = [[g.centre[axis] for g in members] for axis in range(3)]
    t = s.extent_trim_percent
    # The interpolation can underflow, between two tiny values or at a tiny trim percent, which
    # is right, as faint.py allows.
    with np.errstate(under="ignore"):
        low = [float(np.percentile(column, t)) for column in columns]
        high = [float(np.percentile(column, 100 - t)) for column in columns]
    extent = max(top - bottom for bottom, top in zip(low, high))
    if extent <= 0:
        return [()] * len(members)
    edge = extent / s.faint_cube_divisions
    assume(edge > FLOAT32_HALF_STEP)
    return [tuple((p - corner) // edge for p, corner in zip(g.centre, low)) for g in members]


def cube_rule(members: Sequence[Stored], s: CleanSettings) -> list[bool]:
    """The reference for Property 4, which shares no code with faint.py: which of `members` the
    cube rule keeps, worked out in plain Python, one Gaussian at a time, on the whole set.

    The Faint_Gaussians are those whose Real_Opacity is below min_opacity. Each Gaussian goes in
    a dict entry for its Cube, keyed by reference_cubes. A Cube's Cube_Paint sums the Paint of
    every Gaussian in it and its Faint_Paint that of its Faint_Gaussians, each with math.fsum. A
    Cube keeps its Faint_Gaussians when it holds at least faint_min_per_cube of them and its
    Faint_Paint is above faint_min_paint_share times its Cube_Paint, and every other Gaussian is
    kept (Req 2.4, 2.5, 2.9). An empty set keeps nothing (Req 2.10). The reference takes no
    logarithm and shifts no sum, so it's a separate path to the same rule.

    A draw is dropped with assume when a Real_Opacity is within a relative NEAR of min_opacity,
    or when a Cube that holds a Faint_Gaussian has a Faint_Paint within a relative NEAR of
    faint_min_paint_share times its Cube_Paint. A Cube with no Faint_Gaussian has none to keep
    or drop, so it isn't checked: its Faint_Paint of 0 would count as near a share of 0 times
    its Cube_Paint.
    """
    if not members:
        return []
    alphas = [real_opacity(g.opacity) for g in members]
    assume(not any(math.isclose(alpha, s.min_opacity, rel_tol=NEAR) for alpha in alphas))
    faint = [alpha < s.min_opacity for alpha in alphas]
    paints = [reference_paint(alpha, g.scales) for alpha, g in zip(alphas, members)]
    by_cube: dict[tuple[float, ...], list[int]] = {}
    for row, key in enumerate(reference_cubes(members, s)):
        by_cube.setdefault(key, []).append(row)
    keep = [True] * len(members)
    for rows in by_cube.values():
        faint_rows = [row for row in rows if faint[row]]
        if not faint_rows:
            continue
        cube_paint = math.fsum(paints[row] for row in rows)
        faint_paint = math.fsum(paints[row] for row in faint_rows)
        bar = s.faint_min_paint_share * cube_paint
        assume(not math.isclose(faint_paint, bar, rel_tol=NEAR))
        keeps = len(faint_rows) >= s.faint_min_per_cube and faint_paint > bar
        for row in faint_rows:
            keep[row] = keeps
    return keep


@settings(deadline=None)
@given(reference_splats(), valid_settings)
def test_the_faint_filter_keeps_exactly_what_the_cube_rule_keeps(
        data: np.ndarray, values: dict[str, object]) -> None:
    """Feature: clean, Property 4: The Faint_Filter keeps exactly what the cube rule keeps

    **Validates: Requirements 2.4, 2.5, 2.6, 2.7, 2.9, 2.10, 2.20**
    """
    members = as_stored(data)
    expected = cube_rule(members, clean_settings(**values))
    assert faint_answer(members, **values) == expected


# Property 8, in two halves: any finite stored values raise no floating-point error, and a shift
# of every stored scale with a power of 2 on every centre changes no answer.

@settings(deadline=None)
@given(extreme_splats(), valid_settings)
def test_any_finite_stored_values_raise_no_floating_point_error(
        data: np.ndarray, values: dict[str, object]) -> None:
    """Feature: clean, Property 8: Large stored values change nothing and raise no floating-point error

    **Validates: Requirements 2.11**
    """
    # float_errors_raise turns every overflow, invalid value, division by 0 or underflow that
    # faint.py doesn't allow on purpose into a FloatingPointError, which fails the test.
    members = as_stored(data)
    la = log_opacity(np.array([g.opacity for g in members], dtype=np.float64))
    assert np.isfinite(la).all() and (la <= 0).all()
    lp = log_paint(la, np.array([top_two(g.scales) for g in members], dtype=np.float64))
    assert np.isfinite(lp).all()
    faint_answer(members, **values)


# How near its threshold a share test of Property 8's second half may come, as a relative
# distance. Adding c to every stored scale moves each log Paint's rounding by up to about 1e-10,
# so nearer than this the two runs can round to different answers, and the draw is dropped.
SHIFT_NEAR = 1e-6


def near_a_share_threshold(members: Sequence[Stored], s: CleanSettings) -> bool:
    """Whether a share test that faint_keep makes on `members` has a Faint_Paint within a
    relative SHIFT_NEAR of faint_min_paint_share times its Cube_Paint.

    faint_keep makes a share test only when faint_min_paint_share is above 0, and only for a
    Cube that holds a Faint_Gaussian. The Faint_Gaussians come from log_opacity, as faint_keep
    finds them, and the Cubes from reference_cubes. Each Cube's Paints are worked out over its
    largest one, from log Paint, because real_opacity's math.exp would overflow at a stored
    opacity of -3.4e38.
    """
    if s.faint_min_paint_share == 0 or not members:
        return False
    la = log_opacity(np.array([g.opacity for g in members], dtype=np.float64)).tolist()
    bar = -math.inf if s.min_opacity == 0 else math.log(s.min_opacity)
    faint = [s.min_opacity == 1 or alpha < bar for alpha in la]
    lp = [alpha + math.log(math.pi) + top_two(g.scales) for alpha, g in zip(la, members)]
    by_cube: dict[tuple[float, ...], list[int]] = {}
    for row, key in enumerate(reference_cubes(members, s)):
        by_cube.setdefault(key, []).append(row)
    for rows in by_cube.values():
        if not any(faint[row] for row in rows):
            continue
        top = max(lp[row] for row in rows)
        paints = {row: math.exp(lp[row] - top) for row in rows}
        cube_paint = math.fsum(paints.values())
        faint_paint = math.fsum(paints[row] for row in rows if faint[row])
        if math.isclose(faint_paint, s.faint_min_paint_share * cube_paint, rel_tol=SHIFT_NEAR):
            return True
    return False


@settings(deadline=None)
@given(rescaled_splats(), valid_settings)
def test_shifting_every_stored_scale_and_scaling_every_centre_by_a_power_of_2_changes_nothing(
        drawn: Rescaled, values: dict[str, object]) -> None:
    """Feature: clean, Property 8: Large stored values change nothing and raise no floating-point error

    **Validates: Requirements 2.11**
    """
    # In exact arithmetic neither change moves a Cube or a share test: 2**m passes exactly
    # through the percentiles, the Extent and the floor division, and c multiplies every Paint
    # by e**(2c). So a changed answer can only come from an overflow, an underflow or an infinity.
    members = as_stored(drawn.data)
    assume(not near_a_share_threshold(members, clean_settings(**values)))
    assert faint_answer(as_stored(drawn.moved), **values) == faint_answer(members, **values)


# The Isolated_Filter

# The test settings' neighbours and isolation_ratio, 8 and 8.0.
NEIGHBOURS = clean_settings().neighbours
ISOLATION_RATIO = clean_settings().isolation_ratio


def isolated_answer(points: Sequence[Sequence[float]]) -> list[bool]:
    """isolated_keep's answer for the centres `points` at the test settings, as a list, with one
    worker thread, as the test config's max_workers gives. Its answer must be a bool array with
    one entry per Gaussian."""
    keep = isolated_keep(np.array(points, dtype=np.float64), clean_settings(), workers=1)
    assert keep.dtype == np.bool_ and keep.shape == (len(points),)
    return keep.tolist()


def neighbour_distances(points: Sequence[Sequence[float]]) -> list[float]:
    """Each Gaussian's Neighbour_Distance among the centres `points`, worked out by brute force:
    the mean of its NEIGHBOURS smallest Centre_Distances to the others.

    Each row of sorted Centre_Distances starts with a 0 that's dropped, the Gaussian's own or
    that of another Gaussian on its centre, which is the same number. Among others at one
    distance it doesn't matter which count, because each adds that distance to the mean (A10).
    """
    distances = np.sort(centre_distances(np.array(points, dtype=np.float64)), axis=1)
    return distances[:, 1:NEIGHBOURS + 1].mean(axis=1).tolist()


def even_grid(side: int, spacing: float, x0: float = 0.0) -> list[tuple[float, float, float]]:
    """The centres of an even grid of side x side x side Gaussians, `spacing` apart on each axis,
    with its low corner at (x0, 0, 0)."""
    return [(x0 + spacing * x, spacing * y, spacing * z)
            for z in range(side) for y in range(side) for x in range(side)]


def test_a_set_of_neighbours_gaussians_however_far_apart_is_kept_whole_with_no_tree(
        monkeypatch: pytest.MonkeyPatch) -> None:
    sizes: list[int] = []  # the number of points in each tree isolated.py builds, in order

    def spy(points: np.ndarray) -> cKDTree:
        sizes.append(len(points))
        return cKDTree(points)

    monkeypatch.setattr("splatpipe.clean.isolated.cKDTree", spy)
    # neighbours - 1 Gaussians a millimetre apart, and one at the far end of float32's range.
    points = [*((k * 1e-3, 0.0, 0.0) for k in range(NEIGHBOURS - 1)), (3e38, -3e38, 3e38)]
    assert isolated_answer(points) == [True] * NEIGHBOURS
    assert sizes == []
    # With one Gaussian more, the spy sees the one tree. So the spy stands where isolated.py
    # looks cKDTree up, and would have seen a tree above had one been built.
    isolated_answer([*points, (0.0, 1e-3, 0.0)])
    assert sizes == [NEIGHBOURS + 1]


def test_coincident_centres_give_a_neighbour_median_of_0_and_are_kept_with_no_division_by_0(
        ) -> None:
    # neighbours + 1 Gaussians on one spot, so each has neighbours others at distance 0, and one
    # more Gaussian 1 away from them.
    points = [(0.0, 0.0, 0.0)] * (NEIGHBOURS + 1) + [(1.0, 0.0, 0.0)]
    assert neighbour_distances(points) == [0.0] * (NEIGHBOURS + 1) + [1.0]
    # So every Gaussian's Nearest_Neighbours are on the spot, whichever of them the tree picks,
    # and every Neighbour_Median is 0. The Gaussian off the spot is kept too, though its
    # Neighbour_Distance of 1 is above isolation_ratio times its median. float_errors_raise
    # would turn a division by 0 anywhere into an error.
    assert isolated_answer(points) == [True] * (NEIGHBOURS + 2)


def test_one_far_gaussian_next_to_a_tight_cluster_is_dropped() -> None:
    # 27 Gaussians a centimetre apart, and one 0.98 from the nearest of them. Each Gaussian of
    # the cluster has its Nearest_Neighbours in it, and the far one's Neighbour_Distance is
    # about 80 times its neighbours' median, ten times isolation_ratio.
    cluster = even_grid(3, 0.01)
    assert isolated_answer([*cluster, (1.0, 0.01, 0.01)]) == [True] * len(cluster) + [False]


def test_two_even_grids_far_apart_one_100_times_sparser_keep_every_gaussian() -> None:
    # A dense grid of 64 Gaussians 1 apart, and a sparse one of 27 Gaussians 100 apart, 10,000
    # away on x, so each Gaussian's Nearest_Neighbours are in its own grid.
    dense = even_grid(4, 1.0)
    points = [*dense, *even_grid(3, 100.0, x0=10_000.0)]
    # A global distance cutoff measures every Gaussian against one distance, like
    # isolation_ratio times the median Neighbour_Distance of the whole set. The dense grid holds
    # most of the set, so that median is one of the dense grid's, and every Gaussian of the
    # sparse grid has a Neighbour_Distance above the cutoff, so it would drop the whole grid.
    distances = neighbour_distances(points)
    assert min(distances[len(dense):]) > ISOLATION_RATIO * np.median(distances)
    # The Isolated_Filter measures each Gaussian against its own Nearest_Neighbours, which are
    # spaced as it is, so it keeps both grids whole (A5).
    assert isolated_answer(points) == [True] * len(points)


def isolation_rule(centres: Sequence[Sequence[float]], s: CleanSettings) -> list[bool]:
    """The reference for Property 5, which shares no code with isolated.py or neighbours.py:
    which of `centres` the Isolated_Filter's rule keeps, worked out by brute force in plain
    Python, on the whole set it's given.

    A set of neighbours Gaussians or fewer is kept whole (Req 2.14). Otherwise each Gaussian's
    Nearest_Neighbours are the neighbours others nearest to it by math.dist, never itself. Its
    Neighbour_Distance is the mean of its Centre_Distances to them, and its Neighbour_Median is
    the median of their Neighbour_Distances, from statistics.median, each worked out before any
    Gaussian is dropped (Req 2.13). A Gaussian is kept when its Neighbour_Median is 0
    (Req 2.15) or its Neighbour_Distance is at most isolation_ratio times its Neighbour_Median
    (Req 2.12).

    A draw is dropped with assume when a Neighbour_Distance is within a relative NEAR of
    isolation_ratio times a Neighbour_Median above 0. math.dist can differ from the tree's
    distances by a unit in the last place, numpy's mean and median round apart from these, and
    isolated_keep divides where this multiplies, so nearer than that the two can round to
    different answers.
    """
    n, k = len(centres), s.neighbours
    if n <= k:
        return [True] * n
    distance = [[math.dist(p, q) for q in centres] for p in centres]
    # Each Gaussian's Nearest_Neighbours, nearest first, never the Gaussian itself.
    nearest = [sorted((b for b in range(n) if b != a), key=row.__getitem__)[:k]
               for a, row in enumerate(distance)]
    nd = [statistics.fmean(distance[a][b] for b in nearest[a]) for a in range(n)]
    keep = []
    for a in range(n):
        median = statistics.median(nd[b] for b in nearest[a])
        bar = s.isolation_ratio * median
        assume(median == 0 or not math.isclose(nd[a], bar, rel_tol=NEAR))
        keep.append(median == 0 or nd[a] <= bar)
    return keep


@settings(deadline=None)
@given(reference_centres(), valid_settings)
def test_the_isolated_filter_keeps_exactly_what_its_rule_keeps(
        centres: np.ndarray, values: dict[str, object]) -> None:
    """Feature: clean, Property 5: The Isolated_Filter keeps exactly what its rule keeps

    **Validates: Requirements 2.12, 2.13, 2.14, 2.21**
    """
    s = clean_settings(**values)
    expected = isolation_rule(centres.tolist(), s)
    # The step's tree query runs on max(1, max_workers) threads, because scipy refuses 0.
    keep = isolated_keep(centres, s, workers=max(1, s.max_workers))
    assert keep.dtype == np.bool_ and keep.shape == (len(centres),)
    assert keep.tolist() == expected


# The Filter_Order

# The Splat_Data of the Filter_Order tests holds, in row order, BROKEN broken Gaussians, a tight
# cluster of 27 bright Gaussians a centimetre apart, a bright floater 0.98 from the nearest of
# them, and the floater's 7 faint companions, at the other corners of a cube of edge 0.01 that
# has the floater at one corner.
BROKEN = 2
CLUSTER = even_grid(3, 0.01)
FLOATER, *COMPANIONS = even_grid(2, 0.01, x0=1.0)
# The rows that pass every filter: the cluster's.
CLUSTER_ROWS = list(range(BROKEN, BROKEN + len(CLUSTER)))
# The counts run_filters adds for that Splat_Data at the test settings, in order, each with its
# type: the Broken_Filter drops the broken Gaussians, the Faint_Filter the companions and the
# Isolated_Filter the floater.
COUNTS = [("dropped_broken", BROKEN, int), ("dropped_faint", len(COMPANIONS), int),
          ("dropped_isolated", 1, int), ("kept_splats", len(CLUSTER), int)]


def propped_floater() -> np.ndarray:
    """The Filter_Order tests' Splat_Data, made from gaussians, whose plain values make every
    Gaussian drawable and bright, with two changes.

    Each broken Gaussian gets a NaN opacity. Each companion gets a Real_Opacity of 0.001, below
    the test settings' min_opacity of 0.005, so it's a Faint_Gaussian. The 7 companions are fewer
    than faint_min_per_cube, so no Cube holds enough Faint_Gaussians to keep them, and the
    Faint_Filter drops them all, whatever they paint.
    """
    assert len(COMPANIONS) < PER_CUBE
    centres = [*CLUSTER, FLOATER, *COMPANIONS]
    data = gaussians(BROKEN + len(centres))
    for axis, name in enumerate(("x", "y", "z")):
        data[name][BROKEN:] = [centre[axis] for centre in centres]
    data["opacity"][:BROKEN] = np.nan
    data["opacity"][-len(COMPANIONS):] = logit(0.001)
    return data


class Counts:
    """A stand-in for the Step whose add_count run_filters is given. It records each count it's
    given, in order, as (name, value, the value's type)."""

    def __init__(self) -> None:
        self.added: list[tuple[str, object, type]] = []

    def add_count(self, name: str, value: int) -> None:
        self.added.append((name, value, type(value)))


def test_a_gaussian_the_faint_filter_drops_is_no_longer_anyones_neighbour_in_the_isolated_filter(
        ) -> None:
    data = propped_floater()
    # Given everything the Broken_Filter keeps, the companions included, the Isolated_Filter
    # would keep the floater: its Nearest_Neighbours would be the 7 companions and one Gaussian
    # of the cluster, and its Neighbour_Distance would be about its Neighbour_Median.
    drawable = np.arange(BROKEN, len(data))
    assert isolated_answer(positions(data, drawable).tolist()) == [True] * len(drawable)
    # But the Faint_Filter runs first and drops the companions, so the Isolated_Filter gets the
    # cluster and the floater alone. Then the floater's Nearest_Neighbours are in the cluster,
    # and its Neighbour_Distance is about 80 times its Neighbour_Median, ten times
    # isolation_ratio, so the Isolated_Filter drops it. Run the other way round, or each on the
    # whole set, the filters would keep it.
    result = run_filters(data, clean_settings(), Counts().add_count)
    assert result.rows.tolist() == CLUSTER_ROWS
    assert (result.dropped_broken, result.dropped_faint,
            result.dropped_isolated) == (BROKEN, len(COMPANIONS), 1)


def test_the_counts_reach_add_count_in_order_each_as_an_int() -> None:
    counts = Counts()
    run_filters(propped_floater(), clean_settings(), counts.add_count)
    assert counts.added == COUNTS


def test_when_isolated_keep_raises_only_dropped_broken_and_dropped_faint_were_added(
        monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(points: np.ndarray, s: CleanSettings, workers: int) -> NoReturn:
        raise RuntimeError("the stand-in Isolated_Filter failed")

    # Replaced where filters.py looks isolated_keep up.
    monkeypatch.setattr("splatpipe.clean.filters.isolated_keep", fail)
    counts = Counts()
    with pytest.raises(RuntimeError, match="the stand-in Isolated_Filter failed"):
        run_filters(propped_floater(), clean_settings(), counts.add_count)
    # Each count was added as soon as its filter was done, so the two worked out before the
    # failure are there, and the two that need the Isolated_Filter's answer aren't (Req 7.2).
    assert counts.added == COUNTS[:2]


def test_a_max_workers_of_0_runs_with_1_worker_for_the_isolated_filters_tree_query(
        monkeypatch: pytest.MonkeyPatch) -> None:
    workers_given: list[int] = []  # the workers of each isolated_keep call, in order

    def spy(points: np.ndarray, s: CleanSettings, workers: int) -> np.ndarray:
        workers_given.append(workers)
        return isolated_keep(points, s, workers)

    # Replaced where filters.py looks isolated_keep up, with a spy that calls the real one.
    monkeypatch.setattr("splatpipe.clean.filters.isolated_keep", spy)
    result = run_filters(propped_floater(), clean_settings(max_workers=0), Counts().add_count)
    # isolated_keep hands its workers to its tree's query, which scipy refuses at 0. It got the
    # cluster and the floater, more than neighbours Gaussians, so it built its tree and queried
    # it, with 1 worker.
    assert workers_given == [1]
    assert result.rows.tolist() == CLUSTER_ROWS


def filters_in_turn(data: np.ndarray, values: dict[str, object]) -> list[np.ndarray]:
    """The rows of the Splat_Data `data` that each filter keeps when the test calls the three
    itself, one after the other, with the settings `values`: the rows broken_keep keeps, then
    those of them that faint_keep keeps, then those of them that isolated_keep keeps.

    faint_keep gets its inputs from faint_answer, on what broken_keep kept. isolated_keep gets
    the centres of what faint_keep kept from positions, and max(1, max_workers) worker threads,
    as run_filters gives it, because scipy refuses 0.
    """
    s = clean_settings(**values)
    after_broken = np.flatnonzero(broken_keep(data))
    faint_kept = np.array(faint_answer(as_stored(data[after_broken]), **values), dtype=bool)
    after_faint = after_broken[faint_kept]
    isolated_kept = isolated_keep(positions(data, after_faint), s, workers=max(1, s.max_workers))
    return [after_broken, after_faint, after_faint[isolated_kept]]


@settings(deadline=None)
@given(splats(), valid_settings)
def test_the_filters_run_in_order_and_keep_an_ordered_subset(
        data: np.ndarray, values: dict[str, object]) -> None:
    """Feature: clean, Property 1: The filters run in order and keep an ordered subset

    **Validates: Requirements 2.1, 2.17, 2.18**
    """
    before = data.tobytes()
    counts = Counts()
    result = run_filters(data, clean_settings(**values), counts.add_count)
    # The raw bytes are the same, NaN payloads included, so nothing wrote to the Splat_Data and
    # every kept Gaussian holds its values bit for bit.
    assert data.tobytes() == before
    rows = result.rows.tolist()
    assert rows == sorted(set(rows)) and set(rows) <= set(range(len(data)))
    after_broken, after_faint, after_isolated = filters_in_turn(data, values)
    assert rows == after_isolated.tolist()
    # Each filter's drops: the size of the set it was given less the size of the set it kept.
    dropped = [len(data) - len(after_broken), len(after_broken) - len(after_faint),
               len(after_faint) - len(after_isolated)]
    assert [result.dropped_broken, result.dropped_faint, result.dropped_isolated] == dropped
    reported = [(name, value) for name, value, _ in counts.added]
    assert reported == [("dropped_broken", dropped[0]), ("dropped_faint", dropped[1]),
                        ("dropped_isolated", dropped[2]), ("kept_splats", len(after_isolated))]
    assert sum(value for _, value in reported) == len(data)


# Property 6, on dense, opaque grids, with the real config's settings

# The real config.toml, read by path, because conftest.py points CONFIG_PATH at a test config.
REAL_CONFIG = REPO_ROOT / "config.toml"


@functools.cache
def real_settings() -> CleanSettings:
    """The real config.toml's [clean] settings for the Preset called default, checked as the clean
    step checks its own. Every setting the filters read is a Shared_Setting, so every Preset gives
    the same ones."""
    return check_settings(load_config(REAL_CONFIG).step_settings("clean", "default"))


def test_the_real_config_turns_the_faint_filter_off_so_faint_specks_stay() -> None:
    """The real config's min_opacity is 0, so no Gaussian is a Faint_Gaussian and the faint
    filter keeps every one. At 0.005 it removed a faint white wall above a window in a real job,
    whose faint Gaussians were too few in each Cube."""
    assert real_settings().min_opacity == 0.0
    specks = in_a_row(-3.4e38, -1000.0, -40.0, -6.0)  # each alone in its Cube
    points = np.array([g.centre for g in specks], dtype=np.float64).reshape(-1, 3)
    opacity = np.array([g.opacity for g in specks], dtype=np.float64)
    sums = np.array([top_two(g.scales) for g in specks], dtype=np.float64)
    assert faint_keep(points, opacity, sums, real_settings()).tolist() == [True] * 4


@settings(deadline=None)
# Deferred, so the real config is read when this test runs, and a bad one fails this test alone.
@given(st.deferred(lambda: grids(min_opacity=real_settings().min_opacity,
                                 neighbours=real_settings().neighbours)))
def test_a_dense_opaque_grid_loses_nothing(grid: np.ndarray) -> None:
    """Feature: clean, Property 6: A dense, opaque grid loses nothing

    **Validates: Requirements 2.22**
    """
    # No Gaussian of the grid is broken or a Faint_Gaussian, so the Faint_Filter keeps every one
    # without making Cubes, and the Isolated_Filter is given the whole grid. A grid is full of
    # equal Centre_Distances, and no draw is dropped for one, so wherever a tie decides which
    # Gaussians are a Gaussian's Nearest_Neighbours, the tree picks them in its own order (A10).
    counts = Counts()
    result = run_filters(grid, real_settings(), counts.add_count)
    assert result.rows.tolist() == list(range(len(grid)))
    assert counts.added == [("dropped_broken", 0, int), ("dropped_faint", 0, int),
                            ("dropped_isolated", 0, int), ("kept_splats", len(grid), int)]
