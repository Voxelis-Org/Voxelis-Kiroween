"""Tests for the clean step's filters, in splatpipe/clean/, on Splat_Data and centres that the
tests build as arrays themselves (Req 10.1).

Each filter has a part of its own below, and so does the neighbour search that the
Isolated_Filter uses. A part holds its unit tests first and its property tests after them:

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

The autouse fixture float_errors_raise runs every test here inside np.errstate(all="raise"), so
an overflow, an invalid value, a division by 0 or an underflow anywhere in a filter fails the
test. Only the underflow a filter allows on purpose, inside its own np.errstate(under="ignore"),
passes (Req 2.11).
"""

import math
from collections.abc import Iterator

import numpy as np
import pytest
from hypothesis import given, settings
from scipy.spatial import cKDTree

from splatpipe.clean.broken import broken_keep
from splatpipe.clean.neighbours import nearest_others, positions

from splats import centre_sets, gaussians, splats

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
    summed in that order, which is how the tree works it out, so the two agree exactly. math.dist
    can differ from both by one unit in the last place."""
    dx, dy, dz = (float(p) - float(q) for p, q in zip(a, b, strict=True))
    return math.sqrt(dx * dx + dy * dy + dz * dz)


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
        assert distances == [centre_distance(centre, tree.data[other]) for other in others]
        assert distances == sorted(centre_distance(centre, tree.data[other])
                                   for other in range(tree.n) if other != point)[:k]


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
        assert distances == [centre_distance(centres[point], centres[other]) for other in others]
