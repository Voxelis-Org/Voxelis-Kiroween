"""Tests for splatpipe/train/splats.py: the Start_Splats, the Mean_Exposure, the Exposure_Bake and
the drop of non-finite splats.

The functions are pure, so each unit test passes them tiny arrays written here. start_splats
takes the test config's [train] settings, whose start_opacity of 0.5 and start_scale_factor of
0.1 are the real values. Each refusal is an ExpectedFailure whose message says what its
requirement says and holds no traceback (Req 1.18, 3.15, 14.6).

Property 2 draws its distance arrays and factors with Hypothesis, and finds the value each
all-zero row must hold with its own row-wise minimum.

Property 5 draws float32 colour coefficients, as the Trainer gives them to the bake, with a
float64 Mean_Exposure and basis, and works out each splat's colour itself, in float64, before and
after the bake. Its SH_C0 is 1 / (2√π), worked out here.

The drop tests (Req 8.9, D6) start from splats whose 59 values are all finite and all different,
and place each NaN or infinity through FIELDS, the Splat_Layout's 59 fields mapped to the arrays
that hold them, written here and not taken from the code under test. Each drop test checks that
drop_non_finite keeps exactly the splats it should, in their order and bit for bit, and counts
the others.
"""

import math
from collections.abc import Callable
from dataclasses import fields, replace

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from splatpipe.config import load_config
from splatpipe.errors import ExpectedFailure
from splatpipe.train.settings import TrainSettings, read_settings
from splatpipe.train.splats import (
    Splats,
    bake_exposure,
    choose_points,
    drop_non_finite,
    log_start_distances,
    mean_exposure,
    need_points,
    start_log_scales,
    start_splats,
)

# The shape of one splat's part of each array of Splats.
SHAPES = {
    "means": (3,),
    "log_scales": (3,),
    "quats": (4,),
    "opacity_logits": (),
    "sh0": (3,),
    "sh_rest": (15, 3),
}

# The Splat_Layout's 59 fields in their order (Req 8.3), each with the array of Splats that holds
# it and its place in one splat's part of that array. f_rest_k is channel k // 15's coefficient
# k % 15 (Req 8.4), and sh_rest holds the coefficient first and the channel last.
FIELDS = ([(axis, "means", (i,)) for i, axis in enumerate("xyz")]
          + [(f"f_dc_{i}", "sh0", (i,)) for i in range(3)]
          + [(f"f_rest_{k}", "sh_rest", (k % 15, k // 15)) for k in range(45)]
          + [("opacity", "opacity_logits", ())]
          + [(f"scale_{i}", "log_scales", (i,)) for i in range(3)]
          + [(f"rot_{i}", "quats", (i,)) for i in range(4)])


def train_settings() -> TrainSettings:
    """The test config's [train] settings for the fast Preset, read as the step reads them."""
    return read_settings(load_config().step_settings("train", "fast"), "train", "fast")


def refusal(function: Callable[..., object], *args: object) -> str:
    """The message of the ExpectedFailure that function(*args) raises, once it's checked that the
    message holds no traceback (Req 14.6)."""
    with pytest.raises(ExpectedFailure) as caught:
        function(*args)
    message = str(caught.value)
    assert "Traceback" not in message
    return message


def numbered_points(count: int) -> tuple[np.ndarray, np.ndarray]:
    """`count` Sparse_Points, point i at (i, −i, 2i), and their 8-bit colours, point i's (i, i, i),
    so each point and each colour tells which of the file's points it is."""
    places = np.arange(count)
    points = np.column_stack([places, -places, 2 * places]).astype(np.float64)
    colours = np.column_stack([places, places, places]).astype(np.uint8)
    return points, colours


def numbered_splats(count: int) -> Splats:
    """`count` float32 splats in which splat r's field f, in FIELDS' order, holds 59 r + f, so
    every value is finite and different and tells which splat it belongs to.

    Each array starts as NaNs, so a place that FIELDS missed would show up as a dropped splat.
    """
    arrays = {name: np.full((count, *shape), np.nan, dtype=np.float32)
              for name, shape in SHAPES.items()}
    for f, (_, name, place) in enumerate(FIELDS):
        arrays[name][(slice(None), *place)] = 59 * np.arange(count) + f
    return Splats(**arrays)


def plain(splats: Splats) -> list[tuple[str, tuple[int, ...], bytes]]:
    """Each of the six arrays of `splats` as its dtype, shape and bytes, so two Splats compare
    with ==, bit for bit."""
    arrays = [getattr(splats, field.name) for field in fields(Splats)]
    return [(array.dtype.str, array.shape, array.tobytes()) for array in arrays]


def assert_kept(splats: Splats, rows: list[int]) -> None:
    """drop_non_finite on `splats` keeps exactly the splats at `rows`, in that order, with their
    own values and dtypes, and gives the number of the others as an int."""
    kept, dropped = drop_non_finite(splats)
    assert len(kept) == len(rows)
    assert plain(kept) == plain(Splats(**{field.name: getattr(splats, field.name)[rows]
                                          for field in fields(Splats)}))
    assert (dropped, type(dropped)) == (len(splats) - len(rows), int)


# choose_points (Req 3.11)

def test_choose_points_gives_every_point_when_they_fit_the_budget() -> None:
    points, colours = numbered_points(10)
    chosen, their_colours = choose_points(points, colours, 10, 42)
    assert np.array_equal(chosen, points)
    assert np.array_equal(their_colours, colours)


def test_choose_points_gives_the_same_4_of_10_points_in_file_order_for_the_same_seed() -> None:
    points, colours = numbered_points(10)
    chosen, their_colours = choose_points(points, colours, 4, 42)
    places = chosen[:, 0].astype(int).tolist()  # point i's x is i
    assert len(places) == 4 and places == sorted(set(places))  # 4 different points, in file order
    assert np.array_equal(chosen, points[places])
    assert np.array_equal(their_colours, colours[places])  # each with its own colour
    again, again_colours = choose_points(points, colours, 4, 42)
    assert np.array_equal(again, chosen) and np.array_equal(again_colours, their_colours)


# need_points (Req 1.18)

def test_need_points_refuses_3_points_with_3_neighbours_and_passes_4() -> None:
    message = refusal(need_points, 3, 3)
    # The count, and the fewest the step needs: start_scale_neighbours + 1.
    assert ("too few points to start the splats from: 3, and the train step needs at least 4"
            in message)
    need_points(4, 3)  # passes, with no ExpectedFailure


# log_start_distances (Req 3.15)

def test_log_start_distances_refuses_distances_that_are_all_0() -> None:
    message = refusal(log_start_distances, np.zeros((4, 3)))
    assert "The camera solve's points all sit on top of each other" in message


# start_log_scales: Property 2 (Req 3.12, 3.14, 3.16)

# A distance: an exact 0, the subnormal 5e-324 or a float from 0 up to 1e308, about a third of the
# time each, so a row's largest distance can be 0, the smallest float above 0 or one whose square
# overflows.
DISTANCES = st.just(0.0) | st.just(5e-324) | st.floats(0.0, 1e308)


@st.composite
def distance_arrays(draw: st.DrawFn) -> np.ndarray:
    """An (N, k) array of distances, N from 1 to 40 and k from 1 to 5, with up to 3 planted equal
    rows, each k copies of one distance. So all-zero rows, which A6 gives the smallest
    Start_Distance above 0, turn up often even for k = 5, and so do rows of only 5e-324."""
    n = draw(st.integers(1, 40))
    k = draw(st.integers(1, 5))
    rows = draw(st.lists(st.lists(DISTANCES, min_size=k, max_size=k), min_size=n, max_size=n))
    for row in draw(st.lists(st.integers(0, n - 1), max_size=3)):
        rows[row] = [draw(DISTANCES)] * k
    return np.array(rows, dtype=np.float64)


@settings(max_examples=100, deadline=None)
@given(distances=distance_arrays().filter(lambda distances: (distances > 0).any()),
       factor=st.floats(1e-300, 1e300))
def test_every_start_splat_gets_a_finite_log_scale_the_same_on_all_three_axes(
        distances: np.ndarray, factor: float) -> None:
    """Feature: splat-training, Property 2: Every Start_Splat gets a finite log scale, the same on
    all three axes

    **Validates: Requirements 3.16, 3.12, 3.14**
    """
    scales = start_log_scales(distances, factor)
    assert scales.shape == (len(distances), 3)
    assert np.isfinite(scales).all()  # Req 3.16
    assert (scales == scales[:, :1]).all()  # the same on all three axes (Req 3.12)
    # Each all-zero row holds the smallest value of the other rows (Req 3.14, A6), found here with
    # the oracle's own row-wise minimum.
    zero = (distances == 0).all(axis=1)
    assert (scales[zero] == scales[~zero].min(axis=0)).all()


# start_splats (Req 3.12)

def test_start_splats_for_one_splat_gives_the_values_worked_out_by_hand() -> None:
    settings = train_settings()
    assert (settings.start_opacity, settings.start_scale_factor) == (0.5, 0.1)
    # Distances of 1, 5 and 7 to its 3 nearest points give a Start_Distance of
    # √((1 + 25 + 49) / 3) = 5, so a scale of 0.1 × 5 = 0.5.
    start = start_splats(np.array([[1.0, -2.0, 3.0]]), np.array([[255, 0, 51]], dtype=np.uint8),
                         np.array([[1.0, 5.0, 7.0]]), settings)
    assert [getattr(start, field.name).dtype for field in fields(Splats)] == [np.float32] * 6
    assert start.means.tolist() == [[1.0, -2.0, 3.0]]  # the point's position
    # (colour / 255 − 0.5) / SH_C0, where 1 / SH_C0 is 2√π: 255 gives √π, 0 gives −√π, and 51,
    # a colour of 0.2, gives −0.3 × 2√π.
    root_pi = math.sqrt(math.pi)
    assert start.sh0 == pytest.approx(np.array([[root_pi, -root_pi, -0.6 * root_pi]]), rel=1e-6)
    assert start.sh_rest.shape == (1, 15, 3) and not start.sh_rest.any()
    assert start.opacity_logits.tolist() == [0.0]  # log(0.5 / (1 − 0.5))
    assert start.quats.tolist() == [[1.0, 0.0, 0.0, 0.0]]  # the identity rotation, w first
    assert start.log_scales.tolist() == [[float(np.float32(math.log(0.5)))] * 3]


# mean_exposure (Req 6.4)

def test_mean_exposure_of_two_fixes_is_their_element_wise_mean() -> None:
    # Two Exposure_Fixes [A | b], float32 as the Trainer gives them: the starting [I | 0], and
    # one whose mean with it is exact in binary.
    start = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]
    other = [[2, 0.5, 0, 0.5], [0, 3, -1, -0.25], [0.25, 0, 1, 1]]
    a, b = mean_exposure(np.array([start, other], dtype=np.float32))
    assert a.tolist() == [[1.5, 0.25, 0], [0, 2, -0.5], [0.125, 0, 1]]  # Ā
    assert b.tolist() == [0.25, -0.125, 0.5]  # b̄
    assert (a.dtype, b.dtype) == (np.float64, np.float64)


# bake_exposure: Property 5 (Req 6.5, 6.8)

# SH_C0, the band-0 basis value, worked out here and not taken from the code under test.
BAND_0 = 1 / (2 * math.sqrt(math.pi))

# A colour coefficient, a float32 because the Trainer gives the bake float32 splats.
COEFFICIENTS = st.floats(-10, 10, width=32)


@st.composite
def colour_coefficients(draw: st.DrawFn) -> tuple[np.ndarray, np.ndarray]:
    """The sh0 (N, 3) and sh_rest (N, 15, 3) of N splats, N from 1 to 8, each coefficient from
    −10 to 10."""
    n = draw(st.integers(1, 8))
    return (draw(arrays(np.float32, (n, 3), elements=COEFFICIENTS)),
            draw(arrays(np.float32, (n, 15, 3), elements=COEFFICIENTS)))


def splat_colours(basis: np.ndarray, sh0: np.ndarray, sh_rest: np.ndarray) -> np.ndarray:
    """Each splat's colour (N, 3), in float64 and before the clamp at 0: 0.5 plus basis[0] times
    its sh0 and basis[k + 1] times its sh_rest[:, k] for each of the 15 higher-band coefficients
    k, the coefficients being RGB vectors."""
    sh0 = np.asarray(sh0, dtype=np.float64)
    sh_rest = np.asarray(sh_rest, dtype=np.float64)
    return 0.5 + basis[0] * sh0 + sum(value * sh_rest[:, k] for k, value in enumerate(basis[1:]))


@settings(max_examples=100, deadline=None)
@given(coefficients=colour_coefficients(),
       a=arrays(np.float64, (3, 3), elements=st.floats(-4, 4)),
       b=arrays(np.float64, 3, elements=st.floats(-4, 4)),
       higher_bands=arrays(np.float64, 15, elements=st.floats(-2, 2)))
def test_the_exposure_bake_turns_every_colour_c_into_a_c_plus_b(
        coefficients: tuple[np.ndarray, np.ndarray], a: np.ndarray, b: np.ndarray,
        higher_bands: np.ndarray) -> None:
    """Feature: splat-training, Property 5: The Exposure_Bake turns every colour c into Ā·c + b̄

    **Validates: Requirements 6.8, 6.5**
    """
    sh0, sh_rest = coefficients
    splats = replace(numbered_splats(len(sh0)), sh0=sh0, sh_rest=sh_rest)
    basis = np.concatenate([[BAND_0], higher_bands])  # 16 basis values, the first SH_C0
    baked = bake_exposure(splats, a, b)
    before = splat_colours(basis, sh0, sh_rest)
    after = splat_colours(basis, baked.sh0, baked.sh_rest)
    expected = before @ a.T + b  # Ā·c + b̄ for each splat's colour c, a row of `before`
    # A colour's rounding error grows with the values that go into it, so a colour that cancels
    # to about 0 is compared at an absolute tolerance scaled by the largest value involved.
    largest = max(float(np.abs(value).max()) for value in
                  (sh0, sh_rest, a, b, basis, baked.sh0, baked.sh_rest, before, after, expected))
    assert np.allclose(after, expected, rtol=1e-9, atol=1e-9 * largest)


# drop_non_finite (Req 8.9, D6)

def test_splats_with_no_non_finite_value_all_come_back_unchanged_with_a_count_of_0() -> None:
    splats = numbered_splats(4)
    kept, dropped = drop_non_finite(splats)
    assert kept is splats  # given back with no copy
    assert plain(kept) == plain(numbered_splats(4))  # and not changed in place
    assert (dropped, type(dropped)) == (0, int)


@pytest.mark.parametrize(("name", "place"),
                         [pytest.param(name, place, id=field) for field, name, place in FIELDS])
def test_one_nan_in_any_one_of_the_59_fields_drops_that_splat_alone(
        name: str, place: tuple[int, ...]) -> None:
    splats = numbered_splats(3)
    getattr(splats, name)[(1, *place)] = np.nan
    assert_kept(splats, [0, 2])  # the splats before and after it, in their order


def test_plus_and_minus_infinity_drop_their_splats_and_the_others_keep_their_order() -> None:
    splats = numbered_splats(5)
    splats.quats[1, 0] = np.inf  # rot_0
    splats.sh_rest[3, 14, 2] = -np.inf  # f_rest_44
    assert_kept(splats, [0, 2, 4])


def test_a_baked_float64_value_too_big_for_float32_drops_its_splat() -> None:
    # The Exposure_Bake gives float64 colours. 1e39 is finite in float64, but the float32 that the
    # PLY_Writer writes for it is an infinity (D6).
    splats = numbered_splats(3)
    baked = replace(splats, sh0=splats.sh0.astype(np.float64),
                    sh_rest=splats.sh_rest.astype(np.float64))
    baked.sh0[1, 2] = 1e39
    assert_kept(baked, [0, 2])  # the others keep their float64 colours


def test_when_every_splat_holds_a_non_finite_value_none_is_kept_and_all_are_counted() -> None:
    splats = numbered_splats(3)
    splats.means[0, 2] = np.nan
    splats.opacity_logits[1] = np.inf
    splats.log_scales[2, 0] = -np.inf
    assert_kept(splats, [])  # len() of the kept splats is 0, and the count is 3
