"""Unit tests and the property test for Property 5 of a Prior_View's inputs to the loss:
splatpipe/train/prior_views.py.

Every input is built in the test: a few RGB colours, small Brightness arrays written out by hand,
masks of a few stored pixels and the sizes 2 to 4, 3 to 7 and 4 to 4 (Req 18.1, 18.13). Each
oracle is worked out here and never calls the function it checks: OpenCV's own grey conversion
for the Brightness, central differences with clamped indices for the Brightness_Gradient, and
Fractions for where each training pixel's centre sits among the stored pixels, the weights of
its bilinear sample and the stored pixel under its centre (Req 10.6, 10.7, 10.9).

Property 5 checks depth_weight against the test's own product, r × exp(−falloff × g), worked
out in the gradient's own dtype with each Python value cast to that dtype first, as the Trainer's
float32 holds it (Req 11.5, 11.8, 11.9). A float64 oracle can't meet rtol=1e-12 against a
float32 weight, and a float32 tolerance would need an absolute part for the products that
underflow towards 0. r is the Iteration's value from depth_schedule itself, which
test_train_schedules.py checks, because r is what the Trainer passes in.

The settings are the test config's [train] for the fast Preset, read as the step reads them,
except where a test reads the real config by path.
"""

import math
from dataclasses import replace
from fractions import Fraction
from typing import NamedTuple

import cv2
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from splatpipe.config import REPO_ROOT, load_config
from splatpipe.train.prior_views import (brightness, brightness_gradient, depth_weight,
                                         plain_pixels, prior_pixel_mask, prior_taps, upscale,
                                         upscale_taps)
from splatpipe.train.schedules import depth_schedule
from splatpipe.train.settings import TrainSettings, read_settings

HALF = Fraction(1, 2)
# Read by path, so these tests still read it once CONFIG_PATH points at a test config.
REAL_CONFIG = REPO_ROOT / "config.toml"


def train_settings() -> TrainSettings:
    """The test config's [train] settings for the fast Preset, read as the step reads them."""
    return read_settings(load_config().step_settings("train", "fast"), "train", "fast")


# Brightness and its gradient

# Black, white, the three primaries, mid grey and three mixed colours, as 8-bit RGB.
KNOWN_COLOURS = [(0, 0, 0), (255, 255, 255), (255, 0, 0), (0, 255, 0),
                 (0, 0, 255), (128, 128, 128), (12, 200, 77), (250, 31, 140)]


def test_brightness_is_opencvs_rgb_grey_value_over_255_in_float64() -> None:
    pixels = np.array(KNOWN_COLOURS, dtype=np.uint8).reshape(2, 4, 3)
    grey = brightness(pixels)
    assert grey.dtype == np.float64 and grey.shape == (2, 4)
    expected = cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY).astype(np.float64) / 255  # Req 11.1
    assert np.array_equal(grey, expected)
    assert grey[0, 0] == 0 and grey[0, 1] == 1  # black and white, from 0 to 1
    # OpenCV weighs green 0.587, red 0.299 and blue 0.114, so the order shows the pixels were
    # read as RGB, not BGR.
    red, green, blue = grey[0, 2], grey[0, 3], grey[1, 0]
    assert green > red > blue


def clamped_gradient(grey: np.ndarray) -> np.ndarray:
    """The oracle: each pixel's central differences, with each index clamped to the image so a
    border pixel's missing neighbour is the border pixel itself, and their length as float32."""
    height, width = grey.shape

    def at(y: int, x: int) -> float:
        return float(grey[min(max(y, 0), height - 1), min(max(x, 0), width - 1)])

    def length(y: int, x: int) -> float:
        across = (at(y, x + 1) - at(y, x - 1)) / 2
        down = (at(y + 1, x) - at(y - 1, x)) / 2
        return math.sqrt(across * across + down * down)

    return np.array([[length(y, x) for x in range(width)] for y in range(height)], np.float32)


def test_the_gradient_of_a_ramp_is_its_slope_inside_and_half_of_it_at_the_border() -> None:
    # Brightness (3x + 4y) / 64, so the central differences are 3/64 across and 4/64 down inside,
    # and half of those at the border, where the repeated pixel is the pixel itself (Req 11.2).
    rows, columns = np.mgrid[0:3, 0:4]
    grey = (3 * columns + 4 * rows) / 64
    gradient = brightness_gradient(grey)
    assert gradient.dtype == np.float32 and gradient.shape == (3, 4)
    assert np.array_equal(gradient, clamped_gradient(grey))
    assert gradient[1, 1] == gradient[1, 2] == np.float32(5 / 64)  # √(3² + 4²) / 64
    assert gradient[0, 0] == np.float32(2.5 / 64)  # √(1.5² + 2²) / 64 at the corner


@pytest.mark.parametrize("across", [True, False], ids=["across", "down"])
def test_the_gradient_of_a_step_is_half_the_step_on_each_side_of_it_and_0_at_the_border(
        across: bool) -> None:
    step = np.array([[0.0, 0.0, 1.0, 1.0]] * 2)
    grey = step if across else step.T
    gradient = brightness_gradient(grey)
    expected = np.array([[0.0, 0.5, 0.5, 0.0]] * 2, dtype=np.float32)
    assert gradient.dtype == np.float32
    assert np.array_equal(gradient, expected if across else expected.T)


def test_the_gradient_of_a_1_by_1_image_is_0() -> None:
    gradient = brightness_gradient(np.array([[0.7]]))
    assert gradient.dtype == np.float32
    assert gradient.tolist() == [[0.0]]  # each neighbour is the one pixel, repeated


# Plain_Pixels

@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_a_gradient_at_the_cutoff_isnt_plain_and_one_just_below_it_is(dtype: type) -> None:
    # The cutoff is compared in the gradient's own dtype: numpy 2 and torch both turn the Python
    # float plain_gradient into float32 for a float32 gradient. So for float32 the cutoff is
    # float32(0.02), and just below it is the next float32 down (Req 11.3).
    settings = train_settings()
    cutoff = dtype(settings.plain_gradient)
    below = np.nextafter(cutoff, dtype(0))
    gradient = np.array([[0, below], [cutoff, 1]], dtype=dtype)
    plain = plain_pixels(gradient, settings)
    assert plain.dtype == np.bool_
    assert plain.tolist() == [[True, True], [False, False]]


# The Depth_Weight

class WeightCase(NamedTuple):
    """One drawn case of Property 5: the settings depth_weight and depth_schedule read, the
    Planned_Iterations, an Iteration up to them, and the Brightness_Gradients."""

    planned: int
    iteration: int
    start: float
    end: float
    floor: float
    falloff: float
    plain_gradient: float
    gradient: np.ndarray


@st.composite
def weight_cases(draw: st.DrawFn, dtype: type) -> WeightCase:
    """A case as the design's Property 5 strategy draws it, with the gradients a `dtype` array.

    The Iteration is 0, the last or any in between, a third of the time each, and the falloff is
    0 half the time. Each gradient is drawn from 0 to 10, or about half the time is 0, the
    cutoff or one of its two neighbours, all in `dtype`: numpy turns the Python plain_gradient
    into the gradient's dtype to compare, so that's where the cutoff and its neighbours are.
    """
    planned = draw(st.integers(1, 40_000))
    iteration = draw(st.just(0) | st.just(planned) | st.integers(0, planned))
    plain_gradient = draw(st.floats(1e-6, 1))
    cutoff = dtype(plain_gradient)
    planted = [dtype(0), cutoff, np.nextafter(cutoff, dtype(0)), np.nextafter(cutoff, dtype(np.inf))]
    width = 32 if dtype is np.float32 else 64
    values = draw(st.lists(st.sampled_from(planted) | st.floats(0, 10, width=width),
                           min_size=1, max_size=50))
    return WeightCase(planned=planned, iteration=iteration,
                      start=draw(st.floats(1e-6, 1e3)), end=draw(st.floats(1e-6, 1e3)),
                      floor=draw(st.floats(0, 10)),
                      falloff=draw(st.just(0.0) | st.floats(0, 1e3)),
                      plain_gradient=plain_gradient, gradient=np.array(values, dtype=dtype))


@pytest.mark.parametrize("dtype", [np.float64, np.float32], ids=["float64", "float32"])
@settings(max_examples=100, deadline=None)
@given(data=st.data())
def test_the_depth_weight_is_at_least_the_floor_on_a_plain_pixel_and_the_product_elsewhere(
        dtype: type, data: st.DataObject) -> None:
    """Feature: depth-priors, Property 5: The Depth_Weight never falls below the floor on a
    Plain_Pixel, and is the plain product elsewhere

    **Validates: Requirements 11.8, 11.9, 11.5**

    The product is the test's own, in the gradient's dtype, and compared with atol=0, because
    the default atol of 1e-8 would pass any two products below 1e-8. The floor is compared
    exactly, as the weight's dtype holds it. The real floor, 0.1, is inside the drawn range, and
    the test after this one checks it with the real settings.
    """
    case = data.draw(weight_cases(dtype), label="case")
    chosen = replace(train_settings(), steps=case.planned, plain_gradient=case.plain_gradient,
                     depth_gradient_falloff=case.falloff, depth_weight_start=case.start,
                     depth_weight_end=case.end, depth_weight_floor=case.floor)
    rate = depth_schedule(case.start, case.end, case.planned)[case.iteration]  # r (Req 11.4)
    weight = depth_weight(rate, case.gradient, chosen)

    gradient = case.gradient
    plain = gradient < dtype(case.plain_gradient)  # the Plain_Pixels, in the dtype (Req 11.3)
    product = dtype(rate) * np.exp(dtype(-case.falloff) * gradient)
    floor = dtype(case.floor)
    assert weight.dtype == dtype and weight.shape == gradient.shape
    assert np.all(weight[plain] >= floor)  # Req 11.8
    assert np.all(np.isclose(weight[plain], np.maximum(product[plain], floor),
                             rtol=1e-12, atol=0))  # Req 11.5
    assert np.all(np.isclose(weight[~plain], product[~plain], rtol=1e-12, atol=0))  # Req 11.9


@pytest.mark.parametrize("dtype", [np.float64, np.float32], ids=["float64", "float32"])
@pytest.mark.parametrize("preset", ["fast", "default", "quality"])
def test_with_the_real_settings_a_plain_pixels_weight_never_goes_below_a_tenth(
        dtype: type, preset: str) -> None:
    # The real [train], read by path: a floor of 0.1, a cutoff of 0.02, a falloff of 10 and the
    # Depth_Schedule from 1.0 to 0.01 over the Preset's steps (Req 11.5, 11.8, D2).
    real = read_settings(load_config(REAL_CONFIG).step_settings("train", preset), "train", preset)
    assert real.depth_weight_floor == 0.1
    schedule = depth_schedule(real.depth_weight_start, real.depth_weight_end, real.steps)
    cutoff = dtype(real.plain_gradient)
    gradient = np.array([0, np.nextafter(cutoff, dtype(0)), cutoff, 0.5, 10], dtype)  # 2 plain
    for iteration in (0, real.steps // 2, real.steps):
        assert np.all(depth_weight(schedule[iteration], gradient, real)[:2] >= 0.1), iteration
    # At the last Iteration, r is 0.01, so both plain pixels are lifted to the floor as the
    # dtype holds it, and the textured ones keep their products, 0.01 × exp(−10 × g), below it.
    last = depth_weight(schedule[-1], gradient, real)
    assert np.all(last[:2] == dtype(0.1))
    assert np.all(last[2:] < 0.1)


# The scale-up from the Prior_Size to the Training_Size

def sources(stored: int, target: int) -> list[Fraction]:
    """Where each of `target` training pixels' centres sits, in stored pixels: (X + 1/2) · S / T
    − 1/2, clamped to the first and last stored pixels' centres."""
    return [min(max((x + HALF) * stored / target - HALF, Fraction(0)), Fraction(stored - 1))
            for x in range(target)]


def oracle_taps(stored: int, target: int) -> list[tuple[int, int, Fraction]]:
    """(i0, i1, w1) of each training pixel: the stored pixel at or before its clamped source, the
    next one, kept at the last pixel, and the share of the next one, 0 when i1 is i0."""
    taps = []
    for source in sources(stored, target):
        first = math.floor(source)
        second = min(first + 1, stored - 1)
        taps.append((first, second, source - first if second > first else Fraction(0)))
    return taps


def oracle_nearest(stored: int, target: int) -> list[int]:
    """The stored pixel whose extent holds each training pixel's centre: ⌊(X + 1/2) · S / T⌋."""
    return [math.floor((x + HALF) * stored / target) for x in range(target)]


def assert_taps_match(taps: tuple[np.ndarray, np.ndarray, np.ndarray], stored: int,
                      target: int) -> None:
    """Check one axis's (i0, i1, w1) against the oracle: the indices as int64 and each weight as
    the float32 nearest its Fraction."""
    first, second, weight = taps
    expected = oracle_taps(stored, target)
    assert first.dtype == second.dtype == np.int64 and weight.dtype == np.float32
    assert first.tolist() == [i0 for i0, _, _ in expected]
    assert second.tolist() == [i1 for _, i1, _ in expected]
    assert np.array_equal(weight, np.array([float(w1) for _, _, w1 in expected], np.float32))


@pytest.mark.parametrize(("stored", "target"), [(2, 4), (3, 7), (4, 4)])
def test_the_taps_put_each_training_pixels_centre_where_the_fractions_do(
        stored: int, target: int) -> None:
    # 3 to 7 has weights of k/7, and training pixel 3's centre falls on stored pixel 1's, so its
    # w1 is 0. 4 to 4 takes each stored pixel whole (Req 10.6).
    assert_taps_match(upscale_taps(stored, target), stored, target)


@pytest.mark.parametrize(("stored", "target"), [(2, 4), (3, 7), (4, 4)])
def test_the_taps_clamp_at_the_first_and_the_last_stored_pixel(stored: int, target: int) -> None:
    first, second, weight = upscale_taps(stored, target)
    # The first training pixel's centre is at or before stored pixel 0's, so it takes pixel 0
    # whole, and the last's at or after the last stored pixel's, so it takes that one whole.
    assert (first[0], weight[0]) == (0, 0)
    assert (first[-1], second[-1], weight[-1]) == (stored - 1, stored - 1, 0)


def test_prior_taps_take_the_rows_from_the_heights_and_the_columns_from_the_widths() -> None:
    taps = prior_taps(prior_width=2, prior_height=3, width=4, height=7)
    assert_taps_match(taps.rows, 3, 7)
    assert_taps_match(taps.columns, 2, 4)
    assert taps.nearest_rows.dtype == taps.nearest_columns.dtype == np.int64
    assert taps.nearest_rows.tolist() == oracle_nearest(3, 7)
    assert taps.nearest_columns.tolist() == oracle_nearest(2, 4)


# A Prior of 3 × 2 stored pixels, scaled up to 7 × 4 training pixels: the columns have the 3 to
# 7 weights and a centre on a stored centre, and the rows clamp at both ends.
PRIOR_WIDTH, PRIOR_HEIGHT, WIDTH, HEIGHT = 3, 2, 7, 4


def test_upscale_of_a_constant_gives_the_constant() -> None:
    # Powers of two, because then c·(1 − w) and c·w are exact and (1 − w) + w rounds to 1, so the
    # blend gives c exactly. A value like 1e-7 can come out one float32 step off.
    taps = prior_taps(PRIOR_WIDTH, PRIOR_HEIGHT, WIDTH, HEIGHT)
    flat = upscale(np.full((PRIOR_HEIGHT, PRIOR_WIDTH), 2.0, np.float32), taps)
    assert flat.shape == (HEIGHT, WIDTH)
    assert np.all(flat == 2.0)
    colour = np.array([0.5, 2.0, 4.0], np.float32)
    channels = upscale(np.broadcast_to(colour, (PRIOR_HEIGHT, PRIOR_WIDTH, 3)).copy(), taps)
    assert channels.shape == (HEIGHT, WIDTH, 3)
    assert np.all(channels == colour)


def test_upscale_of_a_ramp_gives_the_ramps_values_at_the_training_pixels_centres() -> None:
    # Bilinear sampling of a ramp gives the ramp itself, at each centre's clamped source. The
    # weights and the blend are float32, so the values, up to 5, are within a few float32 steps.
    taps = prior_taps(PRIOR_WIDTH, PRIOR_HEIGHT, WIDTH, HEIGHT)
    rows, columns = np.mgrid[0:PRIOR_HEIGHT, 0:PRIOR_WIDTH].astype(np.float32)
    down, across = sources(PRIOR_HEIGHT, HEIGHT), sources(PRIOR_WIDTH, WIDTH)

    ramp = upscale(columns + 3 * rows, taps)
    expected = np.array([[float(x + 3 * y) for x in across] for y in down])
    np.testing.assert_allclose(ramp, expected, rtol=0, atol=1e-5)

    # Three ramps, one per channel, so the row and column weights reach every channel alike.
    channels = upscale(np.stack([columns, rows, columns + rows], axis=-1), taps)
    expected = np.array([[[float(x), float(y), float(x + y)] for x in across] for y in down])
    np.testing.assert_allclose(channels, expected, rtol=0, atol=1e-5)


# Prior_Pixels

def oracle_weights(stored: int, target: int) -> list[dict[int, Fraction]]:
    """The stored pixels each training pixel's bilinear sample draws on along one axis, with
    their exact weights, 1 − f and f for the clamped source's fraction f, 0s left out."""
    weights = []
    for source in sources(stored, target):
        first = math.floor(source)
        drawn = {first: 1 - (source - first), first + 1: source - first}
        weights.append({pixel: w for pixel, w in drawn.items() if w > 0})
    return weights


def oracle_prior_pixels(mask: np.ndarray, keep: np.ndarray | None, width: int,
                        height: int) -> np.ndarray:
    """The Prior_Pixels, one training pixel at a time: not a Masked_Pixel, a true mask under its
    centre, and a true mask at every stored pixel whose 2-D weight in its sample is above 0."""
    stored_height, stored_width = mask.shape
    rows, columns = oracle_weights(stored_height, height), oracle_weights(stored_width, width)
    near_rows, near_columns = oracle_nearest(stored_height, height), oracle_nearest(stored_width,
                                                                                    width)
    found = np.zeros((height, width), bool)
    for y in range(height):
        for x in range(width):
            drawn = [(r, c) for r, wr in rows[y].items() for c, wc in columns[x].items()
                     if wr * wc > 0]
            found[y, x] = (all(mask[r, c] for r, c in drawn)
                           and mask[near_rows[y], near_columns[x]]
                           and (keep is None or keep[y, x]))
    return found


@pytest.mark.parametrize("transposed", [False, True], ids=["right edge", "bottom edge"])
def test_at_a_masks_edge_a_pixel_drawing_on_a_stored_pixel_with_no_prior_is_left_out(
        transposed: bool) -> None:
    # 3 × 3 stored pixels to 7 × 7, with stored column 2 outside MoGe's mask, or stored row 2.
    taps = prior_taps(3, 3, 7, 7)
    mask = np.ones((3, 3), bool)
    mask[:, 2] = False
    expected = np.zeros((7, 7), bool)
    expected[:, :4] = True
    # Training column 3's centre falls on stored column 1's, so stored column 2 is its i1 with a
    # weight of 0, and it's kept. Column 4's centre is over stored column 1, but its sample takes
    # 3/7 of stored column 2, so it's left out, and so are 5, with 6/7, and 6, with all of it.
    first, second, weight = taps.columns
    assert (first[3], second[3], weight[3]) == (1, 2, 0)
    assert taps.nearest_columns[4] == 1 and weight[4] == np.float32(3 / 7)

    if transposed:
        mask, expected = mask.T, expected.T
    found = prior_pixel_mask(mask, None, taps)
    assert found.dtype == np.bool_ and found.shape == (7, 7)
    assert np.array_equal(found, expected)


@pytest.mark.parametrize(("prior_width", "prior_height", "width", "height"),
                         [(3, 3, 7, 7), (3, 2, 7, 4), (2, 4, 4, 4)])
def test_a_stored_pixel_with_no_prior_leaves_out_exactly_the_pixels_whose_sample_weighs_it(
        prior_width: int, prior_height: int, width: int, height: int) -> None:
    # Each stored pixel in turn is outside MoGe's mask, the corners included, so every tap of the
    # bilinear sample, the diagonal one too, meets a pixel with no Prior somewhere.
    taps = prior_taps(prior_width, prior_height, width, height)
    for row in range(prior_height):
        for column in range(prior_width):
            mask = np.ones((prior_height, prior_width), bool)
            mask[row, column] = False
            assert np.array_equal(prior_pixel_mask(mask, None, taps),
                                  oracle_prior_pixels(mask, None, width, height)), (row, column)


def test_every_masked_pixel_is_left_out() -> None:
    taps = prior_taps(3, 3, 7, 7)
    keep = np.ones((7, 7), bool)
    keep[0, 0] = keep[3, 3] = keep[2, 5] = keep[6, :] = False  # the Masked_Pixels
    everywhere = np.ones((3, 3), bool)
    assert np.array_equal(prior_pixel_mask(everywhere, None, taps), np.ones((7, 7), bool))
    assert np.array_equal(prior_pixel_mask(everywhere, keep, taps), keep)

    edge = everywhere.copy()
    edge[:, 2] = False  # and with the mask's edge as well, both rules leave pixels out
    found = prior_pixel_mask(edge, keep, taps)
    assert np.array_equal(found, oracle_prior_pixels(edge, keep, 7, 7))
    assert not found[~keep].any()
