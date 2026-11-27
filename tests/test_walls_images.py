"""Unit tests for the Walls_Step's score images and window sums: splatpipe/walls/images.py
(Req 9.1, 9.4, 10.5, A4).

Every image is a few dozen to a few hundred pixels, made here from a seeded generator. Each oracle
is written in the test: OpenCV's own calls, a plain Python loop or the closed form, never the
function it checks.
"""
import math

import cv2
import numpy as np
import pytest

from splatpipe.walls.images import (band_weights, bands, high_pass, luma, warp_image,
                                    window_centres, window_ncc, window_sums, window_variance)


def _noise(seed: int, height: int, width: int) -> np.ndarray:
    """A seeded (height, width) float64 noise image from 0 to 255."""
    return np.random.default_rng(seed).uniform(0.0, 255.0, size=(height, width))


def _identity_maps(height: int, width: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Maps that send each pixel to itself, and an `inside` that keeps every pixel."""
    grid_y, grid_x = np.mgrid[0:height, 0:width].astype(np.float64)
    return grid_x, grid_y, np.ones((height, width), dtype=bool)


# --- luma ---------------------------------------------------------------------------------------

# Known BGR colours: black, white, pure blue, green and red, and two mixes.
COLOURS = [(0, 0, 0), (255, 255, 255), (255, 0, 0), (0, 255, 0), (0, 0, 255), (10, 200, 77),
           (133, 7, 250)]


def test_luma_equals_opencv_grey_of_the_float_colour() -> None:
    # luma turns the uint8 Frame into float32 before cv2.cvtColor, so the grey keeps its fractions
    # of a level. cvtColor of the uint8 colour rounds the grey to a whole level, which can be more
    # than 0.5 off the unrounded one, so the oracle is cvtColor of the same colour in float32.
    frame = np.array([COLOURS], dtype=np.uint8)  # (1, 7, 3) BGR
    expected = cv2.cvtColor(frame.astype(np.float32), cv2.COLOR_BGR2GRAY).astype(np.float64)
    got = luma(frame)
    assert got.dtype == np.float64
    assert got.shape == (1, len(COLOURS))
    np.testing.assert_array_equal(got, expected)


def test_luma_is_the_bt601_weighting_of_the_colour() -> None:
    # OpenCV's BGR-to-grey is Y = 0.299 R + 0.587 G + 0.114 B, here to float32's precision.
    frame = np.array([COLOURS], dtype=np.uint8)
    expected = [0.299 * r + 0.587 * g + 0.114 * b for b, g, r in COLOURS]
    np.testing.assert_allclose(luma(frame)[0], expected, atol=1e-3)


def test_luma_refuses_an_image_that_isnt_bgr() -> None:
    with pytest.raises(ValueError, match="BGR"):
        luma(np.zeros((4, 4), dtype=np.uint8))


# --- high_pass and bands ------------------------------------------------------------------------

def test_high_pass_of_a_constant_is_zero() -> None:
    grey = np.full((20, 24), 93.25)
    np.testing.assert_allclose(high_pass(grey, 2.0), 0.0, atol=1e-9)


def test_high_pass_of_a_step_is_minus_then_plus_around_it() -> None:
    # A dark left half (0) and a bright right half (100), the step between columns 11 and 12.
    grey = np.zeros((16, 24))
    grey[:, 12:] = 100.0
    result = high_pass(grey, 2.0)
    row = result[8]
    # Just left of the step the blur is brighter than the pixel, just right of it darker.
    assert row[11] < -10.0
    assert row[12] > 10.0
    assert np.all(row[:12] <= 1e-9)
    assert np.all(row[12:] >= -1e-9)
    # Far from the step the image is flat and the High_Pass is 0.
    np.testing.assert_allclose(row[:2], 0.0, atol=1e-6)
    np.testing.assert_allclose(row[-2:], 0.0, atol=1e-6)
    # The step is the same down every row.
    np.testing.assert_allclose(result, np.broadcast_to(row, result.shape), atol=1e-9)


def test_bands_add_up_to_the_luma_minus_its_last_blur() -> None:
    grey = _noise(1, 40, 48)
    result = bands(grey, [0, 1, 2, 4, 8])
    assert len(result) == 4
    assert all(band.shape == grey.shape and band.dtype == np.float64 for band in result)
    expected = grey - cv2.GaussianBlur(grey, (0, 0), 8.0)
    np.testing.assert_allclose(result[0] + result[1] + result[2] + result[3], expected,
                               atol=1e-9)
    # The first Band is the Luma minus its blur at sigma 1, and the second blur(1) − blur(2).
    np.testing.assert_allclose(result[0], grey - cv2.GaussianBlur(grey, (0, 0), 1.0), atol=1e-12)
    np.testing.assert_allclose(result[1], cv2.GaussianBlur(grey, (0, 0), 1.0)
                               - cv2.GaussianBlur(grey, (0, 0), 2.0), atol=1e-12)


# --- window_sums, window_variance ---------------------------------------------------------------

def _loop_sums(image: np.ndarray, xs: np.ndarray, ys: np.ndarray, size: int) -> list:
    """Each window's sum by a plain Python loop over its pixels."""
    half = size // 2
    sums = []
    for x, y in zip(xs.tolist(), ys.tolist()):
        total = 0
        for row in range(y - half, y + half + 1):
            for col in range(x - half, x + half + 1):
                total += image[row, col].item()
        sums.append(total)
    return sums


def test_window_sums_of_a_float_image_equal_a_loop() -> None:
    image = _noise(2, 9, 11)
    xs, ys = window_centres(11, 9, 3, 2)
    assert xs.size == 20  # x in 1, 3, 5, 7, 9 and y in 1, 3, 5, 7
    got = window_sums(image, (xs, ys), 3)
    assert got.dtype == np.float64
    np.testing.assert_allclose(got, _loop_sums(image, xs, ys, 3), rtol=1e-12)


def test_window_sums_of_a_bool_image_are_exact_counts() -> None:
    mask = np.random.default_rng(3).random((10, 12)) < 0.4
    xs, ys = window_centres(12, 10, 5, 1)
    got = window_sums(mask, (xs, ys), 5)
    assert got.dtype == np.int64
    assert got.tolist() == _loop_sums(mask.astype(np.int64), xs, ys, 5)


def test_window_sums_refuses_an_even_window() -> None:
    with pytest.raises(ValueError, match="odd"):
        window_sums(np.zeros((8, 8)), (np.array([3]), np.array([3])), 4)


@pytest.mark.parametrize("centre", [(0, 3), (3, 0), (7, 3), (3, 7)])
def test_window_sums_refuses_a_window_leaving_the_image(centre: tuple[int, int]) -> None:
    # A 3-pixel window on an 8 × 8 image fits only centres from 1 to 6.
    with pytest.raises(ValueError, match="leaves"):
        window_sums(np.zeros((8, 8)), (np.array([centre[0]]), np.array([centre[1]])), 3)


def test_window_variance_is_the_windows_variance_and_zero_when_flat() -> None:
    image = _noise(4, 9, 9)
    xs, ys = np.array([2, 6]), np.array([2, 6])
    got = window_variance(image, (xs, ys), 5)
    expected = [np.var(image[y - 2:y + 3, x - 2:x + 3]) for x, y in zip(xs, ys)]
    np.testing.assert_allclose(got, expected, rtol=1e-9)
    flat = window_variance(np.full((9, 9), 171.3), (xs, ys), 5)
    assert flat.tolist() == [0.0, 0.0]


# --- warp_image ---------------------------------------------------------------------------------

def test_warp_image_samples_the_source_at_the_maps() -> None:
    source = _noise(5, 6, 8)
    map_x, map_y, inside = _identity_maps(6, 8)
    np.testing.assert_array_equal(warp_image(source, map_x, map_y, inside), source)
    # One pixel to the right: each pixel reads its right-hand neighbour, the last column 0.
    shifted = warp_image(source, map_x + 1.0, map_y, inside)
    np.testing.assert_allclose(shifted[:, :-1], source[:, 1:], atol=1e-12)
    # Half a pixel down is the mean of the two rows, bilinearly.
    half = warp_image(source, map_x, map_y + 0.5, inside)
    np.testing.assert_allclose(half[:-1], (source[:-1] + source[1:]) / 2.0, atol=1e-9)


def test_warp_image_reads_zero_where_not_inside_or_not_finite() -> None:
    source = np.full((5, 5), 40.0)
    map_x, map_y, inside = _identity_maps(5, 5)
    inside[2, 3] = False
    map_x[0, 0] = np.nan
    warped = warp_image(source, map_x, map_y, inside)
    assert warped[2, 3] == 0.0
    assert warped[0, 0] == 0.0
    assert warped[4, 4] == 40.0


# --- window_ncc ---------------------------------------------------------------------------------

def _ncc(prior: np.ndarray, source: np.ndarray, size: int, inside: np.ndarray | None = None,
         flat_variance: float = 0.0) -> np.ndarray:
    """window_ncc of `prior` and `source` with identity maps, over every window of the image."""
    height, width = prior.shape
    map_x, map_y, all_inside = _identity_maps(height, width)
    centres = window_centres(width, height, size, size)
    return window_ncc(prior, source, map_x, map_y, all_inside if inside is None else inside,
                      centres, size, flat_variance)


def test_window_ncc_of_a_window_and_itself_is_one() -> None:
    prior = _noise(6, 14, 21)
    np.testing.assert_allclose(_ncc(prior, prior.copy(), 7), 1.0, atol=1e-9)
    # A brighter, higher-contrast copy still correlates fully: NCC ignores gain and offset.
    np.testing.assert_allclose(_ncc(prior, 3.0 * prior + 20.0, 7), 1.0, atol=1e-9)


def test_window_ncc_of_a_window_and_its_negative_is_minus_one() -> None:
    prior = _noise(7, 14, 21)
    np.testing.assert_allclose(_ncc(prior, -prior, 7), -1.0, atol=1e-9)


def test_window_ncc_of_unrelated_noise_is_near_zero() -> None:
    # Over 225 pixels the NCC of two independent noises has a spread of about 1 / √225 ≈ 0.07.
    scores = _ncc(_noise(8, 30, 45), _noise(9, 30, 45), 15)
    assert scores.size == 6
    assert np.all(np.abs(scores) < 0.3)
    assert abs(float(np.mean(scores))) < 0.15


def test_window_ncc_matches_the_formula_on_correlated_noise() -> None:
    rng = np.random.default_rng(10)
    prior = rng.normal(size=(7, 7))
    source = 0.6 * prior + 0.8 * rng.normal(size=(7, 7))
    got = _ncc(prior, source, 7)
    a, b = prior - prior.mean(), source - source.mean()
    expected = float(np.sum(a * b) / math.sqrt(float(np.sum(a * a)) * float(np.sum(b * b))))
    np.testing.assert_allclose(got, [expected], atol=1e-12)


def test_window_ncc_of_a_flat_window_is_zero() -> None:
    noise = _noise(11, 7, 14)
    flat = np.full((7, 7), 128.7)
    prior = np.hstack([flat, noise[:, 7:]])
    source = noise.copy()
    # The left window is flat in the Prior_Frame and scores 0, the right one is the same noise.
    np.testing.assert_allclose(_ncc(prior, source, 7), [0.0, 1.0], atol=1e-9)
    # Flat in the Source_Frame, and flat in both, score 0 too.
    assert _ncc(noise, np.full((7, 14), 3.0), 7).tolist() == [0.0, 0.0]
    assert _ncc(np.full((7, 7), 5.0), np.full((7, 7), 5.0), 7).tolist() == [0.0]


def test_window_ncc_is_zero_below_flat_variance() -> None:
    # A faint copy has a per-pixel variance far below 1, so a flat_variance of 1 scores it 0.
    prior = _noise(12, 7, 7) / 1000.0
    assert float(np.var(prior)) < 0.01
    assert _ncc(prior, prior.copy(), 7, flat_variance=1.0).tolist() == [0.0]
    np.testing.assert_allclose(_ncc(prior, prior.copy(), 7, flat_variance=0.0), [1.0], atol=1e-9)


def test_window_ncc_is_zero_where_a_pixel_isnt_inside() -> None:
    prior = _noise(13, 7, 14)
    inside = np.ones((7, 14), dtype=bool)
    inside[3, 10] = False  # one pixel of the right window
    np.testing.assert_allclose(_ncc(prior, prior.copy(), 7, inside=inside), [1.0, 0.0],
                               atol=1e-9)


# --- band_weights -------------------------------------------------------------------------------

def _closed_form(rho: float, var: float) -> float:
    """w with w² × var = ρ / (1 − ρ²)."""
    return math.sqrt(rho / ((1.0 - rho * rho) * var))


def test_band_weights_equal_the_closed_form() -> None:
    rhos, variances = [0.5, 0.2, 0.75, 0.05], [4.0, 9.0, 0.3, 120.0]
    got = band_weights(rhos, variances, 0.95)
    assert got.dtype == np.float64 and got.shape == (4,)
    np.testing.assert_allclose(got, [_closed_form(r, v) for r, v in zip(rhos, variances)],
                               rtol=1e-12)
    # w² × var = ρ / (1 − ρ²), the A4 relation itself.
    np.testing.assert_allclose(got * got * np.array(variances),
                               [r / (1.0 - r * r) for r in rhos], rtol=1e-12)


def test_band_weights_cap_the_coherence() -> None:
    got = band_weights([0.99, 1.0, 0.9, 0.5], [2.0, 2.0, 2.0, 2.0], 0.9)
    capped = _closed_form(0.9, 2.0)
    np.testing.assert_allclose(got, [capped, capped, capped, _closed_form(0.5, 2.0)],
                               rtol=1e-12)


def test_band_weights_are_zero_without_coherence_or_variance() -> None:
    got = band_weights([0.0, -0.4, 0.6, 0.6, float("nan"), 0.6],
                       [5.0, 5.0, 0.0, -1.0, 5.0, 5.0], 0.95)
    assert got[:5].tolist() == [0.0, 0.0, 0.0, 0.0, 0.0]
    np.testing.assert_allclose(got[5], _closed_form(0.6, 5.0), rtol=1e-12)


def test_band_weights_refuse_mismatched_lists() -> None:
    with pytest.raises(ValueError, match="one coherence and one variance"):
        band_weights([0.5, 0.5], [1.0], 0.95)
