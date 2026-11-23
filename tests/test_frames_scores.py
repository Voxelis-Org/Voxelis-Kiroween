"""Tests for splatpipe/frames/scores.py: the frame scores, the Scorer and the frame sizes.

Every frame is a tiny uint8 numpy array built here, so no test needs a video or ffmpeg. The
oracle for the Sharpness is a Laplacian written here in numpy, with OpenCV's default border,
reflect-101, which mirrors a frame about its edge pixels without repeating them. Property 1
draws its frames with Hypothesis, up to 12 × 16, and its oracle adds up the Motion's changes in
Python ints. Property 2 draws Displayed_Shapes up to 8000 × 8000, and its oracle works out each
edge's proportional value exactly with fractions.Fraction.
"""

from fractions import Fraction

import numpy as np
import pytest
from hypothesis import example, given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from splatpipe.frames.scores import Scorer, contrast, motion, pick_size, score_shape, sharpness


def noise(height: int, width: int, seed: int) -> np.ndarray:
    """A height × width frame of seeded noise, with any brightness from 0 to 255."""
    return np.random.default_rng(seed).integers(0, 256, size=(height, width), dtype=np.uint8)


def oracle_laplacian(frame: np.ndarray) -> np.ndarray:
    """The 3 × 3 Laplacian of `frame`: each pixel's four neighbours minus 4 times the pixel.

    The border is reflect-101 (dcb|abcd|cba), which is numpy's "reflect" pad.
    """
    padded = np.pad(frame.astype(np.int64), 1, mode="reflect")
    return (padded[:-2, 1:-1] + padded[2:, 1:-1] + padded[1:-1, :-2] + padded[1:-1, 2:]
            - 4 * padded[1:-1, 1:-1])


def columns(left: int, right: int) -> np.ndarray:
    """A 2 × 2 frame whose left column is `left` and right column is `right`.

    Its scores are easy to work out by hand. Its Contrast is |left − right| / 2. With reflect-101
    borders, each pixel's left and right neighbours are both the other column's pixel, and the
    pixels above and below it match it. So its Laplacian is ±2 × (right − left), and its
    Sharpness is 4 × (right − left)².
    """
    return np.array([[left, right], [left, right]], dtype=np.uint8)


# Sharpness (Req 3.2)

@pytest.mark.parametrize(("height", "width"), [(2, 3), (5, 7), (12, 16)])
def test_sharpness_is_the_variance_of_a_numpy_laplacian_with_reflect_101_borders(
        height: int, width: int) -> None:
    frame = noise(height, width, seed=7)
    expected = float(np.var(oracle_laplacian(frame)))
    assert sharpness(frame) == pytest.approx(expected, rel=1e-12)


def test_sharpness_is_0_on_a_flat_frame() -> None:
    assert sharpness(np.full((6, 8), 128, dtype=np.uint8)) == 0.0


# Motion of the first frame (Req 3.4)

def test_motion_with_no_frame_before_is_0() -> None:
    assert motion(None, noise(4, 6, seed=1)) == 0.0


def test_a_scorers_first_frame_gets_a_motion_of_0() -> None:
    scorer = Scorer()
    scorer.add(0.5, noise(4, 6, seed=2))
    assert scorer.result().motion.tolist() == [0.0]


# The Scorer (Req 3.6)

def test_scorer_result_keeps_each_frames_time_and_scores_in_the_order_they_came() -> None:
    # Each Motion is the mean absolute change from the frame before: the mean of |0 − 100| and
    # |60 − 100| is 70, and the mean of |20 − 0| and |220 − 60| is 90.
    scorer = Scorer()
    for time, frame in [(0.0, columns(100, 100)), (0.04, columns(0, 60)),
                        (0.1, columns(20, 220))]:
        scorer.add(time, frame)
    scores = scorer.result()
    assert scores.times.tolist() == [0.0, 0.04, 0.1]
    assert scores.sharpness.tolist() == [0.0, 14400.0, 160000.0]
    assert scores.motion.tolist() == [0.0, 70.0, 90.0]
    assert scores.contrast.tolist() == [0.0, 30.0, 100.0]
    arrays = (scores.times, scores.sharpness, scores.motion, scores.contrast)
    assert [array.dtype for array in arrays] == [np.float64] * 4


# The frame sizes (Req 3.1, 8.1)

def test_score_shape_is_score_width_wide_and_keeps_the_shape() -> None:
    assert score_shape(1920, 1080, 320) == (320, 180)


def test_a_video_narrower_than_score_width_is_scaled_up_for_scoring() -> None:
    assert score_shape(64, 48, 320) == (320, 240)


@pytest.mark.parametrize(("width", "height", "size"), [
    pytest.param(3840, 2160, (1600, 900), id="landscape 4K"),
    pytest.param(2160, 3840, (900, 1600), id="portrait 4K"),
    pytest.param(64, 48, (64, 48), id="smaller than long_edge"),
])
def test_pick_size_brings_the_long_edge_to_long_edge_and_never_scales_a_video_up(
        width: int, height: int, size: tuple[int, int]) -> None:
    assert pick_size(width, height, 1600) == size


# Property tests

def end_frames(height: int, width: int) -> list[np.ndarray]:
    """The height × width frames at the ends of the scores' ranges: all 0, all 255 and, when the
    pixel count is even, half 0 and half 255, whose Contrast is exactly 127.5."""
    frames = [np.zeros((height, width), dtype=np.uint8),
              np.full((height, width), 255, dtype=np.uint8)]
    if height * width % 2 == 0:
        halves = np.repeat(np.array([0, 255], dtype=np.uint8), height * width // 2)
        frames.append(halves.reshape(height, width))
    return frames


@st.composite
def frame_pairs(draw: st.DrawFn) -> tuple[np.ndarray, np.ndarray]:
    """Two uint8 frames of one shape up to 12 × 16, each drawn by arrays or from end_frames."""
    height, width = draw(st.integers(1, 12)), draw(st.integers(1, 16))
    frames = arrays(np.uint8, (height, width)) | st.sampled_from(end_frames(height, width))
    return draw(frames), draw(frames)


def oracle_motion(previous: np.ndarray, frame: np.ndarray) -> float:
    """The mean absolute change from `previous` to `frame`, with every change a Python int."""
    pixels = zip(previous.ravel().tolist(), frame.ravel().tolist(), strict=True)
    changes = [abs(now - before) for before, now in pixels]
    return sum(changes) / len(changes)


@given(frame_pairs())
def test_the_scores_stay_in_range(pair: tuple[np.ndarray, np.ndarray]) -> None:
    """Feature: frames, Property 1: Scores stay in range

    **Validates: Requirements 3.3, 3.4, 3.5, 3.7**
    """
    previous, frame = pair
    frame_motion = motion(previous, frame)
    assert 0 <= frame_motion <= 255
    assert frame_motion == oracle_motion(previous, frame)
    assert motion(None, frame) == 0
    assert sharpness(frame) >= 0
    frame_contrast = contrast(frame)
    assert 0 <= frame_contrast <= 127.5
    assert abs(frame_contrast - float(np.std(frame))) <= 1e-9


HALF = Fraction(1, 2)


def assert_scaled_edge(edge: int, exact: Fraction) -> None:
    """Check an edge worked out from its proportional value `exact`. It's at least 1 and within
    0.5 of `exact`, except when `exact` is below 0.5: then max(1, …) lifts it to exactly 1."""
    assert edge >= 1
    if exact < HALF:
        assert edge == 1
    else:
        assert abs(edge - exact) <= HALF


# Two thin videos, landscape and portrait, whose short edge scales to below 0.5 px, so the
# oracle's exactly-1 case runs every time.
@example(width=8000, height=1, score_width=1, long_edge=1)
@example(width=1, height=8000, score_width=1, long_edge=1)
@given(width=st.integers(1, 8000), height=st.integers(1, 8000),
       score_width=st.integers(1, 4000), long_edge=st.integers(1, 4000))
def test_frame_sizes_keep_the_shape_and_never_grow_a_pick(
        width: int, height: int, score_width: int, long_edge: int) -> None:
    """Feature: frames, Property 2: Frame sizes keep the shape and never grow a Pick

    **Validates: Requirements 3.1, 8.1**
    """
    score_w, score_h = score_shape(width, height, score_width)
    assert score_w == score_width
    assert_scaled_edge(score_h, Fraction(height * score_width, width))

    pick_w, pick_h = pick_size(width, height, long_edge)
    video_long = max(width, height)
    pick_long = min(long_edge, video_long)
    assert max(pick_w, pick_h) == pick_long
    assert pick_w <= width and pick_h <= height
    # Both edges scale by pick_long / video_long, the long one to exactly pick_long.
    assert_scaled_edge(pick_w, Fraction(width * pick_long, video_long))
    assert_scaled_edge(pick_h, Fraction(height * pick_long, video_long))
    if video_long <= long_edge:
        assert (pick_w, pick_h) == (width, height)
