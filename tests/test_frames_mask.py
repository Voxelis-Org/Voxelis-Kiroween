"""Tests for splatpipe/frames/mask.py: the Sample_Buffer, the choice of the Mask_Samples, the
Pixel_Change, the Stuck_Pixels, the choice of the Mask and the Mask image. The memory test also
runs scores.py's Scorer, because the Score_Pass gives each frame to both.

Every frame is a tiny uint8 numpy array built here, so no test needs a video or ffmpeg. The
Sample_Buffer tests feed it frames that are each a new array, and check that the Mask_Samples are
the very arrays it was fed.

The memory test feeds 20,000 new 4 × 4 frames, one at a time, to a Scorer and then to a
SampleBuffer with mask_samples = 3, as the Score_Pass's on_frame does. The test keeps only a
weakref to each frame. CPython frees an array as soon as nothing refers to it, so a frame's
weakref is dead once the Scorer and the buffer have both let it go, with no need for the garbage
collector. A dead frame never comes back, so after each frame the test checks only the frames it
hasn't yet seen die. That keeps each check to a handful of frames.

The Mask tests use mask_ratio, mask_min_coverage and mask_max_coverage as the real config has
them. Their samples are moving noise, some with a planted region that never changes. Each pixel
of the noise is dark, from 0 to 63, in the even samples and bright, from 192 to 255, in the odd
ones, so it changes by at least 129 from one sample to the next, and its Pixel_Change is at least
129. While a planted region holds fewer than half the pixels, the median Pixel_Change is a noise
pixel's, from 129 to 255, so mask_ratio times the median is from 25.8 to 51. So every planted
pixel, whose Pixel_Change is 0, is stuck, and no pixel of the noise is, whatever the seed.

Property 10 feeds the Sample_Buffer streams of 1 to 2,000 frames of 1 × 1, with a mask_samples
from 2 to 40. Every frame has the same pixel, so only identity tells them apart. After every
frame, its oracle works out the stride from the number of frames fed so far, as the smallest
power of 2, k, with ceil(N / k) at most 2 × mask_samples, and never reads the buffer to do it.
The Kept_Range often runs from frame 0 to the last frame, so it often holds every buffered frame.
That's more than mask_samples once the buffer has thinned, so samples_in has to spread them.

Property 11 draws 0 to 8 Mask_Samples of one shape up to 12 × 16 with Hypothesis's arrays. About
half the time a drawn region holds one drawn brightness in every sample, so its pixels never
change. mask_ratio is any float from 0 to 1, often 0, 0.25, 0.5 or 1, so a Pixel_Change often
lands exactly on mask_ratio times the median. mask_min_coverage and mask_max_coverage are two
draws, sorted, each any float from 0 to 1 or a share of the frame's pixels, so the Mask_Coverage
often lands exactly on one of them. The oracle adds up each pixel's changes as Python ints and
divides once by the number of pairs, which gives the same float as pixel_change. It finds the
median by sorting, and the mean of the two middle values is the same float numpy's median gives.
So its rule and its coverage are the definitions' comparisons on the same floats decide_mask has.

Property 12 draws stuck arrays from 1 × 1 to 12 × 16 with Hypothesis's arrays and Pick_Sizes from
1 × 1 to 40 × 40, so the Mask image is sometimes larger than the Score_Frame and sometimes smaller.
Its oracle measures each position as a share of the picture, which the Score_Frame and the Pick
both show whole, and finds with Fraction arithmetic the Score pixel whose area holds each output
pixel's centre. A centre can lie exactly on the line between two Score pixels, as when a
Score_Frame 2 pixels wide gives a Mask 1 pixel wide. A Score pixel's area holds its left and top
edges but not its right and bottom ones, as the design's formula rounds down, so the centre then
belongs to the pixel right of the line or below it.
"""

import weakref
from dataclasses import dataclass
from fractions import Fraction
from itertools import pairwise

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from splatpipe.frames.mask import (MaskChoice, SampleBuffer, decide_mask, mask_image,
                                   pixel_change, spread, stuck_pixels)
from splatpipe.frames.scores import Scorer

# mask_ratio, mask_min_coverage and mask_max_coverage, as the real config has them.
MASK_RATIO = 0.2
MIN_COVERAGE = 0.002
MAX_COVERAGE = 0.25
# How many Mask_Samples the Mask tests give decide_mask.
SAMPLES = 6
# A planted region's brightness, the same in every sample.
PLANTED = 128


def noise(height: int, width: int, seed: int) -> np.ndarray:
    """A height × width frame of seeded noise, with any brightness from 0 to 255."""
    return np.random.default_rng(seed).integers(0, 256, size=(height, width), dtype=np.uint8)


def moving_noise(count: int, height: int, width: int, seed: int = 0) -> list[np.ndarray]:
    """`count` height × width samples of noise in which every pixel changes by at least 129 from
    one sample to the next: it's dark, from 0 to 63, in the even samples and bright, from 192 to
    255, in the odd ones."""
    rng = np.random.default_rng(seed)
    samples = []
    for i in range(count):
        low = 192 if i % 2 else 0
        samples.append(rng.integers(low, low + 64, size=(height, width), dtype=np.uint8))
    return samples


def planted_on_noise(region: np.ndarray, count: int = SAMPLES) -> list[np.ndarray]:
    """`count` samples of moving_noise at the shape of the bool array `region`, with the
    region's pixels at PLANTED in every sample, so they never change."""
    samples = moving_noise(count, *region.shape)
    for sample in samples:
        sample[region] = PLANTED
    return samples


def corner(height: int, width: int, rows: int, columns: int) -> np.ndarray:
    """A height × width bool array that's True in its bottom-right rows × columns, where a
    watermark often sits."""
    region = np.zeros((height, width), dtype=bool)
    region[height - rows:, width - columns:] = True
    return region


def decide(samples: list[np.ndarray]) -> MaskChoice:
    """decide_mask with this file's settings."""
    return decide_mask(samples, ratio=MASK_RATIO, min_coverage=MIN_COVERAGE,
                       max_coverage=MAX_COVERAGE)


# The Sample_Buffer and the Mask_Samples (Req 1.12, 7.1, 7.7)
#
# With mask_samples = 2, the buffer holds at most 4 frames. Frame 4 makes 5, so it keeps 0, 2 and
# 4, and k becomes 2. Frame 8 makes 0, 2, 4, 6 and 8, so it keeps 0, 4 and 8, and k becomes 4.
# Frame 9 isn't a multiple of 4, so it isn't kept.

def ten_frames_in_a_buffer_of_2() -> tuple[SampleBuffer, list[np.ndarray]]:
    """A SampleBuffer with mask_samples = 2 after 10 frames, and those frames, each a new 2 × 3
    array whose brightness is its Frame_Number."""
    frames = [np.full((2, 3), number, dtype=np.uint8) for number in range(10)]
    buffer = SampleBuffer(2)
    for frame in frames:
        buffer.add(frame)
    return buffer, frames


def test_with_mask_samples_2_ten_frames_leave_frame_numbers_0_4_and_8_with_stride_4() -> None:
    buffer, _ = ten_frames_in_a_buffer_of_2()
    assert buffer.numbers == [0, 4, 8]
    assert buffer.stride == 4


@pytest.mark.parametrize(("first", "last", "numbers"), [
    # 0, 4 and 8 lie inside, more than mask_samples, so spread(3, 2) takes the first and the last.
    pytest.param(0, 9, [0, 8], id="0 to 9 gives 0 and 8"),
    # 4 and 8 lie inside, no more than mask_samples, so both are taken.
    pytest.param(3, 9, [4, 8], id="3 to 9 gives 4 and 8"),
    pytest.param(5, 7, [], id="5 to 7 gives none"),
])
def test_the_mask_samples_are_the_buffered_frames_inside_the_kept_range(
        first: int, last: int, numbers: list[int]) -> None:
    buffer, frames = ten_frames_in_a_buffer_of_2()
    samples = buffer.samples_in(first, last)
    assert [number for number, _ in samples] == numbers
    assert all(frame is frames[number] for number, frame in samples)


def test_a_kept_range_with_no_buffered_frame_gives_no_mask() -> None:
    buffer, _ = ten_frames_in_a_buffer_of_2()
    choice = decide([frame for _, frame in buffer.samples_in(5, 7)])
    assert choice.samples == 0
    assert not choice.used


def test_spread_of_3_samples_over_5_places_gives_0_2_and_4() -> None:
    assert spread(5, 3) == [0, 2, 4]


# What the Score_Pass holds at once (Req 1.12, 1.13, 13.7)
#
# The Score_Pass gives each Score_Frame to the Scorer and then to the buffer, and keeps nothing
# itself. The Scorer keeps only its latest frame, to score the next one against, and the buffer
# at most 2 × mask_samples frames. So however long the stream, every frame still alive after
# each one is the current frame, the one before it or one the buffer holds. 20,000 frames are
# over 11 minutes at 30 fps, longer than the 10-minute clip of Req 1.13.

# The long stream's length, its mask_samples and its frame rate.
LONG_STREAM_FRAMES = 20_000
LONG_STREAM_MASK_SAMPLES = 3
LONG_STREAM_FPS = 30


def test_20000_frames_leave_alive_only_the_current_one_the_one_before_and_the_buffers_6() -> None:
    scorer = Scorer()
    buffer = SampleBuffer(LONG_STREAM_MASK_SAMPLES)
    rng = np.random.default_rng(0)
    # A weakref to each frame the test hasn't yet seen die, by Frame_Number.
    alive: dict[int, weakref.ref[np.ndarray]] = {}
    for number in range(LONG_STREAM_FRAMES):
        # A new array, as the decode gives every frame. This variable is the test's only strong
        # reference to it, until the next frame takes its place.
        frame = rng.integers(0, 256, size=(4, 4), dtype=np.uint8)
        alive[number] = weakref.ref(frame)
        scorer.add(number / LONG_STREAM_FPS, frame)
        buffer.add(frame)
        held = buffer.numbers
        assert len(held) <= 2 * LONG_STREAM_MASK_SAMPLES
        alive = {frame_number: ref for frame_number, ref in alive.items() if ref() is not None}
        assert set(alive) <= {number, number - 1, *held}


# The Mask (Req 7.2–7.8)

def test_a_4_by_6_corner_that_never_changes_is_found_as_the_mask_with_its_coverage() -> None:
    # 4 × 6 of a 12 × 16 frame is 24 of its 192 pixels, a Mask_Coverage of 0.125, from
    # mask_min_coverage to mask_max_coverage.
    planted = corner(12, 16, 4, 6)
    choice = decide(planted_on_noise(planted))
    assert choice.samples == SAMPLES
    assert choice.stuck.tolist() == planted.tolist()
    assert choice.coverage == 0.125
    assert choice.used


def test_a_still_scene_whose_median_change_is_0_gives_no_mask() -> None:
    # Every sample is the same picture, so every Pixel_Change is 0. A pixel is stuck only when
    # its Pixel_Change is below mask_ratio × 0, so none is.
    frame = noise(12, 16, seed=3)
    samples = [frame.copy() for _ in range(SAMPLES)]
    assert float(np.median(pixel_change(samples))) == 0.0
    choice = decide(samples)
    assert not choice.stuck.any()
    assert choice.coverage == 0.0
    assert not choice.used


@pytest.mark.parametrize(("height", "width", "rows", "columns"), [
    # 6 × 9 of 12 × 16 is 54 of 192 pixels, a coverage of 0.28125.
    pytest.param(12, 16, 6, 9, id="over 25 %"),
    # 1 pixel of 24 × 32 is 1 of 768, a coverage of about 0.0013.
    pytest.param(24, 32, 1, 1, id="under 0.2 %"),
])
def test_a_planted_region_over_25_or_under_0_2_percent_of_the_frame_gives_no_mask(
        height: int, width: int, rows: int, columns: int) -> None:
    # The region is found, so the coverage is its share of the frame, but that share is outside
    # mask_min_coverage to mask_max_coverage.
    choice = decide(planted_on_noise(corner(height, width, rows, columns)))
    assert choice.coverage == rows * columns / (height * width)
    assert not choice.used


@pytest.mark.parametrize("count", [0, 1])
def test_0_or_1_samples_give_no_mask_and_a_coverage_of_0(count: int) -> None:
    # The samples have the corner that the 4 × 6 corner test finds, but with no pair of samples
    # there's no Pixel_Change.
    choice = decide(planted_on_noise(corner(12, 16, 4, 6), count))
    assert choice.samples == count
    assert choice.stuck is None
    assert choice.coverage == 0.0
    assert not choice.used


@pytest.mark.parametrize(("count", "height", "width"), [(2, 1, 1), (5, 6, 8), (8, 12, 16)])
def test_pixel_change_equals_a_numpy_oracle(count: int, height: int, width: int) -> None:
    # The first two samples are all 255 and all 0, so the first change is the largest, −255,
    # which a uint8 subtraction would wrap round to 1. The others are noise.
    frames = [np.full((height, width), 255, dtype=np.uint8),
              np.zeros((height, width), dtype=np.uint8),
              *(noise(height, width, seed) for seed in range(count - 2))]
    oracle = np.abs(np.diff(np.stack(frames).astype(np.int64), axis=0)).mean(axis=0)
    change = pixel_change(frames)
    assert change.dtype == np.float64
    assert change.tolist() == oracle.tolist()


def test_the_stuck_pixels_are_exactly_the_planted_ones_with_none_added_or_removed() -> None:
    # A lone pixel in a corner, a lone pixel in the middle, a line 1 pixel thin and two pixels
    # that touch only at their corners. Growing the Stuck_Pixels would add the pixels around
    # them, shrinking or smoothing them would remove the lone pixels and the line, and closing
    # them would join the two that touch at their corners.
    planted = np.zeros((12, 16), dtype=bool)
    planted[0, 15] = True
    planted[5, 7] = True
    planted[9, 2:12] = True
    planted[[2, 3], [3, 4]] = True
    samples = planted_on_noise(planted)
    assert stuck_pixels(pixel_change(samples), MASK_RATIO).tolist() == planted.tolist()
    assert decide(samples).stuck.tolist() == planted.tolist()


def test_mask_image_scales_a_2_by_2_stuck_array_to_4_by_4_as_2_by_2_blocks() -> None:
    # The top-right Score pixel is stuck. Each Score pixel covers a 2 × 2 block of the image,
    # which takes its value: 0 for the stuck one and 255 for the others.
    image = mask_image(np.array([[False, True], [False, False]]), width=4, height=4)
    assert image.dtype == np.uint8
    assert image.tolist() == [[255, 255, 0, 0],
                              [255, 255, 0, 0],
                              [255, 255, 255, 255],
                              [255, 255, 255, 255]]
    assert np.unique(image).tolist() == [0, 255]


# Property tests

@dataclass(frozen=True)
class BufferCase:
    """A drawn stream for the Sample_Buffer, as its length and mask_samples, with a
    Kept_Range."""

    count: int  # n, how many frames the stream holds
    samples: int  # m, mask_samples
    first: int  # the Kept_Range's first Frame_Number
    last: int  # its last, inclusive


@st.composite
def buffer_cases(draw: st.DrawFn) -> BufferCase:
    """A stream of 1 to 2,000 frames, a mask_samples from 2 to 40, and a Kept_Range with
    0 <= first <= last <= count − 1.

    The Kept_Range often starts at frame 0 and often ends at the last frame, so it often holds
    every buffered frame. Once the buffer has thinned, that's more than mask_samples, so
    samples_in has to spread them.
    """
    count = draw(st.integers(1, 2000))
    samples = draw(st.integers(2, 40))
    first = draw(st.just(0) | st.integers(0, count - 1))
    last = draw(st.just(count - 1) | st.integers(first, count - 1))
    return BufferCase(count=count, samples=samples, first=first, last=last)


def oracle_stride(fed: int, samples: int) -> int:
    """The stride k after `fed` frames, worked out from that count alone: the smallest power of
    2 with ceil(fed / k) at most 2 × samples."""
    stride = 1
    while (fed + stride - 1) // stride > 2 * samples:
        stride *= 2
    return stride


@given(buffer_cases())
def test_the_sample_buffer_thins_evenly_and_the_mask_samples_stay_in_the_kept_range(
        case: BufferCase) -> None:
    """Feature: frames, Property 10: The Sample_Buffer thins evenly and the Mask_Samples stay in
    the Kept_Range

    **Validates: Requirements 1.12, 7.1**
    """
    buffer = SampleBuffer(case.samples)
    most = 2 * case.samples  # the most frames the buffer may hold
    # Each frame is a new 1 × 1 array with the same pixel, so only identity tells them apart.
    # The test keeps every one, so no two of them can share an id.
    frames: list[np.ndarray] = []
    for fed in range(1, case.count + 1):
        frame = np.zeros((1, 1), dtype=np.uint8)
        frames.append(frame)
        buffer.add(frame)
        stride = oracle_stride(fed, case.samples)
        held = buffer.numbers
        if fed <= most:
            assert len(held) == fed
        else:
            assert case.samples + 1 <= len(held) <= most
        assert held == list(range(0, fed, stride))
        assert buffer.stride == stride
    # The c held Frame_Numbers from first to last, counted from the oracle's own stride.
    stride = oracle_stride(case.count, case.samples)
    inside = [number for number in range(0, case.count, stride)
              if case.first <= number <= case.last]
    pairs = buffer.samples_in(case.first, case.last)
    numbers = [number for number, _ in pairs]
    assert len(numbers) == min(len(inside), case.samples)
    if len(inside) <= case.samples:
        assert numbers == inside
    else:
        # Each Mask_Sample's place among the c. The places increase, so the samples come in
        # Frame_Number order with none twice. They take in the first and the last of the c, and
        # the gaps between neighbours differ by at most 1.
        assert set(numbers) <= set(inside)
        place = {number: index for index, number in enumerate(inside)}
        places = [place[number] for number in numbers]
        gaps = [after - before for before, after in pairwise(places)]
        assert places[0] == 0
        assert places[-1] == len(inside) - 1
        assert min(gaps) >= 1
        assert max(gaps) - min(gaps) <= 1
    # Each pair holds the very array fed with its Frame_Number.
    assert all(frame is frames[number] for number, frame in pairs)


# mask_ratio: any float from 0 to 1, often 0, 0.25, 0.5 or 1. Those are binary fractions, so
# with a whole-number median their product is exact, and a Pixel_Change often equals it. Then
# the pixel isn't below it, so it isn't stuck.
MASK_RATIOS = st.floats(0, 1) | st.sampled_from([0.0, 0.25, 0.5, 1.0])


@dataclass(frozen=True)
class MaskCase:
    """A drawn list of Mask_Samples, with mask_ratio, mask_min_coverage and
    mask_max_coverage."""

    samples: tuple[np.ndarray, ...]
    ratio: float
    min_coverage: float
    max_coverage: float


@st.composite
def mask_cases(draw: st.DrawFn) -> MaskCase:
    """0 to 8 Mask_Samples of one shape up to 12 × 16, with the Mask's settings.

    Each sample and the region are drawn by arrays, which often gives most of their pixels one
    value, or have every pixel drawn on its own. So most pixels of the samples change from one
    sample to the next, and the region can be any set of pixels. About half the time the region
    holds one drawn brightness in every sample, so its pixels never change. mask_min_coverage
    and mask_max_coverage are two draws, sorted, each any float from 0 to 1 or a share
    k / pixels of the frame, so the Mask_Coverage often lands exactly on one of them.
    """
    height, width = draw(st.integers(1, 12)), draw(st.integers(1, 16))
    count = draw(st.integers(0, 8))
    frames = (arrays(np.uint8, (height, width))
              | arrays(np.uint8, (height, width), fill=st.nothing()))
    samples = draw(st.lists(frames, min_size=count, max_size=count))
    if draw(st.booleans()):
        region = draw(arrays(np.bool_, (height, width))
                      | arrays(np.bool_, (height, width), fill=st.nothing()))
        brightness = np.uint8(draw(st.integers(0, 255)))
        samples = [np.where(region, brightness, sample) for sample in samples]
    pixels = height * width
    coverages = st.floats(0, 1) | st.integers(0, pixels).map(lambda k: k / pixels)
    low, high = sorted([draw(coverages), draw(coverages)])
    return MaskCase(samples=tuple(samples), ratio=draw(MASK_RATIOS), min_coverage=low,
                    max_coverage=high)


def oracle_stuck(samples: tuple[np.ndarray, ...], ratio: float) -> list[list[bool]]:
    """The Stuck_Pixels of 2 or more `samples`, row by row, found from the definitions with
    Python ints and plain loops.

    A pixel's Pixel_Change is the sum of its absolute changes from each sample to the next, as
    Python ints, divided once by the number of pairs. That rounds to the same float as
    pixel_change's int32 sum divided by it. The median is the middle Pixel_Change once sorted,
    or the mean of the two middle ones, as numpy's median works it out. A pixel is stuck when its
    Pixel_Change is below mask_ratio times that median.
    """
    frames = [sample.tolist() for sample in samples]
    height, width = len(frames[0]), len(frames[0][0])
    pairs = len(frames) - 1
    change = [[sum(abs(after[y][x] - before[y][x]) for before, after in pairwise(frames)) / pairs
               for x in range(width)] for y in range(height)]
    ordered = sorted(value for row in change for value in row)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        median = ordered[middle]
    else:
        median = (ordered[middle - 1] + ordered[middle]) / 2
    return [[value < ratio * median for value in row] for row in change]


@given(mask_cases())
def test_the_mask_is_unused_or_covers_the_allowed_share(case: MaskCase) -> None:
    """Feature: frames, Property 11: The Mask is unused or covers the allowed share

    **Validates: Requirements 7.3, 7.4, 7.5, 7.6, 7.7, 7.11**
    """
    choice = decide_mask(list(case.samples), ratio=case.ratio, min_coverage=case.min_coverage,
                         max_coverage=case.max_coverage)
    # No Mask, or one that covers from mask_min_coverage to mask_max_coverage (Req 7.11).
    assert not choice.used or case.min_coverage <= choice.coverage <= case.max_coverage
    if len(case.samples) < 2:
        # With no pair of samples there's no Pixel_Change, so there's no Mask (Req 7.7).
        assert not choice.used
    else:
        # The Stuck_Pixels are exactly the oracle's, pixel by pixel, so none was added or
        # removed around another (Req 7.3, 7.6).
        stuck = oracle_stuck(case.samples, case.ratio)
        assert choice.stuck.dtype == np.bool_
        assert choice.stuck.tolist() == stuck
        # The Mask_Coverage is their share of the frame, and the Mask is used exactly when that's
        # from mask_min_coverage to mask_max_coverage, both included (Req 7.4, 7.5).
        coverage = sum(map(sum, stuck)) / (len(stuck) * len(stuck[0]))
        assert choice.coverage == coverage
        assert choice.used == (case.min_coverage <= coverage <= case.max_coverage)


@st.composite
def stuck_arrays(draw: st.DrawFn) -> np.ndarray:
    """A bool array of Stuck_Pixels at a Score_Frame's size, from 1 × 1 to 12 × 16.

    arrays often gives most of the pixels one value, or has every pixel drawn on its own, so the
    stuck pixels can be none, all, a few, most or any set.
    """
    shape = (draw(st.integers(1, 12)), draw(st.integers(1, 16)))
    return draw(arrays(np.bool_, shape) | arrays(np.bool_, shape, fill=st.nothing()))


# A Pick_Size's width or height, from 1 to 40, so the Mask image is sometimes larger than the
# Score_Frame, at most 12 × 16, and sometimes smaller.
PICK_EDGES = st.integers(1, 40)


def oracle_score_pixel(centre: Fraction, pixels: int) -> int:
    """The Score pixel along one edge, from 0 to pixels − 1, whose span holds `centre`.

    `pixels` is how many Score pixels the edge holds, and positions are shares of the edge's
    length, from 0 to 1. So Score pixel j spans j / pixels to (j + 1) / pixels. Each span holds
    its start but not its end, so a centre on the line between two Score pixels belongs to the
    later one.
    """
    return next(pixel for pixel in range(pixels)
                if Fraction(pixel, pixels) <= centre < Fraction(pixel + 1, pixels))


def oracle_mask_image(stuck: np.ndarray, width: int, height: int) -> list[list[int]]:
    """The Mask image at the Pick_Size width × height, row by row, found from the definition.

    Output pixel (x, y) spans x / width to (x + 1) / width of the picture's width and y / height
    to (y + 1) / height of its height, so its centre lies (2x + 1) / (2 × width) across and
    (2y + 1) / (2 × height) down. A Score pixel's area is its column's span across and its row's
    span down, so the Score pixel that holds the centre is the one whose column holds the
    centre's x and whose row holds its y. The output pixel is 0 when that Score pixel is stuck
    and 255 when it isn't.
    """
    score_height, score_width = stuck.shape
    is_stuck = stuck.tolist()
    rows = [oracle_score_pixel(Fraction(2 * y + 1, 2 * height), score_height)
            for y in range(height)]
    columns = [oracle_score_pixel(Fraction(2 * x + 1, 2 * width), score_width)
               for x in range(width)]
    return [[0 if is_stuck[row][column] else 255 for column in columns] for row in rows]


@given(stuck=stuck_arrays(), width=PICK_EDGES, height=PICK_EDGES)
def test_the_mask_image_holds_only_black_and_white_at_the_pick_size(
        stuck: np.ndarray, width: int, height: int) -> None:
    """Feature: frames, Property 12: The Mask image holds only black and white at the Pick_Size

    **Validates: Requirements 7.8**
    """
    image = mask_image(stuck, width, height)
    # An 8-bit image at the Pick_Size, height rows of width pixels, with no value but 0 and 255.
    assert image.dtype == np.uint8
    assert image.shape == (height, width)
    assert set(np.unique(image).tolist()) <= {0, 255}
    # Each pixel takes the value of the Score pixel whose area holds its centre: 0, black, when
    # that Score pixel is stuck, and 255, white, when it isn't.
    assert image.tolist() == oracle_mask_image(stuck, width, height)
