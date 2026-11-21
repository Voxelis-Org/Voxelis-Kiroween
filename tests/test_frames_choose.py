"""Tests for splatpipe/frames/choose.py: the Windows, the Still_Trim and the Jolt_Filter.

The unit tests' clips are a few numpy arrays built here, with frames 0.1 s apart, so no test needs
a video. The Still_Trim and Jolt_Filter tests use Windows of 0.5 s, which hold a frame and the two
frames on each side of it. Half a Window, 0.25 s, is no multiple of the 0.1 s between frames, so no
frame sits on a Window's edge, where rounding in the times could move it in or out. The Window test
puts frames exactly on the edges instead, with times that are binary fractions, so its sums are
exact. Each expected Kept_Range is worked out from the definitions: the Still_Frames come from the
mean Motion over each frame's Window, not from the frame's own Motion.

Property 5 draws its clips with Hypothesis: 1 to 60 frames whose times are running sums of gaps
from 0.001 s to 2 s, and Motions in steps of 1/256. Some gaps and settings are 0.25 s, 0.5 s, 1 s
or 2 s, so frames land exactly on Window edges. Its oracle finds each Window with the definitions'
comparisons on the same floats still_trim gets, so a frame exactly on a Window's edge falls in or
out the same way for both. Every sum of Motions in steps of 1/256 is exact, so the oracle's mean
of a Window is numpy's to the last bit.

Property 6 draws Kept_Ranges of 1 to 60 frames with the same gaps and Window lengths, Sharpness
values from 0 to 2²⁰ in steps of 1/256, and a jolt_ratio from 0 to 1. Its oracle finds each Window
with a plain loop and the same comparisons, and sorts the Window to find its median. Any two of
those Sharpness values add up exactly, so a median that's the mean of the two middle values is
numpy's to the last bit, and the oracle's jolt_ratio times the median is the same float jolt_keep
compares with.
"""

from dataclasses import asdict, dataclass
from itertools import accumulate

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from splatpipe.frames.choose import Kept, jolt_keep, still_trim, window_bounds

# still_motion, min_still_seconds, blip_seconds and jolt_ratio, as the real config has them.
STILL_MOTION = 1.0
MIN_STILL_SECONDS = 2.0
BLIP_SECONDS = 0.5
JOLT_RATIO = 0.6
# The Window of the still test and of the jolt test. The real config's 1 s would put frames 0.5 s
# apart exactly on each other's Window edges.
WINDOW_SECONDS = 0.5
# A walking frame's Motion, well above still_motion.
MOVING = 10.0
FRAMES_PER_SECOND = 10


def clip(*parts: tuple[float, float]) -> tuple[np.ndarray, np.ndarray, float]:
    """A clip of frames 0.1 s apart from 0 s, made of parts given as (seconds, Motion).

    Gives the Frame_Times, the Motions and the Used_Part's end, 0.1 s after the last frame. Frame
    i's time is i / 10, as near as a float gets, and frame 0's Motion is 0, as the Scorer gives it.
    """
    motion = np.concatenate([np.full(round(seconds * FRAMES_PER_SECOND), value, dtype=np.float64)
                             for seconds, value in parts])
    motion[0] = 0.0
    times = np.arange(len(motion)) / FRAMES_PER_SECOND
    return times, motion, len(motion) / FRAMES_PER_SECOND


def trim(times: np.ndarray, motion: np.ndarray, end: float, *,
         window_seconds: float = WINDOW_SECONDS) -> Kept:
    """still_trim with this file's settings."""
    return still_trim(times, motion, end, still_motion=STILL_MOTION,
                      window_seconds=window_seconds, min_still_seconds=MIN_STILL_SECONDS,
                      blip_seconds=BLIP_SECONDS)


def assert_kept(kept: Kept, expected: Kept) -> None:
    """Check every field of `kept`: `first` and `last` exactly, and the four seconds within 1e-9.

    A time like 3.2 s isn't exact in binary, so the difference of two times can be one float away
    from the decimal the test writes.
    """
    want = asdict(expected)
    for name in ("start_seconds", "end_seconds", "cut_start_seconds", "cut_end_seconds"):
        want[name] = pytest.approx(want[name], abs=1e-9)
    assert asdict(kept) == want


def jolt(times: np.ndarray, sharpness: np.ndarray) -> np.ndarray:
    """jolt_keep with this file's jolt_ratio and Window."""
    return jolt_keep(times, sharpness, ratio=JOLT_RATIO, window_seconds=WINDOW_SECONDS)


# Windows (Req 4.1, 5.1, A2)

# 2⁻¹⁰ s, about 1 ms. Adding it to a binary fraction gives another binary fraction.
JUST_PAST = 2.0 ** -10


def test_a_window_holds_its_frame_and_the_frames_exactly_half_its_length_away() -> None:
    # A 1 s Window reaches 0.5 s each way. Frames 1 and 3 are exactly 0.5 s from frame 2, and
    # frames 0 and 4 are 2⁻¹⁰ s further. Every time is a binary fraction, so each time ± 0.5 is
    # exact, and the frames on an edge are exactly on it.
    times = np.array([0.5 - JUST_PAST, 0.5, 1.0, 1.5, 1.5 + JUST_PAST])
    lo, hi = window_bounds(times, 1.0)
    windows = [list(range(low, high)) for low, high in zip(lo.tolist(), hi.tolist(), strict=True)]
    assert windows == [[0, 1], [0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4]]


# The Still_Trim (Req 4)
#
# Each of these clips has 3 s of walking and a still part of zero Motion before or after it. A
# still frame up to 0.2 s from a walking frame has it in its Window, so its mean Motion is at least
# 10 / 5 = 2, which isn't below still_motion. So the 0.2 s of the still part next to the walking
# isn't still. 3 s of zero Motion gives a Still_Start or Still_End of 2.8 s, which is cut, and
# 1.5 s gives 1.3 s, which is under min_still_seconds and kept. A cut Still_End's Kept_Range ends
# where the Still_End starts, at 3.2 s.

@pytest.mark.parametrize(("parts", "expected"), [
    pytest.param([(3.0, 0.0), (3.0, MOVING)],
                 Kept(first=28, last=59, start_seconds=2.8, end_seconds=6.0,
                      cut_start_seconds=2.8, cut_end_seconds=0.0),
                 id="3 s still start is cut"),
    pytest.param([(1.5, 0.0), (3.0, MOVING)],
                 Kept(first=0, last=44, start_seconds=0.0, end_seconds=4.5,
                      cut_start_seconds=0.0, cut_end_seconds=0.0),
                 id="1.5 s still start is kept"),
    pytest.param([(3.0, MOVING), (3.0, 0.0)],
                 Kept(first=0, last=31, start_seconds=0.0, end_seconds=3.2,
                      cut_start_seconds=0.0, cut_end_seconds=2.8),
                 id="3 s still end is cut"),
    pytest.param([(3.0, MOVING), (1.5, 0.0)],
                 Kept(first=0, last=44, start_seconds=0.0, end_seconds=4.5,
                      cut_start_seconds=0.0, cut_end_seconds=0.0),
                 id="1.5 s still end is kept"),
])
def test_a_still_start_or_end_is_cut_from_min_still_seconds_and_kept_when_shorter(
        parts: list[tuple[float, float]], expected: Kept) -> None:
    times, motion, end = clip(*parts)
    assert_kept(trim(times, motion, end), expected)


def test_a_blip_inside_a_still_start_doesnt_end_it_so_the_whole_still_start_is_cut() -> None:
    # A 3 s still start with a 0.3 s Blip 1 s in, then 3 s of walking. This test's Window is
    # 0.05 s, which holds only the frame itself, so the frames that aren't still are exactly the
    # moving ones, and the Blip's run of them lasts 0.3 s, under blip_seconds. With 0.5 s Windows,
    # the Blip's Motion would reach the two frames on each side of it, and the run would last
    # 0.7 s: a Moving_Stretch, so the case wouldn't test a Blip.
    times, motion, end = clip((1.0, 0.0), (0.3, MOVING), (1.7, 0.0), (3.0, MOVING))
    assert_kept(trim(times, motion, end, window_seconds=0.05),
                Kept(first=30, last=59, start_seconds=3.0, end_seconds=6.0,
                     cut_start_seconds=3.0, cut_end_seconds=0.0))


def test_a_still_middle_between_two_moving_stretches_is_kept() -> None:
    # 2 s of walking, 3 s standing still and 2 s of walking. The still middle's Still_Frames last
    # 2.6 s, long enough to be cut at an end, but the Still_Trim cuts only at the ends.
    times, motion, end = clip((2.0, MOVING), (3.0, 0.0), (2.0, MOVING))
    assert_kept(trim(times, motion, end),
                Kept(first=0, last=69, start_seconds=0.0, end_seconds=7.0,
                     cut_start_seconds=0.0, cut_end_seconds=0.0))


def test_a_clip_that_is_still_from_start_to_end_is_kept_whole() -> None:
    # 3 s with a Motion of 0.5, under still_motion. There's no Moving_Stretch, so nothing is cut,
    # though the clip lasts longer than min_still_seconds.
    times, motion, end = clip((3.0, 0.5))
    assert_kept(trim(times, motion, end),
                Kept(first=0, last=29, start_seconds=0.0, end_seconds=3.0,
                     cut_start_seconds=0.0, cut_end_seconds=0.0))


# The Jolt_Filter (Req 5.1)

def test_the_jolt_filter_drops_one_blurred_frame_among_sharp_ones() -> None:
    # Each Window holds 3 to 5 frames, at most one of them the blurred frame, so every median is
    # 100. The blurred frame's 20 is below 0.6 × 100, and the sharp frames' 100 isn't.
    sharpness = np.full(10, 100.0)
    sharpness[4] = 20.0
    keep = jolt(np.arange(10) / 10, sharpness)
    assert keep.dtype == np.bool_
    assert keep.tolist() == [True] * 4 + [False] + [True] * 5


def test_the_jolt_filter_keeps_every_frame_when_every_sharpness_is_0() -> None:
    # Every median is 0, and a Sharpness of 0 is at least 0.6 × 0.
    keep = jolt(np.arange(10) / 10, np.zeros(10))
    assert keep.tolist() == [True] * 10


def test_the_jolt_filter_takes_its_medians_only_from_the_kept_range_frames_it_is_given() -> None:
    # A Used_Part of 20 frames whose first 10 the Still_Trim cut. They're blurred, with a Sharpness
    # of 10, and the Kept_Range's first frame, frame 10, has 50 among frames of 100.
    times = np.arange(20) / 10
    sharpness = np.array([10.0] * 10 + [50.0] + [100.0] * 9)
    # Given the Kept_Range, frame 10's Window holds frames 10 to 12, whose median is 100, so its
    # 50 is below 0.6 × 100 and it's dropped.
    assert jolt(times[10:], sharpness[10:]).tolist() == [False] + [True] * 9
    # Given the whole Used_Part, its Window would also hold the cut frames 8 and 9, and the median
    # would be 50, so it would be kept.
    assert jolt(times, sharpness)[10]


# Property tests

# Seconds that are binary fractions. Mixed into the gaps and the settings below, they put frames
# exactly on Window edges and make Stretches last exactly a setting's length.
ROUND_SECONDS = st.sampled_from([0.25, 0.5, 1.0, 2.0])
# The gap from one Frame_Time to the next, or from the last one to the Used_Part's end.
GAPS = st.floats(0.001, 2) | ROUND_SECONDS
# motion_window_seconds, min_still_seconds, blip_seconds and jolt_window_seconds: from shorter
# than any gap, so a Window can hold only its own frame, to 10 s, longer than many gaps.
SETTING_SECONDS = st.floats(0.001, 10) | ROUND_SECONDS
# Any Motion from 0 to 255 in steps of 1/256. still_motion takes the same steps, so every Motion
# drawn does, and a sum of up to 60 of them is exact whatever order it's added in. So the
# oracle's mean of a Window is the same float as numpy's, even exactly at still_motion.
ANY_MOTION = st.integers(0, 255 * 256).map(lambda k: k / 256)


@dataclass(frozen=True)
class TrimCase:
    """A drawn clip, as its Frame_Times, Motions and Used_Part end, with the Still_Trim's
    settings."""

    times: tuple[float, ...]
    motion: tuple[float, ...]
    end: float
    still_motion: float
    window_seconds: float
    min_still_seconds: float
    blip_seconds: float


@st.composite
def motion_values(draw: st.DrawFn, still_motion: float) -> float:
    """A Motion: four times in five 0, still_motion, 2 × still_motion or 255, so stills, Blips
    and Moving_Stretches all show up, and otherwise ANY_MOTION."""
    level = draw(st.sampled_from([0.0, still_motion, 2 * still_motion, 255.0, None]))
    return draw(ANY_MOTION) if level is None else level


@st.composite
def trim_cases(draw: st.DrawFn) -> TrimCase:
    """1 to 60 frames whose times are running sums of GAPS, with the Used_Part's end one more gap
    after the last. still_motion is from 1/256 to 127.5 in steps of 1/256, so 2 × still_motion
    is a Motion too."""
    stamps = tuple(accumulate(draw(st.lists(GAPS, min_size=2, max_size=61))))
    still_motion = draw(st.integers(1, 255 * 128)) / 256
    motion = draw(st.lists(motion_values(still_motion), min_size=len(stamps) - 1,
                           max_size=len(stamps) - 1))
    return TrimCase(times=stamps[:-1], motion=tuple(motion), end=stamps[-1],
                    still_motion=still_motion, window_seconds=draw(SETTING_SECONDS),
                    min_still_seconds=draw(SETTING_SECONDS), blip_seconds=draw(SETTING_SECONDS))


def oracle_moving_stretches(case: TrimCase) -> list[range]:
    """The Moving_Stretches of `case`, each as the range of its Frame_Numbers, found from the
    definitions with plain loops.

    Frame j is in frame i's Window when t_i − h <= t_j <= t_i + h, with h half of
    window_seconds: the definitions' rule, both edges included, worked out on the same floats
    still_trim gets, so a frame exactly on an edge falls the same way for both. A frame is still
    when its Window's mean Motion is below still_motion. A Moving_Stretch is a run of frames that
    aren't still, as long as it can be, lasting blip_seconds or more from its first frame's time
    to the next frame's time, or to the Used_Part's end.
    """
    times, n = case.times, len(case.times)
    half = case.window_seconds / 2
    still: list[bool] = []
    for i in range(n):
        window = [case.motion[j] for j in range(n)
                  if times[i] - half <= times[j] <= times[i] + half]
        still.append(sum(window) / len(window) < case.still_motion)
    stretches: list[range] = []
    first = 0
    while first < n:
        if still[first]:
            first += 1
            continue
        stop = first  # the first still frame after the run, or n
        while stop < n and not still[stop]:
            stop += 1
        lasts = (times[stop] if stop < n else case.end) - times[first]
        if lasts >= case.blip_seconds:
            stretches.append(range(first, stop))
        first = stop
    return stretches


@given(trim_cases())
def test_the_still_trim_keeps_one_stretch_that_holds_all_the_movement(case: TrimCase) -> None:
    """Feature: frames, Property 5: The Still_Trim keeps one Stretch that holds all the movement

    **Validates: Requirements 4.4, 4.5, 4.7, 4.8, 4.9**
    """
    kept = still_trim(np.array(case.times), np.array(case.motion), case.end,
                      still_motion=case.still_motion, window_seconds=case.window_seconds,
                      min_still_seconds=case.min_still_seconds, blip_seconds=case.blip_seconds)
    n = len(case.times)
    assert 0 <= kept.first <= kept.last <= n - 1
    moving = oracle_moving_stretches(case)
    if not moving:
        assert (kept.first, kept.last) == (0, n - 1)
        assert (kept.cut_start_seconds, kept.cut_end_seconds) == (0.0, 0.0)
    # Every Moving_Stretch frame is in the Kept_Range, so neither cut end holds one.
    moving_frames = {frame for stretch in moving for frame in stretch}
    assert moving_frames <= set(range(kept.first, kept.last + 1))
    # A cut Still_Start runs from frame 0's time to the Kept_Range's first frame's time, and a
    # cut Still_End from the time of the frame after the Kept_Range to the Used_Part's end.
    if kept.first > 0:
        assert case.times[kept.first] - case.times[0] >= case.min_still_seconds
    if kept.last < n - 1:
        assert case.end - case.times[kept.last + 1] >= case.min_still_seconds


# A Sharpness from 0 to 2²⁰ in steps of 1/256. 2²⁰ is more than 1020², the largest variance a
# uint8 frame's Laplacian can have. Any two of them add up exactly, so the mean of a Window's two
# middle values, its median when it holds an even number of frames, is exact.
ANY_SHARPNESS = st.integers(0, 2 ** 28).map(lambda k: k / 256)
# Sharpness values are often 0, 25, 50 or 100, and jolt_ratio often 0, 0.25, 0.5 or 1, so a
# frame's Sharpness is often exactly jolt_ratio times its Window's median, like 50 next to a
# median of 100 with a jolt_ratio of 0.5. That tests the rule's "at least" on its edge.
SHARPNESS = st.sampled_from([0.0, 25.0, 50.0, 100.0]) | ANY_SHARPNESS
JOLT_RATIOS = st.floats(0, 1) | st.sampled_from([0.0, 0.25, 0.5, 1.0])


@dataclass(frozen=True)
class JoltCase:
    """A drawn Kept_Range, as its Frame_Times and Sharpness values, with the Jolt_Filter's
    settings."""

    times: tuple[float, ...]
    sharpness: tuple[float, ...]
    ratio: float
    window_seconds: float


@st.composite
def jolt_cases(draw: st.DrawFn) -> JoltCase:
    """A Kept_Range of 1 to 60 frames whose times are running sums of GAPS, with a Sharpness for
    each frame, a jolt_ratio from JOLT_RATIOS and a jolt_window_seconds from SETTING_SECONDS."""
    times = tuple(accumulate(draw(st.lists(GAPS, min_size=1, max_size=60))))
    sharpness = draw(st.lists(SHARPNESS, min_size=len(times), max_size=len(times)))
    return JoltCase(times=times, sharpness=tuple(sharpness), ratio=draw(JOLT_RATIOS),
                    window_seconds=draw(SETTING_SECONDS))


def oracle_jolt_keep(case: JoltCase) -> list[bool]:
    """Whether each frame of `case` meets the Jolt_Filter's rule, found from the definitions with
    plain loops.

    Frame j is in frame i's Window when t_i − h <= t_j <= t_i + h, with h half of
    window_seconds, on the same floats jolt_keep gets, as in oracle_moving_stretches. The
    Window's median is its middle Sharpness once sorted, or the mean of the two middle ones. A
    frame meets the rule when its Sharpness is at least jolt_ratio times that median.
    """
    times, n = case.times, len(case.times)
    half = case.window_seconds / 2
    meets: list[bool] = []
    for i in range(n):
        window = sorted(case.sharpness[j] for j in range(n)
                        if times[i] - half <= times[j] <= times[i] + half)
        middle = len(window) // 2
        if len(window) % 2:
            median = window[middle]
        else:
            median = (window[middle - 1] + window[middle]) / 2
        meets.append(case.sharpness[i] >= case.ratio * median)
    return meets


@given(jolt_cases())
def test_the_jolt_filter_keeps_exactly_the_frames_that_meet_its_rule(case: JoltCase) -> None:
    """Feature: frames, Property 6: The Jolt_Filter keeps exactly the frames that meet its rule

    **Validates: Requirements 5.1, 5.5**
    """
    keep = jolt_keep(np.array(case.times), np.array(case.sharpness), ratio=case.ratio,
                     window_seconds=case.window_seconds)
    assert keep.dtype == np.bool_
    # Each kept frame meets the rule and each dropped frame breaks it.
    assert keep.tolist() == oracle_jolt_keep(case)
