"""Tests for splatpipe/frames/choose.py: the Windows, the Still_Trim, the Jolt_Filter,
Target_Picks, the Motion_Parts and the Picker.

The unit tests' clips are a few numpy arrays built here, with frames 0.1 s apart, so no test needs
a video. The Still_Trim and Jolt_Filter tests use Windows of 0.5 s, which hold a frame and the two
frames on each side of it. Half a Window, 0.25 s, is no multiple of the 0.1 s between frames, so no
frame sits on a Window's edge, where rounding in the times could move it in or out. The Window test
puts frames exactly on the edges instead, with times that are binary fractions, so its sums are
exact. Each expected Kept_Range is worked out from the definitions: the Still_Frames come from the
mean Motion over each frame's Window, not from the frame's own Motion.

The Target_Picks tests use the fast Preset's base_picks, seconds_per_pick and max_picks, as the
real config has them. The Motion_Parts and Picker tests count frames from 0 at the Kept_Range's
first frame, as the step does for motion_parts and pick, and the tests of pick give it each
frame's Motion_Part by hand. The test of a total Motion of 0 uses times that are binary fractions,
so each frame's share of the Kept_Range's time is exact.

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

Property 7 draws Kept_Ranges of 1 to 60 frames with the same gaps, an end one more gap after the
last frame, and an n from 1 to 400. The first frame's Motion, which the total doesn't count, is
any Motion from 0 to 255. About half the time every Motion after the first is 0, so the total is
0 and the parts split the Kept_Range's time. Otherwise each is 0, a Motion in steps of 1/256 or
any float from 0 to 255.
Its oracle works out each frame's Motion_Position, or its time since the first frame, exactly
with fractions.Fraction. So the only rounding is motion_parts' own, far less than the
1e-9 × total the property allows.

Property 8 draws 1 to 60 frames with a Motion_Part each. The first frame's part, and each step
from one frame's part to the next frame's, are often 0 and otherwise 1 or 2. So the parts never
go down, a part often holds several frames, and some part numbers hold no frame. Each frame is a
Candidate or not, so some parts hold no Candidate, and pick gets the Candidates in a drawn order.
Each Sharpness is one of 1 to 4 values drawn for the case, so ties are common. Its oracle goes
through the parts with plain loops and takes, from each part's Candidates, the earliest one with
their highest Sharpness.

Property 9 draws a Used_Part of 1 to 60 Score_Frames with the same gaps, its end one more gap
after the last frame, and a Kept_Range inside it. Each frame's Motion is drawn as in Property 7,
and about half the time every Motion after the Kept_Range's first frame is 0 up to its last, so
the parts split the Kept_Range's time. Each Sharpness is drawn as in Property 6, often 0, 25, 50
or 100, so ties are common. Each Kept_Range frame is a Candidate or not, and pick gets the
Candidates in a drawn order. n is from 1 to 400 and often 10 or less, so the Candidates often
outnumber it. The test gives motion_parts and pick the Kept_Range's frames, counted from 0 at
its first frame as the step does, and adds the first frame's Frame_Number back to each Pick. Its
checks read only the drawn case and call nothing else from choose.py.
"""

from dataclasses import asdict, dataclass
from fractions import Fraction
from itertools import accumulate, pairwise

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from splatpipe.frames.choose import (Kept, jolt_keep, motion_parts, pick, still_trim, target_picks,
                                     window_bounds)

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


# Target_Picks (Req 6.1, 6.2, A3)
#
# With the fast Preset, Target_Picks is one Pick for every 2 s of Kept_Seconds, so 240 s gives 120
# and 420 s gives 210. 30 s asks for 15 Picks, fewer than base_picks, and 1000 s asks for 500,
# more than max_picks.

# base_picks of the fast Preset, seconds_per_pick and max_picks, as the real config has them.
FAST_BASE_PICKS = 120
SECONDS_PER_PICK = 2.0
MAX_PICKS = 400


@pytest.mark.parametrize(("kept_seconds", "expected"), [
    pytest.param(240.0, 120, id="240 s gives 120"),
    pytest.param(420.0, 210, id="420 s gives 210"),
    pytest.param(30.0, FAST_BASE_PICKS, id="a short clip gives base_picks"),
    pytest.param(1000.0, MAX_PICKS, id="a long clip stops at max_picks"),
])
def test_target_picks_gives_a_pick_per_seconds_per_pick_from_base_picks_to_max_picks(
        kept_seconds: float, expected: int) -> None:
    assert target_picks(kept_seconds, base_picks=FAST_BASE_PICKS,
                        seconds_per_pick=SECONDS_PER_PICK, max_picks=MAX_PICKS) == expected


# The Motion_Parts and the Picker (Req 6.4–6.7, A5)

def test_with_a_total_motion_of_0_the_parts_split_the_kept_range_by_time() -> None:
    # A Kept_Range from 10 s to its end at 14 s, in 2 parts of 2 s, with four frames 0.5 s apart
    # in its first 2 s and one at 13 s. Its first frame's Motion, 9, is the change from the frame
    # before the Kept_Range, which the total doesn't count, so the total is 0. A split by frame
    # count, or one that ended at the last frame's 13 s, would put frame 3 in part 1.
    times = np.array([10.0, 10.5, 11.0, 11.5, 13.0])
    motion = np.array([9.0, 0.0, 0.0, 0.0, 0.0])
    assert motion_parts(times, motion, 14.0, 2).tolist() == [0, 0, 0, 0, 1]


def test_a_part_with_no_candidate_gives_no_pick_so_there_are_fewer_picks_than_parts() -> None:
    # Three parts of two frames each. The Jolt_Filter dropped both frames of part 1, so it holds
    # no Candidate, and the three parts give two Picks: the sharpest Candidates of parts 0 and 2.
    parts = np.array([0, 0, 1, 1, 2, 2])
    sharpness = np.array([40.0, 60.0, 5.0, 8.0, 70.0, 30.0])
    assert pick(parts, np.array([0, 1, 4, 5]), sharpness).tolist() == [1, 4]


def test_two_equally_sharp_candidates_in_one_part_give_the_earlier_one() -> None:
    # Frames 1 and 2 share part 0's highest Sharpness, 50. The Candidates are given latest first,
    # so the earlier frame wins whatever order they come in.
    parts = np.array([0, 0, 0, 1, 1])
    sharpness = np.array([10.0, 50.0, 50.0, 30.0, 20.0])
    assert pick(parts, np.array([4, 3, 2, 1, 0]), sharpness).tolist() == [1, 3]


def test_the_picks_come_in_frame_time_order() -> None:
    # Four parts whose sharpest Candidates sit at different places in them, with the Candidates
    # given in a shuffled order. Frame_Numbers count in Frame_Time order, so the Picks come in
    # Frame_Time order when their Frame_Numbers increase. By Sharpness they'd come 2, 1, 6, 5.
    parts = np.array([0, 0, 1, 1, 1, 2, 3, 3])
    sharpness = np.array([20.0, 80.0, 90.0, 10.0, 30.0, 40.0, 70.0, 60.0])
    assert pick(parts, np.array([5, 2, 7, 0, 3, 6, 1, 4]), sharpness).tolist() == [1, 2, 5, 6]


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


# A Kept_Range frame's Motion. It's often exactly 0, which leaves the Motion_Position where it
# was. Otherwise it's ANY_MOTION, whose sums are exact, so a Motion_Position can land exactly on
# a part's edge, or any float from 0 to 255.
MOTIONS = st.just(0.0) | ANY_MOTION | st.floats(0, 255)


@dataclass(frozen=True)
class PartCase:
    """A drawn Kept_Range, as its Frame_Times, Motions and end, with the Target_Picks n."""

    times: tuple[float, ...]
    motion: tuple[float, ...]
    end: float
    n: int


@st.composite
def part_cases(draw: st.DrawFn) -> PartCase:
    """A Kept_Range of 1 to 60 frames whose times are running sums of GAPS, with its end one more
    gap after the last, and an n from 1 to 400.

    The first frame's Motion is any of MOTIONS. The total doesn't count it, so it can be large
    while the total is 0. About half the time every Motion after the first is 0, so the total is
    0 and the parts split the time. Otherwise each is any of MOTIONS.
    """
    stamps = tuple(accumulate(draw(st.lists(GAPS, min_size=2, max_size=61))))
    after_first = len(stamps) - 2  # how many of the Kept_Range's frames come after its first
    still = st.just((0.0,) * after_first)
    moving = st.lists(MOTIONS, min_size=after_first, max_size=after_first).map(tuple)
    motion = (draw(MOTIONS), *draw(still | moving))
    return PartCase(times=stamps[:-1], motion=motion, end=stamps[-1],
                    n=draw(st.integers(1, 400)))


def oracle_motion_positions(motion: tuple[float, ...]) -> list[Fraction]:
    """Each Kept_Range frame's Motion_Position, worked out exactly with Fraction.

    It's the sum of the Motions after the Kept_Range's first frame, up to and including the
    frame's own, so the first frame's own Motion isn't counted, the first frame's Motion_Position
    is 0 and the last frame's is the total.
    """
    return list(accumulate(map(Fraction, motion[1:]), initial=Fraction(0)))


@given(part_cases())
def test_motion_parts_are_in_range_and_in_order(case: PartCase) -> None:
    """Feature: frames, Property 7: Motion_Parts are in range and in order

    **Validates: Requirements 6.3, 6.4**
    """
    parts = motion_parts(np.array(case.times), np.array(case.motion), case.end, case.n)
    assert parts.shape == (len(case.times),)
    assert np.issubdtype(parts.dtype, np.integer)
    got = parts.tolist()
    assert all(0 <= part <= case.n - 1 for part in got)
    assert all(before <= after for before, after in pairwise(got))
    assert got[0] == 0
    positions = oracle_motion_positions(case.motion)
    total = positions[-1]
    if total > 0:
        assert got[-1] == case.n - 1
        places, whole = positions, total
    else:
        # The parts split the time from the first frame to the Kept_Range's end.
        first = Fraction(case.times[0])
        places = [Fraction(time) - first for time in case.times]
        whole = Fraction(case.end) - first
    # Part p runs from p × whole / n to (p + 1) × whole / n, and each frame's place is in its
    # part's range, give or take 1e-9 × whole for motion_parts' rounding.
    spare = whole / 10**9
    for part, place in zip(got, places, strict=True):
        assert part * whole / case.n - spare <= place <= (part + 1) * whole / case.n + spare


# The first frame's Motion_Part, and the step from one frame's part to the next frame's. It's
# often 0, so a part often holds several frames, and otherwise 1 or 2, so some part numbers hold
# no frame, as when one frame's Motion is large.
PART_STEPS = st.just(0) | st.integers(0, 2)


@dataclass(frozen=True)
class PickCase:
    """A drawn Kept_Range, as each frame's Motion_Part and Sharpness, with its Candidates."""

    parts: tuple[int, ...]
    candidates: tuple[int, ...]
    sharpness: tuple[float, ...]


@st.composite
def pick_cases(draw: st.DrawFn) -> PickCase:
    """A Kept_Range of 1 to 60 frames whose parts are running sums of PART_STEPS, so they never
    go down, with any set of Candidates among its frames and each Sharpness from a small set.

    Each frame is a Candidate or not, so some parts hold no Candidate, and the Candidates come in
    a drawn order. The small set is 1 to 4 values of ANY_SHARPNESS drawn for the case, so two
    Candidates in a part often share the part's highest Sharpness.
    """
    parts = tuple(accumulate(draw(st.lists(PART_STEPS, min_size=1, max_size=60))))
    chosen = draw(st.lists(st.booleans(), min_size=len(parts), max_size=len(parts)))
    candidates = draw(st.permutations([frame for frame, is_candidate in enumerate(chosen)
                                       if is_candidate]))
    levels = draw(st.lists(ANY_SHARPNESS, min_size=1, max_size=4, unique=True))
    sharpness = draw(st.lists(st.sampled_from(levels), min_size=len(parts),
                              max_size=len(parts)))
    return PickCase(parts=parts, candidates=tuple(candidates), sharpness=tuple(sharpness))


def oracle_picks(case: PickCase) -> list[int]:
    """The Picks of `case`, found from the definitions with plain loops.

    For each part that holds a Candidate, in part order, it takes the part's highest Sharpness
    among its Candidates, then the earliest Candidate with that Sharpness. A part with no
    Candidate gives nothing. The parts never go down, so part order is Frame_Number order, the
    order pick gives its Picks in.
    """
    candidates = set(case.candidates)
    picks: list[int] = []
    for part in sorted(set(case.parts)):
        held = [frame for frame, frame_part in enumerate(case.parts)
                if frame_part == part and frame in candidates]
        if held:
            sharpest = max(case.sharpness[frame] for frame in held)
            picks.append(min(frame for frame in held if case.sharpness[frame] == sharpest))
    return picks


@given(pick_cases())
def test_each_motion_part_gives_its_sharpest_earliest_candidate(case: PickCase) -> None:
    """Feature: frames, Property 8: Each Motion_Part gives its sharpest, earliest Candidate

    **Validates: Requirements 6.5, 6.6**
    """
    # int64 even with no Candidates, as the step's Frame_Numbers are.
    candidates = np.array(case.candidates, dtype=np.int64)
    picks = pick(np.array(case.parts), candidates, np.array(case.sharpness))
    assert picks.tolist() == oracle_picks(case)


@dataclass(frozen=True)
class UsedPartCase:
    """A drawn Used_Part, as its Score_Frames' Frame_Times, Motions and Sharpness values and its
    end, with a Kept_Range, the Candidates and the Target_Picks n.

    The Used_Part runs from 0 s, one gap before its first Score_Frame, to `end`, one gap after
    its last. `first`, `last` and `candidates` are Frame_Numbers of the Used_Part, counted from 0
    at its first Score_Frame.
    """

    times: tuple[float, ...]
    motion: tuple[float, ...]
    sharpness: tuple[float, ...]
    end: float
    first: int  # the Kept_Range's first Frame_Number
    last: int  # its last, inclusive
    candidates: tuple[int, ...]
    n: int


@st.composite
def used_part_cases(draw: st.DrawFn) -> UsedPartCase:
    """A Used_Part of 1 to 60 Score_Frames whose times are running sums of GAPS, with its end one
    more gap after the last, a Kept_Range from `first` to `last` inside it, any set of Candidates
    among the Kept_Range's frames, in a drawn order, and an n from 1 to 400.

    Each Score_Frame's Motion is any of MOTIONS and its Sharpness any of SHARPNESS, so ties are
    common. About half the time every Motion after the Kept_Range's first frame is 0 up to its
    last, so the Kept_Range's total is 0 and its parts split its time.
    """
    stamps = tuple(accumulate(draw(st.lists(GAPS, min_size=2, max_size=61))))
    count = len(stamps) - 1  # how many Score_Frames the Used_Part holds
    first = draw(st.integers(0, count - 1))
    last = draw(st.integers(first, count - 1))
    motion = draw(st.lists(MOTIONS, min_size=count, max_size=count))
    if draw(st.booleans()):
        # The total doesn't count the Kept_Range's first frame's own Motion, so it's 0.
        motion[first + 1:last + 1] = [0.0] * (last - first)
    sharpness = draw(st.lists(SHARPNESS, min_size=count, max_size=count))
    kept = range(first, last + 1)
    chosen = draw(st.lists(st.booleans(), min_size=len(kept), max_size=len(kept)))
    candidates = draw(st.permutations([frame for frame, is_candidate
                                       in zip(kept, chosen, strict=True) if is_candidate]))
    # Often 10 or less, so the Candidates of a long Kept_Range outnumber it and the bound of n
    # Picks bites.
    n = draw(st.integers(1, 400) | st.integers(1, 10))
    return UsedPartCase(times=stamps[:-1], motion=tuple(motion), sharpness=tuple(sharpness),
                        end=stamps[-1], first=first, last=last, candidates=tuple(candidates),
                        n=n)


@given(used_part_cases())
def test_picks_are_few_different_inside_the_kept_range_and_in_order(case: UsedPartCase) -> None:
    """Feature: frames, Property 9: Picks are few, different, inside the Kept_Range and in order

    **Validates: Requirements 6.7, 6.8**
    """
    kept = slice(case.first, case.last + 1)
    # The Kept_Range ends at the time of the next frame after it, or at the Used_Part's end when
    # it holds the last frame.
    kept_end = case.times[case.last + 1] if case.last + 1 < len(case.times) else case.end
    parts = motion_parts(np.array(case.times)[kept], np.array(case.motion)[kept], kept_end,
                         case.n)
    # As in the step, pick counts frames from 0 at the Kept_Range's first frame, so adding
    # `first` turns its Picks back into Frame_Numbers of the Used_Part.
    candidates = np.array(case.candidates, dtype=np.int64) - case.first
    picks = pick(parts, candidates, np.array(case.sharpness)[kept])
    frames = [case.first + index for index in picks.tolist()]
    assert len(frames) <= case.n
    # Different Candidates, inside the Kept_Range.
    assert len(set(frames)) == len(frames)
    assert set(frames) <= set(case.candidates)
    assert all(case.first <= frame <= case.last for frame in frames)
    # Inside the Used_Part: each is one of its Score_Frames, with a Frame_Time from its start,
    # 0 s, to before its end.
    assert all(0 <= frame < len(case.times) for frame in frames)
    assert all(0 <= case.times[frame] < case.end for frame in frames)
    assert all(case.times[before] < case.times[after] for before, after in pairwise(frames))
