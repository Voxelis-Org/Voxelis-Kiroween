"""The Still_Trim, the Jolt_Filter, Target_Picks and the Picker, as pure maths on arrays, with no
tool and no file.

Each function that takes frames takes the Score_Frames' Frame_Times and scores as arrays in
Frame_Number order, and the Frame_Times strictly increase.

- Windows (A2): a frame's Window of a length L holds every frame whose Frame_Time is at most
  L / 2 before or after the frame's own, the frame itself included. window_bounds finds each
  Window as an index range with searchsorted.
- The Still_Trim (Req 4): a Still_Frame is a frame whose Window of motion_window_seconds, taken
  from the whole Used_Part, has a mean Motion below still_motion. Each maximal run of frames that
  aren't still is a Moving_Stretch when it lasts blip_seconds or more, and otherwise a Blip,
  which counts as still (A1). A Stretch lasts from its first frame's Frame_Time to the Frame_Time
  of the next frame after it, or to the Used_Part's end when no frame comes after it. The
  Still_Start and the Still_End are cut only when each lasts min_still_seconds or more. So the
  Kept_Range is always one Stretch that holds every Moving_Stretch, and a clip with no
  Moving_Stretch is kept whole (Req 4.7, D2).
- The Jolt_Filter (Req 5.1): a Kept_Range frame is kept when its Sharpness is at least
  jolt_ratio times the median Sharpness of its Window of jolt_window_seconds, taken from the
  Kept_Range only (A2).
- Target_Picks (Req 6.1, A3): one Pick for every seconds_per_pick of Kept_Seconds, rounded up,
  but at least the Preset's base_picks and at most max_picks.
- The Motion_Parts (Req 6.3, 6.4): a Kept_Range frame's Motion_Position is the running total of
  Motion from the Kept_Range's first frame, with the first frame's own Motion not counted. The
  Target_Picks parts split the total evenly, and a frame at Motion_Position pos is in part
  min(n − 1, floor(pos × n / total)). When the total is 0, the parts split the time from the
  first frame to the Kept_Range's end instead (A5).
- The Picker (Req 6.5–6.7): each Motion_Part gives its Candidate with the highest Sharpness, the
  earliest of them on a tie, and a part with no Candidate gives no Pick (A5). The parts never go
  down, so the Picks come in Frame_Time order, at most one per part and all different (Req 6.8).

This module imports only numpy and the standard library.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from itertools import groupby
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Kept:
    """The Kept_Range the Still_Trim gives, with the seconds it cut at each end (Req 4.10)."""

    first: int  # a: the Kept_Range's first Frame_Number
    last: int  # b: its last, inclusive
    start_seconds: float  # times[a]
    end_seconds: float  # times[b + 1], or the Used_Part's end when b is the last frame
    cut_start_seconds: float  # the Still_Start's length when cut, else 0.0
    cut_end_seconds: float  # the Still_End's length when cut, else 0.0


def window_bounds(times: np.ndarray, seconds: float) -> tuple[np.ndarray, np.ndarray]:
    """Each frame's Window of `seconds`, as the index range lo[i]:hi[i] into `times` (A2).

    The Window of frame i holds every frame j with
    times[i] − seconds / 2 <= times[j] <= times[i] + seconds / 2, so frame i is always inside.
    """
    half = seconds / 2
    lo = np.searchsorted(times, times - half, side="left")
    hi = np.searchsorted(times, times + half, side="right")
    return lo, hi


def still_trim(times: np.ndarray, motion: np.ndarray, end: float, *, still_motion: float,
               window_seconds: float, min_still_seconds: float, blip_seconds: float) -> Kept:
    """Cut the Still_Start and the Still_End off the Used_Part's Score_Frames (Req 4).

    `times` and `motion` are the Frame_Times and Motions of every Score_Frame of the Used_Part,
    at least one, and `end` is the Used_Part's end, after the last Frame_Time. The Still_Start
    and the Still_End are each cut when they last `min_still_seconds` or more, and every frame
    is kept when there's no Moving_Stretch.
    """
    still = _over_windows(times, motion, window_seconds, np.mean) < still_motion
    after = np.append(times[1:], end)  # after[q]: the next frame's Frame_Time, or the end
    moving = [(p, q) for p, q in _runs(~still) if after[q] - times[p] >= blip_seconds]
    if not moving:
        return _kept(times, after, 0, len(times) - 1, 0.0, 0.0)
    first_moving, last_moving = moving[0][0], moving[-1][1]
    still_start = float(times[first_moving] - times[0])
    still_end = float(end - after[last_moving])  # 0 when the last Moving_Stretch ends the clip
    cut_start = still_start >= min_still_seconds
    cut_end = still_end >= min_still_seconds
    return _kept(times, after, first_moving if cut_start else 0,
                 last_moving if cut_end else len(times) - 1,
                 still_start if cut_start else 0.0, still_end if cut_end else 0.0)


def jolt_keep(times: np.ndarray, sharpness: np.ndarray, *, ratio: float,
              window_seconds: float) -> np.ndarray:
    """Which Kept_Range frames the Jolt_Filter keeps, as one bool per frame given (Req 5.1).

    `times` and `sharpness` are the Kept_Range's Frame_Times and Sharpness values, so each
    Window takes its frames from the Kept_Range only. A frame is kept when its Sharpness is at
    least `ratio` times the median Sharpness of its Window of `window_seconds`, so every frame
    is kept when every Sharpness is 0.
    """
    return sharpness >= ratio * _over_windows(times, sharpness, window_seconds, np.median)


def _over_windows(times: np.ndarray, values: np.ndarray, seconds: float,
                  stat: Callable[[np.ndarray], Any]) -> np.ndarray:
    """`stat` of `values` over each frame's Window of `seconds`, as float64, one per frame."""
    lo, hi = window_bounds(times, seconds)
    return np.array([float(stat(values[low:high])) for low, high in zip(lo, hi)],
                    dtype=np.float64)


def _runs(flags: np.ndarray) -> list[tuple[int, int]]:
    """The first and last index of each maximal run of True values in `flags`, in order."""
    runs: list[tuple[int, int]] = []
    start = 0
    for flag, group in groupby(flags.tolist()):
        size = sum(1 for _ in group)
        if flag:
            runs.append((start, start + size - 1))
        start += size
    return runs


def _kept(times: np.ndarray, after: np.ndarray, first: int, last: int, cut_start: float,
          cut_end: float) -> Kept:
    """The Kept_Range from frame `first` to frame `last`, as plain Python ints and floats."""
    return Kept(first=int(first), last=int(last), start_seconds=float(times[first]),
                end_seconds=float(after[last]), cut_start_seconds=float(cut_start),
                cut_end_seconds=float(cut_end))


def target_picks(kept_seconds: float, *, base_picks: int, seconds_per_pick: float,
                 max_picks: int) -> int:
    """Target_Picks for a Kept_Range that lasts `kept_seconds` (Req 6.1, A3).

    One Pick for every `seconds_per_pick` of Kept_Seconds, rounded up, but at least `base_picks`,
    the Preset's, and at most `max_picks`.
    """
    return min(max_picks, max(base_picks, math.ceil(kept_seconds / seconds_per_pick)))


def motion_parts(times: np.ndarray, motion: np.ndarray, kept_end: float, n: int) -> np.ndarray:
    """Each Kept_Range frame's Motion_Part, a whole number from 0 to n − 1 (Req 6.3, 6.4).

    `times` and `motion` are the Kept_Range's Frame_Times and Motions, at least one frame,
    `kept_end` is the Kept_Range's end, after its last Frame_Time, and n is Target_Picks. A
    frame's Motion_Position is the running total of Motion from the Kept_Range's first frame,
    with the first frame's own Motion not counted, so the first frame's is 0 and the last
    frame's is the total. When the total is above 0, a frame at Motion_Position pos is in part
    min(n − 1, floor(pos × n / total)). When it's 0, the same formula runs on the frame's time
    since the first frame, against the time from the first frame to `kept_end`. Either way the
    parts never go down, and the first frame is in part 0.
    """
    position = np.concatenate(([0.0], np.cumsum(motion[1:], dtype=np.float64)))
    total = float(position[-1])
    if total > 0:
        return _parts(position, total, n)
    return _parts(times - times[0], kept_end - float(times[0]), n)


def pick(parts: np.ndarray, candidates: np.ndarray, sharpness: np.ndarray) -> np.ndarray:
    """The Picks: each Motion_Part's Candidate with the highest Sharpness (Req 6.5–6.7).

    `parts` and `sharpness` hold one Motion_Part and one Sharpness per frame, for the same frames
    in Frame_Number order, like the Kept_Range's parts from motion_parts. `candidates` holds the
    Candidates' Frame_Numbers among those frames, counted from 0 at the first of them. The walk
    goes through the Candidates in Frame_Number order and keeps, for each part, the first
    Candidate with the highest Sharpness, replacing it only on a strictly higher one, so the
    earliest wins a tie. A part with no Candidate gives no Pick.

    Gives the chosen Frame_Numbers in order, as int64, counted the same way as `candidates`: at
    most one per part and all different. The parts never go down, so that's Frame_Time order.
    """
    part_of, sharp = parts.tolist(), sharpness.tolist()
    best: dict[int, int] = {}  # a part: the Frame_Number of its sharpest Candidate so far
    for frame in np.sort(candidates).tolist():
        part = part_of[frame]
        if part not in best or sharp[frame] > sharp[best[part]]:
            best[part] = frame
    return np.array(sorted(best.values()), dtype=np.int64)


def _parts(position: np.ndarray, total: float, n: int) -> np.ndarray:
    """min(n − 1, floor(position × n / total)) for each frame, as int64, with `total` above 0."""
    return np.minimum(n - 1, np.floor(position * n / total)).astype(np.int64)
