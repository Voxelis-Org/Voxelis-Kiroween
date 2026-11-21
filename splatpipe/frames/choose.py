"""The Still_Trim and the Jolt_Filter, as pure maths on arrays, with no tool and no file.

Each function takes the Score_Frames' Frame_Times and scores as arrays in Frame_Number order,
and the Frame_Times strictly increase.

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

This module imports only numpy and the standard library.
"""

from __future__ import annotations

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
