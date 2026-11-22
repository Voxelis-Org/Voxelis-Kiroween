"""The frame scores and frame sizes, as pure maths on arrays, with no tool and no file.

A frame here is a 2-D uint8 array of brightness from 0 to 255, a scale that is a Fixed_Constant.
Each Score_Frame gets three scores:

- Sharpness (Req 3.2): the variance of OpenCV's 3 × 3 Laplacian (ksize=1) with OpenCV's default
  border, which is reflect-101. The kernel and the border are Fixed_Constants. A variance is
  never below 0.
- Motion (Req 3.3, 3.4): the mean absolute change in brightness from the Score_Frame before,
  from the exact integer sum of the changes, so it's never above 255. With no frame before, as
  for the first Score_Frame, it's 0.
- Contrast (Req 3.5): the standard deviation of the brightness, from the 256-bin histogram with
  Python ints. Its numerator is exact, so it's never above 127.5, and a frame that's half 0 and
  half 255 gives exactly 127.5 (Req 3.7).

The Scorer takes the Score_Frames one at a time, scores each against the one before and then
keeps a reference to that frame alone. So whatever the stream's length, it holds at most two
frames: its own previous one and the caller's current one (Req 1.12). It keeps each frame's
Frame_Time and scores, in the order they came (Req 3.6).

score_shape and pick_size work out sizes from the Displayed_Shape in integers, rounding half up,
and never give an edge below 1 pixel. The Score_Frame is always score_width wide, so a video
narrower than that is scaled up for scoring (Req 3.1). The Pick_Size is the video's own size
when its long edge is at most long_edge, so a Pick is never larger than the video (Req 8.1).

This module imports only numpy, cv2 and the standard library.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

# The 0 to 255 brightness scale, a Fixed_Constant: each histogram bin's level and its square.
_LEVELS = np.arange(256, dtype=np.int64)
_SQUARES = _LEVELS * _LEVELS


def sharpness(frame: np.ndarray) -> float:
    """The variance of the frame's Laplacian: OpenCV's 3 × 3 kernel with its default border."""
    return float(cv2.Laplacian(frame, cv2.CV_64F, ksize=1).var())


def motion(previous: np.ndarray | None, frame: np.ndarray) -> float:
    """The mean absolute change in brightness from `previous` to `frame`, or 0 with no previous.

    The changes are worked out in int16 and summed as int64, so the sum is an exact integer.
    """
    if previous is None:
        return 0.0
    total = np.abs(frame.astype(np.int16) - previous).sum(dtype=np.int64)
    return int(total) / frame.size


def contrast(frame: np.ndarray) -> float:
    """The standard deviation of the frame's brightness, from its 256-bin histogram.

    With n pixels whose levels add up to s1 and whose squared levels add up to s2, all as Python
    ints, the numerator n·s2 − s1² is exact and never below 0.
    """
    counts = np.bincount(frame.ravel(), minlength=_LEVELS.size)
    n = frame.size
    s1 = int(counts @ _LEVELS)
    s2 = int(counts @ _SQUARES)
    return math.sqrt(n * s2 - s1 * s1) / n


@dataclass(frozen=True)
class Scores:
    """Each Score_Frame's Frame_Time and scores, in Frame_Number order, as float64 arrays."""

    times: np.ndarray  # Frame_Time, strictly increasing
    sharpness: np.ndarray
    motion: np.ndarray  # motion[0] == 0
    contrast: np.ndarray


class Scorer:
    """Scores the Score_Frames one at a time, holding only the frame before (Req 1.12)."""

    def __init__(self) -> None:
        self._previous: np.ndarray | None = None
        self._times: list[float] = []
        self._sharpness: list[float] = []
        self._motion: list[float] = []
        self._contrast: list[float] = []

    def add(self, time: float, frame: np.ndarray) -> None:
        """Score `frame` against the frame before, then keep a reference to this frame alone."""
        frame_sharpness = sharpness(frame)
        frame_motion = motion(self._previous, frame)
        frame_contrast = contrast(frame)
        self._times.append(time)
        self._sharpness.append(frame_sharpness)
        self._motion.append(frame_motion)
        self._contrast.append(frame_contrast)
        self._previous = frame

    def result(self) -> Scores:
        """The Frame_Times and scores of every frame added so far, in the order they came."""
        return Scores(times=np.array(self._times, dtype=np.float64),
                      sharpness=np.array(self._sharpness, dtype=np.float64),
                      motion=np.array(self._motion, dtype=np.float64),
                      contrast=np.array(self._contrast, dtype=np.float64))


def score_shape(width: int, height: int, score_width: int) -> tuple[int, int]:
    """The Score_Frame's (width, height) for a Displayed_Shape of width × height (Req 3.1).

    The width is score_width, even for a narrower video, which is scaled up for scoring. The
    height is height × score_width / width, rounded half up, and at least 1.
    """
    return score_width, _scaled_edge(height, score_width, width)


def pick_size(width: int, height: int, long_edge: int) -> tuple[int, int]:
    """The Pick_Size (width, height) for a Displayed_Shape of width × height (Req 8.1).

    It's the video's own size when its long edge is at most long_edge, so a Pick is never larger
    than the video. Otherwise the long edge becomes long_edge, and the short edge
    short × long_edge / long, rounded half up, and at least 1.
    """
    if max(width, height) <= long_edge:
        return width, height
    if width >= height:
        return long_edge, _scaled_edge(height, long_edge, width)
    return _scaled_edge(width, long_edge, height), long_edge


def _scaled_edge(edge: int, new: int, old: int) -> int:
    """edge × new / old, rounded half up in integers, and at least 1 pixel."""
    return max(1, (2 * edge * new + old) // (2 * old))
