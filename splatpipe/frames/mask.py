"""The Sample_Buffer and the Mask, as pure maths on arrays, with no tool and no file.

A frame here is a Score_Frame: a 2-D uint8 array of brightness from 0 to 255.

- The Sample_Buffer (Req 1.12, A15) keeps Score_Frames in memory during the Score_Pass for the
  Mask, with no temporary file. It numbers the frames it's given from 0 and keeps every k-th one
  by Frame_Number, with k = 1 to start. Whenever it then holds more than twice `samples` frames,
  it keeps every other one from the first and doubles k. So after N frames it holds exactly the
  Frame_Numbers of range(0, N, k): all N while N is at most 2 × samples, and from samples + 1 to
  2 × samples after that. It keeps a reference to each frame and never copies one, so it relies
  on every frame it's given being a new array, as the ffmpeg decode gives them.
- The Mask_Samples (Req 7.1) are the buffered frames from the Kept_Range's first Frame_Number to
  its last. When more than `samples` lie there, they're `samples` of them at the places spread
  gives: evenly spread, rounded half up in integers, with the first and the last included.
- The Pixel_Change (Req 7.2) is each pixel's mean absolute change in brightness from one
  Mask_Sample to the next, in Frame_Time order, with the changes added up exactly as int32.
- The Stuck_Pixels (Req 7.3) are the pixels whose Pixel_Change is below mask_ratio times the
  median Pixel_Change of every pixel. So when the median is 0, as in a still scene, no pixel is
  stuck.
- The Mask is used (Req 7.4, 7.5, 7.7) when there are at least 2 Mask_Samples and the
  Stuck_Pixels cover from mask_min_coverage to mask_max_coverage of the frame, both included.
  Nothing grows, shrinks or smooths the Stuck_Pixels (Req 7.6).
- The Mask image (Req 7.8, 7.10) is at the Pick_Size. Each of its pixels takes the value of the
  Score pixel whose area holds the pixel's centre, through integer index arrays, so there's no
  interpolation: 0, black, for a Stuck_Pixel and 255, white, for every other pixel. COLMAP's
  camera mask and the training loss both ignore the black pixels.

This module imports only numpy and the standard library.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise

import numpy as np

# The Mask's two values, Fixed_Constants: black tells COLMAP and the training loss to ignore a
# pixel, and white tells them to use it (Req 7.8, 7.10).
_IGNORE = np.uint8(0)
_USE = np.uint8(255)


class SampleBuffer:
    """Every k-th Score_Frame by Frame_Number, held in memory for the Mask (Req 1.12, A15).

    `samples` is mask_samples, 2 or more. The buffer holds at most twice that many frames once
    `add` returns.
    """

    def __init__(self, samples: int) -> None:
        self._samples = samples
        self._held: list[tuple[int, np.ndarray]] = []  # (Frame_Number, frame), increasing
        self._given = 0  # how many frames add has been given, so the next Frame_Number
        self._stride = 1  # k

    def add(self, frame: np.ndarray) -> None:
        """Take the next Score_Frame, in Frame_Number order, and keep it when its Frame_Number
        is a multiple of k. When that leaves more than 2 × samples pairs, keep every other pair
        from the first and double k.
        """
        number = self._given
        self._given += 1
        if number % self._stride != 0:
            return
        self._held.append((number, frame))
        if len(self._held) > 2 * self._samples:
            self._held = self._held[::2]
            self._stride *= 2

    @property
    def stride(self) -> int:
        """k, a power of 2: the buffer holds the Frame_Numbers that are multiples of k."""
        return self._stride

    @property
    def numbers(self) -> list[int]:
        """The Frame_Numbers the buffer holds, increasing."""
        return [number for number, _ in self._held]

    def samples_in(self, first: int, last: int) -> list[tuple[int, np.ndarray]]:
        """The Mask_Samples of a Kept_Range from Frame_Number `first` to `last`, inclusive.

        They're the held (Frame_Number, frame) pairs whose Frame_Number is from `first` to
        `last`, in order: all of them when there are at most `samples`, and otherwise the
        `samples` of them at the places spread gives, the first and the last included (Req 7.1).
        """
        inside = [(number, frame) for number, frame in self._held if first <= number <= last]
        return [inside[place] for place in spread(len(inside), self._samples)]


def spread(count: int, samples: int) -> list[int]:
    """The places to keep among `count` items in order, at most `samples` of them (Req 7.1).

    Every place from 0 to count − 1 when count is at most samples. Otherwise `samples` places,
    place i being i × (count − 1) / (samples − 1) rounded half up in integers. So the first is
    0, the last is count − 1, they all differ, and the gaps between neighbours differ by at most
    1. `samples` is 2 or more.
    """
    if count <= samples:
        return list(range(count))
    steps = samples - 1
    return [(i * (count - 1) + steps // 2) // steps for i in range(samples)]


def pixel_change(frames: Sequence[np.ndarray]) -> np.ndarray | None:
    """Each pixel's Pixel_Change over the Mask_Samples `frames`, in Frame_Time order (Req 7.2).

    The absolute change in brightness from each sample to the next is added up exactly as int32,
    then divided by the number of pairs, which gives float64. None with fewer than 2 samples.
    """
    if len(frames) < 2:
        return None
    total = np.zeros(frames[0].shape, dtype=np.int32)
    for previous, current in pairwise(frames):
        total += np.abs(current.astype(np.int32) - previous)
    return total / (len(frames) - 1)


def stuck_pixels(change: np.ndarray, ratio: float) -> np.ndarray:
    """The Stuck_Pixels, as a bool array of the Pixel_Change's shape (Req 7.3).

    A pixel is stuck when its Pixel_Change is below `ratio` times the median Pixel_Change of
    every pixel. A Pixel_Change is never below 0, so when the median is 0 no pixel is stuck.
    """
    return change < ratio * float(np.median(change))


@dataclass(frozen=True)
class MaskChoice:
    """Whether the step uses the Mask, and what decided it (Req 7.4–7.7, 7.9)."""

    samples: int  # how many Mask_Samples
    stuck: np.ndarray | None  # bool at the Score_Frame's size, None with < 2 samples
    coverage: float  # the Mask_Coverage, from 0 to 1, 0.0 with < 2 samples
    used: bool


def decide_mask(frames: Sequence[np.ndarray], *, ratio: float, min_coverage: float,
                max_coverage: float) -> MaskChoice:
    """Find the Stuck_Pixels of the Mask_Samples `frames` and decide whether to use the Mask.

    With fewer than 2 samples there's no Mask and the coverage is 0 (Req 7.7). Otherwise the
    Mask_Coverage is the Stuck_Pixels' share of the frame, and the Mask is used when that's from
    `min_coverage` to `max_coverage`, both included (Req 7.4, 7.5). The Stuck_Pixels stay as
    found, with nothing grown, shrunk or smoothed (Req 7.6).
    """
    change = pixel_change(frames)
    if change is None:
        return MaskChoice(samples=len(frames), stuck=None, coverage=0.0, used=False)
    stuck = stuck_pixels(change, ratio)
    coverage = float(stuck.mean())
    return MaskChoice(samples=len(frames), stuck=stuck, coverage=coverage,
                      used=min_coverage <= coverage <= max_coverage)


def mask_image(stuck: np.ndarray, width: int, height: int) -> np.ndarray:
    """The Mask image at the Pick_Size `width` × `height`, from the Stuck_Pixels (Req 7.8).

    `stuck` is at the Score_Frame's size, sh × sw, which may be larger or smaller than the
    Pick_Size. Output pixel (x, y) takes the Score pixel at row (2y + 1) × sh // (2 × height)
    and column (2x + 1) × sw // (2 × width), the one whose area holds the output pixel's
    centre, through integer index arrays. Gives a uint8 image of 0 for each Stuck_Pixel and 255
    for every other pixel, with no other value.
    """
    stuck_height, stuck_width = stuck.shape
    rows = (2 * np.arange(height) + 1) * stuck_height // (2 * height)
    columns = (2 * np.arange(width) + 1) * stuck_width // (2 * width)
    return np.where(stuck[np.ix_(rows, columns)], _IGNORE, _USE)
