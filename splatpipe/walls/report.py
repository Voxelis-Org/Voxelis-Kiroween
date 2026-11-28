"""What the Walls_Step reports: its six counts, its ten timings, its two warnings and its
no-trusted-frame failure (walls spec, Req 18.8, 19, D21).

- Counts holds the six counts of Req 19.1, named as the report names them. Counts.add sets one
  and gives it to StepContext.add_count as an int straight away, so a failed report holds exactly
  the counts the step had worked out (Req 19.3, 19.4). Once they're all known,
  images = passed_images + fallback_images + depth_prior_images
         = solved_images + timed_out_images.
- Timings holds the ten timings of Req 19.2, in seconds. Timings.timed times a block and
  Timings.add takes seconds the step measured itself, like `solve`, from the first Frame_Solve's
  start to the last one's end or the Solve_Cutoff. Each gives the timing to
  StepContext.add_timing when its block ends.
- timed_out_warning is the Req 19.5 warning, for a run with Timed_Out_Frames, and
  not_all_passed_warning the Req 19.6 warning, for a run in which fewer than every Prior_Frame
  passed its Self_Check.
- no_trusted_failure is the Req 18.8 failure, for a run in which no Prior_Frame passed (A15).

This module reads no file and holds no setting. From splatpipe, it imports only
splatpipe.errors, and splatpipe.step for its types.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field, fields
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from splatpipe.errors import ExpectedFailure

if TYPE_CHECKING:
    from splatpipe.step import StepContext

# The report's timings, in Req 19.2's order.
TIMING_NAMES = ("read", "solve", "planes", "fuse", "fallback", "correct", "flatten", "despeckle",
                "write", "demo")
# What set the Walls_Stop_Time, as splatpipe.walls.budget.walls_times names it, in words.
_STOP_LIMITS = {
    "max_minutes": "the walls step's own max_minutes",
    "train_reserve_minutes": ("the train step's train_reserve_minutes before the cap "
                              "deadline"),
}


@dataclass
class Counts:
    """The Walls_Report's six counts (Req 19.1, D21). A count is None until the step knows it."""

    images: int | None = None  # the Prior_Frames
    solved_images: int | None = None  # the Prior_Frames whose Frame_Solve ended
    passed_images: int | None = None  # the Trusted_Frames
    fallback_images: int | None = None  # the Failed_Frames with a Walls_Prior from a Fallback_Depth
    depth_prior_images: int | None = None  # the Depth_Prior_Copies
    timed_out_images: int | None = None  # the Timed_Out_Frames

    def add(self, step: StepContext, name: str, value: int) -> None:
        """Set the count `name` to `value` and add it to the report as an int (Req 19.4). A name
        that isn't one of the six is a bug, and raises ValueError."""
        if name not in _COUNT_NAMES:
            raise ValueError(f"{name!r} isn't a count of the walls step. The counts are "
                             f"{', '.join(_COUNT_NAMES)}.")
        count = int(value)
        step.add_count(name, count)
        setattr(self, name, count)


# The report's counts: Counts' fields, in Req 19.1's order.
_COUNT_NAMES = tuple(item.name for item in fields(Counts))


@dataclass
class Timings:
    """The Walls_Report's ten timings, in seconds, by name (Req 19.2)."""

    seconds: dict[str, float] = field(default_factory=dict)

    def add(self, step: StepContext, name: str, seconds: float) -> None:
        """Set the timing `name` to `seconds` and add it to the report. A name that isn't one of
        the ten is a bug, and raises ValueError."""
        if name not in TIMING_NAMES:
            raise ValueError(f"{name!r} isn't a timing of the walls step. The timings are "
                             f"{', '.join(TIMING_NAMES)}.")
        value = float(seconds)
        step.add_timing(name, value)
        self.seconds[name] = value

    @contextmanager
    def timed(self, step: StepContext, name: str) -> Iterator[None]:
        """Time the block with time.perf_counter, then add its seconds as the timing `name`. A
        block that raises adds no timing, so a failed report holds only the blocks that ended."""
        started = time.perf_counter()
        yield
        self.add(step, name, time.perf_counter() - started)


def timed_out_warning(timed_out: int, images: int, stop_time: datetime, stop_limit: str) -> str:
    """The Req 19.5 warning: the Timed_Out_Frames, the Prior_Frames, the Walls_Stop_Time in UTC,
    and whether `max_minutes` or the Train_Step's `train_reserve_minutes` before the Cap_Deadline
    set it (`stop_limit`, as walls_times gives it), and that those frames keep their depth
    prior."""
    return (f"The walls step ran out of time for {timed_out} of {_frames(images)}: it had to "
            f"stop solving frames in time to be done by {_utc(stop_time)}, which "
            f"{_limit(stop_limit)} set. Those frames keep their depth prior.")


def not_all_passed_warning(passed: int, fallback: int, depth_prior: int, images: int) -> str:
    """The Req 19.6 warning: the Trusted_Frames, the Prior_Frames with a Walls_Prior from a
    Fallback_Depth and the Depth_Prior_Copies, out of the Prior_Frames."""
    return (f"{passed} of {_frames(images)} passed the walls step's self-check and "
            f"{_were(passed)} corrected from their own solve. Of the others, {fallback} got a "
            f"walls prior from a fallback depth and {depth_prior} kept their depth prior "
            "unchanged.")


def no_trusted_failure(images: int, timed_out: int, failed: int,
                       too_few_sources: int) -> ExpectedFailure:
    """The Req 18.8 failure for a run in which no Prior_Frame passed its Self_Check: the
    Prior_Frames, how many timed out, how many failed their Self_Check and, of those, how many
    had too few Source_Frames, and that the Job goes on and trains on the depth priors (A15)."""
    return ExpectedFailure(
        f"None of the {_frames(images)} passed the walls step's self-check, so no walls priors "
        f"were made: {timed_out} ran out of time before their solve ended, and {failed} failed "
        f"their self-check, {too_few_sources} of them because they had too few other frames to "
        "compare with. The job goes on and trains on the depth priors.")


def _frames(count: int) -> str:
    """A count of frames in words, like "1 frame" or "20 frames"."""
    return f"{count} frame" if count == 1 else f"{count} frames"


def _were(count: int) -> str:
    """"was" for one, "were" for any other count."""
    return "was" if count == 1 else "were"


def _utc(moment: datetime) -> str:
    """A time in UTC, to the second, like "2026-11-19 14:25:31 UTC". A naive time is refused with
    ValueError, because it says no time zone."""
    if moment.tzinfo is None or moment.utcoffset() is None:
        raise ValueError("timed_out_warning: the stop time has no time zone")
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _limit(stop_limit: str) -> str:
    """What set the Walls_Stop_Time, in words. Any other name is a bug, and raises ValueError."""
    if stop_limit not in _STOP_LIMITS:
        raise ValueError(f"{stop_limit!r} isn't what sets the walls stop time. It's one of "
                         f"{', '.join(_STOP_LIMITS)}.")
    return _STOP_LIMITS[stop_limit]
