"""The Walls_Step's time budget and its Solve_Order, as pure functions that take the time as an
argument.

The Walls_Stop_Time is the earlier of the step's start + max_minutes and the Cap_Deadline −
train_reserve_minutes, or the start + max_minutes when no Cap_Deadline is set (Req 4.2, D18).
walls_times also says which of the two limits set it, a tie going to max_minutes, the step's own
limit, so the Walls_Record can hold it (Req 4.11). The Solve_Cutoff is finish_minutes before the
Walls_Stop_Time (Req 4.3), and the step starts a Frame_Solve only while may_start(now, cutoff)
holds (Req 4.5). The step reads its start once, when its work starts, and passes it and every
later time in, so the tests pass them a fake clock (Req 4.10).

The Solve_Order spreads the Frame_Solves over the whole video: first, last, middle, then the
quarters and so on, so a run that's cut off still covers all of it (Req 4.4, D17).

This module imports only the standard library and reads no clock.
"""

from __future__ import annotations

from collections import deque
from datetime import datetime, timedelta

MAX_MINUTES = "max_minutes"  # the step's own limit set the Walls_Stop_Time
TRAIN_RESERVE_MINUTES = "train_reserve_minutes"  # the Cap_Deadline's reserve set it


def walls_times(start: datetime, cap_deadline: datetime | None, max_minutes: float,
                train_reserve_minutes: float) -> tuple[datetime, str]:
    """The Walls_Stop_Time and the name of the limit that set it (Req 4.2, D18).

    It is the earlier of `start` + `max_minutes` and `cap_deadline` − `train_reserve_minutes`, or
    `start` + `max_minutes` when `cap_deadline` is None. The name is "max_minutes" or
    "train_reserve_minutes", and a tie goes to "max_minutes".
    """
    own_limit = start + timedelta(minutes=max_minutes)
    if cap_deadline is None:
        return own_limit, MAX_MINUTES
    reserve_limit = cap_deadline - timedelta(minutes=train_reserve_minutes)
    if reserve_limit < own_limit:
        return reserve_limit, TRAIN_RESERVE_MINUTES
    return own_limit, MAX_MINUTES


def solve_cutoff(stop: datetime, finish_minutes: float) -> datetime:
    """The Solve_Cutoff: the Walls_Stop_Time `stop` minus `finish_minutes` (Req 4.3)."""
    return stop - timedelta(minutes=finish_minutes)


def may_start(now: datetime, cutoff: datetime) -> bool:
    """Whether a Frame_Solve may start at `now`: only before the Solve_Cutoff, never at or after
    it (Req 4.5)."""
    return now < cutoff


def solve_order(n: int) -> list[int]:
    """The Solve_Order of `n` Prior_Frames, indexed 0 to `n` − 1 by Pick_Number (Req 4.4, D17).

    First 0, then `n` − 1 when `n` is above 1. Then, from a first-in-first-out queue that starts
    with the interval (0, `n` − 1), each interval (lo, hi) taken from it with hi − lo of 2 or more
    gives its midpoint ⌊(lo + hi) / 2⌋ and puts (lo, mid) and (mid, hi) at the end of the queue,
    until the queue is empty. Every index comes exactly once, for every `n` of 1 or more.
    """
    if n < 1:
        raise ValueError(f"solve_order needs at least one Prior_Frame, but it was given n = {n}.")
    order = [0] if n == 1 else [0, n - 1]
    queue: deque[tuple[int, int]] = deque([(0, n - 1)])
    while queue:
        lo, hi = queue.popleft()
        if hi - lo < 2:
            continue
        mid = (lo + hi) // 2
        order.append(mid)
        queue.append((lo, mid))
        queue.append((mid, hi))
    return order
