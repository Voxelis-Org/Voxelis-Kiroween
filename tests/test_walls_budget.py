"""Unit tests for the Walls_Step's time budget and its Solve_Order: splatpipe/walls/budget.py.

Every time is passed in as a timezone-aware UTC datetime, and the Solve_Cutoff tests move a fake
clock across it, so no test reads the real clock (Req 4.10).
"""

from datetime import datetime, timedelta, timezone

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from splatpipe.walls.budget import (MAX_MINUTES, TRAIN_RESERVE_MINUTES, may_start, solve_cutoff,
                                    solve_order, walls_times)

# The step's start, and the limits the examples use.
START = datetime(2026, 11, 21, 12, 0, 0, tzinfo=timezone.utc)
MAX = 60.0  # max_minutes: the step's own limit gives START + 60 min = 13:00
RESERVE = 40.0  # train_reserve_minutes
FINISH = 3.0  # finish_minutes
OWN_LIMIT = datetime(2026, 11, 21, 13, 0, 0, tzinfo=timezone.utc)
TICK = timedelta(microseconds=1)


class FakeClock:
    """A clock that says the time it was set to, and moves only when told to."""

    def __init__(self, now: datetime) -> None:
        self._now = now

    def now(self) -> datetime:
        return self._now

    def set(self, now: datetime) -> None:
        self._now = now


# ---- walls_times (Req 4.2) ----

def test_no_cap_deadline_gives_start_plus_max_minutes() -> None:
    """With no Cap_Deadline the Walls_Stop_Time is the start + max_minutes, set by max_minutes."""
    assert walls_times(START, None, MAX, RESERVE) == (OWN_LIMIT, MAX_MINUTES)


def test_far_cap_deadline_gives_start_plus_max_minutes() -> None:
    """A Cap_Deadline whose reserve comes after the step's own limit changes nothing: 14:00 − 40
    min is 13:20, later than 13:00."""
    cap = datetime(2026, 11, 21, 14, 0, 0, tzinfo=timezone.utc)
    assert walls_times(START, cap, MAX, RESERVE) == (OWN_LIMIT, MAX_MINUTES)


def test_near_cap_deadline_gives_cap_deadline_minus_reserve() -> None:
    """A Cap_Deadline whose reserve comes first sets the Walls_Stop_Time: 13:30 − 40 min is
    12:50, earlier than 13:00, set by train_reserve_minutes."""
    cap = datetime(2026, 11, 21, 13, 30, 0, tzinfo=timezone.utc)
    stop = datetime(2026, 11, 21, 12, 50, 0, tzinfo=timezone.utc)
    assert walls_times(START, cap, MAX, RESERVE) == (stop, TRAIN_RESERVE_MINUTES)


def test_equal_limits_go_to_max_minutes() -> None:
    """When both limits give 13:00 (13:40 − 40 min), the tie goes to max_minutes."""
    cap = datetime(2026, 11, 21, 13, 40, 0, tzinfo=timezone.utc)
    assert walls_times(START, cap, MAX, RESERVE) == (OWN_LIMIT, MAX_MINUTES)


def test_limit_names_are_the_setting_names() -> None:
    """The names the Walls_Record holds are the [walls] setting names (Req 4.11)."""
    assert (MAX_MINUTES, TRAIN_RESERVE_MINUTES) == ("max_minutes", "train_reserve_minutes")


# ---- solve_cutoff and may_start (Req 4.3, 4.5, 4.10) ----

def test_solve_cutoff_is_finish_minutes_before_the_stop_time() -> None:
    """The Solve_Cutoff is the Walls_Stop_Time − finish_minutes: 13:00 − 3 min is 12:57."""
    cutoff = datetime(2026, 11, 21, 12, 57, 0, tzinfo=timezone.utc)
    assert solve_cutoff(OWN_LIMIT, FINISH) == cutoff


def test_may_start_only_before_the_solve_cutoff() -> None:
    """A fake clock just before the Solve_Cutoff lets a Frame_Solve start, and one at it or just
    after it doesn't (Req 4.5)."""
    stop, _ = walls_times(START, None, MAX, RESERVE)
    cutoff = solve_cutoff(stop, FINISH)
    clock = FakeClock(START)
    assert may_start(clock.now(), cutoff)
    clock.set(cutoff - TICK)
    assert may_start(clock.now(), cutoff)
    clock.set(cutoff)
    assert not may_start(clock.now(), cutoff)
    clock.set(cutoff + TICK)
    assert not may_start(clock.now(), cutoff)


# ---- solve_order (Req 4.4) ----

@pytest.mark.parametrize(("n", "order"), [
    (1, [0]),
    (2, [0, 1]),
    (3, [0, 2, 1]),
    (7, [0, 6, 3, 1, 4, 2, 5]),
])
def test_solve_order_spells_out_small_cases(n: int, order: list[int]) -> None:
    """First, last, middle, then the quarters: for N = 7 the queue gives 3 from (0, 6), then 1
    from (0, 3) and 4 from (3, 6), then 2 from (1, 3) and 5 from (4, 6)."""
    assert solve_order(n) == order


@pytest.mark.parametrize("n", [0, -1])
def test_solve_order_refuses_no_prior_frames(n: int) -> None:
    """There's no Solve_Order of fewer than one Prior_Frame."""
    with pytest.raises(ValueError, match="at least one Prior_Frame"):
        solve_order(n)


# ---- Property 5 (Req 4.12) ----

# N from 1 to 1,000, with the smallest cases 1, 2 and 3 drawn often.
PRIOR_FRAME_COUNTS = st.one_of(st.sampled_from([1, 2, 3]), st.integers(min_value=1, max_value=1000))


@settings(max_examples=100, deadline=None)
@given(n=PRIOR_FRAME_COUNTS)
def test_solve_order_lists_every_index_once(n: int) -> None:
    """Feature: walls, Property 5: The Solve_Order lists every index once.

    For any N from 1 to 1,000, the Solve_Order lists every index from 0 to N − 1 exactly once,
    starting with 0 and, when N is above 1, then N − 1.

    **Validates: Requirements 4.12**
    """
    order = solve_order(n)
    assert sorted(order) == list(range(n))
    assert order[0] == 0
    if n > 1:
        assert order[1] == n - 1
