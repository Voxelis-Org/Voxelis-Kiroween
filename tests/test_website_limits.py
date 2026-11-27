"""The Website_Limits: the Worst_Case_Cost, each limit's refusal, the UTC_Day, give-backs, the
Stop_Time, and Property 1, that the limits are never passed."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from hypothesis import settings
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, initialize, invariant, precondition, rule

from app.errors import PublicRefusal
from app.limits import (TOTAL, LimitFull, Reservation, check_room, day_key, give_back, job_key,
                        stop_line, take, upload_key, utc_day, worst_case_cents)
from cloud.config import load_cloud_config
from cloud.start import worst_case_cost
from app.config import dollars_to_cents
from tests.fake_aws import FakeAws
from tests.website_helpers import REAL_CONFIG, T0, TestClock, website_config

CC = load_cloud_config(REAL_CONFIG)


def seed_upload(fake: FakeAws, account: str, upload_id: str) -> None:
    key = upload_key(account, upload_id)
    fake.items[(key.pk, key.sk)] = {"pk": key.pk, "sk": key.sk, "state": "complete"}


def draft(n: int, cents: int, account: str = "judges") -> Reservation:
    return Reservation(job_id=f"20261201-120000-{n:06x}", account=account,
                       upload_id=f"{n:032x}", preset="fast", cents=cents)


def taken(fake: FakeAws, n: int, cents: int, clock: TestClock, wc) -> Reservation:
    seed_upload(fake, "judges", f"{n:032x}")
    return take(fake, draft(n, cents), 5, clock, wc)


def test_the_worst_case_cents_with_todays_config() -> None:
    wc = website_config()
    assert [worst_case_cents(p, CC, wc) for p in ("fast", "default", "quality")] == [515, 762, 1841]
    assert [dollars_to_cents(worst_case_cost(p, CC)) for p in ("fast", "default", "quality")] == [
        494, 741, 1815]


def test_a_reservation_writes_its_counters_job_and_upload_in_one_transaction() -> None:
    fake, wc, clock = FakeAws(), website_config(), TestClock()
    reservation = taken(fake, 1, 515, clock, wc)
    assert reservation.day == "2026-12-01"
    assert [op for op, _ in fake.calls].count("table_transact") == 1
    assert fake.items[(day_key("2026-12-01").pk, day_key("2026-12-01").sk)]["jobs"] == 1
    assert fake.items[(TOTAL.pk, TOTAL.sk)]["cents"] == 515
    job = fake.items[(job_key("judges", reservation.job_id).pk, f"job#{reservation.job_id}")]
    assert (job["state"], job["cents"], job["steps"], job["launch"]) == ("held", 515, 5, "pending")
    assert fake.items[("account#judges", f"upload#{1:032x}")]["used_by"] == reservation.job_id


def test_each_limit_refuses_by_name_and_a_reservation_just_inside_fits() -> None:
    wc = website_config(daily_jobs=2, daily_cost_usd=10.30, total_cost_usd=15.45)
    fake, clock = FakeAws(), TestClock()
    taken(fake, 1, 515, clock, wc)
    taken(fake, 2, 515, clock, wc)              # exactly the daily cost
    with pytest.raises(LimitFull) as caught:
        taken(fake, 3, 1, clock, wc)
    assert caught.value.limit == "daily_jobs" and "after 00:00 UTC" in caught.value.message
    assert "2 website jobs" in caught.value.message
    clock.advance(days=1)
    with pytest.raises(LimitFull) as caught:
        taken(fake, 4, 516, clock, wc)
    assert caught.value.limit == "total_cost" and "00:00" not in caught.value.message
    taken(fake, 5, 515, clock, wc)              # exactly the total
    wc2 = website_config(daily_cost_usd=5)
    with pytest.raises(LimitFull) as caught:
        check_room(FakeAws(), 515, T0, wc2)
    assert caught.value.limit == "daily_cost" and "$5.00" in caught.value.message


def test_the_day_comes_from_the_clock_take_reads() -> None:
    fake, wc = FakeAws(), website_config(daily_jobs=1)
    clock = TestClock(datetime(2026, 12, 1, 23, 59, 59, tzinfo=timezone.utc))
    first = taken(fake, 1, 515, clock, wc)
    clock.advance(seconds=1)
    second = taken(fake, 2, 515, clock, wc)
    assert (first.day, second.day) == ("2026-12-01", "2026-12-02")
    assert give_back(fake, first)               # after midnight, back to its own day
    assert fake.items[("limits", "day#2026-12-01")] == {"pk": "limits", "sk": "day#2026-12-01",
                                                        "jobs": 0, "cents": 0}
    assert fake.items[("limits", "day#2026-12-02")]["jobs"] == 1
    assert "used_by" not in fake.items[("account#judges", f"upload#{1:032x}")]


def test_a_second_give_back_and_a_conflicted_one_change_nothing() -> None:
    fake, wc, clock = FakeAws(), website_config(), TestClock()
    reservation = taken(fake, 1, 515, clock, wc)
    fake.conflicts = 1
    assert not give_back(fake, reservation)
    assert fake.items[(TOTAL.pk, TOTAL.sk)]["cents"] == 515
    assert give_back(fake, reservation)
    assert not give_back(fake, reservation)
    assert fake.items[(TOTAL.pk, TOTAL.sk)]["cents"] == 0


def test_a_used_upload_a_conflict_and_the_stop_time_refuse() -> None:
    fake, wc, clock = FakeAws(), website_config(), TestClock()
    first = taken(fake, 1, 515, clock, wc)
    with pytest.raises(PublicRefusal) as caught:
        take(fake, Reservation(job_id="20261201-120000-000099", account="judges",
                               upload_id=first.upload_id, preset="fast", cents=515), 5, clock, wc)
    assert caught.value.kind == "bad_request"
    fake.conflicts = 1
    with pytest.raises(PublicRefusal) as caught:
        taken(fake, 3, 515, clock, wc)
    assert caught.value.kind == "conflict" and "Press Start again" in caught.value.message
    clock.now = wc.stop_starting_at
    with pytest.raises(PublicRefusal) as caught:
        taken(fake, 4, 515, clock, wc)
    assert caught.value.kind == "stopped"
    assert stop_line(wc) == "The website stopped starting jobs at 08:00 UTC on 4 February 2027."


def test_utc_day_converts_to_utc() -> None:
    from datetime import timedelta as td
    later = datetime(2026, 12, 2, 1, 0, tzinfo=timezone(td(hours=5)))
    assert utc_day(later) == "2026-12-01"


# Feature: website, Property 1: The Website_Limits are never passed

PRESET_CENTS = (515, 762, 1841)


class LimitsMachine(RuleBasedStateMachine):
    """Starts, give-backs, kept Reservations, conflicts and clock moves, in any order."""

    @initialize(daily_jobs=st.integers(1, 4), daily_usd=st.integers(10, 40),
                total_usd=st.integers(20, 80))
    def setup(self, daily_jobs: int, daily_usd: int, total_usd: int) -> None:
        self.wc = website_config(daily_jobs=daily_jobs, daily_cost_usd=daily_usd,
                                 total_cost_usd=total_usd, stop_starting_at="2026-12-06T08:00:00Z")
        self.fake = FakeAws()
        self.clock = TestClock()
        self.count = 0
        self.held: list[tuple[Reservation, datetime]] = []
        self.released: list[Reservation] = []

    @rule(cents=st.sampled_from(PRESET_CENTS))
    def start(self, cents: int) -> None:
        self.count += 1
        seed_upload(self.fake, "judges", f"{self.count:032x}")
        try:
            check_room(self.fake, cents, self.clock(), self.wc)
        except PublicRefusal:
            pass
        try:
            reservation = take(self.fake, draft(self.count, cents), 5, self.clock, self.wc)
        except PublicRefusal:
            return
        self.held.append((reservation, self.clock()))

    @precondition(lambda self: self.held)
    @rule(data=st.data())
    def give_back_one(self, data: st.DataObject) -> None:
        index = data.draw(st.integers(0, len(self.held) - 1))
        reservation, _ = self.held[index]
        if give_back(self.fake, reservation):
            self.held.pop(index)
            self.released.append(reservation)

    @precondition(lambda self: self.released)
    @rule(data=st.data())
    def give_back_again(self, data: st.DataObject) -> None:
        assert not give_back(self.fake, data.draw(st.sampled_from(self.released)))

    @rule()
    def keep_after_an_unknown_launch(self) -> None:
        pass   # a launch that may have started a machine keeps its Reservation

    @rule()
    def seed_a_conflict(self) -> None:
        self.fake.conflicts = 1

    @rule(minutes=st.integers(1, 3 * 24 * 60))
    def move_the_clock(self, minutes: int) -> None:
        self.clock.advance(minutes=minutes)

    @rule()
    def pass_the_stop_time(self) -> None:
        self.clock.now = max(self.clock.now, self.wc.stop_starting_at)

    @invariant()
    def the_limits_hold(self) -> None:
        if not hasattr(self, "fake"):
            return
        days: dict[str, tuple[int, int]] = {}
        for reservation, at in self.held:
            assert at < self.wc.stop_starting_at
            jobs, cents = days.get(reservation.day, (0, 0))
            days[reservation.day] = (jobs + 1, cents + reservation.cents)
        for (pk, sk), item in self.fake.items.items():
            if pk == "limits" and sk.startswith("day#"):
                jobs, cents = item.get("jobs", 0), item.get("cents", 0)
                assert 0 <= jobs <= self.wc.daily_jobs
                assert 0 <= cents <= self.wc.daily_cost_cents
                assert (jobs, cents) == days.get(sk[4:], (0, 0))
        total = self.fake.items.get(("limits", "total"), {}).get("cents", 0)
        assert 0 <= total <= self.wc.total_cost_cents
        assert total == sum(reservation.cents for reservation, _ in self.held)


LimitsMachine.TestCase.settings = settings(max_examples=100, stateful_step_count=30,
                                           deadline=None)
TestLimitsAreNeverPassed = LimitsMachine.TestCase
