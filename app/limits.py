"""The Website_Limits: the Worst_Case_Cost, the Limit_Counters, and taking or giving back a
Reservation.

A Reservation is one `TransactWriteItems` of four items: the UTC_Day's counters, the total counter,
a new held job item and the Upload marked as used. Each counter's condition lets the write pass only
while the Reservation fits, so the limits can't be passed even by starts at the same moment. A
give-back is one transaction conditioned on the job item still being held, so it happens at most
once. Nothing else writes the counters, and no Job's real cost is written back.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from app.errors import BAD_REQUEST, CONFLICT, PublicRefusal
from cloud.start import worst_case_cost
from cloud.table import (ITEM_ABSENT, Add, And, AtLeast, AtMost, Change, ConditionFailed, Equals,
                         Key, NotExists, Or, Put, Remove, Set, TransactionConflict)

if TYPE_CHECKING:
    from app.config import WebsiteConfig
    from cloud.aws import Aws
    from cloud.config import CloudConfig

LIMITS = "limits"
TOTAL = Key(LIMITS, "total")
_CENTS = 100
_ROUNDING_DIGITS = 6   # clears float noise before the ceiling
_MONTHS = ("January", "February", "March", "April", "May", "June", "July", "August", "September",
           "October", "November", "December")
CONFLICT_MESSAGE = "Another job started at the same moment, so yours didn't. Press Start again."
USED_MESSAGE = ("That upload has already started a job, or it isn't finished. Upload the video "
                "again.")


@dataclass(frozen=True)
class Reservation:
    job_id: str
    account: str
    upload_id: str
    preset: str
    cents: int
    day: str = ""               # the UTC_Day it was taken in, YYYY-MM-DD


class LimitFull(PublicRefusal):
    """A Website_Limit has no room. `limit` is "daily_jobs", "daily_cost" or "total_cost"."""

    limit: str

    def __init__(self, limit: str, message: str) -> None:
        super().__init__(CONFLICT, "limit", message)
        self.limit = limit


def worst_case_cents(preset: str, cc: CloudConfig, wc: WebsiteConfig) -> int:
    """The Reservation's cents: the dearest Fallback_List type for the Time_Cap plus the boot margin,
    rounded up to the next cent."""
    cost = worst_case_cost(preset, cc, wc.boot_margin_minutes)
    return math.ceil(round(cost * _CENTS, _ROUNDING_DIGITS))


def utc_day(now: datetime) -> str:
    """The UTC_Day of an aware time, YYYY-MM-DD."""
    return now.astimezone(timezone.utc).strftime("%Y-%m-%d")


def day_key(day: str) -> Key:
    return Key(LIMITS, f"day#{day}")


def job_key(account: str, job_id: str) -> Key:
    return Key(f"account#{account}", f"job#{job_id}")


def upload_key(account: str, upload_id: str) -> Key:
    return Key(f"account#{account}", f"upload#{upload_id}")


def stop_line(wc: WebsiteConfig) -> str:
    stop = wc.stop_starting_at
    return (f"The website stopped starting jobs at {stop:%H:%M} UTC on {stop.day} "
            f"{_MONTHS[stop.month - 1]} {stop.year}.")


def stopped(wc: WebsiteConfig) -> PublicRefusal:
    return PublicRefusal(CONFLICT, "stopped", stop_line(wc))


def is_stopped(now: datetime, wc: WebsiteConfig) -> bool:
    return now >= wc.stop_starting_at


def check_room(aws: Aws, cents: int, now: datetime, wc: WebsiteConfig) -> None:
    """Refuse at or after the Stop_Time, or when a Reservation of `cents` wouldn't fit a limit."""
    if is_stopped(now, wc):
        raise stopped(wc)
    if cents > wc.daily_cost_cents:
        raise _full("daily_cost", wc)
    if cents > wc.total_cost_cents:
        raise _full("total_cost", wc)
    day, total = aws.table_get_many([day_key(utc_day(now)), TOTAL])
    day = day or {}
    if int(day.get("jobs", 0)) + 1 > wc.daily_jobs:
        raise _full("daily_jobs", wc)
    if int(day.get("cents", 0)) + cents > wc.daily_cost_cents:
        raise _full("daily_cost", wc)
    if int((total or {}).get("cents", 0)) + cents > wc.total_cost_cents:
        raise _full("total_cost", wc)


def take(aws: Aws, draft: Reservation, steps: int, clock: Callable[[], datetime],
         wc: WebsiteConfig) -> Reservation:
    """Take the Reservation in one transaction, at the clock's time, and give it back day-stamped."""
    now = clock()
    reservation = replace(draft, day=utc_day(now))
    check_room(aws, draft.cents, now, wc)
    c = draft.cents
    items = [
        Change(day_key(reservation.day), (Add("jobs", 1), Add("cents", c)),
               And((Or((NotExists("jobs"), AtMost("jobs", wc.daily_jobs - 1))),
                    Or((NotExists("cents"), AtMost("cents", wc.daily_cost_cents - c)))))),
        Change(TOTAL, (Add("cents", c),),
               Or((NotExists("cents"), AtMost("cents", wc.total_cost_cents - c)))),
        Put(_job_item(reservation, steps, now), ITEM_ABSENT),
        Change(upload_key(draft.account, draft.upload_id), (Set("used_by", draft.job_id),),
               And((Equals("state", "complete"), NotExists("used_by")))),
    ]
    try:
        aws.table_transact(items, token=draft.job_id)
    except ConditionFailed as failed:
        raise _refusal(aws, failed.index, reservation.day, wc) from None
    except TransactionConflict:
        raise PublicRefusal(CONFLICT, "conflict", CONFLICT_MESSAGE) from None
    return reservation


def give_back(aws: Aws, reservation: Reservation) -> bool:
    """Give the Reservation back once. False when it was given back already or a conflict stopped it,
    which leaves it held: that can only overcount, never pass a limit."""
    c = reservation.cents
    items = [
        Change(job_key(reservation.account, reservation.job_id), (Set("state", "released"),),
               Equals("state", "held")),
        Change(day_key(reservation.day), (Add("jobs", -1), Add("cents", -c)),
               And((AtLeast("jobs", 1), AtLeast("cents", c)))),
        Change(TOTAL, (Add("cents", -c),), AtLeast("cents", c)),
    ]
    try:
        aws.table_transact(items)
    except (ConditionFailed, TransactionConflict):
        return False
    try:
        aws.table_update(upload_key(reservation.account, reservation.upload_id),
                         (Remove("used_by"),), Equals("used_by", reservation.job_id))
    except ConditionFailed:
        pass  # the Upload item may have expired
    return True


def _job_item(reservation: Reservation, steps: int, now: datetime) -> dict[str, str | int]:
    key = job_key(reservation.account, reservation.job_id)
    return {"pk": key.pk, "sk": key.sk, "state": "held", "preset": reservation.preset,
            "cents": reservation.cents, "day": reservation.day, "started_at": int(now.timestamp()),
            "steps": steps, "upload": reservation.upload_id, "launch": "pending"}


def _refusal(aws: Aws, index: int | None, day: str, wc: WebsiteConfig) -> PublicRefusal:
    if index == 0:
        item = aws.table_get(day_key(day)) or {}
        return _full("daily_jobs" if int(item.get("jobs", 0)) >= wc.daily_jobs else "daily_cost", wc)
    if index == 1:
        return _full("total_cost", wc)
    if index == 3:
        return PublicRefusal(BAD_REQUEST, "bad_request", USED_MESSAGE)
    return PublicRefusal(CONFLICT, "conflict", CONFLICT_MESSAGE)


def _full(limit: str, wc: WebsiteConfig) -> LimitFull:
    again = " Jobs can start again after 00:00 UTC."
    if limit == "daily_jobs":
        return LimitFull(limit, f"Today's limit of {wc.daily_jobs} website jobs is reached, so your "
                                f"job didn't start.{again}")
    if limit == "daily_cost":
        return LimitFull(limit, f"Today's limit of ${wc.daily_cost_cents / _CENTS:.2f} for website "
                                f"jobs has no room for this job, so it didn't start.{again}")
    return LimitFull(limit, f"The website's total limit of ${wc.total_cost_cents / _CENTS:.2f} for "
                            "jobs has no room for this job, so it didn't start.")

