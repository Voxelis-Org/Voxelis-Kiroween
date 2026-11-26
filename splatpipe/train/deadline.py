"""The Cap_Deadline and the Stop_Time, as pure functions that take the time as an argument.

The Cloud_Runner sets SPLATPIPE_CAP_DEADLINE to the Cap_Deadline, the UTC instant by which the
Job's steps should have saved and stopped. read_cap_deadline reads it from the environment it's
given. It gives None when the variable isn't set, so every Planned_Iteration runs (Req 7.2). An
empty value, a value that datetime.fromisoformat refuses and a value with no UTC offset are each
refused with an ExpectedFailure that names the variable, gives its value with repr() and gives
EXAMPLE (Req 7.3). An empty value counts as malformed, not unset, because the User_Data leaves the
variable empty only when working out the deadline failed (A4). Python 3.11's fromisoformat reads
both forms the cloud code writes: 2026-11-21T14:25:30Z from the User_Data and
2026-11-21T14:25:30.123456+00:00 from the Cloud_Runner's isoformat() (Req 7.4). A good value is
turned to UTC.

The Stop_Time is the Cap_Deadline minus clean_reserve_minutes, which gives the Train_Deadline,
then minus save_margin_minutes (Req 7.5). The Trainer calls should_stop(utc_now(), stop) before
each Iteration and starts none at or after the Stop_Time (Req 7.6). stop_time and should_stop take
the times as arguments, so the tests pass them a fake clock (Req 7.11).

From splatpipe, this module imports only splatpipe.errors.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timedelta, timezone

from splatpipe.errors import ExpectedFailure

CAP_DEADLINE_ENV = "SPLATPIPE_CAP_DEADLINE"  # Fixed_Constant: the variable the Cloud_Runner sets
EXAMPLE = "2026-11-21T14:25:30Z"  # the form the messages show


def read_cap_deadline(environ: Mapping[str, str]) -> datetime | None:
    """The Cap_Deadline that `environ` holds, in UTC, or None when CAP_DEADLINE_ENV isn't set.

    An empty value, a value that datetime.fromisoformat refuses and a value with no UTC offset are
    each refused with the Req 7.3 ExpectedFailure (Req 7.1–7.4, A4).
    """
    value = environ.get(CAP_DEADLINE_ENV)
    if value is None:
        return None  # not set: no Stop_Time (Req 7.2)
    deadline = _parsed(value)
    if deadline is None or deadline.utcoffset() is None:
        raise ExpectedFailure(f"{CAP_DEADLINE_ENV} is {value!r}, but it must be a date and time "
                              f"with a UTC offset, like {EXAMPLE}. The cloud runner sets it, so "
                              "on a laptop unset it.")
    return deadline.astimezone(timezone.utc)


def _parsed(value: str) -> datetime | None:
    """The date and time that datetime.fromisoformat reads from `value`, or None when `value` is
    empty (malformed, not unset, A4) or fromisoformat refuses it."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def stop_time(cap_deadline: datetime, clean_reserve_minutes: float,
              save_margin_minutes: float) -> datetime:
    """The Stop_Time: the Cap_Deadline minus clean_reserve_minutes, which is the Train_Deadline,
    then minus save_margin_minutes (Req 7.5)."""
    return cap_deadline - timedelta(minutes=clean_reserve_minutes + save_margin_minutes)


def should_stop(now: datetime, stop: datetime | None) -> bool:
    """Whether the Trainer starts no more Iterations: a Stop_Time is set and `now` is at or after
    it (Req 7.6). With no Stop_Time, it never stops early (Req 7.2)."""
    return stop is not None and now >= stop


def utc_now() -> datetime:
    """The time now, in UTC, which the Trainer passes to should_stop before each Iteration."""
    return datetime.now(timezone.utc)
