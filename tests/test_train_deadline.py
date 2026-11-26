"""Unit tests for the Cap_Deadline and the Stop_Time: splatpipe/train/deadline.py.

Every time is passed in, so no test reads the clock (Req 7.11). The real clean_reserve_minutes and
save_margin_minutes are read from config.toml by path, because tests/conftest.py points the
Config_Loader at a test config.
"""

import tomllib
from datetime import datetime, timedelta, timezone

import pytest

from cloud import runner, userdata
from splatpipe.config import REPO_ROOT
from splatpipe.errors import ExpectedFailure
from splatpipe.train.deadline import (CAP_DEADLINE_ENV, EXAMPLE, read_cap_deadline, should_stop,
                                      stop_time)

# Read by path, so this still reads the real config once CONFIG_PATH points at a test config.
REAL_CONFIG = REPO_ROOT / "config.toml"
# The Cap_Deadline the examples use, and the Stop_Time it gives with the real [train] settings.
CAP_DEADLINE = datetime(2026, 11, 21, 14, 25, 30, tzinfo=timezone.utc)
STOP = datetime(2026, 11, 21, 14, 10, 30, tzinfo=timezone.utc)


def read(value: str) -> datetime:
    """The Cap_Deadline that SPLATPIPE_CAP_DEADLINE set to `value` gives, which must be set."""
    deadline = read_cap_deadline({CAP_DEADLINE_ENV: value})
    assert deadline is not None
    return deadline


def test_unset_variable_gives_no_cap_deadline() -> None:
    """With SPLATPIPE_CAP_DEADLINE not set there's no Cap_Deadline (Req 7.2)."""
    assert read_cap_deadline({}) is None


def test_both_cloud_forms_read_as_utc_times() -> None:
    """The User_Data's Z form and the Cloud_Runner's isoformat() form both read, as UTC times
    123,456 µs apart (Req 7.4)."""
    plain = read("2026-11-21T14:25:30Z")
    precise = read("2026-11-21T14:25:30.123456+00:00")
    assert plain == CAP_DEADLINE
    assert plain.tzinfo == timezone.utc
    assert precise.tzinfo == timezone.utc
    assert precise - plain == timedelta(microseconds=123456)


def test_other_offset_is_turned_to_utc() -> None:
    """A time with a +05:30 offset is turned to the same instant in UTC (Req 7.4)."""
    deadline = read("2026-11-21T19:55:30+05:30")
    assert deadline == CAP_DEADLINE
    assert deadline.tzinfo == timezone.utc


@pytest.mark.parametrize("value", ["", "soon", "2026-11-21T14:25:30"])
def test_bad_value_is_refused_with_a_readable_message(value: str) -> None:
    """An empty value, one fromisoformat refuses and one with no UTC offset are each refused with
    a message that names the variable, gives the value and gives the example (Req 7.3, 14.6)."""
    with pytest.raises(ExpectedFailure) as caught:
        read_cap_deadline({CAP_DEADLINE_ENV: value})
    message = str(caught.value)
    assert "SPLATPIPE_CAP_DEADLINE" in message
    assert repr(value) in message
    assert EXAMPLE in message
    assert "Traceback" not in message


def test_stop_time_with_the_real_settings() -> None:
    """With the real clean_reserve_minutes of 10 and save_margin_minutes of 5, a 14:25:30
    Cap_Deadline gives a 14:10:30 Stop_Time (Req 7.5)."""
    with REAL_CONFIG.open("rb") as file:
        train = tomllib.load(file)["train"]
    clean_reserve, save_margin = train["clean_reserve_minutes"], train["save_margin_minutes"]
    assert (clean_reserve, save_margin) == (10, 5)
    assert stop_time(CAP_DEADLINE, clean_reserve, save_margin) == STOP


def test_should_stop_at_and_after_the_stop_time() -> None:
    """Training goes on a microsecond before the Stop_Time and stops at it and after it
    (Req 7.6, 7.11)."""
    assert should_stop(STOP - timedelta(microseconds=1), STOP) is False
    assert should_stop(STOP, STOP) is True
    assert should_stop(STOP + timedelta(microseconds=1), STOP) is True


def test_no_stop_time_never_stops() -> None:
    """With no Stop_Time, should_stop is false (Req 7.2)."""
    assert should_stop(STOP, None) is False


def test_variable_name_matches_the_cloud_code() -> None:
    """The step reads the variable that the Cloud_Runner and the User_Data set (Req 7.1)."""
    assert CAP_DEADLINE_ENV == runner.CAP_DEADLINE_ENV == userdata.CAP_DEADLINE_ENV
