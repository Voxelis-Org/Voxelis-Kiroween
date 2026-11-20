"""Tests for the JSON files, splatpipe/jsonio.py."""

import json
import math
import re
import sys
import tempfile
from collections import OrderedDict
from datetime import timezone
from enum import IntEnum
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from splatpipe.jsonio import NotJsonError, check_json_value, read_json, write_json

# A high surrogate followed by a low one, as two separate characters. json would read them
# back as the one character U+1F600, so check_json_value refuses them.
SPLIT_PAIR = "\ud83d" + "\ude00"


# Unit tests


class Size(IntEnum):
    SMALL = 1


# Each bad value, and how its message starts, with {} standing for the field path.
BAD_VALUES = [
    pytest.param((1, 2), "{} has the type tuple", id="tuple"),
    pytest.param({1, 2}, "{} has the type set", id="set"),
    pytest.param({1: "x"}, "A key in {} has the type int", id="int key"),
    pytest.param(math.nan, "{} is nan", id="NaN"),
    pytest.param(math.inf, "{} is inf", id="infinity"),
    pytest.param(-math.inf, "{} is -inf", id="minus infinity"),
    pytest.param(Size.SMALL, "{} has the type Size", id="IntEnum"),
    pytest.param(object(), "{} has the type object", id="object"),
    pytest.param(OrderedDict(a=1), "{} has the type OrderedDict", id="dict subclass"),
    pytest.param("x" + SPLIT_PAIR, "{} holds a high surrogate followed by a low surrogate",
                 id="split surrogate pair"),
    pytest.param({SPLIT_PAIR: 1}, "A key in {} holds a high surrogate followed by a low surrogate",
                 id="split surrogate pair in a key"),
]


@pytest.mark.parametrize(("bad", "start"), BAD_VALUES)
def test_check_json_value_refuses_and_names_the_field(bad: object, start: str) -> None:
    with pytest.raises(NotJsonError) as caught:
        check_json_value(bad, "settings")
    assert str(caught.value).startswith(start.format("settings"))


@pytest.mark.parametrize(("bad", "start"), BAD_VALUES)
def test_check_json_value_names_the_path_of_a_nested_value(bad: object, start: str) -> None:
    with pytest.raises(NotJsonError) as caught:
        check_json_value({"a": [0, {"b": bad}]}, "settings")
    assert str(caught.value).startswith(start.format("settings.a[1].b"))


@pytest.mark.parametrize("text", [
    pytest.param("\ud800", id="lone high surrogate"),
    pytest.param("\udc00\ud800", id="low before high"),
    pytest.param("\ud83d \ude00", id="pair split by a space"),
])
def test_lone_surrogates_pass_and_survive_a_write_and_read(tmp_path: Path, text: str) -> None:
    value = {text: [text]}
    check_json_value(value, "warnings")
    path = tmp_path / "warnings.json"
    write_json(path, value)
    assert read_json(path) == value


# Python writes an int as text only up to sys.get_int_max_str_digits() digits, 4300 by default,
# so by default 10 ** limit is a 4301-digit int and 10 ** limit - 1 a 4300-digit one. The sign
# doesn't count as a digit.
@pytest.mark.skipif(sys.get_int_max_str_digits() == 0,
                    reason="Python's limit on an int's digits is off, so no int is too long to write")
@pytest.mark.parametrize("sign", [1, -1], ids=["positive", "negative"])
def test_an_int_too_long_to_write_is_refused_and_one_at_the_limit_survives(
        tmp_path: Path, sign: int) -> None:
    limit = sys.get_int_max_str_digits()
    with pytest.raises(NotJsonError) as caught:
        check_json_value({"a": [sign * 10 ** limit]}, "settings")
    assert str(caught.value).startswith(f"settings.a[0] is an int with more than {limit} digits")
    longest = {"a": [sign * (10 ** limit - 1)]}
    check_json_value(longest, "settings")
    path = tmp_path / "settings.json"
    write_json(path, longest)
    assert read_json(path) == longest


def test_write_json_refuses_a_bad_value_before_making_the_file(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    with pytest.raises(NotJsonError) as caught:
        write_json(path, {"a": [0, {"b": math.nan}]})
    assert str(caught.value).startswith("settings.a[1].b is nan")
    assert not path.exists()


# What a file holds, the error read_json refuses it with, and what our own message says.
BAD_CONTENT = [
    pytest.param(b"[NaN]", NotJsonError, "holds NaN", id="NaN"),
    pytest.param(b'{"a": Infinity}', NotJsonError, "holds Infinity", id="Infinity"),
    pytest.param(b"-Infinity", NotJsonError, "holds -Infinity", id="-Infinity"),
    pytest.param(b'{"a": [1e400]}', NotJsonError, "too big for a float", id="1e400"),
    pytest.param(b"-1e400", NotJsonError, "too big for a float", id="-1e400"),
    pytest.param(b'{"a": "caf\xe9"}', UnicodeDecodeError, None, id="not UTF-8"),
    pytest.param(b'{"a": 1,}', json.JSONDecodeError, None, id="not JSON"),
]


# Task 7's read_earlier counts on every refusal of a file's content being a ValueError.
@pytest.mark.parametrize(("data", "error", "message"), BAD_CONTENT)
def test_read_json_refuses_bad_content_with_a_value_error(
        tmp_path: Path, data: bytes, error: type[ValueError], message: str | None) -> None:
    path = tmp_path / "report.json"
    path.write_bytes(data)
    with pytest.raises(ValueError, match=message) as caught:
        read_json(path)
    assert isinstance(caught.value, error)


# A Step_Report shaped like the design's, with text that isn't ASCII, a character outside
# the BMP (written as an escaped surrogate pair) and a CR LF inside a string.
REPORT = {
    "step": "fake",
    "job_id": "20261119-142530-7f3a9c",
    "preset": "fast",
    "settings": {"fail": "none", "file_bytes": 16, "files": 1},
    "outcome": "failed",
    "started_utc": "2026-11-19T14:25:31.204+00:00",
    "finished_utc": "2026-11-19T14:25:31.209+00:00",
    "duration_seconds": 0.005,
    "timings": {"write_files": 0.0011},
    "counts": {"files_written": 1, "video_bytes": 48213775},
    "warnings": ["Caf\u00e9 at night \U0001f319", "two lines\r\nof text"],
    "error": {
        "readable": "The fake step was set to fail.",
        "expected": True,
        "technical": {"type": "splatpipe.errors.ExpectedFailure", "message": "The fake step was set to fail.",
                      "traceback": "Traceback (most recent call last):\n  ...\n"},
    },
}


def test_a_report_comes_back_equal_from_a_plain_ascii_file(tmp_path: Path) -> None:
    check_json_value(REPORT, "fake")
    path = tmp_path / "fake.json"
    write_json(path, REPORT)
    back = read_json(path)
    assert back == REPORT
    assert list(back) == list(REPORT)
    data = path.read_bytes()
    assert data.isascii()
    assert data.endswith(b"\n")
    assert b"\r" not in data


def test_bools_ints_and_floats_keep_their_types_and_signs(tmp_path: Path) -> None:
    path = tmp_path / "counts.json"
    write_json(path, [True, False, 1, 1.0, 0, -0.0])
    back = read_json(path)
    assert [type(item) for item in back] == [bool, bool, int, float, int, float]
    assert back == [True, False, 1, 1.0, 0, -0.0]
    assert math.copysign(1.0, back[-1]) == -1.0


# Property tests

# A high surrogate directly followed by a low one, as two characters, like SPLIT_PAIR above.
SPLIT_PAIR_FOUND = re.compile(r"[\ud800-\udbff][\udc00-\udfff]")

# st.text() can't give lone surrogates in Hypothesis 6.170 (checked); the second part can, like
# "\ud800", but almost never does. The third part always puts one surrogate between two
# st.text() strings, so about a third of the reports hold a lone surrogate. Strings with a split
# pair are left out: json.loads would join the pair into one character, so check_json_value
# refuses them. The filter is on any_text itself, so it covers dict keys too.
lone_surrogates = st.characters(codec=None, categories=["Cs"])
any_text = (
    st.text()
    | st.text(st.characters(codec=None, exclude_categories=()))
    | st.tuples(st.text(), lone_surrogates, st.text()).map("".join)
).filter(lambda text: SPLIT_PAIR_FOUND.search(text) is None)

json_values = st.recursive(
    st.none() | st.booleans() | st.integers() | st.floats(allow_nan=False, allow_infinity=False) | any_text,
    lambda inner: st.lists(inner, max_size=4) | st.dictionaries(any_text, inner, max_size=4),
    max_leaves=12,
)

utc_times = st.datetimes(timezones=st.just(timezone.utc)).map(lambda t: t.isoformat(timespec="milliseconds"))
seconds = st.floats(min_value=0, allow_infinity=False)

# A Step_Report with the design's fields in the design's order, each filled with values of its kind.
reports = st.fixed_dictionaries({
    "step": st.from_regex(r"[a-z][a-z0-9_]{0,15}", fullmatch=True),
    "job_id": st.from_regex(r"[0-9]{8}-[0-9]{6}-[0-9a-f]{6}", fullmatch=True),
    "preset": st.sampled_from(["fast", "default", "quality"]),
    "settings": st.none() | st.dictionaries(any_text, json_values, max_size=4),
    "outcome": st.sampled_from(["done", "failed"]),
    "started_utc": utc_times,
    "finished_utc": utc_times,
    "duration_seconds": seconds,
    "timings": st.dictionaries(any_text, seconds, max_size=4),
    "counts": st.dictionaries(any_text, st.integers(min_value=0), max_size=4),
    "warnings": st.lists(any_text, max_size=4),
    "error": st.none() | st.dictionaries(any_text, any_text, max_size=4),
})


def typed(value: object) -> object:
    """The value with every type, the key order and each float's sign spelled out, so == is strict.

    Plain == takes True for 1, 1.0 for 1 and -0.0 for 0.0, and ignores the order of dict keys.
    """
    if type(value) is dict:
        return ("dict", [(typed(key), typed(item)) for key, item in value.items()])
    if type(value) is list:
        return ("list", [typed(item) for item in value])
    if type(value) is float:
        return ("float", value.hex())  # exact, and it keeps -0.0 apart from 0.0
    return (type(value).__name__, value)


@settings(deadline=None)
@given(reports)
def test_step_reports_survive_the_round_trip(report: dict[str, object]) -> None:
    """Feature: pipeline-core, Property 5: Step_Reports survive the round trip

    **Validates: Requirements 6.7, 6.9**
    """
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "report.json"
        write_json(path, report)
        back = read_json(path)
        data = path.read_bytes()
    assert typed(back) == typed(report)
    assert data.isascii()
    assert data.endswith(b"\n")
    assert b"\r" not in data
