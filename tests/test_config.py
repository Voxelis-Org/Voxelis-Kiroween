"""Tests for the Config_Loader, splatpipe/config.py."""

import datetime
import math
import re
import tomllib
from collections.abc import Callable
from pathlib import Path
from typing import Any, NamedTuple

import pytest

from splatpipe.config import (NOT_PIPELINE, REPO_ROOT, Config, JobsSettings, check_preset,
                              load_config)
from splatpipe.errors import PipelineError

# The three Presets, written out here rather than taken from the code under test.
PRESET_NAMES = ("fast", "default", "quality")


# Unit tests

# A good [fake] section as tomllib reads it, with two Shared_Settings and one Preset_Setting.
# Config deep-copies its data and nothing here changes this dict, so the tests share it.
FAKE = {"fail": "none", "file_bytes": 16,
        "fast": {"files": 1}, "default": {"files": 2}, "quality": {"files": 3}}


def jobs_section(tmp_path: Path) -> dict[str, Any]:
    """A good [jobs] section, with its Jobs_Folder inside tmp_path."""
    return {"jobs_folder": (tmp_path / "jobs").as_posix(), "default_preset": "fast",
            "max_video_bytes": 1024, "max_video_seconds": 600, "min_used_seconds": 3,
            "ffprobe_timeout_seconds": 30}


def refusal(action: Callable[..., object], *args: object) -> str:
    """Call the action with the args, and give back the message of the PipelineError it raises."""
    with pytest.raises(PipelineError) as caught:
        action(*args)
    return str(caught.value)


def lists_the_presets(message: str) -> bool:
    """Whether the message names each Preset as a word of its own, so default_preset doesn't count."""
    return all(re.search(rf"\b{name}\b", message) for name in PRESET_NAMES)


@pytest.mark.parametrize(("preset", "files"), [("fast", 1), ("default", 2), ("quality", 3)])
def test_step_settings_gives_the_shared_and_preset_settings_together(preset: str, files: int) -> None:
    settings = Config({"fake": FAKE}).step_settings("fake", preset)
    assert settings == {"fail": "none", "file_bytes": 16, "files": files}
    assert (settings.section, settings.preset) == ("fake", preset)


# Each broken [fake] section, and the words its refusal must hold: the section, and the key or
# table. Every case asks for fast, so the checks must cover the default and quality tables too.
BROKEN_LAYOUTS = [
    pytest.param({"fail": "none", "fast": {"files": 1}, "default": {"files": 2}},
                 ["[fake]", "no [fake.quality]"], id="no quality table"),
    pytest.param({"fast": {"files": 1}, "default": {"files": 2, "extra": 0}, "quality": {"files": 3}},
                 ["[fake]", "extra", "not in [fake.fast] or [fake.quality]"],
                 id="a key in only some Preset tables"),
    pytest.param({**FAKE, "files": 0}, ["[fake]", "files"], id="a key both shared and per-Preset"),
    pytest.param({**FAKE, "other": {"files": 4}}, ["[fake]", "[fake.other]"], id="a table named other"),
    pytest.param({"fast": 1, "default": {"files": 2}, "quality": {"files": 3}},
                 ["[fake]", "[fake.fast]"], id="fast = 1"),
]


@pytest.mark.parametrize(("section", "words"), BROKEN_LAYOUTS)
def test_a_broken_step_section_is_refused_naming_the_section_and_the_key_or_table(
        section: dict[str, Any], words: list[str]) -> None:
    message = refusal(Config({"fake": section}).step_settings, "fake", "fast")
    for word in words:
        assert word in message


@pytest.mark.parametrize("data", [
    pytest.param({}, id="no sections"),
    pytest.param({"jobs": 1, "fake": 1}, id="numbers in their place"),
])
def test_a_missing_section_is_refused_by_name(data: dict[str, Any]) -> None:
    config = Config(data)
    assert "[jobs]" in refusal(config.jobs)
    assert "[fake]" in refusal(config.step_settings, "fake", "fast")


# The values a Pipeline_Section can't hold, as tomllib gives them. The datetime is what
# tomllib makes of 1979-05-27T07:32:00Z.
NOT_SETTINGS = [
    pytest.param(math.nan, id="nan"),
    pytest.param(math.inf, id="inf"),
    pytest.param(-math.inf, id="-inf"),
    pytest.param(datetime.date(1979, 5, 27), id="a date"),
    pytest.param(datetime.time(7, 32), id="a time"),
    pytest.param(datetime.datetime(1979, 5, 27, 7, 32, tzinfo=datetime.timezone.utc), id="a datetime"),
]


@pytest.mark.parametrize("value", NOT_SETTINGS)
def test_nan_infinity_and_dates_are_refused_naming_the_section_and_the_key(value: object) -> None:
    message = refusal(Config, {"fake": {**FAKE, "fast": {"files": value}}})
    assert "[fake]" in message and "fast.files" in message


# Where a bad value can sit, and the words its refusal must hold: the section and the key path.
PLACES = [
    pytest.param({"fake": {"fast": {"sizes": [16, math.nan]}}}, ["[fake]", "fast.sizes[1]"],
                 id="in a list"),
    pytest.param({"fake": {"shapes": [{"w": math.inf}]}}, ["[fake]", "shapes[0].w"],
                 id="in a table in a list"),
    pytest.param({"scale": math.nan}, ["top-level", "scale"], id="a top-level key"),
]


@pytest.mark.parametrize(("data", "words"), PLACES)
def test_a_refusal_names_the_key_path_of_the_bad_value(data: dict[str, Any], words: list[str]) -> None:
    message = refusal(Config, data)
    for word in words:
        assert word in message


@pytest.mark.parametrize("name", ["git", "aws", "paid", "devlog"])
def test_the_sections_outside_the_pipeline_may_hold_dates_and_nan(name: str) -> None:
    Config({name: {"day": datetime.date(1979, 5, 27), "ratio": math.nan}})  # raises nothing


# A section a later spec might add. It isn't a Step_Section, so it may hold any table.
CLOUD = {"machine": "g5.4xlarge", "price_per_hour": 2.0, "limits": {"jobs_per_day": 3}}


def test_a_cloud_section_next_to_jobs_and_fake_loads_with_no_code_change(tmp_path: Path) -> None:
    config = Config({"jobs": jobs_section(tmp_path), "fake": FAKE, "cloud": CLOUD})
    assert config.jobs().jobs_folder == tmp_path / "jobs"
    assert [config.step_settings("fake", preset)["files"] for preset in PRESET_NAMES] == [1, 2, 3]


def test_nan_in_the_cloud_section_is_refused_by_name(tmp_path: Path) -> None:
    cloud = {**CLOUD, "price_per_hour": math.nan}
    message = refusal(Config, {"jobs": jobs_section(tmp_path), "fake": FAKE, "cloud": cloud})
    assert "[cloud]" in message and "price_per_hour" in message


def test_jobs_gives_the_six_settings_with_the_seconds_as_floats(tmp_path: Path) -> None:
    jobs = Config({"jobs": jobs_section(tmp_path)}).jobs()
    assert jobs == JobsSettings(jobs_folder=tmp_path / "jobs", default_preset="fast",
                                max_video_bytes=1024, max_video_seconds=600.0,
                                min_used_seconds=3.0, ffprobe_timeout_seconds=30.0)
    seconds = (jobs.max_video_seconds, jobs.min_used_seconds, jobs.ffprobe_timeout_seconds)
    assert [type(value) for value in seconds] == [float, float, float]


@pytest.mark.parametrize("key", ["jobs_folder", "default_preset", "max_video_bytes",
                                 "max_video_seconds", "min_used_seconds", "ffprobe_timeout_seconds"])
def test_a_missing_jobs_setting_is_refused_by_name(tmp_path: Path, key: str) -> None:
    section = jobs_section(tmp_path)
    del section[key]
    message = refusal(Config({"jobs": section}).jobs)
    assert "[jobs]" in message and key in message


# A [jobs] key and a value it can't take: the wrong type, a bool where a number belongs, or a
# number that isn't above 0.
BAD_JOBS_VALUES = [
    pytest.param("jobs_folder", 5, id="a number as the jobs folder"),
    pytest.param("max_video_bytes", "4 GB", id="text as a whole number"),
    pytest.param("max_video_bytes", 4.5, id="a fraction as a whole number"),
    pytest.param("max_video_seconds", "600", id="text as seconds"),
    pytest.param("max_video_bytes", True, id="a bool as a whole number"),
    pytest.param("min_used_seconds", True, id="a bool as seconds"),
    pytest.param("max_video_bytes", 0, id="0 bytes"),
    pytest.param("min_used_seconds", -3, id="minus 3 seconds"),
    pytest.param("ffprobe_timeout_seconds", 0.0, id="0.0 seconds"),
]


@pytest.mark.parametrize(("key", "value"), BAD_JOBS_VALUES)
def test_a_bad_jobs_setting_is_refused_by_name(tmp_path: Path, key: str, value: object) -> None:
    message = refusal(Config({"jobs": {**jobs_section(tmp_path), key: value}}).jobs)
    assert "[jobs]" in message and key in message


def test_a_relative_jobs_folder_is_joined_to_the_repo_root(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = Path(__file__).resolve().parents[1]  # this file is in tests/, at the repo root
    config = Config({"jobs": {**jobs_section(tmp_path), "jobs_folder": "data/jobs"}})
    assert config.jobs().jobs_folder == repo / "data" / "jobs"  # read only: nothing is made
    monkeypatch.chdir(tmp_path)
    assert config.jobs().jobs_folder == repo / "data" / "jobs"


def test_a_missing_step_setting_is_refused_by_name() -> None:
    settings = Config({"fake": FAKE}).step_settings("fake", "default")
    message = refusal(lambda: settings["nope"])
    for word in ("[fake]", "[fake.default]", "nope"):
        assert word in message
    assert settings.get("nope") is None and "nope" not in settings


@pytest.mark.parametrize("name", ["slow", "Fast", "", None])
def test_a_bad_preset_is_refused_listing_the_three(name: object) -> None:
    assert lists_the_presets(refusal(check_preset, name))


def test_step_settings_refuses_a_bad_preset_listing_the_three() -> None:
    assert lists_the_presets(refusal(Config({"fake": FAKE}).step_settings, "fake", "slow"))


def test_a_bad_default_preset_is_refused_listing_the_three(tmp_path: Path) -> None:
    message = refusal(Config({"jobs": {**jobs_section(tmp_path), "default_preset": "slow"}}).jobs)
    assert "[jobs]" in message and "default_preset" in message
    assert lists_the_presets(message)


@pytest.mark.parametrize("content", [
    pytest.param(None, id="no file"),
    pytest.param(b'[jobs]\ndefault_preset = "caf\xe9"\n', id="not UTF-8"),
    pytest.param(b"[jobs\n", id="not TOML"),
])
def test_a_config_file_that_cant_be_loaded_is_refused_naming_the_file(
        tmp_path: Path, content: bytes | None) -> None:
    path = tmp_path / "config.toml"
    if content is not None:
        path.write_bytes(content)
    assert str(path) in refusal(load_config, path)


# Comment check

class Statement(NamedTuple):
    """One TOML statement: the number of its first line, counted from 1, its lines and its dict."""

    line: int
    lines: list[str]
    parsed: dict[str, Any]


def parsed_toml(text: str) -> dict[str, Any] | None:
    """What tomllib makes of the text, or None when tomllib refuses it."""
    try:
        return tomllib.loads(text)
    except tomllib.TOMLDecodeError:
        return None


def statement_end(lines: list[str], start: int) -> tuple[int, dict[str, Any]]:
    """Where the statement starting at lines[start] ends, adding lines until tomllib accepts them.

    A group that never parses fails the test, giving its first line.
    """
    for end in range(start + 1, len(lines) + 1):
        parsed = parsed_toml("\n".join(lines[start:end]))
        if parsed is not None:
            return end, parsed
    pytest.fail(f"Line {start + 1} of the config doesn't parse as TOML, even with the lines after it.")


def split_statements(text: str) -> list[Statement]:
    """The TOML text as statements, so a multi-line array is one statement and its inner lines
    are never read as keys or headers."""
    lines = text.splitlines()
    statements: list[Statement] = []
    start = 0
    while start < len(lines):
        end, parsed = statement_end(lines, start)
        statements.append(Statement(start + 1, lines[start:end], parsed))
        start = end
    return statements


def is_comment_line(statement: Statement) -> bool:
    """Whether the statement is a comment-only line with text after the #."""
    text = statement.lines[0].strip()
    return text.startswith("#") and bool(text[1:].strip())


def has_comment_on_its_line(statement: Statement) -> bool:
    """Whether cutting its first line at some #, keeping the later lines, parses to the same dict,
    with text after that #. A # inside a string fails to parse or parses to another value."""
    first, later = statement.lines[0], statement.lines[1:]
    for index, char in enumerate(first):
        if char == "#" and first[index + 1:].strip():
            if parsed_toml("\n".join([first[:index], *later])) == statement.parsed:
                return True
    return False


def has_comment(previous: Statement | None, statement: Statement) -> bool:
    """Whether a key statement has a comment, right before it or on its own line."""
    return (previous is not None and is_comment_line(previous)) or has_comment_on_its_line(statement)


def uncommented_keys(text: str) -> list[tuple[int, str]]:
    """Each key in a Pipeline_Section of the TOML text that has no comment, as (line, section).

    A header's section is the first key it parses to, so [fake.fast] gives fake. Keys above the
    first header are skipped.
    """
    statements = split_statements(text)
    missing: list[tuple[int, str]] = []
    section: str | None = None
    for previous, statement in zip([None, *statements], statements):
        if statement.lines[0].lstrip().startswith("["):
            section = next(iter(statement.parsed))
        elif (statement.parsed and section is not None and section not in NOT_PIPELINE
              and not has_comment(previous, statement)):
            missing.append((statement.line, section))
    return missing


def test_a_hash_inside_a_string_is_not_a_comment() -> None:
    assert uncommented_keys('[fake]\na = "x#y"\n') == [(2, "fake")]


def test_a_comment_after_a_string_holding_a_hash_counts() -> None:
    assert uncommented_keys('[fake]\na = "x#y"  # real\n') == []


def test_a_comment_two_lines_above_a_key_doesnt_count() -> None:
    assert uncommented_keys("[fake]\n# Size of each file.\n\nfile_bytes = 16\n") == [(4, "fake")]


def test_a_header_line_inside_a_multi_line_array_isnt_a_header() -> None:
    text = '[fake]\n# Names.\nnames = [\n["x"]\n]\nfiles = 1\n'
    assert uncommented_keys(text) == [(6, "fake")]  # files is still in [fake], not in [x]


def test_a_key_under_fake_fast_counts_as_a_key_of_fake() -> None:
    assert uncommented_keys("[fake.fast]\nfiles = 1\n") == [(2, "fake")]


def test_a_key_with_no_comment_under_cloud_fails_the_check() -> None:
    assert uncommented_keys('[cloud]\nmachine = "g5.4xlarge"\n') == [(2, "cloud")]


# The real config.toml

# Read by path, so these tests still read it once CONFIG_PATH points at a test config.
REAL_CONFIG = REPO_ROOT / "config.toml"

# The sections outside the Pipeline, as they are today.
UNCHANGED_SECTIONS = {
    "git": {"main_branch": "main"},
    "aws": {"region": "us-east-1", "project_tag": "project=voxelis-kiroween",
            "read_profile": "kiroween-read", "runner_profile": "kiroween-runner",
            "spend_variable": "APPROVED_SPEND", "stack_name": "voxelis-kiroween"},
    "paid": {"aws": [
        "ec2 run-instances", "ec2 start-instances", "ec2 create-image",
        "ec2 request-spot-instances", "ec2 request-spot-fleet", "ec2 create-fleet",
        "cloudformation create-stack", "cloudformation update-stack", "cloudformation deploy",
        "cloudformation execute-change-set",
    ]},
    "devlog": {"path": "docs/devlog.md", "commit_message": "Devlog: {task}"},
}


def test_the_real_jobs_section_holds_its_six_values() -> None:
    assert load_config(REAL_CONFIG).jobs() == JobsSettings(
        jobs_folder=REPO_ROOT / "data" / "jobs", default_preset="fast",
        max_video_bytes=4294967296, max_video_seconds=600.0, min_used_seconds=3.0,
        ffprobe_timeout_seconds=30.0)


def test_the_real_git_aws_paid_and_devlog_sections_are_unchanged() -> None:
    data = tomllib.loads(REAL_CONFIG.read_text(encoding="utf-8"))
    assert {name: data.get(name) for name in UNCHANGED_SECTIONS} == UNCHANGED_SECTIONS


def test_every_key_in_a_pipeline_section_of_the_real_config_has_a_comment() -> None:
    assert uncommented_keys(REAL_CONFIG.read_text(encoding="utf-8")) == []


@pytest.mark.parametrize("preset", PRESET_NAMES)
def test_the_real_fake_section_loads_for_each_preset(preset: str) -> None:
    settings = load_config(REAL_CONFIG).step_settings("fake", preset)
    assert settings["fail"] == "none" and "files" in settings
