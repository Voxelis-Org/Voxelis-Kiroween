"""Tests for the clean step's settings checks, in splatpipe/clean/settings.py (Req 6.6 to 6.10).

The tests of the kinds, ranges and names build their settings with clean_settings: the test
config's [clean] for the fast Preset, with one setting changed. The tests of the eighteen values
and of a missing setting read the test config itself, with load_config and set_clean.
Property 10 checks the file-name checks against an oracle of its own, on the names that
tests/splats.py draws.
"""

import dataclasses
import itertools
from collections.abc import Callable
from typing import NamedTuple

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from splatpipe.clean.settings import check_plain_name, check_settings
from splatpipe.config import load_config
from splatpipe.errors import ExpectedFailure, PipelineError

from conftest import CLEAN_PRESETS, CLEAN_SHARED
from splats import PRESET, clean_settings, names, three_names

# The three Presets, written out here rather than taken from the code under test.
PRESET_NAMES = ("fast", "default", "quality")


def refusal(**values: object) -> str:
    """The message of the ExpectedFailure that check_settings raises for the test config's
    settings with `values` changed, as clean_settings builds them. It must hold no traceback."""
    with pytest.raises(ExpectedFailure) as caught:
        clean_settings(**values)
    message = str(caught.value)
    assert "Traceback" not in message
    return message


def table_of(key: str) -> str:
    """The table a refusal of `key` names: [clean.<preset>] for a Preset_Setting, which
    clean_settings reads from PRESET's table, and [clean] for a Shared_Setting."""
    return f"[clean.{PRESET}]" if key in CLEAN_PRESETS[PRESET] else "[clean]"


# Unit tests

@pytest.mark.parametrize("preset", PRESET_NAMES)
def test_check_settings_gives_the_eighteen_settings_of_the_test_config(preset: str) -> None:
    settings = load_config().step_settings("clean", preset)
    assert type(settings["extent_trim_percent"]) is int  # the test config says 1, not 1.0
    checked = check_settings(settings)
    assert dataclasses.asdict(checked) == {**CLEAN_SHARED, **CLEAN_PRESETS[preset]}
    assert type(checked.extent_trim_percent) is float and checked.extent_trim_percent == 1.0


# What to leave out with set_clean, and the key it leaves out: a Shared_Setting, and a
# Preset_Setting left out of all three Preset tables, so that the tables still have the same keys.
LEFT_OUT = [
    pytest.param({"neighbours": None}, "neighbours", id="neighbours"),
    pytest.param({f"{preset}__sh_iterations": None for preset in PRESET_NAMES}, "sh_iterations",
                 id="sh_iterations"),
]


@pytest.mark.parametrize("preset", PRESET_NAMES)
@pytest.mark.parametrize(("left_out", "key"), LEFT_OUT)
def test_a_missing_setting_is_the_step_runners_own_refusal(
        set_clean: Callable[..., None], left_out: dict[str, None], key: str, preset: str) -> None:
    set_clean(**left_out)
    settings = load_config().step_settings("clean", preset)
    with pytest.raises(PipelineError) as caught:
        check_settings(settings)
    assert type(caught.value) is PipelineError  # StepSettings' own refusal, not ExpectedFailure
    message = str(caught.value)
    for word in ("[clean]", f"[clean.{preset}]", key):
        assert word in message
    assert "Traceback" not in message


class Range(NamedTuple):
    """A setting's range as Req 6.7 or 6.8 gives it, written out here rather than taken from the
    code under test: what a refusal says the setting must be, values just outside the range, and
    the ends the range includes."""

    wanted: str
    outside: tuple[object, ...]
    ends: tuple[object, ...]


WHOLE_ABOVE_0 = Range("a whole number above 0", outside=(0,), ends=(1,))
FRACTION = Range("a number from 0 to 1", outside=(-0.1, 1.1), ends=(0, 1))
ABOVE_0 = Range("a number above 0", outside=(0,), ends=())

# The whole-number settings of Req 6.7, then the number settings of Req 6.8, each with its range.
WHOLE_NUMBERS = {
    "min_kept_splats": WHOLE_ABOVE_0,
    "max_source_splats": WHOLE_ABOVE_0,
    "neighbours": WHOLE_ABOVE_0,
    "faint_cube_divisions": WHOLE_ABOVE_0,
    "faint_min_per_cube": WHOLE_ABOVE_0,
    "sh_iterations": WHOLE_ABOVE_0,
    "sog_sh_bands": Range("a whole number from 0 to 3", outside=(-1, 4), ends=(0, 3)),
    "max_workers": Range("a whole number from 0 to 8", outside=(-1, 9), ends=(0, 8)),
}
NUMBERS = {
    "min_opacity": FRACTION,
    "warn_kept_fraction": FRACTION,
    "max_removed_fraction": FRACTION,
    "faint_min_paint_share": Range("a number from 0 up to but not including 1",
                                   outside=(-0.1, 1), ends=(0,)),
    "extent_trim_percent": Range("a number from 0 up to but not including 50",
                                 outside=(-0.1, 50), ends=(0,)),
    "isolation_ratio": ABOVE_0,
    "splat_transform_timeout_seconds": ABOVE_0,
}
RANGES = {**WHOLE_NUMBERS, **NUMBERS}

TOO_BIG = 10**400  # an int too big for a float, which a number setting refuses as out of range
# The other values each setting refuses: a bool and a str, a float for a whole number, and
# TOO_BIG for a number.
REFUSED_KINDS = {**dict.fromkeys(WHOLE_NUMBERS, (True, 4.0, "8")),
                 **dict.fromkeys(NUMBERS, (True, "0.5", TOO_BIG))}


def case_id(key: str, value: object) -> str:
    """A test id like `neighbours = 0`, with TOO_BIG as 10**400 rather than its 401 digits."""
    return f"{key} = {'10**400' if value == TOO_BIG else repr(value)}"


REFUSED = [pytest.param(key, value, id=case_id(key, value))
           for key, wanted in RANGES.items() for value in (*REFUSED_KINDS[key], *wanted.outside)]
ENDS = [pytest.param(key, value, id=case_id(key, value))
        for key, wanted in RANGES.items() for value in wanted.ends]


@pytest.mark.parametrize(("key", "value"), REFUSED)
def test_a_value_of_the_wrong_kind_or_out_of_range_is_refused_naming_the_setting(
        key: str, value: object) -> None:
    message = refusal(**{key: value})
    for word in (key, table_of(key), RANGES[key].wanted):
        assert word in message


# Each case changes one setting and leaves the rest at the test config's values. No setting's
# range depends on another's, so each range is checked on its own.
@pytest.mark.parametrize(("key", "value"), ENDS)
def test_the_ends_a_range_includes_pass(key: str, value: object) -> None:
    assert getattr(clean_settings(**{key: value}), key) == value


NAME_SETTINGS = ("train_ply", "clean_ply", "clean_sog")
# Values that aren't Plain_File_Names (Req 6.9), each refused for each of the three names.
NOT_PLAIN_NAMES = [
    pytest.param("../x.ply", id="a .. part"),
    pytest.param("a/b.ply", id="a slash"),
    pytest.param("a\\b.ply", id="a backslash"),
    pytest.param("C:clean.ply", id="a drive"),
    pytest.param("/abs.ply", id="absolute"),
    pytest.param("clean.ply:x", id="a stream"),
    pytest.param(".", id="a dot"),
    pytest.param("..", id="two dots"),
    pytest.param("", id="empty"),
    pytest.param(7, id="a number"),
]
# Names holding a control character, which a refusal never shows.
CONTROL_NAMES = [
    pytest.param("clean\x00.ply", id="a NUL"),
    pytest.param("clean\n.ply", id="a line break"),
]


@pytest.mark.parametrize("key", NAME_SETTINGS)
@pytest.mark.parametrize("name", [*NOT_PLAIN_NAMES, *CONTROL_NAMES])
def test_a_value_that_isnt_a_plain_file_name_is_refused_naming_the_setting(
        key: str, name: object) -> None:
    assert key in refusal(**{key: name})


@pytest.mark.parametrize("key", NAME_SETTINGS)
@pytest.mark.parametrize("name", CONTROL_NAMES)
def test_a_name_holding_a_control_character_isnt_shown(key: str, name: str) -> None:
    message = refusal(**{key: name})
    assert name not in message and repr(name)[1:-1] not in message  # neither as is nor escaped
    assert all(ord(char) >= 0x20 for char in message)


# A name without its setting's exact, lower-case ending, and that ending (Req 6.9).
WRONG_ENDINGS = [
    pytest.param("clean_ply", "clean.PLY", ".ply", id="clean_ply = clean.PLY"),
    pytest.param("clean_ply", "clean.sog", ".ply", id="clean_ply = clean.sog"),
    pytest.param("clean_sog", "clean.ply", ".sog", id="clean_sog = clean.ply"),
]


@pytest.mark.parametrize(("key", "name", "ending"), WRONG_ENDINGS)
def test_a_name_without_its_exact_lower_case_ending_is_refused(
        key: str, name: str, ending: str) -> None:
    message = refusal(**{key: name})
    assert key in message and ending in message


# A train_ply that differs only in case from the test config's clean_ply = "clean.ply" or
# clean_sog = "clean.sog", with the setting and the value it clashes with (Req 6.10). clean_ply and
# clean_sog can't be one name ignoring case, because each must end in its own lower-case ending,
# so a clean_sog of "clean.ply" is refused for its ending first, as WRONG_ENDINGS shows.
CASE_CLASHES = [
    pytest.param("Clean.ply", "clean_ply", "clean.ply", id="train_ply and clean_ply"),
    pytest.param("Clean.sog", "clean_sog", "clean.sog", id="train_ply and clean_sog"),
]


@pytest.mark.parametrize(("train_ply", "other", "other_name"), CASE_CLASHES)
def test_two_names_that_differ_only_in_case_are_refused_naming_both(
        train_ply: str, other: str, other_name: str) -> None:
    message = refusal(train_ply=train_ply)
    for word in ("train_ply", train_ply, other, other_name):
        assert word in message
    assert "names that differ only in case are one file on Windows" in message


# Property tests

# Each file-name setting's ending, written out here rather than taken from the code under test.
# train_ply has none.
ENDINGS: dict[str, str | None] = {"train_ply": None, "clean_ply": ".ply", "clean_sog": ".sog"}
# What a Plain_File_Name can't hold: /, \, : and every character below U+0020, the space.
NOT_IN_NAMES = frozenset("/\\:" + "".join(chr(code) for code in range(0x20)))


def is_plain_name(name: str, ending: str | None) -> bool:
    """The oracle for Property 10, which shares no code with check_plain_name: a name that isn't
    empty, . or .., holds nothing in NOT_IN_NAMES, and ends in exactly `ending`, if there is one."""
    return (name not in ("", ".", "..")
            and NOT_IN_NAMES.isdisjoint(name)
            and (ending is None or name[-len(ending):] == ending))


@settings(deadline=None)
@given(names, st.sampled_from(NAME_SETTINGS), three_names())
def test_the_file_name_checks_accept_exactly_the_plain_distinct_names(
        name: str, key: str, three: dict[str, str]) -> None:
    """Feature: clean, Property 10: The file-name checks accept exactly the plain, distinct names

    **Validates: Requirements 1.12, 4.7, 5.15, 6.9, 6.10**
    """
    ending = ENDINGS[key]
    if is_plain_name(name, ending):
        assert check_plain_name(key, name, ending) == name
    else:
        with pytest.raises(ExpectedFailure) as caught:
            check_plain_name(key, name, ending)
        assert key in str(caught.value)
    # Three names that each pass are refused exactly when two are the same once case is ignored.
    clashes = [(first, second) for first, second in itertools.combinations(NAME_SETTINGS, 2)
               if three[first].casefold() == three[second].casefold()]
    if clashes:
        message = refusal(**three)
        assert any(first in message and second in message for first, second in clashes)
    else:
        checked = clean_settings(**three)
        assert {setting: getattr(checked, setting) for setting in NAME_SETTINGS} == three
