"""Tests for the clean step's settings checks, in splatpipe/clean/settings.py (Req 6.6 to 6.10),
and its limits, in splatpipe/clean/limits.py (Req 3).

The tests of the kinds, ranges and names build their settings with clean_settings: the test
config's [clean] for the fast Preset, with one setting changed. The tests of the eighteen values
and of a missing setting read the test config itself, with load_config and set_clean.
Property 10 checks the file-name checks against an oracle of its own, on the names that
tests/splats.py draws.

The limits have a part of their own after that. Its unit tests call check_source_count and
check_limits on counts they choose, with a Kept made from those counts rather than by
run_filters, and settings from clean_settings. A count one over max_source_splats is refused
with both numbers and the note on more memory, and one equal to it passes (Req 3.1). A
Removed_Fraction over the Removal_Cap fails with the source count, each filter's drops, the
Removed_Fraction, the cap, the note that broken Gaussians don't count toward it and the settings
to loosen, while one exactly at the cap passes, and so does a splat that's 90% broken with
nothing else dropped (Req 3.2, 3.3, A4). One Gaussian under min_kept_splats fails with the
counts and the settings to loosen, and exactly min_kept_splats or one over passes (Req 3.5).
warn_kept_fraction warns the same way, with both counts and the Removed_Fraction (Req 3.6). A
splat over the cap and under min_kept_splats gets the cap's failure (Req 3.4, 3.7). Each failure
is an ExpectedFailure whose message holds no traceback (Req 3.8). Property 7 runs run_filters,
inside np.errstate(all="raise"), and then check_limits, on Splat_Data from splats and settings
from valid_settings. check_limits stops at the Removal_Cap exactly when the Gaussians that pass
the Broken_Filter, by Property 2's oracle can_be_drawn, and aren't kept are more than
max_removed_fraction of the source count, so broken Gaussians never count toward the cap
(Req 2.23, 3.2, 3.4).
"""

import dataclasses
import itertools
import re
from collections.abc import Callable
from typing import NamedTuple

import numpy as np
import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

from splatpipe.clean.filters import Kept, run_filters
from splatpipe.clean.limits import check_limits, check_source_count
from splatpipe.clean.settings import check_plain_name, check_settings
from splatpipe.config import load_config
from splatpipe.errors import ExpectedFailure, PipelineError

from conftest import CLEAN_PRESETS, CLEAN_SHARED
from splats import PRESET, clean_settings, names, splats, three_names, valid_settings
from test_clean_filters import can_be_drawn

# The three Presets, written out here rather than taken from the code under test.
PRESET_NAMES = ("fast", "default", "quality")


def refusal(**values: object) -> str:
    """The message of the ExpectedFailure that check_settings raises for the test config's
    settings with `values` changed, as clean_settings builds them. It must hold no traceback.

    A refusal can echo the values, and Property 10 can draw a name like Traceback.ply, so the
    check reads only the words the code wrote itself: the message with each value taken out, as
    its repr and as its str. The longest go first, so a short name can't split a longer one.
    """
    with pytest.raises(ExpectedFailure) as caught:
        clean_settings(**values)
    message = str(caught.value)
    own_words = message
    shown = [text for value in values.values() for text in (repr(value), str(value))]
    for text in sorted(shown, key=len, reverse=True):
        own_words = own_words.replace(text, "")
    assert "Traceback" not in own_words
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
# A clash whose refusal echoes the word Traceback in both names, which the no-traceback check
# must not take for a traceback.
@example(name="Traceback.ply", key="train_ply",
         three={"train_ply": "Traceback.ply", "clean_ply": "traceback.ply",
                "clean_sog": "clean.sog"})
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


# The limits, in limits.py (Req 3)

# The settings in [clean] that make the faint and isolated filters drop fewer Gaussians, which
# both of check_limits' failures name (Req 3.3, 3.5), written out here rather than taken from the
# code under test.
FILTER_SETTINGS = ("min_opacity", "faint_min_per_cube", "faint_min_paint_share",
                   "faint_cube_divisions", "isolation_ratio")
# A filter's name and the first number after it, which is how many it dropped.
DROPS = re.compile(r"\b(broken|faint|isolated) filter\b\D*(\d+)")


def limits(source: int, *, broken: int = 0, faint: int = 0, isolated: int = 0,
           **values: object) -> str | None:
    """What check_limits gives for a trained splat of `source` Gaussians when the broken, faint
    and isolated filters drop `broken`, `faint` and `isolated` of them: the warning, or None.

    The settings are the test config's with `values` changed, as clean_settings builds them. Its
    limits are the real cap and warning, 0.7 and 0.5, and a min_kept_splats of 1.
    """
    left = source - broken - faint - isolated
    kept = Kept(np.arange(left), broken, faint, isolated)
    return check_limits(source, kept, clean_settings(**values))


def source_count(count: int, **values: object) -> None:
    """check_source_count for a trained splat of `count` Gaussians, under the test config's
    settings with `values` changed."""
    check_source_count(count, clean_settings(**values))


def stopped(check: Callable[..., object], count: int, **values: object) -> str:
    """The message of the ExpectedFailure that check(count, **values) raises, where `check` is
    limits or source_count. It must hold no traceback.

    As in refusal, the check reads only the words the code wrote itself: the message with each
    value the test passed in taken out, as its repr and as its str, the longest first.
    """
    with pytest.raises(ExpectedFailure) as caught:
        check(count, **values)
    message = str(caught.value)
    own_words = message
    shown = [text for value in (count, *values.values()) for text in (repr(value), str(value))]
    for text in sorted(shown, key=len, reverse=True):
        own_words = own_words.replace(text, "")
    assert "Traceback" not in own_words
    return message


def numbers_in(message: str) -> set[str]:
    """Every number `message` shows, as it's written: whole, so 100 isn't found inside 1000, and
    a percentage with its % sign, like 75.0%."""
    return set(re.findall(r"\d+(?:\.\d+)?%?", message))


def drops_in(message: str) -> dict[str, int]:
    """How many each filter dropped, as `message` gives it after the filter's name."""
    return {name: int(count) for name, count in DROPS.findall(message)}


def test_a_count_one_over_max_source_splats_is_refused_with_both_numbers() -> None:
    message = stopped(source_count, 101, max_source_splats=100)
    assert {"101", "100"} <= numbers_in(message)
    for words in ("max_source_splats", "[clean]", "raise", "more memory"):
        assert words in message


def test_a_count_equal_to_max_source_splats_passes() -> None:
    assert source_count(100, max_source_splats=100) is None


def test_over_the_cap_fails_with_the_counts_the_share_the_cap_and_what_to_change() -> None:
    # The faint and isolated filters drop 750 of 1000, 75%, over a cap of 70%. With the 100
    # broken ones it would be 85%, but they don't count toward the cap.
    message = stopped(limits, 1000, broken=100, faint=500, isolated=250,
                      max_removed_fraction=0.7)
    assert {"1000", "75.0%", "70.0%"} <= numbers_in(message)
    assert drops_in(message) == {"broken": 100, "faint": 500, "isolated": 250}
    assert "broken gaussians" in message.lower() and "don't count toward the cap" in message
    for words in (*FILTER_SETTINGS, "max_removed_fraction", "[clean]"):
        assert words in message


def test_a_removed_fraction_exactly_at_the_cap_passes() -> None:
    # 7 of 10 at a cap of 0.7. A warn_kept_fraction of 0 never warns, so passing gives None.
    assert limits(10, faint=4, isolated=3, max_removed_fraction=0.7,
                  warn_kept_fraction=0) is None


def test_a_splat_90_percent_broken_with_nothing_else_dropped_passes_a_cap_of_0_7() -> None:
    # Broken Gaussians don't count toward the cap, so the Removed_Fraction is 0 (A4).
    assert limits(10, broken=9, max_removed_fraction=0.7, warn_kept_fraction=0) is None


def test_one_under_min_kept_splats_left_fails_with_the_counts_and_what_to_change() -> None:
    # 599 of 1000 left, one under a min_kept_splats of 600. The faint and isolated filters drop
    # 151, under the cap.
    message = stopped(limits, 1000, broken=250, faint=100, isolated=51, min_kept_splats=600)
    assert {"1000", "599"} <= numbers_in(message)
    assert drops_in(message) == {"broken": 250, "faint": 100, "isolated": 51}
    for words in (*FILTER_SETTINGS, "min_kept_splats", "[clean]"):
        assert words in message


# Exactly min_kept_splats left, and one over, with the isolated filter dropping one fewer for each
# Gaussian more left. More than half is kept, so warn_kept_fraction's 0.5 gives no warning.
@pytest.mark.parametrize("left", [600, 601], ids=["exactly", "one over"])
def test_exactly_min_kept_splats_left_or_one_over_passes(left: int) -> None:
    assert limits(1000, broken=250, faint=100, isolated=650 - left, min_kept_splats=600) is None


def test_one_under_warn_kept_fraction_warns_with_both_counts_and_the_removed_fraction() -> None:
    # 49 of 100 kept, one under a warn_kept_fraction of 0.5. The faint and isolated filters drop
    # 41, 41%, under the cap.
    warning = limits(100, broken=10, faint=20, isolated=21, warn_kept_fraction=0.5)
    assert warning is not None
    assert {"49", "100", "41.0%"} <= numbers_in(warning)


# Exactly warn_kept_fraction's 50 of 100 kept, and one over, the isolated filter dropping one
# fewer for each Gaussian more kept.
@pytest.mark.parametrize("left", [50, 51], ids=["exactly", "one over"])
def test_exactly_warn_kept_fraction_or_one_over_gives_no_warning(left: int) -> None:
    assert limits(100, broken=10, faint=20, isolated=70 - left, warn_kept_fraction=0.5) is None


def test_a_splat_over_the_cap_and_under_min_kept_splats_gets_the_caps_failure() -> None:
    # The faint and isolated filters drop 800 of 1000, over a cap of 0.7, and leave 200, under a
    # min_kept_splats of 500. A cap of 1 stops nothing, and 200 left pass a min_kept_splats of 1.
    drops = {"faint": 600, "isolated": 200}
    over_both = stopped(limits, 1000, **drops, max_removed_fraction=0.7, min_kept_splats=500)
    over_the_cap = stopped(limits, 1000, **drops, max_removed_fraction=0.7, min_kept_splats=1)
    too_few_left = stopped(limits, 1000, **drops, max_removed_fraction=1.0, min_kept_splats=500)
    assert over_both == over_the_cap != too_few_left


def first_sentence(message: str) -> str:
    """The first sentence of `message`, up to its first full stop followed by a space. A
    percentage like 70.0% holds a full stop too, but a digit follows it."""
    return message.split(". ", 1)[0]


@settings(deadline=None)
@given(splats(), valid_settings)
def test_the_removal_cap_holds_and_broken_gaussians_dont_count_toward_it(
        data: np.ndarray, values: dict[str, object]) -> None:
    """Feature: clean, Property 7: The Removal_Cap holds, and broken Gaussians don't count toward it

    **Validates: Requirements 2.23, 3.2, 3.4**
    """
    s = clean_settings(**values)
    source = len(data)
    with np.errstate(all="raise"):  # as in every filter test (Req 2.11)
        kept = run_filters(data, s, add_count=lambda name, value: None)  # outside a Step
    # The test's own count of the Capped_Removals: the Gaussians that pass the Broken_Filter, by
    # Property 2's oracle, but aren't kept. So no broken Gaussian is among them.
    drawable = {row for row, gaussian in enumerate(data) if can_be_drawn(gaussian)}
    capped = len(drawable - set(kept.rows.tolist()))
    try:
        check_limits(source, kept, s)
    except ExpectedFailure as failure:
        # check_limits' two failures, told apart by their first sentences: the cap's names
        # max_removed_fraction, and the other one names min_kept_splats.
        first = first_sentence(str(failure))
        stopped_at_cap = "max_removed_fraction" in first
        assert stopped_at_cap or "min_kept_splats" in first
    else:
        stopped_at_cap = False
    # Compared in float64, as check_limits compares them, so a share exactly at the cap passes.
    if stopped_at_cap:
        assert capped / source > s.max_removed_fraction
    else:
        # Whatever else check_limits refused or warned about.
        assert capped / source <= s.max_removed_fraction
