"""Tests for the Frames_Step's settings: [frames] in config.toml and splatpipe/frames/settings.py.

The real config is read by path, because tests/conftest.py points CONFIG_PATH at a test config.
The comment check in tests/test_config.py reads every key of the real config, [frames] included,
so no test here checks the comments. Every other test reads the test config, whose [frames] has
the real values except min_picks = 1, base_picks 2, 3 and 4, and both timeouts at 60 s. The
set_frames fixture changes it for one test (Req 9.11).
"""

import tomllib
from collections.abc import Callable, Mapping
from dataclasses import asdict
from typing import Any

import pytest

from splatpipe.config import REPO_ROOT, StepSettings, load_config
from splatpipe.errors import ExpectedFailure, PipelineError
from splatpipe.frames.settings import FramesSettings, read_settings

# Read by path, so these tests still read it once CONFIG_PATH points at a test config.
REAL_CONFIG = REPO_ROOT / "config.toml"
# The three Presets, written out here rather than taken from the code under test.
PRESET_NAMES = ("fast", "default", "quality")

# The 22 Shared_Settings of the real [frames] with the values of Req 9.2, each an int or a float
# as Req 9.2 writes it.
REAL_SHARED = {
    "score_width": 320,
    "tonemap_npl": 100,
    "tonemap_desat": 0.0,
    "still_motion": 1.0,
    "motion_window_seconds": 1,
    "min_still_seconds": 2,
    "blip_seconds": 0.5,
    "jolt_ratio": 0.6,
    "jolt_window_seconds": 1,
    "mask_samples": 240,
    "mask_ratio": 0.2,
    "mask_min_coverage": 0.002,
    "mask_max_coverage": 0.25,
    "jpeg_quality": 95,
    "long_edge": 1600,
    "seconds_per_pick": 2,
    "max_picks": 400,
    "min_picks": 30,
    "min_vcpus": 16,
    "min_memory_gb": 16,
    "ffmpeg_timeout_seconds": 1800,
    "ffprobe_timeout_seconds": 30,
}
# The Preset_Setting base_picks in each Preset table of the real [frames] (Req 9.3).
REAL_BASE_PICKS = {"fast": 120, "default": 150, "quality": 200}
# The test config's base_picks in each Preset table, as tests/conftest.py writes them.
TEST_BASE_PICKS = {"fast": 2, "default": 3, "quality": 4}

# What each kind of value must be, in the words of Req 9.6.
WHOLE_ABOVE_0 = "a whole number above 0"
WHOLE_2_OR_MORE = "a whole number of 2 or more"
WHOLE_1_TO_100 = "a whole number from 1 to 100"
ABOVE_0 = "a number above 0"
ZERO_OR_MORE = "a number of 0 or more"
ZERO_TO_1 = "a number from 0 to 1"

# Every key of [frames] and its Preset tables, with what its value must be (Req 9.6).
KINDS = {
    "score_width": WHOLE_ABOVE_0,
    "tonemap_npl": ABOVE_0,
    "tonemap_desat": ZERO_OR_MORE,
    "still_motion": ABOVE_0,
    "motion_window_seconds": ABOVE_0,
    "min_still_seconds": ABOVE_0,
    "blip_seconds": ABOVE_0,
    "jolt_ratio": ZERO_TO_1,
    "jolt_window_seconds": ABOVE_0,
    "mask_samples": WHOLE_2_OR_MORE,
    "mask_ratio": ZERO_TO_1,
    "mask_min_coverage": ZERO_TO_1,
    "mask_max_coverage": ZERO_TO_1,
    "jpeg_quality": WHOLE_1_TO_100,
    "long_edge": WHOLE_ABOVE_0,
    "seconds_per_pick": ABOVE_0,
    "max_picks": WHOLE_ABOVE_0,
    "min_picks": WHOLE_ABOVE_0,
    "min_vcpus": WHOLE_ABOVE_0,
    "min_memory_gb": WHOLE_ABOVE_0,
    "ffmpeg_timeout_seconds": ABOVE_0,
    "ffprobe_timeout_seconds": ABOVE_0,
    "base_picks": WHOLE_ABOVE_0,
}
# Wrong values for each kind: a bool, which is never a number here, a value outside the kind's
# range at each of its ends, and a value of the wrong type, a float for a whole number and text
# for a number.
WRONG_VALUES = {
    WHOLE_ABOVE_0: [True, 0, 1.5],
    WHOLE_2_OR_MORE: [True, 1, 2.5],
    WHOLE_1_TO_100: [True, 0, 101, 95.0],
    ABOVE_0: [True, 0, "1"],
    ZERO_OR_MORE: [True, -0.5, "0"],
    ZERO_TO_1: [True, -0.1, 1.5, "0.5"],
}


def typed(table: Mapping[str, Any]) -> dict[str, tuple[type, Any]]:
    """Each value of `table` with its type, so that 100 and 100.0, or 1 and true, don't compare
    equal."""
    return {key: (type(value), value) for key, value in table.items()}


def table_of(key: str, preset: str) -> str:
    """The table that holds `key`: [frames.<preset>] for base_picks, else [frames]."""
    return f"[frames.{preset}]" if key == "base_picks" else "[frames]"


def frames_settings(preset: str = "fast", **changes: Any) -> StepSettings:
    """The test config's [frames] settings for `preset`, as the Step_Runner gives them to the
    step, with each key in `changes` set to its value."""
    settings = load_config().step_settings("frames", preset)
    return StepSettings({**settings, **changes}, section="frames", preset=preset)


def read(settings: StepSettings) -> FramesSettings:
    """read_settings on `settings`, with the section and the Preset they're for, as the step
    calls it."""
    return read_settings(settings, settings.section, settings.preset)


def refusal(settings: StepSettings) -> str:
    """The message of the ExpectedFailure that read_settings raises on `settings`."""
    with pytest.raises(ExpectedFailure) as caught:
        read(settings)
    return str(caught.value)


# The real config.toml

def test_the_real_frames_section_has_exactly_the_22_shared_settings_and_base_picks() -> None:
    section = tomllib.loads(REAL_CONFIG.read_text(encoding="utf-8"))["frames"]
    tables = {key: value for key, value in section.items() if isinstance(value, dict)}
    shared = {key: value for key, value in section.items() if key not in tables}
    assert typed(shared) == typed(REAL_SHARED)  # Req 9.2
    assert {name: typed(table) for name, table in tables.items()} == {
        preset: typed({"base_picks": base_picks})
        for preset, base_picks in REAL_BASE_PICKS.items()}  # Req 9.1, 9.3


@pytest.mark.parametrize("preset", PRESET_NAMES)
def test_the_real_frames_section_loads_for_each_preset(preset: str) -> None:
    settings = load_config(REAL_CONFIG).step_settings("frames", preset)
    assert settings == {**REAL_SHARED, "base_picks": REAL_BASE_PICKS[preset]}
    assert read(settings).base_picks == REAL_BASE_PICKS[preset]


# The checks of read_settings

WRONG = [pytest.param(key, value, id=f"{key} {value!r}")
         for key, kind in KINDS.items() for value in WRONG_VALUES[kind]]


@pytest.mark.parametrize(("key", "value"), WRONG)
def test_a_wrong_value_is_refused_naming_the_key_its_table_and_what_it_must_be(
        key: str, value: object) -> None:
    # quality, so the Preset table named is the chosen one, not the test config's default_preset.
    message = refusal(frames_settings("quality", **{key: value}))
    assert f"{key} in {table_of(key, 'quality')}" in message
    assert f"must be {KINDS[key]}" in message


# Values at the ends of their ranges, which pass (Req 9.6). The two coverages change together, so
# mask_min_coverage stays no greater than mask_max_coverage.
EDGE_VALUES = [
    pytest.param({"jpeg_quality": 1}, id="jpeg_quality 1"),
    pytest.param({"jpeg_quality": 100}, id="jpeg_quality 100"),
    pytest.param({"mask_samples": 2}, id="mask_samples 2"),
    pytest.param({"jolt_ratio": 0}, id="jolt_ratio 0"),
    pytest.param({"jolt_ratio": 1}, id="jolt_ratio 1"),
    pytest.param({"mask_ratio": 0}, id="mask_ratio 0"),
    pytest.param({"mask_ratio": 1}, id="mask_ratio 1"),
    pytest.param({"mask_min_coverage": 0, "mask_max_coverage": 0}, id="both coverages 0"),
    pytest.param({"mask_min_coverage": 1, "mask_max_coverage": 1}, id="both coverages 1"),
]


@pytest.mark.parametrize("changes", EDGE_VALUES)
def test_the_values_at_the_ends_of_their_ranges_pass(changes: dict[str, Any]) -> None:
    frames = read(frames_settings(**changes))
    assert {key: getattr(frames, key) for key in changes} == changes


# The Tone_Map's numbers (A13): each wrong value with how its refusal shows it.
TONE_MAP_REFUSALS = [
    pytest.param("tonemap_npl", 0, "0", id="tonemap_npl 0"),
    pytest.param("tonemap_npl", True, "true", id="tonemap_npl true"),
    pytest.param("tonemap_desat", -0.5, "-0.5", id="tonemap_desat -0.5"),
    pytest.param("tonemap_desat", True, "true", id="tonemap_desat true"),
]


@pytest.mark.parametrize(("key", "value", "shown"), TONE_MAP_REFUSALS)
def test_a_wrong_tone_map_number_is_refused_naming_the_key_frames_and_what_it_must_be(
        key: str, value: object, shown: str) -> None:
    message = refusal(frames_settings(**{key: value}))
    assert message.startswith(f"The setting {key} in [frames] is {shown}, but it must be "
                              f"{KINDS[key]}.")


def test_a_tonemap_desat_of_0_passes_as_the_float_0() -> None:
    frames = read(frames_settings(tonemap_desat=0))
    assert (type(frames.tonemap_desat), frames.tonemap_desat) == (float, 0.0)


# Pairs whose low setting is above its high one (Req 9.6, 9.7). Each pair breaks one cross check,
# and the test config's other values pass the other two.
CROSS_CHECKS = [
    pytest.param("mask_min_coverage", 0.3, "mask_max_coverage", 0.125, id="mask coverages"),
    pytest.param("min_picks", 37, "base_picks", 36, id="min_picks above base_picks"),
    pytest.param("base_picks", 56, "max_picks", 55, id="base_picks above max_picks"),
]


@pytest.mark.parametrize(("low", "low_value", "high", "high_value"), CROSS_CHECKS)
def test_a_cross_check_names_both_settings_and_their_values(
        low: str, low_value: object, high: str, high_value: object) -> None:
    message = refusal(frames_settings(**{low: low_value, high: high_value}))
    assert f"{low} in {table_of(low, 'fast')} is {low_value!r}" in message
    assert f"{high} in {table_of(high, 'fast')}, which is {high_value!r}" in message


@pytest.mark.parametrize("key", list(KINDS))
def test_a_missing_key_gives_the_step_runners_message_naming_both_tables_and_the_key(
        key: str) -> None:
    settings = frames_settings("quality")
    del settings[key]
    with pytest.raises(PipelineError) as caught:
        read(settings)
    with pytest.raises(PipelineError) as runners:
        settings[key]  # the Step_Runner's own message for the missing key (Req 9.5)
    message = str(caught.value)
    assert message == str(runners.value)
    for word in ("[frames]", "[frames.quality]", key):
        assert word in message


# The test config and set_frames

@pytest.mark.parametrize(("preset", "base_picks"), list(TEST_BASE_PICKS.items()))
def test_read_settings_on_the_test_config_gives_the_presets_base_picks_and_float_tone_map_numbers(
        preset: str, base_picks: int) -> None:
    settings = load_config().step_settings("frames", preset)
    frames = read(settings)
    assert type(frames) is FramesSettings
    assert asdict(frames) == settings  # each key's value in the field of its name
    assert frames.base_picks == base_picks
    tone_map = [(type(value), value) for value in (frames.tonemap_npl, frames.tonemap_desat)]
    assert tone_map == [(float, 100.0), (float, 0.0)]


# What set_frames can change: a Shared_Setting, which every Preset reads, or base_picks for one
# Preset, which only that Preset reads. Each with the change each Preset then sees.
FRAMES_CHANGES = [
    pytest.param({"mask_samples": 3}, {preset: {"mask_samples": 3} for preset in PRESET_NAMES},
                 id="a shared setting"),
    pytest.param({"base_picks": {"default": 50}},
                 {"fast": {}, "default": {"base_picks": 50}, "quality": {}},
                 id="base_picks for one preset"),
]


@pytest.mark.parametrize(("values", "seen"), FRAMES_CHANGES)
def test_set_frames_changes_only_the_key_it_names(
        set_frames: Callable[..., None], values: dict[str, Any],
        seen: dict[str, dict[str, Any]]) -> None:
    before = {preset: load_config().step_settings("frames", preset) for preset in PRESET_NAMES}
    set_frames(**values)
    after = {preset: load_config().step_settings("frames", preset) for preset in PRESET_NAMES}
    assert after == {preset: {**before[preset], **seen[preset]} for preset in PRESET_NAMES}


def test_the_next_test_sees_the_test_configs_values_again() -> None:
    """pytest runs a file's tests in order, so this runs after the set_frames tests above."""
    for preset, base_picks in TEST_BASE_PICKS.items():
        settings = load_config().step_settings("frames", preset)
        assert (settings["mask_samples"], settings["base_picks"]) == (240, base_picks)
