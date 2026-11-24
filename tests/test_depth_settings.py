"""Tests for the Depth_Step's settings: [depth] in config.toml and splatpipe/depth/settings.py.

The real config is read by path, because tests/conftest.py points CONFIG_PATH at a test config.
The comment check in tests/test_config.py reads every key of the real config, [depth] included,
so no test here checks the comments. Every other test reads the test config, whose [depth] has
the real values of Req 8.2 and three empty Preset tables. The set_depth fixture changes it for
one test (Req 8.8).

Every [depth] setting is a Shared_Setting, so every refusal names [depth], whatever the Preset.
The tests use the quality Preset, so a message that named the Preset's table, or the test
config's default_preset, would show.

The Fallback_List test reads the real config's [cloud] too, with cloud.config's own loader. With
the needs the Start_Command works out for the real Step_List, which holds the Depth_Step, the GPU
memory needed is the train step's 24, 24 and 48 GB, which covers the depth step's 16, and
select_candidates keeps every type on each Preset's Fallback_List (Req 8.5, 18.10).
"""

import tomllib
from collections.abc import Callable, Mapping
from dataclasses import asdict
from typing import Any

import pytest

from cloud.config import load_cloud_config
from cloud.image import MOGE_WEIGHTS
from cloud.select import select_candidates, step_needs_over
from splatpipe.config import REPO_ROOT, StepSettings, load_config
from splatpipe.depth.settings import DepthSettings, read_settings
from splatpipe.errors import ExpectedFailure, PipelineError
from splatpipe.run import STEP_LIST

# Read by path, so these tests still read it once CONFIG_PATH points at a test config.
REAL_CONFIG = REPO_ROOT / "config.toml"
# The three Presets, written out here rather than taken from the code under test.
PRESET_NAMES = ("fast", "default", "quality")

# The 8 Shared_Settings of the real [depth] with the values of Req 8.2, each with the TOML type
# Req 8.2 writes it with, so fit_miss_ratio is the whole number 3.
REAL_SHARED = {
    "long_edge": 512,
    "moge_weights": "/opt/models/moge-2-vitb-normal/model.pt",
    "resolution_level": 9,
    "use_fp16": True,
    "fit_min_points": 10,
    "fit_miss_ratio": 3,
    "fit_max_median_error": 0.15,
    "min_gpu_memory_gb": 16,
}

# What each kind of value must be, in the words of Req 8.3.
WHOLE_ABOVE_0 = "a whole number above 0"
WHOLE_0_TO_9 = "a whole number from 0 to 9"
TRUE_OR_FALSE = "true or false"
TEXT = "a string of one or more characters"
ABOVE_0 = "a number above 0"

# Every key of [depth], with what its value must be (Req 8.3).
KINDS = {
    "long_edge": WHOLE_ABOVE_0,
    "moge_weights": TEXT,
    "resolution_level": WHOLE_0_TO_9,
    "use_fp16": TRUE_OR_FALSE,
    "fit_min_points": WHOLE_ABOVE_0,
    "fit_miss_ratio": ABOVE_0,
    "fit_max_median_error": ABOVE_0,
    "min_gpu_memory_gb": WHOLE_ABOVE_0,
}
# The type each kind's value is kept as: a number above 0 as a float, any other value as it is.
KEPT_AS = {WHOLE_ABOVE_0: int, WHOLE_0_TO_9: int, TRUE_OR_FALSE: bool, TEXT: str, ABOVE_0: float}

# Wrong values for each kind, each with how the refusal shows it, as TOML writes it. For a number:
# a bool, which is never a number here, a value outside the kind's range at each of its ends, and
# a value of the wrong type, a float for a whole number and text for a number. For true or false:
# the numbers 1 and 0.0 and the text "true". For a string: the empty string and a number.
WRONG_VALUES: dict[str, list[tuple[object, str]]] = {
    WHOLE_ABOVE_0: [(True, "true"), (0, "0"), (-3, "-3"), (1.0, "1.0")],
    WHOLE_0_TO_9: [(True, "true"), (-1, "-1"), (10, "10"), (9.0, "9.0")],
    TRUE_OR_FALSE: [(1, "1"), (0.0, "0.0"), ("true", '"true"')],
    TEXT: [("", '""'), (1, "1")],
    ABOVE_0: [(True, "true"), (0, "0"), (-0.5, "-0.5"), ("1", '"1"')],
}


def typed(table: Mapping[str, Any]) -> dict[str, tuple[type, Any]]:
    """Each value of `table` with its type, so that 3 and 3.0, or 1 and true, don't compare
    equal."""
    return {key: (type(value), value) for key, value in table.items()}


def depth_settings(preset: str = "quality", **changes: Any) -> StepSettings:
    """The test config's [depth] settings for `preset`, as the Step_Runner gives them to the
    step, with each key in `changes` set to its value."""
    settings = load_config().step_settings("depth", preset)
    return StepSettings({**settings, **changes}, section="depth", preset=preset)


def read(settings: StepSettings) -> DepthSettings:
    """read_settings on `settings`, with the section and the Preset they're for, as the step
    calls it."""
    return read_settings(settings, settings.section, settings.preset)


def refusal(settings: StepSettings) -> str:
    """The message of the ExpectedFailure that read_settings raises on `settings`."""
    with pytest.raises(ExpectedFailure) as caught:
        read(settings)
    return str(caught.value)


# The real config.toml

def test_the_real_depth_section_has_exactly_the_8_settings_and_empty_preset_tables() -> None:
    section = tomllib.loads(REAL_CONFIG.read_text(encoding="utf-8"))["depth"]
    tables = {key: value for key, value in section.items() if isinstance(value, dict)}
    shared = {key: value for key, value in section.items() if key not in tables}
    assert typed(shared) == typed(REAL_SHARED)  # Req 8.2
    assert tables == {preset: {} for preset in PRESET_NAMES}  # Req 8.1


@pytest.mark.parametrize("preset", PRESET_NAMES)
def test_the_real_depth_section_loads_exactly_the_8_keys_for_each_preset(preset: str) -> None:
    settings = load_config(REAL_CONFIG).step_settings("depth", preset)
    assert typed(settings) == typed(REAL_SHARED)
    depth = read(settings)  # the real values pass the step's checks
    assert type(depth) is DepthSettings
    assert asdict(depth) == settings  # each key's value in the field of its name
    # fit_miss_ratio is written as the whole number 3, and is kept as a float.
    assert {key: type(value) for key, value in asdict(depth).items()} == {
        key: KEPT_AS[kind] for key, kind in KINDS.items()}


def test_the_real_moge_weights_is_the_machine_images_weights_file() -> None:
    section = tomllib.loads(REAL_CONFIG.read_text(encoding="utf-8"))["depth"]
    assert section["moge_weights"] == MOGE_WEIGHTS  # Req 8.4


# The Step_Needs and the Fallback_Lists

# The GPU memory the Start_Command needs over the real Step_List for each Preset, in GB: the train
# step's min_gpu_memory_gb, which is above the depth step's 16 (Req 8.5).
NEEDED_GPU_MEMORY_GB = {"fast": 24.0, "default": 24.0, "quality": 48.0}


@pytest.mark.parametrize("preset", PRESET_NAMES)
def test_every_type_on_each_presets_fallback_list_still_fits_the_real_step_list_with_depth(
        preset: str) -> None:
    """Req 8.5, 18.10. The needs come from the Start_Command's own step_needs_over over the real
    Step_List's names, as cloud/start.py works them out, so the depth step's min_gpu_memory_gb
    counts with no change to the cloud code."""
    names = [entry.name for entry in STEP_LIST]
    assert "depth" in names
    needs = step_needs_over(names, load_config(REAL_CONFIG), preset)
    assert needs.min_gpu_memory_gb == NEEDED_GPU_MEMORY_GB[preset]
    cloud_config = load_cloud_config(REAL_CONFIG)
    fallback = list(cloud_config.preset_plan(preset).fallback)
    assert select_candidates(preset, needs, cloud_config) == fallback  # every type is kept


# The checks of read_settings

WRONG = [pytest.param(key, value, shown, id=f"{key} {value!r}")
         for key, kind in KINDS.items() for value, shown in WRONG_VALUES[kind]]


@pytest.mark.parametrize(("key", "value", "shown"), WRONG)
def test_a_wrong_value_is_refused_naming_the_key_depth_and_what_it_must_be(
        key: str, value: object, shown: str) -> None:
    message = refusal(depth_settings(**{key: value}))
    wanted = KINDS[key]
    assert message == (f"The setting {key} in [depth] is {shown}, but it must be {wanted}. "
                       f"Change it to {wanted}.")  # Req 8.3


@pytest.mark.parametrize("value", [0, 9])
def test_a_resolution_level_at_either_end_of_its_range_passes(value: int) -> None:
    depth = read(depth_settings(resolution_level=value))
    assert (type(depth.resolution_level), depth.resolution_level) == (int, value)


@pytest.mark.parametrize("key", list(KINDS))
def test_a_missing_key_gives_the_step_runners_message_naming_both_tables_and_the_key(
        key: str) -> None:
    settings = depth_settings()
    del settings[key]
    with pytest.raises(PipelineError) as caught:
        read(settings)
    with pytest.raises(PipelineError) as runners:
        settings[key]  # the Step_Runner's own message for the missing key (Req 8.6)
    message = str(caught.value)
    assert type(caught.value) is PipelineError  # not an ExpectedFailure of read_settings
    assert message == str(runners.value)
    for word in ("[depth]", "[depth.quality]", key):
        assert word in message


# The test config and set_depth

def test_set_depth_changes_only_the_key_it_names_for_every_preset(
        set_depth: Callable[..., None]) -> None:
    before = {preset: load_config().step_settings("depth", preset) for preset in PRESET_NAMES}
    set_depth(fit_min_points=3)
    after = {preset: load_config().step_settings("depth", preset) for preset in PRESET_NAMES}
    assert after == {preset: {**before[preset], "fit_min_points": 3} for preset in PRESET_NAMES}
    assert read(depth_settings()).fit_min_points == 3


def test_the_next_test_sees_the_test_configs_values_again() -> None:
    """pytest runs a file's tests in order, so this runs after the set_depth test above. The test
    config's [depth] has the real values (Req 8.8)."""
    for preset in PRESET_NAMES:
        assert typed(load_config().step_settings("depth", preset)) == typed(REAL_SHARED)
