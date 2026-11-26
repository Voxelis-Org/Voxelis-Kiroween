"""Tests for the Train_Step's settings: [train] in config.toml and splatpipe/train/settings.py.

The real config is read by path, because tests/conftest.py points CONFIG_PATH at a test config.
The comment check in tests/test_config.py reads every key of the real config, [train] included,
so no test here checks the comments. The Fallback_List test reads the real config's [cloud] too,
with cloud.config's own loader, and checks that, with the needs the Start_Command works out for
the real Step_List, select_candidates keeps every type on each Preset's Fallback_List (Req 9.9,
14.9). Every other test reads the test config, whose [train] has the real Shared_Settings,
steps 6, 8 and 10, max_splats 50, 60 and 70, and the real min_gpu_memory_gb, 24, 24 and 48. The
set_train fixture changes it for one test (Req 9.15).
"""

import tomllib
from collections.abc import Callable, Mapping
from dataclasses import asdict
from typing import Any

import pytest

from cloud.config import load_cloud_config
from cloud.select import select_candidates, step_needs_over
from splatpipe.config import REPO_ROOT, StepSettings, load_config
from splatpipe.errors import ExpectedFailure, PipelineError
from splatpipe.run import STEP_LIST
from splatpipe.train.settings import TrainSettings, read_settings

# Read by path, so these tests still read it once CONFIG_PATH points at a test config.
REAL_CONFIG = REPO_ROOT / "config.toml"
# The three Presets, written out here rather than taken from the code under test.
PRESET_NAMES = ("fast", "default", "quality")
# The Preset_Settings, which each Preset table [train.<preset>] holds (Req 9.3).
PRESET_KEYS = ("steps", "max_splats", "min_gpu_memory_gb")

# The 15 Shared_Settings that the depth-priors spec adds to [train], with the values of its
# Req 14.1 and in that criterion's order, each an int or a float as it writes it.
NEW_SHARED = {
    "plain_gradient": 0.02,
    "depth_gradient_falloff": 10,
    "depth_weight_start": 1.0,
    "depth_weight_end": 0.01,
    "depth_weight_floor": 0.1,
    "depth_error_cap": 1.0,
    "normal_weight": 0.05,
    "normal_start_step": 3000,
    "normal_every": 4,
    "flatten_reg": 1.0,
    "depth_normal_weight": 0.05,
    "normal_consistency_weight": 0.015,
    "depth_normal_start_fraction": 0.23333333333333334,
    "backproject_share": 0.3333333333333333,
    "backproject_max": 500000,
}
# The 50 Shared_Settings of the real [train] with the values of Req 9.2 and the depth-priors
# spec's Req 14.1, each an int or a float as those criteria write it.
REAL_SHARED = {
    "long_edge": 1600,
    "mask_threshold": 128,
    "seed": 42,
    "sh_degree_interval": 1000,
    "scene_extent_factor": 1.1,
    "near_plane_factor": 0.03,
    "start_opacity": 0.5,
    "start_scale_factor": 0.1,
    "start_scale_neighbours": 3,
    "neighbour_chunk_pairs": 100000000,
    "means_lr": 0.00016,
    "means_lr_final": 0.01,
    "scales_lr": 0.005,
    "quats_lr": 0.001,
    "opacities_lr": 0.05,
    "sh_dc_lr": 0.0025,
    "sh_rest_lr": 0.000125,
    "adam_eps": 1e-15,
    "exposure_lr": 0.001,
    "l1_weight": 0.8,
    "ssim_weight": 0.2,
    "opacity_reg": 0.01,
    "scale_reg": 0.01,
    "erank_reg": 0.01,
    "erank_epsilon": 1e-5,
    "erank_start_fraction": 0.23333333333333334,
    "refine_start_step": 500,
    "refine_every": 100,
    "refine_stop_step": 25000,
    "refine_stop_fraction": 0.9,
    "clean_reserve_minutes": 10,
    "save_margin_minutes": 5,
    "gpu_memory_slack": 0.9,
    **NEW_SHARED,
    "min_vcpus": 4,
    "min_memory_gb": 32,
}
# The Preset_Settings in each Preset table of the real [train] (Req 9.3).
REAL_PRESETS = {
    "fast": {"steps": 7000, "max_splats": 1000000, "min_gpu_memory_gb": 24},
    "default": {"steps": 15000, "max_splats": 2000000, "min_gpu_memory_gb": 24},
    "quality": {"steps": 30000, "max_splats": 3000000, "min_gpu_memory_gb": 48},
}
# The test config's Preset_Settings in each Preset table, as tests/conftest.py writes them.
TEST_PRESETS = {
    "fast": {"steps": 6, "max_splats": 50, "min_gpu_memory_gb": 24},
    "default": {"steps": 8, "max_splats": 60, "min_gpu_memory_gb": 24},
    "quality": {"steps": 10, "max_splats": 70, "min_gpu_memory_gb": 48},
}

# What each kind of value must be, in the words of Req 9.7.
WHOLE_ABOVE_0 = "a whole number above 0"
WHOLE_0_OR_MORE = "a whole number of 0 or more"
WHOLE_1_TO_255 = "a whole number from 1 to 255"
ABOVE_0 = "a number above 0"
ZERO_OR_MORE = "a number of 0 or more"
ABOVE_0_BELOW_1 = "a number above 0 and below 1"
ABOVE_0_AT_MOST_1 = "a number above 0 and at most 1"
ZERO_TO_1 = "a number from 0 to 1"
# The kinds whose values are kept as ints. The values of every other kind are kept as floats.
WHOLE_KINDS = (WHOLE_ABOVE_0, WHOLE_0_OR_MORE, WHOLE_1_TO_255)

# Every key of [train] and its Preset tables, with what its value must be (Req 9.7, and the
# depth-priors spec's Req 14.2).
KINDS = {
    "long_edge": WHOLE_ABOVE_0,
    "mask_threshold": WHOLE_1_TO_255,
    "seed": WHOLE_0_OR_MORE,
    "sh_degree_interval": WHOLE_ABOVE_0,
    "scene_extent_factor": ABOVE_0,
    "near_plane_factor": ABOVE_0,
    "start_opacity": ABOVE_0_BELOW_1,
    "start_scale_factor": ABOVE_0,
    "start_scale_neighbours": WHOLE_ABOVE_0,
    "neighbour_chunk_pairs": WHOLE_ABOVE_0,
    "means_lr": ABOVE_0,
    "means_lr_final": ABOVE_0_AT_MOST_1,
    "scales_lr": ABOVE_0,
    "quats_lr": ABOVE_0,
    "opacities_lr": ABOVE_0,
    "sh_dc_lr": ABOVE_0,
    "sh_rest_lr": ABOVE_0,
    "adam_eps": ABOVE_0,
    "exposure_lr": ABOVE_0,
    "l1_weight": ZERO_OR_MORE,
    "ssim_weight": ZERO_OR_MORE,
    "opacity_reg": ZERO_OR_MORE,
    "scale_reg": ZERO_OR_MORE,
    "erank_reg": ZERO_OR_MORE,
    "erank_epsilon": ABOVE_0,
    "erank_start_fraction": ZERO_TO_1,
    "refine_start_step": WHOLE_0_OR_MORE,
    "refine_every": WHOLE_ABOVE_0,
    "refine_stop_step": WHOLE_0_OR_MORE,
    "refine_stop_fraction": ZERO_TO_1,
    "clean_reserve_minutes": ZERO_OR_MORE,
    "save_margin_minutes": ZERO_OR_MORE,
    "gpu_memory_slack": ABOVE_0_AT_MOST_1,
    "plain_gradient": ABOVE_0,
    "depth_gradient_falloff": ZERO_OR_MORE,
    "depth_weight_start": ABOVE_0,
    "depth_weight_end": ABOVE_0,
    "depth_weight_floor": ZERO_OR_MORE,
    "depth_error_cap": ABOVE_0,
    "normal_weight": ZERO_OR_MORE,
    "normal_start_step": WHOLE_0_OR_MORE,
    "normal_every": WHOLE_ABOVE_0,
    "flatten_reg": ZERO_OR_MORE,
    "depth_normal_weight": ZERO_OR_MORE,
    "normal_consistency_weight": ZERO_OR_MORE,
    "depth_normal_start_fraction": ZERO_TO_1,
    "backproject_share": ZERO_TO_1,
    "backproject_max": WHOLE_0_OR_MORE,
    "min_vcpus": WHOLE_ABOVE_0,
    "min_memory_gb": WHOLE_ABOVE_0,
    "steps": WHOLE_ABOVE_0,
    "max_splats": WHOLE_ABOVE_0,
    "min_gpu_memory_gb": WHOLE_ABOVE_0,
}
# Wrong values for each kind: a bool, which is never a number here, a value outside the kind's
# range at each of its ends, and a value of the wrong type, text for a number and a float for a
# whole number. True and the float are inside the kind's range, so only their type is wrong,
# except True for start_opacity, which must be below 1.
WRONG_VALUES = {
    WHOLE_ABOVE_0: [True, 0, 1.0],
    WHOLE_0_OR_MORE: [True, -1, 0.0],
    WHOLE_1_TO_255: [True, 0, 256, 128.0],
    ABOVE_0: [True, 0, "1"],
    ZERO_OR_MORE: [True, -0.5, "0"],
    ABOVE_0_BELOW_1: [True, 0, 1, "0.5"],
    ABOVE_0_AT_MOST_1: [True, 0, 1.5, "1"],
    ZERO_TO_1: [True, -0.1, 1.5, "0.5"],
}


def typed(table: Mapping[str, Any]) -> dict[str, tuple[type, Any]]:
    """Each value of `table` with its type, so that 10 and 10.0, or 1 and true, don't compare
    equal."""
    return {key: (type(value), value) for key, value in table.items()}


def table_of(key: str, preset: str) -> str:
    """The table that holds `key`: [train.<preset>] for a Preset_Setting, else [train]."""
    return f"[train.{preset}]" if key in PRESET_KEYS else "[train]"


def train_settings(preset: str = "fast", **changes: Any) -> StepSettings:
    """The test config's [train] settings for `preset`, as the Step_Runner gives them to the
    step, with each key in `changes` set to its value."""
    settings = load_config().step_settings("train", preset)
    return StepSettings({**settings, **changes}, section="train", preset=preset)


def read(settings: StepSettings) -> TrainSettings:
    """read_settings on `settings`, with the section and the Preset they're for, as the step
    calls it."""
    return read_settings(settings, settings.section, settings.preset)


def refusal(settings: StepSettings) -> str:
    """The message of the ExpectedFailure that read_settings raises on `settings`."""
    with pytest.raises(ExpectedFailure) as caught:
        read(settings)
    return str(caught.value)


# The real config.toml

def test_the_real_train_section_has_exactly_the_50_shared_settings_and_the_preset_tables() -> None:
    section = tomllib.loads(REAL_CONFIG.read_text(encoding="utf-8"))["train"]
    tables = {key: value for key, value in section.items() if isinstance(value, dict)}
    shared = {key: value for key, value in section.items() if key not in tables}
    # No shared key is in a Preset table and no Preset key is in [train], so each key is in one
    # place (Req 9.4).
    assert len(shared) == 50
    assert typed(shared) == typed(REAL_SHARED)  # Req 9.2, and the depth-priors spec's Req 14.1
    assert {name: typed(table) for name, table in tables.items()} == {
        preset: typed(table) for preset, table in REAL_PRESETS.items()}  # Req 9.1, 9.3


def test_the_real_train_section_has_the_15_new_settings_at_their_values_in_their_place() -> None:
    """The depth-priors spec's Req 14.1: the 15 new keys at that criterion's values, written in
    its order right after gpu_memory_slack and right before min_vcpus."""
    section = tomllib.loads(REAL_CONFIG.read_text(encoding="utf-8"))["train"]
    assert len(NEW_SHARED) == 15
    assert typed({key: section[key] for key in NEW_SHARED}) == typed(NEW_SHARED)
    keys = list(section)
    after = keys.index("gpu_memory_slack") + 1
    assert keys[after:after + len(NEW_SHARED) + 1] == [*NEW_SHARED, "min_vcpus"]


@pytest.mark.parametrize("preset", PRESET_NAMES)
def test_the_real_train_section_loads_exactly_the_53_keys_for_each_preset(preset: str) -> None:
    settings = load_config(REAL_CONFIG).step_settings("train", preset)
    assert len(settings) == 53
    assert typed(settings) == typed({**REAL_SHARED, **REAL_PRESETS[preset]})
    assert asdict(read(settings)) == settings  # and the real values pass the step's checks


@pytest.mark.parametrize("preset", PRESET_NAMES)
def test_every_type_on_each_presets_fallback_list_meets_the_step_needs_of_the_real_step_list(
        preset: str) -> None:
    """Req 9.9, 14.9. The needs come from the Start_Command's own step_needs_over over the real
    Step_List's names, as cloud/start.py works them out, so the train step's needs count."""
    cloud_config = load_cloud_config(REAL_CONFIG)
    needs = step_needs_over([entry.name for entry in STEP_LIST], load_config(REAL_CONFIG), preset)
    fallback = list(cloud_config.preset_plan(preset).fallback)
    assert select_candidates(preset, needs, cloud_config) == fallback  # every type is kept


# The test config

@pytest.mark.parametrize("preset", PRESET_NAMES)
def test_read_settings_on_the_test_config_gives_the_presets_values_and_whole_numbers_as_ints(
        preset: str) -> None:
    settings = load_config().step_settings("train", preset)
    train = read(settings)
    assert type(train) is TrainSettings
    assert asdict(train) == settings  # each key's value in the field of its name
    assert {key: getattr(train, key) for key in PRESET_KEYS} == TEST_PRESETS[preset]
    # clean_reserve_minutes and save_margin_minutes are written as ints, and become floats.
    assert {key: type(value) for key, value in asdict(train).items()} == {
        key: int if kind in WHOLE_KINDS else float for key, kind in KINDS.items()}


# The checks of read_settings

WRONG = [pytest.param(key, value, id=f"{key} {value!r}")
         for key, kind in KINDS.items() for value in WRONG_VALUES[kind]]


@pytest.mark.parametrize(("key", "value"), WRONG)
def test_a_wrong_value_is_refused_naming_the_key_its_table_and_what_it_must_be(
        key: str, value: object) -> None:
    # quality, so the Preset table named is the chosen one, not the test config's default_preset.
    message = refusal(train_settings("quality", **{key: value}))
    wanted = KINDS[key]
    assert message.startswith(f"The setting {key} in {table_of(key, 'quality')} is ")
    assert message.endswith(f", but it must be {wanted}. Change it to {wanted}.")


# Values at the ends of their ranges, which pass (Req 9.7): both ends of mask_threshold, 1 for the
# two numbers that may be at most 1, both ends of the two fractions, and 0 for each of the five
# weights, which turns its Loss_Term off.
EDGE_VALUES = [
    ("mask_threshold", 1),
    ("mask_threshold", 255),
    ("means_lr_final", 1),
    ("gpu_memory_slack", 1),
    ("erank_start_fraction", 0),
    ("erank_start_fraction", 1),
    ("refine_stop_fraction", 0),
    ("refine_stop_fraction", 1),
    ("l1_weight", 0),
    ("ssim_weight", 0),
    ("opacity_reg", 0),
    ("scale_reg", 0),
    ("erank_reg", 0),
]
# The new keys' values at the ends of their ranges, which pass (the depth-priors spec's Req 14.2):
# 0 for each of the four Prior_Term weights, which turns its term off, for depth_gradient_falloff,
# depth_weight_floor and normal_start_step, and for backproject_max, which turns the
# Back_Projected_Splats off; and both ends of the two fractions.
NEW_EDGE_VALUES = [
    ("normal_weight", 0),
    ("flatten_reg", 0),
    ("depth_normal_weight", 0),
    ("normal_consistency_weight", 0),
    ("depth_gradient_falloff", 0),
    ("depth_weight_floor", 0),
    ("normal_start_step", 0),
    ("backproject_max", 0),
    ("depth_normal_start_fraction", 0),
    ("depth_normal_start_fraction", 1),
    ("backproject_share", 0),
    ("backproject_share", 1),
]


@pytest.mark.parametrize(("key", "value"), EDGE_VALUES + NEW_EDGE_VALUES)
def test_the_values_at_the_ends_of_their_ranges_pass(key: str, value: int) -> None:
    assert getattr(read(train_settings(**{key: value})), key) == value


def test_max_splats_below_start_scale_neighbours_plus_1_is_refused_naming_both_settings() -> None:
    message = refusal(train_settings(max_splats=3, start_scale_neighbours=3))
    assert message == ("The setting max_splats in [train.fast] is 3, but it must be at least "
                       "start_scale_neighbours in [train] + 1, which is 4. Raise max_splats or "
                       "lower start_scale_neighbours.")  # Req 9.8


def test_max_splats_of_start_scale_neighbours_plus_1_passes() -> None:
    train = read(train_settings(max_splats=4, start_scale_neighbours=3))
    assert (train.max_splats, train.start_scale_neighbours) == (4, 3)


@pytest.mark.parametrize("key", list(KINDS))
def test_a_missing_key_gives_the_step_runners_message_naming_both_tables_and_the_key(
        key: str) -> None:
    settings = train_settings("quality")
    del settings[key]
    with pytest.raises(PipelineError) as caught:
        read(settings)
    with pytest.raises(PipelineError) as runners:
        settings[key]  # the Step_Runner's own message for the missing key (Req 9.6)
    message = str(caught.value)
    assert message == str(runners.value)
    for word in ("[train]", "[train.quality]", key):
        assert word in message


# set_train

# What set_train can change: a Shared_Setting, which every Preset reads, or a Preset_Setting for
# one Preset, which only that Preset reads. Each with the change each Preset then sees.
TRAIN_CHANGES = [
    pytest.param({"seed": 7}, {preset: {"seed": 7} for preset in PRESET_NAMES},
                 id="a shared setting"),
    pytest.param({"max_splats": {"default": 4}},
                 {"fast": {}, "default": {"max_splats": 4}, "quality": {}},
                 id="max_splats for one preset"),
]


@pytest.mark.parametrize(("values", "seen"), TRAIN_CHANGES)
def test_set_train_changes_only_the_key_it_names(
        set_train: Callable[..., None], values: dict[str, Any],
        seen: dict[str, dict[str, Any]]) -> None:
    before = {preset: load_config().step_settings("train", preset) for preset in PRESET_NAMES}
    set_train(**values)
    after = {preset: load_config().step_settings("train", preset) for preset in PRESET_NAMES}
    assert after == {preset: {**before[preset], **seen[preset]} for preset in PRESET_NAMES}


def test_the_next_test_sees_the_test_configs_values_again() -> None:
    """pytest runs a file's tests in order, so this runs after the set_train tests above."""
    for preset, table in TEST_PRESETS.items():
        settings = load_config().step_settings("train", preset)
        assert (settings["seed"], settings["max_splats"]) == (42, table["max_splats"])
