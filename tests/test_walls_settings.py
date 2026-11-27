"""Tests for the Walls_Step's settings: [walls] in config.toml and splatpipe/walls/settings.py.

The real config is read by path, because tests/conftest.py points CONFIG_PATH at a test config.
The comment check in tests/test_config.py reads every key of the real config, [walls] included,
so no test here checks the comments. Every other test reads the test config, whose [walls] has
the real values of Req 3.2 and whose Preset tables have the real values of Req 3.3. The set_walls
fixture changes it for one test (Req 3.10).

The tests use the quality Preset, so a refusal of max_minutes or train_reserve_minutes names
[walls.quality], and a message that named the test config's default_preset would show.

Each *_min_scale and *_max_scale is checked by kind only as a number above 0. That the pair runs
from below 1 to above 1 is a cross-check of Req 3.5, whose refusal names the pair. In the same way,
only a key's own part of the settings table's kind is in its kind for the other cross-checked
keys, like min_sources, whose kind is a whole number of 2 or more, and finish_minutes, a number
of 0 or more. So an edge that would break a cross-check gets the other setting changed with it.

The Fallback_List test comes in task 10, with the Step_List.
"""

import tomllib
from collections.abc import Callable, Mapping
from dataclasses import asdict
from typing import Any

import pytest

from conftest import MISSING  # what set_walls takes to leave a key out of the test config
from splatpipe.config import REPO_ROOT, StepSettings, load_config
from splatpipe.errors import ExpectedFailure, PipelineError
from splatpipe.walls.settings import WallsSettings, read_settings

# Read by path, so these tests still read it once CONFIG_PATH points at a test config.
REAL_CONFIG = REPO_ROOT / "config.toml"
# The three Presets, written out here rather than taken from the code under test.
PRESET_NAMES = ("fast", "default", "quality")
# The Preset the tests read the test config with.
PRESET = "quality"

# What each kind of value must be, in the words of the module's messages, from the settings
# table of the requirements' Introduction.
WHOLE_ABOVE_0 = "a whole number above 0"
WHOLE_0_OR_MORE = "a whole number of 0 or more"
WHOLE_2_OR_MORE = "a whole number of 2 or more"
WHOLE_0_TO_255 = "a whole number from 0 to 255"
WHOLE_1_TO_100 = "a whole number from 1 to 100"
ODD_3_OR_MORE = "an odd whole number of 3 or more"
NUMBER = "a number"
ABOVE_0 = "a number above 0"
ZERO_OR_MORE = "a number of 0 or more"
ZERO_TO_ONE = "a number from 0 to 1"
ABOVE_0_BELOW_1 = "a number above 0 and below 1"
ABOVE_1 = "a number above 1"
ANGLE = "a number of degrees above 0 and at most 90"
SIGMAS = "a list of 2 or more numbers, the first 0, each above the one before"

# The 83 Shared_Settings of the settings table, in its order, each with its value, written with
# the TOML type the table writes it with, and its kind.
SHARED: dict[str, tuple[object, str]] = {
    "score_long_edge": (640, WHOLE_ABOVE_0),
    "high_pass_sigma": (3.0, ABOVE_0),
    "window_size": (31, ODD_3_OR_MORE),
    "window_step": (4, WHOLE_ABOVE_0),
    "flat_variance": (0.04, ZERO_OR_MORE),
    "occlusion_tolerance": (0.15, ABOVE_0),
    "mask_threshold": (128, WHOLE_0_TO_255),
    "scale_mad_gate": (3.0, ABOVE_0),
    "scale_min_points": (10, WHOLE_ABOVE_0),
    "max_sources": (32, WHOLE_ABOVE_0),
    "source_min_seconds": (1.0, ZERO_OR_MORE),
    "source_fallback_seconds": (0.5, ZERO_OR_MORE),
    "min_sources": (4, WHOLE_2_OR_MORE),
    "source_min_gap": (2, WHOLE_ABOVE_0),
    "source_min_overlap": (0.3, ZERO_TO_ONE),
    "motion_depth_change": (0.01, ABOVE_0),
    "source_min_motion": (0.4, ZERO_OR_MORE),
    "source_motion_cap": (3.0, ABOVE_0),
    "first_min_scale": (0.5, ABOVE_0),
    "first_max_scale": (2.0, ABOVE_0),
    "first_scale_step": (0.015, ABOVE_0),
    "second_min_scale": (0.92, ABOVE_0),
    "second_max_scale": (1.087, ABOVE_0),
    "second_scale_step": (0.004, ABOVE_0),
    "third_min_scale": (0.95, ABOVE_0),
    "third_max_scale": (1.053, ABOVE_0),
    "third_scale_step": (0.0025, ABOVE_0),
    "fit_starts": (5, WHOLE_ABOVE_0),
    "fit_bound": (0.2, ABOVE_0),
    "fit_start_spread": (0.05, ABOVE_0),
    "fit_max_iterations": (400, WHOLE_ABOVE_0),
    "fit_x_tolerance": (0.0001, ABOVE_0),
    "fit_f_tolerance": (0.001, ABOVE_0),
    "band_sigmas": ([0, 1, 2, 4, 8], SIGMAS),
    "band_max_coherence": (0.95, ABOVE_0_BELOW_1),
    "halves_tolerance": (0.015, ABOVE_0),
    "tile_size": (32, WHOLE_ABOVE_0),
    "tile_min_windows": (16, WHOLE_ABOVE_0),
    "z_scale": (1.2, ABOVE_1),
    "min_z": (2.5, NUMBER),
    "z_error_factor": (2.0, ABOVE_0),
    "facet_normal_degrees": (6.0, ANGLE),
    "facet_jump": (0.01, ABOVE_0),
    "facet_jump_distance": (2, WHOLE_ABOVE_0),
    "facet_min_pixels": (4000, WHOLE_ABOVE_0),
    "facet_min_scale": (0.75, ABOVE_0),
    "facet_max_scale": (1.333, ABOVE_0),
    "facet_scale_step": (0.006, ABOVE_0),
    "facet_occlusion_tolerance": (0.5, ABOVE_0),
    "facet_halves_tolerance": (0.03, ABOVE_0),
    "facet_min_change_ratio": (2.5, ZERO_OR_MORE),
    "facet_min_z": (3.0, NUMBER),
    "plane_merge_degrees": (3.0, ANGLE),
    "plane_merge_offset": (0.015, ABOVE_0),
    "plane_min_frames": (3, WHOLE_ABOVE_0),
    "plane_min_share": (0.01, ZERO_TO_ONE),
    "neighbour_frames": (12, WHOLE_ABOVE_0),
    "neighbour_window": (24, WHOLE_ABOVE_0),
    "fuse_tolerance": (0.1, ABOVE_0),
    "fuse_min_values": (3, WHOLE_ABOVE_0),
    "agree_tolerance": (0.03, ZERO_OR_MORE),
    "agree_min_frames": (2, WHOLE_ABOVE_0),
    "projection_min_coverage": (0.3, ZERO_TO_ONE),
    "register_max_degrees": (45.0, ANGLE),
    "register_start_tolerance": (0.5, ABOVE_0),
    "register_end_tolerance": (0.03, ABOVE_0),
    "register_passes": (5, WHOLE_ABOVE_0),
    "register_min_pixels": (1000, WHOLE_ABOVE_0),
    "tukey_constant": (4.685, ABOVE_0),
    "tukey_iterations": (10, WHOLE_ABOVE_0),
    "correction_tolerance": (0.03, ABOVE_0),
    "flatten_degrees": (5.0, ANGLE),
    "flatten_tolerance": (0.03, ZERO_OR_MORE),
    "despeckle_window": (5, ODD_3_OR_MORE),
    "despeckle_min_valid": (9, WHOLE_ABOVE_0),
    "despeckle_tolerance": (0.03, ZERO_OR_MORE),
    "fallback_max_sparse_error": (0.1, ABOVE_0),
    "finish_minutes": (3, ZERO_OR_MORE),
    "workers": (0, WHOLE_0_OR_MORE),
    "demo_frames": (3, WHOLE_0_OR_MORE),
    "demo_jpeg_quality": (90, WHOLE_1_TO_100),
    "min_vcpus": (16, WHOLE_ABOVE_0),
    "min_memory_gb": (32, ABOVE_0),
}
REAL_SHARED = {key: value for key, (value, _) in SHARED.items()}

# The two Preset_Settings of each Preset table, with Req 3.3's values, as whole numbers.
REAL_PRESETS: dict[str, dict[str, object]] = {
    "fast": {"max_minutes": 20, "train_reserve_minutes": 40},
    "default": {"max_minutes": 40, "train_reserve_minutes": 60},
    "quality": {"max_minutes": 60, "train_reserve_minutes": 110},
}
PER_PRESET_KINDS = {"max_minutes": ABOVE_0, "train_reserve_minutes": ZERO_OR_MORE}

# Every key, the 83 Shared_Settings and the 2 Preset_Settings, with its kind.
KINDS = {**{key: kind for key, (_, kind) in SHARED.items()}, **PER_PRESET_KINDS}

# The type each kind's value is kept as: a whole number as an int, a number as a float, and the
# band_sigmas as a tuple.
KEPT_AS = {kind: int for kind in (WHOLE_ABOVE_0, WHOLE_0_OR_MORE, WHOLE_2_OR_MORE,
                                  WHOLE_0_TO_255, WHOLE_1_TO_100, ODD_3_OR_MORE)}
KEPT_AS |= {kind: float for kind in (NUMBER, ABOVE_0, ZERO_OR_MORE, ZERO_TO_ONE,
                                     ABOVE_0_BELOW_1, ABOVE_1, ANGLE)}
KEPT_AS[SIGMAS] = tuple

# Wrong values for each kind, each with how the refusal shows it, as TOML writes it: a bool,
# which is never a number here, the values just past each end of the kind's range, and a value
# of the wrong type, a float for a whole number and text for a number.
WRONG_VALUES: dict[str, list[tuple[object, str]]] = {
    WHOLE_ABOVE_0: [(True, "true"), (0, "0"), (-3, "-3"), (1.0, "1.0"), ("1", '"1"')],
    WHOLE_0_OR_MORE: [(False, "false"), (-1, "-1"), (0.0, "0.0")],
    WHOLE_2_OR_MORE: [(True, "true"), (1, "1"), (2.0, "2.0")],
    WHOLE_0_TO_255: [(True, "true"), (-1, "-1"), (256, "256"), (128.0, "128.0")],
    WHOLE_1_TO_100: [(True, "true"), (0, "0"), (101, "101"), (90.0, "90.0")],
    ODD_3_OR_MORE: [(True, "true"), (2, "2"), (1, "1"), (4, "4"), (31.0, "31.0")],
    NUMBER: [(True, "true"), ("2.5", '"2.5"'), ([2.5], "[2.5]")],
    ABOVE_0: [(True, "true"), (0, "0"), (0.0, "0.0"), (-0.5, "-0.5"), ("1", '"1"')],
    ZERO_OR_MORE: [(False, "false"), (-0.001, "-0.001"), (-1, "-1"), ("0", '"0"')],
    ZERO_TO_ONE: [(True, "true"), (-0.001, "-0.001"), (1.001, "1.001"), ("0.5", '"0.5"')],
    ABOVE_0_BELOW_1: [(True, "true"), (0, "0"), (1, "1"), (1.0, "1.0"), ("0.5", '"0.5"')],
    ABOVE_1: [(True, "true"), (1, "1"), (1.0, "1.0"), (0.5, "0.5"), ("2", '"2"')],
    ANGLE: [(True, "true"), (0, "0"), (90.001, "90.001"), (-6.0, "-6.0"), ("6", '"6"')],
    SIGMAS: [
        ([1, 2, 4], "[1, 2, 4]"),  # doesn't start at 0
        ([0.5, 1], "[0.5, 1]"),
        ([0], "[0]"),  # one sigma
        ([], "[]"),
        ([0, 2, 2], "[0, 2, 2]"),  # doesn't rise
        ([0, 4, 2], "[0, 4, 2]"),
        ([0, True], "[0, true]"),  # a bool where a number belongs
        ([False, 1], "[false, 1]"),
        ([0, "1"], '[0, "1"]'),
        (0, "0"),  # not a list
    ],
}

# The values at the edges of each kind's range, which pass, while the values just past them are
# in WRONG_VALUES: window_size 3 passes and 2 doesn't, mask_threshold 0 and 255 pass and -1 and
# 256 don't, and demo_jpeg_quality 1 and 100 pass and 0 and 101 don't. A kind with no end, like a
# number, gets a value far from the real one.
EDGE_VALUES: dict[str, list[object]] = {
    WHOLE_ABOVE_0: [1],
    WHOLE_0_OR_MORE: [0],
    WHOLE_2_OR_MORE: [2],
    WHOLE_0_TO_255: [0, 255],
    WHOLE_1_TO_100: [1, 100],
    ODD_3_OR_MORE: [3],
    NUMBER: [-1.5, 0],
    ABOVE_0: [1e-06],
    ZERO_OR_MORE: [0, 0.0],
    ZERO_TO_ONE: [0, 1.0],
    ABOVE_0_BELOW_1: [1e-06, 0.999999],
    ABOVE_1: [1.000001],
    ANGLE: [1e-06, 90],
    SIGMAS: [[0, 1], [0, 0.5, 100]],
}
# The keys whose kind's edge can't pass a cross-check, with their lowest value that passes. Each
# *_max_scale must be above 1, and max_sources at least min_sources, which is at least 2.
EDGE_OVERRIDES: dict[str, list[object]] = {
    **{f"{name}_max_scale": [1.000001] for name in ("first", "second", "third", "facet")},
    "max_sources": [2],
}
# The other settings each key's edges are tried with, so that the edge passes the cross-checks.
COMPANIONS: dict[str, dict[str, object]] = {
    "max_sources": {"min_sources": 2},
    "source_min_seconds": {"source_fallback_seconds": 0},
    "register_start_tolerance": {"register_end_tolerance": 1e-06},
    "max_minutes": {"finish_minutes": 0},
}


def typed(table: Mapping[str, Any]) -> dict[str, tuple[type, Any]]:
    """Each value of `table` with its type, so that 3 and 3.0, or 1 and true, don't compare
    equal."""
    return {key: (type(value), value) for key, value in table.items()}


def walls_settings(preset: str = PRESET, **changes: Any) -> StepSettings:
    """The test config's [walls] settings for `preset`, as the Step_Runner gives them to the
    step, with each key in `changes` set to its value."""
    settings = load_config().step_settings("walls", preset)
    return StepSettings({**settings, **changes}, section="walls", preset=preset)


def read(settings: StepSettings) -> WallsSettings:
    """read_settings on `settings`, with the section and the Preset they're for, as the step
    calls it."""
    return read_settings(settings, settings.section, settings.preset)


def refusal(settings: StepSettings) -> str:
    """The message of the ExpectedFailure that read_settings raises on `settings`."""
    with pytest.raises(ExpectedFailure) as caught:
        read(settings)
    return str(caught.value)


def table_of(key: str) -> str:
    """The table that holds `key` in the messages, for the quality Preset."""
    return f"[walls.{PRESET}]" if key in PER_PRESET_KINDS else "[walls]"


# The real config.toml

def test_the_real_walls_section_sits_between_depth_and_train() -> None:
    names = list(tomllib.loads(REAL_CONFIG.read_text(encoding="utf-8")))
    assert names.index("walls") == names.index("depth") + 1  # Req 3.1
    assert names.index("train") == names.index("walls") + 1


def test_the_real_walls_section_has_exactly_the_83_settings_and_the_preset_tables() -> None:
    section = tomllib.loads(REAL_CONFIG.read_text(encoding="utf-8"))["walls"]
    tables = {key: value for key, value in section.items() if isinstance(value, dict)}
    shared = {key: value for key, value in section.items() if key not in tables}
    assert len(REAL_SHARED) == 83
    assert typed(shared) == typed(REAL_SHARED)  # Req 3.2
    assert list(shared) == list(REAL_SHARED)  # in the settings table's order
    assert {preset: typed(table) for preset, table in tables.items()} == {
        preset: typed(table) for preset, table in REAL_PRESETS.items()}  # Req 3.3


@pytest.mark.parametrize("preset", PRESET_NAMES)
def test_the_real_walls_section_loads_exactly_the_85_keys_for_each_preset(preset: str) -> None:
    settings = load_config(REAL_CONFIG).step_settings("walls", preset)
    assert len(settings) == 85
    assert typed(settings) == typed({**REAL_SHARED, **REAL_PRESETS[preset]})
    walls = read(settings)  # the real values pass the step's checks
    assert type(walls) is WallsSettings
    kept = asdict(walls)
    assert kept == {**settings, "band_sigmas": (0.0, 1.0, 2.0, 4.0, 8.0)}
    # A whole number is kept as an int, a number as a float, even when it's written as a whole
    # number like finish_minutes, and band_sigmas as a tuple of floats.
    assert {key: type(value) for key, value in kept.items()} == {
        key: KEPT_AS[kind] for key, kind in KINDS.items()}
    assert {type(sigma) for sigma in walls.band_sigmas} == {float}


# The kinds of read_settings (Req 3.4)

WRONG = [pytest.param(key, value, shown, id=f"{key} {value!r}")
         for key, kind in KINDS.items() for value, shown in WRONG_VALUES[kind]]


@pytest.mark.parametrize(("key", "value", "shown"), WRONG)
def test_a_wrong_value_is_refused_naming_the_key_its_table_and_what_it_must_be(
        key: str, value: object, shown: str) -> None:
    message = refusal(walls_settings(**{key: value}))
    wanted = KINDS[key]
    assert message == (f"The setting {key} in {table_of(key)} is {shown}, but it must be "
                       f"{wanted}. Change it to {wanted}.")


EDGES = [pytest.param(key, value, id=f"{key} {value!r}")
         for key, kind in KINDS.items() for value in EDGE_OVERRIDES.get(key, EDGE_VALUES[kind])]


@pytest.mark.parametrize(("key", "value"), EDGES)
def test_a_value_at_the_edge_of_its_range_passes_and_is_kept(key: str, value: object) -> None:
    walls = read(walls_settings(**{key: value, **COMPANIONS.get(key, {})}))
    kept = getattr(walls, key)
    kind = KINDS[key]
    assert type(kept) is KEPT_AS[kind]
    if kind == SIGMAS:
        assert kept == tuple(float(sigma) for sigma in value)  # type: ignore[union-attr]
    else:
        assert kept == value


# The cross-checks of read_settings (Req 3.5)

SCALE_PASSES = ("first", "second", "third", "facet")

# Each cross-check broken on its own: the changes that break it, and the two settings with their
# values as the refusal shows them. Every other setting is the test config's.
CROSS_CHECKS = [
    *[pytest.param({f"{name}_min_scale": 1.0},
                   (f"{name}_min_scale", "1.0"),
                   (f"{name}_max_scale", repr(REAL_SHARED[f"{name}_max_scale"])),
                   id=f"{name}_min_scale 1.0")
      for name in SCALE_PASSES],
    *[pytest.param({f"{name}_max_scale": 1},
                   (f"{name}_min_scale", repr(REAL_SHARED[f"{name}_min_scale"])),
                   (f"{name}_max_scale", "1"), id=f"{name}_max_scale 1")
      for name in SCALE_PASSES],
    pytest.param({"first_min_scale": 1.5, "first_max_scale": 0.5},
                 ("first_min_scale", "1.5"), ("first_max_scale", "0.5"), id="first both"),
    pytest.param({"source_fallback_seconds": 1.5},
                 ("source_fallback_seconds", "1.5"), ("source_min_seconds", "1.0"),
                 id="source_fallback_seconds 1.5"),
    pytest.param({"min_sources": 33},
                 ("min_sources", "33"), ("max_sources", "32"), id="min_sources 33"),
    pytest.param({"max_sources": 1},
                 ("min_sources", "4"), ("max_sources", "1"), id="max_sources 1"),
    pytest.param({"register_end_tolerance": 0.6},
                 ("register_end_tolerance", "0.6"), ("register_start_tolerance", "0.5"),
                 id="register_end_tolerance 0.6"),
    pytest.param({"despeckle_min_valid": 26},
                 ("despeckle_min_valid", "26"), ("despeckle_window", "5"),
                 id="despeckle_min_valid 26"),
    pytest.param({"despeckle_window": 3, "despeckle_min_valid": 10},
                 ("despeckle_min_valid", "10"), ("despeckle_window", "3"),
                 id="despeckle_window 3"),
    pytest.param({"finish_minutes": 60},
                 ("finish_minutes", "60"), ("max_minutes", "60"), id="finish_minutes 60"),
    pytest.param({"max_minutes": 2.5},
                 ("finish_minutes", "3"), ("max_minutes", "2.5"), id="max_minutes 2.5"),
]


@pytest.mark.parametrize(("changes", "first", "second"), CROSS_CHECKS)
def test_each_cross_check_is_refused_on_its_own_naming_both_settings_and_their_values(
        changes: dict[str, object], first: tuple[str, str], second: tuple[str, str]) -> None:
    message = refusal(walls_settings(**changes))
    (first_key, first_shown), (second_key, second_shown) = first, second
    assert message.startswith(
        f"The setting {first_key} in {table_of(first_key)} is {first_shown} and {second_key} in "
        f"{table_of(second_key)} is {second_shown}, but ")
    assert "but it must be" not in message  # not a kind's refusal


@pytest.mark.parametrize("changes", [
    pytest.param({f"{name}_min_scale": 0.999999, f"{name}_max_scale": 1.000001}, id=name)
    for name in SCALE_PASSES
] + [
    pytest.param({"source_fallback_seconds": 1.0}, id="source_fallback_seconds"),
    pytest.param({"min_sources": 32}, id="min_sources"),
    pytest.param({"register_end_tolerance": 0.5}, id="register_end_tolerance"),
    pytest.param({"despeckle_min_valid": 25}, id="despeckle_min_valid 25"),
    pytest.param({"despeckle_window": 3, "despeckle_min_valid": 9}, id="despeckle_window 3"),
    pytest.param({"finish_minutes": 59.5}, id="finish_minutes"),
])
def test_each_cross_check_passes_at_its_edge(changes: dict[str, object]) -> None:
    walls = read(walls_settings(**changes))
    assert {key: getattr(walls, key) for key in changes} == changes


# A missing key (Req 3.6)

@pytest.mark.parametrize("key", list(KINDS))
def test_a_missing_key_gives_the_step_runners_message_naming_both_tables_and_the_key(
        key: str, set_walls: Callable[..., None]) -> None:
    set_walls(**{key: MISSING})
    settings = load_config().step_settings("walls", PRESET)
    assert key not in settings
    with pytest.raises(PipelineError) as caught:
        read(settings)
    with pytest.raises(PipelineError) as runners:
        settings[key]  # the Step_Runner's own message for the missing key
    message = str(caught.value)
    assert type(caught.value) is PipelineError  # not an ExpectedFailure of read_settings
    assert message == str(runners.value)
    for word in ("[walls]", f"[walls.{PRESET}]", key):
        assert word in message


# The test config and set_walls (Req 3.10)

def test_set_walls_changes_only_the_key_it_names_for_every_preset(
        set_walls: Callable[..., None]) -> None:
    before = {preset: load_config().step_settings("walls", preset) for preset in PRESET_NAMES}
    set_walls(tile_size=8)
    after = {preset: load_config().step_settings("walls", preset) for preset in PRESET_NAMES}
    assert after == {preset: {**before[preset], "tile_size": 8} for preset in PRESET_NAMES}
    assert read(walls_settings()).tile_size == 8


def test_set_walls_sets_a_preset_setting_in_every_preset_table(
        set_walls: Callable[..., None]) -> None:
    set_walls(max_minutes=5)
    for preset in PRESET_NAMES:
        settings = load_config().step_settings("walls", preset)
        assert typed(settings) == typed({**REAL_SHARED, **REAL_PRESETS[preset],
                                         "max_minutes": 5})


def test_the_next_test_sees_the_test_configs_values_again() -> None:
    """pytest runs a file's tests in order, so this runs after the set_walls tests above. The
    test config's [walls] and Preset tables have the real values (Req 3.10)."""
    for preset in PRESET_NAMES:
        settings = load_config().step_settings("walls", preset)
        assert typed(settings) == typed({**REAL_SHARED, **REAL_PRESETS[preset]})
