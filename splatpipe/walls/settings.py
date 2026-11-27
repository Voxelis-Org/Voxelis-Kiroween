"""The Walls_Step's settings: the 83 Shared_Settings of [walls] and the Preset's two, checked.

`read_settings` reads each key with `settings[key]`, so a missing key raises the Step_Runner's own
Readable_Error, which names [walls], [walls.<preset>] and the key (Req 3.6). Each kind of value
has one check, which checks type() exactly, so a bool is never a number: a whole number is a TOML
integer, and a number is a TOML integer or float. A value that fails it is refused with an
ExpectedFailure naming the key, its table and what the value must be, with the value shown as
TOML writes it (Req 3.4). NaN, infinity and TOML dates and times never get here, because the
Config refuses them.

Once every kind passes, the cross-checks of Req 3.5 run in the requirement's order, and each
refusal names both settings, their tables and their values. A rule that ties one setting to
another, or a scale to 1, is a cross-check, not part of the kind: each *_min_scale and
*_max_scale is checked as a number above 0, and then the pair must run from below 1 to above 1,
so a refusal can name the pair.

min_vcpus and min_memory_gb are checked and kept as the cloud-only Step_Needs, and the step never
uses them (Req 3.8).

The kinds are this module's own, so it imports nothing private from another step. From
splatpipe, this module imports only splatpipe.errors.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, NoReturn

from splatpipe.errors import ExpectedFailure


@dataclass(frozen=True)
class WallsSettings:
    """The checked settings of [walls] and its Preset table. Each field is named after its key.

    Whole numbers are ints, other numbers floats, and band_sigmas a tuple of floats.
    """

    # The score images and the windows.
    score_long_edge: int
    high_pass_sigma: float
    window_size: int
    window_step: int
    flat_variance: float
    occlusion_tolerance: float
    mask_threshold: int
    # The Scene_Scale.
    scale_mad_gate: float
    scale_min_points: int
    # The Source_Frames.
    max_sources: int
    source_min_seconds: float
    source_fallback_seconds: float
    min_sources: int
    source_min_gap: int
    source_min_overlap: float
    motion_depth_change: float
    source_min_motion: float
    source_motion_cap: float
    # The three passes' Table_Scales.
    first_min_scale: float
    first_max_scale: float
    first_scale_step: float
    second_min_scale: float
    second_max_scale: float
    second_scale_step: float
    third_min_scale: float
    third_max_scale: float
    third_scale_step: float
    # The Change_Fit.
    fit_starts: int
    fit_bound: float
    fit_start_spread: float
    fit_max_iterations: int
    fit_x_tolerance: float
    fit_f_tolerance: float
    # The Bands.
    band_sigmas: tuple[float, ...]
    band_max_coherence: float
    # The Self_Check.
    halves_tolerance: float
    tile_size: int
    tile_min_windows: int
    z_scale: float
    min_z: float
    z_error_factor: float
    # The Facets.
    facet_normal_degrees: float
    facet_jump: float
    facet_jump_distance: int
    facet_min_pixels: int
    facet_min_scale: float
    facet_max_scale: float
    facet_scale_step: float
    facet_occlusion_tolerance: float
    facet_halves_tolerance: float
    facet_min_change_ratio: float
    facet_min_z: float
    # The World_Planes.
    plane_merge_degrees: float
    plane_merge_offset: float
    plane_min_frames: int
    plane_min_share: float
    # The Neighbour_Frames, the fuse and the fallback.
    neighbour_frames: int
    neighbour_window: int
    fuse_tolerance: float
    fuse_min_values: int
    agree_tolerance: float
    agree_min_frames: int
    projection_min_coverage: float
    register_max_degrees: float
    register_start_tolerance: float
    register_end_tolerance: float
    register_passes: int
    register_min_pixels: int
    tukey_constant: float
    tukey_iterations: int
    # The correction, the flatten, the despeckle and the per-frame choice.
    correction_tolerance: float
    flatten_degrees: float
    flatten_tolerance: float
    despeckle_window: int
    despeckle_min_valid: int
    despeckle_tolerance: float
    fallback_max_sparse_error: float
    # The time budget, the workers and the demos.
    finish_minutes: float
    workers: int
    demo_frames: int
    demo_jpeg_quality: int
    # The cloud-only Step_Needs, checked and kept. The step never uses them (Req 3.8).
    min_vcpus: int
    min_memory_gb: float
    # The Preset_Settings, from [walls.<preset>].
    max_minutes: float
    train_reserve_minutes: float


@dataclass(frozen=True)
class _Kind:
    """A kind of value: what it must be, in words, the exact types it may have, and its range."""

    wanted: str  # for the messages, like "a whole number above 0"
    types: tuple[type, ...]  # compared with type(), so a bool is never an int
    in_range: Callable[[Any], bool]
    keep: Callable[[Any], Any]  # what the checked value is kept as

    def accepts(self, value: object) -> bool:
        """Whether `value` is of this kind. type() is checked exactly: a bool is never a number."""
        return type(value) in self.types and self.in_range(value)


def _whole(wanted: str, in_range: Callable[[int], bool]) -> _Kind:
    """A kind of whole number: a TOML integer, kept as its int."""
    return _Kind(wanted, (int,), in_range, int)


def _number(wanted: str, in_range: Callable[[float], bool]) -> _Kind:
    """A kind of number: a TOML integer or float, kept as a float."""
    return _Kind(wanted, (int, float), in_range, float)


def _rising_from_zero(value: list[Any]) -> bool:
    """Whether `value` holds 2 or more numbers, the first 0, each above the one before."""
    if len(value) < 2 or any(type(item) not in (int, float) for item in value):
        return False
    return value[0] == 0 and all(later > earlier for earlier, later in zip(value, value[1:]))


_WHOLE_ABOVE_ZERO = _whole("a whole number above 0", lambda v: v > 0)
_WHOLE_ZERO_OR_MORE = _whole("a whole number of 0 or more", lambda v: v >= 0)
_WHOLE_AT_LEAST_TWO = _whole("a whole number of 2 or more", lambda v: v >= 2)
_WHOLE_0_TO_255 = _whole("a whole number from 0 to 255", lambda v: 0 <= v <= 255)
_WHOLE_1_TO_100 = _whole("a whole number from 1 to 100", lambda v: 1 <= v <= 100)
_ODD_AT_LEAST_THREE = _whole("an odd whole number of 3 or more", lambda v: v >= 3 and v % 2 == 1)
_NUMBER = _number("a number", lambda v: True)
_NUMBER_ABOVE_ZERO = _number("a number above 0", lambda v: v > 0)
_ZERO_OR_MORE = _number("a number of 0 or more", lambda v: v >= 0)
_ZERO_TO_ONE = _number("a number from 0 to 1", lambda v: 0 <= v <= 1)
_ABOVE_ZERO_BELOW_ONE = _number("a number above 0 and below 1", lambda v: 0 < v < 1)
_ABOVE_ONE = _number("a number above 1", lambda v: v > 1)
_ANGLE = _number("a number of degrees above 0 and at most 90", lambda v: 0 < v <= 90)
_LIST_OF_NUMBERS = _Kind("a list of 2 or more numbers, the first 0, each above the one before",
                         (list,), _rising_from_zero, lambda v: tuple(float(item) for item in v))

# The Shared_Settings of [walls], each with its kind (Req 3.4), in WallsSettings' order, which is
# the order they're checked in. The rules that tie a setting to another, or a scale to 1, are the
# cross-checks of Req 3.5, made once every kind passes.
_SHARED: dict[str, _Kind] = {
    "score_long_edge": _WHOLE_ABOVE_ZERO,
    "high_pass_sigma": _NUMBER_ABOVE_ZERO,
    "window_size": _ODD_AT_LEAST_THREE,
    "window_step": _WHOLE_ABOVE_ZERO,
    "flat_variance": _ZERO_OR_MORE,
    "occlusion_tolerance": _NUMBER_ABOVE_ZERO,
    "mask_threshold": _WHOLE_0_TO_255,
    "scale_mad_gate": _NUMBER_ABOVE_ZERO,
    "scale_min_points": _WHOLE_ABOVE_ZERO,
    "max_sources": _WHOLE_ABOVE_ZERO,
    "source_min_seconds": _ZERO_OR_MORE,
    "source_fallback_seconds": _ZERO_OR_MORE,  # at most source_min_seconds, a cross-check
    "min_sources": _WHOLE_AT_LEAST_TWO,  # at most max_sources, a cross-check
    "source_min_gap": _WHOLE_ABOVE_ZERO,
    "source_min_overlap": _ZERO_TO_ONE,
    "motion_depth_change": _NUMBER_ABOVE_ZERO,
    "source_min_motion": _ZERO_OR_MORE,
    "source_motion_cap": _NUMBER_ABOVE_ZERO,
    # Each pass's scales run from below 1 to above 1, a cross-check.
    "first_min_scale": _NUMBER_ABOVE_ZERO,
    "first_max_scale": _NUMBER_ABOVE_ZERO,
    "first_scale_step": _NUMBER_ABOVE_ZERO,
    "second_min_scale": _NUMBER_ABOVE_ZERO,
    "second_max_scale": _NUMBER_ABOVE_ZERO,
    "second_scale_step": _NUMBER_ABOVE_ZERO,
    "third_min_scale": _NUMBER_ABOVE_ZERO,
    "third_max_scale": _NUMBER_ABOVE_ZERO,
    "third_scale_step": _NUMBER_ABOVE_ZERO,
    "fit_starts": _WHOLE_ABOVE_ZERO,
    "fit_bound": _NUMBER_ABOVE_ZERO,
    "fit_start_spread": _NUMBER_ABOVE_ZERO,
    "fit_max_iterations": _WHOLE_ABOVE_ZERO,
    "fit_x_tolerance": _NUMBER_ABOVE_ZERO,
    "fit_f_tolerance": _NUMBER_ABOVE_ZERO,
    "band_sigmas": _LIST_OF_NUMBERS,
    "band_max_coherence": _ABOVE_ZERO_BELOW_ONE,
    "halves_tolerance": _NUMBER_ABOVE_ZERO,
    "tile_size": _WHOLE_ABOVE_ZERO,
    "tile_min_windows": _WHOLE_ABOVE_ZERO,
    "z_scale": _ABOVE_ONE,
    "min_z": _NUMBER,
    "z_error_factor": _NUMBER_ABOVE_ZERO,
    "facet_normal_degrees": _ANGLE,
    "facet_jump": _NUMBER_ABOVE_ZERO,
    "facet_jump_distance": _WHOLE_ABOVE_ZERO,
    "facet_min_pixels": _WHOLE_ABOVE_ZERO,
    "facet_min_scale": _NUMBER_ABOVE_ZERO,
    "facet_max_scale": _NUMBER_ABOVE_ZERO,
    "facet_scale_step": _NUMBER_ABOVE_ZERO,
    "facet_occlusion_tolerance": _NUMBER_ABOVE_ZERO,
    "facet_halves_tolerance": _NUMBER_ABOVE_ZERO,
    "facet_min_change_ratio": _ZERO_OR_MORE,
    "facet_min_z": _NUMBER,
    "plane_merge_degrees": _ANGLE,
    "plane_merge_offset": _NUMBER_ABOVE_ZERO,
    "plane_min_frames": _WHOLE_ABOVE_ZERO,
    "plane_min_share": _ZERO_TO_ONE,
    "neighbour_frames": _WHOLE_ABOVE_ZERO,
    "neighbour_window": _WHOLE_ABOVE_ZERO,
    "fuse_tolerance": _NUMBER_ABOVE_ZERO,
    "fuse_min_values": _WHOLE_ABOVE_ZERO,
    "agree_tolerance": _ZERO_OR_MORE,
    "agree_min_frames": _WHOLE_ABOVE_ZERO,
    "projection_min_coverage": _ZERO_TO_ONE,
    "register_max_degrees": _ANGLE,
    "register_start_tolerance": _NUMBER_ABOVE_ZERO,
    "register_end_tolerance": _NUMBER_ABOVE_ZERO,  # at most register_start_tolerance, a cross-check
    "register_passes": _WHOLE_ABOVE_ZERO,
    "register_min_pixels": _WHOLE_ABOVE_ZERO,
    "tukey_constant": _NUMBER_ABOVE_ZERO,
    "tukey_iterations": _WHOLE_ABOVE_ZERO,
    "correction_tolerance": _NUMBER_ABOVE_ZERO,
    "flatten_degrees": _ANGLE,
    "flatten_tolerance": _ZERO_OR_MORE,
    "despeckle_window": _ODD_AT_LEAST_THREE,
    "despeckle_min_valid": _WHOLE_ABOVE_ZERO,  # at most despeckle_window², a cross-check
    "despeckle_tolerance": _ZERO_OR_MORE,
    "fallback_max_sparse_error": _NUMBER_ABOVE_ZERO,
    "finish_minutes": _ZERO_OR_MORE,  # below the Preset's max_minutes, a cross-check
    "workers": _WHOLE_ZERO_OR_MORE,
    "demo_frames": _WHOLE_ZERO_OR_MORE,
    "demo_jpeg_quality": _WHOLE_1_TO_100,
    "min_vcpus": _WHOLE_ABOVE_ZERO,
    "min_memory_gb": _NUMBER_ABOVE_ZERO,
}
# The Preset_Settings, which each Preset table [walls.<preset>] holds.
_PER_PRESET: dict[str, _Kind] = {
    "max_minutes": _NUMBER_ABOVE_ZERO,
    "train_reserve_minutes": _ZERO_OR_MORE,
}
_KINDS = _SHARED | _PER_PRESET  # every key, in the order it's checked

# The passes whose Table_Scales must run from below 1 to above 1, in Req 3.5's order.
_SCALE_PAIRS = ("first", "second", "third", "facet")

# A TOML bare key: ASCII letters, digits, _ and -. Any other key is written in quotes.
_BARE_KEY = re.compile(r"[A-Za-z0-9_-]+")


def read_settings(settings: Mapping[str, Any], section: str, preset: str) -> WallsSettings:
    """Read and check every key of the Step's settings, and keep them in a WallsSettings.

    `settings` holds the Shared_Settings of [<section>] and the Preset_Settings of
    [<section>.<preset>], as the Step_Runner's StepSettings gives them. `section` and `preset`
    name those tables in the messages. Every kind is checked first, then the cross-checks.
    """
    tables = {key: _table(key, section, preset) for key in _KINDS}
    values = {key: _checked(settings, key, tables[key], kind) for key, kind in _KINDS.items()}
    _cross_check(settings, values, tables)
    return WallsSettings(**values)


def _table(key: str, section: str, preset: str) -> str:
    """The table that holds `key`: [<section>.<preset>] for a Preset_Setting, else [<section>]."""
    return f"[{section}.{preset}]" if key in _PER_PRESET else f"[{section}]"


def _checked(settings: Mapping[str, Any], key: str, table: str, kind: _Kind) -> Any:
    """settings[key] once it's of `kind`, kept as the kind keeps it."""
    value = settings[key]  # a missing key raises the Step_Runner's own message (Req 3.6)
    if not kind.accepts(value):
        raise ExpectedFailure(f"The setting {key} in {table} is {_shown(value)}, but it must be "
                              f"{kind.wanted}. Change it to {kind.wanted}.")
    return kind.keep(value)


def _cross_check(raw: Mapping[str, Any], values: Mapping[str, Any],
                 tables: Mapping[str, str]) -> None:
    """Make the cross-checks of Req 3.5, in its order, on values whose kinds all passed.

    `raw` gives each value as the Config holds it, for the messages.
    """
    for name in _SCALE_PAIRS:
        _check_scale_pair(name, raw, values, tables)
    _check_at_most("source_fallback_seconds", "source_min_seconds", raw, values, tables)
    _check_at_most("min_sources", "max_sources", raw, values, tables)
    _check_at_most("register_end_tolerance", "register_start_tolerance", raw, values, tables)
    _check_despeckle(raw, values, tables)
    if values["finish_minutes"] >= values["max_minutes"]:
        _refuse("finish_minutes", "max_minutes", raw, tables,
                "finish_minutes must be below max_minutes, so frames can be solved before the "
                "Solve_Cutoff", "Lower finish_minutes or raise max_minutes.")


def _check_scale_pair(name: str, raw: Mapping[str, Any], values: Mapping[str, Any],
                      tables: Mapping[str, str]) -> None:
    """Refuse <name>_min_scale not below 1, or <name>_max_scale not above 1, naming both."""
    low, high = f"{name}_min_scale", f"{name}_max_scale"
    fixes = []
    if values[low] >= 1:
        fixes.append(f"lower {low} below 1")
    if values[high] <= 1:
        fixes.append(f"raise {high} above 1")
    if fixes:
        fix = " and ".join(fixes)
        _refuse(low, high, raw, tables,
                f"{low} must be below 1 and {high} above 1, so the scales tried run from below "
                "the current depth to above it", fix[0].upper() + fix[1:] + ".")


def _check_at_most(key: str, limit: str, raw: Mapping[str, Any], values: Mapping[str, Any],
                   tables: Mapping[str, str]) -> None:
    """Refuse `key` above `limit`, naming both."""
    if values[key] > values[limit]:
        _refuse(key, limit, raw, tables, f"{key} must be at most {limit}",
                f"Lower {key} or raise {limit}.")


def _check_despeckle(raw: Mapping[str, Any], values: Mapping[str, Any],
                     tables: Mapping[str, str]) -> None:
    """Refuse despeckle_min_valid above despeckle_window², the pixels a window holds."""
    window_pixels = values["despeckle_window"] ** 2
    if values["despeckle_min_valid"] > window_pixels:
        _refuse("despeckle_min_valid", "despeckle_window", raw, tables,
                "despeckle_min_valid must be at most despeckle_window squared, "
                f"{window_pixels}, the pixels one window holds",
                "Lower despeckle_min_valid or raise despeckle_window.")


def _refuse(first: str, second: str, raw: Mapping[str, Any], tables: Mapping[str, str],
            rule: str, fix: str) -> NoReturn:
    """Refuse a cross-check, naming both settings, their tables and their values as written."""
    raise ExpectedFailure(f"The setting {first} in {tables[first]} is {_shown(raw[first])} and "
                          f"{second} in {tables[second]} is {_shown(raw[second])}, but {rule}. "
                          f"{fix}")


def _shown(value: object) -> str:
    """`value` as TOML writes it: true or false, a string in double quotes, an array in
    brackets, an inline table in braces, and a number as Python writes it, which TOML reads as
    the same number."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return _quoted(value)
    if isinstance(value, list):
        return "[" + ", ".join(_shown(item) for item in value) + "]"
    if isinstance(value, dict):
        pairs = ", ".join(f"{_key(key)} = {_shown(item)}" for key, item in value.items())
        return f"{{ {pairs} }}" if pairs else "{}"
    return repr(value)


def _quoted(text: str) -> str:
    """`text` as a TOML basic string. JSON's escapes are TOML's, and TOML escapes DEL too."""
    return json.dumps(text, ensure_ascii=False).replace("\x7f", "\\u007f")


def _key(key: str) -> str:
    """`key` as TOML writes it in an inline table: bare when it can be, else in quotes."""
    return key if _BARE_KEY.fullmatch(key) else _quoted(key)
