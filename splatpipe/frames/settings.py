"""The Frames_Step's settings: the Shared_Settings of [frames] and the Preset's base_picks, checked.

`read_settings` reads each key with `settings[key]`, so a missing key raises the Step_Runner's own
Readable_Error, which names [frames], [frames.<preset>] and the key (Req 9.5). Each kind of value
has one check, which checks type() exactly, so a bool is never a number. A value that fails it is
refused with an ExpectedFailure naming the key, its table and what the value must be (Req 9.6).
Then three pairs are checked: mask_min_coverage must be no greater than mask_max_coverage,
min_picks no greater than base_picks, and base_picks no greater than max_picks. A refusal names
both settings and their values (Req 9.6, 9.7). NaN and infinity never get here, because the
Config refuses them.

Every key is checked whatever the video, so a bad tonemap_npl or tonemap_desat fails every Job
the same way, though only an HDR_Clip uses them. min_vcpus and min_memory_gb are checked and kept
for the cloud spec, and the step never uses them (Req 9.10).

From splatpipe, this module imports only splatpipe.errors.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, NoReturn

from splatpipe.errors import ExpectedFailure


@dataclass(frozen=True)
class FramesSettings:
    """The checked settings of [frames] and its Preset table. Each field is named after its key."""

    score_width: int
    tonemap_npl: float  # the Tone_Map's nominal peak brightness, in nits (A13)
    tonemap_desat: float  # the Tone_Map's highlight desaturation, 0 for none (A13)
    still_motion: float
    motion_window_seconds: float
    min_still_seconds: float
    blip_seconds: float
    jolt_ratio: float
    jolt_window_seconds: float
    mask_samples: int
    mask_ratio: float
    mask_min_coverage: float
    mask_max_coverage: float
    jpeg_quality: int
    long_edge: int
    seconds_per_pick: float
    max_picks: int
    min_picks: int
    # The Machine_Needs, kept for the cloud spec. The step never uses them (Req 9.10).
    min_vcpus: int
    min_memory_gb: int
    ffmpeg_timeout_seconds: float
    ffprobe_timeout_seconds: float
    base_picks: int  # the Preset_Setting, from [frames.<preset>]


@dataclass(frozen=True)
class _Kind:
    """A kind of value: what it must be, in words, its range, and whether it's a whole number."""

    wanted: str  # for the messages, like "a whole number above 0"
    in_range: Callable[[Any], bool]
    whole: bool  # True: an int, kept as it is. False: an int or a float, kept as a float.

    def accepts(self, value: object) -> bool:
        """Whether `value` is of this kind. type() is checked exactly: a bool is never a number."""
        types = (int,) if self.whole else (int, float)
        return type(value) in types and self.in_range(value)


_WHOLE_ABOVE_ZERO = _Kind("a whole number above 0", lambda v: v > 0, whole=True)
_WHOLE_TWO_OR_MORE = _Kind("a whole number of 2 or more", lambda v: v >= 2, whole=True)
_WHOLE_ONE_TO_100 = _Kind("a whole number from 1 to 100", lambda v: 1 <= v <= 100, whole=True)
_ABOVE_ZERO = _Kind("a number above 0", lambda v: v > 0, whole=False)
_ZERO_OR_MORE = _Kind("a number of 0 or more", lambda v: v >= 0, whole=False)
_ZERO_TO_ONE = _Kind("a number from 0 to 1", lambda v: 0 <= v <= 1, whole=False)

# The Shared_Settings of [frames], each with its kind (Req 9.6), in FramesSettings' order.
_SHARED: dict[str, _Kind] = {
    "score_width": _WHOLE_ABOVE_ZERO,
    "tonemap_npl": _ABOVE_ZERO,
    "tonemap_desat": _ZERO_OR_MORE,
    "still_motion": _ABOVE_ZERO,
    "motion_window_seconds": _ABOVE_ZERO,
    "min_still_seconds": _ABOVE_ZERO,
    "blip_seconds": _ABOVE_ZERO,
    "jolt_ratio": _ZERO_TO_ONE,
    "jolt_window_seconds": _ABOVE_ZERO,
    "mask_samples": _WHOLE_TWO_OR_MORE,
    "mask_ratio": _ZERO_TO_ONE,
    "mask_min_coverage": _ZERO_TO_ONE,
    "mask_max_coverage": _ZERO_TO_ONE,
    "jpeg_quality": _WHOLE_ONE_TO_100,
    "long_edge": _WHOLE_ABOVE_ZERO,
    "seconds_per_pick": _ABOVE_ZERO,
    "max_picks": _WHOLE_ABOVE_ZERO,
    "min_picks": _WHOLE_ABOVE_ZERO,
    "min_vcpus": _WHOLE_ABOVE_ZERO,
    "min_memory_gb": _WHOLE_ABOVE_ZERO,
    "ffmpeg_timeout_seconds": _ABOVE_ZERO,
    "ffprobe_timeout_seconds": _ABOVE_ZERO,
}
# The Preset_Setting, which each Preset table [frames.<preset>] holds.
_PER_PRESET: dict[str, _Kind] = {"base_picks": _WHOLE_ABOVE_ZERO}
_KINDS = _SHARED | _PER_PRESET  # every key, in the order it's checked
# The pairs (low, high) whose low must be no greater than its high, in the order they're checked.
_NO_GREATER = (("mask_min_coverage", "mask_max_coverage"), ("min_picks", "base_picks"),
               ("base_picks", "max_picks"))


def read_settings(settings: Mapping[str, Any], section: str, preset: str) -> FramesSettings:
    """Read and check every key of the Step's settings, and keep them in a FramesSettings.

    `settings` holds the Shared_Settings of [<section>] and the Preset_Setting of
    [<section>.<preset>], as the Step_Runner's StepSettings gives them. `section` and `preset`
    name those tables in the messages. Whole numbers are kept as ints, other numbers as floats.
    """
    tables = {key: _table(key, section, preset) for key in _KINDS}
    values = {key: _checked(settings, key, tables[key], kind) for key, kind in _KINDS.items()}
    for low, high in _NO_GREATER:
        if values[low] > values[high]:
            _refuse_order(settings, tables, low, high)
    return FramesSettings(**values)


def _table(key: str, section: str, preset: str) -> str:
    """The table that holds `key`: [<section>.<preset>] for the Preset_Setting, else [<section>]."""
    return f"[{section}.{preset}]" if key in _PER_PRESET else f"[{section}]"


def _checked(settings: Mapping[str, Any], key: str, table: str, kind: _Kind) -> int | float:
    """settings[key] once it's of `kind`: a whole number as its int, a number as a float."""
    value = settings[key]  # a missing key raises the Step_Runner's own message (Req 9.5)
    if not kind.accepts(value):
        raise ExpectedFailure(f"The setting {key} in {table} is {_shown(value)}, but it must be "
                              f"{kind.wanted}. Change it to {kind.wanted}.")
    return value if kind.whole else float(value)


def _refuse_order(settings: Mapping[str, Any], tables: Mapping[str, str], low: str,
                  high: str) -> NoReturn:
    """Refuse `low` above `high`, naming both settings, their tables and their values as written."""
    raise ExpectedFailure(f"The setting {low} in {tables[low]} is {_shown(settings[low])}, but "
                          f"it must be no greater than {high} in {tables[high]}, which is "
                          f"{_shown(settings[high])}. Lower {low} or raise {high}.")


def _shown(value: object) -> str:
    """A value about as TOML writes it: true or false for a bool, else Python's repr."""
    if isinstance(value, bool):
        return "true" if value else "false"
    return repr(value)
