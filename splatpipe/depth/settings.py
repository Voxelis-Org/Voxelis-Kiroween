"""The Depth_Step's settings: the eight Shared_Settings of [depth], checked.

`read_settings` reads each key with `settings[key]`, so a missing key raises the Step_Runner's own
Readable_Error, which names [depth], [depth.<preset>] and the key (Req 8.6). Each kind of value
has one check, which checks type() exactly, so a bool is never a number and a number is never
true or false. A value that fails it is refused with an ExpectedFailure naming the key, its table
and what the value must be, with the value shown as TOML writes it (Req 8.3). NaN, infinity and
TOML dates and times never get here, because the Config refuses them.

Every key is a Shared_Setting, so the three Preset tables are empty and every key's table is
[depth]. min_gpu_memory_gb is checked and kept as the cloud-only Step_Need, and the step never
uses it: it checks no amount of GPU memory (Req 3.4, 8.5).

The kinds are this module's own, so it imports nothing private from the Train_Step. From
splatpipe, this module imports only splatpipe.errors.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from splatpipe.errors import ExpectedFailure


@dataclass(frozen=True)
class DepthSettings:
    """The checked settings of [depth]. Each field is named after its key."""

    long_edge: int  # the Prior_Size's longest edge, in pixels
    moge_weights: str  # the Weights_File's path on the Machine_Image
    resolution_level: int  # MoGe-2's resolution_level, from 0 to 9
    use_fp16: bool  # MoGe-2's use_fp16
    fit_min_points: int
    fit_miss_ratio: float
    fit_max_median_error: float
    min_gpu_memory_gb: int  # the cloud-only Step_Need, checked and kept, never used (Req 3.4, 8.5)


@dataclass(frozen=True)
class _Kind:
    """A kind of value: what it must be, in words, the exact types it may have, and its range."""

    wanted: str  # for the messages, like "a whole number above 0"
    types: tuple[type, ...]  # compared with type(), so a bool is never an int
    in_range: Callable[[Any], bool]
    as_float: bool = False  # True: an int or a float, kept as a float. False: kept as it is.

    def accepts(self, value: object) -> bool:
        """Whether `value` is of this kind. type() is checked exactly: a bool is never a number."""
        return type(value) in self.types and self.in_range(value)


_WHOLE_ABOVE_ZERO = _Kind("a whole number above 0", (int,), lambda v: v > 0)
_WHOLE_ZERO_TO_NINE = _Kind("a whole number from 0 to 9", (int,), lambda v: 0 <= v <= 9)
_TRUE_OR_FALSE = _Kind("true or false", (bool,), lambda v: True)
_TEXT = _Kind("a string of one or more characters", (str,), lambda v: len(v) > 0)
_ABOVE_ZERO = _Kind("a number above 0", (int, float), lambda v: v > 0, as_float=True)

# The Shared_Settings of [depth], each with its kind (Req 8.3), in DepthSettings' order, which is
# the order they're checked in.
_KINDS: dict[str, _Kind] = {
    "long_edge": _WHOLE_ABOVE_ZERO,
    "moge_weights": _TEXT,
    "resolution_level": _WHOLE_ZERO_TO_NINE,
    "use_fp16": _TRUE_OR_FALSE,
    "fit_min_points": _WHOLE_ABOVE_ZERO,
    "fit_miss_ratio": _ABOVE_ZERO,
    "fit_max_median_error": _ABOVE_ZERO,
    "min_gpu_memory_gb": _WHOLE_ABOVE_ZERO,
}
# A TOML bare key: ASCII letters, digits, _ and -. Any other key is written in quotes.
_BARE_KEY = re.compile(r"[A-Za-z0-9_-]+")


def read_settings(settings: Mapping[str, Any], section: str, preset: str) -> DepthSettings:
    """Read and check every key of the Step's settings, and keep them in a DepthSettings.

    `settings` holds the Shared_Settings of [<section>], as the Step_Runner's StepSettings gives
    them, and `section` names that table in the messages. `preset` names the Preset, and no key
    depends on it, because every [depth] setting is shared. Whole numbers are kept as ints,
    other numbers as floats.
    """
    table = f"[{section}]"  # every key is a Shared_Setting, whatever the Preset
    values = {key: _checked(settings, key, table, kind) for key, kind in _KINDS.items()}
    return DepthSettings(**values)


def _checked(settings: Mapping[str, Any], key: str, table: str, kind: _Kind) -> Any:
    """settings[key] once it's of `kind`: a number as a float, any other value as it is."""
    value = settings[key]  # a missing key raises the Step_Runner's own message (Req 8.6)
    if not kind.accepts(value):
        raise ExpectedFailure(f"The setting {key} in {table} is {_shown(value)}, but it must be "
                              f"{kind.wanted}. Change it to {kind.wanted}.")
    return float(value) if kind.as_float else value


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
