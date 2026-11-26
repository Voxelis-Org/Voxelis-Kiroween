"""The clean step's settings: the nineteen keys of [clean] and its Preset tables, checked.

`check_settings` turns the StepSettings that the step's `work` gets into a CleanSettings. It
reads every key first, with settings[key], so a missing key is the Step_Runner's own refusal
(Req 6.6). Then it checks the settings in this order and raises ExpectedFailure at the first
problem:

1. The whole numbers (Req 6.7). Only an int counts, so true and 4.0 are refused.
2. The numbers (Req 6.8): an int or a float, never a bool, given back as a float. An int too big
   for a float is refused as out of range.
3. The three file names, each with `check_plain_name` (Req 6.9).
4. The clashes: no two of the three names may be the same once case is ignored (Req 6.10).

The step checks its settings before it opens, writes or removes anything, so a setting can't
reach a path outside out/ or overwrite the trained splat.

From splatpipe, this module imports only splatpipe.step and splatpipe.errors.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields
from pathlib import PurePosixPath, PureWindowsPath
from typing import Any, NoReturn

from splatpipe.errors import ExpectedFailure
# StepSettings comes through splatpipe.step, so this module needs no other Skeleton module.
from splatpipe.step import StepSettings


@dataclass(frozen=True)
class CleanSettings:
    """The clean step's checked settings. Each field is named after its key in the config."""

    # The seventeen Shared_Settings of Req 6.2, from [clean]
    train_ply: str  # Plain_File_Name, read from out/
    clean_ply: str  # Plain_File_Name ending in .ply
    clean_sog: str  # Plain_File_Name ending in .sog
    min_seen_frames: int  # above 0, for the Unseen_Filter
    min_opacity: float  # 0 to 1
    faint_cube_divisions: int  # above 0
    faint_min_per_cube: int  # above 0
    faint_min_paint_share: float  # 0 up to but not including 1
    extent_trim_percent: float  # 0 up to but not including 50
    neighbours: int  # above 0
    isolation_ratio: float  # above 0
    max_removed_fraction: float  # 0 to 1
    min_kept_splats: int  # above 0
    warn_kept_fraction: float  # 0 to 1
    max_source_splats: int  # above 0
    max_workers: int  # 0 to 8, 0 runs splat-transform's encoding inline
    splat_transform_timeout_seconds: float  # above 0
    # The two Preset_Settings of Req 6.3, from [clean.<preset>]
    sog_sh_bands: int  # 0 to 3
    sh_iterations: int  # above 0


@dataclass(frozen=True)
class _Range:
    """A setting's range, as Req 6.7 or 6.8 gives it: a test of a value, and the words for it."""

    holds: Callable[[float], bool]
    words: str


def _above(low: int) -> _Range:
    return _Range(lambda value: value > low, f"above {low}")


def _from_to(low: int, high: int) -> _Range:
    return _Range(lambda value: low <= value <= high, f"from {low} to {high}")


def _up_to(low: int, high: int) -> _Range:
    return _Range(lambda value: low <= value < high, f"from {low} up to but not including {high}")


# Each whole-number setting of Req 6.7 and each number setting of Req 6.8 with its range, in the
# order they're checked. The ranges are the requirements' own, not tunables, and no setting's
# range depends on another's.
_WHOLE_NUMBERS = {
    "min_seen_frames": _above(0),
    "min_kept_splats": _above(0),
    "max_source_splats": _above(0),
    "neighbours": _above(0),
    "faint_cube_divisions": _above(0),
    "faint_min_per_cube": _above(0),
    "sh_iterations": _above(0),
    "sog_sh_bands": _from_to(0, 3),
    "max_workers": _from_to(0, 8),  # 0 runs splat-transform's encoding inline
}
_NUMBERS = {
    "min_opacity": _from_to(0, 1),
    "warn_kept_fraction": _from_to(0, 1),
    "max_removed_fraction": _from_to(0, 1),
    "faint_min_paint_share": _up_to(0, 1),
    "extent_trim_percent": _up_to(0, 50),
    "isolation_ratio": _above(0),
    "splat_transform_timeout_seconds": _above(0),
}
# The three file-name settings of Req 6.9, each with the ending it must have. The step reads
# train_ply by its content, so train_ply has none.
_NAMES: dict[str, str | None] = {"train_ply": None, "clean_ply": ".ply", "clean_sog": ".sog"}
# The pairs of names that must differ once case is ignored (Req 6.10), each with what the step
# would overwrite if they didn't.
_CLASHES = (("train_ply", "clean_ply", "the trained splat"),
            ("train_ply", "clean_sog", "the trained splat"),
            ("clean_ply", "clean_sog", "its own clean splat"))
# The Preset_Settings of Req 6.3, which are in [clean.<preset>]. The rest are in [clean].
_PRESET_SETTINGS = frozenset({"sog_sh_bands", "sh_iterations"})
# The path separators a Plain_File_Name can't hold: / and \, and the running system's own.
_SEPARATORS = frozenset(char for char in ("/", "\\", os.sep, os.altsep) if char)


def check_settings(settings: StepSettings) -> CleanSettings:
    """The clean step's settings for one Preset, each checked, as a CleanSettings.

    Every key is read first, so a missing one is the Step_Runner's own PipelineError, naming
    [clean], [clean.<preset>] and the key (Req 6.6). Then the whole numbers, the numbers, the
    three names and the clashes are checked, in that order, and the first problem raises
    ExpectedFailure (Req 6.7 to 6.10, 8.8).
    """
    values = {field.name: settings[field.name] for field in fields(CleanSettings)}
    checked: dict[str, Any] = {}
    for key, wanted in _WHOLE_NUMBERS.items():
        checked[key] = _whole_number(_table(settings, key), key, values[key], wanted)
    for key, wanted in _NUMBERS.items():
        checked[key] = _number(_table(settings, key), key, values[key], wanted)
    for key, suffix in _NAMES.items():
        checked[key] = check_plain_name(key, values[key], suffix)
    _refuse_clashes(settings.section, checked)
    return CleanSettings(**checked)


def check_plain_name(key: str, value: object, suffix: str | None) -> str:
    """`value`, the setting `key`, when it's a Plain_File_Name that ends in `suffix`, if given.

    A Plain_File_Name names a file inside out/ and nothing else. The refusals come in this
    order: not a str, empty, a path separator, . or .., a :, a control character, an absolute
    path or a drive, and last an ending that isn't exactly `suffix`, in lower case (Req 6.9).
    Each is an ExpectedFailure naming the setting and what's wrong with the value. A value that
    holds a control character is never shown.
    """
    if not isinstance(value, str):
        _refuse_name(key, f"is {_shown(value)}, but it must be a file name in quotes", suffix)
    problem = _path_problem(value)
    if problem is not None:
        _refuse_name(key, problem, suffix)
    if suffix is not None and not value.endswith(suffix):
        raise ExpectedFailure(f"The setting {key} is {value!r}, but it must end in exactly "
                              f"{suffix}, in lower case, because splat-transform picks how to "
                              f"read or write a file from its ending. Change its ending to "
                              f"{suffix}.")
    return value


# The file names

def _refuse_name(key: str, problem: str, suffix: str | None) -> NoReturn:
    """Refuse the file-name setting `key`, saying what's wrong and what it must be instead."""
    ending = "" if suffix is None else f" that ends in {suffix}"
    raise ExpectedFailure(f"The setting {key} {problem}. Change it to a plain file name{ending}, "
                          "with no folder in it.")


def _path_problem(name: str) -> str | None:
    """What stops `name` being a plain file name, in the order Req 6.9's checks come, or None.

    With no separator in it, a name has one part, so not being .. means holding no .. part.
    A name that holds a control character isn't shown, because it would garble the message.
    """
    if not name:
        return "is empty"
    shown = "" if _has_control(name) else f"is {name!r}, which "
    separator = next((char for char in name if char in _SEPARATORS), None)
    if separator is not None:
        return f"{shown}holds a {separator}, so it's a path, not the name of a file in out/"
    if name in (".", ".."):
        return f"{shown}names a folder, not a file in out/"
    if ":" in name:
        return f"{shown}holds a :, and Windows reads a : as a drive or as a stream inside a file"
    if _has_control(name):
        return ("holds a control character, like a line break or a tab, which a file name "
                "can't hold, so it isn't shown here")
    if any(path.is_absolute() or path.drive
           for path in (PurePosixPath(name), PureWindowsPath(name))):
        return f"{shown}is an absolute path or names a drive, so it leads outside out/"
    return None


def _has_control(name: str) -> bool:
    """Whether `name` holds a control character: one below U+0020, the space."""
    return any(ord(char) < 0x20 for char in name)


def _refuse_clashes(section: str, names: Mapping[str, str]) -> None:
    """Refuse two of the three file names that are the same once case is ignored (Req 6.10)."""
    for first, second, overwritten in _CLASHES:
        if names[first].casefold() == names[second].casefold():
            raise ExpectedFailure(
                f"In [{section}], {first} is {names[first]!r} and {second} is "
                f"{names[second]!r}, which are the same name once case is ignored, and names "
                "that differ only in case are one file on Windows. The step would overwrite "
                f"{overwritten}, so give the two settings different names.")


# The whole numbers and the numbers

def _table(settings: StepSettings, key: str) -> str:
    """The table `key` is in: [<section>.<preset>] for a Preset_Setting, else [<section>]."""
    if key in _PRESET_SETTINGS:
        return f"[{settings.section}.{settings.preset}]"
    return f"[{settings.section}]"


def _whole_number(table: str, key: str, value: object, wanted: _Range) -> int:
    """`value` when it's an int in its range. A bool or a float, even 4.0, is refused (Req 6.7)."""
    if type(value) is not int or not wanted.holds(value):
        _refuse(table, key, value, f"a whole number {wanted.words}")
    return value


def _number(table: str, key: str, value: object, wanted: _Range) -> float:
    """`value` as a float, when it's an int or a float in its range but never a bool (Req 6.8).

    Config refuses NaN and infinity already, so the number is finite. An int too big for a
    float is refused as out of range, as [jobs] refuses one.
    """
    words = f"a number {wanted.words}"
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not wanted.holds(value):
        _refuse(table, key, value, words)
    try:
        return float(value)
    except OverflowError as error:  # an int too big for a float
        raise ExpectedFailure(f"In {table}, {key} is {_shown(value)}, which is too big. It must "
                              f"be {words}, so change it to a smaller one.") from error


def _refuse(table: str, key: str, value: object, wanted: str) -> NoReturn:
    """Refuse a whole number or a number, naming its table and saying what it must be."""
    raise ExpectedFailure(f"In {table}, {key} is {_shown(value)}, but it must be {wanted}. "
                          f"Change it to {wanted}.")


def _shown(value: object) -> str:
    """A value about as TOML writes it: true, 0.5, 'text'."""
    if isinstance(value, bool):
        return "true" if value else "false"
    return repr(value)
