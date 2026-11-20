"""The Config_Loader: the Pipeline's settings and Presets, read from config.toml with tomllib.

`load_config()` reads CONFIG_PATH, which tests point at their own file, and returns a `Config`.
A `Config` refuses NaN, infinity and TOML dates and times in every Pipeline_Section, meaning
every top-level key except the sections in NOT_PIPELINE. `jobs()` gives the checked [jobs]
settings, and `step_settings()` gives one Step's Shared_Settings and Preset_Settings for one
Preset. Only `step_settings` reads a Step_Section, by name, so a new section that isn't one,
like [cloud], needs no code here. Every refusal is a PipelineError with a Readable_Error.

From splatpipe, this module imports only splatpipe.errors.
"""

from __future__ import annotations

import copy
import datetime
import math
import tomllib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, NoReturn

from splatpipe.errors import PipelineError

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "config.toml"  # tests point this at their own file
PRESETS = ("fast", "default", "quality")
NOT_PIPELINE = frozenset({"git", "aws", "paid", "devlog"})  # not Pipeline_Sections, so not checked
JOBS_SECTION = "jobs"  # the Skeleton's own section


def load_config(path: Path | None = None) -> Config:
    """Read Config from `path`, or from CONFIG_PATH as it is at call time when `path` is None.

    A file that's missing, can't be read, isn't UTF-8 or isn't valid TOML is refused with a
    PipelineError that names the file.
    """
    file = CONFIG_PATH if path is None else path
    try:
        with open(file, "rb") as handle:
            data = tomllib.load(handle)
    except FileNotFoundError as error:
        raise PipelineError(f"The config file {file} doesn't exist. Check the path, or get "
                            "config.toml back from git if it was removed.") from error
    except OSError as error:
        raise PipelineError(f"The config file {file} can't be read ({error.strerror or error}). "
                            "Check that it's a file you can open.") from error
    except UnicodeDecodeError as error:
        raise PipelineError(f"The config file {file} isn't UTF-8 text. Save it as UTF-8 "
                            "and try again.") from error
    except ValueError as error:  # TOMLDecodeError, or an int with too many digits
        raise PipelineError(f"The config file {file} isn't valid TOML: {error}. Fix it there "
                            "and try again.") from error
    return Config(data, source=str(file))


def check_preset(name: object) -> str:
    """Return `name` when it's one of PRESETS. Refuse anything else, listing the three."""
    if isinstance(name, str) and name in PRESETS:
        return PRESETS[PRESETS.index(name)]  # a plain str: JSON files refuse str subclasses
    raise PipelineError(f"{_shown(name)} isn't a preset. Use {_listed(PRESETS, 'or')}.")


@dataclass(frozen=True)
class JobsSettings:
    """The checked [jobs] settings. Each field is named after its key in [jobs]."""

    jobs_folder: Path  # absolute: a relative path in Config is read from REPO_ROOT
    default_preset: str
    max_video_bytes: int
    max_video_seconds: float
    min_used_seconds: float
    ffprobe_timeout_seconds: float


class StepSettings(dict[str, Any]):
    """One Step's Shared_Settings and Preset_Settings for one Preset, in one dict.

    `settings[key]` with a missing key raises PipelineError, naming [<section>],
    [<section>.<preset>] and the key. `get` and `in` work as for any dict.
    """

    section: str  # the Step_Name, like "fake", used in messages
    preset: str

    def __init__(self, values: Mapping[str, Any], *, section: str, preset: str) -> None:
        super().__init__(values)
        self.section = section
        self.preset = preset

    def __missing__(self, key: str) -> NoReturn:
        name = self.section
        raise PipelineError(
            f"The {name} step needs the setting {key}, but neither [{name}] nor "
            f"[{name}.{self.preset}] in the config has it. Add {key} to [{name}] if it's the "
            f"same for every preset, or to each of {_tables(name, PRESETS)} if it isn't.")


class Config:
    """The data of one config file, with every Pipeline_Section checked."""

    def __init__(self, data: Mapping[str, Any], source: str = "config") -> None:
        """Deep-copy `data`, a dict as tomllib gives it, and check every Pipeline_Section.

        `source` names the data in messages: the file's path, or "config".
        """
        self._data: dict[str, Any] = copy.deepcopy(dict(data))
        self._source = source
        for name, value in self._data.items():
            if name not in NOT_PIPELINE:
                _check_section(source, name, value)

    def jobs(self) -> JobsSettings:
        """The six [jobs] settings, each checked. A relative jobs_folder is read from REPO_ROOT."""
        source = self._source
        keys = _listed([field.name for field in fields(JobsSettings)])
        table = _section(source, self._data, JOBS_SECTION, f"the settings {keys}")
        return JobsSettings(
            jobs_folder=_jobs_folder(source, table),
            default_preset=_jobs_preset(source, table),
            max_video_bytes=_jobs_whole(source, table, "max_video_bytes"),
            max_video_seconds=_jobs_seconds(source, table, "max_video_seconds"),
            min_used_seconds=_jobs_seconds(source, table, "min_used_seconds"),
            ffprobe_timeout_seconds=_jobs_seconds(source, table, "ffprobe_timeout_seconds"),
        )

    def step_settings(self, name: str, preset: str) -> StepSettings:
        """Step `name`'s Shared_Settings and its Preset_Settings for `preset`, in one dict.

        The Preset is checked first, then the section [name], in this order: it exists and is
        a table, its only tables are Preset tables, all three Preset tables exist, they have
        the same keys, and no key is both shared and per-Preset.
        """
        source = self._source
        preset = check_preset(preset)
        section = _section(source, self._data, name, f"its Preset tables {_tables(name, PRESETS)}")
        _refuse_other_tables(source, name, section)
        tables = _preset_tables(source, name, section)
        _refuse_uneven_keys(source, name, tables)
        shared = {key: value for key, value in section.items() if key not in PRESETS}
        _refuse_shared_and_per_preset(source, name, shared, tables)
        values = copy.deepcopy({**shared, **tables[preset]})  # so a Step can't change the Config
        return StepSettings(values, section=name, preset=preset)


# Checking every Pipeline_Section

def _check_section(source: str, name: str, value: object) -> None:
    """Refuse NaN, infinity and TOML dates and times in one Pipeline_Section."""
    if isinstance(value, dict):
        _check_values(source, name, "", value)
    else:  # a top-level key that isn't a table
        _check_values(source, None, name, value)


def _check_values(source: str, section: str | None, path: str, value: object) -> None:
    """Refuse NaN, infinity and dates anywhere in `value`. `path` is like `fast.sizes[2]`."""
    if isinstance(value, dict):
        for key, item in value.items():
            _check_values(source, section, f"{path}.{key}" if path else str(key), item)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _check_values(source, section, f"{path}[{index}]", item)
    elif isinstance(value, float) and not math.isfinite(value):
        raise PipelineError(f"In {source}, {_place(section, path)} is {_shown(value)}. Settings "
                            "must be finite numbers, so change it to one.")
    elif isinstance(value, (datetime.date, datetime.time)):  # a datetime is a date too
        raise PipelineError(f"In {source}, {_place(section, path)} is the date or time "
                            f"{_shown(value)}. Settings can't be TOML dates or times, so write it "
                            "as text in quotes, or as a number.")


def _section(source: str, data: dict[str, Any], name: str, holds: str) -> dict[str, Any]:
    """The top-level table `name`. `holds` says what the section holds, for the messages."""
    if name not in data:
        raise PipelineError(f"{source} has no [{name}] section. Add [{name}], with {holds}.")
    table = data[name]
    if not isinstance(table, dict):
        raise PipelineError(f"In {source}, {name} is {_shown(table)}, but it must be the section "
                            f"[{name}], with {holds}. Replace it with that section.")
    return table


# The [jobs] settings

def _jobs_value(source: str, table: dict[str, Any], key: str) -> object:
    """One [jobs] value, refused by name when it's missing."""
    if key not in table:
        raise PipelineError(f"In {source}, [{JOBS_SECTION}] has no setting {key}. "
                            f"Add {key} to [{JOBS_SECTION}].")
    return table[key]


def _jobs_folder(source: str, table: dict[str, Any]) -> Path:
    """jobs_folder as an absolute path. A relative one is joined to REPO_ROOT, never the cwd."""
    value = _jobs_value(source, table, "jobs_folder")
    # On Windows, REPO_ROOT / "C:jobs" is still relative, to drive C's current folder.
    if not isinstance(value, str) or not value.strip() or not (REPO_ROOT / value).is_absolute():
        raise PipelineError(f"In {source}, {_place(JOBS_SECTION, 'jobs_folder')} is "
                            f"{_shown(value)}, but it must be a folder's path in quotes: a full "
                            'path, or a path from the repo root like "data/jobs". '
                            "Change it to one.")
    return REPO_ROOT / value


def _jobs_preset(source: str, table: dict[str, Any]) -> str:
    """default_preset, refused unless it's one of PRESETS."""
    value = _jobs_value(source, table, "default_preset")
    try:
        return check_preset(value)
    except PipelineError as error:
        raise PipelineError(f"In {source}, {_place(JOBS_SECTION, 'default_preset')} is wrong: "
                            f"{error}") from error


def _jobs_whole(source: str, table: dict[str, Any], key: str) -> int:
    """A whole number above 0. A bool or a float is refused."""
    value = _jobs_value(source, table, key)
    if type(value) is not int or value <= 0:
        _refuse_number(source, key, value, "a whole number above 0")
    return value


def _jobs_seconds(source: str, table: dict[str, Any], key: str) -> float:
    """A number of seconds above 0, an int or a float but never a bool, given back as a float."""
    value = _jobs_value(source, table, key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not value > 0:
        _refuse_number(source, key, value, "a number above 0")
    try:
        return float(value)
    except OverflowError as error:  # an int too big for a float
        raise PipelineError(f"In {source}, {_place(JOBS_SECTION, key)} is too big. Change it "
                            "to a smaller number of seconds.") from error


def _refuse_number(source: str, key: str, value: object, wanted: str) -> NoReturn:
    """Refuse a [jobs] number by name, saying what it must be."""
    raise PipelineError(f"In {source}, {_place(JOBS_SECTION, key)} is {_shown(value)}, but it "
                        f"must be {wanted}. Change it to {wanted}.")


# The layout of a Step_Section

def _refuse_other_tables(source: str, name: str, section: dict[str, Any]) -> None:
    """Every table in a Step_Section is one of its Preset tables."""
    for key, value in section.items():
        if isinstance(value, dict) and key not in PRESETS:
            raise PipelineError(
                f"In {source}, [{name}] holds a table named {key}, but the only tables a step's "
                f"section can hold are its Preset tables {_tables(name, PRESETS)}. Rename "
                f"[{name}.{key}] to one of those, or move its settings into [{name}].")


def _preset_tables(source: str, name: str, section: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """The three Preset tables, by Preset, refused when one is missing or isn't a table."""
    for preset in PRESETS:
        if preset in section and not isinstance(section[preset], dict):
            raise PipelineError(
                f"In {source}, {_place(name, preset)} is {_shown(section[preset])}, but {preset} "
                f"is a preset, so it must be the table [{name}.{preset}]. Move the value into "
                f"[{name}.{preset}] as a setting with its own name.")
    missing = [preset for preset in PRESETS if preset not in section]
    if missing:
        raise PipelineError(
            f"In {source}, [{name}] has no {_tables(name, missing, 'or')}. A step's section "
            f"needs all three Preset tables, {_tables(name, PRESETS)}, with the same settings in "
            f"each, so add {_tables(name, missing)}.")
    return {preset: section[preset] for preset in PRESETS}


def _refuse_uneven_keys(source: str, name: str, tables: dict[str, dict[str, Any]]) -> None:
    """The three Preset tables have the same keys. A refusal names the key and where it's missing."""
    for table in tables.values():
        for key in table:
            having = [preset for preset in PRESETS if key in tables[preset]]
            lacking = [preset for preset in PRESETS if key not in tables[preset]]
            if lacking:
                raise PipelineError(
                    f"In {source}, the setting {key} is in {_tables(name, having)} but not in "
                    f"{_tables(name, lacking, 'or')}. The Preset tables of [{name}] need the same "
                    f"settings, so add {key} to {_tables(name, lacking)}, or remove it from "
                    f"{_tables(name, having)}.")


def _refuse_shared_and_per_preset(source: str, name: str, shared: Mapping[str, Any],
                                  tables: dict[str, dict[str, Any]]) -> None:
    """No key is both a Shared_Setting and a Preset_Setting."""
    for key in shared:
        if any(key in table for table in tables.values()):
            raise PipelineError(
                f"In {source}, the setting {key} is in [{name}] and also in its Preset tables. "
                f"Keep it in one place: in [{name}] if it's the same for every preset, or in "
                f"{_tables(name, PRESETS)} if it isn't.")


# Words for the messages

def _place(section: str | None, path: str) -> str:
    """Where a setting is, like `the setting fast.files in [fake]`."""
    if section is None:
        return f"the top-level setting {path}"
    return f"the setting {path} in [{section}]"


def _tables(name: str, presets: Iterable[str], word: str = "and") -> str:
    """Preset tables as TOML headers, like `[fake.fast], [fake.default] and [fake.quality]`."""
    return _listed([f"[{name}.{preset}]" for preset in presets], word)


def _listed(items: Iterable[str], word: str = "and") -> str:
    """Items joined for a sentence: `a`, `a and b`, `a, b and c`."""
    words = list(items)
    if len(words) < 2:
        return "".join(words)
    return f"{', '.join(words[:-1])} {word} {words[-1]}"


def _shown(value: object) -> str:
    """A value about as TOML writes it: true, nan, 1979-05-27, 'text'."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (datetime.date, datetime.time)):
        return value.isoformat()
    return repr(value)
