"""The Cloud_Config: the `[cloud]` Config section and the reader that loads it.

`load_cloud_config(path)` validates the whole file as pipeline Config first, so the Config_Loader
refuses NaN, infinity and TOML dates or times in `[cloud]` like any Pipeline_Section (Req 2.1).
Then it re-reads the file with `tomllib` and builds a `CloudConfig` from the raw `[cloud]` table,
checking the cloud-specific keys itself (Req 2.2): every missing or wrong-typed value is a
`CloudError` that names the key and what it must be, because `splatpipe/config.py` has no accessor
for `[cloud]`. `CloudConfig.from_mapping(data)` does the same checks on a dict, so selection and
launch tests need no `config.toml` on disk.

A cloud module may import `splatpipe` to call its public functions; `splatpipe` never imports
`cloud`. This module imports no boto3 and no GPU package, so it loads on a laptop.
"""

from __future__ import annotations

import math
import tomllib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import splatpipe.config

from cloud.errors import CloudError

PRESETS = splatpipe.config.PRESETS  # ("fast", "default", "quality"), the Pipeline's three Presets


@dataclass(frozen=True)
class MachineSpec:
    """One Allowed_Type's resources, from a `[cloud.types.<type>]` table."""

    type: str
    vcpus: int
    memory_gb: float
    gpu_memory_gb: float
    price_per_hour_usd: float
    price_date: str


@dataclass(frozen=True)
class RunPlan:
    """The Expected_Time and Time_Cap for the Smoke_Test or the Image_Build, in minutes."""

    expected_minutes: float          # for the money quote
    cap_minutes: float               # the Time_Cap


@dataclass(frozen=True)
class SmokeChecks:
    """The smoke machine's own settings, from `[cloud.smoke]` beside its Expected_Time and Time_Cap."""

    check_timeout_seconds: float     # how long each command check may take
    scene_gaussians: int             # Gaussians in the tiny synthetic scene gsplat renders
    image_pixels: int                # width and height of the render, the SSIM images and the video
    video_seconds: float             # length of the test video the fake-step job is made from


@dataclass(frozen=True)
class PresetPlan:
    """One Preset's ordered Fallback_List and its Expected_Time and Time_Cap, in minutes."""

    fallback: tuple[str, ...]        # ordered Allowed_Type names, primary first
    expected_minutes: float
    cap_minutes: float


@dataclass(frozen=True)
class CloudConfig:
    """The checked `[cloud]` section: the machine types, the Presets, the prices and the times."""

    types: dict[str, MachineSpec]            # one per Allowed_Type
    presets: dict[str, PresetPlan]           # "fast", "default", "quality"
    smoke: RunPlan                           # expected/cap for the smoke test
    smoke_checks: SmokeChecks                # the smoke machine's own settings
    image: RunPlan                           # expected/cap for the image build
    upload_margin_minutes: float
    poll_seconds: float
    ipv4_usd_per_hour: float
    gp3_usd_per_gb_month: float
    root_gb: float                           # root-volume size in GB for every machine (150)
    smoke_instance_type: str
    image_builder_type: str
    base_ami_parameter: str

    def spec(self, instance_type: str) -> MachineSpec:
        """The Machine_Spec for `instance_type`, or a CloudError naming the type if absent."""
        try:
            return self.types[instance_type]
        except KeyError:
            raise CloudError(
                f"No Machine_Spec for the instance type {instance_type!r}. The [cloud.types] "
                f"tables cover {_listed(sorted(self.types))}.") from None

    def preset_plan(self, preset: str) -> PresetPlan:
        """The PresetPlan for `preset`, or a CloudError naming the preset if absent."""
        try:
            return self.presets[preset]
        except KeyError:
            raise CloudError(
                f"No cloud plan for the preset {preset!r}. The Presets are "
                f"{_listed(PRESETS)}.") from None

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> CloudConfig:
        """Build a CloudConfig from a raw `[cloud]` mapping, checking every key (Req 2.2, 2.10)."""
        if not isinstance(data, Mapping):
            raise CloudError(
                "The [cloud] config must be a table of cloud settings. Add a [cloud] section to "
                "config.toml, then run the command again.")
        types = _read_types(data)
        return cls(
            types=types,
            presets=_read_presets(data, types),
            smoke=_read_run_plan(data, "smoke"),
            smoke_checks=_read_smoke_checks(data),
            image=_read_run_plan(data, "image"),
            upload_margin_minutes=_number(data, "upload_margin_minutes", "upload_margin_minutes"),
            poll_seconds=_number(data, "poll_seconds", "poll_seconds"),
            ipv4_usd_per_hour=_number(data, "ipv4_usd_per_hour", "ipv4_usd_per_hour"),
            gp3_usd_per_gb_month=_number(data, "gp3_usd_per_gb_month", "gp3_usd_per_gb_month"),
            root_gb=_number(data, "root_gb", "root_gb"),
            smoke_instance_type=_type_key(data, "smoke_instance_type", types),
            image_builder_type=_type_key(data, "image_builder_type", types),
            base_ami_parameter=_text(data, "base_ami_parameter", "base_ami_parameter"),
        )


def load_cloud_config(path: Path | None = None) -> CloudConfig:
    """Validate the whole file as pipeline Config, then read and check the `[cloud]` table.

    `path` defaults to `splatpipe.config.CONFIG_PATH`, read at call time so a test's temporary
    config file is honoured. `load_config` runs first, so NaN, infinity and TOML dates or times in
    `[cloud]` are refused there like any Pipeline_Section (Req 2.1).
    """
    file = splatpipe.config.CONFIG_PATH if path is None else Path(path)
    splatpipe.config.load_config(file)
    with open(file, "rb") as handle:
        data = tomllib.load(handle)
    cloud = data.get("cloud")
    if not isinstance(cloud, Mapping):
        raise CloudError(
            f"{file} has no [cloud] section. Add [cloud] with the machine types, the Presets, the "
            "prices and the times, then run the command again.")
    return CloudConfig.from_mapping(cloud)


# Reading the tables of [cloud]

def _read_types(cloud: Mapping[str, object]) -> dict[str, MachineSpec]:
    """Every `[cloud.types.<type>]` table as a MachineSpec, keyed by type name."""
    raw = _table(cloud, "types", "the [cloud.types] tables")
    specs: dict[str, MachineSpec] = {}
    for name, table in raw.items():
        where = f'the type "{name}"'
        if not isinstance(table, Mapping):
            raise CloudError(
                f'In the [cloud] config, {where} is {_shown(table)}, but it must be a '
                f'[cloud.types."{name}"] table with vcpus, memory_gb, gpu_memory_gb, '
                "price_per_hour_usd and price_date. Fix it in config.toml, then run the command "
                "again.")
        specs[name] = MachineSpec(
            type=name,
            vcpus=_whole(table, "vcpus", f"vcpus for {where}"),
            memory_gb=_number(table, "memory_gb", f"memory_gb for {where}"),
            gpu_memory_gb=_number(table, "gpu_memory_gb", f"gpu_memory_gb for {where}"),
            price_per_hour_usd=_number(table, "price_per_hour_usd", f"price_per_hour_usd for {where}"),
            price_date=_text(table, "price_date", f"price_date for {where}"),
        )
    return specs


def _read_presets(cloud: Mapping[str, object],
                  types: Mapping[str, MachineSpec]) -> dict[str, PresetPlan]:
    """The three `[cloud.presets.<preset>]` tables as PresetPlans. Exactly fast, default, quality."""
    raw = _table(cloud, "presets", "the [cloud.presets] tables")
    if set(raw) != set(PRESETS):
        raise CloudError(
            f"The [cloud.presets] tables must be exactly {_listed(PRESETS)}, but the config has "
            f"{_listed(sorted(raw)) or 'none'}. Fix them under [cloud.presets] in config.toml, "
            "then run the command again.")
    plans: dict[str, PresetPlan] = {}
    for preset in PRESETS:
        table = _table(raw, preset, f'the "{preset}" preset')
        plans[preset] = PresetPlan(
            fallback=_read_fallback(table, preset, types),
            expected_minutes=_number(table, "expected_minutes", f'expected_minutes for the "{preset}" preset'),
            cap_minutes=_number(table, "cap_minutes", f'cap_minutes for the "{preset}" preset'),
        )
    return plans


def _read_fallback(table: Mapping[str, object], preset: str,
                   types: Mapping[str, MachineSpec]) -> tuple[str, ...]:
    """A Preset's ordered Fallback_List: a list of strings, each a key of `types`."""
    label = f'fallback for the "{preset}" preset'
    value = _get(table, "fallback", label)
    if not isinstance(value, list) or not all(isinstance(entry, str) for entry in value):
        raise CloudError(
            f"In the [cloud] config, {label} is {_shown(value)}, but it must be a list of "
            f"Allowed_Type names in quotes. Fix it under [cloud.presets.{preset}] in config.toml, "
            "then run the command again.")
    for entry in value:
        if entry not in types:
            raise CloudError(
                f'In the [cloud] config, {label} lists "{entry}", which has no [cloud.types] '
                f"table. Every fallback entry must name an Allowed_Type ({_listed(sorted(types))}). "
                f"Fix it under [cloud.presets.{preset}] in config.toml, then run the command again.")
    return tuple(value)


def _read_run_plan(cloud: Mapping[str, object], which: str) -> RunPlan:
    """The `[cloud.smoke]` or `[cloud.image]` table as a RunPlan."""
    table = _table(cloud, which, f"the [cloud.{which}] table")
    return RunPlan(
        expected_minutes=_number(table, "expected_minutes", f"expected_minutes for [cloud.{which}]"),
        cap_minutes=_number(table, "cap_minutes", f"cap_minutes for [cloud.{which}]"),
    )


def _read_smoke_checks(cloud: Mapping[str, object]) -> SmokeChecks:
    """The smoke machine's own settings from the `[cloud.smoke]` table, each checked by name."""
    table = _table(cloud, "smoke", "the [cloud.smoke] table")

    def label(key: str) -> str:
        return f"{key} for [cloud.smoke]"

    return SmokeChecks(
        check_timeout_seconds=_number(table, "check_timeout_seconds", label("check_timeout_seconds")),
        scene_gaussians=_whole(table, "scene_gaussians", label("scene_gaussians")),
        image_pixels=_whole(table, "image_pixels", label("image_pixels")),
        video_seconds=_number(table, "video_seconds", label("video_seconds")),
    )


def _type_key(cloud: Mapping[str, object], key: str, types: Mapping[str, MachineSpec]) -> str:
    """A `[cloud]` value that must name an Allowed_Type: a non-empty string that is a key of `types`."""
    value = _text(cloud, key, key)
    if value not in types:
        raise CloudError(
            f'In the [cloud] config, {key} is "{value}", which has no [cloud.types] table. It must '
            f"name an Allowed_Type ({_listed(sorted(types))}). Fix it under [cloud] in config.toml, "
            "then run the command again.")
    return value


# Checking one value, naming the key and what it must be

def _get(table: Mapping[str, object], key: str, label: str) -> object:
    """The value for `key`, or a CloudError naming the missing key."""
    try:
        return table[key]
    except KeyError:
        raise CloudError(
            f"The [cloud] config is missing {label}. Add {key} in config.toml, then run the command "
            "again.") from None


def _table(parent: Mapping[str, object], key: str, label: str) -> Mapping[str, object]:
    """A sub-table of `[cloud]`, or a CloudError naming the key when it is missing or not a table."""
    value = _get(parent, key, label)
    if not isinstance(value, Mapping):
        raise CloudError(
            f"In the [cloud] config, {label} is {_shown(value)}, but it must be a table. Fix it in "
            "config.toml, then run the command again.")
    return value


def _number(table: Mapping[str, object], key: str, label: str) -> float:
    """A finite number above 0 for `key`, an int or float but never a bool, given back as a float."""
    value = _get(table, key, label)
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value <= 0):
        raise CloudError(
            f"In the [cloud] config, {label} is {_shown(value)}, but it must be a finite number "
            "above 0. Fix it in config.toml, then run the command again.")
    return float(value)


def _whole(table: Mapping[str, object], key: str, label: str) -> int:
    """A whole number above 0 for `key`. A bool or a float is refused."""
    value = _get(table, key, label)
    if type(value) is not int or value <= 0:
        raise CloudError(
            f"In the [cloud] config, {label} is {_shown(value)}, but it must be a whole number "
            "above 0. Fix it in config.toml, then run the command again.")
    return value


def _text(table: Mapping[str, object], key: str, label: str) -> str:
    """A non-empty string for `key`."""
    value = _get(table, key, label)
    if not isinstance(value, str) or not value:
        raise CloudError(
            f"In the [cloud] config, {label} is {_shown(value)}, but it must be a non-empty string. "
            "Fix it in config.toml, then run the command again.")
    return value


# Words for the messages

def _listed(items: Iterable[str]) -> str:
    """Items joined for a sentence: `a`, `a and b`, `a, b and c`."""
    words = [str(item) for item in items]
    if len(words) < 2:
        return "".join(words)
    return f"{', '.join(words[:-1])} and {words[-1]}"


def _shown(value: object) -> str:
    """A value about as TOML writes it: true for a bool, else its repr."""
    if isinstance(value, bool):
        return "true" if value else "false"
    return repr(value)
