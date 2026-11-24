"""Unit tests for the Cloud_Config, cloud/config.py (cloud-jobs task 1.4).

Two things, as the task and design describe. First, the real config.toml `[cloud]` loads through
`load_cloud_config`: there are 8 `types`, each a full Machine_Spec; every Preset Fallback_List is
non-empty and names only Allowed_Types; the pinned primaries and fallbacks and the Q2 minutes are
present; `[cloud]` holds no `[cloud.needs.*]` table; and `smoke_instance_type` and
`image_builder_type` are Allowed_Types (Req 2.3, 2.4, 2.6, 2.7, 2.10). `[cloud.smoke]` also holds
the training check's seven numbers at their A11 values (training spec, Req 12.4) and the depth
check's `depth_normal_start_step` at 8 (depth-priors spec, Req 16.6). Second, corrupting one key
of a valid `[cloud]`: `nan`, `inf` or a TOML date is refused by the Config_Loader naming the key
(Req 2.1), while a missing or wrong-typed key is a `CloudError` naming the key (Req 2.2). A float,
a bool or 0 as one of the training check's numbers is a `CloudError` naming it as
`<key> for [cloud.smoke]` (training spec, Req 12.5), and so is −1, 1.5, `true` or `"8"` as
`depth_normal_start_step`, which may be 0 (depth-priors spec, Req 16.6).

Property 2 (any single corruption of a valid `[cloud]` is refused by name) is the Hypothesis test
at the end of this file (task 1.5): it builds a valid `[cloud]` mapping, corrupts one key, writes it
to a temp config.toml and checks that load_cloud_config refuses it with a message naming that key.
These tests run on the CPU with no network and no AWS: the real-config cases read config.toml by
path, and the corruption cases build a `[cloud]` in memory or write one to a temp config.toml.
"""

import copy
import datetime
import math
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from splatpipe.config import PRESETS, REPO_ROOT, load_config
from splatpipe.errors import PipelineError

from cloud.config import CloudConfig, MachineSpec, load_cloud_config
from cloud.errors import CloudError

# The real config.toml, read by path so these tests still see it once the autouse temp_config
# fixture has pointed splatpipe.config.CONFIG_PATH at a test config with no [cloud] section.
REAL_CONFIG = REPO_ROOT / "config.toml"

# The number of Allowed_Types the [cloud] section defines (Req 2.3).
ALLOWED_TYPE_COUNT = 8

# The pinned primaries and fallbacks (Req 2.4, assumption D): fast and default lead with g5.8xlarge;
# quality is the pinned pair of L40S types, primary then fallback.
FAST_PRIMARY = "g5.8xlarge"
DEFAULT_PRIMARY = "g5.8xlarge"
QUALITY_FALLBACK = ("g6e.4xlarge", "g6e.2xlarge")

# The Q2 minutes as (expected, cap), settled in Q2 (Req 2.5, 2.6).
PRESET_MINUTES = {"fast": (30, 120), "default": (60, 180), "quality": (120, 360)}
SMOKE_MINUTES = (10, 30)
IMAGE_MINUTES = (75, 120)

# The training check's seven [cloud.smoke] numbers, at their A11 values (training spec, Req 12.4).
TRAIN_SMOKE = {"train_images": 4, "train_points": 200, "train_splats": 400, "train_steps": 30,
               "train_refine_start_step": 10, "train_refine_every": 10,
               "train_sh_degree_interval": 5}

# The depth check's Normal_Start in [cloud.smoke] (depth-priors spec, Req 16.6, A5).
DEPTH_NORMAL_START_STEP = 8


@pytest.fixture
def cc() -> CloudConfig:
    """The real config.toml `[cloud]` section, loaded through the Cloud_Config reader."""
    return load_cloud_config(REAL_CONFIG)


# The real config.toml loads

def test_the_real_cloud_section_loads_with_eight_full_machine_specs(cc: CloudConfig) -> None:
    assert isinstance(cc, CloudConfig)
    assert len(cc.types) == ALLOWED_TYPE_COUNT
    for name, spec in cc.types.items():
        assert isinstance(spec, MachineSpec)
        assert spec.type == name
        assert type(spec.vcpus) is int and spec.vcpus > 0
        assert spec.memory_gb > 0
        assert spec.gpu_memory_gb > 0
        assert spec.price_per_hour_usd > 0
        assert isinstance(spec.price_date, str) and spec.price_date


@pytest.mark.parametrize("preset", PRESETS)
def test_each_preset_fallback_is_non_empty_and_all_allowed_types(cc: CloudConfig, preset: str) -> None:
    fallback = cc.presets[preset].fallback
    assert fallback  # non-empty
    for entry in fallback:
        assert entry in cc.types


def test_the_pinned_primaries_and_fallbacks(cc: CloudConfig) -> None:
    assert cc.presets["fast"].fallback[0] == FAST_PRIMARY
    assert cc.presets["default"].fallback[0] == DEFAULT_PRIMARY
    assert cc.presets["quality"].fallback == QUALITY_FALLBACK


@pytest.mark.parametrize("preset", PRESETS)
def test_each_preset_has_its_q2_minutes(cc: CloudConfig, preset: str) -> None:
    plan = cc.presets[preset]
    assert (plan.expected_minutes, plan.cap_minutes) == PRESET_MINUTES[preset]


def test_the_smoke_and_image_runs_have_their_q2_minutes(cc: CloudConfig) -> None:
    assert (cc.smoke.expected_minutes, cc.smoke.cap_minutes) == SMOKE_MINUTES
    assert (cc.image.expected_minutes, cc.image.cap_minutes) == IMAGE_MINUTES


def test_the_smoke_machines_settings_load_and_its_video_is_long_enough_for_a_job(
        cc: CloudConfig) -> None:
    checks = cc.smoke_checks
    assert checks.check_timeout_seconds > 0
    assert type(checks.scene_gaussians) is int and checks.scene_gaussians > 0
    assert type(checks.image_pixels) is int and checks.image_pixels > 0
    # The smoke machine makes a fake-step job from its test video, so the video has to be at least
    # the shortest part of a video a job may use, or make_job refuses it.
    assert checks.video_seconds >= load_config(REAL_CONFIG).jobs().min_used_seconds
    # The training check's numbers are whole numbers at their A11 values (training spec, Req 12.4).
    assert {key: getattr(checks, key) for key in TRAIN_SMOKE} == TRAIN_SMOKE
    assert all(type(getattr(checks, key)) is int for key in TRAIN_SMOKE)
    # The depth check's Normal_Start is the whole number 8 (depth-priors spec, Req 16.6, A5).
    assert checks.depth_normal_start_step == DEPTH_NORMAL_START_STEP
    assert type(checks.depth_normal_start_step) is int


def test_the_smoke_and_image_instance_types_are_allowed_types(cc: CloudConfig) -> None:
    assert cc.smoke_instance_type in cc.types
    assert cc.image_builder_type in cc.types


def test_the_real_cloud_section_holds_no_needs_table() -> None:
    import tomllib

    data = tomllib.loads(REAL_CONFIG.read_text(encoding="utf-8"))
    assert "needs" not in data["cloud"]  # a Step's needs live in that Step's own section


# Corrupting one key: nan, inf or a date is refused by the Config_Loader naming the key (Req 2.1)

# TOML literals the Config_Loader refuses in any Pipeline_Section, so load_cloud_config refuses them
# in [cloud] before the cloud-specific checks run.
BAD_LITERALS = [
    pytest.param("nan", id="nan"),
    pytest.param("inf", id="inf"),
    pytest.param("1979-05-27", id="a date"),
]


@pytest.mark.parametrize("literal", BAD_LITERALS)
def test_nan_inf_or_a_date_in_cloud_is_refused_by_the_config_loader_naming_the_key(
        tmp_path: Path, literal: str) -> None:
    path = tmp_path / "config.toml"
    path.write_text(f"[cloud]\nupload_margin_minutes = {literal}\n", encoding="utf-8")
    with pytest.raises(PipelineError) as caught:
        load_cloud_config(path)
    message = str(caught.value)
    assert "[cloud]" in message and "upload_margin_minutes" in message


# Corrupting one key: a missing or wrong-typed key is a CloudError naming the key (Req 2.2)

def valid_cloud() -> dict[str, Any]:
    """A valid `[cloud]` mapping, built fresh each call, that `from_mapping` accepts."""
    def machine(vcpus: int, memory_gb: float, gpu_memory_gb: float, price: float) -> dict[str, Any]:
        return {"vcpus": vcpus, "memory_gb": memory_gb, "gpu_memory_gb": gpu_memory_gb,
                "price_per_hour_usd": price, "price_date": "2026-11-21"}

    return {
        "types": {
            "g5.xlarge": machine(4, 16, 24, 1.006),
            "g5.8xlarge": machine(32, 128, 24, 2.448),
        },
        "presets": {
            "fast": {"fallback": ["g5.8xlarge"], "expected_minutes": 30, "cap_minutes": 120},
            "default": {"fallback": ["g5.8xlarge"], "expected_minutes": 60, "cap_minutes": 180},
            "quality": {"fallback": ["g5.8xlarge"], "expected_minutes": 120, "cap_minutes": 360},
        },
        "smoke": {"expected_minutes": 10, "cap_minutes": 30, "check_timeout_seconds": 60,
                  "scene_gaussians": 256, "image_pixels": 64, "video_seconds": 4, **TRAIN_SMOKE,
                  "depth_normal_start_step": DEPTH_NORMAL_START_STEP},
        "image": {"expected_minutes": 45, "cap_minutes": 120},
        "upload_margin_minutes": 10,
        "poll_seconds": 20,
        "ipv4_usd_per_hour": 0.005,
        "gp3_usd_per_gb_month": 0.08,
        "root_gb": 150,
        "smoke_instance_type": "g5.xlarge",
        "image_builder_type": "g5.xlarge",
        "base_ami_parameter": "/aws/service/ami/latest",
    }


# Each case corrupts one key of valid_cloud() and names the key its CloudError must mention. The
# first group removes a key, the second gives one the wrong type.
CORRUPTIONS = [
    pytest.param(lambda c: c.pop("upload_margin_minutes"), "upload_margin_minutes",
                 id="no upload_margin_minutes"),
    pytest.param(lambda c: c.pop("types"), "types", id="no types"),
    pytest.param(lambda c: c.pop("presets"), "presets", id="no presets"),
    pytest.param(lambda c: c.pop("smoke_instance_type"), "smoke_instance_type",
                 id="no smoke_instance_type"),
    pytest.param(lambda c: c["types"]["g5.8xlarge"].pop("gpu_memory_gb"), "gpu_memory_gb",
                 id="a type with no gpu_memory_gb"),
    pytest.param(lambda c: c["presets"]["fast"].pop("cap_minutes"), "cap_minutes",
                 id="a preset with no cap_minutes"),
    pytest.param(lambda c: c.update(upload_margin_minutes="soon"), "upload_margin_minutes",
                 id="text where upload_margin_minutes wants a number"),
    pytest.param(lambda c: c.update(root_gb=True), "root_gb", id="a bool as root_gb"),
    pytest.param(lambda c: c.update(base_ami_parameter=5), "base_ami_parameter",
                 id="a number as base_ami_parameter"),
    pytest.param(lambda c: c["types"]["g5.8xlarge"].update(price_per_hour_usd="free"),
                 "price_per_hour_usd", id="text as a type's price"),
    pytest.param(lambda c: c["types"]["g5.8xlarge"].update(vcpus=True), "vcpus",
                 id="a bool as a type's vcpus"),
    pytest.param(lambda c: c["presets"]["fast"].update(fallback="g5.8xlarge"), "fallback",
                 id="a string as a preset fallback"),
    pytest.param(lambda c: c["smoke"].pop("check_timeout_seconds"), "check_timeout_seconds",
                 id="a smoke table with no check_timeout_seconds"),
    pytest.param(lambda c: c["smoke"].update(scene_gaussians=2.5), "scene_gaussians",
                 id="a float as the smoke scene's Gaussian count"),
    pytest.param(lambda c: c["smoke"].update(image_pixels=0), "image_pixels",
                 id="zero smoke image pixels"),
]


def set_smoke(key: str, value: object) -> Callable[[dict[str, Any]], object]:
    """A corruption that sets `key` in the smoke table to `value`."""
    return lambda c: c["smoke"].update({key: value})


# Each of the training check's seven numbers given a float, here its A11 value as a float, so even
# a whole one is refused, a bool, or 0. The CloudError names it as "<key> for [cloud.smoke]"
# (training spec, Req 12.5).
CORRUPTIONS += [
    pytest.param(set_smoke(key, value), f"{key} for [cloud.smoke]", id=f"{what} as {key}")
    for key, a11 in TRAIN_SMOKE.items()
    for what, value in (("a float", float(a11)), ("a bool", True), ("0", 0))
]

# The depth check's depth_normal_start_step given a number below 0, a float, a bool or its value as
# text. 0 is allowed, so it isn't a corruption here. The CloudError names it as
# "depth_normal_start_step for [cloud.smoke]" (depth-priors spec, Req 16.6).
CORRUPTIONS += [
    pytest.param(set_smoke("depth_normal_start_step", value),
                 "depth_normal_start_step for [cloud.smoke]",
                 id=f"{what} as depth_normal_start_step")
    for what, value in (("-1", -1), ("1.5", 1.5), ("true", True), ('"8"', "8"))
]


@pytest.mark.parametrize(("corrupt", "key"), CORRUPTIONS)
def test_a_missing_or_wrong_typed_cloud_key_is_a_cloud_error_naming_the_key(
        corrupt: Callable[[dict[str, Any]], object], key: str) -> None:
    cloud = valid_cloud()
    corrupt(cloud)
    with pytest.raises(CloudError) as caught:
        CloudConfig.from_mapping(cloud)
    assert key in str(caught.value)


# Property 2: Cloud_Config refuses a corrupted key by name (Req 2.1, 2.2)

# A valid [cloud] mapping is built, one key of it is corrupted, and both are written to a temp
# config.toml and read back through load_cloud_config, the entry point the property names. So a
# wrong type that is still valid TOML (a string, a bool, a number where text belongs) is refused by
# the Cloud_Config naming the key (Req 2.2), while NaN, infinity or a TOML date where a number
# belongs is refused first by the Config_Loader naming the key (Req 2.1), exactly as
# load_cloud_config layers the two checks. The mapping is written as TOML by the small writer
# below, because tomllib only reads.

# The 8 Allowed_Type names, whose dots exercise the writer's quoted table keys.
ALLOWED_TYPES = ("g5.xlarge", "g5.2xlarge", "g5.4xlarge", "g5.8xlarge",
                 "g6.2xlarge", "g6.4xlarge", "g6e.2xlarge", "g6e.4xlarge")

# Finite numbers above 0, in a range whose repr is plain decimal (never 1e-05), so the writer's
# output is valid TOML that reads back to a value the Cloud_Config accepts.
cloud_numbers = st.floats(min_value=0.001, max_value=100000, allow_nan=False, allow_infinity=False)
# Non-empty printable-ASCII text, no quote or backslash, for the string-valued keys.
cloud_text = st.text(
    st.characters(min_codepoint=33, max_codepoint=126, exclude_characters='"\\'),
    min_size=1, max_size=16)


@st.composite
def valid_cloud_mappings(draw: st.DrawFn) -> dict[str, Any]:
    """A valid `[cloud]` mapping that from_mapping accepts: 1-3 Allowed_Types, the three Presets
    whose fallbacks and the smoke/image instance types all name those types, and positive times,
    prices and scalars. The smoke table's whole numbers, the training check's seven among them,
    are integers above 0, and the depth check's depth_normal_start_step is an integer of 0 or
    more (depth-priors spec, Req 16.6)."""
    names = draw(st.lists(st.sampled_from(ALLOWED_TYPES), min_size=1, max_size=3, unique=True))
    types = {
        name: {
            "vcpus": draw(st.integers(min_value=1, max_value=128)),
            "memory_gb": draw(cloud_numbers),
            "gpu_memory_gb": draw(cloud_numbers),
            "price_per_hour_usd": draw(cloud_numbers),
            "price_date": draw(cloud_text),
        }
        for name in names
    }
    presets = {
        preset: {
            "fallback": draw(st.lists(st.sampled_from(names), min_size=1,
                                      max_size=len(names), unique=True)),
            "expected_minutes": draw(cloud_numbers),
            "cap_minutes": draw(cloud_numbers),
        }
        for preset in PRESETS
    }
    return {
        "types": types,
        "presets": presets,
        "smoke": {"expected_minutes": draw(cloud_numbers), "cap_minutes": draw(cloud_numbers),
                  "check_timeout_seconds": draw(cloud_numbers),
                  "scene_gaussians": draw(st.integers(min_value=1, max_value=4096)),
                  "image_pixels": draw(st.integers(min_value=1, max_value=4096)),
                  "video_seconds": draw(cloud_numbers),
                  **{key: draw(st.integers(min_value=1, max_value=4096)) for key in TRAIN_SMOKE},
                  "depth_normal_start_step": draw(st.integers(min_value=0, max_value=4096))},
        "image": {"expected_minutes": draw(cloud_numbers), "cap_minutes": draw(cloud_numbers)},
        "upload_margin_minutes": draw(cloud_numbers),
        "poll_seconds": draw(cloud_numbers),
        "ipv4_usd_per_hour": draw(cloud_numbers),
        "gp3_usd_per_gb_month": draw(cloud_numbers),
        "root_gb": draw(cloud_numbers),
        "smoke_instance_type": draw(st.sampled_from(names)),
        "image_builder_type": draw(st.sampled_from(names)),
        "base_ami_parameter": draw(cloud_text),
    }


def corruptible_targets(mapping: dict[str, Any]) -> list[tuple[tuple[str, ...], str]]:
    """Every single-key corruption target in `mapping`, as (path, category). "whole" is a TOML
    integer (vcpus, the smoke scene's Gaussian count and image size, the training check's seven
    numbers and the depth check's depth_normal_start_step); "table" is a sub-table (types,
    presets, smoke, image)."""
    targets: list[tuple[tuple[str, ...], str]] = [
        (("upload_margin_minutes",), "number"),
        (("poll_seconds",), "number"),
        (("ipv4_usd_per_hour",), "number"),
        (("gp3_usd_per_gb_month",), "number"),
        (("root_gb",), "number"),
        (("smoke_instance_type",), "string"),
        (("image_builder_type",), "string"),
        (("base_ami_parameter",), "string"),
        (("types",), "table"),
        (("presets",), "table"),
        (("smoke",), "table"),
        (("image",), "table"),
    ]
    for name in mapping["types"]:
        targets += [
            (("types", name, "vcpus"), "whole"),
            (("types", name, "memory_gb"), "number"),
            (("types", name, "gpu_memory_gb"), "number"),
            (("types", name, "price_per_hour_usd"), "number"),
            (("types", name, "price_date"), "string"),
        ]
    for preset in mapping["presets"]:
        targets += [
            (("presets", preset, "fallback"), "list"),
            (("presets", preset, "expected_minutes"), "number"),
            (("presets", preset, "cap_minutes"), "number"),
        ]
    for which in ("smoke", "image"):
        targets += [
            ((which, "expected_minutes"), "number"),
            ((which, "cap_minutes"), "number"),
        ]
    targets += [
        (("smoke", "check_timeout_seconds"), "number"),
        (("smoke", "scene_gaussians"), "whole"),
        (("smoke", "image_pixels"), "whole"),
        (("smoke", "video_seconds"), "number"),
    ]
    targets += [(("smoke", key), "whole") for key in TRAIN_SMOKE]
    targets.append((("smoke", "depth_normal_start_step"), "whole"))
    return targets


# How each category of key may be corrupted. A number (or the whole-number vcpus) may be removed or
# given a string, a bool, NaN, infinity or a TOML date (the last three where a number belongs, so
# the Config_Loader catches them, Req 2.1). A string, list or table key is removed or given a
# plainly wrong type (Req 2.2).
CORRUPTION_KINDS = {
    "number": ("remove", "to_string", "to_bool", "to_nan", "to_inf", "to_date"),
    "whole": ("remove", "to_string", "to_bool", "to_nan", "to_inf", "to_date"),
    "string": ("remove", "to_number", "to_bool"),
    "list": ("remove", "to_string", "to_number"),
    "table": ("remove", "to_string", "to_number"),
}

# The wrong value each non-removing corruption puts in place.
WRONG_VALUES = {
    "to_string": "corrupted",
    "to_number": 7,
    "to_bool": True,
    "to_nan": float("nan"),
    "to_inf": float("inf"),
    "to_date": datetime.date(2026, 11, 21),
}


def corrupt(mapping: dict[str, Any], path: tuple[str, ...], kind: str) -> None:
    """Apply one corruption to `mapping` in place at `path`."""
    parent: Any = mapping
    for key in path[:-1]:
        parent = parent[key]
    leaf = path[-1]
    if kind == "remove":
        del parent[leaf]
    else:
        parent[leaf] = WRONG_VALUES[kind]


@st.composite
def corruptions(draw: st.DrawFn) -> tuple[dict[str, Any], dict[str, Any], str]:
    """A valid `[cloud]` mapping, a copy with exactly one key corrupted, and that key's name."""
    valid = draw(valid_cloud_mappings())
    path, category = draw(st.sampled_from(corruptible_targets(valid)))
    kind = draw(st.sampled_from(CORRUPTION_KINDS[category]))
    corrupted = copy.deepcopy(valid)
    corrupt(corrupted, path, kind)
    return valid, corrupted, path[-1]


# A small TOML writer for a [cloud] mapping, because tomllib only reads. It writes a table's scalar
# keys before its sub-tables (as TOML requires), quotes a key that isn't bare (the dotted type
# names), and writes nan, inf and a date as the bare TOML literals the Config_Loader must refuse.

def toml_key(key: str) -> str:
    """`key` as a TOML key: bare when it is all letters, digits, `-` or `_`, else quoted."""
    if key and all(ch.isalnum() or ch in "-_" for ch in key):
        return key
    return '"' + key.replace("\\", "\\\\").replace('"', '\\"') + '"'


def toml_value(value: object) -> str:
    """`value` as a TOML scalar or array literal."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return repr(value)
    if isinstance(value, str):
        return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'
    if isinstance(value, datetime.date):  # a date where a number belongs
        return value.isoformat()
    if isinstance(value, list):
        return "[" + ", ".join(toml_value(item) for item in value) + "]"
    raise TypeError(f"the Property 2 TOML writer can't write {value!r}")


def emit_table(prefix: str, table: dict[str, Any], lines: list[str], root: bool = False) -> None:
    """Append the TOML lines for `table` at header `prefix`, scalars first then sub-tables."""
    scalars = [(key, value) for key, value in table.items() if not isinstance(value, dict)]
    subtables = [(key, value) for key, value in table.items() if isinstance(value, dict)]
    if root or scalars:
        if lines:
            lines.append("")
        lines.append(f"[{prefix}]")
        for key, value in scalars:
            lines.append(f"{toml_key(key)} = {toml_value(value)}")
    for key, value in subtables:
        emit_table(f"{prefix}.{toml_key(key)}", value, lines)


def cloud_toml(mapping: dict[str, Any]) -> str:
    """`mapping` written under a `[cloud]` section as TOML text."""
    lines: list[str] = []
    emit_table("cloud", mapping, lines, root=True)
    return "\n".join(lines) + "\n"


@settings(deadline=None)  # it writes a temp config.toml, so a slow disk can't make it flaky
@given(corruptions())
def test_property_2_cloud_config_refuses_a_corrupted_key_by_name(
        case: tuple[dict[str, Any], dict[str, Any], str]) -> None:
    """Feature: cloud-jobs, Property 2: Cloud_Config refuses a corrupted key by name

    **Validates: Requirements 2.1, 2.2**
    """
    valid, corrupted, key = case
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "config.toml"
        path.write_text(cloud_toml(valid), encoding="utf-8")
        assert isinstance(load_cloud_config(path), CloudConfig)  # the un-corrupted mapping loads
        path.write_text(cloud_toml(corrupted), encoding="utf-8")
        with pytest.raises((PipelineError, CloudError)) as caught:
            load_cloud_config(path)
        assert key in str(caught.value)  # the error names the corrupted key
