"""The Train_Step's settings: the Shared_Settings of [train] and the Preset's three, checked.

`read_settings` reads each key with `settings[key]`, so a missing key raises the Step_Runner's own
Readable_Error, which names [train], [train.<preset>] and the key (Req 9.6). Each kind of value
has one check, which checks type() exactly, so a bool is never a number. A value that fails it is
refused with an ExpectedFailure naming the key, its table and what the value must be (Req 9.7).
Then max_splats must be at least start_scale_neighbours + 1, so each Start_Splat has that many
other splats to take its start scale from. A refusal names both settings, their tables and their
values (Req 9.8). NaN and infinity never get here, because the Config refuses them.

min_vcpus and min_memory_gb are checked and kept as the cloud-only Step_Needs, and the step never
uses them (Req 2.11).

From splatpipe, this module imports only splatpipe.errors.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, NoReturn

from splatpipe.errors import ExpectedFailure


@dataclass(frozen=True)
class TrainSettings:
    """The checked settings of [train] and its Preset table. Each field is named after its key."""

    long_edge: int
    mask_threshold: int
    seed: int
    sh_degree_interval: int
    scene_extent_factor: float
    near_plane_factor: float
    start_opacity: float
    start_scale_factor: float
    start_scale_neighbours: int
    neighbour_chunk_pairs: int  # the most point pairs one 3-NN chunk holds on the GPU
    means_lr: float
    means_lr_final: float
    scales_lr: float
    quats_lr: float
    opacities_lr: float
    sh_dc_lr: float
    sh_rest_lr: float
    adam_eps: float
    exposure_lr: float
    l1_weight: float
    ssim_weight: float
    opacity_reg: float
    scale_reg: float
    erank_reg: float
    erank_epsilon: float
    erank_start_fraction: float
    refine_start_step: int
    refine_every: int
    refine_stop_step: int
    refine_stop_fraction: float
    clean_reserve_minutes: float
    save_margin_minutes: float
    gpu_memory_slack: float
    # The depth priors' settings (depth-priors spec, Req 14.1).
    plain_gradient: float
    depth_gradient_falloff: float
    depth_weight_start: float
    depth_weight_end: float
    depth_weight_floor: float
    depth_error_cap: float
    normal_weight: float
    normal_start_step: int
    normal_every: int
    flatten_reg: float
    depth_normal_weight: float
    normal_consistency_weight: float
    depth_normal_start_fraction: float
    backproject_share: float
    backproject_max: int
    # The cloud-only Step_Needs, checked and kept. The step never uses them (Req 2.11).
    min_vcpus: int
    min_memory_gb: int
    # The Preset_Settings, from [train.<preset>].
    steps: int
    max_splats: int
    min_gpu_memory_gb: int


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
_WHOLE_ZERO_OR_MORE = _Kind("a whole number of 0 or more", lambda v: v >= 0, whole=True)
_WHOLE_ONE_TO_255 = _Kind("a whole number from 1 to 255", lambda v: 1 <= v <= 255, whole=True)
_ABOVE_ZERO = _Kind("a number above 0", lambda v: v > 0, whole=False)
_ZERO_OR_MORE = _Kind("a number of 0 or more", lambda v: v >= 0, whole=False)
_ABOVE_ZERO_BELOW_ONE = _Kind("a number above 0 and below 1", lambda v: 0 < v < 1, whole=False)
_ABOVE_ZERO_AT_MOST_ONE = _Kind("a number above 0 and at most 1", lambda v: 0 < v <= 1,
                                whole=False)
_ZERO_TO_ONE = _Kind("a number from 0 to 1", lambda v: 0 <= v <= 1, whole=False)

# The Shared_Settings of [train], each with its kind (Req 9.7), in TrainSettings' order.
_SHARED: dict[str, _Kind] = {
    "long_edge": _WHOLE_ABOVE_ZERO,
    "mask_threshold": _WHOLE_ONE_TO_255,
    "seed": _WHOLE_ZERO_OR_MORE,
    "sh_degree_interval": _WHOLE_ABOVE_ZERO,
    "scene_extent_factor": _ABOVE_ZERO,
    "near_plane_factor": _ABOVE_ZERO,
    "start_opacity": _ABOVE_ZERO_BELOW_ONE,
    "start_scale_factor": _ABOVE_ZERO,
    "start_scale_neighbours": _WHOLE_ABOVE_ZERO,
    "neighbour_chunk_pairs": _WHOLE_ABOVE_ZERO,
    "means_lr": _ABOVE_ZERO,
    "means_lr_final": _ABOVE_ZERO_AT_MOST_ONE,
    "scales_lr": _ABOVE_ZERO,
    "quats_lr": _ABOVE_ZERO,
    "opacities_lr": _ABOVE_ZERO,
    "sh_dc_lr": _ABOVE_ZERO,
    "sh_rest_lr": _ABOVE_ZERO,
    "adam_eps": _ABOVE_ZERO,
    "exposure_lr": _ABOVE_ZERO,
    "l1_weight": _ZERO_OR_MORE,
    "ssim_weight": _ZERO_OR_MORE,
    "opacity_reg": _ZERO_OR_MORE,
    "scale_reg": _ZERO_OR_MORE,
    "erank_reg": _ZERO_OR_MORE,
    "erank_epsilon": _ABOVE_ZERO,
    "erank_start_fraction": _ZERO_TO_ONE,
    "refine_start_step": _WHOLE_ZERO_OR_MORE,
    "refine_every": _WHOLE_ABOVE_ZERO,
    "refine_stop_step": _WHOLE_ZERO_OR_MORE,
    "refine_stop_fraction": _ZERO_TO_ONE,
    "clean_reserve_minutes": _ZERO_OR_MORE,
    "save_margin_minutes": _ZERO_OR_MORE,
    "gpu_memory_slack": _ABOVE_ZERO_AT_MOST_ONE,
    # The depth priors' settings, with the kinds of the depth-priors spec's Req 14.2. A weight of
    # 0 turns its Prior_Term off, and a backproject_max of 0 the Back_Projected_Splats.
    "plain_gradient": _ABOVE_ZERO,
    "depth_gradient_falloff": _ZERO_OR_MORE,
    "depth_weight_start": _ABOVE_ZERO,
    "depth_weight_end": _ABOVE_ZERO,
    "depth_weight_floor": _ZERO_OR_MORE,
    "depth_error_cap": _ABOVE_ZERO,
    "normal_weight": _ZERO_OR_MORE,
    "normal_start_step": _WHOLE_ZERO_OR_MORE,
    "normal_every": _WHOLE_ABOVE_ZERO,
    "flatten_reg": _ZERO_OR_MORE,
    "depth_normal_weight": _ZERO_OR_MORE,
    "normal_consistency_weight": _ZERO_OR_MORE,
    "depth_normal_start_fraction": _ZERO_TO_ONE,
    "backproject_share": _ZERO_TO_ONE,
    "backproject_max": _WHOLE_ZERO_OR_MORE,
    "min_vcpus": _WHOLE_ABOVE_ZERO,
    "min_memory_gb": _WHOLE_ABOVE_ZERO,
}
# The Preset_Settings, which each Preset table [train.<preset>] holds.
_PER_PRESET: dict[str, _Kind] = {
    "steps": _WHOLE_ABOVE_ZERO,
    "max_splats": _WHOLE_ABOVE_ZERO,
    "min_gpu_memory_gb": _WHOLE_ABOVE_ZERO,
}
_KINDS = _SHARED | _PER_PRESET  # every key, in the order it's checked


def read_settings(settings: Mapping[str, Any], section: str, preset: str) -> TrainSettings:
    """Read and check every key of the Step's settings, and keep them in a TrainSettings.

    `settings` holds the Shared_Settings of [<section>] and the Preset_Settings of
    [<section>.<preset>], as the Step_Runner's StepSettings gives them. `section` and `preset`
    name those tables in the messages. Whole numbers are kept as ints, other numbers as floats.
    """
    tables = {key: _table(key, section, preset) for key in _KINDS}
    values = {key: _checked(settings, key, tables[key], kind) for key, kind in _KINDS.items()}
    if values["max_splats"] < values["start_scale_neighbours"] + 1:
        _refuse_splat_budget(values, tables)
    return TrainSettings(**values)


def _table(key: str, section: str, preset: str) -> str:
    """The table that holds `key`: [<section>.<preset>] for a Preset_Setting, else [<section>]."""
    return f"[{section}.{preset}]" if key in _PER_PRESET else f"[{section}]"


def _checked(settings: Mapping[str, Any], key: str, table: str, kind: _Kind) -> int | float:
    """settings[key] once it's of `kind`: a whole number as its int, a number as a float."""
    value = settings[key]  # a missing key raises the Step_Runner's own message (Req 9.6)
    if not kind.accepts(value):
        raise ExpectedFailure(f"The setting {key} in {table} is {_shown(value)}, but it must be "
                              f"{kind.wanted}. Change it to {kind.wanted}.")
    return value if kind.whole else float(value)


def _refuse_splat_budget(values: Mapping[str, int | float],
                         tables: Mapping[str, str]) -> NoReturn:
    """Refuse max_splats below start_scale_neighbours + 1, naming both settings, their tables and
    their values. Both are whole numbers by now, so they're shown as their ints."""
    fewest = values["start_scale_neighbours"] + 1
    raise ExpectedFailure(f"The setting max_splats in {tables['max_splats']} is "
                          f"{values['max_splats']}, but it must be at least "
                          f"start_scale_neighbours in {tables['start_scale_neighbours']} + 1, "
                          f"which is {fewest}. Raise max_splats or lower start_scale_neighbours.")


def _shown(value: object) -> str:
    """A value about as TOML writes it: true or false for a bool, else Python's repr."""
    if isinstance(value, bool):
        return "true" if value else "false"
    return repr(value)
