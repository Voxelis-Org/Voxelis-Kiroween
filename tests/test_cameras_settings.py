"""Tests for the Cameras_Step's settings: [cameras] in config.toml and
splatpipe/cameras/settings.py.

The real config is read by path, because tests/conftest.py points CONFIG_PATH at a test config.
Its Option_Keys are checked against .kiro/skills/colmap-cli/references/options.md through the
colmap_options oracle, which parses options.md with its own code. The comment check in
tests/test_config.py reads every key of the real config, [cameras] included, so the only
comments checked here are the two refine keys', which must say the keys stay true.

The kinds and ranges of the design's "The settings checks" are written out here, not read from
settings.py. Every key of [cameras] and its Preset tables gets values of other kinds and values
just outside its range, which are refused, and the values at the ends of its range, which pass.
Every test but the real config's reads the test config, whose [cameras] is the real one with
min_placed_frames = 1 and every *_timeout_seconds at 60. The set_cameras fixture changes it for
one test, and conftest's MISSING leaves a key out (Req 12.15).
"""

import math
import re
import sys
import tomllib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import pytest

from conftest import MISSING  # what set_cameras takes to leave a key out of the test config
from splatpipe.cameras.level import LevelSettings
from splatpipe.cameras.settings import CamerasSettings, read_settings
from splatpipe.config import REPO_ROOT, StepSettings, load_config
from splatpipe.errors import ExpectedFailure, PipelineError

# Read by path, so these tests still read it once CONFIG_PATH points at a test config.
REAL_CONFIG = REPO_ROOT / "config.toml"
# The three Presets, written out here rather than taken from the code under test.
PRESET_NAMES = ("fast", "default", "quality")

# Each COLMAP command the step runs, with its Time_Limit key (A4, O1).
TIME_LIMIT_KEYS = {
    "feature_extractor": "feature_extractor_timeout_seconds",
    "sequential_matcher": "sequential_matcher_timeout_seconds",
    "view_graph_calibrator": "view_graph_calibrator_timeout_seconds",
    "global_mapper": "global_mapper_timeout_seconds",
    "mapper": "mapper_timeout_seconds",
    "image_undistorter": "image_undistorter_timeout_seconds",
    "image_undistorter_standalone": "image_undistorter_standalone_timeout_seconds",
}
TIME_LIMIT_COMMANDS = {key: command for command, key in TIME_LIMIT_KEYS.items()}

# The 17 Shared_Settings of the real [cameras] that aren't Option_Keys, with the values of O1 to
# O4 and Req 12.3, each an int or a float as the design writes it.
REAL_OWN = {
    "fisheye_fov_degrees": 120.0,  # O2
    "far_overlap": 10,  # O4
    "min_placed_frames": 20,
    "min_placed_share": 0.5,
    "up_agreement": 0.5,
    "axis_point_min_eigen_share": 0.1,  # O3
    "axis_point_min_front_share": 0.5,  # A5
    "heading_min_degrees": 10.0,  # O3
    "feature_extractor_timeout_seconds": 900,  # the Time_Limits of O1
    "sequential_matcher_timeout_seconds": 1800,
    "view_graph_calibrator_timeout_seconds": 600,
    "global_mapper_timeout_seconds": 1800,
    "mapper_timeout_seconds": 3600,
    "image_undistorter_timeout_seconds": 600,
    "image_undistorter_standalone_timeout_seconds": 120,
    "min_vcpus": 16,
    "min_memory_gb": 16,
}
# The near_overlap in each Preset table of the real [cameras] (Req 12.4). The test config's are
# the same.
NEAR_OVERLAP = {"fast": 15, "default": 15, "quality": 25}
# The four settings of the Levelling, which CamerasSettings keeps in its LevelSettings.
LEVEL_KEYS = ("up_agreement", "heading_min_degrees", "axis_point_min_eigen_share",
              "axis_point_min_front_share")


@dataclass(frozen=True)
class Kind:
    """One kind and range of the design's "The settings checks", written out for these tests.

    `wrong` holds values that read_settings must refuse: values of other kinds first, then values
    of the kind just outside the range. `edges` holds the values at the ends of the range, which
    pass.
    """

    kind: Literal["bool", "whole", "number", "name"]
    wrong: tuple[object, ...]
    edges: tuple[object, ...]


TOP = 2**31 - 1  # the largest 32-bit int, the end of a whole number's range with no top
TOO_BIG = 2**31  # one more, which no whole number setting can hold: COLMAP reads them as 32 bits
TINY = math.nextafter(0.0, 1.0)  # the smallest float above 0
NEGATIVE = math.nextafter(0.0, -1.0)  # the largest float below 0
BIGGEST = sys.float_info.max  # the largest finite float, the end of a number's range with no top


def above(bound: float) -> float:
    """The smallest float above `bound`."""
    return math.nextafter(bound, math.inf)


def below(bound: float) -> float:
    """The largest float below `bound`."""
    return math.nextafter(bound, -math.inf)


# A TOML bool, which refuses an int and text.
BOOL = Kind("bool", wrong=(1, "true"), edges=(True, False))
# A TOML bool that is true. It refuses an int and text, then false, the one bool outside its
# range (Req 12.16).
TRUE_ONLY = Kind("bool", wrong=(1, "true", False), edges=(True,))
# The whole numbers. Each refuses a bool, a float and text, then the values just outside its
# range, and 2147483648, which doesn't fit a 32-bit int.
WHOLE_MINUS_1_OR_ABOVE_0 = Kind("whole", wrong=(True, 2.0, "2", -2, 0, TOO_BIG),
                                edges=(-1, 1, TOP))
WHOLE_MINUS_1_OR_MORE = Kind("whole", wrong=(True, 2.0, "2", -2, TOO_BIG), edges=(-1, TOP))
WHOLE_0_OR_MORE = Kind("whole", wrong=(True, 2.0, "2", -1, TOO_BIG), edges=(0, TOP))
WHOLE_ABOVE_0 = Kind("whole", wrong=(True, 2.0, "2", 0, TOO_BIG), edges=(1, TOP))
WHOLE_2_OR_MORE = Kind("whole", wrong=(True, 2.0, "2", 1, TOO_BIG), edges=(2, TOP))
WHOLE_MINUS_1_OR_1_TO_100 = Kind("whole", wrong=(True, 2.0, "2", -2, 0, 101, TOO_BIG),
                                 edges=(-1, 1, 100))
# The numbers, an int or a float. Each refuses a bool and text, then the values just outside its
# range. An end that's a whole number is given as an int, which a decimal option takes too.
ABOVE_0 = Kind("number", wrong=(True, "0.5", 0), edges=(TINY, BIGGEST))
ZERO_OR_MORE = Kind("number", wrong=(True, "0.5", NEGATIVE), edges=(0, BIGGEST))
ZERO_TO_1 = Kind("number", wrong=(True, "0.5", NEGATIVE, above(1.0)), edges=(0, 1))
ABOVE_0_AT_MOST_1 = Kind("number", wrong=(True, "0.5", 0, above(1.0)), edges=(TINY, 1))
DEGREES_0_TO_180 = Kind("number", wrong=(True, "0.5", NEGATIVE, above(180.0)), edges=(0, 180))
# Its bool is false, because true counts as 1, which lies outside this range anyway.
ZERO_BELOW_1 = Kind("number", wrong=(False, "0.5", NEGATIVE, 1), edges=(0, below(1.0)))
ABOVE_0_AT_MOST_360 = Kind("number", wrong=(True, "0.5", 0, above(360.0)), edges=(TINY, 360))
ABOVE_0_BELOW_90 = Kind("number", wrong=(True, "0.5", 0, 90), edges=(TINY, below(90.0)))


def names(*known: str, unknown: str) -> Kind:
    """A choice option: text that's one of `known`. An int and the name `unknown` are refused."""
    return Kind("name", wrong=(1, unknown), edges=known)


# The two keys that must stay true, each with its COLMAP option (Req 12.16, D20).
TRUE_ONLY_KEYS = {
    "global_mapper_ba_refine_extra_params": "--GlobalMapper.ba_refine_extra_params",
    "mapper_ba_refine_extra_params": "--Mapper.ba_refine_extra_params",
}
# The reason of Req 12.16 that their refusal gives, as the design writes it.
REFINE_REASON = ("Bundle adjustment has to refine the distortion: with it off, the distortion "
                 "stays 0, COLMAP's undistorter keeps the camera's own model and copies the "
                 "picks unwarped, and the step can't hand the train step a PINHOLE camera.")

# The design's table of the Option_Keys, row by row: each kind and range with its keys. The
# bools are the keys whose value in the design's [cameras] block is true or false, but the two
# that must stay true.
OPTION_ROWS: tuple[tuple[Kind, tuple[str, ...]], ...] = (
    (BOOL, (
        "sift_extraction_estimate_affine_shape", "sift_extraction_upright",
        "sift_extraction_domain_size_pooling", "feature_matching_guided_matching",
        "feature_matching_skip_geometric_verification", "feature_matching_rig_verification",
        "feature_matching_skip_image_pairs_in_same_frame", "sift_matching_cross_check",
        "sift_matching_cpu_brute_force_matcher", "two_view_geometry_multiple_models",
        "two_view_geometry_compute_relative_pose", "two_view_geometry_detect_watermark",
        "two_view_geometry_multiple_ignore_watermark",
        "two_view_geometry_filter_stationary_matches", "two_view_geometry_use_degensac",
        "two_view_geometry_use_sampson_refinement", "sequential_matching_expand_rig_images",
        "view_graph_calibrator_cross_validate_prior_focal_lengths",
        "view_graph_calibrator_reestimate_relative_pose", "global_mapper_ignore_watermarks",
        "global_mapper_decompose_relative_pose", "global_mapper_multiple_models",
        "global_mapper_skip_rotation_averaging", "global_mapper_skip_track_establishment",
        "global_mapper_skip_global_positioning", "global_mapper_skip_bundle_adjustment",
        "global_mapper_skip_retriangulation", "global_mapper_gp_optimize_positions",
        "global_mapper_gp_optimize_points", "global_mapper_gp_optimize_scales",
        "global_mapper_ba_refine_focal_length", "global_mapper_ba_refine_principal_point",
        "global_mapper_refine_sensor_from_rig", "global_mapper_ba_refine_rig_from_world",
        "global_mapper_ba_refine_points3d",
        "global_mapper_ba_skip_fixed_rotation_stage",
        "global_mapper_ba_skip_joint_optimization_stage", "global_mapper_ra_use_gravity",
        "global_mapper_ra_use_stratified", "mapper_ignore_watermarks", "mapper_multiple_models",
        "mapper_structure_less_registration_fallback", "mapper_structure_less_registration_only",
        "mapper_extract_colors", "mapper_ba_refine_focal_length",
        "mapper_ba_refine_principal_point", "mapper_ba_refine_sensor_from_rig",
        "mapper_fix_existing_frames", "mapper_ba_global_ignore_redundant_points3d",
        "mapper_tri_ignore_two_view_tracks",
    )),
    (TRUE_ONLY, tuple(TRUE_ONLY_KEYS)),
    (WHOLE_MINUS_1_OR_ABOVE_0, (
        "feature_extraction_num_threads", "feature_matching_num_threads",
        "global_mapper_num_threads", "mapper_num_threads", "undistortion_num_threads",
        "image_reader_existing_camera_id", "feature_extraction_max_image_size",
        "undistortion_max_image_size", "mapper_init_image_id1", "mapper_init_image_id2",
        "mapper_ba_local_max_num_iterations", "mapper_ba_global_max_num_iterations",
        "mapper_max_runtime_seconds",
    )),
    (WHOLE_MINUS_1_OR_MORE, (
        "sift_extraction_first_octave", "two_view_geometry_random_seed",
        "global_mapper_random_seed", "mapper_random_seed",
    )),
    (WHOLE_0_OR_MORE, (
        "default_random_seed", "two_view_geometry_min_num_inliers",
        "view_graph_calibrator_relpose_min_num_inliers", "global_mapper_min_num_matches",
        "global_mapper_min_model_size", "global_mapper_ba_num_iterations",
        "global_mapper_ba_min_track_length", "mapper_min_num_matches", "mapper_max_model_overlap",
        "mapper_min_model_size", "mapper_ba_min_num_residuals_for_cpu_multi_threading",
        "mapper_snapshot_frames_freq", "mapper_init_min_num_inliers",
        "mapper_abs_pose_min_num_inliers", "mapper_tri_max_transitivity",
        "mapper_tri_complete_max_transitivity", "mapper_tri_re_max_trials",
    )),
    (WHOLE_ABOVE_0, (
        "sift_extraction_max_num_features", "sift_extraction_num_octaves",
        "sift_extraction_octave_resolution", "sift_extraction_max_num_orientations",
        "sift_extraction_dsp_num_scales", "feature_matching_max_num_matches",
        "two_view_geometry_max_num_trials", "global_mapper_track_required_tracks_per_view",
        "global_mapper_keep_max_num_tracks", "global_mapper_gp_max_num_iterations",
        "global_mapper_ba_ceres_max_num_iterations", "mapper_max_num_models",
        "mapper_init_num_trials", "mapper_ba_global_frames_freq", "mapper_ba_global_points_freq",
        "mapper_ba_global_max_refinements", "mapper_ba_local_max_refinements",
        "mapper_init_max_reg_trials", "mapper_max_reg_trials", "mapper_ba_local_num_images",
    )),
    (WHOLE_2_OR_MORE, (
        "global_mapper_track_min_num_views_per_track",
        "global_mapper_track_max_num_views_per_track",
    )),
    (WHOLE_MINUS_1_OR_1_TO_100, ("undistortion_jpeg_quality",)),
    (ABOVE_0, (
        "image_reader_default_focal_length_factor", "sift_extraction_peak_threshold",
        "sift_extraction_edge_threshold", "sift_extraction_dsp_min_scale",
        "sift_extraction_dsp_max_scale", "sift_matching_max_distance",
        "two_view_geometry_watermark_detection_max_error",
        "two_view_geometry_stationary_matches_max_error", "two_view_geometry_max_error",
        "view_graph_calibrator_min_focal_length_ratio",
        "view_graph_calibrator_max_focal_length_ratio",
        "view_graph_calibrator_max_calibration_error", "view_graph_calibrator_relpose_max_error",
        "global_mapper_track_intra_image_consistency_threshold",
        "global_mapper_gp_loss_function_scale", "global_mapper_ba_ceres_loss_function_scale",
        "global_mapper_tri_complete_max_reproj_error", "global_mapper_tri_merge_max_reproj_error",
        "global_mapper_max_angular_reproj_error_deg", "global_mapper_max_normalized_reproj_error",
        "mapper_min_focal_length_ratio", "mapper_max_focal_length_ratio", "mapper_max_extra_param",
        "mapper_ba_global_frames_ratio", "mapper_ba_global_points_ratio", "mapper_init_max_error",
        "mapper_abs_pose_max_error", "mapper_filter_max_reproj_error",
        "mapper_tri_create_max_angle_error", "mapper_tri_continue_max_angle_error",
        "mapper_tri_merge_max_reproj_error", "mapper_tri_complete_max_reproj_error",
        "mapper_tri_re_max_angle_error", "undistortion_min_scale", "undistortion_max_scale",
    )),
    (ZERO_OR_MORE, (
        "mapper_ba_local_function_tolerance", "mapper_ba_global_function_tolerance",
        "mapper_ba_global_max_refinement_change", "mapper_ba_local_max_refinement_change",
    )),
    (ZERO_TO_1, (
        "two_view_geometry_min_inlier_ratio", "view_graph_calibrator_min_calibrated_pair_ratio",
        "view_graph_calibrator_relpose_min_inlier_ratio", "mapper_init_max_forward_motion",
        "mapper_abs_pose_min_inlier_ratio",
        "mapper_ba_global_ignore_redundant_points3d_min_coverage_gain", "mapper_tri_re_min_ratio",
        "undistortion_blank_pixels",
    )),
    (ABOVE_0_AT_MOST_1, ("sift_matching_max_ratio", "two_view_geometry_confidence")),
    (DEGREES_0_TO_180, (
        "global_mapper_tri_min_angle", "global_mapper_ra_max_rotation_error_deg",
        "global_mapper_min_tri_angle_deg", "mapper_init_min_tri_angle",
        "mapper_filter_min_tri_angle", "mapper_ba_local_min_tri_angle", "mapper_tri_min_angle",
    )),
    (ZERO_BELOW_1, ("undistortion_roi_min_x", "undistortion_roi_min_y")),
    (ABOVE_0_AT_MOST_1, ("undistortion_roi_max_x", "undistortion_roi_max_y")),
    # OPENCV, the first draft's Camera_Model, is now an unknown name (D19).
    (names("RADIAL", "RADIAL_FISHEYE", unknown="OPENCV"), ("camera_model",)),
    (names("l1_root", "l2", unknown="l1"), ("feature_extractor_descriptor_normalization",)),
    (names("UNIFORM", "INLIER_MATCH_COUNT", unknown="INLIER_COUNT"),
     ("global_mapper_ra_reweighting",)),
    (names("COPY", "SOFT_LINK", "HARD_LINK", unknown="MOVE"), ("undistortion_copy_policy",)),
)
OPTION_RANGES = {key: kind for kind, keys in OPTION_ROWS for key in keys}

# The design's table of the other settings (Req 12.10), near_overlap included.
OWN_RANGES = {
    "fisheye_fov_degrees": ABOVE_0_AT_MOST_360,
    "far_overlap": WHOLE_ABOVE_0,
    "min_placed_frames": WHOLE_ABOVE_0,
    "min_placed_share": ZERO_TO_1,
    "up_agreement": ABOVE_0_AT_MOST_1,
    "axis_point_min_eigen_share": ABOVE_0_AT_MOST_1,
    "axis_point_min_front_share": ZERO_TO_1,
    "heading_min_degrees": ABOVE_0_BELOW_90,
    **dict.fromkeys(TIME_LIMIT_KEYS.values(), ABOVE_0),
    "min_vcpus": WHOLE_ABOVE_0,
    "min_memory_gb": WHOLE_ABOVE_0,
    "near_overlap": WHOLE_ABOVE_0,
}
RANGES = {**OWN_RANGES, **OPTION_RANGES}  # every key of [cameras] and its Preset tables

# The TOML type that a real Option_Key of each kind holds. A decimal option holds a float, also
# where options.md writes a whole number, so config.toml shows which options take decimals (C9).
OPTION_TYPES = {"bool": bool, "whole": int, "number": float, "name": str}

# The cross checks whose first setting must be no greater than the second. Both settings of each
# pair have the same range, so an edge value given to both passes the pair.
NO_GREATER = (
    ("sift_extraction_dsp_min_scale", "sift_extraction_dsp_max_scale"),
    ("view_graph_calibrator_min_focal_length_ratio",
     "view_graph_calibrator_max_focal_length_ratio"),
    ("mapper_min_focal_length_ratio", "mapper_max_focal_length_ratio"),
    ("global_mapper_track_min_num_views_per_track", "global_mapper_track_max_num_views_per_track"),
    ("undistortion_min_scale", "undistortion_max_scale"),
)
PARTNERS = {key: other for pair in NO_GREATER for key, other in (pair, pair[::-1])}


def typed(table: Mapping[str, Any]) -> dict[str, tuple[type, Any]]:
    """Each value of `table` with its type, so that 10 and 10.0, or 1 and true, don't compare
    equal."""
    return {key: (type(value), value) for key, value in table.items()}


def real_shared() -> dict[str, Any]:
    """The Shared_Settings of the real [cameras], each as tomllib gives it."""
    section = tomllib.loads(REAL_CONFIG.read_text(encoding="utf-8"))["cameras"]
    return {key: value for key, value in section.items() if not isinstance(value, dict)}


def option_names(colmap_options: Mapping[str, Sequence[Any]]) -> dict[str, str]:
    """Each Option_Key that the oracle finds in options.md, with the COLMAP option it sets."""
    return {option.key: option.option for options in colmap_options.values()
            for option in options if option.sort == "key"}


def table_of(key: str, preset: str) -> str:
    """The table that holds `key`: [cameras.<preset>] for near_overlap, else [cameras]."""
    return f"[cameras.{preset}]" if key == "near_overlap" else "[cameras]"


def cameras_settings(preset: str = "fast", **changes: Any) -> StepSettings:
    """The test config's [cameras] settings for `preset`, as the Step_Runner gives them to the
    step, with each key in `changes` set to its value."""
    settings = load_config().step_settings("cameras", preset)
    return StepSettings({**settings, **changes}, section="cameras", preset=preset)


def read(settings: StepSettings) -> CamerasSettings:
    """read_settings on `settings`, with the section and the Preset they're for, as the step
    calls it."""
    return read_settings(settings, settings.section, settings.preset)


def refusal(settings: StepSettings) -> str:
    """The message of the ExpectedFailure that read_settings raises on `settings`."""
    with pytest.raises(ExpectedFailure) as caught:
        read(settings)
    return str(caught.value)


def kept_value(cameras: CamerasSettings, key: str) -> object:
    """The value that `cameras` keeps for the setting `key`: in `options` for an Option_Key, in
    `level` for a setting of the Levelling, in `timeouts` under its command for a Time_Limit,
    and in the field of the key's name for the rest."""
    if key in OPTION_RANGES:
        return cameras.options[key]
    if key in LEVEL_KEYS:
        return getattr(cameras.level, key)
    if key in TIME_LIMIT_COMMANDS:
        return cameras.timeouts[TIME_LIMIT_COMMANDS[key]]
    return getattr(cameras, key)


def same_value(value: object, text: str) -> bool:
    """Whether `value`, from config.toml, is the value that options.md writes as `text`: a bool
    as 1 or 0, a whole number as the same int, a decimal as the same float, and a name as the
    same text."""
    if type(value) is bool:
        return text == ("1" if value else "0")
    if type(value) is int:
        return re.fullmatch(r"-?[0-9]+", text) is not None and int(text) == value
    if type(value) is float:
        try:
            return float(text) == value
        except ValueError:
            return False
    return value == text


# The real config.toml

def test_the_real_cameras_section_has_exactly_the_designs_190_keys_and_near_overlap(
        colmap_options: Mapping[str, Sequence[Any]]) -> None:
    section = tomllib.loads(REAL_CONFIG.read_text(encoding="utf-8"))["cameras"]
    tables = {key: value for key, value in section.items() if isinstance(value, dict)}
    shared = {key: value for key, value in section.items() if key not in tables}
    # The rows here hold each Option_Key of options.md once: 173, camera_model included.
    assert set(OPTION_RANGES) == set(option_names(colmap_options))
    assert sum(len(keys) for _, keys in OPTION_ROWS) == len(OPTION_RANGES) == 173
    assert set(shared) == set(OPTION_RANGES) | set(REAL_OWN)  # Req 12.1 to 12.3
    assert len(shared) == 190
    assert set(RANGES) == set(shared) | {"near_overlap"}  # so the checks below cover every key
    assert {name: typed(table) for name, table in tables.items()} == {
        preset: typed({"near_overlap": value}) for preset, value in NEAR_OVERLAP.items()}  # 12.4


def test_each_real_option_key_holds_its_value_in_options_md_with_its_kinds_type(
        colmap_options: Mapping[str, Sequence[Any]]) -> None:
    shared = real_shared()
    # Every block that takes an Option_Key gives the same value: --default_random_seed is in
    # all seven blocks, and each undistortion setting in both undistorters' (Req 12.2).
    differing = [(command, option.option, option.text, shared[option.key])
                 for command, options in colmap_options.items() for option in options
                 if option.sort == "key" and option.key != "camera_model"
                 and not same_value(shared[option.key], option.text)]
    assert differing == []
    assert {key: type(shared[key]) for key in OPTION_RANGES} == {
        key: OPTION_TYPES[kind.kind] for key, kind in OPTION_RANGES.items()}  # C9


def test_the_real_cameras_section_holds_the_designs_values_for_the_other_settings() -> None:
    shared = real_shared()
    assert typed({key: shared[key] for key in REAL_OWN}) == typed(REAL_OWN)  # O1 to O4, Req 12.3
    assert typed({"camera_model": shared["camera_model"]}) == typed({"camera_model": "RADIAL"})
    assert shared["far_overlap"] >= 9  # so the Far_Match_Run reaches 256 picks ahead (Req 4.4)
    assert shared["global_mapper_multiple_models"] is True  # Req 12.5
    assert shared["mapper_multiple_models"] is True
    assert shared["undistortion_blank_pixels"] == 0  # Req 8.6


@pytest.mark.parametrize(("key", "option"), list(TRUE_ONLY_KEYS.items()))
def test_the_real_config_holds_true_in_each_refine_key_with_a_comment_saying_it_must_stay_true(
        key: str, option: str) -> None:
    assert real_shared()[key] is True
    lines = REAL_CONFIG.read_text(encoding="utf-8").splitlines()
    at = [number for number, line in enumerate(lines) if line.partition("=")[0].strip() == key]
    assert len(at) == 1
    comment = lines[at[0] - 1]  # the comment right above the key
    assert comment.startswith(f"# {option}: ")
    assert "Must be true" in comment


@pytest.mark.parametrize("preset", PRESET_NAMES)
def test_the_real_cameras_section_loads_for_each_preset(preset: str) -> None:
    settings = load_config(REAL_CONFIG).step_settings("cameras", preset)
    assert typed(settings) == typed({**real_shared(), "near_overlap": NEAR_OVERLAP[preset]})
    assert read(settings).near_overlap == NEAR_OVERLAP[preset]


# The checks of read_settings

WRONG = [pytest.param(key, value, id=f"{key} {value!r}")
         for key, kind in RANGES.items() for value in kind.wrong]


@pytest.mark.parametrize(("key", "value"), WRONG)
def test_a_wrong_value_is_refused_naming_the_key_its_table_and_its_colmap_option(
        colmap_options: Mapping[str, Sequence[Any]], key: str, value: object) -> None:
    # quality, so the Preset table named is the chosen one, not the test config's default_preset.
    message = refusal(cameras_settings("quality", **{key: value}))
    assert f"The setting {key} in {table_of(key, 'quality')}" in message
    if key in OPTION_RANGES:
        assert f"(COLMAP's {option_names(colmap_options)[key]})" in message


def test_a_refusal_reads_as_the_designs_example() -> None:
    assert refusal(cameras_settings(sift_extraction_max_num_features=0)) == (
        "The setting sift_extraction_max_num_features in [cameras] (COLMAP's "
        "--SiftExtraction.max_num_features) is 0, but it must be a whole number above 0. Change "
        "it to a whole number above 0.")


@pytest.mark.parametrize(("key", "option"), list(TRUE_ONLY_KEYS.items()))
def test_a_refine_key_set_to_false_is_refused_with_its_reason_and_true_passes(
        key: str, option: str) -> None:
    message = refusal(cameras_settings(**{key: False}))
    assert message == (f"The setting {key} in [cameras] (COLMAP's {option}) is false, but it "
                       f"must be true. Change it to true. {REFINE_REASON}")  # Req 12.16
    assert "Traceback" not in message  # Req 16.9
    assert read(cameras_settings(**{key: True})).options[key] is True


EDGES = [pytest.param(key, value, id=f"{key} {value!r}")
         for key, kind in RANGES.items() for value in kind.edges]


@pytest.mark.parametrize(("key", "value"), EDGES)
def test_a_value_at_an_end_of_its_range_passes_and_is_kept_where_it_belongs(
        key: str, value: object) -> None:
    # The other setting of a cross check takes the same value, so the pair still passes.
    changes = {key: value} | ({PARTNERS[key]: value} if key in PARTNERS else {})
    cameras = read(cameras_settings("quality", **changes))
    kept = {name: kept_value(cameras, name) for name in changes}
    assert kept == changes
    # An Option_Key keeps its TOML type, so an int given to a decimal option stays an int.
    options = changes.keys() & OPTION_RANGES.keys()
    assert typed({name: kept[name] for name in options}) == typed(
        {name: changes[name] for name in options})


# Each cross check of "The settings checks", broken by values that each pass on their own, with
# the words that name each of its two settings with its value. The crop's check has a pair for
# each direction, and equal edges aren't a crop.
CROSS_CHECKS = [
    pytest.param({"sift_extraction_dsp_min_scale": 3.5, "sift_extraction_dsp_max_scale": 3.25},
                 ("sift_extraction_dsp_min_scale in [cameras] is 3.5",
                  "sift_extraction_dsp_max_scale in [cameras], which is 3.25"),
                 id="DSP-SIFT scales"),
    pytest.param({"view_graph_calibrator_min_focal_length_ratio": 2.5,
                  "view_graph_calibrator_max_focal_length_ratio": 2.0},
                 ("view_graph_calibrator_min_focal_length_ratio in [cameras] is 2.5",
                  "view_graph_calibrator_max_focal_length_ratio in [cameras], which is 2.0"),
                 id="calibration focal length ratios"),
    pytest.param({"mapper_min_focal_length_ratio": 2.5, "mapper_max_focal_length_ratio": 2.0},
                 ("mapper_min_focal_length_ratio in [cameras] is 2.5",
                  "mapper_max_focal_length_ratio in [cameras], which is 2.0"),
                 id="mapper focal length ratios"),
    pytest.param({"global_mapper_track_min_num_views_per_track": 101,
                  "global_mapper_track_max_num_views_per_track": 100},
                 ("global_mapper_track_min_num_views_per_track in [cameras] is 101",
                  "global_mapper_track_max_num_views_per_track in [cameras], which is 100"),
                 id="views per track"),
    pytest.param({"global_mapper_ba_skip_fixed_rotation_stage": True,
                  "global_mapper_ba_skip_joint_optimization_stage": True},
                 ("global_mapper_ba_skip_fixed_rotation_stage in [cameras] is true",
                  "global_mapper_ba_skip_joint_optimization_stage in [cameras] is true"),
                 id="both bundle adjustment stages skipped"),
    pytest.param({"undistortion_min_scale": 2.5, "undistortion_max_scale": 2.0},
                 ("undistortion_min_scale in [cameras] is 2.5",
                  "undistortion_max_scale in [cameras], which is 2.0"),
                 id="undistortion scales"),
    pytest.param({"undistortion_roi_min_x": 0.5, "undistortion_roi_max_x": 0.5},
                 ("undistortion_roi_min_x in [cameras] is 0.5",
                  "undistortion_roi_max_x in [cameras], which is 0.5"),
                 id="crop across"),
    pytest.param({"undistortion_roi_min_y": 0.75, "undistortion_roi_max_y": 0.5},
                 ("undistortion_roi_min_y in [cameras] is 0.75",
                  "undistortion_roi_max_y in [cameras], which is 0.5"),
                 id="crop down"),
]


@pytest.mark.parametrize(("changes", "named"), CROSS_CHECKS)
def test_a_cross_check_is_refused_naming_both_settings_and_their_values(
        changes: dict[str, Any], named: tuple[str, str]) -> None:
    message = refusal(cameras_settings(**changes))
    for words in named:
        assert words in message


@pytest.mark.parametrize("key", list(RANGES))
def test_a_missing_key_gives_the_step_runners_message_naming_both_tables_and_the_key(
        key: str) -> None:
    settings = cameras_settings("quality")
    del settings[key]
    with pytest.raises(PipelineError) as caught:
        read(settings)
    with pytest.raises(PipelineError) as runners:
        settings[key]  # the Step_Runner's own message for the missing key (Req 12.7)
    message = str(caught.value)
    assert message == str(runners.value)
    for words in ("[cameras]", "[cameras.quality]", key):
        assert words in message


# The test config and set_cameras

@pytest.mark.parametrize("preset", PRESET_NAMES)
def test_read_settings_on_the_test_config_gives_the_presets_near_overlap_and_every_setting(
        preset: str) -> None:
    cameras = read(load_config().step_settings("cameras", preset))
    assert type(cameras) is CamerasSettings
    assert cameras.near_overlap == NEAR_OVERLAP[preset]
    real = real_shared()  # the test config's Option_Keys are the real config's (C5)
    assert typed(cameras.options) == typed({key: real[key] for key in OPTION_RANGES})
    assert dict(cameras.timeouts) == dict.fromkeys(TIME_LIMIT_KEYS, 60)  # by command
    assert cameras.level == LevelSettings(up_agreement=0.5, heading_min_degrees=10.0,
                                          axis_point_min_eigen_share=0.1,
                                          axis_point_min_front_share=0.5)
    assert (cameras.camera_model, cameras.fisheye_fov_degrees, cameras.far_overlap,
            cameras.min_placed_frames, cameras.min_placed_share, cameras.min_vcpus,
            cameras.min_memory_gb) == ("RADIAL", 120.0, 10, 1, 0.5, 16, 16)


# What set_cameras can change: a Shared_Setting, which every Preset reads, or near_overlap for
# one Preset, which only that Preset reads. Each with the change each Preset then sees.
CAMERAS_CHANGES = [
    pytest.param({"min_placed_frames": 5},
                 {preset: {"min_placed_frames": 5} for preset in PRESET_NAMES},
                 id="a shared setting"),
    pytest.param({"near_overlap": {"quality": 3}},
                 {"fast": {}, "default": {}, "quality": {"near_overlap": 3}},
                 id="near_overlap for one preset"),
]


def presets_settings() -> dict[str, StepSettings]:
    """The test config's [cameras] settings for each Preset, as it is now."""
    return {preset: load_config().step_settings("cameras", preset) for preset in PRESET_NAMES}


@pytest.mark.parametrize(("values", "seen"), CAMERAS_CHANGES)
def test_set_cameras_changes_only_the_key_it_names(
        set_cameras: Callable[..., None], values: dict[str, Any],
        seen: dict[str, dict[str, Any]]) -> None:
    before = presets_settings()
    set_cameras(**values)
    assert {preset: typed(settings) for preset, settings in presets_settings().items()} == {
        preset: typed({**before[preset], **seen[preset]}) for preset in PRESET_NAMES}


@pytest.mark.parametrize("key", ["mapper_num_threads", "near_overlap"])
def test_set_cameras_with_missing_leaves_the_key_out_for_every_preset(
        set_cameras: Callable[..., None], key: str) -> None:
    before = presets_settings()
    set_cameras(**{key: MISSING})
    assert {preset: typed(settings) for preset, settings in presets_settings().items()} == {
        preset: typed({name: value for name, value in settings.items() if name != key})
        for preset, settings in before.items()}


def test_the_next_test_sees_the_test_configs_values_again() -> None:
    """pytest runs a file's tests in order, so this runs after the set_cameras tests above."""
    for preset, settings in presets_settings().items():
        assert (settings["min_placed_frames"], settings["mapper_num_threads"],
                settings["near_overlap"]) == (1, -1, NEAR_OVERLAP[preset])
