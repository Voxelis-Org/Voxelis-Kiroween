"""The Cameras_Step's settings: [cameras] and the Preset's near_overlap, every one checked.

[cameras] holds one Option_Key for each COLMAP option the step takes from Config, and the step's
own settings. OPTION_KEYS lists the Option_Keys in the order of references/options.md in the
colmap-cli skill, each with its COLMAP option, the COLMAP commands that take it and its Range: the
kind of TOML value it must be, and the range it must lie in. TIMEOUT_KEYS gives each COLMAP
command the step runs its Time_Limit key.

`read_settings` reads each key with `settings[key]`, so a missing key raises the Step_Runner's own
Readable_Error, which names [cameras], [cameras.<preset>] and the key (Req 12.7). Each kind of
value has one check, which checks type() exactly, so a bool is never a number, and a whole number
must also fit the 32-bit int that COLMAP reads it into. Then the Range's own test runs. A value
that fails either is refused with an ExpectedFailure that names the key, its table, COLMAP's
option for an Option_Key, and what the value must be (Req 12.8 to 12.10). Once every key has
passed on its own, the cross checks run, and a refusal names both settings and their values. NaN
and infinity never get here, because the Config refuses them.

Every value is kept as Config gave it, so an Option_Key's int stays an int and its float a float,
and the command lines write each value the way it was given (Req 2.4). min_vcpus and
min_memory_gb are checked and kept for the cloud-jobs spec's Start_Command, and the step never
uses them (Req 12.11).

From splatpipe, this module imports only splatpipe.errors and the step's level.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal, NoReturn

from splatpipe.cameras.level import LevelSettings
from splatpipe.errors import ExpectedFailure

# The value of an Option_Key, as Config gives it: a TOML bool, integer, float or string.
OptionValue = bool | int | float | str

# The bounds of the C++ int that COLMAP 4.2.1 reads each whole-number option into. Every whole
# number in [cameras] must lie within them.
_INT_MIN = -2**31
_INT_MAX = 2**31 - 1


@dataclass(frozen=True)
class Range:
    """What a setting must be: one kind of TOML value, and a range within that kind."""

    kind: Literal["bool", "whole", "number", "name"]
    wanted: str  # for the messages, like "a whole number above 0"
    test: Callable[[Any], bool]  # the range, on a value already of the kind

    def accepts(self, value: object) -> bool:
        """Whether `value` is of the kind and passes the test.

        type() is checked exactly, so a bool is never a number. "bool" takes a TOML bool,
        "whole" a TOML integer that fits a 32-bit int, "number" a TOML integer or float, and
        "name" a TOML string, which the test checks against the Range's names.
        """
        if self.kind == "bool":
            of_kind = type(value) is bool
        elif self.kind == "whole":
            of_kind = type(value) is int and _INT_MIN <= value <= _INT_MAX
        elif self.kind == "number":
            of_kind = type(value) in (int, float)
        else:  # a name
            of_kind = type(value) is str
        return of_kind and self.test(value)


@dataclass(frozen=True)
class OptionKey:
    """One Option_Key: its key, COLMAP's option, the commands that take it, and its Range."""

    key: str  # the key in [cameras], like "sift_extraction_max_num_features"
    option: str  # COLMAP's option, like "--SiftExtraction.max_num_features"
    commands: tuple[str, ...]  # the COLMAP commands that take it
    range: Range


@dataclass(frozen=True)
class CamerasSettings:
    """The checked settings of [cameras] and its Preset table, each value as Config gave it."""

    camera_model: str  # the Camera_Model, "OPENCV" or "OPENCV_FISHEYE", also in options (D5)
    fisheye_fov_degrees: float  # the field of view the Fisheye_Start works from (O2)
    near_overlap: int  # the Preset_Setting, from [cameras.<preset>], for the Near_Match_Run
    far_overlap: int  # --SequentialMatching.overlap of the Far_Match_Run (O4)
    min_placed_frames: int  # the Placed_Check's fewest Placed_Frames (D9)
    min_placed_share: float  # the Placed_Check's smallest share of the Picks (D9)
    level: LevelSettings  # up_agreement, heading_min_degrees, the two axis point shares
    timeouts: Mapping[str, float]  # each COLMAP command's Time_Limit in seconds, by command
    # The Step_Needs, kept for the cloud-jobs spec's Start_Command. The step never uses them.
    min_vcpus: int
    min_memory_gb: int
    options: Mapping[str, OptionValue]  # every Option_Key's value, by key, camera_model included


def _names(*names: str) -> Range:
    """The Range of a choice option: a TOML string that's one of two or more `names`, each
    written as COLMAP takes it."""
    quoted = [f'"{name}"' for name in names]
    return Range("name", f"{', '.join(quoted[:-1])} or {quoted[-1]}", lambda v: v in names)


def _keys(commands: tuple[str, ...], rows: tuple[tuple[str, str, Range], ...]
          ) -> tuple[OptionKey, ...]:
    """One OptionKey for each row (key, option, range), each taken by `commands`."""
    return tuple(OptionKey(key, option, commands, rng) for key, option, rng in rows)


# The Ranges of "The settings checks" in the design, each with the words its refusal uses.
_BOOL = Range("bool", "true or false", lambda v: True)
_WHOLE_MINUS_ONE_OR_ABOVE_ZERO = Range("whole", "a whole number: -1, or above 0",
                                       lambda v: v == -1 or v > 0)
_WHOLE_MINUS_ONE_OR_MORE = Range("whole", "a whole number of -1 or more", lambda v: v >= -1)
_WHOLE_ZERO_OR_MORE = Range("whole", "a whole number of 0 or more", lambda v: v >= 0)
_WHOLE_ABOVE_ZERO = Range("whole", "a whole number above 0", lambda v: v > 0)
_WHOLE_TWO_OR_MORE = Range("whole", "a whole number of 2 or more", lambda v: v >= 2)
_WHOLE_MINUS_ONE_OR_ONE_TO_100 = Range("whole", "a whole number: -1, or from 1 to 100",
                                       lambda v: v == -1 or 1 <= v <= 100)
_ABOVE_ZERO = Range("number", "a number above 0", lambda v: v > 0)
_ZERO_OR_MORE = Range("number", "a number of 0 or more", lambda v: v >= 0)
_ZERO_TO_ONE = Range("number", "a number from 0 to 1", lambda v: 0 <= v <= 1)
_ABOVE_ZERO_TO_ONE = Range("number", "a number above 0 and at most 1", lambda v: 0 < v <= 1)
_DEGREES_ZERO_TO_180 = Range("number", "a number of degrees from 0 to 180",
                             lambda v: 0 <= v <= 180)
_ZERO_TO_BELOW_ONE = Range("number", "a number from 0, below 1", lambda v: 0 <= v < 1)
_ABOVE_ZERO_TO_360 = Range("number", "a number above 0 and at most 360", lambda v: 0 < v <= 360)
_ABOVE_ZERO_BELOW_90 = Range("number", "a number above 0 and below 90", lambda v: 0 < v < 90)

# Each COLMAP command the step runs, in references/options.md's order, with its Time_Limit key.
# sequential_matcher's applies to both Match_Runs (A4).
TIMEOUT_KEYS: Mapping[str, str] = {
    "feature_extractor": "feature_extractor_timeout_seconds",
    "sequential_matcher": "sequential_matcher_timeout_seconds",
    "view_graph_calibrator": "view_graph_calibrator_timeout_seconds",
    "global_mapper": "global_mapper_timeout_seconds",
    "mapper": "mapper_timeout_seconds",
    "image_undistorter": "image_undistorter_timeout_seconds",
    "image_undistorter_standalone": "image_undistorter_standalone_timeout_seconds",
}

# The rows (key, option, range) of each command's Option_Keys, in references/options.md's order.
# The Fixed_Constants, the paths and --SequentialMatching.overlap have no row.
_EVERY_COMMAND = (
    ("default_random_seed", "--default_random_seed", _WHOLE_ZERO_OR_MORE),
)
_FEATURE_EXTRACTOR = (
    ("feature_extractor_descriptor_normalization", "--descriptor_normalization",
     _names("l1_root", "l2")),
    # The one Option_Key whose key breaks the naming rule, and whose value isn't options.md's
    # (Req 12.2, 12.3).
    ("camera_model", "--ImageReader.camera_model", _names("OPENCV", "OPENCV_FISHEYE")),
    ("image_reader_existing_camera_id", "--ImageReader.existing_camera_id",
     _WHOLE_MINUS_ONE_OR_ABOVE_ZERO),
    ("image_reader_default_focal_length_factor", "--ImageReader.default_focal_length_factor",
     _ABOVE_ZERO),
    ("feature_extraction_num_threads", "--FeatureExtraction.num_threads",
     _WHOLE_MINUS_ONE_OR_ABOVE_ZERO),
    ("feature_extraction_max_image_size", "--FeatureExtraction.max_image_size",
     _WHOLE_MINUS_ONE_OR_ABOVE_ZERO),
    ("sift_extraction_max_num_features", "--SiftExtraction.max_num_features", _WHOLE_ABOVE_ZERO),
    ("sift_extraction_first_octave", "--SiftExtraction.first_octave", _WHOLE_MINUS_ONE_OR_MORE),
    ("sift_extraction_num_octaves", "--SiftExtraction.num_octaves", _WHOLE_ABOVE_ZERO),
    ("sift_extraction_octave_resolution", "--SiftExtraction.octave_resolution", _WHOLE_ABOVE_ZERO),
    ("sift_extraction_peak_threshold", "--SiftExtraction.peak_threshold", _ABOVE_ZERO),
    ("sift_extraction_edge_threshold", "--SiftExtraction.edge_threshold", _ABOVE_ZERO),
    ("sift_extraction_estimate_affine_shape", "--SiftExtraction.estimate_affine_shape", _BOOL),
    ("sift_extraction_max_num_orientations", "--SiftExtraction.max_num_orientations",
     _WHOLE_ABOVE_ZERO),
    ("sift_extraction_upright", "--SiftExtraction.upright", _BOOL),
    ("sift_extraction_domain_size_pooling", "--SiftExtraction.domain_size_pooling", _BOOL),
    ("sift_extraction_dsp_min_scale", "--SiftExtraction.dsp_min_scale", _ABOVE_ZERO),
    ("sift_extraction_dsp_max_scale", "--SiftExtraction.dsp_max_scale", _ABOVE_ZERO),
    ("sift_extraction_dsp_num_scales", "--SiftExtraction.dsp_num_scales", _WHOLE_ABOVE_ZERO),
)
_SEQUENTIAL_MATCHER = (
    ("feature_matching_num_threads", "--FeatureMatching.num_threads",
     _WHOLE_MINUS_ONE_OR_ABOVE_ZERO),
    ("feature_matching_guided_matching", "--FeatureMatching.guided_matching", _BOOL),
    ("feature_matching_skip_geometric_verification",
     "--FeatureMatching.skip_geometric_verification", _BOOL),
    ("feature_matching_rig_verification", "--FeatureMatching.rig_verification", _BOOL),
    ("feature_matching_skip_image_pairs_in_same_frame",
     "--FeatureMatching.skip_image_pairs_in_same_frame", _BOOL),
    ("feature_matching_max_num_matches", "--FeatureMatching.max_num_matches", _WHOLE_ABOVE_ZERO),
    ("sift_matching_max_ratio", "--SiftMatching.max_ratio", _ABOVE_ZERO_TO_ONE),
    ("sift_matching_max_distance", "--SiftMatching.max_distance", _ABOVE_ZERO),
    ("sift_matching_cross_check", "--SiftMatching.cross_check", _BOOL),
    ("sift_matching_cpu_brute_force_matcher", "--SiftMatching.cpu_brute_force_matcher", _BOOL),
    ("two_view_geometry_min_num_inliers", "--TwoViewGeometry.min_num_inliers", _WHOLE_ZERO_OR_MORE),
    ("two_view_geometry_multiple_models", "--TwoViewGeometry.multiple_models", _BOOL),
    ("two_view_geometry_compute_relative_pose", "--TwoViewGeometry.compute_relative_pose", _BOOL),
    ("two_view_geometry_detect_watermark", "--TwoViewGeometry.detect_watermark", _BOOL),
    ("two_view_geometry_multiple_ignore_watermark", "--TwoViewGeometry.multiple_ignore_watermark",
     _BOOL),
    ("two_view_geometry_watermark_detection_max_error",
     "--TwoViewGeometry.watermark_detection_max_error", _ABOVE_ZERO),
    ("two_view_geometry_filter_stationary_matches", "--TwoViewGeometry.filter_stationary_matches",
     _BOOL),
    ("two_view_geometry_stationary_matches_max_error",
     "--TwoViewGeometry.stationary_matches_max_error", _ABOVE_ZERO),
    ("two_view_geometry_use_degensac", "--TwoViewGeometry.use_degensac", _BOOL),
    ("two_view_geometry_use_sampson_refinement", "--TwoViewGeometry.use_sampson_refinement", _BOOL),
    ("two_view_geometry_max_error", "--TwoViewGeometry.max_error", _ABOVE_ZERO),
    ("two_view_geometry_confidence", "--TwoViewGeometry.confidence", _ABOVE_ZERO_TO_ONE),
    ("two_view_geometry_max_num_trials", "--TwoViewGeometry.max_num_trials", _WHOLE_ABOVE_ZERO),
    ("two_view_geometry_min_inlier_ratio", "--TwoViewGeometry.min_inlier_ratio", _ZERO_TO_ONE),
    ("two_view_geometry_random_seed", "--TwoViewGeometry.random_seed", _WHOLE_MINUS_ONE_OR_MORE),
    ("sequential_matching_expand_rig_images", "--SequentialMatching.expand_rig_images", _BOOL),
)
_VIEW_GRAPH_CALIBRATOR = (
    ("view_graph_calibrator_cross_validate_prior_focal_lengths",
     "--cross_validate_prior_focal_lengths", _BOOL),
    ("view_graph_calibrator_min_calibrated_pair_ratio", "--min_calibrated_pair_ratio",
     _ZERO_TO_ONE),
    ("view_graph_calibrator_reestimate_relative_pose", "--reestimate_relative_pose", _BOOL),
    ("view_graph_calibrator_min_focal_length_ratio", "--min_focal_length_ratio", _ABOVE_ZERO),
    ("view_graph_calibrator_max_focal_length_ratio", "--max_focal_length_ratio", _ABOVE_ZERO),
    ("view_graph_calibrator_max_calibration_error", "--max_calibration_error", _ABOVE_ZERO),
    ("view_graph_calibrator_relpose_max_error", "--relpose_max_error", _ABOVE_ZERO),
    ("view_graph_calibrator_relpose_min_num_inliers", "--relpose_min_num_inliers",
     _WHOLE_ZERO_OR_MORE),
    ("view_graph_calibrator_relpose_min_inlier_ratio", "--relpose_min_inlier_ratio", _ZERO_TO_ONE),
)
_GLOBAL_MAPPER = (
    ("global_mapper_min_num_matches", "--GlobalMapper.min_num_matches", _WHOLE_ZERO_OR_MORE),
    ("global_mapper_ignore_watermarks", "--GlobalMapper.ignore_watermarks", _BOOL),
    ("global_mapper_num_threads", "--GlobalMapper.num_threads", _WHOLE_MINUS_ONE_OR_ABOVE_ZERO),
    ("global_mapper_random_seed", "--GlobalMapper.random_seed", _WHOLE_MINUS_ONE_OR_MORE),
    ("global_mapper_decompose_relative_pose", "--GlobalMapper.decompose_relative_pose", _BOOL),
    ("global_mapper_multiple_models", "--GlobalMapper.multiple_models", _BOOL),
    ("global_mapper_min_model_size", "--GlobalMapper.min_model_size", _WHOLE_ZERO_OR_MORE),
    ("global_mapper_ba_num_iterations", "--GlobalMapper.ba_num_iterations", _WHOLE_ZERO_OR_MORE),
    ("global_mapper_skip_rotation_averaging", "--GlobalMapper.skip_rotation_averaging", _BOOL),
    ("global_mapper_skip_track_establishment", "--GlobalMapper.skip_track_establishment", _BOOL),
    ("global_mapper_skip_global_positioning", "--GlobalMapper.skip_global_positioning", _BOOL),
    ("global_mapper_skip_bundle_adjustment", "--GlobalMapper.skip_bundle_adjustment", _BOOL),
    ("global_mapper_skip_retriangulation", "--GlobalMapper.skip_retriangulation", _BOOL),
    ("global_mapper_track_intra_image_consistency_threshold",
     "--GlobalMapper.track_intra_image_consistency_threshold", _ABOVE_ZERO),
    ("global_mapper_track_required_tracks_per_view",
     "--GlobalMapper.track_required_tracks_per_view", _WHOLE_ABOVE_ZERO),
    ("global_mapper_track_min_num_views_per_track", "--GlobalMapper.track_min_num_views_per_track",
     _WHOLE_TWO_OR_MORE),
    ("global_mapper_track_max_num_views_per_track", "--GlobalMapper.track_max_num_views_per_track",
     _WHOLE_TWO_OR_MORE),
    ("global_mapper_keep_max_num_tracks", "--GlobalMapper.keep_max_num_tracks", _WHOLE_ABOVE_ZERO),
    ("global_mapper_gp_optimize_positions", "--GlobalMapper.gp_optimize_positions", _BOOL),
    ("global_mapper_gp_optimize_points", "--GlobalMapper.gp_optimize_points", _BOOL),
    ("global_mapper_gp_optimize_scales", "--GlobalMapper.gp_optimize_scales", _BOOL),
    ("global_mapper_gp_loss_function_scale", "--GlobalMapper.gp_loss_function_scale", _ABOVE_ZERO),
    ("global_mapper_gp_max_num_iterations", "--GlobalMapper.gp_max_num_iterations",
     _WHOLE_ABOVE_ZERO),
    ("global_mapper_ba_refine_focal_length", "--GlobalMapper.ba_refine_focal_length", _BOOL),
    ("global_mapper_ba_refine_principal_point", "--GlobalMapper.ba_refine_principal_point", _BOOL),
    ("global_mapper_ba_refine_extra_params", "--GlobalMapper.ba_refine_extra_params", _BOOL),
    ("global_mapper_refine_sensor_from_rig", "--GlobalMapper.refine_sensor_from_rig", _BOOL),
    ("global_mapper_ba_refine_rig_from_world", "--GlobalMapper.ba_refine_rig_from_world", _BOOL),
    ("global_mapper_ba_refine_points3d", "--GlobalMapper.ba_refine_points3D", _BOOL),
    ("global_mapper_ba_min_track_length", "--GlobalMapper.ba_min_track_length",
     _WHOLE_ZERO_OR_MORE),
    ("global_mapper_ba_ceres_loss_function_scale", "--GlobalMapper.ba_ceres_loss_function_scale",
     _ABOVE_ZERO),
    ("global_mapper_ba_ceres_max_num_iterations", "--GlobalMapper.ba_ceres_max_num_iterations",
     _WHOLE_ABOVE_ZERO),
    ("global_mapper_ba_skip_fixed_rotation_stage", "--GlobalMapper.ba_skip_fixed_rotation_stage",
     _BOOL),
    ("global_mapper_ba_skip_joint_optimization_stage",
     "--GlobalMapper.ba_skip_joint_optimization_stage", _BOOL),
    ("global_mapper_tri_complete_max_reproj_error", "--GlobalMapper.tri_complete_max_reproj_error",
     _ABOVE_ZERO),
    ("global_mapper_tri_merge_max_reproj_error", "--GlobalMapper.tri_merge_max_reproj_error",
     _ABOVE_ZERO),
    ("global_mapper_tri_min_angle", "--GlobalMapper.tri_min_angle", _DEGREES_ZERO_TO_180),
    ("global_mapper_ra_use_gravity", "--GlobalMapper.ra_use_gravity", _BOOL),
    ("global_mapper_ra_use_stratified", "--GlobalMapper.ra_use_stratified", _BOOL),
    ("global_mapper_ra_max_rotation_error_deg", "--GlobalMapper.ra_max_rotation_error_deg",
     _DEGREES_ZERO_TO_180),
    ("global_mapper_ra_reweighting", "--GlobalMapper.ra_reweighting",
     _names("UNIFORM", "INLIER_MATCH_COUNT")),
    ("global_mapper_max_angular_reproj_error_deg", "--GlobalMapper.max_angular_reproj_error_deg",
     _ABOVE_ZERO),
    ("global_mapper_max_normalized_reproj_error", "--GlobalMapper.max_normalized_reproj_error",
     _ABOVE_ZERO),
    ("global_mapper_min_tri_angle_deg", "--GlobalMapper.min_tri_angle_deg", _DEGREES_ZERO_TO_180),
)
_MAPPER = (
    ("mapper_min_num_matches", "--Mapper.min_num_matches", _WHOLE_ZERO_OR_MORE),
    ("mapper_ignore_watermarks", "--Mapper.ignore_watermarks", _BOOL),
    ("mapper_multiple_models", "--Mapper.multiple_models", _BOOL),
    ("mapper_max_num_models", "--Mapper.max_num_models", _WHOLE_ABOVE_ZERO),
    ("mapper_max_model_overlap", "--Mapper.max_model_overlap", _WHOLE_ZERO_OR_MORE),
    ("mapper_min_model_size", "--Mapper.min_model_size", _WHOLE_ZERO_OR_MORE),
    ("mapper_init_image_id1", "--Mapper.init_image_id1", _WHOLE_MINUS_ONE_OR_ABOVE_ZERO),
    ("mapper_init_image_id2", "--Mapper.init_image_id2", _WHOLE_MINUS_ONE_OR_ABOVE_ZERO),
    ("mapper_init_num_trials", "--Mapper.init_num_trials", _WHOLE_ABOVE_ZERO),
    ("mapper_structure_less_registration_fallback",
     "--Mapper.structure_less_registration_fallback", _BOOL),
    ("mapper_structure_less_registration_only", "--Mapper.structure_less_registration_only", _BOOL),
    ("mapper_extract_colors", "--Mapper.extract_colors", _BOOL),
    ("mapper_num_threads", "--Mapper.num_threads", _WHOLE_MINUS_ONE_OR_ABOVE_ZERO),
    ("mapper_random_seed", "--Mapper.random_seed", _WHOLE_MINUS_ONE_OR_MORE),
    ("mapper_min_focal_length_ratio", "--Mapper.min_focal_length_ratio", _ABOVE_ZERO),
    ("mapper_max_focal_length_ratio", "--Mapper.max_focal_length_ratio", _ABOVE_ZERO),
    ("mapper_max_extra_param", "--Mapper.max_extra_param", _ABOVE_ZERO),
    ("mapper_ba_refine_focal_length", "--Mapper.ba_refine_focal_length", _BOOL),
    ("mapper_ba_refine_principal_point", "--Mapper.ba_refine_principal_point", _BOOL),
    ("mapper_ba_refine_extra_params", "--Mapper.ba_refine_extra_params", _BOOL),
    ("mapper_ba_refine_sensor_from_rig", "--Mapper.ba_refine_sensor_from_rig", _BOOL),
    ("mapper_ba_local_function_tolerance", "--Mapper.ba_local_function_tolerance", _ZERO_OR_MORE),
    ("mapper_ba_local_max_num_iterations", "--Mapper.ba_local_max_num_iterations",
     _WHOLE_MINUS_ONE_OR_ABOVE_ZERO),
    ("mapper_ba_global_frames_ratio", "--Mapper.ba_global_frames_ratio", _ABOVE_ZERO),
    ("mapper_ba_global_points_ratio", "--Mapper.ba_global_points_ratio", _ABOVE_ZERO),
    ("mapper_ba_global_frames_freq", "--Mapper.ba_global_frames_freq", _WHOLE_ABOVE_ZERO),
    ("mapper_ba_global_points_freq", "--Mapper.ba_global_points_freq", _WHOLE_ABOVE_ZERO),
    ("mapper_ba_global_function_tolerance", "--Mapper.ba_global_function_tolerance",
     _ZERO_OR_MORE),
    ("mapper_ba_global_max_num_iterations", "--Mapper.ba_global_max_num_iterations",
     _WHOLE_MINUS_ONE_OR_ABOVE_ZERO),
    ("mapper_ba_global_max_refinements", "--Mapper.ba_global_max_refinements", _WHOLE_ABOVE_ZERO),
    ("mapper_ba_global_max_refinement_change", "--Mapper.ba_global_max_refinement_change",
     _ZERO_OR_MORE),
    ("mapper_ba_local_max_refinements", "--Mapper.ba_local_max_refinements", _WHOLE_ABOVE_ZERO),
    ("mapper_ba_local_max_refinement_change", "--Mapper.ba_local_max_refinement_change",
     _ZERO_OR_MORE),
    ("mapper_ba_min_num_residuals_for_cpu_multi_threading",
     "--Mapper.ba_min_num_residuals_for_cpu_multi_threading", _WHOLE_ZERO_OR_MORE),
    ("mapper_snapshot_frames_freq", "--Mapper.snapshot_frames_freq", _WHOLE_ZERO_OR_MORE),
    ("mapper_fix_existing_frames", "--Mapper.fix_existing_frames", _BOOL),
    ("mapper_init_min_num_inliers", "--Mapper.init_min_num_inliers", _WHOLE_ZERO_OR_MORE),
    ("mapper_init_max_error", "--Mapper.init_max_error", _ABOVE_ZERO),
    ("mapper_init_max_forward_motion", "--Mapper.init_max_forward_motion", _ZERO_TO_ONE),
    ("mapper_init_min_tri_angle", "--Mapper.init_min_tri_angle", _DEGREES_ZERO_TO_180),
    ("mapper_init_max_reg_trials", "--Mapper.init_max_reg_trials", _WHOLE_ABOVE_ZERO),
    ("mapper_abs_pose_max_error", "--Mapper.abs_pose_max_error", _ABOVE_ZERO),
    ("mapper_abs_pose_min_num_inliers", "--Mapper.abs_pose_min_num_inliers", _WHOLE_ZERO_OR_MORE),
    ("mapper_abs_pose_min_inlier_ratio", "--Mapper.abs_pose_min_inlier_ratio", _ZERO_TO_ONE),
    ("mapper_filter_max_reproj_error", "--Mapper.filter_max_reproj_error", _ABOVE_ZERO),
    ("mapper_filter_min_tri_angle", "--Mapper.filter_min_tri_angle", _DEGREES_ZERO_TO_180),
    ("mapper_max_reg_trials", "--Mapper.max_reg_trials", _WHOLE_ABOVE_ZERO),
    ("mapper_ba_local_num_images", "--Mapper.ba_local_num_images", _WHOLE_ABOVE_ZERO),
    ("mapper_ba_local_min_tri_angle", "--Mapper.ba_local_min_tri_angle", _DEGREES_ZERO_TO_180),
    ("mapper_ba_global_ignore_redundant_points3d", "--Mapper.ba_global_ignore_redundant_points3D",
     _BOOL),
    ("mapper_ba_global_ignore_redundant_points3d_min_coverage_gain",
     "--Mapper.ba_global_ignore_redundant_points3D_min_coverage_gain", _ZERO_TO_ONE),
    ("mapper_max_runtime_seconds", "--Mapper.max_runtime_seconds", _WHOLE_MINUS_ONE_OR_ABOVE_ZERO),
    ("mapper_tri_max_transitivity", "--Mapper.tri_max_transitivity", _WHOLE_ZERO_OR_MORE),
    ("mapper_tri_create_max_angle_error", "--Mapper.tri_create_max_angle_error", _ABOVE_ZERO),
    ("mapper_tri_continue_max_angle_error", "--Mapper.tri_continue_max_angle_error", _ABOVE_ZERO),
    ("mapper_tri_merge_max_reproj_error", "--Mapper.tri_merge_max_reproj_error", _ABOVE_ZERO),
    ("mapper_tri_complete_max_reproj_error", "--Mapper.tri_complete_max_reproj_error", _ABOVE_ZERO),
    ("mapper_tri_complete_max_transitivity", "--Mapper.tri_complete_max_transitivity",
     _WHOLE_ZERO_OR_MORE),
    ("mapper_tri_re_max_angle_error", "--Mapper.tri_re_max_angle_error", _ABOVE_ZERO),
    ("mapper_tri_re_min_ratio", "--Mapper.tri_re_min_ratio", _ZERO_TO_ONE),
    ("mapper_tri_re_max_trials", "--Mapper.tri_re_max_trials", _WHOLE_ZERO_OR_MORE),
    ("mapper_tri_min_angle", "--Mapper.tri_min_angle", _DEGREES_ZERO_TO_180),
    ("mapper_tri_ignore_two_view_tracks", "--Mapper.tri_ignore_two_view_tracks", _BOOL),
)
# The undistortion settings, which image_undistorter and image_undistorter_standalone both take
# with the same values (Req 2.5, 8.3).
_UNDISTORTION = (
    ("undistortion_copy_policy", "--copy_policy", _names("COPY", "SOFT_LINK", "HARD_LINK")),
    ("undistortion_blank_pixels", "--blank_pixels", _ZERO_TO_ONE),
    ("undistortion_min_scale", "--min_scale", _ABOVE_ZERO),
    ("undistortion_max_scale", "--max_scale", _ABOVE_ZERO),
    ("undistortion_max_image_size", "--max_image_size", _WHOLE_MINUS_ONE_OR_ABOVE_ZERO),
    ("undistortion_roi_min_x", "--roi_min_x", _ZERO_TO_BELOW_ONE),
    ("undistortion_roi_min_y", "--roi_min_y", _ZERO_TO_BELOW_ONE),
    ("undistortion_roi_max_x", "--roi_max_x", _ABOVE_ZERO_TO_ONE),
    ("undistortion_roi_max_y", "--roi_max_y", _ABOVE_ZERO_TO_ONE),
    ("undistortion_jpeg_quality", "--jpeg_quality", _WHOLE_MINUS_ONE_OR_ONE_TO_100),
    ("undistortion_num_threads", "--num_threads", _WHOLE_MINUS_ONE_OR_ABOVE_ZERO),
)

# One OptionKey per Option_Key, 173 in all, in references/options.md's order: each command's
# block in turn, with each option where a block first has it.
OPTION_KEYS: tuple[OptionKey, ...] = (
    *_keys(tuple(TIMEOUT_KEYS), _EVERY_COMMAND),
    *_keys(("feature_extractor",), _FEATURE_EXTRACTOR),
    *_keys(("sequential_matcher",), _SEQUENTIAL_MATCHER),
    *_keys(("view_graph_calibrator",), _VIEW_GRAPH_CALIBRATOR),
    *_keys(("global_mapper",), _GLOBAL_MAPPER),
    *_keys(("mapper",), _MAPPER),
    *_keys(("image_undistorter", "image_undistorter_standalone"), _UNDISTORTION),
)

# The Shared_Settings of [cameras] that aren't Option_Keys, each with its Range (Req 12.10), in
# config.toml's order.
_SHARED: dict[str, Range] = {
    "fisheye_fov_degrees": _ABOVE_ZERO_TO_360,
    "far_overlap": _WHOLE_ABOVE_ZERO,
    "min_placed_frames": _WHOLE_ABOVE_ZERO,
    "min_placed_share": _ZERO_TO_ONE,
    "up_agreement": _ABOVE_ZERO_TO_ONE,
    "axis_point_min_eigen_share": _ABOVE_ZERO_TO_ONE,
    "axis_point_min_front_share": _ZERO_TO_ONE,
    "heading_min_degrees": _ABOVE_ZERO_BELOW_90,
    **{key: _ABOVE_ZERO for key in TIMEOUT_KEYS.values()},  # the Time_Limits, in seconds
    "min_vcpus": _WHOLE_ABOVE_ZERO,
    "min_memory_gb": _WHOLE_ABOVE_ZERO,
}
# The Preset_Setting, which each Preset table [cameras.<preset>] holds (Req 12.4).
_PRESET_KEY = "near_overlap"

# The cross checks, which run once every key has passed on its own, in this order.
# The pairs (low, high) of Option_Keys whose low must be no greater than its high.
_NO_GREATER = (
    ("sift_extraction_dsp_min_scale", "sift_extraction_dsp_max_scale"),
    ("view_graph_calibrator_min_focal_length_ratio",
     "view_graph_calibrator_max_focal_length_ratio"),
    ("mapper_min_focal_length_ratio", "mapper_max_focal_length_ratio"),
    ("global_mapper_track_min_num_views_per_track", "global_mapper_track_max_num_views_per_track"),
    ("undistortion_min_scale", "undistortion_max_scale"),
)
# The two Option_Keys that can't both be true: COLMAP's global mapper can't skip both stages of
# its bundle adjustment (F14).
_NOT_BOTH_TRUE = ("global_mapper_ba_skip_fixed_rotation_stage",
                  "global_mapper_ba_skip_joint_optimization_stage")
# The pairs (low, high) of Option_Keys whose low must be below its high: the crop's edges.
_BELOW = (
    ("undistortion_roi_min_x", "undistortion_roi_max_x"),
    ("undistortion_roi_min_y", "undistortion_roi_max_y"),
)


def read_settings(settings: Mapping[str, Any], section: str, preset: str) -> CamerasSettings:
    """Read and check every key of the step's settings, and keep them in a CamerasSettings.

    `settings` holds the Shared_Settings of [<section>] and the Preset_Setting of
    [<section>.<preset>], as the Step_Runner's StepSettings gives them. `section` and `preset`
    name those tables in the messages. The keys are checked one by one: the step's own
    Shared_Settings, then near_overlap, then the Option_Keys. Then the cross checks run.
    """
    shared = f"[{section}]"
    values = {key: _checked(settings, key, shared, rng) for key, rng in _SHARED.items()}
    near_overlap = _checked(settings, _PRESET_KEY, f"[{section}.{preset}]", _WHOLE_ABOVE_ZERO)
    options = {item.key: _checked(settings, item.key, shared, item.range, item.option)
               for item in OPTION_KEYS}
    _cross_check(options, shared)
    return CamerasSettings(
        camera_model=options["camera_model"],
        fisheye_fov_degrees=values["fisheye_fov_degrees"],
        near_overlap=near_overlap,
        far_overlap=values["far_overlap"],
        min_placed_frames=values["min_placed_frames"],
        min_placed_share=values["min_placed_share"],
        level=_level(values),
        timeouts={command: values[key] for command, key in TIMEOUT_KEYS.items()},
        min_vcpus=values["min_vcpus"],
        min_memory_gb=values["min_memory_gb"],
        options=options,
    )


def _level(values: Mapping[str, Any]) -> LevelSettings:
    """The Levelling's four settings, from the checked Shared_Settings."""
    return LevelSettings(up_agreement=values["up_agreement"],
                         heading_min_degrees=values["heading_min_degrees"],
                         axis_point_min_eigen_share=values["axis_point_min_eigen_share"],
                         axis_point_min_front_share=values["axis_point_min_front_share"])


def _checked(settings: Mapping[str, Any], key: str, table: str, rng: Range,
             option: str | None = None) -> Any:
    """settings[key], as Config gave it, once it's of the Range's kind and passes its test.

    `table` is the table that holds the key, and `option` COLMAP's option for an Option_Key.
    """
    value = settings[key]  # a missing key raises the Step_Runner's own message (Req 12.7)
    if not rng.accepts(value):
        must = _must_be(rng, value)
        raise ExpectedFailure(f"The setting {_named(key, table, option)} is {_shown(value)}, "
                              f"but it must be {must}. Change it to {must}.")
    return value


def _named(key: str, table: str, option: str | None) -> str:
    """A setting as a refusal names it: its key and its table, then COLMAP's option for an
    Option_Key, like `default_random_seed in [cameras] (COLMAP's --default_random_seed)`."""
    named = f"{key} in {table}"
    return named if option is None else f"{named} (COLMAP's {option})"


def _must_be(rng: Range, value: object) -> str:
    """What a refused value must be, in words. For a whole number that doesn't fit a 32-bit int,
    the words add the int's bounds, because the Range's words alone would seem to allow it."""
    if rng.kind == "whole" and type(value) is int and not _INT_MIN <= value <= _INT_MAX:
        return f"{rng.wanted} that fits in 32 bits, from {_INT_MIN} to {_INT_MAX}"
    return rng.wanted


def _cross_check(options: Mapping[str, Any], table: str) -> None:
    """Refuse the first Option_Keys that don't fit together, naming both and their values.

    `table` is the table that holds every Option_Key.
    """
    for low, high in _NO_GREATER:
        if options[low] > options[high]:
            _refuse_pair(options, table, low, "no greater than", high)
    first, second = _NOT_BOTH_TRUE
    if options[first] and options[second]:
        raise ExpectedFailure(
            f"The setting {first} in {table} is {_shown(options[first])}, but it can't be true "
            f"while {second} in {table} is {_shown(options[second])} too, because COLMAP's "
            "global mapper can't skip both stages of its bundle adjustment. Set one of them to "
            "false.")
    for low, high in _BELOW:
        if options[low] >= options[high]:
            _refuse_pair(options, table, low, "below", high)


def _refuse_pair(options: Mapping[str, Any], table: str, low: str, relation: str,
                 high: str) -> NoReturn:
    """Refuse `low` for not being `relation` `high`, naming both settings, their table and their
    values as written."""
    raise ExpectedFailure(f"The setting {low} in {table} is {_shown(options[low])}, but it must "
                          f"be {relation} {high} in {table}, which is {_shown(options[high])}. "
                          f"Lower {low} or raise {high}.")


def _shown(value: object) -> str:
    """A value about as TOML writes it: true or false for a bool, else Python's repr."""
    if isinstance(value, bool):
        return "true" if value else "false"
    return repr(value)
