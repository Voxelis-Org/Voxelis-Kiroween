"""Tests for splatpipe/cameras/commands.py, the COLMAP command lines, and
splatpipe/cameras/colmap.py, the COLMAP runner.

The command line tests build the eight runs from the real config, read by path as
REPO_ROOT / "config.toml", because tests/conftest.py points CONFIG_PATH at a test config. They
build them for each Preset, both Camera_Models and both values of mask_used. The Job's folders
are given relative to tmp_path, the tests' working folder, so only Paths.of can make the paths
absolute. The colmap_options oracle in tests/conftest.py parses each command's block in
.kiro/skills/colmap-cli/references/options.md with its own code. Each command line holds its
block's options in the block's order, each value reading back as the block's, with only the
changes the design gives:

- --ImageReader.camera_model is camera_model, where the block has COLMAP's SIMPLE_RADIAL
  (Req 3.2).
- --SequentialMatching.quadratic_overlap is 0 in the Near_Match_Run and 1 in the Far_Match_Run
  (A1), and --SequentialMatching.overlap is the Preset's near_overlap in the first and
  far_overlap in the second (Req 4.1 to 4.3).
- --ImageReader.camera_params, which the block leaves out, holds the Fisheye_Start only for
  RADIAL_FISHEYE, and --ImageReader.camera_mask_path is there only with mask_used (Req 3.3 to
  3.6).
- Each path is absolute and lies where the design's table of builders puts it: inside
  work/frames/ when the block's path starts from $FRAMES, and inside the COLMAP_Folder when it
  starts from $OUT (Req 1.5, 2.2, 3.1, 6.1, 6.2, 6.6, 8.1).

A value reads back as COLMAP reads it, as the kind of its Option_Key's value in the real config:
a bool from 1 or 0, an int from a whole number in decimal, a float from any decimal, and a name
as it is (Req 2.4). format_value and fisheye_start get the design's examples.

Property 1 draws with Hypothesis any settings that read_settings accepts: every key of [cameras]
and the Preset's near_overlap, each from its Range in the design's "The settings checks", which
this file writes out rather than reading settings.py, so the two ba_refine_extra_params keys are
always true (D20). Whole numbers often take their range's
ends, 2147483647 among them. Decimals are TOML floats or TOML integers, and often take -0.0,
5e-324, 1e-300, 1e300 and the largest float wherever their range holds them. The two settings of
each cross check are put in order. It also draws mask_used and a Pick_Size from 1 × 1 to
4000 × 4000, and camera_model takes either Camera_Model, RADIAL or RADIAL_FISHEYE (D19). Its
oracle is colmap_options with this file's own code. Each of the eight command lines holds every
option of its block exactly once and no other option, with --ImageReader.camera_mask_path only
when mask_used is true and --ImageReader.camera_params only for RADIAL_FISHEYE, never for RADIAL.
Its five numbers read back as the Fisheye_Start, which the test works out itself in exact
fractions, without fisheye_start, as f = (L / 2) / (θ / 2), width / 2, height / 2, 0 and 0: cx,
cy, k1 and k2 bit for bit, and f within 1e-12 of it, or as infinity when f lies beyond the
largest float, because the code and the design's formula round differently. Each Fixed_Constant
has its fixed value, each Option_Key's text reads back as its drawn value, and
--SequentialMatching.overlap's as near_overlap or far_overlap. The two undistorters give each
undistortion setting the same text, and no line holds an option that Req 2.6 keeps off.

The runner's tests start no COLMAP. Most replace subprocess.run, as splatpipe.cameras.colmap sees
it, with a stand-in that records its call and writes two of glog's lines to the Command_Log it's
given, then exits with a code or raises TimeoutExpired or FileNotFoundError. One test runs a real
child, sys.executable -c, which writes a line and sleeps until its Time_Limit of 0.2 s stops it
(Req 2.8, 2.10). last_error_line gets Command_Logs as glog writes them under --log_target stderr
(F16, C3), and exit_text an exit code and a signal.

capped gets the Calibration's run, which its builder gives for the real config with the key
view_graph_calibrator_timeout_seconds at 600 s, and a Cap_Deadline in UTC at the edges of C13 and
C14: none, exactly 600 s left, 599 s, a microsecond, exactly 0 and a minute past. Each gives the
run with its Time_Limit and capped_at, every other field as it was, or the Req 17.5 failure
(Req 17.3 to 17.5). Property 8 gives that run, with dataclasses.replace, a Time_Limit key's value
drawn with Hypothesis from 1e-6 to 1e6 s, the ends included, and draws a time now from 2000 to
2100 in UTC with microseconds, and a Cap_Deadline of None or now plus an offset from -2e6 to 2e6 s
in microsecond steps, often exactly the key's value or 0. Its oracle works out the seconds left
with its own timedelta arithmetic, without capped, and gives the run as it was, the run with
those seconds as its Time_Limit and the Cap_Deadline as its capped_at, or the Req 17.5 failure,
which names the command and the Cap_Deadline (Req 17.3 to 17.5, 17.8).

The -h test is the one test here that runs COLMAP, and it's skipped, saying COLMAP isn't
installed, where colmap isn't on the PATH (Req 16.8). It calls smoke.check_options, the
Cameras_Smoke_Check's Option_Check, with the real config and the SmokeSizes of the real
[cloud.smoke], both read by path, in tmp_path. So each of the seven colmap <command> -h runs
through run_colmap within check_timeout_seconds, 60 s, and must list every option that the
builders pass that command for each Preset, both Camera_Models and with and without the Mask,
with no failure, so a mistyped option shows up without a whole COLMAP run (Req 16.7, 16.13,
18.3, A16, C20).
"""

import io
import math
import os
import re
import shutil
import subprocess
import sys
import tomllib
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from conftest import ColmapOption  # an option of options.md, as the colmap_options oracle sorts it
from splatpipe.cameras import commands, smoke
from splatpipe.cameras.colmap import (ColmapRun, capped, colmap_env, exit_text, last_error_line,
                                      run_colmap)
from splatpipe.cameras.commands import Paths, fisheye_start, format_value
from splatpipe.cameras.frames_input import FramesInput, Pick
from splatpipe.cameras.settings import CamerasSettings, read_settings
from splatpipe.config import REPO_ROOT, StepSettings, load_config
from splatpipe.errors import ExpectedFailure

# Read by path, so these tests read it although CONFIG_PATH points at a test config.
REAL_CONFIG = REPO_ROOT / "config.toml"
# The real [cameras] with its Preset tables, each value as tomllib gives it.
REAL = tomllib.loads(REAL_CONFIG.read_text(encoding="utf-8"))["cameras"]
# The three Presets and the two Camera_Models, written out here rather than taken from the code.
PRESET_NAMES = ("fast", "default", "quality")
CAMERA_MODELS = ("RADIAL", "RADIAL_FISHEYE")  # D5, D19
# The Pick_Size the command lines are built for: a landscape Pick at the Frames_Step's long edge
# of 1600 pixels, the size of the design's Fisheye_Start example.
PICK_WIDTH, PICK_HEIGHT = 1600, 900

# Each run the builders give, by its name, with COLMAP's command, whose block in options.md its
# command line follows.
RUN_COMMANDS = {
    "feature_extractor": "feature_extractor",
    "sequential_matcher_near": "sequential_matcher",
    "sequential_matcher_far": "sequential_matcher",
    "view_graph_calibrator": "view_graph_calibrator",
    "global_mapper": "global_mapper",
    "mapper": "mapper",
    "image_undistorter": "image_undistorter",
    "image_undistorter_standalone": "image_undistorter_standalone",
}
NEAR, FAR = "sequential_matcher_near", "sequential_matcher_far"

# The two options that come and go: --ImageReader.camera_params, which options.md's block leaves
# out, only for RADIAL_FISHEYE, and --ImageReader.camera_mask_path, which the block has, only
# with mask_used (Req 3.3 to 3.6).
CAMERA_PARAMS = "--ImageReader.camera_params"
CAMERA_MASK = "--ImageReader.camera_mask_path"


def readable(error: ExpectedFailure) -> str:
    """The Readable_Error of `error`, checked to hold no traceback (Req 16.9)."""
    message = str(error)
    assert "Traceback" not in message
    return message


def read_back(text: str, kind: type) -> object:
    """`text`, a value on a command line, read as a value of `kind` the way COLMAP reads it: a
    bool from 1 or 0, an int from a whole number in decimal, a float from any decimal, and a name
    as it is. Text that isn't a value of `kind` raises ValueError."""
    if kind is bool:
        if text not in ("1", "0"):
            raise ValueError(f"{text!r} isn't 1 or 0")
        return text == "1"
    if kind is int:
        if re.fullmatch(r"-?[0-9]+", text) is None:
            raise ValueError(f"{text!r} isn't a whole number in decimal")
        return int(text)
    if kind is float:
        return float(text)
    return text


# The command lines (Req 1.5, 2.2, 2.3, 2.4, 3.1 to 3.6, 4.1 to 4.3, 6.1, 6.2, 6.6, 8.1)

def real_settings(preset: str, **changes: Any) -> CamerasSettings:
    """read_settings on the real config's [cameras] for `preset`, as the step calls it, with each
    key in `changes` set to its value."""
    settings = load_config(REAL_CONFIG).step_settings("cameras", preset)
    changed = StepSettings({**settings, **changes}, section="cameras", preset=preset)
    return read_settings(changed, "cameras", preset)


def built_runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, preset: str, camera_model: str,
               mask_used: bool) -> tuple[dict[str, ColmapRun], dict[str, Path]]:
    """The eight runs the builders give, by the runs' names, for the real config with `preset`
    and `camera_model`, a Frames_Record with `mask_used`, and the Job folder tmp_path/job, which
    isn't made. With them, the two folders their paths must lie in, absolute, by options.md's
    names for them, $FRAMES and $OUT.

    The Job's folders are given to Paths.of relative to tmp_path, made the working folder, so only
    Paths.of can make the paths absolute (Req 2.2). eight_runs builds the runs.
    """
    monkeypatch.chdir(tmp_path)
    paths = Paths.of(Path("job/work/cameras"), Path("job/work/frames"), Path("job/out"))
    job = tmp_path.resolve() / "job"
    folders = {"$FRAMES": job / "work" / "frames", "$OUT": job / "work" / "cameras" / "colmap"}
    cfg = real_settings(preset, camera_model=camera_model)
    frames = FramesInput(folder=folders["$FRAMES"], pick_width=PICK_WIDTH,
                         pick_height=PICK_HEIGHT, mask_used=mask_used,
                         picks=(Pick(number=0, file="000000.jpg"),))
    env = colmap_env(folders["$OUT"] / "tmp", {"PATH": "/usr/bin:/bin"})
    return eight_runs(cfg, frames, paths, env), folders


def eight_runs(cfg: CamerasSettings, frames: FramesInput, paths: Paths,
               env: Mapping[str, str]) -> dict[str, ColmapRun]:
    """The eight runs the builders give for `cfg`, `frames`, `paths` and `env`, by the runs'
    names.

    The Global_Run is given the Calibration_Copy for RADIAL and the Database for RADIAL_FISHEYE,
    as the solve flow gives it (Req 6.1, 6.2, O9), and the Undistortion the Global_Run's Model 0.
    """
    database = paths.database if cfg.camera_model == "RADIAL_FISHEYE" else paths.calibrated
    runs = (commands.feature_extractor(cfg, frames, paths, env),
            commands.sequential_matcher(cfg, paths, env, far=False),
            commands.sequential_matcher(cfg, paths, env, far=True),
            commands.view_graph_calibrator(cfg, paths, env),
            commands.global_mapper(cfg, paths, env, database),
            commands.mapper(cfg, paths, env),
            commands.image_undistorter(cfg, paths, env, paths.global_models / "0"),
            commands.image_undistorter_standalone(cfg, paths, env))
    return {run.name: run for run in runs}


def designed_paths(camera_model: str) -> dict[str, dict[str, str]]:
    """Where each path option of each run points, by the run's name, inside the folder that its
    block's path in options.md starts from: $FRAMES, the frames step's work folder, or $OUT, the
    camera solve's, whose files the design keeps in the COLMAP_Folder (Req 11.5, O8).

    The places are those of the design's table of builders and "The Work_Folder". The
    Feature_Extraction reads the Grey_Copies, and both mappers and the Undistortion read the
    Colour_Picks (Req 1.5). The Calibration and the Global_Run read the Calibration_Copy, but the
    Global_Run reads the Database for RADIAL_FISHEYE (Req 6.1, 6.2, O9), and the mapper reads the
    Database (Req 6.6). The Undistortion reads the Model that built_runs gives it, and writes
    inside the COLMAP_Folder (Req 8.1). "." is the folder itself.
    """
    global_database = ("database.db" if camera_model == "RADIAL_FISHEYE"
                       else "database_calibrated.db")
    return {
        "feature_extractor": {"--database_path": "database.db", "--image_path": "grey",
                              CAMERA_MASK: "mask.png"},
        NEAR: {"--database_path": "database.db"},
        FAR: {"--database_path": "database.db"},
        "view_graph_calibrator": {"--database_path": "database_calibrated.db"},
        "global_mapper": {"--database_path": global_database, "--image_path": "images",
                          "--output_path": "global"},
        "mapper": {"--database_path": "database.db", "--image_path": "images",
                   "--output_path": "incremental"},
        "image_undistorter": {"--image_path": "images", "--input_path": "global/0",
                              "--output_path": "undistorted"},
        "image_undistorter_standalone": {"--image_path": ".", "--input_file": "mask_camera.txt",
                                         "--output_path": "mask_undistorted"},
    }


def line_value(option: ColmapOption | None, text: str) -> object:
    """The value `text` of `option` on a command line, read back the way the oracle sorts the
    option: a path as a Path, --SequentialMatching.overlap as a whole number, an Option_Key's
    value as the kind of its value in the real config, and a Fixed_Constant's value, or the value
    of an option the block doesn't have, as text."""
    if option is None or option.sort == "fixed":
        return text
    if option.sort == "path":
        return Path(text)
    if option.sort == "overlap":
        return read_back(text, int)
    return read_back(text, type(REAL[option.key]))


def command_line(run: ColmapRun, block: Sequence[ColmapOption]) -> list[tuple[str, object]]:
    """Each option of `run`'s command line after colmap and its command, in order, with its value
    as line_value reads it back by `block`, its command's block in options.md."""
    by_option = {option.option: option for option in block}
    words = run.args[2:]
    return [(option, line_value(by_option.get(option), text))
            for option, text in zip(words[::2], words[1::2], strict=True)]


def designed_line(block: Sequence[ColmapOption], name: str, folders: Mapping[str, Path], *,
                  camera_model: str, preset: str, mask_used: bool) -> list[tuple[str, object]]:
    """The options and values that the run `name`'s command line must hold after its command, in
    order: those of `block`, its block in options.md, each with the block's value as line_value
    reads it back, but for the design's changes.

    --ImageReader.camera_model is `camera_model` (Req 3.2). --SequentialMatching.overlap is the
    Preset's near_overlap in the Near_Match_Run and far_overlap in the Far_Match_Run, and
    --SequentialMatching.quadratic_overlap is 0 and 1 (Req 4.1, 4.2, A1). Without mask_used,
    --ImageReader.camera_mask_path is left out (Req 3.6). Each path is the one designed_paths
    gives, inside the folder in `folders` that the block's path starts from (Req 2.2).
    """
    paths = designed_paths(camera_model)[name]
    line: list[tuple[str, object]] = []
    for option in block:
        if option.option == CAMERA_MASK and not mask_used:
            continue
        value: object
        if option.sort == "path":
            value = folders[option.text.partition("/")[0]] / paths[option.option]
        elif option.sort == "overlap":
            value = REAL[preset]["near_overlap"] if name == NEAR else REAL["far_overlap"]
        elif option.key == "camera_model":
            value = camera_model
        elif option.option == "--SequentialMatching.quadratic_overlap":
            value = "1" if name == FAR else "0"
        else:
            value = line_value(option, option.text)
        line.append((option.option, value))
    return line


@pytest.mark.parametrize("mask_used", [False, True], ids=["no mask", "mask"])
@pytest.mark.parametrize("camera_model", CAMERA_MODELS)
@pytest.mark.parametrize("preset", PRESET_NAMES)
def test_each_command_line_holds_its_blocks_options_in_order_with_only_the_designs_changes(
        colmap_options: Mapping[str, tuple[ColmapOption, ...]], tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch, preset: str, camera_model: str,
        mask_used: bool) -> None:
    runs, folders = built_runs(tmp_path, monkeypatch, preset, camera_model, mask_used)
    assert {name: run.command for name, run in runs.items()} == RUN_COMMANDS
    for name, run in runs.items():
        block = colmap_options[run.command]
        assert run.args[:2] == ("colmap", run.command), name
        line = command_line(run, block)
        params = [value for option, value in line if option == CAMERA_PARAMS]
        line = [(option, value) for option, value in line if option != CAMERA_PARAMS]
        assert line == designed_line(block, name, folders, camera_model=camera_model,
                                     preset=preset, mask_used=mask_used), name
        # Only RADIAL_FISHEYE's Feature_Extraction has the Fisheye_Start, whose five numbers read
        # back as the same floats (Req 2.4, 3.3, 3.4).
        starts = [tuple(float(number) for number in str(text).split(",")) for text in params]
        fisheye = name == "feature_extractor" and camera_model == "RADIAL_FISHEYE"
        assert starts == ([fisheye_start(PICK_WIDTH, PICK_HEIGHT, REAL["fisheye_fov_degrees"])]
                          if fisheye else []), name


# Property 1: Command lines hold the Fixed_Constants and every option once (Req 2.3 to 2.6,
# 2.13, 3.2, 3.3, 3.5, 3.6, 4.1 to 4.3, 8.3)

TOP = 2**31 - 1  # the largest 32-bit int, the top of every whole number's range
TOML_TOP = 2**63 - 1  # the largest TOML integer, the top of a decimal setting's integers
TINY = math.nextafter(0.0, 1.0)  # 5e-324, the smallest float above 0
BIGGEST = sys.float_info.max  # the largest finite float
# Floats from the far ends of a double, drawn wherever a range holds them: -0.0, whose sign must
# come back, the smallest float above 0, 1e-300, 1e300 and the largest finite float.
FAR_FLOATS = (-0.0, TINY, 1e-300, 1e300, BIGGEST)


def whole(low: int, high: int = TOP) -> st.SearchStrategy[int]:
    """A whole number from `low` to `high`, as Config gives a whole-number setting: a TOML
    integer. Both ends come up often, so a range with no top of its own often gives 2147483647,
    the top of the 32-bit int that COLMAP reads these options into."""
    return st.sampled_from((low, high)) | st.integers(low, high)


def number(low: int, high: int | None = None, *, above: bool = False,
           below: bool = False) -> st.SearchStrategy[int | float]:
    """A number from `low` to `high`, as Config gives a decimal setting: a TOML float, or a TOML
    integer, which read_settings takes too. `above` leaves `low` itself out and `below` leaves
    `high` out. With no `high`, the range has no top.

    The floats come from Hypothesis's floats, subnormals included, and often from the range's
    ends, the floats just inside them and FAR_FLOATS, wherever the range holds them. Without a
    top, the integers reach TOML's largest.
    """
    top = BIGGEST if high is None else float(high)

    def inside(value: float) -> bool:
        return ((value > low) if above else (value >= low)) and (
            (value < top) if below else (value <= top))

    ends = [value for value in (float(low), math.nextafter(low, math.inf), top,
                                math.nextafter(top, -math.inf), *FAR_FLOATS) if inside(value)]
    first = low + 1 if above else low
    last = TOML_TOP if high is None else high - 1 if below else high
    return (st.sampled_from(ends) | st.floats(low, top, exclude_min=above, exclude_max=below)
            | st.integers(first, last))


# The Ranges of the design's "The settings checks", each drawing the values it allows.
BOOL = st.booleans()
TRUE = st.just(True)  # the two ba_refine_extra_params keys accept only true (Req 12.16, D20)
WHOLE_MINUS_1_OR_ABOVE_0 = st.just(-1) | whole(1)
WHOLE_MINUS_1_OR_MORE = whole(-1)
WHOLE_0_OR_MORE = whole(0)
WHOLE_ABOVE_0 = whole(1)
WHOLE_2_OR_MORE = whole(2)
WHOLE_MINUS_1_OR_1_TO_100 = st.just(-1) | whole(1, 100)
ABOVE_0 = number(0, above=True)
ZERO_OR_MORE = number(0)
ZERO_TO_1 = number(0, 1)
ABOVE_0_AT_MOST_1 = number(0, 1, above=True)
DEGREES_0_TO_180 = number(0, 180)
ZERO_BELOW_1 = number(0, 1, below=True)
ABOVE_0_AT_MOST_360 = number(0, 360, above=True)
ABOVE_0_BELOW_90 = number(0, 90, above=True, below=True)

# The design's table of the Option_Keys, row by row: each Range with its keys. The bools are the
# keys whose value in the design's [cameras] block is true or false, but the two
# ba_refine_extra_params keys, which are always true.
OPTION_ROWS: tuple[tuple[st.SearchStrategy[Any], tuple[str, ...]], ...] = (
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
        "global_mapper_refine_sensor_from_rig",
        "global_mapper_ba_refine_rig_from_world", "global_mapper_ba_refine_points3d",
        "global_mapper_ba_skip_fixed_rotation_stage",
        "global_mapper_ba_skip_joint_optimization_stage", "global_mapper_ra_use_gravity",
        "global_mapper_ra_use_stratified", "mapper_ignore_watermarks", "mapper_multiple_models",
        "mapper_structure_less_registration_fallback", "mapper_structure_less_registration_only",
        "mapper_extract_colors", "mapper_ba_refine_focal_length",
        "mapper_ba_refine_principal_point",
        "mapper_ba_refine_sensor_from_rig", "mapper_fix_existing_frames",
        "mapper_ba_global_ignore_redundant_points3d", "mapper_tri_ignore_two_view_tracks",
    )),
    (TRUE, ("mapper_ba_refine_extra_params", "global_mapper_ba_refine_extra_params")),
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
    (st.sampled_from(CAMERA_MODELS), ("camera_model",)),
    (st.sampled_from(("l1_root", "l2")), ("feature_extractor_descriptor_normalization",)),
    (st.sampled_from(("UNIFORM", "INLIER_MATCH_COUNT")), ("global_mapper_ra_reweighting",)),
    (st.sampled_from(("COPY", "SOFT_LINK", "HARD_LINK")), ("undistortion_copy_policy",)),
)
OPTION_RANGES = {key: strategy for strategy, keys in OPTION_ROWS for key in keys}

# The design's table of the other settings (Req 12.10), near_overlap included.
OWN_ROWS: tuple[tuple[st.SearchStrategy[Any], tuple[str, ...]], ...] = (
    (ABOVE_0_AT_MOST_360, ("fisheye_fov_degrees",)),
    (WHOLE_ABOVE_0, ("near_overlap", "far_overlap", "min_placed_frames", "min_vcpus",
                     "min_memory_gb")),
    (ZERO_TO_1, ("min_placed_share", "axis_point_min_front_share")),
    (ABOVE_0_AT_MOST_1, ("up_agreement", "axis_point_min_eigen_share")),
    (ABOVE_0_BELOW_90, ("heading_min_degrees",)),
    (ABOVE_0, ("feature_extractor_timeout_seconds", "sequential_matcher_timeout_seconds",
               "view_graph_calibrator_timeout_seconds", "global_mapper_timeout_seconds",
               "mapper_timeout_seconds", "image_undistorter_timeout_seconds",
               "image_undistorter_standalone_timeout_seconds")),
)
RANGES = {**{key: strategy for strategy, keys in OWN_ROWS for key in keys}, **OPTION_RANGES}

# The cross checks of "The settings checks": the pairs whose first must be no greater than the
# second, the two that can't both be true, and the crop's edges, whose first must be below the
# second.
NO_GREATER = (
    ("sift_extraction_dsp_min_scale", "sift_extraction_dsp_max_scale"),
    ("view_graph_calibrator_min_focal_length_ratio",
     "view_graph_calibrator_max_focal_length_ratio"),
    ("mapper_min_focal_length_ratio", "mapper_max_focal_length_ratio"),
    ("global_mapper_track_min_num_views_per_track", "global_mapper_track_max_num_views_per_track"),
    ("undistortion_min_scale", "undistortion_max_scale"),
)
NOT_BOTH_TRUE = ("global_mapper_ba_skip_fixed_rotation_stage",
                 "global_mapper_ba_skip_joint_optimization_stage")
BELOW = (("undistortion_roi_min_x", "undistortion_roi_max_x"),
         ("undistortion_roi_min_y", "undistortion_roi_max_y"))


@st.composite
def accepted_settings(draw: st.DrawFn) -> dict[str, Any]:
    """Every setting of [cameras] and the Preset's near_overlap, as Config gives them, each drawn
    from its Range in RANGES, so read_settings accepts them all.

    Then the two settings of each cross check are put in order: each NO_GREATER pair is sorted,
    the second of NOT_BOTH_TRUE is false when both are drawn true, and each crop's edges are
    sorted, with the lower moved to the float just below the higher when they're drawn equal.
    """
    values = {key: draw(strategy) for key, strategy in RANGES.items()}
    for low, high in NO_GREATER:
        values[low], values[high] = sorted((values[low], values[high]))
    first, second = NOT_BOTH_TRUE
    if values[first] and values[second]:
        values[second] = False
    for low, high in BELOW:
        values[low], values[high] = sorted((values[low], values[high]))
        if values[low] == values[high]:  # both the same float between 0 and 1
            values[low] = math.nextafter(values[high], -math.inf)
    return values


QUADRATIC_OVERLAP = "--SequentialMatching.quadratic_overlap"
# The options that Req 2.6 keeps off every command line: --help, --project_path, --log_path, a GPU
# index, an image list, a vocabulary tree, a loop detection setting, and every option of ALIKED,
# LoMa and LightGlue.
KEPT_OFF = re.compile(r"^--(?:help|project_path|log_path)$|gpu_index|image_list_path|vocab_tree"
                      r"|^--SequentialMatching\.loop_detection_|aliked|loma|lightglue",
                      re.IGNORECASE)


def same(back: object, value: object) -> bool:
    """Whether `back`, read back from a command line, is `value`: of the same type and equal, and
    for a float with the same bits, so -0.0 can't pass for 0.0."""
    if type(back) is not type(value):
        return False
    if isinstance(back, float) and isinstance(value, float):
        return back.hex() == value.hex()
    return back == value


def line_options(run: ColmapRun, name: str) -> list[tuple[str, str]]:
    """Each option on the run `name`'s command line with its value's text, in order, once the
    line is checked to start with colmap and its command and then go on in pairs: an option,
    which starts with --, then its value, which doesn't."""
    assert run.args[:2] == ("colmap", RUN_COMMANDS[name]), name
    words = run.args[2:]
    pairs = list(zip(words[::2], words[1::2], strict=True))
    assert [(option, text) for option, text in pairs
            if not option.startswith("--") or text.startswith("--")] == [], name
    return pairs


def wanted_options(block: Sequence[ColmapOption], name: str, camera_model: str,
                   mask_used: bool) -> set[str]:
    """The options that the run `name`'s command line must hold: those of `block`, its command's
    block in options.md, but --ImageReader.camera_mask_path only with mask_used (Req 3.5, 3.6),
    and --ImageReader.camera_params too in RADIAL_FISHEYE's Feature_Extraction, though the block
    leaves it out, and never for RADIAL (Req 3.3, 3.4)."""
    wanted = {option.option for option in block if mask_used or option.option != CAMERA_MASK}
    if name == "feature_extractor" and camera_model == "RADIAL_FISHEYE":
        wanted.add(CAMERA_PARAMS)
    return wanted


# π to 36 digits, so the oracle's f is exact to far below a float's last bit.
PI = Fraction("3.14159265358979323846264338327950288")
LARGEST = Fraction(BIGGEST)
F_TOLERANCE = Fraction(1, 10**12)  # how far f may lie from the oracle's, relatively


def wanted_fisheye_start(width: int, height: int,
                         fov_degrees: int | float) -> tuple[Fraction, ...]:
    """The Fisheye_Start of Picks `width` by `height` pixels whose field of view across the long
    edge is `fov_degrees`, worked out here in exact fractions, as the design gives it: f, cx,
    cy, k1, k2 with f = (L / 2) / (θ / 2), L the long edge and θ the field of view in radians,
    cx = width / 2, cy = height / 2 and k1 = k2 = 0 (Req 3.4, F2, D19)."""
    half_theta = Fraction(fov_degrees) * PI / 180 / 2
    f = Fraction(max(width, height), 2) / half_theta
    return (f, Fraction(width, 2), Fraction(height, 2), Fraction(0), Fraction(0))


def near_f(back: float, wanted: Fraction) -> bool:
    """Whether `back`, an f read back from a command line, is within F_TOLERANCE of `wanted`,
    relatively, where infinity stands for any number above the largest float."""
    tolerance = wanted * F_TOLERANCE
    if not math.isfinite(back):
        return back == math.inf and wanted + tolerance > LARGEST
    return abs(Fraction(back) - wanted) <= tolerance


def wanted_value(option: ColmapOption, name: str, values: Mapping[str, Any]) -> object:
    """What `option` of the run `name`'s block must read back as for the drawn `values`, unless
    it's a path.

    A Fixed_Constant is its text in options.md, but --SequentialMatching.quadratic_overlap is 0
    in the Near_Match_Run (A1). --SequentialMatching.overlap is near_overlap in the
    Near_Match_Run and far_overlap in the Far_Match_Run (Req 4.1, 4.2). An Option_Key is its
    drawn value (Req 2.3, 2.5).
    """
    if option.sort == "fixed":
        return "0" if option.option == QUADRATIC_OVERLAP and name == NEAR else option.text
    if option.sort == "overlap":
        return values["near_overlap"] if name == NEAR else values["far_overlap"]
    return values[option.key]


def checked_line(run: ColmapRun, name: str, block: Sequence[ColmapOption],
                 values: Mapping[str, Any], mask_used: bool) -> dict[str, str]:
    """Check the run `name`'s command line against `block`, its command's block in options.md,
    for the drawn `values` and `mask_used`, and give each of its options with its value's text.

    The line holds each option of wanted_options once, and no other option, so none that Req 2.6
    keeps off. Each option but a path reads back as its wanted_value: a bool from 1 or 0, an int
    from a whole number in decimal, a float from its text, and a name as it is (Req 2.4).
    """
    pairs = line_options(run, name)
    counts = Counter(option for option, _ in pairs)
    assert [option for option, count in counts.items() if count > 1] == [], name
    assert set(counts) == wanted_options(block, name, values["camera_model"], mask_used), name
    assert [option for option in counts if KEPT_OFF.search(option)] == [], name
    line = dict(pairs)
    for option in block:
        if option.sort != "path":  # a path only has to be there once
            wanted = wanted_value(option, name, values)
            text = line[option.option]
            assert same(read_back(text, type(wanted)), wanted), (name, option.option, text)
    return line


@pytest.fixture(scope="module")
def job_paths(tmp_path_factory: pytest.TempPathFactory) -> Paths:
    """The Paths of an empty Job folder, which Property 1's examples share. The builders only
    name files, so nothing is made in it."""
    job = tmp_path_factory.mktemp("job")
    return Paths.of(job / "work" / "cameras", job / "work" / "frames", job / "out")


@given(values=accepted_settings(), preset=st.sampled_from(PRESET_NAMES),
       mask_used=st.booleans(), width=st.integers(1, 4000), height=st.integers(1, 4000))
def test_command_lines_hold_the_fixed_constants_and_every_option_once(
        colmap_options: Mapping[str, tuple[ColmapOption, ...]], job_paths: Paths,
        values: dict[str, Any], preset: str, mask_used: bool, width: int, height: int) -> None:
    """Feature: cameras, Property 1: Command lines hold the Fixed_Constants and every option once

    **Validates: Requirements 2.3, 2.4, 2.5, 2.6, 2.13, 3.2, 3.3, 3.5, 3.6, 4.1, 4.2, 4.3, 8.3**
    """
    # The Option_Keys drawn are exactly those of options.md.
    assert OPTION_RANGES.keys() == {option.key for block in colmap_options.values()
                                    for option in block if option.sort == "key"}
    cfg = read_settings(StepSettings(values, section="cameras", preset=preset), "cameras", preset)
    frames = FramesInput(folder=job_paths.frames, pick_width=width, pick_height=height,
                         mask_used=mask_used, picks=(Pick(number=0, file="000000.jpg"),))
    env = colmap_env(job_paths.tmp, {"PATH": "/usr/bin:/bin"})
    runs = eight_runs(cfg, frames, job_paths, env)
    assert {name: run.command for name, run in runs.items()} == RUN_COMMANDS
    lines = {name: checked_line(run, name, colmap_options[run.command], values, mask_used)
             for name, run in runs.items()}
    # checked_line kept --ImageReader.camera_params to RADIAL_FISHEYE's Feature_Extraction. There
    # its five numbers read back as the oracle's Fisheye_Start: cx, cy, k1 and k2 bit for bit, and
    # f within F_TOLERANCE (Req 2.4, 3.3, 3.4).
    if values["camera_model"] == "RADIAL_FISHEYE":
        text = lines["feature_extractor"][CAMERA_PARAMS]
        numbers = [read_back(number, float) for number in text.split(",")]
        f, *rest = wanted_fisheye_start(width, height, values["fisheye_fov_degrees"])
        assert len(numbers) == 5, text
        assert near_f(numbers[0], f), (text, f)
        assert all(same(back, float(value)) for back, value in zip(numbers[1:], rest)), text
    # Both undistorters give each of the eleven undistortion settings the same value (Req 2.5,
    # 8.3).
    undistortion = [option.option for option in colmap_options["image_undistorter"]
                    if option.sort == "key" and option.key.startswith("undistortion_")]
    assert len(undistortion) == 11
    assert ([lines["image_undistorter"][option] for option in undistortion]
            == [lines["image_undistorter_standalone"][option] for option in undistortion])


# format_value and fisheye_start (Req 2.4, 3.4)

@pytest.mark.parametrize(("value", "text"), [
    pytest.param(True, "1", id="true"),
    pytest.param(False, "0", id="false"),
    pytest.param(0.02 / 3, "0.006666666666666667", id="0.02 over 3"),
    pytest.param(1e-05, "1e-05", id="1e-05"),
])
def test_format_value_writes_each_value_so_it_reads_back_as_the_same_value(
        value: object, text: str) -> None:
    assert format_value(value) == text  # so a bool is never True or False
    back = read_back(text, type(value))
    assert (type(back), back) == (type(value), value)


def test_fisheye_start_gives_f_from_half_the_long_edge_and_half_the_field_of_view() -> None:
    start = fisheye_start(1600, 900, 120.0)
    f = 800 / (math.pi / 3)  # half the long edge, over half of 120° in radians
    # RADIAL_FISHEYE's five parameters, f, cx, cy, k1, k2 (F2, D19).
    assert len(start) == 5
    assert start[0] == pytest.approx(f, rel=1e-12)
    assert round(start[0], 3) == 763.944
    assert start[1:] == (800, 450, 0, 0)  # cx and cy at the centre, then k1 and k2
    # The text --ImageReader.camera_params gets reads back as the same five floats (Req 2.4).
    text = ",".join(map(format_value, start))
    assert [float(number).hex() for number in text.split(",")] == [
        float(value).hex() for value in start]
    # Upright Picks: the long edge is the height, so f stays and the centre turns with the size.
    assert fisheye_start(900, 1600, 120.0) == (start[0], 450, 800, 0, 0)


# colmap_env and ColmapRun.value (Req 2.2)

def test_colmap_env_drops_test_tmpdir_and_the_glog_keys_and_sets_tmpdir_and_tmp(
        tmp_path: Path) -> None:
    base = {"PATH": "/usr/bin:/bin", "HOME": "/Users/someone", "TEST_TMPDIR": "/tmp/bazel",
            "GLOG_log_dir": "/tmp/glog", "GLOG_logtostderr": "1", "TMPDIR": "/var/tmp",
            "TMP": "/var/tmp"}
    before = dict(base)
    tmp = tmp_path / "colmap" / "tmp"
    assert colmap_env(tmp, base) == {"PATH": "/usr/bin:/bin", "HOME": "/Users/someone",
                                     "TMPDIR": str(tmp), "TMP": str(tmp)}
    assert base == before  # the step's own environment stays as it is


def colmap_run(tmp_path: Path, args: Sequence[str] | None = None, *,
               timeout_seconds: float = 60, log: Path | None = None,
               env: Mapping[str, str] | None = None) -> ColmapRun:
    """A run of COLMAP's mapper that works in tmp_path, with its Command_Log there, or a run of
    `args` in its place. Its Time_Limit is `timeout_seconds`, and its environment is `env`, or a
    small one of its own."""
    database = str(tmp_path / "database.db")
    return ColmapRun(
        name="mapper", command="mapper",
        args=tuple(args) if args is not None else (
            "colmap", "mapper", "--database_path", database, "--Mapper.init_image_id1", "-1"),
        timeout_seconds=timeout_seconds, timeout_key="mapper_timeout_seconds",
        log=log if log is not None else tmp_path / "mapper.log", cwd=tmp_path,
        env=env if env is not None else {"PATH": "/usr/bin:/bin", "TMPDIR": str(tmp_path)})


def test_colmap_run_value_gives_the_argument_after_an_option(tmp_path: Path) -> None:
    run = colmap_run(tmp_path)
    assert run.value("--database_path") == str(tmp_path / "database.db")
    assert run.value("--Mapper.init_image_id1") == "-1"


# capped (Req 17.3 to 17.5, C13 to C15)

# The Time_Limit that the capped tests' run gets from its key, and the Cap_Deadline, in UTC as
# read_cap_deadline gives it, with its text in ISO 8601.
KEY_SECONDS = 600
CAP_DEADLINE = datetime(2026, 11, 23, 5, 0, tzinfo=timezone.utc)
CAP_DEADLINE_TEXT = "2026-11-23T05:00:00+00:00"


def keyed_run(tmp_path: Path) -> ColmapRun:
    """The Calibration's run, as its builder gives it for the real config with the Preset default
    and view_graph_calibrator_timeout_seconds at KEY_SECONDS, for a Job in tmp_path/job, which
    isn't made."""
    job = tmp_path / "job"
    paths = Paths.of(job / "work" / "cameras", job / "work" / "frames", job / "out")
    cfg = real_settings("default", view_graph_calibrator_timeout_seconds=KEY_SECONDS)
    run = commands.view_graph_calibrator(cfg, paths,
                                         colmap_env(paths.tmp, {"PATH": "/usr/bin:/bin"}))
    assert (run.timeout_seconds, run.capped_at) == (KEY_SECONDS, None)
    return run


def test_capped_gives_the_run_as_it_is_with_no_cap_deadline(tmp_path: Path) -> None:
    run = keyed_run(tmp_path)
    assert capped(run, None, CAP_DEADLINE) == run  # with no Cap_Deadline, the time is never read


@pytest.mark.parametrize(("left", "timeout_seconds", "capped_at"), [
    pytest.param(timedelta(seconds=KEY_SECONDS), KEY_SECONDS, None,
                 id="exactly the key's value left"),
    pytest.param(timedelta(seconds=KEY_SECONDS - 1), KEY_SECONDS - 1, CAP_DEADLINE,
                 id="a second less"),
    pytest.param(timedelta(microseconds=1), 1e-6, CAP_DEADLINE, id="a microsecond left"),
])
def test_capped_gives_the_smaller_of_the_keys_value_and_the_seconds_left(
        tmp_path: Path, left: timedelta, timeout_seconds: float,
        capped_at: datetime | None) -> None:
    run = keyed_run(tmp_path)
    got = capped(run, CAP_DEADLINE, CAP_DEADLINE - left)
    # Only the Time_Limit and capped_at may change. Every other field stays as it was.
    assert got == replace(run, timeout_seconds=timeout_seconds, capped_at=capped_at)


@pytest.mark.parametrize("left", [
    pytest.param(timedelta(0), id="exactly 0 s left"),
    pytest.param(timedelta(minutes=-1), id="a minute past"),
])
def test_capped_fails_naming_the_command_and_the_cap_deadline_when_no_time_is_left(
        tmp_path: Path, left: timedelta) -> None:
    with pytest.raises(ExpectedFailure) as caught:
        capped(keyed_run(tmp_path), CAP_DEADLINE, CAP_DEADLINE - left)
    message = readable(caught.value)
    # The command, the Cap_Deadline, a larger cap_minutes under [cloud.presets], and run the Job
    # again (Req 17.5, C15).
    for words in ("COLMAP's view_graph_calibrator", CAP_DEADLINE_TEXT, "a larger cap_minutes",
                  "[cloud.presets]", "run the job again"):
        assert words in message


# Property 8: The Cap_Deadline caps each Time_Limit, and no time left starts nothing (Req 17.3 to
# 17.5, 17.8)

# A Time_Limit key's value from 1e-6 to 1e6 s, the two ends included: whole seconds, as a TOML
# integer gives them, a whole number of microseconds, so an offset of the key's value leaves
# exactly that many seconds, and any float.
KEY_VALUES = (st.sampled_from((1e-6, 1e6)) | st.integers(1, 10**6)
              | st.integers(1, 10**12).map(lambda micros: micros / 10**6)
              | st.floats(1e-6, 1e6))
# A time now from 2000 to 2100, in UTC with microseconds, as utc_now gives it.
NOWS = st.datetimes(datetime(2000, 1, 1), datetime(2100, 1, 1), timezones=st.just(timezone.utc))
# An offset from -2e6 to 2e6 s in microsecond steps.
OFFSETS = st.integers(-2 * 10**12, 2 * 10**12).map(lambda micros: timedelta(microseconds=micros))


@st.composite
def cap_cases(draw: st.DrawFn) -> tuple[float, datetime, datetime | None]:
    """A Time_Limit key's value, a time now, and a Cap_Deadline: None, or now plus an offset from
    OFFSETS, with an offset of exactly the key's value and of 0 each drawn about a quarter of the
    time."""
    key = draw(KEY_VALUES)
    now = draw(NOWS)
    offset = draw(st.none() | st.just(timedelta(seconds=key)) | st.just(timedelta(0)) | OFFSETS)
    return key, now, None if offset is None else now + offset


def cap_oracle(run: ColmapRun, deadline: datetime | None, now: datetime) -> ColmapRun | None:
    """What capped should give for `run` with the Cap_Deadline `deadline` at the time `now`,
    worked out here with timedelta arithmetic: the run as it is, the run with the seconds left as
    its Time_Limit and `deadline` as its capped_at, or None for the Req 17.5 failure."""
    if deadline is None:
        return run  # no Cap_Deadline, so the key's value, with no cap (Req 17.4)
    left = deadline - now  # exact, in whole microseconds
    if left <= timedelta(0):
        return None  # no time left, so no start (Req 17.5)
    seconds_left = left / timedelta(seconds=1)
    if seconds_left >= run.timeout_seconds:
        return run  # at least the key's value left (Req 17.3)
    return replace(run, timeout_seconds=seconds_left, capped_at=deadline)  # Req 17.3


@pytest.fixture(scope="module")
def calibration_run(tmp_path_factory: pytest.TempPathFactory) -> ColmapRun:
    """keyed_run's run, which Property 8's examples share. capped only reads it, and nothing is
    made in its Job folder."""
    return keyed_run(tmp_path_factory.mktemp("capped"))


@given(case=cap_cases())
def test_the_cap_deadline_caps_each_time_limit_and_no_time_left_starts_nothing(
        calibration_run: ColmapRun, case: tuple[float, datetime, datetime | None]) -> None:
    """Feature: cameras, Property 8: The Cap_Deadline caps each Time_Limit, and no time left starts nothing

    **Validates: Requirements 17.3, 17.4, 17.5, 17.8**
    """
    key, now, deadline = case
    run = replace(calibration_run, timeout_seconds=key)
    wanted = cap_oracle(run, deadline, now)
    if wanted is not None:
        # The run as it is, or with only its Time_Limit and capped_at changed.
        assert capped(run, deadline, now) == wanted
        return
    assert deadline is not None
    with pytest.raises(ExpectedFailure) as caught:
        capped(run, deadline, now)
    message = readable(caught.value)
    # The Req 17.5 failure names the command and gives the Cap_Deadline in ISO 8601, in UTC.
    assert f"COLMAP's {run.command}" in message
    assert deadline.isoformat() in message


# run_colmap with a stand-in for subprocess.run (Req 2.2, 2.7, 2.8, 2.10, 11.12)

# What the stand-in writes to the Command_Log it's given: two of glog's lines (F16). The second
# holds b"\xe9", é in Windows-1252, which isn't UTF-8, so it's read back as U+FFFD.
STAND_IN_LOG = (b"I20261123 04:21:05.000001 0x1705ab000 database.cc:211] Loading database\n"
                b"E20261123 04:21:07.318502 0x1705ab000 database.cc:248] Can't open "
                b"/jobs/caf\xe9/database.db\n")
STAND_IN_LAST_LINE = "Can't open /jobs/caf\ufffd/database.db"

Call = tuple[tuple[Any, ...], dict[str, Any]]  # a call's positional and keyword arguments


def replace_run(monkeypatch: pytest.MonkeyPatch, outcome: int | BaseException,
                output: bytes = STAND_IN_LOG) -> list[Call]:
    """Replace subprocess.run, as splatpipe.cameras.colmap sees it, with a stand-in that records
    each call and writes `output` to the file it's given as stdout. Then it raises `outcome` if
    it's an exception, or else gives back a CompletedProcess with `outcome` as its exit code.
    Gives the list of calls."""
    calls: list[Call] = []

    def stand_in(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        calls.append((args, kwargs))
        kwargs["stdout"].write(output)
        if isinstance(outcome, BaseException):
            raise outcome
        return subprocess.CompletedProcess(args[0], outcome)

    monkeypatch.setattr("splatpipe.cameras.colmap.subprocess.run", stand_in)
    return calls


def test_run_colmap_runs_the_argument_list_with_stdin_closed_and_all_its_output_in_its_log(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = replace_run(monkeypatch, 1)
    run = colmap_run(tmp_path)
    result = run_colmap(run)
    [(args, kwargs)] = calls
    assert args == (list(run.args),)  # one argument list, so no shell
    log = kwargs.pop("stdout")  # the Command_Log, opened for writing in binary, closed after
    assert (Path(log.name), log.mode, log.closed) == (run.log, "wb", True)
    assert kwargs == {"stdin": subprocess.DEVNULL, "stderr": subprocess.STDOUT, "cwd": run.cwd,
                      "env": dict(run.env), "timeout": run.timeout_seconds}
    assert run.log.read_bytes() == STAND_IN_LOG
    assert (result.exit_code, result.last_line) == (1, STAND_IN_LAST_LINE)
    assert result.seconds >= 0


def test_run_colmap_gives_no_exit_code_when_colmap_is_stopped_at_its_time_limit(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run = colmap_run(tmp_path)
    # subprocess.run kills COLMAP at its timeout and waits for it, then raises TimeoutExpired.
    replace_run(monkeypatch, subprocess.TimeoutExpired(list(run.args), run.timeout_seconds))
    result = run_colmap(run)
    assert result.exit_code is None
    assert result.last_line == STAND_IN_LAST_LINE  # the log keeps what it wrote before
    assert run.log.read_bytes() == STAND_IN_LOG


def test_run_colmap_fails_saying_colmap_isnt_installed_when_the_program_is_gone(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    replace_run(monkeypatch, FileNotFoundError(2, "No such file or directory", "colmap"),
                output=b"")
    with pytest.raises(ExpectedFailure) as caught:
        run_colmap(colmap_run(tmp_path))
    message = readable(caught.value)
    for words in ("COLMAP isn't installed", "Install COLMAP 4.2.1", ".kiro/steering/tech.md",
                  "run the step again"):
        assert words in message


def test_run_colmap_fails_naming_the_log_and_the_reason_when_its_log_cant_be_opened(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = replace_run(monkeypatch, 0)
    run = colmap_run(tmp_path, log=tmp_path / "logs" / "mapper.log")  # there's no logs/
    with pytest.raises(ExpectedFailure) as caught:
        run_colmap(run)
    message = readable(caught.value)
    assert str(run.log) in message
    assert "No such file or directory" in message
    assert "isn't installed" not in message  # the log's FileNotFoundError isn't a missing colmap
    assert calls == []  # COLMAP never started


# run_colmap with a real child (Req 2.8, 2.10)

LIMIT = 0.2  # the real child's Time_Limit, in seconds
CHILD_SECONDS = 30  # how long the real child would sleep if nothing stopped it
CHILD_LINE = "Processed file [1/8]"


def test_a_real_child_is_stopped_at_its_time_limit_with_its_line_in_the_command_log(
        tmp_path: Path) -> None:
    # Like COLMAP under --log_target stderr, the child writes its line to stderr.
    script = (f"import sys, time; print({CHILD_LINE!r}, file=sys.stderr, flush=True); "
              f"time.sleep({CHILD_SECONDS})")
    tmp = tmp_path / "tmp"
    tmp.mkdir()
    run = colmap_run(tmp_path, (sys.executable, "-c", script), timeout_seconds=LIMIT,
                     env=colmap_env(tmp, os.environ))
    result = run_colmap(run)
    assert result.exit_code is None
    # It ran for about its Time_Limit, far less than the child would have slept.
    assert LIMIT / 2 < result.seconds < CHILD_SECONDS
    assert run.log.read_text(encoding="utf-8").splitlines() == [CHILD_LINE]
    assert result.last_line == CHILD_LINE


# last_error_line and exit_text (C3, F16)

# One line in each of glog's prefix forms (F16), with its text once the prefix is dropped: the
# level letter, the date with its year or without, the time, the thread id in hex or in decimal,
# padded with spaces to 5 characters, and file:line].
GLOG_LINES = [
    pytest.param("I20261123 04:19:19.624758 0x1705ab000 feature_extractor.cc:303] "
                 "Processed file [8/8]", "Processed file [8/8]", id="8-digit date, hex thread"),
    pytest.param("E20261123 04:21:07.318502 140230119581504 sfm.cc:153] "
                 "Failed to create sparse model", "Failed to create sparse model",
                 id="8-digit date, decimal thread"),
    pytest.param("W1123 04:20:41.000250 0x16fb87000 view_graph_calibration.cc:212] "
                 "1 cameras rejected in view graph calibration",
                 "1 cameras rejected in view graph calibration", id="4-digit date, hex thread"),
    pytest.param("E1123 04:20:41.731900    42 sfm.cc:240] View graph calibration failed",
                 "View graph calibration failed", id="4-digit date, decimal thread"),
]


@pytest.mark.parametrize(("line", "text"), GLOG_LINES)
def test_last_error_line_gives_the_last_line_that_isnt_empty_without_glogs_prefix(
        line: str, text: str) -> None:
    log = f"I20261123 04:19:18.000001 0x1705ab000 colmap.cc:45] Starting\n{line}\n\n  \n"
    assert last_error_line(io.StringIO(log)) == text


def test_last_error_line_stops_at_a_crashs_stack_trace_and_gives_the_failed_check() -> None:
    log = ("I20261123 04:21:05.000001 0x1705ab000 global_mapper.cc:120] Running global "
           "positioning\n"
           "F20261123 04:21:07.318502 0x1705ab000 reconstruction.cc:418] Check failed: "
           "existing_frame_ids.count(frame_id) > 0\n"
           "*** Check failure stack trace: ***\n"
           "    @        0x104f2c8a4  google::LogMessage::Fail()\n"
           "    @        0x104f2b9d0  google::LogMessage::SendToLog()\n"
           "    @        0x19b8e1234  abort\n")
    assert last_error_line(io.StringIO(log)) == ("Check failed: "
                                                 "existing_frame_ids.count(frame_id) > 0")


@pytest.mark.parametrize("log", ["", "\n  \n"], ids=["no lines", "only empty lines"])
def test_last_error_line_of_an_empty_log_is_empty(log: str) -> None:
    assert last_error_line(io.StringIO(log)) == ""


@pytest.mark.parametrize(("code", "text"), [
    pytest.param(1, "exit code 1", id="exit code 1"),
    # subprocess gives -6 for a COLMAP that SIGABRT stopped, as a crash does (D8). Signal 6 is
    # SIGABRT on Linux and macOS, where COLMAP runs.
    pytest.param(-6, "exit code -6 (signal SIGABRT)", id="SIGABRT",
                 marks=pytest.mark.skipif(sys.platform == "win32",
                                          reason="signal 6 isn't SIGABRT on Windows")),
])
def test_exit_text_gives_the_exit_code_and_the_signal_that_stopped_colmap(
        code: int, text: str) -> None:
    assert exit_text(code) == text


# The colmap <command> -h test (Req 16.7, 16.8, 16.13, 18.3, A16, C20)

def real_smoke_sizes() -> smoke.SmokeSizes:
    """The SmokeSizes of the real config's [cloud.smoke], read by path, as cloud/smoke_runner.py
    builds them: its seven cameras_* numbers and check_timeout_seconds, each decimal as a float,
    as cloud/config.py reads it."""
    table = tomllib.loads(REAL_CONFIG.read_text(encoding="utf-8"))["cloud"]["smoke"]
    return smoke.SmokeSizes(
        frames=table["cameras_frames"], width=table["cameras_width"],
        height=table["cameras_height"], focal_pixels=float(table["cameras_focal_pixels"]),
        placed_share=float(table["cameras_placed_share"]),
        focal_tolerance=float(table["cameras_focal_tolerance"]),
        timeout_seconds=float(table["cameras_timeout_seconds"]),
        help_timeout_seconds=float(table["check_timeout_seconds"]))


@pytest.mark.skipif(shutil.which("colmap") is None, reason="COLMAP isn't installed")
def test_colmap_help_lists_every_option_the_builders_pass_each_command(tmp_path: Path) -> None:
    # The Option_Check: each of the seven colmap <command> -h runs within check_timeout_seconds,
    # 60 s, and lists every option the builders pass that command for the real config, for each
    # Preset, both Camera_Models and with and without the Mask. Any failure raises.
    smoke.check_options(load_config(REAL_CONFIG), real_smoke_sizes(), tmp_path / smoke.HELP)
