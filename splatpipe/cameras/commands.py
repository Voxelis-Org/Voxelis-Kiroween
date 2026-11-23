"""The camera solve's COLMAP command lines: one builder per COLMAP command, each giving the
ColmapRun that colmap.run_colmap runs.

Paths.of gives every path the step uses, in work/cameras/, work/frames/ and out/, from the three
folders resolved, so every path on a command line is absolute (Req 2.2). The COLMAP_Folder,
work/cameras/colmap/, holds the Database, the Calibration_Copy, each mapper's Models, the
Undistortion's and the Mask_Warp's output, the Mask_Warp's camera line, TMPDIR and the
Command_Logs (O8).

Each builder gives PROGRAM, the command, its paths, the four log options and
--default_random_seed, then the command's options in the order of its block in
references/options.md in the colmap-cli skill. Each Fixed_Constant goes where the block has it, at
its fixed value. Every other option is an Option_Key, at its value in [cameras]: a builder takes
exactly its command's keys from OPTION_KEYS, each once, so no COLMAP default decides anything
(Req 2.3, 2.5). format_value writes each value so that COLMAP reads the same value (Req 2.4), and
the two undistorters take the same eleven undistortion_ keys (Req 8.3).

The Fixed_Constants keep extraction and matching to SIFT and SIFT_BRUTEFORCE on the CPU, with
loop detection off, so nothing is downloaded and SiftGPU never runs (Req 2.6), and keep both
mappers' solvers on the CPU (D4, O7). The two Match_Runs differ only in their pairing: the
Near_Match_Run pairs each Pick with the next near_overlap Picks, and the Far_Match_Run, with
quadratic_overlap on, with the Picks 2^k ahead for k below far_overlap (Req 4.1 to 4.3, A1). Only
the Feature_Extraction has options that come and go: --ImageReader.camera_params with the
Fisheye_Start for OPENCV_FISHEYE (Req 3.3, 3.4), placed where COLMAP's -h lists it, and
--ImageReader.camera_mask_path with the Mask when the Frames_Record has mask_used (Req 3.5, 3.6).
No builder passes --help, --project_path, --log_path, a GPU index, an image list, a vocabulary
tree, a --SequentialMatching.loop_detection_* setting, or an ALIKED, LoMa or LightGlue option.

The Feature_Extraction reads the Grey_Copies, and both mappers and the Undistortion read the
Colour_Picks, which have the same names (Req 1.5). global_mapper reads the database it's given, so
the solve flow chooses the Calibration_Copy or the Database (Req 6.1, 6.2), and mapper always
reads the Database (Req 6.6). Each ColmapRun's name, Time_Limit and Time_Limit key, Command_Log,
working folder and environment come from its command and the Paths.

From splatpipe, this module imports only splatpipe.frames.files and the step's settings,
frames_input and colmap.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from splatpipe.cameras.colmap import PROGRAM, ColmapRun
from splatpipe.cameras.frames_input import FramesInput
from splatpipe.cameras.settings import OPTION_KEYS, TIMEOUT_KEYS, CamerasSettings, OptionValue
from splatpipe.frames.files import GREY, IMAGES, MASK

# The Camera_Model whose camera starts from the Fisheye_Start (D5).
OPENCV_FISHEYE = "OPENCV_FISHEYE"

# The Fixed_Constants of the command lines (D4), each as the words it adds. Every command takes
# the four log options: all of COLMAP's output on stderr, every INFO line, and no colour codes, so
# the Command_Log holds every line as plain text.
LOG_OPTIONS = ("--log_target", "stderr", "--log_level", "0", "--log_severity", "0",
               "--log_color", "0")
# feature_extractor: one camera for every Pick, and SIFT on the CPU, which needs no model weights
# and keeps SiftGPU's non-commercial code from running.
CAMERA_MODE = ("--camera_mode", "1")
EXTRACTION_TYPE = ("--FeatureExtraction.type", "SIFT")
EXTRACTION_GPU = ("--FeatureExtraction.use_gpu", "0")
# sequential_matcher: SIFT_BRUTEFORCE on the CPU, which needs no model weights, and no loop
# detection, which would need a vocabulary tree that COLMAP downloads. The Near_Match_Run pairs
# in a line and the Far_Match_Run by powers of two (D6, A1).
MATCHING_TYPE = ("--FeatureMatching.type", "SIFT_BRUTEFORCE")
MATCHING_GPU = ("--FeatureMatching.use_gpu", "0")
LOOP_DETECTION = ("--SequentialMatching.loop_detection", "0")
NEAR_PAIRING = ("--SequentialMatching.quadratic_overlap", "0")
FAR_PAIRING = ("--SequentialMatching.quadratic_overlap", "1")
# global_mapper and mapper: the build has no CUDA, so the solvers run on the CPU, and bundle
# adjustment uses Ceres, COLMAP 4.2.1's only backend that needs no CUDA (O7).
GLOBAL_POSITIONING_GPU = ("--GlobalMapper.gp_use_gpu", "0")
GLOBAL_BUNDLE_ADJUSTMENT = ("--GlobalMapper.ba_backend", "CERES",
                            "--GlobalMapper.ba_ceres_use_gpu", "0")
MAPPER_BUNDLE_ADJUSTMENT = ("--Mapper.ba_use_gpu", "0", "--Mapper.ba_local_backend", "CERES",
                            "--Mapper.ba_global_backend", "CERES")
# image_undistorter: COLMAP's own layout, images/ and sparse/, which the step reads (A1).
OUTPUT_TYPE = ("--output_type", "COLMAP")


@dataclass(frozen=True)
class Paths:
    """Every path the step uses, each absolute, laid out as the design's "The Work_Folder" says."""

    frames: Path  # work/frames/, the Frames_Step's Work_Folder
    colmap: Path  # work/cameras/colmap/, the COLMAP_Folder, every COLMAP_Command's working folder
    tmp: Path  # colmap/tmp/, TMPDIR and TMP for every COLMAP_Command (A10)
    logs: Path  # colmap/logs/, one Command_Log per COLMAP_Command
    database: Path  # colmap/database.db, the Database
    calibrated: Path  # colmap/database_calibrated.db, the Calibration_Copy
    global_models: Path  # colmap/global/, the Global_Run's Models
    incremental_models: Path  # colmap/incremental/, the Incremental_Run's Models
    undistorted: Path  # colmap/undistorted/, the Undistortion's output
    mask_camera: Path  # colmap/mask_camera.txt, the Mask_Warp's camera line
    mask_warped: Path  # colmap/mask_undistorted/, the Mask_Warp's output
    images: Path  # work/cameras/images/, the Training_Images
    masks: Path  # work/cameras/masks/, the Training_Masks
    sparse: Path  # work/cameras/sparse/, the Levelled_Model
    record: Path  # work/cameras/solve.json, the Solve_Record
    viewer: Path  # out/cameras.json, the Viewer_Cameras

    @classmethod
    def of(cls, work: Path, frames: Path, out: Path) -> Paths:
        """The Paths of the Job whose Cameras_Step Work_Folder is `work`, whose Frames_Step
        Work_Folder is `frames` and whose out folder is `out`.

        The three folders are resolved, so every path is absolute. Nothing on disk is made or
        changed.
        """
        work, frames, out = work.resolve(), frames.resolve(), out.resolve()
        colmap = work / "colmap"
        return cls(frames=frames, colmap=colmap, tmp=colmap / "tmp", logs=colmap / "logs",
                   database=colmap / "database.db",
                   calibrated=colmap / "database_calibrated.db",
                   global_models=colmap / "global", incremental_models=colmap / "incremental",
                   undistorted=colmap / "undistorted", mask_camera=colmap / "mask_camera.txt",
                   mask_warped=colmap / "mask_undistorted", images=work / "images",
                   masks=work / "masks", sparse=work / "sparse", record=work / "solve.json",
                   viewer=out / "cameras.json")


def format_value(value: OptionValue) -> str:
    """`value` as text that COLMAP reads as the same value (Req 2.4, A2).

    A bool is 1 or 0, an int is in decimal, a float is its repr, the shortest text that reads back
    as the same double, like 0.006666666666666667 or 1e-05, and a name is as it is.
    """
    if isinstance(value, bool):  # first, because a bool is an int too
        return "1" if value else "0"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    return value


def fisheye_start(width: int, height: int, fov_degrees: float) -> tuple[float, ...]:
    """The Fisheye_Start of Picks `width` by `height` pixels whose field of view across the long
    edge is `fov_degrees`, in OPENCV_FISHEYE's order fx, fy, cx, cy, k1, k2, k3, k4 (Req 3.4, F2).

    fx = fy = (L / 2) / (θ / 2), with L the long edge and θ the field of view in radians. It's
    worked out as 180·L / (π·fov_degrees), the same formula with the halves cancelled and the
    turn into radians folded into the one division. So every field of view the settings accept
    gives a float: one so small that its radians round to 0, like 5e-324°, gives an infinite f
    rather than a ZeroDivisionError. The principal point is the Pick's centre, and k1 to k4 are 0.
    """
    f = 180 * max(width, height) / (math.pi * fov_degrees)
    return (f, f, width / 2, height / 2, 0.0, 0.0, 0.0, 0.0)


def feature_extractor(cfg: CamerasSettings, frames: FramesInput, paths: Paths,
                      env: Mapping[str, str]) -> ColmapRun:
    """The Feature_Extraction: SIFT on the Grey_Copies, work/frames/grey/, into a new Database,
    with one camera of the Camera_Model for every Pick (Req 1.5, 3.1, 3.2).

    For OPENCV_FISHEYE, --ImageReader.camera_params gives the camera the Fisheye_Start of the
    Pick_Size, joined with commas. For OPENCV it's left out, so the focal length starts from
    --ImageReader.default_focal_length_factor (Req 3.3, 3.4). With mask_used,
    --ImageReader.camera_mask_path gives the Mask, work/frames/mask.png, and without it, it's left
    out (Req 3.5, 3.6).
    """
    camera_params: tuple[str, ...] = ()
    if cfg.camera_model == OPENCV_FISHEYE:
        start = fisheye_start(frames.pick_width, frames.pick_height, cfg.fisheye_fov_degrees)
        camera_params = ("--ImageReader.camera_params", ",".join(map(format_value, start)))
    mask: tuple[str, ...] = ()
    if frames.mask_used:
        mask = ("--ImageReader.camera_mask_path", str(paths.frames / MASK))
    return _colmap_run(
        cfg, paths, env, command="feature_extractor", name="feature_extractor",
        path_options=(("--database_path", paths.database), ("--image_path", paths.frames / GREY)),
        after={"--default_random_seed": CAMERA_MODE,
               "--ImageReader.existing_camera_id": camera_params,
               "--ImageReader.default_focal_length_factor": (*mask, *EXTRACTION_TYPE),
               "--FeatureExtraction.num_threads": EXTRACTION_GPU})


def sequential_matcher(cfg: CamerasSettings, paths: Paths, env: Mapping[str, str], *,
                       far: bool) -> ColmapRun:
    """The Near_Match_Run, or the Far_Match_Run when `far` is True, on the Database (Req 4.1 to
    4.3).

    The Near_Match_Run pairs each Pick with the next near_overlap Picks, with quadratic_overlap 0.
    The Far_Match_Run pairs each Pick with the Picks 2^k ahead for k below far_overlap, with
    quadratic_overlap 1, and COLMAP matches only the pairs the Database doesn't hold yet. Every
    other option is the same in both runs.
    """
    overlap = cfg.far_overlap if far else cfg.near_overlap
    pairing = FAR_PAIRING if far else NEAR_PAIRING
    return _colmap_run(
        cfg, paths, env, command="sequential_matcher",
        name="sequential_matcher_far" if far else "sequential_matcher_near",
        path_options=(("--database_path", paths.database),),
        after={"--default_random_seed": MATCHING_TYPE,
               "--FeatureMatching.num_threads": MATCHING_GPU,
               "--TwoViewGeometry.random_seed": ("--SequentialMatching.overlap",
                                                 format_value(overlap), *pairing),
               "--SequentialMatching.expand_rig_images": LOOP_DETECTION})


def view_graph_calibrator(cfg: CamerasSettings, paths: Paths,
                          env: Mapping[str, str]) -> ColmapRun:
    """The Calibration, on the Calibration_Copy, so the Database stays as the Match_Runs left it
    (Req 5.1, 5.2)."""
    return _colmap_run(cfg, paths, env, command="view_graph_calibrator",
                       name="view_graph_calibrator",
                       path_options=(("--database_path", paths.calibrated),), after={})


def global_mapper(cfg: CamerasSettings, paths: Paths, env: Mapping[str, str],
                  database: Path) -> ColmapRun:
    """The Global_Run on `database`, which the solve flow chooses: the Calibration_Copy for
    OPENCV, or the Database for OPENCV_FISHEYE (Req 6.1, 6.2, O9).

    It reads the Colour_Picks, work/frames/images/, for the points' colours, and writes its
    Models into colmap/global/ (Req 1.5).
    """
    return _colmap_run(
        cfg, paths, env, command="global_mapper", name="global_mapper",
        path_options=(("--database_path", database), ("--image_path", paths.frames / IMAGES),
                      ("--output_path", paths.global_models)),
        after={"--GlobalMapper.keep_max_num_tracks": GLOBAL_POSITIONING_GPU,
               "--GlobalMapper.ba_min_track_length": GLOBAL_BUNDLE_ADJUSTMENT})


def mapper(cfg: CamerasSettings, paths: Paths, env: Mapping[str, str]) -> ColmapRun:
    """The Incremental_Run, on the Database and never on the Calibration_Copy (Req 6.6).

    It reads the Colour_Picks, work/frames/images/, for the points' colours, and writes its
    Models into colmap/incremental/ (Req 1.5).
    """
    return _colmap_run(
        cfg, paths, env, command="mapper", name="mapper",
        path_options=(("--database_path", paths.database), ("--image_path", paths.frames / IMAGES),
                      ("--output_path", paths.incremental_models)),
        after={"--Mapper.ba_local_max_refinement_change": MAPPER_BUNDLE_ADJUSTMENT})


def image_undistorter(cfg: CamerasSettings, paths: Paths, env: Mapping[str, str],
                      model: Path) -> ColmapRun:
    """The Undistortion of the Final_Model in the folder `model`: it warps the Colour_Picks,
    work/frames/images/, to one PINHOLE camera, and writes COLMAP's own layout, images/ and
    sparse/, into colmap/undistorted/ (Req 1.5, 8.1)."""
    return _colmap_run(
        cfg, paths, env, command="image_undistorter", name="image_undistorter",
        path_options=(("--image_path", paths.frames / IMAGES), ("--input_path", model),
                      ("--output_path", paths.undistorted)),
        after={"--default_random_seed": OUTPUT_TYPE})


def image_undistorter_standalone(cfg: CamerasSettings, paths: Paths,
                                 env: Mapping[str, str]) -> ColmapRun:
    """The Mask_Warp: it warps work/frames/mask.png with the camera line in
    colmap/mask_camera.txt, the Final_Model's camera before the Undistortion, into
    colmap/mask_undistorted/, with the Undistortion's values of the undistortion settings
    (Req 8.3)."""
    return _colmap_run(
        cfg, paths, env, command="image_undistorter_standalone",
        name="image_undistorter_standalone",
        path_options=(("--image_path", paths.frames), ("--input_file", paths.mask_camera),
                      ("--output_path", paths.mask_warped)),
        after={})


def _colmap_run(cfg: CamerasSettings, paths: Paths, env: Mapping[str, str], *, command: str,
                name: str, path_options: Sequence[tuple[str, Path]],
                after: Mapping[str, Sequence[str]]) -> ColmapRun:
    """The ColmapRun `name` of COLMAP's `command`.

    Its command line is PROGRAM, `command`, each of `path_options` with its path, the log options,
    then each Option_Key that OPTION_KEYS gives `command`, in OPTION_KEYS's order, which is
    references/options.md's, with its option and its value. --default_random_seed comes first of
    those, because OPTION_KEYS lists it first. `after` holds, by an Option_Key's option, the words
    that come right after that option and its value: the command's Fixed_Constants and the
    options that come and go, so each lands where options.md's block has it.
    """
    args = [PROGRAM, command]
    for option, path in path_options:
        args += (option, str(path))
    args += LOG_OPTIONS
    for item in OPTION_KEYS:
        if command in item.commands:
            args += (item.option, format_value(cfg.options[item.key]), *after.get(item.option, ()))
    return ColmapRun(name=name, command=command, args=tuple(args),
                     timeout_seconds=cfg.timeouts[command], timeout_key=TIMEOUT_KEYS[command],
                     log=paths.logs / f"{name}.log", cwd=paths.colmap, env=env)
