"""The Cameras_Step, which solves the cameras for the Picks that the Frames_Step wrote to
work/frames/: where each Pick was taken from, and the one lens that every Pick shares.

The step runs COLMAP 4.2.1 on the CPU, as the separate program colmap: feature_extractor on the
Grey_Copies, sequential_matcher twice, view_graph_calibrator, global_mapper, the incremental
mapper once when the global mapper gives up, and image_undistorter and
image_undistorter_standalone to one PINHOLE camera. Then it levels and centres the scene with one
rigid move, writes the levelled cameras and points to its Work_Folder, work/cameras/, for the
Training_Step, and copies the levelled cameras to out/cameras.json for the Viewer. Its settings
are in [cameras] in config.toml, and settings.py reads and checks them.

The package is built like the Fake_Step and the Frames_Step. `work(step)` is the step's own code,
which the Step_Runner calls once it has emptied work/cameras/, so no file of an earlier run is
left there (Req 11.13). `run(job_id, preset=None, *, config=None)` runs the step through the
Step_Runner and returns its Step_Report. `__main__.py` is the CLI,
`uv run python -m splatpipe.cameras --job <id> [--preset <name>]` (Req 14.1, 14.2).

`work` goes in this order:

1. It removes an earlier out/cameras.json, before anything else, so a run that fails in the
   step's own code leaves none (Req 11.10, A9).
2. The settings, every one checked before the step reads frames.json (Req 12.8). Then the
   Cap_Deadline, read once from SPLATPIPE_CAP_DEADLINE with read_cap_deadline, so a bad value
   fails with that function's own message before frames.json is read or any COLMAP_Command runs
   (Req 17.1, 17.2, C12). The step caps each COLMAP_Command's Time_Limit at the time left before
   the Cap_Deadline as the command starts, with no cap when the variable isn't set (Req 17.3,
   17.4, D17).
3. The Frames_Contract: frames.json and the files it names, checked before any COLMAP_Command
   runs (Req 1.1 to 1.4). The count picks goes in.
4. COLMAP: when colmap isn't on the PATH, the step fails before any COLMAP_Command (Req 2.7).
5. The folders colmap/, colmap/tmp/ and colmap/logs/, and the COLMAP_Commands' environment, with
   TMPDIR and TMP in colmap/tmp/ (Req 2.2, C2). Every path is absolute.
6. The solve, solve.py's state machine, from the Feature_Extraction to the Final_Model, read
   whole (Req 2 to 7).
7. The Undistortion of the Final_Model, checked (Req 8.1, 8.2). With mask_used, the Mask_Warp of
   the Mask with the Final_Model's camera as the Final_Model holds it before the Undistortion,
   and the Undistorted_Mask (Req 8.3 to 8.5).
8. The Levelling. When it moves the scene, the whole Level_Move goes to every Pose and point at
   once. When it keeps the COLMAP_Frame, the Levelled_Model is the Undistorted_Model as it is,
   bit for bit, and the reason is a warning (Req 9, C10). The count levelled goes in. The highest
   view ray can add a warning about missing ceiling coverage.
9. The writes, in this order: the Training_Images, the Training_Masks with mask_used, the
   Levelled_Model, solve.json, then out/cameras.json, last of all. When out/cameras.json can't
   be written, solve.json is removed too, so a run that fails leaves neither (Req 11.1 to 11.9,
   C6).

Parts 7, 8 and 9 are timed as the stages undistortion, levelling and writing, like the solve's
matching and mapping, and each COLMAP_Command's timing goes in as soon as the command returns, so
a failed report holds what the step had found before it failed (Req 13.2, 13.3). Each count goes
in as an int as soon as it's known (Req 13.1, 13.4). The step writes only inside work/cameras/
and to out/cameras.json, where only the Step_Runner writes the report, and it leaves work/frames/
as it is: COLMAP only reads from it (Req 1.7, 11.11). Every failure the requirements name is an
ExpectedFailure (Req 14.10). An OSError while the step makes a folder or writes a file is the
Req 11.12 failure, which names it and the reason.

`work` reads the module constant TOOLS each time it's called, so a test that replaces the tools
with monkeypatch reaches the step through run_step and the CLI. TOOLS.now is the clock the
Cap_Deadline is read against, so a test can move the time (C16).

From splatpipe, this file imports only splatpipe.errors, splatpipe.frames.files, splatpipe.step,
splatpipe.train.deadline and the step's own modules. Every file in this package imports only the standard library, numpy,
cv2 and splatpipe.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import numpy as np

# The modules level and solve, not their functions of the same names, so splatpipe.cameras.level
# and splatpipe.cameras.solve stay the modules.
from splatpipe.cameras import commands, level, solve
from splatpipe.cameras.colmap import (NOT_INSTALLED, PROGRAM, ColmapRun, Deadline, RunResult,
                                      colmap_env, run_colmap)
from splatpipe.cameras.commands import Paths
from splatpipe.cameras.frames_input import FramesInput, read_frames
from splatpipe.cameras.level import Levelling, move_model, scene_of
from splatpipe.cameras.model import Camera, Model, read_model, write_model
from splatpipe.cameras.outputs import (VIEWER, copy_training_images, mean_reprojection_error,
                                       solve_record, viewer_cameras, write_record,
                                       write_training_masks)
from splatpipe.cameras.settings import CamerasSettings, read_settings
from splatpipe.cameras.solve import Solved, run_or_fail
from splatpipe.cameras.undistort import check_undistorted, mask_camera_line, read_warped_mask
from splatpipe.errors import LOOK_UP_WARNING, ExpectedFailure
from splatpipe.frames.files import MASK
# Config comes through splatpipe.step, as for the Fake_Step and the Frames_Step.
from splatpipe.step import Config, StepContext, run_step
from splatpipe.train.deadline import read_cap_deadline, utc_now

# The Undistortion's own layout in colmap/undistorted/, as --output_type COLMAP writes it (A1):
# the warped Colour_Picks in images/, and the Undistorted_Model, all five files, in sparse/.
_UNDISTORTED_IMAGES = "images"
_UNDISTORTED_MODEL = "sparse"


@dataclass(frozen=True)
class Tools:
    """The tools the step runs. TOOLS holds the real ones, and tests put stand-ins in its place."""

    # find_colmap(), like shutil.which(PROGRAM): the colmap program's path, or None when it isn't
    # on the PATH (Req 2.7).
    find_colmap: Callable[[], str | None]
    # run(colmap_run), like colmap.run_colmap: runs one COLMAP_Command and gives its RunResult.
    run: Callable[[ColmapRun], RunResult]
    # now(), like utc_now: the clock the Cap_Deadline is read against (C16).
    now: Callable[[], datetime] = utc_now


# The real tools. work reads TOOLS each time it's called, so a test can monkeypatch it.
TOOLS = Tools(find_colmap=lambda: shutil.which(PROGRAM), run=run_colmap, now=utc_now)


@dataclass(frozen=True)
class _Flow:
    """What the parts after the solve use: the step, the COLMAP runner, the checked settings, the
    Frames_Contract, the paths, the COLMAP_Commands' environment and the Cap_Deadline."""

    step: StepContext
    run: Callable[[ColmapRun], RunResult]
    cfg: CamerasSettings
    frames: FramesInput
    paths: Paths
    env: Mapping[str, str]
    deadline: Deadline


def run(job_id: str, preset: str | None = None, *,
        config: Config | None = None) -> dict[str, Any]:
    """Run the cameras step on the Job `job_id` and return its Step_Report.

    `preset` replaces the Job_Record's Preset, and `config` defaults to load_config().
    """
    return run_step("cameras", "splatpipe.cameras", job_id, preset=preset, config=config)


def work(step: StepContext) -> None:
    """The cameras step's own code, in the module's order. The Step_Runner turns what it raises
    into a failed report."""
    _remove_earlier_viewer(step.out_folder / VIEWER)
    tools = TOOLS  # read now, so a test's monkeypatch is seen
    cfg = read_settings(step.settings, step.name, step.preset)
    deadline = Deadline(read_cap_deadline(os.environ), tools.now)  # once, before frames.json
    frames = read_frames(step.job.folder / "work" / "frames", step.job.id)  # the Frames_Step's
    step.add_count("picks", int(len(frames.picks)))
    if tools.find_colmap() is None:
        raise ExpectedFailure(NOT_INSTALLED)
    paths = Paths.of(step.work_folder, frames.folder, step.out_folder)
    _make_folders(paths.colmap, paths.tmp, paths.logs)
    env = colmap_env(paths.tmp, os.environ)
    solved = solve.solve(step, tools.run, cfg, frames, paths, env, deadline=deadline)
    flow = _Flow(step=step, run=tools.run, cfg=cfg, frames=frames, paths=paths, env=env,
                 deadline=deadline)
    undistorted, mask = _undistort(flow, solved)
    levelling, levelled = _levelling(flow, undistorted)
    _write(flow, solved, levelling, levelled, mask)


def _remove_earlier_viewer(path: Path) -> None:
    """Remove `path`, the out/cameras.json an earlier run wrote, so a run that fails in the step's
    own code leaves none (Req 11.10, A9). When there's none, there's nothing to do. Any other
    OSError is the Req 11.12 failure, naming the file and the reason."""
    try:
        path.unlink(missing_ok=True)
    except OSError as error:
        raise ExpectedFailure(f"The file {path} from an earlier run couldn't be removed "
                              f"({error.strerror or error}). Check that you can change the "
                              f"files in {path.parent}, then run the step again.") from error


def _make_folders(*folders: Path) -> None:
    """Make each of `folders`, in order, in the Work_Folder that the Step_Runner emptied. An
    OSError, like a full disk, is the Req 11.12 failure, naming the folder and the reason."""
    for folder in folders:
        try:
            folder.mkdir()
        except OSError as error:
            raise _not_written(f"The folder {folder} couldn't be made", folder, error) from error


def _undistort(flow: _Flow, solved: Solved) -> tuple[Model, np.ndarray | None]:
    """The Undistortion of the Final_Model, checked, then with mask_used the Mask_Warp, all timed
    as undistortion. Gives the Undistorted_Model, and the Undistorted_Mask, or None without
    mask_used (Req 8.1 to 8.5).

    Both undistorters run through run_or_fail, so a Timeout fails the step (Req 2.8), and so
    does a non-zero exit (Req 2.9).
    """
    step, paths = flow.step, flow.paths
    with step.timed("undistortion"):
        run_or_fail(step, flow.run,
                    commands.image_undistorter(flow.cfg, paths, flow.env, solved.folder),
                    deadline=flow.deadline)
        undistorted = read_model(paths.undistorted / _UNDISTORTED_MODEL)
        camera = check_undistorted(undistorted, solved.placed_names,
                                   paths.undistorted / _UNDISTORTED_IMAGES)
        mask: np.ndarray | None = None
        if flow.frames.mask_used:
            mask = _warp_mask(flow, solved.model.cameras[0], camera)
    return undistorted, mask


def _warp_mask(flow: _Flow, final: Camera, undistorted: Camera) -> np.ndarray:
    """The Mask_Warp, then the Undistorted_Mask from its output (Req 8.3 to 8.5).

    The camera line in colmap/mask_camera.txt is `final`, the Final_Model's camera as the
    Final_Model holds it before the Undistortion, and the Mask_Warp's command line holds the
    Undistortion's undistortion settings. So its output must be the size of `undistorted`, the
    Undistorted_Camera (F7).
    """
    paths = flow.paths
    _write_camera_line(paths.mask_camera, mask_camera_line(final))
    run_or_fail(flow.step, flow.run,
                commands.image_undistorter_standalone(flow.cfg, paths, flow.env),
                deadline=flow.deadline)
    return read_warped_mask(paths.mask_warped / MASK, undistorted.width, undistorted.height)


def _write_camera_line(path: Path, line: str) -> None:
    """Write the Mask_Warp's camera line `line` to `path`, as it is. An OSError, like a full disk,
    is the Req 11.12 failure, naming the file and the reason."""
    try:
        path.write_text(line, encoding="utf-8", newline="\n")
    except OSError as error:
        raise _not_written(f"The mask's camera line {path} couldn't be written", path,
                           error) from error


def _levelling(flow: _Flow, undistorted: Model) -> tuple[Levelling, Model]:
    """The Levelling of the Undistorted_Model, timed as levelling. Gives what level found, and
    the Levelled_Model (Req 9.1, 9.13, 9.14).

    level never raises (Req 9.12). When it moved the scene, move_model applies the whole
    Level_Move to every Pose and point at once. When it kept the COLMAP_Frame, the
    Levelled_Model is the Undistorted_Model as it is, every Pose and point bit for bit, with no
    identity move applied (C10), and the reason goes into the report as a warning (Req 9.3, 9.7,
    9.10, 9.12). Then levelled goes in, 1 or 0 (Req 13.1).
    """
    step = flow.step
    with step.timed("levelling"):
        scene = scene_of(undistorted, flow.frames.numbers)
        levelling = level.level(scene, flow.cfg.level)
        rotations = scene.rotations
        if levelling.levelled:
            levelled = move_model(undistorted, levelling.move)
            rotations = rotations @ levelling.move.rotation.T
        else:
            levelled = undistorted
            step.add_warning(cast(str, levelling.reason))  # a kept frame always has its reason
        step.add_count("levelled", int(levelling.levelled))
        highest_ray = level.highest_ray_degrees(rotations, undistorted.cameras[0],
                                               levelled=levelling.levelled)
        if highest_ray is not None and highest_ray < flow.cfg.min_highest_ray_degrees:
            step.add_warning(LOOK_UP_WARNING)
    return levelling, levelled


def _write(flow: _Flow, solved: Solved, levelling: Levelling, levelled: Model,
           mask: np.ndarray | None) -> None:
    """Write the step's files, timed as writing, in this order: the Training_Images, the
    Training_Masks, the Levelled_Model, solve.json, then out/cameras.json, last of all (Req 11.2
    to 11.9, A8).

    The Training_Images and the Training_Masks take the Levelled_Model's image names, in its
    order. `mask` is the Undistorted_Mask, or None without mask_used, which writes no masks/
    (Req 11.4). When writing out/cameras.json fails, write_record has removed what it partly
    wrote, and solve.json is removed too, so the failed run leaves neither (C6).
    """
    paths = flow.paths
    with flow.step.timed("writing"):
        names = [image.name for image in levelled.images]
        copy_training_images(paths.undistorted / _UNDISTORTED_IMAGES, paths.images, names)
        if mask is not None:  # with mask_used
            write_training_masks(paths.masks, mask, names)
        write_model(paths.sparse, levelled)
        write_record(paths.record, solve_record(solved, levelling, flow.cfg.camera_model,
                                                mean_reprojection_error(solved.model.points)))
        try:
            write_record(paths.viewer, viewer_cameras(levelled, flow.frames.numbers,
                                                      levelling.levelled))
        except BaseException:  # Ctrl+C included, so no failed run leaves solve.json (C6)
            paths.record.unlink(missing_ok=True)
            raise


def _not_written(problem: str, path: Path, error: OSError) -> ExpectedFailure:
    """The Req 11.12 failure: `problem`, which names the file or folder, the reason the system
    gave, and what to do."""
    return ExpectedFailure(f"{problem} ({error.strerror or error}). Free some space on the disk, "
                           f"or check that you can write to {path.parent}, then run the step "
                           "again.")
