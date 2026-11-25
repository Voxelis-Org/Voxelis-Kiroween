"""The solve flow: COLMAP's extraction, matching, calibration and mapping, the one Fallback and the
Placed_Check, up to the Final_Model, read whole.

solve is the small state machine of the design's "The solve flow". Each state runs at most one
COLMAP_Command through the runner it's given, colmap.run_colmap in the step and a stand-in in the
tests, and the RunResult, with the files the command wrote, decides the next state. The commands
run one at a time, each after the one before has returned, and none runs twice or is retried
(Req 2.11, 2.12):

1. The Feature_Extraction, then the Near_Match_Run and the Far_Match_Run on the same Database.
2. For RADIAL, the Calibration_Copy, then the Calibration on it, so the Database stays as the
   Match_Runs left it (Req 5.1, 5.2). A non-zero exit, a crash included, a rejected camera and no
   image pairs are each a Failed_Calibration (Req 5.4, A3). After an exit of 0, the copy's camera
   row next to the Database's tells the last two from a Calibration that worked (O5). For
   RADIAL_FISHEYE there's no Calibration, because view_graph_calibrator skips fisheye cameras
   (F17), only the Req 5.3 warning.
3. Unless the Calibration failed, the Global_Run: on the Calibration_Copy for RADIAL, and on the
   Database for RADIAL_FISHEYE (Req 6.1, 6.2, O9). It gives up on a non-zero exit, a crash
   included, on an exit of 0 with no Model, and on a Kept_Model that fails the Placed_Check
   (Req 6.3 to 6.5).
4. After a Failed_Calibration, with the Req 5.5 warning, or when the Global_Run gives up, with the
   Req 6.7 warning, the Fallback: the one Incremental_Run, on the Database (Req 5.5, 6.6, 6.7).
   Its non-zero exit fails the step with the Filming_Advice (Req 6.10), and when it writes no
   Model, the Final_Model counts 0 Placed_Frames (Req 6.11).
5. The Final_Model is the Kept_Model of the run that made it (Req 6.9). One that fails the
   Placed_Check is the Placed_Failure, before any Undistortion (Req 7.2, 7.3). Otherwise it's read
   whole and given back.

Every command goes through run_timed, which caps its Time_Limit at the seconds left before the
Cap_Deadline in solve's `deadline` just before it starts, and fails the step with no process
started when no time is left (Req 17.3 to 17.5, C13, C14). A Timeout of any command fails the step
with no Fallback, with the Req 2.8 message under its key's limit and the Req 17.6 message under a
limit the Cap_Deadline set (C15), and so does a non-zero exit of the extractor or of a Match_Run
(Req 2.9). Each command's timing is added as soon as the command returns, whatever its result.
The stage timings matching and mapping are added when their blocks finish, so a failed report
holds the failing command's timing but not its stage's (Req 13.2, 13.3). Each count is added as an
int (Req 13.1, 13.4).

kept_model and placed_needed are pure, and model_counts reads each Model's images.bin. The
Kept_Model has the most Placed_Frames, and the lowest Model number among those with as many
(Req 6.8). The Placed_Check needs placed_needed Placed_Frames, worked out with the exact fraction
of min_placed_share, so 0.3 of 10 Picks needs exactly 3 (Req 7.1, C7). An image counts as a
Placed_Frame when its name is a Pick's file (C11).

From splatpipe, this module imports only splatpipe.errors and the step's colmap, commands,
database, frames_input, model and settings.
"""

from __future__ import annotations

import math
import re
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import timezone
from fractions import Fraction
from pathlib import Path
from typing import Literal, Protocol

from splatpipe.cameras import commands
from splatpipe.cameras.colmap import NO_DEADLINE, ColmapRun, Deadline, RunResult, capped, exit_text
from splatpipe.cameras.commands import RADIAL_FISHEYE, Paths
from splatpipe.cameras.database import calibration_failure, copy_database, read_camera
from splatpipe.cameras.frames_input import FramesInput
from splatpipe.cameras.model import Model, read_images, read_model
from splatpipe.cameras.settings import CamerasSettings
from splatpipe.errors import ExpectedFailure

# A Model's folder name in a mapper's output folder: a decimal number as COLMAP writes it, 0, 1,
# 2 and so on, with no sign and no leading zero, so each number names one folder.
_MODEL_NUMBER = re.compile(r"0|[1-9][0-9]*")
_IMAGES = "images.bin"  # the Model_File that holds exactly the images a Model registers (F3)

# The Filming_Advice, which ends the Req 6.10 failure and the Placed_Failure (Req 7.3).
_FILMING_ADVICE = ("Film the place again: move slower, hold the phone steadier, overlap more "
                   "between views, and film surfaces with more texture. Then make a new job "
                   "from the new video.")
# The warnings of the mapping stage (Req 5.3, 5.5, 6.7).
_FISHEYE = ("The camera model is RADIAL_FISHEYE, so the focal length calibration was skipped, "
            "because COLMAP's view_graph_calibrator only fits pinhole cameras. The fisheye "
            "camera starts from fisheye_fov_degrees = {degrees} in [cameras].")
_CALIBRATION_FAILED = ("The focal length calibration {reason}, so the global mapper didn't run "
                       "and the incremental mapper ran instead.")
_GLOBAL_GAVE_UP = "The global mapper {reason}, so the incremental mapper ran instead."


class Report(Protocol):
    """Where the solve flow adds its timings, counts and warnings. StepContext has these three."""

    def add_timing(self, name: str, seconds: float) -> None: ...

    def add_count(self, name: str, value: int) -> None: ...

    def add_warning(self, text: str) -> None: ...


@dataclass(frozen=True)
class ModelCount:
    """One Model of a mapper run, and how many Picks it places."""

    number: int  # the Model's folder name, 0, 1, ...
    placed: int  # its Placed_Frames


@dataclass(frozen=True)
class Solved:
    """What the solve flow found: how the Calibration and the mappers went, and the Final_Model."""

    calibration_ran: bool  # false for RADIAL_FISHEYE, which has no Calibration
    calibration_failure: str | None  # why, for a Failed_Calibration
    mapper: Literal["global", "incremental"]  # the mapper whose run made the Final_Model
    fallback_reason: str | None  # why the Fallback ran, as its warning says, or None
    models: tuple[ModelCount, ...]  # every Model of the run that made the Final_Model
    final: ModelCount  # the Final_Model
    folder: Path  # the Final_Model's folder
    model: Model  # the Final_Model, read whole
    placed_names: frozenset[str]  # its Placed_Frames' names


@dataclass(frozen=True)
class _Flow:
    """What the states of the mapping stage use: solve's arguments."""

    report: Report
    run: Callable[[ColmapRun], RunResult]
    cfg: CamerasSettings
    frames: FramesInput
    paths: Paths
    env: Mapping[str, str]
    deadline: Deadline

    def needed(self) -> int:
        """The fewest Placed_Frames that pass the Placed_Check for these Picks (Req 7.1)."""
        return placed_needed(len(self.frames.picks), self.cfg.min_placed_frames,
                             self.cfg.min_placed_share)


@contextmanager
def timed(report: Report, name: str) -> Iterator[None]:
    """Time the block with time.perf_counter, then add its seconds to `report` as the timing
    `name`, as StepContext.timed does. A block that raises adds no timing (Req 13.3)."""
    started = time.perf_counter()
    yield
    report.add_timing(name, time.perf_counter() - started)


def run_timed(report: Report, run: Callable[[ColmapRun], RunResult], colmap_run: ColmapRun,
              *, deadline: Deadline = NO_DEADLINE) -> RunResult:
    """Cap `colmap_run` at `deadline`, run it with `run`, add its timing under its name at once,
    whatever the result, and give the result back (Req 13.3, 17.3, 17.4, C13).

    With no time left before the Cap_Deadline, capped raises the Req 17.5 failure before `run`
    is called, so no COLMAP process starts and the command gets no timing (C14). A Timeout is the
    Req 2.8 failure under a limit its key set, and the Req 17.6 failure under a limit the
    Cap_Deadline set (C15). Neither failure ever has a Fallback.
    """
    colmap_run = capped(colmap_run, deadline.at, deadline.now())
    result = run(colmap_run)
    report.add_timing(colmap_run.name, result.seconds)
    if result.exit_code is None:
        raise _timeout_failure(colmap_run)
    return result


def run_or_fail(report: Report, run: Callable[[ColmapRun], RunResult], colmap_run: ColmapRun,
                *, deadline: Deadline = NO_DEADLINE) -> RunResult:
    """run_timed, and the Req 2.9 failure for a non-zero exit, which gives the exit code, the
    Last_Error_Line and the Command_Log.

    The Feature_Extraction, the Match_Runs and both undistorters run through it. The Calibration
    and both mappers run through run_timed, because their non-zero exits have rules of their own.
    """
    result = run_timed(report, run, colmap_run, deadline=deadline)
    if result.exit_code != 0:
        raise ExpectedFailure(
            f"COLMAP's {colmap_run.command} {_stopped_with(result)}. Check that COLMAP 4.2.1 is "
            "installed as .kiro/steering/tech.md says, then run the step again. Its output is in "
            f"{colmap_run.log}.")
    return result


def kept_model(models: Sequence[ModelCount]) -> ModelCount | None:
    """The Kept_Model of one mapper run's `models`: the one with the most Placed_Frames, and the
    one with the lowest number among those with as many, or None for no Model (Req 6.8)."""
    return max(models, key=lambda model: (model.placed, -model.number), default=None)


def placed_needed(picks: int, min_frames: int, min_share: float) -> int:
    """The fewest Placed_Frames that pass the Placed_Check for `picks` Picks: at least
    `min_frames`, and at least `min_share` times `picks` (Req 7.1).

    It's max(min_frames, ⌈Fraction(min_share) × picks⌉), worked out with the exact value of the
    float, so 0.3 of 10 Picks needs exactly 3 (C7). Because a count of Placed_Frames is whole,
    reaching it is the same as reaching both limits.
    """
    return max(min_frames, math.ceil(Fraction(min_share) * picks))


def model_counts(folder: Path, numbers: Mapping[str, int]) -> tuple[ModelCount, ...]:
    """Each Model that a mapper wrote into its output folder `folder`, in number order, with its
    Placed_Frames.

    A Model is a subfolder whose name is a decimal number, 0, 1 and so on, and every other entry
    is ignored. Its Placed_Frames are the images of its images.bin, which holds exactly the images
    the Model registers (F3), whose names are keys of `numbers`, the Picks' files (C11). A missing
    or broken images.bin is the Req 10.4 failure.
    """
    found = sorted(int(entry.name) for entry in folder.iterdir()
                   if _MODEL_NUMBER.fullmatch(entry.name) and entry.is_dir())
    return tuple(ModelCount(number=number, placed=_placed(folder / str(number), numbers))
                 for number in found)


def solve(report: Report, run: Callable[[ColmapRun], RunResult], cfg: CamerasSettings,
          frames: FramesInput, paths: Paths, env: Mapping[str, str],
          *, deadline: Deadline = NO_DEADLINE) -> Solved:
    """Solve the cameras of the Picks in `frames`, from the Feature_Extraction to the Final_Model,
    as the module says, and give back what the flow found, the Final_Model read whole included.

    `run` runs each COLMAP_Command, `cfg` holds the checked settings, `paths` the step's paths,
    whose colmap/, colmap/tmp/ and colmap/logs/ already exist, `env` the COLMAP_Commands'
    environment, and `deadline` the Cap_Deadline that caps every run's Time_Limit (Req 17.3).
    The command and stage timings, the counts fallback_ran, kept_from_global, models_found,
    placed_frames and points3d, and the Req 5.3, 5.5 and 6.7 warnings go into `report`. Every
    failure the requirements name raises an ExpectedFailure.
    """
    run_or_fail(report, run, commands.feature_extractor(cfg, frames, paths, env),
                deadline=deadline)
    with timed(report, "matching"):
        run_or_fail(report, run, commands.sequential_matcher(cfg, paths, env, far=False),
                    deadline=deadline)
        run_or_fail(report, run, commands.sequential_matcher(cfg, paths, env, far=True),
                    deadline=deadline)
    with timed(report, "mapping"):
        return _mapping(_Flow(report, run, cfg, frames, paths, env, deadline))


def _mapping(flow: _Flow) -> Solved:
    """The mapping stage, steps 2 to 5 of the module's list: the Calibration for RADIAL, or the
    Req 5.3 warning for RADIAL_FISHEYE, then the mappers, then the Final_Model."""
    calibration_ran = flow.cfg.camera_model != RADIAL_FISHEYE
    if calibration_ran:
        failure, database = _calibrate(flow), flow.paths.calibrated
    else:
        flow.report.add_warning(_FISHEYE.format(degrees=flow.cfg.fisheye_fov_degrees))
        failure, database = None, flow.paths.database
    mapper, models, fallback_reason = _mappers(flow, failure, database)
    final, folder, model = _final_model(flow, mapper, models)
    numbers = flow.frames.numbers
    return Solved(calibration_ran=calibration_ran, calibration_failure=failure, mapper=mapper,
                  fallback_reason=fallback_reason, models=models, final=final, folder=folder,
                  model=model, placed_names=frozenset(image.name for image in model.images
                                                      if image.name in numbers))


def _calibrate(flow: _Flow) -> str | None:
    """The Calibration_Copy, then the Calibration on it: why it's a Failed_Calibration, or None
    when it worked (Req 5.1, 5.2, 5.4).

    The copy is made once the Far_Match_Run has returned, so no COLMAP process has the Database
    open (Req 2.11). A non-zero exit gives the reason "stopped with <exit code> (<line>)" (A3).
    After an exit of 0, the copy's camera next to the Database's tells a rejected camera and no
    image pairs from a Calibration that worked (O5). A Timeout fails the step (Req 2.8).
    """
    copy_database(flow.paths.database, flow.paths.calibrated)
    result = run_timed(flow.report, flow.run,
                       commands.view_graph_calibrator(flow.cfg, flow.paths, flow.env),
                       deadline=flow.deadline)
    if result.exit_code != 0:
        return _stopped_with(result)
    return calibration_failure(read_camera(flow.paths.database),
                               read_camera(flow.paths.calibrated))


def _mappers(flow: _Flow, failure: str | None, database: Path
             ) -> tuple[Literal["global", "incremental"], tuple[ModelCount, ...], str | None]:
    """The Global_Run on `database`, unless `failure` says why the Calibration failed, then the
    Fallback after a Failed_Calibration or when the Global_Run gives up, each with its warning
    (Req 5.5, 6.7).

    Gives the mapper whose run makes the Final_Model, that run's Models, and why the Fallback
    ran, or None when it didn't (Req 6.9).
    """
    if failure is not None:
        reason = failure
        flow.report.add_warning(_CALIBRATION_FAILED.format(reason=reason))
    else:
        models, reason = _global_run(flow, database)
        if reason is None:
            return "global", models, None
        flow.report.add_warning(_GLOBAL_GAVE_UP.format(reason=reason))
    return "incremental", _incremental_run(flow), reason


def _global_run(flow: _Flow, database: Path) -> tuple[tuple[ModelCount, ...], str | None]:
    """The Global_Run on `database`, into colmap/global/, which is made first: its Models, and
    why it gave up, or None when its Kept_Model passes the Placed_Check (Req 6.1 to 6.5).

    It gives up on a non-zero exit, a crash included, on an exit of 0 with no Model, and on a
    Kept_Model under placed_needed. A Timeout fails the step, with no Fallback (Req 2.8).
    """
    flow.paths.global_models.mkdir()
    result = run_timed(flow.report, flow.run,
                       commands.global_mapper(flow.cfg, flow.paths, flow.env, database),
                       deadline=flow.deadline)
    if result.exit_code != 0:
        return (), _stopped_with(result)
    models = model_counts(flow.paths.global_models, flow.frames.numbers)
    kept, needed = kept_model(models), flow.needed()
    if kept is None:
        return models, "wrote no model"
    if kept.placed < needed:
        return models, (f"placed only {kept.placed} of the {len(flow.frames.picks)} picks in its "
                        f"best model, fewer than the {needed} the step needs")
    return models, None


def _incremental_run(flow: _Flow) -> tuple[ModelCount, ...]:
    """The Fallback, the one Incremental_Run, on the Database and into colmap/incremental/,
    which is made first: its Models, none when it wrote no Model (Req 6.6, 6.11).

    A non-zero exit is the Req 6.10 failure, which gives the exit code, the Last_Error_Line and
    the Filming_Advice. A Timeout fails the step too (Req 2.8).
    """
    flow.paths.incremental_models.mkdir()
    result = run_timed(flow.report, flow.run, commands.mapper(flow.cfg, flow.paths, flow.env),
                       deadline=flow.deadline)
    if result.exit_code != 0:
        raise ExpectedFailure(
            f"COLMAP's incremental mapper stopped with an error ({exit_text(result.exit_code)}: "
            f"{result.last_line}), so the cameras couldn't be solved. {_FILMING_ADVICE}")
    return model_counts(flow.paths.incremental_models, flow.frames.numbers)


def _final_model(flow: _Flow, mapper: Literal["global", "incremental"],
                 models: tuple[ModelCount, ...]) -> tuple[ModelCount, Path, Model]:
    """The Final_Model, the Kept_Model of `models` from `mapper`'s run, with its folder, read
    whole (Req 6.9).

    The counts fallback_ran, kept_from_global, models_found and placed_frames go into the report
    first, with 0 Placed_Frames for no Model (Req 6.11, 13.1). No Model, or a Kept_Model that
    fails the Placed_Check, is then the Placed_Failure, which gives the placed Picks, all the
    Picks, the number needed and the Filming_Advice (Req 7.2, 7.3). Then points3d.
    """
    from_global = mapper == "global"
    kept = kept_model(models)
    placed = 0 if kept is None else kept.placed
    flow.report.add_count("fallback_ran", int(not from_global))
    flow.report.add_count("kept_from_global", int(from_global))
    flow.report.add_count("models_found", len(models))
    flow.report.add_count("placed_frames", placed)
    needed = flow.needed()
    if kept is None or placed < needed:
        raise ExpectedFailure(f"Only {placed} of the {len(flow.frames.picks)} picks could be "
                              f"placed, but the step needs at least {needed}. {_FILMING_ADVICE}")
    output = flow.paths.global_models if from_global else flow.paths.incremental_models
    folder = output / str(kept.number)
    model = read_model(folder)
    flow.report.add_count("points3d", len(model.points.ids))
    return kept, folder, model


def _placed(model: Path, numbers: Mapping[str, int]) -> int:
    """How many images of the images.bin in the Model folder `model` are named after a Pick's
    file, a key of `numbers` (C11)."""
    return sum(1 for image in read_images(model / _IMAGES) if image.name in numbers)


def _timeout_failure(colmap_run: ColmapRun) -> ExpectedFailure:
    """The failure for `colmap_run` stopped at its Time_Limit: the Req 2.8 one, which names the
    Time_Limit's key in [cameras], when its key set the limit, and otherwise the Req 17.6 one,
    which gives the Cap_Deadline in ISO 8601 UTC, the seconds that were left, rounded to a whole
    second, and the cap_minutes advice, with no [cameras] key (C15)."""
    if colmap_run.capped_at is None:
        return ExpectedFailure(
            f"COLMAP's {colmap_run.command} ran longer than its limit of "
            f"{colmap_run.timeout_seconds} seconds, so the step stopped it. Raise "
            f"{colmap_run.timeout_key} in [cameras] in config.toml, then run the step again.")
    when = colmap_run.capped_at.astimezone(timezone.utc).isoformat()  # ISO 8601, in UTC
    return ExpectedFailure(
        f"COLMAP's {colmap_run.command} was still running at the job's cap deadline, {when}, "
        f"after the {round(colmap_run.timeout_seconds)} seconds that were left, so the step "
        "stopped it. Give the job more time with a larger cap_minutes for its preset under "
        "[cloud.presets] in config.toml, then run the job again.")


def _stopped_with(result: RunResult) -> str:
    """How a command that exited with a non-zero code stopped, like "stopped with exit code 1
    (Failed to create sparse model)": its exit code, then its Last_Error_Line."""
    return f"stopped with {exit_text(result.exit_code)} ({result.last_line})"
