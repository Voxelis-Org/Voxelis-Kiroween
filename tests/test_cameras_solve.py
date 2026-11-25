"""Tests for splatpipe/cameras/solve.py, the solve flow up to the Final_Model, and
splatpipe/cameras/database.py, the COLMAP database's one camera and the Calibration_Copy.

Each test of the flow makes its Job with cameras_job from tests/conftest.py, 8 Picks of 16 × 12
unless it asks for other, and gives solve what the Cameras_Step gives it: a real StepContext for
the Job and its Preset with the test config's [cameras], read_settings on the step's settings,
read_frames on the Job's work/frames/, the Paths of its Work_Folder with colmap/, colmap/tmp/ and
colmap/logs/ made, and colmap_env's environment. A test that needs other settings changes them
with set_cameras first. A FakeColmap from fake_colmap stands in for COLMAP with the test's
script, and its runs list every ColmapRun that solve ran, in order. By default the Calibration
works and each mapper writes one Model that places all 8 Picks. The test config's
min_placed_frames is 1 and its min_placed_share the real 0.5, so 8 Picks need 4 Placed_Frames.

- For RADIAL, the runs are the Feature_Extraction, the Near_Match_Run, the Far_Match_Run, the
  Calibration and the Global_Run, one at a time and in that order, and the Calibration_Copy
  appears only after the Far_Match_Run has returned (Req 2.11, 2.12, 5.1). The Calibration reads
  the copy, the Database's camera row stays as the Match_Runs left it, and a Calibration that
  worked sends the Global_Run to the copy (Req 5.2, 6.1). For RADIAL_FISHEYE, no Calibration
  runs, the Req 5.3 warning names RADIAL_FISHEYE and fisheye_fov_degrees, and the Global_Run
  reads the Database (Req 5.3, 6.2, O9, F17).
- Each kind of Failed_Calibration and each way the Global_Run gives up runs the mapper once, on
  the Database, with the Req 5.5 or 6.7 warning giving the reason (Req 5.4, 5.5, 6.3 to 6.7). The
  Final_Model comes from the run that made it, with fallback_ran, kept_from_global and
  models_found to match (Req 6.9, 13.1).
- A Timeout of each command fails, naming its Time_Limit key, with no Fallback (Req 2.8). A
  non-zero exit of the Feature_Extraction or of a Match_Run fails with the exit code and the
  Last_Error_Line (Req 2.9). The Incremental_Run's non-zero exit is the Req 6.10 failure, and its
  exit 0 with no Model is the Placed_Failure with 0 placed (Req 6.11, 7.2, 7.3).
- Each message is compared with the design's wording, with its values filled in, and holds no
  "Traceback" (Req 16.9). After each failure, the report holds the counts, timings and warnings
  worked out before it, the failing command's timing included (Req 13.1 to 13.3). Every list of
  runs is checked to hold no command twice (Req 2.12).
- placed_needed and the Placed_Check at the edges: exactly min_placed_frames and one fewer,
  exactly min_placed_share of the Picks and one fewer, and 0.3 of 10 Picks needing 3 (Req 7.1,
  C7). model_counts ignores folders whose names aren't numbers and images that aren't Picks
  (C11).
- Property 2 draws with Hypothesis one mapper run's Models as 1 to 20 ModelCounts, with unique
  numbers from 0 to 60 in the order drawn, which often isn't number order, and Placed_Frames
  counts from 0 to 4, so ties are common. A plain loop checks that kept_model gives one of them,
  with at least as many Placed_Frames as every other Model and a lower number than every other
  Model with as many (Req 6.8, 6.12).
- copy_database, read_camera and calibration_failure on small SQLite files the tests write, with
  the cameras table of COLMAP 4.2.1 and RADIAL cameras, whose one focal parameter f, params[0],
  is what calibration_failure compares bit for bit (F12, F17, O5, C1).

Signal 6 is SIGABRT on Linux and macOS, where COLMAP runs, but not on Windows, so the crash cases,
exit code -6, are skipped there.
"""

import math
import os
import sqlite3
import struct
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from conftest import CAMERAS_TABLE, FakeColmap  # COLMAP 4.2.1's cameras table (F12), the stand-in
from splatpipe.cameras.colmap import ColmapRun, RunResult, colmap_env
from splatpipe.cameras.commands import Paths
from splatpipe.cameras.database import (DatabaseCamera, calibration_failure, copy_database,
                                        read_camera)
from splatpipe.cameras.frames_input import FramesInput, read_frames
from splatpipe.cameras.model import POINT2D, Camera, Image, Model, Points, read_model, write_model
from splatpipe.cameras.settings import CamerasSettings, read_settings
from splatpipe.cameras.solve import (ModelCount, Solved, kept_model, model_counts, placed_needed,
                                     solve)
from splatpipe.config import load_config
from splatpipe.errors import ExpectedFailure
from splatpipe.jobs import open_job
from splatpipe.step import StepContext

# The runs of the solve flow, by their names, which are also their timings' names.
EXTRACT, NEAR, FAR = "feature_extractor", "sequential_matcher_near", "sequential_matcher_far"
CALIBRATE, GLOBAL, INCREMENTAL = "view_graph_calibrator", "global_mapper", "mapper"
MATCHED = (EXTRACT, NEAR, FAR)  # the runs before the mapping stage, in order
RADIAL_RUNS = (*MATCHED, CALIBRATE, GLOBAL)  # every run for RADIAL when the Global_Run passes
ALL_RUNS = (*RADIAL_RUNS, INCREMENTAL)  # every run for RADIAL when the Global_Run gives up

# A cameras_job Job's Picks, and the Placed_Frames they need with the test config:
# max(1, 0.5 × 8).
PICKS, NEEDED = 8, 4

# The Filming_Advice that ends the Req 6.10 failure and the Placed_Failure, as the design words it.
ADVICE = ("Film the place again: move slower, hold the phone steadier, overlap more between "
          "views, and film surfaces with more texture. Then make a new job from the new video.")

# The Models of the Fallback in the tests that run it: Model 0 places Picks 0 and 1, and Model 1,
# the Kept_Model, Picks 2 to 7. So a Final_Model taken from Model 0, or from the Global_Run's one
# Model of 8 Picks, would show.
FALLBACK_MODELS = (2, 6)

# Signal 6 is SIGABRT on Linux and macOS, where COLMAP runs, but not on Windows.
POSIX_ONLY = pytest.mark.skipif(sys.platform == "win32",
                                reason="signal 6 isn't SIGABRT on Windows")


# The design's wording of the warnings and the Readable_Errors, with the values filled in

def fisheye_warning(degrees: str) -> str:
    """The Req 5.3 warning for a RADIAL_FISHEYE camera that starts from `degrees`."""
    return ("The camera model is RADIAL_FISHEYE, so the focal length calibration was skipped, "
            "because COLMAP's view_graph_calibrator only fits pinhole cameras. The fisheye "
            f"camera starts from fisheye_fov_degrees = {degrees} in [cameras].")


def calibration_warning(reason: str) -> str:
    """The Req 5.5 warning for a Failed_Calibration, with `reason` filled in."""
    return (f"The focal length calibration {reason}, so the global mapper didn't run and the "
            "incremental mapper ran instead.")


def global_warning(reason: str) -> str:
    """The Req 6.7 warning for a Global_Run that gave up, with `reason` filled in."""
    return f"The global mapper {reason}, so the incremental mapper ran instead."


def short_reason(placed: int, picks: int, needed: int) -> str:
    """The Req 6.7 reason for a Global_Run whose Kept_Model places only `placed` of the `picks`
    Picks, under the `needed` that the Placed_Check needs."""
    return (f"placed only {placed} of the {picks} picks in its best model, fewer than the "
            f"{needed} the step needs")


def placed_failure(placed: int, picks: int, needed: int) -> str:
    """The Placed_Failure's Readable_Error (Req 7.3)."""
    return (f"Only {placed} of the {picks} picks could be placed, but the step needs at least "
            f"{needed}. {ADVICE}")


# What solve gets for one Job, as the Cameras_Step gives it

@dataclass(frozen=True)
class SolveJob:
    """What solve takes for one Job, all but the runner: a real StepContext, the checked
    settings, the Frames_Contract, the Paths and the COLMAP_Commands' environment."""

    step: StepContext
    cfg: CamerasSettings
    frames: FramesInput
    paths: Paths
    env: Mapping[str, str]

    def solve(self, run: Callable[[ColmapRun], RunResult]) -> Solved:
        """Run solve on this Job with `run` in place of colmap.run_colmap."""
        return solve(self.step, run, self.cfg, self.frames, self.paths, self.env)

    def failure(self, run: Callable[[ColmapRun], RunResult]) -> str:
        """The Readable_Error of the ExpectedFailure that solve raises on this Job with `run`,
        checked to hold no traceback (Req 16.9)."""
        with pytest.raises(ExpectedFailure) as caught:
            self.solve(run)
        message = str(caught.value)
        assert "Traceback" not in message
        return message


@pytest.fixture
def solve_job(cameras_job: Callable[..., str]) -> Callable[..., SolveJob]:
    """Give solve_job(**options), which makes a Job with cameras_job(**options) and gives solve's
    arguments for it, built as the Cameras_Step builds them, with the test config as it is when
    solve_job is called."""
    def make(**options: Any) -> SolveJob:
        job_id = cameras_job(**options)
        config = load_config()
        job = open_job(job_id, config)
        preset = job.record.preset
        work_folder = job.folder / "work" / "cameras"  # as the Step_Runner makes it
        work_folder.mkdir()
        step = StepContext(name="cameras", preset=preset, job=job,
                           settings=config.step_settings("cameras", preset),
                           work_folder=work_folder, out_folder=job.folder / "out",
                           timings={}, counts={}, warnings=[])
        frames = read_frames(job.folder / "work" / "frames", job.id)
        paths = Paths.of(step.work_folder, frames.folder, step.out_folder)
        for folder in (paths.colmap, paths.tmp, paths.logs):
            folder.mkdir()
        return SolveJob(step=step, cfg=read_settings(step.settings, step.name, step.preset),
                        frames=frames, paths=paths, env=colmap_env(paths.tmp, os.environ))

    return make


def run_names(fake: FakeColmap) -> list[str]:
    """The names of the runs that `fake` stood in for, in order, checked to hold no run twice
    (Req 2.12)."""
    names = [run.name for run in fake.runs]
    assert len(names) == len(set(names)), names
    return names


def ran(fake: FakeColmap, name: str) -> ColmapRun:
    """The one run named `name` that `fake` stood in for."""
    runs = [run for run in fake.runs if run.name == name]
    assert len(runs) == 1, f"{name} ran {len(runs)} times"
    return runs[0]


def pick_names(numbers: Iterable[int]) -> tuple[str, ...]:
    """The files of the Picks with these Pick_Numbers, as the Frames_Contract names them."""
    return tuple(f"{number:06d}.jpg" for number in numbers)


def assert_final_model(job: SolveJob, solved: Solved, *, from_global: bool,
                       models: tuple[ModelCount, ...], final: ModelCount,
                       names: Sequence[str]) -> None:
    """Check that the Final_Model is `final`, the Kept_Model of `models`, the Models of the
    Global_Run when `from_global` is true and of the Incremental_Run otherwise, read whole from
    its folder with the Placed_Frames `names`, and that the report's counts say so (Req 6.8, 6.9,
    13.1)."""
    output = job.paths.global_models if from_global else job.paths.incremental_models
    folder = output / str(final.number)
    assert solved.mapper == ("global" if from_global else "incremental")
    assert (solved.models, solved.final, solved.folder) == (models, final, folder)
    assert [image.name for image in solved.model.images] == list(names)
    assert solved.placed_names == frozenset(names)
    assert job.step.counts == {"fallback_ran": int(not from_global),
                               "kept_from_global": int(from_global),
                               "models_found": len(models), "placed_frames": final.placed,
                               "points3d": len(read_model(folder).points.ids)}


def assert_fallback_model(job: SolveJob, solved: Solved) -> None:
    """Check that the Final_Model is Model 1 of the Incremental_Run's FALLBACK_MODELS."""
    assert_final_model(job, solved, from_global=False,
                       models=(ModelCount(0, 2), ModelCount(1, 6)), final=ModelCount(1, 6),
                       names=pick_names(range(2, 8)))


# The COLMAP databases the tests write

# The Database's camera as COLMAP's feature_extractor makes it for a 16 × 12 Pick: RADIAL, model
# id 3, with f = 1.2 × 16, the principal point at the centre, k1 and k2 at 0, and no focal prior
# (Req 3.3, F2). As (model, width, height, params, prior_focal_length).
START_PARAMS = (19.2, 8.0, 6.0, 0.0, 0.0)  # f, cx, cy, k1, k2
START = (3, 16, 12, START_PARAMS, 0)


def blob(params: Sequence[float]) -> bytes:
    """A camera's parameters as COLMAP's database holds them: little-endian doubles (F12)."""
    return struct.pack(f"<{len(params)}d", *params)


def write_database(path: Path, *cameras: tuple[int, int, int, Sequence[float], int]) -> None:
    """Write a COLMAP database at `path` with COLMAP 4.2.1's cameras table, holding a row for each
    of `cameras`, given as (model, width, height, params, prior_focal_length)."""
    with closing(sqlite3.connect(path)) as database, database:
        database.execute(CAMERAS_TABLE)
        database.executemany(
            "INSERT INTO cameras (model, width, height, params, prior_focal_length) "
            "VALUES (?, ?, ?, ?, ?)",
            [(model, width, height, blob(params), prior)
             for model, width, height, params, prior in cameras])


def read_only(path: Path) -> sqlite3.Connection:
    """A read-only connection to the existing database `path`, which can't change the file."""
    return sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)


Rows = list[tuple[Any, ...]]  # rows of a table, each column as SQLite holds it


def camera_rows(path: Path) -> Rows:
    """Every row of the cameras table of the database `path`, each column as SQLite holds it."""
    with closing(read_only(path)) as database:
        return database.execute("SELECT * FROM cameras ORDER BY camera_id").fetchall()


def dump(path: Path) -> list[str]:
    """The SQL text of the whole database `path`: every table and every row."""
    with closing(read_only(path)) as database:
        return list(database.iterdump())


# The order of the runs (Req 2.11, 2.12, 5.1, 5.2, 5.3, 6.1, 6.2)

def test_for_radial_the_runs_go_one_at_a_time_and_the_copy_appears_after_the_far_match(
        solve_job: Callable[..., SolveJob], fake_colmap: type[FakeColmap]) -> None:
    job = solve_job()
    fake = fake_colmap()
    events: list[tuple[str, str, bool]] = []  # (moment, run, whether the copy exists then)

    def watched(colmap_run: ColmapRun) -> RunResult:
        events.append(("starts", colmap_run.name, job.paths.calibrated.exists()))
        result = fake.run(colmap_run)
        events.append(("returns", colmap_run.name, job.paths.calibrated.exists()))
        return result

    solved = job.solve(watched)
    assert run_names(fake) == list(RADIAL_RUNS)
    # Each run starts after the one before has returned, and the Calibration_Copy is made between
    # the Far_Match_Run's return and the Calibration's start (Req 2.11, 5.1).
    assert events == [("starts", EXTRACT, False), ("returns", EXTRACT, False),
                      ("starts", NEAR, False), ("returns", NEAR, False),
                      ("starts", FAR, False), ("returns", FAR, False),
                      ("starts", CALIBRATE, True), ("returns", CALIBRATE, True),
                      ("starts", GLOBAL, True), ("returns", GLOBAL, True)]
    assert (solved.calibration_ran, solved.calibration_failure) == (True, None)
    assert (solved.mapper, solved.fallback_reason) == ("global", None)
    assert set(job.step.timings) == {*RADIAL_RUNS, "matching", "mapping"}
    assert job.step.warnings == []


def test_the_calibration_reads_the_copy_and_leaves_the_databases_camera_row_unchanged(
        solve_job: Callable[..., SolveJob], fake_colmap: type[FakeColmap]) -> None:
    job = solve_job()
    fake = fake_colmap()
    started: list[tuple[Rows, Rows]] = []  # the Database's and the copy's camera rows

    def watched(colmap_run: ColmapRun) -> RunResult:
        if colmap_run.name == CALIBRATE:  # the Database as the Match_Runs left it, and its copy
            started.append((camera_rows(job.paths.database), camera_rows(job.paths.calibrated)))
        return fake.run(colmap_run)

    job.solve(watched)
    assert run_names(fake) == list(RADIAL_RUNS)
    assert ran(fake, CALIBRATE).value("--database_path") == str(job.paths.calibrated)
    ((database, copy),) = started
    assert copy == database
    assert camera_rows(job.paths.database) == database
    assert camera_rows(job.paths.calibrated) != database  # the Calibration changed the copy
    # A Calibration that worked sends the Global_Run to the copy (Req 6.1).
    assert ran(fake, GLOBAL).value("--database_path") == str(job.paths.calibrated)


def test_for_radial_fisheye_no_calibration_runs_and_the_global_mapper_reads_the_database(
        solve_job: Callable[..., SolveJob], fake_colmap: type[FakeColmap],
        set_cameras: Callable[..., None]) -> None:
    set_cameras(camera_model="RADIAL_FISHEYE", fisheye_fov_degrees=150.0)
    job = solve_job()
    fake = fake_colmap()
    solved = job.solve(fake.run)
    assert run_names(fake) == [*MATCHED, GLOBAL]
    assert not job.paths.calibrated.exists()
    assert ran(fake, GLOBAL).value("--database_path") == str(job.paths.database)
    assert job.step.warnings == [fisheye_warning("150.0")]
    assert (solved.calibration_ran, solved.calibration_failure) == (False, None)
    assert (solved.mapper, solved.fallback_reason) == ("global", None)
    assert set(job.step.timings) == {*MATCHED, GLOBAL, "matching", "mapping"}


def test_a_global_run_that_passes_gives_the_final_model_from_its_kept_model(
        solve_job: Callable[..., SolveJob], fake_colmap: type[FakeColmap]) -> None:
    job = solve_job()
    fake = fake_colmap(global_models=(2, 6))
    solved = job.solve(fake.run)
    assert run_names(fake) == list(RADIAL_RUNS)
    assert_final_model(job, solved, from_global=True, models=(ModelCount(0, 2), ModelCount(1, 6)),
                       final=ModelCount(1, 6), names=pick_names(range(2, 8)))


# The Fallback (Req 5.4, 5.5, 6.3 to 6.7, 6.9)

# Each kind of Failed_Calibration: FakeColmap's script, and the reason the Req 5.5 warning gives.
FAILED_CALIBRATIONS = [
    pytest.param({"exits": {CALIBRATE: (1, "View graph calibration failed")}},
                 "stopped with exit code 1 (View graph calibration failed)", id="exit 1"),
    pytest.param({"exits": {CALIBRATE: (-6, "Check failed: num_cameras > 0")}},
                 "stopped with exit code -6 (signal SIGABRT) (Check failed: num_cameras > 0)",
                 id="exit -6", marks=POSIX_ONLY),
    pytest.param({"calibration": "rejected"},
                 "rejected its focal length estimate and kept the starting one", id="rejected"),
    pytest.param({"calibration": "no_pairs"}, "found no image pairs to calibrate", id="no pairs"),
]


@pytest.mark.parametrize(("script", "reason"), FAILED_CALIBRATIONS)
def test_each_failed_calibration_runs_no_global_mapper_and_the_mapper_once_on_the_database(
        solve_job: Callable[..., SolveJob], fake_colmap: type[FakeColmap],
        script: dict[str, Any], reason: str) -> None:
    job = solve_job()
    fake = fake_colmap(incremental_models=FALLBACK_MODELS, **script)
    solved = job.solve(fake.run)
    assert run_names(fake) == [*MATCHED, CALIBRATE, INCREMENTAL]
    assert ran(fake, INCREMENTAL).value("--database_path") == str(job.paths.database)
    assert job.step.warnings == [calibration_warning(reason)]
    assert (solved.calibration_failure, solved.fallback_reason) == (reason, reason)
    assert_fallback_model(job, solved)
    assert set(job.step.timings) == {*MATCHED, CALIBRATE, INCREMENTAL, "matching", "mapping"}


# Each way the Global_Run gives up: FakeColmap's script, and the reason the Req 6.7 warning gives.
GLOBAL_GIVE_UPS = [
    pytest.param({"exits": {GLOBAL: (1, "Failed to create sparse model")}},
                 "stopped with exit code 1 (Failed to create sparse model)", id="exit 1"),
    pytest.param({"exits": {GLOBAL: (-6, "Check failed: num_frames > 0")}},
                 "stopped with exit code -6 (signal SIGABRT) (Check failed: num_frames > 0)",
                 id="exit -6", marks=POSIX_ONLY),
    pytest.param({"global_models": None}, "wrote no model", id="no model"),
    # Models 0 and 2 tie on 3 Picks, under the 4 that 8 Picks need.
    pytest.param({"global_models": (3, 2, 3)}, short_reason(3, PICKS, NEEDED),
                 id="a kept model under the limit"),
]


@pytest.mark.parametrize(("script", "reason"), GLOBAL_GIVE_UPS)
def test_each_way_the_global_mapper_gives_up_runs_the_mapper_once_on_the_database(
        solve_job: Callable[..., SolveJob], fake_colmap: type[FakeColmap],
        script: dict[str, Any], reason: str) -> None:
    job = solve_job()
    fake = fake_colmap(incremental_models=FALLBACK_MODELS, **script)
    solved = job.solve(fake.run)
    assert run_names(fake) == list(ALL_RUNS)
    assert ran(fake, INCREMENTAL).value("--database_path") == str(job.paths.database)
    assert job.step.warnings == [global_warning(reason)]
    assert (solved.calibration_failure, solved.fallback_reason) == (None, reason)
    assert_fallback_model(job, solved)
    assert set(job.step.timings) == {*ALL_RUNS, "matching", "mapping"}


# Timeouts and non-zero exits (Req 2.8, 2.9, 6.10, 6.11, 7.2, 7.3, 13.3)

# Each run that can reach its Time_Limit: COLMAP's command, its Time_Limit key, the script that
# gets solve to it, and the warnings added before it. The mapper runs only as the Fallback, here
# after a Global_Run that wrote no Model.
TIMEOUTS = [
    pytest.param(EXTRACT, "feature_extractor", "feature_extractor_timeout_seconds", {}, [],
                 id=EXTRACT),
    pytest.param(NEAR, "sequential_matcher", "sequential_matcher_timeout_seconds", {}, [],
                 id=NEAR),
    pytest.param(FAR, "sequential_matcher", "sequential_matcher_timeout_seconds", {}, [],
                 id=FAR),
    pytest.param(CALIBRATE, "view_graph_calibrator", "view_graph_calibrator_timeout_seconds",
                 {}, [], id=CALIBRATE),
    pytest.param(GLOBAL, "global_mapper", "global_mapper_timeout_seconds", {}, [], id=GLOBAL),
    pytest.param(INCREMENTAL, "mapper", "mapper_timeout_seconds", {"global_models": None},
                 [global_warning("wrote no model")], id=INCREMENTAL),
]


@pytest.mark.parametrize(("name", "command", "key", "script", "warnings"), TIMEOUTS)
def test_a_timeout_of_each_command_fails_naming_its_time_limit_key_with_no_fallback(
        solve_job: Callable[..., SolveJob], fake_colmap: type[FakeColmap],
        set_cameras: Callable[..., None], name: str, command: str, key: str,
        script: dict[str, Any], warnings: list[str]) -> None:
    set_cameras(**{key: 75.5})  # a Time_Limit of its own, which the message must give
    job = solve_job()
    fake = fake_colmap(timeouts=(name,), **script)
    assert job.failure(fake.run) == (
        f"COLMAP's {command} ran longer than its limit of 75.5 seconds, so the step stopped it. "
        f"Raise {key} in [cameras] in config.toml, then run the step again.")
    runs = ALL_RUNS[:ALL_RUNS.index(name) + 1]  # nothing runs after it, so no Fallback either
    assert run_names(fake) == list(runs)
    # The matching stage's timing only once both Match_Runs have returned, and never the mapping
    # stage's, whose block failed.
    assert set(job.step.timings) == {*runs, *(() if name in MATCHED else ("matching",))}
    assert job.step.timings[name] == 75.5  # FakeColmap's Timeout lasts the Time_Limit
    assert job.step.counts == {}
    assert job.step.warnings == warnings


# A non-zero exit: its code, the words the messages give it, and the Last_Error_Line.
EXITS = [
    pytest.param(1, "exit code 1", "Invalid options provided.", id="exit 1"),
    pytest.param(-6, "exit code -6 (signal SIGABRT)", "Check failed: num_images > 0",
                 id="exit -6", marks=POSIX_ONLY),
]


@pytest.mark.parametrize(("code", "words", "line"), EXITS)
@pytest.mark.parametrize(("name", "command"), [
    pytest.param(EXTRACT, "feature_extractor", id=EXTRACT),
    pytest.param(NEAR, "sequential_matcher", id=NEAR),
    pytest.param(FAR, "sequential_matcher", id=FAR),
])
def test_a_non_zero_exit_of_the_extractor_or_a_match_run_fails_with_its_code_and_last_line(
        solve_job: Callable[..., SolveJob], fake_colmap: type[FakeColmap], name: str,
        command: str, code: int, words: str, line: str) -> None:
    job = solve_job()
    fake = fake_colmap(exits={name: (code, line)})
    log = job.paths.logs / f"{name}.log"
    assert job.failure(fake.run) == (
        f"COLMAP's {command} stopped with {words} ({line}). Check that COLMAP 4.2.1 is installed "
        f"as .kiro/steering/tech.md says, then run the step again. Its output is in {log}.")
    runs = MATCHED[:MATCHED.index(name) + 1]
    assert run_names(fake) == list(runs)
    assert set(job.step.timings) == set(runs)  # the failing run's included, no stage's
    assert job.step.counts == {}
    assert job.step.warnings == []


@pytest.mark.parametrize(("code", "words", "line"), EXITS)
def test_the_mappers_non_zero_exit_fails_saying_it_stopped_with_an_error_with_the_advice(
        solve_job: Callable[..., SolveJob], fake_colmap: type[FakeColmap], code: int,
        words: str, line: str) -> None:
    job = solve_job()
    fake = fake_colmap(global_models=None, exits={INCREMENTAL: (code, line)})
    assert job.failure(fake.run) == (
        f"COLMAP's incremental mapper stopped with an error ({words}: {line}), so the cameras "
        f"couldn't be solved. {ADVICE}")
    assert run_names(fake) == list(ALL_RUNS)
    assert set(job.step.timings) == {*ALL_RUNS, "matching"}
    assert job.step.counts == {}
    assert job.step.warnings == [global_warning("wrote no model")]


def test_the_mappers_exit_0_with_no_model_is_the_placed_failure_with_0_placed(
        solve_job: Callable[..., SolveJob], fake_colmap: type[FakeColmap]) -> None:
    job = solve_job()
    fake = fake_colmap(global_models=None, incremental_models=None)
    assert job.failure(fake.run) == placed_failure(0, PICKS, NEEDED)
    assert run_names(fake) == list(ALL_RUNS)
    assert set(job.step.timings) == {*ALL_RUNS, "matching"}
    assert job.step.counts == {"fallback_ran": 1, "kept_from_global": 0, "models_found": 0,
                               "placed_frames": 0}
    assert job.step.warnings == [global_warning("wrote no model")]


# The Placed_Check at the edges (Req 7.1 to 7.3, C7)

# The Picks, min_placed_frames and min_placed_share at each edge, and the Placed_Frames they need.
# A Kept_Model that places exactly that many passes, and one that places one fewer fails.
EDGES = [
    pytest.param(8, 5, 0.5, 5, id="min_placed_frames"),  # 5 frames, above half the 8 Picks
    pytest.param(8, 1, 0.5, 4, id="min_placed_share"),  # half the 8 Picks, above 1 frame
    # The float 0.3 lies just below 3/10, so with its exact value 10 Picks need 3 (C7).
    pytest.param(10, 1, 0.3, 3, id="0.3 of 10 picks"),
]


@pytest.mark.parametrize(("picks", "min_frames", "min_share", "needed"), EDGES)
def test_placed_needed_gives_the_larger_of_min_placed_frames_and_the_share_of_the_picks(
        picks: int, min_frames: int, min_share: float, needed: int) -> None:
    assert placed_needed(picks, min_frames, min_share) == needed


@pytest.mark.parametrize(("picks", "min_frames", "min_share", "needed"), EDGES)
def test_a_global_kept_model_with_exactly_the_placed_frames_needed_passes_the_placed_check(
        solve_job: Callable[..., SolveJob], fake_colmap: type[FakeColmap],
        set_cameras: Callable[..., None], picks: int, min_frames: int, min_share: float,
        needed: int) -> None:
    set_cameras(min_placed_frames=min_frames, min_placed_share=min_share)
    job = solve_job(picks=picks)
    fake = fake_colmap(global_models=(needed,))
    solved = job.solve(fake.run)
    assert run_names(fake) == list(RADIAL_RUNS)  # no Fallback
    assert (solved.mapper, solved.final) == ("global", ModelCount(0, needed))
    assert job.step.warnings == []


@pytest.mark.parametrize(("picks", "min_frames", "min_share", "needed"), EDGES)
def test_a_global_kept_model_one_short_runs_the_fallback_whose_model_with_exactly_enough_passes(
        solve_job: Callable[..., SolveJob], fake_colmap: type[FakeColmap],
        set_cameras: Callable[..., None], picks: int, min_frames: int, min_share: float,
        needed: int) -> None:
    set_cameras(min_placed_frames=min_frames, min_placed_share=min_share)
    job = solve_job(picks=picks)
    fake = fake_colmap(global_models=(needed - 1,), incremental_models=(needed,))
    solved = job.solve(fake.run)
    assert run_names(fake) == list(ALL_RUNS)
    assert job.step.warnings == [global_warning(short_reason(needed - 1, picks, needed))]
    assert (solved.mapper, solved.final) == ("incremental", ModelCount(0, needed))


@pytest.mark.parametrize(("picks", "min_frames", "min_share", "needed"), EDGES)
def test_a_final_model_one_short_of_the_placed_frames_needed_is_the_placed_failure(
        solve_job: Callable[..., SolveJob], fake_colmap: type[FakeColmap],
        set_cameras: Callable[..., None], picks: int, min_frames: int, min_share: float,
        needed: int) -> None:
    set_cameras(min_placed_frames=min_frames, min_placed_share=min_share)
    job = solve_job(picks=picks)
    fake = fake_colmap(global_models=(needed - 1,), incremental_models=(needed - 1,))
    assert job.failure(fake.run) == placed_failure(needed - 1, picks, needed)
    assert run_names(fake) == list(ALL_RUNS)
    assert job.step.counts == {"fallback_ran": 1, "kept_from_global": 0, "models_found": 1,
                               "placed_frames": needed - 1}


# model_counts (C11)

def model_of(names: Sequence[str]) -> Model:
    """A Model with one PINHOLE camera and an image for each of `names`, with no 2D points and no
    points."""
    camera = Camera(camera_id=1, model_id=1, width=16, height=12, params=(20.0, 20.0, 8.0, 6.0))
    images = tuple(Image(image_id=place + 1, qvec=(1.0, 0.0, 0.0, 0.0), tvec=(0.0, 0.0, 0.0),
                         camera_id=1, name=name, points2d=np.zeros(0, dtype=POINT2D))
                   for place, name in enumerate(names))
    points = Points(ids=np.zeros(0, dtype="<u8"), xyz=np.zeros((0, 3)),
                    rgb=np.zeros((0, 3), dtype="u1"), errors=np.zeros(0), tracks=())
    return Model(cameras=(camera,), images=images, points=points)


def test_model_counts_ignores_folders_whose_names_arent_numbers_and_images_that_arent_picks(
        tmp_path: Path) -> None:
    numbers = {name: number for number, name in enumerate(pick_names(range(PICKS)))}
    output = tmp_path / "global"
    write_model(output / "10", model_of(pick_names((0, 1))))
    # 000099.jpg is named like a Pick, but there's no Pick 99.
    write_model(output / "2", model_of(("000002.jpg", "other.jpg", "000099.jpg")))
    write_model(output / "0", model_of(pick_names((3, 4, 5))))
    for name in ("sparse", "1a"):  # each holds a Model that places every Pick
        write_model(output / name, model_of(pick_names(range(PICKS))))
    (output / "project.ini").write_text("", encoding="utf-8")
    assert model_counts(output, numbers) == (ModelCount(0, 3), ModelCount(2, 1),
                                             ModelCount(10, 2))


# Property 2: The Kept_Model has the most Placed_Frames and the lowest number among ties (Req 6.8,
# 6.12)

# One mapper run's Models: 1 to 20 of them, with unique numbers from 0 to 60 in the order drawn,
# which often isn't number order, and Placed_Frames counts from 0 to 4, so ties are common.
RUN_MODELS = st.lists(st.builds(ModelCount, number=st.integers(0, 60), placed=st.integers(0, 4)),
                      min_size=1, max_size=20, unique_by=lambda model: model.number)


@given(RUN_MODELS)
def test_the_kept_model_has_the_most_placed_frames_and_the_lowest_number_among_ties(
        models: list[ModelCount]) -> None:
    """Feature: cameras, Property 2: The Kept_Model has the most Placed_Frames and the lowest
    number among ties

    **Validates: Requirements 6.8, 6.12**
    """
    kept = kept_model(models)
    assert kept is not None and kept in models  # one of the run's own Models
    # The oracle, a plain loop. The numbers are unique, so every other Model has another number.
    for other in models:
        if other.number != kept.number:
            assert kept.placed >= other.placed
            if other.placed == kept.placed:
                assert kept.number < other.number


# The database helpers (Req 5.2, 5.4, F12, O5, C1)

def test_copy_database_copies_every_table_and_leaves_the_database_as_it_was(
        tmp_path: Path) -> None:
    database, copy = tmp_path / "database.db", tmp_path / "database_calibrated.db"
    write_database(database, START)
    with closing(sqlite3.connect(database)) as connection, connection:
        connection.execute("CREATE TABLE images (image_id INTEGER PRIMARY KEY NOT NULL, "
                           "name TEXT NOT NULL UNIQUE, camera_id INTEGER NOT NULL)")
        connection.executemany("INSERT INTO images VALUES (?, ?, ?)",
                               [(1, "000000.jpg", 1), (2, "000001.jpg", 1)])
    before = dump(database)
    copy_database(database, copy)
    assert dump(copy) == before
    assert dump(database) == before


def test_read_camera_reads_the_one_camera_row_with_its_params_as_little_endian_doubles(
        tmp_path: Path) -> None:
    path = tmp_path / "database.db"
    params = (19.2, 8.0, 6.0, 0.02, -0.01)  # a RADIAL camera's f, cx, cy, k1, k2
    write_database(path, (3, 16, 12, params, 1))
    assert read_camera(path) == DatabaseCamera(camera_id=1, model_id=3, width=16, height=12,
                                               params=params, prior_focal_length=True)


def test_read_camera_refuses_a_cameras_table_with_two_rows(tmp_path: Path) -> None:
    path = tmp_path / "database.db"
    write_database(path, START, START)
    with pytest.raises(ExpectedFailure) as caught:
        read_camera(path)
    message = str(caught.value)
    assert str(path) in message
    assert "2 cameras" in message.replace(str(path), "")
    assert "Check that COLMAP 4.2.1 is installed" in message
    assert "run the step again" in message
    assert "Traceback" not in message


# What a Calibration can leave in the Calibration_Copy's RADIAL camera: its f, params[0], and
# its prior_focal_length, with the reason calibration_failure gives, or None when it worked (O5,
# F11, F17).
CALIBRATED = [
    pytest.param(0.9 * 19.2, 1, None, id="worked"),
    pytest.param(math.nextafter(19.2, math.inf), 1, None, id="worked, by the smallest step"),
    pytest.param(19.2, 1, "rejected its focal length estimate and kept the starting one",
                 id="rejected"),
    pytest.param(19.2, 0, "found no image pairs to calibrate", id="no pairs"),
]


@pytest.mark.parametrize(("focal", "prior", "reason"), CALIBRATED)
def test_calibration_failure_compares_the_copys_camera_with_the_databases(
        tmp_path: Path, focal: float, prior: int, reason: str | None) -> None:
    database, copy = tmp_path / "database.db", tmp_path / "database_calibrated.db"
    write_database(database, START)
    copy_database(database, copy)
    with closing(sqlite3.connect(copy)) as connection, connection:
        connection.execute("UPDATE cameras SET params = ?, prior_focal_length = ?",
                           (blob((focal, *START_PARAMS[1:])), prior))
    assert calibration_failure(read_camera(database), read_camera(copy)) == reason
