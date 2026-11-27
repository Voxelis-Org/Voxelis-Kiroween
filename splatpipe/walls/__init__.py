"""The Walls_Step: corrects each frame's depth prior with a perspective bas-relief change.

It runs between the Depth_Step and the Train_Step, an Optional_Step. It reads the depth priors in
work/depth/, the camera solve in work/cameras/ and the Frames_Record in work/frames/, and writes
its own priors to work/walls/ in exactly the Prior_Format, at the Depth_Step's Prior_Size, so the
Train_Step reads them with no new code. Its settings are in [walls] in config.toml, and
settings.py reads and checks them. When it fails, the job goes on and trains on the depth priors.

The package is built like the Fake_Step and the Depth_Step (Req 1.1). `work(step)` is the step's
own code, which the Step_Runner calls once it has emptied work/walls/, so no file from an earlier
run is left there (Req 1.10). `run(job_id, preset=None, *, config=None)` runs the step through the
Step_Runner and returns its Step_Report. `__main__.py` is the CLI, `uv run python -m
splatpipe.walls --job <id> [--preset <name>]` (Req 1.2).

`work` goes in the order of Req 1.4 (D15):

1. remove_demos: out/walls/, first of all, so no Walls_Demo of an earlier run is left (Req 1.10).
2. The settings, every one checked (Req 3.4, 3.5).
3. The Cap_Deadline, with the Train_Step's read_cap_deadline (Req 4.1). The clock is read once,
   through the hook utc_now: the step's start, from which walls_times gives the Walls_Stop_Time
   and which limit set it, and solve_cutoff the Solve_Cutoff (Req 4.2, 4.3, 4.10).
4. check_inputs: the input files must be there. Nothing is read (Req 2.2 to 2.5).
5. Timed as read: read_inputs, the Sparse_Model, the Depth_Prior_Set and the Frames_Record,
   joined into the Prior_Frames in Pick_Number order (Req 2), then the Preparation_Pass,
   prepare_frame per Prior_Frame on this process, which writes its arrays to work/walls/scratch/
   and gives its Scene_Scale and Scale_Points, so a Timed_Out_Frame has them too (Req 7.6, C3).
6. The hook WALLS_POOL gives the Worker_Pool, with worker_count's workers (Req 5.1, 5.5).
7. Timed as solve: the Frame_Solves, solve.solve_frame in the Solve_Order, through
   map_frames_until with the Solve_Cutoff and utc_now as its clock. A Frame_Solve not ended by
   the Solve_Cutoff, started or not, is a Timed_Out_Frame (Req 4.4 to 4.6). Every other Prior_Frame
   is a candidate Source_Frame of each frame, the Timed_Out_Frames included, because the
   Preparation_Pass wrote their arrays (Req 4.8). Which frames are Trusted_Frames comes from the
   pool's results, never from which scratch files exist.
8. With no Trusted_Frame, the Req 18.8 failure, before anything is written (A15).
9. The cross-frame stages, each per-frame stage through map_frames, each a barrier (Req 5.3,
   14.4, A11): timed as fuse, the fuse of each Trusted_Frame; timed as planes, the World_Planes,
   once on this process (D15); timed as fallback, the fallback of each Failed_Frame; timed as
   correct, flatten and despeckle, those stages for every frame with a Walls_Depth, a Trusted_Frame
   or a Failed_Frame with a Fallback_Depth; then the per-frame choice of every Prior_Frame
   (Req 17). A Timed_Out_Frame gets none of these but the choice, which gives it a
   Depth_Prior_Copy (Req 4.7, 4.9).
10. Timed as write, the Walls_Prior_Files, one per Prior_Frame in Pick_Number order (Req 17.1,
    18.1). Then work/walls/scratch/ is removed, and the Walls_Record is written, also timed as
    write (Req 18.2).
11. Timed as demo, the Walls_Demos (Req 18.3).
12. Timed as write too, the Walls_Index, last of all: the Depth_Index copied unchanged (Req 18.4,
    18.5, D7). A failure at any point before it leaves no Walls_Index, so the Train_Step reads
    work/depth/ (Req 18.7).

A finally removes work/walls/scratch/ and closes the Worker_Pool, done or failed, so no worker
process is left running and no reader or upload sees the scratch folder (Req 5.6).

Each count is added as an int as soon as it's known, and each timing when its block ends, so a
failed report holds exactly the counts and timings the step had worked out (Req 19.3, 19.4).
The Walls_Record holds only JSON_Values: a number that isn't finite or wasn't worked out is null
(Req 18.2). The step writes only inside work/walls/ and out/walls/ (Req 1.8, 1.9).

`work` reads the module constants WALLS_POOL and utc_now each time it runs, so a test that
replaces them through monkeypatch, with an InlinePool and a fake clock, reaches the step through
run, run_step, the CLI and run_job (Req 5.5, C8).

Every module of the package imports only the standard library, numpy, scipy, cv2 and splatpipe,
and pool.py adds multiprocessing, so the step runs on the CPU only (Req 1.11). From splatpipe,
this module imports only splatpipe.step, the Train_Step's CPU modules colmap, deadline, files,
priors and scene, and the package's own modules.
"""

from __future__ import annotations

import math
import os
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from splatpipe.step import Config, StepContext, run_step
from splatpipe.train import deadline
from splatpipe.train.colmap import Camera
from splatpipe.train.files import CAMERAS, IMAGES
from splatpipe.train.priors import Prior, PriorSet
from splatpipe.train.scene import world_to_camera
from splatpipe.walls.budget import solve_cutoff, solve_order, walls_times
from splatpipe.walls.crossframe import DEPTH_PRIOR, FALLBACK, neighbours
from splatpipe.walls.geometry import normalised_coords
from splatpipe.walls.inputs import FrameInput, Prepared, check_inputs, prepare_frame, read_inputs
from splatpipe.walls.outputs import (SCRATCH, demo_panel, demo_path, depth_prior_copy,
                                     remove_demos, remove_scratch, walls_prior, write_demo,
                                     write_prior_file, write_record, write_walls_index)
from splatpipe.walls.planes import WorldPlane
from splatpipe.walls.pool import Pool, open_pool, worker_count
from splatpipe.walls.report import (Counts, Timings, no_trusted_failure, not_all_passed_warning,
                                    timed_out_warning)
from splatpipe.walls.settings import WallsSettings, read_settings
from splatpipe.walls.solve import (ChoiceResult, ChoiceTask, CorrectTask, DespeckleTask,
                                   FallbackResult, FallbackTask, FlattenTask, FrameSolve,
                                   FuseTask, SolveTask, choice_stage, correct_stage,
                                   despeckle_stage, fallback_stage, flatten_stage, fuse_stage,
                                   load_walls, scratch_world_planes, solve_frame)
from splatpipe.walls.sources import Candidate

# The hook to the Worker_Pool (Req 5.5, A16). work reads it each time it runs, so a test can
# monkeypatch it with a factory that gives an InlinePool.
WALLS_POOL: Callable[[int], Pool] = open_pool
# The clock (C8). work reads it each time it runs, once for the step's start and then as the
# Frame_Solves' clock, so a test can monkeypatch it with a fake one.
utc_now: Callable[[], datetime] = deadline.utc_now

# Walls_Fixed_Constants: the Walls_Record's format number (D7).
RECORD_FORMAT = 1


@dataclass
class _Run:
    """What one run of the step holds while it works: its context, settings, report fields and
    inputs, and the outcomes of each stage, by Prior_Frame name."""

    step: StepContext
    cfg: WallsSettings
    counts: Counts
    timings: Timings
    scratch: Path  # work/walls/scratch/
    priors: PriorSet  # the Depth_Prior_Set
    camera: Camera  # the Shared_Camera
    frames: tuple[FrameInput, ...]  # the Prior_Frames, in Pick_Number order
    prepared: dict[str, Prepared]
    poses: dict[str, np.ndarray]  # each Prior_Frame's world-to-camera matrix
    pool: Pool | None = None
    workers: int = 0
    solves: dict[str, FrameSolve] = field(default_factory=dict)
    timed_out: frozenset[str] = frozenset()
    # The Trusted_Frames' Pick_Numbers by name, in Pick_Number order, from the pool's results.
    trusted: dict[str, int] = field(default_factory=dict)
    planes: tuple[WorldPlane, ...] = ()
    fallbacks: dict[str, FallbackResult] = field(default_factory=dict)
    choices: dict[str, ChoiceResult] = field(default_factory=dict)

    @property
    def failed(self) -> list[FrameInput]:
        """The Failed_Frames: their Frame_Solve ended and didn't pass, in Pick_Number order."""
        return [frame for frame in self.frames
                if frame.name in self.solves and not self.solves[frame.name].passed]

    def has_walls_depth(self, name: str) -> bool:
        """Whether the Prior_Frame `name` gets a Walls_Depth: a Trusted_Frame, or a Failed_Frame
        with a Fallback_Depth."""
        if name in self.trusted:
            return True
        found = self.fallbacks.get(name)
        return found is not None and found.has_fallback

    def neighbours_of(self, frame: FrameInput) -> tuple[tuple[str, ...], tuple[np.ndarray, ...]]:
        """The Neighbour_Frames of `frame`, nearest first, and their poses (Req 14.1, D14)."""
        names = tuple(neighbours(frame.name, frame.pick_number, self.trusted,
                                 self.cfg.neighbour_window, self.cfg.neighbour_frames))
        return names, tuple(self.poses[name] for name in names)


def run(job_id: str, preset: str | None = None, *,
        config: Config | None = None) -> dict[str, Any]:
    """Run the walls step on the Job `job_id` and return its Step_Report.

    `preset` replaces the Job_Record's Preset, and `config` defaults to load_config().
    """
    return run_step("walls", "splatpipe.walls", job_id, preset=preset, config=config)


def work(step: StepContext) -> None:
    """The walls step's own code. The Step_Runner turns what it raises into a failed report."""
    remove_demos(step.out_folder)  # first of all (Req 1.10)
    cfg = read_settings(step.settings, step.name, step.preset)  # Req 3.4, 3.5
    cap_deadline = deadline.read_cap_deadline(os.environ)  # Req 4.1
    clock = utc_now  # read now, so a test's monkeypatch is seen (C8)
    started = clock()  # the step's start, read once (Req 4.2)
    stop, stop_limit = walls_times(started, cap_deadline, cfg.max_minutes,
                                   cfg.train_reserve_minutes)
    cutoff = solve_cutoff(stop, cfg.finish_minutes)  # Req 4.3
    check_inputs(step.job.folder)  # Req 2.5: after the settings and the Cap_Deadline
    scratch = step.work_folder / SCRATCH
    walls: _Run | None = None
    try:
        walls = _read(step, cfg, scratch)
        walls.workers = worker_count(cfg.workers, len(walls.frames))  # Req 5.1
        walls.pool = WALLS_POOL(walls.workers)  # read now, so a test's monkeypatch is seen
        _solve(walls, cutoff, clock, stop, stop_limit)
        _cross_frame(walls)
        _write(walls, _times(started, cap_deadline, stop, cutoff, stop_limit))
    finally:
        try:
            remove_scratch(step.work_folder)  # done or failed (Req 5.6)
        finally:
            if walls is not None and walls.pool is not None:
                walls.pool.close()


def _read(step: StepContext, cfg: WallsSettings, scratch: Path) -> _Run:
    """Timed as read: the inputs, then the Preparation_Pass, which writes every Prior_Frame's
    scratch arrays (Req 2, 7, C3). Adds `images` as soon as the Prior_Frames are known."""
    counts, timings = Counts(), Timings()
    with timings.timed(step, "read"):
        model, priors, frames = read_inputs(step.job.folder)
        counts.add(step, "images", len(frames))
        cameras = step.job.folder / "work" / CAMERAS
        prepared = {frame.name: prepare_frame(scratch, cameras, model, frame, cfg)
                    for frame in frames}
    poses = {frame.name: world_to_camera(np.asarray(frame.image.quaternion, dtype=np.float64),
                                         np.asarray(frame.image.translation, dtype=np.float64))
             for frame in frames}
    return _Run(step=step, cfg=cfg, counts=counts, timings=timings, scratch=scratch,
                priors=priors, camera=model.camera, frames=frames, prepared=prepared,
                poses=poses)


def _solve(walls: _Run, cutoff: datetime, clock: Callable[[], datetime], stop: datetime,
           stop_limit: str) -> None:
    """Timed as solve: the Frame_Solves in the Solve_Order until the Solve_Cutoff (Req 4.4 to
    4.6, 5.2). Adds solved_images, timed_out_images and passed_images, then the Req 19.5 warning
    when some timed out, and fails with Req 18.8's message when no frame passed."""
    step, pool = walls.step, walls.pool
    assert pool is not None
    candidates = [Candidate(name=frame.name, pick_number=int(frame.pick_number),
                            frame_time=float(frame.frame_time),
                            world_to_camera=walls.poses[frame.name])
                  for frame in walls.frames]
    order = solve_order(len(walls.frames)) if walls.frames else []
    tasks = [_solve_task(walls, candidates, index) for index in order]
    with walls.timings.timed(step, "solve"):
        mapped = pool.map_frames_until(solve_frame, tasks, cutoff, clock)
    walls.solves = {tasks[index].name: solved for index, solved in mapped.results.items()}
    walls.timed_out = frozenset(mapped.timed_out)
    walls.trusted = {frame.name: int(frame.pick_number) for frame in walls.frames
                     if frame.name in walls.solves and walls.solves[frame.name].passed}
    images = len(walls.frames)
    walls.counts.add(step, "solved_images", len(walls.solves))
    walls.counts.add(step, "timed_out_images", len(walls.timed_out))
    passed = len(walls.trusted)
    walls.counts.add(step, "passed_images", passed)
    if walls.timed_out:
        step.add_warning(timed_out_warning(len(walls.timed_out), images, stop, stop_limit))
    if passed == 0:
        too_few = sum(1 for solved in walls.solves.values() if solved.result is None)
        raise no_trusted_failure(images, len(walls.timed_out), len(walls.solves), too_few)


def _solve_task(walls: _Run, candidates: Sequence[Candidate], index: int) -> SolveTask:
    """The Frame_Solve task of the Prior_Frame at `index` in Pick_Number order, with every other
    Prior_Frame as a candidate Source_Frame, the Timed_Out_Frames included (Req 4.8)."""
    frame = walls.frames[index]
    return SolveTask(name=frame.name, pick_number=int(frame.pick_number),
                     frame_time=float(frame.frame_time), world_to_camera=walls.poses[frame.name],
                     settings=walls.cfg, camera=walls.camera, scratch=walls.scratch,
                     candidates=tuple(candidate for candidate in candidates
                                      if candidate.name != frame.name))


def _cross_frame(walls: _Run) -> None:
    """The cross-frame stages after the Frame_Solves, in D15's order, each per-frame stage
    through the Worker_Pool's map_frames, then the per-frame choice of every Prior_Frame, which
    adds fallback_images and depth_prior_images and the Req 19.6 warning (Req 5.3, 13 to 17)."""
    step, cfg, pool, timings = walls.step, walls.cfg, walls.pool, walls.timings
    assert pool is not None
    trusted = [frame for frame in walls.frames if frame.name in walls.trusted]
    with timings.timed(step, "fuse"):
        pool.map_frames(fuse_stage, [_fuse_task(walls, frame) for frame in trusted])
    with timings.timed(step, "planes"):  # once, on this process (D15)
        walls.planes = tuple(scratch_world_planes(
            walls.scratch, {frame.name: walls.solves[frame.name].facets for frame in trusted},
            walls.poses, walls.camera, cfg))
    with timings.timed(step, "fallback"):
        found = pool.map_frames(fallback_stage,
                                [_fallback_task(walls, frame) for frame in walls.failed])
    walls.fallbacks = {result.name: result for result in found}
    with_depth = [frame for frame in walls.frames if walls.has_walls_depth(frame.name)]
    with timings.timed(step, "correct"):
        pool.map_frames(correct_stage, [_correct_task(walls, frame) for frame in with_depth])
    with timings.timed(step, "flatten"):
        pool.map_frames(flatten_stage,
                        [FlattenTask(name=frame.name, scratch=walls.scratch,
                                     world_to_camera=walls.poses[frame.name], camera=walls.camera,
                                     planes=walls.planes, trusted=frame.name in walls.trusted,
                                     settings=cfg) for frame in with_depth])
    with timings.timed(step, "despeckle"):
        pool.map_frames(despeckle_stage,
                        [DespeckleTask(name=frame.name, scratch=walls.scratch, settings=cfg)
                         for frame in with_depth])
    chosen = pool.map_frames(choice_stage, [_choice_task(walls, frame) for frame in walls.frames])
    walls.choices = {result.name: result for result in chosen}
    _count_outcomes(walls)


def _fuse_task(walls: _Run, frame: FrameInput) -> FuseTask:
    """The fuse of the Trusted_Frame `frame` with its Neighbour_Frames (Req 14.3)."""
    names, poses = walls.neighbours_of(frame)
    return FuseTask(name=frame.name, scratch=walls.scratch,
                    world_to_camera=walls.poses[frame.name], camera=walls.camera,
                    neighbours=names, neighbour_poses=poses, settings=walls.cfg)


def _fallback_task(walls: _Run, frame: FrameInput) -> FallbackTask:
    """The fallback of the Failed_Frame `frame` (Req 15)."""
    names, poses = walls.neighbours_of(frame)
    return FallbackTask(name=frame.name, scratch=walls.scratch,
                        world_to_camera=walls.poses[frame.name], camera=walls.camera,
                        neighbours=names, neighbour_poses=poses, planes=walls.planes,
                        settings=walls.cfg)


def _correct_task(walls: _Run, frame: FrameInput) -> CorrectTask:
    """The correction of `frame`, which has a Walls_Depth (Req 16.1)."""
    names, poses = walls.neighbours_of(frame)
    return CorrectTask(name=frame.name, scratch=walls.scratch,
                       world_to_camera=walls.poses[frame.name], camera=walls.camera,
                       neighbours=names, neighbour_poses=poses,
                       trusted=frame.name in walls.trusted, settings=walls.cfg)


def _choice_task(walls: _Run, frame: FrameInput) -> ChoiceTask:
    """The per-frame choice of `frame`, every Prior_Frame (Req 17)."""
    found = walls.fallbacks.get(frame.name)
    return ChoiceTask(name=frame.name, scratch=walls.scratch,
                      trusted=frame.name in walls.trusted,
                      timed_out=frame.name in walls.timed_out,
                      has_fallback=found is not None and found.has_fallback,
                      scale_points=int(walls.prepared[frame.name].scale_points),
                      settings=walls.cfg)


def _count_outcomes(walls: _Run) -> None:
    """Add fallback_images and depth_prior_images from the per-frame choice, and the Req 19.6
    warning when fewer than every Prior_Frame passed (Req 19.1, 19.6)."""
    outcomes = [walls.choices[frame.name].outcome for frame in walls.frames]
    fallback = outcomes.count(FALLBACK)
    depth_prior = outcomes.count(DEPTH_PRIOR)
    walls.counts.add(walls.step, "fallback_images", fallback)
    walls.counts.add(walls.step, "depth_prior_images", depth_prior)
    passed, images = len(walls.trusted), len(walls.frames)
    if passed < images:
        walls.step.add_warning(not_all_passed_warning(passed, fallback, depth_prior, images))


def _write(walls: _Run, times: dict[str, Any]) -> None:
    """The Walls_Prior_Files, then the scratch folder's removal, the Walls_Record, the
    Walls_Demos and, last of all, the Walls_Index (Req 18.1 to 18.5). write times the files,
    the record and the index, and demo the demos (Req 19.2)."""
    step, timings = walls.step, walls.timings
    demos = _demo_names(walls)
    seconds = _Stopwatch()
    with seconds:
        kept = _write_priors(walls, frozenset(demos))
    remove_scratch(step.work_folder)  # no reader and no upload sees it
    with seconds:
        write_record(step.work_folder, _record(walls, times, demos))
    timings.add(step, "write", seconds.total)
    with timings.timed(step, "demo"):
        _write_demos(walls, demos, kept)
    with seconds:
        write_walls_index(step.work_folder, walls.priors.index)  # last of all (Req 18.5)
    timings.add(step, "write", seconds.total)


class _Stopwatch:
    """A running sum of the seconds spent in its `with` blocks that ended without an error."""

    def __init__(self) -> None:
        self.total = 0.0
        self._started = 0.0

    def __enter__(self) -> _Stopwatch:
        self._started = time.perf_counter()
        return self

    def __exit__(self, kind: object, error: object, trace: object) -> None:
        if kind is None:
            self.total += time.perf_counter() - self._started


def _write_priors(walls: _Run, demos: frozenset[str]) -> dict[str, Prior]:
    """Write every Prior_Frame's Walls_Prior, in Pick_Number order: from its Walls_Depth when its
    outcome isn't depth_prior, else its Depth_Prior_Copy (Req 17, 18.1). Gives the Walls_Priors
    of the demo frames `demos`, the only ones kept in memory."""
    kept: dict[str, Prior] = {}
    for frame in walls.frames:
        depth_prior = walls.priors.priors[frame.name]
        if walls.choices[frame.name].outcome == DEPTH_PRIOR:
            prior = depth_prior_copy(depth_prior)
        else:
            depth, normal = load_walls(walls.scratch, frame.name)
            prior = walls_prior(depth_prior, depth, normal, depth_prior.mask)
        write_prior_file(walls.step.work_folder, frame.name, prior)
        if frame.name in demos:
            kept[frame.name] = prior
    return kept


def _demo_names(walls: _Run) -> list[str]:
    """The Walls_Demos' Prior_Frames: among the N Prior_Frames whose Walls_Prior isn't a
    Depth_Prior_Copy, in Pick_Number order, those at ⌊(i + 0.5) × N / demo_frames⌋, or every
    one when N is below demo_frames, and none when demo_frames is 0 (Req 18.3)."""
    made = [frame.name for frame in walls.frames
            if walls.choices[frame.name].outcome != DEPTH_PRIOR]
    count, wanted = len(made), walls.cfg.demo_frames
    if wanted <= 0:
        return []
    if count < wanted:
        return made
    return [made[int(math.floor((index + 0.5) * count / wanted))] for index in range(wanted)]


def _write_demos(walls: _Run, demos: Sequence[str], kept: Mapping[str, Prior]) -> None:
    """Write the Walls_Demo of each Prior_Frame in `demos`, out/walls/<stem>.jpg (Req 18.3)."""
    if not demos:
        return
    images = walls.step.job.folder / "work" / CAMERAS / IMAGES
    height, width = walls.priors.index.prior_height, walls.priors.index.prior_width
    x, y = normalised_coords(walls.camera, width, height)
    for name in demos:
        panel = demo_panel(images, name, walls.camera, walls.priors.priors[name], kept[name], x, y)
        write_demo(demo_path(walls.step.out_folder, name), panel, walls.cfg.demo_jpeg_quality)


# The Walls_Record (D7, Req 18.2)

def _times(started: datetime, cap_deadline: datetime | None, stop: datetime, cutoff: datetime,
           stop_limit: str) -> dict[str, Any]:
    """The run's times in the Walls_Record, in UTC (Req 4.11)."""
    return {"started_utc": _utc(started),
            "cap_deadline": None if cap_deadline is None else _utc(cap_deadline),
            "walls_stop_time": _utc(stop), "solve_cutoff": _utc(cutoff),
            "stop_limit": stop_limit}


def _record(walls: _Run, times: Mapping[str, Any], demos: Sequence[str]) -> dict[str, Any]:
    """The Walls_Record: its format, the run's times, the workers, the World_Planes, the
    Walls_Demos' names and one object per Prior_Frame in Pick_Number order (Req 18.2)."""
    return {"format": RECORD_FORMAT, **times, "workers": int(walls.workers),
            "world_planes": len(walls.planes), "demo_frames": list(demos),
            "frames": [_frame_record(walls, frame) for frame in walls.frames]}


def _frame_record(walls: _Run, frame: FrameInput) -> dict[str, Any]:
    """One Prior_Frame's object in the Walls_Record, null for each number that isn't finite or
    wasn't worked out (Req 7.6, 8.8, 10.9, 11.9, 12.8, 15.5, 17.10)."""
    prepared = walls.prepared[frame.name]
    solved = walls.solves.get(frame.name)
    choice = walls.choices[frame.name]
    record: dict[str, Any] = {"name": frame.name, "pick_number": int(frame.pick_number),
                              "scene_scale": _number(prepared.k),
                              "scale_points": int(prepared.scale_points)}
    record.update(_solve_record(solved, prepared))
    record.update(_fallback_record(walls.fallbacks.get(frame.name)))
    record.update({"outcome": choice.outcome, "reason": choice.reason,
                   "sparse_error": _number(choice.sparse_error)})
    return record


def _solve_record(solved: FrameSolve | None, prepared: Prepared) -> dict[str, Any]:
    """The Frame_Solve's fields of a Prior_Frame's record: all null for a Timed_Out_Frame, and
    the passes' fields null for one that had too few Source_Frames (Req 8.7)."""
    result = None if solved is None else solved.result
    return {
        "sources": None if solved is None else len(solved.sources),
        "source_frames": None if solved is None else [source.name for source in solved.sources],
        "source_seconds": None if solved is None else _number(solved.seconds_cut),
        "first_scale": None if result is None else _number(result.first_scale),
        "change": None if result is None else _numbers(result.change),
        "m": _number(prepared.m if result is None else result.m),
        "band_weights": None if result is None else _numbers(result.band_weights),
        "halves_difference": None if result is None else _number(result.halves_difference),
        "tile_low": None if result is None else _number(result.tile_low),
        "tile_high": None if result is None else _number(result.tile_high),
        "tile_low_count": None if result is None else int(result.tile_low_count),
        "tile_high_count": None if result is None else int(result.tile_high_count),
        "passed": None if solved is None else bool(solved.passed),
        "facets": None if solved is None else len(solved.facets),
        "facet_changes": None if solved is None else len(solved.applied),
        "applied_facets": None if solved is None else [
            {"facet": int(change.facet), "change": _numbers(change.change),
             "m": _number(change.m), "halves_difference": _number(change.halves_difference),
             "tile_low": _number(change.tile_low), "tile_high": _number(change.tile_high)}
            for change in solved.applied],
    }


def _fallback_record(found: FallbackResult | None) -> dict[str, Any]:
    """A Failed_Frame's fallback, its share of Covered_Pixels with a median and its fitted
    change, all null for a frame that had no fallback stage (Req 15.5)."""
    if found is None:
        return {"fallback": None, "fallback_coverage": None, "fallback_change": None}
    return {"fallback": found.fallback, "fallback_coverage": _number(found.coverage),
            "fallback_change": None if found.change is None else _numbers(found.change)}


def _number(value: object) -> float | None:
    """`value` as a plain float, or None when it's None or not finite (Req 18.2)."""
    if value is None:
        return None
    number = float(value)  # type: ignore[arg-type]
    return number if math.isfinite(number) else None


def _numbers(values: Sequence[float] | np.ndarray) -> list[float | None]:
    """Each of `values` as a plain float, None where it isn't finite."""
    return [_number(value) for value in np.asarray(values, dtype=np.float64).reshape(-1)]


def _utc(moment: datetime) -> str:
    """A time in UTC as ISO text, to the millisecond, as the Step_Report writes its times."""
    return moment.astimezone(timezone.utc).isoformat(timespec="milliseconds")
