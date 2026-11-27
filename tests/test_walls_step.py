"""Unit tests for the Walls_Step as a whole, run through splatpipe.walls.run (walls spec, Req 1,
4, 5.7, 17 to 19, 22.5 and 22.7).

Every run goes through the Step_Runner, so each test checks the real Step_Report in
out/reports/walls.json. The walls_pool fixture puts InlinePool in place of the Worker_Pool, so
no worker process starts, and FakeClock in place of splatpipe.walls.utc_now, so the Walls_Stop_Time
and the Solve_Cutoff come from times the test chooses. Each Job is made with new_job in the test
config's Jobs_Folder and filled with write_walls_job from tests/camera_files.py: the textured
plane, whose 12 frames all pass their Self_Check, or the plain plane, where none do.

The real step on the textured plane takes several seconds, so the tests that only read a done
run's outputs share one: done_run runs it in the first test that asks for it and keeps what it
left in walls_cache, a session-scoped dict. The failure tests are cheap, because each fails
before any Frame_Solve.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from splatpipe import walls
from splatpipe.config import load_config
from splatpipe.jsonio import read_json
from splatpipe.run import StepEntry, run_job
from splatpipe.train.priors import Prior, read_prior_set
from splatpipe.walls.__main__ import main as walls_main

from camera_files import WallsJob, write_walls_job
from conftest import WALLS, WALLS_PRESETS

# The fixed start the fake clock gives the step, in UTC.
START = datetime(2026, 11, 19, 14, 30, tzinfo=timezone.utc)
# The Job's Preset: new_job's Jobs are made with the test config's default_preset.
PRESET = "fast"
# The textured plane's 12 frames, in Pick_Number order.
FRAMES = tuple(f"{number:06d}.jpg" for number in range(12))
# The Walls_Demos of a run in which every frame gets a Walls_Prior from its Walls_Depth: the
# places ⌊(i + 0.5) × 12 / 3⌋, 2, 6 and 10 (Req 18.3).
DEMOS = ("000002.jpg", "000006.jpg", "000010.jpg")

# The report's six counts and ten timings (Req 19.1, 19.2).
COUNTS = ("images", "solved_images", "passed_images", "fallback_images", "depth_prior_images",
          "timed_out_images")
TIMINGS = ("read", "solve", "planes", "fuse", "fallback", "correct", "flatten", "despeckle",
           "write", "demo")
# The Walls_Record's fields (design, "The Walls_Record"), and each frame's, with the per-frame
# source_frames that the step adds: the Source_Frames' names in rank order.
RECORD_FIELDS = {"format", "started_utc", "cap_deadline", "walls_stop_time", "solve_cutoff",
                 "stop_limit", "workers", "world_planes", "demo_frames", "frames"}
FRAME_FIELDS = {"name", "pick_number", "scene_scale", "scale_points", "sources", "source_seconds",
                "first_scale", "change", "m", "band_weights", "halves_difference", "tile_low",
                "tile_high", "tile_low_count", "tile_high_count", "passed", "facets",
                "facet_changes", "applied_facets", "fallback", "fallback_coverage",
                "fallback_change", "outcome", "reason", "sparse_error", "source_frames"}
PRIOR_ARRAYS = ("inverse_depth", "normal", "mask")
CAP_DEADLINE_ENV = "SPLATPIPE_CAP_DEADLINE"
STALE_DEMO = "out/walls/stale.jpg"  # a Walls_Demo an earlier run left


class FakeClock:
    """A clock that gives `times` in turn, one per call, and then the last of them forever."""

    def __init__(self, times: Sequence[datetime]) -> None:
        self.times = list(times)
        self.calls = 0

    def __call__(self) -> datetime:
        moment = self.times[min(self.calls, len(self.times) - 1)]
        self.calls += 1
        return moment


@dataclass(frozen=True)
class DoneRun:
    """What the shared done run on the textured plane left."""

    job: WallsJob
    report: dict[str, Any]
    before: dict[str, bytes]  # every file of the Job_Folder before the run, by its path
    after: dict[str, bytes]   # and after it
    folders: tuple[str, ...]  # every folder of the Job_Folder after the run
    workers: tuple[int, ...]  # the worker counts the step opened pools with


# Helpers

def folder_of(job_id: str) -> Path:
    """The Job_Folder of `job_id` in the test config's Jobs_Folder."""
    return load_config().jobs().jobs_folder / job_id


def files_in(folder: Path) -> dict[str, bytes]:
    """Every file inside `folder`, by its path from `folder`, with its bytes."""
    return {path.relative_to(folder).as_posix(): path.read_bytes()
            for path in sorted(folder.rglob("*")) if path.is_file()}


def folders_in(folder: Path) -> tuple[str, ...]:
    """Every folder inside `folder`, by its path from `folder`, sorted."""
    return tuple(sorted(path.relative_to(folder).as_posix()
                        for path in folder.rglob("*") if path.is_dir()))


def walls_job(new_job: Callable[..., str], scene: str = "textured") -> tuple[str, WallsJob]:
    """Make a Job, write `scene` into it with write_walls_job and give its Job_Id and what was
    written."""
    job_id = new_job()
    return job_id, write_walls_job(folder_of(job_id), scene)


def use_clock(monkeypatch: pytest.MonkeyPatch, times: Sequence[datetime]) -> FakeClock:
    """Put a FakeClock giving `times` in place of splatpipe.walls.utc_now."""
    clock = FakeClock(times)
    monkeypatch.setattr("splatpipe.walls.utc_now", clock)
    return clock


def utc_text(moment: datetime) -> str:
    """A time as the Walls_Record writes it: ISO text in UTC, to the millisecond."""
    return moment.astimezone(timezone.utc).isoformat(timespec="milliseconds")


def own_cutoff() -> datetime:
    """The Solve_Cutoff of a run from START with no Cap_Deadline: START + max_minutes −
    finish_minutes (Req 4.2, 4.3)."""
    return START + timedelta(minutes=WALLS_PRESETS[PRESET]["max_minutes"]
                             - WALLS["finish_minutes"])


def failed(report: dict[str, Any]) -> str:
    """Check that `report` is a failed run with an Expected_Failure whose Readable_Error holds no
    traceback, that it left no Walls_Index, and give the Readable_Error (Req 1.7, 18.7, 22.7)."""
    assert report["outcome"] == "failed"
    error = report["error"]
    assert error["expected"] is True
    assert "Traceback" not in error["readable"]
    folder = folder_of(report["job_id"])
    assert not (folder / "work" / "walls" / "depth.json").exists()
    assert not (folder / "work" / "walls" / "scratch").exists()
    return error["readable"]


def read_walls(job: WallsJob) -> Any:
    """The Walls_Prior_Set that read_prior_set reads back from work/walls/, every check run."""
    priors = read_prior_set(job.folder / "work" / "walls", job.model)
    assert priors is not None
    return priors


def same_bits(left: Prior, right: Prior) -> bool:
    """Whether two Priors hold the same three arrays, bit for bit."""
    return all(getattr(left, name).dtype == getattr(right, name).dtype
               and getattr(left, name).shape == getattr(right, name).shape
               and getattr(left, name).tobytes() == getattr(right, name).tobytes()
               for name in PRIOR_ARRAYS)


# The shared done run

@pytest.fixture(scope="session")
def walls_cache() -> dict[str, DoneRun]:
    """Where done_run keeps the shared done run once it has run."""
    return {}


@pytest.fixture
def done_run(new_job: Callable[..., str], walls_pool: list[int],
             monkeypatch: pytest.MonkeyPatch, walls_cache: dict[str, DoneRun]) -> DoneRun:
    """The done run on the textured plane, through splatpipe.walls.run with InlinePool and a
    clock that always gives START, so the Solve_Cutoff is never reached. A Walls_Demo left in
    out/walls/ by an earlier run is there before it. It runs once per session, in the first test
    that asks for it, whose tmp_path keeps the Job_Folder, and every other test reads what it
    left and never changes it."""
    if "done" not in walls_cache:
        job_id, job = walls_job(new_job)
        stale = job.folder / STALE_DEMO
        stale.parent.mkdir(parents=True)
        stale.write_bytes(b"a demo from an earlier run")
        use_clock(monkeypatch, [START])
        before = files_in(job.folder)
        report = walls.run(job_id)
        assert report["outcome"] == "done", report["error"]
        walls_cache["done"] = DoneRun(job=job, report=report, before=before,
                                      after=files_in(job.folder),
                                      folders=folders_in(job.folder), workers=tuple(walls_pool))
    return walls_cache["done"]


# A done run (Req 17.1, 18.1 to 18.5, 18.9, 19.1 to 19.4, 19.7)

def test_a_done_run_writes_walls_priors_that_read_prior_set_reads_back_for_every_frame(
        done_run: DoneRun) -> None:
    job = done_run.job
    priors = read_walls(job)
    # Exactly one Walls_Prior per Prior_Frame, at the Depth_Index's Prior_Size (Req 17.1, 18.1).
    assert sorted(priors.priors) == sorted(job.names) == list(FRAMES)
    width, height = job.prior_size
    for name in FRAMES:
        prior, depth_prior = priors.priors[name], job.priors[name]
        assert prior.inverse_depth.shape == (height, width)
        # The mask is the Depth_Prior's exactly (D3), and every frame passed, so its Walls_Prior
        # comes from its Walls_Depth, not a copy.
        assert prior.mask.tobytes() == depth_prior.mask.tobytes()
        assert not np.array_equal(prior.inverse_depth, depth_prior.inverse_depth)
    # The Depth_Priors are 1.25 × the true depth, and the Walls_Priors are much nearer the truth.
    walls_error = [_depth_error(priors.priors[name], job, name) for name in FRAMES]
    depth_error = [_depth_error(job.priors[name], job, name) for name in FRAMES]
    assert max(walls_error) < 0.05 < 0.2 < min(depth_error)


def _depth_error(prior: Prior, job: WallsJob, name: str) -> float:
    """The median over the mask of |z / z_true − 1| for `prior` of frame `name`, z_true from the
    Depth_Prior's depth and the scale write_walls_job put on it."""
    true_inverse = job.priors[name].inverse_depth.astype(np.float64) * job.scale
    ratio = true_inverse[prior.mask] / prior.inverse_depth[prior.mask].astype(np.float64)
    return float(np.median(np.abs(ratio - 1.0)))


def test_a_done_run_writes_a_walls_index_holding_exactly_the_depth_index(
        done_run: DoneRun) -> None:
    folder = done_run.job.folder / "work"
    assert read_json(folder / "walls" / "depth.json") == read_json(folder / "depth" / "depth.json")
    assert read_walls(done_run.job).index == done_run.job.index


def test_a_done_run_writes_a_walls_record_with_the_designs_fields(done_run: DoneRun) -> None:
    record = read_json(done_run.job.folder / "work" / "walls" / "walls.json")
    assert set(record) == RECORD_FIELDS
    stop = START + timedelta(minutes=WALLS_PRESETS[PRESET]["max_minutes"])
    assert {key: record[key] for key in RECORD_FIELDS - {"frames", "world_planes"}} == {
        "format": 1, "started_utc": utc_text(START), "cap_deadline": None,
        "walls_stop_time": utc_text(stop), "solve_cutoff": utc_text(own_cutoff()),
        "stop_limit": "max_minutes", "workers": done_run.workers[0],
        "demo_frames": list(DEMOS)}
    assert done_run.workers == (min(done_run.workers[0], len(FRAMES)),)
    assert type(record["world_planes"]) is int and record["world_planes"] >= 0
    # One object per Prior_Frame, in Pick_Number order, each with the design's fields.
    assert [(frame["name"], frame["pick_number"]) for frame in record["frames"]] == [
        (name, number) for number, name in enumerate(FRAMES)]
    for frame in record["frames"]:
        assert set(frame) == FRAME_FIELDS
        assert (frame["passed"], frame["outcome"], frame["reason"]) == (True, "passed", None)
        assert (frame["fallback"], frame["fallback_coverage"], frame["fallback_change"]) == (
            None, None, None)
        assert len(frame["change"]) == 4
        assert frame["sources"] == len(frame["source_frames"]) >= WALLS["min_sources"]
        assert frame["name"] not in frame["source_frames"]


def test_a_done_run_writes_its_walls_demos_to_out_walls(done_run: DoneRun) -> None:
    demos = done_run.job.folder / "out" / "walls"
    assert sorted(path.name for path in demos.iterdir()) == list(DEMOS)
    width, height = done_run.job.prior_size
    for name in DEMOS:
        panel = cv2.imdecode(np.frombuffer((demos / name).read_bytes(), np.uint8),
                             cv2.IMREAD_COLOR)
        assert panel is not None and panel.shape == (height, 3 * width, 3)


def test_a_done_runs_report_holds_the_six_counts_the_ten_timings_and_every_setting(
        done_run: DoneRun) -> None:
    report = done_run.report
    assert read_json(done_run.job.folder / "out" / "reports" / "walls.json") == report
    counts = report["counts"]
    assert sorted(counts) == sorted(COUNTS)
    assert all(type(value) is int for value in counts.values())  # Req 19.4
    # Req 19.1's two sums.
    assert counts["images"] == (counts["passed_images"] + counts["fallback_images"]
                                + counts["depth_prior_images"])
    assert counts["images"] == counts["solved_images"] + counts["timed_out_images"]
    assert counts == {"images": 12, "solved_images": 12, "passed_images": 12,
                      "fallback_images": 0, "depth_prior_images": 0, "timed_out_images": 0}
    assert sorted(report["timings"]) == sorted(TIMINGS)
    assert all(type(value) is float and value >= 0 for value in report["timings"].values())
    # Every [walls] setting and the Preset's two, which the Step_Runner puts there (Req 19.7).
    assert report["preset"] == PRESET
    assert report["settings"] == {**WALLS, **WALLS_PRESETS[PRESET]}
    # Every frame passed and none timed out, so neither warning is there (Req 19.5, 19.6).
    assert report["warnings"] == []


# What the step writes and leaves (Req 1.8 to 1.10, 22.7)

def test_a_done_run_writes_only_work_walls_out_walls_and_its_report_and_changes_no_input(
        done_run: DoneRun) -> None:
    before, after = done_run.before, done_run.after
    new = set(after) - set(before)
    assert all(path.startswith(("work/walls/", "out/walls/")) or path == "out/reports/walls.json"
               for path in new)
    assert "out/reports/walls.json" in new
    assert {f"work/walls/{name.removesuffix('.jpg')}.npz" for name in FRAMES} <= new
    assert {"work/walls/depth.json", "work/walls/walls.json"} <= new
    # No scratch folder is left, as a folder or a file inside one.
    assert "work/walls/scratch" not in done_run.folders
    assert not any(path.startswith("work/walls/scratch") for path in after)
    # The Walls_Demo an earlier run left is gone, and every other file is the same bytes, every
    # file of work/frames/, work/cameras/ and work/depth/ among them (Req 1.9, 1.10).
    assert set(before) - set(after) == {STALE_DEMO}
    kept = [path for path in before if path != STALE_DEMO]
    assert any(path.startswith("work/frames/") for path in kept)
    assert any(path.startswith("work/cameras/") for path in kept)
    assert any(path.startswith("work/depth/") for path in kept)
    assert all(after[path] == before[path] for path in kept)


def test_a_run_removes_a_walls_demo_an_earlier_run_left_even_when_it_fails(
        new_job: Callable[..., str], walls_pool: list[int]) -> None:
    job_id, job = walls_job(new_job)
    stale = job.folder / STALE_DEMO
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"a demo from an earlier run")
    (job.folder / "work" / "depth" / "depth.json").unlink()
    failed(walls.run(job_id))
    assert not (job.folder / "out" / "walls").exists()


# The failures before any Frame_Solve (Req 1.7, 2.2 to 2.8, 3.4, 4.1, 19.3, 22.7)

def check_early_failure(report: dict[str, Any]) -> str:
    """`failed`, plus: the step had worked out no count and no timing yet (Req 19.3), and wrote
    nothing to work/walls/."""
    message = failed(report)
    assert (report["counts"], report["timings"]) == ({}, {})
    assert list((folder_of(report["job_id"]) / "work" / "walls").iterdir()) == []
    return message


def test_a_bad_setting_fails_before_the_cap_deadline_is_read(
        new_job: Callable[..., str], walls_pool: list[int], set_walls: Callable[..., None],
        monkeypatch: pytest.MonkeyPatch) -> None:
    job_id, _ = walls_job(new_job)
    set_walls(tile_size=True)
    monkeypatch.setenv(CAP_DEADLINE_ENV, "soon")  # malformed too, but read after the settings
    message = check_early_failure(walls.run(job_id))
    assert message.startswith("The setting tile_size in [walls] is true, but it must be ")


def test_a_malformed_cap_deadline_fails_before_the_inputs_are_checked(
        new_job: Callable[..., str], walls_pool: list[int],
        monkeypatch: pytest.MonkeyPatch) -> None:
    job_id, job = walls_job(new_job)
    (job.folder / "work" / "depth" / "depth.json").unlink()  # missing, but checked later
    monkeypatch.setenv(CAP_DEADLINE_ENV, "soon")
    message = check_early_failure(walls.run(job_id))
    assert message.startswith(f"{CAP_DEADLINE_ENV} is 'soon', but it must be a date and time "
                              "with a UTC offset")


def test_a_job_with_no_depth_index_fails_naming_it_and_the_depth_step(
        new_job: Callable[..., str], walls_pool: list[int]) -> None:
    job_id, job = walls_job(new_job)
    path = job.folder / "work" / "depth" / "depth.json"
    path.unlink()
    message = check_early_failure(walls.run(job_id))
    assert message == no_depth_message(path)


def no_depth_message(path: Path) -> str:
    """Req 2.2's Readable_Error for the missing Depth_Index at `path`."""
    return (f"The walls step needs {path}, which the depth step writes, but it isn't there, so "
            "the depth step has to run on this job first. The job goes on and trains as it "
            "would without the walls step.")


@pytest.mark.parametrize("missing", ["images", "sparse/cameras.bin", "sparse/images.bin",
                                     "sparse/points3D.bin"])
def test_a_job_missing_a_camera_solve_path_fails_naming_it_and_the_camera_solve_step(
        new_job: Callable[..., str], walls_pool: list[int], missing: str) -> None:
    job_id, job = walls_job(new_job)
    path = job.folder / "work" / "cameras" / missing
    if path.is_dir():
        for child in path.iterdir():
            child.unlink()
        path.rmdir()
    else:
        path.unlink()
    message = check_early_failure(walls.run(job_id))
    assert message == (f"The walls step needs {path}, which the camera solve step writes, but "
                       "it isn't there. Run the camera solve step on this job first.")


def test_a_job_with_no_frames_record_fails_naming_it_and_the_frames_step(
        new_job: Callable[..., str], walls_pool: list[int]) -> None:
    job_id, job = walls_job(new_job)
    path = job.folder / "work" / "frames" / "frames.json"
    path.unlink()
    message = check_early_failure(walls.run(job_id))
    assert message == (f"The walls step needs {path}, which the frames step writes, but it "
                       "isn't there. Run the frames step on this job first.")


def test_broken_depth_priors_fail_with_the_prior_readers_message(
        new_job: Callable[..., str], walls_pool: list[int]) -> None:
    job_id, job = walls_job(new_job)
    broken = job.folder / "work" / "depth" / "000003.npz"
    broken.write_bytes(b"not a prior file")
    message = check_early_failure(walls.run(job_id))
    assert message.startswith("The walls step can't use the depth priors: ")
    assert str(broken) in message
    assert message.endswith("Run the depth step on this job again. The job goes on and trains "
                            "as it would without the walls step.")


def test_a_frames_record_that_doesnt_match_the_camera_solve_fails_naming_the_image(
        new_job: Callable[..., str], walls_pool: list[int]) -> None:
    job_id, job = walls_job(new_job)
    path = job.folder / "work" / "frames" / "frames.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    record["picks"][4]["file"] = "000099.jpg"
    path.write_text(json.dumps(record), encoding="utf-8")
    message = check_early_failure(walls.run(job_id))
    assert message.startswith(f"The image '000004.jpg' has a depth prior, but no pick in the "
                              f"frames record {path} has it as its file, so frames.json doesn't "
                              "match the camera solve. Run the job again with --fresh: ")
    assert message.endswith(f"{job_id} --fresh")


# No Trusted_Frame (Req 18.7, 18.8, 19.3, A15)

def test_a_run_in_which_no_frame_passes_fails_and_leaves_no_walls_index_or_prior_file(
        new_job: Callable[..., str], walls_pool: list[int],
        monkeypatch: pytest.MonkeyPatch) -> None:
    job_id, job = walls_job(new_job, "plain")
    use_clock(monkeypatch, [START])
    report = walls.run(job_id)
    message = failed(report)
    # The plain plane has the textured plane's cameras, so every frame finds its Source_Frames
    # and fails on its missing texture, none for too few Source_Frames.
    assert message == (
        "None of the 12 frames passed the walls step's self-check, so no walls priors were made: "
        "0 ran out of time before their solve ended, and 12 failed their self-check, 0 of them "
        "because they had too few other frames to compare with. The job goes on and trains on "
        "the depth priors.")
    # Nothing is written to work/walls/ or out/walls/ (Req 18.8).
    assert list((job.folder / "work" / "walls").iterdir()) == []
    assert not (job.folder / "out" / "walls").exists()
    # The report holds the counts worked out before the failure, and the timings of the blocks
    # that ended: read and solve (Req 19.3).
    assert report["counts"] == {"images": 12, "solved_images": 12, "timed_out_images": 0,
                                "passed_images": 0}
    assert sorted(report["timings"]) == ["read", "solve"]


# The Solve_Cutoff (Req 4.1 to 4.3, 4.6 to 4.8, 4.11, 19.5, 19.6)

def test_a_run_cut_off_after_two_solves_copies_the_depth_priors_of_its_timed_out_frames(
        new_job: Callable[..., str], walls_pool: list[int],
        monkeypatch: pytest.MonkeyPatch) -> None:
    job_id, job = walls_job(new_job)
    # A Cap_Deadline close enough that the train step's reserve sets the Walls_Stop_Time: its
    # reserve before the deadline is 10 minutes after START, sooner than max_minutes.
    minutes = WALLS_PRESETS[PRESET]
    cap_deadline = START + timedelta(minutes=minutes["train_reserve_minutes"] + 10)
    assert 10 < minutes["max_minutes"]
    monkeypatch.setenv(CAP_DEADLINE_ENV, cap_deadline.isoformat())
    stop = cap_deadline - timedelta(minutes=minutes["train_reserve_minutes"])
    cutoff = stop - timedelta(minutes=WALLS["finish_minutes"])
    # The step's start, then a time before the cutoff for the first two solves' checks, before
    # and after each, then the cutoff for the third's.
    use_clock(monkeypatch, [START] + [cutoff - timedelta(seconds=1)] * 4 + [cutoff])
    report = walls.run(job_id)
    assert report["outcome"] == "done", report["error"]
    record = read_json(job.folder / "work" / "walls" / "walls.json")
    assert (record["cap_deadline"], record["walls_stop_time"], record["solve_cutoff"],
            record["stop_limit"]) == (utc_text(cap_deadline), utc_text(stop), utc_text(cutoff),
                                      "train_reserve_minutes")
    # The Solve_Order starts with the first and the last frame, so those two were solved.
    solved = {FRAMES[0], FRAMES[-1]}
    timed_out = [name for name in FRAMES if name not in solved]
    frames = {frame["name"]: frame for frame in record["frames"]}
    for name in timed_out:
        assert (frames[name]["outcome"], frames[name]["reason"]) == ("depth_prior", "timed_out")
        assert frames[name]["source_frames"] is None and frames[name]["passed"] is None
        assert frames[name]["fallback"] is None  # no fallback stage (Req 4.7)
    # Each Timed_Out_Frame's Walls_Prior is its Depth_Prior, bit for bit (Req 4.7, 17.5, 17.6).
    priors = read_walls(job)
    assert all(same_bits(priors.priors[name], job.priors[name]) for name in timed_out)
    # The solved frames passed and used Timed_Out_Frames as Source_Frames (Req 4.8).
    for name in solved:
        assert frames[name]["outcome"] == "passed"
        assert set(frames[name]["source_frames"]) & set(timed_out)
        assert not same_bits(priors.priors[name], job.priors[name])
    assert report["counts"] == {"images": 12, "solved_images": 2, "timed_out_images": 10,
                                "passed_images": 2, "fallback_images": 0,
                                "depth_prior_images": 10}
    # Req 19.5's warning, then Req 19.6's.
    assert report["warnings"] == [
        "The walls step ran out of time for 10 of 12 frames: it had to stop solving frames in "
        f"time to be done by {stop.strftime('%Y-%m-%d %H:%M:%S')} UTC, which the train step's "
        "train_reserve_minutes before the cap deadline set. Those frames keep their depth prior.",
        "2 of 12 frames passed the walls step's self-check and were corrected from their own "
        "solve. Of the others, 0 got a walls prior from a fallback depth and 10 kept their "
        "depth prior unchanged."]


# A worker that stops (Req 5.7)

def test_a_worker_that_stops_without_a_result_fails_the_step_naming_its_frame(
        new_job: Callable[..., str], dead_worker_pool: Callable[[str], None],
        monkeypatch: pytest.MonkeyPatch) -> None:
    job_id, job = walls_job(new_job)
    dead_worker_pool(FRAMES[0])  # the first Frame_Solve in the Solve_Order
    use_clock(monkeypatch, [START])
    report = walls.run(job_id)
    message = failed(report)
    assert message == ("A worker process stopped while the walls step solved the frame "
                       "'000000.jpg'. The machine may have run out of memory. The job goes on "
                       "and trains on the depth priors.")
    assert list((job.folder / "work" / "walls").iterdir()) == []
    assert report["counts"] == {"images": 12}
    assert sorted(report["timings"]) == ["read"]


# The CLI (Req 1.1, 1.2)

def test_the_cli_prints_walls_done_and_gives_0_when_the_step_is_done(
        new_job: Callable[..., str], walls_pool: list[int], monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str]) -> None:
    job_id, job = walls_job(new_job)
    # One Frame_Solve before the Solve_Cutoff, so the done run stays short.
    cutoff = own_cutoff()
    use_clock(monkeypatch, [START, cutoff - timedelta(seconds=1),
                            cutoff - timedelta(seconds=1), cutoff])
    assert walls_main(["--job", job_id]) == 0
    out, err = capsys.readouterr()
    assert (out, err) == ("walls: done\n", "")
    assert (job.folder / "work" / "walls" / "depth.json").is_file()


def test_the_cli_prints_the_readable_error_on_stderr_and_gives_1_when_the_step_fails(
        new_job: Callable[..., str], walls_pool: list[int],
        capsys: pytest.CaptureFixture[str]) -> None:
    job_id, job = walls_job(new_job)
    path = job.folder / "work" / "depth" / "depth.json"
    path.unlink()
    assert walls_main(["--job", job_id]) == 1
    out, err = capsys.readouterr()
    assert (out, err) == ("", no_depth_message(path) + "\n")
    assert "Traceback" not in err


# The Job goes on when the walls step fails (Req 1.5, 1.6)

def test_a_job_whose_walls_step_fails_goes_on_trains_and_is_done_with_warnings(
        new_job: Callable[..., str], walls_pool: list[int],
        fake_gpu: Callable[..., Any]) -> None:
    job_id, job = walls_job(new_job)
    depth = job.folder / "work" / "depth"
    for path in depth.iterdir():
        path.unlink()
    depth.rmdir()
    fake_gpu()
    summary = run_job(job_id, steps=(StepEntry("walls", "splatpipe.walls", False),
                                     StepEntry("train", "splatpipe.train", True)))
    message = no_depth_message(depth / "depth.json")
    assert summary["status"] == "done_with_warnings"
    assert summary["steps"][0] == {"step": "walls", "required": False, "outcome": "failed",
                                   "reused": False, "readable_error": message}
    assert summary["steps"][1]["outcome"] == "done"
    assert (summary["failed_step"], summary["readable_error"]) == (None, None)
    log = (job.folder / "out" / "job.log").read_text(encoding="utf-8")
    assert f"WARNING walls: failed, optional so the job goes on: {message}" in log
    assert not (job.folder / "work" / "walls" / "depth.json").exists()
    assert (job.folder / "out" / "train.ply").is_file()
