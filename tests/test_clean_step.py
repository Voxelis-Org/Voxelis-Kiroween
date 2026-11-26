"""End-to-end tests for the clean step: splatpipe/clean/__init__.py and __main__.py.

Each test that runs the step makes its Job with the new_job fixture, and the Job's trained
splat with put_source, so the Job sits in the test config's Jobs_Folder, inside tmp_path, and
nothing reaches the real data/ (Req 10.3). The splats are a few dozen Gaussians from gaussians
in tests/splats.py. The test config's [clean] has min_kept_splats = 1, so splats that small
pass, and set_clean changes one setting for one test.

The step runs through splatpipe.clean.run and splatpipe.clean.__main__.main. The stand_in
fixture patches splatpipe.clean.export_sog, where work looks it up, to call the real export_sog
with tests/fake_splat_transform.py in place of node and cli.mjs, as tests/test_clean_export.py
does, so every run but one needs no Node (Req 10.6).

Done and failed runs: a done run writes both Step_Outputs and a report of exactly the counts and
timings of Req 7.1, with the eighteen settings (Req 5.14, 7.1, 7.3, 7.4, 7.6, 7.7). The warnings
of Req 3.6 and 5.5 land in the report (Req 7.5). A failure at each stage, from reading the splat
to the moves, leaves exactly the counts and timings worked out before it, with expected true and
the step's own message, which holds no traceback (Req 3.8, 7.2, 8.8, 10.4). A Job with no
trained splat fails naming the training step (Req 1.1). A count over max_source_splats fails
before the rows are read (Req 3.1), the limits write neither Step_Output (Req 3.2, 3.5, 3.7),
and a missing setting is the Step_Runner's own refusal (Req 6.6). The CLI prints clean: done, or
the Readable_Error on stderr, and python -m splatpipe.clean runs from the repo root (Req 8.2).
One done run uses the real splat-transform, and is skipped when Node or bin/cli.mjs is missing
(Req 5.1, 10.6).

The two Step_Outputs: a spy on Path.unlink shows the viewer copy removed before the clean
splat, and a removal that fails, for each file in turn, names the file and leaves the files not
yet removed (Req 4.8, 4.9, 4.13). A lone clean splat, as a hard stop leaves it, is gone before
the step writes anything (Req 4.12), and a refused setting leaves an earlier pair as it was
(Req 4.10, 6.9). A failed export, a second move that fails and Ctrl+C in the export each leave
neither output, an earlier pair included, and Ctrl+C still stops the run (Req 4.10, 4.11,
10.10). While the stand-in runs, out/ holds neither new output (Req 4.5, 5.9), and a second done
run with another Preset replaces both (Req 8.6). The trained splat's SHA-256 is the same after a
done run and after a run the Removal_Cap stops (Req 1.11, 10.8), and a done run changes nothing
in the Job_Folder but work/clean/, the two outputs and the report (Req 8.7, 10.11).

The memory test is the one run on a large splat: 20,000 Gaussians, every one a Faint_Gaussian
alone in its Cube, which keeps it, so every stage runs at full size. tracemalloc traces the run,
and a stand-in export reads it as it's called: by then no array the clean phase made is still
held, and the traced peak was at most 600 bytes a Gaussian above the start (Req 8.9).

Each check for a traceback reads only the words the step wrote itself, with the paths and values
the test chose taken out first, as refusal does in tests/test_clean_limits.py (Req 10.4).
"""

import errno
import hashlib
import json
import os
import subprocess
import sys
import tracemalloc
import zipfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NoReturn

import numpy as np
import pytest

from splatpipe import clean
from splatpipe.clean import ply_write
from splatpipe.clean.__main__ import main
from splatpipe.clean.export import export_sog
from splatpipe.clean.ply_read import PlyHeader
from splatpipe.clean.settings import CleanSettings
from splatpipe.config import REPO_ROOT, load_config
from splatpipe.errors import PipelineError
from splatpipe.jobs import check_job_id, open_job
from splatpipe.jsonio import read_json

import fake_splat_transform as fake
from conftest import CLEAN_PRESETS, CLEAN_SHARED
from splats import FAR_BEHIND, PRESET, gaussians, put_cameras, put_source
from test_clean_export import find_stand_in, needs_splat_transform, use_stand_in, without
from test_clean_limits import numbers_in

# How many Gaussians most tests' splats hold: more than the test config's 8 neighbours, so the
# isolated filter searches, and few, so a run is quick.
N = 20
# The counts and the timings of a done Clean_Report, each in the order the step works it out
# (Req 7.1).
COUNTS = ("source_splats", "dropped_broken", "dropped_unseen", "dropped_faint",
          "dropped_isolated", "kept_splats", "clean_ply_bytes", "clean_sog_bytes")
TIMINGS = ("read_cameras", "read_ply", "clean", "write_ply", "export_sog")


def folder_of(job_id: str) -> Path:
    """The Job_Folder of `job_id` as the Step_Runner opens it, resolved, in the test config's
    Jobs_Folder."""
    return open_job(job_id, load_config()).folder


def saved_report(job_id: str) -> Any:
    """The clean step's Step_Report file, out/reports/clean.json, read back (Req 7.6)."""
    return read_json(folder_of(job_id) / "out" / "reports" / "clean.json")


def one_broken() -> np.ndarray:
    """N Gaussians, the first with a NaN centre, which the broken filter drops. The others pass
    every filter."""
    data = gaussians(N)
    data["x"][0] = np.nan
    return data


@dataclass
class StandIn:
    """What the stand_in fixture puts in place of splatpipe.clean.export_sog: the real
    export_sog, with the stand-in script in place of node and cli.mjs. Each call's band count
    goes in `bands`."""

    bands: list[int] = field(default_factory=list)

    def __call__(self, source: Path, target: Path, bands: int, s: CleanSettings,
                 log_file: Path) -> int:
        self.bands.append(bands)
        return export_sog(source, target, bands, s, log_file, find=find_stand_in)


@pytest.fixture
def stand_in(monkeypatch: pytest.MonkeyPatch) -> StandIn:
    """Patch splatpipe.clean.export_sog, where work looks it up, with a StandIn, with the
    stand-in set to ok. A test sets another mode with use_stand_in."""
    use_stand_in(monkeypatch, "ok")
    export = StandIn()
    monkeypatch.setattr(clean, "export_sog", export)
    return export


# Done runs

def test_a_done_run_writes_both_outputs_and_reports_the_counts_timings_and_settings(
        new_job: Callable[..., str], stand_in: StandIn) -> None:
    job_id = new_job()
    put_source(job_id, one_broken())
    report = clean.run(job_id)
    assert (report["outcome"], report["warnings"], report["error"]) == ("done", [], None)
    saved = saved_report(job_id)
    assert saved == report  # at out/reports/clean.json (Req 7.6)
    counts, timings = saved["counts"], saved["timings"]
    assert list(counts) == list(COUNTS) and list(timings) == list(TIMINGS)  # Req 7.1
    assert "removed_percent" not in counts  # Req 7.4
    assert all(type(value) is int for value in counts.values())  # whole numbers (Req 7.3)
    assert all(type(seconds) is float and seconds >= 0 for seconds in timings.values())
    out = folder_of(job_id) / "out"
    assert counts == {"source_splats": N, "dropped_broken": 1, "dropped_unseen": 0,
                      "dropped_faint": 0, "dropped_isolated": 0, "kept_splats": N - 1,
                      "clean_ply_bytes": (out / "clean.ply").stat().st_size,
                      "clean_sog_bytes": len(fake.OK_BYTES)}  # both outputs' sizes (Req 5.14)
    assert (out / "clean.sog").read_bytes() == fake.OK_BYTES
    # The nineteen settings, as the test config gives them (Req 7.7).
    assert saved["settings"] == {**CLEAN_SHARED, **CLEAN_PRESETS[PRESET]}
    assert len(saved["settings"]) == 19


def test_the_kept_fraction_and_band_count_warnings_land_in_the_report(
        new_job: Callable[..., str], stand_in: StandIn) -> None:
    job_id = new_job()
    data = gaussians(N, sh_count=9)  # 9 f_rest_* properties: 1 band of spherical harmonics
    data["x"][:12] = np.nan  # 12 broken, which don't count toward the cap, so 8 of 20 are kept
    put_source(job_id, data)
    report = clean.run(job_id, preset="quality")  # the test config's quality Preset keeps 3 bands
    assert report["outcome"] == "done"  # neither warning stops the step
    kept_little, fewer_bands = report["warnings"]  # Req 7.5
    # Req 3.6's warning, with both counts, then Req 5.5's, with both band counts.
    assert "warn_kept_fraction" in kept_little and {"8", "20"} <= numbers_in(kept_little)
    assert {"3", "1"} <= numbers_in(fewer_bands)
    assert stand_in.bands == [1]  # the trained splat's own band count, in place of the Preset's


def test_a_gaussian_behind_every_training_frame_is_dropped_as_unseen(
        new_job: Callable[..., str], stand_in: StandIn) -> None:
    job_id = new_job()
    data = gaussians(N)
    data["z"][0] = -2 * FAR_BEHIND  # behind both of put_cameras's frames, so no frame sees it
    put_source(job_id, data)
    report = clean.run(job_id)
    assert report["outcome"] == "done", report["error"]
    counts = report["counts"]
    assert (counts["dropped_unseen"], counts["kept_splats"]) == (1, N - 1)


# Failed runs

# A stored opacity whose Real_Opacity, about 0.00005, is far below the test config's min_opacity
# of 0.005, so the Gaussian is a Faint_Gaussian.
FAINT = -10.0


def faint_splat() -> np.ndarray:
    """N Gaussians, the first 15 of them faint. The faint filter drops those 15, 75% of the
    splat, over the test config's cap of 70% (Req 3.2).

    The lattice's spacing of 1 is far wider than a Cube_Edge, about 2 / 128, so each faint
    Gaussian is alone in its Cube, fewer than faint_min_per_cube's 8.
    """
    data = gaussians(N)
    data["opacity"][:15] = FAINT
    return data


def disk_full(file: object, mode: str) -> NoReturn:
    """A stand-in for open in splatpipe/clean/ply_write.py, which raises the OSError a full disk
    gives."""
    raise OSError(errno.ENOSPC, "No space left on device", str(file))


def refuse_replace(monkeypatch: pytest.MonkeyPatch, target: Path) -> None:
    """Make os.replace refuse a move to `target` with the PermissionError Windows gives when
    another program has that file open, and make every other move as before."""
    real_replace = os.replace

    def replace(source: Any, destination: Any, **options: Any) -> None:
        if Path(destination) == target:
            raise PermissionError(errno.EACCES, "Permission denied", str(destination))
        real_replace(source, destination, **options)

    monkeypatch.setattr(os, "replace", replace)


# A way to make a run of the step on the Job fail at one stage. It gets the Job_Id, set_clean
# and monkeypatch.
Breaker = Callable[[str, Callable[..., None], pytest.MonkeyPatch], None]


def no_camera_solve(job_id: str, set_clean: Callable[..., None],
                    monkeypatch: pytest.MonkeyPatch) -> None:
    """Write the trained splat but no work/cameras/sparse/, which the Unseen_Filter needs."""
    put_source(job_id, gaussians(N), cameras=False)


def no_trained_splat(job_id: str, set_clean: Callable[..., None],
                     monkeypatch: pytest.MonkeyPatch) -> None:
    """Leave out/train.ply out of the Job, which has its cameras (Req 1.1, 1.6)."""
    put_cameras(job_id)


def too_many_gaussians(job_id: str, set_clean: Callable[..., None],
                       monkeypatch: pytest.MonkeyPatch) -> None:
    """Write N Gaussians, one more than max_source_splats (Req 3.1)."""
    put_source(job_id, gaussians(N))
    set_clean(max_source_splats=N - 1)


def over_the_cap(job_id: str, set_clean: Callable[..., None],
                 monkeypatch: pytest.MonkeyPatch) -> None:
    """Write the faint splat, whose faint Gaussians are over the Removal_Cap (Req 3.2)."""
    put_source(job_id, faint_splat())


def a_full_disk(job_id: str, set_clean: Callable[..., None],
                monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the clean splat's file fail to open, as on a full disk (Req 4.6)."""
    put_source(job_id, gaussians(N))
    monkeypatch.setattr(ply_write, "open", disk_full, raising=False)  # ahead of the builtin


def a_failing_splat_transform(job_id: str, set_clean: Callable[..., None],
                              monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the stand-in fail as splat-transform can (Req 5.11)."""
    put_source(job_id, gaussians(N))
    use_stand_in(monkeypatch, "fail")


def a_refused_move(job_id: str, set_clean: Callable[..., None],
                   monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the clean splat's move into out/ fail, as when another program has out/clean.ply
    open."""
    put_source(job_id, gaussians(N))
    refuse_replace(monkeypatch, folder_of(job_id) / "out" / "clean.ply")


# Where a run fails, from reading the splat to the moves: how to make it fail there, how many of
# COUNTS and of TIMINGS the step has worked out by then, and words of the step's own message.
STAGES = [
    pytest.param(no_camera_solve, 0, 0,
                 (str(Path("work", "cameras", "sparse", "cameras.bin")), "camera solve step"),
                 id="reading, no camera solve"),
    pytest.param(no_trained_splat, 0, 1,
                 (str(Path("out", "train.ply")), "doesn't exist", "training step"),
                 id="reading, no trained splat"),
    pytest.param(too_many_gaussians, 1, 1, ("max_source_splats",),
                 id="reading, too many gaussians"),
    pytest.param(over_the_cap, 6, 3, ("max_removed_fraction",), id="the limits, over the cap"),
    pytest.param(a_full_disk, 6, 3, ("couldn't be written", "No space left on device"),
                 id="writing, a full disk"),
    pytest.param(a_failing_splat_transform, 7, 4, (f"exit code {fake.FAIL_EXIT_CODE}",),
                 id="the export, splat-transform fails"),
    pytest.param(a_refused_move, 8, 5, ("couldn't move", "Permission denied"),
                 id="the moves, a refused move"),
]


@pytest.mark.parametrize(("breaks", "counts", "timings", "words"), STAGES)
def test_a_failed_report_holds_only_the_counts_and_timings_worked_out_before_the_failure(
        new_job: Callable[..., str], set_clean: Callable[..., None], stand_in: StandIn,
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path, breaks: Breaker, counts: int,
        timings: int, words: tuple[str, ...]) -> None:
    job_id = new_job()
    breaks(job_id, set_clean, monkeypatch)
    report = clean.run(job_id)
    assert saved_report(job_id) == report
    assert report["outcome"] == "failed"
    assert list(report["counts"]) == list(COUNTS[:counts])  # Req 7.2
    assert list(report["timings"]) == list(TIMINGS[:timings])
    error = report["error"]
    assert error["expected"] is True  # Req 3.8, 8.8
    # The step's own ExpectedFailure, whose message is the Readable_Error as it is.
    assert error["technical"]["type"] == "splatpipe.errors.ExpectedFailure"
    assert error["readable"] == error["technical"]["message"]
    for word in words:
        assert word in error["readable"]
    assert "Traceback" not in without(error["readable"], tmp_path)  # Req 10.4


def test_a_count_over_max_source_splats_fails_before_the_rows_are_read(
        new_job: Callable[..., str], set_clean: Callable[..., None], stand_in: StandIn,
        monkeypatch: pytest.MonkeyPatch) -> None:
    job_id = new_job()
    put_source(job_id, gaussians(N))
    reads: list[int] = []  # the vertex count of each header whose rows were read
    real_read_splat = clean.read_splat

    def spy(header: PlyHeader) -> np.ndarray:
        reads.append(header.count)
        return real_read_splat(header)

    monkeypatch.setattr(clean, "read_splat", spy)  # where clean_and_write looks it up
    assert clean.run(job_id)["outcome"] == "done"
    assert reads == [N]  # so the spy sees a run that reads the rows
    set_clean(max_source_splats=N - 1)
    report = clean.run(job_id)
    assert report["outcome"] == "failed" and "max_source_splats" in report["error"]["readable"]
    assert reads == [N]  # this run read no rows (Req 3.1)


# Splats the limits stop, each with the settings that stop it and the setting its message names:
# over the Removal_Cap, and with fewer than min_kept_splats left once the broken one is dropped.
STOPPED = [
    pytest.param(faint_splat, {}, "max_removed_fraction", id="over the removal cap"),
    pytest.param(one_broken, {"min_kept_splats": N}, "min_kept_splats",
                 id="fewer than min_kept_splats left"),
]


@pytest.mark.parametrize(("splat", "values", "setting"), STOPPED)
def test_a_run_the_limits_stop_writes_neither_output(
        new_job: Callable[..., str], set_clean: Callable[..., None], stand_in: StandIn,
        splat: Callable[[], np.ndarray], values: dict[str, object], setting: str) -> None:
    job_id = new_job()
    put_source(job_id, splat())
    set_clean(**values)
    report = clean.run(job_id)
    assert report["outcome"] == "failed" and setting in report["error"]["readable"]
    folder = folder_of(job_id)
    assert not (folder / "out" / "clean.ply").exists()
    assert not (folder / "out" / "clean.sog").exists()
    # Nothing was written at all, because the limits come before the write (Req 3.2, 3.5, 3.7).
    assert list((folder / "work" / "clean").iterdir()) == []


def test_a_missing_setting_fails_with_the_step_runners_own_message(
        new_job: Callable[..., str], set_clean: Callable[..., None], stand_in: StandIn) -> None:
    job_id = new_job()
    put_source(job_id, gaussians(N))
    set_clean(neighbours=None)
    report = clean.run(job_id)
    error = report["error"]
    assert (report["outcome"], error["expected"]) == ("failed", True)
    assert error["technical"]["type"] == "splatpipe.errors.PipelineError"  # not the step's own
    # The settings the Step_Runner gives work refuse the key with the same message (Req 6.6).
    settings = load_config().step_settings("clean", PRESET)
    with pytest.raises(PipelineError) as caught:
        settings["neighbours"]
    assert error["readable"] == str(caught.value)


# The two Step_Outputs (Req 4.5, 4.8 to 4.13, 5.9, 8.6)

# The Step_Outputs' names in the test config, the Clean_Splat first.
OUTPUTS = ("clean.ply", "clean.sog")


def outputs_in(job_id: str) -> dict[str, bytes]:
    """The Step_Outputs in the Job's out/, by name, each with its bytes."""
    out = folder_of(job_id) / "out"
    return {name: (out / name).read_bytes() for name in OUTPUTS if (out / name).exists()}


def done_run(job_id: str, preset: str | None = None) -> dict[str, bytes]:
    """Run the step on the Job to done and give the pair it left in out/, as outputs_in gives
    it. A test runs this first to put an earlier run's pair in out/."""
    report = clean.run(job_id, preset=preset)
    assert report["outcome"] == "done", report["error"]
    pair = outputs_in(job_id)
    assert list(pair) == list(OUTPUTS)
    return pair


def spy_on_unlink(monkeypatch: pytest.MonkeyPatch, refused: Path | None = None) -> list[Path]:
    """Patch Path.unlink to record the path of each call, then do the real removal. Gives the
    list of paths, in the order of the calls.

    A call on `refused` removes nothing and raises the PermissionError Windows gives when
    another program has that file open.
    """
    real_unlink = Path.unlink
    calls: list[Path] = []

    def unlink(self: Path, missing_ok: bool = False) -> None:
        calls.append(self)
        if self == refused:
            raise PermissionError(errno.EACCES, "Permission denied", str(self))
        real_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", unlink)
    return calls


def test_the_viewer_copy_is_removed_before_the_clean_splat(
        new_job: Callable[..., str], stand_in: StandIn, monkeypatch: pytest.MonkeyPatch) -> None:
    job_id = new_job()
    put_source(job_id, gaussians(N))
    done_run(job_id)  # an earlier pair, for the next run to remove
    out = folder_of(job_id) / "out"
    removed = spy_on_unlink(monkeypatch)
    done_run(job_id)
    # The viewer copy first, so out/ never holds it without its clean splat (Req 4.8, 4.13).
    assert [path for path in removed if path.parent == out] == [out / "clean.sog",
                                                                out / "clean.ply"]


# The removal that fails, and the Step_Outputs it leaves in out/: both when it's the viewer
# copy's, because the clean splat's removal comes after it, and the clean splat alone when it's
# the clean splat's.
REFUSED = [
    pytest.param("clean.sog", OUTPUTS, id="the viewer copy"),
    pytest.param("clean.ply", ("clean.ply",), id="the clean splat"),
]


@pytest.mark.parametrize(("refused", "left"), REFUSED)
def test_a_removal_that_fails_names_the_file_and_leaves_the_files_not_yet_removed(
        new_job: Callable[..., str], stand_in: StandIn, monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path, refused: str, left: tuple[str, ...]) -> None:
    job_id = new_job()
    put_source(job_id, gaussians(N))
    earlier = done_run(job_id)
    out = folder_of(job_id) / "out"
    spy_on_unlink(monkeypatch, refused=out / refused)
    report = clean.run(job_id)
    error = report["error"]
    assert (report["outcome"], error["expected"]) == ("failed", True)
    # The file and the reason (Req 4.9), in the step's own words (Req 10.4).
    assert str(out / refused) in error["readable"] and "Permission denied" in error["readable"]
    assert "Traceback" not in without(error["readable"], tmp_path)
    # The files not yet removed, as they were, and never the viewer copy alone (Req 4.10, 4.13).
    assert outputs_in(job_id) == {name: earlier[name] for name in left}


def test_a_lone_clean_splat_that_a_hard_stop_left_is_removed_before_anything_is_written(
        new_job: Callable[..., str], stand_in: StandIn, monkeypatch: pytest.MonkeyPatch) -> None:
    job_id = new_job()
    put_source(job_id, gaussians(N))
    done_run(job_id)
    out = folder_of(job_id) / "out"
    # What a hard stop between the two moves leaves: that run's whole clean splat alone, and no
    # report, because the Step_Runner removed the earlier one before work ran (Req 4.12).
    (out / "clean.sog").unlink()
    (out / "reports" / "clean.json").unlink()
    seen: list[bool] = []  # whether out/clean.ply was there each time the step began to write
    real_write_splat = clean.write_splat

    def spy(path: Path, header: PlyHeader, data: np.ndarray, rows: np.ndarray) -> int:
        seen.append((out / "clean.ply").exists())
        return real_write_splat(path, header, data, rows)

    monkeypatch.setattr(clean, "write_splat", spy)  # where clean_and_write looks it up
    done_run(job_id)
    assert seen == [False]  # the step's first write came after the lone file was gone


def test_a_refused_setting_leaves_the_earlier_pair_as_it_was(
        new_job: Callable[..., str], set_clean: Callable[..., None], stand_in: StandIn) -> None:
    job_id = new_job()
    put_source(job_id, gaussians(N))
    earlier = done_run(job_id)
    # A path that leads out of out/, which check_settings refuses before anything is touched
    # (Req 6.9).
    set_clean(clean_sog="../clean.sog")
    report = clean.run(job_id)
    error = report["error"]
    assert (report["outcome"], error["expected"]) == ("failed", True)
    assert "clean_sog" in error["readable"]
    assert outputs_in(job_id) == earlier  # neither was removed (Req 4.10)


def test_a_failed_export_leaves_neither_output_and_a_following_done_run_writes_both_again(
        new_job: Callable[..., str], stand_in: StandIn, monkeypatch: pytest.MonkeyPatch) -> None:
    job_id = new_job()
    put_source(job_id, gaussians(N))
    earlier = done_run(job_id)
    use_stand_in(monkeypatch, "fail")
    assert clean.run(job_id)["outcome"] == "failed"
    assert outputs_in(job_id) == {}  # the earlier pair included (Req 4.8, 4.10, 10.10)
    use_stand_in(monkeypatch, "ok")
    assert done_run(job_id) == earlier  # the same trained splat and settings give the same pair


def test_a_second_move_that_fails_leaves_neither_output(
        new_job: Callable[..., str], stand_in: StandIn, monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path) -> None:
    job_id = new_job()
    put_source(job_id, gaussians(N))
    done_run(job_id)  # an earlier pair
    folder = folder_of(job_id)
    sog = folder / "out" / "clean.sog"
    refuse_replace(monkeypatch, sog)  # the viewer copy's move, the second
    report = clean.run(job_id)
    error = report["error"]
    assert (report["outcome"], error["expected"]) == ("failed", True)
    assert "couldn't move" in error["readable"] and str(sog) in error["readable"]
    assert "Traceback" not in without(error["readable"], tmp_path)
    assert outputs_in(job_id) == {}  # the clean splat, already moved, was removed (Req 4.10)
    # The clean splat moved first, so it has left work/clean/, and the viewer copy never moved
    # (Req 4.11).
    work = folder / "work" / "clean"
    assert [(work / name).exists() for name in OUTPUTS] == [False, True]


def test_ctrl_c_in_the_export_leaves_neither_output_and_still_stops_the_run(
        new_job: Callable[..., str], stand_in: StandIn, monkeypatch: pytest.MonkeyPatch) -> None:
    job_id = new_job()
    put_source(job_id, gaussians(N))
    done_run(job_id)  # an earlier pair

    def interrupted(source: Path, target: Path, bands: int, s: CleanSettings,
                    log_file: Path) -> NoReturn:
        raise KeyboardInterrupt  # as Ctrl+C does while splat-transform runs

    monkeypatch.setattr(clean, "export_sog", interrupted)
    # The Step_Runner catches no KeyboardInterrupt, so it comes out of run (Req 4.10).
    with pytest.raises(KeyboardInterrupt):
        clean.run(job_id)
    assert outputs_in(job_id) == {}
    # It wrote no report either, so none says done: it removed the earlier one before work ran.
    assert not (folder_of(job_id) / "out" / "reports" / "clean.json").exists()


def test_out_holds_neither_new_output_while_the_stand_in_runs(
        new_job: Callable[..., str], stand_in: StandIn, monkeypatch: pytest.MonkeyPatch) -> None:
    job_id = new_job()
    put_source(job_id, gaussians(N))
    # Each export's view as it starts: the outputs in out/, where it reads and writes, and the
    # bytes of the clean splat it reads.
    seen: list[tuple[dict[str, bytes], Path, Path, bytes]] = []

    def look_then_export(source: Path, target: Path, bands: int, s: CleanSettings,
                         log_file: Path) -> int:
        seen.append((outputs_in(job_id), source, target, source.read_bytes()))
        return stand_in(source, target, bands, s, log_file)

    monkeypatch.setattr(clean, "export_sog", look_then_export)
    pair = done_run(job_id)
    work = folder_of(job_id) / "work" / "clean"
    [(in_out, source, target, clean_splat)] = seen
    assert in_out == {}  # Req 4.5, 5.9
    assert (source, target) == (work / "clean.ply", work / "clean.sog")
    assert pair["clean.ply"] == clean_splat  # the file moved into out/ was whole before it


def bands_named(bands: int) -> bytes:
    """What export_naming_bands writes as the viewer copy for `bands` bands."""
    return fake.OK_BYTES + f"{bands} bands\n".encode()


def test_a_second_done_run_with_another_preset_replaces_both_outputs(
        new_job: Callable[..., str], stand_in: StandIn, monkeypatch: pytest.MonkeyPatch) -> None:
    """The stand-in's viewer copy is rewritten to name its band count, which the Preset sets,
    so each Preset's viewer copy has its own bytes. Every Preset cleans the splat alike
    (Req 6.3), so the second run's trained splat has one broken Gaussian, which gives a clean
    splat of its own too."""

    def export_naming_bands(source: Path, target: Path, bands: int, s: CleanSettings,
                            log_file: Path) -> int:
        stand_in(source, target, bands, s, log_file)
        target.write_bytes(bands_named(bands))
        return target.stat().st_size

    monkeypatch.setattr(clean, "export_sog", export_naming_bands)
    job_id = new_job()
    put_source(job_id, gaussians(N))
    first = done_run(job_id, "fast")  # 0 bands
    put_source(job_id, one_broken())
    second = done_run(job_id, "quality")  # 3 bands
    # Both are the second run's (Req 8.6).
    assert (first["clean.sog"], second["clean.sog"]) == (bands_named(0), bands_named(3))
    counts = saved_report(job_id)["counts"]
    assert counts["kept_splats"] == N - 1
    assert len(second["clean.ply"]) == counts["clean_ply_bytes"]
    assert second["clean.ply"] != first["clean.ply"]


# The trained splat and the Job_Folder

def plain() -> np.ndarray:
    """N Gaussians that pass every filter."""
    return gaussians(N)


# A done run, and a run that the Removal_Cap stops, each with words of its message.
RUNS = [
    pytest.param(plain, "done", (), id="a done run"),
    pytest.param(faint_splat, "failed", ("max_removed_fraction",),
                 id="a run the removal cap stops"),
]


@pytest.mark.parametrize(("splat", "outcome", "words"), RUNS)
def test_the_trained_splat_has_the_same_sha_256_after_the_run(
        new_job: Callable[..., str], stand_in: StandIn, splat: Callable[[], np.ndarray],
        outcome: str, words: tuple[str, ...]) -> None:
    job_id = new_job()
    source = put_source(job_id, splat())
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    report = clean.run(job_id)
    assert report["outcome"] == outcome
    for word in words:
        assert word in report["error"]["readable"]
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before  # Req 1.11, 10.8


def listing(folder: Path) -> dict[str, bytes | None]:
    """Every file and folder under `folder`, by its path from there with / between the parts:
    each file with its bytes, and each folder with None."""
    return {path.relative_to(folder).as_posix(): None if path.is_dir() else path.read_bytes()
            for path in folder.rglob("*")}


def test_a_done_run_changes_only_work_clean_the_two_outputs_and_the_report(
        new_job: Callable[..., str], stand_in: StandIn) -> None:
    job_id = new_job()
    put_source(job_id, gaussians(N))
    folder = folder_of(job_id)
    before = listing(folder)
    done_run(job_id)
    after = listing(folder)
    changed = {name for name in before.keys() | after.keys()
               if name not in before or name not in after or before[name] != after[name]}
    in_work_folder = {name for name in changed
                      if name == "work/clean" or name.startswith("work/clean/")}
    # The two outputs, and the report in the folder the Step_Runner makes for it (Req 8.7,
    # 10.11).
    assert changed - in_work_folder == {"out/clean.ply", "out/clean.sog", "out/reports",
                                        "out/reports/clean.json"}


# The CLI (Req 8.2)

def test_main_prints_clean_done_and_gives_0_when_the_step_is_done(
        new_job: Callable[..., str], stand_in: StandIn,
        capsys: pytest.CaptureFixture[str]) -> None:
    job_id = new_job()
    put_source(job_id, gaussians(N))
    assert main(["--job", job_id]) == 0
    captured = capsys.readouterr()
    assert (captured.out, captured.err) == ("clean: done\n", "")


def test_main_gives_1_and_prints_only_the_readable_error_when_the_step_fails(
        new_job: Callable[..., str], stand_in: StandIn, capsys: pytest.CaptureFixture[str],
        tmp_path: Path) -> None:
    job_id = new_job()  # with no trained splat, so the step fails
    assert main(["--job", job_id]) == 1
    captured = capsys.readouterr()
    readable = saved_report(job_id)["error"]["readable"]
    assert (captured.out, captured.err) == ("", readable + "\n")
    assert "Traceback" not in without(captured.err, tmp_path)


def test_python_m_splatpipe_clean_with_a_bad_job_id_exits_with_1_from_the_repo_root() -> None:
    """In its own process, the CLI reads the real config.toml, whose Jobs_Folder is data/jobs.
    It refuses the Job_Id before it uses any path, so it writes nothing there. Exit code 1 with
    the Job_Id refusal on stderr proves the module path, because a missing module exits with 1
    too, but with Python's own message."""
    bad_id = "../x"
    result = subprocess.run([sys.executable, "-m", "splatpipe.clean", "--job", bad_id],
                            cwd=REPO_ROOT, env={**os.environ, "PYTHONIOENCODING": "utf-8"},
                            capture_output=True, encoding="utf-8", timeout=60)
    with pytest.raises(PipelineError) as caught:  # the Readable_Error it gives
        check_job_id(bad_id)
    assert (result.returncode, result.stdout, result.stderr) == (1, "", f"{caught.value}\n")
    assert "Traceback" not in without(result.stderr, bad_id)


# The real splat-transform (Req 5.1, 10.6)

@needs_splat_transform
def test_a_done_run_with_the_real_splat_transform_moves_a_sog_of_version_2_into_out(
        new_job: Callable[..., str]) -> None:
    job_id = new_job()
    put_source(job_id, gaussians(64))  # 64 Gaussians with 3 bands, as in task 7's real run
    report = clean.run(job_id, preset="default")  # which keeps 1 band, with 1 sh_iteration
    assert report["outcome"] == "done", report["error"]
    sog = folder_of(job_id) / "out" / "clean.sog"
    assert report["counts"]["clean_sog_bytes"] == sog.stat().st_size
    with zipfile.ZipFile(sog) as bundle:  # one bundled .sog
        assert json.loads(bundle.read("meta.json"))["version"] == 2


# Memory (Req 8.9)

# How many Gaussians the memory test's splat holds, as the design gives it: enough that numpy's
# arrays, hundreds of bytes a Gaussian, dwarf the run's own Python objects.
MANY = 20_000
# A megabyte, 10**6 bytes, as the design's Memory section counts them.
MB = 10**6
# The most the traced peak may rise above the start, in bytes a Gaussian. The design's Memory
# table puts numpy's arrays at about 510 bytes a Gaussian at the isolated stage, the largest, and
# the rest is slack.
PEAK_BYTES_PER_GAUSSIAN = 600


def test_no_array_outlives_the_clean_phase_and_the_peak_stays_within_600_bytes_a_gaussian(
        new_job: Callable[..., str], set_clean: Callable[..., None], stand_in: StandIn,
        monkeypatch: pytest.MonkeyPatch) -> None:
    """The design's Memory arithmetic, checked for what tracemalloc traces.

    The splat is the faint stage's worst case at MANY Gaussians: every one a Faint_Gaussian,
    alone in its Cube at faint_cube_divisions 10**6, because the lattice's spacing of 1 is far
    wider than a Cube_Edge, about 27 / 10**6, and kept by its Cube at faint_min_per_cube 1. So
    the faint filter makes MANY Cubes, the isolated filter's tree and query hold all MANY, and the
    write copies them all, as the report's counts show.

    tracemalloc starts once the splat is written, so its bytes don't count. The stand-in export
    reads the traced memory and peak as it's called, after clean_and_write has returned and
    gc.collect() has run. The memory is back within 1 MB of the start then, so no array outlives
    the clean phase, and the peak so far is at most PEAK_BYTES_PER_GAUSSIAN bytes a Gaussian above
    the start.

    tracemalloc sees numpy's array buffers and Python's objects, but not the memory scipy's
    cKDTree allocates in C++ for its nodes, so both bounds cover only what tracemalloc traces. A
    peak above the Splat_Data's own size shows that tracemalloc saw numpy's arrays.
    """
    job_id = new_job()
    data = gaussians(MANY)
    data["opacity"] = FAINT  # every one a Faint_Gaussian
    put_source(job_id, data)
    set_clean(faint_cube_divisions=10**6, faint_min_per_cube=1)
    readings: list[tuple[int, int]] = []  # the traced memory and peak as each export starts

    def measure_then_export(source: Path, target: Path, bands: int, s: CleanSettings,
                            log_file: Path) -> int:
        readings.append(tracemalloc.get_traced_memory())
        return stand_in(source, target, bands, s, log_file)

    monkeypatch.setattr(clean, "export_sog", measure_then_export)
    tracemalloc.start()
    try:
        start, _ = tracemalloc.get_traced_memory()
        report = clean.run(job_id)
    finally:
        tracemalloc.stop()  # so no other test runs traced, even when this one fails
    assert report["outcome"] == "done", report["error"]
    # Full-size work: no filter dropped a Gaussian, all MANY were written, and no stage was
    # skipped.
    out = folder_of(job_id) / "out"
    assert report["counts"] == {"source_splats": MANY, "dropped_broken": 0, "dropped_unseen": 0,
                                "dropped_faint": 0, "dropped_isolated": 0, "kept_splats": MANY,
                                "clean_ply_bytes": (out / "clean.ply").stat().st_size,
                                "clean_sog_bytes": len(fake.OK_BYTES)}
    assert list(report["timings"]) == list(TIMINGS)
    [(current, peak)] = readings
    assert peak - start > data.nbytes  # tracemalloc saw numpy's arrays
    assert current - start <= MB  # no array outlived the clean phase
    assert peak - start <= PEAK_BYTES_PER_GAUSSIAN * MANY
