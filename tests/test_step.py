"""Tests for the Step_Runner and the Fake_Step: splatpipe/step.py and splatpipe/fake/.

Each test makes its Job with the new_job fixture from tests/conftest.py, so the Job sits in the
temporary Jobs_Folder, tmp_path / "jobs", and nothing reaches the real data/ folder. In the
temporary config, [fake] has fail = "none" and file_bytes = 16, and its fast, default and
quality Presets write 1, 2 and 3 files. The set_fail fixture changes fail.
"""

import math
import stat
from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

import splatpipe
from splatpipe import fake
from splatpipe.config import JOBS_SECTION, NOT_PIPELINE, load_config
from splatpipe.errors import PipelineError
from splatpipe.fake.__main__ import main as fake_main
from splatpipe.jsonio import check_json_value, read_json
from splatpipe.step import RESERVED_NAMES, StepContext, check_step_name, run_step


# Step_Reports and the Fake_Step

# The fields of a Step_Report, in the order of the design's data model (Req 6.2, 6.3).
REPORT_FIELDS = ["step", "job_id", "preset", "settings", "outcome", "started_utc",
                 "finished_utc", "duration_seconds", "timings", "counts", "warnings", "error"]
# The warning the Fake_Step adds on every run.
FAKE_WARNING = "This is the fake step. It writes test files and makes no splat."
# The Fake_Step's own Readable_Error for fail = "expected" in [fake] (Req 6.5, 10.5).
EXPECTED_MESSAGE = ('The fake step was set to fail by fail = "expected" in [fake]. '
                    'Set fail = "none" to make it pass.')
# The general Readable_Error for an Unexpected_Error in the fake step (Req 6.6).
BUG_MESSAGE = ("The fake step stopped because of an error in its code. This is a bug, not a "
               "problem with your video. Send out/reports/fake.json to the team.")


def folder_of(job_id: str) -> Path:
    """The Job_Folder of `job_id` in the temporary config's Jobs_Folder."""
    return load_config().jobs().jobs_folder / job_id


def saved_report(job_id: str) -> Any:
    """The fake step's Step_Report file, out/reports/fake.json, read back."""
    return read_json(folder_of(job_id) / "out" / "reports" / "fake.json")


def assert_fake_added(report: dict[str, Any], job_id: str, files: int) -> None:
    """Check the timing, the counts and the warning the Fake_Step adds before its fail check:
    `files` files written, and the size of the Job's copy of the video (Req 6.4, 10.4)."""
    video_bytes = (folder_of(job_id) / "input" / "video.mp4").stat().st_size
    assert list(report["timings"]) == ["write_files"]
    assert report["timings"]["write_files"] >= 0
    assert report["counts"] == {"files_written": files, "video_bytes": video_bytes}
    assert report["warnings"] == [FAKE_WARNING]


def test_fake_run_writes_a_report_with_every_field_and_returns_it(
        new_job: Callable[..., str]) -> None:
    job_id = new_job()
    report = fake.run(job_id)
    assert saved_report(job_id) == report  # Req 6.1
    assert list(report) == REPORT_FIELDS  # Req 6.2
    check_json_value(report, "report")  # only JSON_Values (Req 6.7)
    picked = {key: report[key] for key in ("step", "job_id", "preset", "settings", "outcome",
                                           "error")}
    assert picked == {"step": "fake", "job_id": job_id, "preset": "fast",
                      "settings": {"fail": "none", "file_bytes": 16, "files": 1},
                      "outcome": "done", "error": None}
    started = datetime.fromisoformat(report["started_utc"])
    finished = datetime.fromisoformat(report["finished_utc"])
    assert started.utcoffset() == finished.utcoffset() == timedelta(0)  # both in UTC
    assert started <= finished
    assert report["duration_seconds"] >= 0
    assert_fake_added(report, job_id, files=1)


@pytest.mark.parametrize(("preset", "files"), [("fast", 1), ("default", 2), ("quality", 3)])
def test_the_fake_step_writes_the_presets_files_and_one_result_line(
        new_job: Callable[..., str], preset: str, files: int) -> None:
    job_id = new_job(preset=preset)
    report = fake.run(job_id)
    folder = folder_of(job_id)
    assert (report["preset"], report["settings"]["files"]) == (preset, files)
    written = sorted((folder / "work" / "fake").iterdir())
    assert [path.name for path in written] == [f"file-{index:03d}.txt" for index in range(files)]
    assert [path.stat().st_size for path in written] == [16] * files  # file_bytes in [fake]
    assert_fake_added(report, job_id, files)
    line = (folder / "out" / "fake.txt").read_bytes().decode("utf-8")
    assert line.endswith("\n") and "\n" not in line[:-1] and "\r" not in line  # one LF line


def test_fail_expected_gives_the_steps_own_message_and_keeps_what_it_added(
        new_job: Callable[..., str], set_fail: Callable[..., None]) -> None:
    job_id = new_job()
    set_fail(fake="expected")
    report = fake.run(job_id)
    assert saved_report(job_id) == report
    assert report["outcome"] == "failed"
    error = report["error"]
    assert (error["readable"], error["expected"]) == (EXPECTED_MESSAGE, True)
    technical = error["technical"]
    assert technical["type"] == "splatpipe.errors.ExpectedFailure"
    assert technical["message"] == EXPECTED_MESSAGE
    assert technical["traceback"].startswith("Traceback (most recent call last):")
    assert_fake_added(report, job_id, files=1)


def test_fail_unexpected_gives_the_general_message_and_keeps_what_it_added(
        new_job: Callable[..., str], set_fail: Callable[..., None]) -> None:
    job_id = new_job()
    set_fail(fake="unexpected")
    report = fake.run(job_id)
    assert saved_report(job_id) == report
    assert report["outcome"] == "failed"
    error = report["error"]
    assert (error["readable"], error["expected"]) == (BUG_MESSAGE, False)
    technical = error["technical"]
    assert technical["type"] == "builtins.RuntimeError"
    assert technical["traceback"].startswith("Traceback (most recent call last):")
    assert f"RuntimeError: {technical['message']}" in technical["traceback"]
    assert_fake_added(report, job_id, files=1)


# Step code that adds a bad value to its report, and the field the refusal must name (Req 6.8).
BAD_VALUES = [
    pytest.param(lambda step: step.add_count("frames", 1.0), "counts.frames",
                 id="a float count"),
    pytest.param(lambda step: step.add_timing("solve", math.nan), "timings.solve",
                 id="a NaN timing"),
    pytest.param(lambda step: step.add_warning(""), "warnings[0]", id="an empty warning"),
]


@pytest.mark.parametrize(("work", "field"), BAD_VALUES)
def test_a_bad_value_added_to_the_report_fails_the_step_naming_the_field(
        new_job: Callable[..., str], monkeypatch: pytest.MonkeyPatch,
        work: Callable[[StepContext], None], field: str) -> None:
    monkeypatch.setattr(fake, "work", work)  # run_step looks work up when the Step runs
    job_id = new_job()
    report = fake.run(job_id)
    assert saved_report(job_id) == report
    assert report["outcome"] == "failed"
    error = report["error"]
    assert (error["readable"], error["expected"]) == (BUG_MESSAGE, False)
    assert error["technical"]["type"] == "splatpipe.jsonio.NotJsonError"
    assert field in error["technical"]["message"]
    # The value was refused as it was added, so the report never held it.
    assert (report["timings"], report["counts"], report["warnings"]) == ({}, {}, [])


# Step_Names, opening the Job, the Work_Folder, Presets and the CLI

def tree(folder: Path) -> dict[str, bytes | None]:
    """Every file and folder inside `folder`, by its path relative to `folder`: a file with its
    bytes, a folder with None. Dict equality ignores order, so two trees compare directly."""
    return {path.relative_to(folder).as_posix(): None if path.is_dir() else path.read_bytes()
            for path in folder.rglob("*")}


# Step_Names that break the pattern [a-z][a-z0-9_]*, then every Reserved_Name (Req 7.8).
BAD_STEP_NAMES = [
    pytest.param("", id="empty"),
    pytest.param("Fake", id="upper case"),
    pytest.param("../x", id="path"),
    pytest.param("a-b", id="hyphen"),
    pytest.param("1x", id="starts with a digit"),
    pytest.param("a\n", id="line break"),
    *[pytest.param(name, id=f"reserved {name}") for name in sorted(RESERVED_NAMES)],
]


@pytest.mark.parametrize("name", BAD_STEP_NAMES)
def test_a_bad_or_reserved_step_name_is_refused_and_nothing_is_written(
        new_job: Callable[..., str], tmp_path: Path, name: str) -> None:
    job_id = new_job()
    before = tree(tmp_path)
    with pytest.raises(PipelineError):
        check_step_name(name)
    with pytest.raises(PipelineError):
        run_step(name, "splatpipe.fake", job_id)
    assert tree(tmp_path) == before  # no Step_Report and no Work_Folder


def test_reserved_names_hold_every_skeleton_module_and_section() -> None:
    # The modules are read from splatpipe/ at test time, so a Skeleton module added later and
    # left out of RESERVED_NAMES fails this test.
    modules = {path.stem for path in Path(splatpipe.__file__).parent.glob("*.py")} - {"__init__"}
    assert modules | NOT_PIPELINE | {JOBS_SECTION} <= RESERVED_NAMES


# A Job_Id that fails the Job_Id_Check, and a good Job_Id with no Job_Folder (Req 7.2, 7.3).
@pytest.mark.parametrize("job_id", ["../x", "20261119-142530-000000"],
                         ids=["bad job id", "no job folder"])
def test_a_job_that_cant_be_opened_is_refused_and_nothing_is_written(
        new_job: Callable[..., str], tmp_path: Path, job_id: str) -> None:
    new_job()  # a real Job next to the one that can't be opened
    before = tree(tmp_path)
    with pytest.raises(PipelineError):
        fake.run(job_id)
    assert tree(tmp_path) == before  # no Step_Report


def test_preset_quality_on_the_command_line_replaces_the_job_records_preset(
        new_job: Callable[..., str]) -> None:
    job_id = new_job()  # the Job_Record gets the test config's default Preset, fast
    assert fake_main(["--job", job_id, "--preset", "quality"]) == 0
    report = saved_report(job_id)
    ran_with = (report["preset"], report["settings"]["files"], report["counts"]["files_written"])
    assert ran_with == ("quality", 3, 3)  # Req 7.4
    assert read_json(folder_of(job_id) / "input" / "job.json")["preset"] == "fast"


def test_a_run_empties_only_its_own_report_and_work_folder(new_job: Callable[..., str]) -> None:
    job_id = new_job()
    folder = folder_of(job_id)
    reports = folder / "out" / "reports"
    reports.mkdir()
    (reports / "fake.json").write_bytes(b'{"step": "fake", "outcome": "stale"}\n')
    other_report = reports / "other.json"
    other_report.write_bytes(b'{"step": "other", "outcome": "done"}\n')
    stale = folder / "work" / "fake" / "stale.txt"
    stale.parent.mkdir()
    stale.write_bytes(b"from an earlier run")
    stale.chmod(stat.S_IREAD)  # read-only, so emptying the Work_Folder has to clear the flag
    (folder / "work" / "other").mkdir()
    (folder / "work" / "other" / "keep.txt").write_bytes(b"another step's file")
    before = (tree(folder / "input"), tree(folder / "work" / "other"), other_report.read_bytes())
    report = fake.run(job_id)
    assert report["outcome"] == "done"
    assert saved_report(job_id) == report  # the stale report is gone (Req 7.5)
    assert [path.name for path in stale.parent.iterdir()] == ["file-000.txt"]  # stale.txt too
    after = (tree(folder / "input"), tree(folder / "work" / "other"), other_report.read_bytes())
    assert after == before  # Req 7.6


def test_a_work_folder_linked_outside_the_job_folder_fails_the_step_and_keeps_the_target(
        new_job: Callable[..., str], tmp_path: Path,
        dir_link: Callable[[Path, Path], None]) -> None:
    job_id = new_job()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep.txt").write_bytes(b"not the job's file")
    dir_link(folder_of(job_id) / "work" / "fake", outside)
    report = fake.run(job_id)
    assert saved_report(job_id) == report
    assert report["outcome"] == "failed"
    assert "outside the job folder" in report["error"]["readable"]  # Req 7.7
    assert tree(outside) == {"keep.txt": b"not the job's file"}


# The fail setting, then the exit code and what the CLI prints on stdout and on stderr
# (Req 7.1, 9.1 to 9.3).
CLI_RUNS = [
    pytest.param("none", 0, "fake: done\n", "", id="done"),
    pytest.param("expected", 1, "", EXPECTED_MESSAGE + "\n", id="expected failure"),
    pytest.param("unexpected", 1, "", BUG_MESSAGE + "\n", id="unexpected error"),
]


@pytest.mark.parametrize(("fail", "code", "out", "err"), CLI_RUNS)
def test_the_cli_gives_0_when_done_and_1_with_only_the_readable_error_when_failed(
        new_job: Callable[..., str], set_fail: Callable[..., None],
        capsys: pytest.CaptureFixture[str], fail: str, code: int, out: str, err: str) -> None:
    job_id = new_job()
    set_fail(fake=fail)
    assert fake_main(["--job", job_id]) == code
    captured = capsys.readouterr()
    assert (captured.out, captured.err) == (out, err)
    assert "Traceback" not in captured.err


# Running a Step again

# The Fake_Step's own function and its CLI, each as a run on a Job that gives True when the
# Step is done (Req 7.1).
OWN_RUNS = [
    pytest.param(lambda job_id: fake.run(job_id)["outcome"] == "done", id="function"),
    pytest.param(lambda job_id: fake_main(["--job", job_id]) == 0, id="cli"),
]


@pytest.mark.parametrize("run_done", OWN_RUNS)
def test_a_step_whose_report_says_done_runs_again_through_its_function_and_cli(
        new_job: Callable[..., str], work_calls: list[str],
        run_done: Callable[[str], bool]) -> None:
    job_id = new_job()
    assert run_done(job_id)
    assert saved_report(job_id)["outcome"] == "done"  # the Job's fake report says done
    assert run_done(job_id)
    assert work_calls == ["fake", "fake"]  # the second run ran work again (Req 7.9)


def test_a_done_run_replaces_an_earlier_result_file_with_its_own_line(
        new_job: Callable[..., str]) -> None:
    job_id = new_job()
    result = folder_of(job_id) / "out" / "fake.txt"
    result.write_bytes(b"An earlier run's result,\r\nin two lines.\r\n")
    assert fake.run(job_id)["outcome"] == "done"
    line = f"The fake step ran on job {job_id} with the fast preset.\n"
    assert result.read_bytes() == line.encode("utf-8")  # Req 7.10


def test_a_failed_run_leaves_the_steps_earlier_files_in_out_and_writes_its_report(
        new_job: Callable[..., str], set_fail: Callable[..., None]) -> None:
    job_id = new_job()
    assert fake.run(job_id)["outcome"] == "done"  # writes the earlier out/fake.txt
    out = folder_of(job_id) / "out"
    (out / "fake-old.bin").write_bytes(bytes(range(256)))  # a file an earlier run left in out/
    before = {name: (out / name).read_bytes() for name in ("fake.txt", "fake-old.bin")}
    set_fail(fake="expected")
    report = fake.run(job_id)
    assert (report["outcome"], report["error"]["readable"]) == ("failed", EXPECTED_MESSAGE)
    assert saved_report(job_id) == report  # the failed report replaced the done one
    assert {name: (out / name).read_bytes() for name in before} == before  # Req 7.10
