"""Tests for the Job_Runner: splatpipe/run.py.

The tests run the Fake_Step under the Step_Names fake_a, fake_b and fake_c, and one restart test,
which needs a fourth Step, also under fake. Each is a StepEntry for the module splatpipe.fake,
and each test chooses which of them are required. The temporary
config from tests/conftest.py has a section for each name with fail = "none", and the set_fail
fixture makes a chosen Step fail. The work_calls fixture records whose work runs, in order.
Each test makes its Job with the new_job fixture, so the Job sits in the temporary Jobs_Folder,
tmp_path / "jobs", and nothing reaches the real data/ folder. The property test at the end runs
no Step: it calls first_to_run, run_from and job_status with a stand-in run_one, and touches no
files.
"""

import dataclasses
import importlib
import json
import logging
import os
import shutil
import subprocess
import sys
from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from splatpipe import fake
from splatpipe.config import REPO_ROOT, load_config
from splatpipe.errors import PipelineError
from splatpipe.jobs import open_job
from splatpipe.jsonio import read_json
from splatpipe.run import (STEP_LIST, Earlier, StepEntry, first_to_run, job_status, main,
                           read_earlier, run_from, run_job)
from splatpipe.step import StepContext, check_step_name, run_step

# Whether a Step in a test's Step_List is a Required_Step or an Optional_Step.
REQUIRED = True
OPTIONAL = False
# The Step_Names the tests run the Fake_Step under, each with a section in the test config.
FAKE_NAMES = ("fake_a", "fake_b", "fake_c")
# The Job_Summary's times, which change from run to run.
SUMMARY_TIMES = ("started_utc", "finished_utc", "duration_seconds")


def fake_steps(*required: bool) -> list[StepEntry]:
    """A Step_List of Fake_Steps: fake_a, then fake_b, then fake_c, one for each value in
    `required`, which says whether that Step is required. Each runs the module splatpipe.fake.
    """
    names = FAKE_NAMES[:len(required)]
    return [StepEntry(name, "splatpipe.fake", flag)
            for name, flag in zip(names, required, strict=True)]


def folder_of(job_id: str) -> Path:
    """The Job_Folder of `job_id` in the temporary config's Jobs_Folder."""
    return load_config().jobs().jobs_folder / job_id


def report_path(job_id: str, name: str) -> Path:
    """The path of the Step `name`'s Step_Report in the Job `job_id`."""
    return folder_of(job_id) / "out" / "reports" / f"{name}.json"


def saved_summary(job_id: str) -> Any:
    """The Job_Summary file, out/job.json, read back."""
    return read_json(folder_of(job_id) / "out" / "job.json")


def untimed(summary: dict[str, Any]) -> dict[str, Any]:
    """The Job_Summary without its times, so a test can compare the rest exactly."""
    return {key: value for key, value in summary.items() if key not in SUMMARY_TIMES}


def summary_step(entry: StepEntry, outcome: str, readable_error: str | None = None, *,
                 reused: bool = False) -> dict[str, Any]:
    """The item the Job_Summary's steps should hold for `entry`."""
    return {"step": entry.name, "required": entry.required, "outcome": outcome,
            "reused": reused, "readable_error": readable_error}


def expected_error(name: str) -> str:
    """The Fake_Step's own Readable_Error when the Step `name` has fail = "expected" (Req 10.5)."""
    return (f'The fake step was set to fail by fail = "expected" in [{name}]. '
            'Set fail = "none" to make it pass.')


def log_lines(job_id: str) -> list[str]:
    """The lines of the Job_Log, out/job.log, each without the UTC time it starts with, like
    "INFO fake_a: started". Each time must have the form 2026-11-19T14:25:31Z."""
    lines = []
    for line in (folder_of(job_id) / "out" / "job.log").read_text(encoding="utf-8").splitlines():
        stamp, text = line.split(" ", 1)
        datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%SZ")  # raises ValueError for any other form
        lines.append(text)
    return lines


def assert_ran_in_order(job_id: str, names: Sequence[str], work_calls: Sequence[str]) -> None:
    """Check that the Steps `names`, and no others, ran on the Job `job_id`, in that order and
    with the Job_Record's Preset (Req 8.4).

    `work_calls` comes from the work_calls fixture. Each Step's report must be for that Step and
    this Job, with the record's Preset, and no report's times may come before those of the
    report before it. The times are to the millisecond, so neighbouring Steps can share one, and
    `work_calls` pins the order.
    """
    assert list(work_calls) == list(names)
    preset = read_json(folder_of(job_id) / "input" / "job.json")["preset"]
    reports = [read_json(report_path(job_id, name)) for name in names]
    assert [(report["step"], report["job_id"], report["preset"]) for report in reports] == [
        (name, job_id, preset) for name in names]
    times = [datetime.fromisoformat(report[key]) for report in reports
             for key in ("started_utc", "finished_utc")]
    assert times == sorted(times)


# End to end with Fake_Steps (Req 10.7, 14.7)

def test_a_job_where_every_step_is_done_is_done(
        new_job: Callable[..., str], work_calls: list[str]) -> None:
    job_id = new_job(preset="quality")  # not the default Preset, so the reports show the record's
    steps = fake_steps(REQUIRED, OPTIONAL, REQUIRED)  # Fake_Steps of both kinds (Req 10.7)
    summary = run_job(job_id, steps=steps)
    assert saved_summary(job_id) == summary
    # A first run: fresh is false, and no Step is reused (Req 8.16).
    assert untimed(summary) == {
        "job_id": job_id, "preset": "quality", "fresh": False, "status": "done",
        "steps": [summary_step(entry, "done") for entry in steps],
        "failed_step": None, "readable_error": None,
    }
    assert_ran_in_order(job_id, ["fake_a", "fake_b", "fake_c"], work_calls)
    assert log_lines(job_id) == [  # Req 8.12
        f"INFO job {job_id}: 3 steps, preset quality",
        "INFO fake_a: started", "INFO fake_a: done",
        "INFO fake_b: started", "INFO fake_b: done",
        "INFO fake_c: started", "INFO fake_c: done",
        f"INFO job {job_id}: done",
    ]


def test_a_failed_optional_step_is_logged_and_the_job_is_done_with_warnings(
        new_job: Callable[..., str], set_fail: Callable[..., None],
        work_calls: list[str]) -> None:
    job_id = new_job()
    set_fail(fake_a="expected")
    steps = fake_steps(OPTIONAL, REQUIRED)
    summary = run_job(job_id, steps=steps)
    error = expected_error("fake_a")
    assert saved_summary(job_id) == summary
    assert untimed(summary) == {
        "job_id": job_id, "preset": "fast", "fresh": False, "status": "done_with_warnings",
        "steps": [summary_step(steps[0], "failed", error), summary_step(steps[1], "done")],
        "failed_step": None, "readable_error": None,
    }
    assert_ran_in_order(job_id, ["fake_a", "fake_b"], work_calls)  # the job went on (Req 8.7)
    assert log_lines(job_id) == [
        f"INFO job {job_id}: 2 steps, preset fast",
        "INFO fake_a: started",
        f"WARNING fake_a: failed, optional so the job goes on: {error}",  # Req 8.7
        "INFO fake_b: started", "INFO fake_b: done",
        f"INFO job {job_id}: done with warnings",
    ]


def test_a_failed_required_step_fails_the_job_and_the_later_steps_are_not_run(
        new_job: Callable[..., str], set_fail: Callable[..., None],
        work_calls: list[str]) -> None:
    job_id = new_job()
    set_fail(fake_b="expected")
    steps = fake_steps(REQUIRED, REQUIRED, OPTIONAL)  # even an Optional_Step after it stops
    summary = run_job(job_id, steps=steps)
    error = expected_error("fake_b")
    assert saved_summary(job_id) == summary
    assert untimed(summary) == {  # Req 8.6, 8.8
        "job_id": job_id, "preset": "fast", "fresh": False, "status": "failed",
        "steps": [summary_step(steps[0], "done"), summary_step(steps[1], "failed", error),
                  summary_step(steps[2], "not_run")],
        "failed_step": "fake_b", "readable_error": error,
    }
    assert_ran_in_order(job_id, ["fake_a", "fake_b"], work_calls)
    assert not report_path(job_id, "fake_c").exists()
    assert log_lines(job_id) == [
        f"INFO job {job_id}: 3 steps, preset fast",
        "INFO fake_a: started", "INFO fake_a: done",
        "INFO fake_b: started",
        f"ERROR fake_b: failed, required so the job stops: {error}",
        "INFO fake_c: not run",
        f"INFO job {job_id}: failed",
    ]


# Checks, imports, log and CLI

def files_in(folder: Path) -> dict[str, bytes | None]:
    """Everything under `folder`, by path relative to it: each file's bytes, and None for each
    folder."""
    return {path.relative_to(folder).as_posix(): path.read_bytes() if path.is_file() else None
            for path in folder.rglob("*")}


def refusal(job_id: str, steps: Sequence[StepEntry]) -> str:
    """The Readable_Error that refuses running `steps` on the new Job `job_id`. Checks that the
    Job's out/ is still empty, so no Step ran and no report or log was written."""
    with pytest.raises(PipelineError) as refused:
        run_job(job_id, steps=steps)
    assert list((folder_of(job_id) / "out").iterdir()) == []
    return str(refused.value)


def test_a_step_entry_holds_the_name_the_module_and_whether_the_step_is_required() -> None:
    assert [field.name for field in dataclasses.fields(StepEntry)] == [
        "name", "module", "required"]  # Req 8.1
    entry = StepEntry(name="fake_a", module="splatpipe.fake", required=OPTIONAL)
    assert (entry.name, entry.module, entry.required) == ("fake_a", "splatpipe.fake", False)


def test_the_step_list_holds_the_frames_and_cameras_steps_and_is_the_default_list(
        new_job: Callable[..., str], work_calls: list[str],
        monkeypatch: pytest.MonkeyPatch) -> None:
    # The Frames_Step and then the Cameras_Step, both required (Req 8.2, frames spec, Req 11.3,
    # and cameras spec, Req 14.3).
    assert STEP_LIST == (StepEntry("frames", "splatpipe.frames", REQUIRED),
                         StepEntry("cameras", "splatpipe.cameras", REQUIRED))
    job_id = new_job()
    monkeypatch.setattr("splatpipe.run.STEP_LIST", tuple(fake_steps(REQUIRED)))
    run_job(job_id)  # with no steps given, the Step_List in code runs (Req 8.3)
    # A list the caller gives runs in its place.
    run_job(job_id, steps=[StepEntry("fake_b", "splatpipe.fake", REQUIRED)])
    assert work_calls == ["fake_a", "fake_b"]


def test_an_empty_step_list_is_refused_before_out_changes(
        new_job: Callable[..., str], monkeypatch: pytest.MonkeyPatch) -> None:
    job_id = new_job()
    run_job(job_id, steps=fake_steps(REQUIRED))  # an earlier run fills out/
    out = folder_of(job_id) / "out"
    before = files_in(out)
    monkeypatch.setattr("splatpipe.run.STEP_LIST", ())  # so None runs an empty Step_List
    for steps in (None, []):  # Req 8.10, and frames spec, Req 11.5
        with pytest.raises(PipelineError, match="^No steps are set up yet"):
            run_job(job_id, steps=steps, fresh=True)  # with fresh, a run would replace job.log
    assert files_in(out) == before


def test_a_step_name_listed_twice_is_refused_before_any_report_exists(
        new_job: Callable[..., str]) -> None:
    job_id = new_job()
    message = refusal(job_id, [*fake_steps(REQUIRED, REQUIRED),
                               StepEntry("fake_a", "splatpipe.fake", OPTIONAL)])
    assert "fake_a" in message and "twice" in message  # Req 8.11


@pytest.mark.parametrize("module", ["os", "cloud.spend", "splatpipe", "splatpipex.fake"])
def test_a_module_outside_splatpipe_is_refused_before_any_report_exists(
        new_job: Callable[..., str], module: str) -> None:
    job_id = new_job()
    message = refusal(job_id, [*fake_steps(REQUIRED), StepEntry("fake_b", module, REQUIRED)])
    assert repr(module) in message and "inside splatpipe" in message  # Req 12.2


@pytest.mark.parametrize("name", ["", "Fake", "../x", "a-b", "a\n", "run", "jobs", "aws"],
                         ids=repr)
def test_a_bad_or_reserved_step_name_is_refused_before_any_report_exists(
        new_job: Callable[..., str], name: str) -> None:
    job_id = new_job()
    with pytest.raises(PipelineError) as expected:
        check_step_name(name)  # the Step_Runner's own refusal of the name
    message = refusal(job_id, [*fake_steps(REQUIRED), StepEntry(name, "splatpipe.fake", REQUIRED)])
    assert message == str(expected.value)  # Req 7.8


def test_each_step_module_is_imported_right_before_its_step_runs(
        new_job: Callable[..., str], work_calls: list[str],
        monkeypatch: pytest.MonkeyPatch) -> None:
    job_id = new_job()
    imports: list[tuple[str, list[str]]] = []
    real_import = importlib.import_module

    def spy(name: str) -> ModuleType:
        imports.append((name, list(work_calls)))  # with the Steps whose work had run by then
        return real_import(name)

    monkeypatch.setattr("splatpipe.step.importlib", SimpleNamespace(import_module=spy))
    run_job(job_id, steps=fake_steps(REQUIRED, OPTIONAL, REQUIRED))
    assert imports == [("splatpipe.fake", []), ("splatpipe.fake", ["fake_a"]),
                       ("splatpipe.fake", ["fake_a", "fake_b"])]  # Req 8.5
    assert work_calls == ["fake_a", "fake_b", "fake_c"]


def test_twenty_runs_in_one_process_leave_no_handler_and_no_open_file(
        new_job: Callable[..., str], tmp_path: Path) -> None:
    """Each run removes and closes its Job_Log handler and file and puts the splatpipe logger's
    level back. On Windows, a file left open would stop tmp_path from being deleted."""
    job_id = new_job()
    logger = logging.getLogger("splatpipe")
    level = logger.level
    logger.setLevel(logging.WARNING)  # a level of its own, so the test sees it come back
    try:
        for index in range(20):
            run_job(job_id, steps=fake_steps(REQUIRED, OPTIONAL), fresh=index % 2 == 1)
            assert (logger.handlers, logger.level) == ([], logging.WARNING)
    finally:
        logger.setLevel(level)
    shutil.rmtree(tmp_path)
    assert not tmp_path.exists()


def test_main_with_the_empty_default_list_gives_1_and_says_no_steps_are_set_up(
        new_job: Callable[..., str], monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str]) -> None:
    job_id = new_job()
    monkeypatch.setattr("splatpipe.run.STEP_LIST", ())  # main always runs STEP_LIST
    assert main(["--job", job_id]) == 1  # Req 8.10, 9.2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith("No steps are set up yet")
    assert "Traceback" not in captured.err


# Each Job_Status the CLI can end with: whether each Step is required, the Steps set to fail,
# the exit code, the status in words, and the Step whose Readable_Error goes on stderr.
CLI_ENDINGS = [
    pytest.param((REQUIRED, OPTIONAL), {}, 0, "done", None, id="done"),
    pytest.param((OPTIONAL, REQUIRED), {"fake_a": "expected"}, 0, "done with warnings", None,
                 id="done with warnings"),
    pytest.param((REQUIRED, OPTIONAL), {"fake_a": "expected"}, 1, "failed", "fake_a",
                 id="failed"),
]


@pytest.mark.parametrize(("required", "fail", "code", "status", "failed_step"), CLI_ENDINGS)
def test_main_gives_0_for_done_and_done_with_warnings_and_1_for_failed(
        new_job: Callable[..., str], set_fail: Callable[..., None],
        monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
        required: tuple[bool, ...], fail: dict[str, str], code: int, status: str,
        failed_step: str | None) -> None:
    job_id = new_job()
    set_fail(**fail)
    monkeypatch.setattr("splatpipe.run.STEP_LIST", tuple(fake_steps(*required)))
    assert main(["--job", job_id]) == code  # Req 9.1, 9.4
    captured = capsys.readouterr()
    err = "" if failed_step is None else expected_error(failed_step) + "\n"
    assert (captured.out, captured.err) == (f"Job {job_id}: {status}\n", err)


@pytest.mark.parametrize(("module", "readable_error"), [
    ("splatpipe.fake", "That isn't a job id."),
    ("splatpipe.run", "That isn't a job id."),
])
def test_python_m_with_a_bad_job_id_exits_with_1_from_the_repo_root(
        module: str, readable_error: str) -> None:
    """In its own process, each CLI reads the real config.toml, whose Jobs_Folder is data/jobs.
    Neither writes anything: each reads the config but refuses the id before any path is used,
    and the Readable_Error shows it stopped there."""
    result = subprocess.run([sys.executable, "-m", module, "--job", "../x"], cwd=REPO_ROOT,
                            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
                            capture_output=True, encoding="utf-8", timeout=60)
    assert (result.returncode, result.stdout) == (1, "")  # Req 7.1, 9.2
    assert readable_error in result.stderr
    assert "Traceback" not in result.stderr


# Restart (Req 8.9, 8.13 to 8.20, 9.5)

def each_step(summary: dict[str, Any], key: str) -> list[Any]:
    """The value of `key` for each Step in the Job_Summary's steps, in Step_List order."""
    return [step[key] for step in summary["steps"]]


def test_a_run_after_a_failed_required_step_picks_up_at_that_step(
        new_job: Callable[..., str], set_fail: Callable[..., None],
        work_calls: list[str]) -> None:
    job_id = new_job()
    steps = fake_steps(REQUIRED, REQUIRED, REQUIRED)
    set_fail(fake_b="expected")
    assert each_step(run_job(job_id, steps=steps), "outcome") == ["done", "failed", "not_run"]
    work_folder = folder_of(job_id) / "work" / "fake_a"
    before = (report_path(job_id, "fake_a").read_bytes(), files_in(work_folder))
    set_fail()  # fake_b is fixed
    work_calls.clear()
    summary = run_job(job_id, steps=steps)
    assert work_calls == ["fake_b", "fake_c"]  # Req 8.13
    assert (report_path(job_id, "fake_a").read_bytes(), files_in(work_folder)) == before
    assert summary["steps"] == [summary_step(steps[0], "done", reused=True),  # Req 8.16
                                summary_step(steps[1], "done"), summary_step(steps[2], "done")]


def test_a_failed_optional_step_is_not_run_again_and_still_counts(
        new_job: Callable[..., str], set_fail: Callable[..., None],
        work_calls: list[str]) -> None:
    job_id = new_job()
    steps = fake_steps(REQUIRED, OPTIONAL, REQUIRED)
    set_fail(fake_b="expected", fake_c="expected")
    assert each_step(run_job(job_id, steps=steps), "outcome") == ["done", "failed", "failed"]
    set_fail(fake_b="expected")  # only fake_c is fixed
    work_calls.clear()
    summary = run_job(job_id, steps=steps)
    assert work_calls == ["fake_c"]  # Req 8.13
    assert untimed(summary) == {  # fake_b's reused failure counts (Req 8.17)
        "job_id": job_id, "preset": "fast", "fresh": False, "status": "done_with_warnings",
        "steps": [summary_step(steps[0], "done", reused=True),
                  summary_step(steps[1], "failed", expected_error("fake_b"), reused=True),
                  summary_step(steps[2], "done")],
        "failed_step": None, "readable_error": None,
    }


# The two ways to run a Job with the Fresh_Option, each given the Job_Id.
FRESH_RUNS = [
    pytest.param(lambda job_id: run_job(job_id, fresh=True), id="run_job"),
    pytest.param(lambda job_id: main(["--job", job_id, "--fresh"]), id="main"),
]


@pytest.mark.parametrize("fresh_run", FRESH_RUNS)
def test_fresh_runs_every_step_again(
        new_job: Callable[..., str], work_calls: list[str], monkeypatch: pytest.MonkeyPatch,
        fresh_run: Callable[[str], object]) -> None:
    job_id = new_job()
    steps = fake_steps(REQUIRED, OPTIONAL, REQUIRED)
    monkeypatch.setattr("splatpipe.run.STEP_LIST", tuple(steps))  # main always runs STEP_LIST
    run_job(job_id)  # a done run, so every Step is a Finished_Step
    work_calls.clear()
    fresh_run(job_id)
    assert work_calls == list(FAKE_NAMES)  # Req 8.14
    assert untimed(saved_summary(job_id)) == {
        "job_id": job_id, "preset": "fast", "fresh": True, "status": "done",
        "steps": [summary_step(entry, "done") for entry in steps],  # every reused is false
        "failed_step": None, "readable_error": None,
    }


def test_a_run_removes_the_summary_and_the_reports_it_replaces_before_any_step_runs(
        new_job: Callable[..., str], set_fail: Callable[..., None], work_calls: list[str],
        monkeypatch: pytest.MonkeyPatch) -> None:
    job_id = new_job()
    steps = fake_steps(REQUIRED, REQUIRED, REQUIRED)
    run_job(job_id, steps=steps)  # all three done, with a Job_Summary
    set_fail(fake_b="expected")
    assert run_step("fake_b", "splatpipe.fake", job_id)["outcome"] == "failed"  # by hand
    report_path(job_id, "other").write_text('{"step": "other"}\n', encoding="utf-8")  # unlisted
    kept = {name: report_path(job_id, name).read_bytes() for name in ("fake_a", "other")}
    summary_file = folder_of(job_id) / "out" / "job.json"
    assert summary_file.exists() and report_path(job_id, "fake_c").exists()
    seen: list[tuple[bool, bool, bool]] = []
    recorded_work = fake.work  # the work_calls wrapper, which runs the real work

    def checking_work(step: StepContext) -> None:
        if step.name == "fake_b":  # what's left in out/ as fake_b's work starts
            seen.append((summary_file.exists(), report_path(job_id, "fake_c").exists(),
                         report_path(job_id, "fake_a").exists()))
        recorded_work(step)

    monkeypatch.setattr(fake, "work", checking_work)
    work_calls.clear()
    summary = run_job(job_id, steps=steps)  # fake_b is still set to fail
    assert seen == [(False, False, True)]  # Req 8.9
    assert work_calls == ["fake_b"]
    assert summary["steps"][2] == summary_step(steps[2], "not_run")
    assert not report_path(job_id, "fake_c").exists()
    assert {name: report_path(job_id, name).read_bytes() for name in kept} == kept


def test_main_on_a_job_where_every_step_finished_runs_nothing_and_gives_0(
        new_job: Callable[..., str], set_fail: Callable[..., None], work_calls: list[str],
        monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    job_id = new_job()
    set_fail(fake_a="expected")
    steps = fake_steps(OPTIONAL, REQUIRED, REQUIRED)
    monkeypatch.setattr("splatpipe.run.STEP_LIST", tuple(steps))
    assert run_job(job_id)["status"] == "done_with_warnings"
    work_calls.clear()
    assert main(["--job", job_id]) == 0  # Req 9.5
    assert work_calls == []  # Req 8.15
    out = capsys.readouterr().out
    assert out.startswith(f"Job {job_id}: done with warnings") and "nothing ran" in out
    assert untimed(saved_summary(job_id)) == {  # from the earlier reports (Req 8.15, 8.17)
        "job_id": job_id, "preset": "fast", "fresh": False, "status": "done_with_warnings",
        "steps": [summary_step(steps[0], "failed", expected_error("fake_a"), reused=True),
                  summary_step(steps[1], "done", reused=True),
                  summary_step(steps[2], "done", reused=True)],
        "failed_step": None, "readable_error": None,
    }


# Earlier reports that don't count: each gives the text that replaces fake_a's done report.
REPORTS_THAT_DONT_COUNT = [
    pytest.param(lambda report: "not JSON", id="not JSON"),
    pytest.param(lambda report: "NaN", id="NaN"),
    pytest.param(lambda report: json.dumps([report]), id="a JSON array"),
    pytest.param(lambda report: json.dumps({**report, "step": "fake_b"}), id="another step"),
    pytest.param(lambda report: json.dumps({**report, "job_id": "20261119-142530-000000"}),
                 id="another job"),
    pytest.param(lambda report: json.dumps({**report, "outcome": "not_run"}), id="not_run"),
]


@pytest.mark.parametrize("replacement", REPORTS_THAT_DONT_COUNT)
def test_a_report_that_does_not_count_is_unreadable_and_its_step_runs_again(
        new_job: Callable[..., str], work_calls: list[str],
        replacement: Callable[[dict[str, Any]], str]) -> None:
    job_id = new_job()
    steps = fake_steps(REQUIRED)
    run_job(job_id, steps=steps)  # fake_a is done
    path = report_path(job_id, "fake_a")
    path.write_text(replacement(read_json(path)), encoding="utf-8")
    assert read_earlier(open_job(job_id, load_config()), "fake_a") == Earlier("unreadable")
    work_calls.clear()
    run_job(job_id, steps=steps)
    assert work_calls == ["fake_a"]  # the run starts at fake_a


def test_read_earlier_gives_missing_with_no_report_and_unreadable_for_a_folder(
        new_job: Callable[..., str]) -> None:
    job = open_job(new_job(), load_config())
    assert read_earlier(job, "fake_a") == Earlier("missing")
    report_path(job.id, "fake_a").mkdir(parents=True)  # a folder in the report's place
    assert read_earlier(job, "fake_a") == Earlier("unreadable")


def test_a_finished_step_after_an_unfinished_one_runs_again(
        new_job: Callable[..., str], work_calls: list[str]) -> None:
    job_id = new_job()
    steps = fake_steps(REQUIRED, REQUIRED, REQUIRED)
    run_job(job_id, steps=steps)  # all three done
    report_path(job_id, "fake_b").unlink()
    work_calls.clear()
    summary = run_job(job_id, steps=steps)
    assert work_calls == ["fake_b", "fake_c"]  # though fake_c had finished (Req 8.13)
    assert each_step(summary, "reused") == [True, False, False]


def test_the_job_log_grows_without_fresh_and_starts_again_with_it(
        new_job: Callable[..., str], set_fail: Callable[..., None]) -> None:
    job_id = new_job()
    steps = fake_steps(OPTIONAL, REQUIRED, REQUIRED)
    warning_a = f"WARNING fake_a: failed, optional so the job goes on: {expected_error('fake_a')}"
    set_fail(fake_a="expected", fake_b="expected")
    run_job(job_id, steps=steps)
    first = [
        f"INFO job {job_id}: 3 steps, preset fast",
        "INFO fake_a: started", warning_a,
        "INFO fake_b: started",
        f"ERROR fake_b: failed, required so the job stops: {expected_error('fake_b')}",
        "INFO fake_c: not run",
        f"INFO job {job_id}: failed",
    ]
    set_fail(fake_a="expected")  # fake_b is fixed
    run_job(job_id, steps=steps)
    second = [
        f"INFO job {job_id}: 3 steps, preset fast",
        "INFO skipped, finished in an earlier run: fake_a (failed)",
        "INFO fake_b: started", "INFO fake_b: done",
        "INFO fake_c: started", "INFO fake_c: done",
        f"INFO job {job_id}: done with warnings",
    ]
    assert log_lines(job_id) == first + second  # Req 8.18
    run_job(job_id, steps=steps, fresh=True)
    assert log_lines(job_id) == [  # only this run's lines (Req 8.19)
        f"INFO job {job_id}: 3 steps, preset fast",
        "INFO fake_a: started", warning_a,
        "INFO fake_b: started", "INFO fake_b: done",
        "INFO fake_c: started", "INFO fake_c: done",
        f"INFO job {job_id}: done with warnings",
    ]


@pytest.mark.parametrize("fresh", [False, True], ids=["without fresh", "fresh"])
def test_run_job_runs_the_steps_from_the_start_that_first_to_run_picks(
        new_job: Callable[..., str], set_fail: Callable[..., None], work_calls: list[str],
        fresh: bool) -> None:
    job_id = new_job()
    # The four earlier states over required and optional Steps: fake_a is required and done,
    # fake_b optional and failed, fake_c required and unreadable, and fake optional and missing.
    entries = [*fake_steps(REQUIRED, OPTIONAL, REQUIRED),
               StepEntry("fake", "splatpipe.fake", OPTIONAL)]
    set_fail(fake_b="expected")
    run_job(job_id, steps=entries[:3])
    report_path(job_id, "fake_c").write_text("not JSON", encoding="utf-8")
    job = open_job(job_id, load_config())
    earlier = [read_earlier(job, entry.name) for entry in entries]
    assert [before.state for before in earlier] == ["done", "failed", "unreadable", "missing"]
    start = first_to_run(entries, earlier, fresh)  # worked out before the run
    assert start == (0 if fresh else 2)
    work_calls.clear()
    summary = run_job(job_id, steps=entries, fresh=fresh)
    assert work_calls == [entry.name for entry in entries[start:]]  # Req 8.20
    assert each_step(summary, "reused") == [index < start for index in range(len(entries))]


# Property tests

# Property 7's plans: for each Step in order, whether it's required, the state of its earlier
# report, and its outcome if it runs. Short lists already reach every case the property covers.
plans = st.lists(
    st.tuples(
        st.booleans(),                                                 # required
        st.sampled_from(["missing", "unreadable", "done", "failed"]),  # earlier state
        st.sampled_from(["done", "failed"]),                           # outcome if it runs
    ),
    min_size=1, max_size=8,
)

# A drawn plan: one (required, earlier state, outcome if it runs) tuple for each Step.
Plan = Sequence[tuple[bool, str, str]]


def earlier_error(state: str) -> str | None:
    """The readable error of an Earlier with `state`: "earlier error" when it's failed."""
    return "earlier error" if state == "failed" else None


def new_error(outcome: str) -> str | None:
    """The stand-in run_one's readable error: "new error" when `outcome` is failed."""
    return "new error" if outcome == "failed" else None


def is_finished(required: bool, state: str) -> bool:
    """Whether a Step is a Finished_Step: its earlier state is done, or failed and it's optional."""
    return state == "done" or (state == "failed" and not required)


def run_span(plan: Plan, fresh: bool) -> tuple[int, int]:
    """The oracle's start and end: run_one runs the Steps from start to end - 1, in order.

    start is 0 with `fresh`. Without it, it's the first Step that isn't a Finished_Step, or the
    plan's length when every Step is one. end is one past the first Required_Step from start on
    whose outcome is failed, or the plan's length when there's none.
    """
    unfinished = [index for index, (required, state, _) in enumerate(plan)
                  if not is_finished(required, state)]
    start = 0 if fresh else min(unfinished, default=len(plan))
    stops = [index for index, (required, _, outcome) in enumerate(plan)
             if index >= start and required and outcome == "failed"]
    end = stops[0] + 1 if stops else len(plan)
    return start, end


def expected_steps(entries: Sequence[StepEntry], plan: Plan, start: int,
                   end: int) -> list[dict[str, Any]]:
    """The oracle's Job_Summary steps: reused with the earlier state before start, run from start
    to end - 1 with the drawn outcome, and not_run after."""
    steps = []
    for index, (entry, (_, state, outcome)) in enumerate(zip(entries, plan, strict=True)):
        if index < start:
            steps.append(summary_step(entry, state, earlier_error(state), reused=True))
        elif index < end:
            steps.append(summary_step(entry, outcome, new_error(outcome)))
        else:
            steps.append(summary_step(entry, "not_run"))
    return steps


def expected_status(steps: Sequence[dict[str, Any]]) -> str:
    """Req 8.8's Job_Status of the oracle's steps: failed when a Required_Step failed, done when
    every Step is done, and done_with_warnings otherwise."""
    if any(step["required"] and step["outcome"] == "failed" for step in steps):
        return "failed"
    if all(step["outcome"] == "done" for step in steps):
        return "done"
    return "done_with_warnings"


@given(plans, st.booleans())
def test_a_run_picks_up_at_the_first_unfinished_step(plan: Plan, fresh: bool) -> None:
    """Feature: pipeline-core, Property 7: A run picks up at the first unfinished Step

    **Validates: Requirements 8.6, 8.8, 8.13, 8.14, 8.15, 8.16, 8.17, 8.20**
    """
    entries = [StepEntry(f"s{index}", "splatpipe.fake", required)
               for index, (required, _, _) in enumerate(plan)]
    earlier = [Earlier(state, earlier_error(state)) for _, state, _ in plan]
    outcomes = {f"s{index}": outcome for index, (_, _, outcome) in enumerate(plan)}
    calls: list[StepEntry] = []

    def run_one(entry: StepEntry) -> tuple[str, str | None]:
        calls.append(entry)
        return outcomes[entry.name], new_error(outcomes[entry.name])

    expected_start, end = run_span(plan, fresh)
    start = first_to_run(entries, earlier, fresh)
    assert start == expected_start
    steps = run_from(entries, earlier, start, run_one)
    assert calls == entries[start:end]
    expected = expected_steps(entries, plan, start, end)
    assert steps == expected
    assert job_status(steps) == expected_status(expected)
