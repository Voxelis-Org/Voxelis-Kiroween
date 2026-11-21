"""Unit tests for the Cloud_Runner, cloud/runner.py (cloud-jobs task 6.3).

The Cloud_Runner wraps pipeline-core's `run_job` on the job machine: it rebuilds the Job_Folder from
the Job_Bucket, sets the Cap_Deadline, runs the Step_List, uploads the results, writes the
Spend_Lines, redacts the text logs it uploads, and powers the machine off. These tests drive it on
the CPU with no network, no AWS account and no GPU, against the `FakeAws` seam and the Fake_Step,
with `steps`, `now` and `shutdown` injected (Req 10.11).

- `rebuild_job_folder` downloads `input/`, and any earlier `out/` and `work/`, mapping `jobs/<id>/`
  to the Job_Folder and `work/<id>/` to its `work/`, and leaving the bucket-owned `spend.jsonl` and
  the S3 folder markers alone (Req 10.1).
- A lock-hash drift against the Machine_Image stops with a readable `CloudError` pointing at the
  Image_Build, writes a note to the Job's `out/`, and still powers the machine off (Req 10.6, 10.9).
- `CAP_DEADLINE_ENV` is the machine's start plus the Preset's Time_Cap minus `upload_margin_minutes`,
  as an ISO UTC instant (Req 10.7).
- The uploaded `out/job.log` and `cloud-init-output.log` are Account_Redacted, while the JSON
  reports are uploaded byte-for-byte, so a 12-digit run inside a report survives (Req 10.10).
- `shutdown` runs exactly once, after the final authoritative sync (Req 10.5, 10.6).
- A Spend_Line is written for each Step that ran and one for the machine's own time (Req 12.1, 12.2).

`cloud.runner.REPO_ROOT` is pointed at a temp repo so the archived lock hashes are controlled, and
the machine's environment (the image's recorded hashes and the instance type) is set with
`monkeypatch`. Nothing under `splatpipe/` is changed; the Fake_Step runs through the real `run_job`.
Property 7 (resume) and Property 8 (the uploaded set) are the separate tasks 6.4 and 6.5.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import cloud.runner
import splatpipe.fake
from splatpipe.config import load_config
from splatpipe.jobs import job_folder, open_job, remove_tree
from splatpipe.jsonio import read_json
from splatpipe.run import StepEntry, first_to_run, read_earlier
from splatpipe.step import StepContext

from cloud.errors import CloudError
from cloud.runner import CAP_DEADLINE_ENV, rebuild_job_folder, run_cloud_job
from cloud.spend_file import LABEL_MACHINE

from fake_aws import FakeAws, cloud_config

# The Job_Id the new_job fixture mints, so a test knows the upload keys.
FIXED_ID = "20261119-142530-7f3a9c"
# A real Allowed_Type, so the Spend_Lines take its Config price.
INSTANCE_TYPE = "g5.8xlarge"
# An account-shaped 12-digit run, to show redaction of the text logs and its absence from reports.
ACCOUNT = "123456789012"
# The injected machine start: an aware UTC instant, so the Cap_Deadline is exact.
START = datetime(2026, 11, 20, 0, 0, 0, tzinfo=timezone.utc)
# The Step_Names the Fake_Step runs under, each with a section in the temporary config.
STEP_NAMES = ("fake_a", "fake_b", "fake_c")


def fake_steps(*required: bool) -> list[StepEntry]:
    """A Step_List of Fake_Steps: fake_a, fake_b, fake_c, one per value in `required`, which says
    whether that Step is a Required_Step. Each runs the module splatpipe.fake."""
    names = STEP_NAMES[:len(required)]
    return [StepEntry(name, "splatpipe.fake", flag)
            for name, flag in zip(names, required, strict=True)]


def clock(*times: datetime) -> Callable[[], datetime]:
    """A `now` that returns each given time once, then repeats the last. run_cloud_job calls `now`
    twice, for the machine's start and its end, so two times control the whole run."""
    state = {"i": 0}

    def now() -> datetime:
        value = times[min(state["i"], len(times) - 1)]
        state["i"] += 1
        return value

    return now


def fake_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[str, str]:
    """Point cloud.runner.REPO_ROOT at a temp repo with its own lock files, and return their
    sha-256 hashes, so a test can make the image's recorded hashes match or drift."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "uv.lock").write_bytes(b"version = 1\n")
    (repo / "package-lock.json").write_bytes(b'{"lockfileVersion": 3}\n')
    monkeypatch.setattr(cloud.runner, "REPO_ROOT", repo)
    return (hashlib.sha256((repo / "uv.lock").read_bytes()).hexdigest(),
            hashlib.sha256((repo / "package-lock.json").read_bytes()).hexdigest())


def set_machine_env(monkeypatch: pytest.MonkeyPatch, *, uv_hash: str, package_hash: str) -> None:
    """Set the environment the User_Data exports: the Machine_Image's recorded lock hashes and the
    instance type. The Cap_Deadline variable is set to "" so monkeypatch removes it at teardown,
    even though run_cloud_job overwrites it during the run."""
    monkeypatch.setenv("UV_LOCK_HASH", uv_hash)
    monkeypatch.setenv("PACKAGE_LOCK_HASH", package_hash)
    monkeypatch.setenv("INSTANCE_TYPE", INSTANCE_TYPE)
    monkeypatch.setenv(CAP_DEADLINE_ENV, "")


# rebuild_job_folder (Req 10.1)

def test_rebuild_job_folder_downloads_input_and_earlier_out_and_work(tmp_path: Path) -> None:
    jobs_folder = tmp_path / "jobs"
    jobs_folder.mkdir()
    job_id = FIXED_ID
    aws = FakeAws(objects={
        f"jobs/{job_id}/input/": b"",                               # a folder marker, skipped
        f"jobs/{job_id}/input/job.json": b'{"job_id": "x"}\n',
        f"jobs/{job_id}/input/video.mp4": b"video-bytes",
        f"jobs/{job_id}/out/job.log": b"an earlier log\n",
        f"jobs/{job_id}/out/reports/fake_a.json": b'{"outcome": "done"}\n',
        f"jobs/{job_id}/out/spend.jsonl": b'{"id": "x"}\n',         # bucket-owned, left alone
        f"work/{job_id}/fake_a/file-000.txt": b"work-bytes",
    })

    rebuild_job_folder(aws, job_id, jobs_folder)

    folder = job_folder(job_id, jobs_folder)
    assert (folder / "input" / "job.json").read_bytes() == b'{"job_id": "x"}\n'
    assert (folder / "input" / "video.mp4").read_bytes() == b"video-bytes"
    assert (folder / "out" / "job.log").read_bytes() == b"an earlier log\n"
    assert (folder / "out" / "reports" / "fake_a.json").read_bytes() == b'{"outcome": "done"}\n'
    assert (folder / "work" / "fake_a" / "file-000.txt").read_bytes() == b"work-bytes"
    # The folder marker made no stray file: input/ holds exactly the two downloaded files.
    assert sorted(p.name for p in (folder / "input").iterdir()) == ["job.json", "video.mp4"]
    # spend.jsonl is owned by append_spend_line in the bucket, so the rebuild never downloads it.
    assert not (folder / "out" / "spend.jsonl").exists()
    assert ("s3_download", f"jobs/{job_id}/out/spend.jsonl") not in aws.calls
    assert ("s3_download", f"jobs/{job_id}/input/") not in aws.calls


# The lock-hash backstop (Req 10.6, 10.9)

def test_lock_hash_drift_stops_with_a_readable_error_and_still_powers_off(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_repo(tmp_path, monkeypatch)  # REPO_ROOT's real lock hashes
    set_machine_env(monkeypatch, uv_hash="0" * 64, package_hash="f" * 64)  # the image recorded others
    (tmp_path / "jobs").mkdir()
    aws = FakeAws()
    shutdowns: list[int] = []
    job_id = FIXED_ID

    with pytest.raises(CloudError) as caught:
        run_cloud_job(job_id, aws=aws, config=load_config(), cloud_config=cloud_config(),
                      steps=fake_steps(True), now=clock(START),
                      shutdown=lambda: shutdowns.append(1))

    assert "cloud.image" in str(caught.value)      # a readable message: rebuild the image
    assert shutdowns == [1]                         # still powered off, in the finally (Req 10.6)
    assert f"jobs/{job_id}/out/runner.log" in aws.objects  # the redacted drift note was uploaded
    assert f"jobs/{job_id}/out/job.json" not in aws.objects  # run_job never started


# The Cap_Deadline (Req 10.7)

def test_cap_deadline_is_start_plus_cap_minus_margin(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, new_job: Callable[..., str]) -> None:
    uv_hash, package_hash = fake_repo(tmp_path, monkeypatch)
    set_machine_env(monkeypatch, uv_hash=uv_hash, package_hash=package_hash)
    job_id = new_job()  # the fast preset: cap 120 min, upload margin 10 min
    cc = cloud_config()

    run_cloud_job(job_id, aws=FakeAws(), config=load_config(), cloud_config=cc,
                  steps=fake_steps(True), now=clock(START, START + timedelta(seconds=90)),
                  shutdown=lambda: None)

    margin = cc.preset_plan("fast").cap_minutes - cc.upload_margin_minutes
    expected = (START + timedelta(minutes=margin)).astimezone(timezone.utc).isoformat()
    assert os.environ[CAP_DEADLINE_ENV] == expected


# The log redaction (Req 10.10)

def test_uploaded_text_logs_are_redacted_and_json_reports_are_byte_for_byte(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, new_job: Callable[..., str]) -> None:
    uv_hash, package_hash = fake_repo(tmp_path, monkeypatch)
    set_machine_env(monkeypatch, uv_hash=uv_hash, package_hash=package_hash)
    # cloud-init's own log holds an account id; the Cloud_Runner uploads a redacted copy.
    cloud_init = tmp_path / "cloud-init-output.log"
    cloud_init.write_text(f"cloud-init ran under {ACCOUNT} ok\n", encoding="utf-8")
    monkeypatch.setattr(cloud.runner, "CLOUD_INIT_LOG", cloud_init)

    # A Fake_Step work that logs an account id (into out/job.log) and records a 12-digit count
    # (into its JSON report), so one value shows up in both a text log and a report.
    real_work = splatpipe.fake.work

    def noisy_work(step: StepContext) -> None:
        logging.getLogger("splatpipe.fake").info("saw account %s here", ACCOUNT)
        step.add_count("probe_id", int(ACCOUNT))
        real_work(step)

    monkeypatch.setattr(splatpipe.fake, "work", noisy_work)

    job_id = new_job()
    aws = FakeAws()
    run_cloud_job(job_id, aws=aws, config=load_config(), cloud_config=cloud_config(),
                  steps=fake_steps(True), now=clock(START, START + timedelta(seconds=90)),
                  shutdown=lambda: None)

    job_log = aws.objects[f"jobs/{job_id}/out/job.log"].decode("utf-8")
    assert ACCOUNT not in job_log and "<account>" in job_log
    cloud_init_up = aws.objects[f"jobs/{job_id}/out/cloud-init-output.log"].decode("utf-8")
    assert ACCOUNT not in cloud_init_up and "<account>" in cloud_init_up
    # The JSON report is uploaded unchanged: it equals the local file and keeps its 12-digit count.
    report_key = f"jobs/{job_id}/out/reports/fake_a.json"
    folder = job_folder(job_id, load_config().jobs().jobs_folder)
    assert aws.objects[report_key] == (folder / "out" / "reports" / "fake_a.json").read_bytes()
    assert ACCOUNT in aws.objects[report_key].decode("utf-8")


# The Spend_Lines (Req 12.1, 12.2)

def test_per_step_and_machine_spend_lines_are_written(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, new_job: Callable[..., str]) -> None:
    uv_hash, package_hash = fake_repo(tmp_path, monkeypatch)
    set_machine_env(monkeypatch, uv_hash=uv_hash, package_hash=package_hash)
    job_id = new_job()
    aws = FakeAws()

    run_cloud_job(job_id, aws=aws, config=load_config(), cloud_config=cloud_config(),
                  steps=fake_steps(True, False),  # fake_a required, fake_b optional, both pass
                  now=clock(START, START + timedelta(seconds=120)), shutdown=lambda: None)

    text = aws.objects[f"jobs/{job_id}/out/spend.jsonl"].decode("utf-8")
    lines = [json.loads(part) for part in text.splitlines() if part.strip()]
    by_label = {line["label"]: line for line in lines}
    assert set(by_label) == {"fake_a", "fake_b", LABEL_MACHINE}  # per-Step and machine lines

    folder = job_folder(job_id, load_config().jobs().jobs_folder)
    for name in ("fake_a", "fake_b"):
        line = by_label[name]
        assert (line["job_id"], line["instance_type"], line["estimate"]) == (
            job_id, INSTANCE_TYPE, True)
        assert line["id"] == f"{job_id}:{name}:0"
        # The per-Step seconds come from that Step's report.
        assert line["seconds"] == read_json(folder / "out" / "reports" / f"{name}.json")[
            "duration_seconds"]

    machine = by_label[LABEL_MACHINE]
    assert (machine["instance_type"], machine["estimate"]) == (INSTANCE_TYPE, True)
    # The machine line covers the elapsed wall time outside the Steps.
    step_seconds = by_label["fake_a"]["seconds"] + by_label["fake_b"]["seconds"]
    assert machine["seconds"] == pytest.approx(120 - step_seconds)


# Powering off (Req 10.5, 10.6)

def test_shutdown_is_called_once_after_the_final_sync(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, new_job: Callable[..., str]) -> None:
    uv_hash, package_hash = fake_repo(tmp_path, monkeypatch)
    set_machine_env(monkeypatch, uv_hash=uv_hash, package_hash=package_hash)
    job_id = new_job()
    aws = FakeAws()
    marker = ("shutdown", "")

    run_cloud_job(job_id, aws=aws, config=load_config(), cloud_config=cloud_config(),
                  steps=fake_steps(True), now=clock(START, START + timedelta(seconds=90)),
                  shutdown=lambda: aws.calls.append(marker))

    ops = [op for op, _ in aws.calls]
    assert ops.count("shutdown") == 1  # exactly once (Req 10.6)
    uploads = [i for i, op in enumerate(ops) if op == "s3_upload"]
    assert uploads, "the final sync uploaded the Job's files"
    assert ops.index("shutdown") > uploads[-1]  # after the final authoritative sync (Req 10.5)


# Property 7: the Cloud_Runner resumes exactly where run_job does (Req 10.4, 16.3)

# The four states an earlier Step_Report can be in, as read_earlier sees it: no file at all, a done
# report, a failed report, or bytes that aren't a readable report for this step and job.
EARLIER_STATES = ("none", "done", "failed", "unreadable")


def _record_bytes(job_id: str) -> bytes:
    """A valid Job_Record for `job_id`, as input/job.json, so open_job reads it after the rebuild."""
    record = {"job_id": job_id, "preset": "fast", "start_seconds": 0.0, "end_seconds": 10.0,
              "source_name": "clip.mp4", "source_bytes": 11, "video_seconds": 10.0,
              "video_file": "video.mp4"}
    return json.dumps(record).encode("utf-8")


def _earlier_report_bytes(job_id: str, name: str, state: str) -> bytes | None:
    """The bytes of an earlier Step_Report in the Job_Bucket for `state`, or None for no report.

    "done" and "failed" give a report read_earlier reads as that outcome; "unreadable" gives bytes
    that aren't JSON, so read_earlier calls it unreadable; "none" seeds nothing, a missing report.
    """
    if state == "none":
        return None
    if state == "unreadable":
        return b"{ this is not a readable report"
    outcome = "done" if state == "done" else "failed"
    report: dict[str, object] = {"step": name, "job_id": job_id, "preset": "fast",
                                 "outcome": outcome, "duration_seconds": 1.5}
    if outcome == "failed":
        report["error"] = {"readable": f"the {name} step failed in an earlier run"}
    return json.dumps(report).encode("utf-8")


def _seed_bucket(job_id: str, plan: list[tuple[bool, str]]) -> dict[str, bytes]:
    """The Job_Bucket objects for `plan`: the input the rebuild needs, and the earlier reports."""
    objects = {f"jobs/{job_id}/input/job.json": _record_bytes(job_id),
               f"jobs/{job_id}/input/video.mp4": b"a few bytes"}
    for name, (_required, state) in zip(STEP_NAMES, plan, strict=False):
        report = _earlier_report_bytes(job_id, name, state)
        if report is not None:
            objects[f"jobs/{job_id}/out/reports/{name}.json"] = report
    return objects


@settings(deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(plan=st.lists(st.tuples(st.booleans(), st.sampled_from(EARLIER_STATES)),
                     min_size=1, max_size=len(STEP_NAMES)))
def test_cloud_runner_resumes_exactly_where_run_job_does(
        plan: list[tuple[bool, str]], tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        work_calls: list[str]) -> None:
    """Feature: cloud-jobs, Property 7: The Cloud_Runner resumes exactly where run_job does.

    For any Step_List of required and optional Fake_Steps and any earlier Step_Reports in the
    Job_Bucket (each step: no report, done, failed or unreadable), the Cloud_Runner runs exactly the
    Steps run_job runs: it resumes from the first Step that is not a Finished_Step and never reruns a
    Step that finished before that point, while a Finished_Step after it may run again. The oracle is
    pipeline-core's own read_earlier/first_to_run over the reports the rebuild downloaded, so there
    is no contradiction with pipeline-core's resume (Req 10.4, 16.3).
    """
    # The machine's constants (the image's matching lock hashes and the instance type) are set once:
    # monkeypatch holds across examples, and the temp repo folder is made a single time.
    repo = tmp_path / "repo"
    if not repo.exists():
        uv_hash, package_hash = fake_repo(tmp_path, monkeypatch)
        set_machine_env(monkeypatch, uv_hash=uv_hash, package_hash=package_hash)

    config = load_config()
    cc = cloud_config()
    jobs_folder = config.jobs().jobs_folder
    job_id = FIXED_ID

    # A clean slate for this example: a fresh bucket, and no job folder left by an earlier example.
    folder = jobs_folder / job_id
    if folder.exists():
        remove_tree(folder)
    entries = fake_steps(*[required for required, _state in plan])
    aws = FakeAws(objects=_seed_bucket(job_id, plan))

    # The oracle: run_job's own resume over the reports the Cloud_Runner's rebuild downloads. Read it
    # before run_cloud_job runs, because run_job removes and rewrites the reports it reruns.
    rebuild_job_folder(aws, job_id, jobs_folder)
    job = open_job(job_id, config)
    earlier = [read_earlier(job, entry.name) for entry in entries]
    start = first_to_run(entries, earlier, fresh=False)
    expected_ran = [entry.name for entry in entries[start:]]

    # Run the Cloud_Runner and record which Fake_Steps actually ran (work_calls is cleared so it
    # holds only this example's runs).
    work_calls.clear()
    run_cloud_job(job_id, aws=aws, config=config, cloud_config=cc, steps=entries,
                  now=clock(START, START + timedelta(seconds=90)), shutdown=lambda: None)

    # The Cloud_Runner ran exactly the Steps, in order, that run_job's resume predicts: it never
    # pre-filters a Step, so it resumes exactly where run_job does.
    assert work_calls == expected_ran


# Property 8: the uploaded set equals the Job's own files (Req 10.2, 10.3, 10.5)

# The three Presets the Fake_Step has tables for, varying how many work files each Step writes.
P8_PRESETS = ("fast", "default", "quality")
# The outcome a Fake_Step can be asked for in its own section: pass, fail with its own readable
# error, or fail like a bug. All three leave the Step with a complete report; a failed Required_Step
# stops the Steps after it (they stay not-run, so they write nothing).
FAIL_MODES = ("none", "expected", "unexpected")


def _p8_input(job_id: str, preset: str) -> dict[str, bytes]:
    """The Job_Bucket `input/` a fresh machine downloads: the Job_Record (with `preset`) and the
    video, so the rebuild gives run_job a Job to run under the chosen Preset."""
    record = {"job_id": job_id, "preset": preset, "start_seconds": 0.0, "end_seconds": 10.0,
              "source_name": "clip.mp4", "source_bytes": 11, "video_seconds": 10.0,
              "video_file": "video.mp4"}
    return {f"jobs/{job_id}/input/job.json": json.dumps(record).encode("utf-8"),
            f"jobs/{job_id}/input/video.mp4": b"a few bytes"}


def _uploaded_keys(aws: FakeAws) -> set[str]:
    """Every key the Uploader uploaded through s3_upload, so the bucket-owned spend.jsonl (written
    straight in with s3_put_text) and the seeded input/ are left out."""
    return {key for op, key in aws.calls if op == "s3_upload"}


def _expected_keys(folder: Path, job_id: str) -> set[str]:
    """The keys the Job's local out/ and work/ files map to under the design's mapping: out/ to
    jobs/<id>/out/ and work/ to work/<id>/, with the bucket-owned spend.jsonl left out."""
    keys: set[str] = set()
    out = folder / "out"
    for path in _p8_files(out):
        rel = path.relative_to(out).as_posix()
        if rel == "spend.jsonl":  # append_spend_line owns it in the bucket; it is never uploaded
            continue
        keys.add(f"jobs/{job_id}/out/{rel}")
    work = folder / "work"
    for path in _p8_files(work):
        keys.add(f"work/{job_id}/{path.relative_to(work).as_posix()}")
    return keys


def _p8_files(root: Path) -> list[Path]:
    """Every file under `root`, or none when the folder isn't there."""
    if not root.is_dir():
        return []
    return [path for path in root.rglob("*") if path.is_file()]


@settings(deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(preset=st.sampled_from(P8_PRESETS),
       plan=st.lists(st.tuples(st.booleans(), st.sampled_from(FAIL_MODES)),
                     min_size=1, max_size=len(STEP_NAMES)))
def test_the_uploaded_set_equals_the_jobs_own_files(
        preset: str, plan: list[tuple[bool, str]], tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch, set_fail: Callable[..., None]) -> None:
    """Feature: cloud-jobs, Property 8: The uploaded set equals the Job's own files.

    For any Fake_Step Job the Cloud_Runner runs (any count of required or optional Steps, any mix of
    pass/fail outcomes, any Preset), once run_cloud_job finishes the set of keys the Uploader has
    uploaded equals the Job's out/ files, reports and logs on the machine, with work/ mapped to
    work/<id>/ and the rest to jobs/<id>/, and every uploaded report key (and job.json) holds text
    that parses as a complete JSON object (Req 10.2, 10.3, 10.5).
    """
    # The machine's constants are set once: monkeypatch and tmp_path hold across examples, so the
    # temp repo with its lock files, the image's matching lock hashes, the instance type and a
    # missing cloud-init log (so no cloud-init-output.log is synthesised into out/) are set a single
    # time.
    repo = tmp_path / "repo"
    if not repo.exists():
        uv_hash, package_hash = fake_repo(tmp_path, monkeypatch)
        set_machine_env(monkeypatch, uv_hash=uv_hash, package_hash=package_hash)
        monkeypatch.setattr(cloud.runner, "CLOUD_INIT_LOG", tmp_path / "no-cloud-init-output.log")

    # This example's Step_List, and the fail mode of each Step's own Config section.
    entries = [StepEntry(name, "splatpipe.fake", required)
               for name, (required, _fail) in zip(STEP_NAMES, plan, strict=False)]
    set_fail(**{name: fail for name, (_required, fail) in zip(STEP_NAMES, plan, strict=False)})
    config = load_config()
    jobs_folder = config.jobs().jobs_folder
    job_id = FIXED_ID

    # A clean slate: no job folder from an earlier example, and a fresh bucket holding only input/.
    folder = job_folder(job_id, jobs_folder)
    if folder.exists():
        remove_tree(folder)
    aws = FakeAws(objects=_p8_input(job_id, preset))

    run_cloud_job(job_id, aws=aws, config=config, cloud_config=cloud_config(), steps=entries,
                  now=clock(START, START + timedelta(seconds=90)), shutdown=lambda: None)

    # The uploaded set equals the Job's own out/ and work/ files under the design's mapping.
    assert _uploaded_keys(aws) == _expected_keys(folder, job_id)
    # Every uploaded report key, and job.json, holds text that parses as a complete JSON object.
    for key in _uploaded_keys(aws):
        if key == f"jobs/{job_id}/out/job.json" or (
                key.startswith(f"jobs/{job_id}/out/reports/") and key.endswith(".json")):
            assert isinstance(json.loads(aws.objects[key]), dict)
