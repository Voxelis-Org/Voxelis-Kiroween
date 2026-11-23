"""Unit tests for the four free laptop commands (cloud-jobs task 8.4).

`cloud/status.py`, `cloud/download.py`, `cloud/list.py` and `cloud/stopall.py` are the read-only and
last-resort laptop commands. They are free: Status and Download and List only read, and Stop_All only
terminates, which costs nothing. These tests drive the command functions and their `main` entry
points against the in-memory `FakeAws` seam, on the CPU with no network and no AWS account, and a
tiny bucket each test seeds itself.

They cover, one behaviour per test:

- `status` reads the Job_Summary and the Step reports from the bucket, prints the progress, and
  merges the Job's Spend_Lines into a temp `data/spend.jsonl` without duplicates (Req 13.1, 12.4).
- `download` fetches `jobs/<id>/out/*` into `data/jobs/<id>/out/` and merges the Spend_Lines
  (Req 13.2, 12.4).
- `list_machines` returns our tagged non-terminated instances, and `cloud.list`'s `main` prints each
  one's type, state and launch time with no account id (Req 13.3, 13.5).
- `stop_all` terminates our machines and never calls the Spend_Guard, because terminating is free and
  is the safety switch that must never be gated on an approval (Req 6.5, 13.4).
- Each command builds its `Aws` with the right profile key: the Read_Profile for status, download and
  list, the Runner_Profile for stop-all (Req 1.4, 1.5).
- An AWS error is Account_Redacted before it reaches the terminal, by `run_cloud_cli` (Req 13.6).

Nothing here touches the real `data/` folder (every write goes under `tmp_path`), sets
`APPROVED_SPEND`, or runs a real `aws` command. The `FakeAws` and the seeded bucket come from
`tests/fake_aws.py`.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

import cloud.download
import cloud.list as list_cmd  # 'list' shadows a builtin, so the module is aliased here
import cloud.spend
import cloud.status
import cloud.stopall
from cloud.aws import InstanceInfo
from cloud.download import download
from cloud.errors import CloudError
from cloud.list import list_machines
from cloud.stack import resolve_stack
from cloud.status import status
from cloud.stopall import stop_all
from splatpipe.config import Config

from fake_aws import FakeAws

# A valid Job_Id (YYYYMMDD-HHMMSS-xxxxxx), so check_job_id accepts it and the seeded bucket keys are
# known. It is the same fixed id the pipeline-core fixtures and the Start_Command tests use.
FIXED_ID = "20261119-142530-7f3a9c"

# The Stack name read from the test [aws] table, and the seven Stack_Outputs resolve_stack maps. No
# value holds a 12-digit account id; the structural no-account-literal check lives in the stack test.
STACK_NAME = "voxelis-kiroween"
STACK = {
    "BucketName": "voxelis-kiroween-jobs",
    "JobRoleArn": "arn:aws:iam:::role/voxelis-job",
    "JobInstanceProfileArn": "arn:aws:iam:::instance-profile/voxelis-job",
    "RunnerRoleArn": "arn:aws:iam:::role/voxelis-runner",
    "SecurityGroupId": "sg-00000000",
    "VpcId": "vpc-00000000",
    "GroupName": "voxelis-kiroween",
}
# The StackOutputs that status and download take; built from the fake once, so no ARN is hard-coded
# twice. resolve_stack would raise here if STACK were missing an output.
OUTPUTS = resolve_stack(FakeAws(stack=STACK), STACK_NAME)

# The Job_Summary, the two Step reports and the two Spend_Lines a seeded Job holds. The reports carry
# the per-Step durations status prints; the spend lines carry the ids merge_spend_file dedups on.
SUMMARY = {
    "job_id": FIXED_ID,
    "status": "done",
    "duration_seconds": 12.5,
    "steps": [
        {"step": "alpha", "outcome": "ok", "reused": False},
        {"step": "beta", "outcome": "ok", "reused": True},
    ],
}
REPORTS = {
    "alpha": {"outcome": "ok", "duration_seconds": 1.5},
    "beta": {"outcome": "ok", "duration_seconds": 2.0},
}
SPEND_LINES = [
    {"id": f"{FIXED_ID}:alpha:0", "job_id": FIXED_ID, "label": "alpha",
     "instance_type": "g5.8xlarge", "seconds": 90.0, "estimated_usd": 0.06, "estimate": True},
    {"id": f"{FIXED_ID}:machine:0", "job_id": FIXED_ID, "label": "machine",
     "instance_type": "g5.8xlarge", "seconds": 120.0, "estimated_usd": 0.08, "estimate": True},
]


def _seeded_objects() -> dict[str, bytes]:
    """The `out/` objects of one finished Job in the bucket: the summary, the reports, the spend
    ledger and a log, keyed under `jobs/<id>/out/` the way the Cloud_Runner uploads them."""
    spend_text = "".join(json.dumps(line) + "\n" for line in SPEND_LINES)
    return {
        f"jobs/{FIXED_ID}/out/job.json": json.dumps(SUMMARY).encode("utf-8"),
        f"jobs/{FIXED_ID}/out/reports/alpha.json": json.dumps(REPORTS["alpha"]).encode("utf-8"),
        f"jobs/{FIXED_ID}/out/reports/beta.json": json.dumps(REPORTS["beta"]).encode("utf-8"),
        f"jobs/{FIXED_ID}/out/spend.jsonl": spend_text.encode("utf-8"),
        f"jobs/{FIXED_ID}/out/job.log": b"the run finished\n",
    }


def _instance(instance_id: str, instance_type: str, state: str) -> InstanceInfo:
    """One tagged instance in a given state, for the list and stop-all tests."""
    return InstanceInfo(id=instance_id, type=instance_type, state=state, zone="us-east-1a",
                        launch_time="2026-11-20T00:00:00+00:00", state_reason="")


def _config_with_aws(tmp_path: Path) -> Config:
    """A pipeline Config with [jobs] inside tmp_path and an [aws] with the Stack name, so a command's
    `main` can resolve the Stack and take `data/` from the Jobs_Folder's parent (here tmp_path)."""
    data: dict[str, Any] = {
        "jobs": {
            "jobs_folder": (tmp_path / "jobs").as_posix(),  # absolute, so it stays inside tmp_path
            "default_preset": "fast",
            "max_video_bytes": 1024,
            "max_video_seconds": 600,
            "min_used_seconds": 3,
            "ffprobe_timeout_seconds": 30,
        },
        "aws": {"stack_name": STACK_NAME, "project_tag": "project=voxelis-kiroween"},
    }
    return Config(data, source="test config")


def _patch_laptop_aws(monkeypatch: pytest.MonkeyPatch, module: Any, aws: FakeAws) -> list[str]:
    """Patch `module.laptop_aws` to return `aws` and record the profile key it was built with.

    Every command builds its `Aws` through `laptop_aws(config, profile_key=...)`, so recording that
    key is how a test checks the construction path without a real AWS session (Req 1.4, 1.5).
    """
    keys: list[str] = []

    def fake(config: Config, *, profile_key: str) -> FakeAws:
        keys.append(profile_key)
        return aws

    monkeypatch.setattr(module, "laptop_aws", fake)
    return keys


def _count_lines(path: Path) -> int:
    """How many non-blank lines a JSON-Lines file holds, for the dedup assertions."""
    return len([line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()])


# Req 13.1, 12.4: status reads the summary and the reports and merges the spend without duplicates.

def test_status_reads_summary_and_reports_and_merges_without_duplicates(
        tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    aws = FakeAws(stack=STACK, objects=_seeded_objects())
    spend_path = tmp_path / "spend.jsonl"

    result = status(FIXED_ID, aws=aws, outputs=OUTPUTS, data_root=tmp_path)

    # The Job_Summary came back, so job.json was read, with both steps in order.
    assert result["status"] == "done"
    assert [step["step"] for step in result["steps"]] == ["alpha", "beta"]
    # The printed progress uses the reports' per-Step durations, so the reports were read too.
    out = capsys.readouterr().out
    assert "alpha" in out and "beta" in out
    assert "1.5s" in out  # alpha's duration, which only the report carries
    # Both Spend_Lines merged into the temp data/spend.jsonl.
    assert _count_lines(spend_path) == 2

    # Merging the same Job again adds nothing: the ledger is deduplicated by id (Req 12.4).
    status(FIXED_ID, aws=aws, outputs=OUTPUTS, data_root=tmp_path)
    assert _count_lines(spend_path) == 2


# Req 13.2, 12.4: download fetches out/ into data/jobs/<id>/out/ and merges the spend.

def test_download_fetches_out_folder_into_data_and_merges(tmp_path: Path) -> None:
    aws = FakeAws(stack=STACK, objects=_seeded_objects())

    download(FIXED_ID, aws=aws, outputs=OUTPUTS, data_root=tmp_path)

    out_folder = tmp_path / "jobs" / FIXED_ID / "out"
    assert (out_folder / "job.json").exists()
    assert (out_folder / "reports" / "alpha.json").exists()
    assert (out_folder / "reports" / "beta.json").exists()
    assert (out_folder / "spend.jsonl").exists()
    assert (out_folder / "job.log").exists()
    # A downloaded file is the bucket object, so the Job folder looks like one that ran locally.
    downloaded = json.loads((out_folder / "job.json").read_text(encoding="utf-8"))
    assert downloaded["status"] == "done"
    # The Spend_Lines merged into the temp data/spend.jsonl.
    assert _count_lines(tmp_path / "spend.jsonl") == 2


# Req 13.3, 13.5: list_machines returns our tagged non-terminated instances.

def test_list_machines_returns_our_tagged_non_terminated_instances() -> None:
    running = _instance("i-run", "g5.8xlarge", "running")
    stopped = _instance("i-stop", "g5.2xlarge", "stopped")
    terminated = _instance("i-term", "g5.xlarge", "terminated")
    aws = FakeAws(instances=[running, stopped, terminated])

    got = list_machines(aws=aws)

    # The two live machines are returned; the terminated one is left out.
    assert [machine.id for machine in got] == ["i-run", "i-stop"]


# Req 13.3, 13.5: list's main prints each machine's type, state and launch time, with no account id.

def test_list_main_prints_type_state_and_launch_time_without_account_id(
        monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    aws = FakeAws(instances=[_instance("i-run", "g5.8xlarge", "running")])
    monkeypatch.setattr(list_cmd, "laptop_aws", lambda config, *, profile_key: aws)

    assert list_cmd.main([]) == 0

    out = capsys.readouterr().out
    assert "g5.8xlarge" in out      # the type
    assert "running" in out          # the state
    assert "2026-11-20T00:00:00+00:00" in out  # the launch time
    # Every line is Account_Redacted, so no 12-digit account id reaches the terminal (Req 13.7).
    assert re.search(r"(?<!\d)\d{12}(?!\d)", out) is None


# Req 13.4, 6.5: stop_all terminates our machines and never calls the Spend_Guard.

def test_stop_all_terminates_our_machines_with_no_spend_guard(
        monkeypatch: pytest.MonkeyPatch) -> None:
    aws = FakeAws(instances=[
        _instance("i-run", "g5.8xlarge", "running"),
        _instance("i-stop", "g5.2xlarge", "stopped"),
        _instance("i-term", "g5.xlarge", "terminated"),
    ])
    # Terminating is free and is the last-resort stop, so the Spend_Guard must never run here: record
    # any call to it and assert there was none (Req 6.5).
    guard_calls: list[dict[str, Any]] = []
    monkeypatch.setattr(cloud.spend, "check_spend", lambda **kwargs: guard_calls.append(kwargs))

    ids = stop_all(aws=aws)

    # The two live machines were terminated; the already-terminated one was skipped.
    assert ids == ["i-run", "i-stop"]
    assert {m.id: m.state for m in aws.instances}["i-run"] == "terminated"
    assert {m.id: m.state for m in aws.instances}["i-stop"] == "terminated"
    # Only a tagged read and the terminate touched AWS, and the Spend_Guard was never invoked.
    assert [op for op, _ in aws.calls] == ["tagged_instances", "terminate_instances"]
    assert guard_calls == []


# Req 1.4, 1.5: each command builds its Aws with the right profile key, via its main's construction.

def test_each_command_builds_its_aws_with_the_right_profile_key(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = _config_with_aws(tmp_path)

    # Status and Download read, so they build the Read_Profile Aws; both resolve the Stack, so the
    # fake carries one, and an empty bucket makes each main a clean no-op that returns 0.
    monkeypatch.setattr(cloud.status, "load_config", lambda: config)
    status_keys = _patch_laptop_aws(monkeypatch, cloud.status, FakeAws(stack=STACK))
    assert cloud.status.main([FIXED_ID]) == 0
    assert status_keys == ["read_profile"]

    monkeypatch.setattr(cloud.download, "load_config", lambda: config)
    download_keys = _patch_laptop_aws(monkeypatch, cloud.download, FakeAws(stack=STACK))
    assert cloud.download.main([FIXED_ID]) == 0
    assert download_keys == ["read_profile"]

    # List reads too, and needs no Stack.
    list_keys = _patch_laptop_aws(monkeypatch, list_cmd, FakeAws())
    assert list_cmd.main([]) == 0
    assert list_keys == ["read_profile"]

    # Stop_All starts a machine down, so it builds the Runner_Profile Aws (Req 1.5).
    stopall_keys = _patch_laptop_aws(monkeypatch, cloud.stopall, FakeAws())
    assert cloud.stopall.main([]) == 0
    assert stopall_keys == ["runner_profile"]


# Req 13.6: an AWS error is Account_Redacted before it is printed.

class _RaisingAws(FakeAws):
    """A `FakeAws` whose instance read fails with an AWS-style error carrying a 12-digit account id,
    so the redaction on a command's error path can be checked."""

    def tagged_instances(self, states: Any) -> list[InstanceInfo]:
        raise CloudError("An error occurred (UnauthorizedOperation) calling DescribeInstances: "
                         "arn:aws:iam::123456789012:user/intruder is not authorized.")


def test_aws_errors_are_redacted_before_printing(
        monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(list_cmd, "laptop_aws", lambda config, *, profile_key: _RaisingAws())

    assert list_cmd.main([]) == 1  # the error is caught and turned into exit 1

    err = capsys.readouterr().err
    assert "123456789012" not in err  # the raw account id never reaches the terminal
    assert "<account>" in err          # it was replaced by the redaction marker
