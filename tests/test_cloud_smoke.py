"""Unit tests for the Smoke_Test, cloud/smoke.py (cloud-jobs task 7.5).

The Smoke_Test's GPU work runs only on the real machine, so this CPU suite checks the three things
`run_smoke` and its helpers do on the laptop, all against the `FakeAws` seam with no network, no GPU
and no AWS account (Req 16.1):

- `smoke_marker_key` and `image_id_from_marker` round-trip for any image id: the key is
  `jobs/_smoke/<image_id>.json`, and the inverse recovers the id, so the Start_Command's marker gate
  reads back exactly the id the Smoke_Test wrote (Req 11.3). A key that isn't a Smoke_Marker key is a
  `CloudError`.
- With the `FakeAws`, `run_smoke` passes the Spend_Guard first, so nothing paid starts without a yes:
  without `APPROVED_SPEND=1` it raises and touches no AWS at all; with it, the first `s3_upload` and
  the `run_instances` call come only after `check_spend` has returned (Req 6.1, 11.4). It then launches
  one `smoke_instance_type` from the current Machine_Image with the Requirement 8 request and the
  Smoke_Test's `cap_minutes`, and appends a `smoke-test` Spend_Line to the laptop's `data/spend.jsonl`
  (Req 11.1, 11.4, 12.2). The launch request satisfies Property 4's shape (Req 8.9).
- `smoke_log_key` puts the smoke machine's results beside the marker, and the launch message names
  both keys.

What the smoke machine itself runs, `cloud.smoke_runner`, is tested in `test_cloud_smoke_runner.py`.

`cloud.smoke.REPO_ROOT` is pointed at a committed `git_repo`, the way the module's docstring says a
test controls the Code_Archive and the lock hashes, so the archive and `lock_hashes` need no real
repo state. `check_spend` is the real Spend_Guard, fed each test's own `environ`, reading the repo's
`config.toml` for the Spend_Variable's name. Nothing in `splatpipe/` or `cloud/` is changed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import cloud.smoke
from splatpipe.config import Config

from cloud.aws import ImageInfo
from cloud.errors import CloudError
from cloud.launch import JOB_ID_TAG
from cloud.smoke import image_id_from_marker, run_smoke, smoke_log_key, smoke_marker_key
from cloud.spend import SpendRefused, check_spend as _real_check_spend
from cloud.spend_file import LABEL_SMOKE
from cloud.stack import StackOutputs

from fake_aws import FakeAws, cloud_config, git_repo

# A real-shaped AMI id: hex, so render_user_data's AMI pattern accepts it when run_smoke builds the
# machine's User_Data on the happy path.
IMAGE_ID = "ami-0123456789abcdef0"

# The Project_Tag every launch request carries, as the tag dict a TagSpecifications entry holds.
PROJECT_TAG_DICT = {"Key": "project", "Value": "voxelis-kiroween"}

# The one type the fake offers a zone and default subnet for, so launch_plan lands on the smoke
# type's single zone.
OFFERINGS = {"g5.xlarge": ["us-east-1a"]}
SUBNETS = {"us-east-1a": "subnet-a"}

# The keys a launch request must never carry, so the machine is On-Demand and terminable (Req 8.5, 8.1).
FORBIDDEN_KEYS = ("KeyName", "LaunchTemplate", "InstanceMarketOptions", "DisableApiTermination")

# The seven Stack_Outputs run_smoke reads. No value holds a 12-digit account id (the structural
# redaction check lives in the aws/stack tests, not here).
OUTPUTS = StackOutputs(
    bucket="voxelis-kiroween-jobs",
    job_role_arn="arn:aws:iam::role/voxelis-job",
    job_instance_profile_arn="arn:aws:iam::instance-profile/voxelis-job",
    runner_role_arn="arn:aws:iam::role/voxelis-runner",
    security_group_id="sg-fake",
    vpc_id="vpc-fake",
    group_name="voxelis-kiroween",
)

# The op the patched Spend_Guard records the moment check_spend returns without raising, so a test
# can tell which calls came before the guard. It isn't a real Aws op, so it never clashes with one.
GUARD_MARKER = "check_spend_returned"


def _config(tmp_path: Path) -> Config:
    """A pipeline Config with [jobs] inside tmp_path (so the Spend_File stays in tmp_path) and [aws]
    for the Project_Tag run_smoke reads for the launch request's tags."""
    data: dict[str, Any] = {
        "jobs": {
            "jobs_folder": (tmp_path / "jobs").as_posix(),  # absolute, so it stays inside tmp_path
            "default_preset": "fast",
            "max_video_bytes": 1024,
            "max_video_seconds": 600,
            "min_used_seconds": 3,
            "ffprobe_timeout_seconds": 30,
        },
        "aws": {"project_tag": "project=voxelis-kiroween"},
    }
    return Config(data, source="test config")


def _aws(**overrides: Any) -> FakeAws:
    """A healthy FakeAws for the Smoke_Test: one available tagged image, a full quota, and the smoke
    type offered in one zone with a default subnet."""
    image = ImageInfo(id=IMAGE_ID, name="voxelis-image", state="available",
                      creation_date="2026-11-20T00:00:00+00:00", tags={})
    return FakeAws(images=[image], quota=32.0, offerings=OFFERINGS, subnets=SUBNETS, **overrides)


def _paid(aws: FakeAws) -> list[str]:
    """The paid operations recorded on the fake: the S3 uploads and the RunInstances calls."""
    return [op for op, _ in aws.calls if op in ("s3_upload", "run_instances")]


def _tags_for(request: dict[str, Any], resource_type: str) -> list[dict[str, str]] | None:
    """The EC2 Tags list for `resource_type` in a request's TagSpecifications, or None if absent."""
    for spec in request["TagSpecifications"]:
        if spec["ResourceType"] == resource_type:
            return spec["Tags"]
    return None


# Req 11.3: smoke_marker_key and image_id_from_marker round-trip for any image id.

# Representative image ids: a real-shaped AMI id, the fake's own id shape, a one-character id, ids
# with dots, and ids that themselves end in ".json", which prove the inverse is a true inverse
# rather than a greedy suffix strip.
IMAGE_IDS = [
    "ami-0123456789abcdef0",
    "ami-00000000000000000",
    "ami-fake0001",
    "a",
    "image.with.dots",
    "ends.json",
    "ami-0123456789abcdef0.json",
]


@pytest.mark.parametrize("image_id", IMAGE_IDS)
def test_marker_key_round_trips_for_any_image_id(image_id: str) -> None:
    key = smoke_marker_key(image_id)
    # The key lives under jobs/ so the machine's Job_Role writes it and the Read_Profile reads it.
    assert key == f"jobs/_smoke/{image_id}.json"
    # The id comes straight back out, so the Start_Command's gate reads the id the machine wrote.
    assert image_id_from_marker(key) == image_id
    # And the reverse: for a key this module makes, re-keying the recovered id gives the key back.
    assert smoke_marker_key(image_id_from_marker(key)) == key


@pytest.mark.parametrize("bad_key", ["jobs/_smoke/.json", "other/place/key.json"])
def test_image_id_from_marker_rejects_a_non_marker_key(bad_key: str) -> None:
    # An empty image id and a key outside jobs/_smoke/ are both refused, so a stray key can't pass
    # as a Smoke_Marker.
    with pytest.raises(CloudError):
        image_id_from_marker(bad_key)


@pytest.mark.parametrize("image_id", IMAGE_IDS)
def test_the_smoke_log_sits_beside_the_marker_and_is_never_one(image_id: str) -> None:
    key = smoke_log_key(image_id)
    assert key == f"jobs/_smoke/{image_id}.log"
    # The results go up pass or fail, so their key must never read as a passing marker.
    assert key != smoke_marker_key(image_id)
    with pytest.raises(CloudError):
        image_id_from_marker(key)


# Req 6.1, 11.4: no launch without the Spend_Guard's yes.

def test_run_smoke_refuses_without_approval(tmp_path: Path) -> None:
    # The quote needs no AWS, so check_spend is the first thing run_smoke does; without
    # APPROVED_SPEND=1 it raises SpendRefused and nothing paid, nor any AWS call, happens.
    aws = _aws()
    with pytest.raises(SpendRefused):
        run_smoke(aws=aws, config=_config(tmp_path), cloud_config=cloud_config(),
                  outputs=OUTPUTS, environ={})
    assert aws.calls == []  # not even a read: the guard gates everything


# Req 11.1, 11.4, 8.9, 12.2: the approved path launches one smoke machine and records its Spend_Line.

def test_run_smoke_launches_one_smoke_machine_after_the_spend_guard(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    root = git_repo(tmp_path)  # a committed repo; skips with a reason when git isn't installed
    monkeypatch.setattr(cloud.smoke, "REPO_ROOT", root)
    aws = _aws()
    config = _config(tmp_path)
    cc = cloud_config()

    # Spy on the Spend_Guard: record a marker the instant the real guard returns without raising, so
    # the test can show no paid action came before it. It delegates to the real check_spend, which
    # reads the repo's config.toml for the Spend_Variable's name (Req 6.1).
    def guarded_check_spend(**kwargs: Any) -> None:
        _real_check_spend(**kwargs)
        aws.calls.append((GUARD_MARKER, ""))

    monkeypatch.setattr(cloud.smoke, "check_spend", guarded_check_spend)

    # Capture the UserDataValues run_smoke builds, so the test can assert the launch uses the smoke
    # Time_Cap and the smoke role; the real render still runs, so the request's UserData is real.
    captured_values: dict[str, Any] = {}
    real_render = cloud.smoke.render_user_data

    def spy_render(values: Any) -> str:
        captured_values["values"] = values
        return real_render(values)

    monkeypatch.setattr(cloud.smoke, "render_user_data", spy_render)

    # Capture the exact RunInstances request the code builds, to assert Property 4's shape on it.
    captured_request: dict[str, Any] = {}
    real_run_instances = aws.run_instances

    def capture_run_instances(request: dict[str, Any]) -> Any:
        captured_request["request"] = request
        return real_run_instances(request)

    monkeypatch.setattr(aws, "run_instances", capture_run_instances)

    run_smoke(aws=aws, config=config, cloud_config=cc, outputs=OUTPUTS,
              environ={"APPROVED_SPEND": "1"})

    # The Spend_Guard ran, and nothing paid was recorded before its marker (Req 6.1, 11.4).
    ops = [op for op, _ in aws.calls]
    assert GUARD_MARKER in ops
    before_guard = ops[:ops.index(GUARD_MARKER)]
    assert "s3_upload" not in before_guard
    assert "run_instances" not in before_guard

    # Exactly one machine launched, the smoke_instance_type, from the current image (Req 11.1).
    launched = [key for op, key in aws.calls if op == "run_instances"]
    assert launched == [cc.smoke_instance_type]  # "g5.xlarge"
    assert len(aws.instances) == 1

    # The launch uses the Smoke_Test's Time_Cap and the smoke role (Req 11.1).
    values = captured_values["values"]
    assert values.cap_minutes == int(cc.smoke.cap_minutes)  # the smoke cap_minutes (30)
    assert values.role == "smoke"
    assert values.image_id == IMAGE_ID

    # Property 4's shape, on the very request the code built (Req 8.1-8.6, 8.9).
    request = captured_request["request"]
    assert request["ImageId"] == IMAGE_ID
    assert request["InstanceType"] == cc.smoke_instance_type
    assert request["MinCount"] == request["MaxCount"] == 1
    assert request["InstanceInitiatedShutdownBehavior"] == "terminate"
    assert request["MetadataOptions"] == {"HttpTokens": "required", "HttpEndpoint": "enabled"}
    assert request["IamInstanceProfile"] == {"Arn": OUTPUTS.job_instance_profile_arn}
    assert request["SecurityGroupIds"] == [OUTPUTS.security_group_id]
    assert request["SubnetId"] == SUBNETS["us-east-1a"]
    assert request["UserData"].startswith("#!/bin/bash")
    for resource_type in ("instance", "volume", "network-interface"):
        tags = _tags_for(request, resource_type)
        assert tags is not None
        assert PROJECT_TAG_DICT in tags
    instance_tags = _tags_for(request, "instance")
    assert instance_tags is not None
    assert any(tag["Key"] == JOB_ID_TAG for tag in instance_tags)  # a smoke run carries a Job_Id
    for forbidden in FORBIDDEN_KEYS:
        assert forbidden not in request

    # The Code_Archive was the one upload, under this smoke run's own jobs/<id>/input/ (Req 11.2, 7.4).
    uploads = [key for op, key in aws.calls if op == "s3_upload"]
    assert len(uploads) == 1
    assert uploads[0].startswith("jobs/")
    assert uploads[0].endswith("/input/code.tar")

    # A smoke-test Spend_Line, labelled an estimate, landed in the laptop's spend.jsonl (Req 11.4, 12.2).
    spend_path = tmp_path / "spend.jsonl"
    assert spend_path.exists()
    lines = [json.loads(line) for line in spend_path.read_text(encoding="utf-8").splitlines()
             if line.strip()]
    assert len(lines) == 1
    spend = lines[0]
    assert spend["label"] == LABEL_SMOKE  # "smoke-test"
    assert spend["instance_type"] == cc.smoke_instance_type
    assert spend["estimate"] is True
    assert spend["seconds"] == cc.smoke.expected_minutes * 60

    # The report names the type, the zone, where the machine puts its results, and the Smoke_Marker
    # key it writes on a pass.
    out = capsys.readouterr().out
    assert cc.smoke_instance_type in out
    assert "us-east-1a" in out
    assert smoke_log_key(IMAGE_ID) in out
    assert smoke_marker_key(IMAGE_ID) in out
