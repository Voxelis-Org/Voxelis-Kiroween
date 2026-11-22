"""Shared cloud test helpers: a `FakeAws` over in-memory dicts and a `cloud_config` builder.

`FakeAws` implements the `Aws` protocol without boto3 and with no network, keeping its EC2, S3,
CloudFormation, Service Quotas and SSM state in plain dicts and lists that a test seeds through the
constructor. Every operation appends an `(op, key)` tuple to the ordered `calls` list, so a test can
assert the order AWS was touched in: Property 6, for one, checks that `check_spend` ran before the
first `s3_upload` or `run_instances`. A launch fails with `CapacityUnavailable` or a `CloudError`
only for the `(type, zone)` pairs the test seeds, and otherwise mints a fresh instance.

`cloud_config(**overrides)` builds a valid `CloudConfig` from the real `[cloud]` shape: the 8
machine types, the 3 Presets, the smoke and image plans and the scalars. It seeds no `[cloud.needs]`
table, because a Step's needs live in that Step's own section. Each top-level override replaces its
key, so a test can swap the types, a Preset's fallback, a price or a scalar.

`git_repo(tmp_path)` makes a tiny committed git repo in `tmp_path` for the Start_Command's git
helpers, and skips the test, with a reason, when git isn't installed, the way pipeline-core skips a
test that needs ffprobe.

This module imports no boto3 (it reaches boto3 only indirectly, for the data classes); tests build a
`FakeAws` and never an AWS client.
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from cloud.aws import ImageInfo, InstanceInfo
from cloud.config import CloudConfig
from cloud.errors import CapacityUnavailable, CloudError


class FakeAws:
    """An in-memory `Aws` for tests: seeded dicts, canned replies and an ordered call timeline."""

    def __init__(self, *, region: str = "us-east-1", bucket: str = "voxelis-kiroween-jobs-fake",
                 instances: Sequence[InstanceInfo] | None = None,
                 offerings: Mapping[str, Sequence[str]] | None = None,
                 subnets: Mapping[str, str] | None = None,
                 images: Sequence[ImageInfo] | None = None,
                 console: Mapping[str, str] | None = None, quota: float = 32.0,
                 ssm: Mapping[str, str] | None = None, stack: Mapping[str, str] | None = None,
                 objects: Mapping[str, bytes] | None = None,
                 no_capacity: Sequence[tuple[str, str]] | None = None,
                 unsupported: Sequence[tuple[str, str]] | None = None,
                 launch_failures: Sequence[tuple[str, str]] | None = None) -> None:
        self.region = region
        self.bucket = bucket
        self.calls: list[tuple[str, str]] = []
        self.instances: list[InstanceInfo] = list(instances or ())
        self.offerings: dict[str, list[str]] = {
            name: list(zones) for name, zones in (offerings or {}).items()}
        self.subnets: dict[str, str] = dict(subnets or {})          # zone -> subnet id
        self.images: list[ImageInfo] = list(images or ())
        self.console: dict[str, str] = dict(console or {})
        self.quota = float(quota)
        self.ssm: dict[str, str] = dict(ssm or {})
        self.stack: dict[str, str] = dict(stack or {})
        self.objects: dict[str, bytes] = dict(objects or {})        # S3 key -> bytes
        # A launch for one of these (instance_type, zone) pairs fails instead of succeeding.
        self.no_capacity = set(no_capacity or ())
        self.unsupported = set(unsupported or ())
        self.launch_failures = set(launch_failures or ())
        self._counter = 0

    def _record(self, op: str, key: str = "") -> None:
        self.calls.append((op, key))

    # EC2 reads

    def tagged_instances(self, states: Sequence[str]) -> list[InstanceInfo]:
        self._record("tagged_instances", ",".join(states))
        wanted = set(states)
        return [info for info in self.instances if not wanted or info.state in wanted]

    def type_zones(self, instance_type: str) -> list[str]:
        self._record("type_zones", instance_type)
        return list(self.offerings.get(instance_type, ()))

    def default_subnets(self, vpc_id: str) -> dict[str, str]:
        self._record("default_subnets", vpc_id)
        return dict(self.subnets)

    def tagged_images(self) -> list[ImageInfo]:
        self._record("tagged_images", "")
        return list(self.images)

    def console_output(self, instance_id: str) -> str:
        self._record("console_output", instance_id)
        return self.console.get(instance_id, "")

    def gpu_quota(self) -> float:
        self._record("gpu_quota", "")
        return self.quota

    def ssm_parameter(self, name: str) -> str:
        self._record("ssm_parameter", name)
        try:
            return self.ssm[name]
        except KeyError:
            raise CloudError(f"The fake has no SSM parameter named {name!r}.") from None

    # EC2 writes

    def run_instances(self, request: Mapping[str, Any]) -> InstanceInfo:
        instance_type = str(request.get("InstanceType", ""))
        zone = self._zone_of(request)
        self._record("run_instances", instance_type)
        if (instance_type, zone) in self.no_capacity:
            raise CapacityUnavailable("InsufficientInstanceCapacity", zone, instance_type)
        if (instance_type, zone) in self.unsupported:
            raise CapacityUnavailable("Unsupported", zone, instance_type)
        if (instance_type, zone) in self.launch_failures:
            raise CloudError(f"The fake refused to launch {instance_type} in {zone}.")
        self._counter += 1
        info = InstanceInfo(id=f"i-fake{self._counter:04d}", type=instance_type, state="pending",
                            zone=zone, launch_time="2026-11-20T00:00:00+00:00", state_reason="")
        self.instances.append(info)
        return info

    def stop_instance(self, instance_id: str) -> None:
        self._record("stop_instance", instance_id)
        self.instances = [replace(info, state="stopped") if info.id == instance_id else info
                          for info in self.instances]

    def wait_stopped(self, instance_id: str) -> None:
        self._record("wait_stopped", instance_id)

    def create_image(self, instance_id: str, name: str, tags: Mapping[str, str]) -> str:
        self._record("create_image", name)
        self._counter += 1
        image_id = f"ami-fake{self._counter:04d}"
        self.images.append(ImageInfo(id=image_id, name=name, state="available",
                                     creation_date="2026-11-20T00:00:00+00:00", tags=dict(tags)))
        return image_id

    def wait_image_available(self, image_id: str) -> None:
        self._record("wait_image_available", image_id)

    def terminate_instances(self, instance_ids: Sequence[str]) -> None:
        ids = list(instance_ids)
        self._record("terminate_instances", ",".join(ids))
        gone = set(ids)
        self.instances = [replace(info, state="terminated") if info.id in gone else info
                          for info in self.instances]

    # S3

    def s3_upload(self, local: Path, key: str) -> None:
        self._record("s3_upload", key)
        self.objects[key] = Path(local).read_bytes()

    def s3_download(self, key: str, local: Path) -> None:
        self._record("s3_download", key)
        try:
            data = self.objects[key]
        except KeyError:
            raise CloudError(f"The fake bucket has no object at {key!r}.") from None
        local = Path(local)
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(data)

    def s3_list(self, prefix: str) -> list[str]:
        self._record("s3_list", prefix)
        return sorted(key for key in self.objects if key.startswith(prefix))

    def s3_get_text(self, key: str) -> str:
        self._record("s3_get_text", key)
        try:
            return self.objects[key].decode("utf-8")
        except KeyError:
            raise CloudError(f"The fake bucket has no object at {key!r}.") from None

    def s3_put_text(self, key: str, text: str) -> None:
        self._record("s3_put_text", key)
        self.objects[key] = text.encode("utf-8")

    def s3_exists(self, key: str) -> bool:
        self._record("s3_exists", key)
        return key in self.objects

    # CloudFormation

    def stack_outputs(self, stack_name: str) -> dict[str, str]:
        self._record("stack_outputs", stack_name)
        return dict(self.stack)

    def _zone_of(self, request: Mapping[str, Any]) -> str:
        """The zone a request targets: the subnet's zone from the seeded subnets, else a Placement."""
        subnet = str(request.get("SubnetId", ""))
        for zone, seeded in self.subnets.items():
            if seeded == subnet:
                return zone
        placement = request.get("Placement")
        if isinstance(placement, Mapping) and placement.get("AvailabilityZone"):
            return str(placement["AvailabilityZone"])
        return ""


def cloud_config(**overrides: Any) -> CloudConfig:
    """A valid `CloudConfig` from the real `[cloud]` shape, each top-level override replacing its key."""
    data: dict[str, Any] = _base_cloud_mapping()
    data.update(overrides)
    return CloudConfig.from_mapping(data)


def _base_cloud_mapping() -> dict[str, Any]:
    """The `[cloud]` table as `config.toml` holds it: 8 types, 3 Presets, the plans and the scalars."""
    def spec(vcpus: int, memory_gb: float, gpu_memory_gb: float, price: float) -> dict[str, Any]:
        return {"vcpus": vcpus, "memory_gb": memory_gb, "gpu_memory_gb": gpu_memory_gb,
                "price_per_hour_usd": price, "price_date": "2026-11-21"}

    types = {
        "g5.xlarge": spec(4, 16, 24, 1.006),
        "g5.2xlarge": spec(8, 32, 24, 1.212),
        "g5.4xlarge": spec(16, 64, 24, 1.624),
        "g5.8xlarge": spec(32, 128, 24, 2.448),
        "g6.2xlarge": spec(8, 32, 24, 0.9776),
        "g6.4xlarge": spec(16, 64, 24, 1.3232),
        "g6e.2xlarge": spec(8, 64, 48, 2.24208),
        "g6e.4xlarge": spec(16, 128, 48, 3.00424),
    }
    presets = {
        "fast": {"fallback": ["g5.8xlarge", "g5.4xlarge", "g6.4xlarge", "g5.2xlarge", "g6.2xlarge"],
                 "expected_minutes": 30, "cap_minutes": 120},
        "default": {"fallback": ["g5.8xlarge", "g5.4xlarge", "g6.4xlarge"],
                    "expected_minutes": 60, "cap_minutes": 180},
        "quality": {"fallback": ["g6e.4xlarge", "g6e.2xlarge"],
                    "expected_minutes": 120, "cap_minutes": 360},
    }
    return {
        "types": types,
        "presets": presets,
        "smoke": {"expected_minutes": 10, "cap_minutes": 30, "check_timeout_seconds": 60,
                  "scene_gaussians": 256, "image_pixels": 64, "video_seconds": 4},
        "image": {"expected_minutes": 75, "cap_minutes": 120},
        "upload_margin_minutes": 10,
        "poll_seconds": 20,
        "ipv4_usd_per_hour": 0.005,
        "gp3_usd_per_gb_month": 0.08,
        "root_gb": 150,
        "smoke_instance_type": "g5.xlarge",
        "image_builder_type": "g5.4xlarge",
        "base_ami_parameter":
            "/aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-24.04/latest/ami-id",
    }


def git_repo(tmp_path: Path) -> Path:
    """A tiny committed git repo in `tmp_path`, for the Start_Command's git helpers.

    Skips the test, with a reason, when git isn't installed, the way pipeline-core skips a test that
    needs ffprobe. The repo holds a file under splatpipe/ and under cloud/, a config.toml, a uv.lock
    and a package-lock.json, all committed, so a repo test can dirty one path, archive HEAD, read the
    HEAD sha and hash the lockfiles. It also commits the other dependency manifests the Image_Build
    uploads for its builder: pyproject.toml, .python-version and package.json. Returns the repo's
    path.
    """
    if shutil.which("git") is None:
        pytest.skip("git isn't installed, so this test can't run.")
    root = Path(tmp_path)
    (root / "splatpipe").mkdir()
    (root / "splatpipe" / "run.py").write_text("x = 1\n", encoding="utf-8")
    (root / "cloud").mkdir()
    (root / "cloud" / "start.py").write_text("y = 2\n", encoding="utf-8")
    (root / "config.toml").write_text("[jobs]\n", encoding="utf-8")
    (root / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    (root / "package-lock.json").write_text('{"lockfileVersion": 3}\n', encoding="utf-8")
    (root / "pyproject.toml").write_text('[project]\nname = "test"\n', encoding="utf-8")
    (root / ".python-version").write_text("3.11\n", encoding="utf-8")
    (root / "package.json").write_text('{"private": true}\n', encoding="utf-8")
    # The -c flags keep the commit off the host's git config: a fresh identity, no gpg signing and
    # no line-ending rewrite, so the repo builds the same on every machine.
    base = ["git", "-c", "init.defaultBranch=main", "-c", "core.autocrlf=false",
            "-c", "user.name=Voxelis Test", "-c", "user.email=test@example.invalid",
            "-c", "commit.gpgsign=false"]
    _run_setup_git([*base, "init"], root)
    _run_setup_git([*base, "add", "--", "splatpipe", "cloud", "config.toml", "uv.lock",
                    "package-lock.json", "pyproject.toml", ".python-version", "package.json"], root)
    _run_setup_git([*base, "commit", "-m", "Initial commit"], root)
    return root


def _run_setup_git(command: list[str], cwd: Path) -> None:
    """Run one git command for `git_repo`, raising with git's output when it fails."""
    result = subprocess.run(command, cwd=cwd, capture_output=True, text=True,
                            encoding="utf-8", errors="replace")
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip() or f"exit code {result.returncode}"
        raise RuntimeError(f"couldn't build the test git repo: {detail}")
