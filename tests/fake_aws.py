"""Shared cloud test helpers: a `FakeAws` over in-memory dicts and a `cloud_config` builder.

`FakeAws` implements the `Aws` protocol without boto3 and with no network, keeping its EC2, S3,
CloudFormation, Service Quotas and SSM state in plain dicts and lists that a test seeds through the
constructor. Every operation appends an `(op, key)` tuple to the ordered `calls` list, so a test can
assert the order AWS was touched in: Property 6, for one, checks that `check_spend` ran before the
first `s3_upload` or `run_instances`. A launch fails with `CapacityUnavailable` or a `CloudError`
only for the `(type, zone)` pairs the test seeds, and otherwise mints a fresh instance.

`cloud_config(**overrides)` builds a valid `CloudConfig` from the real `[cloud]` shape: the 8
machine types, the 3 Presets, the smoke plan, the image plan with its builder types (g5.8xlarge,
then g5.4xlarge) and the scalars. It seeds no `[cloud.needs]`
table, because a Step's needs live in that Step's own section. Each top-level override replaces its
key, so a test can swap the types, a Preset's fallback, a price or a scalar.

`git_repo(tmp_path)` makes a tiny committed git repo in `tmp_path` for the Start_Command's git
helpers, and skips the test, with a reason, when git isn't installed, the way pipeline-core skips a
test that needs ffprobe.

This module imports no boto3 (it reaches boto3 only indirectly, for the data classes); tests build a
`FakeAws` and never an AWS client.
"""

from __future__ import annotations

import copy
import hashlib
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from cloud.aws import ImageInfo, InstanceInfo, s3_origin
from cloud.config import CloudConfig
from cloud.errors import CapacityUnavailable, CloudError, QuotaRefused
from cloud.table import (Change, Check, Condition, ConditionFailed, Key, KeyRange, Prefix, Put,
                         Scalar, TransactionConflict, TransactItem, Update, item_key, transact_key)
from tests.fake_table import apply, holds


@dataclass
class FakeMultipart:
    """One multipart upload the fake holds: its key and the parts sent so far."""

    key: str
    parts: dict[int, tuple[str, bytes]] = field(default_factory=dict)   # number -> (ETag, bytes)


@dataclass(frozen=True)
class FakeLink:
    """One Presigned_Link the fake made."""

    kind: str            # "part" or "get"
    key: str
    upload_id: str
    part: int
    expires: datetime
    filename: str | None


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
                 launch_failures: Sequence[tuple[str, str]] | None = None,
                 quota_refused: Sequence[tuple[str, str]] | None = None,
                 table: str | None = "voxelis-kiroween-website-fake") -> None:
        self.region = region
        self.bucket = bucket
        self.table = table
        # The Website_Table: (pk, sk) -> item.
        self.items: dict[tuple[str, str], dict[str, Scalar]] = {}
        self.conflicts = 0                      # the next n transactions raise TransactionConflict
        self.tokens: set[str] = set()           # ClientRequestTokens already applied
        self.multipart: dict[str, FakeMultipart] = {}
        self.links: dict[str, FakeLink] = {}
        self.object_sizes: dict[str, int] = {}  # claimed sizes, beating len(objects[key])
        self.now = datetime(2026, 12, 1, 12, 0, tzinfo=timezone.utc)   # the clock links expire by
        self.instance_jobs: dict[str, str] = {}  # instance id -> job_id tag
        self.job_machines_error = False
        self.copy_failures: set[str] = set()     # source keys whose copy fails
        self.upload_failures: set[str] = set()   # keys whose s3_upload fails
        self.change_sets: list[dict[str, Any]] = []
        self.changes: list[tuple[str, str, str]] = []
        self.deleted_change_sets: list[tuple[str, str]] = []
        self.quota_refused = set(quota_refused or ())
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
        # Each wait_image_available call's (image_id, poll_seconds, max_minutes).
        self.image_waits: list[tuple[str, float, float]] = []
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
        if (instance_type, zone) in self.quota_refused:
            raise QuotaRefused(f"VcpuLimitExceeded: the fake's vCPU limit for {instance_type}.")
        self._counter += 1
        info = InstanceInfo(id=f"i-fake{self._counter:04d}", type=instance_type, state="pending",
                            zone=zone, launch_time="2026-11-20T00:00:00+00:00", state_reason="")
        self.instances.append(info)
        job_id = _job_tag(request)
        if job_id is not None:
            self.instance_jobs[info.id] = job_id
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

    def wait_image_available(self, image_id: str, *, poll_seconds: float,
                             max_minutes: float) -> None:
        self._record("wait_image_available", image_id)
        self.image_waits.append((image_id, poll_seconds, max_minutes))

    def terminate_instances(self, instance_ids: Sequence[str]) -> None:
        ids = list(instance_ids)
        self._record("terminate_instances", ",".join(ids))
        gone = set(ids)
        self.instances = [replace(info, state="terminated") if info.id in gone else info
                          for info in self.instances]

    # S3

    def s3_upload(self, local: Path, key: str) -> None:
        self._record("s3_upload", key)
        if key in self.upload_failures:
            raise CloudError(f"The fake refused to upload {key!r}.")
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

    # DynamoDB, the Website_Table

    def table_get(self, key: Key) -> dict[str, Scalar] | None:
        self._record("table_get", _shown(key))
        self._need_table()
        item = self.items.get((key.pk, key.sk))
        return copy.deepcopy(item) if item is not None else None

    def table_get_many(self, keys: Sequence[Key]) -> list[dict[str, Scalar] | None]:
        wanted = list(keys)
        self._record("table_get_many", ",".join(_shown(key) for key in wanted))
        self._need_table()
        return [copy.deepcopy(self.items.get((key.pk, key.sk))) for key in wanted]

    def table_query(self, pk: str, sk: KeyRange | None = None) -> list[dict[str, Scalar]]:
        self._record("table_query", pk)
        self._need_table()
        if sk is None:
            def inside(value: str) -> bool:
                return True
        elif isinstance(sk, Prefix):
            def inside(value: str) -> bool:
                return value.startswith(sk.value)
        else:
            def inside(value: str) -> bool:
                return sk.low <= value <= sk.high
        return [copy.deepcopy(item) for (p, s), item in sorted(self.items.items())
                if p == pk and inside(s)]

    def table_put(self, item: Mapping[str, Scalar], condition: Condition | None = None) -> None:
        key = item_key(item)
        self._record("table_put", _shown(key))
        self._need_table()
        if not holds(condition, self.items.get((key.pk, key.sk))):
            raise ConditionFailed(None)
        self.items[(key.pk, key.sk)] = dict(item)

    def table_update(self, key: Key, updates: Sequence[Update],
                     condition: Condition | None = None) -> None:
        self._record("table_update", _shown(key))
        self._need_table()
        current = self.items.get((key.pk, key.sk))
        if not holds(condition, current):
            raise ConditionFailed(None)
        self.items[(key.pk, key.sk)] = apply(updates, current, {"pk": key.pk, "sk": key.sk})

    def table_transact(self, items: Sequence[TransactItem], token: str | None = None) -> None:
        entries = list(items)
        self._record("table_transact", token or "")
        self._need_table()
        if token is not None and token in self.tokens:
            return  # DynamoDB applies one token's transaction once
        keys = [transact_key(entry) for entry in entries]
        if len(set(keys)) != len(keys):
            raise CloudError("ValidationException: a transaction can't hold two items on one key.")
        if self.conflicts > 0:
            self.conflicts -= 1
            raise TransactionConflict()
        for index, (entry, key) in enumerate(zip(entries, keys)):
            if not holds(entry.condition, self.items.get((key.pk, key.sk))):
                raise ConditionFailed(index)
        for entry, key in zip(entries, keys):
            if isinstance(entry, Put):
                self.items[(key.pk, key.sk)] = dict(entry.item)
            elif isinstance(entry, Change):
                self.items[(key.pk, key.sk)] = apply(entry.updates, self.items.get((key.pk, key.sk)),
                                                     {"pk": key.pk, "sk": key.sk})
        if token is not None:
            self.tokens.add(token)

    def table_delete(self, key: Key) -> None:
        self._record("table_delete", _shown(key))
        self._need_table()
        self.items.pop((key.pk, key.sk), None)

    # EC2, for "Ended without a result"

    def job_machines(self, job_id: str) -> list[InstanceInfo]:
        self._record("job_machines", job_id)
        if self.job_machines_error:
            raise CloudError("The fake's machine lookup failed.")
        return [info for info in self.instances if self.instance_jobs.get(info.id) == job_id]

    # S3, for the website

    def s3_start_upload(self, key: str) -> str:
        self._record("s3_start_upload", key)
        self._counter += 1
        upload_id = f"fake-upload-{self._counter:04d}"
        self.multipart[upload_id] = FakeMultipart(key=key)
        return upload_id

    def s3_part_link(self, key: str, upload_id: str, part: int, seconds: int) -> str:
        self._record("s3_part_link", key)
        return self._link(FakeLink("part", key, upload_id, part,
                                   self.now + timedelta(seconds=seconds), None))

    def s3_complete_upload(self, key: str, upload_id: str,
                           parts: Sequence[tuple[int, str]]) -> None:
        self._record("s3_complete_upload", key)
        upload = self.multipart.get(upload_id)
        if upload is None or upload.key != key:
            raise CloudError("NoSuchUpload: the fake has no such multipart upload.")
        chunks = []
        for number, etag in parts:
            sent = upload.parts.get(number)
            if sent is None or sent[0] != etag:
                raise CloudError(f"InvalidPart: part {number} wasn't sent with that ETag.")
            chunks.append(sent[1])
        self.objects[key] = b"".join(chunks)
        del self.multipart[upload_id]

    def s3_abort_upload(self, key: str, upload_id: str) -> None:
        self._record("s3_abort_upload", key)
        self.multipart.pop(upload_id, None)

    def s3_download_link(self, key: str, filename: str | None, seconds: int) -> str:
        self._record("s3_download_link", key)
        return self._link(FakeLink("get", key, "", 0, self.now + timedelta(seconds=seconds),
                                   filename))

    def s3_copy(self, source_key: str, key: str, *, part_bytes: int) -> None:
        self._record("s3_copy", key)
        if source_key in self.copy_failures or source_key not in self.objects:
            raise CloudError(f"The fake couldn't copy {source_key!r}.")
        self.objects[key] = self.objects[source_key]
        if source_key in self.object_sizes:
            self.object_sizes[key] = self.object_sizes[source_key]

    def s3_size(self, key: str) -> int | None:
        self._record("s3_size", key)
        if key not in self.objects:
            return None
        return self.object_sizes.get(key, len(self.objects[key]))

    def s3_read_range(self, key: str, offset: int, length: int) -> bytes:
        self._record("s3_read_range", key)
        try:
            data = self.objects[key]
        except KeyError:
            raise CloudError(f"The fake bucket has no object at {key!r}.") from None
        return data[offset:offset + length]

    # The tests' browser

    def put_part(self, url: str, data: bytes) -> tuple[int, str | None]:
        """A browser's PUT of one part: (200, its ETag), or (403, None) for an expired or unknown
        link or a finished upload."""
        link = self.links.get(url)
        if link is None or link.kind != "part" or self.now >= link.expires:
            return 403, None
        upload = self.multipart.get(link.upload_id)
        if upload is None:
            return 403, None
        etag = '"' + hashlib.md5(data).hexdigest() + '"'
        upload.parts[link.part] = (etag, data)
        return 200, etag

    def follow_link(self, url: str) -> tuple[int, bytes]:
        """A browser's GET of a Download_Link: (200, the bytes), or (403, b"")."""
        link = self.links.get(url)
        if link is None or link.kind != "get" or self.now >= link.expires:
            return 403, b""
        if link.key not in self.objects:
            return 404, b""
        return 200, self.objects[link.key]

    # CloudFormation, for the Deploy_Command

    def create_change_set(self, request: Mapping[str, Any]) -> None:
        self._record("create_change_set", str(request.get("ChangeSetName", "")))
        self.change_sets.append(copy.deepcopy(dict(request)))

    def change_set_changes(self, stack_name: str,
                           change_set_name: str) -> list[tuple[str, str, str]]:
        self._record("change_set_changes", change_set_name)
        return list(self.changes)

    def delete_change_set(self, stack_name: str, change_set_name: str) -> None:
        self._record("delete_change_set", change_set_name)
        self.deleted_change_sets.append((stack_name, change_set_name))

    def _link(self, link: FakeLink) -> str:
        self._counter += 1
        url = (f"{s3_origin(self.bucket, self.region)}/{link.key}"
               f"?X-Amz-Signature=fake{self._counter:06d}&X-Amz-Expires=1")
        self.links[url] = link
        return url

    def _need_table(self) -> None:
        if not self.table:
            raise CloudError("No website table is set on the fake.")

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


def _shown(key: Key) -> str:
    return f"{key.pk}|{key.sk}"


def _job_tag(request: Mapping[str, Any]) -> str | None:
    """The job_id tag on a launch request's instance, or None."""
    for spec in request.get("TagSpecifications") or []:
        if spec.get("ResourceType") == "instance":
            for tag in spec.get("Tags") or []:
                if tag.get("Key") == "job_id":
                    return str(tag.get("Value"))
    return None


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
                  "scene_gaussians": 256, "image_pixels": 64, "video_seconds": 4,
                  "train_images": 4, "train_points": 200, "train_splats": 400, "train_steps": 30,
                  "train_refine_start_step": 10, "train_refine_every": 10,
                  "train_sh_degree_interval": 5, "depth_normal_start_step": 8,
                  "cameras_frames": 30, "cameras_width": 320, "cameras_height": 240,
                  "cameras_focal_pixels": 260.0, "cameras_placed_share": 0.8,
                  "cameras_focal_tolerance": 0.02, "cameras_timeout_seconds": 600},
        "image": {"fallback": ["g5.8xlarge", "g5.4xlarge"], "expected_minutes": 75,
                  "cap_minutes": 120},
        "upload_margin_minutes": 10,
        "poll_seconds": 20,
        "ipv4_usd_per_hour": 0.005,
        "gp3_usd_per_gb_month": 0.08,
        "root_gb": 150,
        "smoke_instance_type": "g5.xlarge",
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
