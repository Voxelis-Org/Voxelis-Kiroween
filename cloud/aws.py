"""The Fake_AWS seam: the narrow `Aws` protocol, its one real `BotoAws`, and the Session builders.

Every other `cloud` module talks to AWS only through the `Aws` protocol, so this is the one module
that imports boto3. `make_session` is the only place a `boto3.Session` is built, always with the
region, and the profile named in code when there is one, never through `AWS_PROFILE` or the default
credential chain (Req 1.6, 1.7). `BotoAws` is the one real implementation: it makes its five service
clients from that one Session, turns a capacity error on a launch into `CapacityUnavailable` and
every other AWS error into a `CloudError` carrying `account_redaction(str(error))` so no raw account
id ever reaches a chat, and filters every describe by the Project_Tag so Cloud_Jobs only ever sees
our own resources (Req 1.8, 1.9, 5.5). Tests use the `FakeAws` in `tests/fake_aws.py`, which
implements the same protocol over in-memory dicts.

`laptop_aws` names the region and a profile from `[aws]` for a laptop command; `machine_aws` names
only the region, so the job machine uses its instance role. A cloud module may import `splatpipe`;
`splatpipe` never imports `cloud`. This module imports no GPU package.
"""

from __future__ import annotations

import base64
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from splatpipe.config import Config

from cloud.errors import CapacityUnavailable, CloudError
from cloud.redact import account_redaction

# The repo's real config.toml, read for the Project_Tag. machine_aws has no Config object, so it
# reads the tag here, as cloud.spend reads [aws].spend_variable from the file.
CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.toml"
# Service Quotas code for "Running On-Demand G and VT instances", the G-instance vCPU quota.
GPU_QUOTA_CODE = "L-DB2E81BA"
# A launch ClientError with one of these codes means no On-Demand capacity, not a real failure.
_CAPACITY_CODES = frozenset({"InsufficientInstanceCapacity", "Unsupported"})


@dataclass(frozen=True)
class InstanceInfo:
    """One EC2 instance, the fields Cloud_Jobs reads, as a describe or a launch returns them."""

    id: str
    type: str
    state: str                  # "pending", "running", "shutting-down", "stopping", "stopped", "terminated"
    zone: str
    launch_time: str            # ISO-8601, as AWS returns it
    state_reason: str


@dataclass(frozen=True)
class ImageInfo:
    """One of our Machine_Images, the fields the Start_Command and Image_Build read."""

    id: str
    name: str
    state: str                  # "available", "pending", ...
    creation_date: str
    tags: dict[str, str]


class Aws(Protocol):
    """The one AWS interface every other `cloud` module depends on, over EC2, S3, CloudFormation,
    Service Quotas and SSM. `BotoAws` is the real implementation; tests pass a `FakeAws`."""

    region: str

    # EC2 reads
    def tagged_instances(self, states: Sequence[str]) -> list[InstanceInfo]: ...
    def type_zones(self, instance_type: str) -> list[str]: ...
    def default_subnets(self, vpc_id: str) -> dict[str, str]: ...        # zone -> subnet id
    def tagged_images(self) -> list[ImageInfo]: ...
    def console_output(self, instance_id: str) -> str: ...
    def gpu_quota(self) -> float: ...
    def ssm_parameter(self, name: str) -> str: ...

    # EC2 writes
    def run_instances(self, request: Mapping[str, Any]) -> InstanceInfo: ...
    def stop_instance(self, instance_id: str) -> None: ...
    def wait_stopped(self, instance_id: str) -> None: ...
    def create_image(self, instance_id: str, name: str, tags: Mapping[str, str]) -> str: ...
    def wait_image_available(self, image_id: str) -> None: ...
    def terminate_instances(self, instance_ids: Sequence[str]) -> None: ...

    # S3
    def s3_upload(self, local: Path, key: str) -> None: ...
    def s3_download(self, key: str, local: Path) -> None: ...
    def s3_list(self, prefix: str) -> list[str]: ...
    def s3_get_text(self, key: str) -> str: ...
    def s3_put_text(self, key: str, text: str) -> None: ...
    def s3_exists(self, key: str) -> bool: ...

    # CloudFormation
    def stack_outputs(self, stack_name: str) -> dict[str, str]: ...


def make_session(region: str, profile: str | None) -> boto3.Session:
    """Build the one `boto3.Session`: the region always, the profile named only when there is one.

    This is the only place a Session is made. A named profile beats `AWS_PROFILE` and the default
    chain, so no laptop command can run as the wrong account (Req 1.6).
    """
    if profile is None:
        return boto3.Session(region_name=region)
    return boto3.Session(region_name=region, profile_name=profile)


def laptop_aws(config: Config, *, profile_key: str) -> BotoAws:
    """A `BotoAws` for a laptop command, naming the region and a profile from `[aws]` in `config`.

    `profile_key` is `"read_profile"` or `"runner_profile"`. The region and the profile come from
    the Config the caller already loaded; `[aws]` is not a Pipeline_Section and has no accessor on
    `splatpipe.config.Config`, and this spec adds none, so its parsed table is read directly here.
    """
    aws = _aws_table(config)
    region = _required(aws, "region")
    profile = _required(aws, profile_key)
    return BotoAws(make_session(region, profile), region=region)


def machine_aws(region: str) -> BotoAws:
    """A `BotoAws` whose Session names only the region, so the job machine uses its instance role."""
    return BotoAws(make_session(region, None), region=region)


class BotoAws:
    """The one real `Aws`: five service clients from a single Session, errors made readable and
    account-free, every describe filtered to our own resources by the Project_Tag."""

    def __init__(self, session: boto3.Session, *, region: str, bucket: str | None = None) -> None:
        self.region = region
        # The Job_Bucket is a Stack_Output, unknown at build time; the caller sets it after
        # resolve_stack, before any S3 call. _bucket() refuses an S3 call while it is unset.
        self.bucket = bucket
        self._ec2 = session.client("ec2")
        self._s3 = session.client("s3")
        self._cfn = session.client("cloudformation")
        self._quotas = session.client("service-quotas")
        self._ssm = session.client("ssm")
        self._project_tag = _project_tag()

    # EC2 reads

    def tagged_instances(self, states: Sequence[str]) -> list[InstanceInfo]:
        key, value = self._project_tag
        filters: list[dict[str, Any]] = [{"Name": f"tag:{key}", "Values": [value]}]
        wanted = list(states)
        if wanted:
            filters.append({"Name": "instance-state-name", "Values": wanted})
        result = self._call(lambda: self._ec2.describe_instances(Filters=filters))
        return [_as_instance(raw)
                for reservation in result.get("Reservations", [])
                for raw in reservation.get("Instances", [])]

    def type_zones(self, instance_type: str) -> list[str]:
        result = self._call(lambda: self._ec2.describe_instance_type_offerings(
            LocationType="availability-zone",
            Filters=[{"Name": "instance-type", "Values": [instance_type]}]))
        zones = (str(offering.get("Location", ""))
                 for offering in result.get("InstanceTypeOfferings", []))
        return sorted(zone for zone in zones if zone)

    def default_subnets(self, vpc_id: str) -> dict[str, str]:
        result = self._call(lambda: self._ec2.describe_subnets(Filters=[
            {"Name": "vpc-id", "Values": [vpc_id]},
            {"Name": "default-for-az", "Values": ["true"]}]))
        subnets: dict[str, str] = {}
        for raw in result.get("Subnets", []):
            zone = str(raw.get("AvailabilityZone", ""))
            subnet = str(raw.get("SubnetId", ""))
            if zone and subnet:
                subnets[zone] = subnet
        return subnets

    def tagged_images(self) -> list[ImageInfo]:
        key, value = self._project_tag
        result = self._call(lambda: self._ec2.describe_images(
            Owners=["self"], Filters=[{"Name": f"tag:{key}", "Values": [value]}]))
        return [_as_image(raw) for raw in result.get("Images", [])]

    def console_output(self, instance_id: str) -> str:
        # Latest=True asks for the newest output, which our Nitro machines support. Without it EC2
        # only posts output around state changes, so a line printed long after boot, like the image
        # build's done marker, would never show up.
        result = self._call(lambda: self._ec2.get_console_output(
            InstanceId=instance_id, Latest=True))
        output = result.get("Output") or ""
        try:  # EC2 returns the console output base64-encoded.
            return base64.b64decode(output).decode("utf-8", "replace")
        except (ValueError, TypeError):
            return str(output)

    def gpu_quota(self) -> float:
        result = self._call(lambda: self._quotas.get_service_quota(
            ServiceCode="ec2", QuotaCode=GPU_QUOTA_CODE))
        return float(result.get("Quota", {}).get("Value", 0.0))

    def ssm_parameter(self, name: str) -> str:
        result = self._call(lambda: self._ssm.get_parameter(Name=name))
        return str(result.get("Parameter", {}).get("Value", ""))

    # EC2 writes

    def run_instances(self, request: Mapping[str, Any]) -> InstanceInfo:
        payload = dict(request)
        try:
            result = self._ec2.run_instances(**payload)
        except ClientError as error:
            code = str(error.response.get("Error", {}).get("Code", ""))
            if code in _CAPACITY_CODES:  # no capacity: the launch loop tries the next attempt
                raise CapacityUnavailable(
                    code, _request_zone(payload), str(payload.get("InstanceType", ""))) from error
            raise CloudError(account_redaction(str(error))) from error
        except BotoCoreError as error:
            raise CloudError(account_redaction(str(error))) from error
        instances = result.get("Instances") or []
        if not instances:
            raise CloudError("AWS accepted the launch but returned no instance. Check the machine "
                             "list, and if none started, run the command again.")
        return _as_instance(instances[0])

    def stop_instance(self, instance_id: str) -> None:
        self._call(lambda: self._ec2.stop_instances(InstanceIds=[instance_id]))

    def wait_stopped(self, instance_id: str) -> None:
        self._call(lambda: self._ec2.get_waiter("instance_stopped").wait(
            InstanceIds=[instance_id]))

    def create_image(self, instance_id: str, name: str, tags: Mapping[str, str]) -> str:
        tag_list = [{"Key": key, "Value": value} for key, value in dict(tags).items()]
        kwargs: dict[str, Any] = {"InstanceId": instance_id, "Name": name}
        if tag_list:  # tag the image and its snapshots in the one call, so both carry the tags
            kwargs["TagSpecifications"] = [{"ResourceType": "image", "Tags": tag_list},
                                           {"ResourceType": "snapshot", "Tags": tag_list}]
        result = self._call(lambda: self._ec2.create_image(**kwargs))
        return str(result.get("ImageId", ""))

    def wait_image_available(self, image_id: str) -> None:
        self._call(lambda: self._ec2.get_waiter("image_available").wait(ImageIds=[image_id]))

    def terminate_instances(self, instance_ids: Sequence[str]) -> None:
        ids = list(instance_ids)
        if ids:
            self._call(lambda: self._ec2.terminate_instances(InstanceIds=ids))

    # S3

    def s3_upload(self, local: Path, key: str) -> None:
        self._call(lambda: self._s3.upload_file(str(local), self._bucket(), key))

    def s3_download(self, key: str, local: Path) -> None:
        local = Path(local)
        local.parent.mkdir(parents=True, exist_ok=True)
        self._call(lambda: self._s3.download_file(self._bucket(), key, str(local)))

    def s3_list(self, prefix: str) -> list[str]:
        keys: list[str] = []

        def run() -> None:
            paginator = self._s3.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=self._bucket(), Prefix=prefix):
                keys.extend(str(obj.get("Key", "")) for obj in page.get("Contents", []))

        self._call(run)
        return keys

    def s3_get_text(self, key: str) -> str:
        def run() -> str:
            result = self._s3.get_object(Bucket=self._bucket(), Key=key)
            return result["Body"].read().decode("utf-8")

        return self._call(run)

    def s3_put_text(self, key: str, text: str) -> None:
        self._call(lambda: self._s3.put_object(
            Bucket=self._bucket(), Key=key, Body=text.encode("utf-8")))

    def s3_exists(self, key: str) -> bool:
        try:
            self._s3.head_object(Bucket=self._bucket(), Key=key)
            return True
        except ClientError as error:
            code = str(error.response.get("Error", {}).get("Code", ""))
            if code in ("404", "NoSuchKey", "NotFound"):
                return False
            raise CloudError(account_redaction(str(error))) from error
        except BotoCoreError as error:
            raise CloudError(account_redaction(str(error))) from error

    # CloudFormation

    def stack_outputs(self, stack_name: str) -> dict[str, str]:
        result = self._call(lambda: self._cfn.describe_stacks(StackName=stack_name))
        stacks = result.get("Stacks") or []
        if not stacks:
            raise CloudError(f"No CloudFormation stack named {stack_name!r} was found. Deploy "
                             "cloud/aws-setup.yaml first, then run the command again.")
        return {str(output.get("OutputKey", "")): str(output.get("OutputValue", ""))
                for output in stacks[0].get("Outputs", [])}

    # Helpers

    def _bucket(self) -> str:
        """The Job_Bucket, or a CloudError when the caller hasn't set it from the Stack yet."""
        if not self.bucket:
            raise CloudError("No Job_Bucket is set on this AWS client. Resolve the Stack and set "
                             "its bucket before any S3 call. This is a bug if a user sees it.")
        return self.bucket

    def _call(self, thunk: Any) -> Any:
        """Run one boto3 call, turning any AWS error into a readable, account-free CloudError."""
        try:
            return thunk()
        except (ClientError, BotoCoreError) as error:
            raise CloudError(account_redaction(str(error))) from error


# Turning boto3's dicts into our small data classes

def _as_instance(raw: Mapping[str, Any]) -> InstanceInfo:
    """One instance dict from a describe or a launch as an InstanceInfo."""
    state = raw.get("State") or {}
    placement = raw.get("Placement") or {}
    reason = raw.get("StateReason") or {}
    launch = raw.get("LaunchTime")
    return InstanceInfo(
        id=str(raw.get("InstanceId", "")),
        type=str(raw.get("InstanceType", "")),
        state=str(state.get("Name", "")),
        zone=str(placement.get("AvailabilityZone", "")),
        launch_time=launch.isoformat() if hasattr(launch, "isoformat") else str(launch or ""),
        state_reason=str(reason.get("Message") or raw.get("StateTransitionReason") or ""),
    )


def _as_image(raw: Mapping[str, Any]) -> ImageInfo:
    """One image dict from describe-images as an ImageInfo, its tag list flattened to a dict."""
    tags = {str(tag.get("Key", "")): str(tag.get("Value", "")) for tag in (raw.get("Tags") or [])}
    return ImageInfo(
        id=str(raw.get("ImageId", "")),
        name=str(raw.get("Name", "")),
        state=str(raw.get("State", "")),
        creation_date=str(raw.get("CreationDate", "")),
        tags=tags,
    )


def _request_zone(request: Mapping[str, Any]) -> str:
    """The zone a launch request targets, for a CapacityUnavailable: its Placement if it names one,
    else its subnet as the stand-in. The launch loop knows the real zone and uses its own."""
    placement = request.get("Placement")
    if isinstance(placement, Mapping) and placement.get("AvailabilityZone"):
        return str(placement["AvailabilityZone"])
    return str(request.get("SubnetId", ""))


# Reading [aws] from config

def _aws_table(config: Config) -> Mapping[str, Any]:
    """The parsed `[aws]` table from a loaded Config, or a CloudError when it is missing."""
    data = getattr(config, "_data", None)
    aws = data.get("aws") if isinstance(data, Mapping) else None
    if not isinstance(aws, Mapping):
        raise CloudError("config.toml has no [aws] section. Add [aws] with the region and the "
                         "profile names, then run the command again.")
    return aws


def _required(aws: Mapping[str, Any], key: str) -> str:
    """A non-empty string for `key` in `[aws]`, or a CloudError naming it."""
    value = aws.get(key)
    if not isinstance(value, str) or not value:
        raise CloudError(f"[aws] in config.toml needs {key} as a non-empty string. Add it, then "
                         "run the command again.")
    return value


def _project_tag(config_path: Path = CONFIG_PATH) -> tuple[str, str]:
    """The Project_Tag from `[aws].project_tag`, split into (key, value) for a tag filter."""
    try:
        with open(config_path, "rb") as handle:
            data = tomllib.load(handle)
    except (OSError, ValueError) as error:
        raise CloudError(f"config.toml can't be read for the project tag ({error}). Get it back "
                         "from git, then run the command again.") from error
    aws = data.get("aws")
    raw = aws.get("project_tag") if isinstance(aws, Mapping) else None
    if not isinstance(raw, str) or raw.count("=") != 1 or "" in raw.split("="):
        raise CloudError('[aws] in config.toml needs project_tag as "key=value" text, like '
                         '"project=voxelis-kiroween". Fix it there, then run the command again.')
    key, value = raw.split("=", 1)
    return key, value
