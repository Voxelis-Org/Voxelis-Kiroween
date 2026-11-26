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
only the region, so the job machine uses its instance role, and `job_machine_aws` builds that one
with the region from `[aws]` and the Job_Bucket from the BUCKET the User_Data exports. A cloud
module may import `splatpipe`; `splatpipe` never imports `cloud`. This module imports no GPU
package.
"""

from __future__ import annotations

import base64
import math
import os
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from splatpipe.config import Config

from cloud.errors import CapacityUnavailable, CloudError, QuotaRefused
from cloud.redact import account_redaction
from cloud.table import (And, Between, Change, Check, Condition, ConditionFailed, Equals, Exists,
                         Key, KeyRange, NotExists, Or, Prefix, Put, Remove, Scalar, Set,
                         TransactionConflict, TransactItem, Update, AtLeast, AtMost, Add)

# The repo's real config.toml, read for the Project_Tag. machine_aws has no Config object, so it
# reads the tag here, as cloud.spend reads [aws].spend_variable from the file.
CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.toml"
# Service Quotas code for "Running On-Demand G and VT instances", the G-instance vCPU quota.
GPU_QUOTA_CODE = "L-DB2E81BA"
# The variable the User_Data exports with the Job_Bucket's name, which job_machine_aws reads.
BUCKET_ENV = "BUCKET"
# A launch ClientError with one of these codes means no On-Demand capacity, not a real failure.
_CAPACITY_CODES = frozenset({"InsufficientInstanceCapacity", "Unsupported"})
# A launch ClientError with one of these codes means EC2's vCPU or instance limit refused it.
_QUOTA_CODES = frozenset({"VcpuLimitExceeded", "InstanceLimitExceeded"})
# The tag a Job machine carries with its Job_Id (cloud.launch.JOB_ID_TAG, kept here to avoid a cycle).
_JOB_ID_TAG = "job_id"


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
    bucket: str | None
    table: str | None

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
    def wait_image_available(self, image_id: str, *, poll_seconds: float,
                             max_minutes: float) -> None: ...
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

    # DynamoDB, the Website_Table
    def table_get(self, key: Key) -> dict[str, Scalar] | None: ...
    def table_get_many(self, keys: Sequence[Key]) -> list[dict[str, Scalar] | None]: ...
    def table_query(self, pk: str, sk: KeyRange) -> list[dict[str, Scalar]]: ...
    def table_put(self, item: Mapping[str, Scalar], condition: Condition | None = None) -> None: ...
    def table_update(self, key: Key, updates: Sequence[Update],
                     condition: Condition | None = None) -> None: ...
    def table_transact(self, items: Sequence[TransactItem], token: str | None = None) -> None: ...
    def table_delete(self, key: Key) -> None: ...

    # EC2, for "Ended without a result"
    def job_machines(self, job_id: str) -> list[InstanceInfo]: ...

    # S3, for the website
    def s3_start_upload(self, key: str) -> str: ...
    def s3_part_link(self, key: str, upload_id: str, part: int, seconds: int) -> str: ...
    def s3_complete_upload(self, key: str, upload_id: str,
                           parts: Sequence[tuple[int, str]]) -> None: ...
    def s3_abort_upload(self, key: str, upload_id: str) -> None: ...
    def s3_download_link(self, key: str, filename: str | None, seconds: int) -> str: ...
    def s3_copy(self, source_key: str, key: str, *, part_bytes: int) -> None: ...
    def s3_size(self, key: str) -> int | None: ...
    def s3_read_range(self, key: str, offset: int, length: int) -> bytes: ...

    # CloudFormation, for the Deploy_Command
    def create_change_set(self, request: Mapping[str, Any]) -> None: ...
    def change_set_changes(self, stack_name: str,
                           change_set_name: str) -> list[tuple[str, str, str]]: ...
    def delete_change_set(self, stack_name: str, change_set_name: str) -> None: ...


def s3_origin(bucket: str, region: str) -> str:
    """https://<bucket>.s3.<region>.amazonaws.com, the host of every Presigned_Link and the
    Job_Bucket's entry in the website's Content-Security-Policy."""
    return f"https://{bucket}.s3.{region}.amazonaws.com"


def website_aws(region: str) -> BotoAws:
    """A `BotoAws` naming only the region, so the website function runs as the Website_Role."""
    return BotoAws(make_session(region, None), region=region)


def cloudshell_aws(region: str) -> BotoAws:
    """A `BotoAws` naming only the region, so a CloudShell_Command runs as the Admin_Login."""
    return BotoAws(make_session(region, None), region=region)


def aws_setting(config: Config, key: str) -> str:
    """A non-empty string for `key` in the parsed `[aws]` table, or a CloudError naming it."""
    return _required(_aws_table(config), key)


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


def job_machine_aws(config: Config, environ: Mapping[str, str] | None = None) -> BotoAws:
    """The `BotoAws` a job or smoke machine runs on: its instance role, with the Job_Bucket set.

    The region comes from `[aws].region` in `config`, as on a laptop (the aws-jobs skill), and no
    profile is named, so the instance role is used. The machine can't read the Stack, so the
    Job_Bucket comes from the BUCKET variable the User_Data exports. `environ` defaults to the real
    environment; tests pass their own.
    """
    env = os.environ if environ is None else environ
    bucket = env.get(BUCKET_ENV, "")
    if not bucket:
        raise CloudError(f"The {BUCKET_ENV} environment variable isn't set, so this machine doesn't "
                         "know its job bucket. The machine's user data sets it; this command runs "
                         "only there, started by cloud.start or cloud.smoke.")
    aws = machine_aws(_required(_aws_table(config), "region"))
    aws.bucket = bucket
    return aws


class BotoAws:
    """The one real `Aws`: five service clients from a single Session, errors made readable and
    account-free, every describe filtered to our own resources by the Project_Tag."""

    def __init__(self, session: boto3.Session, *, region: str, bucket: str | None = None) -> None:
        self.region = region
        # The Job_Bucket is a Stack_Output, unknown at build time; the caller sets it after
        # resolve_stack, before any S3 call. _bucket() refuses an S3 call while it is unset.
        self.bucket = bucket
        # The Website_Table's name, set by the caller before the first table call.
        self.table: str | None = None
        self._session = session
        self._links: Any = None   # the presigning S3 client, built on the first link
        self._ec2 = session.client("ec2")
        self._s3 = session.client("s3")
        self._cfn = session.client("cloudformation")
        self._quotas = session.client("service-quotas")
        self._ssm = session.client("ssm")
        self._dynamodb = session.client("dynamodb")
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
            if code in _QUOTA_CODES:  # EC2's own limit: no machine started, and the loop stops
                raise QuotaRefused(account_redaction(str(error))) from error
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

    def wait_image_available(self, image_id: str, *, poll_seconds: float,
                             max_minutes: float) -> None:
        # botocore's waiter gives up after 10 minutes unless told otherwise, but an image's first
        # snapshots can take much longer (the aws-jobs skill), so the caller sets the poll and limit.
        delay = max(1, round(poll_seconds))
        attempts = max(1, math.ceil(max_minutes * 60 / delay))
        self._call(lambda: self._ec2.get_waiter("image_available").wait(
            ImageIds=[image_id], WaiterConfig={"Delay": delay, "MaxAttempts": attempts}))

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
        # A list with the key as its prefix, not a HeadObject. S3 answers a HeadObject of a missing
        # key with 404 only to a caller that may list the bucket, and the Job_Role may list only
        # under jobs/ and work/, so on the machine it would answer 403. The key itself sorts first
        # among the keys it prefixes, so one result is enough.
        result = self._call(lambda: self._s3.list_objects_v2(
            Bucket=self._bucket(), Prefix=key, MaxKeys=1))
        return any(obj.get("Key") == key for obj in result.get("Contents", []))

    # CloudFormation

    def stack_outputs(self, stack_name: str) -> dict[str, str]:
        result = self._call(lambda: self._cfn.describe_stacks(StackName=stack_name))
        stacks = result.get("Stacks") or []
        if not stacks:
            raise CloudError(f"No CloudFormation stack named {stack_name!r} was found. Deploy "
                             "cloud/aws-setup.yaml first, then run the command again.")
        return {str(output.get("OutputKey", "")): str(output.get("OutputValue", ""))
                for output in stacks[0].get("Outputs", [])}

    # DynamoDB, the Website_Table

    def table_get(self, key: Key) -> dict[str, Scalar] | None:
        result = self._table_call(lambda: self._dynamodb.get_item(
            TableName=self._table(), Key=_key(key), ConsistentRead=True))
        raw = result.get("Item")
        return _from_item(raw) if raw else None

    def table_get_many(self, keys: Sequence[Key]) -> list[dict[str, Scalar] | None]:
        wanted = list(keys)
        if not wanted:
            return []
        table = self._table()
        result = self._table_call(lambda: self._dynamodb.transact_get_items(TransactItems=[
            {"Get": {"TableName": table, "Key": _key(key)}} for key in wanted]))
        responses = result.get("Responses") or [{} for _ in wanted]
        return [_from_item(r["Item"]) if r.get("Item") else None for r in responses]

    def table_query(self, pk: str, sk: KeyRange) -> list[dict[str, Scalar]]:
        names = {"#pk": "pk", "#sk": "sk"}
        values: dict[str, Any] = {":pk": {"S": pk}}
        if isinstance(sk, Prefix):
            expression = "#pk = :pk AND begins_with(#sk, :p)"
            values[":p"] = {"S": sk.value}
        else:
            expression = "#pk = :pk AND #sk BETWEEN :lo AND :hi"
            values[":lo"] = {"S": sk.low}
            values[":hi"] = {"S": sk.high}
        items: list[dict[str, Scalar]] = []

        def run() -> None:
            paginator = self._dynamodb.get_paginator("query")
            for page in paginator.paginate(TableName=self._table(), KeyConditionExpression=expression,
                                           ExpressionAttributeNames=names,
                                           ExpressionAttributeValues=values, ConsistentRead=True):
                items.extend(_from_item(raw) for raw in page.get("Items", []))

        self._table_call(run)
        return items

    def table_put(self, item: Mapping[str, Scalar], condition: Condition | None = None) -> None:
        request: dict[str, Any] = {"TableName": self._table(), "Item": _to_item(item)}
        request.update(_Expressions().with_condition(condition))
        self._table_call(lambda: self._dynamodb.put_item(**request))

    def table_update(self, key: Key, updates: Sequence[Update],
                     condition: Condition | None = None) -> None:
        request: dict[str, Any] = {"TableName": self._table(), "Key": _key(key)}
        request.update(_Expressions().with_updates(updates, condition))
        self._table_call(lambda: self._dynamodb.update_item(**request))

    def table_transact(self, items: Sequence[TransactItem], token: str | None = None) -> None:
        table = self._table()
        request: dict[str, Any] = {"TransactItems": [_transact_entry(table, entry) for entry in items]}
        if token is not None:  # one token's transaction applies once, for ten minutes
            request["ClientRequestToken"] = token
        self._table_call(lambda: self._dynamodb.transact_write_items(**request))

    def table_delete(self, key: Key) -> None:
        self._table_call(lambda: self._dynamodb.delete_item(TableName=self._table(), Key=_key(key)))

    # EC2, for "Ended without a result"

    def job_machines(self, job_id: str) -> list[InstanceInfo]:
        key, value = self._project_tag
        filters = [{"Name": f"tag:{key}", "Values": [value]},
                   {"Name": f"tag:{_JOB_ID_TAG}", "Values": [job_id]}]
        result = self._call(lambda: self._ec2.describe_instances(Filters=filters))
        return [_as_instance(raw)
                for reservation in result.get("Reservations", [])
                for raw in reservation.get("Instances", [])]

    # S3, for the website

    def s3_start_upload(self, key: str) -> str:
        result = self._call(lambda: self._s3.create_multipart_upload(Bucket=self._bucket(), Key=key))
        return str(result["UploadId"])

    def s3_part_link(self, key: str, upload_id: str, part: int, seconds: int) -> str:
        params = {"Bucket": self._bucket(), "Key": key, "UploadId": upload_id, "PartNumber": part}
        return str(self._call(lambda: self._link_client().generate_presigned_url(
            "upload_part", Params=params, ExpiresIn=seconds, HttpMethod="PUT")))

    def s3_complete_upload(self, key: str, upload_id: str,
                           parts: Sequence[tuple[int, str]]) -> None:
        listed = [{"PartNumber": number, "ETag": etag} for number, etag in parts]
        self._call(lambda: self._s3.complete_multipart_upload(
            Bucket=self._bucket(), Key=key, UploadId=upload_id, MultipartUpload={"Parts": listed}))

    def s3_abort_upload(self, key: str, upload_id: str) -> None:
        self._call(lambda: self._s3.abort_multipart_upload(
            Bucket=self._bucket(), Key=key, UploadId=upload_id))

    def s3_download_link(self, key: str, filename: str | None, seconds: int) -> str:
        params: dict[str, Any] = {"Bucket": self._bucket(), "Key": key}
        if filename is not None:  # a download; without a name, a link the viewer fetches
            params["ResponseContentDisposition"] = f'attachment; filename="{filename}"'
        return str(self._call(lambda: self._link_client().generate_presigned_url(
            "get_object", Params=params, ExpiresIn=seconds, HttpMethod="GET")))

    def s3_copy(self, source_key: str, key: str, *, part_bytes: int) -> None:
        from boto3.s3.transfer import TransferConfig

        config = TransferConfig(multipart_threshold=part_bytes, multipart_chunksize=part_bytes)
        bucket = self._bucket()
        self._call(lambda: self._s3.copy({"Bucket": bucket, "Key": source_key}, bucket, key,
                                         Config=config))

    def s3_size(self, key: str) -> int | None:
        result = self._call(lambda: self._s3.list_objects_v2(
            Bucket=self._bucket(), Prefix=key, MaxKeys=1))
        for obj in result.get("Contents", []):
            if obj.get("Key") == key:
                return int(obj.get("Size", 0))
        return None

    def s3_read_range(self, key: str, offset: int, length: int) -> bytes:
        def run() -> bytes:
            result = self._s3.get_object(Bucket=self._bucket(), Key=key,
                                         Range=f"bytes={offset}-{offset + length - 1}")
            return bytes(result["Body"].read())

        return self._call(run)

    # CloudFormation, for the Deploy_Command

    def create_change_set(self, request: Mapping[str, Any]) -> None:
        payload = dict(request)
        self._call(lambda: self._cfn.create_change_set(**payload))
        self._call(lambda: self._cfn.get_waiter("change_set_create_complete").wait(
            StackName=payload["StackName"], ChangeSetName=payload["ChangeSetName"]))

    def change_set_changes(self, stack_name: str,
                           change_set_name: str) -> list[tuple[str, str, str]]:
        changes: list[tuple[str, str, str]] = []
        token: str | None = None
        while True:
            kwargs: dict[str, Any] = {"StackName": stack_name, "ChangeSetName": change_set_name}
            if token:
                kwargs["NextToken"] = token
            page = self._call(lambda: self._cfn.describe_change_set(**kwargs))
            for change in page.get("Changes", []):
                resource = change.get("ResourceChange") or {}
                changes.append((str(resource.get("Action", "")),
                                str(resource.get("LogicalResourceId", "")),
                                str(resource.get("Replacement", ""))))
            token = page.get("NextToken")
            if not token:
                return changes

    def delete_change_set(self, stack_name: str, change_set_name: str) -> None:
        self._call(lambda: self._cfn.delete_change_set(
            StackName=stack_name, ChangeSetName=change_set_name))

    # Helpers

    def _table(self) -> str:
        """The Website_Table, or a CloudError when the caller hasn't set it yet."""
        if not self.table:
            raise CloudError("No website table is set on this AWS client. Set it before any table "
                             "call. This is a bug if a user sees it.")
        return self.table

    def _link_client(self) -> Any:
        """The S3 client for Presigned_Links, built on the first link: SigV4, virtual-hosted, on
        the regional endpoint, so every link's host is s3_origin(bucket, region)."""
        if self._links is None:
            from botocore.config import Config as BotoConfig

            self._links = self._session.client(
                "s3", config=BotoConfig(signature_version="s3v4",
                                        s3={"addressing_style": "virtual"}),
                endpoint_url=f"https://s3.{self.region}.amazonaws.com")
        return self._links

    def _table_call(self, thunk: Any) -> Any:
        """Run one DynamoDB call, turning a failed condition or a conflict into its own error."""
        try:
            return thunk()
        except ClientError as error:
            code = str(error.response.get("Error", {}).get("Code", ""))
            if code == "ConditionalCheckFailedException":
                raise ConditionFailed(None) from error
            if code == "TransactionConflictException":
                raise TransactionConflict() from error
            if code == "TransactionCanceledException":
                raise _cancelled(error.response.get("CancellationReasons") or []) from error
            raise CloudError(account_redaction(str(error))) from error
        except BotoCoreError as error:
            raise CloudError(account_redaction(str(error))) from error

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


# DynamoDB's typed values and expressions. Every attribute name and value goes in as a #n or :v
# placeholder, so no name or value is ever pasted into an expression.

def _to_value(value: Scalar) -> dict[str, Any]:
    if isinstance(value, bool):
        return {"BOOL": value}
    if isinstance(value, int):
        return {"N": str(value)}
    if isinstance(value, str):
        return {"S": value}
    raise CloudError(f"The website's table holds only text, whole numbers and true or false, not "
                     f"{type(value).__name__}. This is a bug if a user sees it.")


def _from_value(raw: Mapping[str, Any]) -> Scalar:
    if "S" in raw:
        return str(raw["S"])
    if "BOOL" in raw:
        return bool(raw["BOOL"])
    if "N" in raw:
        return int(str(raw["N"]))
    raise CloudError("The website's table holds a value of a type the website doesn't use.")


def _to_item(item: Mapping[str, Scalar]) -> dict[str, Any]:
    return {name: _to_value(value) for name, value in item.items()}


def _from_item(raw: Mapping[str, Any]) -> dict[str, Scalar]:
    return {str(name): _from_value(value) for name, value in raw.items()}


def _key(key: Key) -> dict[str, Any]:
    return {"pk": {"S": key.pk}, "sk": {"S": key.sk}}


class _Expressions:
    """Placeholder names and values for one request's condition and update expressions."""

    def __init__(self) -> None:
        self.names: dict[str, str] = {}
        self.values: dict[str, Any] = {}

    def name(self, attr: str) -> str:
        for placeholder, known in self.names.items():
            if known == attr:
                return placeholder
        placeholder = f"#n{len(self.names)}"
        self.names[placeholder] = attr
        return placeholder

    def value(self, value: Scalar) -> str:
        placeholder = f":v{len(self.values)}"
        self.values[placeholder] = _to_value(value)
        return placeholder

    def condition(self, condition: Condition) -> str:
        if isinstance(condition, Exists):
            return f"attribute_exists({self.name(condition.attr)})"
        if isinstance(condition, NotExists):
            return f"attribute_not_exists({self.name(condition.attr)})"
        if isinstance(condition, Equals):
            return f"{self.name(condition.attr)} = {self.value(condition.value)}"
        if isinstance(condition, AtMost):
            return f"{self.name(condition.attr)} <= {self.value(condition.limit)}"
        if isinstance(condition, AtLeast):
            return f"{self.name(condition.attr)} >= {self.value(condition.limit)}"
        if isinstance(condition, (And, Or)):
            word = " AND " if isinstance(condition, And) else " OR "
            return "(" + word.join(self.condition(part) for part in condition.parts) + ")"
        raise CloudError("Unknown table condition. This is a bug if a user sees it.")

    def update(self, updates: Sequence[Update]) -> str:
        sets = [f"{self.name(u.attr)} = {self.value(u.value)}" for u in updates if isinstance(u, Set)]
        adds = [f"{self.name(u.attr)} {self.value(u.amount)}" for u in updates if isinstance(u, Add)]
        removes = [self.name(u.attr) for u in updates if isinstance(u, Remove)]
        clauses = [(word, parts) for word, parts in (("SET", sets), ("ADD", adds), ("REMOVE", removes))
                   if parts]
        return " ".join(f"{word} {', '.join(parts)}" for word, parts in clauses)

    def with_condition(self, condition: Condition | None) -> dict[str, Any]:
        request: dict[str, Any] = {}
        if condition is not None:
            request["ConditionExpression"] = self.condition(condition)
        return self._placeholders(request)

    def with_updates(self, updates: Sequence[Update], condition: Condition | None) -> dict[str, Any]:
        request: dict[str, Any] = {"UpdateExpression": self.update(updates)}
        if condition is not None:
            request["ConditionExpression"] = self.condition(condition)
        return self._placeholders(request)

    def _placeholders(self, request: dict[str, Any]) -> dict[str, Any]:
        if self.names:
            request["ExpressionAttributeNames"] = dict(self.names)
        if self.values:
            request["ExpressionAttributeValues"] = dict(self.values)
        return request


def _transact_entry(table: str, entry: TransactItem) -> dict[str, Any]:
    """One TransactWriteItems entry."""
    expressions = _Expressions()
    if isinstance(entry, Put):
        body = {"TableName": table, "Item": _to_item(entry.item),
                **expressions.with_condition(entry.condition)}
        return {"Put": body}
    if isinstance(entry, Change):
        body = {"TableName": table, "Key": _key(entry.key),
                **expressions.with_updates(entry.updates, entry.condition)}
        return {"Update": body}
    if isinstance(entry, Check):
        body = {"TableName": table, "Key": _key(entry.key),
                **expressions.with_condition(entry.condition)}
        return {"ConditionCheck": body}
    raise CloudError("Unknown transaction item. This is a bug if a user sees it.")


def _cancelled(reasons: Sequence[Mapping[str, Any]]) -> Exception:
    """The error for a cancelled transaction, from its reasons in request order."""
    codes = [str(reason.get("Code") or "None") for reason in reasons]
    for index, code in enumerate(codes):
        if code == "ConditionalCheckFailed":
            return ConditionFailed(index)
    if "TransactionConflict" in codes:
        return TransactionConflict()
    listed = ", ".join(code for code in codes if code != "None") or "no reason given"
    return CloudError(f"The website's table cancelled a transaction ({listed}). Try again.")


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
