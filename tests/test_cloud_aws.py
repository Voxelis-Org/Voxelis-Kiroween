"""Unit tests for the AWS seam, cloud/aws.py (cloud-jobs task 2.6).

Two things, offline and with no AWS account, as the task and design describe.

First, the Session builders name the right things. `laptop_aws` builds its `boto3.Session` with the
region from `[aws].region` and the profile from `[aws].<profile_key>`, never `AWS_PROFILE`, so a
laptop command can't run as the wrong account (Req 1.6). `machine_aws` names only the region, so a
job machine falls back on its instance role (Req 1.7). A fake `boto3.Session` records the kwargs the
code passes, and the environment carries values the code must ignore.

`job_machine_aws` builds the machine's one with the region from `[aws].region` and the Job_Bucket
from the BUCKET variable the User_Data exports, and refuses without it. `BotoAws.s3_exists` asks with
a list whose prefix is the key, because the Job_Role may list only under jobs/ and work/, so S3
would answer its HeadObject of a missing key with 403 rather than 404.

Second, `BotoAws.run_instances` converts AWS errors so no other module touches botocore and no raw
account id reaches a chat. A capacity `ClientError` (`InsufficientInstanceCapacity` or `Unsupported`)
becomes a `CapacityUnavailable` the launch loop can catch (Req 5.5); any other `ClientError` or
`BotoCoreError` becomes a `CloudError` whose message is account-redacted (Req 1.9). A fake ec2 client
raises the botocore error in place, so nothing reaches the network.
"""

from __future__ import annotations

import base64
from typing import Any

import pytest
from botocore.exceptions import BotoCoreError, ClientError

import cloud.aws as aws_module
from cloud.aws import BotoAws, job_machine_aws, laptop_aws, machine_aws
from cloud.errors import CapacityUnavailable, CloudError
from splatpipe.config import Config

# Sentinel [aws] values, distinct from any environment variable below, so an assertion on the
# Session's kwargs shows the region and profile were named in code (Req 1.6), not taken from the env.
REGION = "test-region-1"
READ_PROFILE = "read-prof"
RUNNER_PROFILE = "runner-prof"

# A fake 12-digit account id. account_redaction must remove it from any error message (Req 1.9).
ACCOUNT = "123456789012"


def aws_config() -> Config:
    """A Config carrying only an `[aws]` table with the sentinel region and profiles.

    `[aws]` is not a Pipeline_Section, so Config runs no checks over it, and this is all `laptop_aws`
    reads from the Config it is given.
    """
    return Config({"aws": {"region": REGION, "read_profile": READ_PROFILE,
                           "runner_profile": RUNNER_PROFILE}})


def patch_session(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Replace `boto3.Session` with a recorder and give back the list its kwargs land in.

    The fake hands back a throwaway object for every client, so `BotoAws` builds with no network.
    """
    calls: list[dict[str, Any]] = []

    class FakeSession:
        def __init__(self, **kwargs: Any) -> None:
            calls.append(dict(kwargs))

        def client(self, name: str) -> Any:
            return object()

    monkeypatch.setattr(aws_module.boto3, "Session", FakeSession)
    return calls


# The Session builders name the region and the right profile (Req 1.6, 1.7)

@pytest.mark.parametrize(("profile_key", "profile"),
                         [("read_profile", READ_PROFILE), ("runner_profile", RUNNER_PROFILE)])
def test_laptop_aws_names_the_region_and_chosen_profile_not_aws_profile(
        monkeypatch: pytest.MonkeyPatch, profile_key: str, profile: str) -> None:
    # AWS_PROFILE and the region env vars are set to values the code must ignore: it names the
    # profile and the region itself, so the exact kwargs below are the Config's, not the env's.
    monkeypatch.setenv("AWS_PROFILE", "env-profile-ignored")
    monkeypatch.setenv("AWS_REGION", "env-region-ignored")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "env-region-ignored")
    calls = patch_session(monkeypatch)
    aws = laptop_aws(aws_config(), profile_key=profile_key)
    assert calls == [{"region_name": REGION, "profile_name": profile}]
    assert aws.region == REGION


def test_machine_aws_names_the_region_and_no_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = patch_session(monkeypatch)
    aws = machine_aws(REGION)
    assert calls == [{"region_name": REGION}]  # no profile_name key: the instance role is used
    assert aws.region == REGION


def test_job_machine_aws_takes_the_region_from_config_and_the_bucket_from_the_user_data(
        monkeypatch: pytest.MonkeyPatch) -> None:
    # The region env vars hold values the code must ignore: the region comes from [aws].region.
    monkeypatch.setenv("AWS_DEFAULT_REGION", "env-region-ignored")
    calls = patch_session(monkeypatch)
    aws = job_machine_aws(aws_config(), environ={"BUCKET": "voxelis-kiroween-jobs"})
    assert calls == [{"region_name": REGION}]  # no profile: the machine uses its instance role
    assert (aws.region, aws.bucket) == (REGION, "voxelis-kiroween-jobs")


def test_job_machine_aws_refuses_without_the_bucket_variable(
        monkeypatch: pytest.MonkeyPatch) -> None:
    calls = patch_session(monkeypatch)
    with pytest.raises(CloudError) as caught:
        job_machine_aws(aws_config(), environ={})
    assert "BUCKET" in str(caught.value)
    assert calls == []  # refused before any Session is built


# BotoAws.s3_exists asks with a list, which the Job_Role may make under jobs/ and work/

class _ListS3:
    """A stand-in s3 client whose list_objects_v2 records its kwargs and answers with `keys`."""

    def __init__(self, keys: list[str], error: BaseException | None = None) -> None:
        self.kwargs: dict[str, Any] = {}
        self._keys = keys
        self._error = error

    def list_objects_v2(self, **kwargs: Any) -> Any:
        self.kwargs = dict(kwargs)
        if self._error is not None:
            raise self._error
        return {"Contents": [{"Key": key} for key in self._keys]} if self._keys else {}


class _FakeSessionWithS3:
    """A stand-in Session handing `BotoAws` the seeded s3 client and throwaway others."""

    def __init__(self, s3: _ListS3) -> None:
        self._s3 = s3

    def client(self, name: str) -> Any:
        return self._s3 if name == "s3" else object()


KEY = "jobs/20261119-142530-7f3a9c/out/spend.jsonl"


@pytest.mark.parametrize(("keys", "exists"), [
    ([KEY], True),                  # the key itself sorts first among the keys it prefixes
    ([KEY + ".old"], False),        # only a longer key that starts with it
    ([], False),                    # nothing there
])
def test_s3_exists_asks_with_a_list_of_the_key_as_its_prefix(keys: list[str], exists: bool) -> None:
    # A HeadObject of a missing key answers 403 to the Job_Role, which may list only with a prefix
    # under jobs/ or work/, so a list with the key as the prefix answers the question on the machine
    # and on a laptop alike.
    s3 = _ListS3(keys)
    bot = BotoAws(_FakeSessionWithS3(s3), region=REGION, bucket="voxelis-kiroween-jobs")  # type: ignore[arg-type]
    assert bot.s3_exists(KEY) is exists
    assert s3.kwargs == {"Bucket": "voxelis-kiroween-jobs", "Prefix": KEY, "MaxKeys": 1}


def test_s3_exists_turns_an_aws_error_into_a_redacted_cloud_error() -> None:
    error = ClientError({"Error": {"Code": "AccessDenied",
                                   "Message": f"User arn:aws:sts::{ACCOUNT}:assumed-role/x is denied"}},
                        "ListObjectsV2")
    bot = BotoAws(_FakeSessionWithS3(_ListS3([], error)), region=REGION,  # type: ignore[arg-type]
                  bucket="voxelis-kiroween-jobs")
    with pytest.raises(CloudError) as caught:
        bot.s3_exists(KEY)
    assert ACCOUNT not in str(caught.value) and "<account>" in str(caught.value)


# BotoAws.run_instances converts AWS errors (Req 5.5, 1.9)

class _FakeEc2:
    """A stand-in ec2 client whose run_instances raises whatever the test seeds, so no network."""

    def __init__(self, error: BaseException) -> None:
        self._error = error

    def run_instances(self, **kwargs: Any) -> Any:
        raise self._error


class _FakeSessionFor:
    """A stand-in Session handing `BotoAws` the seeded ec2 client and throwaway others."""

    def __init__(self, ec2: _FakeEc2) -> None:
        self._ec2 = ec2

    def client(self, name: str) -> Any:
        return self._ec2 if name == "ec2" else object()


class _FakeBotoCoreError(BotoCoreError):
    """A BotoCoreError whose message carries an account id, to prove the CloudError is redacted."""

    fmt = "could not reach the endpoint for account {account}"


def boto_aws_raising(error: BaseException) -> BotoAws:
    """A `BotoAws` whose ec2 client raises `error` the moment run_instances is called."""
    return BotoAws(_FakeSessionFor(_FakeEc2(error)), region=REGION)


def client_error(code: str, message: str) -> ClientError:
    """A botocore ClientError with the given error code and message, as AWS raises on a bad call."""
    return ClientError({"Error": {"Code": code, "Message": message}}, "RunInstances")


@pytest.mark.parametrize("code", ["InsufficientInstanceCapacity", "Unsupported"])
def test_run_instances_turns_a_capacity_error_into_capacity_unavailable(code: str) -> None:
    bot = boto_aws_raising(client_error(code, "no capacity right now"))
    request = {"InstanceType": "g5.8xlarge", "Placement": {"AvailabilityZone": "us-east-1b"}}
    with pytest.raises(CapacityUnavailable) as caught:
        bot.run_instances(request)
    assert caught.value.code == code
    assert caught.value.instance_type == "g5.8xlarge"
    assert caught.value.zone == "us-east-1b"


def test_run_instances_turns_another_client_error_into_a_redacted_cloud_error() -> None:
    message = f"User arn:aws:iam::{ACCOUNT}:user/sam is not authorized to call RunInstances"
    bot = boto_aws_raising(client_error("UnauthorizedOperation", message))
    with pytest.raises(CloudError) as caught:
        bot.run_instances({"InstanceType": "g5.8xlarge"})
    text = str(caught.value)
    assert ACCOUNT not in text          # the account id is gone
    assert "<account>" in text          # replaced by the redaction marker


def test_run_instances_turns_a_botocore_error_into_a_redacted_cloud_error() -> None:
    bot = boto_aws_raising(_FakeBotoCoreError(account=ACCOUNT))
    with pytest.raises(CloudError) as caught:
        bot.run_instances({"InstanceType": "g5.8xlarge"})
    text = str(caught.value)
    assert ACCOUNT not in text
    assert "<account>" in text


# BotoAws.console_output asks for the newest output and decodes it

class _ConsoleEc2:
    """A stand-in ec2 client that records get_console_output's kwargs and answers like EC2 does,
    with the output base64-encoded."""

    def __init__(self, text: str) -> None:
        self.kwargs: dict[str, Any] = {}
        self._text = text

    def get_console_output(self, **kwargs: Any) -> Any:
        self.kwargs = dict(kwargs)
        return {"Output": base64.b64encode(self._text.encode("utf-8")).decode("ascii")}


def test_console_output_asks_for_the_latest_output_and_decodes_it() -> None:
    # Without Latest=True EC2 only posts console output around state changes, so a line printed long
    # after boot, like the image build's done marker, would never show up.
    ec2 = _ConsoleEc2("booted\nvoxelis-kiroween image build: install complete\n")
    bot = BotoAws(_FakeSessionFor(ec2), region=REGION)  # type: ignore[arg-type]
    text = bot.console_output("i-0abc")
    assert ec2.kwargs == {"InstanceId": "i-0abc", "Latest": True}
    assert text == "booted\nvoxelis-kiroween image build: install complete\n"
