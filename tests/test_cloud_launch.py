"""Unit tests for the launch fallback loop, cloud/launch.py (cloud-jobs task 4.3).

Five behaviours of `launch_with_fallback`, as the task describes, all over a `FakeAws` with no
network and no AWS account. The loop is handed a `plan` of `(instance_type, subnet_id, zone)`
attempts and a `build(attempt, type, subnet)` that mints each attempt's request; the tests drive it
through the fake's seeded failure seams (`no_capacity`, `unsupported`, `launch_failures`):

- a type with no On-Demand capacity in one zone falls through to the next zone of the same type
  (Req 5.1), and the loop reports the type and zone it got (Req 5.3);
- `Unsupported` is a capacity error too, so it moves on to the next type on the Fallback_List (Req
  5.1);
- an ambiguous error (anything that is not a `CapacityUnavailable`) stops the loop at once, launching
  nothing more, because a blind retry could start a second machine (Req 5.5);
- on the happy path the first attempt launches, so exactly one `run_instances` runs and the rest of
  the plan is never tried (Req 5.3, 5.7);
- each attempt is built with its own client token, so repeating one attempt is idempotent while each
  zone/type attempt is a fresh request (Req 5.6).

The `build` the tests pass is the real one the Start_Command uses: a per-attempt `client_token` and
the pure `build_run_instances_request`, recording each request so a test can read back the tokens it
minted. The request carries the attempt's `SubnetId`, which is how the fake tells which zone an
attempt targeted.
"""

from __future__ import annotations

from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from cloud.errors import CloudError
from cloud.launch import JOB_ID_TAG, build_run_instances_request, client_token, launch_with_fallback

from fake_aws import FakeAws, cloud_config

# The Project_Tag every request carries, and a Job_Id-shaped label for the client token.
PROJECT_TAG = ("project", "voxelis-kiroween")
JOB_ID = "20261119-142530-7f3a9c"


def _recording_build(records: list[dict[str, Any]]):
    """A `build(attempt, type, subnet)` like the Start_Command's, recording each request it makes.

    It mints the attempt's client token through `client_token(JOB_ID, attempt)` and the request
    through the pure `build_run_instances_request`, then appends the request to `records` so a test
    can read back the per-attempt tokens. The ARN holds no digits, so the structural no-account-id
    check over `cloud/*.py` has nothing to catch here either.
    """
    def build(attempt: int, instance_type: str, subnet_id: str) -> dict[str, Any]:
        request = build_run_instances_request(
            image_id="ami-fake",
            instance_type=instance_type,
            subnet_id=subnet_id,
            instance_profile_arn="arn:aws:iam::fake:instance-profile/job",
            security_group_id="sg-fake",
            user_data="#!/bin/bash\n",
            client_token=client_token(JOB_ID, attempt),
            project_tag=PROJECT_TAG,
            job_id=JOB_ID,
        )
        records.append(request)
        return request

    return build


# Req 5.1, 5.3: no capacity in one zone falls through to the next zone of the same type.

def test_no_capacity_in_one_zone_falls_through_to_the_next_zone() -> None:
    aws = FakeAws(
        subnets={"us-east-1a": "subnet-a", "us-east-1b": "subnet-b"},
        no_capacity={("g5.8xlarge", "us-east-1a")})
    plan = [
        ("g5.8xlarge", "subnet-a", "us-east-1a"),
        ("g5.8xlarge", "subnet-b", "us-east-1b"),
    ]
    records: list[dict[str, Any]] = []

    instance, instance_type, zone = launch_with_fallback(aws, plan, _recording_build(records))

    # It reports the type and the zone it actually got: the same type, in the second zone (Req 5.3).
    assert instance_type == "g5.8xlarge"
    assert zone == "us-east-1b"
    assert instance.zone == "us-east-1b"
    # Zone A was tried (no capacity) and then zone B launched: two run_instances, one machine.
    assert aws.calls == [("run_instances", "g5.8xlarge"), ("run_instances", "g5.8xlarge")]
    assert len(aws.instances) == 1


# Req 5.1: Unsupported is a capacity error, so the loop moves to the next type.

def test_unsupported_type_moves_to_the_next_type() -> None:
    aws = FakeAws(
        subnets={"us-east-1a": "subnet-a"},
        unsupported={("g5.8xlarge", "us-east-1a")})
    plan = [
        ("g5.8xlarge", "subnet-a", "us-east-1a"),
        ("g5.4xlarge", "subnet-a", "us-east-1a"),
    ]
    records: list[dict[str, Any]] = []

    instance, instance_type, zone = launch_with_fallback(aws, plan, _recording_build(records))

    # The first type was Unsupported, so the second type on the list launched instead.
    assert instance_type == "g5.4xlarge"
    assert zone == "us-east-1a"
    assert aws.calls == [("run_instances", "g5.8xlarge"), ("run_instances", "g5.4xlarge")]
    assert len(aws.instances) == 1


# Req 5.5: an ambiguous error stops the loop with nothing launched.

def test_ambiguous_error_stops_the_loop_with_nothing_launched() -> None:
    aws = FakeAws(
        subnets={"us-east-1a": "subnet-a", "us-east-1b": "subnet-b"},
        launch_failures={("g5.8xlarge", "us-east-1a")})
    plan = [
        ("g5.8xlarge", "subnet-a", "us-east-1a"),
        ("g5.4xlarge", "subnet-b", "us-east-1b"),
    ]
    records: list[dict[str, Any]] = []

    with pytest.raises(CloudError):
        launch_with_fallback(aws, plan, _recording_build(records))

    # The error is not a CapacityUnavailable, so the loop stopped at the first attempt: the next
    # attempt was never tried and no machine exists.
    assert aws.calls == [("run_instances", "g5.8xlarge")]
    assert aws.instances == []


# Req 5.3, 5.7: the happy path launches exactly one machine.

def test_happy_path_launches_exactly_one_machine() -> None:
    aws = FakeAws(subnets={"us-east-1a": "subnet-a"})
    plan = [
        ("g5.8xlarge", "subnet-a", "us-east-1a"),
        ("g5.4xlarge", "subnet-a", "us-east-1a"),
    ]
    records: list[dict[str, Any]] = []

    instance, instance_type, zone = launch_with_fallback(aws, plan, _recording_build(records))

    assert instance_type == "g5.8xlarge"
    assert zone == "us-east-1a"
    # The first attempt succeeded, so run_instances ran exactly once and the rest of the plan was
    # never tried: one machine, never a second.
    assert aws.calls == [("run_instances", "g5.8xlarge")]
    assert len(aws.instances) == 1


# Req 5.6: the client token differs per attempt.

def test_client_token_differs_per_attempt() -> None:
    aws = FakeAws(
        subnets={"us-east-1a": "subnet-a", "us-east-1b": "subnet-b", "us-east-1c": "subnet-c"},
        no_capacity={("g5.8xlarge", "us-east-1a"), ("g5.8xlarge", "us-east-1b")})
    plan = [
        ("g5.8xlarge", "subnet-a", "us-east-1a"),
        ("g5.8xlarge", "subnet-b", "us-east-1b"),
        ("g5.8xlarge", "subnet-c", "us-east-1c"),
    ]
    records: list[dict[str, Any]] = []

    launch_with_fallback(aws, plan, _recording_build(records))

    # Three attempts were built, each with its own token: repeating one attempt is idempotent, but
    # each zone/type attempt is a fresh request.
    tokens = [request["ClientToken"] for request in records]
    assert len(tokens) == 3
    assert len(set(tokens)) == 3


# Req 5.4: an exhausted plan names each type, its zones and AWS's answers.

def test_an_exhausted_plan_names_the_types_zones_and_awss_answers() -> None:
    # A shortage (InsufficientInstanceCapacity) passes, while Unsupported keeps failing, so the
    # message says which answers AWS gave, and how often.
    aws = FakeAws(
        subnets={"us-east-1a": "subnet-a", "us-east-1b": "subnet-b"},
        no_capacity={("g5.8xlarge", "us-east-1a"), ("g5.8xlarge", "us-east-1b"),
                     ("g5.4xlarge", "us-east-1a")},
        unsupported={("g5.4xlarge", "us-east-1b")})
    plan = [
        ("g5.8xlarge", "subnet-a", "us-east-1a"),
        ("g5.8xlarge", "subnet-b", "us-east-1b"),
        ("g5.4xlarge", "subnet-a", "us-east-1a"),
        ("g5.4xlarge", "subnet-b", "us-east-1b"),
    ]

    with pytest.raises(CloudError) as caught:
        launch_with_fallback(aws, plan, _recording_build([]))

    message = str(caught.value)
    assert "g5.8xlarge in us-east-1a, us-east-1b; g5.4xlarge in us-east-1a, us-east-1b" in message
    assert "AWS answered InsufficientInstanceCapacity 3 times and Unsupported once." in message
    assert aws.instances == []


def test_an_exhausted_plan_with_one_answer_says_it_once() -> None:
    aws = FakeAws(subnets={"us-east-1a": "subnet-a"},
                  no_capacity={("g5.8xlarge", "us-east-1a")})

    with pytest.raises(CloudError) as caught:
        launch_with_fallback(aws, [("g5.8xlarge", "subnet-a", "us-east-1a")], _recording_build([]))

    assert "AWS answered InsufficientInstanceCapacity once." in str(caught.value)


# Property 4: Every launch request carries the required settings (Req 8.1-8.6, 8.9, 16.5)

# The 8 Allowed_Types, from the real [cloud] shape: the property draws its instance type from these,
# so it holds "for any Allowed_Type" without hard-coding the list here.
ALLOWED_TYPES = tuple(sorted(cloud_config().types))

# Stand-ins for the opaque strings the caller passes straight through: the AMI id, the subnet id,
# the instance-profile ARN and the security-group id. The builder stores each verbatim, so the
# property must hold for any non-empty string.
OPAQUE = st.text(min_size=1, max_size=64)

# The User_Data a real script would carry: any text, the empty string and newlines included.
USER_DATA = st.text(max_size=256)

# A Job machine's Job_Id is one of these; the builder (used for the Image_Build) may get None
# instead, and then carries no Job_Id tag. The pattern is splatpipe.jobs.JOB_ID, so these are
# real-shaped ids.
JOB_ID_OR_NONE = st.one_of(
    st.none(), st.from_regex(r"[0-9]{8}-[0-9]{6}-[0-9a-f]{6}", fullmatch=True))

# The client token is a fixed collaborator here: Property 4 is about the settings the request
# carries, not the token, so it is held constant rather than varied.
CLIENT_TOKEN = "fake-client-token"

# The Project_Tag as one EC2 tag dict, the form every TagSpecifications entry must contain.
PROJECT_TAG_DICT = {"Key": PROJECT_TAG[0], "Value": PROJECT_TAG[1]}

# The four keys the request must never carry, so the machine is On-Demand, keeps no SSH key pair,
# follows no launch template, and the Stop_All_Command can terminate it (Req 8.1, 8.5).
FORBIDDEN_KEYS = ("KeyName", "LaunchTemplate", "InstanceMarketOptions", "DisableApiTermination")


def _tags_for(request: dict[str, Any], resource_type: str) -> list[dict[str, str]] | None:
    """The EC2 Tags list for `resource_type` in a request's TagSpecifications, or None if absent."""
    for spec in request["TagSpecifications"]:
        if spec["ResourceType"] == resource_type:
            return spec["Tags"]
    return None


@settings(deadline=None)
@given(
    image_id=OPAQUE,
    instance_type=st.sampled_from(ALLOWED_TYPES),
    subnet_id=OPAQUE,
    instance_profile_arn=OPAQUE,
    security_group_id=OPAQUE,
    user_data=USER_DATA,
    job_id=JOB_ID_OR_NONE,
)
def test_property_4_every_launch_request_carries_the_required_settings(
        image_id: str, instance_type: str, subnet_id: str, instance_profile_arn: str,
        security_group_id: str, user_data: str, job_id: str | None) -> None:
    """Feature: cloud-jobs, Property 4: Every launch request carries the required settings

    **Validates: Requirements 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.9, 16.5**
    """
    request = build_run_instances_request(
        image_id=image_id,
        instance_type=instance_type,
        subnet_id=subnet_id,
        instance_profile_arn=instance_profile_arn,
        security_group_id=security_group_id,
        user_data=user_data,
        client_token=CLIENT_TOKEN,
        project_tag=PROJECT_TAG,
        job_id=job_id,
    )

    # Exactly one machine, terminating on shutdown, with IMDSv2 required (Req 8.1, 8.6).
    assert request["MinCount"] == request["MaxCount"] == 1
    assert request["InstanceInitiatedShutdownBehavior"] == "terminate"
    assert request["MetadataOptions"] == {"HttpTokens": "required", "HttpEndpoint": "enabled"}

    # The Job_Profile, the Job_Security_Group, the subnet and the User_Data pass through verbatim
    # (Req 8.3, 8.4).
    assert request["IamInstanceProfile"] == {"Arn": instance_profile_arn}
    assert request["SecurityGroupIds"] == [security_group_id]
    assert request["SubnetId"] == subnet_id
    assert request["UserData"] == user_data

    # The Project_Tag is on the instance, its volume and its network interface (Req 8.2).
    for resource_type in ("instance", "volume", "network-interface"):
        tags = _tags_for(request, resource_type)
        assert tags is not None
        assert PROJECT_TAG_DICT in tags

    # The Job_Id tag is on the instance exactly when a Job_Id is given (Req 8.2).
    instance_tags = _tags_for(request, "instance")
    assert instance_tags is not None
    if job_id is None:
        assert all(tag["Key"] != JOB_ID_TAG for tag in instance_tags)
    else:
        assert {"Key": JOB_ID_TAG, "Value": job_id} in instance_tags

    # No key pair, launch template, market option or API-termination setting (Req 8.5, 8.1).
    for forbidden in FORBIDDEN_KEYS:
        assert forbidden not in request
