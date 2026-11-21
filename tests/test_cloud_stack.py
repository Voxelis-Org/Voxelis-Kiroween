"""Unit tests for the Stack resolver, cloud/stack.py (cloud-jobs task 2.6).

`resolve_stack` reads the Stack's outputs through the `Aws` seam and maps the seven keys to a
`StackOutputs`; a `FakeAws` seeded with a DescribeStacks result stands in for AWS, so there is no
network and no account (Req 1.3). A missing output is a `CloudError` naming it, so a half-deployed
Stack fails in plain words rather than later with a `KeyError`.

Last, a structural check that no `cloud/*.py` file holds a 12-digit account id or an ARN literal, so
nothing account-specific lives in the repo: the Stack supplies every id and ARN at run time
(Req 1.3, 1.8, 1.9). These tests run on the CPU with no network and no AWS.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from splatpipe.config import REPO_ROOT

from cloud.errors import CloudError
from cloud.stack import StackOutputs, resolve_stack

from fake_aws import FakeAws

STACK_NAME = "voxelis-kiroween"

# A DescribeStacks result: the seven output keys cloud/aws-setup.yaml publishes, each with a distinct
# value so a wrong mapping (a field reading another key) would fail. No value is a real id or ARN.
STACK = {
    "BucketName": "bucket-value",
    "JobRoleArn": "job-role-value",
    "JobInstanceProfileArn": "job-profile-value",
    "RunnerRoleArn": "runner-role-value",
    "SecurityGroupId": "sg-value",
    "VpcId": "vpc-value",
    "GroupName": "group-value",
}

# Each output key and the StackOutputs field it must map to, so the mapping test is explicit and an
# accidental swap (bucket reading VpcId, say) is caught.
FIELD_FOR_KEY = {
    "BucketName": "bucket",
    "JobRoleArn": "job_role_arn",
    "JobInstanceProfileArn": "job_instance_profile_arn",
    "RunnerRoleArn": "runner_role_arn",
    "SecurityGroupId": "security_group_id",
    "VpcId": "vpc_id",
    "GroupName": "group_name",
}


def test_resolve_stack_maps_the_seven_outputs() -> None:
    fake = FakeAws(stack=STACK)
    outputs = resolve_stack(fake, STACK_NAME)
    assert isinstance(outputs, StackOutputs)
    for key, field in FIELD_FOR_KEY.items():
        assert getattr(outputs, field) == STACK[key]
    assert ("stack_outputs", STACK_NAME) in fake.calls  # it read the Stack by name


@pytest.mark.parametrize("missing", list(STACK))
def test_a_missing_output_is_a_cloud_error_naming_it(missing: str) -> None:
    partial = {key: value for key, value in STACK.items() if key != missing}
    with pytest.raises(CloudError) as caught:
        resolve_stack(FakeAws(stack=partial), STACK_NAME)
    assert missing in str(caught.value)


# No cloud/*.py holds a 12-digit account id or an ARN literal (Req 1.3, 1.8, 1.9)

CLOUD_PY = sorted((REPO_ROOT / "cloud").glob("*.py"))
# A maximal run of exactly 12 digits is an AWS account id; the redaction uses the same shape.
TWELVE_DIGITS = re.compile(r"(?<!\d)\d{12}(?!\d)")
# Every ARN literal starts with its partition: arn:aws (also arn:aws-cn, arn:aws-us-gov). A bare
# "arn:" would wrongly match field names like job_role_arn:, so the partition prefix is the needle.
ARN_LITERAL = "arn:aws"


def test_the_structural_check_sees_the_cloud_files() -> None:
    # A guard, so the parametrized scan below can't pass vacuously by finding no files.
    names = {path.name for path in CLOUD_PY}
    assert {"aws.py", "stack.py", "errors.py", "redact.py"} <= names


@pytest.mark.parametrize("path", CLOUD_PY, ids=[path.name for path in CLOUD_PY])
def test_no_cloud_file_holds_a_twelve_digit_or_arn_literal(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    assert not TWELVE_DIGITS.search(text), f"{path.name} holds a 12-digit literal (an account id?)"
    assert ARN_LITERAL not in text, f"{path.name} holds an ARN literal"
