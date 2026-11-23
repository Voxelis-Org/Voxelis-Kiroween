"""The Stack resolver: the seven Stack_Outputs Cloud_Jobs reads, and `resolve_stack`.

`StackOutputs` holds the bucket name, the ARNs, the ids and the group name that `cloud/aws-setup.yaml`
publishes as its `Outputs`. `resolve_stack` reads them from AWS by name and maps the seven keys; a
missing output or stack is a `CloudError` naming what's missing (Req 1.3). This module holds no bucket
name, id or ARN literal of its own: every value comes from AWS at run time. The only literals here
are the CloudFormation output key names, which must match `cloud/aws-setup.yaml`'s `Outputs` exactly.

A cloud module may import `splatpipe` to call its public functions; `splatpipe` never imports `cloud`.
This module imports no boto3 and no GPU package.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from cloud.errors import CloudError

if TYPE_CHECKING:
    from cloud.aws import Aws


@dataclass(frozen=True)
class StackOutputs:
    """The seven Stack_Outputs Cloud_Jobs reads, mapped from `cloud/aws-setup.yaml`'s `Outputs`.

    Each field holds a value AWS returns; the comment gives the matching CloudFormation output key.
    """

    bucket: str                    # BucketName
    job_role_arn: str              # JobRoleArn
    job_instance_profile_arn: str  # JobInstanceProfileArn
    runner_role_arn: str           # RunnerRoleArn
    security_group_id: str         # SecurityGroupId
    vpc_id: str                    # VpcId
    group_name: str                # GroupName


# Each StackOutputs field and the CloudFormation output key it comes from, in field order. The keys
# match cloud/aws-setup.yaml's Outputs exactly; changing one there means changing it here too.
_OUTPUT_KEYS: tuple[tuple[str, str], ...] = (
    ("bucket", "BucketName"),
    ("job_role_arn", "JobRoleArn"),
    ("job_instance_profile_arn", "JobInstanceProfileArn"),
    ("runner_role_arn", "RunnerRoleArn"),
    ("security_group_id", "SecurityGroupId"),
    ("vpc_id", "VpcId"),
    ("group_name", "GroupName"),
)


def resolve_stack(aws: Aws, stack_name: str) -> StackOutputs:
    """Read the Stack's outputs by name and map the seven keys.

    A missing output or stack is a `CloudError` naming what's missing (Req 1.3). Holds no bucket
    name, id or ARN literal: every value comes from AWS.
    """
    outputs = aws.stack_outputs(stack_name)
    values: dict[str, str] = {}
    missing: list[str] = []
    for field, key in _OUTPUT_KEYS:
        value = outputs.get(key, "")
        if value:
            values[field] = value
        else:
            missing.append(key)
    if missing:
        names = ", ".join(missing)
        raise CloudError(
            f"The CloudFormation stack {stack_name!r} is missing these outputs: {names}. Deploy "
            "the current cloud/aws-setup.yaml, or update the stack to it, then run the command "
            "again.")
    return StackOutputs(**values)
