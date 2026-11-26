"""The launch request and the in-command fallback over zones and types.

`build_run_instances_request` is a pure function returning the exact `RunInstances` request dict
EC2 receives, so a test asserts on the very dict the code builds (Property 4, Req 8.9). Every
request sets terminate-on-shutdown, requires IMDSv2, names the Job_Profile, the Job_Security_Group
and a subnet, carries the Project_Tag on the instance, its volume and its network interface (and the
Job_Id tag on a Job machine's instance), the User_Data and a per-attempt client token, and names no
key pair, launch template, market option or API-termination setting, so the machine is On-Demand,
the Stop_All_Command can terminate it, and every condition in the Runner_Role's launch policy is met
(Req 8.1-8.6).

`client_token` derives a stable idempotency token from a label that names one run (the Job_Id for a
Job or a Smoke_Test, the image's name for an Image_Build) and the attempt index, so repeating one
attempt starts nothing new while each zone/type attempt gets a fresh token (Req 5.6). `launch_with_fallback` tries each `(type, subnet, zone)` in order: a
`CapacityUnavailable` moves on to the next attempt, any other error (a `CloudError`) stops the loop
with nothing more started, a success returns the instance with its type and zone, and an exhausted
plan is a `CloudError` naming the types and zones tried (Req 5.1, 5.3, 5.4, 5.5).

A cloud module may import `splatpipe` to call its public functions; `splatpipe` never imports
`cloud`. This module talks to AWS only through the `Aws` protocol it is handed, imports no boto3 and
no GPU package, and so loads on a laptop.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

from cloud.errors import CapacityUnavailable, CloudError

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from typing import Any

    from cloud.aws import Aws, InstanceInfo

# The tag key a Job machine carries beside the Project_Tag, holding the machine's Job_Id, so the
# List_Command and the Stop_All_Command can tell which Job a machine is running (Req 8.2).
JOB_ID_TAG = "job_id"


def build_run_instances_request(*, image_id: str, instance_type: str, subnet_id: str,
                                instance_profile_arn: str, security_group_id: str,
                                user_data: str, client_token: str,
                                project_tag: tuple[str, str],
                                job_id: str | None) -> dict[str, Any]:
    """The pure `RunInstances` request dict for one launch attempt.

    It carries `MinCount == MaxCount == 1`, terminate-on-shutdown, IMDSv2 required, the given image,
    type, subnet, instance-profile ARN, security group, User_Data and client token, and the
    Project_Tag on the instance, its volume and its network interface (plus the Job_Id tag on the
    instance when `job_id` is given). It names no key pair, launch template, market option or
    `DisableApiTermination`, so the machine is On-Demand and terminates when it shuts down. Property
    4 asserts on this dict (Req 8.1-8.6, 8.9).
    """
    return {
        "ImageId": image_id,
        "InstanceType": instance_type,
        "MinCount": 1,
        "MaxCount": 1,
        "SubnetId": subnet_id,
        "SecurityGroupIds": [security_group_id],
        "IamInstanceProfile": {"Arn": instance_profile_arn},
        "MetadataOptions": {"HttpTokens": "required", "HttpEndpoint": "enabled"},
        "InstanceInitiatedShutdownBehavior": "terminate",
        "ClientToken": client_token,
        "UserData": user_data,
        "TagSpecifications": [
            {"ResourceType": "instance", "Tags": _tags(project_tag, job_id)},
            {"ResourceType": "volume", "Tags": [_tag(project_tag)]},
            {"ResourceType": "network-interface", "Tags": [_tag(project_tag)]},
        ],
    }


def client_token(label: str, attempt: int) -> str:
    """A stable `RunInstances` client token from a label and the attempt index (Req 5.6).

    `label` names one run: the Job_Id for a Job or a Smoke_Test, the image's name for an
    Image_Build. The token is a deterministic hash of the two, so repeating one attempt mints the
    same token (EC2 treats the retry as the same request and starts nothing new) while each
    zone/type attempt, with its own index, gets a fresh one. Two runs must never share a label: EC2
    remembers a token after its machine has ended and refuses it with other parameters
    (IdempotentParameterMismatch). The 64-character hex digest is within EC2's 64-character
    client-token limit.
    """
    return hashlib.sha256(f"{label}:{attempt}".encode("utf-8")).hexdigest()


def launch_with_fallback(aws: Aws, plan: Sequence[tuple[str, str, str]],
                         build: Callable[[int, str, str], dict[str, Any]]
                         ) -> tuple[InstanceInfo, str, str]:
    """Try each `(type, subnet, zone)` of `plan` in order until one launches (Req 5.1, 5.3-5.5).

    `build(attempt, instance_type, subnet_id)` makes the request for that attempt, with a per-attempt
    client token so a retried attempt is idempotent and each attempt is fresh. A `CapacityUnavailable`
    moves on to the next attempt; any other error (a `CloudError` from `build` or `run_instances`)
    propagates, stopping the loop with nothing more started. On success the instance, its type and
    its zone are returned. An exhausted plan raises a `CloudError` naming the types and zones tried.
    """
    tried: list[tuple[str, str, str]] = []  # (type, zone, AWS's code) of each refused attempt
    for attempt, (instance_type, subnet_id, zone) in enumerate(plan):
        request = build(attempt, instance_type, subnet_id)
        try:
            instance = aws.run_instances(request)
        except CapacityUnavailable as error:
            tried.append((instance_type, zone, error.code))
            continue  # no On-Demand capacity here; try the next zone or type
        return instance, instance_type, zone
    raise CloudError(_exhausted_message(tried))


def _tag(project_tag: tuple[str, str]) -> dict[str, str]:
    """The Project_Tag as one EC2 tag dict."""
    key, value = project_tag
    return {"Key": key, "Value": value}


def _tags(project_tag: tuple[str, str], job_id: str | None) -> list[dict[str, str]]:
    """The instance's tags: the Project_Tag always, the Job_Id tag only for a Job machine."""
    tags = [_tag(project_tag)]
    if job_id is not None:
        tags.append({"Key": JOB_ID_TAG, "Value": job_id})
    return tags


def _exhausted_message(tried: Sequence[tuple[str, str, str]]) -> str:
    """A Readable_Error for an exhausted plan, naming the types and zones that were tried and the
    code AWS answered each with (Req 5.4).

    InsufficientInstanceCapacity means AWS has no machine of that type there right now, while
    Unsupported means it won't start that type there, so the codes tell a shortage that passes from
    a request that will keep failing.
    """
    if not tried:
        return ("No machine could start because no fallback type had a zone with both capacity and a "
                "default subnet to try. Check the preset's fallback list, then run the command again.")
    by_type: dict[str, list[str]] = {}
    codes: dict[str, int] = {}
    for instance_type, zone, code in tried:
        zones = by_type.setdefault(instance_type, [])
        if zone and zone not in zones:
            zones.append(zone)
        codes[code] = codes.get(code, 0) + 1
    parts = [f"{instance_type} in {', '.join(zones)}" if zones else instance_type
             for instance_type, zones in by_type.items()]
    answers = " and ".join(f"{code} {_times(count)}" for code, count in codes.items())
    return ("No machine could start: AWS had no On-Demand capacity for any fallback type in any zone "
            f"tried ({'; '.join(parts)}). AWS answered {answers}. Wait a few minutes and run the "
            "command again, or try a different preset.")


def _times(count: int) -> str:
    """`once`, `twice` or `<n> times`, for how often AWS gave one answer."""
    return {1: "once", 2: "twice"}.get(count, f"{count} times")
