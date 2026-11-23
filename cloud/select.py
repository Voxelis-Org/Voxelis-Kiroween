"""Choosing the machine: the Step_Needs, the hard-needs filter, the quota admission and the plan.

Every function here is pure and order-preserving, so the Start_Command can work out which machine to
launch without touching AWS until it has to. `step_needs_over` reads each Step's own Config section
for the Job's Preset and aggregates the hard needs; `fits_hard_needs` and `select_candidates` keep
the Preset's Fallback_List entries that cover those needs, in order; `running_vcpus` and `admit`
hold the launch to the real G_Quota; and `launch_plan` lays out the ordered (type, subnet, zone)
attempts the launch loop then tries (Req 3, 4, 5).

Memory and GPU memory are hard needs: a type with too little of either is dropped. vCPUs are soft,
because fewer vCPUs only make a Step slower, so they never remove a type (Req 3.2).

A cloud module may import `splatpipe` to call its public functions; `splatpipe` never imports
`cloud`. This module imports no boto3 and no GPU package: it talks to AWS only through the `Aws`
protocol it is handed, so it loads on a laptop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from cloud.errors import CloudError

if TYPE_CHECKING:
    from collections.abc import Sequence

    from splatpipe.config import Config

    from cloud.aws import Aws, InstanceInfo
    from cloud.config import CloudConfig, MachineSpec
    from cloud.stack import StackOutputs


@dataclass(frozen=True)
class Needs:
    """The hard and soft resource needs aggregated over the Steps a machine must run."""

    min_memory_gb: float
    min_gpu_memory_gb: float      # 0 means no GPU need
    min_vcpus: int                # informational only; never excludes a type (soft)


def step_needs_over(step_names: Sequence[str], config: Config, preset: str) -> Needs:
    """The hard needs across the Steps that will run, as the largest need in each dimension.

    Each Step's needs come from its own Config section for `preset` (its Shared_Settings and its
    Preset table), read through `splatpipe.config.Config.step_settings(name, preset)`; this imports
    no Step module. A key a Step omits counts as 0, so a missing key means no need in that dimension,
    and an empty `step_names` gives zero needs (Req 2.7, 3.1, assumption E).
    """
    min_memory_gb: float = 0.0
    min_gpu_memory_gb: float = 0.0
    min_vcpus: int = 0
    for name in step_names:
        settings = config.step_settings(name, preset)
        min_memory_gb = max(min_memory_gb, settings.get("min_memory_gb", 0))
        min_gpu_memory_gb = max(min_gpu_memory_gb, settings.get("min_gpu_memory_gb", 0))
        min_vcpus = max(min_vcpus, settings.get("min_vcpus", 0))
    return Needs(
        min_memory_gb=float(min_memory_gb),
        min_gpu_memory_gb=float(min_gpu_memory_gb),
        min_vcpus=int(min_vcpus),
    )


def fits_hard_needs(spec: MachineSpec, needs: Needs) -> bool:
    """Whether a Machine_Spec covers the hard needs: enough memory and enough GPU memory.

    vCPUs are never checked, because `min_vcpus` is soft (Req 3.2). Every Allowed_Type has a GPU, so
    a zero `min_gpu_memory_gb` imposes no constraint.
    """
    return spec.memory_gb >= needs.min_memory_gb and spec.gpu_memory_gb >= needs.min_gpu_memory_gb


def select_candidates(preset: str, needs: Needs, cc: CloudConfig) -> list[str]:
    """The Preset's Fallback_List, in order, keeping only types that fit the hard needs (Req 3.1, 3.3).

    No type fitting is a `CloudError` naming the unmet need, before anything is uploaded or launched
    (Req 3.4).
    """
    fallback = cc.preset_plan(preset).fallback
    kept = [name for name in fallback if fits_hard_needs(cc.spec(name), needs)]
    if not kept:
        raise CloudError(
            f"No machine type in the {preset} preset's fallback list fits the steps' hard needs: at "
            f"least {needs.min_memory_gb:g} GB of memory and {needs.min_gpu_memory_gb:g} GB of GPU "
            f"memory. The fallback types are {', '.join(fallback)}. Lower the steps' min_memory_gb "
            "or min_gpu_memory_gb, or add a larger type to the preset's fallback list in config.toml, "
            "then run the command again.")
    return kept


def running_vcpus(instances: Sequence[InstanceInfo], cc: CloudConfig) -> int:
    """The sum of our non-terminated machines' vCPUs, for the quota check (Req 4.2)."""
    return sum(cc.spec(instance.type).vcpus for instance in instances)


def admit(candidates: Sequence[str], running: int, quota: float, cc: CloudConfig) -> list[str]:
    """The candidates, in order, that still fit the G_Quota once the running machines are counted.

    A type is kept when `running + its vcpus <= quota` (Req 4.3). A quota of 0 is a `CloudError`
    pointing to the Setup_Guide for asking AWS to raise it (Req 4.4). No candidate fitting is a
    `CloudError` giving the quota, the running total and the smallest candidate's vCPUs, before
    anything is uploaded or launched (Req 4.5).
    """
    if quota == 0:
        raise CloudError(
            "This AWS account has no GPU quota: the limit on Running On-Demand G and VT instances is "
            "0, so no GPU machine can start. Ask AWS to raise the quota as docs/aws-setup.md explains, "
            "then run the command again.")
    kept = [name for name in candidates if running + cc.spec(name).vcpus <= quota]
    if not kept:
        smallest = min((cc.spec(name).vcpus for name in candidates), default=0)
        raise CloudError(
            f"The GPU quota is {quota:g} vCPUs and our running machines already use {running}, so "
            f"there is no room for the smallest fitting type, which needs {smallest} vCPUs. Wait for a "
            "machine to finish, stop one with the stop-all command, or ask AWS to raise the quota, "
            "then run the command again.")
    return kept


def launch_plan(admitted: Sequence[str], aws: Aws, outputs: StackOutputs,
                cc: CloudConfig) -> list[tuple[str, str, str]]:
    """The ordered (instance_type, subnet_id, zone) attempts for the launch loop.

    For each admitted type in order, each zone that both offers the type and has a default subnet of
    the Default_VPC, so the loop tries a type's zones before moving to the next type (Req 5.1).
    """
    subnets = aws.default_subnets(outputs.vpc_id)  # zone -> subnet id
    plan: list[tuple[str, str, str]] = []
    for instance_type in admitted:
        for zone in aws.type_zones(instance_type):
            subnet = subnets.get(zone)
            if subnet is not None:
                plan.append((instance_type, subnet, zone))
    return plan
