"""Spend_Lines and the money quote: what a run is estimated to cost.

A Spend_Line is one JSON object recording what a slice of a machine's time cost: the Job_Id, what
the line is for (a Step, the machine's time outside the Steps, an Image_Build or a Smoke_Test), the
machine type, the seconds and the estimated US-dollar cost, always labelled an estimate (Req 12.3).

This task writes the pure quote helpers the Start_Command uses before the Spend_Guard:
`estimated_cost` turns seconds and a price into dollars, `rough_cost` is the money quote for a whole
run (machine time, the public IPv4 address and the root volume), and `spend_line` builds one line.
The machine-side append and the laptop-side merge join this module in later tasks.

This module imports no boto3 and no GPU package, so it loads on a laptop.
"""

from __future__ import annotations

from typing import Any

from cloud.config import CloudConfig, MachineSpec

LABEL_MACHINE = "machine"        # the machine's time outside the Steps
LABEL_IMAGE = "image-build"
LABEL_SMOKE = "smoke-test"


def estimated_cost(seconds: float, price_per_hour_usd: float) -> float:
    """The estimated US-dollar cost of `seconds` at `price_per_hour_usd`, rounded to cents."""
    return round(seconds / 3600 * price_per_hour_usd, 2)


def rough_cost(minutes: float, spec: MachineSpec, cc: CloudConfig) -> float:
    """The money quote for `minutes` on `spec` (Req 6.3): the machine and its public IPv4 address,
    both billed by the hour, plus the gp3 root volume, billed by the GB-month over a 730-hour month.
    """
    machine_and_ip = minutes / 60 * (spec.price_per_hour_usd + cc.ipv4_usd_per_hour)
    storage = cc.root_gb * cc.gp3_usd_per_gb_month * minutes / (60 * 730)
    return machine_and_ip + storage


def spend_line(*, job_id: str, label: str, instance_type: str, seconds: float,
               price_per_hour_usd: float, index: int) -> dict[str, Any]:
    """One Spend_Line, always labelled an estimate (Req 12.3).

    The `id` is `"<job_id>:<label>:<index>"`, so merging the same line twice is idempotent while a
    genuine rerun, with a higher `index`, writes a fresh line.
    """
    return {
        "id": f"{job_id}:{label}:{index}",
        "job_id": job_id,
        "label": label,
        "instance_type": instance_type,
        "seconds": seconds,
        "estimated_usd": estimated_cost(seconds, price_per_hour_usd),
        "estimate": True,
    }
