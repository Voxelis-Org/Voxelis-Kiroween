"""Spend_Lines and the money quote: what a run is estimated to cost.

A Spend_Line is one JSON object recording what a slice of a machine's time cost: the Job_Id, what
the line is for (a Step, the machine's time outside the Steps, an Image_Build or a Smoke_Test), the
machine type, the seconds and the estimated US-dollar cost, always labelled an estimate (Req 12.3).

The quote helpers the Start_Command uses before the Spend_Guard are pure: `estimated_cost` turns
seconds and a price into dollars, `rough_cost` is the money quote for a whole run (machine time, the
public IPv4 address and the root volume), and `spend_line` builds one line. `append_spend_line` is
the machine-side write: it appends a line to the Job's `out/spend.jsonl` in the bucket through the
`Aws` seam, stamping the line's id from what is already there. `merge_spend_file` is the laptop-side
join: the Status and Download commands merge a Job's lines into `data/spend.jsonl` without duplicates.

This module imports no boto3 and no GPU package, so it loads on a laptop.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from cloud.config import CloudConfig, MachineSpec

if TYPE_CHECKING:
    from cloud.aws import Aws

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


def append_spend_line(aws: Aws, job_id: str, line: Mapping[str, Any]) -> None:
    """Append one Spend_Line to the Job's `out/spend.jsonl` in the bucket (machine side, Req 12.1, 12.2).

    Read `jobs/<job_id>/out/spend.jsonl` (a missing file counts as empty), stamp the line's `id` as
    `"<job_id>:<label>:<n>"`, where `n` is how many lines with that `(job_id, label)` are already
    there, append it and put the whole file back. So re-reading and re-merging the file stays
    idempotent, because the ids don't move, while a genuine rerun of a Step writes a fresh line at a
    higher `n`. The `id` is set here, from the file, since only this write knows the current count.
    """
    key = f"jobs/{job_id}/out/spend.jsonl"
    text = aws.s3_get_text(key) if aws.s3_exists(key) else ""
    label = str(line["label"])
    index = sum(1 for earlier in _read_jsonl(text)
                if earlier.get("job_id") == job_id and earlier.get("label") == label)
    stamped = {**line, "id": f"{job_id}:{label}:{index}"}
    rendered = json.dumps(stamped, ensure_ascii=True, allow_nan=False)
    if text and not text.endswith("\n"):
        text += "\n"
    aws.s3_put_text(key, f"{text}{rendered}\n")


def merge_spend_file(local_path: Path, new_lines: Iterable[Mapping[str, Any]]) -> int:
    """Merge `new_lines` into the laptop's `data/spend.jsonl` (the `local_path`), returning how many
    were added (laptop side, Req 12.4, 12.5).

    The Status_Command and the Download_Command read a Job's `out/spend.jsonl` from the bucket and
    merge its lines here. A line whose `id` is already in the file, or already added earlier in this
    same call, is skipped, so re-merging the same lines adds nothing and the file never holds two
    lines with the same `id`: the merge is idempotent (Req 12.4). It writes only to `local_path`,
    which the caller keeps under `data/`, which git ignores, so no Spend_Line lands in the repo
    (Req 12.5). The file and its folder are made when a line is added; an all-duplicate merge, adding
    nothing, leaves the file untouched.
    """
    path = Path(local_path)
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    seen = {earlier.get("id") for earlier in _read_jsonl(text)}
    added: list[str] = []
    for line in new_lines:
        if line["id"] in seen:
            continue
        seen.add(line["id"])
        added.append(json.dumps(dict(line), ensure_ascii=True, allow_nan=False))
    if not added:
        return 0
    if text and not text.endswith("\n"):
        text += "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "".join(f"{rendered}\n" for rendered in added),
                    encoding="utf-8", newline="\n")
    return len(added)


def _read_jsonl(text: str) -> list[Any]:
    """The JSON objects in a JSON-Lines string, one per non-blank line."""
    return [json.loads(part) for part in text.splitlines() if part.strip()]
