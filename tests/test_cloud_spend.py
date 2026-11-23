"""Unit tests for the Spend_Line builders and the laptop Spend_File merge, cloud/spend_file.py
(cloud-jobs task 8.5).

These cover the pure, laptop-side helpers on the CPU, with no network and no AWS. Property 9, the
Hypothesis property that the merge is idempotent and deduplicated, is task 8.6, in this same file.

- `spend_line` builds one Spend_Line that carries the id, label, instance_type, seconds, the
  estimated dollar cost (seconds / 3600 * price, to cents) and `estimate = True` (Req 12.3).
- `rough_cost` is the money quote: it adds the public-IPv4 term and the gp3 root-volume term to the
  machine's price over the given minutes (Req 6.3).
- `merge_spend_file` writes only to the given `data/` path and dedups by `id` (Req 12.5, 12.4).

Config values come from the shared `cloud_config()` helper, so no config.toml is read.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from cloud.spend_file import LABEL_MACHINE, merge_spend_file, rough_cost, spend_line

from fake_aws import cloud_config

JOB_ID = "20261119-142530-7f3a9c"  # a Job_Id-shaped string, as make_job mints
CC = cloud_config()                # a valid CloudConfig from the real [cloud] shape, built once


# --- spend_line -------------------------------------------------------------

def test_spend_line_carries_every_field() -> None:
    """A Spend_Line carries its id, label, type, seconds, cost and estimate flag (Req 12.3)."""
    price = CC.spec("g5.8xlarge").price_per_hour_usd
    line = spend_line(job_id=JOB_ID, label=LABEL_MACHINE, instance_type="g5.8xlarge",
                      seconds=3600.0, price_per_hour_usd=price, index=0)

    assert line["id"] == f"{JOB_ID}:{LABEL_MACHINE}:0"
    assert line["label"] == LABEL_MACHINE
    assert line["instance_type"] == "g5.8xlarge"
    assert line["seconds"] == 3600.0
    assert line["estimated_usd"] == round(3600.0 / 3600 * price, 2)
    assert line["estimate"] is True


def test_spend_line_estimated_usd_is_rounded_to_cents() -> None:
    """estimated_usd is seconds / 3600 * price rounded to cents, even for a part-hour (Req 12.3)."""
    price = CC.spec("g5.xlarge").price_per_hour_usd  # 1.006, so the cents rounding bites
    seconds = 100.0
    line = spend_line(job_id=JOB_ID, label=LABEL_MACHINE, instance_type="g5.xlarge",
                      seconds=seconds, price_per_hour_usd=price, index=0)

    assert line["estimated_usd"] == round(seconds / 3600 * price, 2)


# --- rough_cost -------------------------------------------------------------

@pytest.mark.parametrize("instance_type, minutes", [("g5.8xlarge", 30.0), ("g6e.4xlarge", 120.0)])
def test_rough_cost_adds_ipv4_and_storage_to_machine(instance_type: str, minutes: float) -> None:
    """rough_cost is the machine price over the minutes plus the IPv4 and gp3 root terms (Req 6.3)."""
    spec = CC.spec(instance_type)
    machine = minutes / 60 * spec.price_per_hour_usd
    ipv4 = minutes / 60 * CC.ipv4_usd_per_hour
    storage = CC.root_gb * CC.gp3_usd_per_gb_month * minutes / (60 * 730)

    # Both added terms are real costs, so the quote is strictly more than the machine alone.
    assert ipv4 > 0
    assert storage > 0
    assert rough_cost(minutes, spec, CC) == pytest.approx(machine + ipv4 + storage)
    assert rough_cost(minutes, spec, CC) > machine


# --- merge_spend_file -------------------------------------------------------

def _line(index: int, label: str = LABEL_MACHINE) -> dict[str, Any]:
    """A Spend_Line with a distinct id per (label, index), built with the real builder."""
    price = CC.spec("g5.8xlarge").price_per_hour_usd
    return spend_line(job_id=JOB_ID, label=label, instance_type="g5.8xlarge",
                      seconds=600.0, price_per_hour_usd=price, index=index)


def _ids(path: Path) -> list[str]:
    """The ids in a Spend_File, in order."""
    return [json.loads(part)["id"]
            for part in path.read_text(encoding="utf-8").splitlines() if part.strip()]


def test_merge_writes_only_under_the_given_data_path(tmp_path: Path) -> None:
    """merge_spend_file writes exactly its data/ path and no file anywhere else (Req 12.5)."""
    local_path = tmp_path / "data" / "spend.jsonl"  # the data/ parent is made by the merge
    before = {p for p in tmp_path.rglob("*") if p.is_file()}

    added = merge_spend_file(local_path, [_line(0), _line(1)])

    after = {p for p in tmp_path.rglob("*") if p.is_file()}
    assert added == 2
    assert local_path.exists()
    # The only file the merge created is the given path, which sits under data/.
    assert after - before == {local_path}


def test_merge_dedups_by_id(tmp_path: Path) -> None:
    """A line whose id is already present is skipped, so the file holds each id once (Req 12.4)."""
    local_path = tmp_path / "data" / "spend.jsonl"
    a, b, c = _line(0), _line(1), _line(2)

    assert merge_spend_file(local_path, [a, b]) == 2
    # b repeats and c is new, so only c is added.
    assert merge_spend_file(local_path, [b, c]) == 1
    assert _ids(local_path) == [a["id"], b["id"], c["id"]]

    # Re-merging lines already present adds nothing and leaves the file as it was.
    assert merge_spend_file(local_path, [a, b, c]) == 0
    assert _ids(local_path) == [a["id"], b["id"], c["id"]]


# --- Property 9: Merging Spend_Lines is idempotent and deduplicated (Req 12.4) --------------------

# One index maps to one Spend_Line, built with the real `_line` builder above, so each index gives a
# line with a distinct id (JOB_ID:machine:<index>) and the same index always gives the same line.
# The range is small next to the list sizes, so a drawn pair of lists shares many ids (the overlap
# the merge must dedup) while still leaving ids in only one of them (the distinct part of the union).
LINE_INDICES = st.integers(min_value=0, max_value=20)


@settings(deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    start_indices=st.lists(LINE_INDICES, unique=True, max_size=12),
    new_indices=st.lists(LINE_INDICES, max_size=12),
)
def test_property_9_merging_spend_lines_is_idempotent_and_deduplicated(
        start_indices: list[int], new_indices: list[int], tmp_path: Path) -> None:
    """Feature: cloud-jobs, Property 9: Merging Spend_Lines is idempotent and deduplicated

    **Validates: Requirements 12.4**

    For any starting Spend_File and any list of Spend_Lines, `merge_spend_file` leaves the file
    holding exactly the union by `id` of the lines already there and the new ones, so merging the
    same lines again adds nothing and the file never holds two lines with the same `id`.
    """
    # The starting file holds distinct ids, as a real Spend_File does; the new-line list is arbitrary
    # and may repeat an id within itself or overlap the starting ids.
    start_lines = [_line(index) for index in start_indices]
    new_lines = [_line(index) for index in new_indices]
    start_ids = {line["id"] for line in start_lines}
    new_ids = {line["id"] for line in new_lines}

    # A clean slate for this example: tmp_path is function-scoped, so Hypothesis reuses it across
    # examples. Drop any Spend_File an earlier example left behind.
    local_path = tmp_path / "data" / "spend.jsonl"
    if local_path.exists():
        local_path.unlink()

    # Write the starting Spend_File directly, independently of the merge, or leave it absent when
    # there is nothing to start with, so the merge also meets the "no file yet" case.
    if start_lines:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_text(
            "".join(json.dumps(line, ensure_ascii=True) + "\n" for line in start_lines),
            encoding="utf-8", newline="\n")

    # First merge: exactly the new ids not already present are added.
    added = merge_spend_file(local_path, new_lines)
    assert added == len(new_ids - start_ids)

    # The file now holds exactly the union by id of the starting and new lines, each id once.
    ids_after = _ids(local_path) if local_path.exists() else []
    assert set(ids_after) == start_ids | new_ids
    assert len(ids_after) == len(set(ids_after))

    # Idempotent: merging the same lines again adds nothing and changes no byte of the file.
    before = local_path.read_bytes() if local_path.exists() else None
    assert merge_spend_file(local_path, new_lines) == 0
    after = local_path.read_bytes() if local_path.exists() else None
    assert after == before
