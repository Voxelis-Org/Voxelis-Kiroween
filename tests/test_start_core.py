"""The Start_Core split in cloud/start.py: launch_job asks the Spend_Guard before it places anything,
the module-global check_spend reaches both callers, and the worst case covers the quote."""

from __future__ import annotations

from pathlib import Path

import pytest

import cloud.start
from cloud.config import load_cloud_config
from cloud.spend import SpendRefused
from cloud.start import (Code, check_image, choose_machine, launch_job, require_step_list,
                         worst_case_cost)
from cloud.spend_file import rough_cost
from tests.website_helpers import REAL_CONFIG, PACKAGE_HASH, UV_HASH, ready_fake, real_config

CC = load_cloud_config(REAL_CONFIG)


def parts(tmp_path: Path):
    fake = ready_fake()
    archive = tmp_path / "code.tar"
    archive.write_bytes(b"tar")
    code = Code("a" * 40, archive, UV_HASH, PACKAGE_HASH)
    config = real_config()
    gate = check_image(fake, config, code)
    choice = choose_machine(fake, gate, require_step_list(), "fast", config, CC)
    return fake, code, config, gate, choice


def test_launch_job_asks_the_spend_guard_first(tmp_path: Path, monkeypatch) -> None:
    fake, code, config, gate, choice = parts(tmp_path)
    order: list[str] = []
    monkeypatch.setattr(cloud.start, "check_spend", lambda **kw: order.append("check_spend"))
    launched = launch_job(fake, gate, choice, "20261201-120000-abcdef", code,
                          lambda prefix, key: order.append(f"place {prefix} {key}"), config, CC,
                          {"APPROVED_SPEND": "1"})
    assert order == ["check_spend", "place jobs/20261201-120000-abcdef/input "
                                    "jobs/20261201-120000-abcdef/input/code.tar"]
    assert (launched.instance_type, launched.zone) == ("g5.8xlarge", "us-east-1a")


def test_a_spend_refusal_places_and_launches_nothing(tmp_path: Path) -> None:
    fake, code, config, gate, choice = parts(tmp_path)
    placed: list[str] = []
    with pytest.raises(SpendRefused):
        launch_job(fake, gate, choice, "20261201-120000-abcdef", code,
                   lambda prefix, key: placed.append(prefix), config, CC, {})
    assert placed == [] and "run_instances" not in [op for op, _ in fake.calls]


def test_the_choice_carries_the_quote_strings(tmp_path: Path) -> None:
    *_, choice = parts(tmp_path)
    assert choice.admitted[0] == "g5.8xlarge" and choice.attempts[0][2] == "us-east-1a"
    assert choice.expected_time == "about 30 min, up to the 120 min time cap"
    assert choice.rough_cost.startswith("about $") and "worst case about $4.94" in choice.rough_cost


@pytest.mark.parametrize("preset", ["fast", "default", "quality"])
@pytest.mark.parametrize("margin", [0, 5, 17.5])
def test_the_worst_case_covers_the_quotes_worst_case(preset: str, margin: float) -> None:
    plan = CC.preset_plan(preset)
    quoted = max(rough_cost(plan.cap_minutes, CC.spec(name), CC) for name in plan.fallback)
    assert worst_case_cost(preset, CC, margin) >= quoted
