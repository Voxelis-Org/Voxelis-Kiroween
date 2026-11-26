"""Unit tests for the machine selection, cloud/select.py (cloud-jobs task 3.2).

Four behaviours, as the task describes, all over the pure functions with no network and no AWS:

- `admit` with a GPU quota of 0 is a `CloudError` pointing to the Setup_Guide, so an account that
  has never asked AWS for quota fails in plain words rather than later with `VcpuLimitExceeded`
  (Req 4.4).
- `select_candidates` with a need no Allowed_Type meets is a `CloudError` that names the unmet
  memory or GPU-memory figure, before anything is uploaded or launched (Req 3.4).
- `step_needs_over` reads each Step's `min_vcpus`, `min_memory_gb` and `min_gpu_memory_gb` from that
  Step's own section for the Job's Preset (through `splatpipe.config.Config.step_settings`), takes
  the largest in each dimension, and counts a key a Step omits as 0 (Req 2.7).
- `launch_plan` lays out the attempts type by type and, within a type, zone by zone, so the launch
  loop tries a type's zones before moving on (Req 5.1); a `FakeAws` supplies the known offerings and
  default subnets.
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from splatpipe.config import Config

from cloud.errors import CloudError
from cloud.select import Needs, admit, launch_plan, select_candidates, step_needs_over
from cloud.stack import StackOutputs

from fake_aws import FakeAws, cloud_config


# admit (Req 4.4)

def test_admit_with_zero_quota_points_to_the_setup_guide() -> None:
    cc = cloud_config()
    with pytest.raises(CloudError) as caught:
        admit(["g5.8xlarge"], running=0, quota=0, cc=cc)
    # The refusal sends the reader to docs/aws-setup.md to ask AWS to raise the quota.
    assert "aws-setup.md" in str(caught.value)


# select_candidates (Req 3.4)

# Each case asks the "default" preset (whose types top out at 128 GB RAM and 24 GB GPU memory) for a
# need no type on its Fallback_List meets, and names the figure the refusal must quote.
UNMET_NEEDS = [
    pytest.param(Needs(min_memory_gb=256, min_gpu_memory_gb=0, min_vcpus=0), "256", id="memory"),
    pytest.param(Needs(min_memory_gb=0, min_gpu_memory_gb=96, min_vcpus=0), "96", id="gpu-memory"),
]


@pytest.mark.parametrize(("needs", "figure"), UNMET_NEEDS)
def test_select_candidates_names_the_unmet_need(needs: Needs, figure: str) -> None:
    cc = cloud_config()
    with pytest.raises(CloudError) as caught:
        select_candidates("default", needs, cc)
    assert figure in str(caught.value)


# step_needs_over (Req 2.7)

def _needs_config() -> Config:
    """A tiny pipeline `Config` with two Step sections that carry some min_* keys and omit others.

    `frames` asks for memory and vCPUs (no GPU-memory key); `train` asks for memory and GPU memory
    (no vCPUs key). The quality Preset raises each Step's heaviest dimension, so the test can also
    show that `step_needs_over` reads the Preset it is given.
    """
    data = {
        "frames": {
            "fast": {"min_memory_gb": 16, "min_vcpus": 16},
            "default": {"min_memory_gb": 16, "min_vcpus": 16},
            "quality": {"min_memory_gb": 24, "min_vcpus": 16},
        },
        "train": {
            "fast": {"min_memory_gb": 32, "min_gpu_memory_gb": 24},
            "default": {"min_memory_gb": 32, "min_gpu_memory_gb": 24},
            "quality": {"min_memory_gb": 32, "min_gpu_memory_gb": 48},
        },
    }
    return Config(data, source="test config")


def test_step_needs_over_reads_each_section_for_the_preset_missing_is_zero() -> None:
    config = _needs_config()
    # One Step: the dimension it omits (GPU memory) counts as 0, not a need.
    assert step_needs_over(["frames"], config, "fast") == Needs(
        min_memory_gb=16.0, min_gpu_memory_gb=0.0, min_vcpus=16)
    # Two Steps: the largest need in each dimension, each Step's omitted key counting as 0.
    assert step_needs_over(["frames", "train"], config, "fast") == Needs(
        min_memory_gb=32.0, min_gpu_memory_gb=24.0, min_vcpus=16)
    # The Preset is read from each Step's own section: quality raises the GPU-memory need to 48.
    assert step_needs_over(["frames", "train"], config, "quality") == Needs(
        min_memory_gb=32.0, min_gpu_memory_gb=48.0, min_vcpus=16)


# launch_plan (Req 5.1)

# A StackOutputs with placeholder values: launch_plan reads only vpc_id, and the FakeAws returns the
# seeded subnets whatever vpc id it is passed.
OUTPUTS = StackOutputs(
    bucket="bucket", job_role_arn="job-role", job_instance_profile_arn="job-profile",
    runner_role_arn="runner-role", security_group_id="sg", vpc_id="vpc", group_name="group")


def test_launch_plan_orders_types_then_zones() -> None:
    aws = FakeAws(
        offerings={"g5.8xlarge": ["us-east-1a", "us-east-1b"], "g5.4xlarge": ["us-east-1a"]},
        subnets={"us-east-1a": "subnet-a", "us-east-1b": "subnet-b"})
    plan = launch_plan(["g5.8xlarge", "g5.4xlarge"], aws, OUTPUTS, cloud_config())
    # Both zones of the first type come before the second type: types first, then zones.
    assert plan == [
        ("g5.8xlarge", "subnet-a", "us-east-1a"),
        ("g5.8xlarge", "subnet-b", "us-east-1b"),
        ("g5.4xlarge", "subnet-a", "us-east-1a"),
    ]


# Property 3: Selection keeps the right types in order (Req 3.1, 3.2, 3.3, 4.2, 4.3, 4.5)

# The preset whose Fallback_List the property drives. cloud_config needs all three presets, so the
# other two carry a trivial valid fallback; only this one is read by select_candidates here.
PRESET = "default"

# Sizes a synthetic type may have for memory_gb and gpu_memory_gb (the Config_Loader needs each
# above 0), and the realistic G-instance vCPU counts.
SPEC_DIMS = [1, 2, 4, 8, 16, 24, 32, 48, 64, 128]
VCPU_VALUES = [1, 2, 4, 8, 16, 32]

# What a Needs dimension may ask for: 0 (no need), the same sizes a type may have (so the >= / <=
# boundaries land exactly), and 256, which no type meets (so the hard-needs filter empties and the
# call raises).
NEED_DIMS = [0, 1, 2, 4, 8, 16, 24, 32, 48, 64, 128, 256]


@st.composite
def selection_cases(draw: st.DrawFn) -> tuple[dict, list[str], Needs, int, int]:
    """One selection scenario: the synthetic Machine_Specs, the preset's Fallback_List over them, an
    aggregate Needs, the running-vCPU total and the GPU quota (whole vCPUs, as a real quota is).

    min_vcpus is drawn past the largest type, so the test also covers the soft-need clause: a type
    that clears memory, GPU memory and the quota must stay however large min_vcpus is. The quota is
    over-sampled around running + a fallback type's vCPUs, so the admit boundary
    (running + vcpus <= quota) is hit exactly as well as spanned.
    """
    names = [f"t{index}" for index in range(draw(st.integers(min_value=1, max_value=5)))]
    types = {
        name: {
            "vcpus": draw(st.sampled_from(VCPU_VALUES)),
            "memory_gb": draw(st.sampled_from(SPEC_DIMS)),
            "gpu_memory_gb": draw(st.sampled_from(SPEC_DIMS)),
            "price_per_hour_usd": 1.0,
            "price_date": "2026-11-21",
        }
        for name in names
    }
    fallback = draw(st.lists(st.sampled_from(names), min_size=1, unique=True))
    needs = Needs(
        min_memory_gb=draw(st.sampled_from(NEED_DIMS)),
        min_gpu_memory_gb=draw(st.sampled_from(NEED_DIMS)),
        min_vcpus=draw(st.integers(min_value=0, max_value=64)),
    )
    running = draw(st.integers(min_value=0, max_value=40))
    boundary = draw(st.sampled_from([running + types[name]["vcpus"] for name in fallback]))
    quota = draw(st.one_of(
        st.sampled_from([0, boundary - 1, boundary, boundary + 1]),
        st.integers(min_value=0, max_value=80)))
    return types, fallback, needs, running, quota


@settings(deadline=None)
@given(selection_cases())
def test_property_3_selection_keeps_the_right_types_in_order(
        case: tuple[dict, list[str], Needs, int, int]) -> None:
    """Feature: cloud-jobs, Property 3: Selection keeps the right types in order

    **Validates: Requirements 3.1, 3.2, 3.3, 4.2, 4.3, 4.5**
    """
    types, fallback, needs, running, quota = case
    presets = {
        "fast": {"fallback": fallback[:1], "expected_minutes": 30, "cap_minutes": 120},
        PRESET: {"fallback": fallback, "expected_minutes": 60, "cap_minutes": 180},
        "quality": {"fallback": fallback[:1], "expected_minutes": 120, "cap_minutes": 360},
    }
    primary = fallback[0]
    cc = cloud_config(types=types, presets=presets, smoke_instance_type=primary,
                      image={"fallback": [primary], "expected_minutes": 75, "cap_minutes": 120})

    # The rule, stated once and directly over the Fallback_List in order: a type belongs exactly
    # when it clears the hard memory and GPU-memory needs and still fits the quota once the running
    # machines are counted. min_vcpus never appears, which is the soft-need clause (Req 3.1, 3.2,
    # 3.3, 4.2, 4.3). This is a sublist of the Fallback_List in its original order.
    expected = [name for name in fallback
                if cc.spec(name).memory_gb >= needs.min_memory_gb
                and cc.spec(name).gpu_memory_gb >= needs.min_gpu_memory_gb
                and running + cc.spec(name).vcpus <= quota]

    try:
        result = admit(select_candidates(PRESET, needs, cc), running, quota, cc)
    except CloudError:
        result = []  # no type qualifies: the call raises, which is the empty result (Req 3.4, 4.5)

    assert result == expected
