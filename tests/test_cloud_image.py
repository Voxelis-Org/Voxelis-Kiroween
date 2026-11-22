"""Unit and structural tests for the Image_Build, cloud/image.py (cloud-jobs task 7.4).

Three kinds of check, all on the CPU with no network, no AWS account and no GPU:

- Structural: `install_script` is the builder's whole install, asserted as text and never run here
  (the real install compiles CUDA code and fetches gigabytes). The task pins what it must hold: the
  COLMAP OFF flags that keep SiftGPU, CGAL and LSD out, `TORCH_CUDA_ARCH_LIST="8.6;8.9"` for the
  A10G/L4/L40S, `npm ci`, gsplat 1.5.3, fused-ssim, the MoGe-2 weights, and a `cloud-init clean`
  before the builder is imaged (Req 9.2, 9.3, 9.4).

- With the `FakeAws`: `build_image` is a Paid_Entry_Point, so it asks the Spend_Guard first and does
  nothing when `APPROVED_SPEND` is unset (Req 9.10). On the approved path it reads the Base_AMI from
  SSM, stops the builder before `create_image` (an API stop never triggers terminate-on-shutdown),
  tags the image with the Project_Tag and the two lock hashes, waits for `available`, terminates the
  builder and records an `image-build` Spend_Line under `data/` (Req 9.1, 9.7, 9.8, 9.9, 12.2). Its
  launch request carries Property 4's required settings (Req 8.9), checked with the Property-4 helper
  and constants reused from `test_cloud_launch`.

- Licence_List: one row for each Machine_Image tool, weight and package, the COLMAP row naming its
  OFF flags (Req 14.5).

The `FakeAws` is subclassed only to record each `run_instances` request (so the launch request can
be read back) and to have the builder's console report the install finished at once, so the console
wait returns on the CPU; the real console polling is a real-machine path. Nothing under `splatpipe/`
or any other file is touched, and no test sets a real `APPROVED_SPEND`: each build passes it as an
in-process `environ` dict straight to the Spend_Guard, the way the Start_Command's tests do.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from splatpipe.config import Config

from cloud.config import CloudConfig
from cloud.image import INSTALL_DONE_MARKER, build_image, install_script
from cloud.launch import JOB_ID_TAG
from cloud.repo import lock_hashes
from cloud.spend import SpendRefused
from cloud.spend_file import LABEL_IMAGE
from cloud.stack import StackOutputs

from fake_aws import FakeAws, cloud_config, git_repo
# Reuse Property 4's own helper and constants, so "the launch request satisfies Property 4's shape"
# is checked by the same assertions task 4 uses, not a re-derived copy.
from test_cloud_launch import FORBIDDEN_KEYS, PROJECT_TAG_DICT, _tags_for

# The COLMAP build flags that keep SiftGPU (non-commercial), CGAL (GPL) and LSD (AGPL) out, as one
# string, so both the install_script and the COLMAP Licence_List row are checked for it verbatim.
OFF_FLAGS = "-DCUDA_ENABLED=OFF -DOPENGL_ENABLED=OFF -DGUI_ENABLED=OFF -DCGAL_ENABLED=OFF -DLSD_ENABLED=OFF"

ROOT = Path(__file__).resolve().parents[1]  # this file is in tests/, at the repo root
LICENCE_LIST = ROOT / "docs" / "third-party-licences.md"

# A real-shaped base AMI id for the seeded SSM parameter; build_image passes it through as the
# builder's ImageId. No value in this file holds a 12-digit account id.
BASE_AMI = "ami-0base00000000000"

# The seven Stack_Outputs build_image is handed. Only the instance-profile ARN, the security group
# and the VPC id are used (the launch request and the default-subnet lookup); none holds an account
# id, since this file doesn't repeat the aws/stack redaction check.
OUTPUTS = StackOutputs(
    bucket="voxelis-kiroween-jobs",
    job_role_arn="arn:aws:iam::role/voxelis-job",
    job_instance_profile_arn="arn:aws:iam::instance-profile/voxelis-job",
    runner_role_arn="arn:aws:iam::role/voxelis-runner",
    security_group_id="sg-fake",
    vpc_id="vpc-fake",
    group_name="voxelis-kiroween",
)

# The one zone the fake offers the builder type in, and its default subnet, so launch_plan lands a
# single attempt on that subnet.
ZONE = "us-east-1a"
SUBNET = "subnet-a"


class _BuilderAws(FakeAws):
    """A `FakeAws` for the Image_Build: records each `run_instances` request and always reports the
    install finished on the builder's console.

    build_image builds its launch request inside a closure and hands it to `launch_with_fallback`,
    which the base `FakeAws` doesn't keep, so this records each request for the Property-4 shape
    check. The builder runs on the Job_Role and signals "install done" by printing the marker to the
    console; on the CPU there is no real builder, so `console_output` returns the marker at once and
    `build_image`'s wait returns on its first poll. The call is still recorded, so the ordered
    timeline stays accurate.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.run_requests: list[dict[str, Any]] = []

    def run_instances(self, request: Any) -> Any:
        self.run_requests.append(dict(request))
        return super().run_instances(request)

    def console_output(self, instance_id: str) -> str:
        super().console_output(instance_id)  # keep the call on the timeline
        return INSTALL_DONE_MARKER


def _config() -> Config:
    """A pipeline Config holding just the `[aws]` the Image_Build reads: the Project_Tag build_image
    tags the image with, and a stack name (unused by build_image, read only by its `main`)."""
    return Config(
        {"aws": {"stack_name": "voxelis-kiroween", "project_tag": "project=voxelis-kiroween"}},
        source="test config")


def _fake_aws(cc: CloudConfig) -> _BuilderAws:
    """A healthy `_BuilderAws`: the Base_AMI parameter resolves, the builder type is offered a zone
    with a default subnet, and the G_Quota admits it."""
    return _BuilderAws(ssm={cc.base_ami_parameter: BASE_AMI},
                       offerings={cc.image_builder_type: [ZONE]},
                       subnets={ZONE: SUBNET}, quota=32.0)


def _last_spend_line(repo_root: Path) -> dict[str, Any]:
    """The last Spend_Line build_image appended to `<repo_root>/data/spend.jsonl`."""
    lines = [line for line in (repo_root / "data" / "spend.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip()]
    return json.loads(lines[-1])


# Structural: install_script pins every tool, weight and package (Req 9.2, 9.3, 9.4)

def test_install_script_holds_the_pinned_build_tools() -> None:
    script = install_script(cloud_config())
    # COLMAP from source with SiftGPU, CGAL and LSD all off.
    assert OFF_FLAGS in script
    # gsplat and fused-ssim compiled for the A10G (8.6) and the L4/L40S (8.9).
    assert 'TORCH_CUDA_ARCH_LIST="8.6;8.9"' in script
    assert "gsplat==1.5.3" in script
    assert "fused-ssim" in script
    # The MoGe-2 depth-model weights (ViT-B), fetched into the image. This must match the
    # licence-cleared weights named in the Licence_List (`Ruicheng/moge-2-vitb-normal`), so code and
    # the licence row can't drift apart.
    assert "Ruicheng/moge-2-vitb-normal" in script
    # Node packages installed reproducibly from the committed lockfile.
    assert "npm ci" in script


def test_install_script_cleans_cloud_init_before_imaging() -> None:
    script = install_script(cloud_config())
    # cloud-init clean wipes the builder's first-boot state so the image boots fresh. It runs before
    # the builder prints the done marker, which is what the Image_Build_Command waits for before it
    # stops and images the builder, so the clean happens before imaging.
    assert "cloud-init clean" in script
    assert script.index("cloud-init clean") < script.index(INSTALL_DONE_MARKER)


# With the FakeAws: build_image asks the Spend_Guard first and builds nothing when it says no.

def test_build_image_refuses_without_approved_spend(tmp_path: Path) -> None:
    cc = cloud_config()
    aws = _fake_aws(cc)
    with pytest.raises(SpendRefused):
        build_image(aws=aws, config=_config(), cloud_config=cc, outputs=OUTPUTS,
                    repo_root=tmp_path, environ={})  # APPROVED_SPEND unset
    # The Spend_Guard comes first, so nothing touched AWS, no builder launched, no image made and no
    # Spend_Line written.
    assert aws.calls == []
    assert aws.run_requests == []
    assert aws.images == []
    assert not (tmp_path / "data" / "spend.jsonl").exists()


# With the FakeAws: the approved build reads SSM, stops before imaging, tags, waits, terminates, records.

def test_build_image_reads_ssm_stops_before_imaging_tags_waits_terminates_records(
        tmp_path: Path) -> None:
    root = git_repo(tmp_path)
    cc = cloud_config()
    aws = _fake_aws(cc)

    image_id = build_image(aws=aws, config=_config(), cloud_config=cc, outputs=OUTPUTS,
                           repo_root=root, environ={"APPROVED_SPEND": "1"})

    ops = [op for op, _ in aws.calls]
    # The first AWS touch after the Spend_Guard is the Base_AMI read from SSM (Req 9.1).
    assert aws.calls[0] == ("ssm_parameter", cc.base_ami_parameter)
    # The builder is stopped through the API before create_image, and create_image before terminate,
    # so imaging runs on a stopped builder and the stop never triggers terminate-on-shutdown (Req 9.9).
    assert ops.index("stop_instance") < ops.index("create_image") < ops.index("terminate_instances")
    # The image is waited to `available` after it is created (Req 9.9).
    assert ops.index("create_image") < ops.index("wait_image_available")

    # Exactly one image is made and returned, tagged with the Project_Tag and the two lock hashes in
    # the one call (the fake models that single tagging call; the real one tags the image and its
    # snapshots) (Req 9.7, 9.8).
    assert len(aws.images) == 1
    created = aws.images[0]
    assert created.id == image_id
    uv_hash, package_hash = lock_hashes(cwd=root)
    assert created.tags == {"project": "voxelis-kiroween",
                            "uv_lock_hash": uv_hash, "package_lock_hash": package_hash}

    # The builder is terminated and ends in the terminated state, so no machine is left running (Req 9.9).
    builder_id = aws.instances[0].id
    assert ("terminate_instances", builder_id) in aws.calls
    assert aws.instances[0].state == "terminated"

    # An image-build Spend_Line is appended under data/, keyed to the new image id and an estimate
    # (Req 12.2, 12.3).
    line = _last_spend_line(root)
    assert line["label"] == LABEL_IMAGE == "image-build"
    assert line["job_id"] == image_id
    assert line["instance_type"] == cc.image_builder_type
    assert line["estimate"] is True
    assert line["id"].startswith(f"{image_id}:image-build:")


# With the FakeAws: the launch request carries Property 4's required settings (Req 8.9).

def test_build_image_launch_request_satisfies_property_4_shape(tmp_path: Path) -> None:
    root = git_repo(tmp_path)
    cc = cloud_config()
    aws = _fake_aws(cc)

    build_image(aws=aws, config=_config(), cloud_config=cc, outputs=OUTPUTS,
                repo_root=root, environ={"APPROVED_SPEND": "1"})

    assert len(aws.run_requests) == 1
    request = aws.run_requests[0]

    # The same shape Property 4 asserts, reusing its helper and constants. The builder carries no
    # Job_Id, so it carries no Job_Id tag.
    assert request["MinCount"] == request["MaxCount"] == 1
    assert request["InstanceInitiatedShutdownBehavior"] == "terminate"
    assert request["MetadataOptions"] == {"HttpTokens": "required", "HttpEndpoint": "enabled"}
    assert request["IamInstanceProfile"] == {"Arn": OUTPUTS.job_instance_profile_arn}
    assert request["SecurityGroupIds"] == [OUTPUTS.security_group_id]
    assert request["SubnetId"] == SUBNET
    assert request["UserData"] == install_script(cc)
    for resource_type in ("instance", "volume", "network-interface"):
        tags = _tags_for(request, resource_type)
        assert tags is not None
        assert PROJECT_TAG_DICT in tags
    instance_tags = _tags_for(request, "instance")
    assert instance_tags is not None
    assert all(tag["Key"] != JOB_ID_TAG for tag in instance_tags)
    for forbidden in FORBIDDEN_KEYS:
        assert forbidden not in request


# Licence_List: the Machine_Image tool rows, the COLMAP row naming its OFF flags (Req 14.5)

def _licence_lines() -> list[str]:
    """The Markdown table-row lines of the Licence_List (every line starting with `|`)."""
    return [line for line in LICENCE_LIST.read_text(encoding="utf-8").splitlines()
            if line.lstrip().startswith("|")]


def _row_with_name(contains: str) -> str:
    """The one Licence_List row whose Name cell (the first cell) contains `contains`, case-insensitively."""
    rows = [line for line in _licence_lines()
            if contains.lower() in line.split("|")[1].strip().lower()]
    assert len(rows) == 1, (
        f"the Licence_List should have one row whose name contains {contains!r}, found {len(rows)}")
    return rows[0]


def test_licence_list_has_the_machine_image_tool_rows() -> None:
    # One row each for the image's tools, weights and packages (Req 14.5): the Base_AMI, ffmpeg (built
    # with zscale), COLMAP, gsplat, fused-ssim, the MoGe-2 weights and Node.
    for name in ("Deep Learning Base", "ffmpeg", "COLMAP", "gsplat", "fused-ssim", "MoGe-2", "Node"):
        _row_with_name(name)  # asserts exactly one such row exists


def test_colmap_licence_row_names_its_off_flags() -> None:
    # The COLMAP row records the OFF flags that keep SiftGPU, CGAL and LSD out (Req 14.5).
    assert OFF_FLAGS in _row_with_name("COLMAP")
