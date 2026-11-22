"""Unit, structural and property tests for the Image_Build, cloud/image.py (cloud-jobs task 7.4).

Four kinds of check, all on the CPU with no network, no AWS account and no GPU:

- Structural: `install_script` is the builder's whole install, asserted as text and never run here
  (the real install compiles CUDA code and fetches gigabytes). It must hold, in order: the apt
  install of the build tools and COLMAP 4.2.1's documented libraries, without the GUI and CGAL
  parts; uv 0.13.9 from its checked tarball; HEAD's five manifests from the bucket, then `uv sync`;
  zimg and ffmpeg; COLMAP with the OFF flags that keep SiftGPU, CGAL and LSD out, plus ONNX off;
  CUDA Toolkit 12.8 and torch 2.7.1 for CUDA 12.8; gsplat 1.5.3 and fused-ssim built from source with
  `TORCH_CUDA_ARCH_LIST="8.6;8.9"`, then an import check; the MoGe-2 weights checked against their
  sha256; `npm ci`; and a `cloud-init clean` before the done marker (Req 9.2, 9.3, 9.4). It uses
  `uv pip --python` into the same venv the job machines run, and a failure trap that powers off.

- Property: for any valid bucket, region and image name, the script is a bash script within the
  16 KB User_Data limit with LF endings, and every value goes in `shlex.quote`d, so splitting its
  line recovers the value.

- With the `FakeAws`: `build_image` is a Paid_Entry_Point, so it asks the Spend_Guard first and does
  nothing when `APPROVED_SPEND` is unset (Req 9.10). On the approved path it reads the Base_AMI from
  SSM, uploads HEAD's manifests under the image's own `jobs/_image/<name>/` prefix before the
  launch, stops the builder before `create_image` (an API stop never triggers terminate-on-shutdown),
  tags the image with the Project_Tag and the two lock hashes, waits for `available`, terminates the
  builder and records an `image-build` Spend_Line under `data/` (Req 9.1, 9.7, 9.8, 9.9, 12.2). A
  failed install ends the wait at once and terminates the builder. Its launch request carries
  Property 4's required settings (Req 8.9), checked with the Property-4 helper and constants reused
  from `test_cloud_launch`.

- Licence_List: one row for each Machine_Image tool, weight and package, the COLMAP row naming its
  flags and the CUDA Toolkit row NVIDIA's licence (Req 14.5).

The `FakeAws` is subclassed only to record each `run_instances` request (so the launch request can
be read back) and to have the builder's console report the install's end at once, so the console
wait returns on the CPU; the real console polling is a real-machine path. No test sets a real
`APPROVED_SPEND`: each build passes it as an in-process `environ` dict straight to the Spend_Guard,
the way the Start_Command's tests do.
"""

from __future__ import annotations

import hashlib
import json
import shlex
from pathlib import Path
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from splatpipe.config import Config

from cloud.config import CloudConfig
from cloud.errors import CloudError
from cloud.image import (INSTALL_DONE_MARKER, INSTALL_FAILED_MARKER, MANIFESTS, MOGE_WEIGHTS,
                         build_image, install_script, manifest_prefix)
from cloud.launch import JOB_ID_TAG
from cloud.repo import head_file, lock_hashes
from cloud.spend import SpendRefused
from cloud.spend_file import LABEL_IMAGE
from cloud.stack import StackOutputs
from cloud.userdata import VENV_PYTHON, UserDataValues, render_user_data

from fake_aws import FakeAws, cloud_config, git_repo
# Reuse Property 4's own helper and constants, so "the launch request satisfies Property 4's shape"
# is checked by the same assertions task 4 uses, not a re-derived copy.
from test_cloud_launch import FORBIDDEN_KEYS, PROJECT_TAG_DICT, _tags_for

# The COLMAP build flags that keep SiftGPU (non-commercial), CGAL (GPL) and LSD (AGPL) out, as one
# string, so both the install_script and the COLMAP Licence_List row are checked for it verbatim.
OFF_FLAGS = "-DCUDA_ENABLED=OFF -DOPENGL_ENABLED=OFF -DGUI_ENABLED=OFF -DCGAL_ENABLED=OFF -DLSD_ENABLED=OFF"
# The two more COLMAP flags: ONNX off leaves out the learned features and ONNX Runtime (SIFT stays),
# and OpenBLAS stands in for Intel's MKL.
EXTRA_FLAGS = "-DONNX_ENABLED=OFF -DBLA_VENDOR=OpenBLAS"

# COLMAP 4.2.1's documented Ubuntu packages (doc/install.rst at the 4.2.1 tag) that the image keeps,
# and the ones it leaves out: the GUI's Qt packages, CGAL, and Intel's MKL, whose licence isn't on
# our list. libglew-dev is kept because 4.2.1's CMake requires GLEW even with OPENGL_ENABLED=OFF.
COLMAP_PACKAGES = (
    "git", "cmake", "ninja-build", "build-essential", "libboost-program-options-dev",
    "libboost-graph-dev", "libboost-system-dev", "libeigen3-dev", "libopenimageio-dev",
    "openimageio-tools", "libmetis-dev", "libgoogle-glog-dev", "libgtest-dev", "libgmock-dev",
    "libsqlite3-dev", "libglew-dev", "libceres-dev", "libsuitesparse-dev", "libcurl4-openssl-dev",
    "libssl-dev",
)
LEFT_OUT_PACKAGES = ("qt6-base-dev", "libqt6opengl6-dev", "libqt6openglwidgets6", "qt6-svg-dev",
                     "libcgal-dev", "libmkl-full-dev")
# What zimg's autotools build and ffmpeg's configure need, and the BLAS that replaces MKL.
OTHER_PACKAGES = ("autoconf", "automake", "libtool", "nasm", "pkg-config", "curl",
                  "libopenblas-openmp-dev")

# The pinned downloads, each with the sha256 the script checks it against.
UV_URL = "https://github.com/astral-sh/uv/releases/download/0.13.9/uv-x86_64-unknown-linux-gnu.tar.gz"
UV_SHA256 = "a7fbaff2b9a271d85c1bdb540941090fd1ce011797b4abeb48e4dcd39c2471d5"
CUDA_KEYRING_URL = ("https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/"
                    "cuda-keyring_1.1-1_all.deb")
CUDA_KEYRING_SHA256 = "d2a6b11c096396d868758b86dab1823b25e14d70333f1dfa74da5ddaf6a06dba"
MOGE_URL = ("https://huggingface.co/Ruicheng/moge-2-vitb-normal/resolve/"
            "ca5f0e07ff01d3e5a364c1d954ed12ee1814b368/model.pt")
MOGE_SHA256 = "16b8110e86d5dc5a849db120ca96ef3a223fd30b0c9146d1d81db504073da5f6"
# fused-ssim at the full 40-character commit gsplat 1.5.3's examples/requirements.txt pins.
FUSED_SSIM = ('"fused-ssim @ git+https://github.com/rahul-goel/fused-ssim@'
              '328dc9836f513d00c4b5bc38fe30478b4435cbb5"')
# The environment both CUDA builds run with: NVIDIA's CUDA 12.8 compiler, for the A10G and the L4/L40S.
BUILD_ENV = 'CUDA_HOME=/usr/local/cuda-12.8 TORCH_CUDA_ARCH_LIST="8.6;8.9"'

ROOT = Path(__file__).resolve().parents[1]  # this file is in tests/, at the repo root
LICENCE_LIST = ROOT / "docs" / "third-party-licences.md"

# A real-shaped base AMI id for the seeded SSM parameter; build_image passes it through as the
# builder's ImageId. No value in this file holds a 12-digit account id.
BASE_AMI = "ami-0base00000000000"

# The seven Stack_Outputs build_image is handed. Only the bucket, the instance-profile ARN, the
# security group and the VPC id are used (the manifests' bucket, the launch request and the
# default-subnet lookup); none holds an account id, since this file doesn't repeat the aws/stack
# redaction check.
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

# A real-shaped image name and its manifest prefix, for the structural tests.
IMAGE_NAME = "voxelis-kiroween-image-20261122-153000"
PREFIX = f"jobs/_image/{IMAGE_NAME}/"


class _BuilderAws(FakeAws):
    """A `FakeAws` for the Image_Build: records each `run_instances` request and reports the end of
    the install on the builder's console at once.

    build_image builds its launch request inside a closure and hands it to `launch_with_fallback`,
    which the base `FakeAws` doesn't keep, so this records each request for the Property-4 shape
    check. The builder runs on the Job_Role and signals the end of the install by printing a marker
    to the console; on the CPU there is no real builder, so `console_output` returns the marker at
    once (the done marker, unless the test asks for the failure marker) and `build_image`'s wait
    returns on its first poll. The call is still recorded, so the ordered timeline stays accurate.
    """

    def __init__(self, *, marker: str = INSTALL_DONE_MARKER, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.run_requests: list[dict[str, Any]] = []
        self.marker = marker

    def run_instances(self, request: Any) -> Any:
        self.run_requests.append(dict(request))
        return super().run_instances(request)

    def console_output(self, instance_id: str) -> str:
        super().console_output(instance_id)  # keep the call on the timeline
        return f"cloud-init output...\n{self.marker}\n"


def _config() -> Config:
    """A pipeline Config holding just the `[aws]` the Image_Build reads: the Project_Tag build_image
    tags the image with, and a stack name (unused by build_image, read only by its `main`)."""
    return Config(
        {"aws": {"stack_name": "voxelis-kiroween", "project_tag": "project=voxelis-kiroween"}},
        source="test config")


def _fake_aws(cc: CloudConfig, *, marker: str = INSTALL_DONE_MARKER) -> _BuilderAws:
    """A healthy `_BuilderAws`: the Base_AMI parameter resolves, the builder type is offered a zone
    with a default subnet, and the G_Quota admits it."""
    return _BuilderAws(marker=marker, ssm={cc.base_ami_parameter: BASE_AMI},
                       offerings={cc.image_builder_type: [ZONE]},
                       subnets={ZONE: SUBNET}, quota=32.0)


def _script(**values: str) -> str:
    """The install script for the real-shaped bucket, region and prefix, or the ones given."""
    kwargs = {"bucket": OUTPUTS.bucket, "region": "us-east-1", "prefix": PREFIX, **values}
    return install_script(cloud_config(), **kwargs)


def _lines(script: str) -> list[str]:
    """The script's lines, without their trailing spaces."""
    return [line.rstrip() for line in script.split("\n")]


def _index(script: str, text: str) -> int:
    """Where `text` first appears in the script, failing the test (with the text) when it doesn't."""
    position = script.find(text)
    assert position >= 0, f"the install script should contain {text!r}"
    return position


def _command(script: str, starts: str) -> str:
    """The whole command whose first line starts with `starts`, its continuation lines joined."""
    lines = _lines(script)
    start = next(i for i, line in enumerate(lines) if line.startswith(starts))
    parts = [lines[start]]
    while parts[-1].endswith("\\"):
        start += 1
        parts.append(lines[start])
    return " ".join(part.rstrip("\\").strip() for part in parts)


def _last_spend_line(repo_root: Path) -> dict[str, Any]:
    """The last Spend_Line build_image appended to `<repo_root>/data/spend.jsonl`."""
    lines = [line for line in (repo_root / "data" / "spend.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip()]
    return json.loads(lines[-1])


# Structural: the script's shape, its failure trap and its values

def test_install_script_is_bash_within_the_user_data_limit_and_stops_on_any_failure() -> None:
    script = _script()
    lines = _lines(script)
    assert lines[0] == "#!/bin/bash"
    assert "\r" not in script
    assert len(script.encode("utf-8")) <= 16384
    assert "set -euo pipefail" in lines
    # No `a && b` chains: under `set -e` a failure before the last command of one is ignored, so
    # every command stands on its own line and any failure stops the install.
    assert "&&" not in script


def test_install_script_powers_off_on_a_failure_and_sets_its_trap_before_the_cap() -> None:
    script = _script()
    # The trap comes before the Time_Cap shutdown, so even a failure on the cap's own line ends the
    # builder, and the cap comes before the first install step.
    assert _index(script, "trap finish EXIT") < _index(script, "shutdown -h +120")
    assert _index(script, "shutdown -h +120") < _index(script, "apt_get update")
    # On a failure the trap prints the marker build_image watches for, then powers off, which
    # terminate-on-shutdown turns into a termination.
    trap = script[_index(script, "finish() {"):_index(script, "trap finish EXIT")]
    assert INSTALL_FAILED_MARKER in trap
    assert "shutdown -h now" in trap
    assert trap.index(INSTALL_FAILED_MARKER) < trap.index("shutdown -h now")


def test_install_script_carries_the_bucket_region_and_manifest_prefix() -> None:
    lines = _lines(_script())
    assert f"BUCKET={OUTPUTS.bucket}" in lines
    assert "REGION=us-east-1" in lines
    assert f"MANIFESTS={PREFIX}" in lines


@pytest.mark.parametrize("values", [
    {"bucket": "Not A Bucket"},
    {"bucket": "voxelis-kiroween-jobs; reboot"},
    {"region": "east"},
    {"prefix": "jobs/someone-else/x/"},
    {"prefix": "jobs/_image/no-trailing-slash"},
    {"prefix": "jobs/_image/a/b/"},
], ids=["spaces", "shell", "region", "outside-image-prefix", "no-slash", "nested"])
def test_install_script_refuses_a_value_that_fails_its_check(values: dict[str, str]) -> None:
    with pytest.raises(CloudError):
        _script(**values)


# Property: every valid value goes in quoted, and the script stays within the limit

BUCKETS = st.from_regex(r"[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]", fullmatch=True)
REGIONS = st.from_regex(r"[a-z]{2}-[a-z]{1,20}-[0-9]{1,4}", fullmatch=True)
IMAGE_NAMES = st.from_regex(r"[A-Za-z0-9][A-Za-z0-9._-]{0,80}", fullmatch=True)


@given(bucket=BUCKETS, region=REGIONS, name=IMAGE_NAMES)
def test_property_the_install_script_is_safe_for_any_valid_values(
        bucket: str, region: str, name: str) -> None:
    """For any valid bucket, region and image name, the install script is a bash script within the
    16 KB EC2 User_Data limit with LF endings, and each value round-trips through `shlex`."""
    prefix = manifest_prefix(name)
    script = install_script(cloud_config(), bucket=bucket, region=region, prefix=prefix)
    lines = _lines(script)
    assert lines[0] == "#!/bin/bash"
    assert "\r" not in script
    assert len(script.encode("utf-8")) <= 16384
    for variable, value in (("BUCKET", bucket), ("REGION", region), ("MANIFESTS", prefix)):
        line = next(line for line in lines if line.startswith(f"{variable}="))
        assert shlex.split(line) == [f"{variable}={value}"]


# Structural: the install's steps, in order

def test_install_script_installs_colmaps_documented_packages_first() -> None:
    script = _script()
    # The apt install of the build tools and libraries is the first install step, before uv and
    # before anything is fetched or built.
    install = _command(script, "apt_get install -y")
    assert _index(script, "apt_get install -y \\\n  build-essential") < _index(script, UV_URL)
    packages = set(install.split()[3:])
    for package in COLMAP_PACKAGES + OTHER_PACKAGES:
        assert package in packages, f"the apt install should include {package}"
    for package in LEFT_OUT_PACKAGES:
        assert package not in script, f"the install should leave out {package}"
    # COLMAP's docs: the folder Ubuntu's OpenImageIO CMake config wants, and OpenBLAS's OpenMP build
    # selected, both before COLMAP's configure.
    configure = _index(script, "cmake -S /tmp/colmap")
    assert _index(script, "mkdir -p /usr/include/opencv4") < configure
    assert _index(script, "update-alternatives --set") < configure
    assert "/usr/lib/x86_64-linux-gnu/openblas-openmp/$lib" in script


def test_install_script_installs_uv_0_13_9_from_its_checked_tarball() -> None:
    script = _script()
    assert UV_URL in script
    assert f'echo "{UV_SHA256}  /tmp/uv.tar.gz" | sha256sum -c -' in script
    assert "install -m 0755 /tmp/uv/uv /tmp/uv/uvx /usr/local/bin/" in script
    # uv is checked and installed before it's first used.
    assert _index(script, "sha256sum -c -") < _index(script, "uv sync")


def test_install_script_fetches_heads_manifests_before_uv_sync() -> None:
    script = _script()
    loop = f"for name in {' '.join(MANIFESTS)}; do"
    assert set(MANIFESTS) == {"pyproject.toml", "uv.lock", ".python-version", "package.json",
                              "package-lock.json"}
    assert loop in script
    copy = 'aws s3 cp "s3://$BUCKET/$MANIFESTS$name" "/opt/job/$name" --region "$REGION"'
    assert copy in script
    assert _index(script, loop) < _index(script, "uv sync --frozen --no-dev")
    # npm ci runs in the same folder, from the package.json and package-lock.json fetched there.
    assert _index(script, "uv sync --frozen --no-dev") < _index(script, "npm ci")


def test_install_script_uses_uv_pip_into_the_venv_the_job_machines_run() -> None:
    script = _script()
    # uv's venv has no pip, so every install goes through `uv pip --python "$VENV_PY"`.
    assert "-m pip" not in script
    installs = [line for line in _lines(script) if "uv pip install" in line]
    assert installs, "the install script should install packages with uv pip"
    assert all('uv pip install --python "$VENV_PY"' in line for line in installs)
    # The venv is the one a job's and the smoke test's User_Data run, and the one the import check uses.
    assert f"VENV_PY={VENV_PYTHON}" in _lines(script)
    user_data = render_user_data(UserDataValues(
        job_id="20261122-153000-7f3a9c", bucket=OUTPUTS.bucket, region="us-east-1",
        preset="default", image_id="ami-0123456789abcdef0",
        code_archive_key="jobs/20261122-153000-7f3a9c/input/code.tar", cap_minutes=180,
        upload_margin_minutes=10, uv_lock_hash="0" * 64, package_lock_hash="1" * 64))
    assert f'{VENV_PYTHON} -m cloud.runner --job "$JOB_ID"' in user_data


def test_install_script_builds_the_gpu_packages_with_cuda_12_8() -> None:
    script = _script()
    # NVIDIA's CUDA Toolkit 12.8, from NVIDIA's Ubuntu 24.04 repo, whose keyring is checked first.
    assert CUDA_KEYRING_URL in script
    assert f'echo "{CUDA_KEYRING_SHA256}  /tmp/cuda-keyring.deb" | sha256sum -c -' in script
    assert _index(script, CUDA_KEYRING_SHA256) < _index(script, "dpkg -i /tmp/cuda-keyring.deb")
    assert _index(script, "dpkg -i /tmp/cuda-keyring.deb") < _index(script, "cuda-toolkit-12-8")
    # torch and torchvision for CUDA 12.8, and never the old CUDA 12.4 build.
    torch = _command(script, 'uv pip install --python "$VENV_PY" --index-url')
    assert "https://download.pytorch.org/whl/cu128" in torch
    assert "torch==2.7.1" in torch and "torchvision==0.22.1" in torch
    assert "cu124" not in script and "2.5.1" not in script
    # The build tools for --no-build-isolation, before the builds that need them.
    tools = _command(script, 'uv pip install --python "$VENV_PY" setuptools')
    for tool in ("setuptools==", "wheel==", "ninja=="):
        assert tool in tools
    # gsplat 1.5.3 from its sdist and fused-ssim at its 40-character commit, both built against
    # CUDA 12.8 for 8.6 and 8.9, after torch and the toolkit.
    gsplat = _command(script, f"{BUILD_ENV} \\")
    assert gsplat == (f'{BUILD_ENV} uv pip install --python "$VENV_PY" --no-build-isolation '
                      "--no-binary gsplat gsplat==1.5.3")
    fused = script[_index(script, gsplat.split()[-1]):]
    assert f'{BUILD_ENV} \\\n  uv pip install --python "$VENV_PY" --no-build-isolation \\\n  {FUSED_SSIM}' in fused
    assert _index(script, "cuda-toolkit-12-8") < _index(script, "gsplat==1.5.3")
    assert _index(script, "torch==2.7.1") < _index(script, "gsplat==1.5.3")
    # The build fails unless both compiled modules import, checked after both builds and before the
    # done marker.
    check = '"$VENV_PY" -c "from gsplat import csrc; import fused_ssim"'
    assert _index(script, FUSED_SSIM) < _index(script, check) < _index(script, INSTALL_DONE_MARKER)


def test_install_script_fetches_the_moge_2_weights_and_checks_their_sha256() -> None:
    script = _script()
    assert MOGE_WEIGHTS == "/opt/models/moge-2-vitb-normal/model.pt"
    assert f"curl -fsSL --retry 3 -o {MOGE_WEIGHTS} \\\n  {MOGE_URL}" in script
    assert f'echo "{MOGE_SHA256}  {MOGE_WEIGHTS}" | sha256sum -c -' in script
    # The weights match the licence-cleared ones the Licence_List names, fetched without
    # huggingface-hub.
    assert "Ruicheng/moge-2-vitb-normal" in script
    assert "huggingface-hub" not in script and "huggingface_hub" not in script


def test_install_script_builds_ffmpeg_with_zimg_and_colmap_with_its_flags() -> None:
    script = _script()
    # zimg at a release tag, installed and registered with the dynamic linker before ffmpeg links it.
    assert "git clone --branch release-3.0.6 --depth 1 https://github.com/sekrit-twc/zimg.git" in script
    assert (_index(script, "make install") < _index(script, "ldconfig")
            < _index(script, "--enable-libzimg"))
    assert "./configure --prefix=/usr/local --enable-gpl --enable-libzimg" in script
    # COLMAP 4.2.1 with SiftGPU, CGAL and LSD out, the ONNX learned features off and OpenBLAS.
    assert "git clone --branch 4.2.1 --depth 1 https://github.com/colmap/colmap.git" in script
    configure = _command(script, "cmake -S /tmp/colmap")
    assert OFF_FLAGS in configure
    assert EXTRA_FLAGS in configure


def test_install_script_cleans_cloud_init_before_imaging() -> None:
    script = _script()
    # cloud-init clean wipes the builder's first-boot state so the image boots fresh. It runs before
    # the builder prints the done marker, which is what the Image_Build_Command waits for before it
    # stops and images the builder, so the clean happens before imaging.
    assert "cloud-init clean" in script
    assert script.index("cloud-init clean") < script.rindex(INSTALL_DONE_MARKER)


# With the FakeAws: build_image asks the Spend_Guard first and builds nothing when it says no.

def test_build_image_refuses_without_approved_spend(tmp_path: Path) -> None:
    cc = cloud_config()
    aws = _fake_aws(cc)
    with pytest.raises(SpendRefused):
        build_image(aws=aws, config=_config(), cloud_config=cc, outputs=OUTPUTS,
                    repo_root=tmp_path, environ={})  # APPROVED_SPEND unset
    # The Spend_Guard comes first, so nothing touched AWS, nothing was uploaded, no builder launched,
    # no image made and no Spend_Line written.
    assert aws.calls == []
    assert aws.objects == {}
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
    # The image is waited to `available` after it is created (Req 9.9), polling every poll_seconds
    # for up to the Image_Build's Time_Cap, not botocore's default 10 minutes.
    assert ops.index("create_image") < ops.index("wait_image_available")
    assert aws.image_waits == [(image_id, cc.poll_seconds, cc.image.cap_minutes)]

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


def test_build_image_uploads_heads_manifests_under_the_images_prefix_before_the_launch(
        tmp_path: Path) -> None:
    root = git_repo(tmp_path)
    # A working-tree edit that isn't committed: the upload must carry HEAD's bytes, not these.
    (root / "uv.lock").write_text("version = 99\n", encoding="utf-8")
    cc = cloud_config()
    aws = _fake_aws(cc)

    build_image(aws=aws, config=_config(), cloud_config=cc, outputs=OUTPUTS,
                repo_root=root, environ={"APPROVED_SPEND": "1"})

    # The five manifests go under the prefix of the image's own name, the same name create_image
    # is given, after the Spend_Guard and the SSM read and before the builder is launched.
    prefix = manifest_prefix(aws.images[0].name)
    uploads = [key for op, key in aws.calls if op == "s3_upload"]
    assert uploads == [f"{prefix}{name}" for name in MANIFESTS]
    ops = [op for op, _ in aws.calls]
    assert ops.index("ssm_parameter") < ops.index("s3_upload")
    assert max(i for i, op in enumerate(ops) if op == "s3_upload") < ops.index("run_instances")
    for name in MANIFESTS:
        assert aws.objects[f"{prefix}{name}"] == head_file(name, cwd=root)
    # So the uv.lock the builder installs from is the one whose hash the image is tagged with.
    uploaded_lock = aws.objects[f"{prefix}uv.lock"]
    assert hashlib.sha256(uploaded_lock).hexdigest() == aws.images[0].tags["uv_lock_hash"]


def test_build_image_ends_the_builder_and_stops_when_the_install_fails(tmp_path: Path) -> None:
    root = git_repo(tmp_path)
    cc = cloud_config()
    aws = _fake_aws(cc, marker=f"{INSTALL_FAILED_MARKER} (exit 1)")

    with pytest.raises(CloudError) as caught:
        build_image(aws=aws, config=_config(), cloud_config=cc, outputs=OUTPUTS,
                    repo_root=root, environ={"APPROVED_SPEND": "1"})

    # The failure marker ends the wait on the first poll with a readable error, the builder is
    # terminated, and no image is made or Spend_Line written.
    assert "install failed" in str(caught.value)
    builder_id = aws.instances[0].id
    assert [op for op, _ in aws.calls].count("console_output") == 1
    assert ("terminate_instances", builder_id) in aws.calls
    assert aws.instances[0].state == "terminated"
    assert aws.images == []
    assert not (root / "data" / "spend.jsonl").exists()


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
    assert request["UserData"] == install_script(
        cc, bucket=OUTPUTS.bucket, region=aws.region, prefix=manifest_prefix(aws.images[0].name))
    for resource_type in ("instance", "volume", "network-interface"):
        tags = _tags_for(request, resource_type)
        assert tags is not None
        assert PROJECT_TAG_DICT in tags
    instance_tags = _tags_for(request, "instance")
    assert instance_tags is not None
    assert all(tag["Key"] != JOB_ID_TAG for tag in instance_tags)
    for forbidden in FORBIDDEN_KEYS:
        assert forbidden not in request


# Licence_List: the Machine_Image tool rows, the COLMAP row naming its flags (Req 14.5)

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
    # with zscale), COLMAP, gsplat, fused-ssim, the MoGe-2 weights, Node, NVIDIA's CUDA Toolkit 12.8,
    # torchvision, the Python build tools and the Ubuntu packages the builds need.
    for name in ("Deep Learning Base", "ffmpeg", "COLMAP", "gsplat", "fused-ssim", "MoGe-2", "Node",
                 "CUDA Toolkit 12.8", "torchvision", "setuptools", "apt archive"):
        _row_with_name(name)  # asserts exactly one such row exists


def test_colmap_licence_row_names_its_flags() -> None:
    # The COLMAP row records the OFF flags that keep SiftGPU, CGAL and LSD out (Req 14.5), and the
    # two more the image builds with.
    row = _row_with_name("COLMAP")
    assert OFF_FLAGS in row
    assert "-DONNX_ENABLED=OFF" in row and "-DBLA_VENDOR=OpenBLAS" in row


def test_cuda_toolkit_licence_row_names_nvidias_eula_and_where_it_lives() -> None:
    row = _row_with_name("CUDA Toolkit 12.8")
    assert "End User License Agreement" in row
    assert "Samarth" in row
    assert "never shipped" in row
