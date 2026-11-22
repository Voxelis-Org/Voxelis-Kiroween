"""The Image_Build: make our private Machine_Image once, from the Base_AMI, with a builder machine.

`install_script` is the builder's whole User_Data: the install, as bash, with every tool, weight and
package pinned. It is asserted structurally and never run in the Test_Suite (Req 9.2, 9.3, 9.4). It
never bakes in the Pipeline's own code; the Start_Command ships that per Job as the Code_Archive
(Req 9.5). The dependency manifests it installs from come from HEAD through the Job_Bucket, because
uv.lock alone is over the 16 KB User_Data limit: `build_image` uploads them under
`jobs/_image/<image name>/` before the launch, and the script fetches them with the AWS CLI. When the
install is done the script prints a done marker, which the Image_Build_Command waits for on the
console. On success it never powers the machine off, because the launch sets terminate-on-shutdown
and an OS shutdown would terminate the builder before it could be imaged. On a failure it prints a
failure marker and powers off, so the builder ends at once and the command stops waiting.

`build_image` drives the build and is a Paid_Entry_Point: it passes the Spend_Guard first (Req 9.10),
then reads the Base_AMI id from the Base_AMI_Parameter, admits the `image_builder_type` against the
G_Quota, uploads the manifests, launches the builder from the Base_AMI with the Requirement 8 launch
request plus a root-volume `BlockDeviceMappings` entry (CreateImage can't resize a volume later) and
`install_script` as the User_Data, waits for the install, stops the builder through the EC2 API
(never triggering terminate-on-shutdown, the skill's stop-first rule), makes the Machine_Image from
the stopped builder with the Project_Tag and the `uv.lock`/`package-lock.json` hashes tagged on the
image and its snapshots in the one call (Req 9.7, 9.8), waits until it is `available`, terminates
the builder, and appends a `LABEL_IMAGE` Spend_Line for the builder's measured time (Req 9.9, 12.2).
It uses the Runner_Profile throughout (Req 1.5).

A cloud module may import `splatpipe` to call its public functions; `splatpipe` never imports
`cloud`. This module imports no boto3 of its own (only `cloud.aws` does) and no GPU package, so it
loads on a laptop.
"""

from __future__ import annotations

import argparse
import json
import posixpath
import re
import shlex
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from splatpipe.config import load_config

from cloud.aws import laptop_aws
from cloud.config import load_cloud_config
from cloud.errors import CloudError, run_cloud_cli
from cloud.launch import build_run_instances_request, client_token, launch_with_fallback
from cloud.repo import head_file, lock_hashes
from cloud.select import admit, launch_plan, running_vcpus
from cloud.spend import check_spend
from cloud.spend_file import LABEL_IMAGE, rough_cost, spend_line
from cloud.stack import resolve_stack
# The shared machine paths, and the User_Data's value checks and size limit, so the builder's script
# is held to the same rules as a job machine's.
from cloud.userdata import JOB_ROOT, VENV_PYTHON, _BUCKET, _MAX_BYTES, _REGION, _pattern

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Any

    from splatpipe.config import Config

    from cloud.aws import Aws
    from cloud.config import CloudConfig, MachineSpec
    from cloud.stack import StackOutputs

# The line install_script prints on the console when the install is done, which build_image waits
# for. It is a fixed string with no account id, safe to read from the console (the skill).
INSTALL_DONE_MARKER = "voxelis-kiroween image build: install complete"
# The line install_script prints when a step fails, just before it powers the builder off. Seeing it
# ends build_image's wait at once, rather than at the Time_Cap. A fixed string too.
INSTALL_FAILED_MARKER = "voxelis-kiroween image build: install failed"
# HEAD's dependency manifests the builder installs from: uv sync reads the first three and npm ci
# the last two. build_image uploads them and install_script downloads them into JOB_ROOT.
MANIFESTS = ("pyproject.toml", "uv.lock", ".python-version", "package.json", "package-lock.json")
# Where in the Job_Bucket one image build's manifests go, as jobs/_image/<image name>/. Under jobs/,
# so the Runner_Role may upload them and the builder's Job_Role may read them, like the
# Smoke_Markers under jobs/_smoke/.
IMAGE_PREFIX = "jobs/_image/"
# Where the image keeps the MoGe-2 weights, and their pinned sha256, which the install checks after
# the download and the smoke machine checks again. The depth-priors spec decides how its step finds
# the file.
MOGE_WEIGHTS = "/opt/models/moge-2-vitb-normal/model.pt"
MOGE_SHA256 = "16b8110e86d5dc5a849db120ca96ef3a223fd30b0c9146d1d81db504073da5f6"
# Where npm ci, run in JOB_ROOT, puts the splat-transform command.
SPLAT_TRANSFORM = f"{JOB_ROOT}/node_modules/.bin/splat-transform"
# The shape of a manifest prefix install_script accepts: IMAGE_PREFIX, one image name, a slash.
_MANIFEST_PREFIX = re.compile(r"jobs/_image/[A-Za-z0-9][A-Za-z0-9._-]*/")
# The non-terminated instance states, so running_vcpus counts every machine still on the G_Quota.
_LIVE_STATES = ("pending", "running", "shutting-down", "stopping", "stopped")
# The builder's root device for the Deep Learning Base AMI (Ubuntu 24.04), whose size the launch
# sets in BlockDeviceMappings because CreateImage can't resize a volume later (Req 9.9, the skill).
_ROOT_DEVICE = "/dev/sda1"
# The read command a timed-out build points a person at, to look at the builder's console by hand.
_CONSOLE_HINT = "uv run python -m cloud.list"


def manifest_prefix(image_name: str) -> str:
    """The Job_Bucket prefix one image build's MANIFESTS go under: `jobs/_image/<image name>/`."""
    return f"{IMAGE_PREFIX}{image_name}/"


def install_script(cc: CloudConfig, *, bucket: str, region: str, prefix: str) -> str:
    """The builder's User_Data: the whole install, as bash. Asserted structurally, never run.

    Pinned throughout (Req 9.2, 9.3, 9.4): the build tools and libraries from Ubuntu's apt archive;
    uv 0.13.9; our Python environment from HEAD's `uv.lock`, which the script fetches from `prefix`
    in `bucket`; ffmpeg with the zscale filter (zimg); COLMAP 4.2.1 from source with SiftGPU, CGAL,
    LSD and the ONNX learned features left out; torch 2.7.1 for CUDA 12.8, then gsplat 1.5.3 and
    fused-ssim compiled with NVIDIA's CUDA Toolkit 12.8 for the A10G (8.6) and the L4 and L40S
    (8.9); the MoGe-2 weights, checked against their sha256; Node and splat-transform from
    `package-lock.json` with `npm ci`. It then removes what a Job machine must not inherit and
    prints the done marker. It never bakes in the Pipeline's own code (Req 9.5).

    On success it leaves the builder running for the Image_Build_Command to stop through the EC2
    API. A failing step prints the failure marker and powers off, which terminate-on-shutdown turns
    into a termination. Each value is checked and `shlex.quote`d like a job machine's User_Data, and
    a script over the 16 KB EC2 User_Data limit is a CloudError.
    """
    cap_minutes = int(cc.image.cap_minutes)
    q = shlex.quote
    bucket = _pattern(bucket, _BUCKET, "bucket name")
    region = _pattern(region, _REGION, "region")
    prefix = _pattern(prefix, _MANIFEST_PREFIX, "manifest prefix")
    names = " ".join(MANIFESTS)
    moge_dir = posixpath.dirname(MOGE_WEIGHTS)
    script = f"""#!/bin/bash
# The image builder's install, run once by cloud-init as root on first boot, every version pinned.
# It never bakes in the Pipeline's own code: each job extracts its Code_Archive over {JOB_ROOT}. On
# success it leaves the builder running for the Image_Build_Command to stop and image.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
# cloud-init runs this without HOME, which uv, npm and git expect.
export HOME=/root
# uv keeps its Python 3.11 under /opt, beside the environment it serves, not in root's home.
export UV_PYTHON_INSTALL_DIR=/opt/uv/python

BUCKET={q(bucket)}
REGION={q(region)}
MANIFESTS={q(prefix)}
VENV_PY={q(VENV_PYTHON)}

# On a failure, say so on the console, give the Image_Build_Command a minute to read it, and power
# off: terminate-on-shutdown then ends the builder at once.
finish() {{
  status=$?
  if [ "$status" -ne 0 ]; then
    echo "{INSTALL_FAILED_MARKER} (exit $status)"
    echo "{INSTALL_FAILED_MARKER} (exit $status)" > /dev/console 2>/dev/null || true
    sleep 60
    shutdown -h now
  fi
}}
trap finish EXIT

# Time_Cap backstop: if the install ever hangs, this OS shutdown (with terminate-on-shutdown) ends
# the builder so it can't run past the time that was approved.
shutdown -h +{cap_minutes}

# Every apt-get waits up to 10 minutes for the dpkg lock, which a first-boot apt run may hold.
apt_get() {{
  apt-get -o DPkg::Lock::Timeout=600 "$@"
}}

# 1. The build tools and libraries zimg, ffmpeg and COLMAP 4.2.1 need. COLMAP's are its install
#    docs' list at the 4.2.1 tag, less Qt (the GUI) and CGAL. libglew-dev stays, because 4.2.1's
#    CMake requires GLEW and OpenGL even with OPENGL_ENABLED=OFF. OpenBLAS's OpenMP build stands in
#    for the docs' Intel MKL, whose licence isn't on our list, as the docs allow.
apt_get update
apt_get install -y \\
  build-essential git curl pkg-config autoconf automake libtool nasm cmake ninja-build \\
  libboost-program-options-dev libboost-graph-dev libboost-system-dev libeigen3-dev \\
  libopenimageio-dev openimageio-tools libmetis-dev libgoogle-glog-dev libgtest-dev libgmock-dev \\
  libsqlite3-dev libglew-dev libceres-dev libsuitesparse-dev libcurl4-openssl-dev libssl-dev \\
  libopenblas-openmp-dev
# COLMAP's docs: Ubuntu's OpenImageIO CMake config wants this folder, and with OpenBLAS its OpenMP
# build must be the one selected.
mkdir -p /usr/include/opencv4
for lib in libblas.so.3 liblapack.so.3 libopenblas.so.0; do
  update-alternatives --set "$lib-x86_64-linux-gnu" "/usr/lib/x86_64-linux-gnu/openblas-openmp/$lib"
done

# 2. uv 0.13.9 into /usr/local/bin, from its release tarball, checked against the release's sha256.
curl -fsSL --retry 3 -o /tmp/uv.tar.gz \\
  https://github.com/astral-sh/uv/releases/download/0.13.9/uv-x86_64-unknown-linux-gnu.tar.gz
echo "a7fbaff2b9a271d85c1bdb540941090fd1ce011797b4abeb48e4dcd39c2471d5  /tmp/uv.tar.gz" | sha256sum -c -
mkdir -p /tmp/uv
tar -xzf /tmp/uv.tar.gz --no-same-owner --strip-components 1 -C /tmp/uv
install -m 0755 /tmp/uv/uv /tmp/uv/uvx /usr/local/bin/

# 3. Our Python environment in {JOB_ROOT}/.venv, from HEAD's manifests, which the
#    Image_Build_Command uploaded for this build: uv.lock alone is over the 16 KB user data limit.
#    Never run uv sync on the image after step 6, because it would remove the GPU packages.
mkdir -p {JOB_ROOT}
for name in {names}; do
  aws s3 cp "s3://$BUCKET/$MANIFESTS$name" "{JOB_ROOT}/$name" --region "$REGION" --only-show-errors
done
cd {JOB_ROOT}
uv sync --frozen --no-dev

# 4. ffmpeg 7.1 with the zscale filter, from zimg 3.0.6. The private image may carry ffmpeg's GPL
#    build because the image is never shared or made public.
git clone --branch release-3.0.6 --depth 1 https://github.com/sekrit-twc/zimg.git /tmp/zimg
cd /tmp/zimg
./autogen.sh
./configure --prefix=/usr/local
make -j"$(nproc)"
make install
ldconfig
git clone --branch n7.1 --depth 1 https://github.com/FFmpeg/FFmpeg.git /tmp/ffmpeg
cd /tmp/ffmpeg
./configure --prefix=/usr/local --enable-gpl --enable-libzimg
make -j"$(nproc)"
make install

# 5. COLMAP 4.2.1 from source. CUDA, OpenGL and the GUI off keep SiftGPU (non-commercial) out, CGAL
#    off keeps out GPL code and LSD off AGPL code, so SIFT runs on the CPU. ONNX off leaves out the
#    learned features and the ONNX Runtime binaries they'd download. SIFT stays.
git clone --branch 4.2.1 --depth 1 https://github.com/colmap/colmap.git /tmp/colmap
cmake -S /tmp/colmap -B /tmp/colmap/build -GNinja -DCMAKE_BUILD_TYPE=Release \\
  -DCUDA_ENABLED=OFF -DOPENGL_ENABLED=OFF -DGUI_ENABLED=OFF -DCGAL_ENABLED=OFF -DLSD_ENABLED=OFF \\
  -DONNX_ENABLED=OFF -DBLA_VENDOR=OpenBLAS
cmake --build /tmp/colmap/build --target install
ldconfig

# 6. The GPU packages, which live only on the image. torch 2.7.1's wheels are built for CUDA 12.8 and
#    torch won't build with an nvcc of another major version, while the base image carries CUDA 13
#    only, so the builds use NVIDIA's CUDA Toolkit 12.8. gsplat 1.5.3, from its sdist, and fused-ssim,
#    at the commit gsplat 1.5.3's examples pin, compile now for the A10G (8.6) and the L4 and L40S
#    (8.9), so nothing compiles when a job starts. fused-ssim's setup.py targets the GPU it sees, the
#    builder's A10G, instead, and its sm_86 code runs on the 8.9 cards too.
curl -fsSL --retry 3 -o /tmp/cuda-keyring.deb \\
  https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
echo "d2a6b11c096396d868758b86dab1823b25e14d70333f1dfa74da5ddaf6a06dba  /tmp/cuda-keyring.deb" | sha256sum -c -
dpkg -i /tmp/cuda-keyring.deb
apt_get update
apt_get install -y cuda-toolkit-12-8
uv pip install --python "$VENV_PY" --index-url https://download.pytorch.org/whl/cu128 \\
  torch==2.7.1 torchvision==0.22.1
uv pip install --python "$VENV_PY" setuptools==84.2.1 wheel==0.48.0 ninja==1.13.2
CUDA_HOME=/usr/local/cuda-12.8 TORCH_CUDA_ARCH_LIST="8.6;8.9" \\
  uv pip install --python "$VENV_PY" --no-build-isolation --no-binary gsplat gsplat==1.5.3
CUDA_HOME=/usr/local/cuda-12.8 TORCH_CUDA_ARCH_LIST="8.6;8.9" \\
  uv pip install --python "$VENV_PY" --no-build-isolation \\
  "fused-ssim @ git+https://github.com/rahul-goel/fused-ssim@328dc9836f513d00c4b5bc38fe30478b4435cbb5"
# The build fails here unless both compiled modules import.
"$VENV_PY" -c "from gsplat import csrc; import fused_ssim"

# 7. The MoGe-2 ViT-B normal weights, pinned to one revision and checked against their sha256.
mkdir -p {moge_dir}
curl -fsSL --retry 3 -o {MOGE_WEIGHTS} \\
  https://huggingface.co/Ruicheng/moge-2-vitb-normal/resolve/ca5f0e07ff01d3e5a364c1d954ed12ee1814b368/model.pt
echo "{MOGE_SHA256}  {MOGE_WEIGHTS}" | sha256sum -c -

# 8. Node 22 and splat-transform, installed with npm ci from the committed package-lock.json, so the
#    version is reproducible.
curl -fsSL https://deb.nodesource.com/setup_22.x | bash -
apt_get install -y nodejs
cd {JOB_ROOT}
npm ci

# 9. Remove what a Job machine must not inherit, then signal the install finished. cloud-init clean
#    clears the builder's user data, logs and machine id so the image boots as a fresh first boot;
#    removing ~/.aws leaves no credentials that could beat the instance role.
rm -rf /root/.aws
cloud-init clean --logs --machine-id

# The Image_Build_Command waits for this line on the console before it stops and images the builder.
echo "{INSTALL_DONE_MARKER}"
echo "{INSTALL_DONE_MARKER}" > /dev/console 2>/dev/null || true
"""
    if len(script.encode("utf-8")) > _MAX_BYTES:
        raise CloudError(f"The image builder's install script came out larger than {_MAX_BYTES} "
                         "bytes, which is the EC2 user data limit. Shorten its comments, or move a "
                         "step into a file the builder downloads from the bucket.")
    return script


def build_image(*, aws: Aws, config: Config, cloud_config: CloudConfig, outputs: StackOutputs,
                repo_root: Path, environ: Mapping[str, str] | None = None) -> str:
    """Build the Machine_Image and return its id. check_spend first (Req 9.10).

    In order: build the money quote from Config and pass the Spend_Guard, so nothing paid starts
    before the yes (Req 9.10); read the Base_AMI id from the Base_AMI_Parameter (Req 9.1); admit the
    `image_builder_type` against the real G_Quota; name the image and upload HEAD's MANIFESTS under
    its `jobs/_image/<name>/` prefix, which the builder downloads; launch the builder from the
    Base_AMI with the Requirement 8 request, a root-volume `BlockDeviceMappings` entry sized from
    `root_gb` and `install_script` as the User_Data (Req 9.9); wait for the install to finish on
    the console, terminating the builder and stopping with a Readable_Error if it fails or runs past
    the Time_Cap; stop the builder through the EC2 API and wait for it to stop (never triggering
    terminate-on-shutdown, the skill's stop-first rule); make the Machine_Image from the stopped
    builder, under the name from before the launch, tagging the image and its snapshots with the
    Project_Tag and the two lock-file hashes in the one call (Req 9.7, 9.8); wait until it is
    `available`; terminate the builder; and append a `LABEL_IMAGE` Spend_Line for the builder's
    measured time to `data/spend.jsonl` (Req 12.2). The Runner_Profile is used throughout (Req 1.5).
    """
    cc = cloud_config
    builder_type = cc.image_builder_type
    spec = cc.spec(builder_type)

    # 1. The money quote, then the Spend_Guard, before anything paid or any AWS read (Req 9.10).
    machine, expected_time, cost = _spend_quote(builder_type, spec, cc)
    check_spend(machine=machine, expected_time=expected_time, rough_cost=cost, environ=environ)

    # 2. The current Base_AMI id, from the public SSM parameter the Read_Policy may read (Req 9.1).
    base_ami = aws.ssm_parameter(cc.base_ami_parameter)

    # 3. Admit the builder type against the real G_Quota, counting our running machines (Req 4).
    quota = aws.gpu_quota()
    running = running_vcpus(aws.tagged_instances(_LIVE_STATES), cc)
    admitted = admit([builder_type], running, quota, cc)
    plan = launch_plan(admitted, aws, outputs, cc)

    # 4. The lock-file hashes to tag on the image, so the Start_Command's pre-check can compare them
    #    with HEAD's and the machine's backstop can re-check (Req 9.8).
    uv_hash, package_hash = lock_hashes(cwd=repo_root)
    project_tag = _project_tag_pair(config)

    # 5. The image's name, fixed before the launch because the builder reads its manifests from
    #    jobs/_image/<name>/. The script is rendered before the upload, so a value it refuses stops
    #    the build before anything goes into the bucket.
    name = f"{project_tag[1]}-image-{_utc_now().strftime('%Y%m%d-%H%M%S')}"
    prefix = manifest_prefix(name)
    user_data = install_script(cc, bucket=outputs.bucket, region=aws.region, prefix=prefix)
    _upload_manifests(aws, repo_root, prefix)

    # 6. Launch the builder from the Base_AMI with the Requirement 8 request plus the root-volume
    #    size, trying each admitted zone in turn (Req 9.9).
    def build(attempt: int, instance_type: str, subnet_id: str) -> dict[str, Any]:
        request = build_run_instances_request(
            image_id=base_ami, instance_type=instance_type, subnet_id=subnet_id,
            instance_profile_arn=outputs.job_instance_profile_arn,
            security_group_id=outputs.security_group_id, user_data=user_data,
            client_token=client_token(LABEL_IMAGE, attempt), project_tag=project_tag, job_id=None)
        request["BlockDeviceMappings"] = [_root_device_mapping(cc.root_gb)]
        return request

    started = _utc_now()
    builder, _type, _zone = launch_with_fallback(aws, plan, build)

    # 7. Wait for the install to finish. A failed install powers itself off; a failed or timed-out
    #    one is terminated here too, so no builder outlives the error. Then stop the builder through
    #    the EC2 API (never triggering terminate-on-shutdown) and wait for it to stop, so CreateImage
    #    runs on a stopped machine and never has to reboot it for consistent snapshots (Req 9.9, the
    #    skill's stop-first rule).
    try:
        _wait_for_install(aws, builder.id, cc)
    except CloudError:
        aws.terminate_instances([builder.id])
        raise
    aws.stop_instance(builder.id)
    aws.wait_stopped(builder.id)

    # 8. Make the image from the stopped builder, under the name its manifests went under, tagging
    #    it and its snapshots with the Project_Tag and the two lock hashes in the one call, then wait
    #    until it is available (Req 9.7, 9.8).
    tags = {project_tag[0]: project_tag[1], "uv_lock_hash": uv_hash, "package_lock_hash": package_hash}
    image_id = aws.create_image(builder.id, name, tags)
    # Up to the Image_Build's Time_Cap, polling every poll_seconds: the first snapshots of a large
    # root volume can take far longer than botocore's default 10 minutes.
    aws.wait_image_available(image_id, poll_seconds=cc.poll_seconds,
                             max_minutes=cc.image.cap_minutes)

    # 9. Terminate the builder and record what it cost, as an estimate, under data/ (Req 9.9, 12.2).
    aws.terminate_instances([builder.id])
    seconds = max(0.0, (_utc_now() - started).total_seconds())
    _append_image_spend_line(repo_root, image_id, builder_type, seconds, spec.price_per_hour_usd)
    return image_id


def main(argv: list[str] | None = None) -> int:
    """The Image_Build_Command CLI: build the Machine_Image and print its id alone on stdout.

    `uv run python -m cloud.image`. A refusal (the Spend_Guard said no, a full quota, no capacity) or
    a failure prints its Readable_Error on stderr and returns 1 through `run_cloud_cli`, which redacts
    any account id first. It builds the Runner_Profile `Aws` from `[aws].runner_profile` (Req 1.5).
    """
    def body() -> int:
        _parser().parse_args(argv)
        config = load_config()
        cloud_config = load_cloud_config()
        aws = laptop_aws(config, profile_key="runner_profile")
        outputs = resolve_stack(aws, _aws_setting(config, "stack_name"))
        aws.bucket = outputs.bucket  # set before any S3 call: the build uploads its manifests
        image_id = build_image(aws=aws, config=config, cloud_config=cloud_config, outputs=outputs,
                               repo_root=Path(__file__).resolve().parents[1])
        print(image_id)
        return 0

    return run_cloud_cli(body)


# Waiting for the install (step 6)

def _wait_for_install(aws: Aws, instance_id: str, cc: CloudConfig) -> None:
    """Wait until install_script prints its done marker on the builder's console (Req 9.9).

    The builder runs on the Job_Role, which has no EC2 permission to stop itself, so it signals the
    install is done by printing `INSTALL_DONE_MARKER`, which cloud-init sends to the console. This
    polls `console_output` every `poll_seconds` until the marker appears, bounded by the Image_Build's
    Time_Cap so a hung install never waits forever but ends in a Readable_Error instead. The
    `INSTALL_FAILED_MARKER` a failing step prints ends the wait at once, with its own error.
    """
    interval = max(1.0, float(cc.poll_seconds))
    attempts = max(1, int(cc.image.cap_minutes * 60 / interval) + 1)
    for attempt in range(attempts):
        console = aws.console_output(instance_id)
        if INSTALL_DONE_MARKER in console:
            return
        if INSTALL_FAILED_MARKER in console:
            raise CloudError(
                f"The image builder's install failed, so no image was made, and the builder "
                f"({instance_id}) has powered itself off. Read its console in your own terminal to "
                "see which step failed, fix that step, then run the image build again.")
        if attempt + 1 < attempts:
            time.sleep(interval)
    raise CloudError(
        f"The image builder ({instance_id}) didn't report the install finished within its "
        f"{cc.image.cap_minutes:g}-minute time cap, so it was ended. Look at the builder's console "
        f"(list the machines with `{_CONSOLE_HINT}` and read its console in your own terminal), "
        "then run the image build again.")


# Uploading the manifests (step 5)

def _upload_manifests(aws: Aws, repo_root: Path, prefix: str) -> None:
    """Upload HEAD's committed MANIFESTS under `prefix` in the Job_Bucket, for the builder to fetch.

    They are HEAD's bytes, not the working tree's, so the uv.lock the builder installs from is the
    one whose hash the image is tagged with (Req 9.8). All five are read from git before the first
    upload, so a missing one stops the build with nothing uploaded.
    """
    contents = {name: head_file(name, cwd=repo_root) for name in MANIFESTS}
    with tempfile.TemporaryDirectory(prefix="voxelis-image-") as tmp:
        for name, data in contents.items():
            path = Path(tmp) / name
            path.write_bytes(data)
            aws.s3_upload(path, f"{prefix}{name}")


# The launch request's root volume (step 5)

def _root_device_mapping(root_gb: float) -> dict[str, Any]:
    """The root-volume `BlockDeviceMappings` entry: a gp3 volume sized from `root_gb`.

    The launch sets the size because CreateImage can't resize a volume later (Req 9.9). The volume
    keeps `DeleteOnTermination` true so no machine leaves a paid volume behind (Req 9.7).
    """
    return {
        "DeviceName": _ROOT_DEVICE,
        "Ebs": {"VolumeSize": int(root_gb), "VolumeType": "gp3", "DeleteOnTermination": True},
    }


# The money quote (step 1)

def _spend_quote(builder_type: str, spec: MachineSpec, cc: CloudConfig) -> tuple[str, str, str]:
    """The machine, expected-time and rough-cost strings `check_spend` shows for approval (Req 6.3).

    The expected cost is the builder's price over the Image_Build's Expected_Time; the worst case is
    the same type run to its Time_Cap, the only backstop that bills.
    """
    expected = rough_cost(cc.image.expected_minutes, spec, cc)
    worst = rough_cost(cc.image.cap_minutes, spec, cc)
    machine = f"a {builder_type} image builder"
    expected_time = (f"about {cc.image.expected_minutes:g} min, up to the "
                     f"{cc.image.cap_minutes:g} min time cap")
    cost = f"about ${expected:.2f}, worst case about ${worst:.2f}"
    return machine, expected_time, cost


# The Spend_Line (step 8)

def _append_image_spend_line(repo_root: Path, image_id: str, instance_type: str, seconds: float,
                             price_per_hour_usd: float) -> None:
    """Append the builder's `LABEL_IMAGE` Spend_Line to `data/spend.jsonl` directly (Req 12.2, 12.5).

    The laptop Image_Build writes straight into `data/spend.jsonl`, which git ignores, keyed by the
    new image id so the line is unique and a re-merge by the Status_Command stays idempotent. The
    line is always an estimate and prices the builder's measured time at the type's Config price.
    """
    line = spend_line(job_id=image_id, label=LABEL_IMAGE, instance_type=instance_type,
                      seconds=seconds, price_per_hour_usd=price_per_hour_usd, index=0)
    path = Path(repo_root) / "data" / "spend.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(line, ensure_ascii=True, allow_nan=False)
    with open(path, "a", encoding="utf-8", newline="\n") as handle:
        handle.write(rendered + "\n")


# Reading [aws], which is not a Pipeline_Section and has no accessor on splatpipe.config.Config

def _aws_setting(config: Config, key: str) -> str:
    """A non-empty string for `key` in the parsed `[aws]` table, or a CloudError naming it.

    `[aws]` is not a Pipeline_Section, so `splatpipe.config.Config` exposes no accessor for it and
    this spec adds none; its parsed table is read directly, as `cloud.aws.laptop_aws` reads the
    region and the profile.
    """
    data = getattr(config, "_data", None)
    aws = data.get("aws") if isinstance(data, dict) else None
    value = aws.get(key) if isinstance(aws, dict) else None
    if not isinstance(value, str) or not value:
        raise CloudError(f"[aws] in config.toml needs {key} as a non-empty string. Add it under "
                         "[aws], then run the command again.")
    return value


def _project_tag_pair(config: Config) -> tuple[str, str]:
    """The Project_Tag from `[aws].project_tag`, split into (key, value) for the launch and tags."""
    raw = _aws_setting(config, "project_tag")
    if raw.count("=") != 1 or "" in raw.split("="):
        raise CloudError('[aws] in config.toml needs project_tag as "key=value" text, like '
                         '"project=voxelis-kiroween". Fix it there, then run the command again.')
    key, value = raw.split("=", 1)
    return key, value


def _utc_now() -> datetime:
    """The time now, as an aware UTC datetime, for the builder's measured time."""
    return datetime.now(timezone.utc)


def _parser() -> argparse.ArgumentParser:
    """The command line: no arguments but `-h`, since the build takes its settings from config.toml."""
    return argparse.ArgumentParser(
        prog="uv run python -m cloud.image", allow_abbrev=False,
        description="Build the private machine image once, from the base AMI, on a builder machine. "
                    "A paid run: it asks the spend guard first. Prints the new image id.")


if __name__ == "__main__":
    sys.exit(main())
