"""The Image_Build: make our private Machine_Image once, from the Base_AMI, with a builder machine.

`install_script` is the builder's whole User_Data: the install, as bash, with every tool, weight and
package pinned. It is asserted structurally and never run in the Test_Suite (Req 9.2, 9.3, 9.4). It
never bakes in the Pipeline's own code; the Start_Command ships that per Job as the Code_Archive
(Req 9.5). The script never powers the machine off itself, because the launch sets
terminate-on-shutdown and an OS shutdown would terminate the builder before it could be imaged; when
the install is done it prints a done marker, which the Image_Build_Command waits for on the console.

`build_image` drives the build and is a Paid_Entry_Point: it passes the Spend_Guard first (Req 9.10),
then reads the Base_AMI id from the Base_AMI_Parameter, admits the `image_builder_type` against the
G_Quota, launches the builder from the Base_AMI with the Requirement 8 launch request plus a
root-volume `BlockDeviceMappings` entry (CreateImage can't resize a volume later) and `install_script`
as the User_Data, waits for the install, stops the builder through the EC2 API (never triggering
terminate-on-shutdown, the skill's stop-first rule), makes the Machine_Image from the stopped builder
with the Project_Tag and the `uv.lock`/`package-lock.json` hashes tagged on the image and its
snapshots in the one call (Req 9.7, 9.8), waits until it is `available`, terminates the builder, and
appends a `LABEL_IMAGE` Spend_Line for the builder's measured time (Req 9.9, 12.2). It uses the
Runner_Profile throughout (Req 1.5).

A cloud module may import `splatpipe` to call its public functions; `splatpipe` never imports
`cloud`. This module imports no boto3 of its own (only `cloud.aws` does) and no GPU package, so it
loads on a laptop.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from splatpipe.config import load_config

from cloud.aws import laptop_aws
from cloud.config import load_cloud_config
from cloud.errors import CloudError, run_cloud_cli
from cloud.launch import build_run_instances_request, client_token, launch_with_fallback
from cloud.repo import lock_hashes
from cloud.select import admit, launch_plan, running_vcpus
from cloud.spend import check_spend
from cloud.spend_file import LABEL_IMAGE, rough_cost, spend_line
from cloud.stack import resolve_stack

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
# The non-terminated instance states, so running_vcpus counts every machine still on the G_Quota.
_LIVE_STATES = ("pending", "running", "shutting-down", "stopping", "stopped")
# The builder's root device for the Deep Learning Base AMI (Ubuntu 24.04), whose size the launch
# sets in BlockDeviceMappings because CreateImage can't resize a volume later (Req 9.9, the skill).
_ROOT_DEVICE = "/dev/sda1"
# The read command a timed-out build points a person at, to look at the builder's console by hand.
_CONSOLE_HINT = "uv run python -m cloud.list"


def install_script(cc: CloudConfig) -> str:
    """The builder's User_Data: the whole install, as bash. Asserted structurally, never run.

    Pinned throughout (Req 9.2, 9.3, 9.4): our Python environment from `uv.lock`; ffmpeg built with
    the zscale filter (libzimg); COLMAP 4.2.1 from source with SiftGPU, CGAL and LSD all off; gsplat
    1.5.3 and fused-ssim compiled for the A10G (8.6) and the L4 and L40S (8.9); the MoGe-2 weights;
    Node and splat-transform from `package-lock.json` with `npm ci`. It then removes what a Job
    machine must not inherit and prints the done marker. It never bakes in the Pipeline's own code
    (Req 9.5) and never powers the machine off itself, because the launch's terminate-on-shutdown
    would turn an OS shutdown into a termination before the builder could be imaged; the
    Image_Build_Command stops the builder through the EC2 API instead.
    """
    cap_minutes = int(cc.image.cap_minutes)
    return f"""#!/bin/bash
# The image builder's install, run once by cloud-init as root on first boot. It never powers the
# machine off itself (terminate-on-shutdown would terminate the builder before it could be imaged)
# and never bakes in the Pipeline's own code: the Start_Command ships that per job as the
# Code_Archive. The committed uv.lock, pyproject.toml, package.json and package-lock.json are
# dependency manifests, not code, and are staged in /opt/job for `uv sync` and `npm ci`.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive

# Time_Cap backstop: if the install ever hangs, this OS shutdown (with terminate-on-shutdown) ends
# the builder so it can't run past the time that was approved. The normal path finishes well inside
# it and leaves the builder running for the Image_Build_Command to stop and image.
shutdown -h +{cap_minutes}

# 1. Our Python environment, from the committed uv.lock, into /opt/job/.venv. The code itself is
#    never baked in; each job extracts its Code_Archive over /opt/job at run time.
mkdir -p /opt/job
cd /opt/job
uv sync --frozen --no-dev

VENV_PY=/opt/job/.venv/bin/python

# 2. ffmpeg, built with the zscale filter (libzimg) for colour-space work. The private image may
#    carry ffmpeg's GPL build because the image is never shared or made public.
git clone --branch release-0.4.1 --depth 1 https://github.com/sekrit-twc/zimg.git /tmp/zimg
cd /tmp/zimg && ./autogen.sh && ./configure --prefix=/usr/local && make -j"$(nproc)" && make install
git clone --branch n7.1 --depth 1 https://github.com/FFmpeg/FFmpeg.git /tmp/ffmpeg
cd /tmp/ffmpeg && ./configure --prefix=/usr/local --enable-gpl --enable-libzimg
make -j"$(nproc)" && make install

# 3. COLMAP 4.2.1 from source, with SiftGPU (non-commercial), CGAL (GPL) and LSD (AGPL) all off, so
#    SIFT runs on the CPU and no copyleft or non-commercial part is built in.
git clone --branch 4.2.1 --depth 1 https://github.com/colmap/colmap.git /tmp/colmap
cmake -S /tmp/colmap -B /tmp/colmap/build -GNinja \\
  -DCUDA_ENABLED=OFF -DOPENGL_ENABLED=OFF -DGUI_ENABLED=OFF -DCGAL_ENABLED=OFF -DLSD_ENABLED=OFF
cmake --build /tmp/colmap/build --target install

# 4. The GPU packages, which live only on the image: torch's CUDA wheels, then gsplat 1.5.3 and
#    fused-ssim compiled for the A10G (8.6) and the L4 and L40S (8.9) so nothing compiles at job
#    start. TORCH_CUDA_ARCH_LIST pins the one image to those architectures (Req 9.3).
"$VENV_PY" -m pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.5.1
TORCH_CUDA_ARCH_LIST="8.6;8.9" "$VENV_PY" -m pip install --no-build-isolation gsplat==1.5.3
TORCH_CUDA_ARCH_LIST="8.6;8.9" "$VENV_PY" -m pip install --no-build-isolation \\
  "fused-ssim @ git+https://github.com/rahul-goel/fused-ssim@328dc9836f513d00c4f5bc5a50cf9e9e139c0a2"

# 5. The MoGe-2 depth model weights, pinned and fetched into the image so a job needs no download.
"$VENV_PY" -m pip install "huggingface-hub==0.26.2"
"$VENV_PY" -m huggingface_hub.commands.huggingface_cli download Ruicheng/moge-2-vitb-normal \\
  --revision v1.0 --local-dir /opt/models/moge-2-vitb-normal

# 6. Node and splat-transform, installed with npm ci from the committed package-lock.json, so the
#    version is reproducible.
curl -fsSL https://deb.nodesource.com/setup_22.x | bash -
apt-get install -y nodejs
cd /opt/job
npm ci

# 7. Remove what a Job machine must not inherit, then signal the install finished. cloud-init clean
#    clears the builder's user data, logs and machine id so the image boots as a fresh first boot;
#    removing ~/.aws leaves no credentials that could beat the instance role.
rm -rf /root/.aws ~/.aws
cloud-init clean --logs --machine-id

# The Image_Build_Command waits for this line on the console before it stops and images the builder.
echo "{INSTALL_DONE_MARKER}"
echo "{INSTALL_DONE_MARKER}" > /dev/console 2>/dev/null || true
"""


def build_image(*, aws: Aws, config: Config, cloud_config: CloudConfig, outputs: StackOutputs,
                repo_root: Path, environ: Mapping[str, str] | None = None) -> str:
    """Build the Machine_Image and return its id. check_spend first (Req 9.10).

    In order: build the money quote from Config and pass the Spend_Guard, so nothing paid starts
    before the yes (Req 9.10); read the Base_AMI id from the Base_AMI_Parameter (Req 9.1); admit the
    `image_builder_type` against the real G_Quota; launch the builder from the Base_AMI with the
    Requirement 8 request, a root-volume `BlockDeviceMappings` entry sized from `root_gb` and
    `install_script` as the User_Data (Req 9.9); wait for the install to finish on the console; stop
    the builder through the EC2 API and wait for it to stop (never triggering terminate-on-shutdown,
    the skill's stop-first rule); make the Machine_Image from the stopped builder, tagging the image
    and its snapshots with the Project_Tag and the two lock-file hashes in the one call (Req 9.7,
    9.8); wait until it is `available`; terminate the builder; and append a `LABEL_IMAGE` Spend_Line
    for the builder's measured time to `data/spend.jsonl` (Req 12.2). The Runner_Profile is used
    throughout (Req 1.5).
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
    user_data = install_script(cc)

    # 5. Launch the builder from the Base_AMI with the Requirement 8 request plus the root-volume
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

    # 6. Wait for the install to finish, then stop the builder through the EC2 API (never triggering
    #    terminate-on-shutdown) and wait for it to stop, so CreateImage runs on a stopped machine and
    #    never has to reboot it for consistent snapshots (Req 9.9, the skill's stop-first rule).
    _wait_for_install(aws, builder.id, cc)
    aws.stop_instance(builder.id)
    aws.wait_stopped(builder.id)

    # 7. Make the image from the stopped builder, tagging it and its snapshots with the Project_Tag
    #    and the two lock hashes in the one call, then wait until it is available (Req 9.7, 9.8).
    name = f"{project_tag[1]}-image-{started.strftime('%Y%m%d-%H%M%S')}"
    tags = {project_tag[0]: project_tag[1], "uv_lock_hash": uv_hash, "package_lock_hash": package_hash}
    image_id = aws.create_image(builder.id, name, tags)
    aws.wait_image_available(image_id)

    # 8. Terminate the builder and record what it cost, as an estimate, under data/ (Req 9.9, 12.2).
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
        aws.bucket = outputs.bucket  # set before any S3 call, though the build writes spend locally
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
    Time_Cap so a hung install never waits forever but ends in a Readable_Error instead.
    """
    interval = max(1.0, float(cc.poll_seconds))
    attempts = max(1, int(cc.image.cap_minutes * 60 / interval) + 1)
    for attempt in range(attempts):
        if INSTALL_DONE_MARKER in aws.console_output(instance_id):
            return
        if attempt + 1 < attempts:
            time.sleep(interval)
    raise CloudError(
        f"The image builder didn't report the install finished within its {cc.image.cap_minutes:g}-"
        f"minute time cap. Look at the builder's console (list the machines with `{_CONSOLE_HINT}` "
        "and read its console in your own terminal), then run the image build again.")


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
