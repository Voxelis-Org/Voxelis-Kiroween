"""The Smoke_Test: a short, cheap run that proves a fresh Machine_Image works before any long Job.

`run_smoke` is the laptop side of the Smoke_Command. It quotes the cost from Config and passes the
Spend_Guard first, so nothing paid starts before the yes (Req 6.1, 11.4). Only then does it find the
current Machine_Image, admit the `smoke_instance_type` against the real G_Quota, upload the
Code_Archive of HEAD and launch one machine from that image with the Requirement 8 request and the
Smoke_Test's Time_Cap (Req 11.1). That machine's User_Data runs every GPU code path for a few
seconds on a tiny synthetic scene and runs the Cloud_Runner end to end on a tiny Fake_Step Job
(Req 11.2); on a pass it writes the Smoke_Marker keyed to the image id (Req 11.3), measures how long
it took to be ready (Req 11.5) and ends itself (Req 11.4). `run_smoke` then records a LABEL_SMOKE
Spend_Line for the machine, labelled an estimate (Req 12.2, 12.3). The GPU work runs only on the
real machine; the CPU Test_Suite checks the launch request, the marker-key round-trip and the
Spend_Line against the Fake_AWS.

`smoke_marker_key` and `image_id_from_marker` are the pure key helpers: the marker lives under
`jobs/` so the machine's Job_Role may write it and the Read_Profile may read it (Req 11.3), and the
two round-trip. `cloud.start` imports `smoke_marker_key` from here (lazily, so `import cloud.start`
works before this module exists) to gate a Job on a passing Smoke_Marker; the name and its result
stay stable.

`main` is the Smoke_Command CLI, `uv run python -m cloud.smoke`. It builds the Runner_Profile `Aws`
from `[aws].runner_profile`, resolves the Stack and runs the Smoke_Test (Req 1.5). The machine-side
GPU checks this module's User_Data points at run only on the real GPU machine, and each later spec
that adds a GPU code path adds that path to them (Req 11.5).

A cloud module may import `splatpipe` to call its public functions; `splatpipe` never imports
`cloud`. This module imports no boto3 (it talks to AWS only through the `Aws` seam) and no GPU
package, so it loads on a laptop.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

from splatpipe.config import load_config
from splatpipe.jobs import new_job_id

from cloud.aws import laptop_aws
from cloud.config import load_cloud_config
from cloud.errors import CloudError, run_cloud_cli
from cloud.launch import build_run_instances_request, client_token, launch_with_fallback
from cloud.repo import archive_head, lock_hashes
from cloud.select import admit, launch_plan, running_vcpus
from cloud.spend import check_spend
from cloud.spend_file import LABEL_SMOKE, rough_cost, spend_line
from cloud.stack import resolve_stack
from cloud.userdata import UserDataValues, render_user_data

if TYPE_CHECKING:
    from typing import Any

    from splatpipe.config import Config

    from cloud.aws import Aws, ImageInfo
    from cloud.config import CloudConfig
    from cloud.stack import StackOutputs

# The repo root on the laptop: cloud/smoke.py's own grandparent, the same way cloud.start and
# cloud.runner find it. Tests point this at a temp repo to control the archive and the lock hashes.
REPO_ROOT = Path(__file__).resolve().parents[1]
# The non-terminated instance states, so running_vcpus counts every machine still on the G_Quota.
_LIVE_STATES = ("pending", "running", "shutting-down", "stopping", "stopped")
# The Image_Build command, named in the refusal that points a user at it when no image is ready.
_IMAGE_COMMAND = "uv run python -m cloud.image"
# The Smoke_Marker key parts: jobs/_smoke/<image_id>.json, under jobs/ so the machine's Job_Role
# writes it and the Read_Profile reads it (Req 11.3).
_MARKER_PREFIX = "jobs/_smoke/"
_MARKER_SUFFIX = ".json"


def smoke_marker_key(image_id: str) -> str:
    """The Smoke_Marker key for `image_id`: `jobs/_smoke/<image_id>.json` (Req 11.3).

    It lives under `jobs/` so the machine (the Job_Role) may write it and the Read_Profile may read
    it. `cloud.start` imports this to gate a Job on a passing marker, so the result stays stable.
    """
    return f"{_MARKER_PREFIX}{image_id}{_MARKER_SUFFIX}"


def image_id_from_marker(key: str) -> str:
    """The Machine_Image id back out of a Smoke_Marker key; the inverse of `smoke_marker_key`.

    `image_id_from_marker(smoke_marker_key(image_id)) == image_id` for a non-empty image id, and
    `smoke_marker_key(image_id_from_marker(key)) == key` for a key this module makes, so the two
    round-trip. A key that isn't a Smoke_Marker key is a `CloudError`.
    """
    if key.startswith(_MARKER_PREFIX) and key.endswith(_MARKER_SUFFIX):
        image_id = key[len(_MARKER_PREFIX):-len(_MARKER_SUFFIX)]
        if image_id:
            return image_id
    raise CloudError(f"That isn't a smoke-marker key: a smoke marker is {_MARKER_PREFIX}<image "
                     f"id>{_MARKER_SUFFIX}, with a non-empty image id. Check the key.")


def run_smoke(*, aws: Aws, config: Config, cloud_config: CloudConfig, outputs: StackOutputs,
              environ: Mapping[str, str] | None = None) -> None:
    """Launch one Smoke_Test machine from the current Machine_Image and record its Spend_Line.

    Every check up to `check_spend` is free and uses Config only; the first S3 upload and the
    RunInstances call come after it returns, so nothing paid starts before the yes (Req 6.1, 11.4).
    The current image, the quota admission, the Code_Archive, the launch request and the LABEL_SMOKE
    Spend_Line are built from the `aws`, `config` and `cloud_config` the caller passes; `environ`
    defaults to the real environment and is only read by the Spend_Guard. On a pass the machine
    writes the Smoke_Marker and ends itself (Req 11.3, 11.4); this function never waits for it.
    """
    cc = cloud_config
    smoke_type = cc.smoke_instance_type
    spec = cc.spec(smoke_type)
    plan = cc.smoke  # the Smoke_Test's Expected_Time and Time_Cap, from [cloud.smoke]

    # 1. The money quote from Config, then the Spend_Guard. Nothing paid runs before it returns
    #    (Req 6.1, 6.3, 11.4). The quote needs no AWS, so a no from the guard costs nothing.
    machine = f"a {smoke_type} GPU machine for the smoke test"
    expected_time = f"about {plan.expected_minutes:g} min, up to the {plan.cap_minutes:g} min time cap"
    cost = (f"about ${rough_cost(plan.expected_minutes, spec, cc):.2f} at "
            f"${spec.price_per_hour_usd:g}/hour (priced {spec.price_date})")
    check_spend(machine=machine, expected_time=expected_time, rough_cost=cost, environ=environ)

    # BotoAws refuses an S3 call until its bucket is set; the bucket is a Stack_Output (Req 1.3).
    aws.bucket = outputs.bucket

    # 2. The current Machine_Image to prove, else point at the Image_Build (Req 11.1).
    image = _newest_available_image(aws)
    # 3. Admit the smoke type against the real G_Quota and lay out the launch attempts (Req 4, 5.1).
    running = running_vcpus(aws.tagged_instances(_LIVE_STATES), cc)
    admitted = admit([smoke_type], running, aws.gpu_quota(), cc)
    attempts = launch_plan(admitted, aws, outputs, cc)

    # 4. The first paid actions. The machine runs cloud.smoke and the Cloud_Runner at HEAD, so it
    #    needs the Code_Archive; upload it under this smoke run's own Job_Id (Req 11.2, 7.4).
    job_id = new_job_id()
    with tempfile.TemporaryDirectory(prefix="voxelis-smoke-") as tmp:
        archive = Path(tmp) / "code.tar"
        archive_head(archive, cwd=REPO_ROOT)
        uv_hash, package_hash = lock_hashes(cwd=REPO_ROOT)
        code_key = f"jobs/{job_id}/input/code.tar"
        aws.s3_upload(archive, code_key)

        # 5. The smoke User_Data (role "smoke" runs cloud.smoke on the machine) and the launch, with
        #    the Smoke_Test's Time_Cap and the Requirement 8 request through the fallback (Req 11.1).
        user_data = render_user_data(UserDataValues(
            job_id=job_id, bucket=outputs.bucket, region=aws.region,
            preset=config.jobs().default_preset, image_id=image.id, code_archive_key=code_key,
            cap_minutes=int(plan.cap_minutes), upload_margin_minutes=int(cc.upload_margin_minutes),
            uv_lock_hash=uv_hash, package_lock_hash=package_hash, role="smoke"))
        project_tag = _project_tag_pair(config)

        def build(attempt: int, instance_type: str, subnet_id: str) -> dict[str, Any]:
            return build_run_instances_request(
                image_id=image.id, instance_type=instance_type, subnet_id=subnet_id,
                instance_profile_arn=outputs.job_instance_profile_arn,
                security_group_id=outputs.security_group_id, user_data=user_data,
                client_token=client_token(job_id, attempt), project_tag=project_tag, job_id=job_id)

        instance, launched_type, launched_zone = launch_with_fallback(aws, attempts, build)

    # 6. Record a LABEL_SMOKE Spend_Line for the machine, labelled an estimate (Req 12.2, 12.3). The
    #    laptop can't time the self-terminating machine, so the estimate is the Expected_Time.
    _append_local_spend_line(config, spend_line(
        job_id=job_id, label=LABEL_SMOKE, instance_type=smoke_type,
        seconds=plan.expected_minutes * 60, price_per_hour_usd=spec.price_per_hour_usd, index=0))

    print(f"Launched a {launched_type} in {launched_zone} ({instance.id}) to smoke-test image "
          f"{image.id}. It writes {smoke_marker_key(image.id)} if it passes, then ends itself.")


def main(argv: list[str] | None = None) -> int:
    """The Smoke_Command CLI: smoke-test the current Machine_Image and return 0.

    A refusal (the Spend_Guard said no, no image is ready, the quota is full, or an AWS error)
    prints its Readable_Error on stderr and returns 1, through `run_cloud_cli`, which redacts any
    account id first. A bad command line is argparse's exit 2.
    """
    def body() -> int:
        _parser().parse_args(argv)
        config = load_config()
        aws = laptop_aws(config, profile_key="runner_profile")
        outputs = resolve_stack(aws, _aws_setting(config, "stack_name"))
        run_smoke(aws=aws, config=config, cloud_config=load_cloud_config(), outputs=outputs)
        return 0

    return run_cloud_cli(body)


# The read-only lookup

def _newest_available_image(aws: Aws) -> ImageInfo:
    """The newest available tagged Machine_Image, or a CloudError pointing at the Image_Build (Req 11.1).

    AWS returns `creation_date` as an ISO-8601 UTC string, so the lexicographic max is the newest.
    """
    available = [image for image in aws.tagged_images() if image.state == "available"]
    if not available:
        raise CloudError(
            f"No Machine_Image is available yet, so there's nothing to smoke-test. Build one with "
            f"`{_IMAGE_COMMAND}`, then run this command again.")
    return max(available, key=lambda image: image.creation_date)


# The Spend_File append (the laptop side; the Status_Command and Download_Command merge later)

def _append_local_spend_line(config: Config, line: Mapping[str, Any]) -> None:
    """Append one Spend_Line to the laptop's `data/spend.jsonl` (Req 12.2, 12.5).

    The Spend_File sits beside the Jobs_Folder, under `data/`, which git ignores, so no Spend_Line
    lands in the repo. A fresh smoke run has its own Job_Id, so its line is new; the merge that the
    Status_Command and Download_Command do dedups the bucket's lines (Req 12.4), which this append
    doesn't need. The file and its folder are made when missing.
    """
    path = config.jobs().jobs_folder.parent / "spend.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(dict(line), ensure_ascii=True, allow_nan=False)
    with open(path, "a", encoding="utf-8", newline="\n") as handle:
        handle.write(f"{rendered}\n")


# Reading [aws], which is not a Pipeline_Section and has no accessor on splatpipe.config.Config

def _aws_setting(config: Config, key: str) -> str:
    """A non-empty string for `key` in the parsed `[aws]` table, or a CloudError naming it.

    `[aws]` is not a Pipeline_Section, so `splatpipe.config.Config` exposes no accessor for it and
    this spec adds none; its parsed table is read directly, as `cloud.aws.laptop_aws` reads it.
    """
    data = getattr(config, "_data", None)
    aws = data.get("aws") if isinstance(data, Mapping) else None
    value = aws.get(key) if isinstance(aws, Mapping) else None
    if not isinstance(value, str) or not value:
        raise CloudError(f"[aws] in config.toml needs {key} as a non-empty string. Add it under "
                         "[aws], then run the command again.")
    return value


def _project_tag_pair(config: Config) -> tuple[str, str]:
    """The Project_Tag from `[aws].project_tag`, split into (key, value) for the launch request."""
    raw = _aws_setting(config, "project_tag")
    if raw.count("=") != 1 or "" in raw.split("="):
        raise CloudError('[aws] in config.toml needs project_tag as "key=value" text, like '
                         '"project=voxelis-kiroween". Fix it there, then run the command again.')
    key, value = raw.split("=", 1)
    return key, value


def _parser() -> argparse.ArgumentParser:
    """The command line: no arguments. The smoke test runs on the current Machine_Image."""
    return argparse.ArgumentParser(
        prog="uv run python -m cloud.smoke", allow_abbrev=False,
        description="Smoke-test the current machine image: launch one short GPU machine that runs "
                    "the GPU code paths and the cloud runner on a tiny fake-step job, and writes a "
                    "smoke marker when it passes. Passes the spend guard first.")


if __name__ == "__main__":
    sys.exit(main())
