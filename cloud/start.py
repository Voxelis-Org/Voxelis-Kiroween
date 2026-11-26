"""The Start_Command: make a Job from a phone video and launch it on one GPU machine.

`start_job` does every free, local and read-only check first and spends money only at the very
end, so a bad video, a dirty repo, a missing image, a drifted lock file, a missing Smoke_Marker or
a full quota fails for nothing, before anyone is asked to approve a paid run (Req 7, 6.1). In order:
refuse an empty Step_List with `splatpipe.run`'s own words (Req 7.2); refuse uncommitted
`splatpipe/`, `cloud/` or `config.toml` (Req 7.3); archive HEAD and hash the lock files (Req 7.4);
resolve the Stack; find the newest available Machine_Image, else point at the Image_Build (Req 7.9);
refuse a lock-hash drift against the image's tags (Req 7.5, 7.6); refuse a missing Smoke_Marker,
a hard gate pointing at the Smoke_Test (Req 7.7); make the Job, whose own checks fail here for free
(Req 7.1); choose the machine against the G_Quota (Req 3, 4, 5); build the money quote and call the
Spend_Guard (Req 6.1, 6.3, 3.6). Only once `check_spend` returns does it upload the video, the
Code_Archive and the Job_Record to `jobs/<id>/input/` (Req 7.8) and call RunInstances through the
fallback loop (Req 5.3), so no upload and no launch ever happens before the yes (Req 6.6).

`main` is the CLI, `uv run python -m cloud.start <video> [--start S] [--end E] [--preset P]`. It
builds the Runner_Profile `Aws` from `[aws].runner_profile` and prints the new Job_Id (Req 1.5).

A cloud module may import `splatpipe` to call its public functions; `splatpipe` never imports
`cloud`. This module imports no boto3 of its own (only `cloud.aws` does) and no GPU package, so it
loads on a laptop. `cloud.smoke` (task 7) owns `smoke_marker_key`; until it exists this module
computes the design's marker key inline, so `import cloud.start` works today.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import splatpipe.run
from splatpipe.config import PRESETS, load_config
from splatpipe.errors import PipelineError
from splatpipe.jobs import open_job
from splatpipe.make import make_job

from cloud.aws import ImageInfo, InstanceInfo, aws_setting, laptop_aws
from cloud.config import load_cloud_config
from cloud.errors import CloudError, run_cloud_cli
from cloud.launch import build_run_instances_request, client_token, launch_with_fallback
from cloud.repo import archive_head, head_commit, lock_hashes, uncommitted
from cloud.select import admit, launch_plan, running_vcpus, select_candidates, step_needs_over
from cloud.spend import check_spend
from cloud.spend_file import rough_cost
from cloud.stack import StackOutputs, resolve_stack
from cloud.userdata import UserDataValues, render_user_data

from cloud.config import PresetPlan
from splatpipe.run import StepEntry

if TYPE_CHECKING:
    from splatpipe.config import Config

    from cloud.aws import Aws
    from cloud.config import CloudConfig

# The paths every Job runs committed, checked for uncommitted changes before anything else (Req 7.3).
_CHECKED_PATHS = ("splatpipe", "cloud", "config.toml")
# The non-terminated instance states, so running_vcpus counts every machine still on the G_Quota.
_LIVE_STATES = ("pending", "running", "shutting-down", "stopping", "stopped")
# The Image_Build and Smoke_Test commands, named in the refusals that point a user at them.
_IMAGE_COMMAND = "uv run python -m cloud.image"
_SMOKE_COMMAND = "uv run python -m cloud.smoke"


@dataclass(frozen=True)
class Code:
    """The code a Job runs: its commit, its Code_Archive on disk and its two lock-file hashes."""

    commit: str
    archive: Path
    uv_lock_hash: str
    package_lock_hash: str


@dataclass(frozen=True)
class Gate:
    """What check_image found: the Stack's outputs and the image every gate passed for."""

    outputs: StackOutputs
    image: ImageInfo


@dataclass(frozen=True)
class Choice:
    """The machine choice and the money quote for one Job."""

    preset: str
    plan: PresetPlan
    candidates: tuple[str, ...]
    admitted: tuple[str, ...]
    attempts: tuple[tuple[str, str, str], ...]    # (instance_type, subnet_id, zone)
    machine: str                                  # the three Spend_Guard strings
    expected_time: str
    rough_cost: str


@dataclass(frozen=True)
class Launched:
    instance: InstanceInfo
    instance_type: str
    zone: str


def start_job(video: str | Path, *, preset: str | None = None, start: float | None = None,
              end: float | None = None, aws: Aws | None = None, config: Config | None = None,
              cloud_config: CloudConfig | None = None, repo_root: Path | None = None,
              environ: Mapping[str, str] | None = None) -> str:
    """Make and launch a Job, and return its Job_Id.

    Every check up to `check_spend` is free, local or read-only; the first S3 upload and the
    RunInstances call come only after it returns, so nothing paid starts before the yes (Req 6.1,
    6.6). `aws` defaults to the Runner_Profile `Aws`, `config` and `cloud_config` to the repo's
    config, `repo_root` to the repo, and `environ` to the real environment; tests pass their own.
    The body composes the Start_Core functions the website calls too, in today's order.
    """
    # 1. Refuse an empty Step_List with the exact words splatpipe.run uses, before anything else.
    entries = require_step_list()
    config = load_config() if config is None else config
    cloud_config = load_cloud_config() if cloud_config is None else cloud_config
    root = Path(__file__).resolve().parents[1] if repo_root is None else Path(repo_root)

    # 2. Refuse uncommitted splatpipe/, cloud/ or config.toml, naming the dirty paths (Req 7.3).
    dirty = uncommitted(_CHECKED_PATHS, cwd=root)
    if dirty:
        raise CloudError(
            f"These have uncommitted changes, so the job wouldn't run the committed code: "
            f"{', '.join(dirty)}. Commit or stash them, then run the command again.")

    with tempfile.TemporaryDirectory(prefix="voxelis-start-") as tmp:
        # 3. Archive HEAD's code and hash HEAD's lock files, all free local git work (Req 7.4).
        archive = Path(tmp) / "code.tar"
        head = head_commit(cwd=root)
        archive_head(archive, cwd=root)
        uv_hash, package_hash = lock_hashes(cwd=root)
        code = Code(commit=head, archive=archive, uv_lock_hash=uv_hash, package_lock_hash=package_hash)

        # 4-7. The Runner_Profile Aws (unless a test passed one), the Stack, the newest image, the
        #      lock-hash check and the Smoke_Marker.
        if aws is None:
            aws = laptop_aws(config, profile_key="runner_profile")
        gate = check_image(aws, config, code)

        # 8. Make the Job. make_job runs every video check here, for free, before any spend (Req 7.1).
        job_id = make_job(video, start=start, end=end, preset=preset, config=config)
        job = open_job(job_id, config)

        # 9-10. The machine choice against the G_Quota and the money quote (Req 3, 4, 5, 6.3).
        choice = choose_machine(aws, gate, entries, job.record.preset, config, cloud_config)

        # 11-12. The Spend_Guard first, then the uploads, then the fallback launch (Req 6.6, 7.8).
        def place_inputs(prefix: str, code_key: str) -> None:
            aws.s3_upload(job.video_path, f"{prefix}/{job.record.video_file}")
            aws.s3_upload(archive, code_key)
            aws.s3_upload(job.folder / "input" / "job.json", f"{prefix}/job.json")

        launched = launch_job(aws, gate, choice, job_id, code, place_inputs, config, cloud_config,
                              environ)
        print(f"Launched a {launched.instance_type} in {launched.zone} ({launched.instance.id}) for "
              f"job {job_id}, running the code at commit {head[:12]}.")
        return job_id


def require_step_list() -> tuple[StepEntry, ...]:
    """`splatpipe.run.STEP_LIST`, read at call time so a test's list is honoured, or the Pipeline's
    own refusal when it's empty (Req 7.2)."""
    entries = splatpipe.run.STEP_LIST
    if not entries:
        raise PipelineError(splatpipe.run._NO_STEPS)
    return tuple(entries)


def check_image(aws: Aws, config: Config, code: Code) -> Gate:
    """Resolve the Stack and set aws.bucket, take the newest available Machine_Image, refuse a
    lock-hash drift against `code`'s hashes, and refuse a missing Smoke_Marker (Req 7.5-7.7, 7.9)."""
    outputs = resolve_stack(aws, aws_setting(config, "stack_name"))
    aws.bucket = outputs.bucket  # BotoAws refuses an S3 call until its bucket is set
    image = _newest_available_image(aws)
    _check_lock_drift(image, code.uv_lock_hash, code.package_lock_hash)
    if not aws.s3_exists(_smoke_marker_key(image.id)):
        raise CloudError(
            f"The current Machine_Image {image.id} hasn't passed the smoke test, so no job "
            f"may run on it. Run `{_SMOKE_COMMAND}`, then run this command again.")
    return Gate(outputs=outputs, image=image)


def choose_machine(aws: Aws, gate: Gate, entries: Sequence[StepEntry], preset: str,
                   config: Config, cc: CloudConfig) -> Choice:
    """The hard needs, the candidates, the quota and the running vCPUs, what the quota admits, the
    ordered launch attempts and the money quote (Req 3, 4, 5, 6.3)."""
    needs = step_needs_over([entry.name for entry in entries], config, preset)
    candidates = select_candidates(preset, needs, cc)
    quota = aws.gpu_quota()
    running = running_vcpus(aws.tagged_instances(_LIVE_STATES), cc)
    admitted = admit(candidates, running, quota, cc)
    attempts = launch_plan(admitted, aws, gate.outputs, cc)
    plan = cc.preset_plan(preset)
    machine, expected_time, cost = _spend_quote(admitted, candidates, plan, cc)
    return Choice(preset=preset, plan=plan, candidates=tuple(candidates), admitted=tuple(admitted),
                  attempts=tuple(attempts), machine=machine, expected_time=expected_time,
                  rough_cost=cost)


def launch_job(aws: Aws, gate: Gate, choice: Choice, job_id: str, code: Code,
               place_inputs: Callable[[str, str], None], config: Config, cc: CloudConfig,
               environ: Mapping[str, str] | None) -> Launched:
    """The Spend_Guard first, before anything paid; then `place_inputs(prefix, code_key)` puts the
    video, the Code_Archive and job.json under jobs/<job_id>/input/; then the User_Data and the
    fallback launch. `check_spend` is read from this module at call time, so a test's patch reaches
    every caller. One yes covers the whole fallback, since only one machine ever launches."""
    check_spend(machine=choice.machine, expected_time=choice.expected_time,
                rough_cost=choice.rough_cost, environ=environ)
    prefix = f"jobs/{job_id}/input"
    code_key = f"{prefix}/code.tar"
    place_inputs(prefix, code_key)
    outputs, image = gate.outputs, gate.image
    user_data = render_user_data(UserDataValues(
        job_id=job_id, bucket=outputs.bucket, region=aws.region, preset=choice.preset,
        image_id=image.id, code_archive_key=code_key, cap_minutes=int(choice.plan.cap_minutes),
        upload_margin_minutes=int(cc.upload_margin_minutes), uv_lock_hash=code.uv_lock_hash,
        package_lock_hash=code.package_lock_hash, role="job"))
    project_tag = _project_tag_pair(config)

    def build(attempt: int, instance_type: str, subnet_id: str) -> dict[str, Any]:
        return build_run_instances_request(
            image_id=image.id, instance_type=instance_type, subnet_id=subnet_id,
            instance_profile_arn=outputs.job_instance_profile_arn,
            security_group_id=outputs.security_group_id, user_data=user_data,
            client_token=client_token(job_id, attempt), project_tag=project_tag, job_id=job_id)

    instance, launched_type, launched_zone = launch_with_fallback(aws, choice.attempts, build)
    return Launched(instance=instance, instance_type=launched_type, zone=launched_zone)


def main(argv: list[str] | None = None) -> int:
    """The Start_Command CLI: make the Job, print its Job_Id alone on stdout and return 0.

    A refusal (an empty Step_List, a dirty repo, a missing image or marker, a full quota, a bad
    video, or a Spend_Guard that said no) prints its Readable_Error on stderr and returns 1, through
    `run_cloud_cli`, which redacts any account id first. A bad command line is argparse's exit 2.
    """
    def body() -> int:
        args = _parser().parse_args(argv)
        config = load_config()
        aws = laptop_aws(config, profile_key="runner_profile")
        print(start_job(args.video, preset=args.preset, start=args.start, end=args.end,
                        aws=aws, config=config))
        return 0

    return run_cloud_cli(body)


# The read-only lookups

def _newest_available_image(aws: Aws) -> ImageInfo:
    """The newest available tagged Machine_Image, or a CloudError pointing at the Image_Build (Req 7.9).

    AWS returns `creation_date` as an ISO-8601 UTC string, so the lexicographic max is the newest.
    """
    available = [image for image in aws.tagged_images() if image.state == "available"]
    if not available:
        raise CloudError(
            f"No Machine_Image is available yet, so no job can run. Build one with "
            f"`{_IMAGE_COMMAND}`, then run this command again.")
    return max(available, key=lambda image: image.creation_date)


def _check_lock_drift(image: ImageInfo, uv_hash: str, package_hash: str) -> None:
    """Refuse a Machine_Image built from different lock files than HEAD's (Req 7.5, 7.6).

    The image carries the sha-256 of the `uv.lock` and `package-lock.json` it was built from as the
    `uv_lock_hash` and `package_lock_hash` tags; either one differing means the machine's packages
    wouldn't match HEAD's code, so the run is refused before any spend.
    """
    if image.tags.get("uv_lock_hash") == uv_hash and image.tags.get("package_lock_hash") == package_hash:
        return
    raise CloudError(
        f"The current Machine_Image {image.id} was built from different lock files than HEAD's: "
        "its uv.lock or package-lock.json hash doesn't match, so its packages wouldn't match the "
        f"code this job would run. Rebuild the image with `{_IMAGE_COMMAND}`, then run this command "
        "again.")


def _smoke_marker_key(image_id: str) -> str:
    """The Smoke_Marker key for `image_id`.

    `cloud.smoke` (task 7) owns `smoke_marker_key`; the import is done here, inside the function, so
    `import cloud.start` succeeds before that module exists. Until it does, the design's key is
    computed inline. A different `ModuleNotFoundError` (a real import bug in `cloud.smoke`) is
    re-raised rather than swallowed.
    """
    try:
        from cloud.smoke import smoke_marker_key
    except ModuleNotFoundError as error:
        if error.name != "cloud.smoke":
            raise
        return f"jobs/_smoke/{image_id}.json"
    return smoke_marker_key(image_id)


# The money quote for the Spend_Guard

def _spend_quote(admitted: Sequence[str], candidates: Sequence[str], plan: PresetPlan,
                 cc: CloudConfig) -> tuple[str, str, str]:
    """The machine, expected-time and rough-cost strings `check_spend` shows for approval (Req 6.3).

    The first admitted type is the one the launch loop tries first, so the expected cost is its
    price over the Preset's Expected_Time (Req 3.6). The worst case is the dearest type that fits
    the hard needs on the whole Fallback_List, run to the Preset's Time_Cap, since a capacity
    fallback could land on a dearer type than the one that launches first.
    """
    first = admitted[0]
    dearest = max(candidates, key=lambda name: cc.spec(name).price_per_hour_usd)
    expected = rough_cost(plan.expected_minutes, cc.spec(first), cc)
    worst = rough_cost(plan.cap_minutes, cc.spec(dearest), cc)
    machine = (f"a {first} GPU machine" if dearest == first
               else f"a {first} GPU machine, worst case a {dearest} on fallback")
    expected_time = f"about {plan.expected_minutes:g} min, up to the {plan.cap_minutes:g} min time cap"
    cost = f"about ${expected:.2f}, worst case about ${worst:.2f}"
    return machine, expected_time, cost


def worst_case_cost(preset: str, cc: CloudConfig, extra_minutes: float = 0.0) -> float:
    """`rough_cost` of the dearest type on the Preset's whole Fallback_List, run for its Time_Cap
    plus `extra_minutes`, in dollars. For the website's Reservations; `start_job` doesn't call it.
    The website passes `[website].boot_margin_minutes`, because a hung machine runs its boot minutes
    before the Time_Cap's shutdown is scheduled."""
    plan = cc.preset_plan(preset)
    dearest = max(plan.fallback, key=lambda name: cc.spec(name).price_per_hour_usd)
    return rough_cost(plan.cap_minutes + extra_minutes, cc.spec(dearest), cc)


# Reading [aws], which is not a Pipeline_Section and has no accessor on splatpipe.config.Config

def _aws_setting(config: Config, key: str) -> str:
    """A non-empty string for `key` in the parsed `[aws]` table, or a CloudError naming it.

    `[aws]` is not a Pipeline_Section, so `splatpipe.config.Config` exposes no accessor for it and
    this spec adds none; its parsed table is read directly, as `cloud.aws.laptop_aws` reads the
    region and the profile.
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
    """The command line: a video, and an optional --start, --end and --preset. There's no --job."""
    parser = argparse.ArgumentParser(
        prog="uv run python -m cloud.start", allow_abbrev=False,
        description="Make a job from a phone video and launch it on one GPU machine in the cloud. "
                    "Prints the new job id.")
    parser.add_argument("video", help="the MP4 or MOV video to make the job from")
    parser.add_argument("--start", type=float,
                        help="where the part of the video to use starts, in seconds (default: 0)")
    parser.add_argument("--end", type=float,
                        help="where the part of the video to use ends, in seconds "
                             "(default: the end of the video)")
    parser.add_argument("--preset", choices=PRESETS,
                        help="the preset (default: default_preset under [jobs] in config.toml)")
    return parser


if __name__ == "__main__":
    sys.exit(main())
