"""The Smoke_Test's machine side: the image's checks and the Smoke_Marker, run on the smoke machine.

`python -m cloud.smoke_runner --job <id>` is what the smoke machine's User_Data runs (role "smoke").
It's separate from the laptop's Smoke_Command, `cloud.smoke`, which only launches the machine.
`run_smoke_test` runs each check in SMOKE_CHECKS, a few seconds each (Req 11.2): torch sees the GPU,
gsplat renders a tiny synthetic scene forwards and backwards, fused-ssim runs forwards and
backwards, ffmpeg has the zscale and tonemap filters, COLMAP, Node and splat-transform run, the
MoGe-2 weights match their pinned sha256, the Trainer trains the tiny Smoke_Scene on the GPU and the
PLY it gives passes the PLY_Check, and the Cloud_Runner runs a tiny Fake_Step Job end to end.
For that last check it makes the Job from a test video, uploads its input, and runs `run_cloud_job`,
which downloads the input again, runs the Step and uploads the results. A failing check doesn't stop
the others. The results are printed on the console and put, Account_Redacted, at
`jobs/_smoke/<image id>.log` in the Job_Bucket, pass or fail. When every check passes it writes the
Smoke_Marker, `jobs/_smoke/<image id>.json`, with how long the machine took to be ready for work
(Req 11.3, 11.5). Either way it powers the machine off (Req 11.4).

SMOKE_CHECKS is the list later specs extend: each spec that adds a GPU code path adds a check for it
here, starting with training (Req 11.5). A check is a function that gets a SmokeContext and gives
back a short line for the log, or raises when the check fails. Its numbers come from
`[cloud.smoke]`.

The GPU checks import torch, gsplat and fused-ssim inside their functions, and the training check
imports the GPU_Module inside its function with `load_gpu`, so this module imports on a laptop.
There the Test_Suite runs `run_smoke_test` with stand-in checks, and the checks that need no GPU,
on the CPU with the Fake_AWS. A cloud module may import `splatpipe` to call its public functions;
`splatpipe` never imports `cloud`. This module imports no boto3 of its own.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from splatpipe.config import load_config
from splatpipe.errors import PipelineError
from splatpipe.frames.ffmpeg import has_filters
from splatpipe.jobs import check_job_id, job_folder, remove_tree
from splatpipe.make import make_job
from splatpipe.run import StepEntry
from splatpipe.train.gpu_api import load_gpu
from splatpipe.train.smoke import SmokeSizes, run_smoke_training

from cloud.aws import job_machine_aws
from cloud.config import load_cloud_config
from cloud.errors import CloudError, run_cloud_cli
from cloud.image import MOGE_SHA256, MOGE_WEIGHTS, SPLAT_TRANSFORM
from cloud.redact import account_redaction
from cloud.runner import run_cloud_job
from cloud.smoke import smoke_log_key, smoke_marker_key

if TYPE_CHECKING:
    from splatpipe.config import Config

    from cloud.aws import Aws
    from cloud.config import CloudConfig

# The variable the User_Data exports with the id of the Machine_Image this machine runs.
IMAGE_ID_ENV = "IMAGE_ID"
# The kernel's seconds since boot. Its first number, read as the checks start, is how long the
# machine took to be ready for work (Req 11.5). Tests point this at a temp file.
PROC_UPTIME = Path("/proc/uptime")
# The two filters the frames step's HDR tone map needs. zscale comes from zimg.
_TONE_MAP_FILTERS = ("zscale", "tonemap")
# The Step the Cloud_Runner check runs: the Fake_Step, required, under the real [fake] section.
_FAKE_STEP = StepEntry("fake", "splatpipe.fake", True)


@dataclass(frozen=True)
class SmokeContext:
    """What a check may use: this smoke run's Job_Id and Aws, the loaded configs, and an empty
    scratch folder of the check's own, removed after it."""

    job_id: str
    aws: Aws
    config: Config
    cloud_config: CloudConfig
    scratch: Path


@dataclass(frozen=True)
class SmokeCheck:
    """One check: its name in the log, and the function that runs it. The function gives back a
    short line for the log, or raises when the check fails."""

    name: str
    run: Callable[[SmokeContext], str]


@dataclass(frozen=True)
class CheckResult:
    """How one check went: whether it passed, its line for the log and how long it took."""

    name: str
    passed: bool
    detail: str
    seconds: float


def _utc_now() -> datetime:
    """The time now, as an aware UTC datetime. The default `now` for run_smoke_test."""
    return datetime.now(timezone.utc)


def _poweroff() -> None:
    """Power the machine off, so terminate-on-shutdown ends it (Req 11.4). The default `shutdown`."""
    subprocess.run(["poweroff"], check=False)


def run_smoke_test(job_id: str, *, aws: Aws, config: Config, cloud_config: CloudConfig,
                   image_id: str, boot_to_ready_seconds: float,
                   checks: Sequence[SmokeCheck] | None = None,
                   now: Callable[[], datetime] = _utc_now,
                   shutdown: Callable[[], None] = _poweroff) -> bool:
    """Run the checks, put their results in the bucket, write the Smoke_Marker on a pass, power off.

    Returns whether every check passed. `checks` defaults to SMOKE_CHECKS, read at call time. Each
    check runs with a scratch folder of its own, and one that raises fails without stopping the
    rest. Each result is printed as it comes, and the whole log is put, Account_Redacted, at
    `smoke_log_key(image_id)`, pass or fail. Only when every check passed is the Smoke_Marker put at
    `smoke_marker_key(image_id)`, with the image id, the time it passed and
    `boot_to_ready_seconds` (Req 11.3, 11.5). The machine is powered off in a `finally`, so it ends
    even on an error (Req 11.4). Tests pass their own `checks`, `now` and `shutdown` and the
    Fake_AWS.
    """
    try:
        listed = SMOKE_CHECKS if checks is None else tuple(checks)
        context = functools.partial(SmokeContext, job_id, aws, config, cloud_config)
        lines = [f"Smoke test of image {image_id}, as job {job_id}. The machine was ready for "
                 f"work {boot_to_ready_seconds:.1f} s after it booted."]
        _say(lines[-1])
        results: list[CheckResult] = []
        for check in listed:
            results.append(_run_check(check, context))
            lines.append(_result_line(results[-1]))
            _say(lines[-1])
        passed = bool(results) and all(result.passed for result in results)
        lines.append(_verdict_line(results, passed, image_id))
        _say(lines[-1])
        aws.s3_put_text(smoke_log_key(image_id), account_redaction("\n".join(lines) + "\n"))
        if passed:
            aws.s3_put_text(smoke_marker_key(image_id),
                            _marker_text(image_id, now(), boot_to_ready_seconds))
        return passed
    finally:
        shutdown()


def main(argv: list[str] | None = None) -> int:
    """The smoke machine's command: `python -m cloud.smoke_runner --job <id>`.

    The smoke machine's User_Data runs it as its last line; it isn't a laptop command. It reads how
    long the machine took to boot, loads the Code_Archive's config.toml, builds the machine's `Aws`
    with `job_machine_aws` and runs `run_smoke_test` on the image the User_Data names in IMAGE_ID,
    which powers the machine off when it ends. It gives 0 when every check passed and 1 otherwise.
    A refusal or an error prints its Readable_Error, Account_Redacted, on stderr and gives 1,
    through `run_cloud_cli`; one that comes before `run_smoke_test` starts leaves the power-off to
    the User_Data's trap.
    """
    def body() -> int:
        args = _parser().parse_args(argv)
        job_id = check_job_id(args.job)
        boot_to_ready = _seconds_since_boot()  # first, so loading the config isn't counted
        config = load_config()
        aws = job_machine_aws(config)
        # _poweroff is looked up here, at call time, so a test can put a recorder in its place.
        passed = run_smoke_test(job_id, aws=aws, config=config, cloud_config=load_cloud_config(),
                                image_id=_image_id(), boot_to_ready_seconds=boot_to_ready,
                                shutdown=_poweroff)
        return 0 if passed else 1

    return run_cloud_cli(body)


# The checks. Each gets a SmokeContext and gives back its line for the log, or raises.

def _check_torch(context: SmokeContext) -> str:
    """torch sees the GPU."""
    import torch  # the GPU packages exist only on the machine image, so they're imported here

    if not torch.cuda.is_available():
        raise CloudError("torch can't see a GPU on this machine: torch.cuda.is_available() is "
                         "False. Check the NVIDIA driver with nvidia-smi, then rebuild the image.")
    return f"torch {torch.__version__} sees the {torch.cuda.get_device_name(0)}"


def _check_gsplat(context: SmokeContext) -> str:
    """gsplat renders a tiny synthetic scene on the GPU, and the gradients come back finite."""
    import torch
    from gsplat import rasterization

    settings = context.cloud_config.smoke_checks
    count, pixels = settings.scene_gaussians, settings.image_pixels
    cuda = torch.device("cuda")
    # A camera at the origin looking down +z, its image as wide as it is far, and the Gaussians on
    # random rays through the image, one unit in front of it and a pixel across there, so every one
    # is in view.
    focal = float(pixels)
    viewmats = torch.eye(4, device=cuda)[None]
    intrinsics = torch.tensor([[focal, 0.0, pixels / 2], [0.0, focal, pixels / 2],
                               [0.0, 0.0, 1.0]], device=cuda)[None]
    rays = (torch.rand(count, 2, device=cuda) * pixels - pixels / 2) / focal
    means = torch.cat([rays, torch.ones(count, 1, device=cuda)], dim=1).requires_grad_()
    quats = torch.zeros(count, 4, device=cuda)
    quats[:, 0] = 1.0  # no rotation: w comes first
    scales = torch.full((count, 3), 1.0 / focal, device=cuda)
    opacities = torch.rand(count, device=cuda)
    colors = torch.rand(count, 3, device=cuda)
    render, alphas, _meta = rasterization(means, quats, scales, opacities, colors, viewmats,
                                          intrinsics, pixels, pixels)
    if tuple(render.shape) != (1, pixels, pixels, 3):
        raise CloudError(f"gsplat rendered an image of shape {tuple(render.shape)}, not "
                         f"(1, {pixels}, {pixels}, 3). Check the image build's gsplat step.")
    if not bool(torch.isfinite(render).all()) or not bool((alphas > 0).any()):
        raise CloudError("gsplat's render of the synthetic scene came back empty or not finite. "
                         "Check the image build's gsplat step, then rebuild the image.")
    render.sum().backward()
    if means.grad is None or not bool(torch.isfinite(means.grad).all()):
        raise CloudError("gsplat's gradients for the synthetic scene came back missing or not "
                         "finite. Check the image build's gsplat step, then rebuild the image.")
    return f"rendered {count} Gaussians at {pixels}x{pixels} pixels, forwards and backwards"


def _check_fused_ssim(context: SmokeContext) -> str:
    """fused-ssim scores an image against itself above a different one, with a finite gradient."""
    import torch
    from fused_ssim import fused_ssim

    pixels = context.cloud_config.smoke_checks.image_pixels
    cuda = torch.device("cuda")
    image = torch.rand(1, 3, pixels, pixels, device=cuda, requires_grad=True)
    other = torch.rand(1, 3, pixels, pixels, device=cuda)
    same = float(fused_ssim(image, image.detach()))
    different_ssim = fused_ssim(image, other)
    different = float(different_ssim)
    if not same > different:  # also False when either is NaN
        raise CloudError(f"fused-ssim scored an image against itself at {same:.3f} and against "
                         f"another at {different:.3f}, but the first must be higher. Check the "
                         "image build's fused-ssim step, then rebuild the image.")
    different_ssim.backward()
    if image.grad is None or not bool(torch.isfinite(image.grad).all()):
        raise CloudError("fused-ssim's gradient came back missing or not finite. Check the image "
                         "build's fused-ssim step, then rebuild the image.")
    return (f"SSIM {same:.3f} for an image against itself and {different:.3f} against another, "
            "with a finite gradient")


def _check_ffmpeg(context: SmokeContext) -> str:
    """ffmpeg has the zscale and tonemap filters, through the frames step's own check."""
    if not has_filters(_TONE_MAP_FILTERS, _timeout(context)):
        raise CloudError("This ffmpeg has no zscale or no tonemap filter, or didn't list its "
                         "filters in time, so the frames step couldn't tone map HDR video. The "
                         "image build compiles ffmpeg with zimg for zscale; check that step, then "
                         "rebuild the image.")
    return "ffmpeg has the zscale and tonemap filters the frames step's tone map needs"


def _check_colmap(context: SmokeContext) -> str:
    """COLMAP runs and prints its version."""
    output = _run_command(["colmap", "version"], _timeout(context))
    line = _line_starting(output, "COLMAP")
    if not line:
        raise CloudError(f"`colmap version` {_printed(output)}, not COLMAP's version. Check the "
                         "image build's COLMAP step, then rebuild the image.")
    return line


def _check_node(context: SmokeContext) -> str:
    """Node runs and prints its version."""
    output = _run_command(["node", "--version"], _timeout(context))
    version = _line_starting(output, "v")
    if not version:
        raise CloudError(f"`node --version` {_printed(output)}, not a version like v22.1.0. "
                         "Check the image build's Node step, then rebuild the image.")
    return f"node {version}"


def _check_splat_transform(context: SmokeContext) -> str:
    """splat-transform, which npm ci installed from package-lock.json, runs and prints its version.

    It prints the version on stderr, where all its status lines go, and keeps stdout for output.
    """
    output = _run_command([SPLAT_TRANSFORM, "--version"], _timeout(context))
    line = _line_starting(output, "splat-transform")
    if not line:
        raise CloudError(f"`splat-transform --version` {_printed(output)}, not its version, like "
                         "splat-transform v3.8.0. Check the image build's npm ci step, then "
                         "rebuild the image.")
    return line


def _check_moge_weights(context: SmokeContext) -> str:
    """The MoGe-2 weights on the image match their pinned sha256."""
    try:
        with open(MOGE_WEIGHTS, "rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
    except OSError as error:
        raise CloudError(f"The MoGe-2 weights can't be read at {MOGE_WEIGHTS} "
                         f"({error.strerror or error}). The image build downloads them; check that "
                         "step, then rebuild the image.") from error
    if digest != MOGE_SHA256:
        raise CloudError(f"The MoGe-2 weights at {MOGE_WEIGHTS} have the sha256 {digest}, not the "
                         f"pinned {MOGE_SHA256}. Rebuild the image.")
    return "the MoGe-2 weights match their pinned sha256"


def _check_training(context: SmokeContext) -> str:
    """The Trainer trains the Smoke_Scene on the GPU, and the PLY it gives passes the PLY_Check.

    The sizes come from `[cloud.smoke]`, and `run_smoke_training` does the rest. `load_gpu`, looked
    up in this module when the check runs, imports the GPU_Module and torch only then, so this
    module still imports on a laptop (Req 12.10) and a test can patch `cloud.smoke_runner.load_gpu`
    with a GPU_Stand_In. Like the gsplat and fused-ssim checks, it has no timeout of its own: the
    log records its seconds (Req 12.11).
    """
    checks = context.cloud_config.smoke_checks
    sizes = SmokeSizes(images=checks.train_images, points=checks.train_points,
                       pixels=checks.image_pixels, splats=checks.train_splats,
                       steps=checks.train_steps, refine_start_step=checks.train_refine_start_step,
                       refine_every=checks.train_refine_every,
                       sh_degree_interval=checks.train_sh_degree_interval)
    return run_smoke_training(context.config, sizes, context.scratch, load_gpu())


def _check_cloud_runner(context: SmokeContext) -> str:
    """The Cloud_Runner runs a tiny Fake_Step Job end to end (Req 11.2).

    The Job is made from a test video ffmpeg writes, under this smoke run's own Job_Id, so its
    `input/` sits beside the Code_Archive the Smoke_Command uploaded, as a real Job's does. Its input
    goes up to the bucket and the local Job_Folder is removed, so `run_cloud_job` downloads it again,
    runs the Fake_Step and uploads the results, keeping the machine on for the rest of the checks.
    The check passes when the Job is done and its summary, log and report are in the bucket.
    """
    settings = context.cloud_config.smoke_checks
    video = context.scratch / "video.mp4"
    _run_command(_test_video_command(video, context), settings.check_timeout_seconds)
    job_id = make_job(video, config=context.config, new_id=lambda: context.job_id)
    folder = job_folder(job_id, context.config.jobs().jobs_folder)
    for path in sorted((folder / "input").iterdir()):
        context.aws.s3_upload(path, f"jobs/{job_id}/input/{path.name}")
    remove_tree(folder)
    summary = run_cloud_job(job_id, aws=context.aws, config=context.config,
                            cloud_config=context.cloud_config, steps=(_FAKE_STEP,),
                            shutdown=_keep_running)
    if summary["status"] != "done":
        raise CloudError(f"The fake-step job {job_id} ended {summary['status']}: "
                         f"{summary['readable_error']}")
    uploaded = set(context.aws.s3_list(f"jobs/{job_id}/out/"))
    missing = [key for key in _fake_job_outputs(job_id) if key not in uploaded]
    if missing:
        raise CloudError(f"The fake-step job {job_id} ran, but the Cloud_Runner didn't upload "
                         f"{', '.join(missing)}.")
    return f"ran the fake-step job {job_id} from the bucket and uploaded its results"


# The checks, in the order they run. The Cloud_Runner check comes last, so the copy of cloud-init's
# log it uploads with the fake-step job already holds every other check's result. Each later spec
# that adds a GPU code path adds a check for it here, starting with training (Req 11.5).
SMOKE_CHECKS: tuple[SmokeCheck, ...] = (
    SmokeCheck("torch-gpu", _check_torch),
    SmokeCheck("gsplat", _check_gsplat),
    SmokeCheck("fused-ssim", _check_fused_ssim),
    SmokeCheck("ffmpeg-zscale", _check_ffmpeg),
    SmokeCheck("colmap", _check_colmap),
    SmokeCheck("node", _check_node),
    SmokeCheck("splat-transform", _check_splat_transform),
    SmokeCheck("moge-weights", _check_moge_weights),
    SmokeCheck("training", _check_training),
    SmokeCheck("cloud-runner", _check_cloud_runner),
)


# Running a check and writing up the results

def _run_check(check: SmokeCheck, context: Callable[[Path], SmokeContext]) -> CheckResult:
    """Run one check in a scratch folder of its own. Any exception fails this check only."""
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="voxelis-smoke-", ignore_cleanup_errors=True) as scratch:
        try:
            detail, passed = check.run(context(Path(scratch))), True
        except Exception as error:  # a failed check is reported, and the other checks still run
            detail, passed = _error_text(error), False
    return CheckResult(check.name, passed, detail, time.perf_counter() - started)


def _error_text(error: Exception) -> str:
    """A failed check's line: our own Readable_Error as it is, anything else with its type."""
    if isinstance(error, (CloudError, PipelineError)):
        return str(error)
    return f"{type(error).__name__}: {error}"


def _result_line(result: CheckResult) -> str:
    """One check's line in the log, like `PASS colmap (0.2 s): COLMAP 4.2.1 ...`."""
    word = "PASS" if result.passed else "FAIL"
    return f"{word} {result.name} ({result.seconds:.1f} s): {result.detail}"


def _verdict_line(results: Sequence[CheckResult], passed: bool, image_id: str) -> str:
    """The log's last line: whether the Smoke_Marker is written, and what to do when it isn't."""
    if passed:
        return f"Every check passed, so the smoke marker goes to {smoke_marker_key(image_id)}."
    failed = sum(1 for result in results if not result.passed)
    return (f"{failed} of {len(results)} checks failed, so no smoke marker was written. Fix the "
            "image build, build a new image, then run the smoke test on it.")


def _marker_text(image_id: str, passed_at: datetime, boot_to_ready_seconds: float) -> str:
    """The Smoke_Marker's JSON: the image id, when it passed (UTC) and its boot-to-ready seconds."""
    marker = {"image_id": image_id,
              "passed_utc": passed_at.astimezone(timezone.utc).isoformat(timespec="seconds"),
              "boot_to_ready_seconds": boot_to_ready_seconds}
    return json.dumps(marker, ensure_ascii=True, allow_nan=False) + "\n"


def _say(line: str) -> None:
    """Print one line of the log, Account_Redacted, at once, for the console and cloud-init's log."""
    print(account_redaction(line), flush=True)


# Small helpers for the checks

def _timeout(context: SmokeContext) -> float:
    """The longest a command check may take, `check_timeout_seconds` in [cloud.smoke]."""
    return context.cloud_config.smoke_checks.check_timeout_seconds


def _run_command(command: Sequence[str], timeout_seconds: float) -> str:
    """Run one command and give back what it printed, stdout first and then stderr, or raise a
    readable CloudError.

    Both streams come back because a tool may print its version on either: splat-transform puts
    it on stderr. A command that isn't installed, runs past `timeout_seconds` or exits with a code
    other than 0 fails, naming the command and the last line it printed.
    """
    shown = " ".join(command)
    try:
        result = subprocess.run(list(command), capture_output=True, stdin=subprocess.DEVNULL,
                                timeout=timeout_seconds)
    except FileNotFoundError as error:
        raise CloudError(f"{command[0]} isn't on this machine, so `{shown}` can't run. The image "
                         "build installs it; check that step, then rebuild the image.") from error
    except subprocess.TimeoutExpired as error:
        raise CloudError(f"`{shown}` didn't finish within {timeout_seconds:g} s, the "
                         "check_timeout_seconds in [cloud.smoke].") from error
    stdout = result.stdout.decode("utf-8", errors="replace")
    stderr = result.stderr.decode("utf-8", errors="replace")
    if result.returncode != 0:
        raise CloudError(f"`{shown}` exited with code {result.returncode}: "
                         f"{_last_line(stderr) or _last_line(stdout) or 'it printed nothing'}")
    return f"{stdout}\n{stderr}"


def _first_line(text: str) -> str:
    """The first line of `text` that isn't blank, stripped, or "" when there is none."""
    return next((line.strip() for line in text.splitlines() if line.strip()), "")


def _line_starting(text: str, prefix: str) -> str:
    """The first line of `text` that starts with `prefix`, stripped, or "" when there is none.

    A tool's version line is found this way, so a warning printed before it doesn't hide it.
    """
    return next((line.strip() for line in text.splitlines() if line.strip().startswith(prefix)), "")


def _printed(output: str) -> str:
    """What a command printed, for a failed check's message: its first line, or that it was nothing."""
    line = _first_line(output)
    return f"printed {line!r}" if line else "printed nothing"


def _last_line(text: str) -> str:
    """The last line of `text` that isn't blank, stripped, or "" when there is none."""
    return next((line.strip() for line in reversed(text.splitlines()) if line.strip()), "")


def _test_video_command(video: Path, context: SmokeContext) -> list[str]:
    """The ffmpeg command that writes the fake-step job's test video: ffmpeg's own test pattern,
    `image_pixels` square for `video_seconds`, in its built-in MPEG-4 encoder, which needs no
    library of its own."""
    settings = context.cloud_config.smoke_checks
    pattern = (f"testsrc=duration={settings.video_seconds:g}:"
               f"size={settings.image_pixels}x{settings.image_pixels}")
    return ["ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error", "-f", "lavfi",
            "-i", pattern, "-c:v", "mpeg4", "-y", str(video)]


def _fake_job_outputs(job_id: str) -> tuple[str, ...]:
    """The keys the fake-step job must leave in the bucket: its summary, its log and its report."""
    out = f"jobs/{job_id}/out"
    return (f"{out}/job.json", f"{out}/job.log", f"{out}/reports/{_FAKE_STEP.name}.json")


def _keep_running() -> None:
    """The fake-step job's `shutdown`: the machine stays on for the rest of the checks."""


def _seconds_since_boot() -> float:
    """The machine's seconds since boot, the first number in /proc/uptime (Req 11.5)."""
    try:
        return float(PROC_UPTIME.read_text(encoding="ascii").split()[0])
    except (OSError, ValueError, IndexError) as error:
        raise CloudError(f"{PROC_UPTIME} can't be read ({error}), so the smoke test can't say how "
                         "long the machine took to be ready. This command runs only on the smoke "
                         "machine.") from error


def _image_id() -> str:
    """The id of the Machine_Image this machine runs, from the IMAGE_ID the User_Data exports."""
    image_id = os.environ.get(IMAGE_ID_ENV, "")
    if not image_id:
        raise CloudError(f"The {IMAGE_ID_ENV} environment variable isn't set, so the smoke test "
                         "doesn't know which machine image it's checking. The machine's user data "
                         "sets it; this command runs only there, started by cloud.smoke.")
    return image_id


def _parser() -> argparse.ArgumentParser:
    """The command line: --job, which is required. The smoke machine's User_Data passes it."""
    parser = argparse.ArgumentParser(
        prog="python -m cloud.smoke_runner", allow_abbrev=False,
        description="Run the machine image's checks on this smoke machine, write the smoke marker "
                    "when they all pass, and power the machine off. The machine's user data runs "
                    "it; it isn't a laptop command.")
    parser.add_argument("--job", required=True,
                        help="the job id cloud.smoke launched this machine for")
    return parser


if __name__ == "__main__":
    sys.exit(main())
