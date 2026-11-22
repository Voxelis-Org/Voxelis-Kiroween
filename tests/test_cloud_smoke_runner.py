"""Unit tests for the smoke machine's side of the Smoke_Test, cloud/smoke_runner.py.

The smoke machine's User_Data runs `python -m cloud.smoke_runner --job <id>`. These tests drive it on
the CPU with no network, no GPU and no AWS account, against the `FakeAws` seam (Req 16.1):

- `run_smoke_test` runs every check, also after one fails, prints the results and puts them,
  Account_Redacted, at `jobs/_smoke/<image id>.log`, writes the Smoke_Marker only when every check
  passed, and powers the machine off once, even on an error (Req 11.2 to 11.5).
- `main` reads how long the machine took to boot from /proc/uptime and the image id from IMAGE_ID,
  and gives 0 on a pass and 1 otherwise.
- SMOKE_CHECKS holds the image's checks, in order.
- The checks that need no GPU run here, with stand-ins for the tools: COLMAP, Node, splat-transform,
  the ffmpeg filter check and the MoGe-2 sha256. The Cloud_Runner check runs for real, on a tiny
  Fake_Step Job, when ffmpeg is installed, and is skipped when it isn't.
- The GPU checks (torch, gsplat and fused-ssim) need the machine image's GPU packages, so their
  bodies run only on the real machine. On a laptop without those packages, each fails on its own and
  the smoke test goes on.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import sys
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

import cloud.runner
import cloud.smoke_runner
from splatpipe.config import load_config
from splatpipe.run import StepEntry

from cloud.errors import CloudError
from cloud.image import SPLAT_TRANSFORM
from cloud.runner import CAP_DEADLINE_ENV
from cloud.smoke import smoke_log_key, smoke_marker_key
from cloud.smoke_runner import SMOKE_CHECKS, SmokeCheck, SmokeContext, run_smoke_test

from fake_aws import FakeAws, cloud_config

IMAGE_ID = "ami-0123456789abcdef0"
JOB_ID = "20261119-142530-7f3a9c"
PASSED_AT = datetime(2026, 11, 20, 10, 15, 0, tzinfo=timezone.utc)
# An account-shaped 12-digit run, which the log must never carry.
ACCOUNT = "123456789012"
# The check_timeout_seconds the fake cloud config gives, from [cloud.smoke].
TIMEOUT = 60.0
# The image's checks, by name.
CHECKS = {check.name: check for check in SMOKE_CHECKS}


def passing(detail: str) -> Callable[[SmokeContext], str]:
    """A check that passes with `detail` as its line."""
    def run(context: SmokeContext) -> str:
        return detail

    return run


def failing(message: str) -> Callable[[SmokeContext], str]:
    """A check that fails with `message` as its Readable_Error."""
    def run(context: SmokeContext) -> str:
        raise CloudError(message)

    return run


def smoke(aws: FakeAws, checks: list[SmokeCheck], shutdowns: list[int]) -> bool:
    """Run the smoke test with `checks` on the Fake_AWS, recording each power-off in `shutdowns`."""
    return run_smoke_test(JOB_ID, aws=aws, config=load_config(), cloud_config=cloud_config(),
                          image_id=IMAGE_ID, boot_to_ready_seconds=142.0, checks=checks,
                          now=lambda: PASSED_AT, shutdown=lambda: shutdowns.append(1))


def log_of(aws: FakeAws) -> str:
    """The results the smoke test put in the bucket."""
    return aws.objects[smoke_log_key(IMAGE_ID)].decode("utf-8")


def context_in(folder: Path, aws: FakeAws | None = None) -> SmokeContext:
    """A SmokeContext for calling one check by hand, with `folder` as its scratch folder."""
    return SmokeContext(JOB_ID, aws or FakeAws(), load_config(), cloud_config(), folder)


# run_smoke_test (Req 11.2 to 11.5)

def test_every_check_passing_writes_the_log_then_the_marker_and_powers_off(
        capsys: pytest.CaptureFixture[str]) -> None:
    aws = FakeAws()
    shutdowns: list[int] = []
    scratches: list[Path] = []

    def check(context: SmokeContext) -> str:
        assert (context.job_id, context.aws) == (JOB_ID, aws)
        assert context.scratch.is_dir() and not any(context.scratch.iterdir())
        scratches.append(context.scratch)
        return "fine"

    assert smoke(aws, [SmokeCheck("one", check), SmokeCheck("two", check)], shutdowns) is True

    assert shutdowns == [1]
    assert json.loads(aws.objects[smoke_marker_key(IMAGE_ID)]) == {
        "image_id": IMAGE_ID, "passed_utc": "2026-11-20T10:15:00+00:00",
        "boot_to_ready_seconds": 142.0}
    log = log_of(aws)
    assert "PASS one" in log and "PASS two" in log
    assert "142.0 s after it booted" in log  # how long the machine took to be ready (Req 11.5)
    # The log went up before the marker, so the marker is the last thing the machine writes.
    puts = [key for op, key in aws.calls if op == "s3_put_text"]
    assert puts == [smoke_log_key(IMAGE_ID), smoke_marker_key(IMAGE_ID)]
    # Each check had a scratch folder of its own, removed after it.
    assert len(set(scratches)) == 2 and not any(path.exists() for path in scratches)
    # The same lines went to the console as they came, for cloud-init's log.
    assert capsys.readouterr().out == log


def test_a_failing_check_fails_the_smoke_test_and_the_others_still_run() -> None:
    aws = FakeAws()
    shutdowns: list[int] = []
    ran: list[str] = []

    def broken(context: SmokeContext) -> str:
        ran.append("broken")
        raise RuntimeError("the GPU fell off")

    def later(context: SmokeContext) -> str:
        ran.append("later")
        return "fine"

    assert smoke(aws, [SmokeCheck("broken", broken), SmokeCheck("later", later)], shutdowns) is False

    assert ran == ["broken", "later"]
    assert smoke_marker_key(IMAGE_ID) not in aws.objects  # no marker without a pass (Req 11.3)
    log = log_of(aws)  # the results still went up, so the failure can be read
    assert "FAIL broken" in log and "RuntimeError: the GPU fell off" in log
    assert "PASS later" in log
    assert "1 of 2 checks failed, so no smoke marker was written" in log
    assert shutdowns == [1]


def test_the_machine_powers_off_even_when_the_bucket_refuses_the_results() -> None:
    class RefusingBucket(FakeAws):
        def s3_put_text(self, key: str, text: str) -> None:
            raise CloudError("The bucket refused the upload.")

    shutdowns: list[int] = []
    with pytest.raises(CloudError):
        smoke(RefusingBucket(), [SmokeCheck("one", passing("fine"))], shutdowns)
    assert shutdowns == [1]  # in the finally (Req 11.4)


def test_no_checks_is_no_pass() -> None:
    aws = FakeAws()
    assert smoke(aws, [], []) is False
    assert smoke_marker_key(IMAGE_ID) not in aws.objects


def test_the_results_are_account_redacted_in_the_bucket_and_on_the_console(
        capsys: pytest.CaptureFixture[str]) -> None:
    aws = FakeAws()
    smoke(aws, [SmokeCheck("leaky", passing(f"saw account {ACCOUNT}")),
                SmokeCheck("louder", failing(f"denied for {ACCOUNT}"))], [])
    for text in (log_of(aws), capsys.readouterr().out):
        assert ACCOUNT not in text
        assert text.count("<account>") == 2


# main, the command the smoke machine's User_Data runs

def patch_machine(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, aws: FakeAws,
                  checks: list[SmokeCheck]) -> list[int]:
    """Point cloud.smoke_runner.main at a fake /proc/uptime, the Fake_AWS, the fake cloud config,
    `checks` and an IMAGE_ID, and record each power-off in the list this returns."""
    uptime = tmp_path / "uptime"
    uptime.write_text("142.37 1012.50\n", encoding="ascii")
    monkeypatch.setattr(cloud.smoke_runner, "PROC_UPTIME", uptime)
    monkeypatch.setattr(cloud.smoke_runner, "job_machine_aws", lambda config: aws)
    monkeypatch.setattr(cloud.smoke_runner, "load_cloud_config", cloud_config)
    monkeypatch.setattr(cloud.smoke_runner, "SMOKE_CHECKS", tuple(checks))
    monkeypatch.setenv("IMAGE_ID", IMAGE_ID)
    poweroffs: list[int] = []
    monkeypatch.setattr(cloud.smoke_runner, "_poweroff", lambda: poweroffs.append(1))
    return poweroffs


def test_main_writes_the_marker_with_the_boot_time_and_powers_off(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    aws = FakeAws()
    poweroffs = patch_machine(monkeypatch, tmp_path, aws, [SmokeCheck("one", passing("fine"))])

    assert cloud.smoke_runner.main(["--job", JOB_ID]) == 0

    assert poweroffs == [1]
    marker = json.loads(aws.objects[smoke_marker_key(IMAGE_ID)])
    assert marker["image_id"] == IMAGE_ID
    assert marker["boot_to_ready_seconds"] == 142.37  # the first number in /proc/uptime


def test_main_gives_1_when_a_check_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    aws = FakeAws()
    poweroffs = patch_machine(monkeypatch, tmp_path, aws, [SmokeCheck("one", failing("no"))])

    assert cloud.smoke_runner.main(["--job", JOB_ID]) == 1

    assert poweroffs == [1]
    assert smoke_marker_key(IMAGE_ID) not in aws.objects
    assert "FAIL one" in log_of(aws)


def test_main_refuses_without_the_image_id(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    aws = FakeAws()
    poweroffs = patch_machine(monkeypatch, tmp_path, aws, [SmokeCheck("one", passing("fine"))])
    monkeypatch.delenv("IMAGE_ID")

    assert cloud.smoke_runner.main(["--job", JOB_ID]) == 1

    assert "IMAGE_ID" in capsys.readouterr().err
    assert aws.calls == []
    assert poweroffs == []  # nothing ran, so the User_Data's poweroff trap ends the machine


# The checks

def test_the_smoke_checks_cover_the_image_in_order() -> None:
    # The user's list, then the Cloud_Runner on a fake-step job last. Later specs add their own GPU
    # paths, starting with training (Req 11.5).
    assert [check.name for check in SMOKE_CHECKS] == [
        "torch-gpu", "gsplat", "fused-ssim", "ffmpeg-zscale", "colmap", "node",
        "splat-transform", "moge-weights", "cloud-runner"]


# What each tool prints for its version, as _run_command gives it back: stdout, a line break, then
# stderr. COLMAP and Node print the version on stdout. splat-transform prints nothing on stdout and
# its version on stderr, where all its status lines go, as the smoke machine found.
@pytest.mark.parametrize(("name", "command", "output", "detail"), [
    ("colmap", ["colmap", "version"],
     "COLMAP 4.2.1 (Commit 1a2b3c4 on 2026-11-20 without CUDA)\n\n",
     "COLMAP 4.2.1 (Commit 1a2b3c4 on 2026-11-20 without CUDA)"),
    ("node", ["node", "--version"], "v22.21.1\n\n", "node v22.21.1"),
    ("splat-transform", [SPLAT_TRANSFORM, "--version"], "\nsplat-transform v3.8.0 (1a2b3c4)\n",
     "splat-transform v3.8.0 (1a2b3c4)"),
])
def test_each_tool_check_runs_its_tool_and_reports_its_version(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str, command: list[str],
        output: str, detail: str) -> None:
    asked: list[tuple[list[str], float]] = []

    def run_command(given: list[str], timeout_seconds: float) -> str:
        asked.append((list(given), timeout_seconds))
        return output

    monkeypatch.setattr(cloud.smoke_runner, "_run_command", run_command)
    assert CHECKS[name].run(context_in(tmp_path)) == detail
    assert asked == [(command, TIMEOUT)]


def test_a_tool_check_finds_its_version_line_after_a_warning(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Node, or a library it loads, may print a warning on stderr before the tool's own version line.
    output = "\n(node:4242) Warning: an example warning\nsplat-transform v3.8.0 (1a2b3c4)\n"
    monkeypatch.setattr(cloud.smoke_runner, "_run_command", lambda given, timeout: output)
    assert CHECKS["splat-transform"].run(context_in(tmp_path)) == "splat-transform v3.8.0 (1a2b3c4)"


@pytest.mark.parametrize(("name", "output"), [
    ("colmap", "Usage: something else\n\n"),
    ("node", "22\n\n"),
    ("splat-transform", "\n\n"),
    ("splat-transform", "\n(node:4242) Warning: an example warning\n"),
])
def test_a_tool_check_fails_when_the_tool_prints_no_version(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str, output: str) -> None:
    monkeypatch.setattr(cloud.smoke_runner, "_run_command", lambda given, timeout: output)
    with pytest.raises(CloudError):
        CHECKS[name].run(context_in(tmp_path))


@pytest.mark.parametrize("has", [True, False])
def test_the_ffmpeg_check_asks_for_the_frames_steps_tone_map_filters(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, has: bool) -> None:
    asked: list[tuple[tuple[str, ...], float]] = []

    def has_filters(names: Any, timeout_seconds: float) -> bool:
        asked.append((tuple(names), timeout_seconds))
        return has

    monkeypatch.setattr(cloud.smoke_runner, "has_filters", has_filters)
    if has:
        assert "zscale" in CHECKS["ffmpeg-zscale"].run(context_in(tmp_path))
    else:
        with pytest.raises(CloudError):
            CHECKS["ffmpeg-zscale"].run(context_in(tmp_path))
    assert asked == [(("zscale", "tonemap"), TIMEOUT)]


def test_the_moge_check_passes_only_on_the_pinned_sha256(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    weights = tmp_path / "model.pt"
    weights.write_bytes(b"a few bytes standing in for the weights")
    digest = hashlib.sha256(weights.read_bytes()).hexdigest()
    monkeypatch.setattr(cloud.smoke_runner, "MOGE_WEIGHTS", str(weights))
    monkeypatch.setattr(cloud.smoke_runner, "MOGE_SHA256", digest)
    check = CHECKS["moge-weights"]
    assert "match" in check.run(context_in(tmp_path))

    monkeypatch.setattr(cloud.smoke_runner, "MOGE_SHA256", "0" * 64)
    with pytest.raises(CloudError) as caught:
        check.run(context_in(tmp_path))
    assert digest in str(caught.value)  # names the sha256 it found

    monkeypatch.setattr(cloud.smoke_runner, "MOGE_WEIGHTS", str(tmp_path / "missing.pt"))
    with pytest.raises(CloudError):
        check.run(context_in(tmp_path))


@pytest.mark.skipif(importlib.util.find_spec("torch") is not None,
                    reason="torch is installed here, so this isn't a machine without the GPU packages")
def test_without_the_gpu_packages_each_gpu_check_fails_on_its_own() -> None:
    aws = FakeAws()
    gpu = [CHECKS[name] for name in ("torch-gpu", "gsplat", "fused-ssim")]
    assert smoke(aws, [*gpu, SmokeCheck("after", passing("fine"))], []) is False
    log = log_of(aws)
    for check in gpu:
        assert f"FAIL {check.name}" in log
    assert "ModuleNotFoundError" in log
    assert "PASS after" in log


# _run_command, which the tool checks use, on real tiny commands

def test_run_command_gives_back_what_the_command_printed() -> None:
    output = cloud.smoke_runner._run_command([sys.executable, "-c", "print('hello')"], TIMEOUT)
    assert output.strip() == "hello"


def test_run_command_gives_back_stdout_then_stderr() -> None:
    # A tool may print its version on stderr, as splat-transform does, so both streams come back,
    # stdout first.
    script = "import sys; sys.stderr.write('on stderr\\n'); sys.stderr.flush(); print('on stdout')"
    output = cloud.smoke_runner._run_command([sys.executable, "-c", script], TIMEOUT)
    assert [line for line in output.splitlines() if line] == ["on stdout", "on stderr"]


def test_run_command_finds_a_version_printed_only_on_stderr() -> None:
    script = "import sys; sys.stderr.write('splat-transform v3.8.0 (1a2b3c4)\\n')"
    output = cloud.smoke_runner._run_command([sys.executable, "-c", script], TIMEOUT)
    assert cloud.smoke_runner._line_starting(output, "splat-transform") == (
        "splat-transform v3.8.0 (1a2b3c4)")


def test_run_command_names_the_exit_code_and_the_last_line_printed() -> None:
    script = "import sys; print('working'); sys.stderr.write('it broke\\n'); sys.exit(3)"
    with pytest.raises(CloudError) as caught:
        cloud.smoke_runner._run_command([sys.executable, "-c", script], TIMEOUT)
    assert "code 3" in str(caught.value) and "it broke" in str(caught.value)


def test_run_command_refuses_a_tool_that_isnt_there() -> None:
    with pytest.raises(CloudError) as caught:
        cloud.smoke_runner._run_command(["voxelis-no-such-tool"], TIMEOUT)
    assert "isn't on this machine" in str(caught.value)


def test_run_command_stops_a_command_that_runs_past_the_timeout() -> None:
    with pytest.raises(CloudError) as caught:
        cloud.smoke_runner._run_command([sys.executable, "-c", "import time; time.sleep(30)"], 0.5)
    assert "check_timeout_seconds" in str(caught.value)


# The Cloud_Runner check, for real, on a tiny Fake_Step Job (Req 11.2)

def machine_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Set up what the User_Data gives the Cloud_Runner: a Code_Archive root whose lock files match
    the image's recorded hashes, the instance type, and no cloud-init log. The Cap_Deadline variable
    is set to "" so monkeypatch removes it afterwards, though run_cloud_job sets it."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "uv.lock").write_bytes(b"version = 1\n")
    (repo / "package-lock.json").write_bytes(b'{"lockfileVersion": 3}\n')
    monkeypatch.setattr(cloud.runner, "REPO_ROOT", repo)
    monkeypatch.setattr(cloud.runner, "CLOUD_INIT_LOG", tmp_path / "no-cloud-init-output.log")
    for variable, name in (("UV_LOCK_HASH", "uv.lock"), ("PACKAGE_LOCK_HASH", "package-lock.json")):
        monkeypatch.setenv(variable, hashlib.sha256((repo / name).read_bytes()).hexdigest())
    monkeypatch.setenv("INSTANCE_TYPE", "g5.xlarge")
    monkeypatch.setenv(CAP_DEADLINE_ENV, "")


def allow_the_test_video(temp_config: Path) -> None:
    """Lift the test config's 1 KB video limit, since the test video is a few tens of KB."""
    text = temp_config.read_text(encoding="utf-8")
    temp_config.write_text(text.replace("max_video_bytes = 1024", "max_video_bytes = 4294967296"),
                           encoding="utf-8")


needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
                                  reason="ffmpeg is not installed")


@needs_ffmpeg
def test_the_cloud_runner_check_runs_a_fake_step_job_through_the_bucket(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, temp_config: Path) -> None:
    machine_repo(tmp_path, monkeypatch)
    allow_the_test_video(temp_config)
    # The Smoke_Command already put the Code_Archive under this smoke run's Job_Id.
    aws = FakeAws(objects={f"jobs/{JOB_ID}/input/code.tar": b"the code archive"})
    passed: dict[str, Any] = {}
    real_run_cloud_job = cloud.smoke_runner.run_cloud_job

    def spy(*args: Any, **kwargs: Any) -> dict[str, Any]:
        passed.update(kwargs)
        return real_run_cloud_job(*args, **kwargs)

    monkeypatch.setattr(cloud.smoke_runner, "run_cloud_job", spy)
    scratch = tmp_path / "scratch"
    scratch.mkdir()

    detail = CHECKS["cloud-runner"].run(context_in(scratch, aws))

    assert JOB_ID in detail
    # The Job's input went up, and the Cloud_Runner downloaded it again into a fresh Job_Folder.
    uploads = [key for op, key in aws.calls if op == "s3_upload"]
    downloads = [key for op, key in aws.calls if op == "s3_download"]
    for name in ("job.json", "video.mp4"):
        assert f"jobs/{JOB_ID}/input/{name}" in uploads
        assert f"jobs/{JOB_ID}/input/{name}" in downloads
    # The fake step ran, and its results are in the bucket.
    assert json.loads(aws.objects[f"jobs/{JOB_ID}/out/job.json"])["status"] == "done"
    assert f"jobs/{JOB_ID}/out/reports/fake.json" in aws.objects
    # It ran the Fake_Step alone, and kept the machine on for the rest of the checks.
    assert passed["steps"] == (StepEntry("fake", "splatpipe.fake", True),)
    assert passed["shutdown"] is not cloud.runner._poweroff
    assert passed["shutdown"]() is None


@needs_ffmpeg
def test_the_cloud_runner_check_fails_when_the_fake_step_job_fails(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, temp_config: Path,
        set_fail: Callable[..., None]) -> None:
    machine_repo(tmp_path, monkeypatch)
    set_fail(fake="expected")
    allow_the_test_video(temp_config)  # after set_fail, which writes the whole test config again
    scratch = tmp_path / "scratch"
    scratch.mkdir()

    with pytest.raises(CloudError) as caught:
        CHECKS["cloud-runner"].run(context_in(scratch))
    assert "ended failed" in str(caught.value)
