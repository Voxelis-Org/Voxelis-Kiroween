"""The Cloud_Runner: the machine-side wrapper that runs one Job through run_job and ends the machine.

The Cloud_Runner only wraps pipeline-core's `run_job`; it never changes what it does (Req 10.1,
10.4). `run_cloud_job` rebuilds the Job_Folder from the Job_Bucket, runs `run_job` with no
Fresh_Option so pipeline-core stays the single source of truth for the Job_Summary, the Job_Log and
each Step_Report, uploads the results as they appear and once more at the end, writes the Spend_Lines,
redacts the text logs it uploads, and powers the machine off in a `finally` so it always ends, even
on an error (Req 10.6). Because `run_job`'s own resume picks up from the first unfinished Step, the
Cloud_Runner inherits that resume exactly, without pre-filtering a single Step (Req 10.4).

`rebuild_job_folder` fetches the Job's `input/`, and any `out/` and `work/` an earlier run left, so
`run_job` sees what a resumed laptop run would. The `Uploader` tails `out/job.log` and uploads a
Step's report only once that Step's outcome line has appeared, so it never uploads a half-written,
non-atomic `write_json` report (Req 10.3); `sync_all` is the final authoritative pass, so the
uploaded set ends up equal to the Job's `out/` and `work/` files (Req 10.5). `spend.jsonl` is owned
by `append_spend_line`, which reads and writes it straight in the bucket, so the rebuild and the
Uploader both leave it alone to avoid clobbering the bucket's copy.

Machine facts come from the environment the User_Data exports: the Machine_Image's recorded lock
hashes (`UV_LOCK_HASH`, `PACKAGE_LOCK_HASH`), which a drift check compares with the extracted
Code_Archive's own lock files (Req 10.9), and the instance type (`INSTANCE_TYPE`) the Spend_Lines
price. The whole runner is testable with injected `steps`, `now` and `shutdown` and the Fake_AWS, so
the Test_Suite exercises it on a CPU with no account, no network and no real `run_job` change
(Req 10.11).

A cloud module may import `splatpipe` to call its public functions; `splatpipe` never imports
`cloud`. This module imports no boto3 (it talks to AWS only through the `Aws` seam) and no GPU
package, so it loads on a laptop.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import subprocess
import threading
import time
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import splatpipe.run
from splatpipe.jobs import job_folder, open_job
from splatpipe.jsonio import read_json

from cloud.errors import CloudError
from cloud.redact import account_redaction
from cloud.spend_file import LABEL_MACHINE, append_spend_line, spend_line

if TYPE_CHECKING:
    from collections.abc import Iterator

    from splatpipe.config import Config
    from splatpipe.run import StepEntry

    from cloud.aws import Aws
    from cloud.config import CloudConfig

CAP_DEADLINE_ENV = "SPLATPIPE_CAP_DEADLINE"   # the Cap_Deadline the Cloud_Runner sets (Req 10.7)

# The extracted Code_Archive's root on the machine: /opt/job, which is cloud/runner.py's own
# grandparent, the same way cloud.start finds the repo root. Tests point this at a temp repo to
# control the lock-file hashes.
REPO_ROOT = Path(__file__).resolve().parents[1]
# cloud-init writes the bootstrap script's output here on the machine. The Cloud_Runner redacts a
# copy into the Job's out/ before upload, so a downloaded copy holds no account id (Req 10.10).
# Tests point it at a temp file.
CLOUD_INIT_LOG = Path("/var/log/cloud-init-output.log")

# The environment variables the User_Data exports for the machine. The lock hashes are the
# Machine_Image's recorded hashes (its tags), surfaced through the User_Data; INSTANCE_TYPE is the
# machine's own type, which the Spend_Lines price (the Job_Role has no EC2 read to ask AWS).
_IMAGE_UV_HASH_ENV = "UV_LOCK_HASH"
_IMAGE_PACKAGE_HASH_ENV = "PACKAGE_LOCK_HASH"
_INSTANCE_TYPE_ENV = "INSTANCE_TYPE"

# The two lock files, in the order their hashes are compared (Req 10.9).
_LOCK_FILES = ("uv.lock", "package-lock.json")
# append_spend_line owns spend.jsonl in the bucket, so the rebuild and the Uploader skip it.
_SPEND_FILE = "spend.jsonl"
# The text logs the Cloud_Runner redacts before upload; the JSON reports are left byte-for-byte.
_JOB_LOG = "job.log"
_RUNNER_LOG = "runner.log"
_CLOUD_INIT_NAME = "cloud-init-output.log"

_LOG = logging.getLogger("cloud.runner")
_LOG_FORMAT = "%(asctime)s %(levelname)s %(message)s"
_LOG_TIME = "%Y-%m-%dT%H:%M:%SZ"  # with time.gmtime, so the runner log's times are in UTC
# A Step's outcome line in out/job.log, after its timestamp and level: "<name>: done" or
# "<name>: failed, ...". Seeing it means run_step has returned and write_json has finished the
# report, so the Uploader may upload it (Req 10.3). The job-status line ("job <id>: done") can't
# match: the token before ": done" is the Job_Id, which starts with a digit.
_OUTCOME = re.compile(r"(?m)^\S+ \S+ ([a-z][a-z0-9_]*): (?:done|failed)")


def _utc_now() -> datetime:
    """The time now, as an aware UTC datetime. The default `now` for run_cloud_job."""
    return datetime.now(timezone.utc)


def _poweroff() -> None:
    """Power the machine off, so terminate-on-shutdown ends it (Req 10.6). The default `shutdown`."""
    subprocess.run(["poweroff"], check=False)


def rebuild_job_folder(aws: Aws, job_id: str, jobs_folder: Path) -> None:
    """Download the Job's files from the Job_Bucket into the local Job_Folder (Req 10.1).

    Fetches `jobs/<id>/input/*`, and any `jobs/<id>/out/*` and `work/<id>/*` an earlier run of the
    same Job left, mapping `jobs/<id>/` to the Job_Folder and `work/<id>/` to its `work/`, so
    `run_job` sees the same reports it would on a resumed laptop run and resumes from the first
    unfinished Step. `spend.jsonl` is left in the bucket, where `append_spend_line` owns it, so a
    resumed run never overwrites it with a stale copy.
    """
    folder = job_folder(job_id, jobs_folder)
    for prefix, local_root in ((f"jobs/{job_id}/input/", folder / "input"),
                               (f"jobs/{job_id}/out/", folder / "out"),
                               (f"work/{job_id}/", folder / "work")):
        for key in aws.s3_list(prefix):
            rel = key[len(prefix):]
            if not rel or rel == _SPEND_FILE:  # a folder marker, or the bucket-owned spend file
                continue
            aws.s3_download(key, local_root / rel)


class Uploader:
    """Uploads the Job's out/ and work/ files to the Job_Bucket as they appear (Req 10.2, 10.3).

    A background thread polls every `poll_seconds` and runs `sync`, one best-effort pass that
    uploads out/ to `jobs/<id>/out/` and work/ to `work/<id>/`. A Step's report
    (`out/reports/<name>.json`) is uploaded only once that Step's outcome line has appeared in
    `out/job.log`, so a half-written, non-atomic `write_json` report is never uploaded (Req 10.3).
    `sync_all` is the final authoritative pass, run after `run_job` returns with no report gate, so
    the uploaded set ends up equal to the Job's out/ and work/ files (Req 10.5). `spend.jsonl` is
    left to `append_spend_line`. The Uploader is best-effort: it swallows its own exceptions, so a
    transient read or upload error never stops the Job.
    """

    def __init__(self, aws: Aws, job_id: str, folder: Path, *, poll_seconds: float) -> None:
        self._aws = aws
        self._job_id = job_id
        self._out = Path(folder) / "out"
        self._work = Path(folder) / "work"
        self._poll_seconds = float(poll_seconds)
        self._stopped = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start the background polling thread (a daemon, so it never keeps the process alive)."""
        self._thread = threading.Thread(target=self._loop, name=f"uploader-{self._job_id}",
                                        daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the background thread and wait for its current pass to finish."""
        self._stopped.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None

    def sync(self) -> None:
        """One best-effort pass, gating each Step's report by its outcome line in out/job.log."""
        try:
            self._upload(self._finished_steps())
        except Exception:  # best-effort: a transient read or upload error never stops the Job
            pass

    def sync_all(self) -> None:
        """The final authoritative pass: upload every out/ and work/ file, reports included."""
        try:
            self._upload(None)
        except Exception:
            pass

    def _loop(self) -> None:
        """Poll until stopped: wait `poll_seconds`, then sync, so a long run uploads as it goes."""
        while not self._stopped.wait(self._poll_seconds):
            self.sync()

    def _upload(self, finished: set[str] | None) -> None:
        """Upload out/ and work/ once. `finished` gates reports; None uploads every report."""
        for path in _files_under(self._out):
            rel = path.relative_to(self._out).as_posix()
            if rel == _SPEND_FILE:  # append_spend_line owns spend.jsonl in the bucket
                continue
            name = _report_step(rel)
            if name is not None and finished is not None and name not in finished:
                continue  # a report whose Step hasn't logged its outcome yet (Req 10.3)
            self._aws.s3_upload(path, f"jobs/{self._job_id}/out/{rel}")
        for path in _files_under(self._work):
            rel = path.relative_to(self._work).as_posix()
            self._aws.s3_upload(path, f"work/{self._job_id}/{rel}")

    def _finished_steps(self) -> set[str]:
        """The Step names whose done/failed outcome line has appeared in out/job.log (Req 10.3)."""
        log = self._out / _JOB_LOG
        if not log.exists():
            return set()
        return set(_OUTCOME.findall(_read_text(log)))


def run_cloud_job(job_id: str, *, aws: Aws, config: Config, cloud_config: CloudConfig,
                  steps: Sequence[StepEntry] | None = None,
                  now: Callable[[], datetime] = _utc_now,
                  shutdown: Callable[[], None] = _poweroff) -> dict[str, Any]:
    """Run one Job on the machine: rebuild the folder, run run_job, upload, tally spend, power off.

    Returns the Job_Summary `run_job` produced. The Cloud_Runner only wraps `run_job`; it never
    changes it (Req 10.1, 10.4). In order: refuse a lock-hash drift against the Machine_Image, still
    powering off (Req 10.9); rebuild the Job_Folder (Req 10.1); set the Cap_Deadline for a Step to
    read (Req 10.7); start the best-effort Uploader (Req 10.2); call `run_job` with no Fresh_Option
    (Req 10.1, 10.4); stop the Uploader and write the per-Step and machine Spend_Lines (Req 12.1,
    12.2); redact the text logs it uploads (Req 10.10); run the final authoritative sync (Req 10.5);
    and, whatever happens, power the machine off in the `finally` (Req 10.6). `steps`, `now` and
    `shutdown` are injected by the tests, with the Fake_AWS, so the whole runner runs on a CPU with
    no account (Req 10.11).
    """
    jobs_folder = config.jobs().jobs_folder
    folder = job_folder(job_id, jobs_folder)
    started = now()
    try:
        _refuse_lock_drift(aws, job_id)                                   # 1
        rebuild_job_folder(aws, job_id, jobs_folder)                      # 2
        job = open_job(job_id, config)
        _set_cap_deadline(started, job.record.preset, cloud_config)       # 3
        uploader = Uploader(aws, job_id, folder, poll_seconds=cloud_config.poll_seconds)
        with _runner_log(folder):
            _LOG.info("cloud runner started for job %s", job_id)
            uploader.start()                                              # 4
            try:
                summary = splatpipe.run.run_job(job_id, steps=steps, config=config)  # 5
            finally:
                uploader.stop()                                           # 6 (stop the Uploader)
            _LOG.info("run_job finished for job %s: %s", job_id, summary["status"])
            _write_spend_lines(aws, job_id, folder, summary, started, now(), cloud_config)  # 6
        _redact_text_logs(folder)                                         # 7
        uploader.sync_all()                                               # 8
        return summary
    finally:
        shutdown()                                                        # 9


# The lock-hash backstop (step 1)

def _refuse_lock_drift(aws: Aws, job_id: str) -> None:
    """Refuse a Job whose lock files differ from the Machine_Image's recorded hashes (Req 10.9).

    The archived uv.lock/package-lock.json (the Code_Archive the machine extracted at REPO_ROOT) are
    hashed and compared with the image's recorded hashes, surfaced by the User_Data as the
    UV_LOCK_HASH and PACKAGE_LOCK_HASH environment variables. A drift writes a redacted note to the
    Job's out/ and raises a CloudError; run_cloud_job's `finally` still powers the machine off.
    """
    archived = tuple(hashlib.sha256((REPO_ROOT / name).read_bytes()).hexdigest()
                     for name in _LOCK_FILES)
    recorded = (os.environ.get(_IMAGE_UV_HASH_ENV, ""),
                os.environ.get(_IMAGE_PACKAGE_HASH_ENV, ""))
    if archived == recorded:
        return
    note = ("This job's uv.lock or package-lock.json doesn't match the machine image's recorded "
            "hashes, so the image's packages wouldn't match this job's code. Rebuild the machine "
            "image with `uv run python -m cloud.image`, then start the job again.")
    aws.s3_put_text(f"jobs/{job_id}/out/{_RUNNER_LOG}", account_redaction(note) + "\n")
    raise CloudError(note)


# The Cap_Deadline (step 3)

def _set_cap_deadline(started: datetime, preset: str, cloud_config: CloudConfig) -> None:
    """Set CAP_DEADLINE_ENV to the machine's start plus the Time_Cap minus the margin (Req 10.7).

    The Time_Cap is the Preset's `cap_minutes` and the margin is `upload_margin_minutes`, both from
    Config. A Step reads the variable to know by when it should have saved and stopped; `splatpipe/`
    reads it in a later spec, so no current Step does and `splatpipe/` is unchanged.
    """
    cap_minutes = cloud_config.preset_plan(preset).cap_minutes
    deadline = started + timedelta(minutes=cap_minutes - cloud_config.upload_margin_minutes)
    os.environ[CAP_DEADLINE_ENV] = deadline.astimezone(timezone.utc).isoformat()


# The Spend_Lines (step 6)

def _write_spend_lines(aws: Aws, job_id: str, folder: Path, summary: dict[str, Any],
                       started: datetime, ended: datetime, cloud_config: CloudConfig) -> None:
    """Append a Spend_Line per Step that ran and one for the rest of the machine's time (Req 12.1, 12.2).

    Each Step that ran this time (not reused, not not-run) gets a line labelled with its Step_Name
    and timed by its report's `duration_seconds`; the machine line covers the rest of the elapsed
    wall time (boot, setup and uploads). Every line takes the type's price from Config and is
    labelled an estimate. The lines go straight into the Job's out/spend.jsonl in the bucket through
    `append_spend_line`, which stamps each id, so re-merging is idempotent.
    """
    instance_type = os.environ.get(_INSTANCE_TYPE_ENV, "")
    price = cloud_config.spec(instance_type).price_per_hour_usd
    step_seconds = 0.0
    for step in summary["steps"]:
        if step["reused"] or step["outcome"] == "not_run":
            continue
        seconds = _report_duration(folder, step["step"])
        step_seconds += seconds
        append_spend_line(aws, job_id, spend_line(
            job_id=job_id, label=step["step"], instance_type=instance_type, seconds=seconds,
            price_per_hour_usd=price, index=0))
    machine_seconds = max(0.0, (ended - started).total_seconds() - step_seconds)
    append_spend_line(aws, job_id, spend_line(
        job_id=job_id, label=LABEL_MACHINE, instance_type=instance_type, seconds=machine_seconds,
        price_per_hour_usd=price, index=0))


def _report_duration(folder: Path, name: str) -> float:
    """The `duration_seconds` in a Step's report, or 0.0 when it's missing or unreadable."""
    try:
        report = read_json(folder / "out" / "reports" / f"{name}.json")
    except (OSError, ValueError, RecursionError):
        return 0.0
    seconds = report.get("duration_seconds") if isinstance(report, dict) else None
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)):
        return 0.0
    return float(seconds)


# The runner's own log and the log redaction (step 7)

@contextmanager
def _runner_log(folder: Path) -> Iterator[None]:
    """Write the cloud.runner logger's lines, INFO and up, to out/runner.log while the Job runs.

    The handler is removed and the file closed on the way out, so the log can be redacted in place
    before it is uploaded (Req 10.10).
    """
    path = folder / "out" / _RUNNER_LOG
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(path, "a", encoding="utf-8", errors="backslashreplace", newline="\n")
    handler = logging.StreamHandler(handle)
    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_LOG_TIME)
    formatter.converter = time.gmtime
    handler.setFormatter(formatter)
    old_level = _LOG.level
    try:
        _LOG.addHandler(handler)
        _LOG.setLevel(logging.INFO)
        yield
    finally:
        _LOG.removeHandler(handler)
        handler.close()
        handle.close()
        _LOG.setLevel(old_level)


def _redact_text_logs(folder: Path) -> None:
    """Redact the text logs in place before upload, leaving the JSON reports unchanged (Req 10.10).

    out/job.log and out/runner.log are rewritten Account_Redacted; cloud-init's own log is read from
    CLOUD_INIT_LOG, redacted and written into out/ so the Uploader carries a safe copy. The JSON
    reports hold no account id, so they are never passed through the redaction.
    """
    out = folder / "out"
    for name in (_JOB_LOG, _RUNNER_LOG):
        path = out / name
        if path.exists():
            _write_text(path, account_redaction(_read_text(path)))
    if CLOUD_INIT_LOG.exists():
        _write_text(out / _CLOUD_INIT_NAME, account_redaction(_read_text(CLOUD_INIT_LOG)))


# Small file helpers

def _files_under(folder: Path) -> list[Path]:
    """Every file under `folder`, sorted, or an empty list when the folder isn't there yet."""
    if not folder.is_dir():
        return []
    return sorted(path for path in folder.rglob("*") if path.is_file())


def _report_step(rel: str) -> str | None:
    """The Step name of an `out/reports/<name>.json` relative path, or None when it isn't a report."""
    prefix, suffix = "reports/", ".json"
    if rel.startswith(prefix) and rel.endswith(suffix) and "/" not in rel[len(prefix):]:
        return rel[len(prefix):-len(suffix)]
    return None


def _read_text(path: Path) -> str:
    """Read a text log as UTF-8, replacing any stray bytes so redaction never fails on the content."""
    return path.read_text(encoding="utf-8", errors="backslashreplace")


def _write_text(path: Path, text: str) -> None:
    """Write `text` to `path` as UTF-8 with LF line endings, making the parent folder if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
