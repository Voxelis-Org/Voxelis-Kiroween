"""The Website_Status of a Job, worked out from its files in the Job_Bucket, and the job list.

Starting while `out/` holds nothing; Running (step n of N) while it holds files but no Job_Summary;
Done, Done with warnings or Failed from the Job_Summary. Ended without a result only once
`ended_check_minutes` have passed, every machine with the Job's tag is terminated or none is found,
and a second read of `out/job.json` still finds none. Nothing caches a status.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any

from app.limits import job_key
from cloud.errors import CloudError
from cloud.table import Key, Prefix, Scalar
from splatpipe.errors import LOOK_UP_WARNING

if TYPE_CHECKING:
    from splatpipe.config import Config

    from app.config import WebsiteConfig
    from cloud.aws import Aws

FINAL = frozenset({"done", "done_with_warnings", "failed", "ended"})
DONE = frozenset({"done", "done_with_warnings"})
SUMMARY = "job.json"
CAMERAS = "cameras.json"
_REPORTS = "reports/"


@dataclass(frozen=True)
class StepTime:
    step: str
    outcome: str
    seconds: float | None


@dataclass(frozen=True)
class JobView:
    job_id: str
    preset: str
    started_at: datetime | None
    status: str
    step: int | None
    steps: int | None
    readable_error: str | None
    warnings: tuple[tuple[str, str], ...]
    run_seconds: float | None
    step_times: tuple[StepTime, ...]
    result_file: str | None
    has_cameras: bool = False
    capture_warning: str | None = None


def out_prefix(job_id: str) -> str:
    return f"jobs/{job_id}/out/"


def status_text(view: JobView) -> str:
    if view.status == "running":
        if view.step is not None and view.steps:
            return f"Running (step {view.step} of {view.steps})"
        return "Running"
    return {"starting": "Starting", "done": "Done", "done_with_warnings": "Done with warnings",
            "failed": "Failed", "ended": "Ended without a result"}.get(view.status, view.status)


def website_status(out_names: Sequence[str], summary: Mapping[str, Any] | None,
                   steps: int | None) -> tuple[str, int | None]:
    """The status and the running step n, pure. Never "ended" on its own."""
    if summary is not None and summary.get("status") in ("done", "done_with_warnings", "failed"):
        return str(summary["status"]), None
    if not out_names:
        return "starting", None
    reports = sum(1 for name in out_names if name.startswith(_REPORTS) and name.endswith(".json"))
    n = reports + 1
    return "running", min(n, steps) if steps else n


def ended_check_due(started_at: datetime | None, now: datetime, minutes: int) -> bool:
    return started_at is not None and now - started_at >= timedelta(minutes=minutes)


def result_file(out_names: Sequence[str], clean: Mapping[str, Any]) -> str | None:
    for name in (clean["clean_sog"], clean["train_ply"]):
        if name in out_names:
            return str(name)
    return None


def read_summary(aws: Aws, job_id: str) -> Mapping[str, Any] | None:
    """The Job_Summary, or None when it's missing or can't be read as JSON yet."""
    try:
        data = json.loads(aws.s3_get_text(out_prefix(job_id) + SUMMARY))
    except (CloudError, ValueError):
        return None
    return data if isinstance(data, Mapping) else None


def job_view(aws: Aws, job_id: str, job_item: Mapping[str, Scalar] | None, config: Config,
             wc: WebsiteConfig, now: datetime, *, timings: bool) -> JobView:
    prefix = out_prefix(job_id)
    names = [key[len(prefix):] for key in aws.s3_list(prefix)]
    summary = read_summary(aws, job_id) if SUMMARY in names else None
    item = job_item or {}
    preset = str(item.get("preset") or (summary or {}).get("preset") or "")
    started = _started(item, summary)
    steps = int(item["steps"]) if "steps" in item else _summary_steps(summary)
    status, step = website_status(names, summary, steps)
    if status in ("starting", "running") and ended_check_due(started, now, wc.ended_check_minutes):
        status, summary = _ended_check(aws, job_id, status, summary)
    step_times, capture_warning = _step_details(aws, prefix, names) if timings else ((), None)
    return _view(job_id, preset, started, status, step, steps, summary, names, config,
                 step_times, capture_warning)


def _ended_check(aws: Aws, job_id: str, status: str,
                 summary: Mapping[str, Any] | None) -> tuple[str, Mapping[str, Any] | None]:
    try:
        machines = aws.job_machines(job_id)
    except CloudError:
        return status, summary              # a failed lookup keeps Starting or Running
    if any(machine.state != "terminated" for machine in machines):
        return status, summary
    again = read_summary(aws, job_id)
    if again is not None and again.get("status") in ("done", "done_with_warnings", "failed"):
        return str(again["status"]), again
    return "ended", None


def _view(job_id: str, preset: str, started: datetime | None, status: str, step: int | None,
          steps: int | None, summary: Mapping[str, Any] | None, names: Sequence[str],
          config: Config, step_times: tuple[StepTime, ...], capture_warning: str | None) -> JobView:
    summary = summary or {}
    entries = [s for s in summary.get("steps", []) if isinstance(s, Mapping)]
    warnings = tuple((str(s.get("step")), str(s.get("readable_error") or ""))
                     for s in entries if s.get("required") is False and s.get("outcome") == "failed")
    result = None
    if status in DONE and preset:
        result = result_file(names, config.step_settings("clean", preset))
    cameras_done = any(s.get("step") == "cameras" and s.get("outcome") == "done" for s in entries)
    run_seconds = summary.get("duration_seconds")
    return JobView(job_id=job_id, preset=preset, started_at=started, status=status,
                   step=step if status == "running" else None, steps=steps,
                   readable_error=str(summary.get("readable_error")) if status == "failed"
                   and summary.get("readable_error") else None,
                   warnings=warnings,
                   run_seconds=float(run_seconds) if isinstance(run_seconds, (int, float)) else None,
                   step_times=step_times, result_file=result,
                   has_cameras=cameras_done and CAMERAS in names, capture_warning=capture_warning)


def _step_details(aws: Aws, prefix: str, names: Sequence[str]
                  ) -> tuple[tuple[StepTime, ...], str | None]:
    times: list[StepTime] = []
    capture_warning = None
    for name in names:
        if not (name.startswith(_REPORTS) and name.endswith(".json")):
            continue
        try:
            report = json.loads(aws.s3_get_text(prefix + name))
        except (CloudError, ValueError):
            continue
        if isinstance(report, Mapping):
            seconds = report.get("duration_seconds")
            step = name[len(_REPORTS):-len(".json")]
            outcome = str(report.get("outcome", ""))
            times.append(StepTime(step=step, outcome=outcome,
                                  seconds=float(seconds) if isinstance(seconds, (int, float)) else None))
            warnings = report.get("warnings")
            if (step == "cameras" and outcome == "done"
                    and isinstance(warnings, list) and LOOK_UP_WARNING in warnings):
                capture_warning = LOOK_UP_WARNING
    return tuple(times), capture_warning


def _started(item: Mapping[str, Scalar], summary: Mapping[str, Any] | None) -> datetime | None:
    if "started_at" in item:
        return datetime.fromtimestamp(int(item["started_at"]), tz=timezone.utc)
    text = (summary or {}).get("started_utc")
    try:
        return datetime.fromisoformat(str(text)) if text else None
    except ValueError:
        return None


def _summary_steps(summary: Mapping[str, Any] | None) -> int | None:
    steps = (summary or {}).get("steps")
    return len(steps) if isinstance(steps, list) else None


def may_see(aws: Aws, account: str, job_id: str) -> tuple[Mapping[str, Scalar] | None,
                                                          Mapping[str, Scalar] | None]:
    """The Account's held job item and the Job's Showcase_Entry, in one read. Either may be None."""
    item, entry = aws.table_get_many([job_key(account, job_id), Key("showcase", f"job#{job_id}")])
    held = item if item is not None and item.get("state") == "held" else None
    return held, entry


def list_jobs(aws: Aws, account: str, config: Config, wc: WebsiteConfig,
              now: datetime) -> list[JobView]:
    items = [item for item in aws.table_query(f"account#{account}", Prefix("job#"))
             if item.get("state") == "held"]
    items.sort(key=lambda item: str(item["sk"]), reverse=True)
    return [job_view(aws, str(item["sk"])[len("job#"):], item, config, wc, now, timings=False)
            for item in items[:wc.max_listed_jobs]]
