"""The Status_Command: show one Job's progress and fold its spend into the laptop ledger.

`status` is free and read-only. It reads the Job's `out/job.json` Job_Summary and its per-Step
reports from the Job_Bucket through the `Aws` seam, prints the Job's progress, and merges the Job's
`out/spend.jsonl` into the laptop's `data/spend.jsonl` without duplicates (Req 13.1, 12.4). The
JSON it reads holds no account id, so its progress lines are printed as they are; the one thing that
could carry an account id is an AWS error, and `run_cloud_cli` prints every error Account_Redacted,
so none reaches the terminal (Req 13.6). A Job that is still running has no Job_Summary yet (it is
written only when the run ends), so a missing summary is reported as "still running" and the
progress is read from the reports that have appeared.

Run as `uv run python -m cloud.status <job id>`. Because it only reads, `main` builds its `Aws` from
the Read_Profile (`[aws].read_profile`) and calls no Spend_Guard (Req 1.4).

A cloud module may import `splatpipe` to call its public functions; `splatpipe` never imports
`cloud`. This module imports no boto3 of its own (only `cloud.aws` does), no `cloud.spend`, and no
GPU package, so it loads on a laptop.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from splatpipe.config import load_config
from splatpipe.jobs import check_job_id

from cloud.aws import laptop_aws
from cloud.errors import CloudError, run_cloud_cli
from cloud.spend_file import merge_spend_file
from cloud.stack import resolve_stack

if TYPE_CHECKING:
    from splatpipe.config import Config

    from cloud.aws import Aws
    from cloud.stack import StackOutputs


def status(job_id: str, *, aws: Aws, outputs: StackOutputs, data_root: Path) -> dict[str, Any]:
    """Report the Job `job_id`'s progress and merge its Spend_Lines into `data/spend.jsonl`.

    Reads `jobs/<id>/out/job.json` and the Step reports from the Job_Bucket with the Read_Profile,
    prints the progress, and merges `jobs/<id>/out/spend.jsonl` into `data_root/spend.jsonl`, which
    the caller keeps under `data/` (Req 13.1, 12.4). Returns the Job_Summary when it exists, else a
    small progress dict synthesised from the reports for a Job that is still running. The JSON read
    here holds no account id, so it is printed unredacted; any AWS error is redacted by
    `run_cloud_cli` before it is printed (Req 13.6).
    """
    job_id = check_job_id(job_id)
    aws.bucket = outputs.bucket  # BotoAws refuses an S3 call until its bucket is set (Req 1.3)

    summary = _read_object(aws, f"jobs/{job_id}/out/job.json")
    reports = _read_reports(aws, job_id)
    _print_progress(job_id, summary, reports)

    added = _merge_job_spend(aws, job_id, data_root)
    _announce_merge(added, data_root / "spend.jsonl")

    if summary is not None:
        return summary
    return {"job_id": job_id, "status": "running", "steps": _steps_from_reports(reports)}


def main(argv: list[str] | None = None) -> int:
    """The Status_Command CLI: print the Job's progress, merge its spend, and return 0.

    It builds the Read_Profile `Aws` from `[aws].read_profile`, resolves the Stack for the bucket,
    and runs `status` with `data/` taken from the Jobs_Folder's parent. Any failure prints its
    Readable_Error on stderr and returns 1, through `run_cloud_cli`, which redacts any account id
    first. A bad command line is argparse's exit 2.
    """
    def body() -> int:
        args = _parser().parse_args(argv)
        config = load_config()
        aws = laptop_aws(config, profile_key="read_profile")
        outputs = resolve_stack(aws, _aws_setting(config, "stack_name"))
        status(args.job, aws=aws, outputs=outputs,
               data_root=config.jobs().jobs_folder.parent)
        return 0

    return run_cloud_cli(body)


# Reading the Job_Summary and the reports from the bucket

def _read_object(aws: Aws, key: str) -> dict[str, Any] | None:
    """The JSON object at `key` in the bucket, or None when it is missing or not a JSON object."""
    if not aws.s3_exists(key):
        return None
    return _parse_object(aws.s3_get_text(key))


def _read_reports(aws: Aws, job_id: str) -> dict[str, dict[str, Any]]:
    """Every Step report under `jobs/<id>/out/reports/`, as a map of Step_Name to its JSON object."""
    prefix = f"jobs/{job_id}/out/reports/"
    reports: dict[str, dict[str, Any]] = {}
    for key in aws.s3_list(prefix):
        rel = key[len(prefix):]
        if rel.endswith(".json") and "/" not in rel:
            report = _parse_object(aws.s3_get_text(key))
            if report is not None:
                reports[rel[: -len(".json")]] = report
    return reports


def _parse_object(text: str) -> dict[str, Any] | None:
    """`text` as a JSON object, or None when it isn't valid JSON or isn't an object."""
    try:
        data = json.loads(text)
    except ValueError:
        return None
    return data if isinstance(data, dict) else None


# Printing the progress (JSON only, so no account id, so no redaction, per Req 13.7)

def _print_progress(job_id: str, summary: dict[str, Any] | None,
                    reports: Mapping[str, dict[str, Any]]) -> None:
    """Print the Job's status and one line per Step, from the Job_Summary and the reports."""
    if summary is None:
        print(f"Job {job_id}: still running, no summary yet (it's written when the job ends).")
        for name in sorted(reports):
            print(_step_line(name, reports[name].get("outcome"), reports[name]))
        return
    header = f"Job {job_id}: {_words(summary.get('status'))}"
    seconds = _duration(summary.get("duration_seconds"))
    if seconds is not None:
        header += f", {seconds:.1f}s"
    print(header)
    steps = summary.get("steps")
    if isinstance(steps, list):
        for step in steps:
            if isinstance(step, dict):
                name = step.get("step")
                name = name if isinstance(name, str) else "?"
                line = _step_line(name, step.get("outcome"), reports.get(name))
                if step.get("reused"):
                    line += " [reused]"
                print(line)
    error = summary.get("readable_error")
    if summary.get("status") == "failed" and isinstance(error, str):
        print(f"  error: {error}")


def _step_line(name: str, outcome: object, report: object) -> str:
    """One Step's progress line: its outcome, and its duration from the report when there is one."""
    line = f"  {name}: {_words(outcome)}"
    seconds = _duration(report.get("duration_seconds")) if isinstance(report, dict) else None
    if seconds is not None:
        line += f" ({seconds:.1f}s)"
    return line


def _steps_from_reports(reports: Mapping[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """The Step progress for the return value when there is no Job_Summary, from the reports."""
    return [{"step": name, "outcome": reports[name].get("outcome")} for name in sorted(reports)]


def _words(value: object) -> str:
    """A status or outcome word for the screen, with underscores as spaces (`not_run` -> not run)."""
    return value.replace("_", " ") if isinstance(value, str) else "unknown"


def _duration(value: object) -> float | None:
    """A finite, non-bool number of seconds as a float, or None when the value isn't one."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


# Merging the Job's Spend_Lines into the laptop ledger

def _merge_job_spend(aws: Aws, job_id: str, data_root: Path) -> int:
    """Merge `jobs/<id>/out/spend.jsonl` into `data_root/spend.jsonl`, returning how many were added.

    A missing spend file counts as nothing to merge. `merge_spend_file` dedups by `id`, so merging
    the same lines twice adds nothing (Req 12.4).
    """
    key = f"jobs/{job_id}/out/spend.jsonl"
    if not aws.s3_exists(key):
        return 0
    return merge_spend_file(data_root / "spend.jsonl", _spend_lines(aws.s3_get_text(key)))


def _spend_lines(text: str) -> list[dict[str, Any]]:
    """The Spend_Lines in a JSON-Lines string: the JSON objects with a string `id`, one per line."""
    lines: list[dict[str, Any]] = []
    for part in text.splitlines():
        if not part.strip():
            continue
        obj = _parse_object(part)
        if obj is not None and isinstance(obj.get("id"), str):
            lines.append(obj)
    return lines


def _announce_merge(added: int, path: Path) -> None:
    """Say how many Spend_Lines the merge added to the laptop ledger."""
    if added == 0:
        print("No new spend lines to record.")
        return
    noun = "line" if added == 1 else "lines"
    print(f"Recorded {added} new spend {noun} in {path}.")


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


def _parser() -> argparse.ArgumentParser:
    """The command line: one job id, so `--help` works and a missing id is argparse's exit 2."""
    parser = argparse.ArgumentParser(
        prog="uv run python -m cloud.status", allow_abbrev=False,
        description="Show one cloud job's progress and record its spend in data/spend.jsonl. "
                    "Read-only and free.")
    parser.add_argument("job", help="the job id that cloud.start printed")
    return parser


if __name__ == "__main__":
    sys.exit(main())
