"""The Download_Command: fetch one Job's results to the laptop and fold its spend into the ledger.

`download` is free and read-only. It fetches `jobs/<id>/out/*` from the Job_Bucket, the Step reports
included, into `data/jobs/<id>/out/` through the `Aws` seam, and merges the Job's `out/spend.jsonl`
into the laptop's `data/spend.jsonl` without duplicates (Req 13.2, 12.4). The files are the Job's own
`out/` artifacts (the Job_Summary, the logs and the reports), so a downloaded Job folder looks like a
Job that ran on the laptop. Any AWS error is printed Account_Redacted by `run_cloud_cli`, so no
account id reaches the terminal (Req 13.6).

Run as `uv run python -m cloud.download <job id>`. Because it only reads, `main` builds its `Aws`
from the Read_Profile (`[aws].read_profile`) and calls no Spend_Guard (Req 1.4).

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


def download(job_id: str, *, aws: Aws, outputs: StackOutputs, data_root: Path) -> None:
    """Fetch the Job `job_id`'s `out/` files to `data/jobs/<id>/out/` and merge its Spend_Lines.

    Reads `jobs/<id>/out/*` from the Job_Bucket with the Read_Profile, the Step reports included,
    into `data_root/jobs/<id>/out/`, and merges `jobs/<id>/out/spend.jsonl` into
    `data_root/spend.jsonl`, which the caller keeps under `data/` (Req 13.2, 12.4). Any AWS error is
    redacted by `run_cloud_cli` before it is printed (Req 13.6).
    """
    job_id = check_job_id(job_id)
    aws.bucket = outputs.bucket  # BotoAws refuses an S3 call until its bucket is set (Req 1.3)

    prefix = f"jobs/{job_id}/out/"
    out_folder = data_root / "jobs" / job_id / "out"
    fetched = 0
    for key in aws.s3_list(prefix):
        rel = key[len(prefix):]
        if not rel:  # the prefix itself, a folder marker
            continue
        aws.s3_download(key, out_folder / rel)
        fetched += 1
    _announce_download(job_id, fetched, out_folder)

    added = _merge_job_spend(aws, job_id, data_root)
    _announce_merge(added, data_root / "spend.jsonl")


def main(argv: list[str] | None = None) -> int:
    """The Download_Command CLI: fetch the Job's results, merge its spend, and return 0.

    It builds the Read_Profile `Aws` from `[aws].read_profile`, resolves the Stack for the bucket,
    and runs `download` with `data/` taken from the Jobs_Folder's parent. Any failure prints its
    Readable_Error on stderr and returns 1, through `run_cloud_cli`, which redacts any account id
    first. A bad command line is argparse's exit 2.
    """
    def body() -> int:
        args = _parser().parse_args(argv)
        config = load_config()
        aws = laptop_aws(config, profile_key="read_profile")
        outputs = resolve_stack(aws, _aws_setting(config, "stack_name"))
        download(args.job, aws=aws, outputs=outputs,
                 data_root=config.jobs().jobs_folder.parent)
        return 0

    return run_cloud_cli(body)


def _announce_download(job_id: str, fetched: int, out_folder: Path) -> None:
    """Say how many files were fetched, and where to, or that the Job has no output yet."""
    if fetched == 0:
        print(f"Job {job_id} has no output files yet, so nothing was downloaded.")
        return
    noun = "file" if fetched == 1 else "files"
    print(f"Downloaded {fetched} {noun} for job {job_id} to {out_folder}.")


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
        try:
            obj = json.loads(part)
        except ValueError:
            continue
        if isinstance(obj, dict) and isinstance(obj.get("id"), str):
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
        prog="uv run python -m cloud.download", allow_abbrev=False,
        description="Download one cloud job's output files to data/jobs/<id>/out/ and record its "
                    "spend in data/spend.jsonl. Read-only and free.")
    parser.add_argument("job", help="the job id that cloud.start printed")
    return parser


if __name__ == "__main__":
    sys.exit(main())
