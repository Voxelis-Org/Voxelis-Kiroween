"""The Stop_All_Command: terminate every one of our GPU machines, the last-resort end.

`stop_all` terminates our non-terminated instances by the Project_Tag through the `Aws` seam and
returns the ids it terminated, so it only ever touches our own resources (Req 13.4, 13.5).
Terminating a machine is free, so this command calls no Spend_Guard and needs no `APPROVED_SPEND`
(Req 6.5): it is the safety switch that stops the money when something is stuck, so it must never be
blocked by the spend gate. Because it starts a machine down, `main` builds its `Aws` from the
Runner_Profile (`[aws].runner_profile`), the only laptop profile allowed to terminate (Req 1.5).

Run as `uv run python -m cloud.stopall`.

A cloud module may import `splatpipe` to call its public functions; `splatpipe` never imports
`cloud`. This module imports no boto3 of its own (only `cloud.aws` does), no `cloud.spend`, and no
GPU package, so it loads on a laptop.
"""

from __future__ import annotations

import argparse
import sys
from typing import TYPE_CHECKING

from splatpipe.config import load_config

from cloud.aws import laptop_aws
from cloud.errors import run_cloud_cli
from cloud.redact import account_redaction

if TYPE_CHECKING:
    from cloud.aws import Aws

# The non-terminated instance states: every machine still alive on the account, so Stop_All reaches
# every machine that could still be costing money, and skips the ones already terminated.
_LIVE_STATES = ("pending", "running", "shutting-down", "stopping", "stopped")


def stop_all(*, aws: Aws) -> list[str]:
    """Terminate our non-terminated instances by the Project_Tag; return the ids terminated (Req 13.4).

    Calls no Spend_Guard: terminating is free, and this is the last-resort end that stops the money,
    so it must never be gated on an approval (Req 6.5).
    """
    ids = [machine.id for machine in aws.tagged_instances(_LIVE_STATES)]
    aws.terminate_instances(ids)
    return ids


def main(argv: list[str] | None = None) -> int:
    """The Stop_All_Command CLI: terminate our machines, print what it ended and return 0.

    It builds the Runner_Profile `Aws` from `[aws].runner_profile` and calls no Spend_Guard, so it
    runs without `APPROVED_SPEND`. Any failure prints its Readable_Error on stderr and returns 1,
    through `run_cloud_cli`, which redacts any account id first. A bad command line is argparse's
    exit 2.
    """
    def body() -> int:
        _parser().parse_args(argv)
        config = load_config()
        aws = laptop_aws(config, profile_key="runner_profile")
        _report(stop_all(aws=aws))
        return 0

    return run_cloud_cli(body)


def _report(ids: list[str]) -> None:
    """Print what Stop_All terminated, Account_Redacted, or a nothing-to-stop line."""
    if not ids:
        print(account_redaction("No machines were running, so there was nothing to stop."))
        return
    noun = "machine" if len(ids) == 1 else "machines"
    print(account_redaction(f"Terminated {len(ids)} {noun}: {', '.join(ids)}."))


def _parser() -> argparse.ArgumentParser:
    """The command line: no arguments, so `--help` works and a stray argument is argparse's exit 2."""
    return argparse.ArgumentParser(
        prog="uv run python -m cloud.stopall", allow_abbrev=False,
        description="Terminate every one of our GPU machines in the cloud, the last-resort way to "
                    "stop the money. Free, so it needs no spend approval.")


if __name__ == "__main__":
    sys.exit(main())
