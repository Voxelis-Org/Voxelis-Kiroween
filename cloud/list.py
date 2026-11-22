"""The List_Command: show our own GPU machines, run as `uv run python -m cloud.list`.

`list_machines` reads our non-terminated instances by the Project_Tag through the `Aws` seam, so it
only ever sees our own resources (Req 13.5). It is free and read-only, so `main` builds its `Aws`
from the Read_Profile (`[aws].read_profile`) and calls no Spend_Guard (Req 1.4, 13.3). `main` prints
each machine's type, state and launch time, every line passed through `account_redaction` first, so
no account id or ARN can reach the terminal (Req 13.5, 13.7).

The module file is named `list.py` on purpose: a module named `list.py` does not bind the builtin
`list` in its own namespace, so the builtin stays available inside this file. It is run as
`uv run python -m cloud.list`.

A cloud module may import `splatpipe` to call its public functions; `splatpipe` never imports
`cloud`. This module imports no boto3 of its own (only `cloud.aws` does) and no GPU package, so it
loads on a laptop.
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
    from cloud.aws import Aws, InstanceInfo

# The non-terminated instance states: every machine still alive on the account, so the list shows
# every machine that could still be costing money, and leaves out the terminated ones.
_LIVE_STATES = ("pending", "running", "shutting-down", "stopping", "stopped")


def list_machines(*, aws: Aws) -> list[InstanceInfo]:
    """Our non-terminated instances, by the Project_Tag, read through the Read_Profile (Req 13.3)."""
    return aws.tagged_instances(_LIVE_STATES)


def main(argv: list[str] | None = None) -> int:
    """The List_Command CLI: print one line per machine on stdout and return 0.

    It builds the Read_Profile `Aws` from `[aws].read_profile` and prints each machine's type, state
    and launch time, every line Account_Redacted so no account id or ARN reaches the terminal. Any
    failure prints its Readable_Error on stderr and returns 1, through `run_cloud_cli`, which redacts
    any account id first. A bad command line is argparse's exit 2.
    """
    def body() -> int:
        _parser().parse_args(argv)
        config = load_config()
        aws = laptop_aws(config, profile_key="read_profile")
        _print_machines(list_machines(aws=aws))
        return 0

    return run_cloud_cli(body)


def _print_machines(machines: list[InstanceInfo]) -> None:
    """Print each machine's type, state and launch time, Account_Redacted, or a no-machines line."""
    if not machines:
        print(account_redaction("No machines are running right now."))
        return
    for machine in machines:
        print(account_redaction(f"{machine.type}  {machine.state}  launched {machine.launch_time}"))


def _parser() -> argparse.ArgumentParser:
    """The command line: no arguments, so `--help` works and a stray argument is argparse's exit 2."""
    return argparse.ArgumentParser(
        prog="uv run python -m cloud.list", allow_abbrev=False,
        description="List our own GPU machines in the cloud, newest-tagged first, with each one's "
                    "type, state and launch time. Read-only and free.")


if __name__ == "__main__":
    sys.exit(main())
