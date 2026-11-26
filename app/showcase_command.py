"""The Showcase_Command, run in AWS CloudShell as the account owner only:

    python3.11 -m app.showcase_command add <job_id> --title T --licence L [--credit C]
    python3.11 -m app.showcase_command list
    python3.11 -m app.showcase_command remove <job_id>
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, TextIO

from splatpipe.config import load_config

from app.cloudshell import CONFIG_PATH, require_cloudshell, run_cloudshell_cli, stack_aws
from app.config import website_config_from
from app.showcase import FOOTAGE_LICENCES, add_entry, check_entry, list_entries, remove_entry
from cloud.aws import cloudshell_aws
from cloud.errors import CloudError

if TYPE_CHECKING:
    from cloud.aws import Aws


def showcase_main(argv: Sequence[str], *, environ: Mapping[str, str],
                  make_aws: Callable[[str], Aws], out: TextIO, config_path: Path,
                  now: datetime) -> int:
    def body() -> int:
        args = _parser().parse_args(list(argv))
        require_cloudshell(environ)
        config = load_config(config_path)
        wc = website_config_from(config)
        if args.action == "add":
            entry = check_entry(args.job_id, args.title, args.licence, args.credit, wc, now)
            aws, _ = stack_aws(config, make_aws, table=True, bucket=True)
            add_entry(aws, entry, config)
            print(f"Added job {entry.job_id} to the showcase.", file=out)
            return 0
        aws, _ = stack_aws(config, make_aws, table=True, bucket=False)
        if args.action == "list":
            for entry in list_entries(aws):
                print("\t".join([entry.job_id, entry.title, entry.licence, entry.credit or ""]),
                      file=out)
            return 0
        if not remove_entry(aws, args.job_id):
            raise CloudError(f"Job {args.job_id} isn't in the showcase. See `list`.")
        print(f"Removed job {args.job_id} from the showcase.", file=out)
        return 0

    return run_cloudshell_cli(body)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python3.11 -m app.showcase_command", allow_abbrev=False,
        description="Add, list or remove the website's showcase entries. Run it only in AWS "
                    "CloudShell, signed in as the account owner.")
    actions = parser.add_subparsers(dest="action", required=True)
    add = actions.add_parser("add", help="show a finished job to every signed-in account")
    add.add_argument("job_id")
    add.add_argument("--title", required=True)
    add.add_argument("--licence", required=True, choices=sorted(FOOTAGE_LICENCES))
    add.add_argument("--credit", help="the footage author's credit; needed for cc-by and cc-by-sa")
    actions.add_parser("list", help="list the entries")
    remove = actions.add_parser("remove", help="remove an entry")
    remove.add_argument("job_id")
    return parser


def main() -> int:
    return showcase_main(sys.argv[1:], environ=os.environ, make_aws=cloudshell_aws, out=sys.stdout,
                         config_path=CONFIG_PATH, now=datetime.now(timezone.utc))


if __name__ == "__main__":
    sys.exit(main())
