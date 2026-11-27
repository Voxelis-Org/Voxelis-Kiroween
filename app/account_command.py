"""The Account_Command: `python3.11 -m app.account_command <name> [--new-password | --remove]`.

Run in AWS CloudShell, signed in as the account owner, only. With no option it makes the Account and
prints its new password; `--new-password` gives it a new one, which ends its Sessions; `--remove`
deletes its table items. A new password is printed once, alone on its line, and goes nowhere else.
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

from app.accounts import check_account_name, make_account, remove_account, replace_password
from app.cloudshell import CONFIG_PATH, require_cloudshell, run_cloudshell_cli, stack_aws
from app.config import website_config_from
from app.web import SecretsRandom
from cloud.aws import cloudshell_aws

if TYPE_CHECKING:
    from app.web import Random
    from cloud.aws import Aws


def account_main(argv: Sequence[str], *, environ: Mapping[str, str],
                 make_aws: Callable[[str], Aws], random: Random, out: TextIO,
                 config_path: Path, now: datetime) -> int:
    def body() -> int:
        args = _parser().parse_args(list(argv))
        require_cloudshell(environ)
        name = check_account_name(args.name)
        config = load_config(config_path)
        wc = website_config_from(config)
        aws, _ = stack_aws(config, make_aws, table=True, bucket=False)
        if args.remove:
            count = remove_account(aws, name)
            print(f"Removed the account {name} and its {count} items in the website's table. "
                  "Its jobs' files stay in the job bucket.", file=out)
            return 0
        if args.new_password:
            password = replace_password(aws, name, wc, random)
        else:
            password = make_account(aws, name, wc, random, now)
        print(password, file=out)
        return 0

    return run_cloudshell_cli(body)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python3.11 -m app.account_command", allow_abbrev=False,
        description="Make a website account, give it a new password, or remove it. Run it only in "
                    "AWS CloudShell, signed in as the account owner.")
    parser.add_argument("name", help="the account's name, like judges")
    choice = parser.add_mutually_exclusive_group()
    choice.add_argument("--new-password", action="store_true",
                        help="give the account a new password, which signs it out everywhere")
    choice.add_argument("--remove", action="store_true",
                        help="remove the account and its items in the website's table")
    return parser


def main() -> int:
    return account_main(sys.argv[1:], environ=os.environ, make_aws=cloudshell_aws,
                        random=SecretsRandom(), out=sys.stdout, config_path=CONFIG_PATH,
                        now=datetime.now(timezone.utc))


if __name__ == "__main__":
    sys.exit(main())
