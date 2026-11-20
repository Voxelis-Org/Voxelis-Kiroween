"""The Spend_Guard: nothing paid starts unless someone said yes to that run.

Every Paid_Entry_Point in cloud/ calls `check_spend()` first, before it creates or starts
anything, with the machine, the expected time and the rough cost. `check_spend()` reads the
Spend_Variable's name from [aws].spend_variable in config.toml and returns only when that
variable is set to exactly "1". Otherwise it raises `SpendRefused`, whose message is a
Readable_Error. It only reads the environment variables and never changes them.

Standard library only: it reads config.toml itself with tomllib and never imports splatpipe,
so it still works when the Pipeline package is broken. It starts nothing and talks to no AWS
service. It only decides.
"""

from __future__ import annotations

import os
import tomllib
from collections.abc import Mapping
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.toml"


class SpendRefused(Exception):
    """Nothing paid may start. The message says why and what to do."""


def check_spend(*, machine: str, expected_time: str, rough_cost: str,
                environ: Mapping[str, str] | None = None,
                config_path: Path | None = None) -> None:
    """Return when the run described may start, and raise SpendRefused when it may not.

    `environ` defaults to os.environ and is only read, with `.get`. `config_path` defaults
    to CONFIG_PATH, the repo's config.toml.
    """
    name = _spend_variable(CONFIG_PATH if config_path is None else config_path)
    _check_description(machine, expected_time, rough_cost)
    values = os.environ if environ is None else environ
    if values.get(name) != "1":  # exactly "1", with no stripping: " 1", "01" and "true" fail
        raise SpendRefused(
            f"Refused to start {machine} for {expected_time} ({rough_cost}): this run costs "
            f"money and {name} isn't set to exactly 1. Get a yes in the chat for that machine, "
            f"time and cost, then run this one command again with {name}=1.")


def _spend_variable(config_path: Path) -> str:
    """The Spend_Variable's name: [aws].spend_variable in the config file, a non-empty string."""
    try:
        with open(config_path, "rb") as handle:
            data = tomllib.load(handle)
    except OSError as error:
        raise SpendRefused(
            f"Refused to start a paid run: the config file {config_path} can't be opened "
            f"({error.strerror or error}). Check the path, or get config.toml back from git if "
            "it was removed, then run the command again.") from error
    except ValueError as error:  # TOMLDecodeError, a file that isn't UTF-8, or a too-long int
        raise SpendRefused(
            f"Refused to start a paid run: the config file {config_path} can't be read as TOML "
            f"({error}). Fix it there, then run the command again.") from error
    aws = data.get("aws")
    name = aws.get("spend_variable") if isinstance(aws, dict) else None
    if not isinstance(name, str) or not name:
        raise SpendRefused(
            f"Refused to start a paid run: in {config_path}, [aws] needs spend_variable, the "
            "name of the environment variable that says yes to a paid run, as text in quotes. "
            "Add it there, or get config.toml back from git, then run the command again.")
    return name


def _check_description(machine: str, expected_time: str, rough_cost: str) -> None:
    """Refuse an empty part, because nobody can approve a run they can't see."""
    parts = {"machine": machine, "expected time": expected_time, "rough cost": rough_cost}
    empty = [label for label, part in parts.items() if not part]
    if empty:
        raise SpendRefused(
            f"Refused to start a paid run: its description has no {' and no '.join(empty)}, so "
            "nobody can say yes to it. Give the machine, the expected time and the rough cost "
            "where the command sets them, in config.toml or in its code, then run it again.")
