"""The COLMAP runner, the only code in the step that starts COLMAP.

COLMAP runs only as the separate program colmap, PROGRAM, through subprocess.run. It's never
linked or imported, and there's no pycolmap (Req 2.1, D12). A ColmapRun holds one COLMAP_Command:
its name, its whole command line, its Time_Limit and that limit's key in [cameras], its
Command_Log, its working folder and its environment. commands.py builds each one.

run_colmap runs the command line as an argument list with no shell and with standard input
closed, in the COLMAP_Folder, with the run's environment (Req 2.2). glog writes all of COLMAP's
output to stderr under --log_target stderr, and stdout goes into the same Command_Log, so the log
holds everything in the order COLMAP wrote it, also when the command fails or runs past its
Time_Limit (Req 2.10, A10). At the Time_Limit, subprocess.run kills COLMAP and waits for it, and
the RunResult's exit_code is None (Req 2.8). A colmap that disappeared after the step found it is
the Req 2.7 failure, and a Command_Log that can't be opened is the Req 11.12 failure. The seconds
come from time.perf_counter around the run. The Last_Error_Line comes from reading the
Command_Log once, line by line, holding one line at a time, so a long log needs no size limit.

colmap_env is pure. It gives the step's environment without TEST_TMPDIR, which glog reads before
TMPDIR, and without every GLOG_ variable, which could move or reformat its log, and with TMPDIR
and TMP set to a folder in the COLMAP_Folder. So the log files that glog writes before COLMAP has
read its options stay in the Work_Folder (C2).

last_error_line gives the last line that isn't empty before glog's "*** Check failure stack
trace: ***" line, without glog's prefix. So after a crash the Readable_Error shows the failed
check, not a stack frame, and the Command_Log keeps the full text (C3). exit_text writes an exit
code for the messages, with the signal's name when a signal stopped COLMAP.

From splatpipe, this module imports only splatpipe.errors.
"""

from __future__ import annotations

import re
import signal
import subprocess
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from splatpipe.errors import ExpectedFailure

PROGRAM = "colmap"  # Fixed_Constant: the COLMAP program the step runs

# The Req 2.7 failure, for a colmap that isn't on the PATH. The step's own check before the first
# COLMAP_Command gives it too.
NOT_INSTALLED = (f"COLMAP isn't installed: the program {PROGRAM} isn't on the PATH. Install "
                 "COLMAP 4.2.1, built without CUDA, OpenGL, the GUI, CGAL and LSD as "
                 ".kiro/steering/tech.md says, then run the step again.")

# The line glog writes after a failed check, before the stack trace (C3).
_STACK_TRACE = "*** Check failure stack trace: ***"
# glog's prefix, like "I20261123 04:19:19.624758 0x1705ab000 sift.cc:764] " (F16): the level
# letter, the date as mmdd or yyyymmdd, the time, the thread id in decimal or hex, which glog pads
# with spaces to 5 characters, and file:line].
_GLOG_PREFIX = re.compile(r"[IWEF](?:[0-9]{8}|[0-9]{4}) [0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]{6} +"
                          r"(?:0x[0-9a-fA-F]+|[0-9]+) [^\s:]+:[0-9]+\] ?")


@dataclass(frozen=True)
class ColmapRun:
    """One COLMAP_Command, as commands.py builds it and run_colmap runs it."""

    name: str  # the run's name, also its timing's and its log's, like "sequential_matcher_near"
    command: str  # COLMAP's command, like "sequential_matcher"
    args: tuple[str, ...]  # the whole command line, from PROGRAM
    timeout_seconds: float  # the Time_Limit
    timeout_key: str  # the Time_Limit key in [cameras], for the Req 2.8 message
    log: Path  # the Command_Log, colmap/logs/<name>.log
    cwd: Path  # the COLMAP_Folder
    env: Mapping[str, str]  # the environment, from colmap_env

    def value(self, option: str) -> str:
        """The argument after `option` in the command line, for the stand-in and the tests.
        `option` must be in the command line, with an argument after it."""
        return self.args[self.args.index(option) + 1]


@dataclass(frozen=True)
class RunResult:
    """What one COLMAP_Command did."""

    exit_code: int | None  # None: stopped at its Time_Limit
    seconds: float  # how long it ran, also when it failed or was stopped
    last_line: str  # the Last_Error_Line, "" when the log has none


def colmap_env(tmp: Path, base: Mapping[str, str]) -> dict[str, str]:
    """The environment of a COLMAP_Command: `base` without TEST_TMPDIR and without every key that
    starts with GLOG_, and with TMPDIR and TMP set to `tmp` (C2, A10). `base` stays as it is."""
    env = {key: value for key, value in base.items()
           if key != "TEST_TMPDIR" and not key.startswith("GLOG_")}
    env["TMPDIR"] = env["TMP"] = str(tmp)
    return env


def run_colmap(run: ColmapRun) -> RunResult:
    """Run `run`, with all its output in its Command_Log, and give its exit code, its seconds and
    its Last_Error_Line.

    The exit code is None when the run reached its Time_Limit (Req 2.8). A colmap that isn't
    there any more is the Req 2.7 failure, and a Command_Log that can't be opened is the Req 11.12
    failure, naming the log.
    """
    exit_code, seconds = _run_into_log(run)
    with run.log.open(encoding="utf-8", errors="replace") as lines:
        last_line = last_error_line(lines)
    return RunResult(exit_code=exit_code, seconds=seconds, last_line=last_line)


def last_error_line(lines: Iterable[str]) -> str:
    """The Last_Error_Line of a Command_Log's `lines`, taken in order (C3).

    The lines are read up to the first one that holds glog's "*** Check failure stack trace:
    ***". Of the lines before it, the last one that isn't empty is kept, without glog's prefix
    when it has one. That's "" when there's no such line.
    """
    last = ""
    for line in lines:
        if _STACK_TRACE in line:
            break
        if line.strip():
            last = line
    text = last.strip()
    prefix = _GLOG_PREFIX.match(text)
    return text[prefix.end():].strip() if prefix else text


def exit_text(code: int) -> str:
    """`code` for the messages: "exit code 1", or "exit code -6 (signal SIGABRT)" for the negative
    code that subprocess gives when a signal stopped COLMAP."""
    if code >= 0:
        return f"exit code {code}"
    try:
        name = signal.Signals(-code).name
    except ValueError:  # a signal Python has no name for, like a real-time signal
        name = str(-code)
    return f"exit code {code} (signal {name})"


def _run_into_log(run: ColmapRun) -> tuple[int | None, float]:
    """Run `run` with stdout and stderr in its Command_Log: its exit code, or None when
    subprocess.run stopped it at its Time_Limit, and the seconds it ran."""
    try:
        log = run.log.open("wb")
    except OSError as error:
        raise _log_not_opened(run, error) from error
    with log:
        start = time.perf_counter()
        try:
            completed = subprocess.run(list(run.args), stdin=subprocess.DEVNULL, stdout=log,
                                       stderr=subprocess.STDOUT, cwd=run.cwd, env=dict(run.env),
                                       timeout=run.timeout_seconds)
            exit_code: int | None = completed.returncode
        except subprocess.TimeoutExpired:  # subprocess.run has already killed COLMAP and waited
            exit_code = None
        except FileNotFoundError as error:  # colmap disappeared after the step found it
            raise ExpectedFailure(NOT_INSTALLED) from error
        return exit_code, time.perf_counter() - start


def _log_not_opened(run: ColmapRun, error: OSError) -> ExpectedFailure:
    """The Req 11.12 failure: `run`'s Command_Log couldn't be opened, the reason the system gave,
    and what to do."""
    return ExpectedFailure(f"The log file {run.log} for COLMAP's {run.command} couldn't be opened "
                           f"for writing ({error.strerror or error}). Free some space on the "
                           f"disk, or check that you can write to {run.log.parent}, then run the "
                           "step again.")
