"""The Job_Runner: it runs the Step_List on one Job, picking up where an earlier run stopped.

`run_job` refuses a bad Step_List before it reads or writes anything. Then it opens the Job and
reads each listed Step's earlier Step_Report with `read_earlier`, the only part that reads one.
`first_to_run` picks the first Step to run: the first that isn't a Finished_Step, or the first
of all with `fresh` (A3). Before any Step runs, `run_job` removes the earlier Job_Summary and the
reports of the Steps it's about to run. `run_from` runs those Steps through the Step_Runner, in
order, until a Required_Step fails, and `job_status` gives the Job_Status. The run's lines go to
the Job_Log, out/job.log, and the Job_Summary to out/job.json. `main` is the CLI,
`uv run python -m splatpipe.run --job <id> [--fresh]`.

The Step_List holds module names as strings, so importing this module imports no Step module
(Req 12.4). `run_step` imports a Step's module only when the Step is about to run (Req 8.5).

From splatpipe, this module imports only splatpipe.errors, splatpipe.config, splatpipe.jsonio,
splatpipe.jobs and splatpipe.step.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from splatpipe.config import Config, load_config
from splatpipe.errors import PipelineError, run_cli
from splatpipe.jobs import Job, open_job
from splatpipe.jsonio import read_json, write_json
from splatpipe.step import check_step_name, run_step


@dataclass(frozen=True)
class StepEntry:
    """One Step in a Step_List (Req 8.1)."""

    name: str  # the Step_Name: its Config sections, its Step_Report and its Work_Folder
    module: str  # the module whose `work` is the Step's code: splatpipe.<Step_Name> for real Steps
    required: bool  # a Required_Step's failure fails the Job, an Optional_Step's doesn't


# The Step_List: the Frames_Step and the Train_Step, both required (Req 8.2, frames spec, D1, and
# train spec, D5, Req 11.4), then the Clean_Step, optional (clean spec, Req 8.4). The camera solve
# spec puts its entry between the Frames_Step and the Train_Step, so the order becomes frames,
# cameras, train, clean. The Fake_Step stays out, so it never runs on real Jobs, and tests that
# run it pass their own list.
STEP_LIST: tuple[StepEntry, ...] = (StepEntry("frames", "splatpipe.frames", True),
                                    StepEntry("train", "splatpipe.train", True),
                                    StepEntry("clean", "splatpipe.clean", False))


@dataclass(frozen=True)
class Earlier:
    """What a Step's report from an earlier run says, read before this run removes anything."""

    state: str  # "missing", "unreadable", "done" or "failed"
    readable_error: str | None = None  # the report's error.readable, kept when state is "failed"


# A Step's module must be inside this package, and the Job_Log's handler goes on its logger.
_PACKAGE = "splatpipe"
# Not __name__, which is __main__ under `python -m splatpipe.run`, outside the Job_Log.
_LOG = logging.getLogger(f"{_PACKAGE}.run")
_LOG_FORMAT = "%(asctime)s %(levelname)s %(message)s"
_LOG_TIME = "%Y-%m-%dT%H:%M:%SZ"  # with time.gmtime, so the Job_Log's times are in UTC
# The JSON spellings of the Step_Outcomes and Job_Statuses, as words for the Job_Log and the screen.
_WORDS = {"done": "done", "failed": "failed", "not_run": "not run",
          "done_with_warnings": "done with warnings"}

_NO_STEPS = ("No steps are set up yet, so there's nothing to run. Each step's spec adds the step "
             "to STEP_LIST in splatpipe/run.py. Until then, run one step on its own, like "
             "uv run python -m splatpipe.fake --job <id>.")


def check_step_list(entries: Sequence[StepEntry]) -> None:
    """Refuse a bad Step_List with a PipelineError, before any Step_Name is used in a path.

    Each Step_Name must pass check_step_name (Req 7.8) and be listed once (Req 8.11), and each
    module must be splatpipe.<something> (Req 12.2).
    """
    seen: set[str] = set()
    for entry in entries:
        name = check_step_name(entry.name)
        if name in seen:
            raise PipelineError(f"The step {name} is in the step list twice. A step can be in the "
                                "list only once, so take the extra one out.")
        seen.add(name)
        _check_module(name, entry.module)


def read_earlier(job: Job, name: str) -> Earlier:
    """What the Step `name`'s earlier report says. It never raises because of what it finds.

    No file gives "missing". A file that can't be read or isn't JSON gives "unreadable", and so
    does a value that isn't a dict for this Step and this Job whose outcome is "done" or "failed".
    Otherwise the state is the report's outcome, and a "failed" one keeps error.readable when
    that's a str.
    """
    try:
        report = read_json(job.folder / "out" / "reports" / f"{name}.json")
    except FileNotFoundError:
        return Earlier("missing")
    except (OSError, ValueError, RecursionError):  # can't open, not UTF-8 or JSON, NaN, too deep
        return Earlier("unreadable")
    if not _is_report(report, name, job.id):
        return Earlier("unreadable")
    if report["outcome"] == "done":
        return Earlier("done")
    return Earlier("failed", _readable(report.get("error")))


def first_to_run(entries: Sequence[StepEntry], earlier: Sequence[Earlier], fresh: bool) -> int:
    """The index of the first Step to run. Pure.

    With `fresh` it's 0 (Req 8.14). Without it, it's the first entry that isn't a Finished_Step,
    or len(entries) when every entry is one (Req 8.13, 8.15).
    """
    if fresh:
        return 0
    for index, (entry, before) in enumerate(zip(entries, earlier, strict=True)):
        if not _finished(entry, before):
            return index
    return len(entries)


def run_from(entries: Sequence[StepEntry], earlier: Sequence[Earlier], start: int,
             run_one: Callable[[StepEntry], tuple[str, str | None]]) -> list[dict[str, Any]]:
    """The run loop. It gives the Job_Summary's steps, in Step_List order, and touches no files.

    Each entry before `start` is reused: it gets its earlier state as its outcome and its earlier
    readable error, and `run_one` isn't called for it (Req 8.16, 8.17). From `start` on, `run_one`
    runs each entry and gives its outcome and readable error. A failed Optional_Step doesn't stop
    the loop (Req 8.7). After a failed Required_Step, `run_one` isn't called again, and every
    later entry is not_run (Req 8.6).
    """
    steps: list[dict[str, Any]] = []
    stopped = False
    for index, (entry, before) in enumerate(zip(entries, earlier, strict=True)):
        if index < start:
            steps.append(_summary_step(entry, before.state, before.readable_error, reused=True))
        elif stopped:
            steps.append(_summary_step(entry, "not_run", None, reused=False))
        else:
            outcome, readable = run_one(entry)
            steps.append(_summary_step(entry, outcome, readable, reused=False))
            stopped = entry.required and outcome == "failed"
    return steps


def job_status(steps: Sequence[Mapping[str, Any]]) -> str:
    """The Job_Status of the Job_Summary's steps (A4, Req 8.8, 8.17).

    "failed" when a Required_Step's outcome is failed, "done_with_warnings" when none is and some
    Step's outcome is failed, a reused one included, and "done" otherwise.
    """
    if _failed_required(steps) is not None:
        return "failed"
    if any(step["outcome"] == "failed" for step in steps):
        return "done_with_warnings"
    return "done"


def run_job(job_id: str, *, steps: Sequence[StepEntry] | None = None,
            config: Config | None = None, fresh: bool = False) -> dict[str, Any]:
    """Run the Step_List, or `steps` when given, on the Job `job_id`, and return the Job_Summary.

    `config` defaults to load_config(), loaded once for every Step. Without `fresh`, the Finished
    Steps at the start of the list are skipped and reused. With it, every Step runs. A refusal is
    a PipelineError and comes before anything is removed or written. The Job_Summary is also
    written to out/job.json. A PipelineError from a Step's run_step, like a Job_Folder deleted
    mid-run, stops the run, and no Job_Summary is written.
    """
    entries = STEP_LIST if steps is None else tuple(steps)
    if not entries:
        raise PipelineError(_NO_STEPS)
    check_step_list(entries)
    if config is None:
        config = load_config()
    job = open_job(job_id, config)
    earlier = [read_earlier(job, entry.name) for entry in entries]
    start = first_to_run(entries, earlier, fresh)
    _remove_earlier(job, entries[start:])
    with _job_log(job, fresh):
        return _run(job, config, entries, earlier, start, fresh)


def main(argv: list[str] | None = None) -> int:
    """The Job_Runner CLI: run the Step_List on one Job and print `Job <id>: <status>` on stdout.

    It gives 0 for done and done with warnings (Req 9.4), which a Job where every Step had already
    finished always is (Req 9.5). For failed it prints the failed Step's Readable_Error on stderr
    and gives 1. A refusal prints its Readable_Error on stderr and gives 1, through run_cli. A bad
    command line is argparse's usage error, exit code 2.
    """
    def body() -> int:
        args = _parser().parse_args(argv)
        summary = run_job(args.job, fresh=args.fresh)
        print(_status_line(summary))
        if summary["status"] == "failed":
            print(summary["readable_error"], file=sys.stderr)
            return 1
        return 0

    return run_cli(body)


# The Step_List and the earlier reports

def _check_module(name: str, module: object) -> None:
    """Refuse a Step's module that isn't splatpipe.<something>."""
    prefix = f"{_PACKAGE}."
    if not (isinstance(module, str) and module.startswith(prefix) and module != prefix):
        raise PipelineError(f"The step {name} has the module {module!r}, but a step's module must "
                            f"be inside {_PACKAGE}, like {prefix}{name}. Change the step's module "
                            "in the step list.")


def _is_report(value: object, name: str, job_id: str) -> bool:
    """Whether `value` is a dict for the Step `name` on the Job `job_id`, done or failed."""
    return (isinstance(value, dict) and value.get("step") == name
            and value.get("job_id") == job_id and value.get("outcome") in ("done", "failed"))


def _readable(error: object) -> str | None:
    """A failed report's error.readable when that's a str, and None when it isn't."""
    readable = error.get("readable") if isinstance(error, dict) else None
    return readable if isinstance(readable, str) else None


def _finished(entry: StepEntry, before: Earlier) -> bool:
    """A Finished_Step: done, or failed for an Optional_Step. `required` comes from the entry."""
    return before.state == "done" or (before.state == "failed" and not entry.required)


def _remove_earlier(job: Job, to_run: Sequence[StepEntry]) -> None:
    """Remove out/job.json and the reports of the Steps about to run (A3, Req 8.9).

    The reports of the skipped Steps and of Steps outside the Step_List stay.
    """
    out = job.folder / "out"
    (out / "job.json").unlink(missing_ok=True)
    for entry in to_run:
        (out / "reports" / f"{entry.name}.json").unlink(missing_ok=True)


# The run

def _run(job: Job, config: Config, entries: Sequence[StepEntry], earlier: Sequence[Earlier],
         start: int, fresh: bool) -> dict[str, Any]:
    """Run the Steps from `start` on and log them, then write the Job_Summary and return it."""
    started_utc, started = _utc_now(), time.perf_counter()
    _log_start(job, entries, earlier, start)
    steps = run_from(entries, earlier, start, _step_runner(job, config))
    for step in steps:
        if step["outcome"] == "not_run":
            _LOG.info("%s: not run", step["step"])
    status = job_status(steps)
    _LOG.info("job %s: %s", job.id, _WORDS[status])
    failed = _failed_required(steps)
    summary = {
        "job_id": job.id,
        "preset": job.record.preset,
        "fresh": fresh,
        "status": status,
        "started_utc": started_utc,
        "finished_utc": _utc_now(),
        "duration_seconds": round(time.perf_counter() - started, 3),
        "steps": steps,
        "failed_step": None if failed is None else failed["step"],
        "readable_error": None if failed is None else failed["readable_error"],
    }
    write_json(job.folder / "out" / "job.json", summary)
    return summary


def _step_runner(job: Job, config: Config) -> Callable[[StepEntry], tuple[str, str | None]]:
    """run_from's `run_one`: run one Step with run_step, with the Job_Record's Preset (Req 8.4)."""
    def run_one(entry: StepEntry) -> tuple[str, str | None]:
        _LOG.info("%s: started", entry.name)
        report = run_step(entry.name, entry.module, job.id, config=config)
        error = report["error"]
        readable = None if error is None else error["readable"]
        _log_outcome(entry, report["outcome"], readable)
        return report["outcome"], readable

    return run_one


def _summary_step(entry: StepEntry, outcome: str, readable_error: str | None, *,
                  reused: bool) -> dict[str, Any]:
    """One item of the Job_Summary's steps."""
    return {"step": entry.name, "required": entry.required, "outcome": outcome,
            "reused": reused, "readable_error": readable_error}


def _failed_required(steps: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    """The first step whose Step is required and whose outcome is failed, or None."""
    return next((step for step in steps if step["required"] and step["outcome"] == "failed"), None)


def _utc_now() -> str:
    """The time now in UTC, to the millisecond, like 2026-11-19T14:25:31.204+00:00."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


# The Job_Log

@contextmanager
def _job_log(job: Job, fresh: bool) -> Iterator[None]:
    """Write the splatpipe logger's lines, INFO and up, to out/job.log in UTC while the run lasts.

    Without `fresh` the lines go at the end of the file, which is started when it's missing
    (Req 8.18). With `fresh` a new file replaces it (Req 8.19). Afterwards, also when the run
    fails, the handler is removed and closed, the file is closed and the logger's old level is
    back, so no file stays open.
    """
    file = open(job.folder / "out" / "job.log", "w" if fresh else "a", encoding="utf-8",
                errors="backslashreplace", newline="\n")
    handler = logging.StreamHandler(file)
    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_LOG_TIME)
    formatter.converter = time.gmtime
    handler.setFormatter(formatter)
    logger = logging.getLogger(_PACKAGE)
    old_level = logger.level
    try:
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        yield
    finally:
        logger.removeHandler(handler)
        handler.close()
        file.close()
        logger.setLevel(old_level)


def _log_start(job: Job, entries: Sequence[StepEntry], earlier: Sequence[Earlier],
               start: int) -> None:
    """The run's first line, then, when Steps are skipped, the line that names them (Req 8.18)."""
    _LOG.info("job %s: %d steps, preset %s", job.id, len(entries), job.record.preset)
    if start > 0:
        skipped = [f"{entry.name} ({_WORDS[before.state]})"
                   for entry, before in zip(entries[:start], earlier[:start], strict=True)]
        _LOG.info("skipped, finished in an earlier run: %s", ", ".join(skipped))


def _log_outcome(entry: StepEntry, outcome: str, readable_error: str | None) -> None:
    """The outcome line of a Step that ran. A failed one's holds its Readable_Error (Req 8.7)."""
    if outcome == "done":
        _LOG.info("%s: done", entry.name)
    elif entry.required:
        _LOG.error("%s: failed, required so the job stops: %s", entry.name, readable_error)
    else:
        _LOG.warning("%s: failed, optional so the job goes on: %s", entry.name, readable_error)


# The command line

def _status_line(summary: Mapping[str, Any]) -> str:
    """`Job <id>: <status in words>`, saying so when nothing ran because every Step had finished."""
    line = f"Job {summary['job_id']}: {_WORDS[summary['status']]}"
    if all(step["reused"] for step in summary["steps"]):
        line += (". Every step had already finished in an earlier run, so nothing ran. Add "
                 "--fresh to run every step again.")
    return line


def _parser() -> argparse.ArgumentParser:
    """The command line: --job, which is required, and --fresh."""
    parser = argparse.ArgumentParser(
        prog="uv run python -m splatpipe.run",
        description="Run the steps of one job in order, from the first step that hasn't "
                    "finished in an earlier run.")
    parser.add_argument("--job", required=True, help="the job id that splatpipe.make printed")
    parser.add_argument("--fresh", action="store_true",
                        help="run every step, also the ones that finished in an earlier run")
    return parser


if __name__ == "__main__":
    sys.exit(main())
