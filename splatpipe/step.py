"""The Step_Runner: it runs one Step on one Job and writes the Step's Step_Report.

`run_step` works in two parts. First it opens the Job: it checks the Step_Name, loads the Config,
opens the Job and picks the Preset, the given one or the Job_Record's. A refusal there raises a
PipelineError and writes no Step_Report (Req 7.2, 7.3, 7.8). Then it runs the Step: it removes the
Step's earlier report, loads the Step's settings, checks and empties the Work_Folder, imports the
Step's module and calls the module's `work` with a StepContext. Any failure from there on goes into
a failed Step_Report. Either way, the report is written to out/reports/<Step_Name>.json and
returned. `run_step` never reads the earlier report, so it runs every Step it's given (Req 7.9). In
the Job_Folder it touches only work/<Step_Name>/ and out/reports/<Step_Name>.json (Req 7.6, 7.10).
`step_main` is the CLI that each Step's __main__ calls.

From splatpipe, this module imports only splatpipe.errors, splatpipe.config, splatpipe.jsonio and
splatpipe.jobs. It imports a Step's module by name, and only when the Step runs.
"""

from __future__ import annotations

import argparse
import importlib
import re
import sys
import time
import traceback
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, NoReturn

from splatpipe.config import PRESETS, Config, StepSettings, check_preset, load_config
from splatpipe.errors import PipelineError, run_cli
from splatpipe.jobs import Job, open_job, remove_tree
from splatpipe.jsonio import NotJsonError, check_json_value, write_json

# The Step_Name pattern and the Reserved_Names stay in code, because they're safety checks: a
# Step_Name becomes part of paths and of a module name. Only explicit ASCII classes, checked with
# fullmatch, as for the Job_Id.
STEP_NAME = re.compile(r"[a-z][a-z0-9_]*")
# The Skeleton's module names, jobs included, which is also the Skeleton's own Config section,
# and the sections in NOT_PIPELINE. A real Step's module is splatpipe.<Step_Name>, so a Step
# called run or config would clash with the Skeleton's own module.
RESERVED_NAMES = frozenset({"errors", "config", "jsonio", "jobs", "probe", "make", "step", "run",
                            "git", "aws", "paid", "devlog"})


def check_step_name(name: object) -> str:
    """Return `name` when it's a str that wholly matches STEP_NAME and isn't a Reserved_Name.

    Anything else is refused with a PipelineError. A name that fails the pattern is left out of
    the message, so a line break or a control character in it can't garble the terminal.
    """
    if not isinstance(name, str) or STEP_NAME.fullmatch(name) is None:
        raise PipelineError("That isn't a step name. A step name has only lowercase letters a-z, "
                            "digits 0-9 and underscores, and starts with a letter, like fake. "
                            "Give the step a name like that.")
    if name in RESERVED_NAMES:
        raise PipelineError(f"A step can't be called {name}, because the pipeline itself uses "
                            "that name for one of its modules or config sections. Give the step "
                            "another name.")
    return name


@dataclass
class StepContext:
    """What a Step's `work` gets: the Job, the Step's settings and folders, and its report fields.

    The Step adds its timings, counts and warnings with the add_* methods. Each refuses a bad name
    or value straight away with NotJsonError, naming the field, so the report never holds a bad
    value (Req 6.8). Adding a name a second time replaces its value.
    """

    name: str  # the Step_Name
    preset: str  # the Preset the Step runs with
    job: Job
    settings: StepSettings  # the Step's Shared_Settings and Preset_Settings for `preset`
    work_folder: Path  # job.folder / "work" / name, emptied before `work` runs
    out_folder: Path  # job.folder / "out"
    timings: dict[str, float]  # seconds, by name
    counts: dict[str, int]  # whole numbers, by name
    warnings: list[str]

    def add_timing(self, name: str, seconds: float) -> None:
        """Add the timing `name`: a finite int or float of seconds, 0 or more, never a bool."""
        field = _named_field("timings", name, seconds)
        if type(seconds) not in (int, float) or seconds < 0:
            _refuse(field, seconds, "a number of seconds, 0 or more")
        self.timings[name] = seconds

    @contextmanager
    def timed(self, name: str) -> Iterator[None]:
        """Time the block with time.perf_counter, then add the seconds as the timing `name`."""
        started = time.perf_counter()
        yield
        self.add_timing(name, time.perf_counter() - started)

    def add_count(self, name: str, value: int) -> None:
        """Add the count `name`: a whole number, 0 or more, as an int. A bool or a float is refused."""
        field = _named_field("counts", name, value)
        if type(value) is not int or value < 0:
            _refuse(field, value, "a whole number, 0 or more, given as an int")
        self.counts[name] = value

    def add_warning(self, text: str) -> None:
        """Add a warning: some text, not empty."""
        field = f"warnings[{len(self.warnings)}]"
        check_json_value(text, field)
        if type(text) is not str or not text:
            _refuse(field, text, "some text, not empty")
        self.warnings.append(text)


def run_step(name: str, module: str, job_id: str, *, preset: str | None = None,
             config: Config | None = None) -> dict[str, Any]:
    """Run the Step `name`, whose code is `work` in the module `module`, on the Job `job_id`.

    Returns the Step_Report, which is also written to out/reports/<name>.json. `preset` replaces
    the Job_Record's Preset (Req 7.4), and `config` defaults to load_config(). A refusal while the
    Job is opened raises a PipelineError and writes no report. A failure while the Step runs gives
    a failed report instead. KeyboardInterrupt and SystemExit aren't caught, so Ctrl+C still stops
    a run.
    """
    name = check_step_name(name)
    if config is None:
        config = load_config()
    job = open_job(job_id, config)
    chosen = job.record.preset if preset is None else check_preset(preset)
    return _run(name, module, job, chosen, config)


def step_main(name: str, module: str, argv: list[str] | None) -> int:
    """The CLI each Step's __main__ calls: `uv run python -m <module> --job <id> [--preset <name>]`.

    A done Step prints `<name>: done` on stdout and gives 0. A failed Step prints its
    Readable_Error on stderr and gives 1, for a Required_Step and an Optional_Step alike
    (Req 9.3). A refusal before the Step runs also prints its Readable_Error on stderr and gives
    1, through run_cli. A bad command line is argparse's usage error, exit code 2.
    """
    def body() -> int:
        args = _parser(name, module).parse_args(argv)
        report = run_step(name, module, args.job, preset=args.preset)
        if report["outcome"] == "done":
            print(f"{name}: done")
            return 0
        print(report["error"]["readable"], file=sys.stderr)
        return 1

    return run_cli(body)


# Running the Step

def _run(name: str, module: str, job: Job, preset: str, config: Config) -> dict[str, Any]:
    """Run the Step on the opened Job, then write its Step_Report, done or failed, and return it."""
    path = _cleared_report(job, name)
    started_utc, started = _utc_now(), time.perf_counter()
    timings: dict[str, float] = {}
    counts: dict[str, int] = {}
    warnings: list[str] = []
    settings: StepSettings | None = None
    error: dict[str, Any] | None = None
    try:
        settings = config.step_settings(name, preset)
        work_folder = _emptied_work_folder(job, name)
        work = importlib.import_module(module).work  # looked up now, so a patched work is seen
        work(StepContext(name=name, preset=preset, job=job, settings=settings,
                         work_folder=work_folder, out_folder=job.folder / "out",
                         timings=timings, counts=counts, warnings=warnings))
    except PipelineError as failure:  # an ExpectedFailure included: its message is readable
        error = _error(failure, str(failure), expected=True)
    except Exception as failure:  # anything else, a failed import included, is a bug
        error = _error(failure, _bug_message(name), expected=False)
    report = {
        "step": name,
        "job_id": job.id,
        "preset": preset,
        # A plain copy: check_json_value accepts only the exact type dict, not StepSettings.
        "settings": None if settings is None else dict(settings),
        "outcome": "done" if error is None else "failed",
        "started_utc": started_utc,
        "finished_utc": _utc_now(),
        "duration_seconds": round(time.perf_counter() - started, 3),
        "timings": timings,
        "counts": counts,
        "warnings": warnings,
        "error": error,
    }
    write_json(path, report)
    return report


def _cleared_report(job: Job, name: str) -> Path:
    """The path of the Step's report, with the earlier report removed and out/reports/ made (A1)."""
    reports = job.folder / "out" / "reports"
    path = reports / f"{name}.json"
    path.unlink(missing_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    return path


def _emptied_work_folder(job: Job, name: str) -> Path:
    """The Step's Work_Folder, emptied, once its resolved path is inside the Job_Folder (A1).

    A Work_Folder that leads outside the Job_Folder is refused before anything is touched
    (Req 7.7).
    """
    folder = job.folder / "work" / name
    resolved = folder.resolve()
    root = job.folder.resolve()
    if root not in resolved.parents:
        raise PipelineError(f"The {name} step's work folder {folder} leads to {resolved}, which "
                            f"is outside the job folder {root}. A step's work folder must be a "
                            "real folder inside the job folder, so remove the link that leads "
                            "out of it, then run the step again.")
    if folder.exists():
        remove_tree(folder)
    folder.mkdir(parents=True)
    return folder


def _error(failure: Exception, readable: str, *, expected: bool) -> dict[str, Any]:
    """The report's error: the Readable_Error, whether it was expected, and the Technical_Detail."""
    kind = type(failure)
    return {
        "readable": readable,
        "expected": expected,
        "technical": {
            "type": f"{kind.__module__}.{kind.__qualname__}",
            "message": str(failure),
            "traceback": "".join(traceback.format_exception(failure)),
        },
    }


def _bug_message(name: str) -> str:
    """The general Readable_Error for an Unexpected_Error in the Step `name` (Req 6.6)."""
    return (f"The {name} step stopped because of an error in its code. This is a bug, not a "
            f"problem with your video. Send out/reports/{name}.json to the team.")


def _utc_now() -> str:
    """The time now in UTC, to the millisecond, like 2026-11-19T14:25:31.204+00:00."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


# The values a Step adds to its report

def _named_field(kind: str, name: object, value: object) -> str:
    """The field `<kind>.<name>`, once `name` is a str a JSON file can hold and `value` a JSON_Value.

    `kind` is timings or counts. A refusal is a NotJsonError that names the field.
    """
    if type(name) is not str:
        raise NotJsonError(f"A name in {kind} has the type {type(name).__name__}, but each name "
                           "must be a str.")
    check_json_value({name: value}, kind)
    return f"{kind}.{name}"


def _refuse(field: str, value: object, wanted: str) -> NoReturn:
    """Refuse a value a Step adds to its report, naming the field and saying what it must be."""
    raise NotJsonError(f"{field} is {value!r}, but it must be {wanted}.")


def _parser(name: str, module: str) -> argparse.ArgumentParser:
    """The command line: --job, which is required, and an optional --preset."""
    parser = argparse.ArgumentParser(prog=f"uv run python -m {module}",
                                     description=f"Run the {name} step on one job.")
    parser.add_argument("--job", required=True, help="the job id that splatpipe.make printed")
    parser.add_argument("--preset", choices=PRESETS,
                        help="the preset (default: the preset in the job's job record)")
    return parser
