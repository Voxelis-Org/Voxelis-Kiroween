"""The Fake_Step: a test step that writes a few small files and one result line, and makes no splat.

It's the working example that every step package copies. A step package has three parts:

- `work(step)`, the Step's own code. The Step_Runner calls it with a StepContext once it has
  emptied the Step's Work_Folder. Every path and setting comes from `step`, so the same code
  runs under any Step_Name, like fake_a and fake_b in the tests.
- `run(job_id, preset=None, *, config=None)`, the plain function. It runs the Step through the
  Step_Runner and returns the Step_Report.
- `__main__.py`, the CLI: `uv run python -m splatpipe.fake --job <id> [--preset <name>]`.

The settings are in [fake] in config.toml. Only tests change `fail`: "expected" makes the step
fail with its own Readable_Error, and "unexpected" makes it fail like a bug.

From splatpipe, this package imports only splatpipe.step and splatpipe.errors.
"""

from __future__ import annotations

from typing import Any

from splatpipe.errors import ExpectedFailure
# Config comes through splatpipe.step, so a step package needs no other Skeleton module.
from splatpipe.step import Config, StepContext, run_step


def run(job_id: str, preset: str | None = None, *,
        config: Config | None = None) -> dict[str, Any]:
    """Run the fake step on the Job `job_id` and return its Step_Report.

    `preset` replaces the Job_Record's Preset, and `config` defaults to load_config().
    """
    return run_step("fake", "splatpipe.fake", job_id, preset=preset, config=config)


def work(step: StepContext) -> None:
    """The fake step's own code. The Step_Runner turns anything it raises into a failed report."""
    video_bytes = step.job.video_path.stat().st_size
    with step.timed("write_files"):
        written = _write_files(step)
    # The counts and the warning come before the fail check, so a failed report has them too.
    step.add_count("files_written", written)
    step.add_count("video_bytes", video_bytes)
    step.add_warning("This is the fake step. It writes test files and makes no splat.")
    _fail_if_asked(step)
    # Only a done run gets here. It replaces the line an earlier run wrote. A failed run leaves
    # that line in place, because the Step_Runner doesn't touch a Step's files in out/.
    line = f"The {step.name} step ran on job {step.job.id} with the {step.preset} preset.\n"
    (step.out_folder / f"{step.name}.txt").write_text(line, encoding="utf-8", newline="\n")


def _write_files(step: StepContext) -> int:
    """Write the Preset's number of small files into the Work_Folder, and return how many."""
    content = b"x" * step.settings["file_bytes"]
    names = [f"file-{index:03d}.txt" for index in range(step.settings["files"])]
    for name in names:
        (step.work_folder / name).write_bytes(content)
    return len(names)


def _fail_if_asked(step: StepContext) -> None:
    """Fail on purpose when the setting fail asks for it. "none" lets the step go on."""
    fail = step.settings["fail"]
    if fail == "expected":
        raise ExpectedFailure(f'The fake step was set to fail by fail = "expected" in '
                              f'[{step.name}]. Set fail = "none" to make it pass.')
    if fail == "unexpected":
        raise RuntimeError(f'The fake step was set to fail with an unexpected error by '
                           f'fail = "unexpected" in [{step.name}].')
    if fail != "none":
        raise ExpectedFailure(f'The setting fail in [{step.name}] is {fail!r}, but it must be '
                              '"none", "expected" or "unexpected". Set fail = "none" to make '
                              'the step pass.')
