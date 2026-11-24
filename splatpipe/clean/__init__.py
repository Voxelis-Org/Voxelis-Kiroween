"""The Clean_Step: it cleans the trained splat and makes the small copy for the web viewer.

It reads the splat the training step wrote, out/train.ply, and drops the Gaussians a viewer
shouldn't see: the broken ones, the ones too faint to add anything and the isolated floaters. It
writes the rest to out/clean.ply as a binary PLY, then runs splat-transform under Node to make
out/clean.sog for the web viewer. Every part of it is CPU work. The file names and every
threshold are settings in [clean] in config.toml.

The package is built like the Fake_Step, and this file holds its two entry points:

- `run(job_id, preset=None, *, config=None)`, the plain function. It runs the step through the
  Step_Runner and returns the Step_Report, which is also out/reports/clean.json.
- `work(step)`, the step's own code. The Step_Runner calls it once it has emptied work/clean/.

The CLI, `uv run python -m splatpipe.clean --job <id> [--preset <name>]`, is __main__.py. The
step is STEP_LIST's last entry, an optional one, and it also runs on its own through these.

`work` runs in four phases, and what out/ holds after a failure depends only on the phase it
stopped in (Req 4.8 to 4.13):

1. `check_settings` checks the eighteen settings before anything in the Job_Folder is opened,
   written or removed, so a refused setting leaves the earlier outputs as they were (Req 6.9).
2. `remove_outputs` removes the earlier out/clean.sog, then the earlier out/clean.ply. A file it
   can't remove stops the step and leaves the files it hadn't removed yet.
3. `clean_and_write` reads the trained splat, runs the three filters and the Requirement 3
   limits, and writes the clean splat into work/clean/. None of its arrays outlive it, and
   gc.collect() runs before node starts. Then `export_sog` makes the viewer copy from it, in
   work/clean/ too. A failure here leaves neither output in out/.
4. `move_outputs` moves the clean splat into out/, then the viewer copy. A failure between the
   two moves, Ctrl+C included, removes what had moved, so out/ again holds neither.

The viewer copy is removed first and moved last, so out/ never holds one without the clean
splat it was made from. `work` does nothing after the moves, so no failure of its own comes once
both outputs are in out/. A hard stop between the two moves, like a killed process, can leave
this run's whole clean splat alone in out/, with no done report, and the next run removes it
first.

The other modules do the work: settings.py checks the settings, ply_read.py reads the trained
splat, filters.py runs the three filters in order, limits.py holds the Requirement 3 limits,
ply_write.py writes the clean splat and export.py runs splat-transform.

From splatpipe, the package imports only splatpipe.step, splatpipe.errors and splatpipe.config,
and this file only splatpipe.step, splatpipe.errors and the package's own modules. Outside the
standard library the package imports only numpy and scipy.spatial, and nothing from cloud/,
boto3, torch or gsplat (Req 8.5).
"""

from __future__ import annotations

import gc
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from splatpipe.clean.export import export_sog
from splatpipe.clean.filters import run_filters
from splatpipe.clean.limits import check_limits, check_source_count
from splatpipe.clean.ply_read import read_header, read_splat
from splatpipe.clean.ply_write import write_splat
from splatpipe.clean.settings import CleanSettings, check_settings
from splatpipe.errors import ExpectedFailure
# Config comes through splatpipe.step, as in the Fake_Step.
from splatpipe.step import Config, StepContext, run_step


@dataclass(frozen=True)
class Written:
    """What clean_and_write leaves for the export: the Clean_Splat's path in work/clean/ and the
    trained splat's SH_Bands. It holds only a path and a number, so no array outlives the clean
    phase."""

    ply: Path
    sh_bands: int


def run(job_id: str, preset: str | None = None, *,
        config: Config | None = None) -> dict[str, Any]:
    """Run the clean step on the Job `job_id` and return its Step_Report.

    `preset` replaces the Job_Record's Preset, and `config` defaults to load_config().
    """
    return run_step("clean", "splatpipe.clean", job_id, preset=preset, config=config)


def work(step: StepContext) -> None:
    """The clean step's own code, in its four phases. The Step_Runner turns anything it raises
    into a failed report."""
    s = check_settings(step.settings)  # Req 6.6 to 6.10, before anything is touched
    ply_out, sog_out = step.out_folder / s.clean_ply, step.out_folder / s.clean_sog
    remove_outputs(sog_out, ply_out)  # Req 4.8: the viewer copy first
    written = clean_and_write(step, s)  # every array it made is freed when it returns
    gc.collect()  # before node starts, for any array held in a reference cycle (Req 8.9)
    bands = viewer_bands(step, s, written.sh_bands)
    sog, log_file = step.work_folder / s.clean_sog, step.work_folder / "splat-transform.log"
    with step.timed("export_sog"):
        sog_bytes = export_sog(written.ply, sog, bands, s, log_file)
    step.add_count("clean_sog_bytes", sog_bytes)
    move_outputs([(written.ply, ply_out), (sog, sog_out)])  # the last thing work does


def clean_and_write(step: StepContext, s: CleanSettings) -> Written:
    """Read the trained splat, run the filters and the limits, and write the Clean_Splat into
    work/clean/. Only its path and the trained splat's SH_Bands come back, so every array made
    here is freed when this returns (Req 8.9).

    The source count is checked before the rows are read (Req 3.1), and the limits before
    anything is written (Req 3.7). Each count is added as soon as it's known, and the timing of
    each stage, read_ply, clean and write_ply, only once that stage is done. So a failure leaves
    exactly the counts and timings worked out before it in the Clean_Report (Req 7.2).
    """
    with step.timed("read_ply"):
        header = read_header(step.out_folder / s.train_ply)  # Req 1.1
        step.add_count("source_splats", header.count)
        check_source_count(header.count, s)  # Req 3.1, before the rows are read
        data = read_splat(header)
    with step.timed("clean"):
        kept = run_filters(data, s, step.add_count)
    warning = check_limits(header.count, kept, s)  # Req 3.2 to 3.7, before anything is written
    if warning is not None:
        step.add_warning(warning)
    ply = step.work_folder / s.clean_ply
    with step.timed("write_ply"):
        size = write_splat(ply, header, data, kept.rows)
    step.add_count("clean_ply_bytes", size)
    return Written(ply=ply, sh_bands=header.sh_bands)


def viewer_bands(step: StepContext, s: CleanSettings, sh_bands: int) -> int:
    """How many bands of spherical harmonics the Viewer_Copy keeps: sog_sh_bands, or the trained
    splat's `sh_bands` when it has fewer (Req 5.4).

    When that lowers sog_sh_bands, the step adds a warning that names both numbers to the
    Clean_Report and goes on, rather than fail (Req 5.5, 7.5).
    """
    bands = min(s.sog_sh_bands, sh_bands)
    if bands < s.sog_sh_bands:
        step.add_warning(f"The {step.preset} preset keeps {s.sog_sh_bands} bands of spherical "
                         "harmonics in the viewer copy, but the trained splat has only "
                         f"{sh_bands}, so the viewer copy keeps {sh_bands}.")
    return bands


def remove_outputs(viewer_copy: Path, clean_splat: Path) -> None:
    """Remove the earlier Viewer_Copy, then the earlier Clean_Splat, from out/ (Req 4.8).

    A file that isn't there is fine. The Clean_Splat is tried only once the Viewer_Copy is gone,
    so out/ never holds an old Viewer_Copy without the Clean_Splat it was made from (Req 4.13).
    An OSError, like a file another program holds open on Windows, is an ExpectedFailure that
    names the file and the reason, and it leaves the files not yet removed (Req 4.9).
    """
    for path in (viewer_copy, clean_splat):
        try:
            path.unlink(missing_ok=True)
        except OSError as error:
            raise ExpectedFailure(
                f"The clean step couldn't remove the earlier {path} "
                f"({error.strerror or error}). Close the program that has that file open, like "
                "a splat viewer, then run the clean step again.") from error


def move_outputs(moves: Sequence[tuple[Path, Path]]) -> None:
    """Move each (source, target) pair into out/ with os.replace, in the order given: the
    Clean_Splat, then the Viewer_Copy (Req 4.11).

    Each move is a rename inside the Job_Folder's one file system, so it's atomic on its own, and
    out/ never holds a half-written file (Req 4.5, 5.9). The moves together aren't, so on any
    BaseException, Ctrl+C included, the files already moved into out/ are removed again and the
    exception goes on up. An OSError goes up as an ExpectedFailure naming the file and the
    reason (Req 4.10).
    """
    moved: list[Path] = []
    try:
        for source, target in moves:
            moved.append(target)  # before the rename, so a Ctrl+C straight after it still sees it
            _move(source, target)
    except BaseException:
        _take_back(moved)
        raise


def _move(source: Path, target: Path) -> None:
    """Rename `source` to `target`. An OSError is an ExpectedFailure naming both and the reason."""
    try:
        os.replace(source, target)
    except OSError as error:
        raise ExpectedFailure(
            f"The clean step couldn't move {source} to {target} ({error.strerror or error}). "
            f"Close any program that has {target.name} open, then run the clean step "
            "again.") from error


def _take_back(moved: list[Path]) -> None:
    """Remove the files in `moved` from out/, the last one first, after a move into out/ failed.

    A file whose move didn't happen isn't there, because remove_outputs removed it, so removing
    it does nothing. A file that can't be removed stops this and stays, with the files moved
    before it, so a Viewer_Copy never stays without its Clean_Splat (Req 4.13). The exception
    that stopped the moves still goes up, never this one, so Ctrl+C still stops the run.
    """
    for target in reversed(moved):
        try:
            target.unlink(missing_ok=True)
        except OSError:
            return
