"""The Depth_Step, which makes depth priors for the Train_Step from the camera solve's frames.

The step runs MoGe-2 on each Frame of the Camera_Solve_Output in work/cameras/, fits each Frame's
inverse depth to the camera solve's own points, and writes the Frames that fit as Prior_Files,
then the Prior_Index, in work/depth/, where the Train_Step reads them. Its settings are in
[depth] in config.toml, and settings.py reads and checks them. It's an Optional_Step: when it
fails, the job goes on and trains without depth priors.

The package is built like the Fake_Step. `work(step)` is the step's own code, which the
Step_Runner calls once it has emptied work/depth/, so no Prior_File or Prior_Index from an
earlier run is left (Req 1.11). `run(job_id, preset=None, *, config=None)` runs the step through
the Step_Runner and returns its Step_Report. `__main__.py` is the CLI, `uv run python -m
splatpipe.depth --job <id> [--preset <name>]` (Req 1.1, 1.2).

`work` goes in the order of Req 1.4:

1. The settings, every one checked (Req 8.3, 8.6).
2. check_inputs: images/ and the three model files must be there. Nothing is read (Req 2.2).
3. The hook DEPTH_GPU gives the Depth_GPU_Module, which refuses a machine with no torch
   (Req 3.2).
4. check_machine: torch must see CUDA, the Weights_File must be a file, and MoGe must import,
   offline (Req 3.2, 3.3, 3.5–3.8). No GPU memory is checked (Req 3.4).
5. Timed as load_model: MoGe-2, loaded strictly from moge_weights (Req 3.9–3.13). A finally that
   starts just before the load ends the steps from here on.
6. The Sparse_Model, with the Train_Step's Model_Reader (Req 2.3–2.6), the Prior_Size, with the
   Training_Size rule and long_edge from [depth] (Req 4.1), and the Field_Of_View (Req 4.3).
7. Each Frame, in images.bin's order (Req 2.8, 4.5): timed into infer, it's decoded and scaled
   to the Prior_Size (Req 2.7, 4.2) and MoGe-2 runs on it (Req 4.4); timed into fit, its
   Fit_Points and its Depth_Fit (Req 5); and, for a Fitted_Frame, timed into write, its
   Prior_File (Req 7.1, 7.10).
8. The finally: release gives the GPU's memory back, with None when the load failed, done or
   failed, so the Train_Step that runs next in the same process gets it (Req 3.14). Then
   peak_gpu_memory_bytes (Req 9.1).
9. With no Fitted_Frame, the step fails with the Req 7.7 message, and work/depth/ is empty.
10. With some Frames left out, the Req 7.9 warning.
11. Timed into write: the Prior_Index, last of all, with the job's Metres_Per_Unit, through a
    temporary file and one rename (Req 6.2, 6.3, 7.3, 7.6).

The step adds each count as an int as soon as it's known. infer, fit and write are running sums
over the Frames, each given to the report again when one of its blocks ends without an error, so
a failed report holds exactly what the step had worked out before it failed (Req 9.2–9.4). A
failure before step 11 leaves no Prior_Index, and a failed index write removes its temporary
file, so a failed run never leaves one (Req 7.8). The step writes only work/depth/<stem>.npz,
work/depth/depth.json.tmp while the index is written, and work/depth/depth.json (Req 1.9).

`work` reads the module constant DEPTH_GPU each time it runs, so a test that replaces it with a
Depth_Stand_In through monkeypatch reaches the step through run, run_step, the CLI and run_job
(Req 1.8). DEPTH_GPU is load_depth_gpu, which imports the Depth_GPU_Module only then.

Only the Depth_GPU_Module, gpu.py, may import torch, MoGe and huggingface_hub. Every other module
of the package imports only the standard library, numpy, cv2 and splatpipe, so importing
splatpipe.depth on a laptop imports no GPU package (Req 3.15). From splatpipe, this module
imports only splatpipe.step, the Train_Step's CPU modules colmap, files, priors and scene, and
the package's CPU modules.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from splatpipe.depth.files import check_inputs, read_frame, write_prior_file, write_prior_index
from splatpipe.depth.fit import (FITTED, FitOutcome, depth_fit, field_of_view, fit_points,
                                 job_metres_per_unit, prior_arrays)
from splatpipe.depth.gpu_api import DepthGpu, check_machine, load_depth_gpu, load_moge, run_moge
from splatpipe.depth.report import Tally, Timings, left_out_warning, no_fit_failure
from splatpipe.depth.settings import DepthSettings, read_settings
# Config comes through splatpipe.step, as for the Fake_Step.
from splatpipe.step import Config, StepContext, run_step
from splatpipe.train.colmap import Camera, ModelImage, SparseModel, read_model
from splatpipe.train.files import CAMERAS, SPARSE
from splatpipe.train.priors import PriorEntry, PriorIndex
from splatpipe.train.scene import training_size

# The hook to the Depth_GPU_Module (Req 1.8). work reads it each time it runs, so a test can
# monkeypatch it with a Depth_Stand_In.
DEPTH_GPU: Callable[[], DepthGpu] = load_depth_gpu


@dataclass(frozen=True)
class _Written:
    """What the Frames' loop gives the Prior_Index: the Shared_Camera, the Prior_Size and the
    entry of each Prior_File written, in images.bin's order."""

    camera: Camera
    width: int  # the Prior_Size
    height: int
    entries: tuple[PriorEntry, ...]


def run(job_id: str, preset: str | None = None, *,
        config: Config | None = None) -> dict[str, Any]:
    """Run the depth step on the Job `job_id` and return its Step_Report.

    `preset` replaces the Job_Record's Preset, and `config` defaults to load_config().
    """
    return run_step("depth", "splatpipe.depth", job_id, preset=preset, config=config)


def work(step: StepContext) -> None:
    """The depth step's own code. The Step_Runner turns what it raises into a failed report."""
    cfg = read_settings(step.settings, step.name, step.preset)  # Req 8.3, 8.6
    cameras = step.job.folder / "work" / CAMERAS
    check_inputs(cameras)  # Req 2.2, before the GPU checks
    gpu = DEPTH_GPU()  # read now, so a test's monkeypatch is seen (Req 1.8, 3.2)
    check_machine(gpu, cfg)  # Req 3.2–3.5
    tally, timings = Tally(), Timings()
    handle: object | None = None
    try:  # from just before the load, so MoGe-2 is released done or failed (Req 3.14)
        with step.timed("load_model"):
            handle = load_moge(gpu, cfg)  # Req 3.9–3.13
        written = _make_priors(step, gpu, handle, cameras, cfg, tally, timings)
    finally:
        gpu.release(handle)  # None when the load failed
        step.add_count("peak_gpu_memory_bytes", int(gpu.peak_memory_bytes()))  # Req 9.1
    if tally.fitted == 0:
        raise no_fit_failure(tally, cfg)  # Req 7.7: no Prior_File was written
    if tally.fitted < tally.frames:
        step.add_warning(left_out_warning(tally))  # Req 7.9
    with _timed_into(step, timings, "write"):
        write_prior_index(step.work_folder, _index(written))  # last of all (Req 7.3, 7.6)


def _make_priors(step: StepContext, gpu: DepthGpu, handle: object, cameras: Path,
                 cfg: DepthSettings, tally: Tally, timings: Timings) -> _Written:
    """Read the Sparse_Model in `cameras`, work/cameras/, then run MoGe-2 on each Frame, fit it
    and write its Prior_File when it's a Fitted_Frame (Req 2.8, 4.5, 5, 7.1).

    `tally` counts the Frames by their fit, and `timings` sums infer, fit and write over them.
    """
    model, width, height = _open_model(step, cameras, cfg)
    fov = field_of_view(model.camera.width, model.camera.fx)  # Req 4.3
    entries: list[PriorEntry] = []
    for image in model.images:  # in images.bin's order
        with _timed_into(step, timings, "infer"):
            rgb = read_frame(cameras, image.name, model.camera, width, height)  # Req 2.7, 4.2
            output = run_moge(gpu, handle, rgb, fov, cfg, image.name)  # Req 3.13, 4.4
        with _timed_into(step, timings, "fit"):
            points = fit_points(model, image, output, width, height)
            outcome = depth_fit(points.targets, points.moge_values, cfg)
        _count_fit(step, tally, outcome.kind)
        if outcome.kind == FITTED:
            with _timed_into(step, timings, "write"):
                write_prior_file(step.work_folder, image.name, prior_arrays(output, outcome))
            entries.append(_entry(image, outcome))
    return _Written(model.camera, width, height, tuple(entries))


def _open_model(step: StepContext, cameras: Path,
                cfg: DepthSettings) -> tuple[SparseModel, int, int]:
    """The Sparse_Model in `cameras`, work/cameras/, and the Prior_Size, as (model, width,
    height), with images, prior_width and prior_height added (Req 2.3, 4.1, 9.1)."""
    model = read_model(cameras / SPARSE)
    step.add_count("images", int(len(model.images)))
    width, height = training_size(model.camera.width, model.camera.height, cfg.long_edge)
    step.add_count("prior_width", int(width))
    step.add_count("prior_height", int(height))
    return model, width, height


def _count_fit(step: StepContext, tally: Tally, kind: str) -> None:
    """Count one more Frame, whose Depth_Fit gave `kind`, and add the four counts of the Frames
    by their fit again, with their new values (Req 9.1, 9.3)."""
    tally.add(kind)
    step.add_count("fitted_images", int(tally.fitted))
    step.add_count("few_points_images", int(tally.few_points))
    step.add_count("flat_images", int(tally.flat))
    step.add_count("poor_fit_images", int(tally.poor_fit))


def _entry(image: ModelImage, outcome: FitOutcome) -> PriorEntry:
    """The Prior_Index's entry of the Fitted_Frame `image`, whose Depth_Fit gave `outcome`, with
    its Image_Pose exactly as images.bin gives it (Req 6.3, 7.3)."""
    fitted = (outcome.scale, outcome.offset, outcome.median_error, outcome.metres_per_unit)
    if any(value is None for value in fitted):
        raise ValueError(f"A prior entry needs a fitted Frame's outcome, not {outcome.kind!r}.")
    scale, offset, median_error, metres_per_unit = (float(value) for value in fitted)
    qw, qx, qy, qz = (float(value) for value in image.quaternion)
    tx, ty, tz = (float(value) for value in image.translation)
    return PriorEntry(name=image.name, quaternion=(qw, qx, qy, qz), translation=(tx, ty, tz),
                      scale=scale, offset=offset, fit_points=int(outcome.fit_points),
                      kept_points=int(outcome.kept_points), median_error=median_error,
                      metres_per_unit=metres_per_unit)


def _index(written: _Written) -> PriorIndex:
    """The Prior_Index of the Prior_Files `written`, one or more, with the job's Metres_Per_Unit,
    the median of the Fitted_Frames' (Req 6.2, 6.3, 7.3)."""
    camera = written.camera
    return PriorIndex(camera=(camera.width, camera.height, camera.fx, camera.fy, camera.cx,
                              camera.cy),
                      prior_width=written.width, prior_height=written.height,
                      metres_per_unit=job_metres_per_unit(written.entries),
                      priors=written.entries)


@contextmanager
def _timed_into(step: StepContext, timings: Timings, name: str) -> Iterator[None]:
    """Time the block with time.perf_counter and, when it ends without an error, add its seconds
    to the running sum `name`, which goes to the report (Req 9.2, 9.3)."""
    started = time.perf_counter()
    yield
    timings.add(step, name, time.perf_counter() - started)
