"""The Train_Step, which trains the Job's 3D Gaussian splat from the camera solve's files.

The step reads the Camera_Solve_Output in work/cameras/, trains the splat on the job machine's
GPU with gsplat 1.5.3, and writes it to out/train.ply, where the Clean_Step reads it. Its
settings are in [train] in config.toml, and settings.py reads and checks them.

The package is built like the Fake_Step. `work(step)` is the step's own code, which the
Step_Runner calls once it has emptied work/train/ (Req 11.8). `run(job_id, preset=None, *,
config=None)` runs the step through the Step_Runner and returns its Step_Report. `__main__.py`
is the CLI, `uv run python -m splatpipe.train --job <id> [--preset <name>]` (Req 11.1, 11.2).

`work` goes in the order of A9 (Req 11.3):

1. It removes out/train.ply, first of all, so a run that fails leaves no splat from an earlier
   run in out/ (Req 8.1).
2. The settings, every one checked (Req 9.6–9.8).
3. The Cap_Deadline from SPLATPIPE_CAP_DEADLINE, and the Stop_Time when it's set (Req 7.1–7.5).
4. check_inputs: images/ and the three model files must be there. Nothing is read (Req 1.2).
5. The hook GPU gives the GPU_Module, whose probe() gives the facts check_gpu decides on
   (Req 2.1–2.10). Nothing is read and nothing goes onto the GPU before this.
6. Timed as load: the Sparse_Model, the Training_Size, the Training_Images and the Mask_Files,
   the scene, the points the Start_Splats start from, with the Req 3.11 warning when they're a
   subset, the 3-NN search on the GPU and the Start_Splats (Req 1.3–1.19, 3.1–3.15).
7. Timed as train: the Trainer (Req 4.17). With no Iteration run, the step fails with the
   Req 7.9 message, and with fewer than steps, it adds the Req 7.8 warning (Req 7.7–7.9).
8. Timed as write_ply: the Mean_Exposure is baked in (Req 6.4–6.7), the splats with a NaN or an
   infinity are dropped, with the Req 8.9 warning giving how many (D6), and the rest are written
   to work/train/train.ply (Req 8.2–8.5).
9. Timed as check_ply: the PLY_Check (Req 8.6, 8.7). When every splat was dropped, the file holds
   no splats and the check fails the run, with the warning already in the report.
10. One os.replace moves the checked file to out/train.ply (Req 8.8).
11. The Train_Record, work/train/train.json, last of all (Req 10.7).

A GpuOutOfMemory from the 3-NN search or the Trainer becomes the Req 2.12 failure. The step adds
each count as an int as soon as it's known, and each timing when its block ends without an error,
so a failed report holds what the step had worked out before it failed (Req 10.1–10.4). It writes
only work/train/train.ply, work/train/train.json and out/train.ply (Req 8.11).

`work` reads the module constant GPU each time it runs, so a test that replaces it with a
GPU_Stand_In through monkeypatch reaches the step through run, run_step, the CLI and run_job
(Req 11.6). GPU is load_gpu, which imports the GPU_Module only then.

Only the GPU_Module, gpu.py, may import torch, gsplat and fused-ssim. Every other module of the
package imports only the standard library, numpy, cv2 and splatpipe, so importing splatpipe.train
on a laptop imports no GPU package (Req 9.11, 9.12). From splatpipe, this module imports only
splatpipe.errors, splatpipe.step and the package's CPU modules.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from splatpipe.errors import ExpectedFailure
# Config comes through splatpipe.step, as for the Fake_Step.
from splatpipe.step import Config, StepContext, run_step
from splatpipe.train.colmap import Camera, SparseModel, read_model
from splatpipe.train.deadline import read_cap_deadline, stop_time
from splatpipe.train.files import (CAMERAS, RECORD, SPARSE, TRAINED_SPLAT, check_inputs,
                                   move_into_out, read_views, train_record, write_record)
from splatpipe.train.gpu_api import (Gpu, GpuOutOfMemory, TrainResult, check_gpu, load_gpu,
                                     out_of_memory_failure)
from splatpipe.train.ply import check_ply, write_ply
from splatpipe.train.scene import Scene, build_scene, training_size
from splatpipe.train.settings import TrainSettings, read_settings
from splatpipe.train.splats import (Splats, bake_exposure, choose_points, drop_non_finite,
                                    mean_exposure, need_points, start_splats)

# The hook to the GPU_Module (Req 11.6). work reads it each time it runs, so a test can
# monkeypatch it with a GPU_Stand_In.
GPU: Callable[[], Gpu] = load_gpu

# The Req 3.11 warning: the Sparse_Points outnumber the Splat_Budget.
_SUBSET = ("The camera solve gave {points:,} points, more than this preset's splat budget of "
           "{budget:,}, so training started from {budget:,} of them, chosen at random.")
# The Req 7.8 warning: the Trainer stopped at the Stop_Time, before the last Planned_Iteration.
_STOPPED_EARLY = ("Training stopped early, after {run:,} of {planned:,} iterations, to leave time "
                  "for the clean step before the job's time cap.")
# The Req 8.9 and 10.5 warning (D6): how many splats drop_non_finite left out.
_DROPPED = "Splats left out of out/train.ply because they held a NaN or an infinity: {count:,}."


def run(job_id: str, preset: str | None = None, *,
        config: Config | None = None) -> dict[str, Any]:
    """Run the train step on the Job `job_id` and return its Step_Report.

    `preset` replaces the Job_Record's Preset, and `config` defaults to load_config().
    """
    return run_step("train", "splatpipe.train", job_id, preset=preset, config=config)


def work(step: StepContext) -> None:
    """The train step's own code. The Step_Runner turns what it raises into a failed report."""
    trained = step.work_folder / TRAINED_SPLAT  # written and checked here, then moved to out/
    (step.out_folder / TRAINED_SPLAT).unlink(missing_ok=True)  # first of all (Req 8.1)
    cfg = read_settings(step.settings, step.name, step.preset)  # Req 9.6–9.8
    step.add_count("steps_planned", int(cfg.steps))
    cap_deadline = read_cap_deadline(os.environ)  # Req 7.1–7.4
    stop = None if cap_deadline is None else stop_time(
        cap_deadline, cfg.clean_reserve_minutes, cfg.save_margin_minutes)  # Req 7.5
    cameras = step.job.folder / "work" / CAMERAS
    check_inputs(cameras)  # Req 1.2, before the GPU checks (A9)
    gpu = _checked_gpu(step, cfg)
    with step.timed("load"):
        camera, scene, start = _load(step, gpu, cameras, cfg)
    with step.timed("train"):
        result = _train(step, gpu, scene, start, cfg, stop)
    _check_steps_run(step, cfg, result)
    with step.timed("write_ply"):
        a, b = _write(step, result, trained)
    with step.timed("check_ply"):
        check_ply(trained)  # Req 8.6, 8.7
    move_into_out(trained, step.out_folder / TRAINED_SPLAT)  # Req 8.8
    write_record(step.work_folder / RECORD, train_record(
        scene=scene, means_lr_start=cfg.means_lr * scene.extent,
        width_ratio=scene.width / camera.width, height_ratio=scene.height / camera.height,
        a=a, b=b, cap_deadline=cap_deadline, stop=stop))  # Req 10.7


def _checked_gpu(step: StepContext, cfg: TrainSettings) -> Gpu:
    """The GPU_Module, through the hook GPU, once the GPU passes check_gpu (Req 2.1–2.10).

    GPU is read here, when the step runs. load_gpu refuses a machine without torch, gsplat or
    fused-ssim (Req 2.2). gpu_memory_bytes is added only when torch sees CUDA.
    """
    gpu = GPU()
    info = gpu.probe()
    if info.cuda:
        step.add_count("gpu_memory_bytes", int(info.total_memory_bytes))
    check_gpu(info, cfg, step.preset)
    return gpu


def _load(step: StepContext, gpu: Gpu, cameras: Path,
          cfg: TrainSettings) -> tuple[Camera, Scene, Splats]:
    """Read the Camera_Solve_Output in `cameras`, work/cameras/, and set up the scene and the
    Start_Splats. Gives back the Shared_Camera, for the Train_Record's ratios, the Scene and the
    Start_Splats."""
    model = read_model(cameras / SPARSE)  # Req 1.3–1.11
    step.add_count("images", int(len(model.images)))
    step.add_count("sparse_points", int(len(model.points)))
    need_points(len(model.points), cfg.start_scale_neighbours)  # Req 1.18
    width, height = training_size(model.camera.width, model.camera.height, cfg.long_edge)
    step.add_count("image_width", int(width))
    step.add_count("image_height", int(height))
    pixels = read_views(cameras, model, width, height, cfg.mask_threshold)  # Req 1.12–1.17
    step.add_count("masked_images", int(sum(view.keep is not None for view in pixels)))
    scene = build_scene(model, pixels, width, height, cfg)  # Req 1.19, 1.20, 3.3, 3.5–3.10
    return model.camera, scene, _start_splats(step, gpu, model, cfg)


def _start_splats(step: StepContext, gpu: Gpu, model: SparseModel,
                  cfg: TrainSettings) -> Splats:
    """The Start_Splats, from every Sparse_Point or, with the Req 3.11 warning, from max_splats of
    them chosen at random, sized by the GPU's 3-NN search (Req 3.11–3.15)."""
    points, colours = choose_points(model.points, model.colours, cfg.max_splats, cfg.seed)
    if len(model.points) > cfg.max_splats:
        step.add_warning(_SUBSET.format(points=len(model.points), budget=cfg.max_splats))
    with _out_of_memory(step.preset, cfg):
        distances = gpu.nearest_distances(points, cfg.start_scale_neighbours,
                                          cfg.neighbour_chunk_pairs)
    start = start_splats(points, colours, distances, cfg)
    step.add_count("start_splats", int(len(start)))
    return start


def _train(step: StepContext, gpu: Gpu, scene: Scene, start: Splats, cfg: TrainSettings,
           stop: datetime | None) -> TrainResult:
    """Run the Trainer on the scene from the Start_Splats, starting no Iteration at or after the
    Stop_Time `stop`, or every Planned_Iteration for None (Req 4.17, 7.6)."""
    with _out_of_memory(step.preset, cfg):
        result = gpu.train(scene, start, cfg, stop)
    step.add_count("steps_run", int(result.steps_run))
    step.add_count("sh_degree", int(result.sh_degree))
    step.add_count("peak_gpu_memory_bytes", int(gpu.peak_memory_bytes()))  # Req 10.1
    return result


def _check_steps_run(step: StepContext, cfg: TrainSettings, result: TrainResult) -> None:
    """Fail with the Req 7.9 message when the Trainer ran no Iteration, and add the Req 7.8
    warning when it stopped before the last Planned_Iteration (Req 7.7–7.9)."""
    if result.steps_run == 0:
        minutes = cfg.clean_reserve_minutes + cfg.save_margin_minutes
        raise ExpectedFailure(
            "No time was left to train before the job's time cap. The step stops "
            f"clean_reserve_minutes plus save_margin_minutes in [{step.name}] before the cap, "
            f"{minutes:g} minutes now. Lower them, or run the job with a longer time cap.")
    if result.steps_run < cfg.steps:
        step.add_warning(_STOPPED_EARLY.format(run=result.steps_run, planned=cfg.steps))


def _write(step: StepContext, result: TrainResult,
           path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Bake the Mean_Exposure into the trained splats, drop those with a NaN or an infinity, and
    write the rest to `path`, work/train/train.ply. Gives back the Mean_Exposure, Ā and b̄.

    `splats` counts the splats written. When every splat was dropped, the file holds none, and
    the PLY_Check fails the run with the warning and its count already in the report (D6).
    """
    a, b = mean_exposure(result.exposures)  # Req 6.4
    kept, dropped = drop_non_finite(bake_exposure(result.splats, a, b))  # Req 6.5, 8.9
    if dropped > 0:
        step.add_warning(_DROPPED.format(count=dropped))  # Req 8.9, 10.5
    step.add_count("splats", int(len(kept)))
    write_ply(path, kept)  # Req 8.2–8.5, 8.10
    step.add_count("ply_bytes", int(path.stat().st_size))
    return a, b


@contextmanager
def _out_of_memory(preset: str, cfg: TrainSettings) -> Iterator[None]:
    """Turn a GpuOutOfMemory from the GPU_Module in the block into the Req 2.12 failure, which
    names the Iteration reached, or the scene's set-up, the Preset and its max_splats."""
    try:
        yield
    except GpuOutOfMemory as error:
        raise out_of_memory_failure(error, preset, cfg.max_splats) from error
