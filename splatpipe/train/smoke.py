"""The Train_Smoke_Check's flow on the CPU side: the Smoke_Scene, and the Trainer's run on it.

cloud/smoke_runner.py's `training` check turns [cloud.smoke]'s numbers into a SmokeSizes and calls
run_smoke_training with the GPU_Module that load_gpu gives. The Test_Suite calls it with the
GPU_Stand_In instead, so the whole check runs on a laptop with no torch (Req 12.2–12.10).

- SmokeSizes: the check's eight numbers from [cloud.smoke].
- smoke_settings (Req 12.3, A11): the [train] settings of the Preset that [jobs].default_preset
  names, with steps, max_splats, refine_start_step, refine_every and sh_degree_interval taken from
  the sizes, checked by the step's own read_settings.
- smoke_scene (Req 12.2): the Smoke_Scene, a Sparse_Model and its images, with everything random
  drawn from one generator seeded with [train]'s seed. One PINHOLE camera, image_pixels square, is
  shared by train_images cameras on the circle of radius 1 in the x–z plane, each looking at the
  origin. The train_points points are uniform in the cube of half-side 0.25 around the origin, so
  every point is 1 ± 0.25·√2, 0.65 to 1.35, in front of every camera. The points' ids are 1 to
  train_points, and no image has Observed_Points (depth-priors spec, Req 2.4), so nothing in the
  training check's run changes. The images are random RGB, and image 0's keep mask leaves out
  its left half, so one image has a mask and keeps pixels. The radius and the half-side are the
  scene's shape in its own unit, Fixed_Constants, as the smoke test's gsplat check puts its
  Gaussians one unit from its camera (Req 9.5).
- run_smoke_training (Req 12.3, 12.7–12.9): the GPU check, the step's own set-up of the scene and
  the Start_Splats, the Trainer, the Exposure_Bake, the PLY_Writer and the PLY_Check. It drops no
  splat, so a splat with a NaN or an infinity fails the PLY_Check, because the Smoke_Scene should
  never make one (D6). It passes only when the Trainer ran every Planned_Iteration, and gives
  back the check's line for the smoke test's log.

The module imports no torch: it reaches the GPU only through the Gpu it's given. From splatpipe,
it imports only splatpipe.config, splatpipe.errors and the package's CPU modules.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from splatpipe.config import Config, StepSettings
from splatpipe.errors import PipelineError
from splatpipe.train.colmap import Camera, ModelImage, SparseModel
from splatpipe.train.files import TRAINED_SPLAT
from splatpipe.train.gpu_api import (Gpu, GpuOutOfMemory, TrainResult, check_gpu,
                                     out_of_memory_failure)
from splatpipe.train.ply import check_ply, write_ply
from splatpipe.train.scene import ImagePixels, Scene, build_scene, training_size
from splatpipe.train.schedules import refine_iterations, refine_stop
from splatpipe.train.settings import TrainSettings, read_settings
from splatpipe.train.splats import (bake_exposure, choose_points, mean_exposure, need_points,
                                    start_splats)

_STEP = "train"  # the Train_Step's section in Config, whose settings the check trains with
# Fixed_Constants: the Smoke_Scene's shape in its own unit (Req 9.5). The cameras sit on a circle
# of _RADIUS around the origin, and the points in the cube of half-side _HALF_SIDE around it.
_RADIUS = 1.0
_HALF_SIDE = 0.25
_FIRST_ID = 1  # COLMAP numbers its cameras, its images and its points from 1


@dataclass(frozen=True)
class SmokeSizes:
    """The Train_Smoke_Check's numbers from [cloud.smoke], which cloud/smoke_runner.py passes in."""

    images: int  # train_images
    points: int  # train_points
    pixels: int  # image_pixels
    splats: int  # train_splats
    steps: int  # train_steps
    refine_start_step: int  # train_refine_start_step
    refine_every: int  # train_refine_every
    sh_degree_interval: int  # train_sh_degree_interval


def smoke_settings(config: Config, sizes: SmokeSizes) -> tuple[str, TrainSettings]:
    """The Preset that [jobs].default_preset names, and the settings the check trains with
    (Req 12.3, A11).

    They're that Preset's [train] settings with steps, max_splats, refine_start_step,
    refine_every and sh_degree_interval replaced by the sizes, checked by the step's own
    read_settings, so the check's numbers are checked as a Job's are.
    """
    preset = config.jobs().default_preset
    values = dict(config.step_settings(_STEP, preset))
    values.update(steps=sizes.steps, max_splats=sizes.splats,
                  refine_start_step=sizes.refine_start_step, refine_every=sizes.refine_every,
                  sh_degree_interval=sizes.sh_degree_interval)
    settings = StepSettings(values, section=_STEP, preset=preset)
    return preset, read_settings(settings, _STEP, preset)


def smoke_scene(sizes: SmokeSizes, seed: int) -> tuple[SparseModel, list[ImagePixels]]:
    """The Smoke_Scene (Req 12.2): its Sparse_Model, and each image's ImagePixels in the model's
    order, as files.read_views gives them for a Job.

    Everything random comes from one np.random.default_rng(seed): the points, uniform in
    [−_HALF_SIDE, _HALF_SIDE]³, then their 8-bit colours, then each image's RGB pixels. The
    Shared_Camera is PINHOLE and `pixels` square, with fx = fy = `pixels`, so each image is as
    wide as it is far, and its principal point at the centre. Camera j of `images` looks at the
    origin from _RADIUS · (sin θ, 0, −cos θ), with θ = 2πj / `images`. Image 0's keep mask leaves
    out its left `pixels` // 2 columns, and the other images have no mask. With 2 or more images,
    the cameras' mean is the origin, so the Scene_Extent is scene_extent_factor × _RADIUS.

    The points' ids are 1 to `points` as uint64, and every image's observed_ids is empty, so no
    point rules a pixel out (depth-priors spec, Req 2.4). Neither draws from the generator.
    """
    generator = np.random.default_rng(seed)
    side = sizes.pixels
    camera = Camera(_FIRST_ID, side, side, fx=float(side), fy=float(side), cx=side / 2,
                    cy=side / 2)
    images = tuple(_camera_on_circle(index, sizes.images) for index in range(sizes.images))
    points = generator.uniform(-_HALF_SIDE, _HALF_SIDE, size=(sizes.points, 3))
    colours = _random_bytes(generator, (sizes.points, 3))
    pixels = [ImagePixels(_random_bytes(generator, (side, side, 3)), None) for _ in images]
    keep = np.ones((side, side), dtype=bool)
    keep[:, :side // 2] = False  # image 0's left half is masked, and its right half kept
    pixels[0] = ImagePixels(pixels[0].pixels, keep)
    point_ids = np.arange(_FIRST_ID, _FIRST_ID + sizes.points, dtype=np.uint64)
    return SparseModel(camera, images, points, colours, point_ids), pixels


def run_smoke_training(config: Config, sizes: SmokeSizes, scratch: Path, gpu: Gpu) -> str:
    """Train the Smoke_Scene with `gpu`'s Trainer and check the splat it gives (Req 12.3,
    12.7–12.9). Gives back the check's line for the smoke test's log.

    Once the settings are read, the GPU check comes first, for the default Preset, so a wrong
    gsplat or a GPU that torch can't see fails the smoke test and not the first Job. Then come
    the step's own need_points, training_size, build_scene, choose_points, 3-NN search and
    start_splats, and the Trainer with no Stop_Time. The Mean_Exposure is baked in and every
    splat the Trainer gave back is written to scratch/train.ply, none dropped, so a splat with a
    NaN or an infinity fails the PLY_Check (Req 12.7, D6). Any failure goes up with its own
    message: a PipelineError from a check, or the Trainer's NonFiniteLoss.
    """
    preset, settings = smoke_settings(config, sizes)
    check_gpu(gpu.probe(), settings, preset)  # Req 2.3–2.10, before anything goes on the GPU
    model, pixels = smoke_scene(sizes, settings.seed)
    need_points(len(model.points), settings.start_scale_neighbours)  # Req 1.18
    width, height = training_size(model.camera.width, model.camera.height, settings.long_edge)
    scene = build_scene(model, pixels, width, height, settings)
    result = _train(gpu, model, scene, settings, preset)
    path = scratch / TRAINED_SPLAT
    a, b = mean_exposure(result.exposures)
    write_ply(path, bake_exposure(result.splats, a, b))  # every splat, none dropped (Req 12.7)
    return _log_line(result, check_ply(path), settings)


def _train(gpu: Gpu, model: SparseModel, scene: Scene, settings: TrainSettings,
           preset: str) -> TrainResult:
    """The Start_Splats from the model's points, sized by `gpu`'s 3-NN search, then trained on
    `scene` by its Trainer with no Stop_Time, once it ran every Planned_Iteration (Req 12.8).

    A GpuOutOfMemory from either GPU call becomes the step's Req 2.12 failure, which names the
    Preset and its max_splats. A run of fewer Iterations than steps is refused with how many ran.
    """
    points, colours = choose_points(model.points, model.colours, settings.max_splats,
                                    settings.seed)
    try:
        distances = gpu.nearest_distances(points, settings.start_scale_neighbours,
                                          settings.neighbour_chunk_pairs)
        start = start_splats(points, colours, distances, settings)
        result = gpu.train(scene, start, settings, None)
    except GpuOutOfMemory as error:
        raise out_of_memory_failure(error, preset, settings.max_splats) from error
    if result.steps_run < settings.steps:
        raise PipelineError(
            f"The Trainer ran only {result.steps_run:,} of the training check's "
            f"{settings.steps:,} iterations, though the check gives it no stop time, so it should "
            "run them all. Check the Trainer in splatpipe/train/gpu.py, then run the smoke test "
            "again.")
    return result


def _log_line(result: TrainResult, splats: int, settings: TrainSettings) -> str:
    """The check's line for the log: the Iterations run, the MCMC refines among them, the
    `splats` written and the SH_Degree reached (Req 12.8).

    The refines are counted with refine_iterations, the condition gsplat's MCMCStrategy checks,
    up to the Refine_Stop that the Trainer gives the strategy.
    """
    stop = refine_stop(settings.steps, settings.refine_stop_step, settings.refine_stop_fraction)
    refines = len(refine_iterations(settings.steps, settings.refine_start_step,
                                    settings.refine_every, stop))
    return (f"trained {_counted(result.steps_run, 'iteration')}, with "
            f"{_counted(refines, 'refine')}, to {_counted(splats, 'splat')} at SH degree "
            f"{result.sh_degree}, and the PLY check passed")


def _camera_on_circle(index: int, count: int) -> ModelImage:
    """Image `index` of `count` in the Smoke_Scene, whose camera looks at the origin from
    _RADIUS · (sin θ, 0, −cos θ), on the circle in the x–z plane, with θ = 2π · index / count.

    Its pose turns the world by θ about y, the unit quaternion (cos θ/2, 0, sin θ/2, 0), then
    moves it _RADIUS along the camera's z, so the origin sits _RADIUS in front of the camera, at
    the image's centre. Its Camera_Centre, −Rᵀt, is then that point on the circle. It has no
    Observed_Points, so its observed_ids is an empty uint64 array.
    """
    theta = 2 * math.pi * index / count
    quaternion = np.array([math.cos(theta / 2), 0.0, math.sin(theta / 2), 0.0])
    translation = np.array([0.0, 0.0, _RADIUS])
    return ModelImage(_FIRST_ID + index, quaternion, translation, _FIRST_ID,
                      f"smoke_{index}.jpg", np.empty(0, dtype=np.uint64))


def _random_bytes(generator: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    """A uint8 array of `shape`, each value drawn by `generator` from 0 to 255, evenly."""
    return generator.integers(0, 256, size=shape, dtype=np.uint8)


def _counted(count: int, noun: str) -> str:
    """A count and its noun, like "1 refine" or "30 iterations"."""
    return f"{count:,} {noun}" if count == 1 else f"{count:,} {noun}s"
