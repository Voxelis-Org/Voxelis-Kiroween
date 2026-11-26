"""The Depth_Smoke_Check's flow on the CPU side: MoGe-2 on a made-up image, the Prior_Smoke_Scene,
and the Trainer's run on it with the Prior_Terms (depth-priors spec, Req 16).

cloud/smoke_runner.py's `depth` check turns [cloud.smoke]'s numbers into a DepthSmokeSizes and
calls run_smoke_depth with the Depth_GPU_Module that load_depth_gpu gives and the GPU_Module that
load_gpu gives. The Test_Suite calls it with the Depth_Stand_In and the GPU_Stand_In instead, so
the whole check runs on a laptop with no torch, MoGe or huggingface_hub.

- DepthSmokeSizes: the Train_Smoke_Check's eight numbers from [cloud.smoke], and its
  depth_normal_start_step, the Normal_Start the check trains with (A5).
- check_moge_output (Req 16.3): MoGe-2's output on the made-up image passes when depth, normal
  and mask are (p, p) float32, (p, p, 3) float32 and (p, p) bool, and depth and normal are finite
  and depth above 0 wherever mask is true. It gives back the number of Geometry_Pixels.
- prior_smoke_scene (Req 16.4, A5): the Prior_Smoke_Scene. It's the training check's Smoke_Scene,
  its cameras, points and images, with each image's rows from p // 2 down painted one grey, so
  each image holds a plain area. Every image gets a made-up Prior at the Prior_Size, p × p, the
  camera's own size, because p is below [depth].long_edge: inverse_depth 1 everywhere, the plane
  through the origin square to the camera, which sits 1 in front of each camera; normal
  (0, 0, −1) everywhere, facing the camera; and mask true everywhere. The Priors and their
  Prior_Index go into the given folder through the Prior_Writer, and come back through the
  Train_Step's own Prior_Reader and checks. The Smoke_Scene has no Observed_Points, so no point
  rules a pixel out of the Back_Projected_Splats.
- run_smoke_depth (Req 16.2–16.9): MoGe's part, with the Depth_Step's own read_settings,
  check_machine, load_moge and run_moge, on a p × p image from np.random.default_rng([train]'s
  seed), whose made-up PINHOLE camera has W = fx = p, and release in a finally. Then the
  Prior_Smoke_Scene in scratch/depth/, the Train_Step's GPU check, and the step's own
  build_scene with the Prior_Set, start_points, the 3-NN search over every Start_Splat,
  start_splats and the Trainer with no Stop_Time, trained with the training check's settings
  except that depth_normal_start_step sets normal_start_step. The Mean_Exposure is baked in,
  every splat is written to scratch/train.ply, none dropped, and the PLY_Check runs. It passes
  only when the Trainer ran every Planned_Iteration and at least one Back_Projected_Splat
  started, and gives back the check's line for the smoke test's log.

Every failure raises with its own message, so the Smoke_Test reports the check as failed and goes
on with the others (Req 16.9): a PipelineError from a check here or from the steps' own checks,
which includes the Depth_Step's Req 3.2 failure on an image with no MoGe code (A19), or the
Trainer's NonFiniteLoss.

The made-up Prior's values, the grey of the plain area, the made-up image's name and its
camera's W = fx are the Prior_Smoke_Scene's shape in its own unit, Fixed_Constants, as the
training check's _RADIUS and _HALF_SIDE are. Every other number comes from config.toml.

The module imports no torch: it reaches the GPUs only through the DepthGpu and the Gpu it's
given. It imports only the standard library, numpy and splatpipe's CPU modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from splatpipe.config import Config
from splatpipe.depth.fit import field_of_view, geometry_pixels
from splatpipe.depth.gpu_api import DepthGpu, MoGeOutput, check_machine, load_moge, run_moge
from splatpipe.depth.settings import read_settings
from splatpipe.errors import PipelineError
from splatpipe.train.backproject import StartPoints, start_points
from splatpipe.train.colmap import ModelImage, SparseModel
from splatpipe.train.files import TRAINED_SPLAT
from splatpipe.train.gpu_api import (Gpu, GpuOutOfMemory, TrainResult, check_gpu,
                                     out_of_memory_failure)
from splatpipe.train.ply import check_ply, write_ply
from splatpipe.train.priors import (PRIOR_FOLDER, BrokenPriors, Prior, PriorEntry, PriorIndex,
                                    PriorSet, prior_path, read_prior_set, write_index,
                                    write_prior)
from splatpipe.train.scene import ImagePixels, Scene, build_scene, training_size
from splatpipe.train.settings import TrainSettings
from splatpipe.train.smoke import SmokeSizes, smoke_scene, smoke_settings
from splatpipe.train.splats import bake_exposure, mean_exposure, need_points, start_splats

_DEPTH_STEP = "depth"  # the Depth_Step's section in Config, whose settings MoGe's part reads
# Fixed_Constants: the Prior_Smoke_Scene's shape in its own unit (Req 16.4, A5).
SMOKE_FRAME = "smoke_moge.jpg"  # the made-up image's name, which run_moge's messages give
_PLAIN_GREY = 128  # the one grey, in each 8-bit RGB channel, of each image's plain lower half
# The made-up Prior: the plane square to the camera through the origin, which the Smoke_Scene's
# cameras look at from 1 away, so its inverse depth is 1, and its normal faces the camera.
_INVERSE_DEPTH = 1.0
_NORMAL = (0.0, 0.0, -1.0)
# The made-up Prior_Index entry's fit, which no Depth_Fit made: the identity scale and offset,
# no Fit_Points, no error and one metre per unit.
_SCALE = 1.0
_OFFSET = 0.0
_FIT_POINTS = 0
_MEDIAN_ERROR = 0.0
_METRES_PER_UNIT = 1.0
# What a failed check says to do next.
_CHECK_MOGE = ("Check the depth step's MoGe-2 call in splatpipe/depth/gpu.py and the image "
               "build's MoGe step, then run the smoke test again.")
_RUN_AGAIN = "then run the smoke test again."


@dataclass(frozen=True)
class DepthSmokeSizes:
    """The Depth_Smoke_Check's numbers from [cloud.smoke], which cloud/smoke_runner.py passes in."""

    train: SmokeSizes  # the Train_Smoke_Check's eight numbers
    normal_start: int  # depth_normal_start_step, the Normal_Start the check trains with (A5)


def check_moge_output(output: MoGeOutput, side: int) -> int:
    """Refuse MoGe-2's `output` on the made-up `side` × `side` image unless its arrays are the
    image's size with Req 4.6's dtypes, and depth and normal are finite and depth above 0
    wherever mask is true (Req 16.3). Gives back the number of Geometry_Pixels.

    Each refusal is a PipelineError naming the array and what was wrong with it.
    """
    wanted = (("depth", (side, side), np.dtype(np.float32)),
              ("normal", (side, side, 3), np.dtype(np.float32)),
              ("mask", (side, side), np.dtype(np.bool_)))
    for name, shape, dtype in wanted:
        array = getattr(output, name)
        if array.shape != shape:
            raise PipelineError(f"MoGe-2's {name} is {array.shape}, but the image is {side} x "
                                f"{side}, so it must be {shape}. {_CHECK_MOGE}")
        if array.dtype != dtype:
            raise PipelineError(f"MoGe-2's {name} is {array.dtype}, but it must be {dtype}. "
                                f"{_CHECK_MOGE}")
    marked = output.mask
    for name in ("depth", "normal"):
        values = getattr(output, name)[marked]
        bad = values[~np.isfinite(values)]
        if bad.size:
            kind = "a NaN" if np.isnan(bad[0]) else "an infinity"
            raise PipelineError(f"MoGe-2's {name} holds {kind} where its mask is true, but it "
                                f"must be finite there. {_CHECK_MOGE}")
    low = output.depth[marked & ~(output.depth > 0)]
    if low.size:
        raise PipelineError(f"MoGe-2's depth holds {float(low[0])} where its mask is true, but it "
                            f"must be above 0 there. {_CHECK_MOGE}")
    return int(np.count_nonzero(geometry_pixels(output)))


def prior_smoke_scene(sizes: SmokeSizes, seed: int,
                      folder: Path) -> tuple[SparseModel, list[ImagePixels], PriorSet]:
    """The Prior_Smoke_Scene (Req 16.4, A5): the Smoke_Scene's Sparse_Model, its images with a
    plain grey lower half, in the model's order, and the Prior_Set read back from `folder`.

    It writes one made-up Prior_File per image and then the Prior_Index into `folder`, made if
    it's missing, with the Prior_Writer, and reads them back with read_prior_set, the Train_Step's
    own reader and checks. Priors that fail a check are a PipelineError with the reader's message.
    Everything here is numpy, with no torch.
    """
    model, pixels = smoke_scene(sizes, seed)
    painted = [_plain_lower_half(image) for image in pixels]
    width, height = model.camera.width, model.camera.height  # the Prior_Size, p × p
    folder.mkdir(parents=True, exist_ok=True)
    for image in model.images:
        write_prior(prior_path(folder, image.name), _made_up_prior(width, height))
    write_index(folder, _made_up_index(model, width, height))
    try:
        priors = read_prior_set(folder, model)
    except BrokenPriors as error:
        raise PipelineError(f"The depth check's made-up priors failed the train step's own "
                            f"checks: {error}. Check the prior format in "
                            f"splatpipe/train/priors.py, {_RUN_AGAIN}") from error
    if priors is None:
        raise PipelineError(f"The depth check wrote its made-up priors to {folder}, but the "
                            "train step's reader found no prior index there. Check the prior "
                            f"format in splatpipe/train/priors.py, {_RUN_AGAIN}")
    return model, painted, priors


def run_smoke_depth(config: Config, sizes: DepthSmokeSizes, scratch: Path,
                    depth_gpu: DepthGpu, gpu: Gpu) -> str:
    """Run MoGe-2 on a made-up image, then train the Prior_Smoke_Scene with its Priors and check
    the splat it gives (Req 16.2–16.9). Gives back the check's line for the smoke test's log.

    The [train] settings are the training check's, with normal_start_step from `sizes`, and
    they're read first, because MoGe's part draws its image from their seed. MoGe's part runs
    next, with the [depth] settings of the same Preset. The Priors go into scratch/depth/, and
    the splat into scratch/train.ply. Before anything trains, the Train_Step's GPU check runs,
    as in the training check, so a wrong gsplat fails here too.
    """
    preset, settings = smoke_settings(config, sizes.train, normal_start_step=sizes.normal_start)
    side = sizes.train.pixels
    geometry = _run_moge_part(config, preset, side, settings.seed, depth_gpu)
    model, pixels, priors = prior_smoke_scene(sizes.train, settings.seed, scratch / PRIOR_FOLDER)
    check_gpu(gpu.probe(), settings, preset)  # Req 2.3–2.10 of the splat-training spec
    need_points(len(model.points), settings.start_scale_neighbours)
    width, height = training_size(model.camera.width, model.camera.height, settings.long_edge)
    scene = build_scene(model, pixels, width, height, settings, priors)
    start, result = _train(gpu, model, scene, settings, preset)
    path = scratch / TRAINED_SPLAT
    a, b = mean_exposure(result.exposures)
    write_ply(path, bake_exposure(result.splats, a, b))  # every splat, none dropped (Req 16.5)
    splats = check_ply(path)
    return (f"MoGe-2 found geometry at {geometry:,} of {side * side:,} pixels; trained "
            f"{_counted(result.steps_run, 'iteration')} with depth priors, from "
            f"{start.back_projected:,} back-projected and {start.sparse:,} sparse splats, to "
            f"{_counted(splats, 'splat')}, and the PLY check passed")


def _run_moge_part(config: Config, preset: str, side: int, seed: int,
                   depth_gpu: DepthGpu) -> int:
    """MoGe's part of the check (Req 16.2, 16.3): the Depth_Step's own settings, machine check,
    load and call on a `side` × `side` RGB image drawn from np.random.default_rng(seed), then
    check_moge_output. Gives back the number of Geometry_Pixels.

    The image's made-up PINHOLE camera has W = fx = `side`, so its Field_Of_View is
    field_of_view(side, side), about 53.1°. MoGe-2 is released in a finally, done or failed, with
    None when the load failed, as the Depth_Step releases it (Req 3.14).
    """
    depth_settings = read_settings(config.step_settings(_DEPTH_STEP, preset), _DEPTH_STEP,
                                   preset)
    check_machine(depth_gpu, depth_settings)  # Req 3.2–3.5
    rgb = np.random.default_rng(seed).integers(0, 256, (side, side, 3), dtype=np.uint8)
    fov_x = field_of_view(side, float(side))
    handle: object | None = None
    try:
        handle = load_moge(depth_gpu, depth_settings)
        output = run_moge(depth_gpu, handle, rgb, fov_x, depth_settings, SMOKE_FRAME)
    finally:
        depth_gpu.release(handle)
    return check_moge_output(output, side)


def _train(gpu: Gpu, model: SparseModel, scene: Scene, settings: TrainSettings,
           preset: str) -> tuple[StartPoints, TrainResult]:
    """The Start_Splats from the Sparse_Points and the Back_Projected_Splats, sized together by
    `gpu`'s 3-NN search, then trained on `scene` by its Trainer with no Stop_Time (Req 16.5,
    16.8). Gives back the start points, for their counts, and the Trainer's result.

    No Back_Projected_Splat is refused before anything goes on the GPU. A GpuOutOfMemory from
    either GPU call becomes the Train_Step's Req 2.12 failure, which names the Preset and its
    max_splats. A run of fewer Iterations than steps is refused with how many ran.
    """
    start = start_points(model, scene, settings)
    if start.back_projected == 0:
        raise PipelineError(
            "No splat started from the depth check's made-up depth priors, though every image "
            "has a prior and a plain grey lower half. Check that backproject_share and "
            "backproject_max in [train] leave room for some, and start_points in "
            f"splatpipe/train/backproject.py, {_RUN_AGAIN}")
    try:
        distances = gpu.nearest_distances(start.points, settings.start_scale_neighbours,
                                          settings.neighbour_chunk_pairs)
        splats = start_splats(start.points, start.colours, distances, settings)
        result = gpu.train(scene, splats, settings, None)
    except GpuOutOfMemory as error:
        raise out_of_memory_failure(error, preset, settings.max_splats) from error
    if result.steps_run < settings.steps:
        raise PipelineError(
            f"The Trainer ran only {result.steps_run:,} of the depth check's "
            f"{settings.steps:,} iterations, though the check gives it no stop time, so it should "
            f"run them all. Check the Trainer in splatpipe/train/gpu.py, {_RUN_AGAIN}")
    return start, result


def _plain_lower_half(image: ImagePixels) -> ImagePixels:
    """`image` with its rows from height // 2 down painted _PLAIN_GREY, so it holds a plain area,
    and its keep mask as it was. The pixels are a copy, so the Smoke_Scene's stay as drawn."""
    pixels = image.pixels.copy()
    pixels[pixels.shape[0] // 2:] = _PLAIN_GREY
    return ImagePixels(pixels, image.keep)


def _made_up_prior(width: int, height: int) -> Prior:
    """The made-up Prior at the Prior_Size `width` × `height`: inverse_depth _INVERSE_DEPTH,
    normal _NORMAL and mask true at every pixel."""
    normal = np.empty((height, width, 3), dtype=np.float32)
    normal[...] = _NORMAL
    return Prior(inverse_depth=np.full((height, width), _INVERSE_DEPTH, dtype=np.float32),
                 normal=normal, mask=np.ones((height, width), dtype=bool))


def _made_up_index(model: SparseModel, width: int, height: int) -> PriorIndex:
    """The Prior_Index of the made-up Priors: the Shared_Camera and every image's pose as the
    model gives them, so the reader's checks pass, at the Prior_Size `width` × `height`."""
    camera = model.camera
    return PriorIndex(camera=(camera.width, camera.height, camera.fx, camera.fy, camera.cx,
                              camera.cy),
                      prior_width=width, prior_height=height,
                      metres_per_unit=_METRES_PER_UNIT,
                      priors=tuple(_made_up_entry(image) for image in model.images))


def _made_up_entry(image: ModelImage) -> PriorEntry:
    """The made-up Prior_Index entry of `image`, with its Image_Pose exactly as the model gives
    it and the made-up fit."""
    qw, qx, qy, qz = (float(value) for value in image.quaternion)
    tx, ty, tz = (float(value) for value in image.translation)
    return PriorEntry(name=image.name, quaternion=(qw, qx, qy, qz), translation=(tx, ty, tz),
                      scale=_SCALE, offset=_OFFSET, fit_points=_FIT_POINTS,
                      kept_points=_FIT_POINTS, median_error=_MEDIAN_ERROR,
                      metres_per_unit=_METRES_PER_UNIT)


def _counted(count: int, noun: str) -> str:
    """A count and its noun, like "1 splat" or "30 iterations"."""
    return f"{count:,} {noun}" if count == 1 else f"{count:,} {noun}s"
