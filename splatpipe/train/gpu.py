# Adapted from gsplat v1.5.3, examples/simple_trainer.py
# https://github.com/nerfstudio-project/gsplat/blob/v1.5.3/examples/simple_trainer.py
#
# Copyright 2025 Nerfstudio Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Changes: cut down to the mcmc preset's training loop on one GPU, with no viewer, evaluation,
# held-out images, pose optimisation, bilateral grid, random background, checkpoints or batch
# scaling. Added: a per-image exposure fix, masked L1 and SSIM losses, the effective-rank term,
# the depth-prior losses and the normal render, a stop time, and our own numbers from
# config.toml.
"""The GPU_Module: the GPU check's facts, the 3-NN search and the Trainer, on torch and gsplat.

This is the one module of the Train_Step that imports torch, gsplat and fused_ssim (Req 9.11).
The step reaches it only through gpu_api.load_gpu, which imports it by name when the step runs,
and its four functions make up gpu_api's Gpu protocol. What they give back is numpy arrays and
plain values, never a torch tensor.

- probe (Req 2.4, 10.1): whether torch sees CUDA, gsplat's version and the GPU's total memory in
  bytes, the facts gpu_api.check_gpu decides on. With CUDA it also resets torch's peak memory
  count, so peak_memory_bytes covers the step's own GPU work and not an earlier step's.
- nearest_distances (Req 3.13): each point's distances to its nearest other points, from
  torch.cdist's difference form on the GPU, one chunk of query points at a time.
- train (Req 4.17): the Trainer, our own loop on gsplat's library API, rasterization() and
  MCMCStrategy, set up and ordered as gsplat's example trainer is. It holds no state between
  calls. With a Prior_Set it also renders the Rendered_Depth and the Normal_Render and adds the
  five Prior_Terms to its loss (depth-priors spec, Req 13).
- peak_memory_bytes (Req 10.1): the most GPU memory torch held at once since probe().

The Trainer sets up, in this order: torch's seed; the views on the GPU, each Prior_View with its
Prior at the Prior_Size and its Brightness_Gradient (depth-priors spec, Req 10.5); with a
Prior_Set, the taps, the rays and the Depth_Schedule of every Iteration; the splats as a
ParameterDict, the Means_LR of every Iteration, one Adam per parameter, the Exposure_Fixes with
their own Adam, which the strategy never sees, and the MCMCStrategy. Then each Iteration goes in
this order: the clock check, the Loss_Terms that schedules.active_terms turns on, the render,
the Exposure_Fix, on a Prior_View the Prior_Terms' inputs, the Loss_Terms of the run's
Loss_Table, the backward pass, the finite check, every optimiser's step, the Means_LR of the
next Iteration, and the strategy's step_post_backward with that rate (Req 4.10).

The Loss_Table is _TERMS, exactly the splat-training spec's five Loss_Terms, with no Prior_Set,
and _TERMS + _PRIOR_TERMS with one (depth-priors spec, Req 13.1, 13.13). With no Prior_Set the
Trainer renders every view in the "RGB" mode, makes no Normal_Render and works out the
splat-training spec's loss. A Prior_View renders in "RGB+ED". A Prior is scaled up to the
Training_Size on each Iteration with prior_views' own functions, run on tensors, so the rule the
Test_Suite checks on numpy is the rule the Trainer runs (Req 10.6, 10.7, 11.7).

Both GPU calls turn torch.cuda.OutOfMemoryError into gpu_api.GpuOutOfMemory, with the Iteration
being run, or None before the first, so the CPU side catches it with no torch (Req 2.12). A loss
that isn't finite raises gpu_api.NonFiniteLoss before any optimiser steps, with every Loss_Term
of the run's Loss_Table and the total in its message (Req 4.16, A7; depth-priors spec,
Req 13.16). The erank Loss_Term is written fresh from D3's formula (Req 5.9), and the five
Prior_Terms from the depth-priors spec's Requirement 13 (Req 13.17). Every division and every
unit-length step in the Prior_Terms works on a value first made safe with torch.where, and the
pixel masks pick with torch.where, so no pixel left out of a term can put a NaN into the loss or
its gradient.

From splatpipe, this module imports only the CPU modules of splatpipe.train.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any, cast

import gsplat
import numpy as np
import torch
from fused_ssim import FusedSSIMMap
from gsplat import MCMCStrategy, rasterization

from splatpipe.train.deadline import should_stop, utc_now
from splatpipe.train.gpu_api import (GpuInfo, GpuOutOfMemory, NonFiniteLoss, TrainResult,
                                     non_finite_message)
from splatpipe.train.prior_views import (PriorTaps, ViewPrior, depth_weight, prior_pixel_mask,
                                         upscale)
from splatpipe.train.scene import Scene, TrainingView
from splatpipe.train.schedules import (active_terms, depth_schedule, image_order,
                                       means_lr_schedule, refine_stop, sh_degree)
from splatpipe.train.settings import TrainSettings
from splatpipe.train.splats import Splats

_DEVICE = torch.device("cuda")  # the job machine's one GPU
# Fixed_Constants: SSIM's constants, the ones fused_ssim itself passes to FusedSSIMMap.
_C1 = 0.01**2
_C2 = 0.03**2
# The Prior_Terms that read the Normal_Render, and the ones that read the Depth_Normals, so
# _prior_inputs makes each only when a term that reads it is on (depth-priors spec, Req 13.10).
_READ_NORMAL_RENDER = frozenset({"normal", "normal_consistency"})
_READ_DEPTH_NORMALS = frozenset({"depth_normal", "normal_consistency"})


def probe() -> GpuInfo:
    """The GPU check's facts: CUDA there or not, gsplat's version, and the GPU's total memory in
    bytes, 0 with no CUDA (Req 2.4).

    With CUDA it also resets torch's peak memory count, so peak_memory_bytes counts from here
    (Req 10.1).
    """
    cuda = torch.cuda.is_available()
    total = 0
    if cuda:
        total = int(torch.cuda.get_device_properties(0).total_memory)
        torch.cuda.reset_peak_memory_stats()
    return GpuInfo(cuda=cuda, gsplat_version=gsplat.__version__, total_memory_bytes=total)


def nearest_distances(points: np.ndarray, neighbours: int, chunk_pairs: int) -> np.ndarray:
    """Each of the (N, 3) points' distances to its `neighbours` nearest other points, as an
    (N, neighbours) float64 array in ascending order (Req 3.13).

    The points are recentred on their mean in float64 and sent to the GPU as float32. Each chunk
    of max(1, chunk_pairs // N) query points is measured against all N with torch.cdist's
    difference form, which keeps the distance between points that sit close together far from
    the origin, where the matmul form loses it. Each query's own column is set to infinity, so
    no point counts as its own neighbour. N is at least neighbours + 1 (Req 1.18, 9.8). torch
    running out of GPU memory becomes GpuOutOfMemory(None) (Req 2.12).
    """
    try:
        return _nearest(points, neighbours, chunk_pairs)
    except torch.cuda.OutOfMemoryError as error:
        raise GpuOutOfMemory(None) from error


def _nearest(points: np.ndarray, neighbours: int, chunk_pairs: int) -> np.ndarray:
    """nearest_distances' search, chunk by chunk."""
    centred = np.asarray(points, dtype=np.float64)
    every = _on_gpu(centred - centred.mean(axis=0), torch.float32)
    count = len(every)
    rows = max(1, chunk_pairs // count)
    chunks = [_nearest_in_chunk(every, first, min(first + rows, count), neighbours)
              for first in range(0, count, rows)]
    return torch.cat(chunks).cpu().numpy().astype(np.float64)


def _nearest_in_chunk(every: torch.Tensor, first: int, end: int,
                      neighbours: int) -> torch.Tensor:
    """The `neighbours` smallest distances, ascending, from each of the points first to end − 1
    of `every` to the other points of `every`, as [end − first, neighbours]."""
    gaps = torch.cdist(every[first:end], every, compute_mode="donot_use_mm_for_euclid_dist")
    own = torch.arange(end - first, device=every.device)
    gaps[own, own + first] = math.inf  # no point counts as its own neighbour
    return torch.topk(gaps, neighbours, dim=1, largest=False).values


def peak_memory_bytes() -> int:
    """The most GPU memory torch held at once since probe(), in bytes (Req 10.1)."""
    return int(torch.cuda.max_memory_allocated())


@dataclass(frozen=True)
class _ViewPrior:
    """One Prior_View's Prior on the GPU, kept at the Prior_Size h × w and scaled up on each
    Iteration, with its Brightness_Gradient at the Training_Size H × W (depth-priors spec,
    Req 10.5, D1)."""

    inverse_depth: torch.Tensor  # [h, w] float32
    normal: torch.Tensor  # [h, w, 3] float32, in the view's camera frame
    mask: torch.Tensor  # [h, w] bool
    gradient: torch.Tensor  # [H, W] float32, the Brightness_Gradient


@dataclass(frozen=True)
class _View:
    """One Training_Image's inputs on the GPU, in one record per image (Req 5.10).

    A Prior_View's record holds its Prior, and every other record an empty prior, None
    (depth-priors spec, Req 10.8).
    """

    pixels: torch.Tensor  # [H, W, 3] uint8, RGB
    keep: torch.Tensor | None  # [H, W] bool, True where it isn't a Masked_Pixel. None: no mask
    keep_count: int  # its kept pixels, counted on the CPU
    viewmat: torch.Tensor  # [1, 4, 4] float32, world to camera
    intrinsics: torch.Tensor  # [1, 3, 3] float32, K at the Training_Size
    index: int  # its place in the Exposure_Fix tensor
    prior: _ViewPrior | None  # its Prior on the GPU, None for an empty prior


# The scene's PriorTaps with its arrays as tensors on the GPU: the index maps int64 and the
# weights float32. prior_views' upscale and prior_pixel_mask take them as they take numpy arrays.
_Taps = PriorTaps


@dataclass(frozen=True)
class _Training:
    """Everything one call of train trains with. train makes it afresh for each call, so nothing
    outlives the call. The strategy swaps the tensors inside `params`, so they're always read
    through params[...]."""

    scene: Scene
    settings: TrainSettings
    views: tuple[_View, ...]  # in the Scene's order
    params: torch.nn.ParameterDict  # the splats, as gsplat's example holds them
    optimizers: dict[str, torch.optim.Optimizer]  # one Adam per key of params
    exposure: torch.nn.Parameter  # [V, 3, 4], each view's Exposure_Fix [A | b]
    exposure_optimizer: torch.optim.Optimizer  # kept out of the strategy's optimisers (Req 6.3)
    strategy: MCMCStrategy
    state: dict[str, Any]  # the strategy's state
    rates: tuple[float, ...]  # the Means_LR of each Iteration from 0 to steps
    table: tuple[_Row, ...]  # the Loss_Table: _TERMS, or _TERMS + _PRIOR_TERMS with a Prior_Set
    has_priors: bool  # whether the Train_Step has a Prior_Set
    taps: _Taps | None  # the Prior_Size to the Training_Size, None with no Prior_Set
    rays: torch.Tensor | None  # [H, W, 3]: ((x + 0.5 − cx) / fx, (y + 0.5 − cy) / fy, 1)
    depth_rates: tuple[float, ...]  # the Depth_Schedule of Iterations 0 to steps, () with none


@dataclass(frozen=True)
class _PriorInputs:
    """One Prior_View's inputs to the Prior_Terms at one Iteration, at the Training_Size H × W,
    every normal in the view's camera frame (depth-priors spec, Req 13)."""

    depth: torch.Tensor  # [H, W] the Rendered_Depth
    inverse_depth: torch.Tensor  # [H, W] the scaled-up prior, 1 outside the Prior_Pixels
    normal: torch.Tensor  # [H, W, 3] the scaled-up prior normal, unit length
    pixels: torch.Tensor  # [H, W] bool, the Prior_Pixels
    weights: torch.Tensor  # [H, W] the Depth_Weight
    rendered: torch.Tensor | None  # [H, W, 3] the Normal_Render, unit length
    rendered_ok: torch.Tensor | None  # [H, W] bool, the Normal_Render's length above 0
    surface: torch.Tensor | None  # [H, W, 3] the Depth_Normal
    surface_ok: torch.Tensor | None  # [H, W] bool, a Depth_Normal there


@dataclass(frozen=True)
class _Inputs:
    """One Iteration's inputs to every Loss_Term."""

    corrected: torch.Tensor  # [H, W, 3], the Corrected_Render
    target: torch.Tensor  # [H, W, 3], the Training_Image
    view: _View
    params: torch.nn.ParameterDict
    settings: TrainSettings
    extent: float  # the Scene_Extent
    prior: _PriorInputs | None  # None on a Training_Image with no Prior


def train(scene: Scene, start: Splats, settings: TrainSettings,
          stop: datetime | None) -> TrainResult:
    """The Trainer (Req 4.17): train the Start_Splats `start` on the scene with the checked
    settings, starting no Iteration at or after the Stop_Time `stop`, or running every
    Planned_Iteration for None (Req 7.6).

    It holds no state between calls. The Means_LR, the Refine_Stop and the Erank_Start, and with
    a Prior_Set the Depth_Schedule and the Depth_Normal_Start, are planned from settings.steps
    whether or not it stops early (Req 7.10; depth-priors spec, Req 11.6). The whole call, the
    upload included, turns torch.cuda.OutOfMemoryError into GpuOutOfMemory, with the Iteration
    being run, or None before the first (Req 2.12).
    """
    iteration: int | None = None  # the Iteration being run, None before the first
    try:
        run = _set_up(scene, start, settings)
        steps_run = 0
        order = image_order(len(run.views), settings.seed)  # Req 4.11
        for iteration, index in zip(range(settings.steps), order):
            if should_stop(utc_now(), stop):  # before anything else (Req 7.6)
                break
            _iterate(run, iteration, run.views[index])
            steps_run = iteration + 1
        return _result(run, steps_run)
    except torch.cuda.OutOfMemoryError as error:
        raise GpuOutOfMemory(iteration) from error


def _set_up(scene: Scene, start: Splats, settings: TrainSettings) -> _Training:
    """The Trainer's set-up, in order: the seed, the views on the GPU with their Priors, with a
    Prior_Set the taps, the rays and the Depth_Schedule, then the splats, the Means_LR, the
    optimisers, the Exposure_Fixes and the strategy.

    The scene has a Prior_Set when it has taps. Then the Loss_Table is _TERMS + _PRIOR_TERMS,
    and without one it's _TERMS, exactly the splat-training spec's five (depth-priors spec,
    Req 13.1, 13.13).
    """
    torch.manual_seed(settings.seed)  # every device, for MCMC's sampling and noise (Req 4.12)
    views = tuple(_view(view, index) for index, view in enumerate(scene.views))
    has_priors = scene.taps is not None
    taps, rays, depth_rates = _prior_set_up(scene, settings)
    params = _parameters(start)
    rates = means_lr_schedule(settings.means_lr * scene.extent, settings.means_lr_final,
                              settings.steps)  # Req 4.4
    optimizers = _optimizers(params, settings, rates[0])
    # Each Exposure_Fix starts at [I | 0] (Req 6.1), with its own Adam (Req 6.3).
    exposure = torch.nn.Parameter(torch.eye(3, 4, device=_DEVICE).repeat(len(views), 1, 1))
    exposure_optimizer = torch.optim.Adam([exposure], lr=settings.exposure_lr,
                                          eps=settings.adam_eps)
    strategy = _strategy(settings)
    strategy.check_sanity(params, optimizers)
    return _Training(scene=scene, settings=settings, views=views, params=params,
                     optimizers=optimizers, exposure=exposure,
                     exposure_optimizer=exposure_optimizer, strategy=strategy,
                     state=strategy.initialize_state(), rates=rates,
                     table=_TERMS + _PRIOR_TERMS if has_priors else _TERMS,
                     has_priors=has_priors, taps=taps, rays=rays, depth_rates=depth_rates)


def _view(view: TrainingView, index: int) -> _View:
    """The TrainingView `view` on the GPU, at place `index` in the Exposure_Fix tensor, with its
    Prior when it has one."""
    if view.keep is None:
        keep, keep_count = None, view.pixels.shape[0] * view.pixels.shape[1]
    else:
        keep, keep_count = _on_gpu(view.keep, torch.bool), int(np.count_nonzero(view.keep))
    return _View(pixels=_on_gpu(view.pixels, torch.uint8), keep=keep, keep_count=keep_count,
                 viewmat=_on_gpu(view.world_to_camera[np.newaxis], torch.float32),
                 intrinsics=_on_gpu(view.intrinsics[np.newaxis], torch.float32), index=index,
                 prior=_prior_on_gpu(view.prior))


def _prior_on_gpu(prior: ViewPrior | None) -> _ViewPrior | None:
    """A Prior_View's Prior on the GPU at the Prior_Size, as read, and its Brightness_Gradient
    as float32, or None for an empty prior (depth-priors spec, Req 10.5, D1)."""
    if prior is None:
        return None
    stored = prior.prior
    return _ViewPrior(inverse_depth=_on_gpu(stored.inverse_depth, torch.float32),
                      normal=_on_gpu(stored.normal, torch.float32),
                      mask=_on_gpu(stored.mask, torch.bool),
                      gradient=_on_gpu(prior.gradient, torch.float32))


def _prior_set_up(scene: Scene, settings: TrainSettings
                  ) -> tuple[_Taps | None, torch.Tensor | None, tuple[float, ...]]:
    """With a Prior_Set, which the scene's taps mark: the taps on the GPU, the rays and
    depth_schedule(depth_weight_start, depth_weight_end, steps), planned from the
    Planned_Iterations whether or not the Trainer stops early (depth-priors spec, Req 11.6).
    With none: None, None and no Depth_Schedule."""
    if scene.taps is None:
        return None, None, ()
    rates = depth_schedule(settings.depth_weight_start, settings.depth_weight_end,
                           settings.steps)
    return _taps_on_gpu(scene.taps), _rays(scene), rates


def _taps_on_gpu(taps: PriorTaps) -> _Taps:
    """`taps` with each index map as an int64 tensor and each weight as a float32 tensor on the
    GPU, uploaded once (depth-priors spec, Req 10.6)."""
    return PriorTaps(rows=_axis_on_gpu(taps.rows), columns=_axis_on_gpu(taps.columns),
                     nearest_rows=_on_gpu(taps.nearest_rows, torch.int64),
                     nearest_columns=_on_gpu(taps.nearest_columns, torch.int64))


def _axis_on_gpu(axis: tuple[np.ndarray, np.ndarray, np.ndarray]
                 ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One axis' taps, (i0, i1, w1), on the GPU as int64, int64 and float32."""
    first, second, weight = axis
    return (_on_gpu(first, torch.int64), _on_gpu(second, torch.int64),
            _on_gpu(weight, torch.float32))


def _rays(scene: Scene) -> torch.Tensor:
    """Each training pixel's ray through the scaled camera, [H, W, 3] float32:
    ((x + 0.5 − cx) / fx, (y + 0.5 − cy) / fy, 1), worked out in float64 (depth-priors spec,
    Req 13.8). Every view holds a copy of the one scaled K, so the first view's is taken."""
    intrinsics = scene.views[0].intrinsics
    across = (np.arange(scene.width) + 0.5 - intrinsics[0, 2]) / intrinsics[0, 0]
    down = (np.arange(scene.height) + 0.5 - intrinsics[1, 2]) / intrinsics[1, 1]
    rays = np.ones((scene.height, scene.width, 3))
    rays[..., 0] = across[np.newaxis, :]
    rays[..., 1] = down[:, np.newaxis]
    return _on_gpu(rays, torch.float32)


def _parameters(start: Splats) -> torch.nn.ParameterDict:
    """The Start_Splats as gsplat's example holds splats (Req 4.2): means [N, 3], scales [N, 3]
    as logs, quats [N, 4], opacities [N] as logits, sh0 [N, 1, 3] and shN [N, 15, 3]."""
    arrays = {"means": start.means, "scales": start.log_scales, "quats": start.quats,
              "opacities": start.opacity_logits, "sh0": start.sh0[:, np.newaxis],
              "shN": start.sh_rest}
    return torch.nn.ParameterDict({name: torch.nn.Parameter(_on_gpu(array, torch.float32))
                                   for name, array in arrays.items()})


def _optimizers(params: torch.nn.ParameterDict, settings: TrainSettings,
                means_lr: float) -> dict[str, torch.optim.Optimizer]:
    """One Adam of one parameter group per key of `params`, with adam_eps (Req 4.2, 4.3).

    The positions start at `means_lr`, the Means_LR of Iteration 0, and the others' rates never
    change (Req 4.5).
    """
    rates = {"means": means_lr, "scales": settings.scales_lr, "quats": settings.quats_lr,
             "opacities": settings.opacities_lr, "sh0": settings.sh_dc_lr,
             "shN": settings.sh_rest_lr}
    return {name: torch.optim.Adam([{"params": params[name], "lr": lr, "name": name}],
                                   eps=settings.adam_eps)
            for name, lr in rates.items()}


def _strategy(settings: TrainSettings) -> MCMCStrategy:
    """gsplat's MCMCStrategy with the Preset's numbers (Req 4.7). noise_lr and min_opacity stay at
    gsplat 1.5.3's defaults. It never holds more than cap_max splats (Req 4.9)."""
    return MCMCStrategy(cap_max=settings.max_splats,
                        refine_start_iter=settings.refine_start_step,
                        refine_every=settings.refine_every,
                        refine_stop_iter=refine_stop(settings.steps, settings.refine_stop_step,
                                                     settings.refine_stop_fraction))


def _iterate(run: _Training, i: int, view: _View) -> None:
    """Iteration i, on `view`, working out the Loss_Terms that active_terms turns on, the
    Erank_Start's rule included (Req 5.7; depth-priors spec, Req 13.14, 13.15)."""
    on = active_terms(i, run.settings, run.has_priors, view.prior is not None)
    rgb, depth, info = _render(run, i, view, with_depth=view.prior is not None)
    fix = run.exposure[view.index]
    corrected = rgb @ fix[:, :3].T + fix[:, 3]  # A·c + b at every pixel (Req 6.2)
    target = view.pixels.float() / 255
    # The render gives a Rendered_Depth exactly on a Prior_View.
    prior = None if depth is None else _prior_inputs(run, i, view, depth, on)
    inputs = _Inputs(corrected, target, view, run.params, run.settings, run.scene.extent, prior)
    terms, loss = _loss(run.table, inputs, on)
    # The loss has no gradient when no term was worked out, as when every weight is 0. Then the
    # backward pass is skipped, and the optimisers step with no gradients, which leaves the
    # splats as they are.
    if loss.requires_grad:
        loss.backward()
    _check_finite(i, terms, loss)
    for optimizer in (*run.optimizers.values(), run.exposure_optimizer):
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    run.optimizers["means"].param_groups[0]["lr"] = run.rates[i + 1]  # Req 4.4
    run.strategy.step_post_backward(params=run.params, optimizers=run.optimizers,
                                    state=run.state, step=i, info=info,
                                    lr=run.rates[i + 1])  # Req 4.10


def _render(run: _Training, i: int, view: _View,
            with_depth: bool) -> tuple[torch.Tensor, torch.Tensor | None, dict[str, Any]]:
    """The render of `view` at Iteration i's SH_Degree, [H, W, 3], its Rendered_Depth, [H, W],
    when `with_depth` and else None, and the info the strategy gets (Req 4.1).

    The classic mode, unpacked, at the Near_Plane and with no background, so a pixel no splat
    covers renders black. With `with_depth`, on a Prior_View, it renders in gsplat's "RGB+ED"
    mode, whose last channel is the expected camera-space z: the accumulated depth over the
    pixel's alpha, which gsplat clamps below, so a pixel no splat covers has a depth of 0.
    Otherwise it renders in "RGB", as the splat-training spec does (depth-priors spec, Req 13.2,
    13.12, 13.13).
    """
    params = run.params
    colors, _alphas, info = rasterization(
        means=params["means"],
        quats=params["quats"],
        scales=torch.exp(params["scales"]),
        opacities=torch.sigmoid(params["opacities"]),
        colors=torch.cat([params["sh0"], params["shN"]], 1),  # [N, 16, 3]
        viewmats=view.viewmat,
        Ks=view.intrinsics,
        width=run.scene.width,
        height=run.scene.height,
        near_plane=run.scene.near_plane,
        sh_degree=sh_degree(i, run.settings.sh_degree_interval),
        packed=False,
        rasterize_mode="classic",
        render_mode="RGB+ED" if with_depth else "RGB",
    )
    if not with_depth:
        return colors[0], None, info
    return colors[0, ..., :3], colors[0, ..., 3], info


def _prior_inputs(run: _Training, i: int, view: _View, depth: torch.Tensor,
                  on: tuple[str, ...]) -> _PriorInputs:
    """The Prior_View `view`'s inputs to the Prior_Terms at Iteration i, from its Rendered_Depth
    `depth` and the names `on` of the Loss_Terms that are on (depth-priors spec, Req 10.6, 10.7,
    11.5, 11.7, 13.10).

    prior_views' own functions run on the GPU's tensors: upscale scales the Prior up to the
    Training_Size, its inverse_depth set to 1 outside the Prior_Pixels so no division there can
    give an infinity, and its normal set to unit length with the safe step; prior_pixel_mask
    gives the Prior_Pixels with the view's keep mask; and depth_weight gives the Depth_Weight at
    Iteration i's Depth_Schedule value. The Normal_Render is made only when normal or
    normal_consistency is on, and the Depth_Normals only when depth_normal or
    normal_consistency is on.
    """
    prior = cast(_ViewPrior, view.prior)  # a Prior_View's record holds its Prior
    taps = cast(_Taps, run.taps)  # a Prior_View means a Prior_Set, which has taps
    pixels = prior_pixel_mask(prior.mask, view.keep, taps)
    inverse_depth = torch.where(pixels, upscale(prior.inverse_depth, taps), 1.0)
    normal, _long = _unit(upscale(prior.normal, taps))
    weights = depth_weight(run.depth_rates[i], prior.gradient, run.settings, torch)
    needed = frozenset(on)
    rendered, rendered_ok = (_normal_render(run, view) if needed & _READ_NORMAL_RENDER
                             else (None, None))
    surface, surface_ok = (_depth_normals(depth, cast(torch.Tensor, run.rays))
                           if needed & _READ_DEPTH_NORMALS else (None, None))
    return _PriorInputs(depth=depth, inverse_depth=inverse_depth, normal=normal, pixels=pixels,
                        weights=weights, rendered=rendered, rendered_ok=rendered_ok,
                        surface=surface, surface_ok=surface_ok)


def _unit(vectors: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """`vectors`, [..., D], each set to unit length, and where each one's squared length is above
    0, [...] bool.

    The safe unit-length step of the depth-priors design: a squared length of 0 is replaced by 1
    before its square root, so such a vector is divided by 1 and stays as it is, and neither the
    value nor its gradient holds a NaN. No small number is hard-coded.
    """
    squared = (vectors * vectors).sum(dim=-1, keepdim=True)
    long = squared > 0
    return vectors / torch.sqrt(torch.where(long, squared, 1.0)), long[..., 0]


def _splat_normals(params: torch.nn.ParameterDict, viewmat: torch.Tensor) -> torch.Tensor:
    """Each splat's Splat_Normal in the camera frame of the [4, 4] world-to-camera `viewmat`,
    [N, 3] (depth-priors spec, Req 13.3).

    Each quaternion (w, x, y, z) is set to unit length with the safe step and turned into its
    rotation matrix by the formula scene.world_to_camera uses. The matrix's columns are the
    splat's axes, scaled by its three scales, so the column of its smallest scale, picked with
    argmin over the log scales and gather, is its shortest axis in the world frame. That axis is
    turned into the camera frame with the view's rotation R, and turned round where its dot
    product with the splat's centre in the camera frame, R·mean + t, is above 0.
    """
    w, x, y, z = _unit(params["quats"])[0].unbind(dim=-1)
    rotations = torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], dim=-1).reshape(-1, 3, 3)
    shortest = params["scales"].argmin(dim=-1)
    axes = rotations.gather(2, shortest[:, None, None].expand(-1, 3, 1))[..., 0]
    rotation, translation = viewmat[:3, :3], viewmat[:3, 3]
    normals = axes @ rotation.T
    centres = params["means"].detach() @ rotation.T + translation
    return torch.where(_dot(normals, centres)[:, None] > 0, -normals, normals)


def _normal_render(run: _Training, view: _View) -> tuple[torch.Tensor, torch.Tensor]:
    """The Normal_Render of `view`, [H, W, 3] at unit length, and where its length is above 0,
    [H, W] bool (depth-priors spec, Req 13.4).

    A second rasterization() of the same splats, with their Splat_Normals as 3-channel colours
    and sh_degree=None, so gsplat uses the colours as given, with no +0.5 and no clamp, and
    negative components survive. The cameras, the size, the Near_Plane, the packing and the mode
    are the colour render's, and with no background a pixel no splat covers renders 0, which the
    safe unit step leaves at 0 and marks as having no length.
    """
    params = run.params
    colors, _alphas, _info = rasterization(
        means=params["means"],
        quats=params["quats"],
        scales=torch.exp(params["scales"]),
        opacities=torch.sigmoid(params["opacities"]),
        colors=_splat_normals(params, view.viewmat[0]),  # [N, 3]
        viewmats=view.viewmat,
        Ks=view.intrinsics,
        width=run.scene.width,
        height=run.scene.height,
        near_plane=run.scene.near_plane,
        sh_degree=None,
        packed=False,
        rasterize_mode="classic",
        render_mode="RGB",
    )
    return _unit(colors[0])


def _depth_normals(depth: torch.Tensor,
                   rays: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The Depth_Normals of the Rendered_Depth `depth`, [H, W, 3] at unit length, and where
    there is one, [H, W] bool (depth-priors spec, Req 13.8).

    Each pixel's point P is its depth × its ray, the pixel's centre back-projected through the
    scaled camera to that z. On the interior pixels, the central differences across,
    (P[y, x + 1] − P[y, x − 1]) / 2, and down, (P[y + 1, x] − P[y − 1, x]) / 2, give
    n = across × down, turned round where its dot product with the pixel's ray is above 0, which
    is facing the camera for any point on the ray in front of it, then set to unit length with
    the safe step. A pixel has a Depth_Normal when it's an interior pixel whose four neighbours
    have a depth above 0 and whose n has a length above 0. The border has none, and its normals
    are 0.
    """
    points = depth[..., None] * rays
    across = (points[1:-1, 2:] - points[1:-1, :-2]) / 2
    down = (points[2:, 1:-1] - points[:-2, 1:-1]) / 2
    crossed = torch.linalg.cross(across, down, dim=-1)
    facing = torch.where(_dot(crossed, rays[1:-1, 1:-1])[..., None] > 0, -crossed, crossed)
    unit, long = _unit(facing)
    covered = depth > 0
    surface = torch.zeros_like(points)
    surface[1:-1, 1:-1] = unit
    found = torch.zeros_like(covered)
    found[1:-1, 1:-1] = (long & covered[1:-1, 2:] & covered[1:-1, :-2] & covered[2:, 1:-1]
                         & covered[:-2, 1:-1])
    return surface, found


def _loss(table: tuple[_Row, ...], inputs: _Inputs,
          on: tuple[str, ...]) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Each Loss_Term's own value, before its weight, by name for every row of `table`, and the
    loss, Σ weight × value over the rows in `on`, with a weight of 1 for a row whose weight
    setting is None, the depth row (Req 5.1; depth-priors spec, Req 13.1, 13.5).

    A row whose name isn't in `on`, the names active_terms gives, isn't worked out and counts as
    0, so a term that's off can't make the loss non-finite (Req 5.7; depth-priors spec,
    Req 13.12, 13.14). With no Prior_Set, `table` is _TERMS and `on` is the splat-training rule,
    so the loss is the splat-training spec's.
    """
    zero = inputs.corrected.new_zeros(())
    terms: dict[str, torch.Tensor] = {}
    loss = zero
    for name, weight_setting, term in table:
        if name not in on:
            terms[name] = zero
            continue
        terms[name] = term(inputs)
        weight = 1.0 if weight_setting is None else getattr(inputs.settings, weight_setting)
        loss = loss + weight * terms[name]
    return terms, loss


def _check_finite(i: int, terms: Mapping[str, torch.Tensor], loss: torch.Tensor) -> None:
    """Raise NonFiniteLoss when Iteration i's loss isn't finite, with the value of every term of
    the run's Loss_Table, the Prior_Terms included with a Prior_Set, and the total in its
    message (Req 4.16, A7; depth-priors spec, Req 13.16).

    One torch.stack(...).tolist() reads every term and the total, six numbers or eleven with a
    Prior_Set, so the CPU waits for the GPU once per Iteration, where gsplat's example reads the
    loss for its progress bar.
    """
    values = torch.stack([*terms.values(), loss]).tolist()
    if not math.isfinite(values[-1]):
        raise NonFiniteLoss(non_finite_message(i, dict(zip([*terms, "total"], values))))


def _result(run: _Training, steps_run: int) -> TrainResult:
    """The trained splats as float32 numpy arrays, read through params[...] because the strategy
    swaps the tensors, each view's Exposure_Fix as a (V, 3, 4) array, the Iterations run and the
    last one's SH_Degree, 0 when none ran (Req 4.17)."""
    params = run.params
    splats = Splats(means=_on_cpu(params["means"]), log_scales=_on_cpu(params["scales"]),
                    quats=_on_cpu(params["quats"]), opacity_logits=_on_cpu(params["opacities"]),
                    sh0=_on_cpu(params["sh0"]).reshape(-1, 3), sh_rest=_on_cpu(params["shN"]))
    last = sh_degree(steps_run - 1, run.settings.sh_degree_interval) if steps_run else 0
    return TrainResult(splats=splats, exposures=_on_cpu(run.exposure), steps_run=steps_run,
                       sh_degree=last)


# The Loss_Terms. Each takes the Iteration's _Inputs, whose Corrected_Render and Training_Image
# are [H, W, 3] floats from 0 to 1, and gives the term's own value.

def _l1(inputs: _Inputs) -> torch.Tensor:
    """The mean absolute difference over the kept pixels' 3 channels (Req 5.2, 5.4)."""
    return _kept_mean((inputs.corrected - inputs.target).abs(), inputs.view)


def _ssim(inputs: _Inputs) -> torch.Tensor:
    """1 − the mean of fused-ssim's SSIM map, with "same" padding, over the kept pixels' 3
    channels, once every Masked_Pixel is 0 in both images (Req 5.3, 5.4). Only img1, the
    render, gets a gradient."""
    img1 = _masked_channels_first(inputs.corrected, inputs.view)
    img2 = _masked_channels_first(inputs.target, inputs.view)
    ssim_map = FusedSSIMMap.apply(_C1, _C2, img1.contiguous(), img2.contiguous(), "same", True)
    return 1 - _kept_mean(ssim_map[0].permute(1, 2, 0), inputs.view)


def _opacity_reg(inputs: _Inputs) -> torch.Tensor:
    """The mean of every splat's opacity, the sigmoid of its logit (Req 5.5)."""
    return torch.sigmoid(inputs.params["opacities"]).mean()


def _scale_reg(inputs: _Inputs) -> torch.Tensor:
    """The mean of every splat's three scales, the exponentials of its log scales, each divided by
    the Scene_Extent (Req 5.5, D7)."""
    return (torch.exp(inputs.params["scales"]) / inputs.extent).mean()


def _erank(inputs: _Inputs) -> torch.Tensor:
    """The Effective_Rank_Term, written fresh from D3's formula (Req 5.6, 5.8, 5.9).

    With s the splat's three scales, q = s² / Σ s², and log q is log_softmax(2 · log s), which is
    finite for any finite log scale. So q · log q is 0 where q underflows to 0, and so is its
    gradient. erank = exp(−Σ q log q) is at least 1, so erank − 1 + erank_epsilon is at least
    erank_epsilon and its log is finite. The term is the mean over splats of
    max(−log(erank − 1 + erank_epsilon), 0) + the splat's smallest scale divided by the
    Scene_Extent (D7).
    """
    log_scales = inputs.params["scales"]
    log_q = torch.log_softmax(2 * log_scales, dim=-1)
    erank = torch.exp(-(log_q.exp() * log_q).sum(dim=-1))
    penalty = torch.clamp(-torch.log(erank - 1 + inputs.settings.erank_epsilon), min=0)
    return (penalty + torch.exp(log_scales).min(dim=-1).values / inputs.extent).mean()


# The Prior_Terms, written fresh from the depth-priors spec's Requirement 13 (Req 13.17). Each is
# a mean over its pixels, taken with _pixel_mean, and every normal is in the view's camera frame.
# active_terms turns depth, normal, depth_normal and normal_consistency on only on a Prior_View,
# and _prior_inputs makes the Normal_Render and the Depth_Normals whenever a term that reads them
# is on.

def _depth(inputs: _Inputs) -> torch.Tensor:
    """The mean over the Prior_Pixels of w · min(|1/z − ι| / ι, depth_error_cap), with w the
    Depth_Weight, z the Rendered_Depth and ι the scaled-up inverse_depth, and the error counted
    as depth_error_cap where z isn't above 0 (depth-priors spec, Req 13.5).

    z is first made safe as 1 where it isn't above 0, and ι is 1 outside the Prior_Pixels, so no
    division can give an infinity. The cap means a splat right in front of the camera adds at
    most depth_error_cap and no gradient beyond it. The Depth_Weight is inside the term, so the
    term goes into the loss as it is.
    """
    prior = _prior(inputs)
    cap = inputs.settings.depth_error_cap
    covered = prior.depth > 0
    safe_depth = torch.where(covered, prior.depth, 1.0)
    error = (1 / safe_depth - prior.inverse_depth).abs() / prior.inverse_depth
    capped = torch.where(covered, torch.clamp(error, max=cap), cap)
    return _pixel_mean(prior.weights * capped, prior.pixels)


def _normal(inputs: _Inputs) -> torch.Tensor:
    """The mean of 1 − n_r · n_p, the Normal_Render's dot product with the scaled-up prior
    normal, over the Prior_Pixels where the Normal_Render's length is above 0 (depth-priors
    spec, Req 13.4, 13.6)."""
    prior = _prior(inputs)
    rendered, rendered_ok = _made(prior.rendered), _made(prior.rendered_ok)
    return _pixel_mean(1 - _dot(rendered, prior.normal), prior.pixels & rendered_ok)


def _flatten(inputs: _Inputs) -> torch.Tensor:
    """The mean over splats of the splat's smallest scale, the exponential of its log scale,
    divided by the Scene_Extent, in units of the Scene_Extent like scale_reg (depth-priors spec,
    Req 13.7, A6). It reads no prior, so with a Prior_Set it's on for every Training_Image."""
    return (torch.exp(inputs.params["scales"]).min(dim=-1).values / inputs.extent).mean()


def _depth_normal(inputs: _Inputs) -> torch.Tensor:
    """The mean of 1 − n_d · n_p, the Depth_Normal's dot product with the scaled-up prior
    normal, over the Prior_Pixels with a Depth_Normal (depth-priors spec, Req 13.9)."""
    prior = _prior(inputs)
    surface, surface_ok = _made(prior.surface), _made(prior.surface_ok)
    return _pixel_mean(1 - _dot(surface, prior.normal), prior.pixels & surface_ok)


def _normal_consistency(inputs: _Inputs) -> torch.Tensor:
    """The mean of 1 − n_r · n_d, the Normal_Render's dot product with the Depth_Normal, over
    the Prior_Pixels with a Depth_Normal where the Normal_Render's length is above 0
    (depth-priors spec, Req 13.4, 13.10)."""
    prior = _prior(inputs)
    rendered, rendered_ok = _made(prior.rendered), _made(prior.rendered_ok)
    surface, surface_ok = _made(prior.surface), _made(prior.surface_ok)
    return _pixel_mean(1 - _dot(rendered, surface), prior.pixels & surface_ok & rendered_ok)


_Term = Callable[[_Inputs], torch.Tensor]
# A Loss_Table row: the term's name, the setting that holds its weight, None for a weight of 1,
# and its function.
_Row = tuple[str, str | None, _Term]
# The named Loss_Terms, each with the setting that holds its weight: exactly the splat-training
# spec's five (Req 5.1; depth-priors spec, Req 13.13).
_TERMS: tuple[_Row, ...] = (
    ("l1", "l1_weight", _l1),
    ("ssim", "ssim_weight", _ssim),
    ("opacity_reg", "opacity_reg", _opacity_reg),
    ("scale_reg", "scale_reg", _scale_reg),
    ("erank", "erank_reg", _erank),
)
# The five Prior_Terms, which a Prior_Set adds after them (depth-priors spec, Req 13.1). depth's
# weight setting is None, meaning 1, because its Depth_Weight is inside the term (Req 13.5).
_PRIOR_TERMS: tuple[_Row, ...] = (
    ("depth", None, _depth),
    ("normal", "normal_weight", _normal),
    ("flatten", "flatten_reg", _flatten),
    ("depth_normal", "depth_normal_weight", _depth_normal),
    ("normal_consistency", "normal_consistency_weight", _normal_consistency),
)


def _prior(inputs: _Inputs) -> _PriorInputs:
    """The Iteration's _PriorInputs, which a Prior_View always has, and active_terms turns the
    Prior_Terms that read them on only on a Prior_View."""
    return cast(_PriorInputs, inputs.prior)


def _made(tensor: torch.Tensor | None) -> torch.Tensor:
    """One of _PriorInputs' optional tensors, which _prior_inputs made because a term that
    reads it is on."""
    return cast(torch.Tensor, tensor)


def _pixel_mean(values: torch.Tensor, pixels: torch.Tensor) -> torch.Tensor:
    """The mean of `values`, [H, W], over the [H, W] bool `pixels`: their sum, picked with
    torch.where so no value elsewhere reaches the loss, over max(count, 1), so a view with no
    such pixel adds 0. The count stays a tensor, so the CPU doesn't wait for it."""
    return torch.where(pixels, values, 0.0).sum() / pixels.sum().clamp_min(1)


def _dot(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    """The dot product of each pair of vectors along the last axis."""
    return (first * second).sum(dim=-1)


def _kept_mean(values: torch.Tensor, view: _View) -> torch.Tensor:
    """The mean of `values`, [H, W, 3], over the 3 channels of the view's kept pixels: every
    value's mean with no mask, else the sum at the kept pixels over 3 × keep_count."""
    if view.keep is None:
        return values.mean()
    return (values * view.keep[..., None]).sum() / (3 * view.keep_count)


def _masked_channels_first(image: torch.Tensor, view: _View) -> torch.Tensor:
    """`image`, [H, W, 3], as [1, 3, H, W], with its Masked_Pixels set to 0 when the view has a
    mask."""
    if view.keep is not None:
        image = image * view.keep[..., None]
    return image.permute(2, 0, 1)[None]


def _on_gpu(array: np.ndarray, dtype: torch.dtype) -> torch.Tensor:
    """A copy of `array` on the GPU, as `dtype`."""
    return torch.as_tensor(array, dtype=dtype, device=_DEVICE)


def _on_cpu(tensor: torch.Tensor) -> np.ndarray:
    """`tensor` as a float32 numpy array on the CPU."""
    return tensor.detach().to(device="cpu", dtype=torch.float32).numpy()
