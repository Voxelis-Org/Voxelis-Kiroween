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
# scaling. Added: a per-image exposure fix, masked L1 and SSIM losses, the effective-rank term, a
# stop time, and our own numbers from config.toml.
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
  calls.
- peak_memory_bytes (Req 10.1): the most GPU memory torch held at once since probe().

The Trainer sets up, in this order: torch's seed, the views on the GPU, the splats as a
ParameterDict, the Means_LR of every Iteration, one Adam per parameter, the Exposure_Fixes with
their own Adam, which the strategy never sees, and the MCMCStrategy. Then each Iteration goes in
this order: the clock check, the render, the Exposure_Fix, the Loss_Terms of _TERMS, the
backward pass, the finite check, every optimiser's step, the Means_LR of the next Iteration, and
the strategy's step_post_backward with that rate (Req 4.10).

Both GPU calls turn torch.cuda.OutOfMemoryError into gpu_api.GpuOutOfMemory, with the Iteration
being run, or None before the first, so the CPU side catches it with no torch (Req 2.12). A loss
that isn't finite raises gpu_api.NonFiniteLoss before any optimiser steps (Req 4.16, A7). The
erank Loss_Term is written fresh from D3's formula (Req 5.9).

From splatpipe, this module imports only the CPU modules of splatpipe.train.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import gsplat
import numpy as np
import torch
from fused_ssim import FusedSSIMMap
from gsplat import MCMCStrategy, rasterization

from splatpipe.train.deadline import should_stop, utc_now
from splatpipe.train.gpu_api import (GpuInfo, GpuOutOfMemory, NonFiniteLoss, TrainResult,
                                     non_finite_message)
from splatpipe.train.scene import Scene, TrainingView
from splatpipe.train.schedules import (erank_start, image_order, means_lr_schedule, refine_stop,
                                       sh_degree)
from splatpipe.train.settings import TrainSettings
from splatpipe.train.splats import Splats

_DEVICE = torch.device("cuda")  # the job machine's one GPU
# Fixed_Constants: SSIM's constants, the ones fused_ssim itself passes to FusedSSIMMap.
_C1 = 0.01**2
_C2 = 0.03**2


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
class _View:
    """One Training_Image's inputs on the GPU, in one record per image (Req 5.10).

    A depth prior adds its per-image input here and in scene.TrainingView.
    """

    pixels: torch.Tensor  # [H, W, 3] uint8, RGB
    keep: torch.Tensor | None  # [H, W] bool, True where it isn't a Masked_Pixel. None: no mask
    keep_count: int  # its kept pixels, counted on the CPU
    viewmat: torch.Tensor  # [1, 4, 4] float32, world to camera
    intrinsics: torch.Tensor  # [1, 3, 3] float32, K at the Training_Size
    index: int  # its place in the Exposure_Fix tensor


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


def train(scene: Scene, start: Splats, settings: TrainSettings,
          stop: datetime | None) -> TrainResult:
    """The Trainer (Req 4.17): train the Start_Splats `start` on the scene with the checked
    settings, starting no Iteration at or after the Stop_Time `stop`, or running every
    Planned_Iteration for None (Req 7.6).

    It holds no state between calls. The Means_LR, the Refine_Stop and the Erank_Start are
    planned from settings.steps whether or not it stops early (Req 7.10). The whole call, the
    upload included, turns torch.cuda.OutOfMemoryError into GpuOutOfMemory, with the Iteration
    being run, or None before the first (Req 2.12).
    """
    iteration: int | None = None  # the Iteration being run, None before the first
    try:
        run = _set_up(scene, start, settings)
        erank_from = erank_start(settings.steps, settings.erank_start_fraction)
        steps_run = 0
        order = image_order(len(run.views), settings.seed)  # Req 4.11
        for iteration, index in zip(range(settings.steps), order):
            if should_stop(utc_now(), stop):  # before anything else (Req 7.6)
                break
            _iterate(run, iteration, run.views[index], iteration >= erank_from)
            steps_run = iteration + 1
        return _result(run, steps_run)
    except torch.cuda.OutOfMemoryError as error:
        raise GpuOutOfMemory(iteration) from error


def _set_up(scene: Scene, start: Splats, settings: TrainSettings) -> _Training:
    """The Trainer's set-up, in order: the seed, the views on the GPU, the splats, the Means_LR,
    the optimisers, the Exposure_Fixes and the strategy."""
    torch.manual_seed(settings.seed)  # every device, for MCMC's sampling and noise (Req 4.12)
    views = tuple(_view(view, index) for index, view in enumerate(scene.views))
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
                     state=strategy.initialize_state(), rates=rates)


def _view(view: TrainingView, index: int) -> _View:
    """The TrainingView `view` on the GPU, at place `index` in the Exposure_Fix tensor."""
    if view.keep is None:
        keep, keep_count = None, view.pixels.shape[0] * view.pixels.shape[1]
    else:
        keep, keep_count = _on_gpu(view.keep, torch.bool), int(np.count_nonzero(view.keep))
    return _View(pixels=_on_gpu(view.pixels, torch.uint8), keep=keep, keep_count=keep_count,
                 viewmat=_on_gpu(view.world_to_camera[np.newaxis], torch.float32),
                 intrinsics=_on_gpu(view.intrinsics[np.newaxis], torch.float32), index=index)


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


def _iterate(run: _Training, i: int, view: _View, erank_on: bool) -> None:
    """Iteration i, on `view`, with the erank Loss_Term when `erank_on` (Req 5.7)."""
    render, info = _render(run, i, view)
    fix = run.exposure[view.index]
    corrected = render @ fix[:, :3].T + fix[:, 3]  # A·c + b at every pixel (Req 6.2)
    target = view.pixels.float() / 255
    terms, loss = _loss(corrected, target, view, run.params, run.settings, erank_on)
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


def _render(run: _Training, i: int, view: _View) -> tuple[torch.Tensor, dict[str, Any]]:
    """The render of `view` at Iteration i's SH_Degree, [H, W, 3], and the info the strategy
    gets (Req 4.1). The classic mode, unpacked, at the Near_Plane and with no background, so a
    pixel no splat covers renders black."""
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
    )
    return colors[0], info


def _loss(corrected: torch.Tensor, target: torch.Tensor, view: _View,
          params: torch.nn.ParameterDict, settings: TrainSettings,
          erank_on: bool) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Each Loss_Term's own value, before its weight, by name, and the loss, the sum of each
    weight × its term (Req 5.1).

    A term whose weight is 0, or erank when `erank_on` is false, isn't worked out and counts as
    0, so a term that's off can't make the loss non-finite (Req 5.7).
    """
    zero = corrected.new_zeros(())
    terms: dict[str, torch.Tensor] = {}
    loss = zero
    for name, weight_setting, term in _TERMS:
        weight = getattr(settings, weight_setting)
        if weight == 0 or (name == "erank" and not erank_on):
            terms[name] = zero
        else:
            terms[name] = term(corrected, target, view, params, settings)
            loss = loss + weight * terms[name]
    return terms, loss


def _check_finite(i: int, terms: Mapping[str, torch.Tensor], loss: torch.Tensor) -> None:
    """Raise NonFiniteLoss when Iteration i's loss isn't finite, with each term's value and the
    total in its message (Req 4.16, A7).

    One torch.stack(...).tolist() reads all six numbers, so the CPU waits for the GPU once per
    Iteration, where gsplat's example reads the loss for its progress bar.
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


# The Loss_Terms. Each takes the Iteration's Corrected_Render and Training_Image, [H, W, 3]
# floats from 0 to 1, the view, the splats and the settings, and gives the term's own value.

def _l1(corrected: torch.Tensor, target: torch.Tensor, view: _View,
        params: torch.nn.ParameterDict, settings: TrainSettings) -> torch.Tensor:
    """The mean absolute difference over the kept pixels' 3 channels (Req 5.2, 5.4)."""
    return _kept_mean((corrected - target).abs(), view)


def _ssim(corrected: torch.Tensor, target: torch.Tensor, view: _View,
          params: torch.nn.ParameterDict, settings: TrainSettings) -> torch.Tensor:
    """1 − the mean of fused-ssim's SSIM map, with "same" padding, over the kept pixels' 3
    channels, once every Masked_Pixel is 0 in both images (Req 5.3, 5.4). Only img1, the
    render, gets a gradient."""
    img1, img2 = _masked_channels_first(corrected, view), _masked_channels_first(target, view)
    ssim_map = FusedSSIMMap.apply(_C1, _C2, img1.contiguous(), img2.contiguous(), "same", True)
    return 1 - _kept_mean(ssim_map[0].permute(1, 2, 0), view)


def _opacity_reg(corrected: torch.Tensor, target: torch.Tensor, view: _View,
                 params: torch.nn.ParameterDict, settings: TrainSettings) -> torch.Tensor:
    """The mean of every splat's opacity, the sigmoid of its logit (Req 5.5)."""
    return torch.sigmoid(params["opacities"]).mean()


def _scale_reg(corrected: torch.Tensor, target: torch.Tensor, view: _View,
               params: torch.nn.ParameterDict, settings: TrainSettings) -> torch.Tensor:
    """The mean of every splat's three scales, the exponentials of its log scales (Req 5.5)."""
    return torch.exp(params["scales"]).mean()


def _erank(corrected: torch.Tensor, target: torch.Tensor, view: _View,
           params: torch.nn.ParameterDict, settings: TrainSettings) -> torch.Tensor:
    """The Effective_Rank_Term, written fresh from D3's formula (Req 5.6, 5.8, 5.9).

    With s the splat's three scales, q = s² / Σ s², and log q is log_softmax(2 · log s), which is
    finite for any finite log scale. So q · log q is 0 where q underflows to 0, and so is its
    gradient. erank = exp(−Σ q log q) is at least 1, so erank − 1 + erank_epsilon is at least
    erank_epsilon and its log is finite. The term is the mean over splats of
    max(−log(erank − 1 + erank_epsilon), 0) + the splat's smallest scale.
    """
    log_scales = params["scales"]
    log_q = torch.log_softmax(2 * log_scales, dim=-1)
    erank = torch.exp(-(log_q.exp() * log_q).sum(dim=-1))
    penalty = torch.clamp(-torch.log(erank - 1 + settings.erank_epsilon), min=0)
    return (penalty + torch.exp(log_scales).min(dim=-1).values).mean()


_Term = Callable[[torch.Tensor, torch.Tensor, _View, torch.nn.ParameterDict, TrainSettings],
                 torch.Tensor]
# The named Loss_Terms, each with the setting that holds its weight (Req 5.1). A depth prior adds
# a row (Req 5.10).
_TERMS: tuple[tuple[str, str, _Term], ...] = (
    ("l1", "l1_weight", _l1),
    ("ssim", "ssim_weight", _ssim),
    ("opacity_reg", "opacity_reg", _opacity_reg),
    ("scale_reg", "scale_reg", _scale_reg),
    ("erank", "erank_reg", _erank),
)


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
