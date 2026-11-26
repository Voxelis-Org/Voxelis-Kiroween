"""The Depth_GPU_Module: MoGe-2 on the GPU, loaded strictly from the image's own weights, offline.

This is the one module of the Depth_Step that imports torch, MoGe and huggingface_hub (Req 3.6,
18.8). It imports torch at its top, and huggingface_hub and MoGe only inside import_moge and
load_model, after HF_HUB_OFFLINE is set. The step reaches it only through
gpu_api.load_depth_gpu, which imports it by name when the step runs, and its six functions make
up gpu_api's DepthGpu protocol. What they give back is numpy arrays, plain values and the
opaque handle _Loaded, never a torch tensor.

It holds no code copied from MoGe: it calls MoGe's API only, as the moge-depth skill gives it
at MoGe's commit 74fbce05 (Req 3.9, 3.10). It never calls MoGe's own loader, never asks the Hub
for a file and never sets the encoder up from DINOv2's original weights, so a Job downloads
nothing (Req 3.10).

- probe (Req 3.3, 9.1): whether torch sees CUDA. With CUDA it also resets torch's peak memory
  count, so peak_memory_bytes counts from the GPU check on.
- import_moge (Req 3.2, 3.6–3.8): sets HF_HUB_OFFLINE to 1 first, then imports
  huggingface_hub.constants and MoGe-2's model class, then checks that huggingface_hub reports
  its offline mode as on. huggingface_hub reads the variable once, when its constants module is
  first imported, so the check fails only when something imported huggingface_hub earlier in
  the process. The variable stays set for the rest of the process.
- load_model (Req 3.9, 3.11, 3.12): our own strict load. torch.load with weights_only=True,
  which loads only tensors and plain values, MoGeModel(**checkpoint["model_config"]), and
  load_state_dict(checkpoint["model"]) with its default strict=True, so a missing, extra or
  misshapen layer is an error and no part of the model runs with random weights. Then the model
  goes to the GPU in eval mode, left in float32, as MoGe's README example does: MoGe's infer
  runs the network under fp16 autocast itself when use_fp16 is on.
- infer (Req 4.2, 4.4, 4.6): the (h, w, 3) uint8 RGB Frame as (3, h, w) floats from 0 to 1 on
  the GPU, MoGe-2's infer with fov_x, resolution_level and use_fp16, and num_tokens,
  force_projection and apply_mask at MoGe's defaults, and its depth, normal and mask copied to
  numpy at the Prior_Size. MoGe's outputs are inference tensors, and copying them to the CPU is
  allowed outside inference mode, so no clone is needed.
- release (Req 3.14): sets the handle's network to None, collects garbage and empties torch's
  GPU memory cache, so the Train_Step that runs next in the same process gets the memory. It
  takes None, for a load that failed.
- peak_memory_bytes (Req 9.1): the most GPU memory torch's allocator held at once since probe().

The errors are gpu_api's, so the CPU side catches them with no torch. A ModuleNotFoundError from
import_moge's two imports is MissingPackage of the missing module's top-level name, because
nothing of ours runs inside them, so it's always a package missing from the image (Req 3.2).
OfflineModeOff is a RuntimeError, an Unexpected_Error (Req 3.8). Each way the checkpoint can be
wrong is WrongCheckpoint with a short reason (Req 3.11): torch.load refusing the file, a value
that isn't a dict holding model_config and model, a TypeError from building the model, and the
strict load refusing the weights. torch.cuda.OutOfMemoryError is a RuntimeError too, so it's
caught on its own, around the move to the GPU alone in load_model, and around the whole GPU
work in infer, and becomes DepthOutOfMemory, with None while loading or the Frame's name
(Req 3.13).

Both out-of-memory paths catch torch's error without naming it, leave the except block, and
only then raise DepthOutOfMemory from None. Raised inside the except block, DepthOutOfMemory
would keep torch's error as its implicit context, even with `from None`, and that error's
traceback keeps the frames of the call that ran out alive, with every tensor they hold, so
release couldn't free them while the step's failure goes up. The frame that raises holds no
tensor either: infer's GPU work runs in _infer_on_gpu, whose frame is gone by then, and
load_model deletes its half-moved network first.

From splatpipe, this module imports only splatpipe.depth.gpu_api.
"""

from __future__ import annotations

import gc
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from splatpipe.depth.gpu_api import (DepthOutOfMemory, MissingPackage, MoGeOutput,
                                     OfflineModeOff, WrongCheckpoint)

_DEVICE = torch.device("cuda")  # the job machine's one GPU
_CHECKPOINT_KEYS = ("model_config", "model")  # what MoGe-2's checkpoint dict holds


@dataclass
class _Loaded:
    """MoGe-2 on the GPU, the handle load_model gives back and infer and release take.

    release sets `net` to None, so the network goes even while the step still holds the handle.
    """

    net: Any  # moge.model.v2.MoGeModel, which only import_moge may import; None once released


def probe() -> bool:
    """Whether torch sees a CUDA GPU (Req 3.3).

    With CUDA it also resets torch's peak memory count, so peak_memory_bytes counts from here
    (Req 9.1).
    """
    cuda = torch.cuda.is_available()
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    return cuda


def import_moge() -> None:
    """Import huggingface_hub and MoGe-2 offline, and check that huggingface_hub is offline.

    HF_HUB_OFFLINE is set to 1 before either import (Req 3.6). A ModuleNotFoundError from the two
    imports becomes MissingPackage of the missing module's top-level name, or of moge when
    Python gives no name (Req 3.2). Then huggingface_hub's offline mode must be on, or
    OfflineModeOff goes up, an Unexpected_Error (Req 3.7, 3.8).
    """
    os.environ["HF_HUB_OFFLINE"] = "1"  # before anything below imports huggingface_hub
    try:
        import huggingface_hub.constants
        from moge.model.v2 import MoGeModel  # noqa: F401  load_model uses it; here it must import
    except ModuleNotFoundError as error:
        raise MissingPackage((error.name or "moge").partition(".")[0]) from error
    if huggingface_hub.constants.HF_HUB_OFFLINE is not True:
        raise OfflineModeOff()


def load_model(weights: Path) -> _Loaded:
    """MoGe-2 loaded strictly from the Weights_File `weights`, on the GPU in eval mode and in
    float32 (Req 3.9, 3.11, 3.12).

    import_moge has run, so importing MoGeModel here is only a lookup in sys.modules. A wrong
    checkpoint raises WrongCheckpoint. torch running out of GPU memory while the model moves to
    the GPU raises DepthOutOfMemory(None), after the except block and with the half-moved
    network deleted, so no traceback keeps its tensors on the GPU (Req 3.13, 3.14).
    """
    from moge.model.v2 import MoGeModel

    net = _strict_load(MoGeModel, weights)
    try:
        net = net.to(_DEVICE)  # the parameters keep their float32 dtype (Req 3.12)
    except torch.cuda.OutOfMemoryError:
        pass  # raised below, once torch's error and the frames it holds are gone
    else:
        return _Loaded(net.eval())
    del net
    raise DepthOutOfMemory(None) from None


def _strict_load(model_class: Any, weights: Path) -> Any:
    """MoGe-2 on the CPU, built from the checkpoint's model_config and loaded strictly with its
    model (Req 3.9, 3.11). Each way the checkpoint can be wrong raises WrongCheckpoint."""
    try:
        checkpoint = torch.load(weights, map_location="cpu", weights_only=True)
    except (pickle.UnpicklingError, EOFError, RuntimeError) as error:
        raise WrongCheckpoint("torch can't read it as a weights-only checkpoint") from error
    if not isinstance(checkpoint, dict):
        raise WrongCheckpoint("it isn't a dict of model_config and model")
    for key in _CHECKPOINT_KEYS:
        if key not in checkpoint:
            raise WrongCheckpoint(f"it holds no {key}")
    try:
        net = model_class(**checkpoint["model_config"])
    except TypeError as error:
        raise WrongCheckpoint("its model_config doesn't build MoGe-2's model") from error
    try:
        net.load_state_dict(checkpoint["model"])  # strict: a missing or unexpected key fails
    except (RuntimeError, TypeError) as error:  # TypeError: a model that isn't a state dict
        raise WrongCheckpoint("its model doesn't load strictly into the model its model_config "
                              "builds") from error
    return net


def infer(model: _Loaded, rgb: np.ndarray, fov_x: float, resolution_level: int,
          use_fp16: bool, frame: str) -> MoGeOutput:
    """MoGe-2's output for the (h, w, 3) uint8 RGB Frame `rgb`, whose horizontal field of view is
    `fov_x` degrees, at the Prior_Size h × w (Req 4.2, 4.4, 4.6).

    torch running out of GPU memory raises DepthOutOfMemory(frame), with `frame` the Frame's
    name, after the except block, so no traceback keeps the tensors of the call that ran out
    (Req 3.13, 3.14).
    """
    try:
        return _infer_on_gpu(model, rgb, fov_x, resolution_level, use_fp16)
    except torch.cuda.OutOfMemoryError:
        pass  # raised below, once torch's error and the frames it holds are gone
    raise DepthOutOfMemory(frame) from None


def _infer_on_gpu(model: _Loaded, rgb: np.ndarray, fov_x: float, resolution_level: int,
                  use_fp16: bool) -> MoGeOutput:
    """infer's GPU work. Its tensors go when it returns, or with torch's error when torch runs
    out of GPU memory."""
    image = torch.from_numpy(rgb).to(_DEVICE).permute(2, 0, 1).float() / 255  # (3, h, w), 0 to 1
    out = model.net.infer(image, fov_x=fov_x, resolution_level=resolution_level,
                          use_fp16=use_fp16)  # num_tokens, force_projection, apply_mask: defaults
    return MoGeOutput(depth=out["depth"].float().cpu().numpy(),
                      normal=out["normal"].float().cpu().numpy(),
                      mask=out["mask"].cpu().numpy().astype(bool))


def release(model: _Loaded | None) -> None:
    """Drop the handle's network, collect garbage and empty torch's GPU memory cache (Req 3.14).

    Takes None, for a load that failed.
    """
    if model is not None:
        model.net = None
    gc.collect()
    torch.cuda.empty_cache()


def peak_memory_bytes() -> int:
    """The most GPU memory torch's allocator held at once since probe(), in bytes (Req 9.1)."""
    return int(torch.cuda.max_memory_allocated())
