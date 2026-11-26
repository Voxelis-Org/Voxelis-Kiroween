"""The Depth_GPU_Module's interface as the CPU sees it, and the checks that turn its errors into
the Depth_Step's Expected_Failures.

The Depth_GPU_Module, gpu.py, is the only file of the Depth_Step that imports torch, MoGe and
huggingface_hub. The step reaches it only through load_depth_gpu, which imports it by name when
the step runs, so importing this module imports none of them (Req 3.15). The DepthGpu protocol is
what the step and the Depth_Smoke_Check call on the Depth_GPU_Module, or on the tests'
Depth_Stand_In, and everything it gives back has the CPU modules' own types, so the fit and the
Prior_Writer work on numpy arrays and the Test_Suite builds them on a laptop.

- MoGeOutput (Req 4.6): one Frame's MoGe_Output at the Prior_Size, as numpy arrays: depth (h, w)
  float32, camera-space z in metres; normal (h, w, 3) float32, unit normals in the camera frame
  that face the camera; and mask (h, w) bool, true where MoGe found geometry. With MoGe's
  apply_mask at its default, depth is +inf and normal 0 where mask is false (moge-depth skill).
- MissingPackage, WrongCheckpoint, DepthOutOfMemory and OfflineModeOff: the four errors the
  Depth_GPU_Module raises, defined here so the CPU side catches them with no torch. The first
  three never reach the Step_Runner, because the functions below turn them into Expected_Failures.
  OfflineModeOff is a RuntimeError and not a PipelineError, so it goes up as it is and the
  Step_Runner reports it as an Unexpected_Error, with its message in the Technical_Detail
  (Req 3.8, A16).
- load_depth_gpu (Req 3.2): imports DEPTH_GPU_MODULE. A missing torch, or a submodule of it, is
  the Expected_Failure saying that the step runs only on a GPU machine image built with MoGe's
  code. Any other error, a bug in our own module included, goes up as it is, an Unexpected_Error,
  so a bug is never reported as the wrong machine.
- check_machine (Req 3.2–3.5): refuses no CUDA, then a moge_weights that isn't a file, before
  anything imports MoGe or loads the file, then a package that MoGe's import is missing. It
  checks no amount of GPU memory: min_gpu_memory_gb in [depth] stays a cloud-only Step_Need.
- load_moge (Req 3.11, 3.13) and run_moge (Req 4.4, 3.13): load MoGe-2 from moge_weights and run
  it on one Frame with resolution_level and use_fp16 from [depth]. A wrong checkpoint and running
  out of GPU memory become their Expected_Failures, the second naming the Frame reached.

The Depth_Step and the Depth_Smoke_Check both load and call MoGe through these functions, so the
smoke check runs the step's own load and call (Req 16.2).

This module imports only the standard library and numpy, and from splatpipe only
splatpipe.errors and splatpipe.depth.settings.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

import numpy as np

from splatpipe.depth.settings import DepthSettings
from splatpipe.errors import ExpectedFailure

DEPTH_GPU_MODULE = "splatpipe.depth.gpu"  # what load_depth_gpu imports: the Depth_GPU_Module
_TORCH = "torch"  # the one GPU package the Depth_GPU_Module imports at its top

_CLOUD_IMAGE = "`uv run python -m cloud.image`"  # the command that rebuilds the Machine_Image
_CLOUD_START = "`uv run python -m cloud.start`"  # the command that runs a Job on the GPU machine
_NO_CUDA = ("The depth step needs an NVIDIA GPU, but torch can't see one on this machine. Check "
            f"the driver with nvidia-smi, or run the job with {_CLOUD_START}.")  # Req 3.3
_NO_PRIORS = "The job goes on and trains without depth priors."  # Req 3.13
_OFFLINE_MODE_OFF = ("huggingface_hub was imported before HF_HUB_OFFLINE was set, so it isn't "
                     "offline and could download. This is a bug in the depth step.")  # Req 3.8


@dataclass(frozen=True)
class MoGeOutput:
    """One Frame's MoGe_Output at the Prior_Size h × w (Req 4.6)."""

    depth: np.ndarray  # (h, w) float32, camera-space z in metres, +inf where mask is false
    normal: np.ndarray  # (h, w, 3) float32, unit, camera frame, facing the camera; 0 off the mask
    mask: np.ndarray  # (h, w) bool, true where MoGe found geometry


class DepthGpu(Protocol):
    """What the step and the Depth_Smoke_Check call on the Depth_GPU_Module, or on a
    Depth_Stand_In."""

    def probe(self) -> bool:
        """Whether torch sees a CUDA GPU. With one, it resets torch's peak memory count, so
        peak_memory_bytes counts from here (Req 9.1)."""
        ...

    def import_moge(self) -> None:
        """Set HF_HUB_OFFLINE to 1, import huggingface_hub and MoGe-2, and check that
        huggingface_hub is offline (Req 3.6, 3.7). Raises MissingPackage or OfflineModeOff."""
        ...

    def load_model(self, weights: Path) -> object:
        """MoGe-2 loaded strictly from the Weights_File `weights`, on the GPU in eval mode and in
        float32 (Req 3.9–3.12), as a handle for infer and release. Raises WrongCheckpoint, or
        DepthOutOfMemory(None)."""
        ...

    def infer(self, model: object, rgb: np.ndarray, fov_x: float, resolution_level: int,
              use_fp16: bool, frame: str) -> MoGeOutput:
        """MoGe-2's output for the (h, w, 3) uint8 RGB Frame `rgb`, whose horizontal field of
        view is `fov_x` degrees, at the Prior_Size h × w (Req 4.2, 4.4, 4.6). Raises
        DepthOutOfMemory(frame), with `frame` the Frame's name."""
        ...

    def release(self, model: object | None) -> None:
        """Drop the handle's network and empty torch's GPU memory cache (Req 3.14). Takes None
        for a load that failed."""
        ...

    def peak_memory_bytes(self) -> int:
        """The most GPU memory torch held at once since probe(), in bytes (Req 9.1)."""
        ...


class MissingPackage(Exception):
    """A package that MoGe-2's import needs can't be imported (Req 3.2).

    `package` is the missing module's top-level name, like "moge", "utils3d_moge" or "httpx".
    """

    package: str

    def __init__(self, package: str) -> None:
        super().__init__(f"the package {package} can't be imported")
        self.package = package


class WrongCheckpoint(Exception):
    """The Weights_File isn't the MoGe-2 ViT-B normal checkpoint (Req 3.11).

    `reason` says why, shortly, like "it holds no model_config" or "the strict load found
    unexpected keys". It goes in brackets in the Req 3.11 message.
    """

    reason: str

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class DepthOutOfMemory(Exception):
    """torch ran out of GPU memory on the Frame `frame`, or while MoGe-2 loaded for None
    (Req 3.13).

    The Depth_GPU_Module raises it in place of torch.cuda.OutOfMemoryError, so the CPU side
    catches it with no torch. Its own short message reads well even where nothing turns it into
    the Req 3.13 failure.
    """

    frame: str | None  # the Frame's name, None while loading

    def __init__(self, frame: str | None) -> None:
        where = "while MoGe-2 loaded" if frame is None else f"on the frame {frame!r}"
        super().__init__(f"torch ran out of GPU memory {where}")
        self.frame = frame


class OfflineModeOff(RuntimeError):
    """huggingface_hub isn't offline after MoGe-2's import, which is a bug in the depth step
    (Req 3.8, A16).

    huggingface_hub reads HF_HUB_OFFLINE once, when it's first imported, so this means something
    imported it before the Depth_GPU_Module set the variable. It isn't a PipelineError, so the
    Step_Runner reports it as an Unexpected_Error, with the message in the Technical_Detail.
    Built with no message, it has the design's.
    """

    def __init__(self, message: str = _OFFLINE_MODE_OFF) -> None:
        super().__init__(message)


def load_depth_gpu() -> DepthGpu:
    """The Depth_GPU_Module, imported by name when the step runs (Req 3.2, 3.15).

    The import goes through this module's own importlib, so a test can replace
    splatpipe.depth.gpu_api.importlib. A ModuleNotFoundError whose missing module, up to its
    first dot, is torch becomes the Req 3.2 Expected_Failure, naming torch. Any other error goes
    up as it is. MoGe and huggingface_hub are imported later, by import_moge.
    """
    try:
        return cast(DepthGpu, importlib.import_module(DEPTH_GPU_MODULE))
    except ModuleNotFoundError as error:
        if (error.name or "").partition(".")[0] != _TORCH:
            raise
        raise ExpectedFailure(_no_package(_TORCH)) from error


def check_machine(gpu: DepthGpu, settings: DepthSettings) -> None:
    """Refuse a machine the step can't run MoGe-2 on.

    In this order: no CUDA GPU (Req 3.3); a moge_weights that isn't a file, before anything
    imports MoGe or loads the file (Req 3.5); and a MissingPackage from import_moge, naming the
    package (Req 3.2). OfflineModeOff from import_moge goes up as it is (Req 3.8). It checks no
    amount of GPU memory (Req 3.4).
    """
    if not gpu.probe():
        raise ExpectedFailure(_NO_CUDA)
    if not Path(settings.moge_weights).is_file():
        raise ExpectedFailure(f"The MoGe-2 weights aren't at {settings.moge_weights}, the "
                              "moge_weights setting in [depth]. The image build downloads them "
                              f"there, so rebuild the machine image with {_CLOUD_IMAGE}.")
    try:
        gpu.import_moge()
    except MissingPackage as error:
        raise ExpectedFailure(_no_package(error.package)) from error


def load_moge(gpu: DepthGpu, settings: DepthSettings) -> object:
    """MoGe-2 loaded from moge_weights, as the Depth_GPU_Module's handle.

    WrongCheckpoint becomes the Req 3.11 failure, with its reason, and DepthOutOfMemory the
    Req 3.13 failure for the load.
    """
    try:
        return gpu.load_model(Path(settings.moge_weights))
    except WrongCheckpoint as error:
        raise ExpectedFailure(f"The file at {settings.moge_weights} isn't the MoGe-2 ViT-B "
                              f"normal checkpoint ({error.reason}). Rebuild the machine image "
                              f"with {_CLOUD_IMAGE}.") from error
    except DepthOutOfMemory as error:
        raise ExpectedFailure("The GPU ran out of memory while the depth step loaded MoGe-2. "
                              f"{_NO_PRIORS}") from error


def run_moge(gpu: DepthGpu, model: object, rgb: np.ndarray, fov_x: float,
             settings: DepthSettings, frame: str) -> MoGeOutput:
    """MoGe-2's output for the Frame `frame`, whose (h, w, 3) uint8 RGB pixels are `rgb` and
    whose horizontal field of view is `fov_x` degrees.

    resolution_level and use_fp16 come from [depth] (Req 4.4). DepthOutOfMemory becomes the
    Req 3.13 failure naming `frame`.
    """
    try:
        return gpu.infer(model, rgb, fov_x, resolution_level=settings.resolution_level,
                         use_fp16=settings.use_fp16, frame=frame)
    except DepthOutOfMemory as error:
        raise ExpectedFailure("The GPU ran out of memory while the depth step ran MoGe-2 on the "
                              f"frame {frame!r}. {_NO_PRIORS}") from error


def _no_package(package: str) -> str:
    """The Req 3.2 message for a machine with no `package`."""
    return (f"The depth step runs only on a GPU machine image built with MoGe's code, and this "
            f"machine has no {package}. Rebuild the machine image with {_CLOUD_IMAGE}, or run "
            f"the job there with {_CLOUD_START}.")
