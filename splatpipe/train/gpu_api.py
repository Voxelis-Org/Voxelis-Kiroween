"""The GPU_Module's interface as the CPU sees it, and the checks that decide on the GPU's facts.

The step reaches the GPU_Module, gpu.py, only through load_gpu, which imports it by name when the
step runs, so importing this module imports no torch, gsplat or fused_ssim (Req 9.12). The Gpu
protocol is what the step and the Train_Smoke_Check call on the GPU_Module, or on the tests'
GPU_Stand_In, and its types are the CPU modules' own.

- load_gpu (Req 2.2): imports GPU_MODULE. A missing torch, gsplat or fused_ssim, or a submodule
  of one, is the Expected_Failure saying that the step runs only on the GPU machine image. Any
  other error, a missing fused_ssim_cuda extension or a bug in our own module included, goes up
  as it is, an Unexpected_Error, so a bug is never reported as the wrong machine.
- check_gpu (Req 2.3–2.5, 2.10, 2.11): decides on the GpuInfo that the GPU_Module's probe()
  gives. It refuses no CUDA, then a gsplat other than GSPLAT_VERSION, then a GPU that gpu_fits
  refuses, and checks nothing else about the machine.
- gpu_need_bytes and gpu_fits (Req 2.6, 2.7): the GPU_Need, gpu_memory_slack ×
  min_gpu_memory_gb × GB bytes, and the GPU_Fit_Check, pure functions of their numbers.
- GpuOutOfMemory and out_of_memory_failure (Req 2.12): the GPU_Module raises GpuOutOfMemory in
  place of torch's out-of-memory error, with the Iteration reached, and the step turns it into
  the Expected_Failure that names the Iteration, the Preset and its max_splats.
- NonFiniteLoss and non_finite_message (Req 4.16, A7): the Trainer raises NonFiniteLoss when an
  Iteration's loss isn't finite. It's a RuntimeError and not a PipelineError, so the Step_Runner
  reports it as an Unexpected_Error, with the Iteration and each Loss_Term's value in its message.

From splatpipe, this module imports only splatpipe.errors, and scene.py, splats.py and
settings.py for their types.
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, cast

import numpy as np

from splatpipe.errors import ExpectedFailure
from splatpipe.train.scene import Scene
from splatpipe.train.settings import TrainSettings
from splatpipe.train.splats import Splats

GPU_MODULE = "splatpipe.train.gpu"  # the module load_gpu imports: the GPU_Module
GPU_PACKAGES = ("torch", "gsplat", "fused_ssim")  # Fixed_Constants: the GPU_Module's GPU imports
GSPLAT_VERSION = "1.5.3"  # the gsplat the GPU_Module is written for (A10)
GB = 10**9  # bytes in a GB, as gpu_memory_gb in [cloud.types] counts them (D1)

_CLOUD_START = "`uv run python -m cloud.start`"  # the command that runs a Job on the GPU machine
_NO_CUDA = ("The train step needs an NVIDIA GPU, but torch can't see one on this machine. Check "
            f"the driver with nvidia-smi, or run the job with {_CLOUD_START}.")  # Req 2.3


@dataclass(frozen=True)
class GpuInfo:
    """The GPU check's facts, which the GPU_Module's probe() gives and check_gpu decides on."""

    cuda: bool  # torch.cuda.is_available()
    gsplat_version: str  # gsplat.__version__
    total_memory_bytes: int  # torch.cuda.get_device_properties(0).total_memory, 0 with no CUDA


@dataclass(frozen=True)
class TrainResult:
    """What the Trainer gives back (Req 4.17)."""

    splats: Splats  # float32 numpy arrays, at most max_splats splats (Req 4.9)
    exposures: np.ndarray  # (V, 3, 4) float32: each Training_Image's Exposure_Fix [A | b]
    steps_run: int  # the Iterations run
    sh_degree: int  # the last Iteration's SH_Degree, 0 when none ran


class Gpu(Protocol):
    """What the step and the Train_Smoke_Check call on the GPU_Module, or on a GPU_Stand_In."""

    def probe(self) -> GpuInfo:
        """The GPU check's facts. peak_memory_bytes counts from here."""
        ...

    def nearest_distances(self, points: np.ndarray, neighbours: int,
                          chunk_pairs: int) -> np.ndarray:
        """Each of the (N, 3) points' distances to its `neighbours` nearest other points, as an
        (N, neighbours) array in ascending order, `chunk_pairs` point pairs at a time (Req 3.13).
        """
        ...

    def train(self, scene: Scene, start: Splats, settings: TrainSettings,
              stop: datetime | None) -> TrainResult:
        """The Trainer (Req 4.17): trains the Start_Splats `start` on the scene, and starts no
        Iteration at or after the Stop_Time `stop`, or runs every Planned_Iteration for None."""
        ...

    def peak_memory_bytes(self) -> int:
        """The most GPU memory torch held at once since probe(), in bytes (Req 10.1)."""
        ...


class GpuOutOfMemory(Exception):
    """torch ran out of GPU memory, at the Iteration `iteration`, or before the first for None.

    The GPU_Module raises it in place of torch.cuda.OutOfMemoryError, so the CPU side catches it
    with no torch. Its own short message reads well even where nothing turns it into the
    Req 2.12 failure.
    """

    iteration: int | None  # the Iteration reached, None before the first

    def __init__(self, iteration: int | None) -> None:
        where = "before training started" if iteration is None else f"at iteration {iteration:,}"
        super().__init__(f"torch ran out of GPU memory {where}")
        self.iteration = iteration


class NonFiniteLoss(RuntimeError):
    """An Iteration's loss isn't finite, which is a bug in the training (Req 4.16, A7).

    Its message is non_finite_message's. It isn't a PipelineError, so the Step_Runner reports it
    as an Unexpected_Error, with the message in the Technical_Detail.
    """


def out_of_memory_failure(error: GpuOutOfMemory, preset: str, max_splats: int) -> ExpectedFailure:
    """The Req 2.12 failure for `error`: that the GPU ran out of memory, at which Iteration or
    while the step set up the scene, the Preset's max_splats, and what to do."""
    if error.iteration is None:
        when = "while the step set up the scene"
    else:
        when = f"at iteration {error.iteration:,} of training"
    return ExpectedFailure(f"The GPU ran out of memory {when}. The {preset} preset trains up to "
                           f"{max_splats:,} splats. Run the job with a preset that has a smaller "
                           "splat budget, or on a machine with a GPU that has more memory.")


def non_finite_message(iteration: int, terms: Mapping[str, float]) -> str:
    """NonFiniteLoss's message: the Iteration and each value of `terms`, in order (Req 4.16).

    `terms` holds the own value, before its weight, of every Loss_Term of the run's Loss_Table,
    then the total: the five splat-training terms, like {"l1": 0.0731, ..., "erank": nan,
    "total": nan}, and with a Prior_Set the five Prior_Terms after them (depth-priors spec,
    Req 13.16). Each value is shown to 3 significant figures.
    """
    values = ", ".join(f"{name} {value:.3g}" for name, value in terms.items())
    return (f"The loss stopped being finite at iteration {iteration:,}: {values}. This is a bug "
            "in the training, not a problem with your video.")


def gpu_need_bytes(min_gpu_memory_gb: int, slack: float) -> float:
    """The GPU_Need, slack × min_gpu_memory_gb × GB bytes (Req 2.6)."""
    return slack * min_gpu_memory_gb * GB


def gpu_fits(total_bytes: int, min_gpu_memory_gb: int, slack: float) -> bool:
    """The GPU_Fit_Check: whether a GPU of `total_bytes` holds at least the GPU_Need (Req 2.6).

    A pure function of its three numbers, so it's tested with no GPU (Req 2.7).
    """
    return total_bytes >= gpu_need_bytes(min_gpu_memory_gb, slack)


def check_gpu(info: GpuInfo, settings: TrainSettings, preset: str) -> None:
    """Refuse a machine whose GPU the step can't train on, from the probe's facts `info`.

    In this order: no CUDA (Req 2.3), a gsplat other than GSPLAT_VERSION (Req 2.5), and a GPU
    whose total memory gpu_fits refuses for the Preset's min_gpu_memory_gb and gpu_memory_slack
    (Req 2.4, 2.10). It checks nothing else about the machine: min_vcpus and min_memory_gb are
    left to the cloud (Req 2.11).
    """
    if not info.cuda:
        raise ExpectedFailure(_NO_CUDA)
    if info.gsplat_version != GSPLAT_VERSION:
        raise ExpectedFailure(f"This machine has gsplat {info.gsplat_version}, but the train step "
                              f"is written for gsplat {GSPLAT_VERSION}. Rebuild the machine "
                              "image.")
    size, slack = settings.min_gpu_memory_gb, settings.gpu_memory_slack
    if not gpu_fits(info.total_memory_bytes, size, slack):
        raise ExpectedFailure(
            f"This GPU has {info.total_memory_bytes / GB:.1f} GB of memory, but the {preset} "
            f"preset needs a {size} GB GPU, and the step accepts {slack} of that, "
            f"{gpu_need_bytes(size, slack) / GB:.1f} GB, because a card reports a little less "
            "than its size. Run the job with a preset that needs less GPU memory, or on a "
            "machine with a larger GPU.")


def load_gpu() -> Gpu:
    """The GPU_Module, imported by name when the step runs (Req 2.2).

    The import goes through this module's own importlib, so a test can replace
    splatpipe.train.gpu_api.importlib. A ModuleNotFoundError whose missing module, up to its
    first dot, is in GPU_PACKAGES becomes the Req 2.2 Expected_Failure, naming that package. Any
    other error goes up as it is.
    """
    try:
        return cast(Gpu, importlib.import_module(GPU_MODULE))
    except ModuleNotFoundError as error:
        package = (error.name or "").partition(".")[0]
        if package not in GPU_PACKAGES:
            raise
        raise ExpectedFailure(f"The train step runs only on the GPU machine image, which has "
                              f"torch, gsplat and fused-ssim, and this machine has no {package}. "
                              f"Run the job there with {_CLOUD_START}.") from error
