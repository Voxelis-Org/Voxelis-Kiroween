"""Unit tests for the GPU interface: splatpipe/train/gpu_api.py.

Every test runs on the CPU, with no GPU and no torch, gsplat or fused_ssim. The GPU_Fit_Check is
a pure function of its three numbers, so it's tested with the byte counts that nvidia-smi shows
for an A10G and an L40S (Req 2.6 to 2.9). check_gpu decides on a GpuInfo written here, as the
GPU_Module's probe() would give it, with the test config's [train] settings (Req 2.3, 2.5, 2.10,
2.11).

load_gpu is reached only with splatpipe.train.gpu_api.importlib replaced by a StandInImportlib,
as tests/test_run.py replaces splatpipe.step.importlib. The stand-in records each name it's asked
to import, then raises an error or gives back a module made here, so no test imports
splatpipe.train.gpu, the GPU_Module. Each load_gpu test checks that the name asked for is
GPU_MODULE and that GPU_MODULE is "splatpipe.train.gpu", so a wrong name can't pass every laptop
test and fail only in the Smoke_Test (Req 2.2).

Each message is compared whole with the one the design gives, written out here, so the test
checks both what it names and that it holds no traceback (Req 14.6).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from types import ModuleType

import pytest

from splatpipe.config import load_config
from splatpipe.errors import ExpectedFailure
from splatpipe.train.gpu_api import (GPU_MODULE, GpuInfo, GpuOutOfMemory, check_gpu, gpu_fits,
                                     load_gpu, non_finite_message, out_of_memory_failure)
from splatpipe.train.settings import TrainSettings, read_settings

A10G_BYTES = 24_146_608_128  # the 23,028 MiB that nvidia-smi shows for a "24 GB" A10G (Req 2.8)
L40S_BYTES = 48_305_799_168  # the 46,068 MiB that nvidia-smi shows for an L40S (Req 2.9)
SLACK = 0.9  # the gpu_memory_slack of Req 2.8 and 2.9

# check_gpu's Readable_Errors, as the design's Error Handling table words them. SMALL_GPU is the
# one for an A10G and the quality preset's 48 GB, with a slack of 0.9.
NO_CUDA = ("The train step needs an NVIDIA GPU, but torch can't see one on this machine. Check "
           "the driver with nvidia-smi, or run the job with `uv run python -m cloud.start`.")
OLD_GSPLAT = ("This machine has gsplat 1.5.2, but the train step is written for gsplat 1.5.3. "
              "Rebuild the machine image.")
SMALL_GPU = ("This GPU has 24.1 GB of memory, but the quality preset needs a 48 GB GPU, and the "
             "step accepts 0.9 of that, 43.2 GB, because a card reports a little less than its "
             "size. Run the job with a preset that needs less GPU memory, or on a machine with a "
             "larger GPU.")
# load_gpu's Req 2.2 Readable_Error, for the missing package.
NO_PACKAGE = ("The train step runs only on the GPU machine image, which has torch, gsplat and "
              "fused-ssim, and this machine has no {package}. Run the job there with "
              "`uv run python -m cloud.start`.")


@dataclass
class StandInImportlib:
    """A stand-in for gpu_api's importlib. Its import_module records each name it's asked to
    import in `names`, then raises `result` when that's an exception, or gives it back."""

    result: ModuleType | Exception
    names: list[str] = field(default_factory=list)

    def import_module(self, name: str) -> ModuleType:
        """Record `name`, then raise or give back `result`."""
        self.names.append(name)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def install_stand_in(monkeypatch: pytest.MonkeyPatch,
                     result: ModuleType | Exception) -> StandInImportlib:
    """A StandInImportlib for `result`, in place of splatpipe.train.gpu_api.importlib for this
    test."""
    stand_in = StandInImportlib(result)
    monkeypatch.setattr("splatpipe.train.gpu_api.importlib", stand_in)
    return stand_in


def check_asked_for_the_gpu_module(stand_in: StandInImportlib) -> None:
    """Check that load_gpu asked `stand_in` for GPU_MODULE alone, and that GPU_MODULE is the
    GPU_Module's real name."""
    assert stand_in.names == [GPU_MODULE]
    assert GPU_MODULE == "splatpipe.train.gpu"


def train_settings(preset: str) -> TrainSettings:
    """The test config's [train] settings for `preset`, read as the step reads them."""
    return read_settings(load_config().step_settings("train", preset), "train", preset)


def refusal(function: Callable[..., object], *args: object) -> str:
    """The message of the ExpectedFailure that function(*args) raises."""
    with pytest.raises(ExpectedFailure) as caught:
        function(*args)
    return str(caught.value)


@pytest.mark.parametrize(("total_bytes", "min_gpu_memory_gb", "fits"), [
    (A10G_BYTES, 24, True),
    (A10G_BYTES, 48, False),
    (L40S_BYTES, 24, True),
    (L40S_BYTES, 48, True),
    (21_600_000_000, 24, True),  # exactly the GPU_Need, 0.9 × 24 × 10⁹ bytes
])
def test_gpu_fit_check_with_a_slack_of_0_9(total_bytes: int, min_gpu_memory_gb: int,
                                           fits: bool) -> None:
    """The GPU_Fit_Check passes a GPU of at least slack × min_gpu_memory_gb × 10⁹ bytes, worked
    out from its three numbers with no GPU: an A10G passes 24 and fails 48, an L40S passes both,
    and exactly the GPU_Need passes (Req 2.6 to 2.9)."""
    assert gpu_fits(total_bytes, min_gpu_memory_gb, SLACK) is fits


@pytest.mark.parametrize(("info", "message"), [
    (GpuInfo(cuda=False, gsplat_version="1.5.2", total_memory_bytes=0), NO_CUDA),
    (GpuInfo(cuda=True, gsplat_version="1.5.2", total_memory_bytes=A10G_BYTES), OLD_GSPLAT),
    (GpuInfo(cuda=True, gsplat_version="1.5.3", total_memory_bytes=A10G_BYTES), SMALL_GPU),
], ids=["no-cuda", "gsplat-1.5.2", "small-gpu"])
def test_check_gpu_refuses_no_cuda_then_another_gsplat_then_a_small_gpu(info: GpuInfo,
                                                                       message: str) -> None:
    """Each GpuInfo has its own fault and every fault checked after it, so each message shows
    which check comes first: no CUDA (Req 2.3), then gsplat 1.5.2 (Req 2.5), then an A10G for
    the quality preset's 48 GB (Req 2.10)."""
    assert refusal(check_gpu, info, train_settings("quality"), "quality") == message


@pytest.mark.parametrize(("vcpus", "memory_gb"), [(1, 1), (1_000_000, 1_000_000)])
def test_check_gpu_passes_a_gpu_that_fits_whatever_min_vcpus_and_min_memory_gb_say(
        vcpus: int, memory_gb: int) -> None:
    """check_gpu checks nothing about the machine but its GPU, so min_vcpus and min_memory_gb,
    from the smallest allowed to more than any machine has, change nothing (Req 2.11)."""
    settings = replace(train_settings("fast"), min_vcpus=vcpus, min_memory_gb=memory_gb)
    info = GpuInfo(cuda=True, gsplat_version="1.5.3", total_memory_bytes=A10G_BYTES)
    assert check_gpu(info, settings, "fast") is None


@pytest.mark.parametrize(("missing", "package"), [
    ("torch", "torch"),
    ("gsplat.strategy", "gsplat"),
    ("fused_ssim", "fused_ssim"),
])
def test_load_gpu_without_a_gpu_package_says_the_step_runs_only_on_the_gpu_image(
        monkeypatch: pytest.MonkeyPatch, missing: str, package: str) -> None:
    """A missing torch, gsplat or fused_ssim, or a submodule of one, is the Req 2.2
    ExpectedFailure naming the package (Req 2.2)."""
    error = ModuleNotFoundError(f"No module named {missing!r}", name=missing)
    stand_in = install_stand_in(monkeypatch, error)
    assert refusal(load_gpu) == NO_PACKAGE.format(package=package)
    check_asked_for_the_gpu_module(stand_in)


def test_load_gpu_lets_a_missing_fused_ssim_cuda_go_up_as_it_is(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """fused_ssim_cuda, the extension a broken image can lack, isn't one of the GPU packages, so
    its error goes up as it is, an Unexpected_Error and never "wrong machine" (Req 2.2)."""
    error = ModuleNotFoundError("No module named 'fused_ssim_cuda'", name="fused_ssim_cuda")
    stand_in = install_stand_in(monkeypatch, error)
    with pytest.raises(ModuleNotFoundError) as caught:
        load_gpu()
    assert caught.value is error
    check_asked_for_the_gpu_module(stand_in)


def test_load_gpu_gives_back_the_module_it_imports(monkeypatch: pytest.MonkeyPatch) -> None:
    """load_gpu gives back the module object that the import gives (Req 2.2)."""
    module = ModuleType("stand_in_gpu")
    stand_in = install_stand_in(monkeypatch, module)
    assert load_gpu() is module
    check_asked_for_the_gpu_module(stand_in)


@pytest.mark.parametrize(("iteration", "preset", "max_splats", "message"), [
    (812, "fast", 1_000_000,
     "The GPU ran out of memory at iteration 812 of training. The fast preset trains up to "
     "1,000,000 splats. Run the job with a preset that has a smaller splat budget, or on a "
     "machine with a GPU that has more memory."),
    (None, "quality", 3_000_000,
     "The GPU ran out of memory while the step set up the scene. The quality preset trains up to "
     "3,000,000 splats. Run the job with a preset that has a smaller splat budget, or on a "
     "machine with a GPU that has more memory."),
], ids=["iteration-812", "scene-set-up"])
def test_out_of_memory_failure_names_where_it_ran_out_the_preset_and_its_max_splats(
        iteration: int | None, preset: str, max_splats: int, message: str) -> None:
    """Running out of GPU memory at Iteration 812, or before the first Iteration while the step
    set up the scene, is the Req 2.12 ExpectedFailure, naming the Preset and its max_splats."""
    failure = out_of_memory_failure(GpuOutOfMemory(iteration), preset, max_splats)
    assert type(failure) is ExpectedFailure
    assert str(failure) == message


def test_non_finite_message_names_the_iteration_each_loss_term_and_the_total() -> None:
    """The message gives the Iteration, the five Loss_Terms' own values and the total, as the
    design's example does (Req 4.16)."""
    terms = {"l1": 0.0731, "ssim": 0.412, "opacity_reg": 0.31, "scale_reg": 0.0204,
             "erank": math.nan, "total": math.nan}
    assert non_finite_message(812, terms) == (
        "The loss stopped being finite at iteration 812: l1 0.0731, ssim 0.412, opacity_reg "
        "0.31, scale_reg 0.0204, erank nan, total nan. This is a bug in the training, not a "
        "problem with your video.")
