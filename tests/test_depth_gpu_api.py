"""Tests for the Depth_GPU_Module's interface on the CPU: splatpipe/depth/gpu_api.py
(depth-priors spec, Req 3.2 to 3.5, 3.11, 3.13, 4.4).

load_depth_gpu is reached only with splatpipe.depth.gpu_api.importlib replaced by
StandInImportlib, so no test imports the Depth_GPU_Module or torch. check_machine, load_moge and
run_moge run on tests/conftest.py's FakeDepthGpu, the Depth_Stand_In, whose call log shows what
each function asked of it and in what order. The settings are the real [depth] values, DEPTH,
checked by read_settings, with a key changed by dataclasses.replace where a test needs it, like
moge_weights pointed into tmp_path.

Each refusal is an ExpectedFailure whose whole message is the one the design's Error Handling
table gives, with moge_weights shown as written and a Frame's name with repr(), and holds no
traceback (Req 18.7).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from splatpipe.depth.gpu_api import (DEPTH_GPU_MODULE, DepthOutOfMemory, MissingPackage,
                                     MoGeOutput, WrongCheckpoint, check_machine, load_depth_gpu,
                                     load_moge, run_moge)
from splatpipe.depth.settings import DepthSettings, read_settings
from splatpipe.errors import ExpectedFailure

from conftest import DEPTH, FakeDepthGpu, FakeMoGe

# The real [depth] settings, checked. Every [depth] setting is shared, so the Preset is any.
SETTINGS = read_settings(DEPTH, "depth", "default")

FRAME = "000002.jpg"  # the Frame a test runs MoGe-2 on
FOV_X = 53.13  # a Field_Of_View in degrees, as the step would give it

# The Readable_Errors of the design's Error Handling table. NO_PACKAGE takes the package's name.
NO_PACKAGE = ("The depth step runs only on a GPU machine image built with MoGe's code, and this "
              "machine has no {package}. Rebuild the machine image with "
              "`uv run python -m cloud.image`, or run the job there with "
              "`uv run python -m cloud.start`.")
NO_CUDA = ("The depth step needs an NVIDIA GPU, but torch can't see one on this machine. Check "
           "the driver with nvidia-smi, or run the job with `uv run python -m cloud.start`.")
NO_WEIGHTS = ("The MoGe-2 weights aren't at {path}, the moge_weights setting in [depth]. The "
              "image build downloads them there, so rebuild the machine image with "
              "`uv run python -m cloud.image`.")
WRONG_CHECKPOINT = ("The file at {path} isn't the MoGe-2 ViT-B normal checkpoint ({reason}). "
                    "Rebuild the machine image with `uv run python -m cloud.image`.")
LOAD_OUT_OF_MEMORY = ("The GPU ran out of memory while the depth step loaded MoGe-2. The job goes "
                      "on and trains without depth priors.")
FRAME_OUT_OF_MEMORY = ("The GPU ran out of memory while the depth step ran MoGe-2 on the frame "
                       "{frame!r}. The job goes on and trains without depth priors.")


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
    """A StandInImportlib for `result`, in place of splatpipe.depth.gpu_api.importlib for this
    test."""
    stand_in = StandInImportlib(result)
    monkeypatch.setattr("splatpipe.depth.gpu_api.importlib", stand_in)
    return stand_in


def refused(call: Callable[[], object]) -> str:
    """The message of the ExpectedFailure that `call` raises, once it's checked that the message
    holds no traceback (Req 18.7)."""
    with pytest.raises(ExpectedFailure) as caught:
        call()
    message = str(caught.value)
    assert "Traceback" not in message
    return message


def with_weights(tmp_path: Path, *, there: bool) -> DepthSettings:
    """The real settings with moge_weights at model.pt in tmp_path, written as a file when
    `there`."""
    weights = tmp_path / "model.pt"
    if there:
        weights.write_bytes(b"not really a checkpoint")
    return replace(SETTINGS, moge_weights=str(weights))


# Req 3.2: load_depth_gpu

def test_load_depth_gpu_imports_the_depth_gpu_module_by_name(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """load_depth_gpu asks importlib for DEPTH_GPU_MODULE alone, which is the Depth_GPU_Module's
    real name, and gives back what it imports."""
    module = ModuleType(DEPTH_GPU_MODULE)
    stand_in = install_stand_in(monkeypatch, module)
    assert load_depth_gpu() is module
    assert stand_in.names == [DEPTH_GPU_MODULE]
    assert DEPTH_GPU_MODULE == "splatpipe.depth.gpu"


@pytest.mark.parametrize("missing", ["torch", "torch.cuda"])
def test_a_missing_torch_is_the_req_3_2_failure_naming_torch(
        monkeypatch: pytest.MonkeyPatch, missing: str) -> None:
    """A ModuleNotFoundError for torch, or a submodule of it, becomes the Req 3.2 failure."""
    stand_in = install_stand_in(monkeypatch, ModuleNotFoundError(f"No module named {missing!r}",
                                                                 name=missing))
    assert refused(load_depth_gpu) == NO_PACKAGE.format(package="torch")
    assert stand_in.names == [DEPTH_GPU_MODULE]


@pytest.mark.parametrize("missing", ["torchvision", "moge", "splatpipe.depth.gpu", None])
def test_another_missing_module_goes_up_as_it_is(monkeypatch: pytest.MonkeyPatch,
                                                 missing: str | None) -> None:
    """A ModuleNotFoundError for any module but torch, torchvision and our own module included,
    or one that names no module, goes up as it is, an Unexpected_Error (Req 3.2)."""
    error = ModuleNotFoundError(f"No module named {missing!r}", name=missing)
    install_stand_in(monkeypatch, error)
    with pytest.raises(ModuleNotFoundError) as caught:
        load_depth_gpu()
    assert caught.value is error


# Req 3.2 to 3.5: check_machine

def test_check_machine_refuses_no_cuda_first(tmp_path: Path) -> None:
    """With no CUDA, no Weights_File and a missing package, check_machine refuses no CUDA, after
    the probe alone (Req 3.3)."""
    gpu = FakeDepthGpu(cuda=False, import_moge_error=MissingPackage("utils3d"))
    assert refused(lambda: check_machine(gpu, with_weights(tmp_path, there=False))) == NO_CUDA
    assert gpu.calls == [("probe", ())]


def test_check_machine_refuses_a_missing_weights_file_before_importing_moge(
        tmp_path: Path) -> None:
    """With CUDA, no Weights_File and a missing package, check_machine refuses the Weights_File,
    with moge_weights as written, and never calls import_moge (Req 3.5)."""
    gpu = FakeDepthGpu(import_moge_error=MissingPackage("utils3d"))
    settings = with_weights(tmp_path, there=False)
    assert refused(lambda: check_machine(gpu, settings)) == NO_WEIGHTS.format(
        path=settings.moge_weights)
    assert gpu.calls == [("probe", ())]


def test_check_machine_refuses_a_missing_package_naming_it(tmp_path: Path) -> None:
    """With CUDA and the Weights_File, a MissingPackage from import_moge is the Req 3.2 failure
    naming the package (Req 3.2)."""
    gpu = FakeDepthGpu(import_moge_error=MissingPackage("utils3d"))
    assert refused(lambda: check_machine(gpu, with_weights(tmp_path, there=True))) == (
        NO_PACKAGE.format(package="utils3d"))
    assert gpu.calls == [("probe", ()), ("import_moge", ())]


def test_check_machine_checks_no_gpu_memory(tmp_path: Path) -> None:
    """A min_gpu_memory_gb no GPU has passes, and check_machine asks only for the probe and the
    import, so it checks no amount of GPU memory (Req 3.4)."""
    gpu = FakeDepthGpu()
    settings = replace(with_weights(tmp_path, there=True), min_gpu_memory_gb=10**9)
    check_machine(gpu, settings)
    assert gpu.calls == [("probe", ()), ("import_moge", ())]


# Req 3.11, 3.13: load_moge

def test_load_moge_loads_the_weights_file_and_gives_back_the_handle(tmp_path: Path) -> None:
    gpu = FakeDepthGpu()
    settings = with_weights(tmp_path, there=True)
    handle = load_moge(gpu, settings)
    assert isinstance(handle, FakeMoGe) and handle is gpu.handle
    assert gpu.calls == [("load_model", (Path(settings.moge_weights),))]


@pytest.mark.parametrize(("error", "message"), [
    pytest.param(WrongCheckpoint("it holds no model_config"),
                 WRONG_CHECKPOINT.format(path="{path}", reason="it holds no model_config"),
                 id="a wrong checkpoint"),
    pytest.param(DepthOutOfMemory(None), LOAD_OUT_OF_MEMORY, id="out of memory"),
])
def test_load_moge_turns_each_load_error_into_its_message(tmp_path: Path, error: Exception,
                                                          message: str) -> None:
    """WrongCheckpoint is the Req 3.11 failure, with moge_weights as written and the reason, and
    DepthOutOfMemory(None) the Req 3.13 failure for the load."""
    gpu = FakeDepthGpu(load_model_error=error)
    settings = with_weights(tmp_path, there=True)
    assert refused(lambda: load_moge(gpu, settings)) == message.format(
        path=settings.moge_weights)


# Req 3.13, 4.4: run_moge

def moge_output() -> MoGeOutput:
    """A MoGe_Output of 3 x 2 pixels, every pixel 1.5 metres away and facing the camera."""
    return MoGeOutput(depth=np.full((2, 3), 1.5, dtype=np.float32),
                      normal=np.tile(np.array([0.0, 0.0, -1.0], dtype=np.float32), (2, 3, 1)),
                      mask=np.ones((2, 3), dtype=bool))


RGB = np.zeros((2, 3, 3), dtype=np.uint8)  # a black Frame of 3 x 2 pixels


@pytest.mark.parametrize(("resolution_level", "use_fp16"), [(9, True), (3, False)])
def test_run_moge_passes_resolution_level_and_use_fp16_from_the_settings(
        resolution_level: int, use_fp16: bool) -> None:
    """run_moge gives infer the handle, the Frame, the Field_Of_View, resolution_level and
    use_fp16 from [depth] and the Frame's name, and gives back what infer gave (Req 4.4)."""
    output = moge_output()
    gpu = FakeDepthGpu(outputs={FRAME: output})
    model = FakeMoGe(Path(SETTINGS.moge_weights))
    settings = replace(SETTINGS, resolution_level=resolution_level, use_fp16=use_fp16)
    assert run_moge(gpu, model, RGB, FOV_X, settings, FRAME) is output
    [(name, args)] = gpu.calls
    assert name == "infer"
    assert args[0] is model and args[1] is RGB
    assert args[2:] == (FOV_X, resolution_level, use_fp16, FRAME)


def test_run_moge_turns_out_of_memory_into_the_req_3_13_failure_naming_the_frame() -> None:
    gpu = FakeDepthGpu(out_of_memory_at=1)
    model = FakeMoGe(Path(SETTINGS.moge_weights))
    assert refused(lambda: run_moge(gpu, model, RGB, FOV_X, SETTINGS, FRAME)) == (
        FRAME_OUT_OF_MEMORY.format(frame=FRAME))
