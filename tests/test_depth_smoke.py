"""Tests for splatpipe/depth/smoke.py: the Depth_Smoke_Check's flow on the CPU side (depth-priors
spec, Req 16.3, 16.4, 16.7–16.9, 18.11).

Every test takes the A11 numbers and depth_normal_start_step from the real config's
[cloud.smoke], and the [train] and [depth] settings from the real config, all read by path,
because tests/conftest.py points CONFIG_PATH at a test config. depth_sizes builds the
DepthSmokeSizes from them as the smoke runner's depth check does. With them the
Prior_Smoke_Scene is the training check's 4 images of 64 x 64 pixels and 200 points, each image
with a grey lower half and a made-up Prior, and the check trains with the fast Preset's settings,
30 Iterations, a Splat_Budget of 400 and a Normal_Start of 8.

check_moge_output runs on made-up MoGe_Outputs: moge_output is a good one, whose mask is false at
OFF_MASK pixels with MoGe's +inf depth and 0 normal there, and each refusal changes one array.

The run tests give run_smoke_depth tests/conftest.py's FakeDepthGpu, the Depth_Stand_In, which
gives moge_output for the made-up image, and FakeGpu, the GPU_Stand_In, whose fake Trainer gives
back the Start_Splats it was given. check_machine needs moge_weights to be a file, so
real_config points it at a stand-in Weights_File in tmp_path, which the stand-in never reads.
Each failure test changes one thing: MoGe's output, the stand-in's Iterations, or backproject_max
in the real config's [train]. Each failure is a PipelineError, whose message the smoke runner
prints as it is, so depth_failure checks that the message names what failed and holds no
traceback (Req 16.9).
"""

from __future__ import annotations

import tomllib
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from splatpipe.config import REPO_ROOT, Config, load_config
from splatpipe.depth.gpu_api import MoGeOutput
from splatpipe.depth.smoke import (SMOKE_FRAME, DepthSmokeSizes, check_moge_output,
                                   prior_smoke_scene, run_smoke_depth)
from splatpipe.errors import PipelineError
from splatpipe.train.backproject import start_points
from splatpipe.train.colmap import SparseModel
from splatpipe.train.ply import check_ply
from splatpipe.train.prior_views import plain_pixels
from splatpipe.train.priors import PRIOR_FOLDER, PRIOR_INDEX, PriorSet, read_prior_set
from splatpipe.train.scene import ImagePixels, Scene, build_scene, training_size
from splatpipe.train.schedules import PRIOR_TERMS, active_terms
from splatpipe.train.settings import TrainSettings
from splatpipe.train.smoke import SmokeSizes, smoke_scene, smoke_settings

from conftest import FakeDepthGpu, FakeGpu

# Read by path, so these tests still read it once CONFIG_PATH points at a test config.
REAL_CONFIG = REPO_ROOT / "config.toml"
SIDE = 64  # [cloud.smoke].image_pixels, which test_the_a11_numbers_are_the_real_configs checks
HALF = SIDE // 2  # the first row of each image's grey lower half
GREY = 128  # the grey of each image's lower half, in every 8-bit RGB channel
OFF_MASK = ((0, 0), (0, 1), (5, 7), (63, 63))  # (row, column): where moge_output finds nothing
# What every refusal of MoGe-2's output says to do next.
CHECK_MOGE = ("Check the depth step's MoGe-2 call in splatpipe/depth/gpu.py and the image "
              "build's MoGe step, then run the smoke test again.")


def real_toml() -> dict[str, Any]:
    """The real config as tomllib reads it."""
    with REAL_CONFIG.open("rb") as file:
        return tomllib.load(file)


def depth_sizes() -> DepthSmokeSizes:
    """The real config's [cloud.smoke] numbers, as the smoke runner's depth check passes them
    in: the training check's eight and depth_normal_start_step."""
    smoke = real_toml()["cloud"]["smoke"]
    train = SmokeSizes(images=smoke["train_images"], points=smoke["train_points"],
                       pixels=smoke["image_pixels"], splats=smoke["train_splats"],
                       steps=smoke["train_steps"],
                       refine_start_step=smoke["train_refine_start_step"],
                       refine_every=smoke["train_refine_every"],
                       sh_degree_interval=smoke["train_sh_degree_interval"])
    return DepthSmokeSizes(train=train, normal_start=smoke["depth_normal_start_step"])


def real_config(weights: Path, **train: Any) -> Config:
    """The real config with [depth]'s moge_weights at `weights` and each [train] key in `train`
    set to its value, like backproject_max=0."""
    data = real_toml()
    data["depth"]["moge_weights"] = str(weights)
    data["train"].update(train)
    return Config(data, source=str(REAL_CONFIG))


def depth_settings() -> TrainSettings:
    """The settings the depth check trains with: the training check's, with the real
    depth_normal_start_step as normal_start_step."""
    sizes = depth_sizes()
    _, settings = smoke_settings(load_config(REAL_CONFIG), sizes.train,
                                 normal_start_step=sizes.normal_start)
    return settings


def scene_with_priors(
        folder: Path) -> tuple[SparseModel, list[ImagePixels], PriorSet, Scene, TrainSettings]:
    """The Prior_Smoke_Scene with its Priors in `folder`, and the Train_Step's Scene of it with
    the Prior_Set, built with the depth check's settings."""
    settings = depth_settings()
    model, pixels, priors = prior_smoke_scene(depth_sizes().train, settings.seed, folder)
    width, height = training_size(model.camera.width, model.camera.height, settings.long_edge)
    return model, pixels, priors, build_scene(model, pixels, width, height, settings,
                                              priors), settings


def moge_output(side: int = SIDE) -> MoGeOutput:
    """A good MoGe_Output for a `side` x `side` image: depth 2 and the normal (0, 0, −1) where
    the mask is true, and, at the OFF_MASK pixels, a false mask with MoGe's +inf depth and 0
    normal."""
    depth = np.full((side, side), 2.0, dtype=np.float32)
    normal = np.zeros((side, side, 3), dtype=np.float32)
    normal[...] = (0.0, 0.0, -1.0)
    mask = np.ones((side, side), dtype=bool)
    for row, column in OFF_MASK:
        depth[row, column] = np.inf
        normal[row, column] = 0.0
        mask[row, column] = False
    return MoGeOutput(depth=depth, normal=normal, mask=mask)


@pytest.fixture
def weights(tmp_path: Path) -> Path:
    """A stand-in Weights_File at tmp_path / "weights" / "model.pt", outside the scratch folder.
    FakeDepthGpu never reads it: check_machine only checks that it's a file (Req 3.5)."""
    path = tmp_path / "weights" / "model.pt"
    path.parent.mkdir()
    path.write_bytes(b"not really a checkpoint")
    return path


@pytest.fixture
def scratch(tmp_path: Path) -> Path:
    """The check's scratch folder, tmp_path / "scratch", made as the smoke runner makes it."""
    path = tmp_path / "scratch"
    path.mkdir()
    return path


def test_the_a11_numbers_are_the_real_configs() -> None:
    """The sizes the other tests rely on: 4 images of 64 x 64, 200 points, 400 splats, 30
    Iterations and a Normal_Start of 8."""
    sizes = depth_sizes()
    assert (sizes.train.images, sizes.train.points, sizes.train.pixels, sizes.train.splats,
            sizes.train.steps, sizes.normal_start) == (4, 200, SIDE, 400, 30, 8)


# check_moge_output (Req 16.3)

def test_a_good_moge_output_passes_and_gives_its_geometry_pixels() -> None:
    """A +inf depth and a 0 normal where the mask is false pass, and those pixels aren't
    Geometry_Pixels."""
    assert check_moge_output(moge_output(), SIDE) == SIDE * SIDE - len(OFF_MASK)


def _nan_normal() -> MoGeOutput:
    """moge_output with a NaN in the normal of a pixel whose mask is true."""
    output = moge_output()
    normal = output.normal.copy()
    normal[10, 20, 1] = np.nan
    return replace(output, normal=normal)


def _zero_depth() -> MoGeOutput:
    """moge_output with a depth of 0 at a pixel whose mask is true."""
    output = moge_output()
    depth = output.depth.copy()
    depth[30, 40] = 0.0
    return replace(output, depth=depth)


# Each wrong MoGe_Output and the whole message that refuses it.
WRONG_OUTPUTS = [
    pytest.param(lambda: replace(moge_output(), depth=moge_output().depth[:, :-1]),
                 f"MoGe-2's depth is (64, 63), but the image is 64 x 64, so it must be (64, 64). "
                 f"{CHECK_MOGE}", id="a wrong shape"),
    pytest.param(lambda: replace(moge_output(), mask=moge_output().mask.astype(np.uint8)),
                 f"MoGe-2's mask is uint8, but it must be bool. {CHECK_MOGE}",
                 id="a wrong dtype"),
    pytest.param(_nan_normal,
                 "MoGe-2's normal holds a NaN where its mask is true, but it must be finite "
                 f"there. {CHECK_MOGE}", id="a NaN normal"),
    pytest.param(_zero_depth,
                 "MoGe-2's depth holds 0.0 where its mask is true, but it must be above 0 there. "
                 f"{CHECK_MOGE}", id="a depth of 0"),
]


@pytest.mark.parametrize(("make", "message"), WRONG_OUTPUTS)
def test_a_wrong_moge_output_is_refused_saying_which_array_and_how(make: Any,
                                                                   message: str) -> None:
    with pytest.raises(PipelineError) as caught:
        check_moge_output(make(), SIDE)
    assert str(caught.value) == message


# The Prior_Smoke_Scene (Req 16.4, 18.11)

def test_the_prior_smoke_scene_passes_the_train_steps_own_prior_checks(tmp_path: Path) -> None:
    folder = tmp_path / PRIOR_FOLDER
    model, _, priors, scene, _ = scene_with_priors(folder)
    again = read_prior_set(folder, model)  # the Train_Step's own reader and checks, once more
    assert again is not None
    names = [image.name for image in model.images]
    assert list(priors.priors) == list(again.priors) == names
    camera = model.camera
    assert again.index.camera == (camera.width, camera.height, camera.fx, camera.fy, camera.cx,
                                  camera.cy)
    assert (again.index.prior_width, again.index.prior_height) == (SIDE, SIDE)
    for name in names:
        prior = again.priors[name]
        assert prior.inverse_depth.dtype == np.float32 and np.all(prior.inverse_depth == 1.0)
        assert prior.normal.dtype == np.float32 and np.all(prior.normal == (0.0, 0.0, -1.0))
        assert prior.mask.dtype == np.bool_ and prior.mask.all()
    # The Scene takes every Prior, so every view is a Prior_View.
    assert scene.taps is not None
    assert all(view.prior is not None for view in scene.views)


def test_the_prior_smoke_scenes_images_have_plain_lower_halves(tmp_path: Path) -> None:
    model, pixels, _, scene, settings = scene_with_priors(tmp_path / PRIOR_FOLDER)
    _, drawn = smoke_scene(depth_sizes().train, settings.seed)
    for image, original in zip(pixels, drawn, strict=True):
        assert np.all(image.pixels[HALF:] == GREY)
        assert np.array_equal(image.pixels[:HALF], original.pixels[:HALF])
        assert (image.keep is None) == (original.keep is None)
        assert image.keep is None or np.array_equal(image.keep, original.keep)
    # Below the row where the grey meets the drawn pixels, every pixel is a Plain_Pixel by the
    # Train_Step's own rule.
    for view in scene.views:
        assert view.prior is not None
        assert plain_pixels(view.prior.gradient, settings)[HALF + 1:].all()
    assert len(model.images) == len(pixels) == 4


# The [cloud.smoke] numbers turn every Prior_Term on and start Back_Projected_Splats (Req 16.7)

def test_with_the_real_config_every_prior_term_comes_on_in_the_30_iterations() -> None:
    """Every image has a Prior, so every Iteration is on a Prior_View. The Depth_Normal_Start is
    round(7/30 × 30) = 7, and the Normal_Start 8 with normal_every 4."""
    settings = depth_settings()
    on = {name: [i for i in range(settings.steps)
                 if name in active_terms(i, settings, True, True)] for name in PRIOR_TERMS}
    every = list(range(30))
    from_7 = list(range(7, 30))
    assert on == {"depth": every, "normal": [8, 12, 16, 20, 24, 28], "flatten": every,
                  "depth_normal": from_7, "normal_consistency": from_7}


def test_with_the_real_config_133_back_projected_splats_start_with_all_200_sparse_points(
        tmp_path: Path) -> None:
    """The Back_Project_Budget is ⌊400 / 3⌋ = 133, far below the Candidate_Pixels, so 133 start,
    and the 200 Sparse_Points all fit in the 267 places left."""
    model, _, _, scene, settings = scene_with_priors(tmp_path / PRIOR_FOLDER)
    start = start_points(model, scene, settings)
    assert (start.back_projected, start.sparse, len(start.points)) == (133, 200, 333)


# The check's run with the two stand-ins (Req 16.8, 16.9)

def test_the_check_writes_and_checks_train_ply_in_the_scratch_folder_and_gives_its_log_line(
        weights: Path, scratch: Path) -> None:
    depth_gpu = FakeDepthGpu(outputs={SMOKE_FRAME: moge_output()})
    gpu = FakeGpu()
    line = run_smoke_depth(real_config(weights), depth_sizes(), scratch, depth_gpu, gpu)
    assert line == ("MoGe-2 found geometry at 4,092 of 4,096 pixels; trained 30 iterations with "
                    "depth priors, from 133 back-projected and 200 sparse splats, to 333 splats, "
                    "and the PLY check passed")
    # Every splat the stand-in gave back, the 333 Start_Splats, is in the checked file.
    assert gpu.start is not None
    assert check_ply(scratch / "train.ply") == len(gpu.start) == 333
    assert (scratch / PRIOR_FOLDER / PRIOR_INDEX).is_file()
    # MoGe's part: the step's own calls, on the made-up image with W = fx = 64, then released.
    data = real_toml()
    assert [name for name, _ in depth_gpu.calls] == ["probe", "import_moge", "load_model",
                                                     "infer", "release"]
    assert depth_gpu.calls[2] == ("load_model", (weights,))
    [(handle, rgb, fov_x, level, fp16, frame)] = [args for name, args in depth_gpu.calls
                                                  if name == "infer"]
    drawn = np.random.default_rng(data["train"]["seed"]).integers(0, 256, (SIDE, SIDE, 3),
                                                                  dtype=np.uint8)
    assert rgb.dtype == np.uint8 and np.array_equal(rgb, drawn)
    assert fov_x == pytest.approx(53.13010235415598)  # 2·atan(1/2), in degrees
    assert (level, fp16, frame) == (data["depth"]["resolution_level"],
                                    data["depth"]["use_fp16"], SMOKE_FRAME)
    assert depth_gpu.calls[-1] == ("release", (handle,)) and handle is depth_gpu.handle
    # Training: the GPU check first, then the 3-NN search and the Trainer with no Stop_Time, on
    # the Scene with every Prior and with the Normal_Start of 8.
    assert [name for name, _ in gpu.calls] == ["probe", "nearest_distances", "train"]
    scene, _, settings, stop = gpu.calls[2][1]
    assert stop is None
    assert (settings.steps, settings.max_splats, settings.normal_start_step) == (30, 400, 8)
    assert scene.taps is not None and all(view.prior is not None for view in scene.views)


def depth_failure(config: Config, scratch: Path, depth_gpu: FakeDepthGpu, gpu: FakeGpu,
                  *phrases: str) -> None:
    """Run the check and check that it fails with a PipelineError whose message holds each of
    `phrases` and no traceback (Req 16.9), and that it wrote no train.ply."""
    with pytest.raises(PipelineError) as caught:
        run_smoke_depth(config, depth_sizes(), scratch, depth_gpu, gpu)
    message = str(caught.value)
    for phrase in phrases:
        assert phrase in message, message
    assert "Traceback" not in message
    assert not (scratch / "train.ply").exists()


def test_a_wrong_moge_output_fails_the_check_after_release_and_before_any_training(
        weights: Path, scratch: Path) -> None:
    wrong = replace(moge_output(), depth=moge_output().depth.astype(np.float64))
    depth_gpu = FakeDepthGpu(outputs={SMOKE_FRAME: wrong})
    gpu = FakeGpu()
    depth_failure(real_config(weights), scratch, depth_gpu, gpu,
                  "MoGe-2's depth is float64, but it must be float32", CHECK_MOGE)
    assert depth_gpu.calls[-1] == ("release", (depth_gpu.handle,))
    assert gpu.calls == []


def test_a_trainer_that_runs_fewer_iterations_fails_the_check_saying_how_many_ran(
        weights: Path, scratch: Path) -> None:
    depth_failure(real_config(weights), scratch,
                  FakeDepthGpu(outputs={SMOKE_FRAME: moge_output()}), FakeGpu(stop_after=29),
                  "The Trainer ran only 29 of the depth check's 30 iterations")


def test_no_back_projected_splat_fails_the_check_before_anything_goes_on_the_gpu(
        weights: Path, scratch: Path) -> None:
    gpu = FakeGpu()
    depth_failure(real_config(weights, backproject_max=0), scratch,
                  FakeDepthGpu(outputs={SMOKE_FRAME: moge_output()}), gpu,
                  "No splat started from the depth check's made-up depth priors",
                  "backproject_max in [train]")
    assert [name for name, _ in gpu.calls] == ["probe"]
