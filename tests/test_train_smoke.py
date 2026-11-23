"""Tests for splatpipe/train/smoke.py: the Train_Smoke_Check's flow on the CPU side (training
spec, Req 12.2, 12.3, 12.6–12.9, 14.10).

Every test takes the A11 numbers from the real config's [cloud.smoke] and the [train] settings
from the real config, both read by path, because tests/conftest.py points CONFIG_PATH at a test
config. With them the Smoke_Scene is 4 images of 64 x 64 pixels and 200 points, and the check
trains with the fast Preset's settings and 30 Iterations, a Splat_Budget of 400, refines from
Iteration 10 every 10 Iterations, and a rise of the SH_Degree every 5 Iterations.

The scene tests run the step's own checks on the Smoke_Scene, need_points, then training_size
and build_scene, and work out each point's depth in front of each camera from the scene's own
world-to-camera matrices. The Schedules test runs schedules.py on the settings smoke_settings
gives.

fake_gpu builds a FakeGpu, the GPU_Stand_In, which the tests pass to run_smoke_training as its
Gpu. It passes the GPU check as a "24 GB" A10G, finds the nearest points by brute force, and its
fake Trainer gives back the Start_Splats it was given, with the Exposure_Fix [I | 0] for every
view, so the check writes the 200 Start_Splats to the scratch folder, tmp_path here. Each
failure test changes one thing about the stand-in: it stops early, runs out of GPU memory in one
of its two GPU calls, gives back one splat with a NaN opacity, or reports gsplat 1.5.2. Each
failure is a PipelineError, whose message the smoke runner prints as it is, so smoke_failure
checks that the message names what failed and holds no traceback (Req 12.9).
"""

import tomllib
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from splatpipe.config import REPO_ROOT, load_config
from splatpipe.errors import PipelineError
from splatpipe.train.gpu_api import Gpu, GpuInfo, GpuOutOfMemory
from splatpipe.train.ply import check_ply, read_ply
from splatpipe.train.scene import build_scene, training_size
from splatpipe.train.schedules import erank_start, refine_iterations, refine_stop, sh_degree
from splatpipe.train.smoke import SmokeSizes, run_smoke_training, smoke_scene, smoke_settings
from splatpipe.train.splats import need_points

# Read by path, so these tests still read it once CONFIG_PATH points at a test config.
REAL_CONFIG = REPO_ROOT / "config.toml"
A10G_BYTES = 24_146_608_128  # FakeGpu's GPU memory by default, a "24 GB" A10G (Req 2.8)


def real_toml() -> dict[str, Any]:
    """The real config as tomllib reads it."""
    with REAL_CONFIG.open("rb") as file:
        return tomllib.load(file)


def a11_sizes() -> SmokeSizes:
    """The A11 numbers: the real config's [cloud.smoke] values, as the smoke runner's training
    check passes them in."""
    smoke = real_toml()["cloud"]["smoke"]
    return SmokeSizes(images=smoke["train_images"], points=smoke["train_points"],
                      pixels=smoke["image_pixels"], splats=smoke["train_splats"],
                      steps=smoke["train_steps"],
                      refine_start_step=smoke["train_refine_start_step"],
                      refine_every=smoke["train_refine_every"],
                      sh_degree_interval=smoke["train_sh_degree_interval"])


# The Smoke_Scene (Req 12.2)

def test_the_smoke_scene_passes_the_steps_own_checks_with_every_point_in_front_of_every_camera(
        ) -> None:
    _, settings = smoke_settings(load_config(REAL_CONFIG), a11_sizes())
    model, pixels = smoke_scene(a11_sizes(), settings.seed)
    need_points(len(model.points), settings.start_scale_neighbours)  # Req 1.18
    width, height = training_size(model.camera.width, model.camera.height, settings.long_edge)
    scene = build_scene(model, pixels, width, height, settings)  # passes Req 3.7 and 3.9
    assert (len(scene.views), scene.width, scene.height, len(model.points)) == (4, 64, 64, 200)
    assert scene.extent == pytest.approx(1.1)  # scene_extent_factor × the circle's radius of 1
    for view in scene.views:
        depths = model.points @ view.world_to_camera[2, :3] + view.world_to_camera[2, 3]
        assert np.all((depths > 0.64) & (depths < 1.36))  # 1 ± 0.25·√2 in front of the camera
    # One image has a mask, and it keeps some of its pixels and masks the others.
    masks = [view.keep for view in scene.views if view.keep is not None]
    assert len(masks) == 1
    assert masks[0].any() and not masks[0].all()


def test_the_same_seed_gives_the_same_smoke_scene() -> None:
    seed = real_toml()["train"]["seed"]
    (model, pixels), (again, pixels_again) = (smoke_scene(a11_sizes(), seed),
                                              smoke_scene(a11_sizes(), seed))
    assert model.camera == again.camera
    for image, image_again in zip(model.images, again.images, strict=True):
        assert ((image.image_id, image.camera_id, image.name)
                == (image_again.image_id, image_again.camera_id, image_again.name))
        assert np.array_equal(image.quaternion, image_again.quaternion)
        assert np.array_equal(image.translation, image_again.translation)
    assert np.array_equal(model.points, again.points)
    assert np.array_equal(model.colours, again.colours)
    for one, other in zip(pixels, pixels_again, strict=True):
        assert np.array_equal(one.pixels, other.pixels)
        assert (one.keep is None) == (other.keep is None)
        assert one.keep is None or np.array_equal(one.keep, other.keep)


# The settings and the Schedules (Req 12.3, 12.6)

def test_smoke_settings_are_the_default_presets_train_settings_with_the_five_smoke_values(
        ) -> None:
    preset, settings = smoke_settings(load_config(REAL_CONFIG), a11_sizes())
    data = real_toml()
    assert preset == data["jobs"]["default_preset"] == "fast"
    train = data["train"]
    shared = {key: value for key, value in train.items() if not isinstance(value, dict)}
    smoke = {"steps": 30, "max_splats": 400, "refine_start_step": 10, "refine_every": 10,
             "sh_degree_interval": 5}  # the A11 numbers (Req 12.3, 12.4)
    assert asdict(settings) == {**shared, **train["fast"], **smoke}


def test_with_the_a11_numbers_erank_starts_at_7_sh_degree_3_at_15_and_the_only_refine_at_20(
        ) -> None:
    _, settings = smoke_settings(load_config(REAL_CONFIG), a11_sizes())
    planned = settings.steps
    assert erank_start(planned, settings.erank_start_fraction) == 7  # round(7/30 × 30)
    degrees = [sh_degree(i, settings.sh_degree_interval) for i in range(planned)]
    assert degrees.index(3) == 15  # the first Iteration at SH_Degree 3
    stop = refine_stop(planned, settings.refine_stop_step, settings.refine_stop_fraction)
    assert refine_iterations(planned, settings.refine_start_step, settings.refine_every,
                             stop) == [20]


# The check's run with the GPU_Stand_In (Req 12.7–12.9)

def test_the_check_writes_and_checks_train_ply_in_the_scratch_folder_and_gives_its_log_line(
        tmp_path: Path, fake_gpu: Callable[..., Any]) -> None:
    gpu = fake_gpu()
    line = run_smoke_training(load_config(REAL_CONFIG), a11_sizes(), tmp_path, gpu)
    assert line == ("trained 30 iterations, with 1 refine, to 200 splats at SH degree 3, and the "
                    "PLY check passed")
    # Every splat the stand-in gave back, the 200 Start_Splats, is in the checked file.
    assert check_ply(tmp_path / "train.ply") == len(gpu.start) == 200


def smoke_failure(gpu: Gpu, scratch: Path, *phrases: str) -> None:
    """Run the check with the stand-in `gpu` and check that it fails with a PipelineError whose
    message holds each of `phrases` and no traceback (Req 12.9)."""
    with pytest.raises(PipelineError) as caught:
        run_smoke_training(load_config(REAL_CONFIG), a11_sizes(), scratch, gpu)
    message = str(caught.value)
    for phrase in phrases:
        assert phrase in message, message
    assert "Traceback" not in message


def test_a_trainer_that_runs_fewer_iterations_fails_the_check_saying_how_many_ran(
        tmp_path: Path, fake_gpu: Callable[..., Any]) -> None:
    smoke_failure(fake_gpu(stop_after=29), tmp_path,
                  "The Trainer ran only 29 of the training check's 30 iterations")  # Req 12.8


# Where the stand-in runs out of GPU memory: the GpuOutOfMemory that the call raises, by its
# FakeGpu option and the Iteration it gives, and where the message says it ran out (Req 2.12).
OUT_OF_MEMORY = [
    pytest.param("nearest_distances_error", None, "while the step set up the scene",
                 id="in the 3-NN search"),
    pytest.param("train_error", 4, "at iteration 4 of training", id="in training"),
]


@pytest.mark.parametrize(("call", "iteration", "where"), OUT_OF_MEMORY)
def test_running_out_of_gpu_memory_fails_the_check_with_the_steps_message(
        tmp_path: Path, fake_gpu: Callable[..., Any], call: str, iteration: int | None,
        where: str) -> None:
    smoke_failure(fake_gpu(**{call: GpuOutOfMemory(iteration)}), tmp_path,
                  f"The GPU ran out of memory {where}", "The fast preset trains up to 400 splats")


def test_a_nan_splat_is_written_not_dropped_so_the_check_fails_at_its_ply_check(
        tmp_path: Path, fake_gpu: Callable[..., Any]) -> None:
    written = tmp_path / "train.ply"
    smoke_failure(fake_gpu(nan_opacity="one"), tmp_path,
                  f"In {written}, 1 splat holds a NaN or an infinity: opacity (1)")  # Req 12.7
    opacity = read_ply(written)["opacity"]  # read with no check
    assert (len(opacity), int(np.isnan(opacity).sum())) == (200, 1)


def test_a_stand_in_that_reports_gsplat_1_5_2_fails_the_gpu_check_before_any_other_gpu_call(
        tmp_path: Path, fake_gpu: Callable[..., Any]) -> None:
    gpu = fake_gpu(GpuInfo(cuda=True, gsplat_version="1.5.2", total_memory_bytes=A10G_BYTES))
    smoke_failure(gpu, tmp_path, "This machine has gsplat 1.5.2",
                  "written for gsplat 1.5.3", "Rebuild the machine image")  # Req 2.5
    assert [name for name, _ in gpu.calls] == ["probe"]
