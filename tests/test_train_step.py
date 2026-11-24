"""Tests for the Train_Step in splatpipe/train/, run through splatpipe.train.run, so the
Step_Runner runs it and writes its real report (training spec, Req 14.4).

Each test makes its Job with new_job from tests/conftest.py, in the test config's Jobs_Folder,
then writes the default Camera_Solve_Output into the Job's work/cameras/ with
write_camera_solve_output from tests/camera_files.py: one 8 x 6 PINHOLE camera, 3 images from
cameras on an arc of the unit circle, each turned about y to look at the origin, 12 points on a
grid around the origin, the 3 images as JPEGs and a mask on the first image only. The Job's
Preset is fast, which trains 6 Iterations with a Splat_Budget of 50 in the test config, so the
step starts from all 12 points, and the long_edge of 1600 leaves the images at 8 x 6.

fake_gpu installs a FakeGpu, the GPU_Stand_In, as the step's GPU hook. It passes the GPU check as
a "24 GB" A10G, finds the nearest points by brute force, and its fake Trainer gives back the
Start_Splats it was given, as float32, with the Exposure_Fix [I | 0] for every view unless the
test sets one. It keeps the scene and the Start_Splats the step gave it, so a test can check
them. With every fix [I | 0], the Mean_Exposure is [I | 0] and the bake leaves every value as it
is, so out/train.ply holds the Start_Splats exactly. The exposure test gives every view
EXPOSURE, whose values float32 holds exactly, and checks the colour that each written splat
shows, 0.5 + SH_C0 · f_dc with no higher band, against Ā·c + b̄, where c is the colour its
Start_Splat showed (Req 6.8). The second-run test gives its first run EXPOSURE too, so the files
of its two runs differ.

layout_fields gives the 59 fields of the Splat_Layout for some splats, as the training spec's
Splat_Layout table maps them, written here rather than taken from ply.py. read_ply reads
out/train.ply back, so a test compares the two.

The failed runs each break one thing: a setting, with set_train or with a test config that leaves
out a key, SPLATPIPE_CAP_DEADLINE, a path of the Camera_Solve_Output, the files that
write_camera_solve_output writes, with write_model's records changed for the scene's own
refusals, or the GpuInfo that FakeGpu's probe gives. The test of a machine with no torch installs
no FakeGpu and replaces splatpipe.train.gpu_api.importlib with a stand-in, as
tests/test_train_gpu_api.py does, so the step's real load_gpu runs and no test imports the
GPU_Module. run_failed checks that each run failed with an Expected_Failure whose Readable_Error
names what its requirement says and holds no traceback (Req 11.7, 14.6). The order tests give a
run two faults, so the message shows which check comes first (A9). FakeGpu can also raise
GpuOutOfMemory or NonFiniteLoss from a GPU call, or give back every splat with a NaN opacity.
The CLI tests run splatpipe.train.__main__.main in this process, so it uses the stand-in too.

The depth-priors spec's tests of training with and without priors (its Req 18.5) run the step on
tests/camera_files.py's plane scene: four 16 x 12 images of the plane z = 3, which the step keeps
at 16 x 12, a mask on the first, and 30 points, all within the Splat_Budget. write_plane_priors
writes its Prior_Set with the Prior_Writer, splatpipe.train.priors' write_prior and write_index,
as the Depth_Step would from fits with a scale of 1 and an offset of 0. plane_prior gives each of
the first three images the plane's inverse depth and normal at a Prior_Size of 8 x 6, half the
Training_Size, with one stored pixel left with no prior, and the fourth image gets no Prior.
Writing the Priors here keeps these tests off the Depth_Step and its stand-in. The oracles are
written here: the Brightness_Gradient from A12's formula and the taps from Req 10.6's, in
Fractions. The broken Prior_Sets are that one with its depth.json cut short, and one written
for an earlier camera solve, in which the second image sat 0.01 further along x. The runs with
no depth.json, with no work/depth/ at all or with only Prior_Files in it, use the default scene,
so their reports hold exactly the splat-training spec's counts, timings and warnings.

conftest's no_cap_deadline removes SPLATPIPE_CAP_DEADLINE before every test, so only a test that
sets it with monkeypatch.setenv has a Cap_Deadline (Req 14.11).
"""

import math
import shutil
import tomllib
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from fractions import Fraction
from functools import partial
from pathlib import Path
from typing import Any, NoReturn

import cv2
import numpy as np
import pytest

from splatpipe import train
from splatpipe.config import Config, load_config
from splatpipe.jsonio import read_json
from splatpipe.train.__main__ import main as train_main
from splatpipe.train.gpu_api import GpuInfo, GpuOutOfMemory, NonFiniteLoss, non_finite_message
from splatpipe.train.ply import check_ply, read_ply
from splatpipe.train.priors import (Prior, PriorEntry, PriorIndex, prior_path, write_index,
                                    write_prior)
from splatpipe.train.scene import world_to_camera
from splatpipe.train.splats import SH_C0, Splats

from camera_files import (ARC_ANGLES, CAMERA_SOLVE, DEFAULT_CAMERA, DEFAULT_IMAGES,
                          DEFAULT_POINTS, IMAGES_FOLDER, PLANE_CAMERA, PLANE_IMAGES, PLANE_POINTS,
                          PLANE_Z, POINTS_BIN, ImageRecord, write_camera_solve_output,
                          write_model)

# The default Camera_Solve_Output: 3 images of 8 x 6 pixels, a mask on the first only, and 12
# points.
IMAGES, POINTS = len(DEFAULT_IMAGES), len(DEFAULT_POINTS)
WIDTH, HEIGHT = DEFAULT_CAMERA.width, DEFAULT_CAMERA.height
STEPS = 6  # the test config's steps for fast, the Job's Preset
A10G_BYTES = 24_146_608_128  # FakeGpu's GPU memory by default, a "24 GB" A10G (Req 2.8)
PEAK_BYTES = 1_234_567_890  # the peak GPU memory the report test gives FakeGpu

# The 50 Shared_Settings of [train] and its 3 Preset_Settings, the report's 53 settings
# (Req 9.2, 9.3, 10.6, and the depth-priors spec's Req 14.1 and 14.3).
SHARED_SETTINGS = ("long_edge", "mask_threshold", "seed", "sh_degree_interval",
                   "scene_extent_factor", "near_plane_factor", "start_opacity",
                   "start_scale_factor", "start_scale_neighbours", "neighbour_chunk_pairs",
                   "means_lr", "means_lr_final", "scales_lr", "quats_lr", "opacities_lr",
                   "sh_dc_lr", "sh_rest_lr", "adam_eps", "exposure_lr", "l1_weight",
                   "ssim_weight", "opacity_reg", "scale_reg", "erank_reg", "erank_epsilon",
                   "erank_start_fraction", "refine_start_step", "refine_every",
                   "refine_stop_step", "refine_stop_fraction", "clean_reserve_minutes",
                   "save_margin_minutes", "gpu_memory_slack", "plain_gradient",
                   "depth_gradient_falloff", "depth_weight_start", "depth_weight_end",
                   "depth_weight_floor", "depth_error_cap", "normal_weight", "normal_start_step",
                   "normal_every", "flatten_reg", "depth_normal_weight",
                   "normal_consistency_weight", "depth_normal_start_fraction",
                   "backproject_share", "backproject_max", "min_vcpus", "min_memory_gb")
PRESET_SETTINGS = ("steps", "max_splats", "min_gpu_memory_gb")
# The Train_Report's timings (Req 10.2).
TIMINGS = ("load", "train", "write_ply", "check_ply")

# The Exposure_Fix [A | b] that the exposure tests give every view. Every value is a sum of
# powers of 2 that float32 holds exactly, so the Mean_Exposure is exactly this.
EXPOSURE = np.array([[1.25, 0.0, 0.125, 0.0625],
                     [0.0, 0.75, 0.0, -0.03125],
                     [0.25, 0.0, 0.5, 0.125]])
# The most a colour worked out from a written f_dc may stray from Ā·c + b̄, from rounding f_dc to
# float32, which is under 1e-7 for these colours.
COLOUR_ERROR = 1e-6

# The three warnings, worded as the design gives them (Req 3.11, 7.8, 8.9, 10.5).
SUBSET_WARNING = ("The camera solve gave {points} points, more than this preset's splat budget "
                  "of {budget}, so training started from {budget} of them, chosen at random.")
STOPPED_EARLY_WARNING = ("Training stopped early, after {run} of {planned} iterations, to leave "
                         "time for the clean step before the job's time cap.")
DROPPED_WARNING = ("Splats left out of out/train.ply because they held a NaN or an infinity: "
                   "{count}.")


def camera_solve_job(new_job: Callable[..., str]) -> str:
    """Make a Job with new_job, write the default Camera_Solve_Output into its work/cameras/, and
    give back its Job_Id."""
    job_id = new_job()
    write_camera_solve_output(folder_of(job_id))
    return job_id


def run_done(job_id: str) -> dict[str, Any]:
    """Run the train step on the Job through splatpipe.train.run, check that it's done, and give
    back its report."""
    report = train.run(job_id)
    assert report["outcome"] == "done", report["error"]
    return report


def folder_of(job_id: str) -> Path:
    """The Job_Folder of `job_id` in the test config's Jobs_Folder."""
    return load_config().jobs().jobs_folder / job_id


def trained_splat(job_id: str) -> Path:
    """The Trained_Splat, out/train.ply, in the Job's folder."""
    return folder_of(job_id) / "out" / "train.ply"


def record_of(job_id: str) -> Any:
    """The Train_Record, work/train/train.json, read back."""
    return read_json(folder_of(job_id) / "work" / "train" / "train.json")


def entries(folder: Path) -> list[str]:
    """Every file and folder inside `folder`, by its path from `folder`, sorted."""
    return sorted(path.relative_to(folder).as_posix() for path in folder.rglob("*"))


def layout_fields(splats: Splats) -> dict[str, np.ndarray]:
    """The 59 fields of the Splat_Layout for `splats`, by name, as the training spec's Splat_Layout
    table maps them: x, y and z the means, f_dc_i sh0[:, i], f_rest_k sh_rest[:, k % 15, k // 15],
    opacity the logits, scale_i the log scales and rot_i the quaternions, w first."""
    fields = {axis: splats.means[:, i] for i, axis in enumerate("xyz")}
    fields |= {f"f_dc_{i}": splats.sh0[:, i] for i in range(3)}
    fields |= {f"f_rest_{k}": splats.sh_rest[:, k % 15, k // 15] for k in range(45)}
    fields["opacity"] = splats.opacity_logits
    fields |= {f"scale_{i}": splats.log_scales[:, i] for i in range(3)}
    fields |= {f"rot_{i}": splats.quats[:, i] for i in range(4)}
    return fields


def assert_same_fields(read: dict[str, np.ndarray], expected: dict[str, np.ndarray]) -> None:
    """Check that `read` has the fields of `expected`, and each field the same values."""
    assert sorted(read) == sorted(expected)
    for name, values in expected.items():
        assert np.array_equal(read[name], values), name


def shown_colours(sh0: np.ndarray) -> np.ndarray:
    """The colour each splat shows from its band-0 coefficients `sh0`, (N, 3), when it has no
    higher band: 0.5 + SH_C0 · sh0, in float64, before the clamp at 0."""
    return 0.5 + SH_C0 * np.asarray(sh0, dtype=np.float64)


def turned_about_y(angle: float, translation: tuple[float, float, float]) -> np.ndarray:
    """The (4, 4) world-to-camera matrix of a rotation of `angle` radians about y, then
    `translation`, the pose camera_files gives each default image on its arc."""
    cos, sin = math.cos(angle), math.sin(angle)
    matrix = np.array([[cos, 0.0, sin, 0.0],
                       [0.0, 1.0, 0.0, 0.0],
                       [-sin, 0.0, cos, 0.0],
                       [0.0, 0.0, 0.0, 1.0]])
    matrix[:3, 3] = translation
    return matrix


# A done run (Req 6.7, 8.2, 8.6, 8.8, 8.12, 10.7)

def test_a_done_run_leaves_a_checked_out_train_ply_of_the_start_splats_and_only_train_json_in_work(
        new_job: Callable[..., str], fake_gpu: Callable[..., Any]) -> None:
    job_id = camera_solve_job(new_job)
    gpu = fake_gpu()
    report = run_done(job_id)
    splat = trained_splat(job_id)
    # out/train.ply passes the PLY_Check with every Start_Splat (Req 8.6, 8.8). Every
    # Exposure_Fix is [I | 0], so the Mean_Exposure baked in leaves every value as the Start_Splat
    # holds it, and read_ply gives back exactly the Start_Splats (Req 6.7, 8.2, 8.12).
    assert check_ply(splat) == len(gpu.start) == POINTS
    assert_same_fields(read_ply(splat), layout_fields(gpu.start))
    # The move took train.ply out of work/train/, which holds only the Train_Record, with every
    # field. The scene's numbers are those of the scene the Trainer got, the 8 x 6 camera wasn't
    # scaled, and with no Cap_Deadline both times are null (Req 8.8, 10.7).
    assert entries(folder_of(job_id) / "work" / "train") == ["train.json"]
    scene = gpu.scene
    assert record_of(job_id) == {
        "near_plane": scene.near_plane,
        "median_scene_depth": scene.median_depth,
        "scene_extent": scene.extent,
        "means_lr_start": report["settings"]["means_lr"] * scene.extent,
        "width_ratio": 1.0,
        "height_ratio": 1.0,
        "mean_exposure": np.eye(3, 4).tolist(),
        "cap_deadline": None,
        "stop_time": None,
    }


# The Train_Report (Req 10.1, 10.2, 10.4, 10.6)

def test_the_report_holds_every_count_as_an_int_the_four_timings_and_the_53_settings(
        new_job: Callable[..., str], fake_gpu: Callable[..., Any]) -> None:
    job_id = camera_solve_job(new_job)
    fake_gpu(peak_bytes=PEAK_BYTES)
    report = run_done(job_id)
    # The Step_Runner's ordinary report, in out/reports/train.json (Req 10.6).
    assert read_json(folder_of(job_id) / "out" / "reports" / "train.json") == report
    assert (report["warnings"], report["error"]) == ([], None)
    # Every count of Req 10.1, each an int (Req 10.4). The 6 Iterations end at SH_Degree 0, with
    # an sh_degree_interval of 1000.
    assert report["counts"] == {
        "steps_planned": STEPS, "gpu_memory_bytes": A10G_BYTES, "images": IMAGES,
        "sparse_points": POINTS, "image_width": WIDTH, "image_height": HEIGHT,
        "masked_images": 1, "start_splats": POINTS, "steps_run": STEPS, "sh_degree": 0,
        "peak_gpu_memory_bytes": PEAK_BYTES, "splats": POINTS,
        "ply_bytes": trained_splat(job_id).stat().st_size}
    assert all(type(value) is int for value in report["counts"].values())
    # The four timings, each a number of seconds (Req 10.2).
    assert sorted(report["timings"]) == sorted(TIMINGS)
    assert all(type(seconds) is float and seconds >= 0
               for seconds in report["timings"].values())
    # The 50 Shared_Settings and the Preset's 3, as the test config has them (Req 10.6).
    assert sorted(report["settings"]) == sorted([*SHARED_SETTINGS, *PRESET_SETTINGS])
    assert len(report["settings"]) == 53
    assert report["settings"] == load_config().step_settings("train", "fast")


# The Cap_Deadline (Req 7.1, 7.5, 10.7)

def test_a_cap_deadline_an_hour_ahead_stops_nothing_and_the_record_holds_it_and_the_stop_time(
        new_job: Callable[..., str], fake_gpu: Callable[..., Any],
        monkeypatch: pytest.MonkeyPatch) -> None:
    job_id = camera_solve_job(new_job)
    gpu = fake_gpu()
    # An hour ahead, in the form the User_Data writes, to the second.
    deadline = (datetime.now(timezone.utc) + timedelta(hours=1)).replace(microsecond=0)
    monkeypatch.setenv("SPLATPIPE_CAP_DEADLINE", deadline.strftime("%Y-%m-%dT%H:%M:%SZ"))
    report = run_done(job_id)
    # The Stop_Time is 10 + 5 minutes before the Cap_Deadline, 45 minutes ahead, so the Trainer
    # got it and still ran every Iteration, with no warning (Req 7.5).
    stop = deadline - timedelta(minutes=15)
    assert [args[3] for name, args in gpu.calls if name == "train"] == [stop]
    assert (report["counts"]["steps_run"], report["warnings"]) == (STEPS, [])
    # Both times in UTC, 15 minutes apart (Req 10.7).
    record = record_of(job_id)
    assert (record["cap_deadline"], record["stop_time"]) == (deadline.isoformat(),
                                                             stop.isoformat())
    assert record["cap_deadline"].endswith("+00:00") and record["stop_time"].endswith("+00:00")


# The Exposure_Fix (Req 6.7)

def test_an_exposure_fix_the_test_sets_shows_in_the_written_colours_as_the_bake_gives_it(
        new_job: Callable[..., str], fake_gpu: Callable[..., Any]) -> None:
    job_id = camera_solve_job(new_job)
    gpu = fake_gpu(exposure=EXPOSURE)
    run_done(job_id)
    written = read_ply(trained_splat(job_id))
    # Every view has EXPOSURE, so the Mean_Exposure is EXPOSURE (Req 6.4, 10.7).
    assert record_of(job_id)["mean_exposure"] == EXPOSURE.tolist()
    # Each written splat shows Ā·c + b̄, where c is the colour its Start_Splat showed (Req 6.7).
    a, b = EXPOSURE[:, :3], EXPOSURE[:, 3]
    colours = shown_colours(np.column_stack([written[f"f_dc_{i}"] for i in range(3)]))
    assert np.allclose(colours, shown_colours(gpu.start.sh0) @ a.T + b,
                       rtol=0, atol=COLOUR_ERROR)
    # The Start_Splats have no higher band, and Ā times 0 is 0. Only the colours change.
    start = layout_fields(gpu.start)
    for name in start:
        if name.startswith("f_rest_"):
            assert np.array_equal(written[name], np.zeros(POINTS)), name
        elif not name.startswith("f_dc_"):
            assert np.array_equal(written[name], start[name]), name


# What the step writes (Req 8.11, 14.5)

def test_a_done_run_adds_only_train_json_out_train_ply_and_its_report_to_the_job_folder(
        new_job: Callable[..., str], fake_gpu: Callable[..., Any]) -> None:
    job_id = camera_solve_job(new_job)
    folder = folder_of(job_id)
    before = entries(folder)
    fake_gpu()
    run_done(job_id)
    # Nothing is gone, and the only new files are the three the step and the Step_Runner write.
    # work/train/ and out/reports/ are the folders that the Step_Runner makes for them.
    assert entries(folder) == sorted([*before, "work/train", "work/train/train.json",
                                      "out/train.ply", "out/reports", "out/reports/train.json"])


# What the Trainer gets (Req 1.19, 1.20)

def test_the_trainer_gets_every_image_in_file_order_with_its_pose_and_splats_at_the_points(
        new_job: Callable[..., str], fake_gpu: Callable[..., Any]) -> None:
    job_id = camera_solve_job(new_job)
    gpu = fake_gpu()
    run_done(job_id)
    # One view for each image of images.bin, in its order, none held out (Req 1.19), each with
    # its image's pose from the model as it is, a turn about y and then (0, 0, 1) (Req 1.20).
    views = gpu.scene.views
    assert [view.name for view in views] == [image.name for image in DEFAULT_IMAGES]
    for view, image, angle in zip(views, DEFAULT_IMAGES, ARC_ANGLES, strict=True):
        assert np.allclose(view.world_to_camera, turned_about_y(angle, image.translation),
                           rtol=0, atol=1e-12), view.name
    # The Start_Splats sit at the model's points, in points3D.bin's order (Req 1.20).
    points = np.array([point.position for point in DEFAULT_POINTS], dtype=np.float32)
    assert np.array_equal(gpu.start.means, points)


# More points than the Splat_Budget (Req 3.11)

def test_more_points_than_max_splats_warn_and_the_training_starts_from_max_splats_of_them(
        new_job: Callable[..., str], fake_gpu: Callable[..., Any],
        set_train: Callable[..., None]) -> None:
    job_id = camera_solve_job(new_job)
    set_train(max_splats={"fast": 5})
    gpu = fake_gpu()
    report = run_done(job_id)
    assert report["warnings"] == [SUBSET_WARNING.format(points=POINTS, budget=5)]
    assert report["counts"]["start_splats"] == len(gpu.start) == 5


# An early stop and a second run (Req 7.7, 7.8, 11.8)

def test_a_stop_after_3_iterations_still_writes_and_checks_the_splat_and_warns(
        new_job: Callable[..., str], fake_gpu: Callable[..., Any]) -> None:
    job_id = camera_solve_job(new_job)
    fake_gpu(stop_after=3)
    report = run_done(job_id)
    # The splat from the 3 Iterations run is written and checked, and in out/ (Req 7.7).
    assert check_ply(trained_splat(job_id)) == POINTS
    assert sorted(report["timings"]) == sorted(TIMINGS)
    assert (report["counts"]["steps_run"], report["counts"]["steps_planned"]) == (3, STEPS)
    # The warning gives the Iterations run and the Planned_Iterations (Req 7.8).
    assert report["warnings"] == [STOPPED_EARLY_WARNING.format(run=3, planned=STEPS)]


def test_a_second_run_on_the_same_job_leaves_only_its_own_files(
        new_job: Callable[..., str], fake_gpu: Callable[..., Any]) -> None:
    job_id = camera_solve_job(new_job)
    # The first run stops after 3 Iterations, with EXPOSURE for every view.
    fake_gpu(stop_after=3, exposure=EXPOSURE)
    run_done(job_id)
    assert record_of(job_id)["mean_exposure"] == EXPOSURE.tolist()
    # The second runs every Iteration, with [I | 0] for every view.
    gpu = fake_gpu()
    report = run_done(job_id)
    folder = folder_of(job_id)
    assert entries(folder / "work" / "train") == ["train.json"]
    assert entries(folder / "out") == ["reports", "reports/train.json", "train.ply"]
    # Each file is the second run's: its report, with no warning, its Train_Record, with the
    # Mean_Exposure [I | 0], and its splat, the Start_Splats as they are (Req 11.8).
    assert read_json(folder / "out" / "reports" / "train.json") == report
    assert (report["counts"]["steps_run"], report["warnings"]) == (STEPS, [])
    assert record_of(job_id)["mean_exposure"] == np.eye(3, 4).tolist()
    assert_same_fields(read_ply(trained_splat(job_id)), layout_fields(gpu.start))


# A splat with a NaN (Req 8.9, 10.5, D6)

def test_a_nan_splat_from_the_trainer_is_dropped_with_the_warning_and_the_run_is_done(
        new_job: Callable[..., str], fake_gpu: Callable[..., Any]) -> None:
    job_id = camera_solve_job(new_job)
    gpu = fake_gpu(nan_opacity="one")
    report = run_done(job_id)
    # The warning gives the count dropped, 1 (Req 8.9, 10.5).
    assert report["warnings"] == [DROPPED_WARNING.format(count=1)]
    # out/train.ply passes the PLY_Check one splat shorter than the Start_Splats, and counts.splats
    # gives the splats written (Req 8.9).
    splat = trained_splat(job_id)
    assert check_ply(splat) == report["counts"]["splats"] == POINTS - 1
    assert report["counts"]["start_splats"] == POINTS
    # The first splat, whose opacity the Trainer gave as NaN, is the one left out, and the others
    # keep their order.
    written = read_ply(splat)
    assert np.array_equal(np.column_stack([written[axis] for axis in "xyz"]),
                          gpu.start.means[1:])


# Training with and without depth priors (depth-priors spec, Req 10.1–10.4, 10.8, 14.3, 14.5)

# The plane scene's Prior_Size here: half its 16 x 12 Training_Size on each side, so a training
# pixel's bilinear sample draws on two stored pixels across and two down, and the scene's taps
# show which Prior_Size they were worked out for.
PRIOR_WIDTH, PRIOR_HEIGHT = PLANE_CAMERA.width // 2, PLANE_CAMERA.height // 2
# The stored pixel, (x, y), where MoGe found no geometry in every Prior, so it has no prior.
NO_GEOMETRY = (0, 0)
# The plane's images that get a Prior: the first three in images.bin. The fourth, 000003.jpg,
# gets none, so its View_Record holds an empty prior.
PRIOR_RECORDS = PLANE_IMAGES[:3]
# The Prior_View whose Frame the Prior_Set test replaces with a ramp, so its Brightness_Gradient
# isn't 0. It has no Mask_File.
RAMP_IMAGE = PLANE_IMAGES[1].name
# A Prior_File's arrays (Req 7.1).
PRIOR_ARRAYS = ("inverse_depth", "normal", "mask")
# The plane's images as an earlier camera solve gave them: the same, but with the second image
# 0.01 further along x.
EARLIER_SOLVE = (PLANE_IMAGES[0],
                 replace(PLANE_IMAGES[1], translation=(PLANE_IMAGES[1].translation[0] + 0.01,
                                                       *PLANE_IMAGES[1].translation[1:])),
                 PLANE_IMAGES[2])
# The Req 10.3 warning, as the design words it, before and after the file and what failed.
BROKEN_PRIORS_HEAD = "The depth priors in work/depth/ weren't used, because "
BROKEN_PRIORS_TAIL = ". Training ran without depth priors."
# The counts of the splat-training spec's Req 10.1, every one that a done run adds.
SPLAT_TRAINING_COUNTS = ("steps_planned", "gpu_memory_bytes", "images", "sparse_points",
                         "image_width", "image_height", "masked_images", "start_splats",
                         "steps_run", "sh_degree", "peak_gpu_memory_bytes", "splats",
                         "ply_bytes")


def plane_job(new_job: Callable[..., str]) -> str:
    """Make a Job with new_job, write the plane scene's Camera_Solve_Output into its
    work/cameras/, with images.bin in PLANE_IMAGES' order and a Mask_File for 000000.jpg only,
    and give back its Job_Id."""
    job_id = new_job()
    write_camera_solve_output(folder_of(job_id), PLANE_CAMERA, PLANE_IMAGES, PLANE_POINTS)
    return job_id


def priors_folder(job_id: str) -> Path:
    """The Job's work/depth/, where the Depth_Step writes the Prior_Set."""
    return folder_of(job_id) / "work" / "depth"


def plane_prior(record: ImageRecord) -> Prior:
    """The plane's Prior for the image `record` at the Prior_Size, as the Depth_Step writes it
    from a fit with a scale of 1 and an offset of 0: at each stored pixel's centre, 1 / the
    camera-space z at which its ray meets the plane z = PLANE_Z, and the plane's unit normal in
    the camera frame, facing the camera. NO_GEOMETRY has no prior: its mask is false and its
    values are 0.

    The ray through (u, v) with a camera-space z of 1 is d = ((u − cx) / fx, (v − cy) / fy, 1).
    Its world z rises by R[:, 2] · d for each unit of z, from the Camera_Centre −Rᵀt, so it meets
    the plane at z = (PLANE_Z − the centre's z) / that rise. Every camera sits below the plane
    and looks up at it, so the rise is above 0 and the normal facing the camera is −R[:, 2].
    """
    camera = PLANE_CAMERA
    matrix = world_to_camera(np.array(record.quaternion), np.array(record.translation))
    rotation = matrix[:3, :3]
    centre = -rotation.T @ matrix[:3, 3]
    across = ((np.arange(PRIOR_WIDTH) + 0.5) * camera.width / PRIOR_WIDTH - camera.cx) / camera.fx
    down = ((np.arange(PRIOR_HEIGHT) + 0.5) * camera.height / PRIOR_HEIGHT - camera.cy) / camera.fy
    rise = (across[np.newaxis] * rotation[0, 2] + down[:, np.newaxis] * rotation[1, 2]
            + rotation[2, 2])
    mask = np.ones((PRIOR_HEIGHT, PRIOR_WIDTH), dtype=bool)
    mask[NO_GEOMETRY[1], NO_GEOMETRY[0]] = False
    inverse_depth = np.where(mask, rise / (PLANE_Z - centre[2]), 0.0).astype(np.float32)
    normal = np.where(mask[..., np.newaxis], -rotation[:, 2], 0.0).astype(np.float32)
    return Prior(inverse_depth=inverse_depth, normal=normal, mask=mask)


def write_plane_priors(job_id: str,
                       records: Sequence[ImageRecord] = PRIOR_RECORDS) -> dict[str, Prior]:
    """Write a Prior_Set for the plane scene into the Job's work/depth/ with the Prior_Writer,
    write_prior and write_index, as the Depth_Step would from fits with a scale of 1 and an offset
    of 0: plane_prior of each of `records`, then the Prior_Index, last, with the plane's camera,
    the Prior_Size and each record's pose. Gives the Priors written, by image name."""
    folder = priors_folder(job_id)
    folder.mkdir(parents=True)
    priors = {str(record.name): plane_prior(record) for record in records}
    for name, prior in priors.items():
        write_prior(prior_path(folder, name), prior)
    listed = tuple(PriorEntry(name=str(record.name), quaternion=record.quaternion,
                              translation=record.translation, scale=1.0, offset=0.0,
                              fit_points=len(PLANE_POINTS), kept_points=len(PLANE_POINTS),
                              median_error=0.0, metres_per_unit=1.0)
                   for record in records)
    camera = (PLANE_CAMERA.width, PLANE_CAMERA.height, PLANE_CAMERA.fx, PLANE_CAMERA.fy,
              PLANE_CAMERA.cx, PLANE_CAMERA.cy)
    write_index(folder, PriorIndex(camera=camera, prior_width=PRIOR_WIDTH,
                                   prior_height=PRIOR_HEIGHT, metres_per_unit=1.0,
                                   priors=listed))
    return priors


def write_ramp(job_id: str, name: str) -> None:
    """Replace the Frame `name` in the Job's work/cameras/images/ with a JPEG of a grey ramp at
    the plane camera's size, 10 levels brighter for each column and 6 for each row, so its
    Brightness_Gradient isn't 0."""
    grey = (10 * np.arange(PLANE_CAMERA.width)[np.newaxis]
            + 6 * np.arange(PLANE_CAMERA.height)[:, np.newaxis]).astype(np.uint8)
    ok, data = cv2.imencode(".jpg", np.repeat(grey[..., np.newaxis], 3, axis=2))
    assert ok
    (folder_of(job_id) / CAMERA_SOLVE / IMAGES_FOLDER / name).write_bytes(data.tobytes())


def gradient_of(pixels: np.ndarray) -> np.ndarray:
    """The Brightness_Gradient of the (H, W, 3) uint8 RGB `pixels`, as A12 defines it: OpenCV's
    grey value / 255, the central differences (I(x + 1) − I(x − 1)) / 2 across and the same down
    with the border pixels repeated outward, and the length of the two, as float32."""
    grey = cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY) / 255
    padded = np.pad(grey, 1, mode="edge")
    across = (padded[1:-1, 2:] - padded[1:-1, :-2]) / 2
    down = (padded[2:, 1:-1] - padded[:-2, 1:-1]) / 2
    return np.hypot(across, down).astype(np.float32)


def taps_oracle(stored: int, target: int) -> tuple[list[int], list[int], list[float], list[int]]:
    """The taps of one axis from `stored` pixels up to `target`, with pixel centres aligned
    (Req 10.6), in Fractions: training pixel X's centre sits at c = (X + 1/2) · S / T stored
    pixels, so its sample is at c − 1/2, clamped at 0. i0 is that place's floor, i1 the next
    pixel but no further than the last, and w1 the place's fraction, or 0 where i1 is i0. The
    nearest stored pixel is the one under c. Gives i0, i1, w1 and the nearest, one per X."""
    first, second, weight, nearest = [], [], [], []
    for x in range(target):
        centre = Fraction(2 * x + 1, 2 * target) * stored
        place = max(centre - Fraction(1, 2), Fraction(0))
        low = math.floor(place)
        high = min(low + 1, stored - 1)
        first.append(low)
        second.append(high)
        weight.append(float(place - low) if high > low else 0.0)
        nearest.append(math.floor(centre))
    return first, second, weight, nearest


def assert_taps(axis: tuple[np.ndarray, np.ndarray, np.ndarray], nearest: np.ndarray,
                stored: int, target: int) -> None:
    """Check one axis of a Scene's PriorTaps, (i0, i1, w1) and the nearest stored pixels, against
    taps_oracle(stored, target)."""
    first, second, weight, closest = taps_oracle(stored, target)
    assert np.array_equal(axis[0], first)
    assert np.array_equal(axis[1], second)
    assert np.array_equal(axis[2], np.array(weight, dtype=np.float32))
    assert np.array_equal(nearest, closest)


def test_a_prior_set_gives_the_trainer_each_prior_views_prior_and_gradient_and_the_taps(
        new_job: Callable[..., str], fake_gpu: Callable[..., Any]) -> None:
    job_id = plane_job(new_job)
    write_ramp(job_id, RAMP_IMAGE)
    written = write_plane_priors(job_id)
    gpu = fake_gpu()
    report = run_done(job_id)
    # The report counts the Prior_Views, the three images with a Prior, as an int, and the
    # Prior_Set adds no warning (Req 10.1, 14.4).
    assert report["counts"]["prior_images"] == len(PRIOR_RECORDS)
    assert type(report["counts"]["prior_images"]) is int
    assert report["warnings"] == []
    # Each Prior_View's View_Record holds its Prior as the Prior_Writer wrote it, at the
    # Prior_Size, and the Brightness_Gradient of its own pixels at the Training_Size. The image
    # with no Prior holds an empty prior, None (Req 10.8).
    scene = gpu.scene
    assert (scene.width, scene.height) == (PLANE_CAMERA.width, PLANE_CAMERA.height)
    assert [view.name for view in scene.views if view.prior is None] == [PLANE_IMAGES[3].name]
    for view in scene.views:
        if view.prior is None:
            continue
        for array in PRIOR_ARRAYS:
            held, prior = getattr(view.prior.prior, array), getattr(written[view.name], array)
            assert held.dtype == prior.dtype and np.array_equal(held, prior), (view.name, array)
        gradient = view.prior.gradient
        assert gradient.dtype == np.float32, view.name
        assert np.allclose(gradient, gradient_of(view.pixels), rtol=1e-6, atol=0), view.name
    # The ramp's gradient isn't 0, so the check above tells one view's pixels from another's.
    ramp = next(view for view in scene.views if view.name == RAMP_IMAGE)
    assert ramp.prior.gradient.max() > 0
    # The taps scale the index's 8 x 6 Prior_Size up to the 16 x 12 Training_Size (Req 10.6).
    assert_taps(scene.taps.rows, scene.taps.nearest_rows, PRIOR_HEIGHT, scene.height)
    assert_taps(scene.taps.columns, scene.taps.nearest_columns, PRIOR_WIDTH, scene.width)


def cut_index_short(job_id: str) -> None:
    """Write the plane's Prior_Set, then cut its Prior_Index to half its bytes."""
    write_plane_priors(job_id)
    index = priors_folder(job_id) / "depth.json"
    data = index.read_bytes()
    index.write_bytes(data[:len(data) // 2])


# Each writes a Prior_Set for the plane scene that fails a check of Req 10.4, with what the
# warning says failed: a Prior_Index cut short, whose reason comes from Python's json, and Priors
# made for an earlier camera solve, whose second image has another pose in images.bin now (A9).
BROKEN_PRIORS = [
    pytest.param(cut_index_short, "can't be read as JSON (", id="a depth.json cut short"),
    pytest.param(partial(write_plane_priors, records=EARLIER_SOLVE),
                 f"has the pose of an earlier camera solve for '{PLANE_IMAGES[1].name}': its "
                 "translation isn't the one images.bin gives",
                 id="priors for an earlier camera solve"),
]


@pytest.mark.parametrize(("write", "what"), BROKEN_PRIORS)
def test_broken_priors_warn_naming_the_file_and_what_failed_and_training_has_no_priors(
        new_job: Callable[..., str], fake_gpu: Callable[..., Any],
        write: Callable[[str], object], what: str) -> None:
    job_id = plane_job(new_job)
    write(job_id)
    gpu = fake_gpu()
    report = run_done(job_id)
    # One warning, naming depth.json by its whole path and what failed, and saying training ran
    # without depth priors (Req 10.3, 14.5).
    index = folder_of(job_id).resolve() / "work" / "depth" / "depth.json"
    [warning] = report["warnings"]
    assert warning.startswith(f"{BROKEN_PRIORS_HEAD}{index} {what}"), warning
    assert warning.endswith(BROKEN_PRIORS_TAIL), warning
    # One bad file drops the whole Prior_Set: no count of it, every view an empty prior and no
    # taps (A9).
    assert "prior_images" not in report["counts"]
    assert [view.prior for view in gpu.scene.views] == [None] * len(PLANE_IMAGES)
    assert gpu.scene.taps is None


def write_prior_files_only(job_id: str) -> None:
    """Write a Prior_File for each image of the default scene into the Job's work/depth/, but no
    Prior_Index, as a Depth_Step stopped before its last write leaves them."""
    folder = priors_folder(job_id)
    folder.mkdir(parents=True)
    prior = Prior(inverse_depth=np.ones((HEIGHT, WIDTH), dtype=np.float32),
                  normal=np.zeros((HEIGHT, WIDTH, 3), dtype=np.float32),
                  mask=np.ones((HEIGHT, WIDTH), dtype=bool))
    for image in DEFAULT_IMAGES:
        write_prior(prior_path(folder, str(image.name)), prior)


@pytest.mark.parametrize("write", [
    pytest.param(None, id="no work/depth/"),
    pytest.param(write_prior_files_only, id="prior files with no depth.json"),
])
def test_with_no_depth_json_the_report_and_the_scene_are_the_splat_training_specs(
        new_job: Callable[..., str], fake_gpu: Callable[..., Any],
        write: Callable[[str], None] | None) -> None:
    job_id = camera_solve_job(new_job)
    if write is None:
        assert not priors_folder(job_id).exists()
    else:
        write(job_id)
    gpu = fake_gpu()
    report = run_done(job_id)
    # Exactly the splat-training spec's counts, timings and warnings, with no new count and no
    # warning about the missing priors, and the 53 settings (Req 10.2, 14.3, A14).
    assert sorted(report["counts"]) == sorted(SPLAT_TRAINING_COUNTS)
    assert sorted(report["timings"]) == sorted(TIMINGS)
    assert report["warnings"] == []
    assert sorted(report["settings"]) == sorted([*SHARED_SETTINGS, *PRESET_SETTINGS])
    assert len(report["settings"]) == 53
    # The Trainer gets a Scene with no Prior_Set: every view an empty prior and no taps (Req 10.8).
    assert [view.prior for view in gpu.scene.views] == [None] * IMAGES
    assert gpu.scene.taps is None


# Failed runs

def run_failed(job_id: str, *phrases: str, preset: str | None = None,
               config: Config | None = None) -> dict[str, Any]:
    """Run the train step on the Job through splatpipe.train.run, with `preset` and `config` when
    they're given, check that it failed with an Expected_Failure whose Readable_Error holds each of
    `phrases` and no traceback, and give back its report (Req 11.7, 14.6)."""
    report = train.run(job_id, preset, config=config)
    assert report["outcome"] == "failed"
    error = report["error"]
    assert error["expected"] is True, error
    readable = error["readable"]
    for phrase in phrases:
        assert phrase in readable, readable
    assert "Traceback" not in readable
    return report


def cameras_tail(*parts: str) -> str:
    """work/cameras/<parts> as a message ends the path on this system, like work\\cameras\\images
    on Windows. Each message gives the whole path, which starts with the Job_Folder."""
    return str(Path("work", "cameras", *parts))


def missing_input_message(path: Path) -> str:
    """The Req 1.2 Readable_Error for `path`, a path of the Camera_Solve_Output that isn't there."""
    return (f"The train step needs {path}, which the camera solve step writes, but it isn't "
            "there. Run the camera solve step on this job first.")


def with_corrupt_points(job_folder: Path) -> None:
    """Write the default Camera_Solve_Output into `job_folder`, then write its model again with
    points3D.bin cut one byte short, so the file ends inside its last point's track."""
    cameras = write_camera_solve_output(job_folder)
    write_model(cameras / "sparse", cut={POINTS_BIN: 1})


# A setting and a Cap_Deadline that the step refuses, and the facts of a machine whose torch sees
# no CUDA GPU, with the Readable_Error of each, as the design's Error Handling table gives them.
BAD_SETTING = {"mask_threshold": 0}
BAD_SETTING_MESSAGE = ("The setting mask_threshold in [train] is 0, but it must be a whole number "
                       "from 1 to 255. Change it to a whole number from 1 to 255.")
BAD_DEADLINE = "soon"
BAD_DEADLINE_MESSAGE = ("SPLATPIPE_CAP_DEADLINE is 'soon', but it must be a date and time with a "
                        "UTC offset, like 2026-11-21T14:25:30Z. The cloud runner sets it, so on a "
                        "laptop unset it.")
NO_CUDA = GpuInfo(cuda=False, gsplat_version="1.5.3", total_memory_bytes=0)
NO_CUDA_MESSAGE = ("The train step needs an NVIDIA GPU, but torch can't see one on this machine. "
                   "Check the driver with nvidia-smi, or run the job with "
                   "`uv run python -m cloud.start`.")


# A9's order (Req 1.2, 2.1, 7.1, 7.3, 9.7, 11.3): each run has two faults, and the one the step
# checks first gives the message.

def test_a_bad_setting_with_a_bad_cap_deadline_gives_the_settings_message(
        new_job: Callable[..., str], fake_gpu: Callable[..., Any],
        set_train: Callable[..., None], monkeypatch: pytest.MonkeyPatch) -> None:
    job_id = camera_solve_job(new_job)
    fake_gpu()
    set_train(**BAD_SETTING)
    monkeypatch.setenv("SPLATPIPE_CAP_DEADLINE", BAD_DEADLINE)
    # The settings are checked before the Cap_Deadline is read (Req 7.1, 9.7).
    assert run_failed(job_id)["error"]["readable"] == BAD_SETTING_MESSAGE


def test_a_bad_cap_deadline_with_no_work_cameras_gives_the_deadlines_message(
        new_job: Callable[..., str], fake_gpu: Callable[..., Any],
        monkeypatch: pytest.MonkeyPatch) -> None:
    job_id = new_job()
    fake_gpu()
    monkeypatch.setenv("SPLATPIPE_CAP_DEADLINE", BAD_DEADLINE)
    # The Cap_Deadline is read before the Camera_Solve_Output is checked (Req 7.3).
    assert run_failed(job_id)["error"]["readable"] == BAD_DEADLINE_MESSAGE


def test_a_missing_work_cameras_gives_the_camera_solve_message_with_no_probe_call(
        new_job: Callable[..., str], fake_gpu: Callable[..., Any]) -> None:
    job_id = new_job()
    gpu = fake_gpu(NO_CUDA)
    # images/ is the first path the step checks, and the GPU isn't even probed (Req 1.2, D5).
    images = folder_of(job_id).resolve() / "work" / "cameras" / "images"
    assert run_failed(job_id)["error"]["readable"] == missing_input_message(images)
    assert gpu.calls == []


def test_a_refusing_probe_with_a_corrupt_points3d_bin_gives_the_gpu_message(
        new_job: Callable[..., str], fake_gpu: Callable[..., Any]) -> None:
    job_id = new_job()
    with_corrupt_points(folder_of(job_id))
    gpu = fake_gpu(NO_CUDA)
    # The GPU is checked before the Sparse_Model is read, and nothing else reaches the GPU
    # (Req 2.1).
    assert run_failed(job_id)["error"]["readable"] == NO_CUDA_MESSAGE
    assert gpu.calls == [("probe", ())]


# The Expected_Failures that need no GPU (Req 11.7, 14.4, 14.6)

# A bad setting names the setting, its table and what it must be (Req 9.7), and a max_splats below
# start_scale_neighbours + 1 names both settings, their tables and their values (Req 9.8).
SETTING_FAILURES = [
    pytest.param(BAD_SETTING, ("mask_threshold in [train] is 0",
                               "it must be a whole number from 1 to 255"), id="a bad setting"),
    pytest.param({"max_splats": {"fast": 3}},
                 ("max_splats in [train.fast] is 3",
                  "start_scale_neighbours in [train] + 1, which is 4"),
                 id="max_splats below start_scale_neighbours + 1"),
]


@pytest.mark.parametrize(("values", "phrases"), SETTING_FAILURES)
def test_a_setting_the_step_refuses_fails_it_naming_the_setting_and_its_table(
        new_job: Callable[..., str], fake_gpu: Callable[..., Any],
        set_train: Callable[..., None], values: dict[str, Any],
        phrases: tuple[str, ...]) -> None:
    job_id = camera_solve_job(new_job)
    fake_gpu()
    set_train(**values)
    run_failed(job_id, *phrases)


def test_a_missing_train_key_fails_the_step_with_the_step_runners_message(
        new_job: Callable[..., str], fake_gpu: Callable[..., Any], temp_config: Path) -> None:
    job_id = camera_solve_job(new_job)
    fake_gpu()
    data = tomllib.loads(temp_config.read_text(encoding="utf-8"))
    del data["train"]["seed"]
    # The Step_Runner's message names the section, the Preset's table and the key (Req 9.6).
    run_failed(job_id, "needs the setting seed", "neither [train] nor [train.fast]",
               config=Config(data))


def test_a_bad_cap_deadline_fails_the_step_naming_the_variable_its_value_and_the_form(
        new_job: Callable[..., str], fake_gpu: Callable[..., Any],
        monkeypatch: pytest.MonkeyPatch) -> None:
    job_id = camera_solve_job(new_job)
    fake_gpu()
    monkeypatch.setenv("SPLATPIPE_CAP_DEADLINE", BAD_DEADLINE)
    run_failed(job_id, "SPLATPIPE_CAP_DEADLINE is 'soon'", "like 2026-11-21T14:25:30Z")  # Req 7.3


@pytest.mark.parametrize("part", ["images", "sparse/cameras.bin", "sparse/images.bin",
                                  "sparse/points3D.bin"])
def test_each_missing_input_path_fails_the_step_naming_it_and_the_camera_solve_step(
        new_job: Callable[..., str], fake_gpu: Callable[..., Any], part: str) -> None:
    job_id = camera_solve_job(new_job)
    path = folder_of(job_id) / "work" / "cameras" / part
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()
    fake_gpu()
    # The message names the path, and that the camera solve step writes it and has to run first
    # (Req 1.2).
    run_failed(job_id, f"{cameras_tail(*Path(part).parts)}, which the camera solve step writes",
               "Run the camera solve step on this job first")


# What the probe gives, with the Preset to run, for a GPU the step can't train on: no CUDA
# (Req 2.3), gsplat 1.5.2 (Req 2.5), and a "24 GB" A10G for the quality preset's 48 GB (Req 2.10).
GPU_FAILURES = [
    pytest.param(NO_CUDA, None, ("needs an NVIDIA GPU", "torch can't see one on this machine"),
                 id="no CUDA"),
    pytest.param(GpuInfo(cuda=True, gsplat_version="1.5.2", total_memory_bytes=A10G_BYTES), None,
                 ("This machine has gsplat 1.5.2", "Rebuild the machine image"),
                 id="gsplat 1.5.2"),
    pytest.param(GpuInfo(cuda=True, gsplat_version="1.5.3", total_memory_bytes=A10G_BYTES),
                 "quality",
                 ("This GPU has 24.1 GB of memory", "the quality preset needs a 48 GB GPU",
                  "accepts 0.9 of that", "a preset that needs less GPU memory",
                  "a machine with a larger GPU"),
                 id="a small GPU"),
]


@pytest.mark.parametrize(("info", "preset", "phrases"), GPU_FAILURES)
def test_a_gpu_the_step_cant_train_on_fails_it_after_the_probe_alone(
        new_job: Callable[..., str], fake_gpu: Callable[..., Any], info: GpuInfo,
        preset: str | None, phrases: tuple[str, ...]) -> None:
    job_id = camera_solve_job(new_job)
    gpu = fake_gpu(info)
    run_failed(job_id, *phrases, preset=preset)
    assert gpu.calls == [("probe", ())]


@dataclass
class NoTorch:
    """A stand-in for splatpipe.train.gpu_api's importlib on a machine with no torch, as
    tests/test_train_gpu_api.py replaces it. Its import_module records each name it's asked to
    import in `names`, then raises the ModuleNotFoundError that a missing torch gives."""

    names: list[str] = field(default_factory=list)

    def import_module(self, name: str) -> NoReturn:
        """Record `name`, then raise as Python does when torch isn't installed."""
        self.names.append(name)
        raise ModuleNotFoundError("No module named 'torch'", name="torch")


def test_a_machine_with_no_torch_fails_the_step_saying_it_runs_only_on_the_gpu_image(
        new_job: Callable[..., str], monkeypatch: pytest.MonkeyPatch) -> None:
    job_id = camera_solve_job(new_job)
    # No FakeGpu, so the step's GPU hook is the real load_gpu, which imports through the stand-in.
    stand_in = NoTorch()
    monkeypatch.setattr("splatpipe.train.gpu_api.importlib", stand_in)
    run_failed(job_id, "The train step runs only on the GPU machine image",
               "this machine has no torch", "`uv run python -m cloud.start`")  # Req 2.2
    assert stand_in.names == ["splatpipe.train.gpu"]


# The default scene with the camera solve's model changed for the scene's own refusals: every
# image at the middle image's pose, so every Camera_Centre is the same; every point 3 back along
# z, where every camera's z for it, 1 − x sin a + (z − 3) cos a, is below −1.7; and every point at
# the origin, 1 in front of every camera.
STILL_IMAGES = tuple(replace(image, quaternion=DEFAULT_IMAGES[1].quaternion,
                             translation=DEFAULT_IMAGES[1].translation)
                     for image in DEFAULT_IMAGES)
POINTS_BEHIND = tuple(replace(point, position=(point.position[0], point.position[1],
                                               point.position[2] - 3.0))
                      for point in DEFAULT_POINTS)
POINTS_ON_ONE_SPOT = tuple(replace(point, position=(0.0, 0.0, 0.0)) for point in DEFAULT_POINTS)

# Each writes a Camera_Solve_Output into a Job_Folder with one fault, with what the step's message
# names: a corrupt model file (Req 1.8), a missing image (Req 1.12), a mask that masks every pixel
# (Req 1.16), 3 points with 3 neighbours (Req 1.18), cameras that didn't move (Req 3.7), no point
# in front of any camera (Req 3.9), and points that all sit on one spot (Req 3.15).
OUTPUT_FAILURES = [
    pytest.param(with_corrupt_points, (cameras_tail("sparse", "points3D.bin"),
                                       "is cut short or isn't a COLMAP binary model"),
                 id="a corrupt model file"),
    pytest.param(partial(write_camera_solve_output, left_out={"000001.jpg"}),
                 ("the image '000001.jpg'", f"{cameras_tail('images')} has no file"),
                 id="a missing image"),
    pytest.param(partial(write_camera_solve_output, fully_masked={"000000.jpg"}),
                 ("The mask '000000.png' in ", f"{cameras_tail('masks')} masks every pixel"),
                 id="a bad mask"),
    pytest.param(partial(write_camera_solve_output, points=DEFAULT_POINTS[:3]),
                 ("too few points to start the splats from: 3", "needs at least 4"),
                 id="too few points"),
    pytest.param(partial(write_camera_solve_output, images=STILL_IMAGES),
                 ("The camera didn't move", "no splat can be trained",
                  "Film while you move through the place"),
                 id="cameras that didn't move"),
    pytest.param(partial(write_camera_solve_output, points=POINTS_BEHIND),
                 ("The camera solve put no point in front of any camera",),
                 id="no point in front of any camera"),
    pytest.param(partial(write_camera_solve_output, points=POINTS_ON_ONE_SPOT),
                 ("The camera solve's points all sit on top of each other",),
                 id="points on one spot"),
]


@pytest.mark.parametrize(("write", "phrases"), OUTPUT_FAILURES)
def test_a_camera_solve_output_the_step_refuses_fails_it_naming_what_is_wrong(
        new_job: Callable[..., str], fake_gpu: Callable[..., Any],
        write: Callable[[Path], object], phrases: tuple[str, ...]) -> None:
    job_id = new_job()
    write(folder_of(job_id))
    fake_gpu()
    run_failed(job_id, *phrases)


# No time left to train (Req 7.9)

def test_a_cap_deadline_already_past_fails_the_step_saying_no_time_was_left_to_train(
        new_job: Callable[..., str], fake_gpu: Callable[..., Any],
        monkeypatch: pytest.MonkeyPatch) -> None:
    job_id = camera_solve_job(new_job)
    fake_gpu()
    past = datetime.now(timezone.utc) - timedelta(minutes=1)
    monkeypatch.setenv("SPLATPIPE_CAP_DEADLINE", past.strftime("%Y-%m-%dT%H:%M:%SZ"))
    report = run_failed(job_id, "No time was left to train before the job's time cap",
                        "clean_reserve_minutes plus save_margin_minutes in [train]")
    # The Stop_Time had passed before the first Iteration, so the Trainer ran none.
    assert report["counts"]["steps_run"] == 0


# Out of GPU memory (Req 2.12, 10.3)

# The counts the step has worked out when it starts the 3-NN search, every count of the load
# before start_splats.
SET_UP_COUNTS = {"steps_planned": STEPS, "gpu_memory_bytes": A10G_BYTES, "images": IMAGES,
                 "sparse_points": POINTS, "image_width": WIDTH, "image_height": HEIGHT,
                 "masked_images": 1}

# The GPU call that runs out of memory, the Iteration FakeGpu's GpuOutOfMemory gives, where the
# message says it ran out, and the counts and the timings worked out before it.
OUT_OF_MEMORY = [
    pytest.param("nearest_distances_error", None, "while the step set up the scene",
                 SET_UP_COUNTS, [], id="in the 3-NN search"),
    pytest.param("train_error", 4, "at iteration 4 of training",
                 {**SET_UP_COUNTS, "start_splats": POINTS}, ["load"], id="in training"),
]


@pytest.mark.parametrize(("call", "iteration", "where", "counts", "timings"), OUT_OF_MEMORY)
def test_running_out_of_gpu_memory_fails_the_step_and_the_report_holds_the_counts_so_far(
        new_job: Callable[..., str], fake_gpu: Callable[..., Any], call: str,
        iteration: int | None, where: str, counts: dict[str, int], timings: list[str]) -> None:
    job_id = camera_solve_job(new_job)
    fake_gpu(**{call: GpuOutOfMemory(iteration)})
    # The message gives where it ran out, the Preset and its max_splats, and what to do.
    report = run_failed(job_id, f"The GPU ran out of memory {where}",
                        "The fast preset trains up to 50 splats",
                        "a preset that has a smaller splat budget", "a GPU that has more memory")
    assert report["counts"] == counts
    assert sorted(report["timings"]) == timings


# A loss that stops being finite (Req 4.16, A7)

# Each Loss_Term's value and the total that the Trainer's NonFiniteLoss gives.
LOSS_TERMS = {"l1": 0.0731, "ssim": 0.412, "opacity_reg": 0.31, "scale_reg": 0.0204,
              "erank": math.nan, "total": math.nan}
# The Step_Runner's general Readable_Error for a bug in the train step.
BUG_MESSAGE = ("The train step stopped because of an error in its code. This is a bug, not a "
               "problem with your video. Send out/reports/train.json to the team.")


def test_a_loss_that_stops_being_finite_is_a_bug_with_the_iteration_and_terms_in_technical(
        new_job: Callable[..., str], fake_gpu: Callable[..., Any]) -> None:
    job_id = camera_solve_job(new_job)
    message = non_finite_message(4, LOSS_TERMS)
    fake_gpu(train_error=NonFiniteLoss(message))
    report = train.run(job_id)
    assert report["outcome"] == "failed"
    error = report["error"]
    assert (error["expected"], error["readable"]) == (False, BUG_MESSAGE)
    technical = error["technical"]
    assert technical["type"] == "splatpipe.train.gpu_api.NonFiniteLoss"
    assert technical["message"] == message
    for words in ("at iteration 4", "l1 0.0731", "ssim 0.412", "opacity_reg 0.31",
                  "scale_reg 0.0204", "erank nan", "total nan"):
        assert words in technical["message"]


# Every splat with a NaN, and the splat of an earlier run (Req 8.1, 8.7, 8.9, 10.5, D6)

def test_with_every_splat_nan_the_run_fails_at_the_ply_check_with_the_warning_and_no_splat(
        new_job: Callable[..., str], fake_gpu: Callable[..., Any]) -> None:
    job_id = camera_solve_job(new_job)
    fake_gpu(nan_opacity="every")
    # drop_non_finite left out every splat, so the file written holds none, and the PLY_Check
    # refuses it with its own message (Req 8.7, 8.9).
    written = folder_of(job_id).resolve() / "work" / "train" / "train.ply"
    report = run_failed(job_id, f"{written} holds no splats")
    # The failed report gives the count dropped, every splat, and none was written (Req 10.5).
    assert report["warnings"] == [DROPPED_WARNING.format(count=POINTS)]
    assert report["counts"]["splats"] == 0
    assert not trained_splat(job_id).exists()


def test_a_failed_run_removes_the_out_train_ply_an_earlier_run_left(
        new_job: Callable[..., str], fake_gpu: Callable[..., Any],
        set_train: Callable[..., None]) -> None:
    job_id = camera_solve_job(new_job)
    fake_gpu()
    run_done(job_id)
    assert trained_splat(job_id).is_file()
    # The settings are the first thing the step checks once it has removed the splat (Req 8.1).
    set_train(**BAD_SETTING)
    run_failed(job_id, BAD_SETTING_MESSAGE)
    assert not trained_splat(job_id).exists()


# The CLI (Req 11.1, 11.2)

def test_the_cli_prints_train_done_and_gives_0_when_the_step_is_done(
        new_job: Callable[..., str], fake_gpu: Callable[..., Any],
        capsys: pytest.CaptureFixture[str]) -> None:
    job_id = camera_solve_job(new_job)
    fake_gpu()
    assert train_main(["--job", job_id]) == 0
    captured = capsys.readouterr()
    assert (captured.out, captured.err) == ("train: done\n", "")


def test_the_cli_prints_the_readable_error_on_stderr_and_gives_1_when_the_step_fails(
        new_job: Callable[..., str], fake_gpu: Callable[..., Any],
        capsys: pytest.CaptureFixture[str]) -> None:
    job_id = new_job()  # with no Camera_Solve_Output, as a Job is until the camera solve lands
    fake_gpu()
    assert train_main(["--job", job_id]) == 1
    captured = capsys.readouterr()
    report = read_json(folder_of(job_id) / "out" / "reports" / "train.json")
    assert (captured.out, captured.err) == ("", report["error"]["readable"] + "\n")
    assert "Traceback" not in captured.err
