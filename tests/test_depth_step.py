"""Tests for the Depth_Step in splatpipe/depth/, run through splatpipe.depth.run, so the
Step_Runner runs it and writes its real report (depth-priors spec, Req 18.4).

Each test makes its Job with new_job from tests/conftest.py, in the test config's Jobs_Folder,
then writes tests/camera_files.py's plane scene into the Job's work/cameras/ with
write_camera_solve_output: four 16 x 12 PINHOLE cameras that look at the plane z = 3 at an angle,
30 points on the plane, each seen by every camera, the four Frames as JPEGs of one colour, and a
Mask_File for the first image in images.bin. images.bin holds the images in the order IMAGES,
which isn't their names' order, so a step that sorted the names or listed images/ would show.
The camera's longer edge is below [depth]'s long_edge of 512, so the Prior_Size is the camera's
own 16 x 12. The Job's Preset is fast.

fake_depth_gpu installs a FakeDepthGpu, the Depth_Stand_In, as the step's DEPTH_GPU hook. It
finds CUDA and gives each Frame plane_moge_output of the plane scene, with MoGe's metres
FAKE_METRES_PER_UNIT (2) times the camera solve's units, so every Frame fits with a scale of 2
and an offset of 0, up to float32's rounding of MoGe's depths. That rounding leaves a few of the
30 points out of the second fit, so the tests check kept_points between fit_min_points and 30
rather than at 30, and the fit's numbers within FIT_ERROR and PRIOR_ERROR.

check_machine wants the Weights_File to be a file before the stand-in's load_model runs
(Req 3.5), so the autouse fixture weights_file writes a stand-in one outside the Jobs_Folder and
sets [depth]'s moge_weights to it with set_depth. depth_settings changes other [depth] keys and
keeps that path, unless the test names moge_weights too.

What a report doesn't show, a test reads from the stand-in's call log, from the files in
work/depth/, read back with the Train_Step's own read_prior_set and with zipfile, or from opened,
which records the path of each file opened from then on. The oracles are written here: the
Field_Of_View from its formula, the warning's wording from the design, the Prior_Index's fields
from Req 7.3, and each Prior from plane_moge_output, never from the step's own code.

The failed runs each break one thing, or two for the order tests, whose message shows which
check comes first (Req 1.4): a [depth] setting, with depth_settings or a test config that leaves
out a key; a path or a file of the Camera_Solve_Output, with write_camera_solve_output's options
or write_model's cut; what FakeDepthGpu does, which can find no CUDA, miss a package, refuse the
checkpoint, run out of memory or find huggingface_hub online; a full disk, from a patched
np.savez or os.replace; or MoGe's outputs, handed to the stand-in, so that no Frame fits. The test
of a machine with no torch installs no FakeDepthGpu and replaces splatpipe.depth.gpu_api.importlib
with NoTorch, so the step's real load_depth_gpu runs and no test imports the Depth_GPU_Module.
run_failed checks that each run failed with an Expected_Failure whose Readable_Error holds what
its requirement says and no traceback, and that it left no Prior_Index (Req 1.7, 7.8, 18.7). Each
message is written here as the design's Error Handling table words it. The test of the frame
that runs out of memory gives splatpipe.depth a Clock in place of time.perf_counter, so each
block the step times into infer, fit or write lasts exactly one second and the report's sums
show which blocks ended. The CLI tests run splatpipe.depth.__main__.main in this process, and
the run_job tests run the Depth_Step and the Train_Step together, the train step with fake_gpu's
FakeGpu, the GPU_Stand_In, on the same plane scene.
"""

import builtins
import errno
import math
import os
import shutil
import tomllib
import zipfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NoReturn

import numpy as np
import pytest

from splatpipe import depth
from splatpipe.config import Config, load_config
from splatpipe.depth.__main__ import main as depth_main
from splatpipe.depth.gpu_api import (DepthOutOfMemory, MissingPackage, MoGeOutput,
                                     OfflineModeOff, WrongCheckpoint)
from splatpipe.jsonio import read_json
from splatpipe.run import StepEntry, run_job
from splatpipe.train.colmap import SparseModel, read_model
from splatpipe.train.priors import read_prior_set

from camera_files import (CAMERA_SOLVE, IMAGES_FOLDER, JUNK, MASKS_FOLDER, PLANE_CAMERA,
                          PLANE_IMAGES, PLANE_POINTS, POINTS_BIN, SPARSE_FOLDER, ImageRecord,
                          plane_moge_output, write_camera_solve_output, write_model)
from conftest import DEPTH, FAKE_METRES_PER_UNIT

# The plane scene's images in images.bin's order here: 000002.jpg, 000000.jpg, 000003.jpg and
# then 000001.jpg.
IMAGES = tuple(PLANE_IMAGES[place] for place in (2, 0, 3, 1))
FRAMES = tuple(image.name for image in IMAGES)
POINT_IDS = tuple(point.point_id for point in PLANE_POINTS)  # 1 to 30, in points3D.bin's order
# The Prior_Size: the camera's own size, since its longer edge is below long_edge (Req 4.1).
PRIOR_WIDTH, PRIOR_HEIGHT = PLANE_CAMERA.width, PLANE_CAMERA.height
# The Field_Of_View, degrees(2 × atan(W / (2 × fx))): about 53.13° for the plane's camera
# (Req 4.3).
FIELD_OF_VIEW = math.degrees(2 * math.atan(PLANE_CAMERA.width / (2 * PLANE_CAMERA.fx)))
PEAK_BYTES = 3_456_789_012  # the peak GPU memory the report test gives FakeDepthGpu

# The most the plane's fit may stray from a scale of FAKE_METRES_PER_UNIT and an offset of 0,
# relative for the scale and the Metres_Per_Unit and absolute for the offset and the median
# error, from MoGe's depths being float32. The plane's Targets are about 1 / 3.
FIT_ERROR = 1e-6
# The most a Prior's Fitted_Inverse_Depth may stray, relative, from 1 / the plane's z: the fit's
# error and the value's rounding to float32.
PRIOR_ERROR = 1e-5

# The Depth_Report's counts, timings and 8 settings (Req 9.1, 9.2, 9.6, 8.2).
COUNTS = ("images", "prior_width", "prior_height", "fitted_images", "few_points_images",
          "flat_images", "poor_fit_images", "peak_gpu_memory_bytes")
TIMINGS = ("load_model", "infer", "fit", "write")
SETTINGS = ("long_edge", "moge_weights", "resolution_level", "use_fp16", "fit_min_points",
            "fit_miss_ratio", "fit_max_median_error", "min_gpu_memory_gb")
# The Prior_Index's fields and each entry's (Req 7.3), and a Prior_File's arrays (Req 7.1).
INDEX_FIELDS = ("format", "camera", "prior_width", "prior_height", "metres_per_unit", "priors")
ENTRY_FIELDS = ("name", "quaternion", "translation", "scale", "offset", "fit_points",
                "kept_points", "median_error", "metres_per_unit")
PRIOR_ARRAYS = ("inverse_depth", "normal", "mask")

# The Req 7.9 warning, worded as the design gives it.
LEFT_OUT_WARNING = ("Depth priors were made for {fitted} of {frames} frames. Of the others, "
                    "{few} had too few of the camera solve's points, {flat} were flat and "
                    "{poor} fitted poorly, so they train without a depth prior.")

# Files that images.bin doesn't name, in images/ and masks/, each holding JUNK, which OpenCV
# can't decode, so a step that read one as a Frame or a Mask_File would fail.
JUNK_FRAMES = ("000004.jpg", "notes.txt")
JUNK_MASKS = ("000004.png", "notes.txt")


@pytest.fixture(autouse=True)
def weights_file(tmp_path: Path, set_depth: Callable[..., None]) -> Path:
    """Write a stand-in Weights_File at tmp_path / "weights" / "model.pt", outside the
    Jobs_Folder, set [depth]'s moge_weights to it, and give its path. FakeDepthGpu never reads
    it: check_machine only checks that it's a file (Req 3.5)."""
    path = tmp_path / "weights" / "model.pt"
    path.parent.mkdir()
    path.write_bytes(b"stand-in MoGe-2 weights")
    set_depth(moge_weights=str(path))
    return path


@pytest.fixture
def depth_settings(weights_file: Path, set_depth: Callable[..., None]) -> Callable[..., None]:
    """Give depth_settings(**values), which rewrites the test config with each [depth] key it
    names set to its value, as set_depth does, and moge_weights still the stand-in Weights_File
    unless `values` names it."""
    def rewrite(**values: Any) -> None:
        set_depth(**{"moge_weights": str(weights_file), **values})

    return rewrite


def plane_job(new_job: Callable[..., str], **options: Any) -> str:
    """Make a Job with new_job, write the plane scene's Camera_Solve_Output into its
    work/cameras/, with images.bin in the order IMAGES and write_camera_solve_output's `options`,
    and give back its Job_Id."""
    job_id = new_job()
    write_camera_solve_output(folder_of(job_id), PLANE_CAMERA, IMAGES, PLANE_POINTS, **options)
    return job_id


def run_done(job_id: str) -> dict[str, Any]:
    """Run the depth step on the Job through splatpipe.depth.run, check that it's done, and give
    back its report."""
    report = depth.run(job_id)
    assert report["outcome"] == "done", report["error"]
    return report


def folder_of(job_id: str) -> Path:
    """The Job_Folder of `job_id` in the test config's Jobs_Folder."""
    return load_config().jobs().jobs_folder / job_id


def cameras_of(job_id: str) -> Path:
    """The Camera_Solve_Output's folder, work/cameras/, in the Job's folder."""
    return folder_of(job_id) / CAMERA_SOLVE


def priors_of(job_id: str) -> Path:
    """The Depth_Step's Work_Folder, work/depth/, in the Job's folder."""
    return folder_of(job_id) / "work" / "depth"


def entries(folder: Path) -> list[str]:
    """Every file and folder inside `folder`, by its path from `folder`, sorted."""
    return sorted(path.relative_to(folder).as_posix() for path in folder.rglob("*"))


def prior_files(frames: tuple[str, ...] | list[str]) -> list[str]:
    """The names of the Prior_Files of `frames`: each Frame's name without .jpg, then .npz."""
    return [frame.removesuffix(".jpg") + ".npz" for frame in frames]


def infer_calls(gpu: Any) -> list[tuple[Any, ...]]:
    """The arguments of each infer call FakeDepthGpu `gpu` recorded, in order: (model, rgb,
    fov_x, resolution_level, use_fp16, frame)."""
    return [args for name, args in gpu.calls if name == "infer"]


def opened(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    """A list that gets the path of each file opened from now on with Path.open, which
    Path.read_bytes, read_text, write_bytes and write_text open with, or with the built-in open,
    which np.load opens a path with. The real open still runs."""
    paths: list[Path] = []
    real_path_open, real_open = Path.open, builtins.open

    def path_open(self: Path, *args: Any, **kwargs: Any) -> Any:
        paths.append(Path(self))
        return real_path_open(self, *args, **kwargs)

    def any_open(file: Any, *args: Any, **kwargs: Any) -> Any:
        if isinstance(file, (str, bytes, os.PathLike)):  # not a file descriptor
            paths.append(Path(os.fsdecode(file)))
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(Path, "open", path_open)
    monkeypatch.setattr(builtins, "open", any_open)
    return paths


def strings_in(value: Any) -> list[str]:
    """Every string among the JSON value `value`'s values, in order. Keys aren't values."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [text for item in value.values() for text in strings_in(item)]
    if isinstance(value, list):
        return [text for item in value for text in strings_in(item)]
    return []


def landing_pixels(record: ImageRecord, keep: int) -> list[tuple[int, int]]:
    """The Prior_Size pixels, (column, row), that the points of `record` land on, but its first
    `keep` points in POINT_IDS. Each of its keypoints sits at its point's projection, and the
    Prior_Size pixel is (min(⌊u · w / W⌋, w − 1), min(⌊v · h / H⌋, h − 1)) (Req 5.1)."""
    return [(min(math.floor(u * PRIOR_WIDTH / PLANE_CAMERA.width), PRIOR_WIDTH - 1),
             min(math.floor(v * PRIOR_HEIGHT / PLANE_CAMERA.height), PRIOR_HEIGHT - 1))
            for u, v, point_id in record.keypoints if point_id in POINT_IDS[keep:]]


# A done run: the Prior_Files and the Prior_Index (Req 6.2, 6.3, 7.1, 7.3, 7.4)

def test_a_done_run_leaves_one_prior_file_per_frame_and_a_prior_index_of_the_planes_fit(
        new_job: Callable[..., str], fake_depth_gpu: Callable[..., Any]) -> None:
    job_id = plane_job(new_job)
    fake_depth_gpu()
    run_done(job_id)
    folder = priors_of(job_id)
    # One Prior_File per Frame, named after its stem, and the Prior_Index (Req 7.1, 7.3).
    assert entries(folder) == sorted([*prior_files(FRAMES), "depth.json"])
    # Each is numpy's uncompressed .npz, holding exactly the three arrays (Req 7.1).
    for name in prior_files(FRAMES):
        with zipfile.ZipFile(folder / name) as archive:
            assert sorted(archive.namelist()) == sorted(f"{array}.npy" for array in PRIOR_ARRAYS)
            assert all(member.compress_type == zipfile.ZIP_STORED
                       for member in archive.infolist()), name
    # The index holds exactly Req 7.3's fields: the plane's camera, the Prior_Size and the job's
    # Metres_Per_Unit, MoGe's metres per camera solve unit (Req 6.2, 6.3).
    index = read_json(folder / "depth.json")
    assert sorted(index) == sorted(INDEX_FIELDS)
    assert index["format"] == 1
    assert index["camera"] == {"width": PLANE_CAMERA.width, "height": PLANE_CAMERA.height,
                               "fx": PLANE_CAMERA.fx, "fy": PLANE_CAMERA.fy,
                               "cx": PLANE_CAMERA.cx, "cy": PLANE_CAMERA.cy}
    assert (index["prior_width"], index["prior_height"]) == (PRIOR_WIDTH, PRIOR_HEIGHT)
    assert index["metres_per_unit"] == pytest.approx(FAKE_METRES_PER_UNIT, rel=FIT_ERROR)
    # One entry per Fitted_Frame, in images.bin's order, with its pose exactly as images.bin
    # holds it and the plane's fit: all 30 points, a scale of FAKE_METRES_PER_UNIT and an offset
    # of 0 (Req 6.3, 7.3).
    assert [entry["name"] for entry in index["priors"]] == list(FRAMES)
    for entry, record in zip(index["priors"], IMAGES, strict=True):
        assert sorted(entry) == sorted(ENTRY_FIELDS)
        assert entry["quaternion"] == list(record.quaternion)
        assert entry["translation"] == list(record.translation)
        assert entry["scale"] == pytest.approx(FAKE_METRES_PER_UNIT, rel=FIT_ERROR)
        assert abs(entry["offset"]) <= FIT_ERROR
        assert entry["fit_points"] == len(POINT_IDS)
        assert DEPTH["fit_min_points"] <= entry["kept_points"] <= len(POINT_IDS)
        assert 0 <= entry["median_error"] <= FIT_ERROR
        assert entry["metres_per_unit"] == pytest.approx(FAKE_METRES_PER_UNIT, rel=FIT_ERROR)
    # The index names no path: its only text is the Frames' names (Req 7.4).
    assert strings_in(index) == list(FRAMES)


def test_read_prior_set_reads_the_done_runs_priors_back_against_the_model_as_the_planes(
        new_job: Callable[..., str], fake_depth_gpu: Callable[..., Any]) -> None:
    job_id = plane_job(new_job)
    fake_depth_gpu()
    run_done(job_id)
    model = read_model(cameras_of(job_id) / SPARSE_FOLDER)
    # The Train_Step's own reader passes every check of the index and the files.
    prior_set = read_prior_set(priors_of(job_id), model)
    assert prior_set is not None
    assert list(prior_set.priors) == list(FRAMES)
    for image in model.images:
        prior = prior_set.priors[image.name]
        plane = plane_moge_output(model, image, PRIOR_WIDTH, PRIOR_HEIGHT, FAKE_METRES_PER_UNIT)
        # MoGe found geometry at every pixel, so every pixel has a prior, whose inverse depth is
        # 1 / the plane's z in camera solve units, MoGe's metres / FAKE_METRES_PER_UNIT, and
        # whose normal is MoGe's (Req 5.11, 7.1).
        assert prior.mask.all(), image.name
        assert np.allclose(prior.inverse_depth,
                           FAKE_METRES_PER_UNIT / plane.depth.astype(np.float64),
                           rtol=PRIOR_ERROR, atol=0), image.name
        assert np.array_equal(prior.normal, plane.normal), image.name


# The Depth_Report (Req 9.1, 9.2, 9.4, 9.6)

def test_the_report_holds_every_count_as_an_int_the_four_timings_and_the_8_settings(
        new_job: Callable[..., str], fake_depth_gpu: Callable[..., Any]) -> None:
    job_id = plane_job(new_job)
    fake_depth_gpu(peak_bytes=PEAK_BYTES)
    report = run_done(job_id)
    # The Step_Runner's ordinary report, in out/reports/depth.json (Req 9.6).
    assert read_json(folder_of(job_id) / "out" / "reports" / "depth.json") == report
    assert (report["step"], report["warnings"], report["error"]) == ("depth", [], None)
    # Every count of Req 9.1, each an int (Req 9.4), with the stand-in's peak memory.
    assert sorted(report["counts"]) == sorted(COUNTS)
    assert report["counts"] == {
        "images": len(FRAMES), "prior_width": PRIOR_WIDTH, "prior_height": PRIOR_HEIGHT,
        "fitted_images": len(FRAMES), "few_points_images": 0, "flat_images": 0,
        "poor_fit_images": 0, "peak_gpu_memory_bytes": PEAK_BYTES}
    assert all(type(value) is int for value in report["counts"].values())
    # The four timings, each a number of seconds (Req 9.2, 9.4).
    assert sorted(report["timings"]) == sorted(TIMINGS)
    assert all(type(seconds) is float and seconds >= 0
               for seconds in report["timings"].values())
    # The 8 settings of [depth], as the test config has them (Req 9.6).
    assert sorted(report["settings"]) == sorted(SETTINGS)
    assert report["settings"] == load_config().step_settings("depth", "fast")


# What MoGe gets, and what the step reads (Req 2.1, 2.8, 4.5)

def test_each_frame_goes_to_infer_once_in_images_bin_order_with_the_field_of_view(
        new_job: Callable[..., str], fake_depth_gpu: Callable[..., Any]) -> None:
    job_id = plane_job(new_job)
    gpu = fake_depth_gpu()
    run_done(job_id)
    calls = infer_calls(gpu)
    # One call per Frame, in images.bin's order, which isn't the names' order (Req 2.8, 4.5).
    assert [frame for *_, frame in calls] == list(FRAMES)
    for model, rgb, fov_x, resolution_level, use_fp16, frame in calls:
        # The loaded model, the Frame at the Prior_Size, the Field_Of_View from the camera's
        # width and fx, and resolution_level and use_fp16 from [depth] (Req 4.3, 4.4).
        assert model is gpu.handle, frame
        assert (rgb.shape, rgb.dtype) == ((PRIOR_HEIGHT, PRIOR_WIDTH, 3), np.uint8), frame
        assert fov_x == pytest.approx(FIELD_OF_VIEW, rel=1e-12), frame
        assert (resolution_level, use_fp16) == (DEPTH["resolution_level"], DEPTH["use_fp16"])


def test_the_step_opens_only_the_frames_images_bin_names_and_no_file_in_masks(
        new_job: Callable[..., str], fake_depth_gpu: Callable[..., Any],
        monkeypatch: pytest.MonkeyPatch) -> None:
    job_id = plane_job(new_job, extra_images=JUNK_FRAMES, extra_masks=JUNK_MASKS)
    cameras = cameras_of(job_id)
    images, masks = cameras / IMAGES_FOLDER, cameras / MASKS_FOLDER
    # masks/ holds the first image's Mask_File and the junk.
    assert entries(masks) == sorted([FRAMES[0].removesuffix(".jpg") + ".png", *JUNK_MASKS])
    fake_depth_gpu()
    paths = opened(monkeypatch)
    # The run is done, so no junk file was decoded as a Frame.
    run_done(job_id)
    resolved = [path.resolve() for path in paths]
    # The files opened in images/ are the Frames images.bin names, each once, in its order, so
    # the junk was never opened (Req 2.8). Nothing in masks/ was opened (Req 2.1).
    assert [path for path in resolved if path.parent == images.resolve()] == [
        (images / frame).resolve() for frame in FRAMES]
    assert [path for path in resolved if path.is_relative_to(masks.resolve())] == []


# A second run (Req 1.11)

def test_a_second_run_after_junk_is_put_in_work_depth_leaves_only_its_own_files(
        new_job: Callable[..., str], fake_depth_gpu: Callable[..., Any]) -> None:
    job_id = plane_job(new_job)
    fake_depth_gpu()
    run_done(job_id)
    folder = priors_of(job_id)
    # A Prior_File of a Frame that images.bin doesn't name, a stray file and a folder.
    for name in ("000009.npz", "notes.txt", "old/000000.npz"):
        (folder / name).parent.mkdir(exist_ok=True)
        (folder / name).write_bytes(JUNK)
    fake_depth_gpu()
    run_done(job_id)
    # The Step_Runner emptied work/depth/, so only the second run's files are there, and the
    # Train_Step's reader reads them (Req 1.11).
    assert entries(folder) == sorted([*prior_files(FRAMES), "depth.json"])
    assert read_prior_set(folder, read_model(cameras_of(job_id) / SPARSE_FOLDER)) is not None


# A Frame left out (Req 7.9, 9.5)

def test_a_frame_left_with_too_few_fit_points_is_left_out_with_the_warning_and_the_counts(
        new_job: Callable[..., str], fake_depth_gpu: Callable[..., Any]) -> None:
    job_id = plane_job(new_job)
    # MoGe finds no geometry under all but fit_min_points - 1 of the second Frame's points, so
    # it has 9 Fit_Points, fewer than fit_min_points (Req 5.2).
    left_out = IMAGES[1]
    gpu = fake_depth_gpu(no_geometry={
        left_out.name: landing_pixels(left_out, DEPTH["fit_min_points"] - 1)})
    report = run_done(job_id)
    fitted = [frame for frame in FRAMES if frame != left_out.name]
    # The warning gives the Fitted_Frames, the Frames and the Frames left out by reason, and the
    # report holds it with the same counts (Req 7.9, 9.1, 9.5).
    assert report["warnings"] == [LEFT_OUT_WARNING.format(
        fitted=len(fitted), frames=len(FRAMES), few=1, flat=0, poor=0)]
    assert {key: report["counts"][key] for key in COUNTS[3:7]} == {
        "fitted_images": len(fitted), "few_points_images": 1, "flat_images": 0,
        "poor_fit_images": 0}
    assert report["counts"]["images"] == len(FRAMES)
    # MoGe still ran on every Frame, but the one left out has no Prior_File and no entry, and the
    # others' Prior_Index reads back (Req 4.5, 7.1, 7.3).
    assert [frame for *_, frame in infer_calls(gpu)] == list(FRAMES)
    folder = priors_of(job_id)
    assert entries(folder) == sorted([*prior_files(fitted), "depth.json"])
    index = read_json(folder / "depth.json")
    assert [entry["name"] for entry in index["priors"]] == fitted
    assert index["metres_per_unit"] == pytest.approx(FAKE_METRES_PER_UNIT, rel=FIT_ERROR)
    prior_set = read_prior_set(folder, read_model(cameras_of(job_id) / SPARSE_FOLDER))
    assert prior_set is not None and list(prior_set.priors) == fitted


# What the step writes (Req 1.9, 18.6)

def test_a_done_run_adds_only_work_depth_files_and_its_report_to_the_job_folder(
        new_job: Callable[..., str], fake_depth_gpu: Callable[..., Any]) -> None:
    job_id = plane_job(new_job)
    folder = folder_of(job_id)
    before = entries(folder)
    fake_depth_gpu()
    run_done(job_id)
    # Nothing is gone, and the only new files are the Prior_Files, the Prior_Index and the
    # Depth_Report. work/depth/ and out/reports/ are the folders the Step_Runner makes for them.
    written = [f"work/depth/{name}" for name in [*prior_files(FRAMES), "depth.json"]]
    assert entries(folder) == sorted([*before, "work/depth", *written, "out/reports",
                                      "out/reports/depth.json"])


# Failed runs

# Every pixel of the Prior_Size, (column, row): FakeDepthGpu's no_geometry for a Frame on which
# MoGe finds no geometry at all, so it has no Fit_Point.
ALL_PIXELS = tuple((x, y) for y in range(PRIOR_HEIGHT) for x in range(PRIOR_WIDTH))

# The stand-in's calls, by name, in a done run: the checks, the load, one infer per Frame in
# images.bin's order, then the release and the peak memory (Req 1.4, 3.14, 9.1).
DONE_CALLS = ["probe", "import_moge", "load_model", *["infer"] * len(FRAMES), "release",
              "peak_memory_bytes"]
# The calls of a run that fails in the load, or once the load is done but before any Frame.
LOAD_CALLS = ["probe", "import_moge", "load_model", "release", "peak_memory_bytes"]

# The Readable_Errors, worded as the design's Error Handling table gives them.
BAD_SETTING = {"resolution_level": 10}
BAD_SETTING_MESSAGE = ("The setting resolution_level in [depth] is 10, but it must be a whole "
                       "number from 0 to 9. Change it to a whole number from 0 to 9.")
NO_CUDA_MESSAGE = ("The depth step needs an NVIDIA GPU, but torch can't see one on this machine. "
                   "Check the driver with nvidia-smi, or run the job with "
                   "`uv run python -m cloud.start`.")
CHECKPOINT_REASON = "it holds no model_config"  # the WrongCheckpoint reason the tests give
# The Req 3.11 Readable_Error for CHECKPOINT_REASON, with {weights} for the Weights_File's path.
WRONG_CHECKPOINT_MESSAGE = ("The file at {weights} isn't the MoGe-2 ViT-B normal checkpoint "
                            f"({CHECKPOINT_REASON}). Rebuild the machine image with "
                            "`uv run python -m cloud.image`.")
LOAD_OUT_OF_MEMORY_MESSAGE = ("The GPU ran out of memory while the depth step loaded MoGe-2. The "
                              "job goes on and trains without depth priors.")
NO_FIT_FAILURE = ("None of the {frames} frames fitted the camera solve's points, so no depth "
                  "priors were made: {few} had fewer than {fit_min_points} of the points, {flat} "
                  "were flat, and {poor} fitted poorly, with a median depth error above "
                  "{fit_max_median_error}. The job goes on and trains without depth priors.")
# The Step_Runner's general Readable_Error for a bug in the depth step, and OfflineModeOff's own
# message, which goes in the Technical_Detail (Req 3.8).
BUG_MESSAGE = ("The depth step stopped because of an error in its code. This is a bug, not a "
               "problem with your video. Send out/reports/depth.json to the team.")
OFFLINE_MODE_OFF_MESSAGE = ("huggingface_hub was imported before HF_HUB_OFFLINE was set, so it "
                            "isn't offline and could download. This is a bug in the depth step.")


def no_package_message(package: str) -> str:
    """The Req 3.2 Readable_Error for a machine with no `package`."""
    return ("The depth step runs only on a GPU machine image built with MoGe's code, and this "
            f"machine has no {package}. Rebuild the machine image with "
            "`uv run python -m cloud.image`, or run the job there with "
            "`uv run python -m cloud.start`.")


def missing_input_message(path: Path) -> str:
    """The Req 2.2 Readable_Error for `path`, a path of the Camera_Solve_Output that isn't there."""
    return (f"The depth step needs {path}, which the camera solve step writes, but it isn't "
            "there. Run the camera solve step on this job first.")


def missing_weights_message(path: Path) -> str:
    """The Req 3.5 Readable_Error for a moge_weights of `path`, which isn't a file."""
    return (f"The MoGe-2 weights aren't at {path}, the moge_weights setting in [depth]. The image "
            "build downloads them there, so rebuild the machine image with "
            "`uv run python -m cloud.image`.")


def frame_out_of_memory_message(frame: str) -> str:
    """The Req 3.13 Readable_Error for running out of GPU memory on the Frame `frame`."""
    return (f"The GPU ran out of memory while the depth step ran MoGe-2 on the frame {frame!r}. "
            "The job goes on and trains without depth priors.")


def not_written_message(path: Path) -> str:
    """The Req 7.10 Readable_Error for the file `path`, on a full disk."""
    return (f"The file {path} couldn't be written (No space left on device). Free some space on "
            f"the disk, or check that you can write to {path.parent}, then run the step again.")


def run_failed(job_id: str, *phrases: str, config: Config | None = None) -> dict[str, Any]:
    """Run the depth step on the Job through splatpipe.depth.run, with `config` when it's given,
    check that it failed with an Expected_Failure whose Readable_Error holds each of `phrases` and
    no traceback, and that it left no Prior_Index and no temporary one, and give back its report
    (Req 1.7, 7.8, 18.7)."""
    report = depth.run(job_id, config=config)
    assert report["outcome"] == "failed"
    error = report["error"]
    assert error["expected"] is True, error
    readable = error["readable"]
    for phrase in phrases:
        assert phrase in readable, readable
    assert "Traceback" not in readable
    folder = priors_of(job_id)
    assert not (folder / "depth.json").exists()
    assert not (folder / "depth.json.tmp").exists()
    return report


def call_names(gpu: Any) -> list[str]:
    """The name of each call FakeDepthGpu `gpu` recorded, in order."""
    return [name for name, _ in gpu.calls]


def resolved_cameras(job_id: str) -> Path:
    """work/cameras/ in the Job's folder, as the step's messages give it, from the resolved
    Job_Folder."""
    return folder_of(job_id).resolve() / CAMERA_SOLVE


def with_corrupt_points(job_id: str) -> None:
    """Write the plane scene's model again into the Job's work/cameras/sparse/, with
    points3D.bin cut one byte short, so the file ends inside its last point's track."""
    write_model(cameras_of(job_id) / SPARSE_FOLDER, PLANE_CAMERA, IMAGES, PLANE_POINTS,
                cut={POINTS_BIN: 1})


def corrupt_points_message(job_id: str) -> str:
    """The start of the Model_Reader's Readable_Error for the Job's points3D.bin, cut short."""
    path = resolved_cameras(job_id) / SPARSE_FOLDER / POINTS_BIN
    return f"{path} is cut short or isn't a COLMAP binary model"


# The order of Req 1.4 (Req 1.4, 3.1, 8.3, 2.2): a done run's calls, and runs with two faults, where
# the one the step checks first gives the message.

def test_a_done_run_checks_loads_runs_each_frame_then_releases_in_the_order_of_req_1_4(
        new_job: Callable[..., str], fake_depth_gpu: Callable[..., Any],
        weights_file: Path) -> None:
    job_id = plane_job(new_job)
    gpu = fake_depth_gpu()
    run_done(job_id)
    # The GPU check, MoGe's import, the load from moge_weights, one infer per Frame, then the
    # release of the very handle the load gave, then the peak memory (Req 1.4, 3.14).
    assert call_names(gpu) == DONE_CALLS
    assert gpu.calls[2] == ("load_model", (weights_file,))
    assert gpu.calls[-2] == ("release", (gpu.handle,))


def test_the_prior_index_is_written_last_after_every_prior_file_and_the_release(
        new_job: Callable[..., str], fake_depth_gpu: Callable[..., Any],
        monkeypatch: pytest.MonkeyPatch) -> None:
    job_id = plane_job(new_job)
    gpu = fake_depth_gpu()
    seen: list[tuple[list[str], list[str]]] = []
    real_write = depth.write_prior_index

    def recorded_write(folder: Path, index: Any) -> None:
        seen.append((entries(folder), call_names(gpu)))
        real_write(folder, index)

    monkeypatch.setattr(depth, "write_prior_index", recorded_write)
    run_done(job_id)
    # When the index is written, once, every Prior_File is there and MoGe-2 is released (Req 1.4,
    # 7.6).
    assert seen == [(sorted(prior_files(FRAMES)), DONE_CALLS)]


def test_a_bad_setting_with_no_work_cameras_gives_the_settings_message(
        new_job: Callable[..., str], fake_depth_gpu: Callable[..., Any],
        depth_settings: Callable[..., None]) -> None:
    job_id = new_job()  # with no Camera_Solve_Output
    gpu = fake_depth_gpu()
    depth_settings(**BAD_SETTING)
    # The settings are checked before the Camera_Solve_Output (Req 8.3).
    assert run_failed(job_id)["error"]["readable"] == BAD_SETTING_MESSAGE
    assert gpu.calls == []


def test_no_work_cameras_gives_the_camera_solve_message_with_no_stand_in_call(
        new_job: Callable[..., str], fake_depth_gpu: Callable[..., Any]) -> None:
    job_id = new_job()
    gpu = fake_depth_gpu(False)  # it would refuse, if the step asked it anything
    # images/ is the first path the step checks, before it reaches the Depth_GPU_Module (Req 2.2).
    assert run_failed(job_id)["error"]["readable"] == missing_input_message(
        resolved_cameras(job_id) / IMAGES_FOLDER)
    assert gpu.calls == []


def test_a_refusing_probe_with_a_corrupt_points3d_bin_gives_the_gpu_message(
        new_job: Callable[..., str], fake_depth_gpu: Callable[..., Any]) -> None:
    job_id = plane_job(new_job)
    with_corrupt_points(job_id)
    gpu = fake_depth_gpu(False)
    # The GPU is checked before the Sparse_Model is read, and nothing else reaches the stand-in
    # (Req 3.1).
    assert run_failed(job_id)["error"]["readable"] == NO_CUDA_MESSAGE
    assert gpu.calls == [("probe", ())]


def test_a_failed_load_with_a_corrupt_points3d_bin_gives_the_loads_message(
        new_job: Callable[..., str], fake_depth_gpu: Callable[..., Any],
        weights_file: Path) -> None:
    job_id = plane_job(new_job)
    with_corrupt_points(job_id)
    gpu = fake_depth_gpu(load_model_error=WrongCheckpoint(CHECKPOINT_REASON))
    # MoGe-2 is loaded before the Sparse_Model is read (Req 1.4, 3.1).
    assert run_failed(job_id)["error"]["readable"] == WRONG_CHECKPOINT_MESSAGE.format(
        weights=weights_file)
    assert call_names(gpu) == LOAD_CALLS


def test_a_corrupt_points3d_bin_is_read_after_the_load_and_before_any_frame(
        new_job: Callable[..., str], fake_depth_gpu: Callable[..., Any]) -> None:
    job_id = plane_job(new_job)
    with_corrupt_points(job_id)
    gpu = fake_depth_gpu()
    # The Model_Reader's own message (Req 2.3), once the load is done and before any infer, and
    # the model loaded is released (Req 1.4, 3.14).
    run_failed(job_id, corrupt_points_message(job_id))
    assert call_names(gpu) == LOAD_CALLS
    assert gpu.calls[-2] == ("release", (gpu.handle,))


# The Expected_Failures that need no GPU (Req 1.7, 18.7), each naming what its requirement says

@pytest.mark.parametrize(("values", "message"), [
    pytest.param(BAD_SETTING, BAD_SETTING_MESSAGE, id="a value out of range"),
    pytest.param({"fit_min_points": True},
                 "The setting fit_min_points in [depth] is true, but it must be a whole number "
                 "above 0. Change it to a whole number above 0.", id="a bool for a number"),
    pytest.param({"use_fp16": 1},
                 "The setting use_fp16 in [depth] is 1, but it must be true or false. Change it "
                 "to true or false.", id="a number for true or false"),
])
def test_a_setting_the_step_refuses_fails_it_naming_the_setting_its_table_and_kind(
        new_job: Callable[..., str], fake_depth_gpu: Callable[..., Any],
        depth_settings: Callable[..., None], values: dict[str, Any], message: str) -> None:
    job_id = plane_job(new_job)
    fake_depth_gpu()
    depth_settings(**values)
    assert run_failed(job_id)["error"]["readable"] == message  # Req 8.3


def test_a_missing_depth_key_fails_the_step_with_the_step_runners_message(
        new_job: Callable[..., str], fake_depth_gpu: Callable[..., Any],
        temp_config: Path) -> None:
    job_id = plane_job(new_job)
    fake_depth_gpu()
    data = tomllib.loads(temp_config.read_text(encoding="utf-8"))
    del data["depth"]["fit_min_points"]
    # The Step_Runner's message names the section, the Preset's table and the key (Req 8.6).
    run_failed(job_id, "The depth step needs the setting fit_min_points",
               "neither [depth] nor [depth.fast]", config=Config(data))


@pytest.mark.parametrize("part", ["images", "sparse/cameras.bin", "sparse/images.bin",
                                  "sparse/points3D.bin"])
def test_each_missing_input_path_fails_the_step_naming_it_and_the_camera_solve_step(
        new_job: Callable[..., str], fake_depth_gpu: Callable[..., Any], part: str) -> None:
    job_id = plane_job(new_job)
    path = cameras_of(job_id) / part
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()
    gpu = fake_depth_gpu()
    # The message names the path, and that the camera solve step writes it (Req 2.2).
    assert run_failed(job_id)["error"]["readable"] == missing_input_message(
        resolved_cameras(job_id) / part)
    assert gpu.calls == []


@dataclass
class NoTorch:
    """A stand-in for splatpipe.depth.gpu_api's importlib on a machine with no torch. Its
    import_module records each name it's asked to import in `names`, then raises the
    ModuleNotFoundError that a missing torch gives."""

    names: list[str] = field(default_factory=list)

    def import_module(self, name: str) -> NoReturn:
        """Record `name`, then raise as Python does when torch isn't installed."""
        self.names.append(name)
        raise ModuleNotFoundError("No module named 'torch'", name="torch")


def test_a_machine_with_no_torch_fails_the_step_saying_it_runs_only_on_the_gpu_image(
        new_job: Callable[..., str], monkeypatch: pytest.MonkeyPatch) -> None:
    job_id = plane_job(new_job)
    # No FakeDepthGpu, so the step's hook is the real load_depth_gpu, which imports through the
    # stand-in.
    stand_in = NoTorch()
    monkeypatch.setattr("splatpipe.depth.gpu_api.importlib", stand_in)
    assert run_failed(job_id)["error"]["readable"] == no_package_message("torch")  # Req 3.2
    assert stand_in.names == ["splatpipe.depth.gpu"]


# What FakeDepthGpu is set to do, the Readable_Error it gives, with {weights} for the
# Weights_File's path, and the stand-in's calls: no CUDA (Req 3.3), a package MoGe's import is
# missing (Req 3.2), a checkpoint the strict load refuses (Req 3.11), and too little GPU memory to
# load MoGe-2 (Req 3.13). A failure before the load releases nothing, and one in it releases the
# None the load left (Req 3.14).
GPU_FAILURES = [
    pytest.param({"cuda": False}, NO_CUDA_MESSAGE, ["probe"], id="no CUDA"),
    pytest.param({"import_moge_error": MissingPackage("utils3d_moge")},
                 no_package_message("utils3d_moge"), ["probe", "import_moge"],
                 id="no utils3d_moge"),
    pytest.param({"load_model_error": WrongCheckpoint(CHECKPOINT_REASON)},
                 WRONG_CHECKPOINT_MESSAGE, LOAD_CALLS, id="a wrong checkpoint"),
    pytest.param({"load_model_error": DepthOutOfMemory(None)}, LOAD_OUT_OF_MEMORY_MESSAGE,
                 LOAD_CALLS, id="out of memory while loading"),
]


@pytest.mark.parametrize(("faults", "message", "calls"), GPU_FAILURES)
def test_a_gpu_or_a_moge_the_step_cant_use_fails_it_with_its_message(
        new_job: Callable[..., str], fake_depth_gpu: Callable[..., Any], weights_file: Path,
        faults: dict[str, Any], message: str, calls: list[str]) -> None:
    job_id = plane_job(new_job)
    gpu = fake_depth_gpu(**faults)
    assert run_failed(job_id)["error"]["readable"] == message.format(weights=weights_file)
    assert call_names(gpu) == calls


def test_a_moge_weights_that_isnt_a_file_fails_naming_it_before_moge_is_imported(
        new_job: Callable[..., str], fake_depth_gpu: Callable[..., Any],
        depth_settings: Callable[..., None], weights_file: Path) -> None:
    job_id = plane_job(new_job)
    missing = weights_file.parent / "missing.pt"
    depth_settings(moge_weights=str(missing))
    gpu = fake_depth_gpu()
    # The path and the rebuild, and nothing imports MoGe or loads a file (Req 3.5).
    assert run_failed(job_id)["error"]["readable"] == missing_weights_message(missing)
    assert call_names(gpu) == ["probe"]


# A Camera_Solve_Output whose Frames the step refuses, with the Train_Step's messages, {images}
# being work/cameras/images/: a Frame with no file, one OpenCV can't decode and one a pixel wider
# than the Shared_Camera (Req 2.7). Each is a Frame after the first in images.bin's order.
FRAME_FAILURES = [
    pytest.param({"left_out": {FRAMES[2]}},
                 f"images.bin names the image {FRAMES[2]!r}, but {{images}} has no file of that "
                 "name.", id="a frame with no file"),
    pytest.param({"junk": {FRAMES[1]}},
                 f"The training image {FRAMES[1]!r} in {{images}} can't be read as a colour JPEG: "
                 "OpenCV couldn't decode it.", id="a frame OpenCV can't decode"),
    pytest.param({"image_sizes": {FRAMES[3]: (PLANE_CAMERA.width + 1, PLANE_CAMERA.height)}},
                 f"The training image {FRAMES[3]!r} in {{images}} is "
                 f"{PLANE_CAMERA.width + 1} x {PLANE_CAMERA.height} pixels, but the shared camera "
                 f"in cameras.bin is {PLANE_CAMERA.width} x {PLANE_CAMERA.height} pixels",
                 id="a frame of another size"),
]


@pytest.mark.parametrize(("options", "phrase"), FRAME_FAILURES)
def test_a_frame_the_step_cant_use_fails_it_with_the_train_steps_message(
        new_job: Callable[..., str], fake_depth_gpu: Callable[..., Any],
        options: dict[str, Any], phrase: str) -> None:
    job_id = plane_job(new_job, **options)
    gpu = fake_depth_gpu()
    images = resolved_cameras(job_id) / IMAGES_FOLDER
    run_failed(job_id, phrase.format(images=images))
    # MoGe-2 ran on the Frames before it, and was released (Req 3.14).
    assert call_names(gpu)[-2:] == ["release", "peak_memory_bytes"]


def no_space(*args: Any, **kwargs: Any) -> NoReturn:
    """Raise the OSError of a full disk, whatever the call."""
    raise OSError(errno.ENOSPC, "No space left on device")


@pytest.mark.parametrize(("target", "written", "left"), [
    pytest.param("numpy.savez", prior_files(FRAMES[:1])[0], prior_files(FRAMES[:1]),
                 id="a prior file"),
    pytest.param("os.replace", "depth.json", prior_files(FRAMES), id="the prior index"),
])
def test_a_full_disk_fails_the_step_naming_the_file_and_the_reason(
        new_job: Callable[..., str], fake_depth_gpu: Callable[..., Any],
        monkeypatch: pytest.MonkeyPatch, target: str, written: str, left: list[str]) -> None:
    job_id = plane_job(new_job)
    fake_depth_gpu()
    monkeypatch.setattr(target, no_space)
    path = folder_of(job_id).resolve() / "work" / "depth" / written
    assert run_failed(job_id)["error"]["readable"] == not_written_message(path)  # Req 7.10
    # No index and no temporary one, so the Train_Step trains without priors (Req 7.6, 7.8). The
    # first Prior_File's write opened its file before np.savez failed.
    assert entries(priors_of(job_id)) == sorted(left)


def test_offline_mode_off_is_an_unexpected_error_with_its_message_in_technical(
        new_job: Callable[..., str], fake_depth_gpu: Callable[..., Any]) -> None:
    job_id = plane_job(new_job)
    gpu = fake_depth_gpu(import_moge_error=OfflineModeOff())
    report = depth.run(job_id)
    assert report["outcome"] == "failed"
    error = report["error"]
    # A bug, not an Expected_Failure: the general message, with OfflineModeOff's own in the
    # Technical_Detail (Req 3.8).
    assert (error["expected"], error["readable"]) == (False, BUG_MESSAGE)
    technical = error["technical"]
    assert technical["type"] == "splatpipe.depth.gpu_api.OfflineModeOff"
    assert technical["message"] == OFFLINE_MODE_OFF_MESSAGE
    assert call_names(gpu) == ["probe", "import_moge"]
    assert not (priors_of(job_id) / "depth.json").exists()


# No Frame fits (Req 7.7, D3)

FLAT_DEPTH = 6.0  # metres, MoGe's depth at every pixel of the flat Frame
OFF_FACTORS = (0.8, 1.0, 1.25)  # what off_output multiplies MoGe's depth under each point by
# The fit_max_median_error that the no-fit test sets. The plane's narrow range of depths keeps the
# median error of off_output's Frame a few percent, below the real 0.15, and far above this.
TIGHT_MEDIAN_ERROR = 0.001


def plane_output(model: SparseModel, frame: str) -> MoGeOutput:
    """plane_moge_output of the plane scene's image named `frame` at the Prior_Size, with MoGe's
    metres FAKE_METRES_PER_UNIT times the camera solve's units: an exact fit."""
    image = next(image for image in model.images if image.name == frame)
    return plane_moge_output(model, image, PRIOR_WIDTH, PRIOR_HEIGHT, FAKE_METRES_PER_UNIT)


def flat_output(model: SparseModel, frame: str) -> MoGeOutput:
    """A MoGe_Output for `frame` whose depth is FLAT_DEPTH at every pixel, every one a
    Geometry_Pixel, so every MoGe_Value is the same and the Spread of the MoGe_Values is 0."""
    plane = plane_output(model, frame)
    return MoGeOutput(depth=np.full_like(plane.depth, FLAT_DEPTH), normal=plane.normal,
                      mask=np.ones_like(plane.mask))


def off_output(model: SparseModel, record: ImageRecord) -> MoGeOutput:
    """The exact plane_output of `record`, with MoGe's depth under each of its points, in its
    keypoints' order, times OFF_FACTORS in turn, so no scale and offset fit every point."""
    plane = plane_output(model, record.name)
    depth_map = plane.depth.copy()
    for number, (x, y) in enumerate(landing_pixels(record, 0)):
        depth_map[y, x] *= OFF_FACTORS[number % len(OFF_FACTORS)]
    return MoGeOutput(depth=depth_map, normal=plane.normal, mask=plane.mask)


def test_no_frame_fitting_fails_with_the_counts_of_each_reason_and_leaves_work_depth_empty(
        new_job: Callable[..., str], fake_depth_gpu: Callable[..., Any],
        depth_settings: Callable[..., None]) -> None:
    job_id = plane_job(new_job)
    depth_settings(fit_max_median_error=TIGHT_MEDIAN_ERROR)
    model = read_model(cameras_of(job_id) / SPARSE_FOLDER)
    # Two Frames with too few Fit_Points, none and fit_min_points - 1, a flat one and one that
    # fits poorly (Req 5.2, 5.3, 5.9).
    gpu = fake_depth_gpu(
        no_geometry={FRAMES[0]: ALL_PIXELS,
                     FRAMES[1]: landing_pixels(IMAGES[1], DEPTH["fit_min_points"] - 1)},
        outputs={FRAMES[2]: flat_output(model, FRAMES[2]),
                 FRAMES[3]: off_output(model, IMAGES[3])})
    report = run_failed(job_id)
    # The message gives the Frames and how many were left out for each reason, and says the job
    # goes on without depth priors (Req 7.7, D3).
    assert report["error"]["readable"] == NO_FIT_FAILURE.format(
        frames=len(FRAMES), few=2, fit_min_points=DEPTH["fit_min_points"], flat=1, poor=1,
        fit_max_median_error=TIGHT_MEDIAN_ERROR)
    assert {key: report["counts"][key] for key in COUNTS[:7]} == {
        "images": len(FRAMES), "prior_width": PRIOR_WIDTH, "prior_height": PRIOR_HEIGHT,
        "fitted_images": 0, "few_points_images": 2, "flat_images": 1, "poor_fit_images": 1}
    assert report["warnings"] == []
    # MoGe-2 ran on every Frame, and nothing was written in work/depth/ (Req 7.7).
    assert call_names(gpu) == DONE_CALLS
    assert entries(priors_of(job_id)) == []


# Out of GPU memory on a Frame (Req 3.13, 7.8, 9.3)

class Clock:
    """A stand-in for time.perf_counter as splatpipe.depth sees it: each call gives one second
    more than the call before, so each block the step times into infer, fit or write, which
    reads the clock once when it starts and once when it ends, lasts exactly one second."""

    def __init__(self) -> None:
        self.seconds = 0.0

    def perf_counter(self) -> float:
        """One second more than the last call gave."""
        self.seconds += 1.0
        return self.seconds


def test_running_out_of_gpu_memory_on_the_third_frame_keeps_the_first_twos_counts_and_timings(
        new_job: Callable[..., str], fake_depth_gpu: Callable[..., Any],
        monkeypatch: pytest.MonkeyPatch) -> None:
    job_id = plane_job(new_job)
    gpu = fake_depth_gpu(peak_bytes=PEAK_BYTES, out_of_memory_at=3)
    monkeypatch.setattr("splatpipe.depth.time", Clock())
    report = run_failed(job_id)
    # The message names the third Frame in images.bin's order (Req 3.13).
    assert report["error"]["readable"] == frame_out_of_memory_message(FRAMES[2])
    # The counts of the first two Frames, both fitted, and the peak memory (Req 9.3).
    assert report["counts"] == {
        "images": len(FRAMES), "prior_width": PRIOR_WIDTH, "prior_height": PRIOR_HEIGHT,
        "fitted_images": 2, "few_points_images": 0, "flat_images": 0, "poor_fit_images": 0,
        "peak_gpu_memory_bytes": PEAK_BYTES}
    # The sums of the first two Frames' blocks, one second each: the third's infer never ended,
    # and the load is timed by the Step_Runner's own clock (Req 9.2, 9.3).
    timings = report["timings"]
    assert sorted(timings) == sorted(TIMINGS)
    assert {name: timings[name] for name in TIMINGS[1:]} == {"infer": 2.0, "fit": 2.0,
                                                            "write": 2.0}
    # The first two Prior_Files and no index, so the Train_Step trains without priors (Req 7.8).
    assert entries(priors_of(job_id)) == sorted(prior_files(FRAMES[:2]))
    assert call_names(gpu)[-2:] == ["release", "peak_memory_bytes"]


# The release (Req 3.14): once the load has started, whatever happens next

RELEASES = [
    pytest.param({}, "done", len(FRAMES), id="a done run"),
    pytest.param({"no_geometry": {frame: ALL_PIXELS for frame in FRAMES}}, "failed",
                 len(FRAMES), id="a run in which no frame fits"),
    pytest.param({"out_of_memory_at": 3}, "failed", 3, id="out of memory on a frame"),
    pytest.param({"load_model_error": WrongCheckpoint(CHECKPOINT_REASON)}, "failed", 0,
                 id="a failed load"),
]


@pytest.mark.parametrize(("faults", "outcome", "infers"), RELEASES)
def test_release_is_called_once_after_the_load_starts_with_the_handle_it_gave(
        new_job: Callable[..., str], fake_depth_gpu: Callable[..., Any],
        faults: dict[str, Any], outcome: str, infers: int) -> None:
    job_id = plane_job(new_job)
    gpu = fake_depth_gpu(**faults)
    assert depth.run(job_id)["outcome"] == outcome
    # Once, after load_model and every infer, and nothing after it but the peak memory. Its
    # argument is the handle the load gave, or None when the load failed.
    names = call_names(gpu)
    assert names == ["probe", "import_moge", "load_model", *["infer"] * infers, "release",
                     "peak_memory_bytes"]
    assert [args for name, args in gpu.calls if name == "release"] == [(gpu.handle,)]
    assert (gpu.handle is None) == ("load_model_error" in faults)


# The CLI (Req 1.1, 1.2)

def test_the_cli_prints_depth_done_and_gives_0_when_the_step_is_done(
        new_job: Callable[..., str], fake_depth_gpu: Callable[..., Any],
        capsys: pytest.CaptureFixture[str]) -> None:
    job_id = plane_job(new_job)
    fake_depth_gpu()
    assert depth_main(["--job", job_id]) == 0
    captured = capsys.readouterr()
    assert (captured.out, captured.err) == ("depth: done\n", "")


def test_the_cli_prints_the_readable_error_on_stderr_and_gives_1_when_the_step_fails(
        new_job: Callable[..., str], fake_depth_gpu: Callable[..., Any],
        capsys: pytest.CaptureFixture[str]) -> None:
    job_id = new_job()  # with no Camera_Solve_Output
    fake_depth_gpu()
    assert depth_main(["--job", job_id]) == 1
    captured = capsys.readouterr()
    report = read_json(folder_of(job_id) / "out" / "reports" / "depth.json")
    assert (captured.out, captured.err) == ("", report["error"]["readable"] + "\n")
    assert "Traceback" not in captured.err


# The Depth_Step in a Job, before the Train_Step (Req 1.5, 1.6)

# The two Steps run_job runs: the Depth_Step, optional, then the Train_Step, required.
DEPTH_AND_TRAIN = (StepEntry("depth", "splatpipe.depth", False),
                   StepEntry("train", "splatpipe.train", True))


def job_log(job_id: str) -> list[str]:
    """The lines of the Job_Log, out/job.log, each without the UTC time it starts with."""
    text = (folder_of(job_id) / "out" / "job.log").read_text(encoding="utf-8")
    return [line.split(" ", 1)[1] for line in text.splitlines()]


def test_a_job_with_a_done_depth_step_then_the_train_step_is_done(
        new_job: Callable[..., str], fake_depth_gpu: Callable[..., Any],
        fake_gpu: Callable[..., Any]) -> None:
    job_id = plane_job(new_job)
    fake_depth_gpu()
    fake_gpu()
    summary = run_job(job_id, steps=DEPTH_AND_TRAIN)
    assert summary["status"] == "done", summary["readable_error"]
    assert [(step["step"], step["outcome"]) for step in summary["steps"]] == [
        ("depth", "done"), ("train", "done")]
    # The depth step left its priors, and the train step its splat.
    assert (priors_of(job_id) / "depth.json").is_file()
    assert (folder_of(job_id) / "out" / "train.ply").is_file()


def test_a_job_whose_depth_step_fails_goes_on_trains_and_is_done_with_warnings(
        new_job: Callable[..., str], fake_depth_gpu: Callable[..., Any],
        fake_gpu: Callable[..., Any]) -> None:
    job_id = plane_job(new_job)
    fake_depth_gpu(False)  # the depth stand-in refuses: torch sees no CUDA
    fake_gpu()
    summary = run_job(job_id, steps=DEPTH_AND_TRAIN)
    # The Depth_Step's entry is failed with its message, the Train_Step trained, and the Job is
    # done with warnings, with no failed Required_Step (Req 1.6, D3).
    assert summary["status"] == "done_with_warnings"
    assert summary["steps"][0] == {"step": "depth", "required": False, "outcome": "failed",
                                   "reused": False, "readable_error": NO_CUDA_MESSAGE}
    assert summary["steps"][1]["outcome"] == "done"
    assert (summary["failed_step"], summary["readable_error"]) == (None, None)
    # The Job_Log holds the Depth_Step's message, and the Job went on (Req 1.5).
    assert f"WARNING depth: failed, optional so the job goes on: {NO_CUDA_MESSAGE}" in job_log(
        job_id)
    assert not (priors_of(job_id) / "depth.json").exists()
    assert (folder_of(job_id) / "out" / "train.ply").is_file()
