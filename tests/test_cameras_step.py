"""Tests for the Cameras_Step in splatpipe/cameras/, run through splatpipe.cameras.run or through
its CLI's main, so the Step_Runner runs it and writes its real report, and in the last test
through run_job, the Job_Runner, after the Frames_Step.

Each test but the last makes its Job with cameras_job from tests/conftest.py, in the temporary
Jobs_Folder: 8 Picks of 16 × 12 pixels, with the Mask, mask.png, when the test asks for it. The
Job's Preset is fast. A FakeColmap from fake_colmap stands in for every COLMAP_Command, and the
step finds it through its TOOLS, so no test needs COLMAP (Req 16.5). The test config's [cameras]
is the real config's with min_placed_frames at 1, so 8 Picks need max(1, 0.5 × 8) = 4
Placed_Frames.

By default the Calibration works and each mapper writes one Model that places all 8 Picks, with
a RADIAL camera at the Pick_Size, with a little distortion, and the fake's 4 points. The fake
image_undistorter turns that camera into a PINHOLE camera of 14 × 10 with fx = fy = its f (F18),
and the fake Mask_Warp gives a three-channel PNG of that size with a grey edge around the Mask's
black corner. The mappers' cameras walk sideways, level, each looking at one point, and COLMAP's
frame holds them tilted. So they agree on which way is up and the Levelling moves the scene, and
their views spread over less than 35°, so the eigenvalue test fails and the Scene_Centre is the
Point_Median. Three changes to that:

- TWO_MODELS: the global mapper's Model 0 places Picks 0 and 1, and Model 1, the Kept_Model,
  Picks 2 to 7. So the Final_Model is Model 1, and its 6 Placed_Frames, PLACED, aren't all the
  Picks.
- upside_down: the mappers turn every other camera upside down, so the cameras' mean up vector is
  about 0 long and the Levelling keeps the COLMAP_Frame (Req 9.3).
- edit_models puts a runner around the FakeColmap that rewrites each Model a mapper writes, with
  the step's own Model_Writer. The test of the two records uses it to list the images from the
  last Pick to the first, as COLMAP may register them, and to give some or all of the points
  COLMAP's error of -1, which means none.

The failed runs each make one failure: sift_extraction_max_num_features at 0, the design's own
example of a refused setting, once with a frames.json cut short too; a SPLATPIPE_CAP_DEADLINE
that read_cap_deadline refuses, set with monkeypatch.setenv, which the autouse no_cap_deadline
otherwise keeps unset; a missing Grey_Copy; a
FakeColmap built with installed=False; mappers whose Models place 3 of the 8 Picks, under the 4
needed; each of the fake image_undistorter's three flags; a non-zero exit of either undistorter
in the fake's script; and a full disk, an OSError with ENOSPC, while a Training_Image is copied
or out/cameras.json is written. The full disk comes from shutil.copyfile or write_json, patched
as splatpipe/cameras/outputs.py sees them, and fails only for that file. run_failed checks that
each run failed with an ExpectedFailure, so the report's error.expected is true, and that its
Readable_Error holds no traceback (Req 14.10, 16.9). A test that needs an earlier run's
out/cameras.json writes one first, and one test puts a runner around the FakeColmap, as
edit_models does, that looks for that file as each COLMAP_Command starts.

The last test runs the real Step_List through run_job, so the Frames_Step makes work/frames/
itself. Its Job comes from new_job, as in tests/test_frames_step.py, with a video of a few bytes
that nothing reads, because a FakeTools from fake_tools stands in for ffprobe, ffmpeg's filter
check and the ffmpeg decode. FakeTools gives the video's 20 frames, 0.5 s apart from 0 s: frame k
is a checkerboard of 25 + 10k ± 20, with the 2 rows of 3 pixels at the bottom right at 0 in every
frame, a corner stuck to the camera. set_frames sets score_width to 8 and long_edge to 16, so the
Picks are 16 × 12, like a cameras_job Job's, and bigger than the 3 × 3 pixels that the fake
image_undistorter needs. Every frame moves, and the test config asks for a Pick every 2 s of
the video's 10 s, so the Frames_Step makes 5 Picks, and the corner's 6 stuck pixels, 0.125 of the
frame, make it use the Mask. 5 Picks need 3 Placed_Frames, at least half of them, and the
FakeColmap's global mapper places all 5 in its one Model, where its default of 8 would be more
Picks than there are. The Step_List goes on to the Train_Step and the Clean_Step, so the test
also gives each the stand-in its own tests use: fake_gpu's FakeGpu for the Train_Step's
GPU_Module, and for the Clean_Step's export the real export_sog with
tests/fake_splat_transform.py, set to ok, in place of node and splat-transform. So the whole Job
runs on a CPU with no COLMAP, GPU or Node.

The tests check:

- The Work_Folder: exactly colmap/, images/, sparse/ and solve.json, plus masks/ with the Mask
  (Req 11.1, 11.4). COLMAP's own files stay in colmap/, laid out as the design says, the
  Undistortion's images/ and sparse/ included, with a Command_Log for each run (Req 11.5). images/
  holds the Undistortion's image for each name in sparse/images.bin (Req 11.2), and masks/ one
  8-bit grey PNG per Placed_Frame at the Undistorted_Camera's size, which is 255 exactly where the
  Mask_Warp's output is 255 in every channel and 0 elsewhere (Req 8.4, 11.3).
- sparse/: exactly the three Model_Files, with one PINHOLE camera with fx = fy that every image
  uses (F18, D19), each Placed_Frame with the Undistorted_Model's image id, camera id, name and
  2D points, and every point with its id, colour, error and track. Every Pose and point is the
  Undistorted_Model's moved by the one Level_Move that solve.json gives (Req 9.13, 10.5 to 10.8).
- Cameras that don't agree on up: every Pose and point of sparse/ is the Undistorted_Model's, bit
  for bit, with the Req 9.3 warning in the report and in solve.json, levelled at 0, and the
  identity as the Level_Move (Req 9.14, 13.5, C10).
- The Mask_Warp's camera line is the Final_Model's RADIAL camera as the Final_Model holds it
  before the Undistortion, written here with this file's own code, and its command line has the
  Undistortion's value of each undistortion setting (Req 8.3).
- solve.json and out/cameras.json hold only JSON_Values and every field of the design's tables,
  in order and of the kinds the design gives, with "RADIAL" as the camera_model.
  mean_reprojection_error is the mean of the errors that aren't -1, or null when no point has
  one (F15). out/cameras.json lists the Placed_Frames in Pick_Number order with their Poses from
  sparse/images.bin, and holds no points (Req 11.7, 11.9).
- The report holds every count of Req 13.1 as an int, a timing for each COLMAP_Command that ran
  and for each stage, the warnings that apply, and every key of [cameras] with the Preset's
  near_overlap in its settings, for the global mapper, the Fallback with the Mask, and the fisheye
  camera (Req 13.1, 13.2, 13.4 to 13.6).
- Outside work/cameras/, a done run adds only out/cameras.json and the report, and every other
  file, work/frames/ included, keeps its hash (Req 1.7, 11.8, 11.11). A second run on the same Job
  leaves nothing of the first run (Req 11.13).
- An earlier out/cameras.json is gone before the first COLMAP_Command runs, and also after a run
  that fails on a setting (Req 11.10).
- A bad setting fails the step with the design's message before the step reads frames.json, also
  when frames.json is cut short, and before any COLMAP_Command runs (Req 12.8). A broken
  Frames_Contract and a missing colmap each fail before any COLMAP_Command runs, the missing
  colmap with the design's message (Req 1.1, 2.7).
- An empty SPLATPIPE_CAP_DEADLINE, "soon" and a time with no UTC offset each fail the step with
  read_cap_deadline's own message before the step reads frames.json, also when frames.json is cut
  short, and before any COLMAP_Command runs, and a bad setting still fails first (Req 17.1, 17.2,
  C12). With a good one and a stand-in clock that stays 42.5 s before it, under the test config's
  60 s, every COLMAP_Command, both undistorters included, gets 42.5 s as its Time_Limit and the
  Cap_Deadline as its capped_at, and the step is done. With none, each gets its Time_Limit key's
  value (Req 17.3, 17.4, C16).
- A Placed_Failure, after which no undistorter runs, each fault of the Undistortion, and a full
  disk while a Training_Image is copied each leave no solve.json and no out/cameras.json, where an
  earlier run's out/cameras.json was (Req 7.2, 8.2, 11.6, 11.10, 16.11). A full disk while
  out/cameras.json is written removes solve.json, which was there, and the partly written
  out/cameras.json (C6). Each full disk's Readable_Error names the file and the reason
  (Req 11.12). An Undistortion that kept the RADIAL camera says that COLMAP keeps a camera's own
  model when its distortion came out as 0 and to run the step again, and names neither
  ba_refine_extra_params key (C8).
- A non-zero exit of the Undistortion or of the Mask_Warp fails with the design's Req 2.9 message,
  and the report holds the timing of each command that ran, the failing one's included, and of the
  stages that finished, with the counts worked out before it (Req 2.9, 13.3).
- Each Readable_Error names what its requirement says, compared whole where the design words it
  (Req 16.9).
- splatpipe.cameras.__main__.main prints cameras: done on stdout and gives 0 for a done run, and
  gives 1 for a failed one, with its Readable_Error on stderr and no traceback (Req 14.2).
- run_job with the real Step_List runs the Frames_Step, the Cameras_Step, the Train_Step and the
  Clean_Step in that order, and the Job is done once all four are. work/frames/ then holds the
  frames spec's layout, mask.png included,
  and work/cameras/ the design's, with masks/, a Training_Image and a Training_Mask for each of
  the 5 Placed_Frames and the three Model_Files, and out/cameras.json lists the 5 Placed_Frames
  (Req 14.4).
"""

import errno
import hashlib
import json
import math
import shutil
import tomllib
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from functools import partial
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

# The stand-ins for COLMAP and for the Frames_Step's tools, the options of options.md as the
# oracle sorts them, and the fake mappers' points.
from conftest import (WALK_POINTS, ColmapOption, FakeColmap, FakeDepthGpu, FakeTools,
                      config_text)
# The splat-transform stand-in, which the run_job test gives the Clean_Step's export.
from test_clean_export import find_stand_in, use_stand_in
from splatpipe import cameras, clean
from splatpipe.cameras import Tools
from splatpipe.cameras.__main__ import main as cameras_main
from splatpipe.cameras.colmap import ColmapRun, RunResult
from splatpipe.cameras.model import (CAMERA_MODELS, FILES, NO_ERROR, Image, Model, Points,
                                     read_images, read_model, write_model)
from splatpipe.clean.export import export_sog
from splatpipe.config import PRESETS, REPO_ROOT, load_config
from splatpipe.errors import ExpectedFailure
from splatpipe.frames.ffmpeg import VideoInfo
from splatpipe.jobs import open_job
from splatpipe.jsonio import check_json_value, read_json, write_json
from splatpipe.run import run_job
from splatpipe.train.colmap import read_model as read_sparse_model
from splatpipe.train.deadline import CAP_DEADLINE_ENV, read_cap_deadline

# The runs of a done run, by their names, which are also the names of their timings and of their
# Command_Logs.
EXTRACT, NEAR, FAR = "feature_extractor", "sequential_matcher_near", "sequential_matcher_far"
CALIBRATE, GLOBAL, INCREMENTAL = "view_graph_calibrator", "global_mapper", "mapper"
UNDISTORT, WARP = "image_undistorter", "image_undistorter_standalone"
# The runs for RADIAL when the Global_Run passes, in order, without the Mask_Warp.
RADIAL_RUNS = (EXTRACT, NEAR, FAR, CALIBRATE, GLOBAL, UNDISTORT)
# The stage timings of a done run (Req 13.2).
STAGES = ("matching", "mapping", "undistortion", "levelling", "writing")
# The COLMAP commands that write Models.
MAPPERS = ("global_mapper", "mapper")

PICKS = 8  # a cameras_job Job's Picks
# The fake global mapper's two Models: Model 0 places Picks 0 and 1, and Model 1, the Kept_Model,
# Picks 2 to 7, whose files are PLACED, in Pick_Number order.
TWO_MODELS = {"global_models": (2, 6)}
PLACED = tuple(f"{number:06d}.jpg" for number in range(2, PICKS))
# The Models of TWO_MODELS, or of the Fallback's same two Models, as solve.json lists them.
TWO_MODEL_COUNTS = [{"number": 0, "placed_frames": 2}, {"number": 1, "placed_frames": 6}]

# The fields of solve.json, in the order of the design's table, and those of them that the
# Levelling gives.
SOLVE_FIELDS = ("camera_model", "calibration_ran", "calibration_failure", "mapper",
                "fallback_reason", "models", "final_model", "mean_reprojection_error", "levelled",
                "level_reason", "mean_up_length", "scene_centre", "heading_camera", "rotation",
                "translation")
LEVEL_FIELDS = SOLVE_FIELDS[SOLVE_FIELDS.index("levelled"):]
# The Level_Move when the Levelling keeps the COLMAP_Frame: the identity and zeros.
IDENTITY = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
ZERO = [0.0, 0.0, 0.0]
# How close the Level_Move's arithmetic must come, on coordinates of a few units.
TOLERANCE = 1e-9

# The real config's [cameras], read by path, because tests/conftest.py points CONFIG_PATH at the
# test config, whose [cameras] has the same keys (C5). Its Shared_Settings are every key but the
# Preset tables.
REAL_CAMERAS = tomllib.loads((REPO_ROOT / "config.toml").read_text(encoding="utf-8"))["cameras"]
SHARED_KEYS = tuple(key for key in REAL_CAMERAS if key not in PRESETS)

# The Req 5.3 warning for the fisheye camera, which starts from the config's fisheye_fov_degrees,
# the Req 6.7 warning for a Global_Run that wrote no Model, and the reason of a rejected
# Calibration, as the design words them.
FISHEYE_WARNING = ("The camera model is RADIAL_FISHEYE, so the focal length calibration was "
                   "skipped, because COLMAP's view_graph_calibrator only fits pinhole cameras. "
                   "The fisheye camera starts from fisheye_fov_degrees = "
                   f"{REAL_CAMERAS['fisheye_fov_degrees']} in [cameras].")
NO_MODEL_WARNING = "The global mapper wrote no model, so the incremental mapper ran instead."
LOOK_UP_WARNING = ("The video never looked up, so the ceiling and the top of the walls will be "
                   "missing. Film a second slow loop with the phone tilted up about 30 degrees.")
REJECTED = "rejected its focal length estimate and kept the starting one"


def disagreement_warning(length: float, up_agreement: float) -> str:
    """The Req 9.3 warning, as the design words it, for a Mean_Up `length` long."""
    return (f"The cameras don't agree on which way is up: their mean up vector is {length} long, "
            f"below up_agreement = {up_agreement} in [cameras]. So the scene was left in "
            "COLMAP's frame, which may be tilted.")


# Running the step

@dataclass(frozen=True)
class Done:
    """A done run of the cameras step: its report, and the Job_Folder it ran in."""

    report: dict[str, Any]
    folder: Path  # the Job_Folder, resolved, as open_job gives it to the Step_Runner

    @property
    def work(self) -> Path:
        """work/cameras/, the step's Work_Folder."""
        return self.folder / "work" / "cameras"

    @property
    def colmap(self) -> Path:
        """work/cameras/colmap/, the COLMAP_Folder."""
        return self.work / "colmap"

    def levelled(self) -> Model:
        """work/cameras/sparse/, the Levelled_Model."""
        return read_model(self.work / "sparse")

    def undistorted(self) -> Model:
        """The Undistorted_Model, which the fake image_undistorter wrote to colmap/undistorted/."""
        return read_model(self.colmap / "undistorted" / "sparse")

    def solve_record(self) -> Any:
        """work/cameras/solve.json, the Solve_Record, read back."""
        return read_json(self.work / "solve.json")

    def viewer(self) -> Any:
        """out/cameras.json, the Viewer_Cameras, read back."""
        return read_json(self.folder / "out" / "cameras.json")


def run_done(job_id: str) -> Done:
    """Run the cameras step on the Job through splatpipe.cameras.run, check that it's done, and
    give back the run."""
    report = cameras.run(job_id)
    assert report["outcome"] == "done", report["error"]
    return Done(report=report, folder=open_job(job_id, load_config()).folder)


def edit_models(monkeypatch: pytest.MonkeyPatch, fake: FakeColmap,
                edit: Callable[[Model], Model]) -> None:
    """Put a runner around `fake` in place of the step's tools. It runs each COLMAP_Command with
    `fake`, then rewrites each Model that a mapper wrote as edit(model), with the step's
    write_model, so the step reads the edited Models. find_colmap stays `fake`'s."""
    def run(colmap_run: ColmapRun) -> RunResult:
        result = fake.run(colmap_run)
        if colmap_run.command in MAPPERS and result.exit_code == 0:
            for folder in Path(colmap_run.value("--output_path")).iterdir():
                write_model(folder, edit(read_model(folder)))
        return result

    monkeypatch.setattr("splatpipe.cameras.TOOLS", Tools(find_colmap=fake.find_colmap, run=run))


def reversed_with_errors(errors: Sequence[float]) -> Callable[[Model], Model]:
    """An edit for edit_models: the Model with its images from the last Pick to the first, as
    COLMAP may register them, and with `errors` as its points' errors, one per point."""
    def edit(model: Model) -> Model:
        assert len(errors) == len(model.points.ids), "one error for each point of the Model"
        points = replace(model.points, errors=np.array(errors, dtype="<f8"))
        return replace(model, images=model.images[::-1], points=points)

    return edit


def ran(fake: FakeColmap, name: str) -> ColmapRun:
    """The one run named `name` that `fake` stood in for."""
    runs = [run for run in fake.runs if run.name == name]
    assert len(runs) == 1, f"{name} ran {len(runs)} times"
    return runs[0]


# The files

def names_in(folder: Path) -> list[str]:
    """The names of the files and folders right inside `folder`, sorted."""
    return sorted(path.name for path in folder.iterdir())


def pick_names(numbers: Iterable[int]) -> list[str]:
    """The files of the Picks with these Pick_Numbers, as the Frames_Contract names them."""
    return [f"{number:06d}.jpg" for number in numbers]


def mask_names(names: Iterable[str]) -> list[str]:
    """The Training_Masks' names for these Placed_Frames' names, like 000002.png for
    000002.jpg."""
    return [f"{Path(name).stem}.png" for name in names]


def work_layout(*, mask: bool) -> list[str]:
    """What names_in gives for the Work_Folder of a done run: colmap/, images/, sparse/ and
    solve.json, plus masks/ with the Mask (Req 11.1, 11.4)."""
    return sorted(["colmap", "images", "sparse", "solve.json", *(["masks"] if mask else [])])


def colmap_layout(*, mask: bool, fallback: bool = False) -> list[str]:
    """What names_in gives for the COLMAP_Folder of a done run for RADIAL, as the design's "The
    Work_Folder" lays it out: incremental/ only when the Fallback ran, and the Mask_Warp's camera
    line and output only with the Mask (O8)."""
    return sorted(["database.db", "database_calibrated.db", "global", "logs", "tmp", "undistorted",
                   *(["incremental"] if fallback else []),
                   *(["mask_camera.txt", "mask_undistorted"] if mask else [])])


def read_image(path: Path) -> np.ndarray:
    """The image in the file at `path`, as cv2.imdecode reads it with its channels and bit depth
    unchanged."""
    image = cv2.imdecode(np.frombuffer(path.read_bytes(), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    assert image is not None, f"cv2.imdecode couldn't read {path}"
    return image


def snapshot(folder: Path, *, skip: Path) -> dict[str, str | None]:
    """Every file and folder inside `folder` but `skip` and what it holds, by its path from
    `folder`, like work/frames/mask.png: a file's SHA-256, and None for a folder."""
    found: dict[str, str | None] = {}
    for path in sorted(folder.rglob("*")):
        if path == skip or skip in path.parents:
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        found[path.relative_to(folder).as_posix()] = digest
    return found


# The models, worked out with this file's own code

def image_fields(model: Model) -> list[tuple[int, int, str, bytes]]:
    """Each image's id, camera id, name and 2D points, the points as their bytes, in the order of
    images.bin."""
    return [(image.image_id, image.camera_id, image.name, image.points2d.tobytes())
            for image in model.images]


def point_fields(points: Points) -> tuple[bytes, bytes, bytes, list[bytes]]:
    """The points' ids, colours, errors and tracks, each as their bytes."""
    return (points.ids.tobytes(), points.rgb.tobytes(), points.errors.tobytes(),
            [track.tobytes() for track in points.tracks])


def pose_bytes(model: Model) -> list[tuple[str, bytes, bytes]]:
    """Each image's name, and its Pose's quaternion and translation as the bytes of their
    doubles, so equal values are equal bit for bit."""
    return [(image.name, np.array(image.qvec, dtype="<f8").tobytes(),
             np.array(image.tvec, dtype="<f8").tobytes()) for image in model.images]


def rotation_of(image: Image) -> np.ndarray:
    """The rotation R of the image's Pose, from its unit quaternion (w, x, y, z) in Hamilton's
    convention."""
    w, x, y, z = image.qvec
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def centre_of(image: Image) -> np.ndarray:
    """The centre of the image's camera, −Rᵀ t."""
    return -rotation_of(image).T @ np.array(image.tvec)


def mean_up_length(model: Model) -> float:
    """The length of the Mean_Up of the model's cameras: the mean of each one's up, Rᵀ(0, −1, 0),
    which is minus the second row of R."""
    return float(np.linalg.norm(np.mean([-rotation_of(image)[1] for image in model.images],
                                        axis=0)))


def close(found: np.ndarray, expected: np.ndarray) -> bool:
    """Whether every number of `found` is within TOLERANCE of the same number of `expected`."""
    return bool(np.allclose(found, expected, rtol=0, atol=TOLERANCE))


# The Work_Folder (Req 8.4, 11.1 to 11.5)

@pytest.mark.parametrize("mask_used", [False, True], ids=["no mask", "the mask"])
def test_the_work_folder_holds_the_designs_layout_and_an_undistorted_image_per_placed_frame(
        cameras_job: Callable[..., str], fake_colmap: type[FakeColmap], mask_used: bool) -> None:
    job_id = cameras_job(mask_used=mask_used)
    fake = fake_colmap(**TWO_MODELS)
    done = run_done(job_id)
    # Exactly colmap/, images/, sparse/ and solve.json, plus masks/ with the Mask (Req 11.1,
    # 11.4).
    assert names_in(done.work) == work_layout(mask=mask_used)
    # COLMAP's own files stay in colmap/, laid out as the design says, the Undistortion's images/
    # and sparse/ included (Req 11.5, O8).
    assert names_in(done.colmap) == colmap_layout(mask=mask_used)
    assert {"images", "sparse"} <= set(names_in(done.colmap / "undistorted"))
    # images/ holds exactly the names in sparse/images.bin, the Final_Model's 6 Placed_Frames,
    # each a copy of its image from the Undistortion (Req 11.2).
    names = [image.name for image in read_images(done.work / "sparse" / "images.bin")]
    assert sorted(names) == list(PLACED)
    assert names_in(done.work / "images") == sorted(names)
    for name in names:
        undistorted = done.colmap / "undistorted" / "images" / name
        assert (done.work / "images" / name).read_bytes() == undistorted.read_bytes()
    # The runs, in order, with the Mask_Warp only with the Mask, and in colmap/logs/ a
    # Command_Log for each, named after it.
    runs = [*RADIAL_RUNS, *([WARP] if mask_used else [])]
    assert [run.name for run in fake.runs] == runs
    logs = done.colmap / "logs"
    assert names_in(logs) == sorted(f"{name}.log" for name in runs)
    assert [run.log for run in fake.runs] == [logs / f"{name}.log" for name in runs]


def test_with_the_mask_masks_holds_the_undistorted_mask_once_for_each_placed_frame(
        cameras_job: Callable[..., str], fake_colmap: type[FakeColmap]) -> None:
    job_id = cameras_job(mask_used=True)
    fake_colmap(**TWO_MODELS)
    done = run_done(job_id)
    (camera,) = done.levelled().cameras  # the Undistorted_Camera
    # The Mask_Warp's output has three channels, with grey between the black corner and the white
    # pixels, as bilinear sampling gives.
    warped = read_image(done.colmap / "mask_undistorted" / "mask.png")
    assert warped.shape == (camera.height, camera.width, 3)
    assert np.any((warped > 0) & (warped < 255))
    # The Undistorted_Mask, worked out here: 255 where every channel is 255, and 0 everywhere
    # else (Req 8.4).
    expected = np.where((warped == 255).all(axis=2), 255, 0).astype(np.uint8)
    # One copy for each Placed_Frame, named after its stem (Req 11.3).
    masks = done.work / "masks"
    assert names_in(masks) == mask_names(PLACED)
    for name in mask_names(PLACED):
        # An 8-bit grey PNG: its header holds the bit depth at byte 24 and the colour type, 0 for
        # grey, at byte 25. The Undistorted_Camera's size, with only 0 and 255.
        data = (masks / name).read_bytes()
        assert (data[24], data[25]) == (8, 0)
        mask = read_image(masks / name)
        assert mask.shape == (camera.height, camera.width)
        assert sorted(np.unique(mask).tolist()) == [0, 255]
        assert np.array_equal(mask, expected)


# The Levelled_Model (Req 9.13, 10.5 to 10.8)

def test_sparse_holds_the_undistorted_model_moved_by_the_level_move_in_the_three_model_files(
        cameras_job: Callable[..., str], fake_colmap: type[FakeColmap]) -> None:
    job_id = cameras_job()
    fake_colmap(**TWO_MODELS)
    done = run_done(job_id)
    # Exactly the three Model_Files, with no rigs.bin or frames.bin (Req 10.5).
    assert names_in(done.work / "sparse") == sorted(FILES)
    levelled, undistorted = done.levelled(), done.undistorted()
    # One PINHOLE camera, the Undistorted_Camera, which every image uses (Req 10.6). From the
    # Final_Model's RADIAL camera, with its one f, the Undistortion gives fx = fy (F18, D19).
    assert levelled.cameras == undistorted.cameras
    (camera,) = levelled.cameras
    assert CAMERA_MODELS[camera.model_id][0] == "PINHOLE"
    fx, fy, _, _ = camera.params
    assert fx == fy
    assert {image.camera_id for image in levelled.images} == {camera.camera_id}
    # Each Placed_Frame with the Undistorted_Model's image id, camera id, name and 2D points, and
    # every point with its id, colour, error and track (Req 10.7, 10.8).
    assert sorted(image.name for image in levelled.images) == list(PLACED)
    assert image_fields(levelled) == image_fields(undistorted)
    assert point_fields(levelled.points) == point_fields(undistorted.points)
    # Every Pose and every point moved by the one Level_Move that solve.json gives, x' = R x + T,
    # with R a rotation, so scale 1. The fake's tilt makes R more than the identity (Req 9.13,
    # 10.7, 10.8).
    record = done.solve_record()
    assert record["levelled"] is True
    rotation, translation = np.array(record["rotation"]), np.array(record["translation"])
    assert np.allclose(rotation.T @ rotation, np.eye(3), rtol=0, atol=1e-12)
    assert abs(np.linalg.det(rotation) - 1) < 1e-12
    assert not close(rotation, np.eye(3))
    for before, after in zip(undistorted.images, levelled.images, strict=True):
        assert close(rotation_of(after), rotation_of(before) @ rotation.T)
        assert close(centre_of(after), rotation @ centre_of(before) + translation)
    assert close(levelled.points.xyz, undistorted.points.xyz @ rotation.T + translation)


# Keeping the COLMAP_Frame (Req 9.3, 9.14, 13.5, C10)

def test_cameras_that_dont_agree_on_up_keep_colmaps_frame_bit_for_bit_with_the_warning(
        cameras_job: Callable[..., str], fake_colmap: type[FakeColmap]) -> None:
    job_id = cameras_job()
    fake_colmap(upside_down=True)
    done = run_done(job_id)
    levelled, undistorted = done.levelled(), done.undistorted()
    # The Levelled_Model is still written as the three Model_Files (Req 9.14), with every Pose and
    # every point as the Undistorted_Model holds it, bit for bit (C10).
    assert names_in(done.work / "sparse") == sorted(FILES)
    assert pose_bytes(levelled) == pose_bytes(undistorted)
    assert levelled.points.xyz.tobytes() == undistorted.points.xyz.tobytes()
    # The Mean_Up's length in solve.json is the one worked out here from the Undistorted_Model,
    # below the test config's up_agreement.
    record = done.solve_record()
    length = record["mean_up_length"]
    assert abs(length - mean_up_length(undistorted)) < 1e-12
    up_agreement = load_config().step_settings("cameras", "fast")["up_agreement"]
    assert length < up_agreement
    # The report holds the Req 9.3 warning, and levelled is 0 (Req 9.3, 13.1, 13.5).
    warning = disagreement_warning(length, up_agreement)
    assert done.report["warnings"] == [warning, LOOK_UP_WARNING]
    assert done.report["counts"]["levelled"] == 0
    # solve.json gives the reason, no Scene_Centre and no Heading_Camera, and the identity as the
    # Level_Move (Req 11.7). json.dumps compares the kinds too, so 1.0 isn't 1.
    expected = {"levelled": False, "level_reason": warning, "mean_up_length": length,
                "scene_centre": None, "heading_camera": None, "rotation": IDENTITY,
                "translation": ZERO}
    assert json.dumps({key: record[key] for key in LEVEL_FIELDS}) == json.dumps(expected)
    assert done.viewer()["levelled"] is False  # Req 11.9


# The Mask_Warp (Req 8.3)

def test_the_mask_warp_takes_the_final_models_own_camera_and_the_undistortions_settings(
        cameras_job: Callable[..., str], fake_colmap: type[FakeColmap],
        colmap_options: Mapping[str, tuple[ColmapOption, ...]]) -> None:
    job_id = cameras_job(mask_used=True)
    fake = fake_colmap()
    done = run_done(job_id)
    # The Final_Model is the Global_Run's Model 0. Its camera, as it holds it before the
    # Undistortion, is the mappers' RADIAL camera at the Pick_Size, with its five parameters f,
    # cx, cy, k1 and k2, not the Undistorted_Camera.
    (camera,) = read_model(done.colmap / "global" / "0").cameras
    assert (CAMERA_MODELS[camera.model_id][0], camera.width, camera.height) == ("RADIAL", 16, 12)
    assert len(camera.params) == 5
    # The camera line, written here: mask.png, the model, the size, then each parameter as repr
    # writes it, which reads back as the same double (F8).
    line = " ".join(["mask.png", "RADIAL", str(camera.width), str(camera.height),
                     *(repr(param) for param in camera.params)]) + "\n"
    camera_file = done.colmap / "mask_camera.txt"
    assert camera_file.read_text(encoding="utf-8") == line
    warp, undistortion = ran(fake, WARP), ran(fake, UNDISTORT)
    assert Path(warp.value("--input_file")) == camera_file
    # Each undistortion setting, the 11 options that the oracle gives an undistortion_ key, has
    # the Undistortion's value in the Mask_Warp.
    settings = [option.option for option in colmap_options[WARP]
                if option.key is not None and option.key.startswith("undistortion_")]
    assert len(settings) == 11
    for option in settings:
        assert warp.value(option) == undistortion.value(option), option


# solve.json and out/cameras.json (Req 11.7 to 11.9)

# Each run's FakeColmap script, the errors that edit_models gives the 4 points of each Model, and
# what the solve flow's fields of solve.json then hold.
SOLVE_RUNS = [
    pytest.param(
        TWO_MODELS, (0.25, NO_ERROR, 0.75, NO_ERROR),
        {"camera_model": "RADIAL", "calibration_ran": True, "calibration_failure": None,
         "mapper": "global", "fallback_reason": None, "models": TWO_MODEL_COUNTS,
         "final_model": 1, "mean_reprojection_error": 0.5},
        id="the global mapper with two errors of -1"),
    pytest.param(
        {"calibration": "rejected", "incremental_models": (2, 6)}, (NO_ERROR,) * 4,
        {"camera_model": "RADIAL", "calibration_ran": True, "calibration_failure": REJECTED,
         "mapper": "incremental", "fallback_reason": REJECTED, "models": TWO_MODEL_COUNTS,
         "final_model": 1, "mean_reprojection_error": None},
        id="the fallback with every error -1"),
]


@pytest.mark.parametrize(("script", "errors", "solved"), SOLVE_RUNS)
def test_solve_json_and_cameras_json_hold_every_field_of_the_designs_tables(
        cameras_job: Callable[..., str], fake_colmap: type[FakeColmap],
        monkeypatch: pytest.MonkeyPatch, script: dict[str, Any], errors: tuple[float, ...],
        solved: dict[str, Any]) -> None:
    job_id = cameras_job()
    fake = fake_colmap(**script)
    edit_models(monkeypatch, fake, reversed_with_errors(errors))
    done = run_done(job_id)
    record, viewer = done.solve_record(), done.viewer()
    check_json_value(record, "solve")  # Req 11.7
    check_json_value(viewer, "cameras")  # Req 11.9
    # Every field of the design's table, in its order. The mean reprojection error skips the
    # errors of -1, and is null when every error is -1 (F15). json.dumps compares the kinds too,
    # so true isn't 1 and 1 isn't 1.0.
    assert list(record) == list(SOLVE_FIELDS)
    assert json.dumps({key: record[key] for key in solved}) == json.dumps(solved)
    # The cameras agree, so the Levelling moved the scene. The Heading_Camera is the first
    # Placed_Frame by Pick_Number, though images.bin lists it last, and the views spread too
    # little for a well defined Axis_Point, so the Scene_Centre is the Point_Median.
    assert record["levelled"] is True
    assert record["level_reason"] is None
    assert type(record["mean_up_length"]) is float
    assert abs(record["mean_up_length"] - 1) < 1e-9
    assert (record["scene_centre"], record["heading_camera"]) == ("point_median", PLACED[0])
    assert [[type(value) for value in row] for row in record["rotation"]] == [[float] * 3] * 3
    assert [type(value) for value in record["translation"]] == [float] * 3
    # out/cameras.json: the Undistorted_Camera, levelled, and each Placed_Frame in Pick_Number
    # order with its Pose from sparse/images.bin, which lists them from the last, and nothing
    # else, so no points (Req 11.9).
    levelled = done.levelled()
    assert [image.name for image in levelled.images] == list(reversed(PLACED))
    (camera,) = levelled.cameras
    fx, fy, cx, cy = camera.params
    poses = {image.name: image for image in levelled.images}
    expected = {
        "camera": {"width": camera.width, "height": camera.height, "fx": fx, "fy": fy,
                   "cx": cx, "cy": cy},
        "levelled": True,
        "frames": [{"name": name, "quaternion_wxyz": list(poses[name].qvec),
                    "translation": list(poses[name].tvec)} for name in PLACED],
    }
    assert json.dumps(viewer) == json.dumps(expected)


# The Cameras_Report (Req 13.1, 13.2, 13.4 to 13.6)

# Each run's settings for set_cameras, whether the Job uses the Mask, its FakeColmap script, the
# runs it makes in order, its fallback_ran, and the warnings that apply.
REPORT_RUNS = [
    pytest.param({}, False, {}, RADIAL_RUNS, 0, [LOOK_UP_WARNING], id="the global mapper"),
    pytest.param({}, True, {"global_models": None},
                 (EXTRACT, NEAR, FAR, CALIBRATE, GLOBAL, INCREMENTAL, UNDISTORT, WARP), 1,
                 [NO_MODEL_WARNING, LOOK_UP_WARNING], id="the fallback with the mask"),
    pytest.param({"camera_model": "RADIAL_FISHEYE"}, False, {},
                 (EXTRACT, NEAR, FAR, GLOBAL, UNDISTORT), 0, [FISHEYE_WARNING, LOOK_UP_WARNING],
                 id="the fisheye camera"),
]


@pytest.mark.parametrize(("settings", "mask_used", "script", "runs", "fallback_ran", "warnings"),
                         REPORT_RUNS)
def test_the_report_holds_every_count_as_an_int_the_timings_of_what_ran_and_the_warnings(
        cameras_job: Callable[..., str], fake_colmap: type[FakeColmap],
        set_cameras: Callable[..., None], settings: dict[str, Any], mask_used: bool,
        script: dict[str, Any], runs: tuple[str, ...], fallback_ran: int,
        warnings: list[str]) -> None:
    set_cameras(**settings)
    job_id = cameras_job(mask_used=mask_used)
    fake = fake_colmap(**script)
    done = run_done(job_id)
    report = done.report
    # The Step_Runner's ordinary report, in out/reports/cameras.json (Req 13.6).
    assert read_json(done.folder / "out" / "reports" / "cameras.json") == report
    assert [run.name for run in fake.runs] == list(runs)
    # Every count of Req 13.1, each an int. The run that made the Final_Model wrote one Model,
    # which places all 8 Picks with the fake's points, and the cameras agree, so the Levelling
    # moved the scene (Req 13.1, 13.4).
    assert report["counts"] == {"picks": PICKS, "placed_frames": PICKS,
                                "points3d": len(WALK_POINTS), "models_found": 1,
                                "fallback_ran": fallback_ran, "kept_from_global": 1 - fallback_ran,
                                "levelled": 1}
    assert all(type(value) is int for value in report["counts"].values())
    # A timing for each COLMAP_Command that ran, named after its run, and for each stage, each a
    # float of seconds (Req 13.2, 13.4).
    assert sorted(report["timings"]) == sorted([*runs, *STAGES])
    assert all(type(seconds) is float and seconds >= 0 for seconds in report["timings"].values())
    assert report["warnings"] == warnings  # Req 13.5
    # Every key of [cameras] and the Preset's near_overlap, as the test config holds them
    # (Req 13.6).
    assert sorted(report["settings"]) == sorted([*SHARED_KEYS, "near_overlap"])
    assert report["settings"] == load_config().step_settings("cameras", "fast")


@pytest.mark.parametrize(("last_pitch", "threshold", "warns"), [
    (0.0, 40.0, True),
    (35.0, 40.0, False),
    (0.0, 5.0, False),
])
def test_upward_coverage_uses_the_highest_view_and_the_configured_threshold(
        cameras_job: Callable[..., str], fake_colmap: type[FakeColmap],
        set_cameras: Callable[..., None], monkeypatch: pytest.MonkeyPatch,
        last_pitch: float, threshold: float, warns: bool) -> None:
    def edit(model: Model) -> Model:
        half_pitch = math.radians(last_pitch) / 2
        images = [replace(image, qvec=(1.0, 0.0, 0.0, 0.0)) for image in model.images]
        images[-1] = replace(images[-1], qvec=(math.cos(half_pitch), -math.sin(half_pitch), 0.0, 0.0))
        return replace(model, images=tuple(images))

    set_cameras(min_highest_ray_degrees=threshold)
    job_id = cameras_job()
    edit_models(monkeypatch, fake_colmap(), edit)
    done = run_done(job_id)
    assert done.viewer()["levelled"] is True
    assert done.report["warnings"] == ([LOOK_UP_WARNING] if warns else [])


# out/, work/frames/ and running the step again (Req 1.7, 11.8, 11.11, 11.13)

def test_a_done_run_adds_only_cameras_json_and_its_report_outside_work_cameras(
        cameras_job: Callable[..., str], fake_colmap: type[FakeColmap]) -> None:
    job_id = cameras_job(mask_used=True)
    folder = open_job(job_id, load_config()).folder
    work = folder / "work" / "cameras"
    before = snapshot(folder, skip=work)
    # The snapshot holds the Frames_Step's files, and out/ is empty.
    assert {"work/frames/frames.json", "work/frames/mask.png", "work/frames/images/000000.jpg",
            "work/frames/grey/000000.jpg"} <= set(before)
    assert [path for path in before if path.startswith("out")] == ["out"]
    fake_colmap()
    run_done(job_id)
    after = snapshot(folder, skip=work)
    # Outside work/cameras/, the run added only out/cameras.json and the report (Req 11.8,
    # 11.11), and every other file, work/frames/ included, keeps its hash (Req 1.7).
    assert sorted(set(after) - set(before)) == ["out/cameras.json", "out/reports",
                                                "out/reports/cameras.json"]
    assert {path: digest for path, digest in after.items() if path in before} == before


def test_a_second_run_on_the_same_job_leaves_only_its_own_files(
        cameras_job: Callable[..., str], fake_colmap: type[FakeColmap]) -> None:
    job_id = cameras_job(mask_used=True)
    # The first run: the Global_Run writes no Model, so the Fallback's Model places all 8 Picks.
    fake_colmap(global_models=None)
    first = run_done(job_id)
    assert names_in(first.colmap) == colmap_layout(mask=True, fallback=True)
    assert names_in(first.work / "images") == pick_names(range(PICKS))
    # The second: the Global_Run's one Model places only the first 6 Picks, which passes.
    fake = fake_colmap(global_models=(6,))
    second = run_done(job_id)
    names = pick_names(range(6))
    # Nothing of the first run is left: no Fallback's Models or Command_Log, and no Training_Image
    # or Training_Mask of Picks 6 and 7 (Req 11.13).
    assert names_in(second.work) == work_layout(mask=True)
    assert names_in(second.colmap) == colmap_layout(mask=True)
    assert names_in(second.colmap / "logs") == sorted(f"{run.name}.log" for run in fake.runs)
    assert names_in(second.colmap / "undistorted" / "images") == names
    assert names_in(second.work / "images") == names
    assert names_in(second.work / "masks") == mask_names(names)
    assert [image.name for image in read_images(second.work / "sparse" / "images.bin")] == names
    assert second.solve_record()["mapper"] == "global"
    assert [frame["name"] for frame in second.viewer()["frames"]] == names


# Failed runs (Req 14.10, 16.9)

NEEDED = 4  # the Placed_Frames that a cameras_job Job's 8 Picks need: max(1, 0.5 × 8)
FULL_DISK = "No space left on device"  # the reason the stand-in full disk gives
# The two records a failed run must not leave, by their paths from the Job_Folder: the
# Solve_Record and the Viewer_Cameras (Req 11.6, 11.10).
RECORDS = ("work/cameras/solve.json", "out/cameras.json")
# What a test writes to out/cameras.json as the Viewer_Cameras of an earlier run.
EARLIER = {"earlier": True}

# The design's own example of a refused setting, and its Readable_Error: the key, its table,
# COLMAP's option and what the value must be (Req 12.8, 12.9).
BAD_SETTING = {"sift_extraction_max_num_features": 0}
BAD_SETTING_ERROR = ("The setting sift_extraction_max_num_features in [cameras] (COLMAP's "
                     "--SiftExtraction.max_num_features) is 0, but it must be a whole number "
                     "above 0. Change it to a whole number above 0.")
# The Req 2.7 Readable_Error, and the Filming_Advice that ends the Placed_Failure, as the design
# words them.
NOT_INSTALLED = ("COLMAP isn't installed: the program colmap isn't on the PATH. Install COLMAP "
                 "4.2.1, built without CUDA, OpenGL, the GUI, CGAL and LSD as "
                 ".kiro/steering/tech.md says, then run the step again.")
ADVICE = ("Film the place again: move slower, hold the phone steadier, overlap more between "
          "views, and film surfaces with more texture. Then make a new job from the new video.")
FRAMES_COMMAND = "uv run python -m splatpipe.frames --job"  # the Frames_Step's CLI


def placed_failure(placed: int, picks: int, needed: int) -> str:
    """The Placed_Failure's Readable_Error, as the design words it (Req 7.3)."""
    return (f"Only {placed} of the {picks} picks could be placed, but the step needs at least "
            f"{needed}. {ADVICE}")


@dataclass(frozen=True)
class Failed:
    """A failed run of the cameras step: its report, and the Job_Folder it ran in."""

    report: dict[str, Any]
    folder: Path  # the Job_Folder, resolved, as open_job gives it to the Step_Runner

    @property
    def readable(self) -> str:
        """The report's Readable_Error."""
        return self.report["error"]["readable"]

    def records_left(self) -> list[str]:
        """Which of RECORDS the run left, by their paths from the Job_Folder."""
        return [path for path in RECORDS if (self.folder / path).exists()]


def folder_of(job_id: str) -> Path:
    """The Job_Folder of `job_id`, resolved, as open_job gives it to the Step_Runner, so the paths
    in the step's messages start with it."""
    return open_job(job_id, load_config()).folder


def run_failed(job_id: str) -> Failed:
    """Run the cameras step on the Job through splatpipe.cameras.run, check that it failed with an
    ExpectedFailure, so the report's error.expected is true, whose Readable_Error holds no
    traceback, and give back the run (Req 14.10, 16.9)."""
    report = cameras.run(job_id)
    assert report["outcome"] == "failed"
    error = report["error"]
    assert error["expected"] is True, error
    assert error["technical"]["type"] == "splatpipe.errors.ExpectedFailure", error
    assert "Traceback" not in error["readable"]
    return Failed(report=report, folder=folder_of(job_id))


def plant_earlier_cameras(job_id: str) -> Path:
    """Write EARLIER to the Job's out/cameras.json, as an earlier run's Viewer_Cameras, and give
    its path."""
    path = folder_of(job_id) / "out" / "cameras.json"
    write_json(path, EARLIER)
    return path


# The earlier out/cameras.json (Req 11.8, 11.10)

def test_an_earlier_cameras_json_is_gone_before_the_first_colmap_command_runs(
        cameras_job: Callable[..., str], fake_colmap: type[FakeColmap],
        monkeypatch: pytest.MonkeyPatch) -> None:
    job_id = cameras_job()
    earlier = plant_earlier_cameras(job_id)
    fake = fake_colmap()
    there: list[bool] = []  # whether out/cameras.json is there as each COLMAP_Command starts

    def run(colmap_run: ColmapRun) -> RunResult:
        there.append(earlier.exists())
        return fake.run(colmap_run)

    monkeypatch.setattr("splatpipe.cameras.TOOLS", Tools(find_colmap=fake.find_colmap, run=run))
    done = run_done(job_id)
    # The step removed it as it started, so it was gone when each COLMAP_Command ran, and the done
    # run wrote its own Viewer_Cameras, last of all (Req 11.8, 11.10).
    assert there == [False] * len(RADIAL_RUNS)
    assert list(done.viewer()) == ["camera", "levelled", "frames"]


def test_an_earlier_cameras_json_is_removed_also_when_the_step_then_fails_on_a_setting(
        cameras_job: Callable[..., str], fake_colmap: type[FakeColmap],
        set_cameras: Callable[..., None]) -> None:
    set_cameras(**BAD_SETTING)
    job_id = cameras_job()
    earlier = plant_earlier_cameras(job_id)
    fake_colmap()
    assert run_failed(job_id).readable == BAD_SETTING_ERROR
    assert not earlier.exists()  # Req 11.10


# Before any COLMAP_Command (Req 1.1, 2.7, 12.8, 16.5)

@pytest.mark.parametrize("cut_short", [False, True],
                         ids=["frames.json as the frames step left it", "frames.json cut short"])
def test_a_bad_setting_fails_the_step_before_it_reads_frames_json_and_before_colmap_runs(
        cameras_job: Callable[..., str], fake_colmap: type[FakeColmap],
        set_cameras: Callable[..., None], cut_short: bool) -> None:
    set_cameras(**BAD_SETTING)
    job_id = cameras_job()
    if cut_short:  # a Frames_Record that the step would refuse if it read it first
        record = folder_of(job_id) / "work" / "frames" / "frames.json"
        record.write_text('{"pick_width": 16, ', encoding="utf-8")
    fake = fake_colmap()
    failed = run_failed(job_id)
    # With frames.json cut short too, it's the setting that fails, so the step hadn't read
    # frames.json (Req 12.8).
    assert failed.readable == BAD_SETTING_ERROR
    # FakeColmap recorded no run, and the step worked out no count, picks included, and no timing.
    assert fake.runs == []
    assert (failed.report["counts"], failed.report["timings"]) == ({}, {})


def test_a_broken_frames_contract_fails_the_step_before_any_colmap_command_runs(
        cameras_job: Callable[..., str], fake_colmap: type[FakeColmap]) -> None:
    job_id = cameras_job()
    grey = folder_of(job_id) / "work" / "frames" / "grey" / "000003.jpg"
    grey.unlink()  # Pick 3's Grey_Copy, which frames.json still lists
    fake = fake_colmap()
    failed = run_failed(job_id)
    # The Readable_Error names the missing file and says to run the frames step again, with the
    # Job's id (Req 1.4).
    assert str(grey) in failed.readable
    assert "missing" in failed.readable.replace(str(grey), "")
    assert f"Run the frames step again with {FRAMES_COMMAND} {job_id}." in failed.readable
    # No COLMAP_Command ran, and the step hadn't got as far as the count picks (Req 1.1).
    assert fake.runs == []
    assert (failed.report["counts"], failed.report["timings"]) == ({}, {})


def test_without_colmap_on_the_path_the_step_fails_before_any_colmap_command_runs(
        cameras_job: Callable[..., str], fake_colmap: type[FakeColmap]) -> None:
    job_id = cameras_job()
    fake = fake_colmap(installed=False)
    failed = run_failed(job_id)
    assert failed.readable == NOT_INSTALLED  # Req 2.7
    # The step had read the Frames_Contract, so picks is in, and then nothing ran.
    assert fake.runs == []
    assert (failed.report["counts"], failed.report["timings"]) == ({"picks": PICKS}, {})


# The Cap_Deadline (Req 17.1 to 17.4, C12, C16)

# The values of SPLATPIPE_CAP_DEADLINE that read_cap_deadline refuses.
REFUSED_DEADLINES = [pytest.param("", id="empty"), pytest.param("soon", id="not a time"),
                     pytest.param("2026-11-21T14:25:30", id="no UTC offset")]
# A good Cap_Deadline, and the text of it that the test sets, in the form the User_Data writes.
CAP_DEADLINE = datetime(2026, 11, 21, 14, 25, 30, tzinfo=timezone.utc)
CAP_DEADLINE_TEXT = "2026-11-21T14:25:30Z"
# The stand-in clock's time, which stays LEFT seconds before CAP_DEADLINE, under the test config's
# 60 s for each Time_Limit key, and enough for every COLMAP_Command of a done run.
LEFT = 42.5
NOW = CAP_DEADLINE - timedelta(seconds=LEFT)


def cap_deadline_refusal(value: str) -> str:
    """read_cap_deadline's own message for SPLATPIPE_CAP_DEADLINE at `value`."""
    with pytest.raises(ExpectedFailure) as refused:
        read_cap_deadline({CAP_DEADLINE_ENV: value})
    return str(refused.value)


@pytest.mark.parametrize("cut_short", [False, True],
                         ids=["frames.json as the frames step left it", "frames.json cut short"])
@pytest.mark.parametrize("value", REFUSED_DEADLINES)
def test_a_refused_cap_deadline_fails_the_step_with_its_own_message_before_frames_json_and_colmap(
        cameras_job: Callable[..., str], fake_colmap: type[FakeColmap],
        monkeypatch: pytest.MonkeyPatch, value: str, cut_short: bool) -> None:
    job_id = cameras_job()
    if cut_short:  # a Frames_Record that the step would refuse if it read it first
        record = folder_of(job_id) / "work" / "frames" / "frames.json"
        record.write_text('{"pick_width": 16, ', encoding="utf-8")
    monkeypatch.setenv(CAP_DEADLINE_ENV, value)
    fake = fake_colmap()
    failed = run_failed(job_id)  # expected: true, with no traceback (Req 14.10, 16.9)
    # read_cap_deadline's own message, also with frames.json cut short, so the step hadn't read
    # frames.json (Req 17.2, C12).
    assert failed.readable == cap_deadline_refusal(value)
    # No COLMAP_Command ran, and the step worked out no count, picks included, and no timing.
    assert fake.runs == []
    assert (failed.report["counts"], failed.report["timings"]) == ({}, {})


def test_a_bad_setting_still_fails_the_step_before_a_refused_cap_deadline(
        cameras_job: Callable[..., str], fake_colmap: type[FakeColmap],
        set_cameras: Callable[..., None], monkeypatch: pytest.MonkeyPatch) -> None:
    set_cameras(**BAD_SETTING)
    job_id = cameras_job()
    monkeypatch.setenv(CAP_DEADLINE_ENV, "soon")
    fake = fake_colmap()
    # The step reads the Cap_Deadline only once the settings checks pass (Req 17.1, C12).
    assert run_failed(job_id).readable == BAD_SETTING_ERROR
    assert fake.runs == []


def test_with_a_cap_deadline_every_colmap_command_gets_the_seconds_left_and_the_step_is_done(
        cameras_job: Callable[..., str], fake_colmap: type[FakeColmap],
        monkeypatch: pytest.MonkeyPatch) -> None:
    job_id = cameras_job(mask_used=True)
    monkeypatch.setenv(CAP_DEADLINE_ENV, CAP_DEADLINE_TEXT)
    fake = fake_colmap(now=lambda: NOW)  # the step's clock, which stays at NOW (C16)
    run_done(job_id)
    keys = load_config().step_settings("cameras", "fast")
    # Every COLMAP_Command ran, both undistorters included. Fewer seconds were left than each one's
    # Time_Limit key gives, so each got the seconds left as its Time_Limit and the Cap_Deadline as
    # its capped_at (Req 17.3, C13).
    assert [run.name for run in fake.runs] == [*RADIAL_RUNS, WARP]
    for run in fake.runs:
        assert keys[run.timeout_key] > LEFT, run.name
        assert (run.timeout_seconds, run.capped_at) == (LEFT, CAP_DEADLINE), run.name


def test_without_a_cap_deadline_every_colmap_command_gets_its_time_limit_keys_value(
        cameras_job: Callable[..., str], fake_colmap: type[FakeColmap]) -> None:
    job_id = cameras_job(mask_used=True)  # no_cap_deadline left SPLATPIPE_CAP_DEADLINE unset
    fake = fake_colmap()
    run_done(job_id)
    keys = load_config().step_settings("cameras", "fast")
    # No cap: each COLMAP_Command got its Time_Limit key's value, with no capped_at (Req 17.4).
    assert [run.name for run in fake.runs] == [*RADIAL_RUNS, WARP]
    for run in fake.runs:
        assert (run.timeout_seconds, run.capped_at) == (keys[run.timeout_key], None), run.name


# Failures that leave no solve.json and no out/cameras.json (Req 7.2, 8.2, 11.6, 11.10, 11.12,
# 16.11, C6)

def test_a_placed_failure_runs_no_undistorter_and_leaves_no_solve_json_or_cameras_json(
        cameras_job: Callable[..., str], fake_colmap: type[FakeColmap]) -> None:
    job_id = cameras_job(mask_used=True)
    plant_earlier_cameras(job_id)
    # The global mapper's one Model places 3 of the 8 Picks, under the 4 needed, so the Fallback
    # runs, and its one Model places 3 too.
    fake = fake_colmap(global_models=(3,), incremental_models=(3,))
    failed = run_failed(job_id)
    assert failed.readable == placed_failure(3, PICKS, NEEDED)  # Req 7.3
    # The Placed_Failure comes before the Undistortion, so neither undistorter ran (Req 7.2).
    assert [run.name for run in fake.runs] == [EXTRACT, NEAR, FAR, CALIBRATE, GLOBAL, INCREMENTAL]
    assert failed.records_left() == []  # Req 11.6, 11.10, 16.11


# Each flag of the fake image_undistorter, and what the Readable_Error says the Undistortion got
# wrong (Req 8.2). A camera that isn't PINHOLE, the RADIAL camera kept as it was, also gets C8's
# words: COLMAP keeps a camera's own model when its distortion came out as 0.
KEPT_MODEL = "COLMAP keeps a camera's own model when the camera's distortion came out as 0."
UNDISTORTION_FAULTS = [
    pytest.param("keep_camera_model", ("left the camera as RADIAL", KEPT_MODEL),
                 id="the camera's own model"),
    pytest.param("second_camera", ("wrote 2 cameras",), id="a second camera"),
    # The Final_Model's last image is Pick 7's.
    pytest.param("leave_out_image", ("left the placed pick 000007.jpg out",),
                 id="an image left out"),
]


@pytest.mark.parametrize(("flag", "wrong"), UNDISTORTION_FAULTS)
def test_each_fault_of_the_undistortion_fails_saying_what_it_got_wrong_and_leaves_no_records(
        cameras_job: Callable[..., str], fake_colmap: type[FakeColmap], flag: str,
        wrong: tuple[str, ...]) -> None:
    job_id = cameras_job()
    plant_earlier_cameras(job_id)
    fake_colmap(**{flag: True})
    failed = run_failed(job_id)
    readable = failed.readable
    # It says that COLMAP's image_undistorter got something wrong, what, and to run the step again
    # (Req 8.2).
    assert "COLMAP's image_undistorter" in readable
    assert all(part in readable for part in wrong)
    assert "run the step again" in readable.lower()
    # It names neither mapper_ba_refine_extra_params nor global_mapper_ba_refine_extra_params,
    # which the settings check keeps true (C8, D20).
    assert "ba_refine_extra_params" not in readable
    assert failed.records_left() == []  # Req 11.6, 11.10, 16.11


def test_a_full_disk_while_a_training_image_is_copied_fails_naming_it_and_leaves_no_records(
        cameras_job: Callable[..., str], fake_colmap: type[FakeColmap],
        monkeypatch: pytest.MonkeyPatch) -> None:
    job_id = cameras_job()
    plant_earlier_cameras(job_id)
    fake_colmap()
    images = folder_of(job_id) / "work" / "cameras" / "images"  # the Training_Images
    copy = shutil.copyfile

    def full_disk_for_training_images(source: Path, target: Path, **options: Any) -> Any:
        """shutil.copyfile, which fails the way a full disk does for each Training_Image."""
        if Path(target).parent == images:
            raise OSError(errno.ENOSPC, FULL_DISK)
        return copy(source, target, **options)

    # As splatpipe/cameras/outputs.py sees it, which copies each Training_Image with it (C4).
    monkeypatch.setattr("splatpipe.cameras.outputs.shutil.copyfile",
                        full_disk_for_training_images)
    failed = run_failed(job_id)
    # It names Pick 0's Training_Image, the first the step copies, and the reason (Req 11.12).
    assert str(images / "000000.jpg") in failed.readable
    assert FULL_DISK in failed.readable
    assert failed.records_left() == []  # Req 11.6, 11.10, 16.11


def test_a_full_disk_while_cameras_json_is_written_removes_solve_json_too_naming_the_file(
        cameras_job: Callable[..., str], fake_colmap: type[FakeColmap],
        monkeypatch: pytest.MonkeyPatch) -> None:
    job_id = cameras_job()
    fake_colmap()
    folder = folder_of(job_id)
    viewer, record = folder / "out" / "cameras.json", folder / "work" / "cameras" / "solve.json"
    record_there: list[bool] = []  # whether solve.json was there as out/cameras.json was written

    def full_disk_for_cameras_json(path: Path, value: object) -> None:
        """write_json, which writes the start of out/cameras.json and then fails the way a full
        disk does."""
        if path != viewer:
            write_json(path, value)
            return
        record_there.append(record.is_file())
        path.write_text('{"camera": ', encoding="utf-8")
        raise OSError(errno.ENOSPC, FULL_DISK)

    # As splatpipe/cameras/outputs.py sees it, which writes both records with it.
    monkeypatch.setattr("splatpipe.cameras.outputs.write_json", full_disk_for_cameras_json)
    failed = run_failed(job_id)
    # solve.json had been written before out/cameras.json, and the failed run left neither:
    # solve.json was removed too, and so was the partly written out/cameras.json (C6).
    assert record_there == [True]
    assert failed.records_left() == []
    # It names the file and the reason (Req 11.12).
    assert str(viewer) in failed.readable
    assert FULL_DISK in failed.readable


# A non-zero exit of either undistorter (Req 2.9, 13.3)

@pytest.mark.parametrize(("name", "line"), [
    pytest.param(UNDISTORT, "Invalid options provided.", id="the undistortion"),
    pytest.param(WARP, "Failed to read the camera of mask.png", id="the mask warp"),
])
def test_a_non_zero_exit_of_either_undistorter_fails_with_its_code_last_line_and_log(
        cameras_job: Callable[..., str], fake_colmap: type[FakeColmap], name: str,
        line: str) -> None:
    job_id = cameras_job(mask_used=True)
    fake = fake_colmap(exits={name: (1, line)})
    failed = run_failed(job_id)
    # The design's Req 2.9 message: the command, its exit code and its Last_Error_Line, to check
    # COLMAP 4.2.1 and run the step again, and its Command_Log. Each undistorter's run is named
    # after its command.
    log = folder_of(job_id) / "work" / "cameras" / "colmap" / "logs" / f"{name}.log"
    assert failed.readable == (
        f"COLMAP's {name} stopped with exit code 1 ({line}). Check that COLMAP 4.2.1 is installed "
        f"as .kiro/steering/tech.md says, then run the step again. Its output is in {log}.")
    # Nothing ran after it, so after the Undistortion's exit not even the Mask_Warp.
    runs = [*RADIAL_RUNS, *([WARP] if name == WARP else [])]
    assert [run.name for run in fake.runs] == runs
    # The report holds the timing of each command that ran, the failing one's included, and of the
    # stages that finished, so not undistortion's, with the counts worked out before (Req 13.3).
    assert sorted(failed.report["timings"]) == sorted([*runs, "matching", "mapping"])
    assert failed.report["counts"] == {"picks": PICKS, "placed_frames": PICKS,
                                       "points3d": len(WALK_POINTS), "models_found": 1,
                                       "fallback_ran": 0, "kept_from_global": 1}


# The CLI (Req 14.2)

def test_the_cli_prints_cameras_done_and_gives_0_when_the_step_is_done(
        cameras_job: Callable[..., str], fake_colmap: type[FakeColmap],
        capsys: pytest.CaptureFixture[str]) -> None:
    job_id = cameras_job()
    fake_colmap()
    assert cameras_main(["--job", job_id]) == 0
    captured = capsys.readouterr()
    assert (captured.out, captured.err) == ("cameras: done\n", "")


def test_the_cli_prints_the_readable_error_on_stderr_and_gives_1_when_the_step_fails(
        cameras_job: Callable[..., str], fake_colmap: type[FakeColmap],
        capsys: pytest.CaptureFixture[str]) -> None:
    job_id = cameras_job()
    fake_colmap(installed=False)
    assert cameras_main(["--job", job_id]) == 1
    captured = capsys.readouterr()
    report = read_json(folder_of(job_id) / "out" / "reports" / "cameras.json")
    assert (report["error"]["expected"], report["error"]["readable"]) == (True, NOT_INSTALLED)
    assert (captured.out, captured.err) == ("", report["error"]["readable"] + "\n")
    assert "Traceback" not in captured.err


# Both steps through the Job_Runner (Req 14.4)

# The Job's video as FakeTools' probe gives it to the Frames_Step: its first video stream that
# isn't cover art, at index 0, 640 × 480, with no colour transfer, so no Tone_Map.
VIDEO = VideoInfo(stream_index=0, width=640, height=480, colour_transfer=None)
# The [frames] settings for set_frames that make the Frames_Step's Score_Frames 8 × 6 and its
# Picks 16 × 12, a cameras_job Job's Pick_Size.
TINY_FRAMES = {"score_width": 8, "long_edge": 16}
SCORE_SHAPE = (6, 8)  # a Score_Frame's rows and columns, the size of each frame FakeTools gives
VIDEO_PICKS = 5  # the Picks the Frames_Step makes of the video's frames


def video_frames() -> list[tuple[float, np.ndarray]]:
    """The video's 20 frames, as FakeTools takes them: 0.5 s apart from 0 s, each 6 rows of 8
    pixels. Frame k is a checkerboard of 25 + 10k ± 20, so every frame moves, with the 2 rows of 3
    pixels at the bottom right at 0 in every frame, a corner stuck to the camera."""
    rows, columns = np.indices(SCORE_SHAPE)
    checkerboard = np.where((rows + columns) % 2 == 0, 20, -20)
    stuck = (rows >= 4) & (columns >= 5)
    return [(k * 0.5, np.where(stuck, 0, 25 + 10 * k + checkerboard).astype(np.uint8))
            for k in range(20)]


def test_run_job_runs_the_frames_step_then_the_cameras_step_and_the_job_is_done(
        new_job: Callable[..., str], tmp_path: Path, temp_config: Path,
        fake_tools: type[FakeTools], fake_colmap: type[FakeColmap],
        fake_depth_gpu: type[FakeDepthGpu], fake_gpu: Callable[..., Any],
        monkeypatch: pytest.MonkeyPatch) -> None:
    job_id = new_job()  # with no work/frames/, which the Frames_Step makes
    # The tiny [frames] settings, a stand-in Weights_File at [depth]'s moge_weights, outside the
    # Jobs_Folder, which the Depth_Step wants to be a file before the stand-in loads it, and
    # fit_min_points at 3, because each Frame sees only the 4 WALK_POINTS. set_frames and
    # set_depth each rewrite the whole test config, so it's written once here.
    weights = tmp_path / "weights" / "model.pt"
    weights.parent.mkdir()
    weights.write_bytes(b"stand-in MoGe-2 weights")
    depth = {"moge_weights": weights.as_posix(), "fit_min_points": 3}
    temp_config.write_text(config_text(tmp_path / "jobs", frames=TINY_FRAMES, depth=depth),
                           encoding="utf-8")
    fake_tools(VIDEO, video_frames())
    # The global mapper's one Model places every Pick, so the Placed_Check passes.
    fake_colmap(global_models=(VIDEO_PICKS,))
    # The Depth_Step's stand-in, with MoGe's output made from the model this Job's Cameras_Step
    # writes in place of the plane scene's, so every Frame fits (depth-priors spec, Req 1.8).
    gpu = fake_depth_gpu()
    sparse = folder_of(job_id) / "work" / "cameras" / "sparse"
    monkeypatch.setattr(gpu, "_plane_model", lambda: read_sparse_model(sparse))
    # The Train_Step's and the Clean_Step's stand-ins, as their own tests install them.
    fake_gpu()
    use_stand_in(monkeypatch, "ok")
    monkeypatch.setattr(clean, "export_sog", partial(export_sog, find=find_stand_in))
    # The Job_Runner with the real Step_List: the Frames_Step, the Cameras_Step, the Depth_Step,
    # the Train_Step and the Clean_Step. The Job is done once all five are (Req 14.4).
    summary = run_job(job_id)
    assert summary["status"] == "done", summary["readable_error"]
    assert [(step["step"], step["outcome"]) for step in summary["steps"]] == [
        ("frames", "done"), ("cameras", "done"), ("depth", "done"), ("train", "done"),
        ("clean", "done")]
    folder = folder_of(job_id)
    # The Depth_Step read the Cameras_Step's model and fitted a prior to every Placed_Frame.
    depth_report = read_json(folder / "out" / "reports" / "depth.json")
    assert depth_report["counts"]["fitted_images"] == VIDEO_PICKS
    names = pick_names(range(VIDEO_PICKS))
    # work/frames/ as the frames spec lays it out: each Pick in images/ and its Grey_Copy in grey/
    # under the same name, frames.json, and mask.png, because the stuck corner makes the
    # Frames_Step use the Mask.
    frames = folder / "work" / "frames"
    record = read_json(frames / "frames.json")
    assert record["mask_used"] is True
    assert [pick["file"] for pick in record["picks"]] == names
    assert names_in(frames) == sorted(["images", "grey", "frames.json", "mask.png"])
    assert names_in(frames / "images") == names
    assert names_in(frames / "grey") == names
    # work/cameras/ as the design lays it out, with masks/ for the Mask: a Training_Image and a
    # Training_Mask for each of the 5 Placed_Frames, and exactly the three Model_Files in sparse/
    # (Req 10.5, 11.1 to 11.5).
    work = folder / "work" / "cameras"
    assert names_in(work) == work_layout(mask=True)
    assert names_in(work / "colmap") == colmap_layout(mask=True)
    assert names_in(work / "images") == names
    assert names_in(work / "masks") == mask_names(names)
    assert names_in(work / "sparse") == sorted(FILES)
    # out/cameras.json, the Viewer_Cameras, lists the 5 Placed_Frames in Pick_Number order.
    viewer = read_json(folder / "out" / "cameras.json")
    assert [frame["name"] for frame in viewer["frames"]] == names
