"""The Cameras_Smoke_Check's own code with the real COLMAP, on the Smoke_Picks in tmp_path
(cameras spec, Req 16.6, 16.13, 18.4, 18.5).

The integration test calls smoke.run_cameras_smoke with the real config and the real
[cloud.smoke] numbers, both read by path as REPO_ROOT / "config.toml", because tests/conftest.py
points CONFIG_PATH at a test config. Its scratch folder is tmp_path and its Job_Id is FIXED_ID.
So it runs the same Option_Check and Smoke_Solve as the machine image, with the step's real
TOOLS, so the real colmap, which is Homebrew's COLMAP 4.2.1 on a laptop (D15, D18). It's skipped,
saying so, where colmap isn't on the PATH (Req 16.8). The Smoke_Picks' scene, with its level path
(D19), is in splatpipe/cameras/smoke.py, and its self-check in tests/test_cameras_smoke.py.

The SmokeSizes come from [cloud.smoke] as cloud/smoke_runner.py builds them: the seven cameras_*
numbers, and check_timeout_seconds for each colmap <command> -h. So the Smoke_Solve's settings
are the real [cameras] of [jobs].default_preset, with each Time_Limit at cameras_timeout_seconds,
600 s (C18). run_cameras_smoke's own checks pass: the Smoke_Solve is done, at least
cameras_placed_share of the 30 Picks, 80 %, are placed, and the Final_Model's f and the
Undistorted_Camera's fx and fy are each within cameras_focal_tolerance, 2 %, of
cameras_focal_pixels, 260 px (Req 18.5, C19).

The colmap commands the test runs itself, model_analyzer and model_converter, run in a folder of
tmp_path with the environment that colmap_env gives the step's COLMAP runs, so TMPDIR is that
folder and glog's log files stay in tmp_path, and each gets a timeout of 60 s (Req 16.13). Then
the test checks, on the SmokeSolve it gets back and the Job_Folder <tmp_path>/job/, that:

- f, fx and fy are each within 2 % of 260 px, both written out here, and fx = fy (F18, D19);
- the Work_Folder holds exactly colmap/, images/, masks/, sparse/ and solve.json, with a
  Training_Image in images/ for each Placed_Frame (Req 11.1);
- sparse/ holds exactly the three Model_Files, which our Model_Reader reads as one PINHOLE camera
  that every image uses, an image for each Placed_Frame (Req 10.5, 10.6);
- colmap model_analyzer --path work/cameras/sparse exits with 0 and logs Cameras: 1, the placed
  count as Registered images: and the number of points our reader reads from
  sparse/points3D.bin as Points:, and colmap model_converter --output_type BIN, from sparse/ into
  a folder the test makes first, writes a cameras.bin, an images.bin and a points3D.bin equal to
  ours byte for byte, because COLMAP keeps every Pose's bytes when it reads a model and writes it
  again (Req 10.2, F5, O6);
- masks/ holds one PNG per Placed_Frame at the camera's size, with only 0 and 255, and 0 over the
  painted corner where the Undistortion moved it, since it crops the Picks by a few pixels: at
  each pixel whose centre, taken through the Undistorted_Camera and then the Final_Model's
  RADIAL camera from the COLMAP_Folder, with its one f and k1 and k2 on the radius, lies inside
  the painted corner of a Pick (Req 8.4, 11.3, D19);
- the levelled cameras' mean up, each camera's Rᵀ(0, −1, 0), points along (0, −1, 0) within
  1e-9. The true path is level and the cameras have no pitch or roll, so their mean up is the
  true vertical, and their centres' heights spread by less than 2 % of the path's length, from
  the first Placed_Frame's centre to the last's. The Heading_Camera that solve.json names looks
  along +Z within 1e-9 once its view, Rᵀ(0, 0, 1), is flattened (Req 9.2, 9.4, 9.5);
- out/cameras.json holds the camera of sparse/, levelled true, and each Placed_Frame in
  Pick_Number order with its Pose from sparse/images.bin (Req 11.9);
- work/frames/ holds the same bytes, file by file, as write_smoke_picks writes again into another
  folder of tmp_path, so the step changed nothing there (Req 1.7).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tomllib
from pathlib import Path

import cv2
import numpy as np
import pytest

from conftest import FIXED_ID  # the Job_Id the tests give their Jobs
from splatpipe.cameras import smoke
from splatpipe.cameras.colmap import colmap_env
from splatpipe.cameras.model import CAMERA_MODELS, FILES, Camera, Image, read_model
from splatpipe.config import REPO_ROOT, load_config
from splatpipe.frames.files import pick_name
from splatpipe.jsonio import read_json

# Read by path, so the test reads it although CONFIG_PATH points at a test config.
REAL_CONFIG = REPO_ROOT / "config.toml"
# The real [cloud.smoke], each value as tomllib gives it.
REAL_SMOKE = tomllib.loads(REAL_CONFIG.read_text(encoding="utf-8"))["cloud"]["smoke"]
# The Cameras_Smoke_Check's numbers, as cloud/smoke_runner.py builds them from [cloud.smoke]:
# the three whole numbers as they are, and the other five as floats, as cloud/config.py reads them.
SIZES = smoke.SmokeSizes(
    frames=REAL_SMOKE["cameras_frames"], width=REAL_SMOKE["cameras_width"],
    height=REAL_SMOKE["cameras_height"],
    focal_pixels=float(REAL_SMOKE["cameras_focal_pixels"]),
    placed_share=float(REAL_SMOKE["cameras_placed_share"]),
    focal_tolerance=float(REAL_SMOKE["cameras_focal_tolerance"]),
    timeout_seconds=float(REAL_SMOKE["cameras_timeout_seconds"]),
    help_timeout_seconds=float(REAL_SMOKE["check_timeout_seconds"]))

# The test's own focal check, written out here: the Smoke_Picks' true f, and how far f, fx and fy
# may be from it, as a share of it.
TRUE_FOCAL = 260.0  # in pixels
FOCAL_SHARE = 0.02
# The black corner painted into every Smoke_Pick: its top left PAINTED_WIDTH × PAINTED_HEIGHT
# pixels, 32 × 24 at 320 × 240. mask.png is 0 there and 255 elsewhere.
PAINTED_WIDTH = SIZES.width // smoke.PAINTED_PART
PAINTED_HEIGHT = SIZES.height // smoke.PAINTED_PART
# The timeout of each colmap command the test runs itself, model_analyzer and model_converter,
# which each read a model of at most 30 images (Req 16.13).
CHECK_TIMEOUT = 60  # in seconds
UP = np.array([0.0, -1.0, 0.0])  # the World_Up of the levelled frame
AHEAD = np.array([0.0, 0.0, 1.0])  # where the Heading_Camera's flattened view points, +Z
TOLERANCE = 1e-9  # how close the levelled directions must come to UP and AHEAD
HEIGHT_SPREAD = 0.02  # how far the centres' heights may spread, as a share of the path's length
# The three counts that colmap model_analyzer logs and the test reads, each after glog's prefix,
# which ends in "] ", like "I20261123 04:19:19.624758 0x1705ab000 model.cc:1234] Points: 1234"
# (F16).
ANALYZER_COUNT = re.compile(r"\] (Cameras|Registered images|Points): ([0-9]+)$", re.MULTILINE)


def colmap_check(folder: Path, *args: str) -> str:
    """Run colmap with `args` and give what it logged, once it has exited with 0.

    It runs in `folder`, with stdin closed and the environment that colmap_env gives the step's
    COLMAP runs, so TMPDIR is `folder` and glog writes its log files there, with a timeout of
    CHECK_TIMEOUT. glog logs to stderr, so stderr goes into stdout and both are read.
    """
    result = subprocess.run(["colmap", *args], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, cwd=folder,
                            env=colmap_env(folder, os.environ), encoding="utf-8",
                            errors="replace", timeout=CHECK_TIMEOUT)
    assert result.returncode == 0, (args, result.returncode, result.stdout)
    return result.stdout


def hashes(folder: Path) -> dict[str, str | None]:
    """Every file and folder inside `folder`, by its path from `folder`, like images/000000.jpg:
    a file's SHA-256, and None for a folder."""
    return {path.relative_to(folder).as_posix():
            hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
            for path in sorted(folder.rglob("*"))}


def names_in(folder: Path) -> list[str]:
    """The names of the files and folders right inside `folder`, sorted."""
    return sorted(path.name for path in folder.iterdir())


def read_png(path: Path) -> np.ndarray:
    """The image in the PNG at `path`, as cv2.imdecode reads it with its channels and bit depth
    unchanged."""
    image = cv2.imdecode(np.frombuffer(path.read_bytes(), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    assert image is not None, f"cv2.imdecode couldn't read {path}"
    return image


def painted_corner(undistorted: Camera, distorted: Camera) -> np.ndarray:
    """Which pixels of a Training_Mask show the corner painted into the Picks: a bool array of
    the height × width of `undistorted`, the Undistorted_Camera, true at each pixel whose centre
    lies inside the corner in a Pick of `distorted`, the Final_Model's RADIAL camera.

    The pixel centred at (u, v), with the top left pixel's centre at (0.5, 0.5), looks along
    x = (u − cx) / fx and y = (v − cy) / fy through the PINHOLE camera. RADIAL's distortion, k1
    and k2 on the radius, takes that to x (1 + k1 r² + k2 r⁴) and y (1 + k1 r² + k2 r⁴), with
    r² = x² + y², and its one focal length f and its principal point then give the place in the
    Pick (F2, D19). The corner is the Pick's top PAINTED_HEIGHT rows of its left PAINTED_WIDTH
    columns. The Mask_Warp samples mask.png bilinearly at that place, so a pixel inside the
    corner takes at least a quarter of its value from the corner's 0s, and the threshold makes
    it 0. A pixel just outside may go either way.
    """
    assert CAMERA_MODELS[distorted.model_id][0] == "RADIAL"
    fx, fy, cx, cy = undistorted.params
    u, v = np.meshgrid(np.arange(undistorted.width) + 0.5, np.arange(undistorted.height) + 0.5)
    x, y = (u - cx) / fx, (v - cy) / fy
    focal, pick_cx, pick_cy, k1, k2 = distorted.params
    r2 = x * x + y * y
    radial = 1 + k1 * r2 + k2 * r2 * r2
    return ((focal * x * radial + pick_cx < PAINTED_WIDTH)
            & (focal * y * radial + pick_cy < PAINTED_HEIGHT))


def rotation_of(image: Image) -> np.ndarray:
    """The rotation R of the image's Pose, from its unit quaternion (w, x, y, z) in Hamilton's
    convention. Its rows are the camera's x, y and z axes in the world."""
    w, x, y, z = image.qvec
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def direction(vector: np.ndarray) -> np.ndarray:
    """`vector` divided by its length."""
    return vector / np.linalg.norm(vector)


def close(found: np.ndarray, expected: np.ndarray) -> bool:
    """Whether every number of `found` is within TOLERANCE of the same number of `expected`."""
    return bool(np.allclose(found, expected, rtol=0, atol=TOLERANCE))


@pytest.mark.skipif(shutil.which("colmap") is None, reason="COLMAP isn't installed")
def test_the_cameras_smoke_check_places_the_smoke_picks_and_reads_back_the_levelled_model(
        tmp_path: Path) -> None:
    config = load_config(REAL_CONFIG)
    # The Option_Check, then the Smoke_Solve with the step's real TOOLS, so the real colmap, and
    # its checks, which raise a PipelineError when they fail (Req 18.4, 18.5).
    solve = smoke.run_cameras_smoke(config, SIZES, tmp_path, FIXED_ID).solve
    # f, fx and fy are within FOCAL_SHARE of TRUE_FOCAL once more, and fx = fy (F18, D19).
    for name, value in (("f", solve.focal), ("fx", solve.fx), ("fy", solve.fy)):
        assert abs(value - TRUE_FOCAL) <= FOCAL_SHARE * TRUE_FOCAL, (name, value)
    assert solve.fx == solve.fy, (solve.fx, solve.fy)
    # The Work_Folder: exactly colmap/, images/, masks/, sparse/ and solve.json, since the Picks
    # use the Mask, with a Training_Image for each name in sparse/images.bin (Req 11.1).
    folder = tmp_path / "job"  # the Smoke_Solve's Job_Folder
    work = folder / "work" / "cameras"
    sparse = work / "sparse"
    assert names_in(work) == ["colmap", "images", "masks", "solve.json", "sparse"]
    # sparse/ holds exactly the three Model_Files, which our reader reads as one PINHOLE camera
    # that every image uses, an image for each Placed_Frame (Req 10.5, 10.6).
    assert names_in(sparse) == sorted(FILES)
    model = read_model(sparse)
    (pinhole,) = model.cameras
    assert CAMERA_MODELS[pinhole.model_id][0] == "PINHOLE"
    assert {image.camera_id for image in model.images} == {pinhole.camera_id}
    images = {image.name: image for image in model.images}
    assert len(images) == len(model.images) == solve.placed
    assert names_in(work / "images") == sorted(images)
    # COLMAP reads sparse/ back. model_analyzer counts the one camera, the Placed_Frames and the
    # points our reader reads (Req 16.6). model_converter, writing it again into a folder made
    # for it, gives our three files byte for byte, beside its own rigs.bin and frames.bin (Req
    # 10.2, F5, O6).
    checks = tmp_path / "colmap_checks"
    checks.mkdir()
    logged = colmap_check(checks, "model_analyzer", "--path", str(sparse))
    assert dict(ANALYZER_COUNT.findall(logged)) == {
        "Cameras": "1", "Registered images": str(solve.placed),
        "Points": str(len(model.points.ids))}, logged
    converted = tmp_path / "converted"
    converted.mkdir()
    colmap_check(checks, "model_converter", "--input_path", str(sparse), "--output_path",
                 str(converted), "--output_type", "BIN")
    for name in FILES:
        assert (converted / name).read_bytes() == (sparse / name).read_bytes(), name
    # masks/: one PNG per Placed_Frame, named after its stem, at the camera's size, with only 0
    # and 255, and 0 over the corner painted black into every Pick, where the Undistortion moved
    # it, as the Final_Model's camera in the COLMAP_Folder gives it (Req 8.4, 11.3).
    record = read_json(work / "solve.json")
    (final,) = read_model(work / "colmap" / record["mapper"] / str(record["final_model"])).cameras
    painted = painted_corner(pinhole, final)
    assert painted.any()
    masks = work / "masks"
    assert names_in(masks) == sorted(f"{Path(name).stem}.png" for name in images)
    for path in sorted(masks.iterdir()):
        mask = read_png(path)
        assert mask.dtype == np.uint8 and mask.shape == (pinhole.height, pinhole.width), path
        assert np.unique(mask).tolist() == [0, 255], path
        assert not mask[painted].any(), path
    # The levelled cameras. Their mean up points along the World_Up. The true path is level and
    # the cameras have no pitch or roll, so their mean up is the true vertical, and their
    # centres, −Rᵀt, keep to one height, up to HEIGHT_SPREAD of the path's length from the first
    # Placed_Frame's centre to the last's. The Heading_Camera that solve.json names has its view
    # along +Z once the part along the World_Up is taken away (Req 9.2, 9.4, 9.5).
    assert record["levelled"] is True, record["level_reason"]
    rotations = {name: rotation_of(image) for name, image in images.items()}
    mean_up = np.mean([-rotation[1] for rotation in rotations.values()], axis=0)
    assert close(direction(mean_up), UP), mean_up
    placed = [pick_name(number) for number in range(SIZES.frames) if pick_name(number) in images]
    centres = [-rotations[name].T @ np.array(images[name].tvec) for name in placed]
    heights = [float(centre @ UP) for centre in centres]
    length = float(np.linalg.norm(centres[-1] - centres[0]))
    assert max(heights) - min(heights) < HEIGHT_SPREAD * length, (heights, length)
    view = rotations[record["heading_camera"]][2]
    assert close(direction(view - (view @ UP) * UP), AHEAD), view
    # out/cameras.json: the camera of sparse/, levelled, and each Placed_Frame in Pick_Number
    # order with its Pose from sparse/images.bin, and nothing else (Req 11.9). json.dumps
    # compares the kinds too, so 1.0 isn't 1.
    fx, fy, cx, cy = pinhole.params
    expected = {
        "camera": {"width": pinhole.width, "height": pinhole.height, "fx": fx, "fy": fy,
                   "cx": cx, "cy": cy},
        "levelled": True,
        "frames": [{"name": name, "quaternion_wxyz": list(images[name].qvec),
                    "translation": list(images[name].tvec)} for name in placed],
    }
    assert json.dumps(read_json(folder / "out" / "cameras.json")) == json.dumps(expected)
    # work/frames/ holds what write_smoke_picks writes again, with [frames]'s jpeg_quality for
    # [jobs].default_preset, file by file, so the step changed nothing there (Req 1.7).
    again = tmp_path / "smoke_picks_again"
    quality = config.step_settings("frames", config.jobs().default_preset)["jpeg_quality"]
    smoke.write_smoke_picks(again, SIZES, quality)
    assert hashes(folder / "work" / "frames") == hashes(again)
