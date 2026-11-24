"""Tests for the Depth_Step's files: splatpipe/depth/files.py (depth-priors spec, Req 2.2, 2.7,
4.1, 4.2, 7.6, 7.10).

Every Camera_Solve_Output is written into the test's tmp_path by tests/camera_files.py's
write_camera_solve_output, with the plane scene: 4 JPEGs of 16 x 12 pixels in one known colour,
IMAGE_BGR. The camera's longer edge is below [depth]'s long_edge of 512, so its Prior_Size is its
own size. The scaling tests write the same scene with WIDE_CAMERA, 640 x 480, whose Prior_Size is
512 x 384, worked out here by hand: each edge × 512 / 640. The INTER_AREA test writes a JPEG of
seeded noise over one Frame, because a Frame of one colour scales to that colour whatever the
interpolation.

A stand-in for Path.open, which Path.read_bytes and Path.read_text open their files with, records
each path it's called on and then runs the real method, so it shows that check_inputs reads
nothing (Req 2.2). The full disk is a stand-in for np.savez, for write_json as
splatpipe.train.priors sees it, and for os.replace, that raises
OSError(errno.ENOSPC, "No space left on device") (Req 7.10).

Each refusal is an ExpectedFailure whose whole message is the one the design's Error Handling
table gives, or the Train_Step's own for a Frame, and holds no traceback (Req 18.7).
"""

from __future__ import annotations

import errno
import os
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any, NoReturn

import cv2
import numpy as np
import pytest

from splatpipe.depth.files import check_inputs, read_frame, write_prior_file, write_prior_index
from splatpipe.errors import ExpectedFailure
from splatpipe.train.colmap import Camera, read_model
from splatpipe.train.priors import Prior, PriorEntry, PriorIndex

from camera_files import (CAMERAS_BIN, IMAGE_BGR, IMAGES_BIN, IMAGES_FOLDER, PLANE_CAMERA,
                          PLANE_IMAGES, PLANE_POINTS, POINTS_BIN, SPARSE_FOLDER, CameraRecord,
                          write_camera_solve_output)

FULL_DISK = "No space left on device"  # the reason the stand-in full disk gives
_AGAIN = "Run the camera solve step on this job again."  # how the Train_Step's messages end

SECOND = PLANE_IMAGES[1].name  # 000001.jpg, so a message can't name the right file by the first

# A camera wider than [depth]'s long_edge of 512, and its Prior_Size: 640 x 480 × 512 / 640.
WIDE_CAMERA = CameraRecord(camera_id=PLANE_CAMERA.camera_id, width=640, height=480, fx=640.0,
                           fy=640.0, cx=320.0, cy=240.0)
WIDE_PRIOR_SIZE = (512, 384)

RGB = IMAGE_BGR[::-1]  # the Frames' colour in RGB order
JPEG_ERROR = 3  # the most a decoded pixel of a one-colour JPEG may stray from it, per channel


def refused(call: Callable[[], object]) -> str:
    """The message of the ExpectedFailure that `call` raises, once it's checked that the message
    holds no traceback (Req 18.7)."""
    with pytest.raises(ExpectedFailure) as caught:
        call()
    message = str(caught.value)
    assert "Traceback" not in message
    return message


def recorded_opens(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    """A list that gets each path that Path.open is called on from now on. The real method still
    runs."""
    calls: list[Path] = []
    real = Path.open

    def recording(self: Path, *args: Any, **kwargs: Any) -> Any:
        calls.append(self)
        return real(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", recording)
    return calls


def plane_output(job_folder: Path, camera: CameraRecord = PLANE_CAMERA, **options: Any) -> Path:
    """The plane scene's Camera_Solve_Output in `job_folder`, with `camera` and
    write_camera_solve_output's keyword arguments, as its work/cameras/ folder."""
    return write_camera_solve_output(job_folder, camera, PLANE_IMAGES, PLANE_POINTS, **options)


def shared_camera(cameras: Path) -> Camera:
    """The Shared_Camera of the Camera_Solve_Output in `cameras`, as the Model_Reader reads it."""
    return read_model(cameras / SPARSE_FOLDER).camera


def decoded(path: Path) -> np.ndarray:
    """The JPEG at `path` as cv2 decodes it, in BGR, with the flags a Frame is decoded with."""
    data = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)


# Req 2.2: check_inputs

# The four paths check_inputs checks, in its order, relative to work/cameras/.
INPUTS = ((IMAGES_FOLDER,), (SPARSE_FOLDER, CAMERAS_BIN), (SPARSE_FOLDER, IMAGES_BIN),
          (SPARSE_FOLDER, POINTS_BIN))


@pytest.mark.parametrize("first", range(len(INPUTS)),
                         ids=[parts[-1] for parts in INPUTS])
def test_check_inputs_refuses_the_first_missing_path_naming_it_and_the_camera_solve_step(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, first: int) -> None:
    """With the path `first` and every path after it in check_inputs' order gone, check_inputs
    refuses that path with the Req 2.2 message. So each of the four is refused in turn, once
    every path before it is there. It opens no file."""
    cameras = plane_output(tmp_path)
    for parts in INPUTS[first:]:
        path = cameras.joinpath(*parts)
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    opened = recorded_opens(monkeypatch)
    missing = cameras.joinpath(*INPUTS[first])
    assert refused(lambda: check_inputs(cameras)) == (
        f"The depth step needs {missing}, which the camera solve step writes, but it isn't there. "
        "Run the camera solve step on this job first.")
    assert opened == []


def test_check_inputs_passes_a_whole_camera_solve_output_and_reads_nothing(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cameras = plane_output(tmp_path)
    opened = recorded_opens(monkeypatch)
    check_inputs(cameras)
    assert opened == []


# Req 2.7, 4.1, 4.2: read_frame

def test_read_frame_scales_a_frame_wider_than_512_pixels_to_the_prior_size_in_rgb(
        tmp_path: Path) -> None:
    """A Frame of the 640 x 480 camera comes back at its Prior_Size, 512 x 384, as uint8 in RGB
    order, so its first channel is IMAGE_BGR's red, 220, and its last the blue, 40, and it's
    cv2's INTER_AREA scaling of the decoded JPEG (Req 4.1, 4.2)."""
    cameras = plane_output(tmp_path, WIDE_CAMERA)
    width, height = WIDE_PRIOR_SIZE
    frame = read_frame(cameras, SECOND, shared_camera(cameras), width, height)
    assert (frame.shape, frame.dtype) == ((height, width, 3), np.uint8)
    assert np.abs(frame.astype(int) - RGB).max() <= JPEG_ERROR
    area = cv2.resize(decoded(cameras / IMAGES_FOLDER / SECOND), WIDE_PRIOR_SIZE,
                      interpolation=cv2.INTER_AREA)
    assert np.array_equal(frame, area[..., ::-1])


def test_read_frame_scales_down_with_inter_area(tmp_path: Path) -> None:
    """A Frame of seeded noise of the 640 x 480 camera comes back as cv2's INTER_AREA scaling of
    its decoded pixels to 512 x 384, in RGB order, which the noise tells from INTER_LINEAR,
    cv2.resize's default (Req 4.2)."""
    cameras = plane_output(tmp_path, WIDE_CAMERA)
    noise = np.random.default_rng(5).integers(0, 256, size=(WIDE_CAMERA.height,
                                                            WIDE_CAMERA.width, 3), dtype=np.uint8)
    ok, data = cv2.imencode(".jpg", noise)
    assert ok
    path = cameras / IMAGES_FOLDER / SECOND
    path.write_bytes(data.tobytes())
    pixels = decoded(path)
    area = cv2.resize(pixels, WIDE_PRIOR_SIZE, interpolation=cv2.INTER_AREA)
    assert not np.array_equal(area, cv2.resize(pixels, WIDE_PRIOR_SIZE,
                                               interpolation=cv2.INTER_LINEAR))
    frame = read_frame(cameras, SECOND, shared_camera(cameras), *WIDE_PRIOR_SIZE)
    assert np.array_equal(frame, area[..., ::-1])


def test_read_frame_keeps_a_camera_smaller_than_512_pixels_at_its_size(tmp_path: Path) -> None:
    """The plane scene's 16 x 12 camera is its own Prior_Size, so its Frame comes back as the
    decoded JPEG itself, only in RGB order (Req 4.1)."""
    cameras = plane_output(tmp_path)
    frame = read_frame(cameras, SECOND, shared_camera(cameras), PLANE_CAMERA.width,
                       PLANE_CAMERA.height)
    assert (frame.shape, frame.dtype) == ((PLANE_CAMERA.height, PLANE_CAMERA.width, 3), np.uint8)
    assert np.abs(frame.astype(int) - RGB).max() <= JPEG_ERROR
    assert np.array_equal(frame, decoded(cameras / IMAGES_FOLDER / SECOND)[..., ::-1])


@pytest.mark.parametrize(("options", "message"), [
    pytest.param({"left_out": {SECOND}},
                 "images.bin names the image '000001.jpg', but {images} has no file of that "
                 f"name. {_AGAIN}", id="a missing file"),
    pytest.param({"junk": {SECOND}},
                 "The training image '000001.jpg' in {images} can't be read as a colour JPEG: "
                 f"OpenCV couldn't decode it. {_AGAIN}", id="junk bytes"),
    pytest.param({"image_sizes": {SECOND: (PLANE_CAMERA.width + 1, PLANE_CAMERA.height)}},
                 "The training image '000001.jpg' in {images} is 17 x 12 pixels, but the shared "
                 "camera in cameras.bin is 16 x 12 pixels, and every training image must be the "
                 f"camera's size. {_AGAIN}", id="a JPEG one pixel wider"),
])
def test_read_frame_refuses_a_frame_with_the_train_steps_messages(
        tmp_path: Path, options: dict[str, Any], message: str) -> None:
    """A Frame with no file, of junk bytes or one pixel wider than the camera is refused with the
    Train_Step's own message, naming the Frame with repr() and its folder (Req 2.7)."""
    cameras = plane_output(tmp_path, **options)
    camera = shared_camera(cameras)
    found = refused(lambda: read_frame(cameras, SECOND, camera, PLANE_CAMERA.width,
                                       PLANE_CAMERA.height))
    assert found == message.format(images=cameras / IMAGES_FOLDER)


# Req 7.6, 7.10: the Prior_File and the Prior_Index on a full disk

def full_disk(*args: object, **kwargs: object) -> NoReturn:
    """A stand-in for np.savez, write_json or os.replace that fails the way a full disk does."""
    raise OSError(errno.ENOSPC, FULL_DISK)


def half_written(path: Path, value: object) -> NoReturn:
    """A stand-in for write_json that writes the start of the file at `path`, then fails the way
    a full disk does, so a temporary file is left for write_index to remove."""
    path.write_bytes(b'{"format": 1, "cam')
    raise OSError(errno.ENOSPC, FULL_DISK)


def not_written(path: Path) -> str:
    """The Req 7.10 message for the file at `path` on the stand-in full disk."""
    return (f"The file {path} couldn't be written ({FULL_DISK}). Free some space on the disk, or "
            f"check that you can write to {path.parent}, then run the step again.")


def depth_folder(tmp_path: Path) -> Path:
    """An empty work/depth/ in tmp_path, as the Step_Runner leaves it before the step runs."""
    folder = tmp_path / "work" / "depth"
    folder.mkdir(parents=True)
    return folder


# One Fitted_Frame's Prior at a Prior_Size of 3 x 2, every pixel with a prior.
PRIOR = Prior(inverse_depth=np.full((2, 3), 0.5, dtype=np.float32),
              normal=np.tile(np.array([0.0, 0.0, -1.0], dtype=np.float32), (2, 3, 1)),
              mask=np.ones((2, 3), dtype=bool))

# A Prior_Index of that one Prior, for the plane scene's camera.
INDEX = PriorIndex(
    camera=(PLANE_CAMERA.width, PLANE_CAMERA.height, PLANE_CAMERA.fx, PLANE_CAMERA.fy,
            PLANE_CAMERA.cx, PLANE_CAMERA.cy),
    prior_width=3, prior_height=2, metres_per_unit=2.0,
    priors=(PriorEntry(name=SECOND, quaternion=(1.0, 0.0, 0.0, 0.0), translation=(0.0, 0.0, 1.0),
                       scale=2.0, offset=0.0, fit_points=30, kept_points=30, median_error=0.0,
                       metres_per_unit=2.0),))


def test_a_prior_file_that_fails_like_a_full_disk_gives_the_req_7_10_message(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """np.savez failing on a full disk is refused, naming 000001.npz, the reason and the folder
    (Req 7.10)."""
    folder = depth_folder(tmp_path)
    monkeypatch.setattr(np, "savez", full_disk)
    message = refused(lambda: write_prior_file(folder, SECOND, PRIOR))
    assert message == not_written(folder / "000001.npz")


@pytest.mark.parametrize(("target", "stand_in"), [
    pytest.param("splatpipe.train.priors.write_json", full_disk, id="write_json fails"),
    pytest.param("splatpipe.train.priors.write_json", half_written,
                 id="write_json fails half-way"),
    pytest.param("os.replace", full_disk, id="os.replace fails"),
])
def test_a_prior_index_that_fails_like_a_full_disk_gives_the_req_7_10_message_and_leaves_none(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, target: str,
        stand_in: Callable[..., NoReturn]) -> None:
    """write_json or os.replace failing on a full disk is refused, naming depth.json, the reason
    and the folder, and leaves neither depth.json nor depth.json.tmp, so a failed run leaves no
    Prior_Index (Req 7.6, 7.10)."""
    folder = depth_folder(tmp_path)
    monkeypatch.setattr(target, stand_in)
    message = refused(lambda: write_prior_index(folder, INDEX))
    assert message == not_written(folder / "depth.json")
    assert sorted(os.listdir(folder)) == []
