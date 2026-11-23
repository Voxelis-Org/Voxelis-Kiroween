"""Tests for the Train_Step's files: splatpipe/train/files.py.

Every Camera_Solve_Output is written into the test's tmp_path by tests/camera_files.py's
write_camera_solve_output (Req 14.1). Its default scene is the model of write_model's default, 3
JPEGs of 8 x 6 pixels in one known colour, IMAGE_BGR, and a mask on the first image only, which
leaves out the image's left half. Each refusal test breaks one thing in it with one of the
function's keyword arguments, on the second image, so a message can't name the right file by
naming the first. The INTER_AREA test writes a JPEG of seeded noise over the second image,
because an image of one colour scales to that colour whatever the interpolation.

Two stand-ins record each path that a Path method is called on and then run the real method. The
one for Path.open, which Path.read_bytes and Path.read_text open their files with, shows that
check_inputs reads nothing (Req 1.2). The one for Path.read_bytes, which read_views reads with,
shows that read_views reads only the files that images.bin names (Req 1.17). The full disk is a
stand-in for os.replace, and for write_json as files.py sees it, that raises
OSError(errno.ENOSPC, "No space left on device") (Req 8.10).

Each refusal is an ExpectedFailure whose message names the file and the reason, and holds no
traceback (Req 14.6). The train_record test gives every number as a numpy value. check_json_value
refuses every numpy scalar, so passing it shows that each value was turned into a plain one
(Req 10.7).
"""

import errno
import os
import shutil
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, NoReturn

import cv2
import numpy as np
import pytest

from splatpipe.errors import ExpectedFailure
from splatpipe.jsonio import check_json_value
from splatpipe.train.colmap import read_model
from splatpipe.train.files import (check_inputs, move_into_out, read_views, train_record,
                                   write_record)
from splatpipe.train.scene import ImagePixels, Scene

from camera_files import (CAMERAS_BIN, DEFAULT_CAMERA, DEFAULT_IMAGES, IMAGE_BGR, IMAGES_BIN,
                          IMAGES_FOLDER, MASKS_FOLDER, POINTS_BIN, SPARSE_FOLDER,
                          write_camera_solve_output)

THRESHOLD = 128  # the real mask_threshold in [train]
FULL_DISK = "No space left on device"  # the reason the stand-in full disk gives

# The default camera's size, 8 x 6, which is also its Training_Size, since it's below long_edge.
WIDTH, HEIGHT = DEFAULT_CAMERA.width, DEFAULT_CAMERA.height
FIRST, SECOND, THIRD = (image.name for image in DEFAULT_IMAGES)  # 000000.jpg to 000002.jpg


def mask_name(image: str) -> str:
    """The name of an image's Mask_File in masks/: its stem, then .png."""
    return image.removesuffix(".jpg") + ".png"


def refused(call: Callable[[], object]) -> str:
    """The message of the ExpectedFailure that `call` raises, once it's checked that the message
    holds no traceback (Req 14.6)."""
    with pytest.raises(ExpectedFailure) as caught:
        call()
    message = str(caught.value)
    assert "Traceback" not in message
    return message


def recorded(monkeypatch: pytest.MonkeyPatch, method: str) -> list[Path]:
    """A list that gets each path that the Path method `method` is called on from now on. The
    real method still runs."""
    calls: list[Path] = []
    real = getattr(Path, method)

    def recording(self: Path, *args: Any, **kwargs: Any) -> Any:
        calls.append(self)
        return real(self, *args, **kwargs)

    monkeypatch.setattr(Path, method, recording)
    return calls


def views_of(cameras: Path, width: int = WIDTH, height: int = HEIGHT) -> list[ImagePixels]:
    """What read_views gives for the Camera_Solve_Output in `cameras`, work/cameras/, at the
    Training_Size `width` x `height`."""
    return read_views(cameras, read_model(cameras / SPARSE_FOLDER), width, height, THRESHOLD)


# Req 1.2: check_inputs

@pytest.mark.parametrize("parts", [
    pytest.param((IMAGES_FOLDER,), id="images"),
    pytest.param((SPARSE_FOLDER, CAMERAS_BIN), id=CAMERAS_BIN),
    pytest.param((SPARSE_FOLDER, IMAGES_BIN), id=IMAGES_BIN),
    pytest.param((SPARSE_FOLDER, POINTS_BIN), id=POINTS_BIN),
])
def test_check_inputs_names_each_missing_path_and_the_camera_solve_step_and_reads_nothing(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, parts: tuple[str, ...]) -> None:
    """With one of the four paths gone from a whole Camera_Solve_Output, check_inputs refuses
    that path, says that the camera solve step writes it and has to run first, and opens no file
    (Req 1.2)."""
    cameras = write_camera_solve_output(tmp_path)
    missing = cameras.joinpath(*parts)
    if missing.is_dir():
        shutil.rmtree(missing)
    else:
        missing.unlink()
    opened = recorded(monkeypatch, "open")
    message = refused(lambda: check_inputs(cameras))
    for words in (str(missing), "which the camera solve step writes",
                  "Run the camera solve step on this job first"):
        assert words in message
    assert opened == []


# Req 1.14, 1.15, 1.17, 3.2: read_views

RGB = IMAGE_BGR[::-1]  # the default images' colour in RGB order
JPEG_ERROR = 3  # the most a decoded pixel of a one-colour JPEG may stray from it, per channel


def test_read_views_gives_rgb_pixels_and_a_keep_mask_only_for_the_image_with_a_mask(
        tmp_path: Path) -> None:
    """Each image comes back at the Training_Size in RGB order, so its first channel is
    IMAGE_BGR's red, 220, and its last the blue, 40 (Req 3.2). Only the first image has a
    Mask_File, so only it gets a keep mask, False in the left half that its mask leaves out
    (Req 1.14, 1.15)."""
    views = views_of(write_camera_solve_output(tmp_path))
    assert len(views) == len(DEFAULT_IMAGES)
    for view in views:
        assert (view.pixels.shape, view.pixels.dtype) == ((HEIGHT, WIDTH, 3), np.uint8)
        assert np.abs(view.pixels.astype(int) - RGB).max() <= JPEG_ERROR
    # The mask is 0 in its first WIDTH // 2 columns and 255 in the rest, and 0 is below 128.
    right_half = np.tile(np.arange(WIDTH) >= WIDTH // 2, (HEIGHT, 1))
    first, *others = views
    assert first.keep is not None and first.keep.dtype == bool
    assert np.array_equal(first.keep, right_half)
    assert [view.keep for view in others] == [None, None]


# A Training_Size below the 8 x 6 camera, as training_size gives it for a long_edge of 5, which
# scales the width by 1.6 and the height by 1.5.
SMALLER = (5, 4)


def test_read_views_scales_an_image_larger_than_the_training_size_with_inter_area(
        tmp_path: Path) -> None:
    """An image of noise at the camera's size comes back as cv2's INTER_AREA scaling of its
    decoded pixels to the Training_Size, in RGB order (Req 3.2)."""
    cameras = write_camera_solve_output(tmp_path)
    noise = np.random.default_rng(5).integers(0, 256, size=(HEIGHT, WIDTH, 3), dtype=np.uint8)
    ok, data = cv2.imencode(".jpg", noise)
    assert ok
    (cameras / IMAGES_FOLDER / SECOND).write_bytes(data.tobytes())
    decoded = cv2.imdecode(data, cv2.IMREAD_COLOR)
    area = cv2.resize(decoded, SMALLER, interpolation=cv2.INTER_AREA)
    # The noise tells INTER_AREA from INTER_LINEAR, cv2.resize's default, at this size.
    assert not np.array_equal(area, cv2.resize(decoded, SMALLER, interpolation=cv2.INTER_LINEAR))
    views = views_of(cameras, *SMALLER)
    assert [view.pixels.shape for view in views] == [(SMALLER[1], SMALLER[0], 3)] * 3
    assert np.array_equal(views[1].pixels, area[..., ::-1])


def test_read_views_never_reads_the_files_that_images_bin_doesnt_name(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """images/ and masks/ each also hold a frame of a fourth image, which images.bin doesn't
    name, and a file that isn't an image, each of junk. read_views reads only each image of
    images.bin and the first image's mask, in images.bin's order (Req 1.17)."""
    cameras = write_camera_solve_output(tmp_path, extra_images=("000003.jpg", "notes.txt"),
                                        extra_masks=("000003.png", "notes.txt"))
    model = read_model(cameras / SPARSE_FOLDER)
    reads = recorded(monkeypatch, "read_bytes")
    read_views(cameras, model, WIDTH, HEIGHT, THRESHOLD)
    images, masks = cameras / IMAGES_FOLDER, cameras / MASKS_FOLDER
    assert reads == [images / FIRST, masks / mask_name(FIRST), images / SECOND, images / THIRD]


# Req 1.12, 1.13, 1.16: each refusal names the file and the reason

@pytest.mark.parametrize(("options", "folder", "name", "reason"), [
    pytest.param({"left_out": {SECOND}}, IMAGES_FOLDER, SECOND, ("has no file of that name",),
                 id="a missing image file"),
    pytest.param({"junk": {SECOND}}, IMAGES_FOLDER, SECOND, ("can't be read as a colour JPEG",),
                 id="junk bytes as a JPEG"),
    pytest.param({"image_sizes": {SECOND: (WIDTH + 1, HEIGHT)}}, IMAGES_FOLDER, SECOND,
                 ("is 9 x 6 pixels", "is 8 x 6 pixels"),
                 id="a JPEG one pixel wider than the camera"),
    pytest.param({"mask_sizes": {SECOND: (WIDTH - 1, HEIGHT)}}, MASKS_FOLDER, mask_name(SECOND),
                 ("is 7 x 6 pixels", "is 8 x 6 pixels"), id="a mask of another size"),
    pytest.param({"fully_masked": {SECOND}}, MASKS_FOLDER, mask_name(SECOND),
                 ("masks every pixel",), id="a mask that masks every pixel"),
])
def test_each_refusal_of_an_image_or_a_mask_names_the_file_and_the_reason(
        tmp_path: Path, options: dict[str, Any], folder: str, name: str,
        reason: tuple[str, ...]) -> None:
    """The message names the file, its name shown with repr() and its folder, and the reason,
    with both sizes when they differ (Req 1.12, 1.13, 1.16)."""
    cameras = write_camera_solve_output(tmp_path, **options)
    message = refused(lambda: views_of(cameras))
    for words in (repr(name), str(cameras / folder), *reason):
        assert words in message


# Req 8.8, 8.10: the move into out/ and the Train_Record on a full disk

def full_disk(*args: object, **kwargs: object) -> NoReturn:
    """A stand-in for os.replace or write_json that fails the way a full disk does."""
    raise OSError(errno.ENOSPC, FULL_DISK)


def test_a_move_into_out_that_fails_like_a_full_disk_names_both_paths_and_the_reason(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    written, target = tmp_path / "work" / "train" / "train.ply", tmp_path / "out" / "train.ply"
    written.parent.mkdir(parents=True)
    target.parent.mkdir()
    written.write_bytes(b"ply\n")
    monkeypatch.setattr(os, "replace", full_disk)
    message = refused(lambda: move_into_out(written, target))
    for words in (str(written), str(target), FULL_DISK):
        assert words in message


# Req 10.7: the Train_Record

DEADLINE = datetime(2026, 11, 21, 14, 25, 30, tzinfo=timezone.utc)
# The Stop_Time, 15 minutes before: the real clean_reserve_minutes plus save_margin_minutes.
STOP = DEADLINE - timedelta(minutes=15)


def record(cap_deadline: datetime | None, stop: datetime | None) -> dict[str, Any]:
    """The Train_Record of a 4 x 3 Training_Size for the 8 x 6 camera, with every number given to
    train_record as a numpy value, as float64 scalars and arrays."""
    scene = Scene(views=(), width=4, height=3, extent=np.float64(2.2),
                  median_depth=np.float64(1.5), near_plane=np.float64(0.045))
    a = np.array([[1.0, 0.0, 0.25], [0.0, 1.0, 0.0], [0.0, 0.5, 1.0]])
    b = np.array([0.125, 0.0, -0.25])
    return train_record(scene=scene, means_lr_start=np.float64(0.000352),
                        width_ratio=np.float64(0.5), height_ratio=np.float64(0.5), a=a, b=b,
                        cap_deadline=cap_deadline, stop=stop)


def test_a_write_record_that_fails_like_a_full_disk_names_the_file_and_the_reason(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "train.json"
    monkeypatch.setattr("splatpipe.train.files.write_json", full_disk)
    message = refused(lambda: write_record(path, record(DEADLINE, STOP)))
    for words in (str(path), FULL_DISK):
        assert words in message


@pytest.mark.parametrize(("cap_deadline", "stop", "cap_text", "stop_text"), [
    pytest.param(DEADLINE, STOP, "2026-11-21T14:25:30+00:00", "2026-11-21T14:10:30+00:00",
                 id="with a cap deadline"),
    pytest.param(None, None, None, None, id="without a cap deadline"),
])
def test_the_train_record_holds_only_plain_json_values(
        cap_deadline: datetime | None, stop: datetime | None, cap_text: str | None,
        stop_text: str | None) -> None:
    """The Mean_Exposure is 3 rows of 4 numbers, [Ā | b̄], and the Cap_Deadline and the
    Stop_Time are ISO 8601 UTC text, both None when no Cap_Deadline was set (Req 10.7)."""
    result = record(cap_deadline, stop)
    check_json_value(result, "train")
    assert result == {
        "near_plane": 0.045,
        "median_scene_depth": 1.5,
        "scene_extent": 2.2,
        "means_lr_start": 0.000352,
        "width_ratio": 0.5,
        "height_ratio": 0.5,
        "mean_exposure": [[1.0, 0.0, 0.25, 0.125], [0.0, 1.0, 0.0, 0.0], [0.0, 0.5, 1.0, -0.25]],
        "cap_deadline": cap_text,
        "stop_time": stop_text,
    }
