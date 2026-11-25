"""Tests for splatpipe/cameras/undistort.py: the checks of the Undistortion's output, the
Mask_Warp's camera line and the Undistorted_Mask.

- check_undistorted on small Undistorted_Models built here, with the Undistortion's images/
  folder in tmp_path holding an empty file per image, since only being a file counts. It refuses
  two cameras, a RADIAL camera that COLMAP kept because its distortion came out as 0, whose
  message says that COLMAP keeps a camera's own model when its distortion came out as 0 and
  names neither mapper_ba_refine_extra_params nor global_mapper_ba_refine_extra_params, which the
  settings check keeps true (C8, F9, D20), an image that uses another camera than the model's one
  camera, naming the image, its camera id and the one camera's id (16.15), an image that isn't a
  Placed_Frame, a Placed_Frame left out of the model, and a missing image file. Each refusal says
  that COLMAP's
  image_undistorter got it wrong and to run the step again, and holds no traceback (Req 8.2,
  16.9). A good model, whose images come in another order than the Placed_Frames, gives its
  PINHOLE camera.
- mask_camera_line for a RADIAL and a RADIAL_FISHEYE camera, each with its five parameters f,
  cx, cy, k1 and k2, whose shortest text needs up to 17 significant digits or an exponent, gives
  one line that, split on single spaces, reads back to mask.png, the model's name, the width, the
  height and every parameter through float() (Req 8.3, F2, F8, D19).
- read_warped_mask on files written here, the PNGs with cv2.imencode: a missing file, bytes that
  aren't an image, a 16-bit PNG, and a PNG one pixel too wide or too tall each fail, naming the
  file and what's wrong and saying to run the step again, with no traceback (Req 8.5, 16.9). A
  three-channel PNG with grey edges, down to pixels with one channel at 254, and a one-channel PNG
  each give one uint8 channel of only 0 and 255, with 255 exactly where every channel is 255
  (Req 8.4).
- undistorted_mask on a 2-D, a one-channel, a three-channel and a four-channel picture, where the
  fourth channel counts like the others (Req 8.4, F9).
- Property 3 draws with Hypothesis uint8 pictures of 1 × 1 to 16 × 16 pixels with 3 channels,
  often on a white background, with channel values drawn mostly from 0, 1, 254 and 255, so
  pixels with one channel just below 255 are common. A plain loop over the pixels checks that
  undistorted_mask gives one uint8 channel of the same height and width, holding only 0 and 255,
  with 255 at exactly the pixels whose three channels are all 255 (Req 8.4, 8.7).
"""

import math
import re
from collections.abc import Iterable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from splatpipe.cameras.model import POINT2D, Camera, Image, Model, Points
from splatpipe.cameras.undistort import (check_undistorted, mask_camera_line, read_warped_mask,
                                         undistorted_mask)
from splatpipe.errors import ExpectedFailure

WIDTH, HEIGHT = 16, 12  # the Undistorted_Camera's size in these tests


# check_undistorted (Req 8.2, C8, F9)

PLACED = ("000000.jpg", "000001.jpg", "000002.jpg")  # the Final_Model's Placed_Frames
# The Undistorted_Camera, with fx = fy (F18), and a RADIAL camera, f, cx, cy, k1, k2, that
# COLMAP's undistorter kept as it was, because its distortion came out as 0 (F9).
PINHOLE = Camera(camera_id=1, model_id=1, width=WIDTH, height=HEIGHT,
                 params=(14.0, 14.0, 8.0, 6.0))
KEPT_RADIAL = Camera(camera_id=1, model_id=3, width=WIDTH, height=HEIGHT,
                     params=(14.0, 8.0, 6.0, 0.0, 0.0))
# C8's words in the refusal of a kept camera.
KEPT_MODEL = "COLMAP keeps a camera's own model when the camera's distortion came out as 0."


def undistorted_model(cameras: Sequence[Camera], names: Sequence[str]) -> Model:
    """An Undistorted_Model with `cameras` and an image for each of `names`, in that order, each
    using camera 1 and with no 2D points, and no points."""
    images = tuple(Image(image_id=place + 1, qvec=(1.0, 0.0, 0.0, 0.0), tvec=(0.0, 0.0, 0.0),
                         camera_id=1, name=name, points2d=np.zeros(0, dtype=POINT2D))
                   for place, name in enumerate(names))
    points = Points(ids=np.zeros(0, dtype="<u8"), xyz=np.zeros((0, 3)),
                    rgb=np.zeros((0, 3), dtype="u1"), errors=np.zeros(0), tracks=())
    return Model(cameras=tuple(cameras), images=images, points=points)


def undistorted_images(tmp_path: Path, names: Iterable[str]) -> Path:
    """The Undistortion's images/ folder, colmap/undistorted/images/ in `tmp_path`, with an empty
    file for each of `names`. check_undistorted only checks that each image's file is a file."""
    folder = tmp_path / "colmap" / "undistorted" / "images"
    folder.mkdir(parents=True)
    for name in names:
        (folder / name).write_bytes(b"")
    return folder


def undistortion_refusal(model: Model, images: Path) -> str:
    """The Readable_Error of check_undistorted on `model`, PLACED and `images`, checked to say
    that COLMAP's image_undistorter got something wrong and to run the step again, and to hold no
    traceback (Req 8.2, 16.9)."""
    with pytest.raises(ExpectedFailure) as caught:
        check_undistorted(model, PLACED, images)
    message = str(caught.value)
    assert "image_undistorter" in message
    assert "run the step again" in message.lower()
    assert "Traceback" not in message
    return message


def test_check_undistorted_gives_the_pinhole_camera_of_a_good_model(tmp_path: Path) -> None:
    names = ("000002.jpg", "000000.jpg", "000001.jpg")  # COLMAP's order, not Pick_Number order
    images = undistorted_images(tmp_path, names)
    assert check_undistorted(undistorted_model((PINHOLE,), names), PLACED, images) == PINHOLE


def test_check_undistorted_refuses_two_cameras(tmp_path: Path) -> None:
    images = undistorted_images(tmp_path, PLACED)
    model = undistorted_model((PINHOLE, replace(PINHOLE, camera_id=2)), PLACED)
    assert "2 cameras" in undistortion_refusal(model, images)


def test_check_undistorted_refuses_a_kept_radial_camera_saying_why_and_naming_no_refine_key(
        tmp_path: Path) -> None:
    images = undistorted_images(tmp_path, PLACED)
    message = undistortion_refusal(undistorted_model((KEPT_RADIAL,), PLACED), images)
    # Word by word, so RADIAL isn't found inside RADIAL_FISHEYE.
    words = re.findall(r"\w+", message)
    assert "RADIAL" in words and "PINHOLE" in words
    # C8's words: COLMAP keeps a camera's own model when its distortion came out as 0. And no word
    # of mapper_ba_refine_extra_params or global_mapper_ba_refine_extra_params, which the settings
    # check keeps true (D20).
    assert KEPT_MODEL in message
    assert "ba_refine_extra_params" not in message


def test_check_undistorted_refuses_an_image_on_another_camera_naming_both_ids(
        tmp_path: Path) -> None:
    images = undistorted_images(tmp_path, PLACED)
    model = undistorted_model((PINHOLE,), PLACED)
    # With every image on the one camera, camera 1, the model passes.
    assert check_undistorted(model, PLACED, images) == PINHOLE
    # The same model with 000001.jpg on camera 2, which the model doesn't hold (16.15).
    moved = tuple(replace(image, camera_id=2) if image.name == "000001.jpg" else image
                  for image in model.images)
    message = undistortion_refusal(replace(model, images=moved), images)
    assert "000001.jpg" in message
    assert "camera id 2" in message
    assert "the id 1" in message


def test_check_undistorted_refuses_an_image_that_isnt_a_placed_frame(tmp_path: Path) -> None:
    names = (*PLACED, "000005.jpg")  # Pick 5, which the Final_Model didn't place
    images = undistorted_images(tmp_path, names)
    assert "000005.jpg" in undistortion_refusal(undistorted_model((PINHOLE,), names), images)


def test_check_undistorted_refuses_a_placed_frame_missing_from_the_model(tmp_path: Path) -> None:
    names = ("000000.jpg", "000002.jpg")
    images = undistorted_images(tmp_path, names)
    assert "000001.jpg" in undistortion_refusal(undistorted_model((PINHOLE,), names), images)


def test_check_undistorted_refuses_a_missing_image_file_naming_it(tmp_path: Path) -> None:
    images = undistorted_images(tmp_path, ("000000.jpg", "000002.jpg"))
    message = undistortion_refusal(undistorted_model((PINHOLE,), PLACED), images)
    assert str(images / "000001.jpg") in message


# mask_camera_line (Req 8.3, F8)

# The Final_Model's camera before the Undistortion, as bundle adjustment might leave it, with its
# five parameters f, cx, cy, k1 and k2 (F2). The shortest text that reads back as some of them
# needs 17 significant digits, like 0.1 + 0.2, or an exponent, so a line written with fewer
# digits would read back to other doubles. The fisheye camera's focal length is the
# Fisheye_Start's for 120° across 1920 pixels.
RADIAL_CAMERA = Camera(camera_id=1, model_id=3, width=1920, height=1080,
                       params=(1000 / 0.7, 959.5, 539.5, -(0.1 + 0.2), 1e-5 / 7))
FISHEYE_CAMERA = Camera(camera_id=1, model_id=9, width=1920, height=1080,
                        params=(960 / (math.pi / 3), 959.5, 539.5, 0.1 + 0.2, -2e-6 / 3))


@pytest.mark.parametrize(("camera", "model"), [
    pytest.param(RADIAL_CAMERA, "RADIAL", id="RADIAL"),
    pytest.param(FISHEYE_CAMERA, "RADIAL_FISHEYE", id="RADIAL_FISHEYE"),
])
def test_mask_camera_line_reads_back_to_the_name_model_size_and_parameters(
        camera: Camera, model: str) -> None:
    line = mask_camera_line(camera)
    assert line.endswith("\n") and line.count("\n") == 1
    name, model_name, width, height, *params = line[:-1].split(" ")
    assert (name, model_name) == ("mask.png", model)
    assert (int(width), int(height)) == (camera.width, camera.height)
    assert tuple(float(param) for param in params) == camera.params


# read_warped_mask (Req 8.4, 8.5)

# The Mask_Warp's output in these tests, WIDTH × HEIGHT: white, with a black rectangle, the grey
# ring one pixel wide that bilinear sampling leaves around it, and around that a faint ring just
# below white.
FAINT, GREY, BLACK = np.s_[2:10, 3:13], np.s_[3:9, 4:12], np.s_[4:8, 5:11]


def warped_picture(faint: int | tuple[int, ...], *channels: int) -> np.ndarray:
    """The Mask_Warp's output with `channels`, or 2-D with none, with its faint ring at
    `faint`."""
    picture = np.full((HEIGHT, WIDTH, *channels), 255, dtype=np.uint8)
    picture[FAINT] = faint
    picture[GREY] = 128
    picture[BLACK] = 0
    return picture


def expected_mask() -> np.ndarray:
    """The Undistorted_Mask of each warped_picture: 0 (ignore) from the faint ring in, and 255
    (use) outside it."""
    mask = np.full((HEIGHT, WIDTH), 255, dtype=np.uint8)
    mask[FAINT] = 0
    return mask


def write_png(path: Path, picture: np.ndarray) -> Path:
    """Write `picture` to `path` as a PNG with cv2.imencode, which keeps its channels and bit
    depth. Gives `path`."""
    ok, data = cv2.imencode(".png", picture)
    assert ok, f"cv2.imencode couldn't encode {path}"
    path.write_bytes(data.tobytes())
    return path


def warped_path(tmp_path: Path) -> Path:
    """The Mask_Warp's output, colmap/mask_undistorted/mask.png in `tmp_path`, with its folder
    made and no file there yet."""
    folder = tmp_path / "colmap" / "mask_undistorted"
    folder.mkdir(parents=True)
    return folder / "mask.png"


def mask_refusal(path: Path) -> str:
    """The Readable_Error of read_warped_mask on `path` at WIDTH × HEIGHT, checked to name the
    file and to say to run the step again, and to hold no traceback (Req 8.5, 16.9)."""
    with pytest.raises(ExpectedFailure) as caught:
        read_warped_mask(path, WIDTH, HEIGHT)
    message = str(caught.value)
    assert str(path) in message
    assert "run the step again" in message.lower()
    assert "Traceback" not in message
    return message


@pytest.mark.parametrize("picture", [
    # In the faint ring only the first channel is below 255, which is enough to ignore the pixel.
    pytest.param(warped_picture((254, 255, 255), 3), id="three channels"),
    pytest.param(warped_picture(254), id="one channel"),
])
def test_read_warped_mask_gives_only_0_and_255_with_255_where_every_channel_is_255(
        tmp_path: Path, picture: np.ndarray) -> None:
    mask = read_warped_mask(write_png(warped_path(tmp_path), picture), WIDTH, HEIGHT)
    assert mask.dtype == np.uint8
    assert mask.shape == (HEIGHT, WIDTH)
    assert np.unique(mask).tolist() == [0, 255]
    assert np.array_equal(mask, expected_mask())


def test_read_warped_mask_refuses_a_missing_file_naming_it(tmp_path: Path) -> None:
    assert "can't be read" in mask_refusal(warped_path(tmp_path))


def test_read_warped_mask_refuses_bytes_that_arent_an_image_naming_the_file(
        tmp_path: Path) -> None:
    path = warped_path(tmp_path)
    path.write_bytes(b"not an image")
    assert "can't be decoded" in mask_refusal(path)


def test_read_warped_mask_refuses_a_16_bit_png_naming_it(tmp_path: Path) -> None:
    path = write_png(warped_path(tmp_path), np.full((HEIGHT, WIDTH, 3), 65535, dtype=np.uint16))
    message = mask_refusal(path)
    assert "16-bit" in message and "8-bit" in message


@pytest.mark.parametrize(("width", "height"), [
    pytest.param(WIDTH + 1, HEIGHT, id="too wide"),
    pytest.param(WIDTH, HEIGHT + 1, id="too tall"),
])
def test_read_warped_mask_refuses_a_png_of_another_size_naming_it_and_both_sizes(
        tmp_path: Path, width: int, height: int) -> None:
    path = write_png(warped_path(tmp_path), np.full((height, width, 3), 255, dtype=np.uint8))
    message = mask_refusal(path)
    assert f"{width} × {height}" in message
    assert f"{WIDTH} × {HEIGHT}" in message


# undistorted_mask (Req 8.4, F9)

# The Undistorted_Mask of each 2 × 3 picture below: 255 only at the two pixels that are 255 in
# every channel.
SMALL_MASK = np.array([[255, 0, 0], [0, 255, 0]], dtype=np.uint8)


@pytest.mark.parametrize("picture", [
    pytest.param([[255, 254, 0], [128, 255, 1]], id="2-D"),
    pytest.param([[[255], [254], [0]], [[128], [255], [1]]], id="one channel"),
    pytest.param([[[255, 255, 255], [254, 255, 255], [255, 255, 254]],
                  [[128, 128, 128], [255, 255, 255], [0, 0, 0]]], id="three channels"),
    pytest.param([[[255, 255, 255, 255], [255, 255, 255, 254], [254, 255, 255, 255]],
                  [[0, 0, 0, 255], [255, 255, 255, 255], [128, 128, 128, 128]]],
                 id="four channels"),
])
def test_undistorted_mask_is_one_uint8_channel_of_255_only_where_every_channel_is_255(
        picture: list[Any]) -> None:
    mask = undistorted_mask(np.array(picture, dtype=np.uint8))
    assert mask.dtype == np.uint8
    assert mask.shape == (2, 3)
    assert np.array_equal(mask, SMALL_MASK)


# Property 3: The Undistorted_Mask is 255 exactly where every channel is 255 (Req 8.4, 8.7)

# A channel value of the Mask_Warp's output, drawn mostly from 0, 1, 254 and 255, the two ends of
# the range and the values next to them.
CHANNEL_VALUES = st.sampled_from((0, 1, 254, 255)) | st.integers(0, 255)


@st.composite
def warped_pictures(draw: st.DrawFn) -> np.ndarray:
    """A uint8 picture of 1 × 1 to 16 × 16 pixels with 3 channels, like the Mask_Warp's output.

    arrays draws some channel values from CHANNEL_VALUES on their own and gives the others one
    background value. The background is white, like most of the Mask_Warp's output, or any
    channel value, or there's none and every value is drawn on its own. On a white background, a
    value drawn below 255 mostly leaves its pixel with one channel below 255, so those pixels are
    common, and so are white ones.
    """
    shape = (draw(st.integers(1, 16)), draw(st.integers(1, 16)), 3)
    background = draw(st.sampled_from((st.just(255), CHANNEL_VALUES, st.nothing())))
    return draw(arrays(np.uint8, shape, elements=CHANNEL_VALUES, fill=background))


@given(warped_pictures())
def test_the_undistorted_mask_is_255_exactly_where_every_channel_is_255(
        picture: np.ndarray) -> None:
    """Feature: cameras, Property 3: The Undistorted_Mask is 255 exactly where every channel is 255

    **Validates: Requirements 8.4, 8.7**
    """
    height, width, _ = picture.shape
    mask = undistorted_mask(picture)
    # One uint8 channel of the picture's height and width.
    assert mask.dtype == np.uint8
    assert mask.shape == (height, width)
    # The oracle, a plain loop over the pixels: each one is 0 or 255, and it's 255 exactly when
    # all three of its channels are 255.
    pixels, values = picture.tolist(), mask.tolist()
    for y in range(height):
        for x in range(width):
            where = f"pixel ({x}, {y}), whose channels are {pixels[y][x]}"
            assert values[y][x] in (0, 255), where
            assert (values[y][x] == 255) == all(channel == 255 for channel in pixels[y][x]), where
