"""The Undistortion's output, checked, and the Undistorted_Mask (Req 8.2 to 8.5).

COLMAP's image_undistorter warps the Final_Model's Colour_Picks to one PINHOLE camera and writes
COLMAP's own layout into colmap/undistorted/: the Undistorted_Model in sparse/ and the warped
picks in images/ (Req 8.1). check_undistorted checks that output before anything else reads it
(Req 8.2): exactly one camera, a PINHOLE one, that every image uses, image names that are exactly
the Final_Model's Placed_Frames, and each image's file in images/. The first check that fails is
an ExpectedFailure that says what the Undistortion got wrong and to run the step again. COLMAP
keeps a camera's own model when every distortion parameter came out within 1e-8 of 0, which is
what bundle adjustment leaves when mapper_ba_refine_extra_params or
global_mapper_ba_refine_extra_params is false (F9). So the refusal of a camera that isn't PINHOLE
also says to set both to true (C8).

With mask_used, COLMAP's image_undistorter_standalone warps the Mask the same way, the Mask_Warp
(Req 8.3). It reads one line per image from its input file: a name, a model name, a width, a
height and the parameters (F8). mask_camera_line gives that line for the Final_Model's camera, as
the Final_Model holds it before the Undistortion. The line's name is MASK, the Mask's file, which
the Mask_Warp reads from work/frames/ and writes under the same name into
colmap/mask_undistorted/. Each parameter is written with repr(float), the shortest text that
reads back as the same double. With the same camera and the same undistortion settings, the
Mask_Warp gives the Undistorted_Camera's size (F7).

The Mask_Warp samples the Mask bilinearly into a three-channel PNG, so the edges of its black
regions come out grey (D10, F8). read_warped_mask reads that file with Path.read_bytes and
decodes it with cv2.imdecode and IMREAD_UNCHANGED, so its channels and bit depth stay as they
are. A file that can't be read or decoded, that isn't 8-bit, or that isn't the
Undistorted_Camera's width and height is the Req 8.5 failure, which names the file, says what's
wrong and says to run the step again. undistorted_mask is pure. It gives the Undistorted_Mask,
one 8-bit grey channel that is 255 (use) where every channel is 255 and 0 (ignore) everywhere
else (Req 8.4, 8.7). It also takes a one-channel image, like the unwarped copy of the Mask that
COLMAP makes for a camera with no distortion (F9).

cv2 only decodes the warped mask's PNG here, and never reads or writes video.

From splatpipe, this module imports only splatpipe.errors, splatpipe.frames.files and the step's
model.
"""

from __future__ import annotations

import os
from collections.abc import Collection
from pathlib import Path

import cv2
import numpy as np

from splatpipe.cameras.model import CAMERA_MODELS, Camera, Model
from splatpipe.errors import ExpectedFailure
from splatpipe.frames.files import MASK

_PINHOLE = "PINHOLE"  # Fixed_Constant: the Undistorted_Camera's model, and the Levelled_Model's
# The Undistorted_Mask's two values, Fixed_Constants: black tells the training loss to ignore a
# pixel, and white tells it to use it (D10).
_IGNORE = np.uint8(0)
_USE = np.uint8(255)

_UNDISTORTER = "COLMAP's image_undistorter"  # the command whose output the Req 8.2 checks read
# The C8 advice, which ends the refusal of an Undistorted_Camera that isn't PINHOLE (F9).
_REFINE_DISTORTION = (
    "COLMAP keeps a camera's own model when the camera's distortion came out as 0, which "
    "happens when mapper_ba_refine_extra_params or global_mapper_ba_refine_extra_params is "
    "false. Set both to true in [cameras] in config.toml, then run the step again.")


def check_undistorted(model: Model, placed: Collection[str], images: Path) -> Camera:
    """The Undistorted_Camera of `model`, the Undistorted_Model, once it passes the Req 8.2
    checks.

    `placed` holds the names of the Final_Model's Placed_Frames, and `images` is the
    Undistortion's images/ folder. In this order: the model holds exactly one camera, the camera
    is PINHOLE, every image uses it, the image names are exactly `placed`, and images/<name> is a
    file for each image. The first check that fails is the Req 8.2 failure, and a camera that
    isn't PINHOLE also gets the C8 advice.
    """
    if len(model.cameras) != 1:
        raise _wrong(f"wrote {len(model.cameras)} cameras into the undistorted model, but it "
                     "should write exactly 1, the camera that every pick shares")
    camera = model.cameras[0]
    name, _ = CAMERA_MODELS[camera.model_id]
    if name != _PINHOLE:
        raise ExpectedFailure(f"{_UNDISTORTER} left the camera as {name}, but it should turn it "
                              f"into a {_PINHOLE} camera with no distortion. "
                              f"{_REFINE_DISTORTION}")
    for image in model.images:
        if image.camera_id != camera.camera_id:
            raise _wrong(f"gave the image {image.name} the camera id {image.camera_id}, but the "
                         f"undistorted model's only camera has the id {camera.camera_id}")
    _check_names(model, frozenset(placed))
    for image in model.images:
        path = images / image.name
        if not os.path.isfile(path):
            raise _wrong(f"didn't write the undistorted image {path}")
    return camera


def mask_camera_line(camera: Camera) -> str:
    """The Mask_Warp's input line for `camera`, the Final_Model's camera as the Final_Model holds
    it before the Undistortion (Req 8.3, F8): "mask.png <MODEL> <width> <height> <p1> … <pN>",
    then a newline.

    The name is MASK, the Mask's file in work/frames/, the model is the camera's model name, and
    each parameter is written with repr(float), the shortest text that reads back as the same
    double.
    """
    name, _ = CAMERA_MODELS[camera.model_id]
    params = (repr(float(param)) for param in camera.params)
    return " ".join((MASK, name, str(camera.width), str(camera.height), *params)) + "\n"


def undistorted_mask(image: np.ndarray) -> np.ndarray:
    """The Undistorted_Mask of the Mask_Warp's `image`, a uint8 image of 2 dimensions, or of 3
    with 1, 3 or 4 channels (Req 8.4). Pure.

    It's one uint8 channel of the image's height and width: 255, use, where every channel is 255,
    and 0, ignore, everywhere else, so the grey edges that bilinear sampling leaves are ignored
    (D10). A one-channel image is like the unwarped copy of the Mask that COLMAP makes for a
    camera with no distortion (F9).
    """
    use = image == _USE
    if image.ndim == 3:
        use = use.all(axis=2)
    return np.where(use, _USE, _IGNORE)


def read_warped_mask(path: Path, width: int, height: int) -> np.ndarray:
    """The Undistorted_Mask of the Mask_Warp's output `path`, which must be an 8-bit image of
    `width` × `height` pixels, the Undistorted_Camera's size (Req 8.4, 8.5).

    The file is read with Path.read_bytes and decoded with cv2.imdecode and IMREAD_UNCHANGED, so
    its channels and bit depth stay as they are. A file that can't be read or decoded, that isn't
    8-bit, or that is another size is the Req 8.5 failure, which names the file and what's wrong.
    """
    try:
        data = path.read_bytes()
    except OSError as error:
        raise _bad_mask(path, f"can't be read ({error.strerror or error})") from error
    image = _decode(data)
    if image is None:
        raise _bad_mask(path, "can't be decoded as an image")
    if image.dtype != np.uint8:
        raise _bad_mask(path, f"is a {image.dtype.itemsize * 8}-bit image, but it should be an "
                              "8-bit one")
    found_height, found_width = image.shape[:2]
    if (found_width, found_height) != (width, height):
        raise _bad_mask(path, f"is {found_width} × {found_height} pixels, but it should be "
                              f"{width} × {height} pixels, the undistorted camera's size")
    return undistorted_mask(image)


def _check_names(model: Model, placed: frozenset[str]) -> None:
    """Refuse the first image of `model` whose name isn't in `placed`, then the first name of
    `placed` in sorted order, which is Pick_Number order, that no image of `model` has
    (Req 8.2)."""
    for image in model.images:
        if image.name not in placed:
            raise _wrong(f"wrote the image {image.name} into the undistorted model, but it isn't "
                         "one of the picks that the final model placed")
    missing = sorted(placed - {image.name for image in model.images})
    if missing:
        raise _wrong(f"left the placed pick {missing[0]} out of the undistorted model")


def _decode(data: bytes) -> np.ndarray | None:
    """The image in `data`, decoded with its channels and bit depth, or None when OpenCV can't
    decode it. For some bytes, like none at all, OpenCV raises cv2.error instead of giving
    None."""
    try:
        return cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    except cv2.error:
        return None


def _wrong(what: str) -> ExpectedFailure:
    """The Req 8.2 failure: what COLMAP's image_undistorter got wrong, then to run the step
    again."""
    return ExpectedFailure(f"{_UNDISTORTER} {what}. Run the step again.")


def _bad_mask(path: Path, what: str) -> ExpectedFailure:
    """The Req 8.5 failure: the Mask_Warp's output `path`, what's wrong with it, and to run the
    step again."""
    return ExpectedFailure(f"The warped mask {path} {what}. Run the step again.")
