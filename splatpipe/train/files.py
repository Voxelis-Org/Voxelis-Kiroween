"""The Train_Step's files: the Camera_Solve_Output it reads, and the Trained_Splat and the
Train_Record it writes.

The names are Fixed_Constants, because they're the contract with the Camera_Solve_Step and the
Clean_Step (Req 1.1):

    work/cameras/images/<name>.jpg   the Training_Images, <name>.jpg being an image's name in
                                     images.bin
    work/cameras/masks/<stem>.png    an image's Mask_File, when it has one, <stem> being the
                                     image's name without .jpg. Black means ignore.
    work/cameras/sparse/             the Sparse_Model: cameras.bin, images.bin and points3D.bin
    work/train/train.ply             the Trained_Splat while it's written and checked
    out/train.ply                    the Trained_Splat once the PLY_Check passes
    work/train/train.json            the Train_Record, written last

check_inputs (Req 1.2) checks that images/ is a folder, then that each model file in sparse/ is a
file, and refuses the first one missing. It reads nothing, so the step runs it before the GPU
checks (A9).

read_views goes through the images in images.bin's order and opens only the files their names
lead to, so every other file in images/ and masks/ is ignored (Req 1.17). Each Training_Image is
read with Path.read_bytes() and decoded with cv2.imdecode and IMREAD_COLOR |
IMREAD_IGNORE_ORIENTATION, so the pixel grid stays the one the camera solve solved, and its size
must be the Shared_Camera's (Req 1.12, 1.13). It's scaled down to the Training_Size with
INTER_AREA when it's larger, then turned from cv2's BGR to RGB (Req 3.2). When masks/<stem>.png
is a file, it's read the same way with IMREAD_GRAYSCALE | IMREAD_IGNORE_ORIENTATION, its size must
be its image's, and keep_mask turns it into the image's keep mask at the Training_Size, which must
keep at least one pixel (Req 1.14, 1.16, 3.4). An image without one gets None, so it trains on
every pixel (Req 1.15). Each refusal is an ExpectedFailure that names the file and the reason,
with both sizes when they differ. Bytes that cv2.imdecode can't decode are refused the same way
whether it gives None or raises cv2.error, as it does for an empty file and for a header beyond
OpenCV's size limit. Names from images.bin are shown with repr(). cv2 only decodes still images
here and never reads or writes video.

move_into_out moves the checked Trained_Splat into out/ with one os.replace, so out/ never holds
a half-written file (Req 8.8). write_record writes the Train_Record with
splatpipe.jsonio.write_json. Both turn an OSError into an ExpectedFailure that names the file and
the reason, like "No space left on device" (Req 8.10). train_record is pure and gives only plain
JSON values (Req 10.7).

From splatpipe, this module imports only splatpipe.errors, splatpipe.jsonio, scene.py for
keep_mask and ImagePixels, and colmap.py and scene.py for their types.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from splatpipe.errors import ExpectedFailure
from splatpipe.jsonio import write_json
from splatpipe.train.colmap import Camera, SparseModel
from splatpipe.train.scene import ImagePixels, Scene, keep_mask

# Fixed_Constants: the names of the Camera_Solve_Output, the Trained_Splat and the Train_Record.
CAMERAS = "cameras"  # the Camera_Solve_Output's folder in work/ (Req 1.1)
IMAGES = "images"  # the Training_Images, in work/cameras/
MASKS = "masks"  # the Mask_Files, in work/cameras/
SPARSE = "sparse"  # the Sparse_Model, in work/cameras/
MODEL_FILES = ("cameras.bin", "images.bin", "points3D.bin")  # the Sparse_Model's files, in sparse/
TRAINED_SPLAT = "train.ply"  # in work/train/ while it's written and checked, then in out/
RECORD = "train.json"  # the Train_Record, in work/train/

_COLOUR = cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION  # a Training_Image, decoded as BGR
_GREY = cv2.IMREAD_GRAYSCALE | cv2.IMREAD_IGNORE_ORIENTATION  # a Mask_File
_AGAIN = "Run the camera solve step on this job again."


def check_inputs(cameras: Path) -> None:
    """Refuse a Camera_Solve_Output in `cameras`, work/cameras/, without images/ or a model file.

    images/ must be a folder and each of MODEL_FILES in sparse/ a file, checked in that order.
    The first one missing is refused with the Req 1.2 message, which names it. Nothing is read.
    """
    images = cameras / IMAGES
    if not images.is_dir():
        raise _missing(images)
    for name in MODEL_FILES:
        path = cameras / SPARSE / name
        if not path.is_file():
            raise _missing(path)


def read_views(cameras: Path, model: SparseModel, width: int, height: int,
               threshold: int) -> list[ImagePixels]:
    """Each Training_Image's ImagePixels at the Training_Size `width` × `height`, in images.bin's
    order (Req 1.12–1.17, 3.2, 3.4).

    `cameras` is work/cameras/, `model` its Sparse_Model and `threshold` the mask_threshold. Only
    images/<name> and masks/<stem>.png of each image's name are opened.
    """
    return [_read_view(cameras, image.name, model.camera, width, height, threshold)
            for image in model.images]


def move_into_out(written: Path, target: Path) -> None:
    """Move the checked Trained_Splat `written`, work/train/train.ply, to `target`, out/train.ply,
    with one os.replace on the same file system (Req 8.8).

    An OSError becomes an ExpectedFailure that names both paths and the reason (Req 8.10).
    """
    try:
        os.replace(written, target)
    except OSError as error:
        raise ExpectedFailure(f"The trained splat {written} couldn't be moved to {target} "
                              f"({_reason(error)}). Free some space on the disk, or check that "
                              f"you can write to {target.parent}, then run the step again."
                              ) from error


def train_record(*, scene: Scene, means_lr_start: float, width_ratio: float, height_ratio: float,
                 a: np.ndarray, b: np.ndarray, cap_deadline: datetime | None,
                 stop: datetime | None) -> dict[str, Any]:
    """The Train_Record, as plain JSON values (Req 10.7).

    `a` and `b` are the Mean_Exposure's Ā, (3, 3), and b̄, (3,), given as 3 rows of 4 floats,
    [Ā | b̄]. The Cap_Deadline and the Stop_Time are given as UTC text from isoformat(), or None
    when no Cap_Deadline was set.
    """
    exposure = np.column_stack((a, b)).astype(np.float64)
    return {
        "near_plane": float(scene.near_plane),
        "median_scene_depth": float(scene.median_depth),
        "scene_extent": float(scene.extent),
        "means_lr_start": float(means_lr_start),
        "width_ratio": float(width_ratio),
        "height_ratio": float(height_ratio),
        "mean_exposure": [[float(value) for value in row] for row in exposure],
        "cap_deadline": _utc_text(cap_deadline),
        "stop_time": _utc_text(stop),
    }


def write_record(path: Path, record: dict[str, Any]) -> None:
    """Write the Train_Record to `path`, work/train/train.json, with write_json, which checks
    every value first. An OSError becomes an ExpectedFailure that names the file and the reason
    (Req 8.10)."""
    try:
        write_json(path, record)
    except OSError as error:
        raise ExpectedFailure(f"The file {path} couldn't be written ({_reason(error)}). Free some "
                              f"space on the disk, or check that you can write to {path.parent}, "
                              "then run the step again.") from error


def _missing(path: Path) -> ExpectedFailure:
    """The Req 1.2 failure for a path of the Camera_Solve_Output that isn't there."""
    return ExpectedFailure(f"The train step needs {path}, which the camera solve step writes, but "
                           "it isn't there. Run the camera solve step on this job first.")


def _read_view(cameras: Path, name: str, camera: Camera, width: int, height: int,
               threshold: int) -> ImagePixels:
    """The image `name`'s RGB pixels at the Training_Size, and its keep mask, or None when it has
    no Mask_File."""
    image = _training_image(cameras / IMAGES, name, camera)
    pixels = cv2.cvtColor(_scaled_down(image, width, height), cv2.COLOR_BGR2RGB)
    keep = _keep(cameras / MASKS, name, _size(image), width, height, threshold)
    return ImagePixels(pixels, keep)


def _training_image(folder: Path, name: str, camera: Camera) -> np.ndarray:
    """The Training_Image `name` in `folder`, images/, as cv2 decodes it, in BGR.

    It must be a file (Req 1.12), a colour image cv2 can decode, and the Shared_Camera's size
    (Req 1.13).
    """
    path = folder / name
    if not path.is_file():
        raise ExpectedFailure(f"images.bin names the image {name!r}, but {folder} has no file of "
                              f"that name. {_AGAIN}")
    shown = f"The training image {name!r} in {folder}"
    image = _decoded(path, _COLOUR, shown, "a colour JPEG")
    found, need = _size(image), (camera.width, camera.height)
    if found != need:
        raise ExpectedFailure(f"{shown} is {_pixels(found)}, but the shared camera in cameras.bin "
                              f"is {_pixels(need)}, and every training image must be the "
                              f"camera's size. {_AGAIN}")
    return image


def _scaled_down(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """The image at the Training_Size `width` × `height`, scaled down with INTER_AREA when it's
    larger (Req 3.2). The Training_Size is never larger than the image."""
    if _size(image) == (width, height):
        return image
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def _keep(folder: Path, name: str, size: tuple[int, int], width: int, height: int,
          threshold: int) -> np.ndarray | None:
    """The keep mask at the Training_Size of the image `name`, whose (width, height) is `size`,
    from its Mask_File in `folder`, masks/, or None when it has none (Req 1.14, 1.15).

    The Mask_File must be a grey image cv2 can decode, of its image's size, that keeps at least one
    pixel once keep_mask has scaled it (Req 1.16, 3.4).
    """
    mask_name = f"{name.removesuffix('.jpg')}.png"
    path = folder / mask_name
    if not path.is_file():
        return None  # no Mask_File, so every pixel trains (Req 1.15)
    shown = f"The mask {mask_name!r} in {folder}"
    grey = _decoded(path, _GREY, shown, "a grey image")
    if _size(grey) != size:
        raise ExpectedFailure(f"{shown} is {_pixels(_size(grey))}, but its image {name!r} is "
                              f"{_pixels(size)}, and a mask must be the size of its image. "
                              f"{_AGAIN}")
    keep = keep_mask(grey, width, height, threshold)
    if not keep.any():
        raise ExpectedFailure(f"{shown} masks every pixel of its image {name!r}: scaled to the "
                              "training size, every pixel is below mask_threshold in [train], "
                              f"{threshold}, so the image has nothing to train on. {_AGAIN}")
    return keep


def _decoded(path: Path, flags: int, shown: str, kind: str) -> np.ndarray:
    """The image in the file at `path`, read with Path.read_bytes() and decoded by cv2.imdecode
    with `flags`. `shown` names the file in the messages, and `kind` says what it must be.

    An OSError is refused with its reason. A decode that gives None is refused, and so is one
    that raises cv2.error, as cv2.imdecode does for no bytes and for a header beyond OpenCV's
    size limit, so every file cv2 can't decode gets the same message.
    """
    try:
        data = path.read_bytes()
    except OSError as error:
        raise ExpectedFailure(f"{shown} couldn't be read ({_reason(error)}). Check that you can "
                              "read it, or run the camera solve step on this job again.") from error
    try:
        image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), flags)
    except cv2.error:
        image = None
    if image is None:
        raise ExpectedFailure(f"{shown} can't be read as {kind}: OpenCV couldn't decode it. "
                              f"{_AGAIN}")
    return image


def _size(image: np.ndarray) -> tuple[int, int]:
    """An image's (width, height) in pixels."""
    return image.shape[1], image.shape[0]


def _pixels(size: tuple[int, int]) -> str:
    """A (width, height) in words, like "8 x 6 pixels"."""
    return f"{size[0]} x {size[1]} pixels"


def _reason(error: OSError) -> str:
    """The reason the system gave for an OSError, like "No space left on device"."""
    return error.strerror or str(error)


def _utc_text(moment: datetime | None) -> str | None:
    """A time as ISO 8601 text in UTC, or None for no time."""
    return None if moment is None else moment.astimezone(timezone.utc).isoformat()
