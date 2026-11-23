"""The Model_Reader: our own reader for the Sparse_Model that the Camera_Solve_Step writes.

It reads COLMAP's little-endian binary layout, as COLMAP's docs and the formats in COLMAP's
read_write_model.py script describe it, with the standard library's struct module and numpy. It
runs no COLMAP program, imports no pycolmap and holds no code copied from COLMAP (Req 1.3).

    cameras.bin    <Q count. Per camera: <iiQQ (camera id, model id, width, height), then the
                   model's parameters as float64. PINHOLE has <4d: fx, fy, cx, cy.
    images.bin     <Q count. Per image: <idddddddi (image id, qw, qx, qy, qz, tx, ty, tz,
                   camera id), the name as UTF-8 bytes ending in NUL, <Q keypoint count, then 24
                   bytes per keypoint (<ddq: x, y, point id).
    points3D.bin   <Q count. Per point: <QdddBBBd (point id, x, y, z, r, g, b, error), <Q track
                   length, then 8 bytes per track entry (<ii: image id, keypoint index).

Each reader reads its file with Path.read_bytes() and walks it with struct.unpack_from through a
_Cursor, whose take() checks that the bytes are there before it unpacks. The keypoints, the
points' ids and errors, and the tracks are read past by moving the offset, once the cursor has
checked that their bytes are there, and none of their values is kept (Req 1.5). Each count is
checked against its smallest record before the loop, 73 bytes per image and 51 per point, so a
huge count is refused at once instead of looping or allocating. A record cut short, bytes after
the last record, a count the file can't hold and a name with no NUL are each refused as "<file>
is cut short or isn't a COLMAP binary model (<what>)" (Req 1.8).

cameras.bin must hold one camera, the Shared_Camera, of the PINHOLE model (Req 1.7). Values are
checked as they're read: width, height, fx and fy above 0, every camera parameter, quaternion,
translation and point position finite, and no quaternion of four zeros, which is what a length
of 0 means here (Req 1.9). A quaternion of tiny or huge values passes, because scene.py's
world_to_camera divides it by its largest absolute value before it scales it to length 1. An
image's name must be UTF-8 and fullmatch IMAGE_NAME, so it can't lead outside images/, and that
refusal gives only the image's id (Req 1.11). images.bin must hold an image, every image must be
the Shared_Camera's, and no two images may share a name (Req 1.10). Names are shown in messages
with repr(), so a control character can't garble the terminal.

read_model reads the three files by name and nothing else in sparse/, so the rigs.bin and
frames.bin that COLMAP 3.12 and later write beside them are ignored (Req 1.6). It keeps the
poses, the camera and the points as the files give them, in the Levelled_Frame, with no change
to the world frame (Req 1.20).

From splatpipe, this module imports only splatpipe.errors.
"""

from __future__ import annotations

import math
import re
import struct
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn

import numpy as np

from splatpipe.errors import ExpectedFailure

PINHOLE = 1  # Fixed_Constants: COLMAP's PINHOLE model id and its parameter count
PINHOLE_PARAMS = 4
IMAGE_NAME = re.compile(r"[^/\\:\x00]+\.jpg")  # Req 1.11, checked with fullmatch

# COLMAP's binary layout, little-endian with no padding (Fixed_Constants). In a format, x is a
# byte read past: _POINT reads past the point's id and its error, keeping neither.
_COUNT = "<Q"
_CAMERA = "<iiQQ"  # camera id, model id, width, height
_PARAMS = f"<{PINHOLE_PARAMS}d"  # fx, fy, cx, cy
_IMAGE = "<idddddddi"  # image id, qw, qx, qy, qz, tx, ty, tz, camera id
_KEYPOINT_BYTES = 24  # <ddq: x, y, point id
_POINT = "<8xdddBBB8xQ"  # <QdddBBBd (point id, x, y, z, r, g, b, error), then the track length
_TRACK_ENTRY_BYTES = 8  # <ii: image id, keypoint index
_SMALLEST_IMAGE = struct.calcsize(_IMAGE) + 1 + struct.calcsize(_COUNT)  # 73: an empty name's NUL
_SMALLEST_POINT = struct.calcsize(_POINT)  # 51: a track with no entry
_PARAM_NAMES = ("fx", "fy", "cx", "cy")
_AGAIN = "Run the camera solve step on this job again."


@dataclass(frozen=True)
class Camera:
    """The Shared_Camera: its id, its size in pixels and its PINHOLE parameters."""

    camera_id: int
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass(frozen=True)
class ModelImage:
    """One image of images.bin: its id, its Image_Pose, its camera's id and its name."""

    image_id: int
    quaternion: np.ndarray  # (4,) float64, w x y z, as stored (not yet normalised)
    translation: np.ndarray  # (3,) float64
    camera_id: int
    name: str


@dataclass(frozen=True)
class SparseModel:
    """The Sparse_Model, in the Levelled_Frame as the camera solve step leaves it."""

    camera: Camera  # the Shared_Camera
    images: tuple[ModelImage, ...]  # in images.bin's order
    points: np.ndarray  # (P, 3) float64
    colours: np.ndarray  # (P, 3) uint8, RGB


def read_model(sparse: Path) -> SparseModel:
    """The Sparse_Model in `sparse`, read from cameras.bin, images.bin and points3D.bin by name.

    Nothing else in `sparse` is read, so rigs.bin and frames.bin are ignored (Req 1.6), and the
    poses, the camera and the points are kept as the files give them (Req 1.20).
    """
    camera = read_cameras(sparse / "cameras.bin")
    images = read_images(sparse / "images.bin", camera.camera_id)
    points, colours = read_points(sparse / "points3D.bin")
    return SparseModel(camera, images, points, colours)


def read_cameras(path: Path) -> Camera:
    """The Shared_Camera, the one PINHOLE camera that cameras.bin must hold (Req 1.4, 1.7, 1.9)."""
    cursor = _Cursor(path)
    (count,) = cursor.take(_COUNT, "the camera count")
    if count != 1:
        raise ExpectedFailure(f"{path} holds {count} cameras, but the camera solve step must "
                              f"write one shared PINHOLE camera. {_AGAIN}")
    camera_id, model_id, width, height = cursor.take(_CAMERA, "the camera's record")
    if model_id != PINHOLE:
        raise ExpectedFailure(f"In {path}, camera {camera_id} has the COLMAP camera model "
                              f"{model_id}, but the train step needs the PINHOLE model "
                              f"({PINHOLE}), so the frames must be undistorted. {_AGAIN}")
    params = cursor.take(_PARAMS, "the camera's parameters")
    _check_camera(path, camera_id, width, height, params)
    cursor.end()
    return Camera(camera_id, width, height, *params)


def read_images(path: Path, camera_id: int) -> tuple[ModelImage, ...]:
    """The images of images.bin in the file's order, each of the Shared_Camera `camera_id`.

    It keeps each image's id, Image_Pose, camera id and name and reads past its keypoints
    (Req 1.4, 1.5). It refuses a file with no image, an image of another camera and two images
    with one name (Req 1.10), a name that isn't UTF-8 or doesn't fullmatch IMAGE_NAME (Req 1.11),
    and a pose that isn't finite or a quaternion of four zeros (Req 1.9).
    """
    cursor = _Cursor(path)
    (count,) = cursor.take(_COUNT, "the image count")
    if count == 0:
        raise ExpectedFailure(f"{path} holds no image, so the train step has nothing to train "
                              f"on. {_AGAIN}")
    cursor.check_count(count, _SMALLEST_IMAGE, "image")
    images: list[ModelImage] = []
    first_with: dict[str, int] = {}  # each name, with the id of the first image that has it
    for place in range(1, count + 1):
        image = _read_image(cursor, f"image {place} of {count}", camera_id)
        if image.name in first_with:
            raise ExpectedFailure(f"In {path}, images {first_with[image.name]} and "
                                  f"{image.image_id} are both named {image.name!r}, but each "
                                  f"image needs a name of its own. {_AGAIN}")
        first_with[image.name] = image.image_id
        images.append(image)
    cursor.end()
    return tuple(images)


def read_points(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """The Sparse_Points of points3D.bin in the file's order: their positions, (P, 3) float64,
    and their RGB colours, (P, 3) uint8.

    Each point's id, error and track are read past (Req 1.5). A position that isn't finite is
    refused, naming the point's place in the file (Req 1.9).
    """
    cursor = _Cursor(path)
    (count,) = cursor.take(_COUNT, "the point count")
    cursor.check_count(count, _SMALLEST_POINT, "point")
    positions: list[tuple[float, float, float]] = []
    colours: list[tuple[int, int, int]] = []
    for place in range(1, count + 1):
        where = f"point {place} of {count}"
        x, y, z, red, green, blue, track = cursor.take(_POINT, where)
        _check_finite(path, where, "position", (x, y, z))
        cursor.skip(track, _TRACK_ENTRY_BYTES, f"the track of {where}")
        positions.append((x, y, z))
        colours.append((red, green, blue))
    cursor.end()
    return (np.array(positions, dtype=np.float64).reshape(-1, 3),
            np.array(colours, dtype=np.uint8).reshape(-1, 3))


class _Cursor:
    """A model file's bytes and the offset of the next byte to read, which only moves forward.

    Every read checks that its bytes are there first, so a file cut short, a count it can't hold
    or a name with no NUL is refused with the Req 1.8 message, never a struct.error.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.data = path.read_bytes()
        self.offset = 0

    def left(self) -> int:
        """How many bytes there are after the offset."""
        return len(self.data) - self.offset

    def take(self, fmt: str, what: str) -> tuple[Any, ...]:
        """The values of struct format `fmt` at the offset, which moves past them. `what` names
        the record for the message when the file ends inside it."""
        size = struct.calcsize(fmt)
        self._need(size, what)
        values = struct.unpack_from(fmt, self.data, self.offset)
        self.offset += size
        return values

    def skip(self, count: int, size: int, what: str) -> None:
        """Move past `count` entries of `size` bytes each, once they're there, keeping no value."""
        self._need(count * size, what)
        self.offset += count * size

    def name(self, what: str) -> bytes:
        """The bytes from the offset up to the next NUL. The offset moves past the NUL."""
        end = self.data.find(b"\x00", self.offset)
        if end < 0:
            raise self.broken(f"the name of {what} has no NUL at its end")
        name = self.data[self.offset:end]
        self.offset = end + 1
        return name

    def check_count(self, count: int, smallest: int, kind: str) -> None:
        """Refuse a count of records, each at least `smallest` bytes, that the bytes left can't
        hold. `kind` names the records in the message, like "image"."""
        if count * smallest > self.left():
            raise self.broken(f"its {kind} count is {count}, which needs at least "
                              f"{_bytes(count * smallest)}, but the file has only "
                              f"{_bytes(self.left())} after the count")

    def end(self) -> None:
        """Refuse bytes after the last record."""
        if self.left():
            raise self.broken(f"it holds {_bytes(self.left())} after its last record")

    def broken(self, what: str) -> ExpectedFailure:
        """The Req 1.8 failure for this file, saying `what` was wrong."""
        return ExpectedFailure(f"{self.path} is cut short or isn't a COLMAP binary model "
                               f"({what}). {_AGAIN}")

    def _need(self, size: int, what: str) -> None:
        """Refuse a read of `size` bytes that runs past the end of the file."""
        if size > self.left():
            raise self.broken(f"it ends inside {what}")


def _read_image(cursor: _Cursor, where: str, camera_id: int) -> ModelImage:
    """The image record at the cursor, checked, with its keypoints read past. `where` is its
    place in the file, like "image 2 of 5", for the Req 1.8 messages."""
    image_id, *pose, image_camera = cursor.take(_IMAGE, where)
    name = _image_name(cursor, where, image_id)
    who = f"image {image_id} ({name!r})"
    quaternion, translation = pose[:4], pose[4:]
    _check_finite(cursor.path, who, "quaternion", quaternion)
    _check_finite(cursor.path, who, "translation", translation)
    if all(value == 0 for value in quaternion):
        _refuse_value(cursor.path, who, f"the quaternion {tuple(quaternion)}",
                      "a quaternion of four zeros has a length of 0, so it gives no rotation")
    if image_camera != camera_id:
        raise ExpectedFailure(f"In {cursor.path}, {who} has the camera id {image_camera}, but "
                              f"every image must be of the shared camera, camera {camera_id} in "
                              f"cameras.bin. {_AGAIN}")
    (keypoints,) = cursor.take(_COUNT, f"the keypoint count of {where}")
    cursor.skip(keypoints, _KEYPOINT_BYTES, f"the keypoints of {where}")
    return ModelImage(image_id, np.array(quaternion, dtype=np.float64),
                      np.array(translation, dtype=np.float64), image_camera, name)


def _image_name(cursor: _Cursor, where: str, image_id: int) -> str:
    """The image's name, read up to its NUL. A name that isn't UTF-8 or doesn't fullmatch
    IMAGE_NAME is refused with the image's id alone, never the raw name (Req 1.11)."""
    raw = cursor.name(where)
    try:
        name = raw.decode("utf-8")
    except UnicodeDecodeError:
        name = ""  # no name fullmatches "", so it's refused below
    if not IMAGE_NAME.fullmatch(name):
        raise ExpectedFailure(f"In {cursor.path}, image {image_id} has a name that isn't UTF-8 "
                              "text of one or more characters and then .jpg, with no /, \\, : "
                              f"or NUL in it, so it can't name a file in images/. {_AGAIN}")
    return name


def _check_camera(path: Path, camera_id: int, width: int, height: int,
                  params: tuple[float, ...]) -> None:
    """Refuse a camera parameter that isn't finite, and a width, height, fx or fy that isn't
    above 0 (Req 1.9)."""
    who = f"camera {camera_id}"
    for label, value in zip(_PARAM_NAMES, params):
        if not math.isfinite(value):
            _refuse_value(path, who, f"{label} = {value!r}", "it must be finite")
    fx, fy = params[0], params[1]
    for label, value in (("width", width), ("height", height), ("fx", fx), ("fy", fy)):
        if value <= 0:
            _refuse_value(path, who, f"{label} = {value!r}", "it must be above 0")


def _check_finite(path: Path, who: str, label: str, values: Sequence[float]) -> None:
    """Refuse `values` unless every one of them is finite (Req 1.9)."""
    if not all(math.isfinite(value) for value in values):
        _refuse_value(path, who, f"the {label} {tuple(values)}",
                      "each of its values must be finite")


def _refuse_value(path: Path, who: str, has: str, but: str) -> NoReturn:
    """Refuse a value that Req 1.9 rules out, naming the file, the camera, image or point, and the
    value."""
    raise ExpectedFailure(f"In {path}, {who} has {has}, but {but}. {_AGAIN}")


def _bytes(count: int) -> str:
    """A number of bytes in words, like "1 byte" or "73 bytes"."""
    return f"{count} byte" if count == 1 else f"{count} bytes"
