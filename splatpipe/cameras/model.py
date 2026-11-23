"""The Model_Files, cameras.bin, images.bin and points3D.bin, in COLMAP 4.2.1's binary layout.

They're the contract the Training_Step's reader parses (D2, O10), laid out as the design's "The
Model_Files" gives them from COLMAP 4.2.1's source (F1 to F5): every number little-endian, records
one after another with no padding, and no header beyond each file's count. This is our own code,
written fresh from those layout tables with struct and numpy, with nothing copied from COLMAP's
scripts or another repo (Req 10.3).

The fixed part of each record goes through struct with an explicit little-endian format, and
struct copies a double's 8 bytes as they are. Each image's 2D points and each point's track go
through np.frombuffer and tobytes() with POINT2D and TRACK_ELEMENT. So every bit of every number
is kept, NaN payloads included.

The decoders walk a memoryview with an offset. A file that ends inside a record or inside its
count, a camera model id that isn't in CAMERA_MODELS, a name with no 0 byte before the end or that
isn't UTF-8, an id given twice, or bytes left after the last record is refused with an
ExpectedFailure that names the file and what's wrong, and says to check that COLMAP 4.2.1 is
installed and to run the step again (Req 10.4).

The encoders write the cameras and the points in increasing id order, as COLMAP's WriteBinary
does, and the images in the Model's order, the order COLMAP registered them in (F3). An id given
twice, a name with a 0 byte in it, a model id that isn't in CAMERA_MODELS or a parameter count
that doesn't match the model is a bug in our code, so they raise ValueError.

read_model reads the three files and ignores any others, rigs.bin and frames.bin included (Req
10.1, F4). read_images reads one images.bin, which holds exactly the images its Model registers
(F3). A missing file is the Req 10.4 failure. write_model makes the folder and writes exactly the
three files, and an OSError is the Req 11.12 failure, naming the file and the reason.

From splatpipe, this module imports only splatpipe.errors.
"""

from __future__ import annotations

import struct
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from splatpipe.errors import ExpectedFailure

# The camera models the Model_Files may hold, by COLMAP's model id: the name and the number of
# parameters (F2). PINHOLE takes fx, fy, cx, cy. OPENCV takes fx, fy, cx, cy, k1, k2, p1, p2, and
# OPENCV_FISHEYE fx, fy, cx, cy, k1, k2, k3, k4.
CAMERA_MODELS: Mapping[int, tuple[str, int]] = {1: ("PINHOLE", 4), 4: ("OPENCV", 8),
                                                5: ("OPENCV_FISHEYE", 8)}
NO_POINT3D = 2**64 - 1  # a 2D point's point3d_id when it has no 3D point, kInvalidPoint3DId (F1)
NO_ERROR = -1.0  # a point's error when it has none (F15)
# An image's 2D point, 24 bytes: its position in pixels, and its 3D point's id or NO_POINT3D.
POINT2D = np.dtype([("x", "<f8"), ("y", "<f8"), ("point3d_id", "<u8")])
# An element of a point's track, 8 bytes: an image's id, and the 2D point's place in its list.
TRACK_ELEMENT = np.dtype([("image_id", "<u4"), ("point2d_idx", "<u4")])
FILES = ("cameras.bin", "images.bin", "points3D.bin")  # the Model_Files

# The fixed parts of the records, little-endian with no padding.
_COUNT = struct.Struct("<Q")  # each file's count, and each image's count of 2D points
_CAMERA = struct.Struct("<IiQQ")  # camera_id, model_id, width, height: 24 bytes
_IMAGE = struct.Struct("<I4d3dI")  # image_id, qw, qx, qy, qz, tx, ty, tz, camera_id: 64 bytes
_POINT = struct.Struct("<Q3d3BdQ")  # point3D_id, x, y, z, r, g, b, error, track_length: 51 bytes
_PARAMS: Mapping[int, struct.Struct] = {  # each camera model's parameters, by model id
    model_id: struct.Struct(f"<{count}d") for model_id, (_, count) in CAMERA_MODELS.items()}
# The camera models, for the messages: PINHOLE (1), OPENCV (4), OPENCV_FISHEYE (5).
_KNOWN = ", ".join(f"{name} ({model_id})" for model_id, (name, _) in CAMERA_MODELS.items())


@dataclass(frozen=True)
class Camera:
    """A camera of cameras.bin."""

    camera_id: int
    model_id: int  # a key of CAMERA_MODELS
    width: int  # in pixels
    height: int  # in pixels
    params: tuple[float, ...]  # the model's count, in the model's order


@dataclass(frozen=True)
class Image:
    """A registered image of images.bin: its Pose, its camera, its name and its 2D points.

    A world point x is at R x + t in the camera, with R the rotation of qvec, a unit quaternion in
    Hamilton's convention, and t the tvec.
    """

    image_id: int
    qvec: tuple[float, float, float, float]  # cam_from_world rotation, w x y z
    tvec: tuple[float, float, float]  # cam_from_world translation
    camera_id: int
    name: str  # like 000000.jpg
    points2d: np.ndarray  # POINT2D, one per 2D point


@dataclass(frozen=True)
class Points:
    """The points of points3D.bin: row i of each array, and tracks[i], belong to the same point."""

    ids: np.ndarray  # <u8, (n,)
    xyz: np.ndarray  # <f8, (n, 3)
    rgb: np.ndarray  # u1, (n, 3)
    errors: np.ndarray  # <f8, (n,), NO_ERROR for a point with none
    tracks: tuple[np.ndarray, ...]  # TRACK_ELEMENT, one array per point


@dataclass(frozen=True)
class Model:
    """A Model's cameras, images and points, as its three Model_Files hold them."""

    cameras: tuple[Camera, ...]
    images: tuple[Image, ...]  # in the file's order, which COLMAP's writer keeps
    points: Points


def encode_cameras(cameras: Sequence[Camera]) -> bytes:
    """The bytes of cameras.bin: the count, then each camera in increasing camera_id order (F3).

    Raises ValueError, a bug in our code, for a camera id given twice, a model id that isn't in
    CAMERA_MODELS, or a number of parameters that isn't the model's.
    """
    _refuse_repeats((camera.camera_id for camera in cameras), "camera")
    parts = [_COUNT.pack(len(cameras))]
    parts += (_camera_record(camera)
              for camera in sorted(cameras, key=lambda camera: camera.camera_id))
    return b"".join(parts)


def decode_cameras(data: bytes, path: Path) -> tuple[Camera, ...]:
    """The cameras in the bytes of cameras.bin, in the file's order.

    Bytes that don't follow the layout are the Req 10.4 failure, which names `path`.
    """
    reader = _Reader(data, path)
    (count,) = reader.unpack(_COUNT, "its count of cameras")
    seen: set[int] = set()
    cameras = []
    for place in range(count):
        camera = _place("camera", place, count)
        camera_id, model_id, width, height = reader.unpack(_CAMERA, camera)
        reader.refuse_repeat(seen, camera_id, "camera")
        if model_id not in CAMERA_MODELS:
            raise reader.broken(f"has the camera model id {model_id} for camera id {camera_id}, "
                                f"but only these camera models can be read: {_KNOWN}")
        params = reader.unpack(_PARAMS[model_id], f"the parameters of {camera}")
        cameras.append(Camera(camera_id, model_id, width, height, params))
    reader.finish(count, "camera")
    return tuple(cameras)


def encode_images(images: Sequence[Image]) -> bytes:
    """The bytes of images.bin: the count, then each image in the given order, the Model's (F3).

    Raises ValueError, a bug in our code, for an image id given twice or a name with a 0 byte in
    it.
    """
    _refuse_repeats((image.image_id for image in images), "image")
    parts = [_COUNT.pack(len(images))]
    parts += (_image_record(image) for image in images)
    return b"".join(parts)


def decode_images(data: bytes, path: Path) -> tuple[Image, ...]:
    """The images in the bytes of images.bin, in the file's order, which is the order COLMAP
    registered them in (F3).

    Bytes that don't follow the layout are the Req 10.4 failure, which names `path`.
    """
    reader = _Reader(data, path)
    (count,) = reader.unpack(_COUNT, "its count of images")
    seen: set[int] = set()
    images = tuple(_decode_image(reader, _place("image", place, count), seen)
                   for place in range(count))
    reader.finish(count, "image")
    return images


def encode_points(points: Points) -> bytes:
    """The bytes of points3D.bin: the count, then each point in increasing id order (F3).

    Raises ValueError, a bug in our code, for a point id given twice.
    """
    ids = points.ids.tolist()
    _refuse_repeats(ids, "point")
    xyz, rgb, errors = points.xyz.tolist(), points.rgb.tolist(), points.errors.tolist()
    parts = [_COUNT.pack(len(ids))]
    for i in sorted(range(len(ids)), key=ids.__getitem__):
        track = np.asarray(points.tracks[i], dtype=TRACK_ELEMENT)
        parts += (_POINT.pack(ids[i], *xyz[i], *rgb[i], errors[i], track.size), track.tobytes())
    return b"".join(parts)


def decode_points(data: bytes, path: Path) -> Points:
    """The points in the bytes of points3D.bin, in the file's order.

    Bytes that don't follow the layout are the Req 10.4 failure, which names `path`.
    """
    reader = _Reader(data, path)
    (count,) = reader.unpack(_COUNT, "its count of points")
    seen: set[int] = set()
    ids: list[int] = []
    xyz: list[tuple[float, float, float]] = []
    rgb: list[tuple[int, int, int]] = []
    errors: list[float] = []
    tracks: list[np.ndarray] = []
    for place in range(count):
        point = _place("point", place, count)
        point_id, x, y, z, r, g, b, error, length = reader.unpack(_POINT, point)
        reader.refuse_repeat(seen, point_id, "point")
        ids.append(point_id)
        xyz.append((x, y, z))
        rgb.append((r, g, b))
        errors.append(error)
        tracks.append(reader.array(TRACK_ELEMENT, length, f"the track of {point}"))
    reader.finish(count, "point")
    return Points(ids=np.array(ids, dtype="<u8"), xyz=np.array(xyz, dtype="<f8").reshape(-1, 3),
                  rgb=np.array(rgb, dtype="u1").reshape(-1, 3),
                  errors=np.array(errors, dtype="<f8"), tracks=tuple(tracks))


def read_images(path: Path) -> tuple[Image, ...]:
    """The images of the images.bin at `path`, which are exactly the images its Model registers
    (F3). A missing or broken file is the Req 10.4 failure."""
    return decode_images(_read(path), path)


def read_model(folder: Path) -> Model:
    """The Model in `folder`, from its three Model_Files.

    Every other file there, rigs.bin and frames.bin included, is ignored (Req 10.1, F4). A missing
    or broken file is the Req 10.4 failure.
    """
    cameras, images, points = (folder / name for name in FILES)
    return Model(cameras=decode_cameras(_read(cameras), cameras),
                 images=decode_images(_read(images), images),
                 points=decode_points(_read(points), points))


def write_model(folder: Path, model: Model) -> None:
    """Make `folder` and write exactly the three Model_Files of `model` into it.

    All three are encoded before anything is written, so a bug's ValueError writes nothing. An
    OSError, like a full disk, is the Req 11.12 failure, naming the folder or the file and the
    reason.
    """
    contents = (encode_cameras(model.cameras), encode_images(model.images),
                encode_points(model.points))
    try:
        folder.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise _not_written(f"The folder {folder} couldn't be made", folder, error) from error
    for name, data in zip(FILES, contents):
        path = folder / name
        try:
            path.write_bytes(data)
        except OSError as error:
            raise _not_written(f"The model file {path} couldn't be written", path,
                               error) from error


class _Reader:
    """Walks the bytes of one Model_File with an offset, refusing what the layout doesn't allow."""

    def __init__(self, data: bytes, path: Path) -> None:
        self.data = bytes(data)  # the same object when it's bytes, for finding each name's end
        self.view = memoryview(self.data)
        self.path = path
        self.offset = 0

    def broken(self, what: str) -> ExpectedFailure:
        """The Req 10.4 failure for this file, where `what` says what's wrong."""
        return _broken(self.path, what)

    def unpack(self, fixed: struct.Struct, part: str) -> tuple[Any, ...]:
        """The values of `fixed` at the offset, moving past them. `part` names them in the
        refusal of a file that ends inside them, like "image 3 of 9"."""
        return fixed.unpack_from(self.view, self._take(fixed.size, part))

    def array(self, dtype: np.dtype, count: int, part: str) -> np.ndarray:
        """A copy of the `count` elements of `dtype` at the offset, moving past them."""
        start = self._take(count * dtype.itemsize, part)
        return np.frombuffer(self.view, dtype=dtype, count=count, offset=start).copy()

    def name(self, part: str) -> str:
        """The UTF-8 name at the offset, moving past it and the 0 byte that ends it."""
        end = self.data.find(b"\0", self.offset)
        if end < 0:
            raise self.broken(f"ends too soon, inside {part}, which has no 0 byte before the end "
                              "of the file")
        try:
            name = self.data[self.offset:end].decode("utf-8")
        except UnicodeDecodeError as error:
            raise self.broken(f"holds {part} in bytes that aren't UTF-8") from error
        self.offset = end + 1
        return name

    def refuse_repeat(self, seen: set[int], record_id: int, kind: str) -> None:
        """Refuse a `kind` id that's already in `seen`, and otherwise add it."""
        if record_id in seen:
            raise self.broken(f"gives the {kind} id {record_id} twice")
        seen.add(record_id)

    def finish(self, count: int, kind: str) -> None:
        """Refuse bytes left after the file's `count` records of `kind`."""
        left = len(self.data) - self.offset
        if left:
            raise self.broken(f"has {_count(left, 'byte')} left after its {_count(count, kind)}")

    def _take(self, size: int, part: str) -> int:
        """The offset, moved `size` bytes on, unless the file ends before that."""
        start = self.offset
        if start + size > len(self.data):
            raise self.broken(f"ends too soon, inside {part}")
        self.offset = start + size
        return start


def _decode_image(reader: _Reader, image: str, seen: set[int]) -> Image:
    """The image record at the reader's offset, which `image` names, like "image 3 of 9"."""
    image_id, qw, qx, qy, qz, tx, ty, tz, camera_id = reader.unpack(_IMAGE, image)
    reader.refuse_repeat(seen, image_id, "image")
    name = reader.name(f"the name of {image}")
    (count,) = reader.unpack(_COUNT, f"the 2D point count of {image}")
    points2d = reader.array(POINT2D, count, f"the 2D points of {image}")
    return Image(image_id, (qw, qx, qy, qz), (tx, ty, tz), camera_id, name, points2d)


def _camera_record(camera: Camera) -> bytes:
    """A camera's record: its id, model id, width and height, then the model's parameters."""
    if camera.model_id not in CAMERA_MODELS:
        raise ValueError(f"The camera id {camera.camera_id} has the camera model id "
                         f"{camera.model_id}, which isn't one of {_KNOWN}.")
    name, count = CAMERA_MODELS[camera.model_id]
    if len(camera.params) != count:
        raise ValueError(f"The camera id {camera.camera_id} has "
                         f"{_count(len(camera.params), 'parameter')}, but its model, {name}, "
                         f"takes {count}.")
    return (_CAMERA.pack(camera.camera_id, camera.model_id, camera.width, camera.height)
            + _PARAMS[camera.model_id].pack(*camera.params))


def _image_record(image: Image) -> bytes:
    """An image's record: its id, Pose and camera id, its name and a 0 byte, then its 2D points."""
    if "\0" in image.name:
        raise ValueError(f"The image id {image.image_id} has the name {image.name!r}, which holds "
                         "a 0 byte, but images.bin ends each name with one.")
    points2d = np.asarray(image.points2d, dtype=POINT2D)
    return b"".join((_IMAGE.pack(image.image_id, *image.qvec, *image.tvec, image.camera_id),
                     image.name.encode("utf-8"), b"\0", _COUNT.pack(points2d.size),
                     points2d.tobytes()))


def _refuse_repeats(ids: Iterable[int], kind: str) -> None:
    """Raise ValueError, a bug in our code, for the first id that `ids` gives twice."""
    seen: set[int] = set()
    for record_id in ids:
        if record_id in seen:
            raise ValueError(f"The {kind} id {record_id} is given twice, but a model holds each "
                             f"{kind} id once.")
        seen.add(record_id)


def _place(kind: str, place: int, count: int) -> str:
    """A record by its place in its file, counted from 1, like "image 3 of 9"."""
    return f"{kind} {place + 1} of {count}"


def _count(number: int, noun: str) -> str:
    """`number` with `noun`, plural unless the number is 1, like "1 camera" or "13 bytes"."""
    return f"{number} {noun}" if number == 1 else f"{number} {noun}s"


def _read(path: Path) -> bytes:
    """The bytes of the Model_File `path`. A missing file is the Req 10.4 failure."""
    try:
        return path.read_bytes()
    except FileNotFoundError as error:
        raise _broken(path, "is missing") from error


def _broken(path: Path, what: str) -> ExpectedFailure:
    """The Req 10.4 failure: the Model_File `path`, what's wrong with it, and what to do."""
    return ExpectedFailure(f"The model file {path} {what}. Check that COLMAP 4.2.1 is installed, "
                           "then run the step again.")


def _not_written(problem: str, path: Path, error: OSError) -> ExpectedFailure:
    """The Req 11.12 failure: `problem`, which names the file, the reason the system gave, and
    what to do."""
    return ExpectedFailure(f"{problem} ({error.strerror or error}). Free some space on the disk, "
                           f"or check that you can write to {path.parent}, then run the step "
                           "again.")
