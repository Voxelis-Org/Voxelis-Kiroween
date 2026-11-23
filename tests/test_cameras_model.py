"""Tests for splatpipe/cameras/model.py: the Model_Files cameras.bin, images.bin and points3D.bin.

The hand-built model has one camera of each model the files may hold, PINHOLE, OPENCV and
OPENCV_FISHEYE, two images, one with no 2D points and one with a 2D point that has NaN
coordinates and no 3D point, and three points, one with an empty track. Its image id 3000000000
and camera id 2³² − 1 don't fit a signed 32-bit field, and its point id 2⁶⁴ − 2 doesn't fit a
signed 64-bit one, so its bytes show that every id field is unsigned.

The bytes that model must encode to are put together here field by field from the layout tables
of the design's "The Model_Files", with struct.pack and explicit little-endian formats, and never
with the encoders (Req 10.2). Task 8's integration test checks the same layout against COLMAP
itself. A decoded model is compared with every float as its 8 bytes and every array as its dtype,
shape and bytes, so a NaN's sign and payload have to come back as they went in.

Property 7 draws models with Hypothesis: up to 5 cameras of the three models, 6 images and 10
points, each kind with ids unique across the whole range of its uint32 or uint64 field. Every
float is any float64, the infinities and NaNs of either sign with any payload, quiet or
signalling, included. Each name has up to 30 characters that UTF-8 can encode, with no 0 byte,
each image 0 to 20 2D points, with NO_POINT3D about half the time, and each point 0 to 8 track
elements. Decoding what the encoders write has to give back the model built again from the drawn
fields, compared the same way, with the cameras and the points in increasing id order and the
images in the order drawn (Req 10.9).

The broken files are the hand-built model's files cut short or with a byte added, or files put
together here with struct.pack, and read_model reads them from tmp_path. Each refusal has to name
the file and what's wrong, say to check that COLMAP 4.2.1 is installed and to run the step again,
and hold no traceback (Req 10.4, 16.9).

The full disk is a stand-in for Path.write_bytes, which write_model writes with, that raises
OSError(errno.ENOSPC, "No space left on device") for one of the three files and writes the others
(Req 11.12).
"""

import errno
import math
import struct
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from splatpipe.cameras.model import (NO_ERROR, NO_POINT3D, POINT2D, TRACK_ELEMENT, Camera, Image,
                                     Model, Points, decode_cameras, decode_images, decode_points,
                                     encode_cameras, encode_images, encode_points, read_images,
                                     read_model, write_model)
from splatpipe.errors import ExpectedFailure

FULL_DISK = "No space left on device"  # the reason the stand-in full disk gives
NAMES = ("cameras.bin", "images.bin", "points3D.bin")  # the three Model_Files


# The fields of the layout tables, each packed little-endian with no padding

def uint8(*values: int) -> bytes:
    """`values` as uint8s, like a point's r, g and b."""
    return struct.pack(f"<{len(values)}B", *values)


def int32(value: int) -> bytes:
    """`value` as an int32, like a camera's model_id."""
    return struct.pack("<i", value)


def uint32(value: int) -> bytes:
    """`value` as a uint32, like a camera_id or an image_id."""
    return struct.pack("<I", value)


def uint64(value: int) -> bytes:
    """`value` as a uint64, like a file's count, a width or a point3D_id."""
    return struct.pack("<Q", value)


def float64(*values: float) -> bytes:
    """`values` as float64s, like a camera's params or an image's qw, qx, qy and qz."""
    return struct.pack(f"<{len(values)}d", *values)


def camera_bytes(camera_id: int, model_id: int, width: int, height: int,
                 params: Sequence[float]) -> bytes:
    """A camera's record in cameras.bin: camera_id, model_id, width, height, then params."""
    return uint32(camera_id) + int32(model_id) + uint64(width) + uint64(height) + float64(*params)


def image_head(image_id: int, qvec: Sequence[float], tvec: Sequence[float],
               camera_id: int) -> bytes:
    """The start of an image's record in images.bin, up to its name: image_id, qw, qx, qy, qz,
    tx, ty, tz and camera_id."""
    return uint32(image_id) + float64(*qvec) + float64(*tvec) + uint32(camera_id)


def image_bytes(image_id: int, qvec: Sequence[float], tvec: Sequence[float], camera_id: int,
                name: bytes, points2d: Sequence[tuple[float, float, int]]) -> bytes:
    """An image's record in images.bin: its start, the name's bytes and a 0 byte, num_points2D,
    then x, y and point3D_id for each 2D point."""
    return (image_head(image_id, qvec, tvec, camera_id) + name + b"\0" + uint64(len(points2d))
            + b"".join(float64(x, y) + uint64(point3d_id) for x, y, point3d_id in points2d))


def point_bytes(point_id: int, xyz: Sequence[float], rgb: Sequence[int], error: float,
                track: Sequence[tuple[int, int]]) -> bytes:
    """A point's record in points3D.bin: point3D_id, x, y, z, r, g, b, error, track_length, then
    image_id and point2D_idx for each track element."""
    return (uint64(point_id) + float64(*xyz) + uint8(*rgb) + float64(error) + uint64(len(track))
            + b"".join(uint32(image_id) + uint32(index) for image_id, index in track))


# The hand-built model

# A NaN with its sign bit set and a payload, whose 8 bytes have to come back as they went in.
PAYLOAD_NAN = struct.unpack("<d", uint64(0xFFF8_0000_0000_0ABC))[0]

# Its cameras, in increasing id order: camera_id, model_id, width, height and params.
CAMERA_FIELDS = (
    (2, 1, 320, 240, (260.0, 261.5, 160.0, 120.0)),  # PINHOLE: fx, fy, cx, cy
    (7, 4, 1920, 1080,  # OPENCV: fx, fy, cx, cy, k1, k2, p1, p2
     (1500.0, 1501.5, 960.5, 540.25, -0.125, 0.0625, 0.001, -0.002)),
    (2**32 - 1, 5, 4000, 3000,  # OPENCV_FISHEYE: fx, fy, cx, cy, k1, k2, k3, k4
     (1154.5, 1155.0, 2000.0, 1500.0, 0.05, -0.01, 0.003, -0.0004)),
)
# Its images, in the Model's order, which isn't id order: image_id, qvec, tvec, camera_id, name,
# and each 2D point's x, y and point3d_id. The first image's 2D point 1 has NaN coordinates and
# no 3D point. The second image has no 2D points. Values like 0.1 and 10.4 have no exact float32,
# so a reader that kept fewer bits than a float64's would show.
IMAGE_FIELDS = (
    (3_000_000_000, (0.7, 0.1, -0.5, 0.5), (0.25, -1.6, 3.1), 7, "000003.jpg",
     ((10.4, 20.25, 5), (math.nan, PAYLOAD_NAN, NO_POINT3D), (300.75, 200.5, 40))),
    (4, (1.0, 0.0, 0.0, 0.0), (-2.0, 0.5, 1.25), 2**32 - 1, "000000.jpg", ()),
)
# Its points, in increasing id order: point3D_id, xyz, rgb, error, and each track element's
# image_id and point2D_idx. Points 5 and 40 are the first image's 2D points 0 and 2. The last
# point has an empty track and no error.
POINT_FIELDS = (
    (5, (0.1, -0.25, 4.2), (255, 0, 128), 0.84, ((3_000_000_000, 0),)),
    (40, (-1.5, 2.3, 6.0), (12, 34, 56), 1.3, ((3_000_000_000, 2),)),
    (2**64 - 2, (3.0, -4.7, 9.25), (0, 200, 7), NO_ERROR, ()),
)


def image_of(image_id: int, qvec: tuple[float, float, float, float],
             tvec: tuple[float, float, float], camera_id: int, name: str,
             points2d: Sequence[tuple[float, float, int]]) -> Image:
    """The Image of an entry of IMAGE_FIELDS."""
    return Image(image_id, qvec, tvec, camera_id, name, np.array(list(points2d), dtype=POINT2D))


def points_of(fields: Sequence[tuple[Any, ...]]) -> Points:
    """The Points of entries of POINT_FIELDS, in the order given."""
    return Points(ids=np.array([point[0] for point in fields], dtype="<u8"),
                  xyz=np.array([point[1] for point in fields], dtype="<f8").reshape(-1, 3),
                  rgb=np.array([point[2] for point in fields], dtype="u1").reshape(-1, 3),
                  errors=np.array([point[3] for point in fields], dtype="<f8"),
                  tracks=tuple(np.array(list(point[4]), dtype=TRACK_ELEMENT)
                               for point in fields))


CAMERAS = tuple(Camera(*fields) for fields in CAMERA_FIELDS)
IMAGES = tuple(image_of(*fields) for fields in IMAGE_FIELDS)
MODEL = Model(cameras=CAMERAS, images=IMAGES, points=points_of(POINT_FIELDS))

# Its records and its three files, put together from the layout tables.
CAMERA_RECORDS = tuple(camera_bytes(*fields) for fields in CAMERA_FIELDS)
IMAGE_RECORDS = tuple(image_bytes(image_id, qvec, tvec, camera_id, name.encode("utf-8"), points2d)
                      for image_id, qvec, tvec, camera_id, name, points2d in IMAGE_FIELDS)
POINT_RECORDS = tuple(point_bytes(*fields) for fields in POINT_FIELDS)
FILES_BYTES = {  # each file's count, then its records
    "cameras.bin": uint64(len(CAMERA_RECORDS)) + b"".join(CAMERA_RECORDS),
    "images.bin": uint64(len(IMAGE_RECORDS)) + b"".join(IMAGE_RECORDS),
    "points3D.bin": uint64(len(POINT_RECORDS)) + b"".join(POINT_RECORDS),
}


# Comparing models bit for bit

def array_key(array: np.ndarray) -> tuple[Any, ...]:
    """An array's dtype, shape and bytes, which are equal only for arrays equal bit for bit."""
    return array.dtype, array.shape, array.tobytes()


def camera_key(camera: Camera) -> tuple[Any, ...]:
    """A camera's fields, with its params as their float64 bytes."""
    return camera.camera_id, camera.model_id, camera.width, camera.height, float64(*camera.params)


def image_key(image: Image) -> tuple[Any, ...]:
    """An image's fields, with its Pose as float64 bytes and its 2D points as array_key gives
    them."""
    return (image.image_id, float64(*image.qvec), float64(*image.tvec), image.camera_id,
            image.name, array_key(image.points2d))


def points_key(points: Points) -> tuple[Any, ...]:
    """Each array of `points` as array_key gives it, with its tracks one by one."""
    return (array_key(points.ids), array_key(points.xyz), array_key(points.rgb),
            array_key(points.errors), tuple(array_key(track) for track in points.tracks))


def model_key(model: Model) -> tuple[Any, ...]:
    """A model's cameras, images and points as the keys above give them, in the model's order."""
    return (tuple(camera_key(camera) for camera in model.cameras),
            tuple(image_key(image) for image in model.images), points_key(model.points))


# The layout (Req 10.2), and reading it back (Req 10.1)

def test_the_hand_built_model_encodes_to_the_bytes_of_the_layout_tables() -> None:
    # The layout tables' point3D_id for a 2D point with no 3D point, and error for a point with
    # none.
    assert (NO_POINT3D, NO_ERROR) == (2**64 - 1, -1.0)
    assert encode_cameras(MODEL.cameras) == FILES_BYTES["cameras.bin"]
    # The images' order isn't id order, so an encoder that sorted them would fail here too.
    assert encode_images(MODEL.images) == FILES_BYTES["images.bin"]
    assert encode_points(MODEL.points) == FILES_BYTES["points3D.bin"]


def test_the_bytes_of_the_layout_tables_decode_back_to_the_hand_built_model() -> None:
    model = Model(cameras=decode_cameras(FILES_BYTES["cameras.bin"], Path("cameras.bin")),
                  images=decode_images(FILES_BYTES["images.bin"], Path("images.bin")),
                  points=decode_points(FILES_BYTES["points3D.bin"], Path("points3D.bin")))
    assert model_key(model) == model_key(MODEL)


def test_cameras_and_points_out_of_id_order_come_out_sorted_and_images_keep_their_order() -> None:
    # Given as ids 2³² − 1, 2 and 7, the cameras are written as 2, 7 and 2³² − 1.
    cameras = encode_cameras([CAMERAS[2], CAMERAS[0], CAMERAS[1]])
    assert cameras == FILES_BYTES["cameras.bin"]
    assert ([camera.camera_id for camera in decode_cameras(cameras, Path("cameras.bin"))]
            == [2, 7, 2**32 - 1])
    # Given as ids 2⁶⁴ − 2, 5 and 40, the points are written as 5, 40 and 2⁶⁴ − 2, each with its
    # own position, colour, error and track.
    points = encode_points(points_of([POINT_FIELDS[2], POINT_FIELDS[0], POINT_FIELDS[1]]))
    assert points == FILES_BYTES["points3D.bin"]
    assert decode_points(points, Path("points3D.bin")).ids.tolist() == [5, 40, 2**64 - 2]
    # The images are written in the order they're given, whichever it is.
    for order in ((0, 1), (1, 0)):
        images = encode_images([IMAGES[place] for place in order])
        assert images == uint64(2) + b"".join(IMAGE_RECORDS[place] for place in order)
        assert ([image.image_id for image in decode_images(images, Path("images.bin"))]
                == [IMAGE_FIELDS[place][0] for place in order])


def test_a_pinhole_camera_record_is_56_bytes() -> None:
    assert len(encode_cameras([CAMERAS[0]])) == 8 + 56  # the file's count, then the record


@pytest.mark.parametrize(("name", "count"), [
    pytest.param("000000.jpg", 0, id="no 2D points"),
    # "größe/000001.jpg" is 16 characters, and 18 bytes in UTF-8.
    pytest.param("größe/000001.jpg", 1, id="a name of 16 characters in 18 bytes"),
    pytest.param("x", 5, id="5 2D points"),
])
def test_an_image_record_is_64_bytes_plus_its_name_and_0_byte_plus_8_plus_24_per_2d_point(
        name: str, count: int) -> None:
    image = Image(1, (1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 1, name,
                  np.zeros(count, dtype=POINT2D))
    assert (len(encode_images([image]))
            == 8 + 64 + len(name.encode("utf-8")) + 1 + 8 + 24 * count)


@pytest.mark.parametrize("length", [0, 1, 3])
def test_a_point_record_is_51_bytes_plus_8_per_track_element(length: int) -> None:
    points = Points(ids=np.array([1], dtype="<u8"), xyz=np.zeros((1, 3)),
                    rgb=np.zeros((1, 3), dtype="u1"), errors=np.array([NO_ERROR]),
                    tracks=(np.zeros(length, dtype=TRACK_ELEMENT),))
    assert len(encode_points(points)) == 8 + 51 + 8 * length


def write_files(folder: Path, files: Mapping[str, bytes]) -> None:
    """Write each of `files` into `folder`, under its name."""
    for name, data in files.items():
        (folder / name).write_bytes(data)


def test_read_model_reads_the_three_files_and_ignores_rigs_bin_and_frames_bin(
        tmp_path: Path) -> None:
    # Neither holds anything a reader could parse, so reading either would fail.
    write_files(tmp_path, {**FILES_BYTES, "rigs.bin": b"not a rig", "frames.bin": b"not a frame"})
    assert model_key(read_model(tmp_path)) == model_key(MODEL)


def test_read_images_gives_the_images_of_a_folders_images_bin(tmp_path: Path) -> None:
    write_files(tmp_path, {"images.bin": FILES_BYTES["images.bin"]})
    assert ([image_key(image) for image in read_images(tmp_path / "images.bin")]
            == [image_key(image) for image in IMAGES])


# Property 7: The Model_Files round trip (Req 10.9)

def unsigned(size: int) -> st.SearchStrategy[int]:
    """A whole number of `size` bytes, unsigned: from Hypothesis's integers, which favour small
    numbers and the ends of the range, or, half the time, from `size` bytes of any value, which
    land in the top half of the range about as often as in the bottom half."""
    any_bytes = st.binary(min_size=size, max_size=size)
    return (st.integers(0, 2 ** (8 * size) - 1)
            | any_bytes.map(lambda data: int.from_bytes(data, "little")))


UINT32S = unsigned(4)  # a camera_id, an image_id or a track element's field
UINT64S = unsigned(8)  # a point3D_id, a 2D point's point3d_id, a width or a height
COLOURS = st.tuples(*[st.integers(0, 255)] * 3)  # a point's r, g and b
# Each camera model the files may hold, by model id, with its number of parameters (F2):
# PINHOLE, OPENCV and OPENCV_FISHEYE.
PARAMETER_COUNTS = {1: 4, 4: 8, 5: 8}


def float_of_bits(bits: int) -> float:
    """The float64 whose 8 bytes are those of the uint64 `bits`, both little-endian."""
    return struct.unpack("<d", uint64(bits))[0]


# A NaN of either sign with any payload, made from its bits: all 11 exponent bits set and a
# mantissa that isn't 0, quiet when the mantissa's top bit is set and signalling when it isn't.
# They're drawn on their own, so NaN payloads are common.
NANS = st.builds(lambda sign, mantissa: float_of_bits(sign << 63 | 0x7FF << 52 | mantissa),
                 st.integers(0, 1), st.integers(1, 2**52 - 1))
# Any float64: Hypothesis's floats, which often give ±0, the infinities and the ends of the
# range, the NaNs above, and the float of any 64 bits.
FLOAT64S = st.floats() | NANS | UINT64S.map(float_of_bits)
# An image's name: up to 30 characters that UTF-8 can encode, with no "\0", which ends each name
# in images.bin.
IMAGE_NAMES = st.text(st.characters(codec="utf-8", exclude_characters="\0"), max_size=30)
# An image's 2D point: x, y, and a point3d_id that's NO_POINT3D about half the time.
POINTS2D = st.tuples(FLOAT64S, FLOAT64S, st.just(NO_POINT3D) | UINT64S)
TRACK_ELEMENTS = st.tuples(UINT32S, UINT32S)  # an image_id and a point2D_idx


def float_tuples(count: int) -> st.SearchStrategy[tuple[float, ...]]:
    """`count` floats of FLOAT64S, as a tuple."""
    return st.tuples(*[FLOAT64S] * count)


@dataclass(frozen=True)
class ModelCase:
    """A drawn model, as entries like CAMERA_FIELDS, IMAGE_FIELDS and POINT_FIELDS, in the order
    drawn."""

    cameras: tuple[tuple[Any, ...], ...]
    images: tuple[tuple[Any, ...], ...]
    points: tuple[tuple[Any, ...], ...]


@st.composite
def model_cases(draw: st.DrawFn) -> ModelCase:
    """Up to 5 cameras, 6 images and 10 points, each kind with its ids unique and drawn from the
    whole range of their field, in the order drawn, which often isn't id order.

    Each camera has one of the three models, any width and height, and any floats for its model's
    parameters. Each image has any Pose, camera id and name, and 0 to 20 2D points. Each point has
    any position, colour and error, and 0 to 8 track elements. The decoders don't check that an
    image's camera, or a track element's image and 2D point, are in the model, so those fields
    are any uint32.
    """
    cameras = []
    for camera_id in draw(st.lists(UINT32S, max_size=5, unique=True)):
        model_id = draw(st.sampled_from(sorted(PARAMETER_COUNTS)))
        cameras.append((camera_id, model_id, draw(UINT64S), draw(UINT64S),
                        draw(float_tuples(PARAMETER_COUNTS[model_id]))))
    images = [(image_id, draw(float_tuples(4)), draw(float_tuples(3)), draw(UINT32S),
               draw(IMAGE_NAMES), draw(st.lists(POINTS2D, max_size=20)))
              for image_id in draw(st.lists(UINT32S, max_size=6, unique=True))]
    points = [(point_id, draw(float_tuples(3)), draw(COLOURS), draw(FLOAT64S),
               draw(st.lists(TRACK_ELEMENTS, max_size=8)))
              for point_id in draw(st.lists(UINT64S, max_size=10, unique=True))]
    return ModelCase(cameras=tuple(cameras), images=tuple(images), points=tuple(points))


def model_of(cameras: Iterable[tuple[Any, ...]], images: Iterable[tuple[Any, ...]],
             points: Sequence[tuple[Any, ...]]) -> Model:
    """The Model of entries like CAMERA_FIELDS, IMAGE_FIELDS and POINT_FIELDS, in the order
    given."""
    return Model(cameras=tuple(Camera(*fields) for fields in cameras),
                 images=tuple(image_of(*fields) for fields in images), points=points_of(points))


@given(model_cases())
def test_the_model_files_round_trip(case: ModelCase) -> None:
    """Feature: cameras, Property 7: The Model_Files round trip

    **Validates: Requirements 10.9**
    """
    model = model_of(case.cameras, case.images, case.points)
    back = Model(cameras=decode_cameras(encode_cameras(model.cameras), Path("cameras.bin")),
                 images=decode_images(encode_images(model.images), Path("images.bin")),
                 points=decode_points(encode_points(model.points), Path("points3D.bin")))
    # The oracle is the model built again from the drawn fields, with the cameras and the points
    # sorted by their ids, the first of their fields, and the images in the order drawn.
    # model_key compares every float as its 8 bytes.
    in_id_order = model_of(sorted(case.cameras, key=lambda fields: fields[0]), case.images,
                           sorted(case.points, key=lambda fields: fields[0]))
    assert model_key(back) == model_key(in_id_order)


# Broken files (Req 10.4)

def read_model_refusal(folder: Path, name: str, data: bytes | None) -> str:
    """The Readable_Error of read_model on a folder of the hand-built model's three files, with
    the file `name` holding `data` instead, or left out when `data` is None."""
    files = {other: contents for other, contents in FILES_BYTES.items() if other != name}
    if data is not None:
        files[name] = data
    write_files(folder, files)
    with pytest.raises(ExpectedFailure) as caught:
        read_model(folder)
    return str(caught.value)


def assert_refused(message: str, path: Path, what: str) -> None:
    """Check that a Req 10.4 refusal names the file and what's wrong, says to check that COLMAP
    4.2.1 is installed and to run the step again, and holds no traceback."""
    assert str(path) in message
    assert what in message
    assert "Check that COLMAP 4.2.1 is installed" in message
    assert "run the step again" in message
    assert "Traceback" not in message


@pytest.mark.parametrize("name", NAMES)
def test_a_missing_model_file_fails_naming_it(tmp_path: Path, name: str) -> None:
    assert_refused(read_model_refusal(tmp_path, name, None), tmp_path / name, "is missing")


def test_read_images_of_a_missing_images_bin_fails_naming_it(tmp_path: Path) -> None:
    with pytest.raises(ExpectedFailure) as caught:
        read_images(tmp_path / "images.bin")
    assert_refused(str(caught.value), tmp_path / "images.bin", "is missing")


@pytest.mark.parametrize(("name", "cut", "part"), [
    # cameras.bin: the count, 8 bytes, then three cameras, each 24 bytes up to height and then its
    # params, 8 bytes each: 4 for camera 1, the PINHOLE one, and 8 for cameras 2 and 3.
    pytest.param("cameras.bin", 0, "inside its count of cameras", id="an empty cameras.bin"),
    pytest.param("cameras.bin", 5, "inside its count of cameras", id="the count of cameras"),
    pytest.param("cameras.bin", 8 + 20, "inside camera 1 of 3", id="a camera"),
    pytest.param("cameras.bin", 8 + 24 + 16, "inside the parameters of camera 1 of 3",
                 id="a camera's params"),
    pytest.param("cameras.bin", 8 + 56, "inside camera 2 of 3", id="between two cameras"),
    pytest.param("cameras.bin", -1, "inside the parameters of camera 3 of 3",
                 id="the last camera"),
    # images.bin: the count, then image 1: 64 bytes up to camera_id, the name 000003.jpg and its
    # 0 byte, 11 bytes, num_points2D, 8 bytes, and three 2D points of 24 bytes, 155 bytes in all.
    # Then image 2, which has no 2D points.
    pytest.param("images.bin", 0, "inside its count of images", id="an empty images.bin"),
    pytest.param("images.bin", 8 + 40, "inside image 1 of 2", id="an image"),
    pytest.param("images.bin", 8 + 64 + 4, "inside the name of image 1 of 2", id="a name"),
    pytest.param("images.bin", 8 + 64 + 11 + 2, "inside the 2D point count of image 1 of 2",
                 id="an image's count of 2D points"),
    pytest.param("images.bin", 8 + 64 + 11 + 8 + 34, "inside the 2D points of image 1 of 2",
                 id="an image's 2D points"),
    pytest.param("images.bin", 8 + 155, "inside image 2 of 2", id="between two images"),
    pytest.param("images.bin", -1, "inside the 2D point count of image 2 of 2",
                 id="the last image"),
    # points3D.bin: the count, then three points, each 51 bytes up to track_length and then 8
    # bytes per track element: one for points 1 and 2, and none for point 3.
    pytest.param("points3D.bin", 3, "inside its count of points", id="the count of points"),
    pytest.param("points3D.bin", 8 + 30, "inside point 1 of 3", id="a point"),
    pytest.param("points3D.bin", 8 + 51 + 4, "inside the track of point 1 of 3",
                 id="a point's track"),
    pytest.param("points3D.bin", -1, "inside point 3 of 3", id="the last point"),
])
def test_a_file_that_ends_too_soon_fails_naming_it_and_where_it_ends(
        tmp_path: Path, name: str, cut: int, part: str) -> None:
    message = read_model_refusal(tmp_path, name, FILES_BYTES[name][:cut])
    assert_refused(message, tmp_path / name, f"ends too soon, {part}")


# 0 is SIMPLE_PINHOLE and 2 SIMPLE_RADIAL, which COLMAP has but the step never reads, and -1 is
# no camera model at all.
@pytest.mark.parametrize("model_id", [0, 2, -1])
def test_an_unknown_camera_model_id_fails_naming_it(tmp_path: Path, model_id: int) -> None:
    data = uint64(1) + camera_bytes(7, model_id, 320, 240, (260.0, 261.5, 160.0, 120.0))
    message = read_model_refusal(tmp_path, "cameras.bin", data)
    assert_refused(message, tmp_path / "cameras.bin",
                   f"has the camera model id {model_id} for camera id 7")


def test_a_name_with_no_0_byte_before_the_end_of_the_file_fails(tmp_path: Path) -> None:
    data = uint64(1) + image_head(4, (1.0, 0.0, 0.0, 0.0), (-2.0, 0.5, 1.25), 7) + b"000000.jpg"
    message = read_model_refusal(tmp_path, "images.bin", data)
    assert_refused(message, tmp_path / "images.bin",
                   "the name of image 1 of 1, which has no 0 byte before the end of the file")


def test_a_name_that_isnt_utf8_fails(tmp_path: Path) -> None:
    # The bytes 0xff and 0xfe never appear in UTF-8.
    data = uint64(1) + image_bytes(4, (1.0, 0.0, 0.0, 0.0), (-2.0, 0.5, 1.25), 7,
                                   b"\xff\xfe.jpg", ())
    message = read_model_refusal(tmp_path, "images.bin", data)
    assert_refused(message, tmp_path / "images.bin",
                   "holds the name of image 1 of 1 in bytes that aren't UTF-8")


@pytest.mark.parametrize(("name", "data", "what"), [
    pytest.param("cameras.bin", uint64(2) + CAMERA_RECORDS[1] * 2, "gives the camera id 7 twice",
                 id="a camera id"),
    pytest.param("images.bin", uint64(2) + IMAGE_RECORDS[1] * 2, "gives the image id 4 twice",
                 id="an image id"),
    pytest.param("points3D.bin", uint64(2) + POINT_RECORDS[1] * 2, "gives the point id 40 twice",
                 id="a point id"),
])
def test_an_id_given_twice_fails_naming_it(tmp_path: Path, name: str, data: bytes,
                                           what: str) -> None:
    assert_refused(read_model_refusal(tmp_path, name, data), tmp_path / name, what)


@pytest.mark.parametrize(("name", "records"), [
    ("cameras.bin", "3 cameras"), ("images.bin", "2 images"), ("points3D.bin", "3 points")])
def test_bytes_after_the_last_record_fail(tmp_path: Path, name: str, records: str) -> None:
    message = read_model_refusal(tmp_path, name, FILES_BYTES[name] + b"\0")
    assert_refused(message, tmp_path / name, f"has 1 byte left after its {records}")


# Bugs in our own code: the encoders raise ValueError

@pytest.mark.parametrize(("encode", "given", "words"), [
    pytest.param(encode_cameras, (CAMERAS[1], CAMERAS[1]), "camera id 7 is given twice",
                 id="a camera id given twice"),
    pytest.param(encode_images, (IMAGES[1], IMAGES[1]), "image id 4 is given twice",
                 id="an image id given twice"),
    pytest.param(encode_points, points_of([POINT_FIELDS[1], POINT_FIELDS[1]]),
                 "point id 40 is given twice", id="a point id given twice"),
    pytest.param(encode_images,
                 (image_of(4, (1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 7, "000\0.jpg", ()),),
                 "holds a 0 byte", id="a name with a 0 byte in it"),
    pytest.param(encode_cameras, (Camera(7, 2, 320, 240, (260.0, 160.0, 120.0, 0.0)),),
                 "camera model id 2", id="a model id it doesn't know"),
    # An OPENCV camera with a PINHOLE camera's 4 params.
    pytest.param(encode_cameras, (Camera(7, 4, 320, 240, (260.0, 261.5, 160.0, 120.0)),),
                 "OPENCV, takes 8", id="a parameter count that doesn't match the model"),
])
def test_the_encoders_raise_value_error_for_each_bug(encode: Callable[[Any], bytes], given: Any,
                                                     words: str) -> None:
    with pytest.raises(ValueError) as caught:
        encode(given)
    assert words in str(caught.value)


# Writing the files (Req 10.2, 11.12)

def test_write_model_makes_the_folder_and_writes_exactly_the_three_files(tmp_path: Path) -> None:
    folder = tmp_path / "work" / "cameras" / "sparse"
    write_model(folder, MODEL)
    assert {path.name: path.read_bytes() for path in folder.iterdir()} == FILES_BYTES


REAL_WRITE_BYTES = Path.write_bytes  # kept before any test puts the stand-in in its place


def full_disk_for(name: str) -> Callable[[Path, bytes], int]:
    """A stand-in for Path.write_bytes that fails the way a full disk does for the file `name`,
    and writes every other file."""
    def write_bytes(path: Path, data: bytes) -> int:
        if path.name == name:
            raise OSError(errno.ENOSPC, FULL_DISK)
        return REAL_WRITE_BYTES(path, data)
    return write_bytes


@pytest.mark.parametrize("name", NAMES)
def test_write_model_on_a_full_disk_fails_naming_the_file_and_the_reason(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    folder = tmp_path / "sparse"
    monkeypatch.setattr(Path, "write_bytes", full_disk_for(name))
    with pytest.raises(ExpectedFailure) as caught:
        write_model(folder, MODEL)
    message = str(caught.value)
    assert str(folder / name) in message
    assert FULL_DISK in message
    assert "Traceback" not in message
