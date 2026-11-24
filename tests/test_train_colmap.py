"""Tests for the Model_Reader: splatpipe/train/colmap.py.

Every model is written by tests/camera_files.py's write_model, the tests' own writer of COLMAP's
binary layout, which imports nothing from colmap.py (Req 14.1). The unit tests write into the
test's tmp_path. Their default scene is a good model: one 8 x 6 PINHOLE camera, 3 images with 12
keypoints each and 12 points with 3 track entries each. Each refusal test breaks one thing in it:
a count, the camera, one image, or a file cut short or given a trailing byte.

Each refusal is an ExpectedFailure whose message names the file it refuses and what its
requirement says, and holds no traceback (Req 1.7 to 1.11, 14.6). The byte sizes the cuts use are
worked out here from COLMAP's layout, not taken from colmap.py.

Property 1 draws whole models with Hypothesis, writes each one into its own
tempfile.TemporaryDirectory() and checks that read_model gives back what was written, every float
bit for bit (Req 1.21).
"""

import math
import struct
import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from splatpipe.errors import ExpectedFailure
from splatpipe.train.colmap import SparseModel, read_model

from camera_files import (CAMERAS_BIN, DEFAULT_CAMERA, DEFAULT_IMAGES, DEFAULT_POINTS, IMAGES_BIN,
                          MODEL_FILES, POINTS_BIN, CameraRecord, ImageRecord, PointRecord,
                          pack_image, write_model)

# What every Req 1.8 refusal says about its file.
CUT_SHORT = "is cut short or isn't a COLMAP binary model"

# The image each image refusal changes is the default scene's second, given an id that isn't its
# place in the file, so a message that gives the image's id can't be giving its place.
FIRST, SECOND, THIRD = DEFAULT_IMAGES
CHANGED_ID = 7
CHANGED = f"image {CHANGED_ID}"
CHANGED_NAME = repr(SECOND.name)  # names are shown with repr()
CAMERA = f"camera {DEFAULT_CAMERA.camera_id}"

# The sizes in COLMAP's layout that the cuts need, in bytes. Each file ends with its last record,
# so a cut of n bytes drops the last n bytes of that record.
COUNT_BYTES = struct.calcsize("<Q")  # a count, a keypoint count or a track length
PARAMS_BYTES = struct.calcsize("<4d")  # the camera's fx, fy, cx and cy, which end cameras.bin
IMAGE_HEAD_BYTES = struct.calcsize("<idddddddi")  # an image's id, pose and camera id
LAST_IMAGE, LAST_POINT = DEFAULT_IMAGES[-1], DEFAULT_POINTS[-1]
# The last image's keypoints, which end images.bin, and the last point's track, which ends
# points3D.bin.
KEYPOINTS_BYTES = struct.calcsize("<ddq") * len(LAST_IMAGE.keypoints)
TRACK_BYTES = struct.calcsize("<ii") * len(LAST_POINT.track)
# The last image's bytes after its head (its name, NUL, keypoint count and keypoints) and after
# its name (its NUL, keypoint count and keypoints).
AFTER_IMAGE_HEAD = len(pack_image(LAST_IMAGE)) - IMAGE_HEAD_BYTES
AFTER_NAME = 1 + COUNT_BYTES + KEYPOINTS_BYTES
# How the reader names the last record of each file in its messages.
LAST_IMAGE_PLACE = f"image {len(DEFAULT_IMAGES)} of {len(DEFAULT_IMAGES)}"
LAST_POINT_PLACE = f"point {len(DEFAULT_POINTS)} of {len(DEFAULT_POINTS)}"

HUGE = 2**60  # a count that no file holds


def images_with(**changes: Any) -> tuple[ImageRecord, ...]:
    """The default images with the second one changed: its id becomes CHANGED_ID and each field in
    `changes` takes its value."""
    return FIRST, replace(SECOND, image_id=CHANGED_ID, **changes), THIRD


def refused(sparse: Path, file: str) -> str:
    """The message of the ExpectedFailure that read_model raises on the model in `sparse`, once
    it's checked that the message names `file`, the file in `sparse` it refuses, and holds no
    traceback (Req 14.6)."""
    with pytest.raises(ExpectedFailure) as caught:
        read_model(sparse)
    message = str(caught.value)
    assert str(sparse / file) in message
    assert "Traceback" not in message
    return message


def plain(model: SparseModel) -> tuple[Any, ...]:
    """`model` with each array as its dtype, shape and bytes, so two models compare with ==, bit
    for bit."""
    def array(values: np.ndarray) -> tuple[str, tuple[int, ...], bytes]:
        return values.dtype.str, values.shape, values.tobytes()

    images = tuple((image.image_id, array(image.quaternion), array(image.translation),
                    image.camera_id, image.name) for image in model.images)
    return model.camera, images, array(model.points), array(model.colours)


# Req 1.6: other files in sparse/

def test_rigs_bin_and_frames_bin_holding_junk_change_nothing(tmp_path: Path) -> None:
    """The rigs.bin and frames.bin that COLMAP 3.12 and later write beside the three files are
    never read, so junk in them changes nothing (Req 1.6)."""
    sparse = tmp_path / "sparse"
    write_model(sparse)
    before = plain(read_model(sparse))
    for name in ("rigs.bin", "frames.bin"):
        (sparse / name).write_bytes(b"\xff\x00junk, not a COLMAP binary model\x01")
    assert plain(read_model(sparse)) == before


# Req 1.7 to 1.10: each refusal names its file and what its requirement says

def case(label: str, file: str, *says: str, **options: Any) -> Any:
    """One refusal: the file it names, the words its message must hold besides, and the
    write_model options that break the default scene."""
    return pytest.param(file, says, options, id=label)


REFUSALS = [
    # Req 1.7: a count of cameras other than 1, giving the count, and a model other than PINHOLE,
    # giving the camera and the model found.
    case("0 cameras", CAMERAS_BIN, "holds 0 cameras", counts={CAMERAS_BIN: 0}),
    case("2 cameras", CAMERAS_BIN, "holds 2 cameras", counts={CAMERAS_BIN: 2}),
    case("model id 0", CAMERAS_BIN, CAMERA, "model 0", "PINHOLE",
         camera=replace(DEFAULT_CAMERA, model_id=0)),
    case("model id 4", CAMERAS_BIN, CAMERA, "model 4", "PINHOLE",
         camera=replace(DEFAULT_CAMERA, model_id=4)),
    # Req 1.8: a file that ends inside each kind of record, cut 1 byte short of the end of the
    # part the message names, then a trailing byte after each file's last record, and a name with
    # no NUL, its file ending just before the NUL.
    case("cut in the camera's record", CAMERAS_BIN, CUT_SHORT, "ends inside the camera's record",
         cut={CAMERAS_BIN: PARAMS_BYTES + 1}),
    case("cut in the camera's parameters", CAMERAS_BIN, CUT_SHORT,
         "ends inside the camera's parameters", cut={CAMERAS_BIN: 1}),
    case("cut in an image's record", IMAGES_BIN, CUT_SHORT, f"ends inside {LAST_IMAGE_PLACE}",
         cut={IMAGES_BIN: AFTER_IMAGE_HEAD + 1}),
    case("cut in a keypoint count", IMAGES_BIN, CUT_SHORT,
         f"ends inside the keypoint count of {LAST_IMAGE_PLACE}",
         cut={IMAGES_BIN: KEYPOINTS_BYTES + 1}),
    case("cut in the keypoints", IMAGES_BIN, CUT_SHORT,
         f"ends inside the keypoints of {LAST_IMAGE_PLACE}", cut={IMAGES_BIN: 1}),
    case("cut in a point's record", POINTS_BIN, CUT_SHORT, f"ends inside {LAST_POINT_PLACE}",
         cut={POINTS_BIN: TRACK_BYTES + 1}),
    case("cut in a track", POINTS_BIN, CUT_SHORT, f"ends inside the track of {LAST_POINT_PLACE}",
         cut={POINTS_BIN: 1}),
    *(case(f"a trailing byte in {file}", file, CUT_SHORT, "1 byte after its last record",
           trailing_byte=(file,)) for file in MODEL_FILES),
    case("a name with no NUL", IMAGES_BIN, CUT_SHORT,
         f"the name of {LAST_IMAGE_PLACE} has no NUL", cut={IMAGES_BIN: AFTER_NAME}),
    # Req 1.9: a value that isn't above 0 or isn't finite, and a quaternion of length 0, naming
    # the camera or the image and the value.
    case("a width of 0", CAMERAS_BIN, CAMERA, "width = 0",
         camera=replace(DEFAULT_CAMERA, width=0)),
    case("an fx of -1", CAMERAS_BIN, CAMERA, "fx = -1.0", camera=replace(DEFAULT_CAMERA, fx=-1.0)),
    case("a NaN translation", IMAGES_BIN, CHANGED, CHANGED_NAME, "translation (nan, 0.0, 1.0)",
         images=images_with(translation=(math.nan, 0.0, 1.0))),
    case("a quaternion of four zeros", IMAGES_BIN, CHANGED, CHANGED_NAME,
         "quaternion (0.0, 0.0, 0.0, 0.0)", images=images_with(quaternion=(0.0, 0.0, 0.0, 0.0))),
    # Req 1.10: no image, an image of another camera and two images with one name, naming the
    # image or images.
    case("no image", IMAGES_BIN, "holds no image", images=()),
    case("an image of camera 2", IMAGES_BIN, CHANGED, CHANGED_NAME, "camera id 2",
         images=images_with(camera_id=2)),
    case("two images named a.jpg", IMAGES_BIN, f"images {FIRST.image_id} and {CHANGED_ID}",
         repr("a.jpg"), images=(replace(FIRST, name="a.jpg"), *images_with(name="a.jpg")[1:])),
]


@pytest.mark.parametrize(("file", "says", "options"), REFUSALS)
def test_each_refusal_names_its_file_and_what_its_requirement_says(
        tmp_path: Path, file: str, says: tuple[str, ...], options: dict[str, Any]) -> None:
    """Each refusal of Req 1.7 to 1.10 is an ExpectedFailure whose message names its file and
    what its requirement says, with no traceback (Req 1.7, 1.8, 1.9, 1.10, 14.6)."""
    sparse = tmp_path / "sparse"
    write_model(sparse, **options)
    message = refused(sparse, file)
    for words in says:
        assert words in message


@pytest.mark.parametrize("file", [IMAGES_BIN, POINTS_BIN])
def test_a_count_of_2_to_the_60_is_refused_at_once(tmp_path: Path, file: str) -> None:
    """A count of 2⁶⁰ images or points is refused against the bytes its smallest records need,
    before the reader loops over a record or makes room for one, so it returns at once (Req 1.8).
    A reader that looped would fail at the record after the file's last instead, which says
    "ends inside", not "count is"."""
    sparse = tmp_path / "sparse"
    write_model(sparse, counts={file: HUGE})
    start = time.perf_counter()
    message = refused(sparse, file)
    assert time.perf_counter() - start < 1
    assert CUT_SHORT in message
    assert f"count is {HUGE}" in message


# Req 1.11: an image's name, which must be a stem and .jpg with no /, \, : or NUL

BAD_NAMES = [
    pytest.param("x/y.jpg", id="a slash"),
    pytest.param("a\\b.jpg", id="a backslash"),
    pytest.param("c:.jpg", id="a colon"),
    pytest.param(".jpg", id="no stem"),
    pytest.param("a.JPG", id="JPG in capitals"),
    pytest.param(b"caf\xe9.jpg", id="bytes that aren't UTF-8"),  # café.jpg in Latin-1
]


@pytest.mark.parametrize("name", BAD_NAMES)
def test_a_bad_name_is_refused_giving_the_images_id_and_not_the_name(
        tmp_path: Path, name: str | bytes) -> None:
    """A name that isn't UTF-8 text of a stem and .jpg, or that holds /, \\ or :, is refused
    with the image's id, so a name can't lead outside images/ (Req 1.11, 14.6). The message never
    shows the name itself."""
    sparse = tmp_path / "sparse"
    write_model(sparse, images=images_with(name=name))
    message = refused(sparse, IMAGES_BIN)
    assert CHANGED in message
    assert repr(name) not in message


# Property 1: the round trip (Req 1.4, 1.5, 1.21)

# −0.0, subnormals and values near ±1e308, each drawn often beside Hypothesis's own finite floats,
# so a reader that lost a sign, a tiny value or a huge one fails.
EDGE_FLOATS = (-0.0, 5e-324, -5e-324, 1e-310, -1e-310, 1e308, -1e308, sys.float_info.max,
               -sys.float_info.max)
FINITE = st.sampled_from(EDGE_FLOATS) | st.floats(allow_nan=False, allow_infinity=False)
ABOVE_0 = (st.sampled_from([value for value in EDGE_FLOATS if value > 0])
           | st.floats(min_value=0.0, exclude_min=True, allow_infinity=False))
# A <i field: a camera id, an image id, or a track entry's image id or keypoint index.
INT32 = st.integers(-2**31, 2**31 - 1)
BYTE = st.integers(0, 255)
# A stem of 1 to 12 characters with no /, \, :, NUL or surrogate, which UTF-8 can't encode.
NAMES = st.text(st.characters(exclude_characters="/\\:\x00", exclude_categories=("Cs",)),
                min_size=1, max_size=12).map(lambda stem: f"{stem}.jpg")

CAMERAS = st.builds(CameraRecord, camera_id=INT32, width=st.integers(1, 10_000),
                    height=st.integers(1, 10_000), fx=ABOVE_0, fy=ABOVE_0, cx=FINITE, cy=FINITE)
KEYPOINTS = st.tuples(FINITE, FINITE, st.integers(-1, 2**63 - 1))  # x, y, point id (<q)
POINTS = st.builds(PointRecord, point_id=st.integers(0, 2**64 - 1),
                   position=st.tuples(FINITE, FINITE, FINITE), colour=st.tuples(BYTE, BYTE, BYTE),
                   error=FINITE, track=st.lists(st.tuples(INT32, INT32), max_size=5).map(tuple))


def images_of(camera_id: int) -> st.SearchStrategy[ImageRecord]:
    """Images of the camera `camera_id`, each with a quaternion that isn't four zeros and 0 to 40
    keypoints."""
    quaternions = st.tuples(FINITE, FINITE, FINITE, FINITE).filter(
        lambda quaternion: any(value != 0 for value in quaternion))
    return st.builds(ImageRecord, image_id=INT32, quaternion=quaternions,
                     translation=st.tuples(FINITE, FINITE, FINITE), camera_id=st.just(camera_id),
                     name=NAMES, keypoints=st.lists(KEYPOINTS, max_size=40).map(tuple))


@st.composite
def models(draw: st.DrawFn) -> tuple[CameraRecord, list[ImageRecord], list[PointRecord]]:
    """A PINHOLE camera, 1 to 6 of its images with names of their own, and 0 to 60 points."""
    camera = draw(CAMERAS)
    images = draw(st.lists(images_of(camera.camera_id), min_size=1, max_size=6,
                           unique_by=lambda image: image.name))
    return camera, images, draw(st.lists(POINTS, max_size=60))


def bits(values: Any) -> np.ndarray:
    """The 64-bit patterns of `values` as float64, so −0.0 differs from 0.0 and every bit counts."""
    return np.array(values, dtype=np.float64).view(np.uint64)


@settings(max_examples=100, deadline=None)
@given(model=models())
def test_the_model_reader_reads_back_what_was_written(
        model: tuple[CameraRecord, list[ImageRecord], list[PointRecord]]) -> None:
    """Feature: splat-training, Property 1: The Model_Reader reads back what was written

    **Validates: Requirements 1.21, 1.4, 1.5**
    """
    camera, images, points = model
    with tempfile.TemporaryDirectory() as folder:
        sparse = Path(folder) / "sparse"
        write_model(sparse, camera, images, points)
        read = read_model(sparse)

    # The Shared_Camera's id, size and parameters (Req 1.4).
    got = read.camera
    assert (got.camera_id, got.width, got.height) == (camera.camera_id, camera.width,
                                                      camera.height)
    assert np.array_equal(bits([got.fx, got.fy, got.cx, got.cy]),
                          bits([camera.fx, camera.fy, camera.cx, camera.cy]))

    # Every image in order, which only holds if each one's keypoints were read past (Req 1.4, 1.5).
    assert len(read.images) == len(images)
    for image, wrote in zip(read.images, images):
        assert (image.image_id, image.camera_id, image.name) == (wrote.image_id, wrote.camera_id,
                                                                 wrote.name)
        assert np.array_equal(image.quaternion.view(np.uint64), bits(wrote.quaternion))
        assert np.array_equal(image.translation.view(np.uint64), bits(wrote.translation))

    # Every point's position and colour in order, which only holds if each one's id, error and
    # track were read past (Req 1.4, 1.5).
    positions = bits([point.position for point in points]).reshape(len(points), 3)
    assert np.array_equal(read.points.view(np.uint64), positions)
    colours = np.array([point.colour for point in points], dtype=np.uint8).reshape(len(points), 3)
    assert read.colours.dtype == np.uint8
    assert np.array_equal(read.colours, colours)
