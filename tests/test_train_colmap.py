"""Tests for the Model_Reader: splatpipe/train/colmap.py.

Every model is written by tests/camera_files.py's write_model, the tests' own writer of COLMAP's
binary layout, which imports nothing from colmap.py (Req 14.1). The unit tests write into the
test's tmp_path. Their default scene is a good model: one 8 x 6 PINHOLE camera, 3 images with 12
keypoints each and 12 points with 3 track entries each. Each refusal test breaks one thing in it:
a count, the camera, one image, or a file cut short or given a trailing byte.

Each refusal is an ExpectedFailure whose message names the file it refuses and what its
requirement says, and holds no traceback (Req 1.7 to 1.11, 14.6). The byte sizes the cuts use are
worked out here from COLMAP's layout, not taken from colmap.py.

The point id tests check the depth-priors spec's Req 2.4 to 2.6: each image's observed_ids are
the distinct ids among its keypoints that are ids of points, in first-keypoint order; a keypoint
id of −1 or of no point is no observation and refuses no model, so every refusal above gives the
same message with such keypoints; ids just above 2⁵³ are never compared as floats; and
observed_rows maps each id to its first point's row. The keypoints that keypoints_with_ids
makes have an x of NaN and a y of infinity, which the reader reads past and never checks.

Property 1 draws whole models with Hypothesis, writes each one into its own
tempfile.TemporaryDirectory() and checks that read_model gives back what was written, every float
bit for bit (Req 1.21). The depth-priors spec widens it (its Req 2.9): point ids from 0 to
2⁶³ − 1, a few at 2⁶³ and above, which no keypoint can name, and repeated ids now and then;
keypoint ids of −1, of points, of no point and repeated within an image. Every point id must come
back, and each image's observed_ids must equal an oracle's, which walks the image's keypoints in
order with a set and never calls the reader's helpers.
"""

import math
import struct
import sys
import tempfile
import time
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from splatpipe.errors import ExpectedFailure
from splatpipe.train.colmap import SparseModel, observed_rows, read_model

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
KEYPOINT_BYTES = struct.calcsize("<ddq")  # one keypoint: x, y and point id
PARAMS_BYTES = struct.calcsize("<4d")  # the camera's fx, fy, cx and cy, which end cameras.bin
IMAGE_HEAD_BYTES = struct.calcsize("<idddddddi")  # an image's id, pose and camera id
LAST_IMAGE, LAST_POINT = DEFAULT_IMAGES[-1], DEFAULT_POINTS[-1]
# The last image's keypoints, which end images.bin, and the last point's track, which ends
# points3D.bin.
KEYPOINTS_BYTES = KEYPOINT_BYTES * len(LAST_IMAGE.keypoints)
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


# Depth-priors spec, Req 2.4 to 2.6: the point ids and each image's Observed_Points

POINT_IDS = [point.point_id for point in DEFAULT_POINTS]  # 1 to 12, one per keypoint
NO_POINT = 1_000  # an id that no point of the default scene has
BIG = 2**53  # 2⁵³ + 1 is 2⁵³ as a float64


def keypoints_with_ids(ids: Sequence[int]) -> tuple[tuple[float, float, int], ...]:
    """One keypoint for each of `ids`, in order, each at an x of NaN and a y of infinity, which
    the reader reads past and never checks or keeps (depth-priors spec, Req 2.6)."""
    return tuple((math.nan, math.inf, point_id) for point_id in ids)


def with_ids_of_no_point(image: ImageRecord) -> ImageRecord:
    """`image` with its keypoints' ids taken in turn from −1, NO_POINT and 1, so it holds −1s,
    ids of no point and one id many times, with its keypoints' count, x and y unchanged."""
    ids = (-1, NO_POINT, 1)
    return replace(image, keypoints=tuple((x, y, ids[place % len(ids)])
                                          for place, (x, y, _) in enumerate(image.keypoints)))


def test_observed_ids_are_the_distinct_ids_of_points_in_first_keypoint_order(
        tmp_path: Path) -> None:
    """An image whose keypoints hold −1, ids of no point (0, NO_POINT and 2⁶³ − 1) and the same
    ids more than once is read with no refusal. Its observed_ids are the distinct ids of points,
    as uint64, in the order of their first keypoint (depth-priors spec, Req 2.4, 2.5, 2.6)."""
    ids = (5, -1, NO_POINT, 3, 5, 0, -1, 3, 2**63 - 1, 1, NO_POINT, 5)
    sparse = tmp_path / "sparse"
    write_model(sparse, images=images_with(keypoints=keypoints_with_ids(ids)))
    model = read_model(sparse)

    observed = model.images[1].observed_ids
    assert observed.dtype == np.uint64
    assert observed.tolist() == [5, 3, 1]
    # The other images' keypoints name every point once, in the points' order.
    for image in (model.images[0], model.images[2]):
        assert image.observed_ids.tolist() == POINT_IDS
    assert model.point_ids.dtype == np.uint64
    assert model.point_ids.tolist() == POINT_IDS


def test_a_model_whose_keypoints_are_all_minus_1_is_read_with_no_observed_points(
        tmp_path: Path) -> None:
    """A model whose every keypoint id is −1 is read with every observed_ids an empty uint64
    array, and the rest of it as the default scene (depth-priors spec, Req 2.4, 2.5)."""
    default, sparse = tmp_path / "default", tmp_path / "sparse"
    write_model(default)
    unseen = [replace(image, keypoints=keypoints_with_ids([-1] * len(image.keypoints)))
              for image in DEFAULT_IMAGES]
    write_model(sparse, images=unseen)
    model = read_model(sparse)

    for image in model.images:
        assert image.observed_ids.dtype == np.uint64
        assert image.observed_ids.shape == (0,)
    assert plain(model) == plain(read_model(default))
    assert model.point_ids.tolist() == POINT_IDS


@pytest.mark.parametrize(("file", "says", "options"), REFUSALS)
def test_keypoints_of_no_point_change_no_refusal(
        tmp_path: Path, file: str, says: tuple[str, ...], options: dict[str, Any]) -> None:
    """Each splat-training refusal gives the same message, naming its file with no traceback,
    when every image's keypoints hold −1s, ids of no point and one id many times. So no model is
    refused for a keypoint's point id, and the reader refuses exactly the models it refused before
    (depth-priors spec, Req 2.5)."""
    sparse = tmp_path / "sparse"
    write_model(sparse, **options)
    before = refused(sparse, file)
    images = [with_ids_of_no_point(image) for image in options.get("images", DEFAULT_IMAGES)]
    write_model(sparse, **{**options, "images": images})
    message = refused(sparse, file)
    assert message == before
    for words in says:
        assert words in message


@pytest.mark.parametrize("cut", [1, KEYPOINT_BYTES, KEYPOINTS_BYTES],
                         ids=["1 byte", "one keypoint", "every keypoint"])
def test_a_keypoint_block_cut_short_is_refused_as_cut_short(tmp_path: Path, cut: int) -> None:
    """images.bin ending 1 byte, one whole keypoint or every keypoint short of the last image's
    keypoints is refused with the reader's cut short message, naming images.bin, before any
    keypoint is read (splat-training Req 1.8; depth-priors spec, Req 2.6)."""
    sparse = tmp_path / "sparse"
    write_model(sparse, images=[with_ids_of_no_point(image) for image in DEFAULT_IMAGES],
                cut={IMAGES_BIN: cut})
    message = refused(sparse, IMAGES_BIN)
    assert CUT_SHORT in message
    assert f"ends inside the keypoints of {LAST_IMAGE_PLACE}" in message


def test_a_keypoint_count_of_2_to_the_60_is_refused_at_once(tmp_path: Path) -> None:
    """An image whose keypoint count is 2⁶⁰, with one keypoint after it, is refused against the
    bytes its keypoints need before any is read, so it returns at once with the cut short
    message naming images.bin (splat-training Req 1.8; depth-priors spec, Req 2.6)."""
    sparse = tmp_path / "sparse"
    write_model(sparse)
    record = pack_image(replace(FIRST, keypoints=keypoints_with_ids([1])))
    count_at = len(record) - KEYPOINT_BYTES - COUNT_BYTES
    record = record[:count_at] + struct.pack("<Q", HUGE) + record[count_at + COUNT_BYTES:]
    (sparse / IMAGES_BIN).write_bytes(struct.pack("<Q", 1) + record)
    start = time.perf_counter()
    message = refused(sparse, IMAGES_BIN)
    assert time.perf_counter() - start < 1
    assert CUT_SHORT in message
    assert "ends inside the keypoints of image 1 of 1" in message


def test_ids_just_above_2_to_the_53_are_never_compared_as_floats(tmp_path: Path) -> None:
    """With a point whose id is 2⁵³, a keypoint id of 2⁵³ + 1, which names no point, is no
    observation. As float64 the two ids are equal, so a reader that compared them as floats
    would count it. The first point's old id, 1, names no point now either (depth-priors spec,
    Req 2.4, 2.5)."""
    points = (replace(DEFAULT_POINTS[0], point_id=BIG), *DEFAULT_POINTS[1:])
    images = (replace(FIRST, keypoints=keypoints_with_ids([BIG + 1])),
              replace(SECOND, keypoints=keypoints_with_ids([BIG + 1, 2, BIG, BIG + 1])), THIRD)
    sparse = tmp_path / "sparse"
    write_model(sparse, images=images, points=points)
    model = read_model(sparse)

    assert model.point_ids.tolist() == [BIG, *POINT_IDS[1:]]
    assert model.images[0].observed_ids.tolist() == []
    assert model.images[1].observed_ids.tolist() == [2, BIG]
    assert model.images[2].observed_ids.tolist() == POINT_IDS[1:]
    assert observed_rows(model, model.images[1]).tolist() == [1, 0]


def test_observed_rows_map_ids_to_rows_and_a_repeated_id_to_its_first_row(
        tmp_path: Path) -> None:
    """observed_rows gives the row of each Observed_Point, as int64 in observed_ids' order, and
    a point id written twice maps to its first row. An image with no Observed_Point gives no row
    (depth-priors spec, Req 2.4)."""
    ids = (10, 30, 20, 30, 40)  # 30 is written twice, at rows 1 and 3
    points = tuple(replace(point, point_id=point_id)
                   for point, point_id in zip(DEFAULT_POINTS, ids))
    images = (replace(FIRST, keypoints=keypoints_with_ids([40, 30, -1, 10, 30, 20])),
              replace(SECOND, keypoints=keypoints_with_ids([-1, NO_POINT])), THIRD)
    sparse = tmp_path / "sparse"
    write_model(sparse, images=images, points=points)
    model = read_model(sparse)

    assert model.point_ids.tolist() == list(ids)
    assert model.images[0].observed_ids.tolist() == [40, 30, 10, 20]
    rows = observed_rows(model, model.images[0])
    assert rows.dtype == np.int64
    assert rows.tolist() == [4, 1, 0, 2]
    none = observed_rows(model, model.images[1])
    assert none.dtype == np.int64
    assert none.shape == (0,)


# Property 1: the round trip (Req 1.4, 1.5, 1.21; depth-priors spec, Req 2.4, 2.5, 2.6, 2.9)

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
QUATERNIONS = st.tuples(FINITE, FINITE, FINITE, FINITE).filter(
    lambda quaternion: any(value != 0 for value in quaternion))
# A point with its id left at 0, which models replaces with one that point_ids draws.
UNNAMED_POINTS = st.builds(PointRecord, point_id=st.just(0),
                           position=st.tuples(FINITE, FINITE, FINITE),
                           colour=st.tuples(BYTE, BYTE, BYTE), error=FINITE,
                           track=st.lists(st.tuples(INT32, INT32), max_size=5).map(tuple))

# A keypoint's point id is <q, so no keypoint names a point id above 2⁶³ − 1. A point's id is
# <Q, and a few are drawn at 2⁶³ and above: 2⁶³, 2⁶⁴ − 1, which is −1 as an int64, and
# Hypothesis's own.
KEYPOINT_ID_MAX = 2**63 - 1
HIGH_POINT_IDS = st.sampled_from([2**63, 2**64 - 1]) | st.integers(2**63, 2**64 - 1)
# The kinds of keypoint id that keypoint_ids draws. −1 comes first, so Hypothesis shrinks a
# keypoint towards no observation.
KEYPOINT_KINDS = ("-1", "a point's", "near a point's", "of no point", "again")


@st.composite
def point_ids(draw: st.DrawFn, count: int) -> list[int]:
    """`count` point ids: most from 0 to 2⁶³ − 1, now and then one at 2⁶³ or above and now and
    then a copy of an id drawn before."""
    ids: list[int] = []
    for _ in range(count):
        kind = draw(st.integers(0, 9))
        if kind == 9 and ids:
            ids.append(draw(st.sampled_from(tuple(ids))))
        elif kind == 8:
            ids.append(draw(HIGH_POINT_IDS))
        else:
            ids.append(draw(st.integers(0, KEYPOINT_ID_MAX)))
    return ids


def no_point_id(candidate: int, taken: set[int]) -> int:
    """`candidate` brought into 0 to 2⁶³ − 1, then moved up past every id in `taken`, wrapping to
    0, so it's a keypoint id that names no point."""
    candidate %= KEYPOINT_ID_MAX + 1
    while candidate in taken:
        candidate = (candidate + 1) % (KEYPOINT_ID_MAX + 1)
    return candidate


@st.composite
def keypoint_ids(draw: st.DrawFn, ids_of_points: Sequence[int]) -> list[int]:
    """0 to 40 keypoint ids for one image. Each is −1; the id of a point that a keypoint can
    name; an id of no point within 2 of a point's id, or of its low 63 bits for an id at 2⁶³ or
    above; another id of no point; or an id this image's keypoints already hold. A kind that has
    nothing to draw from gives −1."""
    nameable = tuple(point_id for point_id in ids_of_points if point_id <= KEYPOINT_ID_MAX)
    taken = set(ids_of_points)
    ids: list[int] = []
    for _ in range(draw(st.integers(0, 40))):
        kind = draw(st.sampled_from(KEYPOINT_KINDS))
        if kind == "a point's" and nameable:
            ids.append(draw(st.sampled_from(nameable)))
        elif kind == "near a point's" and ids_of_points:
            near = draw(st.sampled_from(tuple(ids_of_points))) + draw(st.integers(-2, 2))
            ids.append(no_point_id(near, taken))
        elif kind == "of no point":
            ids.append(no_point_id(draw(st.integers(0, KEYPOINT_ID_MAX)), taken))
        elif kind == "again" and ids:
            ids.append(draw(st.sampled_from(tuple(ids))))
        else:
            ids.append(-1)
    return ids


@st.composite
def images_of(draw: st.DrawFn, camera_id: int, ids_of_points: Sequence[int]) -> ImageRecord:
    """An image of the camera `camera_id` with a quaternion that isn't four zeros and 0 to 40
    keypoints, whose ids keypoint_ids draws against `ids_of_points`."""
    ids = draw(keypoint_ids(ids_of_points))
    keypoints = tuple((draw(FINITE), draw(FINITE), point_id) for point_id in ids)
    return ImageRecord(image_id=draw(INT32), quaternion=draw(QUATERNIONS),
                       translation=draw(st.tuples(FINITE, FINITE, FINITE)), camera_id=camera_id,
                       name=draw(NAMES), keypoints=keypoints)


@st.composite
def models(draw: st.DrawFn) -> tuple[CameraRecord, list[ImageRecord], list[PointRecord]]:
    """A PINHOLE camera, 0 to 60 points whose ids point_ids draws, and 1 to 6 of the camera's
    images with names of their own, whose keypoints name those points, −1 and ids of no point."""
    camera = draw(CAMERAS)
    unnamed = draw(st.lists(UNNAMED_POINTS, max_size=60))
    ids = draw(point_ids(len(unnamed)))
    points = [replace(point, point_id=point_id) for point, point_id in zip(unnamed, ids)]
    images = draw(st.lists(images_of(camera.camera_id, ids), min_size=1, max_size=6,
                           unique_by=lambda image: image.name))
    return camera, images, points


def observed_oracle(image: ImageRecord, points: Sequence[PointRecord]) -> list[int]:
    """The Observed_Points of the depth-priors spec's Req 2.4 for `image`: its keypoints' ids
    walked in order, keeping each one that is a point's id and wasn't kept before. It works on
    Python ints, so no id is compared as a float, and calls nothing of the reader's."""
    of_points = {point.point_id for point in points}
    seen: set[int] = set()
    kept: list[int] = []
    for _, _, point_id in image.keypoints:
        if point_id in of_points and point_id not in seen:
            seen.add(point_id)
            kept.append(point_id)
    return kept


def bits(values: Any) -> np.ndarray:
    """The 64-bit patterns of `values` as float64, so −0.0 differs from 0.0 and every bit counts."""
    return np.array(values, dtype=np.float64).view(np.uint64)


@settings(max_examples=100, deadline=None)
@given(model=models())
def test_the_model_reader_reads_back_what_was_written(
        model: tuple[CameraRecord, list[ImageRecord], list[PointRecord]]) -> None:
    """Feature: splat-training, Property 1: The Model_Reader reads back what was written

    **Validates: Requirements 1.21, 1.4, 1.5** (splat-training spec)

    Feature: depth-priors, Property 1: The Model_Reader reads back what was written, point ids
    and Observed_Points included

    **Validates: Requirements 2.9, 2.4, 2.5, 2.6**
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

    # Every image in order, which only holds if each one's keypoints were read past right (Req
    # 1.4, 1.5), and its Observed_Points as the oracle gives them, as uint64 (depth-priors spec,
    # Req 2.4, 2.5, 2.6).
    assert len(read.images) == len(images)
    for image, wrote in zip(read.images, images):
        assert (image.image_id, image.camera_id, image.name) == (wrote.image_id, wrote.camera_id,
                                                                 wrote.name)
        assert np.array_equal(image.quaternion.view(np.uint64), bits(wrote.quaternion))
        assert np.array_equal(image.translation.view(np.uint64), bits(wrote.translation))
        assert image.observed_ids.dtype == np.uint64
        assert image.observed_ids.tolist() == observed_oracle(wrote, points)

    # Every point id in order, as uint64, the ones at 2⁶³ and above and the repeated ones included
    # (depth-priors spec, Req 2.4).
    assert read.point_ids.dtype == np.uint64
    assert read.point_ids.shape == (len(points),)
    assert read.point_ids.tolist() == [point.point_id for point in points]

    # Every point's position and colour in order, which only holds if each one's error and track
    # were read past right (Req 1.4, 1.5; depth-priors spec, Req 2.6).
    positions = bits([point.position for point in points]).reshape(len(points), 3)
    assert np.array_equal(read.points.view(np.uint64), positions)
    colours = np.array([point.colour for point in points], dtype=np.uint8).reshape(len(points), 3)
    assert read.colours.dtype == np.uint8
    assert np.array_equal(read.colours, colours)
