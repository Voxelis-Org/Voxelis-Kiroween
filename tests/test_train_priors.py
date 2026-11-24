"""Tests for splatpipe/train/priors.py: the Prior_Format's writer, its reader and its checks.

Every unit test starts from one good Prior_Set, written into the test's tmp_path by write_prior and
write_index: two Priors of 4 × 3 pixels for two of the three images of a hand-built Sparse_Model,
whose camera, poses and arrays are this file's own constants. read_prior_set must read it back,
and gives None when there's no depth.json, even with Prior_Files there (Req 10.2).

Each check of Req 10.4 is then broken once, either by editing depth.json's JSON in the test or by
writing the second image's .npz with np.savez directly, and read_prior_set must raise a
BrokenPriors whose path is the file that failed and whose `what` says what failed, worded as the
design's Error Handling gives it. A pose changed by one float step is refused, so poses are
compared value for value.

For Req 7.5, the .npz holds an object array, or the file is a plain pickle, whose unpickling makes
a folder. The reader must refuse it and the folder must not appear, while np.load with
allow_pickle=True, or pickle.loads, makes it, so the payload is live. For Req 7.6, os.replace is
patched to fail with ENOSPC, and write_index must leave neither depth.json nor depth.json.tmp.

Property 4 draws whole Prior_Sets with Hypothesis: 1 to 4 Priors of one Prior_Size from 1 × 1 to
8 × 8, with any finite float32 values and any mask, and a Prior_Index whose floats include −0.0,
subnormals and values near ±1e308. The names are lowercase ASCII, digits, _ and -, starting with a
digit, so each Prior_File's name is a file name on Windows too, and no two fold to one file.
write_prior and write_index write each example into its own tempfile.TemporaryDirectory(), since
tmp_path is shared by all of a test's examples. load_index and load_prior must give back every
index value, each float's 8 bytes compared through struct.pack, and every array with its shape,
its dtype and its bits, and each .npz must hold exactly the three arrays (Req 7.1, 7.3, 7.12).
"""

import errno
import json
import os
import pickle
import string
import struct
import sys
import tempfile
from collections.abc import Callable
from dataclasses import astuple
from pathlib import Path
from typing import Any, NoReturn

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from splatpipe.train.colmap import Camera, ModelImage, SparseModel
from splatpipe.train.priors import (
    PRIOR_INDEX,
    BrokenPriors,
    Prior,
    PriorEntry,
    PriorIndex,
    load_index,
    load_prior,
    prior_path,
    read_prior_set,
    write_index,
    write_prior,
)

# The Shared_Camera and the Image_Poses of the hand-built Sparse_Model, as images.bin would give
# them: quaternions w x y z, not normalised, and translations.
CAMERA = Camera(camera_id=1, width=64, height=48, fx=50.0, fy=50.5, cx=32.0, cy=24.0)
POSES = {
    "000012.jpg": ((0.9, 0.1, -0.2, 0.3), (0.5, -1.25, 3.0)),
    "000013.jpg": ((0.7, -0.3, 0.4, 0.1), (-2.0, 0.75, 1.5)),
    "000014.jpg": ((1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
}
PRIOR_NAMES = ("000012.jpg", "000013.jpg")  # the images with a Prior; 000014.jpg has none
WIDTH, HEIGHT = 4, 3  # the Prior_Size of every good Prior

# The Prior_Index's fields, written here from Req 7.3.
INDEX_KEYS = {"format", "camera", "prior_width", "prior_height", "metres_per_unit", "priors"}
CAMERA_KEYS = {"width", "height", "fx", "fy", "cx", "cy"}
ENTRY_KEYS = {"name", "quaternion", "translation", "scale", "offset", "fit_points",
              "kept_points", "median_error", "metres_per_unit"}


def sparse_model() -> SparseModel:
    """The Sparse_Model every test's Priors are checked against: CAMERA, the three images of
    POSES in their order, and three points."""
    images = tuple(
        ModelImage(image_id=place + 1, quaternion=np.array(quaternion, dtype=np.float64),
                   translation=np.array(translation, dtype=np.float64), camera_id=1, name=name,
                   observed_ids=np.array([1, 2], dtype=np.uint64))
        for place, (name, (quaternion, translation)) in enumerate(POSES.items()))
    points = np.array([[0.0, 0.0, 3.0], [1.0, 0.5, 4.0], [-1.0, 0.25, 5.0]], dtype=np.float64)
    colours = np.array([[200, 10, 10], [10, 200, 10], [10, 10, 200]], dtype=np.uint8)
    return SparseModel(CAMERA, images, points, colours, np.array([1, 2, 3], dtype=np.uint64))


def good_arrays(place: int) -> dict[str, np.ndarray]:
    """The three arrays of the good Prior of the image at `place` in PRIOR_NAMES: inverse_depth
    above 0 in the left two columns and 0 in the right two, the mask true exactly there, and a
    unit normal facing the camera there and 0s elsewhere."""
    inverse_depth = np.array([[0.5, 0.25, 0.0, 0.0],
                              [1.0, 2.0, 0.0, 0.0],
                              [0.125, 4.0, 0.0, 0.0]], dtype=np.float32) * np.float32(place + 1)
    mask = inverse_depth > 0
    normal = np.zeros((HEIGHT, WIDTH, 3), dtype=np.float32)
    normal[mask] = (0.0, 0.6, -0.8)
    return {"inverse_depth": inverse_depth, "normal": normal, "mask": mask}


def good_index() -> PriorIndex:
    """The good Prior_Index: CAMERA, the Prior_Size, and one entry for each of PRIOR_NAMES with
    its pose from POSES."""
    entries = tuple(
        PriorEntry(name=name, quaternion=POSES[name][0], translation=POSES[name][1],
                   scale=2.5 + place, offset=-0.125, fit_points=40 + place, kept_points=38,
                   median_error=0.03125, metres_per_unit=1.75)
        for place, name in enumerate(PRIOR_NAMES))
    camera = (CAMERA.width, CAMERA.height, CAMERA.fx, CAMERA.fy, CAMERA.cx, CAMERA.cy)
    return PriorIndex(camera=camera, prior_width=WIDTH, prior_height=HEIGHT,
                      metres_per_unit=1.75, priors=entries)


def empty_folder(tmp_path: Path) -> Path:
    """An empty tmp_path / "depth", standing for work/depth/."""
    folder = tmp_path / "depth"
    folder.mkdir()
    return folder


def write_good_set(tmp_path: Path, *, index: bool = True) -> Path:
    """Write the good Prior_Set to tmp_path / "depth", its Prior_Files and then, if `index`, its
    Prior_Index, and give the folder."""
    folder = empty_folder(tmp_path)
    for place, name in enumerate(PRIOR_NAMES):
        write_prior(prior_path(folder, name), Prior(**good_arrays(place)))
    if index:
        write_index(folder, good_index())
    return folder


def save_arrays(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Write `arrays` to the .npz file at `path` with np.savez itself, so no dtype is changed."""
    with path.open("wb") as file:
        np.savez(file, **arrays)


def refused(folder: Path, path: Path) -> str:
    """The `what` of the BrokenPriors that read_prior_set(folder, ...) raises, once it's checked
    that the error names the file at `path` first and holds no traceback."""
    with pytest.raises(BrokenPriors) as caught:
        read_prior_set(folder, sparse_model())
    error = caught.value
    assert error.path == path
    assert str(error) == f"{path} {error.what}"
    assert "Traceback" not in str(error)
    return error.what


# The good set, and no index (Req 10.2)

def test_read_prior_set_reads_the_good_set_back(tmp_path: Path) -> None:
    prior_set = read_prior_set(write_good_set(tmp_path), sparse_model())
    assert prior_set is not None
    assert prior_set.index == good_index()
    assert list(prior_set.priors) == list(PRIOR_NAMES)
    for place, name in enumerate(PRIOR_NAMES):
        for label, array in good_arrays(place).items():
            found = getattr(prior_set.priors[name], label)
            assert found.dtype == array.dtype
            assert np.array_equal(found, array)


@pytest.mark.parametrize("make", [
    pytest.param(lambda tmp_path: tmp_path / "depth", id="no folder"),
    pytest.param(empty_folder, id="an empty folder"),
    pytest.param(lambda tmp_path: write_good_set(tmp_path, index=False),
                 id="prior files but no index"),
])
def test_read_prior_set_gives_none_with_no_depth_json(tmp_path: Path,
                                                      make: Callable[[Path], Path]) -> None:
    assert read_prior_set(make(tmp_path), sparse_model()) is None


# The index's checks (Req 10.4)

def edit_index(folder: Path, change: Callable[[Any], None]) -> None:
    """Read depth.json in `folder` as JSON, apply `change` to it and write it back."""
    path = folder / PRIOR_INDEX
    value = json.loads(path.read_text(encoding="utf-8"))
    change(value)
    path.write_text(json.dumps(value), encoding="utf-8")


def set_at(keys: tuple[str | int, ...], value: object) -> Callable[[Any], None]:
    """A change that sets the field at `keys`, like ("priors", 1, "name"), to `value`."""
    def change(index: Any) -> None:
        holder = index
        for key in keys[:-1]:
            holder = holder[key]
        holder[keys[-1]] = value
    return change


CAMERA_CHANGES = [("width", 65), ("height", 47), ("fx", 50.25), ("fy", 50.75), ("cx", 31.5),
                  ("cy", 24.5)]


@pytest.mark.parametrize(("change", "what"), [
    pytest.param(set_at(("format",), 2), "has format 2, but the train step reads only format 1",
                 id="format 2"),
    pytest.param(set_at(("format",), True),
                 "has format true, but the train step reads only format 1", id="format true"),
    *(pytest.param(set_at(("camera", label), value),
                   f"has the camera {label} {value!r}, but cameras.bin's is "
                   f"{getattr(CAMERA, label)!r}", id=f"camera {label}")
      for label, value in CAMERA_CHANGES),
    pytest.param(set_at(("prior_width",), 0),
                 "has prior_width 0, but it must be a whole number above 0", id="prior_width 0"),
    pytest.param(set_at(("prior_height",), 0),
                 "has prior_height 0, but it must be a whole number above 0",
                 id="prior_height 0"),
    pytest.param(set_at(("priors",), []), "lists no priors, but a prior index lists one or more",
                 id="no priors"),
    pytest.param(set_at(("priors", 1, "name"), "x.jpg"),
                 "names the image 'x.jpg', which images.bin doesn't hold", id="an unknown name"),
    pytest.param(set_at(("priors", 1, "name"), "000012.jpg"),
                 "names the image '000012.jpg' twice, but an image has one prior at most",
                 id="a repeated name"),
    # The reader takes no path from the index, so a prior with a path field is refused (A13).
    pytest.param(set_at(("priors", 0, "path"), "000012.npz"),
                 "holds the fields ['fit_points', 'kept_points', 'median_error', "
                 "'metres_per_unit', 'name', 'offset', 'path', 'quaternion', 'scale', "
                 "'translation'] in priors[0], but each prior holds exactly name, quaternion, "
                 "translation, scale, offset, fit_points, kept_points, median_error and "
                 "metres_per_unit", id="a path in a prior"),
])
def test_an_index_that_fails_a_check_is_refused(tmp_path: Path, change: Callable[[Any], None],
                                                what: str) -> None:
    folder = write_good_set(tmp_path)
    edit_index(folder, change)
    assert refused(folder, folder / PRIOR_INDEX) == what


@pytest.mark.parametrize(("label", "place"), [
    *(("quaternion", place) for place in range(4)),
    *(("translation", place) for place in range(3)),
])
def test_a_pose_one_float_step_off_is_refused(tmp_path: Path, label: str, place: int) -> None:
    name = PRIOR_NAMES[1]
    value = POSES[name][0 if label == "quaternion" else 1][place]
    folder = write_good_set(tmp_path)
    edit_index(folder, set_at(("priors", 1, label, place), float(np.nextafter(value, np.inf))))
    assert refused(folder, folder / PRIOR_INDEX) == (
        f"has the pose of an earlier camera solve for '{name}': its {label} isn't the one "
        "images.bin gives")


@pytest.mark.parametrize(("text", "what"), [
    pytest.param('{"format": 1,', "can't be read as JSON (", id="JSON cut short"),
    pytest.param("[]", "holds [], but a prior index is a JSON object", id="a list"),
])
def test_an_index_that_isnt_a_json_object_is_refused(tmp_path: Path, text: str,
                                                     what: str) -> None:
    folder = write_good_set(tmp_path)
    (folder / PRIOR_INDEX).write_text(text, encoding="utf-8")
    assert refused(folder, folder / PRIOR_INDEX).startswith(what)


# Each Prior_File's checks (Req 10.4)

def replaced(name: str, array: np.ndarray) -> dict[str, np.ndarray]:
    """The second image's good arrays, with the array `name` replaced by `array`."""
    return {**good_arrays(1), name: array}


def at_pixel(name: str, where: tuple[int, ...], value: float) -> dict[str, np.ndarray]:
    """The second image's good arrays, with the value at `where` in the array `name` set to
    `value`."""
    arrays = good_arrays(1)
    arrays[name][where] = value
    return arrays


GOOD = good_arrays(1)
ARRAYS_HELD = "but a prior file holds exactly inverse_depth, normal and mask"


@pytest.mark.parametrize(("arrays", "what"), [
    pytest.param({**GOOD, "depth": np.zeros((HEIGHT, WIDTH), dtype=np.float32)},
                 f"holds the arrays ['depth', 'inverse_depth', 'mask', 'normal'], {ARRAYS_HELD}",
                 id="a fourth array"),
    pytest.param({"inverse_depth": GOOD["inverse_depth"], "mask": GOOD["mask"]},
                 f"holds the arrays ['inverse_depth', 'mask'], {ARRAYS_HELD}", id="no normal"),
    pytest.param(replaced("inverse_depth", np.ones((3, 5), dtype=np.float32)),
                 "has inverse_depth of shape (3, 5), but the index's prior size is 3 x 4, so it "
                 "must be (3, 4)", id="inverse_depth's shape"),
    pytest.param(replaced("normal", np.zeros((3, 4, 2), dtype=np.float32)),
                 "has normal of shape (3, 4, 2), but the index's prior size is 3 x 4, so it must "
                 "be (3, 4, 3)", id="normal's shape"),
    pytest.param(replaced("mask", np.zeros((4, 3), dtype=bool)),
                 "has mask of shape (4, 3), but the index's prior size is 3 x 4, so it must be "
                 "(3, 4)", id="mask's shape"),
    pytest.param(replaced("inverse_depth", GOOD["inverse_depth"].astype(np.float64)),
                 "has inverse_depth of dtype float64 ('<f8'), but a prior file's inverse_depth "
                 "is float32 ('<f4')", id="inverse_depth's dtype"),
    pytest.param(replaced("normal", GOOD["normal"].astype(">f4")),
                 "has normal of dtype float32 ('>f4'), but a prior file's normal is float32 "
                 "('<f4')", id="normal big-endian"),
    pytest.param(replaced("mask", GOOD["mask"].astype(np.uint8)),
                 "has mask of dtype uint8 ('|u1'), but a prior file's mask is bool ('|b1')",
                 id="mask's dtype"),
    pytest.param(at_pixel("inverse_depth", (1, 2), np.nan),
                 "holds nan in inverse_depth at the pixel (2, 1), but every value of a prior "
                 "file is finite", id="a NaN in inverse_depth"),
    pytest.param(at_pixel("normal", (2, 3, 1), np.nan),
                 "holds nan in normal at the pixel (3, 2), but every value of a prior file is "
                 "finite", id="a NaN in normal"),
    pytest.param(at_pixel("inverse_depth", (0, 1), 0.0),
                 "has inverse_depth 0.0 at the pixel (1, 0), which its mask marks, but "
                 "inverse_depth must be above 0 at every pixel its mask marks",
                 id="0 under a true mask"),
    pytest.param(at_pixel("inverse_depth", (2, 3), 0.5),
                 "has inverse_depth 0.5 at the pixel (3, 2), which its mask doesn't mark, but "
                 "inverse_depth must be 0 at every pixel its mask doesn't mark",
                 id="non-zero under a false mask"),
])
def test_a_prior_file_that_fails_a_check_is_refused(tmp_path: Path, arrays: dict[str, np.ndarray],
                                                    what: str) -> None:
    folder = write_good_set(tmp_path)
    path = prior_path(folder, PRIOR_NAMES[1])
    save_arrays(path, arrays)
    assert refused(folder, path) == what


def test_a_missing_prior_file_is_refused(tmp_path: Path) -> None:
    folder = write_good_set(tmp_path)
    path = prior_path(folder, PRIOR_NAMES[1])
    path.unlink()
    assert refused(folder, path) == "isn't there, but depth.json lists a prior for '000013.jpg'"


# No code runs when a Prior_File is read (Req 7.5)

class MakesFolder:
    """An object whose unpickling makes the folder at `path`, so a test can see whether loading
    a file ran code."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def __reduce__(self) -> tuple[Callable[[str], None], tuple[str]]:
        return os.mkdir, (str(self.path),)


def object_npz(path: Path, marker: Path) -> None:
    """A real .npz whose inverse_depth is an object array holding a MakesFolder."""
    save_arrays(path, replaced("inverse_depth", np.array([MakesFolder(marker)], dtype=object)))


def load_npz_anyway(path: Path) -> None:
    """Load the object array as only allow_pickle=True would."""
    with np.load(path, allow_pickle=True) as data:
        data["inverse_depth"]


def pickle_file(path: Path, marker: Path) -> None:
    """A plain pickle of a MakesFolder, named as a .npz."""
    path.write_bytes(pickle.dumps(MakesFolder(marker)))


def load_pickle_anyway(path: Path) -> None:
    """Unpickle the file as pickle itself would."""
    pickle.loads(path.read_bytes())


@pytest.mark.parametrize(("write", "load_anyway", "what"), [
    pytest.param(object_npz, load_npz_anyway, "can't be read as a .npz file (Object arrays "
                 "cannot be loaded when allow_pickle=False)", id="an object array"),
    pytest.param(pickle_file, load_pickle_anyway, "can't be read as a .npz file (it doesn't "
                 "start as a zip archive does)", id="a pickle"),
])
def test_a_prior_file_that_would_run_code_is_refused_and_runs_none(
        tmp_path: Path, write: Callable[[Path, Path], None], load_anyway: Callable[[Path], None],
        what: str) -> None:
    folder = write_good_set(tmp_path)
    path = prior_path(folder, PRIOR_NAMES[1])
    marker = tmp_path / "code ran"
    write(path, marker)
    assert refused(folder, path) == what
    assert not marker.exists()
    load_anyway(path)  # the payload is live: loading it with pickling allowed runs it
    assert marker.is_dir()


# Paths (Req 7.4)

def test_prior_path_is_the_stem_with_npz_in_the_folder(tmp_path: Path) -> None:
    assert prior_path(tmp_path, "000012.jpg") == tmp_path / "000012.npz"


def json_strings(value: object) -> list[str]:
    """Every string value in a JSON value, keys left out, in the order they're written."""
    if type(value) is dict:
        return [text for item in value.values() for text in json_strings(item)]
    if type(value) is list:
        return [text for item in value for text in json_strings(item)]
    return [value] if type(value) is str else []


def json_keys(value: object) -> set[str]:
    """Every key in a JSON value, nested ones included."""
    if type(value) is dict:
        return set(value).union(*(json_keys(item) for item in value.values()))
    if type(value) is list:
        return set().union(*(json_keys(item) for item in value))
    return set()


def test_the_index_names_no_path(tmp_path: Path) -> None:
    folder = write_good_set(tmp_path)
    text = (folder / PRIOR_INDEX).read_text(encoding="utf-8")
    value = json.loads(text)
    assert json_strings(value) == list(PRIOR_NAMES)  # its only strings are the images' names
    assert json_keys(value) == INDEX_KEYS | CAMERA_KEYS | ENTRY_KEYS
    assert ".npz" not in text


# Writing the index (Req 7.6)

def test_write_index_leaves_only_depth_json(tmp_path: Path) -> None:
    folder = empty_folder(tmp_path)
    write_index(folder, good_index())
    assert [path.name for path in folder.iterdir()] == [PRIOR_INDEX]
    assert load_index(folder / PRIOR_INDEX) == good_index()


def test_a_failed_rename_leaves_no_index_and_no_temporary_file(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def full_disk(source: object, target: object) -> NoReturn:
        raise OSError(errno.ENOSPC, "No space left on device")

    folder = empty_folder(tmp_path)
    monkeypatch.setattr("splatpipe.train.priors.os.replace", full_disk)
    with pytest.raises(OSError) as caught:
        write_index(folder, good_index())
    assert caught.value.errno == errno.ENOSPC
    assert list(folder.iterdir()) == []


# Property 4: the Prior_Format's round trip (Req 7.1, 7.3, 7.12)

# A Prior_File's arrays and their dtypes, written here from Req 7.1.
ARRAY_DTYPES = {"inverse_depth": "<f4", "normal": "<f4", "mask": "|b1"}

# −0.0, subnormals and values near ±1e308, each drawn often beside Hypothesis's own finite floats,
# so a writer or a reader that lost a sign, a tiny value or a huge one fails.
EDGE_FLOATS = (-0.0, 5e-324, -5e-324, 1e-310, -1e-310, 1e308, -1e308, sys.float_info.max,
               -sys.float_info.max)
FINITE = st.sampled_from(EDGE_FLOATS) | st.floats(allow_nan=False, allow_infinity=False)
# Every finite float32. The bounds are float32's own, so they leave out no value, but they keep
# Hypothesis from drawing float64s too big for float32 and throwing the example away.
FLOAT32_MAX = float(np.finfo(np.float32).max)
FLOAT32S = st.floats(min_value=-FLOAT32_MAX, max_value=FLOAT32_MAX, width=32, allow_nan=False,
                     allow_infinity=False)
POINT_COUNTS = st.integers(0, 10**6)
CAMERA_SIDES = st.integers(1, 10_000)
PRIOR_SIDES = st.integers(1, 8)
# A stem of 1 to 12 characters, a digit then lowercase ASCII letters, digits, _ and -, so it's a
# file name on Windows too: no character Windows refuses, never a reserved name like con, and no
# capital letter, so no two names fold to one file.
STEMS = st.tuples(st.sampled_from(string.digits),
                  st.text(string.ascii_lowercase + string.digits + "_-", max_size=11)).map("".join)
IMAGE_NAMES = STEMS.map(lambda stem: f"{stem}.jpg")


def entries_named(name: str) -> st.SearchStrategy[PriorEntry]:
    """A Prior_Index entry for the image `name`, every float drawn from FINITE."""
    return st.builds(PriorEntry, name=st.just(name),
                     quaternion=st.tuples(FINITE, FINITE, FINITE, FINITE),
                     translation=st.tuples(FINITE, FINITE, FINITE), scale=FINITE, offset=FINITE,
                     fit_points=POINT_COUNTS, kept_points=POINT_COUNTS, median_error=FINITE,
                     metres_per_unit=FINITE)


def priors_sized(width: int, height: int) -> st.SearchStrategy[Prior]:
    """A Prior at the Prior_Size `width` × `height`, with any finite float32 inverse_depth and
    normal and any mask, so no value needs to pass check_prior."""
    return st.builds(Prior, inverse_depth=arrays(np.float32, (height, width), elements=FLOAT32S),
                     normal=arrays(np.float32, (height, width, 3), elements=FLOAT32S),
                     mask=arrays(np.bool_, (height, width)))


@st.composite
def prior_sets(draw: st.DrawFn) -> tuple[PriorIndex, dict[str, Prior]]:
    """A Prior_Index with 1 to 4 entries of distinct names, and each name's Prior at the index's
    Prior_Size."""
    names = draw(st.lists(IMAGE_NAMES, min_size=1, max_size=4, unique=True))
    width, height = draw(PRIOR_SIDES), draw(PRIOR_SIDES)
    camera = (draw(CAMERA_SIDES), draw(CAMERA_SIDES), draw(FINITE), draw(FINITE), draw(FINITE),
              draw(FINITE))
    index = PriorIndex(camera=camera, prior_width=width, prior_height=height,
                       metres_per_unit=draw(FINITE),
                       priors=tuple(draw(entries_named(name)) for name in names))
    return index, {name: draw(priors_sized(width, height)) for name in names}


def exact(value: object) -> object:
    """`value` with every leaf as its type and, for a float, its 8 bytes, so −0.0 differs from 0.0
    and every bit counts. Tuples stay tuples, so a list or a value of another type differs."""
    if type(value) is tuple:
        return tuple(exact(item) for item in value)
    if type(value) is float:
        return float, struct.pack("<d", value)
    return type(value), value


@settings(max_examples=100, deadline=None)
@given(drawn=prior_sets())
def test_the_prior_reader_reads_back_what_the_prior_writer_wrote(
        drawn: tuple[PriorIndex, dict[str, Prior]]) -> None:
    """Feature: depth-priors, Property 4: The Prior_Reader reads back what the Prior_Writer wrote,
    bit for bit

    **Validates: Requirements 7.12, 7.1, 7.3**
    """
    index, priors = drawn
    with tempfile.TemporaryDirectory() as tmp:  # one folder per example
        folder = Path(tmp)
        for name, prior in priors.items():
            write_prior(prior_path(folder, name), prior)
        write_index(folder, index)
        read_index = load_index(folder / PRIOR_INDEX)
        read = {name: load_prior(prior_path(folder, name)) for name in priors}
        held = {}
        for name in priors:
            with np.load(prior_path(folder, name), allow_pickle=False) as data:
                held[name] = sorted(data.files)

    # Every index value, in order, of the same type and with the same bits (Req 7.3).
    assert type(read_index) is PriorIndex
    assert all(type(entry) is PriorEntry for entry in read_index.priors)
    assert exact(astuple(read_index)) == exact(astuple(index))

    # Each Prior_File holds exactly the three arrays, of the same shapes, dtypes and bits
    # (Req 7.1).
    for name, prior in priors.items():
        assert held[name] == sorted(ARRAY_DTYPES)
        for label, dtype in ARRAY_DTYPES.items():
            found, wrote = getattr(read[name], label), getattr(prior, label)
            assert found.dtype.str == dtype
            assert found.shape == wrote.shape
            view = np.uint8 if dtype == "|b1" else np.uint32  # a bool can't be seen as 4 bytes
            assert np.array_equal(found.view(view), wrote.view(view))
