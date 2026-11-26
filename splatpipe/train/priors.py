"""The Prior_Format: the depth priors in work/depth/, which the Depth_Step writes and the
Train_Step reads (depth-priors spec, Req 7 and 10). Both steps use this one module, so they share
one definition of the format (Req 7.11).

    work/depth/<stem>.npz   one Prior_File per Fitted_Frame, <stem> being the image's name in
                            images.bin without .jpg. numpy's uncompressed .npz, holding exactly
                            three arrays at the Prior_Size h x w: inverse_depth (h, w) <f4, the
                            Fitted_Inverse_Depth, above 0 where mask is true and 0 elsewhere;
                            normal (h, w, 3) <f4, MoGe's unit normal in the camera frame, or 0s;
                            and mask (h, w) |b1, true exactly at the pixels with a prior (Req 7.1).
    work/depth/depth.json   the Prior_Index, written last: format (1), camera (width, height, fx,
                            fy, cx, cy), prior_width, prior_height, metres_per_unit and priors,
                            one object per Prior with name, quaternion, translation, scale,
                            offset, fit_points, kept_points, median_error and metres_per_unit
                            (Req 7.3).

Stored pixel (x, y) is centred at ((x + 0.5) × W / w, (y + 0.5) × H / h) in the Shared_Camera's
W × H pixels (Req 7.2). The index names no path: a Prior_File's path is prior_path() of an image
name that the reader has checked against images.bin (Req 7.4, A13).

write_prior writes the three arrays with np.savez, as little-endian float32, float32 and bool.
write_index writes the index with splatpipe.jsonio.write_json to depth.json.tmp in the same
folder, then turns it into depth.json with one os.replace, so a reader never finds a half-written
index. On an OSError it removes the temporary file, then raises (Req 7.6).

load_prior reads with np.load(path, allow_pickle=False), so no object array and no code can be
loaded (Req 7.5), and only once the file starts as a zip archive does, so np.load never takes it
for a .npy or a pickle. load_index reads with read_json and checks every field's JSON type
exactly, so true is never 1 and 512.0 is never a whole number. check_prior checks one Prior's
shapes, dtypes and values. read_prior_set gives None when there's no depth.json (Req 10.2), and
otherwise runs every check of Req 10.4 in order: format 1, the camera, the Prior_Size, one or more
priors, each name in images.bin and only once, each pose, and then each Prior_File.

Every refusal is a BrokenPriors, which names the file and what failed, so one bad file drops the
whole Prior_Set (A9). The Train_Step turns it into a warning and trains without priors.

This module imports no torch. From splatpipe it imports only splatpipe.jsonio, and
splatpipe.train.colmap for its types.
"""

from __future__ import annotations

import contextlib
import json
import operator
import os
import zipfile
import zlib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from splatpipe.jsonio import read_json, write_json
from splatpipe.train.colmap import ModelImage, SparseModel

# Fixed_Constants: the Prior_Format's names and version (Req 7).
PRIOR_FOLDER = "depth"  # the Priors' folder in work/
PRIOR_INDEX = "depth.json"  # the Prior_Index, in work/depth/
PRIOR_FORMAT = 1  # the Prior_Format's version, the index's format
PRIOR_ARRAYS = ("inverse_depth", "normal", "mask")  # a Prior_File's arrays, exactly these

_TEMPORARY = f"{PRIOR_INDEX}.tmp"  # the Prior_Index while write_index writes it
_DTYPES = {"inverse_depth": np.dtype("<f4"), "normal": np.dtype("<f4"), "mask": np.dtype("|b1")}
_FLOAT_ARRAYS = ("inverse_depth", "normal")  # the arrays whose values must be finite
# The first bytes of a zip archive with members and of an empty one, the two that np.load reads
# as a .npz file.
_ZIP_STARTS = (b"PK\x03\x04", b"PK\x05\x06")
# What np.load and NpzFile raise for a file that can't be read or parsed: OSError from the disk,
# ValueError for a bad .npy header, cut-short data or an object array, EOFError and
# zipfile.BadZipFile for a broken archive, zlib.error for broken compressed data, and
# RuntimeError (NotImplementedError included) for an encrypted member or an unknown compression.
_READ_ERRORS = (OSError, ValueError, EOFError, RuntimeError, zipfile.BadZipFile, zlib.error)
# The Prior_Index's fields, in the order write_index writes them (Req 7.3).
_INDEX_FIELDS = ("format", "camera", "prior_width", "prior_height", "metres_per_unit", "priors")
_CAMERA_FIELDS = ("width", "height", "fx", "fy", "cx", "cy")
_ENTRY_FIELDS = ("name", "quaternion", "translation", "scale", "offset", "fit_points",
                 "kept_points", "median_error", "metres_per_unit")
_SHOWN_LENGTH = 40  # the most characters of a JSON value a message shows


@dataclass(frozen=True)
class Prior:
    """One Fitted_Frame's Prior at the Prior_Size h × w."""

    inverse_depth: np.ndarray  # (h, w) float32, above 0 where mask is true and 0 elsewhere
    normal: np.ndarray  # (h, w, 3) float32, unit normals in the camera frame, or 0s
    mask: np.ndarray  # (h, w) bool, true exactly at the pixels with a prior


@dataclass(frozen=True)
class PriorEntry:
    """One Prior's object in the Prior_Index (Req 7.3)."""

    name: str  # the image's name in images.bin
    quaternion: tuple[float, float, float, float]  # w x y z, exactly as images.bin gives it
    translation: tuple[float, float, float]  # exactly as images.bin gives it
    scale: float  # the second fit's scale
    offset: float  # the second fit's offset
    fit_points: int  # the first fit's points
    kept_points: int  # the second fit's points
    median_error: float  # the median Relative_Depth_Error over every Fit_Point
    metres_per_unit: float  # this Frame's Metres_Per_Unit


@dataclass(frozen=True)
class PriorIndex:
    """The Prior_Index, depth.json, whose format is PRIOR_FORMAT (Req 7.3)."""

    camera: tuple[int, int, float, float, float, float]  # width, height, fx, fy, cx, cy
    prior_width: int
    prior_height: int
    metres_per_unit: float  # the job's Metres_Per_Unit, reported and never applied (Req 6.4)
    priors: tuple[PriorEntry, ...]  # in images.bin's order


@dataclass(frozen=True)
class PriorSet:
    """The Priors that read_prior_set has read and checked for one run."""

    index: PriorIndex
    priors: dict[str, Prior]  # by image name, in the index's order


class BrokenPriors(Exception):
    """Priors that fail a check of Req 10.4. `path` is the file and `what` says what failed,
    worded to follow the path, so the error reads like "<path> has the camera fx 1450.0, but
    cameras.bin's is 1452.5". The Train_Step turns it into a warning (Req 10.3, A9)."""

    def __init__(self, path: Path, what: str) -> None:
        super().__init__(path, what)
        self.path = path
        self.what = what

    def __str__(self) -> str:
        return f"{self.path} {self.what}"


def prior_path(folder: Path, name: str) -> Path:
    """The Prior_File in `folder`, work/depth/, of the image `name`: <stem>.npz (Req 7.1, 7.4)."""
    return folder / f"{name.removesuffix('.jpg')}.npz"


def write_prior(path: Path, prior: Prior) -> None:
    """Write `prior` to `path` with np.savez, numpy's uncompressed .npz, as exactly the three
    arrays of PRIOR_ARRAYS, little-endian float32, float32 and bool (Req 7.1).

    The file is opened by its own path, so np.savez never adds .npz to the name.
    """
    arrays = {name: np.asarray(getattr(prior, name), dtype=_DTYPES[name])
              for name in PRIOR_ARRAYS}
    with path.open("wb") as file:
        np.savez(file, allow_pickle=False, **arrays)


def write_index(folder: Path, index: PriorIndex) -> None:
    """Write `index` to depth.json in `folder`, work/depth/, through depth.json.tmp and one
    os.replace on the same file system, so a reader never finds a half-written index (Req 7.6).

    write_json checks every value before it writes anything. On an OSError, the temporary file is
    removed and the error raised again, so a failed write leaves neither file.
    """
    temporary = folder / _TEMPORARY
    try:
        write_json(temporary, _index_value(index))
        os.replace(temporary, folder / PRIOR_INDEX)
    except OSError:
        with contextlib.suppress(OSError):
            temporary.unlink(missing_ok=True)
        raise


def load_prior(path: Path) -> Prior:
    """The Prior in the Prior_File at `path`, read with np.load(path, allow_pickle=False), so no
    object array and no code can be loaded (Req 7.5).

    The file must start as a zip archive does and hold exactly the three arrays of PRIOR_ARRAYS.
    No value is checked here: check_prior does that. A file that can't be read or parsed is a
    BrokenPriors.
    """
    _check_zip_start(path)
    try:
        with np.load(path, allow_pickle=False) as data:
            names = sorted(data.files)
            if names != sorted(PRIOR_ARRAYS):
                raise BrokenPriors(path, f"holds the arrays {names}, but a prior file holds "
                                         "exactly inverse_depth, normal and mask")
            arrays = {name: data[name] for name in PRIOR_ARRAYS}
    except _READ_ERRORS as error:
        raise _not_npz(path, error) from error
    for name, array in arrays.items():
        if not isinstance(array, np.ndarray):  # a member that isn't a .npy comes back as bytes
            raise BrokenPriors(path, f"holds {name} as bytes that aren't a numpy array")
    return Prior(**arrays)


def load_index(path: Path) -> PriorIndex:
    """The Prior_Index at `path`, read with read_json, with every field's JSON type checked
    exactly: `format` the int 1, the camera's width and height ints and its other values ints or
    floats, the Prior_Size ints, and `priors` a list of objects with exactly the fields of
    Req 7.3. A file that can't be read or parsed, or a field of the wrong type, is a BrokenPriors.

    The values aren't checked against the model here: read_prior_set does that.
    """
    try:
        value = read_json(path)
    except (OSError, ValueError, RecursionError) as error:
        raise BrokenPriors(path, f"can't be read as JSON ({_reason(error)})") from error
    if type(value) is not dict:
        raise BrokenPriors(path, f"holds {_shown(value)}, but a prior index is a JSON object")
    _check_format(path, value)
    _check_fields(path, value, "", _INDEX_FIELDS, "a prior index")
    camera = _camera(path, value["camera"])
    prior_width = _whole(path, value["prior_width"], "prior_width")
    prior_height = _whole(path, value["prior_height"], "prior_height")
    metres_per_unit = _number(path, value["metres_per_unit"], "metres_per_unit")
    priors = value["priors"]
    if type(priors) is not list:
        raise BrokenPriors(path, f"has priors {_shown(priors)}, but it must be a list of objects")
    entries = tuple(_entry(path, item, f"priors[{place}]") for place, item in enumerate(priors))
    return PriorIndex(camera, prior_width, prior_height, metres_per_unit, entries)


def check_prior(path: Path, prior: Prior, width: int, height: int) -> None:
    """Refuse a Prior, read from `path`, that doesn't fit the index's Prior_Size `width` ×
    `height` (Req 10.4): inverse_depth (h, w), normal (h, w, 3) and mask (h, w), of dtypes
    float32, float32 and bool, every value finite, and inverse_depth above 0 where mask is true
    and 0 where it's false."""
    shapes = {"inverse_depth": (height, width), "normal": (height, width, 3),
              "mask": (height, width)}
    for name in PRIOR_ARRAYS:
        array = getattr(prior, name)
        if array.shape != shapes[name]:
            raise BrokenPriors(path, f"has {name} of shape {array.shape}, but the index's prior "
                                     f"size is {height} x {width}, so it must be "
                                     f"{shapes[name]}")
        if array.dtype != _DTYPES[name]:
            raise BrokenPriors(path, f"has {name} of dtype {_dtype(array.dtype)}, but a prior "
                                     f"file's {name} is {_dtype(_DTYPES[name])}")
    for name in _FLOAT_ARRAYS:
        _check_finite(path, name, getattr(prior, name))
    _check_mask(path, prior)


def read_prior_set(folder: Path, model: SparseModel) -> PriorSet | None:
    """The Prior_Set in `folder`, work/depth/, checked against `model`, the Sparse_Model, or None
    when `folder` holds no depth.json file (Req 10.2).

    The checks run in Req 10.4's order: format 1 and every field's type, through load_index; the
    camera; the Prior_Size; one or more priors; each name in images.bin and only once; each pose;
    and then each Prior_File, at prior_path() of its checked name, never a path from the index,
    through load_prior and check_prior (A13). The first check that fails raises BrokenPriors, so
    one bad file drops the whole Prior_Set (A9).
    """
    path = folder / PRIOR_INDEX
    if not path.is_file():
        return None
    index = load_index(path)
    _check_camera(path, index, model)
    _check_size(path, index)
    if not index.priors:
        raise BrokenPriors(path, "lists no priors, but a prior index lists one or more")
    images = _check_names(path, index, model)
    _check_poses(path, index, images)
    priors = {entry.name: _read_prior(folder, entry.name, index) for entry in index.priors}
    return PriorSet(index, priors)


def _index_value(index: PriorIndex) -> dict[str, Any]:
    """The Prior_Index as plain JSON values, in the field order of Req 7.3."""
    width, height, *intrinsics = index.camera
    camera: dict[str, Any] = {"width": _int(width), "height": _int(height)}
    camera.update((label, float(value)) for label, value in zip(_CAMERA_FIELDS[2:], intrinsics))
    return {"format": PRIOR_FORMAT,
            "camera": camera,
            "prior_width": _int(index.prior_width),
            "prior_height": _int(index.prior_height),
            "metres_per_unit": float(index.metres_per_unit),
            "priors": [_entry_value(entry) for entry in index.priors]}


def _entry_value(entry: PriorEntry) -> dict[str, Any]:
    """One Prior's object in the Prior_Index, as plain JSON values."""
    return {"name": str(entry.name),
            "quaternion": [float(value) for value in entry.quaternion],
            "translation": [float(value) for value in entry.translation],
            "scale": float(entry.scale),
            "offset": float(entry.offset),
            "fit_points": _int(entry.fit_points),
            "kept_points": _int(entry.kept_points),
            "median_error": float(entry.median_error),
            "metres_per_unit": float(entry.metres_per_unit)}


def _int(value: int) -> int:
    """A whole number as a plain int, numpy's included. A float is refused with TypeError rather
    than rounded."""
    return int(operator.index(value))


def _check_zip_start(path: Path) -> None:
    """Refuse a Prior_File that doesn't start as a zip archive does, so np.load reads it as a
    .npz file and never as a .npy or a pickle."""
    try:
        with path.open("rb") as file:
            start = file.read(len(_ZIP_STARTS[0]))
    except OSError as error:
        raise _not_npz(path, error) from error
    if not start.startswith(_ZIP_STARTS):
        raise BrokenPriors(path, "can't be read as a .npz file (it doesn't start as a zip "
                                 "archive does)")


def _not_npz(path: Path, error: BaseException) -> BrokenPriors:
    """The refusal of a Prior_File that can't be read or parsed, with the reason."""
    return BrokenPriors(path, f"can't be read as a .npz file ({_reason(error)})")


def _check_format(path: Path, value: dict[str, Any]) -> None:
    """Refuse an index whose format isn't the int PRIOR_FORMAT, before any other field."""
    if "format" not in value:
        raise BrokenPriors(path, f"has no format, but the train step reads only format "
                                 f"{PRIOR_FORMAT}")
    found = value["format"]
    if type(found) is not int or found != PRIOR_FORMAT:
        raise BrokenPriors(path, f"has format {_shown(found)}, but the train step reads only "
                                 f"format {PRIOR_FORMAT}")


def _check_fields(path: Path, value: dict[str, Any], field: str, names: Sequence[str],
                  holder: str) -> None:
    """Refuse a JSON object, the one at `field` or the index itself when `field` is "", that
    doesn't hold exactly the fields `names`. `holder` names it in the message, like "the
    camera"."""
    found = sorted(value)
    if found != sorted(names):
        where = f" in {field}" if field else ""
        raise BrokenPriors(path, f"holds the fields {found}{where}, but {holder} holds exactly "
                                 f"{_words(names)}")


def _object(path: Path, value: object, field: str, names: Sequence[str],
            holder: str) -> dict[str, Any]:
    """`value`, the JSON object at `field`, once it's a JSON object of exactly the fields
    `names`."""
    if type(value) is not dict:
        raise BrokenPriors(path, f"has {field} {_shown(value)}, but it must be a JSON object")
    _check_fields(path, value, field, names, holder)
    return value


def _camera(path: Path, value: object) -> tuple[int, int, float, float, float, float]:
    """The index's camera: its width and height ints, and fx, fy, cx and cy as floats."""
    camera = _object(path, value, "camera", _CAMERA_FIELDS, "the camera")
    fx, fy, cx, cy = (_number(path, camera[label], f"camera.{label}")
                      for label in _CAMERA_FIELDS[2:])
    return (_whole(path, camera["width"], "camera.width"),
            _whole(path, camera["height"], "camera.height"), fx, fy, cx, cy)


def _entry(path: Path, value: object, field: str) -> PriorEntry:
    """One Prior's object in the index, at `field`, like "priors[2]"."""
    entry = _object(path, value, field, _ENTRY_FIELDS, "each prior")
    qw, qx, qy, qz = _numbers(path, entry["quaternion"], f"{field}.quaternion", 4)
    tx, ty, tz = _numbers(path, entry["translation"], f"{field}.translation", 3)
    return PriorEntry(name=_text(path, entry["name"], f"{field}.name"),
                      quaternion=(qw, qx, qy, qz),
                      translation=(tx, ty, tz),
                      scale=_number(path, entry["scale"], f"{field}.scale"),
                      offset=_number(path, entry["offset"], f"{field}.offset"),
                      fit_points=_whole(path, entry["fit_points"], f"{field}.fit_points"),
                      kept_points=_whole(path, entry["kept_points"], f"{field}.kept_points"),
                      median_error=_number(path, entry["median_error"],
                                           f"{field}.median_error"),
                      metres_per_unit=_number(path, entry["metres_per_unit"],
                                              f"{field}.metres_per_unit"))


def _whole(path: Path, value: object, field: str) -> int:
    """`value`, the field at `field`, once it's a JSON whole number: an int, never a bool or a
    float."""
    if type(value) is not int:
        raise BrokenPriors(path, f"has {field} {_shown(value)}, but it must be a whole number")
    return value


def _number(path: Path, value: object, field: str) -> float:
    """`value`, the field at `field`, as a float, once it's a JSON number: an int or a float,
    never a bool. An int too big for a float is refused."""
    if type(value) is float:
        return value
    if type(value) is not int:
        raise BrokenPriors(path, f"has {field} {_shown(value)}, but it must be a number")
    try:
        return float(value)
    except OverflowError:
        raise BrokenPriors(path, f"has {field} {_shown(value)}, which is too big for a "
                                 "float") from None


def _numbers(path: Path, value: object, field: str, count: int) -> tuple[float, ...]:
    """`value`, the field at `field`, as `count` floats, once it's a JSON list of `count`
    numbers."""
    if type(value) is not list or len(value) != count:
        raise BrokenPriors(path, f"has {field} {_shown(value)}, but it must be a list of "
                                 f"{count} numbers")
    return tuple(_number(path, item, f"{field}[{place}]") for place, item in enumerate(value))


def _text(path: Path, value: object, field: str) -> str:
    """`value`, the field at `field`, once it's a JSON string."""
    if type(value) is not str:
        raise BrokenPriors(path, f"has {field} {_shown(value)}, but it must be a string")
    return value


def _check_camera(path: Path, index: PriorIndex, model: SparseModel) -> None:
    """Refuse an index made for another camera than cameras.bin's Shared_Camera."""
    shared = model.camera
    need = (shared.width, shared.height, shared.fx, shared.fy, shared.cx, shared.cy)
    for label, found, value in zip(_CAMERA_FIELDS, index.camera, need):
        if found != value:
            raise BrokenPriors(path, f"has the camera {label} {found!r}, but cameras.bin's is "
                                     f"{value!r}")


def _check_size(path: Path, index: PriorIndex) -> None:
    """Refuse a Prior_Size that isn't above 0 on each side."""
    for label, value in (("prior_width", index.prior_width),
                         ("prior_height", index.prior_height)):
        if value <= 0:
            raise BrokenPriors(path, f"has {label} {value}, but it must be a whole number above 0")


def _check_names(path: Path, index: PriorIndex, model: SparseModel) -> dict[str, ModelImage]:
    """Refuse a name that isn't an image's in images.bin, or that the index lists twice. Gives
    images.bin's images by name."""
    images = {image.name: image for image in model.images}
    seen: set[str] = set()
    for entry in index.priors:
        if entry.name not in images:
            raise BrokenPriors(path, f"names the image {entry.name!r}, which images.bin doesn't "
                                     "hold")
        if entry.name in seen:
            raise BrokenPriors(path, f"names the image {entry.name!r} twice, but an image has "
                                     "one prior at most")
        seen.add(entry.name)
    return images


def _check_poses(path: Path, index: PriorIndex, images: dict[str, ModelImage]) -> None:
    """Refuse a prior whose quaternion or translation isn't, value for value, its image's in
    images.bin, so Priors made for an earlier camera solve are dropped (A9)."""
    for entry in index.priors:
        image = images[entry.name]
        for label, found, need in (("quaternion", entry.quaternion, image.quaternion),
                                   ("translation", entry.translation, image.translation)):
            if found != _floats(need):
                raise BrokenPriors(path, f"has the pose of an earlier camera solve for "
                                         f"{entry.name!r}: its {label} isn't the one images.bin "
                                         "gives")


def _read_prior(folder: Path, name: str, index: PriorIndex) -> Prior:
    """The checked Prior of the image `name`, read from prior_path(folder, name)."""
    path = prior_path(folder, name)
    if not path.is_file():
        raise BrokenPriors(path, f"isn't there, but {PRIOR_INDEX} lists a prior for {name!r}")
    prior = load_prior(path)
    check_prior(path, prior, index.prior_width, index.prior_height)
    return prior


def _check_finite(path: Path, name: str, array: np.ndarray) -> None:
    """Refuse a value of the array `name` that isn't finite, naming the first such pixel."""
    bad = ~np.isfinite(array)
    if bad.any():
        where = tuple(int(axis) for axis in np.argwhere(bad)[0])
        raise BrokenPriors(path, f"holds {array[where]} in {name} at the pixel "
                                 f"{_pixel(where)}, but every value of a prior file is finite")


def _check_mask(path: Path, prior: Prior) -> None:
    """Refuse an inverse_depth that isn't above 0 where the mask is true, or isn't 0 where it's
    false, naming the first such pixel."""
    depth, marked = prior.inverse_depth, prior.mask
    for bad, rule in ((marked & ~(depth > 0), "which its mask marks, but inverse_depth must be "
                                              "above 0 at every pixel its mask marks"),
                      (~marked & (depth != 0), "which its mask doesn't mark, but inverse_depth "
                                               "must be 0 at every pixel its mask doesn't mark")):
        if bad.any():
            where = tuple(int(axis) for axis in np.argwhere(bad)[0])
            raise BrokenPriors(path, f"has inverse_depth {depth[where]} at the pixel "
                                     f"{_pixel(where)}, {rule}")


def _floats(values: Iterable[Any]) -> tuple[float, ...]:
    """Values as plain floats, numpy's float64 included, which keeps every bit."""
    return tuple(float(value) for value in values)


def _pixel(where: tuple[int, ...]) -> str:
    """The pixel of an index into an (h, w) or (h, w, 3) array, as "(x, y)"."""
    return f"({where[1]}, {where[0]})"


def _dtype(dtype: np.dtype) -> str:
    """A dtype in words, like "float32 ('<f4')"."""
    return f"{dtype.name} ({dtype.str!r})"


def _words(names: Sequence[str]) -> str:
    """Names as a list in words, like "a, b and c"."""
    return names[0] if len(names) == 1 else f"{', '.join(names[:-1])} and {names[-1]}"


def _shown(value: object) -> str:
    """A JSON value as JSON text, cut to _SHOWN_LENGTH characters, for a message. ASCII only, so
    a control character in a string can't garble the terminal."""
    try:
        text = json.dumps(value, ensure_ascii=True)
    except (ValueError, RecursionError):  # nested too deeply to write back
        text = "a nested value"
    return text if len(text) <= _SHOWN_LENGTH else f"{text[:_SHOWN_LENGTH - 3]}..."


def _reason(error: BaseException) -> str:
    """The reason an error gives: an OSError's from the system, like "No space left on device",
    and any other error's message."""
    if isinstance(error, OSError) and error.strerror:
        return error.strerror
    return str(error) or type(error).__name__
