"""The step's files for the Training_Step and the Viewer: the Training_Images, the Training_Masks,
the Solve_Record and the Viewer_Cameras, with the Work_Folder's names.

The names are Fixed_Constants, because the Work_Folder is the contract the Training_Step reads
(D2):

    work/cameras/colmap/            the COLMAP_Folder, COLMAP's own files (O8)
    work/cameras/images/<name>      the Training_Images, one per Placed_Frame
    work/cameras/masks/<stem>.png   the Training_Masks, only with mask_used
    work/cameras/sparse/            the Levelled_Model, exactly the three Model_Files
    work/cameras/solve.json         the Solve_Record, written last in work/cameras/
    out/cameras.json                the Viewer_Cameras, written last of all

copy_training_images copies each Placed_Frame's warped Colour_Pick from the Undistortion's images/
with shutil.copyfile, which follows a link that undistortion_copy_policy made. It copies rather
than moves, because the Undistortion's images/ stays in the COLMAP_Folder (Req 11.2, 11.5, C4).
write_training_masks encodes the Undistorted_Mask once as a PNG with cv2.imencode and writes the
same bytes as <stem>.png for each Placed_Frame, like 000000.png for 000000.jpg (Req 11.3). Each
makes its folder first. An OSError, like a full disk, or an encode that cv2.imencode reports as
failed, is the Req 11.12 failure, which names the file and the reason.

mean_reprojection_error averages the points' errors that aren't -1, as COLMAP does, and gives None
when no point has one (F15). solve_record and viewer_cameras are pure. They give the Solve_Record
and the Viewer_Cameras with the fields of the design's two tables, in that order, as only plain
int, float, bool, str, None, list and dict values, so check_json_value never refuses one (Req
11.7, 11.9, 13.4). Floats keep full precision. The Viewer_Cameras list the Placed_Frames in
Pick_Number order and hold no points (D3).

write_record writes either record with splatpipe.jsonio.write_json. On an OSError it removes the
partly written file, then raises the Req 11.12 failure, so a failed write leaves no record (C6).
The step writes the Solve_Record after every other file in work/cameras/, and the Viewer_Cameras
last of all (Req 11.6, 11.8).

cv2 only encodes the mask PNG here, and never reads or writes video (Req 12.14).

From splatpipe, this module imports only splatpipe.errors, splatpipe.jsonio and the step's model,
level and solve.
"""

from __future__ import annotations

import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from splatpipe.cameras.level import Levelling
from splatpipe.cameras.model import NO_ERROR, Model, Points
from splatpipe.cameras.solve import Solved
from splatpipe.errors import ExpectedFailure
from splatpipe.jsonio import write_json

# The Work_Folder's names, Fixed_Constants because the Work_Folder is the contract that the
# Training_Step reads (D2).
IMAGES = "images"  # work/cameras/images/, the Training_Images
MASKS = "masks"  # work/cameras/masks/, the Training_Masks, only with mask_used
SPARSE = "sparse"  # work/cameras/sparse/, the Levelled_Model
COLMAP = "colmap"  # work/cameras/colmap/, the COLMAP_Folder
RECORD = "solve.json"  # work/cameras/solve.json, the Solve_Record
VIEWER = "cameras.json"  # out/cameras.json, the Viewer_Cameras

_PNG = ".png"  # the Training_Masks' format and suffix, like 000000.png (Req 11.3)


def copy_training_images(source: Path, target: Path, names: Sequence[str]) -> None:
    """Make `target`, then copy each source/<name> to target/<name>: the Training_Images, from
    the Undistortion's images/ folder `source` (Req 11.2, C4).

    shutil.copyfile follows a link that undistortion_copy_policy made, and the Undistortion's
    images/ stays in the COLMAP_Folder (Req 11.5). An OSError, like a full disk, is the Req 11.12
    failure, naming the file and the reason.
    """
    _make_folder(target)
    for name in names:
        path = target / name
        try:
            shutil.copyfile(source / name, path)
        except OSError as error:
            raise _not_written(f"The undistorted image {source / name} couldn't be copied to "
                               f"{path}", path, _reason(error)) from error


def write_training_masks(folder: Path, mask: np.ndarray, names: Sequence[str]) -> None:
    """Make `folder`, then write the Undistorted_Mask `mask` as folder/<stem>.png for each name,
    like 000000.png for 000000.jpg: the Training_Masks (Req 11.3).

    The mask is encoded once with cv2.imencode, and every file gets the same bytes. An encode
    that cv2.imencode reports as failed, or an OSError, like a full disk, is the Req 11.12
    failure, naming the file and the reason.
    """
    ok, data = cv2.imencode(_PNG, mask)
    _make_folder(folder)
    for name in names:
        _write_mask(folder / f"{Path(name).stem}{_PNG}", ok, data)


def mean_reprojection_error(points: Points) -> float | None:
    """The mean of the points' errors that aren't NO_ERROR, -1, in pixels, as COLMAP's
    ComputeMeanReprojectionError takes it, or None when no point has an error (F15)."""
    errors = np.asarray(points.errors, dtype=np.float64)
    known = errors[errors != NO_ERROR]
    return float(known.mean()) if known.size else None


def solve_record(solved: Solved, levelling: Levelling, camera_model: str,
                 mean_error: float | None) -> dict[str, Any]:
    """The Solve_Record, as plain JSON values in the order of the design's table (Req 11.7,
    13.4). Pure.

    `solved` is what the solve flow found, `levelling` what the Levelling gave, `camera_model`
    the Camera_Model, and `mean_error` the Final_Model's mean_reprojection_error. The Models of
    the run that made the Final_Model are in number order. The Level_Move x' = R x + T gives R as
    a list of 3 rows and T as a list of 3, which are the identity and zeros when the Levelling
    kept the COLMAP_Frame.
    """
    move = levelling.move
    return {
        "camera_model": str(camera_model),
        "calibration_ran": bool(solved.calibration_ran),
        "calibration_failure": _text_or_none(solved.calibration_failure),
        "mapper": str(solved.mapper),
        "fallback_reason": _text_or_none(solved.fallback_reason),
        "models": [{"number": int(model.number), "placed_frames": int(model.placed)}
                   for model in sorted(solved.models, key=lambda model: model.number)],
        "final_model": int(solved.final.number),
        "mean_reprojection_error": _float_or_none(mean_error),
        "levelled": bool(levelling.levelled),
        "level_reason": _text_or_none(levelling.reason),
        "mean_up_length": _float_or_none(levelling.mean_up_length),
        "scene_centre": _text_or_none(levelling.scene_centre),
        "heading_camera": _text_or_none(levelling.heading_camera),
        "rotation": np.asarray(move.rotation, dtype=np.float64).tolist(),
        "translation": np.asarray(move.translation, dtype=np.float64).tolist(),
    }


def viewer_cameras(model: Model, numbers: Mapping[str, int], levelled: bool) -> dict[str, Any]:
    """The Viewer_Cameras of the Levelled_Model `model`, as plain JSON values in the order of the
    design's table (Req 11.9). Pure.

    The camera is the model's one PINHOLE camera, the Undistorted_Camera: its width and height,
    and its fx, fy, cx and cy. Then each Placed_Frame in Pick_Number order, by `numbers`, with its
    name and its Pose as the model's images.bin holds it: the cam_from_world quaternion, w x y z,
    and translation. `levelled` says whether the Levelling moved the scene. The Viewer_Cameras
    hold no points (D3).
    """
    camera = model.cameras[0]
    fx, fy, cx, cy = camera.params
    images = sorted(model.images, key=lambda image: numbers[image.name])
    return {
        "camera": {"width": int(camera.width), "height": int(camera.height), "fx": float(fx),
                   "fy": float(fy), "cx": float(cx), "cy": float(cy)},
        "levelled": bool(levelled),
        "frames": [{"name": str(image.name),
                    "quaternion_wxyz": [float(value) for value in image.qvec],
                    "translation": [float(value) for value in image.tvec]}
                   for image in images],
    }


def write_record(path: Path, record: dict[str, Any]) -> None:
    """Write `record`, the Solve_Record or the Viewer_Cameras, to `path` with write_json, which
    checks every value first.

    On an OSError, like a full disk, the partly written file is removed, and then it's the
    Req 11.12 failure, naming the file and the reason, so a failed write leaves no record (C6).
    """
    try:
        write_json(path, record)
    except OSError as error:
        path.unlink(missing_ok=True)
        raise _not_written(f"The file {path} couldn't be written", path,
                           _reason(error)) from error


def _write_mask(path: Path, ok: bool, data: np.ndarray) -> None:
    """Write the PNG bytes that cv2.imencode gave to the Training_Mask `path`, or refuse when it
    reported a failed encode. Either failure is the Req 11.12 failure, naming `path`."""
    problem = f"The training mask {path} couldn't be written"
    if not ok:
        raise _not_written(problem, path, "OpenCV couldn't encode the undistorted mask as a PNG")
    try:
        path.write_bytes(data.tobytes())
    except OSError as error:
        raise _not_written(problem, path, _reason(error)) from error


def _make_folder(folder: Path) -> None:
    """Make `folder` and its parents, unless it exists. An OSError is the Req 11.12 failure,
    naming the folder and the reason."""
    try:
        folder.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise _not_written(f"The folder {folder} couldn't be made", folder,
                           _reason(error)) from error


def _text_or_none(value: str | None) -> str | None:
    """`value` as a plain str, or None."""
    return None if value is None else str(value)


def _float_or_none(value: float | None) -> float | None:
    """`value` as a plain float, or None."""
    return None if value is None else float(value)


def _reason(error: OSError) -> str:
    """The reason the system gave for `error`, like "No space left on device"."""
    return error.strerror or str(error)


def _not_written(problem: str, path: Path, reason: str) -> ExpectedFailure:
    """The Req 11.12 failure: `problem`, which names the file, the reason, and what to do."""
    return ExpectedFailure(f"{problem} ({reason}). Free some space on the disk, or check that you "
                           f"can write to {path.parent}, then run the step again.")
