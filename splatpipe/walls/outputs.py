"""What the Walls_Step writes: the Walls_Priors, the Walls_Record, the Walls_Demos and the
Walls_Index, and the folders it clears (walls spec, Req 1.10, 17.6 to 17.9, 18).

    work/walls/<stem>.npz     one Walls_Prior_File per Prior_Frame, in the Prior_Format
    work/walls/walls.json     the Walls_Record (D7)
    out/walls/<stem>.jpg      the Walls_Demos (A17)
    work/walls/depth.json     the Walls_Index, written last of all (Req 18.5)
    work/walls/scratch/       the scratch folder, removed before the step returns

- walls_prior (Req 17.7, 17.8, D3, D4, A13): a Prior_Frame's Walls_Prior from its Walls_Depth and
  the normals its change chain gave it. The `mask` is the Depth_Prior's exactly. `inverse_depth`
  is float32(1 / the Walls_Depth) where the mask marks and 0 elsewhere, and a marked pixel whose
  float32 value isn't finite and above 0 takes the Depth_Prior's `inverse_depth` and `normal`.
  `normal` is the change chain's normal, the Depth_Prior's where that isn't finite or has a length
  of 0, and (0, 0, 0) at every unmarked pixel. The Depth_Prior passed check_prior when the step
  read it, so every Walls_Prior passes check_prior at the Prior_Size too (Req 17.9).
- depth_prior_copy (Req 17.6, D2): the Depth_Prior's three arrays, copied unchanged, bit for bit.
- demo_panel (Req 18.3, D4, A17): side by side at the Prior_Size, the Frame, decoded again with
  the Train_Step's read_training_image and scaled_down, the normals of the Depth_Prior's depth and
  the normals of the Walls_Prior's depth. Those normals are for display only: each comes from
  central differences of the points z·(x, y, 1), across and down, crossed, set to unit length and
  turned to face the camera, so a flat wall shows as one flat colour. A pixel on the border, or
  whose own depth or a neighbour's is missing, has no normal. A normal n shows as
  (n + 1) / 2 × 255 in R, G and B, and a pixel with no normal is black. The caller gives the
  Prior_Size Normalised_Coordinates x and y (splatpipe.walls.geometry.normalised_coords), so this
  module needs no camera maths of its own.
- write_prior_file, write_record, write_demo and write_walls_index each turn an OSError into the
  Req 18.6 failure, which names the file and the reason, like "No space left on device".
  write_demo encodes the panel with cv2.imencode as a still JPEG at `demo_jpeg_quality`
  (Req 18.10). cv2 only decodes still images, scales them and encodes JPEGs here, and never reads
  or writes video.
- remove_demos clears out/walls/ at the start of a run (Req 1.10), and remove_scratch removes
  work/walls/scratch/. Nothing else is removed.

From splatpipe, this module imports only splatpipe.errors, splatpipe.jsonio,
splatpipe.train.priors and splatpipe.train.files, and splatpipe.train.colmap for its types.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

from splatpipe.errors import ExpectedFailure
from splatpipe.jsonio import write_json
from splatpipe.train.files import read_training_image, scaled_down
from splatpipe.train.priors import (PRIOR_INDEX, Prior, PriorIndex, prior_path, write_index,
                                    write_prior)

if TYPE_CHECKING:
    from splatpipe.train.colmap import Camera

# Fixed_Constants: the Walls_Step's file and folder names (Req 18, glossary).
DEMOS = "walls"  # the Walls_Demos' folder in out/
RECORD = "walls.json"  # the Walls_Record, in work/walls/
SCRATCH = "scratch"  # the scratch folder, in work/walls/
_DEMO_SUFFIX = ".jpg"

_NO_NORMAL = np.nan  # a display normal's parts at a pixel that has none


def remove_demos(out_folder: Path) -> None:
    """Remove out/walls/ in `out_folder`, the Job's out/, with every Walls_Demo an earlier run left
    there (Req 1.10). Nothing else in out/ is touched. An OSError becomes an ExpectedFailure that
    names the folder and the reason."""
    _remove(out_folder / DEMOS)


def remove_scratch(work_folder: Path) -> None:
    """Remove work/walls/scratch/ in `work_folder`, work/walls/, so no reader and no upload sees
    it. An OSError becomes an ExpectedFailure that names the folder and the reason."""
    _remove(work_folder / SCRATCH)


def demo_path(out_folder: Path, name: str) -> Path:
    """The Walls_Demo of the image `name` in `out_folder`, the Job's out/: walls/<stem>.jpg."""
    return out_folder / DEMOS / f"{name.removesuffix('.jpg')}{_DEMO_SUFFIX}"


def walls_prior(depth_prior: Prior, walls_depth: np.ndarray, walls_normal: np.ndarray,
                covered: np.ndarray) -> Prior:
    """The Walls_Prior of a Prior_Frame from its Walls_Depth (Req 17.7, 17.8, D3, D4, A13).

    `walls_depth` is (h, w) and `walls_normal` (h, w, 3), at the Prior_Size, any float dtype, NaN
    where there's none. `covered` is the frame's Covered_Pixels, which are its Depth_Prior's
    `mask` by definition; a different one is a bug and raises ValueError. The mask is kept
    exactly, so no prior pixel is added or dropped.
    """
    mask = np.asarray(depth_prior.mask, dtype=bool)
    _check_shapes(depth_prior, walls_depth, walls_normal, covered)
    if not np.array_equal(np.asarray(covered, dtype=bool), mask):
        raise ValueError("walls_prior: covered isn't the Depth_Prior's mask")
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        inverse = (1.0 / np.asarray(walls_depth, dtype=np.float64)).astype(np.float32)
    good_depth = mask & np.isfinite(inverse) & (inverse > 0)
    keep_prior = mask & ~good_depth  # the Depth_Prior's value and normal (D3)
    prior_inverse = np.asarray(depth_prior.inverse_depth, dtype=np.float32)
    prior_normal = np.asarray(depth_prior.normal, dtype=np.float32)
    inverse_depth = np.where(good_depth, inverse, np.float32(0))
    inverse_depth = np.where(keep_prior, prior_inverse, inverse_depth).astype(np.float32)
    with np.errstate(over="ignore", invalid="ignore"):
        normal = np.asarray(walls_normal, dtype=np.float64).astype(np.float32)  # as written
        length = np.hypot(np.hypot(normal[..., 0], normal[..., 1]), normal[..., 2])
    good_normal = good_depth & np.isfinite(normal).all(axis=-1) & np.isfinite(length) & (length > 0)
    new_normal = np.where(good_normal[..., np.newaxis], normal, prior_normal)
    new_normal = np.where(mask[..., np.newaxis], new_normal, np.float32(0)).astype(np.float32)
    return Prior(inverse_depth=inverse_depth, normal=new_normal, mask=mask.copy())


def depth_prior_copy(depth_prior: Prior) -> Prior:
    """The Depth_Prior_Copy: the Depth_Prior's three arrays, copied unchanged, bit for bit
    (Req 17.6, D2)."""
    return Prior(inverse_depth=np.array(depth_prior.inverse_depth, copy=True),
                 normal=np.array(depth_prior.normal, copy=True),
                 mask=np.array(depth_prior.mask, copy=True))


def demo_panel(images: Path, name: str, camera: Camera, depth_prior: Prior, walls: Prior,
               x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """One Walls_Demo's panel, (h, 3w, 3) uint8 BGR at the Prior_Size w × h (Req 18.3, A17).

    From left to right: the Frame `name` in `images`, work/cameras/images/, decoded with
    read_training_image against the Shared_Camera `camera` and scaled with scaled_down; the
    normals of the Depth_Prior's depth; the normals of the Walls_Prior `walls`' depth. `x` and
    `y` are the Prior_Size pixels' Normalised_Coordinates, each (h, w). A missing, undecodable or
    wrong-sized Frame is refused with the Train_Step's messages.
    """
    height, width = depth_prior.mask.shape
    for label, coords in (("x", x), ("y", y)):
        if np.shape(coords) != (height, width):
            raise ValueError(f"demo_panel: {label} has the shape {np.shape(coords)}, but the "
                             f"prior size is {(height, width)}")
    frame = scaled_down(read_training_image(images, name, camera), width, height)
    parts = [frame,
             _colours(display_normals(depth_prior.inverse_depth, depth_prior.mask, x, y)),
             _colours(display_normals(walls.inverse_depth, walls.mask, x, y))]
    return np.ascontiguousarray(np.concatenate(parts, axis=1), dtype=np.uint8)


def display_normals(inverse_depth: np.ndarray, mask: np.ndarray, x: np.ndarray,
                    y: np.ndarray) -> np.ndarray:
    """Each pixel's display normal from a depth map, as (h, w, 3) float64, NaN where it has none.

    The points P = z·(x, y, 1), z = 1 / `inverse_depth` where `mask` marks; the central
    differences across, P(right) − P(left), and down, P(below) − P(above), crossed, set to unit
    length and negated when they point away from the camera (n·r above 0). A pixel has no normal
    on the border, where it or one of its four neighbours has no depth, or where the cross
    product isn't finite or has a length of 0. For display only (D4).
    """
    inverse = np.asarray(inverse_depth, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    has_depth = np.asarray(mask, dtype=bool) & np.isfinite(inverse) & (inverse > 0)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        z = np.where(has_depth, 1.0 / np.where(has_depth, inverse, 1.0), np.nan)
    points = np.stack([z * x, z * y, z], axis=-1)
    normals = np.full(points.shape, _NO_NORMAL, dtype=np.float64)
    height, width = has_depth.shape
    if height < 3 or width < 3:
        return normals
    across = points[1:-1, 2:] - points[1:-1, :-2]
    down = points[2:, 1:-1] - points[:-2, 1:-1]
    with np.errstate(over="ignore", invalid="ignore"):
        cross = _cross(across, down)
        length = np.hypot(np.hypot(cross[..., 0], cross[..., 1]), cross[..., 2])
        unit = cross / length[..., np.newaxis]
        rx, ry = x[1:-1, 1:-1], y[1:-1, 1:-1]
        facing = np.where(unit[..., 0] * rx + unit[..., 1] * ry + unit[..., 2] > 0, -1.0, 1.0)
    inner = (has_depth[1:-1, 1:-1] & has_depth[1:-1, 2:] & has_depth[1:-1, :-2]
             & has_depth[2:, 1:-1] & has_depth[:-2, 1:-1]
             & np.isfinite(cross).all(axis=-1) & np.isfinite(length) & (length > 0))
    normals[1:-1, 1:-1] = np.where(inner[..., np.newaxis], unit * facing[..., np.newaxis],
                                   _NO_NORMAL)
    return normals


def write_prior_file(folder: Path, name: str, prior: Prior) -> None:
    """Write the image `name`'s Walls_Prior to its Walls_Prior_File in `folder`, work/walls/, with
    the Prior_Format's write_prior (Req 18.1). An OSError becomes the Req 18.6 failure."""
    path = prior_path(folder, name)
    try:
        write_prior(path, prior)
    except OSError as error:
        raise _not_written(path, error) from error


def write_record(folder: Path, record: dict[str, Any]) -> None:
    """Write the Walls_Record, walls.json in `folder`, work/walls/, with write_json, which checks
    every value first (Req 18.2). An OSError becomes the Req 18.6 failure."""
    path = folder / RECORD
    try:
        write_json(path, record)
    except OSError as error:
        raise _not_written(path, error) from error


def write_demo(path: Path, panel: np.ndarray, quality: int) -> None:
    """Encode `panel` with cv2.imencode as a still JPEG at `quality`, `demo_jpeg_quality`, and
    write it to `path`, out/walls/<stem>.jpg, making out/walls/ when it isn't there (Req 18.3,
    18.10). An OSError becomes the Req 18.6 failure."""
    encoded, data = cv2.imencode(".jpg", panel, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not encoded:
        raise ExpectedFailure(f"The walls demo {path} couldn't be encoded as a JPEG. Set "
                              "demo_frames in [walls] to 0 to make no demos, then run the step "
                              "again.")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data.tobytes())
    except OSError as error:
        raise _not_written(path, error) from error


def write_walls_index(folder: Path, index: PriorIndex) -> None:
    """Write the Walls_Index, depth.json in `folder`, work/walls/, with the Prior_Format's
    write_index, last of all and through a temporary file (Req 18.4, 18.5). An OSError becomes
    the Req 18.6 failure, and write_index has removed the temporary file by then, so no index is
    left (Req 18.7)."""
    try:
        write_index(folder, index)
    except OSError as error:
        raise _not_written(folder / PRIOR_INDEX, error) from error


def _check_shapes(depth_prior: Prior, walls_depth: np.ndarray, walls_normal: np.ndarray,
                  covered: np.ndarray) -> None:
    """Refuse, with ValueError, Walls_Depth, normal or covered arrays that aren't at the
    Depth_Prior's Prior_Size. A mismatch is a bug in the step, not a user's mistake."""
    height, width = np.shape(depth_prior.mask)
    for label, array, shape in (("walls_depth", walls_depth, (height, width)),
                                ("walls_normal", walls_normal, (height, width, 3)),
                                ("covered", covered, (height, width))):
        if np.shape(array) != shape:
            raise ValueError(f"walls_prior: {label} has the shape {np.shape(array)}, but the "
                             f"prior size needs {shape}")


def _cross(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """The cross product of two (..., 3) arrays, part by part with numpy's own loops."""
    ax, ay, az = first[..., 0], first[..., 1], first[..., 2]
    bx, by, bz = second[..., 0], second[..., 1], second[..., 2]
    return np.stack([ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx], axis=-1)


def _colours(normals: np.ndarray) -> np.ndarray:
    """Display normals as (h, w, 3) uint8 BGR: (n + 1) / 2 × 255 in R, G and B, rounded, and black
    where a pixel has no normal (A17). n's x shows in R, y in G and z in B."""
    has_normal = np.isfinite(normals).all(axis=-1)
    safe = np.where(has_normal[..., np.newaxis], normals, -1.0)
    rgb = np.clip(np.rint((safe + 1.0) / 2.0 * 255.0), 0, 255).astype(np.uint8)
    return np.ascontiguousarray(rgb[..., ::-1])


def _remove(path: Path) -> None:
    """Remove the folder at `path` and everything in it, or the file or link at `path`, when it's
    there. An OSError becomes an ExpectedFailure that names the path and the reason."""
    try:
        if path.is_symlink() or path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)
    except OSError as error:
        raise ExpectedFailure(f"The walls step couldn't remove {path} ({_reason(error)}). Check "
                              f"that you can write to {path.parent}, then run the step "
                              "again.") from error


def _not_written(path: Path, error: OSError) -> ExpectedFailure:
    """The Req 18.6 failure for the file at `path`, which couldn't be written."""
    return ExpectedFailure(f"The file {path} couldn't be written ({_reason(error)}). Free some "
                           f"space on the disk, or check that you can write to {path.parent}, "
                           "then run the step again.")


def _reason(error: OSError) -> str:
    """The reason the system gave for an OSError, like "No space left on device"."""
    return error.strerror or str(error)
