"""The Walls_Step's inputs: the Depth_Prior_Set, the Camera_Solve_Output and the Frames_Record,
checked and joined into one FrameInput per Prior_Frame (walls spec, Req 2).

The Walls_Step reads only these, under the names the steps that write them use (Req 2.1):

    work/depth/depth.json and <stem>.npz   the Depth_Prior_Set, which the Depth_Step writes
    work/cameras/images/<name>.jpg         the Frames
    work/cameras/masks/<stem>.png          a Frame's Mask_File, when it has one. Black means ignore.
    work/cameras/sparse/                   the Sparse_Model: cameras.bin, images.bin, points3D.bin
    work/frames/frames.json                the Frames_Record, which the Frames_Step writes

check_inputs (Req 2.2 to 2.5) reads no file. It checks that depth.json is a file, that images/ is
a folder, that each model file is a file and that frames.json is a file, in that order, and
refuses the first one missing, naming it and the step that writes it. The step runs it after the
settings and the Cap_Deadline and before it reads anything else.

read_frames_record (Req 2.7) reads frames.json with Python's json module and uses only `picks`,
and of each Pick only its `number`, `file` and `seconds`. type() is checked exactly, so a bool is
never a number. Every refusal names the field, like picks[3].seconds, and says to run the frames
step on the Job again.

read_inputs (Req 2.6, 2.8 to 2.10) reads the Sparse_Model with the Train_Step's read_model and
the Depth_Prior_Set with read_prior_set and every one of its checks, whose BrokenPriors becomes an
ExpectedFailure with the reader's own message. read_model's refusals are ExpectedFailures already.
Each Prior_Frame, an image with a Depth_Prior, gets the Pick_Number and Frame_Time of the Pick
whose `file` is its name. Every other image in images.bin is left out, so it's never solved,
scored against, projected from or written (D6).

score_size, score_image and score_mask (Req 2.11, 2.12) decode and scale with the Train_Step's own
read_training_image, scaled_down and read_keep_mask, so a Frame or a Mask_File is refused with the
Train_Step's messages, the all-masked one naming [walls]. cv2 only decodes still images and
scales them here, and never reads or writes video (Req 2.13).

The preparation pass (C3) and the scratch folder (C4). prepare_frame runs on the main process,
once per Prior_Frame, before any Frame_Solve, so a Timed_Out_Frame has its arrays too (Req 4.8,
7.6). It decodes the Frame at the Score_Size with score_image and score_mask, works out its Luma,
High_Pass and Bands with images.py, its Scale_Points, Scene_Scale and both Start_Depths with
sources.py, and writes them with np.save into work/walls/scratch/<stem>/, <stem> being the name
without .jpg, as the Prior_Format names a Prior_File. The file names are the constants below,
which solve.py's loaders read, so this module never imports splatpipe.walls.solve (C4).

    At the Score_Size:  luma.npy, high_pass.npy, bands.npy (B, h, w), score_mask.npy,
                        start_depth_score.npy, depth_pixel.npy
    At the Prior_Size:  prior_inverse.npy, prior_normal.npy, prior_mask.npy, start_depth.npy,
                        covered.npy
    scale_points.npy    (n, 3) float64: each Scale_Point's Prior_Size row, column and camera z
    numbers.npy         (3,) float64: k, m and the Scale_Point count

The stages after the Frame_Solves write one file name per stage, so a stage never overwrites a
file another frame of the same stage may still read: every stage reads the depths as the stage
before left them, whatever order the workers take the frames in (Req 14.4, A11). The names are
here with the others, and solve.py's stage tasks use them:

    current_depth.npy, current_normal.npy   the Solved_Depth and its normals (Trusted_Frames)
    facets.npy, facet_numbers.npy           the Facets: a (ph, pw) int32 label image, −1 off every
                                            Facet, and (F, 4) float64, each Facet's normal and m
    fused_depth.npy                         the Fused_Depth (Trusted_Frames)
    fallback_depth.npy, fallback_normal.npy the Fallback_Depth and its normals (Failed_Frames)
    corrected_depth.npy                     after the correction
    flat_depth.npy, walls_normal.npy        after the flatten, its normals the Walls_Depth's last
    walls_depth.npy                         after the despeckle: the Walls_Depth

From splatpipe, this module imports only splatpipe.errors, splatpipe.train's colmap, files,
priors, prior_views and scene modules, and splatpipe.walls's geometry, images and sources.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from splatpipe.errors import ExpectedFailure
from splatpipe.train.colmap import Camera, ModelImage, SparseModel, read_model
from splatpipe.train.files import (CAMERAS, IMAGES, MASKS, MODEL_FILES, SPARSE, read_keep_mask,
                                   read_training_image, scaled_down)
from splatpipe.train.priors import (PRIOR_FOLDER, PRIOR_INDEX, BrokenPriors, Prior, PriorEntry,
                                    PriorSet, read_prior_set)
from splatpipe.train.prior_views import prior_taps
from splatpipe.train.scene import training_size
from splatpipe.walls.geometry import median_inverse_depth
from splatpipe.walls.images import bands, high_pass, luma
from splatpipe.walls.sources import scale_points, scene_scale, start_depth

# Fixed_Constants: the Frames_Record's folder and name, the Frames_Step's contract (Req 2.1).
FRAMES = "frames"  # the Frames_Step's folder in work/
FRAMES_RECORD = "frames.json"  # the Frames_Record, in work/frames/

# The scratch folder's file names, in work/walls/scratch/<stem>/ (C3, C4). The preparation pass.
LUMA = "luma.npy"
HIGH_PASS = "high_pass.npy"
BANDS = "bands.npy"
SCORE_MASK = "score_mask.npy"
START_DEPTH_SCORE = "start_depth_score.npy"
DEPTH_PIXEL = "depth_pixel.npy"
PRIOR_INVERSE = "prior_inverse.npy"
PRIOR_NORMAL = "prior_normal.npy"
PRIOR_MASK = "prior_mask.npy"
START_DEPTH = "start_depth.npy"
COVERED = "covered.npy"
SCALE_POINTS = "scale_points.npy"
NUMBERS = "numbers.npy"
# The Frame_Solve.
CURRENT_DEPTH = "current_depth.npy"
CURRENT_NORMAL = "current_normal.npy"
FACETS = "facets.npy"
FACET_NUMBERS = "facet_numbers.npy"
# The cross-frame stages, one name per stage, so none overwrites what another frame still reads.
FUSED_DEPTH = "fused_depth.npy"
FALLBACK_DEPTH = "fallback_depth.npy"
FALLBACK_NORMAL = "fallback_normal.npy"
CORRECTED_DEPTH = "corrected_depth.npy"
FLAT_DEPTH = "flat_depth.npy"
WALLS_NORMAL = "walls_normal.npy"
WALLS_DEPTH = "walls_depth.npy"

_FRAMES_AGAIN = "Run the frames step on this job again."
_RUN = "uv run python -m splatpipe.run --job"  # the Job's command, for the --fresh message


@dataclass(frozen=True)
class FramePick:
    """The fields of one Pick of the Frames_Record that the Walls_Step uses (Req 2.7)."""

    number: int  # its Pick_Number, a whole number of 0 or more
    file: str  # the image's name in work/cameras/images/ and in images.bin
    seconds: float  # its Frame_Time, finite


@dataclass(frozen=True)
class FrameInput:
    """One Prior_Frame: an image of images.bin with a Depth_Prior (Req 2.9, 2.10)."""

    name: str  # the image's name in images.bin
    pick_number: int  # its Pick's number (D8)
    frame_time: float  # its Pick's seconds (D8)
    image: ModelImage  # its image in images.bin, with its Image_Pose
    prior: Prior  # its Depth_Prior, at the Prior_Size
    entry: PriorEntry  # its object in the Prior_Index


class PrepareSettings(Protocol):
    """The [walls] settings the preparation pass reads. WallsSettings has them all."""

    score_long_edge: int
    mask_threshold: int
    high_pass_sigma: float
    band_sigmas: tuple[float, ...]
    scale_mad_gate: float
    scale_min_points: int


@dataclass(frozen=True)
class Prepared:
    """What the preparation pass gives the main process for one Prior_Frame (Req 7.6, C3)."""

    name: str
    folder: Path  # its scratch folder, work/walls/scratch/<stem>/
    k: float  # its Scene_Scale
    scale_points: int  # its number of Scale_Points
    m: float  # the Prior_Size Start_Depth's Median_Inverse_Depth over the Covered_Pixels


def frame_folder(scratch: Path, name: str) -> Path:
    """The scratch folder of the Prior_Frame `name` in `scratch`, work/walls/scratch/: <stem>/,
    the name without .jpg, as the Prior_Format names a Prior_File."""
    return scratch / name.removesuffix(".jpg")


def prepare_frame(scratch: Path, cameras: Path, model: SparseModel, frame: FrameInput,
                  settings: PrepareSettings) -> Prepared:
    """Write the Prior_Frame `frame`'s scratch arrays to frame_folder(`scratch`, its name) with
    np.save, and give its Scene_Scale, Scale_Point count and m (Req 7, C3).

    `cameras` is work/cameras/ and `model` the Sparse_Model, whose camera is the Shared_Camera.
    The Frame is decoded at the Score_Size with score_image and score_mask, so a missing, broken
    or wrong-sized Frame or Mask_File is refused with the Train_Step's messages. Its Luma,
    High_Pass and Bands come from images.py; its Scale_Points, from the Prior_Depth on its
    Covered_Pixels, its Scene_Scale k and its two Start_Depths from sources.py, the Score_Size
    one through prior_taps from the Prior_Size to the Score_Size.
    """
    camera = model.camera
    size = score_size(camera, settings.score_long_edge)
    grey = luma(score_image(cameras, frame.name, camera, size))
    keep = score_mask(cameras, frame.name, camera, size, settings.mask_threshold)
    prior = frame.prior
    inverse = np.asarray(prior.inverse_depth, dtype=np.float32)
    mask = np.asarray(prior.mask, dtype=bool)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        prior_depth = np.where(mask, 1.0 / inverse.astype(np.float64), np.nan)
    pixels, z, depth_at = scale_points(model, frame.image, camera, mask, prior_depth)
    k, count = scene_scale(z, depth_at, settings.scale_mad_gate, settings.scale_min_points)
    prior_height, prior_width = mask.shape
    taps = prior_taps(prior_width, prior_height, *size)
    prior_start, score_start, depth_pixel = start_depth(inverse, mask, taps, k)
    m = median_inverse_depth(prior_start, mask) if mask.any() else float("nan")
    points = np.stack([pixels[:, 1].astype(np.float64), pixels[:, 0].astype(np.float64),
                       np.asarray(z, dtype=np.float64)], axis=-1).reshape(-1, 3)
    arrays = {
        LUMA: grey, HIGH_PASS: high_pass(grey, settings.high_pass_sigma),
        BANDS: np.stack(bands(grey, settings.band_sigmas)), SCORE_MASK: keep,
        START_DEPTH_SCORE: score_start, DEPTH_PIXEL: depth_pixel,
        PRIOR_INVERSE: inverse, PRIOR_NORMAL: np.asarray(prior.normal, dtype=np.float32),
        PRIOR_MASK: mask, START_DEPTH: prior_start, COVERED: mask, SCALE_POINTS: points,
        NUMBERS: np.array([k, m, count], dtype=np.float64),
    }
    folder = frame_folder(scratch, frame.name)
    folder.mkdir(parents=True, exist_ok=True)
    for file_name, array in arrays.items():
        np.save(folder / file_name, np.ascontiguousarray(array), allow_pickle=False)
    return Prepared(name=frame.name, folder=folder, k=float(k), scale_points=int(count),
                    m=float(m))


def check_inputs(job_folder: Path) -> None:
    """Refuse a Job in `job_folder` without one of the Walls_Step's inputs (Req 2.2 to 2.5).

    work/depth/depth.json must be a file, work/cameras/images/ a folder, each of the Sparse_Model's
    three files a file and work/frames/frames.json a file, checked in that order. The first one
    missing is refused, naming it and the step that writes it. Nothing is read.
    """
    work = job_folder / "work"
    index = work / PRIOR_FOLDER / PRIOR_INDEX
    if not index.is_file():
        raise _no_depth_index(index)
    cameras = work / CAMERAS
    images = cameras / IMAGES
    if not images.is_dir():
        raise _missing(images, "camera solve")
    for name in MODEL_FILES:
        path = cameras / SPARSE / name
        if not path.is_file():
            raise _missing(path, "camera solve")
    record = work / FRAMES / FRAMES_RECORD
    if not record.is_file():
        raise _missing(record, "frames")


def read_frames_record(path: Path) -> tuple[FramePick, ...]:
    """The Picks of the Frames_Record at `path`, in the record's order (Req 2.7).

    It's read with json and must be a JSON object whose `picks` is a list of objects, each with a
    `number`, a whole number of 0 or more, a `file`, a string, and a `seconds`, a finite int or
    float. No `number` and no `file` may appear twice. Every other field is ignored. Each refusal
    names the field and says to run the frames step on the Job again.
    """
    data = _read_json(path)
    if type(data) is not dict:
        raise _refused(f"The frames record {path} must hold a JSON object.")
    if "picks" not in data:
        raise _refused(f"The frames record {path} has no picks.")
    entries = data["picks"]
    if type(entries) is not list:
        raise _wrong(path, "picks", "a list of objects")
    picks = tuple(_pick(path, entry, f"picks[{place}]") for place, entry in enumerate(entries))
    _refuse_repeats(path, [pick.number for pick in picks], "number")
    _refuse_repeats(path, [pick.file for pick in picks], "file")
    return picks


def read_inputs(job_folder: Path) -> tuple[SparseModel, PriorSet, tuple[FrameInput, ...]]:
    """The Sparse_Model, the Depth_Prior_Set and one FrameInput per Prior_Frame, in Pick_Number
    order, of the Job in `job_folder`, once check_inputs has passed (Req 2.6, 2.8 to 2.10).

    The Sparse_Model is read with read_model, then the Depth_Prior_Set with read_prior_set and
    every one of its checks, then the Frames_Record. A BrokenPriors becomes an ExpectedFailure
    with the reader's message. A Prior_Frame whose name is no Pick's `file` is refused with the
    Req 2.8 message. An image in images.bin with no Depth_Prior is left out.
    """
    work = job_folder / "work"
    model = read_model(work / CAMERAS / SPARSE)
    priors = _read_priors(work / PRIOR_FOLDER, model)
    record = work / FRAMES / FRAMES_RECORD
    picks = {pick.file: pick for pick in read_frames_record(record)}
    images = {image.name: image for image in model.images}
    frames = [_frame_input(entry, priors, images, picks, record, job_folder.name)
              for entry in priors.index.priors]
    return model, priors, tuple(sorted(frames, key=lambda frame: frame.pick_number))


def score_size(camera: Camera, long_edge: int) -> tuple[int, int]:
    """The Score_Size of the Shared_Camera `camera`, as (width, height): the Training_Size rule
    with `long_edge`, score_long_edge, so the longer edge is at most `long_edge` and no Frame is
    scaled up (Req 2.11)."""
    return training_size(camera.width, camera.height, long_edge)


def score_image(cameras: Path, name: str, camera: Camera, size: tuple[int, int]) -> np.ndarray:
    """The Frame `name` in `cameras`, work/cameras/, as cv2 decodes it, in BGR, at the Score_Size
    `size`, (width, height) (Req 2.11).

    It's decoded and checked against `camera` with the Train_Step's read_training_image, so a
    Frame with no file, one OpenCV can't decode and one that isn't the Shared_Camera's size are
    refused with the Train_Step's messages, and scaled with its scaled_down, INTER_AREA.
    """
    image = read_training_image(cameras / IMAGES, name, camera)
    return scaled_down(image, *size)


def score_mask(cameras: Path, name: str, camera: Camera, size: tuple[int, int],
               threshold: int) -> np.ndarray:
    """The Score_Mask of the Frame `name`, (height, width) bool at the Score_Size `size`,
    (width, height), true where a pixel is kept (Req 2.12).

    When work/cameras/masks/<stem>.png is a file, it's read with the Train_Step's read_keep_mask
    and `threshold`, mask_threshold from [walls], so it's refused by the Train_Step's rules, the
    all-masked message naming [walls]. A Frame with no Mask_File keeps every pixel (D10, A18).
    """
    width, height = size
    keep = read_keep_mask(cameras / MASKS, name, (camera.width, camera.height), width, height,
                          threshold, section="walls")
    if keep is None:
        return np.ones((height, width), dtype=bool)
    return keep


def _read_priors(folder: Path, model: SparseModel) -> PriorSet:
    """The Depth_Prior_Set in `folder`, work/depth/, read with read_prior_set and every one of its
    checks. A BrokenPriors becomes an ExpectedFailure with its message (Req 2.6). A depth.json
    that went away since check_inputs is the Req 2.2 refusal."""
    try:
        priors = read_prior_set(folder, model)
    except BrokenPriors as error:
        raise ExpectedFailure(f"The walls step can't use the depth priors: {error}. Run the depth "
                              "step on this job again. The job goes on and trains as it would "
                              "without the walls step.") from error
    if priors is None:
        raise _no_depth_index(folder / PRIOR_INDEX)
    return priors


def _frame_input(entry: PriorEntry, priors: PriorSet, images: Mapping[str, ModelImage],
                 picks: Mapping[str, FramePick], record: Path, job_id: str) -> FrameInput:
    """The FrameInput of the Prior_Frame whose Prior_Index object is `entry`, given its Pick's
    number and seconds (Req 2.8, 2.9). read_prior_set has checked that its name is in
    images.bin."""
    pick = picks.get(entry.name)
    if pick is None:
        raise ExpectedFailure(f"The image {entry.name!r} has a depth prior, but no pick in the "
                              f"frames record {record} has it as its file, so frames.json doesn't "
                              "match the camera solve. Run the job again with --fresh: "
                              f"{_RUN} {job_id} --fresh")
    return FrameInput(name=entry.name, pick_number=pick.number, frame_time=pick.seconds,
                      image=images[entry.name], prior=priors.priors[entry.name], entry=entry)


def _read_json(path: Path) -> object:
    """The JSON value in the file at `path`, read with json. A file that can't be read or parsed
    is refused with the reason."""
    try:
        return json.loads(path.read_bytes())
    except OSError as error:
        detail = error.strerror or str(error)
    except (ValueError, RecursionError) as error:  # not UTF-8, not JSON, nested too deeply
        detail = str(error).rstrip(".") or type(error).__name__
    raise _refused(f"The frames record {path} can't be read as JSON ({detail}).")


def _pick(path: Path, entry: object, field: str) -> FramePick:
    """The Pick of one entry of picks, whose place in the record is `field`, like picks[3]."""
    if type(entry) is not dict:
        raise _wrong(path, field, "a JSON object")
    number = _field(path, entry, "number", field)
    if type(number) is not int or number < 0:
        raise _wrong(path, f"{field}.number", "a whole number, 0 or more")
    file = _field(path, entry, "file", field)
    if type(file) is not str:
        raise _wrong(path, f"{field}.file", "a file name in quotes")
    seconds = _seconds(path, _field(path, entry, "seconds", field), f"{field}.seconds")
    return FramePick(number=number, file=file, seconds=seconds)


def _field(path: Path, entry: Mapping[str, Any], key: str, field: str) -> Any:
    """The value of `key` in the Pick at `field`, refused when it's missing."""
    if key not in entry:
        raise _refused(f"The frames record {path} has no {field}.{key}.")
    return entry[key]


def _seconds(path: Path, value: object, field: str) -> float:
    """A Pick's seconds as a float: an int or a float, never a bool, that's finite as a float."""
    seconds = float("nan")
    if type(value) is float:
        seconds = value
    elif type(value) is int:
        try:
            seconds = float(value)
        except OverflowError:  # an int too large for a float
            pass
    if not np.isfinite(seconds):
        raise _wrong(path, field, "a finite number")
    return seconds


def _refuse_repeats(path: Path, values: Sequence[object], key: str) -> None:
    """Refuse a `key`, number or file, that two Picks share, naming both places."""
    first: dict[object, int] = {}
    for place, value in enumerate(values):
        if value in first:
            raise _refused(f"In the frames record {path}, picks[{place}].{key} is {value!r}, the "
                           f"same as picks[{first[value]}].{key}, but each pick must have its "
                           f"own {key}.")
        first[value] = place


def _wrong(path: Path, field: str, wanted: str) -> ExpectedFailure:
    """The Req 2.7 failure for the field `field`, saying what it must be."""
    return _refused(f"In the frames record {path}, {field} must be {wanted}.")


def _refused(problem: str) -> ExpectedFailure:
    """A Req 2.7 refusal: `problem`, then the advice to run the frames step again."""
    return ExpectedFailure(f"{problem} {_FRAMES_AGAIN}")


def _missing(path: Path, writer: str) -> ExpectedFailure:
    """The Req 2.3 or 2.4 failure for an input that isn't there, `writer` naming the step that
    writes it, like "camera solve"."""
    return ExpectedFailure(f"The walls step needs {path}, which the {writer} step writes, but it "
                           f"isn't there. Run the {writer} step on this job first.")


def _no_depth_index(path: Path) -> ExpectedFailure:
    """The Req 2.2 failure for a depth.json that isn't a file."""
    return ExpectedFailure(f"The walls step needs {path}, which the depth step writes, but it "
                           "isn't there, so the depth step has to run on this job first. The job "
                           "goes on and trains as it would without the walls step.")
