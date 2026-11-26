"""The Cameras_Smoke_Check: the Option_Check, then the Smoke_Solve with the real COLMAP (cameras
spec, Req 18, D15).

cloud/smoke_runner.py's `cameras` check turns [cloud.smoke]'s numbers into a SmokeSizes and calls
run_cameras_smoke on the machine image, which has no pytest. The laptop's -h test and
integration test call the same code, with Homebrew's COLMAP, and the smoke check's other tests
call its parts with stand-ins, so the module holds everything both sides need (Req 18.1).

- The Smoke_Picks' scene, moved from tests/test_cameras_real.py: the inside corner of a room, a
  floor, a back wall and a side wall, each 4 m × 4 m, so the scene isn't planar. The world has x
  to the right, y down and z into the room, in metres, so up is -y. The floor is the plane y = 0,
  from x = -2 to 2 and z = 0 to 4, the back wall the plane z = 4 and the side wall the plane
  x = 2, both rising 4 m from the floor, so the three meet at the corner (2, 0, 4). Each has its
  own texture of 1 cm texels, from its own seed, 1, 2 or 3: grey noise smoothed with Gaussians of
  2, 6 and 18 texels, tinted, then 60 rectangles and 60 discs drawn with cv2, then a blur of 1
  texel. The camera's centres are evenly spread along a level line 1.2 m above the floor, from
  (-1, -1.2, 0) to (1, -1.2, 0), so it moves 2 m sideways. It turns only about the vertical, with
  no pitch and no roll, to keep the corner's vertical edge in the Pick's middle column. These
  shapes and seeds are Fixed_Constants of the scene, like the training check's radius and
  half-side. The camera's width, height and f, with its principal point at the Pick's centre in
  COLMAP's convention that the top left pixel's centre is at (0.5, 0.5), and the number of Picks
  come from the SmokeSizes (Req 18.9). The render casts one ray per pixel with numpy, through the
  pixel's centre, and samples the nearest rectangle's texture bilinearly where the ray meets it.
- render_pick gives one Pick and, for each pixel, the rectangle it sees, for the scene's
  self-check.
- write_smoke_picks writes a work/frames/ as the Frames_Contract lays it out, with the names of
  splatpipe/frames/files.py: images/NNNNNN.jpg and grey/NNNNNN.jpg, from grey_copy, at the
  Preset's jpeg_quality, each Pick with a black corner painted into its top left tenth of each
  side, mask.png, 0 over that corner and 255 elsewhere, and frames.json last, with mask_used true
  and made-up times and scores, which the Cameras_Step doesn't read.
- passed_options and listed_options give the options that the step's command lines pass each of
  the seven COLMAP commands, for every Preset, both Camera_Models and with and without the Mask,
  and the options that a `colmap <command> -h` text lists (F16).
- check_options is the Option_Check (Req 18.3, 18.6, C20): `colmap <command> -h` for each of the
  seven commands through run_colmap, so run_colmap stays the only code that starts COLMAP, each
  within check_timeout_seconds and with its Command_Log in the folder it's given.
- smoke_settings and run_smoke_solve are the Smoke_Solve (Req 18.4, C17, C18): the step's own
  work() on the Smoke_Picks, with the [cameras] settings of [jobs].default_preset and every
  Time_Limit key at cameras_timeout_seconds, on a StepContext built here, with an in-memory Job in
  <scratch>/job/ whose Job_Record is never written. So it needs no video, no ffprobe and no
  Jobs_Folder. work() reads the step's real TOOLS and SPLATPIPE_CAP_DEADLINE, as in any run.
- check_smoke_solve is pure (Req 18.5, 18.7, C19): enough placed Picks, and f, fx and fy each
  within the tolerance.
- run_cameras_smoke runs the three in order and gives back the line for the Smoke_Test's log and
  what the Smoke_Solve found (Req 18.8). It writes only inside the scratch folder it's given.

cv2 draws, blurs and encodes the Smoke_Picks' images, and never reads or writes video (Req
12.14). The module imports only the standard library, numpy, cv2 and splatpipe (Req 12.13, 18.1).
From splatpipe, it imports splatpipe.config, splatpipe.errors, splatpipe.frames.files,
splatpipe.jobs, splatpipe.jsonio, splatpipe.step, and the package's work and its other modules.
"""

from __future__ import annotations

import functools
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from splatpipe import cameras
from splatpipe.cameras import commands
from splatpipe.cameras.colmap import PROGRAM, ColmapRun, colmap_env, exit_text, run_colmap
from splatpipe.cameras.commands import RADIAL_FISHEYE, Paths
from splatpipe.cameras.frames_input import FramesInput, Pick
from splatpipe.cameras.model import read_model
from splatpipe.cameras.settings import TIMEOUT_KEYS, CamerasSettings, read_settings
from splatpipe.cameras.solve import placed_needed
from splatpipe.config import PRESETS, Config, StepSettings
from splatpipe.errors import PipelineError
from splatpipe.frames.files import (GREY, IMAGES, MASK, RECORD, grey_copy, pick_name, write_jpeg,
                                    write_png)
from splatpipe.jobs import Job, JobRecord, check_job_id
from splatpipe.jsonio import read_json, write_json
from splatpipe.step import StepContext

JOB, HELP = "job", "help"  # the Smoke_Solve's Job_Folder and the Option_Check's folder in scratch

_STEP = "cameras"  # the Cameras_Step's name and its section in Config
_FRAMES = "frames"  # the Frames_Step's section, whose jpeg_quality the Smoke_Picks take (C18)
# The seven COLMAP commands the step runs, in the order it runs them.
_COMMANDS = tuple(TIMEOUT_KEYS)
_CAMERA_MODELS = ("RADIAL", RADIAL_FISHEYE)  # the Camera_Models the Option_Check covers (D19)
# An option in the list that colmap <command> -h prints. Boost puts each option two spaces in, as
# in "  --SiftExtraction.peak_threshold arg (=0.00667)", and --help in "  -h [ --help ]". A
# description that wraps goes on further in, and each of glog's lines starts with its prefix, so
# neither can pass for an option (F16).
_HELP_OPTION = re.compile(r"^  (?:-\w \[ )?(--[^\s\]]+)", re.MULTILINE)

# The Smoke_Solve's in-memory Job_Record, which the step never reads (C17): it names the
# Smoke_Picks, and its Used_Part is the made-up one of frames.json, one Pick every _PICK_SECONDS.
_SOURCE_NAME = "cameras smoke picks"
_VIDEO_FILE = "video.mp4"
_PICK_SECONDS = 1.0

# Fixed_Constants: the Smoke_Picks' scene (D15).
# The camera's path, in the world's metres, with x to the right, y down and z into the room.
_ROOM_CORNER = (2.0, 0.0, 4.0)  # where the floor and the two walls meet
_EYE_HEIGHT = 1.2  # the camera's height above the floor, the plane y = 0
_PATH_START = (-1.0, -_EYE_HEIGHT, 0.0)  # Pick 0's camera centre
_PATH = (2.0, 0.0, 0.0)  # from Pick 0's centre to the last Pick's: 2 m sideways, level
# The textures.
_TEXEL = 0.01  # each texel's side, in metres
_NOISE_SIGMAS = (2.0, 6.0, 18.0)  # the noise's three scales, as Gaussian sigmas in texels
_NOISE_CONTRAST = 0.18  # the noise's standard deviation, as a share of full brightness
_SHAPES = 60  # rectangles drawn on each texture, and as many discs
_RECTANGLE_SIDES = (6.0, 48.0)  # the range of each rectangle side, in texels
_DISC_RADII = (3, 24)  # the range of each disc's radius, in texels, both ends included
_EDGE_SIGMA = 1.0  # the last blur, a Gaussian sigma in texels, which softens the shapes' edges
# The black corner painted into every Pick is its top left width // PAINTED_PART ×
# height // PAINTED_PART pixels, 32 × 24 at 320 × 240. mask.png is 0 there and 255 elsewhere.
PAINTED_PART = 10


@dataclass(frozen=True)
class SmokeSizes:
    """The Cameras_Smoke_Check's numbers, from [cloud.smoke], which cloud/smoke_runner.py passes
    in."""

    frames: int  # cameras_frames
    width: int  # cameras_width
    height: int  # cameras_height
    focal_pixels: float  # cameras_focal_pixels, the Smoke_Picks' true f
    placed_share: float  # cameras_placed_share
    focal_tolerance: float  # cameras_focal_tolerance, a share of focal_pixels
    timeout_seconds: float  # cameras_timeout_seconds, every Time_Limit key of the Smoke_Solve
    help_timeout_seconds: float  # check_timeout_seconds, each colmap <command> -h


@dataclass(frozen=True)
class SmokeSolve:
    """What the Smoke_Solve found, read from the files the step wrote."""

    folder: Path  # the Smoke_Solve's Job_Folder, <scratch>/job, resolved
    picks: int  # the report count picks
    placed: int  # the report count placed_frames
    mapper: str  # "global" or "incremental", from solve.json
    focal: float  # the Final_Model's f
    fx: float  # the Levelled_Model's PINHOLE fx
    fy: float  # and its fy


@dataclass(frozen=True)
class CamerasSmoke:
    """What run_cameras_smoke gives back."""

    line: str  # the check's line for the Smoke_Test's log
    solve: SmokeSolve  # what the Smoke_Solve found, for the integration test's own checks


@dataclass(frozen=True)
class Rectangle:
    """One rectangle of the scene: the corner its texture starts at and its two edges from there,
    in metres, and the seed of its texture. The texture's columns run along `across` and its rows
    along `down`, one texel every _TEXEL metres."""

    name: str
    origin: tuple[float, float, float]
    across: tuple[float, float, float]
    down: tuple[float, float, float]
    seed: int


# The scene's three rectangles, which meet at _ROOM_CORNER.
RECTANGLES = (
    Rectangle("floor", origin=(-2.0, 0.0, 0.0), across=(4.0, 0.0, 0.0),
              down=(0.0, 0.0, 4.0), seed=1),
    Rectangle("back wall", origin=(-2.0, -4.0, 4.0), across=(4.0, 0.0, 0.0),
              down=(0.0, 4.0, 0.0), seed=2),
    Rectangle("side wall", origin=(2.0, -4.0, 0.0), across=(0.0, 0.0, 4.0),
              down=(0.0, 4.0, 0.0), seed=3),
)


# The Smoke_Picks

def render_pick(number: int, sizes: SmokeSizes) -> tuple[np.ndarray, np.ndarray]:
    """What Pick `number`'s camera sees: a BGR uint8 image of sizes.width × sizes.height, and for
    each pixel the index in RECTANGLES of the rectangle its ray hits first, or -1 for a ray that
    hits none, whose pixel stays black. The painted corner isn't in it."""
    centre, rotation = _camera(number, sizes.frames)
    rays = _rays(rotation, sizes)
    hits = [_hit(rectangle, centre, rays) for rectangle in RECTANGLES]
    distances = np.stack([distance for distance, _ in hits])
    seen = np.where(np.isfinite(distances.min(axis=0)), distances.argmin(axis=0), -1)
    image = np.zeros((sizes.height, sizes.width, 3))
    for index, (rectangle, (_, place)) in enumerate(zip(RECTANGLES, hits)):
        mine = seen == index
        image[mine] = _sample(_texture(rectangle), place[mine])
    return np.rint(image).astype(np.uint8), seen


def write_smoke_picks(folder: Path, sizes: SmokeSizes, jpeg_quality: int) -> None:
    """Write the Smoke_Picks into `folder`, a work/frames/ as the Frames_Step leaves it.

    Pick by Pick, its render with the black corner painted in goes to images/ as a JPEG at
    `jpeg_quality`, and its Grey_Copy, from grey_copy, to grey/ under the same name at the same
    quality. Then come mask.png and, last, frames.json. The JPEGs and the PNG come from
    cv2.imencode, through the Frames_Step's own writers, and frames.json from write_json.
    """
    for name in (IMAGES, GREY):
        (folder / name).mkdir(parents=True, exist_ok=True)
    painted_width, painted_height = _painted_size(sizes)
    for number in range(sizes.frames):
        pick, _ = render_pick(number, sizes)
        pick[:painted_height, :painted_width] = 0
        write_jpeg(folder / IMAGES / pick_name(number), pick, jpeg_quality)
        write_jpeg(folder / GREY / pick_name(number), grey_copy(pick), jpeg_quality)
    mask = np.full((sizes.height, sizes.width), 255, dtype=np.uint8)
    mask[:painted_height, :painted_width] = 0
    write_png(folder / MASK, mask)
    write_json(folder / RECORD, _frames_record(sizes, float(np.mean(mask == 0))))


def _painted_size(sizes: SmokeSizes) -> tuple[int, int]:
    """The width and height of the black corner painted into every Pick, in pixels."""
    return sizes.width // PAINTED_PART, sizes.height // PAINTED_PART


def _frames_record(sizes: SmokeSizes, coverage: float) -> dict[str, Any]:
    """The Smoke_Picks' Frames_Record, with every field in frames_record's order.

    The made-up Used_Part, one Pick every _PICK_SECONDS, is kept whole, and each Pick is its own
    Score_Frame. The Mask is used, with `coverage` its share of 0s. The scores are made up too,
    because the Cameras_Step doesn't read them.
    """
    end = sizes.frames * _PICK_SECONDS
    return {
        "colour_transfer": "bt709", "tone_mapped": False,
        "still_start_seconds": 0.0, "still_end_seconds": 0.0,
        "kept_start_seconds": 0.0, "kept_end_seconds": end,
        "target_picks": sizes.frames, "jolt_filter_skipped": False,
        "pick_width": sizes.width, "pick_height": sizes.height,
        "mask_coverage": coverage, "mask_used": True, "mask_file": MASK,
        "picks": [{"number": number, "file": pick_name(number), "frame_number": number,
                   "seconds": number * _PICK_SECONDS, "sharpness": 100.0,
                   "motion": 0.0 if number == 0 else 2.0, "contrast": 40.0}
                  for number in range(sizes.frames)]}


@functools.cache
def _texture(rectangle: Rectangle) -> np.ndarray:
    """The rectangle's texture, as float32 BGR values from 0 to 255, made from its seed alone.

    Grey noise smoothed at each of the _NOISE_SIGMAS, each band brought to the same standard
    deviation, then added up and tinted with a random colour. Then _SHAPES rectangles and _SHAPES
    discs, drawn in turn, and last a blur of _EDGE_SIGMA texels. It's made once per process.
    """
    rng = np.random.default_rng(rectangle.seed)
    rows = round(float(np.linalg.norm(rectangle.down)) / _TEXEL)
    columns = round(float(np.linalg.norm(rectangle.across)) / _TEXEL)
    noise = np.zeros((rows, columns), dtype=np.float32)
    for sigma in _NOISE_SIGMAS:
        band = cv2.GaussianBlur(rng.standard_normal((rows, columns), dtype=np.float32), (0, 0),
                                sigma)
        noise += band / band.std()
    grey = np.clip(0.5 + _NOISE_CONTRAST * noise / np.sqrt(len(_NOISE_SIGMAS)), 0.0, 1.0)
    image = np.rint(255 * grey[..., None] * rng.uniform(0.6, 1.0, 3)).astype(np.uint8)
    for _ in range(_SHAPES):
        _draw_rectangle(image, rng)
        _draw_disc(image, rng)
    return cv2.GaussianBlur(image.astype(np.float32), (0, 0), _EDGE_SIGMA)


def _draw_rectangle(image: np.ndarray, rng: np.random.Generator) -> None:
    """Fill a rectangle with sides in _RECTANGLE_SIDES, at a random angle, place and colour."""
    rows, columns = image.shape[:2]
    centre = (float(rng.uniform(0, columns)), float(rng.uniform(0, rows)))
    sides = (float(rng.uniform(*_RECTANGLE_SIDES)), float(rng.uniform(*_RECTANGLE_SIDES)))
    corners = cv2.boxPoints((centre, sides, float(rng.uniform(0, 180))))
    cv2.fillConvexPoly(image, np.rint(corners).astype(np.int32), _colour(rng), cv2.LINE_AA)


def _draw_disc(image: np.ndarray, rng: np.random.Generator) -> None:
    """Fill a disc with a radius in _DISC_RADII, at a random place and colour."""
    rows, columns = image.shape[:2]
    centre = (int(rng.integers(0, columns)), int(rng.integers(0, rows)))
    radius = int(rng.integers(_DISC_RADII[0], _DISC_RADII[1], endpoint=True))
    cv2.circle(image, centre, radius, _colour(rng), -1, cv2.LINE_AA)


def _colour(rng: np.random.Generator) -> tuple[int, int, int]:
    """A random BGR colour."""
    blue, green, red = (int(value) for value in rng.integers(0, 256, 3))
    return blue, green, red


def _camera(number: int, picks: int) -> tuple[np.ndarray, np.ndarray]:
    """Pick `number`'s camera of `picks`: its centre in the world, and its rotation R from the
    world to the camera, so a point X is at R (X - centre) in the camera, with x to the right,
    y down and z ahead, as in COLMAP.

    The centres are evenly spread along _PATH, from _PATH_START for Pick 0 to its end for the last
    Pick. The rows of R are the camera's axes in the world. Its y axis is the world's down, so it
    has no pitch and no roll, its z axis is level and points at _ROOM_CORNER's vertical edge, and
    its x axis is y × z.
    """
    centre = np.array(_PATH_START) + np.array(_PATH) * number / (picks - 1)
    ahead = np.array(_ROOM_CORNER) - centre
    ahead[1] = 0.0  # level, so the camera only turns about the vertical
    ahead /= np.linalg.norm(ahead)
    down = np.array([0.0, 1.0, 0.0])
    return centre, np.stack([np.cross(down, ahead), down, ahead])


def _rays(rotation: np.ndarray, sizes: SmokeSizes) -> np.ndarray:
    """Each pixel's ray direction in the world, through the pixel's centre, as an array of
    height × width × 3. In the camera, the ray of the pixel centred at (u, v) is
    ((u - cx) / f, (v - cy) / f, 1), with f = sizes.focal_pixels and (cx, cy) the Pick's
    centre."""
    u, v = np.meshgrid(np.arange(sizes.width) + 0.5, np.arange(sizes.height) + 0.5)
    cx, cy, focal = sizes.width / 2, sizes.height / 2, sizes.focal_pixels
    in_camera = np.stack([(u - cx) / focal, (v - cy) / focal, np.ones_like(u)], axis=-1)
    return in_camera @ rotation  # each direction d becomes Rᵀ d


def _hit(rectangle: Rectangle, centre: np.ndarray,
         rays: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Where each ray from `centre` meets the rectangle's plane: how far along the ray, in units
    of its direction, or inf when the ray misses the rectangle or meets it behind the camera, and
    the point's fractions along `across` and `down`, from 0 to 1 inside the rectangle."""
    origin, across, down = (np.array(edge) for edge in
                            (rectangle.origin, rectangle.across, rectangle.down))
    normal = np.cross(across, down)
    with np.errstate(divide="ignore", invalid="ignore"):  # a ray along the plane never meets it
        distance = ((origin - centre) @ normal) / (rays @ normal)
        offset = centre - origin + distance[..., None] * rays
        place = np.stack([offset @ across / (across @ across), offset @ down / (down @ down)],
                         axis=-1)
        inside = (distance > 0) & np.all((place >= 0) & (place <= 1), axis=-1)
    return np.where(inside, distance, np.inf), place


def _sample(image: np.ndarray, place: np.ndarray) -> np.ndarray:
    """The texture `image` sampled bilinearly at each place, an N × 2 array of fractions from 0 to
    1 along its columns and rows. Texel (row, column) has its centre at ((column + 0.5) / columns,
    (row + 0.5) / rows), and a place within half a texel of an edge takes the edge's values."""
    rows, columns = image.shape[:2]
    x = np.clip(place[:, 0] * columns - 0.5, 0, columns - 1)
    y = np.clip(place[:, 1] * rows - 0.5, 0, rows - 1)
    left = np.minimum(x.astype(np.intp), columns - 2)  # x is 0 or more, so this rounds down
    top = np.minimum(y.astype(np.intp), rows - 2)
    right_share = (x - left)[:, None]
    lower_share = (y - top)[:, None]
    upper = image[top, left] * (1 - right_share) + image[top, left + 1] * right_share
    lower = image[top + 1, left] * (1 - right_share) + image[top + 1, left + 1] * right_share
    return upper * (1 - lower_share) + lower * lower_share


# The Option_Check (Req 18.3, 18.6, C20)

def passed_options(config: Config, sizes: SmokeSizes) -> dict[str, frozenset[str]]:
    """Every option that the builders of commands.py pass each of the seven COLMAP commands, by
    command, in the order the step runs them.

    The runs are built for each Preset, both Camera_Models and both values of mask_used, with
    read_settings of that Preset's [cameras] with camera_model set, a FramesInput of one Pick of
    sizes.width × sizes.height, and Paths in HELP, which needn't exist, since nothing is made or
    run. An option is a word of a command line after the command that starts with --.
    """
    paths = Paths.of(Path(HELP, "work", _STEP), Path(HELP, "work", _FRAMES), Path(HELP, "out"))
    passed: dict[str, set[str]] = {command: set() for command in _COMMANDS}
    for preset in PRESETS:
        for camera_model in _CAMERA_MODELS:
            values = {**config.step_settings(_STEP, preset), "camera_model": camera_model}
            cfg = read_settings(StepSettings(values, section=_STEP, preset=preset), _STEP, preset)
            for mask_used in (False, True):
                frames = FramesInput(folder=paths.frames, pick_width=sizes.width,
                                     pick_height=sizes.height, mask_used=mask_used,
                                     picks=(Pick(number=0, file=pick_name(0)),))
                for run in _runs(cfg, frames, paths):
                    passed[run.command].update(word for word in run.args[2:]
                                               if word.startswith("--"))
    return {command: frozenset(options) for command, options in passed.items()}


def listed_options(help_text: str) -> frozenset[str]:
    """The options that a `colmap <command> -h` text lists: each line's option, two spaces in,
    after an optional short form like "-h [ " (F16)."""
    return frozenset(_HELP_OPTION.findall(help_text))


def check_options(config: Config, sizes: SmokeSizes, folder: Path) -> None:
    """The Option_Check: `colmap <command> -h` for each of the seven commands, in the order the
    step runs them, each within sizes.help_timeout_seconds, checking that it lists every option
    that passed_options gives it (Req 18.3, 18.6, C20).

    Each runs through run_colmap, in `folder`, with its Command_Log at <folder>/<command>.log and
    TMPDIR at <folder>/tmp, both made first. The Cap_Deadline doesn't cap these runs, because
    they aren't the step's. A run that times out or exits with a code other than 0, or a command
    that doesn't list an option, fails with a PipelineError that names the command.
    """
    passed = passed_options(config, sizes)
    tmp = folder / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    env = colmap_env(tmp, os.environ)
    for command in _COMMANDS:
        run = ColmapRun(name=f"{command}_help", command=command, args=(PROGRAM, command, "-h"),
                        timeout_seconds=sizes.help_timeout_seconds,
                        timeout_key="check_timeout_seconds", log=folder / f"{command}.log",
                        cwd=folder, env=env)
        result = run_colmap(run)
        if result.exit_code is None:
            raise PipelineError(
                f"`{PROGRAM} {command} -h` was still running after "
                f"{sizes.help_timeout_seconds:g} s, the check_timeout_seconds in [cloud.smoke], "
                f"so the cameras check stopped it ({_printed(result.last_line)}). Its log is "
                f"{run.log}. Check the image build's COLMAP step, then rebuild the image.")
        if result.exit_code != 0:
            raise PipelineError(
                f"`{PROGRAM} {command} -h` stopped with {exit_text(result.exit_code)} "
                f"({_printed(result.last_line)}), so the cameras check can't see which options "
                f"this COLMAP takes. Its log is {run.log}. Check the image build's COLMAP step, "
                "then rebuild the image.")
        text = run.log.read_text(encoding="utf-8", errors="replace")
        missing = sorted(passed[command] - listed_options(text))
        if missing:
            raise PipelineError(
                f"`{PROGRAM} {command} -h` doesn't list {_options(missing)}, which the cameras "
                f"step passes {command}, so this COLMAP would refuse the step's command line. "
                "Check the image build's COLMAP step, which builds COLMAP 4.2.1, then rebuild "
                "the image.")


def _runs(cfg: CamerasSettings, frames: FramesInput, paths: Paths) -> tuple[ColmapRun, ...]:
    """The eight runs the step's builders give for `cfg`, `frames` and `paths`, in the step's
    order: the Feature_Extraction, both Match_Runs, the Calibration, both mappers and both
    undistorters. The environment is empty, because only the command lines are read."""
    env: Mapping[str, str] = {}
    return (commands.feature_extractor(cfg, frames, paths, env),
            commands.sequential_matcher(cfg, paths, env, far=False),
            commands.sequential_matcher(cfg, paths, env, far=True),
            commands.view_graph_calibrator(cfg, paths, env),
            commands.global_mapper(cfg, paths, env, paths.database),
            commands.mapper(cfg, paths, env),
            commands.image_undistorter(cfg, paths, env, paths.global_models / "0"),
            commands.image_undistorter_standalone(cfg, paths, env))


def _printed(line: str) -> str:
    """A run's Last_Error_Line for the messages, or that it printed nothing."""
    return line or "it printed nothing"


def _options(names: list[str]) -> str:
    """Options for a sentence: "the option --a", or "the options --a, --b and --c"."""
    if len(names) == 1:
        return f"the option {names[0]}"
    return f"the options {', '.join(names[:-1])} and {names[-1]}"


# The Smoke_Solve (Req 18.4, 18.5, 18.7, C17 to C19)

def smoke_settings(config: Config, sizes: SmokeSizes) -> tuple[str, StepSettings]:
    """The Preset that [jobs].default_preset names, and that Preset's [cameras] settings with
    every Time_Limit key of TIMEOUT_KEYS at sizes.timeout_seconds, as a StepSettings that the
    step's read_settings checks (C18). So min_placed_frames stays the real config's."""
    preset = config.jobs().default_preset
    values = dict(config.step_settings(_STEP, preset))
    values.update({key: sizes.timeout_seconds for key in TIMEOUT_KEYS.values()})
    return preset, StepSettings(values, section=_STEP, preset=preset)


def run_smoke_solve(config: Config, sizes: SmokeSizes, scratch: Path, job_id: str) -> SmokeSolve:
    """The Smoke_Solve: the step's own work() on the Smoke_Picks, in <scratch>/job/ (Req 18.4,
    C17).

    It makes <scratch>/job/work/frames/, work/cameras/ and out/, writes the Smoke_Picks with
    [frames]'s jpeg_quality for the Preset, and builds a StepContext named cameras with that
    Preset, smoke_settings' settings, an in-memory Job whose Job_Record is never written, the
    two folders and empty timings, counts and warnings. work() reads the step's real TOOLS and
    the Cap_Deadline, as in any run, and a failure goes up with the step's own ExpectedFailure.
    Then it reads what the step wrote: the mapper and the Final_Model from solve.json, f from
    the Final_Model's cameras.bin and fx and fy from sparse/cameras.bin.
    """
    job_id = check_job_id(job_id)
    preset, settings = smoke_settings(config, sizes)
    made = scratch / JOB
    for part in (made / "work" / _FRAMES, made / "work" / _STEP, made / "out"):
        part.mkdir(parents=True)
    folder = made.resolve()  # a Job_Folder is resolved
    write_smoke_picks(folder / "work" / _FRAMES, sizes,
                      config.step_settings(_FRAMES, preset)["jpeg_quality"])
    seconds = sizes.frames * _PICK_SECONDS
    record = JobRecord(job_id=job_id, preset=preset, start_seconds=0.0, end_seconds=seconds,
                       source_name=_SOURCE_NAME, source_bytes=0, video_seconds=seconds,
                       video_file=_VIDEO_FILE)
    step = StepContext(name=_STEP, preset=preset, job=Job(id=job_id, folder=folder, record=record),
                       settings=settings, work_folder=folder / "work" / _STEP,
                       out_folder=folder / "out", timings={}, counts={}, warnings=[])
    cameras.work(step)
    return _found(step)


def _found(step: StepContext) -> SmokeSolve:
    """What the Smoke_Solve found, from its report counts and the files the step wrote (C19).

    f is params[0] of the camera in the cameras.bin of the Model that solve.json names,
    colmap/global/<final_model>/ or colmap/incremental/<final_model>/, and fx and fy are
    params[0] and params[1] of the Levelled_Model's PINHOLE camera in sparse/cameras.bin.
    """
    paths = Paths.of(step.work_folder, step.job.folder / "work" / _FRAMES, step.out_folder)
    record = read_json(paths.record)
    mapper = record["mapper"]
    models = paths.global_models if mapper == "global" else paths.incremental_models
    final = read_model(models / str(record["final_model"])).cameras[0]
    pinhole = read_model(paths.sparse).cameras[0]
    return SmokeSolve(folder=step.job.folder, picks=step.counts["picks"],
                      placed=step.counts["placed_frames"], mapper=mapper,
                      focal=float(final.params[0]), fx=float(pinhole.params[0]),
                      fy=float(pinhole.params[1]))


def check_smoke_solve(solve: SmokeSolve, sizes: SmokeSizes) -> None:
    """Refuse a Smoke_Solve that placed fewer than cameras_placed_share of the Smoke_Picks, worked
    out exactly as in C7, or whose f, fx or fy lies farther than cameras_focal_tolerance ×
    cameras_focal_pixels from cameras_focal_pixels (Req 18.5, 18.7, C19). A value exactly at the
    tolerance passes. Each refusal is a PipelineError that gives what the check found and what it
    needs, naming the key in [cloud.smoke]."""
    needed = placed_needed(sizes.frames, 0, sizes.placed_share)
    if solve.placed < needed:
        raise PipelineError(
            f"The cameras check's solve placed only {solve.placed} of the {sizes.frames} made-up "
            f"picks, but it must place at least {needed}, cameras_placed_share = "
            f"{sizes.placed_share:g} of them in [cloud.smoke]. Check the image build's COLMAP "
            "step, then run the smoke test again.")
    limit = sizes.focal_tolerance * sizes.focal_pixels
    for name, value in (("the solved focal length f", solve.focal),
                        ("the undistorted camera's fx", solve.fx),
                        ("the undistorted camera's fy", solve.fy)):
        if not abs(value - sizes.focal_pixels) <= limit:  # a NaN isn't within it either
            raise PipelineError(
                f"The cameras check's solve gave {name} = {value:.2f} px, but it must be within "
                f"{limit:.2f} px of the made-up picks' true focal length, cameras_focal_pixels = "
                f"{sizes.focal_pixels:g} px in [cloud.smoke]: cameras_focal_tolerance = "
                f"{sizes.focal_tolerance:g} of it. Check the image build's COLMAP step, then "
                "run the smoke test again.")


def run_cameras_smoke(config: Config, sizes: SmokeSizes, scratch: Path,
                      job_id: str) -> CamerasSmoke:
    """The Cameras_Smoke_Check: the Option_Check in <scratch>/help/, then the Smoke_Solve in
    <scratch>/job/ and its checks. Gives back the line for the Smoke_Test's log and what the
    Smoke_Solve found (Req 18.3 to 18.8). It writes only inside `scratch`."""
    check_options(config, sizes, scratch / HELP)
    solve = run_smoke_solve(config, sizes, scratch, job_id)
    check_smoke_solve(solve, sizes)
    line = (f"every option the cameras step passes is in colmap -h for its {len(_COMMANDS)} "
            f"commands, and the solve of {solve.picks} made-up picks placed {solve.placed} with "
            f"the {solve.mapper} mapper, at f = {solve.focal:.1f} px against "
            f"{sizes.focal_pixels:g} px")
    return CamerasSmoke(line=line, solve=solve)
