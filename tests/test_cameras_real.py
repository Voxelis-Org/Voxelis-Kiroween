"""The Cameras_Step with the real COLMAP, on tiny synthetic Picks that this file makes itself in
the temporary Jobs_Folder (cameras spec, Req 16.6, 16.13).

The scene is the inside corner of a room: three textured rectangles that meet at a corner, a
floor, a back wall and a side wall, each 4 m × 4 m, so the scene isn't planar. The world has x to
the right, y down and z into the room, in metres, so up is -y. The floor is the plane y = 0, from
x = -2 to 2 and z = 0 to 4. The back wall is the plane z = 4 and the side wall the plane x = 2,
both rising 4 m from the floor, so the three meet at the corner (2, 0, 4).

Each rectangle has its own texture of 1 cm texels, made from its own fixed numpy seed alone:
grey noise smoothed at three scales, Gaussians of 2, 6 and 18 texels with the same weight, in a
colour of the texture's own, then rectangles at random angles and discs, each of a random size,
place and colour, which give SIFT plenty of corners. A last blur of 1 texel softens the shapes'
edges, so the render doesn't alias them much where it shrinks the texture.

The camera is a pinhole camera with f = 260 px at 320 × 240 and no distortion, its principal
point at the middle of the Pick, in COLMAP's convention that the top left pixel's centre is at
(0.5, 0.5). Its 30 centres are evenly spread along a level line 1.2 m above the floor, from
(-1, -1.2, 0) to (1, -1.2, 0), so it moves 2 m sideways, along the back wall. Each camera turns
about the vertical only, with no pitch and no roll: its y axis is the world's down, and its z
axis, the way it looks, is level and points at the corner's vertical edge, so that edge stays in
the Pick's middle column. On the way it turns from 36.9° to 14.0° right of +z, and its distance
to that edge shrinks from 5.0 m to 4.1 m. The line lies off to the side of the corner so that
those distances all differ: every optical axis lies in the same level plane, and two cameras at
the same distance from where their axes cross would give view_graph_calibrator a pair that says
nothing about the focal length.

The render casts one ray per pixel with numpy, through the pixel's centre. Each ray hits the
nearest rectangle in front of the camera, and that rectangle's texture is sampled bilinearly
where the ray meets it.

write_picks writes the Picks into a Job's work/frames/, as the Frames_Step leaves it, under the
names in splatpipe/frames/files.py: images/NNNNNN.jpg for each Pick at the test config's
jpeg_quality, which is 95, grey/NNNNNN.jpg from grey_copy at the same quality, mask.png, and
frames.json last. Every Pick has a black corner painted in, its top left 32 × 24 pixels, and
mask.png, a grey PNG at the Pick_Size, is 0 there and 255 elsewhere, so COLMAP ignores those
pixels. frames.json holds every field of the Frames_Record in frames_record's order, with
mask_used true, the Picks spread evenly over the Job's Used_Part, which is kept whole, each its
own Score_Frame, and the made-up scores of cameras_job in tests/conftest.py, which the
Cameras_Step doesn't read. JPEG and PNG bytes come from cv2.imencode, which never touches video,
and frames.json from write_json, as the Frames_Step writes them. Nothing is written outside the
Job, which lies in tmp_path.

The self-check renders all 30 Picks, with no COLMAP. Each one fills the frame, with every ray
hitting a rectangle, sees all three rectangles and isn't uniform.

The integration test runs the Cameras_Step on the Picks with the real COLMAP, through
splatpipe.cameras.run with the step's real TOOLS, so the Step_Runner runs it and writes its real
report (Req 16.6). It's skipped, saying so, where colmap isn't on the PATH (Req 16.8). Its Job
comes from new_job, in the temporary Jobs_Folder, with the test config's Preset, fast. Its
settings are the real [cameras]: set_cameras puts min_placed_frames back at the real config's 20,
which the test config lowers to 1, and gives each Time_Limit, each key of the real [cameras]
that ends in _timeout_seconds, 600 s for a laptop's CPU, where the test config gives 60 s (C5).
So every other value is the real config's, use_gpu 0 included. The colmap commands the test runs
itself, model_analyzer and model_converter, run in a folder of tmp_path with the environment
that colmap_env gives the step's COLMAP runs, so TMPDIR is that folder and glog's log files stay
in tmp_path, and each gets a timeout of 60 s (Req 16.13). The test checks that:

- the report is done, with those settings, and at least 24 of the 30 Picks are placed;
- the Work_Folder holds exactly colmap/, images/, masks/, sparse/ and solve.json, with a
  Training_Image in images/ for each Placed_Frame (Req 11.1);
- sparse/ holds exactly the three Model_Files, which our Model_Reader reads as one PINHOLE camera
  that every image uses, an image for each Placed_Frame (Req 10.5, 10.6);
- colmap model_analyzer --path work/cameras/sparse exits with 0 and logs Cameras: 1, the placed
  count as Registered images: and the points3d count as Points:, and colmap model_converter
  --output_type BIN, from sparse/ into a folder the test makes first, writes a cameras.bin, an
  images.bin and a points3D.bin equal to ours byte for byte, because COLMAP keeps every Pose's
  bytes when it reads a model and writes it again (Req 10.2, F5, O6);
- masks/ holds one PNG per Placed_Frame at the camera's size, with only 0 and 255, and 0 over the
  painted corner where the Undistortion moved it, since it crops the Picks by a few pixels: at
  each pixel whose centre, taken through the Undistorted_Camera and then the Final_Model's
  RADIAL camera from the COLMAP_Folder, with its one f and k1 and k2 on the radius, lies inside
  the painted corner of a Pick (Req 8.4, 11.3, D19);
- the levelled cameras' mean up, each camera's Rᵀ(0, −1, 0), points along (0, −1, 0) within
  1e-9. The true path is level and the cameras have no pitch or roll, so their mean up is the
  true vertical, and their centres' heights spread by less than 2 % of the path's length, from
  the first Placed_Frame's centre to the last's. The Heading_Camera that solve.json names looks
  along +Z within 1e-9 once its view, Rᵀ(0, 0, 1), is flattened (Req 9.2, 9.4, 9.5);
- out/cameras.json holds the camera of sparse/, levelled true, and each Placed_Frame in
  Pick_Number order with its Pose from sparse/images.bin (Req 11.9);
- every file in work/frames/ keeps its hash (Req 1.7).
"""

from __future__ import annotations

import functools
import hashlib
import json
import os
import re
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

# The test config's [frames], whose jpeg_quality the Picks take, and the real config's [cameras],
# read by path, whose settings the integration test runs with.
from conftest import FRAMES, REAL_CAMERAS
from splatpipe import cameras
from splatpipe.cameras.colmap import colmap_env
from splatpipe.cameras.model import CAMERA_MODELS, FILES, Camera, Image, read_model
from splatpipe.config import PRESETS, load_config
from splatpipe.frames.files import GREY, IMAGES, MASK, RECORD, grey_copy, pick_name
from splatpipe.jobs import open_job
from splatpipe.jsonio import read_json, write_json

# The camera: a pinhole camera with no distortion, its principal point at the middle of the Pick.
WIDTH, HEIGHT = 320, 240  # the Pick_Size, in pixels
FOCAL = 260.0  # f, in pixels
CX, CY = WIDTH / 2, HEIGHT / 2  # the principal point, in COLMAP's pixel convention
PICKS = 30  # one Pick per camera, from the start of the path to its end

# The camera's path, in the world's metres, with x to the right, y down and z into the room.
ROOM_CORNER = (2.0, 0.0, 4.0)  # where the floor and the two walls meet
EYE_HEIGHT = 1.2  # the camera's height above the floor, the plane y = 0
PATH_START = (-1.0, -EYE_HEIGHT, 0.0)  # Pick 0's camera centre
PATH = (2.0, 0.0, 0.0)  # from Pick 0's centre to the last Pick's: 2 m sideways, level

# The textures.
TEXEL = 0.01  # each texel's side, in metres
NOISE_SIGMAS = (2.0, 6.0, 18.0)  # the noise's three scales, as Gaussian sigmas in texels
NOISE_CONTRAST = 0.18  # the noise's standard deviation, as a share of full brightness
SHAPES = 60  # rectangles drawn on each texture, and as many discs
RECTANGLE_SIDES = (6.0, 48.0)  # the range of each rectangle side, in texels
DISC_RADII = (3, 24)  # the range of each disc's radius, in texels, both ends included
EDGE_SIGMA = 1.0  # the last blur, a Gaussian sigma in texels, which softens the shapes' edges

# The black corner painted into every Pick: its top left PAINTED_WIDTH × PAINTED_HEIGHT pixels.
# mask.png is 0 there and 255 elsewhere.
PAINTED_WIDTH, PAINTED_HEIGHT = 32, 24

# The smallest standard deviation of a Pick's values, from 0 to 255, for the self-check to call
# it not uniform. The Picks' own are from about 52 to 54.
MIN_SPREAD = 20.0


@dataclass(frozen=True)
class Rectangle:
    """One rectangle of the scene: the corner its texture starts at and its two edges from there,
    in metres, and the seed of its texture. The texture's columns run along `across` and its rows
    along `down`, one texel every TEXEL metres."""

    name: str
    origin: tuple[float, float, float]
    across: tuple[float, float, float]
    down: tuple[float, float, float]
    seed: int


# The scene's three rectangles, which meet at ROOM_CORNER.
RECTANGLES = (
    Rectangle("floor", origin=(-2.0, 0.0, 0.0), across=(4.0, 0.0, 0.0),
              down=(0.0, 0.0, 4.0), seed=1),
    Rectangle("back wall", origin=(-2.0, -4.0, 4.0), across=(4.0, 0.0, 0.0),
              down=(0.0, 4.0, 0.0), seed=2),
    Rectangle("side wall", origin=(2.0, -4.0, 0.0), across=(0.0, 0.0, 4.0),
              down=(0.0, 4.0, 0.0), seed=3),
)


@functools.cache
def texture(rectangle: Rectangle) -> np.ndarray:
    """The rectangle's texture, as float32 BGR values from 0 to 255, made from its seed alone.

    Grey noise smoothed at each of the NOISE_SIGMAS, each band brought to the same standard
    deviation, then added up and tinted with a random colour. Then SHAPES rectangles and SHAPES
    discs, drawn in turn, and last a blur of EDGE_SIGMA texels. It's made once per test run.
    """
    rng = np.random.default_rng(rectangle.seed)
    rows = round(float(np.linalg.norm(rectangle.down)) / TEXEL)
    columns = round(float(np.linalg.norm(rectangle.across)) / TEXEL)
    noise = np.zeros((rows, columns), dtype=np.float32)
    for sigma in NOISE_SIGMAS:
        band = cv2.GaussianBlur(rng.standard_normal((rows, columns), dtype=np.float32), (0, 0),
                                sigma)
        noise += band / band.std()
    grey = np.clip(0.5 + NOISE_CONTRAST * noise / np.sqrt(len(NOISE_SIGMAS)), 0.0, 1.0)
    image = np.rint(255 * grey[..., None] * rng.uniform(0.6, 1.0, 3)).astype(np.uint8)
    for _ in range(SHAPES):
        _draw_rectangle(image, rng)
        _draw_disc(image, rng)
    return cv2.GaussianBlur(image.astype(np.float32), (0, 0), EDGE_SIGMA)


def _draw_rectangle(image: np.ndarray, rng: np.random.Generator) -> None:
    """Fill a rectangle with sides in RECTANGLE_SIDES, at a random angle, place and colour."""
    rows, columns = image.shape[:2]
    centre = (float(rng.uniform(0, columns)), float(rng.uniform(0, rows)))
    sides = (float(rng.uniform(*RECTANGLE_SIDES)), float(rng.uniform(*RECTANGLE_SIDES)))
    corners = cv2.boxPoints((centre, sides, float(rng.uniform(0, 180))))
    cv2.fillConvexPoly(image, np.rint(corners).astype(np.int32), _colour(rng), cv2.LINE_AA)


def _draw_disc(image: np.ndarray, rng: np.random.Generator) -> None:
    """Fill a disc with a radius in DISC_RADII, at a random place and colour."""
    rows, columns = image.shape[:2]
    centre = (int(rng.integers(0, columns)), int(rng.integers(0, rows)))
    radius = int(rng.integers(DISC_RADII[0], DISC_RADII[1], endpoint=True))
    cv2.circle(image, centre, radius, _colour(rng), -1, cv2.LINE_AA)


def _colour(rng: np.random.Generator) -> tuple[int, int, int]:
    """A random BGR colour."""
    blue, green, red = (int(value) for value in rng.integers(0, 256, 3))
    return blue, green, red


def camera(number: int) -> tuple[np.ndarray, np.ndarray]:
    """Pick `number`'s camera: its centre in the world, and its rotation R from the world to the
    camera, so a point X is at R (X - centre) in the camera, with x to the right, y down and z
    ahead, as in COLMAP.

    The centres are evenly spread along PATH, from PATH_START for Pick 0 to its end for the last
    Pick. The rows of R are the camera's axes in the world. Its y axis is the world's down, so it
    has no pitch and no roll, its z axis is level and points at ROOM_CORNER's vertical edge, and
    its x axis is y × z.
    """
    centre = np.array(PATH_START) + np.array(PATH) * number / (PICKS - 1)
    ahead = np.array(ROOM_CORNER) - centre
    ahead[1] = 0.0  # level, so the camera only turns about the vertical
    ahead /= np.linalg.norm(ahead)
    down = np.array([0.0, 1.0, 0.0])
    return centre, np.stack([np.cross(down, ahead), down, ahead])


def render(centre: np.ndarray, rotation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """What the camera at `centre`, with `rotation` from the world to the camera, sees: a BGR
    uint8 image of WIDTH × HEIGHT, and for each pixel the index in RECTANGLES of the rectangle its
    ray hits first, or -1 for a ray that hits none, whose pixel stays black."""
    rays = _rays(rotation)
    hits = [_hit(rectangle, centre, rays) for rectangle in RECTANGLES]
    distances = np.stack([distance for distance, _ in hits])
    seen = np.where(np.isfinite(distances.min(axis=0)), distances.argmin(axis=0), -1)
    image = np.zeros((HEIGHT, WIDTH, 3))
    for index, (rectangle, (_, place)) in enumerate(zip(RECTANGLES, hits)):
        mine = seen == index
        image[mine] = _sample(texture(rectangle), place[mine])
    return np.rint(image).astype(np.uint8), seen


def _rays(rotation: np.ndarray) -> np.ndarray:
    """Each pixel's ray direction in the world, through the pixel's centre, as an array of
    HEIGHT × WIDTH × 3. In the camera, the ray of the pixel centred at (u, v) is
    ((u - CX) / FOCAL, (v - CY) / FOCAL, 1)."""
    u, v = np.meshgrid(np.arange(WIDTH) + 0.5, np.arange(HEIGHT) + 0.5)
    in_camera = np.stack([(u - CX) / FOCAL, (v - CY) / FOCAL, np.ones_like(u)], axis=-1)
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


def write_picks(job_id: str) -> Path:
    """Write the scene's Picks into the work/frames/ of the Job `job_id`, as the Frames_Step
    leaves that folder, and give the folder.

    The Job is found in the test config's Jobs_Folder, which lies in tmp_path. Pick by Pick, its
    render with the black corner painted in goes to images/ as a JPEG at the test config's
    jpeg_quality, and its Grey_Copy, from grey_copy, to grey/ under the same name at the same
    quality. Then come mask.png and, last, frames.json.
    """
    job = open_job(job_id, load_config())
    frames = job.folder / "work" / "frames"
    for name in (IMAGES, GREY):
        (frames / name).mkdir(parents=True)
    quality = (cv2.IMWRITE_JPEG_QUALITY, FRAMES["jpeg_quality"])
    for number in range(PICKS):
        pick, _ = render(*camera(number))
        pick[:PAINTED_HEIGHT, :PAINTED_WIDTH] = 0
        _write_image(frames / IMAGES / pick_name(number), pick, *quality)
        _write_image(frames / GREY / pick_name(number), grey_copy(pick), *quality)
    mask = np.full((HEIGHT, WIDTH), 255, dtype=np.uint8)
    mask[:PAINTED_HEIGHT, :PAINTED_WIDTH] = 0
    _write_image(frames / MASK, mask)
    write_json(frames / RECORD, _frames_record(job.record.start_seconds, job.record.end_seconds,
                                               float(np.mean(mask == 0))))
    return frames


def _frames_record(start: float, end: float, coverage: float) -> dict[str, Any]:
    """The Picks' Frames_Record, with every field in frames_record's order.

    The Used_Part, from `start` to `end`, is kept whole, and the Picks are spread evenly over it,
    each its own Score_Frame. The Mask is used, with `coverage` its share of 0s. The scores are
    cameras_job's made-up ones, because the Cameras_Step doesn't read them.
    """
    return {
        "colour_transfer": "bt709", "tone_mapped": False,
        "still_start_seconds": 0.0, "still_end_seconds": 0.0,
        "kept_start_seconds": start, "kept_end_seconds": end,
        "target_picks": PICKS, "jolt_filter_skipped": False,
        "pick_width": WIDTH, "pick_height": HEIGHT,
        "mask_coverage": coverage, "mask_used": True, "mask_file": MASK,
        "picks": [{"number": number, "file": pick_name(number), "frame_number": number,
                   "seconds": start + (end - start) * number / PICKS, "sharpness": 100.0,
                   "motion": 0.0 if number == 0 else 2.0, "contrast": 40.0}
                  for number in range(PICKS)]}


def _write_image(path: Path, image: np.ndarray, *params: int) -> None:
    """Encode `image` with cv2.imencode in the format of `path`'s suffix, then write it."""
    ok, data = cv2.imencode(path.suffix, image, list(params))
    assert ok, f"cv2.imencode couldn't encode {path}"
    path.write_bytes(data.tobytes())


def test_every_pick_fills_the_frame_sees_all_three_rectangles_and_isnt_uniform() -> None:
    # The self-check of the scene, with no COLMAP. Every pixel's ray hits a rectangle, every Pick
    # sees the floor and both walls, so its view isn't planar, and its values spread widely.
    for number in range(PICKS):
        pick, seen = render(*camera(number))
        assert pick.shape == (HEIGHT, WIDTH, 3) and pick.dtype == np.uint8
        assert (seen >= 0).all(), f"some rays of Pick {number} miss every rectangle"
        missed = [rectangle.name for index, rectangle in enumerate(RECTANGLES)
                  if not (seen == index).any()]
        assert not missed, f"Pick {number} doesn't see the {', '.join(missed)}"
        assert pick.std() > MIN_SPREAD, f"Pick {number} is nearly uniform"


# The integration test with the real COLMAP (Req 16.6)

# The run's Time_Limits: each key of the real [cameras] that ends in _timeout_seconds, at 600 s
# for a laptop's CPU, where the test config gives 60 s (C5).
TIME_LIMITS = {key: 600 for key in REAL_CAMERAS if key.endswith("_timeout_seconds")}
MIN_PLACED = 24  # the Placed_Frames the run must give, of the PICKS
# The timeout of each colmap command the test runs itself, model_analyzer and model_converter,
# which each read a model of at most 30 images (Req 16.13).
CHECK_TIMEOUT = 60  # in seconds
UP = np.array([0.0, -1.0, 0.0])  # the World_Up of the levelled frame
AHEAD = np.array([0.0, 0.0, 1.0])  # where the Heading_Camera's flattened view points, +Z
TOLERANCE = 1e-9  # how close the levelled directions must come to UP and AHEAD
HEIGHT_SPREAD = 0.02  # how far the centres' heights may spread, as a share of the path's length
# The three counts that colmap model_analyzer logs and the test reads, each after glog's prefix,
# which ends in "] ", like "I20261123 04:19:19.624758 0x1705ab000 model.cc:1234] Points: 1234"
# (F16).
ANALYZER_COUNT = re.compile(r"\] (Cameras|Registered images|Points): ([0-9]+)$", re.MULTILINE)


def colmap_check(folder: Path, *args: str) -> str:
    """Run colmap with `args` and give what it logged, once it has exited with 0.

    It runs in `folder`, with stdin closed and the environment that colmap_env gives the step's
    COLMAP runs, so TMPDIR is `folder` and glog writes its log files there, with a timeout of
    CHECK_TIMEOUT. glog logs to stderr, so stderr goes into stdout and both are read.
    """
    result = subprocess.run(["colmap", *args], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, cwd=folder,
                            env=colmap_env(folder, os.environ), encoding="utf-8",
                            errors="replace", timeout=CHECK_TIMEOUT)
    assert result.returncode == 0, (args, result.returncode, result.stdout)
    return result.stdout


def hashes(folder: Path) -> dict[str, str | None]:
    """Every file and folder inside `folder`, by its path from `folder`, like images/000000.jpg:
    a file's SHA-256, and None for a folder."""
    return {path.relative_to(folder).as_posix():
            hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
            for path in sorted(folder.rglob("*"))}


def names_in(folder: Path) -> list[str]:
    """The names of the files and folders right inside `folder`, sorted."""
    return sorted(path.name for path in folder.iterdir())


def read_png(path: Path) -> np.ndarray:
    """The image in the PNG at `path`, as cv2.imdecode reads it with its channels and bit depth
    unchanged."""
    image = cv2.imdecode(np.frombuffer(path.read_bytes(), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    assert image is not None, f"cv2.imdecode couldn't read {path}"
    return image


def painted_corner(undistorted: Camera, distorted: Camera) -> np.ndarray:
    """Which pixels of a Training_Mask show the corner painted into the Picks: a bool array of
    the height × width of `undistorted`, the Undistorted_Camera, true at each pixel whose centre
    lies inside the corner in a Pick of `distorted`, the Final_Model's RADIAL camera.

    The pixel centred at (u, v), with the top left pixel's centre at (0.5, 0.5), looks along
    x = (u − cx) / fx and y = (v − cy) / fy through the PINHOLE camera. RADIAL's distortion, k1
    and k2 on the radius, takes that to x (1 + k1 r² + k2 r⁴) and y (1 + k1 r² + k2 r⁴), with
    r² = x² + y², and its one focal length f and its principal point then give the place in the
    Pick (F2, D19). The corner is the Pick's top PAINTED_HEIGHT rows of its left PAINTED_WIDTH
    columns. The Mask_Warp samples mask.png bilinearly at that place, so a pixel inside the
    corner takes at least a quarter of its value from the corner's 0s, and the threshold makes
    it 0. A pixel just outside may go either way.
    """
    assert CAMERA_MODELS[distorted.model_id][0] == "RADIAL"
    fx, fy, cx, cy = undistorted.params
    u, v = np.meshgrid(np.arange(undistorted.width) + 0.5, np.arange(undistorted.height) + 0.5)
    x, y = (u - cx) / fx, (v - cy) / fy
    focal, pick_cx, pick_cy, k1, k2 = distorted.params
    r2 = x * x + y * y
    radial = 1 + k1 * r2 + k2 * r2 * r2
    return ((focal * x * radial + pick_cx < PAINTED_WIDTH)
            & (focal * y * radial + pick_cy < PAINTED_HEIGHT))


def rotation_of(image: Image) -> np.ndarray:
    """The rotation R of the image's Pose, from its unit quaternion (w, x, y, z) in Hamilton's
    convention. Its rows are the camera's x, y and z axes in the world."""
    w, x, y, z = image.qvec
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def direction(vector: np.ndarray) -> np.ndarray:
    """`vector` divided by its length."""
    return vector / np.linalg.norm(vector)


def close(found: np.ndarray, expected: np.ndarray) -> bool:
    """Whether every number of `found` is within TOLERANCE of the same number of `expected`."""
    return bool(np.allclose(found, expected, rtol=0, atol=TOLERANCE))


@pytest.mark.skipif(shutil.which("colmap") is None, reason="COLMAP isn't installed")
def test_the_real_colmap_places_the_picks_and_reads_back_the_levelled_model_byte_for_byte(
        tmp_path: Path, new_job: Callable[..., str], set_cameras: Callable[..., None]) -> None:
    set_cameras(min_placed_frames=REAL_CAMERAS["min_placed_frames"], **TIME_LIMITS)
    job_id = new_job()
    folder = open_job(job_id, load_config()).folder  # the Job_Folder
    frames = write_picks(job_id)
    before = hashes(frames)
    report = cameras.run(job_id)  # with the step's real TOOLS, so the real colmap
    # The report is done, with every setting at the real config's value but the Time_Limits, and
    # with at least MIN_PLACED of the PICKS placed.
    assert report["outcome"] == "done", report["error"]
    shared = {key: value for key, value in REAL_CAMERAS.items() if key not in PRESETS}
    preset = REAL_CAMERAS[report["preset"]]
    assert report["settings"] == {**shared, **preset, **TIME_LIMITS}
    counts = report["counts"]
    assert counts["picks"] == PICKS
    assert counts["placed_frames"] >= MIN_PLACED, counts
    # The Work_Folder: exactly colmap/, images/, masks/, sparse/ and solve.json, since the Picks
    # use the Mask, with a Training_Image for each name in sparse/images.bin (Req 11.1).
    work = folder / "work" / "cameras"
    sparse = work / "sparse"
    assert names_in(work) == ["colmap", "images", "masks", "solve.json", "sparse"]
    # sparse/ holds exactly the three Model_Files, which our reader reads as one PINHOLE camera
    # that every image uses, an image for each Placed_Frame (Req 10.5, 10.6).
    assert names_in(sparse) == sorted(FILES)
    model = read_model(sparse)
    (pinhole,) = model.cameras
    assert CAMERA_MODELS[pinhole.model_id][0] == "PINHOLE"
    assert {image.camera_id for image in model.images} == {pinhole.camera_id}
    images = {image.name: image for image in model.images}
    assert len(images) == len(model.images) == counts["placed_frames"]
    assert names_in(work / "images") == sorted(images)
    # COLMAP reads sparse/ back. model_analyzer counts the one camera, the Placed_Frames and the
    # Final_Model's points (Req 16.6). model_converter, writing it again into a folder made for
    # it, gives our three files byte for byte, beside its own rigs.bin and frames.bin (Req 10.2,
    # F5, O6).
    checks = tmp_path / "colmap_checks"
    checks.mkdir()
    logged = colmap_check(checks, "model_analyzer", "--path", str(sparse))
    assert dict(ANALYZER_COUNT.findall(logged)) == {
        "Cameras": "1", "Registered images": str(counts["placed_frames"]),
        "Points": str(counts["points3d"])}, logged
    converted = tmp_path / "converted"
    converted.mkdir()
    colmap_check(checks, "model_converter", "--input_path", str(sparse), "--output_path",
                 str(converted), "--output_type", "BIN")
    for name in FILES:
        assert (converted / name).read_bytes() == (sparse / name).read_bytes(), name
    # masks/: one PNG per Placed_Frame, named after its stem, at the camera's size, with only 0
    # and 255, and 0 over the corner painted black into every Pick, where the Undistortion moved
    # it, as the Final_Model's camera in the COLMAP_Folder gives it (Req 8.4, 11.3).
    record = read_json(work / "solve.json")
    (final,) = read_model(work / "colmap" / record["mapper"] / str(record["final_model"])).cameras
    painted = painted_corner(pinhole, final)
    assert painted.any()
    masks = work / "masks"
    assert names_in(masks) == sorted(f"{Path(name).stem}.png" for name in images)
    for path in sorted(masks.iterdir()):
        mask = read_png(path)
        assert mask.dtype == np.uint8 and mask.shape == (pinhole.height, pinhole.width), path
        assert np.unique(mask).tolist() == [0, 255], path
        assert not mask[painted].any(), path
    # The levelled cameras. Their mean up points along the World_Up. The true path is level and
    # the cameras have no pitch or roll, so their mean up is the true vertical, and their
    # centres, −Rᵀt, keep to one height, up to HEIGHT_SPREAD of the path's length from the first
    # Placed_Frame's centre to the last's. The Heading_Camera that solve.json names has its view
    # along +Z once the part along the World_Up is taken away (Req 9.2, 9.4, 9.5).
    assert record["levelled"] is True, record["level_reason"]
    rotations = {name: rotation_of(image) for name, image in images.items()}
    mean_up = np.mean([-rotation[1] for rotation in rotations.values()], axis=0)
    assert close(direction(mean_up), UP), mean_up
    placed = [pick_name(number) for number in range(PICKS) if pick_name(number) in images]
    centres = [-rotations[name].T @ np.array(images[name].tvec) for name in placed]
    heights = [float(centre @ UP) for centre in centres]
    length = float(np.linalg.norm(centres[-1] - centres[0]))
    assert max(heights) - min(heights) < HEIGHT_SPREAD * length, (heights, length)
    view = rotations[record["heading_camera"]][2]
    assert close(direction(view - (view @ UP) * UP), AHEAD), view
    # out/cameras.json: the camera of sparse/, levelled, and each Placed_Frame in Pick_Number
    # order with its Pose from sparse/images.bin, and nothing else (Req 11.9). json.dumps
    # compares the kinds too, so 1.0 isn't 1.
    fx, fy, cx, cy = pinhole.params
    expected = {
        "camera": {"width": pinhole.width, "height": pinhole.height, "fx": fx, "fy": fy,
                   "cx": cx, "cy": cy},
        "levelled": True,
        "frames": [{"name": name, "quaternion_wxyz": list(images[name].qvec),
                    "translation": list(images[name].tvec)} for name in placed],
    }
    assert json.dumps(read_json(folder / "out" / "cameras.json")) == json.dumps(expected)
    # work/frames/ is as write_picks left it (Req 1.7).
    assert hashes(frames) == before
