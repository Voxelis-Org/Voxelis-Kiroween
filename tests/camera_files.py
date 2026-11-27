"""The tests' own writer of COLMAP's binary model, the Sparse_Model that the Train_Step reads.

write_model(sparse, camera, images, points) packs cameras.bin, images.bin and points3D.bin into
the folder `sparse` with struct.pack("<...") in COLMAP's little-endian layout, keypoints, errors
and tracks included:

    cameras.bin    <Q count. Per camera <iiQQ (camera id, model id, width, height), then the
                   PINHOLE parameters <4d (fx, fy, cx, cy).
    images.bin     <Q count. Per image <idddddddi (image id, qw, qx, qy, qz, tx, ty, tz, camera
                   id), the name's bytes and a NUL, <Q keypoint count, then <ddq per keypoint
                   (x, y, point id).
    points3D.bin   <Q count. Per point <QdddBBBd (point id, x, y, z, r, g, b, error), <Q track
                   length, then <ii per track entry (image id, keypoint index).

It's written fresh from the training spec's layout table and imports nothing from
splatpipe.train.colmap, so Property 1's round trip checks the Model_Reader against a writer of its
own (training spec, Req 1.21, 14.1). pack_camera, pack_image and pack_point give one record's
bytes, so a test can work out where a record starts and ends.

For the refusal tests, write_model can break one file at a time. `cut` maps a file name to the
number of bytes to drop from the file's end, `trailing_byte` names the files that get one zero
byte after their last record, and `counts` maps a file name to the count to write in place of the
real one. cameras.bin's count is 1 unless `counts` changes it.

With no camera, images or points given, it writes the default scene: one 8 x 6 PINHOLE camera,
3 images from cameras on an arc of the unit circle in the x-z plane, each looking at the origin,
and 12 points on a small grid around the origin, at least 0.85 in front of every camera. Each image
holds a keypoint at the projection of every point, and each point a track entry for every image.
The records are frozen dataclasses, so a test makes a variant with dataclasses.replace.

write_camera_solve_output(job_folder, camera, images, points) writes the whole Camera_Solve_Output
into job_folder/work/cameras, laid out as the training spec's "The Camera_Solve_Output" gives it:
the model in sparse/ with write_model, each image as images/<name>, and each mask as
masks/<stem>.png, where <stem> is the image's name without ".jpg". An image is a JPEG of one solid
colour, IMAGE_BGR, at the camera's size. A mask is a grey PNG at its image's size, holding only 0
(MASKED, left out) and 255 (KEPT), and it leaves out the image's left half, its first width // 2
columns. Both are encoded with cv2.imencode and written with Path.write_bytes, and cv2 never
touches video. With no scene given, it writes the default scene above, 3 images of 8 x 6 pixels
and 12 points, and only the first image gets a mask. images/ is always made, and masks/ only when
a file goes in it. Its keyword arguments, each keyed by an image's name, like "000000.jpg", break
one thing at a time for the refusal tests (training spec, Req 1.12 to 1.17):

    masked         the images that get a mask, the first image if not given. An image named in
                   mask_sizes or fully_masked gets one too.
    left_out       images that images.bin names but images/ doesn't hold.
    junk           images whose file holds JUNK, bytes that aren't a JPEG.
    image_sizes    an image's (width, height) in place of the camera's.
    mask_sizes     a mask's (width, height) in place of its image's.
    fully_masked   images whose mask masks every pixel.
    extra_images   file names to add to images/ that images.bin doesn't name, each holding JUNK.
    extra_masks    the same for masks/.

The plane scene is the Depth_Step's (depth-priors spec, Req 18.1): PLANE_CAMERA, PLANE_IMAGES and
PLANE_POINTS, records that write_model and write_camera_solve_output take like the default
scene's, so its Frames are JPEGs of one colour too. Four PINHOLE cameras of 16 x 12 pixels sit
below the plane z = PLANE_Z, each in a place of its own, and each looks at the plane at an angle,
so its points' depths differ across its Frame and no Frame is flat. 30 points lie on the plane,
ids 1 to 30, each seen by every camera and no two on one pixel of any image. Each image's
keypoints name its 30 points, with PLANE_STRAYS mixed in: two keypoints of id -1 and one of
PLANE_NO_POINT, the id of no point. The camera's longer edge is below [depth]'s long_edge, so the
Prior_Size is 16 x 12, the camera's own size.

plane_moge_output(model, image, width, height, metres_per_unit, no_geometry) gives MoGe's exact
output for one image of the plane scene, read back with read_model, at the Prior_Size: at each
pixel an Observed_Point lands on, that point's camera-space z × metres_per_unit, and elsewhere the
plane's depth at the pixel's centre × metres_per_unit; the plane's normal facing the camera; and a
mask that's true except at the pixels in `no_geometry`. So every Frame fits with a scale of
metres_per_unit and an offset of 0, up to float32's rounding of the depth, and its
Metres_Per_Unit is the one the test chose. Its projection and its quaternion turn are this
module's own, so it shares no code with the Depth_Step, and it imports nothing from
splatpipe.train.colmap. It imports splatpipe.depth.gpu_api's MoGeOutput when it's called, so the
other tests that use this module never import splatpipe.depth.

The textured plane is the Walls_Step's scene (walls spec, Req 22.2): a wall at z = TEXTURED_Z,
seen by TEXTURED_FRAMES PINHOLE cameras of 128 x 96 pixels on a slow walk past it, each turned
to keep the wall's middle in view, with a faint grey-level noise texture painted on it. Its
helpers, given a frame's index and a size (width, height), work out on the CPU, by meeting each
pixel's ray with the wall, its true depth, normal and Covered mask, its BGR frame, and, for the
scoring, its ScoreSource and its Candidate. The section above textured_camera gives its numbers.
"""

from __future__ import annotations

import math
import struct
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

# Named in plane_moge_output's type hint only. The function imports it when it's called, so only
# the tests that call it import splatpipe.depth.
if TYPE_CHECKING:
    from splatpipe.depth.gpu_api import MoGeOutput
    from splatpipe.train.colmap import Camera
    from splatpipe.walls.scoring import ScoreSource
    from splatpipe.walls.sources import Candidate

CAMERAS_BIN = "cameras.bin"
IMAGES_BIN = "images.bin"
POINTS_BIN = "points3D.bin"
MODEL_FILES = (CAMERAS_BIN, IMAGES_BIN, POINTS_BIN)

PINHOLE_MODEL_ID = 1   # COLMAP's id for the PINHOLE model, whose parameters are fx, fy, cx, cy


@dataclass(frozen=True)
class CameraRecord:
    """One record of cameras.bin. fx, fy, cx and cy are packed whatever the model id says."""

    camera_id: int     # <i
    width: int         # <Q
    height: int        # <Q
    fx: float
    fy: float
    cx: float
    cy: float
    model_id: int = PINHOLE_MODEL_ID   # <i


@dataclass(frozen=True)
class ImageRecord:
    """One record of images.bin: a world-to-camera pose, the image's name and its keypoints."""

    image_id: int                                   # <i
    quaternion: tuple[float, float, float, float]   # qw, qx, qy, qz
    translation: tuple[float, float, float]         # tx, ty, tz
    camera_id: int                                  # <i
    name: str | bytes          # a str is written as UTF-8, bytes as they are (a name that isn't UTF-8)
    keypoints: tuple[tuple[float, float, int], ...] = ()   # x, y and point id (<q, -1 for none)


@dataclass(frozen=True)
class PointRecord:
    """One record of points3D.bin: a position, an RGB colour, an error and a track."""

    point_id: int                            # <Q
    position: tuple[float, float, float]     # x, y, z
    colour: tuple[int, int, int]             # r, g, b, each from 0 to 255
    error: float = 0.0
    track: tuple[tuple[int, int], ...] = ()  # image id and keypoint index, each <i


def pack_camera(camera: CameraRecord) -> bytes:
    """The 56 bytes of `camera`'s record in cameras.bin."""
    return struct.pack("<iiQQ4d", camera.camera_id, camera.model_id, camera.width, camera.height,
                       camera.fx, camera.fy, camera.cx, camera.cy)


def pack_image(image: ImageRecord) -> bytes:
    """`image`'s record in images.bin: 64 bytes, the name and its NUL, then the keypoints."""
    name = image.name.encode("utf-8") if isinstance(image.name, str) else image.name
    head = struct.pack("<idddddddi", image.image_id, *image.quaternion, *image.translation,
                       image.camera_id)
    keypoints = b"".join(struct.pack("<ddq", x, y, point_id) for x, y, point_id in image.keypoints)
    return head + name + b"\x00" + struct.pack("<Q", len(image.keypoints)) + keypoints


def pack_point(point: PointRecord) -> bytes:
    """`point`'s record in points3D.bin: 43 bytes, then the track."""
    head = struct.pack("<QdddBBBd", point.point_id, *point.position, *point.colour, point.error)
    track = b"".join(struct.pack("<ii", image_id, index) for image_id, index in point.track)
    return head + struct.pack("<Q", len(point.track)) + track


# The default scene's camera: 8 x 6 pixels, with fx as wide as the image, a view of about 53°.
DEFAULT_CAMERA = CameraRecord(camera_id=1, width=8, height=6, fx=8.0, fy=8.0, cx=4.0, cy=3.0)

# The default cameras' angles on the arc, in radians. The camera at angle a sits at
# (sin a, 0, -cos a) and looks at the origin: its world-to-camera rotation turns a about y.
ARC_ANGLES = (-0.25, 0.0, 0.25)

# The default scene's 12 points, a 3 x 2 x 2 grid around the origin.
GRID = tuple((x, y, z) for x in (-0.2, 0.0, 0.2) for y in (-0.1, 0.1) for z in (-0.1, 0.1))


def _project(angle: float, point: tuple[float, float, float]) -> tuple[float, float]:
    """Where `point` lands in the image of the default camera at `angle`, in pixels."""
    x, y, z = point
    camera_x = math.cos(angle) * x + math.sin(angle) * z
    camera_z = -math.sin(angle) * x + math.cos(angle) * z + 1.0
    return (DEFAULT_CAMERA.fx * camera_x / camera_z + DEFAULT_CAMERA.cx,
            DEFAULT_CAMERA.fy * y / camera_z + DEFAULT_CAMERA.cy)


def _arc_image(index: int, angle: float) -> ImageRecord:
    """Image `index` of the default scene, named like the Frames_Step's picks, from 000000.jpg."""
    keypoints = tuple((*_project(angle, point), number + 1) for number, point in enumerate(GRID))
    return ImageRecord(image_id=index + 1,
                       quaternion=(math.cos(angle / 2), 0.0, math.sin(angle / 2), 0.0),
                       translation=(0.0, 0.0, 1.0), camera_id=DEFAULT_CAMERA.camera_id,
                       name=f"{index:06d}.jpg", keypoints=keypoints)


DEFAULT_IMAGES = tuple(_arc_image(index, angle) for index, angle in enumerate(ARC_ANGLES))

# Point k is keypoint k of every image, so its track names every image.
DEFAULT_POINTS = tuple(
    PointRecord(point_id=number + 1, position=point, colour=(20 * number, 255 - 20 * number, 128),
                error=0.5, track=tuple((image.image_id, number) for image in DEFAULT_IMAGES))
    for number, point in enumerate(GRID))


def write_model(sparse: Path, camera: CameraRecord = DEFAULT_CAMERA,
                images: Sequence[ImageRecord] = DEFAULT_IMAGES,
                points: Sequence[PointRecord] = DEFAULT_POINTS, *,
                cut: Mapping[str, int] | None = None, trailing_byte: Collection[str] = (),
                counts: Mapping[str, int] | None = None) -> None:
    """Write cameras.bin, images.bin and points3D.bin into `sparse`, making the folder if need be.

    `cut` drops that many bytes from a file's end, `trailing_byte` adds one zero byte to the files
    it names, and `counts` writes its count in place of a file's real one.
    """
    cut = cut or {}
    counts = counts or {}
    records = {CAMERAS_BIN: [pack_camera(camera)],
               IMAGES_BIN: [pack_image(image) for image in images],
               POINTS_BIN: [pack_point(point) for point in points]}
    sparse.mkdir(parents=True, exist_ok=True)
    for name, packed in records.items():
        data = struct.pack("<Q", counts.get(name, len(packed))) + b"".join(packed)
        drop = cut.get(name, 0)
        data = data[:len(data) - drop] + (b"\x00" if name in trailing_byte else b"")
        (sparse / name).write_bytes(data)


# The Camera_Solve_Output's folder in a Job_Folder and the three folders in it. They're written out
# here rather than imported from splatpipe.train, so the tests check the step against the contract.
CAMERA_SOLVE = Path("work", "cameras")
IMAGES_FOLDER, MASKS_FOLDER, SPARSE_FOLDER = "images", "masks", "sparse"

# Every image's one colour, in OpenCV's BGR order. Its blue and red differ, so RGB order shows.
# JPEG is lossy, so a decoded pixel is close to this colour rather than equal to it.
IMAGE_BGR = (40, 120, 220)

# A mask's two values: black leaves a pixel out of the loss and white keeps it.
MASKED, KEPT = 0, 255

# What a junk image and every extra file hold.
JUNK = b"not a JPEG or a PNG"


def write_camera_solve_output(
        job_folder: Path, camera: CameraRecord = DEFAULT_CAMERA,
        images: Sequence[ImageRecord] = DEFAULT_IMAGES,
        points: Sequence[PointRecord] = DEFAULT_POINTS, *,
        masked: Collection[str] | None = None, left_out: Collection[str] = (),
        junk: Collection[str] = (), image_sizes: Mapping[str, tuple[int, int]] | None = None,
        mask_sizes: Mapping[str, tuple[int, int]] | None = None,
        fully_masked: Collection[str] = (), extra_images: Collection[str] = (),
        extra_masks: Collection[str] = ()) -> Path:
    """Write the Camera_Solve_Output into `job_folder`/work/cameras and give back that folder.

    The keyword arguments are the module docstring's. Sizes are (width, height) in pixels.
    """
    cameras = job_folder / CAMERA_SOLVE
    write_model(cameras / SPARSE_FOLDER, camera, images, points)
    image_folder, mask_folder = cameras / IMAGES_FOLDER, cameras / MASKS_FOLDER
    image_folder.mkdir(exist_ok=True)
    image_sizes, mask_sizes = image_sizes or {}, mask_sizes or {}
    names = [image.name for image in images]
    with_mask = set(names[:1] if masked is None else masked) | set(mask_sizes) | set(fully_masked)
    for name in names:
        width, height = image_sizes.get(name, (camera.width, camera.height))
        if name not in left_out:
            data = JUNK if name in junk else _encoded(".jpg", _solid_image(width, height))
            _write(image_folder / name, data)
        if name in with_mask:
            mask = _mask(*mask_sizes.get(name, (width, height)), every_pixel=name in fully_masked)
            _write(mask_folder / (name.removesuffix(".jpg") + ".png"), _encoded(".png", mask))
    for folder, extras in ((image_folder, extra_images), (mask_folder, extra_masks)):
        for name in extras:
            _write(folder / name, JUNK)
    return cameras


def _solid_image(width: int, height: int) -> np.ndarray:
    """A BGR image of `width` x `height` pixels, every one IMAGE_BGR."""
    return np.full((height, width, 3), IMAGE_BGR, dtype=np.uint8)


def _mask(width: int, height: int, *, every_pixel: bool) -> np.ndarray:
    """A grey mask that leaves out its left half, or every pixel."""
    mask = np.full((height, width), KEPT, dtype=np.uint8)
    mask[:, :width if every_pixel else width // 2] = MASKED
    return mask


def _encoded(extension: str, pixels: np.ndarray) -> bytes:
    """`pixels` encoded by cv2.imencode as a ".jpg" or ".png" file's bytes."""
    ok, data = cv2.imencode(extension, pixels)
    if not ok:
        raise ValueError(f"OpenCV couldn't encode a {extension} of shape {pixels.shape}")
    return data.tobytes()


def _write(path: Path, data: bytes) -> None:
    """Write `data` to `path`, making its folder if need be."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


# The plane scene, the Depth_Step's scene, in which every Frame fits exactly (depth-priors spec,
# Req 18.1). Four PINHOLE cameras of 16 x 12 pixels sit below the plane z = PLANE_Z, each in a
# place of its own, and each looks at the plane at an angle of about 20 degrees, so its points'
# depths differ across its Frame and no Frame is flat. PLANE_GRID's 30 points lie on the plane,
# each seen by every camera, and no two land on one pixel of any image. The camera's longer edge
# is below [depth]'s long_edge of 512, so the Prior_Size is the camera's own 16 x 12.
PLANE_Z = 3.0
PLANE_CAMERA = CameraRecord(camera_id=1, width=16, height=12, fx=16.0, fy=16.0, cx=8.0, cy=6.0)

# Each camera's centre and the point on the plane its optical axis goes through, in world units.
PLANE_VIEWS = (
    ((-0.9, -0.7, 0.0), (0.1, 0.05, PLANE_Z)),
    ((1.0, -0.5, 0.3), (-0.05, 0.1, PLANE_Z)),
    ((0.8, 0.8, -0.2), (0.0, -0.1, PLANE_Z)),
    ((-0.7, 0.9, 0.4), (-0.1, 0.0, PLANE_Z)),
)

# The 30 points: a 6 x 5 grid 0.36 apart around the plane's origin, row by row, each nudged by up
# to 0.02 in x and in y. With these nudges, every point lands on a pixel of its own in every image,
# at least a pixel inside the Frame and at least 0.019 pixels from a pixel's edge, so no rounding
# can move it to the next pixel.
PLANE_GRID = tuple((0.36 * (i - 2.5) + 0.01 * ((2 * i + j) % 5 - 2),
                    0.36 * (j - 2) + 0.01 * ((i + 2 * j + 3) % 5 - 2), PLANE_Z)
                   for j in range(5) for i in range(6))

# Point k of PLANE_GRID has the id k + 1. PLANE_NO_POINT is the id of no point in the scene.
PLANE_NO_POINT = 1000

# The keypoints that name no Observed_Point, mixed in among each image's 30 others: each one's
# index among the image's keypoints and its point id, -1 being COLMAP's id for a keypoint with
# no point. They all sit at STRAY_XY, which no Observed_Point lands on in any image.
PLANE_STRAYS = ((0, -1), (11, PLANE_NO_POINT), (23, -1))
STRAY_XY = (0.5, 0.5)


def _rotate(quaternion: Sequence[float], vectors: np.ndarray) -> np.ndarray:
    """`vectors`, (N, 3), turned by the rotation of `quaternion` (w, x, y, z) once it's scaled to
    length 1: v + w·s + q × s with s = 2 q × v, where q is the quaternion's vector part.

    It's written here, rather than taken from splatpipe.train.scene, so the plane scene's
    oracle shares no code with the Depth_Step it checks.
    """
    unit = np.asarray(quaternion, dtype=np.float64) / np.linalg.norm(quaternion)
    vectors = np.asarray(vectors, dtype=np.float64)
    turn = 2.0 * np.cross(unit[1:], vectors)
    return vectors + unit[0] * turn + np.cross(unit[1:], turn)


def _inverse(quaternion: Sequence[float]) -> tuple[float, float, float, float]:
    """The quaternion of the opposite rotation: (w, -x, -y, -z)."""
    w, x, y, z = (float(part) for part in quaternion)
    return (w, -x, -y, -z)


def _looking_at(centre: Sequence[float],
                target: Sequence[float]) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """The world-to-camera quaternion (w, x, y, z) and translation of a camera at `centre` whose
    optical axis goes through `target`.

    The camera-to-world rotation turns the camera's z axis onto the direction d from `centre` to
    `target` the shortest way, about z × d, so the camera's x stays close to the world's.
    images.bin holds the opposite rotation, and the translation is −R × centre.
    """
    direction = np.subtract(target, centre) / np.linalg.norm(np.subtract(target, centre))
    axis = np.cross((0.0, 0.0, 1.0), direction)
    half = math.atan2(float(np.linalg.norm(axis)), float(direction[2])) / 2
    to_world = (math.cos(half), *(math.sin(half) * axis / np.linalg.norm(axis)))
    quaternion = _inverse(to_world)
    translation = -_rotate(quaternion, np.array([centre]))[0]
    return tuple(float(part) for part in quaternion), tuple(float(part) for part in translation)


def _project_points(camera: CameraRecord, quaternion: Sequence[float],
                    translation: Sequence[float],
                    points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The camera-space z of each of the (N, 3) world `points` and where each lands with
    `camera` at its full size, (z, u, v), in COLMAP's pixel convention: u = fx·x / z + cx and
    v = fy·y / z + cy, with the top-left pixel's centre at (0.5, 0.5)."""
    in_camera = _rotate(quaternion, points) + np.asarray(translation, dtype=np.float64)
    z = in_camera[:, 2]
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):  # a test's point at z = 0
        return (z, camera.fx * in_camera[:, 0] / z + camera.cx,
                camera.fy * in_camera[:, 1] / z + camera.cy)


def _plane_image(index: int, centre: Sequence[float], target: Sequence[float]) -> ImageRecord:
    """Image `index` of the plane scene, named like the Frames_Step's picks, from 000000.jpg: a
    keypoint at the projection of every point of PLANE_GRID, in order, with PLANE_STRAYS mixed
    in at their indices."""
    quaternion, translation = _looking_at(centre, target)
    _, u, v = _project_points(PLANE_CAMERA, quaternion, translation, np.array(PLANE_GRID))
    keypoints = [(float(x), float(y), number + 1) for number, (x, y) in enumerate(zip(u, v))]
    for place, point_id in PLANE_STRAYS:  # in index order, so each lands at its own index
        keypoints.insert(place, (*STRAY_XY, point_id))
    return ImageRecord(image_id=index + 1, quaternion=quaternion, translation=translation,
                       camera_id=PLANE_CAMERA.camera_id, name=f"{index:06d}.jpg",
                       keypoints=tuple(keypoints))


PLANE_IMAGES = tuple(_plane_image(index, centre, target)
                     for index, (centre, target) in enumerate(PLANE_VIEWS))


def _keypoint_index(image: ImageRecord, point_id: int) -> int:
    """The index of the keypoint of `image` that names `point_id`."""
    return [keypoint[2] for keypoint in image.keypoints].index(point_id)


# Point k is a keypoint of every image, so its track names every image.
PLANE_POINTS = tuple(
    PointRecord(point_id=number + 1, position=point, colour=(8 * number, 250 - 8 * number, 90),
                error=0.5,
                track=tuple((image.image_id, _keypoint_index(image, number + 1))
                            for image in PLANE_IMAGES))
    for number, point in enumerate(PLANE_GRID))


def plane_moge_output(model: Any, image: Any, width: int, height: int, metres_per_unit: float,
                      no_geometry: Collection[tuple[int, int]] = ()) -> MoGeOutput:
    """MoGe's exact output for the plane scene's `image`, a ModelImage of the SparseModel `model`
    that read_model gives, at the Prior_Size `width` x `height`, with MoGe's metres
    `metres_per_unit` times the camera solve's units.

    - depth, (h, w) float32: at each pixel an Observed_Point lands on, that point's camera-space
      z × metres_per_unit, and at every other pixel the depth of the plane z = PLANE_Z at the
      pixel's centre ((x + 0.5)·W / w, (y + 0.5)·H / h), × metres_per_unit. An Observed_Point
      lands on its Prior_Size pixel (min(⌊u·w / W⌋, w − 1), min(⌊v·h / H⌋, h − 1)) when it's in
      front of the camera and its projection with the Shared_Camera at its full size W x H is
      inside the Frame. When two land on one pixel, the first in the image's order keeps it.
    - normal, (h, w, 3) float32: the plane's unit normal in the camera frame, facing the camera.
    - mask, (h, w) bool: true except at the pixels (x, y), column then row, in `no_geometry`,
      where depth is +inf and normal 0, as MoGe's apply_mask leaves them.

    So each Fit_Point's 1 / z is exactly metres_per_unit × its MoGe_Value, up to float32's
    rounding, and every Frame fits with a scale of metres_per_unit and an offset of 0. The
    Observed_Points are found from model.point_ids, each id at its first row, and the projection
    is this module's own: nothing here comes from splatpipe.train.colmap or the Depth_Step.
    """
    from splatpipe.depth.gpu_api import MoGeOutput  # here, so only the depth tests import it

    quaternion = np.asarray(image.quaternion, dtype=np.float64)
    translation = np.asarray(image.translation, dtype=np.float64)
    depth = _plane_depth(model.camera, quaternion, translation, width, height)
    for (x, y), z in _landings(model, image, width, height).items():
        depth[y, x] = z
    depth = (depth * metres_per_unit).astype(np.float32)
    normal = np.broadcast_to(_facing_normal(quaternion, translation),
                             (height, width, 3)).astype(np.float32)
    mask = np.ones((height, width), dtype=bool)
    for x, y in no_geometry:
        mask[y, x] = False
    depth[~mask] = np.inf
    normal[~mask] = 0.0
    return MoGeOutput(depth=depth, normal=normal, mask=mask)


def _plane_depth(camera: Any, quaternion: np.ndarray, translation: np.ndarray,
                 width: int, height: int) -> np.ndarray:
    """The camera-space z, (h, w) float64, at which the ray through each Prior_Size pixel's
    centre meets the plane z = PLANE_Z. Raises ValueError when a ray doesn't meet it in front of
    the camera."""
    columns = (np.arange(width) + 0.5) * camera.width / width
    rows = (np.arange(height) + 0.5) * camera.height / height
    u, v = np.meshgrid(columns, rows)
    rays = np.stack([(u - camera.cx) / camera.fx, (v - camera.cy) / camera.fy,
                     np.ones_like(u)], axis=-1).reshape(-1, 3)  # camera-space z of 1
    centre = _rotate(_inverse(quaternion), -translation[np.newaxis])[0]
    rise = _rotate(_inverse(quaternion), rays)[:, 2]  # the world z each ray gains per unit of z
    with np.errstate(divide="ignore", invalid="ignore"):
        depth = (PLANE_Z - centre[2]) / rise
    if not np.all(np.isfinite(depth) & (depth > 0)):
        raise ValueError("A pixel's ray doesn't meet the plane in front of this camera.")
    return depth.reshape(height, width)


def _landings(model: Any, image: Any, width: int, height: int) -> dict[tuple[int, int], float]:
    """Each Prior_Size pixel (x, y) that an Observed_Point of `image` lands on, with that point's
    camera-space z. A point behind the camera or outside the Frame lands nowhere, and the first
    point on a pixel keeps it. An id that names no point of `model` is no observation."""
    rows: dict[int, int] = {}
    for row, point_id in enumerate(model.point_ids.tolist()):
        rows.setdefault(int(point_id), row)
    observed = [rows[int(point_id)] for point_id in image.observed_ids.tolist()
                if int(point_id) in rows]
    camera = model.camera
    z, u, v = _project_points(camera, image.quaternion, image.translation,
                              np.asarray(model.points, dtype=np.float64)[observed].reshape(-1, 3))
    landings: dict[tuple[int, int], float] = {}
    for depth, across, down in zip(z.tolist(), u.tolist(), v.tolist()):
        if depth > 0 and 0 <= across < camera.width and 0 <= down < camera.height:
            pixel = (min(math.floor(across * width / camera.width), width - 1),
                     min(math.floor(down * height / camera.height), height - 1))
            landings.setdefault(pixel, depth)
    return landings


def _facing_normal(quaternion: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """The plane's unit normal in the camera frame, (3,) float64, on the camera's side of it."""
    centre = _rotate(_inverse(quaternion), -translation[np.newaxis])[0]
    side = (0.0, 0.0, -1.0) if centre[2] < PLANE_Z else (0.0, 0.0, 1.0)
    return _rotate(quaternion, np.array([side]))[0]


# The textured plane, the Walls_Step's scene (walls spec, Req 22.2, D24, A1, A8).
#
# A wall, the plane z = TEXTURED_Z = 4, faces TEXTURED_FRAMES = 12 PINHOLE cameras of 128 x 96
# pixels with fx = fy = TEXTURED_FOCAL = 192, a view of about 37° x 28°. The cameras walk past it
# along x at y = 0 and z = 0, TEXTURED_STEP = 0.5 apart, from x = −2.75 to +2.75, one frame every
# TEXTURED_SECONDS = 0.5 s, frame i at Pick_Number i and Frame_Time 0.5·i. Each is turned about y
# to look at the wall's middle (0, 0, TEXTURED_Z), so the end cameras look about 34.5° to the
# side, and each sees the wall at a depth of about 3.9 to 5.3. Every camera's whole view lies on
# the textured part of the wall, so every pixel is a Covered_Pixel at every size.
#
# A 1% depth change (motion_depth_change) moves a point, as the median over a frame's pixels, by
# 0.16 to 0.24 pixels in a frame 1 pick away, 0.32 to 0.48 in one 2 picks away, and at least 0.49
# in one 3 or more picks away, against [walls]'s source_min_motion of 0.4: the wall's parallax
# between two frames 3 picks apart is above 40 of the 128 pixels. Because every camera looks at
# the wall's middle, the Overlap of any two frames is at least 0.58, against source_min_overlap's
# 0.3, the two end frames' 0.79. So under the real time (1 s), spacing (2 picks), Overlap and
# Image_Motion rules, choose_sources gives every frame, the first and the last included, 4 or 5
# Source_Frames at source_min_seconds, against min_sources' 4: the middle frames take two or
# three on each side, the end frames four or five on one side.
#
# The texture is grey noise at TEXTURE_LEVELS = 5 levels TEXTURE_LEVEL_STEP = 4 apart around
# mid-grey, 120 to 136, drawn once with the fixed seed TEXTURE_SEED at the nodes of a square grid
# TEXTURE_CELL = 0.07 apart in the wall's own (x, y), and sampled bilinearly, so every camera sees
# the same texture. A cell is about 3 pixels wide at the wall's middle. A frame's grey has a
# spread of about 3.6 levels, and its High_Pass at σ 3 is far above flat_variance (0.04) in every
# window, so all 425 Score_Windows of a 128 x 96 frame (window_size 31, window_step 4) count, in
# 12 Tiles of 32 pixels. The windowed NCC of the High_Pass at the true depth is about 0.98, about
# 0.6 to 0.9 at 1.01 × it, and about 0 to 0.4 at 1.03 × it. Each pixel is the mean of TEXTURE_SAMPLES ×
# TEXTURE_SAMPLES rays spread evenly over it, worked out in float64 and rounded to uint8 only at
# the end, with the same grey in B, G and R. The plain variant is the same wall in one flat grey,
# TEXTURE_GREY, so its High_Pass is 0 everywhere and every window is flat.
#
# For a 128 x 96 Shared_Camera the Score_Size is the camera's own size, because a camera below
# score_long_edge (640) isn't scaled up, and the Depth_Step's Prior_Size rule with long_edge 512
# keeps 128 x 96 too. Each helper takes the size it works at, so a test may use a smaller
# Prior_Size. A pixel (i, j) of a w x h size is the ray through (u, v) = ((i + 0.5)·W / w,
# (j + 0.5)·H / h) of the camera's W x H pixels, the Prior_Format's rule. The scene's oracle, its
# quaternion turn and its rays, is this module's own and shares no code with the Walls_Step.
TEXTURED_Z = 4.0
TEXTURED_FRAMES = 12
TEXTURED_STEP = 0.5           # world units between neighbouring cameras
TEXTURED_SECONDS = 0.5        # seconds between neighbouring frames
TEXTURED_FOCAL = 192.0        # pixels
TEXTURED_CAMERA = CameraRecord(camera_id=1, width=128, height=96, fx=TEXTURED_FOCAL,
                               fy=TEXTURED_FOCAL, cx=64.0, cy=48.0)

# The textured part of the wall, |x| ≤ TEXTURED_HALF_WIDTH and |y| ≤ TEXTURED_HALF_HEIGHT: wider
# than every camera's view of it.
TEXTURED_HALF_WIDTH = 6.0
TEXTURED_HALF_HEIGHT = 4.0

TEXTURE_SEED = 22
TEXTURE_GREY = 128.0
TEXTURE_LEVELS = 5            # the levels 120, 124, 128, 132 and 136
TEXTURE_LEVEL_STEP = 4.0
TEXTURE_CELL = 0.07          # world units, about 3 pixels at the wall's middle
TEXTURE_SAMPLES = 2           # rays per pixel across and down


def _texture_nodes() -> np.ndarray:
    """The texture's grey at each node of its grid, (rows, columns) float64, row j at
    y = −TEXTURED_HALF_HEIGHT + j·TEXTURE_CELL and column i at x = −TEXTURED_HALF_WIDTH +
    i·TEXTURE_CELL, from the fixed seed TEXTURE_SEED."""
    columns = math.ceil(2 * TEXTURED_HALF_WIDTH / TEXTURE_CELL) + 2
    rows = math.ceil(2 * TEXTURED_HALF_HEIGHT / TEXTURE_CELL) + 2
    level = np.random.default_rng(TEXTURE_SEED).integers(0, TEXTURE_LEVELS, size=(rows, columns))
    return TEXTURE_GREY + TEXTURE_LEVEL_STEP * (level - (TEXTURE_LEVELS - 1) / 2)


TEXTURE_NODES = _texture_nodes()


def _textured_view(index: int) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Frame `index`'s world-to-camera quaternion and translation: its camera at
    ((index − (TEXTURED_FRAMES − 1) / 2)·TEXTURED_STEP, 0, 0), looking at the wall's middle."""
    if not 0 <= index < TEXTURED_FRAMES:
        raise ValueError(f"The textured plane has frames 0 to {TEXTURED_FRAMES - 1}, not {index}")
    centre = ((index - (TEXTURED_FRAMES - 1) / 2) * TEXTURED_STEP, 0.0, 0.0)
    return _looking_at(centre, (0.0, 0.0, TEXTURED_Z))


def _textured_image(index: int) -> ImageRecord:
    """Image `index` of the textured plane, named like the Frames_Step's picks, with no
    keypoints."""
    quaternion, translation = _textured_view(index)
    return ImageRecord(image_id=index + 1, quaternion=quaternion, translation=translation,
                       camera_id=TEXTURED_CAMERA.camera_id, name=f"{index:06d}.jpg")


TEXTURED_IMAGES = tuple(_textured_image(index) for index in range(TEXTURED_FRAMES))


def textured_camera() -> Camera:
    """The textured plane's Shared_Camera, as the Model_Reader gives it."""
    from splatpipe.train.colmap import Camera

    record = TEXTURED_CAMERA
    return Camera(camera_id=record.camera_id, width=record.width, height=record.height,
                  fx=record.fx, fy=record.fy, cx=record.cx, cy=record.cy)


def textured_pose(index: int) -> np.ndarray:
    """Frame `index`'s world-to-camera Image_Pose, (4, 4) float64 [[R, t], [0, 0, 0, 1]], with R
    from this module's own quaternion turn."""
    image = TEXTURED_IMAGES[index]
    pose = np.eye(4)
    pose[:3, :3] = _rotate(image.quaternion, np.eye(3)).T   # row j of _rotate's output is R e_j
    pose[:3, 3] = image.translation
    return pose


def _wall_hits(index: int, u: np.ndarray,
               v: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Where the rays of frame `index` through the camera's full-size pixel positions (u, v) meet
    the wall, as (z, x, y, covered): the camera-space z, the wall's own x and y, and whether the
    ray meets the textured part in front of the camera."""
    image = TEXTURED_IMAGES[index]
    camera = TEXTURED_CAMERA
    shape = np.shape(u)
    rays = np.stack([((np.asarray(u, dtype=np.float64) - camera.cx) / camera.fx).ravel(),
                     ((np.asarray(v, dtype=np.float64) - camera.cy) / camera.fy).ravel(),
                     np.ones(int(np.prod(shape)))], axis=-1)  # camera-space z of 1
    turned = _inverse(image.quaternion)
    centre = _rotate(turned, -np.asarray(image.translation, dtype=np.float64)[np.newaxis])[0]
    direction = _rotate(turned, rays)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (TEXTURED_Z - centre[2]) / direction[:, 2]
    x = centre[0] + z * direction[:, 0]
    y = centre[1] + z * direction[:, 1]
    covered = (np.isfinite(z) & (z > 0) & (np.abs(x) <= TEXTURED_HALF_WIDTH)
               & (np.abs(y) <= TEXTURED_HALF_HEIGHT))
    return z.reshape(shape), x.reshape(shape), y.reshape(shape), covered.reshape(shape)


def _pixel_positions(size: tuple[int, int],
                     offset: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """The camera's full-size (u, v) of each pixel of a `size` (w, h) grid, at `offset` across and
    down inside the pixel, 0.5 being its centre, as two (h, w) arrays."""
    width, height = size
    u = (np.arange(width, dtype=np.float64) + offset) * TEXTURED_CAMERA.width / width
    v = (np.arange(height, dtype=np.float64) + offset) * TEXTURED_CAMERA.height / height
    return np.meshgrid(u, v)


def textured_covered(index: int, size: tuple[int, int]) -> np.ndarray:
    """Frame `index`'s Covered mask at `size` (w, h), (h, w) bool: where the ray through a pixel's
    centre meets the textured part of the wall. It is true everywhere in this scene."""
    return _wall_hits(index, *_pixel_positions(size))[3]


def textured_depth(index: int, size: tuple[int, int]) -> np.ndarray:
    """Frame `index`'s true depth at `size` (w, h), (h, w) float64: the camera-space z at which
    the ray through each pixel's centre meets the wall, NaN off the Covered mask."""
    z, _, _, covered = _wall_hits(index, *_pixel_positions(size))
    return np.where(covered, z, np.nan)


def textured_normal(index: int, size: tuple[int, int]) -> np.ndarray:
    """Frame `index`'s true normal at `size` (w, h), (h, w, 3) float64: the wall's unit normal in
    the camera frame, facing the camera, at every Covered_Pixel, and 0 elsewhere."""
    normal = _rotate(TEXTURED_IMAGES[index].quaternion, np.array([[0.0, 0.0, -1.0]]))[0]
    width, height = size
    covered = textured_covered(index, size)
    return np.where(covered[..., np.newaxis], np.broadcast_to(normal, (height, width, 3)), 0.0)


def _texture(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """The texture's grey at the wall's own (x, y), sampled bilinearly from TEXTURE_NODES."""
    rows, columns = TEXTURE_NODES.shape
    across = (np.asarray(x) + TEXTURED_HALF_WIDTH) / TEXTURE_CELL
    down = (np.asarray(y) + TEXTURED_HALF_HEIGHT) / TEXTURE_CELL
    column = np.clip(np.floor(across), 0, columns - 2).astype(np.int64)
    row = np.clip(np.floor(down), 0, rows - 2).astype(np.int64)
    fx, fy = across - column, down - row
    top = TEXTURE_NODES[row, column] * (1 - fx) + TEXTURE_NODES[row, column + 1] * fx
    bottom = TEXTURE_NODES[row + 1, column] * (1 - fx) + TEXTURE_NODES[row + 1, column + 1] * fx
    return top * (1 - fy) + bottom * fy


def textured_frame(index: int, size: tuple[int, int] | None = None, *,
                   plain: bool = False) -> np.ndarray:
    """Frame `index` at `size` (w, h), the camera's own size if not given, as an (h, w, 3) uint8
    BGR image with the same grey in each channel: the mean of the texture over TEXTURE_SAMPLES ×
    TEXTURE_SAMPLES rays spread evenly over each pixel, in float64, then rounded. A ray that
    misses the textured part reads TEXTURE_GREY. `plain` paints the whole wall TEXTURE_GREY."""
    size = size or (TEXTURED_CAMERA.width, TEXTURED_CAMERA.height)
    width, height = size
    grey = np.zeros((height, width))
    for row in range(TEXTURE_SAMPLES):
        for column in range(TEXTURE_SAMPLES):
            u, _ = _pixel_positions(size, (column + 0.5) / TEXTURE_SAMPLES)
            _, v = _pixel_positions(size, (row + 0.5) / TEXTURE_SAMPLES)
            _, x, y, covered = _wall_hits(index, u, v)
            painted = TEXTURE_GREY if plain else np.where(covered, _texture(x, y), TEXTURE_GREY)
            grey = grey + painted
    level = np.clip(np.rint(grey / TEXTURE_SAMPLES ** 2), 0, 255).astype(np.uint8)
    return np.repeat(level[..., np.newaxis], 3, axis=-1)


def textured_candidate(index: int) -> Candidate:
    """Frame `index` as source choice sees it: its name, its Pick_Number `index`, its Frame_Time
    index × TEXTURED_SECONDS and its Image_Pose."""
    from splatpipe.walls.sources import Candidate

    return Candidate(name=TEXTURED_IMAGES[index].name, pick_number=index,
                     frame_time=index * TEXTURED_SECONDS, world_to_camera=textured_pose(index))


def textured_score_source(index: int, half: str, *, prior_size: tuple[int, int],
                          high_pass_sigma: float, band_sigmas: Sequence[float],
                          plain: bool = False) -> ScoreSource:
    """Frame `index` as a Source_Frame of Source_Half `half`, at the Score_Size, the camera's own
    128 x 96, with `high_pass_sigma` and `band_sigmas` from [walls].

    Its High_Pass and Bands are images.py's, from the Luma of textured_frame (plain or not), and
    its Score_Mask keeps every pixel. Its Start_Depth, at `prior_size` (w, h), is the frame's true
    depth, NaN off its Covered mask: the occlusion test compares the warp's depth with the
    Source_Frame's Start_Depth (Req 9.6, A1, D24), and true Observed_Points bring a Source_Frame's
    Start_Depth to about the true depth through its Scene_Scale, whatever one factor its
    Prior_Depth is off by.
    """
    from splatpipe.walls.images import bands, high_pass, luma
    from splatpipe.walls.scoring import ScoreSource

    grey = luma(textured_frame(index, plain=plain))
    return ScoreSource(name=TEXTURED_IMAGES[index].name, world_to_camera=textured_pose(index),
                       high_pass=high_pass(grey, high_pass_sigma),
                       bands=np.stack(bands(grey, band_sigmas)),
                       score_mask=np.ones(grey.shape, dtype=bool),
                       start_depth=textured_depth(index, prior_size),
                       covered=textured_covered(index, prior_size), half=half)
