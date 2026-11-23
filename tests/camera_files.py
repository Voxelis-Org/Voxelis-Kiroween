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
"""

from __future__ import annotations

import math
import struct
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

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
