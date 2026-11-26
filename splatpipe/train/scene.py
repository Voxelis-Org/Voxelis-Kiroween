"""The scene the Trainer trains on, set up from the Sparse_Model with pure numpy functions.

It reads no file and imports no torch (Req 3.10): files.read_views reads the Training_Images and
the Mask_Files, and this module only works on the arrays and numbers it's given.

- training_size (Req 3.1): the camera's size with its longer edge scaled down to long_edge,
  each edge rounded half up in integers, or the camera's own size, so no image is scaled up.
- scaled_intrinsics (Req 3.3): the Shared_Camera's K at the Training_Size.
- world_to_camera (Req 3.5): an Image_Pose's world-to-camera matrix, with the quaternion scaled
  to length 1, and camera_centres gives each image's Camera_Centre, −Rᵀt.
- scene_extent (Req 3.6, 3.7): scene_extent_factor × the largest distance from a Camera_Centre
  to their mean, refusing cameras that all sit in one place.
- median_scene_depth (Req 3.8, 3.9): the Median_Scene_Depth, leaving out each camera with no
  Sparse_Point in front of it, and refusing when no camera has one.
- keep_mask (Req 3.4): a grey Mask_File scaled to the Training_Size, each training pixel taking
  the mask pixel under its centre, as True where it isn't a Masked_Pixel.
- build_scene (Req 1.19, 1.20, 5.10): one TrainingView per image of images.bin, in its order,
  with none held out and the model's poses kept as they are, then the Scene_Extent, the
  Median_Scene_Depth and the Near_Plane. With a Prior_Set (depth-priors spec, Req 10.8), each
  Prior_View's record holds its Prior and Brightness_Gradient, every other record an empty
  prior, None, and the Scene holds the taps from the Prior_Size to the Training_Size.

From splatpipe, this module imports only splatpipe.errors, prior_views.py, and colmap.py,
priors.py and settings.py for their types.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from splatpipe.errors import ExpectedFailure
from splatpipe.train.colmap import Camera, SparseModel
from splatpipe.train.prior_views import PriorTaps, ViewPrior, prior_taps, view_prior
from splatpipe.train.priors import Prior, PriorSet
from splatpipe.train.settings import TrainSettings


@dataclass(frozen=True)
class ImagePixels:
    """One Training_Image's pixels at the Training_Size, as files.read_views gives them."""

    pixels: np.ndarray  # (h, w, 3) uint8, RGB
    keep: np.ndarray | None  # (h, w) bool, True where it isn't a Masked_Pixel. None: no Mask_File


@dataclass(frozen=True)
class TrainingView:
    """One record per Training_Image, holding the image's inputs to the loss (Req 5.10)."""

    name: str  # its name in images.bin
    pixels: np.ndarray  # (h, w, 3) uint8, RGB
    keep: np.ndarray | None  # (h, w) bool, or None when the image has no Mask_File
    intrinsics: np.ndarray  # (3, 3) float64, K scaled to the Training_Size
    world_to_camera: np.ndarray  # (4, 4) float64
    prior: ViewPrior | None = None  # the view's Prior (Req 10.8), None for an empty prior


@dataclass(frozen=True)
class Scene:
    """The views the Trainer trains on, at the Training_Size, and the scene's three distances."""

    views: tuple[TrainingView, ...]  # in images.bin's order
    width: int  # the Training_Size
    height: int
    extent: float  # the Scene_Extent
    median_depth: float  # the Median_Scene_Depth
    near_plane: float  # the Near_Plane
    taps: PriorTaps | None = None  # Prior_Size to Training_Size, None with no Prior_Set


def training_size(width: int, height: int, long_edge: int) -> tuple[int, int]:
    """The Training_Size (Req 3.1) of a `width` × `height` camera, as (width, height).

    When the longer edge L is above `long_edge`, each edge v becomes v × long_edge / L rounded
    half up in integers, (2·v·long_edge + L) // (2·L), and at least 1 pixel, so the longer edge
    is exactly `long_edge`. Otherwise the camera's own size is kept, so no image is scaled up.
    """
    longer = max(width, height)
    if longer <= long_edge:
        return width, height
    return _scaled_edge(width, long_edge, longer), _scaled_edge(height, long_edge, longer)


def _scaled_edge(edge: int, new: int, old: int) -> int:
    """edge × new / old, rounded half up in integers, and at least 1 pixel."""
    return max(1, (2 * edge * new + old) // (2 * old))


def scaled_intrinsics(camera: Camera, width: int, height: int) -> np.ndarray:
    """The Shared_Camera's K at the Training_Size `width` × `height`, as (3, 3) float64 (Req 3.3).

    fx and cx are scaled by sx = width / the camera's width, and fy and cy by sy = height / the
    camera's height.
    """
    sx = width / camera.width
    sy = height / camera.height
    return np.array([[camera.fx * sx, 0.0, camera.cx * sx],
                     [0.0, camera.fy * sy, camera.cy * sy],
                     [0.0, 0.0, 1.0]])


def world_to_camera(quaternion: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """The (4, 4) float64 world-to-camera matrix [[R, t], [0, 0, 0, 1]] of an Image_Pose (Req 3.5).

    The quaternion (w, x, y, z) is divided by its largest absolute value, then scaled to length
    1, so a quaternion of tiny or huge values can't underflow or overflow on the way. The
    Model_Reader has refused a quaternion of four zeros. R is the rotation matrix of the unit
    quaternion, by the standard formula for (w, x, y, z), as COLMAP builds it.
    """
    largest = quaternion / np.max(np.abs(quaternion))
    w, x, y, z = largest / np.linalg.norm(largest)
    matrix = np.eye(4)
    matrix[:3, :3] = [[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                      [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                      [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]]
    matrix[:3, 3] = translation
    return matrix


def camera_centres(world_to_cameras: np.ndarray) -> np.ndarray:
    """The Camera_Centre of each (4, 4) world-to-camera matrix in `world_to_cameras`, (V, 4, 4),
    as (V, 3): −Rᵀt, whose part i is −Σⱼ R[j, i]·t[j] (Req 3.5)."""
    rotations = world_to_cameras[:, :3, :3]
    translations = world_to_cameras[:, :3, 3]
    return -np.einsum("vji,vj->vi", rotations, translations)


def scene_extent(centres: np.ndarray, factor: float) -> float:
    """The Scene_Extent (Req 3.6): `factor` × the largest distance from a Camera_Centre to the
    mean of the `centres`, (V, 3).

    The centres are first taken relative to the first one, which moves their mean with them and
    leaves every distance as it is. So centres that all sit in one place give offsets of exactly
    0 and a largest distance of exactly 0, which a mean taken in place can miss by a rounding:
    three centres at x = 0.1 have a mean of 0.10000000000000002. A largest distance of 0 is
    refused with the Req 3.7 message.
    """
    offsets = centres - centres[0]
    largest = float(np.max(np.linalg.norm(offsets - offsets.mean(axis=0), axis=1)))
    if largest == 0:
        raise ExpectedFailure("The camera didn't move: every image was taken from the same place, "
                              "so no splat can be trained. Film while you move through the place, "
                              "then make a new job from that video.")
    return factor * largest


def median_scene_depth(world_to_cameras: np.ndarray, points: np.ndarray) -> float:
    """The Median_Scene_Depth (Req 3.8) of the (P, 3) Sparse_Points seen by the (V, 4, 4) cameras.

    It goes one camera at a time, so no (V, P) array is made. Each point's camera-space z is
    (R p + t)_z, and a camera's depth is the median of the z above 0. A camera with no point at a
    z above 0 is left out, and the Median_Scene_Depth is the median of the cameras' depths. Both
    medians are numpy.median's: the middle value, or the mean of the two middle values. When no
    camera has a point in front of it, it's refused with the Req 3.9 message.
    """
    depths = []
    for matrix in world_to_cameras:
        z = points @ matrix[2, :3] + matrix[2, 3]
        in_front = z[z > 0]
        if in_front.size:
            depths.append(np.median(in_front))
    if not depths:
        raise ExpectedFailure("The camera solve put no point in front of any camera, so the "
                              "splats have nowhere to start. Run the camera solve step on this "
                              "job again.")
    return float(np.median(depths))


def keep_mask(grey: np.ndarray, width: int, height: int, threshold: int) -> np.ndarray:
    """The (height, width) bool keep mask of a grey (H, W) Mask_File at the Training_Size (Req 3.4).

    Training pixel (x, y) takes the mask pixel at column (2x + 1)·W // (2·width) and row
    (2y + 1)·H // (2·height), the one under the training pixel's centre, through integer index
    maps. cv2.INTER_NEAREST isn't used, because it takes floor(x·W / width), the pixel under the
    training pixel's top-left corner. A pixel whose value is below `threshold` is a Masked_Pixel,
    False, and every other pixel is True.
    """
    mask_height, mask_width = grey.shape
    rows = (2 * np.arange(height) + 1) * mask_height // (2 * height)
    columns = (2 * np.arange(width) + 1) * mask_width // (2 * width)
    return grey[np.ix_(rows, columns)] >= threshold


def build_scene(model: SparseModel, pixels: Sequence[ImagePixels], width: int, height: int,
                settings: TrainSettings, priors: PriorSet | None = None) -> Scene:
    """The Scene at the Training_Size `width` × `height`, from the Sparse_Model and `pixels`,
    each image's ImagePixels in images.bin's order, and the checked Prior_Set `priors`, or None.

    It gives one TrainingView per image, in images.bin's order, holding none out (Req 1.19, 5.10).
    Each view holds its own copy of the one scaled K and the world-to-camera matrix of the
    model's pose, kept as it is with no change to the world frame (Req 1.20). Each image named in
    the Prior_Set holds view_prior's Prior and Brightness_Gradient, and every other image None,
    an empty prior (depth-priors spec, Req 10.8). Then come the Scene_Extent (Req 3.6), the
    Median_Scene_Depth and the Near_Plane, near_plane_factor × the Median_Scene_Depth (Req 3.8),
    and with a Prior_Set the taps from its Prior_Size to the Training_Size, worked out once.
    """
    intrinsics = scaled_intrinsics(model.camera, width, height)
    named = {} if priors is None else priors.priors
    views = tuple(TrainingView(image.name, image_pixels.pixels, image_pixels.keep,
                               intrinsics.copy(),
                               world_to_camera(image.quaternion, image.translation),
                               _view_prior(image_pixels, named.get(image.name)))
                  for image, image_pixels in zip(model.images, pixels, strict=True))
    matrices = np.stack([view.world_to_camera for view in views])
    extent = scene_extent(camera_centres(matrices), settings.scene_extent_factor)
    median_depth = median_scene_depth(matrices, model.points)
    return Scene(views, width, height, extent, median_depth,
                 settings.near_plane_factor * median_depth, _taps(priors, width, height))


def _view_prior(image_pixels: ImagePixels, prior: Prior | None) -> ViewPrior | None:
    """The View_Record's prior of an image whose Prior is `prior`: None, an empty prior, for an
    image with no Prior (Req 10.8)."""
    return None if prior is None else view_prior(image_pixels.pixels, prior)


def _taps(priors: PriorSet | None, width: int, height: int) -> PriorTaps | None:
    """The taps from the Prior_Set's Prior_Size to the Training_Size `width` × `height`, or None
    with no Prior_Set."""
    if priors is None:
        return None
    return prior_taps(priors.index.prior_width, priors.index.prior_height, width, height)
