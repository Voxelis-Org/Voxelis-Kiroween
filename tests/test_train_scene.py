"""Unit tests for the scene: splatpipe/train/scene.py.

Every input is built in the test: sizes, a camera, masks of a few pixels, world-to-camera
matrices and points written out by hand, and for build_scene the default scene of
tests/camera_files.py's write_model, read back with read_model (Req 14.1). Each oracle is worked
out here and never calls the function it checks: the mask pixel under each training pixel's
centre with Fractions, the rotation of a turn about y with cos and sin, each camera's depths from
the full product R p + t, and the default scene's depths from its cameras' angles.

Each refusal is an ExpectedFailure whose message says what its requirement says and holds no
traceback (Req 3.7, 3.9, 14.6).
"""

import math
import statistics
from collections.abc import Callable, Sequence
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from splatpipe.config import load_config
from splatpipe.errors import ExpectedFailure
from splatpipe.train.colmap import Camera, read_model
from splatpipe.train.scene import (ImagePixels, build_scene, camera_centres, keep_mask,
                                   median_scene_depth, scaled_intrinsics, scene_extent,
                                   training_size, world_to_camera)
from splatpipe.train.settings import read_settings

from camera_files import ARC_ANGLES, DEFAULT_CAMERA, DEFAULT_IMAGES, GRID, write_model

LONG_EDGE = 1600  # the real long_edge in [train]
THRESHOLD = 128  # the real mask_threshold in [train]
EXTENT_FACTOR = 1.1  # the real scene_extent_factor in [train]


def refused(call: Callable[[], object]) -> str:
    """The message of the ExpectedFailure that `call` raises, once it's checked that the message
    holds no traceback (Req 14.6)."""
    with pytest.raises(ExpectedFailure) as caught:
        call()
    message = str(caught.value)
    assert "Traceback" not in message
    return message


def pose(rotation: np.ndarray, translation: Sequence[float]) -> np.ndarray:
    """The (4, 4) world-to-camera matrix [[R, t], [0, 0, 0, 1]], built by hand."""
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = translation
    return matrix


def turn_about_y(angle: float) -> np.ndarray:
    """The rotation by `angle` about y, written out: [[cos, 0, sin], [0, 1, 0], [−sin, 0, cos]]."""
    cos, sin = math.cos(angle), math.sin(angle)
    return np.array([[cos, 0.0, sin], [0.0, 1.0, 0.0], [-sin, 0.0, cos]])


# Req 3.1: the Training_Size

@pytest.mark.parametrize(("camera", "size"), [
    pytest.param((3840, 2160), (1600, 900), id="3840 x 2160"),
    pytest.param((2001, 1125), (1600, 900), id="2001 x 1125"),
    pytest.param((2160, 3840), (900, 1600), id="2160 x 3840"),
    pytest.param((1600, 900), (1600, 900), id="1600 x 900 kept"),
    pytest.param((64, 48), (64, 48), id="64 x 48 kept"),
])
def test_training_size_scales_the_longer_edge_to_long_edge_and_never_scales_up(
        camera: tuple[int, int], size: tuple[int, int]) -> None:
    """The longer edge becomes long_edge and the other is rounded to the nearest whole pixel, so
    2001 × 1125 gives 1600 × 899.55, which is 900. A camera whose longer edge is at most long_edge
    keeps its own size (Req 3.1)."""
    got = training_size(*camera, LONG_EDGE)
    assert got == size
    assert all(type(edge) is int for edge in got)


# Req 3.3: the scaled camera

def test_scaled_intrinsics_at_half_size_halves_fx_fy_cx_and_cy() -> None:
    """fx and cx scale by the training width over the camera's, and fy and cy by the heights',
    here 1/2 each (Req 3.3). The four values differ, so a swap shows, and halving is exact."""
    camera = Camera(camera_id=1, width=1920, height=1080, fx=1500.0, fy=1502.0, cx=961.0,
                    cy=539.0)
    intrinsics = scaled_intrinsics(camera, 960, 540)
    assert intrinsics.dtype == np.float64
    assert intrinsics.tolist() == [[750.0, 0.0, 480.5], [0.0, 751.0, 269.5], [0.0, 0.0, 1.0]]


# Req 3.4: the mask scaling and the Masked_Pixels

def under_centre(index: int, new: int, old: int) -> int:
    """Which of `old` mask pixels lies under the centre of training pixel `index`, when `old`
    pixels become `new`: the centre, index + 1/2 in training pixels, is (index + 1/2) × old / new
    in mask pixels, and the pixel under it is that value's floor, worked out exactly."""
    return math.floor((index + Fraction(1, 2)) * Fraction(old, new))


@pytest.mark.parametrize(("size", "taken"), [(4, [1, 3]), (3, [0, 2])])
def test_keep_mask_takes_the_mask_pixel_under_each_training_pixels_centre(
        size: int, taken: list[int]) -> None:
    """A size × size Mask_File scaled to 2 × 2 gives each training pixel the mask pixel under its
    centre (Req 3.4). That's rows and columns 1 and 3 of 4 × 4 and 0 and 2 of 3 × 3, where
    cv2.INTER_NEAREST would take 0 and 2, and 0 and 1. Each mask keeps one pixel and masks the
    rest, at each place in turn, so the keep mask shows which training pixels took that one."""
    assert [under_centre(index, 2, size) for index in range(2)] == taken
    for row in range(size):
        for column in range(size):
            grey = np.zeros((size, size), np.uint8)
            grey[row, column] = 255
            keep = keep_mask(grey, 2, 2, THRESHOLD)
            assert keep.dtype == np.bool_
            assert keep.tolist() == [[under_centre(y, 2, size) == row
                                      and under_centre(x, 2, size) == column for x in range(2)]
                                     for y in range(2)]


def test_keep_mask_masks_127_and_keeps_128_with_a_threshold_of_128() -> None:
    """A pixel below mask_threshold is a Masked_Pixel, False, and one at it is kept (Req 3.4).
    The mask is already at the Training_Size, so each training pixel takes its own mask pixel."""
    grey = np.array([[127, 128], [128, 127]], np.uint8)
    assert keep_mask(grey, 2, 2, THRESHOLD).tolist() == [[False, True], [True, False]]


# Req 3.5: the world-to-camera matrices and the Camera_Centres

ANGLE = 0.25  # radians, the turn about y the tests use
TURN = np.array([math.cos(ANGLE / 2), 0.0, math.sin(ANGLE / 2), 0.0])  # its quaternion, w x y z


def test_world_to_camera_of_the_identity_quaternion_is_the_identity_and_t() -> None:
    """The quaternion (1, 0, 0, 0) gives R = I, beside the translation (Req 3.5)."""
    matrix = world_to_camera(np.array([1.0, 0.0, 0.0, 0.0]), np.array([1.0, 2.0, 3.0]))
    assert matrix.dtype == np.float64
    assert matrix.tolist() == [[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 2.0], [0.0, 0.0, 1.0, 3.0],
                               [0.0, 0.0, 0.0, 1.0]]


@pytest.mark.parametrize("scale", [1.0, 1e-300, 1e300])
def test_world_to_camera_of_a_turn_about_y_is_the_same_times_1e_minus_300_or_1e300(
        scale: float) -> None:
    """The quaternion of a turn about y gives the turn's matrix, and so does that quaternion
    times 1e-300 or 1e300, whose squares would underflow to 0 or overflow to infinity if its
    length were taken as it is (Req 3.5)."""
    translation = (0.5, -1.0, 2.0)
    matrix = world_to_camera(TURN * scale, np.array(translation))
    assert np.allclose(matrix, pose(turn_about_y(ANGLE), translation), rtol=0, atol=1e-15)


def test_camera_centres_are_minus_r_transpose_t() -> None:
    """Each Camera_Centre is −Rᵀt (Req 3.5). The camera turned by ANGLE about y with t = (0, 0, 1)
    sits at (sin, 0, −cos), where −Rt would give (−sin, 0, −cos), and the camera with R = I and
    t = (1, 2, 3) sits at (−1, −2, −3)."""
    matrices = np.stack([pose(turn_about_y(ANGLE), (0.0, 0.0, 1.0)),
                         pose(np.eye(3), (1.0, 2.0, 3.0))])
    centres = camera_centres(matrices)
    assert centres.shape == (2, 3)
    assert np.allclose(centres, [[math.sin(ANGLE), 0.0, -math.cos(ANGLE)], [-1.0, -2.0, -3.0]],
                       rtol=0, atol=1e-15)


# Req 3.6, 3.7: the Scene_Extent

def test_scene_extent_of_four_cameras_on_the_unit_circle_is_1_1() -> None:
    """Four Camera_Centres a quarter turn apart on the unit circle have their mean at its centre
    and each sit 1 from it, so the Scene_Extent is scene_extent_factor × 1 = 1.1 (Req 3.6)."""
    centres = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
    assert scene_extent(centres, EXTENT_FACTOR) == pytest.approx(1.1, rel=1e-12)


def test_scene_extent_refuses_cameras_that_all_sit_in_one_place() -> None:
    """Camera_Centres all in one place are refused: the message says the camera didn't move, so
    no splat can be trained, and to film while moving through the place (Req 3.7, 14.6). Three
    centres at 0.1 have a mean of 0.10000000000000002 when it's taken as they are, so they check
    that the largest distance still comes out as exactly 0."""
    centres = np.full((3, 3), 0.1)
    assert centres.mean(axis=0)[0] != 0.1
    message = refused(lambda: scene_extent(centres, EXTENT_FACTOR))
    for words in ("camera didn't move", "no splat can be trained",
                  "Film while you move through the place"):
        assert words in message


# Req 3.8, 3.9: the Median_Scene_Depth

HALF_TURN = np.diag([-1.0, 1.0, -1.0])  # a half turn about y: the camera looks along world −z
# Five Sparse_Points at world z = 1 to 5.
POINTS = np.array([[0.3, -0.2, 1.0], [-0.4, 0.1, 2.0], [0.2, 0.5, 3.0], [-0.1, -0.3, 4.0],
                   [0.6, 0.2, 5.0]])
SEES_ALL = pose(np.eye(3), (0.0, 0.0, 0.0))  # z = 1 to 5: all 5 in front
SEES_SOME = pose(np.eye(3), (0.0, 0.0, -1.5))  # z = −0.5 to 3.5: 4 in front
FACES_AWAY = pose(HALF_TURN, (0.0, 0.0, 0.0))  # z = −1 to −5: none in front


def in_front(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    """The camera-space z above 0 of `points` for the camera `matrix`, from the full R p + t."""
    z = (points @ matrix[:3, :3].T + matrix[:3, 3])[:, 2]
    return z[z > 0]


def test_median_scene_depth_leaves_out_a_camera_that_faces_away_and_matches_numpy_median() -> None:
    """The camera that faces away from every point is left out, and the rest give the median of
    their depths' medians, both taken with numpy.median (Req 3.8). The camera that sees 4 points
    has the mean of its two middle depths, 2, and the two cameras left the mean of 3 and 2."""
    matrices = np.stack([SEES_ALL, FACES_AWAY, SEES_SOME])
    fronts = [in_front(matrix, POINTS) for matrix in matrices]
    assert [front.size for front in fronts] == [5, 0, 4]
    oracle = np.median([np.median(front) for front in fronts if front.size])
    assert oracle == 2.5
    assert median_scene_depth(matrices, POINTS) == oracle


def test_median_scene_depth_refuses_when_no_camera_has_a_point_in_front() -> None:
    """With every point behind every camera, it's refused: the message says the camera solve put
    no point in front of any camera (Req 3.9, 14.6)."""
    past_them = pose(np.eye(3), (0.0, 0.0, -10.0))  # at z = 10, looking along +z, past them all
    message = refused(lambda: median_scene_depth(np.stack([FACES_AWAY, past_them]), POINTS))
    assert "camera solve put no point in front of any camera" in message


# Req 1.19, 3.8, 5.10: the Scene

def test_build_scene_gives_one_view_per_image_in_file_order_and_the_near_plane(
        tmp_path: Path) -> None:
    """build_scene makes one TrainingView per image of images.bin, holding none out, in the file's
    order, each with its own image's pixels and keep mask (Req 1.19, 5.10). The default scene's
    images are written with ids 3, 1 and 2, so the file's order is neither the ids' nor the
    names'. The Near_Plane is near_plane_factor × the Median_Scene_Depth (Req 3.8), which the
    oracle works out from the default cameras' angles: the camera at angle a has t = (0, 0, 1), so
    a point (x, y, z) sits at depth −sin(a)·x + cos(a)·z + 1, in front of every camera."""
    images = (DEFAULT_IMAGES[2], DEFAULT_IMAGES[0], DEFAULT_IMAGES[1])
    sparse = tmp_path / "sparse"
    write_model(sparse, images=images)
    model = read_model(sparse)
    width, height = DEFAULT_CAMERA.width, DEFAULT_CAMERA.height
    masked = np.ones((height, width), bool)
    masked[0, 0] = False
    pixels = [ImagePixels(np.full((height, width, 3), place, np.uint8), keep)
              for place, keep in enumerate([None, masked, None])]
    settings = read_settings(load_config().step_settings("train", "fast"), "train", "fast")

    scene = build_scene(model, pixels, width, height, settings)

    assert [view.name for view in scene.views] == ["000002.jpg", "000000.jpg", "000001.jpg"]
    assert [int(view.pixels[0, 0, 0]) for view in scene.views] == [0, 1, 2]
    assert [view.keep is None for view in scene.views] == [True, False, True]
    assert np.array_equal(scene.views[1].keep, masked)
    depth = statistics.median(
        statistics.median(-math.sin(angle) * x + math.cos(angle) * z + 1.0 for x, _, z in GRID)
        for angle in ARC_ANGLES)
    assert scene.median_depth == pytest.approx(depth, rel=1e-12)
    assert scene.near_plane == pytest.approx(settings.near_plane_factor * depth, rel=1e-12)
