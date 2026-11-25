"""Tests for splatpipe/cameras/level.py: the Levelling of Requirement 9, which makes the scene
level, faces it along +Z and centres it, and move_model, which applies the Level_Move.

Every test builds its own cameras and points, with no COLMAP (Req 16.1). The conventions are
COLMAP's, as in level.py: a camera's Pose has the cam_from_world rotation R and the translation
t, the rows of R are the camera's x, y and z axes in the world, its Up_Vector is Rᵀ (0, −1, 0),
its View_Direction Rᵀ (0, 0, 1) and its centre −Rᵀ t. `camera` builds R for a camera of a level
scene from its yaw, pitch and roll, `ring` stands cameras on a circle, and most scenes are then
seen through TILT and SHIFT, a rigid move into a made-up COLMAP_Frame, so the Levelling has a
tilt to take out.

Every expected value is worked out here with the test's own numpy code, never with level.py:
seen_through applies a Level_Move, assert_level works out the Mean_Up, the Heading and the
Scene_Centre again in the moved scene, nearest_to_axes finds the Axis_Point by least squares on
the view lines, and rotation_of gives a quaternion's rotation by the quaternion formula. The
warnings are checked against the design's wording of Req 9.3, 9.7, 9.10 and 9.12, with the
values filled in.

Property 4 draws scenes with the design's Hypothesis strategies. rotations() gives a unit
quaternion from four floats in [−1, 1] whose norm is at least 0.1, and scenes() gives 1 to 12
cameras, each with a rotation from rotations() and a centre in [−50, 50]³, 0 to 30 points in
[−50, 50]³, and the four levelling settings across their ranges. Coordinates within tens of units
and quaternion norms away from 0 keep the scenes well conditioned, so the float tolerances hold.
Each drawn scene is a Model, which goes through scene_of, level and move_model as in the step, and
distances works out the distance between every two of the camera centres and the points, before
and after, with each centre −Rᵀ t and R by the quaternion formula. Property 5 levels the same
scenes and checks the Level_Move's rotation R with numpy: RᵀR against I and its determinant
against 1, each within 1e-12, and, when the COLMAP_Frame is kept, the move against exactly the
identity and a zero translation.

Property 6 draws scenes with agreeing_scenes(): a level scene of 1 to 12 cameras from `camera`,
each with a yaw in [−180°, 180°], a pitch and a roll each in [−20°, 20°], so its up is within 28°
of UP, and a centre in [−20, 20]³, and 1 to 30 points in [−20, 20]³, seen through a rigid move
with a rotation from rotations() and a translation in [−100, 100]³, with any up_agreement up to
0.85 and any heading_min_degrees up to 25°. Every camera then has a Heading, so Pick 0 is the
Heading_Camera. level moves the scene, and assert_level works out again in the moved scene the
Mean_Up, Pick 0's flattened View_Direction and the Scene_Centre of the kind level used.
"""

import math
import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from splatpipe.cameras.level import (LevelMove, LevelSettings, Levelling, Scene, axis_point, level,
                                     move_model, quaternion_from_rotation, rotation_onto, scene_of)
from splatpipe.cameras.model import (NO_ERROR, NO_POINT3D, POINT2D, TRACK_ELEMENT, Camera, Image,
                                     Model, Points)
from splatpipe.frames.files import pick_name

# The real config's four levelling settings (O3, A5), written out here.
SETTINGS = LevelSettings(up_agreement=0.5, heading_min_degrees=10.0,
                         axis_point_min_eigen_share=0.1, axis_point_min_front_share=0.5)
UP = np.array([0.0, -1.0, 0.0])  # the World_Up
FORWARD = np.array([0.0, 0.0, 1.0])  # +Z, where the Heading_Camera's Heading has to point
HALF_TURN = np.diag([1.0, -1.0, -1.0])  # P, half a turn about x, which takes UP to −UP
UPSIDE_DOWN = np.diag([-1.0, -1.0, 1.0])  # exactly half a turn about a camera's own view

# The warnings of a kept COLMAP_Frame, in the design's words (Req 9.3, 9.7, 9.10, 9.12).
# Req 9.3, with up_agreement = 0.5 and the Mean_Up's length as the group "length".
NO_AGREEMENT = re.compile(
    r"The cameras don't agree on which way is up: their mean up vector is (?P<length>\S+) long, "
    r"below up_agreement = 0\.5 in \[cameras\]\. So the scene was left in COLMAP's frame, which "
    r"may be tilted\.")
# Req 9.7, with heading_min_degrees = 15.0, the value its test uses.
NO_HEADING = ("Every placed camera looks within heading_min_degrees = 15.0° of straight up or "
              "down, so the scene couldn't be turned to face +Z and was left in COLMAP's frame, "
              "which may be tilted.")
NO_CENTRE = ("The scene's centre couldn't be found: the cameras' view lines don't meet in a well "
             "defined point and the model has no 3D points. So the scene was left in COLMAP's "
             "frame, which may be tilted.")  # Req 9.10


def failed(error: str) -> str:
    """The Req 9.12 warning for `error`, written as `<type>: <message>`."""
    return (f"Levelling the scene failed ({error}), so the scene was left in COLMAP's frame, "
            "which may be tilted.")


# Rotations, worked out here

def unit(vectors: Any) -> np.ndarray:
    """Each vector along the last axis, divided by its length."""
    vectors = np.asarray(vectors, dtype=float)
    return vectors / np.linalg.norm(vectors, axis=-1, keepdims=True)


def rotation_of(quaternion: Sequence[float]) -> np.ndarray:
    """The rotation of the unit quaternion (w, x, y, z), in Hamilton's convention, by the
    quaternion formula R = (w² − v · v) I + 2 v vᵀ + 2 w [v]×, with v = (x, y, z) and [v]× its
    cross-product matrix."""
    w, x, y, z = (float(part) for part in quaternion)
    v = np.array([x, y, z])
    cross = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return (w * w - v @ v) * np.eye(3) + 2.0 * np.outer(v, v) + 2.0 * w * cross


def axis_quaternion(axis: Sequence[float], degrees: float) -> np.ndarray:
    """The unit quaternion of the right-handed turn by `degrees` about `axis`."""
    half = math.radians(degrees) / 2.0
    return np.concatenate([[math.cos(half)], math.sin(half) * unit(axis)])


def turn_about_x(degrees: float) -> np.ndarray:
    """The right-handed turn by `degrees` about the x axis, with its 0s and its 1 exact."""
    c, s = math.cos(math.radians(degrees)), math.sin(math.radians(degrees))
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def turn_about_y(degrees: float) -> np.ndarray:
    """The right-handed turn by `degrees` about the y axis, with its 0s and its 1 exact."""
    c, s = math.cos(math.radians(degrees)), math.sin(math.radians(degrees))
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def turn_about_z(degrees: float) -> np.ndarray:
    """The right-handed turn by `degrees` about the z axis, with its 0s and its 1 exact."""
    c, s = math.cos(math.radians(degrees)), math.sin(math.radians(degrees))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


# The rigid move from a level scene into a made-up COLMAP_Frame: 70° about (1, 2, −0.5), then a
# shift.
TILT = rotation_of(axis_quaternion((1.0, 2.0, -0.5), 70.0))
SHIFT = np.array([3.0, -7.0, 12.0])


# Cameras and scenes

def camera(yaw: float, pitch: float = 0.0, roll: float = 0.0, *,
           upside_down: bool = False) -> np.ndarray:
    """The cam_from_world rotation R of a camera in a level scene, whose up is UP. It looks `yaw`
    degrees from +Z towards +X and `pitch` degrees above the horizon, is rolled `roll` degrees
    about its own view, and is turned exactly half a turn more about its view when it's
    `upside_down`. The columns of Rᵀ are its x, y and z axes in the world."""
    axes = turn_about_y(yaw) @ turn_about_x(pitch) @ turn_about_z(roll)
    return (axes @ UPSIDE_DOWN if upside_down else axes).T


def outward(angle: float) -> np.ndarray:
    """The level unit vector `angle` degrees from +Z towards +X."""
    return np.array([math.sin(math.radians(angle)), 0.0, math.cos(math.radians(angle))])


def ring(angles: Iterable[float], radius: float, centre: Sequence[float], *, facing: str = "in",
         upside_down: bool = False) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """The rotations and the centres of level cameras on the level circle of `radius` round
    `centre`, one at each of `angles`, each looking at the centre ("in") or straight away from
    it ("out"), so every view line goes through the centre."""
    rotations, centres = [], []
    for angle in angles:
        rotations.append(camera(angle + 180.0 if facing == "in" else angle,
                                upside_down=upside_down))
        centres.append(np.asarray(centre, dtype=float) + radius * outward(angle))
    return rotations, centres


def scene(rotations: Sequence[np.ndarray], centres: Sequence[Any], points: Any = ()) -> Scene:
    """The Scene of these cameras, in Pick_Number order and named as Picks 0, 1, 2 and so on, and
    of these points."""
    return Scene(names=tuple(pick_name(number) for number in range(len(rotations))),
                 rotations=np.array(rotations, dtype=float).reshape(-1, 3, 3),
                 centres=np.array(centres, dtype=float).reshape(-1, 3),
                 points=np.array(points, dtype=float).reshape(-1, 3))


def seen_through(cameras: Scene, rotation: np.ndarray, translation: np.ndarray) -> Scene:
    """`cameras` after the rigid move x → rotation x + translation: each camera's R becomes
    R rotationᵀ, and each centre and each point moves like any point."""
    return Scene(names=cameras.names, rotations=cameras.rotations @ rotation.T,
                 centres=cameras.centres @ rotation.T + translation,
                 points=cameras.points @ rotation.T + translation)


def points_near(centre: Sequence[float], count: int, seed: int) -> np.ndarray:
    """`count` points up to 1.5 from `centre` along each axis, the same ones for each seed."""
    spread = np.random.default_rng(seed).uniform(-1.5, 1.5, (count, 3))
    return np.asarray(centre, dtype=float) + spread


CENTRE = np.array([0.4, -1.1, 3.0])  # where the view lines of the rings below meet


def inward_cameras() -> Scene:
    """Six level cameras a sixth of a turn apart on a circle of radius 2.5 round CENTRE, each
    looking at it, with twelve points near it, seen through TILT and SHIFT."""
    rotations, centres = ring(range(0, 360, 60), 2.5, CENTRE)
    return seen_through(scene(rotations, centres, points_near(CENTRE, 12, seed=1)), TILT, SHIFT)


def parallel_cameras(points: Any) -> Scene:
    """Five cameras in a row, each looking along +Z and rolled 30° about its view the same way,
    with `points`. Each view is exactly (0, 0, 1), so A = Σ (I − d dᵀ) is exactly
    diag(5, 5, 0). The roll is the COLMAP_Frame's tilt: TILT would leave A only nearly
    singular."""
    rolled = camera(0.0, 0.0, 30.0)
    rows = [(x, 0.25 * x, -1.0) for x in (-2.0, -1.0, 0.0, 1.0, 2.0)]
    return scene([rolled] * len(rows), rows, points)


def outward_cameras(points: Any) -> Scene:
    """Six level cameras on the circle of inward_cameras, each looking straight away from CENTRE,
    so their view lines meet at CENTRE, behind every one of them, with `points`, seen through
    TILT and SHIFT."""
    rotations, centres = ring(range(0, 360, 60), 2.5, CENTRE, facing="out")
    return seen_through(scene(rotations, centres, points), TILT, SHIFT)


# Models, for scene_of and move_model

CAMERA = Camera(camera_id=1, model_id=1, width=320, height=240,
                params=(260.0, 260.0, 160.0, 120.0))  # PINHOLE: fx, fy, cx, cy


def image(image_id: int, number: int, quaternion: Sequence[float], centre: Sequence[float],
          points2d: Sequence[tuple[float, float, int]] = ()) -> Image:
    """Image `image_id` of Pick `number`, named as the Pick, with the Pose of a camera at `centre`
    whose rotation R has the unit `quaternion`: t = −R C."""
    tvec = -rotation_of(quaternion) @ np.asarray(centre, dtype=float)
    return Image(image_id=image_id, qvec=tuple(np.asarray(quaternion, dtype=float).tolist()),
                 tvec=tuple(tvec.tolist()), camera_id=CAMERA.camera_id, name=pick_name(number),
                 points2d=np.array(list(points2d), dtype=POINT2D))


def ring_images(angles: Sequence[float], radius: float, centre: Sequence[float]) -> list[Image]:
    """The Images of the cameras that `ring` stands at `angles`, looking in, Pick k at angles[k]
    with the image id k + 1. A level camera with the yaw ψ has R = Ry(−ψ), whose quaternion is
    the turn by −ψ about y."""
    return [image(number + 1, number, axis_quaternion((0.0, 1.0, 0.0), -(angle + 180.0)),
                  np.asarray(centre, dtype=float) + radius * outward(angle))
            for number, angle in enumerate(angles)]


def points_at(xyz: np.ndarray) -> Points:
    """Points at `xyz`, with the ids 1, 2, 3 and so on, grey, with no error and empty tracks."""
    count = len(xyz)
    return Points(ids=np.arange(1, count + 1, dtype="<u8"),
                  xyz=np.asarray(xyz, dtype="<f8").reshape(-1, 3),
                  rgb=np.full((count, 3), 128, dtype="u1"), errors=np.full(count, NO_ERROR),
                  tracks=tuple(np.zeros(0, dtype=TRACK_ELEMENT) for _ in range(count)))


def array_key(array: np.ndarray) -> tuple[Any, ...]:
    """An array's dtype, shape and bytes, which are equal only for arrays equal bit for bit."""
    return array.dtype, array.shape, array.tobytes()


# The checks, worked out here

def assert_close(actual: Any, expected: Any, tolerance: float) -> None:
    """Every number of `actual` is within `tolerance` of the same number of `expected`."""
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=tolerance)


def nearest_to_axes(views: np.ndarray, centres: np.ndarray) -> np.ndarray:
    """The point with the smallest sum of squared distances to the view lines, by least squares
    on the lines stacked together: the distance from p to the line through C along d is
    |(I − d dᵀ)(p − C)|."""
    across = np.eye(3) - views[:, :, None] * views[:, None, :]
    targets = np.einsum("nij,nj->ni", across, centres)
    point, *_ = np.linalg.lstsq(across.reshape(-1, 3), targets.reshape(-1), rcond=None)
    return point


def axis_point_of(cameras: Scene, settings: LevelSettings = SETTINGS) -> np.ndarray | None:
    """axis_point of the View_Directions of `cameras`, the third rows of their R, and their
    centres, with the two shares of `settings`."""
    return axis_point(cameras.rotations[:, 2, :], cameras.centres,
                      min_eigen_share=settings.axis_point_min_eigen_share,
                      min_front_share=settings.axis_point_min_front_share)


def assert_level(cameras: Scene, levelling: Levelling, *, heading: int, kind: str) -> None:
    """`levelling` moved `cameras`, with Pick `heading` as the Heading_Camera and a Scene_Centre
    of `kind`. In the moved scene, worked out again here, the Mean_Up points along UP and Pick
    `heading`'s flattened View_Direction along +Z, each within 1e-9, and the Scene_Centre lies
    within 1e-9 times (1 + the scene's size) of the origin (Req 9.4, 9.5, 9.9, 9.11)."""
    assert (levelling.levelled, levelling.reason) == (True, None)
    assert (levelling.heading_camera, levelling.scene_centre) == (cameras.names[heading], kind)
    moved = seen_through(cameras, levelling.move.rotation, levelling.move.translation)
    mean_up = -moved.rotations[:, 1, :].mean(axis=0)  # the Up_Vector is −(R's second row)
    assert_close(mean_up / np.linalg.norm(mean_up), UP, 1e-9)
    x, _, z = moved.rotations[heading, 2, :]
    assert_close(np.array([x, 0.0, z]) / math.hypot(x, z), FORWARD, 1e-9)
    if kind == "axis_point":
        centre = nearest_to_axes(moved.rotations[:, 2, :], moved.centres)
    else:  # the Point_Median, along the levelled axes (A6)
        centre = np.median(moved.points, axis=0)
    size = np.abs(np.concatenate([cameras.centres, cameras.points])).max()
    assert_close(centre, np.zeros(3), 1e-9 * (1.0 + size))


def assert_kept(levelling: Levelling) -> None:
    """`levelling` kept the COLMAP_Frame: exactly the identity rotation and a zero translation,
    with no Scene_Centre (Req 9.13)."""
    assert not levelling.levelled
    assert levelling.scene_centre is None
    assert np.array_equal(levelling.move.rotation, np.eye(3))
    assert np.array_equal(levelling.move.translation, np.zeros(3))


# Cameras that agree and cameras that don't (Req 9.2 to 9.5, 9.11)

def test_cameras_that_agree_come_out_level_facing_plus_z_and_centred() -> None:
    # Eight cameras on half a circle look at its centre, each tilted up to 10° up or down and
    # rolled up to 9°, so they agree on which way is up, but not exactly.
    angles = range(0, 200, 25)
    pitches = (5.0, -8.0, 3.0, 10.0, -4.0, 7.0, -10.0, 2.0)
    rolls = (-6.0, 4.0, 9.0, -3.0, 8.0, -7.0, 1.0, -9.0)
    centre = np.array([0.5, -1.2, 2.0])
    rotations = [camera(angle + 180.0, pitch, roll)
                 for angle, pitch, roll in zip(angles, pitches, rolls, strict=True)]
    centres = [centre + 4.0 * outward(angle) for angle in angles]
    cameras = seen_through(scene(rotations, centres, points_near(centre, 15, seed=2)), TILT,
                           SHIFT)
    levelling = level(cameras, SETTINGS)
    assert_level(cameras, levelling, heading=0, kind="axis_point")
    # The Mean_Up is the mean of every camera's Up_Vector, each counted once (Req 9.2).
    mean_up = -cameras.rotations[:, 1, :].mean(axis=0)
    assert levelling.mean_up_length == pytest.approx(np.linalg.norm(mean_up), abs=1e-12)


def test_cameras_that_dont_agree_keep_the_colmap_frame_with_the_mean_ups_length() -> None:
    # Three cameras stand upright, a third of a turn apart, and three are upside down and look
    # 60° up, also a third of a turn apart, so the mean of their up vectors is (0, −0.25, 0)
    # before TILT.
    rotations = ([camera(yaw) for yaw in (0.0, 120.0, 240.0)]
                 + [camera(yaw, 60.0, upside_down=True) for yaw in (60.0, 180.0, 300.0)])
    centres = [3.0 * outward(angle) for angle in range(0, 360, 60)]
    cameras = seen_through(scene(rotations, centres, points_near((0.0, 0.0, 0.0), 6, seed=3)),
                           TILT, SHIFT)
    length = float(np.linalg.norm(-cameras.rotations[:, 1, :].mean(axis=0)))
    assert length == pytest.approx(0.25, abs=1e-12)
    levelling = level(cameras, SETTINGS)
    assert_kept(levelling)
    match = NO_AGREEMENT.fullmatch(levelling.reason or "")
    assert match is not None, levelling.reason
    assert float(match["length"]) == pytest.approx(length, abs=1e-12)
    assert levelling.mean_up_length == pytest.approx(length, abs=1e-12)


# The Heading_Camera (Req 9.5 to 9.7)

def test_a_first_frame_looking_straight_down_makes_the_next_one_the_heading_camera() -> None:
    # Picks 1 to 4 stand on a circle and look at its centre. Picks 0 and 5 hang above the centre,
    # where up is −y, and look straight down at it, half a turn apart, so their up vectors cancel
    # out and the other four keep the Mean_Up upright.
    centre = np.array([1.0, 0.5, -2.0])
    standing, around = ring((30.0, 120.0, 210.0, 300.0), 3.0, centre)
    rotations = [camera(0.0, -90.0), *standing, camera(180.0, -90.0)]
    centres = [centre + (0.0, -2.0, 0.0), *around, centre + (0.0, -3.0, 0.0)]
    cameras = seen_through(scene(rotations, centres, points_near(centre, 10, seed=4)), TILT,
                           SHIFT)
    assert_level(cameras, level(cameras, SETTINGS), heading=1, kind="axis_point")


def test_every_frame_looking_straight_down_keeps_the_colmap_frame() -> None:
    # Cameras that all look within h of straight down have a Mean_Up at most sin h long, so they
    # never pass the real up_agreement of 0.5. These six look 10° from straight down, a sixth of
    # a turn apart, so their Mean_Up is cos 80° ≈ 0.17 long, which passes an up_agreement of 0.1,
    # and each is within heading_min_degrees = 15° of straight down.
    settings = replace(SETTINGS, up_agreement=0.1, heading_min_degrees=15.0)
    rotations = [camera(yaw, -80.0) for yaw in range(0, 360, 60)]
    centres = [(0.0, -4.0, 0.0) + 0.7 * outward(yaw) for yaw in range(0, 360, 60)]
    cameras = seen_through(scene(rotations, centres, points_near((0.0, 0.0, 0.0), 8, seed=5)),
                           TILT, SHIFT)
    levelling = level(cameras, settings)
    assert_kept(levelling)
    assert levelling.reason == NO_HEADING


def test_the_heading_camera_goes_by_pick_number_when_images_bin_lists_another_order() -> None:
    # Pick k stands at angles[k] on a circle and looks at its centre, and images.bin lists the
    # Picks as 2, 0, 3, 1.
    angles = (210.0, 280.0, 120.0, 350.0)
    centre = np.array([-0.5, 0.3, 1.5])
    images = ring_images(angles, 2.0, centre)
    xyz = points_near(centre, 6, seed=6)
    model = Model(cameras=(CAMERA,), images=tuple(images[number] for number in (2, 0, 3, 1)),
                  points=points_at(xyz))
    cameras = scene_of(model, {pick_name(number): number for number in range(4)})
    rotations, centres = ring(angles, 2.0, centre)
    assert cameras.names == tuple(pick_name(number) for number in range(4))
    assert_close(cameras.rotations, rotations, 1e-12)
    assert_close(cameras.centres, centres, 1e-12)
    assert np.array_equal(cameras.points, xyz)
    assert_level(cameras, level(cameras, SETTINGS), heading=0, kind="axis_point")


# The Scene_Centre (Req 9.8 to 9.11)

def test_cameras_on_a_circle_looking_at_its_centre_have_that_centre_as_their_axis_point() -> None:
    cameras = inward_cameras()
    point = axis_point_of(cameras)
    assert point is not None
    assert_close(point, TILT @ CENTRE + SHIFT, 1e-9)
    assert_level(cameras, level(cameras, SETTINGS), heading=0, kind="axis_point")


def test_exactly_parallel_cameras_fail_the_eigenvalue_share_and_use_the_point_median() -> None:
    cameras = parallel_cameras(points_near((0.5, 0.0, 6.0), 9, seed=7))
    views = cameras.rotations[:, 2, :]
    # A is exactly singular, so numpy.linalg.solve would raise on it, and the Levelling would
    # give the Req 9.12 warning, if the eigenvalue share didn't come first.
    across = sum(np.eye(3) - np.outer(view, view) for view in views)
    assert np.array_equal(across, np.diag([5.0, 5.0, 0.0]))
    assert axis_point_of(cameras) is None
    assert_level(cameras, level(cameras, SETTINGS), heading=0, kind="point_median")


def test_outward_cameras_have_their_axis_point_behind_them_and_use_the_point_median() -> None:
    cameras = outward_cameras(points_near(CENTRE + (0.0, 0.0, 5.0), 9, seed=8))
    views = cameras.rotations[:, 2, :]
    nearest = nearest_to_axes(views, cameras.centres)
    assert_close(nearest, TILT @ CENTRE + SHIFT, 1e-9)
    assert (np.einsum("ni,ni->n", nearest - cameras.centres, views) < 0).all()
    assert axis_point_of(cameras) is None
    assert_level(cameras, level(cameras, SETTINGS), heading=0, kind="point_median")


@pytest.mark.parametrize(("share", "in_front"), [
    # 0.5 is exact in binary, so 5 of 10 is exactly the setting's fraction, and only "at least"
    # passes it.
    pytest.param(0.5, 5, id="5 of 10 at the real 0.5"),
    # C7's own example: the float 0.3 is just below 3/10, and 3 of 10 is enough.
    pytest.param(0.3, 3, id="3 of 10 at 0.3"),
])
def test_an_in_front_share_of_exactly_axis_point_min_front_share_passes(share: float,
                                                                        in_front: int) -> None:
    # Ten cameras a tenth of a turn apart on a circle: the first `in_front` Picks look at its
    # centre and the others straight away from it. Every view line goes through the centre,
    # which is in front of exactly `in_front` of the 10 cameras, compared with an exact fraction
    # of the setting (C7).
    angles = tuple(range(0, 360, 36))
    looking_in, around_in = ring(angles[:in_front], 2.5, CENTRE)
    looking_out, around_out = ring(angles[in_front:], 2.5, CENTRE, facing="out")
    cameras = seen_through(scene(looking_in + looking_out, around_in + around_out,
                                 points_near(CENTRE, 8, seed=9)), TILT, SHIFT)
    settings = replace(SETTINGS, axis_point_min_front_share=share)
    point = axis_point_of(cameras, settings)
    assert point is not None
    assert_close(point, TILT @ CENTRE + SHIFT, 1e-9)
    assert_level(cameras, level(cameras, settings), heading=0, kind="axis_point")


@pytest.mark.parametrize("cameras_with", [
    pytest.param(parallel_cameras, id="parallel cameras"),
    pytest.param(outward_cameras, id="outward-looking cameras"),
])
def test_no_points_and_no_well_defined_axis_point_keep_the_colmap_frame(
        cameras_with: Callable[[Any], Scene]) -> None:
    levelling = level(cameras_with(()), SETTINGS)
    assert_kept(levelling)
    assert levelling.reason == NO_CENTRE


# The half turn (Req 9.4)

def test_a_mean_up_along_0_1_0_takes_the_half_turn_and_still_comes_out_level() -> None:
    # Six cameras on a circle look at its centre, each upside down, so every up vector, and the
    # Mean_Up, is exactly (0, 1, 0). There's no TILT, which would move it.
    rotations, centres = ring(range(0, 360, 60), 2.5, CENTRE, upside_down=True)
    cameras = scene(rotations, centres, points_near(CENTRE, 12, seed=10))
    assert np.array_equal(-cameras.rotations[:, 1, :].mean(axis=0), -UP)
    levelling = level(cameras, SETTINGS)
    assert_level(cameras, levelling, heading=0, kind="axis_point")
    assert levelling.mean_up_length == 1.0


# Errors while the Level_Move is worked out (Req 9.12)

@pytest.mark.parametrize(("part", "place"), [
    pytest.param("qvec", 2, id="a NaN in the rotation"),
    pytest.param("tvec", 1, id="a NaN in the translation"),
])
def test_a_nan_in_a_pose_keeps_the_colmap_frame_with_the_error(part: str, place: int) -> None:
    images = ring_images((0.0, 120.0, 240.0), 2.0, CENTRE)
    broken = list(getattr(images[1], part))
    broken[place] = math.nan
    images[1] = replace(images[1], **{part: tuple(broken)})
    model = Model(cameras=(CAMERA,), images=tuple(images),
                  points=points_at(points_near(CENTRE, 5, seed=11)))
    levelling = level(scene_of(model, {pick_name(number): number for number in range(3)}),
                      SETTINGS)
    assert_kept(levelling)
    assert levelling.reason == failed(
        "ValueError: the pose of 000001.jpg holds a value that isn't a finite number")


def test_an_error_inside_the_maths_keeps_the_colmap_frame_with_the_error(
        monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[np.ndarray] = []

    def broken_solve(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        calls.append(a)
        raise np.linalg.LinAlgError("Singular matrix")

    cameras = inward_cameras()
    monkeypatch.setattr(np.linalg, "solve", broken_solve)
    levelling = level(cameras, SETTINGS)
    assert len(calls) == 1  # the Axis_Point's solve, once the eigenvalue share has passed
    assert_kept(levelling)
    assert levelling.reason == failed("LinAlgError: Singular matrix")


# rotation_onto and Shepperd's method (Req 9.4, 9.13)

def test_rotation_onto_the_world_up_from_itself_is_the_identity() -> None:
    assert np.array_equal(rotation_onto(UP, UP), np.eye(3))


def test_rotation_onto_the_world_up_from_minus_it_is_the_half_turn_about_x() -> None:
    assert np.array_equal(rotation_onto(-UP, UP), HALF_TURN)


def test_rotation_onto_takes_any_unit_vector_to_the_world_up() -> None:
    # 200 random directions, and directions just either side of ±UP and of the level plane,
    # where a · g changes sign.
    edges = [(1e-9, 1.0, 0.0), (0.0, 1.0, 1e-9), (1e-9, -1.0, 0.0), (1.0, 0.0, 0.0),
             (0.0, 1e-9, 1.0), (0.0, -1e-9, 1.0), (0.6, 0.0, -0.8)]
    vectors = np.concatenate([np.random.default_rng(12).normal(size=(200, 3)), edges])
    for a in unit(vectors):
        rotation = rotation_onto(a, UP)
        assert_close(rotation @ a, UP, 1e-12)
        assert_close(rotation.T @ rotation, np.eye(3), 1e-12)
        assert np.linalg.det(rotation) == pytest.approx(1.0, abs=1e-12)


def test_shepperds_method_gives_back_each_quaternion_up_to_its_sign() -> None:
    # 200 random unit quaternions, and ones whose w, x, y or z is the largest part, so Shepperd's
    # method starts from each of its four terms, with the half turns about x, y and z, whose w
    # is 0.
    special = [(1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0),
               (0.0, 0.0, 0.0, 1.0), (0.9, 0.3, -0.2, 0.1), (-0.2, 0.9, 0.3, -0.1),
               (0.1, -0.3, 0.9, 0.2), (0.3, 0.1, -0.2, -0.9), (0.5, 0.5, 0.5, 0.5)]
    quaternions = np.concatenate([np.random.default_rng(13).normal(size=(200, 4)), special])
    for quaternion in unit(quaternions):
        found = quaternion_from_rotation(rotation_of(quaternion))
        assert min(np.abs(found - quaternion).max(), np.abs(found + quaternion).max()) <= 1e-12


# move_model (Req 9.1, 9.13, 10.6 to 10.8)

def test_move_model_moves_every_pose_and_point_and_keeps_everything_else() -> None:
    rng = np.random.default_rng(14)
    quaternions = unit(rng.normal(size=(3, 4)))
    # Image ids in another order than the Picks, a 2D point with NaN coordinates and no 3D point,
    # and an image with no 2D points.
    images = (image(7, 2, quaternions[0], rng.uniform(-5.0, 5.0, 3),
                    [(10.5, 20.25, 3), (math.nan, 1.5, NO_POINT3D)]),
              image(3, 0, quaternions[1], rng.uniform(-5.0, 5.0, 3)),
              image(12, 1, quaternions[2], rng.uniform(-5.0, 5.0, 3), [(300.0, 200.0, 2**64 - 2)]))
    points = Points(ids=np.array([3, 2**64 - 2], dtype="<u8"), xyz=rng.uniform(-5.0, 5.0, (2, 3)),
                    rgb=np.array([[255, 0, 128], [12, 34, 56]], dtype="u1"),
                    errors=np.array([0.75, NO_ERROR]),
                    tracks=(np.array([(7, 0)], dtype=TRACK_ELEMENT),
                            np.array([(12, 0)], dtype=TRACK_ELEMENT)))
    model = Model(cameras=(CAMERA,), images=images, points=points)
    rotation = rotation_of(unit(rng.normal(size=4)))
    translation = rng.uniform(-10.0, 10.0, 3)
    moved = move_model(model, LevelMove(rotation=rotation, translation=translation))
    assert moved.cameras == model.cameras
    for before, after in zip(model.images, moved.images, strict=True):
        assert ((after.image_id, after.camera_id, after.name)
                == (before.image_id, before.camera_id, before.name))
        assert array_key(after.points2d) == array_key(before.points2d)
        assert np.linalg.norm(after.qvec) == pytest.approx(1.0, abs=1e-12)
        turned, was = rotation_of(after.qvec), rotation_of(before.qvec)
        assert_close(turned, was @ rotation.T, 1e-12)  # it sees R_i Rᵀ x' for each moved x'
        # Its centre −Rᵀ t moves to R C + T.
        assert_close(-turned.T @ np.array(after.tvec),
                     rotation @ (-was.T @ np.array(before.tvec)) + translation, 1e-9)
    assert_close(moved.points.xyz, points.xyz @ rotation.T + translation, 1e-9)
    assert ([array_key(part) for part in (moved.points.ids, moved.points.rgb, moved.points.errors)]
            == [array_key(part) for part in (points.ids, points.rgb, points.errors)])
    assert ([array_key(track) for track in moved.points.tracks]
            == [array_key(track) for track in points.tracks])


# The Hypothesis strategies for the levelling (Properties 4 to 6)

COORDINATES = st.tuples(*[st.floats(-50.0, 50.0)] * 3)  # a centre or a point in [−50, 50]³
# The four levelling settings, each across the range the design gives.
LEVEL_SETTINGS = st.builds(LevelSettings, up_agreement=st.floats(0.05, 1.0),
                           heading_min_degrees=st.floats(1.0, 89.0),
                           axis_point_min_eigen_share=st.floats(0.01, 1.0),
                           axis_point_min_front_share=st.floats(0.0, 1.0))


def rotations() -> st.SearchStrategy[tuple[float, ...]]:
    """A unit quaternion (w, x, y, z): four floats in [−1, 1] whose norm is at least 0.1,
    normalised. Keeping the norm away from 0 keeps the normalising well conditioned."""
    return (st.tuples(*[st.floats(-1.0, 1.0)] * 4)
            .filter(lambda parts: math.hypot(*parts) >= 0.1)
            .map(lambda parts: tuple(unit(parts).tolist())))


@dataclass(frozen=True)
class SceneCase:
    """A scene that scenes() draws, in the COLMAP_Frame: each camera's unit quaternion and
    centre, in Pick_Number order, the points, and the four levelling settings."""

    quaternions: tuple[tuple[float, ...], ...]
    centres: tuple[tuple[float, ...], ...]
    points: tuple[tuple[float, ...], ...]
    settings: LevelSettings

    def model(self) -> Model:
        """The Model of these cameras and points: Pick k is image k + 1, with the Pose of its
        quaternion and its centre, and the points have the ids 1, 2, 3 and so on."""
        images = tuple(image(number + 1, number, quaternion, centre)
                       for number, (quaternion, centre)
                       in enumerate(zip(self.quaternions, self.centres, strict=True)))
        return Model(cameras=(CAMERA,), images=images,
                     points=points_at(np.array(self.points, dtype=float).reshape(-1, 3)))


@st.composite
def scenes(draw: st.DrawFn) -> SceneCase:
    """For Properties 4 and 5: 1 to 12 cameras, each with a rotation from rotations() and a
    centre in [−50, 50]³, 0 to 30 points in [−50, 50]³, up_agreement in [0.05, 1],
    heading_min_degrees in [1, 89], axis_point_min_eigen_share in [0.01, 1] and
    axis_point_min_front_share in [0, 1]."""
    cameras = draw(st.lists(st.tuples(rotations(), COORDINATES), min_size=1, max_size=12))
    return SceneCase(quaternions=tuple(quaternion for quaternion, _ in cameras),
                     centres=tuple(centre for _, centre in cameras),
                     points=tuple(draw(st.lists(COORDINATES, max_size=30))),
                     settings=draw(LEVEL_SETTINGS))


def pick_numbers(count: int) -> dict[str, int]:
    """The Pick_Numbers of Picks 0 to `count` − 1, by their names."""
    return {pick_name(number): number for number in range(count)}


# agreeing_scenes(), for Property 6

NEAR_COORDINATES = st.tuples(*[st.floats(-20.0, 20.0)] * 3)  # a centre or a point, in [−20, 20]³
TRANSLATIONS = st.tuples(*[st.floats(-100.0, 100.0)] * 3)  # a translation in [−100, 100]³
# The levelling settings of agreeing scenes: up_agreement up to 0.85 and heading_min_degrees up
# to 25°, which every agreeing scene passes, and the two axis point shares across their ranges,
# as in LEVEL_SETTINGS, so level uses both kinds of Scene_Centre.
AGREEING_SETTINGS = st.builds(LevelSettings, up_agreement=st.floats(0.05, 0.85),
                              heading_min_degrees=st.floats(1.0, 25.0),
                              axis_point_min_eigen_share=st.floats(0.01, 1.0),
                              axis_point_min_front_share=st.floats(0.0, 1.0))


@dataclass(frozen=True)
class AgreeingCase:
    """A scene that agreeing_scenes() draws: each camera's yaw, pitch and roll in degrees and its
    centre in the level scene, in Pick_Number order, the points, the rigid move into the
    COLMAP_Frame as a unit quaternion and a translation, and the four levelling settings."""

    turns: tuple[tuple[float, float, float], ...]
    centres: tuple[tuple[float, ...], ...]
    points: tuple[tuple[float, ...], ...]
    quaternion: tuple[float, ...]
    translation: tuple[float, ...]
    settings: LevelSettings

    def cameras(self) -> Scene:
        """The Scene of these cameras, each built by `camera` from its yaw, pitch and roll, and of
        these points, seen through the rigid move."""
        rotations = [camera(yaw, pitch, roll) for yaw, pitch, roll in self.turns]
        return seen_through(scene(rotations, self.centres, self.points),
                            rotation_of(self.quaternion), np.array(self.translation))


@st.composite
def agreeing_scenes(draw: st.DrawFn) -> AgreeingCase:
    """For Property 6: a level scene of 1 to 12 cameras, as many as scenes() draws, each with a
    yaw in [−180°, 180°], a pitch and a roll each in [−20°, 20°] and a centre in [−20, 20]³, and
    1 to 30 points in [−20, 20]³. Then a rigid move with a rotation from rotations() and a
    translation in [−100, 100]³, up_agreement in [0.05, 0.85], heading_min_degrees in [1, 25],
    and the two axis point shares as in scenes().

    Each camera's up has cos(pitch) cos(roll) ≥ cos² 20° ≈ 0.883 along the common up, so it's
    within 28° of it, and the Mean_Up is at least 0.883 long, which passes any up_agreement up
    to 0.85. After the levelling, each camera's up is within 56° of the World_Up, so its view is
    at least 34° from straight up or down, and any heading_min_degrees up to 25° gives every
    camera a Heading."""
    tilt = st.floats(-20.0, 20.0)  # a pitch or a roll, in degrees
    cameras = draw(st.lists(st.tuples(st.floats(-180.0, 180.0), tilt, tilt, NEAR_COORDINATES),
                            min_size=1, max_size=12))
    return AgreeingCase(turns=tuple((yaw, pitch, roll) for yaw, pitch, roll, _ in cameras),
                        centres=tuple(centre for *_, centre in cameras),
                        points=tuple(draw(st.lists(NEAR_COORDINATES, min_size=1, max_size=30))),
                        quaternion=draw(rotations()), translation=draw(TRANSLATIONS),
                        settings=draw(AGREEING_SETTINGS))


# Property 4: The Level_Move keeps every distance (Req 9.1, 9.13, 9.15)

def distances(model: Model) -> np.ndarray:
    """The distance between every two of the model's camera centres and points, worked out here.
    Each centre is −Rᵀ t, with R from the image's quaternion by the quaternion formula. For a
    quaternion q that isn't a unit, that formula gives |q|² times a rotation, so a Pose with a
    scale shows up as a changed distance."""
    centres = np.array([-rotation_of(each.qvec).T @ np.array(each.tvec)
                        for each in model.images]).reshape(-1, 3)
    places = np.concatenate([centres, model.points.xyz])
    return np.linalg.norm(places[:, None, :] - places[None, :, :], axis=-1)


@given(scenes())
def test_the_level_move_keeps_every_distance(case: SceneCase) -> None:
    """Feature: cameras, Property 4: The Level_Move keeps every distance

    **Validates: Requirements 9.1, 9.13, 9.15**
    """
    model = case.model()
    levelling = level(scene_of(model, pick_numbers(len(model.images))), case.settings)
    moved = move_model(model, levelling.move)
    before = distances(model)
    assert_close(distances(moved), before, 1e-9 * (1.0 + before.max()))


# Property 5: The Level_Move's rotation is a proper rotation (Req 9.1, 9.16)

@given(scenes())
def test_the_level_moves_rotation_is_a_proper_rotation(case: SceneCase) -> None:
    """Feature: cameras, Property 5: The Level_Move's rotation is a proper rotation

    **Validates: Requirements 9.1, 9.16**
    """
    model = case.model()
    levelling = level(scene_of(model, pick_numbers(len(model.images))), case.settings)
    rotation = levelling.move.rotation
    # RᵀR = I keeps every length, so the scale is 1, and a determinant of 1 rules out a mirror.
    assert_close(rotation.T @ rotation, np.eye(3), 1e-12)
    assert np.linalg.det(rotation) == pytest.approx(1.0, abs=1e-12)
    if not levelling.levelled:  # the COLMAP_Frame is kept
        assert_kept(levelling)  # exactly the identity rotation and a zero translation


# Property 6: Cameras that agree come out level, facing +Z and centred (Req 9.2, 9.4, 9.5, 9.9,
# 9.11, 9.17, 9.18)

@given(agreeing_scenes())
def test_agreeing_scenes_come_out_level_facing_plus_z_and_centred(case: AgreeingCase) -> None:
    """Feature: cameras, Property 6: Cameras that agree come out level, facing +Z and centred

    **Validates: Requirements 9.2, 9.4, 9.5, 9.9, 9.11, 9.17, 9.18**
    """
    cameras = case.cameras()
    levelling = level(cameras, case.settings)
    kind = levelling.scene_centre  # with a point, there's always a Scene_Centre
    assert kind is not None, levelling.reason
    # Every camera has a Heading, so Pick 0 is the Heading_Camera. In the moved scene, worked out
    # again here, the Mean_Up points along UP, Pick 0's flattened View_Direction along +Z, and
    # the Scene_Centre of the kind level used lies at the origin.
    assert_level(cameras, levelling, heading=0, kind=kind)
