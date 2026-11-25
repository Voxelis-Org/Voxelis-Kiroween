"""The Levelling (Req 9): the one rigid move that makes the Undistorted_Model level, faces it
along +Z and centres it, with the four settings it takes.

LevelSettings holds the four settings of [cameras] that the Levelling uses. settings.py reads and
checks them, and keeps them in CamerasSettings.level.

The conventions are COLMAP's. A Pose is the cam_from_world rotation R, from the unit quaternion
q = (w, x, y, z) in Hamilton's convention, and the translation t: a world point x is at R x + t in
the camera, which looks along +z with x to the right and y down. So each Placed_Frame i has the
Up_Vector u_i = Rᵢᵀ (0, −1, 0), minus the second row of Rᵢ, the View_Direction d_i = Rᵢᵀ (0, 0, 1),
the third row of Rᵢ, and the centre C_i = −Rᵢᵀ tᵢ.

level works out the Level_Move x' = R x + T from a Scene in eight steps:

1. Finite input: every rotation, centre and point is a finite number.
2. The Mean_Up m = (1/n) Σ u_i must be at least up_agreement long (Req 9.2, 9.3).
3. R_a = rotation_onto(m / |m|, WORLD_UP) turns the Mean_Up to the World_Up (Req 9.4).
4. The Heading_Camera is the first Placed_Frame, by Pick_Number, whose turned View_Direction is
   at least heading_min_degrees from straight up and from straight down (Req 9.6, 9.7).
5. R_t = heading_turn(...) turns its Heading to +Z about the World_Up, and R = R_t R_a (Req 9.5).
6. The Scene_Centre s is the Axis_Point, rotated, when it's well defined, and otherwise the
   Point_Median along the levelled axes (Req 9.8 to 9.10, A6).
7. T = −s takes the Scene_Centre to the origin (Req 9.11).
8. Finite output: R and T are finite numbers.

When a step can't go on, the Levelling keeps the COLMAP_Frame: exactly the identity move, with the
warning of Req 9.3, 9.7 or 9.10. Any exception, and any value that isn't a finite number, keeps it
with the Req 9.12 warning, which names the error's type and message, so level never raises (A14).
move_model applies the whole Level_Move to every Pose and every point at once, with scale 1, and
keeps everything else as it is (Req 9.1, 9.13, 10.6 to 10.8).

This module is pure: it imports only the standard library, numpy and the step's model.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from fractions import Fraction
from typing import Literal

import numpy as np

from splatpipe.cameras.model import Model

WORLD_UP = np.array([0.0, -1.0, 0.0])  # Fixed_Constant: the World_Up
FORWARD = np.array([0.0, 0.0, 1.0])  # Fixed_Constant: where the Heading points, +Z
WORLD_UP.setflags(write=False)
FORWARD.setflags(write=False)
# P = diag(1, −1, −1), the half turn about x, which takes the World_Up to minus itself.
_HALF_TURN = np.diag([1.0, -1.0, -1.0])
_HALF_TURN.setflags(write=False)

_Kind = Literal["axis_point", "point_median"]  # the kinds of Scene_Centre

# The warnings of a Levelling that keeps the COLMAP_Frame (Req 9.3, 9.7, 9.10, 9.12).
_NO_AGREEMENT = ("The cameras don't agree on which way is up: their mean up vector is {length} "
                 "long, below up_agreement = {up_agreement} in [cameras]. So the scene was left "
                 "in COLMAP's frame, which may be tilted.")
_NO_HEADING = ("Every placed camera looks within heading_min_degrees = {degrees}° of straight up "
               "or down, so the scene couldn't be turned to face +Z and was left in COLMAP's "
               "frame, which may be tilted.")
_NO_CENTRE = ("The scene's centre couldn't be found: the cameras' view lines don't meet in a well "
              "defined point and the model has no 3D points. So the scene was left in COLMAP's "
              "frame, which may be tilted.")
_FAILED = ("Levelling the scene failed ({kind}: {message}), so the scene was left in COLMAP's "
           "frame, which may be tilted.")


@dataclass(frozen=True)
class LevelSettings:
    """The Levelling's four settings from [cameras], each as Config gave it."""

    up_agreement: float  # the shortest Mean_Up at which the cameras agree on which way is up
    heading_min_degrees: float  # a camera within this of straight up or down has no Heading
    axis_point_min_eigen_share: float  # the smallest eigenvalue share of a usable Axis_Point
    axis_point_min_front_share: float  # the share of cameras a usable Axis_Point is in front of


@dataclass(frozen=True)
class Scene:
    """What the Levelling works from: the Placed_Frames, in Pick_Number order, and the points,
    all in the COLMAP_Frame."""

    names: tuple[str, ...]  # the Placed_Frames' names, in Pick_Number order
    rotations: np.ndarray  # (n, 3, 3), each camera's R
    centres: np.ndarray  # (n, 3), each camera's -R^T t
    points: np.ndarray  # (m, 3)


@dataclass(frozen=True)
class LevelMove:
    """The Level_Move x' = R x + T: a rotation, then a translation, with scale 1."""

    rotation: np.ndarray  # (3, 3), R
    translation: np.ndarray  # (3,), T


@dataclass(frozen=True)
class Levelling:
    """What level gives: the Level_Move, and what the Levelling found on the way.

    When the COLMAP_Frame is kept, the move is exactly the identity, reason holds the warning and
    scene_centre is None. mean_up_length and heading_camera still hold what the Levelling found
    before the step that kept the frame, and are None when it hadn't found them yet.
    """

    move: LevelMove  # the identity when the COLMAP_Frame is kept
    levelled: bool  # whether the Levelling moved the scene
    reason: str | None  # the warning, when the COLMAP_Frame is kept
    mean_up_length: float | None  # None when the input wasn't finite
    scene_centre: Literal["axis_point", "point_median"] | None  # None when the frame is kept
    heading_camera: str | None  # the Heading_Camera's name


def quaternion_product(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Hamilton's product p ⊗ q of quaternions (w, x, y, z) along the last axis, so the rotation
    of p ⊗ q is the rotation of p times the rotation of q. The shapes broadcast."""
    pw, px, py, pz = np.moveaxis(np.asarray(p, dtype=np.float64), -1, 0)
    qw, qx, qy, qz = np.moveaxis(np.asarray(q, dtype=np.float64), -1, 0)
    return np.stack([pw * qw - px * qx - py * qy - pz * qz,
                     pw * qx + px * qw + py * qz - pz * qy,
                     pw * qy - px * qz + py * qw + pz * qx,
                     pw * qz + px * qy - py * qx + pz * qw], axis=-1)


def quaternion_conjugate(q: np.ndarray) -> np.ndarray:
    """q̄ = (w, −x, −y, −z) of each quaternion along the last axis, which for a unit quaternion is
    the opposite rotation."""
    return np.asarray(q, dtype=np.float64) * np.array([1.0, -1.0, -1.0, -1.0])


def rotation_from_quaternion(q: np.ndarray) -> np.ndarray:
    """The rotation matrix of each unit quaternion (w, x, y, z) along the last axis, in
    Hamilton's convention, as COLMAP builds it: (..., 4) gives (..., 3, 3)."""
    w, x, y, z = np.moveaxis(np.asarray(q, dtype=np.float64), -1, 0)
    rows = ((1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)),
            (2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)),
            (2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)))
    return np.stack([np.stack(row, axis=-1) for row in rows], axis=-2)


def quaternion_from_rotation(rotation: np.ndarray) -> np.ndarray:
    """The unit quaternion (w, x, y, z) of a 3 × 3 rotation matrix, by Shepperd's method.

    It starts from the largest of the trace and the three diagonal terms, which belongs to the
    largest part of q, so s, four times that part, is at least 2 and nothing cancels. q and −q
    are the same rotation, and this gives the one whose largest part is positive.
    """
    r = np.asarray(rotation, dtype=np.float64)
    trace = r[0, 0] + r[1, 1] + r[2, 2]
    start = int(np.argmax([trace, r[0, 0], r[1, 1], r[2, 2]]))
    if start == 0:
        s = 2.0 * math.sqrt(1.0 + trace)  # 4w
        q = (s / 4, (r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s)
    elif start == 1:
        s = 2.0 * math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2])  # 4x
        q = ((r[2, 1] - r[1, 2]) / s, s / 4, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s)
    elif start == 2:
        s = 2.0 * math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2])  # 4y
        q = ((r[0, 2] - r[2, 0]) / s, (r[0, 1] + r[1, 0]) / s, s / 4, (r[1, 2] + r[2, 1]) / s)
    else:
        s = 2.0 * math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1])  # 4z
        q = ((r[1, 0] - r[0, 1]) / s, (r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, s / 4)
    return _unit(np.array(q, dtype=np.float64))


def rotation_onto(a: np.ndarray, g: np.ndarray) -> np.ndarray:
    """A rotation that takes the unit vector `a` to the unit vector `g` (Req 9.4).

    When a · g ≥ 0, it's I + K + K² / (1 + c), with c = a · g and K the cross-product matrix of
    v = a × g, so 1 + c ≥ 1 and the division is safe, and a = g gives I. When a · g < 0, the half
    turn about x, P = diag(1, −1, −1), takes g to −g, so (P a) · g = −(a · g) > 0 and the
    rotation is rotation_onto(P a, g) P, and a = −g gives P. That needs a `g` with no x part, like
    the World_Up.
    """
    a = np.asarray(a, dtype=np.float64)
    g = np.asarray(g, dtype=np.float64)
    if a @ g < 0:
        return _towards(_HALF_TURN @ a, g) @ _HALF_TURN
    return _towards(a, g)


def heading_turn(view: np.ndarray) -> np.ndarray:
    """The turn about the y axis that takes the levelled View_Direction `view` to face +Z
    (Req 9.5).

    With x and z the first and third parts of `view` and r = √(x² + z²), it's
    [[z/r, 0, −x/r], [0, 1, 0], [x/r, 0, z/r]], which keeps the World_Up and takes (x, ·, z) to
    (0, ·, r). For a view with a Heading, r is at least sin(heading_min_degrees) > 0.
    """
    x, z = float(view[0]), float(view[2])
    r = math.hypot(x, z)
    return np.array([[z / r, 0.0, -x / r],
                     [0.0, 1.0, 0.0],
                     [x / r, 0.0, z / r]])


def axis_point(views: np.ndarray, centres: np.ndarray, *, min_eigen_share: float,
               min_front_share: float) -> np.ndarray | None:
    """The Axis_Point of the cameras with these View_Directions and centres, in their own frame,
    when it's well defined, and otherwise None (Req 9.8).

    The Axis_Point p has the smallest sum of squared distances to the Camera_Axes: it solves
    A p = b, with A = Σ (I − d dᵀ) and b = Σ (I − d dᵀ) C. It's well defined when A's smallest
    eigenvalue is at least `min_eigen_share` times its largest, and p lies in front of, with
    (p − C) · d > 0, at least `min_front_share` of the cameras, compared as an exact fraction of
    the setting (C7). The eigenvalue test comes first, so a singular A, like that of parallel
    cameras or of one camera, never reaches numpy.linalg.solve.
    """
    projectors = np.eye(3) - views[:, :, None] * views[:, None, :]  # I − d dᵀ for each camera
    a = projectors.sum(axis=0)
    b = np.einsum("nij,nj->i", projectors, centres)
    eigenvalues = np.linalg.eigvalsh(a)  # in increasing order
    if not eigenvalues[0] >= min_eigen_share * eigenvalues[-1]:
        return None
    point = np.linalg.solve(a, b)
    front = int(np.count_nonzero(np.einsum("ni,ni->n", point - centres, views) > 0))
    return point if front >= Fraction(min_front_share) * len(views) else None


def point_median(points: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    """The Point_Median along the levelled axes: the median of the points' x, of their y and of
    their z, each taken separately, after `rotation` (A6). `points` holds at least one point."""
    return np.median(points @ rotation.T, axis=0)


def level(scene: Scene, settings: LevelSettings) -> Levelling:
    """The Levelling of `scene`: the Level_Move that makes it level, facing +Z and centred, or
    exactly the identity with the warning of Req 9.3, 9.7, 9.10 or 9.12 when it keeps the
    COLMAP_Frame.

    It never raises. Any exception, and any value that isn't a finite number, keeps the
    COLMAP_Frame with the Req 9.12 warning, which names the error's type and message (A14).
    """
    found = _Found()
    try:
        # numpy stays quiet about values that aren't finite, because the checks in _level find
        # them and keep the COLMAP_Frame.
        with np.errstate(all="ignore"):
            return _level(scene, settings, found)
    except Exception as error:  # the Levelling never fails the Job (Req 9.12, 9.14)
        return found.kept(_FAILED.format(kind=type(error).__name__, message=error))


def scene_of(model: Model, numbers: Mapping[str, int]) -> Scene:
    """The Scene of the Undistorted_Model `model`: its Placed_Frames in Pick_Number order, by
    `numbers`, each with its Pose's rotation R and its centre −Rᵀt, and its points.

    Each quaternion is made a unit quaternion first. A Pose that isn't finite, or a quaternion of
    zeros, gives values that aren't finite, which level finds (Req 9.12).
    """
    images = sorted(model.images, key=lambda image: numbers[image.name])
    qvecs = np.array([image.qvec for image in images], dtype=np.float64).reshape(-1, 4)
    tvecs = np.array([image.tvec for image in images], dtype=np.float64).reshape(-1, 3)
    with np.errstate(all="ignore"):
        rotations = rotation_from_quaternion(_unit(qvecs))
        centres = -np.einsum("nji,nj->ni", rotations, tvecs)  # −Rᵢᵀ tᵢ
    return Scene(names=tuple(image.name for image in images), rotations=rotations,
                 centres=centres,
                 points=np.array(model.points.xyz, dtype=np.float64).reshape(-1, 3))


def move_model(model: Model, move: LevelMove) -> Model:
    """`model` moved by the whole Level_Move x' = R x + T, every Pose and every point at once,
    with scale 1 (Req 9.1, 9.13).

    With q_R the unit quaternion of R by Shepperd's method, each image gets
    q' = normalise(q ⊗ q̄_R) and t' = t − R' T, with R' the rotation of q', so its centre moves to
    R C + T. Each point moves to R X + T. The camera, every image's id, camera id, name and 2D
    points, and every point's id, colour, error and track stay as they are (Req 10.6 to 10.8).
    """
    rotation = np.asarray(move.rotation, dtype=np.float64)
    translation = np.asarray(move.translation, dtype=np.float64)
    back = quaternion_conjugate(quaternion_from_rotation(rotation))  # q̄_R
    qvecs = np.array([image.qvec for image in model.images], dtype=np.float64).reshape(-1, 4)
    tvecs = np.array([image.tvec for image in model.images], dtype=np.float64).reshape(-1, 3)
    moved_q = _unit(quaternion_product(qvecs, back))
    moved_t = tvecs - rotation_from_quaternion(moved_q) @ translation
    images = tuple(replace(image, qvec=tuple(q), tvec=tuple(t))
                   for image, q, t in zip(model.images, moved_q.tolist(), moved_t.tolist()))
    xyz = np.asarray(model.points.xyz, dtype=np.float64) @ rotation.T + translation
    return Model(cameras=model.cameras, images=images, points=replace(model.points, xyz=xyz))


@dataclass
class _Found:
    """What level found before a step kept the COLMAP_Frame, which the kept Levelling gives."""

    mean_up_length: float | None = None
    heading_camera: str | None = None

    def kept(self, reason: str) -> Levelling:
        """The Levelling that keeps the COLMAP_Frame for `reason`: exactly the identity move."""
        return Levelling(move=LevelMove(rotation=np.eye(3), translation=np.zeros(3)),
                         levelled=False, reason=reason, mean_up_length=self.mean_up_length,
                         scene_centre=None, heading_camera=self.heading_camera)


def _level(scene: Scene, settings: LevelSettings, found: _Found) -> Levelling:
    """The eight steps of the Levelling, which note in `found` what they find on the way. A value
    that isn't finite raises ValueError, which level turns into the Req 9.12 warning."""
    _refuse_input_not_finite(scene)  # 1
    ups = -scene.rotations[:, 1, :]  # u_i = Rᵢᵀ (0, −1, 0), minus the second row of Rᵢ
    mean_up = ups.sum(axis=0) / len(ups)  # 2 (Req 9.2, 9.3)
    length = float(np.linalg.norm(mean_up))
    if not math.isfinite(length):
        raise ValueError("the length of the cameras' mean up vector isn't a finite number")
    found.mean_up_length = length
    if length < settings.up_agreement:
        return found.kept(_NO_AGREEMENT.format(length=length,
                                               up_agreement=settings.up_agreement))
    to_up = rotation_onto(mean_up / length, WORLD_UP)  # 3: R_a (Req 9.4)
    views = scene.rotations[:, 2, :]  # d_i = Rᵢᵀ (0, 0, 1), the third row of Rᵢ
    turned = views @ to_up.T  # d'_i = R_a d_i
    heading = _heading_camera(turned, settings.heading_min_degrees)  # 4 (Req 9.6, 9.7)
    if heading is None:
        return found.kept(_NO_HEADING.format(degrees=settings.heading_min_degrees))
    found.heading_camera = scene.names[heading]
    rotation = heading_turn(turned[heading]) @ to_up  # 5: R = R_t R_a (Req 9.5)
    centre = _scene_centre(scene, views, rotation, settings)  # 6 (Req 9.8 to 9.10)
    if centre is None:
        return found.kept(_NO_CENTRE)
    kind, scene_centre = centre
    translation = -scene_centre  # 7: T = −s (Req 9.11)
    if not (np.isfinite(rotation).all() and np.isfinite(translation).all()):  # 8
        raise ValueError("the move that levels the scene holds a value that isn't a finite "
                         "number")
    return Levelling(move=LevelMove(rotation=rotation, translation=translation), levelled=True,
                     reason=None, mean_up_length=length, scene_centre=kind,
                     heading_camera=scene.names[heading])


def _refuse_input_not_finite(scene: Scene) -> None:
    """Raise ValueError for the first camera, in Pick_Number order, or the first point that holds
    a value that isn't a finite number."""
    cameras = np.isfinite(scene.rotations).all(axis=(1, 2)) & np.isfinite(scene.centres).all(axis=1)
    broken = np.flatnonzero(~cameras)
    if broken.size:
        raise ValueError(f"the pose of {scene.names[broken[0]]} holds a value that isn't a finite "
                         "number")
    broken = np.flatnonzero(~np.isfinite(scene.points).all(axis=1))
    if broken.size:
        raise ValueError(f"3D point {broken[0] + 1} of {len(scene.points)} holds a value that "
                         "isn't a finite number")


def _heading_camera(turned: np.ndarray, degrees: float) -> int | None:
    """The place of the first turned View_Direction d', in Pick_Number order, with a Heading:
    |d' · g| ≤ cos(`degrees`), so at least `degrees` from straight up and from straight down
    (Req 9.6). None when none has one (Req 9.7)."""
    has_heading = np.abs(turned @ WORLD_UP) <= math.cos(math.radians(degrees))
    places = np.flatnonzero(has_heading)
    return int(places[0]) if places.size else None


def _scene_centre(scene: Scene, views: np.ndarray, rotation: np.ndarray,
                  settings: LevelSettings) -> tuple[_Kind, np.ndarray] | None:
    """The Scene_Centre after `rotation`, with its kind: the Axis_Point, rotated, when it's well
    defined, else the Point_Median along the levelled axes (A6), else None, when the scene has no
    points (Req 9.8 to 9.10)."""
    point = axis_point(views, scene.centres, min_eigen_share=settings.axis_point_min_eigen_share,
                       min_front_share=settings.axis_point_min_front_share)
    if point is not None:
        return "axis_point", rotation @ point
    if len(scene.points):
        return "point_median", point_median(scene.points, rotation)
    return None


def _towards(a: np.ndarray, g: np.ndarray) -> np.ndarray:
    """I + K + K² / (1 + c): the rotation that takes the unit vector `a` to the unit vector `g`
    when c = a · g ≥ 0, with K the cross-product matrix of v = a × g."""
    v = np.cross(a, g)
    k = np.array([[0.0, -v[2], v[1]],
                  [v[2], 0.0, -v[0]],
                  [-v[1], v[0], 0.0]])
    return np.eye(3) + k + k @ k / (1.0 + a @ g)


def _unit(vectors: np.ndarray) -> np.ndarray:
    """Each vector along the last axis, divided by its length."""
    return vectors / np.linalg.norm(vectors, axis=-1, keepdims=True)
