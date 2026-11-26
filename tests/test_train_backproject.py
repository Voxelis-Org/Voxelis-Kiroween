"""Unit tests and the property test for Property 6 of the Back_Projected_Splats:
splatpipe/train/backproject.py (depth-priors spec, Req 12).

Every input is built in the test: Sparse_Models, Scenes and Priors of a few pixels and a few
dozen points, written out by hand (Req 18.1, 18.13). Each Prior_View's Brightness_Gradient comes
from view_prior, and the taps from prior_taps at a Prior_Size equal to the Training_Size, so each
training pixel samples exactly its own stored pixel. Each oracle is worked out here and never
calls the function it checks: the pixel each point lands on from the camera's numbers, the
textured columns of a black and white image, the Candidate_Pixels listed by hand, and each
Back_Projected_Splat's pixel and depth from the test's own float64 projection, R p + t and K's
formula. The Sparse_Points' oracle is choose_points, the splat-training rule that start_points
must keep (Req 12.6, 12.8).

Property 6 draws a scaled PINHOLE camera, an Image_Pose, a pixel and a float32 inverse_depth as
the design's strategy says, and projects the point world_points gives with that same float64
projection, the rotation built from the drawn quaternion by the test's own Hamilton products
(Req 12.3, 12.10).

The settings are the test config's [train] for the fast Preset, read as the step reads them, so
max_splats is 50 and the Back_Project_Budget is ⌊50 / 3⌋ = 16, except where a test reads the real
config by path.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from splatpipe.config import REPO_ROOT, load_config
from splatpipe.train.backproject import (back_project_budget, candidate_pixels, draw_candidates,
                                         landed_pixels, start_points, world_points)
from splatpipe.train.colmap import Camera, ModelImage, SparseModel
from splatpipe.train.prior_views import prior_taps, view_prior
from splatpipe.train.priors import Prior
from splatpipe.train.scene import Scene, TrainingView
from splatpipe.train.settings import TrainSettings, read_settings
from splatpipe.train.splats import choose_points

# Read by path, so these tests still read it once CONFIG_PATH points at a test config.
REAL_CONFIG = REPO_ROOT / "config.toml"


def train_settings() -> TrainSettings:
    """The test config's [train] settings for the fast Preset, read as the step reads them."""
    return read_settings(load_config().step_settings("train", "fast"), "train", "fast")


def intrinsics(focal: float, cx: float, cy: float) -> np.ndarray:
    """K with the focal length `focal` on both axes and the principal point (cx, cy)."""
    return np.array([[focal, 0.0, cx], [0.0, focal, cy], [0.0, 0.0, 1.0]])


def pose(rotation: np.ndarray, translation: Sequence[float]) -> np.ndarray:
    """The (4, 4) world-to-camera matrix [[R, t], [0, 0, 0, 1]], built by hand."""
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = translation
    return matrix


# A half turn about the camera's z axis, R = diag(−1, −1, 1), the rotation of the quaternion
# (0, 0, 0, 1).
HALF_TURN = np.diag([-1.0, -1.0, 1.0])
IDENTITY = pose(np.eye(3), (0.0, 0.0, 0.0))


def prior_of(mask: np.ndarray, inverse_depth: float) -> Prior:
    """A Prior at the size of `mask`: `inverse_depth` and the normal (0, 0, −1), facing the camera,
    where `mask` is true, and 0s where it's false."""
    return Prior(inverse_depth=np.where(mask, inverse_depth, 0.0).astype(np.float32),
                 normal=np.where(mask[..., None], (0.0, 0.0, -1.0), 0.0).astype(np.float32),
                 mask=mask.copy())


def view(name: str, pixels: np.ndarray, matrix: np.ndarray, k: np.ndarray,
         prior: Prior | None = None, keep: np.ndarray | None = None) -> TrainingView:
    """A TrainingView of `pixels`, with view_prior's ViewPrior of `prior`, or no Prior."""
    return TrainingView(name, pixels, keep, k, matrix,
                        None if prior is None else view_prior(pixels, prior))


def project_with(point: np.ndarray, k: np.ndarray,
                 matrix: np.ndarray) -> tuple[float, float, float]:
    """The oracle's projection of the world `point` with the intrinsics `k` and the
    world-to-camera `matrix`, in float64: (x, y, z) = R p + t, then u = fx · x / z + cx and
    v = fy · y / z + cy, given as (u, v, z)."""
    rotation = matrix[:3, :3]
    translation = matrix[:3, 3]
    x, y, z = (float(sum(rotation[i, j] * point[j] for j in range(3)) + translation[i])
               for i in range(3))
    return k[0, 0] * x / z + k[0, 2], k[1, 1] * y / z + k[1, 2], z


def project(point: np.ndarray, the_view: TrainingView) -> tuple[float, float, float]:
    """project_with the view's scaled K and world-to-camera matrix."""
    return project_with(point, the_view.intrinsics, the_view.world_to_camera)


# A7: the Back_Project_Budget

@pytest.mark.parametrize(("preset", "splats", "budget"), [
    ("fast", 1_000_000, 333_333),
    ("default", 2_000_000, 500_000),
    ("quality", 3_000_000, 500_000),
])
def test_the_real_presets_budget_is_a_third_of_their_splats_up_to_backproject_max(
        preset: str, splats: int, budget: int) -> None:
    """⌊1,000,000 / 3⌋ is 333,333, and a third of 2,000,000 or 3,000,000 is above the real
    backproject_max of 500,000 (A7). A backproject_max of 0 turns the Back_Projected_Splats off."""
    real = read_settings(load_config(REAL_CONFIG).step_settings("train", preset), "train", preset)
    assert real.max_splats == splats
    got = back_project_budget(real)
    assert type(got) is int and got == budget
    assert back_project_budget(replace(real, backproject_max=0)) == 0


# Req 12.1 and O5: the landed pixels and the Candidate_Pixels

def test_landed_pixels_marks_the_pixel_a_point_lands_on_and_ignores_one_behind_or_outside(
        ) -> None:
    """On a 4 × 3 image with fx = fy = 2, (cx, cy) = (2, 1.5), R the half turn and t = (0, 0, 1),
    the world point (−0.35, −0.25, 1) sits at (0.35, 0.25, 2) in the camera, so
    u = 2 · 0.35 / 2 + 2 = 2.35 and v = 2 · 0.25 / 2 + 1.5 = 1.75: column 2, row 1 (Req 12.1, O5).
    Each other point lands on no pixel."""
    the_view = view("000000.jpg", np.zeros((3, 4, 3), np.uint8), pose(HALF_TURN, (0.0, 0.0, 1.0)),
                    intrinsics(2.0, 2.0, 1.5))
    points = np.array([
        [-0.35, -0.25, 1.0],  # lands on column 2, row 1
        [-0.25, -0.25, -2.0],  # camera (0.25, 0.25, −1): behind it, though (1.5, 1) is inside
        [-3.0, 0.0, 1.0],  # camera (3, 0, 2): u = 5, right of the image
        [-2.0, 0.0, 1.0],  # camera (2, 0, 2): u = 4, just past the last column
        [3.0, 0.0, 1.0],  # camera (−3, 0, 2): u = −1, left of the image
        [0.0, -4.0, 1.0],  # camera (0, 4, 2): v = 5.5, below the image
    ])
    landed = landed_pixels(points, the_view, 4, 3)
    expected = np.zeros((3, 4), dtype=bool)
    expected[1, 2] = True
    assert landed.dtype == bool and np.array_equal(landed, expected)


def test_candidate_pixels_leave_out_textured_pixels_pixels_with_no_prior_and_landed_pixels(
        ) -> None:
    """A 6 × 4 view, black in columns 0 to 2 and white in 3 to 5, so the gradient is half the
    step, 0.5, in columns 2 and 3 and 0 elsewhere. Its Prior has no value at (0, 0) and (3, 5),
    (1, 4) is a Masked_Pixel, and its one Observed_Point lands on (2, 1): with fx = fy = 4 and
    (cx, cy) = (3, 2), the camera point (−0.375, 0.125, 1) has u = 1.5 and v = 2.5 (Req 12.1)."""
    width, height = 6, 4
    pixels = np.zeros((height, width, 3), dtype=np.uint8)
    pixels[:, 3:] = 255
    mask = np.ones((height, width), dtype=bool)
    mask[0, 0] = mask[3, 5] = False
    keep = np.ones((height, width), dtype=bool)
    keep[1, 4] = False
    the_view = view("000000.jpg", pixels, IDENTITY, intrinsics(4.0, 3.0, 2.0),
                    prior_of(mask, 0.5), keep)
    observed = np.array([[-0.375, 0.125, 1.0]])

    got = candidate_pixels(the_view, observed, prior_taps(width, height, width, height),
                           train_settings(), width, height)

    expected = np.ones((height, width), dtype=bool)
    expected[:, 2:4] = False  # textured
    expected[0, 0] = expected[3, 5] = False  # no Prior there
    expected[1, 4] = False  # a Masked_Pixel, so not a Prior_Pixel
    expected[2, 1] = False  # the Observed_Point lands there
    assert got.dtype == bool and np.array_equal(got, expected)


def test_a_view_with_no_prior_has_no_candidate_pixel() -> None:
    the_view = view("000001.jpg", np.zeros((2, 3, 3), np.uint8), IDENTITY,
                    intrinsics(2.0, 1.5, 1.0))
    got = candidate_pixels(the_view, np.empty((0, 3)), prior_taps(3, 2, 3, 2), train_settings(),
                           3, 2)
    assert got.shape == (2, 3) and got.dtype == bool and not got.any()


# Req 12.2: the draw

def test_draw_candidates_gives_the_same_distinct_sorted_draw_for_the_same_seed() -> None:
    """100 Candidate_Pixels over three views, one of them with none, and a budget of 16. The draw
    is the design's, np.random.default_rng(seed).choice(C, B, replace=False), sorted, so the same
    seed gives the same draw and another seed another (Req 12.2)."""
    counts = [40, 0, 60]
    first = draw_candidates(counts, 16, 42)
    assert first.dtype == np.int64
    assert np.array_equal(first, draw_candidates(counts, 16, 42))
    assert np.array_equal(first, np.sort(np.random.default_rng(42).choice(100, 16, replace=False)))
    assert not np.array_equal(first, draw_candidates(counts, 16, 7))
    drawn = first.tolist()
    assert len(set(drawn)) == 16  # without replacement
    assert drawn == sorted(drawn) and 0 <= drawn[0] and drawn[-1] < 100


def test_draw_candidates_gives_every_candidate_when_there_are_fewer_than_the_budget() -> None:
    assert draw_candidates([2, 0, 3], 16, 42).tolist() == [0, 1, 2, 3, 4]
    assert draw_candidates([2, 0, 3], 5, 42).tolist() == [0, 1, 2, 3, 4]
    nothing = draw_candidates([2, 0, 3], 0, 42)
    assert nothing.dtype == np.int64 and nothing.size == 0


# Req 12.2–12.4, 12.6 and 12.8: the Start_Splats' points
#
# Three views of 3 × 2 pixels with fx = fy = 2 and (cx, cy) = (1.5, 1). 000000.jpg is at the
# world's origin with inverse_depth 0.5 everywhere but (0, 0), where its Prior has no value.
# 000001.jpg has no Prior. 000002.jpg is turned by the half turn and moved by (0.5, −0.25, 1),
# with inverse_depth 0.25 everywhere. Every pixel of both Prior_Views is a Plain_Pixel: their
# colours differ only in blue, by 1 a column and 3 a row, so OpenCV's grey values differ by 1 at
# most and the gradient is at most √(0.5² + 0.5²) / 255, far below plain_gradient.

WIDTH, HEIGHT = 3, 2
CAMERA = Camera(camera_id=1, width=WIDTH, height=HEIGHT, fx=2.0, fy=2.0, cx=1.5, cy=1.0)
K = intrinsics(2.0, 1.5, 1.0)
INVERSE_DEPTHS = {0: 0.5, 2: 0.25}  # each Prior_View's, by its place in the scene
SHIFT = (0.5, -0.25, 1.0)  # 000002.jpg's translation
# The Candidate_Pixels as (view, row, column), in view order and then row-major order. View 0
# leaves out (0, 0), with no Prior, and (1, 2), where point 1 lands: (1, 0.5, 2) gives u = 2.5
# and v = 1.5. View 2 leaves out (0, 0), where point 2 lands: (1.5, 0.25, 1) sits at (−1, −0.5, 2)
# in its camera, so u = 0.5 and v = 0.5. View 2 keeps (0, 1), where point 1 would land, and
# (1, 2), because only a Frame's own Observed_Points rule out its pixels.
CANDIDATES = [(0, 0, 1), (0, 0, 2), (0, 1, 0), (0, 1, 1),
              (2, 0, 1), (2, 0, 2), (2, 1, 0), (2, 1, 1), (2, 1, 2)]


def shaded(blue: int) -> np.ndarray:
    """A 3 × 2 image of red 120 and green 80, with blue `blue` + column + 3 × row."""
    rows, columns = np.mgrid[0:HEIGHT, 0:WIDTH]
    pixels = np.empty((HEIGHT, WIDTH, 3), dtype=np.uint8)
    pixels[..., 0], pixels[..., 1] = 120, 80
    pixels[..., 2] = blue + columns + 3 * rows
    return pixels


def scene_of(with_priors: bool) -> Scene:
    """The three views, with their Priors and the taps, or with no Prior_Set at all."""
    every = np.ones((HEIGHT, WIDTH), dtype=bool)
    no_value = every.copy()
    no_value[0, 0] = False
    first = prior_of(no_value, INVERSE_DEPTHS[0]) if with_priors else None
    third = prior_of(every, INVERSE_DEPTHS[2]) if with_priors else None
    views = (view("000000.jpg", shaded(10), IDENTITY, K, first),
             view("000001.jpg", np.full((HEIGHT, WIDTH, 3), 200, np.uint8), IDENTITY, K),
             view("000002.jpg", shaded(30), pose(HALF_TURN, SHIFT), K, third))
    taps = prior_taps(WIDTH, HEIGHT, WIDTH, HEIGHT) if with_priors else None
    return Scene(views, WIDTH, HEIGHT, extent=1.0, median_depth=2.0, near_plane=0.06, taps=taps)


def model_of(count: int) -> SparseModel:
    """A Sparse_Model of `count` points with ids 1 to `count` and colours all different. Point 1
    is 000000.jpg's one Observed_Point and point 2 is 000002.jpg's, and 000001.jpg observes none.
    The rest sit far off, observed by no image."""
    points = np.column_stack([np.arange(count, dtype=np.float64), np.zeros(count),
                              np.full(count, 10.0)])
    points[0] = (1.0, 0.5, 2.0)
    points[1] = (1.5, 0.25, 1.0)
    colours = np.column_stack([np.arange(count), (3 * np.arange(count)) % 256,
                               np.full(count, 7)]).astype(np.uint8)
    images = (ModelImage(1, np.array([1.0, 0.0, 0.0, 0.0]), np.zeros(3), 1, "000000.jpg",
                         np.array([1], dtype=np.uint64)),
              ModelImage(2, np.array([1.0, 0.0, 0.0, 0.0]), np.zeros(3), 1, "000001.jpg",
                         np.empty(0, dtype=np.uint64)),
              ModelImage(3, np.array([0.0, 0.0, 0.0, 1.0]), np.array(SHIFT), 1, "000002.jpg",
                         np.array([2], dtype=np.uint64)))
    return SparseModel(CAMERA, images, points, colours, np.arange(1, count + 1, dtype=np.uint64))


def candidate_of(point: np.ndarray, scene: Scene) -> tuple[int, int, int]:
    """The one Candidate_Pixel whose centre `point` projects onto in its view, at the camera-space
    z of 1 / that view's inverse_depth, each within a relative error of 10⁻⁹ (Req 12.3)."""
    found = [(index, row, column) for index, row, column in CANDIDATES
             if np.allclose(project(point, scene.views[index]),
                            (column + 0.5, row + 0.5, 1 / INVERSE_DEPTHS[index]),
                            rtol=1e-9, atol=0)]
    assert len(found) == 1, f"{point} is the back-projection of {found}"
    return found[0]


@pytest.mark.parametrize("count", [20, 60])
def test_with_no_prior_set_the_start_points_are_choose_points_for_max_splats(count: int) -> None:
    """With no Prior_Set the Start_Splats are the splat-training spec's: every point when there
    are at most max_splats, 50, and choose_points' subset of 50 when there are 60 (Req 12.8)."""
    settings = train_settings()
    model = model_of(count)
    got = start_points(model, scene_of(with_priors=False), settings)
    points, colours = choose_points(model.points, model.colours, settings.max_splats,
                                    settings.seed)
    assert np.array_equal(got.points, points) and np.array_equal(got.colours, colours)
    assert (got.sparse, got.back_projected) == (min(count, 50), 0)


@pytest.mark.parametrize(("count", "sparse"), [(20, 20), (60, 41)])
def test_with_a_prior_set_every_candidate_follows_the_sparse_points_with_its_colour(
        count: int, sparse: int) -> None:
    """The budget, 16, is above the 9 Candidate_Pixels, so B = 9 and every one is drawn, in view
    and pixel order, after the Sparse_Points. Those are at most 50 − 9 = 41: every one of 20, or
    choose_points' subset of 41 of 60 (Req 12.2, 12.6). Each Back_Projected_Splat has its pixel's
    8-bit colour (Req 12.4)."""
    settings = train_settings()
    scene = scene_of(with_priors=True)
    assert all((the_view.prior.gradient < settings.plain_gradient).all()
               for the_view in scene.views if the_view.prior is not None)
    model = model_of(count)

    got = start_points(model, scene, settings)

    assert (got.sparse, got.back_projected) == (sparse, 9)
    assert got.points.dtype == np.float64 and got.colours.dtype == np.uint8
    assert len(got.points) == len(got.colours) == sparse + 9 <= settings.max_splats
    points, colours = choose_points(model.points, model.colours, settings.max_splats - 9,
                                    settings.seed)
    assert np.array_equal(got.points[:sparse], points)
    assert np.array_equal(got.colours[:sparse], colours)
    drawn = [candidate_of(point, scene) for point in got.points[sparse:]]
    assert drawn == CANDIDATES
    for colour, (index, row, column) in zip(got.colours[sparse:], drawn, strict=True):
        assert np.array_equal(colour, scene.views[index].pixels[row, column])


def test_with_a_budget_below_the_candidates_it_draws_that_many_distinct_ones_in_order() -> None:
    """A backproject_max of 4 makes the budget 4, below the 9 Candidate_Pixels, so B = 4 and the
    Sparse_Points are choose_points' 46 of 60 (Req 12.2, 12.6). The 4 are distinct
    Candidate_Pixels, in view and pixel order, each with its pixel's colour (Req 12.4)."""
    settings = replace(train_settings(), backproject_max=4)
    scene = scene_of(with_priors=True)
    model = model_of(60)

    got = start_points(model, scene, settings)

    assert (got.sparse, got.back_projected) == (46, 4)
    points, colours = choose_points(model.points, model.colours, 46, settings.seed)
    assert np.array_equal(got.points[:46], points) and np.array_equal(got.colours[:46], colours)
    drawn = [candidate_of(point, scene) for point in got.points[46:]]
    assert len(set(drawn)) == 4
    assert drawn == sorted(drawn, key=CANDIDATES.index)
    for colour, (index, row, column) in zip(got.colours[46:], drawn, strict=True):
        assert np.array_equal(colour, scene.views[index].pixels[row, column])


# --- Property 6: projecting a Back_Projected_Splat gives back its pixel and its depth -----------

# Req 12.10's bound: the relative error of each of u, v and z, against its own value.
ROUND_TRIP_BOUND = 1e-9

# The design's strategy: sizes from 1 to 2,000, focal lengths from 200 to 2,000, quaternion
# components from −1 to 1 with a length of at least 0.1, translation components from −4 to 4, and
# a float32 inverse_depth from 0.125 to 2, so z = 1 / inverse_depth is from 0.5 to 8.
SIZES = st.integers(min_value=1, max_value=2_000)
FOCALS = st.floats(min_value=200.0, max_value=2_000.0)
QUATERNIONS = st.tuples(*[st.floats(min_value=-1.0, max_value=1.0)] * 4).filter(
    lambda quaternion: math.hypot(*quaternion) >= 0.1)
TRANSLATIONS = st.tuples(*[st.floats(min_value=-4.0, max_value=4.0)] * 3)
FLOAT32_INVERSE_DEPTHS = st.floats(min_value=0.125, max_value=2.0, width=32)


@dataclass(frozen=True)
class DrawnPixel:
    """One drawn scaled PINHOLE camera, Image_Pose, training pixel and scaled-up inverse_depth."""

    k: np.ndarray  # (3, 3) float64: fx and fy from 200 to 2,000, (cx, cy) inside the image
    quaternion: tuple[float, float, float, float]  # w x y z, of length at least 0.1
    translation: tuple[float, float, float]
    column: int  # x, from 0 to width − 1
    row: int  # y, from 0 to height − 1
    inverse_depth: float  # exactly a float32 value


@st.composite
def drawn_pixels(draw: st.DrawFn) -> DrawnPixel:
    """A camera at a Training_Size from 1 × 1 to 2,000 × 2,000 with cx from 0 to the width and cy
    from 0 to the height, a pose from QUATERNIONS and TRANSLATIONS, any pixel of the image, and
    an inverse_depth from FLOAT32_INVERSE_DEPTHS."""
    width, height = draw(SIZES), draw(SIZES)
    fx, fy = draw(FOCALS), draw(FOCALS)
    cx = draw(st.floats(min_value=0.0, max_value=float(width)))
    cy = draw(st.floats(min_value=0.0, max_value=float(height)))
    k = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
    return DrawnPixel(k=k, quaternion=draw(QUATERNIONS), translation=draw(TRANSLATIONS),
                      column=draw(st.integers(min_value=0, max_value=width - 1)),
                      row=draw(st.integers(min_value=0, max_value=height - 1)),
                      inverse_depth=draw(FLOAT32_INVERSE_DEPTHS))


def hamilton(a: Sequence[float], b: Sequence[float]) -> tuple[float, float, float, float]:
    """The Hamilton product a ⊗ b of the quaternions a and b, each (w, x, y, z)."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw)


def rotation_of(quaternion: Sequence[float]) -> np.ndarray:
    """The (3, 3) float64 rotation R of the unit quaternion q = `quaternion` / its length, worked
    out with Python floats: column j of R is the axis e_j turned by q, the vector part of
    q ⊗ (0, e_j) ⊗ q̄. So the quaternion (0, 0, 0, 1) gives HALF_TURN."""
    length = math.hypot(*quaternion)
    q = tuple(value / length for value in quaternion)
    conjugate = (q[0], -q[1], -q[2], -q[3])
    rotation = np.empty((3, 3))
    for j in range(3):
        axis = (0.0, *(1.0 if i == j else 0.0 for i in range(3)))
        rotation[:, j] = hamilton(hamilton(q, axis), conjugate)[1:]
    return rotation


def test_rotation_of_turns_the_axes_as_the_quaternion_says() -> None:
    """The oracle's own rotation: (0, 0, 0, 1) is the half turn about z, (2, 0, 0, 0) the
    identity once scaled to length 1, and (cos 45°, 0, 0, sin 45°) the quarter turn about z,
    which takes x to y."""
    assert np.allclose(rotation_of((0.0, 0.0, 0.0, 1.0)), HALF_TURN, rtol=0, atol=1e-15)
    assert np.allclose(rotation_of((2.0, 0.0, 0.0, 0.0)), np.eye(3), rtol=0, atol=1e-15)
    half = math.sqrt(0.5)
    quarter = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    assert np.allclose(rotation_of((half, 0.0, 0.0, half)), quarter, rtol=0, atol=1e-15)


@settings(max_examples=100, deadline=None)
@given(drawn=drawn_pixels())
def test_projecting_a_back_projected_splat_gives_back_its_pixel_centre_and_its_depth(
        drawn: DrawnPixel) -> None:
    """Feature: depth-priors, Property 6: Projecting a Back_Projected_Splat gives back its pixel
    and its depth

    world_points places the drawn pixel's centre at the camera-space z of 1 / the drawn float32
    inverse_depth, through the drawn K and the pose [[R, t], [0, 0, 0, 1]] with R from
    rotation_of. The test's own float64 projection of that point with the same K and pose,
    R p + t and K's formula, gives back (x + 0.5, y + 0.5) and that z, each within a relative
    error of 10⁻⁹ of its own value, so the bound holds at the smallest pixel centre, 0.5, too
    (Req 12.3, 12.10).

    **Validates: Requirements 12.10, 12.3**
    """
    matrix = pose(rotation_of(drawn.quaternion), drawn.translation)
    got = world_points(np.array([drawn.column], dtype=np.int64),
                       np.array([drawn.row], dtype=np.int64),
                       np.array([drawn.inverse_depth], dtype=np.float32), drawn.k, matrix)
    assert got.shape == (1, 3) and got.dtype == np.float64
    u, v, z = project_with(got[0], drawn.k, matrix)
    for label, value, wanted in (("u", u, drawn.column + 0.5), ("v", v, drawn.row + 0.5),
                                 ("z", z, 1.0 / drawn.inverse_depth)):
        assert abs(value - wanted) <= ROUND_TRIP_BOUND * wanted, (
            f"{label} is {value!r}, but it should be {wanted!r} within a relative error of 1e-9")
