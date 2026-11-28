"""Tests for the Walls_Step's geometry: splatpipe/walls/geometry.py (walls spec, Req 6, 9.3, 14.2).

The unit tests check each function on hand cases whose answers are worked out here, in the test,
never by calling the function being checked:

- apply_change, depth_factor, compose and changed_normal on small numbers, including a change
  whose new inverse depth is 0, negative or infinite, which gives that pixel no depth and no
  normal, a normal facing away from the camera, which is turned round, and a raw normal of
  length 0, which gives none (Req 6.1 to 6.5).
- normalised_coords at a grid half the camera's size, whose pixels' rays are the mean of the rays
  of the 2 x 2 full-size pixels each one covers, and median_inverse_depth, which reads only the
  Covered_Pixels.
- warp_maps on tests/camera_files.py's plane scene, whose oracle finds each pixel's true depth on
  the plane z = PLANE_Z and projects the point into another camera with its own float64 code, and
  whose keypoints, worked out by camera_files' own quaternion turn, are where every camera sees
  each plane point (Req 9.3).
- zbuffer_project on a handful of points whose landing pixels are worked out by the Depth_Step's
  pixel rule here (Req 14.2).

The property tests, with Hypothesis, check rules that hold for every input:

- Property 1: a change keeps every pixel on its own ray, in front of the camera (Req 6.7).
- Property 2: a change keeps a plane flat and carries its normal (Req 6.8, 6.4).
- Property 3: the Identity_Change gives the depths and normals back (Req 6.9, 6.2).
- Property 4: composing two changes equals applying them in turn (Req 6.10, 6.5).

NaN marks a pixel with no value. Everything runs on the CPU on a few dozen pixels (Req 22.1).
"""

from __future__ import annotations

import math
import struct

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from splatpipe.train.colmap import Camera
from splatpipe.train.scene import scaled_intrinsics, world_to_camera
from splatpipe.walls.geometry import (apply_change, changed_normal, compose, depth_factor,
                                      median_inverse_depth, normalised_coords, warp_maps,
                                      zbuffer_project)

from camera_files import PLANE_CAMERA, PLANE_GRID, PLANE_IMAGES, PLANE_Z

IDENTITY = (1.0, 0.0, 0.0, 0.0)

# The plane scene's Shared_Camera, 16 x 12 pixels, as the Model_Reader gives it.
PLANE = Camera(camera_id=PLANE_CAMERA.camera_id, width=PLANE_CAMERA.width,
               height=PLANE_CAMERA.height, fx=PLANE_CAMERA.fx, fy=PLANE_CAMERA.fy,
               cx=PLANE_CAMERA.cx, cy=PLANE_CAMERA.cy)

# A camera whose principal point is off the centre and whose fx and fy differ, so a mix-up of
# x and y, or of a pixel's corner and its centre, shows.
SKEWED = Camera(camera_id=1, width=16, height=12, fx=20.0, fy=18.0, cx=7.5, cy=6.25)

# One pixel's numbers for the hand cases: depth z = 2, so inverse depth 0.5, at (x, y) = (0.5, 0),
# whose Ray is r = (0.5, 0, 1), with m = 0.5 and the change (1, 0.25, 0, 0.5).
# b·x + c·y + d = 0.125 + 0 + 0.5 = 0.625.
# New inverse depth: 1 × 0.5 + 0.5 × 0.625 = 0.8125.
# Depth_Factor: 0.5 / 0.8125 = 1 / (1 + 0.5 × 2 × 0.625) = 1 / 1.625.
HAND_Z, HAND_X, HAND_Y, HAND_M = 2.0, 0.5, 0.0, 0.5
HAND_CHANGE = (1.0, 0.25, 0.0, 0.5)

# Three changes that leave the hand pixel (z = 2, m = 1, x = y = 0) with no new depth, each by its
# d alone: a new inverse depth of 0.5 + 1 × d.
NO_DEPTH_CHANGES = {
    "zero": ((1.0, 0.0, 0.0, -0.5), 1.0),         # 0.5 − 0.5 = 0
    "negative": ((1.0, 0.0, 0.0, -1.0), 1.0),     # 0.5 − 1 = −0.5
    "infinite": ((1.0, 0.0, 0.0, 1e300), 1e300),  # 0.5 + 1e300 × 1e300 overflows to +inf
}


def _array(*values: float) -> np.ndarray:
    return np.array(values, dtype=np.float64)


# The bas-relief change (Req 6.1 to 6.5)

def test_apply_change_on_a_hand_case() -> None:
    new = apply_change(_array(1 / HAND_Z), HAND_CHANGE, HAND_M, _array(HAND_X), _array(HAND_Y))
    assert new.dtype == np.float64
    np.testing.assert_allclose(new, [0.8125], rtol=1e-15)


def test_apply_change_with_every_part_of_the_change() -> None:
    # 2 × 0.5 + 0.4 × (0.1 × 1 − 0.2 × 0.5 + 0.3) = 1 + 0.4 × 0.3 = 1.12, and at the second pixel
    # 2 × 0.25 + 0.4 × (0.1 × (−1) − 0.2 × 2 + 0.3) = 0.5 + 0.4 × (−0.2) = 0.42.
    new = apply_change(_array(0.5, 0.25), (2.0, 0.1, -0.2, 0.3), 0.4, _array(1.0, -1.0),
                       _array(0.5, 2.0))
    np.testing.assert_allclose(new, [1.12, 0.42], rtol=1e-14)


def test_the_identity_change_gives_back_every_inverse_depth_bit_for_bit() -> None:
    inverse = np.array([[0.1, 1 / 3, 7.25e-5], [np.pi, 2.0 ** -40, 123456.789]])
    x, y = normalised_coords(SKEWED, 3, 2)
    new = apply_change(inverse, IDENTITY, 0.37, x, y)
    assert np.array_equal(new, inverse)


@pytest.mark.parametrize("case", sorted(NO_DEPTH_CHANGES))
def test_a_change_with_no_new_depth_gives_no_depth_no_factor_and_no_normal(case: str) -> None:
    change, m = NO_DEPTH_CHANGES[case]
    z, x, y = _array(2.0), _array(0.0), _array(0.0)
    assert np.isnan(apply_change(1 / z, change, m, x, y)).all()
    assert np.isnan(depth_factor(z, change, m, x, y)).all()
    normal = changed_normal(np.array([[0.0, 0.0, -1.0]]), z, x, y, change, m)
    assert normal.shape == (1, 3)
    assert np.isnan(normal).all()


def test_a_pixel_with_no_new_depth_leaves_its_neighbours_alone() -> None:
    # The change (1, −1, 0, 0) with m = 1: the first pixel, z = 2 at x = 1, gets the new inverse
    # depth 0.5 − 1 < 0, and the second, z = 4 at x = 0, keeps 0.25 and a Depth_Factor of 1.
    z, x, y = _array(2.0, 4.0), _array(1.0, 0.0), _array(0.0, 0.0)
    change = (1.0, -1.0, 0.0, 0.0)
    np.testing.assert_array_equal(apply_change(1 / z, change, 1.0, x, y), [np.nan, 0.25])
    np.testing.assert_array_equal(depth_factor(z, change, 1.0, x, y), [np.nan, 1.0])


def test_depth_factor_on_a_hand_case() -> None:
    factor = depth_factor(_array(HAND_Z), HAND_CHANGE, HAND_M, _array(HAND_X), _array(HAND_Y))
    assert factor.dtype == np.float64
    np.testing.assert_allclose(factor, [1 / 1.625], rtol=1e-15)
    # The old inverse depth over the new one, from the hand case's own numbers.
    np.testing.assert_allclose(factor, [0.5 / 0.8125], rtol=1e-15)


def test_the_identity_change_has_a_depth_factor_of_one() -> None:
    z = np.array([[0.5, 3.0], [12.0, 40.0]])
    x, y = normalised_coords(SKEWED, 2, 2)
    np.testing.assert_array_equal(depth_factor(z, IDENTITY, 0.2, x, y), np.ones((2, 2)))


def test_compose_on_a_hand_case() -> None:
    # m₂ / m₁ = 1 / 0.5 = 2, a₂ = 0.5:
    # (2 × 0.5, 0.5 × 0.1 + 2 × 0.4, 0.5 × 0.2 + 2 × (−0.2), 0.5 × 0.3 + 2 × 0.1)
    # = (1, 0.85, −0.3, 0.35).
    composed = compose((2.0, 0.1, 0.2, 0.3), (0.5, 0.4, -0.2, 0.1), 0.5, 1.0)
    np.testing.assert_allclose(composed, (1.0, 0.85, -0.3, 0.35), rtol=1e-14)


def test_compose_matches_the_two_changes_in_turn_on_a_small_map() -> None:
    # Inverse depths 0.5, 0.25 and 1 at x = 0, 1, −1 and y = 0: m₁ = 0.5.
    # First (2, 0.1, 0, 0.2): 1 + 0.1, 0.5 + 0.15, 2 + 0.05 = 1.1, 0.65, 2.05, so m₂ = 1.1.
    # Second (0.5, 0, 0, 0.1): 0.55 + 0.11, 0.325 + 0.11, 1.025 + 0.11 = 0.66, 0.435, 1.135.
    inverse, x, y = _array(0.5, 0.25, 1.0), _array(0.0, 1.0, -1.0), _array(0.0, 0.0, 0.0)
    first, second, m1, m2 = (2.0, 0.1, 0.0, 0.2), (0.5, 0.0, 0.0, 0.1), 0.5, 1.1
    composed = compose(first, second, m1, m2)
    np.testing.assert_allclose(apply_change(inverse, composed, m1, x, y), [0.66, 0.435, 1.135],
                               rtol=1e-12)


def test_changed_normal_on_a_hand_case() -> None:
    # n = (0, 0, −1) faces the camera along r = (0.5, 0, 1): n·r = −1.
    # m·z·(n·r) = 0.5 × 2 × (−1) = −1, so the raw normal is (0, 0, −1) − (0.25, 0, 0.5)
    # = (−0.25, 0, −1.5), of length √2.3125. Its dot product with r is −1.625 < 0: it already
    # faces the camera.
    normal = changed_normal(np.array([[0.0, 0.0, -1.0]]), _array(HAND_Z), _array(HAND_X),
                            _array(HAND_Y), HAND_CHANGE, HAND_M)
    length = math.sqrt(0.25 ** 2 + 1.5 ** 2)
    np.testing.assert_allclose(normal, [[-0.25 / length, 0.0, -1.5 / length]], rtol=1e-15)


def test_the_identity_change_keeps_a_normal_that_faces_the_camera() -> None:
    # (3, 0, −4) / 5 at r = (0.5, 0, 1): n·r = 0.3 − 0.8 < 0.
    unit = np.array([[0.6, 0.0, -0.8]])
    normal = changed_normal(unit, _array(5.0), _array(0.5), _array(0.0), IDENTITY, 0.3)
    np.testing.assert_allclose(normal, unit, atol=1e-15)


def test_a_normal_facing_away_from_the_camera_is_turned_round() -> None:
    # (0, 0.6, 0.8) at r = (0, 0.25, 1): n·r = 0.15 + 0.8 > 0, so it comes back negated.
    away = np.array([[0.0, 0.6, 0.8]])
    normal = changed_normal(away, _array(3.0), _array(0.0), _array(0.25), IDENTITY, 0.4)
    np.testing.assert_allclose(normal, -away, atol=1e-15)


def test_a_raw_normal_of_length_zero_gives_no_normal() -> None:
    # n = (1, 0, 0) is at right angles to r = (0, 0, 1), so n·r = 0 and with a = 0 the raw normal
    # 0·n + m·z·0·(b, c, d) is 0. The pixel still has a new depth: a + m·z·d = 0 + 1 × 2 × 1 = 2.
    change = (0.0, 0.0, 0.0, 1.0)
    z, x, y = _array(2.0), _array(0.0), _array(0.0)
    assert np.isfinite(depth_factor(z, change, 1.0, x, y)).all()
    normal = changed_normal(np.array([[1.0, 0.0, 0.0]]), z, x, y, change, 1.0)
    assert np.isnan(normal).all()


def test_changed_normal_marks_only_the_pixels_with_no_normal() -> None:
    # The first pixel is the hand case, the second the length-0 case's normal under the hand
    # change with a = 1, which keeps n = (1, 0, 0): a·n + m·z·0·(b, c, d) = (1, 0, 0), and its
    # dot product with r = (0, 0, 1) is 0, not above 0, so it isn't turned round.
    normals = np.array([[0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])
    result = changed_normal(normals, _array(HAND_Z, 2.0), _array(HAND_X, 0.0),
                            _array(HAND_Y, 0.0), HAND_CHANGE, HAND_M)
    length = math.sqrt(0.25 ** 2 + 1.5 ** 2)
    np.testing.assert_allclose(result, [[-0.25 / length, 0.0, -1.5 / length], [1.0, 0.0, 0.0]],
                               rtol=1e-15)


# Normalised_Coordinates and the Median_Inverse_Depth

def test_normalised_coords_at_the_full_size_are_the_pixel_centres() -> None:
    x, y = normalised_coords(SKEWED, SKEWED.width, SKEWED.height)
    assert x.shape == y.shape == (12, 16)
    assert x.dtype == y.dtype == np.float64
    columns = (np.arange(16) + 0.5 - 7.5) / 20.0
    rows = (np.arange(12) + 0.5 - 6.25) / 18.0
    np.testing.assert_allclose(x, np.tile(columns, (12, 1)), rtol=0, atol=1e-15)
    np.testing.assert_allclose(y, np.tile(rows[:, np.newaxis], (1, 16)), rtol=0, atol=1e-15)


def test_normalised_coords_at_half_the_size_cover_the_full_size_pixels() -> None:
    # Pixel (i, j) of the 8 x 6 grid covers the full-size pixels 2i, 2i + 1 across and 2j, 2j + 1
    # down, and its centre (2i + 1, 2j + 1) is midway between theirs, so its ray is their mean.
    full_x, full_y = normalised_coords(SKEWED, 16, 12)
    half_x, half_y = normalised_coords(SKEWED, 8, 6)
    assert half_x.shape == half_y.shape == (6, 8)
    covered_x = full_x.reshape(6, 2, 8, 2).mean(axis=(1, 3))
    covered_y = full_y.reshape(6, 2, 8, 2).mean(axis=(1, 3))
    np.testing.assert_allclose(half_x, covered_x, rtol=0, atol=1e-15)
    np.testing.assert_allclose(half_y, covered_y, rtol=0, atol=1e-15)
    # Pixel (3, 2) by hand: its centre is (7, 5), so x = (7 − 7.5) / 20 and y = (5 − 6.25) / 18.
    assert half_x[2, 3] == pytest.approx(-0.025, abs=1e-15)
    assert half_y[2, 3] == pytest.approx(-1.25 / 18, abs=1e-15)


def test_median_inverse_depth_uses_only_the_covered_pixels() -> None:
    # Covered inverse depths 1, 0.5 and 0.25: the median is 0.5. The uncovered pixels hold a depth
    # of 100, which would pull the median to 0.375 with four covered pixels, and NaN.
    depth = np.array([[1.0, 2.0, np.nan], [4.0, 100.0, 100.0]])
    covered = np.array([[True, True, False], [True, False, False]])
    assert median_inverse_depth(depth, covered) == 0.5


def test_median_inverse_depth_refuses_a_map_with_no_covered_pixel() -> None:
    with pytest.raises(ValueError):
        median_inverse_depth(np.ones((2, 2)), np.zeros((2, 2), dtype=bool))


# The warp on the plane scene (Req 9.3)

SCORE_WIDTH, SCORE_HEIGHT = 8, 6  # half the plane camera's 16 x 12


def _pose(index: int) -> np.ndarray:
    image = PLANE_IMAGES[index]
    return world_to_camera(np.array(image.quaternion, dtype=np.float64),
                           np.array(image.translation, dtype=np.float64))


def _plane_depth(pose: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """The oracle: the camera-space z at which each ray (x, y, 1) of the camera `pose` meets the
    plane z = PLANE_Z. The camera centre is −Rᵀt and the ray's world direction Rᵀ(x, y, 1)."""
    rotation, translation = pose[:3, :3], pose[:3, 3]
    centre = -rotation.T @ translation
    rays = np.stack([x, y, np.ones_like(x)], axis=-1)
    rise = rays @ rotation[:, 2]  # the world z of Rᵀ·ray: Σᵢ R[i, 2]·rayᵢ
    return (PLANE_Z - centre[2]) / rise


def _see(pose: np.ndarray, k: np.ndarray,
         world: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The oracle: where the camera `pose` with K `k` sees the world points (..., 3), as (u, v, z),
    in COLMAP's pixel convention."""
    camera = world @ pose[:3, :3].T + pose[:3, 3]
    z = camera[..., 2]
    return k[0, 0] * camera[..., 0] / z + k[0, 2], k[1, 1] * camera[..., 1] / z + k[1, 2], z


def test_warp_maps_takes_each_pixel_at_its_plane_depth_to_where_another_image_sees_it() -> None:
    prior_pose, source_pose = _pose(0), _pose(1)
    k = scaled_intrinsics(PLANE, SCORE_WIDTH, SCORE_HEIGHT)
    x, y = normalised_coords(PLANE, SCORE_WIDTH, SCORE_HEIGHT)
    depth = _plane_depth(prior_pose, x, y)
    assert np.all(depth > 0)

    centre = -prior_pose[:3, :3].T @ prior_pose[:3, 3]
    world = centre + depth[..., np.newaxis] * (np.stack([x, y, np.ones_like(x)], axis=-1)
                                               @ prior_pose[:3, :3])
    np.testing.assert_allclose(world[..., 2], PLANE_Z, atol=1e-12)  # the oracle's own check
    u, v, z = _see(source_pose, k, world)
    seen = (z > 0) & (u >= 0) & (u < SCORE_WIDTH) & (v >= 0) & (v < SCORE_HEIGHT)
    assert seen.sum() >= 10  # enough of the plane is in both views to mean something

    map_x, map_y, source_z, inside = warp_maps(depth, 1.0, prior_pose, source_pose, k, x, y)
    assert map_x.shape == map_y.shape == source_z.shape == inside.shape == (6, 8)
    np.testing.assert_array_equal(inside, seen)
    np.testing.assert_allclose(map_x, u - 0.5, rtol=0, atol=1e-6)
    np.testing.assert_allclose(map_y, v - 0.5, rtol=0, atol=1e-6)
    np.testing.assert_allclose(source_z, z, rtol=1e-12)


def test_warp_maps_lands_each_plane_point_on_its_keypoint_in_another_image() -> None:
    # camera_files works each keypoint out with its own quaternion turn, at the full size.
    prior, source = PLANE_IMAGES[0], PLANE_IMAGES[2]
    prior_pose, source_pose = _pose(0), _pose(2)
    k = scaled_intrinsics(PLANE, PLANE.width, PLANE.height)
    seen_from = {pid: (u, v) for u, v, pid in prior.keypoints if 1 <= pid <= len(PLANE_GRID)}
    seen_in = {pid: (u, v) for u, v, pid in source.keypoints if 1 <= pid <= len(PLANE_GRID)}
    ids = sorted(seen_from)
    assert len(ids) == len(PLANE_GRID)
    u0 = _array(*(seen_from[pid][0] for pid in ids))
    v0 = _array(*(seen_from[pid][1] for pid in ids))
    points = np.array([PLANE_GRID[pid - 1] for pid in ids])
    *_, z0 = _see(prior_pose, k, points)
    *_, z1 = _see(source_pose, k, points)
    x, y = (u0 - PLANE.cx) / PLANE.fx, (v0 - PLANE.cy) / PLANE.fy

    map_x, map_y, source_z, inside = warp_maps(z0, 1.0, prior_pose, source_pose, k, x, y,
                                               source_size=(PLANE.width, PLANE.height))
    assert inside.all()
    np.testing.assert_allclose(map_x, [seen_in[pid][0] - 0.5 for pid in ids], rtol=0, atol=1e-6)
    np.testing.assert_allclose(map_y, [seen_in[pid][1] - 0.5 for pid in ids], rtol=0, atol=1e-6)
    np.testing.assert_allclose(source_z, z1, rtol=1e-12)


def test_warp_maps_scales_the_depth_by_the_factor() -> None:
    # The same pose for both frames: the point at s·z on the ray through the image centre lands
    # back on the centre, (4, 3) at the Score_Size, with the source z s·z = 0.5 × 3.
    pose = _pose(3)
    k = scaled_intrinsics(PLANE, SCORE_WIDTH, SCORE_HEIGHT)
    map_x, map_y, source_z, inside = warp_maps(_array(3.0), 0.5, pose, pose, k, _array(0.0),
                                               _array(0.0), source_size=(8, 6))
    np.testing.assert_allclose(map_x, [3.5], atol=1e-12)
    np.testing.assert_allclose(map_y, [2.5], atol=1e-12)
    np.testing.assert_allclose(source_z, [1.5], rtol=1e-12)
    assert inside.all()


def test_warp_maps_is_not_inside_behind_the_source_camera_or_outside_its_image() -> None:
    prior_pose = _pose(0)
    # The same camera turned half round about its y axis: diag(−1, 1, −1) after the prior's pose,
    # so every point in front of the prior camera is behind this one, at minus its depth.
    turned = np.diag([-1.0, 1.0, -1.0, 1.0]) @ prior_pose
    k = scaled_intrinsics(PLANE, SCORE_WIDTH, SCORE_HEIGHT)
    depth, x, y = _array(2.0, 2.0), _array(0.0, 0.1), _array(0.0, -0.1)
    *_, source_z, inside = warp_maps(depth, 1.0, prior_pose, turned, k, x, y, source_size=(8, 6))
    np.testing.assert_allclose(source_z, [-2.0, -2.0], rtol=1e-12)
    assert not inside.any()

    # With the prior's own pose, x = 0 lands on the image centre and x = 10 at u = 8 × 10 + 4 = 84,
    # far outside the 8 pixels across. y = −10 lands at v = −77, above the image.
    x, y = _array(0.0, 10.0, 0.0), _array(0.0, 0.0, -10.0)
    *_, inside = warp_maps(_array(2.0, 2.0, 2.0), 1.0, prior_pose, prior_pose, k, x, y,
                           source_size=(8, 6))
    np.testing.assert_array_equal(inside, [True, False, False])


def test_warp_maps_is_not_inside_at_a_pixel_with_no_depth() -> None:
    pose = _pose(1)
    k = scaled_intrinsics(PLANE, SCORE_WIDTH, SCORE_HEIGHT)
    *_, inside = warp_maps(_array(np.nan, 2.0), 1.0, pose, pose, k, _array(0.0, 0.0),
                           _array(0.0, 0.0), source_size=(8, 6))
    np.testing.assert_array_equal(inside, [False, True])


# The z-buffer projection (Req 14.2)

# A 5 x 3 grid over the plane camera's 16 x 12 pixels, so the Depth_Step's rule scales by 5 / 16
# across and 3 / 12 down, neither a whole number of pixels.
GRID_WIDTH, GRID_HEIGHT = 5, 3


def _landing(u: float, v: float) -> tuple[int, int]:
    """The oracle: the Depth_Step's pixel rule, (column, row)."""
    return (min(math.floor(u * GRID_WIDTH / PLANE.width), GRID_WIDTH - 1),
            min(math.floor(v * GRID_HEIGHT / PLANE.height), GRID_HEIGHT - 1))


def _lands_at(point: tuple[float, float, float]) -> tuple[float, float]:
    """The oracle: where a camera point lands with the plane camera at its full size."""
    x, y, z = point
    return PLANE.fx * x / z + PLANE.cx, PLANE.fy * y / z + PLANE.cy


def test_zbuffer_project_keeps_the_nearest_clamps_the_edges_and_drops_what_it_should() -> None:
    just_under_half = 0.5 - 2.0 ** -52        # u = 16·X + 8 lands just inside the right edge
    just_under_three_eighths = 0.375 - 2.0 ** -53  # v = 16·Y + 6 lands just inside the bottom
    centre_far, centre_near, centre_farther = (0.0, 0.0, 2.0), (0.0, 0.0, 1.0), (0.0, 0.0, 3.0)
    right = (just_under_half, 0.0, 1.0)
    bottom = (0.0, just_under_three_eighths, 1.0)
    corner = (2 * just_under_half, 2 * just_under_three_eighths, 2.0)
    on_the_edge = (0.5, 0.0, 1.0)              # u = 16 exactly: outside the image
    behind = (-0.25, -0.25, -1.0)              # lands at (12, 10), but behind the camera
    points = np.array([centre_far, right, centre_near, bottom, corner, on_the_edge, behind,
                       centre_farther])

    # The oracle's preconditions: the edge points land inside, just under W and H.
    assert _lands_at(right)[0] < PLANE.width and _lands_at(bottom)[1] < PLANE.height
    assert _lands_at(corner)[0] < PLANE.width and _lands_at(corner)[1] < PLANE.height
    assert _lands_at(on_the_edge)[0] == PLANE.width

    expected = np.full((GRID_HEIGHT, GRID_WIDTH), np.nan)
    expected[_landing(*_lands_at(centre_near))[::-1]] = 1.0  # (2, 1): the nearest of three
    expected[_landing(*_lands_at(right))[::-1]] = 1.0         # (4, 1)
    expected[_landing(*_lands_at(bottom))[::-1]] = 1.0        # (2, 2)
    expected[_landing(*_lands_at(corner))[::-1]] = 2.0        # (4, 2)
    assert np.isfinite(expected[2, 4]) and np.isfinite(expected[1, 4])
    assert np.isfinite(expected[2, 2]) and np.isfinite(expected[1, 2])
    # The point behind the camera would land on (3, 2), which nothing else hits.
    assert _landing(*_lands_at(behind)) == (3, 2) and np.isnan(expected[2, 3])

    k = scaled_intrinsics(PLANE, PLANE.width, PLANE.height)
    depth, filled = zbuffer_project(points, k, GRID_WIDTH, GRID_HEIGHT,
                                    camera_size=(PLANE.width, PLANE.height))
    assert depth.shape == filled.shape == (GRID_HEIGHT, GRID_WIDTH)
    assert depth.dtype == np.float64 and filled.dtype == bool
    np.testing.assert_array_equal(depth, expected)
    np.testing.assert_array_equal(filled, np.isfinite(expected))
    assert filled.sum() == 4  # every other pixel is unhit and stays unfilled


def test_zbuffer_project_with_a_k_scaled_to_the_grid() -> None:
    # With no camera_size, K is the plane camera at the grid's own 8 x 6: fx = 8, cx = 4, cy = 3.
    # (0.25, −0.25, 2) lands at u = 8 × 0.125 + 4 = 5, v = 8 × (−0.125) + 3 = 2, pixel (5, 2).
    k = scaled_intrinsics(PLANE, 8, 6)
    depth, filled = zbuffer_project(np.array([[0.25, -0.25, 2.0]]), k, 8, 6)
    expected = np.full((6, 8), np.nan)
    expected[2, 5] = 2.0
    np.testing.assert_array_equal(depth, expected)
    np.testing.assert_array_equal(filled, np.isfinite(expected))


def test_zbuffer_project_with_no_point_in_front_fills_nothing() -> None:
    k = scaled_intrinsics(PLANE, PLANE.width, PLANE.height)
    depth, filled = zbuffer_project(np.array([[0.0, 0.0, -1.0], [0.0, 0.0, 0.0]]), k, 4, 3,
                                    camera_size=(PLANE.width, PLANE.height))
    assert np.isnan(depth).all() and not filled.any()


# Property 1 (Req 6.7)

SMALL = 0.5  # the largest |b|, |c| and |d| a drawn change has


@st.composite
def _cameras(draw: st.DrawFn) -> Camera:
    """A Shared_Camera of 16 to 4,000 pixels each way, fx and fy from 200 to 3,000, and its
    principal point anywhere inside the image."""
    width = draw(st.integers(min_value=16, max_value=4000))
    height = draw(st.integers(min_value=16, max_value=4000))
    focal = st.floats(min_value=200.0, max_value=3000.0)
    return Camera(camera_id=1, width=width, height=height, fx=draw(focal), fy=draw(focal),
                  cx=draw(st.floats(min_value=0.0, max_value=float(width))),
                  cy=draw(st.floats(min_value=0.0, max_value=float(height))))


def _cross(p: tuple[float, float, float],
           q: tuple[float, float, float]) -> tuple[float, float, float]:
    """The oracle's own cross product p × q, in float64."""
    return (p[1] * q[2] - p[2] * q[1], p[2] * q[0] - p[0] * q[2], p[0] * q[1] - p[1] * q[0])


@settings(max_examples=100, deadline=None)
@given(camera=_cameras(),
       grid=st.tuples(st.integers(min_value=1, max_value=16),
                      st.integers(min_value=1, max_value=16)),
       place=st.tuples(st.floats(min_value=0.0, max_value=1.0, exclude_max=True),
                       st.floats(min_value=0.0, max_value=1.0, exclude_max=True)),
       z=st.floats(min_value=0.1, max_value=50.0),
       a=st.floats(min_value=0.01, max_value=10.0),
       bcd=st.tuples(*[st.floats(min_value=-SMALL, max_value=SMALL)] * 3),
       m=st.floats(min_value=0.02, max_value=10.0))
def test_a_change_keeps_every_pixel_on_its_own_ray_in_front_of_the_camera(
        camera: Camera, grid: tuple[int, int], place: tuple[float, float], z: float, a: float,
        bcd: tuple[float, float, float], m: float) -> None:
    """Feature: walls, Property 1: A change keeps every pixel on its own ray, in front of the camera

    Any pixel of a grid of 1 to 16 pixels each way over the Shared_Camera, at a depth z from 0.1
    to 50, under a change (a, b, c, d) with a above 0 and |b|, |c|, |d| at most SMALL, and an m
    from 0.02 to 10. The new depth z′ is 1 over apply_change's new inverse depth at the pixel's
    Normalised_Coordinates, and an example whose new inverse depth isn't finite and above 0 (or
    whose z′ overflows) is dropped.

    The oracle builds the pixel's Ray r = (x, y, 1), the old point P = z·r and the new point
    Q = z′·r with its own float64 code, and checks that they are parallel,
    |P × Q| ≤ 10⁻⁹ × |P|·|Q|, and both in front of the camera, z > 0 and z′ > 0: the change
    moves the pixel along its own ray.

    **Validates: Requirements 6.7**
    """
    width, height = grid
    column = min(int(place[0] * width), width - 1)
    row = min(int(place[1] * height), height - 1)
    x_grid, y_grid = normalised_coords(camera, width, height)
    x, y = float(x_grid[row, column]), float(y_grid[row, column])

    change = (a, *bcd)
    new_inverse = float(apply_change(_array(1.0 / z), change, m, _array(x), _array(y))[0])
    assume(math.isfinite(new_inverse) and new_inverse > 0)
    new_z = 1.0 / new_inverse
    assume(math.isfinite(new_z))

    ray = (x, y, 1.0)
    old_point = (z * ray[0], z * ray[1], z * ray[2])
    new_point = (new_z * ray[0], new_z * ray[1], new_z * ray[2])
    cross = _cross(old_point, new_point)
    cross_length = math.hypot(*cross)
    lengths = math.hypot(*old_point) * math.hypot(*new_point)
    assert cross_length <= 1e-9 * lengths
    assert z > 0 and new_z > 0
    assert old_point[2] > 0 and new_point[2] > 0  # both in front of the camera


# Property 2 (Req 6.8, 6.4)

PLANE_SIDE = 16           # the drawn pixels come from a PLANE_SIDE x PLANE_SIDE grid of rays
PLANE_HALF_WIDTH = 0.5    # whose Normalised_Coordinates span [−0.5, 0.5] each way (a 90° view)
PLANE_MAX_TILT = 50.0     # degrees between a drawn plane's normal and the optical axis


def _grid_coordinate(index: int) -> float:
    """The Normalised_Coordinate of column or row `index` of the PLANE_SIDE grid."""
    return ((index + 0.5) / PLANE_SIDE - 0.5) * 2.0 * PLANE_HALF_WIDTH


def _collinear(cells: list[tuple[int, int]]) -> bool:
    """Whether the grid cells all lie on one line, by exact integer cross products."""
    (i0, j0), (i1, j1) = cells[0], cells[1]
    return all((i1 - i0) * (j - j0) - (j1 - j0) * (i - i0) == 0 for i, j in cells[2:])


@settings(max_examples=100, deadline=None)
@given(cells=st.lists(st.integers(min_value=0, max_value=PLANE_SIDE * PLANE_SIDE - 1),
                      min_size=4, max_size=40, unique=True),
       tilt=st.floats(min_value=0.0, max_value=PLANE_MAX_TILT),
       azimuth=st.floats(min_value=0.0, max_value=360.0),
       offset=st.floats(min_value=0.1, max_value=50.0),
       a=st.floats(min_value=0.1, max_value=10.0),
       bcd=st.tuples(*[st.floats(min_value=-SMALL, max_value=SMALL)] * 3))
def test_a_change_keeps_a_plane_flat_and_carries_its_normal(
        cells: list[int], tilt: float, azimuth: float, offset: float, a: float,
        bcd: tuple[float, float, float]) -> None:
    """Feature: walls, Property 2: A change keeps a plane flat and carries its normal

    A plane n·X = δ with a unit normal n tilted up to PLANE_MAX_TILT degrees from the optical
    axis, at any azimuth, and an offset δ from 0.1 to 50, so it never passes through the camera
    centre. 4 to 40 distinct pixels of a PLANE_SIDE x PLANE_SIDE grid whose rays span
    [−0.5, 0.5] each way, not all on one line. With that view and tilt, n·r ≥ cos 50° −
    sin 50° / √2 > 0.1 at every pixel, so every ray meets the plane in front of the camera, at
    the depth z = δ / (n·r), and no ray grazes it. The plane's normal facing the camera is −n.
    A change (a, b, c, d) with a from 0.1 to 10 and |b|, |c|, |d| at most SMALL, with m the
    median inverse depth of the pixels, and an example whose new inverse depths aren't all
    finite and above 0 is dropped.

    The oracle builds each changed point Q = z′·(x, y, 1), with z′ = 1 / apply_change's new
    inverse depth, fits a plane to the points with an SVD of the centred points, and checks
    that every point lies within 10⁻⁶ × |Q| of the fitted plane, and that changed_normal of −n
    at each pixel equals the fitted unit normal facing the camera within 10⁻⁶ in every part.

    **Validates: Requirements 6.8, 6.4**
    """
    grid = [divmod(cell, PLANE_SIDE) for cell in cells]  # (row, column)
    assume(not _collinear(grid))
    y = np.array([_grid_coordinate(row) for row, _ in grid])
    x = np.array([_grid_coordinate(column) for _, column in grid])

    theta, phi = math.radians(tilt), math.radians(azimuth)
    normal = np.array([math.sin(theta) * math.cos(phi), math.sin(theta) * math.sin(phi),
                       math.cos(theta)])
    along = normal[0] * x + normal[1] * y + normal[2]  # n·r
    assert (along > 0.1).all()
    z = offset / along  # each ray meets n·X = δ in front of the camera
    inverse = 1.0 / z
    m = float(np.median(inverse))

    change = (a, *bcd)
    new_inverse = apply_change(inverse, change, m, x, y)
    assume(bool(np.isfinite(new_inverse).all() and (new_inverse > 0).all()))
    new_z = 1.0 / new_inverse
    assume(bool(np.isfinite(new_z).all()))

    points = np.stack([new_z * x, new_z * y, new_z], axis=-1)
    centred = points - points.mean(axis=0)
    fitted = np.linalg.svd(centred)[2][-1]
    fitted = fitted / np.linalg.norm(fitted)
    if float(fitted @ points.mean(axis=0)) > 0:  # face the camera: n·Q < 0 for a point Q on it
        fitted = -fitted

    residuals = np.abs(centred @ fitted)
    distances = np.linalg.norm(points, axis=1)
    assert (residuals <= 1e-6 * distances).all()

    facing = np.broadcast_to(-normal, (len(cells), 3))
    carried = changed_normal(facing, z, x, y, change, m)
    assert np.isfinite(carried).all()
    assert np.abs(carried - fitted[np.newaxis, :]).max() <= 1e-6


# Property 3 (Req 6.9, 6.2)

MAP_SIDE = 32         # the largest depth map drawn, each way
COORD_LIMIT = 10.0    # the largest |x| and |y| drawn: a view of about 168° across
GRAZING = 1e-12       # |n·r| at most this × |r|: the ray grazes the surface, so either sign faces


@st.composite
def _identity_maps(draw: st.DrawFn) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """A depth map of 1 x 1 to MAP_SIDE x MAP_SIDE pixels of depths from 0.1 to 50, the pixels'
    Normalised_Coordinates x and y anywhere in [−COORD_LIMIT, COORD_LIMIT], and a unit normal per
    pixel from two angles, turned round where it faces away from the pixel's Ray (x, y, 1)."""
    shape = (draw(st.integers(min_value=1, max_value=MAP_SIDE)),
             draw(st.integers(min_value=1, max_value=MAP_SIDE)))
    z = draw(hnp.arrays(np.float64, shape, elements=st.floats(min_value=0.1, max_value=50.0)))
    coords = st.floats(min_value=-COORD_LIMIT, max_value=COORD_LIMIT)
    x = draw(hnp.arrays(np.float64, shape, elements=coords))
    y = draw(hnp.arrays(np.float64, shape, elements=coords))
    theta = draw(hnp.arrays(np.float64, shape, elements=st.floats(min_value=0.0,
                                                                   max_value=math.pi)))
    phi = draw(hnp.arrays(np.float64, shape, elements=st.floats(min_value=0.0,
                                                                 max_value=2.0 * math.pi)))
    normal = np.stack([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)],
                      axis=-1)
    along = normal[..., 0] * x + normal[..., 1] * y + normal[..., 2]  # n·r
    normal = np.where((along > 0)[..., np.newaxis], -normal, normal)
    return z, x, y, normal


@settings(max_examples=100, deadline=None)
@given(drawn=_identity_maps(), m=st.floats(min_value=0.01, max_value=10.0))
def test_the_identity_change_gives_the_depths_and_normals_back(
        drawn: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], m: float) -> None:
    """Feature: walls, Property 3: The Identity_Change gives the depths and normals back

    A depth map of 1 x 1 to MAP_SIDE x MAP_SIDE pixels of depths z from 0.1 to 50, any
    Normalised_Coordinates in [−COORD_LIMIT, COORD_LIMIT], a unit normal per pixel turned to face
    the camera, and an m from 0.01 to 10.

    The oracle checks that apply_change with (1, 0, 0, 0) gives back the inverse depth 1.0 / z at
    every pixel bit for bit, through struct.pack("<d", ·), so the Identity_Change gives back the
    same depth, and that changed_normal with (1, 0, 0, 0) gives back each normal within 10⁻⁶ in
    every part. Where the Ray grazes the surface, |n·r| ≤ GRAZING × |r|, the rounding of the unit
    normal may tip which way it faces, so there either n or −n is right.

    **Validates: Requirements 6.9, 6.2**
    """
    z, x, y, normal = drawn
    inverse = 1.0 / z

    new = apply_change(inverse, IDENTITY, m, x, y)
    assert new.shape == inverse.shape
    expected = [struct.pack("<d", value) for value in inverse.ravel().tolist()]
    returned = [struct.pack("<d", value) for value in new.ravel().tolist()]
    assert returned == expected

    carried = changed_normal(normal, z, x, y, IDENTITY, m)
    assert carried.shape == normal.shape
    assert np.isfinite(carried).all()
    same = np.abs(carried - normal).max(axis=-1) <= 1e-6
    turned = np.abs(carried + normal).max(axis=-1) <= 1e-6
    along = normal[..., 0] * x + normal[..., 1] * y + normal[..., 2]
    ray_length = np.sqrt(x * x + y * y + 1.0)
    grazing = np.abs(along) <= GRAZING * ray_length
    assert (same | (grazing & turned)).all()


# Property 4 (Req 6.10, 6.5)

COMPOSE_SIDE = 16       # the largest depth map drawn, each way
COMPOSE_COORDS = 2.0    # the largest |x| and |y| drawn
COMPOSE_A = 10.0        # the largest |a| a drawn change has
COMPOSE_BC = 1.0        # the largest |b| and |c| a drawn change has
LIFT_MIN, LIFT_MAX = 0.01, 2.0  # the drawn share that sets d, below
CANCELLATION = 1e-3     # each ι₂ is at least this share of the sum of its terms' sizes

Change4 = tuple[float, float, float, float]


def _lifted_change(draw: st.DrawFn, inverse: np.ndarray, terms: np.ndarray, m: float,
                   x: np.ndarray, y: np.ndarray) -> Change4:
    """A change (a, b, c, d) of the map `inverse`, of median `m`, whose terms other than d have
    the sizes `terms` before a is applied. a and b, c are drawn; d is set so the smallest new
    inverse depth is a drawn share s from LIFT_MIN to LIFT_MAX of S, the largest size of the new
    inverse depth's terms other than d's: so every new inverse depth is at least s·S > 0, and
    the d term, of size at most (1 + s)·S, keeps the sum of every term's size under (2 + s)·S."""
    a = draw(st.floats(min_value=-COMPOSE_A, max_value=COMPOSE_A))
    part = st.floats(min_value=-COMPOSE_BC, max_value=COMPOSE_BC)
    b, c = draw(part), draw(part)
    share = draw(st.floats(min_value=LIFT_MIN, max_value=LIFT_MAX))
    without_d = a * inverse + m * (b * x + c * y)
    sizes = abs(a) * terms + m * (np.abs(b * x) + np.abs(c * y))
    largest = max(float(sizes.max()), m)
    d = (share * largest - float(without_d.min())) / m
    return a, b, c, d


@st.composite
def _compose_cases(draw: st.DrawFn) -> tuple[np.ndarray, np.ndarray, np.ndarray, Change4,
                                             Change4]:
    """A depth map of 1 x 1 to COMPOSE_SIDE x COMPOSE_SIDE pixels of depths from 0.1 to 50, the
    pixels' Normalised_Coordinates x and y anywhere in [−COMPOSE_COORDS, COMPOSE_COORDS], a first
    change of the map and a second change of the map the first one makes, each from
    _lifted_change. The generator's own sums only pick d; the test's oracle works everything out
    again."""
    shape = (draw(st.integers(min_value=1, max_value=COMPOSE_SIDE)),
             draw(st.integers(min_value=1, max_value=COMPOSE_SIDE)))
    z = draw(hnp.arrays(np.float64, shape, elements=st.floats(min_value=0.1, max_value=50.0)))
    coords = st.floats(min_value=-COMPOSE_COORDS, max_value=COMPOSE_COORDS)
    x = draw(hnp.arrays(np.float64, shape, elements=coords))
    y = draw(hnp.arrays(np.float64, shape, elements=coords))

    inverse0 = 1.0 / z
    m1 = float(np.median(inverse0))
    first = _lifted_change(draw, inverse0, inverse0, m1, x, y)
    a1, b1, c1, d1 = first
    inverse1 = a1 * inverse0 + m1 * (b1 * x + c1 * y + d1)
    terms1 = np.abs(a1) * inverse0 + m1 * (np.abs(b1 * x) + np.abs(c1 * y) + abs(d1))
    m2 = float(np.median(inverse1))
    assume(m2 > 0)
    second = _lifted_change(draw, inverse1, terms1, m2, x, y)
    return z, x, y, first, second


def _usable(inverse: np.ndarray) -> bool:
    return bool(np.isfinite(inverse).all() and (inverse > 0).all())


@settings(max_examples=100, deadline=None)
@given(case=_compose_cases())
def test_composing_two_changes_equals_applying_them_in_turn(
        case: tuple[np.ndarray, np.ndarray, np.ndarray, Change4, Change4]) -> None:
    """Feature: walls, Property 4: Composing two changes equals applying them in turn

    A depth map of 1 x 1 to COMPOSE_SIDE x COMPOSE_SIDE pixels of depths from 0.1 to 50, so
    inverse depths ι₀ from 0.02 to 10, any Normalised_Coordinates in
    [−COMPOSE_COORDS, COMPOSE_COORDS], a first change and a second change of the map the first
    makes. Each change has an a of either sign with |a| at most COMPOSE_A, |b| and |c| at most
    COMPOSE_BC, and a d set by the generator so that its result is above 0 everywhere and is
    never the near-cancellation of much bigger terms (see _lifted_change).

    The oracle applies the changes in turn with its own formula: m₁ = np.median(ι₀),
    ι₁ = a₁·ι₀ + m₁·(b₁x + c₁y + d₁), m₂ = np.median(ι₁) and ι₂ = a₂·ι₁ + m₂·(b₂x + c₂y + d₂).
    An example where ι₁ or ι₂ isn't finite and above 0 at every pixel is dropped. So is one
    where some ι₂ comes from cancellation, smaller than CANCELLATION × T with
    T = |a₂|·(|a₁·ι₀| + m₁·(|b₁x| + |c₁y| + |d₁|)) + m₂·(|b₂x| + |c₂y| + |d₂|), the sum of the
    sizes of the terms that make it: the two ways of working ι₂ out round differently, by a few
    units in the last place of T, which is a relative error above 10⁻⁹ only when ι₂ is under
    about 10⁻⁶ × T. The generator keeps ι₂ ≥ LIFT_MIN / (2 + LIFT_MAX) × T, so this drop is a
    guard that hardly ever fires. Then apply_change of ι₀ with compose(first, second, m₁, m₂)
    and m₁ must give ι₂ within a relative error of 10⁻⁹ at every pixel.

    **Validates: Requirements 6.10, 6.5**
    """
    z, x, y, first, second = case
    a1, b1, c1, d1 = first
    a2, b2, c2, d2 = second
    inverse0 = 1.0 / z
    m1 = float(np.median(inverse0))
    with np.errstate(over="ignore", invalid="ignore"):
        inverse1 = a1 * inverse0 + m1 * (b1 * x + c1 * y + d1)
    assume(_usable(inverse1))
    m2 = float(np.median(inverse1))
    with np.errstate(over="ignore", invalid="ignore"):
        inverse2 = a2 * inverse1 + m2 * (b2 * x + c2 * y + d2)
    assume(_usable(inverse2))

    terms1 = np.abs(a1 * inverse0) + m1 * (np.abs(b1 * x) + np.abs(c1 * y) + abs(d1))
    terms2 = abs(a2) * terms1 + m2 * (np.abs(b2 * x) + np.abs(c2 * y) + abs(d2))
    assume(bool((inverse2 >= CANCELLATION * terms2).all()))

    composed = apply_change(inverse0, compose(first, second, m1, m2), m1, x, y)
    assert composed.shape == inverse2.shape
    assert np.isfinite(composed).all()
    np.testing.assert_allclose(composed, inverse2, rtol=1e-9, atol=0)
