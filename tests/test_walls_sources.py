"""Tests for the Walls_Step's Scene_Scale, Start_Depths and Source_Frames:
splatpipe/walls/sources.py (walls spec, Req 7 and 8, A7, A8).

Every expected value is worked out here, in the test, never by calling the function being
checked:
- scale_points and scene_scale on tests/camera_files.py's plane scene, with a Depth_Prior that
  holds each Observed_Point's own z ÷ 1.3 at the pixel it lands on, as plane_moge_output does
  with a metres_per_unit of 1 / 1.3. The test's own quaternion turn and projection find each
  point's camera-space z and Prior_Size pixel (Req 7.1, 7.2).
  The Prior_Format stores inverse depth as float32, and float32(1.3 / z) is off 1.3 / z by up to
  6 × 10⁻⁸ of itself, which on this scene puts k off 1.3 by about 10⁻⁸. So the test picks each
  point's prior inverse depth first, as the float32 value float32(1.3 / z), and then moves the
  point along camera 0's ray through it, so its camera-space z becomes 1.3 / that float32 value
  in float64. The move is a few parts in 10⁸ of the point's distance, so it stays on its pixel,
  and z × the stored inverse depth is 1.3 to float64's rounding, so k = 1.3 within 10⁻⁹ is an
  honest check of the maths and not of float32's rounding.
- scene_scale's 3-MAD gate on Scale_Points whose r = ln(z / Prior_Depth) are spelled out: a point
  far off is dropped, D is the median absolute deviation unscaled, 9 Scale_Points give k = 1,
  and a set whose D is 0 keeps only the points at the median (Req 7.2, 7.3, A7).
- start_depth on a 4 × 3 Depth_Prior scaled to an 8 × 6 Score_Size, against the test's own
  bilinear taps and prior-pixel rule (Req 7.4, 7.5).
- overlap, image_motion and source_rank on crafted cameras whose answers are counted or have a
  closed form, and choose_sources on candidates that slide sideways from a frame that looks
  down +z at a wall 1 away (Req 8.1–8.6, A8).
Settings come from conftest's WALLS, the real [walls] values. Everything runs on the CPU on a
few dozen points (Req 22.1).
"""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple

import numpy as np
import pytest

from splatpipe.train.colmap import Camera, ModelImage, SparseModel, read_model
from splatpipe.train.prior_views import prior_taps
from splatpipe.walls.sources import (Candidate, SourceChoice, choose_sources, image_motion,
                                     overlap, scale_points, scene_scale, source_rank,
                                     start_depth)

from camera_files import (PLANE_CAMERA, PLANE_GRID, PLANE_IMAGES, PLANE_NO_POINT, PLANE_POINTS,
                          STRAY_XY, PointRecord, write_model)
from conftest import WALLS

SCALE = 1.3  # the Scene_Scale the plane scene's Depth_Prior is built for
MAD_GATE = float(WALLS["scale_mad_gate"])  # 3.0
MIN_POINTS = int(WALLS["scale_min_points"])  # 10
DELTA = float(WALLS["motion_depth_change"])  # 0.01
MOTION_CAP = float(WALLS["source_motion_cap"])  # 3.0

# The [walls] settings choose_sources reads, at their real values.
SOURCE_KEYS = ("max_sources", "source_min_seconds", "source_fallback_seconds", "min_sources",
               "source_min_gap", "source_min_overlap", "motion_depth_change",
               "source_min_motion", "source_motion_cap")
SOURCE_SETTINGS = SimpleNamespace(**{key: WALLS[key] for key in SOURCE_KEYS})

# The plane scene's ids for the two points the exclusion test adds: one behind camera 0 and one
# outside its Frame.
BEHIND_ID = 31
OUTSIDE_ID = 32


# --- The test's own pose maths -----------------------------------------------------------------

def _rotation(quaternion: tuple[float, ...] | np.ndarray) -> np.ndarray:
    """The world-to-camera rotation of the quaternion (w, x, y, z), set to length 1 first."""
    w, x, y, z = np.asarray(quaternion, dtype=np.float64) / np.linalg.norm(quaternion)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _in_camera(quaternion: tuple[float, ...] | np.ndarray,
               translation: tuple[float, ...] | np.ndarray, points: np.ndarray) -> np.ndarray:
    """The (n, 3) world `points` in the camera, R·p + t."""
    return np.asarray(points, dtype=np.float64) @ _rotation(quaternion).T + np.asarray(translation)


def _centre(quaternion: tuple[float, ...], translation: tuple[float, ...]) -> np.ndarray:
    """The camera centre −Rᵀt."""
    return -_rotation(quaternion).T @ np.asarray(translation, dtype=np.float64)


def _pose(rotation: np.ndarray, translation: tuple[float, float, float]) -> np.ndarray:
    """The (4, 4) world-to-camera matrix [[R, t], [0, 0, 0, 1]]."""
    pose = np.eye(4)
    pose[:3, :3] = rotation
    pose[:3, 3] = translation
    return pose


def _k(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])


# --- The plane scene ---------------------------------------------------------------------------

class PlaneScene(NamedTuple):
    model: SparseModel
    image: ModelImage           # image 0, the Prior_Frame
    camera: Camera              # the Shared_Camera, 16 x 12, so the Prior_Size is 16 x 12
    pixels: np.ndarray          # (30, 2) int64, each grid point's Prior_Size (column, row)
    z: np.ndarray               # (30,) float64, each grid point's camera-space z
    inverse_depth: np.ndarray   # (12, 16) float32, the Depth_Prior's inverse_depth


def _plane_scene(tmp_path: Path, *, extras: bool = False) -> PlaneScene:
    """The plane scene read back with read_model, with each grid point moved along camera 0's ray
    so z × the stored float32 inverse depth is 1.3 in float64 (see the module docstring), and
    the Depth_Prior of image 0 that holds 1.3 / z at the pixel each point lands on, 0.5
    elsewhere. With `extras`, image 0 also observes BEHIND_ID, grid point 1 turned through the
    camera centre so it projects onto the same pixel from behind, and OUTSIDE_ID, a point at
    u = 17, v = 6, past the Frame's right edge."""
    first = PLANE_IMAGES[0]
    centre = _centre(first.quaternion, first.translation)
    grid = np.array(PLANE_GRID, dtype=np.float64)
    z = _in_camera(first.quaternion, first.translation, grid)[:, 2]
    inverse32 = (SCALE / z).astype(np.float32)
    wanted = SCALE / inverse32.astype(np.float64)
    moved = centre + (wanted / z)[:, np.newaxis] * (grid - centre)
    points: list[PointRecord] = [replace(point, position=tuple(float(v) for v in moved[place]))
                                 for place, point in enumerate(PLANE_POINTS)]
    images = list(PLANE_IMAGES)
    if extras:
        behind = centre - (moved[0] - centre)
        outside_camera = np.array([0.5625 * 3.0, 0.0, 3.0])  # u = 16·0.5625 + 8 = 17, v = 6
        outside = _rotation(first.quaternion).T @ (outside_camera - np.asarray(first.translation))
        count = len(first.keypoints)
        images[0] = replace(first, keypoints=(*first.keypoints, (*STRAY_XY, BEHIND_ID),
                                              (*STRAY_XY, OUTSIDE_ID)))
        points += [PointRecord(point_id=BEHIND_ID, position=tuple(float(v) for v in behind),
                               colour=(1, 2, 3), track=((first.image_id, count),)),
                   PointRecord(point_id=OUTSIDE_ID, position=tuple(float(v) for v in outside),
                               colour=(1, 2, 3), track=((first.image_id, count + 1),))]
    write_model(tmp_path / "sparse", PLANE_CAMERA, images, points)
    model = read_model(tmp_path / "sparse")
    image = model.images[0]
    in_camera = _in_camera(image.quaternion, image.translation, model.points[:len(PLANE_GRID)])
    u = PLANE_CAMERA.fx * in_camera[:, 0] / in_camera[:, 2] + PLANE_CAMERA.cx
    v = PLANE_CAMERA.fy * in_camera[:, 1] / in_camera[:, 2] + PLANE_CAMERA.cy
    width, height = PLANE_CAMERA.width, PLANE_CAMERA.height  # the Prior_Size is the camera's
    pixels = np.stack([np.minimum(np.floor(u).astype(np.int64), width - 1),
                       np.minimum(np.floor(v).astype(np.int64), height - 1)], axis=-1)
    inverse_depth = np.full((height, width), 0.5, dtype=np.float32)
    inverse_depth[pixels[:, 1], pixels[:, 0]] = inverse32
    return PlaneScene(model, image, model.camera, pixels, in_camera[:, 2], inverse_depth)


def _prior_depth(inverse_depth: np.ndarray) -> np.ndarray:
    """The Prior_Depth, 1 / inverse_depth in float64."""
    return 1.0 / inverse_depth.astype(np.float64)


def test_the_plane_scene_gives_a_scene_scale_of_1_3(tmp_path: Path) -> None:
    scene = _plane_scene(tmp_path)
    covered = np.ones(scene.inverse_depth.shape, dtype=bool)
    assert len(set(map(tuple, scene.pixels.tolist()))) == len(PLANE_GRID)  # a pixel each
    pixels, z, depth_at = scale_points(scene.model, scene.image, scene.camera, covered,
                                       _prior_depth(scene.inverse_depth))
    np.testing.assert_array_equal(pixels, scene.pixels)
    np.testing.assert_allclose(z, scene.z, rtol=1e-12)
    stored = scene.inverse_depth[scene.pixels[:, 1], scene.pixels[:, 0]]
    np.testing.assert_array_equal(depth_at, 1.0 / stored.astype(np.float64))
    k, count = scene_scale(z, depth_at, MAD_GATE, MIN_POINTS)
    assert count == len(PLANE_GRID)
    assert abs(k - SCALE) <= 1e-9
    # scale_min_points of them are enough, and one fewer gives k = 1 (Req 7.3).
    k, count = scene_scale(z[:MIN_POINTS], depth_at[:MIN_POINTS], MAD_GATE, MIN_POINTS)
    assert (count, abs(k - SCALE) <= 1e-9) == (MIN_POINTS, True)
    assert scene_scale(z[:MIN_POINTS - 1], depth_at[:MIN_POINTS - 1], MAD_GATE,
                       MIN_POINTS) == (1.0, MIN_POINTS - 1)


def test_scale_points_leaves_out_points_that_cant_be_scale_points(tmp_path: Path) -> None:
    scene = _plane_scene(tmp_path, extras=True)
    # The stray keypoints, among them one naming PLANE_NO_POINT, the id of no point, name no
    # Observed_Point, so the image observes the 30 grid points and the two extras.
    assert PLANE_NO_POINT not in scene.image.observed_ids.tolist()
    assert scene.image.observed_ids.tolist() == [*range(1, 31), BEHIND_ID, OUTSIDE_ID]
    uncovered = 5  # grid point 6's pixel isn't Covered
    covered = np.ones(scene.inverse_depth.shape, dtype=bool)
    covered[scene.pixels[uncovered, 1], scene.pixels[uncovered, 0]] = False
    pixels, z, depth_at = scale_points(scene.model, scene.image, scene.camera, covered,
                                       _prior_depth(scene.inverse_depth))
    kept = [place for place in range(len(PLANE_GRID)) if place != uncovered]
    np.testing.assert_array_equal(pixels, scene.pixels[kept])
    np.testing.assert_allclose(z, scene.z[kept], rtol=1e-12)
    assert np.all(z > 0)  # the point behind the camera, on grid point 1's pixel, is left out
    assert depth_at.shape == (len(kept),)


# --- The 3-MAD gate ----------------------------------------------------------------------------

def _points_with_r(r: list[float]) -> tuple[np.ndarray, np.ndarray]:
    """Scale_Points whose ln(z / Prior_Depth) are `r`, with Prior_Depths 1 to n."""
    depth = np.arange(1.0, len(r) + 1.0)
    return depth * np.exp(np.array(r)), depth


# Ten offsets around ln 1.3 with a median of 0 among the eleven, and an eleventh that's off.
# The deviations from 0, sorted, are 0, .01, .01, .02, .02, .03, .03, .04, .04, .05 and the
# eleventh's, so D = .03 and the gate keeps |r − r̃| ≤ 3 × .03 = .09. The ten kept have a median
# of (−0.01 + 0) / 2 = −0.005. At 0.1 the point would pass a D scaled to a standard deviation,
# 1.4826 × .03 × 3 ≈ .133, so it shows the gate uses D as it is.
NEAR = [-0.05, -0.04, -0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04]


@pytest.mark.parametrize("off", [0.1, 5.0, -5.0], ids=["just-past-the-gate", "far", "far-below"])
def test_the_mad_gate_drops_a_point_far_off(off: float) -> None:
    # With −5 the sorted r are −5, −0.05 … 0.04, so r̃ is the sixth, −0.01, D = .03 again
    # (deviations 0, .01, .01, .02, .02, .03, .03, .04, .04, .05, 4.99), and the gate keeps the
    # ten near ones, with a median of −0.005 again.
    z, depth = _points_with_r([math.log(SCALE) + value for value in [*NEAR, off]])
    k, count = scene_scale(z, depth, MAD_GATE, MIN_POINTS)
    assert count == 11
    assert k == pytest.approx(SCALE * math.exp(-0.005), rel=1e-12)


def test_a_deviation_of_0_keeps_only_the_points_at_the_median() -> None:
    # Seven of ten at exactly 1.3, so r̃ = ln 1.3 and D = 0. A gate of ≤ 0 × 3 keeps the seven.
    z = np.array([SCALE] * 7 + [2.0, 0.5, 3.0])
    k, count = scene_scale(z, np.ones(10), MAD_GATE, MIN_POINTS)
    assert (k, count) == (pytest.approx(SCALE, rel=1e-15), 10)


# --- The Start_Depths --------------------------------------------------------------------------

def _axis_taps(stored: int, target: int, place: int) -> list[tuple[int, float]]:
    """The stored pixels a bilinear sample of target pixel `place` draws on with a weight above
    0, with their weights: the pixel sits at stored coordinate (place + 0.5)·S / T − 0.5,
    clamped to the first and the last pixel."""
    coordinate = (place + 0.5) * stored / target - 0.5
    if coordinate <= 0:
        return [(0, 1.0)]
    first = math.floor(coordinate)
    weight = coordinate - first
    second = min(first + 1, stored - 1)
    if second == first or weight == 0:
        return [(first, 1.0)]
    return [(first, 1.0 - weight), (second, weight)]


def test_start_depth_is_k_times_the_prior_depth_at_both_sizes() -> None:
    k = SCALE
    prior_w, prior_h, score_w, score_h = 4, 3, 8, 6
    inverse = (np.arange(1, 13, dtype=np.float32) / np.float32(8)).reshape(prior_h, prior_w)
    mask = np.ones((prior_h, prior_w), dtype=bool)
    mask[1, 2] = False
    inverse[1, 2] = 0.0  # a pixel with no prior, as the Prior_Format stores it
    prior_size, score_size, depth_pixels = start_depth(
        inverse, mask, prior_taps(prior_w, prior_h, score_w, score_h), k)

    expected = np.full((prior_h, prior_w), np.nan)
    expected[mask] = k * (1.0 / inverse[mask].astype(np.float64))
    np.testing.assert_allclose(prior_size, expected, rtol=1e-15)
    assert np.isnan(prior_size[1, 2])

    wanted_pixels = np.zeros((score_h, score_w), dtype=bool)
    wanted_depth = np.full((score_h, score_w), np.nan)
    for row in range(score_h):
        for column in range(score_w):
            rows = _axis_taps(prior_h, score_h, row)
            columns = _axis_taps(prior_w, score_w, column)
            nearest = ((2 * row + 1) * prior_h // (2 * score_h),
                       (2 * column + 1) * prior_w // (2 * score_w))
            if mask[nearest] and all(mask[r, c] for r, _ in rows for c, _ in columns):
                wanted_pixels[row, column] = True
                wanted_depth[row, column] = k / sum(wr * wc * float(inverse[r, c])
                                                    for r, wr in rows for c, wc in columns)
    # Stored row 1 feeds Score rows 1 to 4 and stored column 2 feeds Score columns 3 to 6, so the
    # one pixel with no prior leaves out 4 × 4 of the 48 Score_Size pixels.
    assert int(np.count_nonzero(~wanted_pixels)) == 16
    assert not wanted_pixels[1:5, 3:7].any()
    np.testing.assert_array_equal(depth_pixels, wanted_pixels)
    np.testing.assert_allclose(score_size, wanted_depth, rtol=1e-12)  # NaN where wanted is NaN


# --- Overlap, Image_Motion and Source_Rank -----------------------------------------------------

# A source turned 90° about its z axis and moved by dyadic amounts, so the edge points below
# land exactly on u = 0 and u = 16 after the round trip through the world.
TURN = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
SHIFT = (0.25, -0.5, 0.75)
SMALL_K = _k(16.0, 16.0, 8.0, 6.0)  # a 16 x 12 image


def _world_from_source(camera_points: list[tuple[float, float, float]]) -> np.ndarray:
    """World points that the source of pose (TURN, SHIFT) sees at `camera_points`."""
    return (np.array(camera_points) - np.array(SHIFT)) @ TURN  # Rᵀ(c − t), row by row


def test_overlap_counts_the_points_in_front_and_inside() -> None:
    points = _world_from_source([
        (0.0, 0.0, 2.0),      # u = 8, v = 6: inside
        (0.2, 0.1, 1.0),      # u = 11.2, v = 7.6: inside
        (-0.5, 0.0, 1.0),     # u = 0: inside, [0, W) holds its left edge
        (0.5, 0.0, 1.0),      # u = 16: outside, [0, W) leaves out its right edge
        (0.0, 0.375, 1.0),    # v = 12: outside
        (0.0, 0.0, -1.0),     # behind the camera
        (0.0, 0.0, 0.0),      # at the camera centre, z = 0, not in front
    ])
    share = overlap(points, _pose(TURN, SHIFT), SMALL_K, 16, 12)
    assert share == pytest.approx(3 / 7, rel=1e-15)
    assert overlap(np.zeros((0, 3)), _pose(TURN, SHIFT), SMALL_K, 16, 12) == 0.0


def test_image_motion_is_the_median_shift_in_the_source() -> None:
    # The Prior_Frame's camera sits at `centre`, unturned. The source is the same camera moved
    # b = 0.5 along x, so a camera point (X, Y, Z) of the Prior_Frame lands at
    # u = fx·(X − b) / Z + cx, and at (1 + δ) times its depth at fx·(X − b / (1 + δ)) / Z + cx,
    # a shift of fx·b·δ / ((1 + δ)·Z) along u and none along v.
    fx, baseline = 100.0, 0.5
    centre = np.array([0.25, -0.5, 1.0])
    camera_points = np.array([(0.0, 0.0, 2.0), (0.2, 0.1, 4.0), (-0.1, 0.0, 5.0),
                              (0.1, -0.1, 1.0),
                              (3.0, 0.0, 1.0)])  # u = 300 in a 100 x 80 image: not inside
    source = _pose(np.eye(3), tuple(-centre - np.array([baseline, 0.0, 0.0])))
    k = _k(fx, fx, 50.0, 40.0)
    motion = image_motion(camera_points + centre, centre, source, k, 100, 80, DELTA)
    shifts = sorted(fx * baseline * DELTA / ((1 + DELTA) * z) for z in (2.0, 4.0, 5.0, 1.0))
    assert motion == pytest.approx((shifts[1] + shifts[2]) / 2, rel=1e-9)
    # No point inside the source: no motion.
    assert image_motion(camera_points[4:] + centre, centre, source, k, 100, 80, DELTA) == 0.0


@pytest.mark.parametrize("share, motion, rank", [(0.5, 2.0, 1.0), (0.5, 4.0, 1.5),
                                                 (0.0, 5.0, 0.0), (1.0, 3.0, 3.0)])
def test_source_rank_caps_the_motion(share: float, motion: float, rank: float) -> None:
    assert source_rank(share, motion, MOTION_CAP) == pytest.approx(rank, rel=1e-15)


# --- choose_sources ----------------------------------------------------------------------------

# The Prior_Frame: a 640 x 480 camera with fx = fy = 640 at the origin, unturned, its 4 x 3
# Covered_Pixels all at a Start_Depth of 1, and the Score_Size the camera's own. Its pixel
# centres sit at u = 80, 240, 400, 560 and v = 80, 240, 400, so x = −0.375, −0.125, 0.125, 0.375
# and y = −0.25, 0, 0.25. A candidate is the same camera moved b along x.
WIDE = Camera(camera_id=1, width=640, height=480, fx=640.0, fy=640.0, cx=320.0, cy=240.0)
COLUMNS_X = np.array([-0.375, -0.125, 0.125, 0.375])
START = np.ones((3, 4))
COVERED = np.ones((3, 4), dtype=bool)
FRAME = Candidate(name="frame.jpg", pick_number=0, frame_time=0.0, world_to_camera=np.eye(4))


def _slid(name: str, baseline: float, pick: int, seconds: float) -> Candidate:
    return Candidate(name=name, pick_number=pick, frame_time=seconds,
                     world_to_camera=_pose(np.eye(3), (-baseline, 0.0, 0.0)))


def _expected(baseline: float) -> tuple[float, float, float]:
    """A slid candidate's Overlap, Image_Motion and Source_Rank. A column's points land at
    u = 640·(x − b) + 320, inside when −0.5 ≤ x − b < 0.5, and every row lands inside. Every
    point is at z = 1, so each moves fx·b·δ / (1 + δ)."""
    shifted = COLUMNS_X - baseline
    share = float(np.mean((shifted >= -0.5) & (shifted < 0.5)))
    motion = 640.0 * abs(baseline) * DELTA / (1 + DELTA)
    return share, motion, share * min(motion, MOTION_CAP)


# Each b's Overlap × min(Image_Motion, 3), by _expected:
#   0.05: 1 × 0.317 (motion below source_min_motion, 0.4)   0.1: 1 × 0.634 = 0.634
#   0.2: 0.75 × 1.267 = 0.950                                 0.3: 0.75 × 1.901 = 1.426
#   0.4: 0.5 × 2.535 = 1.267                                  0.6: 0.5 × min(3.802, 3) = 1.5
#   0.7: Overlap 0.25, below source_min_overlap, 0.3
def _choose(candidates: list[Candidate]) -> tuple[list[SourceChoice], float]:
    """choose_sources for FRAME, with the real [walls] settings."""
    return choose_sources(FRAME, START, COVERED, WIDE, (640, 480), candidates, SOURCE_SETTINGS)


def test_choose_sources_filters_ranks_and_spaces() -> None:
    candidates = [
        _slid("close.jpg", 0.6, 10, 0.8),   # under source_min_seconds (1.0) from the frame
        _slid("narrow.jpg", 0.7, 20, 5.0),  # Overlap below 0.3
        _slid("still.jpg", 0.05, 22, 6.0),  # Image_Motion below 0.4
        _slid("a.jpg", 0.3, 30, 1.0),       # exactly source_min_seconds away: a candidate
        _slid("b.jpg", 0.3, 26, 8.0),       # a's tie, with a lower Pick_Number
        _slid("c.jpg", 0.4, 40, 9.0),
        _slid("d.jpg", 0.2, 41, 10.0),      # 1 Pick_Number from c, which ranks above it
        _slid("e.jpg", 0.1, 50, 11.0),
        _slid("f.jpg", 0.6, 60, 12.0),
        _slid(FRAME.name, 0.3, 70, 13.0),   # the frame itself is never its own source
    ]
    chosen, seconds = _choose(candidates)
    assert [choice.candidate.name for choice in chosen] == ["f.jpg", "b.jpg", "a.jpg", "c.jpg",
                                                            "e.jpg"]
    assert seconds == WALLS["source_min_seconds"]
    baselines = {"f.jpg": 0.6, "b.jpg": 0.3, "a.jpg": 0.3, "c.jpg": 0.4, "e.jpg": 0.1}
    for choice in chosen:
        share, motion, rank = _expected(baselines[choice.candidate.name])
        assert choice.overlap == pytest.approx(share, rel=1e-12)
        assert choice.image_motion == pytest.approx(motion, rel=1e-9)
        assert choice.rank == pytest.approx(rank, rel=1e-9)


FAR = [_slid("f1.jpg", 0.6, 20, 5.0), _slid("f2.jpg", 0.3, 30, 6.0),
       _slid("f3.jpg", 0.4, 40, 7.0), _slid("f4.jpg", 0.1, 50, 8.0)]
CLOSE = _slid("close.jpg", 0.2, 10, 0.75)      # within source_min_seconds, not the fallback's 0.5
TOO_CLOSE = _slid("too-close.jpg", 0.6, 4, 0.25)  # within both, though it would rank first


@pytest.mark.parametrize("far, names, seconds", [
    (4, ["f1.jpg", "f2.jpg", "f3.jpg", "f4.jpg"], WALLS["source_min_seconds"]),
    (3, ["f1.jpg", "f2.jpg", "f3.jpg", "close.jpg"], WALLS["source_fallback_seconds"]),
    (2, ["f1.jpg", "f2.jpg", "close.jpg"], WALLS["source_fallback_seconds"]),
], ids=["min-sources-no-fallback", "fallback", "fallback-still-short"])
def test_choose_sources_falls_back_only_when_short(far: int, names: list[str],
                                                   seconds: float) -> None:
    assert WALLS["min_sources"] == 4  # the counts above are built around it
    chosen, used = _choose([CLOSE, TOO_CLOSE, *FAR[:far]])
    assert [choice.candidate.name for choice in chosen] == names
    assert used == seconds


def test_choose_sources_stops_at_max_sources() -> None:
    # 40 candidates with the same pose, so the same rank, Pick_Numbers 2 apart, given in
    # reverse: the tie goes to the lower Pick_Number, a gap of exactly source_min_gap is
    # allowed, and the choice stops at max_sources.
    assert (WALLS["max_sources"], WALLS["source_min_gap"]) == (32, 2)
    candidates = [_slid(f"{number:06d}.jpg", 0.3, 2 * number, 1.0 + number)
                  for number in range(40, 0, -1)]
    chosen, used = _choose(candidates)
    assert [choice.candidate.pick_number for choice in chosen] == list(range(2, 66, 2))
    assert used == WALLS["source_min_seconds"]
