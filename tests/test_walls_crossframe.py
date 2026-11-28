"""Tests for the Walls_Step's first cross-frame stages and its per-frame choice:
splatpipe/walls/crossframe.py (walls spec, Req 14 to 17, 22.1, D14, A11, A14, C2, C5).

The unit tests (sub-task 7.2):
- neighbours runs on hand-made Pick_Number maps.
- projected_values runs on the box room of tests/camera_files.py at a Prior_Size of 64 x 48, half
  its camera's 128 x 96, where a frame seen from its own pose gives back its own depth and a
  neighbouring frame gives values close to it, and on a hand-made 8 x 6 camera whose neighbour
  stands ahead of it, so several of the neighbour's points land on one pixel. That case's oracle
  projects each point with its own loops and keeps the smallest z per pixel.
- agreeing, agreeing_median, fuse_frame, correct_frame, despeckle_frame and sparse_error run on
  small hand arrays whose answers are worked out here, away from every tolerance's edge.
- The order test runs the fuse, the correction and the despeckle over four box-room frames in two
  orders, each frame reading the other frames' depths as the stage before left them (Req 14.4).
- frame_choice runs on each row of Req 17's table.

The property test for Property 6 (sub-task 7.3) goes in its own section at the end.

Every oracle is written here and never calls the function it checks. The settings come from the
real config.toml, read by path, through read_settings on its `[walls]` and `[walls.default]`, and
a SimpleNamespace stands in for them where a test sets its own. Everything runs on the CPU on a
few thousand numbers (Req 22.1).
"""

from __future__ import annotations

import math
import statistics
import struct
import tomllib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st
from hypothesis.extra.numpy import arrays

from splatpipe.train.colmap import Camera
from splatpipe.walls.crossframe import (DEPTH_PRIOR, FALLBACK, NO_FALLBACK, PASSED, SPARSE_ERROR,
                                        TIMED_OUT, agreeing, agreeing_median, correct_frame,
                                        despeckle_frame, frame_choice, fuse_frame, neighbours,
                                        projected_values, sparse_error)
from splatpipe.walls.settings import WallsSettings, read_settings

from camera_files import box_camera, box_covered, box_depth, box_pose

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_WALLS = tomllib.loads((REPO_ROOT / "config.toml").read_text(encoding="utf-8"))["walls"]


def real_settings(preset: str = "default") -> WallsSettings:
    """The real `[walls]` settings with the Preset table `preset`, as the step reads them."""
    shared = {key: value for key, value in CONFIG_WALLS.items() if not isinstance(value, dict)}
    return read_settings({**shared, **CONFIG_WALLS[preset]}, "walls", preset)


SETTINGS = real_settings()
PRIOR_SIZE = (64, 48)   # (w, h), half the box room's camera, to keep the tests fast
NAN = math.nan


def same_bits(a: np.ndarray, b: np.ndarray) -> bool:
    """Whether two float64 arrays have the same shape and the same bits, NaN included."""
    a = np.ascontiguousarray(a, dtype=np.float64)
    b = np.ascontiguousarray(b, dtype=np.float64)
    return a.shape == b.shape and a.tobytes() == b.tobytes()


# ---------------------------------------------------------------------------------------------
# The Neighbour_Frames (Req 14.1, D14, A11)


def test_the_real_neighbour_settings_are_the_decided_ones() -> None:
    assert (SETTINGS.neighbour_window, SETTINGS.neighbour_frames) == (24, 12)


def test_neighbours_go_nearest_first_and_a_tie_to_the_lower_pick_number() -> None:
    trusted = {"f": 10, "x": 13, "y": 8, "z": 11, "w": 7}
    got = neighbours("f", 10, trusted, SETTINGS.neighbour_window, SETTINGS.neighbour_frames)
    # Distances: z 1, y 2, w 3 and x 3, a tie that goes to w's lower Pick_Number 7.
    assert got == ["z", "y", "w", "x"]


def test_neighbours_leave_out_frames_past_the_window_and_the_frame_itself() -> None:
    window = SETTINGS.neighbour_window
    trusted = {"self": 30, "edge_low": 30 - window, "edge_high": 30 + window,
               "past_low": 30 - window - 1, "past_high": 30 + window + 1}
    got = neighbours("self", 30, trusted, window, SETTINGS.neighbour_frames)
    assert got == ["edge_low", "edge_high"]


def test_neighbours_leave_out_failed_frames() -> None:
    picks = {f"{number:06d}.jpg": number for number in range(10)}
    failed = {"000004.jpg", "000006.jpg", "000003.jpg"}
    trusted = {name: number for name, number in picks.items() if name not in failed}
    # A Failed_Frame asks for its own Neighbour_Frames too, so the frame needn't be in the map.
    got = neighbours("000005.jpg", 5, trusted, SETTINGS.neighbour_window,
                     SETTINGS.neighbour_frames)
    assert got == ["000007.jpg", "000002.jpg", "000008.jpg", "000001.jpg", "000009.jpg",
                   "000000.jpg"]
    assert not failed & set(got)


def test_neighbours_stop_at_neighbour_frames() -> None:
    count = SETTINGS.neighbour_frames
    trusted = {f"frame{number}": number for number in range(100)}
    got = neighbours("frame50", 50, trusted, SETTINGS.neighbour_window, count)
    expected = []
    for distance in range(1, count):
        expected += [f"frame{50 - distance}", f"frame{50 + distance}"]
    assert got == expected[:count]
    assert len(got) == count


# ---------------------------------------------------------------------------------------------
# The Projected_Values (Req 14.2, A11)


def test_a_frame_projected_from_its_own_pose_gives_its_own_depth() -> None:
    depth = box_depth(4, PRIOR_SIZE)
    covered = box_covered(4, PRIOR_SIZE)
    values = projected_values(box_pose(4), covered, [depth], [box_pose(4)], box_camera())
    width, height = PRIOR_SIZE
    assert values.shape == (height, width, 1)
    assert np.array_equal(np.isfinite(values[..., 0]), covered)
    np.testing.assert_allclose(values[covered, 0], depth[covered], rtol=1e-12)


def test_each_neighbour_gives_its_own_layer_close_to_the_true_depth_and_nan_elsewhere() -> None:
    depth = box_depth(4, PRIOR_SIZE)
    covered = box_covered(4, PRIOR_SIZE)
    empty = np.full_like(depth, NAN)
    values = projected_values(box_pose(4), covered, [box_depth(5, PRIOR_SIZE), empty],
                              [box_pose(5), box_pose(3)], box_camera())
    near, nothing = values[..., 0], values[..., 1]
    landed = np.isfinite(near)
    # Frame 5's points land on most of frame 4's Covered_Pixels, each value within 1% of the true
    # depth there (the pixel centres differ, so the values aren't exact), and on no other pixel.
    assert not (landed & ~covered).any()
    assert landed.sum() > covered.sum() / 2
    assert (landed.sum() < covered.sum())   # some pixels get nothing
    assert np.all(np.abs(near[landed] / depth[landed] - 1) < 0.01)
    # A neighbour with no depth anywhere gives NaN everywhere.
    assert np.isnan(nothing).all()


# A hand camera whose Prior_Size is its own size, and a neighbour one unit ahead of the frame
# along its optical axis, so its points crowd toward the frame's centre and several land on one
# pixel.
HAND_CAMERA = Camera(camera_id=1, width=8, height=6, fx=8.0, fy=8.0, cx=4.0, cy=3.0)
FRAME_POSE = np.eye(4)
AHEAD_POSE = np.array([[1.0, 0.0, 0.0, 0.0],
                       [0.0, 1.0, 0.0, 0.0],
                       [0.0, 0.0, 1.0, -1.0],
                       [0.0, 0.0, 0.0, 1.0]])   # world-to-camera, the centre at world z = 1


def _ahead_depth() -> np.ndarray:
    """The neighbour's depth, rising across and down so the points landing together differ."""
    rows, columns = np.mgrid[0:HAND_CAMERA.height, 0:HAND_CAMERA.width]
    return 1.0 + 0.1 * columns + 0.05 * rows


def _ahead_oracle(depth: np.ndarray) -> tuple[dict[tuple[int, int], float], int]:
    """Each frame pixel's smallest landing z from the neighbour ahead, and the most points that
    land on one pixel, with plain loops."""
    camera = HAND_CAMERA
    landings: dict[tuple[int, int], list[float]] = {}
    for row in range(camera.height):
        for column in range(camera.width):
            d = float(depth[row, column])
            x = ((column + 0.5) - camera.cx) / camera.fx
            y = ((row + 0.5) - camera.cy) / camera.fy
            # The neighbour's camera point (x d, y d, d) is the world point (x d, y d, d + 1),
            # and the frame's camera is the world's.
            px, py, pz = x * d, y * d, d + 1.0
            u, v = camera.fx * px / pz + camera.cx, camera.fy * py / pz + camera.cy
            if 0 <= u < camera.width and 0 <= v < camera.height:
                landings.setdefault((int(math.floor(v)), int(math.floor(u))), []).append(pz)
    return ({pixel: min(zs) for pixel, zs in landings.items()},
            max(len(zs) for zs in landings.values()))


def test_a_neighbour_gives_each_pixel_its_nearest_point_and_nothing_off_the_covered_pixels(
) -> None:
    depth = _ahead_depth()
    nearest, most = _ahead_oracle(depth)
    assert most >= 2   # the case does put several points on one pixel
    covered = np.ones((HAND_CAMERA.height, HAND_CAMERA.width), dtype=bool)
    uncovered = next(iter(nearest))
    covered[uncovered] = False
    values = projected_values(FRAME_POSE, covered, [depth], [AHEAD_POSE], HAND_CAMERA)[..., 0]
    for row in range(HAND_CAMERA.height):
        for column in range(HAND_CAMERA.width):
            if (row, column) in nearest and covered[row, column]:
                assert values[row, column] == pytest.approx(nearest[(row, column)], rel=1e-12)
            else:
                assert math.isnan(values[row, column])


def test_projected_values_refuse_a_missing_pose() -> None:
    with pytest.raises(ValueError):
        projected_values(FRAME_POSE, np.ones((6, 8), dtype=bool), [_ahead_depth()], [],
                         HAND_CAMERA)


# ---------------------------------------------------------------------------------------------
# The Agreeing_Values and their median (Req 15.1, C2)


def test_the_real_agree_settings_are_the_decided_ones() -> None:
    assert (SETTINGS.agree_tolerance, SETTINGS.agree_min_frames) == (0.03, 2)


def test_agreeing_keeps_the_values_within_the_tolerance_of_the_smallest() -> None:
    values = np.array([[2.07, 2.05, NAN, 2.0, 2.059],
                       [NAN, NAN, NAN, NAN, NAN]])
    got = agreeing(values, SETTINGS.agree_tolerance)
    # The smallest is 2.0, the nearest to the camera, wherever it sits: 2.05 (2.5%) and 2.059
    # (2.95%) agree with it, and 2.07 (3.5%) doesn't. A pixel with no values keeps none.
    assert same_bits(got, np.array([[NAN, 2.05, NAN, 2.0, 2.059], [NAN] * 5]))


def test_agreeing_median_needs_agree_min_frames_agreeing_values() -> None:
    values = np.array([[3.0, 3.03, 5.0],    # two agree: their median, (3.0 + 3.03) / 2
                       [3.0, 3.3, 5.0],     # only 3.0 agrees with itself
                       [3.0, 3.03, 3.06],   # three agree: the middle one
                       [NAN, NAN, NAN]])
    got = agreeing_median(values, SETTINGS.agree_tolerance, SETTINGS.agree_min_frames)
    assert got.shape == (4,)
    assert got[0] == pytest.approx(statistics.median([3.0, 3.03]), rel=1e-15)
    assert math.isnan(got[1])
    assert got[2] == 3.03
    assert math.isnan(got[3])
    # With agree_min_frames at 3 only the third pixel has enough.
    three = agreeing_median(values, SETTINGS.agree_tolerance, 3)
    assert np.isnan(three[[0, 1, 3]]).all() and three[2] == 3.03


# ---------------------------------------------------------------------------------------------
# The fuse (Req 14.3)


def test_the_real_fuse_settings_are_the_decided_ones() -> None:
    assert (SETTINGS.fuse_tolerance, SETTINGS.fuse_min_values) == (0.1, 3)


def test_fuse_takes_the_median_of_enough_close_values_and_keeps_the_own_depth_otherwise() -> None:
    own = np.array([[2.0, 2.0, 2.0, 2.0, NAN, 2.0]])
    projected = np.array([[[2.1, 2.15, NAN],      # with the own depth, 3 values: their median
                           [2.1, 2.15, 2.18],     # 4 values: the two middle ones' mean
                           [2.1, 3.0, NAN],       # 2.0 and 2.1 only, under fuse_min_values
                           [2.3, 2.4, 1.7],       # every value more than 10% off
                           [2.0, 2.0, 2.0],       # no own depth: nothing to fuse
                           [1.85, 2.25, 1.95]]])  # 1.85 and 1.95 are close, 2.25 isn't
    before = own.copy()
    got = fuse_frame(own, projected, SETTINGS)
    expected = [statistics.median([2.0, 2.1, 2.15]), statistics.median([2.0, 2.1, 2.15, 2.18]),
                2.0, 2.0, NAN, statistics.median([2.0, 1.85, 1.95])]
    assert same_bits(got[0], np.array(expected))
    assert same_bits(own, before)   # the stage writes a new array


def test_fuse_counts_the_own_depth_among_fuse_min_values() -> None:
    own = np.array([[2.0]])
    projected = np.array([[[2.1, NAN]]])
    two = SimpleNamespace(fuse_tolerance=SETTINGS.fuse_tolerance, fuse_min_values=2)
    assert fuse_frame(own, projected, two)[0, 0] == statistics.median([2.0, 2.1])
    assert fuse_frame(own, projected, SETTINGS)[0, 0] == 2.0


# ---------------------------------------------------------------------------------------------
# The correction (Req 16.1, 16.6)


def test_the_real_correction_tolerance_is_the_decided_one() -> None:
    assert SETTINGS.correction_tolerance == 0.03


def test_correct_moves_only_the_pixels_more_than_the_tolerance_off_in_log_depth() -> None:
    own = np.array([[2.0, 2.0, 2.0, 2.0, NAN, 2.0]])
    projected = np.array([[[2.2, 2.21],    # M = 2.205, |ln(2 / 2.205)| ≈ 0.098: moves
                           [2.04, 2.05],   # M = 2.045, |ln(2 / 2.045)| ≈ 0.022: stays
                           [2.5, NAN],     # one value, under agree_min_frames: stays
                           [2.5, 3.0],     # 3.0 doesn't agree with 2.5, so one value: stays
                           [2.2, 2.21],    # no own depth: stays NaN
                           [1.8, 1.81]]])  # M = 1.805, |ln(2 / 1.805)| ≈ 0.103: moves down
    before = own.copy()
    got = correct_frame(own, projected, SETTINGS)
    assert abs(math.log(2.0 / statistics.median([2.2, 2.21]))) > SETTINGS.correction_tolerance
    assert abs(math.log(2.0 / statistics.median([2.04, 2.05]))) < SETTINGS.correction_tolerance
    expected = [statistics.median([2.2, 2.21]), 2.0, 2.0, 2.0, NAN,
                statistics.median([1.8, 1.81])]
    assert same_bits(got[0], np.array(expected))
    assert same_bits(own, before)


# ---------------------------------------------------------------------------------------------
# The despeckle (Req 16.4, 16.5)


def test_the_real_despeckle_settings_are_the_decided_ones() -> None:
    assert (SETTINGS.despeckle_window, SETTINGS.despeckle_min_valid,
            SETTINGS.despeckle_tolerance) == (5, 9, 0.03)


def _despeckle_oracle(depth: np.ndarray, covered: np.ndarray, window: int, min_valid: int,
                      tolerance: float) -> np.ndarray:
    """Req 16.4 with plain loops, every window read from `depth` as it came in."""
    height, width = depth.shape
    half = window // 2
    out = depth.copy()
    for row in range(height):
        for column in range(width):
            if not covered[row, column]:
                continue
            valid = [float(depth[r, c])
                     for r in range(max(row - half, 0), min(row + half + 1, height))
                     for c in range(max(column - half, 0), min(column + half + 1, width))
                     if covered[r, c] and math.isfinite(depth[r, c]) and depth[r, c] > 0]
            if len(valid) < min_valid:
                continue
            median = statistics.median(valid)
            if abs(float(depth[row, column]) / median - 1) > tolerance:
                out[row, column] = median
    return out


def _despeckle(depth: np.ndarray, covered: np.ndarray) -> np.ndarray:
    return despeckle_frame(depth, covered, SETTINGS.despeckle_window,
                           SETTINGS.despeckle_min_valid, SETTINGS.despeckle_tolerance)


def test_despeckle_moves_a_lone_spike_to_its_windows_median() -> None:
    depth = np.full((7, 7), 2.0)
    depth[3, 3] = 3.0
    covered = np.ones((7, 7), dtype=bool)
    before = depth.copy()
    got = _despeckle(depth, covered)
    assert same_bits(got, np.full((7, 7), 2.0))
    assert same_bits(depth, before)


def test_despeckle_reads_every_window_from_before_the_stage() -> None:
    # Two spikes side by side, and a few uncovered pixels, so a change made in place would move
    # its neighbours' medians.
    rows, columns = np.mgrid[0:7, 0:7]
    depth = 2.0 + 0.004 * rows + 0.003 * columns
    depth[3, 3], depth[3, 4], depth[5, 1] = 3.0, 2.9, 1.5
    covered = np.ones((7, 7), dtype=bool)
    covered[0, 6] = covered[6, 0] = False
    got = _despeckle(depth, covered)
    expected = _despeckle_oracle(depth, covered, SETTINGS.despeckle_window,
                                 SETTINGS.despeckle_min_valid, SETTINGS.despeckle_tolerance)
    assert not same_bits(expected, depth)   # the case does move pixels
    assert same_bits(got, expected)


def test_despeckle_cuts_the_window_at_the_border() -> None:
    rows, columns = np.mgrid[0:6, 0:6]
    depth = 2.0 + 0.01 * (rows + columns)
    depth[0, 0] = 5.0
    covered = np.ones((6, 6), dtype=bool)
    got = _despeckle(depth, covered)
    # The corner's window is cut to rows 0 to 2 and columns 0 to 2: 9 valid pixels, as many as
    # despeckle_min_valid, whose r + c are 1, 1, 2, 2, 2, 3, 3, 4 and the spike, so its median,
    # the 5th, has r + c = 2.
    assert got[0, 0] == 2.0 + 0.01 * 2
    assert same_bits(got[1:, :], depth[1:, :]) and same_bits(got[0, 1:], depth[0, 1:])


def test_despeckle_leaves_a_pixel_whose_window_has_too_few_valid_pixels() -> None:
    rows, columns = np.mgrid[0:6, 0:6]
    depth = 2.0 + 0.01 * (rows + columns)
    depth[0, 0] = 5.0
    covered = np.ones((6, 6), dtype=bool)
    covered[2, 2] = False   # the corner's cut window now has 8 valid pixels, under 9
    got = _despeckle(depth, covered)
    assert got[0, 0] == 5.0
    # An uncovered pixel never moves, NaN or not.
    depth[4, 4] = NAN
    covered[4, 4] = False
    assert math.isnan(_despeckle(depth, covered)[4, 4])


def test_despeckle_refuses_an_even_window() -> None:
    with pytest.raises(ValueError):
        despeckle_frame(np.ones((4, 4)), np.ones((4, 4), dtype=bool), 4, 1, 0.03)


# ---------------------------------------------------------------------------------------------
# The order the frames go through a stage (Req 14.4, 16.5, A11)

ORDER_FRAMES = (4, 5, 6, 7)


def _ripple(index: int, depth: np.ndarray, size: float) -> np.ndarray:
    """`depth` with a fixed ripple of ±`size` on every 3rd and 5th pixel, different per frame."""
    rows, columns = np.mgrid[0:depth.shape[0], 0:depth.shape[1]]
    pattern = (((rows + index) % 3 == 0).astype(float)
               - ((columns + 2 * index) % 5 == 0).astype(float))
    return depth * (1.0 + size * pattern)


def _box_frames() -> dict[str, dict]:
    frames = {}
    for index in ORDER_FRAMES:
        depth = box_depth(index, PRIOR_SIZE)
        frames[f"{index:06d}.jpg"] = {"pick": index, "pose": box_pose(index),
                                      "covered": box_covered(index, PRIOR_SIZE),
                                      "solved": _ripple(index, depth, 0.04)}
    return frames


def _projected(frames: dict[str, dict], name: str, depths: dict[str, np.ndarray]) -> np.ndarray:
    picks = {other: frame["pick"] for other, frame in frames.items()}
    near = neighbours(name, frames[name]["pick"], picks, SETTINGS.neighbour_window,
                      SETTINGS.neighbour_frames)
    return projected_values(frames[name]["pose"], frames[name]["covered"],
                            [depths[other] for other in near],
                            [frames[other]["pose"] for other in near], box_camera())


def _stages(frames: dict[str, dict], order: list[str]) -> list[dict[str, np.ndarray]]:
    """The fuse, the correction and the despeckle over `frames`, each frame taken in `order` and
    reading the other frames' depths as the stage before left them."""
    solved = {name: frame["solved"] for name, frame in frames.items()}
    fused = {name: fuse_frame(solved[name], _projected(frames, name, solved), SETTINGS)
             for name in order}
    walls = {name: _ripple(frames[name]["pick"] + 1, fused[name], 0.1) for name in order}
    corrected = {name: correct_frame(walls[name], _projected(frames, name, fused), SETTINGS)
                 for name in order}
    despeckled = {name: _despeckle(corrected[name], frames[name]["covered"]) for name in order}
    return [fused, corrected, despeckled]


def test_the_order_the_frames_go_through_a_stage_changes_nothing() -> None:
    frames = _box_frames()
    solved = {name: frame["solved"].copy() for name, frame in frames.items()}
    forward = _stages(frames, list(frames))
    backward = _stages(frames, list(reversed(list(frames))))
    for stage_forward, stage_backward in zip(forward, backward):
        for name in frames:
            assert same_bits(stage_forward[name], stage_backward[name])
    # Every stage does move pixels in this case, and no stage touched the depths it read.
    fused, corrected, despeckled = forward
    for name in frames:
        assert not same_bits(fused[name], solved[name])
        assert same_bits(frames[name]["solved"], solved[name])
    assert any(not same_bits(corrected[name], _ripple(frames[name]["pick"] + 1, fused[name], 0.1))
               for name in frames)
    assert any(not same_bits(despeckled[name], corrected[name]) for name in frames)


# ---------------------------------------------------------------------------------------------
# The per-frame choice (Req 17.2 to 17.5, D1, A14, C5)


def test_the_real_choice_settings_are_the_decided_ones() -> None:
    assert (SETTINGS.scale_min_points, SETTINGS.fallback_max_sparse_error) == (10, 0.1)


ENOUGH = SETTINGS.scale_min_points
GOOD = SETTINGS.fallback_max_sparse_error / 2
BAD = SETTINGS.fallback_max_sparse_error * 2


@pytest.mark.parametrize(
    ("trusted", "timed_out", "has_fallback", "error", "points", "expected"),
    [
        pytest.param(True, False, False, None, 0, (PASSED, None), id="trusted"),
        pytest.param(True, False, False, BAD, ENOUGH, (PASSED, None), id="trusted-bad-error"),
        pytest.param(False, False, True, GOOD, ENOUGH, (FALLBACK, None), id="fallback-good"),
        pytest.param(False, False, True, SETTINGS.fallback_max_sparse_error, ENOUGH,
                     (FALLBACK, None), id="fallback-at-the-limit"),
        pytest.param(False, False, True, BAD, ENOUGH, (DEPTH_PRIOR, SPARSE_ERROR),
                     id="fallback-bad-error"),
        pytest.param(False, False, True, NAN, ENOUGH, (DEPTH_PRIOR, SPARSE_ERROR),
                     id="fallback-nan-error"),
        pytest.param(False, False, True, BAD, ENOUGH - 1, (FALLBACK, None),
                     id="fallback-bad-error-too-few-points"),
        pytest.param(False, False, True, None, 0, (FALLBACK, None), id="fallback-no-points"),
        pytest.param(False, False, False, GOOD, ENOUGH, (DEPTH_PRIOR, NO_FALLBACK),
                     id="no-fallback"),
        pytest.param(False, True, False, None, 0, (DEPTH_PRIOR, TIMED_OUT), id="timed-out"),
        pytest.param(False, True, True, GOOD, ENOUGH, (DEPTH_PRIOR, TIMED_OUT),
                     id="timed-out-with-fallback"),
    ],
)
def test_frame_choice_follows_the_table(trusted: bool, timed_out: bool, has_fallback: bool,
                                        error: float | None, points: int,
                                        expected: tuple[str, str | None]) -> None:
    assert frame_choice(trusted, timed_out, has_fallback, error, points, SETTINGS) == expected


def test_frame_choice_refuses_a_timed_out_trusted_frame() -> None:
    with pytest.raises(ValueError):
        frame_choice(True, True, False, None, 0, SETTINGS)


def test_sparse_error_is_the_median_log_ratio_over_the_points_with_depths() -> None:
    point_z = np.array([1.0, 2.0, 4.0, NAN, 3.0, 5.0, -1.0])
    walls_z = np.array([1.1, 2.0, 2.0, 3.0, 0.0, 5.0, 1.0])
    # The 4th point has no z, the 5th no Walls_Depth and the 7th a z behind the camera.
    expected = statistics.median([abs(math.log(1.1 / 1.0)), 0.0, abs(math.log(2.0 / 4.0)), 0.0])
    assert sparse_error(point_z, walls_z) == pytest.approx(expected, rel=1e-15)


def test_sparse_error_is_nan_with_no_point_and_refuses_mismatched_points() -> None:
    assert math.isnan(sparse_error(np.array([NAN, 2.0]), np.array([1.0, NAN])))
    assert math.isnan(sparse_error(np.array([]), np.array([])))
    with pytest.raises(ValueError):
        sparse_error(np.array([1.0, 2.0]), np.array([1.0]))


# ---------------------------------------------------------------------------------------------
# Property 6: the despeckle never moves a steady pixel (sub-task 7.3, Req 16.7)


def _bits(value: float) -> bytes:
    return struct.pack("<d", float(value))


@st.composite
def _despeckle_cases(draw: st.DrawFn) -> tuple[np.ndarray, np.ndarray, int, int, float]:
    """A depth map 3 to 20 pixels wide and high of finite depths above 0, a random Covered mask,
    an odd despeckle_window from 3 to 7, a despeckle_min_valid from 1 to its area and a
    despeckle_tolerance from 0 to 1, sometimes the real one (0.03).

    The depths are a base times a factor per pixel, the factor mostly near 1 or exactly 1 so many
    pixels sit close to their window's median, and sometimes far from it so others don't."""
    height = draw(st.integers(3, 20))
    width = draw(st.integers(3, 20))
    base = draw(st.floats(0.1, 100.0))
    factors = draw(arrays(np.float64, (height, width),
                          elements=st.one_of(st.just(1.0), st.floats(0.97, 1.03),
                                             st.floats(0.2, 5.0))))
    covered = draw(arrays(np.bool_, (height, width)))
    window = draw(st.sampled_from([3, 5, 7]))
    min_valid = draw(st.integers(1, window * window))
    tolerance = draw(st.one_of(st.just(SETTINGS.despeckle_tolerance), st.floats(0.0, 1.0)))
    return base * factors, covered, window, min_valid, tolerance


def _window_valid(depth: np.ndarray, covered: np.ndarray, row: int, column: int,
                  window: int) -> list[float]:
    """The valid pixels of the window centred on (row, column), cut at the border, with loops."""
    height, width = depth.shape
    half = window // 2
    return [float(depth[r, c])
            for r in range(max(row - half, 0), min(row + half + 1, height))
            for c in range(max(column - half, 0), min(column + half + 1, width))
            if covered[r, c] and math.isfinite(depth[r, c]) and depth[r, c] > 0]


@settings(max_examples=100, deadline=None)
@given(_despeckle_cases())
def test_property_6_the_despeckle_never_moves_a_steady_pixel(
        case: tuple[np.ndarray, np.ndarray, int, int, float]) -> None:
    """Feature: walls, Property 6: The despeckle never moves a steady pixel.

    **Validates: Requirements 16.7**

    Every uncovered pixel, every Covered_Pixel whose window has fewer than despeckle_min_valid
    valid pixels, and every Covered_Pixel whose depth z has |z / M − 1| ≤ despeckle_tolerance, M
    its window's median by statistics.median, comes back bit for bit.

    No pixel near the tolerance's edge is left out: statistics.median's even-count median is
    (lo + hi) / 2 in float64, and |z / M − 1| here is the same float64 division, subtraction and
    abs that Req 16.4 states, so the oracle's edge is the requirement's to the last bit, with no
    rounding gap for a near-tie to fall in.
    """
    depth, covered, window, min_valid, tolerance = case
    before = depth.copy()
    got = despeckle_frame(depth, covered, window, min_valid, tolerance)
    assert got.shape == depth.shape
    assert same_bits(depth, before)   # the stage writes a new array
    height, width = depth.shape
    for row in range(height):
        for column in range(width):
            z = float(before[row, column])
            if not covered[row, column]:
                steady = True
            else:
                valid = _window_valid(before, covered, row, column, window)
                steady = (len(valid) < min_valid
                          or abs(z / statistics.median(valid) - 1.0) <= tolerance)
            if steady:
                assert _bits(got[row, column]) == _bits(z), (row, column)
