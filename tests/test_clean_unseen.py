"""Tests for the clean step's Unseen_Filter, splatpipe/clean/unseen.py, and where it runs.

- unseen_keep counts a frame only when a centre is in front of its camera and lands inside the
  frame, on [0, W) x [0, H), and keeps a centre that at least min_seen frames see. The edges are
  checked with exact binary values, so the multiplied-through test is exact there.
- A Hypothesis property checks unseen_keep against the test's own count, which projects each
  centre with a division, for random cameras and centres away from the frame's edges.
- read_views reads a Sparse_Model written with tests/camera_files.py's own writer, refuses a
  missing model file naming it and the camera solve step, and gives the Model_Reader's message
  for a broken one.
- run_filters runs the Unseen_Filter after the Broken_Filter and adds dropped_unseen straight
  after dropped_broken. Unseen Gaussians don't count toward the Removal_Cap, and the limits'
  messages name the unseen filter's drops.

Everything is tiny and on the CPU.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from splatpipe.clean.filters import Kept, run_filters
from splatpipe.clean.limits import check_limits
from splatpipe.clean.unseen import Views, read_views, unseen_keep
from splatpipe.errors import ExpectedFailure
from splatpipe.train.scene import world_to_camera

from camera_files import CameraRecord, ImageRecord, PointRecord, write_model
from splats import clean_settings, gaussians

# A 4 x 3 PINHOLE camera with f = 2, centred at (2, 1.5). A centre at depth 1 straight ahead,
# (x, y, 1), lands on the pixel (2x + 2, 2y + 1.5), so x from -1 up to 1 and y from -0.75 up to
# 0.75 land inside the frame. Every value is exact in binary.
WIDTH, HEIGHT, F, CX, CY = 4, 3, 2.0, 2.0, 1.5


def views(*translations: tuple[float, float, float]) -> Views:
    """Views of frames that look along +z with no turn, one for each translation."""
    count = len(translations)
    return Views(rotations=np.repeat(np.eye(3)[None], count, axis=0),
                 translations=np.array(translations, dtype=np.float64).reshape(count, 3),
                 width=WIDTH, height=HEIGHT, fx=F, fy=F, cx=CX, cy=CY)


def keep(points: list[tuple[float, float, float]], frames: Views, min_seen: int) -> list[bool]:
    """unseen_keep for `points`, as a list."""
    return unseen_keep(np.array(points, dtype=np.float64), frames, min_seen).tolist()


# unseen_keep

def test_a_centre_two_frames_see_is_kept_and_one_only_one_sees_is_dropped_at_2() -> None:
    # The second frame sits 10 to the right, so it sees (10, 0, 1) and the first sees (0, 0, 1).
    two = views((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    one = views((0.0, 0.0, 0.0), (-10.0, 0.0, 0.0))
    assert keep([(0.0, 0.0, 1.0)], two, 2) == [True]
    assert keep([(0.0, 0.0, 1.0)], one, 2) == [False]
    assert keep([(0.0, 0.0, 1.0)], one, 1) == [True]


def test_a_centre_behind_the_camera_or_at_its_depth_is_never_seen() -> None:
    frames = views((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    # Behind it, the centre's pixel would be (2, 1.5), inside, but it's not in front.
    assert keep([(0.0, 0.0, -1.0), (0.0, 0.0, 0.0), (0.0, 0.0, 1e-300)], frames, 1) == [
        False, False, True]


def test_the_frame_holds_its_left_and_top_edges_but_not_its_right_and_bottom() -> None:
    frames = views((0.0, 0.0, 0.0))
    # Pixel u = 0 and v = 0 are inside, u = W = 4 and v = H = 3 are not.
    assert keep([(-1.0, 0.0, 1.0), (1.0, 0.0, 1.0), (0.0, -0.75, 1.0), (0.0, 0.75, 1.0)],
                frames, 1) == [True, False, True, False]


def test_with_no_frames_or_no_centres_nothing_is_seen() -> None:
    assert keep([(0.0, 0.0, 1.0)], views(), 1) == [False]
    assert unseen_keep(np.empty((0, 3)), views((0.0, 0.0, 0.0)), 1).shape == (0,)


def test_a_huge_finite_centre_raises_no_floating_point_error() -> None:
    frames = views((0.0, 0.0, 0.0))
    with np.errstate(all="raise"):  # as the filter tests run (Req 2.11)
        assert keep([(3.4e38, -3.4e38, 1e-38), (1e-45, 0.0, 3.4e38)], frames, 1) == [
            False, True]


# unseen_keep against the test's own count (every input)

unit_quaternions = st.tuples(*[st.floats(-1, 1)] * 4).filter(
    lambda q: 0.1 < float(np.linalg.norm(q)))
coordinates = st.floats(-5, 5, allow_nan=False)
centres = st.lists(st.tuples(coordinates, coordinates, coordinates), min_size=1, max_size=30)
poses = st.lists(st.tuples(unit_quaternions, st.tuples(coordinates, coordinates, coordinates)),
                 min_size=1, max_size=4)


def own_count(points: np.ndarray, frames: Views) -> np.ndarray:
    """How many of `frames` see each of `points`, with each centre projected by a division, and
    every point near an edge, or near its camera's depth of 0, rejected."""
    count = np.zeros(len(points), dtype=np.int64)
    for rotation, translation in zip(frames.rotations, frames.translations):
        for index, point in enumerate(points):
            x, y, z = rotation @ point + translation
            assume(abs(z) > 1e-6)
            u, v = frames.fx * x / z + frames.cx, frames.fy * y / z + frames.cy
            assume(min(abs(u), abs(u - frames.width), abs(v), abs(v - frames.height)) > 1e-6)
            count[index] += z > 0 and 0 <= u < frames.width and 0 <= v < frames.height
    return count


@settings(deadline=None)
@given(centres, poses, st.integers(min_value=1, max_value=4))
def test_unseen_keep_keeps_exactly_the_centres_at_least_min_seen_frames_see(
        points: list[tuple[float, float, float]],
        frame_poses: list[tuple[tuple[float, ...], tuple[float, float, float]]],
        min_seen: int) -> None:
    """Every centre that at least min_seen frames see in front of the camera and inside the frame
    is kept, and no other."""
    matrices = [world_to_camera(np.array(q), np.array(t)) for q, t in frame_poses]
    frames = Views(rotations=np.array([m[:3, :3] for m in matrices]),
                   translations=np.array([m[:3, 3] for m in matrices]),
                   width=WIDTH, height=HEIGHT, fx=F, fy=F, cx=CX, cy=CY)
    array = np.array(points, dtype=np.float64)
    assert unseen_keep(array, frames, min_seen).tolist() == (
        own_count(array, frames) >= min_seen).tolist()


# read_views

CAMERA = CameraRecord(camera_id=1, width=WIDTH, height=HEIGHT, fx=F, fy=F, cx=CX, cy=CY)
IMAGES = (ImageRecord(1, (1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 1, "000000.jpg"),
          ImageRecord(2, (0.0, 1.0, 0.0, 0.0), (1.0, 2.0, 3.0), 1, "000001.jpg"))
POINTS = (PointRecord(1, (0.0, 0.0, 1.0), (0, 0, 0)),)


def test_read_views_gives_each_frames_pose_and_the_shared_camera(tmp_path: Path) -> None:
    write_model(tmp_path, CAMERA, IMAGES, POINTS)
    frames = read_views(tmp_path)
    assert (frames.width, frames.height, frames.fx, frames.fy, frames.cx, frames.cy) == (
        WIDTH, HEIGHT, F, F, CX, CY)
    # The second frame is turned half a turn about x: (x, y, z) goes to (x, -y, -z), then moved.
    assert frames.rotations.tolist() == [np.eye(3).tolist(), np.diag([1.0, -1.0, -1.0]).tolist()]
    assert frames.translations.tolist() == [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]]


@pytest.mark.parametrize("name", ["cameras.bin", "images.bin", "points3D.bin"])
def test_a_missing_model_file_fails_naming_it_and_the_camera_solve_step(
        tmp_path: Path, name: str) -> None:
    write_model(tmp_path, CAMERA, IMAGES, POINTS)
    (tmp_path / name).unlink()
    with pytest.raises(ExpectedFailure) as caught:
        read_views(tmp_path)
    message = str(caught.value)
    assert str(tmp_path / name) in message and "camera solve step" in message


def test_a_broken_model_fails_with_the_model_readers_message(tmp_path: Path) -> None:
    write_model(tmp_path, CAMERA, IMAGES, POINTS, cut={"images.bin": 1})
    with pytest.raises(ExpectedFailure, match="Run the camera solve step on this job again"):
        read_views(tmp_path)


# run_filters and the limits

def test_run_filters_drops_the_unseen_after_the_broken_and_counts_them_next() -> None:
    data = gaussians(20)  # a lattice of centres from 0 to 3 in each coordinate
    data["x"][0] = np.nan  # broken
    data["z"][1:16] = -100.0  # behind both frames: 15 of 20 unseen
    # Both frames 10 behind the lattice, so they see all of it, and a z of -100 is behind them.
    frames = views((0.0, 0.0, 10.0), (0.0, 0.0, 10.0))
    counts: dict[str, int] = {}
    kept = run_filters(data, clean_settings(), counts.__setitem__, views=frames)
    assert list(counts)[:2] == ["dropped_broken", "dropped_unseen"]
    assert (counts["dropped_broken"], counts["dropped_unseen"]) == (1, 15)
    assert kept.dropped_unseen == 15 and kept.rows.tolist() == list(range(16, 20))
    # 16 of 20 dropped, but neither the broken nor the unseen count toward the cap of 70%.
    assert check_limits(20, kept, clean_settings(warn_kept_fraction=0)) is None


def test_without_views_the_unseen_filter_doesnt_run() -> None:
    counts: dict[str, int] = {}
    kept = run_filters(gaussians(20), clean_settings(), counts.__setitem__)
    assert "dropped_unseen" not in counts and kept.dropped_unseen == 0


def test_the_limits_failures_name_the_unseen_filters_drops_and_min_seen_frames() -> None:
    kept = Kept(np.arange(5), dropped_broken=1, dropped_faint=2, dropped_isolated=3,
                dropped_unseen=89)
    with pytest.raises(ExpectedFailure) as caught:
        check_limits(100, kept, clean_settings(min_kept_splats=6))
    message = str(caught.value)
    assert "the unseen filter 89" in message and "min_seen_frames" in message
