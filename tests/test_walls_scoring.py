"""Tests for the Walls_Step's scoring: splatpipe/walls/scoring.py (walls spec, Req 9.2, 9.5 to
9.10, 11.6, 11.8, D11, D24, A1 to A3, A6).

Most tests run on one hand-made scene whose answers can be worked out exactly in the test. A
26 x 20 Score_Size Prior_Frame sits at the origin and sees a fronto-parallel wall at the depth 2
everywhere, and each Source_Frame is the same camera moved along x only, by a shift chosen so the
wall lands `shift` pixels to the right in it. At the Depth_Factor s the warp moves every pixel by
shift / s pixels across, and that's a whole number of pixels for each shift and Table_Scale used
here (shifts of ±2 and 4, scales 0.5, 1 and 2), so cv2.remap reads each warped pixel straight
off one Source pixel and the oracles read the same pixel by plain indexing. The images are seeded
noise: the Prior_Frame's is a window of a wider noise canvas and each Source_Frame's is the
canvas shifted by its shift, plus noise of its own. Moving along x leaves every depth as it is,
so the depth the warp predicts in a Source_Frame is the Prior_Frame's.

Windows of 5 pixels every 6 pixels leave a gap column and row between neighbours, so a pixel can
be put in one window, or in none, and each reason a window or a pair is left out is tested on its
own window, with every other window still in.

The occlusion test is also checked on tests/camera_files.py's plane scene, against each Source
pixel's true depth on the plane, worked out here, with the Start_Depth at 1.1 and 1.25 times it.

The Tile_Score with a standard error of 0 runs on a scene of small whole-number texture that
repeats every 12 pixels across and 6 down, so two Tiles hold the same pixels, every window sum is
exact and the two Tiles' means are the same bits.

Every oracle is written here and never calls the function it checks. The settings the scoring
compares with, `flat_variance`, `occlusion_tolerance`, `fit_bound`, `z_error_factor` and the
pass's scale ranges, come from the real config.toml, read by path. The sizes are tiny ones that
fit the scenes. Everything runs on the CPU on a few hundred pixels (Req 22.1).
"""

from __future__ import annotations

import dataclasses
import math
import tomllib
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest

from splatpipe.train.colmap import Camera
from splatpipe.train.scene import scaled_intrinsics, world_to_camera
from splatpipe.walls.geometry import normalised_coords, warp_maps
from splatpipe.walls.scoring import (ScoreFrame, ScoreSource, Windows, guess_score,
                                     occluded_windows, score_table, score_windows, table_scales,
                                     tile_scores, valid_pairs)

from camera_files import PLANE_CAMERA, PLANE_IMAGES, PLANE_Z

REPO_ROOT = Path(__file__).resolve().parent.parent
WALLS = tomllib.loads((REPO_ROOT / "config.toml").read_text(encoding="utf-8"))["walls"]
FLAT = WALLS["flat_variance"]
TOLERANCE = WALLS["occlusion_tolerance"]
FIT_BOUND = WALLS["fit_bound"]
Z_ERROR = WALLS["z_error_factor"]

# The hand-made scene.
WIDTH, HEIGHT = 26, 20                 # the Score_Size
PRIOR_WIDTH, PRIOR_HEIGHT = 13, 10     # the Source_Frames' Prior_Size, half the Score_Size
SIZE, STEP, TILE = 5, 6, 13            # window_size, window_step and tile_size
HALF = SIZE // 2
DEPTH = 2.0                            # the wall's depth in every camera
FX = 26.0
MARGIN = 4                             # the canvas's extra columns on each side
SCALES = np.array([0.5, 1.0, 2.0])
# The windows' centres: every 6 pixels from 2, keeping the windows inside 26 x 20.
CENTRES = tuple((x, y) for y in (2, 8, 14) for x in (2, 8, 14, 20))
TARGET = CENTRES.index((14, 8))        # window 6, columns 12 to 16 and rows 6 to 10


def _camera(width: int = WIDTH, height: int = HEIGHT) -> Camera:
    return Camera(camera_id=1, width=width, height=height, fx=FX, fy=FX, cx=width / 2,
                  cy=height / 2)


def _frame(height: int = HEIGHT) -> ScoreFrame:
    camera = _camera(WIDTH, height)
    x, y = normalised_coords(camera, WIDTH, height)
    return ScoreFrame(world_to_camera=np.eye(4), k=scaled_intrinsics(camera, WIDTH, height),
                      x=x, y=y)


def _pose(shift: int) -> np.ndarray:
    """A camera moved along x so the wall at DEPTH lands `shift` pixels to the right in it: a
    point (X, Y, Z) lands at fx·(X + t)/Z + cx, so t = shift × DEPTH / fx."""
    pose = np.eye(4)
    pose[0, 3] = shift * DEPTH / FX
    return pose


def _shifted(canvas: np.ndarray, shift: int) -> np.ndarray:
    """The canvas as a Source_Frame `shift` pixels to the right sees it: its pixel j is the
    canvas's column j − shift + MARGIN, so its pixel i + shift is the Prior_Frame's pixel i."""
    return canvas[:, MARGIN - shift:MARGIN - shift + WIDTH].copy()


def _source(name: str, image: np.ndarray, shift: int, half: str = "A") -> ScoreSource:
    prior_shape = (PRIOR_HEIGHT * image.shape[0] // HEIGHT, PRIOR_WIDTH)
    return ScoreSource(name=name, world_to_camera=_pose(shift), high_pass=image,
                       bands=image[np.newaxis], score_mask=np.ones(image.shape, dtype=bool),
                       start_depth=np.full(prior_shape, DEPTH),
                       covered=np.ones(prior_shape, dtype=bool), half=half)


def _noise(seed: int, shape: tuple[int, int], spread: float) -> np.ndarray:
    return np.random.default_rng(seed).normal(0.0, spread, shape)


CANVAS = _noise(1, (HEIGHT, WIDTH + 2 * MARGIN), 3.0)
PRIOR = CANVAS[:, MARGIN:MARGIN + WIDTH].copy()
FRAME = _frame()
ALL = np.ones((HEIGHT, WIDTH), dtype=bool)
WINDOWS = score_windows(PRIOR, ALL, ALL, FRAME.x, FRAME.y, window_size=SIZE, window_step=STEP,
                        tile_size=TILE, flat_variance=FLAT)
SOURCE = _source("right2", _shifted(CANVAS, 2) + _noise(2, (HEIGHT, WIDTH), 1.0), 2)


def _all_but(*left_out: int) -> np.ndarray:
    keep = np.ones(len(CENTRES), dtype=bool)
    keep[list(left_out)] = False
    return keep


def _valid(source: ScoreSource, depth: np.ndarray | None = None,
           scale: float = 1.0) -> np.ndarray:
    """valid_pairs for SOURCE's kind of Source_Frame at `scale`, with the occlusion test made at
    `depth`, the Current_Depth."""
    depth = np.full((HEIGHT, WIDTH), DEPTH) if depth is None else depth
    map_x, map_y, _, inside = warp_maps(depth, scale, FRAME.world_to_camera,
                                        source.world_to_camera, FRAME.k, FRAME.x, FRAME.y,
                                        source_size=source.score_size)
    hidden = occluded_windows(FRAME, WINDOWS, source, depth, TOLERANCE)
    return valid_pairs(WINDOWS, map_x, map_y, inside, source, hidden, window_size=SIZE,
                       flat_variance=FLAT)


# The oracles

def _window(cx: int, cy: int, columns_off: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rows = np.arange(cy - HALF, cy + HALF + 1)
    columns = np.arange(cx - HALF, cx + HALF + 1) + columns_off
    return rows[:, np.newaxis], columns[np.newaxis, :]


def _ncc(first: np.ndarray, second: np.ndarray) -> float:
    a = first - first.mean()
    b = second - second.mean()
    return float(np.sum(a * b) / math.sqrt(float(np.sum(a * a)) * float(np.sum(b * b))))


def _start_pixel(u: float, v: float, width: int, height: int, prior_width: int,
                 prior_height: int) -> tuple[int, int]:
    """The Depth_Step's pixel rule: the Prior_Size (row, column) under (u, v) of the Score_Size."""
    return (min(math.floor(v * prior_height / height), prior_height - 1),
            min(math.floor(u * prior_width / width), prior_width - 1))


def _oracle_occluded(source: ScoreSource, shift: int, cx: int, cy: int) -> bool:
    """Whether window (cx, cy) is occluded for `source`: its centre, warped at the Current_Depth,
    lands at u = cx + 0.5 + shift, v = cy + 0.5, where the predicted depth is DEPTH."""
    height, width = source.high_pass.shape
    u, v = cx + 0.5 + shift, cy + 0.5
    if not 0 <= u < width:
        return True
    row, column = _start_pixel(u, v, width, height, *source.start_depth.shape[::-1])
    start = float(source.start_depth[row, column])
    if not source.covered[row, column] or not math.isfinite(start):
        return True
    return abs(DEPTH - start) > TOLERANCE * start


def _oracle_pair(prior_image: np.ndarray, source: ScoreSource, shift: int, scale: float,
                 cx: int, cy: int) -> float | None:
    """The Window_Score of window (cx, cy) with `source` at `scale`, or None where they aren't a
    Valid_Pair. The warp moves the window shift / scale pixels across, a whole number here."""
    off = shift / scale
    assert off == round(off), "the hand-made scene only uses whole-pixel moves"
    rows, columns = _window(cx, cy, int(round(off)))
    if columns.min() < 0 or columns.max() >= source.high_pass.shape[1]:
        return None
    warped = source.high_pass[rows, columns]
    if not source.score_mask[rows, columns].all() or np.var(warped) < FLAT:
        return None
    if _oracle_occluded(source, shift, cx, cy):
        return None
    return _ncc(prior_image[_window(cx, cy)], warped)


# score_windows (Req 9.2, A2)

def test_score_windows_leaves_out_windows_that_cross_the_border() -> None:
    # On 22 x 16 the grid from 2 every 6 would go on to x = 20 and y = 14, whose windows reach
    # column 22 and row 16, one past the image.
    camera = _camera(22, 16)
    x, y = normalised_coords(camera, 22, 16)
    every = np.ones((16, 22), dtype=bool)
    windows = score_windows(_noise(3, (16, 22), 3.0), every, every, x, y, window_size=SIZE,
                            window_step=STEP, tile_size=TILE, flat_variance=FLAT)
    expected = [(cx, cy) for cy in (2, 8) for cx in (2, 8, 14)]
    assert list(zip(windows.centres_x.tolist(), windows.centres_y.tolist())) == expected
    assert windows.count == 6


def test_score_windows_keeps_every_window_of_a_textured_image() -> None:
    assert list(zip(WINDOWS.centres_x.tolist(), WINDOWS.centres_y.tolist())) == list(CENTRES)
    # Tiles 13 pixels across: 2 Tiles in each row of Tiles, row-major.
    assert WINDOWS.tile.tolist() == [(cy // TILE) * 2 + cx // TILE for cx, cy in CENTRES]
    np.testing.assert_array_equal(WINDOWS.cx, [(cx + 0.5 - 13) / FX for cx, _ in CENTRES])
    np.testing.assert_array_equal(WINDOWS.cy, [(cy + 0.5 - 10) / FX for _, cy in CENTRES])


def test_score_windows_leaves_out_a_non_depth_pixel_a_masked_pixel_and_a_flat_window() -> None:
    high_pass = PRIOR.copy()
    depth_pixel = ALL.copy()
    score_mask = ALL.copy()
    depth_pixel[3, 9] = False         # in window (8, 2)
    score_mask[9, 13] = False         # in window (14, 8)
    high_pass[12:17, 0:5] = 1.5       # window (2, 14), flat: its variance is 0
    depth_pixel[5, 5] = False         # in the gaps between windows, so in none
    score_mask[11, 11] = False
    high_pass[17, :] = 0.0
    windows = score_windows(high_pass, score_mask, depth_pixel, FRAME.x, FRAME.y,
                            window_size=SIZE, window_step=STEP, tile_size=TILE,
                            flat_variance=FLAT)
    expected = [centre for centre in CENTRES if centre not in ((8, 2), (14, 8), (2, 14))]
    assert list(zip(windows.centres_x.tolist(), windows.centres_y.tolist())) == expected


def test_score_windows_keeps_a_window_whose_variance_is_just_above_flat() -> None:
    # A window of ±a in a checkerboard has a variance per pixel of a² × (1 − 1/625), so
    # a = 1.001 × √(flat / (1 − 1/625)) is just above flat_variance and 0.999 × it just below.
    rows, columns = np.indices((SIZE, SIZE))
    checker = np.where((rows + columns) % 2 == 0, 1.0, -1.0)
    edge = math.sqrt(FLAT / (1 - 1 / SIZE ** 4))
    for factor, kept in ((1.001, True), (0.999, False)):
        high_pass = PRIOR.copy()
        high_pass[0:5, 0:5] = factor * edge * checker
        windows = score_windows(high_pass, ALL, ALL, FRAME.x, FRAME.y, window_size=SIZE,
                                window_step=STEP, tile_size=TILE, flat_variance=FLAT)
        assert ((2, 2) in zip(windows.centres_x.tolist(), windows.centres_y.tolist())) is kept


# valid_pairs and the occlusion test (Req 9.5, 9.6, D24, A1, A2)

def test_every_window_is_a_valid_pair_of_the_plain_scene() -> None:
    np.testing.assert_array_equal(_valid(SOURCE), _all_but())


def test_valid_pairs_leaves_out_a_window_with_a_pixel_behind_the_source() -> None:
    # The point of window 6's top-left pixel at depth −2 lies behind the Source_Frame, and still
    # projects to (10, 6), inside its image, so only the test of source_z leaves the pair out.
    depth = np.full((HEIGHT, WIDTH), DEPTH)
    depth[6, 12] = -DEPTH
    map_x, map_y, source_z, inside = warp_maps(depth, 1.0, FRAME.world_to_camera,
                                               SOURCE.world_to_camera, FRAME.k, FRAME.x,
                                               FRAME.y, source_size=SOURCE.score_size)
    assert source_z[6, 12] < 0 and not inside[6, 12]
    assert 0 <= map_x[6, 12] + 0.5 < WIDTH and 0 <= map_y[6, 12] + 0.5 < HEIGHT
    np.testing.assert_array_equal(_valid(SOURCE, depth), _all_but(TARGET))


def test_valid_pairs_leaves_out_a_window_with_a_pixel_outside_the_source() -> None:
    # At s = 0.5 the warp moves every pixel 4 to the right, and the windows at x = 20 reach
    # column 22 + 4 = 26, one past the Source_Frame. The occlusion test is made at the
    # Current_Depth, so it holds at this scale too (A1).
    left_out = [index for index, (cx, _) in enumerate(CENTRES) if cx == 20]
    np.testing.assert_array_equal(_valid(SOURCE, scale=0.5), _all_but(*left_out))


def test_valid_pairs_leaves_out_a_window_that_touches_a_masked_source_pixel() -> None:
    # Prior pixel (13, 7) of window 6 warps to Source pixel (15, 7). Source pixel (13, 7) is what
    # the Prior_Frame's gap column 11 warps to, so masking it leaves every window in.
    mask = SOURCE.score_mask.copy()
    mask[7, 15] = False
    np.testing.assert_array_equal(_valid(dataclasses.replace(SOURCE, score_mask=mask)),
                                  _all_but(TARGET))
    gap = SOURCE.score_mask.copy()
    gap[7, 13] = False
    np.testing.assert_array_equal(_valid(dataclasses.replace(SOURCE, score_mask=gap)),
                                  _all_but())


def test_valid_pairs_leaves_out_a_window_whose_warped_high_pass_is_flat() -> None:
    # Window 6 warps onto Source columns 14 to 18 and rows 6 to 10, which no other window reads.
    high_pass = SOURCE.high_pass.copy()
    high_pass[6:11, 14:19] = 1.5
    flat = dataclasses.replace(SOURCE, high_pass=high_pass, bands=high_pass[np.newaxis])
    np.testing.assert_array_equal(_valid(flat), _all_but(TARGET))


def test_the_warped_centres_land_on_their_own_prior_size_pixels() -> None:
    # Window 6's centre (14, 8) lands at (16.5, 8.5), and 16.5 × 13 / 26 = 8.25, 8.5 × 10 / 20
    # = 4.25: Prior_Size pixel (row 4, column 8). The other centres land on other pixels.
    assert _start_pixel(16.5, 8.5, WIDTH, HEIGHT, PRIOR_WIDTH, PRIOR_HEIGHT) == (4, 8)
    landed = {_start_pixel(cx + 2.5, cy + 0.5, WIDTH, HEIGHT, PRIOR_WIDTH, PRIOR_HEIGHT)
              for cx, cy in CENTRES}
    assert len(landed) == len(CENTRES)


@pytest.mark.parametrize(("start", "occluded"), [(1.8, False), (1.7, True), (2.4, True)])
def test_valid_pairs_leaves_out_a_window_whose_start_depth_disagrees(start: float,
                                                                     occluded: bool) -> None:
    # The warp predicts the depth 2. |2 − 1.8| = 0.2 is within 0.15 × 1.8 = 0.27, while
    # |2 − 1.7| = 0.3 is above 0.255 and |2 − 2.4| = 0.4 above 0.36.
    assert TOLERANCE == 0.15
    start_depth = SOURCE.start_depth.copy()
    start_depth[4, 8] = start
    source = dataclasses.replace(SOURCE, start_depth=start_depth)
    expected = _all_but(TARGET) if occluded else _all_but()
    np.testing.assert_array_equal(_valid(source), expected)
    np.testing.assert_array_equal(_valid(source, scale=2.0), expected)  # at every scale (A1)


def test_valid_pairs_leaves_out_a_window_whose_centre_lands_on_no_covered_pixel() -> None:
    start_depth = SOURCE.start_depth.copy()
    covered = SOURCE.covered.copy()
    start_depth[4, 8] = np.nan
    covered[4, 8] = False
    source = dataclasses.replace(SOURCE, start_depth=start_depth, covered=covered)
    np.testing.assert_array_equal(_valid(source), _all_but(TARGET))


def _plane_pose(index: int) -> np.ndarray:
    image = PLANE_IMAGES[index]
    return world_to_camera(np.array(image.quaternion, dtype=np.float64),
                           np.array(image.translation, dtype=np.float64))


def _plane_depth(pose: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """The z at which each ray (x, y, 1) of the camera `pose` meets the plane z = PLANE_Z."""
    rotation, translation = pose[:3, :3], pose[:3, 3]
    centre = -rotation.T @ translation
    return (PLANE_Z - centre[2]) / (x * rotation[0, 2] + y * rotation[1, 2] + rotation[2, 2])


def test_occluded_windows_on_the_plane_scene_compares_with_the_sources_start_depth() -> None:
    camera = Camera(camera_id=1, width=PLANE_CAMERA.width, height=PLANE_CAMERA.height,
                    fx=PLANE_CAMERA.fx, fy=PLANE_CAMERA.fy, cx=PLANE_CAMERA.cx,
                    cy=PLANE_CAMERA.cy)
    width, height = camera.width, camera.height
    k = scaled_intrinsics(camera, width, height)
    x, y = normalised_coords(camera, width, height)
    prior_pose, source_pose = _plane_pose(0), _plane_pose(1)
    depth = _plane_depth(prior_pose, x, y)
    every = np.ones((height, width), dtype=bool)
    windows = score_windows(_noise(4, (height, width), 3.0), every, every, x, y, window_size=3,
                            window_step=2, tile_size=8, flat_variance=FLAT)
    frame = ScoreFrame(world_to_camera=prior_pose, k=k, x=x, y=y)

    # The oracle: each centre's plane point in the world, seen by the Source_Frame.
    rotation, translation = prior_pose[:3, :3], prior_pose[:3, 3]
    rows, columns = windows.centres_y, windows.centres_x
    z = depth[rows, columns]
    camera_points = np.stack([z * x[rows, columns], z * y[rows, columns], z], axis=-1)
    world = (camera_points - translation) @ rotation
    seen = world @ source_pose[:3, :3].T + source_pose[:3, 3]
    u = camera.fx * seen[:, 0] / seen[:, 2] + camera.cx
    v = camera.fy * seen[:, 1] / seen[:, 2] + camera.cy
    lands = (seen[:, 2] > 0) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    assert 0 < lands.sum() < windows.count  # some centres land in the Source_Frame, some don't

    source_x = (np.arange(width) + 0.5 - camera.cx) / camera.fx
    source_y = (np.arange(height) + 0.5 - camera.cy) / camera.fy
    true = _plane_depth(source_pose, source_x[np.newaxis, :], source_y[:, np.newaxis])
    image = np.zeros((height, width))
    for factor, expected in ((1.0, ~lands), (1.1, ~lands), (1.25, np.ones_like(lands))):
        source = ScoreSource(name="plane1", world_to_camera=source_pose, high_pass=image,
                             bands=image[np.newaxis], score_mask=every,
                             start_depth=factor * true, covered=every, half="A")
        hidden = occluded_windows(frame, windows, source, depth, TOLERANCE)
        np.testing.assert_array_equal(hidden, expected)
    nowhere = ScoreSource(name="plane1", world_to_camera=source_pose, high_pass=image,
                          bands=image[np.newaxis], score_mask=every,
                          start_depth=np.full((height, width), np.nan), covered=~every, half="A")
    assert occluded_windows(frame, windows, nowhere, depth, TOLERANCE).all()


# table_scales (Req 9.7, A3)

@pytest.mark.parametrize(("prefix", "count"),
                         [("first", 94), ("second", 43), ("third", 43), ("facet", 97)])
def test_table_scales_for_the_real_settings(prefix: str, count: int) -> None:
    low, high = WALLS[f"{prefix}_min_scale"], WALLS[f"{prefix}_max_scale"]
    step = WALLS[f"{prefix}_scale_step"]
    assert math.ceil(math.log(high / low) / step) + 1 == count  # the hand count
    scales = table_scales(low, high, step)
    assert scales.shape == (count,)
    assert scales[0] == low and scales[-1] == high
    gaps = np.diff(np.log(scales))
    np.testing.assert_allclose(gaps, math.log(high / low) / (count - 1), rtol=1e-9)
    assert gaps.max() <= step * (1 + 1e-12)


def test_table_scales_on_a_whole_number_of_steps() -> None:
    np.testing.assert_allclose(table_scales(0.5, 2.0, math.log(2.0)), [0.5, 1.0, 2.0],
                               rtol=1e-15)


# score_table (Req 9.8, A1)

def _table_sources() -> list[tuple[ScoreSource, int]]:
    """Half_A: two Source_Frames 2 and 4 pixels to the right, the first with a masked pixel
    in window 6's footprint. Half_B: one 2 pixels to the left, occluded at window 6, whose
    centre lands on Prior_Size pixel (4, 6), and with a masked pixel that window 3 at (20, 2)
    reads at s = 0.5 and s = 1."""
    right2 = _source("right2", _shifted(CANVAS, 2) + _noise(5, (HEIGHT, WIDTH), 1.0), 2, "A")
    mask = right2.score_mask.copy()
    mask[7, 15] = False
    right2 = dataclasses.replace(right2, score_mask=mask)
    right4 = _source("right4", _shifted(CANVAS, 4) + _noise(6, (HEIGHT, WIDTH), 1.0), 4, "A")
    left2 = _source("left2", _shifted(CANVAS, -2) + _noise(7, (HEIGHT, WIDTH), 1.0), -2, "B")
    start_depth = left2.start_depth.copy()
    start_depth[4, 6] = 1.5
    mask = left2.score_mask.copy()
    mask[2, 16] = False
    left2 = dataclasses.replace(left2, start_depth=start_depth, score_mask=mask)
    return [(right2, 2), (right4, 4), (left2, -2)]


def _oracle_table(sources: list[tuple[ScoreSource, int]], half: str | None) -> np.ndarray:
    table = np.zeros((SCALES.shape[0], len(CENTRES)))
    for source, shift in sources:
        if half is not None and source.half != half:
            continue
        for row, scale in enumerate(SCALES):
            for column, (cx, cy) in enumerate(CENTRES):
                score = _oracle_pair(PRIOR, source, shift, float(scale), cx, cy)
                table[row, column] += 0.0 if score is None else score
    return table


def test_score_table_sums_the_window_scores_of_the_valid_pairs() -> None:
    sources = _table_sources()
    settings = {"weights": None, "window_size": SIZE, "flat_variance": FLAT,
                "occlusion_tolerance": TOLERANCE}
    depth = np.full((HEIGHT, WIDTH), DEPTH)
    scored = [source for source, _ in sources]
    table = score_table(FRAME, WINDOWS, scored, depth, SCALES, PRIOR, **settings)
    whole, half_a, half_b = score_table(FRAME, WINDOWS, scored, depth, SCALES, PRIOR, halves=True,
                                        **settings)
    assert table.shape == (3, len(CENTRES))
    np.testing.assert_allclose(table, _oracle_table(sources, None), rtol=0, atol=1e-9)
    np.testing.assert_allclose(half_a, _oracle_table(sources, "A"), rtol=0, atol=1e-9)
    np.testing.assert_allclose(half_b, _oracle_table(sources, "B"), rtol=0, atol=1e-9)
    assert np.array_equal(whole, table)
    assert np.array_equal(half_a + half_b, table)
    # Window 3 at s = 0.5: both Half_A sources leave the image and Half_B's pixel is masked.
    assert table[0, CENTRES.index((20, 2))] == 0.0
    # Window 6 is occluded for Half_B at every scale, and scores in Half_A.
    assert np.all(half_b[:, TARGET] == 0.0)
    assert np.all(half_a[:, TARGET] != 0.0)
    # The true depth scores best: right2 and left2 match the Prior_Frame up to their own noise.
    assert table[1].sum() > table[0].sum() and table[1].sum() > table[2].sum()


# guess_score (Req 9.9, 9.10, D11, A3)

GUESS_TABLE = np.array([[0.3, -0.2, 0.7], [0.9, 0.4, -0.1], [0.5, 0.8, 0.25]])
GUESS_Z = np.ones(3)
GUESS_M = 1.0


def _guess_windows(cx: tuple[float, ...]) -> Windows:
    count = len(cx)
    return Windows(centres_x=np.zeros(count, dtype=np.int64),
                   centres_y=np.zeros(count, dtype=np.int64),
                   tile=np.zeros(count, dtype=np.int64), cx=np.array(cx, dtype=np.float64),
                   cy=np.zeros(count))


CENTRED = _guess_windows((0.0, 0.0, 0.0))


def _guess(change: tuple[float, float, float, float], windows: Windows = CENTRED,
           z: np.ndarray = GUESS_Z) -> float:
    return guess_score(GUESS_TABLE, SCALES, windows, z, change, GUESS_M, FIT_BOUND)


@pytest.mark.parametrize(("a", "row"), [(1.0, 1), (2.0, 0), (0.5, 2)])
def test_guess_score_on_a_table_scale_reads_it_exactly(a: float, row: int) -> None:
    # With b = c = d = 0 the Depth_Factor is 1 / a at every window.
    assert _guess((a, 0.0, 0.0, 0.0)) == float(np.sum(GUESS_TABLE[row]))


def test_guess_score_interpolates_halfway_in_log_scale() -> None:
    # s = √2 is halfway between 1 and 2 in ln(s), and 1/√2 halfway between 0.5 and 1.
    upper = float(np.sum((GUESS_TABLE[1] + GUESS_TABLE[2]) / 2))
    lower = float(np.sum((GUESS_TABLE[0] + GUESS_TABLE[1]) / 2))
    assert _guess((1 / math.sqrt(2), 0.0, 0.0, 0.0)) == pytest.approx(upper, rel=1e-12)
    assert _guess((math.sqrt(2), 0.0, 0.0, 0.0)) == pytest.approx(lower, rel=1e-12)


def test_guess_score_gives_each_window_its_own_factor() -> None:
    # b = 0.2 at x = 0, 2.5 and −5/3 gives s = 1 / (1 + 0.2·x) = 1, 1 / 1.5 and 1 / (2/3), so
    # s = 1, 2/3 and 1.5, each window interpolated on its own.
    windows = _guess_windows((0.0, 2.5, -5 / 3))
    factors = [1.0, 1 / 1.5, 1.5]
    expected = 0.0
    for window, factor in enumerate(factors):
        q = (math.log(factor) - math.log(0.5)) / math.log(2.0)
        lower = min(int(math.floor(q)), 1)
        fraction = q - lower
        expected += (GUESS_TABLE[lower, window] * (1 - fraction)
                     + GUESS_TABLE[lower + 1, window] * fraction)
    assert _guess((1.0, 0.2, 0.0, 0.0), windows) == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("a", [1 / 2.001, 1 / 0.499])
def test_guess_score_rejects_a_factor_outside_the_table_range(a: float) -> None:
    assert _guess((a, 0.0, 0.0, 0.0)) == -math.inf


def test_guess_score_rejects_a_factor_outside_the_range_at_one_window() -> None:
    # b = 0.2 at x = 10 gives s = 1 / 3 there, below 0.5, and s = 1 at the other windows.
    assert _guess((1.0, 0.2, 0.0, 0.0), _guess_windows((0.0, 0.0, 10.0))) == -math.inf


def test_guess_score_rejects_a_factor_that_isnt_finite() -> None:
    # a = 0 with b = c = d = 0 leaves the new inverse depth 0: no depth and no factor.
    assert _guess((0.0, 0.0, 0.0, 0.0)) == -math.inf
    # A window with no Current_Depth has no factor either.
    assert _guess((1.0, 0.0, 0.0, 0.0), z=np.array([1.0, np.nan, 1.0])) == -math.inf
    assert _guess((math.nan, 0.0, 0.0, 0.0)) == -math.inf


@pytest.mark.parametrize("part", [1, 2, 3])
def test_guess_score_rejects_a_number_just_above_fit_bound(part: int) -> None:
    # At x = y = 0, b and c change no factor, and d = ±fit_bound gives 1 / (1 ± 0.2), inside.
    above = np.nextafter(FIT_BOUND, math.inf)
    for sign in (1.0, -1.0):
        at_bound = [1.0, 0.0, 0.0, 0.0]
        at_bound[part] = sign * FIT_BOUND
        assert math.isfinite(_guess(tuple(at_bound)))
        past = [1.0, 0.0, 0.0, 0.0]
        past[part] = sign * above
        assert _guess(tuple(past)) == -math.inf


# tile_scores (Req 11.6, 11.8, A6)

def _oracle_z(prior_image: np.ndarray, sources: list[tuple[ScoreSource, int]],
              centres: tuple[tuple[int, int], ...], factor: float, tile_size: int,
              tile_min_windows: int) -> tuple[float, int]:
    """The Tile_Score against the answer × `factor`, and the Tiles it counted, by A6's rule."""
    pairs: dict[tuple[int, int], list[int]] = defaultdict(lambda: [0, 0])  # windows, pairs
    sums: dict[tuple[int, int], list[float]] = defaultdict(lambda: [0.0, 0.0])
    for cx, cy in centres:
        tile = (cy // tile_size, cx // tile_size)
        count = 0
        for source, shift in sources:
            at_answer = _oracle_pair(prior_image, source, shift, 1.0, cx, cy)
            at_other = _oracle_pair(prior_image, source, shift, factor, cx, cy)
            if at_answer is not None and at_other is not None:
                count += 1
                sums[tile][0] += at_answer
                sums[tile][1] += at_other
        pairs[tile][0] += count > 0
        pairs[tile][1] += count
    counting = [tile for tile, (windows, _) in pairs.items() if windows >= tile_min_windows]
    if len(counting) < 2:
        return math.nan, len(counting)
    means_a = np.array([sums[tile][0] / pairs[tile][1] for tile in counting])
    means_b = np.array([sums[tile][1] / pairs[tile][1] for tile in counting])
    n = len(counting)
    error = Z_ERROR * math.sqrt(np.var(means_a, ddof=1) / n + np.var(means_b, ddof=1) / n)
    return float((means_a.mean() - means_b.mean()) / error), n


def _tile_scores(frame: ScoreFrame, windows: Windows, sources: list[tuple[ScoreSource, int]],
                 prior_image: np.ndarray, height: int, tile_min_windows: int,
                 ) -> tuple[float, float, int, int]:
    return tile_scores(frame, windows, [source for source, _ in sources],
                       np.full((height, WIDTH), DEPTH), prior_image, weights=None, z_scale=2.0,
                       z_error_factor=Z_ERROR, tile_min_windows=tile_min_windows,
                       window_size=SIZE, flat_variance=FLAT, occlusion_tolerance=TOLERANCE)


def test_tile_scores_equal_the_z_test_worked_out_here() -> None:
    # z_scale 2 moves the windows 1 and 4 pixels across, whole numbers. Four Tiles of 13 pixels.
    sources = _table_sources()
    low, high, low_count, high_count = _tile_scores(FRAME, WINDOWS, sources, PRIOR, HEIGHT, 2)
    expected_low, expected_low_count = _oracle_z(PRIOR, sources, CENTRES, 0.5, TILE, 2)
    expected_high, expected_high_count = _oracle_z(PRIOR, sources, CENTRES, 2.0, TILE, 2)
    assert (low_count, high_count) == (expected_low_count, expected_high_count) == (4, 4)
    assert low == pytest.approx(expected_low, rel=1e-9)
    assert high == pytest.approx(expected_high, rel=1e-9)
    assert low > 0 and high > 0  # the true depth beats both wrong ones


def test_tile_scores_count_only_the_tiles_with_tile_min_windows() -> None:
    # With 3, the two Tiles of the bottom row, 2 windows each, drop out.
    sources = _table_sources()
    low, high, low_count, high_count = _tile_scores(FRAME, WINDOWS, sources, PRIOR, HEIGHT, 3)
    assert (low_count, high_count) == (2, 2)
    assert low == pytest.approx(_oracle_z(PRIOR, sources, CENTRES, 0.5, TILE, 3)[0], rel=1e-9)
    assert high == pytest.approx(_oracle_z(PRIOR, sources, CENTRES, 2.0, TILE, 3)[0], rel=1e-9)


def test_tile_scores_give_no_z_with_one_counting_tile() -> None:
    # Tiles of 32 pixels hold the whole 26 x 20 image: one Tile.
    sources = _table_sources()
    windows = score_windows(PRIOR, ALL, ALL, FRAME.x, FRAME.y, window_size=SIZE,
                            window_step=STEP, tile_size=32, flat_variance=FLAT)
    low, high, low_count, high_count = _tile_scores(FRAME, windows, sources, PRIOR, HEIGHT, 2)
    assert math.isnan(low) and math.isnan(high)
    assert (low_count, high_count) == (1, 1)


# The repeating scene: 26 x 14, so two rows of windows at y = 2 and 8, all in one row of Tiles.
REPEAT_HEIGHT = 14
REPEAT_BASE = np.random.default_rng(8).integers(-8, 9, (6, 12)).astype(np.float64)
REPEAT_CANVAS = np.tile(REPEAT_BASE, (3, 3))[:REPEAT_HEIGHT, :WIDTH + 2 * MARGIN]
REPEAT_PRIOR = REPEAT_CANVAS[:, MARGIN:MARGIN + WIDTH].copy()
REPEAT_FRAME = _frame(REPEAT_HEIGHT)


def _repeat_scene() -> tuple[Windows, list[tuple[ScoreSource, int]]]:
    every = np.ones((REPEAT_HEIGHT, WIDTH), dtype=bool)
    windows = score_windows(REPEAT_PRIOR, every, every, REPEAT_FRAME.x, REPEAT_FRAME.y,
                            window_size=SIZE, window_step=STEP, tile_size=TILE,
                            flat_variance=FLAT)
    assert windows.count == 8 and windows.tile.tolist() == [0, 0, 1, 1, 0, 0, 1, 1]
    return windows, [(_source("right2", _shifted(REPEAT_CANVAS, 2), 2), 2)]


def test_a_standard_error_of_zero_with_mean_a_above_mean_b_gives_infinity() -> None:
    # The texture repeats every 12 pixels across and 6 down, so the Tile of the windows at
    # x = 14 and 20 holds the same whole numbers as the Tile at x = 2 and 8, in the same order,
    # and every sum is exact: the two Tiles' means are the same bits. At the answer each warped
    # window is the Prior_Frame's own, and at × 2 it is moved by one pixel and scores less.
    windows, sources = _repeat_scene()
    centres = tuple(zip(windows.centres_x.tolist(), windows.centres_y.tolist()))
    _, high, _, high_count = _tile_scores(REPEAT_FRAME, windows, sources, REPEAT_PRIOR,
                                          REPEAT_HEIGHT, 2)
    assert high_count == 2
    assert high == math.inf
    # The oracle's two Tiles have equal means at each depth, the answer's above the other's.
    source, shift = sources[0]
    at_answer = [_oracle_pair(REPEAT_PRIOR, source, shift, 1.0, cx, cy) for cx, cy in centres]
    at_other = [_oracle_pair(REPEAT_PRIOR, source, shift, 2.0, cx, cy) for cx, cy in centres]
    assert min(at_answer) == pytest.approx(1.0) and max(at_other) < 1.0


@pytest.mark.parametrize("prior_sign", [0.0, -1.0])
def test_a_standard_error_of_zero_with_mean_a_not_above_mean_b_fails(prior_sign: float) -> None:
    # A Prior_Frame image of 0 scores 0 at both depths, so mean_A = mean_B. Its negative scores
    # −1 at the answer and minus the other depth's score there, so mean_A < mean_B. The
    # Valid_Pairs come from the Source_Frame's High_Pass, so they are the same as above.
    windows, sources = _repeat_scene()
    _, high, _, high_count = _tile_scores(REPEAT_FRAME, windows, sources,
                                          prior_sign * REPEAT_PRIOR, REPEAT_HEIGHT, 2)
    assert high_count == 2
    assert math.isnan(high)
