"""Tests for the Walls_Step's fits, passes and self-check: splatpipe/walls/fit.py and
splatpipe/walls/passes.py (walls spec, Req 10, 11, 15.2, 15.3, 22.1, 22.2, A5, A12, C10).

The fits (sub-task 5.4):
- change_fit runs on hand-made Score_Tables over nine Score_Windows whose best guess is known.
  Each window's table is −(j − j_p)² over the Table_Scales j, so it peaks on the Table_Scale j_p.
  The windows' Current_Depths are worked out in the test from a chosen change and the j_p, so the
  chosen change puts every window's Depth_Factor exactly on its peak, scores 0, and beats every
  other guess. guess_score interpolates linearly in ln(s) and the Table_Scales are evenly spaced
  in ln(s), so each window's score is a chord of the parabola, highest exactly on its node.
- tukey_change runs on inverse depths over a small grid of Normalised_Coordinates.

The passes and the self-check (sub-task 5.5):
- The three passes run once, on one frame of the textured plane of tests/camera_files.py, with its
  Start_Depth 1.1 × the true depth, for the First_Pass's scale, the Solved_Change's composition
  and the Third_Pass on the High_Pass.
- The First_Pass's tie, the Halves_Difference and the Self_Check's verdict have exact oracles on
  hand-made inputs: a hand Score_Table for the tie, hand arrays for the Halves_Difference, and a
  Halves_Difference and Tile_Scores handed in for the verdict, each at its limit on its own.
- A guard checks that choose_sources gives every textured-plane frame enough Source_Frames.

Every oracle is written here and never calls the function it checks. The fit settings, the
Table_Ranges and the Tukey settings come from the real config.toml, read by path, and the
settings change_fit takes are read_settings's on its `[walls]` and `[walls.default]`. Everything
runs on the CPU on a few hundred numbers (Req 22.1).
"""

from __future__ import annotations

import math
import statistics
import tomllib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from splatpipe.train.scene import scaled_intrinsics
from splatpipe.walls import passes
from splatpipe.walls.fit import IDENTITY_CHANGE, change_fit, tukey_change
from splatpipe.walls.geometry import normalised_coords
from splatpipe.walls.images import bands, high_pass, luma
from splatpipe.walls.passes import (PassDepth, PriorScore, ThirdPass, band_coherences,
                                    best_scale, first_pass, frame_band_weights,
                                    halves_difference, second_pass, self_check, source_halves,
                                    third_pass)
from splatpipe.walls.scoring import (ScoreFrame, ScoreSource, Windows, score_table,
                                     score_windows, table_scales)
from splatpipe.walls.settings import WallsSettings, read_settings
from splatpipe.walls.sources import choose_sources

from camera_files import (TEXTURED_CAMERA, TEXTURED_FRAMES, TEXTURED_SECONDS, textured_camera,
                          textured_candidate, textured_covered, textured_depth, textured_frame,
                          textured_pose, textured_score_source)

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_WALLS = tomllib.loads((REPO_ROOT / "config.toml").read_text(encoding="utf-8"))["walls"]


def real_settings(preset: str = "default") -> WallsSettings:
    """The real `[walls]` settings with the Preset table `preset`, as the step reads them."""
    shared = {key: value for key, value in CONFIG_WALLS.items() if not isinstance(value, dict)}
    return read_settings({**shared, **CONFIG_WALLS[preset]}, "walls", preset)


SETTINGS = real_settings()


# ---------------------------------------------------------------------------------------------
# The fits (sub-task 5.4): change_fit and tukey_change (Req 10.3, 15.2, 15.3, A5, A12, C10).
# ---------------------------------------------------------------------------------------------

# Nine windows on a 3 × 3 grid of Normalised_Coordinates.
FIT_CX = np.array([-0.3, 0.0, 0.3] * 3)
FIT_CY = np.repeat([-0.3, 0.0, 0.3], 3)
FIT_WINDOWS = Windows(centres_x=np.arange(9, dtype=np.int64),
                      centres_y=np.zeros(9, dtype=np.int64),
                      tile=np.zeros(9, dtype=np.int64), cx=FIT_CX, cy=FIT_CY)


class HandTable:
    """A Score_Table over FIT_WINDOWS whose best guess is known.

    Window p's row is −(j − peaks[p])² over the Table_Scales j of the pass `prefix`. `base` is a
    change (a, b, c, d) relative to m = 1 whose b·x + c·y + d is above 0 at every window and whose
    a is below 1 / s at every peak. Each window's Current_Depth z_p then solves
    1 / s_p = a + z_p·(b·x + c·y + d) for its peak's scale s_p. (b, c, d) and z are rescaled by
    λ and 1 / λ, which leaves z·(b, c, d) and so every Depth_Factor as it is, with λ chosen so the
    median of 1 / z is 1: m = 1 is then the depth map's Median_Inverse_Depth, as change_fit wants.
    `best` is the rescaled change, which puts every window on its peak.
    """

    def __init__(self, prefix: str, base: tuple[float, float, float, float],
                 peaks: list[int]) -> None:
        self.scales = table_scales(CONFIG_WALLS[f"{prefix}_min_scale"],
                                   CONFIG_WALLS[f"{prefix}_max_scale"],
                                   CONFIG_WALLS[f"{prefix}_scale_step"])
        self.peaks = np.asarray(peaks, dtype=np.float64)
        a, b, c, d = base
        line = b * FIT_CX + c * FIT_CY + d
        depth = (1.0 / self.scales[peaks] - a) / line
        assert np.all(line > 0) and np.all(depth > 0)
        rescale = 1.0 / float(np.median(1.0 / depth))
        self.z = depth / rescale
        self.m = 1.0
        assert math.isclose(float(np.median(1.0 / self.z)), self.m, rel_tol=1e-12)
        self.best = (a, b * rescale, c * rescale, d * rescale)
        rows = np.arange(self.scales.shape[0], dtype=np.float64)[:, None]
        self.table = -((rows - self.peaks[None, :]) ** 2)

    def score(self, change: tuple[float, float, float, float]) -> float:
        """The Guess_Score of `change` worked out by hand: each window's Depth_Factor
        1 / (a + m·z·(b·x + c·y + d)), its place between the Table_Scales in ln(s), and the
        straight line between its two neighbouring nodes' values. −∞ outside the Table_Range or
        with |b|, |c| or |d| above `fit_bound`."""
        a, b, c, d = change
        if max(abs(b), abs(c), abs(d)) > SETTINGS.fit_bound:
            return -math.inf
        factor = 1.0 / (a + self.m * self.z * (b * FIT_CX + c * FIT_CY + d))
        low, high = math.log(self.scales[0]), math.log(self.scales[-1])
        if np.any(factor < self.scales[0]) or np.any(factor > self.scales[-1]):
            return -math.inf
        place = (np.log(factor) - low) / (high - low) * (self.scales.shape[0] - 1)
        lower = np.minimum(np.floor(place), self.scales.shape[0] - 2)
        fraction = place - lower
        value_low = -((lower - self.peaks) ** 2)
        value_high = -((lower + 1.0 - self.peaks) ** 2)
        return float(np.sum(value_low * (1.0 - fraction) + value_high * fraction))

    def fit(self) -> tuple[tuple[float, float, float, float], float]:
        """change_fit on this table with the real settings."""
        return change_fit(self.table, self.scales, FIT_WINDOWS, self.z, self.m, SETTINGS)


def inside_table() -> HandTable:
    """A table on the Second_Pass's range whose best guess is inside `fit_bound`."""
    return HandTable("second", (0.97, 0.04, -0.03, 0.05), [6, 8, 10, 7, 9, 11, 5, 12, 9])


def outside_table() -> HandTable:
    """A table on the First_Pass's range whose peak needs a d above `fit_bound`."""
    return HandTable("first", (0.97, 0.15, -0.1, 0.35), [30, 25, 20, 33, 28, 23, 36, 31, 26])


def test_change_fit_finds_the_known_best_guess() -> None:
    """change_fit finds the guess that puts every window on its table's peak, within
    `fit_x_tolerance`, and its Guess_Score, at least the Identity_Change's, is the answer's own
    (Req 10.3, A5)."""
    table = inside_table()
    assert max(abs(number) for number in table.best[1:]) < SETTINGS.fit_bound
    assert table.score(table.best) == pytest.approx(0.0, abs=1e-9)
    identity_score = table.score(IDENTITY_CHANGE)
    assert math.isfinite(identity_score) and identity_score < -1.0

    answer, score = table.fit()

    assert np.max(np.abs(np.subtract(answer, table.best))) < SETTINGS.fit_x_tolerance
    assert score >= identity_score
    assert score == pytest.approx(table.score(answer), abs=1e-9)
    assert max(abs(number) for number in answer[1:]) <= SETTINGS.fit_bound


def test_change_fit_keeps_b_c_and_d_within_the_bound() -> None:
    """When the table's peak needs a d above `fit_bound`, the answer still keeps |b|, |c| and |d|
    within it, and scores above the Identity_Change (Req 10.3, A5)."""
    table = outside_table()
    assert abs(table.best[3]) > SETTINGS.fit_bound
    identity_score = table.score(IDENTITY_CHANGE)
    assert math.isfinite(identity_score)

    answer, score = table.fit()

    assert max(abs(number) for number in answer[1:]) <= SETTINGS.fit_bound
    assert math.isfinite(score) and score > identity_score
    assert score == pytest.approx(table.score(answer), abs=1e-9)


@pytest.mark.parametrize("make_table", [inside_table, outside_table])
def test_change_fit_gives_the_same_bits_twice(make_table) -> None:
    """Two runs on the same table give the same answer and score bit for bit, because the starts
    are a fixed pattern with no random draw (C10, Req 5.9)."""
    first_answer, first_score = make_table().fit()
    second_answer, second_score = make_table().fit()
    assert first_answer == second_answer
    assert first_score == second_score


# A grid of Normalised_Coordinates, 20 across and 15 down.
TUKEY_X, TUKEY_Y = np.meshgrid(np.linspace(-0.5, 0.5, 20), np.linspace(-0.4, 0.4, 15))
TUKEY_CHANGE = (1.1, 0.05, -0.03, 0.02)


def changed_inverse(change: tuple[float, float, float, float], inverse0: np.ndarray,
                    m: float) -> np.ndarray:
    """a·inv0 + m·(b·x + c·y + d) on the grid, worked out by hand."""
    a, b, c, d = change
    return a * inverse0 + m * (b * TUKEY_X + c * TUKEY_Y + d)


def far_off(target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """`target` with a tenth of its inverse depths moved far off, every tenth pixel, alternately
    ×5 and ×0.2, and the mask of the pixels left as they were."""
    moved = target.ravel().copy()
    picked = np.flatnonzero(np.arange(moved.size) % 10 == 3)
    moved[picked[0::2]] *= 5.0
    moved[picked[1::2]] *= 0.2
    kept = np.ones(moved.size, dtype=bool)
    kept[picked] = False
    return moved.reshape(target.shape), kept.reshape(target.shape)


def tukey(inverse0: np.ndarray, target: np.ndarray, m: float) -> tuple[float, float, float, float]:
    """tukey_change on the grid with the real `tukey_constant` and `tukey_iterations`."""
    return tukey_change(inverse0, target, m, TUKEY_X, TUKEY_Y, SETTINGS.tukey_constant,
                        SETTINGS.tukey_iterations)


CURVED = 0.4 + 0.2 * TUKEY_X - 0.1 * TUKEY_Y + 0.3 * TUKEY_X ** 2


def test_tukey_change_recovers_a_change_of_a_curved_surface() -> None:
    """On a curved surface the four numbers are each pinned down: tukey_change gets the exact
    change back from exact targets, and within 10⁻⁶ with a tenth of the targets far off
    (Req 15.2, 15.3, A12).

    This corrects sub-task 5.4's text, which asks for this on a plane tilted to the camera. On
    any plane, tilted or square on, the inverse depth inv0 = (n·r) / δ is itself affine in
    (x, y), so the inv0 column is a combination of the other three and a trades off against
    (b, c, d): the numbers aren't pinned down there. The x² term here keeps inv0 out of the span
    of (x, y, 1).
    """
    m = float(np.median(CURVED))
    target = changed_inverse(TUKEY_CHANGE, CURVED, m)

    exact = tukey(CURVED, target, m)
    assert np.max(np.abs(np.subtract(exact, TUKEY_CHANGE))) < 1e-10

    moved, _ = far_off(target)
    robust = tukey(CURVED, moved, m)
    assert np.max(np.abs(np.subtract(robust, TUKEY_CHANGE))) < 1e-6


def tilted_plane() -> np.ndarray:
    """The inverse depth (n·r) / δ of the plane n·X = δ, tilted to the camera, on the grid."""
    normal = np.array([0.3, -0.2, 1.0]) / np.linalg.norm([0.3, -0.2, 1.0])
    return (normal[0] * TUKEY_X + normal[1] * TUKEY_Y + normal[2]) / 2.0


def square_plane() -> np.ndarray:
    """The inverse depth of a plane seen square on at the depth 2, on the grid."""
    return np.full_like(TUKEY_X, 0.5)


@pytest.mark.parametrize("plane", [tilted_plane, square_plane])
def test_tukey_change_matches_a_plane_depths(plane) -> None:
    """On a plane, tilted or seen square on, a trades off against (b, c, d), so the numbers
    aren't pinned down, but the depths tukey_change gives match the targets within 10⁻⁹ from
    exact targets, and the untouched targets' depths within 10⁻⁹ with a tenth far off
    (Req 15.2, 15.3, A12).

    This corrects sub-task 5.4's text, which says only a plane seen square on trades off: on any
    plane inv0 is affine in (x, y), so its column is a combination of the other three.
    """
    inverse0 = plane()
    m = float(np.median(inverse0))
    target = changed_inverse(TUKEY_CHANGE, inverse0, m)

    exact = tukey(inverse0, target, m)
    depth = 1.0 / changed_inverse(exact, inverse0, m)
    assert np.max(np.abs(depth - 1.0 / target)) < 1e-9

    moved, kept = far_off(target)
    robust = tukey(inverse0, moved, m)
    depth = 1.0 / changed_inverse(robust, inverse0, m)
    assert np.max(np.abs(depth[kept] - 1.0 / target[kept])) < 1e-9


# ---------------------------------------------------------------------------------------------
# The passes and the self-check (sub-task 5.5) (Req 8.6, 10.1, 10.6, 10.8, 11.1–11.3, 11.7, 11.8).
# ---------------------------------------------------------------------------------------------

# The textured plane's Score_Size and Prior_Size, both the camera's own 128 x 96 (see
# tests/camera_files.py), and the Start_Depth's factor on the true depth, as in sub-task 5.6.
TEXTURED_SIZE = (TEXTURED_CAMERA.width, TEXTURED_CAMERA.height)
TEXTURED_START = 1.1
PASS_FRAME = 5     # the one frame whose three passes run here, in the middle of the walk


def textured_prior(index: int, start_scale: float = TEXTURED_START, *,
                   plain: bool = False) -> PriorScore:
    """Textured-plane frame `index` as a Prior_Frame, its Start_Depth `start_scale` × the true
    depth at both sizes, every pixel a Depth_Pixel, kept by the Score_Mask and covered, its
    High_Pass and Bands from the real `high_pass_sigma` and `band_sigmas`, and m worked out here."""
    camera = textured_camera()
    width, height = TEXTURED_SIZE
    x, y = normalised_coords(camera, width, height)
    grey = luma(textured_frame(index, plain=plain))
    covered = textured_covered(index, TEXTURED_SIZE)
    start_depth = start_scale * textured_depth(index, TEXTURED_SIZE)
    frame = ScoreFrame(world_to_camera=textured_pose(index),
                       k=scaled_intrinsics(camera, width, height), x=x, y=y)
    return PriorScore(frame=frame, high_pass=high_pass(grey, SETTINGS.high_pass_sigma),
                      bands=np.stack(bands(grey, SETTINGS.band_sigmas)),
                      score_mask=np.ones((height, width), dtype=bool), depth_pixel=covered,
                      start_depth=start_depth, prior_start_depth=start_depth, covered=covered,
                      prior_x=x, prior_y=y, m=float(np.median(1.0 / start_depth[covered])))


def textured_windows(prior: PriorScore) -> Windows:
    """The Prior_Frame's Score_Windows with the real window, step, Tile and flat settings."""
    return score_windows(prior.high_pass, prior.score_mask, prior.depth_pixel, prior.frame.x,
                         prior.frame.y, window_size=SETTINGS.window_size,
                         window_step=SETTINGS.window_step, tile_size=SETTINGS.tile_size,
                         flat_variance=SETTINGS.flat_variance)


def textured_choice(index: int, start_scale: float = TEXTURED_START) -> tuple[list, float]:
    """choose_sources for textured-plane frame `index`, with its Start_Depth `start_scale` × the
    true depth, every other frame a candidate, and the real settings."""
    start_depth = start_scale * textured_depth(index, TEXTURED_SIZE)
    candidates = [textured_candidate(other) for other in range(TEXTURED_FRAMES)]
    return choose_sources(textured_candidate(index), start_depth,
                          textured_covered(index, TEXTURED_SIZE), textured_camera(),
                          TEXTURED_SIZE, candidates, SETTINGS)


def textured_sources(index: int, *, plain: bool = False) -> list[ScoreSource]:
    """Frame `index`'s Source_Frames as ScoreSources, in rank order, each in its Source_Half, with
    its Start_Depth at the true depth."""
    choices, _ = textured_choice(index)
    return [textured_score_source(choice.candidate.pick_number, half, prior_size=TEXTURED_SIZE,
                                  high_pass_sigma=SETTINGS.high_pass_sigma,
                                  band_sigmas=SETTINGS.band_sigmas, plain=plain)
            for choice, half in zip(choices, source_halves(len(choices)))]


@dataclass(frozen=True)
class Solved:
    """PASS_FRAME's three passes on the textured plane, run once for the module."""
    prior: PriorScore
    windows: Windows
    sources: list[ScoreSource]
    first_scale: float
    first: PassDepth
    second_change: tuple[float, float, float, float]
    second: PassDepth
    weights: np.ndarray
    third: ThirdPass


@pytest.fixture(scope="module")
def solved() -> Solved:
    """The First_Pass, the Second_Pass, the Band_Weights and the Third_Pass of PASS_FRAME."""
    prior = textured_prior(PASS_FRAME)
    windows = textured_windows(prior)
    sources = textured_sources(PASS_FRAME)
    first_scale, first = first_pass(prior, windows, sources, SETTINGS)
    second_change, second = second_pass(prior, windows, sources, first, SETTINGS)
    weights = frame_band_weights(prior, band_coherences(prior, windows, sources, second,
                                                        SETTINGS), SETTINGS)
    third = third_pass(prior, windows, sources, weights, second, SETTINGS)
    return Solved(prior, windows, sources, first_scale, first, second_change, second, weights,
                  third)


FIRST_SCALES = table_scales(SETTINGS.first_min_scale, SETTINGS.first_max_scale,
                            SETTINGS.first_scale_step)


def test_best_scale_takes_the_highest_sum_even_far_from_1() -> None:
    """The Table_Scale with the highest Σ_p C(p, s) wins wherever it lies, and when s and 1/s
    tie exactly the smaller one wins (Req 10.1)."""
    table = np.zeros((FIRST_SCALES.shape[0], 3))
    table[3] = [0.5, 0.2, 0.4]          # 1.1 in all, near the bottom of the range
    table[46] = [0.3, 0.3, 0.3]         # 0.9, the scale nearest 1
    assert best_scale(table, FIRST_SCALES) == FIRST_SCALES[3]

    scales = np.array([0.25, 0.5, 2.0, 4.0])
    tied = np.array([[0.0], [1.0], [1.0], [0.0]])
    assert math.log(0.5) == -math.log(2.0)
    assert best_scale(tied, scales) == 0.5


@pytest.mark.parametrize("near, far", [(50, 40), (44, 52)])
def test_first_pass_takes_the_tied_scale_nearer_1(monkeypatch, near: int, far: int) -> None:
    """With two Table_Scales tied on the highest Σ_p C(p, s), first_pass takes the one nearer 1 in
    ln(s), on either side of 1, and gives s × the Start_Depth with the change (1/s, 0, 0, 0)
    (Req 10.1). The Score_Table is a hand table handed in for score_table, which also checks that
    the First_Pass scores the High_Pass around the Start_Depth on the first_* scales."""
    assert abs(math.log(FIRST_SCALES[near])) < abs(math.log(FIRST_SCALES[far]))
    prior = textured_prior(PASS_FRAME)
    windows = Windows(centres_x=np.array([20, 60, 100]), centres_y=np.array([40, 40, 40]),
                      tile=np.zeros(3, dtype=np.int64), cx=np.zeros(3), cy=np.zeros(3))

    def hand_table(frame, windows_in, sources, current_depth, scales, prior_image, *, weights,
                   halves=False, **_):
        assert np.array_equal(scales, FIRST_SCALES)
        assert prior_image is prior.high_pass and weights is None and not halves
        assert np.array_equal(current_depth, prior.start_depth, equal_nan=True)
        table = np.full((scales.shape[0], windows_in.count), -1.0)
        table[far] = [3.0, 2.0, 1.0]
        table[near] = [1.0, 2.0, 3.0]
        return table

    monkeypatch.setattr(passes, "score_table", hand_table)
    s, depth = first_pass(prior, windows, [], SETTINGS)

    assert s == FIRST_SCALES[near]
    assert depth.change == (1.0 / s, 0.0, 0.0, 0.0)
    assert np.array_equal(depth.score_depth, s * prior.start_depth, equal_nan=True)
    assert np.array_equal(depth.prior_depth, s * prior.prior_start_depth, equal_nan=True)
    covered = prior.covered
    assert depth.m == pytest.approx(float(np.median(1.0 / (s * prior.prior_start_depth[covered]))),
                                    rel=1e-12)


def test_first_pass_finds_the_start_depth_scale_on_the_textured_plane(solved: Solved) -> None:
    """On the textured plane with its Start_Depth 1.1 × the true depth, the First_Pass's scale is
    within one Table_Scale step, in ln(s), of 1 / 1.1 (Req 10.1)."""
    step = math.log(SETTINGS.first_max_scale / SETTINGS.first_min_scale) / (
        FIRST_SCALES.shape[0] - 1)
    assert abs(math.log(solved.first_scale * TEXTURED_START)) <= step
    assert solved.first_scale in FIRST_SCALES


def by_hand(inverse: np.ndarray, change: tuple[float, float, float, float], m: float,
            x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """a·inverse + m·(b·x + c·y + d), worked out here."""
    a, b, c, d = change
    return a * inverse + m * (b * x + c * y + d)


def test_solved_change_is_the_three_changes_in_turn(solved: Solved) -> None:
    """The Solved_Change, applied to the Start_Depth with its m, gives the depth of the three
    passes' changes applied in turn, each with the m of the depth it changes (Req 10.8)."""
    prior, third = solved.prior, solved.third
    covered, x, y = prior.covered, prior.prior_x, prior.prior_y
    inverse0 = 1.0 / prior.prior_start_depth
    m0 = float(np.median(inverse0[covered]))
    assert prior.m == pytest.approx(m0, rel=1e-15)

    first_change = (1.0 / solved.first_scale, 0.0, 0.0, 0.0)
    assert solved.first.change == first_change
    inverse1 = by_hand(inverse0, first_change, m0, x, y)
    m1 = float(np.median(inverse1[covered]))
    inverse2 = by_hand(inverse1, solved.second_change, m1, x, y)
    m2 = float(np.median(inverse2[covered]))
    inverse3 = by_hand(inverse2, third.change, m2, x, y)
    assert solved.first.m == pytest.approx(m1, rel=1e-12)
    assert solved.second.m == pytest.approx(m2, rel=1e-12)

    solved_inverse = by_hand(inverse0, third.solved_change, m0, x, y)
    np.testing.assert_allclose(solved_inverse[covered], inverse3[covered], rtol=1e-12)
    np.testing.assert_allclose(third.answer.prior_depth[covered], 1.0 / inverse3[covered],
                               rtol=1e-12)
    score_inverse = by_hand(1.0 / prior.start_depth, third.solved_change, m0, prior.frame.x,
                            prior.frame.y)
    np.testing.assert_allclose(third.answer.score_depth, 1.0 / score_inverse, rtol=1e-12)


def test_third_pass_scores_the_high_pass_when_every_band_weight_is_0(solved: Solved) -> None:
    """With every Band_Weight 0, the Third_Pass scores the High_Pass: its tables are score_table's
    on the High_Pass of the Prior_Frame and of each Source_Frame, and each Score_Table is the sum
    of its two Half_Tables (Req 10.6, 10.7)."""
    prior = solved.prior
    zeros = np.zeros(len(SETTINGS.band_sigmas))
    third = third_pass(prior, solved.windows, solved.sources, zeros, solved.second, SETTINGS)

    assert third.scored_weights is None
    assert np.array_equal(third.prior_image, prior.high_pass)
    assert np.array_equal(third.band_weights, zeros)
    scales = table_scales(SETTINGS.third_min_scale, SETTINGS.third_max_scale,
                          SETTINGS.third_scale_step)
    expected = score_table(prior.frame, solved.windows, solved.sources,
                           solved.second.score_depth, scales, prior.high_pass, weights=None,
                           window_size=SETTINGS.window_size,
                           flat_variance=SETTINGS.flat_variance,
                           occlusion_tolerance=SETTINGS.occlusion_tolerance, halves=True)
    for built, wanted in zip((third.table, third.half_a, third.half_b), expected):
        assert np.array_equal(built, wanted)
    for made in (third, solved.third):
        np.testing.assert_allclose(made.table, made.half_a + made.half_b, rtol=0, atol=1e-12)


def test_six_ranked_sources_split_into_alternate_halves(solved: Solved) -> None:
    """Six Source_Frames in rank order: Half_A the 1st, 3rd and 5th, Half_B the 2nd, 4th and 6th
    (Req 11.1, D12), and the textured plane's ranked Source_Frames get their halves that way."""
    halves = source_halves(6)
    assert halves == ("A", "B", "A", "B", "A", "B")
    assert [rank for rank, half in enumerate(halves, start=1) if half == "A"] == [1, 3, 5]
    assert [rank for rank, half in enumerate(halves, start=1) if half == "B"] == [2, 4, 6]
    assert source_halves(1) == ("A",) and source_halves(0) == ()
    assert [source.half for source in solved.sources] == [
        "A" if rank % 2 == 0 else "B" for rank in range(len(solved.sources))]


def hand_prior() -> PriorScore:
    """A 3 x 4 Prior_Frame for the Halves_Difference: only its Covered_Pixels and Prior_Size
    Normalised_Coordinates are read."""
    x, y = np.meshgrid(np.linspace(-0.3, 0.3, 4), np.linspace(-0.2, 0.2, 3))
    every = np.ones((3, 4), dtype=bool)
    covered = every.copy()
    covered[0, 0] = covered[2, 3] = False
    return PriorScore(frame=ScoreFrame(world_to_camera=np.eye(4), k=np.eye(3), x=x, y=y),
                      high_pass=np.zeros((3, 4)), bands=np.zeros((1, 3, 4)), score_mask=every,
                      depth_pixel=every, start_depth=np.ones((3, 4)),
                      prior_start_depth=np.ones((3, 4)), covered=covered, prior_x=x, prior_y=y,
                      m=1.0)


def test_halves_difference_is_the_median_worked_out_here() -> None:
    """The Halves_Difference is the median over the Covered_Pixels of |z_A / z_B − 1|, with z_A
    and z_B the current depth through each half's change with its m, a Covered_Pixel with no
    depth counting as infinite, and the pixels off the Covered mask left out (Req 11.3, D12)."""
    prior = hand_prior()
    depth = 2.0 + 0.1 * np.arange(12, dtype=np.float64).reshape(3, 4)
    depth[1, 2] = np.nan          # covered, with no depth
    depth[0, 0] = 1e-6            # off the Covered mask, which would give a wild difference
    current = PassDepth(score_depth=depth, prior_depth=depth, m=0.4, change=IDENTITY_CHANGE)
    change_a = (1.02, 0.01, -0.02, 0.005)
    change_b = (0.99, -0.01, 0.01, 0.0)

    differences = []
    for row in range(3):
        for column in range(4):
            if not prior.covered[row, column]:
                continue
            inverse = 1.0 / depth[row, column]
            x, y = prior.prior_x[row, column], prior.prior_y[row, column]
            z_a = 1.0 / (change_a[0] * inverse + 0.4 * (change_a[1] * x + change_a[2] * y
                                                        + change_a[3]))
            z_b = 1.0 / (change_b[0] * inverse + 0.4 * (change_b[1] * x + change_b[2] * y
                                                        + change_b[3]))
            difference = abs(z_a / z_b - 1.0)
            differences.append(difference if math.isfinite(difference) else math.inf)
    assert len(differences) == 10 and differences.count(math.inf) == 1
    expected = statistics.median(differences)

    assert halves_difference(prior, current, change_a, change_b) == pytest.approx(expected,
                                                                                 rel=1e-12)


def below(value: float) -> float:
    """The float just below `value`."""
    return float(np.nextafter(value, -math.inf))


def above(value: float) -> float:
    """The float just above `value`."""
    return float(np.nextafter(value, math.inf))


PASSING = {"count": SETTINGS.min_sources, "difference": below(SETTINGS.halves_tolerance),
           "low": above(SETTINGS.min_z), "high": above(SETTINGS.min_z)}


@pytest.mark.parametrize("change, passed", [
    ({}, True),
    ({"count": SETTINGS.min_sources - 1}, False),
    ({"difference": SETTINGS.halves_tolerance}, False),
    ({"low": SETTINGS.min_z}, False),
    ({"high": SETTINGS.min_z}, False),
    ({"low": math.nan}, False),
    ({"high": math.inf}, True),
], ids=["passes", "too-few-sources", "difference-at-tolerance", "low-z-at-min",
        "high-z-at-min", "no-low-z", "infinite-high-z"])
def test_self_check_fails_on_each_reason_on_its_own(monkeypatch, solved: Solved, change: dict,
                                                    passed: bool) -> None:
    """self_check passes with at least `min_sources` Source_Frames, a Halves_Difference below
    `halves_tolerance` and both Tile_Scores above `min_z`, and fails with any one of them at its
    limit, the rest just passing, or with a Tile_Score that is NaN, no Z (Req 11.7, 11.8). The
    Halves_Difference and the Tile_Scores are handed in for halves_difference and tile_scores,
    so each sits exactly where the test puts it; the halves' fits are the real ones on the
    Third_Pass's Half_Tables."""
    numbers = {**PASSING, **change}
    third = solved.third
    seen = {}

    def hand_difference(prior, current, change_a, change_b):
        seen["current"] = current
        seen["changes"] = (change_a, change_b)
        return numbers["difference"]

    def hand_tiles(frame, windows, sources, answer_depth, prior_image, *, weights, **_):
        seen["answer"] = answer_depth
        return numbers["low"], numbers["high"], 3, 4

    monkeypatch.setattr(passes, "halves_difference", hand_difference)
    monkeypatch.setattr(passes, "tile_scores", hand_tiles)
    result = self_check(solved.prior, third, solved.windows, solved.sources, SETTINGS,
                        first_scale=solved.first_scale, k=1.25, scale_points=17,
                        source_count=numbers["count"], seconds_cut=SETTINGS.source_min_seconds)

    assert result.passed is passed
    assert seen["current"] is third.current and seen["answer"] is third.answer.score_depth
    for half_change in seen["changes"]:
        assert max(abs(number) for number in half_change[1:]) <= SETTINGS.fit_bound
    assert result.halves_difference == numbers["difference"]
    assert (result.tile_low, result.tile_high) == pytest.approx(
        (numbers["low"], numbers["high"]), nan_ok=True)
    assert (result.tile_low_count, result.tile_high_count) == (3, 4)
    assert (result.k, result.scale_points, result.source_count) == (1.25, 17, numbers["count"])
    assert result.seconds_cut == SETTINGS.source_min_seconds
    assert result.first_scale == solved.first_scale
    assert result.change == third.solved_change and result.m == solved.prior.m
    assert np.array_equal(result.band_weights, third.band_weights)


@pytest.mark.parametrize("start_scale", [1.0, TEXTURED_START])
def test_every_textured_plane_frame_gets_enough_sources(start_scale: float) -> None:
    """A guard on the textured plane of sub-task 5.3: with the real settings, choose_sources
    gives every frame, the first and the last included, at least `min_sources` Source_Frames at
    `source_min_seconds`, with the Start_Depth at the true depth and at the 1.1 × it of sub-task
    5.6, so a failure there points at the passes and not at the scene (Req 8.6)."""
    for index in range(TEXTURED_FRAMES):
        choices, seconds = textured_choice(index, start_scale)
        assert len(choices) >= SETTINGS.min_sources, index
        assert seconds == SETTINGS.source_min_seconds, index
        for choice in choices:
            assert abs(choice.candidate.pick_number - index) * TEXTURED_SECONDS >= seconds

# ---------------------------------------------------------------------------------------------
# The textured plane (sub-task 5.6) goes below this line.
# ---------------------------------------------------------------------------------------------

def plane_solve(index: int, *, plain: bool = False) -> tuple[PriorScore, Windows, SolveResult,
                                                             ThirdPass]:
    """choose_sources, the three passes and self_check on textured-plane frame `index`, with the
    real settings, its own Start_Depth 1.1 × the true depth and its Source_Frames' Start_Depths
    at the true depth (Req 9.6, A1, D24). The test sets the Start_Depths itself and never calls
    scene_scale; the k it hands self_check is the 1.1 the Start_Depth is off by, with no
    Scale_Points."""
    prior = textured_prior(index, plain=plain)
    windows = textured_windows(prior)
    choices, seconds = textured_choice(index)
    sources = [textured_score_source(choice.candidate.pick_number, half, prior_size=TEXTURED_SIZE,
                                     high_pass_sigma=SETTINGS.high_pass_sigma,
                                     band_sigmas=SETTINGS.band_sigmas, plain=plain)
               for choice, half in zip(choices, source_halves(len(choices)))]
    first_scale, first = first_pass(prior, windows, sources, SETTINGS)
    _, second = second_pass(prior, windows, sources, first, SETTINGS)
    weights = frame_band_weights(prior, band_coherences(prior, windows, sources, second,
                                                        SETTINGS), SETTINGS)
    third = third_pass(prior, windows, sources, weights, second, SETTINGS)
    result = self_check(prior, third, windows, sources, SETTINGS, first_scale=first_scale,
                        k=TEXTURED_START, scale_points=0, source_count=len(sources),
                        seconds_cut=seconds)
    return prior, windows, result, third


def depth_error(depth: np.ndarray, true_depth: np.ndarray, covered: np.ndarray) -> float:
    """The median over the Covered_Pixels of |z / z_true − 1|, a Covered_Pixel with no depth
    counting as infinite, worked out here."""
    errors = np.abs(depth[covered] / true_depth[covered] - 1.0)
    return float(np.median(np.where(np.isfinite(errors), errors, np.inf)))


@pytest.mark.parametrize("index", range(TEXTURED_FRAMES))
def test_textured_plane_passes_find_the_true_depth(index: int) -> None:
    """On the textured plane, frame `index` in turn, the ends included, with its Start_Depth 10%
    off the true depth (1.1, inside the First_Pass's ×0.5 to ×2), the three passes' depth is
    within 1% of the true depth, as the median over the Covered_Pixels of |z / z_true − 1|, and
    the Self_Check passes (Req 22.2, 10.1–10.8, 11.7).

    The First_Pass's Current_Depth, the Start_Depth, is about 9% off the Source_Frames' Start_
    Depths at the true depth, inside `occlusion_tolerance`, so its windows aren't occluded; the
    later passes and the Tile_Score, at about the true depth, match them (Req 9.6, A1, D24)."""
    prior, windows, result, third = plane_solve(index)
    true_depth = textured_depth(index, TEXTURED_SIZE)
    start_error = depth_error(prior.prior_start_depth, true_depth, prior.covered)
    assert start_error == pytest.approx(TEXTURED_START - 1.0, rel=1e-9)
    assert windows.count > 0

    error = depth_error(third.answer.prior_depth, true_depth, prior.covered)
    assert error < 0.01, (index, error)
    assert result.passed, (index, result)
    assert result.source_count >= SETTINGS.min_sources
    assert result.halves_difference < SETTINGS.halves_tolerance
    assert result.tile_low > SETTINGS.min_z and result.tile_high > SETTINGS.min_z


@pytest.mark.parametrize("index", [0, PASS_FRAME])
def test_plain_plane_fails_the_self_check(index: int) -> None:
    """The same plane in one flat grey fails the Self_Check: every window is flat, so there is no
    Score_Window and no Z, though choose_sources, which reads only the geometry, still finds
    `min_sources` Source_Frames (Req 22.2, 9.5, 11.5, 11.7)."""
    _, windows, result, _ = plane_solve(index, plain=True)
    assert windows.count == 0
    assert result.source_count >= SETTINGS.min_sources
    assert math.isnan(result.tile_low) and math.isnan(result.tile_high)
    assert (result.tile_low_count, result.tile_high_count) == (0, 0)
    assert not result.passed
