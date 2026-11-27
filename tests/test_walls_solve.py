"""Unit tests for one Frame_Solve, solve_arrays() in splatpipe/walls/solve.py (walls spec,
Req 8.7, 12.7, 22.1, D24).

The scene is the textured plane of tests/camera_files.py at its own 128 x 96, which is both its
Score_Size and its Prior_Size. Each test builds the frame's FrameArrays as the Preparation_Pass
would hand them, with k = 1 and the Depth_Prior's inverse_depth 1 / (1.1 × the true depth) as
float32 on the Covered_Pixels, so its Prior_Depth and its Start_Depth are both 1.1 × the true
depth, and its normal the plane's true normal. `load_source` is a small recorder around
textured_score_source, whose Start_Depth is the Source_Frame's true depth, as true Observed_Points
would make it (D24). The test sets the Start_Depths itself and never calls scene_scale.

- A textured frame passes, its Solved_Depth is within 1% of the true depth at the Prior_Size,
  its normals are unit length and face the camera, only its chosen Source_Frames are loaded, and
  its FrameSolve pickles.
- The same frame on the plain plane fails its Self_Check, with its Facets cut and none refitted.
- With too few candidates it fails with `result` None, no pass run and no Source_Frame loaded.

The settings are read_settings's on the real config.toml's `[walls]` and `[walls.default]`. Every
oracle is written here and never calls the function it checks. Everything runs on the CPU on a
few thousand pixels (Req 22.1).
"""

from __future__ import annotations

import pickle
import tomllib
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest

from splatpipe.walls import solve
from splatpipe.walls.images import bands, high_pass, luma
from splatpipe.walls.scoring import ScoreSource
from splatpipe.walls.settings import WallsSettings, read_settings
from splatpipe.walls.solve import FrameArrays, FrameSolve, solve_arrays
from splatpipe.walls.sources import Candidate

from camera_files import (TEXTURED_CAMERA, TEXTURED_FRAMES, textured_camera, textured_candidate,
                          textured_covered, textured_depth, textured_frame, textured_normal,
                          textured_pose, textured_score_source)

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_WALLS = tomllib.loads((REPO_ROOT / "config.toml").read_text(encoding="utf-8"))["walls"]


def real_settings(preset: str = "default") -> WallsSettings:
    """The real `[walls]` settings with the Preset table `preset`, as the step reads them."""
    shared = {key: value for key, value in CONFIG_WALLS.items() if not isinstance(value, dict)}
    return read_settings({**shared, **CONFIG_WALLS[preset]}, "walls", preset)


SETTINGS = real_settings()

# The camera's own size, both the Score_Size and the Prior_Size of the textured plane.
SIZE = (TEXTURED_CAMERA.width, TEXTURED_CAMERA.height)
PRIOR_SCALE = 1.1      # the Prior_Depth, and with k = 1 the Start_Depth, × the true depth
SOLVE_FRAMES = (0, 5)  # an end of the walk and its middle
PLAIN_FRAME = 5


def frame_arrays(index: int, *, plain: bool = False) -> FrameArrays:
    """Textured-plane frame `index` as the Preparation_Pass hands it: k = 1, the Depth_Prior's
    inverse_depth 1 / (1.1 × the true depth) as float32 on the Covered_Pixels and 0 off them,
    its normal the true one, and both Start_Depths k × the Prior_Depth, NaN off the Covered_Pixels.
    Every pixel is a Depth_Pixel where it is covered and kept by the Score_Mask."""
    covered = textured_covered(index, SIZE)
    true_depth = textured_depth(index, SIZE)
    with np.errstate(divide="ignore", invalid="ignore"):
        inverse = np.where(covered, 1.0 / (PRIOR_SCALE * true_depth), 0.0).astype(np.float32)
        start_depth = np.where(covered, 1.0 / inverse.astype(np.float64), np.nan)
    grey = luma(textured_frame(index, plain=plain))
    return FrameArrays(name=textured_candidate(index).name, pick_number=index,
                       frame_time=textured_candidate(index).frame_time,
                       world_to_camera=textured_pose(index), camera=textured_camera(),
                       high_pass=high_pass(grey, SETTINGS.high_pass_sigma),
                       bands=np.stack(bands(grey, SETTINGS.band_sigmas)),
                       score_mask=np.ones(covered.shape, dtype=bool), depth_pixel=covered,
                       score_start_depth=start_depth, prior_inverse_depth=inverse,
                       prior_normal=textured_normal(index, SIZE).astype(np.float32),
                       covered=covered, prior_start_depth=start_depth, k=1.0, scale_points=0)


def other_candidates(index: int) -> list[Candidate]:
    """Every textured-plane frame but `index`, as source choice sees it."""
    return [textured_candidate(other) for other in range(TEXTURED_FRAMES) if other != index]


@dataclass
class SourceRecorder:
    """load_source for the textured plane: each Source_Frame's ScoreSource, its Start_Depth the
    true depth, with every (name, half) it was asked for recorded in order."""
    plain: bool = False
    calls: list[tuple[str, str]] = field(default_factory=list)

    def __call__(self, name: str, half: str) -> ScoreSource:
        self.calls.append((name, half))
        index = next(other for other in range(TEXTURED_FRAMES)
                     if textured_candidate(other).name == name)
        return textured_score_source(index, half, prior_size=SIZE,
                                     high_pass_sigma=SETTINGS.high_pass_sigma,
                                     band_sigmas=SETTINGS.band_sigmas, plain=self.plain)


@dataclass(frozen=True)
class Solved:
    """One Frame_Solve with what it loaded and how often it refitted a Facet."""
    index: int
    solve: FrameSolve
    loads: list[tuple[str, str]]
    refits: int


def run_solve(index: int, *, plain: bool = False,
              candidates: list[Candidate] | None = None) -> Solved:
    """solve_arrays on textured-plane frame `index`, with the real settings, counting the calls
    of the real refit_facet and of the real first_pass while it runs."""
    recorder = SourceRecorder(plain=plain)
    counts = {"refit": 0, "first": 0}
    real_refit, real_first = solve.refit_facet, solve.first_pass

    def counted_refit(*args, **kwargs):
        counts["refit"] += 1
        return real_refit(*args, **kwargs)

    def counted_first(*args, **kwargs):
        counts["first"] += 1
        return real_first(*args, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(solve, "refit_facet", counted_refit)
        patch.setattr(solve, "first_pass", counted_first)
        result = solve_arrays(frame_arrays(index, plain=plain),
                              other_candidates(index) if candidates is None else candidates,
                              recorder, SETTINGS)
    assert counts["first"] == (0 if result.result is None else 1)
    return Solved(index, result, recorder.calls, counts["refit"])


@pytest.fixture(scope="module", params=SOLVE_FRAMES)
def solved(request: pytest.FixtureRequest) -> Iterator[Solved]:
    """The textured-plane Frame_Solve of each of SOLVE_FRAMES, run once for the module."""
    yield run_solve(request.param)


def depth_error(depth: np.ndarray, true_depth: np.ndarray, covered: np.ndarray) -> float:
    """The median over the Covered_Pixels of |z / z_true − 1|, a Covered_Pixel with no depth
    counting as infinite, worked out here."""
    errors = np.abs(depth[covered] / true_depth[covered] - 1.0)
    return float(np.median(np.where(np.isfinite(errors), errors, np.inf)))


def camera_rays(size: tuple[int, int]) -> np.ndarray:
    """The camera-space ray through each pixel centre of `size` (w, h), (h, w, 3), z = 1, by the
    Prior_Format's pixel-centre rule, worked out here."""
    width, height = size
    camera = TEXTURED_CAMERA
    u = (np.arange(width) + 0.5) * camera.width / width
    v = (np.arange(height) + 0.5) * camera.height / height
    uu, vv = np.meshgrid(u, v)
    return np.stack([(uu - camera.cx) / camera.fx, (vv - camera.cy) / camera.fy,
                     np.ones_like(uu)], axis=-1)


# ---------------------------------------------------------------------------------------------
# A textured frame passes (Req 6.6, 10, 11, 12.4, 12.6, 22.1).
# ---------------------------------------------------------------------------------------------

def test_the_start_depth_is_10_percent_off_the_true_depth() -> None:
    """A guard on the FrameArrays built here: the Start_Depth is k × the Prior_Depth, 1.1 × the
    true depth, so the solve starts 10% off."""
    for index in SOLVE_FRAMES:
        frame = frame_arrays(index)
        assert frame.k == 1.0
        error = depth_error(frame.prior_start_depth, textured_depth(index, SIZE), frame.covered)
        assert error == pytest.approx(PRIOR_SCALE - 1.0, rel=1e-6)


def test_textured_frame_passes_within_1_percent_of_the_true_depth(solved: Solved) -> None:
    """With its Start_Depth 1.1 × the true depth and its Source_Frames' at the true depth,
    solve_arrays passes, and its Solved_Depth at the Prior_Size is within 1% of the true depth,
    as the median over the Covered_Pixels of |z / z_true − 1|."""
    result = solved.solve
    assert result.passed, (solved.index, result.result)
    assert result.result is not None and result.result.k == 1.0
    assert len(result.sources) >= SETTINGS.min_sources
    assert result.result.source_count == len(result.sources)
    assert result.solved_depth is not None and result.solved_depth.dtype == np.float64
    assert result.solved_depth.shape == (SIZE[1], SIZE[0])
    covered = textured_covered(solved.index, SIZE)
    error = depth_error(result.solved_depth, textured_depth(solved.index, SIZE), covered)
    assert error < 0.01, (solved.index, error)


def test_textured_frames_normals_are_unit_and_face_the_camera(solved: Solved) -> None:
    """The passed Frame_Solve's normals are float32, unit length on every Covered_Pixel, and face
    the camera: each one's dot product with its pixel's ray is below 0."""
    normal = solved.solve.solved_normal
    assert normal is not None and normal.dtype == np.float32
    assert normal.shape == (SIZE[1], SIZE[0], 3)
    covered = textured_covered(solved.index, SIZE)
    on = normal[covered].astype(np.float64)
    assert np.all(np.isfinite(on))
    np.testing.assert_allclose(np.sqrt(np.sum(on * on, axis=-1)), 1.0, atol=1e-5)
    facing = np.sum(on * camera_rays(SIZE)[covered], axis=-1)
    assert np.all(facing < 0.0)


def test_textured_frame_loads_only_its_source_frames_in_rank_order(solved: Solved) -> None:
    """load_source is called once for each chosen Source_Frame, in rank order, with its
    Source_Half, alternating A and B, and never for the frame itself (Req 11.1, C4)."""
    result = solved.solve
    assert solved.loads == [(source.name, source.half) for source in result.sources]
    assert [half for _, half in solved.loads] == ["AB"[i % 2] for i in range(len(solved.loads))]
    names = [name for name, _ in solved.loads]
    assert len(set(names)) == len(names)
    assert textured_candidate(solved.index).name not in names
    ranks = [source.source_rank for source in result.sources]
    assert ranks == sorted(ranks, reverse=True)


def test_textured_frame_refits_each_facet(solved: Solved) -> None:
    """A passed Frame_Solve cuts the plane's Facets and refits each one once (Req 12.4, 12.7);
    every applied Facet_Change names one of them, in Facet order."""
    result = solved.solve
    assert len(result.facets) >= 1
    assert solved.refits == len(result.facets)
    indices = [change.facet for change in result.applied]
    assert indices == sorted(set(indices))
    assert all(0 <= facet < len(result.facets) for facet in indices)


def test_frame_solve_pickles(solved: Solved) -> None:
    """A FrameSolve is plain data that comes back from pickle as it went in, so it can cross the
    Worker_Pool's process boundary."""
    result = solved.solve
    back = pickle.loads(pickle.dumps(result))
    assert isinstance(back, FrameSolve)
    assert (back.name, back.sources, back.seconds_cut, back.passed) == (
        result.name, result.sources, result.seconds_cut, result.passed)
    assert back.applied == result.applied
    assert back.result is not None and result.result is not None
    assert back.result.change == result.result.change
    np.testing.assert_array_equal(back.result.band_weights, result.result.band_weights)
    np.testing.assert_array_equal(back.solved_depth, result.solved_depth)
    np.testing.assert_array_equal(back.solved_normal, result.solved_normal)
    assert len(back.facets) == len(result.facets)
    for got, want in zip(back.facets, result.facets):
        np.testing.assert_array_equal(got.pixels, want.pixels)
        np.testing.assert_array_equal(got.normal, want.normal)
        assert got.m == want.m


# ---------------------------------------------------------------------------------------------
# The plain plane fails, with its Facets cut and none refitted (Req 12.7).
# ---------------------------------------------------------------------------------------------

def test_plain_plane_fails_with_its_facets_cut_and_none_refitted() -> None:
    """The same frame on the plain plane runs its passes and fails its Self_Check, so it has no
    Solved_Depth and no normals. Its Facets are still cut, for the flatten, and refit_facet is
    never called (Req 12.7, A9)."""
    plain = run_solve(PLAIN_FRAME, plain=True)
    result = plain.solve
    assert result.result is not None
    assert not result.passed and not result.result.passed
    assert len(result.sources) >= SETTINGS.min_sources
    assert plain.loads == [(source.name, source.half) for source in result.sources]
    assert len(result.facets) >= 1
    assert plain.refits == 0
    assert result.applied == ()
    assert result.solved_depth is None and result.solved_normal is None
    back = pickle.loads(pickle.dumps(result))
    assert back.name == result.name and not back.passed


# ---------------------------------------------------------------------------------------------
# Too few candidates: no pass runs and no Source_Frame loads (Req 8.7, D14).
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("candidates", [
    pytest.param([], id="none"),
    pytest.param([4, 6], id="two-neighbours"),
])
def test_too_few_candidates_fail_with_no_result_and_no_load(candidates: list[int]) -> None:
    """With fewer candidates than `min_sources`, solve_arrays fails with `result` None, runs no
    pass, never calls load_source, and still cuts the Facets (Req 8.7, 12.7)."""
    assert len(candidates) < SETTINGS.min_sources
    few = run_solve(PLAIN_FRAME, candidates=[textured_candidate(i) for i in candidates])
    result = few.solve
    assert result.result is None
    assert not result.passed
    assert few.loads == []
    assert len(result.sources) < SETTINGS.min_sources
    assert result.seconds_cut in (SETTINGS.source_min_seconds, SETTINGS.source_fallback_seconds)
    assert len(result.facets) >= 1
    assert few.refits == 0
    assert result.applied == ()
    assert result.solved_depth is None and result.solved_normal is None
