"""One Frame_Solve of the Walls_Step, on arrays in memory (walls spec, Req 6.6, 8.7, 12.4, 12.6,
12.7, A9, C3, C4).

Task 6 of the spec writes this half of the module, the pure one (C4):
- FrameArrays: what the Preparation_Pass gives one Prior_Frame (C3): its name, Pick_Number,
  Frame_Time and Image_Pose, the Shared_Camera, its Score_Size arrays (High_Pass, Bands,
  Score_Mask, Depth_Pixels and Start_Depth), its Prior_Size arrays (the Depth_Prior's
  `inverse_depth`, `normal` and `mask`, whose true pixels are its Covered_Pixels, and its
  Start_Depth), its Scene_Scale k and its Scale_Point count. m, the Prior_Size Start_Depth's
  Median_Inverse_Depth over the Covered_Pixels, is worked out from those, so it can't disagree
  with them.
- solve_arrays(frame, candidates, load_source, settings): the whole Frame_Solve.
  1. The Source_Frames, with sources.choose_sources on the Prior_Size Start_Depth, in rank order,
     each given its Source_Half by passes.source_halves (Req 11.1). With fewer than
     `min_sources` the frame fails with `result` None, no pass run and no Source_Frame loaded
     (Req 8.7, D14).
  2. Otherwise `load_source(name, half)` loads each chosen Source_Frame's ScoreSource, and only
     those, in rank order, and the First_Pass, the Second_Pass, the Band_Weights, the Third_Pass
     and the Self_Check run (Req 10, 11).
  3. The Facets are cut from the Depth_Prior's own `normal` and `inverse_depth` on its
     Covered_Pixels for every Frame_Solve that ends, failed or passed, for the flatten (Req 12.1,
     12.7, A9).
  4. Only when the Self_Check passes, the Solved_Depth before any Facet_Change is the Start_Depth
     through the Solved_Change with the Start_Depth's m, at both sizes, and the normals are the
     Depth_Prior's `normal` through the same change with geometry.changed_normal (Req 6.4, 6.6,
     10.8). Every Facet is refitted against that one Solved_Depth (Req 12.4), and then the
     Facet_Changes that passed go on, in Facet order, each on its own Facet's pixels with
     facets.apply_facet_change (Req 12.6). Facets hold no pixel in common, so the order changes
     nothing but is fixed anyway.
- FrameSolve: what one Frame_Solve gives back, plain data that pickles, so it can cross the
  process boundary of task 9: the name, the SolveResult or None, the chosen Source_Frames with
  their Source_Rank and Source_Half, the time cut they were chosen with, the Facets, the applied
  Facet_Changes, and the Solved_Depth and its normals at the Prior_Size, None when it failed.

Task 9 adds SolveTask, solve_frame(), the scratch loaders load_frame() and load_source(), and the
stage tasks of the cross-frame work (C4).

Every sum over many pixels is in the walls pure modules this module calls, which keep them out of
BLAS (C14). This module imports only the standard library, numpy, the walls pure modules and two
pure Train_Step modules the walls pure modules already import: splatpipe.train.colmap for the
Camera type and splatpipe.train.scene for the Score_Size K by the Train_Step's own
scaled_intrinsics.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from splatpipe.train.colmap import Camera
from splatpipe.train.scene import scaled_intrinsics
from splatpipe.walls.facets import (Facet, FacetChange, FacetSettings, apply_facet_change,
                                    cut_facets, refit_facet)
from splatpipe.walls.geometry import changed_normal, median_inverse_depth, normalised_coords
from splatpipe.walls.passes import (PassDepth, PriorScore, SolveResult, ThirdPass,
                                    band_coherences, first_pass, frame_band_weights, next_depth,
                                    second_pass, self_check, source_halves, start, third_pass)
from splatpipe.walls.scoring import ScoreFrame, ScoreSource, Windows, score_windows
from splatpipe.walls.sources import Candidate, SourceSettings, choose_sources

# load_source(name, half): the ScoreSource of the Source_Frame `name`, in Source_Half `half`.
LoadSource = Callable[[str, str], ScoreSource]


class SolveSettings(FacetSettings, SourceSettings, Protocol):
    """The `[walls]` settings one Frame_Solve reads, as WallsSettings holds them."""
    window_step: int
    tile_size: int


@dataclass(frozen=True)
class FrameArrays:
    """What the Preparation_Pass gives one Prior_Frame (C3)."""
    name: str                       # its name in images.bin
    pick_number: int                # its Pick's number in frames.json
    frame_time: float               # its Pick's seconds, its Frame_Time
    world_to_camera: np.ndarray     # (4, 4) float64, its Image_Pose
    camera: Camera                  # the Shared_Camera
    # At the Score_Size (h, w).
    high_pass: np.ndarray           # (h, w) float64
    bands: np.ndarray               # (B, h, w) float64
    score_mask: np.ndarray          # (h, w) bool, true where kept
    depth_pixel: np.ndarray         # (h, w) bool, its Depth_Pixels
    score_start_depth: np.ndarray   # (h, w) float64, its Start_Depth, NaN off the Depth_Pixels
    # At the Prior_Size (ph, pw).
    prior_inverse_depth: np.ndarray  # (ph, pw) its Depth_Prior's inverse_depth
    prior_normal: np.ndarray        # (ph, pw, 3) its Depth_Prior's normal
    covered: np.ndarray             # (ph, pw) bool, its Depth_Prior's mask: its Covered_Pixels
    prior_start_depth: np.ndarray   # (ph, pw) float64, k × its Prior_Depth, NaN off covered
    k: float                        # its Scene_Scale
    scale_points: int               # its number of Scale_Points

    @property
    def candidate(self) -> Candidate:
        """The Prior_Frame as source choice sees it."""
        return Candidate(name=self.name, pick_number=int(self.pick_number),
                         frame_time=float(self.frame_time),
                         world_to_camera=np.asarray(self.world_to_camera, dtype=np.float64))

    @property
    def score_size(self) -> tuple[int, int]:
        """The Score_Size as (width, height)."""
        height, width = np.shape(self.high_pass)
        return int(width), int(height)

    @property
    def prior_size(self) -> tuple[int, int]:
        """The Prior_Size as (width, height)."""
        height, width = np.shape(self.covered)
        return int(width), int(height)

    @property
    def m(self) -> float:
        """The Prior_Size Start_Depth's Median_Inverse_Depth over the Covered_Pixels, the m of
        the Solved_Change (Req 6.1, 10.8)."""
        return median_inverse_depth(self.prior_start_depth, np.asarray(self.covered, dtype=bool))


@dataclass(frozen=True)
class ChosenSource:
    """One chosen Source_Frame, as the Walls_Record and the cross-frame stages need it."""
    name: str
    pick_number: int
    source_rank: float   # its Source_Rank, Overlap × min(Image_Motion, source_motion_cap)
    overlap: float
    image_motion: float
    half: str            # its Source_Half, "A" or "B"


@dataclass(frozen=True)
class FrameSolve:
    """What one Frame_Solve gives back. Plain data, so it pickles."""
    name: str
    result: SolveResult | None        # None when it failed before any pass (too few sources)
    sources: tuple[ChosenSource, ...]  # the chosen Source_Frames, in rank order
    seconds_cut: float                # source_min_seconds or source_fallback_seconds (Req 8.8)
    facets: tuple[Facet, ...]         # cut for every Frame_Solve that ends (Req 12.7)
    applied: tuple[FacetChange, ...]  # the applied Facet_Changes, in Facet order
    solved_depth: np.ndarray | None   # (ph, pw) float64, None unless it passed
    solved_normal: np.ndarray | None  # (ph, pw, 3) float32, None unless it passed

    @property
    def passed(self) -> bool:
        """Whether it passed its Self_Check, so the Prior_Frame is a Trusted_Frame."""
        return self.result is not None and self.result.passed


def prior_score(frame: FrameArrays) -> PriorScore:
    """The Prior_Frame's side of the passes, with the Score_Size K by the Train_Step's rule and
    the Normalised_Coordinates of both sizes by the Prior_Format's pixel-centre rule."""
    score_width, score_height = frame.score_size
    prior_width, prior_height = frame.prior_size
    x, y = normalised_coords(frame.camera, score_width, score_height)
    prior_x, prior_y = normalised_coords(frame.camera, prior_width, prior_height)
    score_frame = ScoreFrame(world_to_camera=np.asarray(frame.world_to_camera, dtype=np.float64),
                             k=scaled_intrinsics(frame.camera, score_width, score_height),
                             x=x, y=y)
    return PriorScore(frame=score_frame, high_pass=np.asarray(frame.high_pass, dtype=np.float64),
                      bands=np.asarray(frame.bands, dtype=np.float64),
                      score_mask=np.asarray(frame.score_mask, dtype=bool),
                      depth_pixel=np.asarray(frame.depth_pixel, dtype=bool),
                      start_depth=np.asarray(frame.score_start_depth, dtype=np.float64),
                      prior_start_depth=np.asarray(frame.prior_start_depth, dtype=np.float64),
                      covered=np.asarray(frame.covered, dtype=bool), prior_x=prior_x,
                      prior_y=prior_y, m=frame.m)


def frame_windows(prior: PriorScore, settings: SolveSettings) -> Windows:
    """The Prior_Frame's Score_Windows with the settings' window, step, Tile and flat test."""
    return score_windows(prior.high_pass, prior.score_mask, prior.depth_pixel, prior.frame.x,
                         prior.frame.y, window_size=settings.window_size,
                         window_step=settings.window_step, tile_size=settings.tile_size,
                         flat_variance=settings.flat_variance)


def frame_facets(frame: FrameArrays, settings: SolveSettings) -> tuple[Facet, ...]:
    """The Prior_Frame's Facets, from its Depth_Prior's own `normal` and `inverse_depth` on its
    Covered_Pixels (Req 12.1, A9)."""
    return tuple(cut_facets(frame.prior_normal, frame.prior_inverse_depth, frame.covered,
                            settings))


def solve_arrays(frame: FrameArrays, candidates: Sequence[Candidate], load_source: LoadSource,
                 settings: SolveSettings) -> FrameSolve:
    """One Frame_Solve of the Prior_Frame `frame`, with the other Prior_Frames `candidates` and
    `load_source(name, half)` giving a chosen Source_Frame's ScoreSource in its Source_Half
    (Req 8.7, 10, 11, 12.4, 12.6, 12.7, C4)."""
    choices, seconds_cut = choose_sources(frame.candidate, frame.prior_start_depth, frame.covered,
                                          frame.camera, frame.score_size, candidates, settings)
    halves = source_halves(len(choices))
    chosen = tuple(ChosenSource(name=choice.candidate.name,
                                pick_number=int(choice.candidate.pick_number),
                                source_rank=float(choice.rank), overlap=float(choice.overlap),
                                image_motion=float(choice.image_motion), half=half)
                   for choice, half in zip(choices, halves))
    if len(chosen) < settings.min_sources:
        return FrameSolve(name=frame.name, result=None, sources=chosen,
                          seconds_cut=float(seconds_cut), facets=frame_facets(frame, settings),
                          applied=(), solved_depth=None, solved_normal=None)

    sources = [load_source(source.name, source.half) for source in chosen]
    prior = prior_score(frame)
    windows = frame_windows(prior, settings)
    first_scale, first = first_pass(prior, windows, sources, settings)
    _, second = second_pass(prior, windows, sources, first, settings)
    weights = frame_band_weights(prior, band_coherences(prior, windows, sources, second,
                                                        settings), settings)
    third = third_pass(prior, windows, sources, weights, second, settings)
    result = self_check(prior, third, windows, sources, settings, first_scale=first_scale,
                        k=frame.k, scale_points=frame.scale_points, source_count=len(sources),
                        seconds_cut=seconds_cut)
    facets = frame_facets(frame, settings)
    if not result.passed:
        return FrameSolve(name=frame.name, result=result, sources=chosen,
                          seconds_cut=float(seconds_cut), facets=facets, applied=(),
                          solved_depth=None, solved_normal=None)

    # The Start_Depth through the Solved_Change, with the Start_Depth's m, at both sizes: the
    # change composed onto the Identity_Change is the Solved_Change itself, bit for bit.
    solved = next_depth(prior, start(prior), result.change)
    normal = changed_normal(np.asarray(frame.prior_normal, dtype=np.float64),
                            prior.prior_start_depth, prior.prior_x, prior.prior_y, result.change,
                            prior.m)
    applied = _refits(facets, windows, sources, solved, settings, prior, third)
    depth = solved.prior_depth
    for change in applied:
        depth, normal = apply_facet_change(depth, normal, facets[change.facet], change,
                                           prior.prior_x, prior.prior_y)
    return FrameSolve(name=frame.name, result=result, sources=chosen,
                      seconds_cut=float(seconds_cut), facets=facets, applied=applied,
                      solved_depth=np.asarray(depth, dtype=np.float64),
                      solved_normal=np.asarray(normal, dtype=np.float32))


def _refits(facets: Sequence[Facet], windows: Windows, sources: Sequence[ScoreSource],
            solved: PassDepth, settings: SolveSettings, prior: PriorScore,
            third: ThirdPass) -> tuple[FacetChange, ...]:
    """Each Facet refitted against the one Solved_Depth before any Facet_Change, and the
    Facet_Changes that passed Req 12.5's gates, in Facet order (Req 12.4, 12.5)."""
    changes = (refit_facet(facet, windows, sources, solved, settings, prior=prior, third=third,
                           index=index)
               for index, facet in enumerate(facets))
    return tuple(change for change in changes if change is not None)
