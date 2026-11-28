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

Task 9 writes the other half, what a worker runs (C4, Req 4.8, 5.2, 5.3, 14.4):
- load_frame() and load_source() read a Prior_Frame's scratch arrays back with
  np.load(..., mmap_mode="r"), by inputs.py's file names, into a FrameArrays and a ScoreSource.
  A Source_Frame's ScoreSource gets its start_depth.npy and covered.npy, so every occlusion test
  compares with the Start_Depth the preparation pass worked out, never the raw Prior_Depth (D24).
  The scratch folder holds no pose, Pick_Number or Frame_Time: the task gives them, from the
  Candidates the main process made.
- SolveTask and solve_frame(task), the worker's entry point for one Frame_Solve: it loads the
  frame, runs solve_arrays with load_source, writes the Facets to facets.npy and
  facet_numbers.npy for every Frame_Solve that ends, the Solved_Depth and its normals to
  current_depth.npy and current_normal.npy when it passed, and gives the FrameSolve without those
  two arrays. The main process keeps each FrameSolve's Facets for world_planes and the Walls_Record;
  the flatten's workers read them back with load_facets, so no stage task carries them.
- One task and one module-level function per cross-frame stage, run through the Worker_Pool's
  map_frames (Req 5.3). Each task is plain data that pickles: the frame's name, the scratch
  folder, its pose, the Shared_Camera, its Neighbour_Frames' names and poses, the World_Planes
  where the stage needs them, and the settings, never an array of pixels. Each stage reads the
  files the stage before wrote and writes a file name of its own (inputs.py lists them), so no
  frame overwrites what another frame of the same stage still reads, and the order the workers
  take the frames in changes nothing (Req 14.4, A11):

      fuse_stage       Trusted_Frames: current_depth.npy, its neighbours' too -> fused_depth.npy
      (world planes)   once, on the main process, from the Facets and fused_depth.npy (D15)
      fallback_stage   Failed_Frames: start_depth.npy, prior_normal.npy, its neighbours'
                       fused_depth.npy -> fallback_depth.npy, fallback_normal.npy, when it has one
      correct_stage    frames with a Walls_Depth: fused_depth.npy or fallback_depth.npy, its
                       neighbours' fused_depth.npy -> corrected_depth.npy
      flatten_stage    corrected_depth.npy, current_normal.npy or fallback_normal.npy, the Facets
                       -> flat_depth.npy, walls_normal.npy
      despeckle_stage  flat_depth.npy -> walls_depth.npy
      choice_stage     every Prior_Frame: walls_depth.npy and scale_points.npy -> its outcome,
                       reason and Sparse_Error, given back, not written

  A Timed_Out_Frame gets no stage task after the solve, but keeps its scratch arrays, so it
  stayed a Source_Frame for the frames still solving (Req 4.7, 4.8). Small results, the
  fallback's kind, coverage and change and the choice's outcome, come back as return values.

Every sum over many pixels is in the walls pure modules this module calls, which keep them out of
BLAS, and the stage functions here only load, index and save (C14). This module imports only the
standard library, numpy, the walls pure modules, splatpipe.walls.inputs for the scratch folder's
file names, and two pure Train_Step modules the walls pure modules already import:
splatpipe.train.colmap for the Camera type and splatpipe.train.scene for the Score_Size K by the
Train_Step's own scaled_intrinsics.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol

import numpy as np

from splatpipe.train.colmap import Camera
from splatpipe.train.scene import scaled_intrinsics
from splatpipe.walls.crossframe import (ChoiceSettings, CorrectSettings, FuseSettings,
                                        correct_frame, despeckle_frame, frame_choice, fuse_frame,
                                        projected_values, sparse_error)
from splatpipe.walls.facets import (Facet, FacetChange, FacetSettings, apply_facet_change,
                                    cut_facets, refit_facet)
from splatpipe.walls.fit import FourNumbers
from splatpipe.walls.geometry import changed_normal, median_inverse_depth, normalised_coords
from splatpipe.walls.inputs import (BANDS, CORRECTED_DEPTH, COVERED, CURRENT_DEPTH,
                                    CURRENT_NORMAL, DEPTH_PIXEL, FACET_NUMBERS, FACETS,
                                    FALLBACK_DEPTH, FALLBACK_NORMAL, FLAT_DEPTH, FUSED_DEPTH,
                                    HIGH_PASS, NUMBERS, PRIOR_INVERSE, PRIOR_NORMAL, SCALE_POINTS,
                                    SCORE_MASK, START_DEPTH, START_DEPTH_SCORE, WALLS_DEPTH,
                                    WALLS_NORMAL, frame_folder)
from splatpipe.walls.planes import (FallbackSettings, FlattenSettings, WorldPlane,
                                    WorldPlaneSettings, fallback_medians, flatten_frame,
                                    median_coverage, plane_registration, projection_fallback,
                                    world_planes)
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


# Task 9: the scratch loaders, the Frame_Solve's worker task and the cross-frame stage tasks (C4).

PROJECTION = "projection"  # a Fallback_Depth from the Projection_Fallback (Req 15.2, 15.5)
PLANES = "planes"          # a Fallback_Depth from the Plane_Registration (Req 15.3, 15.5)
_NO_FACET = -1             # facets.npy's label off every Facet


class DespeckleSettings(Protocol):
    """The [walls] settings the despeckle reads."""
    despeckle_window: int
    despeckle_min_valid: int
    despeckle_tolerance: float


class StageSettings(FuseSettings, CorrectSettings, ChoiceSettings, FallbackSettings,
                    FlattenSettings, WorldPlaneSettings, DespeckleSettings, Protocol):
    """The [walls] settings the cross-frame stages read, as WallsSettings holds them."""


def _load(folder: Path, file_name: str) -> np.ndarray:
    """The scratch array `file_name` in `folder`, mapped read-only from its .npy file."""
    return np.load(folder / file_name, mmap_mode="r", allow_pickle=False)


def _save(folder: Path, file_name: str, array: np.ndarray) -> None:
    """Write `array` to `file_name` in `folder` with np.save."""
    np.save(folder / file_name, np.ascontiguousarray(array), allow_pickle=False)


def load_frame(scratch: Path, frame: Candidate, camera: Camera) -> FrameArrays:
    """The FrameArrays of the Prior_Frame `frame`, read from its scratch folder in `scratch`,
    work/walls/scratch/, with `camera` the Shared_Camera (C4). The scratch arrays hold no pose,
    Pick_Number or Frame_Time, so `frame` gives them."""
    folder = frame_folder(scratch, frame.name)
    numbers = _load(folder, NUMBERS)
    return FrameArrays(name=frame.name, pick_number=int(frame.pick_number),
                       frame_time=float(frame.frame_time),
                       world_to_camera=np.asarray(frame.world_to_camera, dtype=np.float64),
                       camera=camera, high_pass=_load(folder, HIGH_PASS),
                       bands=_load(folder, BANDS), score_mask=_load(folder, SCORE_MASK),
                       depth_pixel=_load(folder, DEPTH_PIXEL),
                       score_start_depth=_load(folder, START_DEPTH_SCORE),
                       prior_inverse_depth=_load(folder, PRIOR_INVERSE),
                       prior_normal=_load(folder, PRIOR_NORMAL), covered=_load(folder, COVERED),
                       prior_start_depth=_load(folder, START_DEPTH), k=float(numbers[0]),
                       scale_points=int(numbers[2]))


def load_source(scratch: Path, name: str, half: str, world_to_camera: np.ndarray) -> ScoreSource:
    """The ScoreSource of the Source_Frame `name` in Source_Half `half`, read from its scratch
    folder in `scratch`, with its Image_Pose `world_to_camera` (C4). Its Start_Depth is the
    preparation pass's start_depth.npy, k × its Prior_Depth, with covered.npy, so its occlusion
    tests never see the raw Prior_Depth (D24)."""
    folder = frame_folder(scratch, name)
    return ScoreSource(name=name, world_to_camera=np.asarray(world_to_camera, dtype=np.float64),
                       high_pass=_load(folder, HIGH_PASS), bands=_load(folder, BANDS),
                       score_mask=_load(folder, SCORE_MASK),
                       start_depth=_load(folder, START_DEPTH), covered=_load(folder, COVERED),
                       half=half)


def save_facets(folder: Path, facets: Sequence[Facet], shape: tuple[int, int]) -> None:
    """Write `facets`, at the Prior_Size `shape` (height, width), to facets.npy, an int32 label
    image with each pixel's Facet index and −1 off every Facet, and facet_numbers.npy, (F, 4)
    float64, each Facet's normal and m. Facets hold no pixel in common."""
    labels = np.full(shape, _NO_FACET, dtype=np.int32)
    numbers = np.zeros((len(facets), 4), dtype=np.float64)
    for index, facet in enumerate(facets):
        labels[facet.pixels[:, 0], facet.pixels[:, 1]] = index
        numbers[index, :3] = facet.normal
        numbers[index, 3] = facet.m
    _save(folder, FACETS, labels)
    _save(folder, FACET_NUMBERS, numbers)


def load_facets(folder: Path) -> tuple[Facet, ...]:
    """The Facets save_facets wrote to `folder`, each with its pixels in row-major order, as
    cut_facets gave them."""
    image = _load(folder, FACETS)
    width = int(image.shape[1])
    labels = np.asarray(image).reshape(-1)
    numbers = np.asarray(_load(folder, FACET_NUMBERS), dtype=np.float64)
    order = np.argsort(labels, kind="stable")  # row-major within each label
    ends = np.searchsorted(labels[order], np.arange(len(numbers) + 1) - 0.5)
    facets = []
    for index in range(len(numbers)):
        flat = order[ends[index]:ends[index + 1]].astype(np.int64)
        pixels = np.stack([flat // width, flat % width], axis=-1)
        facets.append(Facet(pixels=pixels, normal=np.array(numbers[index, :3]),
                            m=float(numbers[index, 3])))
    return tuple(facets)


@dataclass(frozen=True)
class SolveTask:
    """One Frame_Solve for a worker: plain data that pickles under `spawn` (Req 5.2, C4)."""
    name: str                          # the Prior_Frame's name
    pick_number: int
    frame_time: float
    world_to_camera: np.ndarray        # (4, 4) float64, its Image_Pose
    settings: SolveSettings
    camera: Camera                     # the Shared_Camera
    scratch: Path                      # work/walls/scratch/
    candidates: tuple[Candidate, ...]  # the other Prior_Frames, Timed_Out_Frames included

    @property
    def candidate(self) -> Candidate:
        """The Prior_Frame itself as source choice sees it."""
        return Candidate(name=self.name, pick_number=int(self.pick_number),
                         frame_time=float(self.frame_time),
                         world_to_camera=np.asarray(self.world_to_camera, dtype=np.float64))


def solve_frame(task: SolveTask) -> FrameSolve:
    """The Frame_Solve of `task`'s Prior_Frame, in a worker (Req 5.2, C4).

    It loads the frame with load_frame, runs solve_arrays with load_source, writes its Facets,
    and when it passed its Solved_Depth and normals to current_depth.npy and current_normal.npy,
    and gives the FrameSolve without those two arrays.
    """
    frame = load_frame(task.scratch, task.candidate, task.camera)
    poses = {candidate.name: candidate.world_to_camera for candidate in task.candidates}

    def load(name: str, half: str) -> ScoreSource:
        return load_source(task.scratch, name, half, poses[name])

    solved = solve_arrays(frame, task.candidates, load, task.settings)
    folder = frame_folder(task.scratch, task.name)
    width, height = frame.prior_size
    save_facets(folder, solved.facets, (height, width))
    if solved.solved_depth is not None and solved.solved_normal is not None:
        _save(folder, CURRENT_DEPTH, solved.solved_depth)
        _save(folder, CURRENT_NORMAL, solved.solved_normal)
    return replace(solved, solved_depth=None, solved_normal=None)


def _projected(scratch: Path, world_to_camera: np.ndarray, covered: np.ndarray,
               neighbours: Sequence[str], neighbour_poses: Sequence[np.ndarray], camera: Camera,
               file_name: str) -> np.ndarray:
    """The Projected_Values of a Prior_Frame from its Neighbour_Frames' `file_name` depths, as
    the stage before left them in `scratch` (Req 14.2, 14.4)."""
    depths = [_load(frame_folder(scratch, name), file_name) for name in neighbours]
    return projected_values(world_to_camera, covered, depths, list(neighbour_poses), camera)


@dataclass(frozen=True)
class FuseTask:
    """The fuse of one Trusted_Frame (Req 14.3)."""
    name: str
    scratch: Path
    world_to_camera: np.ndarray
    camera: Camera
    neighbours: tuple[str, ...]            # its Neighbour_Frames, nearest first
    neighbour_poses: tuple[np.ndarray, ...]
    settings: FuseSettings


def fuse_stage(task: FuseTask) -> str:
    """Fuse the Trusted_Frame's Solved_Depth with its Neighbour_Frames' Solved_Depths, writing
    fused_depth.npy, and give its name (Req 14.3)."""
    folder = frame_folder(task.scratch, task.name)
    covered = _load(folder, COVERED)
    projected = _projected(task.scratch, task.world_to_camera, covered, task.neighbours,
                           task.neighbour_poses, task.camera, CURRENT_DEPTH)
    _save(folder, FUSED_DEPTH, fuse_frame(_load(folder, CURRENT_DEPTH), projected, task.settings))
    return task.name


def scratch_world_planes(scratch: Path, facets: Mapping[str, Sequence[Facet]],
                         poses: Mapping[str, np.ndarray], camera: Camera,
                         settings: WorldPlaneSettings) -> list[WorldPlane]:
    """The World_Planes, built once on the main process after the fuse, from each Trusted_Frame's
    Facets, `facets` by name, kept from its FrameSolve, and its fused_depth.npy (Req 13, D15)."""
    fused = {name: _load(frame_folder(scratch, name), FUSED_DEPTH) for name in facets}
    return world_planes(facets, fused, poses, camera, settings)


@dataclass(frozen=True)
class FallbackTask:
    """The fallback of one Failed_Frame (Req 15)."""
    name: str
    scratch: Path
    world_to_camera: np.ndarray
    camera: Camera
    neighbours: tuple[str, ...]
    neighbour_poses: tuple[np.ndarray, ...]
    planes: tuple[WorldPlane, ...]         # the World_Planes
    settings: FallbackSettings


@dataclass(frozen=True)
class FallbackResult:
    """What the Walls_Record holds of a Failed_Frame's fallback (Req 15.5)."""
    name: str
    fallback: str | None            # PROJECTION, PLANES or None
    coverage: float                 # its share of Covered_Pixels with a median
    change: FourNumbers | None      # the fitted change of its Start_Depth
    m: float | None                 # the m that change is relative to

    @property
    def has_fallback(self) -> bool:
        return self.fallback is not None


def fallback_stage(task: FallbackTask) -> FallbackResult:
    """The Failed_Frame's Fallback_Depth, the Projection_Fallback when enough of its
    Covered_Pixels have a median of its Neighbour_Frames' Fused_Depths, else the
    Plane_Registration, written to fallback_depth.npy and fallback_normal.npy when it gets one
    (Req 15.1–15.5)."""
    folder = frame_folder(task.scratch, task.name)
    covered = _load(folder, COVERED)
    start = _load(folder, START_DEPTH)
    normal = _load(folder, PRIOR_NORMAL)
    m = float(_load(folder, NUMBERS)[1])
    projected = _projected(task.scratch, task.world_to_camera, covered, task.neighbours,
                           task.neighbour_poses, task.camera, FUSED_DEPTH)
    medians = fallback_medians(projected, task.settings)
    coverage = float(median_coverage(medians, covered))
    kind = PROJECTION
    found = projection_fallback(medians, start, normal, covered, m, task.camera, task.settings)
    if found is None:
        kind = PLANES
        found = plane_registration(task.planes, start, normal, covered, m, task.world_to_camera,
                                   task.camera, task.settings)
    if found is None:
        return FallbackResult(task.name, None, coverage, None, None)
    _save(folder, FALLBACK_DEPTH, found.depth)
    _save(folder, FALLBACK_NORMAL, found.normal)
    return FallbackResult(task.name, kind, coverage,
                          tuple(float(number) for number in found.change), float(found.m))


def _walls_start(trusted: bool) -> tuple[str, str]:
    """The depth and normal files a Walls_Depth starts from: the Fused_Depth and the Solved
    normals of a Trusted_Frame, the Fallback_Depth and its normals of a Failed_Frame."""
    return (FUSED_DEPTH, CURRENT_NORMAL) if trusted else (FALLBACK_DEPTH, FALLBACK_NORMAL)


@dataclass(frozen=True)
class CorrectTask:
    """The correction of one Prior_Frame with a Walls_Depth (Req 16.1)."""
    name: str
    scratch: Path
    world_to_camera: np.ndarray
    camera: Camera
    neighbours: tuple[str, ...]
    neighbour_poses: tuple[np.ndarray, ...]
    trusted: bool                   # a Trusted_Frame, or else a Failed_Frame with a fallback
    settings: CorrectSettings


def correct_stage(task: CorrectTask) -> str:
    """Correct the frame's depth to its Neighbour_Frames' Fused_Depths where they agree and it is
    off, writing corrected_depth.npy, and give its name (Req 16.1)."""
    folder = frame_folder(task.scratch, task.name)
    depth_file, _ = _walls_start(task.trusted)
    covered = _load(folder, COVERED)
    projected = _projected(task.scratch, task.world_to_camera, covered, task.neighbours,
                           task.neighbour_poses, task.camera, FUSED_DEPTH)
    _save(folder, CORRECTED_DEPTH,
          correct_frame(_load(folder, depth_file), projected, task.settings))
    return task.name


@dataclass(frozen=True)
class FlattenTask:
    """The flatten of one Prior_Frame with a Walls_Depth (Req 16.2, 16.3)."""
    name: str
    scratch: Path
    world_to_camera: np.ndarray
    camera: Camera
    planes: tuple[WorldPlane, ...]
    trusted: bool
    settings: FlattenSettings


def flatten_stage(task: FlattenTask) -> str:
    """Flatten the frame's Facets, read back with load_facets, on its corrected depth, writing
    flat_depth.npy and walls_normal.npy, and give its name (Req 16.2, 16.3, 16.6)."""
    folder = frame_folder(task.scratch, task.name)
    _, normal_file = _walls_start(task.trusted)
    depth, normal = flatten_frame(load_facets(folder), task.planes,
                                  _load(folder, CORRECTED_DEPTH), _load(folder, normal_file),
                                  task.world_to_camera, task.camera, task.settings)
    _save(folder, FLAT_DEPTH, depth)
    _save(folder, WALLS_NORMAL, normal)
    return task.name


@dataclass(frozen=True)
class DespeckleTask:
    """The despeckle of one Prior_Frame with a Walls_Depth (Req 16.4)."""
    name: str
    scratch: Path
    settings: DespeckleSettings


def despeckle_stage(task: DespeckleTask) -> str:
    """Despeckle the frame's flattened depth, writing walls_depth.npy, its Walls_Depth, and give
    its name (Req 16.4)."""
    folder = frame_folder(task.scratch, task.name)
    settings = task.settings
    _save(folder, WALLS_DEPTH,
          despeckle_frame(_load(folder, FLAT_DEPTH), _load(folder, COVERED),
                          settings.despeckle_window, settings.despeckle_min_valid,
                          settings.despeckle_tolerance))
    return task.name


@dataclass(frozen=True)
class ChoiceTask:
    """The per-frame choice of one Prior_Frame (Req 17.1–17.5)."""
    name: str
    scratch: Path
    trusted: bool
    timed_out: bool
    has_fallback: bool
    scale_points: int               # its Scale_Point count, from the preparation pass
    settings: ChoiceSettings

    @property
    def has_walls_depth(self) -> bool:
        """Whether the stages gave it a Walls_Depth, walls_depth.npy."""
        return not self.timed_out and (self.trusted or self.has_fallback)


@dataclass(frozen=True)
class ChoiceResult:
    """A Prior_Frame's outcome, reason and Sparse_Error, for the Walls_Record (Req 17.10)."""
    name: str
    outcome: str                    # passed, fallback or depth_prior
    reason: str | None              # timed_out, no_fallback or sparse_error for depth_prior
    sparse_error: float | None      # None with no Walls_Depth or too few Scale_Points


def choice_stage(task: ChoiceTask) -> ChoiceResult:
    """The frame's Sparse_Error, from its walls_depth.npy at its Scale_Points when it has a
    Walls_Depth and at least scale_min_points of them, and its outcome (Req 17, A14, C5)."""
    error = None
    if task.has_walls_depth and task.scale_points >= task.settings.scale_min_points:
        folder = frame_folder(task.scratch, task.name)
        points = np.asarray(_load(folder, SCALE_POINTS))
        rows, columns = points[:, 0].astype(np.int64), points[:, 1].astype(np.int64)
        error = sparse_error(points[:, 2], np.asarray(_load(folder, WALLS_DEPTH))[rows, columns])
    outcome, reason = frame_choice(task.trusted, task.timed_out, task.has_fallback, error,
                                   task.scale_points, task.settings)
    return ChoiceResult(task.name, outcome, reason, error)


def load_walls(scratch: Path, name: str) -> tuple[np.ndarray, np.ndarray]:
    """The Walls_Depth and its normals of a Prior_Frame with one, walls_depth.npy and
    walls_normal.npy, read into memory, so they outlive the scratch folder."""
    folder = frame_folder(scratch, name)
    return (np.array(np.load(folder / WALLS_DEPTH, allow_pickle=False)),
            np.array(np.load(folder / WALLS_NORMAL, allow_pickle=False)))
