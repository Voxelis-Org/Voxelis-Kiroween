"""The Frames_Step, which picks the frames of the Job's video that the camera solve works from.

The step reads the Used_Part of the video through the ffmpeg command line, scores every frame,
cuts the still ends, drops the jolted frames and picks sharp frames spread over the camera's
movement. It also finds the pixels stuck to the camera, for the Mask. It writes the Picks, their
grey copies, the Mask and frames.json to its Work_Folder, work/frames/, where the
Camera_Solve_Step reads them. Its settings are in [frames] in config.toml, and settings.py reads
and checks them.

The package is built like the Fake_Step. `work(step)` is the step's own code, which the
Step_Runner calls once it has emptied work/frames/ (Req 11.11). `run(job_id, preset=None, *,
config=None)` runs the step through the Step_Runner and returns its Step_Report. `__main__.py`
is the CLI, `uv run python -m splatpipe.frames --job <id> [--preset <name>]`.

`work` goes in this order:

1. The settings, every one checked before any tool runs (Req 9.5–9.7).
2. The probe: ffprobe gives the video's stream, Displayed_Shape and colour transfer, and the
   step works out the Score_Frame's size and the Pick_Size from them (Req 3.1, 8.1).
3. The Score_Pass: ffmpeg decodes the Job_Record's Used_Part in grey, and each Score_Frame goes
   to a new Scorer and then to a new SampleBuffer. Nothing else keeps a frame, so the step holds
   at most the frame before, the current frame and the buffer's frames (Req 1.12).
4. Choosing: the Still_Trim, Target_Picks, the Jolt_Filter, which is skipped with a warning when
   it keeps fewer frames than Target_Picks (Req 5.2–5.4), the Picker and the Mask. With fewer
   Picks than min_picks, the step fails here, before the Pick_Pass (Req 6.9, 6.10). The
   Score_Pass and the choosing run in one helper that gives back no Score_Frame, so the
   SampleBuffer's frames are freed before the Pick_Pass.
5. The Pick_Pass: ffmpeg decodes the Used_Part again, in colour at the Pick_Size, with the same
   video, stream and filters but for the size and the format (Req 1.7). Each frame whose
   Frame_Time is a Pick's is saved once, as its Colour_Pick in images/ and its Grey_Copy in grey/
   (Req 8.2–8.4). A Pick that ffmpeg doesn't give back fails the step (Req 8.9).
6. mask.png when the step uses the Mask, then frames.json, last of all, so a Work_Folder without
   frames.json is unfinished (Req 8.5, 8.7).

The step writes nothing before the Pick_Pass and nothing to out/, where only the Step_Runner
writes the report (Req 8.8). It adds each count as an int as soon as it's known, and each of the
timings probe, score_pass, choose and pick_pass when its part finishes, so a failed report holds
what the step had worked out before it failed (Req 10.1–10.4).

`work` reads the module constant TOOLS each time it's called, so a test that replaces the tools
with monkeypatch reaches the step through run_step and the CLI.

Every file in this package imports only the standard library, numpy, cv2 and splatpipe.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

from splatpipe.errors import ExpectedFailure
from splatpipe.frames.choose import Kept, jolt_keep, motion_parts, pick, still_trim, target_picks
from splatpipe.frames.ffmpeg import DecodePlan, VideoInfo, decode, probe_stream
from splatpipe.frames.files import (GREY, IMAGES, MASK, RECORD, frames_record, grey_copy,
                                    pick_name, write_jpeg, write_png, write_record)
from splatpipe.frames.mask import MaskChoice, SampleBuffer, decide_mask, mask_image
from splatpipe.frames.scores import Scorer, Scores, pick_size, score_shape
from splatpipe.frames.settings import FramesSettings, read_settings
# Config comes through splatpipe.step, as for the Fake_Step.
from splatpipe.step import Config, StepContext, run_step

# The Req 1.11 failure: the Score_Pass gave no Score_Frame inside the Used_Part.
_NO_FRAMES = ("No frames could be read from the chosen part of the video. Check that the video "
              "plays, then run the step again.")


@dataclass(frozen=True)
class Tools:
    """The tools the step runs. TOOLS holds the real ones, and tests put stand-ins in its place."""

    # probe(video, timeout_seconds), like ffmpeg.probe_stream.
    probe: Callable[[Path, float], VideoInfo]
    # decode(plan, timeout_seconds, on_frame), like ffmpeg.decode, which calls
    # on_frame(Frame_Time, frame) for each frame of the plan's Used_Part, in order.
    decode: Callable[[DecodePlan, float, Callable[[float, np.ndarray], None]], None]


# The real tools. work reads TOOLS each time it's called, so a test can monkeypatch it.
TOOLS = Tools(probe=probe_stream, decode=decode)


@dataclass(frozen=True)
class _Chosen:
    """What the Score_Pass and the choosing give the Pick_Pass and the Frames_Record.

    It holds no Score_Frame, so the SampleBuffer's frames are freed before the Pick_Pass.
    """

    scores: Scores  # every Score_Frame's Frame_Time and scores
    kept: Kept  # the Kept_Range
    target_picks: int
    jolt_filter_skipped: bool
    picks: tuple[int, ...]  # the Picks' Frame_Numbers in the Used_Part, in Frame_Time order
    mask: MaskChoice


def run(job_id: str, preset: str | None = None, *,
        config: Config | None = None) -> dict[str, Any]:
    """Run the frames step on the Job `job_id` and return its Step_Report.

    `preset` replaces the Job_Record's Preset, and `config` defaults to load_config().
    """
    return run_step("frames", "splatpipe.frames", job_id, preset=preset, config=config)


def work(step: StepContext) -> None:
    """The frames step's own code. The Step_Runner turns what it raises into a failed report."""
    tools = TOOLS  # read now, so a test's monkeypatch is seen
    cfg = read_settings(step.settings, step.name, step.preset)
    info, score_size, size = _probe(step, tools, cfg)
    chosen = _score_and_choose(step, tools, cfg, _plan(step, info, score_size, colour=False))
    _pick_pass(step, tools, cfg, _plan(step, info, size, colour=True), chosen)
    _write_mask_and_record(step, info, size, chosen)


def _probe(step: StepContext, tools: Tools,
           cfg: FramesSettings) -> tuple[VideoInfo, tuple[int, int], tuple[int, int]]:
    """Probe the video, then work out the Score_Frame's size and the Pick_Size, timed as probe."""
    with step.timed("probe"):
        info = tools.probe(step.job.video_path, cfg.ffprobe_timeout_seconds)
        score_size = score_shape(info.width, info.height, cfg.score_width)
        size = pick_size(info.width, info.height, cfg.long_edge)
        step.add_count("pick_width", int(size[0]))
        step.add_count("pick_height", int(size[1]))
    return info, score_size, size


def _plan(step: StepContext, info: VideoInfo, size: tuple[int, int], *,
          colour: bool) -> DecodePlan:
    """A Decode_Pass of the Job's video and stream over the Job_Record's Used_Part (Req 1.1),
    with frames of `size` in colour or in grey. Both passes are made here, so they differ only
    in the size and the format (Req 1.7)."""
    record = step.job.record
    width, height = size
    return DecodePlan(video=step.job.video_path, stream_index=info.stream_index,
                      start=record.start_seconds, end=record.end_seconds, width=width,
                      height=height, colour=colour)


def _score_and_choose(step: StepContext, tools: Tools, cfg: FramesSettings,
                      plan: DecodePlan) -> _Chosen:
    """Run the Score_Pass `plan`, then choose the Picks and decide the Mask.

    The Scorer and the SampleBuffer don't outlive this call, so the Score_Frames they hold are
    freed when it returns, before the Pick_Pass (Req 1.12).
    """
    scores, buffer = _score_pass(step, tools, cfg, plan)
    return _choose(step, cfg, scores, buffer)


def _score_pass(step: StepContext, tools: Tools, cfg: FramesSettings,
                plan: DecodePlan) -> tuple[Scores, SampleBuffer]:
    """The Score_Pass, timed as score_pass, with decoded_frames added. With no Score_Frame in the
    Used_Part, the step fails (Req 1.11)."""
    with step.timed("score_pass"):
        scores, buffer = _scored(tools, cfg, plan)
        if scores.times.size == 0:
            raise ExpectedFailure(_NO_FRAMES)
        step.add_count("decoded_frames", int(scores.times.size))
    return scores, buffer


def _scored(tools: Tools, cfg: FramesSettings, plan: DecodePlan) -> tuple[Scores, SampleBuffer]:
    """Decode `plan`, giving each frame to a new Scorer and then to a new SampleBuffer, and give
    back the Scores and the buffer. on_frame keeps no frame itself."""
    scorer, buffer = Scorer(), SampleBuffer(cfg.mask_samples)

    def on_frame(time: float, frame: np.ndarray) -> None:
        scorer.add(time, frame)
        buffer.add(frame)

    tools.decode(plan, cfg.ffmpeg_timeout_seconds, on_frame)
    return scorer.result(), buffer


def _choose(step: StepContext, cfg: FramesSettings, scores: Scores,
            buffer: SampleBuffer) -> _Chosen:
    """The Still_Trim, Target_Picks, the Jolt_Filter, the Picker and the Mask, timed as choose."""
    with step.timed("choose"):
        kept = _still_trim(step, cfg, scores)
        target = target_picks(kept.end_seconds - kept.start_seconds, base_picks=cfg.base_picks,
                              seconds_per_pick=cfg.seconds_per_pick, max_picks=cfg.max_picks)
        step.add_count("target_picks", int(target))
        candidates, skipped = _candidates(step, cfg, scores, kept, target)
        picks = _picks(step, cfg, scores, kept, target, candidates)
        mask = _mask(step, cfg, buffer, kept)
    return _Chosen(scores=scores, kept=kept, target_picks=int(target),
                   jolt_filter_skipped=skipped, picks=picks, mask=mask)


def _still_trim(step: StepContext, cfg: FramesSettings, scores: Scores) -> Kept:
    """The Kept_Range, from the Still_Trim over every Score_Frame of the Used_Part (Req 4)."""
    kept = still_trim(scores.times, scores.motion, step.job.record.end_seconds,
                      still_motion=cfg.still_motion, window_seconds=cfg.motion_window_seconds,
                      min_still_seconds=cfg.min_still_seconds, blip_seconds=cfg.blip_seconds)
    step.add_count("still_start_frames", int(kept.first))
    step.add_count("still_end_frames", int(scores.times.size - 1 - kept.last))
    step.add_count("kept_frames", int(kept.last - kept.first + 1))
    return kept


def _candidates(step: StepContext, cfg: FramesSettings, scores: Scores, kept: Kept,
                target: int) -> tuple[np.ndarray, bool]:
    """The Candidates, counted from 0 at the Kept_Range's first frame, and whether the
    Jolt_Filter was skipped.

    When the Jolt_Filter keeps fewer frames than Target_Picks, it's skipped with a warning and
    every Kept_Range frame is a Candidate (Req 5.2, 5.3). Otherwise the Candidates are exactly
    the frames it kept (Req 5.4).
    """
    span = _span(kept)
    keep = jolt_keep(scores.times[span], scores.sharpness[span], ratio=cfg.jolt_ratio,
                     window_seconds=cfg.jolt_window_seconds)
    kept_by_filter = int(np.count_nonzero(keep))
    if kept_by_filter < target:
        step.add_count("jolt_dropped_frames", 0)
        step.add_warning(f"The jolt filter kept only {kept_by_filter} frames, fewer than the "
                         f"{int(target)} picks this job aims for, so it was skipped and every "
                         "kept frame could be picked.")
        return np.arange(keep.size), True
    step.add_count("jolt_dropped_frames", int(keep.size - kept_by_filter))
    return np.flatnonzero(keep), False


def _picks(step: StepContext, cfg: FramesSettings, scores: Scores, kept: Kept, target: int,
           candidates: np.ndarray) -> tuple[int, ...]:
    """The Picks' Frame_Numbers in the Used_Part, in Frame_Time order (Req 6.3–6.7).

    With fewer Picks than min_picks, the step fails before the Pick_Pass (Req 6.9, 6.10).
    """
    span = _span(kept)
    parts = motion_parts(scores.times[span], scores.motion[span], kept.end_seconds, target)
    # pick counts from 0 at the Kept_Range's first frame, so its first Frame_Number is added.
    picks = tuple((pick(parts, candidates, scores.sharpness[span]) + kept.first).tolist())
    step.add_count("picks", len(picks))
    if len(picks) < cfg.min_picks:
        raise ExpectedFailure(f"Only {len(picks)} of the video's frames could be picked, but the "
                              f"step needs at least {cfg.min_picks}. Film a longer video that "
                              "moves slowly and steadily through the place, in good light, "
                              "without long still parts, then make a new job from it.")
    return picks


def _mask(step: StepContext, cfg: FramesSettings, buffer: SampleBuffer, kept: Kept) -> MaskChoice:
    """Decide the Mask from the Mask_Samples, the buffered Score_Frames inside the Kept_Range
    (Req 7.1–7.7)."""
    samples = [frame for _, frame in buffer.samples_in(kept.first, kept.last)]
    mask = decide_mask(samples, ratio=cfg.mask_ratio, min_coverage=cfg.mask_min_coverage,
                       max_coverage=cfg.mask_max_coverage)
    stuck = 0 if mask.stuck is None else int(np.count_nonzero(mask.stuck))
    step.add_count("mask_samples", int(mask.samples))
    step.add_count("mask_pixels", stuck)
    step.add_count("mask_used", int(mask.used))
    return mask


def _span(kept: Kept) -> slice:
    """The Kept_Range's Score_Frames, as a slice of the Used_Part's arrays."""
    return slice(kept.first, kept.last + 1)


def _pick_pass(step: StepContext, tools: Tools, cfg: FramesSettings, plan: DecodePlan,
               chosen: _Chosen) -> None:
    """The Pick_Pass `plan`, timed as pick_pass. Each frame whose Frame_Time is a Pick's is
    saved once, as its Colour_Pick and its Grey_Copy (Req 8.2–8.4). A Pick that ffmpeg doesn't
    give back fails the step (Req 8.9)."""
    images, grey = step.work_folder / IMAGES, step.work_folder / GREY
    # Each Pick's number by its Frame_Time. Both passes run the same input options and filters
    # but for the size and the format, so ffmpeg gives the same frames with the same times.
    wanted = {float(chosen.scores.times[frame]): number
              for number, frame in enumerate(chosen.picks)}

    def on_frame(time: float, frame: np.ndarray) -> None:
        number = wanted.pop(time, None)  # popped, so each Pick is saved once
        if number is not None:
            name = pick_name(number)
            write_jpeg(images / name, frame, cfg.jpeg_quality)
            write_jpeg(grey / name, grey_copy(frame), cfg.jpeg_quality)

    with step.timed("pick_pass"):
        images.mkdir()
        grey.mkdir()
        tools.decode(plan, cfg.ffmpeg_timeout_seconds, on_frame)
        if wanted:
            raise ExpectedFailure(f"ffmpeg gave back {len(chosen.picks) - len(wanted)} of the "
                                  f"{len(chosen.picks)} picks when it read the video again in "
                                  "colour. Check that the video plays, then run the step again.")


def _write_mask_and_record(step: StepContext, info: VideoInfo, size: tuple[int, int],
                           chosen: _Chosen) -> None:
    """Write mask.png when the step uses the Mask, then frames.json, last of all (Req 8.5, 8.7)."""
    width, height = size
    if chosen.mask.used:
        stuck = cast(np.ndarray, chosen.mask.stuck)  # a used Mask always has its Stuck_Pixels
        write_png(step.work_folder / MASK, mask_image(stuck, width, height))
    record = frames_record(colour_transfer=info.colour_transfer,
                           tone_mapped=False,  # both passes decode without the Tone_Map
                           kept=chosen.kept, target_picks=chosen.target_picks,
                           jolt_filter_skipped=chosen.jolt_filter_skipped, pick_width=width,
                           pick_height=height, mask=chosen.mask, scores=chosen.scores,
                           picks=chosen.picks)
    write_record(step.work_folder / RECORD, record)
