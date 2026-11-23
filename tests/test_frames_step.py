"""Tests for the Frames_Step in splatpipe/frames/, run through splatpipe.frames.run, so the
Step_Runner runs it and writes its real report.

Each test makes its Job with new_job from tests/conftest.py, in the temporary Jobs_Folder. Every
test but the two real ffmpeg ones at the end puts a FakeTools in place of ffprobe, ffmpeg's
filter check and the ffmpeg decode with fake_tools, so only those two need ffmpeg. new_job's
video is 10 s long, and the Job uses all of it unless the test gives a Trim. Its Preset is fast,
whose base_picks is 2 in the test config, so Target_Picks is one Pick for every 2 s of
Kept_Seconds, rounded up and at least 2, unless a test sets base_picks higher.

The stand-in video is 640 × 480, and set_frames sets score_width to 8 and long_edge to 16, so
each Score_Frame is 8 × 6 pixels and each Pick 16 × 12. Every picture a test gives FakeTools has
the Score_Frame's size, 6 rows of 8 pixels, so each Score_Frame is its picture, and each Pick is
its picture scaled up 2 times, with its grey in all three colour channels.

Each picture is a checkerboard of +a and −a on a base brightness, which makes its scores easy to
work out. Its Laplacian is ±8a at every pixel, OpenCV's reflect-101 border included, so its
Sharpness is 64a², and its Contrast is a. When the base brightness changes by d from one frame
to the next and the amplitude by at most d, every pixel changes by d plus or minus the
amplitude's change, so the Motion is d.

- The walk: 20 frames 0.5 s apart, from 0 s to 9.5 s. Frame k's base brightness is 25 + 10k and
  its amplitude 20, so its Sharpness is 25600, its Motion 10, or 0 for frame 0, and its Contrast
  20. Every frame moves, so the Still_Trim keeps all 20 and the Kept_Range runs to the Used_Part's
  end, 10 s, which gives a Target_Picks of 5. Each frame moves as much as the one before, so each
  of the 5 Motion_Parts holds 4 frames, and with every Sharpness the same the earliest of each
  wins: the Picks are frames 0, 4, 8, 12 and 16.
- The walk with a stuck corner: the walk with the 2 rows of 3 pixels at its bottom right at 0 in
  every frame, like a watermark. Those 6 pixels never change and the other 42 change by 10 from
  each frame to the next, so the 6 are the Stuck_Pixels, 0.125 of the frame, and the step uses
  the Mask. The Motion is 8.75 at every frame but the first, so each Motion_Part still holds 4
  frames. The corner makes the Sharpness rise a little from each frame to the next, so the
  Jolt_Filter still keeps every frame. Each frame's mean brightness is 8.75 above the one
  before's.
- The jolts: 40 frames 0.25 s apart, from 0 s to 9.75 s, whose base brightness is 100 and 140 by
  turns, so every frame but the first moves by 40. Frames 10, 11, 26 and 27 are blurred, with an
  amplitude of 3 and a Sharpness of 576, and the others are sharp, with an amplitude of 30 and a
  Sharpness of 57600. A frame's Window of 1 s holds the 2 frames on each side of it, so each
  blurred frame's Window has a sharp median, and the Jolt_Filter keeps the 36 sharp frames. The
  tests set Target_Picks with base_picks. With the same Motion at every frame, 40 Motion_Parts
  hold one frame each, and 20 hold two each, frames 2j and 2j + 1.

The HDR tests run the walk with a colour transfer on the stand-in video. For an HDR_Clip,
arib-std-b67 (HLG) or smpte2084 (PQ), the step asks FakeTools' has_filters for zscale and
tonemap, which it says this ffmpeg has unless it's built with no_filters. So both passes get a
ToneMap of the test config's tonemap_npl and tonemap_desat, 100.0 and 0.0, unless set_frames
changes them. Built with tone_map_failure, FakeTools' first pass with a ToneMap gives the walk's
first 3 frames and then fails with a line of ffmpeg's, as the real decode does when ffmpeg exits
with a non-zero code, so the Score_Pass runs again and gets all 20. A bt709 video and one with no
colour transfer get no filter check and no ToneMap.

The failed runs each make one failure: the jolts with a min_picks above the Picks they give, a
Trim with no frame in it, a Pick_Pass that FakeTools stops early, a Path.write_bytes that fails
like a full disk, the real decode's failure for a missing ffmpeg, which FakeTools raises, and bad
settings. run_failed checks that each failed with an Expected_Failure whose Readable_Error names
what its requirement says and holds no traceback (Req 11.10, 13.9). The CLI tests run
splatpipe.frames.__main__.main in this process, so it uses the stand-in too.

The memory test runs the step on the long stream: 5,000 frames 2 ms apart, from 0 s to 9.998 s,
which FakeTools.stream makes one at a time as its decode reaches them, so nothing holds them all.
Each is a sharp checkerboard like the jolts', whose base brightness is 100 and 140 by turns, so
every frame moves and the step goes on to the Pick_Pass. set_frames sets mask_samples to 3, so
the SampleBuffer holds at most 6 frames. FakeTools gives the step a new array for each pair, not
make's picture, so the test takes its weakrefs from the frames the step's on_frame gets, through a
watch around the stand-in's decode that keeps no frame itself. CPython frees an array as soon as
nothing refers to it, so a frame's weakref is dead once the Scorer and the SampleBuffer have both
let it go. After each frame, at most 8 may be alive: the frame before, the current frame and the
buffer's 6. When the Pick_Pass starts, none may be, because the helper that runs the Score_Pass
and the choosing gives back no Score_Frame (Req 1.12, 13.7).

The Step_List test runs the walk through run_job with no steps given, so the real Step_List runs:
the Frames_Step, then the Train_Step. No camera solve has run, so once the Frames_Step is done the
Train_Step finds no work/cameras/images/ and fails, and so does the Job, with the Train_Step's
message for the missing folder as the Job_Summary's readable_error (splat-training spec, Req 1.2,
11.5, 11.11).

The two real ffmpeg tests run the step with the real TOOLS, so they're skipped, with the reason
"ffmpeg and ffprobe aren't installed", when either tool is missing, like the real tests in
tests/test_frames_ffmpeg.py (Req 13.5). Each makes its clip in tmp_path with ffmpeg, at 10 frames
a second and 64 × 48, smaller than the test config's long_edge of 1600. The first clip is lavfi's
testsrc, 3 s long. The second, the ramp, is 2 s long, and its frame k is flat grey at a
brightness of 16 + 10k, from 16 to 206, so neighbouring frames are 10 apart. The ramp's test runs
the Job through run_job with a Step_List of the Frames_Step's entry alone, so the Job is done once
the step is (Req 11.4, and splat-training spec, Req 11.10). The runs that make the clips have a
limit of 60 s, and the step's own ffprobe and ffmpeg runs take the test config's limits, also
60 s (Req 13.11). make_job would refuse either clip, which is bigger than the test config's
max_video_bytes of 1024, so each test makes the Job with new_job and then copies its clip over
the Job's input/video.mp4. The step reads only that video and the Job_Record's Trim, which runs
from 0 to 10 s, so ffmpeg reads the clip to its end, and Frame_Number k is the clip's frame k.
"""

import errno
import json
import shutil
import subprocess
import weakref
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any, NoReturn

import cv2
import numpy as np
import pytest

from splatpipe import frames
from splatpipe.config import load_config
from splatpipe.errors import ExpectedFailure
from splatpipe.frames.__main__ import main as frames_main
from splatpipe.frames.ffmpeg import DecodePlan, ToneMap, VideoInfo, decode, decode_command
from splatpipe.jobs import open_job
from splatpipe.jsonio import check_json_value, read_json
from splatpipe.run import StepEntry, run_job

# A Frame_Time in seconds and its picture, as FakeTools takes them.
Pair = tuple[float, np.ndarray]

# The stand-in video's first video stream that isn't cover art: at index 1, after an audio
# stream, 640 × 480, with no colour transfer.
INFO = VideoInfo(stream_index=1, width=640, height=480, colour_transfer=None)
# The settings for set_frames that keep every frame tiny: Score_Frames of 8 × 6 and Picks of
# 16 × 12.
TINY = {"score_width": 8, "long_edge": 16}
SCORE_WIDTH, SCORE_HEIGHT = 8, 6
PICK_WIDTH, PICK_HEIGHT = 16, 12

# The 22 Shared_Settings of [frames] (Req 9.2).
SHARED_SETTINGS = ("score_width", "tonemap_npl", "tonemap_desat", "still_motion",
                   "motion_window_seconds", "min_still_seconds", "blip_seconds", "jolt_ratio",
                   "jolt_window_seconds", "mask_samples", "mask_ratio", "mask_min_coverage",
                   "mask_max_coverage", "jpeg_quality", "long_edge", "seconds_per_pick",
                   "max_picks", "min_picks", "min_vcpus", "min_memory_gb",
                   "ffmpeg_timeout_seconds", "ffprobe_timeout_seconds")
# The Frames_Report's timings (Req 10.2).
TIMINGS = ("probe", "score_pass", "choose", "pick_pass")

# +1 and −1 in turn along every row and column, at the Score_Frame's size.
CHECKERBOARD = np.where(np.indices((SCORE_HEIGHT, SCORE_WIDTH)).sum(axis=0) % 2 == 0, 1, -1)
# The stuck corner: the 2 rows of 3 pixels at the bottom right, 0 in every frame of the walk.
CORNER = np.zeros((SCORE_HEIGHT, SCORE_WIDTH), dtype=bool)
CORNER[-2:, -3:] = True
# The blurred frames among the jolts: the frames of Motion_Parts 5 and 13 when 20 parts hold two
# frames each.
BLURRED = (10, 11, 26, 27)
# Settings that ask for more Picks than the jolts give. With a Target_Picks of 20, Motion_Parts 5
# and 13 hold only blurred frames, so no Candidate, and the Picker gives 18 Picks, fewer than the
# min_picks of 20.
TOO_FEW_PICKS = {"min_picks": 20, "base_picks": {"fast": 20}}
FULL_DISK = "No space left on device"  # the reason the stand-in full disk gives


def checkerboard(base: int, amplitude: int) -> np.ndarray:
    """A picture at the Score_Frame's size of base + amplitude and base − amplitude in a
    checkerboard."""
    return (base + amplitude * CHECKERBOARD).astype(np.uint8)


def walk(*, corner: bool = False) -> list[Pair]:
    """The walk's 20 frames, 0.5 s apart from 0 s, with the stuck corner at 0 when `corner`."""
    pictures = (checkerboard(25 + 10 * k, 20) for k in range(20))
    return [(k * 0.5, np.where(CORNER, 0, picture).astype(np.uint8) if corner else picture)
            for k, picture in enumerate(pictures)]


def jolts() -> list[Pair]:
    """The jolts' 40 frames, 0.25 s apart from 0 s, with frames 10, 11, 26 and 27 blurred."""
    return [(k * 0.25, checkerboard(140 if k % 2 else 100, 3 if k in BLURRED else 30))
            for k in range(40)]


def run_done(job_id: str) -> dict[str, Any]:
    """Run the frames step on the Job through splatpipe.frames.run, check that it's done, and give
    back its report."""
    report = frames.run(job_id)
    assert report["outcome"] == "done", report["error"]
    return report


def run_failed(job_id: str, *phrases: str) -> dict[str, Any]:
    """Run the frames step on the Job through splatpipe.frames.run, check that it failed with an
    Expected_Failure whose Readable_Error holds each of `phrases` and no traceback, and give back
    its report (Req 11.10, 13.9)."""
    report = frames.run(job_id)
    assert report["outcome"] == "failed"
    error = report["error"]
    assert error["expected"] is True, error
    readable = error["readable"]
    for phrase in phrases:
        assert phrase in readable, readable
    assert "Traceback" not in readable
    return report


def folder_of(job_id: str) -> Path:
    """The Job_Folder of `job_id` in the temporary config's Jobs_Folder."""
    return load_config().jobs().jobs_folder / job_id


def work_folder(job_id: str) -> Path:
    """The frames step's Work_Folder, work/frames/, in the Job's folder."""
    return folder_of(job_id) / "work" / "frames"


def record_of(job_id: str) -> Any:
    """The Frames_Record, work/frames/frames.json, read back."""
    return read_json(work_folder(job_id) / "frames.json")


def frame_numbers(record: dict[str, Any]) -> list[int]:
    """The Frame_Numbers of the Picks in the Frames_Record, in order."""
    return [entry["frame_number"] for entry in record["picks"]]


def entries(folder: Path) -> list[str]:
    """Every file and folder inside `folder`, by its path from `folder`, sorted."""
    return sorted(path.relative_to(folder).as_posix() for path in folder.rglob("*"))


def layout(picks: int, *, mask: bool) -> list[str]:
    """What entries gives for a Work_Folder with `picks` Picks: images/ and grey/, each with the
    names from 000000.jpg, and frames.json, plus mask.png when `mask` is True."""
    names = [f"{number:06d}.jpg" for number in range(picks)]
    return sorted(["images", "grey", "frames.json", *(["mask.png"] if mask else []),
                   *(f"images/{name}" for name in names), *(f"grey/{name}" for name in names)])


def read_image(path: Path) -> np.ndarray:
    """The image in the file at `path`, as cv2.imdecode reads it with its channels and bit depth
    unchanged."""
    image = cv2.imdecode(np.frombuffer(path.read_bytes(), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    assert image is not None, f"cv2.imdecode couldn't read {path}"
    return image


# The Work_Folder (Req 7.4, 7.5, 7.8, 8.2–8.5)

@pytest.mark.parametrize("corner", [False, True], ids=["no stuck corner", "a stuck corner"])
def test_the_work_folder_holds_the_picks_grey_copies_and_frames_json_and_a_mask_for_a_corner(
        new_job: Callable[..., str], fake_tools: Callable[..., Any],
        set_frames: Callable[..., None], corner: bool) -> None:
    job_id = new_job()
    fake_tools(INFO, walk(corner=corner))
    set_frames(**TINY)
    run_done(job_id)
    work = work_folder(job_id)
    # The 5 Picks have the same names in images/ and grey/, and mask.png is there only for the
    # stuck corner (Req 7.4, 7.5, 8.4, 8.5).
    assert entries(work) == layout(5, mask=corner)
    for number in range(5):
        name = f"{number:06d}.jpg"
        # Each Colour_Pick is in colour at the Pick_Size, and its Grey_Copy is grey at the same
        # size (Req 8.2, 8.3).
        assert read_image(work / "images" / name).shape == (PICK_HEIGHT, PICK_WIDTH, 3)
        assert read_image(work / "grey" / name).shape == (PICK_HEIGHT, PICK_WIDTH)
    if corner:
        # An 8-bit grey PNG at the Pick_Size: the stuck corner, scaled up 2 times, is black, 0,
        # and every other pixel white, 255 (Req 7.8). The PNG's own header holds the bit depth at
        # byte 24 and the colour type at byte 25, 0 for grey, because a grey PNG of 1, 2 or 4 bits
        # would read back as 8-bit too.
        mask = work / "mask.png"
        image = read_image(mask)
        assert image.dtype == np.uint8
        stuck = CORNER.repeat(2, axis=0).repeat(2, axis=1)
        assert image.tolist() == np.where(stuck, 0, 255).tolist()
        data = mask.read_bytes()
        assert (data[24], data[25]) == (8, 0)


def test_each_colour_pick_shows_the_frame_its_frames_json_entry_names(
        new_job: Callable[..., str], fake_tools: Callable[..., Any],
        set_frames: Callable[..., None]) -> None:
    # The walk with the stuck corner, whose frames' mean brightness is 8.75 apart. So a Colour_Pick
    # whose three channels each have a mean within 4 of a frame's mean brightness can only have
    # come from that frame (Req 8.2).
    job_id = new_job()
    pairs = walk(corner=True)
    fake_tools(INFO, pairs)
    set_frames(**TINY)
    run_done(job_id)
    picks = record_of(job_id)["picks"]
    assert len(picks) == 5
    for entry in picks:
        # The Job uses the whole video, so Frame_Number k is the stream's frame k.
        _, picture = pairs[entry["frame_number"]]
        colour = read_image(work_folder(job_id) / "images" / entry["file"])
        channel_means = colour.mean(axis=(0, 1))
        assert np.abs(channel_means - picture.mean()).max() < 4, entry


# The Frames_Report and the Frames_Record (Req 3.6, 8.6, 10.1, 10.2, 10.4, 10.6)

def test_the_report_holds_every_count_as_an_int_the_four_timings_and_every_setting(
        new_job: Callable[..., str], fake_tools: Callable[..., Any],
        set_frames: Callable[..., None]) -> None:
    job_id = new_job()
    fake_tools(INFO, walk(corner=True))
    set_frames(**TINY)
    report = run_done(job_id)
    assert read_json(folder_of(job_id) / "out" / "reports" / "frames.json") == report  # Req 10.6
    # Every count of Req 10.1, each an int (Req 10.4). The 6 Stuck_Pixels are counted at the
    # Score_Frame's size.
    assert report["counts"] == {"decoded_frames": 20, "still_start_frames": 0,
                                "still_end_frames": 0, "kept_frames": 20,
                                "jolt_dropped_frames": 0, "target_picks": 5, "picks": 5,
                                "mask_samples": 20, "mask_pixels": 6, "mask_used": 1,
                                "pick_width": PICK_WIDTH, "pick_height": PICK_HEIGHT}
    assert all(type(value) is int for value in report["counts"].values())
    # The four timings, each a number of seconds (Req 10.2).
    assert sorted(report["timings"]) == sorted(TIMINGS)
    assert all(type(seconds) is float and seconds >= 0
               for seconds in report["timings"].values())
    # The 22 Shared_Settings and the Preset's base_picks, as the test config has them (Req 10.6).
    assert sorted(report["settings"]) == sorted([*SHARED_SETTINGS, "base_picks"])
    assert report["settings"] == load_config().step_settings("frames", "fast")


def test_frames_json_holds_every_field_as_plain_json_values(
        new_job: Callable[..., str], fake_tools: Callable[..., Any],
        set_frames: Callable[..., None]) -> None:
    job_id = new_job()
    fake_tools(INFO, walk())
    set_frames(**TINY)
    run_done(job_id)
    record = record_of(job_id)
    check_json_value(record, "frames")  # Req 8.6
    # Each Pick is frame 4i of the walk, with its Frame_Time and its scores (Req 3.6, 8.6).
    expected = {
        "colour_transfer": None,
        "tone_mapped": False,
        "still_start_seconds": 0.0,
        "still_end_seconds": 0.0,
        "kept_start_seconds": 0.0,
        "kept_end_seconds": 10.0,
        "target_picks": 5,
        "jolt_filter_skipped": False,
        "pick_width": PICK_WIDTH,
        "pick_height": PICK_HEIGHT,
        "mask_coverage": 0.0,
        "mask_used": False,
        "mask_file": None,
        "picks": [{"number": number, "file": f"{number:06d}.jpg", "frame_number": 4 * number,
                   "seconds": 2.0 * number, "sharpness": 25600.0,
                   "motion": 0.0 if number == 0 else 10.0, "contrast": 20.0}
                  for number in range(5)],
    }
    assert record == expected
    # The same JSON text too, so the fields come in the design's order and each value is of the
    # expected kind: false isn't 0, and 16 isn't 16.0.
    assert json.dumps(record) == json.dumps(expected)


# The Used_Part (Req 1.1, 1.7)

def test_both_passes_decode_the_jobs_video_over_the_job_records_trim(
        new_job: Callable[..., str], fake_tools: Callable[..., Any],
        set_frames: Callable[..., None]) -> None:
    job_id = new_job(start=2.5, end=7.25)
    tools = fake_tools(INFO, walk())
    set_frames(**TINY)
    report = run_done(job_id)
    job = open_job(job_id, load_config())
    assert (job.record.start_seconds, job.record.end_seconds) == (2.5, 7.25)
    # ffprobe reads the Job's copy of the video, with the limit from [frames] (Req 1.1).
    assert tools.probes == [(job.video_path, 60.0)]
    # Both passes read the same video and stream over the Trim, and differ only in the size and
    # the format (Req 1.1, 1.7).
    score_pass = DecodePlan(video=job.video_path, stream_index=INFO.stream_index, start=2.5,
                            end=7.25, width=SCORE_WIDTH, height=SCORE_HEIGHT, colour=False)
    pick_pass = replace(score_pass, width=PICK_WIDTH, height=PICK_HEIGHT, colour=True)
    assert tools.plans == [score_pass, pick_pass]
    # FakeTools gives the step only the 10 frames from 2.5 s to 7 s, and the Kept_Range runs to
    # the Trim's end.
    assert report["counts"]["decoded_frames"] == 10
    record = record_of(job_id)
    assert (record["kept_start_seconds"], record["kept_end_seconds"]) == (2.5, 7.25)


# HDR video and the Tone_Map (Req 2.2–2.7, 9.4, 10.5)

# The colour transfers of an HDR_Clip, HLG and PQ, as ffprobe names them (A7).
HLG, PQ = "arib-std-b67", "smpte2084"
# The ToneMap of the test config's tonemap_npl = 100 and tonemap_desat = 0.0.
CONFIG_TONE_MAP = ToneMap(100.0, 0.0)
# The call the step makes to FakeTools' has_filters for an HDR_Clip: zscale and tonemap, with the
# test config's ffprobe_timeout_seconds of 60 as the limit.
FILTER_CHECK = (("zscale", "tonemap"), 60.0)
# The Tone_Map's filters with a tonemap_npl of 203.5 and a tonemap_desat of 0.25: the design's
# text written out, with only those two numbers changed from 100.0 and 0.0, and the hable curve
# and the BT.709 names as they are (A13).
TONE_MAP_203_5 = ("zscale=t=linear:npl=203.5,format=gbrpf32le,zscale=p=bt709,"
                  "tonemap=tonemap=hable:desat=0.25,zscale=t=bt709:m=bt709:r=tv,format=yuv420p")
# The last line of error output of FakeTools' failed pass with the Tone_Map, as ffmpeg prints it
# when zimg fails while it filters a frame.
TONE_MAP_ERROR = "Error while filtering: Generic error in an external library"


def with_transfer(transfer: str | None) -> VideoInfo:
    """The stand-in video, INFO, with the colour transfer `transfer`, or with none for None."""
    return replace(INFO, colour_transfer=transfer)


def tone_maps(plans: list[DecodePlan]) -> list[tuple[bool, ToneMap | None]]:
    """Each plan's colour and ToneMap, in the order FakeTools recorded them: colour False for a
    Score_Pass, in grey, and True for the Pick_Pass."""
    return [(plan.colour, plan.tone_map) for plan in plans]


@pytest.mark.parametrize("transfer", [pytest.param(HLG, id="HLG"), pytest.param(PQ, id="PQ")])
def test_an_hdr_clip_with_both_filters_runs_both_passes_with_the_tone_map(
        new_job: Callable[..., str], fake_tools: Callable[..., Any],
        set_frames: Callable[..., None], transfer: str) -> None:
    job_id = new_job()
    tools = fake_tools(with_transfer(transfer), walk())
    set_frames(**TINY)
    report = run_done(job_id)
    # One filter check, then both passes with the ToneMap of the test config's numbers, and no
    # warning (Req 2.2, 2.6, 9.4).
    assert tools.filter_checks == [FILTER_CHECK]
    assert tone_maps(tools.plans) == [(False, CONFIG_TONE_MAP), (True, CONFIG_TONE_MAP)]
    assert report["warnings"] == []
    # frames.json records the transfer and that the Tone_Map ran (Req 2.7).
    record = record_of(job_id)
    assert record["colour_transfer"] == transfer
    assert record["tone_mapped"] is True


def test_both_passes_tone_map_with_tonemap_npl_and_tonemap_desat_from_the_config(
        new_job: Callable[..., str], fake_tools: Callable[..., Any],
        set_frames: Callable[..., None]) -> None:
    job_id = new_job()
    tools = fake_tools(with_transfer(HLG), walk())
    set_frames(**TINY, tonemap_npl=203.5, tonemap_desat=0.25)
    run_done(job_id)
    # Both plans hold the two numbers set_frames gave (Req 2.2, 9.4).
    tone_map = ToneMap(203.5, 0.25)
    assert tone_maps(tools.plans) == [(False, tone_map), (True, tone_map)]
    # Each plan's ffmpeg command has the Tone_Map's filters with those numbers among its other
    # filters, and the hable curve and the BT.709 names unchanged.
    for plan in tools.plans:
        command = decode_command(plan)
        assert f",{TONE_MAP_203_5}," in command[command.index("-vf") + 1]


def test_an_hdr_clip_without_the_filters_runs_both_passes_without_the_tone_map_and_warns(
        new_job: Callable[..., str], fake_tools: Callable[..., Any],
        set_frames: Callable[..., None]) -> None:
    job_id = new_job()
    tools = fake_tools(with_transfer(PQ), walk(), no_filters=True)
    set_frames(**TINY)
    report = run_done(job_id)  # the step is done all the same
    assert tools.filter_checks == [FILTER_CHECK]
    assert tone_maps(tools.plans) == [(False, None), (True, None)]
    # The warning names the transfer, says the picks' colours may look flat, and says an ffmpeg
    # with zscale fixes them (Req 2.4, 10.5).
    assert report["warnings"] == [
        "The video is HDR (colour transfer smpte2084), but this ffmpeg has no zscale or tonemap "
        "filter, so the frames were read without tone mapping and the picks' colours may look "
        "flat. An ffmpeg built with zscale fixes the colours."]
    record = record_of(job_id)
    assert record["colour_transfer"] == PQ
    assert record["tone_mapped"] is False  # Req 2.7


def test_a_failed_pass_with_the_tone_map_runs_again_without_it_and_the_pick_pass_follows(
        new_job: Callable[..., str], fake_tools: Callable[..., Any],
        set_frames: Callable[..., None]) -> None:
    # The first pass, with the Tone_Map, gives the walk's first 3 frames and then fails.
    job_id = new_job()
    tools = fake_tools(with_transfer(HLG), walk(), tone_map_failure=(3, TONE_MAP_ERROR))
    set_frames(**TINY)
    report = run_done(job_id)
    # The Score_Pass runs again as it was, but without the Tone_Map, and the Pick_Pass takes that
    # choice (Req 2.5, 2.6).
    assert tone_maps(tools.plans) == [(False, CONFIG_TONE_MAP), (False, None), (True, None)]
    assert tools.plans[1] == replace(tools.plans[0], tone_map=None)
    # The warning holds ffmpeg's last line (Req 2.5, 10.5).
    assert report["warnings"] == [
        f"The video is HDR (colour transfer arib-std-b67), but ffmpeg failed to tone map it "
        f"({TONE_MAP_ERROR}), so the frames were read without tone mapping and the picks' "
        "colours may look flat. An ffmpeg built with a working zscale fixes the colours."]
    # The pass that ran again started a new Scorer and a new SampleBuffer, so the counts hold
    # only its 20 frames and none of the 3 the failed pass gave (Req 2.5).
    counts = report["counts"]
    assert (counts["decoded_frames"], counts["mask_samples"]) == (20, 20)
    assert record_of(job_id)["tone_mapped"] is False  # Req 2.7


@pytest.mark.parametrize("transfer", [pytest.param("bt709", id="bt709"),
                                      pytest.param(None, id="no transfer")])
def test_a_clip_that_isnt_hdr_gets_no_filter_check_and_no_tone_map(
        new_job: Callable[..., str], fake_tools: Callable[..., Any],
        set_frames: Callable[..., None], transfer: str | None) -> None:
    job_id = new_job()
    tools = fake_tools(with_transfer(transfer), walk())
    set_frames(**TINY)
    run_done(job_id)
    assert tools.filter_checks == []
    assert tone_maps(tools.plans) == [(False, None), (True, None)]  # Req 2.3
    # frames.json records the transfer ffprobe reported, or null when it reported none (Req 2.7).
    record = record_of(job_id)
    assert record["colour_transfer"] == transfer
    assert record["tone_mapped"] is False


# The Jolt_Filter (Req 5.2–5.4)

def test_a_jolt_filter_that_keeps_fewer_frames_than_target_picks_is_skipped_with_a_warning(
        new_job: Callable[..., str], fake_tools: Callable[..., Any],
        set_frames: Callable[..., None]) -> None:
    # A Target_Picks of 40, more than the 36 frames the Jolt_Filter keeps.
    job_id = new_job()
    fake_tools(INFO, jolts())
    set_frames(**TINY, base_picks={"fast": 40})
    report = run_done(job_id)
    assert report["warnings"] == [
        "The jolt filter kept only 36 frames, fewer than the 40 picks this job aims for, so it "
        "was skipped and every kept frame could be picked."]  # Req 5.3
    assert (report["counts"]["target_picks"], report["counts"]["jolt_dropped_frames"]) == (40, 0)
    record = record_of(job_id)
    assert record["jolt_filter_skipped"] is True
    # Every Kept_Range frame is a Candidate, and each of the 40 Motion_Parts holds one frame, so
    # every frame is a Pick, the blurred ones included (Req 5.2).
    assert frame_numbers(record) == list(range(40))


def test_with_enough_sharp_frames_the_candidates_are_exactly_the_frames_the_filter_kept(
        new_job: Callable[..., str], fake_tools: Callable[..., Any],
        set_frames: Callable[..., None]) -> None:
    # A Target_Picks of 20, no more than the 36 frames the Jolt_Filter keeps.
    job_id = new_job()
    fake_tools(INFO, jolts())
    set_frames(**TINY, base_picks={"fast": 20})
    report = run_done(job_id)
    assert report["warnings"] == []
    assert (report["counts"]["target_picks"], report["counts"]["jolt_dropped_frames"]) == (20, 4)
    record = record_of(job_id)
    assert record["jolt_filter_skipped"] is False
    # Motion_Part j holds frames 2j and 2j + 1. Each sharp pair gives its earlier frame, and the
    # blurred pairs of parts 5 and 13 hold no Candidate, so they give no Pick (Req 5.4).
    assert frame_numbers(record) == [2 * part for part in range(20) if part not in (5, 13)]


# out/ and running the step again (Req 8.8, 11.11)

def test_a_done_run_adds_nothing_to_out_but_its_report(
        new_job: Callable[..., str], fake_tools: Callable[..., Any],
        set_frames: Callable[..., None]) -> None:
    job_id = new_job()
    out = folder_of(job_id) / "out"
    assert entries(out) == []  # new_job leaves out/ empty
    fake_tools(INFO, walk(corner=True))
    set_frames(**TINY)
    run_done(job_id)
    assert entries(out) == ["reports", "reports/frames.json"]  # Req 8.8


def test_a_second_run_on_the_same_job_leaves_only_its_own_files(
        new_job: Callable[..., str], fake_tools: Callable[..., Any],
        set_frames: Callable[..., None]) -> None:
    job_id = new_job()
    # The first run makes 8 Picks and a Mask.
    fake_tools(INFO, walk(corner=True))
    set_frames(**TINY, base_picks={"fast": 8})
    run_done(job_id)
    assert entries(work_folder(job_id)) == layout(8, mask=True)
    # The second makes 5 Picks and no Mask, so the first run's last 3 Picks and its Mask are gone
    # once the Step_Runner has emptied work/frames/ (Req 11.11).
    fake_tools(INFO, walk())
    set_frames(**TINY)
    run_done(job_id)
    assert entries(work_folder(job_id)) == layout(5, mask=False)
    record = record_of(job_id)
    assert (frame_numbers(record), record["mask_used"]) == ([0, 4, 8, 12, 16], False)


# Too few Picks (Req 6.9, 6.10, 10.3, 13.8)

def test_too_few_picks_fail_the_step_before_the_pick_pass_and_leave_work_frames_empty(
        new_job: Callable[..., str], fake_tools: Callable[..., Any],
        set_frames: Callable[..., None]) -> None:
    job_id = new_job()
    tools = fake_tools(INFO, jolts())
    set_frames(**TINY, **TOO_FEW_PICKS)
    # The Readable_Error gives the 18 Picks the video gave and the 20 the step needs, and says
    # what to film differently (Req 6.10).
    report = run_failed(job_id, "18", "20",
                        "a longer video that moves slowly and steadily through the place, in "
                        "good light, without long still parts")
    # FakeTools recorded only the Score_Pass's grey plan, so no Pick_Pass ran, and work/frames/
    # is empty: no Colour_Pick, Grey_Copy, Mask or frames.json (Req 6.9, 13.8).
    assert [plan.colour for plan in tools.plans] == [False]
    assert list(work_folder(job_id).iterdir()) == []
    # The counts worked out before the step failed, picks included, and the timings of the parts
    # that finished, so no choose and no pick_pass (Req 10.3).
    assert report["counts"] == {"pick_width": PICK_WIDTH, "pick_height": PICK_HEIGHT,
                                "decoded_frames": 40, "still_start_frames": 0,
                                "still_end_frames": 0, "kept_frames": 40,
                                "target_picks": 20, "jolt_dropped_frames": 4, "picks": 18}
    assert sorted(report["timings"]) == ["probe", "score_pass"]


# Reading the video and writing the files (Req 1.8, 1.11, 8.7, 8.9, 8.10, 13.8)

def test_a_used_part_with_no_frames_fails_the_step_saying_none_could_be_read(
        new_job: Callable[..., str], fake_tools: Callable[..., Any],
        set_frames: Callable[..., None]) -> None:
    # The Trim starts at 5 s, after the walk's first 10 frames, from 0 s to 4.5 s, which are all
    # that FakeTools has.
    job_id = new_job(start=5.0)
    fake_tools(INFO, walk()[:10])
    set_frames(**TINY)
    run_failed(job_id, "No frames could be read from the chosen part of the video",
               "Check that the video plays")  # Req 1.11


def test_a_pick_pass_that_gives_back_fewer_picks_fails_the_step_and_leaves_no_frames_json(
        new_job: Callable[..., str], fake_tools: Callable[..., Any],
        set_frames: Callable[..., None]) -> None:
    # The walk's Picks are frames 0, 4, 8, 12 and 16, and the Pick_Pass gives only frames 0 to 9,
    # so ffmpeg gives back 3 of the 5.
    job_id = new_job()
    fake_tools(INFO, walk(), pick_pass_frames=10)
    set_frames(**TINY)
    run_failed(job_id, "ffmpeg gave back 3 of the 5 picks")  # Req 8.9
    assert not (work_folder(job_id) / "frames.json").exists()  # Req 8.7, 13.8


def full_disk(*args: object, **kwargs: object) -> NoReturn:
    """A stand-in for Path.write_bytes that fails the way a full disk does."""
    raise OSError(errno.ENOSPC, FULL_DISK)


def test_a_full_disk_fails_the_step_naming_the_file_it_couldnt_write_and_the_reason(
        new_job: Callable[..., str], fake_tools: Callable[..., Any],
        set_frames: Callable[..., None], monkeypatch: pytest.MonkeyPatch) -> None:
    job_id = new_job()
    fake_tools(INFO, walk())
    set_frames(**TINY)
    # Replaced once the Job is made. The Step_Runner writes its report with Path.write_text, so
    # the failed report is still written.
    monkeypatch.setattr(Path, "write_bytes", full_disk)
    # The first file the step writes is the first Colour_Pick, in work/frames/ of the Job_Folder
    # that open_job gives the Step_Runner.
    folder = open_job(job_id, load_config()).folder
    first_pick = folder / "work" / "frames" / "images" / "000000.jpg"
    run_failed(job_id, str(first_pick), FULL_DISK)  # Req 8.10


def missing_ffmpeg() -> ExpectedFailure:
    """The ExpectedFailure that the real decode raises when ffmpeg isn't installed, for FakeTools
    to raise in its place. While decode runs here, subprocess.Popen, as
    splatpipe/frames/ffmpeg.py sees it, raises the FileNotFoundError that a missing program
    gives."""
    def no_ffmpeg(*args: object, **kwargs: object) -> NoReturn:
        raise FileNotFoundError(errno.ENOENT, "No such file or directory", "ffmpeg")

    plan = DecodePlan(video=Path("clip.mp4"), stream_index=0, start=0.0, end=1.0, width=1,
                      height=1, colour=False)
    with pytest.MonkeyPatch.context() as patch, pytest.raises(ExpectedFailure) as caught:
        patch.setattr("splatpipe.frames.ffmpeg.subprocess.Popen", no_ffmpeg)
        decode(plan, 60.0, lambda time, frame: None)
    return caught.value


def test_a_missing_ffmpeg_fails_the_step_naming_it_and_saying_to_install_ffmpeg(
        new_job: Callable[..., str], fake_tools: Callable[..., Any],
        set_frames: Callable[..., None]) -> None:
    job_id = new_job()
    fake_tools(INFO, walk(), error=missing_ffmpeg())
    set_frames(**TINY)
    # The Readable_Error names the missing tool and says to install ffmpeg (Req 1.8).
    run_failed(job_id, "ffmpeg isn't installed", "Install ffmpeg, which includes ffprobe")


# Settings (Req 9.6, 9.7, 10.3)

# A bad setting, whose Readable_Error names the setting, its table and what it must be, and a
# failed cross check, whose Readable_Error names both settings, their tables and their values.
# The test config's base_picks for fast is 2 (Req 9.6, 9.7).
BAD_SETTINGS = [
    pytest.param({"jpeg_quality": 0},
                 ("jpeg_quality in [frames] is 0", "must be a whole number from 1 to 100"),
                 id="a bad setting"),
    pytest.param({"min_picks": 3},
                 ("min_picks in [frames] is 3", "base_picks in [frames.fast], which is 2"),
                 id="a failed cross check"),
]


@pytest.mark.parametrize(("values", "phrases"), BAD_SETTINGS)
def test_a_bad_setting_or_a_failed_cross_check_fails_the_step_before_any_tool_runs(
        new_job: Callable[..., str], fake_tools: Callable[..., Any],
        set_frames: Callable[..., None], values: dict[str, Any],
        phrases: tuple[str, ...]) -> None:
    job_id = new_job()
    tools = fake_tools(INFO, walk())
    set_frames(**values)
    report = run_failed(job_id, *phrases)
    # FakeTools recorded no call, so neither ffprobe nor ffmpeg ran, and the step worked out no
    # count and no timing (Req 9.6, 9.7, 10.3).
    assert (tools.probes, tools.plans) == ([], [])
    assert (report["counts"], report["timings"]) == ({}, {})


# The CLI (Req 11.1, 11.2)

def test_the_cli_prints_frames_done_and_gives_0_when_the_step_is_done(
        new_job: Callable[..., str], fake_tools: Callable[..., Any],
        set_frames: Callable[..., None], capsys: pytest.CaptureFixture[str]) -> None:
    job_id = new_job()
    fake_tools(INFO, walk())
    set_frames(**TINY)
    assert frames_main(["--job", job_id]) == 0
    captured = capsys.readouterr()
    assert (captured.out, captured.err) == ("frames: done\n", "")


def test_the_cli_prints_the_readable_error_on_stderr_and_gives_1_when_the_step_fails(
        new_job: Callable[..., str], fake_tools: Callable[..., Any],
        set_frames: Callable[..., None], capsys: pytest.CaptureFixture[str]) -> None:
    job_id = new_job()
    fake_tools(INFO, jolts())
    set_frames(**TINY, **TOO_FEW_PICKS)
    assert frames_main(["--job", job_id]) == 1
    captured = capsys.readouterr()
    report = read_json(folder_of(job_id) / "out" / "reports" / "frames.json")
    assert (captured.out, captured.err) == ("", report["error"]["readable"] + "\n")
    assert "Traceback" not in captured.err


# What the Score_Pass holds at once (Req 1.12, 1.13, 13.7)
#
# The Score_Pass gives each Score_Frame to the Scorer and then to the SampleBuffer, and keeps
# nothing itself. The Scorer keeps only its latest frame and the buffer at most 2 × mask_samples,
# and the helper that runs the Score_Pass and the choosing gives back neither of them. So however
# long the stream, the step holds at most the frame before, the current frame and the buffer's
# frames, and none of them once the Pick_Pass starts.

# The long stream's length and frame rate, and the mask_samples it runs with.
LONG_STREAM_FRAMES = 5_000
LONG_STREAM_FPS = 500
LONG_STREAM_MASK_SAMPLES = 3
# The most Score_Frames that may be alive after each frame: the frame before, the current frame
# and the SampleBuffer's 2 × mask_samples, 8 in all (Req 1.12).
MOST_ALIVE = 2 + 2 * LONG_STREAM_MASK_SAMPLES

# A decode as the step's tools hold it: decode(plan, timeout_seconds, on_frame).
Decode = Callable[[DecodePlan, float, Callable[[float, np.ndarray], None]], None]


def long_stream_pair(k: int) -> Pair:
    """Pair k of the long stream, which FakeTools makes only when its decode reaches it: at
    k / 500 s, a sharp checkerboard like the jolts', whose base brightness is 100 and 140 by
    turns."""
    return k / LONG_STREAM_FPS, checkerboard(140 if k % 2 else 100, 30)


class ScorePassWatch:
    """A decode for the step that runs the stand-in's own decode and watches the frames the
    Score_Pass gives the step, through weakrefs alone.

    In the Score_Pass, the grey plan, it takes a weakref to each frame the step's on_frame gets.
    Once on_frame has given the frame to the Scorer and the SampleBuffer, it records how many of
    the Score_Pass's frames are still alive. When the Pick_Pass, the colour plan, starts, it
    records how many are alive before the stand-in gives it a frame. It keeps no frame itself.
    """

    def __init__(self, decode: Decode) -> None:
        self._decode = decode
        self._refs: list[weakref.ref[np.ndarray]] = []  # the frames not yet seen dead
        self.alive_after_each_frame: list[int] = []
        self.alive_when_the_pick_pass_starts: list[int] = []

    def decode(self, plan: DecodePlan, timeout_seconds: float,
               on_frame: Callable[[float, np.ndarray], None]) -> None:
        """Run the stand-in's decode on `plan`, recording what the class says."""
        if plan.colour:
            self.alive_when_the_pick_pass_starts.append(self._alive())
            self._decode(plan, timeout_seconds, on_frame)
            return

        def watched(time: float, frame: np.ndarray) -> None:
            self._refs.append(weakref.ref(frame))
            on_frame(time, frame)
            # Counted while this call still holds `frame`, the current frame.
            self.alive_after_each_frame.append(self._alive())

        self._decode(plan, timeout_seconds, watched)

    def _alive(self) -> int:
        """How many of the Score_Pass's frames are still alive. A dead frame never comes back, so
        its weakref is dropped, and each count looks at a handful of frames."""
        self._refs = [ref for ref in self._refs if ref() is not None]
        return len(self._refs)


def test_a_5000_frame_score_pass_leaves_at_most_8_frames_alive_and_none_at_the_pick_pass(
        new_job: Callable[..., str], fake_tools: Any, set_frames: Callable[..., None],
        monkeypatch: pytest.MonkeyPatch) -> None:
    job_id = new_job()
    stand_in = fake_tools.stream(INFO, LONG_STREAM_FRAMES, long_stream_pair)
    # The stand-in's tools, with the watch around its decode.
    watch = ScorePassWatch(stand_in.decode)
    monkeypatch.setattr(frames, "TOOLS", replace(frames.TOOLS, decode=watch.decode))
    set_frames(**TINY, mask_samples=LONG_STREAM_MASK_SAMPLES)
    run_done(job_id)
    # A count after each of the 5,000 frames, none above the frame before, the current frame and
    # the buffer's 6 (Req 1.12, 13.7).
    assert len(watch.alive_after_each_frame) == LONG_STREAM_FRAMES
    assert max(watch.alive_after_each_frame) <= MOST_ALIVE
    # No Score_Frame is alive once the Pick_Pass starts, because the SampleBuffer is freed
    # before it (Req 1.12).
    assert watch.alive_when_the_pick_pass_starts == [0]


# The real Step_List (splat-training spec, Req 1.2, 11.5, 11.11)

def test_run_job_with_the_real_step_list_fails_at_train_once_the_frames_step_is_done(
        new_job: Callable[..., str], fake_tools: Callable[..., Any],
        set_frames: Callable[..., None]) -> None:
    job_id = new_job()
    fake_tools(INFO, walk())
    set_frames(**TINY)
    summary = run_job(job_id)  # with no steps given, the real Step_List runs
    assert [(step["step"], step["outcome"]) for step in summary["steps"]] == [
        ("frames", "done"), ("train", "failed")]
    assert (summary["status"], summary["failed_step"]) == ("failed", "train")
    # No camera solve has run, so the Train_Step finds no work/cameras/images/.
    images = folder_of(job_id).resolve() / "work" / "cameras" / "images"
    assert summary["readable_error"] == (
        f"The train step needs {images}, which the camera solve step writes, but it isn't "
        "there. Run the camera solve step on this job first.")


# The real ffmpeg (Req 8.1, 8.2, 8.4, 8.5, 11.4, 13.4, 13.5, 13.11)

# Skips a test that runs the real ffmpeg and ffprobe when either isn't installed, the same way as
# the real tests in tests/test_frames_ffmpeg.py (Req 13.5).
needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg and ffprobe aren't installed")
# The limit of the ffmpeg run that makes a real clip (Req 13.11). It takes well under a second.
REAL_TIMEOUT = 60
# The real clip's size, smaller than the test config's long_edge of 1600. The edges differ, so a
# width and height that swapped places would show.
CLIP_WIDTH, CLIP_HEIGHT = 64, 48


def make_testsrc_clip(folder: Path) -> Path:
    """Make folder / "testsrc.mp4", a 3 s clip of lavfi's testsrc at 10 frames a second and
    CLIP_WIDTH × CLIP_HEIGHT, with the real ffmpeg, and give its path. It isn't clip.mp4, which
    new_job writes. The codec is mpeg4 because every ffmpeg build has it."""
    path = folder / "testsrc.mp4"
    source = f"testsrc=duration=3:size={CLIP_WIDTH}x{CLIP_HEIGHT}:rate=10"
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i", source,
                    "-c:v", "mpeg4", "-pix_fmt", "yuv420p", "file:" + str(path.resolve())],
                   check=True, timeout=REAL_TIMEOUT)
    return path


def real_job(new_job: Callable[..., str], clip: Path) -> str:
    """Make a Job with new_job, copy `clip` over its video, input/video.mp4, and give its Job_Id.

    make_job would refuse a real clip, which is bigger than the test config's max_video_bytes of
    1024, so new_job makes the Job from its few bytes first. The step reads only the Job's video
    and the Job_Record's Trim, from 0 to 10 s, the length new_job's stand-in probe gave. So the
    Used_Part runs past the clip's end, and ffmpeg reads the clip to its end.
    """
    job_id = new_job()
    shutil.copyfile(clip, open_job(job_id, load_config()).video_path)
    return job_id


@needs_ffmpeg
def test_with_the_real_ffmpeg_a_64_by_48_clip_gives_64_by_48_picks_in_the_designs_layout(
        new_job: Callable[..., str], tmp_path: Path) -> None:
    job_id = real_job(new_job, make_testsrc_clip(tmp_path))
    run_done(job_id)  # with the real TOOLS, so ffprobe and ffmpeg read the clip
    record = record_of(job_id)
    picks = record["picks"]
    assert picks  # the step is done, so it made at least min_picks, 1
    work = work_folder(job_id)
    # images/ and grey/, each with the Picks' names from 000000.jpg, and frames.json, with
    # mask.png only when the step uses the Mask (Req 8.4, 8.5).
    assert entries(work) == layout(len(picks), mask=record["mask_used"])
    # The clip is smaller than long_edge, so it isn't scaled up: each Colour_Pick and its
    # Grey_Copy has the clip's own size (Req 8.1).
    assert (record["pick_width"], record["pick_height"]) == (CLIP_WIDTH, CLIP_HEIGHT)
    for entry in picks:
        assert read_image(work / "images" / entry["file"]).shape == (CLIP_HEIGHT, CLIP_WIDTH, 3)
        assert read_image(work / "grey" / entry["file"]).shape == (CLIP_HEIGHT, CLIP_WIDTH)


# The ramp clip: 2 s at 10 frames a second, so 20 frames 0.1 s apart from 0 s. Frame k is flat
# grey at a brightness of 16 + 10k, from 16 to 206.
RAMP_SECONDS, RAMP_RATE = 2, 10
RAMP_FIRST, RAMP_STEP = 16, 10
# How close each Colour_Pick's mean grey must be to its frame's brightness. Neighbouring frames
# are 10 apart, so a mean within 4 of one frame's brightness is at least 6 from every other
# frame's, and the Colour_Pick can only have come from that frame.
BRIGHTNESS_TOLERANCE = 4


def ramp_brightness(k: int) -> int:
    """The brightness of the ramp clip's frame k, from 0 to 255."""
    return RAMP_FIRST + RAMP_STEP * k


def make_ramp_clip(folder: Path) -> Path:
    """Make folder / "ramp.mp4", the ramp clip at CLIP_WIDTH × CLIP_HEIGHT, with the real ffmpeg,
    and give its path.

    lavfi's color source gives the frames, and geq sets each one's grey from N, its number from
    0. The codec is mpeg4 because every ffmpeg build has it.
    """
    path = folder / "ramp.mp4"
    source = (f"color=c=black:size={CLIP_WIDTH}x{CLIP_HEIGHT}:rate={RAMP_RATE}"
              f":duration={RAMP_SECONDS}")
    grey = f"format=gray,geq=lum={RAMP_FIRST}+{RAMP_STEP}*N"
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i", source,
                    "-vf", grey, "-c:v", "mpeg4", "-pix_fmt", "yuv420p",
                    "file:" + str(path.resolve())],
                   check=True, timeout=REAL_TIMEOUT)
    return path


@needs_ffmpeg
def test_run_job_with_the_real_ffmpeg_is_done_and_each_colour_pick_shows_the_frame_its_entry_names(
        new_job: Callable[..., str], tmp_path: Path) -> None:
    job_id = real_job(new_job, make_ramp_clip(tmp_path))
    # The Job_Runner with a Step_List of the Frames_Step's entry alone, and the real TOOLS, so
    # ffprobe and ffmpeg read the ramp. The Job is done once the step is (Req 11.4, and
    # splat-training spec, Req 11.10).
    summary = run_job(job_id, steps=(StepEntry("frames", "splatpipe.frames", True),))
    assert summary["status"] == "done", summary["readable_error"]
    record = record_of(job_id)
    picks = record["picks"]
    assert picks  # the step is done, so it made at least min_picks, 1
    work = work_folder(job_id)
    # images/ and grey/, each with the Picks' names from 000000.jpg, and frames.json, with
    # mask.png only when the step uses the Mask (Req 8.4, 8.5).
    assert entries(work) == layout(len(picks), mask=record["mask_used"])
    for entry in picks:
        # Each Colour_Pick comes from the frame its entry names, Frame_Number k being the ramp's
        # frame k, so its mean grey is that frame's brightness (Req 8.2, 13.4).
        colour = read_image(work / "images" / entry["file"])
        mean_grey = float(cv2.cvtColor(colour, cv2.COLOR_BGR2GRAY).mean())
        brightness = ramp_brightness(entry["frame_number"])
        assert abs(mean_grey - brightness) <= BRIGHTNESS_TOLERANCE, (entry, mean_grey)
