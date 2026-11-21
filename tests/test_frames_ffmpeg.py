"""Tests for splatpipe/frames/ffmpeg.py: ffprobe, the filter check, the Tone_Map and the ffmpeg
decode.

The unit tests of the pure parts run no tool, so they pass on a machine without ffmpeg (Req 13.6).

- parse_stream gets ffprobe's -show_streams JSON as json.loads gives it, with the fields of a
  phone clip.
- parse_frame_time gets showinfo's lines as ffmpeg prints them on stderr, without the newline.
  The lines marked 9.0.2 are copied from the output of ffmpeg 9.0.2, the version on our laptops.
  The others follow the format strings in libavfilter/vf_showinfo.c of the release they're
  marked with. 6.1 is Ubuntu 24.04's packaged ffmpeg.
- decode_command gets DecodePlans, with and without a ToneMap, and the tests check the argument
  list it gives.
- tone_map_filter gets ToneMaps, and the tests check its text against the design's, which they
  write out in full.
- parse_filters gets the start of what ffmpeg -filters prints, in both of its layouts: 6.1's,
  with three flag columns, and 9.0.2's, with two. Every 9.0.2 line but zscale's is copied from
  the output of the laptops' ffmpeg 9.0.2, which has no zscale. zscale's line is the one 9.0.2
  prints when it's built with libzimg. That line and the 6.1 lines follow the format string of
  show_filters in fftools/opt_common.c and each filter's flags in libavfilter, in the release
  they're marked with.

The unit tests with fake processes run no tool either. They replace subprocess.run and
subprocess.Popen as splatpipe.frames.ffmpeg sees them, with monkeypatch.

- probe_stream gets what a stand-in subprocess.run gives back or raises: a CompletedProcess with
  ffprobe's output, FileNotFoundError or TimeoutExpired.
- has_filters gets the same kinds of stand-in, with the CompletedProcess holding one of the
  -filters listings above.
- decode gets a FakeFfmpeg, whose stdout gives raw frames and whose stderr gives showinfo's lines
  in the format of ffmpeg 9.0.2, or a HungFfmpeg, whose stdout is the read end of an os.pipe()
  that its kill() closes. Every wait in these tests has a short bound, so a broken decode fails
  its test rather than hanging the run.

Property 3 runs no tool either. It gives a new FrameGate a drawn Used_Part and 1 to 60 reported
times, in the order drawn. The Used_Part lies inside a video of at most 10 minutes, so from 0 to
600 s. Each time is None, as for a frame logged with NOPTS, any float from -10 to 700 s, which
reaches past both edges of the Used_Part, the start or the end exactly, or a repeat of a time
drawn before it. Unlike the other property tests' times, they aren't running sums of gaps,
because sorting out times that don't increase is the gate's job. The oracle decides each time
with its own loop: it takes a time that isn't None, lies in the Used_Part and is greater than
every time it took before.

Property 4 runs no tool either. It gives seek_text any float from 0 to 1e6 s, once rounded down,
as for the start, and once rounded up, as for the end. The oracle checks each text's form with a
regular expression of its own. It reads the text and the float back with fractions.Fraction,
which holds both exactly, so its comparisons add no rounding.

The real ffmpeg tests run the real ffmpeg and ffprobe, so they're skipped, with the reason
"ffmpeg and ffprobe aren't installed", when either tool is missing (Req 13.5). Each test makes
its clip in tmp_path with ffmpeg, from lavfi's testsrc at 10 frames a second and 32 × 24 pixels.
Every tool they run gets a limit of 60 s, the runs that make the clips included, except the
filter check of the real Tone_Map test (Req 13.11). They check each Frame_Time to within 1 ms,
because ffmpeg 5.1 and 6.1 write pts_time with six significant digits.

The real Tone_Map test is also skipped, with the reason "this ffmpeg has no zscale or tonemap
filter", when has_filters finds that this ffmpeg lacks either one, as the laptops' 9.0.2 lacks
zscale (Req 13.5). The test runs that check itself, so collecting the tests never runs ffmpeg,
with a limit of 30 s, the ffprobe_timeout_seconds in config.toml that the step gives it. Its
clip's frames are tagged as a phone's HDR clip is: HLG, arib-std-b67, with BT.2020 primaries and
matrix.
"""

import io
import os
import re
import shutil
import subprocess
import threading
from collections.abc import Callable
from dataclasses import dataclass, replace
from fractions import Fraction
from pathlib import Path
from typing import IO, Any, NoReturn

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from splatpipe.errors import ExpectedFailure
from splatpipe.frames.ffmpeg import (NOT_A_FRAME, DecodeFailed, DecodePlan, FrameGate, ToneMap,
                                     VideoInfo, decode, decode_command, has_filters,
                                     parse_filters, parse_frame_time, parse_stream, probe_stream,
                                     seek_text, tone_map_filter)


# parse_stream: the video stream, its Displayed_Shape and its colour transfer (Req 2.1)

def stream(**fields: object) -> dict[str, Any]:
    """A 1920 × 1080 H.264 video stream at index 0 with square pixels, as ffprobe's -show_streams
    JSON gives it, with each of `fields` added or replaced, or left out when it's None. Like many
    clips, it has no color_transfer and no rotation."""
    base: dict[str, Any] = {
        "index": 0, "codec_name": "h264", "codec_type": "video", "width": 1920, "height": 1080,
        "sample_aspect_ratio": "1:1", "pix_fmt": "yuv420p",
        "disposition": {"default": 1, "attached_pic": 0},
        "tags": {"language": "und", "handler_name": "VideoHandler"}}
    return {key: value for key, value in {**base, **fields}.items() if value is not None}


def probed(*streams: dict[str, Any]) -> dict[str, Any]:
    """ffprobe's -show_streams JSON for a file that holds `streams`, in order."""
    return {"streams": list(streams)}


def display_matrix(rotation: int) -> dict[str, Any]:
    """The Display Matrix entry of side_data_list for a picture that players turn by `rotation`
    degrees. ffprobe also prints the matrix itself, which parse_stream doesn't read."""
    return {"side_data_type": "Display Matrix", "rotation": rotation}


def test_parse_stream_skips_cover_art_and_takes_the_video_stream_after_it() -> None:
    cover_art = stream(index=0, codec_name="mjpeg", width=600, height=600,
                       disposition={"default": 0, "attached_pic": 1})
    audio = {"index": 1, "codec_name": "aac", "codec_type": "audio",
             "disposition": {"default": 1, "attached_pic": 0}}
    info = parse_stream(probed(cover_art, audio, stream(index=2)))
    assert info == VideoInfo(stream_index=2, width=1920, height=1080, colour_transfer=None)


# A phone filming upright stores a 1920 × 1080 picture that players turn a quarter turn. ffprobe
# gives that turn as a rotation of -90 in side_data_list, and the rotate tag gives it as "90". A
# half turn is -180 in side_data_list and "180" in the tag.
@pytest.mark.parametrize(("fields", "shape"), [
    pytest.param({"side_data_list": [display_matrix(-90)]}, (1080, 1920), id="side data -90"),
    pytest.param({"tags": {"rotate": "90"}}, (1080, 1920), id="rotate tag 90"),
    pytest.param({"side_data_list": [display_matrix(-180)]}, (1920, 1080), id="side data -180"),
    pytest.param({"tags": {"rotate": "180"}}, (1920, 1080), id="rotate tag 180"),
])
def test_a_quarter_turn_swaps_the_edges_and_a_half_turn_doesnt(
        fields: dict[str, Any], shape: tuple[int, int]) -> None:
    info = parse_stream(probed(stream(**fields)))
    assert (info.width, info.height) == shape


# A 1440 × 1080 picture whose pixels are 4/3 as wide as they're high, as in HDV, shows as
# 1920 × 1080. A ratio of 0:1 means the ratio is unknown, so the pixels count as square.
@pytest.mark.parametrize(("ratio", "width"), [
    pytest.param("4:3", 1920, id="4:3"),
    pytest.param("0:1", 1440, id="0:1"),
])
def test_a_sample_aspect_ratio_of_4_3_widens_the_picture_and_0_1_doesnt(
        ratio: str, width: int) -> None:
    info = parse_stream(probed(stream(width=1440, sample_aspect_ratio=ratio)))
    assert (info.width, info.height) == (width, 1080)


# A phone's HDR clip: HEVC with 10-bit colour in BT.2020 and the HLG transfer, arib-std-b67.
HLG = {"codec_name": "hevc", "pix_fmt": "yuv420p10le", "color_range": "tv",
       "color_space": "bt2020nc", "color_transfer": "arib-std-b67", "color_primaries": "bt2020"}


@pytest.mark.parametrize(("fields", "transfer"), [
    pytest.param({}, None, id="no color_transfer"),
    pytest.param(HLG, "arib-std-b67", id="HLG"),
])
def test_the_colour_transfer_is_none_when_missing_and_otherwise_comes_through_as_it_is(
        fields: dict[str, Any], transfer: str | None) -> None:
    assert parse_stream(probed(stream(**fields))).colour_transfer == transfer


@pytest.mark.parametrize("size", [
    pytest.param({"width": 0, "height": 0}, id="0 x 0"),
    pytest.param({"width": None, "height": None}, id="no width or height"),
])
def test_a_video_stream_without_a_picture_size_is_refused(size: dict[str, Any]) -> None:
    with pytest.raises(ExpectedFailure) as caught:
        parse_stream(probed(stream(**size)))
    message = str(caught.value)
    assert "ffprobe found no picture size" in message
    assert "plays" in message


# parse_frame_time: each frame's time from showinfo (Req 1.6, D5)

# showinfo's line for one frame in each ffmpeg, and the pts_time in it. 5.1 and 6.1 write the
# time with six significant digits, and 7.0 and newer with up to six decimals. 5.1 logs pos:
# after it, and 6.0 and newer duration:. In the 7.1 line, the frame number and the pts fill their
# fields, so no space follows n: or pts:.
FRAME_LINES = [
    pytest.param("[Parsed_showinfo_3 @ 0x55d0c1f4e2c0] n: 250 pts: 750750 pts_time:8.34167 "
                 "pos:  4210785 fmt:gray sar:1/1 s:320x180 i:P iskey:0 type:P ",
                 8.34167, id="5.1"),
    pytest.param("[Parsed_showinfo_3 @ 0x5629a3c87f40] n:  37 pts:    740 pts_time:1.23333 "
                 "duration:     20 duration_time:0.0333333 fmt:bgr24 cl:unspecified "
                 "sar:1/1 s:1600x900 i:P iskey:0 type:B ",
                 1.23333, id="6.1"),
    pytest.param("[Parsed_showinfo_3 @ 0x55b3e2a1d900] n:1250 pts:3753750 pts_time:41.708333 "
                 "duration:   3003 duration_time:0.0333667 fmt:gray cl:unspecified "
                 "sar:1/1 s:320x180 i:P iskey:0 type:P ",
                 41.708333, id="7.1"),
    pytest.param("[Parsed_showinfo_3 @ 0x798701ce40] n:   1 pts:      1 pts_time:0.1     "
                 "duration:      1 duration_time:0.1     fmt:gray cl:unspecified "
                 "sar:1/1 s:16x12 i:P iskey:1 type:I ",
                 0.1, id="9.0.2"),
]

# A frame line with no time, in 9.0.2's format: showinfo logs NOPTS for the pts and the pts_time.
NOPTS_LINE = ("[Parsed_showinfo_3 @ 0x798701ce40] n:   4 pts:  NOPTS pts_time:NOPTS   "
              "duration:      1 duration_time:0.1     fmt:gray cl:unspecified "
              "sar:1/1 s:16x12 i:P iskey:0 type:P ")

# showinfo's lines that aren't a frame's: the config lines it logs when the filters are set up,
# the line on the colours it logs after each frame, and a side data line from 5.1 and 9.0.2.
OTHER_LINES = [
    pytest.param("[Parsed_showinfo_3 @ 0x798701ce40] config in time_base: 1/10, frame_rate: 10/1",
                 id="9.0.2 config in"),
    pytest.param("[Parsed_showinfo_3 @ 0x798701ce40] config out time_base: 0/0, frame_rate: 0/0",
                 id="9.0.2 config out"),
    pytest.param("[Parsed_showinfo_3 @ 0x798701ce40] color_range:pc color_space:unknown "
                 "color_primaries:unknown color_trc:unknown", id="9.0.2 colours"),
    pytest.param("[Parsed_showinfo_3 @ 0x55d0c1f4e2c0]   side data - displaymatrix: rotation of "
                 "-90.00 degrees", id="5.1 side data"),
    pytest.param("[Parsed_showinfo_0 @ 0x7a04c19980]   side data - 3x3 displaymatrix: rotation "
                 "of 90.00 degrees", id="9.0.2 side data"),
]


@pytest.mark.parametrize(("line", "time"), FRAME_LINES)
def test_parse_frame_time_reads_the_pts_time_of_a_frame_line_from_each_ffmpeg(
        line: str, time: float) -> None:
    assert parse_frame_time(line) == time


def test_parse_frame_time_gives_none_for_a_frame_line_with_no_time() -> None:
    assert parse_frame_time(NOPTS_LINE) is None


@pytest.mark.parametrize("line", OTHER_LINES)
def test_parse_frame_time_gives_not_a_frame_for_the_lines_that_arent_a_frames(line: str) -> None:
    assert parse_frame_time(line) is NOT_A_FRAME


# decode_command: the ffmpeg command of a Decode_Pass (Req 1.3, 1.5, 1.7)

def decode_plan(video: Path, *, colour: bool, width: int, height: int,
                tone_map: ToneMap | None = None) -> DecodePlan:
    """The Decode_Pass of the video's stream 2 for the Used_Part from 1.25 s to 7.5 s, giving
    frames of width × height in colour or in grey, with the Tone_Map `tone_map` or, by default,
    without one."""
    return DecodePlan(video=video, stream_index=2, start=1.25, end=7.5, width=width,
                      height=height, colour=colour, tone_map=tone_map)


def after(command: list[str], option: str) -> str:
    """The argument that follows `option` in `command`."""
    return command[command.index(option) + 1]


def test_decode_command_is_a_list_that_gives_the_video_as_file_and_its_absolute_path(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)  # so the relative path below has to be made absolute
    command = decode_command(decode_plan(Path("-clip.mp4"), colour=False, width=320, height=180))
    assert isinstance(command, list)
    assert all(isinstance(argument, str) for argument in command)
    assert command[:2] == ["ffmpeg", "-nostdin"]
    assert command[-1] == "pipe:1"
    assert after(command, "-i") == "file:" + str((tmp_path / "-clip.mp4").resolve())
    assert after(command, "-fps_mode") == "passthrough"
    assert after(command, "-map") == "0:2"


def test_a_grey_plan_and_a_colour_plan_differ_only_in_the_size_and_the_format(
        tmp_path: Path) -> None:
    video = tmp_path / "-clip.mp4"
    # The Score_Pass and the Pick_Pass of a 1920 × 1080 video, with a score_width of 320 and a
    # long_edge of 1600.
    grey = decode_command(decode_plan(video, colour=False, width=320, height=180))
    colour = decode_command(decode_plan(video, colour=True, width=1600, height=900))
    assert len(grey) == len(colour)
    changed = [(index, grey_argument, colour_argument)
               for index, (grey_argument, colour_argument) in enumerate(zip(grey, colour))
               if grey_argument != colour_argument]
    assert changed == [(grey.index("-vf") + 1,
                        "scale=320:180,setsar=1,format=gray,showinfo=checksum=0",
                        "scale=1600:900,setsar=1,format=bgr24,showinfo=checksum=0")]


# tone_map_filter and decode_command with the Tone_Map (Req 2.2, 2.3)

# The Tone_Map's filters with the numbers in config.toml, tonemap_npl = 100 and
# tonemap_desat = 0.0, exactly as the design gives them.
CONFIG_TONE_MAP = ("zscale=t=linear:npl=100.0,format=gbrpf32le,zscale=p=bt709,"
                   "tonemap=tonemap=hable:desat=0.0,zscale=t=bt709:m=bt709:r=tv,format=yuv420p")


def test_the_tone_map_filter_with_the_config_numbers_is_exactly_the_designs_text() -> None:
    assert tone_map_filter(ToneMap(100.0, 0.0)) == CONFIG_TONE_MAP


def test_other_tone_map_numbers_change_only_npl_and_desat_not_hable_or_bt709() -> None:
    # The design's text with its two numbers changed and every other character as it is.
    expected = (CONFIG_TONE_MAP.replace("npl=100.0", "npl=203.5")
                .replace("desat=0.0", "desat=0.25"))
    assert tone_map_filter(ToneMap(203.5, 0.25)) == expected


# The Score_Pass and the Pick_Pass of a 1920 × 1080 video with the Tone_Map, with a score_width
# of 320 and a long_edge of 1600, and the filters each one gives ffmpeg.
@pytest.mark.parametrize(("colour", "width", "height", "filters"), [
    pytest.param(False, 320, 180,
                 f"scale=320:180,setsar=1,{CONFIG_TONE_MAP},format=gray,showinfo=checksum=0",
                 id="grey"),
    pytest.param(True, 1600, 900,
                 f"scale=1600:900,setsar=1,{CONFIG_TONE_MAP},format=bgr24,showinfo=checksum=0",
                 id="colour"),
])
def test_a_plans_tone_map_comes_between_setsar_and_the_format_in_a_grey_and_a_colour_plan(
        tmp_path: Path, colour: bool, width: int, height: int, filters: str) -> None:
    plan = decode_plan(tmp_path / "-clip.mp4", colour=colour, width=width, height=height,
                       tone_map=ToneMap(100.0, 0.0))
    assert after(decode_command(plan), "-vf") == filters


@pytest.mark.parametrize("colour", [pytest.param(False, id="grey"),
                                    pytest.param(True, id="colour")])
def test_a_plan_without_a_tone_map_has_no_zscale_or_tonemap_in_its_command(colour: bool) -> None:
    # The video's path is fixed rather than in tmp_path, whose folder pytest names after the
    # test, so no folder's name can hold either word.
    plan = decode_plan(Path("/clips/-clip.mp4"), colour=colour, width=320, height=180)
    assert [argument for argument in decode_command(plan)
            if "zscale" in argument or "tonemap" in argument] == []


# parse_filters: the filters this ffmpeg has (Req 2.2, 2.4)

# The start of what ffmpeg -filters prints: a header with the legend of its flags and pad types,
# then a line for each filter, which gives the filter's flags, name, pads and description. Each
# listing holds 5 of the hundreds of filters, tonemap and zscale among them.
#
# 6.1, Ubuntu 24.04's packaged ffmpeg, prints three flag columns: timeline support, slice
# threading and command support.
FILTERS_6_1 = """\
Filters:
  T.. = Timeline support
  .S. = Slice threading
  ..C = Command support
  A = Audio input/output
  V = Video input/output
  N = Dynamic number and/or type of input/output
  | = Source or sink filter
 ... setsar            V->V       Set the pixel sample aspect ratio.
 ... showinfo          V->V       Show textual information for each video frame.
 .S. tonemap           V->V       Conversion to/from different dynamic ranges.
 .SC zscale            V->V       Apply resizing, colorspace and bit depth conversion.
 ... testsrc           |->V       Generate test pattern.
"""

# 9.0.2, on our laptops, prints two flag columns, though its legend still shows three, and ends
# its header with a line of dashes. The laptops' build has no libzimg, so it lists no zscale.
# The zscale line here is the one 9.0.2 prints when it's built with libzimg.
FILTERS_9_0_2 = """\
Filters:
  T.. = Timeline support
  .S. = Slice threading
  A = Audio input/output
  V = Video input/output
  N = Dynamic number and/or type of input/output
  | = Source or sink filter
  ------
 .. setsar            V->V       Set the pixel sample aspect ratio.
 .. showinfo          V->V       Show textual information for each video frame.
 .S tonemap           V->V       Conversion to/from different dynamic ranges.
 .S zscale            V->V       Apply resizing, colorspace and bit depth conversion.
 .. testsrc           |->V       Generate test pattern.
"""

# The filters each listing holds.
LISTED = frozenset({"setsar", "showinfo", "tonemap", "zscale", "testsrc"})


def without(listing: str, name: str) -> str:
    """`listing` with the line of the filter `name` taken out."""
    return "".join(line for line in listing.splitlines(keepends=True)
                   if f" {name} " not in line)


@pytest.mark.parametrize("listing", [pytest.param(FILTERS_6_1, id="6.1"),
                                     pytest.param(FILTERS_9_0_2, id="9.0.2")])
def test_parse_filters_finds_zscale_and_tonemap_and_skips_the_header_lines(listing: str) -> None:
    # Exactly the filters' names, so no word of the header, like "Filters:", "=" or "------",
    # is taken for a filter.
    assert parse_filters(listing) == LISTED


# probe_stream and decode with fake processes (Req 1.8, 1.9, 1.10, 1.12, 1.14, 13.6, 13.9)

# The limit the tests give a tool that ends by itself. It's far longer than any of them takes, so
# decode's Timer never fires in them.
TIMEOUT = 10.0
# The limit in the tests of a tool that runs past it.
LIMIT = 0.2
# The longest a HungFfmpeg waits for decode to kill it before it gives up, so a broken decode
# fails its test rather than hanging the run.
SAFETY_SECONDS = 5.0
# The exit code Popen gives a process that kill() stopped with SIGKILL.
KILLED = -9
# The frame size of the grey plan, so each grey frame on stdout is 6 bytes.
WIDTH, HEIGHT = 3, 2
# The log context that each of showinfo's lines starts with, as ffmpeg 9.0.2 logs it.
SHOWINFO = "[Parsed_showinfo_3 @ 0x798701ce40]"
# The two lines showinfo logs when ffmpeg sets the filters up, for a video with a time base of
# 1/1000 at 25 frames a second.
CONFIG_LINES = (f"{SHOWINFO} config in time_base: 1/1000, frame_rate: 25/1\n"
                f"{SHOWINFO} config out time_base: 0/0, frame_rate: 0/0\n").encode()
# What a call to subprocess.run or subprocess.Popen was given: the command and the keyword
# arguments.
Call = tuple[list[str], dict[str, Any]]


class FakeFfmpeg:
    """A stand-in for the ffmpeg process that decode starts with subprocess.Popen.

    Its stdout gives `output` and its stderr gives `log`, and then both end. It runs until decode
    waits for it, when it ends by itself with `exit_code`, or kills it, when it ends with KILLED.
    `events` records each kill() and wait(), in order.
    """

    def __init__(self, output: bytes = b"", log: bytes = b"", exit_code: int = 0) -> None:
        self.stdout: IO[bytes] = io.BytesIO(output)
        self.stderr: IO[bytes] = io.BytesIO(log)
        self.returncode: int | None = None  # None while it runs, as Popen has it
        self.events: list[str] = []
        self._exit_code = exit_code

    @property
    def killed(self) -> bool:
        """Whether decode killed it."""
        return "kill" in self.events

    def poll(self) -> int | None:
        """Its exit code, or None while it runs."""
        return self.returncode

    def wait(self) -> int:
        """Let it end by itself, unless it has ended already, and give its exit code."""
        self.events.append("wait")
        if self.returncode is None:
            self.returncode = self._exit_code
        return self.returncode

    def kill(self) -> None:
        """Stop it, unless it has ended already."""
        self.events.append("kill")
        if self.returncode is None:
            self.returncode = KILLED


class HungFfmpeg(FakeFfmpeg):
    """A stand-in ffmpeg that hangs: it writes nothing and never ends by itself.

    Its stdout is the read end of an os.pipe(), whose write end kill() closes, so a read of stdout
    waits until it's killed. So that a broken decode fails its test rather than hanging the run, a
    timer closes the write end after SAFETY_SECONDS, and wait() fails the test if it hasn't been
    killed by then.
    """

    def __init__(self) -> None:
        super().__init__()
        read_end, write_end = os.pipe()
        self.stdout = os.fdopen(read_end, "rb")
        self._write_end: int | None = write_end
        self._lock = threading.Lock()  # so the write end is closed once, by kill() or the timer
        self._killed = threading.Event()
        self._safety = threading.Timer(SAFETY_SECONDS, self._close_write_end)
        self._safety.daemon = True
        self._safety.start()

    def wait(self) -> int:
        """Wait until it's killed, for at most SAFETY_SECONDS, and give its exit code."""
        if not self._killed.wait(SAFETY_SECONDS):
            raise AssertionError("decode waited for the hung ffmpeg without killing it")
        return super().wait()

    def kill(self) -> None:
        """Stop it, which closes the write end of its stdout, so a read of stdout ends."""
        super().kill()
        self._killed.set()
        self._safety.cancel()
        self._close_write_end()

    def _close_write_end(self) -> None:
        """Close the write end of stdout, unless it's closed already."""
        with self._lock:
            if self._write_end is not None:
                os.close(self._write_end)
                self._write_end = None


def replace_subprocess(monkeypatch: pytest.MonkeyPatch, function: str,
                       outcome: object) -> list[Call]:
    """Replace subprocess.run or subprocess.Popen, as `function` names it and as
    splatpipe.frames.ffmpeg sees it, with a stand-in that records each call, then raises
    `outcome` if it's an exception or else gives it back. Gives the list of calls."""
    calls: list[Call] = []

    def stand_in(command: list[str], **kwargs: Any) -> object:
        calls.append((command, kwargs))
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    monkeypatch.setattr(f"splatpipe.frames.ffmpeg.subprocess.{function}", stand_in)
    return calls


def finished(returncode: int = 0, stdout: bytes = b"", stderr: bytes = b"",
             tool: str = "ffprobe") -> subprocess.CompletedProcess[bytes]:
    """What subprocess.run gives back when `tool` ran, with its output captured as bytes. The
    tool is ffprobe unless another is named."""
    return subprocess.CompletedProcess([tool], returncode, stdout=stdout, stderr=stderr)


def frame_lines(plan: DecodePlan, n: int, pts_time: str) -> bytes:
    """What showinfo logs on stderr for the plan's frame n at `pts_time`, in the format of
    ffmpeg 9.0.2: the frame's line and the colour line that follows it. A pts_time of NOPTS is a
    frame with no time."""
    pts = pts_time if pts_time == "NOPTS" else str(round(float(pts_time) * 1000))
    pixels = "bgr24" if plan.colour else "gray"
    return (f"{SHOWINFO} n:{n:4d} pts:{pts:>7} pts_time:{pts_time:<7} duration:     40 "
            f"duration_time:0.04    fmt:{pixels} cl:unspecified sar:1/1 "
            f"s:{plan.width}x{plan.height} i:P iskey:0 type:P \n"
            f"{SHOWINFO} color_range:pc color_space:unknown color_primaries:unknown "
            "color_trc:unknown\n").encode()


def showinfo_log(plan: DecodePlan, *pts_times: str) -> bytes:
    """What showinfo logs on stderr for the plan's frames at `pts_times`, in order, after its
    config lines."""
    return CONFIG_LINES + b"".join(frame_lines(plan, n, pts_time)
                                   for n, pts_time in enumerate(pts_times))


def grey_plan(tmp_path: Path) -> DecodePlan:
    """The grey Decode_Pass of a video in tmp_path, giving frames of WIDTH × HEIGHT for the
    Used_Part from 1.25 s to 7.5 s. So a frame's Frame_Time is 1.25 s plus its pts_time."""
    return decode_plan(tmp_path / "-clip.mp4", colour=False, width=WIDTH, height=HEIGHT)


def grey_frame(k: int) -> bytes:
    """Grey frame k as ffmpeg writes it on stdout: WIDTH × HEIGHT bytes, each of them k."""
    return bytes([k]) * (WIDTH * HEIGHT)


def grey_frames(count: int) -> bytes:
    """Grey frames 0 to count − 1 as ffmpeg writes them on stdout, one after the other."""
    return b"".join(grey_frame(k) for k in range(count))


def keep_nothing(time: float, frame: np.ndarray) -> None:
    """An on_frame that keeps nothing."""


def probe_failure(tmp_path: Path, timeout: float = TIMEOUT) -> ExpectedFailure:
    """Run probe_stream on a video in tmp_path with the limit `timeout`, and give back the
    ExpectedFailure it raises."""
    with pytest.raises(ExpectedFailure) as caught:
        probe_stream(tmp_path / "-clip.mp4", timeout)
    return caught.value


def decode_failure(tmp_path: Path, timeout: float = TIMEOUT,
                   on_frame: Callable[[float, np.ndarray], None] = keep_nothing
                   ) -> ExpectedFailure:
    """Run decode on grey_plan(tmp_path) with the limit `timeout`, and give back the
    ExpectedFailure it raises."""
    with pytest.raises(ExpectedFailure) as caught:
        decode(grey_plan(tmp_path), timeout, on_frame)
    return caught.value


def readable(error: ExpectedFailure) -> str:
    """The Readable_Error of `error`, checked to hold no traceback (Req 13.9)."""
    message = str(error)
    assert "Traceback" not in message
    return message


@pytest.mark.parametrize(("function", "failure", "tool"), [
    pytest.param("run", probe_failure, "ffprobe", id="ffprobe"),
    pytest.param("Popen", decode_failure, "ffmpeg", id="ffmpeg"),
])
def test_a_missing_tool_is_named_and_the_message_says_to_install_ffmpeg_which_has_ffprobe(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, function: str,
        failure: Callable[[Path], ExpectedFailure], tool: str) -> None:
    replace_subprocess(monkeypatch, function,
                       FileNotFoundError(2, "No such file or directory", tool))
    message = readable(failure(tmp_path))
    assert f"{tool} isn't installed" in message
    assert "Install ffmpeg, which includes ffprobe" in message


def test_an_exit_of_1_from_ffprobe_gives_its_last_stderr_line_even_with_bytes_that_arent_utf8(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # b"\xe9" is é in Windows-1252 but isn't UTF-8, so it's read as U+FFFD.
    stderr = (b"[mov,mp4,m4a,3gp,3g2,mj2 @ 0x7f8e4c004a00] moov atom not found\n"
              b"file:/clips/caf\xe9.mp4: Invalid data found when processing input\n")
    replace_subprocess(monkeypatch, "run", finished(returncode=1, stderr=stderr))
    message = readable(probe_failure(tmp_path))
    assert "(file:/clips/caf\ufffd.mp4: Invalid data found when processing input)" in message
    assert "moov atom not found" not in message


def test_an_exit_of_1_from_ffmpeg_is_decode_failed_with_its_last_stderr_line_not_showinfos(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    plan = grey_plan(tmp_path)
    # ffmpeg logs its errors between showinfo's lines. The last error holds b"\xe9", é in
    # Windows-1252, which isn't UTF-8, so it's read as U+FFFD.
    log = (CONFIG_LINES + frame_lines(plan, 0, "0")
           + b"[h264 @ 0x7f8e4c005000] Invalid NAL unit size (1092 > 87).\n"
           + b"file:/clips/caf\xe9.mp4: Input/output error\n"
           + frame_lines(plan, 1, "0.25"))
    replace_subprocess(monkeypatch, "Popen", FakeFfmpeg(grey_frames(2), log, exit_code=1))
    error = decode_failure(tmp_path)
    assert isinstance(error, DecodeFailed)
    assert error.last_line == "file:/clips/caf\ufffd.mp4: Input/output error"
    message = readable(error)
    assert "(file:/clips/caf\ufffd.mp4: Input/output error)" in message
    assert "Invalid NAL unit size" not in message


def test_ffprobe_output_that_isnt_json_is_refused(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    replace_subprocess(monkeypatch, "run", finished(stdout=b"this is not JSON"))
    assert "ffprobe printed output we couldn't read" in readable(probe_failure(tmp_path))


def test_a_hung_ffmpeg_is_killed_at_the_limit_with_a_message_naming_ffmpeg_timeout_seconds(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    process = HungFfmpeg()
    replace_subprocess(monkeypatch, "Popen", process)
    error = decode_failure(tmp_path, LIMIT)
    assert process.killed
    # The kill makes ffmpeg end with KILLED, but the timeout wins over that exit code.
    assert not isinstance(error, DecodeFailed)
    message = readable(error)
    assert "ffmpeg took longer than 0.2 seconds" in message
    assert "ffmpeg_timeout_seconds in [frames]" in message


def test_a_timed_out_ffprobe_is_refused_with_a_message_naming_ffprobe_timeout_seconds(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # subprocess.run kills ffprobe at its timeout, then raises TimeoutExpired.
    calls = replace_subprocess(monkeypatch, "run", subprocess.TimeoutExpired(["ffprobe"], LIMIT))
    message = readable(probe_failure(tmp_path, LIMIT))
    [(_, kwargs)] = calls
    assert kwargs["timeout"] == LIMIT
    assert "ffprobe took longer than 0.2 seconds" in message
    assert "ffprobe_timeout_seconds in [frames]" in message


# The pairing test's frames: frame k is grey k, and showinfo logs it at PTS_TIMES[k]. In the
# Used_Part from 1.25 s to 7.5 s, the gate refuses frame 0, which has no time, frame 1 at 1.0 s,
# before the start, frame 4 at 1.5 s, the time before it again, and frame 6 at 7.5 s, the end.
PTS_TIMES = ("NOPTS", "-0.25", "0", "0.25", "0.25", "0.5", "6.25")


def test_decode_pairs_the_frames_with_the_times_in_order_and_skips_the_frames_the_gate_refuses(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    plan = grey_plan(tmp_path)
    process = FakeFfmpeg(grey_frames(len(PTS_TIMES)), showinfo_log(plan, *PTS_TIMES))
    calls = replace_subprocess(monkeypatch, "Popen", process)
    given: list[tuple[float, np.ndarray]] = []
    decode(plan, TIMEOUT, lambda time, frame: given.append((time, frame.copy())))
    assert [time for time, _ in given] == [1.25, 1.5, 1.75]
    assert [frame.tobytes() for _, frame in given] == [grey_frame(k) for k in (2, 3, 5)]
    assert all((frame.shape, frame.dtype) == ((HEIGHT, WIDTH), np.uint8) for _, frame in given)
    # decode ran the plan's command with stdin closed (Req 1.3).
    [(command, kwargs)] = calls
    assert command == decode_command(plan)
    assert kwargs["stdin"] == subprocess.DEVNULL


def test_each_frame_is_a_new_array_so_a_frame_that_on_frame_keeps_holds_its_own_pixels(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    plan = decode_plan(tmp_path / "-clip.mp4", colour=True, width=2, height=1)
    # Three colour frames of 2 × 1, each with six bytes of its own: B, G and R for each pixel.
    frames = [bytes(range(6 * k, 6 * k + 6)) for k in range(3)]
    process = FakeFfmpeg(b"".join(frames), showinfo_log(plan, "0", "0.25", "0.5"))
    replace_subprocess(monkeypatch, "Popen", process)
    kept: list[np.ndarray] = []
    decode(plan, TIMEOUT, lambda time, frame: kept.append(frame))
    assert [frame.shape for frame in kept] == [(1, 2, 3)] * 3
    assert [frame.tobytes() for frame in kept] == frames


@pytest.mark.parametrize(("output", "reason"), [
    # Three frames, and showinfo's lines for only two of them.
    pytest.param(grey_frames(3), "a frame came without its time", id="a frame without a time"),
    # One frame and half of the next.
    pytest.param(grey_frames(1) + grey_frame(1)[:3], "the output ended inside a frame",
                 id="output ending inside a frame"),
])
def test_output_that_cant_be_read_says_so_and_to_install_ffmpeg_5_1_or_newer(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, output: bytes, reason: str) -> None:
    log = showinfo_log(grey_plan(tmp_path), "0", "0.25")
    replace_subprocess(monkeypatch, "Popen", FakeFfmpeg(output, log))
    error = decode_failure(tmp_path)
    # A DecodeFailed would make the step run the Score_Pass again without the Tone_Map (Req 2.5).
    assert not isinstance(error, DecodeFailed)
    message = readable(error)
    assert f"ffmpeg's output couldn't be read ({reason})" in message
    assert "Install ffmpeg 5.1 or newer" in message


def test_an_exception_from_on_frame_goes_up_after_ffmpeg_is_killed_and_its_pipes_closed(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    process = FakeFfmpeg(grey_frames(3), showinfo_log(grey_plan(tmp_path), "0", "0.25", "0.5"))
    replace_subprocess(monkeypatch, "Popen", process)
    # What on_frame raises when the first Pick can't be written, as files.py does on a full disk.
    full_disk = ExpectedFailure("The file images/000000.jpg couldn't be written (No space left "
                                "on device).")

    def on_frame(time: float, frame: np.ndarray) -> NoReturn:
        raise full_disk

    assert decode_failure(tmp_path, on_frame=on_frame) is full_disk
    # ffmpeg was still running, so decode killed it and waited for it, and closed its pipes.
    assert process.events[-2:] == ["kill", "wait"]
    assert process.stdout.closed and process.stderr.closed


# has_filters with fake processes (Req 1.8, 2.4, 13.6)

# The command has_filters runs: ffmpeg's filter listing, with no banner and no reading of stdin.
FILTERS_COMMAND = ["ffmpeg", "-hide_banner", "-nostdin", "-filters"]
# The filters the Tone_Map needs, as the step asks for them.
TONE_MAP_FILTERS = ("zscale", "tonemap")


def listed(listing: str, returncode: int = 0) -> subprocess.CompletedProcess[bytes]:
    """What subprocess.run gives back when ffmpeg -filters ran and printed `listing`."""
    return finished(returncode, listing.encode(), tool="ffmpeg")


@pytest.mark.parametrize(("outcome", "found"), [
    pytest.param(listed(FILTERS_6_1), True, id="6.1 lists both"),
    pytest.param(listed(FILTERS_9_0_2), True, id="9.0.2 lists both"),
    # The laptops' ffmpeg 9.0.2, built without libzimg, lists tonemap but no zscale.
    pytest.param(listed(without(FILTERS_9_0_2, "zscale")), False, id="no zscale"),
    pytest.param(listed(without(FILTERS_6_1, "tonemap")), False, id="no tonemap"),
    # subprocess.run kills ffmpeg at its timeout, then raises TimeoutExpired.
    pytest.param(subprocess.TimeoutExpired(FILTERS_COMMAND, TIMEOUT), False, id="timeout"),
    # Both filters are listed, so only the exit code makes it a no.
    pytest.param(listed(FILTERS_9_0_2, returncode=1), False, id="exit of 1"),
])
def test_has_filters_is_true_only_when_ffmpeg_lists_both_filters_and_ends_well(
        monkeypatch: pytest.MonkeyPatch, outcome: object, found: bool) -> None:
    calls = replace_subprocess(monkeypatch, "run", outcome)
    assert has_filters(TONE_MAP_FILTERS, TIMEOUT) is found
    # It ran the listing with stdin closed (Req 1.3) and with the limit it was given.
    [(command, kwargs)] = calls
    assert command == FILTERS_COMMAND
    assert kwargs["stdin"] == subprocess.DEVNULL
    assert kwargs["timeout"] == TIMEOUT


def test_has_filters_without_ffmpeg_gives_the_failure_for_a_missing_tool(
        monkeypatch: pytest.MonkeyPatch) -> None:
    replace_subprocess(monkeypatch, "run",
                       FileNotFoundError(2, "No such file or directory", "ffmpeg"))
    with pytest.raises(ExpectedFailure) as caught:
        has_filters(TONE_MAP_FILTERS, TIMEOUT)
    message = readable(caught.value)
    assert "ffmpeg isn't installed" in message
    assert "Install ffmpeg, which includes ffprobe" in message


# Property tests

# A Used_Part lies inside a video of at most 10 minutes, as max_video_seconds in [jobs] has it,
# so its start and end are from 0 to 600 s.
LONGEST_VIDEO_SECONDS = 600.0
# What a Frame_Gate is given for each frame: None for a frame logged with NOPTS, or a Frame_Time
# from -10 to 700 s, which reaches past both edges of every Used_Part.
REPORTED_TIMES = st.none() | st.floats(-10, 700)


@dataclass(frozen=True)
class GateCase:
    """A drawn Used_Part, with the times reported to the Frame_Gate, in order."""

    start: float
    end: float
    times: tuple[float | None, ...]


@st.composite
def gate_cases(draw: st.DrawFn) -> GateCase:
    """A Used_Part with 0 <= start < end <= 600, and 1 to 60 reported times.

    The start and the end are two different floats drawn on their own and sorted, so the
    Used_Part is sometimes a hair wide and sometimes most of the video. Each time comes from
    REPORTED_TIMES, is the start or the end exactly, or repeats a time drawn before it. So the
    times come in any order, land on both edges and come again.
    """
    start, end = sorted(draw(st.lists(st.floats(0, LONGEST_VIDEO_SECONDS), min_size=2,
                                      max_size=2, unique=True)))
    edges = st.sampled_from((start, end))
    times: list[float | None] = []
    for _ in range(draw(st.integers(1, 60))):
        repeats = [st.sampled_from(times)] if times else []
        times.append(draw(st.one_of(REPORTED_TIMES, edges, *repeats)))
    return GateCase(start=start, end=end, times=tuple(times))


def oracle_takes(case: GateCase) -> list[bool]:
    """Whether the Frame_Gate takes each of the case's times, in order, found with a plain loop
    from the property's rule: a time is taken when it isn't None, is at or after the start, is
    before the end, and is greater than every time taken before it."""
    taken: list[float] = []
    takes: list[bool] = []
    for time in case.times:
        if (time is not None and case.start <= time < case.end
                and all(time > earlier for earlier in taken)):
            taken.append(time)
            takes.append(True)
        else:
            takes.append(False)
    return takes


@given(gate_cases())
def test_the_frame_gate_keeps_exactly_the_used_part_in_order(case: GateCase) -> None:
    """Feature: frames, Property 3: The Frame_Gate keeps exactly the Used_Part, in order

    **Validates: Requirements 1.4, 1.6**
    """
    gate = FrameGate(case.start, case.end)
    # The gate takes or refuses each time as the rule says, so the times it takes lie in the
    # Used_Part and strictly increase.
    assert [gate.accept(time) for time in case.times] == oracle_takes(case)


# What seek_text is given: any float from 0 to 1e6 s, which reaches far past the longest video's
# 600 s. With both bounds given, Hypothesis draws no NaN and no infinity, so every float is finite.
SEEK_SECONDS = st.floats(0, 1e6)
# The form of a seek text: digits, one ".", and exactly six decimals after it. So it has no
# exponent, and no sign or space either.
SEEK_TEXT = re.compile(r"[0-9]+\.[0-9]{6}")
# One microsecond, exactly. Fraction(1e-6) would be the float nearest it, which isn't.
MICROSECOND = Fraction(1, 1_000_000)


@given(SEEK_SECONDS)
def test_the_seek_text_holds_the_used_part(seconds: float) -> None:
    """Feature: frames, Property 4: The seek text holds the Used_Part

    **Validates: Requirements 1.4**
    """
    # decode_command gives -ss the start rounded down and -to the end rounded up.
    start_text, end_text = seek_text(seconds, up=False), seek_text(seconds, up=True)
    assert SEEK_TEXT.fullmatch(start_text)
    assert SEEK_TEXT.fullmatch(end_text)
    exact, start, end = Fraction(seconds), Fraction(start_text), Fraction(end_text)
    # The start's text is at most the time and the end's at least, each within a microsecond, so
    # ffmpeg's cut holds the Used_Part.
    assert exact - MICROSECOND <= start <= exact
    assert exact <= end <= exact + MICROSECOND


# The real ffmpeg tests (Req 1.4, 1.5, 1.6, 13.4, 13.5, 13.11)

# Skips a test that runs the real ffmpeg and ffprobe when either isn't installed (Req 13.5).
needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg and ffprobe aren't installed")
# The limit of every real ffmpeg and ffprobe run, the runs that make the clips included
# (Req 13.11). Each run on these tiny clips takes well under a second.
REAL_TIMEOUT = 60
# The size of every real clip, tiny so the tests stay light on a laptop. The edges differ, so a
# width and height that swapped places would show.
CLIP_WIDTH, CLIP_HEIGHT = 32, 24
# How close each Frame_Time must be to its frame's time in the clip. ffmpeg 5.1 and 6.1 write
# pts_time with six significant digits. The frames are at least 0.1 s apart, so 1 ms still tells
# each frame from its neighbours.
TIME_TOLERANCE = 0.001
# A select filter for the 20 frames of a 2 s testsrc clip, 0.1 s apart from 0 s. It keeps frames
# 0 to 10, from 0 to 1.0 s, and then every third frame, at 1.3, 1.6 and 1.9 s, so the frames it
# keeps are 0.1 s apart and then 0.3 s apart. select leaves each frame it keeps at its own time.
# The quotes stop the commas in its expression from splitting the filters.
VARIABLE_RATE = "select='not(mod(max(n-10,0),3))'"


def make_clip(folder: Path, seconds: int, *options: str) -> Path:
    """Make `folder / "-clip.mp4"`, a clip of lavfi's testsrc that's `seconds` long at 10 frames
    a second and CLIP_WIDTH × CLIP_HEIGHT, with the real ffmpeg, and give its path.

    `options` go to ffmpeg before the clip's name, like a -vf. The name starts with - on purpose.
    The codec is mpeg4 because every ffmpeg build has it. -fps_mode passthrough keeps each frame
    at its own time: without it, ffmpeg writes an mp4 at a constant frame rate and fills the gaps
    of a variable one with repeated frames.
    """
    path = folder / "-clip.mp4"
    source = f"testsrc=duration={seconds}:size={CLIP_WIDTH}x{CLIP_HEIGHT}:rate=10"
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i", source, *options,
                    "-fps_mode", "passthrough", "-c:v", "mpeg4", "-pix_fmt", "yuv420p",
                    "file:" + str(path.resolve())],
                   check=True, timeout=REAL_TIMEOUT)
    return path


def real_plan(video: Path, start: float, end: float) -> DecodePlan:
    """The grey Decode_Pass of the real clip `video` for the Used_Part from `start` to `end`,
    giving frames at the clip's own size. The clip's only stream, its video, is stream 0."""
    return DecodePlan(video=video, stream_index=0, start=start, end=end, width=CLIP_WIDTH,
                      height=CLIP_HEIGHT, colour=False)


def real_frame_times(plan: DecodePlan) -> list[float]:
    """The Frame_Times of the frames that decode gives for `plan` with the real ffmpeg, in
    order."""
    times: list[float] = []
    decode(plan, REAL_TIMEOUT, lambda time, frame: times.append(time))
    return times


@needs_ffmpeg
def test_the_real_ffmpeg_gives_exactly_the_10_frames_of_the_used_part(tmp_path: Path) -> None:
    # The 3 s clip's 30 frames are 0.1 s apart from 0 s. The Used_Part from 1.0 to 2.0 s holds
    # the frame at its start and not the one at its end (Req 1.4).
    clip = make_clip(tmp_path, 3)
    assert real_frame_times(real_plan(clip, 1.0, 2.0)) == pytest.approx(
        [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9], abs=TIME_TOLERANCE)


@needs_ffmpeg
def test_a_variable_frame_rate_gets_its_real_times_with_no_frame_repeated_or_dropped(
        tmp_path: Path) -> None:
    clip = make_clip(tmp_path, 2, "-vf", VARIABLE_RATE)
    # The Used_Part is the whole clip. Each of its 14 frames comes once, at its own time: none is
    # repeated to fill a 0.3 s gap and none is dropped (Req 1.5, 1.6).
    assert real_frame_times(real_plan(clip, 0.0, 2.0)) == pytest.approx(
        [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.3, 1.6, 1.9],
        abs=TIME_TOLERANCE)


@needs_ffmpeg
def test_probe_stream_gives_the_width_and_height_of_a_real_clip(tmp_path: Path) -> None:
    info = probe_stream(make_clip(tmp_path, 1), REAL_TIMEOUT)
    assert (info.width, info.height) == (CLIP_WIDTH, CLIP_HEIGHT)


# The real Tone_Map test (Req 2.2, 13.5, 13.11)

# The limit of the real filter check: ffprobe_timeout_seconds in config.toml, which the step
# gives has_filters.
FILTER_CHECK_TIMEOUT = 30
# A filter that tags the clip's frames as a phone's HDR clip is tagged: the HLG transfer,
# arib-std-b67, with BT.2020 primaries and matrix, which the Tone_Map's zscale reads. It tags the
# frames themselves, because the laptops' ffmpeg 9.0.2 loses the -color_trc and -color_primaries
# output options.
HLG_TAGS = "setparams=color_primaries=bt2020:color_trc=arib-std-b67:colorspace=bt2020nc"


@needs_ffmpeg
def test_the_real_tone_map_gives_an_hlg_clips_frames_at_the_plans_size(tmp_path: Path) -> None:
    # The check runs here, as the test runs, so collecting the tests never runs ffmpeg.
    if not has_filters(TONE_MAP_FILTERS, FILTER_CHECK_TIMEOUT):
        pytest.skip("this ffmpeg has no zscale or tonemap filter")
    clip = make_clip(tmp_path, 1, "-vf", HLG_TAGS)
    # ffprobe finds the clip HLG, so the step would give it the Tone_Map.
    assert probe_stream(clip, REAL_TIMEOUT).colour_transfer == "arib-std-b67"
    # A grey Decode_Pass of the whole clip, with the Tone_Map's numbers in config.toml.
    plan = replace(real_plan(clip, 0.0, 1.0), tone_map=ToneMap(100.0, 0.0))
    shapes: list[tuple[int, ...]] = []
    decode(plan, REAL_TIMEOUT, lambda time, frame: shapes.append(frame.shape))
    # Each of the 1 s clip's 10 frames comes through the Tone_Map, at the plan's size.
    assert shapes == [(plan.height, plan.width)] * 10
