"""Tests for splatpipe/frames/ffmpeg.py: ffprobe and the ffmpeg decode.

The unit tests of the pure parts run no tool, so they pass on a machine without ffmpeg (Req 13.6).

- parse_stream gets ffprobe's -show_streams JSON as json.loads gives it, with the fields of a
  phone clip.
- parse_frame_time gets showinfo's lines as ffmpeg prints them on stderr, without the newline.
  The lines marked 9.0.2 are copied from the output of ffmpeg 9.0.2, the version on our laptops.
  The others follow the format strings in libavfilter/vf_showinfo.c of the release they're
  marked with. 6.1 is Ubuntu 24.04's packaged ffmpeg.
- decode_command gets DecodePlans, and the tests check the argument list it gives.
"""

from pathlib import Path
from typing import Any

import pytest

from splatpipe.errors import ExpectedFailure
from splatpipe.frames.ffmpeg import (NOT_A_FRAME, DecodePlan, VideoInfo, decode_command,
                                     parse_frame_time, parse_stream)


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

def decode_plan(video: Path, *, colour: bool, width: int, height: int) -> DecodePlan:
    """The Decode_Pass of the video's stream 2 for the Used_Part from 1.25 s to 7.5 s, giving
    frames of width × height in colour or in grey."""
    return DecodePlan(video=video, stream_index=2, start=1.25, end=7.5, width=width,
                      height=height, colour=colour)


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
