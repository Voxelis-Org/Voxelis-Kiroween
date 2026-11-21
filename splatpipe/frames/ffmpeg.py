"""ffprobe and the ffmpeg decode, the Frames_Step's two tools, each with its pure parts.

Every frame the step uses comes from the ffmpeg command line (Req 1.2). Each tool runs as an
argument list with no shell and with stdin closed, and gets the video as file: plus its absolute
path, so a file name is never read as an option or a network address (Req 1.3).

probe_stream runs ffprobe, and parse_stream reads its JSON into a VideoInfo for the first video
stream that isn't cover art (Req 2.1): the stream's index, its Displayed_Shape and its colour
transfer. The Displayed_Shape is the stored size with the sample aspect ratio applied to the
width, rounded half up, and with the edges swapped when the rotation turns the picture on its
side, as ffmpeg's own autorotation does.

decode runs one Decode_Pass, with the command decode_command gives:

- -ss and -to are input options from seek_text: the start rounded down and the end rounded up to
  the microsecond, so ffmpeg's cut always holds the Used_Part. The FrameGate then makes the exact
  cut, start <= Frame_Time < end, and drops a frame whose time is missing (NOPTS) or isn't after
  the last time it kept, so the Frame_Times strictly increase (Req 1.4, 1.6).
- scale=W:H,setsar=1 gives every frame the plan's exact size, so each frame on the pipe has a
  known byte count. format=gray gives the Score_Frames and format=bgr24 the Picks' colours.
- showinfo=checksum=0, the last filter, logs each frame's pts_time on stderr, and
  -fps_mode passthrough stops ffmpeg from repeating or dropping frames (Req 1.5). The command
  needs ffmpeg 5.1 or newer (D5).

A daemon thread reads stderr line by line, so ffmpeg never blocks on it, and the k-th frame on
stdout pairs with the k-th showinfo time. A Frame_Time is the -ss value plus the pts_time, which
without -copyts is the time from the start of the file, the clock the Trim uses. decode holds one
frame at a time, and each frame is a new array over its own bytes, so on_frame may keep it
(Req 1.12). A threading.Timer kills ffmpeg at the limit.

Each failure is an ExpectedFailure whose message says what went wrong and what to do, with no
traceback: a missing tool (Req 1.8), a timeout naming its setting in [frames] (Req 1.9), a
non-zero exit with the tool's last line of error output (Req 1.10), raised as DecodeFailed for
ffmpeg, ffprobe output that isn't the JSON we asked for, a stream with no picture size, and
ffmpeg output with a frame that has no showinfo time or that ends inside a frame (Req 1.14).

From splatpipe, this module imports only splatpipe.errors.
"""

from __future__ import annotations

import enum
import json
import math
import queue
import re
import subprocess
import threading
from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import IO, Any, Final, cast

import numpy as np

from splatpipe.errors import ExpectedFailure

_PLAYS = "Check that the file is a complete video that plays, then try again."
_PROBE_UNREADABLE = ("ffprobe printed output we couldn't read. Install a recent ffmpeg, which "
                     "includes ffprobe, and try again.")
_NO_SIZE = f"ffprobe found no picture size in the video. {_PLAYS}"
# Why decode couldn't read ffmpeg's output (Req 1.14).
_NO_TIME = "a frame came without its time"
_PART_FRAME = "the output ended inside a frame"
_SHOWINFO = "[Parsed_showinfo_"  # how every line that ffmpeg's showinfo filter logs starts
# A showinfo frame line: the filter's log context, then n:, pts: and pts_time:, which it captures.
_FRAME_LINE = re.compile(r"\[Parsed_showinfo_[0-9]+ @ [^\]]*\]\s+n:\s*[0-9]+\s+pts:\s*\S+\s+"
                         r"pts_time:(\S+)")
_ASPECT = re.compile(r"([0-9]+):([0-9]+)")  # a sample aspect ratio N:M, like 4:3
_MICROS = 1_000_000  # microseconds in a second


@dataclass(frozen=True)
class VideoInfo:
    """What the step needs to know about the video's first video stream that isn't cover art."""

    stream_index: int  # ffprobe's index of the stream, for -map 0:<index>
    width: int  # the Displayed_Shape: the sample aspect ratio and the rotation applied
    height: int
    colour_transfer: str | None  # as ffprobe reports it, None when it reports none


def parse_stream(data: object) -> VideoInfo:
    """Read ffprobe's -show_streams JSON into the VideoInfo of the first video stream that isn't
    cover art, a still picture whose disposition.attached_pic is 1 (Req 2.1).

    streams must be a list of objects, and the stream's index a whole number of 0 or more, or the
    output is refused as unreadable. The stream's width and height must be whole numbers above 0,
    or it's refused as having no picture size, and so is a video with no such stream.
    """
    streams = _object(data).get("streams")
    if not isinstance(streams, list):
        raise ExpectedFailure(_PROBE_UNREADABLE)
    videos = (stream for stream in map(_object, streams) if _is_video(stream))
    stream = next(videos, None)
    if stream is None:
        raise ExpectedFailure(_NO_SIZE)
    return _video_info(stream)


def probe_stream(path: Path, timeout_seconds: float) -> VideoInfo:
    """Run ffprobe on the video at `path`, for at most `timeout_seconds`, and read its first video
    stream that isn't cover art. Every failure is refused with an ExpectedFailure."""
    command = ["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", _input(path)]
    try:
        result = subprocess.run(command, capture_output=True, stdin=subprocess.DEVNULL,
                                timeout=timeout_seconds)
    except FileNotFoundError as error:
        raise _missing("ffprobe") from error
    except subprocess.TimeoutExpired as error:  # subprocess.run has already killed ffprobe
        raise _too_slow("ffprobe", timeout_seconds, "ffprobe_timeout_seconds") from error
    if result.returncode != 0:
        detail = _last_line(result.stderr) or f"exit code {result.returncode}"
        raise ExpectedFailure(f"ffprobe couldn't read the video ({detail}). {_PLAYS}")
    try:
        data = json.loads(result.stdout)
    except ValueError as error:  # not JSON, or bytes in no Unicode encoding
        raise ExpectedFailure(_PROBE_UNREADABLE) from error
    return parse_stream(data)


def seek_text(seconds: float, *, up: bool = False) -> str:
    """`seconds`, 0 or more, as text for ffmpeg with exactly six decimals and never an exponent.

    The exact value is rounded down to the microsecond, or up when `up` is True. So the start's
    text as -ss and the end's text, rounded up, as -to give a cut that holds the Used_Part.
    """
    micros = Fraction(seconds) * _MICROS
    whole = math.ceil(micros) if up else math.floor(micros)
    return f"{whole // _MICROS}.{whole % _MICROS:06d}"


@dataclass(frozen=True)
class DecodePlan:
    """One Decode_Pass: the video and its stream, the Used_Part, and the frames' size and format."""

    video: Path
    stream_index: int
    start: float  # the Used_Part, from the Job_Record
    end: float
    width: int  # the frame size on the pipe
    height: int
    colour: bool  # bgr24 when True, gray when False


def decode_command(plan: DecodePlan) -> list[str]:
    """The ffmpeg command of the Decode_Pass `plan`, as an argument list for no shell."""
    pixels = "bgr24" if plan.colour else "gray"
    filters = f"scale={plan.width}:{plan.height},setsar=1,format={pixels},showinfo=checksum=0"
    return ["ffmpeg", "-nostdin", "-hide_banner", "-nostats", "-loglevel", "info",
            "-ss", seek_text(plan.start), "-to", seek_text(plan.end, up=True),
            "-i", _input(plan.video),
            "-map", f"0:{plan.stream_index}", "-an", "-sn", "-dn",
            "-vf", filters,
            "-fps_mode", "passthrough", "-f", "rawvideo", "pipe:1"]


class NotAFrame(enum.Enum):
    """The type of NOT_A_FRAME."""

    NOT_A_FRAME = "not a frame"


# parse_frame_time's answer for a line that isn't a frame's, like showinfo's config lines.
NOT_A_FRAME: Final = NotAFrame.NOT_A_FRAME


def parse_frame_time(line: str) -> float | None | NotAFrame:
    """The pts_time of a showinfo frame line, None when it's NOPTS, or NOT_A_FRAME.

    A frame line starts with the Parsed_showinfo_<n> log context and holds n:, pts: and
    pts_time:, as ffmpeg 5.1 and newer log it. Every other line is NOT_A_FRAME: showinfo's config
    and side data lines, and every line that isn't showinfo's.
    """
    match = _FRAME_LINE.match(line)
    if match is None:
        return NOT_A_FRAME
    if match[1] == "NOPTS":
        return None
    try:
        return float(match[1])
    except ValueError:
        return NOT_A_FRAME


class FrameGate:
    """Keeps the frames of the Used_Part whose Frame_Times strictly increase (Req 1.4, 1.6).

    It holds the Used_Part, the -ss value and only the last Frame_Time it accepted.
    """

    def __init__(self, start: float, end: float) -> None:
        self._start = start
        self._end = end
        self._offset = float(seek_text(start))  # the -ss value, which pts_time counts from
        self._last: float | None = None  # the last Frame_Time accepted, None before the first

    def frame_time(self, pts_time: float | None) -> float | None:
        """The Frame_Time of a frame logged at `pts_time`, the -ss value plus pts_time, or None
        for a frame logged with NOPTS."""
        return None if pts_time is None else self._offset + pts_time

    def accept(self, time: float | None) -> bool:
        """Whether to keep a frame at Frame_Time `time`: only when it isn't None, is at or after
        the start, is before the end and is after the last time accepted. A time it accepts
        becomes the last time."""
        if time is None or not self._start <= time < self._end:
            return False
        if self._last is not None and time <= self._last:
            return False
        self._last = time
        return True


class DecodeFailed(ExpectedFailure):
    """ffmpeg exited with a non-zero code in a Decode_Pass (Req 1.10).

    `last_line` is its last line of error output that isn't showinfo's, or its exit code when
    there's none.
    """

    def __init__(self, last_line: str) -> None:
        super().__init__(f"ffmpeg couldn't read the video ({last_line}). {_PLAYS}")
        self.last_line = last_line


def decode(plan: DecodePlan, timeout_seconds: float,
           on_frame: Callable[[float, np.ndarray], None]) -> None:
    """Run the Decode_Pass `plan` for at most `timeout_seconds`, and give each frame the
    FrameGate accepts to on_frame(Frame_Time, frame), in order.

    A frame is a uint8 array of plan.height × plan.width, with a third axis of 3 in BGR order
    when plan.colour is True. Each one is a new array over its own bytes, so on_frame may keep
    it. An exception from on_frame goes up once ffmpeg is stopped and its pipes are closed.
    """
    try:
        process = subprocess.Popen(decode_command(plan), stdin=subprocess.DEVNULL,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError as error:
        raise _missing("ffmpeg") from error
    stdout, stderr = cast(IO[bytes], process.stdout), cast(IO[bytes], process.stderr)
    log = _StderrReader(stderr)
    timed_out = threading.Event()
    timer = threading.Timer(timeout_seconds, _stop, (process, timed_out))
    timer.daemon = True
    timer.start()
    try:
        problem = _read_frames(stdout, plan, log.times, on_frame)
        if problem != _NO_TIME:  # the output has ended, so ffmpeg exits by itself, with its code
            process.wait()
    finally:
        timer.cancel()
        if process.poll() is None:  # on_frame raised, or a frame came without its time
            process.kill()
        process.wait()
        log.join()
        stdout.close()
        stderr.close()
    if timed_out.is_set():  # a timeout wins over every error it causes
        raise _too_slow("ffmpeg", timeout_seconds, "ffmpeg_timeout_seconds")
    if problem != _NO_TIME and process.returncode != 0:
        raise DecodeFailed(log.last_line or f"exit code {process.returncode}")
    if problem is not None:
        raise ExpectedFailure(f"ffmpeg's output couldn't be read ({problem}). Install ffmpeg 5.1 "
                              "or newer, then run the step again.")


class _End(enum.Enum):
    """The type of the marker that follows the last frame time from ffmpeg's stderr."""

    END = "end of stderr"


class _StderrReader:
    """Reads ffmpeg's stderr line by line on a daemon thread, so ffmpeg never blocks on it.

    Each frame line's time, or None for NOPTS, goes on `times`, and _End.END follows the end of
    stderr. `last_line` is the last non-empty line that isn't showinfo's. Read it after join().
    """

    def __init__(self, stderr: IO[bytes]) -> None:
        self.times: queue.Queue[float | None | _End] = queue.Queue()
        self.last_line = ""
        self._stderr = stderr
        self._thread = threading.Thread(target=self._read, daemon=True)
        self._thread.start()

    def join(self) -> None:
        """Wait until the thread has read stderr to its end."""
        self._thread.join()

    def _read(self) -> None:
        """Read every line of stderr, decoded as UTF-8 with bytes that aren't UTF-8 replaced."""
        try:
            for raw in self._stderr:
                line = raw.decode("utf-8", errors="replace").strip()
                time = parse_frame_time(line)
                if time is not NOT_A_FRAME:
                    self.times.put(time)
                elif line and not line.startswith(_SHOWINFO):
                    self.last_line = line
        finally:
            self.times.put(_End.END)


def _read_frames(stdout: IO[bytes], plan: DecodePlan, times: queue.Queue[float | None | _End],
                 on_frame: Callable[[float, np.ndarray], None]) -> str | None:
    """Read whole frames from stdout to its end, pair each with the next time on `times`, and
    give on_frame the frames the FrameGate accepts.

    Gives None when the output ends after a whole frame, or else why it couldn't be read.
    """
    shape = (plan.height, plan.width, 3) if plan.colour else (plan.height, plan.width)
    size = math.prod(shape)
    gate = FrameGate(plan.start, plan.end)
    while data := _read_exactly(stdout, size):
        if len(data) < size:
            return _PART_FRAME
        pts_time = times.get()
        if pts_time is _End.END:
            return _NO_TIME
        time = gate.frame_time(pts_time)
        if time is not None and gate.accept(time):
            on_frame(time, np.frombuffer(data, np.uint8).reshape(shape))
        del data  # so no frame is held while the next one is read (Req 1.12)
    return None


def _read_exactly(stream: IO[bytes], size: int) -> bytes:
    """The next `size` bytes of `stream`, or fewer only when its output ends first."""
    data = stream.read(size)
    while data and len(data) < size:
        more = stream.read(size - len(data))
        if not more:
            break
        data += more
    return data


def _stop(process: subprocess.Popen[bytes], timed_out: threading.Event) -> None:
    """The Timer's action at the limit: mark the Decode_Pass as timed out, then kill ffmpeg."""
    timed_out.set()
    process.kill()


def _video_info(stream: dict[str, Any]) -> VideoInfo:
    """The VideoInfo of one video stream from ffprobe's JSON."""
    width, height, index = stream.get("width"), stream.get("height"), stream.get("index")
    if not (_whole_above_zero(width) and _whole_above_zero(height)):
        raise ExpectedFailure(_NO_SIZE)
    if type(index) is not int or index < 0:
        raise ExpectedFailure(_PROBE_UNREADABLE)
    width = _widened(width, stream.get("sample_aspect_ratio"))
    if _on_its_side(_rotation(stream)):
        width, height = height, width
    transfer = stream.get("color_transfer")
    return VideoInfo(stream_index=index, width=width, height=height,
                     colour_transfer=transfer if isinstance(transfer, str) and transfer else None)


def _widened(width: int, ratio: object) -> int:
    """`width` with a sample aspect ratio of N:M applied: N/M times as wide, rounded half up.

    0:1, N/A, a missing ratio and any other text count as square pixels.
    """
    match = _ASPECT.fullmatch(ratio) if isinstance(ratio, str) else None
    n, m = (int(match[1]), int(match[2])) if match else (0, 0)
    if n == 0 or m == 0:
        return width
    return (2 * width * n + m) // (2 * m)


def _rotation(stream: dict[str, Any]) -> float:
    """The stream's rotation in degrees: the first rotation in side_data_list, else tags.rotate,
    else 0. A value that isn't a finite number counts as 0."""
    side_data = stream.get("side_data_list")
    for entry in side_data if isinstance(side_data, list) else []:
        if isinstance(entry, dict) and "rotation" in entry:
            return _degrees(entry["rotation"])
    tags = stream.get("tags")
    if isinstance(tags, dict) and "rotate" in tags:
        return _degrees(tags["rotate"])
    return 0.0


def _degrees(value: Any) -> float:
    """`value` as a finite number of degrees, from a JSON number or text like "90", else 0."""
    try:
        degrees = float(value)
    except (TypeError, ValueError, OverflowError):  # None, other text, a list, a huge int
        return 0.0
    return degrees if math.isfinite(degrees) else 0.0


def _on_its_side(rotation: float) -> bool:
    """Whether ffmpeg's autorotation turns the picture on its side, swapping its edges."""
    return abs(rotation % 180 - 90) < 1


def _whole_above_zero(value: object) -> bool:
    """Whether `value` is an int above 0. type() is checked exactly, so a bool is never one."""
    return type(value) is int and value > 0


def _is_video(stream: dict[str, Any]) -> bool:
    """True for a video stream that isn't cover art. A missing disposition isn't cover art."""
    disposition = _object(stream.get("disposition", {}))
    return stream.get("codec_type") == "video" and disposition.get("attached_pic") != 1


def _object(value: object) -> dict[str, Any]:
    """`value` when it's a JSON object. Anything else is output we couldn't read."""
    if not isinstance(value, dict):
        raise ExpectedFailure(_PROBE_UNREADABLE)
    return value


def _input(path: Path) -> str:
    """The video for a tool: file: plus its absolute path, never an option or a network address."""
    return "file:" + str(path.resolve())


def _last_line(stderr: bytes) -> str:
    """The last non-empty line a tool printed on stderr, with bytes that aren't UTF-8 replaced."""
    lines = stderr.decode("utf-8", errors="replace").splitlines()
    return next((line.strip() for line in reversed(lines) if line.strip()), "")


def _missing(tool: str) -> ExpectedFailure:
    """The Req 1.8 failure: `tool` isn't installed."""
    return ExpectedFailure(f"{tool} isn't installed. Install ffmpeg, which includes ffprobe, and "
                           f"make sure `{tool}` runs in a terminal.")


def _too_slow(tool: str, seconds: float, setting: str) -> ExpectedFailure:
    """The Req 1.9 failure: `tool` ran past its limit of `seconds`, which `setting` sets."""
    return ExpectedFailure(f"{tool} took longer than {seconds:g} seconds to read the video, so "
                           f"we stopped it. Check that the video plays, or raise {setting} in "
                           "[frames] in config.toml.")
