"""The Video_Probe: what ffprobe says about a video's container format, streams and length.

`probe_video` runs ffprobe as an argument list, never through a shell, and gives it the video
as `file:` plus the absolute path, so a file name is never read as an ffprobe option or a
network address (Req 2.6). ffprobe missing, a timeout, a non-zero exit and output we can't
read are each a PipelineError with a Readable_Error (Req 2.7). `parse_probe` is pure: it reads
ffprobe's JSON into a ProbeResult. Checking the format name against CONTAINER_FAMILY is up to
the Job_Maker, in splatpipe.make.

From splatpipe, this module imports only splatpipe.errors.
"""

from __future__ import annotations

import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from splatpipe.errors import PipelineError

_UNREADABLE = ("ffprobe printed output we couldn't read. Install a recent ffmpeg, which "
               "includes ffprobe, and try again.")


@dataclass(frozen=True)
class ProbeResult:
    """What ffprobe found in a video."""

    format_name: str  # the container format, like "mov,mp4,m4a,3gp,3g2,mj2"
    has_video: bool  # True when there's a video stream that isn't cover art
    video_seconds: float | None  # the Video_Length, or None when there's no usable length


def probe_video(path: Path, timeout_seconds: float) -> ProbeResult:
    """Run ffprobe on the video at `path`, for at most `timeout_seconds`, and read its output.

    Every failure is refused with a PipelineError that says what went wrong and what to do.
    """
    command = ["ffprobe", "-v", "error", "-print_format", "json", "-show_format",
               "-show_streams", "file:" + str(path.resolve())]
    try:
        result = subprocess.run(command, capture_output=True, stdin=subprocess.DEVNULL,
                                timeout=timeout_seconds)
    except FileNotFoundError as error:
        raise PipelineError("ffprobe isn't installed. Install ffmpeg, which includes ffprobe, "
                            "and make sure `ffprobe` runs in a terminal.") from error
    except subprocess.TimeoutExpired as error:
        raise PipelineError(f"ffprobe took longer than {timeout_seconds:g} seconds to read the "
                            "video, so we stopped it. Check that the video plays, or raise "
                            "ffprobe_timeout_seconds under [jobs] in config.toml.") from error
    if result.returncode != 0:
        detail = _first_line(result.stderr) or f"exit code {result.returncode}"
        raise PipelineError(f"ffprobe couldn't read the video ({detail}). Check that the file "
                            "is a complete video that plays, then try again.")
    try:
        data = json.loads(result.stdout)
    except ValueError as error:  # not JSON, or bytes in no Unicode encoding
        raise PipelineError(_UNREADABLE) from error
    return parse_probe(data)


def parse_probe(data: object) -> ProbeResult:
    """Read ffprobe's JSON, from -show_format and -show_streams, into a ProbeResult.

    format must be an object whose format_name is text, and streams a list of objects, or the
    output is refused with a PipelineError. Cover art, a still picture whose
    disposition.attached_pic is 1, isn't counted as a video stream.
    """
    top = _object(data)
    container = _object(top.get("format"))
    format_name = container.get("format_name")
    streams = top.get("streams")
    if not isinstance(format_name, str) or not isinstance(streams, list):
        raise PipelineError(_UNREADABLE)
    videos = [stream for stream in map(_object, streams) if _is_video(stream)]
    return ProbeResult(format_name=format_name, has_video=bool(videos),
                       video_seconds=_length(videos, container))


def _is_video(stream: dict[str, Any]) -> bool:
    """True for a video stream that isn't cover art. A missing disposition isn't cover art."""
    disposition = _object(stream.get("disposition", {}))
    return stream.get("codec_type") == "video" and disposition.get("attached_pic") != 1


def _length(videos: list[dict[str, Any]], container: dict[str, Any]) -> float | None:
    """The first video stream's usable duration, else the container's (Req 2.10)."""
    seconds = _seconds(videos[0].get("duration")) if videos else None
    return seconds if seconds is not None else _seconds(container.get("duration"))


def _seconds(value: Any) -> float | None:
    """`value` in seconds when float() reads it as a finite number above 0, or else None.

    ffprobe prints durations as strings, and "N/A" when it doesn't know one. NaN, infinity,
    0, negatives, other text and a missing value all give None.
    """
    try:
        seconds = float(value)
    except (TypeError, ValueError, OverflowError):  # None, "N/A", other text, a huge int
        return None
    return seconds if math.isfinite(seconds) and seconds > 0 else None


def _object(value: object) -> dict[str, Any]:
    """`value` when it's a JSON object. Anything else is output we couldn't read."""
    if not isinstance(value, dict):
        raise PipelineError(_UNREADABLE)
    return value


def _first_line(stderr: bytes) -> str:
    """The first line ffprobe printed on stderr, with bytes that aren't UTF-8 replaced."""
    return stderr.decode("utf-8", errors="replace").strip().partition("\n")[0].strip()
