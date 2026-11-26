"""The Box_Reader and the Video_Check, pure over `read(offset, length)`.

The website never holds a whole video: it walks the top-level MP4/MOV Boxes by their headers alone,
reads the one `moov` Box, and takes the Video_Length from the first video track's `mdhd`, or the
movie header `mvhd` when that's 0. Every refusal is a `PublicRefusal` with what to do on a phone.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING

from splatpipe.errors import PipelineError
from splatpipe.make import used_part

from app.errors import UNPROCESSABLE, PublicRefusal

if TYPE_CHECKING:
    from splatpipe.config import JobsSettings

    from app.config import WebsiteConfig

VIDEO_FILES = {".mp4": "video.mp4", ".mov": "video.mov"}   # the Job_Record's video_file names
Read = Callable[[int, int], bytes]

_HEADER = 8
_LARGE_HEADER = 16
_WORD = 4
_LONG = 8
_UNKNOWN_32 = 0xFFFFFFFF
_UNKNOWN_64 = 0xFFFFFFFFFFFFFFFF
_BYTES_PER_GB = 1024 ** 3
_SECONDS_PER_MINUTE = 60

BROKEN = ("This file isn't an MP4 or MOV video the pipeline can read. Save it again as MP4 on your "
          "phone and upload it again.")
WRONG_TYPE = "Only MP4 and MOV videos can be used. Pick a .mp4 or .mov file and upload it again."
GONE = "That upload isn't there any more. Upload the video again."
INCOMPLETE = "The upload didn't arrive complete. Upload the video again."
NO_VIDEO = "This file has no video track in it. Pick a video and upload it again."


@dataclass(frozen=True)
class MovieFacts:
    has_video: bool
    video_seconds: float | None


@dataclass(frozen=True)
class VideoFacts:
    video_file: str
    size: int
    video_seconds: float
    start_seconds: float
    end_seconds: float


def refuse(message: str) -> PublicRefusal:
    return PublicRefusal(UNPROCESSABLE, "video", message)


def video_file_for(name: str) -> str:
    """"video.mp4" or "video.mov" from a name's extension, in any letter case."""
    lowered = name.lower()
    for extension, video_file in VIDEO_FILES.items():
        if lowered.endswith(extension):
            return video_file
    raise refuse(WRONG_TYPE)


def find_moov(read: Read, size: int, max_boxes: int, max_moov_bytes: int) -> tuple[int, int]:
    """The `moov` Box's offset and length, reading only top-level headers."""
    offset, count = 0, 0
    while offset < size:
        count += 1
        if count > max_boxes:
            raise refuse(BROKEN)
        header = read(offset, min(_HEADER, size - offset))
        if len(header) < _HEADER:
            raise refuse(BROKEN)
        box_size, kind, header_len = int.from_bytes(header[:_WORD], "big"), header[_WORD:], _HEADER
        to_end = box_size == 0
        if box_size == 1:
            large = read(offset + _HEADER, min(_LONG, size - offset - _HEADER))
            if len(large) < _LONG:
                raise refuse(BROKEN)
            box_size, header_len = int.from_bytes(large, "big"), _LARGE_HEADER
        elif to_end:
            box_size = size - offset
        if box_size < header_len or offset + box_size > size:
            raise refuse(BROKEN)
        if kind == b"moov":
            if box_size > max_moov_bytes:
                raise refuse(BROKEN)
            return offset, box_size
        if to_end:
            break
        offset += box_size
    raise refuse(BROKEN)


def children(data: bytes, start: int, end: int) -> Iterator[tuple[bytes, int, int]]:
    """(type, payload start, payload end) of each Box between `start` and `end` of `data`."""
    offset = start
    while offset < end:
        if end - offset < _HEADER:
            raise refuse(BROKEN)
        box_size = int.from_bytes(data[offset:offset + _WORD], "big")
        kind = data[offset + _WORD:offset + _HEADER]
        header_len = _HEADER
        if box_size == 1:
            if end - offset < _LARGE_HEADER:
                raise refuse(BROKEN)
            box_size = int.from_bytes(data[offset + _HEADER:offset + _LARGE_HEADER], "big")
            header_len = _LARGE_HEADER
        elif box_size == 0:
            box_size = end - offset
        if box_size < header_len or offset + box_size > end:
            raise refuse(BROKEN)
        yield kind, offset + header_len, offset + box_size
        offset += box_size


def _time(data: bytes, start: int, end: int) -> tuple[int, int] | None:
    """(time scale, duration) of an `mvhd` or `mdhd` payload, the duration 0 when unknown."""
    if end - start < _WORD:
        return None
    version = data[start]
    if version == 1:
        if end - start < 32:
            return None
        scale = int.from_bytes(data[start + 20:start + 24], "big")
        duration = int.from_bytes(data[start + 24:start + 32], "big")
        return scale, 0 if duration == _UNKNOWN_64 else duration
    if end - start < 20:
        return None
    scale = int.from_bytes(data[start + 12:start + 16], "big")
    duration = int.from_bytes(data[start + 16:start + 20], "big")
    return scale, 0 if duration == _UNKNOWN_32 else duration


def _track(data: bytes, start: int, end: int) -> tuple[bytes | None, tuple[int, int] | None]:
    """A `trak`'s handler type and its `mdhd` time."""
    handler, times = None, None
    for kind, a, b in children(data, start, end):
        if kind != b"mdia":
            continue
        for inner, c, d in children(data, a, b):
            if inner == b"hdlr" and d - c >= 12:
                handler = data[c + 8:c + 12]
            elif inner == b"mdhd":
                times = _time(data, c, d)
    return handler, times


def read_movie(moov: bytes) -> MovieFacts:
    """Whether the movie has a video track, and its Video_Length in seconds (None if unreadable)."""
    movie, video = None, None
    has_video = False
    for kind, start, end in children(moov, _moov_header(moov), len(moov)):
        if kind == b"mvhd":
            movie = _time(moov, start, end)
        elif kind == b"trak" and not has_video:
            handler, times = _track(moov, start, end)
            if handler == b"vide":
                has_video, video = True, times
    return MovieFacts(has_video=has_video, video_seconds=_seconds(video) or _seconds(movie))


def _moov_header(moov: bytes) -> int:
    """The length of the moov Box's own header: 16 with a 64-bit largesize, else 8."""
    if len(moov) < _HEADER or moov[_WORD:_HEADER] != b"moov":
        raise refuse(BROKEN)
    return _LARGE_HEADER if int.from_bytes(moov[:_WORD], "big") == 1 else _HEADER


def _seconds(times: tuple[int, int] | None) -> float | None:
    if not times or times[0] == 0 or times[1] == 0:
        return None
    return times[1] / times[0]


def check_video(read: Read, size: int | None, stated_size: int, name: str, wc: WebsiteConfig,
                jobs: JobsSettings) -> VideoFacts:
    """The Video_Check, in order: the name, the object still there, its size, the Boxes, the facts."""
    video_file = video_file_for(name)
    if size is None:
        raise refuse(GONE)
    if size > jobs.max_video_bytes:
        raise refuse(f"This video is larger than {jobs.max_video_bytes / _BYTES_PER_GB:g} GB, the "
                     "largest a job takes. Film a shorter video, or trim it on your phone, and "
                     "upload it again.")
    if size != stated_size:
        raise refuse(INCOMPLETE)
    offset, length = find_moov(read, size, wc.max_boxes, wc.max_moov_bytes)
    moov = read(offset, length)
    if len(moov) != length:
        raise refuse(BROKEN)
    facts = read_movie(moov)
    if not facts.has_video:
        raise refuse(NO_VIDEO)
    seconds = facts.video_seconds
    if seconds is None or seconds <= 0:
        raise refuse(BROKEN)
    if seconds > jobs.max_video_seconds:
        raise refuse(f"This video is {seconds:.0f} s long, longer than "
                     f"{jobs.max_video_seconds / _SECONDS_PER_MINUTE:g} minutes, the longest a job "
                     "takes. Trim it on your phone and upload it again.")
    try:
        start, end = used_part(seconds, None, None, jobs.min_used_seconds)
    except PipelineError:
        raise refuse(f"This video is shorter than {jobs.min_used_seconds:g} s, the shortest a job "
                     "may use. Film a longer video and upload it again.") from None
    return VideoFacts(video_file=video_file, size=size, video_seconds=seconds, start_seconds=start,
                      end_seconds=end)
