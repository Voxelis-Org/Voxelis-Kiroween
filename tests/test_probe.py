"""Tests for the Video_Probe, splatpipe/probe.py.

The unit tests replace subprocess.run through monkeypatch, so they never run the real ffprobe
(Req 14.3). The real tests at the end run probe_video, and make_job with the real probe, on a
clip made with ffmpeg. They're skipped when ffmpeg or ffprobe isn't installed (Req 14.4, 14.5).
"""

import json
import shutil
import subprocess
import tomllib
from pathlib import Path
from typing import Any

import pytest

from splatpipe.config import Config, load_config
from splatpipe.errors import PipelineError
from splatpipe.jobs import open_job
from splatpipe.make import CONTAINER_FAMILY, make_job
from splatpipe.probe import ProbeResult, parse_probe, probe_video

# The format name ffprobe reports for MP4 and MOV files: the Container_Family.
FAMILY = "mov,mp4,m4a,3gp,3g2,mj2"


# Unit tests

# The timeout the unit tests give probe_video.
TIMEOUT = 12.5
# The start of the refusal for output that ffprobe printed but the Video_Probe can't read.
UNREADABLE = "ffprobe printed output we couldn't read"
# What a call to subprocess.run was given: the command and the keyword arguments.
Call = tuple[list[str], dict[str, Any]]


def duration_field(duration: str | None) -> dict[str, str]:
    """A duration as ffprobe prints it, or no field at all when `duration` is None."""
    return {} if duration is None else {"duration": duration}


def stream(codec_type: str, duration: str | None = None, attached_pic: int = 0) -> dict[str, Any]:
    """A stream as ffprobe prints it. attached_pic=1 makes it cover art."""
    return {"codec_type": codec_type, "disposition": {"attached_pic": attached_pic},
            **duration_field(duration)}


def output(streams: list[dict[str, Any]], duration: str | None = None) -> dict[str, Any]:
    """ffprobe's JSON from -show_format and -show_streams. The container has a duration unless
    `duration` is None."""
    return {"format": {"format_name": FAMILY, **duration_field(duration)}, "streams": streams}


def printed(data: object) -> bytes:
    """`data` as the JSON bytes ffprobe prints on stdout."""
    return json.dumps(data).encode()


def finished(returncode: int = 0, stdout: bytes = b"",
             stderr: bytes = b"") -> subprocess.CompletedProcess[bytes]:
    """What subprocess.run gives back when ffprobe ran, with its output captured as bytes."""
    return subprocess.CompletedProcess(["ffprobe"], returncode, stdout=stdout, stderr=stderr)


def replace_run(monkeypatch: pytest.MonkeyPatch, outcome: object) -> list[Call]:
    """Replace subprocess.run with a stand-in that records each call, then raises `outcome` if
    it's an exception or else gives it back. Gives the list of calls."""
    calls: list[Call] = []

    def run(command: list[str], **kwargs: Any) -> object:
        calls.append((command, kwargs))
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    monkeypatch.setattr(subprocess, "run", run)
    return calls


def refusal(path: Path) -> str:
    """Run probe_video on `path`, and give back the message of the PipelineError it raises."""
    with pytest.raises(PipelineError) as caught:
        probe_video(path, TIMEOUT)
    return str(caught.value)


def test_probe_video_runs_ffprobe_as_a_list_ending_with_the_absolute_path(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "-clip.mp4").write_bytes(b"")
    monkeypatch.chdir(tmp_path)  # so the relative path below has to be made absolute
    calls = replace_run(monkeypatch, finished(stdout=printed(output([stream("video", "4.0")]))))
    result = probe_video(Path("-clip.mp4"), TIMEOUT)
    [(command, kwargs)] = calls
    assert isinstance(command, list)
    assert command[:7] == ["ffprobe", "-v", "error", "-print_format", "json", "-show_format",
                           "-show_streams"]
    assert command[-1] == "file:" + str((tmp_path / "-clip.mp4").resolve())
    assert kwargs["timeout"] == TIMEOUT
    assert result == ProbeResult(format_name=FAMILY, has_video=True, video_seconds=4.0)


# Each failure, as what the stand-in subprocess.run raises or gives back, and the words its
# refusal must hold.
FAILURES = [
    pytest.param(FileNotFoundError(2, "No such file or directory"),
                 "ffprobe isn't installed. Install ffmpeg", id="ffprobe missing"),
    pytest.param(subprocess.TimeoutExpired(["ffprobe"], TIMEOUT), "12.5 seconds", id="timeout"),
    pytest.param(finished(stdout=b"this is not JSON"), UNREADABLE, id="not JSON"),
    pytest.param(finished(stdout=printed([])), UNREADABLE, id="not an object"),
    pytest.param(finished(stdout=printed({"format": {"format_name": 7}, "streams": []})),
                 UNREADABLE, id="format name not text"),
    pytest.param(finished(stdout=printed({"format": {"format_name": FAMILY}})), UNREADABLE,
                 id="no streams"),
]


@pytest.mark.parametrize(("outcome", "words"), FAILURES)
def test_each_ffprobe_failure_gives_its_own_readable_error(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, outcome: object, words: str) -> None:
    replace_run(monkeypatch, outcome)
    assert words in refusal(tmp_path / "-clip.mp4")


def test_an_exit_of_1_gives_the_first_stderr_line_even_with_bytes_that_arent_utf8(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # b"\xe9" is é in Windows-1252 but isn't UTF-8 here, so it's read as U+FFFD.
    stderr = (b"file:caf\xe9.mp4: Invalid data found when processing input\n"
              b"Error opening input files: Invalid data found when processing input\n")
    replace_run(monkeypatch, finished(returncode=1, stderr=stderr))
    message = refusal(tmp_path / "-clip.mp4")
    assert "file:caf\ufffd.mp4: Invalid data found when processing input" in message
    assert "Error opening input files" not in message


def test_parse_probe_takes_the_length_from_the_video_stream() -> None:
    result = parse_probe(output([stream("video", "4.5")], duration="9.0"))
    assert result == ProbeResult(format_name=FAMILY, has_video=True, video_seconds=4.5)


@pytest.mark.parametrize("duration", ["N/A", None], ids=["N/A", "no duration"])
def test_parse_probe_falls_back_on_the_containers_length(duration: str | None) -> None:
    assert parse_probe(output([stream("video", duration)], duration="9.0")).video_seconds == 9.0


@pytest.mark.parametrize("duration", ["nan", "inf", "0", "-1", None],
                         ids=["nan", "inf", "0", "-1", "no duration"])
def test_parse_probe_gives_no_length_when_neither_is_a_finite_number_above_0(
        duration: str | None) -> None:
    assert parse_probe(output([stream("video", duration)], duration=duration)).video_seconds is None


def test_parse_probe_uses_the_first_video_stream() -> None:
    streams = [stream("audio", "7.0"), stream("video", "4.5"), stream("video", "6.0")]
    assert parse_probe(output(streams, duration="9.0")).video_seconds == 4.5


def test_parse_probe_doesnt_count_cover_art_as_video() -> None:
    cover_art = stream("video", "0.04", attached_pic=1)
    assert not parse_probe(output([cover_art])).has_video
    result = parse_probe(output([cover_art, stream("video", "4.5")]))
    assert (result.has_video, result.video_seconds) == (True, 4.5)


# The real ffprobe tests (Req 14.4, 14.5)

# Skips a test that runs the real ffmpeg and ffprobe when either isn't installed.
needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg and ffprobe aren't installed")


def make_clip(folder: Path) -> Path:
    """Make a 4-second test video, `folder / "-clip.mp4"`, with the real ffmpeg, and give its
    path. The name starts with - on purpose. The codec is mpeg4 because every ffmpeg build has
    it."""
    path = folder / "-clip.mp4"
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi",
                    "-i", "testsrc=duration=4:size=64x48:rate=5", "-c:v", "mpeg4",
                    "-pix_fmt", "yuv420p", "file:" + str(path.resolve())],
                   check=True, timeout=60)
    return path


@needs_ffmpeg
def test_probe_video_reads_a_real_clip_made_with_ffmpeg(tmp_path: Path) -> None:
    result = probe_video(make_clip(tmp_path), load_config().jobs().ffprobe_timeout_seconds)
    assert (result.format_name, result.has_video) == (CONTAINER_FAMILY, True)
    assert result.video_seconds == pytest.approx(4, abs=0.25)


@needs_ffmpeg
def test_make_job_makes_a_job_from_a_real_clip_with_the_real_probe(
        tmp_path: Path, temp_config: Path) -> None:
    clip = make_clip(tmp_path)
    # The test config's size limit is tiny, so only that is raised, above the clip's size. The
    # config is built from the test config, never the real config.toml, so the Jobs_Folder stays
    # in tmp_path.
    data = tomllib.loads(temp_config.read_text(encoding="utf-8"))
    data["jobs"]["max_video_bytes"] = clip.stat().st_size + 1
    config = Config(data)
    job = open_job(make_job(clip, config=config), config)
    assert job.folder.parent == (tmp_path / "jobs").resolve()
    assert job.record.video_seconds == pytest.approx(4, abs=0.25)
    assert job.video_path.read_bytes() == clip.read_bytes()
