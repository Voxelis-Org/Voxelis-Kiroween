"""The Job_Maker: it checks a video and makes a new Job from it.

`make_job` finishes every check before it makes a folder or copies anything (Req 2.1): the
Job_Id and a free Job_Folder, the Preset, the file, the extension, the size, the Video_Probe and
the Trim. Then it makes the Job_Folder with input/, work/ and out/, copies the video into input/
and writes the Job_Record. Once the Job_Folder exists, any failure removes it again. `used_part`
is the Trim, a pure function. `main` is the CLI, `uv run python -m splatpipe.make <video>
[--start S] [--end E] [--preset P]`, which prints the new Job_Id alone on stdout. Every refusal
is a PipelineError with a Readable_Error.

From splatpipe, this module imports only splatpipe.errors, splatpipe.config, splatpipe.jobs and
splatpipe.probe.
"""

from __future__ import annotations

import argparse
import math
import shutil
import sys
from collections.abc import Callable
from pathlib import Path

from splatpipe.config import PRESETS, Config, check_preset, load_config
from splatpipe.errors import PipelineError, run_cli
from splatpipe.jobs import JobRecord, job_folder, new_job_id, remove_tree, write_record
from splatpipe.probe import ProbeResult, probe_video

# In code on purpose: these define what the Pipeline accepts. They aren't tuning numbers.
VIDEO_EXTENSIONS = (".mp4", ".mov")
CONTAINER_FAMILY = "mov,mp4,m4a,3gp,3g2,mj2"

_FOLDERS = ("input", "work", "out")  # the folders inside a Job_Folder


def used_part(video_seconds: float, start: float | None, end: float | None,
              minimum: float) -> tuple[float, float]:
    """The Trim: the Used_Part's start and end, from the Video_Length and an optional start and end.

    A missing start is 0.0 and a missing end is `video_seconds`. A given value comes back
    unchanged, never moved into the video. These are refused with a PipelineError, in this order:
    a given value that's NaN, infinite or negative (Req 3.4), a given value after `video_seconds`
    (Req 3.5), a start that isn't before the end (Req 3.6), and a Used_Part shorter than
    `minimum`, also when no Trim was given (Req 3.7). Values reach the subtraction only once
    they're between 0 and `video_seconds`, so it can't overflow.
    """
    given = [(name, value) for name, value in (("start", start), ("end", end)) if value is not None]
    for name, value in given:
        if not math.isfinite(value) or value < 0:
            raise PipelineError(f"The trim {name} is {value:g}, but it must be a number of "
                                f"seconds, 0 or more. Make the {name} 0 or more, or leave it out.")
    for name, value in given:
        if value > video_seconds:
            raise PipelineError(f"The trim {name}, {value:g} s, is after the end of the video, "
                                f"which is {video_seconds:g} s long. Make the {name} "
                                f"{video_seconds:g} s or less.")
    first = 0.0 if start is None else start
    last = video_seconds if end is None else end
    if not first < last:
        raise PipelineError(f"The trim start, {first:g} s, isn't before the trim end, {last:g} s. "
                            "Give a start that comes before the end.")
    if last - first < minimum:
        raise PipelineError(f"The part of the video to use, from {first:g} s to {last:g} s, is "
                            f"shorter than {minimum:g} s, the shortest part a job may use. Use a "
                            f"longer video, or a start and an end at least {minimum:g} s apart.")
    return first, last


def make_job(video: str | Path, *, start: float | None = None, end: float | None = None,
             preset: str | None = None, config: Config | None = None,
             probe: Callable[[Path, float], ProbeResult] = probe_video,
             new_id: Callable[[], str] = new_job_id) -> str:
    """Check the video at `video`, make a Job from it and return the new Job_Id.

    `start` and `end` are the Trim in seconds. `preset` defaults to [jobs].default_preset and
    `config` to load_config(). Tests pass a stand-in `probe` for the Video_Probe and a fixed
    `new_id`. Every check runs, in the design's order, before anything is made or copied.
    """
    settings = (load_config() if config is None else config).jobs()
    job_id = new_id()
    folder = _free_folder(job_id, settings.jobs_folder)
    chosen = check_preset(settings.default_preset if preset is None else preset)
    path = Path(video)
    size = _checked_size(path, settings.max_video_bytes)
    source = path.resolve()
    length = _checked_length(probe(source, settings.ffprobe_timeout_seconds), path.name,
                             settings.max_video_seconds)
    first, last = used_part(length, _trim_value("start", start), _trim_value("end", end),
                            settings.min_used_seconds)
    record = JobRecord(job_id=job_id, preset=chosen, start_seconds=first, end_seconds=last,
                       source_name=path.name, source_bytes=size, video_seconds=length,
                       video_file="video" + path.suffix.lower())
    _build(folder, settings.jobs_folder, source, record)
    return job_id


def main(argv: list[str] | None = None) -> int:
    """The Job_Maker CLI: make the Job, print its Job_Id alone on stdout and return 0.

    A refusal prints its Readable_Error on stderr and returns 1, through run_cli. A bad command
    line, like one with --job, which this CLI doesn't have, is argparse's usage error: exit
    code 2, and no Job is made (Req 4.10).
    """
    def body() -> int:
        args = _parser().parse_args(argv)
        print(make_job(args.video, start=args.start, end=args.end, preset=args.preset))
        return 0

    return run_cli(body)


# The checks

def _free_folder(job_id: str, jobs_folder: Path) -> Path:
    """The new Job's resolved Job_Folder, refused when something is already there (Req 4.7)."""
    folder = job_folder(job_id, jobs_folder)
    if folder.exists():
        raise _taken(folder)
    return folder


def _checked_size(path: Path, limit: int) -> int:
    """The video's size in bytes, once it's a file with an accepted extension, within `limit`."""
    if not path.is_file():
        raise PipelineError(f"There's no video file at {path}. Check the path, and give the "
                            "video's file, not a folder.")
    if path.suffix.lower() not in VIDEO_EXTENSIONS:
        raise PipelineError(f"Only MP4 and MOV videos are accepted, and {path.name} isn't one. "
                            "Convert the video to MP4 or MOV, then try again.")
    size = path.stat().st_size
    if size > limit:
        raise PipelineError(f"{path.name} is bigger than {limit / 2**30:g} GB, the largest video "
                            "accepted. Use a smaller video, like a shorter clip of it.")
    return size


def _checked_length(result: ProbeResult, name: str, limit: float) -> float:
    """The Video_Length from the Video_Probe's result, once the format, stream and length pass."""
    if result.format_name != CONTAINER_FAMILY:
        raise PipelineError(f"Only MP4 and MOV videos are accepted, but ffprobe reads {name} as "
                            f"the format {result.format_name!r}. Convert the video to MP4 or MOV, "
                            "then try again.")
    if not result.has_video:
        raise PipelineError(f"ffprobe found no video stream in {name}. Use a file that holds a "
                            "video, not only sound.")
    if result.video_seconds is None:
        raise PipelineError(f"ffprobe couldn't tell how long {name} is. Check that the video plays "
                            "to the end, or save it again as MP4, then try again.")
    if result.video_seconds > limit:
        raise PipelineError(f"{name} is {result.video_seconds:g} s long, but the longest video "
                            f"accepted is {limit:g} s. Cut it to {limit:g} s or less, then try "
                            "again.")
    return result.video_seconds


def _trim_value(name: str, value: object) -> float | None:
    """A Trim value as a float, or None when it isn't given. Only an int or a float, not a bool."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PipelineError(f"The trim {name} must be a number of seconds, but it's {value!r}. "
                            "Give a number, or leave it out.")
    try:
        return float(value)
    except OverflowError as error:  # an int too big for a float
        raise PipelineError(f"The trim {name} is too big to be a number of seconds. Give a time "
                            "inside the video, or leave it out.") from error


def _taken(folder: Path) -> PipelineError:
    """The refusal for a Job_Folder that already exists. It isn't ours, so it stays as it is."""
    return PipelineError(f"The job folder {folder} already exists, so it was left as it is and "
                         "no job was made. Try again: each try makes a new job id.")


# Making the Job

def _build(folder: Path, jobs_folder: Path, source: Path, record: JobRecord) -> None:
    """Make the Job_Folder, then fill it. Once it exists, any failure removes it (Req 4.8)."""
    jobs_folder.mkdir(parents=True, exist_ok=True)
    try:
        folder.mkdir(exist_ok=False)
    except FileExistsError as error:  # made since the check, so it isn't ours to remove
        raise _taken(folder) from error
    try:
        _fill(folder, source, record)
    except BaseException as error:  # Ctrl+C included, so no half-made Job is left behind
        remove_tree(folder)
        if isinstance(error, OSError):
            raise PipelineError(f"The video couldn't be copied into a new job "
                                f"({error.strerror or error}), so no job was made. Check that "
                                f"there's free space for it in {folder.parent}, then try "
                                "again.") from error
        raise


def _fill(folder: Path, source: Path, record: JobRecord) -> None:
    """Make input/, work/ and out/, copy the video into input/ and write the Job_Record."""
    for name in _FOLDERS:
        (folder / name).mkdir()
    shutil.copyfile(source, folder / "input" / record.video_file)
    write_record(folder, record)


def _parser() -> argparse.ArgumentParser:
    """The command line: a video, and an optional --start, --end and --preset. There's no --job."""
    parser = argparse.ArgumentParser(
        prog="uv run python -m splatpipe.make", allow_abbrev=False,
        description="Check a phone video and make a new job from it. Prints the new job id.")
    parser.add_argument("video", help="the MP4 or MOV video to make the job from")
    parser.add_argument("--start", type=float,
                        help="where the part of the video to use starts, in seconds (default: 0)")
    parser.add_argument("--end", type=float,
                        help="where the part of the video to use ends, in seconds "
                             "(default: the end of the video)")
    parser.add_argument("--preset", choices=PRESETS,
                        help="the preset (default: default_preset under [jobs] in config.toml)")
    return parser


if __name__ == "__main__":
    sys.exit(main())
