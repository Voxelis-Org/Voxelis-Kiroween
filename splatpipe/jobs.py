"""Job ids, Job_Folders and the Job_Record.

`new_job_id` makes a Job_Id from the UTC time and a random part. `check_job_id` accepts a
string exactly when the whole of it matches JOB_ID, and `job_folder` then refuses a Job_Folder
whose resolved path isn't inside the resolved Jobs_Folder, so a link can't lead out of it.
`open_job` finds a Job and checks its `input/job.json`, which `write_record` writes.
`remove_tree` removes a folder, read-only files included. Every refusal is a PipelineError with
a Readable_Error, except the ValueError for a naive time, which is a bug in the caller.

From splatpipe, this module imports only splatpipe.errors, splatpipe.config and splatpipe.jsonio.
"""

from __future__ import annotations

import os
import re
import secrets
import shutil
import stat
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, NoReturn

from splatpipe.config import Config, check_preset
from splatpipe.errors import PipelineError
from splatpipe.jsonio import read_json, write_json

# The Job_Id pattern stays in code, not in config.toml, because it's a safety check (Req 1.6).
# Only explicit ASCII classes, checked with fullmatch: \d matches non-ASCII digits, and $
# matches before a trailing line break.
JOB_ID = re.compile(r"[0-9]{8}-[0-9]{6}-[0-9a-f]{6}")

_VIDEO_FILES = ("video.mp4", "video.mov")  # the names the Job_Maker gives the copy in input/
_MAKE = "uv run python -m splatpipe.make <video>"  # for the messages


@dataclass(frozen=True)
class JobRecord:
    """The Job_Record, `input/job.json`. Each field is named after its key in the file."""

    job_id: str
    preset: str
    start_seconds: float  # the Used_Part's start
    end_seconds: float  # the Used_Part's end
    source_name: str  # the original file's name, without its folder
    source_bytes: int
    video_seconds: float  # the Video_Length
    video_file: str  # "video.mp4" or "video.mov", inside input/


@dataclass(frozen=True)
class Job:
    """An opened Job: its checked id, its resolved Job_Folder and its checked Job_Record."""

    id: str
    folder: Path  # resolved
    record: JobRecord

    @property
    def video_path(self) -> Path:
        """The copy of the video, in input/."""
        return self.folder / "input" / self.record.video_file


def new_job_id(now: datetime | None = None, random_part: str | None = None) -> str:
    """A Job_Id from `now` in UTC and `random_part`, like `20261119-142530-7f3a9c`.

    `now` defaults to the time now and `random_part` to secrets.token_hex(3). An aware `now`
    is converted to UTC, and a naive one is refused with ValueError. The fields are padded by
    hand, because strftime("%Y") doesn't zero-pad years before 1000 on glibc.
    """
    if now is None:
        now = datetime.now(timezone.utc)
    elif now.utcoffset() is None:
        raise ValueError("new_job_id needs an aware time, like datetime.now(timezone.utc), "
                         "but it got a naive one.")
    t = now.astimezone(timezone.utc)
    part = secrets.token_hex(3) if random_part is None else random_part
    return (f"{t.year:04d}{t.month:02d}{t.day:02d}-{t.hour:02d}{t.minute:02d}{t.second:02d}"
            f"-{part}")


def check_job_id(value: object) -> str:
    """Return `value` when it's a str and the whole of it matches JOB_ID. Refuse anything else.

    The refusal gives the expected form and leaves the value out, so a line break or a control
    character in it can't garble the terminal.
    """
    if isinstance(value, str) and JOB_ID.fullmatch(value) is not None:
        return value
    raise PipelineError("That isn't a job id. A job id has the form YYYYMMDD-HHMMSS-xxxxxx: "
                        "8 digits, a hyphen, 6 digits, a hyphen and 6 characters from 0-9 and "
                        "a-f. Use the id that splatpipe.make printed when it made the job.")


def job_folder(job_id: str, jobs_folder: Path) -> Path:
    """The resolved Job_Folder of `job_id` in `jobs_folder`, after check_job_id.

    Refused unless the resolved Jobs_Folder is among the resolved Job_Folder's parents.
    resolve() follows symlinks and junctions, so a Job_Folder that's a link to a folder
    elsewhere is refused. Windows paths compare without regard to case.
    """
    path = jobs_folder / check_job_id(job_id)
    folder = path.resolve()
    root = jobs_folder.resolve()
    if root not in folder.parents:
        raise PipelineError(f"The job folder {path} leads to {folder}, which is outside the "
                            f"jobs folder {root}. A job folder must be a real folder inside the "
                            "jobs folder, so replace the link with the folder it leads to.")
    return folder


def open_job(job_id: str, config: Config) -> Job:
    """Check `job_id`, then find its Job_Folder in config's Jobs_Folder and read its Job_Record.

    Refused with a PipelineError when job_folder refuses, the folder is missing, input/job.json
    is missing or can't be read as JSON, or a field is missing, has the wrong type or holds a
    bad value.
    """
    jobs_folder = config.jobs().jobs_folder
    folder = job_folder(job_id, jobs_folder)
    if not folder.is_dir():
        raise PipelineError(f"There's no job {job_id} in {jobs_folder}. Check the id, or make "
                            f"the job first with {_MAKE}.")
    path = folder / "input" / "job.json"
    return Job(id=job_id, folder=folder, record=_record(_read_record(path), job_id, path))


def write_record(folder: Path, record: JobRecord) -> None:
    """Write `record` to input/job.json in the Job_Folder `folder`, with write_json."""
    write_json(folder / "input" / "job.json", asdict(record))


def remove_tree(path: Path) -> None:
    """Remove the folder `path` and everything in it, read-only files included.

    rmtree removes links inside the tree without following them.
    """
    shutil.rmtree(path, onerror=_clear_read_only)  # Python 3.11 has onerror, not onexc


def _clear_read_only(function: Callable[[str | os.PathLike[str]], object],
                     path: str | os.PathLike[str], exc_info: object) -> None:
    """rmtree's error handler: clear the read-only flag, then retry the call once.

    If the retry fails too, its error goes up and stops rmtree.
    """
    os.chmod(path, stat.S_IWRITE)
    function(path)


# Reading and checking the Job_Record

def _read_record(path: Path) -> dict[str, Any]:
    """The Job_Record file's JSON object. Anything else is refused, saying what's wrong."""
    try:
        data = read_json(path)
    except FileNotFoundError as error:
        raise PipelineError(f"The job has no job record at {path}. Make the job again "
                            f"with {_MAKE}.") from error
    except OSError as error:
        raise PipelineError(f"The job record {path} can't be read ({error.strerror or error}). "
                            "Check that it's a file you can open.") from error
    except (ValueError, RecursionError) as error:  # not UTF-8, not JSON, NaN, nested too deeply
        detail = str(error).rstrip(".")
        raise PipelineError(f"The job record {path} can't be read as JSON ({detail}). Make the "
                            f"job again with {_MAKE}.") from error
    if not isinstance(data, dict):
        raise PipelineError(f"The job record {path} must hold a JSON object. Make the job again "
                            f"with {_MAKE}.")
    return data


def _record(data: dict[str, Any], job_id: str, path: Path) -> JobRecord:
    """The JobRecord in `data`, with every field checked. A refusal names the field."""
    if _field(data, "job_id", path) != job_id:
        _refuse_field(path, "job_id", f"this job's id, {job_id}")
    return JobRecord(
        job_id=job_id,
        preset=_preset(data, path),
        start_seconds=_seconds(data, "start_seconds", path),
        end_seconds=_seconds(data, "end_seconds", path),
        source_name=_text(data, "source_name", path),
        source_bytes=_byte_count(data, "source_bytes", path),
        video_seconds=_seconds(data, "video_seconds", path),
        video_file=_video_file(data, path),
    )


def _field(data: dict[str, Any], key: str, path: Path) -> Any:
    """The value of `key`, refused by name when it's missing."""
    if key not in data:
        raise PipelineError(f"The job record {path} has no {key}. Make the job again "
                            f"with {_MAKE}.")
    return data[key]


def _preset(data: dict[str, Any], path: Path) -> str:
    """preset, refused unless it's one of the Presets."""
    value = _field(data, "preset", path)
    try:
        return check_preset(value)
    except PipelineError as error:
        raise PipelineError(f"In the job record {path}, the preset is wrong: {error}") from error


def _seconds(data: dict[str, Any], key: str, path: Path) -> float:
    """A finite number of seconds, 0 or more: an int or a float, never a bool."""
    value = _field(data, key, path)
    # Finite and not negative. Python compares an int with a float exactly, so this never
    # overflows, and an int too big for a float is refused too.
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not 0 <= value <= sys.float_info.max):
        _refuse_field(path, key, "a number of seconds, 0 or more")
    return float(value)


def _byte_count(data: dict[str, Any], key: str, path: Path) -> int:
    """A whole number of bytes, 0 or more. A bool or a float is refused."""
    value = _field(data, key, path)
    if type(value) is not int or value < 0:
        _refuse_field(path, key, "a whole number of bytes, 0 or more")
    return value


def _text(data: dict[str, Any], key: str, path: Path) -> str:
    """A str."""
    value = _field(data, key, path)
    if not isinstance(value, str):
        _refuse_field(path, key, "text in quotes")
    return value


def _video_file(data: dict[str, Any], path: Path) -> str:
    """video_file, exactly video.mp4 or video.mov."""
    value = _field(data, "video_file", path)
    if value not in _VIDEO_FILES:
        _refuse_field(path, "video_file", " or ".join(_VIDEO_FILES))
    return value


def _refuse_field(path: Path, key: str, wanted: str) -> NoReturn:
    """Refuse a Job_Record field by name, saying what it must be."""
    raise PipelineError(f"In the job record {path}, {key} must be {wanted}. Fix it there, or "
                        f"make the job again with {_MAKE}.")
