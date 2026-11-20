"""Tests for Job_Ids, Job_Folders and the Job_Record, splatpipe/jobs.py."""

import json
import math
import stat
from collections.abc import Callable
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from splatpipe.config import load_config
from splatpipe.errors import PipelineError
from splatpipe.jobs import (JOB_ID, JobRecord, check_job_id, new_job_id, open_job, remove_tree,
                            write_record)

ID = "20261119-142530-7f3a9c"

# A good Job_Record for ID: the design's example.
RECORD = JobRecord(job_id=ID, preset="fast", start_seconds=1.5, end_seconds=12.48,
                   source_name="Kitchen walk.MOV", source_bytes=48213775, video_seconds=12.48,
                   video_file="video.mov")


# Unit tests


def refusal(action: Callable[..., object], *args: object) -> str:
    """Call the action with the args, and give back the message of the PipelineError it raises."""
    with pytest.raises(PipelineError) as caught:
        action(*args)
    return str(caught.value)


def record_path(tmp_path: Path) -> Path:
    """Where the Job_Record of ID goes in the temporary Jobs_Folder, with its folders made."""
    path = tmp_path / "jobs" / ID / "input" / "job.json"
    path.parent.mkdir(parents=True)
    return path


def record_text(drop: str | None = None, **changes: object) -> bytes:
    """RECORD as JSON, without the field `drop` and with `changes`. json.dumps writes NaN as NaN."""
    data = {**asdict(RECORD), **changes}
    return json.dumps({key: value for key, value in data.items() if key != drop}).encode()


def test_load_config_with_no_path_gives_the_temporary_config(tmp_path: Path) -> None:
    jobs_folder = load_config().jobs().jobs_folder
    assert jobs_folder == tmp_path / "jobs"
    assert tmp_path in jobs_folder.parents


def test_the_job_id_pattern_is_kept_in_code() -> None:
    assert JOB_ID.pattern == "[0-9]{8}-[0-9]{6}-[0-9a-f]{6}"


def test_a_new_job_id_passes_the_check_and_holds_the_utc_time_now() -> None:
    job_id = new_job_id()
    assert check_job_id(job_id) == job_id
    made = datetime.strptime(job_id[:15], "%Y%m%d-%H%M%S").replace(tzinfo=timezone.utc)
    assert abs(datetime.now(timezone.utc) - made) < timedelta(seconds=5)


def test_a_naive_time_is_refused_with_a_value_error() -> None:
    with pytest.raises(ValueError):
        new_job_id(now=datetime(2026, 11, 19, 14, 25, 30))


# A path out of the Jobs_Folder, a trailing line break, upper case and a terminal escape code.
@pytest.mark.parametrize("bad", ["../" + ID, ID + "\n", ID.upper(), "\x1b[2J"],
                         ids=["path", "line break", "upper case", "escape code"])
def test_a_bad_job_id_is_refused_with_the_form_and_without_the_bad_id(bad: str) -> None:
    message = refusal(check_job_id, bad)
    assert "YYYYMMDD-HHMMSS-xxxxxx" in message
    assert bad not in message


def test_a_job_folder_linked_outside_the_jobs_folder_is_refused(
        tmp_path: Path, dir_link: Callable[[Path, Path], None]) -> None:
    outside = tmp_path / "outside"
    (outside / "input").mkdir(parents=True)
    write_record(outside, RECORD)  # a good Job behind the link, so only the link check can refuse
    (tmp_path / "jobs").mkdir()
    dir_link(tmp_path / "jobs" / ID, outside)
    assert "outside the jobs folder" in refusal(open_job, ID, load_config())


def test_write_record_then_open_job_gives_back_the_record(tmp_path: Path) -> None:
    folder = record_path(tmp_path).parents[1]
    write_record(folder, RECORD)
    job = open_job(ID, load_config())
    assert (job.id, job.folder, job.record) == (ID, folder.resolve(), RECORD)
    assert job.video_path == folder.resolve() / "input" / "video.mov"


def test_open_job_refuses_a_job_with_no_folder() -> None:
    assert f"There's no job {ID}" in refusal(open_job, ID, load_config())


# Each refused job.json, None for no file, and the words its refusal must hold.
BAD_RECORDS = [
    pytest.param(None, "has no job record", id="no file"),
    pytest.param(b"{", "can't be read as JSON", id="not JSON"),
    pytest.param(b"[]", "must hold a JSON object", id="not an object"),
    pytest.param(record_text(start_seconds=math.nan), "holds NaN", id="NaN"),
    *[pytest.param(record_text(drop=key), f"has no {key}", id=f"no {key}")
      for key in asdict(RECORD)],
    pytest.param(record_text(preset="slow"), "the preset is wrong", id="not a preset"),
    pytest.param(record_text(start_seconds="1.5"), "start_seconds must be", id="seconds as text"),
    pytest.param(record_text(end_seconds=-1), "end_seconds must be", id="negative seconds"),
    pytest.param(record_text(video_seconds=True), "video_seconds must be", id="seconds as a bool"),
    pytest.param(record_text(source_bytes=1.0), "source_bytes must be", id="bytes as a float"),
    pytest.param(record_text(source_bytes=-1), "source_bytes must be", id="negative bytes"),
    pytest.param(record_text(source_name=None), "source_name must be", id="name as null"),
    *[pytest.param(record_text(video_file=name), "video_file must be video.mp4 or video.mov",
                   id=f"video_file {name}") for name in ["video.mkv", "VIDEO.MP4", "../video.mp4"]],
    pytest.param(record_text(job_id="20261119-142530-000000"), "job_id must be this job's id",
                 id="another job's id"),
]


@pytest.mark.parametrize(("content", "words"), BAD_RECORDS)
def test_open_job_refuses_a_bad_job_record(
        tmp_path: Path, content: bytes | None, words: str) -> None:
    path = record_path(tmp_path)
    if content is not None:
        path.write_bytes(content)
    assert words in refusal(open_job, ID, load_config())


def test_remove_tree_removes_a_tree_with_a_read_only_file_in_it(tmp_path: Path) -> None:
    tree = tmp_path / "tree"
    (tree / "sub").mkdir(parents=True)
    locked = tree / "sub" / "locked.txt"
    locked.write_bytes(b"x")
    locked.chmod(stat.S_IREAD)
    remove_tree(tree)
    assert not tree.exists()


# Property tests

DIGITS, HEX = "0123456789", "0123456789abcdef"
valid_ids = st.from_regex(r"[0-9]{8}-[0-9]{6}-[0-9a-f]{6}", fullmatch=True)
# The last four are an Arabic-Indic 3, a fullwidth 0, a superscript 2 and a zero-width space.
tricky = st.sampled_from(["/", "\\", "..", "\n", "\r", "\x00", " ", "A", "F", "g", "-",
                          "\u0663", "\uff10", "\u00b2", "\u200b"])
# st.text() can't give lone surrogates in Hypothesis 6.170 (checked); this one can, like "\ud800".
any_text = st.text() | st.text(st.characters(codec=None, exclude_categories=()))


@st.composite
def near_misses(draw: st.DrawFn) -> str:
    """A valid id with one character inserted, replaced or deleted at a random position. What's
    inserted, or put in the replaced character's place, comes from tricky."""
    job_id = draw(valid_ids)
    edit = draw(st.sampled_from(["insert", "replace", "delete"]))
    if edit == "insert":
        at = draw(st.integers(min_value=0, max_value=len(job_id)))
        return job_id[:at] + draw(tricky) + job_id[at:]
    at = draw(st.integers(min_value=0, max_value=len(job_id) - 1))
    if edit == "replace":
        return job_id[:at] + draw(tricky) + job_id[at + 1:]
    return job_id[:at] + job_id[at + 1:]


# Any text, valid ids, near misses, and valid ids with a trailing line break, ../ in front,
# backslashes for the hyphens or upper case.
job_id_candidates = st.one_of(
    st.text(), any_text, valid_ids, near_misses(),
    valid_ids.map(lambda s: s + "\n"),
    valid_ids.map(lambda s: "../" + s),
    valid_ids.map(lambda s: s.replace("-", "\\")),
    valid_ids.map(str.upper),
)


def is_job_id(text: str) -> bool:
    """The oracle, written without regular expressions so it can't share a bug with JOB_ID:
    22 characters, `-` at positions 8 and 15, ASCII digits at 0 to 7 and 9 to 14, and characters
    from HEX at 16 to 21."""
    return (len(text) == 22
            and text[8] == text[15] == "-"
            and all(text[i] in DIGITS for i in [*range(0, 8), *range(9, 15)])
            and all(text[i] in HEX for i in range(16, 22)))


@given(job_id_candidates)
def test_the_job_id_check_accepts_exactly_the_pattern(text: str) -> None:
    """Feature: pipeline-core, Property 1: The Job_Id_Check accepts exactly the pattern

    **Validates: Requirements 1.2, 1.3, 1.7**
    """
    if is_job_id(text):
        assert check_job_id(text) == text
    else:
        with pytest.raises(PipelineError):
            check_job_id(text)


# Years 1 to 9999, so a year before 1000 catches an unpadded %Y.
utc_times = st.datetimes(timezones=st.just(timezone.utc))
# 3 random bytes in hex, the same shape as secrets.token_hex(3).
random_parts = st.binary(min_size=3, max_size=3).map(bytes.hex)


@given(utc_times, random_parts)
def test_every_job_id_the_job_maker_makes_passes_the_check(t: datetime, random_part: str) -> None:
    """Feature: pipeline-core, Property 2: Every Job_Id the Job_Maker makes passes the check

    **Validates: Requirements 1.1, 1.8**
    """
    job_id = new_job_id(now=t, random_part=random_part)
    assert check_job_id(job_id) == job_id
    # The fields at their fixed places: YYYY MM DD - HH MM SS - xxxxxx.
    fields = [job_id[0:4], job_id[4:6], job_id[6:8], job_id[9:11], job_id[11:13], job_id[13:15]]
    assert [int(field) for field in fields] == [t.year, t.month, t.day, t.hour, t.minute, t.second]
    assert job_id[16:] == random_part
