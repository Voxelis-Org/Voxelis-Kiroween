"""Tests for splatpipe/cameras/frames_input.py: the Frames_Contract, which the Cameras_Step checks
before any COLMAP_Command runs (Req 1.1).

Each test makes its Job with cameras_job from tests/conftest.py, which writes a tiny valid
work/frames/ as the Frames_Step leaves it, with 8 Picks unless the test asks for other. Then it
breaks one thing there and calls read_frames as the step does, with the Job's work/frames/,
absolute, and the Job's id.

- No frames.json is the Req 1.2 failure: the Frames_Step hasn't finished for the Job.
- Each Req 1.3 fault of the Frames_Record is refused, naming its field.
- Each missing file of Req 1.4 is refused, naming the file. A missing mask.png passes when the
  record's mask_used is false.
- Every refusal says to run the Frames_Step, with its command and the Job's id, and holds no
  traceback (Req 16.9).
- A good record gives its Picks in its own order, and numbers gives each file's Pick_Number
  (Req 1.6).

A field's name is looked for in the message with the frames folder's path taken out, because
that path is inside tmp_path, whose folder name holds the start of the test's name.
"""

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pytest

from splatpipe.cameras.frames_input import FramesInput, Pick, read_frames
from splatpipe.config import load_config
from splatpipe.errors import ExpectedFailure
from splatpipe.jobs import open_job

# The Frames_Step's command, which every refusal gives with the Job's id.
FRAMES_COMMAND = "uv run python -m splatpipe.frames --job"

# A change to the Frames_Record, made in place on the JSON value read from frames.json.
Change = Callable[[dict[str, Any]], None]


def frames_folder(job_id: str) -> Path:
    """The Job's work/frames/, absolute, as the Cameras_Step reads it."""
    return open_job(job_id, load_config()).folder / "work" / "frames"


def change_record(folder: Path, change: Change) -> None:
    """Read frames.json in `folder`, make `change` to it, and write it back as JSON."""
    record = folder / "frames.json"
    data = json.loads(record.read_text(encoding="utf-8"))
    change(data)
    record.write_text(json.dumps(data), encoding="utf-8")


def place(data: Any, path: Sequence[str | int]) -> Any:
    """The value in `data` that the keys and indexes of `path` lead to."""
    for step in path:
        data = data[step]
    return data


def drop(*path: str | int) -> Change:
    """A change that leaves out the key at the end of `path`, like drop("picks", 3, "number")."""
    *parents, key = path

    def change(data: dict[str, Any]) -> None:
        del place(data, parents)[key]

    return change


def put(value: object, *path: str | int) -> Change:
    """A change that sets what's at the end of `path` to `value`, like put(True, "pick_width")."""
    *parents, key = path

    def change(data: dict[str, Any]) -> None:
        place(data, parents)[key] = value

    return change


def refusal(folder: Path, job_id: str) -> str:
    """The Readable_Error of read_frames on `folder`, the Job `job_id`'s work/frames/."""
    with pytest.raises(ExpectedFailure) as caught:
        read_frames(folder, job_id)
    return str(caught.value)


def assert_says_to_run_the_frames_step(message: str, job_id: str, when: str = "again") -> None:
    """Check that a refusal says to run the Frames_Step `when`, with its command and the Job's
    id, and holds no traceback."""
    assert f"Run the frames step {when} with {FRAMES_COMMAND} {job_id}" in message
    assert "Traceback" not in message


# No Frames_Record (Req 1.2)

def test_no_frames_record_fails_saying_the_frames_step_hasnt_finished_for_the_job(
        cameras_job: Callable[..., str]) -> None:
    job_id = cameras_job()
    folder = frames_folder(job_id)
    (folder / "frames.json").unlink()
    message = refusal(folder, job_id)
    assert f"The frames step hasn't finished for the job {job_id}" in message
    assert_says_to_run_the_frames_step(message, job_id, when="first")


# The Frames_Record's faults (Req 1.3)

def test_a_frames_record_that_isnt_json_fails_naming_it(cameras_job: Callable[..., str]) -> None:
    job_id = cameras_job()
    folder = frames_folder(job_id)
    record = folder / "frames.json"
    record.write_text('{"pick_width": 16, ', encoding="utf-8")  # cut short
    message = refusal(folder, job_id)
    assert str(record) in message
    assert "can't be read as JSON" in message
    assert_says_to_run_the_frames_step(message, job_id)


# Each fault, made on cameras_job's record of 8 Picks, and the fields its refusal must name.
RECORD_FAULTS = [
    pytest.param(drop("pick_width"), ("pick_width",), id="no pick_width"),
    pytest.param(drop("pick_height"), ("pick_height",), id="no pick_height"),
    pytest.param(drop("mask_used"), ("mask_used",), id="no mask_used"),
    pytest.param(drop("picks"), ("picks",), id="no picks"),
    # True equals 1, which is a size above 0 and Pick 1's number, so only its kind is wrong.
    pytest.param(put(True, "pick_width"), ("pick_width",), id="a bool for pick_width"),
    pytest.param(put(True, "pick_height"), ("pick_height",), id="a bool for pick_height"),
    pytest.param(put(True, "picks", 1, "number"), ("picks[1].number",), id="a bool for a number"),
    pytest.param(put([], "picks"), ("picks",), id="empty picks"),
    pytest.param(drop("picks", 3, "number"), ("picks[3].number",), id="a pick with no number"),
    pytest.param(drop("picks", 3, "file"), ("picks[3].file",), id="a pick with no file"),
    # Pick 3's name, a file that exists, but not the name of Pick 2.
    pytest.param(put("000003.jpg", "picks", 2, "file"), ("picks[2].file",),
                 id="another pick's file"),
    # Pick 4 again, whose file matches its number, so only the number given twice is wrong.
    pytest.param(put({"number": 4, "file": "000004.jpg"}, "picks", 5),
                 ("picks[5].number", "picks[4].number"), id="a number given twice"),
]


@pytest.mark.parametrize(("change", "fields"), RECORD_FAULTS)
def test_each_fault_in_the_frames_record_fails_naming_its_field(
        cameras_job: Callable[..., str], change: Change, fields: tuple[str, ...]) -> None:
    job_id = cameras_job()
    folder = frames_folder(job_id)
    change_record(folder, change)
    message = refusal(folder, job_id)
    problem = message.replace(str(folder), "")
    for field in fields:
        assert field in problem
    assert_says_to_run_the_frames_step(message, job_id)


# The Frames_Contract's files (Req 1.4)

@pytest.mark.parametrize(("mask_used", "missing"), [
    pytest.param(False, ("images", "000003.jpg"), id="a colour pick"),
    pytest.param(False, ("grey", "000003.jpg"), id="a grey copy"),
    pytest.param(True, ("mask.png",), id="the mask, with mask_used true"),
])
def test_a_missing_file_of_the_frames_contract_fails_naming_it(
        cameras_job: Callable[..., str], mask_used: bool, missing: tuple[str, ...]) -> None:
    job_id = cameras_job(mask_used=mask_used)
    folder = frames_folder(job_id)
    path = folder.joinpath(*missing)
    path.unlink()
    message = refusal(folder, job_id)
    assert str(path) in message
    assert "missing" in message.replace(str(folder), "")
    assert_says_to_run_the_frames_step(message, job_id)


def test_a_missing_mask_passes_when_mask_used_is_false(cameras_job: Callable[..., str]) -> None:
    job_id = cameras_job(mask_used=False)
    folder = frames_folder(job_id)
    assert not (folder / "mask.png").exists()
    assert read_frames(folder, job_id).mask_used is False


# A good record (Req 1.6)

def test_a_good_record_gives_its_picks_in_its_order_and_each_files_pick_number(
        cameras_job: Callable[..., str]) -> None:
    job_id = cameras_job(picks=5, mask_used=True, width=20, height=10)
    folder = frames_folder(job_id)
    order = (3, 0, 4, 1, 2)  # not in number order, so a sort would show

    def reorder(data: dict[str, Any]) -> None:
        data["picks"] = [data["picks"][number] for number in order]  # entry k is Pick k

    change_record(folder, reorder)
    frames = read_frames(folder, job_id)
    assert frames == FramesInput(
        folder=folder, pick_width=20, pick_height=10, mask_used=True,
        picks=tuple(Pick(number=number, file=f"{number:06d}.jpg") for number in order))
    assert frames.numbers == {f"{number:06d}.jpg": number for number in range(5)}
