"""The Frames_Contract, read and checked: the Frames_Record work/frames/frames.json and the files
it names, before any COLMAP_Command runs (Req 1.1, A11).

The names IMAGES, GREY, MASK, RECORD and pick_name come from splatpipe.frames.files, which defines
them as the contract between the two steps. So each Pick's Colour_Pick is images/<file>, its
Grey_Copy is grey/<file>, and its file is pick_name(number), like 000000.jpg. The Mask is
mask.png.

read_frames reads frames.json with splatpipe.jsonio.read_json. A missing file means the
Frames_Step hasn't finished for the Job (Req 1.2), and any other read error is the Req 1.3 failure
saying the file can't be read as JSON. parse_frames_record is pure. It checks only the fields the
step uses, and refuses each of these, naming the field (Req 1.3): a record that isn't an object;
pick_width or pick_height missing or not a whole number above 0; mask_used missing or not true or
false; picks missing, not a list, or empty; a Pick that isn't an object, whose number is missing or
isn't a whole number of 0 or more, or whose file is missing or isn't text; a file other than
pick_name(number); and a number given twice, which would give two Picks one file (Req 1.6). Then
read_frames checks that every images/<file> and grey/<file> is a file, and mask.png too when
mask_used is true, and the Req 1.4 failure names the first one that isn't.

Every refusal is an ExpectedFailure that says to run the Frames_Step, with
uv run python -m splatpipe.frames --job <id> and the Job's id.

From splatpipe, this module imports only splatpipe.errors, splatpipe.jsonio and
splatpipe.frames.files.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from splatpipe.errors import ExpectedFailure
from splatpipe.frames.files import GREY, IMAGES, MASK, RECORD, pick_name
from splatpipe.jsonio import read_json

_FRAMES = "uv run python -m splatpipe.frames --job"  # the Frames_Step's CLI, for the messages
# The two files each Pick has in work/frames/: their folder, and what the messages call them.
_PICK_FILES = ((IMAGES, "colour pick"), (GREY, "grey copy"))


@dataclass(frozen=True)
class Pick:
    """One Pick of the Frames_Record."""

    number: int  # its Pick_Number, its place in time order from 0
    file: str  # pick_name(number), like "000000.jpg", its name in both images/ and grey/


@dataclass(frozen=True)
class FramesInput:
    """The checked Frames_Contract: its folder, and the Frames_Record fields the step uses."""

    folder: Path  # work/frames/, absolute
    pick_width: int  # the Pick_Size in pixels, which the Grey_Copies and the Mask share
    pick_height: int
    mask_used: bool  # whether the Frames_Step uses the Mask, mask.png
    picks: tuple[Pick, ...]  # as the Frames_Record lists them

    @property
    def numbers(self) -> Mapping[str, int]:
        """Each Pick's file and its Pick_Number, so an image that COLMAP names finds its Pick
        (Req 1.6)."""
        return {pick.file: pick.number for pick in self.picks}


def parse_frames_record(data: object, record: Path,
                        job_id: str) -> tuple[int, int, bool, tuple[Pick, ...]]:
    """pick_width, pick_height, mask_used and the Picks of the Frames_Record `data`. Pure.

    `record` is the file that `data` came from, and `job_id` the Job's id, both for the messages.
    Each fault of the module's list is the Req 1.3 failure, which names the field, like
    picks[3].file. Nothing else in the record is checked.
    """
    check = _Check(record, job_id)
    if not isinstance(data, dict):
        raise check.refused(f"The frames record {record} must hold a JSON object.")
    width = _size(check, data, "pick_width")
    height = _size(check, data, "pick_height")
    mask_used = check.field(data, "mask_used", "mask_used")
    if type(mask_used) is not bool:
        raise check.wrong("mask_used", "true or false")
    entries = check.field(data, "picks", "picks")
    if not isinstance(entries, list) or not entries:
        raise check.wrong("picks", "a list with at least one pick")
    return width, height, mask_used, _picks(check, entries)


def read_frames(folder: Path, job_id: str) -> FramesInput:
    """The Frames_Contract of `folder`, the Job's work/frames/, checked (Req 1.1 to 1.4).

    `folder` is absolute, and `job_id` is the Job's id for the messages. The Frames_Record is
    folder/RECORD, checked with parse_frames_record. Then each Pick's Colour_Pick and Grey_Copy
    must be a file, Pick by Pick, and the Mask too when mask_used is true. The first one that
    isn't is the Req 1.4 failure, which names it.
    """
    record = folder / RECORD
    width, height, mask_used, picks = parse_frames_record(_read_record(record, job_id), record,
                                                          job_id)
    missing = _missing_file(folder, mask_used, picks)
    if missing is not None:
        raise _refused(missing, job_id)
    return FramesInput(folder=folder, pick_width=width, pick_height=height, mask_used=mask_used,
                       picks=picks)


class _Check:
    """The refusals for one Frames_Record, which name the file and end with the frames command."""

    def __init__(self, record: Path, job_id: str) -> None:
        self.record = record
        self.job_id = job_id

    def field(self, data: Mapping[str, Any], key: str, name: str) -> Any:
        """The value of `key` in `data`, refused when it's missing. `name` is the field's place in
        the record, like picks[3].number."""
        if key not in data:
            raise self.refused(f"The frames record {self.record} has no {name}.")
        return data[key]

    def wrong(self, name: str, wanted: str) -> ExpectedFailure:
        """The Req 1.3 failure for the field `name`, saying what it must be."""
        return self.refused(f"In the frames record {self.record}, {name} must be {wanted}.")

    def refused(self, problem: str) -> ExpectedFailure:
        """The Req 1.3 failure: `problem`, then the frames command to run."""
        return _refused(problem, self.job_id)


def _size(check: _Check, data: Mapping[str, Any], key: str) -> int:
    """pick_width or pick_height: a whole number above 0. type() is checked exactly, so a bool is
    refused."""
    value = check.field(data, key, key)
    if type(value) is not int or value <= 0:
        raise check.wrong(key, "a whole number above 0")
    return value


def _picks(check: _Check, entries: Sequence[object]) -> tuple[Pick, ...]:
    """The Picks of the entries of picks, in their order, refusing a number given twice."""
    picks: list[Pick] = []
    first: dict[int, int] = {}  # each number, and the index of the entry that gave it first
    for index, entry in enumerate(entries):
        name = f"picks[{index}]"
        pick = _pick(check, entry, name)
        if pick.number in first:
            raise check.refused(f"In the frames record {check.record}, {name}.number is "
                                f"{pick.number}, the same as picks[{first[pick.number]}].number, "
                                "but each pick must have its own number.")
        first[pick.number] = index
        picks.append(pick)
    return tuple(picks)


def _pick(check: _Check, entry: object, name: str) -> Pick:
    """The Pick of one entry of picks, whose place in the record is `name`, like picks[3]."""
    if not isinstance(entry, dict):
        raise check.wrong(name, "a JSON object")
    number = check.field(entry, "number", f"{name}.number")
    if type(number) is not int or number < 0:
        raise check.wrong(f"{name}.number", "a whole number, 0 or more")
    file = check.field(entry, "file", f"{name}.file")
    if not isinstance(file, str):
        raise check.wrong(f"{name}.file", "a file name in quotes")
    expected = pick_name(number)
    if file != expected:
        raise check.refused(f"In the frames record {check.record}, {name}.file is {file!r}, but "
                            f"it must be {expected!r}, the frames step's name for pick {number}.")
    return Pick(number=number, file=file)


def _read_record(record: Path, job_id: str) -> object:
    """The JSON value in the Frames_Record file `record`.

    A missing file is the Req 1.2 failure: the Frames_Step hasn't finished for the Job. Any other
    read error is the Req 1.3 failure, saying the file can't be read as JSON, and why.
    """
    try:
        return read_json(record)
    except FileNotFoundError as error:
        raise _refused(f"The frames step hasn't finished for the job {job_id}, because there's "
                       f"no frames record at {record}.", job_id, "first") from error
    except OSError as error:  # a folder in its place, or no permission to read it
        raise _unreadable(record, error.strerror or str(error), job_id) from error
    except (ValueError, RecursionError) as error:  # not UTF-8, not JSON, NaN, nested too deeply
        raise _unreadable(record, str(error).rstrip("."), job_id) from error


def _missing_file(folder: Path, mask_used: bool, picks: Sequence[Pick]) -> str | None:
    """The Req 1.4 problem for the first file of the Frames_Contract that isn't a file, or None.

    It checks each Pick's Colour_Pick, then its Grey_Copy, Pick by Pick, then the Mask when
    mask_used is true. os.path.isfile gives False for any OSError, where Python 3.11's
    Path.is_file raises some, like for a name too long for the system, which a broken record's
    huge number would give.
    """
    for pick in picks:
        for subfolder, kind in _PICK_FILES:
            path = folder / subfolder / pick.file
            if not os.path.isfile(path):
                return f"The {kind} {path} is missing."
    mask = folder / MASK
    if mask_used and not os.path.isfile(mask):
        return (f"The mask {mask} is missing, though the frames record says the frames step "
                "used one.")
    return None


def _unreadable(record: Path, detail: str, job_id: str) -> ExpectedFailure:
    """The Req 1.3 failure for a Frames_Record that can't be read as JSON, with the reason."""
    return _refused(f"The frames record {record} can't be read as JSON ({detail}).", job_id)


def _refused(problem: str, job_id: str, when: str = "again") -> ExpectedFailure:
    """A refusal: `problem`, then the advice to run the Frames_Step `when`, with its command and
    the Job's id."""
    return ExpectedFailure(f"{problem} Run the frames step {when} with {_FRAMES} {job_id}.")
