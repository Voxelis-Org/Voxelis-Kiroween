"""The Frames_Step's files in its Work_Folder, work/frames/: their names, the Picks' JPEGs, the
Mask's PNG and the Frames_Record.

The names are Fixed_Constants, because they're the contract with the Camera_Solve_Step (A10):

    images/NNNNNN.jpg   the Colour_Picks, at the Pick_Size
    grey/NNNNNN.jpg     the Grey_Copies, with the same names and size
    mask.png            the Mask, only when the step uses one
    frames.json         the Frames_Record, written last

NNNNNN is the Pick's number in Frame_Time order, from 000000, padded to 6 digits (Req 8.4). A
Grey_Copy is its Pick turned grey with global histogram equalisation (A9, Req 8.3).

cv2 only encodes images here and never reads or writes video. write_jpeg and write_png encode
with cv2.imencode and write the bytes with Path.write_bytes, so a full disk is an OSError with its
reason. An OSError, or an encode that cv2.imencode reports as failed, raises an ExpectedFailure
that names the file and the reason, like "No space left on device" (Req 8.10). write_record
writes the Frames_Record with splatpipe.jsonio.write_json and turns an OSError into the same
failure.

frames_record is pure. It gives the Frames_Record (Req 2.7, 3.6, 4.10, 7.9, 8.6) with only plain
int, float, bool, str, None, list and dict values, turning numpy values into Python ones, so
check_json_value never refuses one. Floats keep full precision, because the camera solve may
match on a Pick's seconds.

From splatpipe, this module imports only splatpipe.errors and splatpipe.jsonio when it runs. The
Kept, MaskChoice and Scores types that frames_record reads are imported for type hints only.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

from splatpipe.errors import ExpectedFailure
from splatpipe.jsonio import write_json

if TYPE_CHECKING:  # the types frames_record reads, named in its type hints only
    from splatpipe.frames.choose import Kept
    from splatpipe.frames.mask import MaskChoice
    from splatpipe.frames.scores import Scores

# The Work_Folder's names, Fixed_Constants because they're the contract with the camera solve (A10).
IMAGES = "images"  # the folder of Colour_Picks
GREY = "grey"  # the folder of Grey_Copies, with the Colour_Picks' names
RECORD = "frames.json"  # the Frames_Record, written last
MASK = "mask.png"  # the Mask, written only when the step uses it


def pick_name(number: int) -> str:
    """The file name of Pick `number`, counted from 0 in Frame_Time order, like 000123.jpg."""
    return f"{number:06d}.jpg"


def grey_copy(bgr: np.ndarray) -> np.ndarray:
    """The Grey_Copy of a BGR colour frame: the frame turned grey, then histogram-equalised (A9)."""
    return cv2.equalizeHist(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))


def write_jpeg(path: Path, image: np.ndarray, quality: int) -> None:
    """Write a BGR colour or grey uint8 image to `path` as a JPEG at `quality`, from 1 to 100."""
    ok, data = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    _write_encoded(path, ok, data, "JPEG")


def write_png(path: Path, image: np.ndarray) -> None:
    """Write a grey uint8 image, like the Mask, to `path` as a PNG."""
    ok, data = cv2.imencode(".png", image)
    _write_encoded(path, ok, data, "PNG")


def frames_record(*, colour_transfer: str | None, tone_mapped: bool, kept: Kept,
                  target_picks: int, jolt_filter_skipped: bool, pick_width: int,
                  pick_height: int, mask: MaskChoice, scores: Scores,
                  picks: Iterable[int]) -> dict[str, Any]:
    """The Frames_Record, as plain JSON values in the order of the design's table.

    `kept` is the Still_Trim's Kept_Range, `mask` the step's MaskChoice and `scores` the
    Score_Pass's Scores. `picks` holds the Picks' Frame_Numbers among the Used_Part's
    Score_Frames, in Frame_Time order, so Pick k is pick_name(k) in both images/ and grey/ and
    takes its Frame_Time and scores from `scores`. mask_file is MASK when the step uses the Mask,
    else None.
    """
    return {
        "colour_transfer": None if colour_transfer is None else str(colour_transfer),
        "tone_mapped": bool(tone_mapped),
        "still_start_seconds": float(kept.cut_start_seconds),
        "still_end_seconds": float(kept.cut_end_seconds),
        "kept_start_seconds": float(kept.start_seconds),
        "kept_end_seconds": float(kept.end_seconds),
        "target_picks": int(target_picks),
        "jolt_filter_skipped": bool(jolt_filter_skipped),
        "pick_width": int(pick_width),
        "pick_height": int(pick_height),
        "mask_coverage": float(mask.coverage),
        "mask_used": bool(mask.used),
        "mask_file": MASK if mask.used else None,
        "picks": [_pick_entry(number, int(frame), scores) for number, frame in enumerate(picks)],
    }


def write_record(path: Path, record: dict[str, Any]) -> None:
    """Write the Frames_Record to `path` with write_json, which checks every value first."""
    try:
        write_json(path, record)
    except OSError as error:
        raise _not_written(path, error) from error


def _pick_entry(number: int, frame: int, scores: Scores) -> dict[str, Any]:
    """Pick `number`'s entry: its file name, and its Score_Frame's number, Frame_Time and scores."""
    return {"number": number, "file": pick_name(number), "frame_number": frame,
            "seconds": float(scores.times[frame]),
            "sharpness": float(scores.sharpness[frame]),
            "motion": float(scores.motion[frame]),
            "contrast": float(scores.contrast[frame])}


def _write_encoded(path: Path, ok: bool, data: np.ndarray, kind: str) -> None:
    """Write the bytes cv2.imencode gave, or refuse when it reported a failed encode (Req 8.10)."""
    if not ok:
        raise ExpectedFailure(f"The file {path} couldn't be written (OpenCV couldn't encode the "
                              f"image as a {kind}). Run the step again, and tell the team if it "
                              "fails the same way.")
    try:
        path.write_bytes(data.tobytes())
    except OSError as error:
        raise _not_written(path, error) from error


def _not_written(path: Path, error: OSError) -> ExpectedFailure:
    """The Req 8.10 failure for an OSError, naming the file and the reason the system gave."""
    return ExpectedFailure(f"The file {path} couldn't be written ({error.strerror or error}). "
                           f"Free some space on the disk, or check that you can write to "
                           f"{path.parent}, then run the step again.")
