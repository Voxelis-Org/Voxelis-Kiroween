"""Tests for splatpipe/frames/files.py: the Picks' names, the Grey_Copy, the JPEG and PNG writers
and the Frames_Record.

Every image is a tiny numpy array built here, so no test needs a video or ffmpeg, and every file
goes in tmp_path. The tests read the images back with cv2.imdecode, with their channels and bit
depth as the file holds them.

The full disk is a stand-in that raises OSError(errno.ENOSPC, "No space left on device") from the
Path method each writer writes with: Path.write_bytes for write_jpeg and write_png, and
Path.write_text for write_record, because splatpipe.jsonio.write_json, which write_record calls,
writes with it. The failed encode is a stand-in for cv2.imencode, as files.py sees it, that
reports a failure and gives no bytes.

The Frames_Record test gives frames_record every number and flag as a numpy value: bools and ints,
float64 seconds and Mask_Coverage, the Scores as float64 arrays and the Picks as an int64 array.
check_json_value refuses every numpy scalar, so passing it shows that each value was turned into
a plain Python one.
"""

import errno
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, NoReturn

import cv2
import numpy as np
import pytest

from splatpipe.errors import ExpectedFailure
from splatpipe.frames.choose import Kept
from splatpipe.frames.files import (frames_record, grey_copy, pick_name, write_jpeg, write_png,
                                    write_record)
from splatpipe.frames.mask import MaskChoice
from splatpipe.frames.scores import Scores
from splatpipe.jsonio import check_json_value

FULL_DISK = "No space left on device"  # the reason the stand-in full disk gives
QUALITY = 95  # jpeg_quality in the real config


def colour_frame() -> np.ndarray:
    """A 6 × 8 BGR colour frame of seeded noise, with each channel drawn on its own from 90 to 130.

    Its grey spans only a few dozen levels, so histogram equalisation changes it, and its three
    channels differ, so a grey worked out as RGB rather than BGR would differ too.
    """
    return np.random.default_rng(3).integers(90, 131, size=(6, 8, 3), dtype=np.uint8)


def mask_picture() -> np.ndarray:
    """A 6 × 8 Mask image: a black corner of 2 × 3 pixels, 0, on white, 255."""
    image = np.full((6, 8), 255, dtype=np.uint8)
    image[:2, :3] = 0
    return image


def read_image(path: Path) -> np.ndarray:
    """The image in the file at `path`, as cv2.imdecode reads it with its channels and bit depth
    unchanged."""
    image = cv2.imdecode(np.frombuffer(path.read_bytes(), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    assert image is not None, f"cv2.imdecode couldn't read {path}"
    return image


# The Picks' names (Req 8.4)

@pytest.mark.parametrize(("number", "name"), [(0, "000000.jpg"), (123, "000123.jpg")])
def test_a_picks_file_name_is_its_number_padded_to_6_digits(number: int, name: str) -> None:
    assert pick_name(number) == name


# The Grey_Copy (Req 8.3)

def test_the_grey_copy_is_the_colour_frames_grey_histogram_equalised_at_its_size() -> None:
    frame = colour_frame()
    grey = grey_copy(frame)
    assert (grey.shape, grey.dtype) == ((6, 8), np.uint8)
    assert np.array_equal(grey, cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)))


# The JPEGs (Req 8.3) and the Mask's PNG (Req 7.8)

@pytest.mark.parametrize(("grey", "shape"), [(False, (6, 8, 3)), (True, (6, 8))],
                         ids=["colour pick", "grey copy"])
def test_write_jpeg_writes_a_jpeg_that_reads_back_at_the_images_size(
        tmp_path: Path, grey: bool, shape: tuple[int, ...]) -> None:
    frame = colour_frame()
    path = tmp_path / "000000.jpg"
    write_jpeg(path, grey_copy(frame) if grey else frame, QUALITY)
    image = read_image(path)
    assert (image.shape, image.dtype) == (shape, np.uint8)


def test_write_png_writes_an_8_bit_grey_png_that_holds_only_0_and_255(tmp_path: Path) -> None:
    path = tmp_path / "mask.png"
    write_png(path, mask_picture())
    image = read_image(path)
    assert (image.shape, image.dtype) == ((6, 8), np.uint8)
    assert np.unique(image).tolist() == [0, 255]
    # The PNG's own header holds the bit depth at byte 24 and the colour type at byte 25, 0 for
    # grey. A grey PNG of 1, 2 or 4 bits would also read back as 8-bit above.
    data = path.read_bytes()
    assert (data[24], data[25]) == (8, 0)


# The Frames_Record (Req 2.7, 3.6, 4.10, 7.9, 8.6)

# The Scores of five Score_Frames 0.5 s apart, in a Used_Part from 0.25 s to 2.75 s.
SCORES = Scores(times=np.array([0.25, 0.75, 1.25, 1.75, 2.25]),
                sharpness=np.array([10.0, 12.5, 812.5, 40.25, 99.0]),
                motion=np.array([0.0, 0.3, 2.1, 3.5, 1.75]),
                contrast=np.array([20.0, 21.5, 48.0, 30.1, 33.0]))
# The Kept_Range of frames 2 and 3, as numpy values. The Still_Trim cut frames 0 and 1, which last
# 1 s, at the start, and frame 4, which lasts 0.5 s to the Used_Part's end, at the end.
KEPT = Kept(first=np.int64(2), last=np.int64(3), start_seconds=SCORES.times[2],
            end_seconds=SCORES.times[4], cut_start_seconds=np.float64(1.0),
            cut_end_seconds=np.float64(0.5))


def mask_choice(stuck: int, used: bool) -> MaskChoice:
    """The MaskChoice of 3 Mask_Samples of 4 × 4 whose first `stuck` pixels are Stuck_Pixels, with
    the Mask_Coverage as numpy's float64 and `used` as a numpy bool."""
    pixels = np.arange(16).reshape(4, 4) < stuck
    return MaskChoice(samples=3, stuck=pixels, coverage=pixels.mean(), used=np.bool_(used))


def record(colour_transfer: str | None, tone_mapped: bool, mask: MaskChoice) -> dict[str, Any]:
    """The Frames_Record of KEPT, SCORES and the Picks of frames 2 and 3, with every number and
    flag given to frames_record as a numpy value."""
    return frames_record(colour_transfer=colour_transfer, tone_mapped=np.bool_(tone_mapped),
                         kept=KEPT, target_picks=np.int64(3), jolt_filter_skipped=np.bool_(True),
                         pick_width=np.int64(16), pick_height=np.int64(12), mask=mask,
                         scores=SCORES, picks=np.array([2, 3], dtype=np.int64))


@pytest.mark.parametrize(
    ("colour_transfer", "tone_mapped", "stuck", "used", "coverage", "mask_file"), [
        pytest.param("arib-std-b67", True, 1, True, 0.0625, "mask.png",
                     id="an HDR clip with a mask"),
        pytest.param(None, False, 8, False, 0.5, None, id="no colour transfer and no mask"),
    ])
def test_the_frames_record_holds_every_field_as_plain_json_values(
        colour_transfer: str | None, tone_mapped: bool, stuck: int, used: bool,
        coverage: float, mask_file: str | None) -> None:
    result = record(colour_transfer, tone_mapped, mask_choice(stuck, used))
    check_json_value(result, "frames")
    expected = {
        "colour_transfer": colour_transfer,
        "tone_mapped": tone_mapped,
        "still_start_seconds": 1.0,
        "still_end_seconds": 0.5,
        "kept_start_seconds": 1.25,
        "kept_end_seconds": 2.25,
        "target_picks": 3,
        "jolt_filter_skipped": True,
        "pick_width": 16,
        "pick_height": 12,
        "mask_coverage": coverage,
        "mask_used": used,
        "mask_file": mask_file,
        "picks": [
            {"number": 0, "file": "000000.jpg", "frame_number": 2, "seconds": 1.25,
             "sharpness": 812.5, "motion": 2.1, "contrast": 48.0},
            {"number": 1, "file": "000001.jpg", "frame_number": 3, "seconds": 1.75,
             "sharpness": 40.25, "motion": 3.5, "contrast": 30.1},
        ],
    }
    assert result == expected
    # The same JSON text too, so each value is also of the expected kind: true isn't 1, and 16
    # isn't 16.0.
    assert json.dumps(result, sort_keys=True) == json.dumps(expected, sort_keys=True)


# Writes that fail (Req 8.10, 13.9)

def write_a_pick(path: Path) -> None:
    """Write colour_frame() to `path` as a JPEG at QUALITY."""
    write_jpeg(path, colour_frame(), QUALITY)


def write_a_mask(path: Path) -> None:
    """Write mask_picture() to `path` as a PNG."""
    write_png(path, mask_picture())


def write_a_record(path: Path) -> None:
    """Write a Frames_Record with a Mask to `path`."""
    write_record(path, record("arib-std-b67", True, mask_choice(1, True)))


def full_disk(*args: object, **kwargs: object) -> NoReturn:
    """A stand-in for Path.write_bytes or Path.write_text that fails the way a full disk does."""
    raise OSError(errno.ENOSPC, FULL_DISK)


def failed_encode(*args: object, **kwargs: object) -> tuple[bool, np.ndarray]:
    """A stand-in for cv2.imencode that reports a failed encode and gives no bytes."""
    return False, np.empty(0, dtype=np.uint8)


def assert_names_the_file_and_the_reason(error: ExpectedFailure, path: Path, reason: str) -> None:
    """Check that the failure's Readable_Error names the file and the reason, with no traceback."""
    message = str(error)
    assert str(path) in message
    assert reason in message
    assert "Traceback" not in message


@pytest.mark.parametrize(("write", "name", "method"), [
    pytest.param(write_a_pick, "000000.jpg", "write_bytes", id="write_jpeg"),
    pytest.param(write_a_mask, "mask.png", "write_bytes", id="write_png"),
    # write_record writes through splatpipe.jsonio.write_json, which writes with Path.write_text.
    pytest.param(write_a_record, "frames.json", "write_text", id="write_record"),
])
def test_a_write_that_fails_like_a_full_disk_names_the_file_and_the_reason(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, write: Callable[[Path], None],
        name: str, method: str) -> None:
    path = tmp_path / name
    monkeypatch.setattr(Path, method, full_disk)
    with pytest.raises(ExpectedFailure) as caught:
        write(path)
    assert_names_the_file_and_the_reason(caught.value, path, FULL_DISK)


@pytest.mark.parametrize(("write", "name", "kind"), [
    pytest.param(write_a_pick, "000000.jpg", "JPEG", id="write_jpeg"),
    pytest.param(write_a_mask, "mask.png", "PNG", id="write_png"),
])
def test_a_failed_encode_names_the_file_and_the_reason(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, write: Callable[[Path], None],
        name: str, kind: str) -> None:
    path = tmp_path / name
    monkeypatch.setattr("splatpipe.frames.files.cv2.imencode", failed_encode)
    with pytest.raises(ExpectedFailure) as caught:
        write(path)
    assert_names_the_file_and_the_reason(caught.value, path,
                                         f"couldn't encode the image as a {kind}")
