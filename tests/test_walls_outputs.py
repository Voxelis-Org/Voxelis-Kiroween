"""Unit tests for what the Walls_Step writes and reports, splatpipe/walls/outputs.py and
splatpipe/walls/report.py (walls spec, sub-task 9.6, Req 17.6–17.9, 18.3, 18.6, 18.8, 19.5,
19.6, 22.1).

walls_prior runs on a tiny hand-made Depth_Prior whose pixels each hit one rule of D3, D4 and
A13. The Walls_Demo runs on frame BOX_DEMO_FRAME of the box room of tests/camera_files.py at its
Prior_Size, 128 x 96, whose top half sees the flat back wall: the Walls_Prior's depth is the true
depth and the Depth_Prior's a bumpy one, so the walls third of the panel shows the back wall as
one colour, worked out here from the wall's true normal, and the prior third doesn't. The disk-full
tests make the one write call each writer uses raise ENOSPC for that file only. Every oracle is
worked out here and calls nothing it checks.

Property 7 (sub-task 9.7, Req 17.6, 17.7, 18.9) draws a Depth_Prior at a Prior_Size up to 8 x 8,
a Walls_Depth with a few broken pixels and normals with a few broken ones, writes the Walls_Prior
and the Depth_Prior_Copy with write_prior and write_index, and reads them back with
read_prior_set against a one-image Sparse_Model.
"""

from __future__ import annotations

import errno
import tempfile
import tomllib
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp
from scipy import ndimage

from splatpipe.errors import ExpectedFailure
from splatpipe.train.colmap import Camera, ModelImage, SparseModel
from splatpipe.train.priors import (Prior, PriorEntry, PriorIndex, check_prior, load_prior,
                                    prior_path, read_prior_set, write_index, write_prior)
from splatpipe.walls.geometry import normalised_coords
from splatpipe.walls.outputs import (DEMOS, RECORD, SCRATCH, demo_panel, demo_path,
                                     depth_prior_copy, remove_demos, remove_scratch, walls_prior,
                                     write_demo, write_prior_file, write_record)
from splatpipe.walls.report import no_trusted_failure, not_all_passed_warning, timed_out_warning

from camera_files import (BOX_BACK, BOX_CAMERA, BOX_IMAGES, box_camera, box_covered, box_depth,
                          box_frame, box_normal, box_surface)

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_WALLS = tomllib.loads((REPO_ROOT / "config.toml").read_text(encoding="utf-8"))["walls"]
DEMO_QUALITY = CONFIG_WALLS["demo_jpeg_quality"]   # the real demo_jpeg_quality, 90

BOX_SIZE = (BOX_CAMERA.width, BOX_CAMERA.height)   # the box room's Prior_Size, (w, h)
BOX_DEMO_FRAME = 4         # its top half sees the back wall, its bottom half the floor
BUMP = 0.02                # the Depth_Prior's bumps, as a share of the true depth
BUMP_PERIOD = 8.0          # pixels
JPEG_BLOCK = 16            # a JPEG MCU with 2 x 2 chroma subsampling, in pixels
BLOCK_MARGIN = 2           # pixels kept clear of each MCU's edge, where upsampling blends
JPEG_TOLERANCE = 3         # grey levels
NO_SPACE = OSError(errno.ENOSPC, "No space left on device")


# ---------------------------------------------------------------------------------------------
# walls_prior and depth_prior_copy (Req 17.6 to 17.9, D2, D3, D4, A13)


def small_prior() -> Prior:
    """A 3 x 4 Depth_Prior with its bottom-right pixel unmarked, inverse depths and unit normals
    that differ at every pixel, so a value taken from it is told apart from a new one."""
    mask = np.ones((3, 4), dtype=bool)
    mask[2, 3] = False
    inverse = np.where(mask, np.linspace(0.2, 0.9, 12).reshape(3, 4), 0.0).astype(np.float32)
    tilt = np.linspace(-0.3, 0.3, 12).reshape(3, 4)
    normal = np.stack([tilt, -tilt / 2, -np.ones_like(tilt)], axis=-1)
    normal /= np.linalg.norm(normal, axis=-1, keepdims=True)
    normal[~mask] = 0.0
    return Prior(inverse_depth=inverse, normal=normal.astype(np.float32), mask=mask)


def test_walls_prior_keeps_the_mask_and_takes_the_prior_where_the_new_values_fail() -> None:
    prior = small_prior()
    depth = np.full((3, 4), 2.0)
    depth[0, 1] = 1e300     # 1 / depth rounds to 0 in float32
    depth[0, 2] = 1e-300    # 1 / depth overflows float32
    depth[0, 3] = np.nan    # no Walls_Depth
    depth[1, 0] = -2.0      # an inverse depth below 0
    depth[2, 3] = 3.0       # unmarked: never used
    normal = np.zeros((3, 4, 3))
    normal[...] = (0.6, -0.8, 0.0)   # the change chain's unit normal, unlike any of the prior's
    normal[1, 1] = np.nan   # no normal
    normal[1, 2] = 0.0      # a normal of length 0
    normal[2, 0] = (np.inf, 0.0, -1.0)
    normal[2, 3] = (0.0, 0.0, -1.0)
    made = walls_prior(prior, depth, normal, prior.mask.copy())

    assert made.mask.dtype == np.bool_ and np.array_equal(made.mask, prior.mask)
    assert made.inverse_depth.dtype == np.float32 and made.normal.dtype == np.float32
    from_prior = [(0, 1), (0, 2), (0, 3), (1, 0)]
    for pixel in from_prior:
        assert made.inverse_depth[pixel] == prior.inverse_depth[pixel]
        assert np.array_equal(made.normal[pixel], prior.normal[pixel])
    for pixel in [(1, 1), (1, 2), (2, 0)]:
        assert made.inverse_depth[pixel] == np.float32(0.5)
        assert np.array_equal(made.normal[pixel], prior.normal[pixel])
    for pixel in [(0, 0), (1, 3), (2, 1), (2, 2)]:
        assert made.inverse_depth[pixel] == np.float32(0.5)
        assert np.array_equal(made.normal[pixel], np.float32([0.6, -0.8, 0.0]))
    assert made.inverse_depth[2, 3] == 0.0
    assert np.array_equal(made.normal[2, 3], np.zeros(3, dtype=np.float32))
    check_prior(Path("walls.npz"), made, 4, 3)


def test_walls_prior_refuses_covered_pixels_that_arent_the_mask() -> None:
    prior = small_prior()
    covered = prior.mask.copy()
    covered[2, 3] = True
    with pytest.raises(ValueError, match="mask"):
        walls_prior(prior, np.full((3, 4), 2.0), np.zeros((3, 4, 3)), covered)


def test_depth_prior_copy_is_bit_for_bit() -> None:
    prior = small_prior()
    nan_bits = np.array(prior.normal, copy=True)
    nan_bits.view(np.uint32)[0, 0, 0] = 0x7FC01234   # a NaN with its own payload
    prior = Prior(inverse_depth=prior.inverse_depth, normal=nan_bits, mask=prior.mask)
    copy = depth_prior_copy(prior)
    for name in ("inverse_depth", "normal", "mask"):
        original, copied = getattr(prior, name), getattr(copy, name)
        assert copied.dtype == original.dtype and copied.shape == original.shape
        assert copied.tobytes() == original.tobytes()
        assert not np.shares_memory(copied, original)


# ---------------------------------------------------------------------------------------------
# The Walls_Demo (Req 18.3, 18.10, A17)


def box_priors() -> tuple[Prior, Prior]:
    """Frame BOX_DEMO_FRAME's Depth_Prior, at a bumpy depth, and its Walls_Prior, at the true
    depth, with the true normals, at the Prior_Size."""
    covered = box_covered(BOX_DEMO_FRAME, BOX_SIZE)
    depth = box_depth(BOX_DEMO_FRAME, BOX_SIZE)
    normal = box_normal(BOX_DEMO_FRAME, BOX_SIZE)
    rows, columns = np.indices(covered.shape)
    phase = 2 * np.pi / BUMP_PERIOD
    bumpy = depth * (1 + BUMP * np.sin(phase * columns) * np.cos(phase * rows))
    inverse = np.where(covered, 1.0 / np.where(covered, bumpy, 1.0), 0.0).astype(np.float32)
    depth_prior = Prior(inverse_depth=inverse, normal=normal.astype(np.float32), mask=covered)
    return depth_prior, walls_prior(depth_prior, depth, normal, covered)


def write_box_frame(images: Path) -> str:
    """Write frame BOX_DEMO_FRAME as a JPEG at quality 100 into `images`, and give its name."""
    name = BOX_IMAGES[BOX_DEMO_FRAME].name
    encoded, data = cv2.imencode(".jpg", box_frame(BOX_DEMO_FRAME),
                                 [cv2.IMWRITE_JPEG_QUALITY, 100])
    assert encoded
    images.mkdir(parents=True, exist_ok=True)
    (images / name).write_bytes(data.tobytes())
    return name


def wall_colour() -> np.ndarray:
    """The back wall's colour, BGR: its true unit normal n as (n + 1) / 2 × 255 in R, G, B."""
    surface = box_surface(BOX_DEMO_FRAME, BOX_SIZE)
    normal = box_normal(BOX_DEMO_FRAME, BOX_SIZE)[surface == BOX_BACK][0]
    return np.rint((normal + 1) / 2 * 255)[::-1]


def flat_wall_pixels() -> np.ndarray:
    """The back wall's pixels whose four neighbours see it too, (h, w) bool, off the border."""
    wall = box_surface(BOX_DEMO_FRAME, BOX_SIZE) == BOX_BACK
    inner = ndimage.binary_erosion(wall, structure=ndimage.generate_binary_structure(2, 1),
                                   border_value=0)
    inner[0, :] = inner[-1, :] = inner[:, 0] = inner[:, -1] = False
    return inner


def away_from_blocks(pixels: np.ndarray) -> np.ndarray:
    """The pixels of `pixels` whose whole JPEG MCU is in `pixels`, kept BLOCK_MARGIN clear of the
    MCU's edge. The Prior_Size's width is a multiple of JPEG_BLOCK, so each third starts on an
    MCU's edge."""
    height, width = pixels.shape
    kept = np.zeros_like(pixels)
    for top in range(0, height - JPEG_BLOCK + 1, JPEG_BLOCK):
        for left in range(0, width - JPEG_BLOCK + 1, JPEG_BLOCK):
            if pixels[top:top + JPEG_BLOCK, left:left + JPEG_BLOCK].all():
                kept[top + BLOCK_MARGIN:top + JPEG_BLOCK - BLOCK_MARGIN,
                     left + BLOCK_MARGIN:left + JPEG_BLOCK - BLOCK_MARGIN] = True
    return kept


def test_demo_panel_shows_the_frame_and_the_flat_wall_as_one_colour(tmp_path: Path) -> None:
    images = tmp_path / "images"
    name = write_box_frame(images)
    depth_prior, walls = box_priors()
    x, y = normalised_coords(box_camera(), *BOX_SIZE)
    panel = demo_panel(images, name, box_camera(), depth_prior, walls, x, y)

    width, height = BOX_SIZE
    assert panel.shape == (height, 3 * width, 3) and panel.dtype == np.uint8
    frame = cv2.imdecode(np.fromfile(images / name, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert np.array_equal(panel[:, :width], frame)
    flat = flat_wall_pixels()
    walls_third = panel[:, 2 * width:].astype(np.int64)
    assert np.abs(walls_third[flat] - wall_colour()).max() <= 1
    for third in (panel[:, width:2 * width], panel[:, 2 * width:]):
        assert not third[0].any() and not third[-1].any()   # no normal on the border: black
        assert not third[:, 0].any() and not third[:, -1].any()


def test_write_demo_gives_a_jpeg_with_the_flat_wall_one_colour_in_the_walls_third(
        tmp_path: Path) -> None:
    images = tmp_path / "work" / "cameras" / "images"
    name = write_box_frame(images)
    depth_prior, walls = box_priors()
    x, y = normalised_coords(box_camera(), *BOX_SIZE)
    panel = demo_panel(images, name, box_camera(), depth_prior, walls, x, y)
    path = demo_path(tmp_path / "out", name)
    assert path == tmp_path / "out" / DEMOS / name
    write_demo(path, panel, DEMO_QUALITY)

    decoded = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded is not None
    width, height = BOX_SIZE
    assert decoded.shape == (height, 3 * width, 3)
    checked = away_from_blocks(flat_wall_pixels())
    assert checked.sum() >= 100
    walls_third = decoded[:, 2 * width:].astype(np.int64)
    assert np.abs(walls_third[checked] - wall_colour()).max() <= JPEG_TOLERANCE
    prior_third = decoded[:, width:2 * width].astype(np.int64)[checked]
    assert (prior_third.max(axis=0) - prior_third.min(axis=0)).max() > 10 * JPEG_TOLERANCE


# ---------------------------------------------------------------------------------------------
# Writes that fail (Req 18.6)


def fail_for(monkeypatch: pytest.MonkeyPatch, method: str, target: Path) -> None:
    """Make Path.`method` raise NO_SPACE for `target` only, and work as before for every other
    path."""
    original = getattr(Path, method)

    def failing(self: Path, *args: object, **kwargs: object) -> object:
        if self == target:
            raise NO_SPACE
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, method, failing)


def assert_names_file(error: ExpectedFailure, path: Path) -> None:
    """Req 18.6's message: the file, the reason and what to do, with no traceback."""
    message = str(error)
    assert f"The file {path} couldn't be written (No space left on device)." in message
    assert "run the step again" in message
    assert "Traceback" not in message


def test_write_prior_file_names_the_file_when_the_disk_is_full(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = prior_path(tmp_path, "000004.jpg")
    fail_for(monkeypatch, "open", path)
    with pytest.raises(ExpectedFailure) as caught:
        write_prior_file(tmp_path, "000004.jpg", small_prior())
    assert_names_file(caught.value, path)


def test_write_prior_file_writes_the_prior_format(tmp_path: Path) -> None:
    write_prior_file(tmp_path, "000004.jpg", small_prior())
    read = load_prior(prior_path(tmp_path, "000004.jpg"))
    assert read.inverse_depth.tobytes() == small_prior().inverse_depth.tobytes()


def test_write_record_names_the_file_when_the_disk_is_full(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / RECORD
    fail_for(monkeypatch, "write_text", path)
    with pytest.raises(ExpectedFailure) as caught:
        write_record(tmp_path, {"format": 1, "frames": []})
    assert_names_file(caught.value, path)


def test_write_demo_names_the_file_when_the_disk_is_full(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = demo_path(tmp_path / "out", "000004.jpg")
    fail_for(monkeypatch, "write_bytes", path)
    with pytest.raises(ExpectedFailure) as caught:
        write_demo(path, np.zeros((4, 12, 3), dtype=np.uint8), DEMO_QUALITY)
    assert_names_file(caught.value, path)


# ---------------------------------------------------------------------------------------------
# The folders the step clears (Req 1.10)


def test_remove_demos_removes_only_out_walls(tmp_path: Path) -> None:
    out = tmp_path / "out"
    (out / DEMOS).mkdir(parents=True)
    (out / DEMOS / "000004.jpg").write_bytes(b"old demo")
    (out / "reports").mkdir()
    (out / "reports" / "walls.json").write_text("{}", encoding="utf-8")
    (out / "train.ply").write_bytes(b"splat")
    remove_demos(out)
    assert not (out / DEMOS).exists()
    assert (out / "reports" / "walls.json").is_file() and (out / "train.ply").is_file()
    remove_demos(out)   # nothing there: no error


def test_remove_scratch_removes_only_the_scratch_folder(tmp_path: Path) -> None:
    work = tmp_path / "work" / "walls"
    (work / SCRATCH / "deep").mkdir(parents=True)
    (work / SCRATCH / "deep" / "part.npy").write_bytes(b"scratch")
    (work / "000004.npz").write_bytes(b"prior")
    (work / RECORD).write_text("{}", encoding="utf-8")
    remove_scratch(work)
    assert not (work / SCRATCH).exists()
    assert (work / "000004.npz").is_file() and (work / RECORD).is_file()
    remove_scratch(work)


# ---------------------------------------------------------------------------------------------
# The warnings and the no-trusted-frame failure (Req 18.8, 19.5, 19.6)


def test_timed_out_warning_gives_its_counts_the_stop_time_and_what_set_it() -> None:
    eastern = timezone(timedelta(hours=-5))
    stop = datetime(2026, 11, 19, 9, 25, 31, tzinfo=eastern)   # 14:25:31 UTC
    warning = timed_out_warning(3, 20, stop, "max_minutes")
    assert "for 3 of 20 frames" in warning
    assert "2026-11-19 14:25:31 UTC" in warning
    assert "max_minutes" in warning and "keep their depth prior" in warning
    reserve = timed_out_warning(1, 1, stop, "train_reserve_minutes")
    assert "for 1 of 1 frame:" in reserve
    assert "train_reserve_minutes" in reserve and "cap deadline" in reserve
    for text in (warning, reserve):
        assert "Traceback" not in text
    with pytest.raises(ValueError):
        timed_out_warning(3, 20, datetime(2026, 11, 19, 14, 25, 31), "max_minutes")


def test_not_all_passed_warning_gives_its_counts() -> None:
    warning = not_all_passed_warning(12, 5, 3, 20)
    assert warning.startswith("12 of 20 frames passed")
    assert "5 got a walls prior from a fallback depth" in warning
    assert "3 kept their depth prior unchanged" in warning
    assert "Traceback" not in warning


def test_no_trusted_failure_gives_its_counts() -> None:
    failure = no_trusted_failure(20, 4, 16, 6)
    assert isinstance(failure, ExpectedFailure)
    message = str(failure)
    assert "None of the 20 frames passed" in message
    assert "4 ran out of time" in message
    assert "16 failed their self-check, 6 of them because they had too few" in message
    assert "trains on the depth priors" in message
    assert "Traceback" not in message


# ---------------------------------------------------------------------------------------------
# Property 7 (sub-task 9.7): a Walls_Prior round-trips through the Prior_Format and keeps the mask

PROPERTY_NAME = "000004.jpg"                    # the one image of the property's Sparse_Model
PROPERTY_QUATERNION = (0.9, 0.1, -0.3, 0.2)     # w x y z, as images.bin stores it, not unit
PROPERTY_TRANSLATION = (0.25, -1.5, 3.0)
CAMERA_PER_PRIOR = 4                            # Shared_Camera pixels per Prior_Size pixel
LARGEST_PRIOR_SIDE = 8
# Walls_Depths whose float32 inverse isn't finite and above 0: NaN, inverses of inf, -inf and 0
# and -0, one below 0, one that rounds to 0 in float32 and one that overflows float32 (D3).
BROKEN_DEPTHS = (np.nan, np.inf, -np.inf, 0.0, -0.0, -2.0, 1e300, 1e-300)
# Change-chain normals that aren't finite or have a length of 0 (D4).
BROKEN_NORMALS = ((np.nan, np.nan, np.nan), (0.0, 0.0, 0.0), (np.inf, 0.0, -1.0),
                  (0.0, np.nan, -1.0), (-0.0, 0.0, -0.0))
MOST_BROKEN = 4                                 # pixels forced broken per example, at most


def unit_vectors(raw: np.ndarray) -> np.ndarray:
    """`raw` (h, w, 3) set to unit length, with (0, 0, -1) where it's too short to set."""
    length = np.linalg.norm(raw, axis=-1, keepdims=True)
    return np.where(length >= 1e-3, raw / np.where(length >= 1e-3, length, 1.0),
                    np.array([0.0, 0.0, -1.0]))


def facing_camera(normal: np.ndarray) -> np.ndarray:
    """`normal` (h, w, 3) turned round where it points away from the camera, z above 0."""
    return normal * np.where(normal[..., 2:] > 0, -1.0, 1.0)


@st.composite
def walls_cases(draw: st.DrawFn) -> tuple[Prior, np.ndarray, np.ndarray]:
    """A Depth_Prior at a Prior_Size from 1 x 1 to 8 x 8 with at least one marked pixel, and a
    Walls_Depth and change-chain normals at that size, with a few marked pixels forced broken."""
    height = draw(st.integers(1, LARGEST_PRIOR_SIDE))
    width = draw(st.integers(1, LARGEST_PRIOR_SIDE))
    mask = draw(hnp.arrays(np.bool_, (height, width)))
    mask[draw(st.integers(0, height - 1)), draw(st.integers(0, width - 1))] = True
    positive = st.floats(min_value=2.0 ** -14, max_value=2.0 ** 14, width=32)
    inverse = draw(hnp.arrays(np.float32, (height, width), elements=positive))
    inverse = np.where(mask, inverse, np.float32(0)).astype(np.float32)
    unit_parts = st.floats(min_value=-1.0, max_value=1.0)
    prior_normal = unit_vectors(draw(hnp.arrays(np.float64, (height, width, 3),
                                                elements=unit_parts))).astype(np.float32)
    depth_prior = Prior(inverse_depth=inverse, normal=prior_normal, mask=mask)

    depths = st.one_of(st.floats(min_value=1e-3, max_value=1e3),
                       st.floats(min_value=0.0, exclude_min=True, allow_infinity=False))
    depth = draw(hnp.arrays(np.float64, (height, width), elements=depths))
    unmarked = draw(hnp.arrays(np.float64, (height, width), elements=st.floats()))
    depth = np.where(mask, depth, unmarked)   # never read off the mask: anything at all
    normal = facing_camera(unit_vectors(draw(hnp.arrays(np.float64, (height, width, 3),
                                                        elements=unit_parts))))
    covered = [tuple(int(axis) for axis in pixel) for pixel in np.argwhere(mask)]
    pixels = st.sampled_from(covered)
    for pixel, value in draw(st.lists(st.tuples(pixels, st.sampled_from(BROKEN_DEPTHS)),
                                      max_size=MOST_BROKEN)):
        depth[pixel] = value
    for pixel, value in draw(st.lists(st.tuples(pixels, st.sampled_from(BROKEN_NORMALS)),
                                      max_size=MOST_BROKEN)):
        normal[pixel] = value
    return depth_prior, depth, normal


def expected_walls_prior(depth_prior: Prior, depth: np.ndarray,
                         normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The oracle of Req 17.7, 17.8, D3 and D4, pixel by pixel: the inverse depth and normal a
    Walls_Prior holds. At a marked pixel, float32(1 / depth) and the normal as float32 when it's
    finite and not of length 0, else the Depth_Prior's normal; the Depth_Prior's inverse depth
    and normal where float32(1 / depth) isn't finite and above 0; 0s off the mask."""
    height, width = depth_prior.mask.shape
    inverse = np.zeros((height, width), dtype=np.float32)
    normals = np.zeros((height, width, 3), dtype=np.float32)
    with np.errstate(all="ignore"):
        for pixel in np.ndindex(height, width):
            if not depth_prior.mask[pixel]:
                continue
            value = np.float32(np.float64(1.0) / np.float64(depth[pixel]))
            new_normal = np.asarray(normal[pixel], dtype=np.float64).astype(np.float32)
            if not (np.isfinite(value) and value > 0):
                inverse[pixel] = depth_prior.inverse_depth[pixel]
                normals[pixel] = depth_prior.normal[pixel]
                continue
            inverse[pixel] = value
            usable = bool(np.isfinite(new_normal).all() and (new_normal != 0).any())
            normals[pixel] = new_normal if usable else depth_prior.normal[pixel]
    return inverse, normals


def one_image_model(width: int, height: int) -> tuple[SparseModel, PriorIndex]:
    """A one-image Sparse_Model and the Prior_Index of its one Prior at the Prior_Size
    `width` x `height`, whose camera and pose are the model's, value for value."""
    camera = Camera(camera_id=1, width=CAMERA_PER_PRIOR * width,
                    height=CAMERA_PER_PRIOR * height, fx=3.1 * width, fy=3.3 * height,
                    cx=2.05 * width, cy=1.95 * height)
    image = ModelImage(image_id=1, quaternion=np.array(PROPERTY_QUATERNION, dtype=np.float64),
                       translation=np.array(PROPERTY_TRANSLATION, dtype=np.float64),
                       camera_id=1, name=PROPERTY_NAME, observed_ids=np.empty(0, np.uint64))
    model = SparseModel(camera, (image,), np.empty((0, 3)), np.empty((0, 3), np.uint8),
                        np.empty(0, np.uint64))
    entry = PriorEntry(name=PROPERTY_NAME,
                       quaternion=tuple(float(value) for value in image.quaternion),
                       translation=tuple(float(value) for value in image.translation),
                       scale=1.25, offset=0.01, fit_points=12, kept_points=10,
                       median_error=0.05, metres_per_unit=1.0)
    index = PriorIndex(camera=(camera.width, camera.height, camera.fx, camera.fy, camera.cx,
                               camera.cy),
                       prior_width=width, prior_height=height, metres_per_unit=1.0,
                       priors=(entry,))
    return model, index


def round_trip(folder: Path, prior: Prior) -> Prior:
    """Write `prior` with write_prior and its index with write_index into the new `folder`, and
    give the Prior that read_prior_set reads back from it."""
    height, width = prior.mask.shape
    model, index = one_image_model(width, height)
    folder.mkdir()
    write_prior(prior_path(folder, PROPERTY_NAME), prior)
    write_index(folder, index)
    prior_set = read_prior_set(folder, model)
    assert prior_set is not None and list(prior_set.priors) == [PROPERTY_NAME]
    return prior_set.priors[PROPERTY_NAME]


def assert_same_bits(read: Prior, inverse: np.ndarray, normal: np.ndarray,
                     mask: np.ndarray) -> None:
    """`read`'s three arrays are `inverse`, `normal` and `mask`, bit for bit."""
    assert read.inverse_depth.dtype == np.float32 and read.normal.dtype == np.float32
    assert read.mask.dtype == np.bool_
    assert read.inverse_depth.shape == inverse.shape and read.normal.shape == normal.shape
    assert np.array_equal(read.inverse_depth.view(np.uint32), inverse.view(np.uint32))
    assert np.array_equal(read.normal.view(np.uint32), normal.view(np.uint32))
    assert np.array_equal(read.mask.view(np.uint8), mask.view(np.uint8))


@settings(max_examples=100, deadline=None)
@given(walls_cases())
def test_walls_prior_round_trips_and_keeps_the_mask(
        case: tuple[Prior, np.ndarray, np.ndarray]) -> None:
    """Feature: walls, Property 7: A Walls_Prior round-trips through the Prior_Format and keeps
    the mask.

    **Validates: Requirements 17.6, 17.7, 18.9**
    """
    depth_prior, depth, normal = case
    height, width = depth_prior.mask.shape
    expected_inverse, expected_normal = expected_walls_prior(depth_prior, depth, normal)
    made = walls_prior(depth_prior, depth, normal, depth_prior.mask.copy())
    with tempfile.TemporaryDirectory() as temporary:
        folder = Path(temporary)
        read = round_trip(folder / "walls", made)
        copy = round_trip(folder / "copy", depth_prior_copy(depth_prior))

        assert np.array_equal(read.mask.view(np.uint8), depth_prior.mask.view(np.uint8))
        marked = read.mask
        assert np.isfinite(read.inverse_depth).all()
        assert (read.inverse_depth[marked] > 0).all()
        assert (read.inverse_depth[~marked] == 0).all()
        check_prior(prior_path(folder / "walls", PROPERTY_NAME), read, width, height)
        assert_same_bits(read, made.inverse_depth, made.normal, made.mask)
        assert_same_bits(read, expected_inverse, expected_normal, depth_prior.mask)
        assert_same_bits(copy, depth_prior.inverse_depth, depth_prior.normal, depth_prior.mask)
