"""Unit tests for the Walls_Step's inputs: splatpipe/walls/inputs.py (walls spec, Req 2).

Every Job is made by conftest's new_job in the test config's Jobs_Folder, inside tmp_path, never
in data/ (Req 22.12). Into it go tests/camera_files.py's plane scene as the Camera_Solve_Output,
4 JPEGs of 16 x 12 pixels in one colour, IMAGE_BGR; a Depth_Prior_Set written with the
Prior_Writer, write_prior and write_index, for PRIOR_NAMES, 3 of the scene's 4 images, so
000003.jpg has none; and a frames.json whose Picks name every image, with Pick_Numbers and
seconds that aren't in images.bin's order, so the sort and the join both show.

The scaling tests write the same scene with WIDE_CAMERA, 800 x 600, whose Score_Size with the
test config's score_long_edge of 640 is 640 x 480, worked out by hand: each edge × 640 / 800.

A stand-in for Path.open, which Path.read_bytes and Path.read_text open their files with, records
each path it's called on and then runs the real method, so it shows that check_inputs reads
nothing (Req 2.5).

Each refusal is an ExpectedFailure whose message holds no traceback (Req 22.1). The Frame and
Mask_File refusals are checked against the Train_Step's own messages for the same file.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from splatpipe.config import load_config
from splatpipe.errors import ExpectedFailure
from splatpipe.jobs import open_job
from splatpipe.train.colmap import Camera, SparseModel, read_model
from splatpipe.train.files import read_keep_mask, read_training_image
from splatpipe.train.priors import Prior, PriorEntry, PriorIndex, write_index, write_prior
from splatpipe.walls.inputs import (FramePick, check_inputs, read_frames_record, read_inputs,
                                    score_image, score_mask, score_size)
from splatpipe.walls.settings import read_settings

from camera_files import (CAMERAS_BIN, IMAGE_BGR, IMAGES_BIN, IMAGES_FOLDER, MASKS_FOLDER,
                          PLANE_CAMERA, PLANE_IMAGES, PLANE_POINTS, POINTS_BIN, SPARSE_FOLDER,
                          CameraRecord, write_camera_solve_output, write_model)

NAMES = tuple(image.name for image in PLANE_IMAGES)  # 000000.jpg to 000003.jpg
PRIOR_NAMES = NAMES[:3]  # the Prior_Frames; 000003.jpg has no Depth_Prior
SECOND = NAMES[1]  # so a message can't name the right file by naming the first

# Each image's Pick, by name: Pick_Numbers that aren't images.bin's order, and seconds of their own.
PICK_NUMBERS = {NAMES[0]: 9, NAMES[1]: 2, NAMES[2]: 5, NAMES[3]: 0}
PICK_SECONDS = {NAMES[0]: 4.5, NAMES[1]: 1.0, NAMES[2]: 2.75, NAMES[3]: 0.25}

# A camera wider than the test config's score_long_edge of 640, and its Score_Size by hand.
WIDE_CAMERA = CameraRecord(camera_id=PLANE_CAMERA.camera_id, width=800, height=600, fx=800.0,
                           fy=800.0, cx=400.0, cy=300.0)
WIDE_SCORE_SIZE = (640, 480)
JPEG_ERROR = 3  # the most a decoded pixel of a one-colour JPEG may stray from it, per channel

FRAMES_AGAIN = "Run the frames step on this job again."


def refused(call: Callable[[], object]) -> str:
    """The message of the ExpectedFailure that `call` raises, once it's checked that the message
    holds no traceback (Req 22.1)."""
    with pytest.raises(ExpectedFailure) as caught:
        call()
    message = str(caught.value)
    assert "Traceback" not in message
    return message


def recorded_opens(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    """A list that gets each path that Path.open is called on from now on. The real method still
    runs."""
    calls: list[Path] = []
    real = Path.open

    def recording(self: Path, *args: Any, **kwargs: Any) -> Any:
        calls.append(self)
        return real(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", recording)
    return calls


def job_folder(new_job: Callable[..., str]) -> Path:
    """A new Job's folder, in the test config's Jobs_Folder."""
    return open_job(new_job(), load_config()).folder


def write_priors(job: Path, names: tuple[str, ...] = PRIOR_NAMES) -> dict[str, Prior]:
    """Write a Depth_Prior_Set for `names` into `job`/work/depth/ with the Prior_Writer, made for
    the Sparse_Model already in work/cameras/sparse/, and give each Prior by name. Prior k has
    an inverse depth of k + 1 everywhere but its top-left pixel, which has no prior."""
    model = read_model(job / "work" / "cameras" / SPARSE_FOLDER)
    camera = model.camera
    folder = job / "work" / "depth"
    folder.mkdir(parents=True, exist_ok=True)
    images = {image.name: image for image in model.images}
    priors: dict[str, Prior] = {}
    entries: list[PriorEntry] = []
    for place, name in enumerate(names):
        mask = np.ones((camera.height, camera.width), dtype=bool)
        mask[0, 0] = False
        inverse_depth = np.where(mask, np.float32(place + 1), np.float32(0)).astype(np.float32)
        normal = np.zeros((camera.height, camera.width, 3), dtype=np.float32)
        normal[mask] = (0.0, 0.0, -1.0)
        prior = Prior(inverse_depth=inverse_depth, normal=normal, mask=mask)
        write_prior(folder / f"{name.removesuffix('.jpg')}.npz", prior)
        priors[name] = prior
        image = images[name]
        entries.append(PriorEntry(name=name, quaternion=tuple(image.quaternion.tolist()),
                                  translation=tuple(image.translation.tolist()), scale=1.0,
                                  offset=0.0, fit_points=30, kept_points=30, median_error=0.01,
                                  metres_per_unit=1.0))
    write_index(folder, PriorIndex(
        camera=(camera.width, camera.height, camera.fx, camera.fy, camera.cx, camera.cy),
        prior_width=camera.width, prior_height=camera.height, metres_per_unit=1.0,
        priors=tuple(entries)))
    return priors


def write_record(job: Path, value: object) -> Path:
    """Write `value` as JSON to `job`/work/frames/frames.json and give its path."""
    path = job / "work" / "frames" / "frames.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def picks(names: tuple[str, ...] = NAMES) -> list[dict[str, Any]]:
    """The Frames_Record's picks for `names`, each with fields the Walls_Step doesn't use too."""
    return [{"number": PICK_NUMBERS[name], "file": name, "frame_number": 10 * place,
             "seconds": PICK_SECONDS[name], "sharpness": 100.0}
            for place, name in enumerate(names)]


def plane_job(new_job: Callable[..., str], camera: CameraRecord = PLANE_CAMERA,
              **options: Any) -> Path:
    """A Job with the plane scene's Camera_Solve_Output, written with `camera` and
    write_camera_solve_output's keyword arguments, its Depth_Prior_Set and its frames.json."""
    job = job_folder(new_job)
    write_camera_solve_output(job, camera, PLANE_IMAGES, PLANE_POINTS, **options)
    write_priors(job)
    write_record(job, {"target_picks": len(NAMES), "picks": picks()})
    return job


def shared_camera(job: Path) -> Camera:
    """The Shared_Camera of `job`'s Sparse_Model, as the Model_Reader reads it."""
    return read_model(job / "work" / "cameras" / SPARSE_FOLDER).camera


def mask_threshold() -> int:
    """mask_threshold from [walls] in the test config, read as the step reads its settings."""
    settings = load_config().step_settings("walls", "default")
    return read_settings(settings, settings.section, settings.preset).mask_threshold


# ---- check_inputs (Req 2.2 to 2.5) ----

# The six paths check_inputs checks, in its order, relative to work/, with the step that writes
# each one.
INPUTS = ((("depth", "depth.json"), "depth"),
          (("cameras", IMAGES_FOLDER), "camera solve"),
          (("cameras", SPARSE_FOLDER, CAMERAS_BIN), "camera solve"),
          (("cameras", SPARSE_FOLDER, IMAGES_BIN), "camera solve"),
          (("cameras", SPARSE_FOLDER, POINTS_BIN), "camera solve"),
          (("frames", "frames.json"), "frames"))


@pytest.mark.parametrize("first", range(len(INPUTS)), ids=[parts[-1] for parts, _ in INPUTS])
def test_check_inputs_refuses_the_first_missing_path_naming_it_and_its_step(
        new_job: Callable[..., str], monkeypatch: pytest.MonkeyPatch, first: int) -> None:
    """With the path `first` and every path after it in check_inputs' order gone, check_inputs
    refuses that path, naming it and the step that writes it. So each of the six is refused in
    turn, once every path before it is there. It opens no file."""
    job = plane_job(new_job)
    for parts, _ in INPUTS[first:]:
        path = job.joinpath("work", *parts)
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    opened = recorded_opens(monkeypatch)
    message = refused(lambda: check_inputs(job))
    parts, writer = INPUTS[first]
    assert str(job.joinpath("work", *parts)) in message
    assert f"which the {writer} step writes" in message
    if writer == "depth":
        assert "the depth step has to run on this job first" in message
        assert "trains as it would without the walls step" in message
    else:
        assert f"Run the {writer} step on this job first." in message
    assert opened == []


def test_check_inputs_passes_a_job_with_every_input_and_opens_no_file(
        new_job: Callable[..., str], monkeypatch: pytest.MonkeyPatch) -> None:
    job = plane_job(new_job)
    opened = recorded_opens(monkeypatch)
    check_inputs(job)
    assert opened == []


def test_check_inputs_refuses_a_depth_json_that_is_a_folder(new_job: Callable[..., str]) -> None:
    """depth.json must be a file, so a folder of that name is refused as missing (Req 2.2)."""
    job = plane_job(new_job)
    index = job / "work" / "depth" / "depth.json"
    index.unlink()
    index.mkdir()
    message = refused(lambda: check_inputs(job))
    assert str(index) in message and "which the depth step writes" in message


# ---- read_frames_record (Req 2.7) ----

def test_read_frames_record_gives_the_picks_in_the_records_order(tmp_path: Path) -> None:
    """Only number, file and seconds are kept, in the record's order, and an int's seconds
    becomes a float."""
    path = write_record(tmp_path, {"mask_used": False, "picks": [
        {"number": 4, "file": "000004.jpg", "seconds": 2, "sharpness": 1.0},
        {"number": 0, "file": "000000.jpg", "seconds": 0.5}]})
    found = read_frames_record(path)
    assert found == (FramePick(number=4, file="000004.jpg", seconds=2.0),
                     FramePick(number=0, file="000000.jpg", seconds=0.5))
    assert type(found[0].seconds) is float


def test_read_frames_record_accepts_an_empty_picks_list(tmp_path: Path) -> None:
    assert read_frames_record(write_record(tmp_path, {"picks": []})) == ()


def good_pick(number: int) -> dict[str, Any]:
    """A Pick that passes every check."""
    return {"number": number, "file": f"{number:06d}.jpg", "seconds": 0.5 * number}


def with_second(**change: Any) -> dict[str, Any]:
    """A record of two good Picks whose second has `change` made, a value of None for a key
    meaning that the key is left out."""
    second = good_pick(1)
    for key, value in change.items():
        if value is None:
            del second[key]
        else:
            second[key] = value
    return {"picks": [good_pick(0), second]}


# Each broken record, and the field its message must name.
BROKEN_RECORDS = [
    ("a list", [], "must hold a JSON object"),
    ("a string", "picks", "must hold a JSON object"),
    ("no picks", {"target_picks": 2}, "has no picks"),
    ("picks an object", {"picks": {"number": 0}}, "picks must be a list of objects"),
    ("picks a string", {"picks": "000000.jpg"}, "picks must be a list of objects"),
    ("a pick a number", {"picks": [good_pick(0), 3]}, "picks[1] must be a JSON object"),
    ("a pick a list", {"picks": [good_pick(0), []]}, "picks[1] must be a JSON object"),
    ("no number", with_second(number=None), "has no picks[1].number"),
    ("no file", with_second(file=None), "has no picks[1].file"),
    ("no seconds", with_second(seconds=None), "has no picks[1].seconds"),
    ("number -1", with_second(number=-1), "picks[1].number must be a whole number, 0 or more"),
    ("number 1.5", with_second(number=1.5), "picks[1].number must be a whole number, 0 or more"),
    ("number true", with_second(number=True),
     "picks[1].number must be a whole number, 0 or more"),
    ("number a string", with_second(number="1"),
     "picks[1].number must be a whole number, 0 or more"),
    ("file a number", with_second(file=1), "picks[1].file must be a file name in quotes"),
    ("file a list", with_second(file=["000001.jpg"]),
     "picks[1].file must be a file name in quotes"),
    ("seconds a string", with_second(seconds="0.5"), "picks[1].seconds must be a finite number"),
    ("seconds true", with_second(seconds=True), "picks[1].seconds must be a finite number"),
    ("seconds a list", with_second(seconds=[0.5]), "picks[1].seconds must be a finite number"),
    ("number twice", {"picks": [good_pick(0), {**good_pick(1), "number": 0}]},
     "picks[1].number is 0, the same as picks[0].number"),
    ("file twice", {"picks": [good_pick(0), {**good_pick(1), "file": "000000.jpg"}]},
     "picks[1].file is '000000.jpg', the same as picks[0].file"),
]


@pytest.mark.parametrize("value, field", [(value, field) for _, value, field in BROKEN_RECORDS],
                         ids=[label for label, _, _ in BROKEN_RECORDS])
def test_read_frames_record_refuses_a_broken_record_naming_the_field(
        tmp_path: Path, value: object, field: str) -> None:
    path = write_record(tmp_path, value)
    message = refused(lambda: read_frames_record(path))
    assert str(path) in message
    assert field in message
    assert message.endswith(FRAMES_AGAIN)


@pytest.mark.parametrize("text", ["NaN", "Infinity", "-Infinity", "1e999"])
def test_read_frames_record_refuses_seconds_that_isnt_finite(tmp_path: Path, text: str) -> None:
    """Python's json reads NaN, Infinity and an overflowing float, so each is refused here."""
    path = tmp_path / "frames.json"
    path.write_text(f'{{"picks": [{{"number": 0, "file": "000000.jpg", "seconds": {text}}}]}}',
                    encoding="utf-8")
    message = refused(lambda: read_frames_record(path))
    assert "picks[0].seconds must be a finite number" in message


def test_read_frames_record_refuses_a_file_that_isnt_json(tmp_path: Path) -> None:
    path = tmp_path / "frames.json"
    path.write_text("{picks", encoding="utf-8")
    message = refused(lambda: read_frames_record(path))
    assert f"The frames record {path} can't be read as JSON" in message
    assert message.endswith(FRAMES_AGAIN)


# ---- read_inputs (Req 2.6, 2.8 to 2.10) ----

def test_read_inputs_joins_each_prior_frame_to_its_pick(new_job: Callable[..., str]) -> None:
    """Each Prior_Frame gets its Pick's number and seconds, its image of images.bin and its
    Depth_Prior, in Pick_Number order. 000003.jpg has a Pick but no Depth_Prior, so it's left
    out (Req 2.9, 2.10)."""
    job = plane_job(new_job)
    model, prior_set, frames = read_inputs(job)
    assert isinstance(model, SparseModel)
    assert len(model.images) == len(NAMES)  # images.bin still holds every image
    assert [frame.name for frame in frames] == sorted(PRIOR_NAMES, key=PICK_NUMBERS.__getitem__)
    images = {image.name: image for image in model.images}
    for frame in frames:
        assert frame.pick_number == PICK_NUMBERS[frame.name]
        assert frame.frame_time == PICK_SECONDS[frame.name]
        assert frame.image is images[frame.name]
        assert frame.entry.name == frame.name
        assert frame.prior is prior_set.priors[frame.name]
        place = PRIOR_NAMES.index(frame.name)
        assert np.all(frame.prior.inverse_depth[frame.prior.mask] == np.float32(place + 1))
        assert not frame.prior.mask[0, 0]
    assert set(prior_set.priors) == set(PRIOR_NAMES)


def test_read_inputs_ignores_a_pick_of_no_image(new_job: Callable[..., str]) -> None:
    """A frames.json with a Pick for an image that isn't in images.bin at all is read, and the
    Pick is simply unused."""
    job = plane_job(new_job)
    record = picks() + [{"number": 30, "file": "000030.jpg", "seconds": 9.0}]
    write_record(job, {"picks": record})
    _, _, frames = read_inputs(job)
    assert sorted(frame.name for frame in frames) == sorted(PRIOR_NAMES)


@pytest.mark.parametrize("record", [picks(tuple(name for name in NAMES if name != SECOND)), []],
                         ids=["no pick for 000001.jpg", "no picks"])
def test_read_inputs_refuses_a_prior_frame_with_no_pick(new_job: Callable[..., str],
                                                        record: list[dict[str, Any]]) -> None:
    job = plane_job(new_job)
    path = write_record(job, {"picks": record})
    message = refused(lambda: read_inputs(job))
    if record:
        assert repr(SECOND) in message
    assert str(path) in message
    assert "frames.json doesn't match the camera solve" in message
    assert message.endswith(f"Run the job again with --fresh: uv run python -m splatpipe.run "
                            f"--job {job.name} --fresh")


def test_read_inputs_gives_the_prior_readers_message_for_a_broken_depth_json(
        new_job: Callable[..., str]) -> None:
    job = plane_job(new_job)
    index = job / "work" / "depth" / "depth.json"
    index.write_text("[1, 2]", encoding="utf-8")
    message = refused(lambda: read_inputs(job))
    assert message.startswith(f"The walls step can't use the depth priors: {index} holds [1, 2], "
                              "but a prior index is a JSON object.")
    assert "Run the depth step on this job again." in message
    assert "trains as it would without the walls step" in message


def test_read_inputs_gives_the_prior_readers_message_for_priors_of_an_earlier_solve(
        new_job: Callable[..., str]) -> None:
    """A Depth_Prior_Set whose pose isn't images.bin's fails one of read_prior_set's checks."""
    job = plane_job(new_job)
    index = job / "work" / "depth" / "depth.json"
    value = json.loads(index.read_text(encoding="utf-8"))
    value["priors"][1]["translation"][0] += 0.5
    index.write_text(json.dumps(value), encoding="utf-8")
    message = refused(lambda: read_inputs(job))
    assert f"{index} has the pose of an earlier camera solve for {SECOND!r}" in message


def test_read_inputs_gives_the_model_readers_message_for_a_cut_short_points_file(
        new_job: Callable[..., str]) -> None:
    job = plane_job(new_job)
    sparse = job / "work" / "cameras" / SPARSE_FOLDER
    write_model(sparse, PLANE_CAMERA, PLANE_IMAGES, PLANE_POINTS, cut={POINTS_BIN: 5})
    message = refused(lambda: read_inputs(job))
    assert message.startswith(f"{sparse / POINTS_BIN} is cut short or isn't a COLMAP binary "
                              "model")
    assert message.endswith("Run the camera solve step on this job again.")


# ---- score_size and score_image (Req 2.11) ----

def test_score_size_scales_the_longer_edge_to_score_long_edge() -> None:
    camera = Camera(camera_id=1, width=800, height=600, fx=800.0, fy=800.0, cx=400.0, cy=300.0)
    assert score_size(camera, 640) == WIDE_SCORE_SIZE
    assert score_size(camera, 800) == (800, 600)
    assert score_size(camera, 1000) == (800, 600)


def test_score_image_scales_a_frame_to_the_score_size(new_job: Callable[..., str]) -> None:
    job = plane_job(new_job, WIDE_CAMERA)
    camera = shared_camera(job)
    size = score_size(camera, load_config().step_settings("walls", "default")["score_long_edge"])
    assert size == WIDE_SCORE_SIZE
    image = score_image(job / "work" / "cameras", SECOND, camera, size)
    assert image.shape == (480, 640, 3) and image.dtype == np.uint8
    assert np.all(np.abs(image.astype(int) - np.array(IMAGE_BGR)) <= JPEG_ERROR)  # BGR order


def test_score_image_keeps_a_small_camera_at_its_size(new_job: Callable[..., str]) -> None:
    job = plane_job(new_job)
    camera = shared_camera(job)
    cameras = job / "work" / "cameras"
    size = score_size(camera, load_config().step_settings("walls", "default")["score_long_edge"])
    assert size == (PLANE_CAMERA.width, PLANE_CAMERA.height)
    image = score_image(cameras, SECOND, camera, size)
    assert np.array_equal(image, read_training_image(cameras / IMAGES_FOLDER, SECOND, camera))


@pytest.mark.parametrize("option", [{"left_out": {SECOND}}, {"junk": {SECOND}},
                                    {"image_sizes": {SECOND: (15, 12)}}],
                         ids=["missing", "junk", "wrong size"])
def test_score_image_refuses_a_broken_frame_with_the_train_steps_message(
        new_job: Callable[..., str], option: dict[str, Any]) -> None:
    job = plane_job(new_job, **option)
    camera = shared_camera(job)
    cameras = job / "work" / "cameras"
    message = refused(lambda: score_image(cameras, SECOND, camera, (16, 12)))
    assert message == refused(lambda: read_training_image(cameras / IMAGES_FOLDER, SECOND, camera))
    assert repr(SECOND) in message


# ---- score_mask (Req 2.12) ----

def test_score_mask_keeps_every_pixel_with_no_mask_file(new_job: Callable[..., str]) -> None:
    job = plane_job(new_job, masked=())
    camera = shared_camera(job)
    keep = score_mask(job / "work" / "cameras", SECOND, camera, (16, 12), mask_threshold())
    assert keep.shape == (12, 16) and keep.dtype == bool and keep.all()


def test_score_mask_reads_a_mask_file_at_the_score_size(new_job: Callable[..., str]) -> None:
    """The scene's mask leaves out its 800 x 600 image's left 400 columns, so at 640 x 480 the
    left 320 columns are left out: column 319 takes mask column 639·800 // 1280 = 399, and
    column 320 takes 641·800 // 1280 = 400."""
    job = plane_job(new_job, WIDE_CAMERA, masked={SECOND})
    keep = score_mask(job / "work" / "cameras", SECOND, shared_camera(job), WIDE_SCORE_SIZE,
                      mask_threshold())
    assert keep.shape == (480, 640) and keep.dtype == bool
    assert not keep[:, :320].any() and keep[:, 320:].all()


def test_score_mask_uses_walls_mask_threshold(new_job: Callable[..., str],
                                              set_walls: Callable[..., None]) -> None:
    """A mask of 100 on its left half and 200 on its right: [walls]' mask_threshold of 128
    leaves the left half out, and a mask_threshold of 50 set in [walls] keeps every pixel."""
    job = plane_job(new_job, masked=())
    grey = np.full((12, 16), 200, dtype=np.uint8)
    grey[:, :8] = 100
    masks = job / "work" / "cameras" / MASKS_FOLDER
    masks.mkdir()
    ok, data = cv2.imencode(".png", grey)
    assert ok
    (masks / f"{SECOND.removesuffix('.jpg')}.png").write_bytes(data.tobytes())
    camera = shared_camera(job)
    assert mask_threshold() == 128
    keep = score_mask(job / "work" / "cameras", SECOND, camera, (16, 12), mask_threshold())
    assert not keep[:, :8].any() and keep[:, 8:].all()
    set_walls(mask_threshold=50)
    assert mask_threshold() == 50
    assert score_mask(job / "work" / "cameras", SECOND, camera, (16, 12), mask_threshold()).all()


def test_score_mask_refuses_an_all_black_mask_naming_walls(new_job: Callable[..., str]) -> None:
    job = plane_job(new_job, fully_masked={SECOND})
    camera = shared_camera(job)
    cameras = job / "work" / "cameras"
    threshold = mask_threshold()
    message = refused(lambda: score_mask(cameras, SECOND, camera, (16, 12), threshold))
    assert f"every pixel is below mask_threshold in [walls], {threshold}" in message
    assert "[train]" not in message
    assert message == refused(lambda: read_keep_mask(
        cameras / MASKS_FOLDER, SECOND, (camera.width, camera.height), 16, 12, threshold,
        section="walls"))


def test_score_mask_refuses_a_mask_of_the_wrong_size_with_the_train_steps_message(
        new_job: Callable[..., str]) -> None:
    job = plane_job(new_job, mask_sizes={SECOND: (15, 12)})
    camera = shared_camera(job)
    cameras = job / "work" / "cameras"
    threshold = mask_threshold()
    message = refused(lambda: score_mask(cameras, SECOND, camera, (16, 12), threshold))
    assert message == refused(lambda: read_keep_mask(
        cameras / MASKS_FOLDER, SECOND, (camera.width, camera.height), 16, 12, threshold))
