"""Fixtures for every test: the temporary config, dir_link for the link tests, fake_probe and
new_job, which make Jobs without ffprobe, cameras_job, which makes a Job with a tiny valid
work/frames/ for the Cameras_Step, set_fail, which sets `fail` in the test config, set_frames
and set_cameras, which change its [frames] and [cameras] settings, colmap_options, which gives
the COLMAP options of the camera solve's options.md, work_calls, which records whose Fake_Step
work runs, and fake_tools, which puts a FakeTools in place of the Frames_Step's ffprobe and
ffmpeg.

The autouse fixture temp_config writes a test config into each test's tmp_path and points
splatpipe.config.CONFIG_PATH at it. So load_config() with no path, and every CLI a test runs in
its own process, reads the test config, whose Jobs_Folder is inside tmp_path. No test can make
a Job in the real data/ folder (Req 14.2). The test config also has [frames] with its three
Preset tables, so a test that runs the Frames_Step without a Config of its own gets settings
(frames spec, Req 9.11). It has [cameras] with its three Preset tables for the Cameras_Step:
the real config's, read once, with min_placed_frames at 1 and every *_timeout_seconds at 60 on
top (cameras spec, Req 12.15, C5). set_cameras changes any of its keys, or leaves one out when
given MISSING.

colmap_options parses the seven code blocks of .kiro/skills/colmap-cli/references/options.md
with this file's own code, never with splatpipe/cameras/settings.py, and sorts each option the
way the cameras design does: a Fixed_Constant, a path, --SequentialMatching.overlap, or an
Option_Key with the key that O11's naming rule gives it. It's the oracle of the settings test
and of Property 1.

cameras_job makes a Job with new_job and writes into its work/frames/ what the Frames_Step
leaves there: a JPEG per Pick in images/ and its Grey_Copy in grey/, mask.png when the Job uses
the Mask, and a frames.json with every field of the Frames_Record, under the names in
splatpipe/frames/files.py. So the Cameras_Step's tests start from a finished Frames_Step without
running it or ffmpeg (cameras spec, Req 16.5).

FakeTools gives the Frames_Step synthetic frames, from a list or from a function that makes each
one on demand, and records what the step asked for. It can also report that ffmpeg has no
filters, or fail the first pass with the Tone_Map, for the HDR tests. So the step's tests run it
through splatpipe.frames.run on a machine with no ffmpeg (frames spec, Req 2.4, 2.5, 13.6, 13.7).
"""

from __future__ import annotations

import enum
import json
import os
import re
import shlex
import sys
import tomllib
from collections.abc import Callable, Collection, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from itertools import islice
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final, Literal

import cv2
import numpy as np
import pytest

import splatpipe.config
from splatpipe.config import PRESETS, REPO_ROOT, load_config
from splatpipe.jobs import open_job
from splatpipe.jsonio import write_json
from splatpipe.make import CONTAINER_FAMILY, make_job
from splatpipe.probe import ProbeResult
from splatpipe.step import StepContext

if TYPE_CHECKING:  # named in FakeTools' type hints only, so only its tests import splatpipe.frames
    from splatpipe.frames.ffmpeg import DecodePlan, VideoInfo

# The names the tests run the Fake_Step under. Each gets a section like the real [fake].
STEP_NAMES = ("fake", "fake_a", "fake_b", "fake_c")

# The test config's [jobs]: the real values, except that the Jobs_Folder is in tmp_path and the
# size limit is tiny, so a size test needs only a file one byte over.
JOBS_TOML = """\
[jobs]
jobs_folder = "{jobs_folder}"
default_preset = "fast"
max_video_bytes = 1024
max_video_seconds = 600
min_used_seconds = 3
ffprobe_timeout_seconds = 30
"""

# One Fake_Step section with its three Preset tables, with the keys and values of the real [fake],
# except `fail`, which config_text fills in.
STEP_TOML = """
[{name}]
fail = "{fail}"
file_bytes = 16

[{name}.fast]
files = 1

[{name}.default]
files = 2

[{name}.quality]
files = 3
"""

# The test config's [frames] Shared_Settings: the real values, except that min_picks is 1 and both
# timeouts are 60 s, so tiny clips pass. Each value has the TOML type it has in the real config.
FRAMES = {
    "score_width": 320,
    "tonemap_npl": 100,
    "tonemap_desat": 0.0,
    "still_motion": 1.0,
    "motion_window_seconds": 1,
    "min_still_seconds": 2,
    "blip_seconds": 0.5,
    "jolt_ratio": 0.6,
    "jolt_window_seconds": 1,
    "mask_samples": 240,
    "mask_ratio": 0.2,
    "mask_min_coverage": 0.002,
    "mask_max_coverage": 0.25,
    "jpeg_quality": 95,
    "long_edge": 1600,
    "seconds_per_pick": 2,
    "max_picks": 400,
    "min_picks": 1,
    "min_vcpus": 16,
    "min_memory_gb": 16,
    "ffmpeg_timeout_seconds": 60,
    "ffprobe_timeout_seconds": 60,
}

# The test config's base_picks, the one Preset_Setting of [frames], for each Preset. They're tiny
# for the same reason as min_picks.
BASE_PICKS = {"fast": 2, "default": 3, "quality": 4}

# The real config's [cameras] with its Preset tables, read once by path, because temp_config
# points CONFIG_PATH at the test config.
REAL_CAMERAS = tomllib.loads((REPO_ROOT / "config.toml").read_text(encoding="utf-8"))["cameras"]

# The test config's [cameras] Shared_Settings: the real config's, with two test values on top
# (cameras spec, C5): min_placed_frames at 1, so tiny Jobs pass the Placed_Check, and every
# *_timeout_seconds at 60. Each real value keeps the type tomllib gave it, so toml_value writes it
# back as the same TOML type. The test values are integers.
CAMERAS = {**{key: value for key, value in REAL_CAMERAS.items() if key not in PRESETS},
           "min_placed_frames": 1,
           **{key: 60 for key in REAL_CAMERAS if key.endswith("_timeout_seconds")}}

# The test config's near_overlap, the one Preset_Setting of [cameras], for each Preset: the real
# config's.
NEAR_OVERLAP = {preset: REAL_CAMERAS[preset]["near_overlap"] for preset in PRESETS}


class Missing(enum.Enum):
    """The type of MISSING."""

    MISSING = "missing"


# What set_cameras takes in place of a value to leave that key out of the test config.
MISSING: Final = Missing.MISSING


def toml_value(value: object) -> str:
    """`value` as a TOML value: a bool as true or false, text in double quotes as JSON writes it,
    and a number as Python writes it, which TOML reads back as the same int or float."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value)
    return repr(value)


def frames_text(frames: Mapping[str, Any]) -> str:
    """[frames] and its three Preset tables as TOML, with each key in `frames` set to its value
    and every other key at its value in FRAMES or BASE_PICKS.

    A Shared_Setting takes its new value, like {"mask_samples": 3}. base_picks takes a dict of
    the Presets it changes, like {"base_picks": {"fast": 50}}, and the other Presets keep theirs.
    """
    shared = {**FRAMES, **{key: value for key, value in frames.items() if key != "base_picks"}}
    base_picks = {**BASE_PICKS, **frames.get("base_picks", {})}
    return ("\n[frames]\n"
            + "".join(f"{key} = {toml_value(value)}\n" for key, value in shared.items())
            + "".join(f"\n[frames.{preset}]\nbase_picks = {toml_value(value)}\n"
                      for preset, value in base_picks.items()))


def table_text(name: str, values: Mapping[str, Any]) -> str:
    """The TOML table [name], with a line for each key in `values` whose value isn't MISSING."""
    return f"\n[{name}]\n" + "".join(f"{key} = {toml_value(value)}\n"
                                     for key, value in values.items() if value is not MISSING)


def cameras_text(cameras: Mapping[str, Any]) -> str:
    """[cameras] and its three Preset tables as TOML, with each key in `cameras` set to its value
    and every other key at its value in CAMERAS or NEAR_OVERLAP. A key set to MISSING is left out.

    A Shared_Setting takes its new value, like {"min_placed_frames": 5}. near_overlap takes a dict
    of the Presets it changes, like {"near_overlap": {"quality": 3}}, and the other Presets keep
    theirs. near_overlap set to MISSING is left out of every Preset table.
    """
    shared = {**CAMERAS, **{key: value for key, value in cameras.items() if key != "near_overlap"}}
    changed = cameras.get("near_overlap", {})
    near_overlap = (dict.fromkeys(NEAR_OVERLAP, MISSING) if changed is MISSING
                    else {**NEAR_OVERLAP, **changed})
    return table_text("cameras", shared) + "".join(
        table_text(f"cameras.{preset}", {"near_overlap": value})
        for preset, value in near_overlap.items())


def config_text(jobs_folder: Path, fail: Mapping[str, str] | None = None,
                frames: Mapping[str, Any] | None = None,
                cameras: Mapping[str, Any] | None = None) -> str:
    """The test config as TOML. Each Step_Name in `fail` gets that `fail` value, and every other
    Fake_Step section gets fail = "none". [frames] and then [cameras] come last, with the keys in
    `frames` and `cameras` changed as frames_text and cameras_text say. The Jobs_Folder is
    written with forward slashes, so a Windows backslash can't become a TOML escape."""
    modes = fail or {}
    jobs = JOBS_TOML.format(jobs_folder=jobs_folder.as_posix())
    fakes = "".join(STEP_TOML.format(name=name, fail=modes.get(name, "none"))
                    for name in STEP_NAMES)
    return jobs + fakes + frames_text(frames or {}) + cameras_text(cameras or {})


@pytest.fixture(autouse=True)
def temp_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Write the test config and point splatpipe.config.CONFIG_PATH at it. Gives the file's path.

    The Jobs_Folder is tmp_path / "jobs", which isn't made here. The file goes in a folder of its
    own, tmp_path / "config", because a test in test_config.py needs tmp_path / "config.toml" to
    be missing.
    """
    path = tmp_path / "config" / "config.toml"
    path.parent.mkdir()
    path.write_text(config_text(tmp_path / "jobs"), encoding="utf-8")
    monkeypatch.setattr(splatpipe.config, "CONFIG_PATH", path)
    return path


@pytest.fixture
def set_fail(tmp_path: Path, temp_config: Path) -> Callable[..., None]:
    """Give set_fail(**modes), which rewrites the test config with `fail` set to the given value
    for each Step_Name it names, like set_fail(fake_b="expected").

    Each call writes the whole file again, so every section it doesn't name is back to
    fail = "none", and set_fail() sets them all back. [frames] and [cameras] are back to the test
    config's values too. load_config() reads the file each time it's called, so the next run that
    loads the config sees the change.
    """
    def rewrite(**modes: str) -> None:
        temp_config.write_text(config_text(tmp_path / "jobs", fail=modes), encoding="utf-8")

    return rewrite


@pytest.fixture
def set_frames(tmp_path: Path, temp_config: Path) -> Callable[..., None]:
    """Give set_frames(**values), which rewrites the test config with each [frames] key it names
    set to the given value, like set_frames(mask_samples=3). base_picks, which each Preset has
    its own of, takes a dict of the Presets to change, like set_frames(base_picks={"fast": 50}).

    Like set_fail, each call writes the whole file again from config_text, so every key it doesn't
    name is back to the test config's value, [cameras] is too, and every fail is back to "none".
    The next run that loads the config sees the change. Each test gets a new test config, so the
    next test sees the test config's values again.
    """
    def rewrite(**values: Any) -> None:
        temp_config.write_text(config_text(tmp_path / "jobs", frames=values), encoding="utf-8")

    return rewrite


@pytest.fixture
def set_cameras(tmp_path: Path, temp_config: Path) -> Callable[..., None]:
    """Give set_cameras(**values), which rewrites the test config with each [cameras] key it names
    set to the given value, like set_cameras(min_placed_frames=5), or left out when the value is
    MISSING. near_overlap, which each Preset has its own of, takes a dict of the Presets to
    change, like set_cameras(near_overlap={"quality": 3}), and set_cameras(near_overlap=MISSING)
    leaves it out of every Preset table.

    Like set_frames, each call writes the whole file again from config_text, so every key it
    doesn't name is back to the test config's value, [frames] is too, and every fail is back to
    "none". The next run that loads the config sees the change. Each test gets a new test config,
    so the next test sees the test config's values again.
    """
    def rewrite(**values: Any) -> None:
        temp_config.write_text(config_text(tmp_path / "jobs", cameras=values), encoding="utf-8")

    return rewrite


# The camera solve's COLMAP command lines, one bash code block per command (colmap-cli skill).
OPTIONS_MD = REPO_ROOT / ".kiro" / "skills" / "colmap-cli" / "references" / "options.md"

# A bash code block of options.md, from the line after its ```bash to its closing ```.
BASH_BLOCK = re.compile(r"^```bash\n(.*?)^```", re.MULTILINE | re.DOTALL)

# The options that the cameras design's "The Fixed_Constants" holds in code, but for the paths,
# which are sorted on their own: the four log options of every command, and each command's own.
# options.md gives each its fixed value, except --SequentialMatching.quadratic_overlap, which is 1
# there and in the Far_Match_Run but 0 in the Near_Match_Run (A1).
LOG_OPTIONS = ("--log_target", "--log_level", "--log_severity", "--log_color")
FIXED_OPTIONS = {
    "feature_extractor": ("--camera_mode", "--FeatureExtraction.type",
                          "--FeatureExtraction.use_gpu"),
    "sequential_matcher": ("--FeatureMatching.type", "--FeatureMatching.use_gpu",
                           "--SequentialMatching.loop_detection",
                           "--SequentialMatching.quadratic_overlap"),
    "view_graph_calibrator": (),
    "global_mapper": ("--GlobalMapper.gp_use_gpu", "--GlobalMapper.ba_ceres_use_gpu",
                      "--GlobalMapper.ba_backend"),
    "mapper": ("--Mapper.ba_use_gpu", "--Mapper.ba_local_backend", "--Mapper.ba_global_backend"),
    "image_undistorter": ("--output_type",),
    "image_undistorter_standalone": (),
}

# The folders that every path in options.md starts from: the frames step's work folder and the
# camera solve's.
PATH_FOLDERS = ("$FRAMES", "$OUT")

# The two commands whose options with no group share the undistortion_ keys (O11).
UNDISTORTERS = ("image_undistorter", "image_undistorter_standalone")

# The Option_Keys that O11 names by hand: Req 12.2's camera_model, and default_random_seed, which
# every command takes and which keeps its own name.
OWN_KEYS = {"--ImageReader.camera_model": "camera_model",
            "--default_random_seed": "default_random_seed"}


@dataclass(frozen=True)
class ColmapOption:
    """One option of a command's block in options.md, with its value's text and its sort.

    `sort` is "fixed" for a Fixed_Constant, "path" for a path, "overlap" for
    --SequentialMatching.overlap, which near_overlap or far_overlap sets, and "key" for an
    Option_Key, whose key is `key`.
    """

    option: str  # COLMAP's option, like "--SiftExtraction.max_num_features"
    text: str  # its value as a shell reads the block, like "8192" or "$OUT/database.db"
    sort: Literal["fixed", "path", "overlap", "key"]
    key: str | None = None  # the Option_Key, like "sift_extraction_max_num_features", or None


def option_key(command: str, option: str) -> str:
    """The Option_Key that O11's naming rule gives `option` of `command`.

    --Group.name gets <group>_<name>, with the group in snake case. An option with no group gets
    its command's name in front, or undistortion_ for the two undistorters. The whole key is in
    lower case, so --GlobalMapper.ba_refine_points3D gets global_mapper_ba_refine_points3d.
    OWN_KEYS names the two keys the rule doesn't give.
    """
    if option in OWN_KEYS:
        return OWN_KEYS[option]
    group, _, name = option.removeprefix("--").rpartition(".")
    if group:
        prefix = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", group)  # TwoViewGeometry: Two_View_Geometry
    else:
        prefix = "undistortion" if command in UNDISTORTERS else command
    return f"{prefix}_{name}".lower()


def colmap_option(command: str, option: str, text: str) -> ColmapOption:
    """`option` of `command`, with its value's text, sorted the way the cameras design sorts it."""
    if option in LOG_OPTIONS or option in FIXED_OPTIONS[command]:
        return ColmapOption(option, text, "fixed")
    if text.partition("/")[0] in PATH_FOLDERS:
        return ColmapOption(option, text, "path")
    if option == "--SequentialMatching.overlap":
        return ColmapOption(option, text, "overlap")
    return ColmapOption(option, text, "key", option_key(command, option))


def parse_options(text: str) -> dict[str, tuple[ColmapOption, ...]]:
    """The options of each command in `text`, the text of options.md, by command in the order of
    its code blocks, with each block's options in the block's order.

    A block is one shell command: colmap, the command, then each option followed by its value,
    with a backslash ending each line but the last. It's split as a shell splits it, so the
    backslash joins each line to the next and the quotes around a path come off, but $FRAMES and
    $OUT stay as they are.
    """
    commands: dict[str, tuple[ColmapOption, ...]] = {}
    for block in BASH_BLOCK.findall(text):
        words = shlex.split(block.replace("\\\n", ""))  # words[0] is colmap
        command, rest = words[1], words[2:]
        commands[command] = tuple(colmap_option(command, option, value)
                                  for option, value in zip(rest[::2], rest[1::2]))
    return commands


@pytest.fixture(scope="session")
def colmap_options() -> Mapping[str, tuple[ColmapOption, ...]]:
    """Give the options of the seven COLMAP commands of options.md as parse_options reads them,
    each a ColmapOption with its sort and, for an Option_Key, its key.

    It's the oracle of the settings test and of Property 1, so it reads only options.md, never
    splatpipe/cameras/settings.py. It's session-scoped, so a Hypothesis test can take it, and
    read-only, because every test shares it.
    """
    return MappingProxyType(parse_options(OPTIONS_MD.read_text(encoding="utf-8")))


def make_dir_link(link: Path, target: Path) -> None:
    """Make `link` a link to the folder `target`, or skip the test, saying why, if none can be made.

    It tries a symlink first. Windows allows one only with admin rights or Developer Mode, so there
    it falls back on a directory junction, which needs no extra rights. Path.resolve() follows both.
    """
    try:
        os.symlink(target, link, target_is_directory=True)
        return
    except OSError as error:
        failures = [f"symlink: {error}"]
    if sys.platform == "win32":
        import _winapi  # in the standard library on Windows only

        try:
            _winapi.CreateJunction(str(target), str(link))
            return
        except OSError as error:
            failures.append(f"junction: {error}")
    pytest.skip(f"No folder link can be made here, so this test can't run ({'; '.join(failures)}).")


@pytest.fixture
def dir_link() -> Callable[[Path, Path], None]:
    """Give dir_link(link, target), which makes `link` a link to the folder `target`."""
    return make_dir_link


# What the stand-in Video_Probe reads when a test chooses nothing else: an MP4 with a video
# stream, 10 s long, which the test config accepts.
GOOD_VIDEO = ProbeResult(format_name=CONTAINER_FAMILY, has_video=True, video_seconds=10.0)

FIXED_ID = "20261119-142530-7f3a9c"  # the Job_Id new_job gives every Job it makes


@dataclass
class FakeProbe:
    """A stand-in Video_Probe, so no unit test needs ffprobe (Req 14.3). Every call gets `result`
    back and is recorded in `calls`, as (path, timeout_seconds)."""

    result: ProbeResult = GOOD_VIDEO
    calls: list[tuple[Path, float]] = field(default_factory=list)

    def __call__(self, path: Path, timeout_seconds: float) -> ProbeResult:
        self.calls.append((path, timeout_seconds))
        return self.result


@pytest.fixture
def fake_probe() -> type[FakeProbe]:
    """Give fake_probe(result), which builds a FakeProbe. With no result, it reads GOOD_VIDEO."""
    return FakeProbe


@pytest.fixture
def new_job(tmp_path: Path) -> Callable[..., str]:
    """Give new_job(**options), which makes a Job and returns its Job_Id.

    The video is a few bytes at tmp_path / "clip.mp4", outside the Jobs_Folder. make_job gets a
    FakeProbe that reads GOOD_VIDEO, a new_id that gives FIXED_ID, and `options`: start, end,
    preset or config.
    """
    def make(**options: Any) -> str:
        video = tmp_path / "clip.mp4"
        video.write_bytes(b"a few bytes")
        return make_job(video, probe=FakeProbe(), new_id=lambda: FIXED_ID, **options)

    return make


@pytest.fixture
def cameras_job(new_job: Callable[..., str]) -> Callable[..., str]:
    """Give cameras_job(picks=8, mask_used=False, width=16, height=12), which makes a Job with
    new_job, writes a tiny valid work/frames/ into it, as the Frames_Step leaves it, and returns
    its Job_Id (cameras spec, Req 16.5). Like new_job, it can make one Job per test, because
    each Job gets FIXED_ID.

    images/ gets a JPEG of seeded noise at width × height for each of the `picks` Picks, named
    pick_name(0) onwards, and grey/ gets each one's Grey_Copy, from grey_copy, under the same
    name, both at the test config's jpeg_quality. With mask_used, mask.png is a grey PNG at the
    same size, 0 in the top left corner's height // 4 rows and width // 4 columns, at least one
    of each, and 255 elsewhere. frames.json comes last and holds every field of the
    Frames_Record, in frames_record's order: the Picks spread evenly over the Job's Used_Part,
    which is kept whole, each its own Score_Frame, target_picks equal to `picks`, and
    mask_coverage the Mask's share of 0s, or 0.0 with no Mask. With picks=0, images/, grey/ and
    `picks` are empty.

    The names come from splatpipe/frames/files.py, the contract between the two steps. It's
    imported here, so only the tests that use this fixture import splatpipe.frames.
    """
    from splatpipe.frames.files import GREY, IMAGES, MASK, RECORD, grey_copy, pick_name

    def write_image(path: Path, image: np.ndarray, *params: int) -> None:
        """Encode `image` with cv2.imencode in the format of `path`'s suffix, then write it."""
        ok, data = cv2.imencode(path.suffix, image, list(params))
        assert ok, f"cv2.imencode couldn't encode {path}"
        path.write_bytes(data.tobytes())

    def write_picks(frames: Path, picks: int, width: int, height: int) -> None:
        """Write each Pick's JPEG to images/ and its Grey_Copy's to grey/ in `frames`."""
        quality = (cv2.IMWRITE_JPEG_QUALITY, FRAMES["jpeg_quality"])
        for name in (IMAGES, GREY):
            (frames / name).mkdir(parents=True)
        for number in range(picks):
            pick = np.random.default_rng(number).integers(0, 256, (height, width, 3),
                                                          dtype=np.uint8)
            write_image(frames / IMAGES / pick_name(number), pick, *quality)
            write_image(frames / GREY / pick_name(number), grey_copy(pick), *quality)

    def write_mask(path: Path, width: int, height: int) -> float:
        """Write the Mask to `path` and give its Mask_Coverage, its share of 0s."""
        mask = np.full((height, width), 255, dtype=np.uint8)
        mask[:max(1, height // 4), :max(1, width // 4)] = 0
        write_image(path, mask)
        return float(np.mean(mask == 0))

    def make(*, picks: int = 8, mask_used: bool = False, width: int = 16,
             height: int = 12) -> str:
        job_id = new_job()
        job = open_job(job_id, load_config())
        frames = job.folder / "work" / "frames"
        write_picks(frames, picks, width, height)
        coverage = write_mask(frames / MASK, width, height) if mask_used else 0.0
        start, end = job.record.start_seconds, job.record.end_seconds  # the Used_Part
        write_json(frames / RECORD, {
            "colour_transfer": "bt709", "tone_mapped": False,
            "still_start_seconds": 0.0, "still_end_seconds": 0.0,
            "kept_start_seconds": start, "kept_end_seconds": end,
            "target_picks": picks, "jolt_filter_skipped": False,
            "pick_width": width, "pick_height": height,
            "mask_coverage": coverage, "mask_used": mask_used,
            "mask_file": MASK if mask_used else None,
            "picks": [{"number": number, "file": pick_name(number), "frame_number": number,
                       "seconds": start + (end - start) * number / picks, "sharpness": 100.0,
                       "motion": 0.0 if number == 0 else 2.0, "contrast": 40.0}
                      for number in range(picks)]})
        return job_id

    return make


@pytest.fixture
def work_calls(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Patch splatpipe.fake.work to record each call's Step_Name and then run the real work.
    Gives the list of names, in the order of the calls.

    run_step looks work up each time a Step runs, so the list sees every run in this process,
    through the Job_Runner, splatpipe.fake.run or the Step's own CLI. splatpipe.fake is imported
    here, so only the tests that use this fixture need the Fake_Step.
    """
    from splatpipe import fake

    real_work = fake.work
    names: list[str] = []

    def recorded_work(step: StepContext) -> None:
        names.append(step.name)
        real_work(step)

    monkeypatch.setattr(fake, "work", recorded_work)
    return names


# A synthetic frame for FakeTools: its Frame_Time in seconds, and its picture, a 2-D uint8 array
# of grey levels of any size.
Pair = tuple[float, np.ndarray]


class FakeTools:
    """A stand-in for the Frames_Step's tools, ffprobe, ffmpeg's filter check and the ffmpeg
    decode, so the step's tests need no ffmpeg (frames spec, Req 13.6). The fake_tools fixture
    installs one in their place.

    FakeTools(info, pairs) is built from the VideoInfo that probe gives back and a list of
    synthetic (time, picture) Pairs. FakeTools.stream(info, count, make) is built from a count
    and a function that makes pair k, for k from 0 to count − 1, only when decode reaches it, so
    a test can feed a long stream without holding it (Req 13.7). Each decode call goes through
    the pairs afresh.

    probe, has_filters and decode take the same arguments as ffmpeg.probe_stream,
    ffmpeg.has_filters and ffmpeg.decode, in the order the step gives them. has_filters says
    this ffmpeg has every filter it's asked about, or, with `no_filters` set, that it hasn't.
    decode gives on_frame only the pairs whose times a new FrameGate for the plan's Used_Part
    accepts, in order, as the real decode does. Each frame it gives is a new uint8 array at the
    plan's size, made from its pair's picture: grey for a grey plan, and a BGR frame with that
    grey in all three channels for a colour plan. So each Pick shows the picture of the pair its
    Score_Frame came from, and a test can tell which pair became which Pick.

    It records each probe call as (video, timeout_seconds) in `probes`, each has_filters call as
    (names, timeout_seconds) in `filter_checks`, with the names as a tuple, and each decode
    call's plan in `plans`. Three settings change what decode does. With `pick_pass_frames`
    set, the Pick_Pass, the colour plan, gives only its first pick_pass_frames frames. With
    `tone_map_failure` set to (frames, last_line), the first plan with a ToneMap gives its first
    `frames` frames and then raises DecodeFailed(last_line), as the real decode does when ffmpeg
    exits with a non-zero code, and every later plan runs as usual. With `error` set, decode
    raises it once the plan is recorded, before any frame.
    """

    def __init__(self, info: VideoInfo, pairs: Iterable[Pair], *,
                 pick_pass_frames: int | None = None, no_filters: bool = False,
                 tone_map_failure: tuple[int, str] | None = None,
                 error: Exception | None = None) -> None:
        self.info = info
        self.pairs = pairs
        self.pick_pass_frames = pick_pass_frames
        self.no_filters = no_filters
        self.tone_map_failure = tone_map_failure
        self.error = error
        self.probes: list[tuple[Path, float]] = []
        self.filter_checks: list[tuple[tuple[str, ...], float]] = []
        self.plans: list[DecodePlan] = []

    @classmethod
    def stream(cls, info: VideoInfo, count: int, make: Callable[[int], Pair],
               **settings: Any) -> FakeTools:
        """A FakeTools whose pair k, for k from 0 to count − 1, is make(k), made only when decode
        reaches it. `settings` are FakeTools' own, like pick_pass_frames."""
        return cls(info, _MadePairs(count, make), **settings)

    def probe(self, video: Path, timeout_seconds: float) -> VideoInfo:
        """Record the call and give back the VideoInfo."""
        self.probes.append((video, timeout_seconds))
        return self.info

    def has_filters(self, names: Collection[str], timeout_seconds: float) -> bool:
        """Record the call, then say this ffmpeg has every filter in `names`, or, with no_filters
        set, that it hasn't."""
        self.filter_checks.append((tuple(names), timeout_seconds))
        return not self.no_filters

    def decode(self, plan: DecodePlan, timeout_seconds: float,
               on_frame: Callable[[float, np.ndarray], None]) -> None:
        """Record `plan`, then call on_frame(Frame_Time, frame) for each pair inside its
        Used_Part, in order, keeping none of the frames it gives. With tone_map_failure set, the
        first plan with a ToneMap fails after its first few frames, as the class says."""
        from splatpipe.frames.ffmpeg import DecodeFailed, FrameGate

        failure = self._failure(plan)  # before `plan` is recorded, so plans holds only earlier ones
        self.plans.append(plan)
        if self.error is not None:
            raise self.error
        gate = FrameGate(plan.start, plan.end)
        inside = ((time, picture) for time, picture in self.pairs if gate.accept(time))
        if failure is not None:
            frames, last_line = failure
            for time, picture in islice(inside, frames):
                on_frame(time, _frame(picture, plan))
            raise DecodeFailed(last_line)  # a new error each time, so this FakeTools holds none
        # islice with None gives every pair, so only the Pick_Pass stops early, and only when
        # pick_pass_frames is set.
        for time, picture in islice(inside, self.pick_pass_frames if plan.colour else None):
            on_frame(time, _frame(picture, plan))

    def _failure(self, plan: DecodePlan) -> tuple[int, str] | None:
        """tone_map_failure when `plan` is the first plan with a ToneMap, else None. Call it
        before `plan` is recorded."""
        first = plan.tone_map is not None and all(p.tone_map is None for p in self.plans)
        return self.tone_map_failure if first else None


@dataclass(frozen=True)
class _MadePairs:
    """FakeTools.stream's pairs, make(0) to make(count − 1), made one at a time each time they're
    gone through, so nothing holds them all."""

    count: int
    make: Callable[[int], Pair]

    def __iter__(self) -> Iterator[Pair]:
        return map(self.make, range(self.count))


def _frame(picture: np.ndarray, plan: DecodePlan) -> np.ndarray:
    """A new uint8 frame at the plan's size made from `picture`, each pixel taking the picture's
    pixel whose area holds the pixel's centre: that grey for a grey plan, and a BGR frame with that
    grey in all three channels for a colour plan."""
    height, width = picture.shape
    rows = (2 * np.arange(plan.height) + 1) * height // (2 * plan.height)
    columns = (2 * np.arange(plan.width) + 1) * width // (2 * plan.width)
    grey = picture[np.ix_(rows, columns)]
    return np.repeat(grey[:, :, np.newaxis], 3, axis=2) if plan.colour else grey


@pytest.fixture
def fake_tools(monkeypatch: pytest.MonkeyPatch) -> type[FakeTools]:
    """Give fake_tools, which builds a FakeTools as the class does, with fake_tools(info, pairs,
    ...) or fake_tools.stream(info, count, make, ...), and installs it as the Frames_Step's tools
    for this test. Building another one installs that one instead.

    Installing is monkeypatch.setattr("splatpipe.frames.TOOLS", ...) with a Tools of the
    FakeTools' probe, has_filters and decode. The step reads TOOLS each time it runs, so a run
    through splatpipe.frames.run, run_step or the step's CLI in this process uses the stand-in,
    and the test reads what it recorded afterwards. splatpipe.frames is imported here, so only
    the tests that use this fixture need it.
    """
    from splatpipe.frames import Tools

    class Installed(FakeTools):
        """A FakeTools that installs itself once it's built."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            monkeypatch.setattr("splatpipe.frames.TOOLS",
                                Tools(probe=self.probe, has_filters=self.has_filters,
                                      decode=self.decode))

    return Installed
