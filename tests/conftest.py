"""Fixtures for every test: the temporary config, dir_link for the link tests, fake_probe and
new_job, which make Jobs without ffprobe, set_fail, which sets `fail` in the test config,
set_frames, which changes its [frames] settings, and work_calls, which records whose Fake_Step
work runs.

The autouse fixture temp_config writes a test config into each test's tmp_path and points
splatpipe.config.CONFIG_PATH at it. So load_config() with no path, and every CLI a test runs in
its own process, reads the test config, whose Jobs_Folder is inside tmp_path. No test can make
a Job in the real data/ folder (Req 14.2). The test config also has [frames] with its three
Preset tables, so a test that runs the Frames_Step without a Config of its own gets settings
(frames spec, Req 9.11).
"""

import json
import os
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

import splatpipe.config
from splatpipe.make import CONTAINER_FAMILY, make_job
from splatpipe.probe import ProbeResult
from splatpipe.step import StepContext

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


def config_text(jobs_folder: Path, fail: Mapping[str, str] | None = None,
                frames: Mapping[str, Any] | None = None) -> str:
    """The test config as TOML. Each Step_Name in `fail` gets that `fail` value, and every other
    Fake_Step section gets fail = "none". [frames] comes last, with the keys in `frames` changed
    as frames_text says. The Jobs_Folder is written with forward slashes, so a Windows backslash
    can't become a TOML escape."""
    modes = fail or {}
    jobs = JOBS_TOML.format(jobs_folder=jobs_folder.as_posix())
    fakes = "".join(STEP_TOML.format(name=name, fail=modes.get(name, "none"))
                    for name in STEP_NAMES)
    return jobs + fakes + frames_text(frames or {})


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
    fail = "none", and set_fail() sets them all back. [frames] is back to the test config's values
    too. load_config() reads the file each time it's called, so the next run that loads the config
    sees the change.
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
    name is back to the test config's value, and every fail is back to "none". The next run that
    loads the config sees the change. Each test gets a new test config, so the next test sees the
    test config's values again.
    """
    def rewrite(**values: Any) -> None:
        temp_config.write_text(config_text(tmp_path / "jobs", frames=values), encoding="utf-8")

    return rewrite


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
