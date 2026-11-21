"""Fixtures for every test: the temporary config, dir_link for the link tests, fake_probe and
new_job, which make Jobs without ffprobe, set_fail, which sets `fail` in the test config,
set_frames, which changes its [frames] settings, work_calls, which records whose Fake_Step work
runs, and fake_tools, which puts a FakeTools in place of the Frames_Step's ffprobe and ffmpeg.

The autouse fixture temp_config writes a test config into each test's tmp_path and points
splatpipe.config.CONFIG_PATH at it. So load_config() with no path, and every CLI a test runs in
its own process, reads the test config, whose Jobs_Folder is inside tmp_path. No test can make
a Job in the real data/ folder (Req 14.2). The test config also has [frames] with its three
Preset tables, so a test that runs the Frames_Step without a Config of its own gets settings
(frames spec, Req 9.11).

FakeTools gives the Frames_Step synthetic frames, from a list or from a function that makes each
one on demand, and records what the step asked for. So the step's tests run it through
splatpipe.frames.run on a machine with no ffmpeg (frames spec, Req 13.6, 13.7).
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from itertools import islice
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

import splatpipe.config
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


# A synthetic frame for FakeTools: its Frame_Time in seconds, and its picture, a 2-D uint8 array
# of grey levels of any size.
Pair = tuple[float, np.ndarray]


class FakeTools:
    """A stand-in for the Frames_Step's tools, ffprobe and the ffmpeg decode, so the step's tests
    need no ffmpeg (frames spec, Req 13.6). The fake_tools fixture installs one in their place.

    FakeTools(info, pairs) is built from the VideoInfo that probe gives back and a list of
    synthetic (time, picture) Pairs. FakeTools.stream(info, count, make) is built from a count
    and a function that makes pair k, for k from 0 to count − 1, only when decode reaches it, so
    a test can feed a long stream without holding it (Req 13.7). Each decode call goes through
    the pairs afresh.

    probe and decode take the same arguments as ffmpeg.probe_stream and ffmpeg.decode, in the
    order the step gives them. decode gives on_frame only the pairs whose times a new FrameGate
    for the plan's Used_Part accepts, in order, as the real decode does. Each frame it gives is
    a new uint8 array at the plan's size, made from its pair's picture: grey for a grey plan,
    and a BGR frame with that grey in all three channels for a colour plan. So each Pick shows
    the picture of the pair its Score_Frame came from, and a test can tell which pair became
    which Pick.

    It records each probe call as (video, timeout_seconds) in `probes`, and each decode call's
    plan in `plans`. Two settings change what decode does: with `pick_pass_frames` set, the
    Pick_Pass, the colour plan, gives only its first pick_pass_frames frames, and with `error`
    set, decode raises it once the plan is recorded, before any frame.
    """

    def __init__(self, info: VideoInfo, pairs: Iterable[Pair], *,
                 pick_pass_frames: int | None = None, error: Exception | None = None) -> None:
        self.info = info
        self.pairs = pairs
        self.pick_pass_frames = pick_pass_frames
        self.error = error
        self.probes: list[tuple[Path, float]] = []
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

    def decode(self, plan: DecodePlan, timeout_seconds: float,
               on_frame: Callable[[float, np.ndarray], None]) -> None:
        """Record `plan`, then call on_frame(Frame_Time, frame) for each pair inside its
        Used_Part, in order, keeping none of the frames it gives."""
        from splatpipe.frames.ffmpeg import FrameGate

        self.plans.append(plan)
        if self.error is not None:
            raise self.error
        gate = FrameGate(plan.start, plan.end)
        inside = ((time, picture) for time, picture in self.pairs if gate.accept(time))
        # islice with None gives every pair, so only the Pick_Pass stops early, and only when
        # pick_pass_frames is set.
        for time, picture in islice(inside, self.pick_pass_frames if plan.colour else None):
            on_frame(time, _frame(picture, plan))


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
    FakeTools' probe and decode. The step reads TOOLS each time it runs, so a run through
    splatpipe.frames.run, run_step or the step's CLI in this process uses the stand-in, and the
    test reads what it recorded afterwards. splatpipe.frames is imported here, so only the tests
    that use this fixture need it.
    """
    from splatpipe.frames import Tools

    class Installed(FakeTools):
        """A FakeTools that installs itself once it's built."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            monkeypatch.setattr("splatpipe.frames.TOOLS",
                                Tools(probe=self.probe, decode=self.decode))

    return Installed
