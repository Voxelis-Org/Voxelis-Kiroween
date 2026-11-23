"""Fixtures for every test: the temporary config, dir_link for the link tests, fake_probe and
new_job, which make Jobs without ffprobe, set_fail, which sets `fail` in the test config,
set_frames and set_train, which change its [frames] and [train] settings, set_clean, which sets
its [clean] settings, work_calls, which records whose Fake_Step work runs, fake_tools, which puts
a FakeTools in place of the Frames_Step's ffprobe and ffmpeg, fake_gpu, which puts a FakeGpu in
place of the Train_Step's GPU_Module, and no_cap_deadline, which keeps SPLATPIPE_CAP_DEADLINE out
of every test.

The autouse fixture temp_config writes a test config into each test's tmp_path and points
splatpipe.config.CONFIG_PATH at it. So load_config() with no path, and every CLI a test runs in
its own process, reads the test config, whose Jobs_Folder is inside tmp_path. No test can make
a Job in the real data/ folder (Req 14.2). The test config also has [frames] and [train], each
with its three Preset tables, so a test that runs the Frames_Step or the Train_Step without a
Config of its own gets settings (frames spec, Req 9.11; training spec, Req 9.15). It has [clean]
and its three Preset tables too, from CLEAN_SHARED and CLEAN_PRESETS.

The autouse fixture no_cap_deadline removes SPLATPIPE_CAP_DEADLINE from the environment before
every test, so a Cap_Deadline left in the shell or by another test can't change one. A test that
needs it sets it with monkeypatch.setenv (training spec, Req 14.11).

FakeTools gives the Frames_Step synthetic frames, from a list or from a function that makes each
one on demand, and records what the step asked for. It can also report that ffmpeg has no
filters, or fail the first pass with the Tone_Map, for the HDR tests. So the step's tests run it
through splatpipe.frames.run on a machine with no ffmpeg (frames spec, Req 2.4, 2.5, 13.6, 13.7).

FakeGpu, the GPU_Stand_In, gives the Train_Step the GPU check's facts, finds each point's nearest
points by brute force and fakes the training, giving back the Start_Splats it was given, and
records what the step asked for. It can also raise from either GPU call, or give back one splat
or every splat with a NaN opacity. So the step's tests run it through splatpipe.train.run on a
machine with no GPU and no torch (training spec, Req 11.6, 14.4).
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Callable, Collection, Iterable, Iterator, Mapping
from dataclasses import dataclass, field, fields, replace
from itertools import islice
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pytest

import splatpipe.config
from splatpipe.make import CONTAINER_FAMILY, make_job
from splatpipe.probe import ProbeResult
from splatpipe.step import StepContext

# Named in FakeTools' and FakeGpu's type hints only, so only their tests import splatpipe.frames
# and splatpipe.train.
if TYPE_CHECKING:
    from datetime import datetime

    from splatpipe.frames.ffmpeg import DecodePlan, VideoInfo
    from splatpipe.train.gpu_api import GpuInfo, TrainResult
    from splatpipe.train.scene import Scene
    from splatpipe.train.settings import TrainSettings
    from splatpipe.train.splats import Splats

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

# The test config's 35 [train] Shared_Settings: the real values (training spec, Req 9.2). Each
# value has the TOML type it has in the real config.
TRAIN = {
    "long_edge": 1600,
    "mask_threshold": 128,
    "seed": 42,
    "sh_degree_interval": 1000,
    "scene_extent_factor": 1.1,
    "near_plane_factor": 0.03,
    "start_opacity": 0.5,
    "start_scale_factor": 0.1,
    "start_scale_neighbours": 3,
    "neighbour_chunk_pairs": 100000000,
    "means_lr": 0.00016,
    "means_lr_final": 0.01,
    "scales_lr": 0.005,
    "quats_lr": 0.001,
    "opacities_lr": 0.05,
    "sh_dc_lr": 0.0025,
    "sh_rest_lr": 0.000125,
    "adam_eps": 1e-15,
    "exposure_lr": 0.001,
    "l1_weight": 0.8,
    "ssim_weight": 0.2,
    "opacity_reg": 0.01,
    "scale_reg": 0.01,
    "erank_reg": 0.01,
    "erank_epsilon": 1e-5,
    "erank_start_fraction": 0.23333333333333334,
    "refine_start_step": 500,
    "refine_every": 100,
    "refine_stop_step": 25000,
    "refine_stop_fraction": 0.9,
    "clean_reserve_minutes": 10,
    "save_margin_minutes": 5,
    "gpu_memory_slack": 0.9,
    "min_vcpus": 4,
    "min_memory_gb": 32,
}

# The test config's [train] Preset_Settings, by Preset. steps and max_splats are tiny, so the
# GPU_Stand_In's work stays tiny and a test of a dozen points fits the budget. min_gpu_memory_gb
# keeps its real values, so the GPU_Fit_Check's byte counts mean what they do on the machine.
TRAIN_PRESETS = {
    "fast": {"steps": 6, "max_splats": 50, "min_gpu_memory_gb": 24},
    "default": {"steps": 8, "max_splats": 60, "min_gpu_memory_gb": 24},
    "quality": {"steps": 10, "max_splats": 70, "min_gpu_memory_gb": 48},
}

# The test config's [clean] and its three Preset tables, as tomllib reads them: the real keys
# with the real values, except where a comment gives the real one. clean_toml and clean_settings
# in tests/splats.py copy these dicts and never change them.
CLEAN_SHARED: dict[str, object] = {
    "train_ply": "train.ply",
    "clean_ply": "clean.ply",
    "clean_sog": "clean.sog",
    "min_opacity": 0.005,
    "faint_cube_divisions": 128,
    "faint_min_per_cube": 8,
    "faint_min_paint_share": 0.10,
    "extent_trim_percent": 1,
    "neighbours": 8,
    "isolation_ratio": 8.0,
    "max_removed_fraction": 0.70,
    "min_kept_splats": 1,  # real: 10000. So a splat of a few dozen Gaussians can pass.
    "warn_kept_fraction": 0.5,
    "max_source_splats": 8_000_000,
    "max_workers": 1,  # real: 4. So the tests start no extra threads.
    # real: 3600. So the real splat-transform run has a timeout, like every other subprocess in
    # the suite (Req 10.7).
    "splat_transform_timeout_seconds": 60,
}
# real: 3 bands in each Preset, and 10, 10 and 20 iterations. With 0, 1 and 3 bands, the
# argument-list test sees a different band count for each Preset, and a test can reach the
# Req 5.5 warning with a splat of fewer bands. 1 iteration keeps the real run quick.
CLEAN_PRESETS: dict[str, dict[str, object]] = {
    "fast": {"sog_sh_bands": 0, "sh_iterations": 1},
    "default": {"sog_sh_bands": 1, "sh_iterations": 1},
    "quality": {"sog_sh_bands": 3, "sh_iterations": 1},
}


def toml_value(value: object) -> str:
    """`value` as a TOML value: a bool as true or false, text in double quotes as JSON writes it,
    and a number as Python writes it, which TOML reads back as the same int or float."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        # json.dumps escapes the quote, the backslash and every character below U+0020 the way
        # TOML reads them back. TOML can't hold DEL as it is either, so that's escaped too.
        return json.dumps(value, ensure_ascii=False).replace("\x7f", "\\u007f")
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


def train_text(train: Mapping[str, Any]) -> str:
    """[train] and its three Preset tables as TOML, with each key in `train` set to its value
    and every other key at its value in TRAIN or TRAIN_PRESETS.

    A Shared_Setting takes its new value, like {"seed": 7}. A Preset_Setting, steps, max_splats
    or min_gpu_memory_gb, takes a dict of the Presets it changes, like {"max_splats": {"fast": 4}},
    and the other Presets keep theirs.
    """
    preset_keys = TRAIN_PRESETS["fast"].keys()  # every Preset table has the same keys
    shared = {**TRAIN, **{key: value for key, value in train.items() if key not in preset_keys}}
    presets = {f"train.{preset}": {key: train.get(key, {}).get(preset, value)
                                   for key, value in table.items()}
               for preset, table in TRAIN_PRESETS.items()}
    return "".join(f"\n[{name}]\n"
                   + "".join(f"{key} = {toml_value(value)}\n" for key, value in table.items())
                   for name, table in {"train": shared, **presets}.items())


def clean_toml(changes: Mapping[str, object]) -> str:
    """[clean] and its three Preset tables as TOML: CLEAN_SHARED and CLEAN_PRESETS, with
    `changes` made.

    A key of [clean] is given as it is, like "neighbours", and a key of a Preset table after its
    Preset and two underscores, like "fast__sog_sh_bands". A value of None leaves the key out.
    """
    tables = {"clean": dict(CLEAN_SHARED),
              **{f"clean.{preset}": dict(table) for preset, table in CLEAN_PRESETS.items()}}
    for name, value in changes.items():
        preset, _, key = name.rpartition("__")
        table = tables[f"clean.{preset}" if preset else "clean"]
        if value is None:
            table.pop(key, None)
        else:
            table[key] = value
    return "".join(f"\n[{header}]\n" + "".join(f"{key} = {toml_value(value)}\n"
                                                for key, value in table.items())
                   for header, table in tables.items())


def config_text(jobs_folder: Path, fail: Mapping[str, str] | None = None,
                frames: Mapping[str, Any] | None = None,
                train: Mapping[str, Any] | None = None,
                clean: Mapping[str, object] | None = None) -> str:
    """The test config as TOML. Each Step_Name in `fail` gets that `fail` value, and every other
    Fake_Step section gets fail = "none". [frames] and then [train] come next, with the keys in
    `frames` and `train` changed as frames_text and train_text say. [clean] and its Preset tables
    come last, with the changes in `clean` made, as clean_toml reads them. The Jobs_Folder is
    written with forward slashes, so a Windows backslash can't become a TOML escape."""
    modes = fail or {}
    jobs = JOBS_TOML.format(jobs_folder=jobs_folder.as_posix())
    fakes = "".join(STEP_TOML.format(name=name, fail=modes.get(name, "none"))
                    for name in STEP_NAMES)
    return (jobs + fakes + frames_text(frames or {}) + train_text(train or {})
            + clean_toml(clean or {}))


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


@pytest.fixture(autouse=True)
def no_cap_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove SPLATPIPE_CAP_DEADLINE from the environment for every test, so a Cap_Deadline left
    in the shell or by another test can't reach one (training spec, Req 14.11). A test that needs
    it sets it with monkeypatch.setenv.

    The name is written out, not taken from splatpipe.train, so this file imports no
    splatpipe.train.
    """
    monkeypatch.delenv("SPLATPIPE_CAP_DEADLINE", raising=False)


@pytest.fixture
def set_fail(tmp_path: Path, temp_config: Path) -> Callable[..., None]:
    """Give set_fail(**modes), which rewrites the test config with `fail` set to the given value
    for each Step_Name it names, like set_fail(fake_b="expected").

    Each call writes the whole file again, so every section it doesn't name is back to
    fail = "none", and set_fail() sets them all back. [frames], [train] and [clean] are back to the
    test config's values too. load_config() reads the file each time it's called, so the next run
    that loads the config sees the change.
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
    name is back to the test config's value, in [train] too, and every fail is back to "none". The
    next run that loads the config sees the change. Each test gets a new test config, so the next
    test sees the test config's values again.
    """
    def rewrite(**values: Any) -> None:
        temp_config.write_text(config_text(tmp_path / "jobs", frames=values), encoding="utf-8")

    return rewrite


@pytest.fixture
def set_train(tmp_path: Path, temp_config: Path) -> Callable[..., None]:
    """Give set_train(**values), which rewrites the test config with each [train] key it names
    set to the given value, like set_train(seed=7). steps, max_splats and min_gpu_memory_gb,
    which each Preset has its own of, take a dict of the Presets to change, like
    set_train(max_splats={"fast": 4}).

    Like set_fail and set_frames, each call writes the whole file again from config_text, so
    every key it doesn't name is back to the test config's value, in [frames] too, and every fail
    is back to "none". The next run that loads the config sees the change. Each test gets a new
    test config, so the next test sees the test config's values again.
    """
    def rewrite(**values: Any) -> None:
        temp_config.write_text(config_text(tmp_path / "jobs", train=values), encoding="utf-8")

    return rewrite


@pytest.fixture
def set_clean(tmp_path: Path, temp_config: Path) -> Callable[..., None]:
    """Give set_clean(**values), which rewrites the test config with the [clean] values given,
    like set_clean(neighbours=3). A key of a Preset table comes after its Preset and two
    underscores, like set_clean(fast__sog_sh_bands=1), and None leaves a key out, like
    set_clean(neighbours=None), so a test can check a missing setting.

    Each call writes the whole file again, the way set_fail does, so every setting it doesn't
    name is back to its value in CLEAN_SHARED or CLEAN_PRESETS, every fail is back to "none",
    [frames] is back to the test config's values, and set_clean() sets them all back. A set_fail
    or set_frames call after it puts [clean] back the same way.
    """
    def rewrite(**values: object) -> None:
        temp_config.write_text(config_text(tmp_path / "jobs", clean=values), encoding="utf-8")

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


# FakeGpu's GPU memory by default: the 23,028 MiB that nvidia-smi shows for a "24 GB" A10G, which
# the GPU_Fit_Check passes for a min_gpu_memory_gb of 24 (training spec, Req 2.8).
A10G_BYTES = 24_146_608_128

# The splats whose opacity FakeGpu's nan_opacity makes NaN: the first one, or every one.
NAN_SPLATS = {"one": slice(1), "every": slice(None)}


class FakeGpu:
    """A stand-in for the Train_Step's GPU_Module, the GPU_Stand_In, so the step's tests run its
    whole flow on a CPU with no torch (training spec, Req 11.6, 14.4). The fake_gpu fixture
    installs one as the step's GPU hook.

    FakeGpu(info) is built from the GpuInfo that probe gives back, by default CUDA, gsplat 1.5.3
    and A10G_BYTES. Its four methods take the same arguments as the Gpu protocol's in
    splatpipe.train.gpu_api. It records each call in `calls`, in order, as the method's name and
    a tuple of the arguments it was given, and keeps the scene and the Start_Splats that train
    was last given in `scene` and `start`.

    - probe gives back the info.
    - nearest_distances is brute force in numpy: every pair's distance with np.linalg.norm, each
      point's own distance set to infinity, then the `neighbours` smallest of each row with
      np.sort, ascending, in float64. That's fine for the few hundred points a test draws.
    - train is a fake Trainer. It counts Iterations up to settings.steps, and before each one it
      stops when the real should_stop(utc_now(), stop) says so, or once it has counted
      `stop_after`. It gives back a TrainResult of the Start_Splats as float32 copies, the
      Exposure_Fix `exposure` for every view, [I | 0] when that's None, the Iterations it
      counted, and the last one's SH_Degree from the real sh_degree, 0 when it counted none.
    - peak_memory_bytes gives back `peak_bytes`.

    With `nearest_distances_error` or `train_error` set, that GPU call raises it once the call is
    recorded, like GpuOutOfMemory(812) or NonFiniteLoss(...). With `nan_opacity` set to "one",
    train gives back the first splat with a NaN opacity, and with "every", every splat.
    """

    def __init__(self, info: GpuInfo | None = None, *, stop_after: int | None = None,
                 exposure: np.ndarray | None = None, peak_bytes: int = 0,
                 nearest_distances_error: Exception | None = None,
                 train_error: Exception | None = None,
                 nan_opacity: Literal["one", "every"] | None = None) -> None:
        from splatpipe.train.gpu_api import GpuInfo

        if info is None:
            info = GpuInfo(cuda=True, gsplat_version="1.5.3", total_memory_bytes=A10G_BYTES)
        self.info = info
        self.stop_after = stop_after
        self.exposure = exposure
        self.peak_bytes = peak_bytes
        self.nearest_distances_error = nearest_distances_error
        self.train_error = train_error
        self.nan_opacity = nan_opacity
        self.calls: list[tuple[str, tuple[Any, ...]]] = []
        self.scene: Scene | None = None
        self.start: Splats | None = None

    def probe(self) -> GpuInfo:
        """Record the call and give back the info."""
        self.calls.append(("probe", ()))
        return self.info

    def nearest_distances(self, points: np.ndarray, neighbours: int,
                          chunk_pairs: int) -> np.ndarray:
        """Record the call, raise nearest_distances_error when it's set, then give each of the
        (N, 3) points' distances to its `neighbours` nearest other points, as an (N, neighbours)
        float64 array in ascending order, by brute force. chunk_pairs changes nothing here."""
        self.calls.append(("nearest_distances", (points, neighbours, chunk_pairs)))
        if self.nearest_distances_error is not None:
            raise self.nearest_distances_error
        points = np.asarray(points, dtype=np.float64)
        gaps = np.linalg.norm(points[:, np.newaxis] - points[np.newaxis], axis=-1)
        np.fill_diagonal(gaps, np.inf)
        return np.sort(gaps, axis=1)[:, :neighbours]

    def train(self, scene: Scene, start: Splats, settings: TrainSettings,
              stop: datetime | None) -> TrainResult:
        """Record the call and keep `scene` and `start`, raise train_error when it's set, then
        count the Iterations and give back the TrainResult the class describes."""
        from splatpipe.train.gpu_api import TrainResult
        from splatpipe.train.schedules import sh_degree

        self.calls.append(("train", (scene, start, settings, stop)))
        self.scene, self.start = scene, start
        if self.train_error is not None:
            raise self.train_error
        steps_run = self._count(settings.steps, stop)
        last = sh_degree(steps_run - 1, settings.sh_degree_interval) if steps_run else 0
        return TrainResult(splats=self._splats(start), exposures=self._exposures(len(scene.views)),
                           steps_run=steps_run, sh_degree=last)

    def peak_memory_bytes(self) -> int:
        """Record the call and give back peak_bytes."""
        self.calls.append(("peak_memory_bytes", ()))
        return self.peak_bytes

    def _count(self, steps: int, stop: datetime | None) -> int:
        """The Iterations the fake Trainer runs: up to `steps`, stopping before one when the real
        should_stop(utc_now(), stop) says so, or once it has counted stop_after."""
        from splatpipe.train.deadline import should_stop, utc_now

        counted = 0
        for _ in range(steps):
            if should_stop(utc_now(), stop) or counted == self.stop_after:
                break
            counted += 1
        return counted

    def _splats(self, start: Splats) -> Splats:
        """The Start_Splats as float32 copies, so `start` stays as it was given, with the
        opacities that nan_opacity names made NaN."""
        splats = replace(start, **{item.name: getattr(start, item.name).astype(np.float32)
                                   for item in fields(start)})
        if self.nan_opacity is not None:
            splats.opacity_logits[NAN_SPLATS[self.nan_opacity]] = np.nan
        return splats

    def _exposures(self, views: int) -> np.ndarray:
        """A (views, 3, 4) float32 array holding the Exposure_Fix `exposure` for every view, or
        [I | 0] when that's None."""
        fix = np.eye(3, 4) if self.exposure is None else self.exposure
        return np.broadcast_to(fix, (views, 3, 4)).astype(np.float32)


@pytest.fixture
def fake_gpu(monkeypatch: pytest.MonkeyPatch) -> type[FakeGpu]:
    """Give fake_gpu, which builds a FakeGpu as the class does, with fake_gpu(info, ...), and
    installs it as the Train_Step's GPU_Module for this test. Building another one installs that
    one instead.

    Installing is monkeypatch.setattr("splatpipe.train.GPU", lambda: gpu), with gpu the FakeGpu
    just built. The step reads GPU each time it runs, so a run through splatpipe.train.run,
    run_step, the step's CLI or run_job in this process uses the stand-in, and the test reads
    what it recorded afterwards. monkeypatch imports splatpipe.train to set GPU, and FakeGpu's
    methods import what they need from it when they're called, so only the tests that use this
    fixture import splatpipe.train.
    """

    class Installed(FakeGpu):
        """A FakeGpu that installs itself once it's built."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            monkeypatch.setattr("splatpipe.train.GPU", lambda: self)

    return Installed
