"""Fixtures for every test: the temporary config, dir_link for the link tests, fake_probe and
new_job, which make Jobs without ffprobe, cameras_job, which makes a Job with a tiny valid
work/frames/ for the Cameras_Step, set_fail, which sets `fail` in the test config, set_frames,
set_cameras and set_train, which change its [frames], [cameras] and [train] settings, set_clean,
which sets its [clean] settings, colmap_options, which gives the COLMAP options of the camera
solve's options.md, work_calls, which records whose Fake_Step work runs, fake_tools, which puts a
FakeTools in place of the Frames_Step's ffprobe and ffmpeg, fake_colmap, which gives FakeColmap,
a stand-in for the colmap program, and puts each one it builds in place of the Cameras_Step's
tools, fake_gpu, which puts a FakeGpu in place of the Train_Step's GPU_Module, and
no_cap_deadline, which keeps SPLATPIPE_CAP_DEADLINE out of every test.

The autouse fixture temp_config writes a test config into each test's tmp_path and points
splatpipe.config.CONFIG_PATH at it. So load_config() with no path, and every CLI a test runs in
its own process, reads the test config, whose Jobs_Folder is inside tmp_path. No test can make
a Job in the real data/ folder (Req 14.2). The test config also has [frames], [cameras] and
[train], each with its three Preset tables, so a test that runs the Frames_Step, the Cameras_Step
or the Train_Step without a Config of its own gets settings (frames spec, Req 9.11; cameras spec,
Req 12.15; training spec, Req 9.15). Its [cameras] is the real config's, read once, with
min_placed_frames at 1 and every *_timeout_seconds at 60 on top (cameras spec, C5). set_cameras
changes any of its keys, or leaves one out when given MISSING. It has [clean] and its three
Preset tables too, from CLEAN_SHARED and CLEAN_PRESETS.

The autouse fixture no_cap_deadline removes SPLATPIPE_CAP_DEADLINE from the environment before
every test, so a Cap_Deadline left in the shell or by another test can't change one. A test that
needs it sets it with monkeypatch.setenv (training spec, Req 14.11).

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

FakeColmap stands in for the colmap program in every run of the Cameras_Step, so its tests run
on a machine with no COLMAP (cameras spec, Req 16.5). fake_colmap gives the class, and installs
each FakeColmap it builds as the step's tools, so a run through splatpipe.cameras.run, run_step
or the step's CLI in this process uses it, and finds colmap unless it's built with
installed=False. A test of the solve flow passes its run to solve instead, in place of
colmap.run_colmap. The script says which runs time out, which exit with a given code and line,
what the Calibration does to the Calibration_Copy, how many Picks each Model of each mapper
places, whether the mappers turn every other camera upside down, and what the image_undistorter
gets wrong. FakeColmap records each run and writes its Command_Log. Every other run exits with 0
after writing what the real command would: the Database with its one RADIAL or RADIAL_FISHEYE
camera, the Calibration_Copy's changed camera, each mapper's Models, written with the step's own
Model_Writer, with a RADIAL camera and Poses along a walk seen through a fixed rigid move, the
Undistortion's PINHOLE model and a JPEG per image, and the Mask_Warp's three-channel PNG with a
grey edge around every black region.

FakeGpu, the GPU_Stand_In, gives the Train_Step the GPU check's facts, finds each point's nearest
points by brute force and fakes the training, giving back the Start_Splats it was given, and
records what the step asked for. It can also raise from either GPU call, or give back one splat
or every splat with a NaN opacity. So the step's tests run it through splatpipe.train.run on a
machine with no GPU and no torch (training spec, Req 11.6, 14.4).
"""

from __future__ import annotations

import enum
import json
import math
import os
import re
import shlex
import sqlite3
import struct
import sys
import time
import tomllib
from collections.abc import Callable, Collection, Iterable, Iterator, Mapping, Sequence
from contextlib import closing
from dataclasses import dataclass, field, fields, replace
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

# Named in FakeTools', FakeColmap's and FakeGpu's type hints only, so only the tests that run a
# FakeTools import splatpipe.frames.ffmpeg, only those that use a FakeColmap import
# splatpipe.cameras, and only those that use a FakeGpu import splatpipe.train.
if TYPE_CHECKING:
    from datetime import datetime

    from splatpipe.cameras.colmap import ColmapRun, RunResult
    from splatpipe.cameras.model import Camera, Model
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
                cameras: Mapping[str, Any] | None = None,
                train: Mapping[str, Any] | None = None,
                clean: Mapping[str, object] | None = None) -> str:
    """The test config as TOML. Each Step_Name in `fail` gets that `fail` value, and every other
    Fake_Step section gets fail = "none". [frames], [cameras] and then [train] come next, with the
    keys in `frames`, `cameras` and `train` changed as frames_text, cameras_text and train_text
    say. [clean] and its Preset tables come last, with the changes in `clean` made, as clean_toml
    reads them. The Jobs_Folder is
    written with forward slashes, so a Windows backslash can't become a TOML escape."""
    modes = fail or {}
    jobs = JOBS_TOML.format(jobs_folder=jobs_folder.as_posix())
    fakes = "".join(STEP_TOML.format(name=name, fail=modes.get(name, "none"))
                    for name in STEP_NAMES)
    return (jobs + fakes + frames_text(frames or {}) + cameras_text(cameras or {})
            + train_text(train or {}) + clean_toml(clean or {}))


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
    fail = "none", and set_fail() sets them all back. [frames], [cameras], [train] and [clean] are
    back to the test config's values too. load_config() reads the file each time it's called, so
    the next run that loads the config sees the change.
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
    name is back to the test config's value, in [cameras], [train] and [clean] too, and every fail
    is back to "none". The next run that loads the config sees the change. Each test gets a new
    test config, so the next test sees the test config's values again.
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
    doesn't name is back to the test config's value, [frames], [train] and [clean] are too, and
    every fail is back to "none". The next run that loads the config sees the change. Each test
    gets a new test config, so the next test sees the test config's values again.
    """
    def rewrite(**values: Any) -> None:
        temp_config.write_text(config_text(tmp_path / "jobs", cameras=values), encoding="utf-8")

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


# COLMAP's camera model ids of the two Camera_Models, RADIAL and RADIAL_FISHEYE (cameras design,
# F2, D19).
COLMAP_MODEL_IDS = {"RADIAL": 3, "RADIAL_FISHEYE": 9}

# The cameras table of a COLMAP 4.2.1 database, as its scene/database_sqlite.cc makes it (F12).
# params holds the camera's parameters as little-endian doubles.
CAMERAS_TABLE = """\
CREATE TABLE IF NOT EXISTS cameras
   (camera_id            INTEGER  PRIMARY KEY AUTOINCREMENT  NOT NULL,
    model                INTEGER                             NOT NULL,
    width                INTEGER                             NOT NULL,
    height               INTEGER                             NOT NULL,
    params               BLOB,
    prior_focal_length   INTEGER                             NOT NULL)"""

# What FakeColmap's view_graph_calibrator can do to the Calibration_Copy's camera (O5, F11), and
# the share of the old focal length that a Calibration that worked gives the camera.
CALIBRATIONS = ("worked", "rejected", "no_pairs")
CALIBRATED_SHARE = 0.9

# The fake mappers' scene, in a level world whose up is (0, -1, 0). The camera of the Pick at
# place p of --image_path, by name, stands at (WALK_STEP * p, 0, 0) and looks level at
# WALK_TARGET, around which lie the few WALK_POINTS, each with its colour and its error.
WALK_STEP = 0.5
WALK_TARGET = (2.0, 0.0, 6.0)
WALK_POINTS = np.array(WALK_TARGET) + np.array([(-1.0, -0.5, 0.0), (1.0, -0.5, 0.5),
                                                (0.0, 0.5, -0.5), (0.5, 0.25, 1.0)])
WALK_COLOURS = ((200, 40, 40), (40, 200, 40), (40, 40, 200), (200, 200, 40))
WALK_ERRORS = (0.4, 0.6, 0.5, 0.7)
# The fixed rigid move that takes the level world into COLMAP's frame: a turn of TILT_DEGREES
# about TILT_AXIS, which isn't vertical, then TILT_SHIFT. So the Levelling has a tilt to remove.
TILT_AXIS = (1.0, 0.3, 0.6)
TILT_DEGREES = 35.0
TILT_SHIFT = (1.5, -0.5, 2.0)
# The fake mappers' RADIAL camera: f = MAPPED_FOCAL times the long edge, the principal point at
# the centre, and a little distortion, k1 and k2, neither 0 (D19).
MAPPED_FOCAL = 1.1
MAPPED_DISTORTION = (0.02, -0.01)
# The fake image_undistorter's camera: PINHOLE, COLMAP's model id 1 (F2), with fx = fy = the
# RADIAL camera's f (F18), and UNDISTORTED_SHRINK pixels smaller each way than the camera it
# undistorts.
PINHOLE_ID = 1
UNDISTORTED_SHRINK = 2
# The grey of the edge that the fake image_undistorter_standalone puts around every black region
# of the Mask, like the values that bilinear sampling gives between black and white.
WARPED_EDGE = 128
# What FakeColmap's find_colmap gives in place of shutil.which's answer, unless the FakeColmap is
# built with installed=False. No program is there, and none is run: FakeColmap's run stands in
# for every COLMAP_Command.
FAKE_COLMAP = "/fake/bin/colmap"


@dataclass(kw_only=True)
class FakeColmap:
    """A stand-in for the colmap program, for every run of the Cameras_Step, so the step's tests
    need no COLMAP (cameras spec, Req 16.5). A test builds one with its script through
    fake_colmap, which installs it as the step's tools. A test of the solve flow passes its run
    to solve instead, in place of colmap.run_colmap.

    find_colmap() stands in for shutil.which(PROGRAM): it gives FAKE_COLMAP, or None when the
    FakeColmap is built with installed=False, so the step fails as when colmap isn't on the PATH
    (Req 2.7).

    run(colmap_run) records the ColmapRun in `runs` and writes its Command_Log, a line with the
    command line. Then it follows the script for colmap_run.name:
    - a name in `timeouts` gives a RunResult with exit_code None, after the run's Time_Limit;
    - a name in `exits`, mapped to (code, line), writes the line to the Command_Log too and gives
      that code;
    - every other run exits with 0, after writing what the real command would write.
    A run in `timeouts` or `exits` writes nothing else. The RunResult's seconds are the time the
    stand-in took, or the Time_Limit for a Timeout, and its last_line is the Command_Log's
    Last_Error_Line, read as run_colmap reads it.

    What each run that exits with 0 writes, at the paths on its command line:
    - feature_extractor: a SQLite database at --database_path whose cameras table, laid out as
      COLMAP 4.2.1's, holds one camera of --ImageReader.camera_model at the Pick_Size, the size
      of the first Pick in --image_path. For RADIAL, (f, cx, cy, k1, k2), f is
      --ImageReader.default_focal_length_factor, 1.2 in the config, times the long edge, the
      principal point is the centre, k1 and k2 are 0 and prior_focal_length is 0. For
      RADIAL_FISHEYE, the parameters are --ImageReader.camera_params, the Fisheye_Start, and
      prior_focal_length is 1.
    - sequential_matcher_near and sequential_matcher_far: nothing.
    - view_graph_calibrator, on the database at --database_path, which must exist: with
      `calibration` "worked", f, params[0], becomes CALIBRATED_SHARE times the old f and
      prior_focal_length 1; with "rejected", only prior_focal_length becomes 1; with "no_pairs",
      nothing changes.
    - global_mapper and mapper: one Model folder, 0/, 1/ and so on, in --output_path for each
      count in `global_models` or `incremental_models`, written with the step's write_model.
      None, or no counts, writes no Model. --output_path must be a folder already, as COLMAP's
      mappers need. Model k places the next that many Picks of --image_path by name: Model 0
      the first ones, Model 1 the ones after them, and so on. Each image's id is its Pick's
      place from 1, as the Database gives it. Each Model has one RADIAL camera with a little
      distortion, from MAPPED_FOCAL and MAPPED_DISTORTION, and the few WALK_POINTS. Its Poses lie
      along the walk of WALK_STEP and WALK_TARGET, seen through the fixed rigid move of
      TILT_AXIS, TILT_DEGREES and TILT_SHIFT, so the Levelling has a tilt to remove. With
      upside_down, every other camera, the one at each odd place, is turned a half turn about
      its own view axis, so it's upside down. Then the cameras' mean up vector is about 0 long
      for an even number of them and a third long at most for an odd number from 3, below the real
      up_agreement of 0.5, so the Levelling keeps the COLMAP_Frame (Req 9.3). Each image's 2D
      points are where the points project through the camera, leaving out its distortion, and
      each point's track holds every image of its Model.
    - image_undistorter: the Model at --input_path, read with the step's read_model, written to
      --output_path/sparse/ with the step's write_model, with each RADIAL camera turned PINHOLE
      and UNDISTORTED_SHRINK pixels smaller each way. fx and fy are both the RADIAL camera's f
      (F18), and the principal point scales with the size, as COLMAP's UndistortCamera does (F7).
      The images, with their Poses and 2D points, and the points stay as they are. Each image
      gets a tiny JPEG under its name in --output_path/images/: its Colour_Pick from
      --image_path, resized to its camera's size. Three flags make it go wrong in the ways the
      step's checks refuse (Req 8.2): keep_camera_model keeps each camera as it is, RADIAL, as
      COLMAP does for a camera with no distortion (F9); second_camera adds a copy of the first
      camera with the next id, which no image uses; and leave_out_image leaves the Model's last
      image out, so it gets no JPEG either.
    - image_undistorter_standalone: the picture that the camera line in --input_file names, read
      from --image_path, resized to the size the fake image_undistorter gives the line's camera,
      with WARPED_EDGE on each pixel next to a black one that isn't black itself, like bilinear
      sampling. It's written as a three-channel PNG under the same name in --output_path (F8).
    Both undistorters make --output_path, as COLMAP's do. Resizing is by the nearest pixel.

    The defaults take a default cameras_job, of 8 Picks, all the way: the Calibration works, each
    mapper writes one Model that places all 8 Picks, and both undistorters write what the step
    reads, so the step is done. The fake image_undistorter needs Picks of at least 3 × 3 pixels.
    A script that names a run this stand-in doesn't know, a calibration it doesn't know or a
    count below 0 raises ValueError, and so does a mapper whose counts add up to more Picks than
    --image_path holds.
    """

    timeouts: Collection[str] = ()  # the names of the runs that reach their Time_Limit
    exits: Mapping[str, tuple[int, str]] = field(default_factory=dict)  # name: (code, line)
    calibration: Literal["worked", "rejected", "no_pairs"] = "worked"
    global_models: Sequence[int] | None = (8,)  # the Picks each Model of global_mapper places
    incremental_models: Sequence[int] | None = (8,)  # the Picks each Model of mapper places
    upside_down: bool = False  # both mappers turn every other camera upside down
    keep_camera_model: bool = False  # image_undistorter keeps each camera as it is (F9)
    second_camera: bool = False  # image_undistorter adds a second camera
    leave_out_image: bool = False  # image_undistorter leaves the Model's last image out
    installed: bool = True  # find_colmap finds colmap, or with False gives None (Req 2.7)
    runs: list[ColmapRun] = field(default_factory=list, init=False)  # every run, in order

    def __post_init__(self) -> None:
        """Refuse a script that names a run this stand-in doesn't know, a calibration it doesn't
        know or a count below 0, so a typo can't pass for a real script."""
        known = self._writers()
        unknown = sorted({*self.timeouts, *self.exits} - set(known))
        if unknown:
            raise ValueError(f"FakeColmap has no run named {', '.join(unknown)}. Its runs are "
                             f"{', '.join(known)}.")
        if self.calibration not in CALIBRATIONS:
            raise ValueError(f"FakeColmap's calibration is {self.calibration!r}, but it must be "
                             f"one of {', '.join(CALIBRATIONS)}.")
        for counts in (self.global_models, self.incremental_models):
            if counts is not None and any(type(count) is not int or count < 0
                                          for count in counts):
                raise ValueError("FakeColmap's Model counts must be whole numbers, 0 or more, "
                                 f"but they're {counts}.")

    def find_colmap(self) -> str | None:
        """Stand in for shutil.which(PROGRAM): FAKE_COLMAP, or None when built with
        installed=False."""
        return FAKE_COLMAP if self.installed else None

    def run(self, colmap_run: ColmapRun) -> RunResult:
        """Stand in for run_colmap(colmap_run), as the class says."""
        from splatpipe.cameras.colmap import RunResult, last_error_line

        start = time.perf_counter()
        self.runs.append(colmap_run)
        writers = self._writers()
        name = colmap_run.name
        if name not in writers:
            raise AssertionError(f"FakeColmap doesn't stand in for the run {name}, only for "
                                 f"{', '.join(writers)}.")
        colmap_run.log.write_text(f"FakeColmap ran in place of: {shlex.join(colmap_run.args)}\n",
                                  encoding="utf-8")
        exit_code: int | None
        if name in self.timeouts:
            exit_code = None
        elif name in self.exits:
            exit_code, line = self.exits[name]
            with colmap_run.log.open("a", encoding="utf-8") as log:
                log.write(f"{line}\n")
        else:
            exit_code = 0
            writers[name](colmap_run)
        with colmap_run.log.open(encoding="utf-8", errors="replace") as lines:
            last_line = last_error_line(lines)
        seconds = (float(colmap_run.timeout_seconds) if exit_code is None
                   else time.perf_counter() - start)
        return RunResult(exit_code=exit_code, seconds=seconds, last_line=last_line)

    def _writers(self) -> dict[str, Callable[[ColmapRun], None]]:
        """What each run this stand-in knows writes when it exits with 0, by the run's name."""
        return {"feature_extractor": _extract,
                "sequential_matcher_near": _match,
                "sequential_matcher_far": _match,
                "view_graph_calibrator": lambda run: _calibrate(run, self.calibration),
                "global_mapper": lambda run: _map(run, self.global_models, self.upside_down),
                "mapper": lambda run: _map(run, self.incremental_models, self.upside_down),
                "image_undistorter": lambda run: _undistort(
                    run, keep_camera_model=self.keep_camera_model,
                    second_camera=self.second_camera, leave_out_image=self.leave_out_image),
                "image_undistorter_standalone": lambda run: _warp_mask(
                    run, keep_camera_model=self.keep_camera_model)}


@pytest.fixture
def fake_colmap(monkeypatch: pytest.MonkeyPatch) -> type[FakeColmap]:
    """Give fake_colmap, which builds a FakeColmap with its script, like
    fake_colmap(calibration="rejected", global_models=None), and installs it as the Cameras_Step's
    tools for this test (cameras spec, Req 16.5). Building another one installs that one instead.

    Installing is monkeypatch.setattr("splatpipe.cameras.TOOLS", ...) with a Tools of the
    FakeColmap's find_colmap and run, so find_colmap gives FAKE_COLMAP, or None when the
    FakeColmap is built with installed=False (Req 2.7). The step reads TOOLS each time it runs,
    so a run through splatpipe.cameras.run, run_step or the step's CLI in this process uses the
    stand-in. A test of the solve flow passes its run to solve instead, in place of
    colmap.run_colmap. Either way, its `runs` then lists every ColmapRun it stood in for, in
    order. splatpipe.cameras is imported here, so only the tests that use this fixture need it.
    """
    from splatpipe.cameras import Tools

    class Installed(FakeColmap):
        """A FakeColmap that installs itself as the Cameras_Step's tools once it's built."""

        def __post_init__(self) -> None:
            super().__post_init__()  # a script it refuses installs nothing
            monkeypatch.setattr("splatpipe.cameras.TOOLS",
                                Tools(find_colmap=self.find_colmap, run=self.run))

    return Installed


def _extract(run: ColmapRun) -> None:
    """The fake feature_extractor: a database at --database_path whose cameras table holds one
    camera of --ImageReader.camera_model at the Pick_Size, as FakeColmap says."""
    width, height = _pick_size(Path(run.value("--image_path")))
    if "--ImageReader.camera_params" in run.args:  # the Fisheye_Start, a focal prior (Req 3.4)
        params = tuple(map(float, run.value("--ImageReader.camera_params").split(",")))
        prior = 1
    else:  # RADIAL's f starts at the factor times the long edge, with no prior (Req 3.3)
        focal = float(run.value("--ImageReader.default_focal_length_factor")) * max(width, height)
        params = (focal, width / 2, height / 2, 0.0, 0.0)  # f, cx, cy, k1, k2 (F2)
        prior = 0
    model_id = COLMAP_MODEL_IDS[run.value("--ImageReader.camera_model")]
    with closing(sqlite3.connect(run.value("--database_path"))) as database, database:
        database.execute(CAMERAS_TABLE)
        database.execute("INSERT INTO cameras (model, width, height, params, prior_focal_length) "
                         "VALUES (?, ?, ?, ?, ?)", (model_id, width, height, _blob(params), prior))


def _match(run: ColmapRun) -> None:
    """The fake sequential_matcher, for both Match_Runs: it writes nothing."""


def _calibrate(run: ColmapRun, calibration: str) -> None:
    """The fake view_graph_calibrator: it changes the camera of the database at --database_path,
    the Calibration_Copy, as `calibration` says (O5, F11)."""
    path = Path(run.value("--database_path"))
    if not path.is_file():
        raise AssertionError(f"COLMAP's view_graph_calibrator was given the database {path}, "
                             "which doesn't exist.")
    if calibration == "no_pairs":  # it found no image pairs to calibrate, so it changes nothing
        return
    with closing(sqlite3.connect(path)) as database, database:
        if calibration == "worked":  # a new focal length, RADIAL's one f, params[0] (F17)
            rows = database.execute("SELECT camera_id, params FROM cameras").fetchall()
            for camera_id, blob in rows:
                params = list(struct.unpack(f"<{len(blob) // 8}d", blob))
                params[0] = CALIBRATED_SHARE * params[0]
                database.execute("UPDATE cameras SET params = ? WHERE camera_id = ?",
                                 (_blob(params), camera_id))
        database.execute("UPDATE cameras SET prior_focal_length = 1")  # worked or rejected


def _map(run: ColmapRun, counts: Sequence[int] | None, upside_down: bool) -> None:
    """The fake global_mapper or mapper: one Model in --output_path for each of `counts`, as
    FakeColmap says, and no Model for None or no counts. With `upside_down`, every other camera
    is upside down."""
    from splatpipe.cameras.model import write_model

    output = Path(run.value("--output_path"))
    if not output.is_dir():  # COLMAP's mappers stop at once: "`output_path` is not a directory."
        raise AssertionError(f"COLMAP's {run.command} needs its output folder {output} to exist "
                             "before it runs.")
    if not counts:
        return
    folder = Path(run.value("--image_path"))
    names = sorted(path.name for path in folder.iterdir())
    if sum(counts) > len(names):
        raise ValueError(f"FakeColmap's {run.name} places {sum(counts)} Picks in its Models, but "
                         f"{folder} holds only {len(names)}.")
    width, height = _pick_size(folder)
    start = 0
    for number, count in enumerate(counts):
        write_model(output / str(number),
                    _walk_model(names, range(start, start + count), width, height, upside_down))
        start += count


def _walk_model(names: Sequence[str], places: range, width: int, height: int,
                upside_down: bool) -> Model:
    """The fake mappers' Model that places the Picks at `places` in `names`, the Picks of
    --image_path by name, at the Pick_Size `width` by `height`, as FakeColmap says. With
    `upside_down`, the camera at each odd place is upside down."""
    from splatpipe.cameras.model import POINT2D, TRACK_ELEMENT, Camera, Image, Model, Points

    focal = MAPPED_FOCAL * max(width, height)
    camera = Camera(camera_id=1, model_id=COLMAP_MODEL_IDS["RADIAL"], width=width, height=height,
                    params=(focal, width / 2, height / 2, *MAPPED_DISTORTION))
    tilt = math.radians(TILT_DEGREES)
    move = _matrix(_turn(TILT_AXIS, tilt))  # the level world into COLMAP's frame
    xyz = WALK_POINTS @ move.T + TILT_SHIFT
    ids = np.arange(1, len(xyz) + 1, dtype="<u8")
    images = []
    for place in places:
        centre = np.array([WALK_STEP * place, 0.0, 0.0])
        yaw = math.atan2(WALK_TARGET[0] - centre[0], WALK_TARGET[2] - centre[2])
        # cam_from_world: back from COLMAP's frame to the level world, then into the level
        # camera, which is turned by `yaw` about the y axis to look at WALK_TARGET.
        qvec = _hamilton(_turn((0.0, 1.0, 0.0), -yaw), _turn(TILT_AXIS, -tilt))
        if upside_down and place % 2:  # then a half turn about the camera's view axis, its z
            qvec = _hamilton(_turn((0.0, 0.0, 1.0), math.pi), qvec)
        rotation = _matrix(qvec)
        tvec = -rotation @ (move @ centre + TILT_SHIFT)
        seen = xyz @ rotation.T + tvec  # the points in the camera, all in front of it
        points2d = np.zeros(len(xyz), dtype=POINT2D)
        points2d["x"] = focal * seen[:, 0] / seen[:, 2] + width / 2
        points2d["y"] = focal * seen[:, 1] / seen[:, 2] + height / 2
        points2d["point3d_id"] = ids
        images.append(Image(image_id=place + 1, qvec=tuple(qvec.tolist()),
                            tvec=tuple(tvec.tolist()), camera_id=camera.camera_id,
                            name=names[place], points2d=points2d))
    tracks = tuple(np.array([(image.image_id, index) for image in images], dtype=TRACK_ELEMENT)
                   for index in range(len(xyz)))
    points = Points(ids=ids, xyz=xyz, rgb=np.array(WALK_COLOURS, dtype="u1"),
                    errors=np.array(WALK_ERRORS, dtype="<f8"), tracks=tracks)
    return Model(cameras=(camera,), images=tuple(images), points=points)


def _undistort(run: ColmapRun, *, keep_camera_model: bool, second_camera: bool,
               leave_out_image: bool) -> None:
    """The fake image_undistorter: the Undistorted_Model of the Model at --input_path in
    --output_path/sparse/, and a JPEG per image in --output_path/images/, with the three flags'
    faults, as FakeColmap says."""
    from splatpipe.cameras.model import Model, read_model, write_model

    model = read_model(Path(run.value("--input_path")))
    cameras = tuple(_undistorted_camera(camera, keep_camera_model) for camera in model.cameras)
    if second_camera:  # a copy of the first camera with the next id, which no image uses
        next_id = max(camera.camera_id for camera in cameras) + 1
        cameras += (replace(cameras[0], camera_id=next_id),)
    images = model.images[:-1] if leave_out_image else model.images  # without the last image
    output = Path(run.value("--output_path"))
    output.mkdir(exist_ok=True)  # as COLMAP makes it
    write_model(output / "sparse", Model(cameras=cameras, images=images, points=model.points))
    sizes = {camera.camera_id: (camera.width, camera.height) for camera in cameras}
    picks, folder = Path(run.value("--image_path")), output / "images"
    folder.mkdir(exist_ok=True)
    for image in images:
        pick = _read_picture(picks / image.name, cv2.IMREAD_COLOR)
        _write_picture(folder / image.name, _resized(pick, *sizes[image.camera_id]))


def _warp_mask(run: ColmapRun, *, keep_camera_model: bool) -> None:
    """The fake image_undistorter_standalone: the picture that the camera line in --input_file
    names, at the size the fake image_undistorter gives the line's camera, with a grey edge
    around every black region, as a three-channel PNG in --output_path, as FakeColmap says."""
    from splatpipe.cameras.model import Camera

    line = Path(run.value("--input_file")).read_text(encoding="utf-8")
    name, model, width, height, *params = line.split()  # as mask_camera_line writes it (F8)
    camera = _undistorted_camera(Camera(camera_id=1, model_id=COLMAP_MODEL_IDS[model],
                                        width=int(width), height=int(height),
                                        params=tuple(map(float, params))), keep_camera_model)
    mask = _resized(_read_picture(Path(run.value("--image_path")) / name, cv2.IMREAD_GRAYSCALE),
                    camera.width, camera.height)
    black = mask == 0
    # The pixels next to a black one that aren't black themselves, where bilinear sampling mixes
    # black and white.
    edge = cv2.dilate(black.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & ~black
    warped = np.where(edge, WARPED_EDGE, mask).astype(np.uint8)
    output = Path(run.value("--output_path"))
    output.mkdir(exist_ok=True)  # as COLMAP makes it
    _write_picture(output / name, np.repeat(warped[:, :, np.newaxis], 3, axis=2))


def _undistorted_camera(camera: Camera, keep_camera_model: bool) -> Camera:
    """The camera that the fake image_undistorter gives the RADIAL `camera`: PINHOLE and
    UNDISTORTED_SHRINK pixels smaller each way, with fx = fy = its f and the principal point
    scaled with the size, as COLMAP's UndistortCamera does (F7, F18). With `keep_camera_model`,
    `camera` as it is, still RADIAL, as COLMAP leaves a camera with no distortion (F9)."""
    from splatpipe.cameras.model import Camera

    if keep_camera_model:
        return camera
    width, height = camera.width - UNDISTORTED_SHRINK, camera.height - UNDISTORTED_SHRINK
    f, cx, cy = camera.params[:3]  # RADIAL and RADIAL_FISHEYE both start with these three (F2)
    return Camera(camera_id=camera.camera_id, model_id=PINHOLE_ID, width=width, height=height,
                  params=(f, f, cx * width / camera.width, cy * height / camera.height))


def _read_picture(path: Path, flags: int) -> np.ndarray:
    """The picture in the file `path`, decoded with cv2.imdecode and `flags`, like
    cv2.IMREAD_COLOR."""
    return cv2.imdecode(np.frombuffer(path.read_bytes(), dtype=np.uint8), flags)


def _write_picture(path: Path, picture: np.ndarray) -> None:
    """Encode `picture` with cv2.imencode in the format of `path`'s suffix, then write it."""
    ok, data = cv2.imencode(path.suffix, picture)
    assert ok, f"cv2.imencode couldn't encode {path}"
    path.write_bytes(data.tobytes())


def _resized(picture: np.ndarray, width: int, height: int) -> np.ndarray:
    """`picture` resized to `width` × `height` pixels, each taking its nearest pixel's value."""
    return cv2.resize(picture, (width, height), interpolation=cv2.INTER_NEAREST)


def _pick_size(folder: Path) -> tuple[int, int]:
    """The width and height of the first Pick in `folder` by name, which every Pick shares: the
    Pick_Size, read from the picture itself, as COLMAP reads it."""
    picture = _read_picture(min(folder.iterdir()), cv2.IMREAD_UNCHANGED)
    height, width = picture.shape[:2]
    return width, height


def _blob(params: Sequence[float]) -> bytes:
    """A camera's parameters as a COLMAP database's params holds them: little-endian doubles."""
    return struct.pack(f"<{len(params)}d", *params)


def _turn(axis: Sequence[float], radians: float) -> np.ndarray:
    """The unit quaternion (w, x, y, z) of a right-handed turn by `radians` about `axis`."""
    unit = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    return np.array([math.cos(radians / 2), *(math.sin(radians / 2) * unit)])


def _hamilton(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """The Hamilton product p q of two quaternions (w, x, y, z): the rotation q, then p."""
    return np.array([p[0] * q[0] - p[1:] @ q[1:],
                     *(p[0] * q[1:] + q[0] * p[1:] + np.cross(p[1:], q[1:]))])


def _matrix(q: np.ndarray) -> np.ndarray:
    """The rotation matrix of the unit quaternion q = (w, x, y, z), in Hamilton's convention."""
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


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
