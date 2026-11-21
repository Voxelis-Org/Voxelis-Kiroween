"""Fixtures for every test: the temporary config, dir_link for the link tests, fake_probe and
new_job, which make Jobs without ffprobe, set_fail, which sets `fail` in the test config,
set_clean, which sets its [clean] settings, and work_calls, which records whose Fake_Step work
runs.

The autouse fixture temp_config writes a test config into each test's tmp_path and points
splatpipe.config.CONFIG_PATH at it. So load_config() with no path, and every CLI a test runs in
its own process, reads the test config, whose Jobs_Folder is inside tmp_path. No test can make
a Job in the real data/ folder (Req 14.2).
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
    """`value` as TOML: a bool as true or false, a str in double quotes, and a number as Python
    writes it, which TOML reads back as the same number."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        # json.dumps escapes the quote, the backslash and every character below U+0020 the way
        # TOML reads them back. TOML can't hold DEL as it is either, so that's escaped too.
        return json.dumps(value, ensure_ascii=False).replace("\x7f", "\\u007f")
    return repr(value)


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


def config_text(jobs_folder: Path, clean: Mapping[str, object] | None = None, /,
                **fail: str) -> str:
    """The test config as TOML. Each Step_Name in `fail` gets that `fail` value, and every other
    Fake_Step section gets fail = "none". [clean] and its Preset tables come last, with the
    changes in `clean` made, as clean_toml reads them. The Jobs_Folder is written with forward
    slashes, so a Windows backslash can't become a TOML escape.

    `clean` can only be given by position, so no Step_Name in `fail` lands in it, not even
    "clean"."""
    jobs = JOBS_TOML.format(jobs_folder=jobs_folder.as_posix())
    steps = "".join(STEP_TOML.format(name=name, fail=fail.get(name, "none"))
                    for name in STEP_NAMES)
    return jobs + steps + clean_toml(clean or {})


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
    fail = "none", and set_fail() sets them all back. load_config() reads the file each time it's
    called, so the next run that loads the config sees the change.
    """
    def rewrite(**modes: str) -> None:
        temp_config.write_text(config_text(tmp_path / "jobs", **modes), encoding="utf-8")

    return rewrite


@pytest.fixture
def set_clean(tmp_path: Path, temp_config: Path) -> Callable[..., None]:
    """Give set_clean(**values), which rewrites the test config with the [clean] values given,
    like set_clean(neighbours=3). A key of a Preset table comes after its Preset and two
    underscores, like set_clean(fast__sog_sh_bands=1), and None leaves a key out, like
    set_clean(neighbours=None), so a test can check a missing setting.

    Each call writes the whole file again, the way set_fail does, so every setting it doesn't
    name is back to its value in CLEAN_SHARED or CLEAN_PRESETS, every fail is back to "none", and
    set_clean() sets them all back. A set_fail call after it puts [clean] back the same way.
    """
    def rewrite(**values: object) -> None:
        temp_config.write_text(config_text(tmp_path / "jobs", values), encoding="utf-8")

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
