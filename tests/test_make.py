"""Tests for the Job_Maker and the Trim, splatpipe/make.py.

The unit tests give make_job a stand-in Video_Probe, so none of them runs ffprobe (Req 14.3).
The temporary config from tests/conftest.py puts the Jobs_Folder at tmp_path / "jobs" and the
size limit at 1024 bytes.
"""

import errno
import functools
import math
import os
import stat
import subprocess
import sys
import tomllib
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, NoReturn

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from splatpipe.config import PRESETS, REPO_ROOT, Config, load_config
from splatpipe.errors import PipelineError
from splatpipe.jobs import check_job_id
from splatpipe.jsonio import read_json
from splatpipe.make import CONTAINER_FAMILY, main, make_job, used_part
from splatpipe.probe import ProbeResult


# Unit tests

# The start of the refusal for a video that isn't an MP4 or a MOV.
NOT_MP4_OR_MOV = "Only MP4 and MOV videos are accepted"
# What refusing_probe raises.
PROBE_REFUSAL = "ffprobe isn't installed. Install ffmpeg, which includes ffprobe."


def write_video(folder: Path, name: str = "clip.mp4", size: int = 16) -> Path:
    """A stand-in video: `size` bytes at `folder / name`. Gives its path."""
    path = folder / name
    path.write_bytes(b"x" * size)
    return path


def probed(format_name: str = CONTAINER_FAMILY, has_video: bool = True,
           video_seconds: float | None = 10.0) -> ProbeResult:
    """What the Video_Probe reads. By default, an MP4 with a video stream, 10 s long."""
    return ProbeResult(format_name=format_name, has_video=has_video, video_seconds=video_seconds)


def refusing_probe(path: Path, timeout_seconds: float) -> ProbeResult:
    """A Video_Probe that refuses every video, as probe_video does when ffprobe isn't installed."""
    raise PipelineError(PROBE_REFUSAL)


def changed_config(temp_config: Path, **jobs: object) -> Config:
    """The test config with the [jobs] settings in `jobs` changed, so its Jobs_Folder stays in
    tmp_path."""
    data = tomllib.loads(temp_config.read_text(encoding="utf-8"))
    data["jobs"].update(jobs)
    return Config(data)


def job_folders() -> list[Path]:
    """What the test config's Jobs_Folder holds, or nothing when it doesn't exist."""
    folder = load_config().jobs().jobs_folder
    return sorted(folder.iterdir()) if folder.exists() else []


def refused(video: Path, **options: Any) -> str:
    """Call make_job on `video` with `options`. Check that it raises a PipelineError and leaves
    the Jobs_Folder with no new folder (Req 2.1, 2.13), and give back the error's message."""
    with pytest.raises(PipelineError) as caught:
        make_job(video, **options)
    assert job_folders() == []
    return str(caught.value)


@pytest.mark.parametrize("is_folder", [False, True], ids=["missing path", "a folder"])
def test_a_path_that_isnt_a_file_is_refused(
        tmp_path: Path, fake_probe: Callable[..., Any], is_folder: bool) -> None:
    path = tmp_path / "clip.mp4"
    if is_folder:
        path.mkdir()
    assert "There's no video file" in refused(path, probe=fake_probe())


# Each file the extension and size checks refuse, as its name and its size in bytes, and the
# words its refusal must hold. The test config's size limit, 1024 bytes, is 9.53674e-07 GB.
BAD_FILES = [
    pytest.param("clip.mkv", 16, NOT_MP4_OR_MOV, id=".mkv"),
    pytest.param("clip.mp4.txt", 16, NOT_MP4_OR_MOV, id=".mp4.txt"),
    pytest.param("clip.mp4", 1025, "bigger than 9.53674e-07 GB", id="over the limit"),
]


@pytest.mark.parametrize(("name", "size", "words"), BAD_FILES)
def test_a_bad_extension_or_size_is_refused_before_the_probe_runs(
        tmp_path: Path, fake_probe: Callable[..., Any], name: str, size: int, words: str) -> None:
    probe = fake_probe()
    assert words in refused(write_video(tmp_path, name, size), probe=probe)
    assert probe.calls == []


@pytest.mark.parametrize("name", ["clip.MP4", "clip.Mov"])
def test_upper_and_mixed_case_extensions_pass_the_extension_check(
        tmp_path: Path, fake_probe: Callable[..., Any], name: str) -> None:
    probe = fake_probe()
    make_job(write_video(tmp_path, name), probe=probe)
    assert len(probe.calls) == 1  # the probe runs only once the extension and size checks pass


def test_a_size_limit_of_4294967296_bytes_is_given_as_4_gb(
        tmp_path: Path, temp_config: Path, fake_probe: Callable[..., Any],
        monkeypatch: pytest.MonkeyPatch) -> None:
    video = write_video(tmp_path)
    config = changed_config(temp_config, max_video_bytes=4294967296)
    real_stat = Path.stat

    def bigger_stat(self: Path, *, follow_symlinks: bool = True) -> os.stat_result:
        """The real Path.stat, except that the video's size is one byte over 4 GB."""
        result = real_stat(self, follow_symlinks=follow_symlinks)
        if self != video:
            return result
        fields = list(result)
        fields[stat.ST_SIZE] = 4294967297
        return os.stat_result(fields)

    # Only Path.stat, never os.stat, which is_file, tmp_path and pytest use too.
    monkeypatch.setattr(Path, "stat", bigger_stat)
    probe = fake_probe()
    assert "4 GB" in refused(video, config=config, probe=probe)
    assert probe.calls == []


def test_the_probe_gets_the_resolved_path_and_the_configured_timeout(
        tmp_path: Path, temp_config: Path, fake_probe: Callable[..., Any],
        monkeypatch: pytest.MonkeyPatch) -> None:
    write_video(tmp_path)
    monkeypatch.chdir(tmp_path)  # so the relative path below has to be resolved
    probe = fake_probe()
    make_job("clip.mp4", config=changed_config(temp_config, ffprobe_timeout_seconds=12.5),
             probe=probe)
    assert probe.calls == [((tmp_path / "clip.mp4").resolve(), 12.5)]


# Each Video_Probe result the Job_Maker refuses, and the words its refusal must hold. The format
# name must be exactly CONTAINER_FAMILY. The test config's longest video is 600 s, and its
# shortest Used_Part is 3 s.
BAD_RESULTS = [
    pytest.param(probed(format_name="matroska,webm"), [NOT_MP4_OR_MOV], id="another format"),
    pytest.param(probed(format_name="mp4"), [NOT_MP4_OR_MOV], id="one name of the family"),
    pytest.param(probed(format_name=CONTAINER_FAMILY.upper()), [NOT_MP4_OR_MOV], id="upper case"),
    pytest.param(probed(format_name=CONTAINER_FAMILY + " "), [NOT_MP4_OR_MOV],
                 id="trailing space"),
    pytest.param(probed(has_video=False), ["no video stream"], id="no video stream"),
    pytest.param(probed(video_seconds=None), ["couldn't tell how long"], id="no length"),
    pytest.param(probed(video_seconds=754.25), ["754.25 s", "600 s"], id="over 600 s"),
    pytest.param(probed(video_seconds=2.0), ["shorter than 3 s"], id="2 s with no trim"),
]


@pytest.mark.parametrize(("result", "words"), BAD_RESULTS)
def test_a_bad_probe_result_is_refused(
        tmp_path: Path, fake_probe: Callable[..., Any], result: ProbeResult,
        words: list[str]) -> None:
    message = refused(write_video(tmp_path), probe=fake_probe(result))
    for word in words:
        assert word in message


def test_a_refusal_from_the_probe_is_passed_on(tmp_path: Path) -> None:
    assert PROBE_REFUSAL in refused(write_video(tmp_path), probe=refusing_probe)


# Each Preset or Trim the Job_Maker refuses for a 10-second video, as make_job's options, and
# the words its refusal must hold.
BAD_CHOICES = [
    pytest.param({"preset": "slow"}, list(PRESETS), id="bad preset"),
    pytest.param({"start": 5.0, "end": 4.0}, ["isn't before the trim end"], id="start after end"),
    pytest.param({"end": 12.5}, ["10 s long"], id="end after the video"),
    pytest.param({"start": 8.0}, ["shorter than 3 s"], id="shorter than the minimum"),
    pytest.param({"start": True}, ["it's True"], id="bool start"),
    pytest.param({"end": 10**400}, ["too big"], id="huge int end"),
]


@pytest.mark.parametrize(("options", "words"), BAD_CHOICES)
def test_a_bad_preset_or_trim_is_refused(
        tmp_path: Path, fake_probe: Callable[..., Any], options: dict[str, Any],
        words: list[str]) -> None:
    message = refused(write_video(tmp_path), probe=fake_probe(probed()), **options)
    for word in words:
        assert word in message


# Making the Job

FIXED_ID = "20261119-142530-7f3a9c"  # the Job_Id the tests below give make_job through new_id
# The reason a stand-in for shutil.copyfile or write_record gives, as a full disk would.
FULL_DISK = "No space left on device"
VIDEO_BYTES = bytes(range(256))  # a stand-in video whose copy can't match by chance


def the_jobs_folder() -> Path:
    """The test config's Jobs_Folder. The temporary config doesn't make it."""
    return load_config().jobs().jobs_folder


def full_disk(*args: object, **kwargs: object) -> NoReturn:
    """A stand-in for shutil.copyfile or write_record that fails the way a full disk does."""
    raise OSError(errno.ENOSPC, FULL_DISK)


@pytest.mark.parametrize("folder_exists", [False, True],
                         ids=["no jobs folder yet", "a jobs folder"])
def test_a_good_job_has_input_work_and_out(new_job: Callable[..., str],
                                           folder_exists: bool) -> None:
    jobs_folder = the_jobs_folder()
    if folder_exists:
        jobs_folder.mkdir()
    else:
        assert not jobs_folder.exists()
    folder = jobs_folder / new_job()
    assert {path.name: path.is_dir() for path in folder.iterdir()} == {
        "input": True, "work": True, "out": True}


def test_a_mov_is_copied_to_input_video_mov_and_the_original_stays_as_it_was(
        tmp_path: Path, fake_probe: Callable[..., Any]) -> None:
    video = tmp_path / "Kitchen walk.MOV"
    video.write_bytes(VIDEO_BYTES)
    os.utime(video, (1_600_000_000, 1_600_000_000))  # a time in 2020, so a write would show
    modified = video.stat().st_mtime_ns
    input_folder = the_jobs_folder() / make_job(video, probe=fake_probe()) / "input"
    assert sorted(path.name for path in input_folder.iterdir()) == ["job.json", "video.mov"]
    assert (input_folder / "video.mov").read_bytes() == VIDEO_BYTES
    assert video.read_bytes() == VIDEO_BYTES
    assert video.stat().st_mtime_ns == modified


@pytest.mark.parametrize(("preset", "saved"), [("quality", "quality"), (None, "default")],
                         ids=["chosen preset", "no preset"])
def test_the_job_record_holds_every_field_with_the_chosen_or_default_preset(
        tmp_path: Path, temp_config: Path, fake_probe: Callable[..., Any], preset: str | None,
        saved: str) -> None:
    # [jobs].default_preset is "default" here, not the test config's "fast", so the Preset saved
    # when none is chosen can only come from [jobs].
    config = changed_config(temp_config, default_preset="default")
    video = write_video(tmp_path, "Kitchen walk.MOV", size=48)
    job_id = make_job(video, start=1.5, end=8.25, preset=preset, config=config,
                      probe=fake_probe())
    assert read_json(config.jobs().jobs_folder / job_id / "input" / "job.json") == {
        "job_id": job_id, "preset": saved, "start_seconds": 1.5, "end_seconds": 8.25,
        "source_name": "Kitchen walk.MOV", "source_bytes": 48, "video_seconds": 10.0,
        "video_file": "video.mov"}


def test_without_new_id_the_job_id_comes_from_new_job_id(
        tmp_path: Path, fake_probe: Callable[..., Any]) -> None:
    job_id = make_job(write_video(tmp_path), probe=fake_probe())
    assert check_job_id(job_id) == job_id
    made = datetime.strptime(job_id[:15], "%Y%m%d-%H%M%S").replace(tzinfo=timezone.utc)
    assert abs(datetime.now(timezone.utc) - made) < timedelta(seconds=5)  # the UTC time now
    assert job_folders() == [the_jobs_folder() / job_id]


def test_a_job_folder_that_already_exists_is_refused_and_left_as_it_is(
        tmp_path: Path, fake_probe: Callable[..., Any]) -> None:
    folder = the_jobs_folder() / FIXED_ID
    folder.mkdir(parents=True)
    (folder / "notes.txt").write_bytes(b"not ours")
    with pytest.raises(PipelineError, match="already exists"):
        make_job(write_video(tmp_path), probe=fake_probe(), new_id=lambda: FIXED_ID)
    assert [(path.name, path.read_bytes()) for path in folder.iterdir()] == [
        ("notes.txt", b"not ours")]


@pytest.mark.parametrize("target", ["shutil.copyfile", "splatpipe.make.write_record"])
def test_a_copy_or_record_that_fails_like_a_full_disk_leaves_no_job_folder(
        tmp_path: Path, fake_probe: Callable[..., Any], monkeypatch: pytest.MonkeyPatch,
        target: str) -> None:
    monkeypatch.setattr(target, full_disk)
    message = refused(write_video(tmp_path), probe=fake_probe())
    for words in ("couldn't be copied", FULL_DISK, "free space"):
        assert words in message


# The CLI

@pytest.fixture
def patched_make_job(monkeypatch: pytest.MonkeyPatch, fake_probe: Callable[..., Any]) -> None:
    """Make main's make_job run with a FakeProbe and a new_id that gives FIXED_ID.

    make_job's default probe is bound when the function is defined, so patching probe_video
    wouldn't reach it. So splatpipe.make.make_job, which main looks up when it runs, becomes a
    functools.partial of the real make_job.
    """
    monkeypatch.setattr("splatpipe.make.make_job",
                        functools.partial(make_job, probe=fake_probe(), new_id=lambda: FIXED_ID))


@pytest.mark.usefixtures("patched_make_job")
def test_main_prints_only_the_job_id_and_returns_0(
        tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main([str(write_video(tmp_path))]) == 0
    captured = capsys.readouterr()
    assert (captured.out, captured.err) == (FIXED_ID + "\n", "")
    assert job_folders() == [the_jobs_folder() / FIXED_ID]


@pytest.mark.usefixtures("patched_make_job")
def test_main_with_a_job_id_is_a_usage_error_and_makes_no_job(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as caught:
        main(["--job", FIXED_ID, str(write_video(tmp_path))])
    assert caught.value.code == 2
    assert job_folders() == []


@pytest.mark.usefixtures("patched_make_job")
def test_main_prints_a_refusal_on_stderr_and_returns_1(
        tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main([str(write_video(tmp_path, "clip.mkv"))]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert NOT_MP4_OR_MOV in captured.err
    assert "Traceback" not in captured.err


def test_python_m_splatpipe_make_refuses_a_missing_video_from_the_repo_root(
        tmp_path: Path) -> None:
    """In its own process, the CLI reads the real config.toml, whose Jobs_Folder is data/jobs.
    It stops at the missing file, before anything is made."""
    video = tmp_path / "missing.mp4"
    result = subprocess.run([sys.executable, "-m", "splatpipe.make", str(video)], cwd=REPO_ROOT,
                            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
                            capture_output=True, encoding="utf-8", timeout=60)
    assert (result.returncode, result.stdout) == (1, "")
    assert f"There's no video file at {video}." in result.stderr
    assert "Traceback" not in result.stderr


# Property tests

# Video_Lengths up to 1e6 s and minimum lengths up to 1e3 s, both above 0, subnormals included.
lengths = st.floats(min_value=0, max_value=1e6, exclude_min=True)
minimums = st.floats(min_value=0, max_value=1e3, exclude_min=True)


def trim_values(length: float) -> st.SearchStrategy[float | None]:
    """A Trim start or end for a video `length` s long: missing, any float (st.floats() gives
    NaN, both infinities, negatives and subnormals), a float from -1 s to twice the length, or
    one of the edge values."""
    return st.one_of(st.none(), st.floats(), st.floats(min_value=-1, max_value=2 * length),
                     st.sampled_from([0.0, -0.0, length, -1.0, math.nan, math.inf, -math.inf]))


def rule(length: float, start: float | None, end: float | None, minimum: float) -> bool:
    """Property 3's rule, the oracle: whether used_part must accept the Trim. It's the design's
    four parts, in order, with a missing start counted as 0.0 and a missing end as the length."""
    given_values = [value for value in (start, end) if value is not None]
    s = 0.0 if start is None else start
    e = length if end is None else end
    return (all(math.isfinite(value) and value >= 0 for value in given_values)
            and all(value <= length for value in given_values)
            and s < e
            and e - s >= minimum)


@given(lengths, minimums, st.data())
def test_the_trim_accepts_exactly_the_valid_used_parts(
        length: float, minimum: float, data: st.DataObject) -> None:
    """Feature: pipeline-core, Property 3: The Trim accepts exactly the valid Used_Parts

    **Validates: Requirements 3.4, 3.5, 3.6, 3.7, 3.8**
    """
    start = data.draw(trim_values(length), label="start")
    end = data.draw(trim_values(length), label="end")
    if rule(length, start, end, minimum):
        used_part(length, start, end, minimum)  # it must return, not raise
    else:
        with pytest.raises(PipelineError):
            used_part(length, start, end, minimum)


@st.composite
def accepted_trims(draw: st.DrawFn) -> tuple[float, float | None, float | None, float]:
    """A Video_Length, a Trim and a minimum length, as (length, start, end, minimum), that
    Property 3's rule accepts. It draws mostly valid Trims: a minimum above 0 and up to the
    length, a start that's missing or from 0 to the length less the minimum, and an end that's
    missing or from the start plus the minimum to the length, with a missing start counted as 0."""
    length = draw(lengths)
    minimum = draw(st.floats(min_value=0, max_value=length, exclude_min=True))
    start = draw(st.none() | st.floats(min_value=0, max_value=length - minimum))
    first = 0.0 if start is None else start
    # Rounding can push first + minimum just above the length, and st.floats raises
    # InvalidArgument when its min_value is above its max_value, so drop those draws here.
    assume(first + minimum <= length)
    end = draw(st.none() | st.floats(min_value=first + minimum, max_value=length))
    assume(rule(length, start, end, minimum))  # drops the other rare float-rounding misses
    return length, start, end, minimum


@given(accepted_trims())
def test_an_accepted_trim_gives_back_its_own_values(
        trim: tuple[float, float | None, float | None, float]) -> None:
    """Feature: pipeline-core, Property 4: An accepted Trim gives back its own values

    **Validates: Requirements 3.1, 3.2, 3.3, 3.9**
    """
    length, start, end, minimum = trim
    expected = (0.0 if start is None else start, length if end is None else end)
    assert used_part(length, start, end, minimum) == expected
