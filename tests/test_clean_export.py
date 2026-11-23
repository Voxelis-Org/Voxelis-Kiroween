"""Tests for the export, splatpipe/clean/export.py.

The unit tests check the argument list splat_transform_argv builds for each test Preset (Req 5.2
to 5.4, 5.6, 5.7, 10.9), and that find_splat_transform says which of node and bin/cli.mjs is
missing, or that both are (Req 5.10). Then they run export_sog through the real subprocess.run
on tests/fake_splat_transform.py, a stand-in for splat-transform, so the timeout, the kill and
the decoding are tested on every machine, with no Node (Req 5.11 to 5.14, 10.6). Last, a
stand-in run checks the call itself, a return code of -9 and an OSError (Req 5.8, 5.11). In every
case the whole output lands in the log file.

The real test, at the end, runs export_sog with the real splat-transform on a tiny synthetic
splat, and checks that the Viewer_Copy is a zip whose meta.json says version 2 (Req 5.1, 9.6,
10.7). It's skipped when node isn't on the PATH or the repo has no bin/cli.mjs, with a reason
that says which is missing, and a test with a stand-in PATH checks that reason (Req 10.6).

Each Readable_Error is checked for what its requirement says it names and for no traceback
(Req 10.4). Every subprocess gets a timeout, and every Python child gets PYTHONIOENCODING=utf-8
(Req 10.7).
"""

import errno
import json
import os
import re
import shutil
import subprocess
import sys
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from splatpipe.clean.export import (CLI, TAIL_CHARS, export_sog, find_splat_transform,
                                    splat_transform_argv)
from splatpipe.clean.settings import CleanSettings
from splatpipe.config import REPO_ROOT
from splatpipe.errors import ExpectedFailure

import fake_splat_transform as fake
from conftest import CLEAN_PRESETS
from splats import clean_settings, gaussians, ply_bytes, standard_fields

# The three Presets, written out here rather than taken from the code under test.
PRESET_NAMES = ("fast", "default", "quality")
# splat-transform's script under the repo root, which npm ci installs, written out here rather
# than taken from export.py (Req 5.2).
REPO_CLI = REPO_ROOT / "node_modules" / "@playcanvas" / "splat-transform" / "bin" / "cli.mjs"
# Where the tests with a stand-in run say node is. It's never run.
NODE = "/opt/nodejs/bin/node"
# max_workers for the argument-list test. Every test Preset has 1 sh_iteration, so a different
# number here shows which value follows -i and which follows --max-workers.
WORKERS = 2


# The argument list

@pytest.mark.parametrize("preset", PRESET_NAMES)
def test_the_argument_list_runs_the_repos_cli_with_the_presets_values(
        tmp_path: Path, preset: str) -> None:
    s = clean_settings(**CLEAN_PRESETS[preset], max_workers=WORKERS)
    source, target = tmp_path / "clean.ply", tmp_path / "clean.sog"
    argv = splat_transform_argv(NODE, REPO_ROOT / CLI, source, target, s.sog_sh_bands, s)
    # The global options before the input, -H right after it, and the .sog last (Req 5.2 to 5.4,
    # 5.6).
    assert argv == [NODE, str(REPO_CLI),
                    "-w", "-g", "cpu", "-i", str(CLEAN_PRESETS[preset]["sh_iterations"]),
                    "--max-workers", str(WORKERS),
                    str(source),
                    "-H", str(CLEAN_PRESETS[preset]["sog_sh_bands"]),
                    str(target)]
    # No option that needs a GPU, and no output but the .sog (Req 5.7, 10.9).
    assert not {"--filter-floaters", "--filter-cluster", "--decimate", "-N"} & set(argv)
    assert not [arg for arg in argv if arg.endswith((".voxel.json", ".webp"))]


# Finding node and splat-transform

def put_node_on_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, with_node: bool) -> None:
    """Make an empty folder the whole PATH, with a file in it that shutil.which takes for node
    when `with_node` is true. The working folder becomes tmp_path, because on Windows
    shutil.which looks there first."""
    folder = tmp_path / "path"
    folder.mkdir()
    if with_node:
        node = folder / ("node.exe" if sys.platform == "win32" else "node")
        node.write_bytes(b"")
        node.chmod(0o755)
    monkeypatch.setenv("PATH", str(folder))
    monkeypatch.chdir(tmp_path)


# What find_splat_transform finds: node on the PATH, and cli.mjs under the repo root.
MISSING = [
    pytest.param(False, True, id="no node on the PATH"),
    pytest.param(True, False, id="no node_modules"),
    pytest.param(False, False, id="neither"),
]


@pytest.mark.parametrize(("with_node", "with_cli"), MISSING)
def test_find_says_which_is_missing_to_run_npm_ci_and_node_22(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, with_node: bool, with_cli: bool) -> None:
    put_node_on_path(tmp_path, monkeypatch, with_node=with_node)
    repo = tmp_path / "repo"
    repo.mkdir()
    if with_cli:
        cli = repo.joinpath("node_modules", "@playcanvas", "splat-transform", "bin", "cli.mjs")
        cli.parent.mkdir(parents=True)
        cli.write_text("// splat-transform's script\n", encoding="utf-8")
    with pytest.raises(ExpectedFailure) as caught:
        find_splat_transform(repo)
    message = str(caught.value)
    assert ("`node` isn't on the PATH" in message) is not with_node
    assert ("node_modules/@playcanvas/splat-transform/bin/cli.mjs" in message) is not with_cli
    assert "npm ci" in message and "Node 22 or newer" in message
    assert "Traceback" not in message


# Running export_sog on the files in a folder

# The Clean_Splat export_sog reads, the Viewer_Copy it makes and its log file, each in a test's
# tmp_path. The Clean_Splat isn't made, because neither stand-in reads it.
SOURCE, TARGET, LOG = "clean.ply", "clean.sog", "splat-transform.log"


def export(folder: Path, s: CleanSettings, **options: Any) -> int:
    """Run export_sog on the files in `folder` with the settings' band count, and give what it
    gives. `options` are its run and find."""
    return export_sog(folder / SOURCE, folder / TARGET, s.sog_sh_bands, s, folder / LOG,
                      **options)


def without(message: str, *chosen: object) -> str:
    """`message` with each value the test chose taken out, as its str, the longest first. So a
    check reads only the words export.py wrote itself, as refusal does in
    tests/test_clean_limits.py."""
    for text in sorted(map(str, chosen), key=len, reverse=True):
        message = message.replace(text, "")
    return message


def failure(folder: Path, s: CleanSettings, **options: Any) -> str:
    """The message of the ExpectedFailure that export raises. Its own words, with `folder` and
    NODE taken out, hold no traceback (Req 10.4)."""
    with pytest.raises(ExpectedFailure) as caught:
        export(folder, s, **options)
    message = str(caught.value)
    assert "Traceback" not in without(message, folder, NODE)
    return message


def logged(folder: Path) -> bytes:
    """The bytes of the log file export_sog wrote in `folder`."""
    return (folder / LOG).read_bytes()


# The stand-in script, through the real subprocess.run

# The stand-in, which export_sog runs with this Python in place of node.
STAND_IN = Path(fake.__file__)


def find_stand_in() -> tuple[str, Path]:
    """A find that gives this Python and the stand-in script in place of node and cli.mjs."""
    return sys.executable, STAND_IN


def use_stand_in(monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    """Set the stand-in's mode in the environment its process gets, with its output in UTF-8."""
    monkeypatch.setenv(fake.MODE_VARIABLE, mode)
    monkeypatch.setenv("PYTHONIOENCODING", "utf-8")


# An ANSI colour code, like \x1b[32m or \x1b[1;32m.
COLOUR_CODE = re.compile(r"\x1b\[[0-9;]*m")


def shown_lines(output: str) -> list[str]:
    r"""The lines a terminal shows for `output`, which ends in \n: each line from its last \r on,
    with its colour codes gone."""
    return [COLOUR_CODE.sub("", line.rpartition("\r")[2])
            for line in output.removesuffix("\n").split("\n")]


def test_a_failed_run_gives_the_exit_code_and_the_end_of_its_output_as_a_terminal_shows_it(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    use_stand_in(monkeypatch, "fail")
    message = failure(tmp_path, clean_settings(), find=find_stand_in)
    assert f"exit code {fake.FAIL_EXIT_CODE}" in message
    output = "".join(text for _, text in fake.fail_output())
    shown = shown_lines(output)
    # The tail, as the design words the message around it (Req 5.11).
    tail = message.partition("Its last lines were:\n")[2].rpartition("\nThe whole output is in ")[0]
    lines = tail.split("\n")
    assert len(tail) <= TAIL_CHARS < len("\n".join(shown))  # so the stand-in's output was cut
    assert lines == shown[-len(lines):]  # whole lines from the end, so it starts on a line
    assert len("\n".join(shown[-len(lines) - 1:])) > TAIL_CHARS  # one more line wouldn't fit
    assert "\r" not in tail and "\x1b" not in tail
    assert logged(tmp_path) == output.encode("utf-8")


# What silent and empty leave at the Viewer_Copy's path: no file, and a file of 0 bytes.
NO_VIEWER_COPY = [pytest.param("silent", None, id="silent"),
                  pytest.param("empty", b"", id="empty")]


@pytest.mark.parametrize(("mode", "left"), NO_VIEWER_COPY)
def test_an_exit_of_0_with_no_viewer_copy_or_an_empty_one_says_so(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str, left: bytes | None) -> None:
    use_stand_in(monkeypatch, mode)
    message = failure(tmp_path, clean_settings(), find=find_stand_in)
    target = tmp_path / TARGET
    assert (target.read_bytes() if target.exists() else None) == left
    assert f"wrote no viewer copy at {target}" in message  # Req 5.13
    assert logged(tmp_path) == b""


def test_a_done_run_gives_the_size_of_the_viewer_copy(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    use_stand_in(monkeypatch, "ok")
    size = export(tmp_path, clean_settings(), find=find_stand_in)
    assert size == len(fake.OK_BYTES) == (tmp_path / TARGET).stat().st_size  # Req 5.14
    assert logged(tmp_path) == b""


# How long a stopped stand-in may take to end, and to be gone, in seconds.
FEW_SECONDS = 5


def running(pid: int) -> bool:
    """Whether the process `pid` is still running: tasklist lists it on Windows, and elsewhere
    signal 0 reaches it."""
    if sys.platform == "win32":
        listed = subprocess.run(["tasklist", "/fi", f"PID eq {pid}", "/fo", "csv", "/nh"],
                                capture_output=True, timeout=30, check=True)
        return f'"{pid}"'.encode() in listed.stdout
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def gone(pid: int) -> bool:
    """Whether the process `pid` is gone within FEW_SECONDS."""
    deadline = time.monotonic() + FEW_SECONDS
    while running(pid):
        if time.monotonic() > deadline:
            return False
        time.sleep(0.1)
    return True


def test_a_run_past_the_time_limit_is_stopped_within_a_few_seconds_and_its_process_is_gone(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    use_stand_in(monkeypatch, "hang")
    started = time.monotonic()
    message = failure(tmp_path, clean_settings(splat_transform_timeout_seconds=0.5),
                      find=find_stand_in)
    assert time.monotonic() - started < FEW_SECONDS
    assert "within 0.5 seconds" in message  # Req 5.12
    # The stand-in's one line, printed before it slept, is the whole output.
    output = logged(tmp_path).decode("utf-8")
    assert output.startswith(fake.PID_PREFIX), "the stand-in was stopped before it printed its PID"
    assert gone(int(output.removeprefix(fake.PID_PREFIX)))


# A stand-in run

def find_node() -> tuple[str, Path]:
    """A find that gives NODE and the repo's cli.mjs, for the tests whose run runs nothing."""
    return NODE, REPO_ROOT / CLI


@dataclass
class FakeRun:
    """A stand-in for subprocess.run. Each call is recorded in `calls`, as (args, kwargs). Then it
    raises `error` when that's given, or else writes `sog` to the last argument, the .sog, when
    that's given, and gives back a finished run with `returncode` and `stdout`."""

    returncode: int = 0
    stdout: bytes = b""
    sog: bytes | None = None
    error: OSError | None = None
    calls: list[tuple[object, dict[str, Any]]] = field(default_factory=list)

    def __call__(self, args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        self.calls.append((args, kwargs))
        if self.error is not None:
            raise self.error
        if self.sog is not None:
            Path(args[-1]).write_bytes(self.sog)
        return subprocess.CompletedProcess(args, self.returncode, stdout=self.stdout)


# Output as splat-transform might print it, with its own line ends and a character that isn't
# ASCII, which the log file keeps as they were.
OUTPUT = "Reading clean.ply\r\nWriting clean.sog \u2713\r\n".encode("utf-8")


def test_run_gets_an_argument_list_and_no_shell(tmp_path: Path) -> None:
    s = clean_settings()
    run = FakeRun(stdout=OUTPUT, sog=fake.OK_BYTES)
    assert export(tmp_path, s, run=run, find=find_node) == len(fake.OK_BYTES)
    [(args, kwargs)] = run.calls
    assert type(args) is list  # Req 5.8
    assert args == splat_transform_argv(NODE, REPO_ROOT / CLI, tmp_path / SOURCE,
                                        tmp_path / TARGET, s.sog_sh_bands, s)
    assert not kwargs.get("shell", False)
    assert logged(tmp_path) == OUTPUT


def test_a_return_code_of_minus_9_gives_the_out_of_memory_hint(tmp_path: Path) -> None:
    run = FakeRun(returncode=-9, stdout=OUTPUT)
    message = failure(tmp_path, clean_settings(), run=run, find=find_node)
    assert "exit code -9" in message
    assert "signal 9" in message and "ran out of memory" in message  # Req 5.11
    assert logged(tmp_path) == OUTPUT


def test_an_os_error_names_the_node_path_and_the_reason(tmp_path: Path) -> None:
    run = FakeRun(error=PermissionError(errno.EACCES, "Permission denied"))
    message = failure(tmp_path, clean_settings(), run=run, find=find_node)
    assert NODE in message and "Permission denied" in message
    assert logged(tmp_path) == b""


# The real splat-transform (Req 10.6, 10.7)

def splat_transform_missing(cli: Path = REPO_CLI) -> str:
    """What the real splat-transform run lacks on this machine, as the reason its test is
    skipped: node on the PATH, splat-transform's script at `cli`, or both. "" when it lacks
    neither (Req 10.6)."""
    missing: list[str] = []
    if shutil.which("node") is None:
        missing.append("node isn't on the PATH")
    if not cli.is_file():
        missing.append("splat-transform's bin/cli.mjs isn't there (npm ci at the repo root "
                       "installs it)")
    return ", and ".join(missing)


# Why the real splat-transform can't run on this machine, or "" when it can.
REAL_RUN_MISSING = splat_transform_missing()
# Skips the test that runs the real splat-transform, with that reason, so the whole suite passes
# on a machine with no Node (Req 10.6).
needs_splat_transform = pytest.mark.skipif(bool(REAL_RUN_MISSING), reason=REAL_RUN_MISSING)

# One of the two missing at a time: whether a stand-in node is on the PATH, and whether a
# stand-in cli.mjs is there.
ONE_MISSING = [pytest.param(False, True, id="no node on the PATH"),
               pytest.param(True, False, id="no cli.mjs")]


@pytest.mark.parametrize(("with_node", "with_cli"), ONE_MISSING)
def test_the_real_runs_skip_reason_says_which_is_missing(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, with_node: bool, with_cli: bool) -> None:
    put_node_on_path(tmp_path, monkeypatch, with_node=with_node)
    cli = tmp_path / "cli.mjs"
    if with_cli:
        cli.write_text("// splat-transform's script\n", encoding="utf-8")
    reason = splat_transform_missing(cli)
    assert ("node isn't on the PATH" in reason) is not with_node
    assert ("bin/cli.mjs isn't there" in reason) is not with_cli


@needs_splat_transform
def test_the_real_splat_transform_makes_a_zip_whose_meta_json_says_version_2(
        tmp_path: Path) -> None:
    # A tiny synthetic splat: 64 Gaussians with 45 f_rest_* properties, 3 bands (Req 10.7).
    (tmp_path / SOURCE).write_bytes(ply_bytes(standard_fields(sh_count=45),
                                              gaussians(64, sh_count=45)))
    # -H 1, -i 1, --max-workers 0, which runs the encoding inline, and a 60-second timeout.
    s = clean_settings(sog_sh_bands=1, sh_iterations=1, max_workers=0,
                       splat_transform_timeout_seconds=60)
    export(tmp_path, s)  # the real run and find: node, on the repo's cli.mjs
    with zipfile.ZipFile(tmp_path / TARGET) as sog:
        meta = json.loads(sog.read("meta.json"))
    assert meta["version"] == 2  # splat-transform 3.8.0 writes SOG version 2 (Req 5.1, 9.6)
