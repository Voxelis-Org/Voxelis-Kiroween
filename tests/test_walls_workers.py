"""The Walls_Step's import checks (walls spec, Req 1.11, 2.13, 22.8). They read the files with
ast, through test_imports.py's `imports_of`, and run no Step.

- Every splatpipe/walls/*.py imports only names in sys.stdlib_module_names, numpy, scipy, cv2 and
  splatpipe, a relative import counting as splatpipe, so torch, gsplat, MoGe, boto3 and cloud/
  stay out. pool.py's multiprocessing is in the standard library. None of the files imports
  anything from splatpipe.depth, and none names VideoCapture, VideoWriter or VideoWriter_fourcc,
  cv2's video classes: not in an import like `from cv2 import VideoWriter`, an attribute like
  cv2.VideoCapture under any name the file gives cv2, or a bare name. A docstring or comment that
  names one isn't code, so it doesn't count. The files are found with glob, so each file a later
  task adds is checked as it lands, and the check fails if it finds no file, so it never passes
  by reading nothing.
- No splatpipe/train/*.py imports splatpipe.walls, `from splatpipe import walls` and relative
  imports included, so the Train_Step never depends on the Walls_Step (Req 22.8).
- A fresh process that imports splatpipe.walls and each of its modules, found with glob, holds
  none of torch, gsplat and moge in sys.modules (Req 1.11).

Each check's failure lists the file and the line of every import or video class it found. A
made-up file shows the walls check finding each kind of problem with its line.

Task 9 of the walls spec adds the pool tests and the two-worker test (Req 5.1, 5.4, 5.6 to 5.9,
22.6):

- worker_count with os.sched_getaffinity and os.cpu_count replaced through monkeypatch.
- InlinePool.map_frames_until with a fake clock that the task itself moves on, so the test knows
  exactly when each task starts and ends.
- open_pool(2), the real spawn workers: a worker that calls os._exit, one that raises an
  ExpectedFailure and one that raises ValueError, a worker that reads the thread-limit variables
  while the test process holds its own values again (C15), and no child left after close(). The
  worker functions and their task class are at this module's top level, so a spawn worker
  unpickles them by name: pytest imports this file as `test_walls_workers` with tests/ on
  sys.path (rootdir-relative "prepend" import, tests/ has no __init__.py), and a spawn child
  starts with its parent's sys.path, so it imports the same module. The module imports nothing
  heavy at its top level for that reason: splatpipe.walls, the step, is imported inside the one
  test that runs it.
- One tiny Walls_Step on the textured plane's Job, once with InlinePool and once with open_pool(2),
  with a fixed fake clock, so no Prior_Frame times out. The Walls_Prior_Files are compared array
  by array, bytes, dtype and shape, because np.savez stamps each zip entry with the time it was
  written, so two runs a second apart write different zip bytes for the same arrays. The
  Walls_Record is compared without its times and its worker count.

Every pool is closed in a `with` block or a fixture's finally, so no worker outlives its test.
"""

import ast
import json
import multiprocessing
import os
import subprocess
import sys
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from splatpipe.config import REPO_ROOT, load_config
from splatpipe.errors import ExpectedFailure
from splatpipe.walls.pool import (THREAD_LIMITS, InlinePool, MapResult, Pool, WorkerStopped,
                                  open_pool, stopped_message, worker_count)
from test_imports import imports_in, imports_of, top

SPLATPIPE = REPO_ROOT / "splatpipe"
WALLS_PACKAGE = "splatpipe.walls"
# The top-level names a splatpipe/walls/*.py may import besides the standard library (Req 22.8).
WALLS_IMPORTS = frozenset({"numpy", "scipy", "cv2", "splatpipe"})
# The package no splatpipe/walls/*.py imports from (Req 22.8).
DEPTH_PACKAGE = "splatpipe.depth"
# cv2's names that read and write video, which no splatpipe/walls/*.py names (Req 2.13).
VIDEO_CLASSES = frozenset({"VideoCapture", "VideoWriter", "VideoWriter_fourcc"})
# The GPU packages a fresh process that imports splatpipe.walls never loads, by top level
# (Req 1.11).
NEVER_LOADED_BY_WALLS = frozenset({"torch", "gsplat", "moge"})
# Run in a fresh process with module names as its arguments: it imports splatpipe.walls, then
# each module it's given, and prints the name of every module in sys.modules.
LOADED_BY_WALLS = ("import importlib, json, sys\n"
                   "import splatpipe.walls\n"
                   "for name in sys.argv[1:]:\n"
                   "    importlib.import_module(name)\n"
                   "print(json.dumps(sorted(sys.modules)))\n")


def within(name: str, package: str) -> bool:
    """Whether the dotted module name `name` is `package` or a module inside it."""
    return name == package or name.startswith(f"{package}.")


def walls_problems(source: str | bytes, package: str) -> list[tuple[int, str]]:
    """Every import and name in `source`, a file in `package`, that a splatpipe/walls/*.py may not
    have, as (line, what), in order: an import outside the standard library, numpy, scipy, cv2
    and splatpipe, an import from splatpipe.depth, and a cv2 video class, imported, used as an
    attribute or named bare. `from cv2 import VideoWriter` reads as cv2.VideoWriter."""
    allowed = sys.stdlib_module_names | WALLS_IMPORTS
    found = [(line, name) for line, name in imports_in(source, package)
             if top(name) not in allowed or within(name, DEPTH_PACKAGE)
             or name.removeprefix("cv2.") in VIDEO_CLASSES]
    found += [(node.lineno, ast.unparse(node)) for node in ast.walk(ast.parse(source))
              if (isinstance(node, ast.Attribute) and node.attr in VIDEO_CLASSES)
              or (isinstance(node, ast.Name) and node.id in VIDEO_CLASSES)]
    return sorted(found)


def walls_files() -> list[Path]:
    """Every splatpipe/walls/*.py, found with glob, so each file a later task adds counts."""
    return sorted(SPLATPIPE.glob("walls/*.py"))


def test_the_walls_check_finds_each_problem_with_its_line() -> None:
    """A made-up file that breaks each rule, so the check can fail: torch, an absolute and a
    relative import from splatpipe.depth, `from splatpipe import depth`, an imported and an
    attribute video class, VideoWriter_fourcc as an attribute, and a bare name. The docstring
    names VideoCapture, which isn't code, and stdlib, numpy, scipy, cv2 and splatpipe imports
    pass."""
    sample = ('"""Never calls cv2.VideoCapture."""\n'
              "import multiprocessing\n"
              "import numpy as np\n"
              "from scipy import ndimage\n"
              "import cv2 as image_codec\n"
              "from splatpipe.train import files\n"
              "from . import settings\n"
              "import torch\n"
              "from splatpipe.depth import files as depth_files\n"
              "from ..depth.fit import fit\n"
              "from splatpipe import depth\n"
              "from cv2 import VideoWriter\n"
              "def read(path):\n"
              "    return image_codec.VideoCapture(path)\n"
              "def code():\n"
              "    return image_codec.VideoWriter_fourcc(*'mp4v'), VideoCapture\n")
    assert walls_problems(sample, WALLS_PACKAGE) == [
        (8, "torch"), (9, "splatpipe.depth.files"), (10, "splatpipe.depth.fit.fit"),
        (11, "splatpipe.depth"), (12, "cv2.VideoWriter"),
        (14, "image_codec.VideoCapture"), (16, "VideoCapture"),
        (16, "image_codec.VideoWriter_fourcc")]


def test_the_walls_step_imports_only_stdlib_numpy_scipy_cv2_and_splatpipe() -> None:
    """Req 1.11, 2.13, 22.8: and nothing from splatpipe.depth, and no cv2 video class. The files
    are found with glob, and the test fails when it finds none."""
    paths = walls_files()
    assert paths, ("splatpipe/walls/ holds no .py file, so the walls import check has nothing to "
                   "read. Make splatpipe/walls/__init__.py first.")
    found: list[tuple[str, str]] = []
    for path in paths:
        shown = path.relative_to(REPO_ROOT).as_posix()
        found += [(f"{shown}:{line}", what)
                  for line, what in walls_problems(path.read_bytes(), WALLS_PACKAGE)]
    assert found == []


def test_no_train_file_imports_the_walls_step() -> None:
    """Req 22.8: every splatpipe/train/*.py, found with glob, gpu.py included."""
    paths = sorted(SPLATPIPE.glob("train/*.py"))
    assert paths, "splatpipe/train/ holds no .py file, so this check has nothing to read."
    found = [(where, name) for path in paths for where, name in imports_of(path)
             if within(name, WALLS_PACKAGE)]
    assert found == []


def test_importing_splatpipe_walls_and_its_modules_loads_no_gpu_package() -> None:
    """Req 1.11, 22.8. The modules are every splatpipe/walls/*.py but __init__.py, found with
    glob, so a module a later task adds is imported as it lands. On a laptop, where torch, gsplat
    and moge aren't installed, a module that imported one would fail the fresh process."""
    modules = [f"{WALLS_PACKAGE}.{path.stem}" for path in walls_files()
               if path.stem != "__init__"]
    result = subprocess.run([sys.executable, "-c", LOADED_BY_WALLS, *modules], cwd=REPO_ROOT,
                            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
                            capture_output=True, encoding="utf-8", timeout=60)
    assert result.returncode == 0, result.stderr
    loaded = json.loads(result.stdout)
    assert {WALLS_PACKAGE, *modules} <= set(loaded)
    assert [name for name in loaded if top(name) in NEVER_LOADED_BY_WALLS] == []


# The pool tests (Req 5.1, 5.4, 5.6 to 5.8)

# The worker count the pool tests start, and the step test's real pool (Req 22.6).
TWO_WORKERS = 2
# The frame the stopping and failing worker functions act on, and the other one.
BAD_FRAME = "000002.jpg"
GOOD_FRAME = "000001.jpg"
# What the ExpectedFailure and ValueError raised in a worker say.
EXPECTED_MESSAGE = "The frame 000002.jpg has no texture. The job trains on its depth prior."
VALUE_MESSAGE = "a bad number"
# The fake clock's start, for InlinePool and for the step runs.
START = datetime(2026, 11, 1, 12, 0, tzinfo=timezone.utc)
# The thread-limit variable the test process sets for itself and the one it unsets (C15).
OWN_SET = "OMP_NUM_THREADS"
OWN_UNSET = "OPENBLAS_NUM_THREADS"
OWN_VALUE = "7"


@dataclass(frozen=True)
class FrameTask:
    """A pool task naming its frame, as the step's tasks do. At top level, so spawn unpickles it."""

    name: str


def frame_of(task: FrameTask) -> str:
    """A worker function that gives back its task's frame name."""
    return task.name


def exit_on_bad_frame(task: FrameTask) -> str:
    """A worker function whose process ends with no result on BAD_FRAME, as one the kernel kills."""
    if task.name == BAD_FRAME:
        os._exit(3)
    return task.name


def expected_failure_on_bad_frame(task: FrameTask) -> str:
    """A worker function that raises an ExpectedFailure on BAD_FRAME."""
    if task.name == BAD_FRAME:
        raise ExpectedFailure(EXPECTED_MESSAGE)
    return task.name


def value_error_on_bad_frame(task: FrameTask) -> str:
    """A worker function that raises a ValueError, a bug, on BAD_FRAME."""
    if task.name == BAD_FRAME:
        raise ValueError(VALUE_MESSAGE)
    return task.name


def thread_limits(task: FrameTask) -> dict[str, str | None]:
    """A worker function that gives the worker's thread-limit variables, None for an unset one."""
    return {name: os.environ.get(name) for name in THREAD_LIMITS}


@pytest.fixture
def real_pool() -> Iterator[Pool]:
    """open_pool(2), closed after the test whatever it did, and no child left (Req 5.6)."""
    pool = open_pool(TWO_WORKERS)
    try:
        yield pool
    finally:
        pool.close()
    assert multiprocessing.active_children() == []


def two_tasks() -> tuple[FrameTask, FrameTask]:
    return FrameTask(GOOD_FRAME), FrameTask(BAD_FRAME)


@pytest.mark.parametrize(("workers", "frames", "affinity", "cpus", "expected"), [
    (3, 12, {0, 1, 2, 3, 4, 5}, 8, 3),       # the setting, when above 0
    (0, 12, None, 8, 8),                     # every vCPU: cpu_count with no sched_getaffinity
    (0, 12, {0, 1, 2, 3, 4, 5}, 8, 6),       # the affinity count beats cpu_count
    (0, 4, {0, 1, 2, 3, 4, 5}, 8, 4),        # never more than the frames
    (9, 4, None, 8, 4),
    (0, 12, None, None, 1),                  # cpu_count unknown
    (2, 0, None, 8, 1),                      # at least 1
])
def test_worker_count_is_the_setting_or_every_vcpu_capped_by_the_frames(
        monkeypatch: pytest.MonkeyPatch, workers: int, frames: int, affinity: set[int] | None,
        cpus: int | None, expected: int) -> None:
    """Req 5.1, A16."""
    if affinity is None:
        monkeypatch.delattr(os, "sched_getaffinity", raising=False)
    else:
        monkeypatch.setattr(os, "sched_getaffinity", lambda pid: set(affinity), raising=False)
    monkeypatch.setattr(os, "cpu_count", lambda: cpus)
    assert worker_count(workers, frames) == expected


class SteppedClock:
    """A fake clock that each task moves on by one minute, so the test knows when tasks end."""

    def __init__(self) -> None:
        self.time = START
        self.started: list[str] = []

    def now(self) -> datetime:
        return self.time

    def task(self, task: FrameTask) -> str:
        self.started.append(task.name)
        self.time += timedelta(minutes=1)
        return task.name.upper()


FIVE = tuple(FrameTask(name) for name in ("a.jpg", "b.jpg", "c.jpg", "d.jpg", "e.jpg"))


@pytest.mark.parametrize(("cutoff_minutes", "started", "timed_out"), [
    (2.5, ["a.jpg", "b.jpg", "c.jpg"], ("c.jpg", "d.jpg", "e.jpg")),  # c ends after the cutoff
    (2, ["a.jpg", "b.jpg"], ("b.jpg", "c.jpg", "d.jpg", "e.jpg")),    # b ends at it; c never starts
    (0, [], ("a.jpg", "b.jpg", "c.jpg", "d.jpg", "e.jpg")),          # nothing starts at the cutoff
    (10, ["a.jpg", "b.jpg", "c.jpg", "d.jpg", "e.jpg"], ()),
])
def test_inline_pool_starts_nothing_at_the_cutoff_and_times_out_a_task_that_ends_after_it(
        cutoff_minutes: float, started: list[str], timed_out: tuple[str, ...]) -> None:
    """Req 4.5, 4.6 with InlinePool (C6): a task ending at or after the cutoff is timed out, as the
    real pool would have ended it, and the finished ones come back by index, in order."""
    clock = SteppedClock()
    with InlinePool() as pool:
        mapped = pool.map_frames_until(clock.task, FIVE, START + timedelta(minutes=cutoff_minutes),
                                       clock.now)
    assert clock.started == started
    finished = len(FIVE) - len(timed_out)
    assert mapped == MapResult({index: FIVE[index].name.upper() for index in range(finished)},
                               timed_out)
    assert list(mapped.results) == list(range(finished))


def test_a_worker_that_exits_gives_worker_stopped_naming_its_frame(real_pool: Pool) -> None:
    """Req 5.7: the worker said it took BAD_FRAME before it died."""
    with pytest.raises(WorkerStopped) as raised:
        real_pool.map_frames(exit_on_bad_frame, two_tasks())
    assert raised.value.frame == BAD_FRAME
    assert str(raised.value) == stopped_message(BAD_FRAME)
    assert "Traceback" not in str(raised.value)


def test_a_workers_expected_failure_comes_back_with_the_same_message(real_pool: Pool) -> None:
    """Req 5.8."""
    with pytest.raises(ExpectedFailure) as raised:
        real_pool.map_frames(expected_failure_on_bad_frame, two_tasks())
    assert str(raised.value) == EXPECTED_MESSAGE


def test_a_workers_other_error_is_not_an_expected_failure(real_pool: Pool) -> None:
    """Req 5.8: the Step_Runner reports it as an Unexpected_Error."""
    with pytest.raises(Exception) as raised:
        real_pool.map_frames(value_error_on_bad_frame, two_tasks())
    assert not isinstance(raised.value, ExpectedFailure)
    assert "ValueError" in str(raised.value) and VALUE_MESSAGE in str(raised.value)


def test_workers_run_on_one_thread_and_the_test_process_keeps_its_own_limits(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """Req 5.4, 5.6, C15: each worker inherits "1" for every thread limit, while this process
    holds its own values again once the workers have started, one set and one unset, and
    close() leaves no child."""
    monkeypatch.setenv(OWN_SET, OWN_VALUE)
    monkeypatch.delenv(OWN_UNSET, raising=False)
    pool = open_pool(TWO_WORKERS)
    try:
        assert os.environ.get(OWN_SET) == OWN_VALUE
        assert OWN_UNSET not in os.environ
        limits = pool.map_frames(thread_limits, two_tasks())
        assert pool.map_frames(frame_of, two_tasks()) == [GOOD_FRAME, BAD_FRAME]
    finally:
        pool.close()
    assert limits == [{name: "1" for name in THREAD_LIMITS}] * 2
    assert os.environ.get(OWN_SET) == OWN_VALUE
    assert OWN_UNSET not in os.environ
    assert multiprocessing.active_children() == []


# The two-worker test (Req 5.9, 22.6)

# The Walls_Record's fields that may differ between two runs of the same Job: its times and its
# worker count.
RECORD_TIMES = frozenset({"started_utc", "cap_deadline", "walls_stop_time", "solve_cutoff",
                          "workers"})


@dataclass(frozen=True)
class WallsOutput:
    """What one walls run wrote: each Walls_Prior_File's arrays as (dtype, shape, bytes), by file
    name, the Walls_Record without RECORD_TIMES, and the names of every file it wrote."""

    priors: dict[str, dict[str, tuple[str, tuple[int, ...], bytes]]]
    record: dict[str, Any]
    files: list[str]


def walls_output(job_folder: Path) -> WallsOutput:
    """Read what the Walls_Step wrote into `job_folder`'s work/walls/ and out/walls/."""
    work = job_folder / "work" / "walls"
    priors: dict[str, dict[str, tuple[str, tuple[int, ...], bytes]]] = {}
    for path in sorted(work.glob("*.npz")):
        with np.load(path, allow_pickle=False) as data:
            priors[path.name] = {name: (data[name].dtype.str, data[name].shape,
                                        data[name].tobytes()) for name in data.files}
    record = json.loads((work / "walls.json").read_text(encoding="utf-8"))
    files = sorted(path.relative_to(job_folder).as_posix()
                   for folder in (work, job_folder / "out" / "walls") if folder.exists()
                   for path in folder.rglob("*") if path.is_file())
    return WallsOutput(priors, {key: value for key, value in record.items()
                                if key not in RECORD_TIMES}, files)


def run_walls(job_id: str, monkeypatch: pytest.MonkeyPatch,
              pool: Callable[[int], Pool]) -> dict[str, Any]:
    """Run the Walls_Step on `job_id` through splatpipe.walls.run with `pool` as its Worker_Pool
    and a clock fixed at START, and check that it's done."""
    import splatpipe.walls as walls

    monkeypatch.setattr(walls, "WALLS_POOL", pool)
    monkeypatch.setattr(walls, "utc_now", lambda: START)
    report = walls.run(job_id)
    assert report["outcome"] == "done", report["error"]
    return report


def test_two_workers_write_the_same_walls_priors_as_the_in_process_pool(
        new_job: Callable[..., str], monkeypatch: pytest.MonkeyPatch) -> None:
    """Req 5.9, 22.6, C14: the textured plane's Job, first with InlinePool and then with
    open_pool(2), writes the same Walls_Prior_Files bit for bit and the same Walls_Record but its
    times, and leaves no worker running."""
    from camera_files import write_walls_job
    from splatpipe.walls.crossframe import PASSED

    job_id = new_job()
    job_folder = load_config().jobs().jobs_folder / job_id
    job = write_walls_job(job_folder, "textured")
    opened: list[int] = []

    def two_workers(workers: int) -> Pool:
        opened.append(workers)
        return open_pool(TWO_WORKERS)

    run_walls(job_id, monkeypatch, lambda workers: InlinePool())
    inline = walls_output(job_folder)
    run_walls(job_id, monkeypatch, two_workers)
    spawned = walls_output(job_folder)
    assert multiprocessing.active_children() == []
    assert len(opened) == 1
    assert sorted(inline.priors) == sorted(f"{name.removesuffix('.jpg')}.npz"
                                           for name in job.names)
    # Some Walls_Prior comes from a Walls_Depth, so the comparison isn't of copies only.
    assert any(frame["outcome"] == PASSED for frame in inline.record["frames"])
    assert spawned.priors == inline.priors
    assert spawned.record == inline.record
    assert spawned.files == inline.files