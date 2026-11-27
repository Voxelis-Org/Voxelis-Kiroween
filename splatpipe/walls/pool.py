"""The Walls_Step's Worker_Pool: the real `spawn` workers and the in-process stand-in (Req 5).

`open_pool(workers)` starts that many `multiprocessing.get_context("spawn").Process` workers
(C7), not a `multiprocessing.Pool`, which waits forever for a task whose worker was killed, and
not a `ProcessPoolExecutor`, which can't end a running task at the Solve_Cutoff. Each worker has
its own duplex pipe to the main process: the main process puts the worker's next task on it, and
the worker says which task it took, then gives back the result or the error it raised. A pipe
per worker means that ending one worker at the Solve_Cutoff can only break that worker's own
pipe, which is thrown away with it, and the main process waits with
`multiprocessing.connection.wait` on the pipes and the workers' sentinels together, so it wakes
for a result, for a worker that exited, or for the clock, and never spins (Req 4.6, 5.7).

Every worker starts with the one-thread environment variables set, which it inherits, and calls
`cv2.setNumThreads(1)` before its first task, so the workers don't oversubscribe the vCPUs
(Req 5.4, A16). The main process's own values are put back as soon as each worker has started,
unset ones unset, so the Train_Step and the Clean_Step that run after the Walls_Step in the same
process keep theirs (C15).

`InlinePool` runs each task in the calling process, with OpenCV on one thread while it runs
(C14), so the Test_Suite runs the whole step with no child process (Req 5.5, 22.5). It lives
here, beside the real pool, so both keep one cutoff rule (C6): a task starts only while
now < cutoff, the rule of `budget.may_start`, and one still running at the cutoff is a
Timed_Out_Frame.

A task names its frame through its `name` attribute, a str, as `solve.SolveTask` and the
cross-frame stages' tasks do. `map_frames_until` needs it for the Timed_Out_Frames' names; for
`map_frames` it's only used in messages, and a task without one counts as unnamed. The function
a task runs, `fn`, must be a module-level function, so a `spawn` worker imports it by name: the
workers import only this module, the function's module and what those import.

This module imports only the standard library, cv2 and `splatpipe.errors`.
"""
from __future__ import annotations

import os
import pickle
import traceback
from collections import deque
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from multiprocessing import connection, get_context
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from types import TracebackType
from typing import Any, Generic, Protocol, TypeVar

import cv2

from splatpipe.errors import ExpectedFailure, PipelineError

T = TypeVar("T")
R = TypeVar("R")

# The environment variables that limit numpy's BLAS and the other OpenMP libraries to one thread
# in each worker (Req 5.4, A16). Walls_Fixed_Constants: library names, not settings we tune.
THREAD_LIMITS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
ONE_THREAD = "1"

# The pool's own mechanics, not settings: the longest the main process waits between two reads
# of the clock it's given while tasks run before a cutoff, so a clock that isn't the wall clock
# is still read, and how long an ended worker gets to exit before it's killed.
CLOCK_CHECK_SECONDS = 1.0
END_GRACE_SECONDS = 5.0

_START_METHOD = "spawn"
_TOOK = "took"
_DONE = "done"
_ERROR = "error"


def stopped_message(frame: str | None) -> str:
    """Req 5.7's Readable_Error for a worker process that ended without its result, naming the
    Prior_Frame `frame` it had taken when that's known."""
    where = "its frames" if frame is None else f"the frame {frame!r}"
    return (f"A worker process stopped while the walls step solved {where}. The machine may have "
            "run out of memory. The job goes on and trains on the depth priors.")


class WorkerStopped(ExpectedFailure):
    """A worker process ended without giving back its result (Req 5.7). `frame` is the name of
    the Prior_Frame it had taken, or None when that isn't known."""

    def __init__(self, frame: str | None = None) -> None:
        self.frame = frame
        super().__init__(stopped_message(frame))

    def __reduce__(self) -> tuple[Any, ...]:
        return (type(self), (self.frame,))


class WorkerError(RuntimeError):
    """A worker's error that isn't a PipelineError. It isn't an ExpectedFailure, so the
    Step_Runner reports it as an Unexpected_Error (Req 5.8). Its cause holds the worker's
    traceback."""


class WorkerTraceback(Exception):
    """The traceback text of an error raised in a worker, set as the cause of the error the main
    process raises, so the report's Technical_Detail shows where the worker failed."""

    def __init__(self, text: str) -> None:
        super().__init__(f'\n"""\n{text}"""')


@dataclass(frozen=True)
class MapResult(Generic[R]):
    """What `map_frames_until` gives: the finished tasks' results by task index, in task order,
    and the `name`s of the Timed_Out_Frames, the tasks that didn't finish, in task order
    (Req 4.5, 4.6)."""

    results: dict[int, R]
    timed_out: tuple[str, ...]


class Pool(Protocol):
    """The Worker_Pool the Walls_Step reaches through `splatpipe.walls.WALLS_POOL` (Req 5.5).
    Use it in a `with` block, or call `close()` in a `finally`."""

    def map_frames(self, fn: Callable[[T], R], tasks: Sequence[T]) -> list[R]:
        """Run `fn(task)` for every task and give the results in task order."""
        ...

    def map_frames_until(self, fn: Callable[[T], R], tasks: Sequence[T], cutoff: datetime,
                         now: Callable[[], datetime]) -> MapResult[R]:
        """Start the tasks in order, none at or after `cutoff` by `now()`, end those still
        running at `cutoff`, and give the finished results and the Timed_Out_Frames."""
        ...

    def close(self) -> None:
        """End every worker, leaving none running (Req 5.6)."""
        ...

    def __enter__(self) -> Pool:
        return self

    def __exit__(self, kind: type[BaseException] | None, error: BaseException | None,
                 trace: TracebackType | None) -> None:
        self.close()


def worker_count(workers: int, frames: int) -> int:
    """How many workers the Worker_Pool gets (Req 5.1, A16): `workers` when it's above 0, else
    every vCPU this process may use, `len(os.sched_getaffinity(0))` where the system has it and
    `os.cpu_count()` otherwise, never more than the `frames` Prior_Frames and at least 1."""
    if workers > 0:
        count = workers
    else:
        affinity = getattr(os, "sched_getaffinity", None)
        count = len(affinity(0)) if affinity is not None else (os.cpu_count() or 1)
    return max(1, min(count, frames))


def open_pool(workers: int) -> Pool:
    """Start `workers` spawn worker processes, each on one thread, and give the pool (C7, C15)."""
    if workers < 1:
        raise ValueError(f"open_pool needs at least one worker, but it was given {workers}.")
    return ProcessPool(workers)


def frame_name(task: object) -> str | None:
    """The name of the frame `task` works on: its `name` attribute, or None when it has none."""
    name = getattr(task, "name", None)
    return name if isinstance(name, str) else None


def _required_name(task: object) -> str:
    """The frame name of a task given to `map_frames_until`, which needs it for the
    Timed_Out_Frames."""
    name = frame_name(task)
    if name is None:
        raise TypeError(f"map_frames_until needs every task to name its frame with a str `name` "
                        f"attribute, but {type(task).__name__} has none.")
    return name


@contextmanager
def one_thread_environment() -> Iterator[None]:
    """Set the THREAD_LIMITS to "1" in os.environ, then put the main process's values back,
    unset ones unset, so only a worker started inside the block inherits them (C15)."""
    saved = {name: os.environ.get(name) for name in THREAD_LIMITS}
    os.environ.update({name: ONE_THREAD for name in THREAD_LIMITS})
    try:
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


@contextmanager
def one_cv2_thread() -> Iterator[None]:
    """OpenCV on one thread inside the block, and the count it had put back after (C14)."""
    before = cv2.getNumThreads()
    cv2.setNumThreads(1)
    try:
        yield
    finally:
        cv2.setNumThreads(before)


class InlinePool(Pool):
    """The in-process stand-in for the Worker_Pool (C6): runs `fn(task)` in the calling process,
    with OpenCV on one thread (C14). An error `fn` raises reaches the caller as itself."""

    def map_frames(self, fn: Callable[[T], R], tasks: Sequence[T]) -> list[R]:
        with one_cv2_thread():
            return [fn(task) for task in tasks]

    def map_frames_until(self, fn: Callable[[T], R], tasks: Sequence[T], cutoff: datetime,
                         now: Callable[[], datetime]) -> MapResult[R]:
        """Asks `now()` before each task and starts none at or after `cutoff`, and asks again when
        a task ends: one that ends at or after `cutoff` is timed out, as the real pool would have
        ended it, and so is every task after it."""
        names = [_required_name(task) for task in tasks]
        results: dict[int, R] = {}
        with one_cv2_thread():
            for index, task in enumerate(tasks):
                if not now() < cutoff:
                    return MapResult(results, tuple(names[index:]))
                result = fn(task)
                if not now() < cutoff:
                    return MapResult(results, tuple(names[index:]))
                results[index] = result
        return MapResult(results, ())

    def close(self) -> None:
        """Nothing to end: no process was started."""


def _worker_loop(conn: Connection) -> None:
    """A worker process's whole life: one OpenCV thread, then tasks from `conn` until the None
    that stops it. For each task it says which one it took, runs it and gives back the result,
    or what it raised."""
    cv2.setNumThreads(1)
    while True:
        try:
            message = conn.recv()
        except EOFError:  # the main process closed its end: nothing more will come
            return
        if message is None:
            return
        call, index, payload = message
        conn.send((_TOOK, call, index))
        try:
            # Unpickled here, so a function or task this process can't import is the task's
            # error, reported as such, and not a worker that stopped.
            fn, task = pickle.loads(payload)
            result = fn(task)
        except Exception as error:
            conn.send((_ERROR, call, index, _failure(error)))
            continue
        try:
            conn.send((_DONE, call, index, result))
        except Exception as error:  # the result couldn't be pickled
            conn.send((_ERROR, call, index, _failure(error)))


def _failure(error: Exception) -> tuple[bool, PipelineError | None, str, str, str]:
    """What a worker sends back for `error`: whether it's a readable PipelineError, the error
    itself when it pickles back to the same message, its type, its message and its traceback."""
    readable = isinstance(error, PipelineError)
    copy: PipelineError | None = None
    if isinstance(error, PipelineError):
        try:
            if str(pickle.loads(pickle.dumps(error))) == str(error):
                copy = error
        except Exception:
            copy = None
    kind = type(error)
    return (readable, copy, f"{kind.__module__}.{kind.__qualname__}", str(error),
            "".join(traceback.format_exception(error)))


def _raised(failure: tuple[bool, PipelineError | None, str, str, str],
            frame: str | None) -> Exception:
    """The error the main process raises for a worker's `failure`: the same PipelineError, with
    the same message, or a WorkerError for anything else (Req 5.8)."""
    readable, copy, kind, message, text = failure
    if readable:
        error: Exception = copy if copy is not None else ExpectedFailure(message)
    else:
        where = "a frame" if frame is None else f"the frame {frame!r}"
        error = WorkerError(f"A walls worker raised {kind} on {where}: {message}")
    error.__cause__ = WorkerTraceback(text)
    return error


class _Worker:
    """The main process's handle on one worker: its process, its end of the pipe, the task index
    it was given and the one it said it took."""

    def __init__(self, process: BaseProcess, conn: Connection) -> None:
        self.process = process
        self.conn = conn
        self.given: int | None = None
        self.took: int | None = None
        self.ended = False

    def end(self) -> None:
        """Stop the process now, whatever it's doing, and close the pipe."""
        if self.process.is_alive():
            self.process.terminate()
        self.process.join(END_GRACE_SECONDS)
        if self.process.is_alive():
            self.process.kill()
            self.process.join()
        self.conn.close()
        self.ended = True


class ProcessPool(Pool):
    """The real Worker_Pool: `spawn` worker processes, each fed by its own pipe (C7)."""

    def __init__(self, workers: int) -> None:
        self._context = get_context(_START_METHOD)
        self._call = 0
        self._closed = False
        self._workers: list[_Worker] = []
        try:
            for _ in range(workers):
                self._workers.append(self._start())
        except BaseException:
            self.close()
            raise

    def _start(self) -> _Worker:
        """Start one worker with the one-thread variables set, then put the main process's
        values back (C15)."""
        mine, theirs = self._context.Pipe(duplex=True)
        process = self._context.Process(target=_worker_loop, args=(theirs,), daemon=True,
                                        name="walls-worker")
        try:
            with one_thread_environment():
                process.start()
        finally:
            theirs.close()  # the worker holds its own copy now
        return _Worker(process, mine)

    def map_frames(self, fn: Callable[[T], R], tasks: Sequence[T]) -> list[R]:
        results, _ = self._run(fn, tasks, None, None)
        return [results[index] for index in range(len(tasks))]

    def map_frames_until(self, fn: Callable[[T], R], tasks: Sequence[T], cutoff: datetime,
                         now: Callable[[], datetime]) -> MapResult[R]:
        names = [_required_name(task) for task in tasks]
        results, unfinished = self._run(fn, tasks, cutoff, now)
        return MapResult(results, tuple(names[index] for index in unfinished))

    def close(self) -> None:
        """Stop the idle workers with the None they wait for, end the busy ones, and leave no
        worker process running (Req 5.6). Closing twice does nothing more."""
        if self._closed:
            return
        self._closed = True
        for worker in self._workers:
            if worker.ended:
                continue
            if worker.given is None:
                try:
                    worker.conn.send(None)
                    worker.process.join(END_GRACE_SECONDS)
                except OSError:
                    pass
            worker.end()

    def _run(self, fn: Callable[[T], R], tasks: Sequence[T], cutoff: datetime | None,
             now: Callable[[], datetime] | None) -> tuple[dict[int, R], list[int]]:
        """Run the tasks, each on the first free worker in task order, and give the results by
        index and the indices of the tasks that didn't finish before `cutoff`, in order. With no
        `cutoff`, every task runs. A worker still running at `cutoff` is ended and a fresh one
        started in its place; after an error, the busy workers are ended and replaced when the
        pool is next used."""
        if self._closed:
            raise RuntimeError("The walls worker pool was used after it was closed.")
        self._call += 1
        self._replace_ended()
        pending = deque(range(len(tasks)))
        results: dict[int, R] = {}
        try:
            while pending or self._busy():
                timeout: float | None = None
                if cutoff is not None and now is not None:
                    current = now()
                    if not current < cutoff:  # budget.may_start's rule: never at the cutoff
                        break
                    left = (cutoff - current).total_seconds()
                    timeout = min(max(left, 0.0), CLOCK_CHECK_SECONDS)
                self._give(fn, tasks, pending)
                self._wait(tasks, results, timeout)
        except BaseException:
            self._end_busy()
            raise
        unfinished = sorted([worker.given for worker in self._busy()
                             if worker.given is not None] + list(pending))
        self._end_busy()
        self._replace_ended()
        return results, unfinished

    def _busy(self) -> list[_Worker]:
        return [worker for worker in self._workers if worker.given is not None]

    def _give(self, fn: Callable[[T], R], tasks: Sequence[T], pending: deque[int]) -> None:
        """Put the next pending tasks, in order, on the free workers' pipes. A free worker that
        has exited fails the step (Req 5.7)."""
        for worker in self._workers:
            if not pending:
                return
            if worker.given is not None:
                continue
            if not worker.process.is_alive():
                raise WorkerStopped(None)
            index = pending[0]
            payload = pickle.dumps((fn, tasks[index]))  # a task that won't pickle fails here
            try:
                worker.conn.send((self._call, index, payload))
            except (BrokenPipeError, ConnectionResetError, EOFError) as error:
                raise WorkerStopped(None) from error
            pending.popleft()
            worker.given = index
            worker.took = None

    def _wait(self, tasks: Sequence[T], results: dict[int, Any], timeout: float | None) -> None:
        """Wait up to `timeout` seconds, or until something happens when it's None, for a busy
        worker's message or exit, and handle what came: results first, then exits."""
        busy = self._busy()
        if not busy:
            return
        by_handle: dict[Any, _Worker] = {}
        for worker in busy:
            by_handle[worker.conn] = worker
            by_handle[worker.process.sentinel] = worker
        ready = connection.wait(list(by_handle), timeout)
        for worker in busy:
            if worker.conn in ready:
                self._receive(worker, tasks, results)
        for worker in busy:
            if worker.process.sentinel in ready:
                while worker.given is not None and worker.conn.poll():
                    self._receive(worker, tasks, results)
                if worker.given is not None:
                    raise self._stopped(worker, tasks)

    def _receive(self, worker: _Worker, tasks: Sequence[T], results: dict[int, Any]) -> None:
        """Read one message from `worker`: the task it took, its result, or its error, which is
        raised here (Req 5.8)."""
        try:
            message = worker.conn.recv()
        except (EOFError, ConnectionResetError, BrokenPipeError) as error:
            raise self._stopped(worker, tasks) from error
        kind, call, index = message[:3]
        if call != self._call or index != worker.given:
            return  # from a task this pool no longer waits for
        if kind == _TOOK:
            worker.took = index
        elif kind == _DONE:
            results[index] = message[3]
            worker.given = worker.took = None
        else:
            worker.given = worker.took = None
            raise _raised(message[3], frame_name(tasks[index]))

    def _stopped(self, worker: _Worker, tasks: Sequence[T]) -> WorkerStopped:
        """Req 5.7's failure for a worker that ended without its result, naming the frame it said
        it took."""
        frame = None if worker.took is None else frame_name(tasks[worker.took])
        worker.given = worker.took = None
        worker.end()
        return WorkerStopped(frame)

    def _end_busy(self) -> None:
        """End every worker still running a task; the next use of the pool replaces them."""
        for worker in self._busy():
            worker.end()
            worker.given = worker.took = None

    def _replace_ended(self) -> None:
        """Start a fresh worker in the place of each one that was ended or has exited."""
        for slot, worker in enumerate(self._workers):
            if worker.ended or not worker.process.is_alive():
                if not worker.ended:
                    worker.end()
                self._workers[slot] = self._start()
