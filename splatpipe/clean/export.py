"""Making the Viewer_Copy: splat-transform 3.8.0 turns the Clean_Splat into one bundled .sog.

`find_splat_transform` finds `node` on the PATH and the splat-transform that `npm ci` installed
under the repo root, and says which of them is missing, or that both are (Req 5.10).
`splat_transform_argv` builds the command, and `export_sog` runs it and turns each way it can
fail into a Readable_Error (Req 5.11 to 5.13).

The command starts `node` itself, with splat-transform's bin/cli.mjs as its script, as an
argument list with no shell (Req 5.2, 5.8). npm's node_modules/.bin/splat-transform is a .cmd
shim on Windows, which needs a shell, so starting `node` gives one code path on the Windows
laptops and the Linux job machine, and the child a timeout kills is `node` itself. `-g cpu` puts
splat-transform's GPU work on the CPU, so the step needs no GPU and no Vulkan driver (Req 5.3).

splat-transform's stdout and stderr are caught together, in the order it printed them, and
written whole to the log file in the Work_Folder. A Readable_Error holds only the end of them,
at most TAIL_CHARS characters, cleaned up the way a terminal would show it (Req 5.11, 6.5).

From splatpipe, this module imports only splatpipe.config, splatpipe.errors and
splatpipe.clean.settings.
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

from splatpipe.clean.settings import CleanSettings
from splatpipe.config import REPO_ROOT
from splatpipe.errors import ExpectedFailure

# Format_Constants of splat-transform 3.8.0: its package.json has "engines": {"node": ">=22.0.0"},
# and bin/cli.mjs is its bin entry, which npm ci installs under the repo root (Req 5.2, 5.10).
NODE_VERSION = 22
CLI = Path("node_modules", "@playcanvas", "splat-transform", "bin", "cli.mjs")
# The most characters of splat-transform's output that a Readable_Error holds (Req 5.11, 6.5).
# It shapes a message and changes no result, so it stays in code.
TAIL_CHARS = 2000

# The step's lines land in out/job.log when the Job_Runner runs it, and nowhere when it runs alone.
_LOG = logging.getLogger(__name__)

# What find_splat_transform says is missing, and what to do about it (Req 5.10).
_NODE_MISSING = "Node isn't installed, or `node` isn't on the PATH"
_CLI_MISSING = f"`{CLI.as_posix()}` is missing"
_NO_NODE = (f"{_NODE_MISSING}. The clean step needs Node {NODE_VERSION} or newer for "
            "splat-transform. Install it, then run `npm ci` at the repo root.")
_NO_CLI = (f"splat-transform isn't installed: {_CLI_MISSING}. Run `npm ci` at the repo root, "
           f"with Node {NODE_VERSION} or newer, which splat-transform needs.")
_NO_NODE_OR_CLI = (f"{_NODE_MISSING}, and splat-transform isn't installed either: "
                   f"{_CLI_MISSING}. Install Node {NODE_VERSION} or newer, which splat-transform "
                   "needs, then run `npm ci` at the repo root.")

# An ANSI escape code: ESC, [, its parameters and its final character, like the colour code
# \x1b[31m or \x1b[2K, which clears a line.
_ESCAPE_CODE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
# Every control character, Unicode's category Cc, except \t and \n.
_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")


def find_splat_transform(repo_root: Path = REPO_ROOT) -> tuple[str, Path]:
    """`node`, as shutil.which finds it on the PATH, and splat-transform's bin/cli.mjs under
    `repo_root`, which `npm ci` installs there (Req 5.2).

    Both are looked for before either is refused, so the ExpectedFailure says which one is
    missing, or that both are, says to run `npm ci` at the repo root, and names the Node version
    splat-transform needs (Req 5.10).
    """
    node = shutil.which("node")
    cli = repo_root / CLI
    has_cli = cli.is_file()
    if node is None:
        raise ExpectedFailure(_NO_NODE if has_cli else _NO_NODE_OR_CLI)
    if not has_cli:
        raise ExpectedFailure(_NO_CLI)
    return node, cli


def splat_transform_argv(node: str, cli: Path, source: Path, target: Path, bands: int,
                         s: CleanSettings) -> list[str]:
    """The command that makes the Viewer_Copy `target` from the Clean_Splat `source`.

    splat-transform reads its command line by position, as [GLOBAL OPTIONS] input [ACTIONS]
    output, and an action applies to the input before it. So the global options come first: -w
    to overwrite, -g cpu for its GPU work on the CPU, -i for the SH compression iterations and
    --max-workers for its worker threads (Req 5.3, 5.6). Then the input, then -H, the short form
    of --filter-harmonics, which keeps `bands` bands of spherical harmonics (Req 5.4), and last
    the .sog. No other option goes in, so none that needs a GPU (Req 5.7).

    `source` and `target` are absolute, because the Work_Folder is under the resolved
    Job_Folder, so neither can start with - and be read as an option.
    """
    return [node, str(cli),
            "-w", "-g", "cpu", "-i", str(s.sh_iterations), "--max-workers", str(s.max_workers),
            str(source),
            "-H", str(bands),
            str(target)]


def export_sog(source: Path, target: Path, bands: int, s: CleanSettings, log_file: Path, *,
               run: Callable[..., subprocess.CompletedProcess[bytes]] = subprocess.run,
               find: Callable[[], tuple[str, Path]] = find_splat_transform) -> int:
    """Make the Viewer_Copy `target` from the Clean_Splat `source` with splat-transform, keeping
    `bands` bands of spherical harmonics, and return its size in bytes, for clean_sog_bytes
    (Req 5.1, 5.14).

    `find` gives `node` and bin/cli.mjs, and `run` runs the command, so a test can put a
    stand-in in place of either. The command runs as an argument list with no shell, in
    `target`'s folder, the Work_Folder, so any stray file splat-transform makes lands there, with
    no stdin and for at most splat_transform_timeout_seconds (Req 5.8, 5.12, 8.7). Its stderr
    goes into its stdout, so the output keeps the order it was printed in.

    The whole output, decoded as UTF-8 with any other bytes replaced, goes to `log_file` first,
    whatever happened. Then each failure is an ExpectedFailure (Req 8.8):

    - `node` can't be started: the node path and the OS's reason.
    - splat-transform runs past its time limit, and subprocess.run has killed it and waited for
      it: the limit, what to change, and the tail of the output (Req 5.12).
    - A non-zero exit code: the code, the tail, where the whole output is and the Node version
      it needs. -9 is Linux's signal 9, which often means the machine ran out of memory
      (Req 5.11).
    - An exit code of 0, but no file at `target`, or an empty one: that, and the tail (Req 5.13).

    One INFO line is logged as splat-transform starts, and one once it has ended.
    """
    node, cli = find()
    argv = splat_transform_argv(node, cli, source, target, bands, s)
    _LOG.info("splat-transform: started with %d bands of spherical harmonics, %d SH compression "
              "iterations and %d worker threads", bands, s.sh_iterations, s.max_workers)
    started = time.monotonic()
    try:
        result = run(argv, cwd=target.parent, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                     stderr=subprocess.STDOUT, timeout=s.splat_transform_timeout_seconds,
                     check=False)
    except OSError as error:
        _write_log(log_file, None)
        raise _not_started(node, error) from error
    except subprocess.TimeoutExpired as error:
        seconds = time.monotonic() - started
        output = _write_log(log_file, error.stdout)
        _log_end("stopped at its time limit", seconds, _size(target))
        raise _timed_out(s.splat_transform_timeout_seconds, output) from error
    seconds = time.monotonic() - started
    output = _write_log(log_file, result.stdout)
    size = _size(target)
    _log_end(f"exit code {result.returncode}", seconds, size)
    if result.returncode != 0:
        raise _failed(result.returncode, output, log_file)
    if size == 0:
        raise _no_copy(target, output, log_file)
    return size


# The output and the log

def _write_log(log_file: Path, output: bytes | None) -> str:
    """splat-transform's whole output, decoded as UTF-8 with any other bytes replaced, written
    to `log_file` with its line ends as they were, and given back.

    None is no output: a timeout gives it when nothing was printed, and an OSError has none.
    """
    text = (output or b"").decode("utf-8", errors="replace")
    log_file.write_text(text, encoding="utf-8", newline="\n")
    return text


def _size(path: Path) -> int:
    """The size in bytes of the file at `path`, or 0 when there's no file there."""
    return path.stat().st_size if path.is_file() else 0


def _log_end(how: str, seconds: float, size: int) -> None:
    """The INFO line for splat-transform's end: how it ended, the seconds and the bytes written.

    On Linux it also gives ru_maxrss for RUSAGE_CHILDREN, the peak memory of the largest child
    this process has waited for. Through the step's own CLI, the only child is splat-transform,
    so this is how the first real job measures it (Open item O1).
    """
    _LOG.info("splat-transform: %s after %.1f seconds, %d bytes written%s", how, seconds, size,
              _peak_memory())


def _peak_memory() -> str:
    """On Linux, the largest child's peak memory, as the end of the end line. Elsewhere, nothing.

    The resource module is imported only on Linux, because Windows has none.
    """
    if sys.platform != "linux":
        return ""
    import resource  # Linux only, so not at the top
    peak = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss  # in KiB on Linux
    return f", peak memory of the largest child process so far {peak} KiB"


def _tail(output: str) -> str:
    """The end of splat-transform's output for a Readable_Error, at most TAIL_CHARS characters,
    as a terminal would show it (Req 5.11).

    Three clean-ups come first: \\r\\n becomes \\n, each line keeps only what follows its last
    \\r, which is what a terminal shows once a progress bar has redrawn, and ANSI escape codes,
    colours included, and every other control character but \\n and \\t go. Then the tail is
    the last TAIL_CHARS characters, cut back to the start of a line, so it doesn't start partway
    through one. Characters with no line break among them are the end of one long line, and
    stay. Line breaks at its start and white space at its end go.
    """
    text = output.replace("\r\n", "\n")
    text = "\n".join(line.rpartition("\r")[2] for line in text.split("\n"))
    text = _CONTROL.sub("", _ESCAPE_CODE.sub("", text)).rstrip()
    tail = text[-TAIL_CHARS:]
    if len(tail) < len(text) and text[-len(tail) - 1] != "\n":
        tail = tail[tail.find("\n") + 1:]  # the part of a line it starts with goes
    return tail.lstrip("\n")


def _with_tail(message: str, output: str, after: str = "") -> str:
    """`message`, then the tail of splat-transform's output on lines of its own, when it printed
    anything, then `after`."""
    tail = _tail(output)
    if not tail:
        return f"{message} {after}".rstrip()
    return "\n".join(part for part in (f"{message} Its last lines were:", tail, after) if part)


# The Readable_Errors

def _not_started(node: str, error: OSError) -> ExpectedFailure:
    """`node` couldn't be started, like when it may not be run: the path and the OS's reason."""
    return ExpectedFailure(f"The clean step couldn't start Node at {node} to run splat-transform "
                           f"({error.strerror or error}). Check that `node --version` runs in a "
                           "terminal, then run the clean step again.")


def _timed_out(limit: float, output: str) -> ExpectedFailure:
    """splat-transform ran past its time limit and was stopped (Req 5.12)."""
    return ExpectedFailure(_with_tail(
        f"splat-transform didn't finish within {limit:g} seconds, so the clean step stopped it. "
        "Raise splat_transform_timeout_seconds in [clean], or use a preset with fewer "
        "sh_iterations.", output))


def _failed(code: int, output: str, log_file: Path) -> ExpectedFailure:
    """splat-transform exited with the non-zero code `code` (Req 5.11). A negative code on Linux
    is the signal that stopped it, and signal 9 is what the OOM killer sends."""
    signal = (" It was stopped by signal 9, which often means the machine ran out of memory."
              if code == -9 else "")
    return ExpectedFailure(_with_tail(
        f"splat-transform failed with exit code {code} while it made the viewer copy.{signal}",
        output,
        f"The whole output is in {log_file}. splat-transform 3.8.0 needs Node {NODE_VERSION} or "
        "newer, and `node --version` shows yours."))


def _no_copy(target: Path, output: str, log_file: Path) -> ExpectedFailure:
    """splat-transform exited with 0, but `target` is missing or empty (Req 5.13)."""
    return ExpectedFailure(_with_tail(
        f"splat-transform said it was done, but it wrote no viewer copy at {target}.", output,
        f"Check the whole output in {log_file}, then run the clean step again."))
