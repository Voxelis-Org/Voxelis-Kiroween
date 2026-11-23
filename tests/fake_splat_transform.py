r"""A stand-in for splat-transform, which the clean step's tests run in place of its cli.mjs.

A test gives export_sog a find that returns (sys.executable, this file), so export_sog runs this
script with the argument list it builds for splat-transform, through the real subprocess call,
its timeout, its kill and its decoding, on a machine with no Node (Req 10.6). The script ignores
every argument but the last, the .sog it was told to write, and does what FAKE_SPLAT_TRANSFORM in
its environment says:

- ok writes OK_BYTES, a few bytes, to the .sog and exits with 0.
- fail prints FAIL_CHARS characters of lines, with \r progress redraws and colour codes, to stdout
  and stderr both, as real tools do, and exits with FAIL_EXIT_CODE. Its very last line is
  LAST_LINE, on stderr, in plain text.
- silent exits with 0 and writes nothing.
- empty writes a .sog of 0 bytes and exits with 0.
- hang prints one line on stdout, PID_PREFIX and its process ID, so a test can check that the
  process is gone once it's stopped, then sleeps for HANG_SECONDS, longer than any test's
  timeout. It's a plain sleep, so a kill stops it at once.

Any other value, or none, exits with USAGE_EXIT_CODE and one line on stderr, so a test that
forgets to set it fails loudly. So does a run with no arguments, which has no .sog to write.

The output goes straight to the streams' byte buffers, each piece flushed, so it's the same on
Windows and Linux, with no \r\n put in, and stdout and stderr keep their order when export_sog
merges them. The script imports only the standard library and does nothing on import, so a test
can import its constants. Its name doesn't start with test_, so pytest doesn't collect it.
"""

import itertools
import os
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Literal

# The environment variable that says what the stand-in does, and the values it takes.
MODE_VARIABLE = "FAKE_SPLAT_TRANSFORM"
MODES = ("ok", "fail", "silent", "empty", "hang")
# What ok writes to the .sog.
OK_BYTES = b"stand-in viewer copy\n"
# How many characters fail prints: more than the 2,000 of TAIL_CHARS in splatpipe/clean/export.py
# once the redraws and colour codes are gone, so export_sog has to cut the tail.
FAIL_CHARS = 3000
FAIL_EXIT_CODE = 3
# fail's very last line, which it prints on stderr with a \n after it.
LAST_LINE = "Error: the splat-transform stand-in failed on purpose (FAKE_SPLAT_TRANSFORM=fail)."
# How long hang sleeps, in seconds.
HANG_SECONDS = 60
# The start of hang's one line, which its process ID and a \n follow.
PID_PREFIX = "fake_splat_transform.py: hanging on purpose as process "
# The exit code for a missing or unknown mode, or no arguments.
USAGE_EXIT_CODE = 2

Stream = Literal["stdout", "stderr"]


def say(stream: Stream, text: str) -> None:
    r"""Write text to sys.stdout or sys.stderr as UTF-8 bytes, with no \r\n put in, and flush it,
    so the two streams keep their order when export_sog merges them."""
    buffer = getattr(sys, stream).buffer
    buffer.write(text.encode("utf-8"))
    buffer.flush()


def progress_lines() -> Iterator[tuple[Stream, str]]:
    r"""Lines like a tool's progress, numbered and without end, with the stream each goes to: an
    info line with colour codes on stdout, and after every fourth a progress bar on stderr that
    redraws itself with \r and ends in colour."""
    for n in itertools.count(1):
        yield "stdout", (f"\x1b[32minfo\x1b[39m part {n:02d}: compressing the spherical harmonics "
                         f"of chunk {n:02d}\n")
        if n % 4 == 0:
            yield "stderr", (f"\rchunk {n:02d} [#####.....]  50%"
                             f"\rchunk {n:02d} [##########] \x1b[1;32m100%\x1b[0m\n")


def fail_output() -> list[tuple[Stream, str]]:
    """What fail prints, in order, with the stream each piece goes to: FAIL_CHARS characters in
    all, made of the progress lines that fit, a line of dots that fills what's left, and
    LAST_LINE."""
    output: list[tuple[Stream, str]] = []
    room = FAIL_CHARS - len(LAST_LINE) - 1  # LAST_LINE and its \n come last
    for stream, text in progress_lines():
        if len(text) >= room:  # so the dots line's \n always fits
            break
        output.append((stream, text))
        room -= len(text)
    output.append(("stdout", "." * (room - 1) + "\n"))
    output.append(("stderr", LAST_LINE + "\n"))
    return output


def refuse(problem: str) -> int:
    """Print one line on stderr that names this script and the problem, and give
    USAGE_EXIT_CODE."""
    say("stderr", f"fake_splat_transform.py: {problem}\n")
    return USAGE_EXIT_CODE


def main(argv: list[str]) -> int:
    """Do what FAKE_SPLAT_TRANSFORM says, with argv the arguments after the script, and give the
    exit code."""
    mode = os.environ.get(MODE_VARIABLE)
    if mode not in MODES:
        found = "isn't set" if mode is None else f"is {mode!r}"
        return refuse(f"{MODE_VARIABLE} {found}. Set it to {', '.join(MODES[:-1])} or "
                      f"{MODES[-1]}.")
    if not argv:
        return refuse("it got no arguments, so it has no .sog to write. Give it splat-transform's "
                      "arguments, with the .sog last.")
    target = Path(argv[-1])
    match mode:
        case "ok":
            target.write_bytes(OK_BYTES)
        case "fail":
            for stream, text in fail_output():
                say(stream, text)
            return FAIL_EXIT_CODE
        case "silent":
            pass
        case "empty":
            target.write_bytes(b"")
        case "hang":
            say("stdout", f"{PID_PREFIX}{os.getpid()}\n")
            time.sleep(HANG_SECONDS)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
