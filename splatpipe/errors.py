"""Pipeline errors and the wrapper that every CLI body runs in.

A `PipelineError` carries a Readable_Error: what went wrong and what to do.
A Step raises `ExpectedFailure` for a failure it foresees. `run_cli` prints a
`PipelineError`'s message on stderr and a short bug message for any other
error, and returns 1 for both, never with a traceback. `SystemExit` (argparse's
usage errors and --help) and `KeyboardInterrupt` (Ctrl+C) pass through.

This is the first Skeleton module, so it imports no other splatpipe module.
"""

from __future__ import annotations

import sys
from collections.abc import Callable


# The cameras report and the website share this user-facing warning.
LOOK_UP_WARNING = ("The video never looked up, so the ceiling and the top of the walls will be "
                   "missing. Film a second slow loop with the phone tilted up about 30 degrees.")


class PipelineError(Exception):
    """A refusal or failure whose message is a Readable_Error: what went wrong and what to do."""


class ExpectedFailure(PipelineError):
    """Raised by a Step's own code for a failure it foresees, like too few sharp frames."""


def run_cli(action: Callable[[], int]) -> int:
    """Run a CLI body and return its exit code.

    PipelineError: print its message on stderr and return 1.
    Any other Exception: print a general message with the error's type and message, return 1.
    No traceback is printed in either case. SystemExit and KeyboardInterrupt aren't
    Exceptions, so they pass through: argparse's exit code 2 and Ctrl+C reach the caller.
    """
    try:
        return action()
    except PipelineError as error:
        message = str(error)
    except Exception as error:  # anything else is a bug in our code
        message = (f"splatpipe stopped because of a bug ({type(error).__name__}: {error}). "
                   "Tell the team the command you ran.")
    print(message, file=sys.stderr)
    return 1
