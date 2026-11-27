"""The walls step's CLI: `uv run python -m splatpipe.walls --job <id> [--preset <name>]`.

It prints `walls: done` and exits with 0 when the step is done. When the step fails, or the Job
can't be opened, it prints the Readable_Error on stderr, with no traceback, and exits with 1
(walls spec, Req 1.2). A bad command line exits with 2.
"""

from __future__ import annotations

import sys

from splatpipe.step import step_main


def main(argv: list[str] | None = None) -> int:
    """Run the walls step with the command line `argv`, or sys.argv[1:] when it's None."""
    return step_main("walls", "splatpipe.walls", argv)


if __name__ == "__main__":
    sys.exit(main())
