"""The fake step's CLI: `uv run python -m splatpipe.fake --job <id> [--preset <name>]`.

It prints `fake: done` and exits with 0 when the step is done. When the step fails, or the Job
can't be opened, it prints the Readable_Error on stderr and exits with 1. A bad command line
exits with 2.
"""

from __future__ import annotations

import sys

from splatpipe.step import step_main


def main(argv: list[str] | None = None) -> int:
    """Run the fake step with the command line `argv`, or sys.argv[1:] when it's None."""
    return step_main("fake", "splatpipe.fake", argv)


if __name__ == "__main__":
    sys.exit(main())
