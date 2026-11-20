"""Tests for the docs. Each test reads its doc by path from the repo root, so it checks the real file."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # this file is in tests/, at the repo root
LICENCE_LIST = ROOT / "docs" / "third-party-licences.md"


def table_rows(text: str) -> list[list[str]]:
    """The stripped cells of each Markdown table row in the text, without the |---| divider rows."""
    rows = [[cell.strip() for cell in line.strip()[1:-1].split("|")]
            for line in text.splitlines() if line.startswith("|")]
    return [row for row in rows if not all(cell and not cell.strip("-:") for cell in row)]


def licence_row(name: str) -> dict[str, str]:
    """The Licence_List's one row whose Name cell is the name, with each cell under its heading."""
    header, *rows = table_rows(LICENCE_LIST.read_text(encoding="utf-8"))
    matches = [dict(zip(header, row, strict=True)) for row in rows if row[0] == name]
    assert len(matches) == 1, f"the Licence_List should have one {name!r} row, not {len(matches)}"
    return matches[0]


def test_the_ffmpeg_row_says_what_ffprobe_reads_and_that_the_tools_are_never_vendored_or_linked() -> None:
    """Req 13.6."""
    how = licence_row("ffmpeg and ffprobe")["How we use it"]
    assert "ffprobe reads the video's container, streams and length" in how
    assert "never vendored or linked" in how
