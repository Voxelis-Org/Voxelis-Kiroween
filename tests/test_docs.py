"""Tests for the docs. Each test reads its doc by path from the repo root, so it checks the real file."""

import re
from pathlib import Path

from splatpipe.run import STEP_LIST

ROOT = Path(__file__).resolve().parents[1]  # this file is in tests/, at the repo root
LICENCE_LIST = ROOT / "docs" / "third-party-licences.md"
PIPELINE_DOC = ROOT / "docs" / "pipeline.md"
STRUCTURE_STEERING = ROOT / ".kiro" / "steering" / "structure.md"


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


# The Skeleton's ## headings in docs/pipeline.md. They all come before the first ## Step: section.
SKELETON_HEADINGS = (
    "The job folder", "The job record", "Step reports", "Presets and config.toml", "The step list",
    "How a step is built", "Running one step", "The job runner", "The job summary", "The job log",
    "Running a job again", "A Step's files in out/", "CLIs and exit codes",
)
HEADING = re.compile(r"(#{1,6}) +(.+)")  # a Markdown heading: 1 to 6 #s, then a space and the title

Part = tuple[int, str, str]  # a heading's level, its title, and its part of the text


def doc_parts(text: str) -> list[Part]:
    """Each Markdown heading in the text, in order, with its part: the lines from the heading to the
    next heading of the same or a higher level, or to the end. A line inside a fenced code block is
    never a heading, so a `# comment` line in a TOML or shell example doesn't count as one."""
    lines = text.splitlines()
    marks: list[tuple[int, int, str]] = []  # each heading's line index, level and title
    fenced = False
    for index, line in enumerate(lines):
        if line.lstrip().startswith("```"):
            fenced = not fenced
        elif not fenced and (match := HEADING.match(line)):
            marks.append((index, len(match[1]), match[2].strip()))
    parts = []
    for n, (index, level, title) in enumerate(marks):
        end = next((later for later, other, _ in marks[n + 1:] if other <= level), len(lines))
        parts.append((level, title, "\n".join(lines[index:end])))
    return parts


def doc_part(parts: list[Part], level: int, title: str, where: str = "docs/pipeline.md") -> str:
    """The text of the one part in `parts` with this level and title."""
    found = [part[2] for part in parts if part[:2] == (level, title)]
    assert len(found) == 1, f"{where} should have one {'#' * level} {title} heading, not {len(found)}"
    return found[0]


def pipeline_parts() -> list[Part]:
    """The parts of docs/pipeline.md."""
    return doc_parts(PIPELINE_DOC.read_text(encoding="utf-8"))


def test_the_pipeline_doc_starts_with_the_skeleton_headings() -> None:
    """Req 13.1."""
    titles = [title for level, title, _ in pipeline_parts() if level == 2]
    steps_from = next((n for n, title in enumerate(titles) if title.startswith("Step: ")), len(titles))
    missing = [title for title in SKELETON_HEADINGS if title not in titles[:steps_from]]
    assert not missing, ("docs/pipeline.md should have these ## headings before its first Step "
                         f"section: {missing}")


def test_the_pipeline_doc_has_a_step_section_for_the_fake_step_and_every_listed_step() -> None:
    """Req 13.2, 13.3."""
    parts = pipeline_parts()
    for name in ("fake", *(entry.name for entry in STEP_LIST)):
        doc_part(parts, 2, f"Step: {name}")


def test_the_fake_section_says_the_step_specs_copy_it() -> None:
    """Req 13.3."""
    fake = doc_part(pipeline_parts(), 2, "Step: fake")
    assert "The step specs copy the fake step as their working example" in fake, (
        "The ## Step: fake section should say the step specs copy it as their working example")


def test_running_a_job_again_names_fresh_config_toml_and_running_a_step_by_hand() -> None:
    """Req 13.8."""
    again = doc_part(pipeline_parts(), 2, "Running a job again")
    for words in ("--fresh", "config.toml", "running an earlier step by hand"):
        assert words in again, f"The part headed Running a job again should say {words!r}"


def test_every_step_section_has_files_in_out_and_the_fake_one_lists_out_fake_txt() -> None:
    """Req 13.9."""
    parts = pipeline_parts()
    doc_part(parts, 2, "A Step's files in out/")
    for level, title, text in parts:
        if level == 2 and title.startswith("Step: "):
            doc_part(doc_parts(text), 3, "Files in out/", f"The ## {title} section")
    fake = doc_parts(doc_part(parts, 2, "Step: fake"))
    assert "out/fake.txt" in doc_part(fake, 3, "Files in out/", "The ## Step: fake section")


def test_the_structure_steering_lists_pipeline_md_and_says_step_specs_update_their_section() -> None:
    """Req 13.4, 13.5."""
    lines = STRUCTURE_STEERING.read_text(encoding="utf-8").splitlines()
    docs_lines = [line for line in lines if line.startswith("- `docs/`:")]
    assert len(docs_lines) == 1, f"structure.md should have one `docs/` line, not {len(docs_lines)}"
    assert "pipeline.md" in docs_lines[0], "structure.md's `docs/` line should name pipeline.md"
    others = [line for line in lines if line != docs_lines[0]]
    assert any("Every step spec updates its own section" in line for line in others), (
        "structure.md should have a line saying every step spec updates its own section")
