"""Tests for the docs. Each test reads its doc by path from the repo root, so it checks the real file.

The tests of the Pipeline_Doc's ## Step: frames section also read the real config.toml by path,
and take the Frames_Record's top-level fields from what splatpipe.frames.files.frames_record
gives. So the section has to name the keys the real [frames] holds and the fields the step writes.
The ## Step: train section's Config section test reads the real [train] the same way, so that part
has to name every key the real [train] and its preset tables hold, and the whole section has to
name the depth-priors spec's 15 [train] keys, each of which the real [train] must hold. The
## Step: depth section's Config section test reads the real [depth] the same way.

The Step_Reports' counts and timings are written out here, as the step tests write them out:
tests/test_frames_step.py, tests/test_train_step.py and tests/test_depth_step.py check that a real
run's report holds exactly those names.
"""

import re
import tomllib
from pathlib import Path

import numpy as np

from splatpipe.frames.choose import Kept
from splatpipe.frames.files import frames_record
from splatpipe.frames.mask import MaskChoice
from splatpipe.frames.scores import Scores
from splatpipe.run import STEP_LIST

ROOT = Path(__file__).resolve().parents[1]  # this file is in tests/, at the repo root
LICENCE_LIST = ROOT / "docs" / "third-party-licences.md"
PIPELINE_DOC = ROOT / "docs" / "pipeline.md"
STRUCTURE_STEERING = ROOT / ".kiro" / "steering" / "structure.md"
# Read by path, because tests/conftest.py points splatpipe.config.CONFIG_PATH at a test config.
REAL_CONFIG = ROOT / "config.toml"


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


def test_numpy_and_headless_opencv_say_yes_and_opencv_never_reads_or_writes_video() -> None:
    """Frames spec, Req 12.6. Training spec, Req 13.16: the opencv-python-headless row also names
    the train step's JPEG frames and PNG masks. Depth-priors spec, Req 17.7: it also names the
    depth step's JPEG frames, and still says it never reads or writes video."""
    for name in ("numpy", "opencv-python-headless"):
        assert licence_row(name)["Installed"] == "Yes", f"The {name} row should say Yes"
    how = licence_row("opencv-python-headless")["How we use it"]
    train = ("The train step decodes the camera solve's JPEG frames and PNG masks and scales "
             "them down")
    depth = "The depth step decodes the camera solve's JPEG frames and scales them down"
    for words in (train, depth):
        assert words in how, f"The opencv-python-headless row should say {words!r}"
    assert "never reads or writes video" in how, (
        "The opencv-python-headless row should say it never reads or writes video")


def test_the_ffmpeg_row_names_decoding_the_used_part_zscale_and_tonemap() -> None:
    """Frames spec, Req 12.6."""
    how = licence_row("ffmpeg and ffprobe")["How we use it"]
    for words in ("ffmpeg decodes the frames of the Used_Part", "zscale", "tonemap"):
        assert words in how, f"The ffmpeg and ffprobe row should say {words!r}"


def test_the_zimg_row_names_the_wtfpl_and_zscale() -> None:
    """Frames spec, Req 12.7. Samarth OKed zimg (O1)."""
    row = licence_row("zimg (libzimg)")
    assert "WTFPL" in row["Licence"], "The zimg (libzimg) row should give the WTFPL"
    assert "zscale" in row["How we use it"], "The zimg (libzimg) row should say zscale uses it"


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
    """Req 13.2, 13.3. Frames spec, Req 11.8: with frames in the Step_List, this finds ## Step: frames."""
    parts = pipeline_parts()
    for name in ("fake", *(entry.name for entry in STEP_LIST)):
        doc_part(parts, 2, f"Step: {name}")


def test_the_step_list_part_names_the_frames_step_as_the_first_required_entry() -> None:
    """Frames spec, Req 12.5. Training spec, Req 13.8, 13.14: the list holds the frames step and
    the train step, then the clean step, and the frames spec's three sentences and the old one
    on a later spec adding the clean step's entry are gone."""
    step_list = doc_part(pipeline_parts(), 2, "The step list")
    phrases = ("The list holds the frames step and the train step, both required, and then the "
               "clean step, optional",
               "The camera solve spec puts its entry between the frames step and the train step, "
               "so the order becomes frames, cameras, train, clean",
               "Until the camera solve spec adds its entry, a job runs the frames step and then "
               "fails at the train step, because the camera solve output is missing")
    missing = [words for words in phrases if words not in step_list]
    assert not missing, f"The part headed The step list should say {missing}"
    gone = ("The list holds the frames step as its first, required entry",
            "Each later step spec adds its own entry after the frames step",
            "a job is done once the frames step is done",
            "The list starts empty", "No steps are set up yet",
            "a later spec adds the clean step's optional entry")
    left = [words for words in gone if words in step_list]
    assert not left, f"The part headed The step list should no longer say {left}"


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


# The ## Step: frames section (frames spec, Req 12.1–12.4, 12.8)

FRAMES = "Step: frames"
# The ### parts of a Step's section, the same as the Fake_Step's (Req 12.1).
STEP_PARTS = ("What it reads", "What it writes", "Config section", "Step report fields",
              "Files in out/")
# The Frames_Report's counts (Req 10.1) and timings (Req 10.2).
FRAMES_COUNTS = ("decoded_frames", "still_start_frames", "still_end_frames", "kept_frames",
                 "jolt_dropped_frames", "target_picks", "picks", "mask_samples", "mask_pixels",
                 "mask_used", "pick_width", "pick_height")
FRAMES_TIMINGS = ("probe", "score_pass", "choose", "pick_pass")


def frames_part(title: str) -> str:
    """The text of the ### part with this title in the ## Step: frames section."""
    section = doc_part(pipeline_parts(), 2, FRAMES)
    return doc_part(doc_parts(section), 3, title, f"The ## {FRAMES} section")


def frames_record_fields() -> list[str]:
    """The Frames_Record's top-level fields, as frames_record gives them for a Used_Part from 0 s
    to 1 s with one Score_Frame, kept and picked, and no Mask."""
    one = np.zeros(1)  # one Score_Frame, at 0 s, with scores of 0
    record = frames_record(
        colour_transfer=None, tone_mapped=False,
        kept=Kept(first=0, last=0, start_seconds=0.0, end_seconds=1.0, cut_start_seconds=0.0,
                  cut_end_seconds=0.0),
        target_picks=1, jolt_filter_skipped=False, pick_width=1, pick_height=1,
        mask=MaskChoice(samples=0, stuck=None, coverage=0.0, used=False),
        scores=Scores(times=one, sharpness=one, motion=one, contrast=one), picks=[0])
    return list(record)


def test_the_frames_section_comes_after_the_skeleton_headings_and_has_the_five_parts() -> None:
    """Frames spec, Req 12.1."""
    parts = pipeline_parts()
    section = doc_parts(doc_part(parts, 2, FRAMES))
    for title in STEP_PARTS:
        doc_part(section, 3, title, f"The ## {FRAMES} section")
    titles = [title for level, title, _ in parts if level == 2]
    earlier = titles[:titles.index(FRAMES)]
    missing = [title for title in SKELETON_HEADINGS if title not in earlier]
    assert not missing, (f"docs/pipeline.md should have these ## headings before ## {FRAMES}: "
                         f"{missing}")


def test_the_frames_files_in_out_part_says_it_writes_no_files_there_but_its_report() -> None:
    """Frames spec, Req 12.2."""
    out = frames_part("Files in out/")
    phrases = ("writes no files to `out/` but its report", "which the step runner writes")
    missing = [words for words in phrases if words not in out]
    assert not missing, f"The frames section's Files in out/ part should say {missing}"


def test_the_frames_what_it_reads_part_says_the_machine_image_needs_ffmpeg_5_1_or_newer() -> None:
    """Frames spec, Req 12.8."""
    reads = frames_part("What it reads")
    phrases = ("machine image needs ffmpeg and ffprobe 5.1 or newer", "`-fps_mode`",
               "Ubuntu 24.04's packaged ffmpeg, version 6.1, is new enough")
    missing = [words for words in phrases if words not in reads]
    assert not missing, f"The frames section's What it reads part should say {missing}"


def test_the_frames_what_it_writes_part_names_the_files_the_mask_and_every_record_field() -> None:
    """Frames spec, Req 7.10, 12.3. Training spec, Req 13.12: the camera solve step passes the
    mask on to training."""
    writes = frames_part("What it writes")
    phrases = ("`images/`", "`grey/`", "`frames.json`", "`mask.png`",
               "`--ImageReader.camera_mask_path`",
               "the camera solve step passes it on to training as one mask per image in "
               "`work/cameras/masks/`",
               "Only matching uses the grey copies")
    fields = [f"`{field}`" for field in frames_record_fields()]
    missing = [words for words in (*phrases, *fields) if words not in writes]
    assert not missing, f"The frames section's What it writes part should name {missing}"


def test_the_frames_config_section_names_every_real_key_and_the_machine_needs() -> None:
    """Frames spec, Req 9.10, 12.1, 12.4."""
    config = frames_part("Config section")
    section = tomllib.loads(REAL_CONFIG.read_text(encoding="utf-8"))["frames"]
    keys = [f"`{key}`" for key, value in section.items() if not isinstance(value, dict)]
    phrases = ("`base_picks`", "16 vCPUs", "16 GB", "The step itself never checks the machine")
    missing = [words for words in (*keys, *phrases) if words not in config]
    assert not missing, f"The frames section's Config section part should name {missing}"


def test_the_frames_step_report_fields_part_names_every_count_and_the_four_timings() -> None:
    """Frames spec, Req 12.1, with the counts of Req 10.1 and the timings of Req 10.2."""
    report = frames_part("Step report fields")
    names = [*(f"`counts.{name}`" for name in FRAMES_COUNTS),
             *(f"`timings.{name}`" for name in FRAMES_TIMINGS)]
    missing = [name for name in names if name not in report]
    assert not missing, f"The frames section's Step report fields part should name {missing}"


# The ## Step: train section (training spec, Req 13.1, 13.5 to 13.7, 13.14)

TRAIN = "Step: train"
# The Train_Report's counts (Req 10.1) and timings (Req 10.2).
TRAIN_COUNTS = ("images", "masked_images", "image_width", "image_height", "sparse_points",
                "start_splats", "splats", "steps_planned", "steps_run", "sh_degree",
                "gpu_memory_bytes", "peak_gpu_memory_bytes", "ply_bytes")
TRAIN_TIMINGS = ("load", "train", "write_ply", "check_ply")


def train_part(title: str) -> str:
    """The text of the ### part with this title in the ## Step: train section."""
    section = doc_part(pipeline_parts(), 2, TRAIN)
    return doc_part(doc_parts(section), 3, title, f"The ## {TRAIN} section")


def test_the_train_section_comes_after_the_frames_section_and_has_the_five_parts() -> None:
    """Training spec, Req 13.1."""
    parts = pipeline_parts()
    doc_part(parts, 2, FRAMES)
    section = doc_parts(doc_part(parts, 2, TRAIN))
    for title in STEP_PARTS:
        doc_part(section, 3, title, f"The ## {TRAIN} section")
    titles = [title for level, title, _ in parts if level == 2]
    assert titles.index(FRAMES) < titles.index(TRAIN), (
        f"docs/pipeline.md should have ## {TRAIN} after ## {FRAMES}")


def test_the_train_config_section_names_every_real_key_its_tables_and_the_machine_needs() -> None:
    """Training spec, Req 13.5, 13.14: the keys of the real [train] and of its preset tables."""
    config = train_part("Config section")
    section = tomllib.loads(REAL_CONFIG.read_text(encoding="utf-8"))["train"]
    presets = {name: table for name, table in section.items() if isinstance(table, dict)}
    shared = [key for key, value in section.items() if not isinstance(value, dict)]
    per_preset = list(dict.fromkeys(key for table in presets.values() for key in table))
    tables = ["[train]", *(f"[train.{name}]" for name in presets)]
    names = [f"`{name}`" for name in (*shared, *per_preset, *tables)]
    phrases = ("4 vCPUs", "32 GB", "24 GB", "48 GB")
    missing = [words for words in (*names, *phrases) if words not in config]
    assert not missing, f"The train section's Config section part should name {missing}"


def test_the_train_step_report_fields_part_names_every_count_and_the_four_timings() -> None:
    """Training spec, Req 13.6, 13.14, with the counts of Req 10.1 and the timings of Req 10.2."""
    report = train_part("Step report fields")
    names = [*(f"`counts.{name}`" for name in TRAIN_COUNTS),
             *(f"`timings.{name}`" for name in TRAIN_TIMINGS)]
    missing = [name for name in names if name not in report]
    assert not missing, f"The train section's Step report fields part should name {missing}"


def test_the_train_files_in_out_part_names_the_splat_and_the_clean_steps_limit() -> None:
    """Training spec, Req 13.7, 13.14 (D4): the largest splat budget is within the clean step's."""
    out = train_part("Files in out/")
    phrases = ("`out/train.ply`", "3,000,000 splats", "8,000,000 splats", "`max_source_splats`")
    missing = [words for words in phrases if words not in out]
    assert not missing, f"The train section's Files in out/ part should name {missing}"


# The ## Step: depth section (depth-priors spec, Req 17.1, 17.2, 17.9)

DEPTH = "Step: depth"
# The Depth_Report's counts (Req 9.1) and timings (Req 9.2).
DEPTH_COUNTS = ("images", "prior_width", "prior_height", "fitted_images", "few_points_images",
                "flat_images", "poor_fit_images", "peak_gpu_memory_bytes")
DEPTH_TIMINGS = ("load_model", "infer", "fit", "write")
# The Prior_Index and a Prior_File's three arrays (Req 7.1, 7.3).
PRIOR_NAMES = ("`work/depth/depth.json`", "`inverse_depth`", "`normal`", "`mask`")


def depth_part(title: str) -> str:
    """The text of the ### part with this title in the ## Step: depth section."""
    section = doc_part(pipeline_parts(), 2, DEPTH)
    return doc_part(doc_parts(section), 3, title, f"The ## {DEPTH} section")


def test_the_depth_section_comes_after_frames_and_before_train_and_has_the_five_parts() -> None:
    """Depth-priors spec, Req 17.1, 17.9."""
    parts = pipeline_parts()
    section = doc_parts(doc_part(parts, 2, DEPTH))
    for title in STEP_PARTS:
        doc_part(section, 3, title, f"The ## {DEPTH} section")
    titles = [title for level, title, _ in parts if level == 2]
    assert titles.index(FRAMES) < titles.index(DEPTH) < titles.index(TRAIN), (
        f"docs/pipeline.md should have ## {DEPTH} after ## {FRAMES} and before ## {TRAIN}")


def test_the_depth_config_section_names_every_real_key_and_its_tables() -> None:
    """Depth-priors spec, Req 17.1, 17.9: the keys of the real [depth] and of its preset tables."""
    config = depth_part("Config section")
    section = tomllib.loads(REAL_CONFIG.read_text(encoding="utf-8"))["depth"]
    presets = {name: table for name, table in section.items() if isinstance(table, dict)}
    shared = [key for key, value in section.items() if not isinstance(value, dict)]
    per_preset = list(dict.fromkeys(key for table in presets.values() for key in table))
    tables = ["[depth]", *(f"[depth.{name}]" for name in presets)]
    names = [f"`{name}`" for name in (*shared, *per_preset, *tables)]
    missing = [words for words in names if words not in config]
    assert not missing, f"The depth section's Config section part should name {missing}"


def test_the_depth_step_report_fields_part_names_every_count_and_the_four_timings() -> None:
    """Depth-priors spec, Req 17.9, with the counts of Req 9.1 and the timings of Req 9.2."""
    report = depth_part("Step report fields")
    names = [*(f"`counts.{name}`" for name in DEPTH_COUNTS),
             *(f"`timings.{name}`" for name in DEPTH_TIMINGS)]
    missing = [name for name in names if name not in report]
    assert not missing, f"The depth section's Step report fields part should name {missing}"


def test_the_depth_what_it_writes_part_names_the_prior_index_and_the_three_arrays() -> None:
    """Depth-priors spec, Req 17.2, 17.9: the Prior_Index and the Prior_File's arrays."""
    writes = depth_part("What it writes")
    missing = [words for words in PRIOR_NAMES if words not in writes]
    assert not missing, f"The depth section's What it writes part should name {missing}"


# The 15 [train] keys of the depth-priors spec's Req 14.1, each checked against the real [train].
PRIOR_TRAIN_KEYS = ("plain_gradient", "depth_gradient_falloff", "depth_weight_start",
                    "depth_weight_end", "depth_weight_floor", "depth_error_cap", "normal_weight",
                    "normal_start_step", "normal_every", "flatten_reg", "depth_normal_weight",
                    "normal_consistency_weight", "depth_normal_start_fraction",
                    "backproject_share", "backproject_max")
# The scale-up of a Prior to the Training_Size and the Prior_Pixels (Req 10.6, 10.7).
SCALE_UP_PHRASES = ("scaled up bilinearly", "unit length", "scaled up nearest-neighbour",
                    "counts as a prior pixel")


def test_the_train_section_names_the_prior_keys_and_counts_and_depth_names_the_scale_up() -> None:
    """Depth-priors spec, Req 17.2, 17.3, 17.9: ## Step: train names every new key of the real
    [train], the prior index, counts.prior_images and the Req 10.3 warning, and the depth
    section's What it writes part names the bilinear and nearest-neighbour scale-up."""
    real = tomllib.loads(REAL_CONFIG.read_text(encoding="utf-8"))["train"]
    absent = [key for key in PRIOR_TRAIN_KEYS if key not in real]
    assert not absent, f"The real [train] should hold {absent}"
    section = doc_part(pipeline_parts(), 2, TRAIN)
    names = [*(f"`{key}`" for key in PRIOR_TRAIN_KEYS), "`work/depth/depth.json`",
             "`counts.prior_images`", "Training ran without depth priors."]
    missing = [words for words in names if words not in section]
    assert not missing, f"The ## {TRAIN} section should name {missing}"
    writes = depth_part("What it writes")
    missing = [words for words in SCALE_UP_PHRASES if words not in writes]
    assert not missing, f"The depth section's What it writes part should name {missing}"


# The smoke test and cap deadline parts of ## Running a job in the cloud (training spec,
# Req 12.13, 13.9, 13.11)

CLOUD = "Running a job in the cloud"


def test_the_smoke_test_part_names_the_training_check_train_steps_and_cloud_smoke() -> None:
    """Training spec, Req 12.13, 13.11: the smoke test part lists the training check and says its
    numbers are in [cloud.smoke]."""
    cloud = doc_parts(doc_part(pipeline_parts(), 2, CLOUD))
    smoke = doc_part(cloud, 3, "The smoke test", f"The ## {CLOUD} section")
    phrases = ("training check", "`train_steps`", "`[cloud.smoke]`")
    missing = [words for words in phrases if words not in smoke]
    assert not missing, f"The part headed The smoke test should name {missing}"


def test_the_cap_deadline_part_names_clean_reserve_minutes_and_no_longer_says_no_step_reads_it() -> None:
    """Training spec, Req 13.9: the train step reads the cap deadline."""
    cloud = doc_parts(doc_part(pipeline_parts(), 2, CLOUD))
    deadline = doc_part(cloud, 3, "The cap deadline", f"The ## {CLOUD} section")
    assert "`clean_reserve_minutes`" in deadline, (
        "The part headed The cap deadline should name `clean_reserve_minutes`")
    assert "No current step reads it" not in deadline, (
        "The part headed The cap deadline should no longer say 'No current step reads it'")


# The gsplat-training skill (training spec, Req 13.13)

GSPLAT_TRAINING_SKILL = ROOT / ".kiro" / "skills" / "gsplat-training" / "SKILL.md"


def test_the_gsplat_training_skill_names_out_train_ply_and_no_longer_out_splat_ply() -> None:
    """Training spec, Req 13.13: the clean step reads the splat the train step writes."""
    skill = GSPLAT_TRAINING_SKILL.read_text(encoding="utf-8")
    assert "`out/train.ply`" in skill, "The gsplat-training skill should name `out/train.ply`"
    assert "out/splat.ply" not in skill, (
        "The gsplat-training skill should no longer name out/splat.ply")
