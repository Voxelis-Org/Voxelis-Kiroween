"""Tests for the docs. Each test reads its doc by path from the repo root, so it checks the real file.

The tests of the Pipeline_Doc's ## Step: frames section also read the real config.toml by path,
and take the Frames_Record's top-level fields from what splatpipe.frames.files.frames_record
gives. So the section has to name the keys the real [frames] holds and the fields the step writes.
The tests of its ## Step: cameras section do the same with the real [cameras], and take the
Solve_Record's top-level fields and the Viewer_Cameras' fields from what
splatpipe.cameras.outputs.solve_record and viewer_cameras give for a small example.
The ## Step: train section's Config section test reads the real [train] the same way, so that part
has to name every key the real [train] and its preset tables hold, and the whole section has to
name the depth-priors spec's 15 [train] keys, each of which the real [train] must hold, and its
five Prior_Terms. The
## Step: depth section's Config section test reads the real [depth] the same way. The step list
test takes its entries from the real STEP_LIST, and the skill tests read the gsplat-training and
moge-depth skills by path.

The Step_Reports' counts and timings are written out here, as the step tests write them out:
tests/test_frames_step.py, tests/test_train_step.py and tests/test_depth_step.py check that a real
run's report holds exactly those names.
"""

import re
import tomllib
from pathlib import Path

import numpy as np

from splatpipe.cameras.level import LevelMove, Levelling
from splatpipe.cameras.model import POINT2D, Camera, Image, Model, Points
from splatpipe.cameras.outputs import solve_record, viewer_cameras
from splatpipe.cameras.solve import ModelCount, Solved
from splatpipe.frames.choose import Kept
from splatpipe.frames.files import frames_record, pick_name
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


# The seven COLMAP commands the Cameras_Step runs (cameras spec, Req 15.8).
COLMAP_COMMANDS = ("feature_extractor", "sequential_matcher", "view_graph_calibrator",
                   "global_mapper", "mapper", "image_undistorter", "image_undistorter_standalone")
# The COLMAP row's Installed cell: Homebrew's COLMAP on the laptop, and the machine image that
# has COLMAP 4.2.1 and passed its smoke test (cameras spec, A15).
COLMAP_INSTALLED = ("Yes, on the laptop through Homebrew and on the machine image "
                    "`ami-0fe14782b05fe63b2`.")


def test_the_colmap_row_names_the_cameras_step_commands_and_homebrews_colmap_on_laptops() -> None:
    """Cameras spec, Req 15.8, 16.12 (D18): Homebrew's COLMAP runs the laptop tests, the
    integration test's whole solve included, as Samarth OKed."""
    row = licence_row("COLMAP 4.2.1")
    how = row["How we use it"]
    commands = [f"`{command}`" for command in COLMAP_COMMANDS]
    phrases = ("never vendored or linked", "no pycolmap",
               "Homebrew's COLMAP 4.2.1, which holds SiftGPU, CGAL and LSD", "`-h`", "`use_gpu 0`",
               "the integration test's whole solve included", "as Samarth OKed")
    missing = [words for words in (*commands, *phrases) if words not in how]
    assert not missing, f"The COLMAP 4.2.1 row should say {missing}"
    for old in ("is used only for `-h` and small CPU tests", "the next image build adds it"):
        assert old not in how, f"The COLMAP 4.2.1 row should no longer say {old!r}"
    # The image built on 24 November has every flag, -DDOWNLOAD_ENABLED=OFF included.
    flags = ("the image built on 2026-11-24 (`ami-0300759bb382305d7`) has all seven "
             "`_ENABLED=OFF` flags")
    assert flags in how, f"The COLMAP 4.2.1 row should say {flags!r}"
    assert row["Installed"] == COLMAP_INSTALLED, (
        f"The COLMAP 4.2.1 row's Installed cell should say {COLMAP_INSTALLED!r}")


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
    """Req 13.2, 13.3. Frames spec, Req 11.8, and cameras spec, Req 14.8: with frames, cameras,
    train and clean in the Step_List, this finds ## Step: frames, ## Step: cameras, ## Step: train
    and ## Step: clean."""
    parts = pipeline_parts()
    for name in ("fake", *(entry.name for entry in STEP_LIST)):
        doc_part(parts, 2, f"Step: {name}")


def test_the_step_list_part_names_the_frames_cameras_depth_train_and_clean_entries_in_order() -> None:
    """Frames spec, Req 12.5. Training spec, Req 13.8, 13.14. Cameras spec, Req 14.8, 15.7.
    Depth-priors spec, Req 17.4, 17.9: the list holds the frames step and the cameras step, both
    required, the depth step, optional, the train step, required, and then the clean step,
    optional, each written as the real Step_List's entry. The frames spec's three sentences, the
    old one on a later spec adding the clean step's entry, and every side's words on the camera
    solve spec or the training spec still to add its entry are gone."""
    step_list = doc_part(pipeline_parts(), 2, "The step list")
    phrases = ("The list holds the frames step and the cameras step, both required, the depth "
               "step, optional, the train step, required, and then the clean step, optional",
               "A job is done once all five steps are done, and done with warnings when the three "
               "required steps are done and the depth step or the clean step fails")
    missing = [words for words in phrases if words not in step_list]
    assert not missing, f"The part headed The step list should say {missing}"
    # Each entry as the code writes it, in the real Step_List's order, with frames first.
    entries = [f'`StepEntry("{entry.name}", "{entry.module}", {entry.required})`'
               for entry in STEP_LIST]
    assert STEP_LIST[0].name == "frames" and STEP_LIST[0].required
    unnamed = [entry for entry in entries if entry not in step_list]
    assert not unnamed, f"The part headed The step list should name {unnamed}"
    places = [step_list.index(entry) for entry in entries]
    assert places == sorted(places), (
        f"The part headed The step list should name {entries} in that order")
    gone = ("The list holds the frames step and the train step, both required, and then the "
            "clean step, optional",
            "The camera solve spec puts its entry between the frames step and the train step",
            "so the order becomes frames, cameras, train, clean",
            "a job runs the frames step and then fails at the train step",
            "The list holds the frames step as its first, required entry",
            "Each later step spec adds its own entry after the frames step",
            "a job is done once the frames step is done",
            "The list starts empty", "No steps are set up yet",
            "a later spec adds the clean step's optional entry",
            "The camera solve spec puts its entry between the frames step and the train step",
            "Until the camera solve spec adds its entry",
            "Each later step spec adds its own entry after the cameras step",
            "Until the training spec adds its entry",
            "The camera solve spec puts its entry between the frames step and the depth step",
            "The list holds the frames step, required, the depth step, optional",
            "The list holds the frames step, the cameras step and the train step, all three "
            "required",
            "A job is done once all four steps are done")
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


# The ## Step: cameras section (cameras spec, Req 15.1–15.6, 15.9, 15.10, 16.12)

CAMERAS = "Step: cameras"
# The Work_Folder's parts (Req 11.1) and the Model_Files of the Levelled_Model (Req 10.5).
WORK_FOLDER_PARTS = ("`colmap/`", "`images/`", "`masks/`", "`sparse/`", "`solve.json`")
MODEL_FILES = ("`cameras.bin`", "`images.bin`", "`points3D.bin`")
# The Cameras_Report's counts (Req 13.1) and timings (Req 13.2): one for each COLMAP_Command,
# with the two Match_Runs as sequential_matcher_near and sequential_matcher_far, then the stages.
CAMERAS_COUNTS = ("picks", "placed_frames", "points3d", "models_found", "fallback_ran",
                  "kept_from_global", "levelled")
CAMERAS_TIMINGS = ("feature_extractor", "sequential_matcher_near", "sequential_matcher_far",
                   "view_graph_calibrator", "global_mapper", "mapper", "image_undistorter",
                   "image_undistorter_standalone", "matching", "mapping", "undistortion",
                   "levelling", "writing")


def cameras_part(title: str) -> str:
    """The text of the ### part with this title in the ## Step: cameras section."""
    section = doc_part(pipeline_parts(), 2, CAMERAS)
    return doc_part(doc_parts(section), 3, title, f"The ## {CAMERAS} section")


def one_frame_model() -> Model:
    """A Levelled_Model with one PINHOLE camera and one Placed_Frame, Pick 0, with no 2D points,
    and no points."""
    camera = Camera(camera_id=1, model_id=1, width=2, height=2,  # model 1 is PINHOLE
                    params=(1.0, 1.0, 1.0, 1.0))
    image = Image(image_id=1, qvec=(1.0, 0.0, 0.0, 0.0), tvec=(0.0, 0.0, 0.0), camera_id=1,
                  name=pick_name(0), points2d=np.zeros(0, dtype=POINT2D))
    points = Points(ids=np.zeros(0, dtype="<u8"), xyz=np.zeros((0, 3)),
                    rgb=np.zeros((0, 3), dtype="u1"), errors=np.zeros(0), tracks=())
    return Model(cameras=(camera,), images=(image,), points=points)


def solve_record_fields() -> list[str]:
    """The Solve_Record's top-level fields, as solve_record gives them for a RADIAL camera whose
    Calibration worked, then a global mapper whose one Model places Pick 0, and a Levelling that
    moved the scene with the identity as its Level_Move."""
    placed = ModelCount(number=0, placed=1)
    solved = Solved(calibration_ran=True, calibration_failure=None, mapper="global",
                    fallback_reason=None, models=(placed,), final=placed, folder=Path("0"),
                    model=one_frame_model(), placed_names=frozenset({pick_name(0)}))
    levelling = Levelling(move=LevelMove(rotation=np.eye(3), translation=np.zeros(3)),
                          levelled=True, reason=None, mean_up_length=1.0,
                          scene_centre="axis_point", heading_camera=pick_name(0))
    return list(solve_record(solved, levelling, "RADIAL", None))


def viewer_cameras_fields() -> list[str]:
    """The Viewer_Cameras' fields, as viewer_cameras gives them for one_frame_model, levelled:
    the top-level fields, then the camera's, then those of the frame of Pick 0."""
    viewer = viewer_cameras(one_frame_model(), {pick_name(0): 0}, levelled=True)
    return [*viewer, *viewer["camera"], *viewer["frames"][0]]


def test_the_cameras_section_comes_after_the_frames_section_and_has_the_five_parts() -> None:
    """Cameras spec, Req 15.1, 16.12."""
    parts = pipeline_parts()
    section = doc_parts(doc_part(parts, 2, CAMERAS))
    for title in STEP_PARTS:
        doc_part(section, 3, title, f"The ## {CAMERAS} section")
    titles = [title for level, title, _ in parts if level == 2]
    earlier = titles[:titles.index(CAMERAS)]
    missing = [title for title in (*SKELETON_HEADINGS, FRAMES) if title not in earlier]
    assert not missing, (f"docs/pipeline.md should have these ## headings before ## {CAMERAS}: "
                         f"{missing}")


def test_the_cameras_what_it_reads_part_names_the_frames_files_and_what_colmap_reads() -> None:
    """Cameras spec, Req 15.2, 16.12."""
    reads = cameras_part("What it reads")
    phrases = ("`frames.json`", "`images/`", "`grey/`", "`mask.png`",
               "`feature_extractor` reads the grey copies",
               "`global_mapper`, `mapper` and `image_undistorter` read the colour picks",
               "machine image needs COLMAP 4.2.1", "`.kiro/steering/tech.md`")
    missing = [words for words in phrases if words not in reads]
    assert not missing, f"The cameras section's What it reads part should say {missing}"


def test_the_cameras_what_it_writes_part_names_the_files_the_frame_and_every_record_field() -> None:
    """Cameras spec, Req 15.3, 16.12. The World_Up's minus is U+2212, as the doc writes it."""
    writes = cameras_part("What it writes")
    phrases = (*WORK_FOLDER_PARTS, *MODEL_FILES, "PINHOLE", "(0, −1, 0)", "+Z", "the origin",
               "no pycolmap")
    fields = [f"`{field}`" for field in solve_record_fields()]
    missing = [words for words in (*phrases, *fields) if words not in writes]
    assert not missing, f"The cameras section's What it writes part should name {missing}"


def test_the_cameras_config_section_names_every_real_key_and_the_machine_needs() -> None:
    """Cameras spec, Req 12.11, 15.4, 16.12."""
    config = cameras_part("Config section")
    section = tomllib.loads(REAL_CONFIG.read_text(encoding="utf-8"))["cameras"]
    keys = [f"`{key}`" for key, value in section.items() if not isinstance(value, dict)]
    phrases = ("`near_overlap`", "16 vCPUs", "16 GB", "The step itself never checks the machine")
    missing = [words for words in (*keys, *phrases) if words not in config]
    assert not missing, f"The cameras section's Config section part should name {missing}"


def test_the_cameras_step_report_fields_part_names_every_count_and_timing() -> None:
    """Cameras spec, Req 15.5, 16.12, with the counts of Req 13.1 and the timings of Req 13.2."""
    report = cameras_part("Step report fields")
    names = [*(f"`counts.{name}`" for name in CAMERAS_COUNTS),
             *(f"`timings.{name}`" for name in CAMERAS_TIMINGS)]
    missing = [name for name in names if name not in report]
    assert not missing, f"The cameras section's Step report fields part should name {missing}"


def test_the_cameras_files_in_out_part_names_every_viewer_field_and_when_the_file_changes() -> None:
    """Cameras spec, Req 15.6, 16.12: a run the step runner stops on the config also leaves the
    earlier `work/cameras/`, and one it stops later leaves only `out/cameras.json`."""
    out = cameras_part("Files in out/")
    phrases = ("`out/cameras.json`",
               "removes an earlier `out/cameras.json` when its own code starts",
               "writes a new one last of all",
               "A run that the step runner stops before the step's code starts",
               "it also leaves the earlier run's `work/cameras/`",
               "so only `out/cameras.json` is left")
    fields = [f"`{field}`" for field in viewer_cameras_fields()]
    missing = [words for words in (*phrases, *fields) if words not in out]
    assert not missing, f"The cameras section's Files in out/ part should say {missing}"
    old = "leaves the earlier run's `work/cameras/` and `out/cameras.json` in place"
    assert old not in out, f"The cameras section's Files in out/ part should no longer say {old!r}"


def test_the_cameras_section_names_radial_fx_fy_the_cap_deadline_and_the_refine_refusal() -> None:
    """Cameras spec, Req 15.9, 15.10, 16.12 (D17, D19, D20): the Camera_Models are RADIAL and
    RADIAL_FISHEYE, the levelled camera has fx = fy, each Time_Limit is capped at the time left
    before the Cap_Deadline, and the settings check refuses a false refine key. No OPENCV is left,
    and no words that the step doesn't read the Cap_Deadline."""
    section = doc_part(pipeline_parts(), 2, CAMERAS)
    phrases = ('`"RADIAL"`', '`"RADIAL_FISHEYE"`', "fx = fy",
               "When `SPLATPIPE_CAP_DEADLINE` is set, the step caps each limit at the time left "
               "before the cap deadline",
               "`global_mapper_ba_refine_extra_params` and `mapper_ba_refine_extra_params` must be "
               "true, and the settings check refuses false")
    missing = [words for words in phrases if words not in section]
    assert not missing, f"The ## {CAMERAS} section should say {missing}"
    gone = ("OPENCV", "The step doesn't read `SPLATPIPE_CAP_DEADLINE`")
    left = [words for words in gone if words in section]
    assert not left, f"The ## {CAMERAS} section should no longer say {left}"


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
# The five Prior_Terms the Trainer adds to the Loss_Table with a Prior_Set (Req 13.1).
PRIOR_TERMS = ("depth", "normal", "flatten", "depth_normal", "normal_consistency")
# The scale-up of a Prior to the Training_Size and the Prior_Pixels (Req 10.6, 10.7).
SCALE_UP_PHRASES = ("scaled up bilinearly", "unit length", "scaled up nearest-neighbour",
                    "counts as a prior pixel")


def test_the_train_section_names_the_prior_keys_and_counts_and_depth_names_the_scale_up() -> None:
    """Depth-priors spec, Req 17.2, 17.3, 17.9: ## Step: train names every new key of the real
    [train], the five Prior_Terms, the prior index, counts.prior_images,
    counts.backprojected_splats and the Req 10.3 warning, and the depth section's What it writes
    part names the bilinear and nearest-neighbour scale-up."""
    real = tomllib.loads(REAL_CONFIG.read_text(encoding="utf-8"))["train"]
    absent = [key for key in PRIOR_TRAIN_KEYS if key not in real]
    assert not absent, f"The real [train] should hold {absent}"
    section = doc_part(pipeline_parts(), 2, TRAIN)
    names = [*(f"`{key}`" for key in PRIOR_TRAIN_KEYS), *(f"`{term}`" for term in PRIOR_TERMS),
             "`work/depth/depth.json`",
             "`counts.prior_images`", "`counts.backprojected_splats`",
             "Training ran without depth priors."]
    missing = [words for words in names if words not in section]
    assert not missing, f"The ## {TRAIN} section should name {missing}"
    writes = depth_part("What it writes")
    missing = [words for words in SCALE_UP_PHRASES if words not in writes]
    assert not missing, f"The depth section's What it writes part should name {missing}"


# The smoke test and cap deadline parts of ## Running a job in the cloud (training spec,
# Req 12.13, 13.9, 13.11; cameras spec, Req 15.10, 15.11; depth-priors spec, Req 16.12, 17.5)

CLOUD = "Running a job in the cloud"


def smoke_part() -> str:
    """The text of the ### The smoke test part in the ## Running a job in the cloud section."""
    cloud = doc_parts(doc_part(pipeline_parts(), 2, CLOUD))
    return doc_part(cloud, 3, "The smoke test", f"The ## {CLOUD} section")


def test_the_smoke_test_part_names_the_cameras_and_training_checks_and_their_cloud_smoke_keys() -> None:
    """Training spec, Req 12.13, 13.11, and cameras spec, Req 15.11, 16.12: the smoke test part
    lists the cameras check and the training check and says their numbers are in [cloud.smoke]."""
    smoke = smoke_part()
    phrases = ("cameras check", "`cameras_frames`", "training check", "`train_steps`",
               "`[cloud.smoke]`")
    missing = [words for words in phrases if words not in smoke]
    assert not missing, f"The part headed The smoke test should name {missing}"


def test_the_smoke_test_part_names_the_depth_check_and_depth_normal_start_step() -> None:
    """Depth-priors spec, Req 16.12, 17.5, 17.9: the smoke test part lists the depth check and says
    its number, depth_normal_start_step, is in [cloud.smoke]."""
    smoke = smoke_part()
    phrases = ("depth check", "`depth_normal_start_step`", "`[cloud.smoke]`")
    missing = [words for words in phrases if words not in smoke]
    assert not missing, f"The part headed The smoke test should name {missing}"


def test_the_cap_deadline_part_names_clean_reserve_minutes_and_that_the_cameras_step_reads_it() -> None:
    """Training spec, Req 13.9: the train step reads the cap deadline. Cameras spec, Req 15.10,
    16.12: the cameras step reads it too."""
    cloud = doc_parts(doc_part(pipeline_parts(), 2, CLOUD))
    deadline = doc_part(cloud, 3, "The cap deadline", f"The ## {CLOUD} section")
    assert "`clean_reserve_minutes`" in deadline, (
        "The part headed The cap deadline should name `clean_reserve_minutes`")
    cameras = "The cameras step reads it too"
    assert cameras in deadline, f"The part headed The cap deadline should say {cameras!r}"
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


# The moge-depth skill (depth-priors spec, Req 17.8)

MOGE_DEPTH_SKILL = ROOT / ".kiro" / "skills" / "moge-depth" / "SKILL.md"


def test_the_moge_depth_skill_says_what_the_depth_priors_spec_decided() -> None:
    """Depth-priors spec, Req 17.8: the skill names `moge_weights` in `[depth]` and the image's
    `--no-deps` installs, and its sentences on what the image didn't have yet, the missing licence
    rows, the code's row covering the three git dependencies and the depth-priors spec deciding are
    gone. So is "Step 7 of the image build" for the weights' download, which is step 8 now that
    step 7 installs MoGe's code."""
    skill = MOGE_DEPTH_SKILL.read_text(encoding="utf-8")
    for words in ("`moge_weights` in `[depth]`", "`--no-deps`"):
        assert words in skill, f"The moge-depth skill should name {words}"
    gone = ("The image doesn't have MoGe's code yet", "huggingface-hub has none yet",
            "the code's row covers the three git dependencies", "The depth-priors spec decides",
            "Step 7 of the image build")
    lowered = skill.lower()  # so a phrase counts at a sentence's start or inside one
    left = [words for words in gone if words.lower() in lowered]
    assert not left, f"The moge-depth skill should no longer say {left}"


# The colmap-cli skill (cameras spec, Req 15.12)

COLMAP_CLI_SKILL = ROOT / ".kiro" / "skills" / "colmap-cli" / "SKILL.md"
COLMAP_OPTIONS = ROOT / ".kiro" / "skills" / "colmap-cli" / "references" / "options.md"
# The words in each file that name the two Camera_Models and when camera_params goes in.
COLMAP_CLI_PHRASES = {
    COLMAP_CLI_SKILL: ("RADIAL, one focal length f for square pixels, or RADIAL_FISHEYE for very "
                       "wide lenses",
                       "The cameras step passes it only for RADIAL_FISHEYE"),
    COLMAP_OPTIONS: ("`--ImageReader.camera_model` takes `camera_model`, RADIAL or RADIAL_FISHEYE",
                     "`--ImageReader.camera_params` goes in only for RADIAL_FISHEYE"),
}
# A "Left out:" note: its bullet line and the lines indented under it.
LEFT_OUT_NOTE = re.compile(r"^- Left out:.*(?:\n  .*)*", re.MULTILINE)


def test_the_colmap_cli_skill_names_both_camera_models_and_no_longer_leaves_out_camera_params() -> None:
    """Cameras spec, Req 15.12, 16.12 (D19): both files say the step passes `camera_model`, RADIAL
    or RADIAL_FISHEYE, and `--ImageReader.camera_params` only for RADIAL_FISHEYE, so no note of
    options.md lists `--ImageReader.camera_params` as left out. Each file's text is read with its
    line breaks as spaces, so a phrase that wraps still matches."""
    for path, phrases in COLMAP_CLI_PHRASES.items():
        text = " ".join(path.read_text(encoding="utf-8").split())
        missing = [words for words in phrases if words not in text]
        assert not missing, f"{path.relative_to(ROOT)} should say {missing}"
    notes = LEFT_OUT_NOTE.findall(COLMAP_OPTIONS.read_text(encoding="utf-8"))
    assert notes, "The colmap-cli skill's options.md should have its Left out notes"
    listed = [note for note in notes if "`--ImageReader.camera_params`" in note]
    assert not listed, (
        f"No Left out note of the colmap-cli skill's options.md should list "
        f"`--ImageReader.camera_params`, but these do: {listed}")
