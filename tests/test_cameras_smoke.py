"""Tests for splatpipe/cameras/smoke.py, the Cameras_Smoke_Check's parts that need no COLMAP
(cameras spec, Req 16.13, 16.14, 18.3 to 18.8). None of them is skipped where colmap isn't on the
PATH.

- The scene's self-check, moved from tests/test_cameras_real.py, renders every Smoke_Pick with
  render_pick at the real [cloud.smoke] sizes, read by path: each one fills the frame, with every
  ray hitting a rectangle, sees all three rectangles and isn't uniform.
- The other tests use SIZES, 30 Picks like the real check but of 64 × 48 pixels with f = 52, the
  real 320 × 240 and 260 scaled by a fifth, so the view is the same and the tests stay fast. Its
  placed_share and focal_tolerance are binary fractions, so the pass limits are exact: 15 of the
  30 Picks, and 3.25 px either side of 52.
- write_smoke_picks writes a work/frames/ that read_frames accepts, with the black corner painted
  into every Colour_Pick and Grey_Copy, and mask.png 0 over it and 255 elsewhere.
- passed_options and the scripted -h texts take their oracle from colmap_options, which parses
  options.md: each command's block, with --ImageReader.camera_params too for feature_extractor,
  since the block leaves it out and the builders pass it for RADIAL_FISHEYE.
- check_options runs with smoke.run_colmap replaced by ScriptedHelp, which writes a scripted -h
  text to each run's Command_Log and gives its exit code and Last_Error_Line, as run_colmap does.
- run_smoke_solve runs the step's own work() with FakeColmap and the test config, in a scratch
  folder of tmp_path, with tmp_path as the working folder, so a relative write would show there.
"""

from __future__ import annotations

import math
import tomllib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from conftest import CAMERAS, FIXED_ID, FRAMES, NEAR_OVERLAP
from splatpipe.cameras import smoke
from splatpipe.cameras.colmap import ColmapRun, RunResult, exit_text, last_error_line
from splatpipe.cameras.frames_input import Pick, read_frames
from splatpipe.cameras.model import CAMERA_MODELS, read_model
from splatpipe.cameras.settings import read_settings
from splatpipe.cameras.smoke import (JOB, PAINTED_PART, RECTANGLES, SmokeSizes, SmokeSolve,
                                     check_options, check_smoke_solve, listed_options,
                                     passed_options, render_pick, run_smoke_solve,
                                     smoke_settings, write_smoke_picks)
from splatpipe.config import REPO_ROOT, Config, load_config
from splatpipe.errors import ExpectedFailure, PipelineError
from splatpipe.frames.files import GREY, IMAGES, MASK, pick_name
from splatpipe.jsonio import read_json

# The real [cloud.smoke], read by path, because tests/conftest.py points CONFIG_PATH at a test
# config. The scene's self-check renders the Smoke_Picks at its sizes.
REAL_CONFIG = tomllib.loads((REPO_ROOT / "config.toml").read_text(encoding="utf-8"))
REAL_SMOKE = REAL_CONFIG["cloud"]["smoke"]
REAL_SIZES = SmokeSizes(frames=REAL_SMOKE["cameras_frames"], width=REAL_SMOKE["cameras_width"],
                        height=REAL_SMOKE["cameras_height"],
                        focal_pixels=REAL_SMOKE["cameras_focal_pixels"],
                        placed_share=REAL_SMOKE["cameras_placed_share"],
                        focal_tolerance=REAL_SMOKE["cameras_focal_tolerance"],
                        timeout_seconds=REAL_SMOKE["cameras_timeout_seconds"],
                        help_timeout_seconds=REAL_SMOKE["check_timeout_seconds"])
# The smallest standard deviation of a Pick's values, from 0 to 255, for the self-check to call
# it not uniform. The real Smoke_Picks' own are about 52 or more.
MIN_SPREAD = 20.0

# The no-COLMAP tests' sizes, as the module docstring says.
SIZES = SmokeSizes(frames=30, width=64, height=48, focal_pixels=52.0, placed_share=0.5,
                   focal_tolerance=0.0625, timeout_seconds=600.0, help_timeout_seconds=60.0)
NEEDED = 15  # the Placed_Frames SIZES needs: 0.5 of the 30 Picks
LIMIT = 3.25  # how far f, fx and fy may lie from 52 px: 0.0625 of it
# The most a painted corner's mean value may be in a Colour_Pick or a Grey_Copy, from 0 to 255,
# after the JPEG round trip. The same corner of an unpainted render is about 78 or more.
PAINTED_MEAN = 16.0

# The seven COLMAP commands the step runs, in its order, written out here.
COMMANDS = ("feature_extractor", "sequential_matcher", "view_graph_calibrator", "global_mapper",
            "mapper", "image_undistorter", "image_undistorter_standalone")
# The option that the builders pass feature_extractor for RADIAL_FISHEYE only, which its block in
# options.md leaves out, and the one they pass with mask_used only, which the block has.
CAMERA_PARAMS = "--ImageReader.camera_params"
CAMERA_MASK = "--ImageReader.camera_mask_path"

# glog's lines as colmap <command> -h writes them before Boost's list (F16), and an error line.
GLOG_LINE = "I20261123 04:19:19.624758 0x1705ab000 colmap.cc:86] COLMAP 4.2.1 (without CUDA)"
HELP_ERROR = "Check failed: options_->image_reader->Check()"
ERROR_LINE = f"F20261123 04:19:19.625113 0x1705ab000 option_manager.cc:1022] {HELP_ERROR}"
# The Feature_Extraction's last line in the Cameras_Step failure test.
EXTRACTOR_LINE = "Failed to read the image file format."


def block_options(colmap_options: Mapping[str, Sequence[Any]]) -> dict[str, frozenset[str]]:
    """Every option that the builders pass each command, from options.md: each command's block,
    and --ImageReader.camera_params too for feature_extractor."""
    options = {command: {option.option for option in block}
               for command, block in colmap_options.items()}
    options["feature_extractor"].add(CAMERA_PARAMS)
    return {command: frozenset(names) for command, names in options.items()}


def help_text(options: Iterable[str]) -> str:
    """A colmap <command> -h text as F16 shows it: a glog line, then Boost's list, with -h
    [ --help ] and each of `options` two spaces in, each with a description on the next line,
    further in."""
    lines = [GLOG_LINE, "Options:", "  -h [ --help ]"]
    for option in sorted(options):
        lines += [f"  {option} arg", f"{' ' * 40}What {option} does."]
    return "\n".join(lines) + "\n"


@dataclass
class ScriptedHelp:
    """A stand-in for run_colmap in check_options: each run of `colmap <command> -h` writes its
    command's scripted text to its Command_Log, then gives its scripted exit code and the log's
    Last_Error_Line, read as run_colmap reads it. `ran` lists the commands in order."""

    texts: Mapping[str, tuple[int, str]]  # by command: the exit code and the -h text
    ran: list[str] = field(default_factory=list)

    def __call__(self, run: ColmapRun) -> RunResult:
        assert run.args == ("colmap", run.command, "-h"), run.args
        self.ran.append(run.command)
        code, text = self.texts[run.command]
        run.log.write_text(text, encoding="utf-8")
        with run.log.open(encoding="utf-8") as lines:
            last_line = last_error_line(lines)
        return RunResult(exit_code=code, seconds=0.0, last_line=last_line)


def read_picture(path: Path) -> np.ndarray:
    """The picture in the file `path`, with its channels and bit depth as they are."""
    picture = cv2.imdecode(np.frombuffer(path.read_bytes(), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    assert picture is not None, f"cv2.imdecode couldn't read {path}"
    return picture


def outside(root: Path, scratch: Path) -> dict[str, bytes | None]:
    """Every file and folder in `root` but `scratch` and what's in it, by its path from `root`:
    a file's bytes, and None for a folder."""
    return {path.relative_to(root).as_posix(): path.read_bytes() if path.is_file() else None
            for path in sorted(root.rglob("*"))
            if path != scratch and scratch not in path.parents}


def solved(**changes: Any) -> SmokeSolve:
    """A SmokeSolve that check_smoke_solve passes for SIZES, with each field in `changes` set:
    exactly NEEDED placed, and f and fy exactly LIMIT above 52 px and fx exactly LIMIT below."""
    solve = SmokeSolve(folder=Path("job"), picks=SIZES.frames, placed=NEEDED, mapper="global",
                       focal=52.0 + LIMIT, fx=52.0 - LIMIT, fy=52.0 + LIMIT)
    return replace(solve, **changes)


# The Smoke_Picks

def test_every_smoke_pick_fills_the_frame_sees_all_three_rectangles_and_isnt_uniform() -> None:
    # The scene's self-check. Every pixel's ray hits a rectangle, every Pick sees the floor and
    # both walls, so its view isn't planar, and its values spread widely.
    for number in range(REAL_SIZES.frames):
        pick, seen = render_pick(number, REAL_SIZES)
        assert pick.shape == (REAL_SIZES.height, REAL_SIZES.width, 3) and pick.dtype == np.uint8
        assert seen.shape == (REAL_SIZES.height, REAL_SIZES.width)
        assert (seen >= 0).all(), f"some rays of Pick {number} miss every rectangle"
        missed = [rectangle.name for index, rectangle in enumerate(RECTANGLES)
                  if not (seen == index).any()]
        assert not missed, f"Pick {number} doesn't see the {', '.join(missed)}"
        assert pick.std() > MIN_SPREAD, f"Pick {number} is nearly uniform"


def test_write_smoke_picks_writes_frames_that_read_frames_accepts_with_the_corner_and_mask(
        tmp_path: Path) -> None:
    folder = tmp_path / "frames"
    write_smoke_picks(folder, SIZES, FRAMES["jpeg_quality"])
    frames = read_frames(folder, FIXED_ID)
    assert (frames.pick_width, frames.pick_height) == (SIZES.width, SIZES.height)
    assert frames.mask_used is True
    assert frames.picks == tuple(Pick(number=number, file=pick_name(number))
                                 for number in range(SIZES.frames))
    # The black corner, the top left tenth of each side, in every Colour_Pick and Grey_Copy.
    corner = (slice(SIZES.height // PAINTED_PART), slice(SIZES.width // PAINTED_PART))
    for pick in frames.picks:
        colour = read_picture(folder / IMAGES / pick.file)
        grey = read_picture(folder / GREY / pick.file)
        assert colour.shape == (SIZES.height, SIZES.width, 3), pick.file
        assert grey.shape == (SIZES.height, SIZES.width), pick.file
        assert colour[corner].mean() <= PAINTED_MEAN, pick.file
        assert grey[corner].mean() <= PAINTED_MEAN, pick.file
    # mask.png: a grey PNG at the Pick_Size, 0 over the corner and 255 elsewhere.
    expected = np.full((SIZES.height, SIZES.width), 255, dtype=np.uint8)
    expected[corner] = 0
    mask = read_picture(folder / MASK)
    assert mask.dtype == np.uint8 and np.array_equal(mask, expected)


# The Option_Check (Req 18.3, 18.6)

def test_passed_options_holds_every_option_the_builders_pass_each_command(
        colmap_options: Mapping[str, Sequence[Any]]) -> None:
    passed = passed_options(load_config(), SIZES)
    assert tuple(passed) == COMMANDS
    assert passed == block_options(colmap_options)
    assert {CAMERA_PARAMS, CAMERA_MASK} <= passed["feature_extractor"]


def test_listed_options_reads_boosts_list_but_not_a_wrapped_description_or_glogs_lines() -> None:
    text = (f"{GLOG_LINE}\n"
            "I20261123 04:19:19.624801 0x1705ab000 option_manager.cc:900]   --log_only arg\n"
            "Options:\n"
            "  -h [ --help ]\n"
            "  --database_path arg\n"
            "  --SiftExtraction.peak_threshold arg (=0.0066666666666666671)\n"
            "                                        Peak threshold for detection, which wraps\n"
            "                                        onto a line of its own --wrapped_word\n"
            f"{ERROR_LINE}\n"
            "  --ImageReader.camera_mask_path arg\n")
    assert listed_options(text) == {"--help", "--database_path",
                                    "--SiftExtraction.peak_threshold", CAMERA_MASK}


def test_check_options_fails_a_help_that_exits_with_1_naming_the_command_and_its_last_line(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        colmap_options: Mapping[str, Sequence[Any]]) -> None:
    texts = {command: (0, help_text(options))
             for command, options in block_options(colmap_options).items()}
    texts["mapper"] = (1, f"{GLOG_LINE}\n{ERROR_LINE}\n")
    scripted = ScriptedHelp(texts)
    monkeypatch.setattr(smoke, "run_colmap", scripted)
    with pytest.raises(PipelineError) as raised:
        check_options(load_config(), SIZES, tmp_path / "help")
    message = str(raised.value)
    assert "`colmap mapper -h`" in message
    assert f"{exit_text(1)} ({HELP_ERROR})" in message
    assert scripted.ran == list(COMMANDS[:COMMANDS.index("mapper") + 1])


def test_check_options_fails_a_help_that_lacks_an_option_the_step_passes_naming_both(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        colmap_options: Mapping[str, Sequence[Any]]) -> None:
    options = block_options(colmap_options)
    texts = {command: (0, help_text(names)) for command, names in options.items()}
    texts["feature_extractor"] = (0, help_text(options["feature_extractor"] - {CAMERA_PARAMS}))
    scripted = ScriptedHelp(texts)
    monkeypatch.setattr(smoke, "run_colmap", scripted)
    with pytest.raises(PipelineError) as raised:
        check_options(load_config(), SIZES, tmp_path / "help")
    message = str(raised.value)
    assert "`colmap feature_extractor -h`" in message
    assert f"the option {CAMERA_PARAMS}" in message
    assert scripted.ran == ["feature_extractor"]


# The Smoke_Solve (Req 18.4, 18.5, 18.7, 18.8)

def test_smoke_settings_give_the_default_presets_cameras_with_every_time_limit_at_the_smoke_one(
        temp_config: Path) -> None:
    # The test config's [jobs] with another default_preset, so the Preset can't be a default.
    data = tomllib.loads(temp_config.read_text(encoding="utf-8"))
    data["jobs"]["default_preset"] = "quality"
    config = Config(data)
    preset, settings = smoke_settings(config, SIZES)
    assert preset == "quality"
    assert (settings.section, settings.preset) == ("cameras", "quality")
    time_limits = {key: SIZES.timeout_seconds for key in CAMERAS
                   if key.endswith("_timeout_seconds")}
    assert len(time_limits) == len(COMMANDS)
    assert dict(settings) == {**config.step_settings("cameras", "quality"), **time_limits}
    cfg = read_settings(settings, "cameras", "quality")
    assert dict(cfg.timeouts) == dict.fromkeys(COMMANDS, SIZES.timeout_seconds)
    assert cfg.near_overlap == NEAR_OVERLAP["quality"]


def test_check_smoke_solve_passes_exactly_the_placed_share_and_exactly_the_tolerance() -> None:
    check_smoke_solve(solved(), SIZES)


def test_check_smoke_solve_refuses_one_placed_pick_fewer_naming_cameras_placed_share() -> None:
    with pytest.raises(PipelineError) as raised:
        check_smoke_solve(solved(placed=NEEDED - 1), SIZES)
    message = str(raised.value)
    assert f"placed only {NEEDED - 1} of the {SIZES.frames} made-up picks" in message
    assert f"at least {NEEDED}, cameras_placed_share = 0.5 of them in [cloud.smoke]" in message


@pytest.mark.parametrize(("name", "value", "said"), [
    pytest.param("focal", math.nextafter(52.0 + LIMIT, math.inf), "the solved focal length f",
                 id="f"),
    pytest.param("fx", math.nextafter(52.0 - LIMIT, -math.inf), "the undistorted camera's fx",
                 id="fx"),
    pytest.param("fy", math.nextafter(52.0 + LIMIT, math.inf), "the undistorted camera's fy",
                 id="fy"),
])
def test_check_smoke_solve_refuses_a_focal_length_just_past_the_tolerance_naming_the_keys(
        name: str, value: float, said: str) -> None:
    with pytest.raises(PipelineError) as raised:
        check_smoke_solve(solved(**{name: value}), SIZES)
    message = str(raised.value)
    assert f"{said} = {value:.2f} px, but it must be within {LIMIT:.2f} px" in message
    assert "cameras_focal_pixels = 52 px in [cloud.smoke]" in message
    assert "cameras_focal_tolerance = 0.0625" in message


def test_run_smoke_solve_gives_the_numbers_of_the_files_it_read_and_writes_only_in_scratch(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_colmap: Any) -> None:
    monkeypatch.chdir(tmp_path)  # so a write to a relative path would land where the test looks
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    before = outside(tmp_path, scratch)
    fake = fake_colmap(global_models=(SIZES.frames,))
    solve = run_smoke_solve(load_config(), SIZES, scratch, FIXED_ID)
    assert outside(tmp_path, scratch) == before
    # The numbers, from the files the step wrote: f from the Final_Model's RADIAL camera, which
    # solve.json names, and fx and fy from the Levelled_Model's PINHOLE camera in sparse/.
    folder = (scratch / JOB).resolve()
    work = folder / "work" / "cameras"
    record = read_json(work / "solve.json")
    assert record["mapper"] == "global"
    (final,) = read_model(work / "colmap" / "global" / str(record["final_model"])).cameras
    (pinhole,) = read_model(work / "sparse").cameras
    assert CAMERA_MODELS[final.model_id][0] == "RADIAL"
    assert CAMERA_MODELS[pinhole.model_id][0] == "PINHOLE"
    assert solve == SmokeSolve(folder=folder, picks=SIZES.frames, placed=SIZES.frames,
                               mapper="global", focal=final.params[0], fx=pinhole.params[0],
                               fy=pinhole.params[1])
    assert solve.fx == solve.fy
    # Every COLMAP run had the Smoke_Solve's Time_Limit.
    assert {run.timeout_seconds for run in fake.runs} == {SIZES.timeout_seconds}


def test_run_smoke_solve_passes_a_cameras_step_failure_up_with_the_steps_own_message(
        tmp_path: Path, fake_colmap: Any) -> None:
    fake_colmap(global_models=(SIZES.frames,), exits={"feature_extractor": (1, EXTRACTOR_LINE)})
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    with pytest.raises(ExpectedFailure) as raised:
        run_smoke_solve(load_config(), SIZES, scratch, FIXED_ID)
    assert type(raised.value) is ExpectedFailure
    message = str(raised.value)
    assert message.startswith(
        f"COLMAP's feature_extractor stopped with {exit_text(1)} ({EXTRACTOR_LINE}).")
    log = (scratch / JOB).resolve() / "work" / "cameras" / "colmap" / "logs"
    assert f"Its output is in {log / 'feature_extractor.log'}." in message
