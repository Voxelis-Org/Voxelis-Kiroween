# Implementation Plan: frames

## Overview

This plan builds the Frames_Step in `splatpipe/frames/` in 10 tasks: dependencies and settings, the frame scores, the Still_Trim and the Jolt_Filter, the Picker, the Sample_Buffer, the Mask and the Work_Folder's files, ffprobe and the ffmpeg decode, the step itself, the HDR Tone_Map, the step's section of the Pipeline_Doc, and last the Step_List entry. Each task writes its code with that code's unit tests and Hypothesis property tests, runs the whole suite and commits. The code is Python 3.11. Task 1 adds numpy and `opencv-python-headless` with `uv add`, and no scipy, which the clean-export spec decides. pytest and Hypothesis are already in the dev group.

The plan follows the design's build order. Four choices in that order matter:

- **The docs before the Step_List.** Once `STEP_LIST` holds `frames`, the existing check in `tests/test_docs.py` (`test_the_pipeline_doc_has_a_step_section_for_the_fake_step_and_every_listed_step`) needs a `## Step: frames` section. So task 9 writes that section, and task 10 changes the Step_List together with the "The step list" paragraph that describes it, as D1 asks.
- **The ffmpeg row's tone map words** go in task 1, with the dependencies, because Req 12.6 says they come in the same task that adds the dependencies.
- **The frames import check** (Req 13.10) and the dependency check (Req 9.8) go in task 1, so the test run of every later task checks the new package's imports.
- **`tests/test_frames_files.py`** holds the tests of `files.py`, in task 5.

Some files grow over several tasks, so each piece comes with the first task that uses it:

- `splatpipe/frames/__init__.py`: task 1 makes it with only the package docstring, task 7 writes the step, and task 8 adds the HDR choice.
- `splatpipe/frames/choose.py`: task 3 writes the Still_Trim and the Jolt_Filter, and task 4 adds Target_Picks and the Picker.
- `splatpipe/frames/ffmpeg.py`: task 6 writes it without the Tone_Map, and task 8 adds `TONE_MAP`, `ToneMap`, `tone_map_filter()` and the filter check.
- `tests/conftest.py`: task 1 adds `[frames]` and `set_frames`, task 7 adds `FakeTools`, and task 8 gives `FakeTools` its HDR options.
- `tests/test_frames_ffmpeg.py`: task 6 writes it, and task 8 adds the Tone_Map tests.
- `tests/test_frames_step.py`: task 7 writes it, task 8 adds the HDR cases, and task 10 adds the end-to-end test.
- `tests/test_imports.py`: task 1 adds the frames import check and the dependency check, and task 10 changes the Step_List check.
- `tests/test_docs.py`: task 1 tests the licence rows, task 8 tests a zimg row only with Samarth's OK, task 9 tests the frames section, and task 10 tests "The step list".
- `docs/third-party-licences.md`: task 1 changes the rows, and task 8 adds a zimg row only with Samarth's OK.
- `docs/pipeline.md`: task 9 adds `## Step: frames`, and task 10 rewrites "The step list".

## Working rules for every task

- Work on the existing `feat/frame-picking` branch, never on `main`.
- The only step folder this spec makes is `splatpipe/frames/`. Of the Skeleton, only the Step_List in `splatpipe/run.py` changes, in task 10. `splatpipe/probe.py`, `make.py`, `step.py` and the Fake_Step stay as they are.
- `splatpipe/frames/` imports only the standard library, numpy, cv2 and `splatpipe`. The pure modules (`scores.py`, `choose.py` and `mask.py`) import no `subprocess`, no `pathlib` and no `splatpipe` module. Nothing imports `cloud/`, boto3, torch or gsplat.
- Add no package but numpy and `opencv-python-headless`, in task 1. No scipy, never plain `opencv-python`, and no GPU package in `pyproject.toml`.
- Every frame comes from the ffmpeg command line. cv2 only does maths on frames ffmpeg has decoded and encodes JPEG and PNG images. It never reads or writes video: no `cv2.VideoCapture` and no `cv2.VideoWriter`.
- Every number and setting comes from `[frames]` in `config.toml`, except the design's Fixed_Constants: `HDR_TRANSFERS`, `TONE_MAP`, the Tone_Map's filters with the `hable` curve and BT.709 (their `npl` and `desat` numbers come from `tonemap_npl` and `tonemap_desat`), the Laplacian's kernel and border, the 0 to 255 brightness scale, the Mask's 0 and 255, and the Work_Folder's names (`IMAGES`, `GREY`, `RECORD`, `MASK` and the 6-digit Pick names).
- Tests run on the CPU with no network. The maths tests build tiny numpy frames themselves. Every Job is made in the temporary Jobs_Folder from `tests/conftest.py`, never in the real `data/` folder, and no video or frame goes into git.
- Tests that need ffmpeg use `pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None, reason="ffmpeg and ffprobe aren't installed")`. They make their clips in `tmp_path`. Every subprocess a test runs gets `timeout=60`. Every other test passes on a machine without ffmpeg.
- Property tests use Hypothesis with at least 100 examples (the default), and each docstring says `Feature: frames, Property N: <title>`. Frames are at most 12 × 16 pixels and lists at most 60 frames, except Property 10's streams of up to 2,000 frames of 1 × 1. Times are running sums of gaps from 0.001 s to 2 s, so they strictly increase. Each oracle is written in the test with its own code and never calls the function it checks.
- Every test sub-task is required. None is optional.
- Nothing here costs money. No task touches AWS, sets `APPROVED_SPEND` or runs anything paid.
- The last sub-task of each task is its checkpoint. Run `uv run pytest` and fix what fails until every test passes, `tests/test_guard.py` included. Then stage the task's files by name with `git add <file> ...`, never with `git add -A`, `git add .` or a folder, and commit only those files with `git commit -m "<message>" -- <file> ...`, so nothing else that's staged goes in. Leave `docs/devlog.md` out: the devlog hook adds and commits its own line after each task.
- If a hook blocks something it shouldn't, say so and fix the hook. Never work around it.

## Tasks

- [x] 1. Dependencies, settings and the test config
  - Files: `splatpipe/frames/__init__.py`, `splatpipe/frames/settings.py` and `tests/test_frames_settings.py` (new). `pyproject.toml`, `uv.lock`, `config.toml`, `docs/third-party-licences.md`, `tests/conftest.py`, `tests/test_imports.py` and `tests/test_docs.py` (changed).
  - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7, 9.8, 9.9, 9.10, 9.11, 12.6, 13.1, 13.3, 13.10, 13.11_

  - [x] 1.1 Add numpy and `opencv-python-headless`, with their licence lines
    - Run `uv add numpy opencv-python-headless`. Add no other package, so no scipy. Hypothesis stays in the dev group.
    - Check what `uv.lock` gained. If it brought in any package besides these two, that package gets its own line in the Licence_List in this commit. If its licence isn't on the list in `.kiro/steering/rules.md`, stop and ask Samarth first.
    - In `docs/third-party-licences.md`, the Installed cell of numpy and `opencv-python-headless` becomes "Yes", and scipy's row stays "Not yet". OpenCV's "How we use it" names the Laplacian, histogram equalisation, and JPEG and PNG encoding of frames that ffmpeg has already decoded, and says it never reads or writes video.
    - The "ffmpeg and ffprobe" row's "How we use it" adds that ffmpeg decodes the frames of the Used_Part and tone maps HDR video with `zscale` and `tonemap` when the ffmpeg build has both filters. Keep the words that the existing row test checks.
    - _Requirements: 9.8, 12.6_

  - [x] 1.2 Add `[frames]` and its Preset tables to `config.toml`
    - Append `[frames]`, `[frames.fast]`, `[frames.default]` and `[frames.quality]` after `[fake.quality]`, as the design's "The `[frames]` section of `config.toml`" gives them, with a comment above every key.
    - That's the 22 Shared_Settings with the values of Req 9.2, `tonemap_npl = 100` and `tonemap_desat = 0.0` among them, and `base_picks` 120, 150 and 200. Leave every other section as it is.
    - _Requirements: 9.1, 9.2, 9.3, 9.10_

  - [x] 1.3 Write `splatpipe/frames/settings.py`
    - Make `splatpipe/frames/` with an `__init__.py` that holds only the package docstring for now. Task 7 writes the step in it.
    - `FramesSettings` and `read_settings(settings, section, preset)`, as in the design's "settings.py" section. It reads each key with `settings[key]`, so a missing key raises the Step_Runner's own message.
    - One check for each kind of value, with `type(v)` checked exactly, so a bool is never a number. Each refusal is an `ExpectedFailure` naming the key, its table (`[frames]`, or `[frames.<preset>]` for `base_picks`) and what the value must be.
    - `tonemap_npl` must be a number above 0 and `tonemap_desat` a number of 0 or more, and neither may be a bool. `FramesSettings` keeps both as floats. They're checked for every video, before ffprobe runs, though only an HDR_Clip uses them.
    - Then the cross checks: `mask_min_coverage > mask_max_coverage`, `min_picks > base_picks` and `base_picks > max_picks` are refused, naming both settings and their values.
    - `min_vcpus` and `min_memory_gb` are checked and kept, and nothing in the step uses them.
    - The module imports only the standard library and `splatpipe.errors`.
    - _Requirements: 9.4, 9.5, 9.6, 9.7, 9.9, 9.10_

  - [x] 1.4 Add `[frames]` and `set_frames` to `tests/conftest.py`
    - `config_text` gains a `[frames]` section with its three Preset tables: all 22 Shared_Settings at the real values, `tonemap_npl = 100` and `tonemap_desat = 0.0` included, except `min_picks = 1`, `base_picks` 2, 3 and 4, and both timeouts at 60, so tiny clips pass.
    - A `set_frames(**values)` fixture rewrites the test config with any of the `[frames]` keys changed, a shared one or `base_picks` for one Preset. Like `set_fail`, each call writes the whole file again from `config_text`.
    - Update the module docstring to name the new fixture.
    - _Requirements: 9.11_

  - [x] 1.5 Write unit tests for the settings in `tests/test_frames_settings.py`
    - The real config, read by path as `REPO_ROOT / "config.toml"`, has `[frames]` with exactly the 22 Shared_Settings and the values of Req 9.2, `tonemap_npl = 100` and `tonemap_desat = 0.0` among them, and `base_picks` 120, 150 and 200. `step_settings("frames", preset)` loads it for each of the three Presets. The comment check in `tests/test_config.py` already covers the new keys.
    - Parametrised over every key: a wrong value, a bool where a number belongs included, gives an `ExpectedFailure` naming the key, its table and what it must be. The edge values pass: `jpeg_quality` 1 and 100, `mask_samples` 2, and the ratios and coverages at 0 and 1.
    - The Tone_Map's numbers: `tonemap_npl` of 0 or `true` and `tonemap_desat` of -0.5 or `true` are refused, each naming the key, `[frames]` and what it must be, and `tonemap_desat = 0` passes.
    - Each cross check names both settings and their values. A missing key, read through `StepSettings`, gives the Step_Runner's message with `[frames]`, `[frames.<preset>]` and the key.
    - `read_settings` on the test config gives a `FramesSettings` with the chosen Preset's `base_picks`, and `tonemap_npl` and `tonemap_desat` as the floats 100.0 and 0.0. `set_frames` changes one key for one test, and the next test sees the test config's value again.
    - _Requirements: 9.1, 9.2, 9.3, 9.5, 9.6, 9.7, 9.11, 13.1_

  - [x] 1.6 Write the frames import check, the dependency check and the licence row tests
    - In `tests/test_imports.py`, a new test reads every `splatpipe/frames/*.py` with the file's existing `ast` reader and allows only names in `sys.stdlib_module_names`, numpy, cv2 and `splatpipe`. A failure gives the file and the line.
    - In `tests/test_imports.py`, a dependency test reads `pyproject.toml` and `uv.lock` with `tomllib`. numpy and `opencv-python-headless` are runtime dependencies, and no package in either file is plain `opencv-python`, torch, gsplat or an `nvidia-*` or `cuda-*` package. The test neither requires nor refuses scipy, so the clean-export spec can decide scipy without changing it.
    - The module docstring of `tests/test_imports.py` describes both new tests, and its line saying a step folder like `splatpipe/frames/` can import numpy, scipy and OpenCV no longer names scipy.
    - In `tests/test_docs.py`, the numpy and `opencv-python-headless` rows say "Yes". OpenCV's "How we use it" says it never reads or writes video. The ffmpeg row names decoding the Used_Part's frames, `zscale` and `tonemap`.
    - _Requirements: 9.8, 9.9, 12.6, 13.10_

  - [x] 1.7 Run the tests and commit
    - Run `uv run pytest`. Every test passes, `tests/test_guard.py` included.
    - On `feat/frame-picking`: `git add pyproject.toml uv.lock config.toml docs/third-party-licences.md splatpipe/frames/__init__.py splatpipe/frames/settings.py tests/conftest.py tests/test_frames_settings.py tests/test_imports.py tests/test_docs.py`, then commit those files with the message "Frames: dependencies, settings and the test config".
    - _Requirements: 13.3, 13.11_

- [x] 2. Frame scores and frame sizes
  - Files: `splatpipe/frames/scores.py` and `tests/test_frames_scores.py`, both new.
  - _Requirements: 1.12, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 8.1, 13.1, 13.2, 13.3, 13.11_

  - [x] 2.1 Write `splatpipe/frames/scores.py`
    - `sharpness()`, `motion()`, `contrast()`, `Scores`, `Scorer`, `score_shape()` and `pick_size()`, as in the design's "scores.py" section. The module is pure and imports only numpy, cv2 and the standard library.
    - `sharpness` is `float(cv2.Laplacian(frame, cv2.CV_64F, ksize=1).var())`, with OpenCV's default border.
    - `motion` is 0 with no previous frame. Otherwise it's the exact integer sum of `abs(frame − previous)`, worked out in `int16` and summed as `int64`, divided by the pixel count.
    - `contrast` works from the 256-bin histogram with Python ints, as the design gives it, so a frame that's half 0 and half 255 gives exactly 127.5.
    - `Scorer.add` scores the frame against the one before and then keeps a reference to this frame alone. It appends the four numbers to lists, and `result()` gives them as float64 arrays.
    - `score_shape` and `pick_size` round half up in integers. `pick_size` keeps the video's own size when its long edge is at most `long_edge`, so a Pick is never larger than the video.
    - _Requirements: 1.12, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 8.1_

  - [x] 2.2 Write unit tests for the scores and the sizes in `tests/test_frames_scores.py`
    - `sharpness` equals an oracle Laplacian written in numpy with reflect-101 borders on small frames, and is 0 on a flat frame.
    - `motion(None, frame)` is 0, and a `Scorer`'s first frame gets a Motion of 0.
    - `Scorer.result` keeps the time, Sharpness, Motion and Contrast of each frame, in the order they came.
    - `score_shape(1920, 1080, 320)` gives 320 × 180, and a video narrower than `score_width` is scaled up for scoring.
    - `pick_size` gives 1600 × 900 for 3840 × 2160 and 900 × 1600 for 2160 × 3840 with a `long_edge` of 1600, and keeps 64 × 48 as it is.
    - _Requirements: 3.1, 3.2, 3.4, 3.6, 8.1, 13.1_

  - [x] 2.3 Write the property test for Property 1 in `tests/test_frames_scores.py`, with Hypothesis
    - **Property 1: Scores stay in range**
    - **Validates: Requirements 3.3, 3.4, 3.5, 3.7**
    - Draw pairs of `uint8` frames of one shape up to 12 × 16 with `hypothesis.extra.numpy.arrays`, plus frames that are all 0, all 255, and half 0 and half 255.
    - The oracle works out the mean absolute change with Python ints, checks Sharpness is 0 or more, and checks Contrast is from 0 to 127.5 and within 1e-9 of `numpy.std(frame)`.
    - _Requirements: 3.3, 3.4, 3.5, 3.7, 13.2_

  - [x] 2.4 Write the property test for Property 2 in `tests/test_frames_scores.py`, with Hypothesis
    - **Property 2: Frame sizes keep the shape and never grow a Pick**
    - **Validates: Requirements 3.1, 8.1**
    - Draw Displayed_Shapes from 1 × 1 to 8000 × 8000, and `score_width` and `long_edge` from 1 to 4000.
    - The oracle checks with `fractions.Fraction`: the score width is exact, each worked-out edge is at least 1 and within 0.5 of its proportional value, and the Pick's long edge is `min(long_edge, the video's long edge)`.
    - _Requirements: 3.1, 8.1, 13.2_

  - [x] 2.5 Run the tests and commit
    - Run `uv run pytest`. Every test passes, `tests/test_guard.py` included.
    - On `feat/frame-picking`: `git add splatpipe/frames/scores.py tests/test_frames_scores.py`, then commit those files with the message "Frames: frame scores and sizes".
    - _Requirements: 13.3, 13.11_

- [x] 3. Still trim and jolt filter
  - Files: `splatpipe/frames/choose.py` and `tests/test_frames_choose.py`, both new.
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 4.10, 5.1, 5.5, 13.1, 13.2, 13.3, 13.11_

  - [x] 3.1 Write the first half of `splatpipe/frames/choose.py`
    - `window_bounds()`, `Kept`, `still_trim()` and `jolt_keep()`, as in the design's "choose.py" section. The module is pure and imports only numpy and the standard library.
    - A Window holds every frame whose time is at most half its length before or after the frame's own, the frame itself included. `window_bounds` finds them with `searchsorted`, `"left"` for the low edge and `"right"` for the high one.
    - `still_trim` finds the Still_Frames over the whole Used_Part. It counts each run of frames that aren't still and lasts less than `blip_seconds` as still, and the other runs are the Moving_Stretches. A run's length goes to the next frame's time, or to the Used_Part's end.
    - It cuts the Still_Start and the Still_End only when each lasts `min_still_seconds` or more, and keeps every frame when there's no Moving_Stretch. `Kept` gives the Kept_Range's first and last Frame_Numbers, its start and end in seconds, and the seconds cut at each end, 0.0 where nothing was cut.
    - `jolt_keep` keeps a frame when its Sharpness is at least `jolt_ratio` times the median Sharpness of its Window, with the Windows taken over the Kept_Range frames it's given.
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 4.10, 5.1_

  - [x] 3.2 Write unit tests for the Still_Trim and the Jolt_Filter in `tests/test_frames_choose.py`
    - With frames 0.1 s apart: a still start of 3 s is cut and one of 1.5 s isn't, and the same at the end.
    - A 0.3 s Blip inside a 3 s still start doesn't end it, so the whole start is cut. A still middle between two Moving_Stretches is kept. A clip that's still from start to end is kept whole.
    - Each case checks `first`, `last`, `cut_start_seconds`, `cut_end_seconds`, `start_seconds` and `end_seconds`.
    - A Window holds the frame itself and the frames exactly half its length away, and no frame further.
    - The Jolt_Filter drops one blurred frame among sharp ones, keeps every frame when every Sharpness is 0, and uses only the Kept_Range frames it's given for the medians.
    - _Requirements: 4.1, 4.2, 4.4, 4.5, 4.6, 4.7, 4.8, 4.10, 5.1, 13.1_

  - [x] 3.3 Write the property test for Property 5 in `tests/test_frames_choose.py`, with Hypothesis
    - **Property 5: The Still_Trim keeps one Stretch that holds all the movement**
    - **Validates: Requirements 4.4, 4.5, 4.7, 4.8, 4.9**
    - Draw 1 to 60 frames with increasing times, a Used_Part end after the last time, and positive `still_motion`, `motion_window_seconds`, `min_still_seconds` and `blip_seconds`.
    - Draw the Motions mostly from `[0, still_motion, 2 × still_motion, 255]`, mixed with floats from 0 to 255, so stills, Blips and Moving_Stretches all show up.
    - The oracle finds the Moving_Stretches from the definitions with its own loops. It checks `0 <= first <= last <= n − 1`, that every frame is kept and nothing cut when there's no Moving_Stretch, that every Moving_Stretch frame is inside the Kept_Range, and that each cut end holds no Moving_Stretch frame and lasts at least `min_still_seconds`.
    - _Requirements: 4.4, 4.5, 4.7, 4.8, 4.9, 13.2_

  - [x] 3.4 Write the property test for Property 6 in `tests/test_frames_choose.py`, with Hypothesis
    - **Property 6: The Jolt_Filter keeps exactly the frames that meet its rule**
    - **Validates: Requirements 5.1, 5.5**
    - Draw Kept_Ranges with increasing times, Sharpness values of 0 or more, a `jolt_ratio` from 0 to 1 and a positive `jolt_window_seconds`.
    - The oracle finds each frame's Window with a plain loop, not `searchsorted`, and checks that each kept frame meets the rule and each dropped frame breaks it.
    - _Requirements: 5.1, 5.5, 13.2_

  - [x] 3.5 Run the tests and commit
    - Run `uv run pytest`. Every test passes, `tests/test_guard.py` included.
    - On `feat/frame-picking`: `git add splatpipe/frames/choose.py tests/test_frames_choose.py`, then commit those files with the message "Frames: still trim and jolt filter".
    - _Requirements: 13.3, 13.11_

- [x] 4. Target picks and the picker
  - Files: `splatpipe/frames/choose.py` and `tests/test_frames_choose.py`, both changed.
  - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8, 13.1, 13.2, 13.3, 13.11_

  - [x] 4.1 Write the second half of `splatpipe/frames/choose.py`
    - `target_picks()`, `motion_parts()` and `pick()`, as in the design's "choose.py" section.
    - `target_picks` is `min(max_picks, max(base_picks, math.ceil(kept_seconds / seconds_per_pick)))`.
    - `motion_parts` gives each Kept_Range frame its Motion_Position, the running total of Motion from the Kept_Range's first frame with the first frame's own Motion not counted. Its part is `min(n − 1, floor(pos × n / total))`. When the total is 0, the same formula runs on the time since the Kept_Range's first frame, against the time to the Kept_Range's end.
    - `pick` walks the Candidates in Frame_Number order and keeps, for each part, the first Candidate with the highest Sharpness, replacing it only on a strictly higher one. A part with no Candidate gives no Pick. It returns the chosen Frame_Numbers in order.
    - _Requirements: 6.1, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8_

  - [x] 4.2 Write unit tests for Target_Picks and the Picker in `tests/test_frames_choose.py`
    - `target_picks` gives 120 for 240 s and 210 for 420 s with `fast`'s `base_picks` of 120, `seconds_per_pick` 2 and `max_picks` 400. It gives `base_picks` for a short clip and stops at `max_picks` for a long one.
    - With a total Motion of 0, the parts split the Kept_Range by time.
    - A part with no Candidate gives no Pick, so there are fewer Picks than parts.
    - Two Candidates in one part with the same Sharpness give the earlier one. The Picks come in Frame_Time order.
    - _Requirements: 6.1, 6.2, 6.4, 6.5, 6.6, 6.7, 13.1_

  - [x] 4.3 Write the property test for Property 7 in `tests/test_frames_choose.py`, with Hypothesis
    - **Property 7: Motion_Parts are in range and in order**
    - **Validates: Requirements 6.3, 6.4**
    - Draw Kept_Ranges with increasing times, an end after the last time, Motions from 0 to 255 with all-0 lists included, and n from 1 to 400.
    - The oracle checks that the parts are whole numbers from 0 to n − 1 that never go down, that the first frame is in part 0, and, when the total is above 0, that the last frame is in part n − 1. Each frame's Motion_Position, or its time since the first frame when the total is 0, lies within 1e-9 × total of its part's range.
    - _Requirements: 6.3, 6.4, 13.2_

  - [x] 4.4 Write the property test for Property 8 in `tests/test_frames_choose.py`, with Hypothesis
    - **Property 8: Each Motion_Part gives its sharpest, earliest Candidate**
    - **Validates: Requirements 6.5, 6.6**
    - Draw non-decreasing part lists, any set of Candidates among the frames, and Sharpness values from a small set, so ties are common.
    - The oracle gives, for each part that holds a Candidate, its Candidate with the highest Sharpness and the earliest of them on a tie, and nothing for a part without one.
    - _Requirements: 6.5, 6.6, 13.2_

  - [x] 4.5 Write the property test for Property 9 in `tests/test_frames_choose.py`, with Hypothesis
    - **Property 9: Picks are few, different, inside the Kept_Range and in order**
    - **Validates: Requirements 6.7, 6.8**
    - Draw Score_Frames with increasing times, a Kept_Range inside them, Candidates inside the Kept_Range, Motions, Sharpness values and n from 1 to 400. Run `motion_parts` and then `pick`.
    - Check that the Picks number at most n, are different Candidates inside the Kept_Range and the Used_Part, and have strictly increasing Frame_Times.
    - _Requirements: 6.7, 6.8, 13.2_

  - [x] 4.6 Run the tests and commit
    - Run `uv run pytest`. Every test passes, `tests/test_guard.py` included.
    - On `feat/frame-picking`: `git add splatpipe/frames/choose.py tests/test_frames_choose.py`, then commit those files with the message "Frames: target picks and the picker".
    - _Requirements: 13.3, 13.11_

- [x] 5. Sample buffer, mask and the Work_Folder's files
  - Files: `splatpipe/frames/mask.py`, `splatpipe/frames/files.py`, `tests/test_frames_mask.py` and `tests/test_frames_files.py`, all new.
  - _Requirements: 1.12, 1.13, 2.7, 3.6, 4.10, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 7.9, 7.10, 7.11, 8.3, 8.4, 8.5, 8.6, 8.10, 13.1, 13.2, 13.3, 13.7, 13.9, 13.11_

  - [x] 5.1 Write `splatpipe/frames/mask.py`
    - `SampleBuffer`, `spread()`, `pixel_change()`, `stuck_pixels()`, `MaskChoice`, `decide_mask()` and `mask_image()`, as in the design's "mask.py" section. The module is pure and imports only numpy and the standard library.
    - `SampleBuffer(samples)` holds `(Frame_Number, frame)` pairs in memory, with no temporary file. `add` gives each frame the next Frame_Number from 0, and keeps the pair when that number is a multiple of k, which starts at 1. When the buffer then holds more than 2 × `samples` pairs, it keeps every other pair from the first (`held[::2]`) and doubles k. `stride` gives k, and `numbers` the Frame_Numbers it holds.
    - The buffer keeps a reference to each frame it holds and never copies one, so it relies on `decode`, which task 6 writes, giving a new array for every frame.
    - `samples_in(first, last)` takes the c held pairs whose Frame_Number is from `first` to `last`, in order. It gives all of them when c is at most `samples`, and otherwise the `samples` pairs at the places `spread(c, samples)`.
    - `spread(count, samples)` gives every place from 0 to `count − 1` when `count` is at most `samples`. Otherwise it gives `samples` places spread with the design's round-half-up integer formula, from 0 to `count − 1`, both included.
    - `pixel_change` takes the Mask_Samples as a sequence, adds up each pixel's absolute change from one sample to the next as `int32`, and divides by the number of pairs. It gives `None` with fewer than 2 samples.
    - `stuck_pixels` is `change < ratio × median`, so a median of 0 marks no pixel. `decide_mask` uses the Mask only when the coverage is from `min_coverage` to `max_coverage`, both included, and never grows, shrinks or smooths the stuck pixels.
    - `mask_image` maps each output pixel to the Score pixel whose area holds the output pixel's centre, with integer index arrays. It gives a `uint8` image with 0 for stuck pixels and 255 for the rest, which suits COLMAP's camera mask and the training loss.
    - _Requirements: 1.12, 1.13, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 7.10_

  - [x] 5.2 Write `splatpipe/frames/files.py`
    - The Fixed_Constants `IMAGES`, `GREY`, `RECORD` and `MASK`, and `pick_name()`, `grey_copy()`, `write_jpeg()`, `write_png()`, `frames_record()` and `write_record()`, as in the design's "files.py" section.
    - `write_jpeg` and `write_png` encode with `cv2.imencode` and write with `Path.write_bytes`. An `OSError` or a failed encode raises an `ExpectedFailure` naming the file and the reason, like "No space left on device". So does `write_record`, which calls `splatpipe.jsonio.write_json`.
    - `frames_record` is pure. It gives the design's Frames_Record with only plain `int`, `float`, `bool`, `str`, `None`, `list` and `dict` values, turning numpy values into Python ones.
    - _Requirements: 2.7, 3.6, 4.10, 7.9, 8.3, 8.4, 8.5, 8.6, 8.10_

  - [x] 5.3 Write unit tests for the Sample_Buffer and the Mask in `tests/test_frames_mask.py`
    - With `mask_samples` = 2, 10 frames leave a `SampleBuffer` holding the Frame_Numbers 0, 4 and 8, with `stride` 4. A Kept_Range of 0 to 9 then gives the samples 0 and 8, one of 3 to 9 gives 4 and 8, and one of 5 to 7 gives none, so no Mask. `spread(5, 3)` gives 0, 2 and 4.
    - A 4 × 6 corner that never changes, on samples of moving noise, is found as the Mask, with its coverage.
    - A still scene, whose median change is 0, gives no Mask. A planted region over 25 % of the frame gives no Mask, and so does one under 0.2 %.
    - 0 or 1 samples give no Mask and a coverage of 0.
    - `pixel_change` equals a numpy oracle. The stuck pixels are exactly the planted ones, with no pixel added or removed around them.
    - `mask_image` scales a 2 × 2 stuck array to 4 × 4 as 2 × 2 blocks, and holds only 0 and 255.
    - _Requirements: 1.12, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 13.1_

  - [x] 5.4 Write the property test for Property 10 in `tests/test_frames_mask.py`, with Hypothesis
    - **Property 10: The Sample_Buffer thins evenly and the Mask_Samples stay in the Kept_Range**
    - **Validates: Requirements 1.12, 7.1**
    - Draw n from 1 to 2,000 and `mask_samples` m from 2 to 40. Feed n new 1 × 1 `uint8` frames one at a time to `SampleBuffer(m)`, and keep each one so the test can check identity.
    - After every `add`, with N frames given so far, the buffer holds all N while N ≤ 2m, and m + 1 to 2m after that. Its Frame_Numbers are exactly `range(0, N, k)` and `stride` is k, where the oracle's k is the smallest power of 2 with ceil(N / k) ≤ 2m. The oracle never reads the buffer to work out k.
    - Then draw a Kept_Range `0 <= first <= last <= n − 1`, and let c be the number of held Frame_Numbers from `first` to `last`, counted by the oracle. `samples_in(first, last)` gives min(c, m) pairs in Frame_Number order, each holding the very array fed with its Frame_Number. With c ≤ m they're all c. Otherwise they're m of them, the first and last of the c included, and the gaps between neighbours' places among the c differ by at most 1.
    - _Requirements: 1.12, 7.1, 13.2, 13.7_

  - [x] 5.5 Write the property test for Property 11 in `tests/test_frames_mask.py`, with Hypothesis
    - **Property 11: The Mask is unused or covers the allowed share**
    - **Validates: Requirements 7.3, 7.4, 7.5, 7.6, 7.7, 7.11**
    - Draw 0 to 8 samples of one shape up to 12 × 16, some with a planted region that never changes, a `mask_ratio` from 0 to 1, and `mask_min_coverage <= mask_max_coverage` from 0 to 1.
    - Check that `decide_mask` uses no Mask with fewer than 2 samples, and otherwise uses none or one whose coverage is from `mask_min_coverage` to `mask_max_coverage`. Its stuck pixels are exactly those whose mean absolute change, worked out by the oracle, is below `mask_ratio` times the median.
    - _Requirements: 7.3, 7.4, 7.5, 7.6, 7.7, 7.11, 13.2_

  - [x] 5.6 Write the property test for Property 12 in `tests/test_frames_mask.py`, with Hypothesis
    - **Property 12: The Mask image holds only black and white at the Pick_Size**
    - **Validates: Requirements 7.8**
    - Draw boolean stuck arrays from 1 × 1 to 12 × 16 and Pick_Sizes from 1 × 1 to 40 × 40.
    - Check that `mask_image` gives a `uint8` array of the Pick_Size holding only 0 and 255. The oracle finds, with `Fraction` arithmetic, the Score pixel whose area holds each output pixel's centre, and the output pixel is 0 exactly when that Score pixel is stuck.
    - _Requirements: 7.8, 13.2_

  - [x] 5.7 Write unit tests for the files in `tests/test_frames_files.py`
    - `pick_name(0)` is `000000.jpg` and `pick_name(123)` is `000123.jpg`.
    - `grey_copy` gives a grey image of the colour frame's size, equal to `cv2.equalizeHist` of its grey.
    - `write_jpeg` and `write_png` write files that `cv2.imdecode` reads back at the right size. The PNG is 8-bit grey and holds only 0 and 255.
    - With `Path.write_bytes` patched to raise `OSError(errno.ENOSPC, "No space left on device")`, each writer raises an `ExpectedFailure` naming the file and the reason, with no "Traceback". A failed encode does too.
    - `frames_record`, given numpy values, passes `check_json_value` and holds every field of the design's table. `mask_file` is `"mask.png"` when the Mask is used and `None` when it isn't, and `colour_transfer` can be `None`.
    - _Requirements: 2.7, 3.6, 4.10, 7.9, 8.3, 8.4, 8.6, 8.10, 13.1, 13.9_

  - [x] 5.8 Write the memory test in `tests/test_frames_mask.py`
    - 20,000 frames of 4 × 4 go through a `Scorer` and a `SampleBuffer` with `mask_samples` = 3, one at a time as the Score_Pass feeds them, while the test keeps only a `weakref` to each frame.
    - After every frame, the buffer holds at most 6 frames, and every frame still alive is the current one, the one before it or one the buffer holds.
    - _Requirements: 1.12, 1.13, 13.7_

  - [x] 5.9 Run the tests and commit
    - Run `uv run pytest`. Every test passes, `tests/test_guard.py` included.
    - On `feat/frame-picking`: `git add splatpipe/frames/mask.py splatpipe/frames/files.py tests/test_frames_mask.py tests/test_frames_files.py`, then commit those files with the message "Frames: sample buffer, mask and work folder files".
    - _Requirements: 13.3, 13.11_

- [x] 6. ffprobe and the ffmpeg decode, without the Tone_Map
  - Files: `splatpipe/frames/ffmpeg.py` and `tests/test_frames_ffmpeg.py`, both new.
  - _Requirements: 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 1.10, 1.12, 1.13, 1.14, 2.1, 13.1, 13.2, 13.3, 13.4, 13.5, 13.6, 13.9, 13.11_

  - [x] 6.1 Write `splatpipe/frames/ffmpeg.py`
    - `VideoInfo`, `parse_stream()`, `probe_stream()`, `seek_text()`, `DecodePlan`, `decode_command()`, `parse_frame_time()` with `NOT_A_FRAME`, `FrameGate`, `DecodeFailed` and `decode()`, as in the design's "ffmpeg.py" section. `DecodePlan` has no `tone_map` field yet. Task 8 adds it, with `TONE_MAP`, `ToneMap`, `tone_map_filter()`, `parse_filters()` and `has_filters()`.
    - Each tool runs as an argument list with no shell, with `stdin=subprocess.DEVNULL` and the video as `"file:" + str(path.resolve())`. `probe_stream` passes its limit to `subprocess.run` as `timeout`.
    - `parse_stream` takes the first video stream that isn't cover art. It applies the sample aspect ratio and the rotation to give the Displayed_Shape, and reads `color_transfer` as a non-empty string or `None`.
    - `seek_text` writes exactly six decimals with no exponent, rounding down for the start and up for the end. `decode_command` gives the design's command: `-ss` and `-to` from `seek_text`, `-map 0:<index>`, `scale=W:H,setsar=1,format=<gray|bgr24>,showinfo=checksum=0`, `-fps_mode passthrough`, and `-f rawvideo pipe:1` at the end.
    - `FrameGate` works out each Frame_Time as `float(seek_text(start)) + pts_time`, and accepts it only when it isn't `None`, is inside the Used_Part, and is after the last time it accepted.
    - `decode` reads stderr on a daemon thread and pairs the k-th frame on stdout with the k-th showinfo time. It passes each frame the gate accepts to `on_frame` and holds one frame at a time. Each frame is a new array over its own `bytes`, which `decode` never reuses, so `on_frame` may keep it, as the `Scorer` and the `SampleBuffer` do. A `threading.Timer` kills ffmpeg at the limit. A `finally` cancels the timer, kills and waits for ffmpeg, joins the thread and closes the pipes.
    - Each failure is an `ExpectedFailure` with no traceback: a missing tool, which says to install ffmpeg, a timeout naming `ffmpeg_timeout_seconds` or `ffprobe_timeout_seconds` in `[frames]`, a non-zero exit with the tool's last stderr line (`DecodeFailed` for ffmpeg), ffprobe output that isn't the JSON asked for, a stream with no picture size, and a frame without a showinfo time or output ending inside a frame, which says ffmpeg's output couldn't be read and to install ffmpeg 5.1 or newer.
    - _Requirements: 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 1.10, 1.12, 1.13, 1.14, 2.1_

  - [x] 6.2 Write unit tests for the pure parts in `tests/test_frames_ffmpeg.py`
    - `parse_stream` on sample ffprobe JSON: cover art is skipped, `rotation: -90` in `side_data_list` and `tags.rotate = "90"` swap the edges, and `180` doesn't. A `sample_aspect_ratio` of `4:3` widens the picture and `0:1` doesn't. A missing `color_transfer` gives `None`, and `arib-std-b67` comes through as it is. A stream without a size is refused.
    - `parse_frame_time` reads showinfo lines from ffmpeg 5.1, 6, 7 and 9 (the laptops have 9.0.2), gives `None` for a `NOPTS` line, and gives `NOT_A_FRAME` for side data and config lines.
    - `decode_command` for a video named `-clip.mp4` is a list that starts with `ffmpeg -nostdin`, ends with `pipe:1`, and holds `file:` with the absolute path, `-fps_mode passthrough` and `-map 0:<index>`. A grey plan and a colour plan differ only in the size and the format.
    - _Requirements: 1.3, 1.5, 1.7, 2.1, 13.1_

  - [x] 6.3 Write unit tests with fake processes in `tests/test_frames_ffmpeg.py`
    - Replace `subprocess.run` and `subprocess.Popen` as `splatpipe.frames.ffmpeg` sees them, with `monkeypatch`.
    - A missing ffprobe and a missing ffmpeg each name the tool and say to install ffmpeg, which includes ffprobe.
    - An exit of 1 gives the tool's last stderr line, also when it holds bytes that aren't UTF-8. ffprobe output that isn't JSON is refused.
    - A hung ffmpeg, whose stdout is the read end of an `os.pipe()` that its `kill()` closes, is stopped at a 0.2 s limit with the message naming `ffmpeg_timeout_seconds` in `[frames]`. A timed-out ffprobe names `ffprobe_timeout_seconds`.
    - `decode` pairs frames with times in order and skips the frames the gate refuses. Each frame it gives is a new array, so a frame that `on_frame` keeps still holds its own pixels after the later frames come.
    - It refuses a frame without a time and output that ends inside a frame, saying ffmpeg's output couldn't be read and to install ffmpeg 5.1 or newer. An exception from `on_frame` goes up after the process is cleaned up.
    - No message holds "Traceback".
    - _Requirements: 1.8, 1.9, 1.10, 1.12, 1.14, 13.6, 13.9_

  - [x] 6.4 Write the property test for Property 3 in `tests/test_frames_ffmpeg.py`, with Hypothesis
    - **Property 3: The Frame_Gate keeps exactly the Used_Part, in order**
    - **Validates: Requirements 1.4, 1.6**
    - Draw a Used_Part `start < end` and a sequence of reported times from `st.none() | st.floats(-10, 700)`, mixed with `start`, `end` and repeats of earlier draws, in any order.
    - The oracle takes exactly the times that aren't `None`, are at or after `start`, are before `end`, and are greater than every time taken before them.
    - _Requirements: 1.4, 1.6, 13.2_

  - [x] 6.5 Write the property test for Property 4 in `tests/test_frames_ffmpeg.py`, with Hypothesis
    - **Property 4: The seek text holds the Used_Part**
    - **Validates: Requirements 1.4**
    - Draw finite floats from 0 to 1e6.
    - Check that both texts are digits with one `.` and exactly six decimals, never an exponent. Read back as `Fraction`, the start text is at most x and the end text at least x, each within 1e-6 of x.
    - _Requirements: 1.4, 13.2_

  - [x] 6.6 Write the real ffmpeg tests in `tests/test_frames_ffmpeg.py`
    - Skip them with the reason "ffmpeg and ffprobe aren't installed" when either tool is missing. Make each clip in `tmp_path` with ffmpeg, as an argument list with `timeout=60`.
    - A 3 s `testsrc` clip at 10 fps, decoded with a Used_Part of 1.0 to 2.0, gives exactly the 10 frames from 1.0 to 1.9.
    - A clip whose frames are 0.1 s apart and then 0.3 s apart gives those times within 1 ms, with no frame repeated or dropped.
    - `probe_stream` on a clip gives its width and height.
    - _Requirements: 1.4, 1.5, 1.6, 13.4, 13.5, 13.11_

  - [x] 6.7 Run the tests and commit
    - Run `uv run pytest`. Every test passes, `tests/test_guard.py` included. The real ffmpeg tests run where ffmpeg is installed and are skipped, with their reason, where it isn't.
    - On `feat/frame-picking`: `git add splatpipe/frames/ffmpeg.py tests/test_frames_ffmpeg.py`, then commit those files with the message "Frames: ffprobe and the ffmpeg decode".
    - _Requirements: 13.3, 13.11_

- [x] 7. The step and its CLI
  - Files: `splatpipe/frames/__main__.py` and `tests/test_frames_step.py` (new). `splatpipe/frames/__init__.py` and `tests/conftest.py` (changed).
  - _Requirements: 1.1, 1.7, 1.8, 1.11, 1.12, 1.13, 3.6, 5.2, 5.3, 5.4, 6.9, 6.10, 7.1, 7.4, 7.5, 7.8, 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8, 8.9, 8.10, 9.4, 9.6, 9.7, 10.1, 10.2, 10.3, 10.4, 10.6, 11.1, 11.2, 11.10, 11.11, 13.1, 13.3, 13.4, 13.5, 13.6, 13.7, 13.8, 13.9, 13.11_

  - [x] 7.1 Write the step in `splatpipe/frames/__init__.py`
    - `Tools` with `probe` and `decode`, `TOOLS`, `run()` and `work()`, as in the design's "The step's flow" section, for clips that aren't HDR. Task 8 adds `has_filters`, `HDR_TRANSFERS` and the Tone_Map choice. `run` calls `run_step("frames", "splatpipe.frames", ...)`, like the Fake_Step.
    - `work` reads `TOOLS` when it's called, so a test's `monkeypatch` reaches it through `run_step` and the CLI. It runs the design's order: the settings, the probe, the Score_Pass, choosing and the Mask, the Pick_Pass, `mask.png` when it's used, and `frames.json` last. The parts run inside `step.timed("probe")`, `"score_pass"`, `"choose"` and `"pick_pass"`.
    - Both passes take the Used_Part from the Job_Record's `start_seconds` and `end_seconds`, and the same video, stream and filters but for the size and the format.
    - The Score_Pass feeds each frame to a new `Scorer` and a new `SampleBuffer(mask_samples)`. Its `on_frame` calls `scorer.add(time, frame)`, then `buffer.add(frame)`, and keeps nothing itself.
    - Each count is added with `int(...)` as soon as it's known, so a failed report holds what the step worked out before it failed. Every value bound for JSON is a plain Python type.
    - When the Jolt_Filter keeps fewer frames than Target_Picks, every Kept_Range frame is a Candidate, `jolt_dropped_frames` is 0 and the Req 5.3 warning is added. With fewer Picks than `min_picks`, the step fails with the Req 6.10 message before the Pick_Pass.
    - After the Still_Trim, the Mask_Samples are the frames of `buffer.samples_in(kept.first, kept.last)`, and `decide_mask` decides the Mask. The Score_Pass and choosing run in one helper that gives back the Scores, the Kept_Range, the Picks and the `MaskChoice`, but not the `SampleBuffer`, so the buffered frames are freed before the Pick_Pass.
    - The Pick_Pass makes `images/` and `grey/` and saves each frame whose Frame_Time is a Pick's, once, as its Colour_Pick and its Grey_Copy. With fewer Picks saved than chosen, the step fails with "ffmpeg gave back <k> of the <n> picks".
    - The step writes nothing before the Pick_Pass and nothing to `out/`, and relies on the Step_Runner emptying `work/frames/` before it runs.
    - _Requirements: 1.1, 1.7, 1.11, 1.12, 1.13, 3.6, 5.2, 5.3, 5.4, 6.9, 6.10, 7.1, 7.4, 7.5, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8, 8.9, 9.4, 10.1, 10.2, 10.3, 10.4, 10.6, 11.1, 11.10, 11.11_

  - [x] 7.2 Write `splatpipe/frames/__main__.py`
    - `main(argv)` calls `step_main("frames", "splatpipe.frames", argv)`, and the file ends with `if __name__ == "__main__": sys.exit(main())`.
    - _Requirements: 11.1, 11.2_

  - [x] 7.3 Add `FakeTools` to `tests/conftest.py`
    - `FakeTools`, as in the design's "Stand-ins" section, is a `Tools` built from a `VideoInfo` and a list of synthetic `(time, frame)` pairs. It can also be built from a count and a function that makes pair k on demand, so a test can feed a long stream without holding it.
    - Its `decode` gives only the pairs inside the plan's Used_Part, through `FrameGate`, as `decode` does. It gives grey frames at the plan's size for a grey plan, and colour frames for a colour plan, made from the same pair so a test can tell which frame became which Pick.
    - It records every plan, and it can be set to give fewer frames in the Pick_Pass or to raise a chosen error.
    - A fixture installs it with `monkeypatch.setattr("splatpipe.frames.TOOLS", ...)`.
    - _Requirements: 13.6, 13.7_

  - [x] 7.4 Write unit tests for a done run in `tests/test_frames_step.py`
    - Run the step through `splatpipe.frames.run` with `FakeTools`, so the Step_Runner and the real report are part of each test. Use `set_frames` to make `score_width` and `long_edge` small, so every frame stays tiny.
    - The Work_Folder holds exactly `images/`, `grey/` and `frames.json`, plus `mask.png` when the frames have a corner that never changes. The names in `images/` and `grey/` match, from `000000.jpg`, and each Grey_Copy has its Colour_Pick's size. The Mask is an 8-bit PNG of 0 and 255 at the Pick_Size.
    - Each Colour_Pick's colour shows that it came from the frame its `frames.json` entry names.
    - The report holds every count of Req 10.1 as an `int` and the four timings, and its `settings` holds the 22 Shared_Settings and `base_picks`. `frames.json` passes `check_json_value` and holds every field.
    - The plans use the Job_Record's Trim, for a Job made with `new_job(start=..., end=...)`.
    - With frames that make the Jolt_Filter keep fewer frames than Target_Picks, the step skips it with the Req 5.3 warning. With enough sharp frames, the Candidates are exactly the kept frames.
    - After the run, nothing new is in `out/` but the report. A second run on the same Job leaves only its own files.
    - _Requirements: 1.1, 1.7, 3.6, 5.2, 5.3, 5.4, 7.4, 7.5, 7.8, 8.2, 8.3, 8.4, 8.5, 8.6, 8.8, 10.1, 10.2, 10.4, 10.6, 11.11, 13.1_

  - [x] 7.5 Write unit tests for failed runs and the CLI in `tests/test_frames_step.py`
    - Too few Picks: the Readable_Error gives both numbers and what to film differently, `FakeTools` recorded no Pick_Pass plan, `work/frames/` is empty, and the report holds the counts worked out so far and no `pick_pass` timing.
    - No frames in the Used_Part gives the Req 1.11 message. A Pick_Pass that gives fewer Picks gives the Req 8.9 message and leaves no `frames.json`.
    - A `Path.write_bytes` that raises `OSError(errno.ENOSPC, ...)` fails the step, naming the file and "No space left on device".
    - A bad setting and a failed cross check fail the step before `FakeTools` records any call. A missing ffmpeg, raised by the stand-in, gives the Req 1.8 message.
    - Every failed report has `expected: true` and a `readable` that names what its requirement says and holds no "Traceback".
    - `splatpipe.frames.__main__.main(["--job", job_id])` prints `frames: done` and returns 0 for a done run, and prints the Readable_Error on stderr and returns 1 for a failed one.
    - _Requirements: 1.8, 1.11, 6.9, 6.10, 8.7, 8.9, 8.10, 9.6, 9.7, 10.3, 11.1, 11.2, 11.10, 13.8, 13.9_

  - [x] 7.6 Write the real ffmpeg step test in `tests/test_frames_step.py`
    - Skip it the same way as task 6's real tests. Make a 64 × 48 clip in `tmp_path`.
    - Make the Job with `new_job`, then copy the clip over its `input/video.mp4`. The test config's `max_video_bytes` (1024) and `min_used_seconds` (3) would refuse a real clip in `make_job`, and the step reads only `video_path` and the Job_Record's Trim.
    - Run `splatpipe.frames.run` with the real `TOOLS`. The step is done, the Picks are 64 × 48, and the Work_Folder has the design's layout.
    - _Requirements: 8.1, 8.4, 8.5, 13.4, 13.5, 13.11_

  - [x] 7.7 Write the step's memory test in `tests/test_frames_step.py`
    - Build `FakeTools` from a count of 5,000 and a function that makes each tiny frame on demand, and set `mask_samples` to 3 with `set_frames`. Run the step through `splatpipe.frames.run`.
    - The test keeps only a `weakref` to each frame the Score_Pass is given. After every frame, at most 8 of them are alive: the frame before, the current frame and the buffer's 6.
    - None of them is alive any more when the Pick_Pass starts, because the `SampleBuffer` is freed before it.
    - _Requirements: 1.12, 1.13, 13.7_

  - [x] 7.8 Run the tests and commit
    - Run `uv run pytest`. Every test passes, `tests/test_guard.py` included.
    - On `feat/frame-picking`: `git add splatpipe/frames/__init__.py splatpipe/frames/__main__.py tests/conftest.py tests/test_frames_step.py`, then commit those files with the message "Frames: the step and its CLI".
    - _Requirements: 13.3, 13.11_

- [x] 8. HDR tone map with its fallback
  - Files: `splatpipe/frames/ffmpeg.py`, `splatpipe/frames/__init__.py`, `tests/conftest.py`, `tests/test_frames_ffmpeg.py`, `tests/test_frames_step.py` and `.kiro/specs/frames/requirements.md` (changed). `docs/third-party-licences.md` and `tests/test_docs.py` change only with Samarth's OK on O1.
  - _Requirements: 1.7, 1.8, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 9.4, 10.5, 12.7, 13.1, 13.3, 13.5, 13.6, 13.11_

  - [x] 8.1 Add the Tone_Map to `splatpipe/frames/ffmpeg.py`
    - The Fixed_Constant `TONE_MAP`, the design's filter text from `zscale=t=linear:npl={npl}` to `format=yuv420p`, with `tonemap=tonemap=hable:desat={desat}` and the BT.709 names. Only `{npl}` and `{desat}` are left open.
    - `ToneMap(npl, desat)` holds `tonemap_npl` and `tonemap_desat`. `tone_map_filter(tone_map)` fills `TONE_MAP`'s two numbers in, each written as `repr(float(value))`, like `100.0` or `0.25`, and leaves the rest of the text as it is.
    - `DecodePlan` gains `tone_map: ToneMap | None`. `decode_command` puts `tone_map_filter(plan.tone_map)` after `scale=W:H,setsar=1` and before `format=`, only when `tone_map` isn't `None`, in both kinds of plan.
    - `parse_filters()` and `has_filters()`. `has_filters` runs `["ffmpeg", "-hide_banner", "-nostdin", "-filters"]` with the same `subprocess.run` options and the given limit. A missing ffmpeg is the Req 1.8 failure. A timeout or a non-zero exit gives `False`.
    - _Requirements: 1.7, 1.8, 2.2_

  - [x] 8.2 Add the HDR choice to the step
    - In `splatpipe/frames/__init__.py`: `HDR_TRANSFERS = ("arib-std-b67", "smpte2084")`, and `Tools` and `TOOLS` gain `has_filters`.
    - Inside `step.timed("probe")`, an HDR_Clip runs `TOOLS.has_filters(("zscale", "tonemap"), ffprobe_timeout_seconds)`. With both filters, `tone_map` is `ToneMap(tonemap_npl, tonemap_desat)` from the settings. Without both, `tone_map` is `None`, both passes run without the Tone_Map, and the Req 2.4 warning names the transfer. Any other clip, one with no reported transfer included, gets no filter check and a `tone_map` of `None`.
    - When the Score_Pass with the Tone_Map raises `DecodeFailed`, the step adds the Req 2.5 warning with ffmpeg's last line, sets `tone_map` to `None`, and runs the pass again with a new `Scorer` and a new `SampleBuffer`, so nothing from the failed pass is kept. Any other `DecodeFailed` fails the step as before.
    - The Pick_Pass takes the Score_Pass's final `tone_map`, the same `ToneMap` or `None`. `frames.json` records `colour_transfer` and `tone_mapped`.
    - In `tests/conftest.py`, `FakeTools` gains `has_filters`, a setting to report no filters, and a setting to fail the first pass that has the Tone_Map after a few frames, with a chosen last line.
    - _Requirements: 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 9.4, 10.5_

  - [x] 8.3 Write unit tests for the Tone_Map text, the command and the filter check in `tests/test_frames_ffmpeg.py`
    - `tone_map_filter(ToneMap(100.0, 0.0))` gives exactly the design's text, `zscale=t=linear:npl=100.0,format=gbrpf32le,zscale=p=bt709,tonemap=tonemap=hable:desat=0.0,zscale=t=bt709:m=bt709:r=tv,format=yuv420p`.
    - `ToneMap(203.5, 0.25)` gives the same text with `npl=203.5` and `desat=0.25`, so only the two numbers change, and `hable` and the BT.709 parts stay fixed.
    - `decode_command` with a `ToneMap` has its filter text between `setsar=1` and `format=`, in a grey and a colour plan alike. Without one, no `zscale` or `tonemap` appears.
    - `parse_filters` on a sample `ffmpeg -filters` listing finds `zscale` and `tonemap` and skips the header lines.
    - With fake processes, `has_filters` is true with both filters listed and false with one missing. It's false on a timeout and on an exit of 1, and raises the Req 1.8 failure when ffmpeg is missing.
    - _Requirements: 1.8, 2.2, 2.3, 13.1, 13.6_

  - [x] 8.4 Write the HDR step tests in `tests/test_frames_step.py`
    - An `arib-std-b67` stand-in and a `smpte2084` one, with both filters, run both passes with a `ToneMap`, and `frames.json` has `tone_mapped: true` and the transfer.
    - With `set_frames(tonemap_npl=203.5, tonemap_desat=0.25)`, both recorded plans hold `ToneMap(203.5, 0.25)`, and each plan's `decode_command` holds `npl=203.5` and `desat=0.25` with the `hable` and BT.709 parts unchanged.
    - Without the filters, neither pass has the Tone_Map. The warning names the transfer, says the Picks' colours may look flat, and says an ffmpeg with zscale fixes them. The step is done.
    - A first pass that gives a few frames and then fails with the Tone_Map runs again without it, with the Req 2.5 warning holding ffmpeg's line, and the Pick_Pass runs without it too. `decoded_frames` and `mask_samples` count only the second pass's frames, because the retry starts a new `Scorer` and a new `SampleBuffer`.
    - A `bt709` clip and a clip with no reported transfer make no `has_filters` call and have no Tone_Map. `frames.json` records `"bt709"` or `null`.
    - _Requirements: 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 9.4, 10.5, 13.1, 13.6_

  - [x] 8.5 Write the real Tone_Map test in `tests/test_frames_ffmpeg.py`
    - Skip it when ffmpeg or ffprobe is missing, and also, with the reason "this ffmpeg has no zscale or tonemap filter", when `has_filters(("zscale", "tonemap"), 30)` is false.
    - A tiny clip tagged HLG (`arib-std-b67`), made in `tmp_path` with `timeout=60`, decodes with `ToneMap(100.0, 0.0)` and gives its frames at the plan's size.
    - _Requirements: 2.2, 13.5, 13.11_

  - [x] 8.6 Stop for Samarth's OK on zimg (O1)
    - Before writing anything about zimg, ask in the chat whether the Tone_Map may rely on zimg (libzimg), which ffmpeg's `zscale` needs and which is under the WTFPL, a licence that isn't on the list in `.kiro/steering/rules.md`. Wait for Samarth's answer. Skip the question only when the review has already given his OK.
    - With his OK: add a zimg row to `docs/third-party-licences.md`. It gives the WTFPL, says ffmpeg's `zscale` uses it inside the ffmpeg command line tool on our private machine image, never vendored or linked, and says "Not yet" for Installed. Add a test in `tests/test_docs.py` that the row names the WTFPL and `zscale`.
    - Without his OK: add no zimg row. The code stays as it is, because an ffmpeg without `zscale` already takes the warning path that 8.4 tests, and the step is still done.
    - Either way, write his answer under O1 in `.kiro/specs/frames/requirements.md`.
    - _Requirements: 12.7_

  - [x] 8.7 Run the tests and commit
    - Run `uv run pytest`. Every test passes, `tests/test_guard.py` included. The real Tone_Map test runs or is skipped with its reason.
    - On `feat/frame-picking`: `git add splatpipe/frames/ffmpeg.py splatpipe/frames/__init__.py tests/conftest.py tests/test_frames_ffmpeg.py tests/test_frames_step.py .kiro/specs/frames/requirements.md`, plus `docs/third-party-licences.md tests/test_docs.py` only when 8.6 changed them. Commit those files with the message "Frames: HDR tone map with its fallback".
    - _Requirements: 13.3, 13.11_

- [x] 9. The frames section of the pipeline doc
  - Files: `docs/pipeline.md` and `tests/test_docs.py`, both changed.
  - _Requirements: 7.10, 9.10, 12.1, 12.2, 12.3, 12.4, 12.8, 13.3, 13.11_

  - [x] 9.1 Write `## Step: frames` in `docs/pipeline.md`
    - Put it after `## Step: fake`, with the Fake_Step section's parts: `### What it reads`, `### What it writes`, `### Config section`, `### Step report fields` and `### Files in out/`. Give the CLI, `uv run python -m splatpipe.frames --job <id> [--preset P]`, and `splatpipe.frames.run(job_id, preset=None)`.
    - What it reads: `step.job.video_path`, the Job_Record's `start_seconds` and `end_seconds`, and ffmpeg and ffprobe 5.1 or newer as command line tools. It says the Job's machine image needs ffmpeg and ffprobe 5.1 or newer, because the step uses ffmpeg's `-fps_mode` option, and that Ubuntu 24.04's packaged ffmpeg, version 6.1, is new enough.
    - What it writes: the Work_Folder's layout and the Frames_Record's fields, as the contract the Camera_Solve_Step reads. `frames.json` is written last, so a Work_Folder without it is unfinished. Only matching uses the Grey_Copies. COLMAP takes the Mask as `--ImageReader.camera_mask_path` and the training loss reads the same file, and black means ignore.
    - Config section: each `[frames]` key and `base_picks` for each Preset. The Step_Needs: at least 16 vCPUs and 16 GB of memory on the Job's cloud machine, which the cloud spec reads. The step itself never checks the machine.
    - Step report fields: the counts, the four timings and the three warnings.
    - Files in out/: the step writes no files to `out/` but its report, which the Step_Runner writes.
    - _Requirements: 7.10, 9.10, 12.1, 12.2, 12.3, 12.4, 12.8_

  - [x] 9.2 Write the doc tests for the frames section in `tests/test_docs.py`
    - `## Step: frames` exists after the Skeleton's headings, with its five `###` parts.
    - "Files in out/" says the step writes no files there but its report.
    - "What it reads" says the machine image needs ffmpeg and ffprobe 5.1 or newer, names `-fps_mode`, and says Ubuntu 24.04's packaged ffmpeg is 6.1.
    - "What it writes" names `images/`, `grey/`, `frames.json`, `mask.png`, `--ImageReader.camera_mask_path`, the training loss, matching, and every top-level field of the Frames_Record.
    - "Config section" names every key of the real `[frames]`, read by path, and `base_picks`. It states 16 vCPUs and 16 GB, and that the step never checks the machine.
    - "Step report fields" names every count of Req 10.1 and the four timings.
    - _Requirements: 7.10, 9.10, 12.1, 12.2, 12.3, 12.4, 12.8_

  - [x] 9.3 Run the tests and commit
    - Run `uv run pytest`. Every test passes, `tests/test_guard.py` included.
    - On `feat/frame-picking`: `git add docs/pipeline.md tests/test_docs.py`, then commit those files with the message "Frames: pipeline doc section".
    - _Requirements: 13.3, 13.11_

- [x] 10. The frames step joins the step list
  - Files: `splatpipe/run.py`, `tests/test_run.py`, `tests/test_imports.py`, `tests/test_docs.py`, `tests/test_frames_step.py`, `docs/pipeline.md` and `.kiro/specs/pipeline-core/requirements.md`, all changed.
  - _Requirements: 8.2, 11.3, 11.4, 11.5, 11.6, 11.7, 11.8, 11.9, 12.5, 13.3, 13.4, 13.5, 13.11_

  - [x] 10.1 Change the Step_List in `splatpipe/run.py`
    - `STEP_LIST = (StepEntry("frames", "splatpipe.frames", True),)`. The comment above it says the Frames_Step is the first entry, each later step spec adds its own entry after it, in order, and the Fake_Step stays out.
    - `_NO_STEPS` and the refusal of an empty list that a caller gives stay as they are.
    - _Requirements: 11.3, 11.4, 11.5_

  - [x] 10.2 Update `tests/test_run.py`
    - `test_the_step_list_starts_empty_and_is_the_default_list` gets a name for the new list, like `test_the_step_list_holds_the_frames_step_and_is_the_default_list`, and asserts the list of 10.1.
    - `test_an_empty_step_list_is_refused_before_out_changes`, which passes `steps=None` as well as `[]`, and `test_main_with_the_empty_default_list_gives_1_and_says_no_steps_are_set_up` set `splatpipe.run.STEP_LIST` to `()` with `monkeypatch`. So does any other test that the test run shows relying on an empty default list.
    - `test_python_m_with_a_bad_job_id_exits_with_1_from_the_repo_root` expects "That isn't a job id." for `splatpipe.run` too, and its docstring no longer says the list is empty.
    - _Requirements: 11.5, 11.6_

  - [x] 10.3 Update `tests/test_imports.py`
    - The module docstring and the docstring of `test_each_step_list_entry_is_a_step_inside_splatpipe_with_its_config_section` no longer say the list starts empty.
    - The check now loads `[frames]` from the real config for each of the three Presets.
    - _Requirements: 11.7_

  - [x] 10.4 Rewrite "The step list" in `docs/pipeline.md`
    - In place of the paragraph that says the list starts empty: the Step_List holds the Frames_Step as its first, required entry, each later step spec adds its own entry after it, and a Job is done after the Frames_Step alone until the camera solve spec adds its entry. Keep the list of planned steps.
    - In `tests/test_docs.py`, a test checks those three points, and that "The list starts empty" and "No steps are set up yet" are gone from the section. The existing Step_List check now finds `## Step: frames`.
    - _Requirements: 11.8, 12.5_

  - [x] 10.5 Add the note to pipeline-core's requirements
    - Under Requirement 8, criterion 2 of `.kiro/specs/pipeline-core/requirements.md`, add a note saying that the frames spec added the first Step_List entry, `StepEntry("frames", "splatpipe.frames", True)`. Leave the criterion's own words as they are.
    - _Requirements: 11.9_

  - [x] 10.6 Write the end-to-end test with real ffmpeg in `tests/test_frames_step.py`
    - Skip it the same way as task 6's real tests. Make a 2 s clip in `tmp_path` whose frame k has brightness 16 + 10k, and the Job as in 7.6.
    - `run_job(job_id)` with the real Step_List gives the Job_Status `done`, and `work/frames/` has the design's layout.
    - Each Colour_Pick's mean grey is within 4 of the brightness of the frame that its `frames.json` entry names.
    - _Requirements: 8.2, 11.4, 13.4, 13.5, 13.11_

  - [x] 10.7 Run the tests and commit
    - Run `uv run pytest`. Every test passes, `tests/test_guard.py` included.
    - On `feat/frame-picking`: `git add splatpipe/run.py tests/test_run.py tests/test_imports.py tests/test_docs.py tests/test_frames_step.py docs/pipeline.md .kiro/specs/pipeline-core/requirements.md`, then commit those files with the message "Frames: first entry in the step list".
    - _Requirements: 13.3, 13.11_

## Notes

- Every test sub-task is required, as `.kiro/steering/rules.md` asks. None is marked optional.
- The last sub-task of each task is its checkpoint: the whole suite, then the commit. So the plan has no separate checkpoint tasks and stays at 10 tasks.
- Every acceptance criterion in `requirements.md` is named by at least one task. Each of the design's 12 correctness properties has its own property test sub-task.
- O1 is still open. Task 8 stops for Samarth's answer before it adds anything about zimg, and the step works and warns without `zscale` whatever the answer.
- The graph runs the tasks in order, one at a time, so each task's test run and commit see only finished work. Inside a task, sub-tasks that write different files share a wave.

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "1.2", "1.3", "1.4"] },
    { "id": 1, "tasks": ["1.5", "1.6"] },
    { "id": 2, "tasks": ["1.7"] },
    { "id": 3, "tasks": ["2.1"] },
    { "id": 4, "tasks": ["2.2"] },
    { "id": 5, "tasks": ["2.3"] },
    { "id": 6, "tasks": ["2.4"] },
    { "id": 7, "tasks": ["2.5"] },
    { "id": 8, "tasks": ["3.1"] },
    { "id": 9, "tasks": ["3.2"] },
    { "id": 10, "tasks": ["3.3"] },
    { "id": 11, "tasks": ["3.4"] },
    { "id": 12, "tasks": ["3.5"] },
    { "id": 13, "tasks": ["4.1"] },
    { "id": 14, "tasks": ["4.2"] },
    { "id": 15, "tasks": ["4.3"] },
    { "id": 16, "tasks": ["4.4"] },
    { "id": 17, "tasks": ["4.5"] },
    { "id": 18, "tasks": ["4.6"] },
    { "id": 19, "tasks": ["5.1", "5.2"] },
    { "id": 20, "tasks": ["5.3", "5.7"] },
    { "id": 21, "tasks": ["5.4"] },
    { "id": 22, "tasks": ["5.5"] },
    { "id": 23, "tasks": ["5.6"] },
    { "id": 24, "tasks": ["5.8"] },
    { "id": 25, "tasks": ["5.9"] },
    { "id": 26, "tasks": ["6.1"] },
    { "id": 27, "tasks": ["6.2"] },
    { "id": 28, "tasks": ["6.3"] },
    { "id": 29, "tasks": ["6.4"] },
    { "id": 30, "tasks": ["6.5"] },
    { "id": 31, "tasks": ["6.6"] },
    { "id": 32, "tasks": ["6.7"] },
    { "id": 33, "tasks": ["7.1", "7.2", "7.3"] },
    { "id": 34, "tasks": ["7.4"] },
    { "id": 35, "tasks": ["7.5"] },
    { "id": 36, "tasks": ["7.6"] },
    { "id": 37, "tasks": ["7.7"] },
    { "id": 38, "tasks": ["7.8"] },
    { "id": 39, "tasks": ["8.1"] },
    { "id": 40, "tasks": ["8.2", "8.3"] },
    { "id": 41, "tasks": ["8.4", "8.5"] },
    { "id": 42, "tasks": ["8.6"] },
    { "id": 43, "tasks": ["8.7"] },
    { "id": 44, "tasks": ["9.1"] },
    { "id": 45, "tasks": ["9.2"] },
    { "id": 46, "tasks": ["9.3"] },
    { "id": 47, "tasks": ["10.1", "10.4", "10.5"] },
    { "id": 48, "tasks": ["10.2", "10.3", "10.6"] },
    { "id": 49, "tasks": ["10.7"] }
  ]
}
```
