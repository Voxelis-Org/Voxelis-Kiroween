# Implementation Plan: cameras

## Overview

This plan builds the Cameras_Step in `splatpipe/cameras/` in 10 tasks, in the design's build order: the settings, the model files, the levelling, the Frames_Contract with the COLMAP runner and the command lines, the database and the solve flow, the undistortion and the mask, the outputs and the step itself, the integration test with the real COLMAP, the step's section of the Pipeline_Doc with the COLMAP licence row, and last the Step_List entry. Task 10 ends with 10.9, the merge of `origin/main`, which brings the Training_Step and the Clean_Step, so the Step_List becomes frames, cameras, train, clean. Each task writes its code with that code's unit tests and Hypothesis property tests, runs the whole suite and commits. The code is Python 3.11. The step adds no package: numpy and `opencv-python-headless` are already runtime dependencies, and pytest and Hypothesis are already in the dev group.

Samarth's later answers, D15 to D20, and the small fixes a to f came once every task was done. They add new sub-tasks to six of the tasks, after each one's last sub-task, so the plan stays at 10 tasks, in the design's "Build order of the update (D15 to D20)": 1.7 to 1.10 keep the two refine keys true (D20); 2.5 to 2.8 teach the model files RADIAL and RADIAL_FISHEYE; 4.10 to 4.16 switch the camera to one shared RADIAL camera (D19), with small fix d; 5.7 to 5.15 cap each COLMAP Time_Limit at the Cap_Deadline (D17), with Property 8; 8.4 to 8.11 add the Cameras_Smoke_Check, `splatpipe/cameras/smoke.py` and its place in the Smoke_Test (D15); and 9.5 to 9.11 update the docs, the skill, the licence row (D18), the frames spec and the devlog, small fixes a, b, c and e. Small fix f is in the design only. D16 changes no code. Each group ends with its own test run and commit, so the whole suite passes at every commit.

Three choices in the order matter:

- **The docs before the Step_List.** Once `STEP_LIST` holds `cameras`, the existing check in `tests/test_docs.py` (`test_the_pipeline_doc_has_a_step_section_for_the_fake_step_and_every_listed_step`) needs a `## Step: cameras` section. So task 9 writes that section, and task 10 changes the Step_List together with the "The step list" paragraph that describes it, as D1 asks.
- **The cameras import check** (Req 12.13, 16.10) goes in task 1, so the test run of every later task checks the new package's imports.
- **The `run_job` test of both steps** goes in task 10, because it needs the Step_List entry. Task 7 tests the step through its own `run()` and CLI.

COLMAP on the laptop: only two tests use it, the `colmap <command> -h` check in 4.8 and the integration test in task 8. Both skip, with their reason, when `colmap` isn't on the PATH, so every other task runs on a laptop without COLMAP. Homebrew's COLMAP 4.2.1 runs both, the integration test's whole solve included, as Samarth OKed (D18), and the step always passes `use_gpu 0`. Since 8.6 and 8.7, both run the Cameras_Smoke_Check's code in `splatpipe/cameras/smoke.py`, and the smoke check's other tests in 8.5 need no COLMAP.

Some files grow over several tasks, so each piece comes with the first task that uses it:

- `splatpipe/cameras/__init__.py`: task 1 makes it with only the package docstring, and task 7 writes the step.
- `splatpipe/cameras/level.py`: task 1 makes it with only `LevelSettings`, which `settings.py` needs, and task 3 writes the levelling.
- `tests/conftest.py`: task 1 adds `[cameras]`, `MISSING`, `set_cameras` and the `colmap_options` oracle. Task 4 adds `cameras_job`. Task 5 adds `FakeColmap` and `fake_colmap` for the commands the solve flow runs. Task 7 gives `FakeColmap` the two undistorters and makes `fake_colmap` install it as the step's tools.
- `tests/test_imports.py`: task 1 adds the cameras import check, and task 10 changes the Step_List check's docstring.
- `tests/test_docs.py`: task 9 tests the cameras section and the COLMAP row, and task 10 tests "The step list".
- `docs/pipeline.md`: task 9 adds `## Step: cameras`, and task 10 replaces the first three sentences of "The step list".
- `tests/test_cameras_step.py`: task 7 writes it, and task 10 adds the `run_job` test of both steps.
- 10.9, the merge, changes `config.toml`, `tests/conftest.py`, `tests/test_imports.py`, `tests/test_docs.py`, `docs/pipeline.md` and `docs/third-party-licences.md` once more, with `splatpipe/run.py`, `tests/test_run.py`, `tests/test_frames_step.py` and `docs/devlog.md`, and `tests/test_cameras_step.py` when its `run_job` test needs it.
- The update's groups, 1.7 to 9.11, change many of these files again, and each of their sub-tasks names the files it changes. `config.toml` changes in 1.7, 4.10 and 8.8, `tests/conftest.py` in 4.11 and 5.10, and `tests/test_cameras_colmap.py` in 1.9, 4.12, 4.14, 5.11, 5.12 and 8.7, each in a wave of its own.

## Working rules for every task

- Work on the existing `feat/camera-solve` branch, never on `main`.
- The only step folder this spec makes is `splatpipe/cameras/`. Of the Skeleton, only the Step_List and its comment in `splatpipe/run.py` change, in task 10. These tasks leave the Frames_Step's code, but the one comment that 9.8 renames, `pyproject.toml`, `uv.lock` and `.kiro/steering/tech.md` as they are. Of `cloud/`, only `cloud/config.py` and `cloud/smoke_runner.py` change, in 8.8 and 8.9, and the Cloud_Runner's uploads stay as they are (D16). The merge in 10.9 brings `main`'s own changes to `pyproject.toml`, `uv.lock` and `.kiro/steering/tech.md`, none of them for the Cameras_Step.
- `splatpipe/cameras/` imports only the standard library, numpy, cv2 and `splatpipe`. Nothing imports `cloud/`, boto3, torch, gsplat or pycolmap. `level.py` is pure and imports only the standard library, numpy and the step's `model`.
- COLMAP runs only as the separate program `colmap`, through `subprocess.run` in `run_colmap` in `colmap.py`. It's never linked or imported, and no other module starts a process.
- cv2 only decodes and encodes the mask PNGs, and in `smoke.py` draws, blurs and encodes the Smoke_Picks. It never reads or writes video: no `cv2.VideoCapture` and no `cv2.VideoWriter`.
- Every number and setting comes from `[cameras]` in `config.toml`, except the design's Fixed_Constants, as its "The Fixed_Constants" table lists them: the program `colmap`, the four log options, `--camera_mode 1`, the feature and matcher types, the GPU and bundle adjustment backend options, `--SequentialMatching.loop_detection 0` and `quadratic_overlap`, `--output_type COLMAP`, the paths, the two Camera_Model names, PINHOLE, the World_Up and +Z, the Mask's 0 and 255, the layout and names of the Model_Files, the Work_Folder and `out/cameras.json`, the Cap_Deadline's variable `SPLATPIPE_CAP_DEADLINE`, and the Smoke_Picks' scene shape. The Cameras_Smoke_Check's numbers come from `[cloud.smoke]`.
- Nothing is downloaded while the step runs: SIFT and `SIFT_BRUTEFORCE` only, loop detection off, no vocabulary tree and no model weights.
- Tests run on the CPU with no network. They build tiny synthetic cameras, points, images and databases themselves, in `tmp_path`. Every Job is made in the temporary Jobs_Folder from `tests/conftest.py`, and the Smoke_Solve's Job_Folder in `tmp_path`, never in the real `data/` folder, and no image, database or model goes into git.
- Tests that need COLMAP use `pytest.mark.skipif(shutil.which("colmap") is None, reason="COLMAP isn't installed")`. Only 4.8 and task 8 need it. Every subprocess a test starts itself gets `timeout=60`, and the step's own COLMAP runs take their Time_Limits from the config.
- Property tests use Hypothesis with at least 100 examples (the default), and each docstring says `Feature: cameras, Property N: <title>`. Each oracle is written in the test with its own code and never calls the function it checks.
- Every test sub-task is required. None is optional.
- Nothing here costs money. No task touches AWS, sets `APPROVED_SPEND`, or runs anything paid or in the cloud. The Cameras_Smoke_Check runs on a smoke machine only when someone runs the Smoke_Test, which needs its own yes, and no task here runs it.
- The last sub-task of each task is its checkpoint. Run `uv run pytest` and fix what fails until every test passes, `tests/test_guard.py` included. Then stage the task's files by name with `git add <file> ...`, never with `git add -A`, `git add .` or a folder, and commit only those files with `git commit -m "<message>" -- <file> ...`, so nothing else that's staged goes in. Leave `docs/devlog.md` out: the devlog hook adds and commits its own line after each task. The merge in 10.9 is the one exception: git refuses a commit of named files while a merge is in progress, so 10.9 stages each file it resolves by name, `docs/devlog.md` included, and commits the merge with a plain `git commit -m "<message>"`.
- If a hook blocks something it shouldn't, say so and fix the hook. Never work around it.

## Tasks

- [x] 1. Settings, the test config and the import check
  - Files: `splatpipe/cameras/__init__.py`, `splatpipe/cameras/level.py`, `splatpipe/cameras/settings.py` and `tests/test_cameras_settings.py` (new). `config.toml`, `tests/conftest.py` and `tests/test_imports.py` (changed).
  - Design: "settings.py", "The `[cameras]` section of `config.toml`", "The settings checks", F13, F14, O1 to O4, O7, O11, C5 and C9, and the `tests/conftest.py` and `tests/test_imports.py` rows of "Changes outside `splatpipe/cameras/`".
  - COLMAP on the laptop: not needed.
  - The update for D20 adds 1.7 to 1.10, which name their own files, design sections and requirements.
  - _Requirements: 2.1, 2.3, 4.4, 8.6, 12.1, 12.2, 12.3, 12.4, 12.5, 12.6, 12.7, 12.8, 12.9, 12.10, 12.11, 12.12, 12.13, 12.14, 12.15, 16.2, 16.4, 16.10, 16.13_

  - [x] 1.1 Add `[cameras]` and its Preset tables to `config.toml`
    - Append `[cameras]`, `[cameras.fast]`, `[cameras.default]` and `[cameras.quality]` after `[frames.quality]`, exactly as the design's "The `[cameras]` section of `config.toml`" writes them, with the comment above every key and the group comments. Leave every other section as it is.
    - That's 190 Shared_Settings. The 173 Option_Keys, `camera_model` included, hold the values of `.kiro/skills/colmap-cli/references/options.md`, with decimal options written as TOML floats (C9). The 17 other Shared_Settings hold the values of O1 to O4 and Req 12.3. Each Preset table holds `near_overlap`: 15, 15 and 25.
    - _Requirements: 4.4, 8.6, 12.1, 12.2, 12.3, 12.4, 12.5_

  - [x] 1.2 Write `splatpipe/cameras/settings.py`
    - Make `splatpipe/cameras/` with an `__init__.py` that holds only the package docstring for now, and a `level.py` that holds only `LevelSettings` with its four levelling settings. Task 7 writes the step, and task 3 the levelling.
    - `Range`, `OptionKey`, `OPTION_KEYS`, `TIMEOUT_KEYS`, `CamerasSettings` and `read_settings(settings, section, preset)`, as in the design's "settings.py" section. `OPTION_KEYS` holds one `OptionKey` per Option_Key, in `options.md`'s order, each with its COLMAP option, the commands that take it and its Range from "The settings checks". The keys follow O11's naming rule.
    - Each key is read with `settings[key]`, so a missing key raises the Step_Runner's own message. The kind checks use `type()` exactly, so a bool is never a number, and a whole number must also fit a 32-bit signed int.
    - Then each Range's own test runs, and then the cross checks, once every single key has passed. A refusal is an `ExpectedFailure` that names the key, its table (`[cameras]`, or `[cameras.<preset>]` for `near_overlap`), COLMAP's option for an Option_Key, and what the value must be, like the design's example message. A cross check names both settings and their values.
    - `options` keeps each Option_Key's value as Config gave it, so an int stays an int and a float stays a float. `timeouts` maps each COLMAP command to its Time_Limit, and `sequential_matcher`'s applies to both Match_Runs. `min_vcpus` and `min_memory_gb` are checked and kept, and nothing in the step uses them.
    - The module imports only the standard library, `splatpipe.errors` and the step's `level`.
    - _Requirements: 2.3, 12.6, 12.7, 12.8, 12.9, 12.10, 12.11_

  - [x] 1.3 Add `[cameras]`, `set_cameras` and the options oracle to `tests/conftest.py`
    - `config_text` gains a `[cameras]` section with its three Preset tables: the real config's `[cameras]`, read once from `config.toml` by path with `tomllib`, with `min_placed_frames = 1` and every `*_timeout_seconds` at 60 on top (C5). Each value keeps its TOML type.
    - A `set_cameras(**values)` fixture rewrites the test config with any `[cameras]` key changed. `near_overlap` takes a dict of the Presets it changes, like `set_frames(base_picks=...)`. The sentinel `MISSING` leaves a key out, and for `near_overlap` it leaves the key out of every Preset table. Like `set_frames`, each call writes the whole file again from `config_text`, so `[frames]` and every `fail` are back to the test config's values too.
    - A `colmap_options` fixture parses the seven code blocks of `options.md` with its own code into each command's options and value texts, in order. It sorts each option the way the design does: a Fixed_Constant, a path, `--SequentialMatching.overlap`, or an Option_Key with the key that O11's naming rule gives it. It's the oracle of the settings test in 1.4 and of Property 1 in 4.7, and it never reads `settings.py`.
    - Update the module docstring to name the new fixtures.
    - _Requirements: 12.15_

  - [x] 1.4 Write unit tests for the settings in `tests/test_cameras_settings.py`
    - The real config, read by path as `REPO_ROOT / "config.toml"`, holds exactly the design's 190 keys in `[cameras]` and `near_overlap` in each Preset table. Each Option_Key's value, but for `camera_model`, equals its value in `options.md` from `colmap_options`: a bool as 1 or 0, a number as the same number, a name as the same text. Decimal options hold TOML floats (C9).
    - The other keys hold the values of O1 to O4 and Req 12.3 and 12.4. `far_overlap` is at least 9, both `multiple_models` keys are true, and `undistortion_blank_pixels` is 0.
    - `step_settings("cameras", preset)` loads the real config for each of the three Presets.
    - Parametrised over every key, with the ranges of the design's "The settings checks" written out in the test, not read from `settings.py`: a value of the wrong kind (a bool for a number, a float for a whole number, a str for a number, an unknown name) and a value just outside the range are each refused with an `ExpectedFailure` naming the key, its table and, for an Option_Key, its COLMAP option. The edge values pass, like 2147483647 for a key with no top and -1 where the range allows it, and 2147483648 is refused.
    - Each of the seven cross checks is refused, naming both settings and their values.
    - A missing key, read through `StepSettings`, gives the Step_Runner's message with `[cameras]`, `[cameras.<preset>]` and the key.
    - `read_settings` on the test config gives a `CamerasSettings` with the chosen Preset's `near_overlap`, each Option_Key's value with its own type, the Time_Limits by command, and the `LevelSettings`. `set_cameras` changes one key for one test, `MISSING` leaves one out, and the next test sees the test config's values again.
    - _Requirements: 4.4, 8.6, 12.1, 12.2, 12.3, 12.4, 12.5, 12.7, 12.8, 12.9, 12.10, 12.11, 12.15, 16.2_

  - [x] 1.5 Write the cameras import check in `tests/test_imports.py`
    - A new test reads every `splatpipe/cameras/*.py` with the file's existing `ast` reader and allows only names in `sys.stdlib_module_names`, numpy, cv2 and `splatpipe`, so torch, gsplat, boto3, pycolmap and `cloud/` stay out. A failure gives the file and the line.
    - The same test finds no `cv2.VideoCapture` and no `cv2.VideoWriter` in those files.
    - Together with the existing dependency test, this shows the step needs no new package, so `pyproject.toml` and `uv.lock` stay as they are.
    - The module docstring describes the new test.
    - _Requirements: 2.1, 12.12, 12.13, 12.14, 16.10_

  - [x] 1.6 Run the tests and commit
    - Run `uv run pytest`. Every test passes, `tests/test_guard.py` and the comment check in `tests/test_config.py` included.
    - On `feat/camera-solve`: `git add config.toml splatpipe/cameras/__init__.py splatpipe/cameras/level.py splatpipe/cameras/settings.py tests/conftest.py tests/test_cameras_settings.py tests/test_imports.py`, then commit those files with the message "Cameras: settings, the test config and the import check".
    - _Requirements: 16.4, 16.13_

  - [x] 1.7 Make the two `ba_refine_extra_params` keys true-only in `splatpipe/cameras/settings.py` and `config.toml`
    - Files: `splatpipe/cameras/settings.py` and `config.toml`, changed.
    - Design: "settings.py" (the `why` of `Range` and the true-only Range), "The settings checks" (the row of the two keys), the two keys' comments under "The `[cameras]` section of `config.toml`", C22 and D20.
    - `Range` gains `why: str = ""`, one sentence that the refusal adds after "Change it to ...". It's empty for every Range but a new true-only bool Range, whose `wanted` is "true" and whose test takes only `True`.
    - `mapper_ba_refine_extra_params` and `global_mapper_ba_refine_extra_params` take the true-only Range in place of the plain bool one. So a false value is refused before the step reads `frames.json` or runs any COLMAP_Command, with the design's message: the key, `[cameras]`, COLMAP's option, "it must be true. Change it to true.", then the `why`, that bundle adjustment has to refine the distortion, because with it off the distortion stays 0, COLMAP's undistorter keeps the camera's own model and copies the picks unwarped, and the step can't hand the train step a PINHOLE camera.
    - In `config.toml`, the two keys keep `true`, and their comments end "Must be true: the undistortion needs it.", as the design writes them.
    - _Requirements: 12.8, 12.9, 12.16_

  - [x] 1.8 Write unit tests for the true-only refine keys in `tests/test_cameras_settings.py`
    - Files: `tests/test_cameras_settings.py`, changed.
    - Design: "The settings checks" and the `tests/test_cameras_settings.py` row of "Tests by file".
    - The ranges written out in the test give the two keys "true only", so the parametrised kind and range tests refuse a value that isn't a bool and refuse false as the value just outside the range.
    - For each of the two keys, false is refused with an `ExpectedFailure` whose message names the key, `[cameras]` and its COLMAP option, says it must be true, gives the reason of Req 12.16 and holds no "Traceback". True passes.
    - The real config holds true in both, and both comments say they must stay true.
    - _Requirements: 12.9, 12.16, 16.2, 16.9_

  - [x] 1.9 Update the property test for Property 1 in `tests/test_cameras_colmap.py` for the true-only refine keys
    - **Property 1: Command lines hold the Fixed_Constants and every option once**
    - **Validates: Requirements 2.3, 2.4, 2.5, 2.6, 2.13, 3.2, 3.3, 3.5, 3.6, 4.1, 4.2, 4.3, 8.3**
    - Files: `tests/test_cameras_colmap.py`, changed.
    - Design: Property 1 and its strategy, and D20.
    - The strategy draws `mapper_ba_refine_extra_params` and `global_mapper_ba_refine_extra_params` only as true, so every drawn setting is one that `read_settings` accepts. Every other key is drawn as before, and the property's checks stay as they are.
    - _Requirements: 2.13, 12.16, 16.3_

  - [x] 1.10 Run the tests and commit
    - Run `uv run pytest`. Every test passes, `tests/test_guard.py` and the comment check in `tests/test_config.py` included.
    - On `feat/camera-solve`: `git add splatpipe/cameras/settings.py config.toml tests/test_cameras_settings.py tests/test_cameras_colmap.py`, then `git commit -m "Cameras: the two refine keys must stay true" -- splatpipe/cameras/settings.py config.toml tests/test_cameras_settings.py tests/test_cameras_colmap.py`.
    - _Requirements: 16.4, 16.13_

- [x] 2. The model files
  - Files: `splatpipe/cameras/model.py` and `tests/test_cameras_model.py`, both new.
  - Design: "model.py", "The Model_Files", F1 to F5, O10 and Property 7.
  - COLMAP on the laptop: not needed. Task 8 checks the Model_Writer against COLMAP itself.
  - The update for D19 adds 2.5 to 2.8, which name their own files, design sections and requirements.
  - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.9, 11.12, 16.2, 16.3, 16.4, 16.13_

  - [x] 2.1 Write `splatpipe/cameras/model.py`
    - `CAMERA_MODELS`, `NO_POINT3D`, `NO_ERROR`, `POINT2D`, `TRACK_ELEMENT`, `FILES`, `Camera`, `Image`, `Points`, `Model`, the three encoders and decoders, `read_images()`, `read_model()` and `write_model()`, as in the design's "model.py" section, with the layout of "The Model_Files".
    - It's our own code, written fresh with `struct` and numpy from the layout tables, with nothing copied from COLMAP's scripts or another repo. The fixed parts of each record go through `struct` with explicit little-endian formats. Each image's 2D points and each point's track go through `np.frombuffer` and `tobytes()` with the two dtypes, so every bit is kept, NaN payloads included.
    - Decoding walks a `memoryview` with an offset. It refuses, with an `ExpectedFailure` that names the file, says what's wrong and says to check that COLMAP 4.2.1 is installed and to run the step again: a file that ends inside a record or inside its count, an unknown camera model id, a name with no `\0` before the end or that isn't UTF-8, an id given twice, and bytes after the last record.
    - Encoding writes the cameras and the points in increasing id order and the images in the Model's order (F3). It raises `ValueError`, a bug in our code, for an id given twice, a name holding `\0`, an unknown model id, or a parameter count that doesn't match the model.
    - `read_model` reads the three files and ignores any others, `rigs.bin` and `frames.bin` included (F4). `write_model` makes the folder and writes exactly the three files with `Path.write_bytes`, and an `OSError` is the Req 11.12 failure naming the file and the reason.
    - _Requirements: 10.1, 10.2, 10.3, 10.4, 11.12_

  - [x] 2.2 Write unit tests for the model files in `tests/test_cameras_model.py`
    - A hand-built model with one camera each of PINHOLE, OPENCV and OPENCV_FISHEYE, two images (one with no 2D points, one with NaN coordinates and `NO_POINT3D`) and three points (one with an empty track) encodes to exactly the bytes the test assembles from the design's layout tables with `struct.pack`. Those bytes decode back to the model.
    - Cameras and points given out of id order come out sorted, and images keep their order. A PINHOLE camera record is 56 bytes, and the image and point records have the sizes the design gives.
    - `read_model` reads a folder of the three files with a `rigs.bin` and a `frames.bin` beside them, and ignores both. `read_images` gives the images of a folder's `images.bin`.
    - Each Req 10.4 fault fails, naming the file and what's wrong, with no "Traceback": a missing file, a file that ends inside each kind of record and inside its count, an unknown model id, a name with no `\0`, a name that isn't UTF-8, an id given twice, and bytes after the last record.
    - The encoders raise `ValueError` for each bug the design lists. `write_model` writes exactly three files. With `Path.write_bytes` patched to raise `OSError(errno.ENOSPC, "No space left on device")`, it fails naming the file and the reason.
    - _Requirements: 10.1, 10.2, 10.4, 11.12, 16.2_

  - [x] 2.3 Write the property test for Property 7 in `tests/test_cameras_model.py`, with Hypothesis
    - **Property 7: The Model_Files round trip**
    - **Validates: Requirements 10.9**
    - Draw models with cameras of the three models, images with any Poses, names and 2D points, and points with any positions, colours, errors and tracks. Ids are unique across the whole `uint32` and `uint64` ranges. Floats are any `float64`, NaN payloads and infinities included. Names have up to 30 Unicode characters that UTF-8 can encode, with no `\0`. Each image has 0 to 20 2D points with `NO_POINT3D` common, and each point 0 to 8 track elements.
    - Decoding what the encoders write gives back the same model, with the cameras and the points in id order. The oracle compares every float bit for bit through its bytes.
    - _Requirements: 10.9, 16.3_

  - [x] 2.4 Run the tests and commit
    - Run `uv run pytest`. Every test passes, `tests/test_guard.py` included.
    - On `feat/camera-solve`: `git add splatpipe/cameras/model.py tests/test_cameras_model.py`, then commit those files with the message "Cameras: the model files".
    - _Requirements: 16.4, 16.13_

  - [x] 2.5 Add RADIAL and RADIAL_FISHEYE to `CAMERA_MODELS` in `splatpipe/cameras/model.py`
    - Files: `splatpipe/cameras/model.py`, changed.
    - Design: "model.py", the `cameras.bin` table of "The Model_Files", F2 and D19.
    - `CAMERA_MODELS` gains `3: ("RADIAL", 5)` and `9: ("RADIAL_FISHEYE", 5)`, each with f, cx, cy, k1, k2 in that order, beside PINHOLE, OPENCV and OPENCV_FISHEYE, so `FakeColmap`'s OPENCV cameras still encode until 4.11. 4.10 then drops the two OPENCV models.
    - The comment above `CAMERA_MODELS` gives each model's parameters, and the module docstring names the models.
    - _Requirements: 10.1, 10.2_

  - [x] 2.6 Update the unit tests for the model files in `tests/test_cameras_model.py`
    - Files: `tests/test_cameras_model.py`, changed.
    - Design: the `tests/test_cameras_model.py` row of "Tests by file" and "The Model_Files".
    - The hand-built model has one camera each of PINHOLE, RADIAL and RADIAL_FISHEYE, with the model ids 1, 3 and 9 and records of 56, 64 and 64 bytes. It encodes to exactly the bytes the test assembles with `struct.pack` from the layout tables, and those bytes decode back to the model.
    - The other model tests stay as they are.
    - _Requirements: 10.1, 10.2, 16.2_

  - [x] 2.7 Update the property test for Property 7 in `tests/test_cameras_model.py`
    - **Property 7: The Model_Files round trip**
    - **Validates: Requirements 10.9**
    - Files: `tests/test_cameras_model.py`, changed.
    - Design: Property 7 and its strategy, and D19.
    - The strategy draws the camera model ids 1, 3 and 9, with 4, 5 and 5 parameters, in place of 1, 4 and 5. Everything else is drawn and checked as before, with every float compared bit for bit.
    - _Requirements: 10.9, 16.3_

  - [x] 2.8 Run the tests and commit
    - Run `uv run pytest`. Every test passes, `tests/test_guard.py` included.
    - On `feat/camera-solve`: `git add splatpipe/cameras/model.py tests/test_cameras_model.py`, then `git commit -m "Cameras: RADIAL and RADIAL_FISHEYE in the model files" -- splatpipe/cameras/model.py tests/test_cameras_model.py`.
    - _Requirements: 16.4, 16.13_

- [x] 3. Levelling the scene
  - Files: `splatpipe/cameras/level.py` (changed) and `tests/test_cameras_level.py` (new).
  - Design: "level.py (pure)", "The Hypothesis strategies for the levelling", A6, O3, C7 and Properties 4 to 6.
  - COLMAP on the laptop: not needed. Every test builds its own cameras and points.
  - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7, 9.8, 9.9, 9.10, 9.11, 9.12, 9.13, 9.15, 9.16, 9.17, 9.18, 10.6, 10.7, 10.8, 16.1, 16.2, 16.3, 16.4, 16.13_

  - [x] 3.1 Write the levelling in `splatpipe/cameras/level.py`
    - `WORLD_UP`, `FORWARD`, `Scene`, `LevelMove`, `Levelling`, the quaternion helpers, `rotation_onto()`, `heading_turn()`, `axis_point()`, `point_median()`, `level()`, `scene_of()` and `move_model()`, as in the design's "level.py (pure)" section, next to task 1's `LevelSettings`.
    - `scene_of` takes the Placed_Frames in Pick_Number order, using `numbers`, with each camera's R, its centre −Rᵀt, and the points.
    - `level` follows the design's eight steps: finite input; the Mean_Up against `up_agreement`; `rotation_onto`, with its half turn when a · g < 0; the Heading_Camera, the first Placed_Frame by Pick_Number that has a Heading; the turn to +Z; the Scene_Centre; the translation; and finite output. The Scene_Centre is the Axis_Point when it passes the eigenvalue share and the in-front share, which compares with `Fraction(axis_point_min_front_share)` as C7 says, else the Point_Median along the levelled axes (A6), else the COLMAP_Frame is kept. The eigenvalue test runs before `numpy.linalg.solve`, so cameras whose A is singular, like parallel ones, go to the Point_Median and never reach the solve.
    - Any exception, or any value that isn't finite, keeps the COLMAP_Frame with the Req 9.12 warning naming the error's type and message, so `level` never raises. A kept COLMAP_Frame gives the identity rotation, a zero translation, and the warning of Req 9.3, 9.7, 9.10 or 9.12, worded as the design gives them.
    - `move_model` applies the whole Level_Move to every Pose and every point at once: q_R from Shepperd's method, q'_i = normalise(q_i ⊗ q̄_R), t'_i = t_i − R'_i T and X'_j = R X_j + T. The camera, every image's id, camera id, name and 2D points, and every point's id, colour, error and track stay as they are.
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7, 9.8, 9.9, 9.10, 9.11, 9.12, 9.13, 10.6, 10.7, 10.8_

  - [x] 3.2 Write unit tests for the levelling in `tests/test_cameras_level.py`
    - Cameras that agree come out level, facing +Z and centred. Cameras that don't, half of them upside down, keep the COLMAP_Frame with the Req 9.3 warning, which gives the Mean_Up's length and `up_agreement`.
    - A first Placed_Frame that looks straight down makes the next one the Heading_Camera. Every frame looking straight down keeps the COLMAP_Frame with the Req 9.7 warning. The Heading_Camera is chosen by Pick_Number, also when `images.bin` lists the images in another order.
    - Cameras on a circle looking at its centre have that centre as their Axis_Point. Exactly parallel cameras, whose A is singular, fail the eigenvalue share, and outward-looking cameras have their Axis_Point behind them, so both use the Point_Median, with no Req 9.12 warning. An in-front share of exactly `axis_point_min_front_share` passes (C7). No points and no well defined Axis_Point keep the COLMAP_Frame with the Req 9.10 warning.
    - A Mean_Up along (0, 1, 0) takes the half turn and still comes out level. A NaN in a Pose, and an error raised inside the maths with `monkeypatch` on `numpy.linalg.solve`, each keep the COLMAP_Frame with the Req 9.12 warning naming the error.
    - `rotation_onto` for a = g, a = −g and random a. Shepperd's method gives back each quaternion, up to its sign, checked against the quaternion formula.
    - `move_model` moves each centre to R C + T and each point to R X + T, and keeps the camera, the images' ids, camera ids, names and 2D points, and the points' ids, colours, errors and tracks. A kept COLMAP_Frame gives exactly the identity rotation and a zero translation.
    - _Requirements: 9.2, 9.3, 9.4, 9.5, 9.6, 9.7, 9.8, 9.9, 9.10, 9.11, 9.12, 9.13, 10.6, 10.7, 10.8, 16.1, 16.2_

  - [x] 3.3 Write the property test for Property 4 in `tests/test_cameras_level.py`, with Hypothesis
    - **Property 4: The Level_Move keeps every distance**
    - **Validates: Requirements 9.1, 9.13, 9.15**
    - Draw scenes with the design's `scenes()`: 1 to 12 cameras with rotations from `rotations()` and centres in [−50, 50]³, 0 to 30 points in [−50, 50]³, and levelling settings across the ranges the design gives.
    - Apply the Level_Move that `level` gives with `move_model`. The oracle works out the distance between every two of the camera centres and the points, before and after, with its own numpy code. Each distance stays within 1e-9 times (1 + the largest distance).
    - _Requirements: 9.1, 9.13, 9.15, 16.3_

  - [x] 3.4 Write the property test for Property 5 in `tests/test_cameras_level.py`, with Hypothesis
    - **Property 5: The Level_Move's rotation is a proper rotation**
    - **Validates: Requirements 9.1, 9.16**
    - For the scenes and settings of Property 4, the Level_Move's rotation R has RᵀR within 1e-12 of I and a determinant within 1e-12 of 1, so its scale is 1.
    - When the Levelling keeps the COLMAP_Frame, R is exactly the identity and the translation exactly zero.
    - _Requirements: 9.1, 9.16, 16.3_

  - [x] 3.5 Write the property test for Property 6 in `tests/test_cameras_level.py`, with Hypothesis
    - **Property 6: Cameras that agree come out level, facing +Z and centred**
    - **Validates: Requirements 9.2, 9.4, 9.5, 9.9, 9.11, 9.17, 9.18**
    - Draw scenes with the design's `agreeing_scenes()`: a level scene whose cameras each tilt at most 28° from one common up, with 1 to 30 points, seen through a random rigid move, with any `up_agreement` up to 0.85 and any `heading_min_degrees` up to 25°.
    - `level` moves the scene. In the moved scene, the oracle works out again with its own code the Mean_Up, which points along (0, −1, 0) within 1e-9, the Heading_Camera's flattened View_Direction, which points along +Z within 1e-9, and the Scene_Centre of the kind `level` used, which lies within 1e-9 times (1 + the scene's size) of the origin.
    - _Requirements: 9.2, 9.4, 9.5, 9.9, 9.11, 9.17, 9.18, 16.3_

  - [x] 3.6 Run the tests and commit
    - Run `uv run pytest`. Every test passes, `tests/test_guard.py` included.
    - On `feat/camera-solve`: `git add splatpipe/cameras/level.py tests/test_cameras_level.py`, then commit those files with the message "Cameras: levelling the scene".
    - _Requirements: 16.4, 16.13_

- [x] 4. The Frames_Contract, the COLMAP runner and the command lines
  - Files: `splatpipe/cameras/frames_input.py`, `splatpipe/cameras/colmap.py`, `splatpipe/cameras/commands.py`, `tests/test_cameras_frames_input.py` and `tests/test_cameras_colmap.py` (new). `tests/conftest.py` (changed).
  - Design: "frames_input.py", "colmap.py", "commands.py" with its table of builders, "The Fixed_Constants", F7, F8, F13, F16, A1, C2, C3 and Property 1.
  - COLMAP on the laptop: only the `-h` test in 4.8, which skips without it.
  - The update for D19 and small fix d adds 4.10 to 4.16, which name their own files, design sections and requirements.
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.10, 2.13, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 4.1, 4.2, 4.3, 6.1, 6.2, 6.6, 8.1, 8.3, 11.12, 16.2, 16.3, 16.4, 16.5, 16.7, 16.8, 16.13_

  - [x] 4.1 Write `splatpipe/cameras/frames_input.py`
    - `Pick`, `FramesInput` with its `numbers` property, `parse_frames_record()` and `read_frames()`, as in the design's "frames_input.py" section. The names `IMAGES`, `GREY`, `MASK`, `RECORD` and `pick_name` come from `splatpipe/frames/files.py`, the contract between the two steps.
    - `read_frames` reads `frames.json` with `splatpipe.jsonio.read_json`. `FileNotFoundError` is the Req 1.2 failure, and any other read error is the Req 1.3 failure saying the file can't be read as JSON.
    - `parse_frames_record` is pure and refuses each fault of the design's list, naming the field. Then every `images/<file>` and `grey/<file>` must be a file, and `mask.png` too when `mask_used` is true, or the Req 1.4 failure names the first missing file.
    - Every refusal is an `ExpectedFailure` that says to run the Frames_Step, with `uv run python -m splatpipe.frames --job <id>` and the Job's id.
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.6_

  - [x] 4.2 Write `splatpipe/cameras/colmap.py`
    - `PROGRAM`, `ColmapRun` with its `value()` method, `RunResult`, `colmap_env()`, `run_colmap()`, `last_error_line()` and `exit_text()`, as in the design's "colmap.py" section. It's the only code that starts COLMAP.
    - `colmap_env` is pure. It gives the base environment without `TEST_TMPDIR` and without every key that starts with `GLOG_`, and with `TMPDIR` and `TMP` set to the given folder (C2).
    - `run_colmap` opens the Command_Log for writing in binary, then runs `subprocess.run` with the argument list, `stdin=subprocess.DEVNULL`, stdout to the log, stderr into stdout, the COLMAP_Folder as `cwd`, the run's environment, and its Time_Limit as `timeout`. `TimeoutExpired` gives `exit_code=None`. `FileNotFoundError` is the Req 2.7 failure. A log that can't be opened is the Req 11.12 failure naming it. The seconds come from `time.perf_counter`, and the Last_Error_Line from reading the whole log once, line by line, with `errors="replace"`, so no size limit goes in code.
    - `last_error_line` takes the log's lines in order, stops at glog's `*** Check failure stack trace: ***` line, keeps the last line before it that isn't empty, and drops glog's prefix (C3). `exit_text` gives `exit code 1`, or `exit code -6 (signal SIGABRT)` for a signal.
    - _Requirements: 2.1, 2.2, 2.7, 2.8, 2.10, 11.12_

  - [x] 4.3 Write `splatpipe/cameras/commands.py`
    - `Paths` with `Paths.of()`, the Fixed_Constants of the command lines, `format_value()`, `fisheye_start()` and the seven builders, as in the design's "commands.py" section and its table. `sequential_matcher(far=...)` gives the Near_Match_Run and the Far_Match_Run.
    - `format_value` writes a bool as `1` or `0`, an int with `str`, a float with `repr`, and a name as it is. `fisheye_start` gives (f, f, width / 2, height / 2, 0.0, 0.0, 0.0, 0.0) with f = (L / 2) / (θ / 2).
    - Each builder gives `PROGRAM`, the command, its paths, the four log options and `--default_random_seed`, then the command's options in `options.md`'s order: each Fixed_Constant at its value, and each other option at its Option_Key's value, taking exactly its command's keys from `OPTION_KEYS`, each once. `--SequentialMatching.overlap` comes from `near_overlap` or `far_overlap`. `--ImageReader.camera_params` goes in only for `"OPENCV_FISHEYE"`, and `--ImageReader.camera_mask_path` only with `mask_used`. Both undistorters take the same eleven `undistortion_*` keys.
    - Every path is absolute. The Feature_Extraction reads `work/frames/grey`, and both mappers and the Undistortion read `work/frames/images`. `global_mapper` takes the database it's given, so the solve flow chooses the Calibration_Copy or the Database. Each run's name, Time_Limit and Time_Limit key, Command_Log, working folder and environment come from the command and `paths`.
    - No builder passes `--help`, `--project_path`, `--log_path`, a GPU index, an image list, a vocabulary tree, a `--SequentialMatching.loop_detection_*` setting, or an ALIKED, LoMa or LightGlue option.
    - _Requirements: 1.5, 2.2, 2.3, 2.4, 2.5, 2.6, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 4.1, 4.2, 4.3, 6.1, 6.2, 6.6, 8.1, 8.3_

  - [x] 4.4 Add `cameras_job` to `tests/conftest.py`
    - A `cameras_job` fixture gives `cameras_job(picks=..., mask_used=..., width=..., height=...)`, which makes a Job with `new_job` and writes a tiny valid `work/frames/` into it: `images/` and `grey/` with one small JPEG per Pick, `mask.png` when `mask_used` is true, and a `frames.json` with every field of the Frames_Record. It gives the Job_Id.
    - Update the module docstring to name it.
    - _Requirements: 16.5_

  - [x] 4.5 Write unit tests for the Frames_Contract in `tests/test_cameras_frames_input.py`
    - With no `frames.json`, the failure says the Frames_Step hasn't finished for this Job and gives the frames command with the Job's id.
    - Each Req 1.3 fault fails, naming its field: a record that isn't JSON, each of `pick_width`, `pick_height`, `mask_used` and `picks` missing, a bool where an int belongs, an empty `picks`, a Pick without its `number` or its `file`, a `file` that isn't `pick_name(number)`, and a `number` given twice.
    - Each missing file of Req 1.4 fails, naming the file: a Colour_Pick, a Grey_Copy, and `mask.png` with `mask_used` true. A missing `mask.png` with `mask_used` false passes.
    - Every message says to run the Frames_Step and holds no "Traceback". A good record gives the Picks in the record's order, and `numbers` gives each file's Pick_Number.
    - _Requirements: 1.2, 1.3, 1.4, 1.6, 16.2_

  - [x] 4.6 Write unit tests for the runner and the command lines in `tests/test_cameras_colmap.py`
    - Each builder's command line for the real config, read by path, holds its block's options in `options.md`'s order, each reading back as the same value. The differences are only the ones the design gives: `camera_model`, the A1 values, the Preset's `near_overlap` and `far_overlap`, the conditional options for each Camera_Model and each `mask_used`, and every path absolute and inside its folder.
    - `format_value` gives `1` and `0` for bools, never `True`, and `0.006666666666666667` and `1e-05` for those floats, and each text reads back as the same value. `fisheye_start(1600, 900, 120.0)` gives f = 800 / (π / 3), about 763.944, with cx 800 and cy 450.
    - `colmap_env` drops `TEST_TMPDIR` and the `GLOG_` keys and sets `TMPDIR` and `TMP`. `ColmapRun.value` gives the argument after an option.
    - `run_colmap` with `subprocess.run` replaced as `splatpipe.cameras.colmap` sees it, with `monkeypatch`: the call gets an argument list, `stdin=DEVNULL`, the log as stdout with stderr into it, the cwd, the env and the timeout. `TimeoutExpired` gives `exit_code=None`, `FileNotFoundError` gives the Req 2.7 message, and a log that can't be opened gives the Req 11.12 message.
    - A real child, `sys.executable -c` that prints a line and then sleeps, is stopped at a Time_Limit of 0.2 s, with its line in the Command_Log.
    - `last_error_line` on glog lines with 4-digit and 8-digit dates and with decimal and hex thread ids, on a crash's stack trace, and on an empty log. `exit_text` for 1 and -6.
    - _Requirements: 1.5, 2.2, 2.7, 2.8, 2.10, 3.1, 3.4, 4.1, 4.2, 4.3, 6.1, 6.2, 6.6, 8.1, 11.12, 16.2_

  - [x] 4.7 Write the property test for Property 1 in `tests/test_cameras_colmap.py`, with Hypothesis
    - **Property 1: Command lines hold the Fixed_Constants and every option once**
    - **Validates: Requirements 2.3, 2.4, 2.5, 2.6, 2.13, 3.2, 3.3, 3.5, 3.6, 4.1, 4.2, 4.3, 8.3**
    - Draw settings that `read_settings` accepts, either `mask_used`, either Camera_Model, and Pick_Sizes from 1 × 1 to 4000 × 4000. Each Option_Key's value comes from its Range, with the 32-bit ends of whole numbers and floats like 1e-300 and 1e300 where the range allows them, and the cross-checked pairs put in order.
    - The oracle is `colmap_options`. Each of the eight command lines starts with `colmap` and its command, holds each Fixed_Constant at its value, and holds every other option of its block exactly once, with text that reads back as its Option_Key's value: a bool as `1` or `0`, an int through `int()`, a float through `float()`, and a name as it is.
    - `--SequentialMatching.overlap` is `near_overlap` in the Near_Match_Run and `far_overlap` in the Far_Match_Run. `--ImageReader.camera_params` appears only for OPENCV_FISHEYE, and `--ImageReader.camera_mask_path` only with `mask_used`. Both undistorters give each undistortion setting the same value. No option appears twice, and none of the options of Req 2.6 appears.
    - _Requirements: 2.3, 2.4, 2.5, 2.6, 2.13, 3.2, 3.3, 3.5, 3.6, 4.1, 4.2, 4.3, 8.3, 16.3_

  - [x] 4.8 Write the `colmap <command> -h` test in `tests/test_cameras_colmap.py`
    - Skip it with the reason "COLMAP isn't installed" when `colmap` isn't on the PATH.
    - Run `colmap <command> -h` for each of the seven commands, with `timeout=60`. Check that every option the builders pass for the real config, with and without the Mask and for both Camera_Models, is in that command's list.
    - _Requirements: 16.7, 16.8, 16.13_

  - [x] 4.9 Run the tests and commit
    - Run `uv run pytest`. Every test passes, `tests/test_guard.py` included. The `-h` test runs where COLMAP is installed and is skipped, with its reason, where it isn't.
    - On `feat/camera-solve`: `git add splatpipe/cameras/frames_input.py splatpipe/cameras/colmap.py splatpipe/cameras/commands.py tests/conftest.py tests/test_cameras_frames_input.py tests/test_cameras_colmap.py`, then commit those files with the message "Cameras: frames contract, COLMAP runner and command lines".
    - _Requirements: 16.4, 16.13_

  - [x] 4.10 Switch the step's code to RADIAL and RADIAL_FISHEYE
    - Files: `splatpipe/cameras/settings.py`, `config.toml`, `splatpipe/cameras/commands.py`, `splatpipe/cameras/solve.py`, `splatpipe/cameras/database.py`, `splatpipe/cameras/undistort.py` and `splatpipe/cameras/model.py`, all changed.
    - Design: "settings.py", "commands.py", "database.py", "solve.py" and its Req 5.3 warning, "undistort.py", "model.py", "The `[cameras]` section of `config.toml`", "The settings checks", "The Fixed_Constants", F2, F9, F17, F18, O5, C8, D19 and D20.
    - `settings.py`: the Range of `camera_model` takes `"RADIAL"` or `"RADIAL_FISHEYE"`, and the comment on `CamerasSettings.camera_model` says so. `config.toml`: `camera_model = "RADIAL"`, with the design's comments on `camera_model` and `fisheye_fov_degrees`.
    - `commands.py`: the constant `RADIAL_FISHEYE` takes the place of `OPENCV_FISHEYE`, and `fisheye_start` gives (f, width / 2, height / 2, 0.0, 0.0), RADIAL_FISHEYE's order f, cx, cy, k1, k2. `--ImageReader.camera_params` goes in only for `"RADIAL_FISHEYE"`.
    - `solve.py`: the Calibration runs for `"RADIAL"` and is skipped for `"RADIAL_FISHEYE"`, whose Global_Run reads the Database, and the Req 5.3 warning names RADIAL_FISHEYE. `database.py`: `calibration_failure` compares f, `params[0]`, bit for bit (F17).
    - `undistort.py`: the refusal of a camera that isn't PINHOLE says that COLMAP keeps a camera's own model when its distortion came out as 0, and to run the step again, and no longer names the two refine keys, which the settings check keeps true (C8). `mask_camera_line` writes the Final_Model's RADIAL or RADIAL_FISHEYE camera.
    - `model.py`: `CAMERA_MODELS` drops OPENCV (4) and OPENCV_FISHEYE (5), so it holds PINHOLE, RADIAL and RADIAL_FISHEYE.
    - Every docstring and comment in these files names the new models.
    - _Requirements: 3.2, 3.3, 3.4, 5.1, 5.3, 5.4, 6.1, 6.2, 8.2, 8.3, 10.1, 12.3, 12.9, 12.10_

  - [x] 4.11 Switch `FakeColmap` in `tests/conftest.py` to RADIAL cameras
    - Files: `tests/conftest.py`, changed.
    - Design: the `FakeColmap` stand-in under "Stand-ins", F18 and D19.
    - `COLMAP_MODEL_IDS` gives RADIAL 3 and RADIAL_FISHEYE 9. The fake Feature_Extraction writes one RADIAL camera (f, cx, cy, 0, 0) with f = `--ImageReader.default_focal_length_factor` × the long edge and `prior_focal_length` 0, or a RADIAL_FISHEYE camera with `--ImageReader.camera_params` and 1. A Calibration that works changes f, `params[0]`.
    - The fake mappers' camera is RADIAL, with f = `MAPPED_FOCAL` × the long edge, the principal point at the centre and a little distortion, k1 and k2 not 0. The fake undistorter turns it into PINHOLE with fx = fy = f, 2 pixels smaller each way, and its flag that keeps the camera's model keeps it RADIAL.
    - The module docstring and the comments say RADIAL and RADIAL_FISHEYE.
    - _Requirements: 16.5_

  - [x] 4.12 Update the settings, model, command line and solve tests for RADIAL and RADIAL_FISHEYE
    - Files: `tests/test_cameras_settings.py`, `tests/test_cameras_model.py`, `tests/test_cameras_colmap.py` and `tests/test_cameras_solve.py`, changed.
    - Design: the rows of these four files in "Tests by file", F2, F17 and D19.
    - `tests/test_cameras_settings.py`: the real config holds `camera_model = "RADIAL"`, the Range written out in the test takes `"RADIAL"` and `"RADIAL_FISHEYE"`, and `"OPENCV"` is refused as an unknown name, naming `camera_model`, `[cameras]` and `--ImageReader.camera_model`.
    - `tests/test_cameras_model.py`: OPENCV's id 4 and OPENCV_FISHEYE's id 5 are each refused as an unknown model id, naming the file.
    - `tests/test_cameras_colmap.py`: `fisheye_start(1600, 900, 120.0)` gives (f, 800, 450, 0, 0) with f = 800 / (π / 3), about 763.944, and its text reads back as the same five floats. The real config's `feature_extractor` line has `--ImageReader.camera_model RADIAL` and no `--ImageReader.camera_params`, and with `"RADIAL_FISHEYE"` it has the five numbers. The `-h` test of 4.8 builds its lines for RADIAL and RADIAL_FISHEYE.
    - `tests/test_cameras_solve.py`: the run orders for RADIAL and for RADIAL_FISHEYE, the Req 5.3 warning naming RADIAL_FISHEYE, and `calibration_failure` on SQLite files with RADIAL cameras whose f is the same bit for bit, rejected, or changed, worked.
    - _Requirements: 3.2, 3.3, 3.4, 5.1, 5.3, 5.4, 6.1, 6.2, 10.1, 10.4, 12.3, 12.10, 16.2, 16.5, 16.7_

  - [x] 4.13 Update the step, undistortion and real COLMAP tests for RADIAL and RADIAL_FISHEYE
    - Files: `tests/test_cameras_step.py`, `tests/test_cameras_undistort.py` and `tests/test_cameras_real.py`, changed.
    - Design: the rows of these three files in "Tests by file", "The real COLMAP tests", F9, F18, C8 and D19.
    - `tests/test_cameras_step.py`: `solve.json` says `"RADIAL"`, the Mask_Warp's camera line is the Final_Model's RADIAL camera, and the Levelled_Model's PINHOLE camera has fx = fy. The test of an undistorter that kept the camera's model (C8) checks that the message says COLMAP keeps a camera's own model when its distortion came out as 0 and to run the step again, and names neither refine key.
    - `tests/test_cameras_undistort.py`: `check_undistorted` refuses a RADIAL camera that COLMAP kept, with C8's words and no word of the two refine keys. `mask_camera_line` for a RADIAL and a RADIAL_FISHEYE camera reads back to the same camera with `float()`.
    - `tests/test_cameras_real.py`: the painted corner is taken through the Undistorted_Camera and the Final_Model's RADIAL camera, whose distortion is k1 and k2 on the radius, with one f, so the test no longer asserts OPENCV. The rest of the integration test stays until 8.6.
    - _Requirements: 8.2, 8.3, 8.4, 11.7, 11.9, 16.2, 16.6_

  - [x] 4.14 Update the property test for Property 1 in `tests/test_cameras_colmap.py` for RADIAL and RADIAL_FISHEYE
    - **Property 1: Command lines hold the Fixed_Constants and every option once**
    - **Validates: Requirements 2.3, 2.4, 2.5, 2.6, 2.13, 3.2, 3.3, 3.5, 3.6, 4.1, 4.2, 4.3, 8.3**
    - Files: `tests/test_cameras_colmap.py`, changed.
    - Design: Property 1 and its strategy, and D19.
    - `camera_model` is drawn from `"RADIAL"` and `"RADIAL_FISHEYE"`. `--ImageReader.camera_params` appears only for RADIAL_FISHEYE, with five numbers that read back as the Fisheye_Start that the oracle works out itself, and never for RADIAL. The two refine keys stay drawn only as true, as 1.9 left them.
    - _Requirements: 2.13, 3.2, 3.3, 3.4, 16.3_

  - [x] 4.15 Write the unit test of `check_undistorted`'s other-camera refusal in `tests/test_cameras_undistort.py`
    - Files: `tests/test_cameras_undistort.py`, changed.
    - Design: "undistort.py" and the `tests/test_cameras_undistort.py` row of "Tests by file", which lists this refusal (small fix d).
    - An Undistorted_Model with one PINHOLE camera and an image whose camera id isn't that camera's is refused with an `ExpectedFailure` that names the image, its camera id and the one camera's id, says what the Undistortion got wrong and to run the step again, and holds no "Traceback". The same model with every image on the one camera passes.
    - _Requirements: 8.2, 16.9, 16.15_

  - [x] 4.16 Run the tests and commit
    - Run `uv run pytest`. Every test passes, `tests/test_guard.py` included. Where Homebrew's COLMAP is installed, the `-h` test and the integration test run with the RADIAL camera (D18).
    - On `feat/camera-solve`: `git add splatpipe/cameras/settings.py splatpipe/cameras/commands.py splatpipe/cameras/solve.py splatpipe/cameras/database.py splatpipe/cameras/undistort.py splatpipe/cameras/model.py config.toml tests/conftest.py tests/test_cameras_settings.py tests/test_cameras_model.py tests/test_cameras_colmap.py tests/test_cameras_solve.py tests/test_cameras_step.py tests/test_cameras_undistort.py tests/test_cameras_real.py`, then `git commit -m "Cameras: one shared RADIAL camera, and RADIAL_FISHEYE for wide lenses" -- splatpipe/cameras/settings.py splatpipe/cameras/commands.py splatpipe/cameras/solve.py splatpipe/cameras/database.py splatpipe/cameras/undistort.py splatpipe/cameras/model.py config.toml tests/conftest.py tests/test_cameras_settings.py tests/test_cameras_model.py tests/test_cameras_colmap.py tests/test_cameras_solve.py tests/test_cameras_step.py tests/test_cameras_undistort.py tests/test_cameras_real.py`.
    - _Requirements: 16.4, 16.13_

- [x] 5. The database and the solve flow
  - Files: `splatpipe/cameras/database.py`, `splatpipe/cameras/solve.py` and `tests/test_cameras_solve.py` (new). `tests/conftest.py` (changed).
  - Design: "database.py", "solve.py", "The solve flow" with its state table, the `FakeColmap` stand-in under "Stand-ins", F10, F11, F12, O5, O9, C1, C7, C11 and Property 2.
  - COLMAP on the laptop: not needed. `FakeColmap` stands in for it.
  - The update for D17 adds 5.7 to 5.15, which name their own files, design sections and requirements.
  - _Requirements: 2.8, 2.9, 2.11, 2.12, 5.1, 5.2, 5.3, 5.4, 5.5, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8, 6.9, 6.10, 6.11, 6.12, 7.1, 7.2, 7.3, 11.12, 13.1, 13.2, 13.3, 16.1, 16.2, 16.3, 16.4, 16.5, 16.9, 16.13_

  - [x] 5.1 Write `splatpipe/cameras/database.py`
    - `DatabaseCamera`, `copy_database()`, `read_camera()` and `calibration_failure()`, as in the design's "database.py" section, with Python's built-in `sqlite3`.
    - `copy_database` copies with `sqlite3`'s backup API (C1). A `sqlite3.Error` or an `OSError` is the Req 11.12 failure naming the Calibration_Copy.
    - `read_camera` reads the `cameras` table, with `params` as little-endian doubles. Anything but exactly one row, or a database that can't be read, fails, naming the file and saying to check that COLMAP 4.2.1 is installed and to run the step again.
    - `calibration_failure` is pure. It gives "found no image pairs to calibrate" when the copy's `prior_focal_length` is still false, "rejected its focal length estimate and kept the starting one" when fx and fy are bit for bit the same as before, and `None` when the Calibration worked (O5).
    - _Requirements: 5.2, 5.4, 11.12_

  - [x] 5.2 Write `splatpipe/cameras/solve.py`
    - `Report`, `ModelCount`, `Solved`, `run_timed()`, `run_or_fail()`, `kept_model()`, `placed_needed()`, `model_counts()` and `solve()`, as in the design's "solve.py" section and its state table.
    - `run_timed` adds the run's timing under its name as soon as the run returns, whatever the result. It fails a Timeout with the Req 2.8 message, which names the command, the limit and its key, and runs no Fallback. `run_or_fail` also fails a non-zero exit with the Req 2.9 message, which gives the exit text, the Last_Error_Line and the Command_Log.
    - `kept_model` gives the Model with the most Placed_Frames and the lowest number among those with as many, or `None`. `placed_needed` gives max(`min_placed_frames`, ⌈`Fraction(min_placed_share)` × picks⌉), worked out exactly (C7). `model_counts` lists a mapper's numbered Model folders in number order and counts the images whose names are Picks' files (C11).
    - `solve` follows the state table. It runs the Feature_Extraction, then the Near_Match_Run and the Far_Match_Run inside a `matching` block. Inside a `mapping` block, it makes the Calibration_Copy and runs the Calibration for `"OPENCV"`, or adds the Req 5.3 warning for `"OPENCV_FISHEYE"`. It runs the Global_Run on the Calibration_Copy, or on the Database for the fisheye camera (O9). After a Failed_Calibration, or when the Global_Run gives up, it adds the Req 5.5 or 6.7 warning with its reason and runs the one Fallback on the Database. Both blocks are timed like `StepContext.timed`, so each stage timing is added only when its block finishes.
    - It makes `colmap/global/` and `colmap/incremental/` before the mapper that writes into each. The Incremental_Run's non-zero exit fails with the Req 6.10 message and the Filming_Advice. An Incremental_Run with no Model counts 0 Placed_Frames. A Final_Model under `placed_needed` fails with the Placed_Failure before any Undistortion, giving the placed Picks, all the Picks, the number needed and the Filming_Advice.
    - It adds `fallback_ran`, `kept_from_global`, `models_found`, `placed_frames` and `points3d` as ints, and gives back the Final_Model, read whole with `read_model`. Each COLMAP_Command runs at most once, after the one before has returned, and nothing is retried.
    - _Requirements: 2.8, 2.9, 2.11, 2.12, 5.1, 5.2, 5.3, 5.4, 5.5, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8, 6.9, 6.10, 6.11, 7.1, 7.2, 7.3, 13.1, 13.2, 13.3_

  - [x] 5.3 Add `FakeColmap` and `fake_colmap` to `tests/conftest.py`
    - `FakeColmap`, as in the design's "Stand-ins", stands in for the `colmap` program for the commands the solve flow runs. Its `run(colmap_run)` records each run in `runs` and writes the run's Command_Log. Then it follows its script for the run's name: a name in `timeouts` gives `RunResult(None, ...)`, a name in `exits`, mapped to `(code, line)`, writes the line to the Command_Log and gives that code, and every other run exits with 0 after writing what the real command would write.
    - The Feature_Extraction writes a SQLite `database.db` whose `cameras` table holds one camera of the Camera_Model at the Pick_Size: f = 1.2 × the long edge with `prior_focal_length` 0, or the Fisheye_Start with 1. The Match_Runs write nothing. The Calibration changes the Calibration_Copy as `calibration` says: `"worked"`, `"rejected"` or `"no_pairs"`.
    - `global_mapper` and `mapper` write one Model folder per count in `global_models` or `incremental_models`, with task 2's Model_Writer, and `None` writes no Model. Model k places the next that many Picks of `--image_path`, with Poses along a walk seen through a fixed rigid move, an OPENCV camera with a little distortion, and a few points.
    - The `fake_colmap` fixture gives the class, so a test builds one with its script and passes its `run` to `solve`. Update the module docstring to name both.
    - _Requirements: 16.5_

  - [x] 5.4 Write unit tests for the database and the solve flow in `tests/test_cameras_solve.py`
    - Each test runs `solve` with a `FakeColmap` and a real `StepContext` on a `cameras_job` Job.
    - For OPENCV, the runs are the extractor, the near match, the far match, the Calibration and the Global_Run, in that order. For OPENCV_FISHEYE, there's no Calibration, the Req 5.3 warning names `fisheye_fov_degrees`, and the Global_Run reads the Database. The Calibration_Copy appears only after the Far_Match_Run has returned.
    - The Calibration reads the Calibration_Copy, and the Database's camera row is unchanged afterwards. A Calibration that worked sends the Global_Run to the copy.
    - Each kind of Failed_Calibration (exit 1, exit -6, rejected, no pairs) runs no Global_Run and runs the mapper once, on the Database, with the Req 5.5 warning giving the reason.
    - Each way the Global_Run gives up (exit 1, exit -6, no Model, a Kept_Model under the limit) runs the mapper once, on the Database, with the Req 6.7 warning giving its reason. The Final_Model comes from the run that made it, with `fallback_ran`, `kept_from_global` and `models_found` to match.
    - A Timeout of each command fails, naming its Time_Limit key, with no Fallback. A non-zero exit of the extractor and of each Match_Run fails with the exit code and the Last_Error_Line. The mapper's non-zero exit fails with the Req 6.10 message, and its exit 0 with no Model gives the Placed_Failure with 0 placed.
    - Each failed run's report fields hold the counts and timings worked out so far, the failing command's timing included. Every command runs at most once. Each message names what its requirement says and holds no "Traceback".
    - `placed_needed` and the Placed_Check at the edges: exactly `min_placed_frames` and one fewer, exactly `min_placed_share` of the Picks and one fewer, and 0.3 of 10 Picks needing 3 (C7). `model_counts` ignores folders whose names aren't numbers and images that aren't Picks.
    - `copy_database`, `read_camera` and `calibration_failure` work on small SQLite files the test writes, and `read_camera` refuses a table with two camera rows.
    - _Requirements: 2.8, 2.9, 2.11, 2.12, 5.1, 5.2, 5.3, 5.4, 5.5, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.9, 6.10, 6.11, 7.1, 7.2, 7.3, 13.1, 13.2, 13.3, 16.1, 16.2, 16.5, 16.9_

  - [x] 5.5 Write the property test for Property 2 in `tests/test_cameras_solve.py`, with Hypothesis
    - **Property 2: The Kept_Model has the most Placed_Frames and the lowest number among ties**
    - **Validates: Requirements 6.8, 6.12**
    - Draw 1 to 20 Models with unique numbers from 0 to 60 and counts from 0 to 4, so ties are common.
    - The oracle, a plain loop, checks that the kept Model has at least as many Placed_Frames as every other Model, and a lower number than every other Model with as many.
    - _Requirements: 6.8, 6.12, 16.3_

  - [x] 5.6 Run the tests and commit
    - Run `uv run pytest`. Every test passes, `tests/test_guard.py` included.
    - On `feat/camera-solve`: `git add splatpipe/cameras/database.py splatpipe/cameras/solve.py tests/conftest.py tests/test_cameras_solve.py`, then commit those files with the message "Cameras: the database and the solve flow".
    - _Requirements: 16.4, 16.13_

  - [x] 5.7 Add `Deadline`, `NO_DEADLINE`, `capped()` and `capped_at` to `splatpipe/cameras/colmap.py`
    - Files: `splatpipe/cameras/colmap.py`, changed.
    - Design: "colmap.py" (`ColmapRun.capped_at`, `Deadline`, `NO_DEADLINE` and `capped`), C13, C14 and D17.
    - `ColmapRun` gains `capped_at: datetime | None = None`, the Cap_Deadline when it set `timeout_seconds`. `Deadline` holds the Cap_Deadline `at`, or None, and the clock `now`. `NO_DEADLINE` has no Cap_Deadline.
    - `capped(colmap_run, deadline, now)` is pure. With no Cap_Deadline it gives the run as it is. Otherwise, with `left = (deadline − now).total_seconds()`: at least the run's `timeout_seconds` left gives the run as it is; above 0 and below that gives `dataclasses.replace(colmap_run, timeout_seconds=left, capped_at=deadline)`; 0 or less raises the Req 17.5 `ExpectedFailure`, the design's message, which names the command and the Cap_Deadline in ISO 8601 UTC and says to give the job a larger `cap_minutes` for its preset under `[cloud.presets]` in `config.toml`, then run it again.
    - The module still imports only the standard library and `splatpipe.errors`.
    - _Requirements: 17.3, 17.4, 17.5_

  - [x] 5.8 Cap every COLMAP run in `run_timed` in `splatpipe/cameras/solve.py`
    - Files: `splatpipe/cameras/solve.py`, changed.
    - Design: "solve.py" (`run_timed`, `run_or_fail` and `solve` with their `deadline`), the note under "The solve flow"'s diagram, C13 to C15, and the Cap_Deadline rows of "Error Handling".
    - `run_timed`, `run_or_fail` and `solve` take `deadline: Deadline = NO_DEADLINE`, and `solve` passes its own to every run it makes, through its `_Flow`. `run_timed` first calls `capped(colmap_run, deadline.at, deadline.now())`, so with no time left it raises before the runner is called, and the command gets no timing and no Command_Log.
    - Then it runs the command and adds its timing, and fails a Timeout with the Req 2.8 message when `capped_at` is None, and otherwise with the Req 17.6 message: the command, the Cap_Deadline, the seconds that were left, rounded to a whole second, and the larger `cap_minutes` advice, with no `[cameras]` key. Neither failure starts the Fallback.
    - _Requirements: 2.8, 17.3, 17.4, 17.5, 17.6_

  - [x] 5.9 Read the Cap_Deadline in `work` in `splatpipe/cameras/__init__.py`
    - Files: `splatpipe/cameras/__init__.py`, changed.
    - Design: "The step's flow (`__init__.py`)" (`Tools.now`, `TOOLS` and steps 2, 6 and 7), the `__init__.py` row of "Module layout", C12, C16 and D17.
    - `Tools` gains `now: Callable[[], datetime] = utc_now`, with `utc_now` from `splatpipe.train.deadline`, and `TOOLS` passes `now=utc_now`.
    - Right after `read_settings` and before `read_frames`, `work` reads `deadline = Deadline(read_cap_deadline(os.environ), tools.now)` once, so a bad `SPLATPIPE_CAP_DEADLINE` fails with `read_cap_deadline`'s own message before the Frames_Contract and before any COLMAP run. It passes `deadline` to `solve`, and to the Undistortion's and the Mask_Warp's `run_or_fail` through `_Flow`.
    - The module docstring says the step reads the Cap_Deadline and caps each command's Time_Limit at the time left before it. `splatpipe.train.deadline` is the package's only new import.
    - _Requirements: 14.10, 17.1, 17.2, 17.3, 17.4_

  - [x] 5.10 Give `FakeColmap` a clock in `tests/conftest.py`
    - Files: `tests/conftest.py`, changed.
    - Design: the `FakeColmap` stand-in under "Stand-ins", and C16.
    - `FakeColmap` takes an optional `now`, a stand-in clock, and records in `runs` each run's `timeout_seconds` and `capped_at` as the run got them. `fake_colmap` installs it as `Tools.now`, or `utc_now` when the test gives none, beside `find_colmap` and `run`.
    - The module docstring says so, and that the autouse `no_cap_deadline` keeps every other test uncapped.
    - _Requirements: 16.5_

  - [x] 5.11 Write unit tests for `capped` in `tests/test_cameras_colmap.py`
    - Files: `tests/test_cameras_colmap.py`, changed.
    - Design: the `capped` items of the `tests/test_cameras_colmap.py` row of "Tests by file", and C13 to C15.
    - For a run whose key gives 600 s: no Cap_Deadline gives the run as it is; exactly 600 s left gives it as it is; 599 s left gives 599 s with `capped_at` set; a microsecond left gives a microsecond; exactly 0 s left and a minute past each raise the Req 17.5 failure, which names the command and the Cap_Deadline, says to raise `cap_minutes` under `[cloud.presets]` and holds no "Traceback". Every other field of the run stays as it was.
    - _Requirements: 16.2, 16.9, 17.3, 17.4, 17.5_

  - [x] 5.12 Write the property test for Property 8 in `tests/test_cameras_colmap.py`, with Hypothesis
    - **Property 8: The Cap_Deadline caps each Time_Limit, and no time left starts nothing**
    - **Validates: Requirements 17.3, 17.4, 17.5, 17.8**
    - Files: `tests/test_cameras_colmap.py`, changed.
    - Design: Property 8 and its strategy.
    - Draw key values from 1e-6 to 1e6 seconds, a time now from 2000 to 2100 in UTC with microseconds, and a Cap_Deadline of None or now plus an offset from −2e6 to 2e6 seconds in microsecond steps, with offsets equal to the key's value and to 0 drawn often.
    - The oracle works out the seconds left with its own `timedelta` arithmetic and checks the three cases: the run unchanged; the run with the seconds left as its `timeout_seconds`, `capped_at` set and every other field unchanged; or the `ExpectedFailure` that names the command and the Cap_Deadline.
    - _Requirements: 16.3, 17.3, 17.4, 17.5, 17.8_

  - [x] 5.13 Write unit tests for the capped solve in `tests/test_cameras_solve.py`
    - Files: `tests/test_cameras_solve.py`, changed.
    - Design: the Cap_Deadline items of the `tests/test_cameras_solve.py` row of "Tests by file", "The solve flow", and C13 to C15.
    - With a `Deadline` and a stand-in clock that moves on with each run: every run's `timeout_seconds` is the smaller of its key's value and the seconds left, with `capped_at` set only when the Cap_Deadline gave the limit. A Timeout under a capped limit fails with the Req 17.6 message, which names the command, the Cap_Deadline, the seconds left and `cap_minutes` and no `[cameras]` key, with no Fallback, while a Timeout under a key's limit still gives the Req 2.8 message.
    - A clock that passes the Cap_Deadline before the Global_Run runs nothing more and gives the Req 17.5 message. The report holds the earlier commands' timings and none for the Global_Run, and no Fallback runs.
    - With no `Deadline`, every run gets its key's value, as before.
    - _Requirements: 2.8, 13.3, 16.5, 16.9, 17.3, 17.4, 17.5, 17.6_

  - [x] 5.14 Write unit tests for the step's read of the Cap_Deadline in `tests/test_cameras_step.py`
    - Files: `tests/test_cameras_step.py`, changed.
    - Design: the Cap_Deadline items of the `tests/test_cameras_step.py` row of "Tests by file", C12 and C16.
    - Through `splatpipe.cameras.run` with `FakeColmap`: each `SPLATPIPE_CAP_DEADLINE` that `read_cap_deadline` refuses, set with `monkeypatch.setenv` (an empty value, `"soon"`, and a time with no UTC offset), fails the step with that function's own message and `expected: true`, before `frames.json` is read and before any run, also when `work/frames/` is broken. A bad setting still fails first.
    - With a good one and `fake_colmap(now=...)`, the runs get capped limits and the report is done. With none, every run gets its key's value.
    - _Requirements: 14.10, 16.5, 16.9, 17.1, 17.2, 17.3, 17.4_

  - [x] 5.15 Run the tests and commit
    - Run `uv run pytest`. Every test passes, `tests/test_guard.py` and the cameras import check in `tests/test_imports.py` included. The settings test of 1.4 still finds the seven Time_Limit keys at O1's values.
    - On `feat/camera-solve`: `git add splatpipe/cameras/colmap.py splatpipe/cameras/solve.py splatpipe/cameras/__init__.py tests/conftest.py tests/test_cameras_colmap.py tests/test_cameras_solve.py tests/test_cameras_step.py`, then `git commit -m "Cameras: cap each COLMAP time limit at the cap deadline" -- splatpipe/cameras/colmap.py splatpipe/cameras/solve.py splatpipe/cameras/__init__.py tests/conftest.py tests/test_cameras_colmap.py tests/test_cameras_solve.py tests/test_cameras_step.py`.
    - _Requirements: 16.4, 16.13, 17.7_

- [x] 6. Undistortion and the mask
  - Files: `splatpipe/cameras/undistort.py` and `tests/test_cameras_undistort.py`, both new.
  - Design: "undistort.py", F7, F8, F9, C8 and Property 3.
  - COLMAP on the laptop: not needed. The tests build their own models and PNGs.
  - _Requirements: 8.2, 8.3, 8.4, 8.5, 8.7, 12.14, 16.2, 16.3, 16.4, 16.13_

  - [x] 6.1 Write `splatpipe/cameras/undistort.py`
    - `check_undistorted()`, `mask_camera_line()`, `undistorted_mask()` and `read_warped_mask()`, as in the design's "undistort.py" section. cv2 only decodes the warped mask's PNG.
    - `check_undistorted` gives the Undistorted_Camera after checking that the Undistorted_Model has exactly one camera, that it's PINHOLE, that every image uses it, that its image names are exactly the Placed_Frames' names, and that each image file exists. Each refusal says what the Undistortion got wrong and to run the step again. A camera that isn't PINHOLE also gives the `ba_refine_extra_params` advice of C8.
    - `mask_camera_line` gives one line: `mask.png`, the model name, the width, the height and each parameter as `repr(float)` (F8).
    - `undistorted_mask` is pure. From a `uint8` image of 2 dimensions, or of 3 with 1, 3 or 4 channels, it gives one `uint8` channel with 255 where every channel is 255 and 0 everywhere else.
    - `read_warped_mask` reads the file with `Path.read_bytes` and `cv2.imdecode(..., cv2.IMREAD_UNCHANGED)`. A file that can't be read or decoded, isn't 8-bit, or isn't the Undistorted_Camera's width and height fails, naming the file and what's wrong and saying to run the step again. Otherwise it gives `undistorted_mask` of the image.
    - _Requirements: 8.2, 8.3, 8.4, 8.5, 12.14_

  - [x] 6.2 Write unit tests for the undistortion checks and the mask in `tests/test_cameras_undistort.py`
    - `check_undistorted` refuses, on models the test builds: two cameras, an OPENCV camera, whose message names `mapper_ba_refine_extra_params` and `global_mapper_ba_refine_extra_params`, an image that isn't a Placed_Frame, a Placed_Frame missing from the model, and a missing image file. A good model gives its PINHOLE camera.
    - `mask_camera_line` for an OPENCV and an OPENCV_FISHEYE camera reads back, split on spaces, to the same name, model, size and parameters, each parameter through `float()`.
    - `read_warped_mask` fails on a missing file, bytes that aren't an image, a 16-bit PNG and a PNG of the wrong size, each naming the file, with no "Traceback". A three-channel PNG with grey edges and a one-channel PNG each give a mask of only 0 and 255.
    - `undistorted_mask` on 2-D, one-channel, three-channel and four-channel inputs.
    - _Requirements: 8.2, 8.3, 8.4, 8.5, 16.2_

  - [x] 6.3 Write the property test for Property 3 in `tests/test_cameras_undistort.py`, with Hypothesis
    - **Property 3: The Undistorted_Mask is 255 exactly where every channel is 255**
    - **Validates: Requirements 8.4, 8.7**
    - Draw `uint8` images of 1 × 1 to 16 × 16 pixels with 3 channels, with channel values drawn mostly from 0, 1, 254 and 255, so pixels with one channel just below 255 are common.
    - The oracle checks with its own loop that the result is one `uint8` channel of the same height and width, holds only 0 and 255, and is 255 at exactly the pixels whose three channels are all 255.
    - _Requirements: 8.4, 8.7, 16.3_

  - [x] 6.4 Run the tests and commit
    - Run `uv run pytest`. Every test passes, `tests/test_guard.py` included.
    - On `feat/camera-solve`: `git add splatpipe/cameras/undistort.py tests/test_cameras_undistort.py`, then commit those files with the message "Cameras: undistortion and the mask".
    - _Requirements: 16.4, 16.13_

- [x] 7. The outputs, the step and its CLI
  - Files: `splatpipe/cameras/outputs.py`, `splatpipe/cameras/__main__.py` and `tests/test_cameras_step.py` (new). `splatpipe/cameras/__init__.py` and `tests/conftest.py` (changed).
  - Design: "The step's flow (`__init__.py`)", "outputs.py", "The Work_Folder", "`work/cameras/solve.json` (the Solve_Record)", "`out/cameras.json` (the Viewer_Cameras)", "The Cameras_Report", "Error Handling", the undistorters of the `FakeColmap` stand-in, F15, C4, C6 and C10.
  - COLMAP on the laptop: not needed. `FakeColmap` stands in for it.
  - _Requirements: 1.1, 1.7, 2.7, 2.9, 7.2, 8.1, 8.2, 8.3, 8.4, 8.5, 9.1, 9.3, 9.7, 9.10, 9.12, 9.13, 9.14, 10.5, 10.6, 10.7, 10.8, 11.1, 11.2, 11.3, 11.4, 11.5, 11.6, 11.7, 11.8, 11.9, 11.10, 11.11, 11.12, 11.13, 12.8, 12.14, 13.1, 13.2, 13.3, 13.4, 13.5, 13.6, 14.1, 14.2, 14.10, 16.2, 16.4, 16.5, 16.9, 16.11, 16.13_

  - [x] 7.1 Write `splatpipe/cameras/outputs.py`
    - The Work_Folder's names, `copy_training_images()`, `write_training_masks()`, `mean_reprojection_error()`, `solve_record()`, `viewer_cameras()` and `write_record()`, as in the design's "outputs.py" section.
    - `copy_training_images` copies each name from the Undistortion's `images/` with `shutil.copyfile` (C4). `write_training_masks` encodes the Undistorted_Mask once with `cv2.imencode(".png", ...)` and writes the same bytes as `<stem>.png` for each name. A failed encode or an `OSError` is the Req 11.12 failure naming the file.
    - `mean_reprojection_error` averages the errors that aren't -1, or gives `None` when there are none (F15). `solve_record` and `viewer_cameras` are pure and give only plain JSON values, with the fields of the design's two tables. `viewer_cameras` lists the Placed_Frames in Pick_Number order and holds no points.
    - `write_record` writes with `splatpipe.jsonio.write_json`. On an `OSError` it removes the partly written file, then raises the Req 11.12 failure (C6).
    - _Requirements: 11.2, 11.3, 11.6, 11.7, 11.9, 11.12, 12.14, 13.4_

  - [x] 7.2 Write the step in `splatpipe/cameras/__init__.py`
    - `Tools`, `TOOLS`, `run()` and `work()`, as in the design's "The step's flow" section. `run` calls `run_step("cameras", "splatpipe.cameras", ...)`, like the Frames_Step. `work` reads `TOOLS` when it's called, so a test's `monkeypatch` reaches it through `run_step` and the CLI.
    - `work` runs the design's nine parts in order. It removes an earlier `out/cameras.json` before anything else. It reads the settings, then the Frames_Contract, adding `picks`. It fails with the Req 2.7 message when `find_colmap` gives `None`. It makes `colmap/`, `colmap/tmp/` and `colmap/logs/` and the environment, then runs `solve`.
    - Inside `step.timed("undistortion")`, it runs the Undistortion through `run_or_fail` and checks its output with `check_undistorted`. With `mask_used`, it writes the camera line of the Final_Model's camera, as the Final_Model holds it before the Undistortion, runs the Mask_Warp, and reads the Undistorted_Mask.
    - Inside `step.timed("levelling")`, it works out the Level_Move, adds the warning when the COLMAP_Frame is kept, and adds `levelled`. When the Levelling keeps the COLMAP_Frame, the Levelled_Model is the Undistorted_Model unchanged, with no identity move applied (C10).
    - Inside `step.timed("writing")`, it writes the Training_Images, the Training_Masks with `mask_used`, the Levelled_Model, `solve.json` and `out/cameras.json`, in that order. When writing `out/cameras.json` fails, it removes `solve.json` too before it fails (C6).
    - Each count is added with `int(...)` as soon as it's known. The step writes only inside `work/cameras/` and to `out/cameras.json`, leaves `work/frames/` as it is, and relies on the Step_Runner emptying `work/cameras/` before it runs.
    - _Requirements: 1.1, 1.7, 2.7, 2.9, 7.2, 8.1, 8.2, 8.3, 8.4, 8.5, 9.1, 9.3, 9.7, 9.10, 9.12, 9.13, 9.14, 10.5, 11.1, 11.2, 11.3, 11.4, 11.5, 11.6, 11.8, 11.10, 11.11, 11.12, 11.13, 12.8, 13.1, 13.2, 13.4, 13.5, 13.6, 14.1, 14.10_

  - [x] 7.3 Write `splatpipe/cameras/__main__.py`
    - `main(argv)` calls `step_main("cameras", "splatpipe.cameras", argv)`, and the file ends with `if __name__ == "__main__": sys.exit(main())`.
    - _Requirements: 14.1, 14.2_

  - [x] 7.4 Give `FakeColmap` the two undistorters, and install it as the step's tools
    - The fake `image_undistorter` reads `--input_path` with task 2's Model_Reader. It writes `undistorted/sparse/` with the camera turned PINHOLE and 2 pixels smaller each way, and a tiny JPEG per image in `undistorted/images/`. Flags make it keep the camera's model, add a second camera, or leave out an image.
    - The fake `image_undistorter_standalone` reads `mask_camera.txt` and writes `mask_undistorted/mask.png` as a three-channel PNG at the size the fake undistorter gives, with a grey edge around every black region, like bilinear sampling.
    - A flag makes the fake mappers write every other camera upside down, so the Levelling keeps the COLMAP_Frame.
    - `fake_colmap` now also installs each `FakeColmap` it builds with `monkeypatch.setattr("splatpipe.cameras.TOOLS", ...)`, with `find_colmap` giving a fake path, or `None` when it's built with `installed=False`. The step reads `TOOLS` each time it runs, so a run through `splatpipe.cameras.run`, `run_step` or the CLI in this process uses the stand-in.
    - Update the module docstring.
    - _Requirements: 16.5_

  - [x] 7.5 Write unit tests for a done run in `tests/test_cameras_step.py`
    - Each test runs the step through `splatpipe.cameras.run` with `FakeColmap` on a `cameras_job` Job, so the Step_Runner and the real report are part of each test.
    - The Work_Folder holds exactly `colmap/`, `images/`, `sparse/` and `solve.json`, plus `masks/` with `mask_used`. `images/` holds exactly the names in `sparse/images.bin`. `masks/` holds one PNG per Placed_Frame, at the Undistorted_Camera's size, with only 0 and 255. `colmap/logs/` holds a Command_Log for each run.
    - `sparse/` holds exactly the three Model_Files. They read back as one PINHOLE camera that every image uses, the Placed_Frames with the Undistorted_Model's image ids, camera id, names and 2D points, and every point with its id, colour, error and track.
    - With the fake mappers' upside-down flag, the Levelling keeps the COLMAP_Frame. Then `sparse/`'s Poses and points equal the Undistorted_Model's bit for bit (C10), the report holds the Req 9.3 warning, `levelled` is 0, and `solve.json` gives the reason.
    - The Mask_Warp's camera line is the Final_Model's camera before the Undistortion, and its command line holds the Undistortion's undistortion settings.
    - `solve.json` and `out/cameras.json` pass `check_json_value` and hold every field of the design's two tables. `out/cameras.json` lists the Placed_Frames in Pick_Number order with the Poses in `sparse/images.bin`, and holds no points. `mean_reprojection_error` skips errors of -1 and gives `None` when there are none.
    - The report holds every count of Req 13.1 as an int, every command timing and stage timing of Req 13.2, the warnings that apply, and in `settings` every key of `[cameras]` and the Preset's `near_overlap`.
    - After the run, nothing new is in `out/` but `cameras.json` and the report, and `work/frames/` is unchanged, by hash. A second run on the same Job leaves only its own files.
    - _Requirements: 1.7, 8.3, 8.4, 9.13, 9.14, 10.5, 10.6, 10.7, 10.8, 11.1, 11.2, 11.3, 11.4, 11.5, 11.7, 11.8, 11.9, 11.11, 11.13, 13.1, 13.2, 13.4, 13.5, 13.6, 16.2, 16.5_

  - [x] 7.6 Write unit tests for failed runs and the CLI in `tests/test_cameras_step.py`
    - An earlier `out/cameras.json` is removed when the step starts, also when the step then fails on a setting.
    - A bad setting fails the step before the step reads `frames.json`, also when `work/frames/` is broken, and before `FakeColmap` records any run. A broken Frames_Contract fails before any run, and so does a missing `colmap`, with `installed=False`.
    - A failure in the Placed_Check, in the Undistortion (each flag of the fake undistorter), and in writing a Training_Image (with a write patched to raise `OSError(errno.ENOSPC, "No space left on device")`) leaves no `solve.json` and no `out/cameras.json`. The Placed_Check's failure runs no undistorter.
    - A failed `out/cameras.json` write removes `solve.json` too, and its Readable_Error names the file and the reason.
    - A non-zero exit of the Undistortion and of the Mask_Warp each fails with the Req 2.9 message.
    - Every failed report has `expected: true` and a `readable` that names what its requirement says and holds no "Traceback".
    - `splatpipe.cameras.__main__.main(["--job", job_id])` prints `cameras: done` and returns 0 for a done run, and prints the Readable_Error on stderr and returns 1 for a failed one.
    - _Requirements: 1.1, 2.7, 2.9, 7.2, 8.2, 11.6, 11.10, 11.12, 12.8, 13.3, 14.2, 14.10, 16.5, 16.9, 16.11_

  - [x] 7.7 Run the tests and commit
    - Run `uv run pytest`. Every test passes, `tests/test_guard.py` included.
    - On `feat/camera-solve`: `git add splatpipe/cameras/outputs.py splatpipe/cameras/__init__.py splatpipe/cameras/__main__.py tests/conftest.py tests/test_cameras_step.py`, then commit those files with the message "Cameras: the outputs, the step and its CLI".
    - _Requirements: 16.4, 16.13_

- [x] 8. The integration test with the real COLMAP
  - Files: `tests/test_cameras_real.py`, new.
  - Design: "The real COLMAP tests", item 1, F5 and O6.
  - COLMAP on the laptop: yes. The test skips, with its reason, where `colmap` isn't on the PATH.
  - The update for D15 adds 8.4 to 8.11, which name their own files, design sections and requirements.
  - _Requirements: 1.7, 10.2, 10.5, 10.6, 11.1, 16.4, 16.6, 16.8, 16.13_

  - [x] 8.1 Write the synthetic scene and its picks in `tests/test_cameras_real.py`
    - The scene: three textured rectangles meeting at a corner, a floor, a back wall and a side wall, about 4 m across, so the scene isn't planar. Each has its own texture from a fixed numpy seed: smoothed noise at three scales, plus random rectangles and discs.
    - The camera: a pinhole camera with f = 260 px at 320 × 240 and no distortion. It moves 2 m sideways at a constant height, with no pitch and no roll, and turns to keep looking at the corner, over 30 frames.
    - The render: numpy ray casting. Each pixel's ray hits the nearest rectangle, whose texture is sampled bilinearly.
    - The picks go into the Job's `work/frames/`: `images/NNNNNN.jpg` at JPEG quality 95, `grey/NNNNNN.jpg` from `splatpipe.frames.files.grey_copy`, a black 32 × 24 corner painted into every frame with a `mask.png` that blacks it out, and a `frames.json` with every Frames_Record field and `mask_used` true. Nothing is written outside `tmp_path`.
    - _Requirements: 16.6, 16.13_

  - [x] 8.2 Write the integration test in `tests/test_cameras_real.py`
    - Skip it with the reason "COLMAP isn't installed" when `colmap` isn't on the PATH.
    - Make the Job with `new_job`, write the picks of 8.1, and run `splatpipe.cameras.run` with the real `TOOLS`. `set_cameras` puts `min_placed_frames` back at the real 20 and sets each Time_Limit to 600 s, so every other value is the real config's.
    - The report is done, and at least 24 of the 30 picks are placed. The Work_Folder matches Req 11.1. `sparse/` holds exactly three files, which our reader reads as one PINHOLE camera that every image uses.
    - `colmap model_analyzer --path work/cameras/sparse` exits with 0 and reports `Cameras: 1`, the placed count as `Registered images:` and the `points3d` count as `Points:`. `colmap model_converter --output_type BIN` from `sparse/` into a folder the test makes first writes a `cameras.bin`, an `images.bin` and a `points3D.bin` equal to ours byte for byte (O6). Both runs get `timeout=60`.
    - `masks/` holds one PNG per Placed_Frame at the camera's size, with only 0 and 255 and the painted corner at 0.
    - The levelled cameras' mean up is (0, −1, 0) within 1e-9. Their centres' heights spread by less than 2 % of the path's length, because the true path is level and the cameras have no pitch or roll. The Heading_Camera that `solve.json` names has its flattened view along +Z.
    - `out/cameras.json` matches `sparse/`, and `work/frames/` is unchanged.
    - _Requirements: 1.7, 10.2, 10.5, 10.6, 11.1, 16.6, 16.8, 16.13_

  - [x] 8.3 Run the tests and commit
    - Run `uv run pytest`. Every test passes, `tests/test_guard.py` included. The integration test runs where COLMAP is installed and is skipped, with its reason, where it isn't.
    - On `feat/camera-solve`: `git add tests/test_cameras_real.py`, then commit that file with the message "Cameras: integration test with the real COLMAP".
    - _Requirements: 16.4, 16.13_

  - [x] 8.4 Write `splatpipe/cameras/smoke.py`
    - Files: `splatpipe/cameras/smoke.py`, new.
    - Design: "smoke.py (the Cameras_Smoke_Check)", the Smoke_Picks' row of "The Fixed_Constants", "The real COLMAP tests", C17 to C21 and D15.
    - `SmokeSizes`, `SmokeSolve`, `CamerasSmoke`, `JOB`, `HELP`, `render_pick()`, `write_smoke_picks()`, `passed_options()`, `listed_options()`, `check_options()`, `smoke_settings()`, `run_smoke_solve()`, `check_smoke_solve()` and `run_cameras_smoke()`, as the design gives them.
    - The Smoke_Picks' scene moves from `tests/test_cameras_real.py` as it is: the room corner's three textured rectangles with the seeds 1, 2 and 3, the level path, the ray casting and the bilinear sampling. Its shapes and seeds are Fixed_Constants, and the size, f and number of Picks come from `SmokeSizes`. `write_smoke_picks` writes a `work/frames/` in the Frames_Contract's layout, with the names of `splatpipe/frames/files.py`, the painted corner and `mask.png`, and `frames.json` last.
    - `check_options` runs `colmap <command> -h` for the seven commands through `run_colmap`, each within `help_timeout_seconds` and with its Command_Log in `HELP`, and refuses a run that times out or fails, or an option the builders pass that `-h` doesn't list, with a `PipelineError` that names the command.
    - `run_smoke_solve` runs `splatpipe.cameras.work` with the step's real `TOOLS` on a `StepContext` it builds itself, with an in-memory `Job` in `<scratch>/job/` and the settings of `smoke_settings`. `check_smoke_solve` refuses too few placed Picks, and an f, fx or fy outside the tolerance, naming the `[cloud.smoke]` key. `run_cameras_smoke` runs the three in order and gives back the log line and the `SmokeSolve`. Nothing is written outside the scratch folder.
    - It imports only the standard library, numpy, cv2 and `splatpipe`, and no pytest, so it runs on the machine image. cv2 draws, blurs and encodes the Smoke_Picks' images, and never reads or writes video.
    - _Requirements: 12.13, 12.14, 16.13, 18.1, 18.3, 18.4, 18.5, 18.6, 18.7, 18.8_

  - [x] 8.5 Write unit tests for the smoke check in `tests/test_cameras_smoke.py`
    - Files: `tests/test_cameras_smoke.py`, new.
    - Design: the `tests/test_cameras_smoke.py` row of "Tests by file".
    - With no COLMAP: the scene's self-check, moved from `tests/test_cameras_real.py`, on `render_pick`; `write_smoke_picks` writes a `work/frames/` that `read_frames` accepts, with the painted corner and `mask.png`; `passed_options` holds every option the builders pass, `--ImageReader.camera_params` and `--ImageReader.camera_mask_path` included for `feature_extractor`; `listed_options` on a short `-h` text with a wrapped description and glog lines; and `smoke_settings` gives `[jobs].default_preset` and every Time_Limit key at `cameras_timeout_seconds`, which `read_settings` accepts.
    - `check_smoke_solve` passes exactly the placed share and exactly the tolerance, and refuses one Pick fewer and a focal length just past the tolerance for each of f, fx and fy, naming the `[cloud.smoke]` key.
    - `check_options` with `smoke.run_colmap` patched to write a scripted `-h` text: a `-h` that exits with 1, and one that lacks an option the step passes, each fail, naming the command and either the exit text with the Last_Error_Line or the missing option.
    - `run_smoke_solve` with `fake_colmap(global_models=(30,))` and the test config gives the `SmokeSolve`'s numbers from the files it read, f from the Final_Model's RADIAL camera and fx = fy from `sparse/`, writes nothing outside its scratch folder, and passes a Cameras_Step failure up with the step's own message.
    - _Requirements: 16.13, 16.14, 18.3, 18.4, 18.5, 18.6, 18.7, 18.8_

  - [x] 8.6 Run the integration test on the smoke code in `tests/test_cameras_real.py`
    - Files: `tests/test_cameras_real.py`, changed.
    - Design: "The real COLMAP tests", item 1, F5, F18, O6, C17 to C19, D15, D18 and D19.
    - The scene and its self-check leave the file. The test calls `smoke.run_cameras_smoke` with the real config and the real `[cloud.smoke]` numbers, both read by path, in `tmp_path`, with the step's real `TOOLS`. It's skipped with the reason "COLMAP isn't installed" where `colmap` isn't on the PATH.
    - `run_cameras_smoke`'s own checks pass. Then the test checks again, with its own code, that f, fx and fy are each within 2 % of 260 px and that fx = fy, on the level path that stays (D19).
    - The deeper checks stay, on `<tmp_path>/job/`: the Work_Folder of Req 11.1; `sparse/`'s three files read as one PINHOLE camera that every image uses; `colmap model_analyzer` and the byte-for-byte `colmap model_converter`, each with `timeout=60`; the masks, with the painted corner at 0 through the Final_Model's RADIAL camera; the levelling; `out/cameras.json` against `sparse/`; and `work/frames/` byte for byte against a second `write_smoke_picks`.
    - _Requirements: 1.7, 10.2, 10.5, 10.6, 11.1, 16.6, 16.8, 16.13, 18.4, 18.5_

  - [x] 8.7 Run the `-h` test on the smoke code in `tests/test_cameras_colmap.py`
    - Files: `tests/test_cameras_colmap.py`, changed.
    - Design: "The real COLMAP tests", item 2, C20 and A16.
    - The test calls `smoke.check_options` with the real config and `SmokeSizes` from the real `[cloud.smoke]`, both read by path, in `tmp_path`. So each `-h` runs within `check_timeout_seconds`, 60 s, for each Preset, with and without the Mask and for both Camera_Models, and the test expects no failure. The file's own `-h` reader goes, because `listed_options` takes its place. The test is skipped without COLMAP, with its reason.
    - _Requirements: 16.7, 16.8, 16.13, 18.3_

  - [x] 8.8 Add the seven `cameras_*` numbers to `[cloud.smoke]`, `cloud/config.py` and `tests/fake_aws.py`
    - Files: `config.toml`, `cloud/config.py` and `tests/fake_aws.py`, changed.
    - Design: "The Cameras_Smoke_Check's keys in `[cloud.smoke]`" and C21.
    - Append the seven keys and their comments to `[cloud.smoke]` in `config.toml`, after `train_sh_degree_interval`, exactly as the design writes them.
    - `SmokeChecks` gains `cameras_frames`, `cameras_width` and `cameras_height` as ints, and `cameras_focal_pixels`, `cameras_placed_share`, `cameras_focal_tolerance` and `cameras_timeout_seconds` as floats. `_read_smoke_checks` reads the first three with `_whole` and the other four with `_number`, each labelled `<key> for [cloud.smoke]`.
    - `cloud_config()` in `tests/fake_aws.py` gives the same seven values in its `"smoke"` table.
    - _Requirements: 18.9_

  - [x] 8.9 Add the `cameras` check to `cloud/smoke_runner.py`
    - Files: `cloud/smoke_runner.py`, changed.
    - Design: the paragraph on `_check_cameras` at the end of "smoke.py (the Cameras_Smoke_Check)", and the `cloud/smoke_runner.py` row of "Changes outside `splatpipe/cameras/`".
    - The module imports `splatpipe.cameras.smoke` as a module, `cameras_smoke`, so its `SmokeSizes` doesn't clash with the training check's. `_check_cameras(context)` builds the `SmokeSizes` from the seven `cameras_*` numbers and `check_timeout_seconds`, and returns `cameras_smoke.run_cameras_smoke(context.config, sizes, context.scratch, context.job_id).line`, looked up when the check runs, so a test can put a stand-in in its place.
    - `SMOKE_CHECKS` holds `SmokeCheck("cameras", _check_cameras)` right after `SmokeCheck("colmap", _check_colmap)`, and the module docstring's list of checks names the cameras check.
    - _Requirements: 18.2, 18.8_

  - [x] 8.10 Write unit tests for the cloud side of the cameras smoke check
    - Files: `tests/test_cloud_config.py` and `tests/test_cloud_smoke_runner.py`, changed.
    - Design: the `tests/test_cloud_config.py` and `tests/test_cloud_smoke_runner.py` rows of "Changes outside `splatpipe/cameras/`" and of "Tests by file".
    - `tests/test_cloud_config.py`: the example smoke table and the Hypothesis strategy's both gain the seven values, the wrong-type targets list the seven, three as whole numbers and four as numbers, and the real `[cloud.smoke]` holds the seven with their types. A float for a whole one, a bool, and 0 are each refused, naming the key and `[cloud.smoke]`.
    - `tests/test_cloud_smoke_runner.py`: `test_the_smoke_checks_cover_the_image_in_order` lists `torch-gpu`, `gsplat`, `fused-ssim`, `ffmpeg-zscale`, `colmap`, `cameras`, `node`, `splat-transform`, `moge-weights`, `training` and `cloud-runner`, and its comment says why `cameras` follows `colmap`. A new test runs `CHECKS["cameras"]` with a stand-in for `run_cameras_smoke`, and checks that the stand-in gets the config, the seven numbers and `check_timeout_seconds`, the check's own scratch folder and the smoke run's Job_Id, and that the check gives back the stand-in's line.
    - _Requirements: 16.14, 18.2, 18.8, 18.9_

  - [x] 8.11 Run the tests and commit
    - Run `uv run pytest`. Every test passes, `tests/test_guard.py` and `tests/test_cloud_imports.py` included. Where Homebrew's COLMAP is installed, the `-h` test and the integration test run the smoke code.
    - On `feat/camera-solve`: `git add splatpipe/cameras/smoke.py tests/test_cameras_smoke.py tests/test_cameras_real.py tests/test_cameras_colmap.py config.toml cloud/config.py tests/fake_aws.py cloud/smoke_runner.py tests/test_cloud_config.py tests/test_cloud_smoke_runner.py`, then `git commit -m "Cameras: the cameras smoke check" -- splatpipe/cameras/smoke.py tests/test_cameras_smoke.py tests/test_cameras_real.py tests/test_cameras_colmap.py config.toml cloud/config.py tests/fake_aws.py cloud/smoke_runner.py tests/test_cloud_config.py tests/test_cloud_smoke_runner.py`.
    - _Requirements: 16.4, 16.13_

- [x] 9. The cameras section of the pipeline doc and the COLMAP licence row
  - Files: `docs/pipeline.md`, `docs/third-party-licences.md` and `tests/test_docs.py`, all changed.
  - Design: the `docs/pipeline.md`, `docs/third-party-licences.md` and `tests/test_docs.py` rows of "Changes outside `splatpipe/cameras/`", with its `## Step: cameras` and COLMAP row parts, "The Work_Folder", "The Model_Files", the two records' tables and "The Cameras_Report".
  - COLMAP on the laptop: not needed.
  - The update for D15 to D20 and small fixes a, b, c and e adds 9.5 to 9.11, which name their own files, design sections and requirements.
  - _Requirements: 15.1, 15.2, 15.3, 15.4, 15.5, 15.6, 15.8, 16.4, 16.12, 16.13_

  - [x] 9.1 Write `## Step: cameras` in `docs/pipeline.md`
    - Put it after `## Step: frames`. It opens with the CLI, `uv run python -m splatpipe.cameras --job <id> [--preset P]`, the plain function `splatpipe.cameras.run(job_id, preset=None)` and the order of the work, then has the Frames_Step's five parts.
    - What it reads: `work/frames/frames.json` and the fields the step uses, `images/`, `grey/` and `mask.png`. The Feature_Extraction reads the Grey_Copies, while the mappers and the Undistortion read the Colour_Picks, which share their names. The Job's machine image needs COLMAP 4.2.1 built as `.kiro/steering/tech.md` says.
    - What it writes: the Work_Folder tree, the Model_Files' layout tables and the Pose convention, the one PINHOLE camera, the names that match `work/cameras/images/`, the Training_Masks only with `mask_used`, the World_Up (0, −1, 0), the Heading along +Z and the Scene_Centre at the origin. It says the Training_Step reads these files with its own reader, with no COLMAP or pycolmap, and gives what the COLMAP_Folder holds and the Solve_Record's fields.
    - Config section: a table of every key in `[cameras]` and its Preset tables, with the COLMAP option each Option_Key sets, its value and what it does; the Fixed_Constants; and the Step_Needs, 16 vCPUs, a soft need, and 16 GB of memory, a hard need, as the Pipeline_Doc's "What each step asks the machine for" table plans them. The Start_Command reads them to pick the machine, and the step itself never checks the machine.
    - Step report fields: every count and timing of the Cameras_Report, and when each warning appears.
    - Files in out/: `out/cameras.json` and its fields. The step removes an earlier one when its own code starts and writes a new one last of all, so a run that fails in the step's code leaves none. A run that the step runner stops before the step's code starts leaves the earlier run's `work/cameras/` and `out/cameras.json` in place, next to a failed report.
    - _Requirements: 15.1, 15.2, 15.3, 15.4, 15.5, 15.6_

  - [x] 9.2 Add the step's use to the COLMAP row in `docs/third-party-licences.md`
    - "How we use it" keeps its text from `main` word for word up to "SIFT runs on the CPU.", build flags included, because cloud-jobs Req 14.5 requires them and `tests/test_cloud_image.py` checks the five-flag string `-DCUDA_ENABLED=OFF -DOPENGL_ENABLED=OFF -DGUI_ENABLED=OFF -DCGAL_ENABLED=OFF -DLSD_ENABLED=OFF` in the row verbatim.
    - `main`'s last sentence, "Never vendored, linked or shipped.", gives way to the design's text for the step: "The cameras step runs `feature_extractor`, `sequential_matcher`, `view_graph_calibrator`, `global_mapper`, `mapper`, `image_undistorter` and `image_undistorter_standalone` as separate command line processes on our private machine image, never vendored or linked, never shipped, and with no pycolmap. On laptops, Homebrew's COLMAP 4.2.1, which holds SiftGPU, CGAL and LSD, is used only for `-h` and small CPU tests, always with `use_gpu 0`."
    - "Installed" becomes "Yes, on the laptop through Homebrew. Not yet on the machine image." The licence stays BSD-3-Clause, and the row stays the Licence_List's only row with COLMAP in its name, which `tests/test_cloud_image.py` also checks.
    - _Requirements: 15.8_

  - [x] 9.3 Write the doc tests for the cameras section and the COLMAP row in `tests/test_docs.py`
    - `## Step: cameras` comes after `## Step: frames` and after the Skeleton's headings, with its five `###` parts.
    - What it reads names `frames.json`, `images/`, `grey/` and `mask.png`, says which COLMAP commands read the Grey_Copies and which the Colour_Picks, and names COLMAP 4.2.1 and `.kiro/steering/tech.md`.
    - What it writes names every part of the Work_Folder, the three Model_Files, PINHOLE, (0, −1, 0), +Z, the origin and pycolmap. It also names every top-level field of the Solve_Record, taken from what `outputs.solve_record` gives for a small example, the way the frames tests take the Frames_Record's fields from `frames_record`.
    - Config section names every key of the real `[cameras]`, read by path, and `near_overlap`, and states 16 vCPUs, 16 GB and that the step never checks the machine.
    - Step report fields names every count and timing of the Cameras_Report.
    - Files in out/ names `out/cameras.json` and every field of the Viewer_Cameras, taken from what `outputs.viewer_cameras` gives, says the file is removed when the step starts and written last, and says that a run the step runner stops before the step's code starts leaves the earlier one in place.
    - The COLMAP row names the seven commands, "never vendored or linked", pycolmap, Homebrew's COLMAP 4.2.1, SiftGPU, CGAL and LSD, `-h` and `use_gpu 0`. Its Installed cell says the laptop has it through Homebrew and the machine image doesn't yet.
    - _Requirements: 15.1, 15.2, 15.3, 15.4, 15.5, 15.6, 15.8, 16.12_

  - [x] 9.4 Run the tests and commit
    - Run `uv run pytest`. Every test passes, `tests/test_guard.py` included, and so do the COLMAP row checks in `tests/test_cloud_image.py`, which the full run takes in.
    - On `feat/camera-solve`: `git add docs/pipeline.md docs/third-party-licences.md tests/test_docs.py`, then commit those files with the message "Cameras: pipeline doc section and the COLMAP licence row".
    - _Requirements: 16.4, 16.13_

  - [x] 9.5 Update `## Step: cameras`, "The cap deadline" and "The smoke test" in `docs/pipeline.md`
    - Files: `docs/pipeline.md`, changed.
    - Design: the list "The update for D15 to D20 changes the doc in these places" in "Changes outside `splatpipe/cameras/`", "`out/cameras.json` (the Viewer_Cameras)" and "The Model_Files".
    - `## Step: cameras`: RADIAL and RADIAL_FISHEYE wherever it says OPENCV or OPENCV_FISHEYE, with the Fisheye_Start's five numbers, the Model_Files' ids and parameters for PINHOLE (1), RADIAL (3) and RADIAL_FISHEYE (9), and that the one PINHOLE camera has fx = fy. The two `ba_refine_extra_params` rows and the bullet on them say that the settings check refuses false, and why.
    - The Cap_Deadline in `## Step: cameras`: the order of the work reads it right after the settings, `### What it reads` lists `SPLATPIPE_CAP_DEADLINE` as the train section does, and the paragraph after the order and the Time_Limit bullet say that each limit is capped at the time left before it, in place of "The step doesn't read `SPLATPIPE_CAP_DEADLINE`". The errors list gains a bad `SPLATPIPE_CAP_DEADLINE`, no time left and a limit the cap deadline set, and the undistortion error no longer says to set the refine keys to true.
    - `### Files in out/`: a run that the step runner stops before the step's code starts leaves the earlier run's `out/cameras.json` in place, next to a failed report. When it stops on the config, like a `[cameras]` section with a missing preset table, it also leaves the earlier run's `work/cameras/`. When it stops later, like on a step module that fails to import, it has already emptied `work/cameras/`, so only `out/cameras.json` is left (small fix c).
    - "The cap deadline": the train step's words stay, and a sentence adds that the cameras step reads it too and caps each COLMAP command's time limit at the time left before it. "The smoke test": after "COLMAP, Node and splat-transform run.", the design's bullet on the cameras check, with its numbers in `[cloud.smoke]`, like `cameras_frames`.
    - The words "the step's machine needs" stay, as in the frames and train sections.
    - _Requirements: 15.6, 15.9, 15.10, 15.11_

  - [x] 9.6 Change the Homebrew sentence of the COLMAP row in `docs/third-party-licences.md`
    - Files: `docs/third-party-licences.md`, changed.
    - Design: "The COLMAP row", D18 and A15.
    - The sentence on laptops, which ends "is used only for `-h` and small CPU tests, always with `use_gpu 0`.", becomes the design's: Homebrew's COLMAP 4.2.1, which holds SiftGPU, CGAL and LSD, runs `-h` and the laptop tests, the integration test's whole solve included, always with `use_gpu 0`, as Samarth OKed, with his reasons. The rest of the row stays word for word, and it stays the only row with COLMAP in its name.
    - _Requirements: 15.8_

  - [x] 9.7 Name RADIAL and RADIAL_FISHEYE in the colmap-cli skill
    - Files: `.kiro/skills/colmap-cli/SKILL.md` and `.kiro/skills/colmap-cli/references/options.md`, changed.
    - Design: "The colmap-cli skill" in "Changes outside `splatpipe/cameras/`", F2 and D19 (small fix a).
    - In `SKILL.md`'s `feature_extractor` table, the `--ImageReader.camera_model` and `--ImageReader.camera_params` rows gain the design's sentences on what the cameras step passes. In `references/options.md`, the "Left out" bullet under the `feature_extractor` block no longer names `--ImageReader.camera_params`, and the design's new bullet says that `--ImageReader.camera_model` takes `camera_model`, RADIAL or RADIAL_FISHEYE, and that `--ImageReader.camera_params` goes in only for RADIAL_FISHEYE, with the Fisheye_Start f, cx, cy, k1, k2.
    - Every code block of `references/options.md` stays as it is, `--ImageReader.camera_model SIMPLE_RADIAL` included, because the settings test and Property 1 parse those blocks.
    - _Requirements: 15.12_

  - [x] 9.8 Say Step_Needs in the frames spec and in `splatpipe/frames/settings.py`
    - Files: `.kiro/specs/frames/requirements.md`, `.kiro/specs/frames/design.md`, `.kiro/specs/frames/tasks.md` and `splatpipe/frames/settings.py`, changed.
    - Design: "Step_Needs in the frames spec" in "Changes outside `splatpipe/cameras/`" (small fix b).
    - Each "Machine_Needs" becomes "Step_Needs": in `requirements.md`, the out-of-scope line, the glossary entry, which also says it's the cloud-jobs spec's name, and the three criteria; in `design.md`, the two `tests/test_docs.py` rows; in `tasks.md`, the one sub-task bullet, with no task or sub-task title line or checkbox changed; and in `splatpipe/frames/settings.py`, the comment above `min_vcpus`, with no code changed.
    - `docs/pipeline.md` has no "Machine_Needs", so it stays as it is for this fix.
    - _Requirements: 15.13_

  - [x] 9.9 Add the missing cameras lines to `docs/devlog.md`
    - Files: `docs/devlog.md`, changed.
    - Design: "The devlog lines" and C23 (small fix e).
    - Add one line for each of the sub-tasks 1.1 to 1.6, 2.1 to 2.4, 3.1 to 3.6, 4.1 to 4.9, 5.1 and 5.3, in the hook's format, `- <YYYY-MM-DD HH:MM>+05:30 | feat/camera-solve | cameras <id> <title> | <files>`, with the sub-task's title as this file gives it and the time and files that the design's table takes from the git history, the files as `git show --name-only --format= <commit>` lists them, comma-separated.
    - The 1.x and 2.x lines go between the existing lines of 2026-11-23 21:15 and 23:18, and the 3.x, 4.x, 5.1 and 5.3 lines between 2026-11-23 23:36 and 2026-11-24 11:11, the lines of one minute in sub-task order, so 5.1 and 5.3 follow 4.9. No other line changes.
    - This sub-task has a wave of its own. The devlog hook appends its own line and commits `docs/devlog.md` right after it, with these lines in it, so 9.11 leaves that file out, like every other commit.
    - _Requirements: 15.14_

  - [x] 9.10 Write the doc tests for the update in `tests/test_docs.py`
    - Files: `tests/test_docs.py`, changed.
    - Design: the `tests/test_docs.py` row of the update's table in "Changes outside `splatpipe/cameras/`", and the doc tests row of "Tests by file".
    - `## Step: cameras` names `"RADIAL"`, `"RADIAL_FISHEYE"`, fx = fy, `SPLATPIPE_CAP_DEADLINE` and the refusal of a false `mapper_ba_refine_extra_params` or `global_mapper_ba_refine_extra_params`, and holds neither "OPENCV" nor "The step doesn't read `SPLATPIPE_CAP_DEADLINE`".
    - The Files in out/ test checks the two stopped runs, "it also leaves the earlier run's `work/cameras/`" and "so only `out/cameras.json` is left", in place of "leaves the earlier run's `work/cameras/` and `out/cameras.json` in place".
    - "The cap deadline" says that the cameras step reads it too, and "The smoke test" names the cameras check and `` `cameras_frames` ``, beside the training check's phrases.
    - The colmap-cli skill's `SKILL.md` and `references/options.md` each name RADIAL and RADIAL_FISHEYE, and the notes of `options.md` no longer list `--ImageReader.camera_params` as left out.
    - The COLMAP row also says "the integration test's whole solve included" and "as Samarth OKed", and no longer says "is used only for `-h` and small CPU tests". The Solve_Record example that the tests take the record's fields from uses `"RADIAL"`.
    - _Requirements: 15.6, 15.8, 15.9, 15.10, 15.11, 15.12, 16.12_

  - [x] 9.11 Run the tests and commit
    - Run `uv run pytest`. Every test passes, `tests/test_guard.py`, `tests/test_cloud_docs.py` and the COLMAP row checks in `tests/test_cloud_image.py` included.
    - On `feat/camera-solve`: `git add docs/pipeline.md docs/third-party-licences.md .kiro/skills/colmap-cli/SKILL.md .kiro/skills/colmap-cli/references/options.md .kiro/specs/frames/requirements.md .kiro/specs/frames/design.md .kiro/specs/frames/tasks.md splatpipe/frames/settings.py tests/test_docs.py`, then `git commit -m "Cameras: docs, skill and licence row for RADIAL, the cap deadline and the smoke check; Step_Needs in the frames spec" -- docs/pipeline.md docs/third-party-licences.md .kiro/skills/colmap-cli/SKILL.md .kiro/skills/colmap-cli/references/options.md .kiro/specs/frames/requirements.md .kiro/specs/frames/design.md .kiro/specs/frames/tasks.md splatpipe/frames/settings.py tests/test_docs.py`. `docs/devlog.md` stays out, because the hook committed it after 9.9.
    - _Requirements: 16.4, 16.13_

- [x] 10. The cameras step joins the step list
  - Files: `splatpipe/run.py`, `tests/test_run.py`, `tests/test_frames_step.py`, `tests/test_imports.py`, `tests/test_docs.py`, `tests/test_cameras_step.py`, `docs/pipeline.md` and `.kiro/specs/frames/requirements.md`, all changed.
  - Design: the `splatpipe/run.py`, `tests/test_run.py`, `tests/test_frames_step.py`, `tests/test_imports.py`, `tests/test_docs.py` and frames spec rows of "Changes outside `splatpipe/cameras/`", and its "The step list" words.
  - COLMAP on the laptop: not needed. The `run_job` test uses `FakeColmap`, and the real ffmpeg test runs the frames entry alone.
  - D1 also names `tests/conftest.py`. It already has `[cameras]` from task 1, so a Job that runs the whole Step_List gets the cameras settings, and it needs no change here.
  - _Requirements: 14.3, 14.4, 14.5, 14.6, 14.7, 14.8, 14.9, 15.7, 16.4, 16.13_

  - [x] 10.1 Change the Step_List in `splatpipe/run.py`
    - `STEP_LIST = (StepEntry("frames", "splatpipe.frames", True), StepEntry("cameras", "splatpipe.cameras", True))`. The comment above it says the Cameras_Step is the second entry and each later step spec adds its own entry after it, in order, and keeps the words that the Fake_Step stays out.
    - `_NO_STEPS` and the refusal of an empty list stay as they are.
    - _Requirements: 14.3_

  - [x] 10.2 Update `tests/test_run.py`
    - `test_the_step_list_holds_the_frames_step_and_is_the_default_list` gets a name for the new list, like `test_the_step_list_holds_the_frames_and_cameras_steps_and_is_the_default_list`, and asserts the list of 10.1. Its comment names the cameras spec's Req 14.3.
    - _Requirements: 14.5_

  - [x] 10.3 Update `tests/test_frames_step.py`
    - The real ffmpeg `run_job` test passes `steps=[entry for entry in STEP_LIST if entry.name == "frames"]`, so it still checks the Frames_Step on its own, and the Job is done once that step is.
    - The module docstring and the test's comment say the test runs the Step_List's frames entry alone, no longer that the Frames_Step is the only entry.
    - _Requirements: 14.6_

  - [x] 10.4 Update `tests/test_imports.py`
    - The docstring of `test_each_step_list_entry_is_a_step_inside_splatpipe_with_its_config_section` says the check loads `[frames]` and `[cameras]` from the real config for each of the three Presets, and no longer describes the Frames_Step as the only entry. Its loop already goes through every entry, so it loads `[cameras]` with no code change.
    - _Requirements: 14.7_

  - [x] 10.5 Update "The step list" in `docs/pipeline.md`, with its test
    - In place of the first three sentences of the paragraph about the frames step, from "The list holds the frames step as its first, required entry" to "a job is done once the frames step is done.", write the design's words: "The list holds the frames step and then the cameras step, both required: `StepEntry("frames", "splatpipe.frames", True)` and `StepEntry("cameras", "splatpipe.cameras", True)`. Each later step spec adds its own entry after the cameras step. Until the training spec adds its entry, a job is done once the cameras step is done."
    - Leave the rest of the part as it is on `main`: the paragraph's last sentence, "The planned steps are:", the two bullets "Required steps: picking frames, solving cameras and training." and "Optional steps: depth priors and clean and export.", and the closing sentence "Running a job in the cloud comes in its own spec, and it isn't a step."
    - In `tests/test_docs.py`, the test of "The step list" gets a name that fits and checks the phrases "The list holds the frames step and then the cameras step, both required", "Each later step spec adds its own entry after the cameras step" and "Until the training spec adds its entry, a job is done once the cameras step is done". It checks that "Until the camera solve spec adds its entry", "The list starts empty" and "No steps are set up yet" are gone. The existing Step_List check now finds `## Step: cameras`, and its docstring says so.
    - _Requirements: 14.8, 15.7_

  - [x] 10.6 Add the notes to the frames spec's requirements
    - Under Requirement 11, criteria 3 and 4, and under Requirement 12, criterion 5, of `.kiro/specs/frames/requirements.md`, add the design's note: "The cameras spec added the second Step_List entry, `StepEntry("cameras", "splatpipe.cameras", True)`. This criterion describes the Step_List as the frames spec left it." Leave the criteria's own words as they are.
    - _Requirements: 14.9_

  - [x] 10.7 Write the `run_job` test of both steps in `tests/test_cameras_step.py`
    - Make a Job with `new_job`, install `FakeTools` for the Frames_Step and `FakeColmap` for the Cameras_Step, and run `run_job(job_id)` with the real Step_List.
    - The Job_Status is `done`, both steps are done, `work/frames/` and `work/cameras/` have their designs' layouts, and `out/cameras.json` is written.
    - _Requirements: 14.4_

  - [x] 10.8 Run the tests and commit
    - Run `uv run pytest`. Every test passes, `tests/test_guard.py` included.
    - On `feat/camera-solve`: `git add splatpipe/run.py tests/test_run.py tests/test_frames_step.py tests/test_imports.py tests/test_docs.py tests/test_cameras_step.py docs/pipeline.md .kiro/specs/frames/requirements.md`, then commit those files with the message "Cameras: second entry in the step list".
    - _Requirements: 16.4, 16.13_

  - [x] 10.9 Merge origin/main into feat/camera-solve, run the tests and commit the merge
    - This is one sub-task, not two, because the devlog hook commits `docs/devlog.md` after each sub-task, and git refuses a partial commit while a merge is in progress. So the merge, the fixes, the test run and the merge commit all happen inside it.
    - On `feat/camera-solve`, run `git merge origin/main`.
    - Resolve each of the ten conflicted files keeping both sides, as Samarth's rules and the updated requirements and design say: `STEP_LIST` as frames, cameras, train, clean and its comment in `splatpipe/run.py`, as the design's `splatpipe/run.py` row gives them (Req 14.3); `config.toml` with both sides' sections, in step order, so `[cameras]` and its Preset tables sit between `[frames.quality]` and `[train]`; `tests/conftest.py` with both sides' fixtures and both sides' config sections, `[cameras]` and `main`'s `[train]` and `[clean]`; `docs/pipeline.md` with every `## Step:` section in step order, `frames`, `cameras`, `train` and `clean`, and the design's merged words in "The step list" (Req 15.1, 15.7); the one merged COLMAP row the design gives, in `docs/third-party-licences.md` (Req 15.8); `docs/devlog.md` with every line from both sides, in time order; and `tests/test_run.py`, `tests/test_docs.py`, `tests/test_frames_step.py` and `tests/test_imports.py` using the merged Step_List, as the design's rows for them say (Req 14.5 to 14.8).
    - Any other test that runs the real Step_List, such as the `run_job` test in `tests/test_cameras_step.py` from 10.7 and `main`'s own `run_job` tests, still passes, with the stand-ins it needs or by passing only the entries it means to run (Req 14.4, 14.6).
    - `uv run pytest` passes, `tests/test_guard.py` and `main`'s COLMAP row checks in `tests/test_cloud_image.py` included.
    - Stage each resolved file, and any other file changed to make the tests pass, by name, then run `git commit` for the merge, with a message in the style of `main`'s earlier merges, such as "Merge origin/main into feat/camera-solve: keep both sides of the conflicts (cameras and training), step list frames, cameras, train, clean, devlog in time order". Never `git add -A` or `git add .`, and don't push.
    - _Requirements: 12.1, 12.15, 14.3, 14.4, 14.5, 14.6, 14.7, 14.8, 15.1, 15.7, 15.8, 16.4, 16.13_

## Notes

- Every test sub-task is required, as `.kiro/steering/rules.md` asks. None is marked optional.
- The last sub-task of each task is its checkpoint: the whole suite, then the commit. Each of the update's groups ends with a checkpoint of its own, 1.10, 2.8, 4.16, 5.15, 8.11 and 9.11. So the plan has no separate checkpoint tasks and stays at 10 tasks.
- Every acceptance criterion in `requirements.md` is named by at least one task or sub-task, those of the update's Requirements 17 and 18 and of Requirement 12, criterion 16, Requirement 15, criteria 9 to 14, and Requirement 16, criteria 14 and 15, by the update's sub-tasks. Each of the design's 8 correctness properties has its own property test sub-task: Property 7 in task 2, Properties 4 to 6 in task 3, Property 1 in task 4, Property 2 in task 5, Property 3 in task 6 and Property 8 in 5.12. The update changes Property 1 in 1.9 and 4.14 and Property 7 in 2.7. Properties 4, 5 and 6 are the levelling properties the user asked for: every distance kept, a scale of 1, and an up of (0, −1, 0) when the cameras agree.
- Only 4.8 and task 8 need COLMAP, and both skip without it. After 8.6 and 8.7 they run the Cameras_Smoke_Check's own code. Every other test uses `FakeColmap`, a stand-in for `colmap -h`, or no COLMAP at all, so the suite passes on a laptop without it.
- The spec's review phase has run, and these tasks follow the design's choices O1 to O11 and C1 to C11 as the design gives them. The update's sub-tasks follow C12 to C23, which settle what D15 to D20 leave open.
- The six parent tasks keep their `[x]`, though their new sub-tasks are open, because the task tools key on the exact title lines and checkboxes, which the update leaves as they are. Each parent gains one bullet that names its new sub-tasks.
- Two placements differ from the design's text, each because of where the code lives. The bit for bit check of a kept COLMAP_Frame (C10) is in the step tests of 7.5, because `__init__.py` makes that choice, though the design's test table lists it with the levelling tests. `FakeColmap` comes in two parts: task 5 writes the commands the solve flow runs, and task 7 adds the undistorters and the installing, because `Tools` comes with the step.
- Three small test helpers aren't named in the design: the `colmap_options` oracle in `tests/conftest.py`, shared by the settings test and Property 1 so that `options.md` is parsed in one place; the `FakeColmap` flag that writes every other camera upside down, for the kept COLMAP_Frame in 7.5; and the check in 1.5 that no file uses `cv2.VideoCapture` or `cv2.VideoWriter`, for Req 12.14.
- The graph runs the tasks in order, one at a time, so each task's test run and commit see only finished work. Inside a task, sub-tasks that write different files share a wave.
- The update's sub-tasks run in waves 43 to 67, after 10.9, one group at a time in the order 1, 2, 4, 5, 8, 9. Inside a group, sub-tasks that edit the same file are in different waves, and 9.9, the devlog lines, has a wave of its own, because the devlog hook commits `docs/devlog.md` after it.

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "1.2", "1.3"] },
    { "id": 1, "tasks": ["1.4", "1.5"] },
    { "id": 2, "tasks": ["1.6"] },
    { "id": 3, "tasks": ["2.1"] },
    { "id": 4, "tasks": ["2.2"] },
    { "id": 5, "tasks": ["2.3"] },
    { "id": 6, "tasks": ["2.4"] },
    { "id": 7, "tasks": ["3.1"] },
    { "id": 8, "tasks": ["3.2"] },
    { "id": 9, "tasks": ["3.3"] },
    { "id": 10, "tasks": ["3.4"] },
    { "id": 11, "tasks": ["3.5"] },
    { "id": 12, "tasks": ["3.6"] },
    { "id": 13, "tasks": ["4.1", "4.2", "4.4"] },
    { "id": 14, "tasks": ["4.3", "4.5"] },
    { "id": 15, "tasks": ["4.6"] },
    { "id": 16, "tasks": ["4.7"] },
    { "id": 17, "tasks": ["4.8"] },
    { "id": 18, "tasks": ["4.9"] },
    { "id": 19, "tasks": ["5.1", "5.3"] },
    { "id": 20, "tasks": ["5.2"] },
    { "id": 21, "tasks": ["5.4"] },
    { "id": 22, "tasks": ["5.5"] },
    { "id": 23, "tasks": ["5.6"] },
    { "id": 24, "tasks": ["6.1"] },
    { "id": 25, "tasks": ["6.2"] },
    { "id": 26, "tasks": ["6.3"] },
    { "id": 27, "tasks": ["6.4"] },
    { "id": 28, "tasks": ["7.1", "7.3"] },
    { "id": 29, "tasks": ["7.2", "7.4"] },
    { "id": 30, "tasks": ["7.5"] },
    { "id": 31, "tasks": ["7.6"] },
    { "id": 32, "tasks": ["7.7"] },
    { "id": 33, "tasks": ["8.1"] },
    { "id": 34, "tasks": ["8.2"] },
    { "id": 35, "tasks": ["8.3"] },
    { "id": 36, "tasks": ["9.1", "9.2"] },
    { "id": 37, "tasks": ["9.3"] },
    { "id": 38, "tasks": ["9.4"] },
    { "id": 39, "tasks": ["10.1", "10.5", "10.6"] },
    { "id": 40, "tasks": ["10.2", "10.3", "10.4", "10.7"] },
    { "id": 41, "tasks": ["10.8"] },
    { "id": 42, "tasks": ["10.9"] },
    { "id": 43, "tasks": ["1.7"] },
    { "id": 44, "tasks": ["1.8", "1.9"] },
    { "id": 45, "tasks": ["1.10"] },
    { "id": 46, "tasks": ["2.5"] },
    { "id": 47, "tasks": ["2.6"] },
    { "id": 48, "tasks": ["2.7"] },
    { "id": 49, "tasks": ["2.8"] },
    { "id": 50, "tasks": ["4.10", "4.11"] },
    { "id": 51, "tasks": ["4.12", "4.13"] },
    { "id": 52, "tasks": ["4.14", "4.15"] },
    { "id": 53, "tasks": ["4.16"] },
    { "id": 54, "tasks": ["5.7"] },
    { "id": 55, "tasks": ["5.8"] },
    { "id": 56, "tasks": ["5.9"] },
    { "id": 57, "tasks": ["5.10", "5.11"] },
    { "id": 58, "tasks": ["5.12", "5.13", "5.14"] },
    { "id": 59, "tasks": ["5.15"] },
    { "id": 60, "tasks": ["8.4", "8.8"] },
    { "id": 61, "tasks": ["8.5", "8.6", "8.7", "8.9"] },
    { "id": 62, "tasks": ["8.10"] },
    { "id": 63, "tasks": ["8.11"] },
    { "id": 64, "tasks": ["9.5", "9.6", "9.7", "9.8"] },
    { "id": 65, "tasks": ["9.9"] },
    { "id": 66, "tasks": ["9.10"] },
    { "id": 67, "tasks": ["9.11"] }
  ]
}
```
