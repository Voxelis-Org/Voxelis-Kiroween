# Implementation Plan: splat-training

## Overview

This plan builds the Train_Step in `splatpipe/train/` in 10 tasks, one for each item of the design's "Build order (10 tasks, each writing its own tests)", in that order: the settings, the Schedules and the deadline, the Model_Reader, the scene and the start splats, the PLY, the camera solve files and the GPU interface, the step, the GPU_Module, the smoke test's training check, and last the docs and the Step_List. Each task writes its code with that code's unit tests and Hypothesis property tests, runs the whole suite and commits its files by name. The code is Python 3.11 on numpy and OpenCV, which the Project_File already has, so no task adds a package (A1).

Four of the property tests check the rules asked for by name:

- Property 3, in 2.4: the SH degree only goes up and never above 3.
- Property 4, in 2.5: the positions' learning rate ends at 1% of its start.
- Properties 7 and 8, in 5.4 and 5.5: the PLY check rejects any file with a NaN or a missing field.
- Property 9, in 5.6: writing then reading a PLY gives back the same splats.

Three choices in the order matter:

- **The import check comes first.** Task 1 adds the train import check to `tests/test_imports.py`, with `gpu.py`'s three extra names allowed, and puts `fused-ssim` in `NEVER_PACKAGES`. So the test run of every later task checks the new files as they land.
- **The GPU_Module comes after everything it uses.** Tasks 1 to 7 build and test every CPU part, the step's whole flow included, on a laptop with `FakeGpu`, the GPU_Stand_In. Task 8 writes `gpu.py` against that tested interface. Its code runs only in the Smoke_Test, so its tests read its source.
- **The docs change with the Step_List.** Once `STEP_LIST` holds `train`, the existing check in `tests/test_docs.py` (`test_the_pipeline_doc_has_a_step_section_for_the_fake_step_and_every_listed_step`) needs `## Step: train`, and the real-ffmpeg Job test in `tests/test_frames_step.py` needs the frames entry alone. So task 10 changes `STEP_LIST`, the Pipeline_Doc and the tests that depend on them together.

Some files grow over several tasks, so each piece comes with the first task that uses it:

- `splatpipe/train/__init__.py`: task 1 makes it with only the package docstring, and task 7 writes the step.
- `tests/conftest.py`: task 1 adds `TRAIN`, `TRAIN_PRESETS`, `train_text`, `config_text`'s `train` argument, `set_train` and `no_cap_deadline`, and task 7 adds `FakeGpu` and `fake_gpu`.
- `tests/camera_files.py`: task 3 writes `write_model`, and task 6 adds `write_camera_solve_output`.
- `tests/test_imports.py`: task 1 adds the train import check and `fused-ssim`, task 8 adds `gpu.py`'s source checks and the fresh-process import, and task 10 changes three docstrings: the Step_List check's, and the two that say the clean spec can decide scipy.
- `tests/test_train_settings.py`: task 1 writes it, and task 10 adds the Fallback_List test.
- `config.toml`: task 1 adds `[train]`, and task 9 adds the seven `[cloud.smoke]` keys.
- `docs/pipeline.md` and `tests/test_docs.py`: task 9 changes and tests the smoke test part, and task 10 adds and tests `## Step: train` and four other parts. Before them, task 6 changes the `opencv-python-headless` row of `docs/third-party-licences.md` and its test in `tests/test_docs.py`.

## Working rules for every task

- Work on the existing `feat/splat-training` branch, never on `main`. Others commit to it too, so build on it as it is, with no rebase, reset or force push.
- The only step folder this spec makes is `splatpipe/train/`. Of the Skeleton, only the Step_List in `splatpipe/run.py` changes, in task 10. Outside `splatpipe/train/`, only the files in the design's "Changes outside `splatpipe/train/`" table change. The clean spec's files and the cloud-jobs and frames requirements files stay as they are, and the Clean_Step stays out of `STEP_LIST` (D5, A14).
- No new Python package (A1). `pyproject.toml` and `uv.lock` don't change and nothing new is installed, so the Licence_List gains no line.
- torch, gsplat and fused-ssim are imported only by `splatpipe/train/gpu.py`. Every other file in `splatpipe/train/` imports only the standard library, numpy, cv2 and `splatpipe`, and nothing in `splatpipe/` imports `cloud/` or boto3. The step reaches `gpu.py` only through its `GPU` hook, when it runs.
- No laptop test imports `splatpipe/train/gpu.py`, except to read its source as text. The tests run the step's GPU calls through `FakeGpu`, and reach `load_gpu`'s import only with `splatpipe.train.gpu_api.importlib` replaced by a stand-in.
- cv2 only decodes JPEG and PNG stills. It never reads or writes video.
- Every rate, weight, count, size, factor and time comes from `[train]`, or from `[cloud.smoke]` for the smoke check. Code holds only the Fixed_Constants of the requirements' glossary, the Smoke_Scene's radius of 1 and half-side of 0.25 among them (Req 9.5).
- Write the code fresh. No code from COLMAP, from Inria's gaussian-splatting repo, from the MCMC paper's repo or from the effective-rank paper's repo. Code adapted from gsplat's examples keeps the Apache-2.0 note that the gsplat-training skill gives.
- Tasks 2, 4, 5, 8, 9 and 10 load `skill::gsplat-training` and `skill::splat-files` with `load_resource` before any code is written.
- Nothing here costs money (A12). No task runs the Smoke_Test, an Image_Build, the Start_Command or any `aws` command, and no task sets `APPROVED_SPEND`.
- Tests run on the CPU with no GPU and no network, on tiny synthetic data that each test builds: models, JPEGs, masks and PLYs of a few pixels and a few dozen points, under `tmp_path` or a `tempfile.TemporaryDirectory()`. Every Job is made in the test config's Jobs_Folder, never in `data/`, and nothing a test makes goes into git. The autouse `no_cap_deadline` removes `SPLATPIPE_CAP_DEADLINE` before every test, and a test that needs it sets it with `monkeypatch.setenv`.
- Property tests use Hypothesis with `@settings(max_examples=100, deadline=None)`, and each docstring says `Feature: splat-training, Property N: <title>`. A property test that writes files makes a `tempfile.TemporaryDirectory()` per example, because `tmp_path` is shared by all of a test's examples. Each oracle is written in the test and never calls the function it checks.
- Every test sub-task is required. None is optional.
- The last sub-task of each task is its checkpoint. Run `uv run pytest` and fix what fails until every test passes, `tests/test_guard.py` included. Then stage the task's files by name with `git add <file> ...`, never with `git add -A`, `git add .` or a folder, and commit only those files with `git commit -m "<message>" -- <file> ...`, so nothing else that's staged goes in. Leave `docs/devlog.md` out: the devlog hook adds and commits its own line after each task.
- If a hook blocks something it shouldn't, say so and fix the hook. Never work around it.

## Tasks

- [ ] 1. Settings, the test config and the train import check
  - Files: `splatpipe/train/__init__.py`, `splatpipe/train/settings.py` and `tests/test_train_settings.py` (new). `config.toml`, `tests/conftest.py` and `tests/test_imports.py` (changed).
  - _Requirements: 2.11, 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7, 9.8, 9.9, 9.10, 9.11, 9.13, 9.14, 9.15, 14.3, 14.7, 14.9, 14.11, 14.12_

  - [ ] 1.1 Add `[train]` and its Preset tables to `config.toml`
    - Append `[train]`, `[train.fast]`, `[train.default]` and `[train.quality]` after `[frames.quality]`, as the design's "The `[train]` section of `config.toml`" gives them, with the comment above every key.
    - That's the 35 Shared_Settings of Req 9.2, `neighbour_chunk_pairs = 100000000` among them, and the Preset_Settings of Req 9.3: `steps` 7000, 15000 and 30000, `max_splats` 1000000, 2000000 and 3000000, and `min_gpu_memory_gb` 24, 24 and 48. Each key is in one place only.
    - Leave every other section as it is, so `fast` stays `[jobs].default_preset`.
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.9_

  - [ ] 1.2 Write `splatpipe/train/settings.py`
    - Make `splatpipe/train/` with an `__init__.py` that holds only the package docstring for now. Task 7 writes the step in it.
    - `TrainSettings` and `read_settings(settings, section, preset)`, as in the design's "settings.py" section, built like the Frames_Step's: a `_Kind` per kind of value, a dict of every key with its kind, and `type()` checked exactly, so a bool is never a number. It reads each key with `settings[key]`, so a missing key raises the Step_Runner's own message.
    - Whole numbers stay ints and other numbers become floats. A refusal is an `ExpectedFailure` that names the key, its table (`[train]`, or `[train.<preset>]` for `steps`, `max_splats` and `min_gpu_memory_gb`) and what the value must be, worded as the design's Error Handling table gives it.
    - Then `max_splats` below `start_scale_neighbours` + 1 is refused, naming both settings, their tables and their values.
    - `min_vcpus` and `min_memory_gb` are checked and kept, and nothing in the step uses them.
    - The module imports only the standard library and `splatpipe.errors`.
    - _Requirements: 2.11, 9.5, 9.6, 9.7, 9.8_

  - [ ] 1.3 Add `[train]`, `set_train` and `no_cap_deadline` to `tests/conftest.py`
    - `TRAIN` holds the 35 Shared_Settings at their real values, each with the TOML type it has in the real config. `TRAIN_PRESETS` holds `steps` 6, 8 and 10, `max_splats` 50, 60 and 70, and `min_gpu_memory_gb` 24, 24 and 48, as the design's "The test config's `[train]`" gives them.
    - `train_text(train)` writes `[train]` and its three tables the way `frames_text` writes `[frames]`. A Shared_Setting takes its new value, like `{"seed": 7}`, and a Preset_Setting takes a dict of the Presets it changes, like `{"max_splats": {"fast": 4}}`.
    - `config_text` gains a `train` argument and writes `[train]` after `[frames]`. A new `set_train(**values)` fixture rewrites the whole file, like `set_frames`, so `set_fail`, `set_frames` and `set_train` each leave the other sections at the test config's values.
    - An autouse fixture, `no_cap_deadline`, runs `monkeypatch.delenv("SPLATPIPE_CAP_DEADLINE", raising=False)` for every test. The name is written out, so conftest imports no `splatpipe.train`.
    - The module docstring names the new fixtures.
    - _Requirements: 9.15, 14.11_

  - [ ] 1.4 Write unit tests for the settings in `tests/test_train_settings.py`
    - The real config, read by path as `REPO_ROOT / "config.toml"`, has `[train]` with exactly the 35 Shared_Settings and values of Req 9.2, and the three Preset tables of Req 9.3, each key in one place. `step_settings("train", preset)` loads exactly the 38 keys for each Preset. The comment check in `tests/test_config.py` already covers the new keys.
    - `read_settings` on the test config gives a `TrainSettings` with the chosen Preset's three values, whole numbers as ints and the rest as floats.
    - Parametrised over every key: a value outside its kind, and a bool where a number belongs, gives an `ExpectedFailure` naming the key, its table and what it must be. The edge values pass: `mask_threshold` 1 and 255, `means_lr_final` and `gpu_memory_slack` 1, both fractions 0 and 1, and each weight 0.
    - `max_splats` 3 with `start_scale_neighbours` 3 is refused, naming both settings, their tables and their values, and 4 passes.
    - A missing key, read through `StepSettings`, gives the Step_Runner's message naming `[train]`, `[train.<preset>]` and the key.
    - `set_train` changes one shared key and one Preset's key for one test, and the next test sees the test config's values again.
    - The Fallback_List test comes in task 10, with the Step_List.
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.6, 9.7, 9.8, 9.15, 14.9_

  - [ ] 1.5 Add the train import check and `fused-ssim` to `tests/test_imports.py`
    - A new test reads every `splatpipe/train/*.py` with the file's `ast` reader. Every file but `gpu.py` imports only names in `sys.stdlib_module_names`, numpy, cv2 and `splatpipe`, and `gpu.py` imports only those and torch, gsplat and fused_ssim. A failure gives the file and the line. The test finds the files with glob, so each later task's files are checked as they land.
    - `fused-ssim` joins `NEVER_PACKAGES`, so the dependency check refuses it in `pyproject.toml` and `uv.lock`, as it refuses torch and gsplat.
    - The module docstring describes both.
    - _Requirements: 9.10, 9.11, 9.13, 14.7_

  - [ ] 1.6 Run the tests and commit
    - Run `uv run pytest`. Every test passes, `tests/test_guard.py` included.
    - `pyproject.toml` and `uv.lock` are unchanged, so no package was added and the Licence_List gains no line.
    - On `feat/splat-training`: `git add config.toml splatpipe/train/__init__.py splatpipe/train/settings.py tests/conftest.py tests/test_train_settings.py tests/test_imports.py`, then commit those files with the message "Train: settings, the test config and the import check".
    - _Requirements: 9.10, 9.14, 14.3, 14.12_

- [ ] 2. Schedules and the cap deadline
  - Files: `splatpipe/train/schedules.py`, `splatpipe/train/deadline.py`, `tests/test_train_schedules.py` and `tests/test_train_deadline.py`, all new.
  - _Requirements: 4.4, 4.6, 4.8, 4.11, 4.18, 4.19, 4.20, 5.7, 7.2, 7.3, 7.4, 7.5, 7.6, 7.11, 12.6, 14.1, 14.2, 14.3, 14.6, 14.12_

  - [ ] 2.1 Write `splatpipe/train/schedules.py`
    - Load `skill::gsplat-training` and `skill::splat-files` with `load_resource` before writing code.
    - `MAX_SH_DEGREE`, `sh_degree()`, `means_lr_schedule()`, `refine_stop()`, `erank_start()`, `refine_iterations()` and `image_order()`, as in the design's "schedules.py (pure)" section and the Means_LR row of its decisions table. The module is pure and imports only the standard library and numpy.
    - `means_lr_schedule(start, final, planned)` gives `planned + 1` rates: `start` and `start × final` set outright at the two ends, `start × final ** (i / planned)` between, then a running minimum so no rate rises. No torch scheduler.
    - `refine_stop` and `erank_start` round with Python's `round`. `refine_iterations` keeps each i below `planned` with `refine_start < i < stop` and `i % refine_every == 0`, gsplat 1.5.3's condition for MCMC's relocate and add.
    - `image_order(count, seed)` yields `np.random.default_rng(seed).permutation(count)` pass after pass, from one generator, without end.
    - _Requirements: 4.4, 4.6, 4.8, 4.11, 4.18, 5.7_

  - [ ] 2.2 Write `splatpipe/train/deadline.py`
    - `CAP_DEADLINE_ENV`, `EXAMPLE`, `read_cap_deadline(environ)`, `stop_time()`, `should_stop()` and `utc_now()`, as in the design's "deadline.py" section. The module imports only the standard library and `splatpipe.errors`.
    - `read_cap_deadline` gives `None` when the variable isn't set. An empty value, a value `datetime.fromisoformat` refuses and a value with no UTC offset are each an `ExpectedFailure` with the Req 7.3 message: the variable, its value with `repr()` and `EXAMPLE`. A good value is turned to UTC.
    - `stop_time` is the Cap_Deadline minus the sum of `clean_reserve_minutes` and `save_margin_minutes`. `should_stop(now, stop)` is `stop is not None and now >= stop`, so the tests pass it a fake clock.
    - _Requirements: 7.2, 7.3, 7.4, 7.5, 7.6, 7.11_

  - [ ] 2.3 Write unit tests for the Schedules in `tests/test_train_schedules.py`
    - `sh_degree` at 0, 999, 1,000, 3,000 and 10⁶ with an interval of 1,000 gives 0, 0, 1, 3 and 3.
    - With the real config read by path, `refine_stop` gives 6,300, 13,500 and 25,000 and `erank_start` gives 1,633, 3,500 and 7,000 for `fast`, `default` and `quality`.
    - `means_lr_schedule(start, 0.01, 7000)` gives 7,001 rates, from `start` to 1% of it.
    - `refine_iterations(30, 10, 10, refine_stop(30, 25000, 0.9))` is `[20]`, with the A11 numbers.
    - `image_order(5, 42)` gives passes that are each a permutation of 0 to 4 and aren't all the same, and the same seed gives the same order again.
    - _Requirements: 4.4, 4.6, 4.8, 4.11, 5.7, 12.6, 14.1_

  - [ ] 2.4 Write the property test for Property 3 in `tests/test_train_schedules.py`, with Hypothesis
    - **Property 3: The SH_Degree starts at 0, never goes down and never passes 3**
    - **Validates: Requirements 4.19, 4.6**
    - Draw `sh_degree_interval` from 1 to 10⁶ and two Iterations from 0 to 10⁷, sorted so i ≤ j, with i = j drawn often.
    - Check that `sh_degree(0, interval)` is 0 and `sh_degree(i, interval) <= sh_degree(j, interval) <= 3`. So the SH degree only goes up and never goes above 3.
    - _Requirements: 4.6, 4.19, 14.2_

  - [ ] 2.5 Write the property test for Property 4 in `tests/test_train_schedules.py`, with Hypothesis
    - **Property 4: The Means_LR starts at its rate, never rises and ends at means_lr_final of it**
    - **Validates: Requirements 4.20, 4.4**
    - Draw starting rates from 1e-30 to 1e30, `means_lr_final` from 1e-30 to 1 with 1 itself drawn often, and Planned_Iterations from 1 to 40,000.
    - Check that `means_lr_schedule` gives `planned + 1` rates, the first exactly the start, none above the one before it, and the last within a relative error of 10⁻⁹ of `means_lr_final` × the start. With the real `means_lr_final` of 0.01, that's the positions' learning rate ending at 1% of its start.
    - _Requirements: 4.4, 4.20, 14.2_

  - [ ] 2.6 Write unit tests for the deadline in `tests/test_train_deadline.py`
    - `read_cap_deadline({})` is `None`.
    - `2026-11-21T14:25:30Z` and `2026-11-21T14:25:30.123456+00:00` read as UTC times 123,456 µs apart, and a `+05:30` time is turned to UTC.
    - An empty value, `"soon"` and `2026-11-21T14:25:30` with no offset are each refused with a message that names `SPLATPIPE_CAP_DEADLINE`, gives the value and gives `EXAMPLE`, with no "Traceback".
    - With the real `clean_reserve_minutes` of 10 and `save_margin_minutes` of 5, the `stop_time` of a 14:25:30 Cap_Deadline is 14:10:30.
    - With fixed datetimes, `should_stop` is false a microsecond before the Stop_Time, true at it and after it, and false with no Stop_Time.
    - `CAP_DEADLINE_ENV` equals `cloud.runner.CAP_DEADLINE_ENV` and `cloud.userdata.CAP_DEADLINE_ENV`.
    - _Requirements: 7.2, 7.3, 7.4, 7.5, 7.6, 7.11, 14.1, 14.6_

  - [ ] 2.7 Run the tests and commit
    - Run `uv run pytest`. Every test passes, `tests/test_guard.py` included.
    - On `feat/splat-training`: `git add splatpipe/train/schedules.py splatpipe/train/deadline.py tests/test_train_schedules.py tests/test_train_deadline.py`, then commit those files with the message "Train: schedules and the cap deadline".
    - _Requirements: 14.3, 14.12_

- [ ] 3. The Model_Reader
  - Files: `splatpipe/train/colmap.py`, `tests/camera_files.py` and `tests/test_train_colmap.py`, all new.
  - _Requirements: 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 1.10, 1.11, 1.20, 1.21, 14.1, 14.2, 14.3, 14.6, 14.12_

  - [ ] 3.1 Write `splatpipe/train/colmap.py`
    - `PINHOLE`, `PINHOLE_PARAMS`, `IMAGE_NAME`, `Camera`, `ModelImage`, `SparseModel`, `read_cameras()`, `read_images()`, `read_points()` and `read_model()`, as in the design's "colmap.py (the Model_Reader)" section and its layout table. Write it fresh from COLMAP's docs and the formats in its `read_write_model.py`. It imports only the standard library, numpy and `splatpipe.errors`: no COLMAP program, no pycolmap and no copied code.
    - Each reader reads its file with `Path.read_bytes()` and walks it with `struct.unpack_from` through a small `_Cursor`, whose `take(fmt)` checks that the bytes are there before it unpacks.
    - `read_cameras` refuses a count other than 1 as soon as it reads it, and a model id other than `PINHOLE`. Keypoints, errors, point ids and tracks are skipped by moving the offset, once the cursor has checked that `count × size` bytes are there. Each count is checked against its smallest record, 73 bytes per image and 51 per point, before the loop, so a huge count returns at once.
    - A record cut short, bytes after the last record, a count the file can't hold and a name with no NUL are each refused as "`<file>` is cut short or isn't a COLMAP binary model (`<what>`)".
    - Values are checked as they're read, as Req 1.9 lists them, and only a quaternion of four zeros counts as a length of 0. `read_images` refuses a name that doesn't fullmatch `IMAGE_NAME` or isn't UTF-8, giving only the image's id. It also refuses a camera id other than the Shared_Camera's, two images with one name, and no image at all. Names in messages are shown with `repr()`.
    - `read_model` reads the three files by name and nothing else in `sparse/`, and leaves the world frame as it is.
    - _Requirements: 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 1.10, 1.11, 1.20_

  - [ ] 3.2 Write `write_model` in `tests/camera_files.py`
    - The tests' own writer of COLMAP's binary model, written fresh with `struct.pack("<...")`, as the design's "Stand-ins" section describes it. `write_model(sparse, camera, images, points)` writes `cameras.bin`, `images.bin` and `points3D.bin`, keypoints, errors and tracks included.
    - It can cut a file short, add a trailing byte or write a bad count, for the refusal tests.
    - Its default scene is 3 images from cameras on an arc and 12 points in front of every camera. Task 6 adds the images and masks.
    - _Requirements: 1.21, 14.1, 14.12_

  - [ ] 3.3 Write unit tests for the Model_Reader in `tests/test_train_colmap.py`
    - Each refusal of Req 1.7 to 1.11 is an `ExpectedFailure` that names its file and what the requirement says, with no "Traceback": 0 and 2 cameras, model ids 0 and 4, a file cut short inside each kind of record, one trailing byte, a count of 2⁶⁰ that returns at once, a name with no NUL, a width of 0, an fx of −1, a NaN translation, a quaternion of four zeros, no image, an image of camera 2, two images named `a.jpg`, and the names `x/y.jpg`, `a\b.jpg`, `c:.jpg`, `.jpg`, `a.JPG` and bytes that aren't UTF-8.
    - `rigs.bin` and `frames.bin` holding junk beside the three files change nothing.
    - _Requirements: 1.6, 1.7, 1.8, 1.9, 1.10, 1.11, 14.1, 14.6_

  - [ ] 3.4 Write the property test for Property 1 in `tests/test_train_colmap.py`, with Hypothesis
    - **Property 1: The Model_Reader reads back what was written**
    - **Validates: Requirements 1.21, 1.4, 1.5**
    - Draw as the design's Property 1 strategy says: width and height from 1 to 10,000, fx and fy finite and above 0, cx and cy any finite floats, 1 to 6 images with unique valid names, non-zero quaternions and 0 to 40 keypoints each, and 0 to 60 points with tracks of 0 to 5 entries. Floats include −0.0, subnormals and values near ±1e308.
    - Write each example with `write_model` into its own `tempfile.TemporaryDirectory()`, then read it with `read_model`. The camera, the images in order with their ids, quaternions, translations, camera ids and names, and the points' positions and colours come back equal, with floats compared bit for bit through `.view(np.uint64)`.
    - _Requirements: 1.4, 1.5, 1.21, 14.2_

  - [ ] 3.5 Run the tests and commit
    - Run `uv run pytest`. Every test passes, `tests/test_guard.py` included.
    - On `feat/splat-training`: `git add splatpipe/train/colmap.py tests/camera_files.py tests/test_train_colmap.py`, then commit those files with the message "Train: the COLMAP model reader".
    - _Requirements: 14.3, 14.12_

- [ ] 4. The scene and the start splats
  - Files: `splatpipe/train/scene.py`, `splatpipe/train/splats.py`, `tests/test_train_scene.py` and `tests/test_train_splats.py`, all new.
  - _Requirements: 1.18, 1.19, 1.20, 3.1, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 3.10, 3.11, 3.12, 3.13, 3.14, 3.15, 3.16, 5.10, 6.4, 6.5, 6.6, 6.8, 8.9, 14.1, 14.2, 14.3, 14.6, 14.12_

  - [ ] 4.1 Write `splatpipe/train/scene.py`
    - Load `skill::gsplat-training` and `skill::splat-files` with `load_resource` before writing code.
    - `ImagePixels`, `TrainingView`, `Scene`, `training_size()`, `scaled_intrinsics()`, `world_to_camera()`, `camera_centres()`, `scene_extent()`, `median_scene_depth()`, `keep_mask()` and `build_scene()`, as in the design's "scene.py (pure)" section. The module imports only the standard library, numpy and `splatpipe`, and reads no file.
    - `training_size` rounds half up in integers and never scales up. `world_to_camera` divides the quaternion by its largest absolute value before it scales it to length 1.
    - `scene_extent` refuses a largest distance of 0 with the Req 3.7 message. `median_scene_depth` goes one camera at a time, leaves out each camera with no point at a z above 0, takes both medians with `numpy.median`, and refuses with the Req 3.9 message when no camera has a point in front.
    - `keep_mask` scales with the integer index maps of the design's decisions table, so each training pixel takes the mask pixel under its centre, not the one `cv2.INTER_NEAREST` takes. A pixel below `mask_threshold` is `False`.
    - `build_scene` gives one `TrainingView` per image, in `images.bin`'s order, holding none out and keeping the model's poses as they are, then the Scene_Extent, the Median_Scene_Depth and the Near_Plane.
    - _Requirements: 1.19, 1.20, 3.1, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 3.10, 5.10_

  - [ ] 4.2 Write `splatpipe/train/splats.py`
    - Load `skill::gsplat-training` and `skill::splat-files` with `load_resource` before writing code, for SH_C0 and the colour, opacity and scale conventions.
    - `SH_C0`, `SH_REST`, `Splats` with its `__len__`, `choose_points()`, `need_points()`, `log_start_distances()`, `start_log_scales()`, `start_splats()`, `mean_exposure()`, `bake_exposure()` and `drop_non_finite()`, as in the design's "splats.py (pure)" section. The module imports only the standard library, numpy and `splatpipe`.
    - `choose_points` keeps the subset in file order with `np.sort`. `need_points` refuses a count below `neighbours` + 1 with the Req 1.18 message.
    - `log_start_distances` never forms the root mean square: each row's log is log m + ½ · log(mean((d / m)²)), with m the row's largest distance. A row with m = 0 takes the smallest log of the other rows, and an all-zero array is refused with the Req 3.15 message. `start_log_scales` adds `log(factor)` and gives three equal columns.
    - `start_splats` gives float32 arrays as Req 3.12 says, with `sh_rest` of shape (N, 15, 3).
    - `mean_exposure` and `bake_exposure` work in float64, and the bake changes only the colours.
    - `len(splats)` is N, `means.shape[0]`. `drop_non_finite` rounds each of the six arrays to float32 inside `np.errstate(over="ignore")`, the rounding the PLY_Writer does, and reduces `np.isfinite` over every axis but the first into one `keep` of length N. It gives back each array's `[keep]`, with the input's own values and dtypes, and the number dropped as an `int`. With nothing dropped, it gives back the `Splats` it was given, with no copy.
    - _Requirements: 1.18, 3.11, 3.12, 3.13, 3.14, 3.15, 6.4, 6.5, 6.6, 8.9_

  - [ ] 4.3 Write unit tests for the scene in `tests/test_train_scene.py`
    - `training_size` with a `long_edge` of 1600 gives 1600 × 900 for 3840 × 2160 and for 2001 × 1125, and 900 × 1600 for 2160 × 3840, and keeps 1600 × 900 and 64 × 48 as they are.
    - `scaled_intrinsics` at half size halves fx, fy, cx and cy.
    - `keep_mask` scales 4 × 4 to 2 × 2 and 3 × 3 to 2 × 2 at the pixel centres, checked against an oracle with `Fraction`s. With a threshold of 128, a value of 127 is masked and 128 is kept.
    - `world_to_camera` gives the right matrix for the identity and for a turn about y, and the same matrix for a quaternion times 1e-300 and times 1e300. `camera_centres` gives −Rᵀt.
    - `scene_extent` is 1.1 for four cameras on the unit circle and refuses cameras that all sit in one place.
    - `median_scene_depth` leaves out a camera that faces away from every point, matches `numpy.median`, and refuses when no camera has a point in front.
    - `build_scene` gives one view per image, in file order, and the Near_Plane.
    - Each refusal names what its requirement says, with no "Traceback".
    - _Requirements: 1.19, 3.1, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 3.10, 5.10, 14.1, 14.6_

  - [ ] 4.4 Write unit tests for the splats in `tests/test_train_splats.py`
    - `choose_points` gives every point when they fit, and for 10 points with a budget of 4, the same 4 in file order for the same seed.
    - `need_points` refuses 3 points with 3 neighbours, giving the count and the fewest the step needs, and passes 4.
    - `log_start_distances` refuses an all-zero array with the Req 3.15 message.
    - `start_splats` for one splat gives the colour, the zero `sh_rest`, the logit of 0.5, the identity rotation and the log scale worked out by hand.
    - `mean_exposure` of two fixes is their element-wise mean.
    - `drop_non_finite`: splats with no non-finite value all come back unchanged, with a count of 0. One NaN in one field, tried in each of the 59 fields in turn, drops that splat alone, and +inf and −inf drop theirs. A baked float64 value too big for float32 drops its splat. When every splat holds a non-finite value, `len()` of the kept splats is 0 and the count is all of them. The splats kept stay in their order.
    - _Requirements: 1.18, 3.11, 3.12, 3.15, 6.4, 8.9, 14.1, 14.6_

  - [ ] 4.5 Write the property test for Property 2 in `tests/test_train_splats.py`, with Hypothesis
    - **Property 2: Every Start_Splat gets a finite log scale, the same on all three axes**
    - **Validates: Requirements 3.16, 3.12, 3.14**
    - Draw (N, k) distance arrays with N from 1 to 40 and k from 1 to 5, from exact zeros, the subnormal 5e-324, floats up to 1e308 and a few planted equal rows, filtered to hold a distance above 0, and factors from 1e-300 to 1e300.
    - Check that `start_log_scales` gives an (N, 3) array of finite values with three equal columns, and that each all-zero row holds the smallest value of the other rows, which the oracle finds with its own row-wise minimum.
    - _Requirements: 3.12, 3.14, 3.16, 14.2_

  - [ ] 4.6 Write the property test for Property 5 in `tests/test_train_splats.py`, with Hypothesis
    - **Property 5: The Exposure_Bake turns every colour c into Ā·c + b̄**
    - **Validates: Requirements 6.8, 6.5**
    - Draw 1 to 8 splats with coefficients from −10 to 10, Ā and b̄ with values from −4 to 4, and 16 basis values from −2 to 2 whose first is SH_C0.
    - The oracle works out each splat's colour as 0.5 plus the basis values times the coefficients, before and after `bake_exposure`. Check that the colour after is Ā·c + b̄ with `np.allclose`, at a relative tolerance of 1e-9 and an absolute one of 1e-9 times the largest value involved, in float64.
    - _Requirements: 6.5, 6.8, 14.2_

  - [ ] 4.7 Run the tests and commit
    - Run `uv run pytest`. Every test passes, `tests/test_guard.py` included.
    - On `feat/splat-training`: `git add splatpipe/train/scene.py splatpipe/train/splats.py tests/test_train_scene.py tests/test_train_splats.py`, then commit those files with the message "Train: the scene and the start splats".
    - _Requirements: 14.3, 14.12_

- [ ] 5. The PLY writer, reader and check
  - Files: `splatpipe/train/ply.py` and `tests/test_train_ply.py`, both new.
  - _Requirements: 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.9, 8.10, 8.13, 8.14, 8.15, 8.16, 8.17, 14.1, 14.2, 14.3, 14.6, 14.12_

  - [ ] 5.1 Write `splatpipe/train/ply.py`
    - Load `skill::gsplat-training` and `skill::splat-files` with `load_resource` before writing code. The Splat_Layout is the splat-files skill's standard 3DGS PLY, the layout the gsplat-training skill says `export_splats(format="ply")` writes, but the PLY_Writer is our own numpy code and never calls `export_splats()`.
    - `FIELDS`, `RECORD_BYTES`, `splat_header()`, `splat_records()`, `write_ply()`, `read_ply()` and `check_ply()`, as in the design's "ply.py" section and "The Splat_Layout (`out/train.ply`)". The module imports only the standard library, numpy and `splatpipe`.
    - `splat_records` fills one `np.empty((N, 59), "<f4")` by columns, with `sh_rest.transpose(0, 2, 1).reshape(N, 45)`, so `f_rest_k` holds channel k // 15's coefficient k % 15. Opacities stay logits, scales stay logs and quaternions go out w first. A float32 input is copied bit for bit, and a float64 one is rounded inside `np.errstate(over="ignore")`, so a value too big for float32 becomes an infinity that the check reports.
    - `write_ply` writes the header, then the records' own buffer through `memoryview`, with every splat it's given, 0 included. An `OSError` is the Req 8.10 `ExpectedFailure`, naming the file and `error.strerror`.
    - The header parse that the reader and the check share reads a bounded prefix, `len(splat_header(max(1, size // RECORD_BYTES)))` bytes, and checks every line, the field names in order, any other line with its number, and the file's size.
    - `read_ply` gives one float32 array per field, for 0 splats too. `check_ply` refuses 0 splats and any value that isn't finite, giving how many splats hold one and each field with its count, and gives back N.
    - _Requirements: 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.9, 8.10, 8.13_

  - [ ] 5.2 Write unit tests for the PLY in `tests/test_train_ply.py`
    - `splat_header(1)` is 1,472 bytes and `splat_header(3_000_000)` is 1,478, with exactly the lines of Req 8.3. A written file's size is the header's plus 236 bytes per splat.
    - `check_ply` refuses a file of 0 splats, a `comment` line, a second element, `property double x`, two fields swapped, and a size one byte short or one byte long, each with a message naming the file and what failed. For several NaNs, it gives the count per field.
    - `write_ply` with `Path.open` patched to raise `OSError(errno.ENOSPC, "No space left on device")` fails naming the file and the reason.
    - A baked float64 value too big for float32 is written as an infinity, which the check reports.
    - No message holds "Traceback".
    - _Requirements: 8.3, 8.6, 8.7, 8.10, 14.1, 14.6_

  - [ ] 5.3 Write the property test for Property 6 in `tests/test_train_ply.py`, with Hypothesis
    - **Property 6: A file of finite splats passes the PLY_Check**
    - **Validates: Requirements 8.14, 8.6**
    - Draw 1 to 20 splats, each value from `st.floats(width=32, allow_nan=False, allow_infinity=False)`, −0.0, subnormals and ±3.4e38 included, and write each example's file into its own `tempfile.TemporaryDirectory()`.
    - Check that `check_ply` passes the file and gives back the number of splats.
    - _Requirements: 8.6, 8.14, 14.2_

  - [ ] 5.4 Write the property test for Property 7 in `tests/test_train_ply.py`, with Hypothesis
    - **Property 7: One NaN or infinity anywhere fails the PLY_Check**
    - **Validates: Requirements 8.15**
    - Take the splats of Property 6, draw a record r, a field f and one of NaN, +inf and −inf, and replace the four bytes at the header's length + 236 r + 4 f with `struct.pack("<f", value)`.
    - Check that `check_ply` refuses the file, saying one splat holds a NaN or an infinity and naming field f with a count of 1. So the PLY check rejects any file with a NaN or an infinity in it.
    - _Requirements: 8.15, 14.2_

  - [ ] 5.5 Write the property test for Property 8 in `tests/test_train_ply.py`, with Hypothesis
    - **Property 8: A file missing any one field fails the PLY_Check**
    - **Validates: Requirements 8.16**
    - Take the splats of Property 6 and draw a field index f from 0 to 58. The test writes the file with its own code: `splat_header`'s lines without that field's `property` line, then `np.delete(records, f, axis=1)`.
    - Check that `check_ply` refuses the file with a message that names the field as missing. So the PLY check rejects any file with a field missing.
    - _Requirements: 8.16, 14.2_

  - [ ] 5.6 Write the property test for Property 9 in `tests/test_train_ply.py`, with Hypothesis
    - **Property 9: The PLY_Reader reads back every value the PLY_Writer wrote, bit for bit**
    - **Validates: Requirements 8.17, 8.3, 8.4, 8.5, 8.13**
    - Draw 0 to 20 splats whose values are raw 32-bit patterns, `st.integers(0, 2**32 - 1)` viewed as float32, mixed with ordinary floats, so NaNs with any payload, infinities, −0.0 and subnormals all turn up.
    - Check that `read_ply` of the file `write_ply` writes gives 59 arrays whose bits equal the splats', field by field as the design maps them: `x`, `y` and `z` the means, `f_dc_i` `sh0[:, i]`, `f_rest_k` `sh_rest[:, k % 15, k // 15]`, `opacity` the logits, `scale_i` the log scales and `rot_i` the quaternions, w first. Compare through `.view(np.uint32)`. So writing splats to a PLY and reading it back gives the same splats.
    - _Requirements: 8.3, 8.4, 8.5, 8.13, 8.17, 14.2_

  - [ ] 5.7 Run the tests and commit
    - Run `uv run pytest`. Every test passes, `tests/test_guard.py` included.
    - On `feat/splat-training`: `git add splatpipe/train/ply.py tests/test_train_ply.py`, then commit those files with the message "Train: the PLY writer, reader and check".
    - _Requirements: 14.3, 14.12_

- [ ] 6. The camera solve files and the GPU interface
  - Files: `splatpipe/train/files.py`, `splatpipe/train/gpu_api.py`, `tests/test_train_files.py` and `tests/test_train_gpu_api.py` (new). `tests/camera_files.py`, `docs/third-party-licences.md` and `tests/test_docs.py` (changed).
  - _Requirements: 1.1, 1.2, 1.12, 1.13, 1.14, 1.15, 1.16, 1.17, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 2.10, 2.11, 2.12, 3.2, 3.4, 4.16, 8.8, 8.10, 10.7, 13.16, 14.1, 14.3, 14.6, 14.12_

  - [ ] 6.1 Write `splatpipe/train/files.py`
    - The Fixed_Constant names `CAMERAS`, `IMAGES`, `MASKS`, `SPARSE`, `MODEL_FILES`, `TRAINED_SPLAT` and `RECORD`, and `check_inputs()`, `read_views()`, `move_into_out()`, `train_record()` and `write_record()`, as in the design's "files.py" section. The module imports only the standard library, numpy, cv2 and `splatpipe`.
    - `check_inputs` checks `images/` with `is_dir()`, then each model file in `sparse/` with `is_file()`, in that order, and refuses the first one missing with the Req 1.2 message. It reads nothing.
    - `read_views` goes through the images in `images.bin`'s order and opens only the files they name. Each JPEG is read with `Path.read_bytes()` and `cv2.imdecode` with `IMREAD_COLOR | IMREAD_IGNORE_ORIENTATION`, checked against the Shared_Camera's size, scaled down with `INTER_AREA` when it's larger than the Training_Size, and turned to RGB. `masks/<stem>.png`, when it's there, is decoded with `IMREAD_GRAYSCALE | IMREAD_IGNORE_ORIENTATION`, checked against its image's size, and turned into a keep mask with `keep_mask`. An image without one gets `None`.
    - Each refusal of Req 1.12, 1.13 and 1.16 names the file and the reason, with both sizes when they differ, and a mask that leaves no pixel is refused too.
    - `move_into_out` is one `os.replace`. It and `write_record`, which calls `splatpipe.jsonio.write_json`, turn an `OSError` into the Req 8.10 failure. `train_record` gives only plain JSON values.
    - This is the commit that brings the step's use of OpenCV in, so in `docs/third-party-licences.md` the `opencv-python-headless` row's "How we use it" also says that the train step decodes the camera solve's JPEG frames and PNG masks and scales them down. It keeps "It never reads or writes video". The Licence_List gains no row.
    - _Requirements: 1.1, 1.2, 1.12, 1.13, 1.14, 1.15, 1.16, 1.17, 3.2, 3.4, 8.8, 8.10, 10.7, 13.16_

  - [ ] 6.2 Write `splatpipe/train/gpu_api.py`
    - `GPU_MODULE`, `GPU_PACKAGES`, `GSPLAT_VERSION`, `GB`, `GpuInfo`, `TrainResult`, the `Gpu` protocol, `GpuOutOfMemory`, `NonFiniteLoss`, `out_of_memory_failure()`, `non_finite_message()`, `gpu_need_bytes()`, `gpu_fits()`, `check_gpu()` and `load_gpu()`, as in the design's "gpu_api.py" section. The module imports no torch. It imports `Scene`, `Splats` and `TrainSettings` from the CPU modules, so the protocol's types are real.
    - `gpu_fits(total_bytes, min_gpu_memory_gb, slack)` is `total_bytes >= slack × min_gpu_memory_gb × GB`, a pure function of its three numbers.
    - `check_gpu` gives the probe's total memory to `gpu_fits` and refuses, in this order: no CUDA (Req 2.3), a gsplat version other than 1.5.3 (Req 2.5), and a GPU that `gpu_fits` refuses (Req 2.10). It checks nothing else about the machine.
    - `load_gpu` calls `importlib.import_module(GPU_MODULE)` through its own `import importlib`, so a test can replace `splatpipe.train.gpu_api.importlib`, as `tests/test_run.py` replaces `splatpipe.step.importlib`. A `ModuleNotFoundError` whose `name`, up to its first dot, is in `GPU_PACKAGES` becomes the Req 2.2 `ExpectedFailure` naming the package. Any other error goes up as it is.
    - `NonFiniteLoss` is a `RuntimeError` and not a `PipelineError`, so the Step_Runner reports it as a bug. The messages of `GpuOutOfMemory`, `out_of_memory_failure` and `non_finite_message` are worded as the design gives them.
    - _Requirements: 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.10, 2.11, 2.12, 4.16_

  - [ ] 6.3 Add `write_camera_solve_output` to `tests/camera_files.py`
    - `write_camera_solve_output(job_folder, ...)` writes `work/cameras/`: the model with `write_model`, each image as a JPEG and each mask as a PNG with `cv2.imencode`, as the design's "Stand-ins" section describes it.
    - Its default scene is 3 images of 8 × 6 pixels, the 12 points of `write_model`'s default, and a mask on the first image only.
    - It can leave an image out, write junk bytes as an image, write an image or a mask of another size, write a mask that masks every pixel, and add files to `images/` and `masks/` that `images.bin` doesn't name.
    - _Requirements: 14.1, 14.12_

  - [ ] 6.4 Write unit tests for the files in `tests/test_train_files.py`
    - `check_inputs` refuses each of the four missing paths in turn, naming the path and the camera solve step, and reads nothing.
    - `read_views` on the default scene gives RGB order from a JPEG of a known BGR colour, scales an image larger than the Training_Size with `INTER_AREA`, and gives a keep mask only for the image with a mask. With `Path.read_bytes` wrapped to record its calls, the extra files in `images/` and `masks/` are never read.
    - Each refusal names the file and the reason: a missing image file, junk bytes as a JPEG, a JPEG one pixel wider than the camera with both sizes, a mask of another size, and a mask that masks every pixel.
    - `move_into_out` and `write_record` with `os.replace` or `write_json` raising `OSError(errno.ENOSPC, "No space left on device")` name the file and the reason.
    - `train_record` passes `check_json_value` with and without a Cap_Deadline.
    - No message holds "Traceback".
    - In `tests/test_docs.py`, `test_numpy_and_headless_opencv_say_yes_and_opencv_never_reads_or_writes_video` also checks that the `opencv-python-headless` row names the train step's JPEG frames and PNG masks, and still that it never reads or writes video.
    - _Requirements: 1.2, 1.12, 1.13, 1.14, 1.15, 1.16, 1.17, 3.2, 3.4, 8.8, 8.10, 10.7, 13.16, 14.1, 14.6_

  - [ ] 6.5 Write unit tests for the GPU interface in `tests/test_train_gpu_api.py`
    - `gpu_fits` passes 24,146,608,128 bytes for a `min_gpu_memory_gb` of 24 and refuses it for 48, passes 48,305,799,168 bytes for both, and passes exactly 21,600,000,000 bytes for 24, all with a slack of 0.9.
    - `check_gpu` refuses no CUDA, gsplat 1.5.2 and a small GPU, in that order, each with its message, and passes whatever `min_vcpus` and `min_memory_gb` say.
    - `load_gpu`, with `splatpipe.train.gpu_api.importlib` replaced by a stand-in whose `import_module` raises `ModuleNotFoundError` with `name` set to `torch`, `gsplat.strategy` or `fused_ssim`, gives the Req 2.2 message naming the package. With `fused_ssim_cuda`, the error goes up as it is. The stand-in records the name it's asked to import, and the tests check that it's `GPU_MODULE` and that `GPU_MODULE` is `"splatpipe.train.gpu"`, so a wrong name can't pass every laptop test and fail only in the Smoke_Test. A stand-in that gives back a module object shows `load_gpu` gives back that object. No test imports `splatpipe.train.gpu`.
    - `out_of_memory_failure` names Iteration 812, or the scene set-up for `None`, the Preset and `max_splats`. `non_finite_message` names the Iteration, the five Loss_Terms' values and the total.
    - _Requirements: 2.2, 2.3, 2.5, 2.6, 2.7, 2.8, 2.9, 2.10, 2.11, 2.12, 4.16, 14.1, 14.6_

  - [ ] 6.6 Run the tests and commit
    - Run `uv run pytest`. Every test passes, `tests/test_guard.py` included.
    - On `feat/splat-training`: `git add splatpipe/train/files.py splatpipe/train/gpu_api.py tests/camera_files.py tests/test_train_files.py tests/test_train_gpu_api.py docs/third-party-licences.md tests/test_docs.py`, then commit those files with the message "Train: the camera solve files and the GPU interface".
    - _Requirements: 14.3, 14.12_

- [ ] 7. The step and its CLI
  - Files: `splatpipe/train/__main__.py` and `tests/test_train_step.py` (new). `splatpipe/train/__init__.py` and `tests/conftest.py` (changed).
  - _Requirements: 1.1, 1.2, 1.19, 1.20, 2.1, 2.2, 2.12, 3.11, 4.16, 6.7, 7.1, 7.3, 7.7, 7.8, 7.9, 8.1, 8.2, 8.6, 8.7, 8.8, 8.9, 8.11, 8.12, 9.6, 9.7, 9.8, 9.12, 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7, 11.1, 11.2, 11.3, 11.6, 11.7, 11.8, 14.3, 14.4, 14.5, 14.6, 14.11, 14.12_

  - [ ] 7.1 Write the step in `splatpipe/train/__init__.py`
    - `GPU: Callable[[], Gpu] = load_gpu`, `run()` and `work()`, as in the design's "The step's flow (`__init__.py`)" section. `run` calls `run_step("train", "splatpipe.train", ...)`, like the Fake_Step. `work` reads `GPU` when it runs, so a test's `monkeypatch` reaches it through `run`, `run_step`, the CLI and `run_job`, and `import splatpipe.train` imports no GPU package.
    - `work` follows the design's 11 numbered steps in the order of A9: remove `out/train.ply`, the settings, the Cap_Deadline and the Stop_Time, `check_inputs`, then `GPU()`, the probe and `check_gpu`. Only then come the load, the training, the write, the check, the move and the Train_Record, inside `step.timed("load")`, `"train"`, `"write_ply"` and `"check_ply"` as the design places them.
    - Each count is added with `int(...)` as soon as it's known, at the point the design's Train_Report table gives, so a failed report holds exactly what the step had worked out.
    - `_out_of_memory(preset, cfg)` wraps both GPU calls and turns `GpuOutOfMemory` into `out_of_memory_failure`. With 0 Iterations run, the step fails with the Req 7.9 message, and with fewer than `steps`, it adds the Req 7.8 warning.
    - In the write block, `drop_non_finite` runs on the baked splats, after `bake_exposure` and before `write_ply`. When it dropped any, the step adds the Req 8.9 warning with the count. `splats` is added as `len(kept)`, the splats written. When it dropped every splat, `write_ply` writes 0 splats and the PLY_Check's "holds no splats" fails the run, with no new refusal. The Req 3.11, 7.8 and 8.9 warnings are worded as the design gives them.
    - The step writes only `work/train/train.ply`, `work/train/train.json` and `out/train.ply`, and relies on the Step_Runner emptying `work/train/` first. It imports only the standard library, numpy, `splatpipe.errors`, `splatpipe.step` and the CPU modules of `splatpipe/train/`.
    - _Requirements: 1.1, 1.2, 1.19, 2.1, 2.12, 3.11, 6.7, 7.1, 7.3, 7.7, 7.8, 7.9, 8.1, 8.2, 8.6, 8.7, 8.8, 8.9, 8.11, 8.12, 9.12, 10.1, 10.2, 10.3, 10.4, 10.5, 10.7, 11.1, 11.3, 11.6, 11.7, 11.8_

  - [ ] 7.2 Write `splatpipe/train/__main__.py`
    - `main(argv)` calls `step_main("train", "splatpipe.train", argv)`, and the file ends with `if __name__ == "__main__": sys.exit(main())`.
    - _Requirements: 11.1, 11.2_

  - [ ] 7.3 Add `FakeGpu` and `fake_gpu` to `tests/conftest.py`
    - `FakeGpu`, the GPU_Stand_In, as the design's "Stand-ins" section describes it. It's built with a `GpuInfo`, by default CUDA, gsplat `"1.5.3"` and 24,146,608,128 bytes, and records every call and the scene and Start_Splats it's given.
    - `nearest_distances` is brute force in numpy. `train` counts Iterations up to `settings.steps`, stopping when the real `should_stop(utc_now(), stop)` says so or at its own `stop_after`. It gives back the Start_Splats as float32, an `[I | 0]` Exposure_Fix per view or one the test sets, the Iterations it counted and their last SH_Degree from the real `sh_degree`.
    - It can be set to raise from either GPU call, like `GpuOutOfMemory(812)` or `NonFiniteLoss`, or to give back one splat with a NaN opacity, or every splat with one. `peak_memory_bytes()` gives a number the test sets.
    - The `fake_gpu` fixture installs each `FakeGpu` it builds with `monkeypatch.setattr("splatpipe.train.GPU", lambda: gpu)`, as `fake_tools` installs `TOOLS`. Like `FakeTools`, `FakeGpu` imports what it needs from `splatpipe.train` inside its methods and the fixture, so conftest imports no step package at module level.
    - _Requirements: 11.6, 14.4_

  - [ ] 7.4 Write unit tests for a done run in `tests/test_train_step.py`
    - Run the step through `splatpipe.train.run` with `FakeGpu` and the default Camera_Solve_Output, so the Step_Runner and the real report are part of each test.
    - A done run leaves `out/train.ply`, which passes `check_ply`, holds the Start_Splats with the Mean_Exposure baked in, and equals what `read_ply` gives back. `work/train/` holds only `train.json`, with every Train_Record field and `null` for both times.
    - The report holds every count of Req 10.1 as an `int`, the four timings and the 38 settings.
    - With `SPLATPIPE_CAP_DEADLINE` set an hour ahead, nothing stops, and the Train_Record holds the Cap_Deadline and the Stop_Time in UTC, 15 minutes apart.
    - An Exposure_Fix that the test sets shows in the written colours as the bake gives it.
    - Listing the Job_Folder before and after a done run shows only `work/train/train.json`, `out/train.ply` and `out/reports/train.json` new.
    - The scene `FakeGpu` was given holds every image in file order with the model's poses, and its Start_Splats sit at the model's points.
    - More points than `max_splats`, with `set_train(max_splats={"fast": 5})`, gives the Req 3.11 warning and 5 Start_Splats.
    - A stop after 3 Iterations still writes and checks the splat and adds the Req 7.8 warning. A second run on the same Job leaves only its own files.
    - A NaN splat from the stand-in is dropped: the report holds the Req 8.9 warning with a count of 1, and the run is done with a checked `out/train.ply` one splat shorter than the Start_Splats, which `counts.splats` gives.
    - _Requirements: 1.19, 1.20, 3.11, 6.7, 7.7, 7.8, 8.2, 8.6, 8.8, 8.9, 8.11, 8.12, 10.1, 10.2, 10.4, 10.5, 10.6, 10.7, 11.8, 14.4, 14.5_

  - [ ] 7.5 Write unit tests for failed runs and the CLI in `tests/test_train_step.py`
    - The order of A9: a bad setting with a bad deadline gives the setting's message, a bad deadline with no `work/cameras/` gives the deadline's, a missing `work/cameras/` gives the camera solve message with no probe call, and a refusing probe with a corrupt `points3D.bin` gives the GPU message.
    - Each Expected_Failure that needs no GPU, through `run`, has `expected: true`, a `readable` that names what its requirement says, and no "Traceback": a bad setting, `max_splats` below `start_scale_neighbours` + 1, a missing `[train]` key, a bad deadline, each missing input path, no CUDA, gsplat 1.5.2, a small GPU, a missing torch with `splatpipe.train.gpu_api.importlib` replaced as in 6.5, a corrupt model file, a missing image, a bad mask, too few points, cameras that didn't move, no point in front of any camera, and points that all sit on one spot.
    - A deadline already past fails with the Req 7.9 message. `GpuOutOfMemory` from the 3-NN search and from training gives the Req 2.12 message and the report holds the counts worked out so far.
    - `NonFiniteLoss` gives `expected: false`, the general bug message as `readable`, and the Iteration and the Loss_Terms in `technical`.
    - With every splat NaN, the run fails with the PLY_Check's "holds no splats" message, the failed report holds the Req 8.9 warning with the count, and there's no `out/train.ply`. A failed run removes an `out/train.ply` left by an earlier run.
    - `splatpipe.train.__main__.main(["--job", job_id])` prints `train: done` and returns 0 for a done run, and prints the Readable_Error on stderr with no traceback and returns 1 for a failed one.
    - _Requirements: 1.2, 2.1, 2.2, 2.12, 4.16, 7.1, 7.3, 7.9, 8.1, 8.7, 8.9, 9.6, 9.7, 9.8, 10.3, 10.5, 11.2, 11.3, 11.7, 14.4, 14.6_

  - [ ] 7.6 Run the tests and commit
    - Run `uv run pytest`. Every test passes, `tests/test_guard.py` included.
    - On `feat/splat-training`: `git add splatpipe/train/__init__.py splatpipe/train/__main__.py tests/conftest.py tests/test_train_step.py`, then commit those files with the message "Train: the step and its CLI".
    - _Requirements: 14.3, 14.11, 14.12_

- [ ] 8. The GPU_Module and the Trainer
  - Files: `splatpipe/train/gpu.py` (new). `tests/test_imports.py` and `tests/test_cloud_imports.py` (changed).
  - _Requirements: 2.2, 2.4, 2.12, 3.2, 3.13, 4.1, 4.2, 4.3, 4.5, 4.7, 4.9, 4.10, 4.12, 4.13, 4.14, 4.15, 4.16, 4.17, 4.18, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 5.10, 6.1, 6.2, 6.3, 7.6, 7.10, 8.2, 9.11, 9.12, 10.1, 14.3, 14.7, 14.8, 14.12_

  - [ ] 8.1 Write `splatpipe/train/gpu.py`
    - Load `skill::gsplat-training` and `skill::splat-files` with `load_resource` before writing code. Follow gsplat 1.5.3's API exactly as the skill and the design's "gpu.py (the GPU_Module and the Trainer)" section give it.
    - The file starts with the licence note the gsplat-training skill gives for code adapted from `examples/simple_trainer.py`, from `# Adapted from gsplat v1.5.3, examples/simple_trainer.py` to `# limitations under the License.`, then the design's own "Changes:" lines.
    - It imports torch, gsplat for `__version__`, `MCMCStrategy` and `rasterization` from gsplat, `FusedSSIMMap` from fused_ssim, numpy, the standard library and the CPU modules of `splatpipe/train/`, and nothing else.
    - It never calls `rasterization_inria_wrapper()`, `rasterization_2dgs_inria_wrapper()` or `export_splats()`, and names none of them anywhere in the file, comments and docstrings included, because 8.2's check refuses the names. It holds no code from Inria's repo or from the MCMC or effective-rank papers' repos.
    - `probe()`, `nearest_distances()`, `train()` and `peak_memory_bytes()`, as the design gives them. `train` holds no state between calls and runs the design's numbered steps in order: the seed, the views on the GPU, the `ParameterDict`, the rates from `means_lr_schedule`, one Adam per key, the Exposure_Fixes with their own Adam kept out of the strategy's optimisers, `MCMCStrategy` with no `noise_lr` or `min_opacity`, then the loop.
    - Each Iteration, in this order: `should_stop` first; `rasterization()` with `packed=False`, `rasterize_mode="classic"`, the Near_Plane, the Iteration's SH_Degree and no `backgrounds`; the Exposure_Fix; the five Loss_Terms from one `_TERMS` table, each skipped at a weight of 0 and `erank` skipped before the Erank_Start; `loss.backward()`, skipped when every weight is 0; one `torch.stack(...).tolist()` that raises `NonFiniteLoss` before any optimiser step; each optimiser's `step()` and `zero_grad(set_to_none=True)`; the positions' rate set to `rates[i + 1]`; then `strategy.step_post_backward(..., step=i, info=info, lr=rates[i + 1])`.
    - The terms are worked out as the design's table gives them: `l1` and `ssim` over the kept pixels' 3 channels, `ssim` from `FusedSSIMMap.apply(C1, C2, img1, img2, "same", True)` with both images set to 0 at Masked_Pixels, and `erank` from `torch.log_softmax(2 × scales, -1)` with D3's formula, written fresh.
    - `torch.cuda.OutOfMemoryError` becomes `GpuOutOfMemory` with the Iteration, or `None` before the first. The splats come back to the CPU as float32 numpy arrays, read through `params[...]` because the strategy swaps the tensors.
    - _Requirements: 2.4, 2.12, 3.2, 3.13, 4.1, 4.2, 4.3, 4.5, 4.7, 4.9, 4.10, 4.12, 4.13, 4.14, 4.15, 4.16, 4.17, 4.18, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 5.10, 6.1, 6.2, 6.3, 7.6, 7.10, 8.2, 9.11, 10.1_

  - [ ] 8.2 Write `gpu.py`'s source checks and the fresh-process import in `tests/test_imports.py`
    - Read `splatpipe/train/gpu.py` as text and never import it. It starts with the skill's Apache-2.0 note, with `# Copyright 2025 Nerfstudio Team` and `# Licensed under the Apache License, Version 2.0` before its first import, and the text holds none of the names `rasterization_inria_wrapper`, `rasterization_2dgs_inria_wrapper` and `export_splats`.
    - It's the only file in `splatpipe/train/` that imports torch, gsplat or fused_ssim, and it imports all three.
    - `splatpipe.train.gpu_api.GPU_MODULE` names this file: its dots turned into folders, plus `.py`, is `splatpipe/train/gpu.py`, so the module `load_gpu` asks for is the one these checks read.
    - A fresh process that runs `import splatpipe.train` and imports each of its CPU modules has none of torch, gsplat and fused_ssim in `sys.modules`.
    - _Requirements: 2.2, 4.14, 4.15, 8.2, 9.11, 9.12, 14.7_

  - [ ] 8.3 Change the splatpipe GPU check in `tests/test_cloud_imports.py`
    - `test_nothing_under_splatpipe_imports_cloud_boto3_or_a_gpu_package` keeps refusing `cloud` and boto3 in every file under `splatpipe/`, and every GPU package in every file but `splatpipe/train/gpu.py`, which may import torch, gsplat and fused_ssim and no other GPU package (A14).
    - The test's docstring and the module docstring say so.
    - _Requirements: 14.8_

  - [ ] 8.4 Run the tests and commit
    - Run `uv run pytest`. Every test passes, `tests/test_guard.py` included.
    - On `feat/splat-training`: `git add splatpipe/train/gpu.py tests/test_imports.py tests/test_cloud_imports.py`, then commit those files with the message "Train: the GPU module and its trainer".
    - _Requirements: 14.3, 14.12_

- [ ] 9. The smoke test's training check
  - Files: `splatpipe/train/smoke.py` and `tests/test_train_smoke.py` (new). `config.toml`, `cloud/config.py`, `cloud/smoke_runner.py`, `tests/fake_aws.py`, `tests/test_cloud_config.py`, `tests/test_cloud_smoke_runner.py`, `docs/pipeline.md` and `tests/test_docs.py` (changed).
  - _Requirements: 12.1, 12.2, 12.3, 12.4, 12.5, 12.6, 12.7, 12.8, 12.9, 12.10, 12.11, 12.12, 12.13, 13.11, 14.3, 14.10, 14.12_

  - [ ] 9.1 Write `splatpipe/train/smoke.py`
    - Load `skill::gsplat-training` and `skill::splat-files` with `load_resource` before writing code.
    - `SmokeSizes`, `smoke_settings()`, `smoke_scene()` and `run_smoke_training()`, as in the design's "smoke.py" section and the Smoke_Scene row of its decisions table. The module imports only the standard library, numpy and `splatpipe`, and no torch.
    - `smoke_settings` takes `[jobs].default_preset`'s `[train]` settings, replaces `steps`, `max_splats`, `refine_start_step`, `refine_every` and `sh_degree_interval` with the sizes, and checks the result with `read_settings`.
    - `smoke_scene` draws everything from one generator seeded with `[train]`'s `seed`: the PINHOLE camera, the cameras on the unit circle looking at the origin, the points in the cube of half-side 0.25, the images, and a keep mask on image 0 that leaves out its left half.
    - `run_smoke_training(config, sizes, scratch, gpu)` runs `check_gpu` on `gpu.probe()`, then the step's own `need_points`, `training_size`, `build_scene`, `choose_points`, `nearest_distances`, `start_splats` and `train`. A `GpuOutOfMemory` from either GPU call becomes `out_of_memory_failure`, as in the step.
    - It refuses a run of fewer Iterations than `steps` with a `PipelineError` saying how many ran, bakes the Mean_Exposure, writes every splat the Trainer gave back to `scratch/train.ply` with `write_ply`, runs `check_ply`, and gives back the design's log line with the numbers filled in, the refines counted with `refine_iterations`.
    - It calls no `drop_non_finite`, so a splat with a NaN or an infinity fails its PLY_Check, because the Smoke_Scene should never make one.
    - _Requirements: 12.2, 12.3, 12.7, 12.8, 12.9_

  - [ ] 9.2 Add the seven `[cloud.smoke]` keys to the configs
    - In `config.toml`, append `train_images = 4`, `train_points = 200`, `train_splats = 400`, `train_steps = 30`, `train_refine_start_step = 10`, `train_refine_every = 10` and `train_sh_degree_interval = 5` to `[cloud.smoke]` after `video_seconds`, each with the comment the design's "The `[cloud.smoke]` keys (A11)" gives it.
    - In `cloud/config.py`, `SmokeChecks` gains seven `int` fields with the keys' names after `video_seconds`, and `_read_smoke_checks` reads each with `_whole(table, key, label(key))`.
    - In `tests/fake_aws.py`, `_base_cloud_mapping`'s smoke table gains the seven keys at their A11 values.
    - _Requirements: 12.4, 12.5_

  - [ ] 9.3 Add the `training` check to `cloud/smoke_runner.py`
    - `_check_training(context)`, as the design gives it, turns `context.cloud_config.smoke_checks` into a `SmokeSizes` and returns `run_smoke_training(context.config, sizes, context.scratch, load_gpu())`.
    - The module imports `SmokeSizes` and `run_smoke_training` from `splatpipe.train.smoke` and `load_gpu` from `splatpipe.train.gpu_api` at the top, and calls `load_gpu()` inside the function, so `gpu.py` and torch are imported only there and a test can patch `cloud.smoke_runner.load_gpu`. The module imports no GPU package of its own.
    - `SMOKE_CHECKS` gains `SmokeCheck("training", _check_training)` just before `cloud-runner`, which stays last, and the module docstring's list of checks names it. The check has no timeout of its own: the log records its seconds, which shows Req 12.11 on the machine.
    - _Requirements: 12.1, 12.10, 12.11_

  - [ ] 9.4 Change the smoke test part of `docs/pipeline.md`, with its doc test
    - In "The smoke test", add a bullet for the training check: the real Trainer trains a tiny made-up scene on the GPU for a few Iterations, with one MCMC refine, then the PLY_Writer and the PLY_Check run on its splat, with its numbers in `[cloud.smoke]`. The last sentence says the training check is the first of the checks later specs add.
    - In `tests/test_docs.py`, a test checks that the part names the training check, `train_steps` and `[cloud.smoke]`.
    - _Requirements: 12.13, 13.11_

  - [ ] 9.5 Write unit tests for the smoke flow in `tests/test_train_smoke.py`
    - `smoke_scene` with the A11 numbers from the real config passes the step's own checks on the CPU: `need_points`, and `build_scene` with a Scene_Extent of 1.1 and every point between 0.64 and 1.36 in front of every camera. One image has a mask that keeps pixels, and the same seed gives the same scene.
    - `smoke_settings` takes `fast`, the default Preset, with the five values replaced and every other `[train]` value as the config has it.
    - With the real config and the A11 numbers, the Schedules turn `erank` on at Iteration 7, reach SH_Degree 3 at Iteration 15, and refine at Iteration 20 only.
    - `run_smoke_training` with a `FakeGpu` writes and checks `train.ply` in the scratch folder and gives the log line with its numbers. It fails with a readable message when the stand-in runs fewer Iterations, runs out of memory, gives a NaN splat or reports gsplat 1.5.2. The NaN splat is written, not dropped, so the check fails at its PLY_Check.
    - _Requirements: 12.2, 12.3, 12.6, 12.7, 12.8, 12.9, 14.10_

  - [ ] 9.6 Write unit tests for the smoke keys and the smoke runner's check
    - In `tests/test_cloud_config.py`, the fixed valid mapping gains the seven keys, each key gets corruption cases for a float, a bool and 0 that name it as `<key> for [cloud.smoke]`, and the real-config test checks their A11 values. The cloud-jobs spec's Property 2 strategy, `valid_cloud_mappings` and `corruptible_targets`, gains them as whole numbers.
    - In `tests/test_cloud_smoke_runner.py`, the order test has `training` just before `cloud-runner`.
    - The laptop test of the GPU checks includes `training`, with `splatpipe.train.gpu_api.importlib` replaced so its `import_module` raises `ModuleNotFoundError` for torch, as in 6.5. The check fails on its own with the Req 2.2 message and the next check still runs, and no test imports `gpu.py`.
    - A new test runs the check with `cloud.smoke_runner.load_gpu` patched to give a `FakeGpu`. It passes, the stand-in got the `[cloud.smoke]` numbers, and the check gives its log line.
    - _Requirements: 12.1, 12.4, 12.5, 12.9, 12.10, 12.13, 14.10_

  - [ ] 9.7 Run the tests and commit
    - Run `uv run pytest`. Every test passes, `tests/test_guard.py` included.
    - On `feat/splat-training`: `git add splatpipe/train/smoke.py config.toml cloud/config.py cloud/smoke_runner.py tests/fake_aws.py tests/test_cloud_config.py tests/test_cloud_smoke_runner.py tests/test_train_smoke.py docs/pipeline.md tests/test_docs.py`, then commit those files with the message "Train: the smoke test's training check".
    - This task runs no Smoke_Test. The Smoke_Command uploads the Code_Archive, so the new check runs on the current Machine_Image with no Image_Build, but the Start_Command's gate still passes on a Smoke_Marker written before the check existed (A12). So once the spec is built, the Smoke_Test runs again before the first training Job, as a separate paid run that needs the person's yes. Kiro first tells them the machine (`smoke_instance_type`, a g5.xlarge), the expected time and the rough cost, and only after their yes runs `uv run python -m cloud.smoke` with `APPROVED_SPEND=1` on that one command.
    - _Requirements: 12.12, 14.3, 14.12_

- [ ] 10. The docs and the Step_List
  - Files: `splatpipe/run.py`, `docs/pipeline.md`, `.kiro/skills/gsplat-training/SKILL.md`, `tests/test_run.py`, `tests/test_frames_step.py`, `tests/test_imports.py`, `tests/test_train_settings.py`, `tests/test_docs.py` and `tests/test_cloud_docs.py`, all changed.
  - _Requirements: 9.9, 11.4, 11.5, 11.9, 11.10, 11.11, 13.1, 13.2, 13.3, 13.4, 13.5, 13.6, 13.7, 13.8, 13.9, 13.10, 13.12, 13.13, 13.14, 13.15, 14.3, 14.9, 14.12_

  - [ ] 10.1 Write `## Step: train` in `docs/pipeline.md`
    - Load `skill::gsplat-training` and `skill::splat-files` with `load_resource` before writing.
    - Put it after `## Step: frames`, as the design's "The Pipeline_Doc" section gives it: a short paragraph with the CLI, `uv run python -m splatpipe.train --job <id> [--preset P]`, `splatpipe.train.run(job_id, preset=None)` and the order of the work from A9, then `### What it reads`, `### What it writes`, `### Config section`, `### Step report fields` and `### Files in out/`.
    - What it reads: the Camera_Solve_Output as the contract the camera solve spec writes, the masks matched by stem, where a value below 128 counts as black, so the camera solve step should write only 0 and 255, what the step refuses, `SPLATPIPE_CAP_DEADLINE` and the settings.
    - What it writes: the Splat_Layout's 59 fields, `work/train/` and the Train_Record's fields.
    - Config section: every key of `[train]` and its Preset tables with its table, value and meaning, and the Step_Needs: 4 vCPUs, 32 GB of memory, and a 24 GB GPU for `fast` and `default` or a 48 GB one for `quality`, of which the step checks only the GPU's memory, against the GPU_Need.
    - Step report fields: each count as `counts.<name>`, each timing as `timings.<name>`, the three warnings, the kinds of Expected_Failure, and that a loss that stops being finite is a bug, with the Iteration and the Loss_Terms in `technical`. The Req 8.9 warning gives how many splats with a NaN or an infinity were left out of `out/train.ply`, `counts.splats` counts the splats written, and a run that leaves out every splat fails at the PLY_Check's "holds no splats", with the warning in its report.
    - Files in out/: the step removes `out/train.ply` when it starts and writes it whole once the PLY_Check passes. The Clean_Step reads it from there, and it's the Job's result when the Clean_Step fails or doesn't run. The largest Splat_Budget, 3,000,000 splats, is within the 8,000,000 splats the Clean_Step takes, its `max_source_splats` (D4).
    - _Requirements: 13.1, 13.2, 13.3, 13.4, 13.5, 13.6, 13.7_

  - [ ] 10.2 Change the four other parts of `docs/pipeline.md`
    - "The step list": the three new sentences of the design's table, "The list holds the frames step and then the train step, both required" with both entries, "The camera solve spec puts its entry between the two, and a later spec adds the clean step's optional entry after the train step's, so the order becomes frames, cameras, train, clean" and "Until the camera solve spec adds its entry, a job runs the frames step and then fails at the train step, because the camera solve output is missing", in place of the three old ones. The planned steps' line "Optional steps: depth priors and clean and export." becomes "Optional steps: depth priors and clean."
    - "The cap deadline": "The train step reads it and stops training early, so that `out/train.ply` is written and checked `clean_reserve_minutes` before the cap deadline", in place of "No current step reads it; the training spec will."
    - "What each step asks the machine for": the Training row becomes "4 vCPUs, 32 GB RAM and a GPU with 24 GB (fast and default) or 48 GB (quality)", and the row "Clean and export | 8 vCPUs, 8 GB RAM" becomes "Clean | 8 vCPUs, 8 GB RAM".
    - The frames section's "What it writes": "the camera solve step passes it on to training as one mask per image in `work/cameras/masks/`", in place of "the training loss reads the same file" (A13).
    - _Requirements: 13.8, 13.9, 13.10, 13.12_

  - [ ] 10.3 Change the gsplat-training skill's line on the clean step
    - In `.kiro/skills/gsplat-training/SKILL.md`, the bullet on the step that reads the trained splat, which names `out/splat.ply`, becomes the design's sentence: "The clean step reads `out/train.ply`, which the training step writes, as a binary PLY in the usual 3DGS layout (its requirements, A1), and `export_splats(format="ply")` writes that layout. An MCMC splat has at most `cap_max` Gaussians, and the clean step refuses one with more than its `max_source_splats`, 8,000,000, so keep the cap at or under that setting." Nothing else in the skill changes.
    - _Requirements: 13.13_

  - [ ] 10.4 Change the Step_List in `splatpipe/run.py`
    - `STEP_LIST = (StepEntry("frames", "splatpipe.frames", True), StepEntry("train", "splatpipe.train", True))`. The comment above it says the camera solve spec puts its entry between the two and a later spec adds the Clean_Step's optional entry after the Train_Step's, so the order becomes frames, cameras, train, clean. It also says the Clean_Step stays out of the list for now (D5) and the Fake_Step stays out.
    - `_NO_STEPS` and the refusal of an empty list that a caller gives stay as they are.
    - _Requirements: 11.4, 11.5_

  - [ ] 10.5 Update the tests that depend on the Step_List
    - In `tests/test_run.py`, `test_the_step_list_holds_the_frames_step_and_is_the_default_list` becomes `test_the_step_list_holds_the_frames_and_train_steps_and_is_the_default_list` and asserts the list of 10.4.
    - In `tests/test_frames_step.py`, the real-ffmpeg `run_job` test passes `steps=(StepEntry("frames", "splatpipe.frames", True),)`, so it still expects `done`, and its comments and the module docstring no longer say the frames step is the list's only entry. A new test runs `run_job` with the real Step_List, `fake_tools` and the `walk()` frames: the Job ends `failed` at `train`, the frames step is `done`, and the Job_Summary's `readable_error` is the Req 1.2 message naming `work/cameras/images`. Any other test that the run shows relying on the frames-only list gets the frames entry alone, the same way.
    - In `tests/test_imports.py`, the Step_List check's docstring names the Train_Step, whose `[train]` it now loads from the real config for each Preset. The two docstrings that say which spec can decide scipy, the module's and `test_pyproject_and_uv_lock_hold_numpy_and_headless_opencv_and_no_gpu_or_plain_opencv`'s, say "the clean spec".
    - In `tests/test_train_settings.py`, the Fallback_List test: with the real Config and the real Step_List, `select_candidates` keeps every type on each Preset's Fallback_List, with the needs from the Start_Command's own `step_needs_over`.
    - _Requirements: 9.9, 11.5, 11.9, 11.10, 11.11, 14.9_

  - [ ] 10.6 Write the doc tests in `tests/test_docs.py` and `tests/test_cloud_docs.py`
    - In `tests/test_docs.py`, the step list test looks for the three new sentences and checks that the three old ones are gone. The frames test looks for the new mask words in place of "the training loss reads the same file".
    - New train tests: `## Step: train` comes after `## Step: frames`, with its five parts. Its Config section names every key of the real `[train]`, read by path, its tables, and "4 vCPUs", "32 GB", "24 GB" and "48 GB". Its Step report fields part names `counts.<name>` for each of the 13 counts and `timings.<name>` for each of the 4 timings. Its Files in out/ part names `out/train.ply`, "3,000,000 splats", "8,000,000 splats" and `max_source_splats`.
    - The cap deadline part names `clean_reserve_minutes` and no longer says "No current step reads it". The gsplat-training skill names `out/train.ply` and no longer `out/splat.ply`.
    - In `tests/test_cloud_docs.py`, `STEP_NEEDS_PLAN["Training"]` becomes `("4 vCPUs", "32 GB RAM", "24 GB", "48 GB")`, and the key `"Clean and export"` becomes `"Clean"`, with its `("8 vCPUs", "8 GB RAM")` unchanged.
    - _Requirements: 13.1, 13.7, 13.8, 13.9, 13.10, 13.12, 13.13, 13.14, 13.15_

  - [ ] 10.7 Run the whole suite one last time and commit
    - Run `uv run pytest`. This is the spec's final full-suite run: every test passes, `tests/test_guard.py` included, on the CPU with no network.
    - Check that `git log --oneline 1898e51..HEAD -- pyproject.toml uv.lock` lists none of this spec's commits, so no package was added (A1).
    - On `feat/splat-training`: `git add splatpipe/run.py docs/pipeline.md .kiro/skills/gsplat-training/SKILL.md tests/test_run.py tests/test_frames_step.py tests/test_imports.py tests/test_train_settings.py tests/test_docs.py tests/test_cloud_docs.py`, then commit those files with the message "Train: pipeline doc section and the step list entry".
    - _Requirements: 14.3, 14.12_

## Notes

- Every test sub-task is required, as `.kiro/steering/rules.md` asks. None is marked optional, so no `*` appears anywhere.
- The last sub-task of each task is its checkpoint: the whole suite, then the commit by name. So the plan has no separate checkpoint tasks and stays at the design's 10, and 10.7 is the final full-suite run.
- Every acceptance criterion in `requirements.md` is named by at least one task. Each of the design's 9 correctness properties has its own property test sub-task, with Hypothesis: Property 1 in 3.4, Properties 3 and 4 in 2.4 and 2.5, Properties 2 and 5 in 4.5 and 4.6, and Properties 6 to 9 in 5.3 to 5.6. The drop of non-finite splats (D6) has unit tests and no property, as the design keeps it: 4.4 tests `drop_non_finite`, and 7.4 and 7.5 test the step's drop and its warning.
- Task 8's code runs only on the GPU, so its tests read its source and check the imports. The Trainer first runs in the Smoke_Test's `training` check, and Req 12.11, the check's time on the g5.xlarge, shows only in that run's log.
- No laptop test imports `gpu.py`. 6.5, 7.5 and 9.6 replace `splatpipe.train.gpu_api.importlib` instead, which gives the same Req 2.2 message on any machine, with torch installed or not, and 6.5 and 8.2 check that `load_gpu` asks for the module whose file the source checks read. The review kept this and changed the design's build order to match.
- No task runs anything paid. After the spec is built, the Smoke_Test re-run of 9.7 is a separate paid run, started only after the person's yes.
- The graph runs the tasks in order, one at a time, so each task's test run and commit see only finished work. Inside a task, sub-tasks that write different files share a wave, tests come after the code they test, and each task's commit gates the next task's first sub-tasks.

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "1.2", "1.3"] },
    { "id": 1, "tasks": ["1.4", "1.5"] },
    { "id": 2, "tasks": ["1.6"] },
    { "id": 3, "tasks": ["2.1", "2.2"] },
    { "id": 4, "tasks": ["2.3", "2.6"] },
    { "id": 5, "tasks": ["2.4"] },
    { "id": 6, "tasks": ["2.5"] },
    { "id": 7, "tasks": ["2.7"] },
    { "id": 8, "tasks": ["3.1", "3.2"] },
    { "id": 9, "tasks": ["3.3"] },
    { "id": 10, "tasks": ["3.4"] },
    { "id": 11, "tasks": ["3.5"] },
    { "id": 12, "tasks": ["4.1", "4.2"] },
    { "id": 13, "tasks": ["4.3", "4.4"] },
    { "id": 14, "tasks": ["4.5"] },
    { "id": 15, "tasks": ["4.6"] },
    { "id": 16, "tasks": ["4.7"] },
    { "id": 17, "tasks": ["5.1"] },
    { "id": 18, "tasks": ["5.2"] },
    { "id": 19, "tasks": ["5.3"] },
    { "id": 20, "tasks": ["5.4"] },
    { "id": 21, "tasks": ["5.5"] },
    { "id": 22, "tasks": ["5.6"] },
    { "id": 23, "tasks": ["5.7"] },
    { "id": 24, "tasks": ["6.1", "6.2", "6.3"] },
    { "id": 25, "tasks": ["6.4", "6.5"] },
    { "id": 26, "tasks": ["6.6"] },
    { "id": 27, "tasks": ["7.1", "7.2", "7.3"] },
    { "id": 28, "tasks": ["7.4"] },
    { "id": 29, "tasks": ["7.5"] },
    { "id": 30, "tasks": ["7.6"] },
    { "id": 31, "tasks": ["8.1"] },
    { "id": 32, "tasks": ["8.2", "8.3"] },
    { "id": 33, "tasks": ["8.4"] },
    { "id": 34, "tasks": ["9.1", "9.2", "9.4"] },
    { "id": 35, "tasks": ["9.3", "9.5"] },
    { "id": 36, "tasks": ["9.6"] },
    { "id": 37, "tasks": ["9.7"] },
    { "id": 38, "tasks": ["10.1", "10.3", "10.4"] },
    { "id": 39, "tasks": ["10.2", "10.5"] },
    { "id": 40, "tasks": ["10.6"] },
    { "id": 41, "tasks": ["10.7"] }
  ]
}
```
