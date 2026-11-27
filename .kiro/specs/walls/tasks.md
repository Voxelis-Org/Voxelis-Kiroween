# Implementation Plan: walls

## Overview

This plan builds the Walls_Step in `splatpipe/walls/` and changes one function of the Train_Step so it trains on what the Walls_Step writes, in 10 tasks, each building on the one before: the `[walls]` settings, the inputs and the time budget; the bas-relief change, its normals and the warp; the score images and the score table; the scene scale, the start depth and the source frames; the fits, the three passes and the self-check; the facets and one frame's whole solve; the first cross-frame stages (the neighbour frames, the projected values, the fuse, the correction, the despeckle and the per-frame choice); the second (the world planes, the projection fallback, the plane registration and the flatten); the step itself, with its worker pool, outputs, report and CLI; and last the Train_Step's folder choice, the Step_List entry, the docs, the new expected times and the end-to-end Job test. Each task writes its code with that code's unit tests and Hypothesis property tests, runs the whole suite and commits its files by name. The code is Python 3.11 on numpy, scipy and OpenCV, which the Project_File already has, so `pyproject.toml` and `uv.lock` don't change and no package is added (Req 21.13).

No task starts until the user says go. Once every task is done, Samarth runs a real Job with the Walls_Step himself. That run is paid, it isn't a task, and nothing in this plan starts it.

The property tests the request asks for by name:

- Property 1, in 2.4: a Bas_Relief_Change keeps every pixel on its own ray, in front of the camera.
- Property 2, in 2.5: a Bas_Relief_Change keeps a plane flat, and the Changed_Normal is the new plane's normal.
- Property 3, in 2.6: the Identity_Change (1, 0, 0, 0) gives back the same inverse depths bit for bit, and the same normals.
- Property 6, in 7.3: the despeckle never moves a pixel within `despeckle_tolerance` (3%) of its window's median, nor one whose window has too few valid pixels.

The design adds three more: Property 4, in 2.7, composing two changes equals applying them in turn; Property 5, in 1.10, the Solve_Order lists every index once; and Property 7, in 9.7, a Walls_Prior round-trips through the Prior_Format and keeps the mask. The textured-plane test of Req 22.2, in 5.6, checks that the passes find the right scale within 1% and that the Self_Check passes there and fails on a plain plane. It's a unit test, not a property.

Five choices in the order matter:

- **The pure maths comes first.** Tasks 2 to 8 build every rule of the method as numpy, scipy and OpenCV functions on arrays, each tested on tiny made-up arrays and scenes, before any code reads a Job, writes a file or starts a process. Task 1 gives them the settings they read, the inputs they're fed from and the time budget.
- **The step comes after everything it calls.** Task 9 writes the worker pool, the scratch folder, the outputs, the report and `work`, once every stage they run is tested. Its tests run the whole step through `run()` with the in-process pool and a fake clock.
- **The Step_List entry and the Train_Step's change come last.** Task 10 puts the Walls_Step in `STEP_LIST` once the step is whole, so no Job runs a half-built step, and the tests that assert the Step_List change in that task (Req 22.10). The Train_Step starts reading `work/walls/` in the same task.
- **The docs come with the Step_List entry.** Task 10 writes the `## Step: walls` section of `docs/pipeline.md`, with its citation of Kriegman and Belhumeur (ECCV 1998), every other part of the Pipeline_Doc this spec changes, the licence cell and the new expected times, once the code they describe is done. No earlier task changes `docs/pipeline.md` or `docs/third-party-licences.md`.
- **Each test scene comes with the first task that uses it.** Task 2's warp test uses the plane scene `tests/camera_files.py` already has. Task 5 adds the textured plane, task 6 the box room and task 9 the helper that writes a whole Job's inputs for the Walls_Step.

Calls this plan makes where the design's first draft left room or disagreed with itself, C1 to C13, and two the review added, C14 and C15. The review folded each into the design, which now says the same. Each is small to change:

- **C1, the mask reader.** Req 2.12 wants each Mask_File read with the Train_Step's rules and refusals, but that reader is the private `_keep` in `splatpipe/train/files.py`, whose all-masked message names `[train]`. Task 1 makes it public as `read_keep_mask`, with a `section` argument that only that message uses, so the Train_Step's behaviour and messages stay the same. The design's "Changes outside `splatpipe/walls/`" table now lists this file.
- **C2, the Agreeing_Values.** The glossary's "within `agree_tolerance` of the nearest of them" is read as the nearest to the camera: the smallest Projected_Value at the pixel, as a Projected_Value is itself the nearest depth a Neighbour_Frame gives.
- **C3, where the Scene_Scale is worked out.** The design's first draft had the preparation pass write the Start_Depths to the scratch folder and also had `solve_frame` work them out. This plan works out every Prior_Frame's Scale_Points, Scene_Scale and both Start_Depths on the main process, in task 9's preparation pass, so a Timed_Out_Frame has them too, for the Walls_Record (Req 7.6) and the Sparse_Error.
- **C4, `solve.py` over two tasks.** Task 6 writes `FrameSolve` and `solve_arrays()`, the whole Frame_Solve on arrays in memory. Task 9 adds `SolveTask`, `solve_frame()`, which loads the scratch arrays and calls `solve_arrays`, `load_frame()` and `load_source()`, which read the scratch arrays back, and the worker tasks of the cross-frame stages. The loaders live in `solve.py`, beside their one caller, because they build `solve.py`'s `FrameArrays`: in `inputs.py` they would make `inputs.py` and `solve.py` import each other. `inputs.py` holds the scratch folder's file names, which its preparation pass writes and the loaders read.
- **C5, the per-frame choice.** Task 7 puts `sparse_error()` and `frame_choice()` in `crossframe.py`, pure, where the design's first draft had them in `outputs.py`. `walls_prior()` and `depth_prior_copy()` stay in `outputs.py`, in task 9, with Property 7.
- **C6, `InlinePool`.** It lives in `splatpipe/walls/pool.py`, as the design's module table says, so the stand-in and the real pool share one cutoff rule. The design's first draft put it in `tests/conftest.py` in its "Stand-ins" section; `tests/conftest.py` holds only the `walls_pool` fixture and the dead-worker factory.
- **C7, the real pool.** The design's first draft named `multiprocessing.get_context("spawn").Pool`. That pool waits forever for a task whose worker process was killed, so it can't give Req 5.7's failure, and Python 3.11's `ProcessPoolExecutor` can't end a running task at the Solve_Cutoff (Req 4.6). Task 9 builds the pool from the `spawn` context's `Process` workers fed by queues instead, which keeps `spawn`, the one-thread limits and each worker's `cv2.setNumThreads(1)`.
- **C8, the clock.** `work` reads the time through `splatpipe.walls.utc_now`, which is `splatpipe.train.deadline.utc_now`, and hands it to the pool, so a test's `monkeypatch` gives the whole step a fake clock.
- **C9, the walls fallback warning.** The design's first draft of `_read_priors` reused `_BROKEN_PRIORS`, which ends "Training ran without depth priors." That's wrong when training falls back to `work/depth/` (Req 20.2), so task 10 adds a warning text of its own for broken walls priors.
- **C10, the Change_Fit's starts.** Req 5.9 wants the same priors bit for bit whatever the worker count, and `[walls]` holds no seed, so the `fit_starts` − 1 starts other than the Identity_Change are a fixed pattern, and each start's simplex is spread by `fit_start_spread`, as task 5 says.
- **C11, plane normals.** A fitted plane normal's sign is arbitrary, so every angle between two plane normals is taken from |n₁·n₂|, and offsets are compared once the normals face the same way.
- **C12, the Fallback_List test.** Req 22.9's check that every type on each Preset's Fallback_List still meets the Step_Needs needs the Walls_Step in the real Step_List, so it comes in task 10, not with task 1's settings tests.
- **C13, test files.** Three test files the design's "Tests by file" table didn't name: `tests/test_walls_inputs.py` (task 1), `tests/test_walls_solve.py` (task 6) and, in task 10, a row in `tests/test_cloud_docs.py`'s `STEP_NEEDS_PLAN` for the new Step_Needs row. `tests/test_walls_workers.py` starts in task 1 with the import checks, so each later file is checked as it lands.
- **C14, the same bits whatever the workers (added at review).** Req 5.9 and the two-worker test of 9.8 want the same Walls_Priors bit for bit from `InlinePool`, in the test process, whose BLAS may use many threads, and from spawned workers on one thread, and a BLAS call's last bits can change with its thread count. So code that runs in a worker keeps every sum over many pixels in numpy's own loops (elementwise arithmetic, `np.sum`, `np.median`, `np.einsum` with its default `optimize=False`), applies a pose to many points elementwise and never with `@` or `np.dot`, and calls LAPACK only on matrices of at most 4 × 4. `InlinePool` runs its tasks with `cv2.setNumThreads(1)`, as the workers do.
- **C15, thread limits that don't leak (added at review).** The Walls_Step runs in the same process as the Train_Step and the Clean_Step after it, so `open_pool` sets the one-thread variables in `os.environ` only while it starts each worker, which inherits them, and then puts the main process's own values back, unset ones unset.

Some files grow over several tasks, so each piece comes with the first task that uses it:

- `splatpipe/walls/__init__.py`: task 1 makes it with only the package docstring, and task 9 writes the step.
- `splatpipe/walls/inputs.py`: task 1 writes the input checks and readers, and task 9 the preparation pass and the scratch folder's file names.
- `splatpipe/walls/solve.py`: task 6 writes `FrameArrays`, `FrameSolve` and `solve_arrays`, and task 9 `SolveTask`, `solve_frame`, the scratch loaders and the stage tasks (C4).
- `config.toml`: task 1 adds `[walls]` and its Preset tables, and task 10 changes `expected_minutes` in `[cloud.presets.*]`.
- `tests/conftest.py`: task 1 adds `WALLS`, `WALLS_PRESETS`, `walls_text`, `config_text`'s `walls` argument and `set_walls`, and task 9 the `walls_pool` fixture and the dead-worker factory.
- `tests/camera_files.py`: task 5 adds the textured plane, task 6 the box room and task 9 the walls Job writer.
- `tests/test_walls_workers.py`: task 1 writes the import checks, and task 9 adds the pool tests and the two-worker test.
- `tests/test_walls_settings.py`: task 1 writes it, and task 10 adds the Fallback_List test.
- `docs/pipeline.md`, `tests/test_docs.py` and `docs/third-party-licences.md`: task 10 only.

## Working rules for every task

- Work on the existing `feat/walls` branch, never on `main`. Build on it as it is, with no rebase, reset or force push.
- The only new step folder is `splatpipe/walls/`. Outside it, only the files in the design's "Changes outside `splatpipe/walls/`" table, `splatpipe/train/files.py` (C1), `tests/test_cloud_docs.py` (C13) and the test files each task names change or are made. `pyproject.toml`, `uv.lock`, everything under `app/`, `cloud/image.py`, `cloud/select.py`, `cloud/start.py`, `cloud/runner.py`, the Smoke_Test (`cloud/smoke.py` and `cloud/smoke_runner.py`) and the Fallback_Lists stay exactly as they are (Req 21.13, D16, D20). Nothing in the Depth_Step, the Prior_Format's fields in `splatpipe/train/priors.py`, or the Train_Step's losses, Start_Splats and counts changes.
- Every file in `splatpipe/walls/` imports only the standard library, numpy, scipy, cv2 and `splatpipe`, and `pool.py` adds `multiprocessing`. No file imports torch, gsplat, MoGe, `splatpipe.depth`, `cloud` or boto3. `splatpipe.walls` imports `splatpipe.train`'s CPU modules, and nothing in `splatpipe.train` imports `splatpipe.walls` (Req 1.11, 22.8).
- cv2 only decodes JPEG and PNG stills, converts colours, scales, blurs, box-filters and remaps images, and encodes the Walls_Demos as JPEGs. It never reads or writes video, and no file names a cv2 video class (Req 2.13, 18.10).
- Code that runs in a worker, in tasks 2 to 9, keeps its sums over many pixels out of BLAS, as C14 says: elementwise numpy, `np.sum`, `np.median` and `np.einsum` with its default `optimize=False`, poses applied elementwise, and LAPACK only on matrices of at most 4 × 4 (Req 5.9).
- Every size, count, share, angle, tolerance, time and quality comes from `[walls]` or its Preset table. Code holds only the Walls_Fixed_Constants of the requirements' glossary (Req 3.7).
- Write the code fresh from the requirements and the design, on the library APIs of numpy, scipy and OpenCV. No code from any other repo: none from Kriegman and Belhumeur's paper or any repo that implements it, no multi-view stereo code, and nothing from Inria's gaussian-splatting repo (Req 1.12).
- Load `skill::moge-depth`, `skill::gsplat-training` and `skill::splat-files` before writing any code in the tasks that touch the Prior_Format, the normals or the Train_Step: tasks 1, 2, 4, 6, 8, 9 and 10.
- Nothing here costs money. No task runs anything on AWS, an `aws` command, `cloud.start`, `cloud.image` or `cloud.smoke`, starts a Job, an Image_Build or a Smoke_Test, or sets `APPROVED_SPEND`. No task reads anything in the job bucket.
- Tests run on the CPU with no GPU and no network, on tiny synthetic data that each test builds: models, frames, masks, priors and scenes of a few dozen to a few thousand pixels, under `tmp_path` or a `tempfile.TemporaryDirectory()`. Every Job is made in the test config's Jobs_Folder, never in `data/`, and nothing a test makes goes into git (Req 22.12). Nothing heavy runs on the laptop: the only tests that start real worker processes are task 9's pool tests and its two-worker test, each with 2 workers on a tiny scene.
- Property tests use Hypothesis with `@settings(max_examples=100, deadline=None)`, and each docstring says `Feature: walls, Property N: <title>`. A property test that writes files makes a `tempfile.TemporaryDirectory()` per example. Each oracle is written in the test and never calls the function it checks.
- Every unit test and property test sub-task is required. None is optional (Req 22.4).
- No task adds a package or a Licence_List row. If one turns out to need a package, stop and ask the user first.
- The last sub-task of each task is its checkpoint. Run `uv run pytest` and fix what fails until every test passes, `tests/test_guard.py` included. Then stage the task's files by name with `git add <file> ...`, never with `git add -A`, `git add .` or a folder, and commit only those files with `git commit -m "<message>" -- <file> ...`. Leave `docs/devlog.md` out: the devlog hook adds and commits its own line after each task.
- If a hook blocks something it shouldn't, say so and fix the hook. Never work around it.

## Tasks

- [ ] 1. The `[walls]` settings, the inputs and the time budget
  - Files: `splatpipe/walls/__init__.py`, `splatpipe/walls/settings.py`, `splatpipe/walls/budget.py`, `splatpipe/walls/inputs.py`, `tests/test_walls_settings.py`, `tests/test_walls_budget.py`, `tests/test_walls_inputs.py` and `tests/test_walls_workers.py` (new). `config.toml`, `splatpipe/train/files.py` and `tests/conftest.py` (changed).
  - _Requirements: 1.11, 1.12, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 2.10, 2.11, 2.12, 2.13, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 3.10, 4.2, 4.3, 4.4, 4.5, 4.10, 4.12, 22.1, 22.3, 22.4, 22.8, 22.9, 22.12_

  - [ ] 1.1 Add `[walls]` and its Preset tables to `config.toml`
    - Put `[walls]` after `[depth.quality]` and before `[train]`, with the 83 Shared_Settings of the requirements' settings table, in its order and at its values, and a short comment on the line above every key.
    - The comments of the D22 keys (`scale_min_points`, `fit_start_spread`, `fit_max_iterations`, `fit_x_tolerance`, `fit_f_tolerance`, `band_max_coherence`, `tile_min_windows`, `register_min_pixels`, `tukey_constant`, `tukey_iterations` and `demo_frames`) and of `workers` and `demo_jpeg_quality` say that the value is a default untested on real footage. `mask_threshold`'s comment says it matches `[train]`'s.
    - `[walls.fast]`, `[walls.default]` and `[walls.quality]` hold exactly `max_minutes` (20, 40, 60) and `train_reserve_minutes` (40, 60, 110), each with its comment.
    - `min_vcpus` (16) and `min_memory_gb` (32) are the Step_Needs that `cloud/select.py` already reads, and the cloud code doesn't change. `[cloud.presets.*]` stays as it is until task 10.
    - _Requirements: 3.1, 3.2, 3.3, 3.8, 3.9_

  - [ ] 1.2 Write `splatpipe/walls/settings.py`
    - Load `skill::moge-depth`, `skill::gsplat-training` and `skill::splat-files` before writing code.
    - Make `splatpipe/walls/` with an `__init__.py` that holds only the package docstring for now. Task 9 writes the step in it.
    - `WallsSettings`, a frozen dataclass with one field per key, the 83 Shared_Settings and the 2 Preset_Settings, and `read_settings(settings, section, preset)`, as in the design's "Settings (`settings.py`)" section, with its own small table of kinds, so it imports nothing private from another step. The module imports only the standard library and `splatpipe.errors`.
    - Each key is read with `settings[key]`, so a missing key raises the Step_Runner's own message naming the section, the table and the key (Req 3.6). Each value's `type()` is checked exactly: a whole number is a TOML integer, a number is a TOML integer or float, and a bool is never a number.
    - A wrong kind is an `ExpectedFailure`: "The setting `<key>` in `<table>` is `<value>`, but it must be `<kind>`. Change it to `<kind>`.", with the value shown as TOML writes it, the table `[walls]` for a Shared_Setting and `[walls.<preset>]` for `max_minutes` and `train_reserve_minutes`.
    - `band_sigmas` is a list of 2 or more numbers, the first 0, each above the one before. `window_size` and `despeckle_window` are odd whole numbers of at least 3, the angles are above 0 and at most 90, and every other kind is as the settings table gives it.
    - Once every kind passes, the cross-checks of Req 3.5, in the requirement's order, each refused with an `ExpectedFailure` that names both settings and their values: each `*_min_scale` below 1 and its `*_max_scale` above 1, `source_fallback_seconds` at most `source_min_seconds`, `min_sources` at most `max_sources`, `register_end_tolerance` at most `register_start_tolerance`, `despeckle_min_valid` at most `despeckle_window`², and `finish_minutes` below the Preset's `max_minutes`.
    - `min_vcpus` and `min_memory_gb` are checked and kept, and the step never uses them (Req 3.8).
    - _Requirements: 3.4, 3.5, 3.6, 3.7, 3.8_

  - [ ] 1.3 Write `splatpipe/walls/budget.py`
    - `walls_times()`, `solve_cutoff()`, `may_start()` and `solve_order()`, as in the design's "The time budget (`budget.py`, pure)" section. The module imports only the standard library and reads no clock: every time comes in as an argument (Req 4.10).
    - `walls_times(start, cap_deadline, max_minutes, train_reserve_minutes)` gives the Walls_Stop_Time, the earlier of `start` + `max_minutes` and `cap_deadline` − `train_reserve_minutes`, or `start` + `max_minutes` with no Cap_Deadline, and `"max_minutes"` or `"train_reserve_minutes"`, whichever set it, a tie going to `"max_minutes"`.
    - `solve_cutoff(stop, finish_minutes)` is the Walls_Stop_Time − `finish_minutes`, and `may_start(now, cutoff)` is `now < cutoff`.
    - `solve_order(n)` is Req 4.4's order: 0, then n − 1 when n is above 1, then the midpoints from a first-in-first-out queue of intervals that starts with (0, n − 1), for every n of 1 or more.
    - _Requirements: 4.2, 4.3, 4.4, 4.5, 4.10_

  - [ ] 1.4 Make the Train_Step's mask reader public in `splatpipe/train/files.py`
    - `_keep` becomes `read_keep_mask(folder, name, size, width, height, threshold, section="train")`, and `_read_view` calls it, so the Train_Step reads and refuses its Mask_Files exactly as before (C1).
    - `section` names the Config section in the all-masked message, "every pixel is below mask_threshold in [<section>]", and changes nothing else. Its docstring says the Walls_Step reads its Mask_Files with it too.
    - _Requirements: 2.12_

  - [ ] 1.5 Write `splatpipe/walls/inputs.py`
    - `FrameInput`, `check_inputs()`, `read_frames_record()`, `read_inputs()`, `score_size()`, `score_image()` and `score_mask()`, as in the design's "Inputs (`inputs.py`)" section. The module imports only the standard library, numpy, cv2, `splatpipe.errors` and the Train_Step's `colmap`, `files`, `priors` and `scene` modules. Task 9 adds the preparation pass and the scratch loaders.
    - `FrameInput` is frozen and holds one Prior_Frame's name, its Pick_Number, its Frame_Time, its `ModelImage` with its pose, its Depth_Prior and its `PriorEntry`.
    - `check_inputs(job_folder)` reads no file. It checks, in this order, `work/depth/depth.json`, `work/cameras/images/`, each of the Sparse_Model's three files and `work/frames/frames.json`, and refuses the first one missing with the message of Req 2.2, 2.3 or 2.4, each naming the path and the step that writes it. The `depth.json` message says that the depth step has to run first and that the Job goes on and trains as it would without the walls step.
    - `read_frames_record(path)` reads with `json`, uses only `picks` and, of each Pick, only `number`, `file` and `seconds`, and refuses each case of Req 2.7 with an `ExpectedFailure` that names the field and says to run the frames step on the Job again. `seconds` must be a finite int or float, never a bool.
    - `read_inputs(job_folder)` calls `read_model` on `work/cameras/sparse/`, then `read_prior_set(work/depth, model)`, whose `BrokenPriors` becomes an `ExpectedFailure` with the reader's message, then `read_frames_record`. It joins each Prior_Frame to the Pick whose `file` is its name, giving it the Pick's `number` and `seconds`, and refuses a Prior_Frame with no Pick with Req 2.8's message, which names the image, says that `frames.json` doesn't match the camera solve and says to run the Job again with `--fresh`. It gives the model, the Depth_Prior_Set and the `FrameInput`s in Pick_Number order, and ignores every image in `images.bin` that has no Depth_Prior (Req 2.10).
    - `score_size(camera, long_edge)` is `training_size(camera.width, camera.height, long_edge)`. `score_image` decodes with `read_training_image` and scales to the Score_Size with `scaled_down`. `score_mask` gives `read_keep_mask(..., section="walls")` with `mask_threshold` from `[walls]` when `work/cameras/masks/<stem>.png` is a file, and every pixel kept otherwise.
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 2.10, 2.11, 2.12, 2.13_

  - [ ] 1.6 Add `[walls]` and `set_walls` to `tests/conftest.py`
    - `WALLS`, the 83 Shared_Settings at their real values with their real TOML types; `WALLS_PRESETS`, `max_minutes` and `train_reserve_minutes` for each Preset; `walls_text(walls)`; and a `walls` argument to `config_text`, which writes `[walls]` and its three Preset tables between `[depth]`'s tables and `[train]`.
    - A `set_walls(**values)` fixture built like `set_depth`, which rewrites the test config with each named key changed for that test. A Preset_Setting given to it is set in all three Preset tables. The module docstring names the new helpers.
    - _Requirements: 3.10, 22.12_

  - [ ] 1.7 Write the walls import checks in `tests/test_walls_workers.py`
    - A test reads every `splatpipe/walls/*.py`, found with glob, with `ast`, and checks that each imports only names in `sys.stdlib_module_names`, numpy, scipy, cv2 and `splatpipe`, imports nothing from `splatpipe.depth`, and names none of cv2's video classes (`VideoCapture`, `VideoWriter` and `VideoWriter_fourcc`). A failure gives the file and the line. Each later file is checked as it lands.
    - A test reads every `splatpipe/train/*.py` with `ast` and checks that none imports `splatpipe.walls`.
    - A fresh process that imports `splatpipe.walls` and each of its modules, found with glob, has none of torch, gsplat and moge in `sys.modules`.
    - Task 9 adds the pool tests and the two-worker test to this file. The module docstring says so.
    - _Requirements: 1.11, 2.13, 22.8_

  - [ ] 1.8 Write unit tests for the settings in `tests/test_walls_settings.py`
    - The real config, read by path, has `[walls]` with exactly the 83 Shared_Settings at the settings table's values and TOML types, and three Preset tables with exactly `max_minutes` and `train_reserve_minutes` at Req 3.3's values. `step_settings("walls", preset)` loads exactly 85 keys for each Preset. The comment check in `tests/test_config.py` already covers the new keys.
    - Parametrised over every key: a value outside its kind, and a bool where a number belongs, gives the Req 3.4 message naming the key, its table and the kind. The edges of each range pass and the values just past them don't, among them `window_size` 3 and 2, `mask_threshold` 0, 255 and 256, `demo_jpeg_quality` 1 and 101, and a `band_sigmas` that doesn't start at 0, has one sigma or doesn't rise.
    - Each cross-check of Req 3.5 is refused on its own, naming both settings and their values.
    - A missing key gives the Step_Runner's message naming `[walls]`, `[walls.<preset>]` and the key. `set_walls` changes one key for one test, and the next test sees the test config's values again.
    - The Fallback_List test comes in task 10, with the Step_List (C12).
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.10, 22.1, 22.9_

  - [ ] 1.9 Write unit tests for the time budget in `tests/test_walls_budget.py`
    - `walls_times` with no Cap_Deadline gives start + `max_minutes`, with a far Cap_Deadline the same, with a near one Cap_Deadline − `train_reserve_minutes`, and with both equal `"max_minutes"`, each with the limit that set it.
    - `solve_cutoff` and `may_start` just before, at and just after the Solve_Cutoff, with a fake clock.
    - `solve_order` for N = 1, 2, 3 and 7 gives [0], [0, 1], [0, 2, 1] and [0, 6, 3, 1, 4, 2, 5], spelled out in the test.
    - _Requirements: 4.2, 4.3, 4.4, 4.5, 4.10, 22.1_

  - [ ] 1.10 Write the property test for Property 5 in `tests/test_walls_budget.py`, with Hypothesis
    - **Property 5: The Solve_Order lists every index once**
    - **Validates: Requirements 4.12**
    - Draw N from 1 to 1,000, with 1, 2 and 3 drawn often, as the design's Property 5 strategy says.
    - Check that `sorted(solve_order(N))` is `list(range(N))`, that the first index is 0, and that the second is N − 1 when N is above 1.
    - _Requirements: 4.12, 22.3_

  - [ ] 1.11 Write unit tests for the inputs in `tests/test_walls_inputs.py`
    - Build each Job with `tests/camera_files.py`'s plane scene, a `frames.json` whose Picks name its images, and a Depth_Prior_Set written with the Prior_Writer.
    - `check_inputs` refuses each of the six missing paths in turn, naming the path and the step that writes it, and opens no file.
    - `read_frames_record` refuses JSON that isn't an object, a `picks` that isn't a list of objects, each missing field, a `number` of −1, 1.5 and `true`, a `file` that isn't a string, a `seconds` that isn't a number, and a `number` or a `file` given twice, each naming the field.
    - `read_inputs` gives each Prior_Frame its Pick_Number and Frame_Time, leaves out an image with no Depth_Prior, refuses a Prior_Frame whose name is no Pick's `file` with the `--fresh` message, and gives the reader's message for a broken `depth.json` and a cut-short `points3D.bin`.
    - `score_image` scales a frame to the Score_Size, keeps a camera smaller than `score_long_edge` at its size, and refuses a missing frame, junk bytes and a frame of the wrong size with the Train_Step's messages. `score_mask` keeps every pixel with no Mask_File, reads one with `[walls]`'s `mask_threshold`, and refuses an all-black Mask_File with a message naming `[walls]`. The Train_Step's own mask tests pass unchanged.
    - No message holds "Traceback".
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 2.10, 2.11, 2.12, 22.1, 22.12_

  - [ ] 1.12 Run the tests and commit
    - Run `uv run pytest`. Every test passes, `tests/test_guard.py` and `tests/test_train_files.py` included.
    - On `feat/walls`: `git add config.toml splatpipe/train/files.py splatpipe/walls/__init__.py splatpipe/walls/settings.py splatpipe/walls/budget.py splatpipe/walls/inputs.py tests/conftest.py tests/test_walls_settings.py tests/test_walls_budget.py tests/test_walls_inputs.py tests/test_walls_workers.py`, then commit only those files with `git commit -m "Walls: the [walls] settings, the inputs and the time budget" -- <the same files>`.
    - _Requirements: 22.4, 22.12_

- [ ] 2. The bas-relief change, its normals and the warp
  - Files: `splatpipe/walls/geometry.py` and `tests/test_walls_geometry.py` (new).
  - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8, 6.9, 6.10, 9.3, 9.11, 14.2, 22.1, 22.3, 22.4, 22.12_

  - [ ] 2.1 Write the bas-relief change in `splatpipe/walls/geometry.py`
    - Load `skill::moge-depth`, `skill::gsplat-training` and `skill::splat-files` before writing code.
    - `normalised_coords()`, `rays()`, `median_inverse_depth()`, `apply_change()`, `depth_factor()`, `compose()` and `changed_normal()`, as in the design's "The bas-relief change (`geometry.py`, pure)" section, all in float64. The module imports only the standard library, numpy and `splatpipe.train.scene`.
    - `normalised_coords(camera, width, height)` centres pixel (i, j) of a w × h grid at ((i + 0.5) · W / w, (j + 0.5) · H / h) in the Shared_Camera's pixels, the Prior_Format's rule, and gives x = (u − cx) / fx and y = (v − cy) / fy, so the same ray is described at the Score_Size and at the Prior_Size.
    - `apply_change` computes `a * inverse + m * (b * x + c * y + d)` exactly so, so the Identity_Change gives back the input inverse depths bit for bit (Req 6.2). A pixel whose new inverse depth isn't finite and above 0 gets no new depth, and every function marks a pixel with no new depth or no new normal with NaN.
    - `compose(first, second, m1, m2)` is Req 6.5's formula. `changed_normal` is a·n + m·z·(n·r)·(b, c, d), set to unit length and turned round when its dot product with r is above 0, with no new normal for a length of 0 or a part that isn't finite (D4).
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6_

  - [ ] 2.2 Write the projection and the warp in `splatpipe/walls/geometry.py`
    - `back_project()`, `project()`, `warp_maps()` and `zbuffer_project()`, as in the design's "Back-projection, projection and the warp" section, with the Image_Poses from `world_to_camera` and the Shared_Camera at the Score_Size from `scaled_intrinsics`.
    - `warp_maps` gives OpenCV's sample coordinates (u − 0.5, v − 0.5), the depth the warp predicts in the Source_Frame, and `inside`, true where that depth is above 0 and the point lands on [0, W) × [0, H) of the Source_Frame. It applies the two poses to the points elementwise, never with `@` or `np.dot` (C14).
    - `zbuffer_project` lands each point on a pixel by the Depth_Step's rule, (min(⌊u · w / W⌋, w − 1), min(⌊v · h / H⌋, h − 1)), drops points behind the camera or outside the image, keeps the nearest depth per pixel with `np.lexsort`, and leaves a pixel no point lands on unfilled.
    - _Requirements: 9.3, 9.11, 14.2_

  - [ ] 2.3 Write unit tests for the geometry in `tests/test_walls_geometry.py`
    - `apply_change`, `depth_factor`, `compose` and `changed_normal` on hand cases worked out in the test. A change whose new inverse depth is 0, negative or infinite gives that pixel no depth and no normal. A normal facing away from the camera is turned round, and a raw normal of length 0 gives none.
    - `normalised_coords` at a grid half the camera's size gives the rays of the full-size grid's pixel centres it covers, and `median_inverse_depth` uses only the Covered_Pixels.
    - On `tests/camera_files.py`'s plane scene, `warp_maps` takes a pixel of one image at its true depth on the plane to the sub-pixel where another image sees the same point, within 10⁻⁶ pixels, and gives that point's camera-space z there. `inside` is false for a point behind the Source_Frame's camera and for one outside its image.
    - `zbuffer_project` keeps the nearer of two points on one pixel, clamps a point at the right and bottom edges by the Depth_Step's rule, drops a point behind the camera and leaves an unhit pixel unfilled.
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 9.3, 14.2, 22.1_

  - [ ] 2.4 Write the property test for Property 1 in `tests/test_walls_geometry.py`, with Hypothesis
    - **Property 1: A change keeps every pixel on its own ray, in front of the camera**
    - **Validates: Requirements 6.7**
    - Draw as the design's Property 1 strategy says: fx and fy from 200 to 3,000, cx and cy inside the image, any pixel, z from 0.1 to 50, and a change with a above 0 and small |b|, |c| and |d|, dropped with `assume` when the new inverse depth isn't finite and above 0.
    - The oracle builds the old point z · r and the new point z′ · r with its own float64 code. Check that their cross product is within 10⁻⁹ of their lengths' product and that both z and z′ are above 0. So a change moves every pixel along its own ray.
    - _Requirements: 6.7, 22.3_

  - [ ] 2.5 Write the property test for Property 2 in `tests/test_walls_geometry.py`, with Hypothesis
    - **Property 2: A change keeps a plane flat and carries its normal**
    - **Validates: Requirements 6.8, 6.4**
    - Draw as the design's Property 2 strategy says: a plane with a unit normal and an offset away from 0, 4 to 40 pixels whose rays meet it in front of the camera, their depths read off the plane, and a change whose new inverse depths there are finite and above 0.
    - The oracle fits a plane to the changed points with an SVD of the centred points. Check that every point lies within 10⁻⁶ of its distance from the camera centre of that plane, and that each Changed_Normal equals the fitted normal facing the camera within 10⁻⁶. So a flat wall stays flat.
    - _Requirements: 6.4, 6.8, 22.3_

  - [ ] 2.6 Write the property test for Property 3 in `tests/test_walls_geometry.py`, with Hypothesis
    - **Property 3: The Identity_Change gives the depths and normals back**
    - **Validates: Requirements 6.9, 6.2**
    - Draw as the design's Property 3 strategy says: a depth map from 1 × 1 to 32 × 32 pixels of depths from 0.1 to 50, unit normals turned to face the camera, any Normalised_Coordinates and m from 0.01 to 10.
    - Check that `apply_change` with (1, 0, 0, 0) gives back 1.0 / z at every pixel bit for bit, compared through `struct.pack("<d", value)`, and that `changed_normal` gives back each normal within 10⁻⁶. So (1, 0, 0, 0) gives back the same depth.
    - _Requirements: 6.2, 6.9, 22.3_

  - [ ] 2.7 Write the property test for Property 4 in `tests/test_walls_geometry.py`, with Hypothesis
    - **Property 4: Composing two changes equals applying them in turn**
    - **Validates: Requirements 6.10, 6.5**
    - Draw as the design's Property 4 strategy says: a depth map and its median inverse depth m₁, a first change, the changed map and its median m₂, and a second change, both results finite and above 0.
    - The oracle applies the two changes in turn with its own formula. Check that `apply_change` of the first map with `compose(c1, c2, m1, m2)` and m₁ gives the same inverse depths within a relative error of 10⁻⁹.
    - _Requirements: 6.5, 6.10, 22.3_

  - [ ] 2.8 Run the tests and commit
    - Run `uv run pytest`. Every test passes, `tests/test_guard.py` included.
    - On `feat/walls`: `git add splatpipe/walls/geometry.py tests/test_walls_geometry.py`, then commit only those files with `git commit -m "Walls: the bas-relief change, its normals and the warp" -- <the same files>`.
    - _Requirements: 22.4, 22.12_

- [ ] 3. The score images and the score table
  - Files: `splatpipe/walls/images.py`, `splatpipe/walls/scoring.py`, `tests/test_walls_images.py` and `tests/test_walls_scoring.py` (new).
  - _Requirements: 9.1, 9.2, 9.4, 9.5, 9.6, 9.7, 9.8, 9.9, 9.10, 9.11, 10.5, 11.4, 11.5, 11.6, 11.8, 22.1, 22.4, 22.12_

  - [ ] 3.1 Write `splatpipe/walls/images.py`
    - `luma()`, `high_pass()`, `bands()`, `band_weights()`, `band_mix()`, `window_centres()`, `window_sums()` and `window_ncc()`, as in the design's "Images and scoring" section. The module imports only the standard library, numpy and cv2.
    - `luma` is `cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)` as float64 from 0 to 255. `high_pass` takes away `cv2.GaussianBlur(grey, (0, 0), high_pass_sigma)`. `bands` gives one octave Band per pair of neighbouring `band_sigmas`, a sigma of 0 being the Luma itself, so `[0, 1, 2, 4, 8]` gives 4.
    - `band_weights` caps each Coherence at `band_max_coherence` and gives w = √(ρ / ((1 − ρ²) · var)), and 0 where ρ isn't above 0 or the variance is 0 (A4).
    - `window_centres` gives the grid every `window_step` pixels from ⌊`window_size` / 2⌋ in from the top left, keeping only windows inside the image. `window_sums` reads `cv2.boxFilter(image, cv2.CV_64F, (W, W), normalize=False)` at the centres, and counts masked or invalid pixels with corner reads of an integral image.
    - `window_ncc` samples the Source_Frame's image at the warp's maps with `cv2.remap(..., cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)` and gives each window's zero-mean NCC, cov / √(var_I · var_J) from the five sums, with 0 where either variance is 0.
    - _Requirements: 9.1, 9.4, 10.5_

  - [ ] 3.2 Write `splatpipe/walls/scoring.py`
    - `Windows`, `ScoreSource`, `score_windows()`, `occluded()`, `valid_pairs()`, `table_scales()`, `score_table()`, `guess_score()` and `tile_scores()`, as in the design's "Images and scoring" section. The module imports only the standard library, numpy, `splatpipe.walls.geometry` and `splatpipe.walls.images`.
    - `ScoreSource`, which the design's signatures leave open, is frozen and holds what one Source_Frame brings: its name, its world-to-camera matrix, its High_Pass, its Bands and its Score_Mask at the Score_Size, its Start_Depth, k × its Prior_Depth, and its Covered_Pixels at the Prior_Size, and its Source_Half, A or B (D24).
    - `score_windows` keeps the windows of Req 9.2: inside the image, all Depth_Pixels, all kept by the Score_Mask, and a High_Pass variance of at least `flat_variance`.
    - `occluded` and `valid_pairs` make the tests of Req 9.5 and 9.6, with the occlusion test comparing the warp's predicted depth with the Source_Frame's Start_Depth, made once per Score_Table at the Current_Depth and held at every Table_Scale, a window centre that lands where the Source_Frame has no Start_Depth, on no Covered_Pixel of it, counting as occluded, and the flat test always on the warped High_Pass (D24, A1, A2).
    - `table_scales(min_scale, max_scale, step)` gives ⌈ln(max / min) / step⌉ + 1 scales evenly spaced in ln(s), both ends included (A3).
    - `score_table` warps each Source_Frame at each Table_Scale, scores the image it's asked for, the High_Pass or the Band_Mix with the Prior_Frame's own weights, and gives C(p, s), with 0 where p has no Valid_Pair. When asked, it gives the two Half_Tables from the same warps.
    - `guess_score` looks C(p, s(p)) up by linear interpolation in ln(s) and gives −∞ when any s(p) isn't finite or leaves the Table_Range, or |b|, |c| or |d| is above `fit_bound` (D11).
    - `tile_scores` gives the two Tile_Scores, against × and ÷ `z_scale`, and the Tiles each counted, with the rules of Req 11.6 and 11.8: no Z with fewer than 2 Tiles, and +∞ for a standard error of 0 when mean_A − mean_B is above 0 (A6).
    - _Requirements: 9.2, 9.5, 9.6, 9.7, 9.8, 9.9, 9.10, 9.11, 11.4, 11.5, 11.6, 11.8_

  - [ ] 3.3 Write unit tests for the score images in `tests/test_walls_images.py`
    - `luma` equals `cv2.cvtColor` of known colours. `high_pass` of a constant is 0, and of a step is ± around it. `bands` of `[0, 1, 2, 4, 8]` gives 4 Bands that add up to the Luma minus its blur at sigma 8.
    - `window_ncc` is 1 for a window matched to itself, −1 for its negative, near 0 for two unrelated noise windows, and 0 for a flat window. `window_sums` equals a plain loop's sums on a small image.
    - `band_weights` equals the closed form, caps ρ at `band_max_coherence`, and gives 0 where ρ is 0 or below or the variance is 0.
    - _Requirements: 9.1, 9.4, 10.5, 22.1_

  - [ ] 3.4 Write unit tests for the scoring in `tests/test_walls_scoring.py`
    - `score_windows` leaves out a window that crosses the border, one with a pixel that isn't a Depth_Pixel, one with a masked pixel and a flat one.
    - `valid_pairs` leaves out a pair for each of its reasons on its own: a pixel behind the Source_Frame, one outside it, a masked Source pixel, a flat warped window and an occluded centre, whose predicted depth is more than `occlusion_tolerance` off the Source_Frame's Start_Depth, and a centre on no Covered_Pixel, where the Source_Frame has no Start_Depth.
    - `table_scales` gives 94, 43, 43 and 97 scales for the real First, Second, Third and Facet settings, evenly spaced in ln(s) with both ends included.
    - `score_table` on a hand-made case sums the Window_Scores of the Valid_Pairs, is 0 where a window has none, and its two Half_Tables add up to it.
    - `guess_score` hits a node exactly, interpolates halfway between two in ln(s), and rejects a factor outside the Table_Range, a non-finite one and a |b| just above `fit_bound`.
    - `tile_scores` equals a z-test worked out in the test, gives no Z with 1 counting Tile, +∞ for a standard error of 0 with mean_A above mean_B, and a failure for it with mean_A at or below mean_B.
    - _Requirements: 9.2, 9.5, 9.6, 9.7, 9.8, 9.9, 9.10, 11.6, 11.8, 22.1_

  - [ ] 3.5 Run the tests and commit
    - Run `uv run pytest`. Every test passes, `tests/test_guard.py` included.
    - On `feat/walls`: `git add splatpipe/walls/images.py splatpipe/walls/scoring.py tests/test_walls_images.py tests/test_walls_scoring.py`, then commit only those files with `git commit -m "Walls: the score images and the score table" -- <the same files>`.
    - _Requirements: 22.4, 22.12_

- [ ] 4. The scene scale, the start depth and the source frames
  - Files: `splatpipe/walls/sources.py` and `tests/test_walls_sources.py` (new).
  - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 22.1, 22.4, 22.12_

  - [ ] 4.1 Write `splatpipe/walls/sources.py`
    - Load `skill::moge-depth`, `skill::gsplat-training` and `skill::splat-files` before writing code.
    - `scale_points()`, `scene_scale()`, `start_depth()`, `Candidate`, `overlap()`, `image_motion()`, `source_rank()`, `SourceChoice` and `choose_sources()`, as in the design's "Scene scale and sources (`sources.py`, pure)" section. The module imports only the standard library, numpy, `splatpipe.walls.geometry` and the Train_Step's `colmap`, `scene` and `prior_views` modules.
    - `scale_points` takes a Prior_Frame's Observed_Points through `observed_rows`, keeps those in front of its camera whose projection with the Shared_Camera at full size lands inside the Frame on a Covered_Pixel by the Depth_Step's pixel rule, and gives each one's camera-space z and the Prior_Depth at its pixel (A7).
    - `scene_scale` gives k = exp(the median of the r with |r − r̃| at most `scale_mad_gate` × D), with D the unscaled median absolute deviation, only the r at the median kept when D is 0, and k = 1 with fewer than `scale_min_points` Scale_Points, together with the Scale_Point count.
    - `start_depth` gives the Prior_Size Start_Depth, k × the Prior_Depth on the Covered_Pixels, and the Score_Size one, k / the Depth_Prior's `inverse_depth` scaled up with `prior_taps` and `upscale`, with the Depth_Pixels from `prior_pixel_mask(mask, None, taps)`, because the Score_Mask is a test of its own (Req 7.5).
    - `Candidate` is frozen and holds what source choice needs from another Prior_Frame: its name, Pick_Number, Frame_Time and world-to-camera matrix.
    - `choose_sources` takes the candidates at least `source_min_seconds` away in time, drops those below `source_min_overlap` or `source_min_motion`, ranks by Overlap × min(Image_Motion, `source_motion_cap`), ties to the lower Pick_Number, and takes each in rank order whose Pick_Number is at least `source_min_gap` from every one already chosen, up to `max_sources`. With fewer than `min_sources`, it chooses again with `source_fallback_seconds`. It gives the `SourceChoice`s in rank order and the time it used.
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 8.1, 8.2, 8.3, 8.4, 8.5, 8.6_

  - [ ] 4.2 Write unit tests for the scene scale and the sources in `tests/test_walls_sources.py`
    - On `tests/camera_files.py`'s plane scene with a Depth_Prior that holds each Observed_Point's own z ÷ 1.3 at the pixel it lands on, as `plane_moge_output` does, `scene_scale` gives k = 1.3 within 10⁻⁹. One point moved far off is dropped by the gate, 9 Scale_Points give k = 1, and a set whose deviation is 0 keeps only the points at the median.
    - `scale_points` leaves out a point behind the camera, one outside the Frame, one on a pixel that isn't Covered and a keypoint that names no point.
    - `start_depth` is k × the Prior_Depth at the Prior_Size, and at a Score_Size twice the Prior_Size it gives the scaled-up depth and leaves out each pixel `prior_pixel_mask` leaves out.
    - `overlap`, `image_motion` and `source_rank` equal values worked out in the test for crafted cameras. `choose_sources` drops a candidate too close in time, one below the Overlap and one below the Image_Motion, ranks a tie to the lower Pick_Number, skips a candidate 1 Pick_Number from one already chosen, stops at `max_sources`, and drops to `source_fallback_seconds` only when the first pass gives fewer than `min_sources`, with the time it used.
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 22.1_

  - [ ] 4.3 Run the tests and commit
    - Run `uv run pytest`. Every test passes, `tests/test_guard.py` included.
    - On `feat/walls`: `git add splatpipe/walls/sources.py tests/test_walls_sources.py`, then commit only those files with `git commit -m "Walls: the scene scale, the start depth and the source frames" -- <the same files>`.
    - _Requirements: 22.4, 22.12_

- [ ] 5. The fits, the three passes and the self-check
  - Files: `splatpipe/walls/fit.py`, `splatpipe/walls/passes.py` and `tests/test_walls_passes.py` (new). `tests/camera_files.py` (changed).
  - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7, 10.8, 11.1, 11.2, 11.3, 11.4, 11.5, 11.7, 11.8, 15.2, 15.3, 22.1, 22.2, 22.4, 22.12_

  - [ ] 5.1 Write `splatpipe/walls/fit.py`
    - `change_fit()` and `tukey_change()`, as in the design's "The fits (`fit.py`, pure)" section. The module imports only the standard library, numpy, `scipy.optimize` and `splatpipe.walls.scoring`.
    - `change_fit` runs `scipy.optimize.minimize(method="Nelder-Mead")` on −Guess_Score from `fit_starts` starts, with `maxiter=fit_max_iterations`, `xatol=fit_x_tolerance` and `fatol=fit_f_tolerance`. The starts are a fixed pattern, with no random draw (C10): the Identity_Change, then start j for j from 1 moves number (j − 1) mod 4 of (a, b, c, d) by +`fit_start_spread` in the first round of four and by −`fit_start_spread` in the next, and so on. Each start's `initial_simplex` is the start and the start moved by `fit_start_spread` along each of the four numbers.
    - A rejected guess scores −∞. `change_fit` keeps the best guess it evaluated over every start, so its Guess_Score is at least the Identity_Change's, and gives it with its score.
    - `tukey_change(inverse0, target, m, x, y, constant, iterations)` is iteratively reweighted least squares on the columns (inverse0, m·x, m·y, m), with Tukey's weights at `tukey_constant` median absolute deviations of the residuals for `tukey_iterations` rounds and no `fit_bound` (A12). With a deviation of 0 the weights stay as they were. Each round builds the 4 × 4 weighted normal equations with `np.einsum` and solves them with `np.linalg.lstsq`, which also copes with the rank-deficient case of a wall seen square on (C14).
    - _Requirements: 10.3, 15.2, 15.3_

  - [ ] 5.2 Write `splatpipe/walls/passes.py`
    - `SolveResult`, `ThirdPass`, `first_pass()`, `second_pass()`, `band_coherences()`, `third_pass()` and `self_check()`, as in the design's "The three passes and the self-check (`passes.py`, pure)" section. The module imports only the standard library, numpy and the walls pure modules.
    - `first_pass` takes the Table_Scale with the highest Σ_p C(p, s) on the High_Pass, the one nearest 1 in ln(s) on a tie, and gives its scale and s × the Start_Depth. `second_pass` runs `change_fit` on its High_Pass table around the First_Pass's depth.
    - `band_coherences` gives each Band's mean Window_Score over the Valid_Pairs at the Second_Pass's depth, and `third_pass` scores the Band_Mix with `band_weights`, or the High_Pass when every weight is 0, building the Score_Table and the two Half_Tables around the Second_Pass's depth and running `change_fit` on the Score_Table.
    - The Solved_Change is the three passes' changes composed with `compose`, relative to the Start_Depth, so k stays a number of its own (Req 10.8).
    - `self_check` refits each Source_Half on its Half_Table, works out the Halves_Difference over the Covered_Pixels and the two Tile_Scores at the answer, and passes only with at least `min_sources` Source_Frames, a Halves_Difference below `halves_tolerance` and both Tile_Scores above `min_z`. It gives the `SolveResult` with the numbers the caller hands it: k, the Scale_Point count, the Source_Frame count and the time they were chosen with.
    - _Requirements: 10.1, 10.2, 10.4, 10.5, 10.6, 10.7, 10.8, 11.1, 11.2, 11.3, 11.4, 11.5, 11.7, 11.8_

  - [ ] 5.3 Add the textured plane to `tests/camera_files.py`
    - A textured-plane scene: a plane seen by about 12 cameras of about 128 × 96 pixels from a slow walk past it, each at a Frame_Time 0.5 s after the one before. The cameras turn to keep the plane's middle in view as they walk, so the Overlap stays above `source_min_overlap` between frames far apart, and they're far enough apart that a 1% depth change moves a point by more than `source_min_motion` (0.4 px), which at 128 pixels wide takes a parallax of about 40 pixels. So `choose_sources` finds at least `min_sources` Source_Frames for every frame, the first and the last included, under the real time, spacing, Overlap and Image_Motion rules, and a test in 5.5 checks that it does.
    - Its texture is a deterministic grey-level noise of a few levels around mid-grey, from a fixed seed in the helper, sampled bilinearly in the plane's own coordinates, so every camera sees the same texture. Each frame is rendered on the CPU by intersecting each pixel's ray with the plane. A plain variant is the same plane in one flat grey.
    - Helpers give each frame's true depth, normal and Covered mask at the Prior_Size and at the Score_Size, its BGR frame, its `ScoreSource` and its `Candidate`. The `ScoreSource` helper holds the frame's true depth as its Start_Depth, because the occlusion test compares the warp's depth with the Source_Frame's Start_Depth (Req 9.6, A1, D24), and true Observed_Points bring a Source_Frame's Start_Depth to about the true depth through its Scene_Scale, whatever one factor its Prior_Depth is off by.
    - _Requirements: 22.2_

  - [ ] 5.4 Write unit tests for the fits in `tests/test_walls_passes.py`
    - On a hand-made Score_Table whose best guess is known, `change_fit` finds it and gives a Guess_Score at least the Identity_Change's. Its answer keeps |b|, |c| and |d| within `fit_bound`, and two runs give the same answer bit for bit.
    - On a plane tilted to the camera, where the four numbers are each pinned down, `tukey_change` gets an exact change back from exact targets, and still gets it within 10⁻⁶ when a tenth of the targets are moved far off. On a plane seen square on, where a and d trade off, the depths it gives still match the targets within 10⁻⁹.
    - _Requirements: 10.3, 15.2, 15.3, 22.1_

  - [ ] 5.5 Write unit tests for the passes and the self-check in `tests/test_walls_passes.py`
    - `first_pass` takes the best Table_Scale, and on a tie between two scales the one nearer 1 in ln(s).
    - The Solved_Change equals the three passes' changes composed in turn, checked by applying each to the Start_Depth.
    - With every Band_Weight 0, `third_pass` scores the High_Pass.
    - Six Source_Frames in rank order split into Half_A, the 1st, 3rd and 5th, and Half_B, the 2nd, 4th and 6th. The Halves_Difference equals a median worked out in the test.
    - `self_check` fails with fewer than `min_sources` Source_Frames, with a Halves_Difference at `halves_tolerance`, and with either Tile_Score at `min_z`, each on its own.
    - A guard on the scene of 5.3: with the real settings, `choose_sources` gives every textured-plane frame at least `min_sources` Source_Frames, so a failure in 5.6 points at the passes and not at the scene.
    - _Requirements: 8.6, 10.1, 10.6, 10.8, 11.1, 11.2, 11.3, 11.7, 22.1_

  - [ ] 5.6 Write the textured-plane test in `tests/test_walls_passes.py`
    - Run `choose_sources`, the three passes and `self_check` on the textured plane, with the real `[walls]` settings, each frame in turn with its own Start_Depth at 1.1 × the true depth and its Source_Frames' Start_Depths at the true depth, from the helper of 5.3. The test sets the Start_Depths itself and doesn't call `scene_scale`, which would take the scale from the points.
    - 1.1 is a known scale inside the First_Pass's range, so the passes start 10% off the true depth. The First_Pass's Current_Depth is then about 9% off the Source_Frames' Start_Depths, inside `occlusion_tolerance` (15%), so the occlusion test still passes windows, and the later passes and the Tile_Score, at about the true depth, match them (Req 9.6, A1, D24). A 25% start would occlude every window of the First_Pass, and putting the same 25% on the Source_Frames' Start_Depths too would occlude every window of the Second_Pass, the Third_Pass and the Tile_Score instead, once the First_Pass had found the scale.
    - Check that the passes' depth is within 1% of the true depth, as the median over the Covered_Pixels of |z / z_true − 1|, so the passes find the right scale within 1% from a start 10% off, and that the Self_Check passes.
    - The same plane with no texture fails the Self_Check.
    - _Requirements: 22.2_

  - [ ] 5.7 Run the tests and commit
    - Run `uv run pytest`. Every test passes, `tests/test_guard.py` included.
    - On `feat/walls`: `git add splatpipe/walls/fit.py splatpipe/walls/passes.py tests/camera_files.py tests/test_walls_passes.py`, then commit only those files with `git commit -m "Walls: the fits, the three passes and the self-check" -- <the same files>`.
    - _Requirements: 22.4, 22.12_

- [ ] 6. Facets and one frame's whole solve
  - Files: `splatpipe/walls/facets.py`, `splatpipe/walls/solve.py`, `tests/test_walls_facets.py` and `tests/test_walls_solve.py` (new). `tests/camera_files.py` (changed).
  - _Requirements: 6.6, 8.7, 12.1, 12.2, 12.3, 12.4, 12.5, 12.6, 12.7, 22.1, 22.4, 22.12_

  - [ ] 6.1 Write `splatpipe/walls/facets.py`
    - Load `skill::moge-depth`, `skill::gsplat-training` and `skill::splat-files` before writing code.
    - `Facet`, `FacetChange`, `jump_pixels()`, `cut_facets()`, `refit_facet()` and `apply_facet_change()`, as in the design's "Facets (`facets.py`, pure)" section and its decisions table. The module imports only the standard library, numpy, scipy and the walls pure modules.
    - `jump_pixels` marks a Covered_Pixel whose `inverse_depth` and that of a Covered_Pixel `facet_jump_distance` pixels across or down differ by more than `facet_jump` of the smaller.
    - `cut_facets` is the design's seed-and-grow on the Depth_Prior's `normal` at the Prior_Size: seeds in row-major order among the Covered_Pixels that aren't Jump_Pixels, 4-connected growth by breadth-first search while a neighbour's normal is within `facet_normal_degrees` of the region's running mean normal, then the clean-up that frees each pixel more than that angle from the final mean. A region of at least `facet_min_pixels` pixels is a Facet, with its normal and its Median_Inverse_Depth.
    - `refit_facet` builds the Facet's Score_Table and two Half_Tables on the Band_Mix around the Solved_Depth, before any Facet_Change, from `facet_min_scale` to `facet_max_scale` with `facet_scale_step`, over the Score_Windows whose centre's Prior_Size pixel, by `keep_mask`'s nearest-pixel rule, lies in the Facet, with the occlusion test made at the Solved_Depth against each Source_Frame's Start_Depth with `facet_occlusion_tolerance` (D24). It runs `change_fit` with the Facet's own m, and gives a `FacetChange` only when the three gates of Req 12.5 pass.
    - `apply_facet_change` gives the Facet's pixels the change's depth and Changed_Normal and leaves every other pixel as it was.
    - _Requirements: 12.1, 12.2, 12.3, 12.4, 12.5, 12.6_

  - [ ] 6.2 Write `FrameSolve` and `solve_arrays()` in `splatpipe/walls/solve.py`
    - `FrameArrays`, `FrameSolve` and `solve_arrays(frame, candidates, load_source, settings)`, one Frame_Solve on arrays in memory (C4). `FrameArrays` holds what the preparation pass of task 9 gives a Prior_Frame: its name, pose, Score_Size arrays, Prior_Size arrays, k, Scale_Point count and both Start_Depths. `load_source` gives a chosen Source_Frame's `ScoreSource`, so only the Source_Frames are loaded. The module imports only the standard library, numpy and the walls pure modules.
    - `solve_arrays` chooses the Source_Frames and, with fewer than `min_sources`, gives a failed `FrameSolve` whose `result` is `None`, with no pass run (Req 8.7). Otherwise it runs the three passes and the Self_Check.
    - It cuts the Facets of every Frame_Solve that ends, for the flatten, and refits them only when the Self_Check passes (Req 12.7).
    - A passed Frame_Solve's Solved_Depth is the Prior_Size Start_Depth through the Solved_Change, with m from the Prior_Size Start_Depth, and its normals are the Depth_Prior's `normal` through the same change with `changed_normal`. Then each applied Facet_Change, in Facet order, goes on its own Facet's pixels.
    - _Requirements: 6.6, 8.7, 12.4, 12.6, 12.7_

  - [ ] 6.3 Add the box room to `tests/camera_files.py`
    - A small box room: two walls and a floor, each with the noise texture of 5.3, seen by a few cameras of about 128 × 96 pixels, with its Observed_Points on the surfaces.
    - Helpers give each frame's true depth, normal and Covered mask at the Prior_Size, so its Depth_Prior's `normal` is the surface's own and the edges between surfaces are depth or normal jumps.
    - _Requirements: 22.1_

  - [ ] 6.4 Write unit tests for the facets in `tests/test_walls_facets.py`
    - On the box room, `jump_pixels` marks the pixels along the edge where a wall meets the floor nearer the camera.
    - `cut_facets` gives one Facet per surface a frame sees with at least `facet_min_pixels` pixels, drops a smaller patch, holds no Jump_Pixel, puts no pixel in two Facets, keeps every normal within `facet_normal_degrees` of its Facet's normal, and makes each Facet 4-connected.
    - On the textured plane, with a Solved_Depth tilted off the true plane so the Facet has a real change to find, `refit_facet` gives a `FacetChange` when the three gates pass, and `None` when each gate alone fails: a Halves_Difference above `facet_halves_tolerance`, a change not above `facet_min_change_ratio` × it, and a Tile_Score below `facet_min_z`.
    - `apply_facet_change` changes only the Facet's pixels.
    - _Requirements: 12.1, 12.2, 12.3, 12.4, 12.5, 12.6, 22.1_

  - [ ] 6.5 Write unit tests for one frame's solve in `tests/test_walls_solve.py`
    - On the textured plane, as in 5.6, with the frame's `FrameArrays` holding k = 1 and its Prior_Depth and Start_Depth at 1.1 × the true depth, and `load_source` giving its Source_Frames' Start_Depths at the true depth, `solve_arrays` passes, its Solved_Depth is within 1% of the true depth at the Prior_Size, and its normals are unit length and face the camera.
    - On the plain plane it fails, with its Facets cut and none refitted.
    - With too few candidates it fails with `result` `None` and `load_source` never called.
    - _Requirements: 8.7, 12.7, 22.1_

  - [ ] 6.6 Run the tests and commit
    - Run `uv run pytest`. Every test passes, `tests/test_guard.py` included.
    - On `feat/walls`: `git add splatpipe/walls/facets.py splatpipe/walls/solve.py tests/camera_files.py tests/test_walls_facets.py tests/test_walls_solve.py`, then commit only those files with `git commit -m "Walls: facets and one frame's whole solve" -- <the same files>`.
    - _Requirements: 22.4, 22.12_

- [ ] 7. Cross-frame work I: neighbours, projected values, the fuse, the correction, the despeckle and the per-frame choice
  - Files: `splatpipe/walls/crossframe.py` and `tests/test_walls_crossframe.py` (new).
  - _Requirements: 14.1, 14.2, 14.3, 14.4, 15.1, 16.1, 16.4, 16.5, 16.6, 16.7, 17.1, 17.2, 17.3, 17.4, 17.5, 22.1, 22.3, 22.4, 22.12_

  - [ ] 7.1 Write `splatpipe/walls/crossframe.py`
    - `neighbours()`, `projected_values()`, `agreeing()`, `agreeing_median()`, `fuse_frame()`, `correct_frame()`, `despeckle_frame()`, `sparse_error()` and `frame_choice()`, as in the design's "Cross-frame work" and "The per-frame choice" sections. The module imports only the standard library, numpy and `splatpipe.walls.geometry`.
    - `neighbours` gives the Trusted_Frames other than the frame within `neighbour_window` Pick_Numbers, the nearest first, ties to the lower Pick_Number, at most `neighbour_frames` (D14).
    - `projected_values` gives an (h, w, k) array with at most one value per Covered_Pixel from each Neighbour_Frame, its nearest point there through `zbuffer_project`, and NaN where it has none.
    - `agreeing` keeps a pixel's Projected_Values within `agree_tolerance` of the nearest of them, the smallest depth (C2). `agreeing_median` gives their median where there are at least `agree_min_frames`, and NaN elsewhere (Req 15.1).
    - `fuse_frame`, `correct_frame` and `despeckle_frame` follow Req 14.3, 16.1 and 16.4. Each reads only the depths it's given from before the stage and writes a new array, so no pixel's change within a stage moves another's (Req 16.5), and each changes depths only (Req 16.6).
    - `sparse_error` is the median over the Scale_Points of |ln(z / z_point)|. `frame_choice` gives each Prior_Frame its outcome, `passed`, `fallback` or `depth_prior`, and for `depth_prior` its reason, `timed_out`, `no_fallback` or `sparse_error`, by Req 17.2 to 17.5 and A14 (C5).
    - _Requirements: 14.1, 14.2, 14.3, 14.4, 15.1, 16.1, 16.4, 16.5, 16.6, 17.1, 17.2, 17.3, 17.4, 17.5_

  - [ ] 7.2 Write unit tests for the cross-frame stages in `tests/test_walls_crossframe.py`
    - `neighbours` orders by Pick_Number distance, breaks a tie to the lower Pick_Number, leaves out frames past `neighbour_window`, Failed_Frames and the frame itself, and stops at `neighbour_frames`.
    - `projected_values` gives one value per pixel per Neighbour_Frame, its nearest, and NaN where nothing lands. `agreeing` keeps the values within 3% of the smallest, and `agreeing_median` needs `agree_min_frames` of them.
    - `fuse_frame` takes the median of the own depth and the values within `fuse_tolerance` when there are at least `fuse_min_values`, and keeps the own depth otherwise. `correct_frame` moves only the pixels more than `correction_tolerance` off in ln(depth). `despeckle_frame` moves a lone spike to its window's median, cuts the window at the border, and leaves a pixel whose window has too few valid pixels.
    - Running a stage over the frames in two orders gives the same depths bit for bit (Req 14.4).
    - `frame_choice` on each row of Req 17's table: a Trusted_Frame, a Failed_Frame with a Fallback_Depth and a good Sparse_Error, one with a bad Sparse_Error, one with a bad Sparse_Error but too few Scale_Points, one with no Fallback_Depth and a Timed_Out_Frame. `sparse_error` equals a hand case.
    - _Requirements: 14.1, 14.2, 14.3, 14.4, 15.1, 16.1, 16.4, 16.5, 16.6, 17.2, 17.3, 17.4, 17.5, 22.1_

  - [ ] 7.3 Write the property test for Property 6 in `tests/test_walls_crossframe.py`, with Hypothesis
    - **Property 6: The despeckle never moves a steady pixel**
    - **Validates: Requirements 16.7**
    - Draw as the design's Property 6 strategy says: a depth map from 3 × 3 to 20 × 20 pixels of finite depths above 0, a random Covered mask, `despeckle_window` an odd number from 3 to 7, `despeckle_min_valid` from 1 to the window's area, and `despeckle_tolerance` from 0 to 1.
    - The oracle works out each pixel's window, cut at the border, its valid pixels and their median with its own loops. Check that every pixel within `despeckle_tolerance` of its window's median, and every pixel whose window has fewer than `despeckle_min_valid` valid pixels, comes back bit for bit, compared through `struct.pack("<d", value)`. So with the real tolerance the despeckle never moves a pixel within 3% of its window's median.
    - _Requirements: 16.7, 22.3_

  - [ ] 7.4 Run the tests and commit
    - Run `uv run pytest`. Every test passes, `tests/test_guard.py` included.
    - On `feat/walls`: `git add splatpipe/walls/crossframe.py tests/test_walls_crossframe.py`, then commit only those files with `git commit -m "Walls: neighbours, the fuse, the correction, the despeckle and the per-frame choice" -- <the same files>`.
    - _Requirements: 22.4, 22.12_

- [ ] 8. Cross-frame work II: world planes, the projection fallback, the plane registration and the flatten
  - Files: `splatpipe/walls/planes.py` and `tests/test_walls_planes.py` (new).
  - _Requirements: 13.1, 13.2, 13.3, 13.4, 15.1, 15.2, 15.3, 15.4, 16.2, 16.3, 16.5, 16.6, 22.1, 22.4, 22.12_

  - [ ] 8.1 Write `splatpipe/walls/planes.py`
    - Load `skill::moge-depth`, `skill::gsplat-training` and `skill::splat-files` before writing code.
    - `WorldPlane`, `robust_plane()`, `world_planes()`, `camera_plane()`, `projection_fallback()`, `plane_registration()` and `flatten_frame()`, as in the design's "Cross-frame work" section and its decisions table. The module imports only the standard library, numpy and the walls pure modules.
    - `robust_plane` is the Tukey-reweighted principal components of the decisions table: the weighted mean and the 3 × 3 weighted covariance, built with `np.einsum` (C14), the normal from `np.linalg.eigh`, the unscaled median absolute deviation of the distances, Tukey's weights at `tukey_constant` deviations, for `tukey_iterations` rounds, the weights left as they were when the deviation is 0.
    - `world_planes` puts every Trusted_Frame's Facets at their Fused_Depth in the world, fits each Facet_Plane, and merges them largest Facet first: a Facet joins the first World_Plane within `plane_merge_degrees` whose offset is within `plane_merge_offset` × its Facet's camera distance, and each join refits the World_Plane through all its Facets' points. It keeps the World_Planes with Facets from at least `plane_min_frames` Prior_Frames and at least `plane_min_share` of the pixels of the largest such World_Plane (A10). Every angle between plane normals comes from |n₁·n₂|, and offsets are compared with the normals facing the same way (C11).
    - `camera_plane` turns a World_Plane into the camera frame, n_c = R·n_w and δ_c = δ_w + n_c·t, and a plane's depth along a ray r is δ_c / (n_c·r).
    - `projection_fallback` and `plane_registration` make a Failed_Frame's Fallback_Depth as Req 15.2 and 15.3 say, with `tukey_change` of the Start_Depth, the registration's tolerance falling geometrically from `register_start_tolerance` to `register_end_tolerance` over `register_passes` passes. Each gives `None` when Req 15.4 says there's no Fallback_Depth, and otherwise the depth, the normals and the fitted change.
    - `flatten_frame` follows Req 16.2 and 16.3. It reads only the depths from before the stage (Req 16.5) and changes a pixel's normal only when it moves the pixel (Req 16.6).
    - _Requirements: 13.1, 13.2, 13.3, 13.4, 15.1, 15.2, 15.3, 15.4, 16.2, 16.3, 16.5, 16.6_

  - [ ] 8.2 Write unit tests for the planes and the fallbacks in `tests/test_walls_planes.py`
    - `robust_plane` fits an exact plane, keeps every point of a perfect one, and still fits it within 10⁻⁶ with one point moved far off.
    - On the box room's frames as Trusted_Frames, `world_planes` merges each surface's Facets into one World_Plane, keeps the room's three planes, drops a plane whose Facets come from fewer than `plane_min_frames` frames and one below `plane_min_share`, and merges two Facets whose fitted normals point opposite ways.
    - `projection_fallback` gets a plane back from agreeing medians at the Covered_Pixels, gives each pixel with a median that median and every other pixel the fitted change's depth, and gives `None` below `projection_min_coverage`.
    - `plane_registration` registers a Start_Depth off by a scale and a tilt to the box room's World_Planes, with the tolerance of each pass as the test works it out, and gives `None` with no World_Plane and with fewer than `register_min_pixels` matched in a pass.
    - `flatten_frame` moves a pixel within `flatten_tolerance` of its plane depth onto the plane and gives it the plane's normal facing the camera, takes a World_Plane's normal within `flatten_degrees` and the Facet's own robust normal otherwise, and leaves a pixel past the tolerance with its depth and its normal.
    - _Requirements: 13.1, 13.2, 13.3, 13.4, 15.1, 15.2, 15.3, 15.4, 16.2, 16.3, 16.6, 22.1_

  - [ ] 8.3 Run the tests and commit
    - Run `uv run pytest`. Every test passes, `tests/test_guard.py` included.
    - On `feat/walls`: `git add splatpipe/walls/planes.py tests/test_walls_planes.py`, then commit only those files with `git commit -m "Walls: world planes, the fallbacks and the flatten" -- <the same files>`.
    - _Requirements: 22.4, 22.12_

- [ ] 9. The walls step: its worker pool, outputs, report and CLI
  - Files: `splatpipe/walls/__main__.py`, `splatpipe/walls/pool.py`, `splatpipe/walls/outputs.py`, `splatpipe/walls/report.py`, `tests/test_walls_outputs.py` and `tests/test_walls_step.py` (new). `splatpipe/walls/__init__.py`, `splatpipe/walls/inputs.py`, `splatpipe/walls/solve.py`, `tests/conftest.py`, `tests/camera_files.py` and `tests/test_walls_workers.py` (changed).
  - _Requirements: 1.1, 1.2, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 1.10, 4.1, 4.5, 4.6, 4.7, 4.8, 4.9, 4.11, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 6.5, 7.6, 8.8, 10.9, 11.9, 12.8, 13.5, 15.5, 17.1, 17.6, 17.7, 17.8, 17.9, 17.10, 18.1, 18.2, 18.3, 18.4, 18.5, 18.6, 18.7, 18.8, 18.9, 18.10, 19.1, 19.2, 19.3, 19.4, 19.5, 19.6, 19.7, 22.5, 22.6, 22.7, 22.12_

  - [ ] 9.1 Write `splatpipe/walls/pool.py`
    - Load `skill::moge-depth`, `skill::gsplat-training` and `skill::splat-files` before writing code.
    - `WorkerStopped`, the `Pool` protocol, `MapResult`, `worker_count()`, `open_pool()` and `InlinePool`, as in the design's "The worker pool (`pool.py`)" section, with the calls C6 and C7. The module imports only the standard library, cv2 and `splatpipe.errors`.
    - `worker_count(workers, frames)` is `workers` when above 0, else `len(os.sched_getaffinity(0))` where the system has it and `os.cpu_count()` otherwise, never more than the Prior_Frames and at least 1.
    - `open_pool(workers)` starts that many `multiprocessing.get_context("spawn").Process` workers fed by a task queue, each first calling `cv2.setNumThreads(1)`. Around each worker's start, the first ones and any fresh ones, it sets `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `NUMEXPR_NUM_THREADS` and `VECLIB_MAXIMUM_THREADS` to `"1"` in `os.environ`, which the worker inherits, and then puts the main process's own values back, unset ones unset, so the Train_Step and the Clean_Step after it keep theirs (C15). Each worker says which task it took before it runs it.
    - `map_frames(fn, tasks)` gives the results in task order. `map_frames_until(fn, tasks, cutoff, now)` starts tasks in order, none at or after the `cutoff`, ends the workers still running at the `cutoff` and starts fresh ones in their place, and gives the finished results with the names of the Timed_Out_Frames.
    - A worker that exits with no result raises `WorkerStopped` with Req 5.7's message, naming the frame it had taken when that's known. A worker's `ExpectedFailure` is raised again as the same message, and any other error as an error that isn't an `ExpectedFailure`, so the Step_Runner reports it as an Unexpected_Error (Req 5.8). `close()` ends every worker and leaves none running.
    - `InlinePool` runs `fn(task)` in the calling process, with `cv2.setNumThreads(1)` while it runs and the main process's count put back after (C14). Its `map_frames_until` asks `now()` before each task and starts none at or after the `cutoff`, and asks again when a task ends: a task that ends at or after the `cutoff` counts as timed out, as the real pool would have ended it.
    - _Requirements: 4.5, 4.6, 5.1, 5.2, 5.4, 5.5, 5.6, 5.7, 5.8_

  - [ ] 9.2 Write the preparation pass, the scratch loaders and the worker tasks
    - In `splatpipe/walls/inputs.py`, the scratch folder's file names, as the design's "The scratch folder" lists them, and `prepare_frame()`, which decodes one Prior_Frame with `score_image` and `score_mask`, works out its Luma, High_Pass and Bands, its Scale_Points, Scene_Scale and both Start_Depths (C3), and writes the design's scratch arrays to `work/walls/scratch/<stem>/` with `np.save`. `inputs.py` imports `splatpipe.walls.images` and `splatpipe.walls.sources` for this, and never `splatpipe.walls.solve`.
    - In `splatpipe/walls/solve.py`, `load_frame()` and `load_source()`, which read the scratch arrays back with `np.load(..., mmap_mode="r")`, by `inputs.py`'s file names, into a `FrameArrays` and a `ScoreSource` (C4). `load_source` gives the `ScoreSource` the Source_Frame's `start_depth.npy` and `covered.npy`, so the occlusion test compares with the Start_Depth the preparation pass worked out, never the raw Prior_Depth (D24). They live here, beside `solve_frame`, so `inputs.py` and `solve.py` don't import each other.
    - In `splatpipe/walls/solve.py`, `SolveTask`, plain data a `spawn` worker can pickle: the frame's name, its pose, the settings, the Shared_Camera, the scratch folder and the other Prior_Frames' `Candidate`s. `solve_frame(task)` loads the frame with `load_frame`, calls `solve_arrays` with `load_source`, writes the Solved_Depth and its normals to `current_depth.npy` and `current_normal.npy`, and gives the `FrameSolve` without its arrays.
    - The worker tasks of the cross-frame stages, one per stage, each reading the depths and normals the stage before left in the scratch folder and writing the frame's new ones: the fuse, the fallback, the correction, the flatten, the despeckle and the per-frame choice. The World_Planes are built once, on the main process, between the fuse and the fallback (D15).
    - _Requirements: 4.8, 5.2, 5.3, 7.6, 14.4_

  - [ ] 9.3 Write `splatpipe/walls/outputs.py` and `splatpipe/walls/report.py`
    - `remove_demos()`, `write_prior_file()`, `write_record()`, `write_demo()`, `demo_panel()`, `walls_prior()`, `depth_prior_copy()` and `remove_scratch()`, as in the design's "Outputs (`outputs.py`)" and "The per-frame choice and the Walls_Prior" sections. The module imports only the standard library, numpy, cv2, `splatpipe.errors`, `splatpipe.jsonio`, `splatpipe.train.priors` and `splatpipe.train.files`, whose `read_training_image` and `scaled_down` decode the demo's Frame.
    - `walls_prior` keeps the Depth_Prior's `mask` exactly, writes float32(1 / the Walls_Depth) where the mask is true and 0 elsewhere, takes the Depth_Prior's `inverse_depth` and `normal` at a marked pixel whose float32 value isn't finite and above 0, takes the Depth_Prior's `normal` where the change chain's normal isn't finite or has a length of 0, and (0, 0, 0) at every unmarked pixel (D3, D4, A13). Every Walls_Prior passes `check_prior` at the Prior_Size. `depth_prior_copy` gives the Depth_Prior's three arrays unchanged, bit for bit (D2).
    - `demo_panel` puts side by side, at the Prior_Size, the Frame, decoded again with `read_training_image` and `scaled_down`, the normals of the Depth_Prior's depth and the normals of the Walls_Prior's depth, each from finite differences of its depth map for display only, coloured (n + 1) / 2 × 255 with black where there's no normal. `write_demo` encodes it with `cv2.imencode(".jpg", panel, [cv2.IMWRITE_JPEG_QUALITY, demo_jpeg_quality])`.
    - Each writer turns an `OSError` into Req 18.6's message, naming the file and the reason.
    - In `report.py`, `Counts`, `Timings`, the Req 19.5 warning for Timed_Out_Frames, the Req 19.6 warning when fewer than every Prior_Frame passed, and `no_trusted_failure()`, Req 18.8's `ExpectedFailure`, which gives the Prior_Frames, how many timed out, how many failed their Self_Check and, of those, how many had too few Source_Frames, and says that the Job goes on and trains on the depth priors.
    - _Requirements: 17.6, 17.7, 17.8, 17.9, 18.1, 18.3, 18.6, 18.8, 18.10, 19.5, 19.6_

  - [ ] 9.4 Write the step in `splatpipe/walls/__init__.py` and `splatpipe/walls/__main__.py`
    - `WALLS_POOL: Callable[[int], Pool] = open_pool`, `utc_now`, which is `splatpipe.train.deadline.utc_now` (C8), `run()` and `work()`, as in the design's "The step's flow (`__init__.py`)" section. `run` calls `run_step("walls", "splatpipe.walls", ...)`. `work` reads `WALLS_POOL` and `utc_now` when it runs, so a test's `monkeypatch` reaches them through `run`, `run_step`, the CLI and `run_job`.
    - `work` follows Req 1.4's order: remove `out/walls/`; `read_settings`; `read_cap_deadline(os.environ)`; `walls_times` and `solve_cutoff` from `utc_now()`, read once; `check_inputs`; `read_inputs`; the preparation pass; the Frame_Solves in the Solve_Order through `map_frames_until`; the Req 18.8 failure when no Prior_Frame is a Trusted_Frame; then the fuse, the World_Planes, the fallback, the correction, the flatten, the despeckle and the per-frame choice, each per-frame stage through `map_frames`; then the Walls_Prior_Files, the removal of the scratch folder, the Walls_Record, the Walls_Demos and, last of all, `write_index(work/walls, depth_index)`.
    - A Timed_Out_Frame gets a Depth_Prior_Copy and no fallback, correction, flatten or despeckle, and stays a Source_Frame of the others (Req 4.7, 4.8).
    - The Walls_Record holds `format` 1, the run's times and which limit set the Walls_Stop_Time, the workers, the World_Planes and the Walls_Demos' names, and one object per Prior_Frame in Pick_Number order with the design's fields, null for a number that isn't finite or wasn't worked out (Req 18.2). The Walls_Demos are at the places ⌊(i + 0.5) × N / `demo_frames`⌋ among the N Prior_Frames whose Walls_Prior isn't a Depth_Prior_Copy (Req 18.3).
    - Each count is added with `int(...)` as soon as it's known and each timing when its block ends, so a failed report holds exactly what the step had worked out (Req 19.3, 19.4). `read` times the inputs and the preparation pass. A `finally` removes `work/walls/scratch/` and closes the pool, done or failed.
    - `__main__.py`'s `main(argv)` calls `step_main("walls", "splatpipe.walls", argv)`, and the file ends with `if __name__ == "__main__": sys.exit(main())`.
    - _Requirements: 1.1, 1.2, 1.4, 1.7, 1.8, 1.9, 1.10, 4.1, 4.6, 4.7, 4.8, 4.9, 4.11, 5.3, 5.6, 6.5, 7.6, 8.8, 10.9, 11.9, 12.8, 13.5, 15.5, 17.1, 17.10, 18.2, 18.3, 18.4, 18.5, 18.7, 18.8, 19.1, 19.2, 19.3, 19.4, 19.7_

  - [ ] 9.5 Add the pool fixture and the walls Job writer to the test helpers
    - In `tests/conftest.py`, a `walls_pool` fixture that installs `InlinePool` with `monkeypatch.setattr("splatpipe.walls.WALLS_POOL", ...)`, and a dead-worker factory whose pool raises `WorkerStopped` for a frame the test names. conftest imports `splatpipe.walls` only inside the fixtures.
    - In `tests/camera_files.py`, `write_walls_job(job_folder, scene, ...)`, which writes a scene of 5.3 or 6.3 as a Job's inputs: `work/frames/frames.json` with a Pick per frame, `work/cameras/` with its frames as JPEGs at a quality high enough to keep the texture, its Sparse_Model and, when asked, Mask_Files, and `work/depth/` with a Depth_Prior per frame off by a known scale, written with the Prior_Writer.
    - _Requirements: 22.5, 22.12_

  - [ ] 9.6 Write unit tests for the outputs in `tests/test_walls_outputs.py`
    - `walls_prior` keeps the mask exactly, takes the Depth_Prior's value and normal at a pixel whose float32 inverse depth rounds to 0 or overflows, takes the Depth_Prior's normal at a pixel with no normal, and gives (0, 0, 0) at every unmarked pixel. `depth_prior_copy` is bit for bit.
    - `demo_panel` and `write_demo` give a JPEG that `cv2.imdecode` reads, three Prior_Sizes wide, in which the box room's flat wall is one colour in the walls half.
    - `write_prior_file`, `write_record` and `write_demo`, with the write raising `OSError(errno.ENOSPC, "No space left on device")`, each give Req 18.6's message naming the file.
    - `no_trusted_failure` and the two warnings give their counts, and none holds "Traceback".
    - _Requirements: 17.6, 17.7, 17.8, 17.9, 18.3, 18.6, 18.8, 19.5, 19.6, 22.1_

  - [ ] 9.7 Write the property test for Property 7 in `tests/test_walls_outputs.py`, with Hypothesis
    - **Property 7: A Walls_Prior round-trips through the Prior_Format and keeps the mask**
    - **Validates: Requirements 17.6, 17.7, 18.9**
    - Draw as the design's Property 7 strategy says: a Prior_Size from 1 × 1 to 8 × 8, a `mask` with at least one true pixel, `inverse_depth` above 0 under it, random unit normals, and a Walls_Depth of positive float64 on the Covered_Pixels with a few pixels forced to a non-finite or non-positive inverse depth. Each example writes into its own `tempfile.TemporaryDirectory()` with `write_prior` and `write_index`, and reads back with `read_prior_set` against a one-image Sparse_Model whose camera and pose match the index.
    - Check that the read-back `mask` is the Depth_Prior's bit for bit, that `inverse_depth` is finite and above 0 exactly where the mask is true and 0 elsewhere, that `check_prior` passes, and that the three arrays read back bit for bit, compared through `.view(np.uint32)` and `.view(np.uint8)`. Also check that `depth_prior_copy` reads back as its Depth_Prior's three arrays, bit for bit.
    - _Requirements: 17.6, 17.7, 18.9, 22.3_

  - [ ] 9.8 Write the pool tests and the two-worker test in `tests/test_walls_workers.py`
    - `worker_count` with `os.sched_getaffinity` and `os.cpu_count` replaced gives the `workers` setting, every vCPU for 0, the affinity count over `cpu_count` where both exist, never more than the frames and at least 1.
    - `InlinePool.map_frames_until` with a fake clock starts no task at or after the cutoff, counts a task that ends after it as timed out, and gives the rest in order.
    - With `open_pool(2)`: a worker function that calls `os._exit` gives `WorkerStopped` naming its frame, one that raises an `ExpectedFailure` gives the same message, and one that raises `ValueError` gives an error that isn't an `ExpectedFailure`. A worker function that reads `OMP_NUM_THREADS` gives `"1"`, while the test process's `os.environ` holds its own values again, one set and one unset by `monkeypatch`, once the workers have started (C15). After `close()`, `multiprocessing.active_children()` is empty. The worker functions live at the module's top level, so `spawn` imports them by name.
    - One tiny Walls_Step run on the textured plane's Job, once with `InlinePool` and once with `open_pool(2)`, writes the same Walls_Prior_Files bit for bit and the same Walls_Record values but its times (Req 5.9, 22.6).
    - _Requirements: 5.1, 5.4, 5.6, 5.7, 5.8, 5.9, 22.6_

  - [ ] 9.9 Write unit tests for the step in `tests/test_walls_step.py`
    - Run the step through `splatpipe.walls.run` with the `walls_pool` fixture, a fake clock through `splatpipe.walls.utc_now`, and Jobs from `write_walls_job`, so the Step_Runner and the real report are part of each test.
    - A done run on the textured plane: `read_prior_set(work/walls, model)` reads every Walls_Prior back, the Walls_Index holds exactly the Depth_Index's values, `walls.json` holds the design's fields, `out/walls/` holds the Walls_Demos, the report's six counts hold Req 19.1's two sums with every count an `int`, and the report holds the ten timings and every `[walls]` and Preset setting.
    - The failures, each with its Readable_Error, `expected: true` and no "Traceback": a bad setting, a malformed `SPLATPIPE_CAP_DEADLINE`, no `work/depth/depth.json`, a missing camera solve path, no `frames.json`, broken depth priors, and a `frames.json` that doesn't match the camera solve.
    - A run on the plain plane, where no Prior_Frame passes, fails with Req 18.8's message and leaves no Walls_Index and no Walls_Prior_File. With `SPLATPIPE_CAP_DEADLINE` set close, the Walls_Stop_Time comes from `train_reserve_minutes`, and with a fake clock that reaches the Solve_Cutoff after two Frame_Solves, the Timed_Out_Frames get Depth_Prior_Copies, the report holds Req 19.5's warning, and the solved frames used Timed_Out_Frames as Source_Frames. The dead-worker factory fails the step with Req 5.7's message.
    - A file left in `out/walls/` by an earlier run is gone. Listing the Job_Folder before and after a done run shows only `work/walls/` files, `out/walls/` files and `out/reports/walls.json` new, with no `work/walls/scratch/`, and every file in `work/frames/`, `work/cameras/` and `work/depth/` the same bytes.
    - `splatpipe.walls.__main__.main(["--job", job_id])` prints `walls: done` and gives 0, or prints the Readable_Error on stderr with no traceback and gives 1.
    - `run_job` with `steps=(StepEntry("walls", "splatpipe.walls", False), StepEntry("train", "splatpipe.train", True))`, `FakeGpu` and a Job with no `work/depth/` gives `done_with_warnings`, the walls entry `failed` with the Req 2.2 message in the Job_Log and the Job_Summary.
    - _Requirements: 1.1, 1.2, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 1.10, 4.1, 4.6, 4.7, 4.8, 4.11, 5.7, 17.1, 18.1, 18.2, 18.4, 18.5, 18.7, 18.8, 18.9, 19.1, 19.2, 19.3, 19.4, 19.5, 19.6, 19.7, 22.5, 22.7_

  - [ ] 9.10 Run the tests and commit
    - Run `uv run pytest`. Every test passes, `tests/test_guard.py` included.
    - On `feat/walls`: `git add splatpipe/walls/__init__.py splatpipe/walls/__main__.py splatpipe/walls/pool.py splatpipe/walls/outputs.py splatpipe/walls/report.py splatpipe/walls/inputs.py splatpipe/walls/solve.py tests/conftest.py tests/camera_files.py tests/test_walls_outputs.py tests/test_walls_step.py tests/test_walls_workers.py`, then commit only those files with `git commit -m "Walls: the step, its worker pool, outputs, report and CLI" -- <the same files>`.
    - _Requirements: 22.4, 22.12_

- [ ] 10. Training reads the walls priors, the step list, the docs and the expected times
  - Files: `splatpipe/train/__init__.py`, `splatpipe/run.py`, `config.toml`, `docs/pipeline.md`, `docs/third-party-licences.md`, `tests/test_train_step.py`, `tests/test_run.py`, `tests/test_frames_step.py`, `tests/test_cameras_step.py`, `tests/test_imports.py`, `tests/test_walls_settings.py`, `tests/test_docs.py`, `tests/test_cloud_docs.py`, `tests/test_cloud_config.py` and `tests/test_start_core.py`, all changed.
  - _Requirements: 1.3, 1.5, 1.6, 3.8, 20.1, 20.2, 20.3, 20.4, 20.5, 20.6, 21.1, 21.2, 21.3, 21.4, 21.5, 21.6, 21.7, 21.8, 21.9, 21.10, 21.11, 21.12, 21.13, 21.14, 21.15, 22.4, 22.9, 22.10, 22.11, 22.12_

  - [ ] 10.1 Choose the prior folder in `_read_priors` in `splatpipe/train/__init__.py`
    - Load `skill::moge-depth`, `skill::gsplat-training` and `skill::splat-files` before writing code.
    - `WALLS_FOLDER = "walls"` and `_read_priors` as in the design's "The train change", the one place that picks the Train_Step's prior folder (Req 20.5).
    - With `work/walls/depth.json` a file, it reads `read_prior_set(work/walls, model)`. A `BrokenPriors` there adds a warning of its own, which names the file and what failed and says that training fell back to the depth priors in `work/depth/` (C9), and then the step reads `work/depth/` exactly as today, with `_BROKEN_PRIORS` naming `work/depth/` when those fail too.
    - With no `work/walls/depth.json`, the flow is today's exactly, with no warning about the Walls_Step. `prior_images` counts the Priors of the Prior_Set the step trains with, from whichever folder. No setting, count, timing, Loss_Term or Start_Splat changes (Req 20.4). The docstring says so.
    - _Requirements: 20.1, 20.2, 20.3, 20.4, 20.5, 20.6_

  - [ ] 10.2 Write unit tests for the prior folder in `tests/test_train_step.py`
    - With `FakeGpu` and the plane scene: a Job with a walls Prior_Set whose `inverse_depth` differs from the depth one trains with the walls Priors, and `prior_images` counts them.
    - Broken walls priors give the walls warning naming `work/walls/` and the file, and the step trains with the depth Priors. With both broken, the report holds both warnings and the Scene has no priors.
    - Walls Prior_Files with no `work/walls/depth.json` are ignored, and the depth Priors are used with no warning.
    - The regression check: a run with no `work/walls/` gives exactly today's counts, timings and warnings, and the splat-training and depth-priors tests in this file pass with no change to what they expect.
    - _Requirements: 20.1, 20.2, 20.3, 20.4, 20.6, 22.11_

  - [ ] 10.3 Put the Walls_Step in the Step_List in `splatpipe/run.py`
    - `STEP_LIST` becomes Req 1.3's six entries: frames, cameras, depth, `StepEntry("walls", "splatpipe.walls", False)`, train and clean.
    - The comment above it names the Walls_Step as optional, from this spec (walls spec, Req 1.3), and still says that the Fake_Step stays out.
    - _Requirements: 1.3_

  - [ ] 10.4 Update the tests that depend on the Step_List
    - In `tests/test_run.py`, the Step_List test becomes `test_the_step_list_holds_the_frames_cameras_depth_walls_train_and_clean_steps_and_is_the_default_list` and asserts the six entries.
    - In `tests/test_frames_step.py`, the Job through every entry but the cameras one expects frames `done`, depth `failed`, walls `failed` with the Req 2.2 message naming `work/depth/depth.json`, train `failed` and clean `not_run`. The module docstring names the Walls_Step.
    - In `tests/test_cameras_step.py`, the Job through the real Step_List uses the `walls_pool` fixture, so no worker process starts, and expects frames, cameras and depth `done`, walls `failed` with the Req 18.8 message, train and clean `done`, and the Job `done_with_warnings` (A21). Its 5 Prior_Frames have Pick_Numbers 0 to 4, so no frame can have 4 Source_Frames 2 Pick_Numbers apart, and every one fails its Self_Check with too few Source_Frames before any pass runs (Req 8.7): the message gives 5 Prior_Frames, 0 timed out and 5 failed, all 5 with too few Source_Frames. Its 16 × 12 frames couldn't hold a 31 × 31 Score_Window either. The test's comments and the module docstring say why.
    - In `tests/test_imports.py`, the Step_List check's docstring names the Walls_Step, whose `[walls]` it now loads from the real config for each Preset.
    - In `tests/test_walls_settings.py`, the Fallback_List test (C12): with the real Config and the real Step_List, `step_needs_over` gives each Preset's needs with the Walls_Step's `min_memory_gb` of 32 counted, and `select_candidates` keeps every type on each Preset's Fallback_List. If a type no longer fits, stop and ask the user, because the Fallback_Lists stay as they are (D16).
    - _Requirements: 1.5, 1.6, 3.8, 22.9, 22.10_

  - [ ] 10.5 Change the expected times and the tests that pin them
    - In `config.toml`, `expected_minutes` in `[cloud.presets.fast]`, `[cloud.presets.default]` and `[cloud.presets.quality]` becomes 50, 100 and 180, and each `cap_minutes` stays at 120, 180 and 360 (D19).
    - In `tests/test_cloud_config.py`, `PRESET_MINUTES` becomes `{"fast": (50, 120), "default": (100, 180), "quality": (180, 360)}`, its comment says why, and the fixed sample mapping's `expected_minutes` become 50, 100 and 180.
    - In `tests/test_start_core.py`, the quote's `expected_time` becomes "about 50 min, up to the 120 min time cap".
    - Search the repo, `.kiro/` aside, for any other doc that states the old Expected_Times, and change it in this task (Req 21.12). Today none does.
    - _Requirements: 21.11, 21.12_

  - [ ] 10.6 Write `## Step: walls` and change the other parts of `docs/pipeline.md`
    - `## Step: walls` after `## Step: depth` and before `## Step: train`, built like the depth section: a paragraph with the CLI, `uv run python -m splatpipe.walls --job <id> [--preset P]`, and `splatpipe.walls.run(job_id, preset=None)`, that the step is optional and runs on the CPU only, and the order of its work. Then `### What it reads`, `### What it writes`, `### Config section`, `### Step report fields` and `### Files in out/`.
    - The method in our own words, quoting no text from the paper: the Bas_Relief_Change and why it keeps every pixel on its ray and every plane flat, the Scene_Scale and the Start_Depth, the Source_Frames, the windowed NCC Score_Table, the three passes and the Band_Mix, the Self_Check, the Facets, the World_Planes, the fuse, the fallback, the correction, the flatten, the despeckle and the per-frame choice. It cites D. J. Kriegman and P. N. Belhumeur, "What Shadows Reveal about Object Structure", ECCV 1998, pp. 399–414, with its journal version in JOSA A 18(8), 2001 (Req 21.2).
    - What it writes: the Walls_Prior_Files and the Walls_Index in the Prior_Format at the Depth_Step's Prior_Size, the Walls_Index a copy of the Depth_Index written last, which Prior_Frames get a Depth_Prior_Copy, and the Walls_Record's fields (Req 21.3). Config section: every `[walls]` and Preset key with its table, value and meaning, the defaults untested on real footage marked (Req 21.4). Step report fields: the six counts as `counts.<name>`, the ten timings as `timings.<name>`, the two warnings and the Expected_Failures (Req 21.5). Files in out/: the Walls_Demos in `out/walls/`, removed first at the start of each run.
    - "The step list": the list holds the frames step and the cameras step, both required, the depth step and the walls step, both optional, the train step, required, and then the clean step, optional, with the six entries as `STEP_LIST` writes them; a Job is done once all six steps are done, and done with warnings when the three required steps are done and the depth step, the walls step or the clean step fails; and the planned optional steps name the walls step. It still holds `StepEntry("clean", "splatpipe.clean", False)`, which `tests/test_clean_repo.py` looks for (Req 21.6).
    - "What each step asks the machine for": a row "Correcting walls | 16 vCPUs, 32 GB RAM", and a sentence that the walls step's needs live in `[walls]` as `min_vcpus`, a soft need, and `min_memory_gb`, and that `max_minutes` cuts the step short on a machine with fewer vCPUs (Req 21.7).
    - "The cap deadline": the walls step reads `SPLATPIPE_CAP_DEADLINE` too and stops solving frames in time to leave the train step its `train_reserve_minutes` (Req 21.8). The same part, or "Running a job in the cloud", states the new Expected_Times of 50, 100 and 180 minutes, the rough cost the walls step adds to a Job of each Preset, about $0.80, $1.60 and $3.00, and that the website's start page shows the new times once the account owner redeploys (Req 21.12).
    - `## Step: train` says that the step reads `work/walls/` when `work/walls/depth.json` is there and `work/depth/` otherwise, with the walls fallback warning (Req 21.9). `## Step: depth` says that the walls step reads the depth priors too and leaves them as they are (Req 21.10).
    - Read every part this task wrote against the code of tasks 1 to 9 once more, and fix any sentence that doesn't say what the code does.
    - _Requirements: 21.1, 21.2, 21.3, 21.4, 21.5, 21.6, 21.7, 21.8, 21.9, 21.10, 21.12_

  - [ ] 10.7 Change the `opencv-python-headless` row in `docs/third-party-licences.md`
    - Its "How we use it" cell adds that the walls step decodes the camera solve's frames and masks, blurs and warps them and writes its demo JPEGs, and it still says that it never reads or writes video. No row is added, for a dependency or for the paper, which the Pipeline_Doc cites (A20).
    - _Requirements: 21.13, 21.14_

  - [ ] 10.8 Write the doc tests in `tests/test_docs.py` and `tests/test_cloud_docs.py`
    - In `tests/test_docs.py`: the step list test looks for the new sentences, with the entries from the real `STEP_LIST`, and checks that the old five-step sentences are gone. `## Step: walls` comes after `## Step: depth` and before `## Step: train`, with its five parts. Its parts name every key of the real `[walls]` and its Preset tables, read by path, every count and timing of Req 19, `work/walls/depth.json`, `walls.json` and `out/walls/`, and the citation's authors, title, ECCV 1998, pages 399–414 and JOSA A 18(8), 2001.
    - Also in `tests/test_docs.py`: `## Step: train` names `work/walls/` and the fallback to `work/depth/`, `## Step: depth` names the walls step, "The cap deadline" names the walls step and `train_reserve_minutes`, the Pipeline_Doc states 50, 100 and 180 minutes, and the `opencv-python-headless` row names the walls step and still says it never reads or writes video.
    - In `tests/test_cloud_docs.py`, `STEP_NEEDS_PLAN` gains `"Correcting walls": ("16 vCPUs", "32 GB RAM")` (C13), and a test checks the sentence that the walls step's needs live in `[walls]`, with `min_vcpus` soft and `max_minutes` cutting the step short.
    - _Requirements: 21.6, 21.7, 21.15_

  - [ ] 10.9 Run the whole suite one last time and commit
    - Run `uv run pytest`. This is the spec's final full-suite run: every test passes, `tests/test_guard.py` and every earlier spec's tests included, on the CPU with no network.
    - Check that `git diff main --stat -- pyproject.toml uv.lock app cloud/image.py cloud/select.py cloud/start.py cloud/runner.py cloud/smoke.py cloud/smoke_runner.py` lists nothing, so no package, website file, image or smoke check changed (Req 21.13).
    - On `feat/walls`: `git add splatpipe/train/__init__.py splatpipe/run.py config.toml docs/pipeline.md docs/third-party-licences.md tests/test_train_step.py tests/test_run.py tests/test_frames_step.py tests/test_cameras_step.py tests/test_imports.py tests/test_walls_settings.py tests/test_docs.py tests/test_cloud_docs.py tests/test_cloud_config.py tests/test_start_core.py`, then commit only those files with `git commit -m "Walls: training reads the walls priors, the step list, the docs and the expected times" -- <the same files>`.
    - The spec ends here and nothing paid runs. Samarth runs the first real Job on the Walls_Step himself, after the tasks are done, and it isn't a task. The website's start page shows the new expected times once the account owner redeploys, which isn't a task either.
    - _Requirements: 21.13, 22.4, 22.12_

## Notes

- Every unit test and property test sub-task is required, as `.kiro/steering/rules.md` and Req 22.4 ask. None is marked optional, so no `*` appears anywhere.
- The plan has 10 tasks, within rules.md's limit. The last sub-task of each task is its checkpoint: the whole suite, then the commit by name. So there are no separate checkpoint tasks, and 10.9 is the final full-suite run.
- Every acceptance criterion in `requirements.md` is named by at least one task. Each of the design's 7 correctness properties has its own property test sub-task, with Hypothesis: Property 5 in 1.10, Properties 1 to 4 in 2.4 to 2.7, Property 6 in 7.3 and Property 7 in 9.7.
- The textured-plane test of Req 22.2 is 5.6, and 6.5 checks the same scene through one whole Frame_Solve. The step-level tests of Req 22.5 and 22.7 are 9.9, the two-worker test of Req 22.6 is 9.8, and the end-to-end Job test of Req 22.10 is in 10.4.
- The Train_Step's behaviour with no `work/walls/` stays exactly as the depth-priors spec left it: 10.2 checks it, and every earlier train test stays in the suite and passes.
- No task runs anything paid. The one paid run left after the spec, Samarth's first real Job, isn't a task. For it, Kiro first tells the person in the chat the machine, the expected time and the rough cost, and only after their yes runs the start command with `APPROVED_SPEND=1`.
- The graph runs the tasks in order, one at a time, so each task's test run and commit see only finished work. Inside a task, sub-tasks that write different files share a wave, tests come after the code they test, sub-tasks that write the same file sit in different waves, and each task's commit gates the next task's first sub-tasks.

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "1.2", "1.3", "1.4", "1.6", "1.7"] },
    { "id": 1, "tasks": ["1.5", "1.8", "1.9"] },
    { "id": 2, "tasks": ["1.10", "1.11"] },
    { "id": 3, "tasks": ["1.12"] },
    { "id": 4, "tasks": ["2.1"] },
    { "id": 5, "tasks": ["2.2"] },
    { "id": 6, "tasks": ["2.3"] },
    { "id": 7, "tasks": ["2.4"] },
    { "id": 8, "tasks": ["2.5"] },
    { "id": 9, "tasks": ["2.6"] },
    { "id": 10, "tasks": ["2.7"] },
    { "id": 11, "tasks": ["2.8"] },
    { "id": 12, "tasks": ["3.1"] },
    { "id": 13, "tasks": ["3.2", "3.3"] },
    { "id": 14, "tasks": ["3.4"] },
    { "id": 15, "tasks": ["3.5"] },
    { "id": 16, "tasks": ["4.1"] },
    { "id": 17, "tasks": ["4.2"] },
    { "id": 18, "tasks": ["4.3"] },
    { "id": 19, "tasks": ["5.1", "5.3"] },
    { "id": 20, "tasks": ["5.2", "5.4"] },
    { "id": 21, "tasks": ["5.5"] },
    { "id": 22, "tasks": ["5.6"] },
    { "id": 23, "tasks": ["5.7"] },
    { "id": 24, "tasks": ["6.1", "6.3"] },
    { "id": 25, "tasks": ["6.2", "6.4"] },
    { "id": 26, "tasks": ["6.5"] },
    { "id": 27, "tasks": ["6.6"] },
    { "id": 28, "tasks": ["7.1"] },
    { "id": 29, "tasks": ["7.2"] },
    { "id": 30, "tasks": ["7.3"] },
    { "id": 31, "tasks": ["7.4"] },
    { "id": 32, "tasks": ["8.1"] },
    { "id": 33, "tasks": ["8.2"] },
    { "id": 34, "tasks": ["8.3"] },
    { "id": 35, "tasks": ["9.1", "9.3", "9.5"] },
    { "id": 36, "tasks": ["9.2", "9.6"] },
    { "id": 37, "tasks": ["9.4", "9.7"] },
    { "id": 38, "tasks": ["9.8", "9.9"] },
    { "id": 39, "tasks": ["9.10"] },
    { "id": 40, "tasks": ["10.1", "10.3", "10.5", "10.6", "10.7"] },
    { "id": 41, "tasks": ["10.2", "10.4", "10.8"] },
    { "id": 42, "tasks": ["10.9"] }
  ]
}
```
