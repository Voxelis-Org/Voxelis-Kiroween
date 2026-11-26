# Implementation Plan: depth-priors

## Overview

This plan builds the Depth_Step in `splatpipe/depth/` and changes the Train_Step so it trains with what the Depth_Step writes, in 10 tasks, each building on the one before: the Model_Reader's point ids and the Prior_Format, the `[depth]` settings and the fit, the Depth_Step on the CPU, the Depth_GPU_Module, training's reading of the Priors, the Back_Projected_Splats, the Prior_Terms, the smoke test's depth check, MoGe on the machine image, and last the Step_List entry and the docs wrap-up. Each task writes its code with that code's unit tests and Hypothesis property tests, runs the whole suite and commits its files by name. The code is Python 3.11 on numpy and OpenCV, which the Project_File already has, so `pyproject.toml` and `uv.lock` don't change (Req 15.5).

The property tests the request asks for by name:

- Property 2, in 2.8: the inverse-depth fit gets the exact scale and offset back when the MoGe_Values are an exact affine of the Targets.
- Property 3, in 2.9: it still gets them back when up to a tenth of the points are moved far off.
- Property 5, in 5.6: the Depth_Weight never goes below `depth_weight_floor` (0.1) on a Plain_Pixel.
- Property 7, in 7.4: with no Prior_Set, an Iteration works out exactly the splat-training spec's five Loss_Terms. 7.5 also checks that `_TERMS` in `gpu.py` is exactly those five.

The design adds three more: Property 1, in 1.5, the Model_Reader's widened round trip; Property 4, in 1.7, the Prior_Format's round trip; and Property 6, in 6.4, the back-projection's round trip.

Training with no priors keeps working exactly as the splat-training spec says, and four sub-tasks check it: 5.7 (a run with no `work/depth/` gives the splat-training counts, timings and warnings, and a Scene with no priors), 6.5 (the same Start_Splats as `choose_points` gives), and 7.4 and 7.5 (the five terms and the five-row `_TERMS`). Every splat-training test stays in the suite and passes.

Four choices in the order matter:

- **The Prior_Format comes first.** Both steps use it and the Model_Reader's Observed_Points, so task 1 builds them in `splatpipe/train/`, and `splatpipe.depth` imports `splatpipe.train`, never the other way.
- **The GPU modules come after everything they use.** Tasks 2 and 3 build and test every CPU part of the Depth_Step on a laptop with `FakeDepthGpu`, and task 4 writes `splatpipe/depth/gpu.py` against that tested interface. Tasks 5 and 6 do the same for the Train_Step's CPU side before task 7 changes the Trainer. The GPU code runs only in the Smoke_Test's depth check, so its laptop tests read its source.
- **The Step_List entry comes last.** Task 10 puts the Depth_Step in `STEP_LIST` once the step, its GPU module, the train changes, the smoke check and the image install are all in. The step list part of the Pipeline_Doc and the tests that assert the Step_List change in that task (Req 18.12).
- **The docs change with the code.** The task that builds or changes a part of a step writes its part of `docs/pipeline.md` and that part's test: task 4 the `## Step: depth` section, tasks 5, 6 and 7 the parts of `## Step: train` they change, task 8 the smoke test part and task 10 the step list and the Step_Needs paragraph. Task 3 changes the `opencv-python-headless` row and task 9 adds and changes the rows for the image's new packages, with rules.md's certifi line (D5).

No task runs anything paid. Building the Machine_Image and running the Smoke_Test are paid runs that the user starts after the spec is built. They are not tasks, and no task sets `APPROVED_SPEND` (Req 15.6).

Some files grow over several tasks, so each piece comes with the first task that uses it:

- `splatpipe/depth/__init__.py`: task 2 makes it with only the package docstring, and task 3 writes the step.
- `splatpipe/depth/gpu_api.py`: task 2 writes `MoGeOutput`, which the fit reads, and task 3 writes the rest.
- `config.toml`: task 2 adds `[depth]`, task 5 the 15 `[train]` keys and task 8 `[cloud.smoke]`'s `depth_normal_start_step`, each in the task whose code first reads it.
- `tests/conftest.py`: task 2 adds `DEPTH`, `depth_text`, `config_text`'s `depth` argument and `set_depth`, task 3 `FakeDepthGpu` and `fake_depth_gpu`, and task 5 the 15 new `TRAIN` keys.
- `tests/camera_files.py`: task 2 adds the plane scene and `plane_moge_output`.
- `tests/test_imports.py`: task 2 adds the depth import check, task 4 `gpu.py`'s source checks, the fresh-process import and `NEVER_PACKAGES`, task 7 the Loss_Table checks, and task 10 the Step_List check's docstring.
- `splatpipe/train/__init__.py`: task 5 reads the Prior_Set and task 6 starts the Back_Projected_Splats.
- `splatpipe/train/schedules.py`: task 5 adds `depth_schedule`, and task 7 the term tables, `depth_normal_start`, `normal_on` and `active_terms`.
- `tests/test_depth_settings.py`: task 2 writes it, and task 10 adds the Fallback_List test.
- `cloud/image.py`: task 2 changes the comment above `MOGE_WEIGHTS`, with the test that keeps `moge_weights` equal to it, and task 9 adds the install step.
- `docs/pipeline.md` and `tests/test_docs.py`: tasks 4, 5, 6, 7, 8 and 10, as above. `docs/third-party-licences.md`: tasks 3 and 9.

## Working rules for every task

- Work on the existing `feat/depth-priors` branch, never on `main`. Build on it as it is, with no rebase, reset or force push.
- The only new step folder is `splatpipe/depth/`. Outside `splatpipe/depth/` and `splatpipe/train/`, only the files in the design's "Changes outside `splatpipe/depth/` and `splatpipe/train/`" table and the test files of its "Tests by file" table change or are made. `pyproject.toml`, `uv.lock`, `cloud/select.py`, `cloud/start.py`, `cloud/runner.py` and the Fallback_Lists stay as they are (Req 8.5, 15.5). `.kiro/steering/rules.md` changes only its named-exceptions line for certifi and tqdm, in task 9, which Samarth OKs (D5). The splat-training requirements file stays as it is, and the Pipeline_Doc says what's true now (A18).
- `splatpipe/depth/gpu.py` is the only file that imports MoGe or huggingface_hub, both inside functions, and it imports torch at its top. `splatpipe/train/gpu.py` stays the only other file that imports torch, and it imports no GPU package besides torch, gsplat and fused_ssim. Every other file in `splatpipe/depth/` and `splatpipe/train/` imports only the standard library, numpy, cv2 and `splatpipe`. `splatpipe.depth` imports `splatpipe.train`, never the other way, and nothing in `splatpipe/` imports `cloud/` or boto3.
- No laptop test imports either GPU module, except to read its source as text. The tests run the GPU calls through `FakeDepthGpu` and `FakeGpu`, and reach `load_depth_gpu`'s import only with `splatpipe.depth.gpu_api.importlib` replaced by a stand-in.
- cv2 only decodes JPEG and PNG stills, converts colours and scales images. It never reads or writes video.
- Every size, count, ratio, weight and path comes from `[depth]`, `[train]` or `[cloud.smoke]`. Code holds only the Fixed_Constants of the requirements' glossary (Req 8.7).
- Write the code fresh. No code from Inria's gaussian-splatting repo, the 2DGS repo or MoGe's repo: the Depth_GPU_Module calls MoGe's API only. `splatpipe/train/gpu.py` keeps its Apache-2.0 note for the code adapted from gsplat's example trainer.
- Load skills with `load_resource` before writing any code: `skill::moge-depth` in tasks 2, 3, 4, 8, 9 and 10; `skill::gsplat-training` in tasks 6, 7 and 8; `skill::splat-files` in task 8; and `skill::aws-jobs` in tasks 8 and 9.
- Nothing here costs money. No task runs the Image_Build, the Smoke_Test, the Start_Command or any `aws` command, and no task sets `APPROVED_SPEND` (Req 15.6).
- Tests run on the CPU with no GPU and no network, on tiny synthetic data that each test builds: models, JPEGs, Priors and PLYs of a few pixels and a few dozen points, under `tmp_path` or a `tempfile.TemporaryDirectory()`. Every Job is made in the test config's Jobs_Folder, never in `data/`, and nothing a test makes goes into git (Req 18.13). Nothing heavy runs on the laptop.
- Property tests use Hypothesis with `@settings(max_examples=100, deadline=None)`, and each docstring says `Feature: depth-priors, Property N: <title>`. A property test that writes files makes a `tempfile.TemporaryDirectory()` per example. Each oracle is written in the test and never calls the function it checks.
- Every unit test and property test sub-task is required. None is optional (Req 18.3).
- Whoever adds a package to the Machine_Image adds its `docs/third-party-licences.md` row in the same task and commit, and stops to ask the user before committing if a licence isn't on rules.md's list.
- The last sub-task of each task is its checkpoint. Run `uv run pytest` and fix what fails until every test passes, `tests/test_guard.py` included. Then stage the task's files by name with `git add <file> ...`, never with `git add -A`, `git add .` or a folder, and commit only those files with `git commit -m "<message>" -- <file> ...`. Leave `docs/devlog.md` out: the devlog hook adds and commits its own line after each task.
- If a hook blocks something it shouldn't, say so and fix the hook. Never work around it.

## Tasks

- [x] 1. The Model_Reader's point ids and the Prior_Format
  - Files: `splatpipe/train/priors.py` and `tests/test_train_priors.py` (new). `splatpipe/train/colmap.py`, `splatpipe/train/smoke.py` and `tests/test_train_colmap.py` (changed).
  - _Requirements: 2.3, 2.4, 2.5, 2.6, 2.9, 6.3, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.11, 7.12, 10.2, 10.4, 14.8, 14.9, 18.1, 18.2, 18.3, 18.13_

  - [x] 1.1 Keep the point ids and the Observed_Points in `splatpipe/train/colmap.py`
    - `ModelImage.observed_ids`, `SparseModel.point_ids` and `observed_rows()`, as in the design's "Changes to the Model_Reader".
    - `_POINT` becomes `"<QdddBBB8xQ"`, which keeps the point id and still reads past the error. Tracks are still skipped. The ids go into one `(P,)` uint64 array.
    - `_read_image` reads the keypoint count, checks that the 24 × count bytes are there with `_Cursor._need`, and reads them as one `np.frombuffer` structured array (`<f8`, `<f8`, `<i8`), keeping only the id column. It keeps the ids of 0 or more, distinct, in the order of their first keypoint. x and y are never kept.
    - `read_model` turns each image's kept ids into uint64 first, then keeps only the ids that are ids of points, with `np.isin` against `point_ids`, so both arrays are uint64 and no id is compared as a float64. A keypoint id of −1 or of no point is no observation, and no model is refused for it, so the reader refuses exactly the models it refused before.
    - `observed_rows` maps each observed id to the row of the first point with that id, through a sorted copy of `point_ids` and `np.searchsorted`.
    - The module docstring says what the reader keeps now. The splat-training requirements file stays as it is (A18).
    - _Requirements: 2.3, 2.4, 2.5, 2.6, 14.8_

  - [x] 1.2 Pass the new fields in `splatpipe/train/smoke.py`
    - `smoke_scene` gives point ids 1 to P and an empty uint64 `observed_ids` for every image, so the training check's run is unchanged.
    - _Requirements: 2.4_

  - [x] 1.3 Write the Prior_Format in `splatpipe/train/priors.py`
    - The Fixed_Constants `PRIOR_FOLDER`, `PRIOR_INDEX`, `PRIOR_FORMAT` and `PRIOR_ARRAYS`, and `Prior`, `PriorEntry`, `PriorIndex`, `PriorSet`, `BrokenPriors`, `prior_path()`, `write_prior()`, `write_index()`, `load_prior()`, `load_index()`, `check_prior()` and `read_prior_set()`, as in the design's "The Prior_Format" sections. The module imports only the standard library, numpy, `splatpipe.jsonio` and `splatpipe.train.colmap` for types. It imports no torch and nothing from `splatpipe.depth`.
    - `write_prior` is `np.savez`, uncompressed, with the three arrays as little-endian float32, float32 and bool. `write_index` writes the index's JSON_Values with `write_json` to `depth.json.tmp` in the same folder, then `os.replace`s it to `depth.json`. On an `OSError` it removes the temporary file, then raises.
    - `load_prior` reads with `np.load(path, allow_pickle=False)` as a context manager and wants exactly the three names. `load_index` reads with `read_json` and checks every field's JSON type exactly, so `true` is never 1. A file that can't be read or parsed, or a field of the wrong type, is a `BrokenPriors`.
    - `check_prior` checks each array's shape at `prior_height` × `prior_width`, its dtype, that every value is finite, and that `inverse_depth` is above 0 where `mask` is true and 0 where it's false.
    - `read_prior_set(folder, model)` gives `None` when `folder / PRIOR_INDEX` isn't a file. Otherwise it runs the design's checks in the design's order and reads each Prior_File at `prior_path(folder, name)` from a checked name, never from a path in the index. The first check that fails raises `BrokenPriors(path, what)`, so one bad file drops the whole Prior_Set (A9). Each `what` is worded like the design's Error Handling examples.
    - _Requirements: 6.3, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.11, 10.2, 10.4_

  - [x] 1.4 Write unit tests for the point ids in `tests/test_train_colmap.py`
    - With `write_model`'s records, an image whose keypoints hold −1, ids of no point and the same id twice gives `observed_ids` that are distinct, in first-keypoint order, and only ids of points. No model is refused for them, and a model whose keypoints are all −1 is read with every `observed_ids` empty.
    - Every splat-training refusal is still refused with its message, and a keypoint block cut short is refused with the reader's "cut short" message, naming `images.bin`.
    - A keypoint id of 2⁵³ + 1 that names no point, next to a point whose id is 2⁵³, is no observation, so the ids are never compared as floats.
    - `observed_rows` maps ids to rows, and a point id written twice maps to its first row.
    - No message holds "Traceback".
    - _Requirements: 2.4, 2.5, 2.6, 18.1_

  - [x] 1.5 Widen the property test for Property 1 in `tests/test_train_colmap.py`, with Hypothesis
    - **Property 1: The Model_Reader reads back what was written, point ids and Observed_Points included**
    - **Validates: Requirements 2.9, 2.4, 2.5, 2.6**
    - Widen the splat-training spec's Property 1 test as the design's strategy says: point ids from 0 to 2⁶³ − 1, a few at 2⁶³ and above, and duplicates now and then; keypoint ids from −1, from the points' ids, from ids of no point, and repeated within an image. The docstring keeps the splat-training tag and adds `Feature: depth-priors, Property 1: ...`.
    - Check that the camera, the images, the point positions, colours and point ids come back bit for bit, and that each image's `observed_ids` equal the oracle's, which walks the keypoints in order with a `set`.
    - _Requirements: 2.4, 2.5, 2.6, 2.9, 18.2_

  - [x] 1.6 Write unit tests for the Prior_Format in `tests/test_train_priors.py`
    - `read_prior_set` gives `None` with no `depth.json`, and with Prior_Files but no index (10.2).
    - Each check of Req 10.4 is refused with a `BrokenPriors` naming the file and what failed: JSON that won't parse, `format` 2 and `true`, each camera value changed, a size of 0, no priors, an unknown name, a repeated name, a changed quaternion value and translation value, a missing `.npz`, a file with a fourth array, an array of another shape or dtype, a NaN, a 0 `inverse_depth` under a true mask and a non-zero one under a false mask.
    - An `.npz` holding an object array is refused by `np.load` with `allow_pickle=False` (7.5). The index names no path (7.4). `prior_path` of `000012.jpg` is `000012.npz` in the folder.
    - `write_index` leaves only `depth.json`, and with `os.replace` raising `OSError(errno.ENOSPC, "No space left on device")` it leaves neither `depth.json` nor `depth.json.tmp` (7.6).
    - _Requirements: 7.4, 7.5, 7.6, 10.2, 10.4, 18.1, 18.13_

  - [x] 1.7 Write the property test for Property 4 in `tests/test_train_priors.py`, with Hypothesis
    - **Property 4: The Prior_Reader reads back what the Prior_Writer wrote, bit for bit**
    - **Validates: Requirements 7.12, 7.1, 7.3**
    - Draw as the design's Property 4 strategy says: 1 to 4 entries with unique names of lowercase ASCII letters, digits, `_` and `-` that start with a digit, plus `.jpg`, so each is a file name on Windows too; finite values with −0.0, subnormals and values near ±1e308; Prior_Sizes from 1 × 1 to 8 × 8; and float32 array values with no NaN or infinity. Each example writes into its own `tempfile.TemporaryDirectory()`.
    - Check that `load_index` gives back every index value, floats compared through `struct.pack("<d", x)`, and that `load_prior` gives back the three arrays with the same shapes, dtypes and bits, compared through `.view(np.uint32)`. Each `.npz` holds exactly the three names.
    - _Requirements: 7.1, 7.3, 7.12, 18.2_

  - [x] 1.8 Run the tests and commit
    - Run `uv run pytest`. Every test passes, `tests/test_guard.py` included. The train import check and the fresh-process import find `priors.py` by glob, so it imports no torch (14.9).
    - On `feat/depth-priors`: `git add splatpipe/train/colmap.py splatpipe/train/smoke.py splatpipe/train/priors.py tests/test_train_colmap.py tests/test_train_priors.py`, then commit those files with the message "Depth: the model reader's point ids and the prior format".
    - _Requirements: 14.9, 18.3, 18.13_

- [x] 2. The `[depth]` settings and the fit
  - Files: `splatpipe/depth/__init__.py`, `splatpipe/depth/settings.py`, `splatpipe/depth/gpu_api.py`, `splatpipe/depth/fit.py`, `tests/test_depth_settings.py` and `tests/test_depth_fit.py` (new). `config.toml`, `cloud/image.py`, `tests/conftest.py`, `tests/camera_files.py` and `tests/test_imports.py` (changed).
  - _Requirements: 3.4, 4.3, 4.6, 4.7, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 5.10, 5.11, 5.12, 5.13, 6.1, 6.2, 6.4, 8.1, 8.2, 8.3, 8.4, 8.6, 8.7, 8.8, 15.11, 18.1, 18.2, 18.3, 18.8, 18.10, 18.13_

  - [x] 2.1 Add `[depth]` to `config.toml` and change `MOGE_WEIGHTS`'s comment
    - Put `[depth]` and its three Preset tables between `[frames.quality]` and `[train]`, as the design's "The `[depth]` section of `config.toml`" gives them: the 8 Shared_Settings with the comment above each key, and `[depth.fast]`, `[depth.default]` and `[depth.quality]` empty, each with the comment that every `[depth]` setting is shared.
    - In `cloud/image.py`, the comment above `MOGE_WEIGHTS` says that the Depth_Step finds the Weights_File through `moge_weights` in `[depth]`, which a test keeps equal to `MOGE_WEIGHTS`, in place of "The depth-priors spec decides how its step finds the file". Nothing else in `cloud/image.py` changes here.
    - _Requirements: 8.1, 8.2, 8.7, 15.11_

  - [x] 2.2 Write `splatpipe/depth/settings.py`
    - Make `splatpipe/depth/` with an `__init__.py` that holds only the package docstring for now. Task 3 writes the step in it.
    - `DepthSettings` and `read_settings(settings, section, preset)`, as in the design's "`splatpipe/depth/settings.py`" section, with its own small table of kinds, so it imports nothing private from the Train_Step. Each key is read with `settings[key]`, so a missing key raises the Step_Runner's own message, and each value's `type()` is checked exactly, so a bool is never a number.
    - A refusal is an `ExpectedFailure`: "The setting `<key>` in [depth] is `<value>`, but it must be `<kind>`. Change it to `<kind>`.", with the value shown as TOML writes it.
    - `min_gpu_memory_gb` is checked and kept, and nothing in the step uses it (3.4).
    - The module imports only the standard library and `splatpipe.errors`.
    - _Requirements: 3.4, 8.3, 8.6, 8.7_

  - [x] 2.3 Write `MoGeOutput` and `splatpipe/depth/fit.py`
    - Load `skill::moge-depth` with `load_resource` before writing code.
    - Start `splatpipe/depth/gpu_api.py` with its module docstring and `MoGeOutput`, as the design gives it. It imports only the standard library and numpy. Task 3 adds the rest.
    - `field_of_view()`, `geometry_pixels()`, `FitPoints`, `fit_points()`, `spread()`, `FitOutcome`, `depth_fit()`, `prior_arrays()` and `job_metres_per_unit()`, as in the design's "`splatpipe/depth/fit.py` (pure)" section. The module imports no torch.
    - `fit_points` keeps the Observed_Points in front of the camera, whose projection with the Shared_Camera at its full size lands inside the Frame, on a Prior_Size pixel (min(⌊u · w / W⌋, w − 1), min(⌊v · h / H⌋, h − 1)) that is a Geometry_Pixel. Targets are 1 / z and MoGe_Values 1 / MoGe's depth there, both float64.
    - `depth_fit` runs the design's steps 1 to 8 in float64 and gives `"few_points"`, `"flat"`, `"poor_fit"` or `"fitted"`. The median Relative_Depth_Error is over every Fit_Point, the dropped ones included, worked out as |t / p − 1|, and is infinite where p isn't above 0.
    - `prior_arrays` works out the fitted value in float64 inside `np.errstate`, rounds it to float32, and keeps a pixel only where it's a Geometry_Pixel, the value is finite and above 0, and every normal component is finite. Every other pixel gets 0s and `mask` false.
    - _Requirements: 4.3, 4.6, 4.7, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 5.10, 5.11, 6.1, 6.2, 6.4_

  - [x] 2.4 Add `[depth]`, `set_depth` and the plane scene to the test helpers
    - In `tests/conftest.py`: `DEPTH`, the 8 settings at their real values with their real TOML types; `depth_text(depth)`; a `depth` argument to `config_text`, which writes `[depth]` and its three empty Preset tables between `[frames]` and `[train]`; and a `set_depth(**values)` fixture built like `set_frames`. The module docstring names them.
    - In `tests/camera_files.py`, the plane scene: four cameras of 16 × 12 pixels that look at the plane z = 3 at an angle, from different places, so no Frame is flat; 30 points on the plane, each seen by every camera and no two on one pixel of any image; each image's keypoints naming its points, with a few −1s and ids of no point mixed in. The records work with `write_model` and `write_camera_solve_output`, so the Frames are JPEGs of one colour.
    - `plane_moge_output(model, image, width, height, metres_per_unit, no_geometry=())` gives a `MoGeOutput` at the Prior_Size: at each pixel an Observed_Point lands on, that point's z × `metres_per_unit`, and elsewhere the plane's depth at the pixel centre; the plane's normal facing the camera; and a mask that's true except at the pixels in `no_geometry`. So every Frame fits exactly, with the Metres_Per_Unit the test chose. It imports nothing from `splatpipe.train.colmap`.
    - _Requirements: 8.8, 18.1, 18.13_

  - [x] 2.5 Add the depth import check to `tests/test_imports.py`
    - A new test reads every `splatpipe/depth/*.py` with the file's `ast` reader. Every file but `gpu.py` imports only names in `sys.stdlib_module_names`, numpy, cv2 and `splatpipe`, and `gpu.py` imports only those and torch, moge and huggingface_hub. A failure gives the file and the line. The test finds the files with glob, so each later file is checked as it lands. Task 4 checks where `gpu.py` imports them.
    - _Requirements: 18.8_

  - [x] 2.6 Write unit tests for the settings in `tests/test_depth_settings.py`
    - The real config, read by path, has `[depth]` with exactly the 8 Shared_Settings and values of Req 8.2 and three empty Preset tables, and `step_settings("depth", preset)` loads exactly the 8 keys for each Preset. The comment check in `tests/test_config.py` already covers the new keys.
    - The real `moge_weights` equals `cloud.image.MOGE_WEIGHTS`.
    - Parametrised over every key: a value outside its kind, a bool where a number belongs and a number for `use_fp16` give an `ExpectedFailure` naming the key, `[depth]` and the kind. `resolution_level` −1 and 10 and an empty `moge_weights` are refused, and `resolution_level` 0 and 9 pass.
    - A missing key gives the Step_Runner's message naming `[depth]`, `[depth.<preset>]` and the key. `set_depth` changes one key for one test, and the next test sees the test config's values again.
    - The Fallback_List test comes in task 10, with the Step_List.
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.6, 8.8, 18.1, 18.10_

  - [x] 2.7 Write unit tests for the fit in `tests/test_depth_fit.py`
    - `field_of_view(1920, 960)` is 90°. `geometry_pixels` leaves out a false mask, an infinite depth and a depth of 0.
    - `fit_points` on the plane scene keeps the points in front, inside the Frame and on a Geometry_Pixel, and leaves out one behind the camera, one outside the image, one on a pixel with no geometry, one whose keypoint id was −1 and one whose id names no point.
    - `depth_fit` gives `"few_points"` for 9 points and for a second fit left with 9, `"flat"` for equal MoGe_Values in the first fit and in the second and for a second-fit scale of 0, and `"poor_fit"` when the dropped points alone push the median error over. A hand-made set whose median error is exactly 0.15 fits and one just above doesn't.
    - `prior_arrays` gives 0s and `mask` false at a negative fitted value, at one that rounds to 0 or overflows in float32, and at a non-finite normal.
    - The plane scene's Metres_Per_Unit is the one chosen, per Frame and for the job.
    - _Requirements: 4.3, 4.7, 5.1, 5.2, 5.3, 5.6, 5.7, 5.8, 5.9, 5.11, 6.1, 6.2, 18.1_

  - [x] 2.8 Write the property test for Property 2 in `tests/test_depth_fit.py`, with Hypothesis
    - **Property 2: Exactly affine data is recovered**
    - **Validates: Requirements 5.12, 5.4**
    - Draw as the design's Property 2 strategy says: n from 20 to 100 distinct MoGe_Values k / 20 for whole numbers k from 1 to 400, a scale log-uniform from 10⁻³ to 10³, and an offset that keeps every Target above 0, with the Targets the scale × MoGe_Value + the offset in float64.
    - Check that `depth_fit` with the real `[depth]` settings gives `"fitted"`, the scale within a relative error of 10⁻⁹ and the offset within 10⁻⁹ × max(|offset|, the largest Target). So the fit gets the exact scale and offset back.
    - _Requirements: 5.4, 5.12, 18.2_

  - [x] 2.9 Write the property test for Property 3 in `tests/test_depth_fit.py`, with Hypothesis
    - **Property 3: Up to a tenth of points moved far off are dropped, and the true fit is recovered**
    - **Validates: Requirements 5.13, 5.5**
    - Draw as the design's Property 3 strategy says: an arithmetic progression of 20 to 100 MoGe_Values in a shuffled order, the scale and offset as in Property 2, and 1 to ⌊n / 10⌋ Targets each moved up by λ × R, with λ from 6 to 7 and R the true Targets' range.
    - Check that `depth_fit` gives `"fitted"` with the true scale and offset within the bounds of Property 2, and that `kept_points` equals the oracle's count, which leaves out exactly the moved points. So the fit still gets them back when a few points are way off.
    - _Requirements: 5.5, 5.13, 18.2_

  - [x] 2.10 Run the tests and commit
    - Run `uv run pytest`. Every test passes, `tests/test_guard.py` included.
    - On `feat/depth-priors`: `git add config.toml cloud/image.py splatpipe/depth/__init__.py splatpipe/depth/settings.py splatpipe/depth/gpu_api.py splatpipe/depth/fit.py tests/conftest.py tests/camera_files.py tests/test_imports.py tests/test_depth_settings.py tests/test_depth_fit.py`, then commit those files with the message "Depth: the [depth] settings and the fit".
    - _Requirements: 18.3, 18.13_

- [x] 3. The depth step on the CPU, its stand-in and its CLI
  - Files: `splatpipe/depth/__main__.py`, `splatpipe/depth/files.py`, `splatpipe/depth/report.py`, `tests/test_depth_files.py`, `tests/test_depth_gpu_api.py` and `tests/test_depth_step.py` (new). `splatpipe/depth/__init__.py`, `splatpipe/depth/gpu_api.py`, `splatpipe/train/files.py`, `tests/conftest.py`, `docs/third-party-licences.md` and `tests/test_docs.py` (changed).
  - _Requirements: 1.1, 1.2, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 1.11, 2.1, 2.2, 2.3, 2.7, 2.8, 3.1, 3.2, 3.3, 3.4, 3.5, 3.8, 3.11, 3.13, 3.14, 3.15, 4.1, 4.2, 4.4, 4.5, 6.2, 6.3, 7.6, 7.7, 7.8, 7.9, 7.10, 8.3, 8.6, 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 17.7, 18.1, 18.3, 18.4, 18.6, 18.7, 18.13_

  - [x] 3.1 Finish `splatpipe/depth/gpu_api.py`
    - Load `skill::moge-depth` with `load_resource` before writing code.
    - `DEPTH_GPU_MODULE`, the `DepthGpu` protocol, `MissingPackage`, `WrongCheckpoint`, `DepthOutOfMemory`, `OfflineModeOff`, `load_depth_gpu()`, `check_machine()`, `load_moge()` and `run_moge()`, as in the design's "`splatpipe/depth/gpu_api.py`" section. The module imports no torch, MoGe or huggingface_hub.
    - `load_depth_gpu` calls `importlib.import_module(DEPTH_GPU_MODULE)` through this module's own `import importlib`, so a test can replace it. A `ModuleNotFoundError` whose name, up to its first dot, is `torch` becomes the Req 3.2 failure naming torch, and anything else goes up as it is.
    - `check_machine` refuses no CUDA (3.3), then a `moge_weights` that isn't a file, before `import_moge` runs (3.5), then a `MissingPackage` from `import_moge`, naming the package (3.2). It checks no GPU memory (3.4). `OfflineModeOff` is a `RuntimeError` and goes up as it is (3.8).
    - `load_moge` turns `WrongCheckpoint` into the Req 3.11 failure and `DepthOutOfMemory` into the Req 3.13 failure "while the depth step loaded MoGe-2". `run_moge` passes `resolution_level` and `use_fp16` from the settings and turns `DepthOutOfMemory` into the Req 3.13 failure naming the Frame.
    - Each message is worded as the design's Error Handling table gives it.
    - _Requirements: 3.2, 3.3, 3.4, 3.5, 3.8, 3.11, 3.13, 3.15, 4.4_

  - [x] 3.2 Write `splatpipe/depth/files.py` and `splatpipe/depth/report.py`
    - In `splatpipe/train/files.py`, `_training_image` and `_scaled_down` become public, `read_training_image` and `scaled_down`, with their callers changed and their behaviour the same, so the Depth_Step decodes frames with the Train_Step's code and Expected_Failures.
    - `check_inputs()`, `read_frame()`, `write_prior_file()` and `write_prior_index()`, as in the design's "`splatpipe/depth/files.py`" section. `check_inputs` checks `images/`, then each model file, with `train.files`' names, and refuses the first one missing with the Req 2.2 message, reading nothing. `read_frame` decodes with `read_training_image`, scales down with `INTER_AREA` when the Frame is larger than the Prior_Size, and gives the (h, w, 3) uint8 RGB frame. The two writers turn an `OSError` into the Req 7.10 message.
    - `Tally`, `Timings`, `left_out_warning()` and `no_fit_failure()`, as in the design's "`splatpipe/depth/report.py`" section, with the Req 7.9 and Req 7.7 texts.
    - This is the commit that brings the depth step's use of OpenCV in, so in `docs/third-party-licences.md` the `opencv-python-headless` row's "How we use it" also says that the depth step decodes the camera solve's JPEG frames and scales them down. It keeps "It never reads or writes video".
    - _Requirements: 2.2, 2.3, 2.7, 4.1, 4.2, 7.6, 7.7, 7.9, 7.10, 9.1, 9.2, 17.7_

  - [x] 3.3 Write the step in `splatpipe/depth/__init__.py` and `splatpipe/depth/__main__.py`
    - `DEPTH_GPU: Callable[[], DepthGpu] = load_depth_gpu`, `run()` and `work()`, as in the design's "The depth step's flow" section. `run` calls `run_step("depth", "splatpipe.depth", ...)`. `work` reads `DEPTH_GPU` when it runs, so a test's `monkeypatch` reaches it through `run`, `run_step`, the CLI and `run_job`, and `import splatpipe.depth` imports no torch, MoGe or huggingface_hub.
    - `work` follows the design's 11 numbered steps in the order of Req 1.4: the settings, `check_inputs`, `DEPTH_GPU()`, `check_machine`, the load inside `step.timed("load_model")`, `read_model` with `images`, `prior_width` and `prior_height`, the Field_Of_View, then each Frame in `images.bin`'s order with its `infer`, `fit` and `write` blocks.
    - The `finally` that starts just before the load calls `release(handle)`, with `None` when the load failed, then adds `peak_gpu_memory_bytes`. With no Fitted_Frame the step raises `no_fit_failure`, leaving `work/depth/` empty. With some Frames left out it adds the Req 7.9 warning. The Prior_Index is written last, with the job's Metres_Per_Unit.
    - Each count is added with `int(...)` as soon as it's known and each timing as a running sum, so a failed report holds exactly what the step had worked out.
    - The step writes only `work/depth/<stem>.npz`, `work/depth/depth.json.tmp` and `work/depth/depth.json`, and relies on the Step_Runner emptying `work/depth/` first.
    - `__main__.py`'s `main(argv)` calls `step_main("depth", "splatpipe.depth", argv)`, and the file ends with `if __name__ == "__main__": sys.exit(main())`.
    - _Requirements: 1.1, 1.2, 1.4, 1.7, 1.8, 1.9, 1.11, 2.1, 2.8, 3.1, 3.14, 3.15, 4.1, 4.5, 6.2, 6.3, 7.6, 7.7, 7.8, 7.9, 9.1, 9.2, 9.3, 9.4, 9.5, 9.6_

  - [x] 3.4 Add `FakeDepthGpu` and `fake_depth_gpu` to `tests/conftest.py`
    - `FakeDepthGpu`, the Depth_Stand_In, as the design's "Stand-ins" section describes it. It's built with whether CUDA is there and records every call in order: `probe`, `import_moge`, `load_model` with its path, each `infer` with its arguments, `release` with its handle, and `peak_memory_bytes`, which gives a number the test sets.
    - `infer` gives the MoGe_Output the test hands it for each Frame, or by default `plane_moge_output` of the plane scene. It can be set to raise `MissingPackage("moge")` or `OfflineModeOff` from `import_moge`, `WrongCheckpoint` or `DepthOutOfMemory(None)` from `load_model`, or `DepthOutOfMemory(name)` from the n-th `infer`.
    - The `fake_depth_gpu` fixture installs each one it builds with `monkeypatch.setattr("splatpipe.depth.DEPTH_GPU", lambda: gpu)`. `FakeDepthGpu` imports what it needs from `splatpipe.depth` inside its methods and the fixture, so conftest imports no step package at module level.
    - _Requirements: 1.8, 18.4_

  - [x] 3.5 Write unit tests for the files and the GPU interface
    - In `tests/test_depth_files.py`: `check_inputs` refuses each of the four missing paths in turn, naming the path and the camera solve step, and reads nothing. `read_frame` scales a JPEG of a known BGR colour down with `INTER_AREA` to the Prior_Size of a camera wider than 512 pixels and gives RGB, and keeps a camera smaller than 512 at its size. It refuses a missing file, junk bytes and a JPEG one pixel wider with the Train_Step's messages. `write_prior_file` and `write_prior_index` with `np.savez`, `write_json` or `os.replace` raising `OSError(errno.ENOSPC, "No space left on device")` give the Req 7.10 message, and the index write leaves no `depth.json` or `depth.json.tmp`.
    - In `tests/test_depth_gpu_api.py`: `load_depth_gpu`, with `splatpipe.depth.gpu_api.importlib` replaced by a stand-in, turns a missing `torch` into the Req 3.2 failure, lets another missing name through, and asks for `DEPTH_GPU_MODULE`, which is `"splatpipe.depth.gpu"`. `check_machine` refuses no CUDA, then a missing weights file with no `import_moge` call, then `MissingPackage("utils3d")` naming it, in that order, and checks no memory. `load_moge` and `run_moge` turn `WrongCheckpoint` and `DepthOutOfMemory` into their messages, naming the frame, and `run_moge` passes `resolution_level` and `use_fp16` from the settings.
    - In `tests/test_docs.py`, the `opencv-python-headless` row's test also checks that the row names the depth step's JPEG frames, and still that it never reads or writes video.
    - No message holds "Traceback".
    - _Requirements: 2.2, 2.7, 3.2, 3.3, 3.4, 3.5, 3.11, 3.13, 4.1, 4.2, 4.4, 7.6, 7.10, 17.7, 18.1, 18.7_

  - [x] 3.6 Write unit tests for a done run in `tests/test_depth_step.py`
    - Run the step through `splatpipe.depth.run` with `FakeDepthGpu` and the plane scene's Camera_Solve_Output, so the Step_Runner and the real report are part of each test.
    - A done run leaves one `.npz` per Frame and `depth.json`, whose fields and Metres_Per_Unit are the plane's and which `read_prior_set` reads back against the model. The report holds every count of Req 9.1 as an `int`, the four timings and the 8 settings.
    - The frames go to `infer` in `images.bin`'s order with the Field_Of_View, a junk file in `images/` and a `masks/` folder are never opened, and a second run after a junk file is put in `work/depth/` leaves no junk.
    - One Frame whose `no_geometry` leaves too few Fit_Points gives the Req 7.9 warning with the counts.
    - Listing the Job_Folder before and after a done run shows only `work/depth/` files and `out/reports/depth.json` new.
    - _Requirements: 1.9, 1.11, 2.1, 2.8, 4.5, 6.2, 6.3, 7.1, 7.3, 7.4, 7.9, 9.1, 9.2, 9.4, 9.5, 9.6, 18.4, 18.6_

  - [x] 3.7 Write unit tests for failed runs, the release and the CLI in `tests/test_depth_step.py`
    - The order of Req 1.4 from the stand-in's call log and from pairs of faults: a bad setting with no `work/cameras/` gives the setting's message, no `work/cameras/` gives the camera solve message with no stand-in call, a refusing probe with a corrupt `points3D.bin` gives the GPU message, and the model is read only after `load_model`.
    - Every Expected_Failure that needs no GPU, each naming what its requirement says, with `expected: true` and no "Traceback". `OfflineModeOff` gives `expected: false` with its message in `technical`.
    - No Frame fitting fails with the Req 7.7 message and leaves `work/depth/` empty. `DepthOutOfMemory` on the third Frame fails naming it, leaves no index, and the report holds the counts and timings of the first two.
    - `release` is called once after a done run, a no-fit run, an out-of-memory run and a failed load, and never before `load_model`.
    - `splatpipe.depth.__main__.main(["--job", job_id])` prints `depth: done` and returns 0, or prints the Readable_Error on stderr with no traceback and returns 1.
    - `run_job` with `steps=(StepEntry("depth", "splatpipe.depth", False), StepEntry("train", "splatpipe.train", True))`, both stand-ins and the plane scene gives `done`. With the depth stand-in refusing, it gives `done with warnings`, with the depth step's entry `failed` and its message.
    - _Requirements: 1.2, 1.4, 1.5, 1.6, 1.7, 3.1, 3.8, 3.13, 3.14, 7.7, 7.8, 8.3, 8.6, 9.3, 18.4, 18.7_

  - [x] 3.8 Run the tests and commit
    - Run `uv run pytest`. Every test passes, `tests/test_guard.py` and the splat-training tests included.
    - On `feat/depth-priors`: `git add splatpipe/depth/__init__.py splatpipe/depth/__main__.py splatpipe/depth/gpu_api.py splatpipe/depth/files.py splatpipe/depth/report.py splatpipe/train/files.py tests/conftest.py tests/test_depth_files.py tests/test_depth_gpu_api.py tests/test_depth_step.py docs/third-party-licences.md tests/test_docs.py`, then commit those files with the message "Depth: the step on the CPU, its stand-in and its CLI".
    - _Requirements: 18.3, 18.13_

- [x] 4. The Depth_GPU_Module and the depth section of the Pipeline_Doc
  - Files: `splatpipe/depth/gpu.py` (new). `tests/test_imports.py`, `tests/test_cloud_imports.py`, `docs/pipeline.md` and `tests/test_docs.py` (changed).
  - _Requirements: 3.2, 3.6, 3.7, 3.8, 3.9, 3.10, 3.11, 3.12, 3.13, 3.14, 3.15, 4.2, 4.4, 4.6, 9.1, 15.5, 17.1, 17.2, 17.9, 18.3, 18.8, 18.9, 18.13_

  - [x] 4.1 Write `splatpipe/depth/gpu.py`
    - Load `skill::moge-depth` with `load_resource` before writing code, and follow MoGe's API at commit `74fbce05` exactly as the skill and the design's "`splatpipe/depth/gpu.py`" section give it. The docstring says the module holds no code copied from MoGe and calls MoGe's API only.
    - It imports the standard library, numpy and torch at its top, `splatpipe.depth.gpu_api`, and huggingface_hub and moge only inside `import_moge` and `load_model`. Nowhere in its text, comments and docstrings included, does it name `from_pretrained`, `hf_hub_download` or `init_weights`, because 4.2's check refuses the names.
    - `probe()`: `torch.cuda.is_available()`, and with CUDA `torch.cuda.reset_peak_memory_stats()`.
    - `import_moge()`: `os.environ["HF_HUB_OFFLINE"] = "1"` as its first statement, then `import huggingface_hub.constants` and `from moge.model.v2 import MoGeModel` inside a `try` that turns `ModuleNotFoundError` into `MissingPackage` of the top-level name, then `OfflineModeOff` with the design's message when `huggingface_hub.constants.HF_HUB_OFFLINE` isn't `True`.
    - `load_model(weights)`: `torch.load(path, map_location="cpu", weights_only=True)`, `MoGeModel(**checkpoint["model_config"])`, `load_state_dict(checkpoint["model"])` with its default `strict=True`, then `.to("cuda").eval()` in float32, giving back `_Loaded(net)`. Each refusal of the decisions table becomes `WrongCheckpoint` with a short reason, and `torch.cuda.OutOfMemoryError` around the `.to("cuda")` alone becomes `DepthOutOfMemory(None)`, raised `from None`.
    - `infer(...)`: the RGB frame to a (3, h, w) float tensor from 0 to 1 on the GPU, `model.net.infer(image, fov_x=..., resolution_level=..., use_fp16=...)` with `force_projection` and `apply_mask` at MoGe's defaults, and a `MoGeOutput` of numpy arrays: `depth` (h, w) float32, `normal` (h, w, 3) float32 and `mask` (h, w) bool. Out of memory becomes `DepthOutOfMemory(frame)`, raised `from None`.
    - `release(model)` sets the handle's network to `None`, runs `gc.collect()` and `torch.cuda.empty_cache()`, and takes `None` too. `peak_memory_bytes()` is `torch.cuda.max_memory_allocated()`.
    - _Requirements: 3.2, 3.6, 3.7, 3.8, 3.9, 3.10, 3.11, 3.12, 3.13, 3.14, 4.2, 4.4, 4.6, 9.1_

  - [x] 4.2 Write `gpu.py`'s source checks and the fresh-process import in `tests/test_imports.py`
    - Read `splatpipe/depth/gpu.py` as text and with `ast`, and never import it. It imports torch only at its top, and moge and huggingface_hub only inside functions. Its text sets `HF_HUB_OFFLINE` on a line before those imports, passes `weights_only=True`, and names none of `from_pretrained`, `hf_hub_download` and `init_weights`.
    - `splatpipe.depth.gpu_api.DEPTH_GPU_MODULE` with its dots turned into folders, plus `.py`, is `splatpipe/depth/gpu.py`, so the module `load_depth_gpu` asks for is the one these checks read.
    - A fresh process that imports `splatpipe.depth` and each of its CPU modules, every `splatpipe/depth/*.py` but `gpu.py` found with glob, has none of torch, moge, utils3d_moge and huggingface_hub in `sys.modules`.
    - `NEVER_PACKAGES` gains `moge`, `utils3d-moge`, `huggingface-hub` and `hf-xet`, so the dependency check refuses them in `pyproject.toml` and `uv.lock`. The module docstring says so.
    - _Requirements: 3.6, 3.9, 3.10, 3.15, 15.5, 18.8_

  - [x] 4.3 Change the GPU package check in `tests/test_cloud_imports.py`
    - `GPU_PACKAGES` gains `huggingface_hub`, `utils3d_moge`, `utils3d` and `hf_xet`. The check keeps refusing `cloud` and boto3 in every file under `splatpipe/`, and every GPU package in every file but the two GPU modules: `splatpipe/train/gpu.py` may import torch, gsplat and fused_ssim, and `splatpipe/depth/gpu.py` may import torch, moge and huggingface_hub. The test's docstring and the module docstring say so.
    - _Requirements: 18.9_

  - [x] 4.4 Write `## Step: depth` in `docs/pipeline.md`
    - Put it after `## Step: frames` and before `## Step: train`, built like the train section, as the design's "The Pipeline_Doc" section gives it: a paragraph with the CLI, `uv run python -m splatpipe.depth --job <id> [--preset P]`, and `splatpipe.depth.run(job_id, preset=None)`, that the step is optional, and the order of its work. Then `### What it reads`, `### What it writes`, `### Config section`, `### Step report fields` and `### Files in out/`.
    - What it reads: the Camera_Solve_Output's frames and Sparse_Model, read with the train step's reader, and no masks; `[depth]`; the MoGe-2 weights at `moge_weights`. That it runs only on a GPU machine image with MoGe's code, loads the weights by path with `HF_HUB_OFFLINE=1`, and downloads nothing.
    - What it writes: the Prior_Format, `work/depth/<stem>.npz` with `inverse_depth`, `normal` and `mask`, their dtypes, units, camera frame and pixel centres, and `work/depth/depth.json` with its fields, written last. Then the fit in words: the Fit_Points, the two passes, the 15% check, the Metres_Per_Unit, and that nothing is rescaled. Task 5 adds how the train step scales a Prior up.
    - Config section: every `[depth]` key with its table and value, the empty Preset tables, and the need of a GPU with 16 GB, which the step never checks. Step report fields: every count as `counts.<name>` and timing as `timings.<name>`, the warning, the kinds of Expected_Failure, and that the job goes on and trains without priors when the step fails. Files in out/: none but its report.
    - _Requirements: 17.1, 17.2_

  - [x] 4.5 Write the depth section's doc tests in `tests/test_docs.py`
    - `## Step: depth` comes after `## Step: frames` and before `## Step: train`, with its five parts. Its parts name every key of the real `[depth]`, read by path, every count and timing of the Depth_Report as `counts.<name>` and `timings.<name>`, `work/depth/depth.json`, and `inverse_depth`, `normal` and `mask`.
    - _Requirements: 17.1, 17.2, 17.9_

  - [x] 4.6 Run the tests and commit
    - Run `uv run pytest`. Every test passes, `tests/test_guard.py` included. No test imports `splatpipe/depth/gpu.py`.
    - On `feat/depth-priors`: `git add splatpipe/depth/gpu.py tests/test_imports.py tests/test_cloud_imports.py docs/pipeline.md tests/test_docs.py`, then commit those files with the message "Depth: the GPU module and the pipeline doc's depth section".
    - _Requirements: 18.3, 18.13_

- [x] 5. Training reads the priors: the settings, the plain pixels, the depth weight and the scale-up
  - Files: `splatpipe/train/prior_views.py` and `tests/test_train_prior_views.py` (new). `config.toml`, `splatpipe/train/settings.py`, `splatpipe/train/schedules.py`, `splatpipe/train/scene.py`, `splatpipe/train/__init__.py`, `tests/conftest.py`, `tests/test_train_settings.py`, `tests/test_train_schedules.py`, `tests/test_train_step.py`, `docs/pipeline.md` and `tests/test_docs.py` (changed).
  - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.6, 10.7, 10.8, 10.9, 11.1, 11.2, 11.3, 11.4, 11.5, 11.6, 11.7, 11.8, 11.9, 14.1, 14.2, 14.3, 14.5, 14.6, 14.7, 14.9, 14.10, 17.2, 17.3, 17.9, 18.1, 18.2, 18.3, 18.5, 18.13_

  - [x] 5.1 Add the 15 `[train]` keys to `config.toml`, `splatpipe/train/settings.py` and the test config
    - In `config.toml`, put the 15 Shared_Settings of Req 14.1 after `gpu_memory_slack` and before `min_vcpus`, with the comment above each key that the design's "The `[train]` additions" gives it. `plain_gradient`'s comment says it's untested on real footage (O2).
    - `TrainSettings` gains the 15 keys in the same place, with the kinds of the design's "Changes to the Train_Step's other modules", from the existing kind table. That makes 50 Shared_Settings and 3 Preset_Settings.
    - In `tests/conftest.py`, `TRAIN` gains the 15 keys at their real values, so a test that runs the Train_Step without a Config of its own gets every setting.
    - _Requirements: 14.1, 14.2, 14.10_

  - [x] 5.2 Write `splatpipe/train/prior_views.py` and the Depth_Schedule
    - `brightness()`, `brightness_gradient()`, `plain_pixels()`, `depth_weight()`, `PriorTaps`, `upscale_taps()`, `prior_taps()`, `upscale()`, `prior_pixel_mask()`, `ViewPrior` and `view_prior()`, as in the design's "Prior views on the CPU" section. The module imports only the standard library, numpy, cv2 and `splatpipe`.
    - `brightness` is `cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY) / 255` in float64. `brightness_gradient` pads with `mode="edge"`, takes the central differences and gives float32. `plain_pixels` is `gradient < plain_gradient`.
    - `depth_weight(rate, gradient, settings, xp=np)` uses only `xp.exp`, `xp.where` and operators, so the Trainer calls the same function with `xp=torch` (11.7).
    - `upscale_taps` works out the taps in integers, with q = (2X + 1) · S − T over 2T and the clamps at both ends. `upscale` and `prior_pixel_mask` use only indexing and arithmetic, so they take numpy arrays and torch tensors alike. `prior_pixel_mask` keeps a pixel only when it isn't a Masked_Pixel and every stored pixel its bilinear sample draws on with a weight above 0 has a prior, and the nearest stored pixel has one too.
    - In `splatpipe/train/schedules.py`, `depth_schedule(start, end, planned)` gives `planned + 1` values, `start` and `end` set outright at the two ends and `start × (end / start) ** (i / planned)` between, with no running minimum.
    - _Requirements: 10.6, 10.7, 10.9, 11.1, 11.2, 11.3, 11.4, 11.5, 11.6, 11.7_

  - [x] 5.3 Read the Prior_Set in the scene and the step
    - In `splatpipe/train/scene.py`, `TrainingView` gains `prior: ViewPrior | None = None` and `Scene` gains `taps: PriorTaps | None = None`. `build_scene(model, pixels, width, height, settings, priors=None)` calls `view_prior` for each image named in the Prior_Set and `prior_taps(prior_width, prior_height, width, height)` once. An image with no Prior gets `None`.
    - In `splatpipe/train/__init__.py`, step 6 reads the Prior_Set as the design's "Changes to the Train_Step's flow" gives it: after `read_views` and inside the `load` timing, `read_prior_set(step.job.folder / "work" / PRIOR_FOLDER, model)`, with a `BrokenPriors` turned into the Req 10.3 warning and no Prior_Set. With a Prior_Set it adds `prior_images`, then calls `build_scene` with it.
    - With no `work/depth/depth.json`, the flow is the splat-training spec's exactly, with no new count or warning (A14).
    - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.8, 14.3, 14.5, 14.6, 14.7_

  - [x] 5.4 Write unit tests for the new settings in `tests/test_train_settings.py`
    - The real config, read by path, has `[train]` with exactly the 50 Shared_Settings and the 3 Preset_Settings, the 15 new ones at Req 14.1's values, and `step_settings("train", preset)` loads exactly 53 keys for each Preset.
    - Parametrised over the 15 new keys: a value outside its kind, and a bool where a number belongs, gives the splat-training message naming the key, `[train]` and what it must be. The edges pass: each weight, `depth_gradient_falloff`, `depth_weight_floor`, `normal_start_step` and `backproject_max` at 0, and both fractions at 0 and 1.
    - _Requirements: 14.1, 14.2, 18.1_

  - [x] 5.5 Write unit tests for the prior views and the Depth_Schedule
    - In `tests/test_train_prior_views.py`: `brightness` of known colours against `cv2.cvtColor`. `brightness_gradient` of a ramp, a step and a 1 × 1 image, with the border repeated. `plain_pixels` at the cutoff itself is false and just below it true. `upscale_taps` from 2 to 4, 3 to 7 and 4 to 4 against an oracle with `Fraction`s, with the clamps at both ends. `upscale` of a constant gives the constant and of a ramp the ramp's values at the centres. `prior_pixel_mask` at a mask's edge leaves out each pixel whose sample uses a false pixel with a weight above 0, keeps one whose weight there is 0, and leaves out every Masked_Pixel.
    - In `tests/test_train_schedules.py`: `depth_schedule` gives 1.0 at Iteration 0, 0.01 at the last and 0.1 halfway, for each real Preset's `steps`.
    - _Requirements: 10.6, 10.7, 10.9, 11.1, 11.2, 11.3, 11.4, 11.6, 18.1_

  - [x] 5.6 Write the property test for Property 5 in `tests/test_train_prior_views.py`, with Hypothesis
    - **Property 5: The Depth_Weight never falls below the floor on a Plain_Pixel, and is the plain product elsewhere**
    - **Validates: Requirements 11.8, 11.9, 11.5**
    - Draw as the design's Property 5 strategy says: Planned_Iterations from 1 to 40,000 with an Iteration up to it, the start and end from 10⁻⁶ to 10³, the floor from 0 to 10, the falloff from 0 to 10³, `plain_gradient` from 10⁻⁶ to 1, and 1 to 50 gradients from 0 to 10 with 0, `plain_gradient` and its float neighbours planted, as float64 and as float32 arrays.
    - With r the Iteration's `depth_schedule` value, check that every Plain_Pixel's weight is at least the floor, compared as the weight's dtype holds it, and equals max(r × exp(−falloff × g), floor), and that every other pixel's weight equals r × exp(−falloff × g) with `np.isclose(rtol=1e-12)`. So the depth weight never goes below 0.1 on a plain pixel with the real floor.
    - _Requirements: 11.5, 11.8, 11.9, 18.2_

  - [x] 5.7 Write unit tests for training with and without priors in `tests/test_train_step.py`
    - With `FakeGpu`, a Prior_Set written for the plane scene with the Prior_Writer gives the Trainer a Scene whose Prior_Views hold their Priors and Brightness_Gradients and whose taps match the Prior_Size, and the report the count `prior_images`.
    - A broken `depth.json`, and a Prior_File for an earlier camera solve with a changed pose, each give the Req 10.3 warning naming the file and what failed, and a Scene with no priors.
    - The regression check: a run with no `work/depth/` gives exactly the splat-training counts, timings and warnings, no new count or warning and the 53 settings, and the Scene `FakeGpu` gets has `prior` `None` on every view and `taps` `None`. The splat-training tests in this file pass with no change to what they expect, except the settings' count.
    - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.8, 14.3, 14.5, 18.5_

  - [x] 5.8 Change the train and depth sections of `docs/pipeline.md`, with their doc tests
    - In `## Step: train`: What it reads says the step reads `work/depth/` when `depth.json` is there, what it checks, and that one failed check gives one warning and training without priors. Config section names the 15 new keys with their values and meanings. Step report fields names `counts.prior_images` and the Req 10.3 warning.
    - In `## Step: depth`'s What it writes: how the train step scales a Prior up, bilinear for `inverse_depth` and `normal` with each normal set to unit length and nearest-neighbour for `mask`, and which training pixels count as Prior_Pixels.
    - In `tests/test_docs.py`, a test checks that `## Step: train` names every new key of the real `[train]`, read by path, `work/depth/depth.json` and `counts.prior_images`, and that the depth section's What it writes names the bilinear and nearest-neighbour scale-up.
    - _Requirements: 17.2, 17.3, 17.9_

  - [x] 5.9 Run the tests and commit
    - Run `uv run pytest`. Every test passes, `tests/test_guard.py` included. The import checks find `prior_views.py` by glob (14.9).
    - On `feat/depth-priors`: `git add config.toml splatpipe/train/settings.py splatpipe/train/prior_views.py splatpipe/train/schedules.py splatpipe/train/scene.py splatpipe/train/__init__.py tests/conftest.py tests/test_train_settings.py tests/test_train_prior_views.py tests/test_train_schedules.py tests/test_train_step.py docs/pipeline.md tests/test_docs.py`, then commit those files with the message "Depth: training reads the priors".
    - _Requirements: 14.9, 18.3, 18.13_

- [x] 6. Back-projected starting splats
  - Files: `splatpipe/train/backproject.py` and `tests/test_train_backproject.py` (new). `splatpipe/train/__init__.py`, `tests/test_train_step.py`, `docs/pipeline.md` and `tests/test_docs.py` (changed).
  - _Requirements: 12.1, 12.2, 12.3, 12.4, 12.5, 12.6, 12.7, 12.8, 12.9, 12.10, 14.4, 14.5, 14.6, 17.3, 18.1, 18.2, 18.3, 18.5, 18.13_

  - [x] 6.1 Write `splatpipe/train/backproject.py`
    - Load `skill::gsplat-training` with `load_resource` before writing code.
    - `back_project_budget()`, `landed_pixels()`, `candidate_pixels()`, `draw_candidates()`, `world_points()`, `StartPoints` and `start_points()`, as in the design's "Back-projected starting splats" section. The module imports only the standard library, numpy and `splatpipe`.
    - `back_project_budget` is min(⌊`backproject_share` × `max_splats`⌋, `backproject_max`) with `math.floor`. `landed_pixels` marks the one training pixel (⌊u⌋, ⌊v⌋) each Observed_Point in front of the camera lands on, through the view's scaled K and world-to-camera matrix (O5). `candidate_pixels` is the Prior_Pixels that are Plain_Pixels and not landed on.
    - `draw_candidates` draws B = min(budget, C) numbers with its own `np.random.default_rng(seed).choice(C, B, replace=False)`, sorted. `world_points` back-projects each pixel centre at z = 1 / the scaled-up `inverse_depth`, all in float64.
    - `start_points` with no Prior_Set is exactly `choose_points(model.points, model.colours, max_splats, seed)`. With one, it numbers the Candidate_Pixels in view order and then row-major order, draws, maps the numbers back through the cumulative counts, and gives the Sparse_Points of `choose_points(..., max_splats − B, seed)` first, then the Back_Projected_Splats with their pixels' 8-bit colours. Each view's candidate mask is worked out once.
    - _Requirements: 12.1, 12.2, 12.3, 12.4, 12.6, 12.8, 12.9_

  - [x] 6.2 Start the Back_Projected_Splats in `splatpipe/train/__init__.py`
    - `_start_splats` calls `start_points(model, scene, cfg)` in place of `choose_points`, inside the `load` timing. With a Prior_Set it adds `backprojected_splats`.
    - When `start.sparse` is below the number of Sparse_Points, the warning is Req 3.11's with no Prior_Set and the design's Req 12.7 text with one.
    - `nearest_distances` and `start_splats` run on every Start_Splat together, so the Back_Projected_Splats take the same size rule, opacity, rotation and zero `f_rest`, and `start_splats` counts them all.
    - _Requirements: 12.4, 12.5, 12.6, 12.7, 12.8, 14.4, 14.5, 14.6_

  - [x] 6.3 Write unit tests for the back-projection in `tests/test_train_backproject.py`
    - `back_project_budget` gives 333,333, 500,000 and 500,000 for the real Presets, and 0 with `backproject_max` 0.
    - `landed_pixels` marks a point's pixel and ignores a point behind the camera and one outside the image. `candidate_pixels` leaves out textured, non-Prior and landed pixels.
    - `draw_candidates` gives the same draw for the same seed, distinct numbers, and every candidate when there are fewer than the budget.
    - `start_points` with no Prior_Set equals `choose_points`. With one, it gives the drawn pixels' colours, the Sparse_Points first, and at most `max_splats` − B Sparse_Points.
    - _Requirements: 12.1, 12.2, 12.4, 12.6, 12.8, 18.1_

  - [x] 6.4 Write the property test for Property 6 in `tests/test_train_backproject.py`, with Hypothesis
    - **Property 6: Projecting a Back_Projected_Splat gives back its pixel and its depth**
    - **Validates: Requirements 12.10, 12.3**
    - Draw as the design's Property 6 strategy says: width and height from 1 to 2,000, fx and fy from 200 to 2,000, cx and cy inside the image, quaternions with a length of at least 0.1, translations from −4 to 4, any pixel, and a float32 `inverse_depth` from 0.125 to 2.
    - The test projects the point `world_points` gives with its own float64 code, R x + t and K's formula. Check that it gives the pixel centre (x + 0.5, y + 0.5) and a camera-space z of 1 / `inverse_depth`, each within a relative error of 10⁻⁹.
    - _Requirements: 12.3, 12.10, 18.2_

  - [x] 6.5 Write unit tests for the step's Start_Splats in `tests/test_train_step.py`
    - With `FakeGpu` and a Prior_Set for the plane scene, whose Frames are one colour and so all Plain_Pixels: the Start_Splats hold the Back_Projected_Splats after the Sparse_Points, `counts.backprojected_splats` and `counts.start_splats` count them, and the sizes come from one `nearest_distances` call over every Start_Splat.
    - More Sparse_Points than `max_splats` − B leaves gives the Req 12.7 warning with the points, the budget, B and the points used.
    - The regression check: with no `work/depth/`, the Start_Splats `FakeGpu` gets equal the splat-training spec's, from `choose_points`, with no Back_Projected_Splat and no new count.
    - _Requirements: 12.5, 12.7, 12.8, 14.4, 14.5, 18.5_

  - [x] 6.6 Change the train section of `docs/pipeline.md`, with its doc test
    - `## Step: train` says how the step starts the Back_Projected_Splats on plain Prior_Pixels where no sparse point lands, their budget from `backproject_share` and `backproject_max`, and that the Sparse_Points fill the rest. Step report fields names `counts.backprojected_splats`, that `counts.start_splats` counts them too, and the Req 12.7 warning.
    - In `tests/test_docs.py`, the train test also looks for `counts.backprojected_splats`.
    - _Requirements: 17.3_

  - [x] 6.7 Run the tests and commit
    - Run `uv run pytest`. Every test passes, `tests/test_guard.py` included.
    - On `feat/depth-priors`: `git add splatpipe/train/backproject.py splatpipe/train/__init__.py tests/test_train_backproject.py tests/test_train_step.py docs/pipeline.md tests/test_docs.py`, then commit those files with the message "Depth: back-projected starting splats".
    - _Requirements: 18.3, 18.13_

- [x] 7. The prior loss terms
  - Files: `splatpipe/train/schedules.py`, `splatpipe/train/gpu.py`, `splatpipe/train/gpu_api.py`, `tests/test_train_schedules.py`, `tests/test_train_gpu_api.py`, `tests/test_imports.py`, `docs/pipeline.md` and `tests/test_docs.py`, all changed.
  - _Requirements: 10.5, 10.6, 11.6, 11.7, 13.1, 13.2, 13.3, 13.4, 13.5, 13.6, 13.7, 13.8, 13.9, 13.10, 13.11, 13.12, 13.13, 13.14, 13.15, 13.16, 13.17, 13.18, 14.9, 17.3, 18.1, 18.2, 18.3, 18.13_

  - [x] 7.1 Add the term tables and `active_terms` to `splatpipe/train/schedules.py`
    - `SPLAT_TERMS`, `PRIOR_TERMS`, `TERM_WEIGHTS`, `depth_normal_start()`, `normal_on()` and `active_terms()`, as in the design's "Changes to the Schedules". The module stays pure and imports no torch.
    - `depth_normal_start` is `round(fraction × planned)`, the Erank_Start's rounding. `normal_on` is `iteration >= start and (iteration - start) % every == 0`.
    - `active_terms(i, settings, has_priors, view_has_prior)` gives the names in Loss_Table order. The rule for the five splat-training terms moves into it unchanged from `gpu.py`'s `_loss`. Only with `has_priors` come `depth` and `normal` on a Prior_View, `flatten` on every Iteration, and `depth_normal` and `normal_consistency` on a Prior_View from the Depth_Normal_Start, each only when its weight isn't 0.
    - _Requirements: 13.6, 13.9, 13.10, 13.11, 13.12, 13.13, 13.14, 13.15_

  - [x] 7.2 Add the Prior_Terms to the Trainer in `splatpipe/train/gpu.py`
    - Load `skill::gsplat-training` with `load_resource` before writing code, and follow gsplat 1.5.3's API as the skill and the design's "Changes to the Trainer" give it. The licence note stays, and its "Changes:" line adds "the depth-prior losses and the normal render". The file names none of `rasterization_inria_wrapper`, `rasterization_2dgs_inria_wrapper` and `export_splats`, and the Prior_Terms are written fresh from Requirement 13's formulas.
    - `_ViewPrior`, `_PriorInputs` and `_Inputs`, and the changes to `_View` and `_Training`. `_set_up` uploads each Prior_View's Prior at the Prior_Size with its gradient, and with a Prior_Set the taps, the rays and `depth_schedule(depth_weight_start, depth_weight_end, steps)`. The table is `_TERMS + _PRIOR_TERMS` with a Prior_Set and `_TERMS`, exactly the splat-training five, without one.
    - Each Iteration: `on = active_terms(...)`; `_render` in `"RGB+ED"` on a Prior_View and `"RGB"` otherwise; `_prior_inputs` with `upscale`, `prior_pixel_mask` and `depth_weight(..., torch)` from `prior_views`, the Normal_Render only when `normal` or `normal_consistency` is on, and the Depth_Normals only when `depth_normal` or `normal_consistency` is on; then `_loss(run.table, inputs, on)`. The order after the loss stays the splat-training spec's.
    - `_splat_normals`, `_normal_render` with `sh_degree=None`, `_depth_normals` with the sign turned against the pixel's ray, and the five terms of the design's table, each a sum over its pixels divided by `max(count, 1)`, with every division and unit-length step made safe with `torch.where` as the design's decisions table says: in a unit-length step a squared length of 0 is replaced by 1 before its square root, so no small number is hard-coded.
    - `NonFiniteLoss` lists every term of `run.table` and the total. In `splatpipe/train/gpu_api.py`, `non_finite_message`'s docstring says it lists every Loss_Term of the run's Loss_Table.
    - _Requirements: 10.5, 10.6, 11.6, 11.7, 13.1, 13.2, 13.3, 13.4, 13.5, 13.6, 13.7, 13.8, 13.9, 13.10, 13.12, 13.13, 13.14, 13.16, 13.17_

  - [x] 7.3 Write unit tests for the term schedules in `tests/test_train_schedules.py`
    - `depth_normal_start` gives 1,633, 3,500 and 7,000 for the real Presets. `normal_on` at 2,999, 3,000, 3,003 and 3,004 with the real `normal_start_step` and `normal_every`.
    - `active_terms` with a Prior_Set: a view with no Prior gets the five splat-training terms and `flatten`; a Prior_View gets `depth` from Iteration 0, `normal` on its Iterations, and `depth_normal` and `normal_consistency` from the Depth_Normal_Start; each weight of 0 turns its term off.
    - _Requirements: 13.6, 13.9, 13.10, 13.11, 13.12, 13.14, 13.15, 18.1_

  - [x] 7.4 Write the property test for Property 7 in `tests/test_train_schedules.py`, with Hypothesis
    - **Property 7: With no Prior_Set, an Iteration works out exactly the splat-training terms**
    - **Validates: Requirements 13.18, 13.13, 13.14**
    - Draw as the design's Property 7 strategy says: the five splat-training weights each 0 a third of the time and otherwise from 10⁻⁶ to 10, `erank_start_fraction` from 0 to 1 with 0, 7/30 and 1 drawn often, Planned_Iterations from 1 to 40,000, the Iteration below them with the Erank_Start and its neighbours drawn often, either value of `view_has_prior`, and every Prior_Term setting across its whole valid range. The settings go through `read_settings`.
    - Check that `active_terms(i, settings, False, view_has_prior)` equals the oracle: each of `l1`, `ssim`, `opacity_reg` and `scale_reg` whose weight isn't 0, then `erank` when its weight isn't 0 and i is at or after round(`erank_start_fraction` × Planned_Iterations), in that order, and never a Prior_Term. So with no priors, training uses exactly the splat-training losses.
    - _Requirements: 13.13, 13.14, 13.18, 18.2_

  - [x] 7.5 Write the Trainer's source checks and the message test
    - In `tests/test_imports.py`, read `splatpipe/train/gpu.py` with `ast` and never import it: `_TERMS` holds exactly the splat-training spec's five rows, `l1`, `ssim`, `opacity_reg`, `scale_reg` and `erank` in that order, and the names and weight settings of `_TERMS` and `_PRIOR_TERMS` equal `SPLAT_TERMS`, `PRIOR_TERMS` and `TERM_WEIGHTS`. Its text calls `active_terms` and `depth_weight` and still names none of the banned functions. `gpu.py` still imports no GPU package besides torch, gsplat and fused_ssim (14.9).
    - In `tests/test_train_gpu_api.py`, `non_finite_message` with the ten terms names each of them, the Iteration and the total.
    - _Requirements: 11.7, 13.1, 13.13, 13.16, 13.17, 14.9, 18.1_

  - [x] 7.6 Change the train section of `docs/pipeline.md`, with its doc test
    - `## Step: train` names the five Prior_Terms, `depth`, `normal`, `flatten`, `depth_normal` and `normal_consistency`, with their weights and when each is on, and says that with no priors the loss holds only the five splat-training terms and the step renders no depth or normals. Its Step report fields say that the loss-not-finite message lists every Loss_Term of the run's loss table.
    - In `tests/test_docs.py`, the train test also looks for the five Prior_Term names.
    - _Requirements: 17.3_

  - [x] 7.7 Run the tests and commit
    - Run `uv run pytest`. Every test passes, `tests/test_guard.py` included. The Trainer's new code runs only on a GPU, in the Smoke_Test's depth check of task 8, which the user runs.
    - On `feat/depth-priors`: `git add splatpipe/train/schedules.py splatpipe/train/gpu.py splatpipe/train/gpu_api.py tests/test_train_schedules.py tests/test_train_gpu_api.py tests/test_imports.py docs/pipeline.md tests/test_docs.py`, then commit those files with the message "Depth: the prior loss terms".
    - _Requirements: 18.3, 18.13_

- [x] 8. The smoke test's depth check
  - Files: `splatpipe/depth/smoke.py` and `tests/test_depth_smoke.py` (new). `splatpipe/train/smoke.py`, `config.toml`, `cloud/config.py`, `cloud/smoke_runner.py`, `tests/fake_aws.py`, `tests/test_cloud_config.py`, `tests/test_cloud_smoke_runner.py`, `docs/pipeline.md` and `tests/test_docs.py` (changed).
  - _Requirements: 16.1, 16.2, 16.3, 16.4, 16.5, 16.6, 16.7, 16.8, 16.9, 16.10, 16.11, 16.12, 17.5, 18.3, 18.11, 18.13_

  - [x] 8.1 Write `splatpipe/depth/smoke.py`
    - Load `skill::moge-depth`, `skill::gsplat-training` and `skill::splat-files` with `load_resource` before writing code.
    - `DepthSmokeSizes`, `check_moge_output()`, `prior_smoke_scene()` and `run_smoke_depth()`, as in the design's "`splatpipe/depth/smoke.py`" section. The module imports only the standard library, numpy and `splatpipe`, and no torch.
    - In `splatpipe/train/smoke.py`, `smoke_settings(config, sizes, **overrides)` takes extra `[train]` values to replace, for the depth check's `normal_start_step`. The training check's own run is unchanged.
    - `run_smoke_depth` runs MoGe's part with the step's own `read_settings`, `check_machine`, `load_moge` and `run_moge` on a made-up `image_pixels` × `image_pixels` image from a generator seeded with `[train]`'s `seed`, with `fov_x = field_of_view(p, p)`, and `release` in a `finally`. `check_moge_output` checks the shapes, dtypes and finite values and gives the Geometry_Pixels.
    - `prior_smoke_scene` takes the training check's `smoke_scene`, paints each image's rows from p // 2 down one grey, gives every image a made-up Prior (inverse depth 1, normal (0, 0, −1), mask true), and writes and reads it back with the Prior_Writer and `read_prior_set`.
    - It then trains with the Prior_Set through `build_scene`, `start_points`, `nearest_distances`, `start_splats` and `gpu.train`, bakes the Mean_Exposure, writes every splat with `write_ply` and runs `check_ply`. It passes only when the Trainer ran every Iteration and at least one Back_Projected_Splat started, and gives back the design's log line. Every failure raises with a message saying what failed.
    - _Requirements: 16.2, 16.3, 16.4, 16.5, 16.8, 16.9_

  - [x] 8.2 Add `depth_normal_start_step` to the configs
    - In `config.toml`, append `depth_normal_start_step = 8` to `[cloud.smoke]` after `train_sh_degree_interval`, with the design's comment.
    - In `cloud/config.py`, `SmokeChecks` gains `depth_normal_start_step: int`, which `_read_smoke_checks` reads with a new `_whole_or_zero` helper: a whole number of 0 or more, never a bool or a float, refused with the loader's Readable_Error naming `depth_normal_start_step for [cloud.smoke]`.
    - In `tests/fake_aws.py`, `_base_cloud_mapping`'s smoke table gains the key at 8.
    - _Requirements: 16.6_

  - [x] 8.3 Add the `depth` check to `cloud/smoke_runner.py`
    - Load `skill::aws-jobs` with `load_resource` before writing code.
    - `_check_depth(context)` builds `DepthSmokeSizes` from `context.cloud_config.smoke_checks` and calls `run_smoke_depth(context.config, sizes, context.scratch, load_depth_gpu(), load_gpu())`. The module imports `DepthSmokeSizes` and `run_smoke_depth` from `splatpipe.depth.smoke` and `load_depth_gpu` from `splatpipe.depth.gpu_api` at the top and calls both loaders inside the function, looked up in its own module when it runs, so it still imports on a laptop and a test can patch each.
    - `SMOKE_CHECKS` gains `SmokeCheck("depth", _check_depth)` after `moge-weights` and before `training`, and the module docstring's list of checks names it. The check has no time limit of its own. On today's Machine_Image it fails with the Req 3.2 message until the user runs the next Image_Build (A19).
    - _Requirements: 16.1, 16.10, 16.11_

  - [x] 8.4 Change the smoke test part of `docs/pipeline.md`, with its doc test
    - In "The smoke test", add the design's bullet for the depth check after the MoGe-2 weights bullet, with `depth_normal_start_step` in `[cloud.smoke]`. The last sentence names the training check and the depth check as the checks later specs added.
    - In `tests/test_docs.py`, a test checks that the part names the depth check, `depth_normal_start_step` and `[cloud.smoke]`.
    - _Requirements: 16.12, 17.5_

  - [x] 8.5 Write unit tests for the depth smoke flow in `tests/test_depth_smoke.py`
    - `check_moge_output` passes a good output and refuses a wrong shape, a wrong dtype, a NaN normal and a depth of 0 where the mask is true, each with its message.
    - `prior_smoke_scene` passes the Train_Step's own prior checks on the CPU, and its images have plain lower halves.
    - With the real Config and `[cloud.smoke]`, `active_terms` turns every Prior_Term on at least once in the 30 Iterations, and `start_points` starts 133 Back_Projected_Splats and all 200 Sparse_Points.
    - `run_smoke_depth` with `FakeDepthGpu` and `FakeGpu` writes and checks `train.ply` in the scratch folder and gives the log line with its numbers. It fails with a readable message when MoGe's output is wrong, the stand-in runs fewer Iterations or no Back_Projected_Splat starts.
    - _Requirements: 16.3, 16.4, 16.7, 16.8, 16.9, 18.11_

  - [x] 8.6 Write unit tests for the smoke key and the smoke runner's check
    - In `tests/test_cloud_config.py`, the fixed valid mapping gains the key, its corruption cases −1, 1.5, `true` and `"8"` each name `depth_normal_start_step for [cloud.smoke]`, and the real-config test checks 8. The Property 2 strategy, `valid_cloud_mappings` and `corruptible_targets`, gains it as a whole number of 0 or more.
    - In `tests/test_cloud_smoke_runner.py`, the order test has `depth` between `moge-weights` and `training`. The laptop test of the GPU checks includes `depth`, with `splatpipe.depth.gpu_api.importlib` replaced so its `import_module` raises `ModuleNotFoundError` for torch: the check fails on its own with the Req 3.2 message and the next check still runs. A new test runs `_check_depth` with `cloud.smoke_runner.load_depth_gpu` and `load_gpu` patched with stand-ins, and it passes with its log line.
    - _Requirements: 16.1, 16.6, 16.9, 16.10, 16.12, 18.11_

  - [x] 8.7 Run the tests and commit
    - Run `uv run pytest`. Every test passes, `tests/test_guard.py` included.
    - On `feat/depth-priors`: `git add splatpipe/depth/smoke.py splatpipe/train/smoke.py config.toml cloud/config.py cloud/smoke_runner.py tests/fake_aws.py tests/test_cloud_config.py tests/test_cloud_smoke_runner.py tests/test_depth_smoke.py docs/pipeline.md tests/test_docs.py`, then commit those files with the message "Depth: the smoke test's depth check".
    - This task runs no Smoke_Test. The depth check passes only on an image built with the MoGe_Packages, so it first runs after task 9's install is built, in a paid run the user starts (16.11).
    - _Requirements: 16.11, 18.3, 18.13_

- [x] 9. MoGe on the machine image, its licence rows and rules.md's certifi line
  - Files: `cloud/image.py`, `tests/test_cloud_image.py`, `docs/third-party-licences.md` and `.kiro/steering/rules.md`, all changed.
  - _Requirements: 15.1, 15.2, 15.3, 15.4, 15.6, 15.7, 15.8, 15.9, 15.10, 15.12, 18.3, 18.13_

  - [x] 9.1 Check the pins, then add install step 7 to `cloud/image.py`
    - Load `skill::moge-depth` and `skill::aws-jobs` with `load_resource` before writing code.
    - First, before the install changes, read MoGe's `uv.lock` at commit `74fbce054ebed49800de42d0ad0e83495065719a` and check that each of the 14 PyPI pins in the design's O3 table is the lock's version, and that huggingface-hub's tree on Linux x86-64 and Python 3.11 holds no package outside D4's list but hf-xet, which D4 leaves off. This is a free, read-only fetch.
    - If a pin differs, or the tree holds another package, stop and ask the user before changing anything (O3).
    - Then `install_script` gains step 7 after the GPU packages, exactly as the design's "The install's new step" gives it: MoGe's code and utils3d-moge from their pinned git commits with `uv pip install --python "$VENV_PY" --no-deps`, then the 14 pinned PyPI packages, huggingface-hub 1.26.0 and its 13 packages at the O3 pins, with no hf-xet, also with `uv pip install --python "$VENV_PY" --no-deps` (15.12). Step 7's comment gives the reason hf-xet stays off, as the design's comment does: huggingface-hub imports it only inside functions and downloads over plain HTTP without it, and the depth step downloads nothing. Then `HF_HUB_OFFLINE=1 "$VENV_PY" -c "from moge.model.v2 import MoGeModel"`, which fails the build under the existing trap, so the build proves that nothing on the MoGe-2 path is missing. The weights become step 8, Node step 9 and the clean-up step 10.
    - The module's and `install_script`'s docstrings name the new step. Measure the script with the real Config and a sample bucket and prefix: it stays under EC2's 16,384-byte User_Data limit, about 9.0 KB (the design's estimate is 9,027 bytes), which `install_script` already enforces. Nothing goes in `pyproject.toml` or `uv.lock`.
    - _Requirements: 15.1, 15.2, 15.3, 15.4, 15.12_

  - [x] 9.2 Add and change the licence rows in `docs/third-party-licences.md`, and change rules.md's certifi line
    - New rows, each "Cloud image only" and "Not yet" installed, as the design's "The Licence_List" table gives them: huggingface-hub, click, httpx, httpcore, h11, anyio, idna, pyyaml and utils3d-moge, each with its licence, version or commit, how we use it and what it comes in through. The huggingface-hub row says it's installed with `--no-deps`, so hf-xet stays off the image. There is no hf-xet row (15.7, 15.12).
    - Changed rows: filelock, fsspec, typing-extensions and packaging with their pinned versions and "Comes in through ... huggingface-hub" as the design gives each, fsspec's with the note on its `LICENSE` at the `2026.7.0` tag; certifi 2026.7.22, "Used unmodified. Comes in only through httpx and httpcore, on the machine image, which Samarth OKs. Cloud image only.", which names no requests, because requests isn't installed anywhere; tqdm 4.70.0, "Used unmodified. Comes in through huggingface-hub. Cloud image only."; and the MoGe code row, installed with `--no-deps` at its commit, with only utils3d-moge of its git dependencies, and pipeline and FlexGEMM off.
    - In `.kiro/steering/rules.md`, the named-exceptions line for certifi and tqdm becomes exactly D5's line: "- certifi (MPL-2.0) and tqdm (MPL-2.0 and MIT): used unmodified. On the cloud machine image, certifi comes in through httpx and httpcore, and tqdm through huggingface-hub. Samarth OKs certifi's." So it says that certifi comes in only through httpx and httpcore, on the machine image, used unmodified, that tqdm still comes in through huggingface-hub, and records Samarth's OK. It names no requests, which isn't installed anywhere: our `uv.lock` has neither requests nor certifi, and the image doesn't add requests (D5). No other line of rules.md changes.
    - _Requirements: 15.7, 15.8, 15.9, 15.12_

  - [x] 9.3 Write the image tests in `tests/test_cloud_image.py`
    - Step 7 installs each MoGe_Package with its exact pin and `--no-deps`, after the GPU packages' import check and before the weights, and runs `HF_HUB_OFFLINE=1 "$VENV_PY" -c "from moge.model.v2 import MoGeModel"`. No line of the script outside its comments names `moderngl`, `trimesh`, `gradio`, `flex-gemm`, `hf-xet`, `hf_xet` or `pipeline`, case ignored: step 7's comment names the packages that stay off, hf-xet among them, and the script's opening comment says "Pipeline". The script stays within 16 KB, at about 9.0 KB by the design's estimate.
    - `test_install_script_fetches_the_moge_2_weights_and_checks_their_sha256` keeps its curl and sha256 checks, and its assertion that the script names no huggingface package becomes the design's: the weights are still fetched with curl from the pinned revision, and `huggingface-hub` appears only as `huggingface-hub==1.26.0` in step 7.
    - Each row of 9.2's new rows exists once, and no licence row is hf-xet's. The certifi row names httpx and httpcore, `.kiro/steering/rules.md` holds D5's line for certifi and tqdm exactly, and neither the certifi row nor that line names requests. The MoGe code row says `--no-deps` and its commit, and the filelock, fsspec, packaging and typing-extensions rows name huggingface-hub.
    - _Requirements: 15.1, 15.2, 15.3, 15.4, 15.7, 15.8, 15.9, 15.10, 15.12_

  - [x] 9.4 Run the tests and commit
    - Run `uv run pytest`. Every test passes, `tests/test_guard.py` included.
    - On `feat/depth-priors`: `git add cloud/image.py tests/test_cloud_image.py docs/third-party-licences.md .kiro/steering/rules.md`, then commit those files with the message "Depth: MoGe on the machine image, its licence rows and rules.md's certifi line".
    - This task runs no Image_Build. The new install reaches a Machine_Image only when the user runs the next Image_Build, a paid run (15.6).
    - _Requirements: 15.6, 18.3, 18.13_

- [x] 10. The Step_List entry, the docs wrap-up and the moge-depth skill
  - Files: `splatpipe/run.py`, `docs/pipeline.md`, `.kiro/skills/moge-depth/SKILL.md`, `tests/test_run.py`, `tests/test_frames_step.py`, `tests/test_imports.py`, `tests/test_depth_settings.py`, `tests/test_docs.py` and `tests/test_cloud_docs.py`, all changed.
  - _Requirements: 1.3, 1.10, 8.5, 15.5, 15.6, 15.9, 17.4, 17.6, 17.8, 17.9, 18.3, 18.10, 18.12, 18.13_

  - [x] 10.1 Change the Step_List in `splatpipe/run.py`
    - `STEP_LIST = (StepEntry("frames", "splatpipe.frames", True), StepEntry("depth", "splatpipe.depth", False), StepEntry("train", "splatpipe.train", True), StepEntry("clean", "splatpipe.clean", False))`.
    - The comment above it says the camera solve spec puts its entry between the Frames_Step and the Depth_Step, so the order becomes frames, cameras, depth, train, clean, and that the Fake_Step stays out.
    - _Requirements: 1.3_

  - [x] 10.2 Change the step list and Step_Needs parts of `docs/pipeline.md`
    - "The step list": the design's three new sentences, "The list holds the frames step, required, the depth step, optional, the train step, required, and then the clean step, optional" with the four entries, the camera solve sentence and the "Until the camera solve spec adds its entry" sentence, in place of the three they replace. It still holds `StepEntry("clean", "splatpipe.clean", False)`, which `tests/test_clean_repo.py` looks for.
    - "What each step asks the machine for": the Depth priors row stays "a GPU with 16 GB VRAM", and the paragraph after the table says "The depth step's need is `min_gpu_memory_gb` in `[depth]`, 16, which the train step's 24 GB already covers."
    - Read `## Step: depth`, `## Step: train` and "The smoke test" once more against the code of tasks 2 to 9, and fix any sentence that no longer says what the code does.
    - _Requirements: 17.4, 17.6_

  - [x] 10.3 Change the moge-depth skill's sentences
    - Load `skill::moge-depth` with `load_resource` before writing.
    - In `.kiro/skills/moge-depth/SKILL.md`, the sentences of the design's "The moge-depth skill" section, each with the design's new text: the pin table's line on MoGe's code, the line on how the step finds the weights, which now names `moge_weights` in `[depth]`, and the "On our image" column for utils3d_moge and huggingface_hub.
    - In the bullet on licence rows under "Our rules that touch this", the sentence "MoGe's code and weights have rows, and the code's row covers the three git dependencies." becomes the design's, so it matches what this spec installs: "MoGe's code and weights have rows, and so does utils3d-moge, which the image installs with `--no-deps` at the commit MoGe pins. MoGe's other git dependencies, pipeline and FlexGEMM, aren't installed, as the code's row says."
    - In the same bullet, the "huggingface-hub has none yet, and nor do click, hf-xet, ..." sentence becomes the design's: huggingface-hub 1.26.0 and the packages that come in through it have their rows, and hf-xet, which huggingface-hub declares, has none, because the image leaves it off. The "One way to keep the list to what runs: ... The depth-priors spec decides." sentences become the design's, which say what D4 decided: `moge` and `utils3d_moge` from their pinned commits with `--no-deps`, and huggingface-hub and its 13 packages at the versions in MoGe's `uv.lock`, each with `--no-deps` too, from the next image build, with moderngl, trimesh, gradio, pipeline and FlexGEMM off. Nothing else in the skill changes.
    - _Requirements: 17.8_

  - [x] 10.4 Update the tests that depend on the Step_List
    - In `tests/test_run.py`, `test_the_step_list_holds_the_frames_train_and_clean_steps_and_is_the_default_list` becomes `test_the_step_list_holds_the_frames_depth_train_and_clean_steps_and_is_the_default_list` and asserts the list of 10.1.
    - The real-Step_List Job test is `test_run_job_with_the_real_step_list_fails_at_train_once_the_frames_step_is_done` in `tests/test_frames_step.py`. It now expects frames `done`, depth `failed` with the Req 2.2 message naming `work/cameras/images`, train `failed` and clean `not_run`, and the Job `failed` at train. Any other test that the run shows relying on the three-entry list gets updated the same way.
    - In `tests/test_imports.py`, the Step_List check's docstring names the Depth_Step, whose `[depth]` it now loads from the real config for each Preset.
    - In `tests/test_depth_settings.py`, the Fallback_List test: with the real Config and the real Step_List, `step_needs_over` gives 24, 24 and 48 GB for the three Presets, and `select_candidates` keeps every type on each Preset's Fallback_List.
    - _Requirements: 1.10, 8.5, 18.10, 18.12_

  - [x] 10.5 Write the last doc tests in `tests/test_docs.py` and `tests/test_cloud_docs.py`
    - In `tests/test_docs.py`, the step list test looks for the three new sentences and checks that the three old ones are gone. A test checks that the moge-depth skill names `moge_weights` and `--no-deps` and no longer says "The image doesn't have MoGe's code yet", "huggingface-hub has none yet", "the code's row covers the three git dependencies" or that the depth-priors spec decides.
    - In `tests/test_cloud_docs.py`, a test checks the Step_Needs paragraph's sentence on `min_gpu_memory_gb` in `[depth]`, and `STEP_NEEDS_PLAN`'s Depth priors row keeps its 16 GB.
    - _Requirements: 17.4, 17.6, 17.8, 17.9_

  - [x] 10.6 Run the whole suite one last time and commit
    - Run `uv run pytest`. This is the spec's final full-suite run: every test passes, `tests/test_guard.py` and every splat-training test included, on the CPU with no network.
    - Check that `git log --oneline main..HEAD -- pyproject.toml uv.lock` lists none of this spec's commits and that `git diff main -- .kiro/steering/rules.md` changes only the named-exceptions line for certifi and tqdm, as task 9 left it (15.5, 15.9, D5).
    - On `feat/depth-priors`: `git add splatpipe/run.py docs/pipeline.md .kiro/skills/moge-depth/SKILL.md tests/test_run.py tests/test_frames_step.py tests/test_imports.py tests/test_depth_settings.py tests/test_docs.py tests/test_cloud_docs.py`, then commit those files with the message "Depth: the step list entry and the docs wrap-up".
    - This task runs nothing paid and the spec ends here. Building the Machine_Image and running the Smoke_Test are paid runs that the user starts, and they are not tasks of this spec (15.6).
    - _Requirements: 15.5, 15.6, 15.9, 18.3, 18.13_

## Notes

- Every unit test and property test sub-task is required, as `.kiro/steering/rules.md` and Req 18.3 ask. None is marked optional, so no `*` appears anywhere.
- The plan has 10 tasks, within rules.md's limit. The last sub-task of each task is its checkpoint: the whole suite, then the commit by name. So there are no separate checkpoint tasks, and 10.6 is the final full-suite run.
- Every acceptance criterion in `requirements.md` is named by at least one task. Each of the design's 7 correctness properties has its own property test sub-task, with Hypothesis: Property 1 in 1.5, Property 4 in 1.7, Properties 2 and 3 in 2.8 and 2.9, Property 5 in 5.6, Property 6 in 6.4 and Property 7 in 7.4.
- The CPU unit tests cover, with no GPU, the fit (2.7), the plain-pixel mask and the weight schedule (5.5), the back-projection (6.3) and the prior files (1.6), with the Model_Reader (1.4), the step's flow (3.6, 3.7) and the term schedules (7.3). The GPU paths, `splatpipe/depth/gpu.py` and the Trainer's Prior_Terms, are covered only by the Smoke_Test's depth check, so their laptop tests read their source (4.2, 7.5) and no laptop test imports either GPU module.
- With no Prior_Set, training stays exactly as the splat-training spec says: 5.7, 6.5, 7.4 and 7.5 check it, and every splat-training test stays in the suite.
- The test of a Job through the real Step_List lives in `tests/test_frames_step.py`, as the design's "Changes outside" table says, so 10.4 changes it there.
- The comment above `MOGE_WEIGHTS` changes in 2.1, with the test that keeps `moge_weights` equal to it, and not with the install step in task 9.
- 9.1 is the one place where the plan may stop for the user: if a pin differs from MoGe's `uv.lock`, or huggingface-hub's tree there holds a package outside D4's list other than hf-xet, the install doesn't change until the user answers. hf-xet stays off the image, with no pin and no licence row (D4).
- rules.md changes in one place only, its named-exceptions line for certifi and tqdm, in 9.2, which Samarth OKs (D5). 9.4 stages it by name, and 10.6 checks that no other line changed.
- No task runs anything paid. Once the spec is built, two paid runs are left for the user to start, and neither is a task: the Image_Build, `uv run python -m cloud.image`, which puts the MoGe_Packages on a new image, and then the Smoke_Test, `uv run python -m cloud.smoke`, on that image. For each, Kiro first tells the person the machine (`image_builder_type` or `smoke_instance_type` in `[cloud]`), the expected time and the rough cost, and only after their yes runs that one command with `APPROVED_SPEND=1`.
- Until the Image_Build runs, the image that runs Jobs has no MoGe code. On it the Depth_Step fails at its import check and the Train_Step trains without priors, so Jobs keep working but end `done with warnings`, and a Smoke_Test there fails the depth check. The new image gets a new id, and the Start_Command needs a passing Smoke_Test on it before any Job runs there (A19).
- The graph runs the tasks in order, one at a time, so each task's test run and commit see only finished work. Inside a task, sub-tasks that write different files share a wave, tests come after the code they test, and each task's commit gates the next task's first sub-tasks.

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "1.3"] },
    { "id": 1, "tasks": ["1.2", "1.4", "1.6"] },
    { "id": 2, "tasks": ["1.5", "1.7"] },
    { "id": 3, "tasks": ["1.8"] },
    { "id": 4, "tasks": ["2.1", "2.2", "2.5"] },
    { "id": 5, "tasks": ["2.3"] },
    { "id": 6, "tasks": ["2.4"] },
    { "id": 7, "tasks": ["2.6", "2.7"] },
    { "id": 8, "tasks": ["2.8"] },
    { "id": 9, "tasks": ["2.9"] },
    { "id": 10, "tasks": ["2.10"] },
    { "id": 11, "tasks": ["3.1", "3.2"] },
    { "id": 12, "tasks": ["3.3", "3.4"] },
    { "id": 13, "tasks": ["3.5", "3.6"] },
    { "id": 14, "tasks": ["3.7"] },
    { "id": 15, "tasks": ["3.8"] },
    { "id": 16, "tasks": ["4.1", "4.4"] },
    { "id": 17, "tasks": ["4.2", "4.3", "4.5"] },
    { "id": 18, "tasks": ["4.6"] },
    { "id": 19, "tasks": ["5.1"] },
    { "id": 20, "tasks": ["5.2", "5.4"] },
    { "id": 21, "tasks": ["5.3", "5.5"] },
    { "id": 22, "tasks": ["5.6", "5.7", "5.8"] },
    { "id": 23, "tasks": ["5.9"] },
    { "id": 24, "tasks": ["6.1"] },
    { "id": 25, "tasks": ["6.2", "6.3"] },
    { "id": 26, "tasks": ["6.4", "6.5", "6.6"] },
    { "id": 27, "tasks": ["6.7"] },
    { "id": 28, "tasks": ["7.1", "7.6"] },
    { "id": 29, "tasks": ["7.2", "7.3"] },
    { "id": 30, "tasks": ["7.4", "7.5"] },
    { "id": 31, "tasks": ["7.7"] },
    { "id": 32, "tasks": ["8.1", "8.2", "8.4"] },
    { "id": 33, "tasks": ["8.3", "8.5"] },
    { "id": 34, "tasks": ["8.6"] },
    { "id": 35, "tasks": ["8.7"] },
    { "id": 36, "tasks": ["9.1"] },
    { "id": 37, "tasks": ["9.2"] },
    { "id": 38, "tasks": ["9.3"] },
    { "id": 39, "tasks": ["9.4"] },
    { "id": 40, "tasks": ["10.1", "10.2", "10.3"] },
    { "id": 41, "tasks": ["10.4", "10.5"] },
    { "id": 42, "tasks": ["10.6"] }
  ]
}
```
