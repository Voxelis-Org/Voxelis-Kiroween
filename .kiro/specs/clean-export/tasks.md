# Implementation Plan: clean-export

## Overview

This plan builds step 08, `splatpipe/clean_export/`, in the design's build order of 9 tasks: the dependencies and licences, the settings, the PLY reader and writer, the Non_Finite_Filter and the Opacity_Filter, the Outlier_Filter, the cleaning and its limits, Splat_Transform, the step itself, and last the Pipeline_Doc section. Each task writes its code with that code's unit and property tests, runs the whole suite and commits.

The cleaning follows the decisions made at review: the Outlier_Filter's local ratio over 8 neighbours at 8 times (D1), the Opacity_Filter at 0.005 with faint surfaces kept (D2), no scale filter (D3), never more than 70% removed (D4), the three cleaning properties (D5), Splat_Transform 3.8.0 (D6), and every one of those numbers in `[clean_export]` with a comment (D7).

Requirement 8, criterion 4 binds the later spec that adds this step's Step_List entry, so no task here covers it.

Some test files grow over several tasks, so each piece comes with the first task that uses it:

- `tests/conftest.py`: task 2 adds the test config's `[clean_export]` and `set_clean_export`, task 7 adds `stand_in_transform`, and task 8 adds `splat_job`.
- `tests/test_clean_filters.py`: tasks 4 and 5 test the filters one by one, and task 6 adds `clean`, the limits and Properties 1 to 3.
- `tests/test_clean_setup.py`: task 1 tests the dependencies and licences, and task 9 adds the doc tests.

## Working rules for every task

- Work on `feat/clean-export`, never on `main`.
- Inside `splatpipe/`, only `splatpipe/clean_export/` imports numpy and scipy. No Skeleton module and no pipeline-core test changes, and `STEP_LIST` stays empty (A4). No GPU package goes in `pyproject.toml`.
- Every number and setting comes from `[clean_export]` in `config.toml`, except the Format_Constants, `PLY_TYPES`, `PLY_NAMES`, `POSITION`, `SCALES`, `ROTATION`, `REQUIRED`, `SH_BANDS` and π in Paint, and the two constants the design keeps in code on purpose, `PRESET_KEYS` and `NODE_NEEDED`.
- Write the code fresh. No code from Inria's gaussian-splatting repo or any other splat project.
- Tests run on the CPU with no network and only tiny synthetic splats that the tests build as bytes. No test writes into the real `data/` folder, and no splat file goes in git.
- Subprocess tests pass a timeout, set `PYTHONIOENCODING=utf-8` in the child's environment and read its output as UTF-8.
- Property tests use Hypothesis with at least 100 examples, the default, and `@settings(deadline=None)`. Each docstring says `Feature: clean-export, Property N: <title>`.
- Every test sub-task is required. None is optional.
- The suite must pass on a machine without Node. Only the real Splat_Transform test needs it, and it's skipped there with its reason.
- Nothing here costs money. No task touches AWS, sets `APPROVED_SPEND` or runs anything paid. Task 1 uses the network only to install packages with `uv add` and `npm`.
- The last sub-task of each task is its checkpoint. Run `uv run pytest` and fix what fails until every test passes, `tests/test_guard.py` included. Then stage the task's files by name with `git add <file> ...`, never with `git add -A`, `git add .` or a folder, and commit only those files with `git commit -m "<message>" -- <file> ...`. Leave `docs/devlog.md` out: the devlog hook adds and commits its own line after each task.
- If a hook blocks something it shouldn't, say so and fix the hook. Never work around it.

## Tasks

- [ ] 1. Dependencies and licences
  - Files: `package.json`, `package-lock.json` and `tests/test_clean_setup.py` (new), `pyproject.toml`, `uv.lock`, `docs/third-party-licences.md` and `README.md` (changed).
  - _Requirements: 6.10, 6.11, 6.12, 6.14, 8.5, 9.6, 9.7_

  - [ ] 1.1 Add numpy and scipy
    - Run `uv add numpy scipy`, so both go in `[project].dependencies`, not the dev group.
    - Check that `uv.lock` holds no `torch`, `gsplat`, `nvidia-*` or `cuda-*` package.
    - _Requirements: 6.10_

  - [ ] 1.2 Pin Splat_Transform 3.8.0
    - Write `package.json` as in the design, then run `npm install --save-exact @playcanvas/splat-transform@3.8.0`, which writes `package-lock.json`. Then run `npm ci`. `node_modules/` is already in `.gitignore`.
    - Check that `package-lock.json` is under 1 MB and that `node_modules/@playcanvas/splat-transform/package.json` says version 3.8.0 and `"bin": {"splat-transform": "bin/cli.mjs"}`.
    - _Requirements: 6.11_

  - [ ] 1.3 Check every npm licence and update the Licence_List
    - List every package in `package-lock.json` with its `license` field. Read the `LICENSE` file and any third-party notices of each package that ships a native or WASM binary, `webgpu`'s Dawn build in particular, for the libraries bundled in it.
    - Check each licence against the permissive list in `.kiro/steering/rules.md`. IF any is copyleft, non-commercial, unclear or not on the list, stop and ask Samarth, as the rules say, and add nothing.
    - Look at what the installed numpy and scipy wheels bundle, in their `.libs` folders, and note on their rows any bundled library the rows don't name yet, like the Visual C++ runtime in scipy's Windows wheel.
    - In `docs/third-party-licences.md`: numpy and scipy become installed. The splat-transform 3.8.0 row's licence cell lists the licences found, with the bundled libraries noted, and its Installed cell says "Yes, on the laptop. Not yet on the machine image." Add a Node.js row as the design says, after reading Node's own `LICENSE` file for its bundled parts.
    - _Requirements: 6.12, 9.6_

  - [ ] 1.4 Add Node and `npm ci` to `README.md`
    - Under "Install": the clean and export step also needs Node 22 or newer, and then `npm ci` at the repo root, which installs splat-transform 3.8.0 into `node_modules/`.
    - _Requirements: 9.7_

  - [ ] 1.5 Write the setup tests in `tests/test_clean_setup.py`
    - `pyproject.toml`'s runtime dependencies name numpy and scipy, and no runtime or dev dependency is a GPU package. `uv.lock` names no `torch`, `gsplat`, `nvidia-*` or `cuda-*` package.
    - `package.json` pins `@playcanvas/splat-transform` to exactly `"3.8.0"`, and `package-lock.json`'s `node_modules/@playcanvas/splat-transform` entry has version 3.8.0.
    - No file under `splatpipe/`, read with `rglob` and `imports_of` from `tests/test_imports.py`, imports torch, torchvision, torchaudio, gsplat, moge, fused_ssim, nvdiffrast, cupy, pycuda, triton, jax or tensorflow. A test of the checker itself shows that a sample `import torch` inside a function is caught.
    - The Licence_List has one Node.js row, whose "How we use it" says it's never vendored, linked or shipped, and its splat-transform row names 3.8.0 and starts its Installed cell with "Yes". Use `licence_row` from `tests/test_docs.py`.
    - `README.md` names `npm ci` and Node 22.
    - _Requirements: 6.10, 6.11, 6.14, 8.5, 9.6, 9.7_

  - [ ] 1.6 Run the tests and commit
    - Run `uv run pytest`. Every test passes.
    - `git add pyproject.toml uv.lock package.json package-lock.json docs/third-party-licences.md README.md tests/test_clean_setup.py`, then commit those files with the message "Clean export: numpy, scipy, splat-transform 3.8.0 and their licences".
    - _Requirements: 6.12_

- [ ] 2. Settings
  - Files: `splatpipe/clean_export/__init__.py`, `splatpipe/clean_export/settings.py` and `tests/test_clean_settings.py` (new), `config.toml` and `tests/conftest.py` (changed).
  - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8, 6.9, 6.13, 6.15_

  - [ ] 2.1 Add `[clean_export]` to `config.toml`
    - Append the design's section, with its comments, after `[fake.quality]`: `min_opacity = 0.005`, `surface_cubes = 128`, `surface_min_faint = 8`, `surface_min_share = 0.1`, `neighbours = 8`, `isolation_ratio = 8` and `max_removed_fraction = 0.7` in `[clean_export]` itself, and `web_sh_bands` 0, 2 and 3 in the three Preset tables.
    - It holds no `outlier_sigma` and no `max_scale` (D1, D3).
    - _Requirements: 6.1, 6.2, 6.3, 6.4_

  - [ ] 2.2 Write `splatpipe/clean_export/__init__.py` and `settings.py`
    - `__init__.py` is only a docstring for now, saying what the step does. Task 8 adds `work` and `run`.
    - `settings.py` has `PRESET_KEYS`, `CleanSettings`, `is_plain_file_name()`, `as_written()` and `read_settings()`, as in the design's "settings.py" section.
    - `read_settings` reads each key with `values[key]`, so a missing one raises the Skeleton's message. It checks the three file names first, then the numbers, each with the design's message, naming `[clean_export.<preset>]` for `web_sh_bands` and `[clean_export]` for the rest.
    - A file name that isn't a `str` is refused before any string method touches it. A whole number is `type(value) is int`, and a number is never a `bool`. An `OverflowError` from `float()` is refused with the same message.
    - `as_written(value)` is `Fraction(repr(value))`, so `0.7` is exactly 7/10.
    - _Requirements: 6.5, 6.6, 6.7, 6.8, 6.9_

  - [ ] 2.3 Add the test config's `[clean_export]` and `set_clean_export` to `tests/conftest.py`
    - `config_text` gains a keyword-only `clean_export` argument, a mapping of values that replace the section's, where `None` leaves a key out. Its defaults are the design's test values. Existing calls stay as they are.
    - `set_clean_export(**values)` rewrites the test config with those values, like `set_fail`.
    - _Requirements: 6.15_

  - [ ] 2.4 Write unit tests for the settings in `tests/test_clean_settings.py`
    - The real `[clean_export]`, read by path, loads for each of the three Presets and holds the values of Req 6.4. Its raw TOML tables, read with `tomllib`, hold exactly the shared keys of Req 6.2 in `[clean_export]` and `web_sh_bands` alone in each Preset table, because the Skeleton only refuses a key that's in both places. `read_settings` accepts it for each Preset.
    - Each bad kind of value is refused, naming the key and the right table: `true`, `8.0` for a whole number, `"8"`, 0, -1, 1.5 for a fraction, 4 for `web_sh_bands`, and a 400-digit integer for a number.
    - Each bad file name of the design's test table is refused, naming the setting. `clean_ply` equal to `web_sog`, and either equal to `source_ply`, are refused.
    - `as_written(0.7) == Fraction(7, 10)` and `as_written(1) == 1`.
    - A missing key raises the Skeleton's message, naming `[clean_export]`, `[clean_export.fast]` and the key, through `read_settings` on a real `StepSettings` from `Config.step_settings`.
    - `set_clean_export(neighbours=3)` changes what `load_config()` gives, and `set_clean_export(neighbours=None)` leaves the key out.
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.6, 6.7, 6.8, 6.9, 6.13, 6.15_

  - [ ] 2.5 Run the tests and commit
    - Run `uv run pytest`. Every test passes, the comment check on the real config in `tests/test_config.py` included.
    - `git add config.toml splatpipe/clean_export/__init__.py splatpipe/clean_export/settings.py tests/conftest.py tests/test_clean_settings.py`, then commit those files with the message "Clean export: settings and the [clean_export] section".

- [ ] 3. The PLY reader and writer
  - Files: `splatpipe/clean_export/ply.py`, `tests/splats.py` and `tests/test_clean_ply.py` (new).
  - _Requirements: 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 4.1, 4.2, 4.3, 4.5, 10.3, 10.4, 10.5_

  - [ ] 3.1 Write `splatpipe/clean_export/ply.py`
    - The Format_Constants, `PlyHeader`, `read_header()`, `read_splat()` and `write_splat()`, as in the design's "ply.py" section.
    - `read_header` follows the design's six steps in order. It keeps `comment` lines and drops `obj_info` lines. A message names the file as `shown` gives it, names a bad type or name with `repr`, and points at a bad line by its number, never by echoing the line.
    - `write_splat` writes the header as Latin-1 with `\n` line endings and each type's short spelling from `PLY_NAMES`, swaps a big-endian array with `byteswap().view(...)`, writes with `tofile`, then calls `flush()` and `os.fsync()`. An `OSError` becomes the design's readable error naming the file.
    - _Requirements: 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 4.1, 4.2, 4.5_

  - [ ] 3.2 Write `tests/splats.py`
    - `splat_dtype`, `make_splat`, `ply_bytes` and `dense_grid`, as in the design's "Helpers and fixtures". `ply_bytes` writes its header by hand from the dtype, so it shares no code with `ply.py`.
    - _Requirements: 10.3_

  - [ ] 3.3 Write unit tests for the PLY reader and writer in `tests/test_clean_ply.py`
    - Little-endian and big-endian files read to the same values, a `\r\n` header reads, comments come back in order, and an `obj_info` line is read and left out of the written file.
    - Each refusal in the design's test table for this file names the file and its problem, and none echoes a whole bad line or holds "Traceback".
    - `write_splat` writes `binary_little_endian`, the kept count and the short type spellings, so `float32` comes out as `float`. Writing into a folder that doesn't exist raises the readable error naming the file.
    - _Requirements: 1.2, 1.4, 1.5, 1.6, 1.7, 1.8, 4.1, 4.2, 4.5, 10.4_

  - [ ] 3.4 Write the property test for Property 4 in `tests/test_clean_ply.py`, with Hypothesis
    - **Property 4: The Clean_Splat survives the round trip**
    - **Validates: Requirements 1.2, 4.1, 4.2, 4.3**
    - Draw the properties, their order, their types and spellings, the byte order, the comment lines and whether there's an `obj_info` line with Hypothesis, and the 1 to 40 Gaussians' bytes from `np.random.default_rng(seed)`, as the design says, so no example runs past Hypothesis's size limit. Use a `tempfile.TemporaryDirectory` for each example.
    - Compare the bytes, not the values, so NaN payloads count.
    - _Requirements: 1.2, 4.1, 4.2, 4.3, 10.5_

  - [ ] 3.5 Run the tests and commit
    - Run `uv run pytest`. Every test passes.
    - `git add splatpipe/clean_export/ply.py tests/splats.py tests/test_clean_ply.py`, then commit those files with the message "Clean export: PLY reader and writer".

- [ ] 4. The Non_Finite_Filter and the Opacity_Filter
  - Files: `splatpipe/clean_export/filters.py` and `tests/test_clean_filters.py` (new).
  - _Requirements: 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.12, 2.13, 10.12_

  - [ ] 4.1 Write `non_finite()` and `faint_to_drop()` in `splatpipe/clean_export/filters.py`
    - As in the design's "filters.py" section. `faint_to_drop` does the faint test on the logit, returns early when fewer than `surface_min_faint` Gaussians are faint, finds each Gaussian's cube from the Extent with `np.unique`, and works out Paint and the shares in log space with the per-cube log-sum-exp.
    - Write out the cases of `min_opacity` 0 and 1, of a share of 0 and 1, and of an Extent of 0, so no `math.log(0)`, `math.log1p(-1)` or division by 0 happens. The share test is the small helper `more_than_share`. Neither function changes its inputs, and each returns an empty result for empty input.
    - There's no scale filter. A Gaussian's size only enters through its Paint.
    - _Requirements: 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.12, 2.13_

  - [ ] 4.2 Write unit tests for the two filters in `tests/test_clean_filters.py`
    - The Non_Finite_Filter drops NaN, `inf` and `-inf` in each Required_Property and a rotation of all 0, and keeps a NaN in `f_dc_0`.
    - Real_Opacity 0.004 is dropped and 0.006 kept. In one cube, 8 Faint_Splats that paint well over 10% are kept, 7 are dropped, and 8 that paint just under 10% are dropped. `more_than_share(math.log(0.1), 0.0, 0.1)` is false, so a share of exactly 10% doesn't count as more. Faint_Splats spread one per cube are dropped. Identical positions make one cube.
    - With `np.errstate(over="raise", invalid="raise", divide="raise")`, opacity logits and log scales of ±3e38 go through. `min_opacity` 0 and 1 and `surface_min_share` 0 and 1 work.
    - A Gaussian with log scales of 10 is kept. Empty input drops none.
    - _Requirements: 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.12, 2.13, 10.12_

  - [ ] 4.3 Run the tests and commit
    - Run `uv run pytest`. Every test passes.
    - `git add splatpipe/clean_export/filters.py tests/test_clean_filters.py`, then commit those files with the message "Clean export: non-finite and opacity filters, with faint surfaces kept".

- [ ] 5. The Outlier_Filter
  - Files: `splatpipe/clean_export/filters.py` and `tests/test_clean_filters.py` (changed).
  - _Requirements: 2.9, 2.10, 2.11, 2.13, 10.12_

  - [ ] 5.1 Write `isolated()` in `splatpipe/clean_export/filters.py`
    - As in the design: `KDTree(...).query(..., k=neighbours + 1, workers=-1)`, each Gaussian's own entry taken out, or the last entry where its own isn't in the list, then `mean > ratio * median` with the median over the neighbours' own means.
    - It uses no cutoff worked out over the whole splat (D1). With `neighbours` or fewer Gaussians, or none, it drops none.
    - _Requirements: 2.9, 2.10, 2.11, 2.13_

  - [ ] 5.2 Write unit tests for the Outlier_Filter in `tests/test_clean_filters.py`
    - A Gaussian 100 units from a tight cluster is dropped. In a separate splat, a tight cluster and a sparse, even shell of Gaussians 50 units around it, like a sky, are all kept. Keep the two apart: in one splat, the lone Gaussian's nearest neighbours would be shell Gaussians, and it would stay unless the shell were dense.
    - A Gaussian built at exactly 8 times its Neighbour_Median is kept and one just over is dropped. Build it from distances that are exact in floating point.
    - 8 Gaussians or fewer are all kept. 12 Gaussians at one spot plus one nearby run with no error, and the result follows the rule.
    - The result matches a brute-force version written in the test with `itertools` and `math.dist` on a small set with no tied distances.
    - _Requirements: 2.9, 2.10, 2.11, 2.13, 10.12_

  - [ ] 5.3 Run the tests and commit
    - Run `uv run pytest`. Every test passes.
    - `git add splatpipe/clean_export/filters.py tests/test_clean_filters.py`, then commit those files with the message "Clean export: outlier filter with the local ratio".

- [ ] 6. The cleaning and its limits
  - Files: `splatpipe/clean_export/filters.py` and `tests/test_clean_filters.py` (changed).
  - _Requirements: 2.1, 2.8, 2.12, 2.14, 2.15, 2.16, 3.1, 3.2, 3.3, 3.4, 3.6, 3.7, 3.8, 7.3, 10.5_

  - [ ] 6.1 Write `Cleaning`, `clean()`, `check_kept()` and `kept_warning()` in `splatpipe/clean_export/filters.py`
    - `clean` runs the Filter_Order on rows by index, copies only the positions, opacities and scales to `float64`, and gives its counts as Python `int`s. It never changes `data`.
    - `check_kept` raises the design's 70% message first, comparing with `as_written`, then the `min_kept_splats` message. The 70% message gives the kept count too. `kept_warning` gives the design's warning or `None`, also with `as_written`.
    - _Requirements: 2.1, 2.16, 3.1, 3.2, 3.3, 3.6, 3.7, 3.8, 7.3_

  - [ ] 6.2 Write unit tests for the cleaning and its limits in `tests/test_clean_filters.py`
    - A NaN Gaussian isn't anyone's neighbour, and a faint Gaussian the Opacity_Filter dropped doesn't count in the Outlier_Filter. The counts add up and are Python `int`s.
    - At `max_removed_fraction = 0.7`, 7 removed of 10 passes and 8 of 10 fails. The message holds every count with thousands separators, the 70%, `out/splat.ply`, that nothing was written, and the settings to loosen.
    - `min_kept_splats` fails with its message. `kept_warning` warns below `warn_kept_fraction` and is `None` at it.
    - _Requirements: 2.1, 2.16, 3.1, 3.2, 3.3, 3.6, 3.7, 3.8, 7.3_

  - [ ] 6.3 Write the property test for Property 1 in `tests/test_clean_filters.py`, with Hypothesis
    - **Property 1: The cleaning never removes more than `max_removed_fraction`**
    - **Validates: Requirements 2.8, 3.1, 3.2, 3.3, 3.4, 3.6**
    - Use the design's `splats()` and `clean_settings()` strategies, with Hypothesis drawing the shape and a seeded rng the bulk values, and numpy's overflow, invalid-value and divide-by-zero errors set to raise around `clean`.
    - Check the message by its count phrases, like `remove 8 of the 10 Gaussians`, not by bare numbers.
    - _Requirements: 2.8, 3.1, 3.2, 3.3, 3.4, 3.6, 10.5_

  - [ ] 6.4 Write the property test for Property 2 in `tests/test_clean_filters.py`, with Hypothesis
    - **Property 2: The cleaning never changes the Gaussians it keeps**
    - **Validates: Requirements 2.1, 2.14**
    - Use the same strategies, whose Gaussians carry a `uid` property. Compare bytes, so NaN payloads count.
    - _Requirements: 2.1, 2.14, 10.5_

  - [ ] 6.5 Write the property test for Property 3 in `tests/test_clean_filters.py`, with Hypothesis
    - **Property 3: A Dense_Grid of opaque, finite Gaussians loses nothing**
    - **Validates: Requirements 2.12, 2.15**
    - Draw the lattice's shape, spacing, corner and SH bands with Hypothesis and the per-Gaussian values from a seeded rng, as the design says, and read the settings from the real `config.toml` by path, for the `default` Preset.
    - _Requirements: 2.12, 2.15, 10.5_

  - [ ] 6.6 Run the tests and commit
    - Run `uv run pytest`. Every test passes.
    - `git add splatpipe/clean_export/filters.py tests/test_clean_filters.py`, then commit those files with the message "Clean export: the cleaning, the 70% limit and its properties".

- [ ] 7. Splat_Transform
  - Files: `splatpipe/clean_export/export.py` and `tests/test_clean_export.py` (new), `tests/conftest.py` (changed).
  - _Requirements: 5.1, 5.2, 5.3, 5.5, 5.7, 5.8, 5.9, 5.10, 5.13, 10.6, 10.7, 10.9_

  - [ ] 7.1 Write `splatpipe/clean_export/export.py`
    - `PACKAGE`, `ROOT_PACKAGE`, `NODE_NEEDED`, `command()`, `arguments()` and `make_web_copy()`, as in the design's "export.py" section.
    - `command` finds Node with `shutil.which`, reads the installed package's `package.json` for its version and `bin`, and checks the version against the root `package.json`'s pin and the entry script against `PACKAGE`. `make_web_copy` looks `command` up when it runs, so tests can patch it.
    - `make_web_copy` runs an argument list with no shell, stderr merged into stdout, and the configured timeout, and turns each failure into the design's message.
    - _Requirements: 5.1, 5.2, 5.3, 5.5, 5.7, 5.8, 5.9, 5.10, 5.13_

  - [ ] 7.2 Add `stand_in_transform` to `tests/conftest.py`
    - `stand_in_transform(mode)` writes the design's small Python script into `tmp_path`, patches `splatpipe.clean_export.export.command` to give `[sys.executable, <script>]`, and returns a function that reads the arguments the script recorded. The modes are `write`, `fail`, `silent`, `empty` and `hang`.
    - _Requirements: 10.6_

  - [ ] 7.3 Write unit tests for Splat_Transform's handling in `tests/test_clean_export.py`
    - `arguments(...)` has `-g cpu` before the input, `--filter-harmonics` with the band count, the output last, and none of the GPU-only options.
    - With `shutil.which` patched to give `None`, `command()` says Node is missing. With `PACKAGE` patched to an empty folder, it says splat-transform isn't installed. With a `package.json` there for 3.9.0, it names both versions. Each message says to run `npm ci` and names Node 22.
    - With `stand_in_transform`: `fail` gives exit code 3 and lines 10 to 29, `hang` with a 1-second limit gives the timeout message within a few seconds, `silent` and `empty` give the no-file message, and `write` passes. A spy on `subprocess.run` shows a list and no `shell=True`.
    - _Requirements: 5.2, 5.3, 5.5, 5.7, 5.8, 5.9, 5.10, 5.13, 10.6, 10.9_

  - [ ] 7.4 Write the real Splat_Transform test in `tests/test_clean_export.py`
    - Skip it with "Node isn't installed" when `shutil.which("node")` is `None`, and with "splat-transform isn't installed: run npm ci at the repo root" when `PACKAGE / "package.json"` is missing.
    - Write a 4×4×4 Dense_Grid with 1 SH band and one `comment` line through `write_splat`, run `make_web_copy` with 1 band and a 120-second limit, and check that the `.sog` starts with `PK\x03\x04` and that its `meta.json` has `"version": 2` and a count of 64.
    - _Requirements: 5.1, 10.6, 10.7_

  - [ ] 7.5 Run the tests and commit
    - Run `uv run pytest`. Every test passes, and the real Splat_Transform test runs here, because task 1 installed it.
    - `git add splatpipe/clean_export/export.py tests/conftest.py tests/test_clean_export.py`, then commit those files with the message "Clean export: web copy with splat-transform 3.8.0".

- [ ] 8. The step and its CLI
  - Files: `splatpipe/clean_export/__main__.py` and `tests/test_clean_step.py` (new), `splatpipe/clean_export/__init__.py` and `tests/conftest.py` (changed).
  - _Requirements: 1.1, 1.4, 1.9, 1.10, 3.1, 3.5, 3.9, 4.4, 4.6, 4.7, 5.4, 5.6, 5.11, 5.12, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 8.1, 8.2, 8.3, 8.5, 8.6, 8.7, 8.8, 10.4, 10.8, 10.10, 10.11_

  - [ ] 8.1 Write `work` and `run` in `splatpipe/clean_export/__init__.py`, and `__main__.py`
    - `work` follows the design's eleven steps in order, with every path from `step` and the settings. It removes both earlier outputs before it reads anything, adds `source_splats` before the `max_source_splats` check and the other cleaning counts before `check_kept`, and moves both outputs into `out/` only once both exist. When the Web_Copy's move fails, it removes the Clean_Splat it just moved before it raises.
    - `run` and `__main__.py` copy the Fake_Step's, with `clean_export` and `splatpipe.clean_export`.
    - _Requirements: 1.1, 1.4, 1.10, 3.1, 3.9, 4.4, 4.6, 4.7, 5.4, 5.6, 5.11, 5.12, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 8.1, 8.2, 8.6, 8.7, 8.8_

  - [ ] 8.2 Add `splat_job` to `tests/conftest.py`
    - `splat_job(data, **ply_options)` makes a Job with `new_job`, writes `out/splat.ply` from `ply_bytes`, and returns the Job_Id.
    - _Requirements: 10.3_

  - [ ] 8.3 Write unit tests for a done run in `tests/test_clean_step.py`
    - With `stand_in_transform("write")`, a done run writes `out/clean.ply`, which reads back with the kept count, and `out/web.sog`. The report has outcome done, every count and timing of Req 7.1 as `int`s and seconds, and the 16 settings.
    - The source's bytes are the same after the run. Listing the Job_Folder before and after shows changes only under `work/clean_export/`, the two outputs and the report. A second done run replaces both outputs.
    - `web_sh_bands = 3` on a splat with 1 band passes `--filter-harmonics 1` and warns. A run under `warn_kept_fraction` warns and is done. `STEP_LIST` is still `()`.
    - _Requirements: 1.9, 4.7, 5.4, 5.12, 7.1, 7.3, 7.4, 7.5, 7.6, 8.3, 8.6, 8.7, 10.8, 10.11_

  - [ ] 8.4 Write unit tests for failed runs and the CLI in `tests/test_clean_step.py`
    - A run over 70% fails with the design's message and `expected: true`, has the counts up to `kept_splats` and the timings `read_ply` and `clean` only, and leaves neither output in `out/`, also when both were there from an earlier run. The source's bytes are the same.
    - A missing source, a too-big source and a bad setting each fail with their message. A too-big source's report has `source_splats` only. A `fail` stand-in leaves neither output in `out/`. With `os.replace` patched to fail on the Web_Copy's move only, `out/` holds neither output afterwards.
    - `clean_export.__main__.main(["--job", id])` returns 0 and prints `clean_export: done`, and returns 1 with the readable error on stderr and no "Traceback".
    - A subprocess runs `[sys.executable, "-m", "splatpipe.clean_export", "--job", "../x"]` from the repo root. It exits with 1, prints the readable error on stderr, holds no "Traceback" and writes no file.
    - _Requirements: 1.4, 1.9, 3.1, 3.5, 3.9, 4.4, 5.6, 7.2, 8.2, 8.8, 10.4, 10.8, 10.10_

  - [ ] 8.5 Run the tests and commit
    - Run `uv run pytest`. Every test passes.
    - `git add splatpipe/clean_export/__init__.py splatpipe/clean_export/__main__.py tests/conftest.py tests/test_clean_step.py`, then commit those files with the message "Clean export: the step and its CLI".

- [ ] 9. The Pipeline_Doc section
  - Files: `docs/pipeline.md` and `tests/test_clean_setup.py` (changed).
  - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 9.8_

  - [ ] 9.1 Write `## Step: clean_export` in `docs/pipeline.md`
    - After `## Step: fake`, with the parts the design lists: the opening paragraph, `### What it reads`, `### What it writes`, `### Config section`, `### Step report fields` and `### Files in out/`.
    - It describes the three filters with their settings, the 70% limit and what happens past it, and that there's no scale filter because big Gaussians are often the sky and the far background. It states the A1 contract and that the step never writes `out/splat.ply`.
    - "The step list" gains the design's sentence about this step's entry: the first step spec that adds an entry adds this one too, optional and last.
    - _Requirements: 9.1, 9.3, 9.4, 9.5, 9.8_

  - [ ] 9.2 Add the doc tests to `tests/test_clean_setup.py`
    - With `doc_parts` and `doc_part` from `tests/test_docs.py`: `docs/pipeline.md` has one `## Step: clean_export` with each of its five `###` parts.
    - Its "Files in out/" part names `out/clean.ply` and `out/web.sog` and says the step removes both before it writes either. Its "What it reads" part names `out/splat.ply` and says the step never writes it. "The step list" names the clean and export step.
    - The section names all seven cleaning settings of Req 6.4: `min_opacity`, `surface_cubes`, `surface_min_faint`, `surface_min_share`, `neighbours`, `isolation_ratio` and `max_removed_fraction`.
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 9.8_

  - [ ] 9.3 Run the tests and commit
    - Run `uv run pytest`. Every test passes, `tests/test_docs.py` included.
    - `git add docs/pipeline.md tests/test_clean_setup.py`, then commit those files with the message "Clean export: pipeline doc section".
