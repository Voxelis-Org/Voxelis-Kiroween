# Implementation Plan: clean

## Overview

This plan builds the Clean_Step in `splatpipe/clean/` in 9 tasks, in the order of the design's module layout: the settings and the `[clean]` config, the PLY reader and writer, the Broken_Filter with the neighbour search, the Faint_Filter, the Isolated_Filter, the Filter_Order with the Requirement 3 limits, and the splat-transform subprocess with its lockfile. Then `run()` and `work()` join them, and the last task writes the Pipeline_Doc section and the README line. Each Licence_List change goes in the same commit as the dependency or tool it describes, as `.kiro/steering/rules.md` asks, so task 2 makes the numpy and scipy ones and task 7 the rest. Each filter module is pure, so it's built and tested on arrays before the step that joins them exists. Each task writes its code with that code's unit and property tests, runs the whole suite and commits. The code is Python 3.11 with numpy and scipy, plus PlayCanvas splat-transform 3.8.0 run under Node. The tests use pytest and Hypothesis, which is already in the dev group.

Some files grow over several tasks, so each piece comes with the first task that uses it:

- `tests/splats.py`, the shared test helpers and Hypothesis strategies: task 1 writes `clean_settings` and the name strategy, task 2 adds the PLY builder, tasks 2 to 6 add each property's strategies, and task 8 adds `put_source`.
- `tests/test_clean_repo.py`, the checks on the repo's own files: task 1 writes the real-config and import checks, task 2 adds the dependency checks and the numpy and scipy rows' checks, task 7 the lockfile checks and the splat-transform, npm and Node rows' checks, task 8 the package's entry points, and task 9 the doc and README checks.
- `tests/test_clean_limits.py`: task 1 tests the settings checks, and task 6 adds the Requirement 3 limits.
- `tests/test_clean_filters.py`: tasks 3, 4 and 5 test one filter each, and task 6 adds the Filter_Order.
- `docs/third-party-licences.md`: task 2 marks the numpy and scipy rows installed, and task 7 names every package in the lockfile with its licence, marks the splat-transform 3.8.0 row installed, adds Node's line and adds the machine-image note to npm's.

## Working rules for every task

- Work on `feat/clean`, never on `main`.
- The step lives in `splatpipe/clean/`. Change no Skeleton module and no pipeline-core test. `STEP_LIST` stays empty, so `tests/test_run.py` keeps its `assert STEP_LIST == ()` and its three "No steps are set up yet" assertions (Req 8.3, A9). `tests/conftest.py` is shared, and this spec only adds the `[clean]` test config and `set_clean` to it.
- `splatpipe/clean/` imports only the standard library, numpy, `scipy.spatial` and `splatpipe`. It imports nothing from `cloud/`, boto3, torch or gsplat (Req 8.5).
- Task 2 adds numpy and scipy with `uv add`, as runtime dependencies. Hypothesis is already in the dev group. Add no GPU package, torch, gsplat, CUDA or `nvidia-*` wheel and no boto3, anywhere. Node runs splat-transform and nothing else.
- Every number comes from `config.toml`, except what the design keeps in code on purpose: the Format_Constants `PLY_TYPES`, `REQUIRED`, `SH_BANDS`, `LOG_PI`, `FLOAT32_HALF_STEP`, `HUGE_DIVISIONS`, `NODE_VERSION` and `CLI`, and `TAIL_CHARS` (Req 6.5).
- Tests run on the CPU with no network and no GPU, on tiny synthetic splats that the tests build as bytes, in `tmp_path` or a `tempfile.TemporaryDirectory()`. No splat file goes in git, and no test writes into the real `data/` (Req 10.1, 10.3). A test that needs the real `config.toml` reads it by path, as `REPO_ROOT / "config.toml"`, because `tests/conftest.py` points `CONFIG_PATH` at a test config.
- Every filter test runs inside `np.errstate(all="raise")`, so an overflow, an invalid value or a division by 0 fails it (Req 2.11).
- Subprocess tests pass a timeout, set `PYTHONIOENCODING=utf-8` in the child's environment and read its output as UTF-8 (Req 10.7).
- Property tests use Hypothesis with at least 100 examples (the default) and `@settings(deadline=None)`. Each docstring says `Feature: clean, Property N: <title>` and `**Validates: Requirements ...**`, as pipeline-core's do. The strategies live in `tests/splats.py`, as the design's "Strategies" section gives them. Every strategy for a property whose result depends on the Isolated_Filter's neighbour search, but the dense grid's and Property 3's, drops draws with tied Centre_Distances. A strategy for the Broken_Filter or the Faint_Filter alone may keep them, as Property 8's first half does. Property 3's draws coincident pairs and drops only a draw with a tie at a Gaussian's `k`-th nearest (Req 10.13).
- Every test sub-task is required (Req 10.2).
- Nothing here costs money. No task touches AWS, sets `APPROVED_SPEND` or runs anything paid.
- The `splat-files` skill has the PLY layout and splat-transform 3.8.0's options. Install and run splat-transform the way this spec says, though: `npm ci` at the repo root, then `node node_modules/@playcanvas/splat-transform/bin/cli.mjs`. Never use the skill's `npm install -g` or `npx`, because the step downloads nothing at run time (Req 5.2, A8).
- Nothing over 1 MB goes in git without asking first. Check the size of `uv.lock` and `package-lock.json` before committing them.
- The last sub-task of each task is its checkpoint. Run `uv run pytest` and fix what fails until every test passes, `tests/test_guard.py` and every pipeline-core test included. Then stage the task's files by name with `git add <file> ...`, never with `git add -A`, `git add .` or a folder, and commit only those files with `git commit -m "<title>" -m "<body>" -- <file> ...`, so nothing else that's staged goes in. The title follows the repo's `Topic: what` style, as `Clean step: <what>`, and the body says what the commit adds and why. Keep backticks and `$` out of both, because PowerShell reads them inside double quotes. Leave `docs/devlog.md` out: the devlog hook adds and commits its own line after each task.
- If a hook blocks something it shouldn't, say so and fix the hook. Never work around it.

## Tasks

- [x] 1. Settings and the `[clean]` config
  - Files: `splatpipe/clean/__init__.py`, `splatpipe/clean/settings.py`, `tests/splats.py`, `tests/test_clean_limits.py` and `tests/test_clean_repo.py` (new), `config.toml` and `tests/conftest.py` (changed).
  - _Requirements: 1.12, 4.7, 5.15, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8, 6.9, 6.10, 6.14, 6.15, 6.16, 6.17, 6.18, 8.5, 8.8, 10.1, 10.2, 10.4_

  - [x] 1.1 Write `splatpipe/clean/settings.py`
    - `CleanSettings`, `check_settings()` and `check_plain_name()`, as in the design's "settings.py" section. Make `splatpipe/clean/__init__.py` too, holding only the package's docstring for now. Task 8 adds `run()` and `work()`.
    - `check_settings` reads every key with `settings[key]`, so a missing key is the Step_Runner's own refusal through `StepSettings.__missing__`, with no new code (Req 6.6).
    - Then it checks in the design's order and raises `ExpectedFailure` at the first problem: the whole numbers, where only `type(value) is int` counts, so `True` and `4.0` are refused; the numbers, an `int` or a `float` but never a `bool`, where an `int` too big for a float is out of range; the three names; and the clashes. Each message names the setting, `[clean]` or `[clean.<preset>]`, the value and what it must be. Floats come back as `float(value)`.
    - `check_plain_name` refuses, in the design's order: a value that isn't a `str`, `""`, a name holding `/`, `\`, `os.sep` or `os.altsep`, `.` and `..`, a name holding `:`, a name holding a control character below U+0020, which it doesn't echo, and a name that `PurePosixPath` or `PureWindowsPath` calls absolute or gives a drive. With a suffix, the name must end in exactly `.ply` or `.sog`, in lower case.
    - The clashes compare `train_ply`, `clean_ply` and `clean_sog` after `casefold()`, name the two settings and their values, and say that names differing only in case are one file on Windows.
    - From `splatpipe`, import only `splatpipe.step` and `splatpipe.errors`.
    - _Requirements: 1.12, 4.7, 5.15, 6.5, 6.6, 6.7, 6.8, 6.9, 6.10, 8.8_

  - [x] 1.2 Add `[clean]` and its three Preset tables to `config.toml`
    - Append `[clean]`, `[clean.fast]`, `[clean.default]` and `[clean.quality]` after `[fake]`, exactly as the design's "Sections added to config.toml" gives them: the sixteen Shared_Settings in `[clean]`, and `sog_sh_bands` and `sh_iterations` in each Preset table, with the values of Req 6.4 and a comment on the line directly above every key.
    - Leave every other section exactly as it is.
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.17, 6.18_

  - [x] 1.3 Add the `[clean]` test config to `tests/conftest.py`, and `clean_settings` to `tests/splats.py`
    - `CLEAN_SHARED` and `CLEAN_PRESETS` hold the real keys with the real values, except `min_kept_splats = 1`, `max_workers = 1`, `splat_transform_timeout_seconds = 60`, `sh_iterations = 1` in every Preset, and `sog_sh_bands` of 0, 1 and 3 in `fast`, `default` and `quality`, as the design's "The test config's [clean]" section gives them.
    - `config_text` writes `[clean]` and its three Preset tables into every test config, after the Fake_Step sections. `set_fail` keeps working as it does now.
    - `set_clean(**values)` rewrites the test config with chosen `[clean]` values for one test, the way `set_fail` does for `fail`. A Preset table's key is given as `fast__sog_sh_bands=1`, and a value of `None` leaves the key out, so a test can check a missing setting.
    - `clean_settings(**values)`, in the new `tests/splats.py`, builds a `CleanSettings` from those dicts with chosen changes, for the tests of pure functions.
    - _Requirements: 6.16_

  - [x] 1.4 Write unit tests for the settings checks in `tests/test_clean_limits.py`
    - `check_settings` on the test config's `step_settings("clean", preset)` gives the eighteen values for each Preset, and `extent_trim_percent = 1` comes back as `1.0`.
    - A setting left out of `[clean]` with `set_clean(neighbours=None)`, and `sh_iterations` left out of all three Preset tables, each give the Step_Runner's `PipelineError`, naming `[clean]`, `[clean.<preset>]` and the key (Req 6.6).
    - Every whole-number setting of Req 6.7 is refused for `True`, a `float` such as `4.0`, a `str`, and a value just outside its range, like `neighbours = 0`, `sog_sh_bands = 4` and `max_workers = 9`. The ends of each range pass: `max_workers` and `sog_sh_bands` at 0, `sog_sh_bands` at 3 and `max_workers` at 8.
    - Every number setting of Req 6.8 is refused for `True`, a `str`, `10**400` and a value just outside its range: `min_opacity` at -0.1 and 1.1, `faint_min_paint_share` at 1, `extent_trim_percent` at 50, and `isolation_ratio` and `splat_transform_timeout_seconds` at 0. The ends that the ranges include pass, like `min_opacity` at 0 and 1, and `faint_min_paint_share` and `extent_trim_percent` at 0. Each range is checked on its own, because no setting's range depends on another's.
    - Each refusal names the setting, its section and what it must be.
    - The names: `../x.ply`, `a/b.ply`, `a\b.ply`, `C:clean.ply`, `/abs.ply`, `clean.ply:x`, `.`, `..`, `""`, `7`, and a name holding `\x00` or `\n`, are each refused, naming the setting, and a name with a control character isn't echoed. `clean.PLY` and `clean.sog` are refused for `clean_ply`, and `clean.ply` for `clean_sog` (Req 6.9).
    - `train_ply = "Clean.ply"` next to `clean_ply = "clean.ply"` is refused, naming both settings and saying that names differing only in case are one file on Windows, and so are the other two pairs (Req 6.10).
    - Every refusal but the missing setting's is an `ExpectedFailure`, and no message holds "Traceback".
    - _Requirements: 6.6, 6.7, 6.8, 6.9, 6.10, 8.8, 10.4_

  - [x] 1.5 Write the property test for Property 10 in `tests/test_clean_limits.py`, with Hypothesis
    - **Property 10: The file-name checks accept exactly the plain, distinct names**
    - **Validates: Requirements 1.12, 4.7, 5.15, 6.9, 6.10**
    - Add the design's "Names" strategy to `tests/splats.py`: `st.text()`; good names with `/`, `\`, `:`, `.`, `\x00`, `\n`, `\u2215` or `\uff0f` put in at a random place; `..`, `C:clean.ply`, `/abs.ply`, `clean.ply:x` and `clean.PLY`; and for the second half, good names with their case variants, so clashes turn up often.
    - The oracle accepts a name that isn't empty, isn't `.` or `..`, holds none of `/`, `\`, `:` or a character below U+0020, and ends in exactly the suffix when there is one. It shares no code with `check_plain_name`.
    - For three names that pass, `check_settings` refuses them exactly when two are equal after `str.casefold`, naming both settings.
    - _Requirements: 1.12, 4.7, 5.15, 6.9, 6.10_

  - [x] 1.6 Write the real-config and import checks in `tests/test_clean_repo.py`
    - Read `REPO_ROOT / "config.toml"` by path. `[clean]` loads with `step_settings("clean", preset)` for each of the three Presets, its keys are exactly the sixteen of Req 6.2 and the two of Req 6.3, its values are those of Req 6.4, and `check_settings` accepts it (Req 6.1 to 6.4, 6.14).
    - `max_source_splats` is at most `2 * 10**9 // 248`, a quarter of 8 GB at 62 float32 properties a Gaussian (Req 6.17).
    - Read `config.toml` as text: every key in `[clean]`, `[clean.fast]`, `[clean.default]` and `[clean.quality]` has a `#` comment with text on the line directly above it. A comment at the end of the key's line, or two lines above it, doesn't count here, though `tests/test_config.py` accepts the first. Test the helper on a short sample of each case too (Req 6.18).
    - No `*.py` under `splatpipe/`, found with `rglob`, imports `torch`, `torchvision`, `gsplat`, `cupy`, `pycuda`, `triton`, `nvidia` or `cuda`, or a submodule of one (Req 6.15).
    - Every `*.py` in `splatpipe/clean/` imports only names in `sys.stdlib_module_names`, `numpy`, `scipy` and `splatpipe`, and nothing from `cloud` (Req 8.5).
    - Read the files with `ast`, the way `tests/test_imports.py` does, and leave that file as it is. A failure lists the file and the line.
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.14, 6.15, 6.17, 6.18, 8.5_

  - [x] 1.7 Run the tests and commit
    - Run `uv run pytest`. Every test passes, `tests/test_guard.py` included, and so does the comment check in `tests/test_config.py`, which now reads `[clean]` too.
    - On `feat/clean`: `git add splatpipe/clean/__init__.py splatpipe/clean/settings.py config.toml tests/conftest.py tests/splats.py tests/test_clean_limits.py tests/test_clean_repo.py`, then commit those files with the title "Clean step: settings and the [clean] config" and the body "Adds the clean step's eighteen settings to config.toml, each with a comment above it, and the same section to the test config. settings.py checks every kind, range and file name before the step opens, writes or removes anything, so a bad setting can't reach a path outside out/ or overwrite the trained splat."
    - _Requirements: 10.1, 10.2_

- [x] 2. PLY reader and writer, with numpy and scipy
  - Files: `splatpipe/clean/ply_read.py`, `splatpipe/clean/ply_write.py` and `tests/test_clean_ply.py` (new), `pyproject.toml`, `uv.lock`, `docs/third-party-licences.md`, `tests/splats.py` and `tests/test_clean_repo.py` (changed).
  - _Requirements: 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 1.10, 1.11, 4.1, 4.2, 4.3, 4.4, 4.6, 5.14, 6.5, 6.11, 8.8, 9.7, 10.1, 10.2, 10.3, 10.4, 10.5, 10.12_

  - [x] 2.1 Add numpy and scipy, and mark them installed in `docs/third-party-licences.md`
    - Run `uv add numpy scipy`, which puts both in `[project].dependencies` with the lower bounds uv writes, and updates `uv.lock`. scipy is for task 3's neighbour search.
    - Keep the comment above `dependencies` that keeps the GPU packages out. Add no other package.
    - In `docs/third-party-licences.md`, change the numpy and scipy rows' Installed cell from "Not yet" to "Yes". Keep the rest of both rows as it is, their bundled-library notes included.
    - _Requirements: 6.11, 9.7_

  - [x] 2.2 Write `splatpipe/clean/ply_read.py`
    - The Format_Constants `PLY_TYPES`, `REQUIRED` and `SH_BANDS`, then `PlyProperty`, `PlyHeader` with its packed `dtype(byte_order)`, `read_header()` and `read_splat()`, as in the design's "ply_read.py" section.
    - `read_header` opens the file with `open(path, "rb")` and never with a write mode. It reads the header with `readline()`, decodes each line as Latin-1, and strips one `\n` and then one `\r`. It keeps every `comment` and `obj_info` line verbatim, in order, and finds every property by name, never by position.
    - It refuses each of these with an `ExpectedFailure` that names the file: a first line that isn't `ply`; `ascii`, saying splat-transform can't read an ascii PLY either; a header line it can't read, with its line number and the line; a `property` line before any `element`; a bad count; an unknown type; a property named twice in one element; a list property in `vertex` or in an element before it, naming it; no `vertex` element; missing Required_Properties, naming each; a count of 0; an `f_rest_*` count that isn't 0, 9, 24 or 45, naming the count and the four allowed; and a file shorter than `data_offset + count × row size`, saying it's cut short or still being written.
    - A missing file gets the message that names the file and says the training step writes it, so training has to run first. Any other `OSError` gives the file and `error.strerror`.
    - `read_splat` reads with `np.fromfile(..., count=header.count, offset=header.data_offset)`, refuses a short result as cut short, and turns a big-endian array little-endian in place with `byteswap(inplace=True)` and a `view`, so every value keeps its bits.
    - _Requirements: 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 1.10, 1.11, 6.5, 8.8_

  - [x] 2.3 Write `splatpipe/clean/ply_write.py`
    - `header_bytes()` and `write_splat()`, as in the design's "ply_write.py" section.
    - `header_bytes` writes `format binary_little_endian 1.0`, every kept comment in its order, `element vertex <count>`, and each property with its declared type in the source's order, encoded as Latin-1.
    - `write_splat` makes the one copy `data[rows]`, opens the file with mode `"xb"`, writes the header and then `kept.tofile(handle)`, and returns the file's size for `clean_ply_bytes`. An `OSError` while opening or writing removes the partial file and raises an `ExpectedFailure` naming the file and `error.strerror`.
    - _Requirements: 4.1, 4.2, 4.3, 4.6, 5.14, 8.8_

  - [x] 2.4 Add the PLY builder to `tests/splats.py`
    - `ply_bytes(fields, rows, *, byte_order="<", comments=(), before=(), after=(), eol=b"\n")` writes the header by hand and packs the rows with `struct`, from its own table of PLY types to `struct` codes. It shares no code with `ply_read.py` or `ply_write.py`. A bulk path packs with numpy, for task 8's test of 20,000 Gaussians.
    - `standard_fields(sh_count=45, normals=True)` gives A1's layout, and `gaussians(n, ...)` fills plain values: unit quaternions, Real_Opacities above `min_opacity`, stored scales from -8 to 3 and centres on a jittered lattice.
    - _Requirements: 10.1, 10.3_

  - [x] 2.5 Write unit tests for the reader and the writer in `tests/test_clean_ply.py`
    - Both byte orders, `\r\n` header lines, an element before and one after `vertex`, and every type in `PLY_TYPES` read back right. Reading leaves the file's bytes the same.
    - `f_rest_*` counts of 0, 9, 24 and 45 give 0 to 3 SH_Bands, and 1, 8, 44 and 46 are refused, naming the count (Req 1.5).
    - A missing file is refused, naming the file and the training step (Req 1.6). A folder in its place gives the OS's reason.
    - Not a PLY, `ascii`, no `vertex`, a bad header line, an unknown type, a property listed twice, a missing `scale_2`, as in a 2DGS splat, and `opacity_linear` in place of `opacity` are each refused, naming the file and the problem, with missing properties by name (Req 1.7). So are a list property in `vertex` (Req 1.8), a count of 0 (Req 1.9), and a file one byte or one row short (Req 1.10).
    - The carry-through example: a splat with `nx`, `ny`, `nz`, a `double` property called `confidence` and the comments `SplatRenderMode: mip` and `postshot.anti_aliasing=1`, written for a subset of its rows and read back, has the same property names, order, types, values and comments (Req 10.12).
    - A write that raises `OSError` with `ENOSPC` partway through names the file and the reason, and leaves no partial file (Req 4.6).
    - No message holds "Traceback".
    - _Requirements: 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 1.10, 1.11, 4.1, 4.2, 4.3, 4.6, 10.4, 10.12_

  - [x] 2.6 Write the property test for Property 9 in `tests/test_clean_ply.py`, with Hypothesis
    - **Property 9: A Clean_Splat reads back bit for bit**
    - **Validates: Requirements 1.2, 1.3, 1.4, 4.1, 4.2, 4.3, 4.4**
    - This is the bit-for-bit PLY round trip. Add the design's "Layouts" and "Values" strategies to `tests/splats.py`: A1's layout shuffled, extra properties with any type in `PLY_TYPES`, either byte order, an `f_rest_*` count of 0, 9, 24 or 45, Latin-1 comments that sometimes hold `SplatRenderMode: mip` or `postshot.anti_aliasing=1`, an optional `element meta 2` before or after `vertex`, and values from `st.from_dtype`. The subset of rows is any increasing one that isn't empty, because the reader refuses a count of 0.
    - Build the bytes with `ply_bytes`, read them, write the subset and read it back, in a `tempfile.TemporaryDirectory()` the test makes itself. Check the names, their order and types, the comments, `binary_little_endian`, the count, and every field's raw bytes, compared as unsigned integers so NaN payloads, infinities and -0.0 count.
    - _Requirements: 1.2, 1.3, 1.4, 4.1, 4.2, 4.3, 4.4, 10.5_

  - [x] 2.7 Add the dependency and licence-row checks to `tests/test_clean_repo.py`
    - `pyproject.toml`'s `[project].dependencies` name numpy and scipy, and neither they nor the dev group name a GPU package: torch, torchvision, gsplat, cupy, pycuda, triton, or a name that starts with `nvidia-` or `cuda-`.
    - `uv.lock` holds no package with one of those names, so `uv sync` on a laptop installs none.
    - Read `docs/third-party-licences.md` by path and find a row by its Name cell, the way `licence_row` in `tests/test_docs.py` does, and leave that file as it is. The numpy and scipy rows each start their Installed cell with "Yes" (Req 9.7).
    - _Requirements: 6.11, 9.7_

  - [x] 2.8 Run the tests and commit
    - Run `uv run pytest`. Every test passes, `tests/test_guard.py` included. Check that `uv.lock` is under 1 MB.
    - On `feat/clean`: `git add pyproject.toml uv.lock docs/third-party-licences.md splatpipe/clean/ply_read.py splatpipe/clean/ply_write.py tests/splats.py tests/test_clean_ply.py tests/test_clean_repo.py`, then commit those files with the title "Clean step: PLY reader and writer, with numpy and scipy" and the body "Adds numpy and scipy as runtime dependencies, with their licence-list rows marked installed, and ply_read.py and ply_write.py, which read a binary 3DGS PLY in either byte order by property name and write the kept rows back little-endian with every property and comment carried through. The round-trip property checks that every value comes back bit for bit, NaN payloads included."
    - _Requirements: 10.1, 10.2_

- [x] 3. Broken filter and the neighbour search
  - Files: `splatpipe/clean/neighbours.py`, `splatpipe/clean/broken.py` and `tests/test_clean_filters.py` (new), `tests/splats.py` (changed).
  - _Requirements: 2.2, 2.3, 2.11, 2.16, 2.19, 8.9, 10.1, 10.2, 10.3, 10.5, 10.13_

  - [x] 3.1 Write `splatpipe/clean/neighbours.py`
    - `positions()` and `nearest_others()`, as in the design's "neighbours.py" section.
    - `positions` fills a new C-ordered `(len(rows), 3)` float64 array column by column from `x`, `y` and `z` at `rows`.
    - `nearest_others` queries with `k=list(range(1, k + 2))`, so scipy always keeps the second axis, and passes `workers` on. It drops each point from its own row wherever it is, and a row that doesn't hold the point loses its last column. It turns the self mask into the mask of the others in place, and frees the raw arrays one at a time, as the design's "Memory" section counts them.
    - _Requirements: 2.16, 8.9_

  - [x] 3.2 Write `splatpipe/clean/broken.py`
    - `broken_keep()`, as in the design's "broken.py" section: `np.isfinite` on every field in `REQUIRED`, and a quaternion that isn't four zeros, tested with `== 0` on each component and no squares.
    - It reads no field outside `REQUIRED`.
    - _Requirements: 2.2, 2.3, 2.19_

  - [x] 3.3 Write unit tests for the Broken_Filter and the neighbour search in `tests/test_clean_filters.py`
    - `broken_keep` drops a Gaussian with NaN, +infinity or -infinity in each Required_Property in turn, and one with the quaternion `(0, -0.0, 0, 0)`. It keeps one whose only NaN is in a colour property, and one whose quaternion is the smallest subnormal float32 in one component and 0 in the rest (Req 2.2, 2.3).
    - Quaternion components of ±3e38 raise nothing (Req 2.11).
    - `positions` gives the float32 centres exactly, as float64.
    - `nearest_others` with more than `k + 1` coincident centres, and with another point at distance 0 ahead of a point in its own row, gives every point `k` others, none of them itself (Req 2.16).
    - Every test here runs inside `np.errstate(all="raise")`.
    - _Requirements: 2.2, 2.3, 2.11, 2.16, 10.1_

  - [x] 3.4 Write the property test for Property 2 in `tests/test_clean_filters.py`, with Hypothesis
    - **Property 2: The Broken_Filter keeps exactly the Gaussians that can be drawn**
    - **Validates: Requirements 2.2, 2.3, 2.19**
    - Add the design's "Splats" strategy to `tests/splats.py`, which Properties 1 and 7 use too: 1 to 60 Gaussians; centres from lattice points, clusters and far outliers up to ±3e38, with the index-based jitter; stored opacities and scales mostly in their plain ranges, plus the extremes and the exact logit of `min_opacity`; quaternions mostly unit, plus all-zero ones; and NaNs and infinities in Required_Properties, up to every Gaussian, and in Carried_Properties. Add the brute-force tie check that drops, with `assume`, a draw in which a Gaussian has two equal Centre_Distances to others (Req 10.13).
    - The oracle looks at each Gaussian with `math.isfinite` and `==`, one value at a time. Running `broken_keep` again on what it kept keeps every one.
    - _Requirements: 2.2, 2.3, 2.19, 10.5, 10.13_

  - [x] 3.5 Write the property test for Property 3 in `tests/test_clean_filters.py`, with Hypothesis
    - **Property 3: No Gaussian is its own neighbour**
    - **Validates: Requirements 2.16**
    - Draw sets of more than `k` centres, coincident pairs included, and keep only those that meet the rule of Req 10.13 for that set and its `k`, with `assume`.
    - Each Gaussian gets `k` tree indices, none its own and none twice. They're exactly its `k` nearest others found by brute force, with the brute-force Centre_Distances.
    - _Requirements: 2.16, 10.13_

  - [x] 3.6 Run the tests and commit
    - Run `uv run pytest`. Every test passes, `tests/test_guard.py` included.
    - On `feat/clean`: `git add splatpipe/clean/neighbours.py splatpipe/clean/broken.py tests/splats.py tests/test_clean_filters.py`, then commit those files with the title "Clean step: broken filter and neighbour search" and the body "Adds broken.py, which drops the Gaussians nothing can draw, those with a NaN or infinity in a required property or a zero-length rotation, and leaves a bad colour as it is. neighbours.py finds each Gaussian's nearest other Gaussians with a KD-tree and never counts a Gaussian as its own neighbour, even when centres coincide."
    - _Requirements: 10.1, 10.2_

- [x] 4. Faint filter
  - Files: `splatpipe/clean/faint.py` (new), `tests/splats.py` and `tests/test_clean_filters.py` (changed).
  - _Requirements: 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 2.10, 2.11, 2.20, 6.5, 8.9, 10.1, 10.2, 10.3, 10.5_

  - [x] 4.1 Write `splatpipe/clean/faint.py`
    - The Format_Constants `LOG_PI`, `FLOAT32_HALF_STEP` and `HUGE_DIVISIONS`, then `log_opacity()`, `log_paint()`, `cube_groups()` and `faint_keep()`, as in the design's "faint.py" section. The Faint_Filter builds no tree and searches for no neighbours.
    - `log_opacity` is `-np.logaddexp(0.0, -stored)`, and `log_paint` is `log_alpha + LOG_PI + top_two`, so no `exp` of a stored scale is ever taken.
    - `cube_groups` follows the design's six steps: the Trimmed_Box from `np.percentile`, one axis at a time; the Extent as `max(0.0, longest side)`; one Cube when the Extent is 0; the Cube_Edge, with the centre path when the edge is at most `FLOAT32_HALF_STEP` or the divisions are `HUGE_DIVISIONS` or more; the Cube coordinates, made in place with `np.subtract` and `np.floor_divide` in one `(3, n)` float64 array and never cast to an integer; and the groups, from `np.lexsort` and a scan for changes.
    - `faint_keep` follows the design's nine steps. An empty set gives an empty answer. `min_opacity` at 1 makes every Gaussian a Faint_Gaussian without trusting `la < 0`. A set with no Faint_Gaussian is kept whole at once, with no Cubes. Each Cube's Paints are shifted by its largest log Paint before `exp`, and the sums come from `np.bincount`. A Cube keeps its Faint_Gaussians when `faint_count >= min(faint_min_per_cube, n + 1)` and `faint_paint > faint_min_paint_share * cube_paint`, and the paint condition is taken as met when the share is 0.
    - Only the underflow each step allows runs inside `np.errstate(under="ignore")`. Free the arrays as the design's "Memory" section counts them.
    - _Requirements: 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 2.10, 2.11, 6.5, 8.9_

  - [x] 4.2 Write unit tests for the Faint_Filter in `tests/test_clean_filters.py`
    - `log_opacity` of 0, ±1, ±40, ±1000 and ±3.4e38 is finite and at most 0, and `log_paint` with stored scales of ±3.4e38 is finite (Req 2.11).
    - The cube rule: a Cube of `faint_min_per_cube` Faint_Gaussians that paint more than `faint_min_paint_share` of it keeps them all, one with a stored opacity of -1000 included. The same Cube with one Faint_Gaussian fewer, or a little less Faint_Paint, drops them all. A Gaussian at or above `min_opacity` beside them is always kept (Req 2.4, 2.5, 2.9).
    - `min_opacity` at 0 makes no Faint_Gaussian, and at 1 makes every Gaussian one, a stored opacity of 3.4e38 included. A `faint_min_paint_share` of 0 keeps a Cube whose Faint_Paint underflows next to a bright Gaussian.
    - With `extent_trim_percent` at 0 and corners at 0 and 32, a centre exactly on a Cube face goes to the Cube above it, and one a float64 step below the face goes to the Cube below (Req 2.7).
    - 200 Gaussians with one far outlier: at `extent_trim_percent` 0 the outlier sets the Cube_Edge, and at 1 it doesn't, and lands in a far Cube of the same size (Req 2.6, 2.7). A centre at 3e38 raises nothing (Req 2.11).
    - Every centre the same gives one Cube, and so does a set of one (Req 2.8). An empty set gives an empty answer (Req 2.10).
    - `HUGE_DIVISIONS` divisions, and subnormal centres with a tiny Extent, take the centre path, and each distinct centre gets a Cube of its own.
    - Every test here runs inside `np.errstate(all="raise")`.
    - _Requirements: 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 2.10, 2.11, 10.1_

  - [x] 4.3 Write the property test for Property 4 in `tests/test_clean_filters.py`, with Hypothesis
    - **Property 4: The Faint_Filter keeps exactly what the cube rule keeps**
    - **Validates: Requirements 2.4, 2.5, 2.6, 2.7, 2.9, 2.10, 2.20**
    - Add the design's "Splats for the references" strategy for this property, and its "Settings" strategy, to `tests/splats.py`: 0 to 60 Gaussians, so the empty set is drawn too, with finite Required_Properties, stored opacities from -12 to 12 and stored scales from -8 to 3.
    - The reference is the design's, in plain Python on the whole set it's given: the logistic Real_Opacity, the Trimmed_Box from `np.percentile`, each Cube as the tuple of `(p - low) // edge` with Python's `//` as a dict key, Paint from `math.exp`, and the sums with `math.fsum`. It takes no logarithm and shifts no sum.
    - Drop with `assume` a draw with a Real_Opacity within a relative 1e-9 of `min_opacity`, one with a Cube whose Faint_Paint is within a relative 1e-9 of `faint_min_paint_share` times its Cube_Paint, and one whose Cube_Edge is above 0 but at most `FLOAT32_HALF_STEP`.
    - _Requirements: 2.4, 2.5, 2.6, 2.7, 2.9, 2.10, 2.20, 10.5_

  - [x] 4.4 Write the property test for Property 8 in `tests/test_clean_filters.py`, with Hypothesis
    - **Property 8: Large stored values change nothing and raise no floating-point error**
    - **Validates: Requirements 2.11**
    - Add the design's "Splats for the extremes" strategies to `tests/splats.py`.
    - First half: for any finite float32 stored opacities, stored scales and centres, ±3.4e38 and the subnormals included, `faint_keep` raises no `FloatingPointError` under `np.errstate(all="raise")`, `log_opacity` is finite and at most 0, and `log_paint` is finite.
    - Second half: adding a constant `c` of up to ±10⁶ to every stored scale, and multiplying every centre by `2**m` with `m` from -20 to 20, leaves `faint_keep`'s answer the same. The strategy draws the values so that every one stays exact in float32. Drop with `assume` a draw whose share test is within a relative 10⁻⁶ of its threshold.
    - _Requirements: 2.11_

  - [x] 4.5 Run the tests and commit
    - Run `uv run pytest`. Every test passes, `tests/test_guard.py` included.
    - On `feat/clean`: `git add splatpipe/clean/faint.py tests/splats.py tests/test_clean_filters.py`, then commit those files with the title "Clean step: faint filter" and the body "Adds faint.py, which judges every faint Gaussian with the rest of its cube, so a faint sheet that paints a visible surface stays and scattered faint specks go, with no opacity floor. It works in the log domain and never casts a cube to an integer, so no finite stored value can overflow it, and the properties check it against a plain-Python reference."
    - _Requirements: 10.1, 10.2_

- [x] 5. Isolated filter
  - Files: `splatpipe/clean/isolated.py` (new), `tests/splats.py` and `tests/test_clean_filters.py` (changed).
  - _Requirements: 2.12, 2.13, 2.14, 2.15, 2.16, 2.21, 8.9, 10.1, 10.2, 10.3, 10.5, 10.13_

  - [x] 5.1 Write `splatpipe/clean/isolated.py`
    - `isolated_keep()`, as in the design's "isolated.py" section, with the `workers` it's given.
    - A set of `neighbours` Gaussians or fewer is kept whole, with no tree. Otherwise it builds the step's one `cKDTree`, gets every point's nearest others with `nearest_others`, and works out every Neighbour_Distance and Neighbour_Median from that same set before it drops any Gaussian.
    - The median is `np.median(nd[idx], axis=1, overwrite_input=True)`, and `dist` and `idx` are freed as the design says. A Gaussian is kept when its median is 0 or `nd / med <= isolation_ratio`, with the division taken only where `med > 0`.
    - _Requirements: 2.12, 2.13, 2.14, 2.15, 2.16, 8.9_

  - [x] 5.2 Write unit tests for the Isolated_Filter in `tests/test_clean_filters.py`
    - A set of exactly `neighbours` Gaussians, however far apart, is kept whole, and no tree is built, as a spy on `cKDTree` shows (Req 2.14).
    - Coincident centres give a Neighbour_Median of 0 and are kept, with no division by 0 (Req 2.15).
    - One far Gaussian next to a tight cluster is dropped (Req 2.12).
    - The test is local: two even grids far apart, one 100 times sparser than the other, both keep every Gaussian, which a global distance cutoff wouldn't (A5).
    - Every test here runs inside `np.errstate(all="raise")`.
    - _Requirements: 2.12, 2.14, 2.15, 10.1_

  - [x] 5.3 Write the property test for Property 5 in `tests/test_clean_filters.py`, with Hypothesis
    - **Property 5: The Isolated_Filter keeps exactly what its rule keeps**
    - **Validates: Requirements 2.12, 2.13, 2.14, 2.21**
    - Add the design's "Splats for the references" strategy for this property to `tests/splats.py`: 1 to 60 finite centres, with task 3's tie check.
    - The reference keeps the whole set when it holds `neighbours` Gaussians or fewer. Otherwise it works out by brute force with `math.dist`, on the set it's given, each Gaussian's `neighbours` nearest others, its Neighbour_Distance, and the median of those neighbours' Neighbour_Distances with `statistics.median`. It keeps the Gaussian when the median is 0 or the distance is at most `isolation_ratio` times the median.
    - Drop with `assume` a draw within a relative 1e-9 of that threshold.
    - _Requirements: 2.12, 2.13, 2.14, 2.21, 10.5, 10.13_

  - [x] 5.4 Run the tests and commit
    - Run `uv run pytest`. Every test passes, `tests/test_guard.py` included.
    - On `feat/clean`: `git add splatpipe/clean/isolated.py tests/splats.py tests/test_clean_filters.py`, then commit those files with the title "Clean step: isolated filter" and the body "Adds isolated.py, which drops a Gaussian whose mean distance to its nearest neighbours is many times the median of theirs. The test is local, so a sparse but even background, like a far wall or the sky, survives where a global distance cutoff would eat it."
    - _Requirements: 10.1, 10.2_

- [ ] 6. Filter order and the removal limits
  - Files: `splatpipe/clean/filters.py` and `splatpipe/clean/limits.py` (new), `tests/splats.py`, `tests/test_clean_filters.py` and `tests/test_clean_limits.py` (changed).
  - _Requirements: 2.1, 2.13, 2.17, 2.18, 2.22, 2.23, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 7.2, 7.3, 8.8, 8.9, 10.1, 10.2, 10.4, 10.5, 10.13_

  - [x] 6.1 Write `splatpipe/clean/filters.py`
    - `Kept` and `run_filters()`, as in the design's "filters.py" section: `broken_keep` on the Splat_Data, `faint_keep` on what it kept, then `isolated_keep` on what that kept, each given the set before it as increasing index arrays, never as copies of rows.
    - Build the faint inputs at `rows1`: the positions, and the stored opacity and the sum of the two largest stored scales, in float64. Free `points1` and the faint arrays before the isolated stage, as the design's "Memory" section says. Give the isolated stage `max(1, max_workers)` workers, because scipy refuses 0.
    - Add `dropped_broken`, `dropped_faint`, then `dropped_isolated` and `kept_splats` through `add_count` as soon as each is known, each as an `int`. Never write to the Splat_Data.
    - _Requirements: 2.1, 2.17, 2.18, 7.2, 7.3, 8.9_

  - [x] 6.2 Write `splatpipe/clean/limits.py`
    - `check_source_count()` and `check_limits()`, as in the design's "limits.py" section, pure and raising `ExpectedFailure`.
    - `check_source_count` refuses a count above `max_source_splats`, giving both numbers and saying the setting in `[clean]` can be raised on a machine with more memory.
    - `check_limits` works out the Removed_Fraction from the faint and isolated drops only. It checks the Removal_Cap, then `min_kept_splats`, then `warn_kept_fraction`, in that order, and returns the warning or `None`. The two failure messages give the counts and each filter's drops, the first gives the fraction and the cap as percentages and says that broken Gaussians don't count toward the cap, and both name the settings in `[clean]` to loosen, as the design lists them. It never trims the removals back to the cap.
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 8.8_

  - [ ] 6.3 Write unit tests for the Filter_Order in `tests/test_clean_filters.py`
    - A Gaussian the Faint_Filter drops is no longer anyone's neighbour in the Isolated_Filter, which shows the Filter_Order and that each filter gets the set the one before it kept (Req 2.1, 2.13).
    - The counts reach `add_count` in order, each as an `int`, as each filter finishes. When `isolated_keep` is made to raise, only `dropped_broken` and `dropped_faint` were added (Req 2.17, 7.2, 7.3).
    - `max_workers` at 0 runs, and the query gets 1 worker.
    - Every test here runs inside `np.errstate(all="raise")`.
    - _Requirements: 2.1, 2.13, 2.17, 7.2, 7.3, 10.1_

  - [ ] 6.4 Write the property test for Property 1 in `tests/test_clean_filters.py`, with Hypothesis
    - **Property 1: The filters run in order and keep an ordered subset**
    - **Validates: Requirements 2.1, 2.17, 2.18**
    - This is the user's "it never changes the Gaussians it keeps" property, for the filters. Task 2's Property 9 carries it through the Clean_Splat.
    - Use task 3's "Splats" strategy and task 4's "Settings" strategy. `run_filters` gives back increasing rows, none repeated, and they're exactly the rows that calling `broken_keep`, `faint_keep` and `isolated_keep` one after the other gives. The Splat_Data's raw bytes are the same after `run_filters` as before, so every kept Gaussian holds its values bit for bit. Each `dropped_*` count is its stage's input size less its output size, and the three of them plus `kept_splats` add up to the source count.
    - _Requirements: 2.1, 2.17, 2.18, 10.5_

  - [ ] 6.5 Write the property test for Property 6 in `tests/test_clean_filters.py`, with Hypothesis
    - **Property 6: A dense, opaque grid loses nothing**
    - **Validates: Requirements 2.22**
    - This is the user's "it removes nothing from a dense grid of opaque, finite Gaussians" property.
    - Add the design's "Grids" strategy to `tests/splats.py`: `nx`, `ny` and `nz` from 1 to 6 with a product above 8, a spacing of `2**e` per axis for `e` from -8 to 8, an origin of whole multiples of the spacing, stored opacities at least 0.001 above the logit of `min_opacity`, random unit quaternions and any finite scales. It has no tie check, because a grid is full of ties (Req 10.13).
    - With the real `config.toml`'s `[clean]` settings, read by path and checked with `check_settings`, `run_filters` keeps every Gaussian: all three `dropped_*` counts are 0, whatever order the KD-tree puts the equally near Gaussians in.
    - _Requirements: 2.22, 10.5, 10.13_

  - [ ] 6.6 Write unit tests for the limits in `tests/test_clean_limits.py`
    - A count one over `max_source_splats` is refused, with both numbers and the note on more memory, and a count equal to it passes (Req 3.1).
    - The Removal_Cap's message holds the source count, each filter's drops, the Removed_Fraction, the cap, the note that broken Gaussians don't count and the settings to loosen (Req 3.2, 3.3). A share exactly at the cap, 7 of 10 at 0.7, passes. A splat where 90% are broken and nothing else is dropped passes with the cap at 0.7 (A4).
    - `min_kept_splats` one under, exactly and one over (Req 3.5). `warn_kept_fraction` the same way, with both counts and the Removed_Fraction in the warning (Req 3.6).
    - A splat that breaks both the cap and `min_kept_splats` gives the cap's message (Req 3.4, 3.7).
    - Every failure is an `ExpectedFailure`, and no message holds "Traceback" (Req 3.8).
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 10.4_

  - [ ] 6.7 Write the property test for Property 7 in `tests/test_clean_limits.py`, with Hypothesis
    - **Property 7: The Removal_Cap holds, and broken Gaussians don't count toward it**
    - **Validates: Requirements 2.23, 3.2, 3.4**
    - This is the user's "the step never removes more than 70%, with broken Gaussians outside the cap" property. The Config's `max_removed_fraction` is 0.70, and the settings strategy draws it often, beside the rest of 0 to 1.
    - Use task 3's "Splats" strategy, which also draws splats where the Broken_Filter takes more than `max_removed_fraction`.
    - `run_filters` then `check_limits` raises the Removal_Cap's `ExpectedFailure` exactly when the Gaussians that passed the Broken_Filter but were dropped later, counted by the test from Property 2's oracle and the kept rows, are more than `max_removed_fraction` of the source count, compared in float64 as `check_limits` does. When it doesn't raise the cap's failure, the share is at most `max_removed_fraction`, whatever else it refuses or warns about. Tell the cap's failure from the `min_kept_splats` one by its first sentence.
    - _Requirements: 2.23, 3.2, 3.4, 10.5_

  - [ ] 6.8 Run the tests and commit
    - Run `uv run pytest`. Every test passes, `tests/test_guard.py` included.
    - On `feat/clean`: `git add splatpipe/clean/filters.py splatpipe/clean/limits.py tests/splats.py tests/test_clean_filters.py tests/test_clean_limits.py`, then commit those files with the title "Clean step: filter order and removal limits" and the body "Adds filters.py, which runs the broken, faint and isolated filters in order on index arrays, so a kept Gaussian's values never change, and limits.py, which stops the step when the faint and isolated filters would take more than max_removed_fraction of the splat or leave fewer than min_kept_splats. Broken Gaussians don't count toward the cap, and a dense, even grid loses nothing."
    - _Requirements: 10.1, 10.2_

- [ ] 7. splat-transform export, the lockfile and its licences
  - Files: `package.json`, `package-lock.json`, `splatpipe/clean/export.py`, `tests/fake_splat_transform.py` and `tests/test_clean_export.py` (new), `docs/third-party-licences.md` and `tests/test_clean_repo.py` (changed).
  - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.6, 5.7, 5.8, 5.9, 5.10, 5.11, 5.12, 5.13, 5.14, 6.5, 6.12, 6.13, 8.7, 8.8, 9.6, 9.7, 10.1, 10.2, 10.4, 10.6, 10.7, 10.9_

  - [ ] 7.1 Pin splat-transform 3.8.0 in `package.json` and `package-lock.json`
    - Check that `node --version` is 22 or newer. If Node is missing or older, stop and tell the user, because splat-transform 3.8.0 needs Node 22.
    - Write `package.json` at the repo root as the design's "Supporting changes" gives it: `"private": true`, the name `voxelis-kiroween-tools`, `"license": "MIT"`, `"engines": {"node": ">=22"}` and the dependency `"@playcanvas/splat-transform": "3.8.0"`, with no `^` or `~`.
    - Run `npm install` once to make `package-lock.json`, then `npm ci`, which installs exactly the lockfile. The lockfile resolves `node_modules/@playcanvas/splat-transform` to 3.8.0, and `node_modules/@playcanvas/splat-transform/bin/cli.mjs` exists. `node_modules/` stays in `.gitignore`.
    - _Requirements: 6.12_

  - [ ] 7.2 Check every package in the lockfile against the licence rules, and update `docs/third-party-licences.md` for splat-transform, its packages, Node and npm
    - Read every entry of `package-lock.json`'s `packages` with its `license` field. Where the field is missing or unclear, read the package's own licence file in `node_modules/`. For `webgpu`, also read what its prebuilt Dawn native binaries bundle, from its licence and notice files.
    - Check each licence against the permissive list in `.kiro/steering/rules.md`, and the Dawn binaries against its rule for bundled libraries. If any package or bundled library is copyleft, non-commercial, has no clear licence, or has a licence that isn't on the list, stop and ask the user before doing anything else in this task.
    - Then, in the splat-transform row, replace the general "Its npm packages are MIT, BSD-3-Clause and ISC" with each package in the lockfile and its licence: splat-transform's own dependencies `webgpu` and `@adobe/spz`, its peer dependency `playcanvas`, which npm installs too, and every other package the lockfile holds. Note the Dawn native binaries on `webgpu`'s entry as bundled libraries. Give a package a row of its own when it doesn't fit in that cell.
    - Mark the splat-transform 3.8.0 row installed: its Installed cell says "Yes", and its name keeps the pinned 3.8.0.
    - Add a row named Node.js, worded like the other command line tools' rows, with Node's own licence from the LICENSE file of the installed Node, "Separate command line tool that runs splat-transform. Never vendored, linked or shipped." as how we use it, and "Yes, on the laptop. Not yet on the machine image." under Installed.
    - The npm row keeps its licence and its use. Its Installed cell becomes "Yes, on the laptop. Not yet on the machine image.", like Node's, because the clean step runs on the job machine too, so the machine image will need `npm ci` as well as Node.
    - _Requirements: 6.13, 9.7_

  - [ ] 7.3 Write `splatpipe/clean/export.py`
    - `NODE_VERSION`, `CLI`, `TAIL_CHARS`, `find_splat_transform()`, `splat_transform_argv()` and `export_sog()`, as in the design's "export.py" section.
    - `find_splat_transform` checks both `shutil.which("node")` and `(repo_root / CLI).is_file()` before it refuses. Its message says which is missing, or that both are, to run `npm ci` at the repo root, and that splat-transform needs Node `NODE_VERSION` or newer.
    - `splat_transform_argv` gives `[node, cli, "-w", "-g", "cpu", "-i", <sh_iterations>, "--max-workers", <max_workers>, <source>, "-H", <bands>, <target>]` in that order, with absolute paths and no other option.
    - `export_sog` runs that list with no shell, with `cwd` the Work_Folder, `stdin=subprocess.DEVNULL`, stderr merged into stdout as bytes, and `timeout=splat_transform_timeout_seconds`. It writes the whole output, decoded as UTF-8 with `errors="replace"`, to the log file first. Then it handles each outcome in the design's table: an `OSError`, a timeout, a non-zero exit with the out-of-memory hint for -9, and an exit of 0 with no file or an empty one are each an `ExpectedFailure`. When the file has bytes, it returns the size for `clean_sog_bytes`.
    - The tail in a Readable_Error is the last `TAIL_CHARS` characters, cut back to the start of a line, after the design's three clean-ups: `\r\n` becomes `\n`, each line keeps only what follows its last `\r`, and colour codes and every other control character but `\n` and `\t` go.
    - Log one INFO line with `logging.getLogger(__name__)` as splat-transform starts and one as it ends. On Linux only, the end line gives `resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss`. Import `resource` only there, because Windows has no such module.
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.6, 5.7, 5.8, 5.9, 5.10, 5.11, 5.12, 5.13, 5.14, 6.5, 8.7, 8.8_

  - [ ] 7.4 Write the stand-in for splat-transform in `tests/fake_splat_transform.py`
    - A Python script that a test runs in place of `cli.mjs`, by giving `export_sog` a `find` that returns `(sys.executable, <this script>)`.
    - It reads `FAKE_SPLAT_TRANSFORM` from its environment: `ok` writes a few bytes to its last argument, `fail` prints 3,000 characters of lines with `\r` progress redraws and colour codes and exits with 3, `silent` exits with 0 and writes nothing, `empty` writes a file of 0 bytes, and `hang` sleeps for 60 seconds.
    - _Requirements: 10.6_

  - [ ] 7.5 Write unit tests for the export in `tests/test_clean_export.py`
    - The argument list for each test Preset: `node` and the repo's `bin/cli.mjs` first, then `-w`, `-g cpu`, `-i` with the Preset's `sh_iterations` and `--max-workers` with the Shared_Setting, all before the input, then `-H` with the band count right after the input, and the `.sog` last. It holds none of `--filter-floaters`, `--filter-cluster`, `--decimate`, `-N`, `.voxel.json` or `.webp` (Req 5.2, 5.3, 5.4, 5.6, 5.7, 10.9).
    - A `find` with no `node` on the PATH, one whose repo root has no `node_modules`, and one with neither, each give the Req 5.10 message, naming what's missing, `npm ci` and Node 22.
    - With the stand-in: `fail` gives the exit code and a tail of at most `TAIL_CHARS` characters that starts on a line, with no `\r`, no colour codes and no "Traceback" (Req 5.11). `hang` with a 0.5-second timeout gives the limit in the message within a few seconds, and the child is gone afterwards (Req 5.12). `silent` and `empty` give the Req 5.13 message, and `ok` gives the file's size (Req 5.13, 5.14).
    - With a stand-in `run`: it's called with a list and without `shell=True` (Req 5.8), a return code of -9 gives the out-of-memory hint, and an `OSError` names the `node` path and the reason.
    - The whole output lands in the log file in every case.
    - _Requirements: 5.2, 5.3, 5.4, 5.6, 5.7, 5.8, 5.10, 5.11, 5.12, 5.13, 5.14, 10.4, 10.6, 10.7, 10.9_

  - [ ] 7.6 Write the real splat-transform test in `tests/test_clean_export.py`
    - Skip it unless `shutil.which("node")` finds Node and the repo has `bin/cli.mjs`, with a reason that says which is missing, so the whole suite passes on a machine with no Node (Req 10.6).
    - Write a splat of 64 Gaussians with 45 `f_rest_*` properties into `tmp_path` with `ply_bytes`, and run the real `export_sog` on it with `-H 1`, `-i 1`, `--max-workers 0` and a 60-second timeout (Req 10.7).
    - The `.sog` is a zip whose `meta.json` says `"version": 2` (Req 5.1, 9.6).
    - _Requirements: 5.1, 5.3, 9.6, 10.6, 10.7_

  - [ ] 7.7 Add the lockfile and licence-row checks to `tests/test_clean_repo.py`
    - `package.json` pins `@playcanvas/splat-transform` at exactly `3.8.0`, and `package-lock.json` resolves it to 3.8.0 (Req 6.12).
    - The name of each package in `package-lock.json`, apart from the root entry, appears in the Licence_List (Req 6.13).
    - With task 2.7's row lookup, the Node.js, splat-transform 3.8.0 and npm rows each start their Installed cell with "Yes", and the Node.js row says it's never vendored, linked or shipped (Req 9.7).
    - _Requirements: 6.12, 6.13, 9.7_

  - [ ] 7.8 Run the tests and commit
    - Run `uv run pytest`. Every test passes, `tests/test_guard.py` included. The real splat-transform test runs where Node is installed and is skipped, with its reason, where it isn't. Check that `package-lock.json` is under 1 MB.
    - On `feat/clean`: `git add package.json package-lock.json docs/third-party-licences.md splatpipe/clean/export.py tests/fake_splat_transform.py tests/test_clean_export.py tests/test_clean_repo.py`, then commit those files with the title "Clean step: splat-transform 3.8.0 export, with its lockfile and licences" and the body "Adds export.py, which runs splat-transform 3.8.0 under node, on the CPU and with no shell, to write the viewer copy into work/clean/, and turns each way it can fail into a readable error. package.json and package-lock.json pin 3.8.0 so npm ci installs the same tree everywhere, and the licence list names every package in that tree, each checked against our permissive list. The licence list also marks splat-transform 3.8.0 installed, gains a line for Node, which runs it, and notes that Node and npm aren't on the machine image yet. A stand-in script tests the failures on machines without Node."
    - _Requirements: 10.1, 10.2_

- [ ] 8. Running the step: `run()`, `work()` and the CLI
  - Files: `splatpipe/clean/__main__.py` and `tests/test_clean_step.py` (new), `splatpipe/clean/__init__.py`, `tests/splats.py` and `tests/test_clean_repo.py` (changed).
  - _Requirements: 1.1, 1.11, 3.1, 3.2, 3.5, 3.6, 3.7, 3.8, 4.5, 4.8, 4.9, 4.10, 4.11, 4.12, 4.13, 5.1, 5.5, 5.9, 5.14, 6.6, 6.9, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 8.1, 8.2, 8.3, 8.5, 8.6, 8.7, 8.8, 8.9, 10.1, 10.2, 10.3, 10.4, 10.6, 10.7, 10.8, 10.10, 10.11_

  - [ ] 8.1 Write `run()` and `work()` in `splatpipe/clean/__init__.py`
    - `run()`, `work(step)`, `clean_and_write()`, `Written`, `viewer_bands()`, `remove_outputs()` and `move_outputs()`, as in the design's "__init__.py (run and work)" and "The two-output guarantee" sections. Rewrite the package's docstring to say what it now holds.
    - `work` runs the four phases in order: `check_settings`, before anything in the Job_Folder is opened, written or removed; `remove_outputs`, the Viewer_Copy first; `clean_and_write`, then `gc.collect()` before `node` starts; and `export_sog` into the Work_Folder, then `move_outputs`. It does nothing after `move_outputs` returns.
    - `clean_and_write` reads `out/<train_ply>` with `read_header` and adds `source_splats`. It runs `check_source_count` before `read_splat`, and `run_filters` and `check_limits` before anything is written, then adds the warning, writes `work/clean/<clean_ply>` and adds `clean_ply_bytes`. Its stages are timed as `read_ply`, `clean` and `write_ply`, and `work` times the export as `export_sog`. It returns only a path and the SH_Bands, so none of its arrays outlive it.
    - `viewer_bands` gives `min(sog_sh_bands, the Source_Splat's SH_Bands)`, and adds the design's warning when that lowers them.
    - `remove_outputs` calls `unlink(missing_ok=True)` on the Viewer_Copy, then on the Clean_Splat. It turns an `OSError` into an `ExpectedFailure` that names the file and the reason and says to close the program that has it open.
    - `move_outputs` calls `os.replace` for the Clean_Splat, then for the Viewer_Copy. On any `BaseException` it removes whatever it had moved into `out/` and raises again, an `OSError` as an `ExpectedFailure` naming the file.
    - _Requirements: 1.1, 3.1, 3.2, 3.5, 3.7, 3.8, 4.5, 4.8, 4.9, 4.10, 4.11, 4.12, 4.13, 5.1, 5.5, 5.9, 5.14, 6.9, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 8.1, 8.6, 8.7, 8.8, 8.9_

  - [ ] 8.2 Write `splatpipe/clean/__main__.py`
    - `main(argv)`, which calls `step_main("clean", "splatpipe.clean", argv)`, and `if __name__ == "__main__": sys.exit(main())`, as the Fake_Step's does.
    - _Requirements: 8.1, 8.2_

  - [ ] 8.3 Add `put_source` to `tests/splats.py`
    - `put_source(job_id, data)` writes a splat as `out/train.ply` in a Job made with the `new_job` fixture.
    - _Requirements: 10.3_

  - [ ] 8.4 Write end-to-end tests for done and failed runs in `tests/test_clean_step.py`
    - Run the step through `splatpipe.clean.run` and `splatpipe.clean.__main__.main`, with `splatpipe.clean.export_sog` patched to call the real `export_sog` with task 7's stand-in.
    - A done run writes both Step_Outputs. Its report, at `out/reports/clean.json`, has exactly the counts and timings of Req 7.1, as `int`s and seconds, no `removed_percent`, and the eighteen settings (Req 5.14, 7.1, 7.3, 7.4, 7.6, 7.7).
    - The Req 3.6 warning, and the Req 5.5 warning from a splat with fewer SH_Bands than the Preset keeps, land in `warnings` (Req 3.6, 5.5, 7.5).
    - A failed report at each stage, from reading the splat to the moves, holds exactly the counts and timings worked out before the failure (Req 7.2). Every failure has `expected: true`, and its `readable` is the step's own message (Req 3.8, 8.8).
    - A count over `max_source_splats` fails before the rows are read, as a spy on `read_splat` shows (Req 3.1). A run that stops at the Removal_Cap, and one that leaves fewer than `min_kept_splats`, write neither Step_Output (Req 3.2, 3.5, 3.7).
    - A setting left out with `set_clean` fails with the Step_Runner's message (Req 6.6). A Job with no `out/train.ply` fails with the message that names the training step (Req 1.1).
    - `main(["--job", job_id])` prints `clean: done` and returns 0. When the step fails, it returns 1 and prints the readable error on stderr, with no "Traceback" (Req 8.2).
    - A subprocess runs `[sys.executable, "-m", "splatpipe.clean", "--job", "../x"]` from the repo root, with a 60-second timeout. It exits with 1 and prints the readable error with no "Traceback", which proves the module path (Req 8.2).
    - One done run uses the real splat-transform, skipped as in task 7 when Node or `bin/cli.mjs` is missing (Req 5.1, 10.6).
    - _Requirements: 1.1, 3.1, 3.2, 3.5, 3.6, 3.7, 3.8, 5.1, 5.5, 5.14, 6.6, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 8.2, 8.8, 10.4, 10.6, 10.7_

  - [ ] 8.5 Write end-to-end tests for the two Step_Outputs and the trained splat in `tests/test_clean_step.py`
    - A spy on `Path.unlink` shows the Viewer_Copy removed before the Clean_Splat. A removal that raises `PermissionError`, for each file in turn, names the file and leaves the files not yet removed (Req 4.8, 4.9, 4.13).
    - A lone `out/clean.ply`, as a hard stop leaves it, is gone before anything is written (Req 4.12).
    - A setting that `check_settings` refuses leaves an earlier pair in `out/` as it was (Req 4.10, 6.9).
    - With the stand-in set to `fail`, `out/` holds neither Step_Output, an earlier run's pair included, and a following done run writes both again (Req 4.8, 4.10, 10.10).
    - A second rename that raises, and a Ctrl+C raised by a stand-in `export_sog`, each leave neither Step_Output, and the Ctrl+C still stops the run (Req 4.10, 4.11, 10.10).
    - While the stand-in runs, `out/` doesn't hold the new Clean_Splat yet (Req 4.5, 5.9).
    - A second done run with another Preset replaces both Step_Outputs (Req 8.6).
    - `out/train.ply`'s SHA-256 is the same after a done run and after a run that stops at the Removal_Cap (Req 1.11, 10.8).
    - Listing the Job_Folder before and after a done run shows only `work/clean/`, the two Step_Outputs and the report changed (Req 8.7, 10.11).
    - _Requirements: 1.11, 4.5, 4.8, 4.9, 4.10, 4.11, 4.12, 4.13, 5.9, 6.9, 8.6, 8.7, 10.8, 10.10, 10.11_

  - [ ] 8.6 Write the memory test in `tests/test_clean_step.py`
    - As in the design's "The memory test" section: run the step through `run()` on 20,000 Gaussians on a jittered lattice, built with the bulk path of `ply_bytes`, every one a Faint_Gaussian, with `set_clean` values that put each in a Cube of its own (`faint_cube_divisions` at 10⁶) and make each Cube keep it (`faint_min_per_cube` at 1).
    - Start `tracemalloc` first, and patch `export_sog` with a stand-in that reads `tracemalloc.get_traced_memory()` when it's called.
    - At that moment the traced memory is back within 1 MB of where it started, so no array outlives the clean phase. The traced peak is at most 600 bytes per Gaussian above the start.
    - _Requirements: 8.9_

  - [ ] 8.7 Add the package checks to `tests/test_clean_repo.py`
    - `splatpipe.clean` has `work` and `run`, and `splatpipe.clean.__main__` has `main` (Req 8.1).
    - Task 1's import checks now read the whole package, and they still pass (Req 8.5).
    - _Requirements: 8.1, 8.5_

  - [ ] 8.8 Run the tests and commit
    - Run `uv run pytest`. Every test passes, `tests/test_guard.py` and `tests/test_run.py`'s `assert STEP_LIST == ()` included (Req 8.3).
    - On `feat/clean`: `git add splatpipe/clean/__init__.py splatpipe/clean/__main__.py tests/splats.py tests/test_clean_step.py tests/test_clean_repo.py`, then commit those files with the title "Clean step: run, work and the CLI" and the body "Joins the settings, the reader, the three filters, the limits, the writer and the export in splatpipe/clean/__init__.py, with python -m splatpipe.clean as its CLI. Both outputs are made in work/clean/ and move into out/ only once both are whole, the viewer copy removed first and moved last, so out/ never holds a viewer copy without the clean splat it came from. The step stays out of STEP_LIST until a spec with a step worth running adds the entries."
    - _Requirements: 8.3, 10.1, 10.2_

- [ ] 9. Pipeline doc and README
  - Files: `docs/pipeline.md`, `README.md` and `tests/test_clean_repo.py` (changed).
  - _Requirements: 8.3, 8.4, 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.8, 10.1, 10.2_

  - [ ] 9.1 Write the `## Step: clean` section of `docs/pipeline.md`
    - Put it after `## Step: fake`, with the parts the Fake_Step's section has: `### What it reads`, `### What it writes`, `### Config section`, `### Step report fields` and `### Files in out/`. Give the CLI, `uv run python -m splatpipe.clean --job <id> [--preset P]`, and the plain function, `splatpipe.clean.run(job_id, preset=None)`.
    - "What it reads" states the training step's contract from A1: a standard 3DGS Binary_PLY at `out/<train_ply>`, `binary_little_endian` and float32, with the properties A1 lists. It says that the step carries every other property through unchanged, a NaN or an infinity in a colour included, and never writes that file (Req 9.4).
    - "What it writes" lists the filters in their order, the Requirement 3 limits and the two Step_Outputs. It says that every part of the step is CPU work for the 8 vCPUs and 8 GB a Step gets on the job machine, and that the Viewer_Copy is SOG meta version 2 because splat-transform is pinned at 3.8.0, so the web viewer has to be read against the version 2 spec (Req 9.6). It says the step needs Node 22 or newer and `npm ci` at the repo root.
    - "Config section" lists the eighteen settings, with their tables, values and what each does. "Step report fields" lists the counts and timings of Req 7.1, the warnings of Req 3.6 and 5.5, and why there's no `removed_percent`.
    - "Files in out/" names both Step_Outputs and states the guarantees of the design's "The two-output guarantee": the step removes both before it writes either, the Viewer_Copy first; a failure after that removal leaves neither; a hard stop can leave a whole Clean_Splat alone, this run's between the two moves into `out/` or the earlier one between the two removals, with no done Clean_Report, and the next run removes it first; a Viewer_Copy never sits in `out/` without the Clean_Splat it was made from; and the pair is to be trusted only when `out/reports/clean.json` says done (Req 9.3).
    - In "The step list", add that the clean step isn't in `STEP_LIST` yet, that the first spec with a Step worth running adds the entries, and that the clean step's goes last, as an optional step (Req 8.3, 8.4, 9.5). Call the planned optional step "clean" in place of "clean and export".
    - _Requirements: 8.3, 8.4, 9.1, 9.3, 9.4, 9.5, 9.6_

  - [ ] 9.2 Add the clean step's line to `README.md`
    - Under "Install", after `uv sync`: the clean step also needs Node 22 or newer and `npm ci` at the repo root, which installs splat-transform 3.8.0 into `node_modules/`, and the tests that run it are skipped without them.
    - _Requirements: 9.8_

  - [ ] 9.3 Add the doc and README checks to `tests/test_clean_repo.py`
    - Read `docs/pipeline.md` the way `tests/test_docs.py` does, and leave that file as it is. The `## Step: clean` section has its five `###` parts, each found by its heading's name, because `STEP_LIST` is empty and `tests/test_docs.py` doesn't reach this section (Req 9.1, 9.2).
    - Its "Files in out/" part says that both outputs are removed first, that a failure after that leaves neither, that a hard stop can leave a whole clean splat alone, that a viewer copy never sits there without its clean splat, and to trust the pair only when `out/reports/clean.json` says done (Req 9.3).
    - The section names `train.ply` and `binary_little_endian` and says other properties are carried through unchanged (Req 9.4). It says SOG version 2 and 8 GB (Req 9.6).
    - "The step list" says that the clean step isn't in `STEP_LIST` yet (Req 9.5).
    - The README names `npm ci` and Node 22 (Req 9.8).
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.8_

  - [ ] 9.4 Run the tests and commit
    - Run `uv run pytest`. Every test passes, `tests/test_guard.py` and `tests/test_docs.py` included, which checks that the new section has a "Files in out/" part too.
    - Check with `git diff --name-only main -- tests/` that, of the files in `tests/`, this spec changed only `tests/conftest.py` and added only `tests/splats.py`, `tests/fake_splat_transform.py` and its own `tests/test_clean_*.py`, so no pipeline-core test changed (Req 8.3).
    - On `feat/clean`: `git add docs/pipeline.md README.md tests/test_clean_repo.py`, then commit those files with the title "Clean step: pipeline doc and README" and the body "Writes the clean step's section of docs/pipeline.md, with what it reads and writes, its eighteen settings, its report fields and the two-output guarantee, so the other step specs can build on it. The README says the step needs Node 22 and npm ci."
    - _Requirements: 8.3, 10.1, 10.2_

## Notes

- Every test sub-task is required, as `.kiro/steering/rules.md` asks.
- The last sub-task of each task is its checkpoint: the whole suite, then the commit. So the plan has no separate checkpoint tasks and stays at 9 tasks.
- Every acceptance criterion in `requirements.md` is named by at least one task.
- The user's three property tests are Property 7 in task 6.7 (the step never removes more than 70%, with broken Gaussians outside the cap), Property 1 in task 6.4 (it never changes the Gaussians it keeps, with Property 9 in task 2.6 carrying that through the file), and Property 6 in task 6.5 (a dense grid of opaque, finite Gaussians loses nothing). The bit-for-bit PLY round trip is Property 9, in task 2.6.
- The lockfile's licence check is task 7.2, before any code in task 7 is written.
- The graph runs the tasks in order, one at a time, so each task's test run and commit see only finished work. Inside a task, sub-tasks that write different files share a wave.

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "1.2"] },
    { "id": 1, "tasks": ["1.3"] },
    { "id": 2, "tasks": ["1.4", "1.6"] },
    { "id": 3, "tasks": ["1.5"] },
    { "id": 4, "tasks": ["1.7"] },
    { "id": 5, "tasks": ["2.1"] },
    { "id": 6, "tasks": ["2.2", "2.3", "2.4"] },
    { "id": 7, "tasks": ["2.5", "2.7"] },
    { "id": 8, "tasks": ["2.6"] },
    { "id": 9, "tasks": ["2.8"] },
    { "id": 10, "tasks": ["3.1", "3.2"] },
    { "id": 11, "tasks": ["3.3"] },
    { "id": 12, "tasks": ["3.4"] },
    { "id": 13, "tasks": ["3.5"] },
    { "id": 14, "tasks": ["3.6"] },
    { "id": 15, "tasks": ["4.1"] },
    { "id": 16, "tasks": ["4.2"] },
    { "id": 17, "tasks": ["4.3"] },
    { "id": 18, "tasks": ["4.4"] },
    { "id": 19, "tasks": ["4.5"] },
    { "id": 20, "tasks": ["5.1"] },
    { "id": 21, "tasks": ["5.2"] },
    { "id": 22, "tasks": ["5.3"] },
    { "id": 23, "tasks": ["5.4"] },
    { "id": 24, "tasks": ["6.1", "6.2"] },
    { "id": 25, "tasks": ["6.3", "6.6"] },
    { "id": 26, "tasks": ["6.4", "6.7"] },
    { "id": 27, "tasks": ["6.5"] },
    { "id": 28, "tasks": ["6.8"] },
    { "id": 29, "tasks": ["7.1"] },
    { "id": 30, "tasks": ["7.2"] },
    { "id": 31, "tasks": ["7.3", "7.4"] },
    { "id": 32, "tasks": ["7.5", "7.7"] },
    { "id": 33, "tasks": ["7.6"] },
    { "id": 34, "tasks": ["7.8"] },
    { "id": 35, "tasks": ["8.1", "8.2", "8.3"] },
    { "id": 36, "tasks": ["8.4", "8.7"] },
    { "id": 37, "tasks": ["8.5"] },
    { "id": 38, "tasks": ["8.6"] },
    { "id": 39, "tasks": ["8.8"] },
    { "id": 40, "tasks": ["9.1", "9.2"] },
    { "id": 41, "tasks": ["9.3"] },
    { "id": 42, "tasks": ["9.4"] }
  ]
}
```
