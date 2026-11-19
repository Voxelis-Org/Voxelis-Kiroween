# Implementation Plan: pipeline-core

## Overview

This plan builds the Skeleton of `splatpipe/`, the Spend_Guard and the Pipeline_Doc in the design's build order of 9 tasks: errors and JSON files, the Config_Loader, Job_Ids and the test config, the Video_Probe, the Job_Maker, the Step_Runner with the Fake_Step, the Job_Runner with the restart rule, the Spend_Guard, and last the Pipeline_Doc and the import checks. Each task writes its code with that code's unit and property tests, runs the whole suite and commits. The code is Python 3.11 with the standard library only. The tests use pytest and Hypothesis, which are already in the dev group.

Some test files grow over several tasks, as the design's build order lists, so each piece comes with the first task that uses it:

- `tests/conftest.py`: task 3 writes the temporary config and `dir_link`, task 5 adds `fake_probe` and `new_job`, and task 6 adds the helper that sets `fail` and `work_calls`.
- `tests/test_docs.py`: task 4 tests the ffmpeg row it changes (Req 13.6), and task 9 adds the other doc tests.
- `tests/test_probe.py`: task 4 runs the real `probe_video` on the clip, and task 5 adds the real `make_job` test, because `make_job` and `CONTAINER_FAMILY` only exist from task 5.

## Working rules for every task

- Work on `feat/pipeline-core`, never on `main`.
- `splatpipe/` and `cloud/spend.py` use the standard library only. Add no packages, so `pyproject.toml` and `uv.lock` stay as they are. The only step folder this spec makes is `splatpipe/fake/`.
- Every number and setting comes from `config.toml`, except the constants the design keeps in code on purpose: `JOB_ID`, `STEP_NAME`, `RESERVED_NAMES`, `PRESETS`, `NOT_PIPELINE`, `VIDEO_EXTENSIONS` and `CONTAINER_FAMILY`.
- Tests run on the CPU with no network and only tiny synthetic files. From task 3 on, every test gets the temporary config and Jobs_Folder from `tests/conftest.py`, and no test writes into the real `data/` folder. Any config a test builds itself keeps `jobs_folder` inside `tmp_path`.
- Subprocess tests pass `timeout=60`, set `PYTHONIOENCODING=utf-8` in the child's environment and read its output as UTF-8, so a hung child can't stall the suite and a Windows code page can't break the decoding.
- Property tests use Hypothesis with at least 100 examples (the default), and each docstring says `Feature: pipeline-core, Property N: <title>`. Properties 5 and 6 touch files on every example, so they use `@settings(deadline=None)` to stay steady on a slow disk.
- Every test sub-task is required. None is optional.
- Nothing here costs money. No task touches AWS, sets `APPROVED_SPEND` or runs anything paid.
- The last sub-task of each task is its checkpoint. Run `uv run pytest` and fix what fails until every test passes, `tests/test_guard.py` included. Then stage the task's files by name with `git add <file> ...`, never with `git add -A`, `git add .` or a folder, and commit only those files with `git commit -m "<message>" -- <file> ...`, so nothing else that's staged goes in. Leave `docs/devlog.md` out: the devlog hook adds and commits its own line after each task.
- If a hook blocks something it shouldn't, say so and fix the hook. Never work around it.

## Tasks

- [ ] 1. Errors and JSON files
  - Files: `splatpipe/errors.py`, `splatpipe/jsonio.py`, `tests/test_errors.py` and `tests/test_jsonio.py`, all new.
  - _Requirements: 6.7, 6.8, 6.9, 9.1, 9.2, 14.1, 14.6_

  - [ ] 1.1 Write `splatpipe/errors.py`
    - `PipelineError`, `ExpectedFailure` and `run_cli(action)`, as in the design's "errors.py" section.
    - `run_cli` returns the action's code. On a `PipelineError` it prints the message on stderr and returns 1. On any other `Exception` it prints the design's bug message, with the error's type and message, and returns 1. It never prints a traceback, and `SystemExit` and `KeyboardInterrupt` pass through.
    - _Requirements: 9.1, 9.2_

  - [ ] 1.2 Write `splatpipe/jsonio.py`
    - `NotJsonError`, `check_json_value()`, `write_json()` and `read_json()`, as in the design's "jsonio.py" section.
    - Accept only the exact types `dict` with `str` keys, `list`, `str`, `int`, finite `float`, `bool` and `None`. Each refusal names the field path, like `settings.sizes[2]`.
    - `write_json` checks the whole value before it writes, then writes ASCII JSON with `\n` line endings. `read_json` refuses `NaN`, `Infinity` and `-Infinity` with `NotJsonError`, through the `parse_constant` hook of `json.loads`.
    - Every refusal of a file's content is then a `ValueError`, because `UnicodeDecodeError`, `json.JSONDecodeError` and `NotJsonError` all are. Task 7's `read_earlier` relies on that.
    - _Requirements: 6.7, 6.8_

  - [ ] 1.3 Write unit tests for `run_cli` in `tests/test_errors.py`
    - An action that returns 0 gives 0 and prints nothing on stderr.
    - A `PipelineError` prints only its message on stderr and gives 1. A `ValueError` prints the bug message with `ValueError` and its message and gives 1. Neither output holds "Traceback".
    - An `ExpectedFailure` is a `PipelineError`. A `SystemExit(2)` from the action is raised again.
    - _Requirements: 9.1, 9.2_

  - [ ] 1.4 Write unit tests for the JSON files in `tests/test_jsonio.py`
    - `check_json_value` refuses a tuple, a set, an `int` key, NaN, infinity, minus infinity, an `IntEnum` and an `object()`, each with its field path in the message, nested ones too (`settings.a[1].b`).
    - `write_json` with a bad value raises before the file exists.
    - `read_json` refuses a file holding `NaN`, `Infinity` or `-Infinity` with `NotJsonError`.
    - _Requirements: 6.7, 6.8_

  - [ ] 1.5 Write the property test for Property 5 in `tests/test_jsonio.py`, with Hypothesis
    - **Property 5: Step_Reports survive the round trip**
    - **Validates: Requirements 6.7, 6.9**
    - Use the design's `json_values` and `any_text` strategies, so lone surrogates are covered, and fill each Step_Report field with values of its kind.
    - Compare strictly: the same types all the way down, the same key order and the same float signs. The file is ASCII, ends with `\n` and holds no `\r`.
    - Write into a `tempfile.TemporaryDirectory()` the test makes itself, because Hypothesis refuses `tmp_path` with `@given`.
    - _Requirements: 6.7, 6.9, 14.6_

  - [ ] 1.6 Run the tests and commit
    - Run `uv run pytest`. Every test passes, `tests/test_guard.py` included.
    - On `feat/pipeline-core`: `git add splatpipe/errors.py splatpipe/jsonio.py tests/test_errors.py tests/test_jsonio.py`, then commit those files with the message "Pipeline core: errors and JSON files".
    - _Requirements: 14.1_

- [ ] 2. Config loader and presets
  - Files: `splatpipe/config.py` and `tests/test_config.py` (new), `config.toml` (changed).
  - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 5.10, 5.11, 10.2, 14.1_

  - [ ] 2.1 Write `splatpipe/config.py`
    - `REPO_ROOT`, `CONFIG_PATH`, `PRESETS`, `NOT_PIPELINE`, `JOBS_SECTION`, `Config`, `load_config()`, `check_preset()`, `JobsSettings` and `StepSettings`, as in the design's "config.py" section.
    - `load_config()` with no path reads `CONFIG_PATH` at call time, so tests can point it at their own file. A missing file or a TOML error is a `PipelineError` that names the file.
    - `Config` refuses NaN, infinity and TOML dates and times in every Pipeline_Section, naming the section and the key.
    - `jobs()` checks the six `[jobs]` keys and joins a relative `jobs_folder` to `REPO_ROOT`. `step_settings()` runs the design's five layout checks in order and returns the shared and Preset settings together. A missing key in `StepSettings` names the section, the Preset table and the key.
    - `Config` keeps no list of Step_Sections. Only `step_settings` reads one, by name. So a section a later spec adds that isn't a Step_Section, like `[cloud]`, needs no code change and is still checked as a Pipeline_Section.
    - _Requirements: 5.4, 5.5, 5.6, 5.7, 5.8, 5.9_

  - [ ] 2.2 Add the `[jobs]` and `[fake]` sections to `config.toml`
    - Append `[jobs]`, `[fake]`, `[fake.fast]`, `[fake.default]` and `[fake.quality]` as the design's "Sections added to config.toml" gives them, with a comment above every key.
    - Leave `[git]`, `[aws]`, `[paid]` and `[devlog]` exactly as they are.
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 10.2_

  - [ ] 2.3 Write unit tests for the Config_Loader in `tests/test_config.py`
    - `step_settings` merges the shared and Preset settings.
    - Each broken layout is refused, naming the section and the key or table: a missing Preset table, a key in only some Preset tables, a key both shared and per-Preset, a non-Preset table like `[fake.other]`, and `fast = 1`.
    - NaN, `inf`, a date, a time and a datetime in a Pipeline_Section are refused by name. A date in `[devlog]` is not.
    - A config with a `[cloud]` table next to `[jobs]` and `[fake]` loads with no code change, and `nan` in `[cloud]` is refused by name.
    - A missing `[jobs]` key, a wrong type, a bool where a number belongs, a number that isn't above 0 and a missing Step setting read through `StepSettings` are each refused by name.
    - A bad Preset and a bad default Preset are refused, listing all three. A missing file and broken TOML are refused, naming the file.
    - A relative `jobs_folder` is joined to the repo root, also after `monkeypatch.chdir` to another folder.
    - _Requirements: 5.4, 5.5, 5.6, 5.7, 5.8, 5.9_

  - [ ] 2.4 Write the comment check and the real-config tests in `tests/test_config.py`
    - Write the comment-check helper from the design's "The config comment check" section, with `tomllib` as the judge.
    - Test the helper: `a = "x#y"` has no comment, `a = "x#y"  # real` has one, a comment two lines above doesn't count, a `["x"]` line inside a multi-line array isn't a header, a key under `[fake.fast]` counts as `[fake]`'s, and a key with no comment under `[cloud]`, a section that isn't a Step_Section, fails the check.
    - Read the real config by path, as `REPO_ROOT / "config.toml"`, so these tests still read it after task 3 points `CONFIG_PATH` at a temporary file. Check that `[jobs]` holds its six values, `[git]`, `[aws]`, `[paid]` and `[devlog]` equal their values today, every key in a Pipeline_Section has a comment, and `[fake]` loads with `step_settings("fake", preset)` for each of the three Presets. Task 9's Step_List check loads the other Step_Sections the same way.
    - _Requirements: 5.1, 5.2, 5.3, 5.10, 5.11, 10.2_

  - [ ] 2.5 Run the tests and commit
    - Run `uv run pytest`. Every test passes, `tests/test_guard.py` included.
    - On `feat/pipeline-core`: `git add splatpipe/config.py config.toml tests/test_config.py`, then commit those files with the message "Pipeline core: config loader and presets".
    - _Requirements: 14.1_

- [ ] 3. Job ids, job folders and the test config
  - Files: `splatpipe/jobs.py`, `tests/conftest.py` and `tests/test_jobs.py`, all new.
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 7.2, 7.3, 14.1, 14.2, 14.6_

  - [ ] 3.1 Write `splatpipe/jobs.py`
    - `JOB_ID`, `new_job_id()`, `check_job_id()`, `job_folder()`, `JobRecord`, `Job`, `open_job()`, `write_record()` and `remove_tree()`, as in the design's "jobs.py" section.
    - Keep the pattern `[0-9]{8}-[0-9]{6}-[0-9a-f]{6}` in code and check it with `fullmatch`, never with `\d` or `$`. A refusal gives the expected form and leaves the bad string out.
    - `new_job_id` builds the id from zero-padded fields, not `strftime("%Y")`, and refuses a naive time.
    - `job_folder` refuses unless the resolved Jobs_Folder is among the resolved Job_Folder's parents. `open_job` checks every Job_Record field. `remove_tree` clears a read-only flag and retries once.
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 7.2, 7.3_

  - [ ] 3.2 Write `tests/conftest.py` with the temporary config
    - An autouse fixture writes a test `config.toml` into `tmp_path` and points `splatpipe.config.CONFIG_PATH` at it with `monkeypatch`.
    - Its `jobs_folder` is `(tmp_path / "jobs").as_posix()`. Its `max_video_bytes` is tiny, like 1024, so a size test needs only a file one byte over. It has `[fake]`, `[fake_a]`, `[fake_b]` and `[fake_c]`, each with its three Preset tables and `fail = "none"`.
    - Add `dir_link(link, target)`, as in the design's "Fixtures" section: it tries `os.symlink(target, link, target_is_directory=True)`, falls back on Windows to a directory junction with `_winapi.CreateJunction(str(target), str(link))`, which needs no admin rights, and skips the test with the reason only when neither works.
    - _Requirements: 14.2_

  - [ ] 3.3 Write unit tests for job ids and job folders in `tests/test_jobs.py`
    - `load_config()` with no path gives the temporary config, and its Jobs_Folder is inside `tmp_path`.
    - `JOB_ID.pattern` is the pattern string. `new_job_id()` with no arguments passes `check_job_id`, and its time is within a few seconds of the UTC time now. A naive `now` raises `ValueError`. A refusal gives `YYYYMMDD-HHMMSS-xxxxxx` and not the bad string.
    - A Job_Folder that's a link to a folder outside the Jobs_Folder is refused. The test makes the link with `dir_link`, so it runs on Windows without admin rights.
    - `write_record` then `open_job` gives back the record. `open_job` refuses a missing folder, a missing or broken `job.json`, one holding `NaN`, one lacking a field, a wrong type or value, a `video_file` other than `video.mp4` or `video.mov`, and a `job_id` that differs from its folder's.
    - `remove_tree` removes a tree with a read-only file in it.
    - _Requirements: 1.1, 1.3, 1.4, 1.5, 1.6, 7.2, 7.3, 14.2_

  - [ ] 3.4 Write the property test for Property 1 in `tests/test_jobs.py`, with Hypothesis
    - **Property 1: The Job_Id_Check accepts exactly the pattern**
    - **Validates: Requirements 1.2, 1.3, 1.7**
    - Use the design's strategy: any text, valid ids, near misses made with `tricky`, and valid ids with a trailing newline, a `../` prefix, backslashes or upper case.
    - The oracle checks the length and each position by hand, without regular expressions.
    - _Requirements: 1.2, 1.3, 1.7, 14.6_

  - [ ] 3.5 Write the property test for Property 2 in `tests/test_jobs.py`, with Hypothesis
    - **Property 2: Every Job_Id the Job_Maker makes passes the check**
    - **Validates: Requirements 1.1, 1.8**
    - Draw UTC times over years 1 to 9999, and the random part as 3 random bytes in hex. The id passes the check, and its fields give back the time and the random part.
    - _Requirements: 1.1, 1.8, 14.6_

  - [ ] 3.6 Run the tests and commit
    - Run `uv run pytest`. Every test passes, `tests/test_guard.py` included.
    - On `feat/pipeline-core`: `git add splatpipe/jobs.py tests/conftest.py tests/test_jobs.py`, then commit those files with the message "Pipeline core: job ids, job folders and the test config".
    - _Requirements: 14.1_

- [ ] 4. Video probe, and ffprobe in the licence list
  - Files: `splatpipe/probe.py`, `tests/test_probe.py` and `tests/test_docs.py` (new), `docs/third-party-licences.md` (changed).
  - _Requirements: 2.6, 2.7, 2.9, 2.10, 2.11, 13.6, 13.7, 14.1, 14.3, 14.4, 14.5_

  - [ ] 4.1 Write `splatpipe/probe.py`
    - `ProbeResult`, `probe_video()` and `parse_probe()`, as in the design's "probe.py" section.
    - `probe_video` runs ffprobe as an argument list, never a shell string, with the given timeout, `stdin=subprocess.DEVNULL`, and `"file:" + str(path.resolve())` as the last argument. Each failure (not installed, a timeout, a non-zero exit, output it can't read) is a `PipelineError` with the design's message.
    - `parse_probe` is pure. It skips cover art, takes the length from the first video stream and falls back to the container's. A length counts only when it's a finite number above 0.
    - _Requirements: 2.6, 2.7, 2.9, 2.10, 2.11_

  - [ ] 4.2 Change the ffmpeg row of `docs/third-party-licences.md`
    - Rename the row "ffmpeg and ffprobe" and keep its licence cell.
    - Its "How we use it" cell says that ffprobe reads the video's container, streams and length when a Job is made, and that ffmpeg extracts the frames. It also says both are separate command line tools, on our private machine image and on laptops for the tests, never vendored or linked.
    - Set the Installed cell to "Yes, on the laptop. Not yet on the machine image." when `Get-Command ffmpeg, ffprobe` finds both. `Get-Command` only looks the tools up.
    - _Requirements: 13.6, 13.7_

  - [ ] 4.3 Write unit tests for the Video_Probe in `tests/test_probe.py`
    - With `subprocess.run` replaced through `monkeypatch`, for a file named `-clip.mp4`: the command is a list that starts `ffprobe -v error -print_format json -show_format -show_streams` and ends with `file:` plus the absolute path, and the given timeout is passed on.
    - ffprobe missing, a timeout, an exit of 1 and output that isn't JSON or has the wrong shape each give their own readable error. The exit-1 error gives ffprobe's first stderr line, even when it holds bytes that aren't UTF-8.
    - `parse_probe` takes the stream's duration, falls back to the container's for `"N/A"` or none, gives `None` for `"nan"`, `"inf"`, `"0"`, `"-1"` or nothing, uses the first video stream, and doesn't count cover art as video.
    - _Requirements: 2.6, 2.7, 2.9, 2.10, 2.11, 14.3_

  - [ ] 4.4 Write the real ffprobe test in `tests/test_probe.py`
    - Skip it, with the reason "ffmpeg and ffprobe aren't installed", when `shutil.which("ffmpeg")` or `shutil.which("ffprobe")` is `None`.
    - Make `tmp_path / "-clip.mp4"` with the design's ffmpeg command, as an argument list with a 60-second timeout.
    - `probe_video` gives the format name `mov,mp4,m4a,3gp,3g2,mj2`, a video stream and a length within 0.25 s of 4. Task 5 adds the `make_job` part.
    - _Requirements: 14.4, 14.5_

  - [ ] 4.5 Write the licence row test in `tests/test_docs.py`
    - The Licence_List has an "ffmpeg and ffprobe" row. Its "How we use it" cell says what ffprobe reads and that the tools are never vendored or linked.
    - _Requirements: 13.6_

  - [ ] 4.6 Run the tests and commit
    - Run `uv run pytest`. Every test passes, `tests/test_guard.py` included. The real ffprobe test runs where ffmpeg is installed and is skipped, with its reason, where it isn't.
    - On `feat/pipeline-core`: `git add splatpipe/probe.py docs/third-party-licences.md tests/test_probe.py tests/test_docs.py`, then commit those files in one commit with the message "Pipeline core: video probe, and ffprobe in the ffmpeg licence row".
    - _Requirements: 13.7, 14.1_

- [ ] 5. Job maker and trim
  - Files: `splatpipe/make.py` and `tests/test_make.py` (new), `tests/conftest.py` and `tests/test_probe.py` (changed).
  - _Requirements: 1.1, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 2.11, 2.12, 2.13, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 4.10, 9.1, 9.2, 14.1, 14.3, 14.4, 14.5, 14.6_

  - [ ] 5.1 Write `splatpipe/make.py`
    - `VIDEO_EXTENSIONS` and `CONTAINER_FAMILY` with the design's comment above them, `used_part()`, `make_job()` and `main()`, as in the design's "make.py" section.
    - `used_part` runs its four checks in the design's order and returns the given values unchanged.
    - `make_job` finishes every check, in the design's order, before it makes a folder or copies anything. It passes `[jobs].ffprobe_timeout_seconds` to the probe, and the size refusal gives the limit as `f"{limit / 2**30:g} GB"`. A Trim value must be an `int`, a `float` or `None`, never a `bool`, and one too big for a float is refused.
    - It then makes the Job_Folder with `exist_ok=False`, copies the video to `input/video.mp4` or `input/video.mov`, and writes the Job_Record. Once the folder exists, any failure removes it, and an `OSError` becomes a readable error about the copy and the free space.
    - `main` takes the video, `--start`, `--end` and `--preset`, uses `allow_abbrev=False` and has no `--job` option. It prints only the new Job_Id and returns 0. The file ends with `if __name__ == "__main__": sys.exit(main())`, so `python -m splatpipe.make` runs it.
    - _Requirements: 1.1, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 2.11, 2.12, 2.13, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 4.10, 9.1, 9.2_

  - [ ] 5.2 Add `fake_probe` and `new_job` to `tests/conftest.py`
    - `fake_probe(...)` gives a stand-in Video_Probe that returns a chosen `ProbeResult` and records its calls.
    - `new_job(...)` makes a Job from a few bytes named `clip.mp4`, with the fake probe and a fixed `new_id`, and returns its id.
    - _Requirements: 14.3_

  - [ ] 5.3 Write unit tests for the checks in `tests/test_make.py`
    - Each failing check leaves the Jobs_Folder with no new folder: a bad Preset, a missing path, a folder, `.mkv`, `.mp4.txt`, a file over the limit, the wrong format name, no video stream, no length, over 600 s, a Trim start after the end, a `bool` or a huge `int` as a Trim value, and a probe that raises a `PipelineError`.
    - `.MP4` and `.Mov` pass the extension check. The size refusal gives the configured limit. Check the "4 GB" text for a limit of 4 294 967 296 bytes without writing a big file: with `monkeypatch`, wrap `Path.stat` so that only the video's result gets a bigger `st_size`, and call the real `Path.stat` for every other path. Never patch `os.stat` itself, because `is_file`, `tmp_path` and pytest use it too.
    - The probe isn't called when the extension or the size fails. It's called with the resolved path and the configured timeout. The format name must match exactly.
    - Over the length limit, the message gives both numbers. The Trim messages give the Video_Length and the minimum, and a 2-second video with no Trim is refused.
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 2.11, 2.12, 2.13, 3.5, 3.6, 3.7, 4.6, 14.3_

  - [ ] 5.4 Write the property test for Property 3 in `tests/test_make.py`, with Hypothesis
    - **Property 3: The Trim accepts exactly the valid Used_Parts**
    - **Validates: Requirements 3.4, 3.5, 3.6, 3.7, 3.8**
    - Use the design's strategy: the start and the end are each missing or any float, NaN, infinities and negatives included, drawn with `st.data()` after the length. The oracle is the design's four-part rule.
    - _Requirements: 3.4, 3.5, 3.6, 3.7, 3.8, 14.6_

  - [ ] 5.5 Write the property test for Property 4 in `tests/test_make.py`, with Hypothesis
    - **Property 4: An accepted Trim gives back its own values**
    - **Validates: Requirements 3.1, 3.2, 3.3, 3.9**
    - Use the design's composite strategy, which draws mostly valid Trims, with `assume` on Property 3's rule.
    - Before drawing the end, call `assume(start + minimum <= length)`, with a missing start counted as 0. Rounding can push `start + minimum` just above `length`, and `st.floats(min_value=..., max_value=...)` raises `InvalidArgument` when the minimum is above the maximum, which would fail the test at random.
    - _Requirements: 3.1, 3.2, 3.3, 3.9, 14.6_

  - [ ] 5.6 Write unit tests for making the Job and for the CLI in `tests/test_make.py`
    - A good Job has `input/`, `work/` and `out/`, also when the Jobs_Folder didn't exist yet. A `.MOV` is copied to `input/video.mov`, and the original's bytes and modified time don't change.
    - The Job_Record holds every field, with the chosen Preset or, when none is chosen, `[jobs].default_preset`. Without `new_id`, the id comes from `new_job_id` and passes `check_job_id`.
    - A Job_Folder that already exists for a fixed `new_id` is refused and stays unchanged. A `shutil.copyfile` or `write_record` patched to raise `OSError`, like a full disk, leaves no Job_Folder and gives a readable error.
    - `main([video])` prints only the id and returns 0. `main(["--job", id, video])` exits with 2 and makes no Job. A refused video prints the readable error on stderr, with no "Traceback", and returns 1.
    - A subprocess runs `[sys.executable, "-m", "splatpipe.make", str(tmp_path / "missing.mp4")]` from the repo root. It exits with 1 and prints the readable error on stderr, with no "Traceback". It reads the real `config.toml` but makes nothing, because it stops at the missing file.
    - `make_job`'s default probe is bound when the function is defined, so for the `main` tests patch `splatpipe.make.make_job` with a `functools.partial` that adds the fake probe and a fixed `new_id`.
    - _Requirements: 1.1, 4.1, 4.2, 4.3, 4.4, 4.5, 4.7, 4.8, 4.9, 4.10, 9.1, 9.2, 14.3_

  - [ ] 5.7 Add the real `make_job` test to `tests/test_probe.py`
    - In the real ffprobe test from task 4, compare the format name with `CONTAINER_FAMILY`.
    - Add a second real test, skipped the same way: `make_job` with the real probe makes a Job from the clip. Pass it a config built from the test config with only `max_video_bytes` raised above the clip's size, because the test config's limit is tiny. Never build it from the real `config.toml`, whose `jobs_folder` is the real `data/jobs`.
    - _Requirements: 14.4, 14.5_

  - [ ] 5.8 Run the tests and commit
    - Run `uv run pytest`. Every test passes, `tests/test_guard.py` included.
    - On `feat/pipeline-core`: `git add splatpipe/make.py tests/conftest.py tests/test_make.py tests/test_probe.py`, then commit those files with the message "Pipeline core: job maker and trim".
    - _Requirements: 14.1_

- [ ] 6. Step runner and fake step
  - Files: `splatpipe/step.py`, `splatpipe/fake/__init__.py`, `splatpipe/fake/__main__.py` and `tests/test_step.py` (new), `tests/conftest.py` (changed).
  - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 7.9, 7.10, 9.1, 9.2, 9.3, 10.1, 10.3, 10.4, 10.5, 10.6, 14.1_

  - [ ] 6.1 Write `splatpipe/step.py`
    - `STEP_NAME`, `RESERVED_NAMES`, `check_step_name()`, `StepContext`, `run_step()` and `step_main()`, as in the design's "step.py" section.
    - `run_step` opens the Job first: it checks the Step_Name, loads the config, runs `open_job`, and takes the given Preset or the Job_Record's. A failure here raises and writes no report.
    - Then it removes the old report, loads the settings, checks that the Work_Folder is inside the Job_Folder, empties it, imports the module and runs `work`. A `PipelineError` gives its own readable message, and any other `Exception` gives the design's general message. It writes the report with `write_json` and returns it. The report's `settings` is `dict(settings)`, a plain copy, because `check_json_value` accepts only the exact type `dict` and would refuse `StepSettings`. `KeyboardInterrupt` and `SystemExit` aren't caught.
    - `run_step` never reads the Step's earlier report, so it runs every Step it's given, a Finished_Step included. In `out/` it touches only `out/reports/<name>.json`. It leaves the Step's other files there for the Step's own code to replace or remove.
    - The `add_*` methods refuse a bad name or value at once with `NotJsonError`, naming the field.
    - `step_main` parses `--job` and `--preset` inside `run_cli`. It prints `<name>: done` and returns 0, or prints the readable error on stderr and returns 1.
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 7.9, 7.10, 9.1, 9.2, 9.3_

  - [ ] 6.2 Write the Fake_Step in `splatpipe/fake/`
    - `__init__.py` has `work(step)` and `run(job_id, preset=None, *, config=None)`, as in the design's "fake/" section.
    - `__main__.py` has `main(argv)`, which calls `step_main("fake", "splatpipe.fake", argv)`, and ends with `if __name__ == "__main__": sys.exit(main())`.
    - `work` adds its counts, timing and warning before it checks `fail`, so a failed report still has them. It takes every path from `step`, so it runs the same as `fake`, `fake_a` or `fake_b`.
    - After the `fail` check, `work` writes `out/<name>.txt` as one UTF-8 line ending in `\n`, replacing the one from an earlier run. A failed run writes no result file and leaves an earlier one in place.
    - _Requirements: 7.1, 7.10, 10.1, 10.3, 10.4, 10.5, 10.6_

  - [ ] 6.3 Add the `fail` helper and `work_calls` to `tests/conftest.py`
    - The `fail` helper rewrites the temporary config with a different `fail` for chosen Step_Names, for example `fake_b="expected"`.
    - `work_calls` patches `splatpipe.fake.work` with `monkeypatch`, to a wrapper that records each call's Step_Name and then runs the real `work`. It returns the list of names. `run_step` looks `work` up when it runs, so the wrapper sees every run, through the Job_Runner or a Step's own function and CLI.
    - _Requirements: 7.9, 10.5, 10.6_

  - [ ] 6.4 Write unit tests for Step_Reports and the Fake_Step in `tests/test_step.py`
    - `splatpipe.fake.run(job_id)` writes `out/reports/fake.json` with every field and returns the same report. The Step writes the Preset's number of files into `work/fake/` and writes `out/fake.txt`. The report has the counts `files_written` and `video_bytes`, a timing and the warning.
    - `fail = "expected"` gives the Step's own message and `expected: true`. `fail = "unexpected"` gives the general message naming the Step, and the `RuntimeError` in `technical`. Both keep the timings, counts and warning added before the failure.
    - With `splatpipe.fake.work` patched to call `add_count(..., 1.0)`, the Step fails with `counts.<name>` in the technical message. Bad values for `add_timing` and `add_warning` fail it the same way.
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8, 7.1, 10.1, 10.3, 10.4, 10.5, 10.6_

  - [ ] 6.5 Write unit tests for folders, Presets and the CLI in `tests/test_step.py`
    - `check_step_name` refuses `""`, `Fake`, `../x`, `a-b`, `1x`, `a\n` and every reserved name. `RESERVED_NAMES` holds every `splatpipe/*.py` stem except `__init__`, plus `NOT_PIPELINE` and `jobs`. `run_step` with a refused name raises a `PipelineError` and writes no report.
    - `--preset quality` overrides the Job_Record's Preset and is recorded in the report.
    - A stale report and a stale read-only file in the Work_Folder are gone after a run. `input/`, `work/other/` and `out/reports/other.json` stay byte for byte the same.
    - A Work_Folder linked with `dir_link` to a folder outside the Job_Folder fails the Step, and the target keeps its files.
    - A bad Job_Id or a missing Job_Folder raises a `PipelineError` and writes no report.
    - `splatpipe.fake.__main__.main([...])` returns 0 when the Step is done and 1 when it failed, with the readable error on stderr and no "Traceback".
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 9.1, 9.2, 9.3_

  - [ ] 6.6 Write unit tests for running a Step again in `tests/test_step.py`
    - On a Job whose `fake` report already says `done`, a second `splatpipe.fake.run(job_id)` and a second `splatpipe.fake.__main__.main(["--job", job_id])` both run `work` again, as `work_calls` shows.
    - A done run replaces an earlier `out/fake.txt` with its own line.
    - A run with `fail = "expected"` leaves an earlier `out/fake.txt` and an earlier `out/fake-old.bin` byte for byte the same, while `out/reports/fake.json` becomes the new failed report.
    - _Requirements: 7.9, 7.10_

  - [ ] 6.7 Run the tests and commit
    - Run `uv run pytest`. Every test passes, `tests/test_guard.py` included.
    - On `feat/pipeline-core`: `git add splatpipe/step.py splatpipe/fake/__init__.py splatpipe/fake/__main__.py tests/conftest.py tests/test_step.py`, then commit those files with the message "Pipeline core: step runner and fake step".
    - _Requirements: 14.1_

- [ ] 7. Job runner and the restart rule
  - Files: `splatpipe/run.py` and `tests/test_run.py`, both new.
  - _Requirements: 7.1, 7.8, 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8, 8.9, 8.10, 8.11, 8.12, 8.13, 8.14, 8.15, 8.16, 8.17, 8.18, 8.19, 8.20, 9.1, 9.2, 9.4, 9.5, 10.7, 12.2, 12.4, 14.1, 14.6, 14.7_

  - [ ] 7.1 Write `splatpipe/run.py`
    - `StepEntry`, `STEP_LIST = ()`, `Earlier`, `check_step_list()`, `read_earlier()`, `first_to_run()`, `run_from()`, `job_status()`, `run_job(..., fresh=False)` and `main()`, as in the design's "run.py" section. The Step_List holds module names as strings, and `run.py` imports no step package.
    - `read_earlier` is the only part that reads a file, and it never raises because of what it finds. No file gives `missing`. Any other `OSError`, a `ValueError`, a `RecursionError`, or a value that isn't a `dict` with this Step's `step`, this Job's `job_id` and an `outcome` of `"done"` or `"failed"` gives `unreadable`. For `"failed"` it keeps `error.readable` when that's a string.
    - `first_to_run` is pure. With `fresh` it returns 0. Without it, it returns the index of the first entry that isn't a Finished_Step, or `len(entries)` when every entry is one. Whether a Step is required comes from its `StepEntry`, never from the report, and the two lists are paired with `zip(..., strict=True)`.
    - `run_from` touches no files. Each entry before `start` gets its earlier outcome and readable error with `reused: true`, and `run_one` isn't called for it. From `start` on, `run_one` runs each entry, with `reused: false`, and a failed Optional_Step doesn't stop the loop. After a failed Required_Step, every later entry is `not_run`.
    - `job_status` gives `failed` when a Required_Step's outcome is `failed`, `done_with_warnings` when none is and some Step's is, a reused one included, and `done` otherwise.
    - `run_job` refuses an empty list, a bad or reserved Step_Name, a name listed twice and a module outside `splatpipe`, all before it writes anything. Then it loads the config once, opens the Job, reads each listed Step's earlier report and picks `start`. Before any Step runs, it removes `out/job.json` and the reports of `entries[start:]`, and leaves the reports of the skipped Steps and of unlisted Steps alone.
    - Each Step from `start` on runs through `run_step` with the Job_Record's Preset, so a skipped Step's module is never imported. `run_job` writes `out/job.json` with `fresh`, each Step's `reused`, and `failed_step` and `readable_error` when a Required_Step failed, and returns it. When every Step had already finished, no Step runs and the summary comes from the earlier reports.
    - The Job_Log is a handler on the `splatpipe` logger that writes `out/job.log` in UTC. `run_job` opens the file itself, with mode `"a"` without `fresh` and `"w"` with it. A `finally` removes and closes the handler, closes the file and puts the logger's old level back.
    - The run's first line gives the Job_Id, the number of Steps and the Preset. When `start > 0`, the next line names the skipped Steps with their earlier outcomes. Each Step that runs gets a start line and an outcome line, a failed Optional_Step's with its readable error, and each `not_run` Step gets a "not run" line.
    - `main` parses `--job` and `--fresh` inside `run_cli`, calls `run_job(job_id, fresh=args.fresh)` and prints `Job <id>: <status in words>`. When every Step was reused, the line adds that nothing ran and that `--fresh` runs every Step again. It returns 0 for `done` and `done_with_warnings`. A Job where every Step had already finished is always one of those two, because a failed Required_Step is never a Finished_Step. For `failed` it prints the failed Step's readable error on stderr and returns 1. The file ends with `if __name__ == "__main__": sys.exit(main())`.
    - _Requirements: 7.8, 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8, 8.9, 8.10, 8.11, 8.12, 8.13, 8.14, 8.15, 8.16, 8.17, 8.18, 8.19, 9.1, 9.2, 9.4, 9.5, 12.2, 12.4_

  - [ ] 7.2 Write end-to-end tests with Fake_Steps in `tests/test_run.py`
    - Each test lists `StepEntry` items that run `splatpipe.fake` as `fake_a`, `fake_b` and `fake_c`, and sets `fail` with the conftest helper.
    - All done gives `done`. `fake_a` optional and failing, then `fake_b` done, gives `done_with_warnings` and the log line for `fake_a`.
    - `fake_a` done, `fake_b` required and failing, then `fake_c`, gives `failed`, `fake_c` as `not_run` with no report, and the summary's `failed_step` and `readable_error`.
    - The reports show the Steps ran in order with the Job_Record's Preset. `job.log` has a start line and an outcome line for each Step that ran, and a "not run" line for each Step that didn't.
    - A first run's summary has `fresh: false` and every `reused` false.
    - _Requirements: 8.4, 8.6, 8.7, 8.8, 8.12, 8.16, 10.7, 14.7_

  - [ ] 7.3 Write unit tests for the Job_Runner's checks, imports, log and CLI in `tests/test_run.py`
    - `StepEntry` holds `name`, `module` and `required`. `STEP_LIST` is empty and is the default list.
    - An empty list is refused before `out/` changes. A duplicate name is refused before any report exists. A module outside `splatpipe` is refused.
    - A spy on `importlib.import_module` in `splatpipe.step` shows that each module is imported right before its Step runs.
    - After 20 runs in one process, half of them with `fresh`, `tmp_path` can be deleted, the `splatpipe` logger has no handler left, and its old level is back.
    - With the empty default list, `run.main` returns 1 and prints the "No steps are set up yet" error. With `splatpipe.run.STEP_LIST` patched through `monkeypatch`, it returns 0 for `done` and `done_with_warnings`, and 1 for `failed`, with the readable error on stderr.
    - A bad or reserved Step_Name in the list is refused before any report exists.
    - Subprocesses run `[sys.executable, "-m", "splatpipe.fake", "--job", "../x"]` and `[sys.executable, "-m", "splatpipe.run", "--job", "../x"]` from the repo root. Each exits with 1 and prints the readable error on stderr, with no "Traceback". Neither writes anything: `splatpipe.fake` reads the real `config.toml` but refuses the id before any path is used, and `splatpipe.run` refuses the empty Step_List before it reads anything.
    - _Requirements: 7.1, 7.8, 8.1, 8.2, 8.3, 8.5, 8.10, 8.11, 9.1, 9.2, 9.4, 12.2_

  - [ ] 7.4 Write the property test for Property 7 in `tests/test_run.py`, with Hypothesis
    - **Property 7: A run picks up at the first unfinished Step**
    - **Validates: Requirements 8.6, 8.8, 8.13, 8.14, 8.15, 8.16, 8.17, 8.20**
    - Use the design's `plans` strategy: lists of 1 to 8 tuples of whether the Step is required, its earlier state (`missing`, `unreadable`, `done` or `failed`) and its outcome if it runs (`done` or `failed`). Draw `fresh` with `st.booleans()`.
    - Step `i` is `StepEntry(f"s{i}", "splatpipe.fake", required)`, with `Earlier(state, "earlier error" if state == "failed" else None)`. The stand-in `run_one` records each call and returns the drawn outcome, with `"new error"` when it's `failed`, so the test can tell an earlier readable error from a new one.
    - Check what `first_to_run` returns, which Steps `run_one` ran and in what order, each Step's `outcome`, `readable_error` and `reused`, and what `job_status` gives. The oracle works all of these out from the drawn tuples with its own code and never calls `first_to_run`, `run_from` or `job_status`.
    - The test touches no files and no config, so it needs no temporary folder.
    - _Requirements: 8.6, 8.8, 8.13, 8.14, 8.15, 8.16, 8.17, 8.20, 14.6_

  - [ ] 7.5 Write the restart tests in `tests/test_run.py`
    - Use `fake_a`, `fake_b` and `fake_c`, the `fail` helper and `work_calls`, as the design's "The restart tests" section gives them. `main` always runs `STEP_LIST`, which is empty, so a case that calls `main` first sets `splatpipe.run.STEP_LIST` to the case's list with `monkeypatch`.
    - Picking up after a failed Required_Step: `fake_a` is done, `fake_b` is required and fails, and `fake_c` isn't run. With `fake_b` fixed, a run without `fresh` runs `work` for `fake_b` and `fake_c` only. `fake_a`'s report and Work_Folder stay byte for byte the same, and only `fake_a` has `reused: true`.
    - A failed Optional_Step isn't run again: `fake_a` is done, `fake_b` is optional and fails, and `fake_c` is required and fails. With only `fake_c` fixed, the next run runs `work` for `fake_c` only, and the status is `done_with_warnings`.
    - Fresh: after a done run, `run_job(..., fresh=True)` and `main(["--job", job_id, "--fresh"])` each run `work` for every Step. Every `reused` is false, and the summary has `fresh: true`.
    - What's removed at the start: all three Steps are done. Then `fake_b`, a Required_Step, is run by hand with `run_step` and `fail = "expected"`, and an unlisted `out/reports/other.json` is added. In the next run, without `fresh` and with `fake_b` still set to fail, a wrapper around `work` checks, as `fake_b`'s `work` starts, that `out/job.json` and `out/reports/fake_c.json` are gone and `out/reports/fake_a.json` is still there. After the run, `fake_c` is `not_run` with no report, and `fake_a.json` and `other.json` stay byte for byte the same.
    - Everything finished: after a `done_with_warnings` run, with `fake_a` optional and failing and `fake_b` and `fake_c` done, `main(["--job", job_id])` runs no `work`, says that nothing ran and returns 0. Every `reused` is true, `fake_a` keeps its readable error, and the status is still `done_with_warnings`.
    - Reports that don't count: `fake_a` is done, then its report is replaced by text that isn't JSON, a file holding `NaN`, a JSON array, a report whose `step` is `fake_b`, one whose `job_id` is another Job's, or one whose `outcome` is `"not_run"`. Each time, the next run starts at `fake_a`. `read_earlier` gives `missing` with no file, and `unreadable` for each of these and for a folder in the report's place, without raising.
    - A Finished_Step after an unfinished one: all three Steps are done, then `fake_b`'s report is removed. A run without `fresh` skips `fake_a` and runs `fake_b` and `fake_c`, though `fake_c` had finished.
    - The Job_Log: after two runs without `fresh`, `job.log` holds both runs' lines in order, and the second run's lines include the one that names the skipped Steps. After a run with `fresh`, it holds only that run's lines.
    - `run_job` follows the decision: on a Job whose earlier reports mix the four states over required and optional Steps, the test works out `start` with `first_to_run` from `read_earlier` before the run. Then `work_calls` starts at `entries[start]`, and only `entries[:start]` have `reused: true`, with and without `fresh`.
    - _Requirements: 8.8, 8.9, 8.13, 8.14, 8.15, 8.16, 8.17, 8.18, 8.19, 8.20, 9.5_

  - [ ] 7.6 Run the tests and commit
    - Run `uv run pytest`. Every test passes, `tests/test_guard.py` included.
    - On `feat/pipeline-core`: `git add splatpipe/run.py tests/test_run.py`, then commit those files with the message "Pipeline core: job runner and the restart rule".
    - _Requirements: 14.1_

- [ ] 8. Spend guard
  - Files: `cloud/spend.py` and `tests/test_spend.py`, both new.
  - _Requirements: 11.1, 11.2, 11.3, 11.4, 11.5, 11.6, 11.7, 11.8, 14.1, 14.6_

  - [ ] 8.1 Write `cloud/spend.py`
    - `SpendRefused` and `check_spend(*, machine, expected_time, rough_cost, environ=None, config_path=None)`, as in the design's "cloud/spend.py" section. It uses the standard library only and never imports `splatpipe`.
    - It reads the variable's name from `[aws].spend_variable` and only reads `environ`, with `.get`. It refuses when the config can't be read, the name is missing or empty, a description part is empty, or the value isn't exactly `"1"`. The refusal gives the design's message.
    - _Requirements: 11.1, 11.2, 11.3, 11.4, 11.5, 11.6, 11.7_

  - [ ] 8.2 Write unit tests for the Spend_Guard in `tests/test_spend.py`
    - The refusal message has the machine, the time, the cost, the variable's name and what to do.
    - A temporary config naming `SPLATPIPE_TEST_SPEND` is honoured. The test sets that variable to "1" with `monkeypatch.setenv` and never sets `APPROVED_SPEND`.
    - Every call that leaves `environ` as `None` uses that temporary config, with `SPLATPIPE_TEST_SPEND` set or removed through `monkeypatch`, so an `APPROVED_SPEND` left in a developer's shell can't change a result.
    - A missing file, broken TOML, no `[aws]`, no `spend_variable`, an empty name and an empty description part are each refused.
    - `os.environ` is the same before and after allowed and refused calls, and a `MappingProxyType` passed as `environ` is never written.
    - _Requirements: 11.2, 11.3, 11.5, 11.6, 11.7_

  - [ ] 8.3 Write the property test for Property 6 in `tests/test_spend.py`, with Hypothesis
    - **Property 6: The Spend_Guard allows only exactly "1"**
    - **Validates: Requirements 11.4, 11.5, 11.8**
    - Use the design's strategy: no value, any text, and near misses such as `" 1"`, `"1\n"`, `"01"`, `"1.0"`, fullwidth 1 and Arabic-Indic 1.
    - It reads the real `config.toml` and passes its own dict as `environ`. It never touches `os.environ`, so `APPROVED_SPEND` is never set.
    - _Requirements: 11.4, 11.5, 11.8, 14.6_

  - [ ] 8.4 Run the tests and commit
    - Run `uv run pytest`. Every test passes, `tests/test_guard.py` included.
    - On `feat/pipeline-core`: `git add cloud/spend.py tests/test_spend.py`, then commit those files with the message "Pipeline core: spend guard".
    - _Requirements: 14.1_

- [ ] 9. Pipeline doc, structure steering and import checks
  - Files: `docs/pipeline.md` and `tests/test_imports.py` (new), `.kiro/steering/structure.md` and `tests/test_docs.py` (changed).
  - _Requirements: 5.11, 8.2, 11.1, 12.1, 12.2, 12.3, 12.4, 12.5, 13.1, 13.2, 13.3, 13.4, 13.5, 13.8, 13.9, 14.1_

  - [ ] 9.1 Write `docs/pipeline.md`
    - Start with the Skeleton, as the design's "Docs" section lists it: the Job_Folder layout, the Job_Record fields, the Step_Report format, the Presets and the Config layout, the Step_List with the planned Required_Steps and Optional_Steps, the Job_Runner, the Job_Summary, the Job_Log, and the CLIs with their exit codes, `--fresh` included.
    - Add a part headed "Running a job again". It says which Steps a run without `--fresh` skips and which it runs, that before any Step runs it removes the summary and the reports of the Steps it's about to run, that the skipped Steps show `"reused": true` and count toward the Job_Status, and that a Job where every Step has finished runs nothing and exits with 0. It says how `job.log` grows without `--fresh` and starts again with it, and that a single Step's CLI always runs its Step. It says to use `--fresh` after changing settings in `config.toml` or a Step's code, or after running an earlier Step by hand, because the Job_Runner doesn't check whether a Finished_Step's inputs changed.
    - Add a part headed "A Step's files in out/". It says the Step_Runner removes only a Step's own report and empties only its own Work_Folder. It leaves the Step's other files in `out/` alone, so each Step's own code replaces or removes what it wrote there in an earlier run.
    - Then write `## Step: fake`: what it reads, what it writes, its Config section and its Step_Report fields, and a part headed "Files in out/" that lists `out/fake.txt`, which a done run replaces and a failed run leaves in place. Say that the step specs copy the Fake_Step as their working example.
    - _Requirements: 13.1, 13.2, 13.3, 13.8, 13.9_

  - [ ] 9.2 Change `.kiro/steering/structure.md`
    - The `docs/` line lists `pipeline.md`.
    - Add a line saying that every step spec updates its own section of `docs/pipeline.md` as part of its tasks, so the Steps the three of us build keep fitting together.
    - _Requirements: 13.4, 13.5_

  - [ ] 9.3 Add the doc tests to `tests/test_docs.py`
    - `docs/pipeline.md` has the Skeleton's headings, and a `## Step: <name>` section for `fake` and for every entry in `STEP_LIST`. The `fake` section says that the step specs copy it.
    - The part headed "Running a job again" names `--fresh`, `config.toml` and running a Step by hand.
    - The part headed "A Step's files in out/" exists, each `## Step: <name>` section has a part headed "Files in out/", and the `fake` one lists `out/fake.txt`.
    - The `docs/` line of `structure.md` names `pipeline.md`, and another line says that step specs update their own section.
    - _Requirements: 13.1, 13.2, 13.3, 13.4, 13.5, 13.8, 13.9_

  - [ ] 9.4 Write the import checks in `tests/test_imports.py`
    - Read the files with `ast` and run no Step, as the design's "The import checks" section gives it.
    - No AWS: no `*.py` under `splatpipe/`, found with `rglob`, imports `cloud`, `boto3`, `botocore`, `s3transfer`, `awscrt`, `awscli`, `aiobotocore` or `aioboto3`, or a submodule of one. The check reads `import`, `from ... import`, and `importlib.import_module` or `__import__` calls with a constant string, anywhere in the file. A failure gives the file and the line.
    - Standard library only: `splatpipe/*.py`, `splatpipe/fake/**/*.py` and `cloud/spend.py` import only names in `sys.stdlib_module_names`, plus `splatpipe` for the first two. In `splatpipe/*.py`, each `splatpipe.X` import names a Skeleton module, never a step package.
    - A subprocess runs `import splatpipe.run` and finds only Skeleton modules among the `splatpipe.*` names in `sys.modules`.
    - Each `STEP_LIST` entry's module is `splatpipe.<name>`, its name passes `check_step_name`, its Step_Section loads from the real config, read by path, with `step_settings(name, preset)` for each of the three Presets, and it isn't the Fake_Step. The list starts empty, so this check loads nothing until a step spec adds its entry.
    - _Requirements: 5.11, 8.2, 11.1, 12.1, 12.2, 12.3, 12.4, 12.5_

  - [ ] 9.5 Run the tests and commit
    - Run `uv run pytest`. Every test passes, `tests/test_guard.py` included.
    - On `feat/pipeline-core`: `git add docs/pipeline.md .kiro/steering/structure.md tests/test_docs.py tests/test_imports.py`, then commit those files with the message "Pipeline core: pipeline doc, structure steering and import checks".
    - _Requirements: 14.1_

## Notes

- Every test sub-task is required, as `.kiro/steering/rules.md` asks. None is marked optional.
- The last sub-task of each task is its checkpoint: the whole suite, then the commit. So the plan has no separate checkpoint tasks and stays at the design's 9.
- Every acceptance criterion in `requirements.md` is named by at least one task.
- The graph runs the tasks in order, one at a time, so each task's test run and commit see only finished work. Inside a task, sub-tasks that write different files share a wave.

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "1.2"] },
    { "id": 1, "tasks": ["1.3", "1.4"] },
    { "id": 2, "tasks": ["1.5"] },
    { "id": 3, "tasks": ["1.6"] },
    { "id": 4, "tasks": ["2.1", "2.2"] },
    { "id": 5, "tasks": ["2.3"] },
    { "id": 6, "tasks": ["2.4"] },
    { "id": 7, "tasks": ["2.5"] },
    { "id": 8, "tasks": ["3.1", "3.2"] },
    { "id": 9, "tasks": ["3.3"] },
    { "id": 10, "tasks": ["3.4"] },
    { "id": 11, "tasks": ["3.5"] },
    { "id": 12, "tasks": ["3.6"] },
    { "id": 13, "tasks": ["4.1", "4.2"] },
    { "id": 14, "tasks": ["4.3"] },
    { "id": 15, "tasks": ["4.4", "4.5"] },
    { "id": 16, "tasks": ["4.6"] },
    { "id": 17, "tasks": ["5.1"] },
    { "id": 18, "tasks": ["5.2"] },
    { "id": 19, "tasks": ["5.3"] },
    { "id": 20, "tasks": ["5.4"] },
    { "id": 21, "tasks": ["5.5"] },
    { "id": 22, "tasks": ["5.6", "5.7"] },
    { "id": 23, "tasks": ["5.8"] },
    { "id": 24, "tasks": ["6.1"] },
    { "id": 25, "tasks": ["6.2", "6.3"] },
    { "id": 26, "tasks": ["6.4"] },
    { "id": 27, "tasks": ["6.5"] },
    { "id": 28, "tasks": ["6.6"] },
    { "id": 29, "tasks": ["6.7"] },
    { "id": 30, "tasks": ["7.1"] },
    { "id": 31, "tasks": ["7.2"] },
    { "id": 32, "tasks": ["7.3"] },
    { "id": 33, "tasks": ["7.4"] },
    { "id": 34, "tasks": ["7.5"] },
    { "id": 35, "tasks": ["7.6"] },
    { "id": 36, "tasks": ["8.1"] },
    { "id": 37, "tasks": ["8.2"] },
    { "id": 38, "tasks": ["8.3"] },
    { "id": 39, "tasks": ["8.4"] },
    { "id": 40, "tasks": ["9.1", "9.2"] },
    { "id": 41, "tasks": ["9.3", "9.4"] },
    { "id": 42, "tasks": ["9.5"] }
  ]
}
```
