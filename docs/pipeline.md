# The pipeline

`splatpipe/` turns one phone video into a 3D Gaussian splat, one step at a time. This doc starts with the skeleton, the shared code that every step plugs into: the job folder, the settings, the reports and the commands that run steps. Then each step has its own section, like the fake step's at the end, with the same parts. Each step spec adds or updates its own section as part of its tasks, so the steps we build separately keep fitting together.

## The job folder

`splatpipe.make` makes one job folder per video, inside the jobs folder. The jobs folder is `jobs_folder` in `[jobs]`: `data/jobs/`, which git ignores.

```
data/jobs/                      the jobs folder
  20261119-142530-7f3a9c/       one job folder, named after its job id
    input/
      video.mov                 the copy of the video: video.mp4 or video.mov
      job.json                  the job record
    work/
      <name>/                   a step's work folder, for its in-between files
    out/
      reports/
        <name>.json             a step's report
      job.json                  the job summary
      job.log                   the job log
      fake.txt                  a step's own files, like this one from the fake step
```

- `splatpipe.make` makes `input/`, `work/` and `out/`, copies the video into `input/` and writes the job record.
- The step runner empties a step's work folder before the step runs, and writes the step's report after it.
- A step's own code writes its in-between files to its work folder and its results to `out/`.
- The job runner writes the job summary and the job log.

A job id is the UTC date and time the job was made, then 6 random hex characters: `YYYYMMDD-HHMMSS-xxxxxx`. Every CLI and function that gets a job id checks that the whole id matches `[0-9]{8}-[0-9]{6}-[0-9a-f]{6}` before it uses the id in a path. It also refuses a job folder that's a link leading outside the jobs folder.

## The job record

`input/job.json` holds what `splatpipe.make` found and what the person chose:

```json
{
  "job_id": "20261119-142530-7f3a9c",
  "preset": "fast",
  "start_seconds": 1.5,
  "end_seconds": 12.48,
  "source_name": "Kitchen walk.MOV",
  "source_bytes": 48213775,
  "video_seconds": 12.48,
  "video_file": "video.mov"
}
```

| Field | What it holds |
|---|---|
| `job_id` | The job id, the same as the job folder's name. |
| `preset` | The job's preset: `fast`, `default` or `quality`. |
| `start_seconds` | Where the part of the video to use starts, in seconds. 0 when no start was given. |
| `end_seconds` | Where that part ends, in seconds. The video's length when no end was given. |
| `source_name` | The original file's name, without its folder. |
| `source_bytes` | The original file's size in bytes. |
| `video_seconds` | The whole video's length in seconds, from ffprobe. |
| `video_file` | The copy's name in `input/`: `video.mp4` or `video.mov`. |

Steps use only the part of the video from `start_seconds` to `end_seconds`. They find the copy as `step.job.video_path`, so they never guess its extension.

## Step reports

When a step runs, the step runner writes its report to `out/reports/<name>.json`, done or failed, in place of the one from an earlier run. A step that can't start writes no report, as "Running one step" says.

```json
{
  "step": "fake",
  "job_id": "20261119-142530-7f3a9c",
  "preset": "fast",
  "settings": {"fail": "none", "file_bytes": 16, "files": 1},
  "outcome": "done",
  "started_utc": "2026-11-19T14:25:31.204+00:00",
  "finished_utc": "2026-11-19T14:25:31.209+00:00",
  "duration_seconds": 0.005,
  "timings": {"write_files": 0.0011},
  "counts": {"files_written": 1, "video_bytes": 48213775},
  "warnings": ["This is the fake step. It writes test files and makes no splat."],
  "error": null
}
```

| Field | What it holds |
|---|---|
| `step` | The step name. |
| `job_id` | The job id. |
| `preset` | The preset the step ran with. |
| `settings` | The settings the step used: its shared settings and its preset's settings, in one object. `null` when they couldn't be loaded. |
| `outcome` | `"done"` or `"failed"`. |
| `started_utc`, `finished_utc` | When the step started and finished, in UTC, to the millisecond. |
| `duration_seconds` | How long the step took, in seconds, rounded to 3 decimals. |
| `timings` | The named timings the step added, in seconds. |
| `counts` | The named whole-number counts the step added. |
| `warnings` | The warnings the step added. |
| `error` | `null` when the step is done, and the error when it failed. |

A failed step's `error` looks like this:

```json
"error": {
  "readable": "The fake step was set to fail by fail = \"expected\" in [fake]. Set fail = \"none\" to make it pass.",
  "expected": true,
  "technical": {
    "type": "splatpipe.errors.ExpectedFailure",
    "message": "The fake step was set to fail by fail = \"expected\" in [fake]. Set fail = \"none\" to make it pass.",
    "traceback": "Traceback (most recent call last):\n  File ..."
  }
}
```

- `readable` is the readable error: a message for the person who made the job, saying what went wrong and what to do, with no traceback. When the step raised `ExpectedFailure`, or the skeleton refused something, like a missing setting, `readable` is that error's own message and `expected` is `true`.
- For any other error, a failed import included, `readable` is a general message and `expected` is `false`: "The <name> step stopped because of an error in its code. This is a bug, not a problem with your video. Send out/reports/<name>.json to the team."
- `technical` is for the team: the error's type, its message and its full traceback.

A failed report keeps the timings, counts and warnings the step added before it failed. A step adds them with `step.add_timing(name, seconds)` or `with step.timed(name):`, `step.add_count(name, value)` and `step.add_warning(text)`. A report holds only plain JSON: objects with string keys, arrays, strings, finite numbers, `true`, `false` and `null`. An `add_*` call with any other value, like a float count or NaN seconds, fails the step as a bug, with the field named in `technical`.

## Presets and config.toml

A preset is `fast`, `default` or `quality`. A job gets its preset when it's made, from `--preset` or else from `default_preset` in `[jobs]`, and keeps it in its job record. Each step reads its numbers for the job's preset. A step's own CLI can run it with another preset, with `--preset`, and the report records the preset it used. The job runner always uses the job record's preset.

Every number and setting lives in `config.toml` at the repo root, read with `tomllib`. `[git]`, `[aws]`, `[paid]` and `[devlog]` aren't the pipeline's, and they stay as they are. Every other section is a pipeline section: `[jobs]`, each step's section, and any section a later spec adds, like `[cloud]`. In a pipeline section:

- Every key has a short comment, at the end of its line or on the line just above it. The tests check this.
- No value is NaN, infinity, or a TOML date or time. Loading the config refuses one, naming the section and the key.

`[jobs]` holds the skeleton's own settings:

| Key | Value | What it is |
|---|---|---|
| `jobs_folder` | `"data/jobs"` | The folder that holds every job folder. A relative path is read from the repo root. |
| `default_preset` | `"fast"` | The preset a job gets when none is chosen. |
| `max_video_bytes` | `4294967296` | The largest video accepted, in bytes (4 GB). |
| `max_video_seconds` | `600` | The longest video accepted, in seconds (10 minutes). |
| `min_used_seconds` | `3` | The shortest part of the video a job may use, in seconds. |
| `ffprobe_timeout_seconds` | `30` | The longest time ffprobe may take to read a video, in seconds. |

Each step has a section named after it:

- `[<name>]` holds the step's shared settings, the same for every preset.
- `[<name>.fast]`, `[<name>.default]` and `[<name>.quality]` hold its preset settings. All three must be there, with the same keys.
- A key is either shared or in the preset tables, never both, and the section holds no other tables.

The step gets its shared settings and its preset's settings together, as `step.settings`. A section that breaks these rules fails the step with a readable error that names the section and the key or table. So does a setting the step asks for that isn't in its section. The fake step's section, at the end, is a small example.

## The step list

A step name has lowercase ASCII letters, digits and underscores, and starts with a letter, like `fake`. It names the step's module, `splatpipe.<name>`, its config section, its report and its work folder. A step can't take a name the skeleton uses: `errors`, `config`, `jsonio`, `jobs`, `probe`, `make`, `step`, `run`, `git`, `aws`, `paid` or `devlog`.

The step list is `STEP_LIST` in `splatpipe/run.py`: the steps the job runner runs, in order. Each entry is a `StepEntry(name, module, required)`. When a required step fails, the job fails. When an optional step fails, the job goes on without it. The fake step isn't in the list, so it never runs on real jobs. The tests pass their own lists.

The list starts empty, and each step's spec adds its own entry. Until then, `splatpipe.run` refuses with "No steps are set up yet", so run one step on its own. The planned steps are:

- Required steps: picking frames, solving cameras and training.
- Optional steps: depth priors, plain walls and clean.

The clean step isn't in `STEP_LIST` yet. The first spec with a step worth running adds the entries, and puts the clean step's last, as an optional step.

Running a job in the cloud comes in its own spec, and it isn't a step.

## How a step is built

Each step is a package, `splatpipe/<name>/`, built like the fake step:

- `__init__.py` defines `work(step)`, the step's own code, and `run(job_id, preset=None, *, config=None)`, its plain function, which runs the step through the step runner and returns the report.
- `__main__.py` defines `main(argv)`, which calls `step_main("<name>", "splatpipe.<name>", argv)`, and ends with `if __name__ == "__main__": sys.exit(main())`.

`work(step)` gets a `StepContext` from `splatpipe.step`:

- `step.name` and `step.preset`.
- `step.job`, with its `id`, its `folder`, its `record`, which holds the job record's fields, and its `video_path`, the copy of the video.
- `step.settings`: the step's shared settings and its preset's settings.
- `step.work_folder`, already emptied, and `step.out_folder`, the job's `out/`.
- `add_timing`, `timed`, `add_count` and `add_warning`, which add to the report.

For a failure it foresees, like too few sharp frames, `work` raises `ExpectedFailure` from `splatpipe.errors`, with a readable error. Anything else it raises counts as a bug.

## Running one step

The step runner, `run_step` in `splatpipe/step.py`, runs one step on one job. A step's CLI, its `run` function and the job runner all go through it:

1. It checks the step name, loads `config.toml` and opens the job, checking the job id and the job record. It takes the preset given with `--preset`, or else the job record's. When any of this fails, it stops with a readable error and writes no report.
2. It removes the step's earlier report.
3. It loads the step's settings, checks that the work folder is inside the job folder, empties it, imports the step's module and calls its `work(step)`. A failure from here on gives a failed report.
4. It writes the report, done or failed, and returns it.

It never reads the step's earlier report, so it always runs the step. In the job folder, the step runner itself touches only the step's work folder and its report.

## The job runner

The job runner, `run_job` in `splatpipe/run.py` with its CLI `splatpipe.run`, runs the step list on one job:

1. It refuses an empty step list, a bad or reserved step name, a step listed twice and a module outside `splatpipe`, before it reads or writes anything.
2. It loads `config.toml` once and opens the job.
3. It reads each step's earlier report and picks the first step to run, as the part below on running a job again says.
4. Before any step runs, it removes `out/job.json` and the reports of the steps it's about to run.
5. It runs those steps in order through the step runner, with the job record's preset. It imports a step's module only when the step is about to run.
6. When a required step fails, the job stops, and every later step is not run. When an optional step fails, its name and readable error go to the job log, and the job goes on.
7. It writes the job summary and returns it.

The job status is one of these. The job log and the CLIs use the words, and the job summary uses the JSON spellings.

- done (`"done"`): every step is done.
- done with warnings (`"done_with_warnings"`): every required step is done and at least one optional step failed. Warnings inside a step's report don't change the status.
- failed (`"failed"`): a required step failed.

## The job summary

`out/job.json` sums up the run. In this example, from the tests, the step list runs the fake step three times. `fake_a` is optional and failed, and `fake_b` is required and failed, so `fake_c` wasn't run. The messages are shortened here.

```json
{
  "job_id": "20261119-142530-7f3a9c",
  "preset": "fast",
  "fresh": false,
  "status": "failed",
  "started_utc": "2026-11-19T14:25:31.200+00:00",
  "finished_utc": "2026-11-19T14:25:31.240+00:00",
  "duration_seconds": 0.04,
  "steps": [
    {"step": "fake_a", "required": false, "outcome": "failed", "reused": false, "readable_error": "The fake step was set to fail ..."},
    {"step": "fake_b", "required": true, "outcome": "failed", "reused": false, "readable_error": "The fake step was set to fail ..."},
    {"step": "fake_c", "required": true, "outcome": "not_run", "reused": false, "readable_error": null}
  ],
  "failed_step": "fake_b",
  "readable_error": "The fake step was set to fail ..."
}
```

- `preset` is the job record's, and `fresh` says whether the run had `--fresh`.
- `status` is `"done"`, `"done_with_warnings"` or `"failed"`.
- `steps` has one item per step in the step list, in order. Its `outcome` is `"done"`, `"failed"` or `"not_run"`. `reused` is `true` for a finished step the run skipped, whose `outcome` and `readable_error` come from its earlier report. It's `false` for a step that ran or wasn't reached.
- `failed_step` and the top-level `readable_error` name the required step that failed and give its readable error. They're `null` unless the status is `"failed"`.

## The job log

`out/job.log` gets a line for each thing the job runner does, with the time in UTC and a level. The example's first run wrote these lines, with the messages shortened:

```
2026-11-19T14:25:31Z INFO job 20261119-142530-7f3a9c: 3 steps, preset fast
2026-11-19T14:25:31Z INFO fake_a: started
2026-11-19T14:25:31Z WARNING fake_a: failed, optional so the job goes on: The fake step was set to fail ...
2026-11-19T14:25:31Z INFO fake_b: started
2026-11-19T14:25:31Z ERROR fake_b: failed, required so the job stops: The fake step was set to fail ...
2026-11-19T14:25:31Z INFO fake_c: not run
2026-11-19T14:25:31Z INFO job 20261119-142530-7f3a9c: failed
```

A run starts with a line that gives the job id, the number of steps and the preset. Each step that runs gets a started line and an outcome line, each step the run didn't reach gets a "not run" line, and the last line gives the job status. A step that logs with `logging.getLogger(__name__)`, at INFO or above, lands in the job log too when the job runner runs it.

## Running a job again

A run without `--fresh` picks up where the last one stopped.

- A step is finished when its report is for this step and this job and says done, or says failed and the step is optional in the step list. A missing report, one that can't be read as JSON, one for another step or job, one with another outcome, and a required step's failed report don't count.
- Without `--fresh`, the job runner skips the finished steps at the start of the step list. It runs every step from the first unfinished one on, a finished step after that one included, because that step's inputs may have changed.
- Before any step runs, it removes the summary and the reports of the steps it's about to run, so a step that ends up not run has no report left over from an earlier run. The skipped steps keep their reports, and so do steps outside the step list. A refused run removes nothing.
- The skipped steps show `"reused": true` in the summary, and their earlier outcomes count toward the job status, like those of the steps that ran. So a skipped optional step that failed makes the job done with warnings.
- When every step has finished, nothing runs. The summary comes from the earlier reports, the CLI says that nothing ran and that `--fresh` runs every step again, and it exits with 0. Such a job is always done or done with warnings, because a required step that failed never counts as finished.
- Without `--fresh`, the run adds its lines to the end of `job.log`, and starts the file when it's missing. When it skips steps, its second line names them, with their earlier outcomes. `--fresh` runs every step, the finished ones included, and starts a new `job.log`.
- A single step's CLI, like `uv run python -m splatpipe.fake --job <id>`, always runs its step, even when its report already says done. Only the job runner skips finished steps.

Use `--fresh` after changing settings in `config.toml` or a step's code, or after running an earlier step by hand, because the job runner doesn't check whether a finished step's inputs changed.

In the example, once `fake_b` is fixed, a run without `--fresh` skips `fake_a`, because a failed optional step is finished, and runs `fake_b` and `fake_c`. It adds these lines to the end of the same `job.log`:

```
2026-11-19T14:40:12Z INFO job 20261119-142530-7f3a9c: 3 steps, preset fast
2026-11-19T14:40:12Z INFO skipped, finished in an earlier run: fake_a (failed)
2026-11-19T14:40:12Z INFO fake_b: started
2026-11-19T14:40:12Z INFO fake_b: done
2026-11-19T14:40:12Z INFO fake_c: started
2026-11-19T14:40:12Z INFO fake_c: done
2026-11-19T14:40:12Z INFO job 20261119-142530-7f3a9c: done with warnings
```

Its summary has `"status": "done_with_warnings"`, because the reused failure of `fake_a` counts, and these steps:

```json
"steps": [
  {"step": "fake_a", "required": false, "outcome": "failed", "reused": true, "readable_error": "The fake step was set to fail ..."},
  {"step": "fake_b", "required": true, "outcome": "done", "reused": false, "readable_error": null},
  {"step": "fake_c", "required": true, "outcome": "done", "reused": false, "readable_error": null}
]
```

With `--fresh`, `job.log` would hold only this run, and `fake_a` would run again in place of the skipped line.

## A Step's files in out/

The step runner removes only a step's own report and empties only its own work folder. It leaves the step's other files in `out/` alone, so each step's own code replaces or removes what it wrote there in an earlier run. Each step's section lists the files the step writes to `out/` in a part headed "Files in out/", or says it writes none.

## CLIs and exit codes

| CLI | What it does |
|---|---|
| `uv run python -m splatpipe.make <video> [--start S] [--end E] [--preset P]` | Checks the video, makes a job from it and prints the new job id alone on stdout. |
| `uv run python -m splatpipe.<name> --job <id> [--preset P]` | Runs one step on one job, even when it finished before. `--preset` replaces the job record's preset for this run. Prints `<name>: done` when the step is done. |
| `uv run python -m splatpipe.run --job <id> [--fresh]` | Runs the step list on one job, from the first unfinished step, or every step with `--fresh`. Prints `Job <id>: <status>`, like `Job 20261119-142530-7f3a9c: done with warnings`. |

`splatpipe.make` finishes every check before it makes a folder or copies anything. It checks the preset, that the video is a file ending in `.mp4` or `.mov` in any case, and its size against `max_video_bytes`. Then ffprobe must read it as MP4 or MOV, the format `mov,mp4,m4a,3gp,3g2,mj2`, with a video stream and a length of at most `max_video_seconds`. Last comes the trim. `--start` and `--end` must be 0 or more and inside the video, the start must come before the end, and the part between them must be at least `min_used_seconds` long, also when no trim is given. The original video stays as it was.

Each CLI has a plain function too, which raises `PipelineError` with the readable error where the CLI refuses:

- `make_job(video, *, start=None, end=None, preset=None)` in `splatpipe.make` returns the job id.
- `run(job_id, preset=None)` in each step's package returns the step's report.
- `run_job(job_id, *, fresh=False)` in `splatpipe.run` returns the job summary.

| Exit code | When |
|---|---|
| 0 | `make` made the job, a step is done, or the job is done or done with warnings, a job where every step had already finished included. |
| 1 | A refusal, a failed step or a failed job. The readable error goes to stderr, with no traceback. |
| 2 | A bad command line. argparse prints the usage and the error. |

A step's CLI exits with 1 when the step failed, for a required step and an optional step alike. When the job failed, `splatpipe.run` prints the failed step's readable error on stderr. A bug in the skeleton itself prints "splatpipe stopped because of a bug (<type>: <message>). Tell the team the command you ran." and exits with 1.

## Step: fake

The fake step is a test step in `splatpipe/fake/`. It does trivial CPU work and makes no splat. The step specs copy the fake step as their working example: its package, its code, its CLI and the parts of this section. It isn't in the step list, so it never runs on real jobs. The tests put it in their own step lists under other names, like `fake_a`, and its paths and section follow the name: `[fake_a]`, `work/fake_a/`, `out/fake_a.txt` and `out/reports/fake_a.json`.

Its CLI is `uv run python -m splatpipe.fake --job <id> [--preset <name>]`, and its plain function is `splatpipe.fake.run(job_id, preset=None)`.

### What it reads

- The size of the job's copy of the video, through `step.job.video_path`.
- Its settings, from `[fake]` and from the job's preset table, like `[fake.fast]`.

### What it writes

1. `files` small files of `file_bytes` bytes each into `work/fake/`, named `file-000.txt`, `file-001.txt` and so on.
2. Its timing, counts and warning, into its report.
3. Then it checks `fail`. `"expected"` fails the step with its own readable error. `"unexpected"` raises a `RuntimeError`, so the step fails like a bug. Any other value except `"none"` fails it with a readable error that names the setting.
4. When it didn't fail, `out/fake.txt`, below.

The timing, the counts and the warning come before the `fail` check, so a failed report has them too.

### Config section

| Key | Table | Value | What it is |
|---|---|---|---|
| `fail` | `[fake]` | `"none"` | How the step fails on purpose: `"none"`, `"expected"` or `"unexpected"`. Only tests change it. |
| `file_bytes` | `[fake]` | `16` | The size of each small file, in bytes. |
| `files` | `[fake.fast]`, `[fake.default]` and `[fake.quality]` | `1`, `2` and `3` | The number of small files. |

The real `config.toml` keeps `fail = "none"`, with a comment above every key. Tests set `fail` in their own config.

### Step report fields

- `settings`: `fail`, `file_bytes` and `files`.
- `timings.write_files`: the seconds it took to write the small files.
- `counts.files_written`: the number of small files, the preset's `files`.
- `counts.video_bytes`: the size of the job's copy of the video, in bytes.
- `warnings`: "This is the fake step. It writes test files and makes no splat."
- `error`, when it failed. With `fail = "expected"`, `readable` is its own message, as in the example under "Step reports", and `expected` is `true`. With `fail = "unexpected"`, `readable` is the general bug message, `expected` is `false`, and `technical` holds the `RuntimeError`.

### Files in out/

- `out/fake.txt`: one UTF-8 line, like `The fake step ran on job 20261119-142530-7f3a9c with the fast preset.` A done run replaces it. A failed run writes none and leaves an earlier one in place.

Its report, `out/reports/fake.json`, comes from the step runner, like every step's.

## Step: clean

The clean step is in `splatpipe/clean/`. It reads the splat the training step wrote, drops the Gaussians a viewer shouldn't see and writes the rest as the clean splat, a binary PLY. Then it makes the viewer copy, a small `.sog` for the web viewer, from the clean splat with PlayCanvas splat-transform. These two files are what the person who filmed the place gets back. The step is built like the fake step. It isn't in the step list yet, as "The step list" says, so run it on its own.

Its CLI is `uv run python -m splatpipe.clean --job <id> [--preset P]`, and its plain function is `splatpipe.clean.run(job_id, preset=None)`.

### What it reads

- `out/<train_ply>`, which is `out/train.ply` with the real config: the splat the training step writes. This is the contract between the two steps: the training step writes it as a standard 3DGS binary PLY, `binary_little_endian` with float32 values, whose `vertex` element has one row per Gaussian and these properties:
  - `x`, `y` and `z`, the centre.
  - `f_dc_0` to `f_dc_2`, the base colour.
  - `f_rest_*`, the higher spherical harmonics: 45 of them at SH degree 3, or 0, 9 or 24 at a lower degree.
  - `opacity`, stored before the sigmoid.
  - `scale_0` to `scale_2`, stored as logarithms.
  - `rot_0` to `rot_3`, the rotation quaternion.
  - Sometimes `nx`, `ny` and `nz`, which nothing uses.
- Its settings, from `[clean]` and from the job's preset table, like `[clean.fast]`.

The step reads meaning only into `x`, `y`, `z`, `opacity`, `scale_0` to `scale_2` and `rot_0` to `rot_3`. It carries every other property through to the clean splat unchanged, a NaN or an infinity in a colour included. It never writes `out/<train_ply>`, so the trained splat survives a failed clean.

### What it writes

The step works in this order, and each filter gets what the one before it kept:

1. It fails when the trained splat holds more than `max_source_splats` (8,000,000) Gaussians, before it reads their rows.
2. The broken filter drops every Gaussian with a NaN or an infinity in `x`, `y`, `z`, `opacity`, `scale_0` to `scale_2` or `rot_0` to `rot_3`, or with a rotation of zero length, because nothing can draw one. It looks at no other property.
3. The faint filter judges every faint Gaussian together with the rest of its cube. A Gaussian is faint when its real opacity, the sigmoid of its `opacity`, is below `min_opacity` (0.005). Every Gaussian that isn't faint is kept, and no Gaussian is dropped for its opacity alone. Space is cut into cubes whose side is the trimmed extent divided by `faint_cube_divisions` (128). The trimmed extent is the longest side of the box that leaves out `extent_trim_percent` (1%) of the centres at each end of each axis, so a few far-off Gaussians can't make the cubes huge. A cube keeps all its faint Gaussians when it holds at least `faint_min_per_cube` (8) of them and they paint more than `faint_min_paint_share` (10%) of what the whole cube paints, and drops them all otherwise. A Gaussian's paint is its real opacity × π × its two largest real scales, the exponentials of its `scale_*`. So a sheet of faint Gaussians that together paint a visible surface stays, and scattered faint specks go.
4. The isolated filter drops every Gaussian whose mean distance to its `neighbours` (8) nearest other Gaussians is more than `isolation_ratio` (8) times the median of that number over those neighbours. The test is a local ratio, so the sky and the far background, sparse but even, survive.
5. The limits, all checked before either output is written, in this order:
   - The removal cap. The faint and isolated filters together may remove at most `max_removed_fraction` (70%) of the trained splat's Gaussians. Broken Gaussians are always removed and don't count toward the cap. Past the cap, the step fails and writes neither output, rather than trim the removals back to the cap.
   - Fewer than `min_kept_splats` (10,000) Gaussians left fails the step the same way.
   - Keeping less than `warn_kept_fraction` (50%) of the trained splat's Gaussians adds a warning, and the step goes on.
6. `work/clean/<clean_ply>`, the clean splat: the kept Gaussians as a `binary_little_endian` PLY, with every property of the trained splat in the same order and with the same type, and the trained splat's `comment` lines.
7. `work/clean/<clean_sog>`, the viewer copy: splat-transform 3.8.0 makes it from the clean splat, as one bundled `.sog` file that keeps at most `sog_sh_bands` bands of spherical harmonics. The step writes splat-transform's whole output to `work/clean/splat-transform.log`.
8. Then it moves both into `out/`, as "Files in out/" says.

When the step fails, the trained splat is still in `out/`, as it was. The clean step goes in the step list as an optional step, so once it's there, a failed clean ends the job done with warnings.

Every part of the step is CPU work. The Python side uses numpy and scipy, and splat-transform runs with `-g cpu`, which puts its GPU work on the CPU. The step is sized for the 8 vCPUs and 8 GB a step gets on the job machine: `max_workers` is at most 8, and `max_source_splats` keeps the trained splat's data, at 62 float32 properties a Gaussian, under a quarter of the 8 GB.

The viewer copy is SOG meta version 2, because splat-transform is pinned at 3.8.0, so the web viewer has to be checked against the version 2 spec.

The step needs Node 22 or newer, and `npm ci` at the repo root, which installs splat-transform 3.8.0 into `node_modules/`.

### Config section

| Key | Table | Value | What it is |
|---|---|---|---|
| `train_ply` | `[clean]` | `"train.ply"` | The trained splat's file in `out/`, which the training step writes and this step reads. |
| `clean_ply` | `[clean]` | `"clean.ply"` | The clean splat's file in `out/`. It must end in `.ply`. |
| `clean_sog` | `[clean]` | `"clean.sog"` | The viewer copy's file in `out/`. It must end in `.sog`. |
| `min_opacity` | `[clean]` | `0.005` | The real opacity below which a Gaussian is faint. |
| `faint_cube_divisions` | `[clean]` | `128` | The side of the faint filter's cubes is the trimmed extent divided by this. |
| `faint_min_per_cube` | `[clean]` | `8` | The fewest faint Gaussians a cube must hold to keep them. |
| `faint_min_paint_share` | `[clean]` | `0.10` | A cube keeps its faint Gaussians only when they paint more than this share of what the cube paints. |
| `extent_trim_percent` | `[clean]` | `1` | The percent of the centres left out at each end of each axis for the trimmed extent. |
| `neighbours` | `[clean]` | `8` | How many nearest other Gaussians the isolated filter looks at. |
| `isolation_ratio` | `[clean]` | `8.0` | The isolated filter drops a Gaussian whose mean distance to its nearest neighbours is more than this many times the median of that number over those neighbours. |
| `max_removed_fraction` | `[clean]` | `0.70` | The removal cap: the largest share of the trained splat's Gaussians that the faint and isolated filters may remove together. |
| `min_kept_splats` | `[clean]` | `10000` | The fewest Gaussians the clean splat may hold. Fewer fails the step. |
| `warn_kept_fraction` | `[clean]` | `0.5` | The step warns when it keeps less than this share of the trained splat's Gaussians. |
| `max_source_splats` | `[clean]` | `8000000` | The most Gaussians the step takes in. More fails the step before it reads their rows. |
| `max_workers` | `[clean]` | `4` | The worker threads for the isolated filter's neighbour search and for splat-transform, at most 8. At 0, splat-transform encodes inline and the search uses 1. |
| `splat_transform_timeout_seconds` | `[clean]` | `3600` | The longest time splat-transform may take to make the viewer copy, in seconds. Past it, the step stops splat-transform and fails. |
| `sog_sh_bands` | `[clean.fast]`, `[clean.default]` and `[clean.quality]` | `3`, `3` and `3` | The bands of higher spherical harmonics the viewer copy keeps, from 0 to 3. |
| `sh_iterations` | `[clean.fast]`, `[clean.default]` and `[clean.quality]` | `10`, `10` and `20` | The iterations of splat-transform's spherical harmonic compression. |

All three presets clean the splat the same way, and only `sh_iterations` differs between their viewer copies. The real `config.toml` has a comment on the line directly above every key.

### Step report fields

- `counts.source_splats`: the Gaussians in the trained splat.
- `counts.dropped_broken`, `counts.dropped_faint` and `counts.dropped_isolated`: the Gaussians each filter dropped.
- `counts.kept_splats`: the Gaussians left after the three filters.
- `counts.clean_ply_bytes` and `counts.clean_sog_bytes`: the sizes of the clean splat and the viewer copy, in bytes.
- `timings.read_ply`, `timings.clean`, `timings.write_ply` and `timings.export_sog`: the seconds it took to read the trained splat, run the three filters, write the clean splat and run splat-transform.
- `warnings`, which gets a warning in each of these cases:
  - When the step keeps less than `warn_kept_fraction` of the trained splat's Gaussians. The warning gives both counts and the share the faint and isolated filters removed.
  - When the preset's `sog_sh_bands` is more than the bands the trained splat has. The warning names both numbers, and the viewer copy keeps the trained splat's bands.

There's no `removed_percent` count, because `step.add_count` takes only whole numbers. The share the faint and isolated filters removed is in the first warning and in the readable error of a run that stops at the removal cap.

### Files in out/

- `out/<clean_ply>`, which is `out/clean.ply` with the real config: the clean splat.
- `out/<clean_sog>`, which is `out/clean.sog` with the real config: the viewer copy, made from that clean splat.

A done run replaces both. Once its settings pass their checks, the step removes both from `out/`, the viewer copy first, before it writes either. It makes the new ones in `work/clean/` and moves them into `out/` only once both are whole, the clean splat first and the viewer copy straight after. Each move is a rename, so `out/` never holds a half-written file. So:

- A failure after that removal leaves neither file in `out/`, an earlier run's pair included. That holds for an expected failure, a bug and Ctrl+C alike. When a failure falls between the two moves, the step removes the clean splat it had already moved.
- A hard stop the step can't see, like a killed process or a power cut, can leave a whole clean splat alone in `out/`: this run's, between the two moves into `out/`, or the earlier one, between the two removals. No done report goes with it, and the next run of the step removes it first.
- A viewer copy never sits in `out/` without the clean splat it was made from, because the viewer copy is removed first and moved last.

Trust the pair only when the step's report, `out/reports/clean.json`, says done. That report comes from the step runner, like every step's.
