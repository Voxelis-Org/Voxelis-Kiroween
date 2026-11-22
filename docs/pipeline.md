# The pipeline

`splatpipe/` turns one phone video into a 3D Gaussian splat, one step at a time. This doc starts with the skeleton, the shared code that every step plugs into: the job folder, the settings, the reports and the commands that run steps. Then each step has its own section, like the fake step's below, with the same parts. Each step spec adds or updates its own section as part of its tasks, so the steps we build separately keep fitting together.

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

The step gets its shared settings and its preset's settings together, as `step.settings`. A section that breaks these rules fails the step with a readable error that names the section and the key or table. So does a setting the step asks for that isn't in its section. The fake step's section, below, is a small example.

## The step list

A step name has lowercase ASCII letters, digits and underscores, and starts with a letter, like `fake`. It names the step's module, `splatpipe.<name>`, its config section, its report and its work folder. A step can't take a name the skeleton uses: `errors`, `config`, `jsonio`, `jobs`, `probe`, `make`, `step`, `run`, `git`, `aws`, `paid` or `devlog`.

The step list is `STEP_LIST` in `splatpipe/run.py`: the steps the job runner runs, in order. Each entry is a `StepEntry(name, module, required)`. When a required step fails, the job fails. When an optional step fails, the job goes on without it. The fake step isn't in the list, so it never runs on real jobs. The tests pass their own lists.

The list holds the frames step as its first, required entry: `StepEntry("frames", "splatpipe.frames", True)`. Each later step spec adds its own entry after the frames step. Until the camera solve spec adds its entry, a job is done once the frames step is done. The planned steps are:

- Required steps: picking frames, solving cameras and training.
- Optional steps: depth priors and clean and export.

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

## Running a job in the cloud

The heavy work runs on one GPU machine in AWS, one machine per job. The code that drives it lives under `cloud/`, so `splatpipe/` never learns where it runs. The cloud run is not a step and adds no `STEP_LIST` entry: a job in the cloud runs the same step list as a laptop run, through the same `run_job`. It's built in its own spec.

The laptop commands, each run as `uv run python -m cloud.<name>`:

| Command | What it does |
|---|---|
| `cloud.start <video> [--start S] [--end E] [--preset P]` | Makes the job, checks the repo is committed and the machine image is current, quotes the machine, the expected time and the rough cost, and after the yes uploads the video, the code and the job record and launches one machine. Paid; uses the runner profile. |
| `cloud.status <job-id>` | Reads the job summary and reports from the bucket and reports the job's progress. Free; read profile. |
| `cloud.download <job-id>` | Fetches the job's `out/` and reports into `data/jobs/<job-id>/out/`. Free; read profile. |
| `cloud.list` | Lists our machines that aren't terminated, with each one's type, state and launch time. Free; read profile. |
| `cloud.stopall` | Terminates every one of our machines that isn't terminated, the last-resort way to end a run. Free; runner profile. |

On the machine, the Cloud_Runner rebuilds the job folder from the bucket (the `input/`, plus any `out/` and `work/` from an earlier run of the same job), calls `run_job` without `--fresh` so it resumes from the first unfinished step exactly as a laptop run does, uploads each step's report, `work/` and `out/` as they appear with a final sync at the end, writes the spend lines, and powers the machine off when the job ends.

### The cap deadline

The Cloud_Runner sets one environment variable for the steps, `SPLATPIPE_CAP_DEADLINE`: the absolute UTC instant, in ISO 8601, by which a step should have saved and stopped. It's the machine's start time plus the preset's time cap, minus `upload_margin_minutes` from `[cloud]`, leaving room to upload the results before the machine's hard time-cap shutdown. No current step reads it; the training spec will.

### What each step asks the machine for

Each step records the machine it needs in its own config section, per preset, as `min_vcpus`, `min_memory_gb` and `min_gpu_memory_gb` (its shared settings and its preset table, like every other setting). A missing key means no need in that dimension. `min_memory_gb` and `min_gpu_memory_gb` are hard needs: `cloud.start` keeps only the types that meet the largest of each over the job's steps. `min_vcpus` is soft, because fewer vCPUs only make a step slower, so it never rules a type out. The plan for the steps we build:

| Step | What it asks for |
|---|---|
| Picking frames | 16 vCPUs, 16 GB RAM |
| Solving cameras | 16 vCPUs, 16 GB RAM |
| Depth priors | a GPU with 16 GB VRAM |
| Training | a GPU, 32 GB RAM |
| Clean and export | 8 vCPUs, 8 GB RAM |

Each step spec adds its own keys when it's built. There is no `[cloud.needs.*]` table; a step's needs live in the step's own section.

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

## Step: frames

The frames step is in `splatpipe/frames/`, built like the fake step. It picks the frames of the job's video that the camera solve step works from, and saves them in its work folder, `work/frames/`.

Its CLI is `uv run python -m splatpipe.frames --job <id> [--preset P]`, and its plain function is `splatpipe.frames.run(job_id, preset=None)`.

It works in this order:

1. It checks every setting, before any tool runs.
2. The probe: ffprobe reads the video's size, rotation and colour transfer. For an HDR video, the step also asks ffmpeg whether it has the filters to tone map it.
3. The score pass: ffmpeg decodes the part of the video to use as small grey score frames, `score_width` pixels wide. The step gives each score frame a sharpness, a motion and a contrast. It keeps those numbers and the time of every frame, but holds only an evenly thinned set of the frames themselves, at most twice `mask_samples`, for the mask.
4. The still trim cuts off the still part at the start and the one at the end, each when it lasts `min_still_seconds` or more. Still parts in the middle stay, and a video that's still from start to end is kept whole. The frames left are the kept range. From its length, the step works out `target_picks`, the most picks it makes.
5. The jolt filter drops the kept frames that are blurred next to their neighbours. When it keeps fewer frames than `target_picks`, the step skips it, with a warning.
6. The step splits the kept range into `target_picks` parts, each with an equal share of the kept range's motion, or of its time when the total motion is 0. It picks the sharpest frame left in each part, the earliest on a tie. A part with no frame left gives no pick. With fewer picks than `min_picks`, the step fails here, before the pick pass.
7. It looks for pixels stuck to the camera, like a drone's propeller or a watermark, in the mask samples. These are the thinned frames that lie in the kept range, or `mask_samples` of them spread evenly, the first and the last included, when more lie there. Then it decides whether to use the mask.
8. The pick pass: ffmpeg decodes the same part of the video again, in colour at the pick size. The step saves each pick and its grey copy.
9. It writes `mask.png` when it uses the mask, then `frames.json`, last of all.

### What it reads

- The job's copy of the video, through `step.job.video_path`.
- The job record's `start_seconds` and `end_seconds`. The step uses each frame whose time is at or after `start_seconds` and before `end_seconds`. A frame's time comes from the video's own timestamps, in seconds from the start of the video, so a video with a variable frame rate gets its frames' real times. A frame with no time, or with a time that isn't after the last frame the step took, is skipped, so the times always go up.
- Its settings, from `[frames]` and from the job's preset table, like `[frames.fast]`.
- ffmpeg and ffprobe 5.1 or newer, as command line tools. ffprobe reads the size, rotation, sample aspect ratio and colour transfer of the video's first video stream that isn't cover art. ffmpeg decodes every frame the step uses and hands it over through a pipe.

The job's machine image needs ffmpeg and ffprobe 5.1 or newer, because the step uses ffmpeg's `-fps_mode` option. It runs ffmpeg with `-fps_mode passthrough`, so no frame is repeated or dropped to keep a steady frame rate. Ubuntu 24.04's packaged ffmpeg, version 6.1, is new enough.

An HDR video is one whose colour transfer is `arib-std-b67` (HLG) or `smpte2084` (PQ). For an HDR video, the step asks ffmpeg whether it has the `zscale` and `tonemap` filters. With both, both passes tone map each frame to SDR BT.709, with `tonemap_npl` and `tonemap_desat`, before it's made grey or saved. Without them, both passes read the frames without tone mapping, and the report gets a warning. When ffmpeg fails in the score pass with the tone map, the step adds a warning and runs the score pass again without it, and the pick pass runs without it too. Every other video, one whose colour transfer ffprobe doesn't report included, is read without tone mapping.

### What it writes

The step writes only into its work folder. The camera solve step reads these files, so their layout and the fields of `frames.json` are the contract between the two steps. The names are fixed in `splatpipe/frames/files.py`, not in `config.toml`.

```
work/frames/
  images/000000.jpg ...   the picks, in colour
  grey/000000.jpg ...     the grey copies, with the picks' names and size
  mask.png                the mask, only when the step uses one
  frames.json             the frames record, written last
```

- `images/` holds the picks, as JPEGs at `jpeg_quality`. Each is named after its number in time order, from `000000.jpg`, padded to 6 digits. A pick is the same video frame as its score frame, in colour at the pick size. The pick size is the video's size as a player shows it, with its rotation and sample aspect ratio applied, scaled so the long edge is `long_edge` pixels. A video whose long edge is `long_edge` or shorter keeps its own size, so a pick is never larger than the video.
- `grey/` holds a grey copy of each pick, with the pick's name and size. Each one is the pick turned grey with global histogram equalisation (OpenCV's `equalizeHist`), as a JPEG at `jpeg_quality`. Only matching uses the grey copies.
- `mask.png` is one mask for every pick: an 8-bit grey PNG at the pick size that holds only 0 and 255. Black (0) marks a pixel stuck to the camera, and white (255) every other pixel. Each mask pixel copies the score frame pixel under its centre, and nothing grows, shrinks or smooths the stuck pixels. COLMAP takes the mask as its single camera mask, `--ImageReader.camera_mask_path`, and the training loss reads the same file. For both, black means ignore. The step writes the mask only when it uses it: when it has at least 2 mask samples and the stuck pixels cover from `mask_min_coverage` to `mask_max_coverage` of the frame.
- `frames.json` is the frames record, below. The step writes it after every pick, every grey copy and the mask. So a run that fails leaves no `frames.json`, and a work folder without it is unfinished, whatever else it holds.

The step writes nothing before the pick pass. The step runner empties `work/frames/` before the step runs, so a run that fails before the pick pass leaves it empty.

`frames.json` describes the picks and the mask. This example is from a 3840 × 2160 HDR video at 30 frames a second, with the `fast` preset, the part to use from 1.5 s to 61.2 s, and an ffmpeg without `zscale`:

```json
{
  "colour_transfer": "arib-std-b67",
  "tone_mapped": false,
  "still_start_seconds": 2.4,
  "still_end_seconds": 0.0,
  "kept_start_seconds": 3.9,
  "kept_end_seconds": 61.2,
  "target_picks": 120,
  "jolt_filter_skipped": false,
  "pick_width": 1600,
  "pick_height": 900,
  "mask_coverage": 0.0125,
  "mask_used": true,
  "mask_file": "mask.png",
  "picks": [
    {"number": 0, "file": "000000.jpg", "frame_number": 72, "seconds": 3.9,
     "sharpness": 812.5, "motion": 2.1, "contrast": 48.0}
  ]
}
```

| Field | What it holds |
|---|---|
| `colour_transfer` | The video's colour transfer as ffprobe reports it, like `"bt709"` or `"arib-std-b67"`, or `null` when ffprobe reports none. |
| `tone_mapped` | Whether both passes ran with the tone map. |
| `still_start_seconds`, `still_end_seconds` | The seconds the still trim cut at the start and at the end, 0.0 when it cut nothing there. |
| `kept_start_seconds`, `kept_end_seconds` | Where the kept range starts and ends, in seconds. It starts at its first frame's time. It ends at the time of the frame after its last, or at `end_seconds` when no frame comes after it. |
| `target_picks` | The number of parts the kept range was split into, so the most picks the step could make. |
| `jolt_filter_skipped` | Whether the step skipped the jolt filter. |
| `pick_width`, `pick_height` | The pick size, in pixels, which is also the grey copies' and the mask's size. |
| `mask_coverage` | The share of a score frame's pixels that are stuck, from 0 to 1. 0.0 with fewer than 2 mask samples. |
| `mask_used` | Whether the step uses the mask. |
| `mask_file` | `"mask.png"` when the step uses the mask, else `null`. |
| `picks` | One object per pick, in time order. |

Each object in `picks` holds:

| Field | What it holds |
|---|---|
| `number` | The pick's number in time order, from 0. |
| `file` | Its file name in both `images/` and `grey/`, like `"000000.jpg"`. |
| `frame_number` | Its score frame's place among the score frames of the part of the video to use, from 0. |
| `seconds` | Its time, in seconds from the start of the video, on the same clock as `start_seconds` and `end_seconds`. |
| `sharpness` | The variance of its score frame's Laplacian. |
| `motion` | Its score frame's mean absolute change in brightness from the score frame before, from 0 to 255, or 0 for the first score frame. |
| `contrast` | The standard deviation of its score frame's brightness, from 0 to 127.5. |

Every value is plain JSON. Floats keep full precision, because the camera solve step may match on `seconds`.

### Config section

| Key | Table | Value | What it is |
|---|---|---|---|
| `score_width` | `[frames]` | `320` | The width of the grey score frames, in pixels. Their height keeps the video's shape, so a narrower video is scaled up for scoring. |
| `tonemap_npl` | `[frames]` | `100` | The nominal peak brightness, in nits, that `zscale` maps an HDR video from. |
| `tonemap_desat` | `[frames]` | `0.0` | How much `tonemap` desaturates an HDR video's bright highlights. 0 turns it off. |
| `still_motion` | `[frames]` | `1.0` | A frame is still when the mean motion over its window is below this, on the 0 to 255 brightness scale. |
| `motion_window_seconds` | `[frames]` | `1` | The length of the window for the still test, in seconds. |
| `min_still_seconds` | `[frames]` | `2` | The shortest still start or still end that the still trim cuts, in seconds. |
| `blip_seconds` | `[frames]` | `0.5` | A run of frames that aren't still counts as still when it's shorter than this many seconds. |
| `jolt_ratio` | `[frames]` | `0.6` | The jolt filter keeps a frame when its sharpness is at least this share of the median sharpness over its window. |
| `jolt_window_seconds` | `[frames]` | `1` | The length of the window for the jolt test, in seconds. |
| `mask_samples` | `[frames]` | `240` | The most mask samples, the score frames used to find the stuck pixels. |
| `mask_ratio` | `[frames]` | `0.2` | A pixel is stuck when its mean change in brightness, from one mask sample to the next, is below this share of the median over every pixel. |
| `mask_min_coverage` | `[frames]` | `0.002` | The smallest share of the frame the stuck pixels must cover for the mask to be used. |
| `mask_max_coverage` | `[frames]` | `0.25` | The largest share of the frame the stuck pixels may cover for the mask to be used. |
| `jpeg_quality` | `[frames]` | `95` | The JPEG quality of the picks and their grey copies, from 1 to 100. |
| `long_edge` | `[frames]` | `1600` | The long edge of the picks, in pixels. A smaller video isn't scaled up. |
| `seconds_per_pick` | `[frames]` | `2` | Seconds of kept video per pick, when that gives more picks than `base_picks`. |
| `max_picks` | `[frames]` | `400` | The most picks a job gets. |
| `min_picks` | `[frames]` | `30` | The fewest picks a job can go on with. With fewer, the step fails before the pick pass, and its readable error says what to film differently. |
| `min_vcpus` | `[frames]` | `16` | The fewest vCPUs the job's cloud machine needs for this step. |
| `min_memory_gb` | `[frames]` | `16` | The least memory the job's cloud machine needs for this step, in GB. |
| `ffmpeg_timeout_seconds` | `[frames]` | `1800` | The longest time each ffmpeg pass over the video may take, in seconds. |
| `ffprobe_timeout_seconds` | `[frames]` | `30` | The longest time ffprobe, or ffmpeg listing its filters, may take, in seconds. |
| `base_picks` | `[frames.fast]`, `[frames.default]` and `[frames.quality]` | `120`, `150` and `200` | The fewest picks the step aims for. |

- A frame's window holds the frames whose time is at most half the window's length before or after its own, the frame itself included. The still test takes them from every frame of the part of the video to use, and the jolt test from the kept range only.
- `target_picks` is min(`max_picks`, max(`base_picks`, ceil(kept seconds / `seconds_per_pick`))), where the kept seconds are `kept_end_seconds` minus `kept_start_seconds`. So with the `fast` preset, 240 s of kept video gives a `target_picks` of 120, and 420 s gives 210.
- The step checks every setting before ffprobe or ffmpeg runs, `tonemap_npl` and `tonemap_desat` included, though only an HDR video uses them. A value of the wrong kind fails the step with a readable error that names the key, its table and what it must be. A bool is never a number here. The step also fails when `mask_min_coverage` is greater than `mask_max_coverage`, `min_picks` greater than `base_picks`, or `base_picks` greater than `max_picks`, naming both settings and their values.
- `[frames]` has its own `ffprobe_timeout_seconds`, apart from the one in `[jobs]`, because a step sees only its own section.

`min_vcpus` and `min_memory_gb` are the step's machine needs: the job's cloud machine needs at least 16 vCPUs and 16 GB of memory for this step. The cloud spec reads them to pick the machine. The step itself never checks the machine, so it runs the same on a laptop. It checks the two settings like the others, but never uses them.

### Step report fields

- `settings`: the 22 settings in `[frames]` and the preset's `base_picks`.
- `timings.probe`: ffprobe, and for an HDR video the check for the `zscale` and `tonemap` filters.
- `timings.score_pass`: the score pass with its scoring, its second run included when ffmpeg failed with the tone map.
- `timings.choose`: the still trim, the jolt filter, the picks and the mask.
- `timings.pick_pass`: the pick pass with its JPEG writes.
- `counts.pick_width` and `counts.pick_height`: the pick size, in pixels.
- `counts.decoded_frames`: the score frames in the part of the video to use.
- `counts.still_start_frames` and `counts.still_end_frames`: the frames the still trim cut at the start and at the end.
- `counts.kept_frames`: the frames in the kept range.
- `counts.target_picks`: the number of parts the kept range was split into.
- `counts.jolt_dropped_frames`: the frames the jolt filter dropped, 0 when the step skipped it.
- `counts.picks`: the picks the step chose.
- `counts.mask_samples`: the mask samples the step used to find the stuck pixels.
- `counts.mask_pixels`: the stuck pixels, at the score frames' size.
- `counts.mask_used`: 1 when the step uses the mask, 0 when it doesn't.
- `warnings`, each only when it applies:
  - For an HDR video when ffmpeg has no `zscale` or `tonemap` filter: "The video is HDR (colour transfer <transfer>), but this ffmpeg has no zscale or tonemap filter, so the frames were read without tone mapping and the picks' colours may look flat. An ffmpeg built with zscale fixes the colours."
  - For an HDR video when ffmpeg failed in the score pass with the tone map, with ffmpeg's last line of error output: "The video is HDR (colour transfer <transfer>), but ffmpeg failed to tone map it (<last line>), so the frames were read without tone mapping and the picks' colours may look flat. An ffmpeg built with a working zscale fixes the colours."
  - When the step skipped the jolt filter: "The jolt filter kept only <k> frames, fewer than the <n> picks this job aims for, so it was skipped and every kept frame could be picked."
- `error`, when it failed. The step raises `ExpectedFailure` for each failure it foresees, so `expected` is `true` and `readable` is the step's own message, saying what went wrong and what to do. Those failures are a bad setting, a missing ffmpeg or ffprobe, a tool that runs past its time limit, fails or prints output the step can't read, a video with no picture size, no frames in the part of the video to use, fewer picks than `min_picks`, a pick that ffmpeg doesn't give back in the pick pass, and a file that can't be written. Any other error is a bug, with the general message.

The counts are whole numbers, so the seconds the still trim cut and the mask coverage are in `frames.json` instead. A failed report holds the counts and timings the step had worked out before it failed.

### Files in out/

The frames step writes no files to `out/` but its report, `out/reports/frames.json`, which the step runner writes, like every step's.
