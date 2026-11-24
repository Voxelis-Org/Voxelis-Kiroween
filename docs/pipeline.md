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

The list holds the frames step and then the cameras step, both required: `StepEntry("frames", "splatpipe.frames", True)` and `StepEntry("cameras", "splatpipe.cameras", True)`. Each later step spec adds its own entry after the cameras step. Until the training spec adds its entry, a job is done once the cameras step is done. The planned steps are:

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

## Step: cameras

The cameras step is in `splatpipe/cameras/`, built like the fake step and the frames step. It solves the cameras for the picks that the frames step left in `work/frames/`: where each pick was taken from, and the one lens that every pick shares. It runs COLMAP 4.2.1 on the CPU, then levels and centres the scene. It saves the cameras, the points and the undistorted picks in its work folder, `work/cameras/`, for the training step, and the cameras alone in `out/cameras.json` for the web viewer.

Its CLI is `uv run python -m splatpipe.cameras --job <id> [--preset P]`, and its plain function is `splatpipe.cameras.run(job_id, preset=None)`.

It works in this order:

1. It removes an earlier `out/cameras.json`, before anything else.
2. It checks every setting, before it reads `frames.json` or runs any COLMAP command.
3. It reads the frames record, `work/frames/frames.json`, and checks that every pick's colour pick and grey copy is there, and the mask when the record says the frames step used one. Then it checks that the program `colmap` is on the PATH. All of this comes before any COLMAP command runs.
4. Feature extraction: `colmap feature_extractor` finds SIFT features in the grey copies and saves them in a new database, with one camera of `camera_model` for every pick. With the mask, COLMAP drops the features on the mask's black pixels.
5. Matching: `colmap sequential_matcher` runs twice on the same database. The near match run pairs each pick with the next `near_overlap` picks. Then the far match run pairs each pick i with pick i + 2^k, for k from 0 to `far_overlap` − 1, and COLMAP matches only the pairs the database doesn't hold yet.
6. Calibration, for `"OPENCV"` only: the step copies the database to the calibration copy, then runs `colmap view_graph_calibrator` on the copy, which works out the focal length from the matches, because the picks carry no EXIF. The calibration has failed when it exits with an error, rejects its focal length estimate or finds no image pairs to calibrate. `view_graph_calibrator` only fits pinhole cameras, so for `"OPENCV_FISHEYE"` the step skips it, and the camera starts from `fisheye_fov_degrees` instead.
7. Mapping: unless the calibration failed, `colmap global_mapper` runs, on the calibration copy, or on the database for `"OPENCV_FISHEYE"`. It gives up when it exits with an error, writes no model, or its kept model fails the placed check. After a failed calibration, or when the global mapper gives up, the step runs the fallback once: `colmap mapper`, the incremental mapper, on the database.
8. The placed check: the final model must place at least `min_placed_frames` picks, and at least `min_placed_share` of them. Otherwise the step fails here, before the undistortion, with a readable error that says how to film the place again.
9. Undistortion: `colmap image_undistorter` warps the final model's colour picks to one PINHOLE camera with no distortion. With the mask, `colmap image_undistorter_standalone` then warps `mask.png` with the same camera and the same settings, and the step makes the undistorted mask from its output.
10. Levelling: one rigid move, with scale 1, makes the scene level, turns it so the heading camera looks along +Z, and moves the scene centre to the origin. When it can't, the step keeps COLMAP's frame, adds a warning and goes on.
11. It writes the training images, the training masks when the frames step used the mask, the levelled model and `solve.json`, then `out/cameras.json`, last of all.

A mapper can write several models, `0/`, `1/` and so on, when it can't join every pick into one. A placed frame is a pick that a model gives a camera pose. A mapper's kept model is its model with the most placed frames, and the one with the lowest number among those with as many. The final model is the incremental mapper's kept model when the fallback ran, and the global mapper's otherwise.

COLMAP runs as a separate program, never linked or imported. The step runs its commands one at a time, each at most once, and retries nothing. Each command has its own time limit in `[cameras]`, and a command that runs past it fails the step, with no fallback. Every option of each command is on its command line, from `[cameras]` or fixed in code, so no COLMAP default decides anything.

### What it reads

The step reads the frames step's work folder, `work/frames/`:

- `frames.json`, the frames record. Of its fields, the step uses only `pick_width`, `pick_height`, `mask_used` and `picks`, and of each pick only its `number` and its `file`.
- `images/`, the colour picks, and `grey/`, the grey copies: each pick's `images/<file>` and `grey/<file>`.
- `mask.png`, the mask, when the frames record has `mask_used`.

It also reads its settings, from `[cameras]` and from the job's preset table, like `[cameras.fast]`, and it runs COLMAP 4.2.1, the command line program `colmap`.

The step checks the frames record and its files before any COLMAP command runs. A missing `frames.json` means the frames step hasn't finished for the job. A record that can't be read as JSON, a field that's missing or holds the wrong kind of value, an empty `picks`, a pick whose `file` isn't the frames step's name for its `number`, like `000000.jpg` for 0, a `number` given twice, and a missing colour pick, grey copy or mask each fail the step. The readable error names the field or the file, and says to run the frames step with `uv run python -m splatpipe.frames --job <id>`. The number of picks is the number of entries in `picks`, and each image that COLMAP names belongs to the pick whose `file` is that name.

`feature_extractor` reads the grey copies, while `global_mapper`, `mapper` and `image_undistorter` read the colour picks. A grey copy has its colour pick's name and size, and COLMAP knows an image by its name inside the folder it gets as `--image_path`, so to COLMAP both are `000000.jpg`. That way the features come from the evened-out grey copies, and the points' colours and the training images from the colour picks. Each COLMAP command reads and writes these files, with `colmap/` short for `work/cameras/colmap/`:

| Command | Reads | Writes |
|---|---|---|
| `feature_extractor` | The grey copies in `work/frames/grey/`, and with `mask_used`, `work/frames/mask.png` as `--ImageReader.camera_mask_path`. | A new database, `colmap/database.db`. |
| `sequential_matcher`, for both match runs | The database. | The matches, into the database. |
| `view_graph_calibrator` | The calibration copy, `colmap/database_calibrated.db`, which the step makes from the database first. | The focal length, into the calibration copy. |
| `global_mapper` | The calibration copy, or the database for `"OPENCV_FISHEYE"`, and the colour picks in `work/frames/images/`, for the points' colours. | Its models, into `colmap/global/`. |
| `mapper` | The database, and the colour picks in `work/frames/images/`, for the points' colours. | Its models, into `colmap/incremental/`. |
| `image_undistorter` | The final model, and the colour picks in `work/frames/images/`. | The warped colour picks and the undistorted model, into `colmap/undistorted/`. |
| `image_undistorter_standalone` | `work/frames/mask.png`, with the final model's camera from `colmap/mask_camera.txt`. | The warped mask, into `colmap/mask_undistorted/`. |

The step leaves every file in `work/frames/` as it is.

The job's machine image needs COLMAP 4.2.1, built from source without CUDA, OpenGL, the GUI, CGAL and LSD, as `.kiro/steering/tech.md` says, so SiftGPU, CGAL and LSD are never in it. SIFT runs on the CPU, and the step passes `--FeatureExtraction.use_gpu 0` and `--FeatureMatching.use_gpu 0` anyway. The step runs `colmap` through `subprocess` and uses no pycolmap. When `colmap` isn't on the PATH, the step fails before any COLMAP command runs, and its readable error says to install COLMAP 4.2.1 as `.kiro/steering/tech.md` says. Nothing is downloaded while the step runs: SIFT and `SIFT_BRUTEFORCE` need no model weights, and loop detection, which would need a vocabulary tree that COLMAP downloads, stays off.

### What it writes

The step writes only into its work folder, `work/cameras/`, and to `out/cameras.json`. The training step reads `work/cameras/` with its own small reader, with no COLMAP and no pycolmap installed, so the layout below, the model files' layout and the fields of `solve.json` are the contract between the two steps. The names and the layout are fixed in code, not in `config.toml`.

```
work/cameras/
  colmap/                              COLMAP's own files
    tmp/                               TMPDIR and TMP for every COLMAP command
    logs/                              one log per COLMAP command that ran
      feature_extractor.log
      sequential_matcher_near.log
      sequential_matcher_far.log
      view_graph_calibrator.log        OPENCV only
      global_mapper.log                unless the calibration failed
      mapper.log                       only when the fallback ran
      image_undistorter.log
      image_undistorter_standalone.log only with mask_used
    database.db                        the database
    database_calibrated.db             the calibration copy, OPENCV only
    global/                            the global mapper's models, 0/, 1/, ..., and project.ini
    incremental/                       the incremental mapper's models, only when the fallback ran
    undistorted/                       the undistortion's output
      images/000000.jpg ...            the warped colour picks
      sparse/                          the undistorted model, all five files
      stereo/, run-colmap-geometric.sh, run-colmap-photometric.sh
    mask_camera.txt                    the mask warp's camera line, only with mask_used
    mask_undistorted/mask.png          the mask warp's output, only with mask_used
  images/000000.jpg ...                the training images, one per placed frame
  masks/000000.png ...                 the training masks, only with mask_used
  sparse/                              the levelled model, exactly three files
    cameras.bin
    images.bin
    points3D.bin
  solve.json                           the solve record, written last in work/cameras/
out/
  cameras.json                         the viewer cameras, written last of all
```

`work/cameras/` holds nothing else. The step runner empties it before the step runs, so nothing of an earlier run is left in it.

- `colmap/` is the COLMAP folder. It holds COLMAP's own files, for the team: `tmp/`, `logs/`, `database.db`, `database_calibrated.db`, `global/`, `incremental/`, `undistorted/`, `mask_camera.txt` and `mask_undistorted/`. Every COLMAP command runs with `colmap/` as its working folder and with `TMPDIR` and `TMP` set to `tmp/`, so COLMAP writes nothing outside the work folder. `logs/` holds each command's whole output, also when the command failed or ran past its time limit. The calibration changes only `database_calibrated.db`, so `database.db` stays as the match runs left it, for the incremental mapper. `global/` is there only when the global mapper ran, so not after a failed calibration. `mask_camera.txt` holds the final model's camera, as it was before the undistortion, for the mask warp. The undistortion writes its own `images/` and `sparse/` into `undistorted/`, so none of COLMAP's files land in `work/cameras/sparse/`.
- `images/` holds the training images: each placed frame's warped colour pick, copied from `colmap/undistorted/images/` under its name in `sparse/images.bin`, the pick's `file`, like `000000.jpg`. It holds no other file. With `undistortion_blank_pixels` at 0, the undistortion crops each warped pick so that no pixel is blank, so the undistorted camera's size and principal point can differ a little from the picks'.
- `masks/` holds the training masks, only when the frames record has `mask_used`. There's one per placed frame, named after it without `.jpg`, like `000000.png`, and each is the same undistorted mask: an 8-bit grey PNG at the undistorted camera's size that holds only 0 and 255. Black (0) tells the training loss to ignore a pixel, and white (255) to use it. A pixel is white only where every channel of the mask warp's output is 255, so the grey edges that the warp's bilinear sampling leaves around the black parts come out black. Without `mask_used` there's no `masks/` folder.
- `sparse/` is the levelled model: exactly `cameras.bin`, `images.bin` and `points3D.bin`, in COLMAP 4.2.1's binary layout, below, with no `rigs.bin`, no `frames.bin` and no other file. COLMAP 4.2.1 reads the folder as it is, because it reads `rigs.bin` and `frames.bin` only when they exist. `cameras.bin` holds exactly one camera, the undistorted camera, of the model PINHOLE, and every image uses it. `images.bin` holds one image per placed frame, with the undistorted model's image id, camera id and 2D points, its pose moved by the level move, and its name, the pick's `file`. So the image `000000.jpg` has its training image at `work/cameras/images/000000.jpg`. `points3D.bin` holds every point of the undistorted model, with its id, colour, error and track, and its position moved by the level move.
- `solve.json` is the solve record, below. The step writes it after every training image, every training mask and the levelled model. So a run that fails in the step's code leaves no `solve.json`, and the training step can treat a work folder without it as unfinished, whatever else it holds.

The levelling gives the levelled model its frame. The level move is one rigid move, a rotation and then a translation, with scale 1, and the step applies the whole of it to every pose and every point at once:

1. It averages every placed frame's up vector, Rᵀ(0, −1, 0), the direction that's up in the camera's picture. When the mean is at least `up_agreement` long, the cameras agree, and the move rotates the mean onto the world up, (0, −1, 0).
2. It turns the scene about the world up, so the heading camera's heading points along +Z. A camera's heading is its view direction, Rᵀ(0, 0, 1), after that rotation, with its vertical part taken away. A camera has a heading when it looks at least `heading_min_degrees` away from straight up and from straight down, and the heading camera is the placed frame with the lowest pick number that has one.
3. It moves the scene centre to the origin. The scene centre is the axis point, the point nearest every placed frame's view line, when that point is well defined: the smallest eigenvalue of Σ(I − d dᵀ), over the placed frames' view directions d, is at least `axis_point_min_eigen_share` of the largest, and the point lies in front of at least `axis_point_min_front_share` of the placed frames. Otherwise the scene centre is the point median: the median of the points' x, of their y and of their z, each taken separately along the levelled axes.

When the cameras' mean up vector is shorter than `up_agreement`, no placed frame has a heading, the axis point isn't well defined and the model has no points, or working out the move fails in any other way, the step keeps COLMAP's frame. Then the level move is the identity, every pose and point stays exactly as the undistorted model holds it, and the report gets a warning. The levelling never fails the step.

The model files are in COLMAP 4.2.1's binary layout, as COLMAP's source at the `4.2.1` tag writes it in `src/colmap/scene/reconstruction_io_binary.cc`, and the training step's reader parses exactly this layout. The integration test checks the step's files against COLMAP itself: `colmap model_converter` writes the same bytes from `sparse/`. Every number is little-endian, records follow each other with no padding, and there's no header beyond each file's count.

`cameras.bin` holds:

| Field | Type | Notes |
|---|---|---|
| `num_cameras` | `uint64` | 1 in the levelled model. |
| then, per camera: `camera_id` | `uint32` | |
| `model_id` | `int32` | 1 PINHOLE, 4 OPENCV, 5 OPENCV_FISHEYE. The levelled model holds PINHOLE only. |
| `width`, `height` | `uint64`, `uint64` | In pixels. |
| `params` | `float64` × the model's count | PINHOLE: fx, fy, cx, cy. OPENCV: fx, fy, cx, cy, k1, k2, p1, p2. OPENCV_FISHEYE: fx, fy, cx, cy, k1, k2, k3, k4. |

Cameras are in increasing `camera_id` order. A PINHOLE record is 56 bytes.

`images.bin` holds:

| Field | Type | Notes |
|---|---|---|
| `num_reg_images` | `uint64` | Only registered images are written, so this is the number of placed frames. |
| then, per image: `image_id` | `uint32` | |
| `qw`, `qx`, `qy`, `qz` | `float64` × 4 | The cam_from_world rotation as a unit quaternion, in Hamilton's convention. |
| `tx`, `ty`, `tz` | `float64` × 3 | The cam_from_world translation. |
| `camera_id` | `uint32` | The one camera's id. |
| `name` | bytes, then one `0` byte | UTF-8 with no length in front, like `000000.jpg`. Its training image is `work/cameras/images/<name>`. |
| `num_points2D` | `uint64` | |
| then, per 2D point: `x`, `y` | `float64`, `float64` | In pixels. The upper-left image corner is (0, 0), so the upper-left pixel's centre is (0.5, 0.5). The undistortion writes NaN for a point it couldn't move. |
| `point3D_id` | `uint64` | 18446744073709551615 (2⁶⁴ − 1) when the 2D point has no 3D point. |

Images are in the order the model registered them, which COLMAP keeps when it reads and writes a model. An image record is 64 bytes, plus its name and the `0` byte, plus 8, plus 24 bytes per 2D point.

`points3D.bin` holds:

| Field | Type | Notes |
|---|---|---|
| `num_points3D` | `uint64` | |
| then, per point: `point3D_id` | `uint64` | |
| `x`, `y`, `z` | `float64` × 3 | In the levelled world. |
| `r`, `g`, `b` | `uint8` × 3 | From the colour picks. |
| `error` | `float64` | The mean reprojection error in pixels, -1 for none. |
| `track_length` | `uint64` | |
| then, per track element: `image_id` | `uint32` | |
| `point2D_idx` | `uint32` | The 2D point's place in that image's list, from 0. |

Points are in increasing `point3D_id` order. A point record is 51 bytes plus 8 per track element.

The poses follow COLMAP's convention. A world point x is at R x + t in the camera, with R the rotation of (qw, qx, qy, qz) and t = (tx, ty, tz). The camera looks along its +z axis, with +x to the right and +y down in the picture. Its centre is −Rᵀ t. When `solve.json` has `"levelled": true`, the world's up is (0, −1, 0), the heading camera's view, flattened, points along +Z, and the scene centre is at the origin, with the scale COLMAP solved. When it has `"levelled": false`, the poses and points are in COLMAP's frame, which may be tilted.

`solve.json` says how the cameras were solved and levelled. In this example, the calibration and the global mapper worked, and the numbers are rounded:

```json
{
  "camera_model": "OPENCV",
  "calibration_ran": true,
  "calibration_failure": null,
  "mapper": "global",
  "fallback_reason": null,
  "models": [{"number": 0, "placed_frames": 118}, {"number": 1, "placed_frames": 4}],
  "final_model": 0,
  "mean_reprojection_error": 0.84,
  "levelled": true,
  "level_reason": null,
  "mean_up_length": 0.97,
  "scene_centre": "axis_point",
  "heading_camera": "000000.jpg",
  "rotation": [[0.99, 0.01, -0.12], [-0.01, 1.0, 0.02], [0.12, -0.02, 0.99]],
  "translation": [-0.41, 0.12, -2.3]
}
```

| Field | What it holds |
|---|---|
| `camera_model` | The camera model, `"OPENCV"` or `"OPENCV_FISHEYE"`. |
| `calibration_ran` | Whether the calibration ran: `false` for `"OPENCV_FISHEYE"`. |
| `calibration_failure` | Why the calibration failed, like `"rejected its focal length estimate and kept the starting one"`, or `null` when it didn't. |
| `mapper` | Which mapper made the final model: `"global"` or `"incremental"`. |
| `fallback_reason` | Why the fallback ran: the failed calibration's reason or the global mapper's, as in its warning. `null` when it didn't run. |
| `models` | Each model of the mapper run that made the final model, in number order, with its `number` and its `placed_frames`. |
| `final_model` | The final model's number. |
| `mean_reprojection_error` | The mean of the final model's point errors that aren't -1, in pixels, or `null` when no point has one. |
| `levelled` | Whether the levelling moved the scene. |
| `level_reason` | The warning when the levelling kept COLMAP's frame, else `null`. |
| `mean_up_length` | The length of the cameras' mean up vector, from 0 to 1, or `null` when the input wasn't finite. |
| `scene_centre` | `"axis_point"`, `"point_median"`, or `null` when COLMAP's frame was kept. |
| `heading_camera` | The heading camera's name, or `null`. |
| `rotation`, `translation` | The level move, x' = R x + T, as a 3 × 3 list of rows and a list of 3. The identity and zeros when COLMAP's frame was kept. |

Every value is plain JSON, and floats keep full precision.

### Config section

`[cameras]` holds 190 settings, and each preset table holds one more, `near_overlap`. The step's own settings come first:

| Key | Table | Value | What it is |
|---|---|---|---|
| `camera_model` | `[cameras]` | `"OPENCV"` | COLMAP's `--ImageReader.camera_model`: the camera every pick shares, `"OPENCV"`, or `"OPENCV_FISHEYE"` for very wide lenses. An `"OPENCV"` camera's focal length starts at `image_reader_default_focal_length_factor` times the picks' long edge, with no focal prior, and the calibration works it out. |
| `fisheye_fov_degrees` | `[cameras]` | `120.0` | The field of view across the picks' long edge, in degrees, that an `"OPENCV_FISHEYE"` camera starts from, passed as `--ImageReader.camera_params`: fx = fy = (L / 2) / (θ / 2), with L the long edge in pixels and θ this angle in radians, cx and cy at the picks' centre, and k1 to k4 at 0. That gives the camera the focal prior COLMAP needs to keep fisheye pairs. 120° is a guess until we have real fisheye clips. |
| `far_overlap` | `[cameras]` | `10` | COLMAP's `--SequentialMatching.overlap` in the far match run: pick i meets pick i + 2^k, for k from 0 to this − 1. Keep it at 9 or more, so that with 400 picks the far match run reaches 256 picks ahead. |
| `min_placed_frames` | `[cameras]` | `20` | The fewest picks the final model must place. |
| `min_placed_share` | `[cameras]` | `0.5` | The smallest share of the picks, from 0 to 1, that the final model must place. |
| `up_agreement` | `[cameras]` | `0.5` | The shortest mean up vector, from 0 to 1, at which the cameras agree on which way is up. Below it, the levelling keeps COLMAP's frame. |
| `axis_point_min_eigen_share` | `[cameras]` | `0.1` | The axis point counts only when the smallest eigenvalue of Σ(I − d dᵀ) is at least this share of the largest. |
| `axis_point_min_front_share` | `[cameras]` | `0.5` | The axis point counts only when it's in front of at least this share of the placed frames. |
| `heading_min_degrees` | `[cameras]` | `10.0` | A camera that looks within this many degrees of straight up or down has no heading. |
| `feature_extractor_timeout_seconds` | `[cameras]` | `900` | The longest time `colmap feature_extractor` may run, in seconds. |
| `sequential_matcher_timeout_seconds` | `[cameras]` | `1800` | The longest time each of the two `colmap sequential_matcher` runs may take, in seconds. |
| `view_graph_calibrator_timeout_seconds` | `[cameras]` | `600` | The longest time `colmap view_graph_calibrator` may run, in seconds. |
| `global_mapper_timeout_seconds` | `[cameras]` | `1800` | The longest time `colmap global_mapper` may run, in seconds. |
| `mapper_timeout_seconds` | `[cameras]` | `3600` | The longest time `colmap mapper`, the incremental mapper, may run, in seconds. |
| `image_undistorter_timeout_seconds` | `[cameras]` | `600` | The longest time `colmap image_undistorter` may run, in seconds. |
| `image_undistorter_standalone_timeout_seconds` | `[cameras]` | `120` | The longest time `colmap image_undistorter_standalone`, which warps the mask, may run, in seconds. |
| `min_vcpus` | `[cameras]` | `16` | The vCPUs the job's cloud machine should have for this step, a soft need. |
| `min_memory_gb` | `[cameras]` | `16` | The least memory the job's cloud machine needs for this step, in GB, a hard need. |
| `near_overlap` | `[cameras.fast]`, `[cameras.default]` and `[cameras.quality]` | `15`, `15` and `25` | COLMAP's `--SequentialMatching.overlap` in the near match run: each pick meets the next this many picks. |

Every other key in `[cameras]`, like `camera_model` above, sets one COLMAP option, which its comment in `config.toml` names. These option keys are named after their options. `--Group.name` becomes `<group>_<name>`, with the group in snake case and the whole key in lower case, so `--SiftExtraction.max_num_features` is `sift_extraction_max_num_features`, and `--GlobalMapper.ba_refine_points3D` is `global_mapper_ba_refine_points3d`. An option with no group gets its command's name in front, like `feature_extractor_descriptor_normalization` for `--descriptor_normalization`. The undistortion settings, which both undistorters take, start with `undistortion_`, and `--default_random_seed`, which every command takes, keeps its own name. Each value is the one in the colmap-cli skill's `.kiro/skills/colmap-cli/references/options.md`, which is COLMAP 4.2.1's default unless that file's notes say otherwise, except `camera_model`, which is `"OPENCV"` where that file has COLMAP's default, `SIMPLE_RADIAL`. A decimal option holds a TOML float, like `10.0` where that file writes `10`. The step writes each value on the command line so that COLMAP reads the same value: a bool as `1` or `0`, a whole number in decimal, a decimal with Python's `repr`, the shortest text that reads back as the same double, and a name as it is.

Every COLMAP command takes:

| Key | COLMAP option | Value | What it does |
|---|---|---|---|
| `default_random_seed` | `--default_random_seed` | `0` | The seed each COLMAP thread's random generator starts from. |

`colmap feature_extractor` takes these, and `camera_model` as `--ImageReader.camera_model`:

| Key | COLMAP option | Value | What it does |
|---|---|---|---|
| `feature_extractor_descriptor_normalization` | `--descriptor_normalization` | `"l1_root"` | How SIFT descriptors are normalised, `"l1_root"` or `"l2"`. |
| `image_reader_existing_camera_id` | `--ImageReader.existing_camera_id` | `-1` | -1 makes a new camera instead of reusing one in the database. |
| `image_reader_default_focal_length_factor` | `--ImageReader.default_focal_length_factor` | `1.2` | The starting focal length, as this times the long edge. |
| `feature_extraction_num_threads` | `--FeatureExtraction.num_threads` | `-1` | Threads for SIFT, -1 for every core. |
| `feature_extraction_max_image_size` | `--FeatureExtraction.max_image_size` | `3200` | Longer picks are scaled down to this before SIFT, in pixels. |
| `sift_extraction_max_num_features` | `--SiftExtraction.max_num_features` | `8192` | Most SIFT features kept per pick. |
| `sift_extraction_first_octave` | `--SiftExtraction.first_octave` | `-1` | -1 doubles each pick first, which finds more small features. |
| `sift_extraction_num_octaves` | `--SiftExtraction.num_octaves` | `4` | Octaves in SIFT's scale space. |
| `sift_extraction_octave_resolution` | `--SiftExtraction.octave_resolution` | `3` | Levels per octave. |
| `sift_extraction_peak_threshold` | `--SiftExtraction.peak_threshold` | `0.006666666666666667` | Contrast a feature needs, 0.02 / 3. Change it with `sift_extraction_octave_resolution`. |
| `sift_extraction_edge_threshold` | `--SiftExtraction.edge_threshold` | `10.0` | Features more edge-like than this are dropped. |
| `sift_extraction_estimate_affine_shape` | `--SiftExtraction.estimate_affine_shape` | `false` | Fit an affine shape to each feature. Slower, more matches. |
| `sift_extraction_max_num_orientations` | `--SiftExtraction.max_num_orientations` | `2` | Most orientations per feature. |
| `sift_extraction_upright` | `--SiftExtraction.upright` | `false` | Give every feature the upright orientation. |
| `sift_extraction_domain_size_pooling` | `--SiftExtraction.domain_size_pooling` | `false` | DSP-SIFT descriptors. Slower, more matches. |
| `sift_extraction_dsp_min_scale` | `--SiftExtraction.dsp_min_scale` | `0.16666666666666666` | Smallest scale DSP-SIFT pools over, 1 / 6. |
| `sift_extraction_dsp_max_scale` | `--SiftExtraction.dsp_max_scale` | `3.0` | Largest scale DSP-SIFT pools over. |
| `sift_extraction_dsp_num_scales` | `--SiftExtraction.dsp_num_scales` | `10` | Scales DSP-SIFT pools over. |

It also gets `--ImageReader.camera_mask_path` with `work/frames/mask.png` when the frames record has `mask_used`, and `--ImageReader.camera_params` with the fisheye start when `camera_model` is `"OPENCV_FISHEYE"`. Each is left out otherwise.

Both `colmap sequential_matcher` runs take these, with the same values:

| Key | COLMAP option | Value | What it does |
|---|---|---|---|
| `feature_matching_num_threads` | `--FeatureMatching.num_threads` | `-1` | Threads for matching, -1 for every core. |
| `feature_matching_guided_matching` | `--FeatureMatching.guided_matching` | `false` | A second pass guided by each pair's geometry. Slower, more matches. |
| `feature_matching_skip_geometric_verification` | `--FeatureMatching.skip_geometric_verification` | `false` | Keep the raw matches without checking their geometry. |
| `feature_matching_rig_verification` | `--FeatureMatching.rig_verification` | `false` | Check pairs as camera rigs. Each pick here is its own frame. |
| `feature_matching_skip_image_pairs_in_same_frame` | `--FeatureMatching.skip_image_pairs_in_same_frame` | `false` | Don't match the images of one rig frame. |
| `feature_matching_max_num_matches` | `--FeatureMatching.max_num_matches` | `32768` | Most matches kept per pair. |
| `sift_matching_max_ratio` | `--SiftMatching.max_ratio` | `0.8` | Ratio test between the best and the second-best match. |
| `sift_matching_max_distance` | `--SiftMatching.max_distance` | `0.7` | Largest descriptor distance a match may have. |
| `sift_matching_cross_check` | `--SiftMatching.cross_check` | `true` | Keep only the matches that agree both ways. |
| `sift_matching_cpu_brute_force_matcher` | `--SiftMatching.cpu_brute_force_matcher` | `false` | Compare every pair of descriptors instead of using the faiss index. |
| `two_view_geometry_min_num_inliers` | `--TwoViewGeometry.min_num_inliers` | `15` | Pairs with fewer inliers are marked degenerate. |
| `two_view_geometry_multiple_models` | `--TwoViewGeometry.multiple_models` | `false` | Look for more than one geometry per pair. |
| `two_view_geometry_compute_relative_pose` | `--TwoViewGeometry.compute_relative_pose` | `false` | Work out each pair's relative pose while checking it. |
| `two_view_geometry_detect_watermark` | `--TwoViewGeometry.detect_watermark` | `true` | Mark the pairs whose matches look like a watermark. |
| `two_view_geometry_multiple_ignore_watermark` | `--TwoViewGeometry.multiple_ignore_watermark` | `true` | With `two_view_geometry_multiple_models`, leave out watermark geometries. |
| `two_view_geometry_watermark_detection_max_error` | `--TwoViewGeometry.watermark_detection_max_error` | `4.0` | Inlier threshold of the watermark test, in pixels. |
| `two_view_geometry_filter_stationary_matches` | `--TwoViewGeometry.filter_stationary_matches` | `false` | Drop matches that sit at the same pixel in both picks. |
| `two_view_geometry_stationary_matches_max_error` | `--TwoViewGeometry.stationary_matches_max_error` | `4.0` | How close counts as the same pixel, in pixels. |
| `two_view_geometry_use_degensac` | `--TwoViewGeometry.use_degensac` | `false` | Use DEGENSAC for the fundamental matrix. |
| `two_view_geometry_use_sampson_refinement` | `--TwoViewGeometry.use_sampson_refinement` | `true` | Refine each geometry on the Sampson error. |
| `two_view_geometry_max_error` | `--TwoViewGeometry.max_error` | `4.0` | RANSAC's inlier threshold, in pixels. |
| `two_view_geometry_confidence` | `--TwoViewGeometry.confidence` | `0.999` | RANSAC's confidence, above 0 and at most 1. |
| `two_view_geometry_max_num_trials` | `--TwoViewGeometry.max_num_trials` | `10000` | Most RANSAC trials per pair. |
| `two_view_geometry_min_inlier_ratio` | `--TwoViewGeometry.min_inlier_ratio` | `0.0` | Pairs with a smaller inlier share are marked degenerate. 0 is off. |
| `two_view_geometry_random_seed` | `--TwoViewGeometry.random_seed` | `0` | RANSAC's seed, so the pair checks repeat. -1 uses the thread generators. |
| `sequential_matching_expand_rig_images` | `--SequentialMatching.expand_rig_images` | `true` | Also match each rig frame's other images. Each pick is its own frame. |

`--SequentialMatching.overlap` comes from `near_overlap` in the near match run, and from `far_overlap` in the far match run.

`colmap view_graph_calibrator` takes these:

| Key | COLMAP option | Value | What it does |
|---|---|---|---|
| `view_graph_calibrator_cross_validate_prior_focal_lengths` | `--cross_validate_prior_focal_lengths` | `true` | Check trusted focal lengths against the pairs. |
| `view_graph_calibrator_min_calibrated_pair_ratio` | `--min_calibrated_pair_ratio` | `0.5` | Share of a camera's pairs that must be calibrated already for the check above. |
| `view_graph_calibrator_reestimate_relative_pose` | `--reestimate_relative_pose` | `true` | Refit each kept pair's relative pose with the new focal length. |
| `view_graph_calibrator_min_focal_length_ratio` | `--min_focal_length_ratio` | `0.1` | Estimates below this share of the starting focal length are rejected. |
| `view_graph_calibrator_max_focal_length_ratio` | `--max_focal_length_ratio` | `10.0` | Estimates above this many times the starting focal length are rejected. |
| `view_graph_calibrator_max_calibration_error` | `--max_calibration_error` | `2.0` | Pairs with a larger calibration error are marked degenerate. |
| `view_graph_calibrator_relpose_max_error` | `--relpose_max_error` | `1.0` | Inlier threshold of the relative pose refit, in pixels. |
| `view_graph_calibrator_relpose_min_num_inliers` | `--relpose_min_num_inliers` | `30` | Fewest inliers a refitted pair keeps. |
| `view_graph_calibrator_relpose_min_inlier_ratio` | `--relpose_min_inlier_ratio` | `0.25` | Smallest inlier share a refitted pair keeps. |

`colmap global_mapper` takes these:

| Key | COLMAP option | Value | What it does |
|---|---|---|---|
| `global_mapper_min_num_matches` | `--GlobalMapper.min_num_matches` | `15` | Pairs with fewer inlier matches are left out. |
| `global_mapper_ignore_watermarks` | `--GlobalMapper.ignore_watermarks` | `false` | Leave out the pairs marked as watermarks. |
| `global_mapper_num_threads` | `--GlobalMapper.num_threads` | `-1` | Threads, -1 for every core. |
| `global_mapper_random_seed` | `--GlobalMapper.random_seed` | `0` | Seed of rotation averaging, positioning and retriangulation. -1 uses the clock. |
| `global_mapper_decompose_relative_pose` | `--GlobalMapper.decompose_relative_pose` | `true` | Work out each pair's relative pose from its geometry first. |
| `global_mapper_multiple_models` | `--GlobalMapper.multiple_models` | `true` | One model per connected part of the view graph. |
| `global_mapper_min_model_size` | `--GlobalMapper.min_model_size` | `3` | Models with fewer placed frames are dropped. |
| `global_mapper_ba_num_iterations` | `--GlobalMapper.ba_num_iterations` | `3` | Rounds of bundle adjustment. |
| `global_mapper_skip_rotation_averaging` | `--GlobalMapper.skip_rotation_averaging` | `false` | Skip rotation averaging. |
| `global_mapper_skip_track_establishment` | `--GlobalMapper.skip_track_establishment` | `false` | Skip building the tracks. |
| `global_mapper_skip_global_positioning` | `--GlobalMapper.skip_global_positioning` | `false` | Skip global positioning. |
| `global_mapper_skip_bundle_adjustment` | `--GlobalMapper.skip_bundle_adjustment` | `false` | Skip bundle adjustment. |
| `global_mapper_skip_retriangulation` | `--GlobalMapper.skip_retriangulation` | `false` | Skip retriangulation. |
| `global_mapper_track_intra_image_consistency_threshold` | `--GlobalMapper.track_intra_image_consistency_threshold` | `10.0` | Most pixels between a track's points in one picture. |
| `global_mapper_track_required_tracks_per_view` | `--GlobalMapper.track_required_tracks_per_view` | `2147483647` | Tracks per frame that end the track building. 2147483647 is no limit. |
| `global_mapper_track_min_num_views_per_track` | `--GlobalMapper.track_min_num_views_per_track` | `3` | Fewest frames a track must be seen in. |
| `global_mapper_track_max_num_views_per_track` | `--GlobalMapper.track_max_num_views_per_track` | `100` | Tracks seen in more frames are dropped. |
| `global_mapper_keep_max_num_tracks` | `--GlobalMapper.keep_max_num_tracks` | `2147483647` | Most tracks kept, the longest first. 2147483647 is no limit. |
| `global_mapper_gp_optimize_positions` | `--GlobalMapper.gp_optimize_positions` | `true` | Global positioning moves the camera positions. |
| `global_mapper_gp_optimize_points` | `--GlobalMapper.gp_optimize_points` | `true` | Global positioning moves the points. |
| `global_mapper_gp_optimize_scales` | `--GlobalMapper.gp_optimize_scales` | `true` | Global positioning fits the scales. |
| `global_mapper_gp_loss_function_scale` | `--GlobalMapper.gp_loss_function_scale` | `0.1` | Scale of the loss that damps outliers in global positioning. |
| `global_mapper_gp_max_num_iterations` | `--GlobalMapper.gp_max_num_iterations` | `100` | Most solver iterations in global positioning. |
| `global_mapper_ba_refine_focal_length` | `--GlobalMapper.ba_refine_focal_length` | `true` | Bundle adjustment refines the focal length. |
| `global_mapper_ba_refine_principal_point` | `--GlobalMapper.ba_refine_principal_point` | `false` | Bundle adjustment refines the principal point. |
| `global_mapper_ba_refine_extra_params` | `--GlobalMapper.ba_refine_extra_params` | `true` | Bundle adjustment refines the distortion. Keep it true for the undistortion. |
| `global_mapper_refine_sensor_from_rig` | `--GlobalMapper.refine_sensor_from_rig` | `true` | Refine each camera's place in its rig. Each pick is its own frame. |
| `global_mapper_ba_refine_rig_from_world` | `--GlobalMapper.ba_refine_rig_from_world` | `true` | Bundle adjustment refines the frames' poses. |
| `global_mapper_ba_refine_points3d` | `--GlobalMapper.ba_refine_points3D` | `true` | Bundle adjustment refines the points. |
| `global_mapper_ba_min_track_length` | `--GlobalMapper.ba_min_track_length` | `3` | Points seen in fewer frames stay out of bundle adjustment. |
| `global_mapper_ba_ceres_loss_function_scale` | `--GlobalMapper.ba_ceres_loss_function_scale` | `1.0` | Scale of the loss that damps outliers in bundle adjustment. |
| `global_mapper_ba_ceres_max_num_iterations` | `--GlobalMapper.ba_ceres_max_num_iterations` | `200` | Most solver iterations per bundle adjustment. |
| `global_mapper_ba_skip_fixed_rotation_stage` | `--GlobalMapper.ba_skip_fixed_rotation_stage` | `false` | Skip the bundle adjustment stage that holds the rotations fixed. |
| `global_mapper_ba_skip_joint_optimization_stage` | `--GlobalMapper.ba_skip_joint_optimization_stage` | `false` | Skip the stage that refines everything together. |
| `global_mapper_tri_complete_max_reproj_error` | `--GlobalMapper.tri_complete_max_reproj_error` | `15.0` | Retriangulation's largest error to complete a track, in pixels. |
| `global_mapper_tri_merge_max_reproj_error` | `--GlobalMapper.tri_merge_max_reproj_error` | `15.0` | Retriangulation's largest error to merge two tracks, in pixels. |
| `global_mapper_tri_min_angle` | `--GlobalMapper.tri_min_angle` | `1.0` | Retriangulation's smallest triangulation angle, in degrees. |
| `global_mapper_ra_use_gravity` | `--GlobalMapper.ra_use_gravity` | `false` | Rotation averaging uses gravity priors. The picks have none. |
| `global_mapper_ra_use_stratified` | `--GlobalMapper.ra_use_stratified` | `true` | With gravity priors, solve the gravity-only pairs first. |
| `global_mapper_ra_max_rotation_error_deg` | `--GlobalMapper.ra_max_rotation_error_deg` | `10.0` | Pairs off the averaged rotations by more are cut, in degrees. 0 keeps all. |
| `global_mapper_ra_reweighting` | `--GlobalMapper.ra_reweighting` | `"UNIFORM"` | How rotation averaging weighs the pairs, `"UNIFORM"` or `"INLIER_MATCH_COUNT"`. |
| `global_mapper_max_angular_reproj_error_deg` | `--GlobalMapper.max_angular_reproj_error_deg` | `1.0` | Global positioning's outlier threshold, in degrees. |
| `global_mapper_max_normalized_reproj_error` | `--GlobalMapper.max_normalized_reproj_error` | `0.01` | Bundle adjustment's outlier threshold, in normalised image units. |
| `global_mapper_min_tri_angle_deg` | `--GlobalMapper.min_tri_angle_deg` | `1.0` | Smallest triangulation angle a point keeps, in degrees. |

`colmap mapper`, the incremental mapper, takes these:

| Key | COLMAP option | Value | What it does |
|---|---|---|---|
| `mapper_min_num_matches` | `--Mapper.min_num_matches` | `15` | Pairs with fewer inlier matches are left out. |
| `mapper_ignore_watermarks` | `--Mapper.ignore_watermarks` | `false` | Leave out the pairs marked as watermarks. |
| `mapper_multiple_models` | `--Mapper.multiple_models` | `true` | Start a new model from the picks the first couldn't place. |
| `mapper_max_num_models` | `--Mapper.max_num_models` | `50` | Most models. |
| `mapper_max_model_overlap` | `--Mapper.max_model_overlap` | `20` | Most frames a new model may share with the earlier ones. |
| `mapper_min_model_size` | `--Mapper.min_model_size` | `10` | Later models with fewer placed frames are dropped. |
| `mapper_init_image_id1` | `--Mapper.init_image_id1` | `-1` | Image id of the starting pair's first picture, -1 to let the mapper choose. |
| `mapper_init_image_id2` | `--Mapper.init_image_id2` | `-1` | Image id of the starting pair's second picture, -1 to let the mapper choose. |
| `mapper_init_num_trials` | `--Mapper.init_num_trials` | `200` | Most tries to find a starting pair. |
| `mapper_structure_less_registration_fallback` | `--Mapper.structure_less_registration_fallback` | `true` | Place a frame from its matches alone when its points can't. |
| `mapper_structure_less_registration_only` | `--Mapper.structure_less_registration_only` | `false` | Place frames only from their matches. |
| `mapper_extract_colors` | `--Mapper.extract_colors` | `true` | Give the points their colours from the colour picks. |
| `mapper_num_threads` | `--Mapper.num_threads` | `-1` | Threads, -1 for every core. |
| `mapper_random_seed` | `--Mapper.random_seed` | `0` | The mapper's seed, so runs repeat. -1 uses the thread generators. |
| `mapper_min_focal_length_ratio` | `--Mapper.min_focal_length_ratio` | `0.1` | Frames whose focal length falls below this share of the start are filtered out. |
| `mapper_max_focal_length_ratio` | `--Mapper.max_focal_length_ratio` | `10.0` | Frames whose focal length rises above this many times the start are filtered out. |
| `mapper_max_extra_param` | `--Mapper.max_extra_param` | `1.0` | Frames whose distortion goes past this are filtered out. |
| `mapper_ba_refine_focal_length` | `--Mapper.ba_refine_focal_length` | `true` | Bundle adjustment refines the focal length. |
| `mapper_ba_refine_principal_point` | `--Mapper.ba_refine_principal_point` | `false` | Bundle adjustment refines the principal point. |
| `mapper_ba_refine_extra_params` | `--Mapper.ba_refine_extra_params` | `true` | Bundle adjustment refines the distortion. Keep it true for the undistortion. |
| `mapper_ba_refine_sensor_from_rig` | `--Mapper.ba_refine_sensor_from_rig` | `true` | Refine each camera's place in its rig. Each pick is its own frame. |
| `mapper_ba_local_function_tolerance` | `--Mapper.ba_local_function_tolerance` | `0.0` | Ceres' function tolerance in local bundle adjustment. |
| `mapper_ba_local_max_num_iterations` | `--Mapper.ba_local_max_num_iterations` | `25` | Most solver iterations per local bundle adjustment. |
| `mapper_ba_global_frames_ratio` | `--Mapper.ba_global_frames_ratio` | `1.1` | Growth in frames since the last global bundle adjustment that starts the next. |
| `mapper_ba_global_points_ratio` | `--Mapper.ba_global_points_ratio` | `1.1` | Growth in points since the last global bundle adjustment that starts the next. |
| `mapper_ba_global_frames_freq` | `--Mapper.ba_global_frames_freq` | `500` | Frames added since the last global bundle adjustment that start the next. |
| `mapper_ba_global_points_freq` | `--Mapper.ba_global_points_freq` | `250000` | Points added since the last global bundle adjustment that start the next. |
| `mapper_ba_global_function_tolerance` | `--Mapper.ba_global_function_tolerance` | `0.0` | Ceres' function tolerance in global bundle adjustment. |
| `mapper_ba_global_max_num_iterations` | `--Mapper.ba_global_max_num_iterations` | `50` | Most solver iterations per global bundle adjustment. |
| `mapper_ba_global_max_refinements` | `--Mapper.ba_global_max_refinements` | `5` | Most rounds of global bundle adjustment and filtering. |
| `mapper_ba_global_max_refinement_change` | `--Mapper.ba_global_max_refinement_change` | `0.0005` | The rounds above stop when less than this share of observations change. |
| `mapper_ba_local_max_refinements` | `--Mapper.ba_local_max_refinements` | `2` | Most rounds of local bundle adjustment and filtering. |
| `mapper_ba_local_max_refinement_change` | `--Mapper.ba_local_max_refinement_change` | `0.001` | The rounds above stop when less than this share of observations change. |
| `mapper_ba_min_num_residuals_for_cpu_multi_threading` | `--Mapper.ba_min_num_residuals_for_cpu_multi_threading` | `50000` | Smaller bundle adjustments run on one thread. |
| `mapper_snapshot_frames_freq` | `--Mapper.snapshot_frames_freq` | `0` | Write a snapshot every this many frames, 0 for none. |
| `mapper_fix_existing_frames` | `--Mapper.fix_existing_frames` | `false` | Hold a continued model's frames fixed. The step never continues a model. |
| `mapper_init_min_num_inliers` | `--Mapper.init_min_num_inliers` | `100` | Inliers the starting pair needs. |
| `mapper_init_max_error` | `--Mapper.init_max_error` | `4.0` | RANSAC's inlier threshold for the starting pair, in pixels. |
| `mapper_init_max_forward_motion` | `--Mapper.init_max_forward_motion` | `0.95` | Largest share of the starting pair's motion that may point straight ahead. |
| `mapper_init_min_tri_angle` | `--Mapper.init_min_tri_angle` | `16.0` | Parallax the starting pair needs, in degrees. |
| `mapper_init_max_reg_trials` | `--Mapper.init_max_reg_trials` | `2` | Most tries to start a model from each picture. |
| `mapper_abs_pose_max_error` | `--Mapper.abs_pose_max_error` | `12.0` | RANSAC's inlier threshold to place the next frame, in pixels. |
| `mapper_abs_pose_min_num_inliers` | `--Mapper.abs_pose_min_num_inliers` | `30` | Inliers a frame needs to be placed. |
| `mapper_abs_pose_min_inlier_ratio` | `--Mapper.abs_pose_min_inlier_ratio` | `0.25` | Inlier share a frame needs to be placed. |
| `mapper_filter_max_reproj_error` | `--Mapper.filter_max_reproj_error` | `4.0` | Observations with a larger reprojection error are filtered out, in pixels. |
| `mapper_filter_min_tri_angle` | `--Mapper.filter_min_tri_angle` | `1.5` | Points with a smaller triangulation angle are filtered out, in degrees. |
| `mapper_max_reg_trials` | `--Mapper.max_reg_trials` | `3` | Most tries to place each frame. |
| `mapper_ba_local_num_images` | `--Mapper.ba_local_num_images` | `6` | Frames in each local bundle adjustment. |
| `mapper_ba_local_min_tri_angle` | `--Mapper.ba_local_min_tri_angle` | `6.0` | Smallest triangulation angle that ties a frame into local bundle adjustment. |
| `mapper_ba_global_ignore_redundant_points3d` | `--Mapper.ba_global_ignore_redundant_points3D` | `false` | Leave redundant points out of global bundle adjustment. |
| `mapper_ba_global_ignore_redundant_points3d_min_coverage_gain` | `--Mapper.ba_global_ignore_redundant_points3D_min_coverage_gain` | `0.05` | Coverage gain below which a point is redundant. |
| `mapper_max_runtime_seconds` | `--Mapper.max_runtime_seconds` | `-1` | The mapper stops after this long and keeps what it built. -1 is no limit. |
| `mapper_tri_max_transitivity` | `--Mapper.tri_max_transitivity` | `1` | How far triangulation follows chains of matches. |
| `mapper_tri_create_max_angle_error` | `--Mapper.tri_create_max_angle_error` | `2.0` | Largest angle error to create a point, in degrees. |
| `mapper_tri_continue_max_angle_error` | `--Mapper.tri_continue_max_angle_error` | `2.0` | Largest angle error to extend a track, in degrees. |
| `mapper_tri_merge_max_reproj_error` | `--Mapper.tri_merge_max_reproj_error` | `4.0` | Largest reprojection error to merge two tracks, in pixels. |
| `mapper_tri_complete_max_reproj_error` | `--Mapper.tri_complete_max_reproj_error` | `4.0` | Largest reprojection error to complete a track, in pixels. |
| `mapper_tri_complete_max_transitivity` | `--Mapper.tri_complete_max_transitivity` | `5` | How far completing a track follows chains of matches. |
| `mapper_tri_re_max_angle_error` | `--Mapper.tri_re_max_angle_error` | `5.0` | Largest angle error when retriangulating, in degrees. |
| `mapper_tri_re_min_ratio` | `--Mapper.tri_re_min_ratio` | `0.2` | Frames with less than this share of their matches triangulated are retriangulated. |
| `mapper_tri_re_max_trials` | `--Mapper.tri_re_max_trials` | `1` | Most retriangulation rounds per frame. |
| `mapper_tri_min_angle` | `--Mapper.tri_min_angle` | `1.5` | Smallest triangulation angle, in degrees. |
| `mapper_tri_ignore_two_view_tracks` | `--Mapper.tri_ignore_two_view_tracks` | `true` | Don't triangulate points seen in only two frames. |

`colmap image_undistorter` and `colmap image_undistorter_standalone` both take these, with the same values, so the mask warp crops and scales the mask exactly as the undistortion does the colour picks:

| Key | COLMAP option | Value | What it does |
|---|---|---|---|
| `undistortion_copy_policy` | `--copy_policy` | `"COPY"` | How picks that need no warp are copied, `"COPY"`, `"SOFT_LINK"` or `"HARD_LINK"`. |
| `undistortion_blank_pixels` | `--blank_pixels` | `0.0` | 0 crops so no pixel is blank, 1 keeps the whole view with black edges. |
| `undistortion_min_scale` | `--min_scale` | `0.2` | Smallest rescale the crop may need. |
| `undistortion_max_scale` | `--max_scale` | `2.0` | Largest rescale the crop may need. |
| `undistortion_max_image_size` | `--max_image_size` | `-1` | Longest edge of the warped pictures, in pixels, -1 for no cap. |
| `undistortion_roi_min_x` | `--roi_min_x` | `0.0` | Left edge of the crop, as a share of the width. |
| `undistortion_roi_min_y` | `--roi_min_y` | `0.0` | Top edge of the crop, as a share of the height. |
| `undistortion_roi_max_x` | `--roi_max_x` | `1.0` | Right edge of the crop, as a share of the width. |
| `undistortion_roi_max_y` | `--roi_max_y` | `1.0` | Bottom edge of the crop, as a share of the height. |
| `undistortion_jpeg_quality` | `--jpeg_quality` | `100` | JPEG quality of the warped picks, from 1 to 100, or -1 for 100. |
| `undistortion_num_threads` | `--num_threads` | `-1` | Threads, -1 for every core. |

- The step checks every setting before it reads `frames.json` or runs any COLMAP command. A value of the wrong kind or outside its range fails the step with a readable error that names the key, its table, COLMAP's option for an option key, and what the value must be. A bool is never a number here, and every whole number must fit a 32-bit signed int, the type COLMAP reads its whole-number options into. An option key holds the kind of value its option takes in COLMAP 4.2.1: a TOML bool for an option that's on or off, a TOML integer for a whole number, a TOML integer or float for a decimal, and for a choice, a TOML string that's one of the names this build can run. The ranges are our own sanity bounds, and COLMAP's own checks still run, so a value COLMAP refuses fails its command.
- The step also fails, naming both settings and their values, when `sift_extraction_dsp_min_scale` is greater than `sift_extraction_dsp_max_scale`, `view_graph_calibrator_min_focal_length_ratio` greater than `view_graph_calibrator_max_focal_length_ratio`, `mapper_min_focal_length_ratio` greater than `mapper_max_focal_length_ratio`, `global_mapper_track_min_num_views_per_track` greater than `global_mapper_track_max_num_views_per_track`, or `undistortion_min_scale` greater than `undistortion_max_scale`. It fails too when `undistortion_roi_min_x` isn't below `undistortion_roi_max_x`, `undistortion_roi_min_y` isn't below `undistortion_roi_max_y`, or `global_mapper_ba_skip_fixed_rotation_stage` and `global_mapper_ba_skip_joint_optimization_stage` are both true.
- A key that's missing from `[cameras]` or the preset table fails the step with the step runner's own readable error, which names the section, the table and the key.
- Keep `global_mapper_ba_refine_extra_params` and `mapper_ba_refine_extra_params` true. With either one false, bundle adjustment leaves the distortion at 0, COLMAP's undistorter then keeps the camera's own model, and the step fails because the undistorted camera isn't PINHOLE.
- The real config keeps `global_mapper_multiple_models` and `mapper_multiple_models` true, so a mapper that can't join every pick into one model still writes each part, and the step keeps the part with the most placed frames. It keeps `undistortion_blank_pixels` at 0, so neither the training images nor the training masks have blank edges.
- Each `*_timeout_seconds` is a guess for the largest job the settings allow, 400 picks of 1600 pixels on the long edge, on an 8-vCPU machine, with a few times the expected time to spare. The first real cloud runs should replace them with measured times. The step doesn't read `SPLATPIPE_CAP_DEADLINE`: each time limit catches a hung COLMAP command, and the machine's time cap stays the backstop.

These values stay in code, not in `config.toml`, because the build, our rules or the step's own design force them:

| What | Value | Why it's in code |
|---|---|---|
| The program | `colmap` | The step runs COLMAP only as this separate program. |
| The log options, for every command | `--log_target stderr`, `--log_level 0`, `--log_severity 0` and `--log_color 0` | Each command's log keeps every line of its output, with no colour codes. |
| `feature_extractor` | `--camera_mode 1`, `--FeatureExtraction.type SIFT` and `--FeatureExtraction.use_gpu 0` | One camera for every pick. SIFT needs no model weights, and SiftGPU is non-commercial. |
| `sequential_matcher` | `--FeatureMatching.type SIFT_BRUTEFORCE`, `--FeatureMatching.use_gpu 0`, `--SequentialMatching.loop_detection 0`, and `--SequentialMatching.quadratic_overlap` at `0` in the near match run and `1` in the far match run | `SIFT_BRUTEFORCE` needs no model weights, and loop detection would need a vocabulary tree that COLMAP downloads, so nothing is downloaded. The pairing is what makes the two match runs. |
| `global_mapper` | `--GlobalMapper.gp_use_gpu 0`, `--GlobalMapper.ba_ceres_use_gpu 0` and `--GlobalMapper.ba_backend CERES` | The build has no CUDA, and COLMAP 4.2.1's only other bundle adjustment backend, CASPAR, needs it. |
| `mapper` | `--Mapper.ba_use_gpu 0`, `--Mapper.ba_local_backend CERES` and `--Mapper.ba_global_backend CERES` | The same. |
| `image_undistorter` | `--output_type COLMAP` | The step reads COLMAP's own layout, `images/` and `sparse/`. |
| The paths | As under "What it reads" and "What it writes", each absolute. | The step decides where every file goes. |
| The camera model names | `OPENCV` and `OPENCV_FISHEYE` | The two values `camera_model` can take. |
| The levelled model's camera | PINHOLE | The training step's contract: one camera with no distortion. |
| The world up and the heading | (0, −1, 0) and +Z | The levelled model's frame, which the training step and the viewer rely on. |
| The mask's values | 0 and 255 | Black means ignore and white means use, for COLMAP and the training loss alike. |
| The model files' layout, the work folder's names and `out/cameras.json` | As under "What it writes" and "Files in out/". | They're the contract the training step and the viewer read. |

`min_vcpus` and `min_memory_gb` are the step's machine needs, as the table under "What each step asks the machine for" plans them: 16 vCPUs, a soft need, and 16 GB of memory, a hard need. `cloud.start` reads them for the job's preset to pick the machine. Memory rules out a type with less, while fewer vCPUs only make the step slower, so the step can also run on an 8-vCPU type from a preset's fallback list. The step needs no GPU, so `[cameras]` has no `min_gpu_memory_gb`. The step itself never checks the machine, so it runs the same on a laptop with COLMAP installed. It checks the two settings like the others, but never uses them.

### Step report fields

- `settings`: the 190 settings in `[cameras]` and the preset's `near_overlap`.
- `counts.picks`: the picks in the frames record.
- `counts.placed_frames`: the final model's placed frames, 0 when the incremental mapper wrote no model.
- `counts.points3d`: the final model's points.
- `counts.models_found`: the models of the mapper run that made the final model.
- `counts.fallback_ran`: 1 when the fallback ran, 0 when it didn't.
- `counts.kept_from_global`: 1 when the final model came from the global mapper, 0 when it came from the incremental mapper.
- `counts.levelled`: 1 when the levelling moved the scene, 0 when it kept COLMAP's frame.
- `timings.feature_extractor`, `timings.sequential_matcher_near`, `timings.sequential_matcher_far`, `timings.view_graph_calibrator`, `timings.global_mapper`, `timings.mapper`, `timings.image_undistorter` and `timings.image_undistorter_standalone`: each COLMAP command that ran, in seconds, also when it failed or ran past its time limit. `sequential_matcher_near` is the near match run, and `sequential_matcher_far` the far match run.
- `timings.matching`: both match runs.
- `timings.mapping`: the calibration copy, the calibration, the mappers, choosing the final model and reading it.
- `timings.undistortion`: the undistortion, its checks, the mask warp and the undistorted mask.
- `timings.levelling`: working out the level move and applying it.
- `timings.writing`: the training images, the training masks, the levelled model, `solve.json` and `out/cameras.json`.
- `warnings`, each only when it applies:
  - When `camera_model` is `"OPENCV_FISHEYE"`: "The camera model is OPENCV_FISHEYE, so the focal length calibration was skipped, because COLMAP's view_graph_calibrator only fits pinhole cameras. The fisheye camera starts from fisheye_fov_degrees = <degrees> in [cameras]."
  - When the calibration failed, so the global mapper didn't run: "The focal length calibration <reason>, so the global mapper didn't run and the incremental mapper ran instead." The reason is "stopped with <exit code> (<last line>)", "found no image pairs to calibrate" or "rejected its focal length estimate and kept the starting one".
  - When the global mapper gave up: "The global mapper <reason>, so the incremental mapper ran instead." The reason is "stopped with <exit code> (<last line>)", "wrote no model" or "placed only <k> of the <n> picks in its best model, fewer than the <needed> the step needs". In this warning and the one above, <exit code> reads like `exit code 1`, or `exit code -6 (signal SIGABRT)` when a signal stopped COLMAP. <last line> is the last line that isn't empty in the command's output, before any `*** Check failure stack trace: ***` line, without glog's prefix of level, date, time, thread, file and line.
  - When the cameras' mean up vector is shorter than `up_agreement`: "The cameras don't agree on which way is up: their mean up vector is <length> long, below up_agreement = <value> in [cameras]. So the scene was left in COLMAP's frame, which may be tilted."
  - When no placed frame has a heading: "Every placed camera looks within heading_min_degrees = <degrees>° of straight up or down, so the scene couldn't be turned to face +Z and was left in COLMAP's frame, which may be tilted."
  - When the axis point isn't well defined and the model has no points: "The scene's centre couldn't be found: the cameras' view lines don't meet in a well defined point and the model has no 3D points. So the scene was left in COLMAP's frame, which may be tilted."
  - When working out the level move raises any other error or gives a value that isn't a finite number: "Levelling the scene failed (<type>: <message>), so the scene was left in COLMAP's frame, which may be tilted."
- `error`, when it failed. The step raises `ExpectedFailure` for each failure it foresees, so `expected` is `true` and `readable` is the step's own message, saying what went wrong and what to do. Those failures are an earlier `out/cameras.json` that can't be removed, a bad setting, a missing or broken frames record, a missing colour pick, grey copy or mask, no `colmap` on the PATH, and a COLMAP command that runs past its time limit, whose message names the command, the limit and its `*_timeout_seconds` key. So are a non-zero exit of `feature_extractor`, of a match run or of either undistorter, with the exit code, COLMAP's last line and the command's log, a non-zero exit of the incremental mapper, too few placed frames, a database that can't be read or doesn't hold exactly one camera, a model file that's missing or broken, an undistortion that doesn't give one PINHOLE camera with exactly the placed frames and their images, a warped mask that can't be read, isn't 8-bit or isn't the undistorted camera's size, and a file or folder that can't be written, like on a full disk. The incremental mapper's error and the placed check's say to film the place again: move slower, hold the phone steadier, overlap more between views, and film surfaces with more texture. A missing setting fails with the step runner's own readable error, also with `expected` at `true`. Any other error is a bug, with the general message. A failed calibration, a global mapper that gives up and anything in the levelling aren't failures, and only add warnings.

The counts are whole numbers, so the mean reprojection error, the level move and the reasons are in `solve.json` instead. A failed report holds the counts and timings the step had worked out before it failed. Each command's timing goes in as soon as the command returns, so a failed report has the failing command's timing, but a stage's timing goes in only when its stage finishes. A run that fails the placed check, for example, has `counts.placed_frames` and its mappers' timings, but no `counts.points3d` and no `timings.mapping`.

### Files in out/

- `out/cameras.json`: the viewer cameras, the levelled cameras for the web viewer, with no points. In this example the numbers are rounded, and only the first frame is shown:

```json
{
  "camera": {"width": 1552, "height": 873, "fx": 1290.2, "fy": 1290.2, "cx": 776.1, "cy": 436.4},
  "levelled": true,
  "frames": [
    {"name": "000000.jpg", "quaternion_wxyz": [0.99, 0.01, -0.12, 0.0], "translation": [0.31, -0.02, 1.84]}
  ]
}
```

| Field | What it holds |
|---|---|
| `camera` | The undistorted camera, the one PINHOLE camera of `sparse/cameras.bin`: `width` and `height` in pixels, and `fx`, `fy`, `cx` and `cy`. |
| `levelled` | Whether the levelling moved the scene, as in `solve.json`. |
| `frames` | One object per placed frame, in pick number order. |

Each object in `frames` holds:

| Field | What it holds |
|---|---|
| `name` | Its name, the pick's `file`, like `"000000.jpg"`. |
| `quaternion_wxyz` | Its pose's rotation from `sparse/images.bin`: the cam_from_world unit quaternion, as w, x, y, z. |
| `translation` | Its pose's translation from `sparse/images.bin`. |

The poses follow the convention under "What it writes", and floats keep full precision.

The step removes an earlier `out/cameras.json` when its own code starts, before it does anything else, and writes a new one last of all, after `solve.json`. So a run that fails in the step's code leaves no `out/cameras.json`. When writing `out/cameras.json` itself fails, the step removes what it partly wrote and `solve.json` too, so that run leaves neither.

A run that the step runner stops before the step's code starts, like one on a `[cameras]` section with a missing preset table, leaves the earlier run's `work/cameras/` and `out/cameras.json` in place, next to a failed report. So a reader that needs this run's cameras checks first that `out/reports/cameras.json` says done.

Its report, `out/reports/cameras.json`, comes from the step runner, like every step's.
