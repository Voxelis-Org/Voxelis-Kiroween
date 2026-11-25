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

The list holds the frames step and the cameras step, both required, the depth step, optional, the train step, required, and then the clean step, optional: `StepEntry("frames", "splatpipe.frames", True)`, `StepEntry("cameras", "splatpipe.cameras", True)`, `StepEntry("depth", "splatpipe.depth", False)`, `StepEntry("train", "splatpipe.train", True)` and `StepEntry("clean", "splatpipe.clean", False)`. A job is done once all five steps are done, and done with warnings when the three required steps are done and the depth step or the clean step fails. The planned steps are:

- Required steps: picking frames, solving cameras and training.
- Optional steps: depth priors and clean.

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

On the machine, the Cloud_Runner rebuilds the job folder from the bucket (the `input/`, plus any `out/` and `work/` from an earlier run of the same job), calls `run_job` without `--fresh` so it resumes from the first unfinished step exactly as a laptop run does, uploads each step's report, `work/` and `out/` as they appear with a final sync at the end, writes the spend lines, and powers the machine off when the job ends. The machine's user data starts it as `python -m cloud.runner --job <id>`, from the code it extracted to `/opt/job`, once it has read the machine's own instance type from the instance metadata with an IMDSv2 token, for the spend lines. It isn't a laptop command.

### The cap deadline

The Cloud_Runner sets one environment variable for the steps, `SPLATPIPE_CAP_DEADLINE`: the absolute UTC instant, in ISO 8601, by which a step should have saved and stopped. It's the machine's start time plus the preset's time cap, minus `upload_margin_minutes` from `[cloud]`, leaving room to upload the results before the machine's hard time-cap shutdown. The train step reads it and stops training early, so that `out/train.ply` is written and checked `clean_reserve_minutes` before the cap deadline. The cameras step reads it too, and caps each COLMAP command's time limit at the time left before it.

### What each step asks the machine for

Each step records the machine it needs in its own config section, per preset, as `min_vcpus`, `min_memory_gb` and `min_gpu_memory_gb` (its shared settings and its preset table, like every other setting). A missing key means no need in that dimension. `min_memory_gb` and `min_gpu_memory_gb` are hard needs: `cloud.start` keeps only the types that meet the largest of each over the job's steps. `min_vcpus` is soft, because fewer vCPUs only make a step slower, so it never rules a type out. The plan for the steps we build:

| Step | What it asks for |
|---|---|
| Picking frames | 16 vCPUs, 16 GB RAM |
| Solving cameras | 16 vCPUs, 16 GB RAM |
| Depth priors | a GPU with 16 GB VRAM |
| Training | 4 vCPUs, 32 GB RAM and a GPU with 24 GB (fast and default) or 48 GB (quality) |
| Clean | 8 vCPUs, 8 GB RAM |

Each step spec adds its own keys when it's built. There is no `[cloud.needs.*]` table; a step's needs live in the step's own section. The depth step's need is `min_gpu_memory_gb` in `[depth]`, 16, which the train step's 24 GB already covers.

### The smoke test

`uv run python -m cloud.smoke` proves a new machine image works before any long job. It launches one machine of the `smoke_instance_type` (a g5.xlarge) from the newest image, and that machine's user data runs `python -m cloud.smoke_runner --job <id>`, which isn't a laptop command either. It runs these checks, a few seconds each, and a failed check doesn't stop the others:

- torch sees the GPU.
- gsplat renders a tiny synthetic scene, forwards and backwards.
- fused-ssim runs forwards and backwards.
- ffmpeg has the `zscale` and `tonemap` filters the frames step's tone map needs.
- COLMAP, Node and splat-transform run.
- In the cameras check, every option the cameras step passes is in each COLMAP command's `-h`, and COLMAP solves 30 made-up picks with the step's own code, placing at least 80 % of them with the focal length within 2 % of the truth. The cameras check's numbers are in `[cloud.smoke]`, like `cameras_frames`, the number of made-up picks.
- The MoGe-2 weights match their pinned sha256.
- In the depth check, MoGe-2 loads from the image's weights the way the depth step loads it and runs on a tiny made-up image. Before anything trains, the train step's own GPU check runs, as in the training check. Then the real trainer trains a tiny made-up scene with made-up depth priors, with every prior loss and some splats started from the priors, and the PLY writer and the PLY check run on its splat. Its own number, `depth_normal_start_step`, is in `[cloud.smoke]`.
- In the training check, the real trainer trains a tiny made-up scene on the GPU for a few iterations, with one MCMC refine, then the PLY writer and the PLY check run on its splat. The training check's numbers are in `[cloud.smoke]`, like `train_steps`, the number of iterations.
- The Cloud_Runner runs a tiny fake-step job end to end. The job is made from a test video under the smoke run's own job id, its `input/` goes up to the bucket, and `run_cloud_job` downloads it again, runs the fake step and uploads the results.

The results go to `jobs/_smoke/<image id>.log` in the bucket, pass or fail, with the account id redacted. When every check passes, the machine writes the smoke marker, `jobs/_smoke/<image id>.json`, which records how long the machine took to be ready for work after it booted. `cloud.start` checks for that marker before it starts a job on the image. Then the machine powers itself off. The check timeout, the scene's size and the test video's length are in `[cloud.smoke]`. Each later spec that adds a GPU code path adds a check for it to `SMOKE_CHECKS` in `cloud/smoke_runner.py`, and the training check and the depth check are the GPU checks later specs added. The cameras spec added the cameras check, which runs COLMAP on the CPU.

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
- `mask.png` is one mask for every pick: an 8-bit grey PNG at the pick size that holds only 0 and 255. Black (0) marks a pixel stuck to the camera, and white (255) every other pixel. Each mask pixel copies the score frame pixel under its centre, and nothing grows, shrinks or smooths the stuck pixels. COLMAP takes the mask as its single camera mask, `--ImageReader.camera_mask_path`, and the camera solve step passes it on to training as one mask per image in `work/cameras/masks/`. For both, black means ignore. The step writes the mask only when it uses it: when it has at least 2 mask samples and the stuck pixels cover from `mask_min_coverage` to `mask_max_coverage` of the frame.
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
3. It reads the cap deadline from `SPLATPIPE_CAP_DEADLINE`, when that's set, also before it reads `frames.json` or runs any COLMAP command.
4. It reads the frames record, `work/frames/frames.json`, and checks that every pick's colour pick and grey copy is there, and the mask when the record says the frames step used one. Then it checks that the program `colmap` is on the PATH. All of this comes before any COLMAP command runs.
5. Feature extraction: `colmap feature_extractor` finds SIFT features in the grey copies and saves them in a new database, with one camera of `camera_model` for every pick. With the mask, COLMAP drops the features on the mask's black pixels.
6. Matching: `colmap sequential_matcher` runs twice on the same database. The near match run pairs each pick with the next `near_overlap` picks. Then the far match run pairs each pick i with pick i + 2^k, for k from 0 to `far_overlap` − 1, and COLMAP matches only the pairs the database doesn't hold yet.
7. Calibration, for `"RADIAL"` only: the step copies the database to the calibration copy, then runs `colmap view_graph_calibrator` on the copy, which works out the focal length f from the matches, because the picks carry no EXIF. The calibration has failed when it exits with an error, rejects its focal length estimate or finds no image pairs to calibrate. `view_graph_calibrator` only fits pinhole cameras, so for `"RADIAL_FISHEYE"` the step skips it, and the camera starts from the fisheye start, worked out from `fisheye_fov_degrees`, instead.
8. Mapping: unless the calibration failed, `colmap global_mapper` runs, on the calibration copy, or on the database for `"RADIAL_FISHEYE"`. It gives up when it exits with an error, writes no model, or its kept model fails the placed check. After a failed calibration, or when the global mapper gives up, the step runs the fallback once: `colmap mapper`, the incremental mapper, on the database.
9. The placed check: the final model must place at least `min_placed_frames` picks, and at least `min_placed_share` of them. Otherwise the step fails here, before the undistortion, with a readable error that says how to film the place again.
10. Undistortion: `colmap image_undistorter` warps the final model's colour picks to one PINHOLE camera with no distortion. With the mask, `colmap image_undistorter_standalone` then warps `mask.png` with the same camera and the same settings, and the step makes the undistorted mask from its output.
11. Levelling: one rigid move, with scale 1, makes the scene level, turns it so the heading camera looks along +Z, and moves the scene centre to the origin. When it can't, the step keeps COLMAP's frame, adds a warning and goes on.
12. It writes the training images, the training masks when the frames step used the mask, the levelled model and `solve.json`, then `out/cameras.json`, last of all.

A mapper can write several models, `0/`, `1/` and so on, when it can't join every pick into one. A placed frame is a pick that a model gives a camera pose. A mapper's kept model is its model with the most placed frames, and the one with the lowest number among those with as many. The final model is the incremental mapper's kept model when the fallback ran, and the global mapper's otherwise.

COLMAP runs as a separate program, never linked or imported. The step runs its commands one at a time, each at most once, and retries nothing. Each command has its own time limit in `[cameras]`. When `SPLATPIPE_CAP_DEADLINE` is set, the step caps each limit at the time left before the cap deadline as the command starts, so a command gets the smaller of the two. A command that runs past its time limit fails the step, with no fallback, and so does a cap deadline that has passed before a command can start. Every option of each command is on its command line, from `[cameras]` or fixed in code, so no COLMAP default decides anything.

### What it reads

The step reads the frames step's work folder, `work/frames/`:

- `frames.json`, the frames record. Of its fields, the step uses only `pick_width`, `pick_height`, `mask_used` and `picks`, and of each pick only its `number` and its `file`.
- `images/`, the colour picks, and `grey/`, the grey copies: each pick's `images/<file>` and `grey/<file>`.
- `mask.png`, the mask, when the frames record has `mask_used`.

It also reads `SPLATPIPE_CAP_DEADLINE`, the cap deadline, when the cloud runner sets it, as "The cap deadline" says. It reads it with the train step's own reader, `read_cap_deadline` in `splatpipe/train/deadline.py`, so a value that one step refuses, the other refuses with the same readable error. It reads its settings, from `[cameras]` and from the job's preset table, like `[cameras.fast]`, and it runs COLMAP 4.2.1, the command line program `colmap`.

The step checks the frames record and its files before any COLMAP command runs. A missing `frames.json` means the frames step hasn't finished for the job. A record that can't be read as JSON, a field that's missing or holds the wrong kind of value, an empty `picks`, a pick whose `file` isn't the frames step's name for its `number`, like `000000.jpg` for 0, a `number` given twice, and a missing colour pick, grey copy or mask each fail the step. The readable error names the field or the file, and says to run the frames step with `uv run python -m splatpipe.frames --job <id>`. The number of picks is the number of entries in `picks`, and each image that COLMAP names belongs to the pick whose `file` is that name.

`feature_extractor` reads the grey copies, while `global_mapper`, `mapper` and `image_undistorter` read the colour picks. A grey copy has its colour pick's name and size, and COLMAP knows an image by its name inside the folder it gets as `--image_path`, so to COLMAP both are `000000.jpg`. That way the features come from the evened-out grey copies, and the points' colours and the training images from the colour picks. Each COLMAP command reads and writes these files, with `colmap/` short for `work/cameras/colmap/`:

| Command | Reads | Writes |
|---|---|---|
| `feature_extractor` | The grey copies in `work/frames/grey/`, and with `mask_used`, `work/frames/mask.png` as `--ImageReader.camera_mask_path`. | A new database, `colmap/database.db`. |
| `sequential_matcher`, for both match runs | The database. | The matches, into the database. |
| `view_graph_calibrator` | The calibration copy, `colmap/database_calibrated.db`, which the step makes from the database first. | The focal length, into the calibration copy. |
| `global_mapper` | The calibration copy, or the database for `"RADIAL_FISHEYE"`, and the colour picks in `work/frames/images/`, for the points' colours. | Its models, into `colmap/global/`. |
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
      view_graph_calibrator.log        RADIAL only
      global_mapper.log                unless the calibration failed
      mapper.log                       only when the fallback ran
      image_undistorter.log
      image_undistorter_standalone.log only with mask_used
    database.db                        the database
    database_calibrated.db             the calibration copy, RADIAL only
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
- `sparse/` is the levelled model: exactly `cameras.bin`, `images.bin` and `points3D.bin`, in COLMAP 4.2.1's binary layout, below, with no `rigs.bin`, no `frames.bin` and no other file. COLMAP 4.2.1 reads the folder as it is, because it reads `rigs.bin` and `frames.bin` only when they exist. `cameras.bin` holds exactly one camera, the undistorted camera, of the model PINHOLE, and every image uses it. Its fx and fy are equal: the solved RADIAL or RADIAL_FISHEYE camera has one focal length, f, and the undistortion keeps it as both fx and fy, changing only the size and the principal point. `images.bin` holds one image per placed frame, with the undistorted model's image id, camera id and 2D points, its pose moved by the level move, and its name, the pick's `file`. So the image `000000.jpg` has its training image at `work/cameras/images/000000.jpg`. `points3D.bin` holds every point of the undistorted model, with its id, colour, error and track, and its position moved by the level move.
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
| `model_id` | `int32` | 1 PINHOLE, 3 RADIAL, 9 RADIAL_FISHEYE. The levelled model holds PINHOLE only. |
| `width`, `height` | `uint64`, `uint64` | In pixels. |
| `params` | `float64` × the model's count | PINHOLE: fx, fy, cx, cy, with fx = fy in the levelled model. RADIAL: f, cx, cy, k1, k2. RADIAL_FISHEYE: f, cx, cy, k1, k2. |

Cameras are in increasing `camera_id` order. A PINHOLE record is 56 bytes, and a RADIAL or RADIAL_FISHEYE record 64.

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
  "camera_model": "RADIAL",
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
| `camera_model` | The camera model, `"RADIAL"` or `"RADIAL_FISHEYE"`. |
| `calibration_ran` | Whether the calibration ran: `false` for `"RADIAL_FISHEYE"`. |
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
| `camera_model` | `[cameras]` | `"RADIAL"` | COLMAP's `--ImageReader.camera_model`: the camera every pick shares, `"RADIAL"`, with one focal length, for square pixels, or `"RADIAL_FISHEYE"` for very wide lenses. Both have five parameters: f, cx, cy, k1 and k2. A `"RADIAL"` camera's focal length starts at `image_reader_default_focal_length_factor` times the picks' long edge, with no focal prior, and the calibration works it out. |
| `fisheye_fov_degrees` | `[cameras]` | `120.0` | The field of view across the picks' long edge, in degrees, that a `"RADIAL_FISHEYE"` camera starts from. The fisheye start is its five numbers, passed as `--ImageReader.camera_params` in the order f, cx, cy, k1, k2: f = (L / 2) / (θ / 2), with L the long edge in pixels and θ this angle in radians, cx and cy at the picks' centre, width / 2 and height / 2, and k1 and k2 at 0. That gives the camera the focal prior COLMAP needs to keep fisheye pairs. 120° is a guess until we have real fisheye clips. |
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

Every other key in `[cameras]`, like `camera_model` above, sets one COLMAP option, which its comment in `config.toml` names. These option keys are named after their options. `--Group.name` becomes `<group>_<name>`, with the group in snake case and the whole key in lower case, so `--SiftExtraction.max_num_features` is `sift_extraction_max_num_features`, and `--GlobalMapper.ba_refine_points3D` is `global_mapper_ba_refine_points3d`. An option with no group gets its command's name in front, like `feature_extractor_descriptor_normalization` for `--descriptor_normalization`. The undistortion settings, which both undistorters take, start with `undistortion_`, and `--default_random_seed`, which every command takes, keeps its own name. Each value is the one in the colmap-cli skill's `.kiro/skills/colmap-cli/references/options.md`, which is COLMAP 4.2.1's default unless that file's notes say otherwise, except `camera_model`, which is `"RADIAL"` where that file has COLMAP's default, `SIMPLE_RADIAL`. A decimal option holds a TOML float, like `10.0` where that file writes `10`. The step writes each value on the command line so that COLMAP reads the same value: a bool as `1` or `0`, a whole number in decimal, a decimal with Python's `repr`, the shortest text that reads back as the same double, and a name as it is.

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

It also gets `--ImageReader.camera_mask_path` with `work/frames/mask.png` when the frames record has `mask_used`, and `--ImageReader.camera_params` with the fisheye start's five numbers, joined with commas, when `camera_model` is `"RADIAL_FISHEYE"`. Each is left out otherwise.

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
| `global_mapper_ba_refine_extra_params` | `--GlobalMapper.ba_refine_extra_params` | `true` | Bundle adjustment refines the distortion. Must be true: the undistortion needs it, so the settings check refuses false. |
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
| `mapper_ba_refine_extra_params` | `--Mapper.ba_refine_extra_params` | `true` | Bundle adjustment refines the distortion. Must be true: the undistortion needs it, so the settings check refuses false. |
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
- `global_mapper_ba_refine_extra_params` and `mapper_ba_refine_extra_params` must be true, and the settings check refuses false, with a readable error that says why. Bundle adjustment has to refine the distortion: with it off, the distortion stays 0, COLMAP's undistorter keeps the camera's own model and copies the picks unwarped, and the step can't hand the train step a PINHOLE camera. Refusing false in the settings check stops the step before any COLMAP command, instead of at the undistortion check after the whole solve.
- The real config keeps `global_mapper_multiple_models` and `mapper_multiple_models` true, so a mapper that can't join every pick into one model still writes each part, and the step keeps the part with the most placed frames. It keeps `undistortion_blank_pixels` at 0, so neither the training images nor the training masks have blank edges.
- Each `*_timeout_seconds` is a guess for the largest job the settings allow, 400 picks of 1600 pixels on the long edge, on an 8-vCPU machine, with a few times the expected time to spare. The first real cloud runs should replace them with measured times. Each time limit catches a hung COLMAP command. When `SPLATPIPE_CAP_DEADLINE` is set, the step caps each limit at the time left before the cap deadline as the command starts, so a hung command fails the step by the cap deadline at the latest, whatever its key holds, and the step's failed report is written before the machine's time cap. The time cap stays the backstop.

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
| The camera model names | `RADIAL` and `RADIAL_FISHEYE` | The two values `camera_model` can take. |
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
  - When `camera_model` is `"RADIAL_FISHEYE"`: "The camera model is RADIAL_FISHEYE, so the focal length calibration was skipped, because COLMAP's view_graph_calibrator only fits pinhole cameras. The fisheye camera starts from fisheye_fov_degrees = <degrees> in [cameras]."
  - When the calibration failed, so the global mapper didn't run: "The focal length calibration <reason>, so the global mapper didn't run and the incremental mapper ran instead." The reason is "stopped with <exit code> (<last line>)", "found no image pairs to calibrate" or "rejected its focal length estimate and kept the starting one".
  - When the global mapper gave up: "The global mapper <reason>, so the incremental mapper ran instead." The reason is "stopped with <exit code> (<last line>)", "wrote no model" or "placed only <k> of the <n> picks in its best model, fewer than the <needed> the step needs". In this warning and the one above, <exit code> reads like `exit code 1`, or `exit code -6 (signal SIGABRT)` when a signal stopped COLMAP. <last line> is the last line that isn't empty in the command's output, before any `*** Check failure stack trace: ***` line, without glog's prefix of level, date, time, thread, file and line.
  - When the cameras' mean up vector is shorter than `up_agreement`: "The cameras don't agree on which way is up: their mean up vector is <length> long, below up_agreement = <value> in [cameras]. So the scene was left in COLMAP's frame, which may be tilted."
  - When no placed frame has a heading: "Every placed camera looks within heading_min_degrees = <degrees>° of straight up or down, so the scene couldn't be turned to face +Z and was left in COLMAP's frame, which may be tilted."
  - When the axis point isn't well defined and the model has no points: "The scene's centre couldn't be found: the cameras' view lines don't meet in a well defined point and the model has no 3D points. So the scene was left in COLMAP's frame, which may be tilted."
  - When working out the level move raises any other error or gives a value that isn't a finite number: "Levelling the scene failed (<type>: <message>), so the scene was left in COLMAP's frame, which may be tilted."
- `error`, when it failed. The step raises `ExpectedFailure` for each failure it foresees, so `expected` is `true` and `readable` is the step's own message, saying what went wrong and what to do. Those failures are an earlier `out/cameras.json` that can't be removed, a bad setting, a bad `SPLATPIPE_CAP_DEADLINE`, a missing or broken frames record, a missing colour pick, grey copy or mask, no `colmap` on the PATH, a COLMAP command that runs past a time limit its `*_timeout_seconds` key set, whose message names the command, the limit and the key, a cap deadline that leaves no time when a COLMAP command is about to start, and a COLMAP command that runs past a time limit the cap deadline set. So are a non-zero exit of `feature_extractor`, of a match run or of either undistorter, with the exit code, COLMAP's last line and the command's log, a non-zero exit of the incremental mapper, too few placed frames, a database that can't be read or doesn't hold exactly one camera, a model file that's missing or broken, an undistortion that doesn't give one PINHOLE camera with exactly the placed frames and their images, a warped mask that can't be read, isn't 8-bit or isn't the undistorted camera's size, and a file or folder that can't be written, like on a full disk. A bad `SPLATPIPE_CAP_DEADLINE` is an empty value, one that isn't a date and time, or one with no UTC offset. Its message is the train step's: it gives the value, says it must be a date and time with a UTC offset, like `2026-11-21T14:25:30Z`, and says to unset it on a laptop. When no time is left before the cap deadline, the command doesn't start and gets no timing, and the message names the command and the cap deadline. When a command runs past a time limit the cap deadline set, the message names the command and the cap deadline and gives the seconds that were left. Both of these say to give the job more time with a larger `cap_minutes` for its preset under `[cloud.presets]` in `config.toml`, then run the job again. When the undistorted camera isn't PINHOLE, the message says that COLMAP keeps a camera's own model when the camera's distortion came out as 0, and to run the step again. The incremental mapper's error and the placed check's say to film the place again: move slower, hold the phone steadier, overlap more between views, and film surfaces with more texture. A missing setting fails with the step runner's own readable error, also with `expected` at `true`. Any other error is a bug, with the general message. A failed calibration, a global mapper that gives up and anything in the levelling aren't failures, and only add warnings.

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
| `camera` | The undistorted camera, the one PINHOLE camera of `sparse/cameras.bin`: `width` and `height` in pixels, and `fx`, `fy`, `cx` and `cy`, with `fx` = `fy`. |
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

A run that the step runner stops before the step's code starts leaves the earlier run's `out/cameras.json` in place, next to a failed report. When it stops on the config, like a `[cameras]` section with a missing preset table, it also leaves the earlier run's `work/cameras/`. When it stops later, like on a step module that fails to import, it has already emptied `work/cameras/`, so only `out/cameras.json` is left. So a reader that needs this run's cameras checks first that `out/reports/cameras.json` says done.

Its report, `out/reports/cameras.json`, comes from the step runner, like every step's.

## Step: depth

The depth step is in `splatpipe/depth/`, built like the fake step. It makes depth priors for the train step on the job machine's GPU. It runs MoGe-2 on each frame of the camera solve output in `work/cameras/`, fits each frame's depth to the camera solve's own points, and writes the frames that fit as prior files in its work folder, `work/depth/`. It's an optional step: when it fails, the job goes on and trains without depth priors.

Its CLI is `uv run python -m splatpipe.depth --job <id> [--preset P]`, and its plain function is `splatpipe.depth.run(job_id, preset=None)`.

It works in this order:

1. It checks every setting.
2. It checks that the camera solve output's `images/` folder and its three model files are there, without reading them. This comes before the GPU checks, so a job without the camera solve output fails with the camera solve's message on any machine, a laptop included.
3. It imports the GPU module, `splatpipe/depth/gpu.py`, which needs torch.
4. It checks the machine: torch must see a CUDA GPU, the MoGe-2 weights must be a file at `moge_weights`, and MoGe-2 must import, offline. It checks no amount of GPU memory.
5. It loads MoGe-2 from the weights onto the GPU. It reads nothing of the camera solve output before this.
6. It reads the sparse model with the train step's reader, and works out the prior size and the field of view.
7. For each frame, in `images.bin`'s order, it decodes the frame and scales it down to the prior size, runs MoGe-2 on it, fits it to the camera solve's points and, when it fits, writes its prior file.
8. It releases MoGe-2 and empties torch's GPU memory cache, so the train step that runs next in the same process gets the GPU's memory. This happens once the load has started, whether the step is done or fails.
9. When no frame fitted, the step fails here. When some frames were left out, it adds a warning.
10. Last of all, it writes the prior index, `work/depth/depth.json`.

### What it reads

- The camera solve output in `work/cameras/`: the frames in `images/` and the sparse model in `sparse/`, the same files the train step reads, under the train step's names, as "Step: train" lays them out. It reads no masks, so it never opens `masks/`.
- Its settings, from `[depth]`. Every one is shared, so the preset tables are empty.
- The MoGe-2 weights, the `Ruicheng/moge-2-vitb-normal` checkpoint, at `moge_weights`: `/opt/models/moge-2-vitb-normal/model.pt`, where the image build downloads them and checks their sha256. The step doesn't check the sha256 again, because the image build and the smoke test do.
- torch, MoGe's code (`moge`, with `utils3d_moge`) and huggingface_hub, which only a GPU machine image built with MoGe's code has. Only the GPU module, `splatpipe/depth/gpu.py`, imports them: torch at its top, and MoGe and huggingface_hub only inside its functions. The step imports that module only when it runs. So `splatpipe.depth` imports on a laptop, and there the step fails at its GPU checks with a readable error that says to run the job with `uv run python -m cloud.start`.

The step reads the sparse model with the train step's own reader, so it refuses exactly the models the train step refuses, with the same messages. The reader also keeps each point's id from `points3D.bin` and each image's observed points: the distinct point ids among the image's keypoints that are ids of points in `points3D.bin`, in the order of their first keypoint. A keypoint whose point id is −1, or isn't the id of a point, is no observation, and no model is refused for it. The reader still keeps no keypoint's x and y and no point's error or track.

The step goes through every image in `images.bin`, in that file's order, and opens only the files those images name, so it ignores every other file in `images/`. It decodes each frame as the train step decodes a training image: as stored, ignoring any EXIF orientation, as a colour JPEG at the shared camera's size. A frame with no file, one that can't be decoded and one that isn't the shared camera's size fail the step with the train step's messages. The step scales the frame down to the prior size with OpenCV's `INTER_AREA` when it's larger, and turns it from BGR to RGB.

The prior size is the train step's training size rule applied to the shared camera with `long_edge` from `[depth]`: a camera whose longer edge is above 512 pixels gets priors whose longer edge is 512, keeping its shape, and a smaller camera's priors keep its size. So a 1920 × 1080 camera gets 512 × 288 priors.

MoGe-2 loads from the weights file by its path, and the step downloads nothing:

- The GPU module sets `HF_HUB_OFFLINE=1` before it imports huggingface_hub or MoGe, so the Hugging Face client makes no requests. After the import, it checks that huggingface_hub really is offline.
- It loads the file with `torch.load(path, map_location="cpu", weights_only=True)`, so the file can't run code, builds `MoGeModel(**checkpoint["model_config"])` from `moge.model.v2`, and loads `checkpoint["model"]` strictly. A missing or unexpected key fails the step, instead of leaving part of MoGe-2 with random weights.
- It moves MoGe-2 to the GPU, in eval mode and in float32.
- It never calls `from_pretrained()`, `hf_hub_download()` or `init_weights()`, so it fetches neither MoGe-2's weights from the Hugging Face Hub nor DINOv2's original weights.

MoGe-2 gets each frame as RGB floats from 0 to 1, each 8-bit value divided by 255 on the GPU. Its `infer()` gets the field of view as `fov_x`, `resolution_level` and `use_fp16` from `[depth]`, and `force_projection` and `apply_mask` at MoGe's defaults. The field of view is degrees(2 × atan(W / (2 × fx))), from the shared camera's full width W and its fx, both in pixels, so scaling the frame doesn't change it. MoGe-2 gives back, at the prior size, each pixel's depth, the camera-space z in metres as MoGe-2 estimates them, its unit normal, and a mask of where it found geometry. A pixel is a geometry pixel only where the mask is true and the depth is finite and above 0, so sky, and any other pixel where MoGe-2 found no geometry, never gets a prior.

### What it writes

The step writes only into its work folder: `work/depth/<stem>.npz`, `work/depth/depth.json.tmp` while it writes the index, and `work/depth/depth.json`. These files are the prior format, the contract between the depth step and the train step, and anything that makes priors later writes the same. One module, `splatpipe/train/priors.py`, defines it for both steps, with its writer and its reader, and imports no torch. The names are fixed there, not in `config.toml`.

```
work/depth/
  000000.npz ...   the prior files, one for each fitted frame, named after its image
  depth.json       the prior index, written last
```

A prior file is numpy's uncompressed `.npz`, written with `np.savez` and read with `np.load(path, allow_pickle=False)`, so reading one can run no code. Its name is its image's name in `images.bin` with `.jpg` swapped for `.npz`: the prior of `000012.jpg` is `000012.npz`. It holds exactly three arrays at the prior size, h = `prior_height` rows by w = `prior_width` columns:

| Array | Shape | dtype | What it holds |
|---|---|---|---|
| `inverse_depth` | (h, w) | float32, little-endian | The fitted inverse depth, scale × (1 / MoGe's depth) + offset, in 1 / camera solve units: an estimate of 1 / the pixel's camera-space z in the camera solve's own units. Above 0 at each pixel with a prior, and 0 at every other pixel. |
| `normal` | (h, w, 3) | float32, little-endian | MoGe-2's unit normal at each pixel with a prior, in the camera frame of the image's pose, with x right, y down and z forward, facing the camera as MoGe-2 gives it: a wall square to the camera has z near −1. (0, 0, 0) at every other pixel. |
| `mask` | (h, w) | bool | True exactly at the pixels with a prior. |

Stored pixel (x, y) is centred at ((x + 0.5) × W / w, (y + 0.5) × H / h) in the shared camera's pixels, where W × H is the camera's size and the top-left pixel's centre is (0.5, 0.5), as COLMAP and MoGe both put it. Every value in a prior file is finite.

`depth.json`, the prior index, describes the priors. It holds only plain JSON, and floats keep full precision. This example is from a 1920 × 1080 camera, with one of its priors shown:

```json
{
  "format": 1,
  "camera": {"width": 1920, "height": 1080, "fx": 1450.0, "fy": 1450.0, "cx": 960.0, "cy": 540.0},
  "prior_width": 512,
  "prior_height": 288,
  "metres_per_unit": 1.8734,
  "priors": [
    {"name": "000000.jpg", "quaternion": [0.9982, 0.0121, -0.0583, 0.0034],
     "translation": [0.412, -0.077, 1.935], "scale": 0.5312, "offset": 0.0071,
     "fit_points": 412, "kept_points": 377, "median_error": 0.0431, "metres_per_unit": 1.9021}
  ]
}
```

| Field | What it holds |
|---|---|
| `format` | 1, the prior format's version, as a whole number. |
| `camera` | The shared camera the priors were made for: its `width` and `height` as whole numbers, and its `fx`, `fy`, `cx` and `cy` as `cameras.bin` gives them. |
| `prior_width`, `prior_height` | The prior size w and h, in pixels, whole numbers above 0. |
| `metres_per_unit` | The job's metres per camera solve unit, an estimate that nothing applies. |
| `priors` | One object per fitted frame, in `images.bin`'s order. |

Each object in `priors` holds:

| Field | What it holds |
|---|---|
| `name` | The image's name in `images.bin`. |
| `quaternion`, `translation` | The image's pose, (w, x, y, z) and (x, y, z), exactly as `images.bin` gives them, before any normalising, so a reader can tell priors made for an earlier camera solve. |
| `scale`, `offset` | The second pass's scale and offset. |
| `fit_points`, `kept_points` | The fit points of the first pass and of the second. |
| `median_error` | The median relative depth error, over every fit point. |
| `metres_per_unit` | This frame's metres per camera solve unit. |

The index names no path. A reader derives each prior file's path from a name it has checked against `images.bin`, never from a path in the index.

The step writes the prior index after every prior file, to `depth.json.tmp` in the same folder, and one rename on the same file system turns it into `depth.json`, so a reader never finds a half-written index. When that write fails, the step removes the temporary file. So a run that fails, at any point, leaves no `depth.json`, and the format's reader, `read_prior_set` in `splatpipe/train/priors.py`, reads a `work/depth/` with no `depth.json` as no priors, whatever else it holds. A run where no frame fits leaves `work/depth/` empty. The step runner empties `work/depth/` before the step runs, so no file from an earlier run is left.

The train step trains at its own training size, which is usually larger than the prior size, so a prior is scaled up to it per training pixel, with pixel centres aligned. `splatpipe/train/prior_views.py` holds the rule, in `upscale_taps`, `upscale` and `prior_pixel_mask`, with no torch. Along each axis, training pixel X of T sits at the stored coordinate (X + 0.5) × S / T − 0.5 of the S stored pixels, worked out in whole numbers, so the centres line up as COLMAP and MoGe put them.

- `inverse_depth` and `normal` are scaled up bilinearly: each training pixel blends the four stored pixels around its stored coordinate, which is clamped to the first and the last stored pixel at the edges. Each scaled-up normal is then set to unit length, n / ‖n‖, and a blend of length 0 stays (0, 0, 0).
- `mask` is scaled up nearest-neighbour: each training pixel takes the mask of the stored pixel under its centre, ⌊(2X + 1) × S / (2T)⌋ along each axis.

A training pixel counts as a prior pixel only when its training image's mask, if it has one, doesn't leave it out of the loss, the stored pixel under its centre has a prior, and every stored pixel its bilinear sample draws on with a weight above 0 has a prior too. So the 0s at the pixels without a prior never blend into the depth at the edge of MoGe's mask. A stored pixel without a prior that the sample gives a weight of 0, as when a training pixel's centre falls exactly on a stored pixel's centre, doesn't count against it.

MoGe-2's depth is in metres as it estimates them, with a scale of its own for each frame, while the camera solve's model has a scale of its own too. So the step fits each frame's inverse depth to the camera solve's own points before it writes a prior. A frame's fit points are its observed points that are in front of its camera, whose projection (u, v) with the shared camera at its full size lands inside the frame, and whose prior-size pixel is a geometry pixel. That pixel is (min(⌊u × w / W⌋, w − 1), min(⌊v × h / H⌋, h − 1)), the stored pixel whose footprint holds the projection. Each fit point's target is 1 / its camera-space z, and its MoGe value is 1 / MoGe's depth at its pixel.

The fit has two passes, worked out in float64. A set of numbers' spread is their mean distance from their median.

1. A frame with fewer than `fit_min_points` (10) fit points is left out, with too few points.
2. The first pass: the scale is the targets' spread over the MoGe values' spread, and the offset is the targets' median minus the scale times the MoGe values' median. A frame whose MoGe values have a spread of 0 is left out as flat.
3. Each point's miss is |scale × MoGe value + offset − target|. The second pass keeps the points whose miss is at most `fit_miss_ratio` (3) times the median miss, so a few points far off the rest can't pull the fit.
4. When fewer than `fit_min_points` points are kept, the frame is left out with too few points. When the kept MoGe values have a spread of 0, it's left out as flat.
5. The second pass fits the kept points the same way, and its scale and offset are the frame's. A scale of 0 leaves the frame out as flat. The scale is a ratio of spreads, so it's never below 0.
6. The 15% check: over every fit point, the ones the second pass dropped included, the fitted inverse depth p is scale × MoGe value + offset, and the relative depth error is |target / p − 1|, which is |1 / p − z| / z. Where p isn't above 0, the error is infinite. A frame whose median relative depth error is above `fit_max_median_error` (0.15) is left out as a poor fit.

Every frame that passes is a fitted frame, and gets a prior file. At each geometry pixel, its fitted inverse depth is the second pass's scale × (1 / MoGe's depth) + offset, worked out in float64 and stored as float32. A pixel keeps its prior only where that float32 value is finite and above 0 and every component of MoGe's normal is finite. Every other pixel gets 0s and a false mask.

A fitted frame's metres per unit is the median, over the second pass's kept points, of target / MoGe value, which is MoGe's depth over the camera solve's z: how many metres one camera solve unit is, as MoGe-2 sees it. The job's is the median of the fitted frames'. Both go in `depth.json` as an estimate only. The step changes no camera, pose or point, and nothing is rescaled, so the scene stays in the camera solve's units.

### Config section

| Key | Table | Value | What it is |
|---|---|---|---|
| `long_edge` | `[depth]` | `512` | The longest edge of the depth priors, in pixels. A camera with a longer edge gets priors scaled down to it, keeping its shape. A smaller camera isn't scaled up. |
| `moge_weights` | `[depth]` | `"/opt/models/moge-2-vitb-normal/model.pt"` | The MoGe-2 ViT-B normal weights on the machine image, where the image build downloads them. A test keeps it equal to `MOGE_WEIGHTS` in `cloud/image.py`. |
| `resolution_level` | `[depth]` | `9` | How much detail MoGe-2 sees, from 0 to 9, passed to its `infer()`. 9 is MoGe's default, its most detail. |
| `use_fp16` | `[depth]` | `true` | Whether MoGe-2 runs its network in half precision, as it does by default. The model itself stays in float32. |
| `fit_min_points` | `[depth]` | `10` | The fewest fit points a frame needs in each of its two passes to get a depth prior. |
| `fit_miss_ratio` | `[depth]` | `3` | The second pass leaves out the points that miss the first pass's fit by more than this times the median miss. |
| `fit_max_median_error` | `[depth]` | `0.15` | A frame gets no depth prior when its median relative depth error is above this. |
| `min_gpu_memory_gb` | `[depth]` | `16` | The least GPU memory the job's cloud machine needs for this step, in GB. The step never checks it. |

`[depth.fast]`, `[depth.default]` and `[depth.quality]` are empty, each with a comment above it saying that every `[depth]` setting is shared, so every preset makes the same priors. The real `config.toml` has a comment on the line directly above every key.

- The step checks every setting before it checks the camera solve output. A value of the wrong kind fails the step with a readable error that names the key, its table and what it must be, like "The setting resolution_level in [depth] is 10, but it must be a whole number from 0 to 9. Change it to a whole number from 0 to 9." A bool is never a number here. A missing key fails the step with the step runner's message, which names `[depth]`, the preset's table and the key.
- `long_edge`, `fit_min_points` and `min_gpu_memory_gb` must be whole numbers above 0, `resolution_level` a whole number from 0 to 9, `use_fp16` true or false, `moge_weights` a string of one or more characters, and `fit_miss_ratio` and `fit_max_median_error` numbers above 0.

`min_gpu_memory_gb` is the step's one machine need: the job's cloud machine needs a GPU with 16 GB for this step. The cloud spec reads it to pick the machine. The step itself checks it like the other settings, but never uses it, so it runs on any GPU that torch can see.

### Step report fields

- `settings`: the 8 settings in `[depth]`. The preset tables add none.
- `timings.load_model`: building MoGe-2 and loading its weights onto the GPU.
- `timings.infer`: decoding the frames, scaling them down and running MoGe-2 on them, summed over the frames.
- `timings.fit`: the fit points and the two passes, summed over the frames.
- `timings.write`: writing the prior files and the prior index.
- `counts.images`: the images in `images.bin`, which are the frames.
- `counts.prior_width` and `counts.prior_height`: the prior size, in pixels.
- `counts.fitted_images`: the fitted frames, each with a prior file.
- `counts.few_points_images`: the frames left out with too few fit points, in either pass.
- `counts.flat_images`: the frames left out as flat.
- `counts.poor_fit_images`: the frames left out as a poor fit, with a median relative depth error above `fit_max_median_error`.
- `counts.peak_gpu_memory_bytes`: the most GPU memory torch's allocator held at once from the GPU check on, from `torch.cuda.max_memory_allocated()`. The step adds it once MoGe-2 has started to load, whether the step is done or fails.
- `warnings`, only when the step left out some frames but fitted at least one: "Depth priors were made for <fitted> of <frames> frames. Of the others, <few> had too few of the camera solve's points, <flat> were flat and <poor> fitted poorly, so they train without a depth prior."
- `error`, when it failed. The step raises `ExpectedFailure` for each failure it foresees, so `expected` is `true` and `readable` is the step's own message, saying what went wrong and what to do. Those failures are:
  - a bad or missing setting;
  - a camera solve output that's missing, or that the train step's reader refuses, and a frame with no file, one that can't be decoded and one that isn't the shared camera's size, each with the train step's message;
  - a machine without torch, MoGe's code, huggingface_hub or a package they import, which names the package and says that the step runs only on a GPU machine image built with MoGe's code; one where torch sees no CUDA GPU; and one with no file at `moge_weights`, which says to rebuild the machine image with `uv run python -m cloud.image`;
  - a file at `moge_weights` that isn't the MoGe-2 ViT-B normal checkpoint, because it can't be loaded, isn't a dict that holds `model_config` and `model`, has a `model_config` that doesn't build MoGe-2, or fails the strict load;
  - torch running out of GPU memory while the step loads MoGe-2, or while it runs MoGe-2 on a frame, which the error names;
  - no fitted frame: "None of the <frames> frames fitted the camera solve's points, so no depth priors were made: <few> had fewer than <fit_min_points> of the points, <flat> were flat, and <poor> fitted poorly, with a median depth error above <fit_max_median_error>. The job goes on and trains without depth priors.";
  - a prior file or the prior index that can't be written, like on a full disk, with the file and the reason.

huggingface_hub found online after MoGe's import is a bug in the step, because it means something imported huggingface_hub before the GPU module set `HF_HUB_OFFLINE`. The step then fails with the general bug message, `expected` is `false`, and the message in `technical` says so. Any other error is a bug too, with the general message.

When the step fails, for any reason, the job runner writes its readable error to the job log and goes on with the next step, because the step is optional. So a job whose required steps are all done ends done with warnings, with the depth step's entry `failed`. A failed run leaves no `depth.json`, so the job trains without depth priors.

The counts are whole numbers, so the scales, offsets, errors and metres per unit are in `depth.json` instead. The step adds each count as soon as it knows it, and each timing as a running sum over the frames, so a failed report holds the counts and timings the step had worked out before it failed.

### Files in out/

The depth step writes no files to `out/` but its report, `out/reports/depth.json`, which the step runner writes, like every step's.

## Step: train

The train step is in `splatpipe/train/`, built like the fake step. It trains the job's 3D Gaussian splat on the job machine's GPU, from the frames, cameras and points that the camera solve step writes to `work/cameras/`, and writes the splat to `out/train.ply`, where the clean step reads it. It trains with its own trainer, written on gsplat 1.5.3's `rasterization()` and MCMC strategy.

Its CLI is `uv run python -m splatpipe.train --job <id> [--preset P]`, and its plain function is `splatpipe.train.run(job_id, preset=None)`.

It works in this order:

1. It removes `out/train.ply`, first of all, so a run that fails leaves no splat from an earlier run in `out/`.
2. It checks every setting.
3. It reads the cap deadline from `SPLATPIPE_CAP_DEADLINE`, when that's set, and works out the stop time.
4. It checks that the camera solve output's `images/` folder and its three model files are there, without reading them. This comes before the GPU checks, so a job without the camera solve output fails with the camera solve's message on any machine, a laptop included.
5. It imports the GPU module, `splatpipe/train/gpu.py`, and checks the GPU: torch must see a CUDA GPU, gsplat must be 1.5.3, and the GPU must have the memory the preset needs. Nothing is read and nothing goes onto the GPU before this.
6. The load: it reads the sparse model, the training images and their masks, then the depth priors in `work/depth/` when `work/depth/depth.json` is there, and sets up the scene at the training size. It starts a splat at each point of the sparse model, or at `max_splats` of them chosen at random when there are more, and sizes each splat from its nearest other points, which the GPU finds. With depth priors, it also starts splats from the priors, on plain surfaces where the sparse model has no point, and the sparse model's points get only what those leave of `max_splats`, as "What it reads" says.
7. The training: the trainer trains the splats on the GPU for the preset's `steps` iterations, unless the stop time comes first. Each iteration renders one training image, turns the render's colours with that image's own exposure fix, works out the loss against the image with its masked pixels left out, and updates the splats and the fix. With depth priors, the loss also holds the five prior loss terms, which pull the rendered depth and the splats' normals toward the priors and keep the splats flat, as "Config section" says. The trainer goes through every image once per pass, in a new order each pass. Every `refine_every` iterations for most of the run, gsplat's MCMC strategy moves nearly transparent splats onto opaque ones and adds new ones, up to `max_splats`.
8. The write: it bakes the mean of the images' exposure fixes into the splats' colours, leaves out every splat with a NaN or an infinity, and writes the rest to `work/train/train.ply`.
9. The PLY check reads the file back and checks its header, its size and that every value is finite.
10. One move puts the checked file at `out/train.ply`.
11. Last of all, it writes the train record, `work/train/train.json`.

### What it reads

- The camera solve output in `work/cameras/`, below.
- The depth priors in `work/depth/`, which the depth step writes, only when `work/depth/depth.json` is there. Below.
- `SPLATPIPE_CAP_DEADLINE`, the cap deadline, when the cloud runner sets it.
- Its settings, from `[train]` and from the job's preset table, like `[train.fast]`.
- torch, gsplat 1.5.3 and fused-ssim, which only the job's machine image has. Only the GPU module, `splatpipe/train/gpu.py`, imports them, and the step imports that module only when it runs. So `splatpipe.train` imports on a laptop, and there the step fails at its GPU checks with a readable error that says to run the job with `uv run python -m cloud.start`.

The camera solve output is the contract between the camera solve step and this one, and the camera solve spec writes exactly this layout. The names are fixed in `splatpipe/train/files.py`, not in `config.toml`.

```
work/cameras/
  images/<stem>.jpg ...   the undistorted colour frames, one for each image in images.bin
  masks/<stem>.png ...    an image's mask, only when it has one
  sparse/
    cameras.bin           one shared PINHOLE camera
    images.bin            each image's id, pose, camera id, name and keypoints
    points3D.bin          each point's id, position, 8-bit RGB colour, error and track
```

- `images/` holds the training images: one undistorted colour JPEG for each image in `images.bin`, under the name `images.bin` gives it, at the shared camera's size. A name is a stem of one or more characters with no `/`, `\`, `:` or NUL, then `.jpg` in lower case. The step decodes each image as it's stored, ignoring any EXIF orientation, so the pixels stay the ones the camera solve worked on. An image larger than the training size is scaled down to it with OpenCV's `INTER_AREA`.
- `masks/` holds the masks. A mask is matched to its image by stem: the mask of `000123.jpg` is `masks/000123.png`. Each image's mask is optional on its own. An image with a mask leaves its masked pixels out of the loss, and an image without one trains on every pixel. A mask is a grey PNG at its image's size, and black means ignore: any value below 128, the `mask_threshold`, counts as black. So the camera solve step should write only 0 and 255. At a smaller training size, each training pixel takes the mask pixel under its centre.
- `sparse/` holds COLMAP's binary model. `cameras.bin` holds one camera, shared by every image, of COLMAP's PINHOLE model (model id 1), with its width, height, fx, fy, cx and cy in pixels. `images.bin` holds each image's id, its pose as a quaternion (qw, qx, qy, qz) and a translation, its camera id and its name. `points3D.bin` holds each point's id, position and 8-bit RGB colour. The step keeps each point's id and each image's observed points, the distinct point ids among its keypoints that are ids of points in `points3D.bin`, as "Step: depth" says under "What it reads". It still reads past the keypoints' x and y, the points' errors and the tracks, keeping none of them, and ignores every other file in `sparse/`, like the `rigs.bin` and `frames.bin` that COLMAP 3.12 and later write.
- The model is in the levelled frame, the world frame the camera solve step leaves it in. The step uses the poses, the camera and the points exactly as they are, so the trained splat lies in the same frame as the cameras.
- The step reads the model with its own reader, written with numpy and Python's `struct` module. It runs no COLMAP program and uses no pycolmap.
- It trains on every image in `images.bin`, in that file's order, and holds none out for evaluation. It opens only the files those images name, so it ignores any other file in `images/` and `masks/`.

The step fails with a readable error that names the file and says what to do when:

- `images/` or one of the three model files is missing. The error names the path and says that the camera solve step writes it, so that step has to run first.
- `cameras.bin` holds more or fewer than one camera, or a camera whose model isn't PINHOLE, because the step needs undistorted frames.
- A model file ends inside a record, holds bytes after its last record, or gives a count its size can't hold.
- The camera's width, height, fx or fy isn't above 0, a camera parameter, a pose or a point's position isn't finite, or a quaternion is four zeros.
- `images.bin` holds no image, an image of another camera, two images with one name, or a name that isn't a stem and `.jpg`. A bad name is given by its image's id, never as the name itself.
- `points3D.bin` holds fewer than `start_scale_neighbours` + 1 points, 4 with the real settings.
- An image has no file in `images/`, or its file can't be decoded as a colour JPEG or isn't the shared camera's size. When the sizes differ, the error gives both.
- A mask can't be decoded as a grey image, isn't its image's size, or masks every pixel once it's scaled to the training size.

The depth priors are the prior format that "Step: depth" lays out under "What it writes", and the step reads them with that format's reader, `read_prior_set` in `splatpipe/train/priors.py`. With no `work/depth/depth.json`, whatever else `work/depth/` holds, the step trains without depth priors, exactly as it would with no depth step, and adds no count or warning about them. When `work/depth/depth.json` is there, the step reads the priors once it has read the sparse model, the training images and their masks, and before it sets up the scene and the starting splats. It checks, in this order:

1. that `depth.json` reads as a JSON object with `format` 1 and exactly the prior index's fields, each of its JSON type, so `true` is never 1 and `512.0` is never a whole number;
2. that its `camera` is the shared camera in `cameras.bin`, with an equal width, height, fx, fy, cx and cy;
3. that `prior_width` and `prior_height` are whole numbers above 0;
4. that `priors` lists one or more priors;
5. that each prior's `name` is the name of an image in `images.bin`, and is listed once;
6. that each prior's `quaternion` and `translation` equal, value for value, its image's pose in `images.bin`, so priors made for an earlier camera solve aren't used;
7. that each prior file, at the path that `prior_path` derives from the checked name and never at a path from the index, is there, reads as a `.npz` file with `allow_pickle=False`, and holds exactly `inverse_depth`, `normal` and `mask` at `prior_height` × `prior_width`, as float32, float32 and bool, with every value finite and `inverse_depth` above 0 where `mask` is true and 0 where it's false.

The first check that fails drops every prior, so one bad file means the step trains without depth priors. It then adds one warning that names the file and what failed, "The depth priors in work/depth/ weren't used, because <file> <what>. Training ran without depth priors.", and goes on as it would with no `depth.json`. Priors that fail a check never fail the step.

When every check passes, each training image that `depth.json` names is a prior view, and the scene keeps two things for it: its prior at the prior size, as read, and its training image's brightness gradient at the training size. That gradient is the length of the central-difference gradient of the image's grey value from 0 to 1, OpenCV's RGB-to-grey value divided by 255, with the border pixels repeated outward, kept as float32. A training image with no prior keeps no prior. The scene also works out, once per run, how to scale a prior up from the prior size to the training size, as "Step: depth" says under "What it writes". The step adds `counts.prior_images`.

With depth priors, some of the starting splats start from the priors, on plain surfaces where the camera solve found no points. In each prior view, a training pixel is a candidate when it's a prior pixel, as "Step: depth" says under "What it writes", when it's a plain pixel, whose brightness gradient is below `plain_gradient`, and when none of that image's own observed points lands on it. A point lands on a training pixel when it's in front of the camera and its projection (u, v), through the view's camera at the training size and the image's pose, falls in that pixel, (⌊u⌋, ⌊v⌋). So each point rules out only the one pixel it lands on.

The budget for these splats is the smaller of ⌊`backproject_share` × `max_splats`⌋ and `backproject_max`: 333,333 for `fast`, and 500,000 for `default` and `quality`. The candidates are numbered view by view in `images.bin`'s order, and row by row within each view. A random generator seeded with `seed`, used for nothing else, draws as many of them as the budget allows, or every candidate when there are fewer, without replacement and with an equal chance for each. Each drawn pixel's splat starts at the world point of the pixel's centre, at the camera-space depth of 1 / the prior's `inverse_depth` there, scaled up to the training size, through the view's camera at the training size and the image's pose, all worked out in float64. It takes the training pixel's 8-bit colour. A `backproject_max` of 0 starts no splat from the priors.

The sparse model's points fill the rest of `max_splats`: every point when they fit in `max_splats` less the splats drawn from the priors, or otherwise that many of them, chosen at random as with no depth priors. The starting splats hold the sparse model's points first, then the splats from the priors, view by view and pixel by pixel. All of them go through one nearest-point search on the GPU together, so each splat's starting size comes from its `start_scale_neighbours` nearest other starting splats, of either kind, and every splat starts at `start_opacity`, with no rotation and higher spherical harmonics of 0. The step adds `counts.backprojected_splats`. With no depth priors, or with priors that failed a check, the starting splats are the sparse model's points alone, as without a depth step.

`SPLATPIPE_CAP_DEADLINE` is the cap deadline the cloud runner sets, as "The cap deadline" says. When it's set, the stop time is the cap deadline minus `clean_reserve_minutes` and `save_margin_minutes`, 15 minutes before it with the real settings, and the trainer starts no iteration at or after the stop time. That leaves `save_margin_minutes` to copy the splats off the GPU and to write and check the file, so `out/train.ply` is in place `clean_reserve_minutes` before the cap deadline, for the clean step. When the variable isn't set, as on a laptop, every planned iteration runs. Its value must be a date and time with a UTC offset, like `2026-11-21T14:25:30Z` or `2026-11-21T14:25:30.123456+00:00`. An empty value, a value that isn't a date and time, and one with no UTC offset each fail the step with a readable error. An empty value doesn't count as unset, because the machine's user data leaves the variable empty only when working out the deadline failed.

### What it writes

The step writes only `work/train/train.ply`, `work/train/train.json` and `out/train.ply`. Its report comes from the step runner.

```
work/train/
  train.ply             the trained splat while it's written and checked, then moved to out/
  train.json            the train record, written last, once out/train.ply is in place
out/
  train.ply             the trained splat, removed when the step starts, there again once it's done
  reports/train.json    the step's report, which the step runner writes
```

A done run leaves `work/train/` holding only `train.json`, because the move takes `train.ply` out of it. A run that fails before the move leaves no `out/train.ply` and no `train.json`, and when the PLY check fails, the file it refused stays in `work/train/`. The step runner empties `work/train/` before the step runs.

`train.ply` is the trained splat: a binary little-endian PLY in the standard 3DGS layout, the one gsplat's `export_splats(format="ply")` writes and splat-transform 3.8.0 reads. The step writes it with its own numpy code, in `splatpipe/train/ply.py`, which also fixes the layout's field names. The header is ASCII, each line ending in `\n`, with nothing else in it: no comment, no other element and no normals.

```
ply
format binary_little_endian 1.0
element vertex <N>
property float x
property float y
property float z
property float f_dc_0
property float f_dc_1
property float f_dc_2
property float f_rest_0
...                          one line for each of f_rest_1 to f_rest_43
property float f_rest_44
property float opacity
property float scale_0
property float scale_1
property float scale_2
property float rot_0
property float rot_1
property float rot_2
property float rot_3
end_header
```

Then come N records of 236 bytes, one per splat, each holding the 59 fields as little-endian float32 values, in the header's order:

| Fields | Byte offset in the record | What they hold |
|---|---|---|
| `x`, `y`, `z` | 0, 4, 8 | The splat's centre, in the levelled frame. |
| `f_dc_0` to `f_dc_2` | 12 to 20 | Band 0 of the spherical harmonics for R, G and B, with the mean exposure fix baked in. |
| `f_rest_0` to `f_rest_44` | 24 to 200 | Bands 1 to 3, 15 coefficients for each colour channel, channel by channel: `f_rest_k` is channel k // 15's coefficient k % 15, so R's 15 come first, then G's, then B's. |
| `opacity` | 204 | The opacity as a logit, before the sigmoid. |
| `scale_0` to `scale_2` | 208 to 216 | The log of the splat's size along each of its axes. |
| `rot_0` to `rot_3` | 220 to 232 | The rotation as a quaternion, w first, as the trainer holds it, not scaled to length 1. |

The header is 1,472 bytes for 1 to 9 splats and one byte longer for each extra digit of N. So a file is 1,471 bytes plus the digits of N plus 236 bytes per splat: 708,001,478 bytes for 3,000,000 splats.

While it trains, each training image has its own exposure fix: a 3 × 4 colour transform [A | b] that turns each rendered colour c into A·c + b, so the phone's changing exposure doesn't end up as blotches in the splat. Before the write, the step bakes the mean fix, the element-wise mean of every image's fix, into every splat's colours, so a viewer that shows the splat with no fix shows the average frame's exposure. Then it leaves out every splat with a NaN or an infinity in any of its 59 values, judging each value as the float32 it would be written as, so a baked colour too big for float32 leaves its splat out too. The splats left are written in the trainer's order.

`train.json`, the train record, holds the numbers that aren't whole and so can't be report counts. This example is from a run with a cap deadline:

```json
{
  "near_plane": 0.0912,
  "median_scene_depth": 3.04,
  "scene_extent": 4.87,
  "means_lr_start": 0.0007792,
  "width_ratio": 1.0,
  "height_ratio": 1.0,
  "mean_exposure": [[1.021, 0.004, -0.008, 0.012],
                    [0.002, 0.993, 0.006, -0.004],
                    [-0.006, 0.001, 1.017, 0.003]],
  "cap_deadline": "2026-11-21T14:25:30+00:00",
  "stop_time": "2026-11-21T14:10:30+00:00"
}
```

| Field | What it holds |
|---|---|
| `near_plane` | The near plane, `near_plane_factor` times the median scene depth. The renderer leaves out every splat nearer a camera than this. |
| `median_scene_depth` | The median scene depth: for each camera with at least one point in front of it, the median depth of the points in front of it, then the median of those. |
| `scene_extent` | The scene extent: `scene_extent_factor` times the largest distance from a camera centre to the cameras' mean position. |
| `means_lr_start` | The positions' learning rate at iteration 0, `means_lr` times the scene extent. |
| `width_ratio`, `height_ratio` | The training size's width and height over the shared camera's. 1.0 when the step didn't scale the images down. |
| `mean_exposure` | The mean exposure fix as 3 rows of 4 numbers, R's row first. Each row holds that row of the mean A, then the mean b's value for it. |
| `cap_deadline` | The cap deadline in UTC, in ISO 8601, or `null` when `SPLATPIPE_CAP_DEADLINE` isn't set. |
| `stop_time` | The stop time in UTC, or `null` with no cap deadline. |

Every value is plain JSON, and floats keep full precision.

### Config section

| Key | Table | Value | What it is |
|---|---|---|---|
| `long_edge` | `[train]` | `1600` | The longest edge of the images the step trains at, in pixels. A camera with a longer edge is scaled down to it, keeping its shape. A smaller camera isn't scaled up. |
| `mask_threshold` | `[train]` | `128` | A training pixel whose mask value, from 0 to 255, is below this is left out of the loss. A whole number from 1 to 255. |
| `seed` | `[train]` | `42` | The seed for the random subset of points, the draw of the splats that start from the depth priors, the order of the images and torch's random numbers. |
| `sh_degree_interval` | `[train]` | `1000` | The iterations between each rise of the spherical harmonics degree, which starts at 0 and stops at 3. |
| `scene_extent_factor` | `[train]` | `1.1` | The scene extent is this times the largest distance from a camera centre to the cameras' mean position. |
| `near_plane_factor` | `[train]` | `0.03` | The near plane is this times the median scene depth. The renderer leaves out splats nearer a camera than the near plane. |
| `start_opacity` | `[train]` | `0.5` | The opacity every splat starts at, above 0 and below 1. |
| `start_scale_factor` | `[train]` | `0.1` | A splat's starting size, the same on all three axes, is this times the root mean square of its distances to its `start_scale_neighbours` nearest other points. |
| `start_scale_neighbours` | `[train]` | `3` | How many nearest other points set a splat's starting size. |
| `neighbour_chunk_pairs` | `[train]` | `100000000` | The most point pairs one chunk of the nearest-point search holds on the GPU, about 400 MB. |
| `means_lr` | `[train]` | `0.00016` | The starting learning rate of the splats' positions, times the scene extent. |
| `means_lr_final` | `[train]` | `0.01` | The share of its start that the positions' learning rate falls to, exponentially, by the end of the last planned iteration. Above 0 and at most 1. |
| `scales_lr` | `[train]` | `0.005` | The learning rate of the splats' log scales. |
| `quats_lr` | `[train]` | `0.001` | The learning rate of the splats' rotations. |
| `opacities_lr` | `[train]` | `0.05` | The learning rate of the splats' opacity logits. |
| `sh_dc_lr` | `[train]` | `0.0025` | The learning rate of the splats' band-0 colours. |
| `sh_rest_lr` | `[train]` | `0.000125` | The learning rate of the splats' higher spherical harmonics. |
| `adam_eps` | `[train]` | `1e-15` | Adam's epsilon, for every optimiser. |
| `exposure_lr` | `[train]` | `0.001` | The learning rate of each image's exposure fix. |
| `l1_weight` | `[train]` | `0.8` | The weight of the L1 colour loss between the image and its render after the exposure fix, with the masked pixels left out. 0 turns it off. |
| `ssim_weight` | `[train]` | `0.2` | The weight of the SSIM loss, 1 − SSIM between the image and its render after the exposure fix, with the masked pixels left out. 0 turns it off. |
| `opacity_reg` | `[train]` | `0.01` | The weight of the splats' mean opacity, which fades splats the images don't need. 0 turns it off. |
| `scale_reg` | `[train]` | `0.01` | The weight of the splats' mean scale in units of the scene extent, which keeps them small. 0 turns it off. |
| `erank_reg` | `[train]` | `0.01` | The weight of the effective-rank term, which keeps splats from turning into needles. 0 turns it off. |
| `erank_epsilon` | `[train]` | `1e-5` | A small number inside the effective-rank term's log, so the term stays finite. |
| `erank_start_fraction` | `[train]` | `0.23333333333333334` | The share of the iterations from which the effective-rank term is on, the float nearest 7/30. |
| `refine_start_step` | `[train]` | `500` | The iteration after which MCMC starts moving and adding splats. |
| `refine_every` | `[train]` | `100` | The iterations between MCMC's refines. |
| `refine_stop_step` | `[train]` | `25000` | The iteration at which MCMC stops refining, at the latest. |
| `refine_stop_fraction` | `[train]` | `0.9` | MCMC also stops refining at this share of the iterations, when that comes first. |
| `clean_reserve_minutes` | `[train]` | `10` | The minutes before the cap deadline by which the splat must be written and checked, left for the clean step. |
| `save_margin_minutes` | `[train]` | `5` | How many minutes before that time the last iteration may start, leaving time to copy the splats off the GPU and to write and check the file. |
| `gpu_memory_slack` | `[train]` | `0.9` | The share of the preset's `min_gpu_memory_gb` that the GPU must report, because a "24 GB" card reports a little less. |
| `plain_gradient` | `[train]` | `0.02` | A training pixel whose brightness gradient, in grey from 0 to 1 per pixel, is below this is a plain pixel, where the depth loss's weight never drops below `depth_weight_floor`. Splats start from the depth priors only on plain pixels. Above 0. Untested on real footage. |
| `depth_gradient_falloff` | `[train]` | `10` | The depth loss's weight at a pixel is the iteration's depth weight times exp(−this × the pixel's brightness gradient), so it's lower on textured pixels. 0 or more. |
| `depth_weight_start` | `[train]` | `1.0` | The depth weight at the first iteration. Above 0. |
| `depth_weight_end` | `[train]` | `0.01` | The depth weight after the last planned iteration. It falls exponentially from `depth_weight_start` in between, over the planned iterations. Above 0. |
| `depth_weight_floor` | `[train]` | `0.1` | The lowest depth loss weight on a plain pixel: a plain pixel whose weight would be lower gets this, while a textured pixel's can go lower. 0 or more. |
| `depth_error_cap` | `[train]` | `1.0` | The largest relative inverse depth error one pixel adds to the depth loss, so a splat right in front of the camera can't take it over. Above 0. |
| `normal_weight` | `[train]` | `0.05` | The weight of the normal loss, between the splats' rendered normals and the priors' normals. 0 or more, and 0 turns it off. |
| `normal_start_step` | `[train]` | `3000` | The iteration at which the normal loss starts. A whole number, 0 or more. |
| `normal_every` | `[train]` | `4` | From `normal_start_step`, the normal loss runs on every this many iterations. A whole number above 0. |
| `flatten_reg` | `[train]` | `1.0` | The weight of the splats' mean smallest scale in units of the scene extent, which keeps them flat, with depth priors only. 0 or more, and 0 turns it off. |
| `depth_normal_weight` | `[train]` | `0.05` | The weight of the loss between the rendered depth's normals and the priors' normals. 0 or more, and 0 turns it off. |
| `normal_consistency_weight` | `[train]` | `0.015` | The weight of the loss between the splats' rendered normals and the rendered depth's normals. 0 or more, and 0 turns it off. |
| `depth_normal_start_fraction` | `[train]` | `0.23333333333333334` | The share of the iterations from which the two rendered-depth normal losses are on, the float nearest 7/30. From 0 to 1. |
| `backproject_share` | `[train]` | `0.3333333333333333` | The share of the preset's splat budget, `max_splats`, that may start from the depth priors, the float nearest 1/3. From 0 to 1. |
| `backproject_max` | `[train]` | `500000` | The most splats that may start from the depth priors. A whole number, 0 or more, and 0 turns that off. |
| `min_vcpus` | `[train]` | `4` | The fewest vCPUs the job's cloud machine needs for this step. |
| `min_memory_gb` | `[train]` | `32` | The least memory the job's cloud machine needs for this step, in GB. |
| `steps` | `[train.fast]`, `[train.default]` and `[train.quality]` | `7000`, `15000` and `30000` | The iterations the step trains for, the planned iterations. |
| `max_splats` | `[train.fast]`, `[train.default]` and `[train.quality]` | `1000000`, `2000000` and `3000000` | The splat budget: the most splats the step trains, MCMC's `cap_max`. |
| `min_gpu_memory_gb` | `[train.fast]`, `[train.default]` and `[train.quality]` | `24`, `24` and `48` | The least GPU memory the job's machine needs for this step, in GB. The step checks it. |

- An iteration worked out from a share of `steps` is rounded to the nearest whole iteration. So the effective-rank term is on from iteration 1,633 with `fast`, 3,500 with `default` and 7,000 with `quality`. MCMC refines every 100 iterations after iteration 500 and before iteration 6,300, 13,500 and 25,000. The spherical harmonics degree goes up by one every 1,000 iterations and reaches 3 at iteration 3,000.
- The step checks every setting before it reads the cap deadline or any file. A value of the wrong kind fails the step with a readable error that names the key, its table and what it must be, like "The setting mask_threshold in [train] is 0, but it must be a whole number from 1 to 255. Change it to a whole number from 1 to 255." A bool is never a number here. A missing key fails the step with the step runner's message, which names `[train]`, the preset's table and the key. The step also fails when `max_splats` is below `start_scale_neighbours` + 1, naming both settings, their tables and their values.

The 15 settings from `plain_gradient` to `backproject_max` are the depth prior settings. They set the prior loss terms, with their weights, the depth loss's weight per pixel and when each term is on, and how many splats may start from the depth priors. The step checks them and reports them in `settings` with or without depth priors. `plain_gradient`, `backproject_share` and `backproject_max` set the splats that start from the depth priors, as "What it reads" says, and `plain_gradient` and the other twelve set the prior loss terms below. Without depth priors, none of the 15 changes the run.

Without depth priors, that is with no `work/depth/depth.json` or with priors that failed a check, the loss holds only the five splat-training terms, `l1`, `ssim`, `opacity_reg`, `scale_reg` and `erank`, each on by its own rule: the first four whenever their weight isn't 0, and `erank` when its weight isn't 0, from the iteration the effective-rank term starts at. The trainer renders every training image in colour only, in gsplat's `"RGB"` mode, and renders no depth and no normals.

With depth priors, the trainer adds the five prior loss terms after those five, in this order: `depth`, `normal`, `flatten`, `depth_normal` and `normal_consistency`. On a prior view it renders the colours and the depth together, in gsplat's `"RGB+ED"` mode. The rendered depth z is gsplat's expected depth, a camera-space z, and it's 0 where no splat covers the pixel. Each term goes into the loss times its weight, and a mean over no pixels counts as 0. Every normal below is in the view's camera frame, faces the camera and has unit length. The prior pixels are the ones "Step: depth" lays out under "What it writes".

- `depth`, with no weight setting, so a weight of 1, on every iteration of a prior view. It's the relative inverse-depth L1: the mean, over the view's prior pixels, of the pixel's depth weight × min(|1/z − 1/z_prior| / (1/z_prior), `depth_error_cap`), where 1/z_prior is the prior's `inverse_depth` scaled up to the training size. So each pixel adds at most `depth_error_cap`, 1, times its weight, and a pixel where z isn't above 0 counts at the cap, so a splat right in front of the camera can't take the loss over. The depth weight is inside the term: the iteration's depth weight, which falls exponentially from `depth_weight_start` (1.0) at iteration 0 to `depth_weight_end` (0.01) after the last planned iteration, times exp(−`depth_gradient_falloff` × the pixel's brightness gradient). On a plain pixel it never goes below `depth_weight_floor` (0.1). A textured pixel's can go lower.
- `normal`, weight `normal_weight` (0.05), on iteration i of a prior view when i is at or after `normal_start_step` (3,000) and i − `normal_start_step` is a multiple of `normal_every` (4): iterations 3,000, 3,004, 3,008 and so on. It's the mean, over the prior pixels where the normal render's length is above 0, of 1 − the dot product of the normal render and the prior's normal, scaled up to the training size. The normal render is a second render of the same splats, with the same camera and near plane and no spherical harmonics, whose colour is each splat's normal: the axis of its smallest scale, turned into the camera frame and to face the camera. Each of its pixels whose length is above 0 is then set to unit length.
- `flatten`, weight `flatten_reg` (1.0), on every iteration, of a prior view or not, but only with depth priors. It's the mean over every splat of its smallest scale divided by the scene extent, so it's in units of the scene extent like `scale_reg`, and it keeps the splats flat.
- `depth_normal`, weight `depth_normal_weight` (0.05), on every iteration of a prior view from the depth-normal start. It's the mean, over the prior pixels that have a depth normal, of 1 − the dot product of the depth normal and the prior's normal. A pixel's depth normal comes from the rendered depth: each pixel's centre goes back through the view's camera at the training size to its z, and the depth normal is the unit cross product of those points' central differences across and down, turned to face the camera. A pixel on the image's border, one whose four neighbours don't all have a rendered depth above 0, and one whose cross product has length 0 have none.
- `normal_consistency`, weight `normal_consistency_weight` (0.015), on every iteration of a prior view from the depth-normal start. It's the mean, over the prior pixels that have a depth normal and where the normal render's length is above 0, of 1 − the dot product of the normal render and the depth normal, so the splats' normals agree with the surface their depth draws.

The depth-normal start is round(`depth_normal_start_fraction` × `steps`), rounded like the effective-rank term's start: iteration 1,633 with `fast`, 3,500 with `default` and 7,000 with `quality`. A term whose weight is 0, or whose iteration rule is off, counts as 0 and isn't worked out. The trainer makes the normal render only on the iterations where `normal` or `normal_consistency` is on, and the depth normals only where `depth_normal` or `normal_consistency` is on. A training image with no prior, in a run with depth priors, is rendered in the `"RGB"` mode with no depth, and its iterations work out the five splat-training terms and `flatten`, with `depth`, `normal`, `depth_normal` and `normal_consistency` counted as 0. From the depth-normal start on, each iteration of a prior view renders twice, once for the colours and the depth and once for the normals, so those iterations take longer.

`min_vcpus`, `min_memory_gb` and `min_gpu_memory_gb` are the step's machine needs: the job's cloud machine needs at least 4 vCPUs, 32 GB of memory and a 24 GB GPU for `fast` and `default`, or a 48 GB GPU for `quality`. The cloud spec reads them to pick the machine. Of the three, the step itself checks only the GPU's memory, against the GPU need: `gpu_memory_slack` × `min_gpu_memory_gb` × 10⁹ bytes, since a GB is 10⁹ bytes here, as in `[cloud.types]`. So `fast` and `default` take a GPU whose total memory, as torch reports it, is at least 21.6 GB, and `quality` one with at least 43.2 GB, which no 24 GB card has. The step checks `min_vcpus` and `min_memory_gb` like the other settings, but never uses them.

### Step report fields

- `settings`: the 50 settings in `[train]` and the preset's `steps`, `max_splats` and `min_gpu_memory_gb`, with or without depth priors.
- `timings.load`: reading the camera solve output, reading and checking the depth priors, and setting up the scene and the starting splats, the splats from the depth priors and the nearest-point search on the GPU included.
- `timings.train`: the trainer, the upload of the images to the GPU and the copy of the splats back included.
- `timings.write_ply`: the exposure bake, leaving out the splats with a NaN or an infinity, and writing `work/train/train.ply`.
- `timings.check_ply`: the PLY check.
- `counts.steps_planned`: the planned iterations, the preset's `steps`.
- `counts.gpu_memory_bytes`: the GPU's total memory in bytes, as torch reports it. The step adds it only when torch sees a CUDA GPU.
- `counts.images`: the images in `images.bin`, which are the training images.
- `counts.sparse_points`: the points in `points3D.bin`.
- `counts.image_width` and `counts.image_height`: the training size, in pixels.
- `counts.masked_images`: the training images with a mask.
- `counts.prior_images`: the prior views, the training images with a depth prior. The step adds it only when it has read and checked the depth priors, so a run with no `work/depth/depth.json`, or with priors that failed a check, has no `prior_images`.
- `counts.backprojected_splats`: the splats that started from the depth priors, the smaller of their budget and the number of candidate pixels, as "What it reads" says. The step adds it whenever it has read and checked the depth priors, even when it's 0, and only then, so a run without them has no `backprojected_splats`.
- `counts.start_splats`: the splats the training started from. With depth priors, it counts the sparse model's points used and the splats from the depth priors together.
- `counts.steps_run`: the iterations run.
- `counts.sh_degree`: the spherical harmonics degree of the last iteration, 0 when none ran.
- `counts.peak_gpu_memory_bytes`: the most GPU memory torch's allocator held at once from the GPU check on, from `torch.cuda.max_memory_allocated()`.
- `counts.splats`: the splats the step wrote: those the trainer gave back, less those left out for a NaN or an infinity.
- `counts.ply_bytes`: the trained splat's size in bytes.
- `warnings`, each only when it applies:
  - When the depth priors fail a check of "What it reads": "The depth priors in work/depth/ weren't used, because <file> <what>. Training ran without depth priors.", where <file> is the path of the file that failed and <what> says what failed, like "has the camera fx 1450.0, but cameras.bin's is 1452.5". The step then trains without depth priors. With no `work/depth/depth.json` there's no warning.
  - Without depth priors, when the camera solve gave more points than the splat budget: "The camera solve gave <points> points, more than this preset's splat budget of <max_splats>, so training started from <max_splats> of them, chosen at random."
  - With depth priors, in place of the one above, when the camera solve gave more points than the splat budget leaves after the splats from the depth priors: "The camera solve gave <points> points, more than the <left> places this preset's splat budget of <max_splats> leaves after <backprojected_splats> splats started from the depth priors, so training started from <left> of the points, chosen at random.", where <left> is `max_splats` less `counts.backprojected_splats`.
  - When the trainer stopped at the stop time, before the last planned iteration: "Training stopped early, after <steps_run> of <steps> iterations, to leave time for the clean step before the job's time cap."
  - When the step left out splats that held a NaN or an infinity: "Splats left out of out/train.ply because they held a NaN or an infinity: <n>.", where <n> is how many it left out, and `counts.splats` counts the splats it wrote. A run that leaves out every splat writes a file of no splats and fails at the PLY check's "holds no splats", with this warning in its report.
- `error`, when it failed. The step raises `ExpectedFailure` for each failure it foresees, so `expected` is `true` and `readable` is the step's own message, saying what went wrong and what to do. Those failures are:
  - a bad or missing setting, or `max_splats` below `start_scale_neighbours` + 1;
  - a bad `SPLATPIPE_CAP_DEADLINE`;
  - a camera solve output that's missing or that the step refuses, as "What it reads" lists, too few points included;
  - a machine without torch, gsplat or fused-ssim, one where torch sees no CUDA GPU, a gsplat other than 1.5.3, or a GPU with less memory than the preset needs, like "This GPU has 24.1 GB of memory, but the quality preset needs a 48 GB GPU, and the step accepts 0.9 of that, 43.2 GB, because a card reports a little less than its size. Run the job with a preset that needs less GPU memory, or on a machine with a larger GPU.";
  - cameras that all sit in one place, no point in front of any camera, or points that all sit on top of each other;
  - torch running out of GPU memory in the nearest-point search or in training. The error gives the iteration it reached, or says it happened while the step set up the scene, gives the preset's `max_splats`, and says to use a preset with a smaller splat budget or a GPU with more memory;
  - no time left to train, because the stop time came before the first iteration;
  - a file that can't be written or moved, like on a full disk;
  - a trained splat that fails the PLY check: no splats, a header other than the layout's, a size other than the header's plus 236 bytes per splat, or a value that isn't finite, with how many splats hold one and each field's count.

A loss that stops being finite is a bug in the training, not a problem with the video. The trainer checks the loss after each iteration's backward pass, before any optimiser step, so the bad values never reach a splat, and it stops at once. The step then fails with the general bug message and `expected` is `false`. The message in `technical` gives the iteration, the value of every loss term in the run's loss table, each before its weight and 0 for a term that was off on that iteration, and the total. Without depth priors, the loss table holds the five splat-training terms, `l1`, `ssim`, `opacity_reg`, `scale_reg` and `erank`. With depth priors, the five prior loss terms, `depth`, `normal`, `flatten`, `depth_normal` and `normal_consistency`, come after them. Any other error is a bug too, with the general message.

The counts are whole numbers, so the scene's distances, the starting learning rate and the mean exposure fix are in `train.json` instead. The step adds each count as soon as it knows it, in the order above, so a failed report holds the counts and timings the step had worked out before it failed.

### Files in out/

- `out/train.ply`: the trained splat, in the layout under "What it writes". The step removes it when it starts, so a run that fails leaves none, not even an earlier run's. It writes it whole once the PLY check passes: one move takes the checked file from `work/train/`, so `out/` never holds a half-written splat. The clean step reads it from there, and it's the job's result when the clean step fails or doesn't run. The largest splat budget, `quality`'s 3,000,000 splats, is within the 8,000,000 splats the clean step takes, its `max_source_splats`.

Its report, `out/reports/train.json`, comes from the step runner, like every step's.

## Step: clean

The clean step is in `splatpipe/clean/`. It reads the splat the training step wrote, drops the Gaussians a viewer shouldn't see and writes the rest as the clean splat, a binary PLY. Then it makes the viewer copy, a small `.sog` for the web viewer, from the clean splat with PlayCanvas splat-transform. These two files are what the person who filmed the place gets back. The step is built like the fake step. It's the step list's last entry, an optional one, and it also runs on its own.

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

When the step fails, the trained splat is still in `out/`, as it was. The clean step is an optional step in the step list, so a failed clean ends the job done with warnings.

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
| `sog_sh_bands` | `[clean.fast]`, `[clean.default]` and `[clean.quality]` | `0`, `3` and `3` | The bands of higher spherical harmonics the viewer copy keeps, from 0 to 3. The fast preset keeps none, because splat-transform's spherical harmonic compression runs on the CPU and took over an hour on the first real job's 912,131 Gaussians, while with 0 bands it takes seconds. |
| `sh_iterations` | `[clean.fast]`, `[clean.default]` and `[clean.quality]` | `10`, `10` and `20` | The iterations of splat-transform's spherical harmonic compression. |

All three presets clean the splat the same way, and only `sog_sh_bands` and `sh_iterations` differ between their viewer copies. The clean splat, `out/clean.ply`, keeps every band in every preset. The real `config.toml` has a comment on the line directly above every key.

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
