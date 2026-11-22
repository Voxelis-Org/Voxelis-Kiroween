# Implementation Plan: cloud-jobs

## Overview

This plan builds Cloud_Jobs, the cloud wrapper under `cloud/`, in the design's build order of 9 tasks: the `[cloud]` Config section and its reader plus the dependencies; redaction, errors, the AWS seam and the Stack resolver; machine-type selection; the launch request and the User_Data; the Start_Command; the machine-side Cloud_Runner; the Image_Build and the Smoke_Test; the free Status, Download, List and Stop_All commands with the Spend_File merge; and last the docs, steering, skill and import checks. Each task is a vertical slice: it writes its module(s) and the unit and property tests that cover them in the same task, runs the whole suite, and commits by name. The code wraps the Pipeline's public functions (`make_job`, `run_job`) and changes nothing in `splatpipe/`.

The three mandated property tests land in the tasks that own them: the launch-request shape (Property 4, Req 8.9) in task 4, nothing-paid-before-the-Spend_Guard (Property 6, Req 6.6) in task 5, and resume-from-the-first-unfinished-Step (Property 7, Req 10.4) in task 6. The other properties are Property 1 (redaction, task 2), Property 2 (Config, task 1), Property 3 (selection, task 3), Property 8 (uploaded set, task 6) and Property 9 (Spend_File merge, task 8).

Some test and code files grow over several tasks, so each piece comes with the first task that uses it:

- `tests/fake_aws.py`: task 2 writes `FakeAws` and the `cloud_config(**overrides)` helper, task 5 adds the `git_repo(tmp_path)` helper.
- `cloud/spend_file.py`: task 5 writes the quote helpers, task 6 the machine-side writing, task 8 the laptop merge.
- `tests/test_cloud_docs.py`: task 1 tests the dependency and licence parts it adds, task 9 the doc, steering and skill parts.
- `docs/third-party-licences.md`: task 1 adds the boto3 tree and the npm rows; task 7 adds the Machine_Image tool rows, in the same commit as the `install_script` that names those tools (`.kiro/steering/rules.md`: the licence row lands in the same commit as the tool, which matters most for the GPL-sensitive ffmpeg and COLMAP builds).

## Working rules for every task

- Work on `feat/cloud-runner`, never on `main`.
- All new code lives under `cloud/`. No task changes any file under `splatpipe/`, any pipeline-core or clean-export test, or `cloud/aws-setup.yaml`. `splatpipe.run.STEP_LIST` stays `()`, so no real Job can run and no pipeline-core test changes; the Stack already exposes every output and permission Cloud_Jobs needs, so the template is untouched (Req 1.10, 16.6).
- boto3 is imported only inside `cloud/`, and only in `cloud/aws.py`. A `cloud` module may import `splatpipe` to call its public functions, never the reverse. No `cloud` module imports torch, gsplat, fused-ssim or any other GPU package, so every `cloud` module imports on a laptop.
- Every number and setting comes from `config.toml`. The `[cloud]` prices, each type's `memory_gb` and `gpu_memory_gb`, and `root_gb = 150` are the AWS us-east-1 Linux On-Demand values published 2026-11-21, each type carrying that `price_date`; write exactly those, don't re-derive them. The fallback-list ordering beyond the pinned primaries, and the Smoke_Marker gate as a hard refusal in the Start_Command, are kept as the design states.
- The AWS seam is a narrow `Aws` protocol in `cloud/aws.py`. `BotoAws` is the one real implementation and the only place a `boto3.Session` is built; tests pass a `FakeAws` that implements the same protocol over in-memory dicts and records an ordered `calls` timeline. Following the design, nothing paid happens before `check_spend` returns, where "paid" is the first `s3_upload` and the `run_instances` call; the free local and read-only work before it (the Code_Archive, `make_job`'s local folder, the quota, image and marker reads for the money quote) is not a paid run.
- Tests run on the CPU with no network, no GPU and no AWS account, against the `FakeAws` seam and the Fake_Step, building any tiny synthetic data in the test. No task calls AWS, sets `APPROVED_SPEND`, or runs anything paid, and no test runs a raw `aws` command. The paid commands (Start, Image_Build, Smoke_Test) are built and unit-tested against `FakeAws` only; a real run happens later, by hand, after the person in the chat says yes, with `APPROVED_SPEND=1` set on that one command — never in these tasks (Req 16.1, 16.8).
- Tests reuse pipeline-core's `tests/conftest.py` fixtures, which point `splatpipe.config.CONFIG_PATH` at a temporary `config.toml` whose `jobs_folder` is inside `tmp_path`. No test writes into the real `data/` folder, and any config a test builds itself keeps `jobs_folder` inside `tmp_path`. The shared cloud test helpers (`FakeAws`, `cloud_config`, `git_repo`) live in `tests/fake_aws.py`; no existing conftest fixture or pipeline-core test is changed.
- Subprocess and CLI tests pass `timeout=60`, set `PYTHONIOENCODING=utf-8` in the child's environment, and read its output as UTF-8.
- Property tests use Hypothesis with at least 100 examples (the default), and each docstring says `Feature: cloud-jobs, Property N: <title>`. A property test that touches the filesystem uses `@settings(deadline=None)` so a slow disk can't make it flaky.
- Every test sub-task is required. None is optional (`.kiro/steering/rules.md`: the unit test and property test tasks are always required).
- Whoever adds a dependency, tool, model weight or dataset adds its `docs/third-party-licences.md` row in the same task and commit, and stops to ask the user before committing if any package has a copyleft, non-commercial or unclear licence (Req 14.6).
- The last sub-task of each task is its checkpoint. Run `uv run pytest`; every test passes, `tests/test_guard.py` and the unchanged pipeline-core tests included. Then stage the task's files by name with `git add <file> ...`, never with `git add -A`, `git add .` or a folder, and commit only those files with `git commit -m "<message>" -- <file> ...`. Leave `docs/devlog.md` out: the devlog hook adds and commits its own line after each task.
- If a hook blocks something it shouldn't, say so and fix the hook. Never work around it.

## Tasks

- [x] 1. The `[cloud]` config section, the Cloud_Config and the dependencies
  - Files: `config.toml`, `pyproject.toml`, `uv.lock`, `package.json`, `package-lock.json`, `docs/third-party-licences.md` (changed); `cloud/config.py`, `tests/test_cloud_config.py`, `tests/test_cloud_docs.py` (new).
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 2.10, 14.1, 14.2, 14.3, 14.4, 14.6_

  - [x] 1.1 Append the `[cloud]` section to `config.toml`
    - Append `[cloud]` as the design's "Sections added to `config.toml`" gives it, with a comment above every key and table, which the Config_Loader's Pipeline_Section comment check requires. There is no `[cloud.needs.*]` table: a Step's needs live in that Step's own section.
    - A quoted `[cloud.types."<type>"]` Machine_Spec table for each of the 8 Allowed_Types, each with `vcpus`, `memory_gb`, `gpu_memory_gb`, `price_per_hour_usd` and `price_date = "2026-11-21"`. Write exactly these us-east-1 Linux On-Demand values as vcpus / memory_gb / gpu_memory_gb / price_per_hour_usd: g5.xlarge 4 / 16 / 24 / 1.006, g5.2xlarge 8 / 32 / 24 / 1.212, g5.4xlarge 16 / 64 / 24 / 1.624, g5.8xlarge 32 / 128 / 24 / 2.448, g6.2xlarge 8 / 32 / 24 / 0.9776, g6.4xlarge 16 / 64 / 24 / 1.3232, g6e.2xlarge 8 / 64 / 48 / 2.24208, g6e.4xlarge 16 / 128 / 48 / 3.00424 (A10G 24 GB on g5, L4 24 GB on g6, L40S 48 GB on g6e).
    - A `[cloud.presets.<preset>]` table for fast, default and quality, each with an ordered `fallback`, `expected_minutes` and `cap_minutes` (fast 30/120, default 60/180, quality 120/360; primaries g5.8xlarge for fast and default, g6e.4xlarge → g6e.2xlarge for quality). `[cloud.smoke]` and `[cloud.image]` with expected/cap (10/30 and 45/120). The scalars `upload_margin_minutes`, `poll_seconds`, `ipv4_usd_per_hour`, `gp3_usd_per_gb_month`, `root_gb = 150`, `smoke_instance_type`, `image_builder_type` and `base_ami_parameter`.
    - Leave `[git]`, `[aws]`, `[paid]`, `[devlog]`, `[jobs]`, `[fake]` and the Fake_Step Preset tables exactly as they are.
    - _Requirements: 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9_

  - [x] 1.2 Write `cloud/config.py`
    - `CloudConfig`, `MachineSpec`, `RunPlan`, `PresetPlan`, `load_cloud_config()`, `CloudConfig.from_mapping()` and the accessors `spec()` and `preset_plan()`, as in the design's "cloud/config.py" section. `MachineSpec` has `vcpus`, `memory_gb`, `gpu_memory_gb`, `price_per_hour_usd` and `price_date`; there is no `StepNeeds` type and no `[cloud.needs]` reader.
    - `load_cloud_config` calls `splatpipe.config.load_config(path)` first so the Config_Loader refuses NaN, infinity and TOML dates in `[cloud]` like any Pipeline_Section, then re-reads the file's `[cloud]` table with `tomllib`. Every missing or wrong-typed key is a `CloudError` naming the key and what it must be; prices finite and `> 0`, minutes `> 0`, `memory_gb` and `gpu_memory_gb` finite and `> 0`, `price_date` a non-empty string.
    - Check the cross-table rules: every `fallback` entry is a key of `types`, `smoke_instance_type` and `image_builder_type` are keys of `types`, and there are exactly the three Presets. Add no accessor to `splatpipe/config.py`.
    - _Requirements: 2.1, 2.2, 2.10_

  - [x] 1.3 Add boto3, the npm lockfiles and their Licence_List rows
    - `uv add boto3` so the Project_File gains boto3 as a main runtime dependency and `uv.lock` updates; add no GPU package, so `uv sync` on a laptop still installs no torch, gsplat, fused-ssim or CUDA wheel.
    - Create root `package.json` and `package-lock.json` pinning splat-transform 3.8.0 for the Machine_Image's `npm ci`; the laptop tests never run npm.
    - In `docs/third-party-licences.md`, mark boto3 installed and add a row for each package `uv.lock` gained (botocore, s3transfer, jmespath, python-dateutil, six, urllib3), each with its permissive licence, and a row for each package in the generated `package-lock.json` that the splat-transform row doesn't already cover. Stop and ask the user before committing if any is copyleft, non-commercial or unclear.
    - _Requirements: 14.1, 14.2, 14.3, 14.4, 14.6_

  - [x] 1.4 Write unit tests for the Cloud_Config in `tests/test_cloud_config.py`
    - The real `config.toml` `[cloud]` loads; there are 8 `types`, each with a full Machine_Spec (`vcpus`, `memory_gb`, `gpu_memory_gb`, `price_per_hour_usd`, `price_date`); each Preset `fallback` is non-empty and every entry is an Allowed_Type; the pinned primaries and fallbacks and the Q2 minutes are present; `[cloud]` holds no `[cloud.needs.*]` table; `smoke_instance_type` and `image_builder_type` are Allowed_Types.
    - A `[cloud]` with `nan`, `inf` or a TOML date is refused by the Config_Loader naming the key; a missing or wrong-typed key is a `CloudError` naming the key.
    - _Requirements: 2.3, 2.4, 2.6, 2.7, 2.10_

  - [x] 1.5 Write the property test for Property 2 in `tests/test_cloud_config.py`, with Hypothesis
    - **Property 2: Cloud_Config refuses a corrupted key by name**
    - **Validates: Requirements 2.1, 2.2**
    - For any valid `[cloud]` mapping and any single corruption of it (a required key removed, or one key given a wrong type, NaN, infinity or a TOML date where a number belongs), `load_cloud_config` raises an error whose message names the corrupted key, while the un-corrupted mapping loads to a `CloudConfig`.
    - _Requirements: 2.1, 2.2_

  - [x] 1.6 Write the dependency and licence tests in `tests/test_cloud_docs.py`
    - `pyproject.toml` has boto3 in the main dependencies and no GPU package; `package.json` and `package-lock.json` pin splat-transform 3.8.0.
    - The Licence_List has rows for boto3, botocore, s3transfer, jmespath, python-dateutil, six and urllib3, and a row for each package in `package-lock.json`.
    - _Requirements: 14.1, 14.2, 14.3, 14.4_

  - [x] 1.7 Run the tests and commit
    - Run `uv run pytest`; every test passes, `tests/test_guard.py` included.
    - Stage by name: `config.toml`, `cloud/config.py`, `pyproject.toml`, `uv.lock`, `package.json`, `package-lock.json`, `docs/third-party-licences.md`, `tests/test_cloud_config.py`, `tests/test_cloud_docs.py`; commit with the message "Cloud jobs: the [cloud] config section, the Cloud_Config and the dependencies".
    - _Requirements: 16.1, 16.2, 16.8_

- [x] 2. Redaction, errors, the AWS seam and the Stack
  - Files: `cloud/redact.py`, `cloud/errors.py`, `cloud/aws.py`, `cloud/stack.py`, `tests/fake_aws.py`, `tests/test_cloud_redact.py`, `tests/test_cloud_aws.py`, `tests/test_cloud_stack.py`, all new.
  - _Requirements: 1.2, 1.3, 1.6, 1.7, 1.8, 1.9, 5.5, 10.10, 13.6, 13.7_

  - [x] 2.1 Write `cloud/redact.py`
    - `account_redaction(text)` using `(?<!\d)\d{12}(?!\d)` to replace every maximal run of exactly 12 digits with `<account>`, leaving an 11- or 13-digit run alone, as in the design's "cloud/redact.py" section.
    - _Requirements: 13.7_

  - [x] 2.2 Write `cloud/errors.py`
    - `CloudError`, `CapacityUnavailable` (with `code`, `zone`, `instance_type`) and `run_cloud_cli(action)`, as in the design's "cloud/errors.py" section. `run_cloud_cli` mirrors `splatpipe.errors.run_cli` but also catches `SpendRefused` (from `cloud.spend`) and `CloudError`, prints every message through `account_redaction`, prints no traceback, and lets `SystemExit` and `KeyboardInterrupt` pass through.
    - _Requirements: 1.9, 5.5, 13.6_

  - [x] 2.3 Write `cloud/aws.py` and `tests/fake_aws.py`
    - `make_session()`, the `Aws` protocol, `BotoAws`, `laptop_aws()`, `machine_aws()`, and the `InstanceInfo` and `ImageInfo` data classes, as in the design's "The Fake_AWS seam" section. `make_session` is the only place a `boto3.Session` is built. `laptop_aws` names the region from `[aws].region` and the profile from `[aws].<profile_key>`, never `AWS_PROFILE`; `machine_aws` names only the region. `BotoAws` is the only module importing boto3, converts a capacity `ClientError` (`InsufficientInstanceCapacity`/`Unsupported`) into `CapacityUnavailable` and any other AWS error into a `CloudError` carrying `account_redaction(str(error))`, and filters every describe by the Project_Tag.
    - `tests/fake_aws.py`: `FakeAws`, implementing the `Aws` protocol over in-memory dicts and recording an ordered `calls` list of `(op, key)` tuples, plus the `cloud_config(**overrides)` helper that builds a valid `CloudConfig` from a dict.
    - _Requirements: 1.2, 1.6, 1.7, 1.8, 1.9, 5.5, 16.1_

  - [x] 2.4 Write `cloud/stack.py`
    - `StackOutputs` (the seven outputs) and `resolve_stack(aws, stack_name)`, as in the design's "cloud/stack.py" section. It maps the seven keys from DescribeStacks, a missing output or stack is a `CloudError` naming what's missing, and it holds no bucket name, id or ARN literal.
    - _Requirements: 1.3_

  - [x] 2.5 Write the property test and examples for redaction in `tests/test_cloud_redact.py`, with Hypothesis
    - **Property 1: Account_Redaction clears exactly 12-digit runs**
    - **Validates: Requirements 13.6, 13.7, 1.9, 10.10**
    - For any string, `account_redaction` replaces every maximal run of exactly 12 digits with `<account>`, leaves every other-length digit run unchanged, changes no non-digit character, and is idempotent, so the result holds no maximal 12-digit run.
    - Examples: a real-shaped ARN's 12-digit field becomes `<account>`; an 11- and a 13-digit number survive; a 12-digit number that is not an account id is still replaced.
    - _Requirements: 13.6, 13.7, 1.9, 10.10_

  - [x] 2.6 Write unit tests in `tests/test_cloud_aws.py` and `tests/test_cloud_stack.py`
    - aws: with `boto3.Session` patched to record its kwargs, `laptop_aws` names the region and the chosen profile and never `AWS_PROFILE`; `machine_aws` names the region and no profile; with botocore stubbed, `BotoAws.run_instances` turns a capacity `ClientError` into `CapacityUnavailable` and any other into a redacted `CloudError`.
    - stack: `resolve_stack` maps the seven outputs from a `FakeAws` DescribeStacks; a missing output is a `CloudError` naming it; a structural check that no `cloud/*.py` holds a 12-digit literal or an `arn:` literal.
    - _Requirements: 1.2, 1.3, 1.6, 1.7, 1.8, 5.5_

  - [x] 2.7 Run the tests and commit
    - Run `uv run pytest`; every test passes.
    - Stage `cloud/redact.py`, `cloud/errors.py`, `cloud/aws.py`, `cloud/stack.py`, `tests/fake_aws.py`, `tests/test_cloud_redact.py`, `tests/test_cloud_aws.py`, `tests/test_cloud_stack.py` by name; commit with the message "Cloud jobs: redaction, errors, the AWS seam and the Stack".
    - _Requirements: 16.1, 16.2, 16.8_

- [x] 3. Selecting the machine
  - Files: `cloud/select.py` and `tests/test_cloud_select.py`, both new.
  - _Requirements: 2.7, 3.1, 3.2, 3.3, 3.4, 3.5, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 5.1_

  - [x] 3.1 Write `cloud/select.py`
    - `Needs`, `step_needs_over()`, `fits_hard_needs()`, `select_candidates()`, `running_vcpus()`, `admit()` and `launch_plan()`, all pure, as in the design's "cloud/select.py" section.
    - `step_needs_over(step_names, config, preset)` reads each Step's own section for the Preset through `splatpipe.config.Config.step_settings(name, preset)` and takes the largest `min_memory_gb`, the largest `min_gpu_memory_gb` and the largest `min_vcpus` (soft); a key a Step omits counts as 0, so a missing key means no need; an empty list gives zero needs. `fits_hard_needs` treats `memory_gb` and `gpu_memory_gb` as hard (`spec.memory_gb >= needs.min_memory_gb and spec.gpu_memory_gb >= needs.min_gpu_memory_gb`) and never checks vCPUs. `select_candidates` keeps the Preset's Fallback_List in order, dropping types that fail the hard needs, and raises naming the unmet need when none fit. `admit` keeps candidates where `running + spec.vcpus <= quota`, raises pointing to the Setup_Guide when the quota is 0, and raises giving the quota, the running total and the vCPUs when nothing fits. `launch_plan` returns ordered `(type, subnet, zone)` attempts, each admitted type then each zone that offers it and has a default subnet.
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 5.1_

  - [x] 3.2 Write unit tests in `tests/test_cloud_select.py`
    - `admit` with a quota of 0 raises pointing to the Setup_Guide; `select_candidates` raises naming the unmet memory or GPU-memory need; `step_needs_over` reads `min_vcpus`/`min_memory_gb`/`min_gpu_memory_gb` from a Step's own section for a Preset and treats a missing key as no need; `launch_plan` orders types first, then zones.
    - _Requirements: 3.4, 4.4, 4.5, 5.1, 2.7_

  - [x] 3.3 Write the property test for Property 3 in `tests/test_cloud_select.py`, with Hypothesis
    - **Property 3: Selection keeps the right types in order**
    - **Validates: Requirements 3.1, 3.2, 3.3, 4.2, 4.3, 4.5**
    - For any Machine_Specs, Fallback_List, aggregate `Needs`, GPU quota and running-vCPU total, `admit(select_candidates(...), running, quota, cc)` returns a sublist of the Fallback_List in its original order with a type present exactly when its `memory_gb` is at least `min_memory_gb`, its `gpu_memory_gb` is at least `min_gpu_memory_gb`, and `running + its vcpus <= quota`; its vCPU count alone never removes it; when no type qualifies the call raises.
    - _Requirements: 3.1, 3.2, 3.3, 4.2, 4.3, 4.5_

  - [x] 3.4 Run the tests and commit
    - Run `uv run pytest`; every test passes.
    - Stage `cloud/select.py` and `tests/test_cloud_select.py` by name; commit with the message "Cloud jobs: selecting the machine".
    - _Requirements: 16.1, 16.2, 16.8_

- [x] 4. The launch request and the User_Data
  - Files: `cloud/launch.py`, `cloud/userdata.py`, `tests/test_cloud_launch.py`, `tests/test_cloud_userdata.py`, all new.
  - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8, 8.9, 10.7_

  - [x] 4.1 Write `cloud/launch.py`
    - `build_run_instances_request()`, `client_token()` and `launch_with_fallback()`, as in the design's "cloud/launch.py" section. The request dict carries `MinCount == MaxCount == 1`, `InstanceInitiatedShutdownBehavior=terminate`, IMDSv2 `MetadataOptions`, the instance-profile ARN, the security group, the subnet, the Project_Tag on the instance, the volume and the network interface (and the Job_Id tag on the instance only when a Job_Id is given), the User_Data, a per-attempt client token, and no `KeyName`, `LaunchTemplate`, `InstanceMarketOptions` or `DisableApiTermination` key.
    - `launch_with_fallback` tries each `(type, subnet, zone)` in order: on `CapacityUnavailable` it moves to the next attempt; on any other error (a `CloudError`) it stops and raises, starting nothing more; on success it returns the instance, its type and its zone; an exhausted plan is a `CloudError` listing the types and zones tried. `client_token` gives a fresh token per attempt.
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7_

  - [x] 4.2 Write `cloud/userdata.py`
    - `CAP_DEADLINE_ENV = "SPLATPIPE_CAP_DEADLINE"`, `UserDataValues` and `render_user_data()`, as in the design's "cloud/userdata.py" section. The rendered bash script begins `#!/bin/bash`, uses only `\n` line endings, is at most 16384 bytes, places the `poweroff` trap before the Time_Cap `shutdown` line and the cap line before any download or run, and `shlex.quote`s every interpolated value after checking it (the Job_Id through `splatpipe.jobs.check_job_id`, the bucket, region and hashes against safe patterns, the minutes as whole numbers). It raises a `CloudError` if the text ever exceeds 16 KB.
    - _Requirements: 8.7, 8.8, 10.7_

  - [x] 4.3 Write unit tests for the fallback loop in `tests/test_cloud_launch.py`
    - Over a `FakeAws`: capacity in zone A fails then zone B succeeds; one type's `Unsupported` moves to the next type; an ambiguous error stops the loop with nothing launched; exactly one successful `run_instances`; the client token differs per attempt.
    - _Requirements: 5.1, 5.3, 5.4, 5.5, 5.6, 5.7_

  - [x] 4.4 Write the property test for Property 4 in `tests/test_cloud_launch.py`, with Hypothesis
    - **Property 4: Every launch request carries the required settings**
    - **Validates: Requirements 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.9**
    - For any image id, Allowed_Type, subnet, instance-profile ARN, security-group id, Job_Id (or none) and User_Data, `build_run_instances_request` returns a dict with `MinCount == MaxCount == 1`, `InstanceInitiatedShutdownBehavior == "terminate"`, the IMDSv2 `MetadataOptions`, the given profile ARN, security group and subnet, the Project_Tag on the instance, the volume and the network interface (and the Job_Id tag on the instance exactly when a Job_Id is given), the given User_Data, and no `KeyName`, `LaunchTemplate`, `InstanceMarketOptions` or `DisableApiTermination` key.
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.9, 16.5_

  - [x] 4.5 Write the property test and examples for the User_Data in `tests/test_cloud_userdata.py`, with Hypothesis
    - **Property 5: Every User_Data script is safe and complete**
    - **Validates: Requirements 8.7, 8.8, 10.7**
    - For any valid `UserDataValues`, `render_user_data` returns text that begins `#!/bin/bash`, uses only `\n` line endings, is at most 16384 bytes, places the `poweroff` trap before the Time_Cap line and the cap line before any download or run, carries the Time_Cap from `cap_minutes`, and interpolates every value `shlex.quote`d so splitting the line with `shlex` recovers the original value.
    - Examples: a tricky Job_Id-shaped value is quoted; a value failing its pattern raises; the Cap_Deadline variable name is the documented `CAP_DEADLINE_ENV` constant.
    - _Requirements: 8.7, 8.8, 10.7_

  - [x] 4.6 Run the tests and commit
    - Run `uv run pytest`; every test passes.
    - Stage `cloud/launch.py`, `cloud/userdata.py`, `tests/test_cloud_launch.py`, `tests/test_cloud_userdata.py` by name; commit with the message "Cloud jobs: the launch request and the User_Data".
    - _Requirements: 16.1, 16.2, 16.8_

- [x] 5. The Start_Command
  - Files: `cloud/repo.py`, `cloud/spend_file.py`, `cloud/start.py`, `tests/test_cloud_repo.py`, `tests/test_cloud_start.py` (new); `tests/fake_aws.py` (changed, add `git_repo`).
  - _Requirements: 1.5, 3.4, 3.6, 3.7, 4.5, 5.3, 6.1, 6.2, 6.3, 6.4, 6.6, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 7.9, 12.3_

  - [x] 5.1 Write `cloud/repo.py` and the `git_repo` helper
    - `uncommitted()`, `head_commit()`, `archive_head()` and `lock_hashes()`, as in the design's "cloud/repo.py" section, shelling out to `git`. `uncommitted` reports paths among `splatpipe/`, `cloud/` and `config.toml` with changes; `archive_head` writes a `git archive HEAD` tar; `lock_hashes` gives the sha-256 of `HEAD`'s `uv.lock` and `package-lock.json`.
    - Add `git_repo(tmp_path)` to `tests/fake_aws.py`: it makes a tiny committed repo and skips with a reason when `git` isn't found.
    - _Requirements: 7.3, 7.4, 7.5, 7.6_

  - [x] 5.2 Write the quote helpers in `cloud/spend_file.py`
    - `estimated_cost()`, `rough_cost()`, `spend_line()` and the labels `LABEL_MACHINE`, `LABEL_IMAGE`, `LABEL_SMOKE`, as in the design's "cloud/spend_file.py" section. `rough_cost` adds the IPv4 and root-volume-storage terms to the machine price over the given minutes. (The machine-side append and the laptop merge come in tasks 6 and 8.)
    - _Requirements: 6.3, 12.3_

  - [x] 5.3 Write `cloud/start.py`
    - `start_job()` and `main()`, in the design's order, free and read-only work first: refuse an empty `STEP_LIST` with `splatpipe.run`'s own "No steps are set up yet…" text; refuse uncommitted `splatpipe/`, `cloud/` or `config.toml`; `archive_head` and `lock_hashes`; build the Runner_Profile `Aws` and `resolve_stack`; find the newest `available` tagged Machine_Image, else a `CloudError` pointing to the Image_Build_Command; refuse a lock-hash drift against the image's tags; refuse a missing Smoke_Marker pointing to the Smoke_Command; `make_job`; select, read the quota and running vCPUs, `admit` and `launch_plan`; build the quote (the first admitted type over the Expected_Time and the worst case, the dearest fitting type on the Fallback_List run to its Time_Cap) and call `check_spend`; only then upload the video, the Code_Archive and the Job_Record to `jobs/<id>/input/`; `render_user_data` and `launch_with_fallback`; report the type and zone. `main` parses `video`, `--start`, `--end` and `--preset` inside `run_cloud_cli` and builds the `Aws` from `[aws].runner_profile`. The Smoke_Marker gate stays a hard refusal as the design states.
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 7.9, 3.6, 3.7, 5.3, 1.5_

  - [x] 5.4 Write unit tests in `tests/test_cloud_repo.py`
    - A dirty `splatpipe/`, `cloud/` or `config.toml` is listed by `uncommitted`; a clean repo gives none; `archive_head` writes a tar and `head_commit` the sha; `lock_hashes` changes when a lockfile changes. Uses `git_repo` and skips with a reason when `git` is absent.
    - _Requirements: 7.3, 7.4_

  - [x] 5.5 Write unit tests in `tests/test_cloud_start.py`
    - An empty `STEP_LIST` raises the "No steps are set up yet" text with no upload or launch; each of the uncommitted, lock-drift, no-marker, no-image, quota-full and unmet-need refusals happens before any paid action; on the happy path the video, Code_Archive and Job_Record are uploaded to `jobs/<id>/input/` and one machine launches, reporting its type and zone; the quote's rough and worst-case numbers come from the Config prices. Uses a committed `git_repo`, a `FakeAws` with an image, a marker and a quota, and a monkeypatched fake `STEP_LIST` that never changes the shipped empty default.
    - _Requirements: 6.3, 7.1, 7.2, 7.3, 7.5, 7.6, 7.7, 7.8, 7.9, 3.4, 4.5, 5.3_

  - [x] 5.6 Write the property test for Property 6 in `tests/test_cloud_start.py`, with Hypothesis
    - **Property 6: Nothing paid happens before the Spend_Guard returns**
    - **Validates: Requirements 6.1, 6.6**
    - For any Preset, chosen type and inputs, when the Start_Command runs against the `FakeAws`, no `s3_upload` and no `run_instances` call is recorded before `check_spend` has returned without raising; and when the Spend_Guard raises, the `FakeAws` records no `s3_upload` and no `run_instances` at all.
    - _Requirements: 6.1, 6.6, 16.4_

  - [x] 5.7 Run the tests and commit
    - Run `uv run pytest`; every test passes.
    - Stage `cloud/repo.py`, `cloud/spend_file.py`, `cloud/start.py`, `tests/fake_aws.py`, `tests/test_cloud_repo.py`, `tests/test_cloud_start.py` by name; commit with the message "Cloud jobs: the Start_Command".
    - _Requirements: 16.1, 16.2, 16.8_

- [x] 6. The Cloud_Runner
  - Files: `cloud/runner.py` and `tests/test_cloud_runner.py` (new); `cloud/spend_file.py` (changed, the machine-side writing).
  - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7, 10.8, 10.9, 10.10, 10.11, 12.1, 12.2_

  - [x] 6.1 Write the machine-side writing in `cloud/spend_file.py`
    - `append_spend_line(aws, job_id, line)`, as in the design's "cloud/spend_file.py" section: read `jobs/<job_id>/out/spend.jsonl` from the bucket, append the line with its `id = "<job_id>:<label>:<n>"`, and put it back, so re-merging is idempotent and a genuine rerun writes a higher `n`.
    - _Requirements: 12.1, 12.2_

  - [x] 6.2 Write `cloud/runner.py`
    - `CAP_DEADLINE_ENV`, `rebuild_job_folder()`, `Uploader` and `run_cloud_job()`, as in the design's "cloud/runner.py" section. `run_cloud_job` compares the archived lock hashes with the image's (a drift writes a redacted note, uploads it, raises a `CloudError`, and still powers off in `finally`); rebuilds the Job_Folder; sets `CAP_DEADLINE_ENV` to the machine's start plus the Time_Cap minus `upload_margin_minutes`; starts the best-effort `Uploader`, which tails `out/job.log` and uploads a Step's report only after its outcome line appears; calls `run_job(job_id, steps=..., config=...)` with no Fresh_Option; writes the per-Step and `LABEL_MACHINE` Spend_Lines; applies `account_redaction` to the text logs (`out/job.log`, its own log, `cloud-init-output.log`) before upload while leaving the JSON reports unchanged; runs the final authoritative `sync_all`; and in `finally` calls `shutdown()` (default `poweroff`). It is testable with injected `steps`, `now`, `shutdown` and the `FakeAws`.
    - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.6, 10.7, 10.8, 10.9, 10.10, 10.11, 12.1, 12.2_

  - [x] 6.3 Write unit tests in `tests/test_cloud_runner.py`
    - `rebuild_job_folder` downloads `input/` and any earlier `out/` and `work/`; a lock-hash drift stops with a readable error and still powers off; `CAP_DEADLINE_ENV` is set to start plus cap minus margin; the uploaded `job.log` and `cloud-init-output.log` are redacted while the JSON reports are byte-for-byte; `shutdown` is called exactly once, after the final sync; per-Step and `machine` Spend_Lines are written.
    - _Requirements: 10.1, 10.6, 10.7, 10.9, 10.10, 12.1, 12.2_

  - [x] 6.4 Write the property test for Property 7 in `tests/test_cloud_runner.py`, with Hypothesis
    - **Property 7: The Cloud_Runner resumes exactly where run_job does**
    - **Validates: Requirements 10.4**
    - For any set of earlier Step_Reports in the Job_Bucket over a Step_List of required and optional Fake_Steps, the Cloud_Runner runs exactly the Steps `run_job` runs: it resumes from the first Step that is not a Finished_Step and never reruns a Step that finished before that resume point, while a Finished_Step after the first unfinished one may run again. The oracle is pipeline-core's own `read_earlier`/`first_to_run` applied to the reports the Cloud_Runner downloaded, so there is no contradiction with pipeline-core's resume.
    - _Requirements: 10.4, 16.3_

  - [x] 6.5 Write the property test for Property 8 in `tests/test_cloud_runner.py`, with Hypothesis
    - **Property 8: The uploaded set equals the Job's own files**
    - **Validates: Requirements 10.2, 10.3, 10.5**
    - For any Fake_Step Job the Cloud_Runner runs, once it finishes the set of keys the Uploader has uploaded equals the Job's `out/` files, reports and logs on the machine (with `work/` mapped to `work/<id>/` and the rest to `jobs/<id>/`), and every uploaded report key holds text that parses as a complete JSON object.
    - _Requirements: 10.2, 10.3, 10.5_

  - [x] 6.6 Run the tests and commit
    - Run `uv run pytest`; every test passes.
    - Stage `cloud/runner.py`, `cloud/spend_file.py`, `tests/test_cloud_runner.py` by name; commit with the message "Cloud jobs: the Cloud_Runner".
    - _Requirements: 16.1, 16.2, 16.8_

- [x] 7. The Image_Build and the Smoke_Test
  - Files: `cloud/image.py`, `cloud/smoke.py`, `tests/test_cloud_image.py`, `tests/test_cloud_smoke.py` (new); `docs/third-party-licences.md` (changed, the Machine_Image tool rows).
  - _Requirements: 1.5, 6.1, 8.9, 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7, 9.8, 9.9, 9.10, 11.1, 11.2, 11.3, 11.4, 11.5, 12.2, 14.5, 14.6_

  - [x] 7.1 Write `cloud/image.py`
    - `install_script()`, `build_image()` and `main()`, as in the design's "cloud/image.py" section. `install_script` holds, every version pinned: `uv sync` from `uv.lock`; ffmpeg built with zscale; COLMAP 4.2.1 from source with `-DCUDA_ENABLED=OFF -DOPENGL_ENABLED=OFF -DGUI_ENABLED=OFF -DCGAL_ENABLED=OFF -DLSD_ENABLED=OFF`; gsplat 1.5.3 and fused-ssim compiled with `TORCH_CUDA_ARCH_LIST="8.6;8.9"`; the MoGe-2 weights; Node and `npm ci`; a `cloud-init clean` and the removal of `~/.aws` before imaging; it never bakes in the Pipeline's code. `build_image` passes `check_spend` first, reads the Base_AMI from SSM, `admit`s the `image_builder_type`, launches the builder through `launch_with_fallback` with a root-volume `BlockDeviceMappings` entry, stops it before `create_image` (an API stop never triggers terminate-on-shutdown), tags the image and its snapshots with the Project_Tag and the two lock hashes in the same call, waits until `available`, terminates the builder and appends a `LABEL_IMAGE` Spend_Line. The Runner_Profile is used throughout.
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7, 9.8, 9.9, 9.10, 6.1, 1.5, 12.2_

  - [x] 7.2 Write `cloud/smoke.py`
    - `smoke_marker_key()`, `image_id_from_marker()`, `run_smoke()` and `main()`, as in the design's "cloud/smoke.py" section. `run_smoke` passes `check_spend` first, finds the current Machine_Image, `admit`s the `smoke_instance_type`, launches one from that image with the Requirement 8 request and the Smoke_Test's Time_Cap, whose User_Data runs every GPU code path on a tiny synthetic scene and runs the Cloud_Runner end to end on a tiny Fake_Step Job; on pass the machine writes the Smoke_Marker keyed to the image id, measures boot-to-ready, and ends itself; `run_smoke` appends a `LABEL_SMOKE` Spend_Line. The GPU work runs only on the real machine. Uses the Runner_Profile.
    - _Requirements: 11.1, 11.2, 11.3, 11.4, 11.5, 6.1, 1.5, 12.2_

  - [x] 7.3 Add the Machine_Image tool rows to `docs/third-party-licences.md`
    - A row for the Base_AMI, gsplat 1.5.3, fused-ssim, the MoGe-2 weights, Node, ffmpeg built with zscale and COLMAP 4.2.1, each with its licence and how the Machine_Image uses it; the COLMAP row notes the build flags that keep SiftGPU, CGAL and LSD out. These rows land in this task's commit alongside the `install_script` that names the tools. Verify zimg's licence for the ffmpeg-with-zscale row before committing, and ask the user before committing if zimg or any tool is not permissive.
    - _Requirements: 14.5, 14.6_

  - [x] 7.4 Write unit and structural tests in `tests/test_cloud_image.py`
    - Structural: `install_script` holds the COLMAP OFF flags, `TORCH_CUDA_ARCH_LIST="8.6;8.9"`, `npm ci`, gsplat 1.5.3, fused-ssim and the MoGe-2 weights, and a `cloud-init clean` before imaging; the Licence_List has the Machine_Image tool rows, the COLMAP row naming its OFF flags.
    - With the `FakeAws`: `build_image` calls `check_spend` first, reads the Base_AMI from SSM, stops the builder before `create_image`, tags the image and snapshots with the Project_Tag and the two lock hashes, waits for `available`, terminates the builder and appends an `image-build` Spend_Line; its launch request satisfies Property 4's shape.
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.7, 9.8, 9.9, 9.10, 8.9, 14.5_

  - [x] 7.5 Write unit tests in `tests/test_cloud_smoke.py`
    - `smoke_marker_key` and `image_id_from_marker` round-trip for any image id; with the `FakeAws`, `run_smoke` calls `check_spend` first, launches one `smoke_instance_type` from the current image with the Requirement 8 request and the smoke `cap_minutes`, and appends a `smoke-test` Spend_Line; its launch request satisfies Property 4's shape. The GPU paths are noted as real-machine-only and not run on the CPU.
    - _Requirements: 11.1, 11.3, 11.4, 6.1, 8.9_

  - [x] 7.6 Run the tests and commit
    - Run `uv run pytest`; every test passes.
    - Stage `cloud/image.py`, `cloud/smoke.py`, `docs/third-party-licences.md`, `tests/test_cloud_image.py`, `tests/test_cloud_smoke.py` by name; commit with the message "Cloud jobs: the Image_Build and the Smoke_Test".
    - _Requirements: 16.1, 16.2, 16.8_

- [x] 8. Status, download, list, stop-all and the Spend_File merge
  - Files: `cloud/status.py`, `cloud/download.py`, `cloud/list.py`, `cloud/stopall.py`, `tests/test_cloud_commands.py`, `tests/test_cloud_spend.py` (new); `cloud/spend_file.py` (changed, the merge).
  - _Requirements: 1.4, 1.5, 6.3, 6.5, 12.3, 12.4, 12.5, 13.1, 13.2, 13.3, 13.4, 13.5, 13.6, 13.7_

  - [x] 8.1 Write the Spend_File merge in `cloud/spend_file.py`
    - `merge_spend_file(local_path, new_lines)`, as in the design's "cloud/spend_file.py" section: add the new lines to `data/spend.jsonl`, skipping any `id` already there, return how many were added, be idempotent, and write only under `data/`.
    - _Requirements: 12.4, 12.5_

  - [x] 8.2 Write `cloud/status.py` and `cloud/download.py`
    - `status()`/`main()`: read `jobs/<id>/out/job.json` and the reports from the bucket with the Read_Profile, report the Job's progress, and `merge_spend_file` the Job's `out/spend.jsonl` into `data/spend.jsonl`, redacting any AWS error before printing.
    - `download()`/`main()`: fetch `jobs/<id>/out/*` and the reports into `data/jobs/<id>/out/` with the Read_Profile, and merge the Spend_Lines.
    - Each `main` runs inside `run_cloud_cli` and builds its `Aws` from `[aws].read_profile`.
    - _Requirements: 13.1, 13.2, 13.6, 12.4, 1.4_

  - [x] 8.3 Write `cloud/list.py` and `cloud/stopall.py`
    - `list_machines()`/`main()`: list our non-terminated instances by the Project_Tag with the Read_Profile, printing each type, state and launch time but no account id or ARN. The module file `cloud/list.py` is run as `python -m cloud.list` and keeps the builtin `list` available inside it.
    - `stop_all()`/`main()`: terminate our non-terminated instances by the Project_Tag with the Runner_Profile, as the last-resort end, calling no Spend_Guard because terminating is free.
    - _Requirements: 13.3, 13.4, 13.5, 6.5, 1.4, 1.5_

  - [x] 8.4 Write unit tests in `tests/test_cloud_commands.py`
    - With the `FakeAws` and a seeded bucket: `status` reads the summary and reports and merges Spend_Lines without duplicates; `download` fetches `out/` into `data/jobs/<id>/out/` and merges Spend_Lines; `list_machines` returns our tagged non-terminated instances and `main` prints type, state and launch time with no account id; `stop_all` terminates them with no `check_spend` call; each command built its `Aws` with the right profile key; AWS errors are redacted before printing.
    - _Requirements: 13.1, 13.2, 13.3, 13.4, 13.5, 13.6, 12.4, 1.4, 1.5, 6.5_

  - [x] 8.5 Write unit tests in `tests/test_cloud_spend.py`
    - `spend_line` carries the id, label, type, seconds, `estimated_usd == seconds/3600*price` and `estimate == true`; `rough_cost` adds the IPv4 and root-storage terms; `merge_spend_file` writes only under `data/`.
    - _Requirements: 12.3, 12.5, 6.3_

  - [x] 8.6 Write the property test for Property 9 in `tests/test_cloud_spend.py`, with Hypothesis
    - **Property 9: Merging Spend_Lines is idempotent and deduplicated**
    - **Validates: Requirements 12.4**
    - For any starting Spend_File and any list of Spend_Lines, `merge_spend_file` leaves the file holding exactly the union by `id` of the lines already there and the new ones, so merging the same lines again adds nothing and the file never holds two lines with the same `id`.
    - _Requirements: 12.4_

  - [x] 8.7 Run the tests and commit
    - Run `uv run pytest`; every test passes.
    - Stage `cloud/spend_file.py`, `cloud/status.py`, `cloud/download.py`, `cloud/list.py`, `cloud/stopall.py`, `tests/test_cloud_commands.py`, `tests/test_cloud_spend.py` by name; commit with the message "Cloud jobs: status, download, list, stop-all and the Spend_File merge".
    - _Requirements: 16.1, 16.2, 16.8_

- [x] 9. Docs, steering, the skill and the import checks
  - Files: `tests/test_cloud_imports.py` (new); `docs/pipeline.md`, `README.md`, `.kiro/steering/tech.md`, `.kiro/skills/aws-jobs/SKILL.md`, `tests/test_cloud_docs.py` (changed).
  - _Requirements: 1.1, 1.2, 15.1, 15.2, 15.3, 15.4, 15.5, 15.6, 15.7, 16.6, 16.7_

  - [x] 9.1 Update `docs/pipeline.md`, `README.md`, `.kiro/steering/tech.md` and the aws-jobs skill
    - `docs/pipeline.md`: a cloud-run section describing the Start_Command, the Cloud_Runner, the Status, Download, List and Stop_All commands; the per-Step Step_Needs plan of Requirement 2's note (picking frames and solving cameras 16 vCPUs/16 GB, depth priors a GPU with 16 GB VRAM, plain walls 32 GB RAM and as many vCPUs as it can get, training a GPU and 32 GB RAM, clean and export 8 vCPUs/8 GB), noting each Step carries these as `min_vcpus`, `min_memory_gb` and `min_gpu_memory_gb` in its own section for each Preset with a missing key meaning no need; the `SPLATPIPE_CAP_DEADLINE` variable's name and meaning; and that the cloud run is not a Step and adds no Step_List entry.
    - `README.md`: a "Run your own copy" section listing what someone needs (an AWS account with GPU quota, the Stack from the aws-setup spec, and building the Machine_Image once, about $1.20 and about 45 minutes) and the commands in order, noting those figures are estimates to check against the Config prices.
    - `.kiro/steering/tech.md`: change the "one job machine runs at a time" line to the quota-based admission of Q3.
    - `.kiro/skills/aws-jobs/SKILL.md`: change the "a fallback type is a new run with a new yes" line to the one-yes-covers-the-Fallback_List rule, change the "only one job machine runs at a time" note to the quota-based admission, and add that Cloud_Jobs applies the Account_Redaction so Kiro may read the redacted console output to debug a machine.
    - _Requirements: 15.1, 15.2, 15.3, 15.4, 15.5, 15.6, 15.7_

  - [x] 9.2 Write the import checks in `tests/test_cloud_imports.py`
    - Read the files with `ast` and run nothing. No file under `splatpipe/` imports `cloud`, boto3, torch, gsplat or another GPU package; boto3 appears only under `cloud/`; no `cloud/*.py` imports a GPU package at module level, so every `cloud` module imports on a laptop. A failure gives the file and the line.
    - _Requirements: 1.1, 1.2, 16.7_

  - [x] 9.3 Add the doc, steering and skill tests to `tests/test_cloud_docs.py`
    - `docs/pipeline.md` has the cloud-run section, the `SPLATPIPE_CAP_DEADLINE` variable, the per-Step Step_Needs plan, and the statement that the cloud run is not a Step. `README.md` has the "Run your own copy" section with the ~$1.20/~45-minute note. `.kiro/steering/tech.md`'s one-at-a-time line is now quota-based. The aws-jobs skill has the one-yes-covers-the-Fallback_List line, the quota-based admission line and the console redaction note.
    - _Requirements: 15.1, 15.2, 15.3, 15.4, 15.5, 15.6, 15.7_

  - [x] 9.4 Run the tests and commit
    - Run `uv run pytest`; every test passes. Confirm `splatpipe.run.STEP_LIST` is still `()` and no pipeline-core or clean-export test changed.
    - Stage `docs/pipeline.md`, `README.md`, `.kiro/steering/tech.md`, `.kiro/skills/aws-jobs/SKILL.md`, `tests/test_cloud_imports.py`, `tests/test_cloud_docs.py` by name; commit with the message "Cloud jobs: docs, steering, the skill and the import checks".
    - _Requirements: 16.1, 16.2, 16.6, 16.8_

## Notes

- Every test sub-task is required, as `.kiro/steering/rules.md` asks. None is marked optional, so no `*` postfix appears.
- The last sub-task of each task is its checkpoint: the whole suite, then the commit by name. So the plan has no separate checkpoint or test-only tasks and stays at the design's 9.
- The three mandated property tests sit in the tasks that own them: Property 4 (the launch-request shape, Req 8.9) in task 4, Property 6 (nothing paid before the Spend_Guard, Req 6.6) in task 5, and Property 7 (resume from the first unfinished Step, Req 10.4) in task 6. Properties 1, 2, 3, 5, 8 and 9 sit in tasks 2, 1, 3, 4, 6 and 8.
- Tasks 7 and 9 carry unit and structural tests only. The Image_Build is IaC-like, the Smoke_Test's GPU paths run only on a real machine, and the docs and import checks are structural, so none has a universal property, per the design's Correctness Properties section and the workflow's "when NOT to use PBT".
- Every acceptance criterion in `requirements.md` is named by at least one task; Req 1.10 is honoured by leaving `cloud/aws-setup.yaml` untouched, as the design's "Stack coverage" shows every AWS action maps to an existing policy.
- No task adds a `STEP_LIST` entry, changes any `splatpipe/` module, changes a pipeline-core or clean-export test, or changes `cloud/aws-setup.yaml`. No task calls AWS, sets `APPROVED_SPEND` or runs anything paid; the paid commands are exercised only against the `FakeAws`.
- The graph runs the tasks in order, one at a time, so each task's test run and commit see only finished work. Inside a task, sub-tasks that write different files share a wave, tests come after the code they test, and each task's commit gates the next task's first sub-task.

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "1.2", "1.3"] },
    { "id": 1, "tasks": ["1.4", "1.6"] },
    { "id": 2, "tasks": ["1.5"] },
    { "id": 3, "tasks": ["1.7"] },
    { "id": 4, "tasks": ["2.1"] },
    { "id": 5, "tasks": ["2.2"] },
    { "id": 6, "tasks": ["2.3"] },
    { "id": 7, "tasks": ["2.4"] },
    { "id": 8, "tasks": ["2.5", "2.6"] },
    { "id": 9, "tasks": ["2.7"] },
    { "id": 10, "tasks": ["3.1"] },
    { "id": 11, "tasks": ["3.2"] },
    { "id": 12, "tasks": ["3.3"] },
    { "id": 13, "tasks": ["3.4"] },
    { "id": 14, "tasks": ["4.1", "4.2"] },
    { "id": 15, "tasks": ["4.3", "4.5"] },
    { "id": 16, "tasks": ["4.4"] },
    { "id": 17, "tasks": ["4.6"] },
    { "id": 18, "tasks": ["5.1", "5.2"] },
    { "id": 19, "tasks": ["5.3", "5.4"] },
    { "id": 20, "tasks": ["5.5"] },
    { "id": 21, "tasks": ["5.6"] },
    { "id": 22, "tasks": ["5.7"] },
    { "id": 23, "tasks": ["6.1"] },
    { "id": 24, "tasks": ["6.2"] },
    { "id": 25, "tasks": ["6.3"] },
    { "id": 26, "tasks": ["6.4"] },
    { "id": 27, "tasks": ["6.5"] },
    { "id": 28, "tasks": ["6.6"] },
    { "id": 29, "tasks": ["7.1", "7.2", "7.3"] },
    { "id": 30, "tasks": ["7.4", "7.5"] },
    { "id": 31, "tasks": ["7.6"] },
    { "id": 32, "tasks": ["8.1", "8.3"] },
    { "id": 33, "tasks": ["8.2", "8.5"] },
    { "id": 34, "tasks": ["8.4", "8.6"] },
    { "id": 35, "tasks": ["8.7"] },
    { "id": 36, "tasks": ["9.1"] },
    { "id": 37, "tasks": ["9.2", "9.3"] },
    { "id": 38, "tasks": ["9.4"] }
  ]
}
```
