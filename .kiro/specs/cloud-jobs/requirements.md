# Requirements Document

## Introduction

The Pipeline's heavy work runs on a GPU machine in AWS, one machine per Job. This spec builds the cloud wrapper, Cloud_Jobs, that runs a whole Job on one EC2 machine so the laptop can sleep. Cloud_Jobs uploads the video and the Pipeline's own code, starts one machine from our Machine_Image, lets the Cloud_Runner on that machine run the Step_List through pipeline-core's `run_job` and upload the results, and ends the machine when the Job ends. Around that run sit the laptop commands: a Start_Command that launches a Job, a Status_Command and a Download_Command that read and fetch its results, a List_Command that shows our machines and a Stop_All_Command that terminates them, an Image_Build_Command that makes the Machine_Image, and a Smoke_Command that proves a fresh image works before any long run.

Cloud_Jobs only wraps the Pipeline functions; it never changes what they do. It all lives under `cloud/`, so `splatpipe/` never learns where it runs. It finds the Job_Bucket, the Job_Profile, the Job_Security_Group, the Default_VPC and the Runner_Role through the Stack's outputs, read by the Stack's name in `[aws].stack_name`, never through names or ids kept in the repo. The Status_Command, the Download_Command and the List_Command use the Read_Profile; only the Start_Command, the Image_Build_Command, the Smoke_Command and the Stop_All_Command use the Runner_Profile.

Every launch costs money, so every launch goes through the Spend_Guard first, and Kiro quotes the machine, the expected time and the rough cost in the chat and runs the paid command with `APPROVED_SPEND=1` only after the person says yes. One machine per Job, no automatic retries or relaunches, and every machine ends itself when its Job ends, with a hard Time_Cap as a backstop.

### In scope

- The cloud wrapper in `cloud/`: resolving the Stack_Outputs by `[aws].stack_name`; choosing the machine type with the Fallback_List and the Step_Needs filter; reading the real G_Quota and admitting a launch; the Spend_Guarded Start_Command (the uncommitted-change check, the Code_Archive of `HEAD`, the uv.lock and package-lock pre-check, the quota-room check, the upload of the video, the code and the Job_Record, the launch request, and the machine/time/cost quote); the machine-side User_Data and Cloud_Runner (resume through `run_job`, per-Step upload, self-shutdown, the Time_Cap, the Cap_Deadline, the Spend_Lines and the log redaction); the Image_Build; the Smoke_Test; the Status_Command, Download_Command, List_Command and Stop_All_Command; the `[cloud]` Config section; boto3 and its Licence_List rows; the README's "Run your own copy" section; the Pipeline_Doc updates for the cloud run, the Step_Needs plan and the Cap_Deadline; and the Tech_Steering and AWS_Jobs_Skill updates for the one-yes, the quota-based admission and the redaction.

### Out of scope

- Changing any `splatpipe/` module or any pipeline-core or clean-export test.
- Adding a Step_List entry. `STEP_LIST` stays empty, so no real Job can run until a Step spec adds an entry, and this spec changes no pipeline-core test (Requirement 16).
- Any GPU package in `pyproject.toml` or the laptop install. The GPU packages live only in the Machine_Image.
- Deploying or changing the Stack. The Stack already exists from the aws-setup spec, and this spec needs no change to `cloud/aws-setup.yaml` (Requirement 1).
- Any real paid AWS run while carrying out this spec. The tests run on a CPU with a Fake_Step and a Fake_AWS, no account and no network.

### Task budget

The whole spec stays within 10 tasks, which the design and tasks phases enforce. Each task writes its own unit tests and property tests, both required, never optional.

### Settled in clarification

The clarify step put four money and machine questions to the user, who settled each one. Each is marked in the requirement it belongs to.

- **Q1 — One yes covers the Fallback_List (Requirement 3, Requirement 6):** the person's yes for a Start_Command covers every type on that Preset's Fallback_List, because only one of them ever launches. Before asking, Kiro quotes the first choice (type, price per hour, expected time, rough cost) and the worst case: the dearest type on the list running to its Time_Cap. The AWS_Jobs_Skill's line that a fallback type is "a new run with a new yes" changes to match (Requirement 15).
- **Q2 — Expected times and Time_Caps (Requirement 2):** the expected times are 30 min (fast), 60 min (default), 120 min (quality), 10 min (smoke test) and 45 min (image build). The Time_Caps are 120 min (fast), 180 min (default), 360 min (quality), 30 min (smoke test) and 120 min (image build). A Time_Cap matters only when something hangs, and a tight cap would cut a long video's Job short, so the caps are deliberately generous. The worst-case cost quote of Q1 is therefore large, which is acceptable because the cap only bills when something hangs.
- **Q3 — Machines at the same time (Requirement 4):** admission is by the real quota. The Start_Command adds up the vCPUs of our running machines (jobs, smoke tests and image builds, found by the Project_Tag) and launches when the chosen type still fits the real G_Quota read from Service Quotas, so under 32 vCPUs two 16-vCPU fallback machines could run side by side. The Tech_Steering's and the AWS_Jobs_Skill's "one job machine runs at a time" both change to match (Requirement 15). The rule "one machine per Job at most" still holds: it means one machine per Job, not one machine in the whole account.
- **Q4 — The Spend_File (Requirement 12):** the machine writes the Spend_Lines into the Job's `out/` in the Job_Bucket, and the Status_Command and the Download_Command add the new ones to the laptop's `data/spend.jsonl` without duplicates. Besides a line per Step, there is a line for the machine's time outside the Steps (boot, setup, uploads) and one for each Image_Build and Smoke_Test, so the Spend_File adds up to roughly what the machines cost. Every line is labelled an estimate.

### Assumptions confirmed at review

The user didn't ask for these in the first request. Each is the recommended answer, marked in the requirement it belongs to, for the review to confirm, and each is a small change if the review decides otherwise.

- **A — Per-Step upload without changing pipeline-core (Requirement 10):** the Cloud_Runner rebuilds the Job_Folder from the Job_Bucket (the `input/`, plus any `out/` and `work/` from an earlier run of the same Job), then calls pipeline-core's public `run_job(job_id)` without the Fresh_Option, which already resumes from the first unfinished Step and is the single source of truth for the Job_Summary, the Job_Log and each Step_Report. A best-effort Uploader uploads each Step_Report, the Step's `work/` and `out/` files, the Job_Log and the Job_Summary as they appear, and a final authoritative sync runs after `run_job` returns. Because `write_json` in `splatpipe/jsonio.py` is a plain, non-atomic `path.write_text`, the Uploader never uploads a Step_Report that is still being written: it uploads a Step_Report only after the Step that owns it has returned.
- **B — The resume property versus pipeline-core Requirement 8.13 (Requirement 10):** the property is stated exactly as pipeline-core defines resume. The Cloud_Runner resumes from the first Step that is not a Finished_Step and never reruns a Step that finished before that resume point; a Finished_Step that sits after an unfinished one may still run again, because pipeline-core reruns it (its inputs may have changed). The property test matches pipeline-core so there is no contradiction.
- **C — Telling a Step it is close to the Time_Cap (Requirement 10):** the Cloud_Runner sets an environment variable, the Cap_Deadline, holding the absolute UTC instant by which a Step should have saved and stopped (the machine's start plus the Time_Cap, minus `upload_margin_minutes` from Config). `splatpipe/` stays unchanged in this spec: no current Step reads the Cap_Deadline, and the training spec will read it. The variable's name and meaning go in the Pipeline_Doc (Requirement 15).
- **D — The quality Fallback_List (Requirement 2, Requirement 3):** the quality Preset's primary type is g6e.4xlarge and its fallback is g6e.2xlarge, keeping the 48 GB L40S class. The Fallback_Lists live in Config so they are tunable, and the user may add the 24 GB types later if cheaper quality fallbacks are wanted.
- **E — Where a Step's needs live (Requirement 2, Requirement 3):** each Step keeps its needs in its own Config section, not in a `[cloud.needs.*]` table (there is none). For each Step in the Step_List, the Start_Command reads `min_vcpus`, `min_memory_gb` and `min_gpu_memory_gb` from that Step's settings for the Job's Preset (its Shared_Settings and its Preset table, through pipeline-core's `step_settings`); a missing key means no need in that dimension. Each future Step spec adds these keys to its own section (for example `[frames]` on the frame-picking branch, with `min_vcpus = 16` and `min_memory_gb = 16`), and the tests put them in a Fake_Step's section. This spec records the user's stated per-Step numbers in the Pipeline_Doc as the plan.
- **F — The Smoke_Test gate (Requirement 7, Requirement 11):** the Start_Command refuses to launch a Job unless a passing Smoke_Marker exists in the Job_Bucket for the current Machine_Image id, with a Readable_Error pointing to the Smoke_Command. The Smoke_Test runs the real Cloud_Runner end to end on a tiny Fake_Step Job as well as the GPU code paths. The review may downgrade the gate to a team rule.
- **G — The splat-transform lockfile overlap with clean-export (Requirement 9, Requirement 14):** the repo root has no `package.json` or `package-lock.json` yet. This spec creates them, pinning splat-transform 3.8.0; the Machine_Image installs them with `npm ci`; and the machine stops with a Readable_Error if the Job's `package-lock.json` differs from the image's, the same rule as uv.lock. The clean-export spec reuses these files rather than recreating them, and the Licence_List gains the rows for splat-transform's npm tree from the lockfile.
- **H — Redaction in uploaded log files (Requirement 10, Requirement 13):** the machine replaces every 12-digit number with `<account>` in the text logs it uploads (the Job_Log, the Cloud_Runner's own log and `cloud-init-output.log`), so a log downloaded into `data/` is safe for Kiro to open. JSON reports are left as they are, because they hold no account id. This is in addition to the rule that the laptop commands redact AWS errors, console output and job logs before printing.
- **I — The laptop uv.lock pre-check (Requirement 7, Requirement 9):** when the Machine_Image is built, its `uv.lock` hash and its `package-lock.json` hash are recorded as tags on the image. The Start_Command compares `HEAD`'s `uv.lock` and `package-lock.json` with the image's tags and refuses before launching when they differ, to save a paid boot; the machine still re-checks as a backstop.

## Glossary

Terms this spec reuses from the pipeline-core spec and the Pipeline_Doc: Pipeline, Skeleton, Step, Step_Name, Step_List, Required_Step, Optional_Step, Finished_Step, Fresh_Option, Job, Job_Id, Job_Folder, Job_Record, Jobs_Folder, Step_Report, Step_Outcome, Job_Runner (`run_job`), Job_Summary, Job_Log, Job_Status, Preset, Config, Config_Loader, Pipeline_Section, Step_Section, Shared_Setting, Preset_Setting, Expected_Failure, Readable_Error, Fake_Step, Spend_Guard, Spend_Variable, Paid_Entry_Point, Pipeline_Doc, Licence_List, Structure_Steering and Test_Suite.

Terms this spec reuses from the aws-setup spec: Stack, Stack_Region, Stack_Output, Account_Owner, Job_Bucket, Bucket_Job_Folder (`jobs/<Job_Id>/`), Bucket_Work_Folder (`work/<Job_Id>/`), Work_File, Job_Role, Job_Profile, Job_Security_Group, Default_VPC, Runner_Role, Read_Policy, Read_Profile, Runner_Profile, Project_Tag, Project_Group, Amazon_Image, Machine_Image, Base_AMI_Parameter, Allowed_Type, G_Quota, IMDSv2, Debug_Shell, Session_Manager_Plugin and Setup_Guide.

New terms this spec defines:

- **Cloud_Jobs**: the cloud wrapper this spec builds, under `cloud/`. It wraps the Pipeline functions and never changes what they do.
- **Base_AMI**: AWS's Deep Learning Base OSS Nvidia Driver GPU AMI for Ubuntu 24.04, whose current id is in the public Base_AMI_Parameter `/aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-24.04/latest/ami-id`.
- **Start_Command**: the laptop command and function that make a Job from a video and launch one machine to run it. A Paid_Entry_Point that uses the Runner_Profile.
- **Status_Command**: the laptop command that reports a Job's progress from the Step_Reports and the Job_Summary in the Job_Bucket. It uses the Read_Profile and is free.
- **Download_Command**: the laptop command that fetches a Job's `out/` and reports from the Job_Bucket into `data/`. It uses the Read_Profile and is free.
- **List_Command**: the laptop command that lists every instance carrying the Project_Tag that isn't terminated. It uses the Read_Profile and is free.
- **Stop_All_Command**: the laptop command that terminates every instance carrying the Project_Tag that isn't terminated. It uses the Runner_Profile and is free (terminating costs nothing), so it calls no Spend_Guard.
- **Image_Build**: making the Machine_Image from the Base_AMI with a builder machine. The **Image_Build_Command** drives it, a Paid_Entry_Point that uses the Runner_Profile.
- **Smoke_Test**: a short run on a g5.xlarge that exercises every GPU code path on a tiny synthetic scene and runs the Cloud_Runner end to end on a tiny Fake_Step Job. The **Smoke_Command** drives it, a Paid_Entry_Point that uses the Runner_Profile.
- **Smoke_Marker**: a small object the Smoke_Test writes under `jobs/` in the Job_Bucket when it passes, keyed to the Machine_Image id it ran on, which the Start_Command checks before launching a Job.
- **Cloud_Runner**: the machine-side runner that rebuilds the Job_Folder from the Job_Bucket, calls `run_job`, uploads the results and ends the machine.
- **User_Data**: the launch script (also called the Bootstrap) that cloud-init runs once as root on the machine's first boot, which sets the Cap_Deadline, schedules the Time_Cap shutdown, downloads the inputs, starts the Cloud_Runner and powers the machine off.
- **Code_Archive**: a `git archive` of `HEAD` that the Start_Command uploads to the Job's `input/`, so the machine runs the Pipeline's code at a committed revision.
- **Time_Cap**: the hardest limit on a machine's life, in minutes, from Config per Preset and for the Smoke_Test and the Image_Build. When the Time_Cap is reached, the machine powers off whatever else is happening.
- **Expected_Time**: the time a run is expected to take, in minutes, from Config per Preset and for the Smoke_Test and the Image_Build, used in the money quote and the rough cost.
- **Cap_Deadline**: the absolute UTC instant by which a Step should have saved and stopped: the machine's start plus the Time_Cap, minus `upload_margin_minutes`. The Cloud_Runner puts it in an environment variable for the Steps.
- **Step_Needs**: one Step's resource needs, read from the Step's own Config section (its Shared_Settings and its Preset table) for the Job's Preset: `min_vcpus`, `min_memory_gb` and `min_gpu_memory_gb`. A missing key means no need in that dimension, and there is no `[cloud.needs.*]` table.
- **Machine_Spec**: one Allowed_Type's resources, from the `[cloud.types.<type>]` table in Config: `vcpus`, `memory_gb`, `gpu_memory_gb`, `price_per_hour_usd` and a `price_date`.
- **Fallback_List**: the ordered list of Allowed_Types for one Preset, from Config, the primary type first. The Start_Command launches the first type on the list that fits the Step_Needs and the G_Quota and has capacity.
- **Running_Vcpus**: the sum of the vCPUs of our machines (by the Project_Tag) that aren't terminated, which the Start_Command adds to the chosen type's vCPUs to check the G_Quota.
- **Spend_Line**: one line of the Spend_File: a JSON object with the Job_Id, what the line is for (a Step, the machine's time outside the Steps, an Image_Build or a Smoke_Test), the machine type, the seconds and the estimated cost in US dollars, labelled an estimate.
- **Spend_File**: `data/spend.jsonl` on the laptop, and the Job's copy under `out/` in the Job_Bucket, holding Spend_Lines.
- **Account_Redaction**: replacing every run of exactly 12 digits with `<account>` in text a command prints or a machine uploads, so an AWS account id never reaches a chat, a log in the repo or `data/`.
- **Cloud_Config**: the `[cloud]` Config section and the reader in `cloud/` that loads it.
- **Fake_AWS**: an in-process stand-in for the AWS clients (EC2, S3, CloudFormation and Service Quotas) that records calls and returns canned replies, so the tests run with no account and no network.
- **README**: `README.md` at the repo root.
- **Tech_Steering**: `.kiro/steering/tech.md`.
- **AWS_Jobs_Skill**: the `aws-jobs` skill at `.kiro/skills/aws-jobs/SKILL.md`.
- **Project_File**: `pyproject.toml` at the repo root, with `uv.lock` next to it.

## Requirements

### Requirement 1: Where Cloud_Jobs lives and how it reaches AWS

**User Story:** As a team member, I want all the cloud code under `cloud/`, finding the Stack's resources by the Stack's name and using the right profile for each command, so that `splatpipe/` never learns where it runs and nothing account-specific lands in the repo.

#### Acceptance Criteria

1. THE Cloud_Jobs code SHALL live only under `cloud/`, and no file under `splatpipe/` SHALL import `cloud/`, boto3 or any other AWS code, which the Test_Suite checks across every file under `splatpipe/`.
2. THE Cloud_Jobs code SHALL import boto3 only inside `cloud/`, and MAY import `splatpipe` to call its public functions, because `splatpipe/` must never import `cloud/` but not the reverse.
3. THE Cloud_Jobs code SHALL find the Job_Bucket's name, the Job_Profile's ARN, the Job_Security_Group's id, the Default_VPC's id and the Runner_Role's ARN only from the Stack_Outputs, read from the Stack named by `[aws].stack_name`, and SHALL hold no bucket name, resource id, ARN or account id in the repo.
4. THE Status_Command, THE Download_Command and THE List_Command SHALL use the Read_Profile named by `[aws].read_profile`.
5. THE Start_Command, THE Image_Build_Command, THE Smoke_Command and THE Stop_All_Command SHALL use the Runner_Profile named by `[aws].runner_profile`.
6. WHEN Cloud_Jobs code on a laptop makes an AWS client, THE Cloud_Jobs code SHALL name both the profile and the region in `boto3.Session(profile_name=..., region_name=...)`, with the region from `[aws].region`, and SHALL never rely on `AWS_PROFILE`, `AWS_REGION`, `AWS_DEFAULT_REGION` or the default credential chain.
7. WHEN Cloud_Jobs code runs on a Job_Machine, THE Cloud_Jobs code SHALL make its AWS clients with `boto3.Session(region_name=...)` and no profile, so the machine uses its instance role.
8. WHEN Cloud_Jobs looks up an AWS resource, THE Cloud_Jobs code SHALL look up only our own resources, by the Project_Tag or by a name the Stack gives, and SHALL run no command that lists, names or describes any resource this project doesn't make.
9. THE Cloud_Jobs code SHALL keep every AWS account id, access key and ARN out of the repo, the chat and any log, including through the Account_Redaction of Requirement 13.
10. THIS spec SHALL need no change to `cloud/aws-setup.yaml`, because the Stack already exposes every Stack_Output and every permission Cloud_Jobs needs; IF the design finds a gap in the Stack, THEN THE design SHALL raise the gap for the Account_Owner to fix by updating the Stack, rather than work around it.

### Requirement 2: The `[cloud]` Config section

**User Story:** As a team member, I want every machine type, price, time and per-Step need in Config, so that we can tune the cloud run without changing code.

**Settled in clarification:** Q2 (criteria 5 and 6). **Assumptions confirmed at review:** D (criterion 4), E (criterion 7).

#### Acceptance Criteria

1. THE Config SHALL hold a `[cloud]` Pipeline_Section, and Cloud_Config SHALL read it, with the Config_Loader's existing checks refusing NaN, infinity and a TOML date or time in `[cloud]`.
2. THE Cloud_Config SHALL refuse a missing or wrong-typed `[cloud]` value with a Readable_Error that names the key and what the key must be, because `splatpipe/config.py` has no accessor for `[cloud]`.
3. THE `[cloud]` section SHALL hold a `[cloud.types.<type>]` Machine_Spec table for each of the 8 Allowed_Types, each with `vcpus`, `memory_gb`, `gpu_memory_gb`, `price_per_hour_usd` and a `price_date`, and the `price_per_hour_usd` SHALL be the On-Demand price for the Stack_Region on the `price_date`, taken from AWS's On-Demand pricing page (filled from the 2026-11-21 us-east-1 Linux list).
4. THE `[cloud]` section SHALL hold a `[cloud.presets.<preset>]` table for each Preset, each with an ordered Fallback_List of Allowed_Type names (fast and default primary g5.8xlarge, quality primary g6e.4xlarge, quality fallback g6e.2xlarge by assumption D), an `expected_minutes` and a `cap_minutes`.
5. THE `[cloud]` section SHALL hold an `expected_minutes` and a `cap_minutes` for the Smoke_Test and for the Image_Build (settled in Q2: expected 10 and 45, caps 30 and 120).
6. THE `[cloud.presets.<preset>]` tables SHALL hold the Expected_Times and Time_Caps settled in Q2: fast expected 30 and cap 120, default expected 60 and cap 180, quality expected 120 and cap 360, all in minutes.
7. THE Start_Command SHALL read each Step's Step_Needs from that Step's own Config section for the Job's Preset (its Shared_Settings and its Preset table) as `min_vcpus`, `min_memory_gb` and `min_gpu_memory_gb`, a missing key meaning no need in that dimension, and THE Config SHALL hold no `[cloud.needs.*]` section (assumption E).
8. THE `[cloud]` section SHALL hold an `upload_margin_minutes` for the Cap_Deadline, a poll interval in seconds for the Status_Command and the Cloud_Runner's Uploader, a `root_gb` root-volume size (150, used for every machine), and any other number Cloud_Jobs needs, and SHALL hold no hard-coded number anywhere in the Cloud_Jobs code.
9. THE every key in the `[cloud]` section and its tables SHALL have a short comment, which the Test_Suite checks, as the Config_Loader requires for a Pipeline_Section.
10. THE Test_Suite SHALL check that `[cloud]` loads from the real Config, that each Allowed_Type has a Machine_Spec, that each Preset has a Fallback_List whose entries are all Allowed_Types, and that `[cloud]` holds no `[cloud.needs.*]` table.

Note: the per-Step needs the Pipeline_Doc records as the plan (Requirement 15) are picking frames and solving cameras 16 vCPUs and 16 GB RAM, depth priors a GPU with 16 GB VRAM, training a GPU and 32 GB RAM, and clean and export 8 vCPUs and 8 GB RAM. Each Step expresses these in its own section as `min_vcpus`, `min_memory_gb` and `min_gpu_memory_gb` for each Preset, a missing key meaning no need; for example frame-picking's `[frames]` sets `min_vcpus = 16` and `min_memory_gb = 16`.

### Requirement 3: Choosing a machine type that fits every Step

**User Story:** As someone paying for the machine, I want the machine chosen to cover every Step's memory and GPU needs and to fall back to another type when the first is unavailable, so that a Job neither fails for want of memory nor waits forever for one type.

**Settled in clarification:** Q1 (criteria 6 and 7). **Assumptions confirmed at review:** D (criterion 5), E (criteria 1 and 2).

#### Acceptance Criteria

1. WHEN the Start_Command chooses a machine for a Job, THE Start_Command SHALL work out the Step_Needs of every Step the machine must run, and SHALL keep only the Allowed_Types whose Machine_Spec `memory_gb` is at least the largest `min_memory_gb` over those Steps and whose `gpu_memory_gb` is at least the largest `min_gpu_memory_gb` over those Steps.
2. THE Start_Command SHALL treat `min_vcpus` as a soft need, so a type with fewer vCPUs than a Step asks for is still allowed, because fewer vCPUs only make a Step slower, while `min_memory_gb` and `min_gpu_memory_gb` are hard needs.
3. THE Start_Command SHALL consider the types in the Job's Preset's Fallback_List, in order, and SHALL never choose a type that is not on that Fallback_List.
4. IF no type on the Fallback_List meets the hard Step_Needs, THEN THE Start_Command SHALL stop with a Readable_Error that names the unmet need, before it starts or uploads anything.
5. THE Start_Command SHALL take the Fallback_List and the Machine_Specs from `[cloud]` and each Step's Step_Needs from that Step's own section, so changing a type's size or a Step's needs changes no code (assumptions D and E).
6. BEFORE the Spend_Guard, THE Start_Command SHALL quote the first type on the Fallback_List that fits (its type, its `price_per_hour_usd`, the Preset's Expected_Time and a rough cost) and the worst case (the dearest fitting type on the Fallback_List run to its Time_Cap), because one yes covers the whole Fallback_List (Q1).
7. THE person's one yes SHALL cover every type on the Fallback_List, because only one type ever launches for the Job, so moving to a fallback type needs no new yes (Q1).

### Requirement 4: Fitting the launch into the real GPU quota

**User Story:** As someone running jobs on a shared account, I want the Start_Command to read the account's real GPU quota and start only when the machine fits, so that it tells me in plain words when there is no room instead of letting AWS refuse the launch.

**Settled in clarification:** Q3 (criteria 2, 3 and 4).

#### Acceptance Criteria

1. WHEN the Start_Command chooses a machine, THE Start_Command SHALL read the G_Quota (quota code `L-DB2E81BA`) for the Stack_Region from Service Quotas with the Read_Policy's permission, rather than assume a value.
2. THE Start_Command SHALL work out the Running_Vcpus by describing our machines (by the Project_Tag) that aren't terminated, counting jobs, Smoke_Tests and Image_Builds alike (Q3).
3. THE Start_Command SHALL launch a type only when the Running_Vcpus plus that type's `vcpus` are at most the G_Quota, so under a 32-vCPU quota one g5.8xlarge runs at a time while two 16-vCPU fallback machines could run side by side (Q3).
4. IF the G_Quota is 0, THEN THE Start_Command SHALL stop with a Readable_Error that says the account has no GPU quota and points to the Setup_Guide for asking AWS to raise it (Q3).
5. IF every fitting type on the Fallback_List would take the Running_Vcpus over the G_Quota, THEN THE Start_Command SHALL stop with a Readable_Error that gives the G_Quota, the Running_Vcpus and the type's vCPUs, before it uploads or launches anything, instead of letting AWS refuse the launch with `VcpuLimitExceeded`.
6. THE Start_Command SHALL make every quota and describe call with the Runner_Profile (which holds the Read_Policy) and look up only our own machines by the Project_Tag.

### Requirement 5: Capacity, zones and the Fallback_List at launch

**User Story:** As someone starting a Job, I want the Start_Command to try other zones and then the next type when a type has no capacity, so that a busy zone doesn't block my Job, without ever starting a second machine by mistake.

**Settled in clarification:** Q1 (criterion 2).

#### Acceptance Criteria

1. WHEN a launch fails because AWS has no On-Demand capacity for the type in a zone (`InsufficientInstanceCapacity` or `Unsupported`), THE Start_Command SHALL try the other zones of the Stack_Region that offer the type, and then the next fitting type on the Fallback_List, until a launch succeeds or the list is exhausted.
2. THE Start_Command SHALL treat moving to another zone or type as part of the same approved run, not as a retry or a relaunch, because nothing starts or bills until a launch succeeds (Q1).
3. WHEN a launch succeeds, THE Start_Command SHALL report which type and which zone it got.
4. IF no fitting type on the Fallback_List has capacity in any zone, THEN THE Start_Command SHALL stop with a Readable_Error that says which types and zones were tried, and SHALL start nothing.
5. IF a launch fails with an error that is not a clear capacity error (for example a timeout or an ambiguous error), THEN THE Start_Command SHALL stop rather than try another zone or type, because a `RunInstances` idempotency token only counts within one zone and a blind retry could start a second machine.
6. THE Start_Command SHALL pass a `RunInstances` client token derived from the Job_Id and the attempt, so repeating one attempt starts nothing new, and an approved rerun of the same Job uses a fresh token.
7. THE Start_Command SHALL never start more than one machine for one Job, and SHALL never relaunch a machine by itself after a Job's machine has started.

### Requirement 6: The Spend_Guard and the money quote

**User Story:** As the person paying, I want every paid command to pass the Spend_Guard first and to tell me the machine, the time and the cost, so that nothing paid starts without my yes and I know what I am approving.

#### Acceptance Criteria

1. WHEN the Start_Command, the Image_Build_Command or the Smoke_Command runs, THE command SHALL call the Spend_Guard's `check_spend()` with the machine, the Expected_Time and the rough cost, before it creates, uploads or starts anything.
2. IF the Spend_Variable (`APPROVED_SPEND`) is not set to exactly `1`, THEN THE Spend_Guard SHALL refuse the run with its Readable_Error and THE command SHALL start nothing.
3. BEFORE asking for the yes, THE command SHALL print the chosen type, its `price_per_hour_usd` with its `price_date`, the Expected_Time and the rough cost (the Expected_Time times the price, plus the public IPv4 address and the root volume for the same time), and for the Start_Command the worst case of Requirement 3, criterion 6.
4. THE command SHALL use no interactive prompt, because Kiro tells the person the numbers in the chat and then runs the command once with `APPROVED_SPEND=1` set on that one command.
5. THE List_Command and THE Stop_All_Command SHALL call no Spend_Guard, because listing and terminating cost nothing, so they run without `APPROVED_SPEND`.
6. FOR ALL Presets, chosen types and inputs, no upload and no launch SHALL happen unless the Spend_Guard has already returned without raising (property), which the Test_Suite checks by having the Fake_AWS record every call and asserting the Spend_Guard was satisfied before the first upload or `RunInstances`.

### Requirement 7: The Start_Command's checks, code upload and one command

**User Story:** As someone starting a Job, I want one command that checks the repo is committed, uploads the video and the Pipeline's code, and launches the machine, so that every Job runs code that is in a commit and I run only one command.

**Assumptions confirmed at review:** F (criterion 7), G (criterion 6), I (criterion 5).

#### Acceptance Criteria

1. WHEN the Start_Command runs, THE Start_Command SHALL make the Job from the video with pipeline-core's `make_job` (which runs its own checks), so the Job_Record and the Job_Folder are made the one way the Pipeline makes them.
2. IF the Step_List is empty, THEN THE Start_Command SHALL stop with the same Readable_Error that `splatpipe.run` gives ("No steps are set up yet..."), before it uploads or launches anything, because no real Job can run until a Step spec adds a Step_List entry.
3. IF `splatpipe/`, `cloud/` or `config.toml` has uncommitted changes, THEN THE Start_Command SHALL stop with a Readable_Error that names what is uncommitted, so every Job runs code that is in a commit.
4. THE Start_Command SHALL make the Code_Archive with `git archive HEAD`, record the `HEAD` commit in the Job, and upload the Code_Archive so the machine runs that code, so a code change needs no new Machine_Image and only a `uv.lock` or tool change does.
5. IF `HEAD`'s `uv.lock` or `package-lock.json` differs from the Machine_Image's recorded hashes, THEN THE Start_Command SHALL stop with a Readable_Error that says to rebuild the Machine_Image, before launching, to save a paid boot (assumption I).
6. THE Start_Command SHALL compare both `HEAD`'s `uv.lock` and `HEAD`'s `package-lock.json`, because the Machine_Image installs the npm tree with `npm ci` and a drift there breaks splat-transform the same way a `uv.lock` drift breaks the Python env (assumption G).
7. IF no passing Smoke_Marker exists in the Job_Bucket for the current Machine_Image id, THEN THE Start_Command SHALL stop with a Readable_Error that points to the Smoke_Command, before launching (assumption F).
8. WHEN every check has passed and the person has said yes, THE Start_Command SHALL upload the video, the Code_Archive and the Job_Record to the Job's `input/` under `jobs/<Job_Id>/input/` with the Runner_Profile, then launch one machine.
9. THE Start_Command SHALL find the current Machine_Image as our newest image carrying the Project_Tag that is `available`, and SHALL stop with a Readable_Error pointing to the Image_Build_Command IF there is none.

### Requirement 8: The launch request

**User Story:** As the Account_Owner, I want every launch request Cloud_Jobs builds to carry the settings the Runner_Role demands and to end itself, so that no launch is refused for a missing setting and no machine outlives its Job.

#### Acceptance Criteria

1. WHEN the Start_Command, the Image_Build_Command or the Smoke_Command builds a launch request, THE launch request SHALL set `InstanceInitiatedShutdownBehavior` to `terminate`.
2. THE launch request SHALL put the Project_Tag `project=voxelis-kiroween` on the instance, its volume and its network interface, and SHALL put the Job_Id tag on a Job machine's instance.
3. THE launch request SHALL set `IamInstanceProfile` to the Job_Profile from the `JobInstanceProfileArn` Stack_Output.
4. THE launch request SHALL set `SecurityGroupIds` to the Job_Security_Group from the `SecurityGroupId` Stack_Output, and SHALL place the machine in a default subnet of the Default_VPC from the `VpcId` Stack_Output.
5. THE launch request SHALL name no key pair and no launch template.
6. THE launch request SHALL set `MetadataOptions` to `HttpTokens=required` and `HttpEndpoint=enabled`, so IMDSv2 is required, which the Runner_Role demands.
7. THE launch request SHALL carry User_Data that holds a Time_Cap, so every machine powers off at its Time_Cap even if its work hangs.
8. THE User_Data SHALL be at most 16 KB before base64, SHALL reach EC2 with LF line endings, and SHALL have every value the Start_Command writes into it checked and quoted with `shlex.quote`.
9. FOR ALL Presets, fitting types, zones and Job_Ids, EVERY launch request Cloud_Jobs builds SHALL have `InstanceInitiatedShutdownBehavior=terminate`, the Project_Tag (and the Job_Id tag on a Job instance) on the instance, the volume and the network interface, the Job_Profile, the Job_Security_Group, a subnet of the Default_VPC, no key pair, no launch template, IMDSv2 with `HttpTokens=required` and `HttpEndpoint=enabled`, and a Time_Cap in the User_Data (property), which the Test_Suite checks on the request the code builds, with the Fake_AWS.

### Requirement 9: The Machine_Image and its build

**User Story:** As a team member, I want our own private machine image with every tool and weight pinned and built once, so that a job machine is working minutes after it starts and a code change never needs a new image.

**Assumptions confirmed at review:** G (criterion 4), I (criterion 8).

#### Acceptance Criteria

1. THE Image_Build SHALL make the Machine_Image from the Base_AMI, whose current id the Image_Build_Command reads from the Base_AMI_Parameter with the Read_Policy's permission.
2. THE Machine_Image SHALL hold, every version pinned: our Python environment from `uv.lock`, ffmpeg built with zscale, COLMAP 4.2.1 built from source with `CUDA_ENABLED=OFF`, `OPENGL_ENABLED=OFF`, `GUI_ENABLED=OFF`, `CGAL_ENABLED=OFF` and `LSD_ENABLED=OFF`, gsplat 1.5.3, fused-ssim, the MoGe-2 weights, Node and splat-transform 3.8.0.
3. THE Image_Build SHALL compile gsplat and fused-ssim into the Machine_Image with `TORCH_CUDA_ARCH_LIST="8.6;8.9"`, so the one image runs on the A10G (8.6) and on the L4 and L40S (8.9) and nothing compiles when a Job starts.
4. THE Image_Build SHALL install the npm tree with `npm ci` from the repo's `package-lock.json`, so splat-transform's version is reproducible (assumption G).
5. THE Machine_Image SHALL hold the tools, the weights and the Python environment but never the Pipeline's own code, because the Start_Command uploads the Code_Archive for each Job.
6. THE Machine_Image SHALL stay private, never shared, copied, exported or made public, because it carries GPL builds of ffmpeg and COLMAP's dependencies, and the Runner_Role has no action that would share it.
7. THE Image_Build SHALL tag the Machine_Image and its snapshots with the Project_Tag in the same request that makes the image, which the Runner_Role demands, and SHALL keep the root volume's `DeleteOnTermination` true so no Job leaves a paid volume behind.
8. THE Image_Build SHALL record the `uv.lock` hash and the `package-lock.json` hash as tags on the Machine_Image, so the Start_Command's pre-check and the machine's backstop can compare them (assumptions G and I).
9. WHEN the Image_Build runs, THE Image_Build_Command SHALL launch the builder (a g5.4xlarge) from the Base_AMI with the launch request of Requirement 8 and a Time_Cap, run the install script from the builder's User_Data, then stop the builder through the EC2 API (which never triggers terminate-on-shutdown), make the Machine_Image from the stopped builder, wait until the image is `available`, and terminate the builder.
10. THE Image_Build_Command SHALL be a Paid_Entry_Point that passes the Spend_Guard first and uses the Runner_Profile.

### Requirement 10: The Cloud_Runner on the machine

**User Story:** As someone whose laptop is asleep, I want the machine to run my Job's Steps in order, upload each result as it is done, pick up where a rerun left off, and always end itself, so that I get my results and never pay for a machine that outlived its Job.

**Assumptions confirmed at review:** A (criteria 1, 2, 3, 5 and 6), B (criterion 4), C (criterion 7), H (criterion 10).

#### Acceptance Criteria

1. WHEN the Cloud_Runner starts, THE Cloud_Runner SHALL rebuild the Job_Folder from the Job_Bucket, downloading the Job's `input/` and any `out/` and `work/` from an earlier run of the same Job, then call pipeline-core's `run_job(job_id)` without the Fresh_Option, so `run_job` stays the single source of truth for the Job_Summary, the Job_Log and each Step_Report (assumption A).
2. WHILE `run_job` runs, THE Cloud_Runner's Uploader SHALL upload each Step_Report, the Step's `work/` and `out/` files, the Job_Log and the Job_Summary as they appear, mapping `work/` to `work/<Job_Id>/` and the rest to `jobs/<Job_Id>/`, and SHALL run a final authoritative sync after `run_job` returns (assumption A).
3. THE Uploader SHALL upload a Step_Report only after the Step that owns it has returned, and SHALL never upload a Step_Report while `write_json` is still writing it, because `write_json` is a non-atomic `path.write_text` (assumption A).
4. FOR ALL sets of Step_Reports already in the Job_Bucket, THE Cloud_Runner SHALL run exactly the Steps `run_job` runs: it resumes from the first Step that is not a Finished_Step and never reruns a Step that finished before that resume point, while a Finished_Step after the first unfinished one may run again, exactly as pipeline-core defines resume (property, assumption B).
5. FOR ALL runs, THE set of files the Uploader has uploaded once the Cloud_Runner finishes SHALL equal the Job's `out/`, reports and logs on the machine, and every uploaded Step_Report SHALL parse as a complete JSON object (property), which the Test_Suite checks with the Fake_AWS.
6. WHEN `run_job` returns, done or failed, THE Cloud_Runner SHALL upload the final Job_Summary and Job_Log and then power the machine off with `shutdown -h` or `poweroff`, so the terminate-on-shutdown behaviour ends the machine (assumption A).
7. WHEN the Cloud_Runner starts, THE Cloud_Runner SHALL set the Cap_Deadline environment variable to the machine's start plus the Preset's Time_Cap minus `upload_margin_minutes`, for a Step to read, and `splatpipe/` SHALL stay unchanged in this spec because no current Step reads it (assumption C).
8. IF the Time_Cap is reached, THEN the machine SHALL power off whatever else is happening, through the shutdown the User_Data scheduled, so a hang still ends the machine.
9. IF the Job's `uv.lock` or `package-lock.json` differs from the Machine_Image's recorded hashes, THEN THE Cloud_Runner SHALL stop at once with a Readable_Error that says to rebuild the Machine_Image, as a backstop to the Start_Command's pre-check, and SHALL still end the machine.
10. WHEN the Cloud_Runner uploads a text log (the Job_Log, its own log or `cloud-init-output.log`), THE Cloud_Runner SHALL apply the Account_Redaction to that log first, so a log downloaded into `data/` holds no account id, while leaving the JSON reports unchanged because they hold no account id (assumption H).
11. THE Cloud_Runner SHALL be testable with an injected Step_List of Fake_Steps and the Fake_AWS, so the Test_Suite exercises it on a CPU with no account, no network and no real `run_job` change.

### Requirement 11: The Smoke_Test

**User Story:** As a team member, I want a short, cheap run that proves a fresh image works before any long job, so that I never spend hours on a machine whose image is broken.

**Assumptions confirmed at review:** F (criteria 3 and 5).

#### Acceptance Criteria

1. WHEN the Smoke_Command runs, THE Smoke_Test SHALL launch one g5.xlarge from the current Machine_Image with the launch request of Requirement 8 and the Smoke_Test's Time_Cap.
2. THE Smoke_Test SHALL run every GPU code path for a few seconds on a tiny synthetic scene, and SHALL run the Cloud_Runner end to end on a tiny Fake_Step Job (boot, download, run, upload, shutdown).
3. WHEN the Smoke_Test passes, THE Smoke_Test SHALL write a Smoke_Marker under `jobs/` in the Job_Bucket, keyed to the Machine_Image id it ran on, which the Start_Command checks (assumption F).
4. THE Smoke_Command SHALL be a Paid_Entry_Point that passes the Spend_Guard first and uses the Runner_Profile, and THE machine SHALL end itself when the Smoke_Test ends.
5. THE Smoke_Test SHALL measure how long the machine takes to be ready for work, and each later spec that adds a GPU code path SHALL add that code path to the Smoke_Test (assumption F).

### Requirement 12: Spend_Lines and the Spend_File

**User Story:** As the person paying, I want a running tally of what the machines cost, so that I can see roughly where the money went without reading an AWS bill.

**Settled in clarification:** Q4 (criteria 1, 2, 3 and 4).

#### Acceptance Criteria

1. AFTER each Step, THE Cloud_Runner SHALL append a Spend_Line to the Job's Spend_File under `out/` in the Job_Bucket, with the Job_Id, the Step, the machine type, the seconds and the estimated cost (Q4).
2. THE Cloud_Runner SHALL append a Spend_Line for the machine's time outside the Steps (boot, setup and uploads), and THE Image_Build and THE Smoke_Test SHALL each append a Spend_Line for their machine, so the Spend_File adds up to roughly what the machines cost (Q4).
3. THE every Spend_Line SHALL be labelled an estimate, and SHALL take the price from the type's `price_per_hour_usd` in Config (Q4).
4. WHEN the Status_Command or the Download_Command runs, THE command SHALL add the Job's new Spend_Lines to the laptop's `data/spend.jsonl` without duplicating a line already there (Q4).
5. THE `data/spend.jsonl` SHALL stay under `data/`, which git ignores, so no Spend_Line lands in the repo.

### Requirement 13: Status, download, list and stop-all

**User Story:** As a team member, I want free commands to check a Job, fetch its results, list our machines and stop them all, so that I can watch and clean up without paying and without leaking the account id.

**Assumptions confirmed at review:** H (criterion 6).

#### Acceptance Criteria

1. WHEN the Status_Command runs, THE Status_Command SHALL report the Job's progress from the Step_Reports and the Job_Summary in the Job_Bucket, with the Read_Profile, and add any new Spend_Lines to `data/spend.jsonl`.
2. WHEN the Download_Command runs, THE Download_Command SHALL fetch the Job's `out/` and reports from the Job_Bucket into the Job's `data/jobs/<Job_Id>/` folder, with the Read_Profile, and add any new Spend_Lines to `data/spend.jsonl`.
3. WHEN the List_Command runs, THE List_Command SHALL list every instance carrying the Project_Tag that isn't terminated, with the Read_Profile, showing each instance's type, state and launch time but no account id or ARN.
4. WHEN the Stop_All_Command runs, THE Stop_All_Command SHALL terminate every instance carrying the Project_Tag that isn't terminated, with the Runner_Profile, as the last-resort way to end a machine, and SHALL call no Spend_Guard because terminating is free.
5. THE List_Command and THE Stop_All_Command SHALL look up only our own machines, by the Project_Tag, and SHALL run no command that would show another project's resources.
6. WHEN any Cloud_Jobs command prints an AWS error message, console output or a job log, THE command SHALL apply the Account_Redaction first, so the account id never reaches the chat, which lets Kiro read the output to debug a machine (assumption H).
7. THE Account_Redaction SHALL replace every run of exactly 12 digits with `<account>`, and the Test_Suite SHALL check it on strings that hold an account id and on strings that hold a 12-digit number that is not one.

### Requirement 14: Dependencies and licences

**User Story:** As a team member, I want boto3 and every new tool recorded with its licence in the same commit, so that the laptop install stays permissive and GPU-free and every licence is accounted for.

**Assumptions confirmed at review:** G (criterion 4).

#### Acceptance Criteria

1. THE Project_File SHALL gain boto3 as a main runtime dependency, because the laptop runs the Start_Command, the Status_Command, the Download_Command, the List_Command, the Stop_All_Command, the Image_Build_Command and the Smoke_Command, and boto3 is not a GPU package.
2. THE Project_File SHALL gain no GPU package, so `uv sync` on a laptop still installs no torch, no gsplat, no fused-ssim and no CUDA wheel.
3. THE Licence_List SHALL mark boto3 as installed and gain a row for each package that adding boto3 puts in `uv.lock` (botocore, s3transfer, jmespath, python-dateutil, six and urllib3), each with its licence, all of which are permissive.
4. THE repo root SHALL gain a `package.json` and a `package-lock.json` pinning splat-transform 3.8.0, and THE Licence_List SHALL gain a row for each package in that lockfile whose licence the splat-transform row doesn't already cover (assumption G).
5. THE Licence_List SHALL gain or update a row for the Base_AMI, gsplat 1.5.3, fused-ssim, the MoGe-2 weights, Node, ffmpeg built with zscale and COLMAP 4.2.1, each with its licence and how the Machine_Image uses it, and the COLMAP row SHALL note the build flags that keep SiftGPU, CGAL and LSD out.
6. WHEN the Builder adds a dependency or a tool, THE Builder SHALL add its Licence_List row in the same commit, and SHALL stop and ask the user before committing IF any package has a copyleft, non-commercial or unclear licence.

### Requirement 15: Docs, steering and the skill

**User Story:** As one of the people building the steps, I want the cloud run written up where the rest is and the steering and the skill brought up to date, so that the steps we build separately keep fitting together and the rules match what the code does.

**Assumptions confirmed at review:** C (criterion 2), D (criterion 2), E (criterion 2).

#### Acceptance Criteria

1. THE README SHALL gain a "Run your own copy" section that lists what someone needs (an AWS account with GPU quota, the Stack from the aws-setup spec, and building the Machine_Image once, about $1.20 and about 45 minutes) and the commands in order, and SHALL note those figures are estimates to check against the Config prices.
2. THE Pipeline_Doc SHALL gain a section on the cloud run that describes the Start_Command, the Cloud_Runner, the Status_Command, the Download_Command, the List_Command and the Stop_All_Command, records the per-Step Step_Needs plan of Requirement 2's note (assumptions D and E), and documents the Cap_Deadline environment variable's name and meaning so the training spec can rely on it (assumption C).
3. THE Tech_Steering SHALL change its "one job machine runs at a time" line to the quota-based admission of Q3, so admission is by the real G_Quota and more than one machine may run when the quota allows.
4. THE AWS_Jobs_Skill SHALL change its line that a fallback type is "a new run with a new yes" to the one-yes-covers-the-Fallback_List rule of Q1.
5. THE AWS_Jobs_Skill SHALL change its "Only one job machine runs at a time" note to the quota-based admission of Q3.
6. THE AWS_Jobs_Skill's note on reading a machine's console output SHALL say that Cloud_Jobs applies the Account_Redaction, so Kiro may read the redacted output to debug a machine.
7. THE Pipeline_Doc SHALL say that the cloud run is not a Step and adds no Step_List entry, so running a Job in the cloud runs the same Step_List as a laptop run.

### Requirement 16: Tests

**User Story:** As a team member, I want every part of Cloud_Jobs tested on a laptop CPU with a Fake_Step and a Fake_AWS, so that the tests are quick, free and safe to run with no account and no network.

#### Acceptance Criteria

1. THE Test_Suite SHALL test Cloud_Jobs on a CPU with a Fake_Step and the Fake_AWS, with no account, no network and no GPU, building any tiny synthetic data in the test.
2. THE Test_Suite's unit tests and property tests SHALL both be required, never optional.
3. THE Test_Suite SHALL hold the property test of Requirement 10, criterion 4, written with Hypothesis: for all sets of Step_Reports already present, the Cloud_Runner resumes from the first Step that is not a Finished_Step and never reruns a Step finished before that resume point, matching pipeline-core's resume exactly.
4. THE Test_Suite SHALL hold the property test of Requirement 6, criterion 6, written with Hypothesis: for all Presets, chosen types and inputs, no upload and no launch happens unless the Spend_Guard has already returned without raising.
5. THE Test_Suite SHALL hold the property test of Requirement 8, criterion 9, written with Hypothesis: for all Presets, fitting types, zones and Job_Ids, every launch request Cloud_Jobs builds has terminate-on-shutdown, the Project_Tag (and the Job_Id tag on a Job instance) on the instance, the volume and the network interface, the Job_Profile, the Job_Security_Group, a subnet of the Default_VPC, no key pair, no launch template, IMDSv2 with `HttpTokens=required` and `HttpEndpoint=enabled`, and a Time_Cap in the User_Data.
6. THE Test_Suite SHALL add no Step_List entry and SHALL change no pipeline-core or clean-export test, so `tests/test_run.py`'s assertions that `STEP_LIST == ()` and that `splatpipe.run` refuses with "No steps are set up yet" still hold.
7. THE Test_Suite SHALL check that no file under `splatpipe/` imports `cloud/`, boto3, torch, gsplat or another GPU package, and that the Cloud_Jobs code imports boto3 only inside `cloud/`.
8. THE Test_Suite SHALL make no AWS call and set `APPROVED_SPEND` for no test that would start anything, so running the Test_Suite spends nothing.
