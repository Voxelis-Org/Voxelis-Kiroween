# Requirements Document

## Introduction

The Pipeline (`splatpipe/`) turns a phone video (MP4 or MOV, up to 10 minutes) into a clean 3D Gaussian splat (`.ply`) plus a small copy for the web viewer (`.sog`). This spec builds only the Skeleton that the steps plug into: making a job from a video, the job folder, the settings and presets in `config.toml`, the step reports, running one step or a whole job, picking a job up where it stopped, a Fake_Step for the tests, the Spend_Guard for the cloud code, and `docs/pipeline.md`.

Each real step gets its own spec after this one: picking frames, solving cameras, running it in the cloud, training, cleaning and export, then depth priors and plain walls.

Out of scope:

- Step logic, and folders for the real steps. Only the Fake_Step is built here.
- The cloud wrapper. It comes in its own spec and is not a step. Nothing in this spec runs anything paid.
- New dependencies. The Skeleton, the Fake_Step and the Spend_Guard use only the Python standard library.

### Assumptions confirmed at review

The user didn't ask for these in the first request. At review, the user confirmed A1, A2 and A4 as written and replaced A3 with the restart rule below. Each one is marked in the requirement it belongs to.

- **A1 (Requirement 7):** When a step starts, the Step_Runner removes the step's old Step_Report and empties the step's own Work_Folder, so files from an earlier run can't leak into the new run. The new Step_Report replaces the old one.
- **A2 (Requirement 8):** The Step_List lives in code and starts empty. Each entry names the step (its Step_Name), the module that holds it and whether it's required. The Fake_Step is not in the Step_List, so it never runs on real jobs, and tests pass their own list. With an empty list, `splatpipe.run` refuses with a Readable_Error and a non-zero exit code.
- **A3 (Requirements 7, 8, 9 and 13):** A whole-Job run skips the steps that already finished (done, or an Optional_Step that failed), so a restarted Job picks up where it stopped. Without the Fresh_Option, the Job_Runner skips the finished steps at the start of the Step_List and runs every step from the first unfinished one on. The Fresh_Option (`--fresh`) runs every step. Before any step runs, a run removes the earlier Job_Summary and only the Step_Reports of the steps it's about to run, so a step marked "not run" has no report left over from an earlier run. The Job_Summary marks each skipped step as `reused`. A run without the Fresh_Option adds its lines to the end of the Job_Log, and a run with it starts a new Job_Log. A single step's CLI always runs its step. This A3 replaces the first draft's, under which every run removed the reports of every step in its Step_List.
- **A4 (Requirements 8 and 9):** "Done with warnings" means every Required_Step is done and at least one Optional_Step failed. Warnings inside a Step_Report don't change the Job_Status. The Job_Runner CLI exits with 0 for "done" and for "done with warnings".

## Glossary

- **Pipeline**: the `splatpipe/` package, which turns one phone video into a splat, one step at a time.
- **Skeleton**: the part of the Pipeline this spec builds: making jobs, settings, reports, the Step_Runner and the Job_Runner. It holds no step logic.
- **Step**: one part of the Pipeline's work on one Job, like picking frames. Each Step is a plain Python function plus a CLI.
- **Step_Name**: a Step's short name in lowercase ASCII letters, digits and underscores, starting with a letter, like `fake`. It names the Step's Config sections, its Step_Report and its Work_Folder. A real Step's module is `splatpipe.<Step_Name>`.
- **Reserved_Name**: a name a Step can't take: the name of a Skeleton module (like `run` or `config`), the Skeleton's own Config section, or one of `git`, `aws`, `paid` and `devlog`.
- **Required_Step**: a Step whose failure fails the Job. Picking frames, solving cameras and training will be Required_Steps once their specs add them.
- **Optional_Step**: a Step the Job can do without. Depth priors, plain walls, and clean and export will be Optional_Steps once their specs add them.
- **Step_List**: the ordered list, in code, of the Steps the Job_Runner runs.
- **Job**: one video going through the Pipeline, with its own Job_Id and Job_Folder.
- **Job_Id**: a Job's id, made by the Job_Maker from the UTC date and time and a random part, like `20261119-142530-7f3a9c`: 8 digits, a hyphen, 6 digits, a hyphen and 6 lowercase hex characters.
- **Job_Id_Check**: the check that every CLI and function of the Pipeline runs on a Job_Id it receives.
- **Jobs_Folder**: the folder that holds every Job_Folder: `data/jobs/` (set in Config), or a temporary folder in tests.
- **Job_Folder**: `<Jobs_Folder>/<Job_Id>/`, holding the folders `input/`, `work/` and `out/`.
- **Job_Maker**: the function and CLI that check a video and make a new Job from it.
- **Video_Probe**: the part of the Job_Maker that runs ffprobe on a video and returns the video's container format, streams and length. Tests can replace it.
- **Container_Family**: the format name that ffprobe reports for MP4 and MOV files: `mov,mp4,m4a,3gp,3g2,mj2`.
- **Video_Length**: the whole video's length in seconds, from the first video stream's duration, or from the container's duration when the stream has none.
- **Trim**: an optional start and an optional end, in seconds, that pick the part of the video to use.
- **Used_Part**: the part of the video the Pipeline uses: from the Trim start (0 s when none is given) to the Trim end (the Video_Length when none is given).
- **Job_Record**: `input/job.json` in the Job_Folder.
- **Preset**: one of `fast`, `default` and `quality`. Each Step reads its numbers for the chosen Preset.
- **Config**: `config.toml` at the repo root, read with `tomllib`.
- **Config_Loader**: the code that reads Config and gives the Skeleton and each Step their settings.
- **Pipeline_Sections**: every Config section except `[git]`, `[aws]`, `[paid]` and `[devlog]`: the Skeleton's own Config section, the Step_Sections, and any other section a later spec adds, like `[cloud]` for machine types and prices. Every key in a Pipeline_Section needs a comment (Requirement 5, criteria 2 and 10), and no value in a Pipeline_Section can be NaN, infinite or a TOML date or time (Requirement 5, criterion 7).
- **Step_Section**: a Pipeline_Section that holds a Step's Shared_Settings and Preset_Settings, with the section's Preset tables. Only `[fake]` and the section of each Step in the Step_List in code are Step_Sections. Any other section a later spec adds, like `[cloud]`, is not a Step_Section and needs no code change, but it's still a Pipeline_Section.
- **Shared_Setting**: a Step setting in `[<Step_Name>]`, the same for every Preset.
- **Preset_Setting**: a Step setting set once in each of `[<Step_Name>.fast]`, `[<Step_Name>.default]` and `[<Step_Name>.quality]`.
- **Step_Runner**: the shared code that runs one Step on one Job: it opens the Job, loads the settings, gets the Work_Folder ready, runs the Step's own code, times it and writes the Step_Report.
- **Work_Folder**: `work/<Step_Name>/` in the Job_Folder, where a Step keeps its in-between files.
- **Step_Report**: the JSON report a Step writes to `out/reports/<Step_Name>.json` in the Job_Folder.
- **Step_Outcome**: `done` or `failed` in a Step_Report. The Job_Summary also uses `not run` for Steps the Job_Runner didn't reach. The JSON files spell them `"done"`, `"failed"` and `"not_run"`. The Job_Log and the CLIs use the words.
- **Expected_Failure**: a failure a Step foresees and describes with its own Readable_Error, like "too few sharp frames".
- **Unexpected_Error**: any other error while a Step runs, including a Step module that fails to import.
- **Readable_Error**: a message a normal user can read: what went wrong and what to do about it, with no traceback.
- **Technical_Detail**: an error's type, its message and its traceback, for the team.
- **JSON_Value**: an object with string keys, an array, a string, a finite number, `true`, `false` or `null`, nested in any way.
- **Job_Runner**: the function `run_job` and the CLI `uv run python -m splatpipe.run --job <id> [--fresh]` that run the Step_List on one Job.
- **Finished_Step**: a Step whose Step_Report at `out/reports/<Step_Name>.json` reads as a JSON object whose `step` is the Step_Name, whose `job_id` is the Job's Job_Id and whose `outcome` is `"done"`, or `"failed"` when the Step is an Optional_Step in the Step_List the Job_Runner is running. Any other Step is not a Finished_Step: a Step with no Step_Report, a Step whose Step_Report can't be read as a JSON object, a Step whose Step_Report is for another Step or another Job, and a Required_Step whose Step_Report says `"failed"`.
- **Fresh_Option**: the Job_Runner's option to run every Step in the Step_List, Finished_Steps included: `fresh=True` for the Job_Runner's function (`run_job(..., fresh=True)`) and `--fresh` for the Job_Runner CLI. The Fresh_Option is off unless the caller gives it.
- **Job_Status**: `done`, `done with warnings` or `failed`. The Job_Summary spells them `"done"`, `"done_with_warnings"` and `"failed"`. The Job_Log and the CLIs use the words.
- **Job_Summary**: `out/job.json` in the Job_Folder, written by the Job_Runner.
- **Job_Log**: `out/job.log` in the Job_Folder, written by the Job_Runner.
- **Fake_Step**: the test Step in `splatpipe/fake/`. It does trivial CPU work, and it's the working example the step specs copy.
- **Spend_Guard**: the function in `cloud/spend.py` that every Paid_Entry_Point calls first.
- **Spend_Variable**: the environment variable named by `[aws].spend_variable` in Config, currently `APPROVED_SPEND`.
- **Paid_Entry_Point**: a function or CLI in `cloud/` that starts something that costs money. This spec builds none.
- **Pipeline_Doc**: `docs/pipeline.md`.
- **Licence_List**: `docs/third-party-licences.md`.
- **Structure_Steering**: `.kiro/steering/structure.md`.
- **CLI**: a command line tool run with `uv run python -m <module>`.
- **Test_Suite**: the tests in `tests/`, run with `uv run pytest`.

## Requirements

### Requirement 1: Job ids

**User Story:** As a team member, I want the Pipeline to make every Job_Id itself and to accept only that one pattern, so that a Job_Id can never point at a path outside the Jobs_Folder.

#### Acceptance Criteria

1. WHEN the Job_Maker makes a Job, THE Job_Maker SHALL make the Job_Id from the current UTC date and time and a random part of 6 lowercase hex characters, in the form `YYYYMMDD-HHMMSS-xxxxxx`.
2. WHEN a CLI or function of the Pipeline receives a Job_Id, THE Job_Id_Check SHALL check that the whole string matches the pattern `[0-9]{8}-[0-9]{6}-[0-9a-f]{6}` before the Job_Id is used in any path.
3. IF a Job_Id doesn't match the pattern, THEN THE Job_Id_Check SHALL refuse the Job_Id with a Readable_Error.
4. WHEN a Job_Id matches the pattern, THE Job_Id_Check SHALL check that the Job_Folder's resolved path is inside the Jobs_Folder's resolved path.
5. IF the Job_Folder's resolved path is outside the Jobs_Folder's resolved path, THEN THE Job_Id_Check SHALL refuse the Job_Id with a Readable_Error.
6. THE Pipeline SHALL keep the Job_Id pattern in code, as an exception to the rule that every number lives in Config, because the pattern is a safety check.
7. FOR ALL strings, THE Job_Id_Check SHALL accept the string exactly when the whole string matches the pattern, so every string that holds `/`, `\`, `..`, a line break, a non-ASCII digit, an uppercase letter or any other character outside the pattern is refused (property).
8. FOR ALL UTC times and random parts, THE Job_Id_Check SHALL accept the Job_Id that the Job_Maker makes from those times and random parts (property).

Note for the design: check with `re.fullmatch` and explicit ASCII classes (`[0-9]`, `[0-9a-f]`), never `\d` or `$`. `$` matches before a trailing line break, and `\d` matches non-ASCII digits.

### Requirement 2: Checking the video

**User Story:** As someone who films a place, I want my video checked before anything is copied, so that I find out straight away, in words I understand, when the video can't be used.

#### Acceptance Criteria

1. THE Job_Maker SHALL finish every check (the Job_Id, a free Job_Folder, the Preset, the file, the extension, the size, the Video_Probe and the Trim) before the Job_Maker creates a Job_Folder or copies anything.
2. IF the video path doesn't exist or isn't a regular file, THEN THE Job_Maker SHALL refuse the video with a Readable_Error.
3. IF the video's extension, in any mix of upper and lower case, is not `.mp4` or `.mov`, THEN THE Job_Maker SHALL refuse the video with a Readable_Error that says only MP4 and MOV videos are accepted.
4. IF the video's size is more than the size limit in Config (4 294 967 296 bytes), THEN THE Job_Maker SHALL refuse the video with a Readable_Error that gives the limit in GB ("4 GB").
5. THE Job_Maker SHALL check the extension and the size before the Video_Probe runs.
6. WHEN the Video_Probe reads a video, THE Video_Probe SHALL run ffprobe with the timeout set in Config and pass the video as a local file path, so a file name is never read as an ffprobe option or a network address.
7. IF ffprobe isn't installed, runs past the timeout, exits with an error or prints output the Video_Probe can't read, THEN THE Job_Maker SHALL refuse the video with a Readable_Error that says what went wrong and what to do.
8. IF ffprobe reports a format other than the Container_Family, THEN THE Job_Maker SHALL refuse the video with a Readable_Error.
9. IF ffprobe reports no video stream, THEN THE Job_Maker SHALL refuse the video with a Readable_Error.
10. THE Video_Probe SHALL take the Video_Length from the first video stream's duration, and from the container's duration when the stream has no duration.
11. IF neither the first video stream nor the container gives a finite Video_Length above 0 s, THEN THE Job_Maker SHALL refuse the video with a Readable_Error.
12. IF the Video_Length is more than the length limit in Config (600 s), THEN THE Job_Maker SHALL refuse the video with a Readable_Error that gives the Video_Length and the limit.
13. IF any check fails, THEN THE Job_Maker SHALL leave the Jobs_Folder without a new Job_Folder and copy nothing.

### Requirement 3: The trim

**User Story:** As someone who films a place, I want to pick the part of my video to use with an optional start and end in seconds, so that I can cut off the shaky bits at either end.

#### Acceptance Criteria

1. WHERE no Trim start is given, THE Job_Maker SHALL start the Used_Part at 0 s.
2. WHERE no Trim end is given, THE Job_Maker SHALL end the Used_Part at the Video_Length.
3. WHERE a Trim start or a Trim end is given, THE Job_Maker SHALL use the given value unchanged, without moving the value into the video.
4. IF a Trim start or a Trim end is NaN, infinite or negative, THEN THE Job_Maker SHALL refuse the Trim with a Readable_Error.
5. IF a Trim start or a Trim end is after the Video_Length, THEN THE Job_Maker SHALL refuse the Trim with a Readable_Error that gives the Video_Length.
6. IF the Used_Part's start is not before the Used_Part's end, THEN THE Job_Maker SHALL refuse the Trim with a Readable_Error.
7. IF the Used_Part is shorter than the minimum length in Config (3 s), THEN THE Job_Maker SHALL refuse the video with a Readable_Error that gives the minimum length, also when no Trim is given.
8. FOR ALL Video_Lengths and Trims, with each of the start and the end given or not and given as any number including NaN, infinity and negative numbers, THE Job_Maker SHALL accept exactly the Trims whose Used_Part starts at or after 0 s, ends at or before the Video_Length, has a start before the end and lasts at least the minimum length (property).
9. FOR ALL Video_Lengths and accepted Trims, THE Job_Maker SHALL give a Used_Part whose start equals the given start (0 s when none is given) and whose end equals the given end (the Video_Length when none is given) (property).

### Requirement 4: Making the job

**User Story:** As someone who films a place, I want my checked video copied into its own Job_Folder with a record of my choices, so that the Steps can work on it while my original file stays as it was.

#### Acceptance Criteria

1. WHEN every check passes, THE Job_Maker SHALL create the Job_Folder with the folders `input/`, `work/` and `out/`, creating the Jobs_Folder too when the Jobs_Folder doesn't exist yet.
2. WHEN every check passes, THE Job_Maker SHALL copy the video to `input/video.mp4` or `input/video.mov`, picked by the video's extension, and leave the original file in place and unchanged.
3. WHEN the video is copied, THE Job_Maker SHALL write the Job_Record with the Job_Id, the Preset, the Used_Part's start and end in seconds, the original file name without the folder path, the file's size in bytes, the Video_Length in seconds and the copy's file name in `input/` (`video.mp4` or `video.mov`).
4. WHERE the caller chooses a Preset, THE Job_Maker SHALL save that Preset in the Job_Record.
5. WHERE the caller chooses no Preset, THE Job_Maker SHALL save the default Preset set in Config (`fast`) in the Job_Record.
6. IF the chosen Preset is not one of the Presets, THEN THE Job_Maker SHALL refuse with a Readable_Error that lists the Presets.
7. IF the Job_Folder for the new Job_Id already exists, THEN THE Job_Maker SHALL refuse with a Readable_Error and leave the existing Job_Folder unchanged.
8. IF copying the video or writing the Job_Record fails, THEN THE Job_Maker SHALL remove the Job_Folder the Job_Maker started and refuse with a Readable_Error.
9. WHEN the Job_Maker CLI is run with a video path and an optional start, end and Preset, THE Job_Maker CLI SHALL make the Job and print the new Job_Id on standard output.
10. IF the Job_Maker CLI is given a Job_Id, THEN THE Job_Maker CLI SHALL refuse with a non-zero exit code and make no Job.

### Requirement 5: Settings and presets in config.toml

**User Story:** As a team member, I want every number of the Pipeline in Config with a short comment, and three Presets for each Step, so that we can tune the Pipeline later without changing code.

#### Acceptance Criteria

1. THE Config SHALL hold the Skeleton's numbers and settings in a Pipeline_Section: the Jobs_Folder, the default Preset (`fast`), the size limit in bytes (4 294 967 296), the length limit (600 s), the minimum Used_Part length (3 s) and the ffprobe timeout in seconds.
2. THE Config SHALL give every key in the Pipeline_Sections a short comment, on the key's own line or on the line directly above the key.
3. THE Config SHALL keep the `[git]`, `[aws]`, `[paid]` and `[devlog]` sections unchanged.
4. THE Config SHALL hold each Step's Shared_Settings in `[<Step_Name>]` and the Step's Preset_Settings in `[<Step_Name>.fast]`, `[<Step_Name>.default]` and `[<Step_Name>.quality]`.
5. WHEN a Step asks for the Step's settings for a Preset, THE Config_Loader SHALL return the Step's Shared_Settings together with the Preset_Settings of that Preset.
6. IF a Step's section is missing a Preset table, has a key in only some of the Preset tables, has a key that is both shared and per-Preset, or holds a table whose name is not a Preset, THEN THE Config_Loader SHALL refuse with a Readable_Error that names the section and the key or table.
7. IF a value in a Pipeline_Section is NaN, infinite, or a TOML date or time, THEN THE Config_Loader SHALL refuse with a Readable_Error that names the section and the key.
8. IF a setting that the Skeleton or a Step asks for is missing from Config, THEN THE Config_Loader SHALL refuse with a Readable_Error that names the section and the key.
9. IF a Preset name given to the Pipeline, or the default Preset in Config, is not `fast`, `default` or `quality`, THEN THE Config_Loader SHALL refuse with a Readable_Error that lists the three Presets.
10. THE Test_Suite SHALL check that every key in the Pipeline_Sections of the real Config has a comment.
11. THE Test_Suite SHALL check that each Step_Section (`[fake]` and the section of every Step in the Step_List in code) loads from the real Config for each of the three Presets.

### Requirement 6: Step reports

**User Story:** As a team member, I want every Step to write a JSON report with its settings, timings, counts, warnings and any error, so that a normal user can read what went wrong and we can see why.

#### Acceptance Criteria

1. WHEN a Step finishes or fails, THE Step_Runner SHALL write the Step's Step_Report to `out/reports/<Step_Name>.json` in the Job_Folder.
2. THE Step_Report SHALL hold the Step_Name, the Job_Id, the Preset, the settings the Step used, the Step_Outcome, the start and finish times in UTC, the duration in seconds, the named timings the Step added, the named whole-number counts the Step added and the warnings the Step added.
3. WHEN a Step fails, THE Step_Report SHALL also hold an error with a Readable_Error, whether the failure was an Expected_Failure, and the Technical_Detail.
4. WHEN a Step fails, THE Step_Report SHALL keep the timings, counts and warnings the Step added before the failure.
5. WHEN a Step raises an Expected_Failure, THE Step_Runner SHALL record the Expected_Failure's own Readable_Error in the Step_Report.
6. WHEN a Step raises an Unexpected_Error, THE Step_Runner SHALL record a general Readable_Error that names the Step and says what to do, in the Step_Report.
7. THE Step_Report SHALL hold only JSON_Values.
8. IF a Step adds a value to the Step's Step_Report that is not a JSON_Value, THEN THE Step_Runner SHALL refuse the value with an error that names the field.
9. FOR ALL Step_Reports made of JSON_Values, THE Step_Runner SHALL write the Step_Report so that reading the file back gives a Step_Report equal to the original (round-trip property).

### Requirement 7: Running one step

**User Story:** As a team member, I want to run any one Step on its own on one Job, so that I can build and tune a Step without running the whole Pipeline.

**Assumptions confirmed at review:** A1 (criteria 5 and 6) and A3 (criterion 9).

#### Acceptance Criteria

1. THE Pipeline SHALL give every Step a plain Python function and a CLI, `uv run python -m splatpipe.<Step_Name> --job <id> [--preset <name>]`, that run the Step on one Job through the Step_Runner.
2. WHEN the Step_Runner opens a Job, THE Step_Runner SHALL run the Job_Id_Check and read the Job_Record before the Step's own code runs.
3. IF the Job_Folder doesn't exist, or the Job_Record is missing, unreadable or lacks a field, THEN THE Step_Runner SHALL refuse with a Readable_Error and write no Step_Report.
4. WHERE a Preset is given to the Step's function or CLI, THE Step_Runner SHALL run the Step with that Preset instead of the Preset in the Job_Record, and record the Preset used in the Step_Report.
5. WHEN a Step starts, THE Step_Runner SHALL remove the Step's earlier Step_Report and empty the Step's Work_Folder before the Step's own code runs.
6. THE Step_Runner SHALL leave the `input/` folder, the other Steps' Work_Folders and the other Steps' Step_Reports unchanged.
7. IF the Work_Folder's resolved path is outside the Job_Folder's resolved path, THEN THE Step_Runner SHALL leave the Work_Folder's contents in place and fail the Step with a Readable_Error.
8. IF a Step_Name given to the Step_Runner or listed in a Step_List doesn't match the pattern `[a-z][a-z0-9_]*` or is a Reserved_Name, THEN THE Step_Runner and THE Job_Runner SHALL refuse the Step_Name with a Readable_Error before the Step_Name is used in any path, and write no Step_Report.
9. THE Step_Runner SHALL run every Step the Step_Runner is given, Finished_Steps included, so a Step's own function and CLI (`uv run python -m splatpipe.<Step_Name>`) always run the Step, and only the Job_Runner skips Finished_Steps.
10. THE Step_Runner SHALL leave a Step's files in `out/`, other than the Step's Step_Report, unchanged, so each Step's own code handles the files the Step wrote to `out/` in an earlier run when the Step runs again.

### Requirement 8: The Step_List and the Job_Runner

**User Story:** As a team member, I want one command that runs every Step of a Job in order, so that a Required_Step that fails stops the Job and an Optional_Step that fails is skipped with a reason.

**Assumptions confirmed at review:** A2 (criteria 1, 2, 3 and 10), A3 (criteria 8, 9 and 13 to 20) and A4 (criterion 8).

#### Acceptance Criteria

1. THE Step_List SHALL hold, for each Step in order, the Step_Name, the module that holds the Step, and whether the Step is a Required_Step or an Optional_Step.
2. THE Step_List SHALL start empty in this spec, without a Fake_Step entry. Each step spec adds the entry for the spec's own Step.

   Note: the frames spec added the first Step_List entry, `StepEntry("frames", "splatpipe.frames", True)`, so the Step_List in code is no longer empty.

3. THE Job_Runner SHALL run the Step_List in code by default, and run a Step_List given by the caller instead when one is given.
4. WHEN the Job_Runner runs a Job, THE Job_Runner SHALL run the Steps in the Step_List's order, with the Preset from the Job_Record.
5. THE Job_Runner SHALL import each Step's module only when that Step is about to run.
6. IF a Required_Step fails, THEN THE Job_Runner SHALL stop the Job, mark every later Step `not run`, and set the Job_Status to `failed`.
7. IF an Optional_Step fails, THEN THE Job_Runner SHALL write the Step_Name and the Step's Readable_Error to the Job_Log and go on with the next Step.
8. WHEN the Job ends, THE Job_Runner SHALL write the Job_Summary with the Job_Status (`done` when every Step is done, `done with warnings` when every Required_Step is done and at least one Optional_Step failed, `failed` when a Required_Step failed), each Step's Step_Outcome, including the earlier Step_Outcome of each skipped Finished_Step and `not run` for each Step the Job_Runner didn't reach, and the failed Required_Step's Readable_Error when the Job failed.
9. WHEN the Job_Runner has checked the Step_List, the Job_Id and the Job_Record, THE Job_Runner SHALL remove the earlier Job_Summary and the earlier Step_Reports of the Steps the Job_Runner is about to run, before any Step runs, and leave the Step_Reports of the skipped Finished_Steps and of the Steps outside the Step_List unchanged, so a Step marked `not run` has no Step_Report left over from an earlier run.
10. IF the Step_List is empty, THEN THE Job_Runner SHALL refuse with a Readable_Error that says no Steps are set up yet, before the Job_Runner writes anything.
11. IF a Step_List names the same Step_Name twice, THEN THE Job_Runner SHALL refuse with a Readable_Error before any Step runs.
12. THE Job_Runner SHALL write the Job_Log to `out/job.log`, with a line for the start and a line for the Step_Outcome of each Step the Job_Runner runs, and a `not run` line for each Step the Job_Runner didn't reach.
13. WHEN the Job_Runner runs a Job without the Fresh_Option, THE Job_Runner SHALL skip each Finished_Step that comes before the first Step that isn't a Finished_Step, and run the Steps from that first Step on, in the Step_List's order, including every Finished_Step after that first Step.
14. WHERE the Fresh_Option is given (`fresh=True` to the Job_Runner's function, `--fresh` to the Job_Runner CLI), THE Job_Runner SHALL run the Steps from the first Step in the Step_List on, including every Finished_Step.
15. WHEN the Job_Runner runs a Job without the Fresh_Option and every Step in the Step_List is a Finished_Step, THE Job_Runner SHALL run no Step and write the Job_Summary from the Steps' earlier Step_Reports.
16. THE Job_Summary SHALL hold a `reused` flag for each Step: `true` for a skipped Finished_Step, whose Step_Outcome the Job_Runner took from the Step's earlier Step_Report, and `false` for a Step that ran or that the Job_Runner didn't reach.
17. THE Job_Runner SHALL count each skipped Finished_Step toward the Job_Status with the Step's earlier Step_Outcome, the same as a Step that ran, so a skipped Optional_Step that failed earlier gives the Job_Status `done with warnings` when every Required_Step is done.
18. WHEN the Job_Runner runs a Job without the Fresh_Option, THE Job_Runner SHALL add the run's lines to the end of the existing Job_Log, starting the Job_Log when the Job_Log doesn't exist yet, with a line that names the skipped Finished_Steps when the Job_Runner skips at least one Finished_Step.
19. WHERE the Fresh_Option is given, THE Job_Runner SHALL start a new Job_Log in place of the earlier Job_Log.
20. FOR ALL Step_Lists whose Steps are each a Required_Step or an Optional_Step, all earlier Step_Report states for each Step (no Step_Report, `done`, `failed` or unreadable), all outcomes for each Step when the Step runs (`done` or `failed`) and the Fresh_Option given or not, THE Job_Runner SHALL run exactly the Steps from the first Step that isn't a Finished_Step on (from the first Step in the Step_List when the Fresh_Option is given, and no Step when the Fresh_Option isn't given and every Step is a Finished_Step), in order, up to and including the first Required_Step that fails, and set the Job_Status as criterion 8 says, counting each skipped Finished_Step with the Step's earlier Step_Outcome (property).

Note: criterion 13 runs a Finished_Step again when the Finished_Step comes after a Step that isn't finished, because the Finished_Step's inputs may have changed. In normal use the Finished_Steps always come first, because a run removes the Step_Reports of the Steps the run is about to run (criterion 9). A Finished_Step after an unfinished Step only happens after someone runs a Step by hand or changes or removes a Step_Report, or after the Step_List changes.

### Requirement 9: Command line behaviour

**User Story:** As someone running a CLI, I want a clear exit code and a message I can read, so that scripts can tell whether it worked and people know what to do when it didn't.

**Assumptions confirmed at review:** A3 (criterion 5) and A4 (criterion 4).

#### Acceptance Criteria

1. WHEN a CLI of the Pipeline succeeds, THE CLI SHALL exit with code 0.
2. IF a CLI of the Pipeline fails, THEN THE CLI SHALL print the Readable_Error on standard error, without a traceback, and exit with a non-zero code.
3. WHEN a Step's CLI runs, THE Step's CLI SHALL exit with code 0 when the Step_Outcome is `done` and with a non-zero code when the Step_Outcome is `failed`, for Required_Steps and Optional_Steps alike.
4. WHEN the Job_Runner CLI finishes a Job, THE Job_Runner CLI SHALL exit with code 0 for the Job_Status `done` or `done with warnings`, and with a non-zero code for the Job_Status `failed`.
5. WHEN the Job_Runner CLI runs without the Fresh_Option on a Job where every Step in the Step_List is a Finished_Step, THE Job_Runner CLI SHALL exit with code 0.

### Requirement 10: The fake step

**User Story:** As a team member, I want a Fake_Step that does trivial CPU work, so that the Skeleton can be tested end to end on a laptop and the step specs have a working example to copy.

#### Acceptance Criteria

1. THE Fake_Step SHALL live in `splatpipe/fake/`, with the Fake_Step's own function and CLI, `uv run python -m splatpipe.fake --job <id> [--preset <name>]`.
2. THE Config SHALL hold a `[fake]` section with the Fake_Step's Shared_Settings and with Preset_Settings for `fast`, `default` and `quality`, including the number of files the Fake_Step writes, each key with a comment.
3. WHEN the Fake_Step runs, THE Fake_Step SHALL read the size of the Job's input video, write as many small files into the Fake_Step's Work_Folder as the Preset says, and write one small result file into `out/`.
4. WHEN the Fake_Step finishes, THE Fake_Step's Step_Report SHALL hold the settings for the Preset used, at least one named timing, a count of the files written that equals the Preset's number, a count equal to the input video's size in bytes, and one warning.
5. WHERE a test sets the Fake_Step to fail with an Expected_Failure, THE Fake_Step SHALL fail with an Expected_Failure that carries the Fake_Step's own Readable_Error.
6. WHERE a test sets the Fake_Step to fail with an Unexpected_Error, THE Fake_Step SHALL raise an error that is not an Expected_Failure.
7. WHERE a test puts the Fake_Step in a Step_List as a Required_Step or as an Optional_Step, THE Job_Runner SHALL run the Fake_Step like any other Step.

### Requirement 11: The spend guard

**User Story:** As the account owner, I want one Spend_Guard that every paid entry point calls first, so that nothing paid starts unless someone said yes to that run.

#### Acceptance Criteria

1. THE Spend_Guard SHALL live in `cloud/spend.py` and use only the Python standard library.
2. WHEN the Spend_Guard is called, THE Spend_Guard SHALL take a short description of what is about to start: the machine, the expected time and the rough cost.
3. THE Spend_Guard SHALL read the Spend_Variable's name from `[aws].spend_variable` in Config.
4. WHILE the Spend_Variable is set to exactly `1`, THE Spend_Guard SHALL let the call go on.
5. IF the Spend_Variable is unset or set to any value other than exactly `1`, THEN THE Spend_Guard SHALL refuse with an error whose message gives the description, says the run was refused because the run costs money and the Spend_Variable isn't set to exactly `1`, and says what to do: get a yes in the chat for that machine, time and cost, then run that one command again with the Spend_Variable set to `1`.
6. IF the Spend_Guard can't read Config or `[aws].spend_variable`, or a part of the description is empty, THEN THE Spend_Guard SHALL refuse with a Readable_Error.
7. THE Spend_Guard SHALL leave the environment variables unchanged.
8. FOR ALL values of the Spend_Variable, THE Spend_Guard SHALL refuse every value other than exactly `1`, and refuse when the Spend_Variable is unset (property).

Note: this spec builds no Paid_Entry_Point. The cloud spec's Paid_Entry_Points call the Spend_Guard before they create or start anything, as `.kiro/steering/cloud.md` already requires.

### Requirement 12: No cloud code and no new dependencies in the pipeline

**User Story:** As a team member, I want the Pipeline kept free of cloud code and extra packages, so that every Step runs the same on a laptop and on the job machine.

#### Acceptance Criteria

1. THE Pipeline SHALL import no module named `cloud`, `boto3`, `botocore`, `s3transfer`, `awscrt`, `awscli`, `aiobotocore` or `aioboto3`, and no submodule of one.
2. THE Step_List SHALL name only modules inside `splatpipe`.
3. THE Skeleton and the Fake_Step SHALL import only the Python standard library and modules of the Pipeline.
4. WHEN the Job_Runner's module is imported, THE Job_Runner SHALL import no Step module.
5. THE Test_Suite SHALL check criteria 1 to 4 by reading every Python file in `splatpipe/` and `cloud/spend.py`, without running the Steps.

### Requirement 13: Docs

**User Story:** As one of the three people building the Steps, I want one document that says what each Step reads and writes, so that the Steps we build separately keep fitting together.

**Assumption confirmed at review:** A3 (criterion 8).

#### Acceptance Criteria

1. THE Pipeline_Doc SHALL start with the Skeleton: the Job_Folder layout, the Job_Record fields, the Step_Report format, the Presets and the Config layout, the Step_List and which planned Steps are Required_Steps or Optional_Steps, the Job_Runner, the Job_Summary, the Job_Log and the CLIs with the CLIs' exit codes.
2. THE Pipeline_Doc SHALL have one section per Step that says what the Step reads, what the Step writes, the Step's Config section and the Step's Step_Report fields.
3. THE Pipeline_Doc SHALL hold the Fake_Step's section and say that the step specs copy the Fake_Step as the step specs' working example.
4. THE Structure_Steering SHALL list `pipeline.md` in the Structure_Steering's `docs/` line.
5. THE Structure_Steering SHALL say that every step spec updates the Step's own section in the Pipeline_Doc as part of the spec's tasks, so the Steps the three of us build keep fitting together.
6. THE Licence_List SHALL cover ffprobe in the ffmpeg row: ffprobe reads the video's container, streams and length when a Job is made, and ffmpeg and ffprobe are separate command line tools on the machine image and on laptops for the tests, never vendored or linked.
7. THE Licence_List's ffmpeg row SHALL change in the same task as the Video_Probe.
8. THE Pipeline_Doc SHALL explain which Steps the Job_Runner skips when a Job runs again, and say to use the Fresh_Option after changing settings or after running an earlier Step by hand, because the Job_Runner doesn't check whether a Finished_Step's inputs changed.
9. THE Pipeline_Doc SHALL say that the Step_Runner leaves a Step's files in `out/`, other than the Step's Step_Report, unchanged, so each step spec handles the Step's own files in `out/` when the Step runs again, and list in each Step's section the files the Step writes to `out/`.

### Requirement 14: Tests

**User Story:** As a team member, I want the Skeleton tested on a laptop CPU with no network, so that the tests are quick, free and safe to run anywhere.

#### Acceptance Criteria

1. THE Test_Suite SHALL run on the CPU with no network and only tiny synthetic data.
2. THE Test_Suite SHALL make every Job in a temporary Jobs_Folder, never in the real `data/` folder.
3. THE Test_Suite SHALL replace the Video_Probe in unit tests, so the unit tests need no ffprobe.
4. THE Test_Suite SHALL hold at least one test that runs the real Video_Probe on a tiny clip made with ffmpeg in a temporary folder.
5. IF ffmpeg or ffprobe isn't installed, THEN THE Test_Suite SHALL skip the tests that need ffmpeg or ffprobe and say why.
6. THE Test_Suite SHALL hold the property tests in Requirement 1 (criteria 7 and 8), Requirement 3 (criteria 8 and 9), Requirement 6 (criterion 9), Requirement 8 (criterion 20) and Requirement 11 (criterion 8), written with Hypothesis and required, never optional.
7. THE Test_Suite SHALL test the Job_Runner end to end with Fake_Steps: a Job where every Step is done, a Job where an Optional_Step fails and a Job where a Required_Step fails before a later Step.
