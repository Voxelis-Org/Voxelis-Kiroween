# Requirements Document

## Introduction

This spec builds step 08, the Clean_Export_Step: the last step of the Pipeline. It reads the splat the training step wrote, throws away the parts a viewer shouldn't see (stray floaters, near-transparent blobs, giant smears and anything with a NaN in it), writes the Clean_Splat as a binary `.ply`, and then makes the Web_Copy, a small `.sog` for the web viewer. Those two files are what the person who filmed the place gets back, as `.kiro/steering/product.md` says.

The step is an Optional_Step: when it fails, the Job is done with warnings and the trained splat is still there. The pipeline-core spec's Glossary and the Pipeline_Doc's "The step list" section both already say so.

Everything in this step runs on a CPU. The cleaning is numpy and scipy maths on arrays we read ourselves, and the Web_Copy comes from Splat_Transform, a Node command line tool, run with its GPU work switched off. Nothing here needs a GPU, torch, gsplat or AWS. On a splat of the size we train, the cleaning is the heaviest CPU work the Pipeline does on one machine, so Requirement 3 caps the number of Gaussians the step will take rather than letting it fill the laptop's memory.

Out of scope:

- The training step that writes the Source_Splat. This spec only reads its file and states the contract in A1.
- The Step_List entry. By A4 this spec doesn't add one, so no pipeline-core test changes.
- Viewing the Web_Copy. The web viewer and the app come in the app spec.
- Splat_Transform's GPU-only features (`--filter-floaters`, `--filter-cluster`, `.voxel.json` output, `.webp` output). The step does its own cleaning on the CPU instead, so it runs the same on a laptop and on the job machine.
- Changing any Skeleton module or any pipeline-core test. This spec adds `splatpipe/clean_export/`, one Config section, one section of the Pipeline_Doc, the test config's new section, and its own tests.

### Assumptions to confirm at review

The user didn't ask for these in the first request. Each is marked in the requirement it belongs to, and each is a small change if the review decides otherwise.

- **A1 (Requirement 1, criteria 1 and 2):** The training step writes the Source_Splat to `out/splat.ply` in the Job_Folder, as a Binary_PLY in the usual 3DGS layout. The name is `source_ply` in Config, so agreeing a different name with whoever builds training costs one line and no code. This step never writes `out/splat.ply`, so the trained splat survives a failed clean.
- **A2 (Requirement 4, criterion 1 and Requirement 5, criterion 1):** The Clean_Splat is `out/clean.ply` and the Web_Copy is `out/web.sog`. Both names are in Config.
- **A3 (Requirement 6, criteria 8 to 11):** Splat_Transform is pinned in a `package.json` and `package-lock.json` at the repo root and installed with `npm ci`, so the version is reproducible and every package in its tree can be checked against the Licence_List from the lockfile. The step runs the locally installed binary and never downloads anything at run time. The Licence_List names version 3.8.0 today; 3.12.0 is the newest. The review picks the version, and Requirement 6, criterion 10 makes the licence check part of the same task either way.
- **A4 (Requirement 8, criteria 3 and 4):** This spec adds **no** Step_List entry. `STEP_LIST` stays empty, because `tests/test_run.py` asserts `STEP_LIST == ()` and that `splatpipe.run` refuses with "No steps are set up yet" (three more assertions), and because this step is the last one: on its own in the list it would make every Job "done with warnings" having produced nothing. The entry belongs to whichever spec first has a Step worth running, and that spec renegotiates pipeline-core's Requirement 8.2 and changes those tests once, for everyone. Until then the step runs through its own CLI and its own `run()`, which is how the Pipeline_Doc already tells us to run a step that isn't in the list. Requirement 6, criterion 12 and Requirement 9, criterion 2 carry the checks that `tests/test_imports.py` and `tests/test_docs.py` would have carried had the entry been there.
- **A5 (Requirement 2, criterion 6):** The Outlier_Filter is statistical outlier removal over the k nearest neighbours, which is why this step adds scipy as well as numpy. If the review would rather keep the dependency list to numpy alone, the filter becomes a radial trim about the robust centre, and criteria 2.6, 2.7 and 2.12 change.

## Glossary

- **Clean_Export_Step**: the Step this spec builds, in `splatpipe/clean_export/`. Its Step_Name is `clean_export`, so its Config section is `[clean_export]`, its Work_Folder is `work/clean_export/` and its Step_Report is `out/reports/clean_export.json`.
- **Source_Splat**: the splat the training step wrote, which this step reads: `out/<source_ply>` in the Job_Folder, by A1 `out/splat.ply`.
- **Gaussian**: one splat in a splat file: a position, a rotation, three scales, an opacity and its colour, which is a set of spherical harmonic coefficients.
- **Splat_Data**: the Gaussians of one splat file, read into memory as numpy arrays, one array per PLY property, each with one row per Gaussian.
- **PLY_Header**: the text header at the start of a PLY file: its format line, its `element vertex <count>` line and one `property <type> <name>` line per property.
- **Binary_PLY**: a PLY whose format line says `binary_little_endian` or `binary_big_endian`. An `ascii` PLY is not a Binary_PLY, and Splat_Transform rejects one.
- **Splat_Properties**: the PLY properties of a 3DGS splat: `x`, `y`, `z`, the unused normals `nx`, `ny`, `nz`, the base colour `f_dc_0` to `f_dc_2`, the higher spherical harmonics `f_rest_0` upwards, `opacity`, `scale_0` to `scale_2` and the rotation quaternion `rot_0` to `rot_3`.
- **Required_Properties**: the Splat_Properties this step needs in order to clean: `x`, `y`, `z`, `opacity`, `scale_0`, `scale_1`, `scale_2` and `rot_0` to `rot_3`.
- **Format_Constants**: the numbers of the PLY and 3DGS formats that stay in code, not in Config, because they describe the file format and not our choices: the `f_rest_*` counts 0, 9, 24 and 45 and the 3 coefficients per band behind them, the logistic function behind Real_Opacity, the exponential behind Real_Scale, the 4 components of a rotation quaternion, and the names in Splat_Properties. This is the same exception pipeline-core's Requirement 1, criterion 6 makes for the Job_Id pattern.
- **SH_Bands**: how many bands of higher spherical harmonics a splat holds, from the number of `f_rest_*` properties: 3 coefficients per Gaussian per band, so `f_rest_*` counts of 0, 9, 24 and 45 mean 0, 1, 2 and 3 bands.
- **Real_Opacity**: a Gaussian's opacity as a viewer sees it, between 0 and 1: the logistic function of the PLY's `opacity`, which is a logit.
- **Real_Scale**: a Gaussian's three sizes in world units: the exponential of the PLY's `scale_0`, `scale_1` and `scale_2`, which are logarithms.
- **Non_Finite_Filter**: the filter that drops every Gaussian with a NaN or an infinity in any of its Required_Properties, or whose rotation quaternion has zero length, because no viewer can draw one.
- **Opacity_Filter**: the filter that drops every Gaussian whose Real_Opacity is below `min_opacity`.
- **Scale_Filter**: the filter that drops every Gaussian whose largest Real_Scale is above `max_scale`.
- **Neighbour_Distance**: for one Gaussian, the mean distance from its position to the positions of its `neighbours` nearest other Gaussians. A Gaussian is never its own neighbour.
- **Outlier_Threshold**: `mean + outlier_sigma * standard deviation` of the Neighbour_Distances of **the Gaussians the Outlier_Filter was given**, worked out once, before anything is dropped.
- **Outlier_Filter**: the filter that drops every Gaussian whose Neighbour_Distance is above the Outlier_Threshold (A5).
- **Filter_Order**: the order the filters run in: Non_Finite_Filter, Opacity_Filter, Scale_Filter, then Outlier_Filter.
- **Threshold_Filters**: the Non_Finite_Filter, the Opacity_Filter and the Scale_Filter. Each judges one Gaussian on its own values alone, so running it again on what it kept keeps everything. The Outlier_Filter is not one of them: it judges a Gaussian against the whole set, so dropping Gaussians changes both the Neighbour_Distances and the Outlier_Threshold, and a second pass can drop more.
- **Kept_Splats**: the Gaussians left after every filter.
- **Clean_Splat**: the Binary_PLY this step writes from the Kept_Splats: `out/<clean_ply>` in the Job_Folder, by A2 `out/clean.ply`.
- **Web_Copy**: the small copy for the web viewer that Splat_Transform makes from the Clean_Splat: `out/<web_sog>` in the Job_Folder, by A2 `out/web.sog`. A `.sog` is one bundled file.
- **Step_Outputs**: the Clean_Splat and the Web_Copy together. Requirement 4, criterion 7 keeps them in step, so `out/` never holds a Web_Copy made from a different Clean_Splat.
- **Splat_Transform**: the PlayCanvas `@playcanvas/splat-transform` command line tool (MIT, Node 22 or newer), run as a subprocess to write the Web_Copy.
- **Splat_Transform_Missing**: the case where Node or the locally installed Splat_Transform binary isn't there, which is an Expected_Failure with a Readable_Error that says how to install it.
- **Plain_File_Name**: a Config value that names a file inside `out/` and nothing else: a non-empty string with no `/`, no `\`, no path separator of the running system, not `.` or `..`, holding no `..` part, and not an absolute path or a Windows drive-relative path.
- **Clean_Report**: the Clean_Export_Step's Step_Report, with the fields Requirement 7 lists.

Terms this spec uses as the Pipeline_Doc and the pipeline-core spec define them: Pipeline, Skeleton, Step, Step_Name, Job, Job_Id, Job_Folder, Job_Record, Step_Runner, Step_Report, Step_List, Required_Step, Optional_Step, Fake_Step, Work_Folder, Config, Preset, Shared_Setting, Preset_Setting, Step_Section, Expected_Failure, Unexpected_Error, Readable_Error, Job_Runner, Job_Status, JSON_Value, Pipeline_Doc, Licence_List, CLI and Test_Suite.

## Requirements

### Requirement 1: Reading the trained splat

**User Story:** As a team member, I want the step to read the trained splat itself and to say plainly when it can't, so that a missing or odd splat file gives a message instead of a traceback.

**Assumptions confirmed at review:** A1 (criteria 1 and 2).

#### Acceptance Criteria

1. WHEN the Clean_Export_Step runs, THE Clean_Export_Step SHALL read the Source_Splat from `out/<source_ply>` in the Job_Folder, where `source_ply` is a Shared_Setting (A1).
2. THE Clean_Export_Step SHALL read a Binary_PLY in either byte order, taking the `vertex` element's `element vertex <count>` rows and its Splat_Properties into Splat_Data (A1).
3. THE Clean_Export_Step SHALL parse the PLY_Header itself, with the Splat_Properties and the `f_rest_*` counts as Format_Constants in code.
4. IF the Source_Splat doesn't exist, THEN THE Clean_Export_Step SHALL fail with an Expected_Failure whose Readable_Error names the file and says that the training step writes it, so this step needs training to have run first.
5. IF the Source_Splat isn't a PLY, is an `ascii` PLY, has no `vertex` element, or lacks any of the Required_Properties, THEN THE Clean_Export_Step SHALL fail with an Expected_Failure whose Readable_Error names the file and what was wrong with it, naming the missing properties when that's the reason.
6. IF the Source_Splat's `vertex` count is 0, THEN THE Clean_Export_Step SHALL fail with an Expected_Failure saying the trained splat holds no Gaussians.
7. IF the Source_Splat's data is shorter than its `vertex` count and properties need, THEN THE Clean_Export_Step SHALL fail with an Expected_Failure saying the file is cut short or still being written.
8. THE Clean_Export_Step SHALL work out the SH_Bands from the number of `f_rest_*` properties, and SHALL fail with an Expected_Failure naming the count IF that count isn't 0, 9, 24 or 45.
9. THE Clean_Export_Step SHALL leave the Source_Splat's bytes exactly as they were, which the Test_Suite checks by comparing the file before and after a done run and a failed run (Requirement 10, criterion 8).
10. THE Clean_Export_Step SHALL read the Source_Splat only from inside the Job_Folder, which Requirement 6, criterion 7 secures by making `source_ply` a Plain_File_Name.

### Requirement 2: Cleaning the splat

**User Story:** As someone who films a place, I want the floaters and the invisible blobs taken out of my splat, so that what I get back looks like the place I filmed and not a cloud of specks.

**Assumptions confirmed at review:** A5 (criteria 6, 7 and 12).

#### Acceptance Criteria

1. WHEN the Clean_Export_Step has read the Splat_Data, THE Clean_Export_Step SHALL run the filters in Filter_Order and keep the Gaussians that pass every one.
2. THE Non_Finite_Filter SHALL drop every Gaussian that has a NaN or an infinity in any Required_Property, or whose rotation quaternion `rot_0` to `rot_3` has zero length.
3. THE Opacity_Filter SHALL drop every Gaussian whose Real_Opacity is below the Preset_Setting `min_opacity`, where Real_Opacity is the logistic function of the PLY's `opacity` (a Format_Constant).
4. THE Scale_Filter SHALL drop every Gaussian whose largest Real_Scale is above the Preset_Setting `max_scale`, where Real_Scale is the exponential of the PLY's `scale_0`, `scale_1` and `scale_2` (a Format_Constant).
5. THE Scale_Filter SHALL compare scales so that a large `scale_*` raises no overflow warning and gives no infinity that escapes the filter, which the Test_Suite checks with numpy's floating-point errors set to raise.
6. THE Outlier_Filter SHALL drop every Gaussian whose Neighbour_Distance is above the Outlier_Threshold, where `neighbours` and `outlier_sigma` are Preset_Settings and a Gaussian is never its own neighbour (A5).
7. THE Outlier_Filter SHALL work out the Outlier_Threshold once, from the Gaussians it was given, before it drops any (A5).
8. WHERE the Gaussians the Outlier_Filter is given number `neighbours` or fewer, THE Outlier_Filter SHALL keep them all rather than fail, because a Gaussian can't have more neighbours than there are other Gaussians.
9. WHERE every Neighbour_Distance is the same, so their standard deviation is 0, THE Outlier_Filter SHALL keep every Gaussian.
10. FOR ALL Splat_Data and settings, EACH filter SHALL give back a subset of the Gaussians it was given, in their original order, with their values unchanged, and SHALL never add or duplicate a Gaussian (property).
11. FOR ALL Splat_Data and settings, EVERY Gaussian a Threshold_Filter keeps SHALL satisfy that filter's rule, EVERY Gaussian it drops SHALL break it, and running that filter again on what it kept SHALL keep all of them (property).
12. FOR ALL Splat_Data and settings, EVERY Gaussian the Outlier_Filter keeps SHALL have a Neighbour_Distance no greater than the Outlier_Threshold of the set it was given, and EVERY Gaussian it drops SHALL have one above it. THE Outlier_Filter is not required to be idempotent, because dropping Gaussians changes both the Neighbour_Distances and the Outlier_Threshold (property, A5).
13. THE Clean_Export_Step SHALL count how many Gaussians each filter dropped, for the Clean_Report.

### Requirement 3: Keeping enough of the splat, and not too much of it

**User Story:** As someone who films a place, I want to be told when cleaning would leave almost nothing, so that I don't get back an empty file and think the pipeline worked.

#### Acceptance Criteria

1. IF fewer than `min_kept_splats` Gaussians are left after every filter, THEN THE Clean_Export_Step SHALL fail with an Expected_Failure and write neither Step_Output, where `min_kept_splats` is a Shared_Setting.
2. THE Readable_Error for criterion 1 SHALL give how many Gaussians the Source_Splat held, how many are left, how many each filter dropped, and that the settings in `[clean_export]` can be loosened.
3. WHERE the Kept_Splats are fewer than `warn_kept_fraction` of the Gaussians the Source_Splat held but at least `min_kept_splats`, THE Clean_Export_Step SHALL add a warning to the Clean_Report saying how much was dropped, and SHALL go on, where `warn_kept_fraction` is a Shared_Setting.
4. IF the Source_Splat's `vertex` count is above `max_source_splats`, THEN THE Clean_Export_Step SHALL fail with an Expected_Failure, before it reads the Gaussians into memory, whose Readable_Error gives both numbers and says that the setting can be raised on a machine with more memory, where `max_source_splats` is a Shared_Setting.

### Requirement 4: Writing the clean splat

**User Story:** As someone who films a place, I want the clean splat as a normal `.ply`, so that I can open it in any splat viewer or editor.

**Assumptions confirmed at review:** A2 (criterion 1).

#### Acceptance Criteria

1. WHEN the Kept_Splats pass Requirement 3, THE Clean_Export_Step SHALL write them to the Clean_Splat as a `binary_little_endian` PLY holding the same Splat_Properties, in the same order, as the Source_Splat (A2).
2. THE Clean_Export_Step SHALL write the Clean_Splat's PLY_Header with `element vertex <number of Kept_Splats>`.
3. FOR ALL Splat_Data, reading back a Clean_Splat the step wrote SHALL give Splat_Data with the same properties, in the same order, holding bit-for-bit the same values as the Kept_Splats (round-trip property).
4. THE Clean_Export_Step SHALL write the Clean_Splat into its Work_Folder first and move it into `out/` only once it's whole, so a failure or a crash can't leave a half-written Clean_Splat in `out/`.
5. IF writing the Clean_Splat fails because the disk is full or the file can't be written, THEN THE Clean_Export_Step SHALL fail with an Expected_Failure whose Readable_Error names the file and the reason.
6. THE Clean_Export_Step SHALL write the Clean_Splat only inside the Job_Folder, which Requirement 6, criterion 7 secures by making `clean_ply` a Plain_File_Name.
7. WHEN the Clean_Export_Step starts its work on a Job, THE Clean_Export_Step SHALL remove both Step_Outputs from `out/` before it writes either, so that `out/` holds either both Step_Outputs from this run or neither, and never a Web_Copy made from an earlier Clean_Splat. The Step_Runner leaves a Step's files in `out/` alone, so the step does this itself.

### Requirement 5: The small copy for the web viewer

**User Story:** As someone who films a place, I want a small copy of my splat, so that it opens in a browser without downloading hundreds of megabytes.

**Assumptions confirmed at review:** A2 (criterion 1).

#### Acceptance Criteria

1. WHEN the Clean_Splat is written, THE Clean_Export_Step SHALL run Splat_Transform on it to write the Web_Copy as one bundled `.sog` file (A2).
2. THE Clean_Export_Step SHALL pass Splat_Transform the option that puts its GPU work on the CPU (`-g cpu`), so the step needs no GPU and gives the same result on a laptop and on the job machine.
3. THE Clean_Export_Step SHALL pass Splat_Transform the option that keeps only `web_sh_bands` bands of higher spherical harmonics (`--filter-harmonics <web_sh_bands>`), where `web_sh_bands` is a Preset_Setting.
4. WHERE `web_sh_bands` is above the Source_Splat's SH_Bands, THE Clean_Export_Step SHALL pass the Source_Splat's SH_Bands instead and SHALL add a warning to the Clean_Report naming both numbers, rather than fail.
5. THE Clean_Export_Step SHALL pass Splat_Transform no option that needs a GPU, so not `--filter-floaters`, `--filter-cluster`, `.voxel.json` output or `.webp` output, which the Test_Suite checks by asserting on the argument list the step builds (Requirement 10, criterion 9).
6. THE Clean_Export_Step SHALL have Splat_Transform write into the Work_Folder and SHALL move the result into `out/` only once Splat_Transform has succeeded and the file exists, so a failed run leaves no half-written Web_Copy in `out/`.
7. IF Node or the locally installed Splat_Transform binary isn't there (Splat_Transform_Missing), THEN THE Clean_Export_Step SHALL fail with an Expected_Failure whose Readable_Error says to run `npm ci` at the repo root and names the Node version Splat_Transform needs (a Format_Constant of the tool, not a tunable).
8. IF Splat_Transform exits with a non-zero code, THEN THE Clean_Export_Step SHALL fail with an Expected_Failure whose Readable_Error holds Splat_Transform's own last lines of output, trimmed to a length the step sets, with no traceback.
9. IF Splat_Transform hasn't finished within `splat_transform_timeout_seconds`, THEN THE Clean_Export_Step SHALL stop it and fail with an Expected_Failure naming the limit, where `splat_transform_timeout_seconds` is a Shared_Setting.
10. IF Splat_Transform exits with 0 but the Web_Copy file it was told to write doesn't exist or is empty, THEN THE Clean_Export_Step SHALL fail with an Expected_Failure saying so.
11. THE Clean_Export_Step SHALL write the Web_Copy only inside the Job_Folder, which Requirement 6, criterion 7 secures by making `web_sog` a Plain_File_Name.
12. THE Clean_Export_Step SHALL record both Step_Outputs' sizes in bytes in the Clean_Report.
13. THE Clean_Export_Step SHALL run Splat_Transform with no shell, as an argument list, so a file name can't be read as a command.

### Requirement 6: Settings, presets and dependencies

**User Story:** As a team member, I want every threshold of this step in Config with three Presets, so that we can tune the cleaning without changing code.

**Assumptions confirmed at review:** A3 (criteria 8 to 11), A5 (criterion 13).

#### Acceptance Criteria

1. THE Config SHALL hold a `[clean_export]` Step_Section with the three Preset tables `[clean_export.fast]`, `[clean_export.default]` and `[clean_export.quality]`, each with the same keys, and every key SHALL have a short comment.
2. THE Shared_Settings SHALL be `source_ply`, `clean_ply`, `web_sog`, `min_kept_splats`, `warn_kept_fraction`, `max_source_splats` and `splat_transform_timeout_seconds`.
3. THE Preset_Settings SHALL be `min_opacity`, `max_scale`, `neighbours`, `outlier_sigma` and `web_sh_bands`.
4. THE Clean_Export_Step SHALL take every threshold, count, size, fraction and time limit from Config, and SHALL hold in code only the Format_Constants.
5. IF a setting is missing from `[clean_export]` and its Preset table, THEN THE Step_Runner's existing check SHALL fail the Step with its Readable_Error, which already names the section, the table and the key, with no new code in this step.
6. IF a setting is present but isn't the kind of value it must be, THEN THE Clean_Export_Step SHALL fail with an Expected_Failure naming the setting, its section and what it must be: `min_kept_splats`, `max_source_splats` and `neighbours` whole numbers above 0, `warn_kept_fraction` a number from 0 to 1, `min_opacity` a number from 0 to 1, `max_scale`, `outlier_sigma` and `splat_transform_timeout_seconds` numbers above 0, and `web_sh_bands` a whole number from 0 to 3.
7. IF `source_ply`, `clean_ply` or `web_sog` isn't a Plain_File_Name, THEN THE Clean_Export_Step SHALL fail with an Expected_Failure naming the setting and the value's problem, before it opens, writes or removes anything, so a setting can't reach a path outside `out/`.
8. IF `clean_ply` and `web_sog` are the same name, or either is the same as `source_ply`, THEN THE Clean_Export_Step SHALL fail with an Expected_Failure naming them, so the step can't overwrite the trained splat or its own other output.
9. THE `pyproject.toml` SHALL gain numpy and scipy as runtime dependencies (A5), and SHALL gain no GPU package, so `uv sync` on a laptop still installs no torch, no gsplat and no CUDA wheel.
10. THE repo root SHALL gain a `package.json` and a `package-lock.json` that pin one version of Splat_Transform (A3).
11. THE Licence_List SHALL gain or update a line for every package in `package-lock.json` whose licence isn't already covered by the Splat_Transform line, with any bundled library noted, in the same task that adds the lockfile, and the task SHALL check every one of those licences against the permissive list in `.kiro/steering/rules.md` and stop for the user IF any package has a copyleft, non-commercial or unclear licence (A3).
12. THE Test_Suite SHALL check that `[clean_export]` loads from the real Config for each of the three Presets, and that its keys are exactly those of criteria 2 and 3, because `STEP_LIST` is empty by A4 and so the Step_List check in `tests/test_imports.py` doesn't cover this section.
13. THE Test_Suite SHALL check that no file under `splatpipe/` imports torch, gsplat or another GPU package, because `tests/test_imports.py` checks only the AWS names across every file under `splatpipe/` and checks the standard-library-only rule with `glob`, which leaves a step folder like `splatpipe/clean_export/` out.
14. THE `tests/conftest.py` test config SHALL gain a `[clean_export]` section with its three Preset tables, so a test that runs the step through its CLI or its `run()` without passing a Config of its own gets settings, and the tests SHALL be able to change any one of them for one test.

### Requirement 7: The step report

**User Story:** As a team member, I want the step's report to say how much it threw away and how big the two files are, so that I can tell a good clean from one that ate the scene.

#### Acceptance Criteria

1. WHEN the Clean_Export_Step is done, THE Clean_Report SHALL hold the counts `source_splats`, `dropped_non_finite`, `dropped_opacity`, `dropped_scale`, `dropped_outlier`, `kept_splats`, `clean_ply_bytes` and `web_sog_bytes`, and the timings `read_ply`, `clean`, `write_ply` and `export_web`.
2. WHEN the Clean_Export_Step fails, THE Clean_Report SHALL hold exactly those counts and timings of criterion 1 that the step had already worked out before it failed, and no others, so a failed Clean_Report says how far the step got.
3. THE Clean_Export_Step SHALL add each count as a whole number and each timing as a number of seconds, so the Step_Runner's checks on report values never refuse one and never fail the step as a bug.
4. WHERE the step warns under Requirement 3, criterion 3 or Requirement 5, criterion 4, THE Clean_Report SHALL hold that warning.
5. THE Clean_Report SHALL be the Step_Runner's ordinary Step_Report, written to `out/reports/clean_export.json`, with no change to the Step_Runner.
6. THE Clean_Report's `settings` SHALL be the twelve settings of Requirement 6, criteria 2 and 3, which the Step_Runner puts there itself.

### Requirement 8: Running the step

**User Story:** As a team member, I want to run the clean and export step on its own on one Job, so that I can tune the cleaning without training again.

**Assumptions confirmed at review:** A4 (criteria 3 and 4).

#### Acceptance Criteria

1. THE Clean_Export_Step SHALL be a package `splatpipe/clean_export/` built like the Fake_Step: `work(step)`, `run(job_id, preset=None, *, config=None)` and a `__main__.py` whose `main(argv)` calls the Skeleton's `step_main`.
2. THE CLI `uv run python -m splatpipe.clean_export --job <id> [--preset P]` SHALL run the step on one Job, print `clean_export: done` on stdout and exit with 0 when the step is done, and print the Readable_Error on stderr with no traceback and exit with 1 when it failed.
3. THIS spec SHALL leave `STEP_LIST` empty and SHALL change no pipeline-core test (A4).
4. WHERE a later spec adds this step's Step_List entry, THE entry SHALL be optional and SHALL sit after the training step's, so later specs insert their entries before it (A4).
5. THE Clean_Export_Step SHALL import only the standard library, numpy, scipy and `splatpipe`, and SHALL import nothing from `cloud/`, boto3, torch or gsplat, which Requirement 6, criterion 13 and the existing AWS check in `tests/test_imports.py` together check.
6. WHEN the Clean_Export_Step runs again on a Job it already ran on, THE Clean_Export_Step SHALL replace both Step_Outputs, by Requirement 4, criterion 7.
7. THE Clean_Export_Step SHALL write every in-between file into its Work_Folder, which the Step_Runner empties before each run, and SHALL write nothing anywhere else in the Job_Folder but its two Step_Outputs.
8. THE Clean_Export_Step SHALL raise `ExpectedFailure` for every failure Requirements 1, 3, 4, 5 and 6 name, so the Clean_Report's `error.expected` is `true` and its `readable` is the step's own message.

### Requirement 9: Docs and licences

**User Story:** As one of the three people building the steps, I want this step written up where the others are, so that the steps we build separately keep fitting together.

#### Acceptance Criteria

1. THE Pipeline_Doc SHALL gain a `## Step: clean_export` section with the same parts as the Fake_Step's, including a `### What it reads`, a `### What it writes`, a `### Config section`, a `### Step report fields` and a `### Files in out/` part, which are the headings `tests/test_docs.py` looks for.
2. THE Test_Suite SHALL check criterion 1's headings for this section by name, because `STEP_LIST` is empty by A4 and so `tests/test_docs.py` doesn't reach this section through the Step_List.
3. THE Pipeline_Doc's `### Files in out/` part for this step SHALL name both Step_Outputs and SHALL say that the step removes both before it writes either, so `out/` never holds a mismatched pair (Requirement 4, criterion 7).
4. THE Pipeline_Doc SHALL state the contract this step relies on: that the training step writes the Source_Splat as a Binary_PLY at `out/<source_ply>` (A1), and that this step never writes that file.
5. THE Pipeline_Doc's "The step list" section SHALL say that this step's entry isn't in `STEP_LIST` yet and which spec adds it (A4).
6. THE Licence_List SHALL mark numpy, scipy and Splat_Transform as installed, with the pinned Splat_Transform version, and SHALL gain a line for Node itself worded like the other command line tools' lines: a separate tool on our machines, never vendored, linked or shipped.
7. THE `README.md` SHALL say that this step needs `npm ci` at the repo root and Node 22 or newer, next to how the rest is installed.

### Requirement 10: Tests

**User Story:** As a team member, I want this step tested on a laptop CPU with tiny files and no network, so that the tests are quick, free and safe to run anywhere.

#### Acceptance Criteria

1. THE Test_Suite SHALL test the PLY reader and writer, every filter, the Requirement 3 limits, the Config checks, the report fields and the CLI, on tiny synthetic splats the tests build themselves, with no network and no GPU.
2. THE Test_Suite's unit tests and property tests SHALL both be required, never optional.
3. THE Test_Suite SHALL build its synthetic splats as bytes in the test, so no splat file is committed to git and nothing is written into the real `data/`.
4. THE Test_Suite SHALL cover every Expected_Failure of Requirements 1, 3, 4, 5 and 6, and SHALL check that each Readable_Error names what the requirement says it names and holds no traceback.
5. THE Test_Suite SHALL use property tests with Hypothesis for the properties marked in Requirement 2, criteria 10, 11 and 12 and Requirement 4, criterion 3.
6. THE Test_Suite SHALL skip the tests that actually run Splat_Transform when Node or the installed binary isn't there, with a reason that says which is missing, and SHALL test the step's handling of Splat_Transform's failures (criteria 7 to 10 of Requirement 5) with a stand-in instead, so the whole suite passes on a machine with no Node.
7. THE Test_Suite SHALL NOT run Splat_Transform on anything but a tiny synthetic splat, and SHALL give every subprocess a timeout, as the pipeline-core spec's working rules require.
8. THE Test_Suite SHALL check that a done run and a failed run both leave the Source_Splat's bytes unchanged (Requirement 1, criterion 9).
9. THE Test_Suite SHALL check the argument list the step builds for Splat_Transform: that it holds `-g cpu` and `--filter-harmonics` with the Preset's band count, and none of the GPU-only options of Requirement 5, criterion 5.
10. THE Test_Suite SHALL check that a run which fails at the Web_Copy leaves neither Step_Output in `out/`, and that a second done run replaces both (Requirement 4, criterion 7 and Requirement 8, criterion 6).
11. THE Test_Suite SHALL check that the step writes nothing in the Job_Folder outside its Work_Folder, its two Step_Outputs and its Step_Report, by listing the Job_Folder before and after a run.
