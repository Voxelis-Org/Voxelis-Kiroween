# Requirements Document

## Introduction

This spec builds the Clean_Step, the last step of the Pipeline. It reads the splat the training step wrote, drops the Gaussians a viewer shouldn't see (the broken ones, the ones too faint to contribute, and the isolated floaters), writes the Clean_Splat as a binary `.ply`, and then makes the Viewer_Copy, a small `.sog` for the web viewer. Those two files are what the person who filmed the place gets back, as `.kiro/steering/product.md` says.

The Clean_Step is an Optional_Step: when it fails, the Job ends done with warnings and the trained splat is still there, untouched. The pipeline-core spec's Glossary and the Pipeline_Doc's "The step list" section both already say so.

The Clean_Step runs on the Job's cloud machine, and every part of it is CPU work: numpy and scipy maths on arrays the step reads itself, plus Splat_Transform as a subprocess with its GPU work switched off. Nothing here needs a GPU, torch, gsplat or AWS, so the step gives the same result on a laptop as on the job machine. The step is sized for the 8 vCPUs and 8 GB a Step gets on that machine, which is why Requirement 3 caps the number of Gaussians the step will take rather than letting the arrays fill the machine's memory.

This spec replaces the retired `clean-export` spec, which covered the same step under an older hand-off. What carries over: the PLY reader and writer, the "left almost nothing" safety, the Plain_File_Name rules, the subprocess cases and the whole-folder discipline. What changed: the Step_Name is `clean`, the input is `out/train.ply`, the faint and isolated rules are local and scale-free instead of global, a removal cap replaces the old global thresholds as the main safety, and there is no max-scale filter.

Out of scope:

- The training step that writes the Source_Splat. This spec only reads that file and states the contract in A1.
- A max-scale filter (A6). The retired spec dropped a Gaussian whose largest Real_Scale was above a limit, to catch giant smears. The Isolated_Filter catches most floaters, and a later spec can add a max-scale rule if the training output needs one.
- The Step_List entry. By A9 this spec adds none, so no pipeline-core test changes.
- Viewing the Viewer_Copy. The web viewer and the app come in the app spec, which also decides which splat-transform version the viewer is pinned to.
- Splat_Transform's GPU-only features (`--filter-floaters`, `--filter-cluster`, `.voxel.json` output and `.webp` output), and `--decimate`, which has to be the last action and has to write a `.ply`, so it would need a second run. The step does its own cleaning on the CPU instead.
- 2DGS splats, which have `scale_0` and `scale_1` but no `scale_2`, and PLYs that carry `opacity_linear` in place of `opacity`. Training writes three scales and a logit opacity (A1), so the Clean_Step refuses either of those as a missing Required_Property.
- Changing any Skeleton module or any pipeline-core test. This spec adds `splatpipe/clean/`, one Config section, one section of the Pipeline_Doc, the test config's new section, a root `package.json` and `package-lock.json`, and its own tests.

### Assumptions to confirm at review

The user didn't ask for these in the first request. Each is marked in the requirement it belongs to, and each is a small change if the review decides otherwise. A3, A10, A11 and A12 record decisions from the first review.

- **A1 (Requirement 1, criteria 1 to 5, Requirement 6, criterion 4 and Requirement 9, criterion 4):** The training step writes the Source_Splat to `out/train.ply` in the Job_Folder as a standard 3DGS Binary_PLY: `binary_little_endian`, float32 values, with `x`, `y`, `z`, `f_dc_0` to `f_dc_2`, `f_rest_*` (45 of them at SH degree 3), `opacity` stored before the sigmoid, `scale_0` to `scale_2` stored as logarithms, `rot_0` to `rot_3`, and sometimes `nx`, `ny`, `nz`. The name is `train_ply` in Config, so agreeing a different name with whoever builds training costs one line and no code. The Clean_Step never writes `out/train.ply`, so the trained splat survives a failed clean.
- **A2 (Requirement 4, criterion 1, Requirement 5, criterion 1 and Requirement 6, criterion 4):** The Clean_Splat is `out/clean.ply` and the Viewer_Copy is `out/clean.sog`. Both names are in Config.
- **A3 (Requirement 2, criteria 4 to 7, 9 and 10, and Requirement 6, criterion 4):** The Faint_Filter is a cube rule, which came from the review in place of the first draft's opacity floor and neighbour test. A Faint_Gaussian is one whose Real_Opacity is below `min_opacity` 0.005. There is no opacity floor, so no Gaussian is dropped for its opacity alone: every Faint_Gaussian is judged together with the other Gaussians in its Cube. The Cube_Edge is the Extent divided by `faint_cube_divisions` 128, and a Cube keeps its Faint_Gaussians when it holds at least `faint_min_per_cube` 8 of them and its Faint_Paint is greater than `faint_min_paint_share` 0.10 times its Cube_Paint. So a sheet of faint Gaussians that together paint a visible surface stays, and faint Gaussians that are few, or that paint little beside the Gaussians around them, go. The Extent comes from the Trimmed_Box, from the 1st to the 99th percentile of the centres with `extent_trim_percent` 1, so that a few far-off Gaussians can't make the Cubes huge. Those far-off Gaussians still land in Cubes of the same size outside the Trimmed_Box. The Faint_Filter uses no neighbour search.
- **A4 (Requirement 3, criteria 2 to 6 and Requirement 6, criterion 4):** Broken Gaussians are always removed and don't count toward the Removal_Cap. The cap is on the faint and isolated removals together, as a fraction of the Source_Splat's Gaussian count, with `max_removed_fraction` 0.70. Past the cap the step stops: it writes neither Step_Output rather than trimming the removals back to the cap, because stopping needs no ranking of Gaussians. The retired spec's "left almost nothing" safety stays as well, as `min_kept_splats` (a failure) and `warn_kept_fraction` (a warning). The settled Shared_Setting list didn't name those last two, and the safety needs a number for each, so this spec adds both as Shared_Settings. `min_kept_splats` is 10 000, because a phone-video scene trains to hundreds of thousands of Gaussians or more, so fewer than 10 000 is a scene that has gone. `warn_kept_fraction` is 0.5, so the step warns once the clean keeps less than half.
- **A5 (Requirement 2, criteria 12 to 15 and Requirement 6, criterion 4):** The Isolated_Filter is a local ratio, with `neighbours` 8 and `isolation_ratio` 8.0. The user was explicit that the test has to be local: a global distance cutoff eats the sky and the far background. The Isolated_Filter is the only filter that searches for neighbours.
- **A6 (the out-of-scope list):** This spec has no max-scale filter.
- **A7 (Requirement 3, criterion 1, Requirement 6, criteria 4, 7 and 17, Requirement 8, criterion 9 and Requirement 9, criterion 6):** A Step on the Job's cloud machine gets 8 vCPUs and 8 GB of memory. So `max_workers` is at most 8, and the Config's 4 is Splat_Transform's own default. `max_source_splats` is at most the number of Gaussians whose Splat_Data takes a quarter of 8 GB, which at 62 float32 properties per Gaussian (248 bytes) is 2 × 10⁹ / 248, or 8 064 516, and the Config's 8 000 000 is the largest round number under that. The remaining three quarters are for the one neighbour search, which is the Isolated_Filter's KD-tree, for the filters' own arrays and for the copy the step writes. That budget holds at `neighbours` 8. `neighbours` has no upper bound beyond being a whole number above 0, so a much larger value can need more memory than a Step gets.
- **A8 (Requirement 5, criteria 2 and 10, Requirement 6, criteria 12 and 13, and Requirement 9, criterion 6):** Splat_Transform 3.8.0 is pinned in a `package.json` and `package-lock.json` at the repo root and installed with `npm ci`, so the version is reproducible and every package in its tree can be checked against the Licence_List from the lockfile. That tree holds Splat_Transform's own dependencies, `webgpu`, which ships Dawn native binaries, and `@adobe/spz`, and its peer dependency `playcanvas`, which npm installs too. The step starts `node`, found on the PATH, with the installed package's `bin/cli.mjs`, and downloads nothing at run time.
- **A9 (Requirement 6, criterion 14, Requirement 8, criteria 3 and 4, and Requirement 9, criteria 2 and 5):** This spec adds **no** Step_List entry. `STEP_LIST` stays empty, because `tests/test_run.py` asserts `STEP_LIST == ()` and asserts the Job_Runner's "No steps are set up yet" refusal in three places, and because this step is the last one: alone in the list it would make every Job "done with warnings" having produced nothing. The entry belongs to whichever spec first has a Step worth running, and that spec renegotiates pipeline-core's Requirement 8, criterion 2 and changes those tests once, for everyone. Until then the Clean_Step runs through its own CLI and its own `run()`, which is how the Pipeline_Doc already tells us to run a step that isn't in the list. Requirement 6, criterion 14 and Requirement 9, criterion 2 carry the checks that `tests/test_imports.py` and `tests/test_docs.py` would have carried had the entry been there.
- **A10 (the Glossary's Nearest_Neighbours, Requirement 2, criteria 12, 21 and 22, and Requirement 10, criterion 13):** When more than one other Gaussian is at the Centre_Distance of a Gaussian's `neighbours`-th nearest, which of them count among its Nearest_Neighbours is left to the order of scipy's KD-tree, which the Isolated_Filter searches. That order is the same on every run with the same data and scipy version, and isn't otherwise specified, so a scipy upgrade could change which of two equally near Gaussians counts. The choice can change a Neighbour_Median but not a Neighbour_Distance, and real training output almost never has such ties. The property tests whose result depends on the Isolated_Filter's neighbour search avoid tied cases, and the dense-grid property, whose grid is full of ties, holds whatever the order.
- **A11 (Requirement 2, criterion 3 and Requirement 9, criterion 4):** The Broken_Filter looks at the Required_Properties only, so a NaN or an infinity in a colour property (`f_dc_*` or `f_rest_*`), or in any other Carried_Property, is carried into the Clean_Splat unchanged and reaches Splat_Transform. The review accepted this choice: training gives finite colours unless it diverges, and a diverged run has non-finite centres too, which the Broken_Filter drops. What Splat_Transform 3.8.0 does with a non-finite colour isn't documented. It may fail, which costs the run both Step_Outputs and gives a Readable_Error under Requirement 5, criterion 11, or it may spoil some of the Viewer_Copy's colours.
- **A12 (Requirement 2, criterion 8):** When the Extent is 0, the Cube_Edge would be 0, so the Faint_Filter puts every Gaussian in one Cube and judges all the Faint_Gaussians together. The Extent is 0 when the Trimmed_Box is a single point, as with a set of one Gaussian or a set whose centres nearly all coincide.

## Glossary

- **Clean_Step**: the Step this spec builds, in `splatpipe/clean/`. Its Step_Name is `clean`, so its Config section is `[clean]`, its Work_Folder is `work/clean/` and its Step_Report is `out/reports/clean.json`.
- **Source_Splat**: the splat the training step wrote, which the Clean_Step reads: `out/<train_ply>` in the Job_Folder, by A1 `out/train.ply`.
- **Gaussian**: one splat in a splat file: a centre, a rotation, three scales, an opacity and its colour, which is a set of spherical harmonic coefficients.
- **Splat_Data**: the Gaussians of one splat file, read into memory as numpy arrays, one array per PLY property, each with one row per Gaussian.
- **PLY_Header**: the text header at the start of a PLY file: its format line, its `comment` lines, its `element vertex <count>` line and one `property <type> <name>` line per property.
- **Binary_PLY**: a PLY whose format line says `binary_little_endian` or `binary_big_endian`. An `ascii` PLY is not a Binary_PLY, and Splat_Transform rejects one.
- **Splat_Properties**: the PLY properties of a standard 3DGS splat: `x`, `y`, `z`, the unused normals `nx`, `ny`, `nz`, the base colour `f_dc_0` to `f_dc_2`, the higher spherical harmonics `f_rest_0` upwards, `opacity`, `scale_0` to `scale_2` and the rotation quaternion `rot_0` to `rot_3`.
- **Required_Properties**: the Splat_Properties the Clean_Step needs in order to clean: `x`, `y`, `z`, `opacity`, `scale_0`, `scale_1`, `scale_2` and `rot_0` to `rot_3`.
- **Carried_Properties**: every other property of the Source_Splat's `vertex` element, which the Clean_Step doesn't read any meaning into and carries through unchanged: `nx`, `ny`, `nz`, the colour properties, and any non-standard property a producer added.
- **Format_Constants**: the numbers and names of the PLY and 3DGS formats that stay in code, not in Config, because they describe the file format and not our choices: the PLY scalar types and their sizes, the `f_rest_*` counts 0, 9, 24 and 45 and the 3 coefficients per band behind them, the logistic function behind Real_Opacity, the exponential behind Real_Scale, the π in Paint, the 4 components of a rotation quaternion, the names in Splat_Properties, float32's smallest gap and largest value, which bound how small a Cube_Edge can matter, and the Node version Splat_Transform needs and the path of its `bin` entry. This is the same exception pipeline-core's Requirement 1, criterion 6 makes for the Job_Id pattern.
- **SH_Bands**: how many bands of higher spherical harmonics a splat holds, from the number of `f_rest_*` properties: 3 coefficients per Gaussian per band, so `f_rest_*` counts of 0, 9, 24 and 45 mean 0, 1, 2 and 3 bands.
- **Real_Opacity**: a Gaussian's opacity as a viewer sees it, between 0 and 1: the logistic function of the PLY's `opacity`, which is a logit.
- **Real_Scale**: one of a Gaussian's three sizes in world units: the exponential of the PLY's `scale_0`, `scale_1` or `scale_2`, which are logarithms.
- **Centre_Distance**: the Euclidean distance between two Gaussians' centres. A Gaussian is never its own neighbour, so a Gaussian's nearest neighbours are always other Gaussians.
- **Broken_Filter**: the filter that drops every Gaussian with a NaN or an infinity in any Required_Property, or whose rotation quaternion has zero length, because nothing can draw one. It looks at no other property, so a NaN or an infinity in a colour stays (A11).
- **Faint_Gaussian**: a Gaussian whose Real_Opacity is below `min_opacity` (A3).
- **Trimmed_Box**: the box, with its sides along the x, y and z axes, that runs on each axis from the `extent_trim_percent` percentile to the (100 − `extent_trim_percent`) percentile of the centres of the Gaussians the Faint_Filter was given, each percentile interpolated linearly between the two nearest values, as numpy's `percentile` does by default. By A3 it runs from the 1st to the 99th percentile.
- **Extent**: the length of the Trimmed_Box's longest side.
- **Cube_Edge**: the Extent divided by `faint_cube_divisions`, which is the length of every Cube's side.
- **Cube**: one of the cubes, each with sides one Cube_Edge long, that tile all of space, with one Cube's low corner at the Trimmed_Box's low corner. On each axis a Cube runs from its low face up to, but not including, its high face, so every point is in exactly one Cube. A Gaussian is in the Cube its centre falls in, so a Gaussian whose centre is outside the Trimmed_Box is in a Cube of the same size outside it. When the Extent is 0, there is one Cube, which holds every Gaussian (A12).
- **Paint**: for one Gaussian, its Real_Opacity × π × the product of its two largest Real_Scales, which is the opacity-weighted area of the Gaussian's largest slice through its centre, taken at one standard deviation.
- **Cube_Paint**: for one Cube, the summed Paint of every Gaussian in the Cube, the Faint_Gaussians and the rest alike.
- **Faint_Paint**: for one Cube, the summed Paint of the Faint_Gaussians in the Cube.
- **Faint_Filter**: the filter that drops the Gaussians too faint to contribute: it keeps every Gaussian that isn't a Faint_Gaussian, and keeps the Faint_Gaussians of a Cube only when the Cube holds at least `faint_min_per_cube` of them and its Faint_Paint is greater than `faint_min_paint_share` times its Cube_Paint (A3).
- **Nearest_Neighbours**: for one Gaussian, the `neighbours` other Gaussians with the smallest Centre_Distance to it. A Gaussian is never one of its own Nearest_Neighbours. When more than one other Gaussian is at the Centre_Distance of the `neighbours`-th nearest, which of them count is left to the order of scipy's KD-tree, which the Isolated_Filter searches (A10).
- **Neighbour_Distance**: for one Gaussian, the mean Centre_Distance to its Nearest_Neighbours.
- **Neighbour_Median**: for one Gaussian, the median of the Neighbour_Distances of its Nearest_Neighbours.
- **Isolated_Filter**: the filter that drops every Gaussian whose Neighbour_Distance is above `isolation_ratio` times its own Neighbour_Median (A5). The test is local, so a sparse but even region, like the far background, survives.
- **Filter_Order**: the order the filters run in: Broken_Filter, then Faint_Filter, then Isolated_Filter. Each filter is given what the one before it kept.
- **Local_Filters**: the Faint_Filter and the Isolated_Filter. Each judges a Gaussian against the set it was given, the Faint_Filter through the Gaussian's Cube and the Isolated_Filter through its Nearest_Neighbours, so dropping Gaussians changes the judgement and a second pass can drop more. The Broken_Filter is not one: it judges a Gaussian on its own values alone, so running it again on what it kept keeps everything.
- **Kept_Splats**: the Gaussians left after every filter.
- **Capped_Removals**: the Gaussians the Faint_Filter and the Isolated_Filter dropped, counted together. The Broken_Filter's drops are not Capped_Removals (A4).
- **Removed_Fraction**: the Capped_Removals divided by the Source_Splat's Gaussian count.
- **Removal_Cap**: the rule that the Removed_Fraction may be at most `max_removed_fraction` (A4).
- **Clean_Splat**: the Binary_PLY the Clean_Step writes from the Kept_Splats: `out/<clean_ply>` in the Job_Folder, by A2 `out/clean.ply`.
- **Viewer_Copy**: the small copy for the web viewer that Splat_Transform makes from the Clean_Splat: `out/<clean_sog>` in the Job_Folder, by A2 `out/clean.sog`. A `.sog` is one bundled file, and Splat_Transform 3.8.0 writes SOG meta version 2.
- **Step_Outputs**: the Clean_Splat and the Viewer_Copy together. Requirement 4, criteria 8 to 13 keep them in step, so `out/` never holds a Viewer_Copy without the Clean_Splat it was made from.
- **Splat_Transform**: the PlayCanvas `@playcanvas/splat-transform` command line tool, version 3.8.0 (MIT, Node 22 or newer), run as a subprocess to write the Viewer_Copy.
- **Splat_Transform_Missing**: the case where `node` isn't on the PATH, or the locally installed Splat_Transform's `node_modules/@playcanvas/splat-transform/bin/cli.mjs` isn't under the repo root, which is an Expected_Failure with a Readable_Error that says how to install it.
- **Plain_File_Name**: a Config value that names a file inside `out/` and nothing else: a non-empty string with no `/`, no `\`, no `:`, no control character (a character below U+0020), no path separator of the running system, not `.` or `..`, holding no `..` part, and not an absolute path or a Windows drive-relative path. The `:` rule also refuses an NTFS stream name like `clean.ply:x`.
- **Clean_Report**: the Clean_Step's Step_Report, with the fields Requirement 7 lists.

Terms this spec uses as the Pipeline_Doc and the pipeline-core spec define them: Pipeline, Skeleton, Step, Step_Name, Job, Job_Id, Job_Folder, Job_Record, Step_Runner, Step_Report, Step_List, Required_Step, Optional_Step, Fake_Step, Work_Folder, Config, Preset, Shared_Setting, Preset_Setting, Step_Section, Expected_Failure, Unexpected_Error, Readable_Error, Job_Runner, Job_Status, JSON_Value, Pipeline_Doc, Licence_List, CLI and Test_Suite.

## Requirements

### Requirement 1: Reading the trained splat

**User Story:** As a team member, I want the Clean_Step to read the trained splat itself and to say plainly when it can't, so that a missing or odd splat file gives a message instead of a traceback.

**Assumptions to confirm at review:** A1 (criteria 1 to 5).

#### Acceptance Criteria

1. WHEN the Clean_Step runs, THE Clean_Step SHALL read the Source_Splat from `out/<train_ply>` in the Job_Folder, where `train_ply` is a Shared_Setting (A1).
2. THE Clean_Step SHALL read a Binary_PLY in either byte order, taking the `vertex` element's `element vertex <count>` rows into Splat_Data, one array per property (A1).
3. THE Clean_Step SHALL parse the PLY_Header itself and find every property by name, with the Splat_Properties and the `f_rest_*` counts as Format_Constants in code (A1).
4. THE Clean_Step SHALL keep every property of the `vertex` element in Splat_Data, each with its name, its declared type and its position in the row, the Required_Properties and the Carried_Properties alike (A1).
5. THE Clean_Step SHALL work out the SH_Bands from the number of `f_rest_*` properties, and SHALL fail with an Expected_Failure naming the count IF that count isn't 0, 9, 24 or 45 (A1).
6. IF the Source_Splat doesn't exist, THEN THE Clean_Step SHALL fail with an Expected_Failure whose Readable_Error names the file and says that the training step writes it, so this step needs training to have run first.
7. IF the Source_Splat isn't a PLY, is an `ascii` PLY, has no `vertex` element, or lacks any of the Required_Properties, THEN THE Clean_Step SHALL fail with an Expected_Failure whose Readable_Error names the file and what was wrong with it, naming the missing properties when that's the reason.
8. IF the `vertex` element declares a list property, THEN THE Clean_Step SHALL fail with an Expected_Failure naming that property, because a list property has no fixed size in the row and so can't be carried through.
9. IF the Source_Splat's `vertex` count is 0, THEN THE Clean_Step SHALL fail with an Expected_Failure saying the trained splat holds no Gaussians.
10. IF the Source_Splat's data is shorter than its `vertex` count and properties need, THEN THE Clean_Step SHALL fail with an Expected_Failure saying the file is cut short or still being written.
11. THE Clean_Step SHALL leave the Source_Splat's bytes exactly as they were, on a done run and on a failed run alike, which the Test_Suite checks by comparing the file before and after each (Requirement 10, criterion 8).
12. THE Clean_Step SHALL read the Source_Splat only from inside the Job_Folder, which Requirement 6, criterion 9 secures by making `train_ply` a Plain_File_Name.

### Requirement 2: Cleaning the splat

**User Story:** As someone who films a place, I want the broken, faint and isolated splats taken out, so that what I get back looks like the place I filmed and not a cloud of specks around it.

**Assumptions to confirm at review:** A3 (criteria 4 to 7, 9 and 10), A5 (criteria 12 to 15), A10 (criteria 12, 21 and 22), A11 (criterion 3) and A12 (criterion 8).

#### Acceptance Criteria

1. WHEN the Clean_Step has read the Splat_Data, THE Clean_Step SHALL run the filters in Filter_Order and keep the Gaussians that pass every one.
2. THE Broken_Filter SHALL drop every Gaussian that has a NaN or an infinity in any Required_Property, or whose rotation quaternion `rot_0` to `rot_3` has zero length.
3. THE Broken_Filter SHALL judge a Gaussian on the Gaussian's Required_Properties alone, so the Broken_Filter keeps a Gaussian whose only NaN or infinity is in a Carried_Property, such as a colour, and the Clean_Step carries that value into the Clean_Splat unchanged (A11).
4. THE Faint_Filter SHALL keep every Gaussian that isn't a Faint_Gaussian, where `min_opacity` is a Shared_Setting (A3).
5. THE Faint_Filter SHALL decide on every Faint_Gaussian by criterion 9's test on the Faint_Gaussian's Cube, so no Gaussian is dropped for its Real_Opacity alone, however low that is (A3).
6. THE Faint_Filter SHALL work out the Trimmed_Box, the Extent, the Cube_Edge, every Gaussian's Cube and Paint, and every Cube's Cube_Paint and Faint_Paint once, from the whole set the Faint_Filter was given, which is what the Broken_Filter kept, before the Faint_Filter drops any Gaussian, where `extent_trim_percent` and `faint_cube_divisions` are Shared_Settings (A3).
7. THE Faint_Filter SHALL put every Gaussian in the Cube the Gaussian's centre falls in, a centre outside the Trimmed_Box included, so far-off Gaussians land in Cubes of the same Cube_Edge outside the Trimmed_Box and don't set the Cube_Edge (A3).
8. IF the Extent is 0, THEN THE Faint_Filter SHALL put every Gaussian the Faint_Filter was given in one Cube (A12).
9. THE Faint_Filter SHALL keep the Faint_Gaussians of every Cube that holds at least `faint_min_per_cube` Faint_Gaussians and whose Faint_Paint is greater than `faint_min_paint_share` times the Cube's Cube_Paint, and SHALL drop the Faint_Gaussians of every other Cube, where `faint_min_per_cube` and `faint_min_paint_share` are Shared_Settings. The share test is that product, with no division, so a Cube whose Cube_Paint is 0 keeps no Faint_Gaussians, because 0 isn't greater than 0 (A3).
10. WHERE the set the Faint_Filter is given holds no Gaussian, THE Faint_Filter SHALL give back no Gaussian, because an empty set has no Trimmed_Box (A3).
11. THE Faint_Filter SHALL work out Real_Opacity, Paint and every Gaussian's Cube so that no finite stored `opacity`, `scale_*` or centre, however far from 0, raises a floating-point warning or gives an infinity that changes which Gaussians the Faint_Filter keeps, which the Test_Suite checks with numpy's floating-point errors set to raise.
12. THE Isolated_Filter SHALL drop every Gaussian whose Neighbour_Distance is above `isolation_ratio` times that Gaussian's own Neighbour_Median, where `neighbours` and `isolation_ratio` are Shared_Settings (A5, A10).
13. THE Isolated_Filter SHALL work out every Neighbour_Distance once, from the set the Isolated_Filter was given, before the Isolated_Filter drops any Gaussian, and SHALL take each Gaussian's Neighbour_Median from the Neighbour_Distances of that same set (A5).
14. WHERE the set the Isolated_Filter is given holds `neighbours` Gaussians or fewer, THE Isolated_Filter SHALL keep every one, because a Gaussian can't have more neighbours than there are other Gaussians (A5).
15. WHERE a Gaussian's Neighbour_Median is 0, THE Isolated_Filter SHALL keep that Gaussian, rather than divide by zero (A5).
16. THE Isolated_Filter SHALL take the distance between two Gaussians as the Centre_Distance, and SHALL never count a Gaussian among its own Nearest_Neighbours.
17. THE Clean_Step SHALL count how many Gaussians each filter dropped, for the Clean_Report.
18. FOR ALL Splat_Data and settings, EACH filter SHALL give back a subset of the Gaussians the filter was given, in their original order, with every property value unchanged, and SHALL never add or duplicate a Gaussian (property).
19. FOR ALL Splat_Data, EVERY Gaussian the Broken_Filter keeps SHALL have finite Required_Properties and a rotation quaternion of non-zero length, EVERY Gaussian the Broken_Filter drops SHALL break one of those two rules, and running the Broken_Filter again on what it kept SHALL keep all of them (property).
20. FOR ALL Splat_Data and settings, EVERY Gaussian the Faint_Filter is given that isn't a Faint_Gaussian SHALL be kept, EVERY Faint_Gaussian the Faint_Filter keeps SHALL be in a Cube that meets both conditions of criterion 9, and EVERY Faint_Gaussian the Faint_Filter drops SHALL be in a Cube that fails at least one of them, with the Cubes, their Cube_Paint and their Faint_Paint worked out on the whole set the Faint_Filter was given (property).
21. FOR ALL Splat_Data and settings, EVERY Gaussian the Isolated_Filter keeps SHALL satisfy the Isolated_Filter's rule of criteria 12, 14 and 15 as worked out on the set the Isolated_Filter was given, and EVERY Gaussian the Isolated_Filter drops SHALL break it. Neither Local_Filter is required to be idempotent, because dropping Gaussians changes the Trimmed_Box, the Cubes' Cube_Paint and Faint_Paint, the Neighbour_Distances and the Neighbour_Medians (property).
22. FOR ALL Splat_Data holding more than `neighbours` Gaussians laid out on an evenly spaced grid, with finite properties, unit quaternions and every Real_Opacity at or above `min_opacity`, THE Clean_Step SHALL keep every Gaussian, whatever order the KD-tree puts the grid's equally near Gaussians in, by A10 (property).
23. FOR ALL Splat_Data and settings, THE Clean_Step SHALL either fail with the Removal_Cap's Expected_Failure of Requirement 3, criterion 2 or leave a Removed_Fraction of at most `max_removed_fraction` (property).

### Requirement 3: Stopping when the clean would eat the scene

**User Story:** As someone who films a place, I want the step to stop when cleaning would take most of my splat away, so that I don't get back a hollow file and think the pipeline worked.

**Assumptions to confirm at review:** A4 (criteria 2 to 6) and A7 (criterion 1).

#### Acceptance Criteria

1. IF the Source_Splat's `vertex` count is above `max_source_splats`, THEN THE Clean_Step SHALL fail with an Expected_Failure, before the Clean_Step reads the Gaussians into memory, whose Readable_Error gives both numbers and says that the setting can be raised on a machine with more memory, where `max_source_splats` is a Shared_Setting (A7).
2. IF the Removed_Fraction is above `max_removed_fraction`, THEN THE Clean_Step SHALL fail with an Expected_Failure and write neither Step_Output, where `max_removed_fraction` is a Shared_Setting (A4).
3. THE Readable_Error for criterion 2 SHALL give the Source_Splat's Gaussian count, how many Gaussians each filter dropped, the Removed_Fraction and the cap, and SHALL say that the Broken_Filter's drops don't count toward the cap and which settings in `[clean]` can be loosened (A4).
4. WHEN the Removed_Fraction is above `max_removed_fraction`, THE Clean_Step SHALL stop, rather than keep part of the faint and isolated removals and write the Step_Outputs anyway, so the Clean_Step never has to rank one Gaussian against another (A4).
5. IF fewer than `min_kept_splats` Gaussians are left after every filter, THEN THE Clean_Step SHALL fail with an Expected_Failure and write neither Step_Output, whose Readable_Error gives the Source_Splat's Gaussian count, how many Gaussians are left, how many each filter dropped and which settings in `[clean]` can be loosened, where `min_kept_splats` is a Shared_Setting (A4).
6. WHERE the Kept_Splats are fewer than `warn_kept_fraction` of the Source_Splat's Gaussian count but at least `min_kept_splats`, and the Removed_Fraction is at most `max_removed_fraction`, THE Clean_Step SHALL add a warning to the Clean_Report that gives both counts and the Removed_Fraction, and SHALL go on, where `warn_kept_fraction` is a Shared_Setting (A4).
7. THE Clean_Step SHALL check the Removal_Cap, then `min_kept_splats`, then `warn_kept_fraction`, and SHALL finish all three checks before the Clean_Step writes either Step_Output, so the Readable_Error for a given splat and settings is always the same one.
8. WHEN the Clean_Step fails under this requirement, THE Clean_Step SHALL raise `ExpectedFailure`, so that the Job_Runner ends the Job done with warnings, because the Clean_Step is an Optional_Step, and the Source_Splat stays in `out/` for another attempt with looser settings.

### Requirement 4: Writing the clean splat

**User Story:** As someone who films a place, I want the clean splat as a normal `.ply`, so that I can open it in any splat viewer or editor.

**Assumptions to confirm at review:** A2 (criterion 1).

#### Acceptance Criteria

1. WHEN the Kept_Splats pass Requirement 3, THE Clean_Step SHALL write the Kept_Splats to the Clean_Splat as a `binary_little_endian` PLY holding the same properties, in the same order and with the same declared types as the Source_Splat's `vertex` element, the Carried_Properties included and unchanged (A2).
2. THE Clean_Step SHALL write the Clean_Splat's PLY_Header with `element vertex <number of Kept_Splats>`.
3. THE Clean_Step SHALL carry the Source_Splat's PLY_Header `comment` lines through to the Clean_Splat unchanged and in their original order, because a comment such as `SplatRenderMode` says how the scene has to be rendered.
4. FOR ALL Splat_Data, reading back a Clean_Splat the Clean_Step wrote SHALL give Splat_Data with the same properties, in the same order, holding bit-for-bit the same values as the Kept_Splats (round-trip property).
5. THE Clean_Step SHALL write the Clean_Splat into the Work_Folder first and move the file into `out/` only once the file is whole, so a failure or a crash can't leave a half-written Clean_Splat in `out/`.
6. IF writing the Clean_Splat fails because the disk is full or the file can't be written, THEN THE Clean_Step SHALL fail with an Expected_Failure whose Readable_Error names the file and the reason.
7. THE Clean_Step SHALL write the Clean_Splat only inside the Job_Folder, which Requirement 6, criterion 9 secures by making `clean_ply` a Plain_File_Name.
8. WHEN the Clean_Step starts its work on a Job, THE Clean_Step SHALL remove both Step_Outputs from `out/`, the Viewer_Copy first, before the Clean_Step writes either. The Step_Runner leaves a Step's files in `out/` alone, so the Clean_Step does this itself.
9. IF the Clean_Step can't remove an earlier Step_Output under criterion 8, THEN THE Clean_Step SHALL fail with an Expected_Failure whose Readable_Error names the file and the reason, before the Clean_Step writes either Step_Output.
10. IF the Clean_Step stops with an Expected_Failure, an Unexpected_Error or an interrupt such as Ctrl+C after criterion 8 has removed the earlier Step_Outputs, THEN THE Clean_Step SHALL leave neither Step_Output in `out/`. A failure before criterion 8's removal, such as a setting that Requirement 6, criterion 9 refuses, leaves the earlier Step_Outputs where they were, and a failure under criterion 9 leaves those the Clean_Step hadn't yet removed.
11. THE Clean_Step SHALL move the Step_Outputs into `out/` only once both are whole in the Work_Folder, the Clean_Splat first and the Viewer_Copy straight after.
12. IF a hard stop that the Clean_Step can't see, such as a killed process or a power cut, falls between the Clean_Step's two moves into `out/`, THEN THE Clean_Step SHALL leave behind, of the two Step_Outputs, only this run's whole Clean_Splat, with no done Clean_Report, and SHALL remove that Clean_Splat first when the Clean_Step next runs on the Job, by criterion 8. A hard stop between criterion 8's two removals can likewise leave the earlier Clean_Splat alone, which the next run removes the same way.
13. THE Clean_Step SHALL leave a Viewer_Copy in `out/` only together with the Clean_Splat that the Viewer_Copy was made from, at every point of a run, a hard stop included.

### Requirement 5: The viewer copy

**User Story:** As someone who films a place, I want a small copy of my splat, so that the splat opens in a browser without downloading hundreds of megabytes.

**Assumptions to confirm at review:** A2 (criterion 1) and A8 (criteria 2 and 10).

#### Acceptance Criteria

1. WHEN the Clean_Splat is written, THE Clean_Step SHALL run Splat_Transform on the Clean_Splat the Clean_Step just wrote, to make the Viewer_Copy as one bundled `.sog` file (A2).
2. THE Clean_Step SHALL run the locally installed Splat_Transform by starting `node`, found on the PATH, with the script `node_modules/@playcanvas/splat-transform/bin/cli.mjs` under the repo root, which is the `bin` entry in Splat_Transform 3.8.0's `package.json`, and SHALL download nothing at run time (A8). Starting `node` itself lets the Clean_Step pass an argument list with no shell (criterion 8), which npm's `node_modules/.bin/splat-transform` doesn't allow on Windows, where it is a `.cmd` shim that needs a shell.
3. THE Clean_Step SHALL pass Splat_Transform the option that puts its GPU work on the CPU (`-g cpu`), so the Clean_Step needs no GPU, gives the same result on a laptop and on the job machine, and needs no Vulkan driver on the machine image.
4. THE Clean_Step SHALL pass Splat_Transform the option that keeps only `sog_sh_bands` bands of higher spherical harmonics (`-H <sog_sh_bands>`, the short form of `--filter-harmonics`), where `sog_sh_bands` is a Preset_Setting.
5. WHERE `sog_sh_bands` is above the Source_Splat's SH_Bands, THE Clean_Step SHALL pass the Source_Splat's SH_Bands instead and SHALL add a warning to the Clean_Report naming both numbers, rather than fail.
6. THE Clean_Step SHALL pass Splat_Transform the spherical harmonic compression iterations (`-i <sh_iterations>`) and the worker count (`--max-workers <max_workers>`), where `sh_iterations` is a Preset_Setting and `max_workers` is a Shared_Setting.
7. THE Clean_Step SHALL pass Splat_Transform no option that needs a GPU, so not `--filter-floaters`, `--filter-cluster`, `.voxel.json` output or `.webp` output, which the Test_Suite checks by asserting on the argument list the Clean_Step builds (Requirement 10, criterion 9).
8. THE Clean_Step SHALL run Splat_Transform with no shell, as an argument list, so a file name can't be read as a command.
9. THE Clean_Step SHALL have Splat_Transform write into the Work_Folder, and SHALL move the result into `out/` only once Splat_Transform has succeeded and the file exists, so a failed run leaves no half-written Viewer_Copy in `out/`.
10. IF Splat_Transform_Missing, THEN THE Clean_Step SHALL fail with an Expected_Failure whose Readable_Error says which of `node` and `bin/cli.mjs` is missing, says to run `npm ci` at the repo root, and names the Node version Splat_Transform needs (a Format_Constant of the tool, not a tunable).
11. IF Splat_Transform exits with a non-zero code, THEN THE Clean_Step SHALL fail with an Expected_Failure whose Readable_Error holds Splat_Transform's own last lines of output, trimmed to at most `TAIL_CHARS` characters, which is 2000 by Requirement 6, criterion 5, with no traceback.
12. IF Splat_Transform hasn't finished within `splat_transform_timeout_seconds`, THEN THE Clean_Step SHALL stop Splat_Transform and fail with an Expected_Failure naming the limit, where `splat_transform_timeout_seconds` is a Shared_Setting.
13. IF Splat_Transform exits with 0 but the Viewer_Copy file Splat_Transform was told to write doesn't exist or is empty, THEN THE Clean_Step SHALL fail with an Expected_Failure saying so.
14. THE Clean_Step SHALL record both Step_Outputs' sizes in bytes in the Clean_Report.
15. THE Clean_Step SHALL write the Viewer_Copy only inside the Job_Folder, which Requirement 6, criterion 9 secures by making `clean_sog` a Plain_File_Name.

### Requirement 6: Settings, presets and dependencies

**User Story:** As a team member, I want every threshold of this step in Config with three Presets, so that we can tune the cleaning and the viewer copy without changing code.

**Assumptions to confirm at review:** A1 to A5 (criterion 4), A7 (criteria 4, 7 and 17), A8 (criteria 12 and 13) and A9 (criterion 14).

#### Acceptance Criteria

1. THE Config SHALL hold a `[clean]` Step_Section with the three Preset tables `[clean.fast]`, `[clean.default]` and `[clean.quality]`, each with the same keys.
2. THE Shared_Settings SHALL be these seventeen: `train_ply`, `clean_ply`, `clean_sog`, `min_seen_frames`, `min_opacity`, `faint_cube_divisions`, `faint_min_per_cube`, `faint_min_paint_share`, `extent_trim_percent`, `neighbours`, `isolation_ratio`, `max_removed_fraction`, `min_kept_splats`, `warn_kept_fraction`, `max_source_splats`, `max_workers` and `splat_transform_timeout_seconds`.
3. THE Preset_Settings SHALL be `sog_sh_bands` and `sh_iterations`, so all three Presets clean the splat identically and only the Viewer_Copy differs between them.
4. THE Config SHALL set `train_ply` to `"train.ply"`, `clean_ply` to `"clean.ply"`, `clean_sog` to `"clean.sog"`, `min_seen_frames` to 2, `min_opacity` to 0.005, `faint_cube_divisions` to 128, `faint_min_per_cube` to 8, `faint_min_paint_share` to 0.10, `extent_trim_percent` to 1, `neighbours` to 8, `isolation_ratio` to 8.0, `max_removed_fraction` to 0.70, `min_kept_splats` to 10 000, `warn_kept_fraction` to 0.5, `max_source_splats` to 8 000 000, `max_workers` to 4 and `splat_transform_timeout_seconds` to 3600, and SHALL set `sog_sh_bands` to 0 in all three Presets, so every Viewer_Copy skips Splat_Transform's spherical harmonic compression, which under `-g cpu` ran past `splat_transform_timeout_seconds` on the first real job's 912,131 Gaussians, while the Clean_Splat keeps every band, and `sh_iterations` to 10 in `[clean.fast]`, 10 in `[clean.default]` and 20 in `[clean.quality]` (A1 to A5, A7).
5. THE Clean_Step SHALL take every threshold, count, size, fraction, worker count and time limit from Config, and SHALL hold in code only the Format_Constants and `TAIL_CHARS`, which is 2000, the most characters of Splat_Transform's output that the Readable_Error of Requirement 5, criterion 11 holds. `TAIL_CHARS` stays in code because it shapes a message and changes no result.
6. IF a setting is missing from `[clean]` and its Preset table, THEN THE Step_Runner's existing check SHALL fail the Step with the Step_Runner's Readable_Error, which already names the section, the table and the key, with no new code in this step.
7. IF a setting that must be a whole number is present but isn't a whole number in its range, THEN THE Clean_Step SHALL fail with an Expected_Failure naming the setting, its section and what the setting must be, where those settings and ranges are: `min_kept_splats`, `max_source_splats`, `neighbours`, `faint_cube_divisions`, `faint_min_per_cube` and `sh_iterations` whole numbers above 0; `sog_sh_bands` a whole number from 0 to 3; and `max_workers` a whole number from 0 to 8, where 0 runs Splat_Transform's encoding inline (A7).
8. IF a setting that must be a number is present but isn't a number in its range, THEN THE Clean_Step SHALL fail with an Expected_Failure naming the setting, its section and what the setting must be, where those settings and ranges are: `min_opacity`, `warn_kept_fraction` and `max_removed_fraction` numbers from 0 to 1; `faint_min_paint_share` a number from 0 up to but not including 1; `extent_trim_percent` a number from 0 up to but not including 50; and `isolation_ratio` and `splat_transform_timeout_seconds` numbers above 0.
9. IF `train_ply`, `clean_ply` or `clean_sog` isn't a Plain_File_Name, or `clean_ply` doesn't end in `.ply` or `clean_sog` doesn't end in `.sog`, THEN THE Clean_Step SHALL fail with an Expected_Failure naming the setting and the value's problem, before the Clean_Step opens, writes or removes anything, so a setting can't reach a path outside `out/`. Both endings are exact and in lower case, because Splat_Transform picks its reader and writer from a file's extension.
10. IF any two of `train_ply`, `clean_ply` and `clean_sog` are the same name when compared ignoring case, THEN THE Clean_Step SHALL fail with an Expected_Failure naming them, so the Clean_Step can't overwrite the trained splat or its own other Step_Output, because names that differ only in case are one file on Windows.
11. THE `pyproject.toml` SHALL gain numpy and scipy as runtime dependencies, and SHALL gain no GPU package, so `uv sync` on a laptop still installs no torch, no gsplat and no CUDA wheel.
12. THE repo root SHALL gain a `package.json` and a `package-lock.json` that pin Splat_Transform 3.8.0, installed with `npm ci` (A8).
13. THE Licence_List SHALL gain or update a line for every package in `package-lock.json` whose licence isn't already covered by the Splat_Transform line, with any bundled library noted, in the same task that adds the lockfile. That task SHALL check every package's licence against the permissive list in `.kiro/steering/rules.md`, naming Splat_Transform 3.8.0's own dependencies `webgpu` and `@adobe/spz` and its peer dependency `playcanvas`, and noting the Dawn native binaries that `webgpu` ships as bundled libraries, and SHALL stop for the user IF any package has a copyleft, non-commercial or unclear licence, or any other licence that isn't on that list (A8).
14. THE Test_Suite SHALL check that `[clean]` loads from the real Config for each of the three Presets, and that its keys are exactly those of criteria 2 and 3, because `STEP_LIST` is empty by A9 and so the Step_List check in `tests/test_imports.py` doesn't cover this section.
15. THE Test_Suite SHALL check that no file under `splatpipe/` imports torch, gsplat or another GPU package, because `tests/test_imports.py` checks only the AWS names across every file under `splatpipe/` and checks the standard-library-only rule with `glob`, which leaves a step folder like `splatpipe/clean/` out.
16. THE `tests/conftest.py` test config SHALL gain a `[clean]` section with its three Preset tables, so a test that runs the Clean_Step through its CLI or its `run()` without passing a Config of its own gets settings, and the tests SHALL be able to change any one of those settings for one test.
17. THE Config's `max_source_splats` SHALL be at most 8 064 516, the number of Gaussians whose Splat_Data fits in a quarter of 8 GB (2 × 10⁹ bytes) at 62 float32 properties (248 bytes) per Gaussian, and THE Test_Suite SHALL check the real Config's value against that limit (A7).
18. THE Config SHALL give every setting in `[clean]`, `[clean.fast]`, `[clean.default]` and `[clean.quality]` a comment, on the line directly above the setting, that says what the setting does, and THE Test_Suite SHALL check, by reading `config.toml` as text, that each of those settings has one.

### Requirement 7: The step report

**User Story:** As a team member, I want the step's report to say how much each filter threw away and how big the two files are, so that I can tell a good clean from one that ate the scene.

#### Acceptance Criteria

1. WHEN the Clean_Step is done, THE Clean_Report SHALL hold the counts `source_splats`, `dropped_broken`, `dropped_faint`, `dropped_isolated`, `kept_splats`, `clean_ply_bytes` and `clean_sog_bytes`, and the timings `read_ply`, `clean`, `write_ply` and `export_sog`.
2. WHEN the Clean_Step fails, THE Clean_Report SHALL hold exactly those counts and timings of criterion 1 that the Clean_Step had already worked out before the failure, and no others, so a failed Clean_Report says how far the Clean_Step got.
3. THE Clean_Step SHALL add each count as a whole number and each timing as a number of seconds, so the Step_Runner's checks on report values never refuse one and never fail the Step as a bug.
4. THE Clean_Report SHALL hold no `removed_percent` count, because `step.add_count` refuses a float; the Removed_Fraction lives in the warning text of Requirement 3, criterion 6 and in the Readable_Error of Requirement 3, criterion 3.
5. WHERE the Clean_Step warns under Requirement 3, criterion 6 or Requirement 5, criterion 5, THE Clean_Report SHALL hold that warning.
6. THE Clean_Report SHALL be the Step_Runner's ordinary Step_Report, written to `out/reports/clean.json`, with no change to the Step_Runner.
7. THE Clean_Report's `settings` SHALL be the eighteen settings of Requirement 6, criteria 2 and 3, which the Step_Runner puts there itself.

### Requirement 8: Running the step

**User Story:** As a team member, I want to run the clean step on its own on one Job, so that I can tune the cleaning without training again.

**Assumptions to confirm at review:** A7 (criterion 9) and A9 (criteria 3 and 4).

#### Acceptance Criteria

1. THE Clean_Step SHALL be a package `splatpipe/clean/` built like the Fake_Step: `work(step)`, `run(job_id, preset=None, *, config=None)` and a `__main__.py` whose `main(argv)` calls the Skeleton's `step_main`.
2. THE CLI `uv run python -m splatpipe.clean --job <id> [--preset P]` SHALL run the Clean_Step on one Job, print `clean: done` on stdout and exit with 0 when the Step is done, and print the Readable_Error on stderr with no traceback and exit with 1 when the Step failed.
3. THIS spec SHALL leave `STEP_LIST` empty and SHALL change no pipeline-core test, so `tests/test_run.py` keeps its `assert STEP_LIST == ()` and its three assertions on the Job_Runner's "No steps are set up yet" refusal (A9).
4. WHERE a later spec adds the Clean_Step's Step_List entry, THE entry SHALL be optional and SHALL sit after the training step's, so later specs insert their entries before it (A9).
5. THE Clean_Step SHALL import only the standard library, numpy, scipy and `splatpipe`, and SHALL import nothing from `cloud/`, boto3, torch or gsplat, which Requirement 6, criterion 15 and the existing AWS check in `tests/test_imports.py` together check.
6. WHEN the Clean_Step runs again on a Job it already ran on, THE Clean_Step SHALL replace both Step_Outputs, by Requirement 4, criterion 8.
7. THE Clean_Step SHALL write every in-between file into the Work_Folder, which the Step_Runner empties before each run, and SHALL write nothing anywhere else in the Job_Folder but its two Step_Outputs.
8. THE Clean_Step SHALL raise `ExpectedFailure` for every failure Requirements 1, 3, 4, 5 and 6 name, apart from a missing setting, which Requirement 6, criterion 6 leaves to the Step_Runner's own check, so the Clean_Report's `error.expected` is `true` and its `readable` is the Clean_Step's own message.
9. THE Clean_Step SHALL do all of its work on the CPU, within the 8 vCPUs and 8 GB of memory a Step gets on the Job's cloud machine, which `max_workers` at no more than 8 and `max_source_splats` hold it to (A7).

### Requirement 9: Docs and licences

**User Story:** As one of the three people building the steps, I want this step written up where the others are, so that the steps we build separately keep fitting together.

**Assumptions to confirm at review:** A1 and A11 (criterion 4), A7 and A8 (criterion 6) and A9 (criteria 2 and 5).

#### Acceptance Criteria

1. THE Pipeline_Doc SHALL gain a `## Step: clean` section with the same parts as the Fake_Step's, including a `### What it reads`, a `### What it writes`, a `### Config section`, a `### Step report fields` and a `### Files in out/` part, which are the headings `tests/test_docs.py` looks for.
2. THE Test_Suite SHALL check criterion 1's headings for this section by name, because `STEP_LIST` is empty by A9 and so `tests/test_docs.py`, which walks `("fake", *STEP_LIST)`, doesn't reach this section (A9).
3. THE Pipeline_Doc's `### Files in out/` part for this step SHALL name both Step_Outputs and SHALL state the guarantees of Requirement 4, criteria 8 to 13: that the Clean_Step removes both before it writes either; that a failure after that removal leaves neither; that a hard stop can leave a whole Clean_Splat alone, this run's between the two moves into `out/` or the earlier one between the two removals, with no done Clean_Report, and that the next run removes it first; that a Viewer_Copy never sits in `out/` without the Clean_Splat it was made from; and that the pair is to be trusted only when `out/reports/clean.json` says done.
4. THE Pipeline_Doc SHALL state the contract this step relies on: that the training step writes the Source_Splat as a standard 3DGS Binary_PLY at `out/<train_ply>` with the properties A1 lists, that the Clean_Step carries every other property through unchanged, a NaN or an infinity in a colour included, and that the Clean_Step never writes that file (A1, A11).
5. THE Pipeline_Doc's "The step list" section SHALL say that this step's entry isn't in `STEP_LIST` yet and which spec adds it (A9).
6. THE Pipeline_Doc's `## Step: clean` section SHALL say that every part of the Clean_Step is CPU work needing 8 vCPUs and 8 GB, and that the Viewer_Copy is SOG meta version 2 because Splat_Transform is pinned at 3.8.0, so the web viewer has to be read against the version 2 spec (A7, A8).
7. THE Licence_List SHALL mark numpy, scipy, Splat_Transform 3.8.0 and npm as installed, with the pinned Splat_Transform version, and SHALL gain a line for Node itself worded like the other command line tools' lines: a separate tool on our machines, never vendored, linked or shipped.
8. THE `README.md` SHALL say that this step needs `npm ci` at the repo root and Node 22 or newer, next to how the rest is installed.

### Requirement 10: Tests

**User Story:** As a team member, I want this step tested on a laptop CPU with tiny files and no network, so that the tests are quick, free and safe to run anywhere.

**Assumptions to confirm at review:** A10 (criterion 13).

#### Acceptance Criteria

1. THE Test_Suite SHALL test the PLY reader and writer, each of the three filters, the Requirement 3 limits, the Config checks, the Clean_Report fields and the CLI, on tiny synthetic splats the tests build themselves, with no network and no GPU.
2. THE Test_Suite's unit tests and property tests SHALL both be required, never optional.
3. THE Test_Suite SHALL build its synthetic splats as bytes in the test, so no splat file is committed to git and nothing is written into the real `data/`.
4. THE Test_Suite SHALL cover every Expected_Failure of Requirements 1, 3, 4, 5 and 6, and SHALL check that each Readable_Error names what the requirement says it names and holds no traceback.
5. THE Test_Suite SHALL use property tests with Hypothesis for the properties marked in Requirement 2, criteria 18 to 23 and Requirement 4, criterion 4.
6. THE Test_Suite SHALL skip the tests that actually run Splat_Transform when `node` isn't on the PATH or the installed `bin/cli.mjs` isn't there, with a reason that says which is missing, and SHALL test the Clean_Step's handling of Splat_Transform's failures (Requirement 5, criteria 10 to 13) with a stand-in instead, so the whole Test_Suite passes on a machine with no Node.
7. THE Test_Suite SHALL run Splat_Transform on nothing but a tiny synthetic splat, and SHALL give every subprocess a timeout.
8. THE Test_Suite SHALL check that a done run and a failed run both leave the Source_Splat's bytes unchanged (Requirement 1, criterion 11).
9. THE Test_Suite SHALL check the argument list the Clean_Step builds for Splat_Transform: that the list starts with `node` and the installed `bin/cli.mjs`, and holds `-g cpu`, `-H` (the short form of `--filter-harmonics`) with the Preset's band count, `-i` with the Preset's iterations and `--max-workers` with the Shared_Setting, and none of the GPU-only options of Requirement 5, criterion 7.
10. THE Test_Suite SHALL check that a run which fails at the Viewer_Copy, and a run whose move of the Viewer_Copy into `out/` fails, each leave neither Step_Output in `out/`, a pair from an earlier run included, and that a second done run replaces both (Requirement 4, criteria 8, 10 and 11 and Requirement 8, criterion 6).
11. THE Test_Suite SHALL check that the Clean_Step writes nothing in the Job_Folder outside the Work_Folder, its two Step_Outputs and its Step_Report, by listing the Job_Folder before and after a run.
12. THE Test_Suite SHALL check the carry-through of Requirement 1, criterion 4 and Requirement 4, criteria 1 and 3 on a synthetic splat that holds `nx`, `ny`, `nz`, a non-standard property and a header `comment` line, by reading the Clean_Splat back and comparing its properties, their order, their types, their values and the comment with the Source_Splat's.
13. THE Test_Suite's property tests whose result depends on the Isolated_Filter's neighbour search SHALL avoid Splat_Data in which more than one other Gaussian is at the Centre_Distance of a Gaussian's `neighbours`-th nearest, apart from the dense-grid property of Requirement 2, criterion 22, which SHALL be tested on grids full of such ties and SHALL hold whatever the KD-tree's order (A10). Property tests of the Broken_Filter or the Faint_Filter alone may draw such ties, because neither filter searches for neighbours.
