# Requirements Document

## Introduction

This spec builds the Walls_Step and changes the Train_Step so it can use what the Walls_Step writes. Plain walls have almost no texture, so COLMAP finds few points on them, and MoGe-2's depth there is only roughly right, so the splat comes out bumpy. The Walls_Step corrects each frame's depth prior with Kriegman and Belhumeur's perspective bas-relief family (ECCV 1998): the changes 1/z′ = a/z + b·x + c·y + d, with x and y the normalised image coordinates. Every change of that family keeps each pixel on its own ray and keeps every plane flat, so per frame the step only searches four numbers. It scores each guess with the faint texture a wall still has, like paint stipple or weave. In one window that texture is far below the noise, but summed over a whole frame and many other frames it decides.

The Walls_Step is an Optional_Step between the Depth_Step and the Train_Step: `StepEntry("walls", "splatpipe.walls", False)`, so the order becomes frames, cameras, depth, walls, train, clean. It reads the Depth_Step's priors in `work/depth/` and the camera solve in `work/cameras/` with the Train_Step's Model_Reader, and writes its own priors to `work/walls/` in exactly the Prior_Format, at the Depth_Step's Prior_Size, leaving the depth priors as they are. The Train_Step reads `work/walls/` when the Walls_Step is done, and `work/depth/` otherwise, and nothing else in training changes. When the Walls_Step fails, the Job goes on and trains on the depth priors. The Walls_Step uses the Job_Folder, the Step_Runner, the Step_Report and the Config of the pipeline-core spec, and it's built like the Fake_Step and the Depth_Step. On the cloud side it follows the cloud-jobs spec: its Step_Needs go in `[walls]`, which `cloud/select.py` already reads for every Step.

The Walls_Step runs on the CPU only, with numpy, scipy and OpenCV, which are already in `uv.lock`, so it adds no package and the Machine_Image stays as it is. It uses every vCPU the machine has, through worker processes. The user's estimate is about 30 minutes for 190 frames on 32 vCPUs, and about twice that on 16. So the step reads the Cap_Deadline as the Train_Step does and stops at a Preset's `max_minutes`, or earlier when the Train_Step would no longer get its time. Frames it didn't finish keep their depth prior.

The method, in the order the step runs it. Each part has its requirement below, and every number is in `[walls]`:

- Each change is written relative to the current depth, 1/z′ = a/z + m·(b·x + c·y + d), with m the frame's median inverse depth, so the change that changes nothing is (1, 0, 0, 0).
- The scene scale k from the frame's sparse points, with a 3-MAD gate, gives the start depth, k × the depth prior.
- Up to 32 source frames per frame, chosen by time, spacing, overlap and parallax.
- A score table per pass: zero-mean NCC of high-passed luma over 31 × 31 windows every 4 pixels, summed over the sources, so any guess of the four numbers scores by table lookup.
- Three passes: one scale from ×0.5 to ×2, then a first fit of the four numbers, then a last fit on a mix of luma bands weighted by how well each band agrees across frames.
- A self-check with no ground truth: the sources split into two halves that must agree, and a tile score Z.
- Planar facets of each frame, each refitted on its own and kept only when it passes its own checks.
- Cross-frame work: world planes, the fuse, a fallback depth for frames that failed their self-check, the correction, the flatten, the despeckle and the per-frame choice.
- A side-by-side JPEG of 3 frames for the demo, and a `## Step: walls` section in the Pipeline_Doc that explains the method in our own words and cites the paper.

Out of scope:

- Running anything on AWS. No task starts a Job, an Image_Build or a Smoke_Test, and no task sets `APPROVED_SPEND`. Once the tasks are done, Samarth runs a real Job on the Walls_Step himself.
- Any change to the Machine_Image, `cloud/image.py`, the Smoke_Test's checks, `cloud/select.py`, the Start_Command, the Cloud_Runner or the Website's code.
- Any change to the Depth_Step, its files, the Prior_Format's fields, or the Train_Step's losses, Start_Splats and counts.
- The Preset's Time_Caps and the Website's spend limits, which stay as they are.
- Tuning the new numbers on real footage, and measuring the Train_Step's real times.

### Task budget

The whole spec stays within 10 tasks, which the design and tasks phases plan. Each task writes its own unit tests and property tests, both required, never optional. If the design can't fit in 10 tasks, it stops and says so before the tasks are written (D23). Once the requirements, the design and the tasks are written, the three files are committed by path and `feat/walls` is pushed, and no task starts until the user says go.

### Decisions made before this draft

The user chose these when asked, before the requirements were written.

- **D1 (Requirements 15 and 17): Failed frames and the per-frame choice.** A Prior_Frame that fails its Self_Check gets a Fallback_Depth: projected from the Trusted_Frames, or else registered to the World_Planes. Only a Prior_Frame whose Walls_Depth came from a Fallback_Depth and whose Sparse_Error is above `fallback_max_sparse_error` (0.1) drops to its Depth_Prior. This replaces the request's sentence that a frame failing its self-check trains on its depth prior.
- **D2 (Requirement 17): The depth prior's version.** Every Prior_Frame that keeps its Depth_Prior gets an unchanged copy of all three arrays of its Depth_Prior's Prior_File.
- **D3 (Requirement 17): The mask.** Each Walls_Prior keeps exactly its Depth_Prior's `mask`, so no prior pixel is added or dropped. A pixel whose new inverse depth isn't finite and above 0 keeps its Depth_Prior's value.
- **D4 (Requirements 6, 16, 17 and 18): The normals.** The `normal` array holds MoGe's normal carried exactly through the Prior_Frame's or the Facet's Bas_Relief_Change: for a pixel at depth z with Ray r and normal n, the new normal is proportional to a·n + m·z·(n·r)·(b, c, d), set to unit length and turned to face the camera. Flattened pixels take their World_Plane's normal, or their Facet's own robust normal. In a Walls_Demo, both normal images are worked out from each depth map by finite differences, for display only.
- **D5 (Requirements 18 and 20): How training picks its priors.** The Train_Step treats the Walls_Step as done when `work/walls/depth.json` exists, which the Walls_Step writes last of all, so a failed run leaves none. If those priors fail the Train_Step's checks, the Train_Step adds a warning that says it fell back, and reads `work/depth/` as it does now. No new Train_Report count, and nothing else in training changes.
- **D6 (Requirements 2 and 8): Only frames with a depth prior.** Only Prior_Frames are solved or used as Source_Frames. A Frame with no Depth_Prior gets no Walls_Prior, as in the Depth_Step.
- **D7 (Requirement 18): The index and the record.** The Walls_Index copies the Depth_Index unchanged: its camera, its Prior_Size, its `metres_per_unit` and every entry with its fit fields. The per-frame walls numbers (k, a, b, c, d, the Halves_Difference, the Tile_Scores and the outcome) go in the Walls_Record, `work/walls/walls.json`.
- **D8 (Requirements 2, 8 and 14): Frame times and pick numbers.** A Prior_Frame's Frame_Time is its Pick's `seconds` in `work/frames/frames.json`, matched by the Pick's `file` to the image's name. "2 frames apart" and "within 24 frames" count Pick_Numbers.
- **D9 (Requirements 9 and 12 to 16): Sizes.** The scoring runs on Luma at the Score_Size, 640 pixels on the long edge. The Facets, the fuse, the fallback, the correction, the flatten and the despeckle run at the Prior_Size, so 4,000 pixels, 2 pixels and 5 × 5 are Prior_Size pixels.
- **D10 (Requirements 2 and 9): Masked windows.** A window that touches a masked pixel, from the camera solve's `work/cameras/masks/<stem>.png` where black means ignore, in the Prior_Frame or in the warped Source_Frame, is skipped.
- **D11 (Requirements 9 and 10): Guesses outside the table.** A guess of the four numbers whose Depth_Factor leaves the pass's Table_Range at any Score_Window is rejected.
- **D12 (Requirement 11): The halves.** The ranked Source_Frames split alternately: the 1st, 3rd, … in Half_A and the 2nd, 4th, … in Half_B. Each half refits only the Third_Pass's four numbers, on its half of that pass's Score_Table. The halves agree when the median over the Prior_Frame's Covered_Pixels of |z_A / z_B − 1| is under 1.5%.
- **D13 (Requirements 11 and 12): Z twice.** The Tile_Score is worked out against ×1.2 and against ÷1.2, and both must be above 2.5. For a Facet, both must be at least 3.
- **D14 (Requirements 8, 11, 14 and 15): Neighbours, overlap and too few sources.** The Trusted_Frames that the fuse, the fallback and the correction use are the Neighbour_Frames: Trusted_Frames within 24 Pick_Numbers, the nearest 12 first, never the Prior_Frame itself. The projection fallback has enough overlap when at least 30% of the Prior_Frame's Covered_Pixels get an agreeing projected value. A Prior_Frame with fewer than 4 Source_Frames, even after dropping to 0.5 s, fails its Self_Check.
- **D15 (Requirements 1 and 13 to 17): The order of the cross-frame work.** The fuse (Trusted_Frames), then the World_Planes, then the fallback (Failed_Frames), then the correction (every Prior_Frame with a Walls_Depth), then the flatten, then the despeckle, then the per-frame choice. The World_Planes are built once, from the Trusted_Frames' Facets, and both the fallback and the flatten use them.
- **D16 (Requirements 3 and 21): Soft vCPUs.** `min_vcpus` stays a soft need, with no change to the cloud code, and `max_minutes` cuts off a slow machine. A `fast` Job can land on 8 vCPUs, and a `quality` Job runs on 16 or 8.
- **D17 (Requirement 4): A spread-out order.** The Prior_Frames are solved in the Solve_Order: first, last, middle, then the quarters, and so on, so a run that's cut off still covers the whole video.
- **D18 (Requirements 3, 4 and 21): The time budget.** Each Preset gets `train_reserve_minutes` in `[walls]`: 40 for `fast`, 60 for `default` and 110 for `quality`, chosen with no measured training times yet. The Walls_Stop_Time is the earlier of the step's start + `max_minutes` and the Cap_Deadline − `train_reserve_minutes`. No Frame_Solve starts at or after the Solve_Cutoff, `finish_minutes` (3) before the Walls_Stop_Time, and the Frame_Solves still running then give up, so their Prior_Frames keep their Depth_Prior and count as timed out.
- **D19 (Requirement 21): Expected times.** `expected_minutes` in `[cloud.presets.*]` rises from 30 to 50 for `fast`, from 60 to 100 for `default` and from 120 to 180 for `quality`. `cap_minutes` stays as it is, so the Website's reservations don't change. The Walls_Step adds roughly $0.80 to a `fast` Job, $1.60 to a `default` Job and $3.00 to a `quality` Job. The Website shows the new times after the account owner redeploys.
- **D20 (Requirement 21): No smoke check.** The Smoke_Test gains no check, because the Walls_Step is CPU-only numpy, scipy and OpenCV code that the Test_Suite tests in full on a laptop.
- **D21 (Requirement 19): The report's counts.** The Walls_Report counts `images` (the Prior_Frames), `solved_images`, `passed_images`, `fallback_images`, `depth_prior_images` and `timed_out_images`. A Timed_Out_Frame keeps its Depth_Prior and gets no fallback or correction.
- **D22 (Requirement 3): Numbers the request doesn't give.** These go in `[walls]` as defaults, each with a comment that says it's untested on real footage: the Change_Fit's start spread, iteration cap and tolerances; a cap on the Coherence (0.95); the fewest windows a Tile needs; Tukey's constant and iteration count; the fewest Scale_Points for k, below which k = 1; the fewest matched pixels for the Plane_Registration; and the Walls_Demo count (3). Requirement 3 adds the few more that every number in Config needs (A16 to A18).
- **D23 (Task budget): One spec.** One spec of at most 10 top-level tasks.

### Decisions made at review

The user reviewed the draft and chose this.

- **D24 (Requirements 4, 9, 11 and 12): The occlusion test uses the Source_Frame's Start_Depth.** The occlusion test compares the depth the warp predicts in a Source_Frame with that Source_Frame's Start_Depth, its Scene_Scale k × its Prior_Depth, which the Walls_Step already works out for every Prior_Frame (Requirement 7), and not with its raw Prior_Depth. That holds for the passes' test with `occlusion_tolerance` (15%), the Facet refit's looser test with `facet_occlusion_tolerance` (50%) and the Tile_Score's test made at the answer. A window centre that lands where the Source_Frame has no Start_Depth still counts as occluded.

### Assumptions to confirm at review

The user didn't decide these. Each is the recommended answer, marked in the requirement it belongs to, and each is a small change if the review decides otherwise.

- **A1 (Requirement 9): When the occlusion test is made.** A window is occluded for a Source_Frame when, at the window's centre, the depth the warp predicts in the Source_Frame and the Source_Frame's own Start_Depth there differ by more than `occlusion_tolerance` (15%) (D24). The test is made once per pass, at the pass's Current_Depth, and the same Valid_Pairs then hold at every Table_Scale. A Facet's refit makes it at the Solved_Depth with `facet_occlusion_tolerance` (50%), and the Tile_Score uses the test made at the answer, both against the Source_Frame's Start_Depth too. The reason: a test made at each Table_Scale would skip every window whose scaled depth is more than 15% from the Source_Frame's Start_Depth, which would shrink the First_Pass's ×0.5 to ×2 to about ±15% and leave the Tile_Score's ×1.2, 20% away, with no window at all. A window centre that lands where the Source_Frame has no Start_Depth counts as occluded.
- **A2 (Requirement 9): Score_Windows and Valid_Pairs.** Window centres lie on a grid every `window_step` Score_Size pixels, starting half a window in from the top left, so each window lies inside the image. A Score_Window has a Start_Depth at every one of its pixels, touches no masked pixel and isn't flat in the Prior_Frame. A Source_Frame's pair with it is a Valid_Pair at a Depth_Factor when every warped pixel lands in front of the Source_Frame's camera and inside its image, touches no masked pixel of the Source_Frame, the warped window isn't flat and it isn't occluded. The flat test always uses the High_Pass, in every pass, so the same windows are flat in all three passes.
- **A3 (Requirement 9): The Table_Scales.** A pass's Table_Scales run from its minimum to its maximum scale, both included, evenly spaced in ln(s), with n = ⌈ln(max / min) / step⌉ + 1 scales, so neighbouring scales are at most `step` apart in ln(s). A Guess_Score looks up C(p, s) by linear interpolation in ln(s) between the two neighbouring Table_Scales.
- **A4 (Requirement 10): The Band_Weights.** A Band's Coherence is capped at `band_max_coherence`. A Band whose Coherence isn't above 0, or whose variance is 0, gets a weight of 0. A Band's variance is taken over the Prior_Frame's Depth_Pixels that no mask hides. When every weight is 0, the Third_Pass scores the High_Pass instead of the Band_Mix. The Source_Frames' Band_Mixes use the Prior_Frame's own weights.
- **A5 (Requirements 10 to 12): One fit for every table.** Every Change_Fit, in the Second_Pass, the Third_Pass, the Source_Halves and the Facets, uses the same Nelder-Mead settings and the same `fit_bound` on |b|, |c| and |d|, from `fit_starts` starts: the Identity_Change and `fit_starts` − 1 others spread by `fit_start_spread`. A Facet's change takes m from the Facet's own Median_Inverse_Depth. The Third_Pass's tables, the Half_Tables, the Tile_Scores and the Facets' tables all score the Band_Mix.
- **A6 (Requirement 11): The Tile_Score.** For each of ×`z_scale` and ÷`z_scale`, a Tile counts when at least `tile_min_windows` of its Score_Windows have a Valid_Pair at both the answer and the other depth. The Tile's two means are over the same Valid_Pairs, and Z = (mean_A − mean_B) / (`z_error_factor` × √(var_A / n + var_B / n)) over the n counting Tiles, with sample variances. With fewer than 2 counting Tiles there's no Z and the Self_Check fails. A standard error of 0 gives Z = +∞ when mean_A − mean_B is above 0, and a failed Self_Check otherwise.
- **A7 (Requirement 7): The Scene_Scale.** The 3-MAD gate uses the median absolute deviation as it is, not scaled to a standard deviation. With a deviation of 0, only the Scale_Points at the median stay. The Scale_Points follow the Depth_Step's Fit_Point rule, on the Depth_Prior's Covered_Pixels.
- **A8 (Requirement 8): Overlap and motion.** The Overlap is taken over every Covered_Pixel of the Prior_Frame at its Start_Depth. The Image_Motion is measured in the Source_Frame's Score_Size pixels, as the median over the Covered_Pixels that land inside it. Source_Frames are chosen greedily in Source_Rank order, and ties go to the lower Pick_Number.
- **A9 (Requirement 12): Facets.** The Facets are cut from the Depth_Prior's own `normal` and `inverse_depth`, are 4-connected, and leave out every Jump_Pixel. Only a Trusted_Frame's Facets are refitted, inside its Frame_Solve, from the Score_Windows whose centres fall on the Facet.
- **A10 (Requirement 13): World_Planes.** The World_Planes are built after the fuse, from the Trusted_Frames' Fused_Depths. Facets join World_Planes greedily, the largest Facet first. A Facet's camera distance is the distance from its Prior_Frame's Camera_Centre to its Facet_Plane. The 1% rule counts against the largest World_Plane that has Facets from `plane_min_frames` or more Prior_Frames.
- **A11 (Requirements 14 to 16): The cross-frame stages.** Each stage reads the other Prior_Frames' depths as the stage before it left them, so the order the Prior_Frames go through a stage doesn't change its result. Neighbour_Frames at the same Pick_Number distance go lower Pick_Number first. A Projected_Value lands on a Prior_Size pixel by the Depth_Step's pixel rule, and each Neighbour_Frame gives at most one value per pixel, its nearest.
- **A12 (Requirement 15): The fallback's fits.** The Tukey fits of the four numbers are made in inverse depth, with no `fit_bound`, because they fit geometry rather than search a table. The Plane_Registration's tolerance falls geometrically from `register_start_tolerance` to `register_end_tolerance` over `register_passes` passes. A pixel matches the World_Plane nearest its current depth among those within the angle and the tolerance. A Failed_Frame that gets neither fallback keeps its Depth_Prior.
- **A13 (Requirements 16 and 17): Normals outside the flatten.** The fuse, the projected values, the correction and the despeckle change depths only. A pixel they change keeps the normal that its Prior_Frame's or Facet's change gave it. A pixel whose new normal isn't finite, or has a length of 0, keeps its Depth_Prior's normal.
- **A14 (Requirement 17): The sparse check.** The per-frame choice uses the Scale_Points, at the final Walls_Depth. A Prior_Frame with fewer than `scale_min_points` Scale_Points isn't off its points, so it keeps its Fallback_Depth.
- **A15 (Requirements 4 and 18): No trusted frame, and the time after the cutoff.** When no Prior_Frame passes its Self_Check, which includes a run in which every Prior_Frame timed out, the step fails, because no Fallback_Depth can be made and every Walls_Prior would copy its Depth_Prior. The cross-frame stages and the writes run after the Solve_Cutoff with no cutoff of their own, so `finish_minutes` is the time they're meant to fit in, untested on real footage.
- **A16 (Requirement 5): Workers.** `workers = 0` means every vCPU the process may use, and a whole number above 0 sets the count, never more than the Prior_Frames. Each worker runs numpy, scipy and OpenCV on one thread. A worker process that ends without a result fails the step. The step reaches the Worker_Pool through one module-level hook that the Test_Suite replaces with an in-process stand-in, as the Train_Step reaches its GPU_Module.
- **A17 (Requirement 18): The Walls_Demos.** They're chosen among the Prior_Frames whose Walls_Prior isn't a Depth_Prior_Copy, spread by Pick_Number. Each panel is at the Prior_Size. A normal n shows as the colour (n + 1) / 2 × 255 in R, G and B, and a pixel with no normal is black. `demo_jpeg_quality` (90) sets the JPEG quality.
- **A18 (Requirements 2 and 3): Masks.** The Walls_Step reads each Mask_File with the Train_Step's rules, and has its own `mask_threshold` (128, as in `[train]`), because a Step sees only its own section.
- **A19 (Requirements 4 and 6): Two more property tests.** Besides the property tests the user asked for, the composition of two Bas_Relief_Changes gets a property test, and the Solve_Order gets one that it lists every Prior_Frame once, as testing.md asks for every rule that should hold for every input.
- **A20 (Requirement 21): Licences.** No dependency, tool, model weight or dataset is added, so the Licence_List gains no row. The paper isn't a dependency, so it gets no row either: the Pipeline_Doc cites it. The `opencv-python-headless` row's "How we use it" cell adds the Walls_Step's use.
- **A21 (Requirement 22): The end-to-end Job test.** The Test_Suite's Job through the real Step_List, in `tests/test_cameras_step.py`, expects the Walls_Step's outcome on its tiny scene, as the design works it out, and the Job_Status that follows from it.
- **A22 (Glossary): The stop time's name.** The Walls_Step's stop time is the Walls_Stop_Time, because the splat-training spec already uses Stop_Time for the Train_Step's, and the website spec for the Website's.

### The settings

Every number of the method is a Shared_Setting in `[walls]`, except the two per-Preset settings at the end. The ones marked D22 are defaults that the comment in `config.toml` calls untested on real footage. "Whole" means a whole number given as a TOML integer, and a bool is never a number.

| Key | Table | Value | Kind | What it is |
|---|---|---|---|---|
| `score_long_edge` | `[walls]` | `640` | whole, above 0 | Long edge of the Score_Size, in pixels. A smaller camera isn't scaled up. |
| `high_pass_sigma` | `[walls]` | `3.0` | number above 0 | Sigma of the blur the High_Pass takes away, in Score_Size pixels. |
| `window_size` | `[walls]` | `31` | odd whole, at least 3 | Side of a Score_Window, in Score_Size pixels. |
| `window_step` | `[walls]` | `4` | whole, above 0 | Spacing of the Score_Windows' centres, in Score_Size pixels. |
| `flat_variance` | `[walls]` | `0.04` | number, 0 or more | A window whose High_Pass variance is under this, in grey levels squared, is flat. |
| `occlusion_tolerance` | `[walls]` | `0.15` | number above 0 | A window is occluded when the Source_Frame's Start_Depth disagrees by more than this (D24). |
| `mask_threshold` | `[walls]` | `128` | whole, 0 to 255 | A Mask_File pixel below this is masked, as in `[train]` (A18). |
| `scale_mad_gate` | `[walls]` | `3.0` | number above 0 | The Scene_Scale keeps the Scale_Points within this many median absolute deviations. |
| `scale_min_points` | `[walls]` | `10` | whole, above 0 | Fewest Scale_Points for a Scene_Scale; with fewer, k = 1 (D22). |
| `max_sources` | `[walls]` | `32` | whole, above 0 | Most Source_Frames per Prior_Frame. |
| `source_min_seconds` | `[walls]` | `1.0` | number, 0 or more | Shortest time between a Prior_Frame and its Source_Frames, in seconds. |
| `source_fallback_seconds` | `[walls]` | `0.5` | number, 0 or more, at most `source_min_seconds` | The shorter time used when the first gives too few Source_Frames. |
| `min_sources` | `[walls]` | `4` | whole, at least 2, at most `max_sources` | Fewest Source_Frames a Prior_Frame needs. |
| `source_min_gap` | `[walls]` | `2` | whole, above 0 | Fewest Pick_Numbers between two Source_Frames of one Prior_Frame. |
| `source_min_overlap` | `[walls]` | `0.3` | number from 0 to 1 | Smallest Overlap of a Source_Frame. |
| `motion_depth_change` | `[walls]` | `0.01` | number above 0 | The depth change the Image_Motion is measured for, 1%. |
| `source_min_motion` | `[walls]` | `0.4` | number, 0 or more | Smallest Image_Motion of a Source_Frame, in Score_Size pixels. |
| `source_motion_cap` | `[walls]` | `3.0` | number above 0 | The Image_Motion counts up to this in the Source_Rank, in Score_Size pixels. |
| `first_min_scale` | `[walls]` | `0.5` | number above 0, below 1 | Smallest Table_Scale of the First_Pass. |
| `first_max_scale` | `[walls]` | `2.0` | number above 1 | Largest Table_Scale of the First_Pass. |
| `first_scale_step` | `[walls]` | `0.015` | number above 0 | Largest step between the First_Pass's Table_Scales, in ln(s). |
| `second_min_scale` | `[walls]` | `0.92` | number above 0, below 1 | Smallest Table_Scale of the Second_Pass. |
| `second_max_scale` | `[walls]` | `1.087` | number above 1 | Largest Table_Scale of the Second_Pass. |
| `second_scale_step` | `[walls]` | `0.004` | number above 0 | Largest step between the Second_Pass's Table_Scales, in ln(s). |
| `third_min_scale` | `[walls]` | `0.95` | number above 0, below 1 | Smallest Table_Scale of the Third_Pass. |
| `third_max_scale` | `[walls]` | `1.053` | number above 1 | Largest Table_Scale of the Third_Pass. |
| `third_scale_step` | `[walls]` | `0.0025` | number above 0 | Largest step between the Third_Pass's Table_Scales, in ln(s). |
| `fit_starts` | `[walls]` | `5` | whole, above 0 | Nelder-Mead starts of each Change_Fit. |
| `fit_bound` | `[walls]` | `0.2` | number above 0 | Largest \|b\|, \|c\| and \|d\| of a Change_Fit's guess. |
| `fit_start_spread` | `[walls]` | `0.05` | number above 0 | How far the starts other than the Identity_Change are spread (D22). |
| `fit_max_iterations` | `[walls]` | `400` | whole, above 0 | Most Nelder-Mead iterations per start (D22). |
| `fit_x_tolerance` | `[walls]` | `0.0001` | number above 0 | Nelder-Mead's tolerance on the four numbers (D22). |
| `fit_f_tolerance` | `[walls]` | `0.001` | number above 0 | Nelder-Mead's tolerance on the Guess_Score (D22). |
| `band_sigmas` | `[walls]` | `[0, 1, 2, 4, 8]` | list of 2 or more numbers, the first 0, each above the one before | Blur sigmas of the octave Bands, in Score_Size pixels. 0 is the Luma itself. |
| `band_max_coherence` | `[walls]` | `0.95` | number above 0, below 1 | Cap on a Band's Coherence (D22). |
| `halves_tolerance` | `[walls]` | `0.015` | number above 0 | The Source_Halves agree when the Halves_Difference is under this. |
| `tile_size` | `[walls]` | `32` | whole, above 0 | Side of a Tile, in Score_Size pixels. |
| `tile_min_windows` | `[walls]` | `16` | whole, above 0 | Fewest Score_Windows a Tile needs to count in the Tile_Score (D22). |
| `z_scale` | `[walls]` | `1.2` | number above 1 | The Tile_Score compares the answer with the answer × and ÷ this. |
| `min_z` | `[walls]` | `2.5` | number | A Prior_Frame's two Tile_Scores must be above this. |
| `z_error_factor` | `[walls]` | `2.0` | number above 0 | The Tile_Score's standard error is multiplied by this. |
| `facet_normal_degrees` | `[walls]` | `6.0` | number above 0, at most 90 | A Facet's normals lie within this many degrees of its normal. |
| `facet_jump` | `[walls]` | `0.01` | number above 0 | Inverse depths that differ by more than this share make a Jump_Pixel. |
| `facet_jump_distance` | `[walls]` | `2` | whole, above 0 | The distance across or down, in Prior_Size pixels, the jump test compares over. |
| `facet_min_pixels` | `[walls]` | `4000` | whole, above 0 | Fewest Prior_Size pixels of a Facet. |
| `facet_min_scale` | `[walls]` | `0.75` | number above 0, below 1 | Smallest Table_Scale of a Facet's refit. |
| `facet_max_scale` | `[walls]` | `1.333` | number above 1 | Largest Table_Scale of a Facet's refit. |
| `facet_scale_step` | `[walls]` | `0.006` | number above 0 | Largest step between a Facet's Table_Scales, in ln(s). |
| `facet_occlusion_tolerance` | `[walls]` | `0.5` | number above 0 | The looser occlusion test of a Facet's refit. |
| `facet_halves_tolerance` | `[walls]` | `0.03` | number above 0 | A Facet's Source_Halves must agree within this. |
| `facet_min_change_ratio` | `[walls]` | `2.5` | number, 0 or more | A Facet_Change must be more than this times its Source_Halves' difference. |
| `facet_min_z` | `[walls]` | `3.0` | number | A Facet's two Tile_Scores must be at least this. |
| `plane_merge_degrees` | `[walls]` | `3.0` | number above 0, at most 90 | Facet_Planes merge when their normals are within this many degrees. |
| `plane_merge_offset` | `[walls]` | `0.015` | number above 0 | … and their offsets within this share of the camera distance. |
| `plane_min_frames` | `[walls]` | `3` | whole, above 0 | Fewest Prior_Frames whose Facets make a World_Plane. |
| `plane_min_share` | `[walls]` | `0.01` | number from 0 to 1 | Smallest share of the largest World_Plane's pixels a World_Plane needs. |
| `neighbour_frames` | `[walls]` | `12` | whole, above 0 | Most Neighbour_Frames per Prior_Frame. |
| `neighbour_window` | `[walls]` | `24` | whole, above 0 | Neighbour_Frames are within this many Pick_Numbers. |
| `fuse_tolerance` | `[walls]` | `0.1` | number above 0 | The fuse takes the Projected_Values within this share of the pixel's depth. |
| `fuse_min_values` | `[walls]` | `3` | whole, above 0 | Fewest values, the pixel's own included, the fuse takes a median of. |
| `agree_tolerance` | `[walls]` | `0.03` | number, 0 or more | Agreeing_Values are within this share of the nearest Projected_Value. |
| `agree_min_frames` | `[walls]` | `2` | whole, above 0 | Fewest Agreeing_Values for the fallback and the correction. |
| `projection_min_coverage` | `[walls]` | `0.3` | number from 0 to 1 | Smallest share of Covered_Pixels with Agreeing_Values for the Projection_Fallback. |
| `register_max_degrees` | `[walls]` | `45.0` | number above 0, at most 90 | A pixel matches a World_Plane whose normal is within this many degrees of its own. |
| `register_start_tolerance` | `[walls]` | `0.5` | number above 0 | The Plane_Registration's depth tolerance in its first pass. |
| `register_end_tolerance` | `[walls]` | `0.03` | number above 0, at most `register_start_tolerance` | The Plane_Registration's depth tolerance in its last pass. |
| `register_passes` | `[walls]` | `5` | whole, above 0 | Passes of the Plane_Registration. |
| `register_min_pixels` | `[walls]` | `1000` | whole, above 0 | Fewest matched pixels in each Plane_Registration pass (D22). |
| `tukey_constant` | `[walls]` | `4.685` | number above 0 | Tukey's constant, in median absolute deviations, for every robust fit (D22). |
| `tukey_iterations` | `[walls]` | `10` | whole, above 0 | Reweighting rounds of every robust fit (D22). |
| `correction_tolerance` | `[walls]` | `0.03` | number above 0 | The correction moves a pixel more than this off in ln(depth). |
| `flatten_degrees` | `[walls]` | `5.0` | number above 0, at most 90 | A Facet takes a World_Plane's normal within this many degrees. |
| `flatten_tolerance` | `[walls]` | `0.03` | number, 0 or more | Only pixels within this share of the Facet's plane move. |
| `despeckle_window` | `[walls]` | `5` | odd whole, at least 3 | Side of the despeckle's window, in Prior_Size pixels. |
| `despeckle_min_valid` | `[walls]` | `9` | whole, above 0, at most `despeckle_window`² | Fewest valid pixels the window needs. |
| `despeckle_tolerance` | `[walls]` | `0.03` | number, 0 or more | A pixel more than this share off its window's median takes the median. |
| `fallback_max_sparse_error` | `[walls]` | `0.1` | number above 0 | A Fallback_Depth whose Sparse_Error is above this drops to the Depth_Prior. |
| `finish_minutes` | `[walls]` | `3` | number, 0 or more, below every `max_minutes` | The Solve_Cutoff comes this many minutes before the Walls_Stop_Time. |
| `workers` | `[walls]` | `0` | whole, 0 or more | Worker processes; 0 means every vCPU (A16). |
| `demo_frames` | `[walls]` | `3` | whole, 0 or more | Walls_Demos written; 0 writes none (D22). |
| `demo_jpeg_quality` | `[walls]` | `90` | whole, 1 to 100 | JPEG quality of the Walls_Demos (A17). |
| `min_vcpus` | `[walls]` | `16` | whole, above 0 | Fewest vCPUs the Job's machine should have for this step, a soft need. The step never checks it. |
| `min_memory_gb` | `[walls]` | `32` | number above 0 | Least memory the Job's machine needs for this step, in GB, a hard need. The step never checks it. |
| `max_minutes` | `[walls.fast]`, `[walls.default]`, `[walls.quality]` | `20`, `40`, `60` | number above 0 | Longest the step solves frames for, from its start, in minutes. |
| `train_reserve_minutes` | `[walls.fast]`, `[walls.default]`, `[walls.quality]` | `40`, `60`, `110` | number, 0 or more | Minutes before the Cap_Deadline the step leaves for the Train_Step (D18). |

## Glossary

- **Walls_Step**: the Step this spec builds, in `splatpipe/walls/`. Its Step_Name is `walls`, so its Config section is `[walls]`, its Work_Folder is `work/walls/` and its Step_Report is `out/reports/walls.json`.
- **Walls_Report**: the Walls_Step's Step_Report, with the fields of Requirement 19.
- **Depth_Prior_Set**: the Prior_Set that the Prior_Reader reads from `work/depth/`, which the Depth_Step writes.
- **Depth_Index**: `work/depth/depth.json`, the Depth_Prior_Set's Prior_Index.
- **Depth_Prior**: one Prior of the Depth_Prior_Set.
- **Prior_Frame**: a Frame that the Depth_Index lists, so it has a Depth_Prior.
- **Covered_Pixel**: a Prior_Size pixel of a Prior_Frame where its Depth_Prior's `mask` is true.
- **Prior_Depth**: at a Covered_Pixel, 1 / the Depth_Prior's `inverse_depth`, in camera solve units.
- **Frame_Time**: a Prior_Frame's time, its Pick's `seconds` in the Frames_Record, matched by the Pick's `file` (D8).
- **Score_Size**: the width and height the scoring works at: the Train_Step's Training_Size rule applied to the Shared_Camera with `score_long_edge` (640).
- **Luma**: a Prior_Frame's grey values from 0 to 255, as floats, from OpenCV's BGR-to-grey conversion of the Frame scaled down to the Score_Size with `INTER_AREA`.
- **High_Pass**: the Luma minus the Luma blurred by a Gaussian of sigma `high_pass_sigma` (3) Score_Size pixels.
- **Band**: one octave difference-of-Gaussians band of the Luma: the Luma blurred with one sigma of `band_sigmas` minus the Luma blurred with the next, where a sigma of 0 means the Luma itself. `[0, 1, 2, 4, 8]` gives 4 Bands.
- **Coherence**: ρ, one Band's mean Window_Score over a Prior_Frame's Valid_Pairs at the Second_Pass's depth.
- **Band_Weight**: one Band's weight w in the Band_Mix, such that w² × the Band's variance = ρ / (1 − ρ²) (A4).
- **Band_Mix**: the sum of a Prior_Frame's Bands, each times its Band_Weight, which the Third_Pass scores.
- **Score_Mask**: a Prior_Frame's keep mask at the Score_Size, from its Mask_File by the Train_Step's `keep_mask` rule with `mask_threshold`, or every pixel kept when it has no Mask_File. A pixel it doesn't keep is masked.
- **Normalised_Coordinates**: a pixel's (x, y) = ((u − cx) / fx, (v − cy) / fy), with (u, v) the pixel's centre in the Shared_Camera's pixels by the Prior_Format's pixel-centre rule.
- **Ray**: a pixel's r = (x, y, 1), from its Normalised_Coordinates.
- **Median_Inverse_Depth**: m, the median of a depth map's inverse depths over its Covered_Pixels, or over a Facet's pixels for a Facet's change.
- **Bas_Relief_Change**: four numbers (a, b, c, d) that change a depth map z into z′ by 1/z′ = a/z + m × (b·x + c·y + d) at each pixel, with m the Median_Inverse_Depth of z and (x, y) the pixel's Normalised_Coordinates: Kriegman and Belhumeur's perspective bas-relief family, written relative to the depth it changes.
- **Identity_Change**: the Bas_Relief_Change (1, 0, 0, 0), which changes nothing.
- **Depth_Factor**: s(p) = z′(p) / z(p), what a Bas_Relief_Change multiplies the depth at pixel p by.
- **Changed_Normal**: the normal a Bas_Relief_Change gives a pixel at depth z with Ray r and normal n: a·n + m·z·(n·r)·(b, c, d), set to unit length and turned so its dot product with r is below 0 (D4).
- **Scale_Points**: a Prior_Frame's Observed_Points that are in front of its camera, whose projection lands inside it on a Covered_Pixel by the Depth_Step's pixel rule (A7).
- **Scene_Scale**: k, the factor from a Prior_Frame's Scale_Points that brings its Prior_Depth to the camera solve's points (Requirement 7).
- **Start_Depth**: k × the Prior_Depth at each Covered_Pixel, the depth a Prior_Frame's First_Pass starts from.
- **Depth_Pixel**: a Score_Size pixel of a Prior_Frame that has a Start_Depth, by the Train_Step's prior-pixel rule for scaling a Prior (Requirement 7).
- **Current_Depth**: the depth a pass, a Half_Table or a Facet's refit builds its Score_Table around.
- **Source_Frame**: one of up to `max_sources` other Prior_Frames whose score images a Prior_Frame is scored against (Requirement 8).
- **Overlap**: the share of a Prior_Frame's Covered_Pixels whose point at the Start_Depth lands in front of a candidate Source_Frame's camera and inside its image.
- **Image_Motion**: how far, in the candidate's Score_Size pixels, a Covered_Pixel's point moves in the candidate when its depth changes by `motion_depth_change` (1%), as the median over the Covered_Pixels that land inside it (A8).
- **Source_Rank**: a candidate's Overlap × min(Image_Motion, `source_motion_cap`).
- **Score_Window**: a `window_size` × `window_size` window of a Prior_Frame at the Score_Size, centred on the grid every `window_step` pixels, that the scoring uses (A2).
- **Window_Score**: the zero-mean normalised cross-correlation (NCC) between one Score_Window's pixels in the Prior_Frame's score image and the same pixels of one Source_Frame's score image, warped into the Prior_Frame.
- **Valid_Pair**: a Score_Window and a Source_Frame whose Window_Score counts at a Depth_Factor (Requirement 9, A2).
- **Table_Scales**: the Depth_Factors a Score_Table holds (A3).
- **Table_Range**: from a Score_Table's smallest Table_Scale to its largest.
- **Score_Table**: C(p, s), for each Score_Window p and each Table_Scale s, the sum of the Window_Scores of p's Valid_Pairs at the Current_Depth × s.
- **Guess_Score**: F, the score of one guess of the four numbers: the sum over the Score_Windows p of C(p, s(p)), with s(p) the guess's Depth_Factor at p's centre.
- **Change_Fit**: the Nelder-Mead search for the Bas_Relief_Change with the highest Guess_Score on one Score_Table (Requirement 10, A5).
- **First_Pass**, **Second_Pass** and **Third_Pass**: a Prior_Frame's three passes (Requirement 10).
- **Solved_Change**: the one Bas_Relief_Change of the Start_Depth that the three passes make together.
- **Source_Halves**: Half_A, the 1st, 3rd, … ranked Source_Frames, and Half_B, the 2nd, 4th, … (D12).
- **Half_Table**: a Score_Table summed over one Source_Half's Valid_Pairs only.
- **Halves_Difference**: the median over the Covered_Pixels, or over a Facet's pixels, of |z_A / z_B − 1|, with z_A and z_B the depths each Source_Half's own change gives.
- **Tile**: a `tile_size` × `tile_size` square of the Score_Size, the Tiles laid from the top left.
- **Tile_Score**: Z, a two-sample z across the Tiles of the mean Window_Score at the answer against × or ÷ `z_scale`, with the standard error times `z_error_factor` (A6).
- **Self_Check**: the test of Requirement 11 that a Prior_Frame's answer passes or fails, with no ground truth.
- **Preparation_Pass**: the Walls_Step's pass over every Prior_Frame, on the main process after it reads the inputs and before any Frame_Solve starts, that works out each Prior_Frame's Scale_Points, its Scene_Scale and its Start_Depth at the Prior_Size and the Score_Size, so a Timed_Out_Frame has them too (Requirement 7, D24).
- **Frame_Solve**: one Prior_Frame's own work in a worker process, from the Start_Depth the Preparation_Pass worked out: its Source_Frames, its three passes, its Self_Check, its Facets and, when it passes, its Facets' refits.
- **Trusted_Frame**: a Prior_Frame whose Frame_Solve passed its Self_Check.
- **Failed_Frame**: a Prior_Frame whose Frame_Solve ended and didn't pass its Self_Check.
- **Timed_Out_Frame**: a Prior_Frame whose Frame_Solve hadn't ended by the Solve_Cutoff, started or not.
- **Facet**: a planar piece of a Prior_Frame's Depth_Prior at the Prior_Size (Requirement 12).
- **Jump_Pixel**: a Covered_Pixel whose inverse depth and that of a Covered_Pixel `facet_jump_distance` pixels away across or down differ by more than `facet_jump` of the smaller of the two.
- **Facet_Change**: a Bas_Relief_Change refitted for one Facet of a Trusted_Frame, relative to the Solved_Depth.
- **Facet_Plane**: the plane, a unit normal n and an offset δ with n·X = δ, that a robust fit gives through a Facet's points.
- **World_Plane**: a plane in the world that Facet_Planes of several Trusted_Frames make together (Requirement 13).
- **Neighbour_Frames**: a Prior_Frame's Trusted_Frames within `neighbour_window` (24) Pick_Numbers, at most `neighbour_frames` (12), the nearest first, never the Prior_Frame itself (D14).
- **Projected_Value**: at a Covered_Pixel of one Prior_Frame, the nearest depth that one Neighbour_Frame's depth map gives there, its points projected into the Prior_Frame through a z-buffer.
- **Agreeing_Values**: a pixel's Projected_Values within `agree_tolerance` (3%) of the nearest of them.
- **Solved_Depth**: a Trusted_Frame's Start_Depth through its Solved_Change, with its applied Facet_Changes on their Facets.
- **Fused_Depth**: a Trusted_Frame's depth after the fuse (Requirement 14).
- **Fallback_Depth**: a Failed_Frame's depth from the Projection_Fallback or the Plane_Registration (Requirement 15).
- **Projection_Fallback** and **Plane_Registration**: the two ways Requirement 15 makes a Fallback_Depth.
- **Walls_Depth**: a Trusted_Frame's Fused_Depth or a Failed_Frame's Fallback_Depth, as the correction, the flatten and the despeckle change it in turn.
- **Sparse_Error**: the median over a Prior_Frame's Scale_Points of |ln(z / z_point)|, with z the Walls_Depth at the point's pixel and z_point the point's camera-space z.
- **Depth_Prior_Copy**: a Walls_Prior that is the unchanged copy of all three arrays of its Depth_Prior (D2).
- **Walls_Prior**: one Prior_Frame's Prior in the Prior_Format, as the Walls_Step writes it.
- **Walls_Prior_File**: `work/walls/<stem>.npz`, one Walls_Prior.
- **Walls_Index**: `work/walls/depth.json`, the Prior_Index of the Walls_Priors, written last.
- **Walls_Record**: `work/walls/walls.json`, the Walls_Step's per-frame numbers and outcomes (D7).
- **Walls_Demo**: `out/walls/<stem>.jpg`, one side-by-side JPEG for the demo (Requirement 18).
- **Solve_Order**: the order the Frame_Solves start in (Requirement 4, D17).
- **Worker_Pool**: the Walls_Step's worker processes (Requirement 5).
- **Walls_Stop_Time**: the time by which the Walls_Step should be done (D18, A22).
- **Solve_Cutoff**: the Walls_Stop_Time minus `finish_minutes`, after which no Frame_Solve runs.
- **Kriegman_Belhumeur_Paper**: D. J. Kriegman and P. N. Belhumeur, "What Shadows Reveal about Object Structure", ECCV 1998, pp. 399–414, with its journal version in JOSA A 18(8), 2001.
- **Walls_Fixed_Constants**: what stays in code, not in Config, because it describes a format, a library, a contract or the maths and not a choice we tune: the names `walls.json`, `out/walls/` and `<stem>.jpg`; the Walls_Record's `format` number (1) and its field names; the Prior_Format's names, which the Walls_Step reuses from `splatpipe/train/priors.py`; OpenCV's grey conversion; the Solve_Order's rule; the colour mapping of A17; the sample variance's need for 2 Tiles; the internal scratch folder `work/walls/scratch/` and its file names; the one thread per worker of A16 and the names of the thread-limit environment variables; the fixed pattern of the Change_Fit's starts, which `fit_starts` and `fit_start_spread` size; and `SPLATPIPE_CAP_DEADLINE`. This is the same exception the depth-priors spec's Fixed_Constants make.

Terms this spec uses as the pipeline-core spec and the Pipeline_Doc define them: Skeleton, Step, Step_Name, Job, Jobs_Folder, Job_Folder, Step_Runner, Step_Report, Step_List, Required_Step, Optional_Step, Fake_Step, Work_Folder, Config, Preset, Shared_Setting, Step_Section, Expected_Failure, Unexpected_Error, Readable_Error, JSON_Value, Job_Runner, Job_Summary, Job_Status, Job_Log, Pipeline_Doc, Licence_List, CLI and Test_Suite.

Terms it uses as the depth-priors spec defines them: Depth_Step, Frame, Prior_Size, Observed_Points, Fit_Point, Prior, Prior_File, Prior_Index, Prior_Format, Prior_Reader and Prior_Set.

Terms it uses as the splat-training spec defines them: Train_Step, Camera_Solve_Output, Mask_File, Sparse_Model, Model_Reader, Shared_Camera, Image_Pose, Camera_Centre, Training_Size, Start_Splats, Loss_Term, GPU_Module, GPU_Stand_In, Train_Report and Stop_Time.

Terms it uses as the cloud-jobs spec defines them: Cloud_Runner, Start_Command, Cap_Deadline, Time_Cap, Expected_Time, Step_Needs, Fallback_List, Smoke_Test, Image_Build and Project_File. From the aws-setup spec it uses Machine_Image, from the website spec, Website, from the frames spec, Frames_Record and Pick, and from the cameras spec, Pick_Number.

## Requirements

### Requirement 1: The walls step and the step list

**User Story:** As someone who films a place, I want the depth priors corrected between the depth step and training, and the job to go on with the depth priors when they can't be, so that plain walls come out flat and a failed correction never costs me my splat.

**Decided:** D15 (criterion 4).

#### Acceptance Criteria

1. THE Walls_Step SHALL be a package `splatpipe/walls/` built like the Fake_Step and the Depth_Step: `work(step)`, `run(job_id, preset=None, *, config=None)` and a `__main__.py` whose `main(argv)` calls the Skeleton's `step_main`.
2. THE CLI `uv run python -m splatpipe.walls --job <id> [--preset P]` SHALL run the Walls_Step on one Job, print `walls: done` on standard output and exit with 0 when the step is done, and print the Readable_Error on standard error with no traceback and exit with 1 when the step failed.
3. THE Step_List in `splatpipe/run.py` SHALL be `(StepEntry("frames", "splatpipe.frames", True), StepEntry("cameras", "splatpipe.cameras", True), StepEntry("depth", "splatpipe.depth", False), StepEntry("walls", "splatpipe.walls", False), StepEntry("train", "splatpipe.train", True), StepEntry("clean", "splatpipe.clean", False))`, with the Step_List's comment naming the Walls_Step as optional and this spec.
4. THE Walls_Step SHALL do its work in this order: remove `out/walls/`; check the settings; read the Cap_Deadline and work out the Walls_Stop_Time and the Solve_Cutoff; check that its input files are there; read the Sparse_Model; read the Depth_Prior_Set; read the Frames_Record; the Preparation_Pass; run the Frame_Solves in the Worker_Pool; fail when no Prior_Frame is a Trusted_Frame; the fuse; the World_Planes; the fallback; the correction; the flatten; the despeckle; the per-frame choice; and write the Walls_Prior_Files, the Walls_Record, the Walls_Demos and, last of all, the Walls_Index (D15).
5. IF the Walls_Step fails, THEN THE Job_Runner SHALL write the Walls_Step's Readable_Error to the Job_Log and go on with the next Step, as for every Optional_Step.
6. WHEN the Walls_Step failed and every Required_Step is done, THE Job_Runner SHALL give the Job the Job_Status `done with warnings`, with the Job_Summary's entry for the Walls_Step holding the outcome `failed` and the Walls_Step's Readable_Error.
7. THE Walls_Step SHALL raise `ExpectedFailure` for every failure that Requirements 2 to 19 name, so the Walls_Report's `error.expected` is `true` and its `readable` is the step's own message, and every other error SHALL reach the Step_Runner as an Unexpected_Error.
8. THE Walls_Step SHALL write only inside `work/walls/` and `out/walls/`, and SHALL write nothing else to `out/` other than the Walls_Report, which the Step_Runner writes.
9. THE Walls_Step SHALL leave every file in `work/frames/`, `work/cameras/` and `work/depth/` as it is.
10. WHEN the Walls_Step runs again on a Job, THE Walls_Step SHALL rely on the Step_Runner emptying `work/walls/` first, and SHALL remove `out/walls/` first of all, so no file from an earlier run is left in either.
11. THE Walls_Step SHALL run on the CPU only and import no torch, gsplat, MoGe or other GPU package, so it runs the same on a laptop and on the job machine.
12. THE Walls_Step's code SHALL be written fresh from this spec, on the library APIs of numpy, scipy and OpenCV, with no code copied from any other repo, as rules.md requires.

### Requirement 2: Reading the inputs

**User Story:** As a team member building the depth step, I want the walls step to read the depth priors, the camera solve and the frames record with the readers the other steps use, so that one hand-off serves every step and a broken input fails with the same message everywhere.

**Decided:** D6 (criterion 10), D8 (criteria 7 to 9) and D10 (criterion 12). **Assumptions (to confirm at review):** A18 (criterion 12).

#### Acceptance Criteria

1. THE Walls_Step SHALL read the Depth_Prior_Set from `work/depth/`, the Camera_Solve_Output from `work/cameras/` (the Frames in `images/`, the Mask_Files in `masks/` and the Sparse_Model in `sparse/`) and the Frames_Record, `work/frames/frames.json`, and nothing else in the Job_Folder.
2. IF `work/depth/depth.json` isn't a file, THEN THE Walls_Step SHALL fail with an Expected_Failure whose Readable_Error names the path, says that the depth step writes it, so the depth step has to run first, and says that the Job goes on and trains as it would without the walls step.
3. IF `work/cameras/images/` or any of the Sparse_Model's three files is missing, THEN THE Walls_Step SHALL fail with an Expected_Failure whose Readable_Error names the missing path and says that the camera solve step writes it.
4. IF `work/frames/frames.json` isn't a file, THEN THE Walls_Step SHALL fail with an Expected_Failure whose Readable_Error names the path and says that the frames step writes it.
5. THE Walls_Step SHALL make the checks of criteria 2 to 4, without reading any file, after the settings and the Cap_Deadline and before it reads anything else.
6. THE Walls_Step SHALL read the Sparse_Model with the Train_Step's Model_Reader, and the Depth_Prior_Set with the Prior_Reader's `read_prior_set` and every one of its checks, and IF either refuses its files, THEN THE Walls_Step SHALL fail with an Expected_Failure that gives the reader's message, which names the file and what failed.
7. THE Walls_Step SHALL read the Frames_Record with Python's `json` module and use only its `picks`, and of each Pick only its `number`, a whole number of 0 or more, its `file`, a string, and its `seconds`, a finite number, and IF the record isn't a JSON object, `picks` isn't a list of objects, any of those fields is missing or of the wrong kind, or a `number` or a `file` appears twice, THEN THE Walls_Step SHALL fail with an Expected_Failure whose Readable_Error names the field and says to run the frames step on the Job again.
8. IF a Prior_Frame's name isn't the `file` of a Pick, THEN THE Walls_Step SHALL fail with an Expected_Failure whose Readable_Error names the image, says that `frames.json` doesn't match the camera solve, and says to run the Job again with `--fresh`.
9. THE Walls_Step SHALL give each Prior_Frame its Pick's `number` as its Pick_Number and its Pick's `seconds` as its Frame_Time (D8).
10. THE Walls_Step SHALL solve, score against, project from and write a Walls_Prior for only the Prior_Frames, and SHALL ignore every other image in `images.bin` (D6).
11. THE Walls_Step SHALL decode each Prior_Frame with the Train_Step's `read_training_image`, so a Frame with no file, one that can't be decoded and one that isn't the Shared_Camera's size fail with the Train_Step's Expected_Failures, and SHALL scale it to the Score_Size with the Train_Step's `scaled_down`.
12. WHEN `work/cameras/masks/<stem>.png` is a file for a Prior_Frame, THE Walls_Step SHALL read it with the Train_Step's rules for a Mask_File, refusing it with the same Expected_Failures, and SHALL make its Score_Mask with the Train_Step's `keep_mask` and `mask_threshold` from `[walls]`, and WHERE a Prior_Frame has no Mask_File, THE Walls_Step SHALL keep every pixel of its Score_Mask (D10, A18).
13. THE Walls_Step SHALL read every input with numpy, OpenCV and the standard library, and cv2 SHALL decode and scale only still images, never reading or writing video.

### Requirement 3: Settings and needs

**User Story:** As a team member, I want every number of the walls step in Config, checked before the step reads anything, and its machine needs where the cloud code already finds them, so that we can tune it without changing code.

**Decided:** D16 (criterion 8), D18 (criterion 3) and D22 (criteria 2 and 9). **Assumptions (to confirm at review):** A16, A17 and A18 (criterion 2).

#### Acceptance Criteria

1. THE Config SHALL hold a `[walls]` Step_Section after `[depth]`'s tables and before `[train]`, with the three Preset tables `[walls.fast]`, `[walls.default]` and `[walls.quality]`, and with a short comment on the line above every key.
2. THE `[walls]` section SHALL hold exactly the Shared_Settings of the settings table in the Introduction, with the values that table gives.
3. THE Preset tables SHALL hold exactly `max_minutes` and `train_reserve_minutes`: 20 and 40 in `[walls.fast]`, 40 and 60 in `[walls.default]`, and 60 and 110 in `[walls.quality]` (D18).
4. IF a `[walls]` setting is present but isn't the kind of value the settings table gives for it, THEN THE Walls_Step SHALL fail with an Expected_Failure, before it reads the Cap_Deadline, whose Readable_Error names the setting, its table and what it must be, where a bool is never a number.
5. IF `first_min_scale`, `second_min_scale`, `third_min_scale` or `facet_min_scale` isn't below 1, a matching maximum isn't above 1, `source_fallback_seconds` is above `source_min_seconds`, `min_sources` is above `max_sources`, `register_end_tolerance` is above `register_start_tolerance`, `despeckle_min_valid` is above `despeckle_window`², or `finish_minutes` isn't below the Preset's `max_minutes`, THEN THE Walls_Step SHALL fail with an Expected_Failure whose Readable_Error names both settings and their values.
6. IF a setting is missing from `[walls]` or the Preset's table, THEN THE Step_Runner's existing check SHALL fail the Walls_Step with the Step_Runner's Readable_Error, which names the section, the table and the key.
7. THE Walls_Step SHALL take every size, count, share, angle, tolerance, time and quality from Config, and hold in code only the Walls_Fixed_Constants.
8. THE `min_vcpus` and `min_memory_gb` in `[walls]` SHALL be the Walls_Step's Step_Needs, which the Start_Command takes into the needs over every Step as `cloud/select.py` already does, with no change to the cloud code, so `min_vcpus` stays a soft need that rules out no machine type and `min_memory_gb` a hard one (D16).
9. THE comment of each setting that D22 names, and of `mask_threshold`, `workers` and `demo_jpeg_quality`, SHALL say that its value is a default untested on real footage, except where it gives a rule, as `mask_threshold`'s does by matching `[train]`'s.
10. THE `tests/conftest.py` test config SHALL gain a `[walls]` section with its three Preset tables, so a test that runs the Walls_Step without a Config of its own gets settings, and a test SHALL be able to change any one of the settings for that test, as `set_frames` does for `[frames]`.

### Requirement 4: The time budget

**User Story:** As someone who films a place, I want the walls step to stop in time for training, and to spend that time spread over the whole video, so that a slow machine costs me some correction and never my splat.

**Decided:** D17 (criteria 4 and 5), D18 (criteria 2, 3, 6 and 7) and D24 (criterion 8). **Assumptions (to confirm at review):** A15 (criterion 9) and A19 (criterion 12).

#### Acceptance Criteria

1. THE Walls_Step SHALL read the Cap_Deadline from `SPLATPIPE_CAP_DEADLINE` with the Train_Step's `read_cap_deadline`, after the settings and before the input checks, and IF the value is malformed, THEN THE Walls_Step SHALL fail with that function's Expected_Failure.
2. THE Walls_Stop_Time SHALL be the earlier of the step's start + `max_minutes` and the Cap_Deadline − `train_reserve_minutes`, and WHERE no Cap_Deadline is set, THE Walls_Stop_Time SHALL be the step's start + `max_minutes`, with the step's start read once, when its work starts (D18).
3. THE Solve_Cutoff SHALL be the Walls_Stop_Time − `finish_minutes`.
4. THE Solve_Order SHALL list the Prior_Frames, sorted by Pick_Number and indexed from 0 to N − 1, as: 0; then N − 1 when N is above 1; then, from a first-in-first-out queue that starts with the interval (0, N − 1), for each interval (lo, hi) taken from the queue with hi − lo of 2 or more, the index ⌊(lo + hi) / 2⌋, putting (lo, mid) and (mid, hi) at the end of the queue, until the queue is empty (D17).
5. THE Walls_Step SHALL start the Frame_Solves in the Solve_Order, each as soon as a worker is free, and SHALL start no Frame_Solve at or after the Solve_Cutoff.
6. WHEN the Solve_Cutoff comes while Frame_Solves are still running, THE Walls_Step SHALL end each of them and count its Prior_Frame, and every Prior_Frame whose Frame_Solve never started, as a Timed_Out_Frame (D18).
7. THE Walls_Step SHALL give each Timed_Out_Frame a Depth_Prior_Copy, and no Fallback_Depth, correction, flatten or despeckle (D18, D21).
8. THE Walls_Step SHALL keep using a Timed_Out_Frame as a Source_Frame of other Prior_Frames, because a Source_Frame needs only its Depth_Prior, its Start_Depth and its Frame (D24).
9. THE Walls_Step SHALL run the cross-frame stages and the writes after the Frame_Solves end, with no cutoff of their own (A15).
10. THE Walls_Step SHALL work out the Walls_Stop_Time, the Solve_Cutoff and whether a Frame_Solve may start with pure functions that take the time as an argument, so the Test_Suite passes them a fake clock.
11. THE Walls_Record SHALL hold the step's start, the Cap_Deadline or null, the Walls_Stop_Time, the Solve_Cutoff, and which of the two limits of criterion 2 set the Walls_Stop_Time.
12. FOR ALL N from 1 to 1,000, THE Solve_Order SHALL list every index from 0 to N − 1 exactly once, starting with 0 and, when N is above 1, then N − 1 (property, A19).

### Requirement 5: Using every vCPU

**User Story:** As a team member, I want the walls step to use every vCPU of the job machine without fighting over them, so that the frames are solved in about the time we planned and the step leaves nothing running.

**Assumptions (to confirm at review):** A16 (criteria 1, 4, 5 and 7).

#### Acceptance Criteria

1. THE Worker_Pool SHALL have `workers` worker processes when `workers` is above 0, and one per vCPU the process may use when `workers` is 0, counted with `os.sched_getaffinity(0)` where the system has it and with `os.cpu_count()` otherwise, and never more than the number of Prior_Frames (A16).
2. THE Walls_Step SHALL run every Frame_Solve in a worker process of the Worker_Pool, so as many Frame_Solves run side by side as the Worker_Pool has workers.
3. THE Walls_Step SHALL run the per-frame work of the fuse, the fallback, the correction, the flatten, the despeckle and the per-frame choice in the Worker_Pool too.
4. THE Walls_Step SHALL limit each worker process's OpenCV and numpy work to one thread, so the workers together use the vCPUs without oversubscribing them (A16).
5. THE Walls_Step SHALL reach the Worker_Pool through one module-level hook that the Test_Suite can replace with an in-process stand-in, as the Train_Step reaches its GPU_Module (A16).
6. WHEN the Walls_Step ends, done or failed, THE Walls_Step SHALL leave no worker process running.
7. IF a worker process ends without giving back its result, THEN THE Walls_Step SHALL fail with an Expected_Failure whose Readable_Error says that a worker process stopped, names the Prior_Frame it was working on when that's known, says that the machine may have run out of memory, and says that the Job goes on and trains on the depth priors (A16).
8. WHEN a worker raises an `ExpectedFailure`, THE Walls_Step SHALL fail with that same Expected_Failure and Readable_Error, and WHEN a worker raises any other error, THE Walls_Step SHALL fail with an Unexpected_Error.
9. WHEN no Prior_Frame times out, THE Walls_Step SHALL write the same Walls_Priors, bit for bit, and the same Walls_Record values, other than its times, whatever the number of workers and whatever order the Frame_Solves end in.

### Requirement 6: The bas-relief change and its normals

**User Story:** As a team member, I want one small, tested function for the four-number change and the normals it gives, so that every pass, fit and fallback changes depth the same way and planes stay planes.

**Decided:** D4 (criterion 4). **Assumptions (to confirm at review):** A19 (criteria 5 and 10).

#### Acceptance Criteria

1. THE Walls_Step SHALL apply a Bas_Relief_Change (a, b, c, d) to a depth map in inverse depth, in float64: at each pixel with depth z and Normalised_Coordinates (x, y), the new inverse depth is a × (1/z) + m × (b·x + c·y + d), with m the Median_Inverse_Depth of the depth map it changes.
2. WHEN the Bas_Relief_Change is the Identity_Change, THE Walls_Step SHALL give back every inverse depth unchanged, bit for bit.
3. THE Walls_Step SHALL give a Bas_Relief_Change's Depth_Factor at a pixel as the old inverse depth divided by the new one, and a new inverse depth that isn't finite and above 0 SHALL give that pixel no new depth.
4. THE Walls_Step SHALL give each pixel a Bas_Relief_Change touches its Changed_Normal: a·n + m·z·(n·r)·(b, c, d) from the pixel's normal n, depth z and Ray r, set to unit length and, when its dot product with r is above 0, turned round, and a result of length 0 or with a part that isn't finite SHALL give the pixel no new normal (D4).
5. THE Walls_Step SHALL compose two Bas_Relief_Changes, the first (a₁, b₁, c₁, d₁) on a depth map of Median_Inverse_Depth m₁ and the second (a₂, b₂, c₂, d₂) on the result, of Median_Inverse_Depth m₂, into the one change (a₁·a₂, a₂·b₁ + (m₂ / m₁)·b₂, a₂·c₁ + (m₂ / m₁)·c₂, a₂·d₁ + (m₂ / m₁)·d₂) of the first depth map, and THE Walls_Record SHALL hold each Prior_Frame's Solved_Change in that form, relative to its Start_Depth.
6. THE Bas_Relief_Change, the Changed_Normal and their composition SHALL be numpy functions, shared by every pass, fit and fallback of the Walls_Step.
7. FOR ALL Shared_Cameras, pixels, depths above 0 and Bas_Relief_Changes whose new inverse depth at the pixel is finite and above 0, THE point at the new depth on the pixel's Ray SHALL lie on the same ray from the camera centre as the old point, in front of the camera, within a relative error of 10⁻⁹ (property).
8. FOR ALL planes that don't pass through the camera centre, sets of 4 or more pixels whose Rays meet the plane in front of the camera, and Bas_Relief_Changes whose new inverse depths at those pixels are finite and above 0, THE changed points SHALL lie on one plane, within a relative error of 10⁻⁶ of their distance from the camera centre, and THE Changed_Normal at each of them SHALL be that plane's unit normal facing the camera, within 10⁻⁶ (property).
9. FOR ALL depth maps of finite depths above 0, unit normals facing the camera and Normalised_Coordinates, THE Identity_Change SHALL give back the same inverse depths bit for bit and the same normals within 10⁻⁶ (property).
10. FOR ALL pairs of Bas_Relief_Changes and depth maps on which both changes give finite inverse depths above 0, THE composed change of criterion 5 applied to the first depth map SHALL give the same inverse depths as the two changes applied in turn, within a relative error of 10⁻⁹ (property, A19).

### Requirement 7: The scene scale and the start depth

**User Story:** As someone who films a place, I want each frame's depth prior first brought to the camera solve's own points, so that the search starts from the right size of room and spends its range on the shape.

**Assumptions (to confirm at review):** A7 (criteria 1 and 2).

#### Acceptance Criteria

1. THE Walls_Step SHALL take each Prior_Frame's Scale_Points from its Observed_Points, with each point's camera-space z from the Prior_Frame's Image_Pose and its Prior_Size pixel by the Depth_Step's pixel rule, and SHALL keep only the points in front of the camera whose projection lands inside the Frame on a Covered_Pixel (A7).
2. THE Walls_Step SHALL work out each Scale_Point's r = ln(z / the Prior_Depth at its pixel), the median r̃ of the r, the median absolute deviation D of |r − r̃|, and the Scene_Scale k = exp(the median of the r with |r − r̃| at most `scale_mad_gate` × D) (A7).
3. IF a Prior_Frame has fewer than `scale_min_points` Scale_Points, THEN THE Walls_Step SHALL give it the Scene_Scale k = 1.
4. THE Walls_Step SHALL give each Covered_Pixel the Start_Depth k × its Prior_Depth.
5. THE Walls_Step SHALL make the Start_Depth at the Score_Size from the Depth_Prior's `inverse_depth` scaled to the Score_Size with the Train_Step's rule for scaling a Prior (`upscale_taps`, `upscale` and `prior_pixel_mask`), divided by k, and SHALL count a Score_Size pixel as a Depth_Pixel only where that rule gives the pixel a prior.
6. THE Walls_Record SHALL hold each Prior_Frame's Scene_Scale and its number of Scale_Points.

### Requirement 8: Choosing the source frames

**User Story:** As a team member, I want each frame scored against the other frames that see the same surface from far enough away, so that the faint texture's parallax decides the depth and the codec's shared noise doesn't.

**Decided:** D6 (criterion 1), D8 (criteria 1 and 5) and D14 (criterion 7). **Assumptions (to confirm at review):** A8 (criteria 2 to 5).

#### Acceptance Criteria

1. THE Walls_Step SHALL take as a Prior_Frame's candidates the other Prior_Frames whose Frame_Time differs from its own by at least `source_min_seconds` (D6, D8).
2. THE Walls_Step SHALL work out each candidate's Overlap over every Covered_Pixel of the Prior_Frame at its Start_Depth, and SHALL drop a candidate whose Overlap is below `source_min_overlap` (A8).
3. THE Walls_Step SHALL work out each candidate's Image_Motion as the median, over the Covered_Pixels whose point lands inside the candidate, of the distance in the candidate's Score_Size pixels between the projections of the point at the Start_Depth and at (1 + `motion_depth_change`) × the Start_Depth, and SHALL drop a candidate whose Image_Motion is below `source_min_motion` (A8).
4. THE Walls_Step SHALL rank the candidates left by Source_Rank, the highest first, with ties going to the lower Pick_Number (A8).
5. THE Walls_Step SHALL choose the Source_Frames in rank order, taking each candidate whose Pick_Number differs by at least `source_min_gap` from that of every Source_Frame already chosen, until it has `max_sources` (D8).
6. IF criteria 1 to 5 give fewer than `min_sources` Source_Frames, THEN THE Walls_Step SHALL choose again with `source_fallback_seconds` in place of `source_min_seconds`.
7. IF a Prior_Frame still has fewer than `min_sources` Source_Frames, THEN THE Walls_Step SHALL fail its Self_Check without running any pass (D14).
8. THE Walls_Record SHALL hold each Prior_Frame's number of Source_Frames and the time, `source_min_seconds` or `source_fallback_seconds`, they were chosen with.

### Requirement 9: Windows, warping and the score table

**User Story:** As a team member, I want each guess scored by summing many weak window correlations across many frames into one table, so that a frame's four numbers can be searched quickly and the faint texture of a wall still decides.

**Decided:** D9 (criteria 1 and 2), D10 (criteria 2 and 5), D11 (criterion 10) and D24 (criteria 5 and 6). **Assumptions (to confirm at review):** A1 (criteria 5 and 6), A2 (criteria 2 and 5) and A3 (criteria 7 and 9).

#### Acceptance Criteria

1. THE Walls_Step SHALL score at the Score_Size, on the High_Pass in the First_Pass and the Second_Pass and on the Band_Mix in the Third_Pass, for the Prior_Frame and for each Source_Frame alike (D9).
2. THE Walls_Step SHALL take as a Prior_Frame's Score_Windows the windows centred every `window_step` Score_Size pixels across and down, starting ⌊`window_size` / 2⌋ pixels in from the top left, whose pixels all lie inside the image, are all Depth_Pixels and are all kept by the Score_Mask, and whose High_Pass variance is at least `flat_variance` (D10, A2).
3. THE Walls_Step SHALL warp a Source_Frame into a Prior_Frame at a Depth_Factor s by taking each Depth_Pixel's point at s × the Current_Depth on its Ray, moving it into the Source_Frame's camera through both Image_Poses, projecting it with the Shared_Camera scaled to the Score_Size, and sampling the Source_Frame's score image there bilinearly.
4. THE Walls_Step SHALL work out a Window_Score as the zero-mean normalised cross-correlation of the Score_Window's pixels in the Prior_Frame's score image and in the warped Source_Frame's score image.
5. THE Walls_Step SHALL count a Score_Window and a Source_Frame as a Valid_Pair at a Depth_Factor only when every pixel of the window warps to a point in front of the Source_Frame's camera and inside its image, touches no pixel that the Source_Frame's Score_Mask masks, the warped window's High_Pass variance is at least `flat_variance`, and the window isn't occluded for the Source_Frame (D10, D24, A1, A2).
6. THE Walls_Step SHALL count a Score_Window as occluded for a Source_Frame when, at the window's centre warped at the Current_Depth, the depth the warp predicts in the Source_Frame differs by more than `occlusion_tolerance` of it from the Source_Frame's Start_Depth, its Scene_Scale k × its Prior_Depth, at the Prior_Size pixel under the warped centre by the Depth_Step's pixel rule, or that pixel has no Start_Depth because it isn't one of the Source_Frame's Covered_Pixels, with the test made once per Score_Table and holding at every Table_Scale (D24, A1).
7. THE Walls_Step SHALL give each Score_Table its Table_Scales from the pass's minimum to its maximum scale, both included, evenly spaced in ln(s), with ⌈ln(max / min) / step⌉ + 1 scales (A3).
8. THE Walls_Step SHALL build each Score_Table C(p, s) as the sum, over the Valid_Pairs of Score_Window p at Table_Scale s, of their Window_Scores, with 0 where p has no Valid_Pair.
9. THE Walls_Step SHALL work out a guess's Guess_Score as the sum over the Score_Windows p of C(p, s(p)), with s(p) the guess's Depth_Factor at p's centre on the Current_Depth, looked up by linear interpolation in ln(s) between the two neighbouring Table_Scales (A3).
10. IF a guess's Depth_Factor at any Score_Window isn't finite or lies outside the Table_Range, or its |b|, |c| or |d| is above `fit_bound`, THEN THE Walls_Step SHALL reject the guess (D11).
11. THE warping, the Window_Scores and the Score_Tables SHALL be numpy and OpenCV functions that the Test_Suite tests with no worker process.

### Requirement 10: The three passes and the band mix

**User Story:** As someone who films a place, I want each frame's depth found coarse to fine and finished on the image detail that really agrees across frames, so that the search can't lock onto noise and ends at the right shape.

**Decided:** D11 (criterion 3). **Assumptions (to confirm at review):** A4 (criteria 5 and 6) and A5 (criteria 3 and 7).

#### Acceptance Criteria

1. THE First_Pass SHALL build a Score_Table on the High_Pass around the Start_Depth from `first_min_scale` to `first_max_scale` with `first_scale_step`, and SHALL take the Table_Scale s with the highest sum over the Score_Windows of C(p, s), the one nearest 1 in ln(s) on a tie, so its change is (1/s, 0, 0, 0) and its depth s × the Start_Depth.
2. THE Second_Pass SHALL build a Score_Table on the High_Pass around the First_Pass's depth from `second_min_scale` to `second_max_scale` with `second_scale_step`, and SHALL take the Change_Fit on it as its change.
3. THE Change_Fit SHALL run scipy's Nelder-Mead from `fit_starts` starts, the Identity_Change and `fit_starts` − 1 others spread by `fit_start_spread`, with `fit_max_iterations`, `fit_x_tolerance` and `fit_f_tolerance`, scoring a rejected guess as worse than every other, and SHALL give back the guess with the highest Guess_Score of every guess it evaluated, so its Guess_Score is at least the Identity_Change's (D11, A5).
4. THE Walls_Step SHALL work out each Band's Coherence ρ as the mean Window_Score on that Band over the Prior_Frame's Valid_Pairs at the Second_Pass's depth.
5. THE Walls_Step SHALL give each Band the Band_Weight w with w² × the Band's variance = ρ / (1 − ρ²), with ρ capped at `band_max_coherence`, and a weight of 0 where ρ isn't above 0 or the Band's variance is 0 (A4).
6. THE Third_Pass SHALL score the Band_Mix of the Prior_Frame and of each Source_Frame, each made with the Prior_Frame's own Band_Weights, and WHERE every Band_Weight is 0, THE Third_Pass SHALL score the High_Pass instead (A4).
7. THE Third_Pass SHALL build a Score_Table and the two Half_Tables around the Second_Pass's depth from `third_min_scale` to `third_max_scale` with `third_scale_step`, and SHALL take the Change_Fit on the Score_Table as its change (A5).
8. THE Walls_Step SHALL give each Prior_Frame whose passes ran its Solved_Change: the First_Pass's, the Second_Pass's and the Third_Pass's changes composed with Requirement 6, criterion 5, relative to the Start_Depth, so the Scene_Scale k stays a number of its own in the Walls_Record.
9. THE Walls_Record SHALL hold each Prior_Frame's First_Pass scale, its Solved_Change (a, b, c, d) with its m, and its four Band_Weights.

### Requirement 11: The self-check

**User Story:** As someone who films a place, I want each frame's answer checked against itself with no ground truth, so that a frame whose texture couldn't decide its depth doesn't get a confident wrong one.

**Decided:** D12 (criteria 1 to 3), D13 (criterion 7), D14 (criterion 7) and D24 (criterion 5). **Assumptions (to confirm at review):** A6 (criteria 4 to 6 and 8).

#### Acceptance Criteria

1. THE Walls_Step SHALL split each Prior_Frame's Source_Frames, in rank order, into the Source_Halves: Half_A the 1st, 3rd, 5th and so on, and Half_B the 2nd, 4th, 6th and so on (D12).
2. THE Walls_Step SHALL run the Change_Fit on each Source_Half's Half_Table of the Third_Pass, giving each half its own change of the Second_Pass's depth (D12).
3. THE Walls_Step SHALL work out the Halves_Difference over the Prior_Frame's Covered_Pixels, with z_A and z_B the Second_Pass's depth through each half's change (D12).
4. THE Walls_Step SHALL take as a Tile's Score_Windows those whose centres lie in the Tile, and as the answer the Second_Pass's depth through the Third_Pass's change (A6).
5. THE Walls_Step SHALL work out, for each of the answer × `z_scale` and the answer ÷ `z_scale`, the mean Window_Score of each Tile at the answer and at that depth, over the Score_Windows' Valid_Pairs at both depths, with the occlusion test of Requirement 9, criterion 6, made at the answer against each Source_Frame's Start_Depth (D24, A1, A6).
6. THE Walls_Step SHALL work out each Tile_Score over the Tiles with at least `tile_min_windows` such Score_Windows as Z = (mean_A − mean_B) / (`z_error_factor` × √(var_A / n + var_B / n)), with mean_A and var_A the mean and sample variance of the Tiles' means at the answer, mean_B and var_B those at the other depth, and n the number of those Tiles (A6).
7. THE Walls_Step SHALL pass a Prior_Frame's Self_Check only when it has at least `min_sources` Source_Frames, its Halves_Difference is below `halves_tolerance`, and both its Tile_Scores are above `min_z`, and SHALL fail it otherwise (D13, D14).
8. IF fewer than 2 Tiles count for a Tile_Score, THEN THE Walls_Step SHALL fail the Self_Check, and WHERE the standard error of a Tile_Score is 0, THE Walls_Step SHALL give it the value +∞ when mean_A − mean_B is above 0 and fail the Self_Check otherwise (A6).
9. THE Walls_Record SHALL hold each Prior_Frame's Halves_Difference, both its Tile_Scores, the numbers of Tiles they counted and whether it passed its Self_Check.

### Requirement 12: Facets

**User Story:** As someone who films a place, I want each flat piece of a frame refitted on its own when the evidence for it is strong, so that a wall the whole-frame change can't fit still comes out flat and in the right place.

**Decided:** D9 (criteria 1 to 3), D13 (criterion 5) and D24 (criterion 4). **Assumptions (to confirm at review):** A5 (criterion 4) and A9 (criteria 1, 3, 4 and 7).

#### Acceptance Criteria

1. THE Walls_Step SHALL cut each solved Prior_Frame's Depth_Prior into Facets at the Prior_Size, from the Depth_Prior's `normal` and `inverse_depth` (D9, A9).
2. THE Walls_Step SHALL count a Covered_Pixel as a Jump_Pixel when its `inverse_depth` and that of a Covered_Pixel `facet_jump_distance` pixels away across or down differ by more than `facet_jump` of the smaller of the two.
3. THE Walls_Step SHALL make each Facet a 4-connected set of Covered_Pixels, holding no Jump_Pixel, whose every normal lies within `facet_normal_degrees` of the Facet's normal, the normalised mean of its pixels' normals, with at least `facet_min_pixels` pixels, and with no Covered_Pixel in two Facets (A9).
4. WHEN a Prior_Frame passes its Self_Check, THE Walls_Step SHALL, inside its Frame_Solve, build each Facet a Score_Table and its two Half_Tables on the Band_Mix around the Solved_Depth before any Facet_Change, from `facet_min_scale` to `facet_max_scale` with `facet_scale_step`, over the Score_Windows whose centres' Prior_Size pixel by the Train_Step's `keep_mask` rule lies in the Facet, with the occlusion test of Requirement 9, criterion 6, made at the Solved_Depth against each Source_Frame's Start_Depth with `facet_occlusion_tolerance` in place of `occlusion_tolerance`, and SHALL run the Change_Fit on each, with m the Facet's Median_Inverse_Depth (D24, A5, A9).
5. THE Walls_Step SHALL apply a Facet_Change only when its Halves_Difference over the Facet's pixels is at most `facet_halves_tolerance`, the median over the Facet's pixels of |z_facet / z − 1| is above `facet_min_change_ratio` × that Halves_Difference, and both its Tile_Scores, over the Tiles of its own Score_Windows, are at least `facet_min_z` (D13).
6. WHEN the Walls_Step applies a Facet_Change, THE Walls_Step SHALL give the Facet's pixels the Facet_Change's depth and Changed_Normal and leave every other pixel as it was.
7. THE Walls_Step SHALL refit no Facet of a Failed_Frame, and SHALL keep every Failed_Frame's and Trusted_Frame's Facets for the flatten (A9).
8. THE Walls_Record SHALL hold each Prior_Frame's number of Facets and of applied Facet_Changes, and each applied Facet_Change's numbers.

### Requirement 13: World planes

**User Story:** As someone who films a place, I want the flat surfaces that several trusted frames agree on joined into planes of the room, so that frames that failed can be put back on them and every frame's walls end up on the same plane.

**Decided:** D15 (criterion 1). **Assumptions (to confirm at review):** A10 (criteria 1, 3 and 4).

#### Acceptance Criteria

1. THE Walls_Step SHALL build the World_Planes once, after the fuse, from every Facet of every Trusted_Frame, with each Facet's points at their Fused_Depth in the world (D15, A10).
2. THE Walls_Step SHALL fit each Facet's Facet_Plane through its world points with a robust principal component fit, reweighting each point's distance from the plane with Tukey's weights at `tukey_constant` median absolute deviations, for `tukey_iterations` rounds.
3. THE Walls_Step SHALL merge the Facet_Planes, the largest Facet first, each Facet joining the first World_Plane whose normal lies within `plane_merge_degrees` of its own and whose offset differs from its own by at most `plane_merge_offset` × its Facet's camera distance, or else starting a new one, and SHALL refit each World_Plane through all its Facets' points with criterion 2's fit after each join (A10).
4. THE Walls_Step SHALL keep only the World_Planes with Facets from at least `plane_min_frames` Prior_Frames and at least `plane_min_share` of the pixels of the largest World_Plane with Facets from at least `plane_min_frames` Prior_Frames (A10).
5. THE Walls_Record SHALL hold the number of World_Planes kept.

### Requirement 14: Neighbour frames and the fuse

**User Story:** As someone who films a place, I want each trusted frame's depth checked against its trusted neighbours, so that one frame's small mistakes are outvoted where its neighbours agree.

**Decided:** D14 (criterion 1) and D15 (criterion 3). **Assumptions (to confirm at review):** A11 (criteria 1, 2 and 4).

#### Acceptance Criteria

1. THE Walls_Step SHALL give each Prior_Frame its Neighbour_Frames: the Trusted_Frames other than itself whose Pick_Numbers are within `neighbour_window` of its own, at most `neighbour_frames` of them, the nearest Pick_Numbers first and the lower Pick_Number on a tie (D14, A11).
2. THE Walls_Step SHALL give each Covered_Pixel of a Prior_Frame at most one Projected_Value from each Neighbour_Frame, the nearest of that Neighbour_Frame's points that project onto the pixel by the Depth_Step's pixel rule, in front of the camera (A11).
3. THE Walls_Step SHALL fuse each Trusted_Frame: at each Covered_Pixel, it takes the pixel's own Solved_Depth and the Projected_Values from its Neighbour_Frames' Solved_Depths that are within `fuse_tolerance` of that own depth, and WHEN those are at least `fuse_min_values` values, the pixel's Fused_Depth is their median, and otherwise its own Solved_Depth.
4. THE Walls_Step SHALL make every cross-frame stage read the other Prior_Frames' depths as the stage before it left them, so the order it takes the Prior_Frames in doesn't change the result (A11).

### Requirement 15: Fallback depths

**User Story:** As someone who films a place, I want a frame that failed its self-check to get its depth from the trusted frames around it, so that it still trains on a corrected wall when its own evidence was too weak.

**Decided:** D1 (criteria 1 to 4) and D14 (criterion 2). **Assumptions (to confirm at review):** A12 (criteria 2 to 4).

#### Acceptance Criteria

1. THE Walls_Step SHALL give each Covered_Pixel of each Failed_Frame its Projected_Values from its Neighbour_Frames' Fused_Depths, and WHEN at least `agree_min_frames` of them are Agreeing_Values, the median of the Agreeing_Values.
2. WHEN at least `projection_min_coverage` of a Failed_Frame's Covered_Pixels have such a median, THE Walls_Step SHALL make its Fallback_Depth with the Projection_Fallback: one Tukey fit of a Bas_Relief_Change of its Start_Depth to those medians in inverse depth, with `tukey_constant` and `tukey_iterations`; each pixel with a median takes the median, each other Covered_Pixel takes the fitted change's depth, and every Covered_Pixel takes the fitted change's Changed_Normal (D1, D14, A12).
3. WHEN fewer than `projection_min_coverage` of a Failed_Frame's Covered_Pixels have a median and there are World_Planes, THE Walls_Step SHALL make its Fallback_Depth with the Plane_Registration: `register_passes` passes whose depth tolerance falls geometrically from `register_start_tolerance` to `register_end_tolerance`, each matching every Covered_Pixel to the World_Plane nearest its current depth whose normal in the camera frame is within `register_max_degrees` of the pixel's current normal and whose depth along the pixel's Ray, in front of the camera, is within the pass's tolerance of the pixel's current depth, then one Tukey fit of a Bas_Relief_Change of the Start_Depth to the matched pixels' plane depths, so the Fallback_Depth and its normals are the Start_Depth and the normals through the last pass's change (A12).
4. IF a Failed_Frame has no World_Plane to register to, or fewer than `register_min_pixels` pixels match in any Plane_Registration pass, THEN THE Walls_Step SHALL give the Failed_Frame no Fallback_Depth, so it gets a Depth_Prior_Copy (A12).
5. THE Walls_Record SHALL hold each Failed_Frame's fallback, `projection`, `planes` or null, its share of Covered_Pixels with a median, and its fitted change.

### Requirement 16: The correction, the flatten and the despeckle

**User Story:** As someone who films a place, I want every corrected frame pulled onto what its trusted neighbours agree on, its flat pieces made truly flat, and its stray pixels removed, so that the walls the train step sees are smooth.

**Decided:** D4 (criterion 3), D9 (criteria 1 to 4) and D15 (criteria 1 to 4). **Assumptions (to confirm at review):** A11 (criterion 5) and A13 (criterion 6).

#### Acceptance Criteria

1. THE Walls_Step SHALL correct each Prior_Frame with a Walls_Depth: at each Covered_Pixel whose Projected_Values from its Neighbour_Frames' Fused_Depths hold at least `agree_min_frames` Agreeing_Values, with median M, and whose own depth z has |ln(z / M)| above `correction_tolerance`, the pixel takes the depth M.
2. THE Walls_Step SHALL flatten each Facet of each Prior_Frame with a Walls_Depth: it fits the Facet's robust normal through its points at the Walls_Depth in the camera frame with Requirement 13, criterion 2's fit, takes the normal n of the World_Plane, in the camera frame, nearest in angle to that robust normal when one is within `flatten_degrees`, and else the robust normal itself, and takes the offset δ as the median over the Facet's pixels of n·X.
3. WHEN a flattened Facet's pixel has a depth z within `flatten_tolerance` of its plane depth z_plane = δ / (n·r), with z_plane finite and above 0, THE Walls_Step SHALL give the pixel the depth z_plane and the normal n turned to face the camera, and SHALL leave every other pixel of the Facet as it was (D4).
4. THE Walls_Step SHALL despeckle each Prior_Frame with a Walls_Depth: for each Covered_Pixel, the `despeckle_window` × `despeckle_window` window centred on it, cut at the image's border, has as valid pixels its Covered_Pixels with a finite depth above 0, the pixel itself included, and when the window has at least `despeckle_min_valid` valid pixels and the pixel's depth z has |z / M − 1| above `despeckle_tolerance`, with M the valid pixels' median, the pixel takes the depth M.
5. THE correction, the flatten and the despeckle SHALL each read the depths from before that stage, for every pixel, so no pixel's change within a stage moves another's (A11).
6. THE correction and the despeckle SHALL change depths only, and THE flatten SHALL change a pixel's normal only when it moves the pixel (A13).
7. FOR ALL depth maps of finite depths above 0 with any masks, `despeckle_window` and `despeckle_min_valid` settings and `despeckle_tolerance` of 0 or more, THE despeckle SHALL leave unchanged, bit for bit, every pixel whose depth is within `despeckle_tolerance` of its window's median, and every pixel whose window has fewer than `despeckle_min_valid` valid pixels (property).

### Requirement 17: The per-frame choice and the walls priors

**User Story:** As someone who films a place, I want every frame to end with one depth prior that is either the corrected one or exactly the old one, so that training always gets a prior it can trust for every frame that had one.

**Decided:** D1 (criteria 2 to 4), D2 (criterion 6), D3 (criterion 7) and D4 (criterion 8). **Assumptions (to confirm at review):** A13 (criterion 8) and A14 (criterion 3).

#### Acceptance Criteria

1. THE Walls_Step SHALL give every Prior_Frame exactly one Walls_Prior and every other Frame none.
2. THE Walls_Step SHALL give each Trusted_Frame a Walls_Prior made from its Walls_Depth.
3. WHEN a Failed_Frame has a Fallback_Depth and either a Sparse_Error of at most `fallback_max_sparse_error` or fewer than `scale_min_points` Scale_Points, THE Walls_Step SHALL give it a Walls_Prior made from its Walls_Depth (D1, A14).
4. IF a Failed_Frame with a Fallback_Depth has at least `scale_min_points` Scale_Points and a Sparse_Error above `fallback_max_sparse_error`, THEN THE Walls_Step SHALL give it a Depth_Prior_Copy (D1).
5. THE Walls_Step SHALL give each Timed_Out_Frame and each Failed_Frame with no Fallback_Depth a Depth_Prior_Copy.
6. THE Walls_Step SHALL make a Depth_Prior_Copy hold its Depth_Prior's `inverse_depth`, `normal` and `mask`, each unchanged, bit for bit (D2).
7. THE Walls_Step SHALL make a Walls_Prior from a Walls_Depth with its Depth_Prior's `mask` exactly, `inverse_depth` the float32 of 1 / the Walls_Depth at each pixel the mask marks and 0 at every other, and at each marked pixel whose float32 value isn't finite and above 0, the Depth_Prior's `inverse_depth` and `normal` (D3).
8. THE Walls_Step SHALL give each marked pixel of a Walls_Prior made from a Walls_Depth the unit normal, facing the camera, that Requirements 6, 12, 15 and 16 give it, and its Depth_Prior's `normal` where that normal isn't finite or has a length of 0, and (0, 0, 0) at every pixel the mask doesn't mark (D4, A13).
9. THE Walls_Step SHALL make every Walls_Prior pass the Prior_Format's `check_prior` at the Prior_Size.
10. THE Walls_Record SHALL hold each Prior_Frame's outcome, `passed`, `fallback` or `depth_prior`, its Sparse_Error or null, and for `depth_prior` its reason: `timed_out`, `no_fallback` or `sparse_error`.

### Requirement 18: Writing the walls priors and the demo

**User Story:** As a team member, I want the walls priors written in exactly the depth priors' format, with their own record and a few pictures that show the walls, so that training reads them with no new code and the demo shows the difference.

**Decided:** D4 (criterion 3), D5 (criteria 5 and 7) and D7 (criteria 2 and 4). **Assumptions (to confirm at review):** A15 (criterion 8) and A17 (criterion 3).

#### Acceptance Criteria

1. THE Walls_Step SHALL write each Walls_Prior to its Walls_Prior_File with the Prior_Format's `write_prior`, at `prior_path(work/walls, name)`, at the Depth_Index's Prior_Size.
2. THE Walls_Step SHALL write the Walls_Record, `work/walls/walls.json`, holding only JSON_Values: `format` (1), the run's numbers of Requirement 4, criterion 11, the number of workers and of World_Planes and the names of the Walls_Demos' Prior_Frames, and one object per Prior_Frame in Pick_Number order with its `name`, its Pick_Number and the per-frame numbers that Requirements 7 to 12, 15 and 17 give, with null for a number that isn't finite or wasn't worked out (D7).
3. WHERE `demo_frames` is above 0, THE Walls_Step SHALL write one Walls_Demo, `out/walls/<stem>.jpg`, for each of the Prior_Frames whose Walls_Prior isn't a Depth_Prior_Copy at the places ⌊(i + 0.5) × N / `demo_frames`⌋, i from 0 to `demo_frames` − 1, in Pick_Number order among those N Prior_Frames, or for every one of them when N is below `demo_frames`: a JPEG at `demo_jpeg_quality` that holds side by side the Frame at the Prior_Size, the normals of the Depth_Prior's depth and the normals of the Walls_Prior's depth, each normal worked out from its depth map by finite differences for display only, so a flat wall shows as one flat colour (D4, A17).
4. THE Walls_Step SHALL write the Walls_Index, `work/walls/depth.json`, with the Prior_Format's `write_index`, holding exactly the Depth_Index's values: its camera, its Prior_Size, its `metres_per_unit` and every entry with its fit fields, in its order (D7).
5. THE Walls_Step SHALL write the Walls_Prior_Files first, then the Walls_Record, then the Walls_Demos, and the Walls_Index last of all, so a reader that finds the Walls_Index finds every Walls_Prior_File (D5).
6. IF writing a Walls_Prior_File, the Walls_Record, a Walls_Demo or the Walls_Index fails because the disk is full or the file can't be written, THEN THE Walls_Step SHALL fail with an Expected_Failure whose Readable_Error names the file and the reason.
7. WHEN the Walls_Step fails for any reason, THE Walls_Step SHALL leave no Walls_Index in `work/walls/`, so the Train_Step reads `work/depth/` (D5).
8. IF no Prior_Frame is a Trusted_Frame, THEN THE Walls_Step SHALL fail with an Expected_Failure, before it writes any Walls_Prior_File, the Walls_Record, any Walls_Demo or the Walls_Index, whose Readable_Error gives the number of Prior_Frames, how many timed out, how many failed their Self_Check and, of those, how many had too few Source_Frames, and says that the Job goes on and trains on the depth priors (A15).
9. THE Prior_Reader SHALL read the Walls_Index and the Walls_Prior_Files with `read_prior_set(work/walls, model)` and every one of its checks, with no field added to the Prior_Format, which the Test_Suite checks on the files a run writes.
10. THE Walls_Step SHALL encode the Walls_Demos with `cv2.imencode` as still JPEGs, never reading or writing video.

### Requirement 19: The walls step's report

**User Story:** As a team member, I want the walls step's report to say how many frames it solved, trusted, fell back on and ran out of time for, so that I can tell a good walls run from a cut-off or a poor one.

**Decided:** D21 (criterion 1).

#### Acceptance Criteria

1. WHEN the Walls_Step is done, THE Walls_Report SHALL hold the counts `images` (the Prior_Frames), `solved_images` (the Prior_Frames whose Frame_Solve ended), `passed_images` (the Trusted_Frames), `fallback_images` (the Failed_Frames whose Walls_Prior was made from a Fallback_Depth), `depth_prior_images` (the Depth_Prior_Copies) and `timed_out_images` (the Timed_Out_Frames), so `images` = `passed_images` + `fallback_images` + `depth_prior_images` = `solved_images` + `timed_out_images` (D21).
2. WHEN the Walls_Step is done, THE Walls_Report SHALL hold the timings `read` (the inputs), `solve` (the Frame_Solves, from the first's start to the last's end or the Solve_Cutoff), `planes`, `fuse`, `fallback`, `correct`, `flatten`, `despeckle`, `write` (the Walls_Prior_Files, the Walls_Record and the Walls_Index) and `demo`, each in seconds.
3. WHEN the Walls_Step fails, THE Walls_Report SHALL hold exactly the counts and timings of criteria 1 and 2 that the step had worked out before it failed.
4. THE Walls_Step SHALL add each count as a whole number given as an int and each timing as a number of seconds, so the Step_Runner's checks never refuse one.
5. WHERE the Walls_Step has Timed_Out_Frames, THE Walls_Report SHALL hold a warning that gives the Timed_Out_Frames, the Prior_Frames, the Walls_Stop_Time in UTC and whether `max_minutes` or the Train_Step's `train_reserve_minutes` before the Cap_Deadline set it, and says that those frames keep their depth prior.
6. WHERE fewer Prior_Frames passed their Self_Check than there are Prior_Frames, THE Walls_Report SHALL hold a warning that gives the Trusted_Frames, the Prior_Frames with a Walls_Prior from a Fallback_Depth and the Depth_Prior_Copies, out of the Prior_Frames.
7. THE Walls_Report SHALL be the Step_Runner's ordinary Step_Report, written to `out/reports/walls.json`, with every setting of `[walls]` and of the Preset's table, which the Step_Runner puts there itself.

### Requirement 20: Training reads the walls priors

**User Story:** As someone who films a place, I want training to use the walls priors when the walls step made them and the depth priors otherwise, so that the walls step can only help and training never has to know how a prior was made.

**Decided:** D5 (criteria 1 to 4).

#### Acceptance Criteria

1. WHEN `work/walls/depth.json` exists in the Job_Folder, THE Train_Step SHALL read its Prior_Set from `work/walls/` with `read_prior_set` and every one of its checks, at the place where it reads `work/depth/` now (D5).
2. IF the Prior_Set in `work/walls/` fails a check, THEN THE Train_Step SHALL add a warning to the Train_Report that names the file and what failed and says that training fell back to the depth priors in `work/depth/`, and SHALL then read `work/depth/` exactly as the depth-priors spec says, with that spec's own warning when those fail too (D5).
3. WHERE `work/walls/depth.json` doesn't exist, THE Train_Step SHALL read `work/depth/` exactly as the depth-priors spec says, and SHALL add no warning about the Walls_Step.
4. THE Train_Step SHALL change nothing else: no new setting, count, timing or Loss_Term and no change to the Start_Splats, with `prior_images` counting the Priors of the Prior_Set it trains with, from whichever folder (D5).
5. THE Train_Step SHALL choose between `work/walls/` and `work/depth/` in one function, `_read_priors` in `splatpipe/train/__init__.py`, the one place that picks the Train_Step's prior folder.
6. THE Train_Step SHALL keep naming, in the depth-priors spec's warning for broken priors, the folder it read, so a reader tells `work/walls/` from `work/depth/`.

### Requirement 21: Docs, cloud times and licences

**User Story:** As one of the three people building the steps, I want the walls step written up next to the other steps in our own words, its cost in the quotes, and its licences checked, so that the team and the judges can see what it does, what it costs and that it adds nothing we can't use.

**Decided:** D16 (criterion 7), D18 (criterion 8), D19 (criteria 11 and 12) and D20 (criterion 13). **Assumptions (to confirm at review):** A20 (criterion 14).

#### Acceptance Criteria

1. THE Pipeline_Doc SHALL gain a `## Step: walls` section, after `## Step: depth` and before `## Step: train`, with the parts `### What it reads`, `### What it writes`, `### Config section`, `### Step report fields` and `### Files in out/`.
2. THE `## Step: walls` section SHALL explain the method in our own words, from the Bas_Relief_Change and why it keeps every pixel on its ray and every plane flat to the per-frame choice, SHALL cite the Kriegman_Belhumeur_Paper by its authors, title, ECCV 1998 and pages 399–414, with its JOSA A 18(8) 2001 version, and SHALL quote no text from the paper.
3. THE `### What it writes` part SHALL say that the Walls_Prior_Files and the Walls_Index are the Prior_Format at the Depth_Step's Prior_Size, that the Walls_Index copies the Depth_Index, which Prior_Frames get a Depth_Prior_Copy, the Walls_Record's fields, and that the Walls_Index is written last.
4. THE `### Config section` part SHALL give every key of `[walls]` and of its Preset tables with its table, its value and what it is, and SHALL say which values are defaults untested on real footage.
5. THE `### Step report fields` part SHALL give the counts, the timings, the warnings and the Expected_Failures of Requirements 2 to 19, and THE `### Files in out/` part SHALL say that the step writes the Walls_Demos to `out/walls/` and removes that folder first, at the start of each run.
6. THE Pipeline_Doc's "The step list" part SHALL say that the list holds the frames step and the cameras step, both required, the depth step and the walls step, both optional, the train step, required, and then the clean step, optional, with each entry as the real Step_List writes it, that a Job is done once all six steps are done, and that it's done with warnings when the three required steps are done and the depth step, the walls step or the clean step fails, and its list of the planned optional steps SHALL name the walls step.
7. THE Pipeline_Doc's Step_Needs table SHALL gain a row for the walls step, 16 vCPUs and 32 GB RAM, and say that the needs live in `[walls]` as `min_vcpus`, a soft need, and `min_memory_gb`, and that `max_minutes` cuts the step short on a machine with fewer vCPUs (D16).
8. THE Pipeline_Doc's "The cap deadline" part SHALL say that the walls step reads `SPLATPIPE_CAP_DEADLINE` too and stops solving frames in time to leave the train step its `train_reserve_minutes` (D18).
9. THE Pipeline_Doc's `## Step: train` section SHALL say that the Train_Step reads `work/walls/` when `work/walls/depth.json` is there and `work/depth/` otherwise, and give the fallback warning of Requirement 20, criterion 2.
10. THE Pipeline_Doc's `## Step: depth` section SHALL say that the walls step reads the depth priors too and leaves them as they are.
11. THE `[cloud.presets.*]` tables SHALL have `expected_minutes` 50 for `fast`, 100 for `default` and 180 for `quality`, with `cap_minutes` unchanged at 120, 180 and 360, so the Website's reservations don't change (D19).
12. THE Pipeline_Doc SHALL state the three new Expected_Times, the rough cost the Walls_Step adds to a Job of each Preset (about $0.80, $1.60 and $3.00), and that the Website's start page shows the new times once the account owner redeploys, and every other doc that states the old Expected_Times SHALL change with them, in the same task as `config.toml`, as SHALL `tests/test_cloud_config.py`'s `PRESET_MINUTES` and `tests/test_start_core.py`'s quote (D19).
13. THE Project_File, `uv.lock`, `app/pyproject.toml` and `app/uv.lock` SHALL gain no package, and `cloud/image.py`, the Smoke_Test's checks and the Machine_Image SHALL stay as they are, so no Image_Build is needed (D20).
14. THE Licence_List SHALL gain no row, because the Walls_Step adds no dependency, tool, model weight or dataset and the Kriegman_Belhumeur_Paper is none of these, and its `opencv-python-headless` row's "How we use it" cell SHALL add that the walls step decodes the camera solve's frames and masks, blurs and warps them and writes its demo JPEGs, and SHALL still say that it never reads or writes video (A20).
15. THE `tests/test_docs.py` SHALL check the step list sentences of criterion 6 in place of the old ones, that the `## Step: walls` section's parts name every key of the real `[walls]` and its Preset tables, every count and timing of Requirement 19, `work/walls/depth.json`, `walls.json` and `out/walls/`, and that the section cites the Kriegman_Belhumeur_Paper.

### Requirement 22: Tests

**User Story:** As a team member, I want every part of the walls step tested on a laptop with tiny made-up scenes, so that the tests are quick, free and safe to run anywhere, and the first real job only runs code that is already tested.

**Assumptions (to confirm at review):** A19 (criterion 3) and A21 (criterion 10).

#### Acceptance Criteria

1. THE Test_Suite SHALL test with unit tests, on the CPU with tiny synthetic data: the Bas_Relief_Change, the Changed_Normal and their composition; the settings and their refusals; the Walls_Stop_Time, the Solve_Cutoff and the Solve_Order with a fake clock; the Frames_Record checks; the Scene_Scale with its gate and with too few Scale_Points; the Source_Frames' time rule, the drop to `source_fallback_seconds`, the spacing, the Overlap, the Image_Motion and the rank; the warping, which takes a pixel at its true depth to the pixel that sees the same point in the Source_Frame; the Window_Scores and each reason a pair isn't a Valid_Pair; the Score_Table, the interpolation and the rejected guesses; the Change_Fit; the Bands and the Band_Weights; the Source_Halves, the Halves_Difference and the Tile_Score; the Facets, the Jump_Pixels and the Facet_Change's tests; the World_Planes; the Neighbour_Frames and the Projected_Values; the fuse; the Projection_Fallback and the Plane_Registration; the correction, the flatten and the despeckle; the per-frame choice; the Walls_Priors, the Depth_Prior_Copies, the Walls_Index, the Walls_Record and the Walls_Demos; and the report fields.
2. THE Test_Suite SHALL render a textured plane, seen from at least 3 cameras, with faint random texture, give the passes a Start_Depth off from the true depth by a known scale inside the First_Pass's range, and check that the passes' depth is within 1% of the true depth, as the median over the Covered_Pixels of |z / z_true − 1|, that the Self_Check passes, and that the same plane with no texture fails the Self_Check.
3. THE Test_Suite SHALL hold the property tests of Requirement 4, criterion 12, Requirement 6, criteria 7 to 10, and Requirement 16, criterion 7, written with Hypothesis (A19).
4. THE Test_Suite's unit tests and property tests SHALL both be required, never optional.
5. THE Test_Suite SHALL run the Walls_Step through `run()` on the CPU, with the in-process stand-in for the Worker_Pool and a fake clock, to test: a done run whose Walls_Prior_Files and Walls_Index `read_prior_set` reads back; a run with no `work/depth/depth.json`; a run with broken depth priors; a Frames_Record that doesn't match the camera solve; a run in which no Prior_Frame passes, which leaves no Walls_Index; a run cut off at the Solve_Cutoff, whose Timed_Out_Frames get Depth_Prior_Copies; a worker that ends without a result; and that `out/walls/` from an earlier run is removed.
6. THE Test_Suite SHALL run one tiny Walls_Step with the in-process stand-in and with 2 real worker processes and check that both write the same Walls_Priors, bit for bit.
7. THE Test_Suite SHALL check, by listing the Job_Folder before and after a run, that the Walls_Step writes nothing outside `work/walls/`, `out/walls/` and the Walls_Report, and that every Readable_Error of its Expected_Failures names what its requirement says and holds no traceback.
8. THE Test_Suite SHALL check that every file in `splatpipe/walls/` imports only names in the standard library, numpy, scipy, `cv2` and `splatpipe`, names none of cv2's video classes, and that `import splatpipe.walls` in a fresh process loads no torch, gsplat or moge, and `tests/test_cloud_imports.py` SHALL keep refusing `cloud` and boto3 in every file under `splatpipe/`.
9. THE Test_Suite SHALL check that `[walls]` loads from the real Config for each of the three Presets with exactly the keys of Requirement 3, and that, with the real Config and Step_List, every type on each Preset's Fallback_List still meets the Step_Needs that the Start_Command works out.
10. THE `tests/test_run.py` test that asserts the Step_List SHALL assert the Step_List of Requirement 1, criterion 3; `tests/test_frames_step.py`'s Job through the Step_List without the cameras entry SHALL expect the Walls_Step failed between the depth and train entries; and `tests/test_cameras_step.py`'s Job through the real Step_List SHALL expect the Walls_Step's entry with the outcome the design gives for its tiny scene, all in the same task that changes the Step_List (A21).
11. THE Test_Suite SHALL run the Train_Step through `run()` on the CPU with a GPU_Stand_In, to test a run that trains with the walls priors, a run with broken walls priors that warns and trains with the depth priors, and a run with no `work/walls/` whose Train_Report holds exactly the counts, timings and warnings it holds now.
12. THE Test_Suite SHALL run on the CPU with no network and no AWS, make every Job in a temporary Jobs_Folder and never in the real `data/` folder, build every image, model, prior and mask in the test, and commit none to git.
