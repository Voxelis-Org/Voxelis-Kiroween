# Requirements Document

## Introduction

This spec builds the Cameras_Step, the second step of the Pipeline. It solves the cameras for the Picks that the Frames_Step wrote to `work/frames/`: where each Pick was taken from, and the one lens that every Pick shares. It runs COLMAP 4.2.1 on the CPU, as a command line tool, in this order:

1. `feature_extractor` on the Grey_Copies, with one shared camera and the Mask.
2. `sequential_matcher` twice, first for the near pairs and then for the far pairs.
3. `view_graph_calibrator`, because the Picks carry no EXIF, so the global mapper would have no focal length to rely on.
4. `global_mapper`, then the incremental `mapper` once if the global mapper gives up.
5. `image_undistorter` to one PINHOLE camera on the Colour_Picks, and `image_undistorter_standalone` on the Mask.

Then the step levels and centres the scene with one rigid move, saves the levelled cameras and points in its Work_Folder, `work/cameras/`, for the Training_Step, and copies the levelled cameras to `out/cameras.json` for the Viewer.

The Cameras_Step is a Required_Step and the second entry in the Step_List. It uses the Job_Folder, the Step_Runner, the Step_Report and the Presets from the pipeline-core spec, reads `work/frames/` as the frames spec lays it out, and it's built like the Fake_Step and the Frames_Step.

In the cloud, the Cloud_Runner from the cloud-jobs spec runs the step on the Job's machine through `run_job`, the same way a laptop run does. The step's Step_Needs ask for 16 vCPUs, a soft need, and 16 GB of memory, a hard need, but the step itself never checks the machine, so it runs the same on a laptop with COLMAP installed. It reads the Cap_Deadline and caps each COLMAP command's Time_Limit at the time left before it (D17), so a hung command fails the step before the Cap_Deadline, and the machine's Time_Cap stays the backstop. Its maths is tested on the laptop with tiny made-up cameras and points and no COLMAP, and one integration test runs the real COLMAP when it's installed. The same solve of a tiny made-up scene runs on the machine image as the Cameras_Smoke_Check, one of the Smoke_Test's checks (D15).

COLMAP is a command line tool the step calls, never something it links or imports. The machine image builds it without CUDA, OpenGL, the GUI, CGAL and LSD, so its non-commercial SiftGPU code, GPL CGAL and AGPL LSD are never in it, and the step passes `--FeatureExtraction.use_gpu 0` and `--FeatureMatching.use_gpu 0` anyway. The colmap-cli skill in `.kiro/skills/colmap-cli/` holds what we checked about each command, and its `references/options.md` gives each command's full option list with COLMAP 4.2.1's defaults.

The training spec, now on `main`, was written against the hand-off in D2, so this spec keeps to it exactly.

Out of scope:

- The Training_Step, which reads `work/cameras/` with its own reader, and the Viewer, which reads `out/cameras.json`. This spec writes the files they read and states the contract in the Pipeline_Doc.
- The cloud wrapper, which the cloud-jobs spec (`.kiro/specs/cloud-jobs/`) built and which is already on `main`: its Start_Command reads the Step_Needs to pick the machine, and its Cloud_Runner runs the Job there. This spec changes `cloud/` only to add the Cameras_Smoke_Check to the Smoke_Test's checks and its numbers to `[cloud.smoke]` (D15), and leaves the Cloud_Runner's uploads as they are (D16). Nothing in this spec runs anything paid or anything in the cloud: the Cameras_Smoke_Check runs only when someone runs the Smoke_Test, which needs its own yes.
- GPU work and dense reconstruction. Nothing here needs a GPU, CUDA, torch, gsplat or AWS.
- Loop detection, vocabulary trees, exhaustive matching, and learned features and matchers (ALIKED, LoMa and LightGlue). The step uses SIFT and `SIFT_BRUTEFORCE` only, which need no model weights. It keeps loop detection off, because loop detection needs a vocabulary tree that COLMAP would download at run time. So nothing is downloaded while the step runs, whether or not the build has ONNX or download support. Homebrew's laptop COLMAP has both.
- Changing the Frames_Step, the pipeline-core spec's code or the Job_Record. The Camera_Model is a setting, the same for every Job (D5). Skeleton modules change only in the Step_List and its comment in `splatpipe/run.py`. The one change to the Frames_Step's files is a comment in `splatpipe/frames/settings.py`, whose Machine_Needs becomes Step_Needs (Requirement 15, criterion 13).
- `.kiro/steering/tech.md`, which this spec leaves as it is.

### Decisions made before this draft

The user chose these, in the request or when asked, before the requirements were written.

- **D1 (Requirement 14):** The Cameras_Step goes into the Step_List as its second entry, after the Frames_Step, and as a Required_Step. `main` has since added the Training_Step's and the Clean_Step's entries, so after the merge of `origin/main` the Step_List is frames, cameras, train, clean: `STEP_LIST = (StepEntry("frames", "splatpipe.frames", True), StepEntry("cameras", "splatpipe.cameras", True), StepEntry("train", "splatpipe.train", True), StepEntry("clean", "splatpipe.clean", False))`. A Job is done once all four steps are done, and done with warnings when the three Required_Steps are done and the Clean_Step fails. Every place that assumes another Step_List changes in the same work, the way the frames spec's D1 did: `splatpipe/run.py`, `tests/test_run.py`, `tests/test_imports.py`, `tests/test_docs.py`, `tests/conftest.py`, the Pipeline_Doc's "The step list" section, and notes in the frames spec's requirements. `tests/test_frames_step.py` is one of these places too, because its job runner tests run entries of the real Step_List and check the Job's outcome.
- **D2 (Requirements 8, 10 and 11):** The hand-off to the Training_Step. The user fixed it, because the training spec is being written against it.
  - `work/cameras/sparse/` holds the Levelled_Model in COLMAP 4.2.1's binary format, exactly three files: `cameras.bin`, with exactly one shared PINHOLE camera, the Undistorted_Camera, which has fx = fy since D19, and `images.bin` and `points3D.bin`. It holds no `rigs.bin` or `frames.bin`.
  - Each name in `images.bin` matches its file in `work/cameras/images/<name>`. The names are the Frames_Step's, like `000000.jpg`, so that file is `work/cameras/images/000000.jpg`.
  - `work/cameras/masks/000000.png` and the others exist only when the Frames_Record has `mask_used`. There's one mask per Placed_Frame, each the same Undistorted_Mask, since every image shares one camera.
  - The Training_Step reads these files with its own small reader, with no COLMAP or pycolmap installed. That doesn't mean another format.
  - However the Levelling is done, the Levelled_Model is written in that binary format, matching 4.2.1's layout byte for byte, checked against the 4.2.1 source.
  - COLMAP's unlevelled model and its other files stay in `work/cameras/colmap/`. The Undistortion writes its own `images/` and `sparse/` inside its output folder, so they land under `work/cameras/colmap/`, never in `work/cameras/sparse/`.
- **D3 (Requirement 11):** `out/cameras.json` is a copy of the levelled cameras for the Viewer: cameras only, no points. The Step_Runner never cleans `out/`, so the Cameras_Step replaces or removes `out/cameras.json` itself.
- **D4 (Requirements 2, 3 and 12):** Each tunable COLMAP option gets one flat key in `[cameras]`, an Option_Key, with a comment, about 180 of them, and the step checks each one's kind before COLMAP runs.
  - The values that the build and the rules force are Fixed_Constants in code, like the frames step's: `--FeatureExtraction.use_gpu 0`, `--FeatureMatching.use_gpu 0`, `--FeatureExtraction.type SIFT`, `--FeatureMatching.type SIFT_BRUTEFORCE`, `--SequentialMatching.loop_detection 0`, the CUDA solver options off (`--GlobalMapper.gp_use_gpu 0`, `--GlobalMapper.ba_ceres_use_gpu 0` and `--Mapper.ba_use_gpu 0`), the log options (`--log_target stderr`, `--log_level 0`, `--log_severity 0` and `--log_color 0`), the paths, and the one shared camera (`--camera_mode 1`).
  - Each undistortion setting is one key, which both `image_undistorter` and `image_undistorter_standalone` use.
  - Every other option in `.kiro/skills/colmap-cli/references/options.md` is passed with its value from `[cameras]`, so no COLMAP default decides anything.
- **D5 (Requirements 3, 5 and 12):** The camera model is the Shared_Setting `camera_model`, `"RADIAL"` or `"RADIAL_FISHEYE"`, the same for every Job, with no change to the pipeline-core spec or the Job_Record. `"RADIAL"` is the one shared camera of the request, with one focal length f for both axes, because phone video has square pixels. `"RADIAL_FISHEYE"` is for very wide lenses. It starts from a configured field of view across the long edge, `fisheye_fov_degrees`: f = (long edge / 2) / (field of view / 2, in radians), the principal point at the image centre, and k1 and k2 at 0, passed as `--ImageReader.camera_params`. That gives the camera a focal prior, which COLMAP 4.2.1 needs to keep fisheye pairs. The Calibration is skipped for `"RADIAL_FISHEYE"`, since `view_graph_calibrator` only fits pinhole-type cameras. D19 put these two in place of the first draft's `"OPENCV"` and `"OPENCV_FISHEYE"`.
- **D6 (Requirements 4 and 12):** The Near_Match_Run uses `quadratic_overlap 0`, with an overlap of 15 for `fast` and `default` and 25 for `quality`, a Preset_Setting. The Far_Match_Run, on the same Database, uses `quadratic_overlap 1` and only adds the far pairs, because COLMAP skips the pairs it already has. Its overlap must be at least 9 to reach 256 Picks ahead with 400 Picks. It's a key, with the value the design proposes.
- **D7 (Requirements 5 and 6):** The Calibration runs on a copy of the Database. If it exits with code 1, rejects the camera (`1 cameras rejected`) or finds no pairs, the step skips the Global_Run and goes straight to the Incremental_Run, which is then the one Fallback.
- **D8 (Requirements 2, 6 and 12):** The Global_Run gives up when it exits with a non-zero code for any reason other than the step's own Timeout (exit code 1 with `Failed to create sparse model`, or a crash such as 134, which Python's `subprocess` gives as -6), when it writes no Model, or when its Kept_Model fails the Placed_Check. Any of these starts the one Incremental_Run, which reads the original Database, not the Calibration_Copy. Any Timeout fails the step with no Fallback, and each COLMAP command has its own Time_Limit in `[cameras]`. Both mappers run with multiple models on, and the step keeps the Model with the most Placed_Frames, the lowest Model number on a tie. "Placed" counts the Picks in the kept Model.
- **D9 (Requirements 7 and 12):** The step fails with an Expected_Failure, and so the Job fails, when fewer than 20 Picks, or under half of them, are placed in the Final_Model, after the Fallback when there was one. Both numbers are keys. The Readable_Error tells the person how to film better: move slower, hold the phone steadier, overlap more between views, and film surfaces with more texture.
- **D10 (Requirement 8):** The Undistortion warps the Colour_Picks to PINHOLE. The Mask_Warp warps `mask.png` with `image_undistorter_standalone`, which samples bilinearly into a three-channel PNG. Then every pixel below 255 becomes 0 (ignore) and the rest become 255, in one 8-bit grey channel. Both run with `--blank_pixels 0`.
- **D11 (Requirements 9 and 12):** The Levelling is one rigid move with scale 1. It averages every Placed_Frame's up vector, Rᵀ(0, −1, 0). If they roughly agree, with the mean at least `up_agreement` (0.5) long, it rotates the mean to (0, −1, 0), turns the scene about the vertical so the first Placed_Frame looks along +Z, and moves the centre to the origin. If the mean is shorter, it skips the whole move, keeps COLMAP's frame and adds a warning.
  - The centre is the point nearest all the camera axes when that's well defined: the smallest eigenvalue of Σ(I − d dᵀ) is at least `axis_point_min_eigen_share` of the largest, and the point is in front of at least half the cameras. Otherwise the centre is the median of the points.
  - If the first Placed_Frame, the one with the lowest Pick_Number, looks nearly straight up or down, so its heading is undefined, the turn uses the next Placed_Frame that has a heading. "Nearly" is the key `heading_min_degrees`.
  - If anything in the Levelling fails, the step keeps COLMAP's frame, and the Levelling never fails the Job. So if no Placed_Frame has a heading, or the median is needed and the model has no points, the step keeps COLMAP's frame and adds a warning. The user wasn't asked about these two cases, but they follow from that rule.
- **D12 (Requirements 2, 10 and 15):** COLMAP runs as a command line tool through `subprocess`, with a time limit, never linked or imported, and with no pycolmap. A missing `colmap` program fails the step with a Readable_Error. Our own fresh code reads and writes the model files. Nothing is downloaded while the step runs.
- **D13 (Requirements 12 and 15):** `min_vcpus = 16` and `min_memory_gb = 16` go in `[cameras]`, like `[frames]`, as the step's Step_Needs for the cloud-jobs spec's Start_Command. The step checks their values like the frames step does but never uses them, so it never checks the machine.
- **D14 (Requirement 16):** Unit tests cover the Levelling maths and the Placed_Check on synthetic cameras, with no COLMAP. Property tests check that the Levelling keeps every distance between cameras and points, keeps the scale at 1, and gives an up vector of (0, −1, 0) when the cameras agree. A real COLMAP run is an integration test that's skipped when COLMAP isn't installed, and it checks that the Levelled_Model reads back through `colmap model_analyzer` and through our own reader. Every test is required, never optional, and runs on the CPU with tiny synthetic data and no network.

### Decisions made after the spec was finished

Samarth answered these once every task was done. Each one changes the requirements it names, and the new sub-tasks after task 10.9 build them.

- **D15 (Requirements 16 and 18):** The Smoke_Test gets a cameras check, `SmokeCheck("cameras", _check_cameras)` in `SMOKE_CHECKS` in `cloud/smoke_runner.py`, right after the colmap check. The machine image has no pytest, because it's installed with `uv sync --no-dev`, so the check's code lives in `splatpipe/cameras/smoke.py`, like `splatpipe/train/smoke.py`: the Option_Check of every option the step passes against each command's `-h`, then the Smoke_Solve of a 30-frame made-up scene. The `-h` test in `tests/test_cameras_colmap.py` and the integration test in `tests/test_cameras_real.py` use the same code.
- **D16 (Requirement 11):** Files that an earlier run uploaded to the Job_Bucket and a rerun removed stay in the bucket, and `cloud/` stays as it is for them. They never feed a step: the Step_Runner empties a step's Work_Folder before the step runs, and the Step_Reports decide which steps are done. Deleting them would also need a delete right that our roles don't have.
- **D17 (Requirements 2, 13 and 17):** The step caps each COLMAP command's Time_Limit at the time left before the Cap_Deadline, which it reads from `SPLATPIPE_CAP_DEADLINE` with `read_cap_deadline` from `splatpipe/train/deadline.py`, with no cap when the variable isn't set. The seven Time_Limit keys keep their values (O1), and the first real runs tune them.
- **D18 (Requirements 15 and 16):** Homebrew's COLMAP 4.2.1 may run the laptop tests, the integration test's whole solve included. Samarth OKs it: it's a separate program on a developer's laptop, never shipped, linked or vendored, the tests use only CPU SIFT and the mappers, and the machine image's own build has none of SiftGPU, CGAL and LSD. This replaces A15's "only for `-h` and small CPU tests".
- **D19 (Requirements 3, 5, 6, 10, 12, 15 and 16):** One shared RADIAL camera (f, cx, cy, k1, k2) in place of OPENCV, because phone video has square pixels, and RADIAL_FISHEYE, with the same starting parameters, in place of OPENCV_FISHEYE for very wide lenses. On the integration test's level path, with no pitch, no roll and a constant height, the OPENCV camera's fy wasn't pinned down: fx came out at 259.7 px against a true 260 px, but fy came out at 191 to 209 px over three runs. The Undistortion still hands the Training_Step one shared PINHOLE camera in `work/cameras/sparse/`, now with fx = fy. The integration test keeps its level path, because that's the case that broke, and checks that f comes out within 2 % of the truth.
- **D20 (Requirements 8 and 12):** `mapper_ba_refine_extra_params` and `global_mapper_ba_refine_extra_params` accept only true, and the settings check refuses false. With them off, the distortion stays 0, COLMAP's undistorter keeps the camera's own model and copies the Picks unwarped, and the step would fail only at its undistortion check, after the whole solve.

### Facts checked in the COLMAP 4.2.1 source during clarify

- Sequential pairing: in quadratic mode, Pick i is paired only with Pick i + 2^k, for k from 0 to overlap − 1. In linear mode, Pick i is paired with Picks i + 1 to i + overlap.
- `view_graph_calibrator` only fits pinhole-type cameras. Every camera it fits gets `has_prior_focal_length = true`, even a rejected one, whose focal length goes back to its starting value. "No image pairs to calibrate" returns success, with exit code 0. A Ceres failure exits with code 1.
- Two-view geometry: a shared pinhole-type camera without a focal prior uses the shared-focal solver. With focal priors on both cameras, it uses the calibrated solver on bearing vectors, which handles fisheye. A fisheye pair without a focal prior is marked DEGENERATE.
- Binary model files: `ReadBinary` needs `cameras.bin`, `images.bin` and `points3D.bin`, and reads `rigs.bin` and `frames.bin` only when they exist. For a folder without them, `ReadImagesBinary` makes one trivial rig per camera. `WriteBinary` always writes all five files. Each `images.bin` record starts with the image id, the cam_from_world quaternion (w, x, y, z), the translation and the camera id. The rest of the record is O10.

### Assumptions for the review

The user didn't decide these. Each one is marked in the requirement it belongs to, for the user to confirm or change at review.

- **A1 (Requirements 2 and 4):** `--SequentialMatching.quadratic_overlap`, 0 in the Near_Match_Run and 1 in the Far_Match_Run, and `--output_type COLMAP` in the Undistortion are Fixed_Constants, not Option_Keys. D6 defines the two Match_Runs by their `quadratic_overlap`, and the step reads the Undistortion's output in COLMAP's own layout, so these values are part of the step's design, like `--camera_mode 1`.
- **A2 (Requirements 2 and 12):** Each Option_Key holds the TOML type of its option's kind. An option that is on or off is a TOML bool, `true` or `false`, which the step passes as 1 or 0. A whole-number option is a TOML integer, and a decimal option a TOML integer or decimal, never a bool. A choice option is a TOML string, written as COLMAP takes it, like `"l1_root"` or `"CERES"`. Values that COLMAP reads as "none" or "no limit", like -1, stay as COLMAP's numbers.
- **A3 (Requirement 5):** A Calibration that exits with any non-zero code other than by a Timeout is a Failed_Calibration, a crash included, the same as the Global_Run's rule in D8. D7 named exit code 1.
- **A4 (Requirements 2 and 12):** Each COLMAP command the step runs has one Time_Limit key, and the `sequential_matcher` key's limit applies to each of the two Match_Runs.
- **A5 (Requirements 9 and 12):** The share of Placed_Frames that the Axis_Point must lie in front of is a key too, `axis_point_min_front_share = 0.5`, because `.kiro/steering/python.md` keeps every threshold in Config. D11 says "at least half".
- **A6 (Requirement 9):** "The median of the points" is the Point_Median: the median of the points' x, of their y and of their z, each taken separately along the levelled axes, after the rotation and the turn. A median taken along COLMAP's axes would give another point.
- **A7 (Requirement 6):** The user wasn't asked what happens when the Fallback itself fails. When the Incremental_Run exits with a non-zero code other than by a Timeout, the step fails with a Readable_Error that gives the exit code, the Last_Error_Line and the Filming_Advice. When it exits with code 0 and writes no Model, the Final_Model has 0 Placed_Frames, so the Placed_Check fails with its own Readable_Error.
- **A8 (Requirements 11 and 13):** A Step_Report's counts are whole numbers, so the mean reprojection error, the Level_Move and the reasons for a Fallback or a kept COLMAP_Frame go in the Solve_Record, `work/cameras/solve.json`, a work-folder record like the Frames_Step's `frames.json`. The step writes it after the Training_Images, the Training_Masks and the Levelled_Model, so a Work_Folder without it is unfinished, and writes the Viewer_Cameras after it, last of all.
- **A9 (Requirements 11 and 15):** The step removes an earlier `out/cameras.json` as soon as its own code starts, so a run that fails never leaves cameras from an earlier run next to a Work_Folder that the Step_Runner has emptied.
- **A10 (Requirement 2):** Each COLMAP_Command's output goes to a Command_Log of its own in the COLMAP_Folder, and `TMPDIR` points inside the COLMAP_Folder. So the logs stay with the Job for the team, and glog writes nothing to `/tmp`.
- **A11 (Requirement 1):** Before any COLMAP_Command runs, the step checks the Frames_Contract: that `frames.json` reads with the fields the step uses, that each Pick's `file` is the name the Frames_Contract gives its `number`, and that each Colour_Pick and Grey_Copy, and the Mask when `mask_used` is true, exists. So a broken `work/frames/` gives a clear error instead of a COLMAP failure.
- **A12 (Requirement 13):** The report's counts and timings have the names Requirement 13 gives them. A COLMAP_Command that fails or runs past its Time_Limit still gets its timing, so a failed report shows how long the failing command ran.
- **A13 (Requirement 11):** The Viewer_Cameras hold the Undistorted_Camera, whether the Levelling moved the scene, and each Placed_Frame's name and levelled Pose in COLMAP's convention. The viewer spec may ask for more.
- **A14 (Requirement 9):** Any error while the Levelling works out the Level_Move, one the code doesn't foresee included, keeps the COLMAP_Frame with a warning that names the error, because D11 says the Levelling never fails the Job. Reading the Undistorted_Model and writing the Levelled_Model aren't part of the Levelling, so their failures still fail the step.
- **A15 (Requirement 15):** The Licence_List's COLMAP row also covers Homebrew's COLMAP 4.2.1 on laptops, which holds SiftGPU, CGAL and LSD and runs the laptop tests with `use_gpu 0`, the integration test's whole solve included, as D18 allows. Its Installed column says COLMAP is on the laptop through Homebrew and on the machine image `ami-0fe14782b05fe63b2`, which Samarth says has COLMAP 4.2.1 and passed its smoke test on 23 Nov 2026.
- **A16 (Requirement 16):** A test checks every option the step passes against `colmap <command> -h`, skipped without COLMAP, so a mistyped option name shows up without a whole COLMAP run.

### Open design items

The design proposes these and the user reviews them. They aren't gaps in the requirements: each requirement that depends on one names it.

- **O1 (Requirement 12):** The Time_Limit of each COLMAP command.
- **O2 (Requirements 3 and 12):** The value of `fisheye_fov_degrees`.
- **O3 (Requirements 9 and 12):** The values of `axis_point_min_eigen_share` and of `heading_min_degrees`, the "nearly straight up or down" threshold.
- **O4 (Requirements 4 and 12):** The value of `far_overlap`, at least 9.
- **O5 (Requirement 5):** How the step detects a rejected camera and a Calibration with no pairs: from the log lines `N cameras rejected in view graph calibration` and `No image pairs to calibrate`, or by comparing the camera's focal length in the Database and in the Calibration_Copy.
- **O6 (Requirements 9 and 10):** How the Level_Move is applied and the Levelled_Model written: by our own Model_Writer, or by `colmap model_transformer`, with only its `cameras.bin`, `images.bin` and `points3D.bin` copied into `work/cameras/sparse/`, so `rigs.bin` and `frames.bin` stay out.
- **O7 (Requirement 12):** Whether the bundle adjustment backends `--GlobalMapper.ba_backend`, `--Mapper.ba_local_backend` and `--Mapper.ba_global_backend` are Fixed_Constants or Option_Keys that accept only `CERES`, since the CASPAR backends need CUDA.
- **O8 (Requirement 11):** The COLMAP_Folder's layout.
- **O9 (Requirement 6):** For `"RADIAL_FISHEYE"`, which database the Global_Run reads, since there's no Calibration_Copy.
- **O10 (Requirement 10):** The rest of each `images.bin` record (the name and the 2D points), which file tells whether a Model registers an image, and the whole layout of the three Model_Files, checked against `src/colmap/scene/reconstruction_io_binary.cc` at the 4.2.1 tag.
- **O11 (Requirements 11 and 12):** The names and comments of the Option_Keys and the Time_Limit keys, the range each Option_Key's check allows, and the exact fields of the Solve_Record and the Viewer_Cameras.

## Glossary

- **Cameras_Step**: the Step this spec builds, in `splatpipe/cameras/`. Its Step_Name is `cameras`, so its Config section is `[cameras]`, its Work_Folder is `work/cameras/` and its Step_Report is `out/reports/cameras.json`.
- **Frames_Contract**: the layout of `work/frames/` and the fields of the Frames_Record, as the frames spec's A10 and Requirement 8 and the Pipeline_Doc's frames section give them: the Colour_Picks in `images/`, the Grey_Copies in `grey/`, the Mask `mask.png` only when the Frames_Step uses one, and the Frames_Record `frames.json`, written last.
- **Pick_Number**: a Pick's `number` in the Frames_Record, its place in time order from 0.
- **Training_Step**: the step after this one, built in its own spec, `.kiro/specs/splat-training/`, which reads `work/cameras/`. Its Step_Name is `train`, and it's a Required_Step.
- **Clean_Step**: the step after the Training_Step, built in its own spec, `.kiro/specs/clean/`, which cleans the trained splat and makes the Viewer's copy. Its Step_Name is `clean`, and it's an Optional_Step.
- **Viewer**: the web viewer, built in its own spec, which reads the Viewer_Cameras.
- **COLMAP**: the COLMAP 4.2.1 command line program `colmap`, built for the machine image as `.kiro/steering/tech.md` says. On a laptop it's Homebrew's COLMAP 4.2.1, which runs the laptop tests, the integration test's whole solve included, always with `use_gpu 0` (D18).
- **COLMAP_Command**: one run of `colmap` with one of its commands, like `colmap feature_extractor`, and that run's options.
- **Option_Key**: a key in `[cameras]` that holds the value of one COLMAP option, with a comment that names the option (D4).
- **Fixed_Constants**: what stays in code, not in Config, because the build, the rules or the step's own design force it. This is the same exception that the frames spec's Fixed_Constants and the pipeline-core spec's Job_Id pattern make. They are:
  - the program name `colmap`;
  - `--FeatureExtraction.use_gpu 0`, `--FeatureMatching.use_gpu 0`, `--FeatureExtraction.type SIFT`, `--FeatureMatching.type SIFT_BRUTEFORCE` and `--SequentialMatching.loop_detection 0`;
  - `--GlobalMapper.gp_use_gpu 0`, `--GlobalMapper.ba_ceres_use_gpu 0` and `--Mapper.ba_use_gpu 0`;
  - `--log_target stderr`, `--log_level 0`, `--log_severity 0` and `--log_color 0`;
  - `--camera_mode 1`, and the paths;
  - `--SequentialMatching.quadratic_overlap` 0 in the Near_Match_Run and 1 in the Far_Match_Run, and `--output_type COLMAP` (A1);
  - the two Camera_Model names, the PINHOLE model of the Levelled_Model, the World_Up (0, −1, 0) and the +Z of the Heading;
  - the Mask's 0 and 255;
  - the layout of the Model_Files, and the file and folder names of the Work_Folder and of `out/cameras.json`.
- **Time_Limit**: the longest time a COLMAP_Command may run, in seconds: the value of its command's Time_Limit key in `[cameras]` (A4, O1), or the seconds left before the Cap_Deadline when fewer are left (D17, Requirement 17).
- **Timeout**: a COLMAP_Command running past its Time_Limit.
- **Last_Error_Line**: the last line that isn't empty in what a COLMAP_Command wrote to its standard error, where `--log_target stderr` sends all of COLMAP's output.
- **COLMAP_Folder**: `work/cameras/colmap/`, which holds COLMAP's own files: the Database, the Calibration_Copy, every Model, the output of the Undistortion and of the Mask_Warp, the camera file for the Mask_Warp, and the Command_Logs. Its layout is O8.
- **Command_Log**: the file in the COLMAP_Folder that holds one COLMAP_Command's output (A10).
- **Database**: the COLMAP database in the COLMAP_Folder that the Feature_Extraction makes and the Match_Runs fill.
- **Calibration_Copy**: a copy of the Database that the Calibration changes, so the Database stays as the Match_Runs left it.
- **Feature_Extraction**: the `colmap feature_extractor` run.
- **Match_Runs**: the Near_Match_Run and the Far_Match_Run.
- **Near_Match_Run**: the first `colmap sequential_matcher` run, with `--SequentialMatching.quadratic_overlap 0` and `--SequentialMatching.overlap` from `near_overlap`, which pairs each Pick with the next `near_overlap` Picks in name order.
- **Far_Match_Run**: the second `colmap sequential_matcher` run, on the same Database, with `--SequentialMatching.quadratic_overlap 1` and `--SequentialMatching.overlap` from `far_overlap`, which pairs each Pick i with Pick i + 2^k for k from 0 to `far_overlap` − 1, and matches only the pairs the Database doesn't hold yet.
- **Camera_Model**: the value of the Shared_Setting `camera_model`, `"RADIAL"` or `"RADIAL_FISHEYE"` (D5, D19).
- **Fisheye_Start**: the starting camera for `"RADIAL_FISHEYE"`, worked out from `fisheye_fov_degrees` (Requirement 3, criterion 4).
- **Calibration**: the `colmap view_graph_calibrator` run on the Calibration_Copy.
- **Failed_Calibration**: a Calibration that exits with a non-zero code other than by a Timeout, rejects the camera, or finds no image pairs to calibrate (D7, A3).
- **Global_Run**: the `colmap global_mapper` run.
- **Incremental_Run**: the `colmap mapper` run, which runs only as the Fallback.
- **Fallback**: the one Incremental_Run that the step makes after a Failed_Calibration or after the Global_Run gives up (Requirement 6, criteria 3 to 5).
- **Model**: one numbered folder, `0/`, `1/` and so on, that a mapper writes into its output folder, holding one reconstruction.
- **Placed_Frame**: a Pick that a Model registers, so it has a camera pose in that Model.
- **Kept_Model**: of the Models that one mapper run writes, the one with the most Placed_Frames, and the one with the lowest Model number among those with equally many.
- **Final_Model**: the Kept_Model the step goes on with: the Global_Run's when the Global_Run didn't give up, and the Incremental_Run's when the Fallback ran.
- **Placed_Check**: the check that a Kept_Model has at least `min_placed_frames` Placed_Frames and at least `min_placed_share` times the number of Picks.
- **Filming_Advice**: the words that tell the person how to film better: move slower, hold the phone steadier, overlap more between views, and film surfaces with more texture.
- **Placed_Failure**: the Expected_Failure for a Final_Model that fails the Placed_Check (Requirement 7).
- **Undistortion**: the `colmap image_undistorter` run that warps the Final_Model's Colour_Picks to the Undistorted_Camera.
- **Undistorted_Model**: the model the Undistortion writes into the `sparse/` of its output folder: the Undistorted_Camera, the Placed_Frames with the Final_Model's Poses, and the Final_Model's points.
- **Undistorted_Camera**: the one PINHOLE camera of the Undistorted_Model.
- **Mask_Warp**: the `colmap image_undistorter_standalone` run that warps the Mask the way the Undistortion warps the Colour_Picks.
- **Undistorted_Mask**: the Mask after the Mask_Warp and the threshold of Requirement 8, criterion 4, at the Undistorted_Camera's size.
- **Pose**: a camera's cam_from_world rotation R and translation t, as COLMAP holds them. A world point x is at R x + t in the camera's coordinates, where the camera looks along its +z axis, with x to the right and y down. The camera's centre is at −Rᵀt.
- **COLMAP_Frame**: the coordinates that the Undistorted_Model's Poses and points come in from COLMAP.
- **Up_Vector**: for one camera whose Pose has the rotation R, Rᵀ(0, −1, 0): the direction that is up in the camera's picture, in the model's world.
- **View_Direction**: for one camera whose Pose has the rotation R, Rᵀ(0, 0, 1): the direction the camera looks, in the model's world.
- **Camera_Axis**: the line through a camera's centre along its View_Direction.
- **Mean_Up**: the mean of every Placed_Frame's Up_Vector.
- **World_Up**: (0, −1, 0), which the Levelling makes the up direction of the scene.
- **Heading**: a camera's View_Direction after the rotation of Requirement 9, criterion 4, with its part along the World_Up taken away. A camera has a Heading when that View_Direction is at least `heading_min_degrees` from straight up and from straight down.
- **Heading_Camera**: the Placed_Frame with the lowest Pick_Number that has a Heading.
- **Axis_Point**: the point with the smallest sum of squared distances to the Placed_Frames' Camera_Axes.
- **Point_Median**: the median of the Undistorted_Model's points' x, of their y and of their z, each taken separately, after the rotation and the turn of Requirement 9, criteria 4 and 5 (A6).
- **Scene_Centre**: the Axis_Point after the rotation and the turn of Requirement 9, criteria 4 and 5, when the Axis_Point is well defined (Requirement 9, criterion 8), and the Point_Median otherwise.
- **Levelling**: the part of the Cameras_Step that works out the Level_Move from the Undistorted_Model and applies the Level_Move to every Pose and point (Requirement 9).
- **Level_Move**: the one rigid move, a rotation then a translation with scale 1, that takes the COLMAP_Frame to the Levelled_Model's frame. It's the identity when the Levelling keeps the COLMAP_Frame (D11).
- **Model_Files**: `cameras.bin`, `images.bin` and `points3D.bin`, in COLMAP 4.2.1's binary layout.
- **Model_Reader**: our own code that reads a Model's cameras, images and points.
- **Model_Writer**: our own code that writes Model_Files.
- **Levelled_Model**: `work/cameras/sparse/`, the Undistorted_Model moved by the Level_Move, as the three Model_Files (D2).
- **Training_Images**: the files `work/cameras/images/<name>`, one undistorted colour image for each Placed_Frame.
- **Training_Masks**: the files `work/cameras/masks/<stem>.png`, one copy of the Undistorted_Mask for each Placed_Frame, only when the Frames_Record has `mask_used`.
- **Solve_Record**: `work/cameras/solve.json`, which says how the cameras were solved and levelled (A8).
- **Viewer_Cameras**: `out/cameras.json`, the copy of the levelled cameras for the Viewer (D3).
- **Cameras_Report**: the Cameras_Step's Step_Report, with the fields Requirement 13 lists.
- **Step_Needs**: the Cameras_Step's `min_vcpus` and `min_memory_gb` in `[cameras]`, the cloud-jobs spec's name for a step's needs. The Start_Command reads the two keys for the Job's Preset to pick the machine. Memory is a hard need and vCPUs a soft one, so the step can run on a type from a Preset's Fallback_List with as few as 8 vCPUs. `[cameras]` has no `min_gpu_memory_gb`, because the step needs no GPU, and a missing key means no need.
- **Cameras_Smoke_Check**: the Smoke_Test's `cameras` check, whose code is in `splatpipe/cameras/smoke.py` and which `cloud/smoke_runner.py` runs on the machine image: the Option_Check, then the Smoke_Solve (D15, Requirement 18).
- **Option_Check**: the check that `colmap <command> -h` lists every option that the Cameras_Step's command lines pass that command.
- **Smoke_Picks**: the made-up Picks that `splatpipe/cameras/smoke.py` renders of a textured room corner, `cameras_frames` of them at `cameras_width` × `cameras_height` pixels, through a pinhole camera with f = `cameras_focal_pixels` and no distortion that moves along a level path with no pitch and no roll, with a black corner painted into each and a Mask that blacks it out. They're the integration test's scene, moved out of `tests/test_cameras_real.py`.
- **Smoke_Solve**: the Cameras_Step's own code, run with the real COLMAP on the Smoke_Picks, in a Job_Folder inside the Cameras_Smoke_Check's scratch folder that has no Job_Record file and no Jobs_Folder around it.

Terms this spec uses as the frames spec defines them: Frames_Step, Frames_Record, Pick, Colour_Pick, Grey_Copy, Mask and Pick_Size.

Terms this spec uses as the cloud-jobs spec defines them: Cloud_Runner, Start_Command, Time_Cap, Cap_Deadline, Fallback_List and Smoke_Test.

Terms this spec uses as the aws-setup spec defines them: Job_Bucket.

Terms this spec uses as the pipeline-core spec and the Pipeline_Doc define them: Pipeline, Skeleton, Step, Step_Name, Job, Job_Id, Jobs_Folder, Job_Folder, Job_Record, Step_Runner, Step_Report, Step_List, Required_Step, Optional_Step, Fake_Step, Work_Folder, Config, Preset, Shared_Setting, Preset_Setting, Step_Section, Expected_Failure, Unexpected_Error, Readable_Error, Job_Runner, Job_Status, JSON_Value, Pipeline_Doc, Licence_List, CLI and Test_Suite.

## Requirements

### Requirement 1: Reading the frames step's files

**User Story:** As a team member building the camera solve, I want the step to read the Frames_Step's files through the Frames_Contract, so that the cameras come from exactly the Picks the Frames_Step chose.

**Assumptions:** A11 (criteria 1, 3 and 4).

#### Acceptance Criteria

1. THE Cameras_Step SHALL read the Frames_Record from `work/frames/frames.json` and check the Frames_Contract, as criteria 2 to 4 say, before any COLMAP_Command runs (A11).
2. IF `work/frames/frames.json` doesn't exist, THEN THE Cameras_Step SHALL fail with an Expected_Failure whose Readable_Error says that the Frames_Step hasn't finished for this Job, and says to run the Frames_Step first with `uv run python -m splatpipe.frames --job <id>`.
3. IF the Frames_Record can't be read as JSON, or one of its fields `pick_width`, `pick_height`, `mask_used` and `picks` is missing or holds the wrong kind of value, or `picks` holds no Pick, or a Pick lacks its `number` or its `file`, or a Pick's `file` isn't the name the Frames_Contract gives the Pick's `number`, THEN THE Cameras_Step SHALL fail with an Expected_Failure whose Readable_Error names the field and says to run the Frames_Step again (A11).
4. IF a Pick's Colour_Pick or Grey_Copy is missing, or the Frames_Record's `mask_used` is true and the Mask is missing, THEN THE Cameras_Step SHALL fail with an Expected_Failure whose Readable_Error names the missing file and says to run the Frames_Step again (A11).
5. THE Cameras_Step SHALL give COLMAP the Grey_Copies' folder for the Feature_Extraction, and the Colour_Picks' folder for both mappers' point colours and for the Undistortion.
6. THE Cameras_Step SHALL take the number of Picks as the number of entries in the Frames_Record's `picks`, and match each image that COLMAP names to the Pick whose `file` is that name, with that Pick's Pick_Number.
7. THE Cameras_Step SHALL leave every file in `work/frames/` unchanged.

Note: COLMAP knows an image by its path from `--image_path`, so a Grey_Copy and its Colour_Pick are both `000000.jpg` to COLMAP. That's what lets the Feature_Extraction read the Grey_Copies while the mappers and the Undistortion read the Colour_Picks, which have the same names and size.

### Requirement 2: Running COLMAP

**User Story:** As a team member, I want every COLMAP command run the same safe way, with every option written out and a time limit, so that no COLMAP default, stray file or hung process decides what happens.

**Decided:** D4 (criteria 3 to 5), D8 (criterion 8) and D12 (criteria 1, 6 and 7). **Assumptions:** A1 (criterion 3), A2 (criterion 4), A4 (criterion 8) and A10 (criteria 2 and 10).

#### Acceptance Criteria

1. THE Cameras_Step SHALL run COLMAP only as the separate program `colmap`, through Python's `subprocess`, and import no COLMAP code and no pycolmap (D12).
2. THE Cameras_Step SHALL run each COLMAP_Command as an argument list with no shell, with standard input closed, with every path as an absolute path, and with the environment variable `TMPDIR` set to a folder inside the COLMAP_Folder, so the log files that glog writes before COLMAP has read its options stay in the Work_Folder (A10).
3. THE Cameras_Step SHALL pass each COLMAP_Command the Fixed_Constants of its command, and every other option that `.kiro/skills/colmap-cli/references/options.md` gives for that command, each option exactly once and with the value of its Option_Key in `[cameras]`, so no COLMAP default decides anything (D4, A1).
4. THE Cameras_Step SHALL write each Option_Key's value on the command line so that COLMAP reads the same value: a bool as `1` or `0`, a whole number in decimal, a decimal number with enough digits to give back the same float, and a name exactly as the Option_Key holds the name (A2).
5. THE Cameras_Step SHALL pass `--default_random_seed` to every COLMAP_Command with the value of the option's one Option_Key, and pass each undistortion setting to both the Undistortion and the Mask_Warp with the value of the setting's one Option_Key (D4).
6. THE Cameras_Step SHALL run SIFT features and `SIFT_BRUTEFORCE` matching only, with loop detection off and with no vocabulary tree, no model weights and no address to download from, so nothing is downloaded while the step runs (D12).
7. IF the `colmap` program isn't on the PATH, THEN THE Cameras_Step SHALL fail with an Expected_Failure, before any COLMAP_Command runs, whose Readable_Error says that COLMAP isn't installed and says to install COLMAP 4.2.1, built as `.kiro/steering/tech.md` says, and run the step again (D12).
8. IF a COLMAP_Command runs past a Time_Limit that its Time_Limit key in `[cameras]` set, THEN THE Cameras_Step SHALL stop the command and fail with an Expected_Failure, with no Fallback, whose Readable_Error names the command, gives the Time_Limit in seconds and names the Time_Limit key in `[cameras]` that sets the Time_Limit (D8, A4).
9. IF a COLMAP_Command other than the Calibration, the Global_Run and the Incremental_Run exits with a non-zero code, THEN THE Cameras_Step SHALL fail with an Expected_Failure whose Readable_Error names the command, gives its exit code and its Last_Error_Line, and says to check that COLMAP 4.2.1 is installed and to run the step again.
10. THE Cameras_Step SHALL save each COLMAP_Command's output to a Command_Log of its own in the COLMAP_Folder, also when the command fails or runs past its Time_Limit (A10).
11. THE Cameras_Step SHALL run the COLMAP_Commands one at a time, each after the one before has exited, so at most one COLMAP process has the Database or the Calibration_Copy open.
12. THE Cameras_Step SHALL run each of the Feature_Extraction, the Near_Match_Run, the Far_Match_Run, the Calibration, the Global_Run, the Incremental_Run, the Undistortion and the Mask_Warp at most once in a run of the step, with no retry.
13. FOR ALL settings that the checks of Requirement 12 accept, both values of the Frames_Record's `mask_used` and both Camera_Models, THE Cameras_Step SHALL build each command line with each Fixed_Constant of its command at the Fixed_Constant's value, with every other option of criterion 3 exactly once and at its Option_Key's value, written as criterion 4 says, and with no option for loop detection, a vocabulary tree, or a learned feature or matcher (property).

Note: Requirement 5 says what a non-zero exit of the Calibration does, and Requirement 6 what one of the Global_Run or the Incremental_Run does. Requirement 17 says what happens when the Cap_Deadline sets a Time_Limit, or leaves no time to start a command.

### Requirement 3: Finding features

**User Story:** As someone who films a place, I want the step to find features on the evened-out grey copies, with one camera for every Pick, so that every Pick is described with the same lens and the stuck pixels are left out.

**Decided:** D4 (criterion 1), D5 (criteria 2 to 4) and D19 (criteria 3 and 4). **Open for the design:** O2 (criterion 4).

#### Acceptance Criteria

1. WHEN the checks of Requirement 1 pass, THE Cameras_Step SHALL run the Feature_Extraction into a new Database, on the Grey_Copies' folder, with `--camera_mode 1`, `--FeatureExtraction.type SIFT` and `--FeatureExtraction.use_gpu 0` (D4).
2. THE Cameras_Step SHALL give every Pick the same one camera, of the Camera_Model, through `--ImageReader.camera_model` (D5).
3. WHERE the Camera_Model is `"RADIAL"`, THE Cameras_Step SHALL leave `--ImageReader.camera_params` out, so the camera's focal length f starts at `--ImageReader.default_focal_length_factor` times the long edge of the Pick_Size, with no focal prior (D5, D19).
4. WHERE the Camera_Model is `"RADIAL_FISHEYE"`, THE Cameras_Step SHALL pass `--ImageReader.camera_params` with the Fisheye_Start, in the order f, cx, cy, k1, k2, each number written as Requirement 2, criterion 4 says: f = (L / 2) / (θ / 2), where L is the larger of `pick_width` and `pick_height` and θ is `fisheye_fov_degrees` in radians, cx = `pick_width` / 2, cy = `pick_height` / 2, and k1 = k2 = 0 (D5, D19, O2).
5. WHERE the Frames_Record's `mask_used` is true, THE Cameras_Step SHALL pass the Mask to the Feature_Extraction as `--ImageReader.camera_mask_path`, so COLMAP drops the features on the Mask's black pixels.
6. WHERE the Frames_Record's `mask_used` is false, THE Cameras_Step SHALL leave `--ImageReader.camera_mask_path` out.

### Requirement 4: Matching the picks

**User Story:** As someone who films a place, I want each Pick matched with its near neighbours and with a few far ones, so that the walk holds together from start to end.

**Decided:** D6 (criteria 1 to 4). **Assumptions:** A1 (criteria 1 and 2). **Open for the design:** O4 (criterion 4).

#### Acceptance Criteria

1. WHEN the Feature_Extraction is done, THE Cameras_Step SHALL run the Near_Match_Run on the Database, with `--SequentialMatching.quadratic_overlap 0` and with `--SequentialMatching.overlap` set to the Preset_Setting `near_overlap` (D6, A1).
2. WHEN the Near_Match_Run is done, THE Cameras_Step SHALL run the Far_Match_Run on the same Database, with `--SequentialMatching.quadratic_overlap 1` and with `--SequentialMatching.overlap` set to `far_overlap`, so COLMAP adds only the far pairs that the Near_Match_Run didn't match (D6, A1).
3. THE Cameras_Step SHALL run both Match_Runs with `--FeatureMatching.type SIFT_BRUTEFORCE`, `--FeatureMatching.use_gpu 0` and `--SequentialMatching.loop_detection 0`, and with the same value of every other matching option.
4. THE real Config SHALL hold a `far_overlap` of at least 9, so the Far_Match_Run reaches the Pick 256 places ahead, the farthest power of two inside the 400 Picks that `max_picks` in `[frames]` allows (D6, O4).

### Requirement 5: Calibrating the focal length

**User Story:** As someone who films with a phone, I want the step to work out the lens's focal length from the matches, so that the global mapper has a focal length although the video carries no EXIF.

**Decided:** D5 (criterion 3), D7 (criteria 1 to 5) and D19 (criteria 1 and 3). **Assumptions:** A3 (criterion 4). **Open for the design:** O5 (criterion 4).

#### Acceptance Criteria

1. WHERE the Camera_Model is `"RADIAL"`, WHEN the Far_Match_Run has exited, THE Cameras_Step SHALL copy the Database to the Calibration_Copy and run the Calibration on the Calibration_Copy (D7, D19).
2. THE Cameras_Step SHALL keep the Database as the Match_Runs left the Database, for the Incremental_Run (D7, D8).
3. WHERE the Camera_Model is `"RADIAL_FISHEYE"`, THE Cameras_Step SHALL run no Calibration, because `view_graph_calibrator` only fits pinhole-type cameras, and add a warning to the Cameras_Report that says the Calibration was skipped for the fisheye camera, which starts from `fisheye_fov_degrees` (D5, D19).
4. THE Cameras_Step SHALL count as a Failed_Calibration each Calibration that exits with a non-zero code other than by a Timeout, each Calibration that rejects the camera, and each Calibration that finds no image pairs to calibrate, and SHALL detect the last two, which COLMAP 4.2.1 reports with exit code 0, in the way the design chooses (D7, A3, O5).
5. IF a Failed_Calibration happens, THEN THE Cameras_Step SHALL run no Global_Run, run the Incremental_Run as the Fallback, and add a warning to the Cameras_Report that says why the Calibration failed and that the incremental mapper ran instead (D7).

Note: COLMAP 4.2.1 marks a rejected camera's focal length as trusted after putting the focal length back to its starting value, so an exit code of 0 doesn't show that the Calibration worked.

### Requirement 6: Mapping and the fallback

**User Story:** As someone who films a place, I want the fast global mapper tried first and the incremental mapper tried once when the global mapper gives up, so that a hard video still gets cameras without the Job running on and on.

**Decided:** D7 (criterion 6), D8 (criteria 1 to 12) and D19 (criteria 1 and 2). **Assumptions:** A7 (criteria 10 and 11). **Open for the design:** O9 (criterion 2).

#### Acceptance Criteria

1. WHERE the Camera_Model is `"RADIAL"`, WHEN the Calibration has exited and isn't a Failed_Calibration, THE Cameras_Step SHALL run the Global_Run on the Calibration_Copy, with the Colour_Picks' folder as `--image_path`, with `--GlobalMapper.gp_use_gpu 0` and with `--GlobalMapper.ba_ceres_use_gpu 0` (D8).
2. WHERE the Camera_Model is `"RADIAL_FISHEYE"`, WHEN the Far_Match_Run has exited, THE Cameras_Step SHALL run the Global_Run with no Calibration, on the database that the design chooses, with the options of criterion 1 (O9).
3. IF the Global_Run exits with a non-zero code other than by a Timeout, such as code 1 after `Failed to create sparse model`, or a crash such as code 134, which Python's `subprocess` gives as -6, THEN THE Cameras_Step SHALL run the Fallback (D8).
4. IF the Global_Run exits with code 0 and writes no Model, THEN THE Cameras_Step SHALL run the Fallback (D8).
5. IF the Global_Run's Kept_Model fails the Placed_Check, THEN THE Cameras_Step SHALL run the Fallback (D8).
6. WHEN the Cameras_Step runs the Fallback, THE Cameras_Step SHALL run the Incremental_Run once, on the Database and never on the Calibration_Copy, with the Colour_Picks' folder as `--image_path` and with `--Mapper.ba_use_gpu 0` (D7, D8).
7. WHEN the Global_Run gives up, as criteria 3 to 5 say, THE Cameras_Step SHALL add a warning to the Cameras_Report that gives the reason, either the Global_Run's exit code and Last_Error_Line, or that the Global_Run wrote no Model, or the Kept_Model's number of Placed_Frames next to the number the Placed_Check needs, and says that the incremental mapper ran instead (D8).
8. THE Cameras_Step SHALL choose each mapper run's Kept_Model from that run's Models: the Model with the most Placed_Frames, and the Model with the lowest Model number among those with equally many (D8).
9. THE Cameras_Step SHALL take the Final_Model from the Incremental_Run's Models when the Fallback ran, and from the Global_Run's Models otherwise (D8).
10. IF the Incremental_Run exits with a non-zero code other than by a Timeout, THEN THE Cameras_Step SHALL fail with an Expected_Failure whose Readable_Error says that the incremental mapper stopped with an error, gives its exit code and its Last_Error_Line, and gives the Filming_Advice (A7).
11. IF the Incremental_Run exits with code 0 and writes no Model, THEN THE Cameras_Step SHALL count 0 Placed_Frames for the Final_Model, so the Placed_Check fails (A7).
12. FOR ALL lists of at least one Model, each with any number of Placed_Frames, THE Cameras_Step SHALL choose a Kept_Model that has at least as many Placed_Frames as every other Model, and a lower Model number than every other Model with as many (property).

Note: Requirement 2, criterion 8, and Requirement 17, criteria 5 and 6, cover a Global_Run or an Incremental_Run that runs past its Time_Limit or has no time left to start: the step fails, with no Fallback.

### Requirement 7: The placed-frames check

**User Story:** As someone who films a place, I want the Job to stop with clear advice when too few frames could be placed, so that I know to film again rather than get a broken splat.

**Decided:** D9 (criteria 1 to 3).

#### Acceptance Criteria

1. THE Placed_Check SHALL pass a Kept_Model whose Placed_Frames number at least `min_placed_frames` and at least `min_placed_share` times the number of Picks, and fail every other Kept_Model (D9).
2. IF the Final_Model fails the Placed_Check, THEN THE Cameras_Step SHALL fail with the Placed_Failure, an Expected_Failure, before the Undistortion runs (D9).
3. THE Placed_Failure's Readable_Error SHALL give how many Picks were placed, out of how many Picks, and the fewest Placed_Frames the step needs, and SHALL give the Filming_Advice: move slower, hold the phone steadier, overlap more between views, and film surfaces with more texture (D9).

### Requirement 8: Undistorting the picks and the mask

**User Story:** As a team member building the training, I want the Picks and the Mask warped to one camera with no distortion, so that the training can use a plain pinhole camera.

**Decided:** D2 (criterion 1) and D10 (criteria 1 to 7).

#### Acceptance Criteria

1. WHEN the Final_Model passes the Placed_Check, THE Cameras_Step SHALL run the Undistortion on the Final_Model, with the Colour_Picks' folder as `--image_path`, with `--output_type COLMAP` and with an output folder inside the COLMAP_Folder, so the Undistortion's own `images/` and `sparse/` land in the COLMAP_Folder (D2, D10).
2. IF the Undistorted_Model holds other than exactly one camera, a camera that isn't PINHOLE, or an image that isn't a Placed_Frame of the Final_Model, or the Undistortion leaves out a Placed_Frame's image, THEN THE Cameras_Step SHALL fail with an Expected_Failure whose Readable_Error says what the Undistortion got wrong and says to run the step again.
3. WHERE the Frames_Record's `mask_used` is true, WHEN the Undistortion is done, THE Cameras_Step SHALL run the Mask_Warp on the Mask, with the Final_Model's camera as the Final_Model holds the camera before the Undistortion, and with the same value of every undistortion setting as the Undistortion (D10).
4. WHEN the Mask_Warp is done, THE Cameras_Step SHALL make the Undistorted_Mask from the Mask_Warp's output, as one 8-bit grey channel that holds 0 (ignore) for each pixel that is below 255 in any channel, and 255 (use) for each pixel that is 255 in every channel (D10).
5. IF the Mask_Warp's output can't be read, or isn't at the Undistorted_Camera's width and height, THEN THE Cameras_Step SHALL fail with an Expected_Failure whose Readable_Error names the file, says what's wrong and says to run the step again.
6. THE real Config SHALL hold 0 in the Option_Key of `--blank_pixels`, which the Undistortion and the Mask_Warp both use, so neither output has blank edges (D10).
7. FOR ALL 8-bit images with three channels, THE Cameras_Step SHALL give an Undistorted_Mask that holds only 0 and 255 and holds 255 exactly where every channel is 255 (property).

Note: since D20, the settings check keeps both `ba_refine_extra_params` keys true (Requirement 12, criterion 16). So a camera that isn't PINHOLE in criterion 2 can come only from a bundle adjustment that left every distortion parameter within 1e-8 of 0, which COLMAP's undistorter copies unwarped. Criterion 2's Readable_Error says so and says to run the step again. From the RADIAL camera, the Undistortion gives a PINHOLE camera with fx = fy = f (D19).

### Requirement 9: Levelling the scene

**User Story:** As someone looking at the splat, I want the scene the right way up and in the middle of the view, so that the Viewer and the training start from a level, centred scene.

**Decided:** D11 (criteria 1 to 18). **Assumptions:** A5 (criterion 8), A6 (criterion 9) and A14 (criterion 12). **Open for the design:** O3 (criteria 6 and 8) and O6 (criterion 13).

#### Acceptance Criteria

1. WHEN the Undistortion, and the Mask_Warp when the Mask_Warp runs, are done, THE Levelling SHALL work out the Level_Move from the Poses of the Undistorted_Model's Placed_Frames and from the Undistorted_Model's points, and apply the Level_Move to every Pose and every point of the Undistorted_Model, with scale 1 (D11).
2. THE Levelling SHALL work out the Mean_Up from every Placed_Frame's Up_Vector, Rᵀ(0, −1, 0), with each Placed_Frame counted once (D11).
3. IF the Mean_Up is shorter than `up_agreement`, THEN THE Levelling SHALL keep the COLMAP_Frame, with a warning in the Cameras_Report that gives the Mean_Up's length and `up_agreement` and says the scene was left in COLMAP's frame, which may be tilted (D11).
4. WHEN the Mean_Up is at least `up_agreement` long, THE Levelling SHALL rotate the scene so the Mean_Up points along the World_Up, (0, −1, 0) (D11).
5. WHEN the Levelling has rotated the Mean_Up to the World_Up, THE Levelling SHALL turn the scene about the World_Up so the Heading_Camera's Heading points along +Z (D11).
6. THE Levelling SHALL take as the Heading_Camera the first Placed_Frame, by Pick_Number, when that Placed_Frame has a Heading, and otherwise the next Placed_Frame, by Pick_Number, that has a Heading (D11, O3).
7. IF no Placed_Frame has a Heading, THEN THE Levelling SHALL keep the COLMAP_Frame, with a warning in the Cameras_Report that says every camera looks nearly straight up or down, so the scene was left in COLMAP's frame (D11).
8. THE Levelling SHALL count the Axis_Point as well defined when the smallest eigenvalue of Σ(I − d dᵀ), summed over the Placed_Frames' View_Directions d, is at least `axis_point_min_eigen_share` times the largest eigenvalue, and the Axis_Point lies in front of at least `axis_point_min_front_share` of the Placed_Frames, where a point p lies in front of a camera with the centre c and the View_Direction d when (p − c) · d > 0 (D11, A5, O3).
9. THE Levelling SHALL take the Axis_Point, after the rotation and the turn, as the Scene_Centre when the Axis_Point is well defined, and the Point_Median otherwise (D11, A6).
10. IF the Axis_Point isn't well defined and the Undistorted_Model holds no points, THEN THE Levelling SHALL keep the COLMAP_Frame, with a warning in the Cameras_Report that says the scene's centre couldn't be found, so the scene was left in COLMAP's frame (D11).
11. WHEN the Levelling has turned the scene, THE Levelling SHALL move every camera and every point by the one translation that takes the Scene_Centre to the origin (D11).
12. IF working out the Level_Move raises any other error, or gives a value that isn't a finite number, THEN THE Levelling SHALL keep the COLMAP_Frame, with a warning in the Cameras_Report that names the error and says the scene was left in COLMAP's frame (D11, A14).
13. THE Levelling SHALL either apply the whole Level_Move, its rotation, its turn and its translation together, or keep the COLMAP_Frame, so no part of the Level_Move is applied without the rest (D11, O6).
14. THE Cameras_Step SHALL write the Levelled_Model both when the Levelling moves the scene and when the Levelling keeps the COLMAP_Frame, so the Levelling never fails the Job (D11).
15. FOR ALL sets of Poses and points, THE Levelling SHALL give a Level_Move that keeps the distance between every two of the camera centres and the points, within a tolerance for floating point (property).
16. FOR ALL sets of Poses and points, THE Levelling SHALL give a Level_Move whose rotation R has RᵀR = I and a determinant of 1, so its scale is 1, within a tolerance for floating point (property).
17. FOR ALL sets of Poses and points whose Mean_Up is at least `up_agreement` long, with at least one Placed_Frame that has a Heading and with a Scene_Centre, THE Levelling SHALL give a Levelled_Model whose Mean_Up points along (0, −1, 0), within a tolerance for floating point (property).
18. FOR ALL sets of Poses and points of criterion 17, THE Levelling SHALL give a Levelled_Model in which the Heading_Camera's Heading points along +Z and the Scene_Centre, worked out again, is at the origin, within a tolerance for floating point (property).

Note for the design: the rotation that takes the Mean_Up to the World_Up isn't unique when the Mean_Up points along (0, 1, 0). Every such rotation gives the same Level_Move, because the turn of criterion 5 then fixes the Heading.

### Requirement 10: The model files

**User Story:** As a team member building the training, I want the levelled cameras and points in COLMAP's own binary files, so that my small reader can read them without COLMAP installed.

**Decided:** D2 (criteria 2 and 5 to 8), D12 (criteria 1 to 3) and D19 (criteria 1 and 9). **Open for the design:** O6 (criterion 5) and O10 (criteria 1 and 2).

#### Acceptance Criteria

1. THE Model_Reader SHALL read the cameras, images and points of a Model that COLMAP 4.2.1 wrote, and of a folder that holds only the three Model_Files, for cameras of the models PINHOLE, RADIAL and RADIAL_FISHEYE, and SHALL tell which images the Model registers (O10, D19).
2. THE Model_Writer SHALL write `cameras.bin`, `images.bin` and `points3D.bin` byte for byte as COLMAP 4.2.1's `WriteBinary` writes those three files, in the layout checked against `src/colmap/scene/reconstruction_io_binary.cc` at the 4.2.1 tag (D2, O10).
3. THE Model_Reader and the Model_Writer SHALL be our own code, written fresh with the Python standard library and numpy, with nothing copied from COLMAP's scripts or from another repo (D12).
4. IF a Model_File is missing, ends too soon, or holds a value that the layout doesn't allow, THEN THE Cameras_Step SHALL fail with an Expected_Failure whose Readable_Error names the file and what's wrong, and says to check that COLMAP 4.2.1 is installed and to run the step again.
5. THE Levelled_Model SHALL be the folder `work/cameras/sparse/`, holding exactly `cameras.bin`, `images.bin` and `points3D.bin` in the layout of criterion 2, with no `rigs.bin`, no `frames.bin` and no other file (D2, O6).
6. THE Levelled_Model's `cameras.bin` SHALL hold exactly one camera, the Undistorted_Camera, of the model PINHOLE, and every image in the Levelled_Model's `images.bin` SHALL use that camera (D2).
7. THE Levelled_Model's `images.bin` SHALL hold one image for each Placed_Frame, with the image id, the camera id and the 2D points that the Undistorted_Model holds for that image, the image's Pose moved by the Level_Move, and the image's name, the Pick's `file`, like `000000.jpg` (D2).
8. THE Levelled_Model's `points3D.bin` SHALL hold every point of the Undistorted_Model, with the id, the colour, the error and the track that the Undistorted_Model holds for that point, and the point's position moved by the Level_Move (D2).
9. FOR ALL models whose cameras are of the models PINHOLE, RADIAL and RADIAL_FISHEYE, with any images that have any Poses, names and 2D points, and any points that have any colours, errors and tracks, THE Model_Reader SHALL read back from the Model_Writer's files a model equal to the one written (round-trip property).

### Requirement 11: The work folder and out/

**User Story:** As a team member building the training or the Viewer, I want the step's files in one known layout, so that my step can read them without guessing.

**Decided:** D2 (criteria 1 to 5), D3 (criteria 8 to 10) and D16 (the note after criterion 13). **Assumptions:** A8 (criteria 6 to 8), A9 (criterion 10) and A13 (criterion 9). **Open for the design:** O8 (criterion 5) and O11 (criteria 7 and 9).

#### Acceptance Criteria

1. THE Cameras_Step SHALL lay out the Work_Folder as the contract the Training_Step reads: `work/cameras/colmap/` (the COLMAP_Folder), `work/cameras/images/` (the Training_Images), `work/cameras/masks/` (the Training_Masks, only when the Frames_Record's `mask_used` is true), `work/cameras/sparse/` (the Levelled_Model) and `work/cameras/solve.json` (the Solve_Record) (D2).
2. THE Cameras_Step SHALL save each Placed_Frame's colour image from the Undistortion as `work/cameras/images/<name>`, where `<name>` is the Placed_Frame's name in the Levelled_Model's `images.bin`, and SHALL hold no other file in `work/cameras/images/` (D2).
3. WHERE the Frames_Record's `mask_used` is true, THE Cameras_Step SHALL save a copy of the Undistorted_Mask as `work/cameras/masks/<stem>.png` for each Placed_Frame, where `<stem>` is the Placed_Frame's name without `.jpg`, like `000000.png` (D2).
4. WHERE the Frames_Record's `mask_used` is false, THE Cameras_Step SHALL write no `work/cameras/masks/` folder (D2).
5. THE Cameras_Step SHALL keep COLMAP's own files in the COLMAP_Folder, the Undistortion's `images/` and `sparse/` included, and write none of COLMAP's own files into `work/cameras/sparse/` (D2, O8).
6. THE Cameras_Step SHALL write the Solve_Record after the Training_Images, the Training_Masks and the Levelled_Model, so a run that fails in the step's own code leaves no Solve_Record, and the Training_Step can treat a Work_Folder without a Solve_Record as unfinished (A8).
7. THE Solve_Record SHALL hold only JSON_Values: the Camera_Model; whether the Calibration ran, and why when it was a Failed_Calibration; which mapper made the Final_Model, and why when the Fallback ran; the number of Placed_Frames in each Model of the mapper run that made the Final_Model, and the Final_Model's Model number; the Final_Model's mean reprojection error, the mean of the errors that the Final_Model's `points3D.bin` holds, in pixels; whether the Levelling moved the scene, and why when the Levelling kept the COLMAP_Frame; the Mean_Up's length; which Scene_Centre was used; the Heading_Camera's name; and the Level_Move's rotation and translation (A8, O11).
8. WHEN the Solve_Record is written, THE Cameras_Step SHALL write the Viewer_Cameras to `out/cameras.json`, last of all (D3, A8).
9. THE Viewer_Cameras SHALL hold only JSON_Values: the Undistorted_Camera's width, height, fx, fy, cx and cy; whether the Levelling moved the scene; and, for each Placed_Frame in Pick_Number order, the Placed_Frame's name and the Placed_Frame's Pose from the Levelled_Model's `images.bin`; and SHALL hold no points (D3, A13, O11).
10. WHEN the Cameras_Step's own code starts, THE Cameras_Step SHALL remove an earlier `out/cameras.json` before any other work, so a run that fails in the step's own code leaves no Viewer_Cameras (D3, A9).
11. THE Cameras_Step SHALL write only inside the Work_Folder and to `out/cameras.json`, where only the Step_Runner writes the Cameras_Report.
12. IF removing the earlier `out/cameras.json`, or writing a file that the Cameras_Step writes itself, like the Calibration_Copy, a Command_Log, a Training_Image, a Training_Mask, a Model_File, the Solve_Record or the Viewer_Cameras, fails because the disk is full or the file can't be written, THEN THE Cameras_Step SHALL fail with an Expected_Failure whose Readable_Error names the file and the reason.
13. WHEN the Cameras_Step runs again on a Job, THE Cameras_Step SHALL rely on the Step_Runner emptying `work/cameras/` first, so every file of an earlier run is gone before the new run writes.

Note (D16): a file that an earlier run uploaded to the Job_Bucket and a rerun removed stays in the bucket, and `cloud/` stays as it is for it. Such a file never feeds a step, because the Step_Runner empties a step's Work_Folder before the step runs (criterion 13) and the Step_Reports decide which steps are done. Deleting it would also need a delete right that our roles don't have.

### Requirement 12: Settings, presets and dependencies

**User Story:** As a team member, I want every COLMAP option and every number of this step in Config, so that we can see everything COLMAP runs with and tune the step without changing code.

**Decided:** D4 (criteria 2, 6 and 9), D5 (criteria 2 and 3), D6 (criteria 2 to 4), D8 (criteria 3 and 5), D9 and D11 (criterion 3), D13 (criteria 3 and 11), D19 (criteria 3 and 10) and D20 (criterion 16). **Assumptions:** A2 (criteria 8 and 9), A4 and A5 (criterion 3). **Open for the design:** O1 to O4 (criterion 3), and O7 and O11 (criteria 2 and 9).

#### Acceptance Criteria

1. THE Config SHALL hold a `[cameras]` Step_Section with the three Preset tables `[cameras.fast]`, `[cameras.default]` and `[cameras.quality]`, each with the same keys, and with a short comment for every key.
2. THE `[cameras]` section SHALL hold one Option_Key for each COLMAP option that Requirement 2, criterion 3 takes from Config, about 180 Option_Keys, each with a comment that names its option and with the value that `references/options.md` gives the option, which is COLMAP 4.2.1's default unless the notes there say otherwise, with two exceptions: `--ImageReader.camera_model`, whose Option_Key is `camera_model` (criterion 3), and `--SequentialMatching.overlap`, which takes `near_overlap` in the Near_Match_Run and `far_overlap` in the Far_Match_Run (D4, D5, D6, O7, O11).
3. THE `[cameras]` section SHALL hold `camera_model = "RADIAL"`, the Option_Key of `--ImageReader.camera_model`, and these Shared_Settings: `fisheye_fov_degrees` (O2), `far_overlap`, at least 9 (O4), `min_placed_frames = 20`, `min_placed_share = 0.5`, `up_agreement = 0.5`, `axis_point_min_eigen_share` (O3), `axis_point_min_front_share = 0.5` (A5), `heading_min_degrees` (O3), one Time_Limit key for each COLMAP command the step runs (A4, O1), `min_vcpus = 16` and `min_memory_gb = 16` (D5, D6, D8, D9, D11, D13).
4. THE Preset tables SHALL each hold one Preset_Setting, `near_overlap`: 15 in `[cameras.fast]`, 15 in `[cameras.default]` and 25 in `[cameras.quality]` (D6).
5. THE real Config SHALL hold `true` in the Option_Keys of `--GlobalMapper.multiple_models` and `--Mapper.multiple_models`, so both mappers run with multiple models on (D8).
6. THE Cameras_Step SHALL take every threshold, count, share, angle and time limit from Config, and hold in code only the Fixed_Constants (D4).
7. IF a setting is missing from `[cameras]` and its Preset table, THEN THE Step_Runner's existing check SHALL fail the Step with the Step_Runner's Readable_Error, which names the section, the table and the key.
8. IF a setting is present but isn't the kind of value the setting must be, or lies outside the setting's range, THEN THE Cameras_Step SHALL fail with an Expected_Failure, before the step reads the Frames_Record or runs any COLMAP_Command, whose Readable_Error names the setting, its table and what the setting must be, where a bool never counts as a number (A2).
9. THE Cameras_Step SHALL check each Option_Key as the kind that its option is in COLMAP 4.2.1, within the range the design gives the Option_Key: a TOML bool for an option that is on or off, a TOML integer for a whole-number option, a TOML integer or decimal for a decimal option, and a TOML string for a choice option, from the names this build can run (D4, A2, O7, O11).
10. THE Cameras_Step SHALL check the other settings: `camera_model` either `"RADIAL"` or `"RADIAL_FISHEYE"`; `fisheye_fov_degrees` a number above 0 and at most 360; `near_overlap`, `far_overlap` and `min_placed_frames` whole numbers above 0; `min_placed_share` and `axis_point_min_front_share` numbers from 0 to 1; `up_agreement` and `axis_point_min_eigen_share` numbers above 0 and at most 1; `heading_min_degrees` a number above 0 and below 90; each Time_Limit a number above 0; and `min_vcpus` and `min_memory_gb` whole numbers above 0.
11. THE Cameras_Step SHALL run the same whatever the machine's CPUs and memory, and leave the Step_Needs for the cloud-jobs spec's Start_Command to read (D13).
12. THE `pyproject.toml` SHALL gain no dependency for the Cameras_Step.
13. THE Cameras_Step SHALL import only the Python standard library, numpy, OpenCV (`cv2`) and `splatpipe`, and import nothing from `cloud/`, boto3, torch, gsplat or pycolmap.
14. THE Cameras_Step SHALL use OpenCV only on images and image files, never to read or write video, as `.kiro/steering/tech.md` requires.
15. THE `tests/conftest.py` test config SHALL gain a `[cameras]` section with the section's three Preset tables, so a test that runs the step through the step's CLI or `run()` without a Config of the test's own gets settings, and a test SHALL be able to change any one of the settings for that test.
16. IF `mapper_ba_refine_extra_params` or `global_mapper_ba_refine_extra_params` is false, THEN THE Cameras_Step SHALL fail with an Expected_Failure, before the step reads the Frames_Record or runs any COLMAP_Command, whose Readable_Error names the key, its table and COLMAP's option, and says that the key must stay true because, with it off, the distortion stays 0, COLMAP's undistorter keeps the camera's own model and copies the Picks unwarped, and the step can't hand the Training_Step a PINHOLE camera (D20).

### Requirement 13: The step report

**User Story:** As a team member, I want the step's report to say how many frames were placed, which mapper placed them and how long each COLMAP command took, so that I can tell a good solve from one that barely held together.

**Assumptions:** A8 (criterion 4) and A12 (criteria 1 to 3).

#### Acceptance Criteria

1. WHEN the Cameras_Step is done, THE Cameras_Report SHALL hold the counts `picks` (the Picks in the Frames_Record), `placed_frames` (the Final_Model's Placed_Frames), `points3d` (the Final_Model's points), `models_found` (the number of Models of the mapper run that made the Final_Model), `fallback_ran` (1 when the Fallback ran, 0 when the Fallback didn't run), `kept_from_global` (1 when the Final_Model came from the Global_Run, 0 when the Final_Model came from the Incremental_Run) and `levelled` (1 when the Levelling moved the scene, 0 when the Levelling kept the COLMAP_Frame) (A12).
2. WHEN the Cameras_Step is done, THE Cameras_Report SHALL hold a timing in seconds for each COLMAP_Command that ran, named after its command, with the two Match_Runs as `sequential_matcher_near` and `sequential_matcher_far`, and a timing in seconds for each stage: `matching` (both Match_Runs), `mapping` (the Calibration, the mappers and choosing the Final_Model), `undistortion` (the Undistortion, the Mask_Warp and the Undistorted_Mask), `levelling`, and `writing` (the Training_Images, the Training_Masks, the Levelled_Model, the Solve_Record and the Viewer_Cameras) (A12).
3. WHEN the Cameras_Step fails, THE Cameras_Report SHALL hold the counts and timings of criteria 1 and 2 that the step had worked out before the step failed, the timing of a COLMAP_Command that failed or ran past its Time_Limit included (A12).
4. THE Cameras_Step SHALL add each count as an int and each timing as a number of seconds, so the Step_Runner's checks never refuse one, and SHALL keep the mean reprojection error, the Level_Move and the reasons in the Solve_Record (A8).
5. WHERE the step warns under Requirement 5, criteria 3 and 5, Requirement 6, criterion 7, or Requirement 9, criteria 3, 7, 10 and 12, THE Cameras_Report SHALL hold that warning.
6. THE Cameras_Report SHALL be the Step_Runner's ordinary Step_Report, written to `out/reports/cameras.json`, with the settings of Requirement 12, which the Step_Runner puts there itself.

### Requirement 14: Running the step and the step list

**User Story:** As a team member, I want the Cameras_Step to run on its own and as the second step of every Job, so that `splatpipe.run` solves the cameras right after picking the frames.

**Decided:** D1 (criteria 3 to 9).

#### Acceptance Criteria

1. THE Cameras_Step SHALL be a package `splatpipe/cameras/` built like the Fake_Step and the Frames_Step: `work(step)`, `run(job_id, preset=None, *, config=None)` and a `__main__.py` whose `main(argv)` calls the Skeleton's `step_main`.
2. THE CLI `uv run python -m splatpipe.cameras --job <id> [--preset P]` SHALL run the step on one Job, print `cameras: done` on standard output and exit with 0 when the step is done, and print the Readable_Error on standard error, with no traceback, and exit with 1 when the step failed.
3. THE Step_List in `splatpipe/run.py` SHALL be `(StepEntry("frames", "splatpipe.frames", True), StepEntry("cameras", "splatpipe.cameras", True), StepEntry("train", "splatpipe.train", True), StepEntry("clean", "splatpipe.clean", False))`, with the Step_List's comment saying that the Frames_Step, the Cameras_Step and the Training_Step are Required_Steps and the Clean_Step is an Optional_Step, in that order, and naming the spec of each entry (D1).
4. WHEN the Job_Runner runs the Step_List on a Job, THE Job_Runner SHALL run the Cameras_Step once the Frames_Step is done and before the Training_Step, and give the Job the Job_Status `done` once the Frames_Step, the Cameras_Step, the Training_Step and the Clean_Step are all done (D1).
5. THE `tests/test_run.py` test that asserts the Step_List SHALL assert the Step_List of criterion 3 (D1).
6. THE `tests/test_frames_step.py` tests that run `run_job` SHALL each run only the Step_List entries the test means to check, or give each step the test runs the stand-in that the step needs, so the real ffmpeg test still checks the Frames_Step on its own and no test's outcome depends on whether COLMAP is installed (D1).
7. THE `tests/test_imports.py` Step_List check SHALL load `[frames]`, `[cameras]`, `[train]` and `[clean]` from the real Config for each of the three Presets, with the check's docstring naming the four entries of criterion 3 in order (D1).
8. THE `tests/test_docs.py` Step_List check SHALL find a `## Step:` section in the Pipeline_Doc for each entry of criterion 3, and the test of the part headed "The step list" SHALL check the words of Requirement 15, criterion 7, and check that the words saying that the camera solve spec or the training spec still has to add its entry are gone (D1).
9. THE frames spec's requirements document SHALL gain a note under Requirement 11, criteria 3 and 4, and under Requirement 12, criterion 5, saying that the cameras spec added the second Step_List entry, and leave the criteria's own words unchanged (D1).
10. THE Cameras_Step SHALL raise `ExpectedFailure` for every failure that Requirements 1, 2, 6, 7, 8, 10, 11, 12 and 17 name, other than the missing setting of Requirement 12, criterion 7, which the Step_Runner's own check refuses, so the Cameras_Report's `error.expected` is `true` and the report's `readable` is the step's own message.

### Requirement 15: Docs and licences

**User Story:** As one of the three people building the steps, I want this step written up next to the others, so that the training, the Viewer and the cloud wrapper fit with it.

**Decided:** D1 (criterion 7), D2 (criterion 3), D3 (criterion 6), D4 (criterion 4), D12 (criterion 8), D13 (criterion 4), D15 (criterion 11), D17 (criterion 10), D18 (criterion 8), D19 (criteria 9 and 12) and D20 (criterion 9). **Assumptions:** A9 (criterion 6) and A15 (criterion 8).

#### Acceptance Criteria

1. THE Pipeline_Doc SHALL gain a `## Step: cameras` section after `## Step: frames`, with the same parts as the Frames_Step's section: a `### What it reads`, a `### What it writes`, a `### Config section`, a `### Step report fields` and a `### Files in out/` part.
2. THE `### What it reads` part SHALL name the Frames_Contract files the step reads, say that the Feature_Extraction reads the Grey_Copies while the mappers and the Undistortion read the Colour_Picks, and say that the Job's machine image needs COLMAP 4.2.1 built as `.kiro/steering/tech.md` says.
3. THE `### What it writes` part SHALL give the Work_Folder's layout and the Levelled_Model as the contract the Training_Step reads: the three Model_Files and their layout, the one PINHOLE camera, the names that match the files in `work/cameras/images/`, the Training_Masks only when `mask_used` is true, the Pose convention, the World_Up (0, −1, 0), the Heading along +Z and the Scene_Centre at the origin, that the Training_Step reads these files with its own reader without COLMAP or pycolmap, what the COLMAP_Folder holds, and the Solve_Record's fields (D2).
4. THE `### Config section` part SHALL name every key in the real `[cameras]` and its Preset tables, say which COLMAP option each Option_Key sets, list the Fixed_Constants, and state the Step_Needs, 16 vCPUs, a soft need, and 16 GB of memory, a hard need, as the Pipeline_Doc's "What each step asks the machine for" table plans them, and that the step itself never checks the machine (D4, D13).
5. THE `### Step report fields` part SHALL name every count and timing of Requirement 13 and say when the step adds each warning.
6. THE `### Files in out/` part SHALL describe the Viewer_Cameras and their fields, say that the step removes an earlier `out/cameras.json` when the step's own code starts and writes a new one last of all, so a run that fails in the step's own code leaves none, and say that a run the Step_Runner stops before the step's code starts leaves the earlier `out/cameras.json` next to a failed Cameras_Report: with the earlier `work/cameras/` too when the Step_Runner stops on the step's Config section, like a `[cameras]` section with a missing Preset table, and with `work/cameras/` already emptied when it stops later, like on a step module that fails to import, because `splatpipe/step.py` empties the Work_Folder before it imports the step (D3, A9).
7. THE Pipeline_Doc's "The step list" section SHALL say that the Step_List holds the Frames_Step, the Cameras_Step and the Training_Step, all three required, and then the Clean_Step, optional, with their four entries, and that a Job is done once all four steps are done and done with warnings when the three Required_Steps are done and the Clean_Step fails, in place of both sides' sentences on the list's entries: `main`'s, which say the camera solve spec still has to add its entry, and this branch's, which say the training spec still has to add its entry (D1).
8. THE Licence_List's COLMAP row SHALL be one row that keeps `main`'s facts, which are the licence with PoseLib and faiss, the eight build flags and what each one does, the note on the image built on 2026-11-22, and the libraries in the apt archive row that COLMAP links, and that says how the step uses COLMAP: the commands the Cameras_Step runs, as separate command line processes on our private machine image, never vendored or linked into our code, and with no pycolmap; that on laptops Homebrew's COLMAP 4.2.1, which holds SiftGPU, CGAL and LSD, runs `-h` and the laptop tests with `use_gpu 0`, the integration test's whole solve included, as Samarth OKed, with his reasons from D18; and, in the Installed column, that COLMAP is on the laptop through Homebrew and on the machine image `ami-0fe14782b05fe63b2` (D12, D18, A15).
9. THE Pipeline_Doc's `## Step: cameras` section SHALL name the Camera_Models `"RADIAL"` and `"RADIAL_FISHEYE"` and the Fisheye_Start's five numbers, give the Model_Files' model ids and parameters for PINHOLE, RADIAL and RADIAL_FISHEYE, say that the Undistorted_Camera has fx = fy, and say that the settings check refuses a false `mapper_ba_refine_extra_params` or `global_mapper_ba_refine_extra_params`, with no word of OPENCV or OPENCV_FISHEYE left in the section (D19, D20).
10. THE Pipeline_Doc's `## Step: cameras` section SHALL say that the step caps each COLMAP command's Time_Limit at the time left before `SPLATPIPE_CAP_DEADLINE`, in place of the words that the step doesn't read it, and the part headed "The cap deadline" SHALL say that the cameras step reads it too (D17).
11. THE Pipeline_Doc's part headed "The smoke test" SHALL list the Cameras_Smoke_Check after the COLMAP check and say that its numbers, like `cameras_frames`, are in `[cloud.smoke]` (D15).
12. THE colmap-cli skill, `.kiro/skills/colmap-cli/SKILL.md` and `references/options.md`, SHALL say that the step passes `--ImageReader.camera_model` from `camera_model`, RADIAL or RADIAL_FISHEYE, and passes `--ImageReader.camera_params` with the Fisheye_Start only for RADIAL_FISHEYE, in place of the note that the step never passes `--ImageReader.camera_params`, and SHALL leave each code block of `references/options.md` as it is (D19).
13. THE frames spec's `requirements.md`, `design.md` and `tasks.md`, and the comment in `splatpipe/frames/settings.py`, SHALL say Step_Needs where they say Machine_Needs, with no task or sub-task title line changed, so every step's spec uses the cloud-jobs spec's term.
14. THE devlog, `docs/devlog.md`, SHALL gain one line for each of the cameras sub-tasks 1.1 to 4.9, 5.1 and 5.3, in the hook's line format and in time order among the other lines, with the time and files the design gives each line from the git history.

Note on criterion 12: the skill names neither OPENCV nor OPENCV_FISHEYE today. Its `--ImageReader.camera_model` row gives COLMAP's default, `SIMPLE_RADIAL`, which the code block of `references/options.md` keeps, so criterion 12 adds the step's two Camera_Models next to it.

Note on criterion 13: the Pipeline_Doc holds no "Machine_Needs". Its frames section says "the step's machine needs" in plain words, as its cameras and train sections do, so the Pipeline_Doc stays as it is for criterion 13.

Note: on `main`, the COLMAP row holds what cloud-jobs Requirement 14.5 asks for, the build flags that keep SiftGPU, CGAL and LSD out, and three more flags. `tests/test_cloud_image.py` checks the five-flag string `-DCUDA_ENABLED=OFF -DOPENGL_ENABLED=OFF -DGUI_ENABLED=OFF -DCGAL_ENABLED=OFF -DLSD_ENABLED=OFF` word for word in that row, checks that the row holds `-DONNX_ENABLED=OFF`, `-DDOWNLOAD_ENABLED=OFF` and `-DBLA_VENDOR=OpenBLAS`, and checks that exactly one Licence_List row has COLMAP in its name. So criterion 8's words join the row's text, which stays as `main` has it up to "It links the libraries in the apt archive row.", and the row stays the only one with COLMAP in its name.

### Requirement 16: Tests

**User Story:** As a team member, I want the step's maths tested on a laptop CPU with made-up cameras, and the real COLMAP run tested when COLMAP is there, so that the tests are quick, free and safe to run anywhere.

**Decided:** D14 (criteria 1, 3, 4, 6 and 8), D15 (criteria 6, 7 and 14), D17 (criterion 5), D18 (criterion 6) and D19 (criterion 6). **Assumptions:** A16 (criterion 7).

#### Acceptance Criteria

1. THE Test_Suite SHALL test the Levelling and the Placed_Check with unit tests on synthetic cameras and points that the tests build themselves, with no COLMAP: for the Levelling, cameras that agree and cameras that don't, a first Placed_Frame that looks straight down, no Placed_Frame with a Heading, an Axis_Point that is well defined and one that isn't, no points, a Mean_Up that points along (0, 1, 0), and an error while the Level_Move is worked out; and for the Placed_Check, exactly `min_placed_frames` Placed_Frames and one fewer, and exactly `min_placed_share` of the Picks and fewer (D14).
2. THE Test_Suite SHALL test the Model_Reader and the Model_Writer, every setting check, the Frames_Contract checks, the Fisheye_Start, the choice of the Kept_Model, the Undistorted_Mask's threshold, the command lines built from Config, the Solve_Record, the Viewer_Cameras and the report fields with unit tests.
3. THE Test_Suite SHALL hold the property tests of Requirement 2, criterion 13, Requirement 6, criterion 12, Requirement 8, criterion 7, Requirement 9, criteria 15 to 18, Requirement 10, criterion 9, and Requirement 17, criterion 8, written with Hypothesis (D14).
4. THE Test_Suite's unit tests and property tests SHALL both be required, never optional (D14).
5. THE Test_Suite SHALL test, with a stand-in for COLMAP and a stand-in clock, the order of the COLMAP_Commands, the Fallback after each way the Global_Run gives up and after each kind of Failed_Calibration, the Timeouts, the Time_Limits that the Cap_Deadline sets, a Cap_Deadline that leaves no time, a value of `SPLATPIPE_CAP_DEADLINE` that `read_cap_deadline` refuses, the non-zero exits, the missing `colmap` program and the Frames_Contract failures, so the whole suite passes on a machine with no COLMAP (D17).
6. THE Test_Suite SHALL hold an integration test that runs the Cameras_Smoke_Check's own code with the real COLMAP, with Homebrew's COLMAP on a laptop, on the Smoke_Picks in a temporary folder, keeps the Smoke_Picks' level path, and checks that the Final_Model's f and the Undistorted_Camera's fx and fy each come out within 2 % of the true 260 px, that `work/cameras/sparse/` reads back through `colmap model_analyzer` and through the Model_Reader and is what `colmap model_converter` writes byte for byte, that the Training_Masks, the Levelling and the Viewer_Cameras are right, and that the Work_Folder's layout matches Requirement 11 (D14, D15, D18, D19).
7. THE Test_Suite SHALL hold a test that runs the Cameras_Smoke_Check's Option_Check, `colmap <command> -h` for each COLMAP command the step runs, and checks that the command lists every option the step passes that command (A16, D15).
8. IF the `colmap` program isn't on the PATH, THEN THE Test_Suite SHALL skip the tests that need COLMAP and say why (D14).
9. THE Test_Suite SHALL check that each Readable_Error of the step's Expected_Failures names what its requirement says the Readable_Error names, and holds no traceback.
10. THE Test_Suite SHALL check that every file in `splatpipe/cameras/` imports only the standard library, numpy, `cv2` and `splatpipe`, so torch, gsplat, boto3, pycolmap and `cloud/` stay out (Requirement 12, criterion 13).
11. THE Test_Suite SHALL check that a run that fails in the Placed_Check, in the Undistortion or while writing a Training_Image leaves no Solve_Record and no `out/cameras.json` (Requirement 11, criteria 6 and 10).
12. THE Test_Suite SHALL check that the Pipeline_Doc's `## Step: cameras` section has its five parts and names every key of the real `[cameras]`, every count and timing of Requirement 13 and every part of the Work_Folder, that the section and the Pipeline_Doc's cloud parts say what Requirement 15, criteria 6 and 9 to 11 say, that the colmap-cli skill says what Requirement 15, criterion 12 says, and that the Licence_List's COLMAP row says what Requirement 15, criterion 8 says.
13. THE Test_Suite SHALL run on the CPU with no network, make every Job in a temporary Jobs_Folder, or the Smoke_Solve's Job_Folder in a temporary folder, and never in the real `data/` folder, commit no image, database or model to git, and give every subprocess a timeout.
14. THE Test_Suite SHALL test the Cameras_Smoke_Check's parts that need no COLMAP with unit tests, the Smoke_Picks, the options the Option_Check looks for, the Option_Check's refusals of Requirement 18, criterion 6, with a stand-in for `colmap <command> -h`, the Smoke_Solve with the stand-in for COLMAP, and the pass and fail limits of Requirement 18, criterion 5, and SHALL test the `cameras` entry of `SMOKE_CHECKS`, its place in the list and the numbers it passes from `[cloud.smoke]`, with a stand-in for the Cameras_Smoke_Check's code (D15).
15. THE Test_Suite SHALL test each refusal of Requirement 8, criterion 2 with a unit test, the refusal of an image that uses another camera than the Undistorted_Model's one camera included.

### Requirement 17: The cap deadline

**User Story:** As a team member running Jobs in the cloud, I want every COLMAP command stopped by the Job's Cap_Deadline at the latest, so that a slow or hung command fails the step with a clear report before the machine's Time_Cap shuts the machine down.

**Decided:** D17 (criteria 1 to 8).

#### Acceptance Criteria

1. WHEN the settings checks of Requirement 12 pass, THE Cameras_Step SHALL read the Cap_Deadline once, from the environment variable `SPLATPIPE_CAP_DEADLINE`, with `read_cap_deadline` from `splatpipe/train/deadline.py`, before the step reads the Frames_Record (D17).
2. IF `read_cap_deadline` refuses the value of `SPLATPIPE_CAP_DEADLINE`, THEN THE Cameras_Step SHALL fail with `read_cap_deadline`'s own Expected_Failure, before the step reads the Frames_Record or runs any COLMAP_Command (D17).
3. WHERE `SPLATPIPE_CAP_DEADLINE` is set, THE Cameras_Step SHALL give each COLMAP_Command, as the command starts, the smaller of the value of its Time_Limit key and the seconds left before the Cap_Deadline as its Time_Limit (D17).
4. WHERE `SPLATPIPE_CAP_DEADLINE` isn't set, THE Cameras_Step SHALL give each COLMAP_Command the value of its Time_Limit key as its Time_Limit, with no cap (D17).
5. IF no time is left before the Cap_Deadline when a COLMAP_Command is about to start, THEN THE Cameras_Step SHALL start no COLMAP_Command and fail with an Expected_Failure, with no Fallback and with no timing for the command, whose Readable_Error names the command, gives the Cap_Deadline, and says to give the Job a longer Time_Cap, with a larger `cap_minutes` for the Job's Preset in `[cloud.presets]`, and to run the Job again.
6. IF a COLMAP_Command runs past a Time_Limit that the Cap_Deadline set, THEN THE Cameras_Step SHALL stop the command and fail with an Expected_Failure, with no Fallback, whose Readable_Error names the command, gives the Time_Limit in seconds and the Cap_Deadline, and says to give the Job a longer Time_Cap, with a larger `cap_minutes` for the Job's Preset in `[cloud.presets]`, and to run the Job again.
7. THE real Config SHALL keep the seven Time_Limit keys at the values the design's O1 gives them (D17).
8. FOR ALL values of a Time_Limit key above 0, Cap_Deadlines or none, and times now, THE Cameras_Step SHALL give a COLMAP_Command the key's value as its Time_Limit when there's no Cap_Deadline or at least that many seconds are left, the seconds left, marked as set by the Cap_Deadline, when fewer are left, and no start when no time is left (property).

Note: the Training_Step reads the same variable with the same function, so a value that one step refuses, the other refuses with the same Readable_Error.

### Requirement 18: The cameras smoke check

**User Story:** As a team member building a machine image, I want the Smoke_Test to check the image's COLMAP the way the Cameras_Step runs it, so that a COLMAP that refuses one of the step's options, or solves cameras badly, fails the Smoke_Test and not the first Job.

**Decided:** D15 (criteria 1 to 9) and D19 (criterion 5).

#### Acceptance Criteria

1. THE Cameras_Smoke_Check SHALL be code in `splatpipe/cameras/smoke.py` that runs on the machine image without pytest, and SHALL import only the standard library, numpy, cv2 and `splatpipe` (D15).
2. THE Smoke_Test's `SMOKE_CHECKS` in `cloud/smoke_runner.py` SHALL hold `SmokeCheck("cameras", _check_cameras)` right after `SmokeCheck("colmap", _check_colmap)`, so the checks run as `torch-gpu`, `gsplat`, `fused-ssim`, `ffmpeg-zscale`, `colmap`, `cameras`, `node`, `splat-transform`, `moge-weights`, `training` and `cloud-runner` (D15).
3. WHEN the Cameras_Smoke_Check runs, THE Cameras_Smoke_Check SHALL run the Option_Check first: `colmap <command> -h` for each of the seven COLMAP commands that the Cameras_Step runs, each within the Smoke_Test's `check_timeout_seconds`, checking that each command lists every option that the step's command lines pass that command for the Config the check is given, for each Preset, for both Camera_Models, and with and without the Mask (D15).
4. WHEN the Option_Check passes, THE Cameras_Smoke_Check SHALL write the Smoke_Picks in the Frames_Contract's layout and run the Smoke_Solve with the real COLMAP, with the `[cameras]` settings of the Preset that `[jobs].default_preset` names and every Time_Limit key at `cameras_timeout_seconds` (D15).
5. THE Cameras_Smoke_Check SHALL pass only when the Smoke_Solve finishes with no failure, places at least `cameras_placed_share` of the Smoke_Picks, and gives a Final_Model whose f, and an Undistorted_Camera whose fx and fy, each lie within `cameras_focal_tolerance` times `cameras_focal_pixels` of `cameras_focal_pixels` (D15, D19).
6. IF a `colmap <command> -h` run fails, or a command doesn't list an option that the step passes it, THEN THE Cameras_Smoke_Check SHALL fail with a Readable_Error that names the command and either the run's exit and Last_Error_Line or each missing option.
7. IF the Smoke_Solve fails, places too few Smoke_Picks or gives a focal length outside the tolerance, THEN THE Cameras_Smoke_Check SHALL fail with a Readable_Error that gives the Cameras_Step's own Readable_Error, or the numbers the check found next to the numbers it needs.
8. THE Cameras_Smoke_Check SHALL write only inside the scratch folder the Smoke_Test gives it, and SHALL give back one line for the Smoke_Test's log, which says that the Option_Check passed and gives the Smoke_Solve's placed Picks, its mapper and its f (D15).
9. THE `[cloud.smoke]` table of the real Config SHALL hold the Cameras_Smoke_Check's numbers, each with a comment: `cameras_frames = 30`, `cameras_width = 320`, `cameras_height = 240`, `cameras_focal_pixels = 260.0`, `cameras_placed_share = 0.8`, `cameras_focal_tolerance = 0.02` and `cameras_timeout_seconds = 600`, which `cloud/config.py` SHALL read into its `SmokeChecks` and check like the training check's numbers, and which `cloud_config()` in `tests/fake_aws.py` SHALL hold too (D15).
