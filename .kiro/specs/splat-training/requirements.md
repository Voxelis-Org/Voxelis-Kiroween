# Requirements Document

## Introduction

This spec builds the Train_Step, which trains the 3D Gaussian splat on the Job's GPU machine: a 24 GB GPU (an NVIDIA A10G or L4) for the `fast` and `default` Presets, a 48 GB L40S for `quality`, and at least 32 GB of memory. The step reads the levelled cameras, the sparse points and the undistorted frames that the Camera_Solve_Step writes to `work/cameras/`. It trains with our own small trainer on gsplat 1.5.3's `rasterization()` and its MCMC strategy, and writes the Trained_Splat to `out/train.ply` in the standard 3DGS PLY layout. The Clean_Step reads the splat from there, and when that step fails or doesn't run, `out/train.ply` is the Job's result.

The Train_Step is a Required_Step and the second entry in the Step_List, after the Frames_Step. It uses the Job_Folder, the Step_Runner, the Step_Report and the Presets of the pipeline-core spec, and it's built like the Fake_Step and the Frames_Step. On the cloud side it follows the cloud-jobs spec: its Step_Needs go in `[train]` for the Start_Command, it reads the Cap_Deadline that the Cloud_Runner sets, and the Smoke_Test gains a check for it.

torch, gsplat and fused-ssim exist only on the Machine_Image, so the training runs only there, in one module, the GPU_Module. Everything around it is CPU code that imports no GPU package: reading the camera solve's files, the scene's numbers, the schedules, the exposure maths, the PLY writer and checker, the settings and the deadline. The Test_Suite tests all of that on a laptop CPU with tiny made-up data. The Smoke_Test's new check runs the real trainer for a few Iterations on the GPU before any Job does.

The camera solve spec isn't written yet, so this spec takes the hand-off below as the contract, and the camera solve spec will write exactly this:

- `work/cameras/images/<name>.jpg`: the undistorted colour frames.
- `work/cameras/masks/<name>.png`: an image's mask, when it has one. Black means ignore.
- `work/cameras/sparse/`: COLMAP's binary model, with `cameras.bin` (one shared PINHOLE camera), `images.bin` and `points3D.bin`, in the levelled world frame.

The Train_Step reads these files with our own small reader, with no COLMAP and no pycolmap.

This spec trains without depth priors. A later spec adds them, and Requirement 5 keeps the loss and the per-image data open to a new term.

Out of scope:

- The Camera_Solve_Step. This spec only reads the files it will write, and states that contract in the Pipeline_Doc.
- Depth priors, and any code for them.
- The clean spec's files, and the Clean_Step's Step_List entry. The Clean_Step stays out of `STEP_LIST` for now, and a later spec adds its entry after the Train_Step's (D5).
- Any change to the Project_File, `uv.lock` or the Machine_Image (A1).
- Running the Smoke_Test or anything else that costs money. No task of this spec is a paid run (A12).
- Evaluation images and scores (PSNR, SSIM, LPIPS), pose optimisation, per-image appearance beyond the Exposure_Fix, bilateral grids, the antialiased and 3DGUT modes, checkpoints, resuming a stopped training run, TensorBoard, the viewer and trajectory videos.
- Changing the Cloud_Runner, its Uploader, the Start_Command or the Fallback_Lists (O1).

### Task budget

The whole spec stays within 10 tasks, which the design and tasks phases plan. Each task writes its own unit tests and property tests, both required, never optional.

### Decisions made before this draft

The user chose these when asked, before the requirements were written.

- **D1 (Requirement 2): The GPU memory check.** Before the step loads anything, it reads the GPU's total memory from torch and refuses with an Expected_Failure when that memory is below the Preset's `min_gpu_memory_gb` times a slack setting in `[train]`, `gpu_memory_slack = 0.9`. GB means 10⁹ bytes, as in `gpu_memory_gb` in `[cloud.types]`. The slack is there because a "24 GB" A10G or L4 reports a little under 24 GB: nvidia-smi shows about 23,028 MiB, and about 46,068 MiB for an L40S. So `quality`, which needs 48 GB, refuses every 24 GB card. The comparison is a pure function of a byte count, so it's tested without a GPU. There is no trial Iteration. `min_vcpus` and `min_memory_gb` stay cloud-only Step_Needs that the step never checks, like the Frames_Step's.
- **D2 (Requirements 3 and 5): The loss and the start scale.** The photometric loss is 0.8 × L1 + 0.2 × (1 − SSIM), as gsplat's `ssim_lambda` of 0.2 makes it. Each Start_Splat's scale is 0.1 times the root mean square of the point's distances to its 3 nearest other points, the same on all three axes, stored as a log.
- **D3 (Requirement 5): The Effective_Rank_Term.** The term is 0.01 times the mean over splats of max(−log(erank − 1 + 10⁻⁵), 0) plus the splat's smallest scale in linear units divided by the Scene_Extent (D7), where erank = exp(−Σᵢ qᵢ log qᵢ) and qᵢ = sᵢ² / Σⱼ sⱼ² over the splat's three scales sᵢ. The term is on from 7/30 of the Planned_Iterations.
- **D4 (Requirement 13): The contract the Clean_Step reads.** This spec records the contract the Clean_Step reads in the Pipeline_Doc's new `## Step: train` section: `out/train.ply`, its fields, and the largest Splat_Budget, 3,000,000 splats, which is within the 8,000,000 splats the Clean_Step takes (its `max_source_splats`).
- **D5 (Requirements 1, 11 and 13): The Step_List until the camera solve lands.** `STEP_LIST` gains `StepEntry("train", "splatpipe.train", True)` after the Frames_Step. The camera solve spec puts its entry between the two when it lands, and a later spec adds the Clean_Step's entry after the Train_Step's, so the order becomes frames, cameras, train, clean. Until the camera solve lands, a real Job runs the Frames_Step and then fails at the Train_Step, with a Readable_Error saying that the camera solve output is missing.

### Decisions made at review

The user reviewed the draft and chose these.

- **D6 (Requirements 8, 10, 12 and 14): Splats with a NaN or an infinity.** After the Mean_Exposure is baked in, which can overflow a value to an infinity once it's rounded to float32, the Train_Step drops every splat that has a NaN or an infinity in any field, checked as float32, before it writes `out/train.ply`, and adds a warning with the count to the Train_Report, rather than failing the step. The PLY_Check still refuses any file with a NaN or an infinity. A run with no splat left still fails, at the PLY_Check's existing "no splats" refusal, and the failed report holds the warning with the count. The Train_Smoke_Check drops nothing, so it still fails on any splat with a NaN or an infinity, because its tiny scene should never make one. gsplat's `export_splats` also drops such splats, but with no warning, and in 1.5.3 one non-finite opacity makes it drop every splat.
- **D7 (Requirement 5): The scale terms in units of the Scene_Extent.** The user chose this after the spec was built. gsplat's example rescales every scene to about 1 unit across before it trains, and the 0.01 weights of `scale_reg` and of the Effective_Rank_Term's smallest-scale part were tuned at that size. The Train_Step trains in the camera solve's units instead (Requirement 1, criterion 20), and COLMAP's mapper rescales its models to about 10 units across, so in those units the two terms would pull about 10 times harder and shrink the splats. So in `scale_reg` and in the Effective_Rank_Term's smallest-scale part, each scale is divided by the Scene_Extent before the mean is taken, and neither term depends on the scene's units. The Effective_Rank_Term's log part stays as it is, because erank depends only on the ratios of the scales. No test is added: a laptop can't import the GPU_Module, so the Smoke_Test runs the change on the GPU.

### Assumptions to confirm at review

The user didn't decide these. Each is the recommended answer, marked in the requirement it belongs to, and each is a small change if the review decides otherwise.

- **A1 (Requirements 3, 9 and 14): No new Python package.** This spec adds no package to the Project_File or `uv.lock`, because changing `uv.lock` makes the Machine_Image stale, which costs an Image_Build of about 75 minutes and about $2, then a Smoke_Test. So the search for each Start_Splat's 3 nearest neighbours runs with torch on the GPU, and only the formula on top of the distances is a numpy function. torch, gsplat and fused-ssim are imported only by the GPU_Module, and the Train_Step imports the GPU_Module only inside its own code, when it runs, so every other module of the step imports on a laptop. Nothing new is installed, so the Licence_List gains no line.
- **A3 (Requirement 10): The Train_Record.** The numbers that aren't whole, and so can't be Step_Report counts, go in the Train_Record, `work/train/train.json`: the Near_Plane, the Median_Scene_Depth, the Scene_Extent, the starting Means_LR, the ratios of the Training_Size's width and height to the Shared_Camera's, the Mean_Exposure, the Cap_Deadline and the Stop_Time.
- **A4 (Requirements 7 and 9): The save margin and an empty deadline.** `save_margin_minutes = 5`: the Trainer stops 5 minutes before the Train_Deadline, which leaves time to copy the splats off the GPU and to write and check a Trained_Splat of up to about 708 MB (3,000,000 splats of 236 bytes). An empty `SPLATPIPE_CAP_DEADLINE` counts as malformed rather than unset, because the User_Data leaves the variable empty only when working out the deadline failed.
- **A5 (Requirements 1 and 13): Masks, one image at a time.** Each Mask_File is optional on its own: an image with a Mask_File leaves out its Masked_Pixels, and an image without one trains on every pixel. Files in `images/` and `masks/` that `images.bin` doesn't name are ignored. A Mask_File that masks every pixel of its image is refused, because the image would have no pixel left for the Loss_Terms.
- **A6 (Requirement 3): Start rotations and points on top of each other.** Every Start_Splat starts with the identity rotation. gsplat's example draws random ones, but a splat whose three scales are equal looks the same at any rotation. A Start_Splat whose nearest other points sit exactly on it has a Start_Distance of 0, whose log isn't finite, so it takes the smallest Start_Distance above 0 instead, and the step fails when every Start_Distance is 0.
- **A7 (Requirement 4): A loss that stops being finite.** The Trainer checks the loss after every Iteration and stops at once when the loss isn't finite, rather than training on for hours to a splat that the PLY_Check refuses. The cause is a fault in the training rather than in the person's video, so the step fails as an Unexpected_Error, with the Iteration and the Loss_Terms' values in the report's Technical_Detail. Reading the loss makes the CPU wait for the GPU once per Iteration, as gsplat's example already does for its progress bar.
- **A8 (Requirements 4, 5 and 9): Whole Iterations from fractions, and the image order.** An Iteration worked out from a fraction of the Planned_Iterations is rounded to the nearest whole Iteration. So the Effective_Rank_Term starts at Iteration 1,633 for `fast`, 3,500 for `default` and 7,000 for `quality`, and the MCMC strategy stops refining at Iteration 6,300, 13,500 and 25,000. `erank_start_fraction` is written as 0.23333333333333334, the float nearest 7/30. The Trainer trains on one Training_Image per Iteration and goes through every image once per pass, in an order drawn afresh for each pass from a generator seeded with `seed`, as gsplat's example does with its shuffled loader.
- **A9 (Requirements 1, 2, 7 and 11): The order of the step's work.** The step removes an earlier `out/train.ply`, checks its settings, reads the Cap_Deadline, checks that the Camera_Solve_Output's files are there, imports the GPU_Module and checks the GPU, and only then reads the Sparse_Model, the Training_Images and the Mask_Files. So the GPU check comes before anything is loaded (D1). Checking that the files are there doesn't load them, and checking it first means a Job without the camera solve's files fails with the camera solve's message everywhere, a laptop included (D5). The review can move that check after the GPU check.
- **A10 (Requirement 2): The gsplat version.** The step refuses a gsplat whose version isn't 1.5.3, because the Trainer is written against 1.5.3's API.
- **A11 (Requirement 12): The training check's numbers.** The Train_Smoke_Check uses the `[train]` settings of the Preset that `[jobs].default_preset` names (`fast`), except for seven new `[cloud.smoke]` keys: `train_images = 4`, `train_points = 200`, `train_splats = 400`, `train_steps = 30`, `train_refine_start_step = 10`, `train_refine_every = 10` and `train_sh_degree_interval = 5`. Its images are `image_pixels` (64) pixels square. With these numbers the 30 Iterations turn the Effective_Rank_Term on at Iteration 7, reach SH_Degree 3 at Iteration 15, and include one MCMC refine, at Iteration 20, which also adds splats.
- **A12 (Requirement 12): The Smoke_Marker gate.** The Smoke_Command uploads the Code_Archive, so the Train_Smoke_Check runs on the current Machine_Image with no rebuild. But the Start_Command's gate looks only for a Smoke_Marker for the image's id, and a marker written before this spec still passes it. So the team runs the Smoke_Test again, a paid run that needs the person's yes, before the first training Job. Making the marker record which checks ran is left out.
- **A13 (Requirement 13): The frames section's mask sentence.** The Pipeline_Doc's frames section says of `work/frames/mask.png` that "the training loss reads the same file", and a test in `tests/test_docs.py` looks for that phrase. The Train_Step reads only the Camera_Solve_Output, so the sentence changes to say that the camera solve step passes the Mask on to training as one mask per image in `work/cameras/masks/`, and the test's phrase changes with it.
- **A14 (Requirement 14): Checks from other built specs.** Two checks from other built specs change, and those specs' requirements files stay as they are. The cloud-jobs spec's Requirement 16, criterion 7 test in `tests/test_cloud_imports.py` refuses torch, gsplat and fused-ssim in every file under `splatpipe/`, so the GPU_Module would fail it. The frames spec's Requirement 12, criterion 5 sentence in the Pipeline_Doc says that a Job is done after the Frames_Step alone, and `tests/test_docs.py` looks for that sentence.

### Open question (resolved)

- **O1 (no requirement): The Uploader sent every file on every pass.** Resolved by f67ce0c, "Uploader: skip files unchanged since they last went up". The Uploader now sends a file again only when its size or modified time changed, so `out/train.ply` goes up once after the move into `out/`, and again only if it changes. A pass can still catch `work/train/train.ply` while it's being written. That partial copy goes up again on the next pass once the file changes, and nothing reads `work/<Job_Id>/` in the bucket, so it does no harm. This spec still doesn't change the Uploader.

## Glossary

- **Train_Step**: the Step this spec builds, in `splatpipe/train/`. Its Step_Name is `train`, so its Config section is `[train]`, its Work_Folder is `work/train/` and its Step_Report is `out/reports/train.json`.
- **Camera_Solve_Step**: the Step that goes between the Frames_Step and the Train_Step, built in its own spec, whose Step_Name is `cameras`. It writes the Camera_Solve_Output.
- **Clean_Step**: the clean spec's Step, whose Step_Name is `clean`. It reads the Trained_Splat.
- **Camera_Solve_Output**: the files the Train_Step reads from `work/cameras/`: the Training_Images, the Mask_Files and the Sparse_Model.
- **Training_Image**: one undistorted colour frame, `work/cameras/images/<name>.jpg`, where `<name>.jpg` is the image's name in `images.bin`.
- **Mask_File**: an image's optional mask, `work/cameras/masks/<stem>.png`, where `<stem>` is the image's name without `.jpg`. Black means ignore.
- **Masked_Pixel**: a training pixel whose Mask_File value, read as grey from 0 to 255 and scaled to the Training_Size, is below `mask_threshold` (128). The Loss_Terms that compare pixels leave it out.
- **Sparse_Model**: COLMAP's binary model in `work/cameras/sparse/`: `cameras.bin`, `images.bin` and `points3D.bin`.
- **Model_Reader**: our own reader for the Sparse_Model, written with numpy and the standard library's `struct` module.
- **Shared_Camera**: the one camera in `cameras.bin`, of COLMAP's PINHOLE model, with a width and a height in pixels and the parameters fx, fy, cx and cy.
- **Image_Pose**: an image's rotation quaternion (w, x, y, z) and translation t from `images.bin`. Together they take a world point into the camera's frame, whose axes are x right, y down and z forward, as COLMAP and gsplat both use.
- **Camera_Centre**: an image's position in the world, −Rᵀt, where R is the rotation matrix of the image's quaternion.
- **Sparse_Points**: the points in `points3D.bin`, each with a position and an 8-bit RGB colour.
- **Levelled_Frame**: the world frame of the Sparse_Model, as the Camera_Solve_Step leaves it.
- **Training_Size**: the width and height in pixels that the Train_Step trains at (Requirement 3, criterion 1).
- **Scene_Extent**: `scene_extent_factor` (1.1) times the largest distance from a Camera_Centre to the mean of the Camera_Centres, as gsplat's example works it out.
- **Median_Scene_Depth**: the median, over the cameras with at least one Sparse_Point in front of them, of each such camera's median camera-space z of the Sparse_Points in front of it, z above 0. A median here is numpy's: the middle value, or the mean of the two middle values.
- **Near_Plane**: `near_plane_factor` (0.03) times the Median_Scene_Depth. The renderer leaves out every splat nearer a camera than the Near_Plane.
- **Splat_Budget**: the most splats a Preset trains, `max_splats`, which is the MCMC strategy's `cap_max`.
- **Start_Splats**: the splats training starts from: one for each Sparse_Point, or for each point of the subset when the points outnumber the Splat_Budget.
- **Start_Distance**: for one Start_Splat, the root mean square of its distances to its `start_scale_neighbours` (3) nearest other Start_Splats.
- **Iteration**: one round of the Trainer, which renders one Training_Image, works out the loss, runs the backward pass and updates the splats. Iterations count from 0. The `steps` in the Preset tables, and `steps_planned` and `steps_run` in the Train_Report, count Iterations.
- **Planned_Iterations**: the Preset's `steps`: 7,000 for `fast`, 15,000 for `default` and 30,000 for `quality`.
- **Schedules**: the pure functions, importing no torch, that give an Iteration's SH_Degree and Means_LR, and the Refine_Stop and the Erank_Start.
- **SH_Degree**: the degree of spherical harmonics an Iteration renders with (Requirement 4, criterion 6).
- **Means_LR**: the learning rate of the splats' positions at an Iteration (Requirement 4, criterion 4).
- **Refine_Stop**: the Iteration at which the MCMC strategy stops relocating and adding splats (Requirement 4, criterion 8).
- **Erank_Start**: the first Iteration whose loss holds the `erank` Loss_Term (Requirement 5, criterion 7).
- **Loss_Term**: one named part of the loss, with a weight from Config: `l1`, `ssim`, `opacity_reg`, `scale_reg` and `erank` (Requirement 5).
- **Effective_Rank_Term**: the `erank` Loss_Term, the regulariser against needle-shaped splats from Hyung et al., "Effective Rank Analysis and Regularization for Enhanced 3D Gaussian Splatting" (NeurIPS 2024), as D3 gives it.
- **Exposure_Fix**: one 3 × 4 colour transform [A | b] per Training_Image, which turns each rendered colour c into A·c + b.
- **Corrected_Render**: a render after its Training_Image's Exposure_Fix. The Loss_Terms compare it with the Training_Image.
- **Mean_Exposure**: [Ā | b̄], the element-wise mean of every Training_Image's Exposure_Fix.
- **Exposure_Bake**: the numpy function that writes the Mean_Exposure into the splats' colours (Requirement 6, criterion 5).
- **SH_C0**: 0.28209479177387814, the band-0 spherical harmonic constant, 1 / (2√π).
- **Train_Deadline**: the Cap_Deadline minus `clean_reserve_minutes`: the time by which the Trained_Splat is written and checked, leaving that many minutes for the Clean_Step.
- **Stop_Time**: the Train_Deadline minus `save_margin_minutes`. The Trainer starts no Iteration at or after it.
- **Trained_Splat**: `out/train.ply`, the splat the Train_Step writes.
- **Splat_Layout**: the Trained_Splat's file layout (Requirement 8, criterion 3).
- **PLY_Writer**: our numpy function that writes splats in the Splat_Layout.
- **PLY_Reader**: our numpy function that reads a file in the Splat_Layout back into arrays.
- **PLY_Check**: the check the Train_Step runs on the written file before moving it into `out/` (Requirement 8, criterion 6).
- **GPU_Module**: `splatpipe/train/gpu.py`, the one module of the Train_Step that imports torch, gsplat and fused-ssim. The Trainer lives there.
- **Trainer**: the function in the GPU_Module that trains the splat, which the Train_Step and the Train_Smoke_Check both call.
- **GPU_Need**: `gpu_memory_slack` × `min_gpu_memory_gb` × 10⁹ bytes, for the Job's Preset.
- **GPU_Fit_Check**: the pure function that compares a GPU's total memory in bytes with the GPU_Need.
- **GPU_Stand_In**: what the Test_Suite puts in place of the GPU_Module, so the Train_Step runs on a CPU in the tests.
- **Train_Record**: `work/train/train.json`, which holds the numbers that can't be Step_Report counts (A3).
- **Train_Report**: the Train_Step's Step_Report, with the fields Requirement 10 lists.
- **SMOKE_CHECKS**: the ordered list of the Smoke_Test's checks in `cloud/smoke_runner.py`.
- **Train_Smoke_Check**: the check named `training` that this spec adds to SMOKE_CHECKS.
- **Smoke_Scene**: the tiny made-up scene the Train_Smoke_Check trains on.
- **Fixed_Constants**: what stays in code, not in Config, because it describes a format, a library or a contract and not a choice we tune: SH_C0; the Splat_Layout's 59 field names, their order and their type, and the highest SH_Degree, 3, whose 45 `f_rest_*` fields the layout holds; COLMAP's binary layout, the PINHOLE model's id, 1, and its 4 parameters; the file and folder names of the Camera_Solve_Output, the Trained_Splat and the Train_Record; the variable name `SPLATPIPE_CAP_DEADLINE`; GB as 10⁹ bytes; SSIM's constants C1 = 0.01² and C2 = 0.03², which `fused_ssim` itself passes to `FusedSSIMMap`, and fused-ssim's 11 × 11 Gaussian window with σ = 1.5, which its CUDA code fixes; gsplat's version, 1.5.3 (A10); and the Smoke_Scene's shape in its own unit, cameras on a circle of radius 1 around points in a cube of half-side 0.25, as the Smoke_Test's gsplat check already puts its Gaussians one unit from its camera. This is the same exception the pipeline-core spec's Requirement 1, criterion 6 makes for the Job_Id pattern.

Terms this spec uses as the pipeline-core spec and the Pipeline_Doc define them: Skeleton, Step, Step_Name, Job, Job_Id, Jobs_Folder, Job_Folder, Step_Runner, Step_Report, Step_List, Required_Step, Fake_Step, Work_Folder, Config, Config_Loader, Preset, Shared_Setting, Preset_Setting, Step_Section, Expected_Failure, Unexpected_Error, Readable_Error, Technical_Detail, Job_Runner, Job_Summary, JSON_Value, Pipeline_Doc, Licence_List, CLI and Test_Suite.

Terms it uses as the cloud-jobs spec defines them: Cloud_Runner, Cap_Deadline, User_Data, Code_Archive, Start_Command, Step_Needs, Fallback_List, Smoke_Test, Smoke_Command, Smoke_Marker, Image_Build, Cloud_Config and Project_File. From the aws-setup spec it uses Machine_Image, and from the frames spec, Frames_Step and Mask.

## Requirements

### Requirement 1: Reading the camera solve's files

**User Story:** As a team member building the camera solve, I want the train step to read one fixed layout and to say plainly when the files are missing or wrong, so that my step knows exactly what to write and a bad hand-off gives a message instead of a traceback.

**Decided:** D5 (criterion 2). **Assumptions (to confirm at review):** A5 (criteria 14 to 17) and A9 (criterion 2).

#### Acceptance Criteria

1. THE Train_Step SHALL read the Camera_Solve_Output only from `work/cameras/` in the Job_Folder: the Training_Images in `images/`, the Mask_Files in `masks/`, and the Sparse_Model's `cameras.bin`, `images.bin` and `points3D.bin` in `sparse/`, with these names as Fixed_Constants.
2. IF `work/cameras/images/` or any of the Sparse_Model's three files is missing, THEN THE Train_Step SHALL fail with an Expected_Failure, before the GPU checks of Requirement 2, whose Readable_Error names the missing path and says that the camera solve step writes that path, so the camera solve step has to run first (D5, A9).
3. THE Model_Reader SHALL read the Sparse_Model with numpy and the standard library's `struct` module, in COLMAP's little-endian binary layout as COLMAP's docs and COLMAP's `read_write_model.py` script describe the layout, and SHALL run no COLMAP program, import no pycolmap and hold no code copied from COLMAP.
4. THE Model_Reader SHALL take the Shared_Camera's id, model, width, height, fx, fy, cx and cy from `cameras.bin`, each image's id, Image_Pose, camera id and name from `images.bin`, and each point's position and colour from `points3D.bin`.
5. THE Model_Reader SHALL read past each image's keypoints and each point's error and track, keeping none of those values.
6. THE Model_Reader SHALL ignore every other file in `work/cameras/sparse/`, such as the `rigs.bin` and `frames.bin` that COLMAP 3.12 and later write beside the three files. COLMAP's docs say its newer cameras and images files keep the older layout.
7. IF `cameras.bin` holds a number of cameras other than 1, or the camera's model isn't PINHOLE, THEN THE Train_Step SHALL fail with an Expected_Failure whose Readable_Error names the file and gives the count or the model found.
8. IF a Sparse_Model file ends inside a record, holds bytes after the last record, or gives a count that the file's size can't hold, THEN THE Train_Step SHALL fail with an Expected_Failure whose Readable_Error names the file and says that the file is cut short or isn't a COLMAP binary model.
9. IF the Shared_Camera's width, height, fx or fy isn't above 0, a camera parameter, quaternion, translation or point position isn't finite, or an image's quaternion has a length of 0, THEN THE Train_Step SHALL fail with an Expected_Failure whose Readable_Error names the file and the camera, image or point.
10. IF `images.bin` holds no image, an image whose camera id isn't the Shared_Camera's, or two images with the same name, THEN THE Train_Step SHALL fail with an Expected_Failure whose Readable_Error names the file and the image.
11. IF an image's name in `images.bin` isn't a stem of one or more characters followed by `.jpg`, with no `/`, `\`, `:` or NUL character in the name, THEN THE Train_Step SHALL fail with an Expected_Failure whose Readable_Error gives the image's id, so a name can't lead outside `images/`.
12. IF an image that `images.bin` names has no file in `images/`, THEN THE Train_Step SHALL fail with an Expected_Failure whose Readable_Error names the image and the folder.
13. IF a Training_Image can't be read as a colour JPEG, or the image's width and height differ from the Shared_Camera's, THEN THE Train_Step SHALL fail with an Expected_Failure whose Readable_Error names the file and, when the sizes differ, gives both sizes.
14. WHERE `masks/<stem>.png` exists for a Training_Image named `<stem>.jpg`, THE Train_Step SHALL use that file as the image's Mask_File (A5).
15. WHERE a Training_Image has no Mask_File, THE Train_Step SHALL train on every pixel of the image (A5).
16. IF a Mask_File can't be read as a grey image, the Mask_File's width and height differ from its Training_Image's, or every pixel of the Mask_File is a Masked_Pixel, THEN THE Train_Step SHALL fail with an Expected_Failure whose Readable_Error names the file and the reason (A5).
17. THE Train_Step SHALL ignore every file in `images/` and `masks/` that no image in `images.bin` names (A5).
18. IF `points3D.bin` holds fewer than `start_scale_neighbours` + 1 points, THEN THE Train_Step SHALL fail with an Expected_Failure whose Readable_Error gives the count and the fewest points the step needs.
19. THE Train_Step SHALL train on every image in `images.bin` and hold none out for evaluation.
20. THE Train_Step SHALL use the Image_Poses, the Shared_Camera and the Sparse_Points in the Levelled_Frame exactly as the Sparse_Model gives the poses, the camera and the points, with no change to the world frame, so the Trained_Splat lies in the same frame as the cameras.
21. FOR ALL Sparse_Models of one PINHOLE camera with any images and any points that the Test_Suite's own writer writes, THE Model_Reader SHALL read back the same camera, the same Image_Poses, camera ids and names, and the same point positions and colours, bit for bit (property).

### Requirement 2: Checking the GPU before anything loads

**User Story:** As someone who films a place, I want the job to say at once when the machine's GPU is too small for my preset, so that it never trains for an hour and then crashes.

**Decided:** D1 (criteria 1, 4 and 6 to 11). **Assumptions (to confirm at review):** A9 (criterion 1) and A10 (criterion 5).

#### Acceptance Criteria

1. WHEN the Camera_Solve_Output's files are there, THE Train_Step SHALL import the GPU_Module and run the checks of criteria 2 to 10 before the Train_Step reads the Sparse_Model, any Training_Image or any Mask_File, and before anything goes onto the GPU (D1, A9).
2. IF torch, gsplat or fused-ssim can't be imported, THEN THE Train_Step SHALL fail with an Expected_Failure whose Readable_Error says that the step runs only on the GPU machine image and that a Job runs there through `uv run python -m cloud.start`.
3. IF torch sees no CUDA GPU, THEN THE Train_Step SHALL fail with an Expected_Failure whose Readable_Error says that the step needs an NVIDIA GPU and that torch can't see one on this machine.
4. THE Train_Step SHALL read the GPU's total memory in bytes from torch, as `torch.cuda.get_device_properties(0).total_memory`, and give the byte count to the GPU_Fit_Check with the Preset's `min_gpu_memory_gb` and `gpu_memory_slack` (D1).
5. IF the installed gsplat's version isn't 1.5.3, THEN THE Train_Step SHALL fail with an Expected_Failure whose Readable_Error gives the version found and says to rebuild the machine image (A10).
6. THE GPU_Fit_Check SHALL pass a GPU whose total memory in bytes is at least the GPU_Need, `gpu_memory_slack` × `min_gpu_memory_gb` × 10⁹ bytes, where GB means 10⁹ bytes, as in `gpu_memory_gb` in `[cloud.types]` (D1).
7. THE GPU_Fit_Check SHALL be a pure function of the byte count, `min_gpu_memory_gb` and `gpu_memory_slack` that imports no torch, so the Test_Suite tests the GPU_Fit_Check with no GPU (D1).
8. WHEN the GPU_Fit_Check gets 24,146,608,128 bytes, the 23,028 MiB that nvidia-smi shows for a "24 GB" A10G, with a `gpu_memory_slack` of 0.9, THE GPU_Fit_Check SHALL pass the GPU for a `min_gpu_memory_gb` of 24 and refuse the GPU for 48 (D1).
9. WHEN the GPU_Fit_Check gets 48,305,799,168 bytes, the 46,068 MiB that nvidia-smi shows for an L40S, with a `gpu_memory_slack` of 0.9, THE GPU_Fit_Check SHALL pass the GPU for a `min_gpu_memory_gb` of 24 and for 48 (D1).
10. IF the GPU_Fit_Check refuses the GPU, THEN THE Train_Step SHALL fail with an Expected_Failure whose Readable_Error gives the GPU's memory in GB, the Preset, the Preset's `min_gpu_memory_gb` and the slack, and says to run the Job with a Preset that needs less GPU memory or on a machine with a larger GPU (D1).
11. THE Train_Step SHALL run no trial Iteration to measure memory, and SHALL leave `min_vcpus` and `min_memory_gb` unchecked as cloud-only Step_Needs, so the step runs the same whatever the machine's vCPUs and memory, like the Frames_Step (D1).
12. IF torch runs out of GPU memory while the Train_Step loads the scene, sets up the Start_Splats or trains, THEN THE Train_Step SHALL fail with an Expected_Failure whose Readable_Error says that the GPU ran out of memory, gives the Iteration reached when training had started and the Preset's `max_splats`, and says to use a Preset with a smaller Splat_Budget or a GPU with more memory.

### Requirement 3: Setting up the scene

**User Story:** As someone who films a place, I want the training to start from the camera solve's points, at sizes that fit the scene, so that the splat grows from what the cameras saw.

**Decided:** D2 (criterion 12). **Assumptions (to confirm at review):** A1 (criterion 13) and A6 (criteria 12, 14 and 15).

#### Acceptance Criteria

1. THE Train_Step SHALL set the Training_Size to the Shared_Camera's width and height times `long_edge` / the longer of the two, each rounded to the nearest whole pixel, when the longer is above `long_edge`, and to the Shared_Camera's own width and height otherwise, so no image is scaled up.
2. THE Train_Step SHALL scale every Training_Image to the Training_Size and read the image's colours in RGB order, from 0 to 1, as each 8-bit value / 255.
3. THE Train_Step SHALL scale the Shared_Camera's fx and cx by the training width / the camera's width, and fy and cy by the training height / the camera's height.
4. THE Train_Step SHALL scale each Mask_File to the Training_Size with nearest-neighbour sampling, so each training pixel takes the value of the mask pixel under the training pixel's centre, and SHALL count as a Masked_Pixel each training pixel whose value is below `mask_threshold`.
5. THE Train_Step SHALL make each image's world-to-camera matrix from the image's Image_Pose, with the quaternion scaled to length 1, and each image's Camera_Centre as −Rᵀt.
6. THE Train_Step SHALL set the Scene_Extent to `scene_extent_factor` × the largest distance from a Camera_Centre to the mean of the Camera_Centres.
7. IF the largest distance from a Camera_Centre to the mean of the Camera_Centres is 0, THEN THE Train_Step SHALL fail with an Expected_Failure whose Readable_Error says that the camera didn't move, so no splat can be trained, and says to film while moving through the place.
8. THE Train_Step SHALL set the Near_Plane to `near_plane_factor` × the Median_Scene_Depth, leaving out of the median each camera with no Sparse_Point in front of the camera, so no splat can sit right on a camera.
9. IF no camera has a Sparse_Point in front of the camera, THEN THE Train_Step SHALL fail with an Expected_Failure whose Readable_Error says that the camera solve put no point in front of any camera.
10. THE Train_Step SHALL work out the Training_Size, the scaled camera, the Camera_Centres, the Scene_Extent, the Median_Scene_Depth and the Near_Plane with numpy functions that import no torch.
11. WHERE the Sparse_Points outnumber the Splat_Budget, THE Train_Step SHALL start from `max_splats` of the points, chosen at random with a generator seeded with `seed`, and SHALL add a warning to the Train_Report that gives both counts.
12. THE Train_Step SHALL give each Start_Splat the point's position; an `f_dc` of (the point's 8-bit colour / 255 − 0.5) / SH_C0 for each channel; 45 `f_rest` coefficients of 0; an opacity of logit(`start_opacity`); the identity rotation, (1, 0, 0, 0) with w first; and on all three axes a log scale of log(`start_scale_factor` × the Start_Distance) (D2, A6).
13. THE GPU_Module SHALL find each Start_Splat's `start_scale_neighbours` nearest other Start_Splats with torch on the GPU, and THE Train_Step SHALL turn the distances into log scales with a numpy function that imports no torch (A1).
14. IF a Start_Splat's Start_Distance is 0, THEN THE Train_Step SHALL give the Start_Splat the smallest Start_Distance above 0 among the Start_Splats (A6).
15. IF every Start_Distance is 0, THEN THE Train_Step SHALL fail with an Expected_Failure whose Readable_Error says that the camera solve's points all sit on top of each other (A6).
16. FOR ALL arrays of neighbour distances of 0 or more with at least one distance above 0, THE Train_Step SHALL give every Start_Splat a finite log scale, the same on all three axes (property).

### Requirement 4: Training with gsplat's MCMC strategy

**User Story:** As someone who films a place, I want my splat trained the way gsplat's MCMC recipe trains one, with my preset's numbers, so that I get a clean splat in the time the preset allows.

**Assumptions (to confirm at review):** A7 (criterion 16) and A8 (criteria 8 and 11).

#### Acceptance Criteria

1. THE Trainer SHALL render each Iteration's Training_Image with gsplat 1.5.3's `rasterization()` in the classic mode with `packed=False`, the Near_Plane as `near_plane`, the Training_Size, the scaled camera and the image's world-to-camera matrix, the splats' spherical harmonics at the Iteration's SH_Degree, and no background colour, so a pixel that no splat covers renders black.
2. THE Trainer SHALL hold the splats as gsplat's example does, as torch parameters for the positions, the log scales, the rotations, the opacities as logits, `sh0` for the band-0 coefficients and `shN` for the 15 higher coefficients of each channel, each with an Adam optimiser of one parameter group and `adam_eps`.
3. THE Trainer SHALL start the learning rates at the Means_LR of Iteration 0 for the positions and at `scales_lr`, `quats_lr`, `opacities_lr`, `sh_dc_lr` and `sh_rest_lr` for the others, which are gsplat's example rates.
4. THE Schedules SHALL give the Means_LR of Iteration i, for i from 0 to Planned_Iterations, as `means_lr` × Scene_Extent × `means_lr_final`^(i / Planned_Iterations), where Iteration i trains at the Means_LR of i, so the rate has fallen to `means_lr_final` (1%) of the rate's start when the last Planned_Iteration ends.
5. THE Trainer SHALL keep every learning rate other than the positions' the same through the run.
6. THE Schedules SHALL give Iteration i the SH_Degree min(i // `sh_degree_interval`, 3), so the degree goes up by one every 1,000 Iterations, to 3.
7. THE Trainer SHALL use gsplat's `MCMCStrategy` with `cap_max` = `max_splats`, `refine_start_iter` = `refine_start_step`, `refine_every` = `refine_every` and `refine_stop_iter` = the Refine_Stop, and SHALL pass no `noise_lr` or `min_opacity`, which stay at gsplat 1.5.3's defaults of 5e5 and 0.005.
8. THE Schedules SHALL give the Refine_Stop as min(`refine_stop_step`, round(`refine_stop_fraction` × Planned_Iterations)), which is 6,300 for `fast`, 13,500 for `default` and 25,000 for `quality` (A8).
9. THE Trainer SHALL hold at most `max_splats` splats at every Iteration, so the Trained_Splat holds at most the Splat_Budget.
10. THE Trainer SHALL call the strategy's `step_post_backward` once each Iteration, after the optimisers' step and the Means_LR update, with the Means_LR just set, in the order gsplat's example uses.
11. THE Trainer SHALL train on one Training_Image per Iteration and visit every Training_Image once per pass, in an order drawn for each pass from a generator seeded with `seed` (A8).
12. THE Trainer SHALL seed torch's random generators with `seed`.
13. THE Trainer SHALL run with no pose optimisation, no held-out images, no random background, no antialiased or 3DGUT mode, no bilateral grid, no viewer, no LPIPS, no TensorBoard and no checkpoints.
14. THE Trainer SHALL use only gsplat's library API, never `rasterization_inria_wrapper()` or `rasterization_2dgs_inria_wrapper()`, and SHALL hold no code from Inria's gaussian-splatting repo or from the MCMC paper's own repo, as rules.md and the gsplat-training skill require.
15. WHERE code in the GPU_Module is adapted from gsplat's examples, THE GPU_Module SHALL keep the examples' Apache-2.0 licence note, as the gsplat-training skill gives the note.
16. IF an Iteration's loss isn't finite, THEN THE Trainer SHALL stop at once with an error that the Step_Runner counts as an Unexpected_Error, whose message gives the Iteration and each Loss_Term's value (A7).
17. THE Trainer SHALL be one function, which the Train_Step and the Train_Smoke_Check both call, that takes the scene, the checked settings and the Stop_Time or none, and gives back the trained splats as numpy arrays, every Training_Image's Exposure_Fix, the Iterations run and the last Iteration's SH_Degree.
18. THE Schedules SHALL be pure functions that import no torch, which the Trainer calls.
19. FOR ALL `sh_degree_interval` values of 1 or more and all Iterations i and j with i ≤ j, THE Schedules SHALL give an SH_Degree of 0 at Iteration 0, and an SH_Degree at i that is no greater than the SH_Degree at j and never above 3 (property).
20. FOR ALL starting rates above 0, `means_lr_final` values above 0 and at most 1, and Planned_Iterations of 1 or more, THE Schedules SHALL give the starting rate at Iteration 0, a Means_LR that never rises from one Iteration to the next, and `means_lr_final` × the starting rate, within a relative error of 10⁻⁹, at Iteration Planned_Iterations, when the last Planned_Iteration has ended (property).

### Requirement 5: The loss

**User Story:** As someone who films a place, I want the training to compare only the parts of each frame that show the place, and to keep splats from turning into needles, so that the splat looks right from new viewpoints too.

**Decided:** D2 (criteria 2 and 3), D3 (criteria 6 to 9) and D7 (criteria 5 and 6). **Assumptions (to confirm at review):** A8 (criterion 7).

#### Acceptance Criteria

1. THE Trainer SHALL make each Iteration's loss the sum of the named Loss_Terms `l1`, `ssim`, `opacity_reg`, `scale_reg` and `erank`, each the Loss_Term's value times the Loss_Term's weight from Config: `l1_weight` (0.8), `ssim_weight` (0.2), `opacity_reg` (0.01), `scale_reg` (0.01) and `erank_reg` (0.01).
2. THE Trainer SHALL work out the `l1` Loss_Term as the mean absolute difference between the Corrected_Render and the Training_Image, over the 3 colour channels of every pixel that isn't a Masked_Pixel (D2).
3. THE Trainer SHALL work out the `ssim` Loss_Term as 1 − the mean of fused-ssim's per-pixel SSIM map, from `FusedSSIMMap` with `"same"` padding, over the 3 colour channels of every pixel that isn't a Masked_Pixel, after setting every Masked_Pixel to 0 in both the Corrected_Render and the Training_Image (D2).
4. THE Trainer SHALL leave every Masked_Pixel out of every Loss_Term that compares pixels.
5. THE Trainer SHALL work out the `opacity_reg` Loss_Term as the mean of every splat's opacity, the sigmoid of the splat's logit, as gsplat's `mcmc` preset does, and the `scale_reg` Loss_Term as the mean of every splat's three scales, the exponentials of the splat's log scales, each divided by the Scene_Extent, so the `scale_reg` Loss_Term doesn't depend on the scene's units (D7).
6. THE Trainer SHALL work out the `erank` Loss_Term as the mean over splats of max(−log(erank − 1 + `erank_epsilon`), 0) + the splat's smallest scale divided by the Scene_Extent, where erank = exp(−Σᵢ qᵢ log qᵢ) and qᵢ = sᵢ² / Σⱼ sⱼ² over the splat's three scales sᵢ in linear units (D3, D7).
7. THE Trainer SHALL add the `erank` Loss_Term only from the Erank_Start, round(`erank_start_fraction` × Planned_Iterations), which is Iteration 1,633 for `fast`, 3,500 for `default` and 7,000 for `quality` (D3, A8).
8. THE Trainer SHALL keep the `erank` Loss_Term finite for every splat by taking qᵢ log qᵢ as 0 where qᵢ is 0, so a splat whose smallest scales round to 0 can't make the loss NaN (D3).
9. THE Trainer SHALL compute the Effective_Rank_Term from the formula of D3, written fresh, with no code from the paper's own repository (D3).
10. THE Trainer SHALL keep each Training_Image's inputs (the image's pixels, Masked_Pixels, camera, world-to-camera matrix and Exposure_Fix) in one record per image and build the loss from the named Loss_Terms, so the depth priors spec can add a Loss_Term, with its own weight in `[train]` and its own per-image input such as a depth map, without changing the five Loss_Terms of criterion 1.

### Requirement 6: The exposure fix

**User Story:** As someone who films a place with a phone that changes its exposure as I walk, I want each frame's brightness and colour shift learned on its own, so that the splat doesn't fill with blotches that explain them.

#### Acceptance Criteria

1. THE Trainer SHALL give each Training_Image an Exposure_Fix, starting at A = the 3 × 3 identity and b = 0.
2. THE Trainer SHALL apply each Training_Image's Exposure_Fix to every pixel of the image's render, turning colour c into A·c + b, to make the Corrected_Render that the Loss_Terms compare with the Training_Image.
3. THE Trainer SHALL train the Exposure_Fixes with one Adam optimiser at `exposure_lr` (0.001) with `adam_eps`, kept apart from the optimisers the MCMC strategy is given, because the strategy resizes every optimiser the strategy holds along with the splats.
4. WHEN training ends, THE Train_Step SHALL work out the Mean_Exposure, [Ā | b̄], as the element-wise mean of every Training_Image's A and of every b.
5. THE Exposure_Bake SHALL multiply the RGB vector of each of a splat's higher-band coefficients, in `f_rest`, by Ā, and SHALL set the splat's `f_dc` to Ā·f_dc + (Ā·0.5·**1** + b̄ − 0.5·**1**) / SH_C0, where **1** is (1, 1, 1).
6. THE Exposure_Bake SHALL be a numpy function that imports no torch.
7. THE Train_Step SHALL write the Trained_Splat with the Mean_Exposure baked in, so a viewer that shows the splat with no Exposure_Fix shows the average frame's exposure.
8. FOR ALL splats' coefficients, Mean_Exposures and vectors of spherical harmonic basis values whose band-0 value is SH_C0, THE Exposure_Bake SHALL give each splat the colour Ā·c + b̄, to within floating-point rounding, where c is the splat's colour before the bake: 0.5 plus the sum of each basis value times the matching coefficient's RGB vector, taken before the clamp at 0 (property).

### Requirement 7: Stopping before the time cap

**User Story:** As someone who films a place, I want the training to stop in time and still save my splat when the job runs short of time, so that a long video gives me a splat that trained for less time rather than nothing.

**Assumptions (to confirm at review):** A4 (criteria 3 and 5) and A9 (criterion 3).

#### Acceptance Criteria

1. WHEN the Train_Step has checked its settings, THE Train_Step SHALL read the Cap_Deadline from the environment variable `SPLATPIPE_CAP_DEADLINE`, which the Cloud_Runner sets.
2. WHERE `SPLATPIPE_CAP_DEADLINE` isn't set, THE Train_Step SHALL run every Planned_Iteration with no Stop_Time.
3. IF `SPLATPIPE_CAP_DEADLINE` is set but empty, isn't a date and time that Python 3.11's `datetime.fromisoformat` reads, or has no UTC offset, THEN THE Train_Step SHALL fail with an Expected_Failure, before the Train_Step checks the Camera_Solve_Output, whose Readable_Error names the variable, gives the variable's value and gives an example of the form the value must have (A4, A9).
4. THE Train_Step SHALL accept both forms the cloud code writes: `2026-11-21T14:25:30Z`, from the User_Data, and `2026-11-21T14:25:30.123456+00:00`, from the Cloud_Runner's `isoformat()`.
5. THE Train_Step SHALL set the Train_Deadline to the Cap_Deadline minus `clean_reserve_minutes` (10) and the Stop_Time to the Train_Deadline minus `save_margin_minutes` (5) (A4).
6. WHILE a Stop_Time is set, THE Trainer SHALL check the clock before each Iteration and SHALL start no Iteration at or after the Stop_Time.
7. WHEN the Trainer stops before the last Planned_Iteration, THE Train_Step SHALL still write and check the Trained_Splat from the Iterations run, as Requirement 8 says for a full run.
8. WHEN the Trainer stops before the last Planned_Iteration, THE Train_Step SHALL add a warning to the Train_Report that gives the Iterations run and the Planned_Iterations and says that training stopped early to leave time for the clean step before the Job's time cap.
9. IF the clock is at or after the Stop_Time before the first Iteration, THEN THE Train_Step SHALL fail with an Expected_Failure whose Readable_Error says that no time was left to train before the Job's time cap and names `clean_reserve_minutes` and `save_margin_minutes` in `[train]`.
10. THE Trainer SHALL plan the Means_LR, the Refine_Stop and the Erank_Start from the Planned_Iterations whether or not the Trainer stops early.
11. THE Train_Step SHALL read the Cap_Deadline, work out the Stop_Time and decide before each Iteration whether to stop with pure functions that take the time as an argument and import no torch, so the Test_Suite tests those functions with a fake clock.

### Requirement 8: Writing and checking the trained splat

**User Story:** As a team member building the clean step, I want the trained splat in one known layout, checked before the step calls itself done, so that my step can read it without guessing and never gets a broken file.

**Decided:** D6 (criterion 9).

#### Acceptance Criteria

1. WHEN the Train_Step's own code starts, THE Train_Step SHALL first remove `out/train.ply` when that file exists, so a run that fails leaves no splat from an earlier run in `out/`. A failure that the Step_Runner raises before the step's own code runs, such as a `[train]` section that breaks the Config_Loader's rules, still fails the Job, because the Train_Step is a Required_Step.
2. WHEN training ends, THE Train_Step SHALL write the splats, with the Mean_Exposure baked in, to `work/train/train.ply` with the PLY_Writer, our own numpy code, and not with gsplat's `export_splats()`.
3. THE PLY_Writer SHALL write the Splat_Layout: the header lines `ply`, `format binary_little_endian 1.0`, `element vertex <N>`, one `property float <name>` line for each of the 59 fields `x`, `y`, `z`, `f_dc_0` to `f_dc_2`, `f_rest_0` to `f_rest_44`, `opacity`, `scale_0` to `scale_2` and `rot_0` to `rot_3`, in that order, and `end_header`, each line ending in `\n`; then N little-endian records of the 59 fields as float32, 236 bytes each, with no other element, no comment and no normals.
4. THE PLY_Writer SHALL write the `f_rest_*` fields channel-major, so `f_rest_k` holds colour channel k // 15's coefficient k % 15, with each channel's coefficients in band order, as the splat-files skill describes.
5. THE PLY_Writer SHALL write each opacity as its logit, before the sigmoid, each scale as its log, and each rotation's four values w first, as the Trainer holds the values.
6. WHEN the PLY_Writer has written `work/train/train.ply`, THE Train_Step SHALL run the PLY_Check on the written file, which passes only a file whose header is exactly the Splat_Layout's, with an N of 1 or more, whose size is the header's size plus N × 236 bytes, and whose every value is finite.
7. IF the PLY_Check fails, THEN THE Train_Step SHALL fail with an Expected_Failure whose Readable_Error names the file and what failed: no splats, a header that differs from the Splat_Layout's, with the fields missing or out of place, a size that doesn't match N, or how many splats hold a NaN or an infinity and in which fields.
8. WHEN the PLY_Check passes, THE Train_Step SHALL move `work/train/train.ply` to `out/train.ply` with one rename on the same file system, so `out/` never holds a half-written Trained_Splat.
9. WHEN the Mean_Exposure is baked in, THE Train_Step SHALL drop every splat that has a NaN or an infinity in any of the splat's 59 fields, judging each value as the float32 that the PLY_Writer writes for it, with a numpy function that imports no torch, keeping the other splats in their order, before the PLY_Writer writes `work/train/train.ply`, and SHALL add a warning to the Train_Report that gives how many splats the Train_Step dropped, when the Train_Step dropped any (D6). The drop comes after the bake, and judges float32 values, because a baked value can overflow to an infinity when it's rounded to float32. The Train_Report's `splats` count stays the number of splats written. When the Train_Step drops every splat, the PLY_Writer writes a file of 0 splats, which the PLY_Check refuses with criterion 7's existing "no splats" Expected_Failure, and the failed Train_Report already holds the warning with the count, so the report says why.
10. IF writing or moving the Trained_Splat fails because the disk is full or the file can't be written, THEN THE Train_Step SHALL fail with an Expected_Failure whose Readable_Error names the file and the reason.
11. THE Train_Step SHALL write only inside `work/train/` and to `out/train.ply` in the Job_Folder, beside the Train_Report that the Step_Runner writes.
12. THE Train_Step SHALL leave `out/train.ply` unchanged once the step is done, so the Clean_Step reads the Trained_Splat from `out/train.ply`, and `out/train.ply` stays the Job's result when the Clean_Step fails or doesn't run.
13. THE PLY_Reader SHALL read a file in the Splat_Layout back into one float32 array for each field.
14. FOR ALL sets of 1 or more splats with finite float32 values, THE PLY_Check SHALL pass the file that the PLY_Writer writes from those splats (property).
15. FOR ALL files that the PLY_Writer writes from 1 or more splats with finite values, THE PLY_Check SHALL refuse the file once any one value in the file is set to NaN, to positive infinity or to negative infinity (property).
16. FOR ALL files that the PLY_Writer writes from 1 or more splats with finite values, THE PLY_Check SHALL refuse a file written the same way with any one of the 59 fields left out of its header and its records (property).
17. FOR ALL sets of splats, of any count and with any float32 values, NaN and infinities included, THE PLY_Reader SHALL read back from the PLY_Writer's file the same values, bit for bit (property).

### Requirement 9: Settings, presets and dependencies

**User Story:** As a team member, I want every number of the training in Config, with three Presets, so that we can tune the training without changing code.

**Assumptions (to confirm at review):** A1 (criteria 10 to 14), A4 and A8 (criterion 2).

#### Acceptance Criteria

1. THE Config SHALL hold a `[train]` Step_Section with the three Preset tables `[train.fast]`, `[train.default]` and `[train.quality]`, each with the same keys, and with a short comment for every key.
2. THE `[train]` section SHALL hold these Shared_Settings: `long_edge = 1600`, `mask_threshold = 128`, `seed = 42`, `sh_degree_interval = 1000`, `scene_extent_factor = 1.1`, `near_plane_factor = 0.03`, `start_opacity = 0.5`, `start_scale_factor = 0.1`, `start_scale_neighbours = 3`, `neighbour_chunk_pairs = 100000000`, `means_lr = 0.00016`, `means_lr_final = 0.01`, `scales_lr = 0.005`, `quats_lr = 0.001`, `opacities_lr = 0.05`, `sh_dc_lr = 0.0025`, `sh_rest_lr = 0.000125`, `adam_eps = 1e-15`, `exposure_lr = 0.001`, `l1_weight = 0.8`, `ssim_weight = 0.2`, `opacity_reg = 0.01`, `scale_reg = 0.01`, `erank_reg = 0.01`, `erank_epsilon = 1e-5`, `erank_start_fraction = 0.23333333333333334`, `refine_start_step = 500`, `refine_every = 100`, `refine_stop_step = 25000`, `refine_stop_fraction = 0.9`, `clean_reserve_minutes = 10`, `save_margin_minutes = 5`, `gpu_memory_slack = 0.9`, `min_vcpus = 4` and `min_memory_gb = 32` (A4, A8).
3. THE Preset tables SHALL hold these Preset_Settings, for `fast`, `default` and `quality` in turn: `steps` 7000, 15000 and 30000; `max_splats` 1000000, 2000000 and 3000000; and `min_gpu_memory_gb` 24, 24 and 48. `fast` stays the default Preset, as `[jobs].default_preset` already says.
4. THE Config SHALL hold each `[train]` key either as a Shared_Setting or in every Preset table, never both, as the Config_Loader requires.
5. THE Train_Step SHALL take every rate, weight, count, size, factor and time from Config, and hold in code only the Fixed_Constants.
6. IF a setting is missing from `[train]` and its Preset table, THEN THE Step_Runner's existing check SHALL fail the Step with the Step_Runner's Readable_Error, which names the section, the table and the key.
7. IF a setting is present but isn't the kind of value the setting must be, THEN THE Train_Step SHALL fail with an Expected_Failure, before the Train_Step reads the Cap_Deadline, whose Readable_Error names the setting, the setting's table and what the setting must be: `long_edge`, `sh_degree_interval`, `start_scale_neighbours`, `neighbour_chunk_pairs`, `refine_every`, `steps`, `max_splats`, `min_vcpus`, `min_memory_gb` and `min_gpu_memory_gb` whole numbers above 0; `seed`, `refine_start_step` and `refine_stop_step` whole numbers of 0 or more; `mask_threshold` a whole number from 1 to 255; `scene_extent_factor`, `near_plane_factor`, `start_scale_factor`, `means_lr`, `scales_lr`, `quats_lr`, `opacities_lr`, `sh_dc_lr`, `sh_rest_lr`, `adam_eps`, `exposure_lr` and `erank_epsilon` numbers above 0; `l1_weight`, `ssim_weight`, `opacity_reg`, `scale_reg`, `erank_reg`, `clean_reserve_minutes` and `save_margin_minutes` numbers of 0 or more, where a weight of 0 turns its Loss_Term off; `start_opacity` a number above 0 and below 1; `means_lr_final` and `gpu_memory_slack` numbers above 0 and at most 1; and `erank_start_fraction` and `refine_stop_fraction` numbers from 0 to 1. A bool is never a number here.
8. IF `max_splats` is below `start_scale_neighbours` + 1, THEN THE Train_Step SHALL fail with an Expected_Failure, before the Train_Step reads the Cap_Deadline, whose Readable_Error names both settings and their values.
9. THE Config's `min_vcpus`, `min_memory_gb` and `min_gpu_memory_gb` in `[train]` and its Preset tables SHALL be the Train_Step's Step_Needs, which the Start_Command reads to pick the machine: at least 4 vCPUs, 32 GB of memory, and a 24 GB GPU for `fast` and `default` or a 48 GB GPU for `quality`.
10. THE Project_File SHALL gain no package, so `uv.lock` and the Machine_Image stay unchanged, and `uv sync` on a laptop still installs no torch, no gsplat, no fused-ssim and no CUDA wheel (A1).
11. THE Train_Step's modules other than the GPU_Module SHALL import only the Python standard library, numpy, OpenCV (`cv2`) and `splatpipe`, and THE GPU_Module SHALL import, besides those, only torch, gsplat and fused-ssim (A1).
12. THE Train_Step SHALL import the GPU_Module only inside the step's own code, when the step runs, so importing `splatpipe.train` on a laptop imports no GPU package (A1).
13. THE Train_Step SHALL import nothing from `cloud/` or boto3 (A1).
14. WHERE a task of this spec adds a dependency, a tool, a model weight or a dataset, THE Licence_List SHALL gain that item's line, with the item's licence and how we use the item, in the same commit (A1).
15. THE `tests/conftest.py` test config SHALL gain a `[train]` section with the section's three Preset tables, so a test that runs the Train_Step without a Config of its own gets settings, and a test SHALL be able to change any one of the settings for that test, as `set_frames` does for `[frames]`.

### Requirement 10: The step report

**User Story:** As a team member, I want the step's report to say what the training started from, how far it got and how much GPU memory it used, so that I can tell a good run from a cut-short one and tune the presets.

**Decided:** D6 (criterion 5). **Assumptions (to confirm at review):** A3 (criterion 7).

#### Acceptance Criteria

1. WHEN the Train_Step is done, THE Train_Report SHALL hold the counts `images` (the Training_Images), `masked_images` (the Training_Images with a Mask_File), `image_width` and `image_height` (the Training_Size), `sparse_points`, `start_splats`, `splats` (in the Trained_Splat), `steps_planned` (the Planned_Iterations), `steps_run` (the Iterations run), `sh_degree` (the last Iteration's SH_Degree), `gpu_memory_bytes` (the GPU's total memory, as torch reports the total), `peak_gpu_memory_bytes` (the most memory torch's allocator held at once during the step, from `torch.cuda.max_memory_allocated()`) and `ply_bytes` (the Trained_Splat's size).
2. WHEN the Train_Step is done, THE Train_Report SHALL hold the timings `load` (reading the Camera_Solve_Output and setting up the scene and the Start_Splats), `train`, `write_ply` and `check_ply`, each in seconds.
3. WHEN the Train_Step fails, THE Train_Report SHALL hold exactly the counts and timings of criteria 1 and 2 that the step had worked out before the step failed.
4. THE Train_Step SHALL add each count as a whole number given as an int and each timing as a number of seconds, so the Step_Runner's checks never refuse one.
5. WHERE the step warns under Requirement 3, criterion 11, Requirement 7, criterion 8 or Requirement 8, criterion 9, THE Train_Report SHALL hold that warning. The warning of Requirement 8, criterion 9 reads "Splats left out of out/train.ply because they held a NaN or an infinity: `<n>`.", where `<n>` is the number of splats dropped (D6).
6. THE Train_Report SHALL be the Step_Runner's ordinary Step_Report, written to `out/reports/train.json`, with the 38 settings of Requirement 9, criteria 2 and 3, which the Step_Runner puts there itself.
7. WHEN the Trained_Splat is in `out/`, THE Train_Step SHALL write the Train_Record, `work/train/train.json`, holding only JSON_Values: the Near_Plane, the Median_Scene_Depth, the Scene_Extent, the starting Means_LR, the ratios of the Training_Size's width and height to the Shared_Camera's, the Mean_Exposure as 3 rows of 4 numbers, and the Cap_Deadline and the Stop_Time as ISO 8601 UTC times, both `null` when no Cap_Deadline was set (A3).

### Requirement 11: Running the step and the step list

**User Story:** As a team member, I want the train step to run on its own and as the step after frames in every job, so that `splatpipe.run` goes on from the picked frames to a splat once the camera solve lands.

**Decided:** D5 (criteria 4, 5 and 11). **Assumptions (to confirm at review):** A9 (criterion 3).

#### Acceptance Criteria

1. THE Train_Step SHALL be a package `splatpipe/train/` built like the Fake_Step: `work(step)`, `run(job_id, preset=None, *, config=None)` and a `__main__.py` whose `main(argv)` calls the Skeleton's `step_main`.
2. THE CLI `uv run python -m splatpipe.train --job <id> [--preset P]` SHALL run the step on one Job, print `train: done` on standard output and exit with 0 when the step is done, and print the Readable_Error on standard error with no traceback and exit with 1 when the step failed.
3. THE Train_Step SHALL do the step's work in this order: remove an earlier `out/train.ply`; check the settings; read the Cap_Deadline; check that the Camera_Solve_Output's files are there; import the GPU_Module and check the GPU; read the Sparse_Model, the Training_Images and the Mask_Files; set up the scene and the Start_Splats; train; write, check and move the Trained_Splat; and write the Train_Record (A9).
4. THE Step_List in `splatpipe/run.py` SHALL be `(StepEntry("frames", "splatpipe.frames", True), StepEntry("train", "splatpipe.train", True))`, with the Step_List's comment saying that the camera solve spec puts its entry between the two (D5).
5. WHILE the Step_List holds no Camera_Solve_Step entry, THE Job_Runner SHALL run the Frames_Step and then fail a Job that has no Camera_Solve_Output at the Train_Step, with the Readable_Error of Requirement 1, criterion 2 as the Job_Summary's `readable_error` (D5).
6. THE Train_Step SHALL reach the GPU_Module through one module-level hook that the Test_Suite can replace with a GPU_Stand_In, as the Frames_Step reaches ffmpeg through `TOOLS`, so the step's own code runs on a CPU in the tests.
7. THE Train_Step SHALL raise `ExpectedFailure` for every failure that Requirements 1, 2, 3, 7, 8 and 9 name, so the Train_Report's `error.expected` is `true` and the report's `readable` is the step's own message.
8. WHEN the Train_Step runs again on a Job, THE Train_Step SHALL rely on the Step_Runner emptying `work/train/` first, and on Requirement 8, criterion 1 for `out/train.ply`.
9. THE `tests/test_run.py` test that asserts the Step_List SHALL assert the Step_List of criterion 4, in the same task that changes the Step_List.
10. THE `tests/test_frames_step.py` test that runs the Job_Runner with the real ffmpeg SHALL give the Job_Runner a Step_List of the Frames_Step's entry alone, so the test still expects the Job to be done.
11. THE Test_Suite SHALL check criterion 5 on a laptop, with the Frames_Step's stand-in tools: a Job run through the real Step_List ends `failed` at the Train_Step, with the Frames_Step done and the Readable_Error of Requirement 1, criterion 2 (D5).

### Requirement 12: The smoke test's training check

**User Story:** As a team member, I want the smoke test to run the real trainer on the GPU for a few iterations, so that a broken image or a bug in the training code shows up in a short smoke test and not an hour into a job.

**Decided:** D6 (criterion 7). **Assumptions (to confirm at review):** A11 (criteria 3 to 6) and A12 (criterion 12).

#### Acceptance Criteria

1. THE SMOKE_CHECKS SHALL gain the Train_Smoke_Check, named `training`, just before `cloud-runner`, which stays last.
2. THE Train_Smoke_Check SHALL build the Smoke_Scene with a numpy function that imports no torch: `train_images` images of `image_pixels` × `image_pixels` pixels sharing one PINHOLE camera, with a mask on at least one image, `train_points` Sparse_Points in front of every camera, and cameras set apart so the Scene_Extent is above 0, all drawn from a generator seeded with `[train]`'s `seed`.
3. THE Train_Smoke_Check SHALL run the Trainer on the GPU with the `[train]` settings of the Preset that `[jobs].default_preset` names, except that `[cloud.smoke]` sets the Planned_Iterations (`train_steps`), the Splat_Budget (`train_splats`), `refine_start_step` (`train_refine_start_step`), `refine_every` (`train_refine_every`) and `sh_degree_interval` (`train_sh_degree_interval`) (A11).
4. THE `[cloud.smoke]` table SHALL gain `train_images = 4`, `train_points = 200`, `train_splats = 400`, `train_steps = 30`, `train_refine_start_step = 10`, `train_refine_every = 10` and `train_sh_degree_interval = 5`, each with a comment (A11).
5. THE Cloud_Config SHALL gain a field in `SmokeChecks` for each new key, which the loader in `cloud/config.py` reads and checks as a whole number above 0, refusing a bad value with the loader's Readable_Error that names the key (A11).
6. THE `[cloud.smoke]` numbers SHALL make the Train_Smoke_Check's run turn the `erank` Loss_Term on, reach SH_Degree 3 and include at least one MCMC refine, which the Test_Suite checks on the CPU with the Schedules and the real Config (A11).
7. WHEN the Trainer returns, THE Train_Smoke_Check SHALL bake the Mean_Exposure in, write every splat the Trainer gave back into the check's scratch folder with the PLY_Writer, dropping none, and run the PLY_Check on the file, so a splat with a NaN or an infinity fails the PLY_Check and so the Train_Smoke_Check, because the Smoke_Scene should never make one (D6).
8. THE Train_Smoke_Check SHALL pass only when the Trainer ran every planned Iteration and the PLY_Check passed, and SHALL give back a line for the log with the Iterations run, the splats written and the SH_Degree reached.
9. IF the Trainer or the PLY_Check fails, THEN THE Train_Smoke_Check SHALL raise an error whose message says what failed, so the Smoke_Test reports the check as failed and goes on with the other checks.
10. THE Train_Smoke_Check SHALL import the GPU_Module inside the check's own function, as the other GPU checks import torch, so `cloud/smoke_runner.py` still imports on a laptop.
11. THE Train_Smoke_Check SHALL finish within `[cloud.smoke]`'s `check_timeout_seconds` (60) on the Smoke_Test's g5.xlarge, in line with the few seconds each check takes (cloud-jobs Requirement 11, criterion 2).
12. THE Smoke_Test SHALL run the Train_Smoke_Check on the current Machine_Image with no Image_Build, because the Smoke_Command uploads the Code_Archive that holds the check. Running the Smoke_Test again is a paid run that needs the person's yes, and no task of this spec runs the Smoke_Test (A12).
13. THE `tests/test_cloud_smoke_runner.py` test of the checks' order, the `[cloud.smoke]` tables in `tests/fake_aws.py` and `tests/test_cloud_config.py`, and the Pipeline_Doc's smoke test part SHALL change in the same task as SMOKE_CHECKS.

### Requirement 13: Docs and the gsplat skill

**User Story:** As one of the three people building the steps, I want the train step and its hand-offs written up next to the other steps, so that the camera solve and the clean step fit with it.

**Decided:** D4 (criterion 7) and D5 (criterion 8). **Assumptions (to confirm at review):** A5 (criterion 2) and A13 (criterion 12).

#### Acceptance Criteria

1. THE Pipeline_Doc SHALL gain a `## Step: train` section, after `## Step: frames`, with the same parts as the Fake_Step's section: `### What it reads`, `### What it writes`, `### Config section`, `### Step report fields` and `### Files in out/`.
2. THE `### What it reads` part SHALL state the Camera_Solve_Output as the contract the camera solve spec writes: `work/cameras/images/<name>.jpg`, the undistorted colour frames; `work/cameras/masks/<name>.png` when an image has a mask, where black means ignore and a value below 128 counts as black, so the camera solve step should write only 0 and 255; `work/cameras/sparse/` with COLMAP's binary `cameras.bin` (one shared PINHOLE camera), `images.bin` and `points3D.bin` in the Levelled_Frame; the match of `<name>.jpg` to `masks/<name>.png` by stem; and what the step refuses (A5).
3. THE `### What it reads` part SHALL also name `SPLATPIPE_CAP_DEADLINE` and the step's settings.
4. THE `### What it writes` part SHALL give the Splat_Layout, the Work_Folder's files and the Train_Record's fields.
5. THE `### Config section` part SHALL give every key of `[train]` and its Preset tables, with the key's table, value and meaning, and SHALL state the Step_Needs: 4 vCPUs, 32 GB of memory, and a 24 GB GPU for `fast` and `default` or a 48 GB one for `quality`, of which the step itself checks only the GPU's memory, against the GPU_Need.
6. THE `### Step report fields` part SHALL give the counts, the timings, the warnings and the kinds of Expected_Failure of Requirements 10 and 11.
7. THE `### Files in out/` part SHALL say that the step removes `out/train.ply` when the step starts and writes the file whole once the PLY_Check passes, that the Clean_Step reads the file from there, that the file is the Job's result when the Clean_Step fails or doesn't run, and that the largest Splat_Budget is 3,000,000 splats, which is within the 8,000,000 splats the Clean_Step takes (D4).
8. THE Pipeline_Doc's "The step list" section SHALL say that the list holds the Frames_Step and then the Train_Step, both required, that the camera solve spec puts its entry between the two and a later spec adds the Clean_Step's entry after the Train_Step's, so the order becomes frames, cameras, train, clean, and that until the camera solve spec's entry is there a Job runs the Frames_Step and fails at the Train_Step because the camera solve output is missing, in place of the sentence that says a Job is done once the Frames_Step is done (D5), and SHALL give the planned optional steps as "depth priors and clean", in place of "depth priors and clean and export".
9. THE Pipeline_Doc's "The cap deadline" part SHALL say that the Train_Step reads `SPLATPIPE_CAP_DEADLINE` and stops training early so that `out/train.ply` is written and checked `clean_reserve_minutes` before the Cap_Deadline, in place of "No current step reads it; the training spec will."
10. THE Pipeline_Doc's Step_Needs table SHALL give Training as 4 vCPUs, 32 GB RAM and a GPU with 24 GB (fast and default) or 48 GB (quality), in place of "a GPU, 32 GB RAM", and SHALL name the clean step's row "Clean", in place of "Clean and export", with its 8 vCPUs and 8 GB RAM unchanged.
11. THE Pipeline_Doc's "The smoke test" part SHALL list the Train_Smoke_Check among the checks and say that the check's numbers are in `[cloud.smoke]`.
12. THE Pipeline_Doc's frames section SHALL say that the camera solve step passes the Mask on to training as one mask per image in `work/cameras/masks/`, in place of "the training loss reads the same file" (A13).
13. THE gsplat-training skill's line that says the clean step reads `out/splat.ply` SHALL say that the Clean_Step reads `out/train.ply`, which the Train_Step writes, and that the Clean_Step takes at most 8,000,000 splats, the Clean_Step's `max_source_splats`, so the MCMC strategy's `cap_max` stays at or under that number.
14. THE `tests/test_docs.py` SHALL check the step list sentence of criterion 8 in place of the old one, the frames phrase of criterion 12 in place of the old one, and that the `## Step: train` section's parts name every key of the real `[train]`, every count and timing of Requirement 10, and `out/train.ply`.
15. THE `tests/test_cloud_docs.py` row check for Training SHALL look for "4 vCPUs", "32 GB RAM", "24 GB" and "48 GB".
16. THE Licence_List's `opencv-python-headless` row SHALL say in its "How we use it" cell that the Train_Step decodes the Camera_Solve_Output's JPEG frames and PNG masks and scales them down, and SHALL still say that it never reads or writes video, which `tests/test_docs.py` checks.

### Requirement 14: Tests

**User Story:** As a team member, I want the step's CPU parts tested on a laptop with tiny made-up data, so that the tests are quick, free and safe to run anywhere, and the GPU only runs code that is already tested.

**Decided:** D6 (criteria 1 and 4). **Assumptions (to confirm at review):** A1 (criterion 7) and A14 (criterion 8).

#### Acceptance Criteria

1. THE Test_Suite SHALL test with unit tests, on the CPU with tiny synthetic data and no GPU: the Schedules (the SH_Degree, the Means_LR, the Refine_Stop and the Erank_Start, the examples of Requirement 4, criterion 8 and Requirement 5, criterion 7 included), the Training_Size and the scaled camera, the Masked_Pixels and the mask scaling, the Camera_Centres, the Scene_Extent, the Median_Scene_Depth and the Near_Plane (cameras with no point in front included), the start log scales (Start_Distances of 0 included), the subset of Sparse_Points, the Mean_Exposure and the Exposure_Bake, the drop of Requirement 8, criterion 9, which keeps exactly the splats that are finite in every field, in their order, and gives the count dropped, the PLY_Writer, the PLY_Reader and the PLY_Check, the Model_Reader with a small writer of COLMAP's binary model in the tests, the setting checks, the Cap_Deadline's two forms and its refusals, the Stop_Time and the stop decision with a fake clock, the GPU_Fit_Check with the byte counts of Requirement 2, and the report fields.
2. THE Test_Suite SHALL hold the property tests of Requirement 1, criterion 21, Requirement 3, criterion 16, Requirement 4, criteria 19 and 20, Requirement 6, criterion 8, and Requirement 8, criteria 14 to 17, written with Hypothesis.
3. THE Test_Suite's unit tests and property tests SHALL both be required, never optional.
4. THE Test_Suite SHALL run the Train_Step through `run()` on the CPU with a GPU_Stand_In, to test the order of the step's work, every Expected_Failure that needs no GPU, a done run that leaves a checked Trained_Splat in `out/` and the Train_Record in `work/train/`, an early stop's warning, the drop of a NaN splat from the GPU_Stand_In with the warning of Requirement 8, criterion 9 in a run that is done with a checked `out/train.ply`, the failure of a run in which every splat has a NaN or an infinity with the PLY_Check's "no splats" Expected_Failure, that warning in the Train_Report and no `out/train.ply`, and that a failed run leaves no `out/train.ply` from an earlier run (D6).
5. THE Test_Suite SHALL check, by listing the Job_Folder before and after a run with the GPU_Stand_In, that the Train_Step writes nothing outside `work/train/`, `out/train.ply` and the Train_Report.
6. THE Test_Suite SHALL check that each Readable_Error of the step's Expected_Failures names what the requirement says the Readable_Error names and holds no traceback.
7. THE Test_Suite SHALL check that every file in `splatpipe/train/` other than the GPU_Module imports only names in the standard library, numpy, `cv2` and `splatpipe`, that the GPU_Module imports, besides those, only torch, gsplat and fused_ssim, and that `import splatpipe.train` in a fresh process loads no torch, gsplat or fused_ssim (A1).
8. THE `tests/test_cloud_imports.py` check of the cloud-jobs spec's Requirement 16, criterion 7 SHALL keep refusing `cloud` and boto3 in every file under `splatpipe/`, and SHALL keep refusing every GPU package in every file except the GPU_Module, which may import torch, gsplat and fused_ssim and no other GPU package (A14).
9. THE Test_Suite SHALL check that `[train]` loads from the real Config for each of the three Presets with exactly the keys of Requirement 9, and that, with the real Config and Step_List, every type on each Preset's Fallback_List meets the Step_Needs that the Start_Command works out.
10. THE Test_Suite SHALL check the Train_Smoke_Check's place in SMOKE_CHECKS, the Smoke_Scene on the CPU against the step's own scene checks, the Schedules of Requirement 12, criterion 6, and the new `[cloud.smoke]` keys and their refusals.
11. THE Test_Suite SHALL run the Train_Step's tests with `SPLATPIPE_CAP_DEADLINE` removed from the environment unless a test sets the variable, so a deadline left by another test or by the shell can't change those tests.
12. THE Test_Suite SHALL run on the CPU with no network, make every Job in a temporary Jobs_Folder and never in the real `data/` folder, build every image, model and splat file in the test, and commit none to git.
