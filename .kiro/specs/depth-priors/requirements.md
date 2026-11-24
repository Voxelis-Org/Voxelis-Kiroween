# Requirements Document

## Introduction

This spec builds the Depth_Step and changes the Train_Step so it can use what the Depth_Step writes. The Depth_Step runs MoGe-2 on every frame of the camera solve, fits each frame's depth map to the camera solve's own points, and writes the frames that fit as depth priors in `work/depth/`. When the priors are there, the Train_Step starts extra splats on plain surfaces from them and adds five loss terms that pull the rendered depth and normals toward them. This helps training where COLMAP found few points, like plain walls, floors and ceilings.

The Depth_Step is an Optional_Step between the Camera_Solve_Step and the Train_Step: `StepEntry("depth", "splatpipe.depth", False)`, so the order becomes frames, cameras, depth, train, clean. When it fails, or when no frame fits, the Job goes on and the Train_Step trains without priors, exactly as the splat-training spec says. The Depth_Step uses the Job_Folder, the Step_Runner, the Step_Report and the Config of the pipeline-core spec, and it's built like the Fake_Step and the Train_Step. On the cloud side it follows the cloud-jobs spec: its Step_Needs go in `[depth]` for the Start_Command, and the Smoke_Test gains a check for it.

MoGe, torch and the packages MoGe imports exist only on the Machine_Image. The Machine_Image already keeps MoGe-2's weights at `/opt/models/moge-2-vitb-normal/model.pt`, but not MoGe's code, so this spec adds MoGe's code and the packages its MoGe-2 path imports to the image's install script. Only the Depth_GPU_Module imports MoGe, inside a function, and only the two GPU modules import torch. Everything else is CPU code that the Test_Suite tests on a laptop with tiny made-up data: the fit, the prior files, the plain-pixel mask, the depth weight, the back-projection and the settings. The Smoke_Test's new check runs every new GPU path before any Job does.

The camera solve spec isn't on main yet, so this spec takes the hand-off below as the contract, the same one the splat-training spec reads, and the camera solve spec writes exactly this:

- `work/cameras/images/<name>.jpg`: the undistorted colour frames.
- `work/cameras/masks/<name>.png`: an image's mask, when it has one. Black means ignore.
- `work/cameras/sparse/`: COLMAP's binary model, with `cameras.bin` (one shared PINHOLE camera), `images.bin` and `points3D.bin`. The camera solve spec on `origin/feat/camera-solve` writes `images.bin` with each image's 2D points and their point ids, and `points3D.bin` with each point's id and track, as COLMAP 4.2.1 writes them.

The Depth_Step and the Train_Step both read these files with the Train_Step's own Model_Reader, which this spec extends to keep the points each image observes.

Out of scope:

- Building the Machine_Image and running the Smoke_Test. Both are paid runs that the user starts. No task of this spec is a paid run.
- The Camera_Solve_Step and its Step_List entry.
- Rescaling the scene to metres. The Depth_Step only reports its estimate of the scale.
- Any change to the Project_File or `uv.lock`.
- MoGe-1, MoGe-3, MoGe's command line, its demo app and its training code.
- Changing the Cloud_Runner, the Start_Command, the Fallback_Lists or `cloud/select.py`.
- Editing `.kiro/steering/rules.md` beyond its certifi line (D5).
- Tuning the new numbers on real footage.

### Task budget

The whole spec stays within 10 tasks, which the design and tasks phases plan. Each task writes its own unit tests and property tests, both required, never optional.

### Decisions made before this draft

The user chose these when asked: D1 to D4 before the requirements were written, and D5 and D4's change for hf-xet after the requirements were reviewed.

- **D1 (Requirements 4, 10 and 11): Prior size and training size.** The Depth_Step stores each Prior at MoGe's size, with its long side at most 512 pixels. The Train_Step keeps the Priors on the GPU at that size, about 1 GB for 400 frames. When it trains on a frame, it scales that frame's Prior up to the Training_Size: bilinear for the inverse depth and the normals, nearest-neighbour for MoGe's geometry mask. The depth loss, the normal losses and the plain-pixel test all work per training pixel.
- **D2 (Requirement 11): The 0.1 floor.** The floor applies to the final per-pixel Depth_Weight: the Depth_Schedule (1.0 down to 0.01, exponentially over the run) × exp(−10 × the Brightness_Gradient). On a Plain_Pixel the weight is raised to 0.1 whenever it falls below that. Textured pixels can go lower. The property test checks this final per-pixel weight.
- **D3 (Requirements 1 and 7): No frame fits.** When no Frame fits, the Depth_Step fails with an Expected_Failure and a Readable_Error that gives the reason. The Job ends `done with warnings`, the Job_Summary names the Depth_Step, and the Train_Step finds no Priors and trains without them.
- **D4 (Requirements 15 and 17): The packages on the image.** MoGe and utils3d_moge go on the Machine_Image from their pinned commits with `--no-deps`, so moderngl, trimesh and gradio stay out. huggingface-hub 1.26.0 goes on with click, filelock, fsspec, httpx, packaging, pyyaml, tqdm, typing-extensions, anyio, certifi, httpcore, idna and h11, each pinned at the version in MoGe's `uv.lock` at the pinned commit, each installed with `--no-deps` and each with a Licence_List row. hf-xet, which huggingface-hub declares, is left off the image, with no Licence_List row and no question about the Rust crates its wheel bundles. The reason: huggingface_hub 1.26.0 imports hf_xet only inside `xet_get` and falls back to plain HTTP without it, and the Depth_Step never downloads anything, because `HF_HUB_OFFLINE=1` is set and the weights are a local file. huggingface_hub's source at tag v1.26.0 bears this out: every `hf_xet` import in the package is inside a function, and the download code checks for hf_xet through its installed metadata, not by importing it. The Image_Install's build-time import check of MoGe with `HF_HUB_OFFLINE=1` is what proves that nothing on the MoGe-2 path is missing. Rows already exist for filelock, fsspec, packaging, typing-extensions, certifi and tqdm. The certifi row says it "comes in through requests", but on the image it also comes in through httpx and httpcore, so that row changes (D5).
- **D5 (Requirement 15): rules.md's certifi line.** Samarth OKs changing rules.md's named-exceptions line for certifi to say that on the Machine_Image certifi also comes in through httpx and httpcore, used unmodified. rules.md is a steering file, so the task that changes the Licence_List's certifi row changes only that one line of rules.md, and the line records Samarth's OK.

### Assumptions to confirm at review

The user didn't decide these. Each is the recommended answer, marked in the requirement it belongs to, and each is a small change if the review decides otherwise.

- **A1 (Requirement 2): A frame's sparse points.** A Frame's Observed_Points are the Sparse_Points it observes: the keypoints in `images.bin` whose point id is the id of a point in `points3D.bin`. The fit's Target is 1 / that point's camera-space z in the Frame's Image_Pose. So the Model_Reader keeps each image's keypoint point ids and each point's id. Today it reads past the keypoints (`cursor.skip(keypoints, ...)` in `splatpipe/train/colmap.py`), so this changes the splat-training spec's Requirement 1, criterion 5, "keeping none of those values". A keypoint whose point id is −1, or names no point, counts as no observation, and the Model_Reader refuses no model for it, so the Train_Step refuses exactly the models it refused before.
- **A2 (Requirement 5): The fit's points.** A Fit_Point is an Observed_Point in front of the Frame's camera, z above 0, whose projection lands inside the Frame, on a Prior_Size pixel where MoGe found geometry. Its MoGe_Value is 1 / MoGe's depth at that pixel. The 10-point minimum applies before the first fit and again to the points left for the second fit. The 15% check is the median Relative_Depth_Error, with the second fit, over every Fit_Point of the Frame, the dropped ones included. A Frame whose MoGe Spread is 0 is left out. The scale, a ratio of Spreads, is never negative, and a Frame whose second-fit scale is 0 is left out as flat too. A pixel where scale × MoGe_Value + offset isn't above 0 gets no Prior.
- **A3 (Requirement 6): Metres per unit.** The estimate is reported only and never applied. Each Fitted_Frame's value is the median, over the Fit_Points of its second fit, of MoGe's metric depth / COLMAP's z, and the estimate is the median of those values over the Fitted_Frames. It goes in the Prior_Index, because Step_Report counts must be whole numbers (pipeline-core, Requirement 6).
- **A4 (Requirements 11 and 13): Schedules.** The Depth_Schedule runs over the Planned_Iterations, like the positions' learning rate (splat-training, Requirement 7, criterion 10). The Normal_Term starts at an absolute Iteration, `normal_start_step = 3000`, set in Config, and is on at that Iteration and every `normal_every` (4) Iterations after it. The Depth_Normal_Term and the Consistency_Term start at round(7/30 × the Planned_Iterations), rounded as the splat-training spec rounds the Erank_Start: Iteration 1,633 for `fast`, 3,500 for `default` and 7,000 for `quality`.
- **A5 (Requirement 16): The smoke check's numbers.** The Depth_Smoke_Check gets its own step number in `[cloud.smoke]`, `depth_normal_start_step = 8`, so every Prior_Term runs within the training check's 30 Iterations. It reuses the training check's other `[cloud.smoke]` numbers.
- **A6 (Requirement 13): The flatten term.** The Flatten_Term is on only when there is a Prior_Set, on every Iteration. With no Prior_Set, the Loss_Table holds exactly the five splat-training terms.
- **A7 (Requirement 12): Back-projected splats.** The Back_Projected_Splats are drawn at random, with a generator seeded with `seed`, from the Candidate_Pixels of every Prior_View: training pixels that are Plain_Pixels and Prior_Pixels, and that no projected Observed_Point of that Frame lands on. Each Observed_Point rules out only the one training pixel it lands on. There are at most min(floor(`backproject_share` × `max_splats`), `backproject_max`) of them: 333,333 for `fast` and 500,000 for `default` and `quality`. The Sparse_Points fill the rest of the Splat_Budget. Each Back_Projected_Splat takes its pixel's colour, a depth from the scaled-up Prior, and a size from the same 3-nearest-neighbour rule as the other Start_Splats.
- **A8 (Requirement 13): Rendered depth and normals.** The Rendered_Depth is gsplat's expected depth, from the same render as the colours in the `"RGB+ED"` mode. A Splat_Normal is the splat's shortest axis, turned to face the camera. The Normal_Render is a second render of the same splats with their Splat_Normals as colours, made only on the Iterations where the Normal_Term or the Consistency_Term is on. The Depth_Normal comes from the central differences of the Rendered_Depth's back-projected points. The two terms on the Depth_Normal, like the Depth_Term and the Normal_Term, count only Prior_Pixels.
- **A9 (Requirements 7 and 10): Broken priors.** When `work/depth/` holds Priors that don't pass the Train_Step's checks, the Train_Step adds a warning and trains without priors, rather than failing a Required_Step over an optional input. It's all or nothing: one bad Prior_File drops the whole Prior_Set. The Depth_Step writes the Prior_Index last, so a half-written run is never read. The Prior_Index records the Shared_Camera and each Prior's Image_Pose, so Priors made for an earlier camera solve are found out and dropped.
- **A10 (Requirements 3 and 8): The weights path and the needs.** `[depth]` holds the Weights_File's path, `/opt/models/moge-2-vitb-normal/model.pt`, because nothing in `splatpipe/` may import `cloud/` (python.md). A test checks that it equals `MOGE_WEIGHTS` in `cloud/image.py`. `min_gpu_memory_gb = 16` is a cloud-only Step_Need: the Start_Command takes the largest need over every Step (`cloud/select.py`), and the Train_Step's 24 GB already covers 16. The Depth_Step checks that torch sees CUDA, but not the GPU's memory. Every `[depth]` key is a Shared_Setting, so the three Preset tables of `[depth]` are there, as the Config_Loader requires, and empty.
- **A11 (Requirement 1): The Step_List on this branch.** The Step_List becomes frames, depth, train, clean, with `StepEntry("depth", "splatpipe.depth", False)` in `splatpipe/run.py`. The camera solve spec adds its entry between the Frames_Step and the Depth_Step when it lands. Until then, a Job without a Camera_Solve_Output fails the Depth_Step, goes on, and fails at the Train_Step, as the splat-training spec's Requirement 11, criterion 5 says.
- **A12 (Requirement 11): Brightness and its gradient.** A pixel's Brightness is its grey value from 0 to 1, as OpenCV's RGB-to-grey conversion gives it (0.299 R + 0.587 G + 0.114 B) / 255, on the Training_Image at the Training_Size. Its Brightness_Gradient is the length of the gradient from central differences, (I(x+1) − I(x−1)) / 2 across and the same down, with the border pixels repeated outward.
- **A13 (Requirements 7 and 10): The Prior_Format and how training samples it.** One Prior_File per Fitted_Frame, `work/depth/<stem>.npz`, in numpy's uncompressed `.npz` format, read with `allow_pickle=False`, plus the Prior_Index, `work/depth/depth.json`. The normals stay in the camera frame, as MoGe gives them. The Train_Step derives each Prior_File's path from an image name it has checked against `images.bin`, never from a path in the index. Scaling up follows D1, and on top of D1, a training pixel whose bilinear sample draws on a stored pixel without a Prior gets no Prior, so the 0s outside MoGe's mask never blend into the depth at the mask's edge.
- **A14 (Requirement 14): A run without priors.** With no Prior_Set, the Train_Step trains, writes and reports exactly as the splat-training spec says, with one exception: the new `[train]` keys appear in every Train_Report's `settings`, because the Step_Runner puts every setting of `[train]` there. The new counts appear only when the Train_Step trains with a Prior_Set, and a Job with no `work/depth/depth.json` gets no new warning.
- **A15 (Requirements 2 and 4): What the Depth_Step reads and how it calls MoGe.** The Depth_Step reads no Mask_File, because the Train_Step leaves Masked_Pixels out of every Prior_Term itself. It doesn't read `SPLATPIPE_CAP_DEADLINE`, because MoGe's README gives about 60 ms per image for its larger ViT-L model, so 400 Frames should take minutes, far less than a Job's time cap. It decodes each Frame as the Train_Step does and scales it down with `INTER_AREA`. It calls `infer()` with MoGe's defaults, `resolution_level = 9` and `use_fp16 = true`, held in `[depth]`. It doesn't check the Weights_File's sha256, which the Image_Build and the Smoke_Test already check.
- **A16 (Requirement 3): GPU memory and offline mode.** The Depth_Step and the Train_Step run in one process in a Job, so the Depth_Step releases the MoGe_Model and empties torch's memory cache when it ends, done or failed, and the Train_Step gets the GPU's memory. After it imports MoGe, the Depth_GPU_Module checks that huggingface_hub really is offline, which fails only when something imported huggingface_hub before the variable was set.
- **A17 (Requirement 18): Two more property tests.** Besides the property tests the user asked for, the Prior_Format gets a round-trip property test and the back-projection a property test that projecting a Back_Projected_Splat's centre gives back its pixel and its depth, as testing.md asks for every rule that should hold for every input.
- **A18 (Requirement 14): The splat-training spec's criteria.** Requirement 14 lists the splat-training criteria this spec changes. The splat-training requirements file stays as it is, as that spec's A14 did for the cloud-jobs and frames specs, and the Pipeline_Doc says what's true now.
- **A19 (Requirement 16): Until the image is rebuilt.** The Machine_Image that runs Jobs now has no MoGe code. On it, the Depth_Step fails at its import check and the Train_Step trains without priors, so Jobs keep working, but each Job that trains ends `done with warnings` instead of `done`, with the Depth_Step's entry `failed`, as Requirement 1, criterion 6 says for any failed Depth_Step. A Smoke_Test on that image fails the Depth_Smoke_Check. The next Image_Build makes a new image id, and the Start_Command's gate then needs a passing Smoke_Test on that image before any Job runs on it.

### Open points to confirm

- **O2 (Requirement 11): The plain-pixel cutoff.** The request puts the cutoff in Config but gives no number. This draft writes `plain_gradient = 0.02` on A12's gradient, a change of about 5 grey levels per pixel, meant to sit above JPEG noise and below most texture. It's untested on real footage. The review sets the number.
- **O3 (Requirement 15): The exact pins.** MoGe's `uv.lock` at commit `74fbce05` pins huggingface-hub 1.26.0, httpx 0.28.1, httpcore 1.0.9, anyio 4.14.2 and idna 3.18. On Python 3.11 anyio needs only idna and typing-extensions, and httpcore needs certifi and h11, so D4's list is the whole tree but for hf-xet, which D4 leaves off. The design reads the pins of click, filelock, fsspec, packaging, pyyaml, tqdm, typing-extensions, certifi and h11 from the same file. torch 2.7.1 already puts filelock, fsspec and typing-extensions on the image, so the design checks that MoGe's pins still meet torch's and gsplat's requirements. If a pin would break them, or the lock names a package outside D4's list other than hf-xet, that goes back to the user before the task adds it.
- **O4 (Requirement 13): The cost of the consistency term.** The Consistency_Term is on every Iteration from the Depth_Normal_Start on Prior_Views, so from there on those Iterations render twice, once for the colours and depth and once for the Splat_Normals. That makes those Iterations slower. The request asks for it this way, so the draft keeps it.
- **O5 (Requirement 12): One pixel per point.** "Where no sparse point lands" is read literally: each Observed_Point rules out the one training pixel it lands on. At 1,600 pixels wide and a few thousand points per frame, almost every Plain_Pixel stays a Candidate_Pixel. If the intent is to keep the new splats away from areas that COLMAP already covers, the review can widen the rule, for example to a block of pixels around each point.

## Glossary

- **Depth_Step**: the Step this spec builds, in `splatpipe/depth/`. Its Step_Name is `depth`, so its Config section is `[depth]`, its Work_Folder is `work/depth/` and its Step_Report is `out/reports/depth.json`.
- **Depth_GPU_Module**: `splatpipe/depth/gpu.py`, the one module of the Depth_Step that imports torch, MoGe and huggingface_hub.
- **Depth_Stand_In**: what the Test_Suite puts in place of the Depth_GPU_Module, so the Depth_Step runs on a CPU in the tests.
- **Depth_Report**: the Depth_Step's Step_Report, with the fields of Requirement 9.
- **Weights_File**: the MoGe-2 ViT-B normal checkpoint on the Machine_Image, at the path `moge_weights` in `[depth]`: `/opt/models/moge-2-vitb-normal/model.pt`, the `MOGE_WEIGHTS` of `cloud/image.py`.
- **MoGe_Model**: `moge.model.v2.MoGeModel` from MoGe's code at commit `74fbce054ebed49800de42d0ad0e83495065719a`, built and loaded from the Weights_File.
- **MoGe_Output**: what `MoGeModel.infer()` gives for one Frame, at the Prior_Size: `depth`, camera-space z in metres; `normal`, unit normals in the camera frame that face the camera; and `mask`, true where MoGe found geometry.
- **Frame**: one image that `images.bin` names, with its colour JPEG `work/cameras/images/<name>.jpg`, the file the splat-training spec calls a Training_Image.
- **Prior_Size**: the width and height of every Prior: the Shared_Camera's size scaled down by the Train_Step's Training_Size rule to `[depth]`'s `long_edge` (512), or the camera's own size when its longer edge is 512 or less.
- **Field_Of_View**: the horizontal field of view the Depth_Step gives MoGe, in degrees: degrees(2 × atan(W / (2 × fx))), from the Shared_Camera's width W and fx.
- **Geometry_Pixel**: a Prior_Size pixel where MoGe's `mask` is true and MoGe's `depth` is finite and above 0.
- **Observed_Points**: one Frame's distinct Sparse_Points whose ids are the point id of one of the Frame's keypoints in `images.bin` (A1).
- **Fit_Point**: an Observed_Point of a Frame that is in front of the Frame's camera (camera-space z above 0), whose projection with the Shared_Camera lands inside the Frame, and whose Prior_Size pixel under that projection is a Geometry_Pixel (A2).
- **Target**: a Fit_Point's 1 / z, its inverse depth in the camera solve's units.
- **MoGe_Value**: a Fit_Point's 1 / MoGe's `depth` at the Prior_Size pixel under its projection, in 1 / metres.
- **Spread**: the mean absolute deviation of some values around their median, mean(|x − median(x)|), with numpy's median: the middle value, or the mean of the two middle values.
- **Depth_Fit**: the two-pass fit of Requirement 5, which gives one Frame's scale and offset so that scale × MoGe_Value + offset matches the Target.
- **Miss**: one Fit_Point's |scale × MoGe_Value + offset − Target| with the first fit's scale and offset.
- **Relative_Depth_Error**: one Fit_Point's |1 / (scale × MoGe_Value + offset) − z| / z with the second fit's scale and offset, or infinity when scale × MoGe_Value + offset isn't above 0.
- **Fitted_Frame**: a Frame whose Depth_Fit passes every check of Requirement 5.
- **Fitted_Inverse_Depth**: scale × (1 / MoGe's `depth`) + offset at a Geometry_Pixel of a Fitted_Frame, with that Frame's second-fit scale and offset.
- **Metres_Per_Unit**: the estimate of how many metres one unit of the camera solve is (Requirement 6, A3).
- **Prior**: one Fitted_Frame's depth prior at the Prior_Size: its Fitted_Inverse_Depth, its normals and a mask of the pixels that have a prior.
- **Prior_File**: `work/depth/<stem>.npz`, one Prior, where `<stem>` is the Frame's name without `.jpg`.
- **Prior_Index**: `work/depth/depth.json`, the list of the Priors and what they were made from, which the Depth_Step writes last.
- **Prior_Format**: the layout of the Prior_Index and the Prior_Files (Requirement 7), the one contract between whatever makes Priors and the Train_Step.
- **Prior_Writer** and **Prior_Reader**: our numpy functions that write and read the Prior_Format.
- **Prior_Set**: the Priors the Train_Step has read and checked for one run, or none.
- **Prior_View**: a Training_Image that has a Prior in the Prior_Set.
- **Prior_Pixel**: a training pixel of a Prior_View that isn't a Masked_Pixel and that has a Prior once the Prior is scaled up to the Training_Size (Requirement 10, criterion 7).
- **Brightness** and **Brightness_Gradient**: a training pixel's grey value and the length of its gradient (A12).
- **Plain_Pixel**: a training pixel whose Brightness_Gradient is below `plain_gradient` in `[train]`.
- **Depth_Schedule**: the Iteration's share of the depth loss, from `depth_weight_start` (1.0) down to `depth_weight_end` (0.01), exponentially over the Planned_Iterations.
- **Depth_Weight**: one training pixel's weight in the Depth_Term at one Iteration (Requirement 11, criterion 5).
- **Loss_Table**: the Trainer's table of named Loss_Terms, `_TERMS` in `splatpipe/train/gpu.py`, each with its weight.
- **View_Record**: the Trainer's one record per Training_Image that holds the image's inputs to the loss (splat-training, Requirement 5, criterion 10).
- **Rendered_Depth**: the expected depth that gsplat 1.5.3's `rasterization()` gives in the `"RGB+ED"` render mode: each splat's blending weight times its camera-space z, summed, then divided by the pixel's alpha (A8).
- **Splat_Normal**: a splat's shortest axis, the axis of its smallest scale, as a unit vector in the camera frame, with its sign chosen so it faces the camera (A8).
- **Normal_Render**: a render of the splats with their Splat_Normals as colours, scaled to unit length at each pixel (A8).
- **Depth_Normal**: the unit normal at a training pixel of the surface the Rendered_Depth makes (A8).
- **Prior_Terms**: the five Loss_Terms this spec adds: `depth` (the Depth_Term), `normal` (the Normal_Term), `flatten` (the Flatten_Term), `depth_normal` (the Depth_Normal_Term) and `normal_consistency` (the Consistency_Term).
- **Normal_Start**: the first Iteration of the Normal_Term, `normal_start_step` (3000).
- **Depth_Normal_Start**: the first Iteration of the Depth_Normal_Term and the Consistency_Term, round(`depth_normal_start_fraction` × Planned_Iterations).
- **Candidate_Pixel**: a training pixel a Back_Projected_Splat can start from (A7).
- **Back_Projected_Splat**: a Start_Splat placed at a Candidate_Pixel, at the depth its Prior gives (Requirement 12).
- **Back_Project_Budget**: the most Back_Projected_Splats one run starts from, min(floor(`backproject_share` × `max_splats`), `backproject_max`).
- **Depth_Smoke_Check**: the check named `depth` that this spec adds to SMOKE_CHECKS.
- **Prior_Smoke_Scene**: the tiny made-up scene with made-up Priors that the Depth_Smoke_Check trains on.
- **MoGe_Packages**: what the Image_Install adds for MoGe: MoGe's code and utils3d_moge at their pinned commits, and huggingface-hub 1.26.0 with the other packages D4 lists, each with `--no-deps`. hf-xet isn't one of them (D4).
- **Image_Install**: `install_script` in `cloud/image.py`, the Image_Build's whole install.
- **Fixed_Constants**: what stays in code, not in Config, because it describes a format, a library or a contract and not a choice we tune: the Prior_Format's file names (`depth.json` and `<stem>.npz`), its array names, its `format` number (1) and its field names; MoGe's class `moge.model.v2.MoGeModel` and the checkpoint's keys `model_config` and `model`; the variable name `HF_HUB_OFFLINE` and its value `1`; the pinned commits and versions of the MoGe_Packages in `cloud/image.py`; OpenCV's grey weights and the central-difference stencil of A12; and the Prior_Smoke_Scene's shape in its own unit. This is the same exception the pipeline-core spec's Requirement 1, criterion 6 makes for the Job_Id pattern.

Terms this spec uses as the pipeline-core spec and the Pipeline_Doc define them: Skeleton, Step, Step_Name, Job, Job_Id, Jobs_Folder, Job_Folder, Step_Runner, Step_Report, Step_List, Required_Step, Optional_Step, Finished_Step, Fake_Step, Work_Folder, Config, Config_Loader, Preset, Shared_Setting, Preset_Setting, Step_Section, Expected_Failure, Unexpected_Error, Readable_Error, Technical_Detail, Job_Runner, Job_Summary, Job_Status, JSON_Value, Pipeline_Doc, Licence_List, CLI and Test_Suite.

Terms it uses as the splat-training spec defines them: Train_Step, Camera_Solve_Step, Clean_Step, Camera_Solve_Output, Training_Image, Mask_File, Masked_Pixel, Sparse_Model, Model_Reader, Shared_Camera, Image_Pose, Camera_Centre, Sparse_Points, Training_Size, Scene_Extent, Splat_Budget, Start_Splats, Start_Distance, Iteration, Planned_Iterations, Schedules, Means_LR, Erank_Start, Loss_Term, Mean_Exposure, GPU_Module, Trainer, GPU_Stand_In, Train_Report, Train_Record, PLY_Writer, PLY_Check, SMOKE_CHECKS, Train_Smoke_Check and Smoke_Scene.

Terms it uses as the cloud-jobs spec defines them: Cloud_Runner, User_Data, Start_Command, Step_Needs, Fallback_List, Smoke_Test, Smoke_Command, Smoke_Marker, Image_Build, Cloud_Config and Project_File. From the aws-setup spec it uses Machine_Image, and from the frames spec, Frames_Step.

## Requirements

### Requirement 1: The depth step and the step list

**User Story:** As someone who films a place, I want depth priors made between the camera solve and training, and the job to go on without them when they can't be made, so that plain surfaces train better and a failed prior never costs me my splat.

**Decided:** D3 (criterion 6). **Assumptions (to confirm at review):** A11 (criteria 3 and 10).

#### Acceptance Criteria

1. THE Depth_Step SHALL be a package `splatpipe/depth/` built like the Fake_Step: `work(step)`, `run(job_id, preset=None, *, config=None)` and a `__main__.py` whose `main(argv)` calls the Skeleton's `step_main`.
2. THE CLI `uv run python -m splatpipe.depth --job <id> [--preset P]` SHALL run the Depth_Step on one Job, print `depth: done` on standard output and exit with 0 when the step is done, and print the Readable_Error on standard error with no traceback and exit with 1 when the step failed.
3. THE Step_List in `splatpipe/run.py` SHALL be `(StepEntry("frames", "splatpipe.frames", True), StepEntry("depth", "splatpipe.depth", False), StepEntry("train", "splatpipe.train", True), StepEntry("clean", "splatpipe.clean", False))`, with the Step_List's comment saying that the camera solve spec puts its entry between the Frames_Step and the Depth_Step, so the order becomes frames, cameras, depth, train, clean (A11).
4. THE Depth_Step SHALL do its work in this order: check the settings; check that the Camera_Solve_Output's files are there; import the Depth_GPU_Module and check the GPU; check that the Weights_File is there; import MoGe and load the MoGe_Model; read the Sparse_Model; for each Frame in `images.bin`'s order, decode the Frame, run MoGe, fit the Frame and write its Prior_File when the Frame is a Fitted_Frame; release the GPU's memory; and write the Prior_Index last.
5. IF the Depth_Step fails, THEN THE Job_Runner SHALL write the Depth_Step's Readable_Error to the Job_Log and go on with the next Step, as for every Optional_Step.
6. WHEN the Depth_Step failed and every Required_Step is done, THE Job_Runner SHALL give the Job the Job_Status `done with warnings`, with the Job_Summary's entry for the Depth_Step holding the outcome `failed` and the Depth_Step's Readable_Error (D3).
7. THE Depth_Step SHALL raise `ExpectedFailure` for every failure that Requirements 2 to 9 name, except the Unexpected_Error of Requirement 3, criterion 8, so the Depth_Report's `error.expected` is `true` and its `readable` is the step's own message.
8. THE Depth_Step SHALL reach the Depth_GPU_Module through one module-level hook that the Test_Suite can replace with a Depth_Stand_In, as the Train_Step reaches its GPU_Module, so the step's own code runs on a CPU in the tests.
9. THE Depth_Step SHALL write only inside `work/depth/`, and SHALL write nothing to `out/` other than the Depth_Report, which the Step_Runner writes.
10. WHILE the Step_List holds no Camera_Solve_Step entry, THE Job_Runner SHALL run a Job that has no Camera_Solve_Output through the Frames_Step, fail the Depth_Step with the Readable_Error of Requirement 2, criterion 2, go on, and fail the Job at the Train_Step, as the splat-training spec's Requirement 11, criterion 5 says (A11).
11. WHEN the Depth_Step runs again on a Job, THE Depth_Step SHALL rely on the Step_Runner emptying `work/depth/` first, so no Prior_File or Prior_Index from an earlier run is left.

### Requirement 2: Reading the camera solve's files

**User Story:** As a team member building the camera solve, I want the depth step to read the same files the train step reads, with the same reader, so that one hand-off serves both steps.

**Assumptions (to confirm at review):** A1 (criteria 4 to 6 and 9) and A15 (criteria 1 and 7).

#### Acceptance Criteria

1. THE Depth_Step SHALL read the Camera_Solve_Output only from `work/cameras/` in the Job_Folder: the Frames in `images/` and the Sparse_Model in `sparse/`, and SHALL read no Mask_File (A15).
2. IF `work/cameras/images/` or any of the Sparse_Model's three files is missing, THEN THE Depth_Step SHALL fail with an Expected_Failure, before the GPU checks of Requirement 3, whose Readable_Error names the missing path and says that the camera solve step writes that path, so the camera solve step has to run first.
3. THE Depth_Step SHALL read the Sparse_Model with the Train_Step's Model_Reader, and IF the Model_Reader refuses the Sparse_Model, THEN THE Depth_Step SHALL fail with the Model_Reader's Expected_Failure.
4. THE Model_Reader SHALL also give each point's id from `points3D.bin` and each image's Observed_Points: the distinct point ids among the image's keypoints that are ids of points in `points3D.bin`, in the order of their first keypoint (A1).
5. THE Model_Reader SHALL count a keypoint whose point id is −1, or isn't the id of a point in `points3D.bin`, as no observation, and SHALL refuse no Sparse_Model because of a keypoint's point id, so the Train_Step refuses exactly the Sparse_Models that the splat-training spec's Requirement 1 refuses (A1).
6. THE Model_Reader SHALL keep reading past each keypoint's x and y and each point's error and track, keeping none of those values (A1).
7. THE Depth_Step SHALL decode each Frame as the Train_Step decodes a Training_Image: read as stored with `cv2.imdecode`, ignoring any EXIF orientation, as a colour JPEG at the Shared_Camera's size, turned from BGR to RGB, and SHALL fail with the Train_Step's Expected_Failures for a Frame that has no file in `images/`, can't be decoded or isn't the Shared_Camera's size (A15).
8. THE Depth_Step SHALL go through every image in `images.bin`, in that file's order, and SHALL ignore every file in `images/` that no image in `images.bin` names.
9. FOR ALL Sparse_Models of one PINHOLE camera with any images, any keypoints with any point ids and any points that the Test_Suite's own writer writes, THE Model_Reader SHALL read back the same camera, Image_Poses, camera ids, names, point positions, colours and point ids, bit for bit, and each image's Observed_Points as criterion 4 gives them (property). This widens the splat-training spec's Requirement 1, criterion 21.

### Requirement 3: The GPU and the MoGe model

**User Story:** As a team member, I want the depth step to load MoGe-2 only from the image's own weights, with no network and no silent fallback, so that a job never downloads anything and a wrong file fails instead of running with random weights.

**Assumptions (to confirm at review):** A10 (criterion 4) and A16 (criteria 8 and 14).

#### Acceptance Criteria

1. WHEN the Camera_Solve_Output's files are there, THE Depth_Step SHALL import the Depth_GPU_Module, run the checks of criteria 2, 3 and 5, and load the MoGe_Model before the Depth_Step reads the Sparse_Model or any Frame.
2. IF torch, MoGe's code or huggingface_hub can't be imported, THEN THE Depth_Step SHALL fail with an Expected_Failure whose Readable_Error names the missing package, says that the step runs only on a GPU machine image built with MoGe's code, and says to rebuild the machine image or to run the job there through `uv run python -m cloud.start`.
3. IF torch sees no CUDA GPU, THEN THE Depth_Step SHALL fail with an Expected_Failure whose Readable_Error says that the step needs an NVIDIA GPU and that torch can't see one on this machine.
4. THE Depth_Step SHALL check no amount of GPU memory, so `min_gpu_memory_gb` in `[depth]` stays a cloud-only Step_Need (A10).
5. IF the Weights_File isn't a file, THEN THE Depth_Step SHALL fail with an Expected_Failure, before anything imports MoGe or loads the Weights_File, whose Readable_Error gives the path and says to rebuild the machine image.
6. THE Depth_GPU_Module SHALL set the environment variable `HF_HUB_OFFLINE` to `1` before it imports MoGe or huggingface_hub, and SHALL import MoGe and huggingface_hub only inside a function, never at the module's top.
7. WHEN the Depth_GPU_Module has imported MoGe, THE Depth_GPU_Module SHALL check that huggingface_hub reports its offline mode as on.
8. IF huggingface_hub reports its offline mode as off, THEN THE Depth_GPU_Module SHALL stop with an error that the Step_Runner counts as an Unexpected_Error, saying that huggingface_hub was imported before `HF_HUB_OFFLINE` was set (A16).
9. THE Depth_GPU_Module SHALL load the Weights_File with `torch.load(path, map_location="cpu", weights_only=True)`, build the MoGe_Model as `MoGeModel(**checkpoint["model_config"])`, and load `checkpoint["model"]` with a strict `load_state_dict`, so a missing or unexpected key is an error.
10. THE Depth_GPU_Module SHALL call no `from_pretrained()`, `hf_hub_download()` or `init_weights()`, so no Job downloads anything.
11. IF the checkpoint isn't a dict that holds `model_config` and `model`, or the strict load fails, THEN THE Depth_Step SHALL fail with an Expected_Failure whose Readable_Error says that the file at the Weights_File's path isn't the MoGe-2 ViT-B normal checkpoint and says to rebuild the machine image.
12. THE Depth_GPU_Module SHALL move the MoGe_Model to the GPU, put it in eval mode and keep it in float32, as MoGe's README example does.
13. IF torch runs out of GPU memory while the Depth_Step loads the MoGe_Model or runs MoGe, THEN THE Depth_Step SHALL fail with an Expected_Failure whose Readable_Error says that the GPU ran out of memory, gives the Frame reached, and says that the job trains without depth priors.
14. WHEN the Depth_Step ends after the MoGe_Model has started to load, done or failed, THE Depth_Step SHALL release the MoGe_Model and empty torch's GPU memory cache, so the Train_Step that runs next in the same process gets the GPU's memory (A16).
15. THE Depth_Step SHALL import the Depth_GPU_Module only inside the step's own code, when the step runs, so importing `splatpipe.depth` on a laptop imports no torch, MoGe or huggingface_hub.

### Requirement 4: Running MoGe on each frame

**User Story:** As someone who films a place, I want MoGe-2 run on every frame with the real camera's field of view, so that each depth map has the right shape for the camera that took it.

**Decided:** D1 (criteria 1 and 2). **Assumptions (to confirm at review):** A15 (criteria 2 and 4).

#### Acceptance Criteria

1. THE Depth_Step SHALL set the Prior_Size with the Train_Step's Training_Size rule (splat-training, Requirement 3, criterion 1), applied to the Shared_Camera's width and height with `long_edge` from `[depth]` (512), so no Frame is scaled up (D1).
2. THE Depth_Step SHALL scale each Frame to the Prior_Size with OpenCV's `INTER_AREA` when the Frame is larger, and SHALL give MoGe the Frame's RGB colours as floats from 0 to 1, each 8-bit value / 255 (D1, A15).
3. THE Depth_Step SHALL give MoGe the Field_Of_View as `fov_x`, worked out with a numpy or standard-library function that imports no torch.
4. THE Depth_Step SHALL call `infer()` with `resolution_level` and `use_fp16` from `[depth]` and with `force_projection` and `apply_mask` at MoGe's defaults (A15).
5. THE Depth_Step SHALL run MoGe on every Frame, one Frame at a time, in `images.bin`'s order.
6. THE Depth_GPU_Module SHALL give back each MoGe_Output as numpy arrays at the Prior_Size: `depth` (h, w) float32, `normal` (h, w, 3) float32 and `mask` (h, w) bool.
7. THE Depth_Step SHALL treat a Prior_Size pixel as having geometry only when the pixel is a Geometry_Pixel, so a pixel of sky, or any pixel where MoGe found no geometry, gets no Prior.

### Requirement 5: Fitting each frame to the camera solve

**User Story:** As someone who films a place, I want each depth map fitted to the camera solve's own points, and a frame used only when the fit is good, so that the priors agree with the cameras the splat is trained from.

**Assumptions (to confirm at review):** A2 (criteria 1, 3, 7, 8 and 11).

#### Acceptance Criteria

1. THE Depth_Step SHALL take each Frame's Fit_Points from the Frame's Observed_Points, with each Fit_Point's camera-space z from the Frame's Image_Pose, its projection from the Shared_Camera, and its Prior_Size pixel as the pixel under that projection scaled to the Prior_Size (A2).
2. IF a Frame has fewer than `fit_min_points` (10) Fit_Points, THEN THE Depth_Step SHALL leave the Frame out as a Frame with too few points.
3. IF the Spread of a Frame's MoGe_Values is 0, THEN THE Depth_Step SHALL leave the Frame out as a flat Frame (A2).
4. THE Depth_Fit SHALL set the scale to the Spread of the Targets divided by the Spread of the MoGe_Values, and the offset to the median of the Targets minus the scale times the median of the MoGe_Values.
5. THE Depth_Fit SHALL work out each Fit_Point's Miss with the first fit's scale and offset, then fit again, with criterion 4's formulas, on only the Fit_Points whose Miss is at most `fit_miss_ratio` (3) times the median Miss.
6. IF fewer than `fit_min_points` Fit_Points are left for the second fit, or the Spread of their MoGe_Values is 0, THEN THE Depth_Step SHALL leave the Frame out, as a Frame with too few points or as a flat Frame (A2).
7. IF the second fit's scale is 0, THEN THE Depth_Step SHALL leave the Frame out as a flat Frame (A2).
8. THE Depth_Fit SHALL work out the median Relative_Depth_Error with the second fit's scale and offset over every Fit_Point of the Frame, the Fit_Points the second fit left out included (A2).
9. IF the median Relative_Depth_Error is above `fit_max_median_error` (0.15), THEN THE Depth_Step SHALL leave the Frame out as a Frame whose fit is poor, and WHEN the median is at most `fit_max_median_error`, THE Depth_Step SHALL keep the Frame as a Fitted_Frame.
10. THE Depth_Fit and the Fit_Point selection SHALL be numpy functions that import no torch, so the Test_Suite tests them with no GPU.
11. THE Depth_Step SHALL give a Fitted_Frame's Prior_Size pixel a prior only where the pixel is a Geometry_Pixel and the Fitted_Inverse_Depth is above 0 (A2).
12. FOR ALL sets of at least 2 × `fit_min_points` distinct MoGe_Values above 0, all scales above 0 and all offsets that keep every Target above 0, where every Target is exactly the scale times its MoGe_Value plus the offset, THE Depth_Fit SHALL keep the Frame as a Fitted_Frame and give back the scale within a relative error of 10⁻⁹ and the offset within 10⁻⁹ × the larger of |offset| and the largest Target (property).
13. FOR ALL such sets in which at most a tenth of the Targets are moved far off their true values, THE Depth_Fit SHALL keep the Frame as a Fitted_Frame and give back the true scale and offset within the bounds of criterion 12, with the largest true Target in the offset's bound (property).

Note for the design: criterion 12 uses at least 2 × `fit_min_points` points because the points whose Miss is at most the median Miss are at least half of them, so the second fit always keeps `fit_min_points`. Its MoGe_Values are distinct because, with exact data, every Miss is rounding noise and rounding decides which points the second fit keeps: if most points shared one MoGe_Value, the kept points could all be that one, and the Frame would come out flat. Its offset bound is relative to the largest Target because an offset near 0 has no meaningful relative error, while its rounding is relative to the Targets it's subtracted from. For criterion 13, the first fit's Spreads take in the moved points, so the design gives the generator bounds that make the 3 × rule drop every moved point: MoGe_Values spread across a range rather than bunched around their median, and moved Targets off by a bounded factor of the true Targets' range. The design shows that these bounds guarantee the drop, so the property holds for every case the generator draws.

### Requirement 6: Metres per camera solve unit

**User Story:** As a team member, I want an estimate of how many metres one camera solve unit is, so that we know the scene's real size without the pipeline changing it.

**Assumptions (to confirm at review):** A3 (criteria 1 to 3).

#### Acceptance Criteria

1. THE Depth_Step SHALL work out each Fitted_Frame's Metres_Per_Unit as the median, over the Fit_Points of the Frame's second fit, of MoGe's `depth` at the Fit_Point's Prior_Size pixel divided by the Fit_Point's camera-space z (A3).
2. THE Depth_Step SHALL work out the job's Metres_Per_Unit as the median of the Fitted_Frames' Metres_Per_Unit (A3).
3. THE Prior_Index SHALL hold the job's Metres_Per_Unit and each Fitted_Frame's (A3).
4. THE Depth_Step SHALL change no camera, pose or point, and THE Train_Step SHALL keep training in the camera solve's units, so the Metres_Per_Unit is only a reported estimate.

### Requirement 7: Writing the priors

**User Story:** As a team member, I want the priors written in one documented format, so that the train step reads them the same way whatever made them.

**Decided:** D3 (criteria 7 and 8). **Assumptions (to confirm at review):** A9 (criterion 6) and A13 (criteria 1 to 5).

#### Acceptance Criteria

1. THE Prior_Writer SHALL write each Fitted_Frame's Prior to its Prior_File, `work/depth/<stem>.npz`, in numpy's uncompressed `.npz` format, holding exactly three arrays at the Prior_Size h × w (A13):
   - `inverse_depth`, (h, w) float32: the Fitted_Inverse_Depth, in 1 / camera solve units, at each pixel that has a prior, and 0 at every other pixel;
   - `normal`, (h, w, 3) float32: MoGe's unit normal in the camera frame, with x right, y down and z forward and facing the camera, at each pixel that has a prior, and 0 at every other pixel;
   - `mask`, (h, w) bool: true exactly at the pixels that have a prior.
2. THE Prior_Format SHALL place stored pixel (x, y) at the centre ((x + 0.5) × W / w, (y + 0.5) × H / h) in the Shared_Camera's pixels, where W × H is the Shared_Camera's size, as COLMAP and MoGe both put pixel centres at half-pixel offsets (A13).
3. THE Prior_Writer SHALL write the Prior_Index, `work/depth/depth.json`, holding only JSON_Values: `format` (1); `camera`, the Shared_Camera's `width`, `height`, `fx`, `fy`, `cx` and `cy`; `prior_width` and `prior_height`; `metres_per_unit`; and `priors`, one object per Fitted_Frame in `images.bin`'s order, with the Frame's `name`, its Image_Pose as `quaternion` (w, x, y, z) and `translation` exactly as `images.bin` gives them, and its `scale`, `offset`, `fit_points` (the first fit's points), `kept_points` (the second fit's points), `median_error` and `metres_per_unit` (A13).
4. THE Prior_Index SHALL name no file path, so a reader derives each Prior_File's path from a Frame's name (A13).
5. THE Prior_Reader SHALL read the Prior_Index with Python's `json` module and each Prior_File with `numpy.load(path, allow_pickle=False)`, so reading a Prior_File can run no code (A13).
6. WHEN every Prior_File is written, THE Depth_Step SHALL write the Prior_Index last, to a temporary file in `work/depth/` that one rename on the same file system turns into `work/depth/depth.json`, so a reader never finds a half-written Prior_Index (A9).
7. IF no Frame is a Fitted_Frame, THEN THE Depth_Step SHALL fail with an Expected_Failure, leaving no file in `work/depth/`, whose Readable_Error gives the number of Frames, how many had too few points, how many were flat and how many fitted poorly, and says that the job goes on and trains without depth priors (D3).
8. WHEN the Depth_Step fails for any reason, THE Depth_Step SHALL leave no Prior_Index in `work/depth/`, so the Train_Step trains without priors (D3).
9. WHERE the Depth_Step leaves out some Frames but keeps at least one Fitted_Frame, THE Depth_Step SHALL add a warning to the Depth_Report that gives the Fitted_Frames, the Frames, and how many Frames had too few points, were flat or fitted poorly.
10. IF writing a Prior_File or the Prior_Index fails because the disk is full or the file can't be written, THEN THE Depth_Step SHALL fail with an Expected_Failure whose Readable_Error names the file and the reason.
11. THE Prior_Writer and the Prior_Reader SHALL be one numpy module that imports no torch, used by both the Depth_Step and the Train_Step, so the two steps share one definition of the Prior_Format.
12. FOR ALL Prior_Sets of one or more Priors of any Prior_Size with any finite values, THE Prior_Reader SHALL read back from the Prior_Writer's files the same Prior_Index values and the same arrays, bit for bit (property, A17).

### Requirement 8: Settings, needs and the weights path

**User Story:** As a team member, I want every number of the depth step in Config, and its machine needs where the cloud code finds them, so that we can tune it without changing code.

**Assumptions (to confirm at review):** A10 (criteria 1, 4 and 5) and A15 (criterion 2).

#### Acceptance Criteria

1. THE Config SHALL hold a `[depth]` Step_Section with the three Preset tables `[depth.fast]`, `[depth.default]` and `[depth.quality]`, each empty, with a comment above each table saying that every `[depth]` setting is shared, and with a short comment for every key (A10).
2. THE `[depth]` section SHALL hold these Shared_Settings: `long_edge = 512`, `moge_weights = "/opt/models/moge-2-vitb-normal/model.pt"`, `resolution_level = 9`, `use_fp16 = true`, `fit_min_points = 10`, `fit_miss_ratio = 3`, `fit_max_median_error = 0.15` and `min_gpu_memory_gb = 16` (A15).
3. IF a `[depth]` setting is present but isn't the kind of value the setting must be, THEN THE Depth_Step SHALL fail with an Expected_Failure, before the Depth_Step checks the Camera_Solve_Output, whose Readable_Error names the setting, its table and what it must be: `long_edge`, `fit_min_points` and `min_gpu_memory_gb` whole numbers above 0; `resolution_level` a whole number from 0 to 9; `use_fp16` true or false; `moge_weights` a string of one or more characters; and `fit_miss_ratio` and `fit_max_median_error` numbers above 0. A bool is never a number here.
4. THE Test_Suite SHALL check that the real Config's `moge_weights` equals `MOGE_WEIGHTS` in `cloud/image.py` (A10).
5. THE `min_gpu_memory_gb` in `[depth]` SHALL be the Depth_Step's one Step_Need, which the Start_Command takes into the largest need over every Step, as `cloud/select.py` already does, with no change to the cloud code (A10).
6. IF a setting is missing from `[depth]`, THEN THE Step_Runner's existing check SHALL fail the Depth_Step with the Step_Runner's Readable_Error, which names the section, the table and the key.
7. THE Depth_Step SHALL take every size, count, ratio and path from Config and hold in code only the Fixed_Constants.
8. THE `tests/conftest.py` test config SHALL gain a `[depth]` section with its three Preset tables, so a test that runs the Depth_Step without a Config of its own gets settings, and a test SHALL be able to change any one of the settings for that test, as `set_frames` does for `[frames]`.

### Requirement 9: The depth step's report

**User Story:** As a team member, I want the depth step's report to say how many frames fitted and why the others didn't, so that I can tell a good prior run from a poor one.

#### Acceptance Criteria

1. WHEN the Depth_Step is done, THE Depth_Report SHALL hold the counts `images` (the Frames), `prior_width` and `prior_height` (the Prior_Size), `fitted_images` (the Fitted_Frames), `few_points_images`, `flat_images` and `poor_fit_images` (the Frames left out for each reason) and `peak_gpu_memory_bytes` (the most memory torch's allocator held at once from the GPU check on, from `torch.cuda.max_memory_allocated()`).
2. WHEN the Depth_Step is done, THE Depth_Report SHALL hold the timings `load_model` (building and loading the MoGe_Model), `infer` (decoding the Frames and running MoGe), `fit` (the Fit_Points and the Depth_Fits) and `write` (the Prior_Files and the Prior_Index), each in seconds, summed over the Frames.
3. WHEN the Depth_Step fails, THE Depth_Report SHALL hold exactly the counts and timings of criteria 1 and 2 that the step had worked out before the step failed.
4. THE Depth_Step SHALL add each count as a whole number given as an int and each timing as a number of seconds, so the Step_Runner's checks never refuse one.
5. WHERE the Depth_Step warns under Requirement 7, criterion 9, THE Depth_Report SHALL hold that warning.
6. THE Depth_Report SHALL be the Step_Runner's ordinary Step_Report, written to `out/reports/depth.json`, with the 8 settings of Requirement 8, criterion 2, which the Step_Runner puts there itself.

### Requirement 10: Training reads the priors

**User Story:** As someone who films a place, I want training to use the priors when they're there and to train as before when they aren't, so that the priors can only help.

**Decided:** D1 (criteria 5 to 7). **Assumptions (to confirm at review):** A9 (criteria 3 and 4), A13 (criteria 4 and 7) and A14 (criterion 2).

#### Acceptance Criteria

1. WHEN `work/depth/depth.json` exists in the Job_Folder, THE Train_Step SHALL read the Prior_Set with the Prior_Reader once the Train_Step has read the Sparse_Model, the Training_Images and the Mask_Files, and before the Train_Step sets up the Start_Splats.
2. WHERE `work/depth/depth.json` doesn't exist, THE Train_Step SHALL train with no Prior_Set, exactly as the splat-training spec says, and SHALL add no warning about the missing priors (A14).
3. IF the Prior_Index or any Prior_File fails a check of criterion 4, THEN THE Train_Step SHALL add a warning to the Train_Report that names the file and what failed and says that training ran without depth priors, and SHALL train with no Prior_Set (A9).
4. THE Train_Step SHALL check that the Prior_Index reads as a JSON object with `format` 1; that its `camera` equals the Shared_Camera's six values; that `prior_width` and `prior_height` are whole numbers above 0; that `priors` is a list of one or more objects; that each `name` is the name of an image in `images.bin` and appears once; that each Prior's `quaternion` and `translation` equal that image's Image_Pose in `images.bin`; and that each Prior_File, at the path derived from its `name`, holds exactly the three arrays of Requirement 7, criterion 1, at `prior_height` × `prior_width`, of those dtypes, with every value finite, `inverse_depth` above 0 where `mask` is true and 0 where it's false (A9, A13).
5. WHERE the Train_Step has a Prior_Set, THE Trainer SHALL keep every Prior on the GPU at the Prior_Size, about 1 GB for 400 Priors of 512 × 288 pixels (D1).
6. WHEN the Trainer trains on a Prior_View, THE Trainer SHALL scale the view's Prior up to the Training_Size with pixel centres aligned: bilinear sampling for `inverse_depth` and `normal`, with each scaled-up normal set to unit length, and nearest-neighbour sampling for `mask`, so each training pixel takes the mask of the stored pixel under its centre (D1).
7. THE Trainer SHALL count as a Prior_Pixel each training pixel of a Prior_View that isn't a Masked_Pixel, whose scaled-up `mask` is true, and whose bilinear sample draws only on stored pixels whose `mask` is true (D1, A13).
8. THE Trainer SHALL keep each Prior_View's Prior in the view's View_Record, as the splat-training spec's Requirement 5, criterion 10 left room for, and SHALL give a Training_Image with no Prior an empty prior in its View_Record.
9. THE Train_Step SHALL work out every Prior_Pixel test and the scaling of criterion 6 per training pixel, at the Training_Size (D1).

### Requirement 11: Plain pixels and the depth weight

**User Story:** As someone who films a place, I want the depth prior to count most on plain surfaces and to keep counting there all through training, so that the walls COLMAP can't see stay flat.

**Decided:** D2 (criteria 5 and 8). **Assumptions (to confirm at review):** A4 (criteria 4 and 6) and A12 (criteria 1 to 3).

#### Acceptance Criteria

1. THE Train_Step SHALL work out each training pixel's Brightness from the Training_Image at the Training_Size as its grey value from 0 to 1, as OpenCV's RGB-to-grey conversion gives it, divided by 255 (A12).
2. THE Train_Step SHALL work out each training pixel's Brightness_Gradient as the length of the gradient from central differences, (I(x + 1) − I(x − 1)) / 2 across and (I(y + 1) − I(y − 1)) / 2 down, with the border pixels repeated outward (A12).
3. THE Train_Step SHALL count as a Plain_Pixel each training pixel whose Brightness_Gradient is below `plain_gradient` (0.02, O2) (A12).
4. THE Schedules SHALL give the Depth_Schedule of Iteration i, for i from 0 to Planned_Iterations, as `depth_weight_start` × (`depth_weight_end` / `depth_weight_start`)^(i / Planned_Iterations), so it falls exponentially from 1.0 to 0.01 over the run (A4).
5. THE Train_Step SHALL work out the Depth_Weight of a training pixel at Iteration i as the Depth_Schedule of i × exp(−`depth_gradient_falloff` × the pixel's Brightness_Gradient), raised to `depth_weight_floor` (0.1) when the pixel is a Plain_Pixel and the product is below `depth_weight_floor`, so a textured pixel's weight can go lower and a Plain_Pixel's can't (D2).
6. THE Trainer SHALL plan the Depth_Schedule from the Planned_Iterations whether or not the Trainer stops early, as the splat-training spec's Requirement 7, criterion 10 plans the Means_LR (A4).
7. THE Brightness, the Brightness_Gradient, the Plain_Pixels, the Depth_Schedule and the Depth_Weight SHALL be worked out by numpy functions that import no torch, and THE Trainer SHALL get every training pixel's Depth_Weight from those functions' rule, so the rule the Test_Suite tests is the rule the Trainer uses.
8. FOR ALL Planned_Iterations of 1 or more, Iterations from 0 to Planned_Iterations, `depth_weight_start` and `depth_weight_end` above 0, `depth_weight_floor` and `depth_gradient_falloff` of 0 or more, `plain_gradient` above 0 and Brightness_Gradients of 0 or more, THE Depth_Weight of every Plain_Pixel SHALL be at least `depth_weight_floor` (property).
9. FOR ALL the same inputs, THE Depth_Weight of every training pixel that isn't a Plain_Pixel SHALL equal the Depth_Schedule × exp(−`depth_gradient_falloff` × its Brightness_Gradient), within floating-point rounding (property).

### Requirement 12: Back-projected starting splats

**User Story:** As someone who films a place, I want training to start splats on plain surfaces where COLMAP found no points, so that those surfaces have splats to train from the first iteration.

**Assumptions (to confirm at review):** A7 (criteria 1 to 7) and O5 (criterion 1).

#### Acceptance Criteria

1. WHERE the Train_Step has a Prior_Set, THE Train_Step SHALL count as a Candidate_Pixel each training pixel of a Prior_View that is a Plain_Pixel and a Prior_Pixel, and that no Observed_Point of that Frame lands on, where a point lands on the training pixel under its projection with the scaled camera, and only points in front of the camera land (A7, O5).
2. THE Train_Step SHALL start the run from min(the Back_Project_Budget, the number of Candidate_Pixels) Back_Projected_Splats, drawn without replacement and with an equal chance for each Candidate_Pixel of every Prior_View, by a generator seeded with `seed` (A7).
3. THE Train_Step SHALL place each Back_Projected_Splat at the world point of its Candidate_Pixel's centre at the camera-space z of 1 / the scaled-up `inverse_depth` at that pixel, through the view's scaled camera and the view's Image_Pose (A7).
4. THE Train_Step SHALL give each Back_Projected_Splat the pixel's 8-bit colour, and the opacity, rotation and `f_rest` coefficients that the splat-training spec's Requirement 3, criterion 12 gives every Start_Splat (A7).
5. THE Train_Step SHALL size every Start_Splat, the Back_Projected_Splats included, with the splat-training spec's rule: `start_scale_factor` × its Start_Distance to its `start_scale_neighbours` nearest other Start_Splats, found over every Start_Splat together on the GPU (A7).
6. WHERE the Train_Step has a Prior_Set, THE Train_Step SHALL start from at most `max_splats` − the number of Back_Projected_Splats Sparse_Points, chosen at random as the splat-training spec's Requirement 3, criterion 11 chooses them when there are more, so the Start_Splats never outnumber the Splat_Budget (A7).
7. WHERE the Sparse_Points outnumber what criterion 6 leaves for them, THE Train_Step SHALL add a warning to the Train_Report that gives the points, the Splat_Budget, the Back_Projected_Splats and the points used, in place of the splat-training spec's subset warning.
8. WHERE the Train_Step has no Prior_Set, THE Train_Step SHALL start from exactly the Start_Splats of the splat-training spec's Requirement 3, with no Back_Projected_Splat.
9. THE Candidate_Pixels, the draw and the world points SHALL be worked out by numpy functions that import no torch.
10. FOR ALL Prior_Views with any camera, any Image_Pose with a quaternion of length above 0, and any Candidate_Pixel whose scaled-up `inverse_depth` is above 0, THE Train_Step SHALL place the Back_Projected_Splat so that projecting its centre with the view's scaled camera and Image_Pose gives back the pixel's centre and a camera-space z of 1 / that `inverse_depth`, within a relative error of 10⁻⁹ (property, A17).

### Requirement 13: The prior loss terms

**User Story:** As someone who films a place, I want the rendered depth and the splats' normals pulled toward the fitted priors, and the splats kept flat, so that plain surfaces come out as surfaces rather than fog.

**Assumptions (to confirm at review):** A4 (criteria 6, 9, 10 and 11), A6 (criteria 1, 7 and 13) and A8 (criteria 2 to 4, 8 and 10). **Open point:** O4 (criterion 10).

#### Acceptance Criteria

1. WHERE the Train_Step has a Prior_Set, THE Trainer SHALL add the five Prior_Terms to the Loss_Table after the five splat-training Loss_Terms, each with its own name: `depth`, `normal`, `flatten`, `depth_normal` and `normal_consistency` (A6).
2. WHEN the Trainer renders a Prior_View, THE Trainer SHALL render the colours and the Rendered_Depth together, in gsplat 1.5.3's `"RGB+ED"` render mode, with every other argument as the splat-training spec's Requirement 4, criterion 1 gives it (A8).
3. THE Trainer SHALL make each splat's Splat_Normal from the rotation's axis of the splat's smallest scale, turned into the camera frame with the view's world-to-camera rotation, with its sign chosen so its dot product with the splat's centre in the camera frame isn't above 0 (A8).
4. THE Trainer SHALL make the Normal_Render with a second `rasterization()` of the same splats, with the Splat_Normals as 3-channel colours and no spherical harmonics, and with the near plane, the mode and the packing of the colour render, and SHALL leave out of the `normal` and `normal_consistency` Loss_Terms, the two that use the Normal_Render, each pixel where the Normal_Render's length is 0 (A8).
5. THE Trainer SHALL work out the `depth` Loss_Term on each Iteration of a Prior_View as the mean, over the view's Prior_Pixels, of the pixel's Depth_Weight × min(|1/z − 1/z_prior| / (1/z_prior), `depth_error_cap`), where z is the Rendered_Depth and 1/z_prior the scaled-up `inverse_depth`, with the error counted as `depth_error_cap` (1) where z isn't above 0, so a splat right in front of the camera can't dominate the loss. The Depth_Weight is inside the term, so the Trainer adds the term to the loss as it is.
6. THE Trainer SHALL work out the `normal` Loss_Term as the mean, over the Prior_Pixels, of 1 − the dot product of the Normal_Render and the scaled-up prior normal, weighted by `normal_weight` (0.05), on each Iteration i of a Prior_View with i at or after the Normal_Start and (i − the Normal_Start) a multiple of `normal_every` (4) (A4).
7. THE Trainer SHALL work out the `flatten` Loss_Term on every Iteration as the mean over splats of the splat's smallest scale divided by the Scene_Extent, in units of the Scene_Extent like `scale_reg`, weighted by `flatten_reg` (1.0) (A6).
8. THE Trainer SHALL work out each Prior_Pixel's Depth_Normal from the Rendered_Depth's points, each pixel's centre back-projected through the scaled camera to its z, as the unit cross product of the central differences across and down, with its sign chosen to face the camera, and SHALL give no Depth_Normal to a pixel on the image's border or to a pixel whose four neighbours don't all have a Rendered_Depth above 0 (A8).
9. THE Trainer SHALL work out the `depth_normal` Loss_Term as the mean, over the Prior_Pixels with a Depth_Normal, of 1 − the dot product of the Depth_Normal and the scaled-up prior normal, weighted by `depth_normal_weight` (0.05), on each Iteration of a Prior_View from the Depth_Normal_Start (A4).
10. THE Trainer SHALL work out the `normal_consistency` Loss_Term as the mean, over the Prior_Pixels with a Depth_Normal, of 1 − the dot product of the Normal_Render and the Depth_Normal, weighted by `normal_consistency_weight` (0.015), on each Iteration of a Prior_View from the Depth_Normal_Start, and SHALL make the Normal_Render only on the Iterations where the `normal` or the `normal_consistency` Loss_Term is on (A4, A8, O4).
11. THE Schedules SHALL give the Depth_Normal_Start as round(`depth_normal_start_fraction` × Planned_Iterations), with the splat-training spec's rounding for the Erank_Start: Iteration 1,633 for `fast`, 3,500 for `default` and 7,000 for `quality` (A4).
12. WHEN the Trainer trains on a Training_Image with no Prior while the Train_Step has a Prior_Set, THE Trainer SHALL count the `depth`, `normal`, `depth_normal` and `normal_consistency` Loss_Terms as 0 for that Iteration, and SHALL render the image in the `"RGB"` mode.
13. WHERE the Train_Step has no Prior_Set, THE Trainer SHALL use exactly the Loss_Table of the splat-training spec's Requirement 5, criterion 1, the Loss_Terms `l1`, `ssim`, `opacity_reg`, `scale_reg` and `erank` in that order, each on by that spec's rules, render every Training_Image in the `"RGB"` mode and make no Normal_Render (A6).
14. THE Trainer SHALL count a Prior_Term whose weight is 0, or whose Iteration rule of criteria 6, 9 and 10 is off, as 0 and not work it out, as the splat-training Trainer does for its own Loss_Terms, so a term that's off can't make the loss non-finite.
15. THE Trainer SHALL decide which Loss_Terms an Iteration works out with one pure function that imports no torch, which takes the Iteration, the checked settings, whether the Train_Step has a Prior_Set and whether the Iteration's Training_Image has a Prior.
16. WHEN an Iteration's loss isn't finite, THE Trainer SHALL stop as the splat-training spec's Requirement 4, criterion 16 says, with the value of every Loss_Term in the Loss_Table, the Prior_Terms included when there is a Prior_Set, and the total in the error's message.
17. THE Trainer SHALL compute every Prior_Term from this requirement's formulas, written fresh on gsplat's library API, with no code from Inria's gaussian-splatting repo, the 2DGS repo or any other repo, as rules.md requires.
18. FOR ALL checked settings, Planned_Iterations and Iterations, with no Prior_Set, THE function of criterion 15 SHALL give exactly the Loss_Terms that the splat-training spec's Requirement 5 works out: each of `l1`, `ssim`, `opacity_reg` and `scale_reg` whose weight isn't 0, and `erank` when its weight isn't 0 and the Iteration is at or after the Erank_Start, in the Loss_Table's order (property).

### Requirement 14: Training settings, report and the splat-training criteria

**User Story:** As a team member, I want the new training numbers in Config and the train report to say what the priors added, so that we can tune them and tell a prior run from a plain one.

**Assumptions (to confirm at review):** A14 (criteria 3 and 4) and A18 (criterion 8).

#### Acceptance Criteria

1. THE `[train]` section SHALL gain these Shared_Settings, each with a short comment: `plain_gradient = 0.02` (O2), `depth_gradient_falloff = 10`, `depth_weight_start = 1.0`, `depth_weight_end = 0.01`, `depth_weight_floor = 0.1`, `depth_error_cap = 1.0`, `normal_weight = 0.05`, `normal_start_step = 3000`, `normal_every = 4`, `flatten_reg = 1.0`, `depth_normal_weight = 0.05`, `normal_consistency_weight = 0.015`, `depth_normal_start_fraction = 0.23333333333333334`, `backproject_share = 0.3333333333333333` and `backproject_max = 500000`.
2. IF a new `[train]` setting is present but isn't the kind of value the setting must be, THEN THE Train_Step SHALL fail with the splat-training spec's Requirement 9, criterion 7 Expected_Failure, with these kinds: `plain_gradient`, `depth_weight_start`, `depth_weight_end` and `depth_error_cap` numbers above 0; `depth_gradient_falloff`, `depth_weight_floor`, `normal_weight`, `flatten_reg`, `depth_normal_weight` and `normal_consistency_weight` numbers of 0 or more, where a weight of 0 turns its Prior_Term off; `normal_start_step` and `backproject_max` whole numbers of 0 or more, where a `backproject_max` of 0 turns the Back_Projected_Splats off; `normal_every` a whole number above 0; and `depth_normal_start_fraction` and `backproject_share` numbers from 0 to 1.
3. WHERE the Train_Step has no Prior_Set, THE Train_Report SHALL hold exactly the counts, timings and warnings of the splat-training spec's Requirement 10, and its `settings` SHALL hold the new keys of criterion 1 as well (A14).
4. WHERE the Train_Step has a Prior_Set, THE Train_Report SHALL also hold the counts `prior_images` (the Prior_Views) and `backprojected_splats` (the Back_Projected_Splats), and `start_splats` SHALL count every Start_Splat, the Back_Projected_Splats included (A14).
5. WHERE the Train_Step warns under Requirement 10, criterion 3 or Requirement 12, criterion 7, THE Train_Report SHALL hold that warning.
6. THE Train_Step SHALL count reading and checking the Prior_Set and setting up the Back_Projected_Splats in its `load` timing.
7. THE Train_Step SHALL do its work in the splat-training spec's order (its Requirement 11, criterion 3), with the Prior_Set read and checked after the Training_Images and the Mask_Files and before the scene and the Start_Splats are set up.
8. THE splat-training spec's criteria that this spec changes SHALL be these, and that spec's requirements file SHALL stay as it is (A18): Requirement 1, criteria 5 and 21 (Requirement 2 of this spec); Requirement 3, criterion 11 (Requirement 12); Requirement 5, criteria 1 and 10 (Requirements 10 and 13); Requirement 9, criteria 2 and 15 (criteria 1 and 10); Requirement 10, criteria 1, 5 and 6 (criteria 3 to 5); Requirement 11, criteria 3, 4, 9 and 11 (criterion 7 and Requirements 1 and 18); Requirement 13, criterion 8 (Requirement 17); and Requirement 14, criterion 9 (Requirement 18).
9. THE Train_Step's modules SHALL keep the splat-training spec's import rules (its Requirement 9, criteria 11 to 13): the Prior_Format module is part of `splatpipe` and imports no torch, and THE GPU_Module SHALL import no GPU package besides torch, gsplat and fused-ssim.
10. THE `tests/conftest.py` test config's `[train]` section SHALL gain the new keys of criterion 1, so a test that runs the Train_Step without a Config of its own still gets every setting.

### Requirement 15: MoGe on the machine image

**User Story:** As a team member, I want MoGe's code and exactly the packages its MoGe-2 path needs on the machine image, each pinned and each with its licence row, so that the depth step runs there and nothing we can't use comes with it.

**Decided:** D4 (criteria 1 to 3, 7, 8, 10 and 12) and D5 (criterion 9). **Assumptions (to confirm at review):** A19 (criterion 6). **Open point:** O3 (criterion 2).

#### Acceptance Criteria

1. THE Image_Install SHALL install, after the GPU packages and with `--no-deps`, MoGe's code from `git+https://github.com/microsoft/MoGe` at commit `74fbce054ebed49800de42d0ad0e83495065719a` and utils3d_moge from `git+https://github.com/EasternJournalist/utils3d-moge` at commit `62f09d58509485564e24d5d9f6aac9ee9ebc0c37`, so moderngl, trimesh, gradio, pipeline, FlexGEMM and MoGe's other declared dependencies stay off the Machine_Image (D4).
2. THE Image_Install SHALL install huggingface-hub 1.26.0 and click, filelock, fsspec, httpx 0.28.1, packaging, pyyaml, tqdm, typing-extensions, anyio 4.14.2, certifi, httpcore 1.0.9, idna 3.18 and h11, each pinned with `==` at the version MoGe's `uv.lock` gives at the pinned commit, with `--no-deps`, so no other package comes in (D4, O3).
3. WHEN the MoGe_Packages are installed, THE Image_Install SHALL fail the Image_Build unless `from moge.model.v2 import MoGeModel` imports in the image's Python environment with `HF_HUB_OFFLINE=1`, as the Image_Install already does for gsplat and fused-ssim. This import also loads huggingface_hub's download code, through the `hf_hub_download` import at the top of `moge/model/v2.py`, so the check proves that nothing on the MoGe-2 path is missing with hf-xet left off (D4).
4. THE Image_Install SHALL stay within the 16 KB EC2 User_Data limit, which its existing check enforces.
5. THE Project_File and `uv.lock` SHALL gain no package, so `uv sync` on a laptop still installs no torch, MoGe, utils3d_moge, huggingface-hub or hf-xet.
6. THE spec's tasks SHALL end before any Image_Build and any Smoke_Test, because both are paid runs that the user starts, and no task SHALL set `APPROVED_SPEND` (A19).
7. THE Licence_List SHALL gain a row, in the same task that changes the Image_Install, for each of huggingface-hub, click, httpx, httpcore, h11, anyio, idna, pyyaml and utils3d_moge, with the package's licence, its version or commit, how we use it, what it comes in through, and "Cloud image only", and SHALL gain no row for hf-xet (D4).
8. THE Licence_List's rows for filelock, fsspec, packaging and typing-extensions SHALL also say that they come in through huggingface-hub, and the MoGe code row SHALL say that MoGe's code goes in with `--no-deps` at its pinned commit and that only utils3d_moge, of its git dependencies, is installed (D4).
9. THE task that changes the Image_Install SHALL change both the Licence_List's certifi row and rules.md's named-exceptions line for certifi, so each says that on the Machine_Image certifi also comes in through httpx and httpcore, used unmodified, with rules.md's line recording that Samarth OKs it, and SHALL change no other line of rules.md, because rules.md is a steering file (D5).
10. THE `tests/test_cloud_image.py` tests SHALL check that the Image_Install installs each MoGe_Package with its pin and `--no-deps`, runs the import check of criterion 3, and names none of moderngl, trimesh, gradio, flex-gemm, hf-xet, `hf_xet` or MoGe's `pipeline` package on any line outside its comments, because the comments name the packages that stay off and the script's opening comment already says "Pipeline".
11. THE comment above `MOGE_WEIGHTS` in `cloud/image.py` SHALL say that the Depth_Step finds the Weights_File through `moge_weights` in `[depth]`, which a test keeps equal to `MOGE_WEIGHTS`.
12. THE Image_Install SHALL leave hf-xet off the Machine_Image, though huggingface-hub 1.26.0 declares it, because huggingface_hub 1.26.0 imports hf_xet only inside `xet_get` and other functions and falls back to plain HTTP without it, and the Depth_Step downloads nothing: `HF_HUB_OFFLINE=1` is set and the Weights_File is a local file (D4).

### Requirement 16: The smoke test's depth check

**User Story:** As a team member, I want the smoke test to run MoGe-2 and the trainer with the prior losses on the GPU, so that a broken image or a bug in the new GPU code shows up in a short smoke test and not in a job.

**Assumptions (to confirm at review):** A5 (criteria 4 to 6) and A19 (criterion 11).

#### Acceptance Criteria

1. THE SMOKE_CHECKS SHALL gain the Depth_Smoke_Check, named `depth`, after `moge-weights` and before `training`.
2. THE Depth_Smoke_Check SHALL load the MoGe_Model with the Depth_Step's own load of Requirement 3, from the real Config's `moge_weights`, and run MoGe once, with the Depth_Step's own call of Requirement 4, on a made-up RGB image of `image_pixels` × `image_pixels` pixels with a made-up PINHOLE camera's Field_Of_View, drawn from a generator seeded with `[train]`'s `seed`.
3. THE Depth_Smoke_Check SHALL pass MoGe's part only when the MoGe_Output's three arrays have the image's size and the dtypes of Requirement 4, criterion 6, and `depth` and `normal` are finite and `depth` is above 0 wherever `mask` is true.
4. THE Depth_Smoke_Check SHALL then build the Prior_Smoke_Scene with numpy functions that import no torch: the Smoke_Scene's cameras and points, images that each hold a plain area of one colour, and a made-up Prior for every image, written in the Prior_Format into the check's scratch folder and read back with the Train_Step's Prior_Reader and checks (A5).
5. THE Depth_Smoke_Check SHALL run the Train_Step's own Start_Splats with Back_Projected_Splats and the Trainer with the Prior_Set on the GPU, with the training check's `[train]` settings and `[cloud.smoke]` numbers, except that `[cloud.smoke]`'s `depth_normal_start_step` sets the Normal_Start, then bake the Mean_Exposure in, write every splat with the PLY_Writer, dropping none, and run the PLY_Check, as the Train_Smoke_Check does (A5).
6. THE `[cloud.smoke]` table SHALL gain `depth_normal_start_step = 8`, with a comment, and THE Cloud_Config SHALL gain a field in `SmokeChecks` for it, which the loader in `cloud/config.py` reads and checks as a whole number of 0 or more, refusing a bad value with the loader's Readable_Error that names the key (A5).
7. THE `[cloud.smoke]` numbers SHALL make the Depth_Smoke_Check's run work out every Prior_Term on at least one Iteration and start at least one Back_Projected_Splat, which the Test_Suite checks on the CPU with the Schedules, the function of Requirement 13, criterion 15, the Prior_Smoke_Scene and the real Config.
8. THE Depth_Smoke_Check SHALL pass only when MoGe's part passed, the Trainer ran every planned Iteration, at least one Back_Projected_Splat started and the PLY_Check passed, and SHALL give back a line for the log with the MoGe_Output's Geometry_Pixels, the Back_Projected_Splats, the Iterations run and the splats written.
9. IF any part of the Depth_Smoke_Check fails, THEN THE Depth_Smoke_Check SHALL raise an error whose message says what failed, so the Smoke_Test reports the check as failed and goes on with the other checks.
10. THE Depth_Smoke_Check SHALL import the Depth_GPU_Module and the GPU_Module inside the check's own function, so `cloud/smoke_runner.py` still imports on a laptop, and SHALL run with no time limit of its own, as the Train_Smoke_Check does.
11. THE Depth_Smoke_Check SHALL pass only on a Machine_Image built with the MoGe_Packages, so a Smoke_Test on the current Machine_Image fails the check until the user runs the Image_Build, and no task of this spec runs the Image_Build or the Smoke_Test (A19).
12. THE `tests/test_cloud_smoke_runner.py` test of the checks' order, the `[cloud.smoke]` tables in `tests/fake_aws.py` and `tests/test_cloud_config.py`, and the Pipeline_Doc's smoke test part SHALL change in the same task as SMOKE_CHECKS.

### Requirement 17: Docs and the MoGe skill

**User Story:** As one of the three people building the steps, I want the depth step and its prior format written up next to the other steps, so that whatever makes priors later fits the train step without reading its code.

**Decided:** D4 (criterion 8). **Assumptions (to confirm at review):** A11 (criterion 4) and A13 (criterion 2).

#### Acceptance Criteria

1. THE Pipeline_Doc SHALL gain a `## Step: depth` section, after `## Step: frames` and before `## Step: train`, with the parts `### What it reads`, `### What it writes`, `### Config section`, `### Step report fields` and `### Files in out/`.
2. THE `### What it writes` part SHALL give the whole Prior_Format: the Prior_Files' name, encoding, arrays, dtypes, units, camera frame and pixel centres; the Prior_Index's fields; that the Prior_Index is written last; and how the Train_Step scales a Prior up and which pixels count as Prior_Pixels (A13).
3. THE Pipeline_Doc's `## Step: train` section SHALL say that the Train_Step reads `work/depth/` when the Prior_Index is there, what it checks and when it warns, how it starts Back_Projected_Splats, the five Prior_Terms and when each is on, the new `[train]` keys, the new counts and warnings, and that the loss-not-finite message lists every Loss_Term of the Loss_Table.
4. THE Pipeline_Doc's "The step list" section SHALL say that the list holds the Frames_Step, the Depth_Step (optional), the Train_Step and the Clean_Step (optional), that the camera solve spec puts its entry between the Frames_Step and the Depth_Step, so the order becomes frames, cameras, depth, train, clean, and that until then a Job runs the Frames_Step, fails the Depth_Step and goes on, and fails at the Train_Step because the camera solve output is missing (A11).
5. THE Pipeline_Doc's "The smoke test" part SHALL list the Depth_Smoke_Check among the checks and say that its number is in `[cloud.smoke]`.
6. THE Pipeline_Doc's Step_Needs table SHALL keep the depth priors row as a GPU with 16 GB VRAM, and say that the need lives in `[depth]` as `min_gpu_memory_gb`.
7. THE Licence_List's `opencv-python-headless` row SHALL say in its "How we use it" cell that the Depth_Step decodes the camera solve's JPEG frames and scales them down, and SHALL still say that it never reads or writes video.
8. THE moge-depth skill's sentences that say the image has no MoGe code yet, that huggingface-hub and the packages that come in through it have no Licence_List rows yet, or that the depth-priors spec decides how the step finds the Weights_File or which packages go on the image, SHALL say what this spec decided: `moge_weights` in `[depth]`, and MoGe and utils3d_moge with `--no-deps` plus huggingface-hub and the other packages of D4, also with `--no-deps` and each with its Licence_List row, with hf-xet left off and given no row, on the image once the user runs the next Image_Build.
9. THE `tests/test_docs.py` SHALL check the step list sentence of criterion 4 in place of the old one, that the `## Step: depth` section's parts name every key of the real `[depth]`, every count and timing of Requirement 9, `work/depth/depth.json` and the three array names, and that the `## Step: train` section names every new key of the real `[train]`.

### Requirement 18: Tests

**User Story:** As a team member, I want every CPU part of the priors tested on a laptop with tiny made-up data, so that the tests are quick, free and safe to run anywhere, and the GPU only runs code that is already tested.

**Assumptions (to confirm at review):** A17 (criterion 2).

#### Acceptance Criteria

1. THE Test_Suite SHALL test with unit tests, on the CPU with tiny synthetic data and no GPU: the Model_Reader's point ids and Observed_Points (keypoints with −1 and with ids of no point included); the Prior_Size and the Field_Of_View; the Fit_Points (points behind the camera, outside the Frame and on pixels with no geometry included); the Depth_Fit's two passes, each reason a Frame is left out and the 15% boundary; the Metres_Per_Unit; the Prior_Writer and the Prior_Reader, and every check of Requirement 10, criterion 4; the scaling of a Prior to the Training_Size and the Prior_Pixels, the mask's edge included; the Brightness, the Brightness_Gradient and the Plain_Pixels; the Depth_Schedule at its first and last Iteration and the Depth_Weight; the Normal_Start, the Depth_Normal_Start and the function of Requirement 13, criterion 15; the Candidate_Pixels, the draw, the Back_Project_Budget and the world points; the setting checks of `[depth]` and the new `[train]` keys; and the report fields.
2. THE Test_Suite SHALL hold the property tests of Requirement 2, criterion 9, Requirement 5, criteria 12 and 13, Requirement 7, criterion 12, Requirement 11, criteria 8 and 9, Requirement 12, criterion 10, and Requirement 13, criterion 18, written with Hypothesis (A17).
3. THE Test_Suite's unit tests and property tests SHALL both be required, never optional.
4. THE Test_Suite SHALL run the Depth_Step through `run()` on the CPU with a Depth_Stand_In, to test the order of the step's work, every Expected_Failure that needs no GPU, a done run that leaves the Prior_Files and the Prior_Index in `work/depth/`, a run in which some Frames are left out with the warning of Requirement 7, criterion 9, a run in which no Frame fits that fails and leaves `work/depth/` empty, and that the step releases the GPU's memory when it's done and when it fails.
5. THE Test_Suite SHALL run the Train_Step through `run()` on the CPU with a GPU_Stand_In, to test a run with a Prior_Set that gives the Trainer the Priors, the Back_Projected_Splats and the counts of Requirement 14, criterion 4; a run with a broken Prior_Index and one with a Prior_File for an earlier camera solve, each with the warning of Requirement 10, criterion 3 and no Prior_Set; and a run with no `work/depth/`, whose Train_Report holds exactly the splat-training counts, timings and warnings.
6. THE Test_Suite SHALL check, by listing the Job_Folder before and after a run with the Depth_Stand_In, that the Depth_Step writes nothing outside `work/depth/` and the Depth_Report.
7. THE Test_Suite SHALL check that each Readable_Error of the Depth_Step's Expected_Failures names what the requirement says the Readable_Error names and holds no traceback.
8. THE Test_Suite SHALL check that every file in `splatpipe/depth/` other than the Depth_GPU_Module imports only names in the standard library, numpy, `cv2` and `splatpipe`; that the Depth_GPU_Module imports, besides those, only torch at its top and MoGe and huggingface_hub inside functions; that the Depth_GPU_Module's source sets `HF_HUB_OFFLINE` before those imports, passes `weights_only=True` and calls no `from_pretrained`, `hf_hub_download` or `init_weights`; and that `import splatpipe.depth` in a fresh process loads no torch, moge, utils3d_moge or huggingface_hub.
9. THE `tests/test_cloud_imports.py` check SHALL keep refusing `cloud` and boto3 in every file under `splatpipe/`, and SHALL keep refusing every GPU package in every file except the two GPU modules: the GPU_Module, which may import torch, gsplat and fused_ssim, and the Depth_GPU_Module, which may import torch, moge and huggingface_hub.
10. THE Test_Suite SHALL check that `[depth]` and `[train]` load from the real Config for each of the three Presets with exactly the keys of Requirements 8 and 14, and that, with the real Config and Step_List, every type on each Preset's Fallback_List still meets the Step_Needs that the Start_Command works out.
11. THE Test_Suite SHALL check the Depth_Smoke_Check's place in SMOKE_CHECKS, the Prior_Smoke_Scene on the CPU against the Train_Step's own Prior checks, the numbers of Requirement 16, criterion 7, and the new `[cloud.smoke]` key and its refusals.
12. THE `tests/test_run.py` test that asserts the Step_List SHALL assert the Step_List of Requirement 1, criterion 3, and the splat-training spec's test of a Job through the real Step_List SHALL expect the Frames_Step done, the Depth_Step failed with the Readable_Error of Requirement 2, criterion 2, and the Train_Step failed, in the same task that changes the Step_List.
13. THE Test_Suite SHALL run on the CPU with no network, make every Job in a temporary Jobs_Folder and never in the real `data/` folder, build every image, model, Prior and splat file in the test, and commit none to git.
