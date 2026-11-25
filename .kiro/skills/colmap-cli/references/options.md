# COLMAP 4.2.1 options the camera solve passes

Part of the `colmap-cli` skill. Each block is a full command the way the step runs it, with every
option that can change what the command does, so that no COLMAP default decides anything. The
option lists come from `colmap <command> -h` on Homebrew's COLMAP 4.2.1 (November 2026), which has
the same options as the image's build for these commands.

Values are 4.2.1's defaults, except where a command's notes say otherwise. Where `-h` rounds a
decimal, the value is the exact one from the source. The step takes every value from its
`config.toml` section; these are where that section starts. `$FRAMES` is the frames step's work
folder and `$OUT` the camera solve's, as in SKILL.md.

Every call starts with these five, which every command has:

- `--log_target stderr`: all output goes to stderr, where the step captures it. Once the options
  have parsed, glog writes no log files. A command whose options fail to parse has already
  written them, to `$TMPDIR`, or `/tmp` without it.
- `--log_level 0` and `--log_severity 0`: glog's verbosity, and the lowest level it shows, 0 for
  INFO. The log keeps every INFO line.
- `--log_color 0`: no colour codes in the captured log. With the default 1, glog colours warning
  and error lines whenever `TERM` names a colour terminal, even through a pipe. The option exists
  only in builds against glog 0.4 or newer, and a build without it stops with `unrecognised
  option`.
- `--default_random_seed 0`: the seed each thread's random generator starts from, 0 by default.
  Extraction and undistortion have no random parts, but passing it on every call keeps the rule
  simple.

Every call leaves out `--help`, `--project_path`, which in 4.2.1 replaces every other flag, and
`--log_path`, the folder for log files, which `--log_target stderr` doesn't write.

## feature_extractor

```bash
colmap feature_extractor \
  --database_path "$OUT/database.db" \
  --image_path "$FRAMES/grey" \
  --log_target stderr \
  --log_level 0 \
  --log_severity 0 \
  --log_color 0 \
  --default_random_seed 0 \
  --camera_mode 1 \
  --descriptor_normalization l1_root \
  --ImageReader.camera_model SIMPLE_RADIAL \
  --ImageReader.existing_camera_id -1 \
  --ImageReader.default_focal_length_factor 1.2 \
  --ImageReader.camera_mask_path "$FRAMES/mask.png" \
  --FeatureExtraction.type SIFT \
  --FeatureExtraction.num_threads -1 \
  --FeatureExtraction.use_gpu 0 \
  --FeatureExtraction.max_image_size 3200 \
  --SiftExtraction.max_num_features 8192 \
  --SiftExtraction.first_octave -1 \
  --SiftExtraction.num_octaves 4 \
  --SiftExtraction.octave_resolution 3 \
  --SiftExtraction.peak_threshold 0.006666666666666667 \
  --SiftExtraction.edge_threshold 10 \
  --SiftExtraction.estimate_affine_shape 0 \
  --SiftExtraction.max_num_orientations 2 \
  --SiftExtraction.upright 0 \
  --SiftExtraction.domain_size_pooling 0 \
  --SiftExtraction.dsp_min_scale 0.16666666666666666 \
  --SiftExtraction.dsp_max_scale 3 \
  --SiftExtraction.dsp_num_scales 10
```

- `--camera_mode 1` (default -1) gives every pick the same camera. It overrides
  `--ImageReader.single_camera`, `--ImageReader.single_camera_per_folder` and
  `--ImageReader.single_camera_per_image`, so the step leaves those three out.
- `--ImageReader.camera_mask_path` only goes in when `frames.json` has `mask_used`.
- `--FeatureExtraction.use_gpu 0` is the default in the image's build and 1 in brew's.
- `--FeatureExtraction.max_image_size 3200` replaces the default -1, which means 3200 for SIFT.
  Any value of at least the picks' long edge keeps them at full size.
- `--SiftExtraction.peak_threshold` and `--SiftExtraction.dsp_min_scale` are the exact defaults,
  0.02 / 3 and 1 / 6, which `-h` shows as 0.00667 and 0.167. `peak_threshold` doesn't follow
  `octave_resolution`, so change the two together.
- `--ImageReader.existing_camera_id -1` makes a new camera instead of reusing one from the
  database. A database left over from an earlier run would have one.
- `--ImageReader.camera_model` takes `camera_model`, RADIAL or RADIAL_FISHEYE, in place of the
  block's `SIMPLE_RADIAL`, COLMAP's default. `--ImageReader.camera_params` goes in only for
  RADIAL_FISHEYE, with the Fisheye_Start f, cx, cy, k1, k2, just after
  `--ImageReader.camera_model`. For RADIAL it's left out, so the focal starts from
  `default_focal_length_factor`.
- Left out: `--image_list_path` and `--ImageReader.mask_path`, inputs the step doesn't have. Left
  empty, COLMAP reads every image below `--image_path` and uses no per-image masks.
  `--FeatureExtraction.gpu_index`, and all the `--AlikedExtraction.*` and `--LomaExtraction.*`
  options, which a build without ONNX can't use.

## sequential_matcher

```bash
colmap sequential_matcher \
  --database_path "$OUT/database.db" \
  --log_target stderr \
  --log_level 0 \
  --log_severity 0 \
  --log_color 0 \
  --default_random_seed 0 \
  --FeatureMatching.type SIFT_BRUTEFORCE \
  --FeatureMatching.num_threads -1 \
  --FeatureMatching.use_gpu 0 \
  --FeatureMatching.guided_matching 0 \
  --FeatureMatching.skip_geometric_verification 0 \
  --FeatureMatching.rig_verification 0 \
  --FeatureMatching.skip_image_pairs_in_same_frame 0 \
  --FeatureMatching.max_num_matches 32768 \
  --SiftMatching.max_ratio 0.8 \
  --SiftMatching.max_distance 0.7 \
  --SiftMatching.cross_check 1 \
  --SiftMatching.cpu_brute_force_matcher 0 \
  --TwoViewGeometry.min_num_inliers 15 \
  --TwoViewGeometry.multiple_models 0 \
  --TwoViewGeometry.compute_relative_pose 0 \
  --TwoViewGeometry.detect_watermark 1 \
  --TwoViewGeometry.multiple_ignore_watermark 1 \
  --TwoViewGeometry.watermark_detection_max_error 4 \
  --TwoViewGeometry.filter_stationary_matches 0 \
  --TwoViewGeometry.stationary_matches_max_error 4 \
  --TwoViewGeometry.use_degensac 0 \
  --TwoViewGeometry.use_sampson_refinement 1 \
  --TwoViewGeometry.max_error 4 \
  --TwoViewGeometry.confidence 0.999 \
  --TwoViewGeometry.max_num_trials 10000 \
  --TwoViewGeometry.min_inlier_ratio 0 \
  --TwoViewGeometry.random_seed 0 \
  --SequentialMatching.overlap 10 \
  --SequentialMatching.quadratic_overlap 1 \
  --SequentialMatching.expand_rig_images 1 \
  --SequentialMatching.loop_detection 0
```

- `--FeatureMatching.use_gpu 0` is the default in the image's build and 1 in brew's.
- `--TwoViewGeometry.random_seed 0` replaces the default -1. Each RANSAC run then starts from the
  same seed, so the pair checks repeat.
- `--TwoViewGeometry.min_inlier_ratio 0` turns that check off. RANSAC's own 0.25 minimum inlier
  ratio and 100 minimum trials have no option, so those stay as built.
- Left out: `--FeatureMatching.gpu_index`, `--SiftMatching.lightglue_min_score`,
  `--SiftMatching.lightglue_model_path`, and all the `--AlikedMatching.*` and `--LomaMatching.*`
  options, which a build without ONNX can't use. With loop detection off for good, also
  `--SequentialMatching.vocab_tree_path`, `--SequentialMatching.num_threads` (loop detection only)
  and the seven `--SequentialMatching.loop_detection_*` settings.

## view_graph_calibrator

```bash
colmap view_graph_calibrator \
  --database_path "$OUT/database_global.db" \
  --log_target stderr \
  --log_level 0 \
  --log_severity 0 \
  --log_color 0 \
  --default_random_seed 0 \
  --cross_validate_prior_focal_lengths 1 \
  --min_calibrated_pair_ratio 0.5 \
  --reestimate_relative_pose 1 \
  --min_focal_length_ratio 0.1 \
  --max_focal_length_ratio 10 \
  --max_calibration_error 2 \
  --relpose_max_error 1 \
  --relpose_min_num_inliers 30 \
  --relpose_min_inlier_ratio 0.25
```

- It has no seed of its own. Its relative pose RANSAC follows `--default_random_seed`.

## global_mapper

```bash
colmap global_mapper \
  --database_path "$OUT/database_global.db" \
  --image_path "$FRAMES/images" \
  --output_path "$OUT/sparse_global" \
  --log_target stderr \
  --log_level 0 \
  --log_severity 0 \
  --log_color 0 \
  --default_random_seed 0 \
  --GlobalMapper.min_num_matches 15 \
  --GlobalMapper.ignore_watermarks 0 \
  --GlobalMapper.num_threads -1 \
  --GlobalMapper.random_seed 0 \
  --GlobalMapper.decompose_relative_pose 1 \
  --GlobalMapper.multiple_models 1 \
  --GlobalMapper.min_model_size 3 \
  --GlobalMapper.ba_num_iterations 3 \
  --GlobalMapper.skip_rotation_averaging 0 \
  --GlobalMapper.skip_track_establishment 0 \
  --GlobalMapper.skip_global_positioning 0 \
  --GlobalMapper.skip_bundle_adjustment 0 \
  --GlobalMapper.skip_retriangulation 0 \
  --GlobalMapper.track_intra_image_consistency_threshold 10 \
  --GlobalMapper.track_required_tracks_per_view 2147483647 \
  --GlobalMapper.track_min_num_views_per_track 3 \
  --GlobalMapper.track_max_num_views_per_track 100 \
  --GlobalMapper.keep_max_num_tracks 2147483647 \
  --GlobalMapper.gp_use_gpu 0 \
  --GlobalMapper.gp_optimize_positions 1 \
  --GlobalMapper.gp_optimize_points 1 \
  --GlobalMapper.gp_optimize_scales 1 \
  --GlobalMapper.gp_loss_function_scale 0.1 \
  --GlobalMapper.gp_max_num_iterations 100 \
  --GlobalMapper.ba_refine_focal_length 1 \
  --GlobalMapper.ba_refine_principal_point 0 \
  --GlobalMapper.ba_refine_extra_params 1 \
  --GlobalMapper.refine_sensor_from_rig 1 \
  --GlobalMapper.ba_refine_rig_from_world 1 \
  --GlobalMapper.ba_refine_points3D 1 \
  --GlobalMapper.ba_min_track_length 3 \
  --GlobalMapper.ba_backend CERES \
  --GlobalMapper.ba_ceres_use_gpu 0 \
  --GlobalMapper.ba_ceres_loss_function_scale 1 \
  --GlobalMapper.ba_ceres_max_num_iterations 200 \
  --GlobalMapper.ba_skip_fixed_rotation_stage 0 \
  --GlobalMapper.ba_skip_joint_optimization_stage 0 \
  --GlobalMapper.tri_complete_max_reproj_error 15 \
  --GlobalMapper.tri_merge_max_reproj_error 15 \
  --GlobalMapper.tri_min_angle 1 \
  --GlobalMapper.ra_use_gravity 0 \
  --GlobalMapper.ra_use_stratified 1 \
  --GlobalMapper.ra_max_rotation_error_deg 10 \
  --GlobalMapper.ra_reweighting UNIFORM \
  --GlobalMapper.max_angular_reproj_error_deg 1 \
  --GlobalMapper.max_normalized_reproj_error 0.01 \
  --GlobalMapper.min_tri_angle_deg 1
```

- `--GlobalMapper.gp_use_gpu 0` and `--GlobalMapper.ba_ceres_use_gpu 0` replace the defaults 1 and
  1. This build has no CUDA.
- `--GlobalMapper.random_seed 0` replaces the default -1, under which rotation averaging and
  positioning are seeded from the clock. It seeds them and retriangulation, and turns off a
  custom solver ordering in positioning, which can make positioning slower.
- 2147483647 means no limit for the two track counts.
- Left out: `--GlobalMapper.image_list_path`, an input the step doesn't have, so every image is
  used. And the GPU indices `--GlobalMapper.gp_gpu_index` and `--GlobalMapper.ba_gpu_index`.

## mapper

```bash
colmap mapper \
  --database_path "$OUT/database.db" \
  --image_path "$FRAMES/images" \
  --output_path "$OUT/sparse" \
  --log_target stderr \
  --log_level 0 \
  --log_severity 0 \
  --log_color 0 \
  --default_random_seed 0 \
  --Mapper.min_num_matches 15 \
  --Mapper.ignore_watermarks 0 \
  --Mapper.multiple_models 1 \
  --Mapper.max_num_models 50 \
  --Mapper.max_model_overlap 20 \
  --Mapper.min_model_size 10 \
  --Mapper.init_image_id1 -1 \
  --Mapper.init_image_id2 -1 \
  --Mapper.init_num_trials 200 \
  --Mapper.structure_less_registration_fallback 1 \
  --Mapper.structure_less_registration_only 0 \
  --Mapper.extract_colors 1 \
  --Mapper.num_threads -1 \
  --Mapper.random_seed 0 \
  --Mapper.min_focal_length_ratio 0.1 \
  --Mapper.max_focal_length_ratio 10 \
  --Mapper.max_extra_param 1 \
  --Mapper.ba_refine_focal_length 1 \
  --Mapper.ba_refine_principal_point 0 \
  --Mapper.ba_refine_extra_params 1 \
  --Mapper.ba_refine_sensor_from_rig 1 \
  --Mapper.ba_local_function_tolerance 0 \
  --Mapper.ba_local_max_num_iterations 25 \
  --Mapper.ba_global_frames_ratio 1.1 \
  --Mapper.ba_global_points_ratio 1.1 \
  --Mapper.ba_global_frames_freq 500 \
  --Mapper.ba_global_points_freq 250000 \
  --Mapper.ba_global_function_tolerance 0 \
  --Mapper.ba_global_max_num_iterations 50 \
  --Mapper.ba_global_max_refinements 5 \
  --Mapper.ba_global_max_refinement_change 0.0005 \
  --Mapper.ba_local_max_refinements 2 \
  --Mapper.ba_local_max_refinement_change 0.001 \
  --Mapper.ba_use_gpu 0 \
  --Mapper.ba_local_backend CERES \
  --Mapper.ba_global_backend CERES \
  --Mapper.ba_min_num_residuals_for_cpu_multi_threading 50000 \
  --Mapper.snapshot_frames_freq 0 \
  --Mapper.fix_existing_frames 0 \
  --Mapper.init_min_num_inliers 100 \
  --Mapper.init_max_error 4 \
  --Mapper.init_max_forward_motion 0.95 \
  --Mapper.init_min_tri_angle 16 \
  --Mapper.init_max_reg_trials 2 \
  --Mapper.abs_pose_max_error 12 \
  --Mapper.abs_pose_min_num_inliers 30 \
  --Mapper.abs_pose_min_inlier_ratio 0.25 \
  --Mapper.filter_max_reproj_error 4 \
  --Mapper.filter_min_tri_angle 1.5 \
  --Mapper.max_reg_trials 3 \
  --Mapper.ba_local_num_images 6 \
  --Mapper.ba_local_min_tri_angle 6 \
  --Mapper.ba_global_ignore_redundant_points3D 0 \
  --Mapper.ba_global_ignore_redundant_points3D_min_coverage_gain 0.05 \
  --Mapper.max_runtime_seconds -1 \
  --Mapper.tri_max_transitivity 1 \
  --Mapper.tri_create_max_angle_error 2 \
  --Mapper.tri_continue_max_angle_error 2 \
  --Mapper.tri_merge_max_reproj_error 4 \
  --Mapper.tri_complete_max_reproj_error 4 \
  --Mapper.tri_complete_max_transitivity 5 \
  --Mapper.tri_re_max_angle_error 5 \
  --Mapper.tri_re_min_ratio 0.2 \
  --Mapper.tri_re_max_trials 1 \
  --Mapper.tri_min_angle 1.5 \
  --Mapper.tri_ignore_two_view_tracks 1
```

- `--Mapper.random_seed 0` replaces the default -1.
- `--Mapper.ba_local_max_num_iterations 25` and `--Mapper.ba_global_max_num_iterations 50` replace
  the defaults -1, which mean these Ceres values.
- `--Mapper.max_runtime_seconds -1` means no limit. A cap below the job's time limit stops the run
  and keeps what's built.
- `--Mapper.init_image_id1 -1` and `--Mapper.init_image_id2 -1` let the mapper choose the
  starting pair. `--Mapper.snapshot_frames_freq 0` writes no snapshots.
  `--Mapper.fix_existing_frames 0` only matters with `--input_path`.
- Left out: `--input_path`, a model to continue from. `--Mapper.image_list_path`,
  `--Mapper.constant_rig_list_path`, `--Mapper.constant_camera_list_path` and
  `--Mapper.snapshot_path`, inputs the step doesn't have. Left empty, the mapper uses every image
  and holds no camera or rig fixed. And the GPU index `--Mapper.ba_gpu_index`.

## image_undistorter

```bash
colmap image_undistorter \
  --image_path "$FRAMES/images" \
  --input_path "$OUT/sparse/0" \
  --output_path "$OUT/undistorted" \
  --log_target stderr \
  --log_level 0 \
  --log_severity 0 \
  --log_color 0 \
  --default_random_seed 0 \
  --output_type COLMAP \
  --copy_policy COPY \
  --blank_pixels 0 \
  --min_scale 0.2 \
  --max_scale 2 \
  --max_image_size -1 \
  --roi_min_x 0 \
  --roi_min_y 0 \
  --roi_max_x 1 \
  --roi_max_y 1 \
  --jpeg_quality 100 \
  --num_threads -1
```

- `--jpeg_quality 100` replaces the default -1, which writes at 100.
- `--copy_policy` defaults to `copy`. Its case doesn't matter, unlike `--output_type`'s.
- `--max_image_size -1` means no cap.
- Left out: `--image_list_path`, an input the step doesn't have, so every registered image is
  undistorted. And `--num_patch_match_src_images` (20), which only goes into
  `stereo/patch-match.cfg`.

## image_undistorter_standalone

For warping `mask.png` the same way as the picks. The camera options must match the
`image_undistorter` call exactly.

```bash
colmap image_undistorter_standalone \
  --image_path "$FRAMES" \
  --input_file "$OUT/mask_camera.txt" \
  --output_path "$OUT/mask_undistorted" \
  --log_target stderr \
  --log_level 0 \
  --log_severity 0 \
  --log_color 0 \
  --default_random_seed 0 \
  --copy_policy COPY \
  --blank_pixels 0 \
  --min_scale 0.2 \
  --max_scale 2 \
  --max_image_size -1 \
  --roi_min_x 0 \
  --roi_min_y 0 \
  --roi_max_x 1 \
  --roi_max_y 1 \
  --jpeg_quality 100 \
  --num_threads -1
```

- `mask_camera.txt` holds one line, `mask.png` followed by the original camera from the model being
  undistorted, in `cameras.txt` order: `<MODEL> <WIDTH> <HEIGHT> <params...>`.
- `--jpeg_quality` doesn't apply to the PNG, but passing it keeps the two calls alike.

## Other commands the skill mentions

- `exhaustive_matcher` takes the same `--FeatureMatching.*`, `--SiftMatching.*` and
  `--TwoViewGeometry.*` options as `sequential_matcher`, with `--ExhaustiveMatching.block_size 50`
  in place of the `--SequentialMatching.*` ones.
- `database_cleaner` takes `--database_path` and `--type`, given once: `all`, `images`, `features`,
  `matches` or `two_view_geometries`.
- `model_analyzer` takes `--path` and `--verbose 0`.
- `model_converter` takes `--input_path`, `--output_path`, `--output_type` (`BIN`, `TXT`, `NVM`,
  `Bundler`, `VRML`, `PLY`, `R3D` or `CAM`) and `--skip_distortion 0`.
- `automatic_reconstructor` isn't used. In video mode it always turns on loop detection with the
  built-in vocabulary tree, which the image's build doesn't have.
