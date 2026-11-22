---
name: colmap-cli
description: COLMAP 4.2.1 command line on our CPU-only build (no CUDA, OpenGL or GUI), for feature_extractor, sequential_matcher, view_graph_calibrator, global_mapper, the incremental mapper and image_undistorter, with their main options and defaults, what they write and how they usually fail. Use when writing, running or debugging COLMAP commands for the camera solve, reading a COLMAP database or model folder, or choosing between the incremental and the global mapper.
metadata:
  sources: COLMAP docs (colmap.github.io, and doc/ at the 4.2.1 tag) and the COLMAP source at the 4.2.1 tag, read 2026
---

# COLMAP 4.2.1 command line

Notes on the COLMAP commands the camera solve needs: `feature_extractor`, `sequential_matcher`,
`view_graph_calibrator`, `global_mapper`, `mapper` and `image_undistorter`, for COLMAP 4.2.1 as
we build it. They come from the COLMAP docs and the source at the `4.2.1` tag on GitHub; links
are at the bottom. Defaults are the ones 4.2.1 uses in our build, read from the source.

colmap.github.io follows COLMAP's main branch, which has moved past 4.2.1. It already describes
things 4.2.1 doesn't have, such as `single_view_calibrator`, the `FLEXGLUE` matcher and the
`SKEWED_PINHOLE` camera model, and its `/legacy/` snapshots stop at 4.0. The 4.2.1 wording is in
`doc/*.rst` at the tag. On the machine, `colmap <command> -h` has the final say on option names
and defaults.

## Our build

COLMAP 4.2.1 from source with `-DCUDA_ENABLED=OFF -DOPENGL_ENABLED=OFF -DGUI_ENABLED=OFF
-DCGAL_ENABLED=OFF -DLSD_ENABLED=OFF` (tech.md). At run time that means:

- SiftGPU isn't compiled in. SIFT extraction and matching only run on the CPU: COLMAP's bundled
  VLFeat code extracts, and a faiss index matches (brute force with
  `--SiftMatching.cpu_brute_force_matcher 1`).
- `--FeatureExtraction.use_gpu` and `--FeatureMatching.use_gpu` default to `0` in this build. The
  help listings in the docs show `1` because they come from a GPU build. Pass `0` on every
  extractor and matcher call anyway, so the command means the same on any build.
- A `1` stops the command before it starts, with `Cannot use GPU feature extraction without CUDA
  or OpenGL support. Consider setting use_gpu to false.` (or the feature matching version), then
  `Invalid options provided.`.
- Never turn GPU SIFT on, and never suggest a build that has it: SiftGPU is non-commercial.
- Bundle adjustment is Ceres on the CPU. `--Mapper.ba_use_gpu`, `--BundleAdjustmentCeres.use_gpu`
  and the `CASPAR` backend need CUDA, so leave them off. The global mapper asks for GPU solvers
  by default; its section says what to pass.
- tech.md doesn't set `ONNX_ENABLED` or `DOWNLOAD_ENABLED`, which default to ON. So the binary
  can fetch learned-model weights and vocabulary trees when an option asks for one. SIFT with
  the default `SIFT_BRUTEFORCE` matcher needs no download, and it's all we use.
- To check a binary, `colmap version` prints `COLMAP 4.2.1 (Commit ... without GPU support, ...)`.
  In the logs, `Creating SIFT CPU feature extractor` and `Creating SIFT CPU feature matcher`
  confirm the CPU path.

## Every command

- Options are `--Group.name value`, and booleans take `0` or `1` (`true` and `false` work too).
  `colmap <command> -h` lists every option with this build's defaults.
- An unknown option stops the command with `Failed to parse options - unrecognised option ...`.
  Guides for COLMAP 3.x use names 4.2.1 has dropped, like `--SiftExtraction.use_gpu`,
  `--SiftExtraction.num_threads`, `--SiftExtraction.max_image_size`, `--SiftMatching.use_gpu`,
  `--SiftMatching.guided_matching` and `--SiftMatching.max_num_matches`. In 4.2.1 those live
  under `--FeatureExtraction.*` and `--FeatureMatching.*`, and the RANSAC settings under
  `--TwoViewGeometry.*`. One line of 4.2.1's own FAQ still says `--SiftExtraction.max_image_size`.
- Paths are checked before anything runs: `--image_path` must be an existing folder, and
  `--database_path` must not be a folder and must sit in one that exists. A failed check prints
  the check, then `Invalid options provided.`, and exits 1.
- The database is opened with create, so a mistyped database file name quietly gives a new,
  empty database. The matchers then find nothing to match, and the mappers fail later with
  `No images with matches` or `Cannot continue with empty pose graph`.
- Only one COLMAP process may have a database open; the source says not to open it from two at
  once. It runs in SQLite's WAL mode, so `database.db-wal` and `database.db-shm` sit beside it
  while it's open. Copy or read it only after the command that wrote it has exited.
- In 4.2.1, `--project_path file.ini` takes options from the file only and silently ignores
  every other flag on the line. COLMAP's main branch has since made flags win, and the live docs
  describe that. Pass flags and leave `--project_path` out. The mappers still write a
  `project.ini` that records the options they ran with.
- Most commands exit 0 even when they skipped images, so check what they wrote (see "Checking a
  run").
- `feature_extractor`, the matchers, `mapper` and `image_undistorter` treat the first SIGINT or
  SIGTERM as a request to stop: they finish at a safe point, keep what they have, and exit 130 or
  143. A second signal kills them. `view_graph_calibrator` and `global_mapper` have no such
  handling, so a signal kills them at once: the global mapper leaves no model, and the
  calibrator can leave its database half-changed.
- A failed internal check ends the process with an abort, not a clean exit code, after printing
  what failed.
- glog writes to stderr and, by default, also to log files in `$GOOGLE_LOG_DIR` or `/tmp`.
  `--log_target stderr` keeps it to stderr.
- The random parts (RANSAC, the mappers) aren't repeatable by default, with their seed at -1.
  Setting `--TwoViewGeometry.random_seed`, `--Mapper.random_seed` or `--GlobalMapper.random_seed`
  to 0 or more makes them repeatable. Multi-threaded runs can still differ a little; COLMAP's own
  determinism tests run single-threaded.

## How the commands fit together

This is how they fit the frames step's output. `$FRAMES` is `work/frames/`, which holds
`images/`, `grey/`, `mask.png` and `frames.json`. `$OUT` stands for the camera solve's own work
folder, which its spec will name. Every setting not shown is COLMAP's default; the step's own
values belong in `config.toml`.

```bash
# 1. SIFT on the grey copies, one shared camera, the stuck-pixel mask when frames.json has mask_used
colmap feature_extractor \
  --database_path "$OUT/database.db" \
  --image_path "$FRAMES/grey" \
  --ImageReader.single_camera 1 \
  --ImageReader.camera_model SIMPLE_RADIAL \
  --ImageReader.camera_mask_path "$FRAMES/mask.png" \
  --FeatureExtraction.use_gpu 0

# 2. Match each pick with the picks after it, in name order
colmap sequential_matcher \
  --database_path "$OUT/database.db" \
  --FeatureMatching.use_gpu 0

# 3a. Incremental mapper
mkdir -p "$OUT/sparse"
colmap mapper \
  --database_path "$OUT/database.db" \
  --image_path "$FRAMES/images" \
  --output_path "$OUT/sparse"

# 3b. Or the global mapper, on a copy of the database
cp "$OUT/database.db" "$OUT/database_global.db"
colmap view_graph_calibrator --database_path "$OUT/database_global.db"
mkdir -p "$OUT/sparse_global"
colmap global_mapper \
  --database_path "$OUT/database_global.db" \
  --image_path "$FRAMES/images" \
  --output_path "$OUT/sparse_global" \
  --GlobalMapper.gp_use_gpu 0 \
  --GlobalMapper.ba_ceres_use_gpu 0

# 4. Undistort the colour picks of the chosen model for training
colmap image_undistorter \
  --image_path "$FRAMES/images" \
  --input_path "$OUT/sparse/0" \
  --output_path "$OUT/undistorted" \
  --output_type COLMAP
```

- COLMAP knows an image by its path relative to `--image_path`, so `grey/000000.jpg` and
  `images/000000.jpg` are both `000000.jpg`. That's what lets extraction read the grey copies
  while the mapper and the undistorter read the colour picks, which share their names and size.
  The mapper only reads images for point colours, and training can start the splat from those
  points, so give it `images/`.
- Leave `--ImageReader.camera_mask_path` out when `frames.json` has no mask. A missing mask file
  doesn't make COLMAP fail (see below).
- 3a or 3b: the FAQ calls the incremental mapper the most reliable and best tested. It calls the
  global mapper faster on large sets with a good match graph, but less tolerant of bad matches
  and dependent on good focal lengths, hence `view_graph_calibrator` for our EXIF-less frames.

## feature_extractor

Finds SIFT keypoints and descriptors in every image under `--image_path` and stores them in the
database, along with the cameras.

| Option | Default | What it does |
|---|---|---|
| `--database_path` | required | SQLite file, created if missing. |
| `--image_path` | required | Read recursively. Every readable image below it counts. |
| `--image_list_path` | none | Text file of image names to read instead of all of them. |
| `--ImageReader.single_camera` | 0 | 1 gives every image the same camera, so one set of intrinsics. |
| `--ImageReader.camera_model` | `SIMPLE_RADIAL` | Starting camera model. See the cameras page. |
| `--ImageReader.camera_params` | empty | Known intrinsics, comma-separated in the model's order. Marks the focal length as trusted. |
| `--ImageReader.default_focal_length_factor` | 1.2 | Without EXIF or `camera_params`, the focal starts at this times the longer side, and isn't trusted. |
| `--ImageReader.camera_mask_path` | empty | One mask for all images. Keypoints on 0 pixels are dropped. |
| `--ImageReader.mask_path` | empty | Folder of per-image masks named `<image name>.png`. |
| `--camera_mode` | -1 | Shorthand for the camera flags: 0 auto, 1 single, 2 per folder, 3 per image. -1 leaves them alone. |
| `--FeatureExtraction.use_gpu` | 0 | Always pass 0. |
| `--FeatureExtraction.num_threads` | -1 | -1 uses every core, one SIFT extractor per thread. |
| `--FeatureExtraction.max_image_size` | -1 | Longer images are scaled down before SIFT. -1 means 3200 for SIFT. |
| `--FeatureExtraction.type` | `SIFT` | Keep SIFT. |
| `--SiftExtraction.max_num_features` | 8192 | Per image. Keeps the larger-scale ones. |
| `--SiftExtraction.first_octave` | -1 | -1 doubles the image first: more small features, more RAM. |
| `--SiftExtraction.peak_threshold` | 0.0067 (0.02 / 3) | Lower finds more features in low-contrast images. |
| `--SiftExtraction.estimate_affine_shape`, `--SiftExtraction.domain_size_pooling` | 0, 0 | DSP-SIFT with affine shapes: more matches, slower, through a separate CPU extractor. |

All its output goes into the database: `cameras`, `rigs` and `frames` (one trivial rig per
camera, one frame per image), `images` (with names relative to `--image_path`), `keypoints`,
`descriptors`, and `pose_priors` for EXIF GPS or orientation. Our picks have no EXIF, because
the frames step encodes them with OpenCV.

How it fails:

- **Every frame gets its own camera.** In the default camera mode, a camera is recognised by its
  EXIF make, model, focal length and image size, and an image without them gets a new camera.
  Our picks have no EXIF, so without `--ImageReader.single_camera 1` each pick gets separate
  intrinsics. The log's `Camera: #N` line counts up and `cameras` gets one row per image. A clip
  that switches lens or zoom breaks the one-camera assumption too.
- **Images skipped with exit 0.** An unreadable file, or a size that doesn't fit the shared
  camera, is logged as a warning with a status such as `BITMAP_ERROR` or
  `CAMERA_SINGLE_DIM_ERROR`, and the run goes on. Compare the `images` count with the number of
  picks.
- **Silent mask problems.** A missing or unreadable mask file gives one error line,
  `Mask at ... does not exist.` or `Failed to read invalid mask file at: ... No mask is going to
  be used.`, and extraction runs without a mask. The mask is read at each keypoint's pixel, with
  no scaling: a 0 pixel drops the keypoint, and so does a position outside a mask that's too
  small. So the mask must be exactly the image size. Any non-zero value keeps a keypoint.
- **A rerun changes nothing.** Images that already have features in the database are skipped,
  so new options don't apply to them. Start from a new database.
- **Too many images.** `--image_path work/frames` would take the colour picks, the grey copies
  and `mask.png` as images. Point it at `grey/` itself.
- **RAM.** With all threads and the default size and octave, it warns that CPU SIFT can use a
  lot of RAM per thread. That warning is expected. If memory does run out, lower
  `--FeatureExtraction.num_threads` or `--FeatureExtraction.max_image_size`, or set
  `--SiftExtraction.first_octave 0`.
- **Few features.** Each image logs `Features: N (SIFT)`. Blur and plain surfaces give few. The
  FAQ's way to more matches is DSP-SIFT with affine shapes, plus guided matching in the next
  step.

## sequential_matcher

Matches each image with the ones after it in name order, then checks the geometry of each pair.
Made for video.

Images are sorted by name as text, whatever their order in the database. With
`quadratic_overlap` on, the default, image i is paired with i+1, i+2, i+4 and so on up to
i+2^(overlap-1), so the default 10 pairs reach 512 images ahead. With it off, i is paired with
i+1 to i+overlap. Our zero-padded six-digit names sort in time order.

| Option | Default | What it does |
|---|---|---|
| `--database_path` | required | |
| `--SequentialMatching.overlap` | 10 | Pairs per image. |
| `--SequentialMatching.quadratic_overlap` | 1 | Spread the pairs at 1, 2, 4, … images ahead. |
| `--SequentialMatching.loop_detection` | 0 | Also match every `loop_detection_period` (10)th image with its `loop_detection_num_images` (50) most similar images, found with a vocabulary tree. |
| `--SequentialMatching.vocab_tree_path` | empty | Empty means COLMAP's default SIFT tree, downloaded at run time. |
| `--FeatureMatching.use_gpu` | 0 | Always pass 0. |
| `--FeatureMatching.num_threads` | -1 | All cores. |
| `--FeatureMatching.guided_matching` | 0 | A second pass guided by the pair's geometry: more matches, slower. |
| `--SiftMatching.max_ratio` | 0.8 | Ratio test between the best and second-best match. |
| `--SiftMatching.cross_check` | 1 | Keep only matches that agree both ways. |
| `--TwoViewGeometry.min_num_inliers` | 15 | Fewer marks the pair degenerate. |
| `--TwoViewGeometry.max_error` | 4 | RANSAC inlier threshold, in pixels. |

It writes rows to `matches` (raw descriptor matches) and `two_view_geometries` (the inliers, the
fitted model and a `config` code) for each pair. Only `two_view_geometries` feeds the mappers.
The `config` codes are 0 undefined, 1 degenerate (rejected), 2 calibrated (essential matrix),
3 uncalibrated (fundamental matrix), 4 planar, 5 panoramic (pure rotation), 6 planar or
panoramic, 7 watermark, 8 multiple and 9 calibrated rig.

How it fails:

- **A rerun changes nothing.** Pairs that already have both rows are skipped, so new matching
  options don't apply to them. Start from a new database, or run
  `colmap database_cleaner --database_path <db> --type matches`, which clears the matches and
  the two-view geometries. `--type` takes one value, although the features page's example
  passes it twice.
- **The end of a loop never meets its start.** Without loop detection, a walk that comes back to
  where it began is only matched to its neighbours, and the drift stays in the model. Loop
  detection needs a vocabulary tree. By default COLMAP downloads one from its GitHub releases at
  run time and caches it in `~/.cache/colmap`, which fails with no network. The tree is a data
  file, so our licence rules apply: check its licence with Samarth, add its line, put it on the
  image and pass it with `--SequentialMatching.vocab_tree_path`. The other way is
  `exhaustive_matcher` on the same database afterwards. It skips the pairs that are already
  matched, and the tutorial says exhaustive matching suits sets of up to a few hundred images.
  400 picks make about 80,000 pairs, though, so time it on the CPU before relying on it.
- **Pure rotation.** Standing in one spot and turning gives pairs with no baseline, stored as 5
  or 6, which give the mappers no parallax to triangulate from. The tutorial's capture advice is
  to move between views.
- **Fast motion.** Picks that are far apart drop under 15 inliers and become degenerate. A
  larger `--SequentialMatching.overlap` adds pairs.

## view_graph_calibrator

Estimates focal lengths from the fundamental matrices of the matched pairs and writes them back
to the database, with refreshed two-view geometries. Run it after matching and before
`global_mapper` when the focal lengths aren't trusted, which is our case. It changes the
database in place and takes no `--image_path`.

Its options have no group prefix.

| Option | Default | What it does |
|---|---|---|
| `--database_path` | required | The copy to change. |
| `--cross_validate_prior_focal_lengths`, `--min_calibrated_pair_ratio` | 1, 0.5 | For cameras with trusted focals, upgrade their uncalibrated pairs when more than this share of their pairs is already calibrated. |
| `--min_focal_length_ratio`, `--max_focal_length_ratio` | 0.1, 10 | An estimate outside this range of the starting focal is rejected. |
| `--max_calibration_error` | 2 | Pairs with a larger error are marked degenerate. |
| `--reestimate_relative_pose` | 1 | Refit each kept pair's relative pose with the new focals. |
| `--relpose_max_error`, `--relpose_min_num_inliers`, `--relpose_min_inlier_ratio` | 1, 30, 0.25 | Limits for that refit. |

What it changes:

- Each pinhole-type camera without a trusted focal gets its focal from one fit over all its
  pairs. Fisheye models are skipped. Cameras with a trusted focal, from EXIF or
  `--ImageReader.camera_params`, are held fixed.
- Then every pinhole-type camera is marked as trusted, even one whose estimate it rejected.
- Pairs over `max_calibration_error` become degenerate (1). The rest are refitted as calibrated
  pairs with the new focals, from the raw matches and under the stricter `relpose_*` limits,
  which can reject more of them.
- The log shows `Calibrating view graph with N pairs`, `K cameras rejected in view graph
  calibration`, `Invalid / total number of two-view geometry: X / N` and `View graph calibration
  completed successfully`.

How it fails:

- **No undo.** Copy the database first, once the matcher has exited, and keep the original for
  the incremental mapper; the docs say to work on a copy. A kill midway can leave the copy
  half-changed, so start again from the original.
- **Success that did nothing.** `No image pairs to calibrate` still exits 0: matching found no
  usable pairs. `No cameras to optimize` means every camera was already trusted, for example on
  a second run over the same copy.
- **A rejected camera.** With our one camera, `1 cameras rejected in view graph calibration`
  means the estimate fell outside 0.1 to 10 times the starting focal. The focal stays at the
  guess, 1.2 times the longer side, but is now marked as trusted, so the global mapper relies on
  it and no longer warns about missing focal lengths. Treat it as a failed calibration.
- **Hard failure.** `Ceres solver failed`, then `View graph calibration failed`, exit 1.
- **A split graph.** When most pairs turn degenerate, the view graph can fall apart, and the
  global mapper returns several small models.

## global_mapper

Global SfM, from GLOMAP, which is now part of COLMAP: rotation averaging over the whole view
graph, then tracks, global positioning of cameras and points, bundle adjustment and
retriangulation, for all images at once. Since 4.2.0 it reconstructs each connected part of the
view graph separately by default.

| Option | Default | What it does |
|---|---|---|
| `--database_path` | required | The calibrated copy. |
| `--image_path` | required | Only read for point colours. |
| `--output_path` | required | Must exist. |
| `--GlobalMapper.multiple_models` | 1 | One model per connected component. 0 keeps the largest only, the behaviour before 4.2.0. |
| `--GlobalMapper.min_model_size` | 3 | Fewer registered frames and the model is dropped. |
| `--GlobalMapper.min_num_matches` | 15 | Pairs with fewer inliers are left out. |
| `--GlobalMapper.ra_max_rotation_error_deg` | 10 | Pairs that disagree with the averaged rotations by more are cut, which can split the graph. |
| `--GlobalMapper.ba_num_iterations` | 3 | Bundle adjustment rounds. |
| `--GlobalMapper.ba_refine_focal_length`, `ba_refine_extra_params`, `ba_refine_principal_point` | 1, 1, 0 | What bundle adjustment may change in the cameras. |
| `--GlobalMapper.gp_use_gpu`, `--GlobalMapper.ba_ceres_use_gpu` | 1, 1 | Pass 0 on our build. |
| `--GlobalMapper.num_threads` | -1 | All cores. |
| `--GlobalMapper.random_seed` | -1 | -1 isn't repeatable. 0 or more is. |

```text
sparse_global/
  project.ini                  the options it ran with
  0/                           the model with the most 3D points
    cameras.bin images.bin points3D.bin rigs.bin frames.bin
  1/ ...                       more components, with multiple_models 1
```

The log names the components: `Found N connected component(s) after rotation filtering`, then
`Reconstructing component i / N with M images` for each, and finally `Kept K reconstruction(s),
discarded X with fewer than 3 registered frames, and failed to reconstruct Y`.

How it fails:

- **No output folder.** `` `output_path` is not a directory. ``, exit 1. Make it first.
- **No trusted focals.** `Less than 50% of cameras have prior focal lengths...` is logged before
  and after the run. It always fires on our frames unless `view_graph_calibrator` ran first or
  the intrinsics were given.
- **Several models.** A weakly connected clip comes back as `0/`, `1/` and so on. Check the
  counts before taking `0/`.
- **Nothing to map.** `Cannot continue with empty pose graph`, or every component failing with
  `Global mapping failed` or coming out too small, ends in `Failed to create sparse model` and
  exit 1.
- **GPU warnings.** With the defaults it logs `Requested to use GPU for bundle adjustment, but
  COLMAP was compiled without CUDA support. Falling back to CPU-based solvers.` It does run on
  the CPU, so the warning is harmless, and the two `use_gpu 0` flags stop the request.
- **Killed, nothing saved.** A signal or a time cap that ends it leaves no model, because the
  global mapper can't resume. Give it room inside the job's time limit.
- **Bad matches hurt more.** The docs say it copes worse with outlier matches than the
  incremental mapper.

## mapper (incremental)

Incremental SfM: it starts from one well-matched pair, then adds one image at a time and
triangulates new points, with local bundle adjustment after each image and global bundle
adjustment as the model grows. COLMAP's default and most tested mapper.

| Option | Default | What it does |
|---|---|---|
| `--database_path` | required | The original, uncalibrated database. |
| `--image_path` | required | Only read for point colours. |
| `--output_path` | required | Must exist. |
| `--input_path` | empty | A model to continue from. The result goes straight into `--output_path`. |
| `--Mapper.multiple_models` | 1 | Start a new model from images the first couldn't take. |
| `--Mapper.min_model_size` | 10 | Later models with fewer registered images are dropped; for small sets the limit is half the images. The first model is always kept. |
| `--Mapper.min_num_matches` | 15 | Pairs with fewer inliers are left out. |
| `--Mapper.init_min_num_inliers` | 100 | For the starting pair. |
| `--Mapper.init_min_tri_angle` | 16 | Degrees of parallax the starting pair needs. |
| `--Mapper.init_image_id1`, `--Mapper.init_image_id2` | -1 | Name the starting pair by image id. |
| `--Mapper.abs_pose_min_num_inliers` | 30 | To register the next image. |
| `--Mapper.ba_refine_focal_length`, `ba_refine_extra_params`, `ba_refine_principal_point` | 1, 1, 0 | What bundle adjustment may change in the cameras. |
| `--Mapper.ba_global_frames_ratio`, `--Mapper.ba_global_points_ratio` | 1.1, 1.1 | Growth that triggers a global bundle adjustment. |
| `--Mapper.max_extra_param` | 1 | Images whose distortion goes past this are filtered as bogus. |
| `--Mapper.max_runtime_seconds` | -1 | Stop after this long and keep what's built. |
| `--Mapper.num_threads` | -1 | All cores. |
| `--Mapper.random_seed` | -1 | -1 isn't repeatable. 0 or more is. |
| `--Mapper.ba_use_gpu` | 0 | Leave 0. |

```text
sparse/
  0/   cameras.bin images.bin points3D.bin rigs.bin frames.bin project.ini   the first model built
  1/   ...                                                                    later models
```

Each model is written to the next number when it's finished, in the order they were built, not
by size. `0/` is usually the biggest but not always, so choose by registered image count. Models
are binary; see "Checking a run" for text.

`automatic_reconstructor --data_type video` tunes the mapper for video. It halves
`init_min_tri_angle` to 8, raises both `ba_global_*_ratio` options to 1.4 and lifts the
`max_extra_param` limit. `mapper` on its own applies no preset, so pass those flags to get the
same.

How it fails:

- **No output folder.** `` `output_path` is not a directory. ``, exit 1.
- **Nothing to map.** `No images with matches`, then `Failed to create any sparse model`,
  exit 1.
- **No starting pair.** After `=> No good initial image pair found.` it retries up to four
  times, logging `=> Relaxing the initialization constraints.` each time. It halves the inlier
  count and the angle in turn, down to 25 inliers and 4 degrees. If nothing works there's no
  model. The usual cause is too little parallax (turning on the spot, a camera that hardly
  moves), little texture or few matches. Lower `--Mapper.init_min_tri_angle` or
  `--Mapper.init_min_num_inliers`, or name the pair.
- **It stops growing.** Repeated `=> Could not register, trying another image.` means the rest
  doesn't connect. The leftover images may start `1/`, and later models below `min_model_size`
  are dropped with `Discarding reconstruction due to insufficient size`.
- **Grey or black points.** Points take their colours from `--image_path`. Given `grey/` they
  come out grey. Given a folder without the database's names they stay black, with
  `Could not read image ... at path ...` per image.
- **A partial model with exit 0.** `--Mapper.max_runtime_seconds` stops it with `Reached maximum
  runtime of N seconds.` and keeps what's built. Check the registered image count.
- **Interrupted.** On SIGINT or SIGTERM it keeps and writes the model in progress and exits 130
  or 143. Continue it with `--input_path sparse/0 --output_path sparse/0`.
- **The calibrated copy.** The database `view_graph_calibrator` changed has its focal marked as
  trusted and fewer usable pairs. Give the incremental mapper the original unless you mean to.

## image_undistorter

Warps one model's images to distortion-free pinhole cameras and writes them with a matching
model, in the folder layout COLMAP's dense tools use.

| Option | Default | What it does |
|---|---|---|
| `--image_path` | required | The images to warp: the colour picks. |
| `--input_path` | required | One model folder, such as `sparse/0`. |
| `--output_path` | required | Created if missing, but not its parents. |
| `--output_type` | `COLMAP` | `COLMAP`, `PMVS` or `CMP-MVS`, case-sensitive. |
| `--max_image_size` | -1 | Cap on the longer side of the output. |
| `--blank_pixels` | 0 | 0 crops to pixels the original covers. 1 keeps the whole view, with black edges. |
| `--min_scale`, `--max_scale` | 0.2, 2 | Limits on the rescale that fitting `blank_pixels` needs. |
| `--roi_min_x`, `--roi_min_y`, `--roi_max_x`, `--roi_max_y` | 0, 0, 1, 1 | Crop, as fractions of the original. |
| `--jpeg_quality` | -1 | 1 to 100 for JPEG output. -1 writes at 100. |
| `--copy_policy` | `COPY` | `COPY`, `SOFT_LINK` or `HARD_LINK`, for images that need no warp. |
| `--num_threads` | -1 | All cores. |

```text
undistorted/
  images/000000.jpg ...                 warped picks, same names
  sparse/                               the model with PINHOLE cameras, no 0/ level
    cameras.bin images.bin points3D.bin rigs.bin frames.bin
  stereo/                               dense scaffolding: empty depth_maps/, normal_maps/ and
                                        consistency_graphs/, plus patch-match.cfg and fusion.cfg
  run-colmap-geometric.sh
  run-colmap-photometric.sh             dense scripts; patch_match_stereo needs CUDA, so ignore them
```

- Every camera comes out `PINHOLE` (fx, fy, cx, cy). With `blank_pixels` at 0 the image is
  cropped and rescaled so no pixel is empty, so its size and principal point usually change.
- gsplat 1.5.3's example COLMAP parser looks for `<dir>/sparse/0`, falls back to
  `<dir>/sparse`, and reads `<dir>/images`, so this folder works as its data folder as it is.
- Images whose camera has no distortion left (every distortion parameter zero, as with
  `PINHOLE`) are copied or linked rather than re-encoded, unless `--max_image_size` is set.

How it fails:

- **Lower-case type.** `--output_type colmap` fails with ``Invalid `output_type` ...``, exit 1.
- **The wrong model level.** `--input_path sparse` instead of `sparse/0` aborts with
  `rigs, cameras, frames, images, points3D files do not exist at ...`.
- **A missing parent folder** aborts in a check.
- **Missing images with exit 0.** `Cannot read image at path: ...` leaves that image out of
  `images/`, but `sparse/` still lists it. Compare the file count with the registered images.
- **Interrupted.** A signal stops it with `Stopped image undistortion before writing the sparse
  model and stereo configuration files.`, so there's no `sparse/`. Rerun it, ideally into an
  empty folder.
- **The mask no longer lines up.** `mask.png` is at the pick size, and undistortion moves and
  resizes everything. `image_undistorter_standalone` warps any image given a camera line in
  `--input_file` (`<name> <MODEL> <WIDTH> <HEIGHT> <params...>`, the fields of `cameras.txt`),
  with the same warp. Give it the camera from the model you undistorted and the same
  `--blank_pixels`, `--min_scale`, `--max_scale` and `--max_image_size`. It reads the mask as
  RGB and samples bilinearly, so expect a three-channel PNG with grey edges, and threshold it
  back to 0 and 255.

## Checking a run

`colmap model_analyzer --path <model>` prints `Registered images`, `Points`, `Mean track length`
and `Mean reprojection error`, among others.

For text files, convert into a folder that already exists:

```bash
mkdir -p "$OUT/sparse_txt"
colmap model_converter --input_path "$OUT/sparse/0" --output_path "$OUT/sparse_txt" --output_type TXT
```

`images.txt` has two lines per image. The first holds the image id, the world-to-camera rotation
as a quaternion (QW QX QY QZ, Hamilton convention), the translation, the camera id and the name.
The second lists its 2D points. The camera centre is −Rᵀt, and the camera looks down +z with x
to the right and y down. `cameras.txt` has one line per camera: id, model, width, height, then
the parameters.

The database can be read with Python's built-in `sqlite3`, once no COLMAP command has it open:

```python
import sqlite3

con = sqlite3.connect("database.db")
print(con.execute("SELECT COUNT(*) FROM images").fetchone())            # picks read
print(con.execute("SELECT COUNT(*) FROM cameras").fetchone())           # 1 with single_camera
print(con.execute("SELECT MIN(rows), AVG(rows) FROM keypoints").fetchone())
print(con.execute("SELECT config, COUNT(*) FROM two_view_geometries GROUP BY config").fetchall())
con.close()
```

In `two_view_geometries`, `rows` is the inlier count. `cameras.params` is a blob of float64 in
the model's parameter order, and `cameras.prior_focal_length` is the trusted-focal flag.

## Our rules that touch this

- COLMAP is a command line tool on our private machine image, built with the tech.md flags.
  Don't vendor it, don't link it into our code, and never make the image public.
- Always pass `--FeatureExtraction.use_gpu 0` and `--FeatureMatching.use_gpu 0`. Never GPU SIFT.
- Anything COLMAP would download, like a vocabulary tree or learned model weights, needs
  Samarth's OK and a line in `docs/third-party-licences.md` first. So would pycolmap, if our
  code used it to read models. Reading the text files or the database with the standard library
  adds nothing.
- Databases, models and undistorted images are job files under `data/`, never in git.
- The camera solve's COLMAP numbers (overlap, thresholds, seeds, time limits) belong in its
  `config.toml` section, not in code.
- Tests that need COLMAP skip when it isn't installed, and the laptop only runs small CPU tests.

## Links

COLMAP docs, which follow the main branch:

- Command-line interface: <https://colmap.github.io/cli.html>
- Tutorial (capture advice, matching modes, reconstruction): <https://colmap.github.io/tutorial.html>
- FAQ (masks, intrinsics, CPU-only use, incremental against global): <https://colmap.github.io/faq.html>
- Feature extraction and matching types: <https://colmap.github.io/features.html>
- Camera models: <https://colmap.github.io/cameras.html>
- Output format: <https://colmap.github.io/format.html>
- Database format: <https://colmap.github.io/database.html>
- Installation and build options: <https://colmap.github.io/install.html>
- Changelog (the 4.2.0 global mapper change, the 4.2.1 fixes): <https://colmap.github.io/changelog.html>

The 4.2.1 tag:

- Docs source, the text for our version: <https://github.com/colmap/colmap/tree/4.2.1/doc>
- Where every option is registered: <https://github.com/colmap/colmap/blob/4.2.1/src/colmap/controllers/option_manager.cc>
- The commands: <https://github.com/colmap/colmap/blob/4.2.1/src/colmap/exe/feature.cc>,
  <https://github.com/colmap/colmap/blob/4.2.1/src/colmap/exe/sfm.cc> and
  <https://github.com/colmap/colmap/blob/4.2.1/src/colmap/exe/image.cc>
- SIFT defaults: <https://github.com/colmap/colmap/blob/4.2.1/src/colmap/feature/sift.h>
- Sequential pairing: <https://github.com/colmap/colmap/blob/4.2.1/src/colmap/controllers/pairing.h>
- Two-view geometry: <https://github.com/colmap/colmap/blob/4.2.1/src/colmap/estimators/two_view_geometry.h>
- Incremental mapper: <https://github.com/colmap/colmap/blob/4.2.1/src/colmap/controllers/incremental_pipeline.h>
  and <https://github.com/colmap/colmap/blob/4.2.1/src/colmap/sfm/incremental_mapper.h>
- Global mapper: <https://github.com/colmap/colmap/blob/4.2.1/src/colmap/controllers/global_pipeline.h>
  and <https://github.com/colmap/colmap/blob/4.2.1/src/colmap/sfm/global_mapper.h>
- View graph calibration: <https://github.com/colmap/colmap/blob/4.2.1/src/colmap/estimators/view_graph_calibration.h>
- Undistortion: <https://github.com/colmap/colmap/blob/4.2.1/src/colmap/image/undistortion.h>
  and <https://github.com/colmap/colmap/blob/4.2.1/src/colmap/controllers/undistorters.cc>

gsplat 1.5.3's COLMAP parser, for the undistorted folder layout:
<https://github.com/nerfstudio-project/gsplat/blob/v1.5.3/examples/datasets/colmap.py>
