---
name: gsplat-training
description: Training a splat with gsplat 1.5.3. What rasterization() takes and gives back, the MCMC strategy and its settings, and how the example trainer (examples/simple_trainer.py) is put together, from its COLMAP loader to the PLY it writes. Use when writing or reviewing the training step, calling rasterization() or a densification strategy, choosing MCMCStrategy settings, adapting code from gsplat's examples, installing gsplat on the machine image, or reading a PLY that gsplat exported.
metadata:
  sources: the v1.5.3 tag of gsplat on GitHub (library source, examples/ and README), the 1.5.3 docs at docs.gsplat.studio/versions/1.5.3, gsplat's 1.5.3 files on PyPI, and the torch 2.7.1 extension builder, fused-ssim, torchmetrics and imageio-ffmpeg pages the trainer depends on, read November 2026
---

# Training with gsplat 1.5.3

Notes on three parts of gsplat 1.5.3: the `rasterization()` function, the MCMC densification strategy, and the example trainer, `examples/simple_trainer.py`. They come from the `v1.5.3` tag of the gsplat repo and the 1.5.3 docs, plus the pages of torch, fused-ssim, torchmetrics and imageio-ffmpeg that the trainer depends on, written in our own words. The block of adapted example code near the end carries the example's licence note. Where a short expression in backticks elsewhere is gsplat's own code, it's from the same Apache-2.0 source, Copyright 2025 Nerfstudio Team. Links are at the bottom.

A few of gsplat's functions and one example helper wrap or copy code under Inria's non-commercial licence. "Our rules that touch this" names them.

Read the docs pages under `versions/1.5.3/`, not `main/`. The `main` pages describe a newer `rasterization()` with more arguments and render modes than 1.5.3 has.

gsplat, torch and fused-ssim only exist on the machine image (tech.md), so nothing here runs on a laptop.

## Installing and compiling

- PyPI has 1.5.3 as an sdist and as a pure-Python wheel, `gsplat-1.5.3-py3-none-any.whl`, and pip picks the wheel. gsplat's publish workflow builds both with `BUILD_NO_CUDA=1 python -m build`, so the wheel holds no compiled CUDA module. The README says the same: an install from PyPI builds the CUDA code on the first run.
- With the wheel, the CUDA code compiles the first time gsplat needs it, not at install. `gsplat/cuda/_backend.py` tries the prebuilt module `gsplat.csrc` first. When that's missing, it compiles with torch's `cpp_extension` into `~/.cache/torch_extensions/py<python>_cu<cuda>/gsplat_cuda/` (under `TORCH_EXTENSIONS_DIR` instead, when that's set) and says this "may take a few minutes the first time". The cache is per user, and later runs load it.
- That compile needs `nvcc`. With no CUDA toolkit, gsplat prints "No CUDA toolkit found. gsplat will be disabled." and every CUDA call fails.
- Installing from source compiles the module during the install, through `setup.py`. That's the sdist, with `--no-binary gsplat`, or the README's `git+https://github.com/nerfstudio-project/gsplat.git` pinned to `@v1.5.3`. Both carry the CUDA sources and the glm headers, which the repo keeps as a git submodule. `setup.py` imports torch and the repo has no `pyproject.toml`, so torch goes in first and pip needs `--no-build-isolation`. `TORCH_CUDA_ARCH_LIST` picks the GPU architectures, `MAX_JOBS` defaults to 10, and `BUILD_NO_CUDA=1` skips the compile. On this path torch 2.7.1 also refuses an `nvcc` whose CUDA major version differs from the one torch was built with.
- gsplat also publishes prebuilt wheels for some torch and CUDA pairs at `https://docs.gsplat.studio/whl` (README).
- So `TORCH_CUDA_ARCH_LIST=... pip install gsplat==1.5.3` compiles nothing. For nothing to compile when a job starts, install from source on the image, so that `gsplat.csrc` exists.
- Our image does that (`cloud/image.py`). Its base image carries only CUDA 13.2 to 13.5 and no 12.x, so it installs torch 2.7.1 and torchvision 0.22.1 from PyTorch's cu128 index, NVIDIA's `cuda-toolkit-12-8` apt package for an nvcc that matches, and setuptools, wheel and ninja. Then it runs `uv pip install --no-build-isolation --no-binary gsplat gsplat==1.5.3` with `CUDA_HOME=/usr/local/cuda-12.8` and `TORCH_CUDA_ARCH_LIST="8.6;8.9"`, and the build fails unless `from gsplat import csrc` imports. That recipe worked on two real machines on that base image.
- Warming the compile cache during the image build is the weaker option. In each new process, torch 2.7.1 regenerates its ninja file and runs ninja over the cached build. If the compile flags differ, ninja compiles the module again. Without `TORCH_CUDA_ARCH_LIST`, the flags follow the machine's own GPU, so the cache only holds when every job runs as the same user with the same `TORCH_CUDA_ARCH_LIST` as the build. `_backend.py`'s docstring warms it with `python -c "from gsplat.cuda._backend import _C"`, but also sets `FAST_COMPILE=1`, which compiles with `-O0` for debugging, so leave that out.

## rasterization()

`from gsplat import rasterization` renders N Gaussians into C camera views in one call. The inputs are torch tensors on the GPU. Every shape may also have leading batch dimensions, written `[..., N, 3]` in the docs, to render several scenes at once. One scene needs none, so the tables leave them out.

### What it takes

The Gaussians:

| Argument | Shape | Holds |
| --- | --- | --- |
| `means` | `[N, 3]` | centres, in world space |
| `quats` | `[N, 4]` | rotations as `(w, x, y, z)`. gsplat normalises them, so they needn't be unit length. |
| `scales` | `[N, 3]` | size along each axis, so positive. The trainer passes `exp()` of the log scales it stores. |
| `opacities` | `[N]` | from 0 to 1. The trainer passes `sigmoid()` of the logits it stores. |
| `colors` | `[N, D]` or `[C, N, D]`, or with `sh_degree` set, `[N, K, 3]` or `[C, N, K, 3]` | see "Colours" below. The forms with `C` give each view its own values. |
| `covars` | `[N, 3, 3]`, optional | full covariances. When given, `quats` and `scales` are ignored. |

The cameras:

| Argument | Shape | Holds |
| --- | --- | --- |
| `viewmats` | `[C, 4, 4]` | world-to-camera matrices, the inverse of each camera-to-world. The camera axes are OpenCV's, as in COLMAP: x right, y down, z forward. |
| `Ks` | `[C, 3, 3]` | pinhole intrinsics, in pixels |
| `width`, `height` | ints | the image size |

Gradients reach the Gaussians and `viewmats`, which the trainer's pose optimisation relies on. They don't reach `Ks`.

Colours work two ways:

- `sh_degree=None`, the default: `colors` are final values with any number of channels D. Above `channel_chunk` (32) channels they render 32 at a time, which is slower.
- `sh_degree` set: `colors` are spherical harmonic coefficients, K per colour channel, laid out coefficient first and channel last. `(sh_degree + 1)²` must be at most K, so a low degree uses only the first bands of a full set. gsplat evaluates them in the direction from the camera centre to the Gaussian's centre, adds 0.5 and clamps below at 0, like Inria's renderer. Nothing clamps above, so a render can go over 1.

Options, with 1.5.3's defaults:

| Argument | Default | Effect |
| --- | --- | --- |
| `near_plane`, `far_plane` | `0.01`, `1e10` | the depth range kept, in scene units |
| `radius_clip` | `0.0` | skips Gaussians whose projected radius is this many pixels or less. It speeds up large scenes, and 0 turns it off. |
| `eps2d` | `0.3` | added to each projected 2D covariance, so nothing renders smaller than about 3 pixels |
| `packed` | `True` | keeps the in-between results as a list of the visible (view, Gaussian) pairs: less memory, sometimes slower. `False` keeps dense `[C, N, ...]` arrays, and is the trainer's default. |
| `tile_size` | `16` | tile size in pixels. Other sizes are untested. |
| `backgrounds` | `None` | `[C, D]` colour behind the Gaussians, added as `(1 - alpha) * background` |
| `render_mode` | `"RGB"` | `"RGB"`, `"D"`, `"ED"`, `"RGB+D"` or `"RGB+ED"`, see below |
| `sparse_grad` | `False` | returns the gradients of `means`, `quats` and `scales` as COO sparse tensors. Only with `packed=True`, and it needs a sparse optimiser such as `torch.optim.SparseAdam`. |
| `absgrad` | `False` | also computes absolute gradients of the 2D centres, read as `meta["means2d"].absgrad`. DefaultStrategy's `absgrad` mode (the AbsGS paper) uses them. MCMC doesn't. |
| `rasterize_mode` | `"classic"` | `"antialiased"` scales each opacity by `sqrt(det(Σ) / det(Σ + eps2d·I))`, the 2D filter from Mip-Splatting, where Σ is the projected covariance |
| `channel_chunk` | `32` | channels rendered per pass |
| `distributed` | `False` | spreads the Gaussians over several GPUs |
| `camera_model` | `"pinhole"` | also `"ortho"`, `"fisheye"`, and `"ftheta"`, which only works with `with_ut` |
| `segmented` | `False` | segmented radix sort, usually slower overall |
| `with_ut`, `with_eval3d` | `False` | 3DGUT: project with an unscented transform, and evaluate the Gaussians in 3D. Both need `packed=False`, `sparse_grad=False`, and `quats` and `scales` rather than `covars`. The example trainer only allows `with_ut` together with `with_eval3d`. |
| `radial_coeffs`, `tangential_coeffs`, `thin_prism_coeffs`, `ftheta_coeffs` | `None` | OpenCV-style lens distortion: radial `[C, 6]` for pinhole or `[C, 4]` for fisheye, tangential `[C, 2]`, thin prism `[C, 4]`. Only with `with_ut=True`. |
| `rolling_shutter`, `viewmats_rs` | `RollingShutterType.GLOBAL`, `None` | rolling shutter, from a second world-to-camera matrix per view. Only with `with_ut=True`. COLMAP gives one pose per image, so we have no `viewmats_rs`. |

Render modes: `"D"` is the accumulated depth, each Gaussian's blending weight times its camera-space z, summed. `"ED"` is that divided by the pixel's alpha, the expected depth. `"RGB+D"` and `"RGB+ED"` add the depth after the colour channels. Depth here is z along the camera's axis, not the distance along the ray.

### What it gives back

`render_colors, render_alphas, meta = rasterization(...)`:

- `render_colors`, `[C, H, W, X]`. X is D (3 for SH colour) in `"RGB"`, 1 in `"D"` and `"ED"`, and D + 1 in the two `"RGB+"` modes.
- `render_alphas`, `[C, H, W, 1]`: each pixel's accumulated opacity.
- `meta`: a dict of in-between results. The trainer calls it `info` and passes it to the strategy.

The keys of `meta` in 1.5.3, read from the code. The docstring's example prints an older list.

| Key | Unpacked | Packed | Holds |
| --- | --- | --- | --- |
| `radii` | `[C, N, 2]` int32 | `[nnz, 2]` | projected radius in pixels, in x and y. 0 means the Gaussian isn't in that view. |
| `means2d` | `[C, N, 2]` | `[nnz, 2]` | projected centres in pixels. DefaultStrategy reads their gradients. |
| `depths` | `[C, N]` | `[nnz]` | camera-space z of each centre |
| `conics` | `[C, N, 3]` | `[nnz, 3]` | upper triangle of the inverse 2D covariance |
| `opacities` | `[C, N]` | `[nnz]` | per-view opacities, after the antialiasing factor |
| `batch_ids`, `camera_ids`, `gaussian_ids` | `None` | `[nnz]` | the batch, view and Gaussian of each packed row |
| `tile_width`, `tile_height`, `tile_size`, `tiles_per_gauss`, `isect_ids`, `flatten_ids`, `isect_offsets` | | | the tile binning and depth sort |
| `width`, `height`, `n_batches`, `n_cameras` | | | sizes |

Unpacked, `(meta["radii"] > 0).all(-1)` marks the Gaussians each view drew, and the trainer's viewer counts them that way.

## The MCMC strategy

`gsplat.MCMCStrategy` follows the paper "3D Gaussian Splatting as Markov Chain Monte Carlo". The original 3DGS clones, splits and prunes Gaussians by their gradients, sizes and opacities, and gsplat's `DefaultStrategy` does that. MCMC keeps the count under a fixed cap instead. It moves nearly transparent Gaussians onto opaque ones, adds new ones where opacity is high, and jitters the nearly transparent ones.

### What it expects

- `params`: a dict of `torch.nn.Parameter`, or a `torch.nn.ParameterDict`, with at least `means`, `scales`, `quats` and `opacities`, stored raw: `scales` as logs and `opacities` as logits. The strategy applies `exp()` and `sigmoid()` itself. It copies and grows any other keys, like `sh0` and `shN`, alongside.
- `optimizers`: a dict with the same keys as the trainable parameters. Each optimizer has exactly one param group, holding its one parameter. `check_sanity()` asserts both.
- It swaps in new parameter tensors and edits the optimizer state in place. So read the parameters through `params[...]` every step, and never hold on to an old tensor.

### Settings

| Field | Default | What it does |
| --- | --- | --- |
| `cap_max` | `1_000_000` | the most Gaussians allowed |
| `noise_lr` | `5e5` | scales the position noise, see step 4 below |
| `refine_start_iter` | `500` | relocating and adding start after this step |
| `refine_stop_iter` | `25_000` | and stop before this one |
| `refine_every` | `100` | how often they run |
| `min_opacity` | `0.005` | a Gaussian at or below this opacity counts as dead. It's also the lowest opacity a moved or new Gaussian gets. |
| `verbose` | `False` | prints how many Gaussians were moved and added |

### What one step does

Call `initialize_state()` once. The state it returns holds a 51 × 51 table of binomial coefficients for the relocation formula. After each backward pass, call `step_post_backward(params, optimizers, state, step, info, lr)`, where `lr` is the current learning rate of `means`. MCMC has no `step_pre_backward` of its own (the base class's does nothing), and it never reads `info`.

On a refine step, when `refine_start_iter < step < refine_stop_iter` and `step % refine_every == 0` (steps 600 to 24,900 with the defaults):

1. **Relocate.** Every dead Gaussian becomes a copy of a live one. The sources are drawn at random in proportion to opacity, with replacement. Each source and its copies get a lower opacity and a new scale from the paper's equation 9, chosen so that together they render about like the source did alone. Opacities are clamped to `[min_opacity, 1 - ε]`, and the sources' Adam moments are zeroed.
2. **Add.** The new count is `min(cap_max, int(1.05 * N))`, so the count grows by 5% per refine until it reaches the cap. The sources are drawn from all the Gaussians in the same way and split the same way, and the new Gaussians start with zeroed Adam moments.
3. `torch.cuda.empty_cache()`.

On every step, from the first to the last:

4. **Noise.** Each centre moves by normal random noise, multiplied by the Gaussian's 3D covariance matrix, by `lr * noise_lr`, and by an opacity gate, `sigmoid(100 * (0.005 - opacity))`. The gate is 0.62 at opacity 0, 0.5 at 0.005, about 0.01 at 0.05 and practically 0 above 0.1, so opaque Gaussians stay where they are. The noise shrinks as the `means` learning rate decays.

MCMC never deletes a Gaussian. Once the count reaches the cap, adding stops and relocation only moves Gaussians, so a run that reaches the cap ends with exactly `cap_max`. Gaussians that fade after the last refine stay in that set, nearly transparent. Starting from 100,000 COLMAP points, the default cap of 1,000,000 is reached after 48 refines, around step 5,300 (our arithmetic, not the docs').

### The example's preset

`python simple_trainer.py mcmc` starts from the trainer's default config and changes five fields:

| Field | `default` preset | `mcmc` preset |
| --- | --- | --- |
| `init_opa`, the starting opacity | `0.1` | `0.5` |
| `init_scale`, a factor on the starting size | `1.0` | `0.1` |
| `opacity_reg` | `0.0` | `0.01` |
| `scale_reg` | `0.0` | `0.01` |
| `strategy` | `DefaultStrategy(verbose=True)` | `MCMCStrategy(verbose=True)` |

The two regularisers add `opacity_reg * mean(sigmoid(opacities))` and `scale_reg * mean(exp(scales))` to the loss. They pull opacities and sizes down, so Gaussians the images don't need fade below `min_opacity` and get relocated.

Strategy fields are set on the command line like `--strategy.cap-max 1000000`, as `examples/benchmarks/mcmc.sh` does. `--steps_scaler` scales `refine_start_iter`, `refine_stop_iter` and `refine_every` along with the step counts, but not `cap_max` or `noise_lr`.

### What it costs

The 1.5.3 docs' evaluation page, averaged over the 7 Mip-NeRF 360 scenes on an A100. The page gives neither the step count nor the image sizes. `benchmarks/mcmc.sh` runs the same scenes with the default 30,000 steps, at a data factor of 4 outdoors and 2 indoors.

| Run | PSNR | SSIM | LPIPS | Gaussians | GPU memory | Minutes |
| --- | --- | --- | --- | --- | --- | --- |
| DefaultStrategy | 29.00 | 0.87 | 0.14 | 3,237,318 | 5.62 GB | 19.39 |
| MCMC, cap 1M | 29.18 | 0.87 | 0.14 | 1,000,000 | 1.98 GB | 15.42 |
| MCMC, cap 2M | 29.53 | 0.88 | 0.13 | 2,000,000 | 3.43 GB | 21.79 |
| MCMC, cap 3M | 29.65 | 0.89 | 0.12 | 3,000,000 | 4.99 GB | 27.63 |

Memory and time follow the cap. Our machines have different GPUs, so use the ratios, not the minutes.

## The example trainer

`examples/simple_trainer.py` is a script, not part of the gsplat package, so `pip install gsplat` doesn't bring it. It imports the modules next to it in `examples/` and the packages in `examples/requirements.txt`.

### The files

| File | What it does |
| --- | --- |
| `simple_trainer.py` | the `Config`, the `Runner` class that trains and evaluates, and the command line |
| `datasets/colmap.py` | `Parser` reads a COLMAP model, and `Dataset` serves one image at a time |
| `datasets/normalize.py` | the transforms that normalise the world frame |
| `datasets/traj.py` | camera paths for the trajectory videos |
| `utils.py` | `knn` (with scikit-learn), `rgb_to_sh`, `set_random_seed`, and the pose and appearance modules |
| `gsplat_viewer.py`, `lib_bilagrid.py` | the browser viewer and the optional bilateral grid |
| `benchmarks/*.sh` | benchmark runs, among them `basic.sh`, which the docs name for their default-strategy numbers, and `mcmc.sh` |

### Loading the data

`Parser(data_dir, factor, normalize, test_every)`:

- Reads the COLMAP model from `<data_dir>/sparse/0/`, or from `<data_dir>/sparse/` when there's no `0/`. It uses `SceneManager` from a fork of pycolmap, `rmbrualla/pycolmap`, pinned by commit in `requirements.txt`. Images COLMAP didn't register are left out.
- Loads the images from `<data_dir>/images/`, or from `images_<factor>/` when `--data_factor` is above 1, and then that folder has to exist. The default factor is 4, so full-size images need `--data_factor 1`. COLMAP's image names must be paths under `images/`. If the first file in `images_<factor>/` is a `.jpg`, it uses none of them and writes its own bicubic PNG copies of `images/` to `images_<factor>_png/`. It pairs the files of the two folders by sorted position, so both must hold the same set.
- Takes K from COLMAP's camera, divides it by the factor, then rescales it to the size of the first image on disk.
- Accepts the SIMPLE_PINHOLE, PINHOLE, SIMPLE_RADIAL, RADIAL, OPENCV and OPENCV_FISHEYE camera models. For a distorted camera, `Dataset` undistorts each image as it loads it, with OpenCV's `remap`, and crops to the valid area. For fisheye it also makes a mask of the valid pixels.
- Sorts the images by name and holds every `test_every`-th one out of training, for evaluation: indices 0, 8, 16 and so on with the default of 8. No setting trains on all of them. `test_every=1` puts every image in the evaluation set, and a `test_every` above the image count still holds out the first one.
- With `normalize_world_space` on, the default, moves the world in three steps. First it rotates it so the cameras' average up direction is +z. It recentres it on the median, over the cameras, of the point on each camera's forward ray closest to the origin, and scales it so the median camera distance is 1. Then it moves the points' median to the origin and rotates the world onto the point cloud's principal axes, with the smallest spread on z. Last, it turns it 180° about x if the points' median z is above their mean, which assumes more points near the floor. `parser.transform` is the 4 × 4 matrix from COLMAP's world to this frame. The splat trains, and exports, in the normalised frame.
- Sets `scene_scale` to the largest distance from a camera to the cameras' mean position. The trainer multiplies the `means` learning rate by `1.1 * scene_scale * global_scale`.

`Dataset(parser, split, patch_size, load_depths)` gives one dict per image: `K` `[3, 3]`, `camtoworld` `[4, 4]`, `image` `[H, W, 3]` as floats from 0 to 255, `image_id` (its index within the split), `mask` for fisheye only, and with `load_depths` the COLMAP points it sees with their depths, for the depth loss.

### Building the Gaussians

`create_splats_with_optimizers()`:

- Starts from COLMAP's points and their colours (`init_type="sfm"`), or from `init_num_pts` random points within `init_extent * scene_scale` of the origin on each axis (`"random"`).
- Sizes each Gaussian by the root mean square of the distances to its 3 nearest neighbours, times `init_scale`, the same on all three axes, and stores the log.
- Gives random quaternions from `torch.rand`, which aren't normalised, and opacity `logit(init_opa)`.
- Sets `sh0 = (rgb - 0.5) / 0.28209479177387814`, shaped `[N, 1, 3]`, and `shN` to zeros, shaped `[N, (sh_degree + 1)² - 1, 3]`.
- Makes one Adam optimizer per parameter, with eps 1e-15. The learning rates are 1.6e-4 × scene scale for `means`, 5e-3 for `scales`, 1e-3 for `quats`, 5e-2 for `opacities`, 2.5e-3 for `sh0` and 2.5e-3 / 20 for `shN`. A batch size B above 1, counting the batch on every GPU, multiplies the rates by √B, divides eps by √B, and sets each beta to `1 - B * (1 - beta)`.

So the keys are `means`, `scales`, `quats`, `opacities`, `sh0` and `shN`, the ones MCMC needs and `export_splats()` takes. With `--app_opt`, the colour keys become `features` and `colors`.

### The training loop

`Runner.train()` makes an `ExponentialLR` scheduler on `means` that reaches 0.01 × its starting rate at `max_steps`, more schedulers when the pose or grid extras are on, and a `DataLoader` with batch size 1, shuffling and 4 workers. Then each step, in this order:

1. Takes the next image, restarting the loader when it runs out.
2. Renders it with `rasterize_splats()`: `exp()` of the scales, `sigmoid()` of the opacities, `sh0` and `shN` joined as the colours, and `viewmats` from `inv(camtoworld)`. The SH degree starts at 0 and goes up one every `sh_degree_interval` (1,000) steps, up to `sh_degree` (3).
3. Applies the optional extras: the bilateral grid, and a random background behind the render (`random_bkgd`).
4. Calls `strategy.step_pre_backward()`.
5. Computes the loss, `0.8 * L1 + 0.2 * (1 - SSIM)` with `ssim_lambda` at 0.2 and SSIM from fused-ssim, plus the depth, grid and regularisation terms that are on.
6. Calls `loss.backward()`.
7. Writes TensorBoard scalars every `tb_every` (100) steps.
8. Saves a checkpoint at the save steps and the last step, and with `--save_ply` a PLY at the PLY steps and the last step, both before this step's update.
9. Calls `step()` and `zero_grad(set_to_none=True)` on every optimizer, then the scheduler's `step()`.
10. Calls `strategy.step_post_backward()`. MCMC gets `lr=schedulers[0].get_last_lr()[0]`, the `means` rate the scheduler just set for the next step.
11. At the eval steps, measures PSNR, SSIM and LPIPS on the held-out images and renders a trajectory video.
12. Updates the viewer, when it's on.

`fused_ssim(img1, img2, padding="same")` takes `[B, C, H, W]` images and returns the mean SSIM. Only `img1` gets a gradient, so the render goes first. `padding="valid"`, which the trainer uses, drops a 5-pixel border. It has no mask argument. The examples pin fused-ssim at commit `328dc9836f513d00c4b5bc38fe30478b4435cbb5`. Its `setup.py` compiles for the GPU it sees, with its own `-arch` flag, and torch then ignores `TORCH_CUDA_ARCH_LIST`. On our A10G image builder that's sm_86, which also runs on the L4 and L40S (8.9). With no GPU visible it falls back to sm_75, sm_80 and sm_89.

When `Dataset` returns a `mask` (fisheye only), the trainer sets the render to 0 outside it and leaves the photo as it is. That works for fisheye because `remap` already left those photo pixels black.

### Settings

`Config` fields worth knowing, with their defaults:

| Field | Default | Note |
| --- | --- | --- |
| `data_dir`, `data_factor`, `result_dir` | `data/360_v2/garden`, `4`, `results/garden` | the input folder, the image downscale factor, and where results go |
| `max_steps` | `30_000` | |
| `eval_steps`, `save_steps`, `ply_steps` | `[7_000, 30_000]` each | `--eval_steps -1` turns evaluation off, as the benchmarks do |
| `save_ply` | `False` | no PLY without it |
| `test_every`, `normalize_world_space` | `8`, `True` | see "Loading the data" |
| `init_type` | `"sfm"` | or `"random"` |
| `sh_degree`, `sh_degree_interval` | `3`, `1_000` | |
| `ssim_lambda` | `0.2` | |
| `packed`, `antialiased`, `random_bkgd` | `False` | `antialiased` sets `rasterize_mode="antialiased"` |
| `batch_size`, `steps_scaler` | `1`, `1.0` | `steps_scaler` multiplies the step counts, the SH interval and the strategy's refine steps |
| `disable_viewer`, `disable_video` | `False` | see "How it runs" |
| `lpips_net` | `"alex"` | `"vgg"` matches the LPIPS of the original 3DGS repo |
| `pose_opt`, `app_opt`, `use_bilateral_grid`, `depth_loss`, `with_ut`, `visible_adam`, `sparse_grad` | `False` | the extras: camera pose refinement, per-image appearance, a bilateral grid, a depth loss against COLMAP's points, 3DGUT, the selective Adam from Taming 3DGS, and sparse gradients |

### What it writes

Under `--result_dir`:

```text
cfg.yml                              the config it ran with
ckpts/ckpt_<step>_rank0.pt           {"step": ..., "splats": ...}, the raw parameters
ply/point_cloud_<step>.ply           only with --save_ply
stats/train_step<step>_rank0.json    peak GPU memory in GB, seconds since the start, Gaussian count
stats/val_step<step>.json            psnr, ssim, lpips, seconds per image, Gaussian count
renders/val_step<step>_<i>.png       photo and render side by side
videos/traj_<step>.mp4               the trajectory video
tb/                                  TensorBoard logs
```

Steps in file names count from 0, so after 30,000 steps the last files say `29999`.

### How it runs

We never run it (see "Our rules that touch this"). This is how it's meant to run, for reading its code and its benchmark scripts.

It runs from inside `examples/`, because it imports `datasets`, `utils` and `gsplat_viewer` by their top-level names. The command line comes from tyro: a preset name, `default` or `mcmc`, then any config field as a flag. The script runs through `gsplat.distributed.cli`, which needs CUDA and starts one process per visible GPU, turning the viewer off when there's more than one. `--ckpt <file>` skips training and only evaluates and renders from the checkpoint.

- With the viewer on, the default, the script serves it on port 8080 while it trains, then prints "Viewer running... Ctrl+C to exit." and sleeps for 1,000,000 seconds, about 11.5 days. `--disable_viewer` turns that off. Code adapted from the example leaves the viewer out, since every machine ends when its job ends (rules.md).
- `--disable_video` skips the trajectory videos, which need imageio's ffmpeg plugin.
- `--eval_steps -1` skips evaluation, but the trainer still builds the LPIPS metric when it starts.
- The script and its modules import viser, nerfview, tyro, torchmetrics, TensorBoard, imageio, PyYAML, OpenCV, SciPy, scikit-learn, matplotlib and the pycolmap fork, among others, when it starts.

## export_splats()

`gsplat.export_splats(means, scales, quats, opacities, sh0, shN, format="ply", save_to=None)` returns the file as bytes, and also writes it when `save_to` is set. It takes the raw parameters, with `sh0` as `[N, 1, 3]` and `shN` as `[N, K, 3]`. The formats:

- `"ply"`: the standard 3DGS PLY. It's `binary_little_endian` with one `element vertex` and float columns in this order: `x y z`, `f_dc_0` to `f_dc_2`, the `f_rest_*` columns, `opacity`, `scale_0` to `scale_2`, `rot_0` to `rot_3`. The `f_rest_*` columns are channel-major, all of R, then G, then B, as the splat-files skill describes. Opacity stays a logit, scales stay logs, and the quaternions go out as stored, not normalised. There are no normals and no header comments, so nothing marks a splat trained with `antialiased` (splat-files has the comment viewers read).
- `"splat"`, for antimatter15's viewer, and `"ply_compressed"`, SuperSplat's compressed PLY, which also drops Gaussians with opacity at or below 1/255.

Before writing, it drops every Gaussian with a NaN or an infinity in any field. In 1.5.3 the opacity check reduces over all Gaussians at once (`any(dim=0)` on a 1-D tensor), so one non-finite opacity drops every Gaussian and the PLY holds 0 vertices. So drop the rows with a non-finite opacity yourself before calling it.

## A cut-down MCMC loop

The `mcmc` preset's training loop with every optional part removed. `splats` is the `ParameterDict` from `create_splats_with_optimizers()`, on the GPU. `scene_scale` is the trainer's, 1.1 × the Parser's. `batches` yields `Dataset` items through a `DataLoader` without end, and `device` and `max_steps` are what they say. If any of this goes into our code, its licence note goes with it (rules.md).

```python
# Adapted from gsplat v1.5.3, examples/simple_trainer.py
# https://github.com/nerfstudio-project/gsplat/blob/v1.5.3/examples/simple_trainer.py
#
# Copyright 2025 Nerfstudio Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Changes: cut down to the mcmc preset's training loop on one GPU, with no viewer,
# evaluation, depth loss, pose or appearance optimisation or batch scaling, and with the
# example's default numbers written in. Ours come from config.toml.

import torch
import torch.nn.functional as F
from fused_ssim import fused_ssim
from gsplat import MCMCStrategy, export_splats, rasterization

lrs = {"means": 1.6e-4 * scene_scale, "scales": 5e-3, "quats": 1e-3,
       "opacities": 5e-2, "sh0": 2.5e-3, "shN": 2.5e-3 / 20}
optimizers = {
    name: torch.optim.Adam([{"params": splats[name], "lr": lr, "name": name}], eps=1e-15)
    for name, lr in lrs.items()
}
strategy = MCMCStrategy(cap_max=1_000_000)
strategy.check_sanity(splats, optimizers)
state = strategy.initialize_state()
scheduler = torch.optim.lr_scheduler.ExponentialLR(
    optimizers["means"], gamma=0.01 ** (1.0 / max_steps)
)

for step in range(max_steps):
    data = next(batches)
    camtoworlds = data["camtoworld"].to(device)  # [1, 4, 4]
    Ks = data["K"].to(device)  # [1, 3, 3]
    pixels = data["image"].to(device) / 255.0  # [1, H, W, 3]
    height, width = pixels.shape[1:3]

    colors, alphas, info = rasterization(
        means=splats["means"],
        quats=splats["quats"],
        scales=torch.exp(splats["scales"]),
        opacities=torch.sigmoid(splats["opacities"]),
        colors=torch.cat([splats["sh0"], splats["shN"]], 1),  # [N, K, 3]
        viewmats=torch.linalg.inv(camtoworlds),
        Ks=Ks,
        width=width,
        height=height,
        sh_degree=min(step // 1000, 3),
        packed=False,
    )

    l1loss = F.l1_loss(colors, pixels)
    ssimloss = 1.0 - fused_ssim(
        colors.permute(0, 3, 1, 2), pixels.permute(0, 3, 1, 2), padding="valid"
    )
    loss = l1loss * 0.8 + ssimloss * 0.2
    loss += 0.01 * torch.sigmoid(splats["opacities"]).mean()  # opacity_reg
    loss += 0.01 * torch.exp(splats["scales"]).mean()  # scale_reg
    loss.backward()

    for optimizer in optimizers.values():
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    scheduler.step()

    strategy.step_post_backward(
        params=splats, optimizers=optimizers, state=state, step=step, info=info,
        lr=scheduler.get_last_lr()[0],
    )

export_splats(
    means=splats["means"], scales=splats["scales"], quats=splats["quats"],
    opacities=splats["opacities"], sh0=splats["sh0"], shN=splats["shN"],
    format="ply", save_to="splat.ply",
)
```

## Our rules that touch this

- gsplat, torch and fused-ssim go only in the machine image, never in `pyproject.toml` (tech.md). Laptops don't have them, so laptop tests can only import the parts of the training step that don't import them.
- Our training step is our own code on gsplat's library API: `rasterization()`, `MCMCStrategy` and `export_splats()`. The examples aren't part of the package. Anything adapted from them keeps the licence note above (rules.md). gsplat is Apache-2.0, "Copyright 2025 Nerfstudio Team", and the repo has no NOTICE file at v1.5.3.
- Never call `rasterization_inria_wrapper()` or `rasterization_2dgs_inria_wrapper()`. They run Inria's CUDA rasterizer, or 2DGS's, which is based on it, and their docstrings say that backend has its own licence. rules.md bans Inria's code.
- Use gsplat's `MCMCStrategy`, never code from the MCMC paper's own repo (`ubc-vision/3dgs-mcmc`), which is under Inria's non-commercial Gaussian-Splatting License.
- Never copy `colormap()` from `examples/utils.py`. Its comment points to the 2DGS repo (`hbb1/2d-gaussian-splatting`), also under Inria's non-commercial licence. The examples' other borrowed code is permissive: `datasets/traj.py` says it's from Google's multinerf (Apache-2.0), `datasets/normalize.py` cites nerf-factory (Apache-2.0), and `rotation_6d_to_matrix()` in `utils.py` says it's adapted from pytorch3d.
- We never run gsplat's example trainer or install its `examples/requirements.txt`, and we never use LPIPS: our trainer measures PSNR and SSIM only. So none of the example's licence questions reach our machines:
  - its requirements' plain `opencv-python` (we only ever install `opencv-python-headless`), the ffmpeg executable inside `imageio[ffmpeg]` (our ffmpeg is only the command line tool on the machine image), and its `numpy<2.0.0` pin;
  - the packages in it with no line in `docs/third-party-licences.md`, among them the pycolmap fork, viser, nerfview, tyro, torchmetrics, tensorboard, tensorly, scikit-learn, matplotlib, splines and fused-bilagrid;
  - the pretrained AlexNet or VGG16 ImageNet weights that torchmetrics' LPIPS downloads, which the trainer loads at start-up.
- So read the example for how it's put together, and write our trainer on gsplat's library API, with no viewer, no trajectory videos and no LPIPS. Anything it needs that the image doesn't have gets its line in `docs/third-party-licences.md` in the commit that adds it.
- The training step reads `SPLATPIPE_CAP_DEADLINE`, the UTC time by which it should have saved and stopped (`docs/pipeline.md`, "The cap deadline"). The example has no deadline, so our loop has to check the clock and write the PLY in time.
- The frames step writes `mask.png`, where black means ignore, for COLMAP and the training loss (`docs/pipeline.md`). The example never reads it, and fused-ssim takes no mask. Masking the way the example masks fisheye images means setting both the render and the photo to 0 where the mask is black.
- The clean step reads `out/train.ply`, which the training step writes, as a binary PLY in the usual 3DGS layout (its requirements, A1), and `export_splats(format="ply")` writes that layout. An MCMC splat has at most `cap_max` Gaussians, and the clean step refuses one with more than its `max_source_splats`, 8,000,000, so keep the cap at or under that setting.
- Every number, like the step count, the cap, the learning rates and the loss weights, goes in `config.toml` (python.md).
- Checkpoints, PLYs and TensorBoard logs go in `data/`, never in git (rules.md).

## Links

- gsplat at the pinned tag: <https://github.com/nerfstudio-project/gsplat/tree/v1.5.3>
- README, with the install options: <https://github.com/nerfstudio-project/gsplat/blob/v1.5.3/README.md>
- Licence: <https://github.com/nerfstudio-project/gsplat/blob/v1.5.3/LICENSE>
- 1.5.3 on PyPI, its sdist and wheel: <https://pypi.org/project/gsplat/1.5.3/#files>
- `setup.py`, the source build: <https://github.com/nerfstudio-project/gsplat/blob/v1.5.3/setup.py>
- The publish workflow, which builds the PyPI files with `BUILD_NO_CUDA=1`: <https://github.com/nerfstudio-project/gsplat/blob/v1.5.3/.github/workflows/publish.yml>
- `MANIFEST.in` and `.gitmodules`, what the sdist carries: <https://github.com/nerfstudio-project/gsplat/blob/v1.5.3/MANIFEST.in> and <https://github.com/nerfstudio-project/gsplat/blob/v1.5.3/.gitmodules>
- `_backend.py`, the compile on first use: <https://github.com/nerfstudio-project/gsplat/blob/v1.5.3/gsplat/cuda/_backend.py>
- `rasterization()`: <https://github.com/nerfstudio-project/gsplat/blob/v1.5.3/gsplat/rendering.py>
- `fully_fused_projection()`, the shapes behind `meta`: <https://github.com/nerfstudio-project/gsplat/blob/v1.5.3/gsplat/cuda/_wrapper.py>
- `MCMCStrategy`: <https://github.com/nerfstudio-project/gsplat/blob/v1.5.3/gsplat/strategy/mcmc.py>
- Relocation, adding and noise: <https://github.com/nerfstudio-project/gsplat/blob/v1.5.3/gsplat/strategy/ops.py> and <https://github.com/nerfstudio-project/gsplat/blob/v1.5.3/gsplat/relocation.py>
- The strategy base class: <https://github.com/nerfstudio-project/gsplat/blob/v1.5.3/gsplat/strategy/base.py>
- `export_splats()`: <https://github.com/nerfstudio-project/gsplat/blob/v1.5.3/gsplat/exporter.py>
- The example trainer: <https://github.com/nerfstudio-project/gsplat/blob/v1.5.3/examples/simple_trainer.py>
- Its COLMAP loader and normalisation: <https://github.com/nerfstudio-project/gsplat/blob/v1.5.3/examples/datasets/colmap.py> and <https://github.com/nerfstudio-project/gsplat/blob/v1.5.3/examples/datasets/normalize.py>
- Its helpers: <https://github.com/nerfstudio-project/gsplat/blob/v1.5.3/examples/utils.py>
- Its requirements: <https://github.com/nerfstudio-project/gsplat/blob/v1.5.3/examples/requirements.txt>
- The MCMC benchmark script: <https://github.com/nerfstudio-project/gsplat/blob/v1.5.3/examples/benchmarks/mcmc.sh>
- Docs, rasterization: <https://docs.gsplat.studio/versions/1.5.3/apis/rasterization.html>
- Docs, densification strategies: <https://docs.gsplat.studio/versions/1.5.3/apis/strategy.html>
- Docs, fitting a COLMAP capture: <https://docs.gsplat.studio/versions/1.5.3/examples/colmap.html>
- Docs, data conventions: <https://docs.gsplat.studio/versions/1.5.3/conventions/data_conventions.html>
- Docs, evaluation and the MCMC numbers: <https://docs.gsplat.studio/versions/1.5.3/tests/eval.html>
- fused-ssim at the examples' pinned commit: <https://github.com/rahul-goel/fused-ssim/blob/328dc9836f513d00c4b5bc38fe30478b4435cbb5/fused_ssim/__init__.py>
- torch 2.7.1's `cpp_extension`, with the build folder, the CUDA version check and the JIT rebuild: <https://github.com/pytorch/pytorch/blob/v2.7.1/torch/utils/cpp_extension.py>
- fused-ssim's `setup.py` at that commit, with its own architecture flags: <https://github.com/rahul-goel/fused-ssim/blob/328dc9836f513d00c4b5bc38fe30478b4435cbb5/setup.py>
- torchmetrics' LPIPS, which loads the torchvision weights: <https://github.com/Lightning-AI/torchmetrics/blob/master/src/torchmetrics/functional/image/lpips.py>
- imageio-ffmpeg, on the ffmpeg in its wheels: <https://github.com/imageio/imageio-ffmpeg>
- 3D Gaussian Splatting as Markov Chain Monte Carlo: <https://arxiv.org/abs/2404.09591>
- The paper's own repo and its licence, which we can't use: <https://github.com/ubc-vision/3dgs-mcmc/blob/main/LICENSE.md>
- Mip-Splatting, behind `antialiased`: <https://arxiv.org/abs/2311.16493>
- AbsGS, behind `absgrad`: <https://arxiv.org/abs/2404.10484>
- 3DGUT, behind `with_ut`: <https://arxiv.org/abs/2412.12507>
