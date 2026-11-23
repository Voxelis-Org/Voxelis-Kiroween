---
name: moge-depth
description: Depth, normals and a mask of where there is geometry from one image, with MoGe-2 and the Ruicheng/moge-2-vitb-normal weights that the machine image keeps at /opt/models/moge-2-vitb-normal/model.pt. How to load MoGe-2 from that file with no download, what MoGeModel.infer() takes (an RGB image and the horizontal field of view) and gives back (metric depth, normals, the mask, points and intrinsics), the Python packages MoGe's code imports and declares, and the licences of its code and its weights. Use when writing or reviewing the depth-priors step, loading or calling MoGe, installing MoGe on the machine image, or turning a COLMAP camera into MoGe's field of view.
metadata:
  sources: microsoft/MoGe at commit 74fbce054ebed49800de42d0ad0e83495065719a (README, LICENSE, docs, configs, pyproject.toml, uv.lock, moge/model/v2.py and the code it imports, the scripts, baselines/moge.py and the training code), the Ruicheng/moge-2-vitb-normal card and file list at revision ca5f0e07ff01d3e5a364c1d954ed12ee1814b368, the facebook/dinov2-base card, utils3d-moge, pipeline and FlexGEMM at the commits MoGe pins, huggingface_hub 1.26.0's source, docs and PyPI page, the PyTorch 2.7 docs and COLMAP's database docs, read November 2026
---

# MoGe-2 depth from one image

Notes for loading MoGe-2 from the weights file on our machine image and running it on one frame. They come from the MoGe repo at commit `74fbce05`, the `Ruicheng/moge-2-vitb-normal` model card at revision `ca5f0e07`, and the pages of the packages MoGe depends on, read on GitHub and the Hugging Face Hub. Everything is in our own words: content from those sources was rephrased for compliance with licensing restrictions. Nothing was installed or run to write these notes, and the example code is ours. Where a short expression in backticks is MoGe's own code, it's from MoGe's MIT-licensed source, Copyright (c) Microsoft Corporation. Links are at the bottom.

MoGe, torch and the rest exist only on the machine image (tech.md), so none of this runs on a laptop.

## What we pin

| What | Pin |
| --- | --- |
| MoGe's code | `microsoft/MoGe` at commit `74fbce054ebed49800de42d0ad0e83495065719a`, "V3 (#158)" from 19 August 2026, the head of `main` when we read it. The repo has no tags, so a commit is the only pin. Its `pyproject.toml` calls the package `moge` 3.0.0. |
| The class | `moge.model.v2.MoGeModel`. The same commit also has MoGe-1 (`moge.model.v1`) and MoGe-3 (`moge.model.v3`), and `moge infer` now defaults to v3. |
| The weights | `model.pt` from `Ruicheng/moge-2-vitb-normal` at revision `ca5f0e07ff01d3e5a364c1d954ed12ee1814b368`: 419,110,160 bytes, sha256 `16b8110e86d5dc5a849db120ca96ef3a223fd30b0c9146d1d81db504073da5f6`. The Hub lists that sha256 as the file's LFS hash, and it's the `MOGE_SHA256` in `cloud/image.py`. |
| On the image | `/opt/models/moge-2-vitb-normal/model.pt`. Step 7 of the image build downloads it from that revision and checks its sha256, and the smoke test checks it again (`_check_moge_weights` in `cloud/smoke_runner.py`). The image doesn't have MoGe's code yet. |

## The model

MoGe-2 takes one image and predicts the scene's geometry as the camera sees it: a point map, a metric scale, normals and a mask, which `infer()` turns into depth in metres. `moge-2-vitb-normal` is the ViT-B checkpoint with the normal head: 104M parameters, the second smallest of MoGe-2's four checkpoints (README). Its card names `facebook/dinov2-base` as the base model.

In `moge/model/v2.py`, the model is:

- a DINOv2 encoder, built from MoGe's own copy of DINOv2's code;
- a convolutional neck;
- heads for the points, the normals and the mask;
- an MLP on the encoder's class token, which predicts one metric scale per image.

The class builds each head only when the checkpoint's config asks for it, and the README lists metric scale and normals for this checkpoint.

The checkpoint is a dict that holds `model_config`, the keyword arguments `MoGeModel(...)` is built with, and `model`, the state dict. So the architecture and settings like `num_tokens_range` come from the checkpoint, and the code's defaults only fill gaps. On the machine, `torch.load(path, map_location="cpu", weights_only=True)["model_config"]` shows the real values.

The README's only speed figure is for ViT-L: 60 ms per image in FP16 on an A100 or RTX 3090.

## Loading it from the local file

At this commit, `MoGeModel.from_pretrained(path)`:

1. Uses `path` as it is when `Path(path).exists()`. Otherwise it treats the argument as a Hub repo id and calls `hf_hub_download(repo_id=..., filename="model.pt")`.
2. Loads the file with `torch.load(..., map_location="cpu", weights_only=True)`. With `weights_only=True`, the unpickler only loads tensors, primitive types, dicts and types registered as safe, so the file can't run code the way a full pickle load can (PyTorch docs).
3. Builds `MoGeModel(**model_config)`, after merging in any `model_kwargs` it was given.
4. Loads the state dict with `strict=False`. Missing or unexpected keys only cause a warning that lists them. A comment in the code says this lets a checkpoint leave out optional heads.

With the file in place, nothing goes to the network:

- The encoder builds DINOv2 with `pretrained=False`, so no DINOv2 weights are fetched. Only `init_weights()` fetches them, through `torch.hub` from `dl.fbaipublicfiles.com`. MoGe's training code calls it at the start of every run, before it loads any checkpoint on top. Never call it: it would download, and it would overwrite the fine-tuned encoder with DINOv2's original weights.
- `huggingface_hub` is still imported, at the top of `moge/model/v2.py`, so it has to be installed even though a local load never calls it.

What our code should add, because MoGe's doesn't:

- Check that the file is there first, and fail with a readable error if it isn't. A missing file sends the absolute path to `hf_hub_download` as a repo id. huggingface_hub 1.26.0 refuses an id with more than one `/`, or one that isn't a `str` (a `Path`, say), with an `HFValidationError` before it makes any request. So nothing downloads, but the error talks about repo ids.
- Set `HF_HUB_OFFLINE=1` in the environment before anything imports huggingface_hub. Then the Hub client makes no HTTP requests at all. Version 1.26.0 reads the variable once, when `huggingface_hub.constants` is imported, so setting it later does nothing.
- Make a wrong checkpoint fail, instead of running with random weights in a part it lacks. Either turn the two load warnings into errors, or do the load yourself: build `MoGeModel(**checkpoint["model_config"])`, then call `load_state_dict(checkpoint["model"])`, which is strict by default. The example below does that. The sha256 checks already catch a corrupt or different file.

Then move the model with `.to("cuda")` and call `.eval()`, as MoGe's scripts do. Leave it in fp32, as the README's example does: `infer()` already runs the network under fp16 autocast by default, and works out depth and intrinsics in fp32. MoGe's `moge infer --fp16` casts the whole model with `.half()` instead.

## What infer() takes

`model.infer(image, num_tokens=None, resolution_level=9, force_projection=True, apply_mask=True, fov_x=None, use_fp16=True)` runs under `torch.inference_mode()`.

| Argument | Default | What it is |
| --- | --- | --- |
| `image` | | `(3, H, W)`, or `(B, 3, H, W)` for a batch of same-size images: RGB as floats from 0 to 1. It's moved to the model's device and dtype. Don't normalise it, because the encoder subtracts ImageNet's mean and divides by its standard deviation itself. The README says aspect ratios from 2:1 to 1:2 work. |
| `fov_x` | `None` | the horizontal field of view in degrees, as a number or as a `(B,)` tensor with one per image. With `None`, MoGe estimates it from its own point map. |
| `num_tokens` | `None` | how many ViT patches to cut the image into, as an int. The docstring and the CLI suggest 1,200 to 2,500. The docstring also offers `'least'` and `'most'`, which this commit's code doesn't handle. |
| `resolution_level` | `9` | 0 to 9, used when `num_tokens` is `None`. It picks `num_tokens` linearly across the checkpoint's `num_tokens_range`, from the low end at 0 to the high end at 9. The code's default range, and the one in MoGe's v2 training config, is 1,200 to 3,600, and training used counts across that range. |
| `force_projection` | `True` | rebuilds the point map from the depth and the intrinsics, so each point lies exactly on its pixel's ray. |
| `apply_mask` | `True` | sets depth and points to `+inf`, and normals to 0, where the mask is false. |
| `use_fp16` | `True` | runs the network under fp16 autocast. It has no effect on a model that's already fp16. |

How much detail the network sees: the encoder resizes the image (bilinear, antialiased) to a grid of 14-pixel patches, about `num_tokens` of them in the image's aspect ratio. The model builds five feature levels from that grid, each twice the size of the one before, the heads predict at the finest, and the maps are resized bilinearly to the input size. For a 16:9 frame at 3,600 tokens, that's 80 × 45 patches, a 1120 × 630 input to the ViT and predictions at 1280 × 720. The maps always come back at the input size, but the network only sees the image at the ViT's input size, so a larger frame gets larger maps and no more detail.

### The field of view

- MoGe assumes a pinhole camera with square pixels, the principal point at the image centre and no distortion. Its focal solve says so in its docstring, and the intrinsics it returns always have the principal point at the centre.
- With `fov_x`, MoGe sets its focal length from it and solves only for the depth shift. Without it, it solves for both. The solve runs on the CPU, one image at a time: scipy's `least_squares` with Levenberg-Marquardt, on a 64 × 64 nearest-neighbour copy of the masked point map. When fewer than two of those samples are in the mask, it skips the solve and uses a shift of 0, and a focal length of 1 when it had no `fov_x`.
- Our camera solve gives one PINHOLE camera for undistorted frames (`docs/pipeline.md`, "Step: train"). From its width W and its fx, both in pixels, `fov_x = degrees(2 * atan(W / (2 * fx)))`. The field of view doesn't change when an image is scaled, as long as W and fx come from the same size. MoGe's evaluation code, `baselines/moge.py`, gets `fov_x` from known intrinsics with the same formula, through utils3d-moge's `intrinsics_to_fov`, which also assumes a centred principal point.
- MoGe then returns intrinsics whose fx matches COLMAP's. Their fy equals fx in pixels and their principal point is the centre, whatever COLMAP found, because `infer()` takes nothing else about the camera.

## What infer() gives back

A dict of tensors on the model's device, float32 except the mask. Shapes are for one image, and a batch adds a leading B.

| Key | Shape | Holds |
| --- | --- | --- |
| `depth` | `(H, W)` | camera-space z, in metres as MoGe-2 estimates them: distance along the optical axis, not along the ray. `+inf` where the mask is false, with `apply_mask`. |
| `normal` | `(H, W, 3)` | unit normals in the camera frame, facing the camera: a wall square to the camera has z near -1. Zeros where the mask is false, with `apply_mask`. Only checkpoints with a normal head return them, and ours has one. |
| `mask` | `(H, W)` | bool, true where there is geometry. In training, the mask head learns 0 where the ground-truth depth is infinite, like sky, and 1 elsewhere, unless a dataset's config leaves pixels of unknown depth out. `infer()` thresholds the head's sigmoid at 0.5 and also drops pixels whose depth isn't above 0. |
| `points` | `(H, W, 3)` | camera-space points in metres. With `force_projection`, each lies on the ray through its pixel's centre, at that pixel's depth. |
| `intrinsics` | `(3, 3)` | normalised pinhole intrinsics: fx and cx divided by the width, fy and cy by the height, with cx and cy at 0.5. |

The camera frame has OpenCV's axes, x right, y down and z forward, like COLMAP's and gsplat's.

Using them:

- Pixel intrinsics: multiply the first row by the width and the second by the height. MoGe puts pixel centres at (i + 0.5) / W, with 0 at the image's top-left corner. COLMAP uses the same convention, with the top-left pixel's centre at (0.5, 0.5), so the result compares directly with `cameras.bin`. In OpenCV's convention, with pixel centres on whole numbers, cx is W / 2 - 0.5.
- Scale: one predicted factor per image multiplies depth and points. The camera solve's model has a scale of its own, so a depth prior has to be fitted to that scale before it means anything there, for example per image against the sparse points that image sees. gsplat's `"D"` and `"ED"` renders are camera-space z too (gsplat-training skill), so the two compare after that fit.
- World frame: rotate the normals by the image's camera-to-world rotation.
- Infinities: with `apply_mask`, depth and points hold `+inf` where the mask is false, and arithmetic on them gives inf or NaN. Select the masked pixels first, or pass `apply_mask=False` and use `mask` yourself.
- Inference tensors: tensors made under `torch.inference_mode()` can't be used in computations that autograd records after it ends, or changed in place outside it. Clone them outside inference mode, which gives ordinary tensors, before using them that way.
- MoGe's mask isn't the camera solve's `masks/<stem>.png`. That one marks pixels to leave out of training, with black meaning ignore. MoGe's marks the pixels with finite geometry.

## An example

Our own code, written for this skill rather than copied from MoGe. Its load does what `from_pretrained()` does at this commit, but strictly. It decodes the frame the way the train step does, and only runs on the machine image. In a step, the path and the numbers come from `config.toml` and `cameras.bin`, and the errors are the step's readable errors.

```python
import math
import os
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"  # before anything imports huggingface_hub

import cv2
import numpy as np
import torch
from moge.model.v2 import MoGeModel

weights = Path("/opt/models/moge-2-vitb-normal/model.pt")
if not weights.is_file():
    raise RuntimeError(f"The MoGe-2 weights aren't at {weights}. Rebuild the machine image.")
checkpoint = torch.load(weights, map_location="cpu", weights_only=True)
model = MoGeModel(**checkpoint["model_config"])
model.load_state_dict(checkpoint["model"])  # strict: a missing or unexpected key is an error
model = model.to("cuda").eval()

# As stored, ignoring EXIF orientation, like splatpipe/train/files.py, then BGR to RGB.
data = np.frombuffer(Path("work/cameras/images/000000.jpg").read_bytes(), dtype=np.uint8)
bgr = cv2.imdecode(data, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
image = torch.from_numpy(rgb).to("cuda").permute(2, 0, 1).float() / 255  # (3, H, W), 0 to 1

width, fx = 1920, 1450.0  # the shared PINHOLE camera's width and fx, from cameras.bin
fov_x = math.degrees(2 * math.atan(width / (2 * fx)))

out = model.infer(image, fov_x=fov_x)
depth = out["depth"].clone()    # (H, W), metres, +inf where the mask is false
normal = out["normal"].clone()  # (H, W, 3), unit normals, zeros where the mask is false
mask = out["mask"].clone()      # (H, W), bool, true where there is geometry
```

## The Python packages

### What the MoGe-2 path imports

Traced through `from moge.model.v2 import MoGeModel`, `from_pretrained()` and `infer()` at the pinned commit:

| Package | Why | Licence | On our image |
| --- | --- | --- | --- |
| torch | the whole model | BSD-3-Clause style, with NVIDIA's licence for its CUDA packages (its row in `docs/third-party-licences.md`) | yes, 2.7.1 for CUDA 12.8 |
| numpy | the depth-shift and focal solve | BSD-3-Clause, with bundled libraries (its row) | yes, from `uv.lock` |
| scipy | `scipy.signal`, imported at the top of `moge/utils/geometry_numpy.py`, and `scipy.optimize.least_squares` for the solve that every `infer()` call runs | BSD-3-Clause, with bundled libraries (its row) | yes, from `uv.lock` |
| opencv-python-headless | `cv2` is imported at the top of `geometry_numpy.py`. The MoGe-2 path calls none of its functions. | Apache-2.0, with FFmpeg under LGPL-2.1 in its wheels (its row) | yes, from `uv.lock` |
| utils3d_moge | `utils3d.pt.intrinsics_from_focal_center` and `utils3d.pt.depth_map_to_point_map`. MoGe imports it as `utils3d_moge`, or as the older `utils3d` when that fails. | MIT | no |
| huggingface_hub | `hf_hub_download`, imported at the top of `v2.py` | Apache-2.0 | no |

Our image's versions meet MoGe's bounds: Python 3.11 (MoGe wants 3.10 or later), torch 2.7.1 (`>=2.4`), numpy `>=2` and scipy `>=1.8`. A comment in MoGe's `pyproject.toml` suggests opencv-python-headless 4.10.0.84 if OpenCV gives trouble. Ours is 5.1 or later (our `pyproject.toml`), and the MoGe-2 path only imports it.

Beyond these:

- utils3d_moge 1.7, at its pinned commit, declares moderngl, numpy and scipy. It imports its submodules lazily, and the two torch functions MoGe-2 calls bring in only torch and `typing_extensions`, which torch already needs. So moderngl isn't imported on this path. moderngl 5.12.0 (MIT) needs glcontext (MIT).
- huggingface_hub 1.26.0, the version in MoGe's `uv.lock`, needs click, filelock, fsspec, hf-xet (on x86-64 and 64-bit ARM), httpx, packaging, pyyaml, tqdm and typing-extensions. In the same lock, httpx 0.28.1 needs anyio, certifi, httpcore and idna, and httpcore needs h11.
- torchvision is declared but isn't on this path. The training data loader imports it.
- xformers is optional. `moge/model/modules/dinov2/layers/swiglu_ffn.py` tries to import it and falls back to plain torch, and only the ViT-g encoder uses that layer. ViT-B's blocks use a plain MLP.

### What pyproject.toml declares

Installing the `moge` package with its dependencies brings all of these. MoGe's `requirements.txt` only installs the package itself, with `-e .`.

| Dependency | Pin | MoGe uses it for |
| --- | --- | --- |
| click | `>=8.3.1` | the `moge` command line |
| numpy, scipy | `>=2`, `>=1.8` | the geometry |
| opencv-python-headless | none | image work in the scripts and the data code |
| pillow | none | the scripts and the data code |
| matplotlib | none | colour maps, in `moge/utils/vis.py` |
| trimesh | `>=4.11` | the scripts' mesh export and viewer |
| huggingface-hub | none | `from_pretrained()`'s download |
| requests, tqdm | none | requests for `moge/utils/download.py`, a URL downloader, and tqdm for progress bars |
| torch, torchvision | `>=2.4`, `>=0.19` | the model. `[tool.uv.sources]` sends uv to PyTorch's cu130 index for them, and the README says plain pip ignores that. |
| utils3d_moge | git, `62f09d58509485564e24d5d9f6aac9ee9ebc0c37` | the geometry helpers |
| pipeline | git, `1c511390d90226c00c101f34b84df26a0f8789b4` | the training and evaluation data loaders. It has no dependencies, and installs a top-level module named `pipeline`, which is why ours is `splatpipe` (structure.md). |
| flex-gemm | git, `b2fadb29d41846c7981ade6801ffc689fae119cf` | MoGe-3's sparse refinement, through `moge/model/modules/flex_sparse_blocks.py`. It needs torch, triton (triton-windows on Windows) and filelock, and builds CUDA kernels only when `FLEX_GEMM_BUILD_CUDA` or `BUILD_CUDA` is set. The README says it's why macOS isn't supported. |
| gradio, starlette | `>=6.0`, none | the demo app. MoGe's lock resolves gradio 6.22.0, which pulls in fastapi, uvicorn, pandas, pydantic, httpx and more. |

The `train` extra adds accelerate, sympy, wandb, tensorboard and mlflow.

## Licences

### The code

- MoGe is MIT, "Copyright (c) Microsoft Corporation". Its `LICENSE` holds the MIT text and then the whole Apache-2.0 text, and `pyproject.toml` says `license = "MIT"`.
- MoGe's copy of DINOv2, everything under `moge/model/modules/dinov2/`, is Apache-2.0. Each file starts with Meta's header, "Copyright (c) Meta Platforms, Inc. and affiliates", which points to the licence in the repo's root `LICENSE`, the Apache text above. The README's licence section says the same. The repo has no NOTICE file.
- The demo app's viewer ships three.js files in `moge/utils/gradio_3d_viewer/assets/`: MIT, "Copyright © 2010-2024 three.js authors", with their own `LICENSE`. They're package data, so an install of `moge` carries them.
- The git dependencies are all MIT: utils3d-moge (Copyright (c) 2022 EasternJournalist), pipeline (Copyright (c) 2025 Ruicheng Wang) and FlexGEMM (Copyright (c) 2025 Jianfeng Xiang).

### The weights

- The model card at revision `ca5f0e07` is metadata only: `license: mit`, `base_model: facebook/dinov2-base` and `pipeline_tag: depth-estimation`. It has no text, so it says nothing about the training data, the intended use or the limits. The Hub tags the model as a fine-tune of `facebook/dinov2-base`.
- `facebook/dinov2-base`, the weights it was fine-tuned from, is Apache-2.0 on its own card.
- So the weights are MIT, from an Apache-2.0 base. rules.md allows both, and the two MoGe rows already in `docs/third-party-licences.md` say the same.

## Our rules that touch this

- MoGe and torch go only on the machine image, never in `pyproject.toml` or the laptop install (tech.md), and the packages only MoGe needs, like utils3d-moge and huggingface-hub, belong with it. The train step keeps torch, gsplat and fused-ssim in one module, `splatpipe/train/gpu.py`, which it imports only when it runs, and `tests/test_imports.py` checks that. The depth-priors step can keep MoGe's imports the same way, so `splatpipe` still imports on a laptop.
- `cloud/image.py` names the weights path, but nothing in `splatpipe/` may import from `cloud/` (python.md), so the step can't read the path from there. The depth-priors spec decides how the step finds the file, for example the same path in `config.toml`.
- The weights come only from the image, checked by sha256 when it's built and again by the smoke test. Load them by path, never by repo id, with `HF_HUB_OFFLINE=1`, and never call `init_weights()`, so a job downloads nothing.
- Every package the image installs gets its row in `docs/third-party-licences.md` in the same commit (rules.md). MoGe's code and weights have rows, and the code's row covers the three git dependencies. huggingface-hub has none yet, and nor do click, hf-xet, httpx, pyyaml, anyio, httpcore, idna and h11, which come in through it. Installing `moge` with its dependencies would also bring trimesh, matplotlib and gradio's whole tree, none of which the MoGe-2 path imports. One way to keep the list to what runs: install `moge` and `utils3d_moge` from their pinned commits with `--no-deps`, plus huggingface-hub and what it needs. The depth-priors spec decides.
- MoGe asks for `opencv-python-headless`, the one rules.md allows, never plain `opencv-python`. Frames still come from ffmpeg: the step decodes the camera solve's JPEGs, as the train step does, and never a video.
- Each spec that adds a GPU code path adds a check to `SMOKE_CHECKS` in `cloud/smoke_runner.py` (`docs/pipeline.md`, "The smoke test"). For MoGe, that check would load the weights and run `infer()` on a small image.
- `num_tokens` or `resolution_level`, whether to pass `fov_x`, `apply_mask` and any thresholds go in `config.toml` (python.md).
- The weights and any depth or normal maps stay in `data/` or on the machine, never in git (rules.md).
- Our step calls MoGe's API. If it ever adapts MoGe's code instead, the copy keeps MoGe's MIT notice, or Meta's Apache-2.0 header for anything from `moge/model/modules/dinov2/` (rules.md).

## Links

- MoGe at the pinned commit: <https://github.com/microsoft/MoGe/tree/74fbce054ebed49800de42d0ad0e83495065719a>
- README, with the checkpoint table, the minimal example and the licence section: <https://github.com/microsoft/MoGe/blob/74fbce054ebed49800de42d0ad0e83495065719a/README.md>
- Licence, MIT then Apache-2.0: <https://github.com/microsoft/MoGe/blob/74fbce054ebed49800de42d0ad0e83495065719a/LICENSE>
- `pyproject.toml`, `requirements.txt` and `uv.lock`: <https://github.com/microsoft/MoGe/blob/74fbce054ebed49800de42d0ad0e83495065719a/pyproject.toml>, <https://github.com/microsoft/MoGe/blob/74fbce054ebed49800de42d0ad0e83495065719a/requirements.txt> and <https://github.com/microsoft/MoGe/blob/74fbce054ebed49800de42d0ad0e83495065719a/uv.lock>
- `MoGeModel` for MoGe-2, with `from_pretrained()`, `forward()` and `infer()`: <https://github.com/microsoft/MoGe/blob/74fbce054ebed49800de42d0ad0e83495065719a/moge/model/v2.py>
- The encoder, and DINOv2's builders with the `pretrained` download: <https://github.com/microsoft/MoGe/blob/74fbce054ebed49800de42d0ad0e83495065719a/moge/model/modules/dinov2_encoder.py> and <https://github.com/microsoft/MoGe/blob/74fbce054ebed49800de42d0ad0e83495065719a/moge/model/modules/dinov2/hub/backbones.py>
- Where training calls `init_weights()`: <https://github.com/microsoft/MoGe/blob/74fbce054ebed49800de42d0ad0e83495065719a/moge/train/checkpoint.py>
- The optional xformers import: <https://github.com/microsoft/MoGe/blob/74fbce054ebed49800de42d0ad0e83495065719a/moge/model/modules/dinov2/layers/swiglu_ffn.py>
- The focal and shift solve: <https://github.com/microsoft/MoGe/blob/74fbce054ebed49800de42d0ad0e83495065719a/moge/utils/geometry_torch.py> and <https://github.com/microsoft/MoGe/blob/74fbce054ebed49800de42d0ad0e83495065719a/moge/utils/geometry_numpy.py>
- The inference script, with `--fov_x` and `--fp16`: <https://github.com/microsoft/MoGe/blob/74fbce054ebed49800de42d0ad0e83495065719a/moge/scripts/infer.py>
- The evaluation wrapper, from intrinsics to `fov_x`: <https://github.com/microsoft/MoGe/blob/74fbce054ebed49800de42d0ad0e83495065719a/baselines/moge.py>
- The normal map's colours, which show the sign convention: <https://github.com/microsoft/MoGe/blob/74fbce054ebed49800de42d0ad0e83495065719a/moge/utils/vis.py>
- Normal estimation: <https://github.com/microsoft/MoGe/blob/74fbce054ebed49800de42d0ad0e83495065719a/docs/normal.md>
- Training, its config, its data and the mask's targets: <https://github.com/microsoft/MoGe/blob/74fbce054ebed49800de42d0ad0e83495065719a/docs/train.md>, <https://github.com/microsoft/MoGe/blob/74fbce054ebed49800de42d0ad0e83495065719a/configs/train/v2.json>, <https://github.com/microsoft/MoGe/blob/74fbce054ebed49800de42d0ad0e83495065719a/moge/train/dataloader.py>, <https://github.com/microsoft/MoGe/blob/74fbce054ebed49800de42d0ad0e83495065719a/moge/train/train_moge12.py> and <https://github.com/microsoft/MoGe/blob/74fbce054ebed49800de42d0ad0e83495065719a/moge/train/losses.py>
- Where `pipeline`, `flex_gemm` and requests are imported: <https://github.com/microsoft/MoGe/blob/74fbce054ebed49800de42d0ad0e83495065719a/moge/test/dataloader.py>, <https://github.com/microsoft/MoGe/blob/74fbce054ebed49800de42d0ad0e83495065719a/moge/model/modules/flex_sparse_blocks.py> and <https://github.com/microsoft/MoGe/blob/74fbce054ebed49800de42d0ad0e83495065719a/moge/utils/download.py>
- The model card at the pinned revision: <https://huggingface.co/Ruicheng/moge-2-vitb-normal/blob/ca5f0e07ff01d3e5a364c1d954ed12ee1814b368/README.md>
- Its files, with `model.pt`'s size and sha256: <https://huggingface.co/Ruicheng/moge-2-vitb-normal/tree/ca5f0e07ff01d3e5a364c1d954ed12ee1814b368>
- The base model's card: <https://huggingface.co/facebook/dinov2-base>
- utils3d-moge at its pin, with its licence, the torch transforms (`intrinsics_from_focal_center`, `intrinsics_to_fov`) and maps (`depth_map_to_point_map`, `point_map_to_normal_map`): <https://github.com/EasternJournalist/utils3d-moge/tree/62f09d58509485564e24d5d9f6aac9ee9ebc0c37>, <https://github.com/EasternJournalist/utils3d-moge/blob/62f09d58509485564e24d5d9f6aac9ee9ebc0c37/utils3d_moge/torch/transforms.py> and <https://github.com/EasternJournalist/utils3d-moge/blob/62f09d58509485564e24d5d9f6aac9ee9ebc0c37/utils3d_moge/torch/maps.py>
- pipeline at its pin: <https://github.com/EasternJournalist/pipeline/tree/1c511390d90226c00c101f34b84df26a0f8789b4>
- FlexGEMM at its pin, with its `setup.py`: <https://github.com/JeffreyXiang/FlexGEMM/tree/b2fadb29d41846c7981ade6801ffc689fae119cf>
- moderngl and glcontext: <https://pypi.org/project/moderngl/5.12.0/> and <https://pypi.org/project/glcontext/3.0.0/>
- huggingface_hub 1.26.0 on PyPI, with its licence and dependencies: <https://pypi.org/project/huggingface-hub/1.26.0/>
- huggingface_hub's environment variables, `HF_HUB_OFFLINE` among them: <https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables>
- Where 1.26.0 reads `HF_HUB_OFFLINE`, validates repo ids and checks `hf_hub_download`'s arguments: <https://github.com/huggingface/huggingface_hub/blob/v1.26.0/src/huggingface_hub/constants.py>, <https://github.com/huggingface/huggingface_hub/blob/v1.26.0/src/huggingface_hub/utils/_validators.py> and <https://github.com/huggingface/huggingface_hub/blob/v1.26.0/src/huggingface_hub/file_download.py>
- `torch.load` and `weights_only`: <https://docs.pytorch.org/docs/2.7/generated/torch.load.html>
- Inference mode, and cloning inference tensors: <https://docs.pytorch.org/docs/2.7/notes/autograd.html#inference-mode> and <https://docs.pytorch.org/cppdocs/api/autograd/modes.html>
- COLMAP's pixel convention, under "Keypoints and Descriptors": <https://colmap.github.io/database.html>
- MoGe-2: Accurate Monocular Geometry with Metric Scale and Sharp Details: <https://arxiv.org/abs/2507.02546>
