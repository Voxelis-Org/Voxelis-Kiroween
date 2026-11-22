# Third-party licences

Every dependency, tool, model weight and dataset we use, with its licence and how we use it. Whoever adds one adds its line here in the same commit. The licence rules are in `.kiro/steering/rules.md`.

"Not yet" means it isn't in `uv.lock`, a `package-lock.json` or the machine image yet.

| Name | Licence | How we use it | Installed |
|---|---|---|---|
| pytest | MIT. Its dependencies are BSD-3-Clause (colorama), MIT (iniconfig, pluggy), Apache-2.0 or BSD-2-Clause (packaging) and BSD-2-Clause (Pygments). | Runs the tests. Dev group only. | Yes |
| Hypothesis | MPL-2.0. Its dependency sortedcontainers is Apache-2.0. | Property tests. Dev group, tests only. | Yes |
| cfn-lint | MIT-0. Its dependencies are MIT (PyYAML), BSD-3-Clause (jsonpatch, jsonpointer, networkx, sympy, mpmath), Apache-2.0 and CNRI-Python (regex) and PSF-2.0 (typing-extensions). | The template test in `tests/test_aws_setup.py` runs it on `cloud/aws-setup.yaml`. Dev group only. | Yes |
| boto3 | Apache-2.0 | Talks to AWS, only inside `cloud/`. | Yes |
| botocore | Apache-2.0 | The low-level core boto3 is built on. Comes in through boto3, only inside `cloud/`. | Yes |
| s3transfer | Apache-2.0 | Manages S3 uploads and downloads for boto3. Comes in through boto3, only inside `cloud/`. | Yes |
| jmespath | MIT | Filters AWS responses for boto3. Comes in through boto3, only inside `cloud/`. | Yes |
| python-dateutil | Apache-2.0 and BSD-3-Clause (dual-licensed) | Parses AWS timestamps for botocore. Comes in through botocore, only inside `cloud/`. | Yes |
| six | MIT | Python 2/3 compatibility shim. Comes in through python-dateutil, only inside `cloud/`. | Yes |
| urllib3 | MIT | The HTTP client botocore uses to reach AWS. Comes in through botocore, only inside `cloud/`. | Yes |
| AWS Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 24.04) | Amazon-owned. An AWS-provided AMI, used under AWS's terms. | The base image the Image_Build launches its builder machine from. Used only to build our own private Machine_Image on EC2; never redistributed, shared or made public. | Not yet |
| COLMAP 4.2.1 | BSD-3-Clause | Command line tool on our private machine image. Finds the camera poses. Built from source with `-DCUDA_ENABLED=OFF -DOPENGL_ENABLED=OFF -DGUI_ENABLED=OFF -DCGAL_ENABLED=OFF -DLSD_ENABLED=OFF`: the first three keep SiftGPU (non-commercial) out, `-DCGAL_ENABLED=OFF` keeps out CGAL (GPL) and `-DLSD_ENABLED=OFF` keeps out LSD (AGPL). SIFT runs on the CPU. Never vendored, linked or shipped. | Not yet |
| ffmpeg and ffprobe | LGPL or GPL, depending on the build | ffprobe reads the video's container, streams and length when a Job is made. ffmpeg decodes the frames of the Used_Part and tone maps HDR video with `zscale` and `tonemap` when the ffmpeg build has both filters. Both are separate command line tools, on our private machine image and on laptops for the tests, never vendored or linked. | Yes, on the laptop. Not yet on the machine image. |
| zimg (libzimg) | WTFPL, version 2 | ffmpeg's `zscale` filter uses it in the HDR tone map, inside the ffmpeg command line tool on our private machine image. Never vendored or linked into our code. | Not yet |
| gsplat 1.5.3 | Apache-2.0 | Trains the splat. Cloud image only. | Not yet |
| fused-ssim | MIT | SSIM loss during training. Cloud image only. | Not yet |
| MoGe code | MIT, except its DINOv2 code, which is Apache-2.0. Its git dependencies utils3d-moge, pipeline and FlexGEMM are all MIT. | The depth model. Cloud image only. | Not yet |
| MoGe-2 ViT-B normal weights (`Ruicheng/moge-2-vitb-normal`) | MIT | Weights for the depth model. Cloud image only, never in git. | Not yet |
| numpy | BSD-3-Clause, with parts under 0BSD, MIT, Zlib and CC0. Its wheels bundle OpenBLAS and LAPACK (BSD-3-Clause and its Open MPI variant) and the GCC runtime library, such as libgfortran (GPL-3.0-or-later with the GCC runtime library exception). The Linux wheels also bundle libquadmath (LGPL-2.1-or-later), and the Windows wheels bundle Microsoft's Visual C++ runtime (`msvcp140.dll`, under Microsoft's licence). | CPU maths. | Yes |
| scipy | BSD-3-Clause. Its wheels bundle OpenBLAS and LAPACK (BSD-3-Clause and its Open MPI variant) and the GCC runtime library, such as libgfortran (GPL-3.0-or-later with the GCC runtime library exception). The Linux wheels also bundle libquadmath (LGPL-2.1-or-later). | CPU maths. | Not yet |
| opencv-python-headless | Apache-2.0. Its wheels ship FFmpeg under LGPL-2.1. | The Laplacian, histogram equalisation, and JPEG and PNG encoding of frames that ffmpeg has already decoded. It never reads or writes video. | Yes |
| torch | BSD-3-Clause style, plus bundled code under Apache-2.0 (also with the LLVM exception), BSD-2-Clause, BSL-1.0, ISC, MIT and Zlib. Its `nvidia-*` and `cuda-*` packages are under NVIDIA's licence. | Runs gsplat and the depth model. Cloud image only. | Not yet |
| certifi | MPL-2.0 | Used unmodified. Comes in through requests. | Not yet |
| tqdm | MPL-2.0 and MIT | Used unmodified. Comes in through huggingface-hub. | Not yet |
| PlayCanvas splat-transform 3.8.0 (`@playcanvas/splat-transform`) | MIT. Its npm packages are MIT, BSD-3-Clause and ISC. | Makes the small copy of the splat for the web viewer. Node only. | Not yet |
| Node | MIT. Its distribution bundles components under other permissive licences that Node's own licence file lists (among them V8 under BSD-3-Clause, libuv under MIT, OpenSSL under Apache-2.0, zlib under Zlib and ICU under the Unicode licence). | Runs PlayCanvas splat-transform to make the small web-viewer copy of the splat. Cloud image only. | Not yet |
| git | GPL-2.0 | Separate command line tool that the hooks and people run. Never vendored, linked or shipped. | Yes, on the laptop |
| npm | Artistic-2.0 | Separate command line tool that installs splat-transform's packages. Never vendored, linked or shipped. | Yes, on the laptop |
| AWS CLI v2 | Apache-2.0 | Separate command line tool that Kiro and people run. Never vendored, linked or shipped. | Yes, on the laptop |
| awslabs.aws-iac-mcp-server | Apache-2.0 | Kiro runs its cfn-guard compliance check on `cloud/aws-setup.yaml` through the aws-infrastructure-as-code power. Our code doesn't import or ship it. | Yes, on the laptop, through the Kiro power |
| AWS Session Manager plugin | Apache-2.0 | Separate command line tool that people run with the AWS CLI to open a shell on a job machine. Never vendored, linked or shipped. | Not yet |
| uv | MIT or Apache-2.0 | Separate command line tool that runs the hooks and manages the Python environment. Never vendored, linked or shipped. | Yes, on the laptop |
