# Third-party licences

Every dependency, tool, model weight and dataset we use, with its licence and how we use it. Whoever adds one adds its line here in the same commit. The licence rules are in `.kiro/steering/rules.md`.

"Not yet" means it isn't in `uv.lock`, a `package-lock.json` or the machine image yet.

| Name | Licence | How we use it | Installed |
|---|---|---|---|
| pytest | MIT. Its dependencies are BSD-3-Clause (colorama), MIT (iniconfig, pluggy), Apache-2.0 or BSD-2-Clause (packaging) and BSD-2-Clause (Pygments). | Runs the tests. Dev group only. | Yes |
| Hypothesis | MPL-2.0. Its dependency sortedcontainers is Apache-2.0. | Property tests. Dev group, tests only. | Yes |
| boto3 | Apache-2.0 | Talks to AWS, only inside `cloud/`. | Not yet |
| ffmpeg and ffprobe | LGPL or GPL, depending on the build | ffprobe reads the video's container, streams and length when a Job is made. ffmpeg extracts the frames. Both are separate command line tools, on our private machine image and on laptops for the tests, never vendored or linked. | Yes, on the laptop. Not yet on the machine image. |
| COLMAP 4.2.1 | BSD-3-Clause | Command line tool on our private machine image. Finds the camera poses. Built without CUDA, OpenGL, the GUI, CGAL and LSD. | Not yet |
| gsplat 1.5.3 | Apache-2.0 | Trains the splat. Cloud image only. | Not yet |
| fused-ssim | MIT | SSIM loss during training. Cloud image only. | Not yet |
| MoGe code | MIT, except its DINOv2 code, which is Apache-2.0. Its git dependencies utils3d-moge, pipeline and FlexGEMM are all MIT. | The depth model. Cloud image only. | Not yet |
| MoGe-2 ViT-B normal weights (`Ruicheng/moge-2-vitb-normal`) | MIT | Weights for the depth model. Cloud image only, never in git. | Not yet |
| numpy | BSD-3-Clause, with parts under 0BSD, MIT, Zlib and CC0. Its wheels bundle OpenBLAS and LAPACK (BSD-3-Clause and its Open MPI variant) and the GCC runtime library, such as libgfortran (GPL-3.0-or-later with the GCC runtime library exception). The Linux wheels also bundle libquadmath (LGPL-2.1-or-later), and the Windows wheels bundle Microsoft's Visual C++ runtime (`msvcp140.dll`, under Microsoft's licence). | CPU maths. | Yes |
| scipy | BSD-3-Clause. Its wheels bundle OpenBLAS and LAPACK (BSD-3-Clause and its Open MPI variant) and the GCC runtime library, such as libgfortran (GPL-3.0-or-later with the GCC runtime library exception). The Linux wheels also bundle libquadmath (LGPL-2.1-or-later). | CPU maths. | Yes |
| opencv-python-headless | Apache-2.0. Its wheels ship FFmpeg under LGPL-2.1. | CPU maths. Never used to read or write video. | Not yet |
| torch | BSD-3-Clause style, plus bundled code under Apache-2.0 (also with the LLVM exception), BSD-2-Clause, BSL-1.0, ISC, MIT and Zlib. Its `nvidia-*` and `cuda-*` packages are under NVIDIA's licence. | Runs gsplat and the depth model. Cloud image only. | Not yet |
| certifi | MPL-2.0 | Used unmodified. Comes in through requests. | Not yet |
| tqdm | MPL-2.0 and MIT | Used unmodified. Comes in through huggingface-hub. | Not yet |
| PlayCanvas splat-transform 3.8.0 (`@playcanvas/splat-transform`) | MIT. Its npm packages are MIT, BSD-3-Clause and ISC. | Makes the small copy of the splat for the web viewer. Node only. | Not yet |
| git | GPL-2.0 | Separate command line tool that the hooks and people run. Never vendored, linked or shipped. | Yes, on the laptop |
| npm | Artistic-2.0 | Separate command line tool that installs splat-transform's packages. Never vendored, linked or shipped. | Yes, on the laptop |
| AWS CLI v2 | Apache-2.0 | Separate command line tool that Kiro and people run. Never vendored, linked or shipped. | Yes, on the laptop |
| uv | MIT or Apache-2.0 | Separate command line tool that runs the hooks and manages the Python environment. Never vendored, linked or shipped. | Yes, on the laptop |
