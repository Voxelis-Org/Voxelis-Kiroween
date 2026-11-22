---
inclusion: always
---

# Tech stack

- Python 3.11, managed with uv. Run things with `uv run`, add packages with `uv add`.
- pytest for tests, Hypothesis for property tests. Hypothesis stays in the dev group because it's a test dependency only.
- numpy, scipy and OpenCV for the CPU maths. OpenCV is always `opencv-python-headless`, never plain `opencv-python`. We never use cv2 to read or write video.
- ffmpeg as a command line tool on the server. Frames always come from the ffmpeg command line.
- COLMAP 4.2.1 as a command line tool on the server, built from source with `-DCUDA_ENABLED=OFF -DOPENGL_ENABLED=OFF -DGUI_ENABLED=OFF -DCGAL_ENABLED=OFF -DLSD_ENABLED=OFF`. The first three keep SiftGPU (non-commercial) out. The last two keep out CGAL (GPL) and LSD (AGPL), which we don't use: we do no Delaunay meshing and no line detection. Two more flags: `-DONNX_ENABLED=OFF` leaves out the learned features and the ONNX Runtime binaries they'd download, while SIFT stays, and `-DBLA_VENDOR=OpenBLAS` builds with OpenBLAS's OpenMP build in place of Intel's MKL, whose licence isn't on our list.
- SIFT runs on the CPU. Still pass `--FeatureExtraction.use_gpu 0` to `colmap feature_extractor` and `--FeatureMatching.use_gpu 0` to whichever matcher we run.
- gsplat 1.5.3 for training.
- The GPU packages (torch, gsplat, the depth model) only go in the cloud machine image. Never add them to `pyproject.toml` or the laptop install. That includes torch's CUDA wheels and the NVIDIA CUDA libraries that come with them.
- AWS EC2 for the cloud: one GPU machine per job, started from our own machine image. The job files live in S3.
- The GPU quota is per account and region. Ours allows 32 vCPUs of G instances (the GPU machines) at a time in us-east-1. Admission is by that real G quota, read from Service Quotas: the start command adds up the vCPUs of our running machines (jobs, smoke tests and image builds, found by our tag) and launches when the chosen type still fits the quota, so more than one of our machines may run at once when the quota allows. It's one machine per job, not one machine in the whole account. Fast and default jobs run on a g5.8xlarge, with 32 vCPUs on 16 cores and one NVIDIA A10G, so under a 32-vCPU quota one of those runs at a time while two 16-vCPU fallback machines could run side by side. A new AWS account starts at 0 and has to ask AWS for more before any job can start.
- boto3 only inside `cloud/`.
- Node only for PlayCanvas splat-transform.
- Settings live in `config.toml` at the root, read with Python's built-in `tomllib`.
