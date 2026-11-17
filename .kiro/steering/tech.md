---
inclusion: always
---

# Tech stack

- Python 3.11, managed with uv. Run things with `uv run`, add packages with `uv add`.
- pytest for tests, Hypothesis for property tests. Hypothesis stays in the dev group because it's a test dependency only.
- numpy, scipy and opencv for the CPU maths.
- ffmpeg and COLMAP 4.2.1 as command line tools on the server.
- gsplat 1.5.3 for training.
- The GPU packages (torch, gsplat, the depth model) only go in the cloud machine image. Never add them to `pyproject.toml` or the laptop install.
- AWS EC2 for the cloud: one GPU machine per job, started from our own machine image. The job files live in S3.
- boto3 only inside `cloud/`.
- Node only for PlayCanvas splat-transform.
- Settings live in `config.toml` at the root, read with Python's built-in `tomllib`.
