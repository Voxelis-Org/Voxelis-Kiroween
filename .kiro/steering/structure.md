---
inclusion: always
---

# Structure

- `splatpipe/`: the pipeline. Each step gets its own subfolder when its spec is built. Don't make step folders before that. It isn't called `pipeline/` because MoGe (the depth model) depends on a package that installs a top-level `pipeline` module, which would clash with ours on the cloud machine.
- `cloud/`: only the AWS code, plus `aws-setup.yaml`, the one CloudFormation template that makes our AWS resources.
- `app/`: the website, with its own App_Project (`app/pyproject.toml` and `app/uv.lock`, boto3 only) and the 3D viewer in `app/viewer/`. The website's packages never go in the root `pyproject.toml` or `uv.lock`.
- `scripts/`: helper scripts.
- `tests/`: all the tests.
- `docs/`: `devlog.md` and `kiro-usage.md`, which we fill as we go for the write-up, `pipeline.md`, and `aws-setup.md`, the guide to setting up AWS.
- `docs/pipeline.md`: one section per step, with what it reads, what it writes, its config section and its report fields. Every step spec updates its own section as part of its tasks, so the steps the three of us build keep fitting together.
- `docs/third-party-licences.md`: a line for every dependency, tool, model weight and dataset, with its licence and how we use it.
- `data/` (at the root, gitignored): every video, frame, splat and weight goes here, never anywhere else in the repo.
- `config.toml` (at the root): every number and setting, read with `tomllib`.
- `.kiro/`: steering, specs, agents and hooks. Never gitignore it, the judges need to see it.
