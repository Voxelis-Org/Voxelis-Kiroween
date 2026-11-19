---
inclusion: always
---

# Structure

- `splatpipe/`: the pipeline. Each step gets its own subfolder when its spec is built. Don't make step folders before that. It isn't called `pipeline/` because MoGe (the depth model) depends on a package that installs a top-level `pipeline` module, which would clash with ours on the cloud machine.
- `cloud/`: only the AWS code.
- `scripts/`: helper scripts.
- `tests/`: all the tests.
- `docs/`: `devlog.md` and `kiro-usage.md`. We fill them as we go, for the write-up.
- `docs/third-party-licences.md`: a line for every dependency, tool, model weight and dataset, with its licence and how we use it.
- `data/` (at the root, gitignored): every video, frame, splat and weight goes here, never anywhere else in the repo.
- `config.toml` (at the root): every number and setting, read with `tomllib`.
- `.kiro/`: steering, specs, agents and hooks. Never gitignore it, the judges need to see it.
