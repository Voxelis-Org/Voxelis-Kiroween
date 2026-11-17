---
inclusion: always
---

# Structure

- `pipeline/`: the pipeline. Each step gets its own subfolder when its spec is built. Don't make step folders before that.
- `cloud/`: only the AWS code.
- `scripts/`: helper scripts.
- `tests/`: all the tests.
- `docs/`: `devlog.md` and `kiro-usage.md`. We fill them as we go, for the write-up.
- `data/` (at the root, gitignored): every video, frame, splat and weight goes here, never anywhere else in the repo.
- `config.toml` (at the root): every number and setting, read with `tomllib`.
- `.kiro/`: steering, specs, agents and hooks. Never gitignore it, the judges need to see it.
