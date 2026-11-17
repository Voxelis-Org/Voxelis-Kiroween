---
inclusion: fileMatch
fileMatchPattern: "**/*.py"
---

# Python

- Every pipeline step is a plain function plus a small CLI, so it can be run on its own (`uv run python -m pipeline.<step>`).
- Nothing in `pipeline/` imports from `cloud/`, boto3 or any other AWS code.
- Every number comes from `config.toml`, read with `tomllib`. Don't hard-code thresholds, sizes, counts or time limits.
- Use type hints.
- Keep functions small.
- Write error messages a normal user can read: what went wrong and what to do about it.
