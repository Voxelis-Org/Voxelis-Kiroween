---
inclusion: fileMatch
fileMatchPattern: "tests/**"
---

# Tests

- CPU only. No GPU, no network.
- Tiny synthetic data only.
- Use Hypothesis property tests whenever a rule should hold for every input.
- Tests that need ffmpeg or COLMAP are skipped when the tool isn't installed, for example with `pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is not installed")`.
- Run them with `uv run pytest`.
