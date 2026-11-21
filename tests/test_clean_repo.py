"""Checks on the repo's own files for the clean step. Each test reads the real file by path from
the repo root, and none runs the step.

- The real config.toml, read by path, because conftest.py points CONFIG_PATH at a test config.
  [clean] holds the sixteen Shared_Settings of Req 6.2, and each of its three Preset tables the
  two Preset_Settings of Req 6.3, with the values of Req 6.4. It loads for each Preset, and
  check_settings accepts it (Req 6.1 to 6.4, 6.14). max_source_splats fits in a quarter of the
  job machine's 8 GB (Req 6.17).
- The comments (Req 6.18): read as text, every key in [clean] and its Preset tables has a comment
  with text on the line directly above it. tests/test_config.py also accepts a comment at the end
  of a key's line, so this file checks the stricter rule itself, on statements split by that
  file's own split_statements.
- The imports, read with ast by tests/test_imports.py's own imports_of: no *.py under splatpipe/,
  found with rglob, imports a GPU package (Req 6.15), and every *.py in splatpipe/clean/ imports
  only the standard library, numpy, scipy and splatpipe, so nothing from cloud/ (Req 8.5). Each
  failure lists the file and the line of every import it found.
"""

import sys
import tomllib
from typing import Any

import pytest

from splatpipe.clean.settings import check_settings
from splatpipe.config import REPO_ROOT, load_config

from test_config import is_comment_line, split_statements
from test_imports import imports_of, top

SPLATPIPE = REPO_ROOT / "splatpipe"
CLEAN = SPLATPIPE / "clean"
# The real config, read by path, because conftest.py points CONFIG_PATH at a test config.
REAL_CONFIG = REPO_ROOT / "config.toml"

# The sixteen Shared_Settings of Req 6.2 with the values Req 6.4 gives them, written out here
# rather than taken from the code under test or the test config.
REAL_SHARED = {
    "train_ply": "train.ply",
    "clean_ply": "clean.ply",
    "clean_sog": "clean.sog",
    "min_opacity": 0.005,
    "faint_cube_divisions": 128,
    "faint_min_per_cube": 8,
    "faint_min_paint_share": 0.10,
    "extent_trim_percent": 1,
    "neighbours": 8,
    "isolation_ratio": 8.0,
    "max_removed_fraction": 0.70,
    "min_kept_splats": 10_000,
    "warn_kept_fraction": 0.5,
    "max_source_splats": 8_000_000,
    "max_workers": 4,
    "splat_transform_timeout_seconds": 3600,
}
# Each Preset's table: the two Preset_Settings of Req 6.3, with the values Req 6.4 gives them.
REAL_PRESETS = {
    "fast": {"sog_sh_bands": 3, "sh_iterations": 10},
    "default": {"sog_sh_bands": 3, "sh_iterations": 10},
    "quality": {"sog_sh_bands": 3, "sh_iterations": 20},
}
# The most Gaussians whose Splat_Data fits in a quarter of the job machine's 8 GB, 2 * 10**9
# bytes, at 62 float32 properties, or 248 bytes, a Gaussian: 8,064,516 (Req 6.17, A7).
MOST_SOURCE_SPLATS = 2 * 10**9 // 248
# The GPU packages that nothing under splatpipe/ may import, by top-level name (Req 6.15).
GPU_NAMES = frozenset({"torch", "torchvision", "gsplat", "cupy", "pycuda", "triton", "nvidia",
                       "cuda"})
# What a file in splatpipe/clean/ may import, by top-level name (Req 8.5). cloud isn't one.
CLEAN_MAY_IMPORT = sys.stdlib_module_names | {"numpy", "scipy", "splatpipe"}


def real_clean_section() -> dict[str, Any]:
    """[clean] of the real config.toml, its Preset tables included, as tomllib reads it."""
    return tomllib.loads(REAL_CONFIG.read_text(encoding="utf-8"))["clean"]


def uncommented_clean_keys(text: str) -> list[tuple[int, str]]:
    """Each key of [clean] and its Preset tables in the TOML text whose line directly above isn't
    a # comment with text, as (line, key), with lines counted from 1 (Req 6.18).

    The text is split into statements as tests/test_config.py splits it, and a header's section
    is the first key it parses to, as there, so [clean.fast] gives clean. Unlike that file's
    check, a comment at the end of the key's own line doesn't count, and nor does one with a
    blank line between it and the key. Keys in other sections aren't read.
    """
    statements = split_statements(text)
    missing: list[tuple[int, str]] = []
    section: str | None = None
    for above, statement in zip([None, *statements], statements):
        if statement.lines[0].lstrip().startswith("["):
            section = next(iter(statement.parsed))
        elif statement.parsed and section == "clean" and not (
                above is not None and is_comment_line(above)):
            missing.append((statement.line, next(iter(statement.parsed))))
    return missing


# The real config.toml

def test_the_real_clean_section_holds_the_sixteen_shared_and_the_two_preset_settings() -> None:
    """Req 6.1 to 6.4. Read as tomllib reads it, [clean] holds the sixteen Shared_Settings and its
    three Preset tables, and each Preset table holds the two Preset_Settings, all with the values
    of Req 6.4, and nothing else."""
    assert real_clean_section() == {**REAL_SHARED, **REAL_PRESETS}


@pytest.mark.parametrize("preset", list(REAL_PRESETS))
def test_the_real_clean_section_loads_for_each_preset_and_check_settings_accepts_it(
        preset: str) -> None:
    """Req 6.14. STEP_LIST is empty, so tests/test_imports.py doesn't load this section."""
    settings = load_config(REAL_CONFIG).step_settings("clean", preset)
    assert settings == {**REAL_SHARED, **REAL_PRESETS[preset]}
    check_settings(settings)  # raises nothing


def test_the_real_max_source_splats_fits_in_a_quarter_of_8_gb() -> None:
    """Req 6.17."""
    assert real_clean_section()["max_source_splats"] <= MOST_SOURCE_SPLATS


# The comments

@pytest.mark.parametrize(("text", "missing"), [
    pytest.param("[clean]\n# How many neighbours to look at.\nneighbours = 8\n", [],
                 id="a comment directly above"),
    pytest.param("[clean]\nneighbours = 8  # How many neighbours to look at.\n",
                 [(2, "neighbours")], id="a comment at the end of the line"),
    pytest.param("[clean.fast]\n# Iterations of the compression.\n\nsh_iterations = 10\n",
                 [(4, "sh_iterations")], id="a comment two lines above"),
    pytest.param("[clean.quality]\n#\nsh_iterations = 20\n", [(3, "sh_iterations")],
                 id="a # with no text"),
])
def test_only_a_comment_with_text_on_the_line_directly_above_a_clean_key_counts(
        text: str, missing: list[tuple[int, str]]) -> None:
    assert uncommented_clean_keys(text) == missing


def test_every_key_in_the_real_clean_tables_has_a_comment_on_the_line_directly_above() -> None:
    """Req 6.18."""
    assert uncommented_clean_keys(REAL_CONFIG.read_text(encoding="utf-8")) == []


# The imports

def test_nothing_under_splatpipe_imports_a_gpu_package() -> None:
    """Req 6.15. Every *.py under splatpipe/, found with rglob, so splatpipe/clean/ and every step
    folder added later count too."""
    found = [(where, name) for path in sorted(SPLATPIPE.rglob("*.py"))
             for where, name in imports_of(path) if top(name) in GPU_NAMES]
    assert found == []


def test_splatpipe_clean_imports_only_the_standard_library_numpy_scipy_and_splatpipe() -> None:
    """Req 8.5. cloud isn't in the standard library, so an import from cloud/ fails this too."""
    found = [(where, name) for path in sorted(CLEAN.rglob("*.py"))
             for where, name in imports_of(path) if top(name) not in CLEAN_MAY_IMPORT]
    assert found == []
