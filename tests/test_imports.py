"""The import checks (Req 12): they read the Pipeline's files with ast and run no Step (Req 12.5).

`imports_in` collects every module name a file imports, with its line: each `import`, each
`from ... import`, and each `import_module` or `__import__` call whose first argument is a
constant string, anywhere in the file, inside functions too. A relative name is resolved against
the file's own package first, so `from .step import run_step` in splatpipe/run.py reads as
splatpipe.step.run_step. A from-import reads as its module plus each imported name, because
`from splatpipe import fake` imports the module splatpipe.fake.

- No AWS (Req 12.1): no *.py under splatpipe/, found with rglob so that step folders count too,
  imports cloud, boto3, botocore, s3transfer, awscrt, awscli, aiobotocore or aioboto3, or a
  submodule of one.
- Standard library only (Req 12.3, 11.1): splatpipe/*.py and splatpipe/fake/ import only names
  in sys.stdlib_module_names and splatpipe, and cloud/spend.py only the standard library. The
  Skeleton is read with glob, not rglob, so a step folder like splatpipe/frames/ can import
  numpy, scipy and OpenCV.
- No Step module on import (Req 12.4): each splatpipe.X that a splatpipe/*.py imports is a
  Skeleton module, a splatpipe/X.py, and a fresh process that imports splatpipe.run has only
  Skeleton modules among its splatpipe.* modules.
- The Step_List (Req 12.2, 8.2, 5.11): each STEP_LIST entry is splatpipe.<name> with a good
  Step_Name, isn't the Fake_Step, and its Step_Section loads from the real config for each Preset.

Each failure lists the file and the line of every import it found.
"""

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

from splatpipe.config import PRESETS, REPO_ROOT, load_config
from splatpipe.run import STEP_LIST
from splatpipe.step import check_step_name

SPLATPIPE = REPO_ROOT / "splatpipe"
# The real config, read by path, because conftest.py points CONFIG_PATH at a test config.
REAL_CONFIG = REPO_ROOT / "config.toml"
# The AWS code that nothing in splatpipe/ may import, by top-level name (Req 12.1).
AWS_NAMES = frozenset({"cloud", "boto3", "botocore", "s3transfer", "awscrt", "awscli",
                       "aiobotocore", "aioboto3"})
# The Skeleton's modules: X for each splatpipe/X.py.
SKELETON_MODULES = frozenset(path.stem for path in SPLATPIPE.glob("*.py")) - {"__init__"}
# The calls that import the module their first argument names.
IMPORT_CALLS = frozenset({"import_module", "__import__"})
# Run in a fresh process, because this one has imported more of splatpipe through conftest.py.
LOADED_BY_RUN = ("import json, sys, splatpipe.run; "
                 "print(json.dumps(sorted(n for n in sys.modules if n.startswith('splatpipe.'))))")


def absolute(name: str, package: str) -> str:
    """`name` with its leading dots, if it has any, resolved against `package`, as Python does."""
    rest = name.lstrip(".")
    level = len(name) - len(rest)
    if level == 0:
        return name
    base = package.rsplit(".", level - 1)[0]
    return f"{base}.{rest}" if rest else base


def constant_import(node: ast.AST) -> str | None:
    """The constant string that an import_module or __import__ call names, or None."""
    if not isinstance(node, ast.Call) or not node.args:
        return None
    if isinstance(node.func, ast.Name):
        called = node.func.id
    elif isinstance(node.func, ast.Attribute):
        called = node.func.attr
    else:
        return None
    first = node.args[0]
    if called in IMPORT_CALLS and isinstance(first, ast.Constant) and isinstance(first.value, str):
        return first.value
    return None


def imports_in(source: str | bytes, package: str) -> list[tuple[int, str]]:
    """Every module name that `source`, a file in `package`, imports, as (line, name), in order."""
    found: list[tuple[int, str]] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            found += [(node.lineno, alias.name) for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            module = absolute("." * node.level + (node.module or ""), package)
            found += [(node.lineno, f"{module}.{alias.name}") for alias in node.names]
        elif (name := constant_import(node)) is not None:
            found.append((node.lineno, absolute(name, package)))
    return sorted(found)


def imports_of(path: Path) -> list[tuple[str, str]]:
    """Every module name that the file at `path` imports, as ("<file>:<line>", name).

    The file's package is the dotted path of its folder, like splatpipe.fake for both files in
    splatpipe/fake/. The file is read as bytes, so ast decodes it as Python would.
    """
    package = ".".join(path.parent.relative_to(REPO_ROOT).parts)
    shown = path.relative_to(REPO_ROOT).as_posix()
    return [(f"{shown}:{line}", name) for line, name in imports_in(path.read_bytes(), package)]


def top(name: str) -> str:
    """The top-level package of a dotted module name."""
    return name.partition(".")[0]


def stdlib_only_files() -> list[tuple[Path, bool]]:
    """The files that may import only the standard library, each with whether it may import
    splatpipe too: the Skeleton, read with glob and not rglob, the Fake_Step and the Spend_Guard.
    """
    pipeline = sorted(SPLATPIPE.glob("*.py")) + sorted(SPLATPIPE.glob("fake/**/*.py"))
    return [(path, True) for path in pipeline] + [(REPO_ROOT / "cloud" / "spend.py", False)]


def test_the_reader_finds_every_kind_of_import_with_its_line() -> None:
    """Imports inside functions count, relative names are resolved against the package, and an
    import_module call whose name isn't a constant string isn't read."""
    sample = ("import os.path\n"
              "from boto3.session import Session\n"
              "from . import config\n"
              "from ..step import run_step\n"
              "\n"
              "def later(name):\n"
              "    import cloud.spend\n"
              "    importlib.import_module('botocore.exceptions')\n"
              "    __import__('awscrt')\n"
              "    import_module('.helper')\n"
              "    importlib.import_module(name)\n")
    assert imports_in(sample, "splatpipe.fake") == [
        (1, "os.path"), (2, "boto3.session.Session"), (3, "splatpipe.fake.config"),
        (4, "splatpipe.step.run_step"), (7, "cloud.spend"), (8, "botocore.exceptions"),
        (9, "awscrt"), (10, "splatpipe.fake.helper")]


def test_nothing_under_splatpipe_imports_aws_code() -> None:
    """Every *.py under splatpipe/, found with rglob, so a step folder added later counts too."""
    found = [(where, name) for path in sorted(SPLATPIPE.rglob("*.py"))
             for where, name in imports_of(path) if top(name) in AWS_NAMES]
    assert found == []


def test_the_skeleton_fake_step_and_spend_guard_import_only_the_standard_library() -> None:
    found: list[tuple[str, str]] = []
    for path, splatpipe_too in stdlib_only_files():
        allowed = sys.stdlib_module_names | ({"splatpipe"} if splatpipe_too else set())
        found += [(where, name) for where, name in imports_of(path) if top(name) not in allowed]
    assert found == []


def test_the_skeleton_imports_no_step_package() -> None:
    """Each splatpipe.X that a splatpipe/*.py imports, `from splatpipe import X` included,
    names a splatpipe/X.py."""
    found = [(where, name) for path in sorted(SPLATPIPE.glob("*.py"))
             for where, name in imports_of(path)
             if name.startswith("splatpipe.") and name.split(".")[1] not in SKELETON_MODULES]
    assert found == []


def test_importing_splatpipe_run_loads_only_skeleton_modules() -> None:
    result = subprocess.run([sys.executable, "-c", LOADED_BY_RUN], cwd=REPO_ROOT,
                            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
                            capture_output=True, encoding="utf-8", timeout=60)
    assert result.returncode == 0, result.stderr
    loaded = json.loads(result.stdout)
    assert "splatpipe.run" in loaded
    assert [name for name in loaded
            if name.removeprefix("splatpipe.") not in SKELETON_MODULES] == []


def test_each_step_list_entry_is_a_step_inside_splatpipe_with_its_config_section() -> None:
    """STEP_LIST starts empty, so this loads no Step_Section until a step spec adds its entry."""
    config = load_config(REAL_CONFIG)
    for entry in STEP_LIST:
        assert check_step_name(entry.name) == entry.name
        assert entry.module == f"splatpipe.{entry.name}"  # Req 12.2
        assert entry.module != "splatpipe.fake"  # the Fake_Step never runs on real Jobs (Req 8.2)
        for preset in PRESETS:
            config.step_settings(entry.name, preset)  # Req 5.11
