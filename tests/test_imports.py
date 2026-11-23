"""The import checks (Req 12) and the dependency check: they read the Pipeline's files with ast,
the GPU_Module's as text too, and pyproject.toml and uv.lock with tomllib, and run no Step
(Req 12.5).

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
  numpy and OpenCV.
- No Step module on import (Req 12.4): each splatpipe.X that a splatpipe/*.py imports is a
  Skeleton module, a splatpipe/X.py, and a fresh process that imports splatpipe.run has only
  Skeleton modules among its splatpipe.* modules.
- The Step_List (Req 12.2, 8.2, 5.11): each STEP_LIST entry is splatpipe.<name> with a good
  Step_Name, isn't the Fake_Step, and its Step_Section loads from the real config for each Preset.
- The Frames_Step's imports (frames spec, Req 9.9, 13.10): every splatpipe/frames/*.py imports
  only names in sys.stdlib_module_names, numpy, cv2 and splatpipe, so torch, gsplat, boto3 and
  cloud/ stay out.
- The Train_Step's imports (splat-training spec, Req 9.11, 9.13, 14.7): every
  splatpipe/train/*.py but gpu.py imports only names in sys.stdlib_module_names, numpy, cv2 and
  splatpipe, and gpu.py, the GPU_Module, imports only those and torch, gsplat and fused_ssim, so
  boto3 and cloud/ stay out of all of them. The files are found with glob, so each new file is
  checked as soon as it lands.
- The GPU_Module (splat-training spec, Req 2.2, 4.14, 4.15, 8.2, 9.11, 9.12, 14.7): gpu.py is
  read as text and with ast and never imported, because torch, gsplat and fused_ssim exist only
  on the Machine_Image. It starts with the Apache-2.0 note that the gsplat-training skill gives
  for code adapted from gsplat's example trainer, with the note's copyright and licence lines
  before its first import, and its text, comments and docstrings included, names none of
  rasterization_inria_wrapper, rasterization_2dgs_inria_wrapper and export_splats. It's the only
  splatpipe/train/*.py that imports torch, gsplat or fused_ssim, and it imports all three.
  gpu_api.GPU_MODULE, the module load_gpu imports, names that file. A fresh process that imports
  splatpipe.train and each of its other modules, found with glob, holds none of torch, gsplat
  and fused_ssim in sys.modules.
- The dependencies (frames spec, Req 9.8, and splat-training spec, Req 9.10): numpy and
  opencv-python-headless are runtime dependencies in pyproject.toml and in the project's own
  entry in uv.lock, and no package in either file is plain opencv-python, torch, gsplat,
  fused-ssim or an nvidia-* or cuda-* package. Names are compared as PEP 503 normalises them.
  The check neither requires nor refuses scipy, so the clean spec can decide scipy without
  changing it.

Each import check's failure lists the file and the line of every import it found, and the
dependency check's failure lists the file of every package it found.
"""

import ast
import json
import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

from splatpipe.config import PRESETS, REPO_ROOT, load_config
from splatpipe.run import STEP_LIST
from splatpipe.step import check_step_name
from splatpipe.train.gpu_api import GPU_MODULE

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
# The top-level names a splatpipe/frames/*.py may import besides the standard library (frames
# spec, Req 9.9).
FRAMES_IMPORTS = frozenset({"numpy", "cv2", "splatpipe"})
# The top-level names a splatpipe/train/*.py may import besides the standard library, and the
# GPU_Module's file, which may import torch, gsplat and fused_ssim too (splat-training spec,
# Req 9.11).
TRAIN_IMPORTS = frozenset({"numpy", "cv2", "splatpipe"})
GPU_MODULE_FILE = SPLATPIPE / "train" / "gpu.py"
GPU_MODULE_SHOWN = GPU_MODULE_FILE.relative_to(REPO_ROOT).as_posix()  # as imports_of shows it
# The GPU packages, by the top-level names the GPU_Module imports them as.
GPU_NAMES = frozenset({"torch", "gsplat", "fused_ssim"})
GPU_MODULE_IMPORTS = TRAIN_IMPORTS | GPU_NAMES
# The licence note that the gsplat-training skill gives for code adapted from gsplat's
# examples/simple_trainer.py, which gpu.py starts with (splat-training spec, Req 4.15).
GSPLAT_NOTE = (
    "# Adapted from gsplat v1.5.3, examples/simple_trainer.py",
    "# https://github.com/nerfstudio-project/gsplat/blob/v1.5.3/examples/simple_trainer.py",
    "#",
    "# Copyright 2025 Nerfstudio Team",
    "#",
    '# Licensed under the Apache License, Version 2.0 (the "License");',
    "# you may not use this file except in compliance with the License.",
    "# You may obtain a copy of the License at",
    "#",
    "#     http://www.apache.org/licenses/LICENSE-2.0",
    "#",
    "# Unless required by applicable law or agreed to in writing, software",
    '# distributed under the License is distributed on an "AS IS" BASIS,',
    "# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.",
    "# See the License for the specific language governing permissions and",
    "# limitations under the License.",
)
# The gsplat functions gpu.py never names, comments and docstrings included: the two wrappers run
# Inria's rasterizer, or 2DGS's, which is based on it, and the step writes its splat with its own
# PLY_Writer (splat-training spec, Req 4.14, 8.2).
NEVER_IN_GPU_MODULE = ("rasterization_inria_wrapper", "rasterization_2dgs_inria_wrapper",
                       "export_splats")
# Run in a fresh process with module names as its arguments: it imports splatpipe.train, then
# each module it's given, and prints the name of every module in sys.modules.
LOADED_BY_TRAIN = ("import importlib, json, sys\n"
                   "import splatpipe.train\n"
                   "for name in sys.argv[1:]:\n"
                   "    importlib.import_module(name)\n"
                   "print(json.dumps(sorted(sys.modules)))\n")
PYPROJECT = REPO_ROOT / "pyproject.toml"
UV_LOCK = REPO_ROOT / "uv.lock"
# The runtime dependencies the Frames_Step needs (frames spec, Req 9.8).
FRAMES_DEPENDENCIES = frozenset({"numpy", "opencv-python-headless"})
# The packages that never go in pyproject.toml or uv.lock, by their normalised names: plain
# opencv-python, which installs the same cv2 module as the headless one, and the GPU packages,
# which only go in the cloud machine image (frames spec, Req 9.8, and splat-training spec,
# Req 9.10).
NEVER_PACKAGES = frozenset({"opencv-python", "torch", "gsplat", "fused-ssim"})
NEVER_PREFIXES = ("nvidia-", "cuda-")
# The first character of a requirement like "numpy>=2.4.6" that can't be part of its package name.
NAME_END = re.compile(r"[^A-Za-z0-9._-]")


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


def package_name(requirement: str) -> str:
    """The package that a requirement like "numpy>=2.4.6" names, normalised as PEP 503 does, so
    "OpenCV_Python" reads as opencv-python."""
    name = NAME_END.split(requirement.strip(), maxsplit=1)[0]
    return re.sub(r"[-_.]+", "-", name).lower()


def pyproject_packages(pyproject: dict[str, Any]) -> list[str]:
    """Every package that pyproject.toml names, normalised: its runtime dependencies, its extras
    and its dependency groups."""
    project = pyproject["project"]
    lists = [project.get("dependencies", []), *project.get("optional-dependencies", {}).values(),
             *pyproject.get("dependency-groups", {}).values()]
    return [package_name(requirement) for requirements in lists for requirement in requirements]


def never_a_dependency(name: str) -> bool:
    """Whether a package, by its normalised name, must never be in pyproject.toml or uv.lock."""
    return name in NEVER_PACKAGES or name.startswith(NEVER_PREFIXES)


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
    """The entries are the Frames_Step, the Train_Step and then the Clean_Step, so this loads
    [frames], [train] and [clean] from the real config for each of the three Presets (frames
    spec, Req 11.7, and splat-training spec, Req 14.9)."""
    config = load_config(REAL_CONFIG)
    for entry in STEP_LIST:
        assert check_step_name(entry.name) == entry.name
        assert entry.module == f"splatpipe.{entry.name}"  # Req 12.2
        assert entry.module != "splatpipe.fake"  # the Fake_Step never runs on real Jobs (Req 8.2)
        for preset in PRESETS:
            config.step_settings(entry.name, preset)  # Req 5.11


def test_the_frames_step_imports_only_the_standard_library_numpy_cv2_and_splatpipe() -> None:
    """Frames spec, Req 9.9, 13.10."""
    allowed = sys.stdlib_module_names | FRAMES_IMPORTS
    found = [(where, name) for path in sorted(SPLATPIPE.glob("frames/*.py"))
             for where, name in imports_of(path) if top(name) not in allowed]
    assert found == []


def test_the_train_step_imports_only_the_standard_library_numpy_cv2_and_splatpipe() -> None:
    """Splat-training spec, Req 9.11, 9.13, 14.7. gpu.py, the GPU_Module, may import torch, gsplat
    and fused_ssim as well, and no other file may. The files are found with glob, so each file a
    later task adds is checked as it lands."""
    found: list[tuple[str, str]] = []
    for path in sorted(SPLATPIPE.glob("train/*.py")):
        allowed = sys.stdlib_module_names | (
            GPU_MODULE_IMPORTS if path == GPU_MODULE_FILE else TRAIN_IMPORTS)
        found += [(where, name) for where, name in imports_of(path) if top(name) not in allowed]
    assert found == []


def test_gpu_py_starts_with_the_apache_note_of_gsplats_example_trainer() -> None:
    """Splat-training spec, Req 4.15. gpu.py is read as text and never imported, and the note's
    copyright and licence lines come before its first import."""
    text = GPU_MODULE_FILE.read_text(encoding="utf-8")
    lines = text.splitlines()
    assert tuple(lines[:len(GSPLAT_NOTE)]) == GSPLAT_NOTE
    first_import = imports_in(text, "splatpipe.train")[0][0]
    before_imports = lines[:first_import - 1]
    assert "# Copyright 2025 Nerfstudio Team" in before_imports
    assert any(line.startswith("# Licensed under the Apache License, Version 2.0")
               for line in before_imports)


def test_gpu_py_names_no_inria_wrapper_and_no_export_splats() -> None:
    """Splat-training spec, Req 4.14, 8.2: anywhere in its text, comments and docstrings
    included."""
    text = GPU_MODULE_FILE.read_text(encoding="utf-8")
    assert [name for name in NEVER_IN_GPU_MODULE if name in text] == []


def test_gpu_py_is_the_only_train_file_that_imports_torch_gsplat_or_fused_ssim() -> None:
    """Splat-training spec, Req 9.11, 14.7: and gpu.py imports all three."""
    found = [(where, name) for path in sorted(SPLATPIPE.glob("train/*.py"))
             for where, name in imports_of(path) if top(name) in GPU_NAMES]
    assert {where.rpartition(":")[0] for where, _ in found} == {GPU_MODULE_SHOWN}, found
    assert {top(name) for _, name in found} == GPU_NAMES, found


def test_the_module_load_gpu_imports_is_the_file_these_checks_read() -> None:
    """Splat-training spec, Req 2.2: gpu_api.GPU_MODULE with its dots turned into folders, plus
    .py, is splatpipe/train/gpu.py."""
    assert GPU_MODULE.replace(".", "/") + ".py" == GPU_MODULE_SHOWN


def test_importing_splatpipe_train_and_its_cpu_modules_loads_no_gpu_package() -> None:
    """Splat-training spec, Req 9.12, 14.7. The CPU modules are every splatpipe/train/*.py but
    __init__.py and gpu.py, found with glob, so a module a later task adds is imported as it
    lands. On a laptop, where torch, gsplat and fused_ssim aren't installed, a CPU module that
    imported one would fail the fresh process."""
    modules = [f"splatpipe.train.{path.stem}" for path in sorted(SPLATPIPE.glob("train/*.py"))
               if path != GPU_MODULE_FILE and path.stem != "__init__"]
    result = subprocess.run([sys.executable, "-c", LOADED_BY_TRAIN, *modules], cwd=REPO_ROOT,
                            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
                            capture_output=True, encoding="utf-8", timeout=60)
    assert result.returncode == 0, result.stderr
    loaded = json.loads(result.stdout)
    assert {"splatpipe.train", *modules} <= set(loaded)
    assert [name for name in loaded if top(name) in GPU_NAMES] == []


def test_pyproject_and_uv_lock_hold_numpy_and_headless_opencv_and_no_gpu_or_plain_opencv() -> None:
    """Frames spec, Req 9.8. scipy is neither required nor refused, so the clean spec can decide
    it without changing this test."""
    pyproject = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    lock = {package["name"]: package
            for package in tomllib.loads(UV_LOCK.read_text(encoding="utf-8"))["package"]}
    runtime = {package_name(requirement) for requirement in pyproject["project"]["dependencies"]}
    assert FRAMES_DEPENDENCIES <= runtime
    own_entry = lock[package_name(pyproject["project"]["name"])]
    assert FRAMES_DEPENDENCIES <= {dependency["name"] for dependency in own_entry["dependencies"]}
    named = [("pyproject.toml", name) for name in pyproject_packages(pyproject)]
    named += [("uv.lock", name) for name in lock]
    assert [(where, name) for where, name in named if never_a_dependency(name)] == []
