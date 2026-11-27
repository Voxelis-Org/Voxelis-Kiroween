"""The import checks (Req 12) and the dependency check: they read the Pipeline's files with ast,
the two GPU modules' as text too, and pyproject.toml and uv.lock with tomllib, and run no Step
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
- The Cameras_Step's imports (cameras spec, Req 2.1, 12.13, 12.14, 16.10): every
  splatpipe/cameras/*.py imports only names in sys.stdlib_module_names, numpy, cv2 and
  splatpipe, so torch, gsplat, boto3, pycolmap and cloud/ stay out. None of those files names
  VideoCapture or VideoWriter, cv2's video classes: not in an import like `from cv2 import
  VideoWriter`, an attribute like cv2.VideoCapture under any name the file gives cv2, or a bare
  name. A docstring or comment that names one isn't code, so it doesn't count. With the
  dependency check, this shows the step needs no new package (Req 12.12).
- The Train_Step's imports (splat-training spec, Req 9.11, 9.13, 14.7, and depth-priors spec,
  Req 14.9): every splatpipe/train/*.py but gpu.py imports only names in
  sys.stdlib_module_names, numpy, cv2 and splatpipe, and gpu.py, the GPU_Module, imports only
  those and torch, gsplat and fused_ssim, so boto3, cloud/ and every other GPU package stay out
  of all of them. The files are found with glob, so each new file is checked as soon as it
  lands.
- The GPU_Module (splat-training spec, Req 2.2, 4.14, 4.15, 8.2, 9.11, 9.12, 14.7): gpu.py is
  read as text and with ast and never imported, because torch, gsplat and fused_ssim exist only
  on the Machine_Image. It starts with the Apache-2.0 note that the gsplat-training skill gives
  for code adapted from gsplat's example trainer, with the note's copyright and licence lines
  before its first import, and its text, comments and docstrings included, names none of
  rasterization_inria_wrapper, rasterization_2dgs_inria_wrapper and export_splats, the
  Prior_Terms included (depth-priors spec, Req 13.17). It's the only splatpipe/train/*.py that
  imports torch, gsplat or fused_ssim, and it imports all three. gpu_api.GPU_MODULE, the module
  load_gpu imports, names that file. A fresh process that imports splatpipe.train and each of
  its other modules, found with glob, holds none of torch, gsplat and fused_ssim in sys.modules.
- The Trainer's Loss_Table (depth-priors spec, Req 11.7, 13.1, 13.13, 13.15): gpu.py is read
  with ast and never imported. Its _TERMS holds exactly the splat-training spec's five rows, l1,
  ssim, opacity_reg, scale_reg and erank with their weight settings, in that order, and the
  names and weight settings of _TERMS and _PRIOR_TERMS are SPLAT_TERMS, PRIOR_TERMS and
  TERM_WEIGHTS from schedules.py. gpu.py calls active_terms and depth_weight, each a Call node
  in ast, imported from its CPU module and not defined in gpu.py, so the rules the laptop tests
  check are the ones the Trainer uses. A made-up file shows that a name mentioned only in a
  docstring or a string is no call.
- The Depth_Step's imports (depth-priors spec, Req 18.8): every splatpipe/depth/*.py but gpu.py
  imports only names in sys.stdlib_module_names, numpy, cv2 and splatpipe, a relative import
  counting as splatpipe, and gpu.py, the Depth_GPU_Module, imports only those and torch, moge
  and huggingface_hub. The files are found with glob, so each new file is checked as soon as it
  lands, and the check fails if it finds no file, so it never passes by reading nothing.
- The Depth_GPU_Module (depth-priors spec, Req 3.2, 3.6, 3.9, 3.10, 3.15, 18.8): gpu.py is read
  as text and with ast and never imported, because torch, MoGe and huggingface_hub exist only on
  the Machine_Image. It imports all three: torch only in import statements of the module's own
  body, at its top, and moge and huggingface_hub only inside functions. Each function that
  imports huggingface_hub, import_moge among them, sets os.environ["HF_HUB_OFFLINE"] to "1" in
  a statement of its own body, on a line before its first moge or huggingface_hub import, by
  ast's line numbers, because the docstring names both first. load_model's moge import isn't
  read, because it runs after import_moge has imported MoGe. gpu.py calls torch.load, every call
  with weights_only=True, and its text, comments and docstrings included, names none of
  from_pretrained, hf_hub_download and init_weights. gpu_api.DEPTH_GPU_MODULE, the module
  load_depth_gpu imports, names that file. A fresh process that imports splatpipe.depth and
  each of its other modules, found with glob, holds none of torch, moge, utils3d_moge and
  huggingface_hub in sys.modules. A made-up file that breaks each rule shows each check failing,
  with its line.
- The dependencies (frames spec, Req 9.8, splat-training spec, Req 9.10, and depth-priors spec,
  Req 15.5): numpy and opencv-python-headless are runtime dependencies in pyproject.toml and in
  the project's own entry in uv.lock, and no package in either file is plain opencv-python,
  torch, gsplat, fused-ssim, an nvidia-* or cuda-* package, moge, utils3d-moge, huggingface-hub
  or hf-xet. MoGe's code and the packages only it needs go only on the Machine_Image, which
  leaves hf-xet off, so a laptop's uv sync installs none of them. Names are compared as PEP 503
  normalises them, so utils3d_moge, huggingface_hub and hf_xet are refused too. The check
  neither requires nor refuses scipy, so the clean spec can decide scipy without changing it.

Each import check's failure lists the file and the line of every import it found, and the
cameras check's failure does the same for every video class it found. The dependency check's
failure lists the file of every package it found.
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
from splatpipe.depth.gpu_api import DEPTH_GPU_MODULE
from splatpipe.run import STEP_LIST
from splatpipe.step import check_step_name
from splatpipe.train.gpu_api import GPU_MODULE
from splatpipe.train.schedules import PRIOR_TERMS, SPLAT_TERMS, TERM_WEIGHTS

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
# The top-level names a splatpipe/cameras/*.py may import besides the standard library (cameras
# spec, Req 12.13).
CAMERAS_IMPORTS = frozenset({"numpy", "cv2", "splatpipe"})
# cv2's classes that read and write video, which no splatpipe/cameras/*.py names (cameras spec,
# Req 12.14).
VIDEO_CLASSES = frozenset({"VideoCapture", "VideoWriter"})
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
TRAIN_PACKAGE = "splatpipe.train"  # the package gpu.py's relative imports resolve against
# The splat-training spec's Loss_Table, its Requirement 5, criterion 1: each Loss_Term's name and
# the setting that holds its weight, in order. gpu.py's _TERMS holds exactly these rows, the
# whole Loss_Table with no Prior_Set (depth-priors spec, Req 13.13).
SPLAT_TRAINING_ROWS = (("l1", "l1_weight"), ("ssim", "ssim_weight"),
                       ("opacity_reg", "opacity_reg"), ("scale_reg", "scale_reg"),
                       ("erank", "erank_reg"))
# The CPU functions gpu.py calls, each with the module it imports it from: active_terms decides
# each Iteration's Loss_Terms and depth_weight gives every training pixel's Depth_Weight
# (depth-priors spec, Req 11.7, 13.15).
TRAINER_CALLS = (("active_terms", "splatpipe.train.schedules"),
                 ("depth_weight", "splatpipe.train.prior_views"))
# Run in a fresh process with module names as its arguments: it imports splatpipe.train, then
# each module it's given, and prints the name of every module in sys.modules.
LOADED_BY_TRAIN = ("import importlib, json, sys\n"
                   "import splatpipe.train\n"
                   "for name in sys.argv[1:]:\n"
                   "    importlib.import_module(name)\n"
                   "print(json.dumps(sorted(sys.modules)))\n")
# The top-level names a splatpipe/depth/*.py may import besides the standard library, and the
# Depth_GPU_Module's file, which may import torch, moge and huggingface_hub too (depth-priors
# spec, Req 18.8).
DEPTH_IMPORTS = frozenset({"numpy", "cv2", "splatpipe"})
DEPTH_GPU_MODULE_FILE = SPLATPIPE / "depth" / "gpu.py"
DEPTH_GPU_MODULE_SHOWN = DEPTH_GPU_MODULE_FILE.relative_to(REPO_ROOT).as_posix()
DEPTH_PACKAGE = "splatpipe.depth"  # the package gpu.py's relative imports resolve against
DEPTH_GPU_NAMES = frozenset({"torch", "moge", "huggingface_hub"})
DEPTH_GPU_MODULE_IMPORTS = DEPTH_IMPORTS | DEPTH_GPU_NAMES
# The two names gpu.py imports only inside functions, after HF_HUB_OFFLINE is set, because
# huggingface_hub reads the variable once, when it's first imported, and MoGe's model module
# imports huggingface_hub at its top (depth-priors spec, Req 3.6).
HUB_NAMES = frozenset({"moge", "huggingface_hub"})
# The statement that sets HF_HUB_OFFLINE, as ast.unparse writes it, so either kind of quote reads
# the same (depth-priors spec, Req 3.6).
SET_HUB_OFFLINE = "os.environ['HF_HUB_OFFLINE'] = '1'"
# The functions gpu.py never names, comments and docstrings included: MoGe's own loader and the
# Hub's download, either of which can fetch the weights, and the encoder's set-up from DINOv2's
# original weights, which downloads them and overwrites MoGe's (depth-priors spec, Req 3.10).
NEVER_IN_DEPTH_GPU_MODULE = ("from_pretrained", "hf_hub_download", "init_weights")
# The names a fresh process that imports splatpipe.depth and its CPU modules never loads, by top
# level (depth-priors spec, Req 3.15).
NEVER_LOADED_BY_DEPTH = frozenset({"torch", "moge", "utils3d_moge", "huggingface_hub"})
# Run in a fresh process with module names as its arguments: it imports splatpipe.depth, then
# each module it's given, and prints the name of every module in sys.modules.
LOADED_BY_DEPTH = ("import importlib, json, sys\n"
                   "import splatpipe.depth\n"
                   "for name in sys.argv[1:]:\n"
                   "    importlib.import_module(name)\n"
                   "print(json.dumps(sorted(sys.modules)))\n")
# The kinds of node whose body is a function's, so an import inside one runs only when it's called.
FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef)
PYPROJECT = REPO_ROOT / "pyproject.toml"
UV_LOCK = REPO_ROOT / "uv.lock"
# The runtime dependencies the Frames_Step needs (frames spec, Req 9.8).
FRAMES_DEPENDENCIES = frozenset({"numpy", "opencv-python-headless"})
# The packages that never go in pyproject.toml or uv.lock, by their normalised names: plain
# opencv-python, which installs the same cv2 module as the headless one, the GPU packages, which
# only go in the cloud machine image, and MoGe's code with the packages only it needs, which go
# only there too, hf-xet included though the image leaves it off (frames spec, Req 9.8,
# splat-training spec, Req 9.10, and depth-priors spec, Req 15.5).
NEVER_PACKAGES = frozenset({"opencv-python", "torch", "gsplat", "fused-ssim", "moge",
                            "utils3d-moge", "huggingface-hub", "hf-xet"})
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
    return imports_below(ast.parse(source), package)


def imports_below(tree: ast.AST, package: str) -> list[tuple[int, str]]:
    """Every module name imported in `tree`, a parsed file in `package` or a part of one, like a
    function, as (line, name), in order."""
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
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


def video_classes_of(path: Path) -> list[tuple[str, str]]:
    """Every attribute or name in the file at `path` that is one of VIDEO_CLASSES, as
    ("<file>:<line>", its code), like cv2.VideoCapture under any name the file gives cv2. A
    docstring or comment that names one isn't code, so it isn't read."""
    shown = path.relative_to(REPO_ROOT).as_posix()
    found = [(node.lineno, ast.unparse(node)) for node in ast.walk(ast.parse(path.read_bytes()))
             if (isinstance(node, ast.Attribute) and node.attr in VIDEO_CLASSES)
             or (isinstance(node, ast.Name) and node.id in VIDEO_CLASSES)]
    return [(f"{shown}:{line}", code) for line, code in sorted(found)]


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


def misplaced_depth_gpu_imports(source: str | bytes, package: str) -> list[tuple[int, str]]:
    """The torch, moge and huggingface_hub imports in `source` that are out of place, as
    (line, name): torch anywhere but in an import statement of the module's own body, at its top,
    and moge or huggingface_hub anywhere but inside a function (depth-priors spec, Req 3.6,
    18.8)."""
    tree = ast.parse(source)
    at_top = {pair for statement in tree.body
              if isinstance(statement, (ast.Import, ast.ImportFrom))
              for pair in imports_below(statement, package)}
    in_functions = {pair for node in ast.walk(tree) if isinstance(node, FUNCTIONS)
                    for pair in imports_below(node, package)}
    return [(line, name) for line, name in imports_below(tree, package)
            if (top(name) == "torch" and (line, name) not in at_top)
            or (top(name) in HUB_NAMES and (line, name) not in in_functions)]


def functions_importing_huggingface_hub(source: str | bytes,
                                        package: str) -> list[ast.FunctionDef]:
    """Each function in `source`, nested ones included, that imports huggingface_hub."""
    return [node for node in ast.walk(ast.parse(source)) if isinstance(node, FUNCTIONS)
            and any(top(name) == "huggingface_hub" for _, name in imports_below(node, package))]


def hub_imports_before_offline(source: str | bytes, package: str) -> list[tuple[int, str]]:
    """The first moge or huggingface_hub import, as (line, name), of each function in `source`
    that imports huggingface_hub with no statement of its own body setting HF_HUB_OFFLINE to 1 on
    an earlier line (depth-priors spec, Req 3.6).

    Line numbers come from ast, because the module's docstring names both the variable and the
    imports first. A function that imports moge and not huggingface_hub, like gpu.py's
    load_model, isn't read: it runs after import_moge has imported MoGe, so its import is only a
    lookup in sys.modules.
    """
    found: list[tuple[int, str]] = []
    for function in functions_importing_huggingface_hub(source, package):
        line, name = next(pair for pair in imports_below(function, package)
                          if top(pair[1]) in HUB_NAMES)
        if not any(ast.unparse(statement) == SET_HUB_OFFLINE and statement.lineno < line
                   for statement in function.body):
            found.append((line, name))
    return sorted(found)


def torch_load_calls(source: str | bytes) -> list[tuple[int, bool]]:
    """Each torch.load call in `source`, as (line, whether it passes weights_only=True)
    (depth-priors spec, Req 3.9)."""
    return [(node.lineno, any(keyword.arg == "weights_only"
                              and isinstance(keyword.value, ast.Constant)
                              and keyword.value.value is True for keyword in node.keywords))
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.Call) and ast.unparse(node.func) == "torch.load"]


def lines_naming(text: str, names: tuple[str, ...]) -> list[tuple[int, str]]:
    """Each line of `text` that names one of `names`, as (line, name), comments and docstrings
    included."""
    return [(number, name) for number, line in enumerate(text.splitlines(), start=1)
            for name in names if name in line]


def table_rows(source: str | bytes, table: str) -> list[tuple[str, str | None]]:
    """The (name, weight setting) of each row of the Loss_Table `table`, read with ast from the
    one statement of `source`'s module body that assigns it, and never run.

    Each row must be a tuple of three: the term's name and its weight setting, both constants,
    the setting None for a weight of 1, and the term's function, a name (depth-priors spec,
    Req 13.1, 13.13).
    """
    values = [statement.value for statement in ast.parse(source).body
              if isinstance(statement, (ast.Assign, ast.AnnAssign))
              for target in (statement.targets if isinstance(statement, ast.Assign)
                             else [statement.target])
              if isinstance(target, ast.Name) and target.id == table]
    assert len(values) == 1, f"{table} is assigned {len(values)} times in the module's body"
    value = values[0]
    assert isinstance(value, ast.Tuple), f"{table} isn't a tuple of rows"
    rows: list[tuple[str, str | None]] = []
    for row in value.elts:
        assert (isinstance(row, ast.Tuple) and len(row.elts) == 3
                and isinstance(row.elts[2], ast.Name)), ast.unparse(row)
        rows.append((ast.literal_eval(row.elts[0]), ast.literal_eval(row.elts[1])))
    return rows


def called_names(source: str | bytes) -> set[str]:
    """The name of each function `source` calls, as a name, like f(x), or as an attribute, like
    m.f(x), read from ast's Call nodes, so a name only mentioned in a comment, a docstring or a
    string isn't a call (depth-priors spec, Req 11.7)."""
    names: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            names.add(node.func.id)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            names.add(node.func.attr)
    return names


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
    """The entries are the Frames_Step, the Cameras_Step, the Depth_Step, the Walls_Step, the
    Train_Step and then the Clean_Step, so this loads [frames], [cameras], [depth], [walls],
    [train] and [clean] from the real config for each of the three Presets (frames spec,
    Req 11.7; cameras spec, Req 14.7; splat-training spec, Req 14.9; depth-priors spec,
    Req 18.10; walls spec, Req 1.3, 22.9)."""
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


def test_the_cameras_step_imports_only_stdlib_numpy_cv2_and_splatpipe_and_no_video_class() -> None:
    """Cameras spec, Req 2.1, 12.13, 12.14 and 16.10. `from cv2 import VideoWriter` reads as
    cv2.VideoWriter, so the import reader finds it, and video_classes_of finds the attributes and
    names in the code. With the dependency check below, this shows that the step needs no new
    package (Req 12.12)."""
    allowed = sys.stdlib_module_names | CAMERAS_IMPORTS
    paths = sorted(SPLATPIPE.glob("cameras/*.py"))
    found = [(where, name) for path in paths for where, name in imports_of(path)
             if top(name) not in allowed or name.removeprefix("cv2.") in VIDEO_CLASSES]
    found += [use for path in paths for use in video_classes_of(path)]
    assert found == []


def test_the_train_step_imports_only_the_standard_library_numpy_cv2_and_splatpipe() -> None:
    """Splat-training spec, Req 9.11, 9.13, 14.7, and depth-priors spec, Req 14.9. gpu.py, the
    GPU_Module, may import torch, gsplat and fused_ssim as well, and no other GPU package, and no
    other file may import any of them. The files are found with glob, so each file a later task
    adds is checked as it lands."""
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
    included, and still with the Prior_Terms in it (depth-priors spec, Req 13.17)."""
    text = GPU_MODULE_FILE.read_text(encoding="utf-8")
    assert [name for name in NEVER_IN_GPU_MODULE if name in text] == []


def test_gpu_py_is_the_only_train_file_that_imports_torch_gsplat_or_fused_ssim() -> None:
    """Splat-training spec, Req 9.11, 14.7, and depth-priors spec, Req 14.9: and gpu.py imports
    all three."""
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


def test_the_trainer_checks_read_the_rows_and_only_real_calls() -> None:
    """The helpers that the Trainer's Loss_Table checks use, on a made-up file: a row's function
    isn't read, a None weight setting is kept, and active_terms, named only in the docstring and
    in a string, is no call, while prior_views.depth_weight(i) is one."""
    sample = ('"""Calls active_terms."""\n'
              "from splatpipe.train import prior_views\n"
              "_TERMS: tuple = (\n"
              "    ('l1', 'l1_weight', _l1),\n"
              "    ('depth', None, _depth),\n"
              ")\n"
              "def step(i):\n"
              "    on = 'active_terms'\n"
              "    return prior_views.depth_weight(i)\n")
    assert table_rows(sample, "_TERMS") == [("l1", "l1_weight"), ("depth", None)]
    assert called_names(sample) == {"depth_weight"}


def test_gpu_py_terms_are_exactly_the_splat_training_specs_five_rows() -> None:
    """Depth-priors spec, Req 13.13: gpu.py's _TERMS, read with ast and never imported, holds
    exactly the splat-training spec's five Loss_Terms with their weight settings, l1, ssim,
    opacity_reg, scale_reg and erank in that order (its Requirement 5, criterion 1), so with no
    Prior_Set the Loss_Table is that spec's."""
    assert tuple(table_rows(GPU_MODULE_FILE.read_bytes(), "_TERMS")) == SPLAT_TRAINING_ROWS


def test_gpu_py_loss_tables_hold_the_schedules_names_and_weight_settings() -> None:
    """Depth-priors spec, Req 13.1, 13.13: the names of gpu.py's _TERMS and _PRIOR_TERMS, read
    with ast, are SPLAT_TERMS and PRIOR_TERMS in order, the names active_terms gives, and each
    row's weight setting is TERM_WEIGHTS's, depth's None included. So the terms the Trainer
    works out are the ones the laptop tests of active_terms check."""
    source = GPU_MODULE_FILE.read_bytes()
    terms = table_rows(source, "_TERMS")
    prior_terms = table_rows(source, "_PRIOR_TERMS")
    assert tuple(name for name, _ in terms) == SPLAT_TERMS
    assert tuple(name for name, _ in prior_terms) == PRIOR_TERMS
    assert dict(terms + prior_terms) == TERM_WEIGHTS


def test_gpu_py_calls_active_terms_and_depth_weight_from_their_cpu_modules() -> None:
    """Depth-priors spec, Req 11.7, 13.15: gpu.py calls active_terms and depth_weight, each a
    Call node in ast, as a name or an attribute, so a mention in a comment or a docstring doesn't
    count. Each is imported from its CPU module, the function itself or the module, and gpu.py
    defines no function by that name, so the rules the laptop tests check are the ones the
    Trainer uses."""
    text = GPU_MODULE_FILE.read_text(encoding="utf-8")
    called = called_names(text)
    imported = {name for _, name in imports_in(text, TRAIN_PACKAGE)}
    defined = {node.name for node in ast.walk(ast.parse(text)) if isinstance(node, FUNCTIONS)}
    problems = [f"{GPU_MODULE_SHOWN} never calls {name}" for name, _ in TRAINER_CALLS
                if name not in called]
    problems += [f"{GPU_MODULE_SHOWN} doesn't import {name} from {module}"
                 for name, module in TRAINER_CALLS
                 if f"{module}.{name}" not in imported and module not in imported]
    problems += [f"{GPU_MODULE_SHOWN} defines its own {name}" for name, _ in TRAINER_CALLS
                 if name in defined]
    assert problems == []


def test_the_depth_step_imports_only_the_standard_library_numpy_cv2_and_splatpipe() -> None:
    """Depth-priors spec, Req 18.8. gpu.py, the Depth_GPU_Module, may import torch, moge and
    huggingface_hub as well, and no other file may. A relative import resolves to splatpipe. The
    files are found with glob, so each file a later task adds is checked as it lands, and the
    test fails when it finds none, so it never passes by reading nothing."""
    paths = sorted(SPLATPIPE.glob("depth/*.py"))
    assert paths, ("splatpipe/depth/ holds no .py file, so the depth import check has nothing to "
                   "read. Make splatpipe/depth/__init__.py first.")
    found: list[tuple[str, str]] = []
    for path in paths:
        allowed = sys.stdlib_module_names | (
            DEPTH_GPU_MODULE_IMPORTS if path == DEPTH_GPU_MODULE_FILE else DEPTH_IMPORTS)
        found += [(where, name) for where, name in imports_of(path) if top(name) not in allowed]
    assert found == []


def test_the_depth_gpu_checks_find_each_violation_with_its_line() -> None:
    """The helpers that the Depth_GPU_Module's checks use, on a made-up file that breaks each
    rule, so the checks can fail. guarded() breaks none, with double quotes that ast.unparse
    reads as single ones."""
    sample = ('"""Calls from_pretrained."""\n'
              "import os\n"
              "import moge\n"
              "def import_moge():\n"
              "    import torch\n"
              "    import huggingface_hub.constants\n"
              "    os.environ['HF_HUB_OFFLINE'] = '1'\n"
              "def load(path):\n"
              "    return torch.load(path, weights_only=False)\n"
              "def later():\n"
              "    if True:\n"
              '        os.environ["HF_HUB_OFFLINE"] = "1"\n'
              "    from moge.model.v2 import MoGeModel\n"
              "    import huggingface_hub\n"
              "def guarded(path):\n"
              '    os.environ["HF_HUB_OFFLINE"] = "1"\n'
              "    import huggingface_hub\n"
              "    return torch.load(path, map_location='cpu', weights_only=True)\n")
    assert misplaced_depth_gpu_imports(sample, DEPTH_PACKAGE) == [(3, "moge"), (5, "torch")]
    assert hub_imports_before_offline(sample, DEPTH_PACKAGE) == [
        (6, "huggingface_hub.constants"), (13, "moge.model.v2.MoGeModel")]
    assert torch_load_calls(sample) == [(9, False), (18, True)]
    assert lines_naming(sample, NEVER_IN_DEPTH_GPU_MODULE) == [(1, "from_pretrained")]


def test_depth_gpu_py_imports_torch_at_its_top_and_moge_and_huggingface_hub_inside_functions(
) -> None:
    """Depth-priors spec, Req 3.6, 18.8. gpu.py is read as text and never imported, because
    torch, MoGe and huggingface_hub exist only on the Machine_Image. It imports all three."""
    text = DEPTH_GPU_MODULE_FILE.read_text(encoding="utf-8")
    found = [(f"{DEPTH_GPU_MODULE_SHOWN}:{line}", name)
             for line, name in misplaced_depth_gpu_imports(text, DEPTH_PACKAGE)]
    assert found == []
    imported = {top(name) for _, name in imports_in(text, DEPTH_PACKAGE)}
    assert DEPTH_GPU_NAMES <= imported, sorted(DEPTH_GPU_NAMES - imported)


def test_depth_gpu_py_sets_hf_hub_offline_before_it_imports_moge_or_huggingface_hub() -> None:
    """Depth-priors spec, Req 3.6. In each function that imports huggingface_hub, import_moge
    among them, a statement of the function's own body sets os.environ["HF_HUB_OFFLINE"] to "1"
    on a line before the function's first moge or huggingface_hub import. The lines come from
    ast, since the docstring names both first, and load_model, which imports moge only after
    import_moge has, isn't read."""
    text = DEPTH_GPU_MODULE_FILE.read_text(encoding="utf-8")
    importers = [function.name
                 for function in functions_importing_huggingface_hub(text, DEPTH_PACKAGE)]
    assert "import_moge" in importers, importers
    found = [(f"{DEPTH_GPU_MODULE_SHOWN}:{line}", name)
             for line, name in hub_imports_before_offline(text, DEPTH_PACKAGE)]
    assert found == []


def test_depth_gpu_py_loads_the_weights_with_weights_only() -> None:
    """Depth-priors spec, Req 3.9: gpu.py calls torch.load, and every call passes
    weights_only=True, read with ast, since the docstring names it too."""
    calls = torch_load_calls(DEPTH_GPU_MODULE_FILE.read_text(encoding="utf-8"))
    assert calls, f"{DEPTH_GPU_MODULE_SHOWN} calls no torch.load"
    assert [f"{DEPTH_GPU_MODULE_SHOWN}:{line}" for line, weights_only in calls
            if not weights_only] == []


def test_depth_gpu_py_names_no_moge_loader_hub_download_or_dinov2_set_up() -> None:
    """Depth-priors spec, Req 3.10: anywhere in its text, comments and docstrings included."""
    text = DEPTH_GPU_MODULE_FILE.read_text(encoding="utf-8")
    assert [(f"{DEPTH_GPU_MODULE_SHOWN}:{line}", name)
            for line, name in lines_naming(text, NEVER_IN_DEPTH_GPU_MODULE)] == []


def test_the_module_load_depth_gpu_imports_is_the_file_these_checks_read() -> None:
    """Depth-priors spec, Req 3.2, 18.8: gpu_api.DEPTH_GPU_MODULE with its dots turned into
    folders, plus .py, is splatpipe/depth/gpu.py."""
    assert DEPTH_GPU_MODULE.replace(".", "/") + ".py" == DEPTH_GPU_MODULE_SHOWN


def test_importing_splatpipe_depth_and_its_cpu_modules_loads_no_torch_moge_or_huggingface_hub(
) -> None:
    """Depth-priors spec, Req 3.15, 18.8. The CPU modules are every splatpipe/depth/*.py but
    __init__.py and gpu.py, found with glob, so a module a later task adds is imported as it
    lands. On a laptop, where none of the four is installed, a CPU module that imported one would
    fail the fresh process."""
    modules = [f"{DEPTH_PACKAGE}.{path.stem}" for path in sorted(SPLATPIPE.glob("depth/*.py"))
               if path != DEPTH_GPU_MODULE_FILE and path.stem != "__init__"]
    result = subprocess.run([sys.executable, "-c", LOADED_BY_DEPTH, *modules], cwd=REPO_ROOT,
                            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
                            capture_output=True, encoding="utf-8", timeout=60)
    assert result.returncode == 0, result.stderr
    loaded = json.loads(result.stdout)
    assert {DEPTH_PACKAGE, *modules} <= set(loaded)
    assert [name for name in loaded if top(name) in NEVER_LOADED_BY_DEPTH] == []


def test_the_dependency_check_refuses_moge_and_the_packages_only_it_needs() -> None:
    """Depth-priors spec, Req 15.5, in the spellings MoGe's and huggingface_hub's own files use,
    so the check below refuses each of them in pyproject.toml and uv.lock."""
    spellings = ("moge", "utils3d_moge @ git+https://github.com/EasternJournalist/utils3d-moge",
                 "huggingface-hub==1.26.0", "huggingface_hub>=0.20", "hf-xet", "hf_xet>=1.1")
    assert [spelling for spelling in spellings
            if not never_a_dependency(package_name(spelling))] == []


def test_pyproject_and_uv_lock_hold_numpy_and_headless_opencv_and_no_gpu_or_plain_opencv() -> None:
    """Frames spec, Req 9.8, splat-training spec, Req 9.10, and depth-priors spec, Req 15.5: no
    GPU package, no MoGe package and no plain opencv-python. scipy is neither required nor
    refused, so the clean spec can decide it without changing this test."""
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
