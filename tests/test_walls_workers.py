"""The Walls_Step's import checks (walls spec, Req 1.11, 2.13, 22.8). They read the files with
ast, through test_imports.py's `imports_of`, and run no Step.

- Every splatpipe/walls/*.py imports only names in sys.stdlib_module_names, numpy, scipy, cv2 and
  splatpipe, a relative import counting as splatpipe, so torch, gsplat, MoGe, boto3 and cloud/
  stay out. pool.py's multiprocessing is in the standard library. None of the files imports
  anything from splatpipe.depth, and none names VideoCapture, VideoWriter or VideoWriter_fourcc,
  cv2's video classes: not in an import like `from cv2 import VideoWriter`, an attribute like
  cv2.VideoCapture under any name the file gives cv2, or a bare name. A docstring or comment that
  names one isn't code, so it doesn't count. The files are found with glob, so each file a later
  task adds is checked as it lands, and the check fails if it finds no file, so it never passes
  by reading nothing.
- No splatpipe/train/*.py imports splatpipe.walls, `from splatpipe import walls` and relative
  imports included, so the Train_Step never depends on the Walls_Step (Req 22.8).
- A fresh process that imports splatpipe.walls and each of its modules, found with glob, holds
  none of torch, gsplat and moge in sys.modules (Req 1.11).

Each check's failure lists the file and the line of every import or video class it found. A
made-up file shows the walls check finding each kind of problem with its line.

Task 9 of the walls spec adds the pool tests and the two-worker test to this file.
"""

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

from splatpipe.config import REPO_ROOT
from test_imports import imports_in, imports_of, top

SPLATPIPE = REPO_ROOT / "splatpipe"
WALLS_PACKAGE = "splatpipe.walls"
# The top-level names a splatpipe/walls/*.py may import besides the standard library (Req 22.8).
WALLS_IMPORTS = frozenset({"numpy", "scipy", "cv2", "splatpipe"})
# The package no splatpipe/walls/*.py imports from (Req 22.8).
DEPTH_PACKAGE = "splatpipe.depth"
# cv2's names that read and write video, which no splatpipe/walls/*.py names (Req 2.13).
VIDEO_CLASSES = frozenset({"VideoCapture", "VideoWriter", "VideoWriter_fourcc"})
# The GPU packages a fresh process that imports splatpipe.walls never loads, by top level
# (Req 1.11).
NEVER_LOADED_BY_WALLS = frozenset({"torch", "gsplat", "moge"})
# Run in a fresh process with module names as its arguments: it imports splatpipe.walls, then
# each module it's given, and prints the name of every module in sys.modules.
LOADED_BY_WALLS = ("import importlib, json, sys\n"
                   "import splatpipe.walls\n"
                   "for name in sys.argv[1:]:\n"
                   "    importlib.import_module(name)\n"
                   "print(json.dumps(sorted(sys.modules)))\n")


def within(name: str, package: str) -> bool:
    """Whether the dotted module name `name` is `package` or a module inside it."""
    return name == package or name.startswith(f"{package}.")


def walls_problems(source: str | bytes, package: str) -> list[tuple[int, str]]:
    """Every import and name in `source`, a file in `package`, that a splatpipe/walls/*.py may not
    have, as (line, what), in order: an import outside the standard library, numpy, scipy, cv2
    and splatpipe, an import from splatpipe.depth, and a cv2 video class, imported, used as an
    attribute or named bare. `from cv2 import VideoWriter` reads as cv2.VideoWriter."""
    allowed = sys.stdlib_module_names | WALLS_IMPORTS
    found = [(line, name) for line, name in imports_in(source, package)
             if top(name) not in allowed or within(name, DEPTH_PACKAGE)
             or name.removeprefix("cv2.") in VIDEO_CLASSES]
    found += [(node.lineno, ast.unparse(node)) for node in ast.walk(ast.parse(source))
              if (isinstance(node, ast.Attribute) and node.attr in VIDEO_CLASSES)
              or (isinstance(node, ast.Name) and node.id in VIDEO_CLASSES)]
    return sorted(found)


def walls_files() -> list[Path]:
    """Every splatpipe/walls/*.py, found with glob, so each file a later task adds counts."""
    return sorted(SPLATPIPE.glob("walls/*.py"))


def test_the_walls_check_finds_each_problem_with_its_line() -> None:
    """A made-up file that breaks each rule, so the check can fail: torch, an absolute and a
    relative import from splatpipe.depth, `from splatpipe import depth`, an imported and an
    attribute video class, VideoWriter_fourcc as an attribute, and a bare name. The docstring
    names VideoCapture, which isn't code, and stdlib, numpy, scipy, cv2 and splatpipe imports
    pass."""
    sample = ('"""Never calls cv2.VideoCapture."""\n'
              "import multiprocessing\n"
              "import numpy as np\n"
              "from scipy import ndimage\n"
              "import cv2 as image_codec\n"
              "from splatpipe.train import files\n"
              "from . import settings\n"
              "import torch\n"
              "from splatpipe.depth import files as depth_files\n"
              "from ..depth.fit import fit\n"
              "from splatpipe import depth\n"
              "from cv2 import VideoWriter\n"
              "def read(path):\n"
              "    return image_codec.VideoCapture(path)\n"
              "def code():\n"
              "    return image_codec.VideoWriter_fourcc(*'mp4v'), VideoCapture\n")
    assert walls_problems(sample, WALLS_PACKAGE) == [
        (8, "torch"), (9, "splatpipe.depth.files"), (10, "splatpipe.depth.fit.fit"),
        (11, "splatpipe.depth"), (12, "cv2.VideoWriter"),
        (14, "image_codec.VideoCapture"), (16, "VideoCapture"),
        (16, "image_codec.VideoWriter_fourcc")]


def test_the_walls_step_imports_only_stdlib_numpy_scipy_cv2_and_splatpipe() -> None:
    """Req 1.11, 2.13, 22.8: and nothing from splatpipe.depth, and no cv2 video class. The files
    are found with glob, and the test fails when it finds none."""
    paths = walls_files()
    assert paths, ("splatpipe/walls/ holds no .py file, so the walls import check has nothing to "
                   "read. Make splatpipe/walls/__init__.py first.")
    found: list[tuple[str, str]] = []
    for path in paths:
        shown = path.relative_to(REPO_ROOT).as_posix()
        found += [(f"{shown}:{line}", what)
                  for line, what in walls_problems(path.read_bytes(), WALLS_PACKAGE)]
    assert found == []


def test_no_train_file_imports_the_walls_step() -> None:
    """Req 22.8: every splatpipe/train/*.py, found with glob, gpu.py included."""
    paths = sorted(SPLATPIPE.glob("train/*.py"))
    assert paths, "splatpipe/train/ holds no .py file, so this check has nothing to read."
    found = [(where, name) for path in paths for where, name in imports_of(path)
             if within(name, WALLS_PACKAGE)]
    assert found == []


def test_importing_splatpipe_walls_and_its_modules_loads_no_gpu_package() -> None:
    """Req 1.11, 22.8. The modules are every splatpipe/walls/*.py but __init__.py, found with
    glob, so a module a later task adds is imported as it lands. On a laptop, where torch, gsplat
    and moge aren't installed, a module that imported one would fail the fresh process."""
    modules = [f"{WALLS_PACKAGE}.{path.stem}" for path in walls_files()
               if path.stem != "__init__"]
    result = subprocess.run([sys.executable, "-c", LOADED_BY_WALLS, *modules], cwd=REPO_ROOT,
                            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
                            capture_output=True, encoding="utf-8", timeout=60)
    assert result.returncode == 0, result.stderr
    loaded = json.loads(result.stdout)
    assert {WALLS_PACKAGE, *modules} <= set(loaded)
    assert [name for name in loaded if top(name) in NEVER_LOADED_BY_WALLS] == []
