"""The import checks (cloud-jobs task 9.2): read the files with ast and run nothing (Req 1.1, 1.2, 16.7).

Three boundaries keep the laptop and the cloud apart. Each is checked by parsing every file with
ast, so no module is imported and no Step runs:

- No file under `splatpipe/` imports `cloud` or boto3, so the Pipeline never learns where it runs
  and never depends on the cloud wrapper (Req 1.1). No file under `splatpipe/` imports a GPU
  package either, except `splatpipe/train/gpu.py`, the GPU_Module, which may import torch, gsplat
  and fused_ssim and no other GPU package (Req 16.7; splat-training spec, Req 14.8, A14). The
  Train_Step loads the GPU_Module only when it runs, so the rest of the Pipeline still imports on
  a laptop with no GPU libraries.
- boto3 is imported only under `cloud/`, so the AWS SDK lives only in the cloud wrapper (Req 1.2).
- No `cloud/*.py` imports a GPU package at module level, so every `cloud` module imports on a laptop
  that has no GPU libraries; those packages live only in the Machine_Image. A lazy import inside a
  function is allowed, because it only runs on the machine (Req 16.7).

Each failure names the file and the line of every offending import.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]  # this file is in tests/, at the repo root
SPLATPIPE = REPO_ROOT / "splatpipe"
CLOUD = REPO_ROOT / "cloud"

# The GPU-only packages, by top-level import name: PyTorch and its siblings, the splat trainer
# gsplat, fused-ssim, the MoGe depth model, nvdiffrast, and the NVIDIA CUDA wheels torch pulls in
# (tech.md, rules.md). None is installed on a laptop, so importing one at module level would make
# `import cloud.<module>` fail on a laptop.
GPU_PACKAGES = frozenset({
    "torch", "torchvision", "torchaudio",
    "gsplat", "fused_ssim", "moge", "nvdiffrast", "nvidia",
})

# The GPU_Module, the one file under splatpipe/ that may import a GPU package, and the only GPU
# packages it may import (splat-training spec, Req 14.8, A14).
GPU_MODULE_FILE = SPLATPIPE / "train" / "gpu.py"
GPU_MODULE_PACKAGES = frozenset({"torch", "gsplat", "fused_ssim"})


class _Imports(ast.NodeVisitor):
    """Collect every import in one file as (line, top_level_name, module_level).

    `top_level_name` is the first dotted part of the imported package: `import torch.nn` and
    `from torch import nn` both name `torch`. A relative import (`from . import x`) can never name an
    outside package, so it resolves to the file's own top-level package. `module_level` is False only
    for an import inside a function body, where it runs lazily when the function is called rather than
    at import time; a class-body import still runs at import time, so it counts as module level.
    """

    def __init__(self, package: str) -> None:
        self.package = package  # the file's own top-level package, "splatpipe" or "cloud"
        self.found: list[tuple[int, str, bool]] = []
        self._in_function = 0  # > 0 while inside a def, where imports are deferred

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.found.append((node.lineno, alias.name.split(".", 1)[0], self._in_function == 0))

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        top = self.package if node.level else (node.module or "").split(".", 1)[0]
        if top:
            self.found.append((node.lineno, top, self._in_function == 0))

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._in_function += 1
        self.generic_visit(node)
        self._in_function -= 1

    visit_AsyncFunctionDef = visit_FunctionDef  # type: ignore[assignment]


def imports_of(path: Path) -> list[tuple[str, str, bool]]:
    """Every import in the file at `path` as ("<file>:<line>", top_level_name, module_level).

    The file is read as bytes so ast decodes it exactly as Python would, and its own top-level
    package is the first part of its repo-relative path.
    """
    package = path.relative_to(REPO_ROOT).parts[0]
    shown = path.relative_to(REPO_ROOT).as_posix()
    finder = _Imports(package)
    finder.visit(ast.parse(path.read_bytes()))
    return [(f"{shown}:{line}", top, module_level)
            for line, top, module_level in sorted(finder.found)]


def test_the_reader_finds_imports_with_their_line_and_whether_they_are_module_level() -> None:
    """A plain import, a from-import and a relative import are module level; an import inside a
    function is not, and the relative import resolves to the file's own package."""
    sample = ("import torch\n"                 # 1: module level
              "from boto3.session import S\n"   # 2: module level
              "from . import errors\n"          # 3: module level, relative -> the file's package
              "\n"
              "def later():\n"
              "    import gsplat\n")             # 6: inside a function, not module level
    finder = _Imports("cloud")
    finder.visit(ast.parse(sample))
    assert sorted(finder.found) == [
        (1, "torch", True),
        (2, "boto3", True),
        (3, "cloud", True),
        (6, "gsplat", False),
    ]


def test_nothing_under_splatpipe_imports_cloud_boto3_or_a_gpu_package() -> None:
    """Every *.py under splatpipe/, step folders included (rglob), so a folder added later counts
    too. None imports cloud or boto3, and none but splatpipe/train/gpu.py, the GPU_Module, imports
    a GPU package, anywhere in the file. The GPU_Module may import torch, gsplat and fused_ssim and
    no other GPU package (Req 1.1, 16.7; splat-training spec, Req 14.8, A14)."""
    offenders: list[str] = []
    for path in sorted(SPLATPIPE.rglob("*.py")):
        allowed = GPU_MODULE_PACKAGES if path == GPU_MODULE_FILE else frozenset()
        forbidden = ({"cloud", "boto3"} | GPU_PACKAGES) - allowed
        offenders += [f"{where} imports {top}"
                      for where, top, _module_level in imports_of(path)
                      if top in forbidden]
    assert offenders == [], ("splatpipe must import no cloud, boto3 or GPU package, but "
                             "splatpipe/train/gpu.py may import torch, gsplat and fused_ssim: "
                             + "; ".join(offenders))


def test_boto3_is_imported_only_under_cloud() -> None:
    """Across the Pipeline and the cloud wrapper, boto3 is imported only in files under cloud/, so
    the AWS SDK stays inside the cloud wrapper (Req 1.2)."""
    offenders = [where
                 for path in sorted(SPLATPIPE.rglob("*.py")) + sorted(CLOUD.glob("*.py"))
                 for where, top, _module_level in imports_of(path)
                 if top == "boto3" and CLOUD not in path.parents]
    assert offenders == [], "boto3 may be imported only under cloud/: " + "; ".join(offenders)


def test_no_cloud_module_imports_a_gpu_package_at_module_level() -> None:
    """No cloud/*.py imports a GPU package at module level, so every cloud module imports on a
    laptop with no GPU libraries; a lazy import inside a function is allowed (Req 16.7)."""
    offenders = [f"{where} imports {top}"
                 for path in sorted(CLOUD.glob("*.py"))
                 for where, top, module_level in imports_of(path)
                 if module_level and top in GPU_PACKAGES]
    assert offenders == [], "a cloud module imports a GPU package at module level: " + "; ".join(offenders)
