"""What app/ may import: the standard library, app, cloud and splatpipe's top-level modules, never
boto3 or botocore and never a private name. And splatpipe/ never imports app."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

from tests.website_helpers import REPO_ROOT

APP = REPO_ROOT / "app"
SPLATPIPE_TOP = {path.stem for path in (REPO_ROOT / "splatpipe").glob("*.py")}


def app_files() -> list[Path]:
    return sorted(path for path in APP.rglob("*.py") if "viewer" not in path.parts)


def imports(path: Path) -> list[tuple[str, list[str]]]:
    """(module, imported names) for every import in a file."""
    found: list[tuple[str, list[str]]] = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            found.extend((alias.name, []) for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            found.append((node.module, [alias.name for alias in node.names]))
    return found


def allowed(module: str) -> bool:
    top = module.split(".")[0]
    if top in {"app", "cloud"}:
        return True
    if top == "splatpipe":
        parts = module.split(".")
        return len(parts) == 1 or (len(parts) == 2 and parts[1] in SPLATPIPE_TOP)
    return top in sys.stdlib_module_names


@pytest.mark.parametrize("path", app_files(), ids=lambda path: str(path.relative_to(REPO_ROOT)))
def test_an_app_module_imports_only_what_it_may(path: Path) -> None:
    for module, names in imports(path):
        if module == "__future__":
            continue
        assert module.split(".")[0] not in {"boto3", "botocore"}, module
        assert allowed(module), module
        assert not any(name.startswith("_") for name in names), (module, names)
        assert not any(part.startswith("_") for part in module.split(".")), module


@pytest.mark.parametrize("path", app_files(), ids=lambda path: str(path.relative_to(REPO_ROOT)))
def test_no_app_module_uses_secrets_manager_or_parameter_store(path: Path) -> None:
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            assert node.value not in {"secretsmanager", "ssm"}


def test_splatpipe_never_imports_app() -> None:
    for path in sorted((REPO_ROOT / "splatpipe").rglob("*.py")):
        for module, _ in imports(path):
            assert module.split(".")[0] != "app", path


def test_no_test_writes_the_spend_variable_into_the_environment() -> None:
    name = "APPROVED" + "_SPEND"
    for path in sorted((REPO_ROOT / "tests").glob("*.py")):
        text = path.read_text(encoding="utf-8")
        assert f'os.environ["{name}"]' not in text, path
        assert f'setenv("{name}"' not in text, path
