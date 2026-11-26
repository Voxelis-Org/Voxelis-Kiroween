"""The App_Project, app/pyproject.toml and app/uv.lock: boto3 only, at the root lock's versions."""

from __future__ import annotations

import tomllib

from tests.website_helpers import REPO_ROOT

BOTO3_CLOSURE = {"boto3", "botocore", "s3transfer", "jmespath", "python-dateutil", "six", "urllib3"}


def lock_versions(path) -> dict[str, str]:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    return {package["name"]: package["version"] for package in data["package"]
            if "registry" in package.get("source", {})}


def test_the_app_project_lists_only_boto3_pinned() -> None:
    project = tomllib.loads((REPO_ROOT / "app" / "pyproject.toml").read_text(encoding="utf-8"))
    assert project["project"]["name"] == "voxelis-website"
    assert project["project"]["dependencies"] == ["boto3==1.44.9"]
    assert project["project"]["requires-python"] == ">=3.11,<3.12"
    pinned = {line.split("==")[0] for line in project["tool"]["uv"]["constraint-dependencies"]}
    assert pinned == BOTO3_CLOSURE - {"boto3"}


def test_the_app_lock_holds_boto3s_closure_at_the_root_locks_versions() -> None:
    app = lock_versions(REPO_ROOT / "app" / "uv.lock")
    root = lock_versions(REPO_ROOT / "uv.lock")
    assert set(app) == BOTO3_CLOSURE
    for name, version in app.items():
        assert root[name] == version, name


def test_the_root_project_doesnt_hold_the_website() -> None:
    root = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert root["project"]["name"] == "voxelis-kiroween"
    assert "voxelis-website" not in (REPO_ROOT / "uv.lock").read_text(encoding="utf-8")
