"""The Deploy_Command and the lock export, with a temporary git repo as the clone, a fake installer,
a fake engine fetch and the Fake_AWS. Nothing here reaches the network or AWS."""

from __future__ import annotations

import io
import shutil
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from app.deploy import (build_bundle, deploy_main, spend_strings, would_lose_table, zip_tree)
from app.lockfile import export_requirements
from cloud.errors import CloudError
from tests.fake_aws import FakeAws
from tests.website_helpers import REPO_ROOT, STACK, website_config

NOW = datetime(2026, 12, 1, 12, 0, tzinfo=timezone.utc)
CLOUDSHELL = {"AWS_EXECUTION_ENV": "CloudShell"}
YES = {**CLOUDSHELL, "APPROVED_SPEND": "1"}


def copy_into(repo: Path, relative: str) -> None:
    source = REPO_ROOT / relative
    target = repo / relative
    if source.is_dir():
        shutil.copytree(source, target, ignore=shutil.ignore_patterns("vendor", "__pycache__"))
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(source, target)


@pytest.fixture
def clone(tmp_path: Path) -> Path:
    if shutil.which("git") is None:
        pytest.skip("git isn't installed, so this test can't run.")
    repo = tmp_path / "repo"
    repo.mkdir()
    for relative in ("app", "config.toml", "uv.lock", "package-lock.json",
                     "scripts/fetch_viewer_engine.py", "cloud/aws-setup.yaml"):
        copy_into(repo, relative)
    for package in ("cloud", "splatpipe"):
        (repo / package).mkdir(exist_ok=True)
        for path in (REPO_ROOT / package).glob("*.py"):
            shutil.copy(path, repo / package / path.name)
    (repo / "splatpipe" / "frames").mkdir()
    (repo / "splatpipe" / "frames" / "step.py").write_text("x = 1\n", encoding="utf-8")
    (repo / "tests").mkdir()
    (repo / "tests" / "test_x.py").write_text("x = 1\n", encoding="utf-8")
    (repo / "docs").mkdir()
    (repo / "docs" / "notes.md").write_text("notes\n", encoding="utf-8")
    base = ["git", "-c", "init.defaultBranch=main", "-c", "core.autocrlf=false", "-c",
            "user.name=T", "-c", "user.email=t@example.invalid", "-c", "commit.gpgsign=false"]
    for command in (["init", "-q"], ["add", "--", "."], ["commit", "-q", "-m", "c"]):
        subprocess.run([*base, *command], cwd=repo, check=True, capture_output=True)
    return repo


def fake_install(requirements: Path, target: Path, env) -> None:
    assert "boto3==1.44.9" in requirements.read_text()
    assert env["PIP_CACHE_DIR"].startswith(str(requirements.parent))
    (target / "boto3").mkdir()
    (target / "boto3" / "__init__.py").write_text("__version__ = '1.44.9'\n")


def fake_fetch(script: Path, vendor: Path) -> None:
    assert script.name == "fetch_viewer_engine.py" and script.is_file()
    vendor.mkdir(parents=True, exist_ok=True)
    (vendor / "playcanvas.min.mjs").write_text("export const engine = 1;\n")
    (vendor / "PLAYCANVAS-LICENSE.txt").write_text("MIT\n")


def deployed_fake(**stack) -> FakeAws:
    fake = FakeAws(stack={**{k: v for k, v in STACK.items() if k != "WebsiteTableName"}, **stack})
    fake.table = None
    return fake


def deploy(clone: Path, fake: FakeAws, environ=YES, argv=()) -> tuple[int, str]:
    out = io.StringIO()
    code = deploy_main(list(argv), environ=environ, make_aws=lambda region: fake, repo_root=clone,
                       install=fake_install, out=out, now=NOW, folder=clone.parent,
                       fetch_engine=fake_fetch)
    return code, out.getvalue()


def test_a_deploy_uploads_the_bundle_and_makes_a_change_set(clone: Path) -> None:
    fake = deployed_fake()
    fake.changes = [("Add", "WebsiteTable", "False"), ("Modify", "JobBucket", "False")]
    code, out = deploy(clone, fake)
    assert code == 0, out
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=clone, capture_output=True,
                            text=True).stdout.strip()
    key = f"website/{commit}.zip"
    assert [op for op, _ in fake.calls] == ["stack_outputs", "s3_upload", "create_change_set",
                                            "change_set_changes"]
    request = fake.change_sets[0]
    assert request["ChangeSetName"] == f"website-{commit[:12]}-20261201120000"
    assert {"ParameterKey": "WebsiteCodeKey", "ParameterValue": key} in request["Parameters"]
    assert {"ParameterKey": "WebsiteCodeBucket",
            "ParameterValue": STACK["BucketName"]} in request["Parameters"]
    assert request["TemplateBody"] == (REPO_ROOT / "cloud" / "aws-setup.yaml").read_text()
    assert request["Tags"] == [{"Key": "project", "Value": "voxelis-kiroween"}]
    assert "Add WebsiteTable False" in out and "execute-change-set" in out and "arn:" not in out
    with zipfile.ZipFile(io.BytesIO(fake.objects[key])) as bundle:
        names = set(bundle.namelist())
    for wanted in ("config.toml", "app/handler.py", "app/lambda_function.py", "app/static/app.js",
                   "app/viewer/embed.js", "app/viewer/vendor/playcanvas.min.mjs", "cloud/aws.py",
                   "splatpipe/config.py", "baked/commit.json", "baked/code.tar", "boto3/__init__.py"):
        assert wanted in names, wanted
    for unwanted in ("app/uv.lock", "app/pyproject.toml", "app/viewer/README.md",
                     "splatpipe/frames/step.py", "tests/test_x.py", "docs/notes.md",
                     "cloud/aws-setup.yaml"):
        assert unwanted not in names, unwanted
    assert fake.table is None and not [op for op, _ in fake.calls if op.startswith("table_")]


def test_the_same_commit_builds_the_same_bytes(clone: Path) -> None:
    wc = website_config()
    first = build_bundle(clone, wc, fake_install, clone.parent / "one", fake_fetch).zip_path.read_bytes()
    second = build_bundle(clone, wc, fake_install, clone.parent / "two", fake_fetch).zip_path.read_bytes()
    assert first == second


def test_no_yes_means_no_aws_call(clone: Path, capsys) -> None:
    fake = deployed_fake()
    assert deploy(clone, fake, environ=CLOUDSHELL)[0] == 1
    err = capsys.readouterr().err
    assert "website GPU jobs (up to 10 and $52 a UTC day)" in err
    assert "every day until 2027-02-04 08:00 UTC" in err and "at most $500 in total" in err
    assert fake.calls == []


@pytest.mark.parametrize(("environ", "words"), [({}, "only in AWS CloudShell"),
                                                ({"AWS_EXECUTION_ENV": "AWS_Lambda_python3.11"},
                                                 "only in AWS CloudShell")])
def test_outside_cloudshell_is_refused(clone: Path, capsys, environ, words) -> None:
    fake = deployed_fake()
    assert deploy(clone, fake, environ=environ)[0] == 1
    assert words in capsys.readouterr().err and fake.calls == []


def test_outside_the_deploy_folder_and_a_dirty_tree_are_refused(clone: Path, tmp_path: Path,
                                                                 capsys) -> None:
    fake = deployed_fake()
    out = io.StringIO()
    assert deploy_main([], environ=YES, make_aws=lambda r: fake, repo_root=clone,
                       install=fake_install, out=out, now=NOW, folder=tmp_path / "elsewhere",
                       fetch_engine=fake_fetch) == 1
    assert "/tmp/voxelis-website" in capsys.readouterr().err
    (clone / "app" / "handler.py").write_text("changed\n")
    assert deploy(clone, fake)[0] == 1
    assert "uncommitted changes: app." in capsys.readouterr().err and fake.calls == []


def test_a_compiled_file_or_a_too_big_bundle_is_refused(clone: Path) -> None:
    def compiled_install(requirements, target, env):
        fake_install(requirements, target, env)
        (target / "boto3" / "speedups.so").write_bytes(b"\x7fELF")

    with pytest.raises(CloudError, match="compiled file"):
        build_bundle(clone, website_config(), compiled_install, clone.parent / "a", fake_fetch)
    with pytest.raises(CloudError, match="Lambda's limit"):
        build_bundle(clone, website_config(max_bundle_bytes=1000), fake_install,
                     clone.parent / "b", fake_fetch)


def test_a_change_set_that_would_lose_the_table_is_deleted(clone: Path, capsys) -> None:
    fake = deployed_fake(WebsiteTableName="table-fake")
    fake.changes = [("Modify", "WebsiteTable", "True")]
    assert deploy(clone, fake)[0] == 1
    assert "WebsiteTable" in capsys.readouterr().err
    assert len(fake.deleted_change_sets) == 1


@pytest.mark.parametrize(("changes", "deployed", "lost"), [
    ([("Add", "WebsiteTable", "False")], False, False),
    ([("Add", "WebsiteTable", "False")], True, True),
    ([("Modify", "WebsiteFunction", "False")], True, False),
    ([("Modify", "WebsiteFunction", "True")], True, True),
    ([("Remove", "WebsiteFunctionUrl", "False")], True, True),
    ([("Modify", "WebsiteFunctionUrl", "Conditional")], True, True),
    ([("Modify", "JobBucket", "True")], True, False),
])
def test_would_lose_table(changes, deployed, lost) -> None:
    assert bool(would_lose_table(changes, deployed)) is lost


def test_spend_strings_and_arguments() -> None:
    assert spend_strings(website_config(daily_cost_usd=52.5)) == (
        "website GPU jobs (up to 10 and $52.50 a UTC day)", "every day until 2027-02-04 08:00 UTC",
        "at most $500 in total")


def test_the_lock_export_lists_boto3s_closure_with_hashes() -> None:
    text = export_requirements((REPO_ROOT / "app" / "uv.lock").read_text())
    names = [line.split("==")[0] for line in text.splitlines() if "==" in line]
    assert names == sorted(["boto3", "botocore", "jmespath", "python-dateutil", "s3transfer", "six",
                            "urllib3"])
    assert "voxelis-website" not in text and "--hash=sha256:" in text
    twice = '[[package]]\nname = "a"\nversion = "1"\nsource = { registry = "x" }\n' \
            'wheels = [{ hash = "sha256:00" }]\n'
    with pytest.raises(CloudError, match="twice"):
        export_requirements(twice + twice)
    with pytest.raises(CloudError, match="no hashed wheel"):
        export_requirements('[[package]]\nname = "a"\nversion = "1"\nsource = { registry = "x" }\n')


def test_zip_tree_round_trips(tmp_path: Path) -> None:
    (tmp_path / "t" / "a").mkdir(parents=True)
    (tmp_path / "t" / "a" / "x.txt").write_bytes(b"x")
    (tmp_path / "t" / "y.bin").write_bytes(bytes(range(256)))
    zip_tree(tmp_path / "t", tmp_path / "1.zip")
    zip_tree(tmp_path / "t", tmp_path / "2.zip")
    assert (tmp_path / "1.zip").read_bytes() == (tmp_path / "2.zip").read_bytes()
    with zipfile.ZipFile(tmp_path / "1.zip") as archive:
        assert archive.read("a/x.txt") == b"x" and archive.read("y.bin") == bytes(range(256))
