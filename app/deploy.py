"""The Deploy_Command: `APPROVED_SPEND=1 python3.11 -m app.deploy`, run by the account owner only,
in AWS CloudShell, from a clone under `[website].deploy_folder`.

It builds the website bundle from HEAD's commit (never from the working tree), asks the Spend_Guard
with the website limits, which is the account owner's one yes for every job the website starts,
uploads the bundle to `website/<commit>.zip` in the job bucket, and makes, but doesn't execute, a
change set. It refuses a change set that would lose the website's table or change its address.
"""

from __future__ import annotations

import io
import os
import re
import shutil
import subprocess
import sys
import tarfile
import zipfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, TextIO

from splatpipe.config import load_config

from app.baked import write_baked
from app.cloudshell import require_cloudshell, run_cloudshell_cli
from app.config import website_config_from
from app.lockfile import export_requirements
from cloud.aws import aws_setting, cloudshell_aws
from cloud.errors import CloudError
from cloud.repo import archive_head, head_commit, lock_hashes, uncommitted
from cloud.spend import check_spend

if TYPE_CHECKING:
    from app.config import WebsiteConfig
    from cloud.aws import Aws

Installer = Callable[[Path, Path, Mapping[str, str]], None]      # install(requirements, target, env)
EngineFetcher = Callable[[Path, Path], None]                       # fetch(script, vendor folder)

# The most --template-body takes (CloudFormation quotas).
MAX_TEMPLATE_BODY_BYTES = 51_200
CHECKED_PATHS = ("app", "splatpipe", "cloud", "config.toml")
COMPILED = (".so", ".pyd", ".dylib")
ZIP_TIME = (1980, 1, 1, 0, 0, 0)
FILE_MODE = 0o644
BUNDLE_KEY = re.compile(r"website/[0-9a-f]{40}\.zip")
_CENTS = 100
_ENGINE_SCRIPT = "scripts/fetch_viewer_engine.py"
_LOSING = {"WebsiteTable": ("Remove", "Add"), "WebsiteFunction": ("Remove",),
           "WebsiteFunctionUrl": ("Remove",)}


@dataclass(frozen=True)
class Bundle:
    zip_path: Path
    commit: str
    uv_lock_hash: str
    package_lock_hash: str
    template_body: str


def deploy_main(argv: Sequence[str], *, environ: Mapping[str, str],
                make_aws: Callable[[str], Aws], repo_root: Path, install: Installer,
                out: TextIO, now: datetime, folder: Path | None = None,
                fetch_engine: EngineFetcher | None = None) -> int:
    def body() -> int:
        if argv:
            raise CloudError("The deploy command takes no arguments. Run "
                             "`APPROVED_SPEND=1 python3.11 -m app.deploy`.")
        require_cloudshell(environ)
        root = Path(repo_root).resolve()
        config = load_config(root / "config.toml")
        wc = website_config_from(config)
        base = Path(str(wc.deploy_folder)) if folder is None else Path(folder)
        if not root.is_relative_to(base.resolve()):
            raise CloudError(f"Run the deploy command from a clone inside {wc.deploy_folder}, as "
                             "docs/aws-setup.md says, so nothing goes in the home folder.")
        dirty = uncommitted(CHECKED_PATHS, cwd=root)
        if dirty:
            raise CloudError(f"These have uncommitted changes: {', '.join(dirty)}. The website must "
                             "run a commit, so commit or stash them, then deploy again.")
        bundle = build_bundle(root, wc, install, base, fetch_engine or script_fetcher(sys.executable))
        if len(bundle.template_body.encode("utf-8")) > MAX_TEMPLATE_BODY_BYTES:
            raise CloudError("cloud/aws-setup.yaml is over 51,200 bytes, the most a change set's "
                             "template body takes. Shorten it, then deploy again.")
        machine, expected_time, cost = spend_strings(wc)
        check_spend(machine=machine, expected_time=expected_time, rough_cost=cost, environ=environ)
        return _publish(config, make_aws, bundle, out, now)

    return run_cloudshell_cli(body)


def _publish(config: Any, make_aws: Callable[[str], Aws], bundle: Bundle, out: TextIO,
             now: datetime) -> int:
    stack_name = aws_setting(config, "stack_name")
    aws = make_aws(aws_setting(config, "region"))
    outputs = aws.stack_outputs(stack_name)
    bucket = outputs.get("BucketName", "")
    if not bucket:
        raise CloudError(f"The stack {stack_name!r} has no BucketName output. Create the stack "
                         "first, as docs/aws-setup.md says.")
    aws.bucket = bucket
    key = f"website/{bundle.commit}.zip"
    name = f"website-{bundle.commit[:12]}-{now:%Y%m%d%H%M%S}"
    request = change_set_request(stack_name, name, bundle, bucket, key,
                                 _project_tag(aws_setting(config, "project_tag")), outputs)
    aws.s3_upload(bundle.zip_path, key)
    aws.create_change_set(request)
    changes = aws.change_set_changes(stack_name, name)
    why = would_lose_table(changes, "WebsiteTableName" in outputs)
    if why:
        aws.delete_change_set(stack_name, name)
        raise CloudError(f"{why} The change set was deleted and nothing changed.")
    for action, logical, replacement in changes:
        print(f"{action} {logical} {replacement}", file=out)
    print(f"Made the change set {name}. Review it, then execute it:", file=out)
    print(f"  aws cloudformation describe-change-set --stack-name {stack_name} "
          f"--change-set-name {name} --region {aws.region}", file=out)
    print(f"  aws cloudformation execute-change-set --stack-name {stack_name} "
          f"--change-set-name {name} --region {aws.region}", file=out)
    return 0


def spend_strings(wc: WebsiteConfig) -> tuple[str, str, str]:
    stop = wc.stop_starting_at
    return (f"website GPU jobs (up to {wc.daily_jobs} and {_dollars(wc.daily_cost_cents)} a UTC day)",
            f"every day until {stop:%Y-%m-%d %H:%M} UTC",
            f"at most {_dollars(wc.total_cost_cents)} in total")


def change_set_request(stack_name: str, change_set_name: str, bundle: Bundle, bucket: str,
                       key: str, project_tag: tuple[str, str],
                       outputs: Mapping[str, str]) -> dict[str, Any]:
    """The change set, with the code parameters only from the Stack's own bucket and the commit."""
    if bucket != outputs.get("BucketName") or key != f"website/{bundle.commit}.zip" \
            or not BUNDLE_KEY.fullmatch(key):
        raise CloudError("The website's code must come from the stack's own job bucket, under "
                         "website/<commit>.zip. This is a bug if you see it.")
    tag_key, tag_value = project_tag
    return {
        "StackName": stack_name,
        "ChangeSetName": change_set_name,
        "ChangeSetType": "UPDATE",
        "TemplateBody": bundle.template_body,
        "Capabilities": ["CAPABILITY_IAM"],
        "Parameters": [
            {"ParameterKey": "DefaultVpcId", "UsePreviousValue": True},
            {"ParameterKey": "WebsiteCodeBucket", "ParameterValue": bucket},
            {"ParameterKey": "WebsiteCodeKey", "ParameterValue": key},
        ],
        "Tags": [{"Key": tag_key, "Value": tag_value}],
        "Description": f"Website at commit {bundle.commit[:12]}",
    }


def would_lose_table(changes: Sequence[tuple[str, str, str]], deployed: bool) -> str | None:
    """Why a change set would lose the table or change the address, once the website is deployed."""
    if not deployed:
        return None
    # A code update shows the URL as a Conditional replacement, because the change set can't read
    # the modified function's GetAtt. The function's ARN is the URL's only input that can change,
    # and an update that doesn't replace the function keeps it, so that one pair is safe.
    function_kept = ("Modify", "WebsiteFunction", "False") in changes
    for action, logical, replacement in changes:
        if function_kept and (action, logical, replacement) == ("Modify", "WebsiteFunctionUrl",
                                                               "Conditional"):
            continue
        if logical in _LOSING and (action in _LOSING[logical] or replacement in ("True", "Conditional")):
            return (f"This change set would {action.lower()} {logical} (replacement: {replacement}), "
                    "which would lose the website's table or change its address.")
    return None


# Building the bundle

def build_bundle(repo_root: Path, wc: WebsiteConfig, install: Installer, folder: Path,
                 fetch_engine: EngineFetcher) -> Bundle:
    folder = Path(folder)
    build = folder / "build"
    if build.exists():
        shutil.rmtree(build)
    (build / "baked").mkdir(parents=True)
    commit = head_commit(cwd=repo_root)
    archive = build / "baked" / "code.tar"
    archive_head(archive, cwd=repo_root)
    uv_hash, package_hash = lock_hashes(cwd=repo_root)
    with tarfile.open(archive) as tar:
        files = _committed_files(tar)
    for name, data in files.items():
        if _bundled(name):
            _write(build / name, data)
    write_baked(build, commit, uv_hash, package_hash)
    requirements = folder / "requirements.txt"
    requirements.write_text(export_requirements(files["app/uv.lock"].decode("utf-8")),
                            encoding="utf-8")
    install(requirements, build, _install_env(folder))
    tools = folder / "tools"
    _write(tools / _ENGINE_SCRIPT, files[_ENGINE_SCRIPT])
    fetch_engine(tools / _ENGINE_SCRIPT, build / "app" / "viewer" / "vendor")
    _check_tree(build, wc.max_bundle_bytes)
    zip_path = folder / f"website-{commit}.zip"
    zip_tree(build, zip_path)
    return Bundle(zip_path=zip_path, commit=commit, uv_lock_hash=uv_hash,
                  package_lock_hash=package_hash,
                  template_body=files["cloud/aws-setup.yaml"].decode("utf-8"))


def _committed_files(tar: tarfile.TarFile) -> dict[str, bytes]:
    files: dict[str, bytes] = {}
    for member in tar.getmembers():
        path = PurePosixPath(member.name)
        if not member.isfile() or path.is_absolute() or ".." in path.parts:
            continue
        handle = tar.extractfile(member)
        if handle is not None:
            files[str(path)] = handle.read()
    return files


def _bundled(name: str) -> bool:
    """Whether a committed file goes in the bundle: app/ without its project files and docs, the
    top-level cloud and splatpipe modules, and config.toml."""
    path = PurePosixPath(name)
    if name == "config.toml":
        return True
    if path.parts[0] == "app":
        return name not in ("app/uv.lock", "app/pyproject.toml") and path.suffix != ".md"
    return len(path.parts) == 2 and path.parts[0] in ("cloud", "splatpipe") and path.suffix == ".py"


def _write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _install_env(folder: Path) -> dict[str, str]:
    cache = folder / "cache"
    return {**os.environ, "XDG_CACHE_HOME": str(cache), "PIP_CACHE_DIR": str(cache / "pip")}


def _check_tree(build: Path, max_bytes: int) -> None:
    files = [path for path in build.rglob("*") if path.is_file()]
    compiled = [path for path in files if path.suffix in COMPILED]
    if compiled:
        raise CloudError(f"The bundle holds a compiled file, {compiled[0].relative_to(build)}, but "
                         "boto3's packages are pure Python, so the wrong package came in. Check "
                         "app/uv.lock.")
    if sum(path.stat().st_size for path in files) > max_bytes:
        raise CloudError(f"The bundle is over {max_bytes} bytes unzipped, Lambda's limit.")


def zip_tree(root: Path, zip_path: Path) -> None:
    """Zip `root` with sorted entries, fixed times and one mode, so one tree gives the same bytes."""
    paths = sorted(path for path in Path(root).rglob("*") if path.is_file())
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            info = zipfile.ZipInfo(path.relative_to(root).as_posix(), date_time=ZIP_TIME)
            info.external_attr = FILE_MODE << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, path.read_bytes())
    Path(zip_path).write_bytes(buffer.getvalue())


def pip_installer(python: str) -> Installer:
    def install(requirements: Path, target: Path, env: Mapping[str, str]) -> None:
        subprocess.run([python, "-m", "pip", "install", "--target", str(target), "--no-deps",
                        "--require-hashes", "--only-binary=:all:", "--no-compile", "--no-cache-dir",
                        "--disable-pip-version-check", "-r", str(requirements)],
                       check=True, env=dict(env))
    return install


def script_fetcher(python: str) -> EngineFetcher:
    def fetch(script: Path, vendor: Path) -> None:
        subprocess.run([python, str(script), "--into", str(vendor)], check=True)
    return fetch


def _dollars(cents: int) -> str:
    return f"${cents // _CENTS}" if cents % _CENTS == 0 else f"${cents / _CENTS:.2f}"


def _project_tag(raw: str) -> tuple[str, str]:
    key, sep, value = raw.partition("=")
    if not sep or not key or not value:
        raise CloudError('[aws] project_tag in config.toml must be "key=value".')
    return key, value


def main() -> int:
    return deploy_main(sys.argv[1:], environ=os.environ, make_aws=cloudshell_aws,
                       repo_root=Path(__file__).resolve().parents[1],
                       install=pip_installer(sys.executable), out=sys.stdout,
                       now=datetime.now(timezone.utc))


if __name__ == "__main__":
    sys.exit(main())
