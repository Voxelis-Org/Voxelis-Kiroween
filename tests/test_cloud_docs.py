"""Tests for the cloud-jobs dependencies and their Licence_List rows (cloud-jobs task 1).

Each test reads the real file at the repo root, so it checks what `uv sync` and `npm ci` would
install and what `docs/third-party-licences.md` records. These run on the CPU with no network.

The doc, steering and skill checks for this file are a later sub-task (task 9) and are not here yet.
"""

import json
import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # this file is in tests/, at the repo root
PYPROJECT = ROOT / "pyproject.toml"
PACKAGE_JSON = ROOT / "package.json"
PACKAGE_LOCK = ROOT / "package-lock.json"
LICENCE_LIST = ROOT / "docs" / "third-party-licences.md"

SPLAT_TRANSFORM = "@playcanvas/splat-transform"
SPLAT_TRANSFORM_VERSION = "3.8.0"

# The packages adding boto3 puts in uv.lock, each needing its own Licence_List row (Req 14.3).
BOTO3_TREE = ("boto3", "botocore", "s3transfer", "jmespath", "python-dateutil", "six", "urllib3")

# GPU packages the laptop install must never pull in (Req 14.2), as normalised distribution names.
GPU_NAMES = frozenset({"torch", "gsplat", "fused-ssim"})
GPU_PREFIXES = ("nvidia", "cuda")


def requirement_name(spec: str) -> str:
    """The distribution name at the start of a PEP 508 dependency string, normalised (PEP 503):
    lowercased, with every run of `-`, `_` or `.` turned into a single `-`. So `fused_ssim>=1`
    and `fused-ssim` both give `fused-ssim`."""
    match = re.match(r"[A-Za-z0-9._-]+", spec.strip())
    return re.sub(r"[-_.]+", "-", match.group(0)).lower() if match else ""


def is_gpu_package(name: str) -> bool:
    """True when a normalised distribution name is one of the GPU packages: torch, gsplat,
    fused-ssim, or an NVIDIA/CUDA wheel such as `nvidia-cuda-runtime-cu12` (Req 14.2)."""
    return name in GPU_NAMES or any(
        name == prefix or name.startswith(prefix + "-") for prefix in GPU_PREFIXES)


def declared_dependencies() -> list[str]:
    """Every dependency string pyproject.toml declares: the main runtime list and the dev groups,
    because `uv sync` installs both on a laptop."""
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    specs = list(data.get("project", {}).get("dependencies", []))
    for group in data.get("dependency-groups", {}).values():
        specs.extend(spec for spec in group if isinstance(spec, str))
    return specs


def table_rows(text: str) -> list[list[str]]:
    """The stripped cells of each Markdown table row in the text, without the |---| divider rows."""
    rows = [[cell.strip() for cell in line.strip()[1:-1].split("|")]
            for line in text.splitlines() if line.startswith("|")]
    return [row for row in rows if not all(cell and not cell.strip("-:") for cell in row)]


def licence_row(contains: str) -> dict[str, str]:
    """The Licence_List's one row whose Name cell contains `contains`, each cell under its heading."""
    header, *rows = table_rows(LICENCE_LIST.read_text(encoding="utf-8"))
    matches = [dict(zip(header, row, strict=True)) for row in rows if contains in row[0]]
    assert len(matches) == 1, f"the Licence_List should have one {contains!r} row, not {len(matches)}"
    return matches[0]


def lockfile_licences() -> set[str]:
    """Every distinct `license` across the package-lock.json node_modules entries (Req 14.4)."""
    lock = json.loads(PACKAGE_LOCK.read_text(encoding="utf-8"))
    return {entry["license"] for path, entry in lock["packages"].items()
            if path.startswith("node_modules/") and "license" in entry}


def test_pyproject_has_boto3_in_the_main_dependencies() -> None:
    """Req 14.1."""
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    names = [requirement_name(spec) for spec in data["project"]["dependencies"]]
    assert "boto3" in names, "pyproject.toml's [project].dependencies should include boto3"


def test_pyproject_declares_no_gpu_package() -> None:
    """Req 14.2."""
    gpu = sorted({name for spec in declared_dependencies()
                  if is_gpu_package(name := requirement_name(spec))})
    assert not gpu, (
        f"pyproject.toml should declare no GPU package (torch, gsplat, fused-ssim, a CUDA/NVIDIA "
        f"wheel); they belong only in the cloud machine image, but found {gpu}")


def test_package_json_pins_splat_transform_at_3_8_0() -> None:
    """Req 14.4."""
    data = json.loads(PACKAGE_JSON.read_text(encoding="utf-8"))
    pinned = data.get("dependencies", {}).get(SPLAT_TRANSFORM)
    assert pinned == SPLAT_TRANSFORM_VERSION, (
        f"package.json should pin {SPLAT_TRANSFORM} at exactly {SPLAT_TRANSFORM_VERSION}, "
        f"not {pinned!r}")


def test_package_lock_pins_splat_transform_at_3_8_0() -> None:
    """Req 14.4."""
    lock = json.loads(PACKAGE_LOCK.read_text(encoding="utf-8"))
    required = lock["packages"][""]["dependencies"].get(SPLAT_TRANSFORM)
    resolved = lock["packages"][f"node_modules/{SPLAT_TRANSFORM}"]["version"]
    assert required == SPLAT_TRANSFORM_VERSION, (
        f"package-lock.json's root should require {SPLAT_TRANSFORM} {SPLAT_TRANSFORM_VERSION}, "
        f"not {required!r}")
    assert resolved == SPLAT_TRANSFORM_VERSION, (
        f"package-lock.json should resolve {SPLAT_TRANSFORM} to {SPLAT_TRANSFORM_VERSION}, "
        f"not {resolved!r}")


def test_the_licence_list_has_a_row_for_boto3_and_each_package_it_pulls_in() -> None:
    """Req 14.3."""
    for name in BOTO3_TREE:
        row = licence_row(name)  # asserts exactly one row with this Name exists
        assert row["Licence"], f"the {name!r} Licence_List row should name a licence"
    assert licence_row("boto3")["Installed"] == "Yes", (
        "the Licence_List should mark boto3 as installed")


def test_the_splat_transform_row_covers_every_npm_licence_in_the_lockfile() -> None:
    """Req 14.4. One splat-transform row covers the whole npm tree, so no npm package needs its
    own row: every licence resolved in package-lock.json is MIT, BSD-3-Clause or ISC, which the
    splat-transform row already names."""
    licence = licence_row(SPLAT_TRANSFORM)["Licence"]
    for named in ("MIT", "BSD-3-Clause", "ISC"):
        assert named in licence, f"the splat-transform Licence_List row should name {named}"
    uncovered = sorted(lic for lic in lockfile_licences() if lic not in licence)
    assert not uncovered, (
        f"the splat-transform row should cover every npm licence, but {uncovered} is not named; "
        "add a Licence_List row for any npm package its licence doesn't cover (Req 14.4)")
