"""Tests for the cloud-jobs dependencies and docs (cloud-jobs tasks 1 and 9).

Each test reads the real file at the repo root. The task-1 part checks what `uv sync` and `npm ci`
would install and what `docs/third-party-licences.md` records. The task-9 part checks the cloud-run
write-up in `docs/pipeline.md`, the README's "Run your own copy" section, and the quota-based
admission, one-yes and redaction updates in the Tech_Steering and the aws-jobs skill (Req 15).
These run on the CPU with no network.
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
PIPELINE_DOC = ROOT / "docs" / "pipeline.md"
README = ROOT / "README.md"
TECH_STEERING = ROOT / ".kiro" / "steering" / "tech.md"
AWS_JOBS_SKILL = ROOT / ".kiro" / "skills" / "aws-jobs" / "SKILL.md"

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


# --- Task 9: the doc, steering and skill checks (Req 15) ---------------------------------------

# The per-Step Step_Needs plan of Requirement 2's note, as {Step: substrings the row must contain}.
STEP_NEEDS_PLAN = {
    "Picking frames": ("16 vCPUs", "16 GB RAM"),
    "Solving cameras": ("16 vCPUs", "16 GB RAM"),
    "Depth priors": ("GPU", "16 GB VRAM"),
    "Plain walls": ("32 GB RAM",),
    "Training": ("GPU", "32 GB RAM"),
    "Clean and export": ("8 vCPUs", "8 GB RAM"),
}


def markdown_section(text: str, heading: str) -> str:
    """The text of the one `## <heading>` section: from its heading line to the line before the next
    `#` or `##` heading, or the end. A `###` subsection of the section stays inside it, and a line in
    a fenced ``` block is never read as a heading (so a `# comment` in a code example doesn't count)."""
    lines = text.splitlines()
    starts, fenced = [], False
    for index, line in enumerate(lines):
        if line.lstrip().startswith("```"):
            fenced = not fenced
        elif not fenced and line.strip() == f"## {heading}":
            starts.append(index)
    assert len(starts) == 1, f"expected one '## {heading}' heading, found {len(starts)}"
    start, end, fenced = starts[0], len(lines), False
    for index in range(start + 1, len(lines)):
        if lines[index].lstrip().startswith("```"):
            fenced = not fenced
        elif not fenced and re.match(r"#{1,2} ", lines[index]):
            end = index
            break
    return "\n".join(lines[start:end])


def test_the_pipeline_doc_has_the_cloud_run_section() -> None:
    """Req 15.2: the Pipeline_Doc describes the cloud run and its commands."""
    section = markdown_section(PIPELINE_DOC.read_text(encoding="utf-8"), "Running a job in the cloud")
    for command in ("cloud.start", "cloud.status", "cloud.download", "cloud.list", "cloud.stopall"):
        assert command in section, (
            f"the cloud-run section of docs/pipeline.md should describe {command}")
    assert "Cloud_Runner" in section, "the cloud-run section should describe the Cloud_Runner"


def test_the_pipeline_doc_documents_the_cap_deadline_variable() -> None:
    """Req 15.2 (assumption C): the Cap_Deadline environment variable's name and meaning."""
    section = markdown_section(PIPELINE_DOC.read_text(encoding="utf-8"), "Running a job in the cloud")
    assert "SPLATPIPE_CAP_DEADLINE" in section, (
        "the cloud-run section should name the SPLATPIPE_CAP_DEADLINE environment variable")
    assert "upload_margin_minutes" in section, (
        "the cloud-run section should explain the cap deadline is the time cap minus "
        "upload_margin_minutes")


def test_the_pipeline_doc_records_the_per_step_step_needs_plan() -> None:
    """Req 15.2 (assumptions D and E): the per-Step Step_Needs plan and the three min_* keys."""
    section = markdown_section(PIPELINE_DOC.read_text(encoding="utf-8"), "Running a job in the cloud")
    for key in ("min_vcpus", "min_memory_gb", "min_gpu_memory_gb"):
        assert key in section, f"the Step_Needs plan should name {key}"
    rows = {cells[0]: cells[1] for cells in table_rows(section) if len(cells) == 2}
    for step, needs in STEP_NEEDS_PLAN.items():
        assert step in rows, f"the Step_Needs plan should have a row for {step!r}"
        for need in needs:
            assert need in rows[step], (
                f"the {step!r} Step_Needs row should say {need!r}, not {rows[step]!r}")


def test_the_pipeline_doc_says_the_cloud_run_is_not_a_step() -> None:
    """Req 15.7: the cloud run is not a Step and adds no Step_List entry."""
    section = markdown_section(PIPELINE_DOC.read_text(encoding="utf-8"), "Running a job in the cloud")
    lowered = section.lower()
    assert "not a step" in lowered, "the cloud-run section should say the cloud run is not a step"
    assert "adds no" in lowered and "step_list" in lowered, (
        "the cloud-run section should say the cloud run adds no STEP_LIST entry")


def test_the_readme_has_the_run_your_own_copy_section() -> None:
    """Req 15.1: the README's "Run your own copy" section and the image-build estimate."""
    section = markdown_section(README.read_text(encoding="utf-8"), "Run your own copy")
    assert "GPU quota" in section, (
        "the Run your own copy section should list an AWS account with GPU quota")
    for figure in ("$1.20", "45 min"):
        assert figure in section, (
            f"the Run your own copy section should give the image-build estimate {figure!r}")
    assert "estimate" in section.lower(), (
        "the Run your own copy section should note the figures are estimates to check against "
        "the Config prices")


def test_the_tech_steering_admission_is_quota_based() -> None:
    """Req 15.3: the Tech_Steering's one-at-a-time line is now the quota-based admission."""
    text = TECH_STEERING.read_text(encoding="utf-8")
    assert "Admission is by that real G quota" in text, (
        "tech.md should say admission is by the real G quota, read from Service Quotas")
    assert "more than one of our machines may run at once when the quota allows" in text, (
        "tech.md should say more than one of our machines may run when the quota allows")
    assert "one machine per job, not one machine in the whole account" in text, (
        "tech.md should keep one machine per job, not one machine in the whole account")


def test_the_aws_jobs_skill_says_one_yes_covers_the_fallback_list() -> None:
    """Req 15.4: a fallback type is covered by the one yes, not a new run with a new yes."""
    text = AWS_JOBS_SKILL.read_text(encoding="utf-8")
    assert "One yes covers the whole fallback list" in text, (
        "the aws-jobs skill should say one yes covers the whole fallback list")
    assert "part of the same approved run" in text, (
        "the aws-jobs skill should say moving to a fallback type is part of the same approved run")


def test_the_aws_jobs_skill_admission_is_quota_based() -> None:
    """Req 15.5: the skill's one-job-machine note is now the quota-based admission."""
    text = AWS_JOBS_SKILL.read_text(encoding="utf-8")
    assert "Admission is by the real GPU quota" in text, (
        "the aws-jobs skill should say admission is by the real GPU quota")
    assert "could run side by side" in text, (
        "the aws-jobs skill should say two 16-vCPU fallback machines could run side by side")


def test_the_aws_jobs_skill_notes_the_console_output_redaction() -> None:
    """Req 15.6: the skill's console-output note says Cloud_Jobs applies the Account_Redaction."""
    text = AWS_JOBS_SKILL.read_text(encoding="utf-8")
    assert "Cloud_Jobs applies the account redaction" in text, (
        "the aws-jobs skill should say Cloud_Jobs applies the account redaction")
    assert "read the redacted console output to debug a machine" in text, (
        "the aws-jobs skill should say Kiro may read the redacted console output to debug a machine")
