"""Tests for the User_Data, cloud/userdata.py (cloud-jobs task 4.5).

`render_user_data` turns a checked `UserDataValues` into the bootstrap bash script cloud-init runs
once as root on first boot. Property 5 (the Hypothesis test) checks, for any valid `UserDataValues`,
that the script is safe and complete: it begins `#!/bin/bash`, uses only `\\n` line endings, stays
within the 16384-byte EC2 User_Data limit, sets the `poweroff` trap before the Time_Cap `shutdown`
line and the cap line before any download or run, carries the Time_Cap from `cap_minutes`, and
interpolates every value `shlex.quote`d so splitting the line with `shlex` recovers the original
value (Req 8.7, 8.8, 10.7).

The first three example tests pin the shapes the task names: a Job_Id-shaped value goes in
`shlex.quote`d and round-trips, a value that fails its pattern is refused with a `CloudError`, and
the Cap_Deadline variable is the documented `CAP_DEADLINE_ENV` constant. The rest pin the machine
side: a Job's machine runs `cloud.runner` and the smoke machine `cloud.smoke_runner`, each of whose
command lines takes the `--job` the script passes, and the instance type the Spend_Lines price comes
from the instance metadata with an IMDSv2 token. These run on the CPU with no network and no AWS:
rendering is pure string work.
"""

import importlib
import shlex
from collections.abc import Callable

import pytest
from hypothesis import given
from hypothesis import strategies as st

from cloud.errors import CloudError
from cloud.userdata import (CAP_DEADLINE_ENV, RUNNER_MODULE, VENV_PYTHON, UserDataValues,
                            render_user_data)
from splatpipe.config import PRESETS
from splatpipe.jobs import JOB_ID


def line_index(lines: list[str], predicate: Callable[[str], bool]) -> int:
    """The index of the first line matching `predicate`. A miss fails the test, showing the script."""
    for index, line in enumerate(lines):
        if predicate(line):
            return index
    raise AssertionError("No line matched the predicate in:\n" + "\n".join(lines))


# Strategies for the fields of a valid UserDataValues. Each matches the checker the field goes
# through in render_user_data: the Job_Id pattern, check_preset's Presets, the safe patterns for the
# bucket, region, image id, code-archive key and lock hashes, and whole-number minutes. The region
# and key lengths are bounded so a run never approaches the 16 KB limit (real values are short).
JOB_IDS = st.from_regex(JOB_ID.pattern, fullmatch=True)
BUCKETS = st.from_regex(r"[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]", fullmatch=True)
REGIONS = st.from_regex(r"[a-z]{2}-[a-z]{1,20}-[0-9]{1,4}", fullmatch=True)
AMIS = st.from_regex(r"ami-[0-9a-f]{8,17}", fullmatch=True)
KEYS = st.from_regex(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,40}", fullmatch=True)
HASHES = st.from_regex(r"[0-9a-f]{64}", fullmatch=True)
MINUTES = st.integers(min_value=0, max_value=100_000)

valid_values = st.builds(
    UserDataValues,
    job_id=JOB_IDS,
    bucket=BUCKETS,
    region=REGIONS,
    preset=st.sampled_from(PRESETS),
    image_id=AMIS,
    code_archive_key=KEYS,
    cap_minutes=MINUTES,
    upload_margin_minutes=MINUTES,
    uv_lock_hash=HASHES,
    package_lock_hash=HASHES,
    role=st.sampled_from(["job", "smoke"]),
)


# Property 5: Every User_Data script is safe and complete (Req 8.7, 8.8, 10.7)

@given(valid_values)
def test_property_5_every_user_data_script_is_safe_and_complete(values: UserDataValues) -> None:
    """Feature: cloud-jobs, Property 5: Every User_Data script is safe and complete

    **Validates: Requirements 8.7, 8.8, 10.7**
    """
    script = render_user_data(values)
    lines = script.split("\n")

    # Safe and within the EC2 User_Data limits: a bash shebang, LF-only endings, at most 16 KB.
    assert lines[0] == "#!/bin/bash"
    assert "\r" not in script
    assert len(script.encode("utf-8")) <= 16384

    # The poweroff trap is set before the Time_Cap shutdown line, and the cap line comes before any
    # download or run, so a failure on the cap line or a hang below still powers the machine off.
    trap_index = line_index(lines, lambda s: s == "trap 'poweroff' EXIT")
    cap_index = line_index(lines, lambda s: s.startswith("shutdown -h "))
    download_index = line_index(lines, lambda s: s.startswith("aws s3 cp "))
    run_index = line_index(lines, lambda s: s.endswith('--job "$JOB_ID"'))
    assert trap_index < cap_index < download_index
    assert cap_index < run_index

    # The Time_Cap line carries cap_minutes.
    assert shlex.split(lines[cap_index]) == ["shutdown", "-h", f"+{values.cap_minutes}"]

    # Every interpolated value goes in shlex.quoted, and splitting its line with shlex recovers the
    # original value exactly.
    interpolated = {
        "JOB_ID": values.job_id,
        "BUCKET": values.bucket,
        "AWS_DEFAULT_REGION": values.region,
        "PRESET": values.preset,
        "IMAGE_ID": values.image_id,
        "CODE_ARCHIVE_KEY": values.code_archive_key,
        "UV_LOCK_HASH": values.uv_lock_hash,
        "PACKAGE_LOCK_HASH": values.package_lock_hash,
        "CAP_MINUTES": str(values.cap_minutes),
        "UPLOAD_MARGIN_MINUTES": str(values.upload_margin_minutes),
    }
    for name, value in interpolated.items():
        line = f"export {name}={shlex.quote(value)}"
        assert line in lines
        assert shlex.split(line) == ["export", f"{name}={value}"]


# Example tests

def example_values(**overrides: object) -> UserDataValues:
    """A valid UserDataValues, with any field replaced for an example."""
    fields: dict[str, object] = dict(
        job_id="20261119-142530-7f3a9c",
        bucket="voxelis-kiroween-jobs",
        region="us-east-1",
        preset="fast",
        image_id="ami-0123456789abcdef0",
        code_archive_key="jobs/20261119-142530-7f3a9c/input/code.tar",
        cap_minutes=120,
        upload_margin_minutes=5,
        uv_lock_hash="a" * 64,
        package_lock_hash="b" * 64,
    )
    fields.update(overrides)
    return UserDataValues(**fields)  # type: ignore[arg-type]


def test_a_tricky_job_id_shaped_value_is_shlex_quoted() -> None:
    # The Job_Id pattern forbids shell metacharacters, so the trickiest valid id carries hyphens and
    # hex only; it still goes in shlex.quoted, and splitting its line with shlex recovers it exactly.
    job_id = "20261119-142530-7f3a9c"
    lines = render_user_data(example_values(job_id=job_id)).split("\n")
    line = f"export JOB_ID={shlex.quote(job_id)}"
    assert line in lines
    assert shlex.split(line) == ["export", f"JOB_ID={job_id}"]


def test_a_value_failing_its_pattern_raises() -> None:
    # A bucket with a space and uppercase fails the safe bucket pattern, so rendering is refused.
    with pytest.raises(CloudError):
        render_user_data(example_values(bucket="Not a bucket!"))


def test_the_cap_deadline_variable_is_the_documented_constant() -> None:
    assert CAP_DEADLINE_ENV == "SPLATPIPE_CAP_DEADLINE"
    script = render_user_data(example_values())
    assert f"export {CAP_DEADLINE_ENV}=" in script


# The machine's entry points

@pytest.mark.parametrize(("role", "module"), [("job", "cloud.runner"),
                                              ("smoke", "cloud.smoke_runner")])
def test_each_role_runs_its_own_machine_entry_point(role: str, module: str) -> None:
    # A Job's machine runs the Cloud_Runner, and the smoke machine runs the smoke test's own checks,
    # never cloud.smoke, the laptop command that launches it.
    lines = render_user_data(example_values(role=role)).split("\n")
    assert lines[-2] == f'{VENV_PYTHON} -m {module} --job "$JOB_ID"'
    assert lines[-1] == ""  # the script ends with one line break


@pytest.mark.parametrize("module_name", sorted(set(RUNNER_MODULE.values())))
def test_every_machine_entry_point_takes_the_job_id(module_name: str) -> None:
    # The run line passes --job "$JOB_ID", so each entry point's own command line must take it, as
    # the laptop's cloud.smoke, which takes no arguments, never could.
    module = importlib.import_module(module_name)
    assert callable(module.main)
    assert module._parser().parse_args(["--job", "20261119-142530-7f3a9c"]).job == (
        "20261119-142530-7f3a9c")
    with pytest.raises(SystemExit):
        module._parser().parse_args([])  # --job is required


def test_the_instance_type_comes_from_imdsv2_before_the_run() -> None:
    # The fallback picks the instance type at launch, so the script asks the instance metadata for
    # it, with an IMDSv2 session token, which the launch makes required. Each read is a plain
    # assignment before its export, so set -e stops the script when curl fails, which an
    # `export X="$(...)"` would hide.
    lines = render_user_data(example_values()).split("\n")
    token = line_index(lines, lambda s: s.startswith('IMDS_TOKEN="$(curl -fsS -X PUT '))
    read = line_index(lines, lambda s: s.startswith('INSTANCE_TYPE="$(curl -fsS -H '))
    export = lines.index("export INSTANCE_TYPE")
    cap = line_index(lines, lambda s: s.startswith("shutdown -h "))
    run = line_index(lines, lambda s: s.endswith('--job "$JOB_ID"'))
    assert cap < token < read < export < run
    assert "/latest/api/token" in lines[token]
    assert "X-aws-ec2-metadata-token-ttl-seconds:" in lines[token]
    assert '"X-aws-ec2-metadata-token: $IMDS_TOKEN"' in lines[read]
    assert lines[read].endswith('/latest/meta-data/instance-type)"')
    assert not any(line.startswith("export INSTANCE_TYPE=") for line in lines)
