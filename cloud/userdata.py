"""The User_Data: the bootstrap bash script cloud-init runs once as root on first boot.

`render_user_data` turns a checked `UserDataValues` into the launch script that powers the machine
off when it ends, schedules the Time_Cap shutdown as a backstop, carries the Job's settings and the
Cap_Deadline to the Cloud_Runner, fetches the Code_Archive with the instance role and runs the
Cloud_Runner at the committed revision. The script begins `#!/bin/bash`, uses only `\\n` line
endings and is at most 16 KB, the EC2 User_Data limit (Req 8.7, 8.8).

The `poweroff` trap is written before the Time_Cap `shutdown` line, so a failure on the cap line
itself still powers the machine off, and the cap line comes before any download or run, so a hang
anywhere later still ends the machine (the AWS_Jobs_Skill's ordering, Req 8.7, 10.8). Every value is
checked and then `shlex.quote`d before it goes into the script: the Job_Id through
`splatpipe.jobs.check_job_id`, the Preset through `splatpipe.config.check_preset`, the bucket,
region, image id, code-archive key and lock hashes against safe patterns, and the minutes as whole
numbers. The script never holds AWS keys; the machine uses its instance role (Req 1.7).

`render_user_data` serves a Job (`cloud.runner`) and the Smoke_Test (`cloud.smoke`); the Image_Build
passes its own install script as the whole User_Data instead, so there is no `image` case here.

A cloud module may import `splatpipe` to call its public functions; `splatpipe` never imports
`cloud`. This module imports no boto3 and no GPU package, so it loads on a laptop.
"""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass

from splatpipe.config import check_preset
from splatpipe.jobs import check_job_id

from cloud.errors import CloudError

CAP_DEADLINE_ENV = "SPLATPIPE_CAP_DEADLINE"   # the Cap_Deadline variable the Cloud_Runner sets (Req 10.7)

# The machine paths a Job, the Smoke_Test and the Image_Build share: the folder the Code_Archive is
# extracted into, and the Python of the environment the Image_Build sets up there with uv. cloud.image
# installs into this same VENV_PYTHON, so the two can't drift apart.
JOB_ROOT = "/opt/job"
VENV_PYTHON = f"{JOB_ROOT}/.venv/bin/python"

_MAX_BYTES = 16384   # the EC2 User_Data size limit (Req 8.7)

# Safe patterns for the values that have no checker of their own. shlex.quote already makes every
# value safe in the shell; these patterns refuse anything misshapen early, with a readable error
# naming the field, and keep the script to the characters we expect from the Stack and the Config.
_BUCKET = re.compile(r"[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]")   # an S3 bucket name, 3-63 chars
_REGION = re.compile(r"[a-z]{2}-[a-z]+-[0-9]+")             # like us-east-1
_HASH = re.compile(r"[0-9a-f]{64}")                         # a sha-256 hex digest
_AMI = re.compile(r"ami-[0-9a-f]{8,17}")                    # an AMI id
_KEY = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]*")           # an S3 object key, like jobs/<id>/input/code.tar

# role -> the on-machine entry point module. render_user_data builds the bootstrap for a Job and for
# the Smoke_Test only; the Image_Build passes install_script() as its whole User_Data instead.
_RUNNER_MODULE = {"job": "cloud.runner", "smoke": "cloud.smoke"}


@dataclass(frozen=True)
class UserDataValues:
    """The checked inputs the bootstrap script interpolates. Each field is a value the machine needs."""

    job_id: str
    bucket: str
    region: str
    preset: str
    image_id: str
    code_archive_key: str
    cap_minutes: int
    upload_margin_minutes: int
    uv_lock_hash: str
    package_lock_hash: str
    role: str = "job"            # "job", "smoke" or "image"


def _pattern(value: object, pattern: re.Pattern[str], field: str) -> str:
    """Return `value` when it's a str the whole of which matches `pattern`. Refuse anything else.

    The refusal names the field and the expected form and leaves the raw value out, so a line break
    or a control character in it can't garble the terminal, like `check_job_id` does.
    """
    if isinstance(value, str) and pattern.fullmatch(value) is not None:
        return value
    raise CloudError(f"The {field} isn't valid for the machine's bootstrap script: it must match "
                     f"{pattern.pattern!r}. Check the stack output or the config it came from.")


def _whole(value: object, field: str) -> int:
    """Return `value` when it's a whole number of minutes (an int of 0 or more). Refuse anything else."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CloudError(f"{field} must be a whole number of minutes (an int of 0 or more) for the "
                         "machine's bootstrap script.")
    return value


def render_user_data(values: UserDataValues) -> str:
    """The bootstrap bash script, LF-only, at most 16 KB (Req 8.7, 8.8). Property 5 asserts on it.

    Every value is checked and then shlex.quoted. The `poweroff` trap comes before the Time_Cap
    `shutdown` line, and the cap line comes before any download or run (Req 8.7, 10.8). A rendered
    text over 16 KB is a `CloudError`.
    """
    module = _RUNNER_MODULE.get(values.role)
    if module is None:
        raise CloudError(f"render_user_data builds the bootstrap for a job or a smoke machine, but "
                         f"it got role {values.role!r}. The image build passes its own install "
                         "script as the whole user data instead.")
    job_id = check_job_id(values.job_id)
    preset = check_preset(values.preset)
    bucket = _pattern(values.bucket, _BUCKET, "bucket name")
    region = _pattern(values.region, _REGION, "region")
    image_id = _pattern(values.image_id, _AMI, "image id")
    code_key = _pattern(values.code_archive_key, _KEY, "code archive key")
    uv_hash = _pattern(values.uv_lock_hash, _HASH, "uv.lock hash")
    pkg_hash = _pattern(values.package_lock_hash, _HASH, "package-lock.json hash")
    cap_minutes = _whole(values.cap_minutes, "cap_minutes")
    margin_minutes = _whole(values.upload_margin_minutes, "upload_margin_minutes")

    q = shlex.quote
    lines = [
        "#!/bin/bash",
        "set -euo pipefail",
        # Power off when the script ends, done or failed. This trap is set before the Time_Cap
        # shutdown line, so a failure on the cap line itself still powers the machine off.
        "trap 'poweroff' EXIT",
        # The Time_Cap backstop, before any download or run, so a hang anywhere below still ends the
        # machine (Req 8.7, 10.8).
        f"shutdown -h +{q(str(cap_minutes))}",
        # The Job's settings for the Cloud_Runner, each value shlex-quoted (Req 8.8).
        f"export JOB_ID={q(job_id)}",
        f"export BUCKET={q(bucket)}",
        f"export AWS_DEFAULT_REGION={q(region)}",
        f"export PRESET={q(preset)}",
        f"export IMAGE_ID={q(image_id)}",
        f"export CODE_ARCHIVE_KEY={q(code_key)}",
        f"export UV_LOCK_HASH={q(uv_hash)}",
        f"export PACKAGE_LOCK_HASH={q(pkg_hash)}",
        f"export CAP_MINUTES={q(str(cap_minutes))}",
        f"export UPLOAD_MARGIN_MINUTES={q(str(margin_minutes))}",
        # The Cap_Deadline: the machine's start plus the Time_Cap, minus the upload margin, as an
        # absolute UTC instant the Steps can read (Req 10.7).
        f'export {CAP_DEADLINE_ENV}="$(date -u -d "+$((CAP_MINUTES - UPLOAD_MARGIN_MINUTES)) minutes" +%Y-%m-%dT%H:%M:%SZ)"',
        # Fetch the Code_Archive with the instance role, extract it to /opt/job, and run the
        # Cloud_Runner at the committed revision.
        f"mkdir -p {JOB_ROOT}",
        'aws s3 cp "s3://$BUCKET/$CODE_ARCHIVE_KEY" /tmp/code.tar',
        f"tar -xf /tmp/code.tar -C {JOB_ROOT}",
        f"cd {JOB_ROOT}",
        f'{VENV_PYTHON} -m {module} --job "$JOB_ID"',
    ]
    script = "\n".join(lines) + "\n"
    if len(script.encode("utf-8")) > _MAX_BYTES:
        raise CloudError(f"The machine's bootstrap script came out larger than {_MAX_BYTES} bytes, "
                         "which is the EC2 user data limit. Shorten the values, or drop a params "
                         "file into the bucket and fetch it instead.")
    return script
