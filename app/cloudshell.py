"""The CloudShell guard every CloudShell_Command runs first, and their CLI wrapper.

CloudShell's own `.bashrc` sets `AWS_EXECUTION_ENV=CloudShell`. The check guards against running a
command on a laptop by mistake; it isn't a security boundary, because anyone can set the variable.
The guard hook blocks Kiro from running these modules on a laptop as a second layer.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING

from cloud.aws import aws_setting
from cloud.errors import CloudError, run_cloud_cli

if TYPE_CHECKING:
    from splatpipe.config import Config

    from cloud.aws import Aws

EXECUTION_ENV = "AWS_EXECUTION_ENV"
CLOUDSHELL = "CloudShell"
# The bundle's or the clone's config.toml, two folders above this file.
CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.toml"


def require_cloudshell(environ: Mapping[str, str]) -> None:
    if environ.get(EXECUTION_ENV) != CLOUDSHELL:
        raise CloudError("This command runs only in AWS CloudShell, signed in as the account owner. "
                         "Open CloudShell in us-east-1 and follow docs/aws-setup.md.")


def run_cloudshell_cli(action: Callable[[], int]) -> int:
    """`cloud.errors.run_cloud_cli`: every message printed Account_Redacted, with no traceback."""
    return run_cloud_cli(action)


def stack_aws(config: Config, make_aws: Callable[[str], Aws], *, table: bool,
              bucket: bool) -> tuple[Aws, dict[str, str]]:
    """The CloudShell client for `[aws].region`, with the website table and the bucket set from the
    Stack's outputs when asked. Gives the client and the outputs."""
    aws = make_aws(aws_setting(config, "region"))
    stack_name = aws_setting(config, "stack_name")
    outputs = aws.stack_outputs(stack_name)
    for wanted, output, attr in ((table, "WebsiteTableName", "table"), (bucket, "BucketName", "bucket")):
        if not wanted:
            continue
        value = outputs.get(output, "")
        if not value:
            raise CloudError(f"The stack {stack_name!r} has no {output} output, so the website isn't "
                             "deployed yet. Deploy it as docs/aws-setup.md says, then run this again.")
        setattr(aws, attr, value)
    return aws, outputs
