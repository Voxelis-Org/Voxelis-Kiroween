"""The Website_Function's entry point: `app.lambda_function.handler`.

The first request of a container builds the `Deps` and keeps it: the Website_Role's client with the
table from `WEBSITE_TABLE`, the three configs from the bundle's `config.toml`, the Baked_Commit, the
static files and the clock. The handler never reads the spend variable from the environment.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from splatpipe.config import load_config

from app.baked import read_baked
from app.config import website_config_from
from app.handler import handle, load_static
from app.web import Deps, SecretsRandom
from cloud.aws import aws_setting, website_aws
from cloud.config import load_cloud_config
from cloud.redact import account_redaction

BUNDLE = Path(__file__).resolve().parents[1]
TABLE_VARIABLE = "WEBSITE_TABLE"
_DEPS: Deps | None = None


class RedactFilter(logging.Filter):
    """Applies account_redaction to every log record, the handler's and boto3's alike."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = account_redaction(record.getMessage())
        record.args = ()
        return True


def now() -> datetime:
    return datetime.now(timezone.utc)


def build_deps(environ: Any = os.environ, root: Path = BUNDLE) -> Deps:
    config = load_config(root / "config.toml")
    aws = website_aws(aws_setting(config, "region"))
    aws.table = environ[TABLE_VARIABLE]
    log = logging.getLogger("voxelis.website")
    log.setLevel(logging.INFO)
    for handler in logging.getLogger().handlers:
        handler.addFilter(RedactFilter())
    log.addFilter(RedactFilter())
    return Deps(aws=aws, config=config, cloud=load_cloud_config(root / "config.toml"),
                website=website_config_from(config), baked=read_baked(root),
                static=load_static(root), random=SecretsRandom(), clock=now, log=log)


def handler(event: dict[str, Any], context: object) -> dict[str, Any]:
    global _DEPS
    if _DEPS is None:
        _DEPS = build_deps()
    return handle(event, _DEPS, now())
