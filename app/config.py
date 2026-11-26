"""The `[website]` reader: `WebsiteConfig`, `ScryptSettings` and `dollars_to_cents`.

`load_website_config` validates the whole file with splatpipe's Config_Loader first, so NaN,
infinity and TOML dates in `[website]` are refused there, then reads `[website]` with
`WebsiteConfig.from_mapping`. Every refusal is a `PipelineError` that names the key and says what it
must be. A key the reader doesn't know is refused too, so a typo can't silently fall back.

The floors that come from AWS stay in code, each with its source: S3 parts of at least 5 MiB, at
most 10,000 parts in one upload, and presigned links of at most 7 days
(https://docs.aws.amazon.com/AmazonS3/latest/userguide/qfacts.html).
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from splatpipe.config import Config, load_config
from splatpipe.errors import PipelineError

SECTION = "website"
# S3's smallest part, except the last one, and the most parts one multipart upload may have.
S3_MIN_PART_BYTES = 5 * 1024 * 1024
S3_MAX_PARTS = 10_000
# The longest a SigV4 presigned link may last: 7 days.
S3_MAX_LINK_SECONDS = 7 * 24 * 60 * 60
# The Stop_Time's one accepted form.
STOP_TIME = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z")
_STOP_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
_MODULE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*")
_CENTS = 100

# Each whole-number key and its floor. The upper bounds that AWS sets are checked on their own.
_WHOLE_FLOORS: dict[str, int] = {
    "daily_jobs": 1, "boot_margin_minutes": 0, "upload_part_bytes": S3_MIN_PART_BYTES,
    "upload_link_seconds": 1, "part_link_retries": 0, "download_link_seconds": 1,
    "copy_part_bytes": S3_MIN_PART_BYTES, "upload_record_hours": 1, "max_boxes": 1,
    "max_moov_bytes": 1, "ended_check_minutes": 0, "signin_max_failures": 1,
    "signin_window_minutes": 1, "signin_wait_minutes": 1, "signin_token_minutes": 1,
    "session_days": 1, "token_bytes": 16, "password_length": 16, "scrypt_n": 2, "scrypt_r": 1,
    "scrypt_p": 1, "scrypt_dklen": 16, "scrypt_salt_bytes": 8, "max_body_bytes": 1024,
    "max_listed_jobs": 1, "max_title_chars": 1, "max_credit_chars": 1, "max_name_chars": 1,
    "static_cache_seconds": 0, "max_bundle_bytes": 1,
}
_DOLLAR_KEYS = ("daily_cost_usd", "total_cost_usd")
_TEXT_KEYS = ("stop_starting_at", "deploy_folder")
_LIST_KEYS = ("cloudshell_modules",)
KEYS = frozenset(_WHOLE_FLOORS) | frozenset(_DOLLAR_KEYS) | frozenset(_TEXT_KEYS) | frozenset(_LIST_KEYS)


@dataclass(frozen=True)
class ScryptSettings:
    """scrypt's settings for one password hash."""

    n: int
    r: int
    p: int
    dklen: int
    salt_bytes: int


@dataclass(frozen=True)
class WebsiteConfig:
    """The checked `[website]` settings. Each field is named after its key, except the two cents."""

    stop_starting_at: datetime          # aware, UTC
    daily_jobs: int
    daily_cost_cents: int
    total_cost_cents: int
    boot_margin_minutes: int
    upload_part_bytes: int
    upload_link_seconds: int
    part_link_retries: int
    download_link_seconds: int
    copy_part_bytes: int
    upload_record_hours: int
    max_boxes: int
    max_moov_bytes: int
    ended_check_minutes: int
    signin_max_failures: int
    signin_window_minutes: int
    signin_wait_minutes: int
    signin_token_minutes: int
    session_days: int
    token_bytes: int
    password_length: int
    scrypt: ScryptSettings
    max_body_bytes: int
    max_listed_jobs: int
    max_title_chars: int
    max_credit_chars: int
    max_name_chars: int
    static_cache_seconds: int
    cloudshell_modules: tuple[str, ...]
    deploy_folder: PurePosixPath
    max_bundle_bytes: int

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any], *,
                     max_video_bytes: int | None = None) -> WebsiteConfig:
        """Check `[website]` as a mapping and build the settings, refusing a bad key by name.

        `max_video_bytes` is `[jobs].max_video_bytes`, for the 10,000-part check of the upload's
        part size; None skips that check.
        """
        unknown = sorted(set(data) - KEYS)
        if unknown:
            raise PipelineError(f"[website] in config.toml has a key it doesn't know: {unknown[0]}. "
                                "Check its spelling, or remove it.")
        whole = {key: _whole(data, key, floor) for key, floor in _WHOLE_FLOORS.items()}
        _check_links(whole)
        _check_parts(whole["upload_part_bytes"], max_video_bytes)
        if whole["scrypt_n"] & (whole["scrypt_n"] - 1):
            raise PipelineError("[website] scrypt_n in config.toml must be a power of 2, like 32768.")
        scrypt = ScryptSettings(n=whole.pop("scrypt_n"), r=whole.pop("scrypt_r"),
                                p=whole.pop("scrypt_p"), dklen=whole.pop("scrypt_dklen"),
                                salt_bytes=whole.pop("scrypt_salt_bytes"))
        return cls(stop_starting_at=_stop_time(data), daily_cost_cents=_dollars(data, "daily_cost_usd"),
                   total_cost_cents=_dollars(data, "total_cost_usd"), scrypt=scrypt,
                   cloudshell_modules=_modules(data), deploy_folder=_deploy_folder(data), **whole)


def load_website_config(path: Path | None = None) -> WebsiteConfig:
    """Validate the whole file with splatpipe's Config_Loader, then read `[website]`."""
    config = load_config(path)
    return website_config_from(config)


def website_config_from(config: Config) -> WebsiteConfig:
    """`[website]` from a loaded Config, with `[jobs].max_video_bytes` for the part-count check."""
    data = getattr(config, "_data", {}).get(SECTION)
    if not isinstance(data, Mapping):
        raise PipelineError("config.toml has no [website] section. Get config.toml back from git.")
    return WebsiteConfig.from_mapping(data, max_video_bytes=config.jobs().max_video_bytes)


def dollars_to_cents(value: float) -> int:
    """Whole cents from the .2f text of `value`, as the Start_Command's quote prints it."""
    dollars, _, cents = f"{value:.2f}".partition(".")
    return int(dollars) * _CENTS + int(cents) * (-1 if dollars.startswith("-") else 1)


# Reading each kind of key

def _present(data: Mapping[str, Any], key: str) -> Any:
    if key not in data:
        raise PipelineError(f"[website] in config.toml needs {key}. Get config.toml back from git, "
                            "or add the key with its comment.")
    return data[key]


def _whole(data: Mapping[str, Any], key: str, floor: int) -> int:
    value = _present(data, key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise PipelineError(f"[website] {key} in config.toml must be a whole number, "
                            f"{floor} or more, but it is {value!r}.")
    if value < floor:
        raise PipelineError(f"[website] {key} in config.toml must be {floor} or more, "
                            f"but it is {value}.")
    return value


def _dollars(data: Mapping[str, Any], key: str) -> int:
    value = _present(data, key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise PipelineError(f"[website] {key} in config.toml must be a number of US dollars, "
                            f"but it is {value!r}.")
    cents = dollars_to_cents(float(value))
    if cents < 1:
        raise PipelineError(f"[website] {key} in config.toml must be at least $0.01, "
                            f"but it is {value!r}.")
    return cents


def _stop_time(data: Mapping[str, Any]) -> datetime:
    value = _present(data, "stop_starting_at")
    if not isinstance(value, str) or not STOP_TIME.fullmatch(value):
        raise PipelineError("[website] stop_starting_at in config.toml must be UTC text written "
                            f'as YYYY-MM-DDTHH:MM:SSZ, like "2027-02-04T08:00:00Z", but it is {value!r}.')
    try:
        return datetime.strptime(value, _STOP_FORMAT).replace(tzinfo=timezone.utc)
    except ValueError as error:
        raise PipelineError(f"[website] stop_starting_at in config.toml isn't a real time: {value!r}. "
                            "Write a real UTC time, like \"2027-02-04T08:00:00Z\".") from error


def _modules(data: Mapping[str, Any]) -> tuple[str, ...]:
    value = _present(data, "cloudshell_modules")
    if (not isinstance(value, list) or not value
            or not all(isinstance(name, str) and _MODULE.fullmatch(name) for name in value)):
        raise PipelineError("[website] cloudshell_modules in config.toml must be a non-empty list "
                            'of dotted module names, like ["app.deploy"].')
    return tuple(value)


def _deploy_folder(data: Mapping[str, Any]) -> PurePosixPath:
    value = _present(data, "deploy_folder")
    folder = PurePosixPath(value) if isinstance(value, str) else None
    if (folder is None or not folder.is_absolute() or len(folder.parts) < 3
            or folder.parts[1] != "tmp" or ".." in folder.parts):
        raise PipelineError("[website] deploy_folder in config.toml must be a folder inside /tmp, "
                            f'like "/tmp/voxelis-website", but it is {value!r}.')
    return folder


def _check_links(whole: Mapping[str, int]) -> None:
    for key in ("upload_link_seconds", "download_link_seconds"):
        if whole[key] > S3_MAX_LINK_SECONDS:
            raise PipelineError(f"[website] {key} in config.toml must be at most "
                                f"{S3_MAX_LINK_SECONDS} seconds (7 days), S3's longest link.")


def _check_parts(part_bytes: int, max_video_bytes: int | None) -> None:
    if max_video_bytes is not None and math.ceil(max_video_bytes / part_bytes) > S3_MAX_PARTS:
        raise PipelineError(f"[website] upload_part_bytes in config.toml is too small: the largest "
                            f"video would need more than {S3_MAX_PARTS} parts, S3's limit. Make it "
                            "larger.")
