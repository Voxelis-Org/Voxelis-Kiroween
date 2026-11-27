"""The sign-in limit: Sign_In_Failure items and the lock, per Sign_In_Address.

Each failure is its own item, `signin#<address>` / `failure#<10-digit epoch seconds>#<8 hex>`, so
two failures at the same moment can't overwrite each other's count. The address's lock is
`signin#<address>` / `lock`, holding `until`. One query from the window's start to `lock` returns
the window's failures and the lock together. Nothing is counted per Account.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

from cloud.table import Between, Scalar

if TYPE_CHECKING:
    from collections.abc import Sequence

    from app.config import WebsiteConfig
    from app.web import Random
    from cloud.aws import Aws

_SUFFIX_BYTES = 4
_EPOCH_DIGITS = 10
LOCK = "lock"


def partition(address: str) -> str:
    return f"signin#{address}"


def locked_until(aws: Aws, address: str, now: datetime, wc: WebsiteConfig) -> datetime | None:
    """When the address's running lock ends, or None when there is none."""
    return _lock_of(_window(aws, address, now, wc), now)


def record_failure(aws: Aws, address: str, now: datetime, wc: WebsiteConfig,
                   random: Random) -> datetime | None:
    """Record one failure; start a lock when the window now holds enough. Gives the lock's end."""
    seconds = int(now.timestamp())
    window = timedelta(minutes=wc.signin_window_minutes)
    suffix = random.token_bytes(_SUFFIX_BYTES).hex()
    aws.table_put({"pk": partition(address), "sk": f"failure#{seconds:0{_EPOCH_DIGITS}d}#{suffix}",
                   "expires_at": int((now + window).timestamp())})
    failures = [item for item in _window(aws, address, now, wc) if item["sk"] != LOCK]
    if len(failures) < wc.signin_max_failures:
        return None
    until = now + timedelta(minutes=wc.signin_wait_minutes)
    aws.table_put({"pk": partition(address), "sk": LOCK, "until": int(until.timestamp()),
                   "expires_at": int(until.timestamp())})
    return until


def _window(aws: Aws, address: str, now: datetime,
            wc: WebsiteConfig) -> Sequence[dict[str, Scalar]]:
    start = int(now.timestamp()) - wc.signin_window_minutes * 60 + 1
    return aws.table_query(partition(address),
                           Between(f"failure#{max(start, 0):0{_EPOCH_DIGITS}d}", LOCK))


def _lock_of(items: Sequence[dict[str, Scalar]], now: datetime) -> datetime | None:
    for item in items:
        if item["sk"] == LOCK:
            until = datetime.fromtimestamp(int(item["until"]), tz=timezone.utc)
            return until if until > now else None
    return None
