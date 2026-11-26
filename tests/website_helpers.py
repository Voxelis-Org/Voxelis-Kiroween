"""Shared website test helpers: `website_config`, `make_deps`, `event`, `sign_in_as` and `TestClock`.

`website_config(**overrides)` builds a `WebsiteConfig` from the real `[website]` in the repo's
`config.toml`, with scrypt made cheap (N=2, r=1, p=1) unless a test overrides it, so a hash takes
microseconds. `make_deps(fake, **overrides)` loads the repo's real `config.toml` by path, because the
autouse `temp_config` fixture's Config has no `[aws]`, `[cloud]` or `[website]`, and gives `Deps` a
settable `TestClock` and a seeded `Random`.
"""

from __future__ import annotations

import base64
import json
import logging
import random as stdlib_random
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from splatpipe.config import load_config

from app.config import WebsiteConfig

REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_CONFIG = REPO_ROOT / "config.toml"
FAST_SCRYPT = {"scrypt_n": 2, "scrypt_r": 1, "scrypt_p": 1}
# A time well before the Stop_Time, for tests that don't care about the date.
T0 = datetime(2026, 12, 1, 12, 0, tzinfo=timezone.utc)


def website_mapping() -> dict[str, Any]:
    """The real `[website]` table, as tomllib reads it."""
    with open(REAL_CONFIG, "rb") as handle:
        return dict(tomllib.load(handle)["website"])


def website_config(**overrides: Any) -> WebsiteConfig:
    """A `WebsiteConfig` from the real `[website]`, cheap scrypt, then each override by key."""
    data = website_mapping()
    data.update(FAST_SCRYPT)
    data.update(overrides)
    return WebsiteConfig.from_mapping(data, max_video_bytes=4294967296)


class SeededRandom:
    """A `Random` from a seed, so a test's tokens repeat."""

    def __init__(self, seed: int = 7) -> None:
        self._random = stdlib_random.Random(seed)

    def token_bytes(self, n: int) -> bytes:
        return self._random.randbytes(n)

    def choice(self, alphabet: str) -> str:
        return self._random.choice(alphabet)


@dataclass
class TestClock:
    """A settable clock: `clock()` gives `now`, and `advance` moves it."""

    __test__ = False   # not a pytest test class
    now: datetime = T0

    def __call__(self) -> datetime:
        return self.now

    def advance(self, **delta: float) -> None:
        self.now = self.now + timedelta(**delta)


def real_config():
    """The repo's real Config, read by path."""
    return load_config(REAL_CONFIG)


def event(method: str, path: str, *, cookies: Mapping[str, str] | None = None,
          headers: Mapping[str, str] | None = None, body: bytes | str | None = None,
          query: str = "", source_ip: str = "203.0.113.7", base64_body: bool = False,
          host: str = "abc123.lambda-url.us-east-1.on.aws") -> dict[str, Any]:
    """A Lambda function URL payload-2.0 event."""
    raw = body.encode("utf-8") if isinstance(body, str) else (body or b"")
    result: dict[str, Any] = {
        "version": "2.0", "rawPath": path, "rawQueryString": query,
        "headers": {"host": host, **{k.lower(): v for k, v in (headers or {}).items()}},
        "requestContext": {"domainName": host,
                           "http": {"method": method, "path": path, "sourceIp": source_ip}},
        "isBase64Encoded": base64_body,
    }
    if cookies:
        result["cookies"] = [f"{name}={value}" for name, value in cookies.items()]
    if body is not None:
        result["body"] = base64.b64encode(raw).decode("ascii") if base64_body else raw.decode("utf-8")
    return result


def make_deps(fake: Any, **overrides: Any) -> Any:
    """`Deps` over a FakeAws, the repo's real configs, a seeded Random and a TestClock.

    `website=` replaces the whole WebsiteConfig, and `wc_overrides=` changes some of its keys."""
    from app.web import Deps
    from cloud.config import load_cloud_config

    wc_overrides = overrides.pop("wc_overrides", {})
    values: dict[str, Any] = {
        "aws": fake, "config": real_config(), "cloud": load_cloud_config(REAL_CONFIG),
        "website": website_config(**wc_overrides), "baked": None, "static": {},
        "random": SeededRandom(), "clock": TestClock(), "log": quiet_logger(),
    }
    values.update(overrides)
    return Deps(**values)


def make_account_item(deps: Any, name: str, password: str, uid: str = "u" * 32) -> None:
    """Seed an Account's profile item with `password`."""
    import base64 as b64

    from app.accounts import hash_password, profile_key

    salt = b"salt-for-" + name.encode()
    made = hash_password(password, salt, deps.website.scrypt)
    scrypt = deps.website.scrypt
    deps.aws.items[(profile_key(name).pk, "profile")] = {
        "pk": profile_key(name).pk, "sk": "profile", "hash": b64.b64encode(made).decode(),
        "salt": b64.b64encode(salt).decode(), "n": scrypt.n, "r": scrypt.r, "p": scrypt.p,
        "dklen": scrypt.dklen, "uid": uid, "created_at": 0}


def sign_in_as(deps: Any, name: str, now: datetime | None = None) -> tuple[dict[str, str], str]:
    """Seed an Account and a Session for it. Gives the cookies and the CSRF_Token."""
    from app.sessions import SESSION_COOKIE, new_token, session_key

    now = now or deps.clock()
    if (f"account#{name}", "profile") not in deps.aws.items:
        make_account_item(deps, name, "password-" + name)
    uid = deps.aws.items[(f"account#{name}", "profile")]["uid"]
    session_id, csrf = new_token(deps), new_token(deps)
    key = session_key(name, session_id)
    deps.aws.items[(key.pk, key.sk)] = {
        "pk": key.pk, "sk": key.sk, "csrf": csrf, "uid": uid, "created_at": int(now.timestamp()),
        "expires_at": int((now + timedelta(days=deps.website.session_days)).timestamp())}
    return {SESSION_COOKIE: f"{name}.{session_id}"}, csrf


def json_body(response: Mapping[str, Any]) -> Any:
    """The JSON a handler answer holds."""
    text = response.get("body", "")
    if response.get("isBase64Encoded"):
        text = base64.b64decode(text).decode("utf-8")
    return json.loads(text)


def quiet_logger(name: str = "voxelis.website.test") -> logging.Logger:
    log = logging.getLogger(name)
    log.propagate = True
    return log
