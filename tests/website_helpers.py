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


# A whole website over the fake: the Stack, an image that passed the smoke test, capacity, and a
# Baked_Commit whose lock hashes match the image's tags.

STACK = {"BucketName": "voxelis-kiroween-jobs-fake", "JobRoleArn": "arn:aws:iam::ACCT:role/job",
         "JobInstanceProfileArn": "arn:aws:iam::ACCT:instance-profile/job",
         "RunnerRoleArn": "arn:aws:iam::ACCT:role/runner", "SecurityGroupId": "sg-fake",
         "VpcId": "vpc-fake", "GroupName": "voxelis-kiroween", "WebsiteTableName": "table-fake"}
UV_HASH, PACKAGE_HASH = "b" * 64, "c" * 64
GPU_TYPES = ("g5.xlarge", "g5.2xlarge", "g5.4xlarge", "g5.8xlarge", "g6.2xlarge", "g6.4xlarge",
             "g6e.2xlarge", "g6e.4xlarge")


def ready_fake(**kwargs: Any) -> Any:
    from cloud.aws import ImageInfo
    from tests.fake_aws import FakeAws

    image = ImageInfo(id="ami-0123456789abcdef0", name="voxelis-1", state="available",
                      creation_date="2026-11-30T00:00:00Z",
                      tags={"uv_lock_hash": UV_HASH, "package_lock_hash": PACKAGE_HASH})
    values: dict[str, Any] = {"stack": dict(STACK), "images": [image],
                              "objects": {"jobs/_smoke/ami-0123456789abcdef0.json": b"{}"},
                              "offerings": {name: ["us-east-1a"] for name in GPU_TYPES},
                              "subnets": {"us-east-1a": "subnet-a"}, "quota": 32.0}
    values.update(kwargs)
    fake = FakeAws(**values)
    fake.bucket = None
    return fake


def baked_commit(tmp_path: Path) -> Any:
    from app.baked import BakedCommit

    archive = tmp_path / "baked-code.tar"
    archive.write_bytes(b"fake code archive")
    return BakedCommit(commit="a" * 40, uv_lock_hash=UV_HASH, package_lock_hash=PACKAGE_HASH,
                       archive=archive)


STATIC = {"/static/style.css": b"body{}", "/favicon.ico": b"\x00\x00\x01\x00",
          "/static/app.js": b"'use strict';", "/viewer/viewer.css": b".splat-viewer{}",
          "/viewer/embed.js": b"import './viewer.js';", "/viewer/viewer.js": b"export {};",
          "/viewer/tour.js": b"export {};", "/viewer/loading.js": b"export {};",
          "/viewer/haunt.js": b"export {};",
          "/viewer/vendor/playcanvas.min.mjs": b"export const engine = 1;" * 100}


def website(tmp_path: Path, **overrides: Any) -> Any:
    """`Deps` for a whole website over `ready_fake()`, with the fake static files."""
    fake = overrides.pop("fake", None) or ready_fake()
    overrides.setdefault("baked", baked_commit(tmp_path))
    overrides.setdefault("static", dict(STATIC))
    return make_deps(fake, **overrides)


JOB_ID = "20261201-120000-abcdef"


def seed_job(deps: Any, job_id: str = JOB_ID, *, account: str | None = "judges",
             status: str | None = "done", files: tuple[str, ...] = ("clean.sog", "cameras.json"),
             state: str = "held", started_at: datetime | None = None,
             reports: tuple[str, ...] = ("frames", "cameras", "depth", "train", "clean")) -> None:
    """A Job's job item (unless account is None) and its out/ files, with a Job_Summary when
    `status` is given."""
    prefix = f"jobs/{job_id}/out/"
    if account is not None:
        key = (f"account#{account}", f"job#{job_id}")
        deps.aws.items[key] = {"pk": key[0], "sk": key[1], "state": state, "preset": "fast",
                               "cents": 515, "day": "2026-12-01",
                               "started_at": int((started_at or T0).timestamp()), "steps": 5,
                               "upload": "0" * 32, "launch": "launched"}
    for name in files:
        deps.aws.objects[prefix + name] = b"splat bytes " + name.encode()
    for step in reports:
        deps.aws.objects[f"{prefix}reports/{step}.json"] = json.dumps(
            {"step": step, "outcome": "done", "duration_seconds": 12.5}).encode()
    if status is not None:
        steps = [{"step": s, "required": s not in ("depth", "clean"), "reused": False,
                  "outcome": "failed" if (status == "done_with_warnings" and s == "clean")
                  else "done", "readable_error": "The clean step ran out of time."
                  if (status == "done_with_warnings" and s == "clean") else None}
                 for s in reports]
        summary = {"job_id": job_id, "preset": "fast", "status": status,
                   "started_utc": "2026-12-01T12:00:00.000+00:00", "duration_seconds": 1830.0,
                   "steps": steps, "failed_step": "train" if status == "failed" else None,
                   "readable_error": "Training ran out of GPU memory." if status == "failed"
                   else None}
        deps.aws.objects[prefix + "job.json"] = json.dumps(summary).encode()


# Small MP4 and MOV files, built Box by Box.

def box(kind: bytes, payload: bytes = b"", *, large: bool = False) -> bytes:
    if large:
        return (1).to_bytes(4, "big") + kind + (16 + len(payload)).to_bytes(8, "big") + payload
    return (8 + len(payload)).to_bytes(4, "big") + kind + payload


def time_header(kind: bytes, timescale: int, duration: int, version: int = 0) -> bytes:
    if version == 1:
        payload = bytes([1, 0, 0, 0]) + bytes(16) + timescale.to_bytes(4, "big") + \
            duration.to_bytes(8, "big") + bytes(4)
    else:
        payload = bytes(4) + bytes(8) + timescale.to_bytes(4, "big") + duration.to_bytes(4, "big") + \
            bytes(4)
    return box(kind, payload)


def track(handler: bytes, timescale: int, duration: int, version: int = 0) -> bytes:
    hdlr = box(b"hdlr", bytes(4) + bytes(4) + handler + bytes(12) + b"\x00")
    return box(b"trak", box(b"mdia", time_header(b"mdhd", timescale, duration, version) + hdlr))


def mp4(seconds: float = 10.0, *, video: bool = True, timescale: int = 600,
        track_duration: int | None = None, mdat_bytes: int = 64, moov_first: bool = False,
        version: int = 0) -> bytes:
    duration = int(seconds * timescale)
    tracks = track(b"soun", 44100, int(seconds * 44100))
    if video:
        tracks += track(b"vide", timescale, duration if track_duration is None else track_duration,
                        version)
    moov = box(b"moov", time_header(b"mvhd", 1000, int(seconds * 1000), version) + tracks)
    ftyp = box(b"ftyp", b"isom" + bytes(4) + b"isomiso2mp41")
    mdat = box(b"mdat", bytes(mdat_bytes))
    return ftyp + (moov + mdat if moov_first else mdat + moov)


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
