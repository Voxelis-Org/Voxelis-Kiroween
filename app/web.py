"""The website's request handler and its data: `Request`, `Response`, `Random` and `Deps`.

`handle(event, deps, now)` is a pure function of a Lambda payload-2.0 event, the `Deps` bundle and
the time, so every test drives it with plain dicts and the Fake_AWS. `app.lambda_function` builds
the real `Deps` once per container.
"""

from __future__ import annotations

import logging
import secrets
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from splatpipe.config import Config

    from app.baked import BakedCommit
    from app.config import WebsiteConfig
    from cloud.aws import Aws
    from cloud.config import CloudConfig
    from cloud.stack import StackOutputs


@dataclass(frozen=True)
class Request:
    method: str
    path: str
    query: Mapping[str, str]
    headers: Mapping[str, str]           # names lowercased
    cookies: Mapping[str, str]
    body: bytes
    source_ip: str                       # requestContext.http.sourceIp, the Sign_In_Address
    host: str                            # requestContext.domainName


@dataclass
class Response:
    status: int
    content_type: str
    body: bytes
    headers: dict[str, str] = field(default_factory=dict)
    cookies: list[str] = field(default_factory=list)


class Random(Protocol):
    def token_bytes(self, n: int) -> bytes: ...
    def choice(self, alphabet: str) -> str: ...


class SecretsRandom:
    """The real `Random`, backed by `secrets`."""

    def token_bytes(self, n: int) -> bytes:
        return secrets.token_bytes(n)

    def choice(self, alphabet: str) -> str:
        return secrets.choice(alphabet)


@dataclass
class Deps:
    aws: Aws
    config: Config
    cloud: CloudConfig
    website: WebsiteConfig
    baked: BakedCommit | None
    static: Mapping[str, bytes]
    random: Random
    clock: Callable[[], datetime]        # aware UTC now; read at the Reservation and before launch
    log: logging.Logger
    stack: StackOutputs | None = None    # resolved by need, cached per container
