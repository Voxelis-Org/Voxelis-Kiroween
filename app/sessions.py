"""Sessions: the Session_Cookie, the one-read Session check, signing in and out, and CSRF.

The Session_Cookie holds `<account name>.<id>`, the id `token_bytes` random bytes in unpadded
base64url. The session item, `account#<name>` / `session#<SHA-256 hex of the id>`, never holds the
id itself. One `table_get_many` of the profile and the session decides a Valid_Session: both exist,
the session isn't ended or expired by the handler's clock, and its `uid` equals the profile's.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING
from urllib.parse import parse_qs

from app import signin_limit
from app.accounts import password_matches, profile_key, read_account, valid_name
from app.errors import BAD_REQUEST, FORBIDDEN, TOO_MANY, PublicRefusal
from app.web import Response
from cloud.table import ITEM_ABSENT, ITEM_THERE, Key, Set

if TYPE_CHECKING:
    from app.web import Deps, Request

SESSION_COOKIE = "__Host-voxelis-session"
SIGNIN_COOKIE = "__Host-voxelis-signin"
SEE_OTHER = 303
_TOKEN = re.compile(r"[A-Za-z0-9_-]+")
_DAY_SECONDS = 24 * 60 * 60
NO_MATCH = "That name and password don't match an account."
CSRF_MESSAGE = ("This form has expired or came from another site. Open the page again and send it "
                "from there.")


@dataclass(frozen=True)
class Session:
    account: str
    key: Key
    csrf: str
    uid: str


def token_length(token_bytes: int) -> int:
    """The unpadded base64url length of `token_bytes` bytes: 43 for 32."""
    return (4 * token_bytes + 2) // 3


def new_token(deps: Deps) -> str:
    raw = deps.random.token_bytes(deps.website.token_bytes)
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def well_formed_token(value: str | None, token_bytes: int) -> bool:
    return (isinstance(value, str) and len(value) == token_length(token_bytes)
            and _TOKEN.fullmatch(value) is not None)


def same_token(given: str | None, expected: str | None) -> bool:
    if not given or not expected:
        return False
    return hmac.compare_digest(given.encode("utf-8", "replace"), expected.encode("utf-8", "replace"))


def parse_session_cookie(value: str | None, token_bytes: int) -> tuple[str, str] | None:
    """(account name, id) from a well-formed Session_Cookie, else None, with no AWS call."""
    if not value or "." not in value:
        return None
    name, _, session_id = value.partition(".")
    if not valid_name(name) or not well_formed_token(session_id, token_bytes):
        return None
    return name, session_id


def session_key(name: str, session_id: str) -> Key:
    digest = hashlib.sha256(session_id.encode("ascii")).hexdigest()
    return Key(f"account#{name}", f"session#{digest}")


def current(request: Request, deps: Deps, now: datetime) -> Session | None:
    """The request's Valid_Session, or None. At most one table read, and none for a bad cookie."""
    parsed = parse_session_cookie(request.cookies.get(SESSION_COOKIE), deps.website.token_bytes)
    if parsed is None:
        return None
    name, session_id = parsed
    key = session_key(name, session_id)
    profile, item = deps.aws.table_get_many([profile_key(name), key])
    if profile is None or item is None or item.get("ended") is True:
        return None
    if int(item.get("expires_at", 0)) <= now.timestamp() or item.get("uid") != profile.get("uid"):
        return None
    return Session(account=name, key=key, csrf=str(item.get("csrf", "")), uid=str(item["uid"]))


def form(request: Request) -> dict[str, str]:
    """A form body's fields, the first value of each."""
    try:
        text = request.body.decode("utf-8")
    except UnicodeDecodeError:
        return {}
    return {name: values[0] for name, values in parse_qs(text, keep_blank_values=True).items()}


def sign_in(request: Request, deps: Deps, now: datetime) -> Response:
    """Check the form token, the address's lock and the password, then make a Session."""
    wc = deps.website
    fields = form(request)
    if not same_token(fields.get("csrf"), request.cookies.get(SIGNIN_COOKIE)):
        raise PublicRefusal(FORBIDDEN, "csrf", CSRF_MESSAGE)
    address = request.source_ip
    until = signin_limit.locked_until(deps.aws, address, now, wc)
    if until is not None:
        minutes = max(1, -(-int((until - now).total_seconds()) // 60))
        raise PublicRefusal(TOO_MANY, "signin_wait", f"Too many failed sign-ins from here. Wait "
                                                     f"{minutes} minutes, then try again.")
    name = fields.get("name", "").strip()
    account = None
    if valid_name(name):
        item = deps.aws.table_get(profile_key(name))
        account = read_account(name, item) if item is not None else None
    if not password_matches(fields.get("password", ""), account, wc.scrypt):
        signin_limit.record_failure(deps.aws, address, now, wc, deps.random)
        raise PublicRefusal(BAD_REQUEST, "bad_request", NO_MATCH)
    assert account is not None
    return _start_session(account.name, account.uid, deps, now)


def _start_session(name: str, uid: str, deps: Deps, now: datetime) -> Response:
    wc = deps.website
    session_id, csrf = new_token(deps), new_token(deps)
    lifetime = timedelta(days=wc.session_days)
    key = session_key(name, session_id)
    deps.aws.table_put({"pk": key.pk, "sk": key.sk, "csrf": csrf, "uid": uid,
                        "created_at": int(now.timestamp()),
                        "expires_at": int((now + lifetime).timestamp())}, ITEM_ABSENT)
    response = redirect("/jobs")
    response.cookies.append(set_cookie(SESSION_COOKIE, f"{name}.{session_id}",
                                       wc.session_days * _DAY_SECONDS, "Lax"))
    response.cookies.append(clear_cookie(SIGNIN_COOKIE))
    return response


def sign_out(session: Session, deps: Deps, now: datetime) -> Response:
    """End the Session, clear its cookie and go to the sign-in page. Nothing is deleted."""
    deps.aws.table_update(session.key, (Set("ended", True),), ITEM_THERE)
    response = redirect("/signin")
    response.cookies.append(clear_cookie(SESSION_COOKIE))
    return response


def redirect(location: str) -> Response:
    return Response(SEE_OTHER, "text/plain; charset=utf-8", b"", headers={"Location": location})


def set_cookie(name: str, value: str, max_age: int, same_site: str) -> str:
    return f"{name}={value}; Path=/; Secure; HttpOnly; SameSite={same_site}; Max-Age={max_age}"


def clear_cookie(name: str) -> str:
    return f"{name}=; Path=/; Secure; HttpOnly; SameSite=Lax; Max-Age=0"
