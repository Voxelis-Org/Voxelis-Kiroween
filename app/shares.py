"""Share links: a secret link to one finished Job's result page and viewer, read-only, with no
sign-in.

The Job's owner makes the link; the website shows it once, on the page the button opens, and keeps
only its SHA-256 in the table, the way it keeps Session ids. A new link turns the old one off, and
the owner can turn the link off at any time. A link works only while both of these agree:

- `share` / `token#<SHA-256 hex>` holds the Job_Id and the owner, with `state` "on";
- the owner's held job item holds the same hash in `share_hash`.

A wrong, malformed or turned-off link gets the same 404 as a page that doesn't exist.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

from app.errors import bad_request, not_found
from app.limits import job_key
from app.sessions import new_token, well_formed_token
from cloud.table import ITEM_ABSENT, ConditionFailed, Equals, Key, Remove, Set

if TYPE_CHECKING:
    from collections.abc import Mapping

    from app.sessions import Session
    from app.web import Deps
    from cloud.table import Scalar

SHARE = "share"
NOT_FINISHED = "Only a finished job with a splat can have a share link."


@dataclass(frozen=True)
class SharedJob:
    job_id: str
    account: str
    job_item: Mapping[str, Scalar]


def token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("ascii")).hexdigest()


def share_key(digest: str) -> Key:
    return Key(SHARE, f"token#{digest}")


def make_link(deps: Deps, session: Session, job_id: str, job_item: Mapping[str, Scalar],
              now: datetime) -> str:
    """A new share token for the Account's own finished Job, turning any older link off."""
    old = job_item.get("share_hash")
    token = new_token(deps)
    digest = token_hash(token)
    key = share_key(digest)
    deps.aws.table_put({"pk": key.pk, "sk": key.sk, "job_id": job_id, "account": session.account,
                        "state": "on", "created_at": int(now.timestamp())}, ITEM_ABSENT)
    try:
        deps.aws.table_update(job_key(session.account, job_id), (Set("share_hash", digest),),
                              Equals("state", "held"))
    except ConditionFailed:
        _turn_off(deps, digest)
        raise not_found() from None
    if isinstance(old, str):
        _turn_off(deps, old)
    return token


def turn_off(deps: Deps, session: Session, job_id: str, job_item: Mapping[str, Scalar]) -> None:
    """Turn the Job's share link off. Nothing is deleted."""
    digest = job_item.get("share_hash")
    if not isinstance(digest, str):
        raise bad_request("This job has no share link to turn off.")
    try:
        deps.aws.table_update(job_key(session.account, job_id), (Remove("share_hash"),),
                              Equals("share_hash", digest))
    except ConditionFailed:
        pass  # another request changed it already
    _turn_off(deps, digest)


def shared_job(deps: Deps, token: str) -> SharedJob:
    """The Job a share token opens, or the 404. No AWS call for a malformed token."""
    if not well_formed_token(token, deps.website.token_bytes):
        raise not_found()
    digest = token_hash(token)
    share = deps.aws.table_get(share_key(digest))
    if share is None or share.get("state") != "on":
        raise not_found()
    account, job_id = str(share.get("account", "")), str(share.get("job_id", ""))
    item = deps.aws.table_get(job_key(account, job_id))
    if item is None or item.get("state") != "held" or item.get("share_hash") != digest:
        raise not_found()
    return SharedJob(job_id=job_id, account=account, job_item=item)


def _turn_off(deps: Deps, digest: str) -> None:
    try:
        deps.aws.table_update(share_key(digest), (Set("state", "off"),), Equals("state", "on"))
    except ConditionFailed:
        pass
