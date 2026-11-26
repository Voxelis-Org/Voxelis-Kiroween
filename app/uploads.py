"""Uploads: the browser sends the video straight to `uploads/<id>` in parts, never through the
website. The website begins the multipart upload, makes one Upload_Link per part, and completes it.

An Upload item, `account#<name>` / `upload#<32 hex>`, is read only under the signed-in Account's own
partition, so another Account's Upload is simply not there. Each state change is conditioned on the
state it comes from: open, then complete, refused or aborted.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

from app.errors import BAD_REQUEST, PublicRefusal, bad_request, not_found
from app.limits import is_stopped, stopped, upload_key
from app.video import INCOMPLETE, video_file_for
from cloud.table import ITEM_ABSENT, ConditionFailed, Equals, Set

if TYPE_CHECKING:
    from app.sessions import Session
    from app.web import Deps

UPLOAD_ID = re.compile(r"[0-9a-f]{32}")
ETAG = re.compile(r'"[0-9a-f]{32}"')
_ID_BYTES = 16
WRONG_STATE = "That upload isn't open any more. Upload the video again."


@dataclass(frozen=True)
class Upload:
    id: str
    account: str
    key: str
    s3_upload_id: str
    name: str
    size: int
    parts: int
    state: str
    used_by: str | None


def clean_source_name(name: str, max_chars: int) -> str:
    """The part after the last / or \\, without control characters, stripped, at most max_chars."""
    tail = re.split(r"[/\\]", name)[-1]
    kept = "".join(ch for ch in tail if unicodedata.category(ch) != "Cc").strip()
    return kept[:max_chars].strip()


def parts_needed(size: int, part_bytes: int) -> int:
    return max(1, -(-size // part_bytes))


def read_upload(deps: Deps, account: str, upload_id: str) -> Upload | None:
    if not UPLOAD_ID.fullmatch(upload_id):
        return None
    item = deps.aws.table_get(upload_key(account, upload_id))
    if item is None:
        return None
    return Upload(id=upload_id, account=account, key=str(item["key"]),
                  s3_upload_id=str(item["s3_upload_id"]), name=str(item["name"]),
                  size=int(item["size"]), parts=int(item["parts"]), state=str(item["state"]),
                  used_by=str(item["used_by"]) if "used_by" in item else None)


def begin(deps: Deps, session: Session, name: object, size: object, now: datetime) -> Upload:
    wc, jobs = deps.website, deps.config.jobs()
    if is_stopped(now, wc):
        raise stopped(wc)
    if not isinstance(name, str):
        raise bad_request("Pick a video file to upload.")
    clean = clean_source_name(name, wc.max_name_chars)
    if not clean:
        raise bad_request("That file has no name the website can use. Rename it and try again.")
    video_file_for(clean)
    if isinstance(size, bool) or not isinstance(size, int) or size < 1:
        raise bad_request("That file is empty. Pick a video and try again.")
    if size > jobs.max_video_bytes:
        raise PublicRefusal(BAD_REQUEST, "bad_request",
                            f"This video is larger than the {jobs.max_video_bytes} bytes a job "
                            "takes. Film a shorter video, or trim it on your phone, and try again.")
    upload_id = deps.random.token_bytes(_ID_BYTES).hex()
    key = f"uploads/{upload_id}"
    parts = parts_needed(size, wc.upload_part_bytes)
    s3_upload_id = deps.aws.s3_start_upload(key)
    item_key = upload_key(session.account, upload_id)
    expires = now + timedelta(hours=wc.upload_record_hours)
    deps.aws.table_put({"pk": item_key.pk, "sk": item_key.sk, "key": key,
                        "s3_upload_id": s3_upload_id, "name": clean, "size": size, "parts": parts,
                        "state": "open", "created_at": int(now.timestamp()),
                        "expires_at": int(expires.timestamp())}, ITEM_ABSENT)
    return Upload(upload_id, session.account, key, s3_upload_id, clean, size, parts, "open", None)


def part_link(deps: Deps, session: Session, upload_id: str, part: int) -> str:
    upload = _open_upload(deps, session, upload_id)
    if not 1 <= part <= upload.parts:
        raise not_found()
    return deps.aws.s3_part_link(upload.key, upload.s3_upload_id, part,
                                 deps.website.upload_link_seconds)


def check_parts(raw: Any, count: int) -> list[tuple[int, str]]:
    """Part numbers 1 to count in order, each with an S3 ETag, or a refusal."""
    if not isinstance(raw, list) or len(raw) != count:
        raise bad_request(INCOMPLETE)
    parts: list[tuple[int, str]] = []
    for expected, entry in enumerate(raw, start=1):
        if (not isinstance(entry, list) or len(entry) != 2 or entry[0] != expected
                or isinstance(entry[0], bool) or not isinstance(entry[1], str)
                or not ETAG.fullmatch(entry[1])):
            raise bad_request(INCOMPLETE)
        parts.append((expected, entry[1]))
    return parts


def complete(deps: Deps, session: Session, upload_id: str, raw_parts: Any) -> Upload:
    upload = _open_upload(deps, session, upload_id)
    parts = check_parts(raw_parts, upload.parts)
    deps.aws.s3_complete_upload(upload.key, upload.s3_upload_id, parts)
    size = deps.aws.s3_size(upload.key)
    good = size is not None and size == upload.size and size <= deps.config.jobs().max_video_bytes
    state = "complete" if good else "refused"
    _move(deps, session, upload, state)
    if not good:
        raise bad_request(INCOMPLETE)
    return Upload(**{**upload.__dict__, "state": state})


def abort(deps: Deps, session: Session, upload_id: str) -> None:
    upload = _open_upload(deps, session, upload_id)
    deps.aws.s3_abort_upload(upload.key, upload.s3_upload_id)
    _move(deps, session, upload, "aborted")


def _open_upload(deps: Deps, session: Session, upload_id: str) -> Upload:
    upload = read_upload(deps, session.account, upload_id)
    if upload is None:
        raise not_found()
    if upload.state != "open":
        raise bad_request(WRONG_STATE)
    return upload


def _move(deps: Deps, session: Session, upload: Upload, state: str) -> None:
    try:
        deps.aws.table_update(upload_key(session.account, upload.id), (Set("state", state),),
                              Equals("state", "open"))
    except ConditionFailed:
        raise bad_request(WRONG_STATE) from None


def describe(upload: Upload, part_bytes: int) -> dict[str, Any]:
    return {"upload": upload.id, "parts": upload.parts, "part_bytes": part_bytes}

