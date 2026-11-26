"""Showcase_Entries: finished Jobs the account owner shows every signed-in Account, with the
footage's licence and credit."""

from __future__ import annotations

import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

from splatpipe.errors import PipelineError
from splatpipe.jobs import check_job_id

from app.statuses import DONE, out_prefix, read_summary, result_file
from cloud.errors import CloudError
from cloud.table import ITEM_ABSENT, ConditionFailed, Key, Prefix, Scalar

if TYPE_CHECKING:
    from splatpipe.config import Config

    from app.config import WebsiteConfig
    from cloud.aws import Aws

FOOTAGE_LICENCES = {"own": "our own video", "cc0": "CC0", "cc-by": "CC BY",
                    "cc-by-sa": "CC BY-SA", "pexels": "the Pexels licence"}
NEEDS_CREDIT = frozenset({"cc-by", "cc-by-sa"})
SHOWCASE = "showcase"


@dataclass(frozen=True)
class ShowcaseEntry:
    job_id: str
    title: str
    licence: str
    credit: str | None
    added_at: int


def entry_key(job_id: str) -> Key:
    return Key(SHOWCASE, f"job#{job_id}")


def check_entry(job_id: str, title: str, licence: str, credit: str | None,
                wc: WebsiteConfig, now: datetime) -> ShowcaseEntry:
    try:
        job_id = check_job_id(job_id)
    except PipelineError as error:
        raise CloudError(str(error)) from None
    if not title.strip() or len(title) > wc.max_title_chars or _has_control(title):
        raise CloudError(f"The title must be 1 to {wc.max_title_chars} characters with no control "
                         "characters.")
    if licence not in FOOTAGE_LICENCES:
        raise CloudError(f"{licence!r} isn't a footage licence we take. Use one of "
                         f"{', '.join(FOOTAGE_LICENCES)}.")
    if credit is not None and (len(credit) > wc.max_credit_chars or _has_control(credit)):
        raise CloudError(f"The credit must be at most {wc.max_credit_chars} characters with no "
                         "control characters.")
    if licence in NEEDS_CREDIT and not (credit or "").strip():
        raise CloudError(f"Footage under {FOOTAGE_LICENCES[licence]} needs its author's credit. Add "
                         "--credit with the author's name and where the footage came from.")
    return ShowcaseEntry(job_id=job_id, title=title.strip(), licence=licence,
                         credit=credit.strip() if credit else None, added_at=int(now.timestamp()))


def add_entry(aws: Aws, entry: ShowcaseEntry, config: Config) -> None:
    """Add a finished Job with a splat, from the website or a laptop. An entry already there is
    refused."""
    prefix = out_prefix(entry.job_id)
    names = [key[len(prefix):] for key in aws.s3_list(prefix)]
    summary = read_summary(aws, entry.job_id)
    status = (summary or {}).get("status")
    preset = (summary or {}).get("preset")
    if status not in DONE or not isinstance(preset, str):
        raise CloudError(f"Job {entry.job_id} hasn't finished as done or done with warnings, so it "
                         "can't go in the showcase. Check its status first.")
    if result_file(names, config.step_settings("clean", preset)) is None:
        raise CloudError(f"Job {entry.job_id} has no splat in its out/ folder, so it can't go in "
                         "the showcase.")
    key = entry_key(entry.job_id)
    item: dict[str, Scalar] = {"pk": key.pk, "sk": key.sk, "title": entry.title,
                               "licence": entry.licence, "added_at": entry.added_at}
    if entry.credit:
        item["credit"] = entry.credit
    try:
        aws.table_put(item, ITEM_ABSENT)
    except ConditionFailed:
        raise CloudError(f"Job {entry.job_id} is already in the showcase. Remove it first with "
                         f"`remove {entry.job_id}`, then add it again.") from None


def read_entry(item: Mapping[str, Scalar]) -> ShowcaseEntry:
    return ShowcaseEntry(job_id=str(item["sk"])[len("job#"):], title=str(item.get("title", "")),
                         licence=str(item.get("licence", "")),
                         credit=str(item["credit"]) if "credit" in item else None,
                         added_at=int(item.get("added_at", 0)))


def list_entries(aws: Aws) -> list[ShowcaseEntry]:
    entries = [read_entry(item) for item in aws.table_query(SHOWCASE, Prefix("job#"))]
    return sorted(entries, key=lambda entry: (entry.added_at, entry.job_id), reverse=True)


def remove_entry(aws: Aws, job_id: str) -> bool:
    key = entry_key(job_id)
    if aws.table_get(key) is None:
        return False
    aws.table_delete(key)
    return True


def licence_line(entry: ShowcaseEntry) -> str | None:
    if entry.licence == "cc-by":
        return f"Footage: {entry.credit}, CC BY"
    if entry.licence == "cc-by-sa":
        return (f"Footage: {entry.credit}, CC BY-SA. This splat is shared under CC BY-SA 4.0.")
    return f"Footage: {entry.credit}" if entry.credit else None


def _has_control(text: str) -> bool:
    return any(unicodedata.category(ch) == "Cc" for ch in text)
