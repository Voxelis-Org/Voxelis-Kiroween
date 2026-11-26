"""Accounts: names, passwords, scrypt hashes and the Account items in the Website_Table.

An Account's profile item, `account#<name>` / `profile`, holds its scrypt hash and salt (base64),
the scrypt settings the hash was made with, and a random `uid`. Every new password and every
re-made Account gets a new `uid`, and a Session whose `uid` differs is signed out, so a new password
ends every Session at once and a removed Account's Sessions never come back.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import re
import string
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

from app.config import ScryptSettings
from cloud.errors import CloudError
from cloud.table import ITEM_ABSENT, ITEM_THERE, ConditionFailed, Key, Scalar, Set

if TYPE_CHECKING:
    from app.config import WebsiteConfig
    from app.web import Random
    from cloud.aws import Aws

# The Account_Name_Pattern, checked with fullmatch. It stays in code because it's a safety check.
ACCOUNT_NAME = re.compile(r"[a-z][a-z0-9-]{0,31}")
PASSWORD_ALPHABET = string.ascii_letters + string.digits
# OpenSSL's scrypt allocates 128 * r * (n + p + 2) bytes; twice that is a safe maxmem.
_MAXMEM_FACTOR = 256
_UID_BYTES = 16


@dataclass(frozen=True)
class Account:
    name: str
    uid: str
    password_hash: bytes
    salt: bytes
    settings: ScryptSettings


def check_account_name(value: object) -> str:
    """`value` when it matches the Account_Name_Pattern, else a CloudError giving the pattern."""
    if isinstance(value, str) and ACCOUNT_NAME.fullmatch(value):
        return value
    raise CloudError(f"{value!r} isn't an account name. Use 1 to 32 characters of lowercase letters, "
                     "digits and hyphens, starting with a letter (the pattern "
                     f"{ACCOUNT_NAME.pattern}).")


def valid_name(value: object) -> bool:
    return isinstance(value, str) and ACCOUNT_NAME.fullmatch(value) is not None


def profile_key(name: str) -> Key:
    return Key(f"account#{name}", "profile")


def new_password(length: int, random: Random) -> str:
    return "".join(random.choice(PASSWORD_ALPHABET) for _ in range(length))


def hash_password(password: str, salt: bytes, settings: ScryptSettings) -> bytes:
    """scrypt with the hash's own settings, so a later change of [website] still checks old hashes."""
    maxmem = _MAXMEM_FACTOR * settings.r * (settings.n + settings.p + 2)
    return hashlib.scrypt(password.encode("utf-8"), salt=salt, n=settings.n, r=settings.r,
                          p=settings.p, dklen=settings.dklen, maxmem=maxmem)


def password_matches(password: str, account: Account | None, configured: ScryptSettings) -> bool:
    """Exactly one scrypt either way: with no Account, a dummy hash with the configured settings, so
    an unknown name takes as long as a wrong password."""
    if account is None:
        hash_password(password, bytes(configured.salt_bytes), configured)
        return False
    made = hash_password(password, account.salt, account.settings)
    return hmac.compare_digest(made, account.password_hash)


def read_account(name: str, item: Mapping[str, Scalar]) -> Account:
    settings = ScryptSettings(n=int(item["n"]), r=int(item["r"]), p=int(item["p"]),
                              dklen=int(item["dklen"]), salt_bytes=0)
    return Account(name=name, uid=str(item["uid"]),
                   password_hash=base64.b64decode(str(item["hash"])),
                   salt=base64.b64decode(str(item["salt"])), settings=settings)


def make_account(aws: Aws, name: str, wc: WebsiteConfig, random: Random, now: datetime) -> str:
    """Make the Account and give back its new password. A taken name changes nothing."""
    password = new_password(wc.password_length, random)
    item = {"pk": profile_key(name).pk, "sk": "profile", "created_at": int(now.timestamp()),
            **_secret_fields(password, wc, random)}
    try:
        aws.table_put(item, ITEM_ABSENT)
    except ConditionFailed:
        raise CloudError(f"An account named {name} already exists. To give it a new password, run "
                         "the command again with --new-password.") from None
    return password


def replace_password(aws: Aws, name: str, wc: WebsiteConfig, random: Random) -> str:
    """Give the Account a new password and a new uid, which ends every Session of it."""
    password = new_password(wc.password_length, random)
    updates = tuple(Set(attr, value) for attr, value in _secret_fields(password, wc, random).items())
    try:
        aws.table_update(profile_key(name), updates, ITEM_THERE)
    except ConditionFailed:
        raise CloudError(f"There's no account named {name}. Make it by running the command without "
                         "--new-password.") from None
    return password


def remove_account(aws: Aws, name: str) -> int:
    """Delete the Account's own items (its profile, Sessions, Uploads, job items and share links)
    and give back how many. The Job_Bucket's files, the Limit_Counters, the Sign_In_Failures and
    the Showcase_Entries stay."""
    items = aws.table_query(profile_key(name).pk)
    for item in items:
        aws.table_delete(Key(str(item["pk"]), str(item["sk"])))
    return len(items)


def _secret_fields(password: str, wc: WebsiteConfig, random: Random) -> dict[str, Scalar]:
    salt = random.token_bytes(wc.scrypt.salt_bytes)
    made = hash_password(password, salt, wc.scrypt)
    return {"hash": base64.b64encode(made).decode("ascii"),
            "salt": base64.b64encode(salt).decode("ascii"),
            "n": wc.scrypt.n, "r": wc.scrypt.r, "p": wc.scrypt.p, "dklen": wc.scrypt.dklen,
            "uid": random.token_bytes(_UID_BYTES).hex()}
