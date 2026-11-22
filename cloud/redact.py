"""Account_Redaction: keep AWS account ids out of chats, logs in the repo and data/.

One helper, `account_redaction(text)`, replaces every maximal run of exactly 12 digits with
`<account>`, so an 11- or 13-digit run is left alone. Laptop commands redact AWS errors, console
output and job logs before printing; the machine redacts the text logs it uploads. JSON reports
are never passed through it, because they hold no account id (Req 13.7).

Standard library only.
"""

from __future__ import annotations

import re

_ACCOUNT = re.compile(r"(?<!\d)\d{12}(?!\d)")


def account_redaction(text: str) -> str:
    """Replace every maximal run of exactly 12 digits with <account>.

    An 11- or 13-digit run is left alone, because the lookarounds require the run's neighbours to
    be non-digits (or the string's edge).
    """
    return _ACCOUNT.sub("<account>", text)
