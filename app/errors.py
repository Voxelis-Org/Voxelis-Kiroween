"""`PublicRefusal`: a refusal whose message is a Readable_Error that's safe to show in a browser.

A page shows the message in its message area, and a Script_Request gets
`{"error": kind, "message": message}` with the status. The handler passes every message through
`account_redaction` and `html.escape` before it reaches a page.
"""

from __future__ import annotations

# The HTTP statuses the refusals use, named once so the modules read as words.
BAD_REQUEST = 400
FORBIDDEN = 403
NOT_FOUND = 404
CONFLICT = 409
TOO_LARGE = 413
UNPROCESSABLE = 422
TOO_MANY = 429
SERVER = 500
UNAVAILABLE = 503


class PublicRefusal(Exception):
    """A refusal a person may see: its HTTP status, its kind and its Readable_Error."""

    status: int
    kind: str
    message: str

    def __init__(self, status: int, kind: str, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.kind = kind
        self.message = message


def not_found() -> PublicRefusal:
    """The one 404, the same whether or not the thing exists."""
    return PublicRefusal(NOT_FOUND, "not_found", "There's nothing here. Check the link, or go back "
                                                 "to your jobs.")


def bad_request(message: str) -> PublicRefusal:
    return PublicRefusal(BAD_REQUEST, "bad_request", message)
