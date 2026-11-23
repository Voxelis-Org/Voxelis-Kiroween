"""Cloud_Jobs errors and the CLI wrapper that every cloud command body runs in.

`CloudError` is a refusal or failure whose message is a Readable_Error, already Account_Redacted.
`CapacityUnavailable` is not a `CloudError`: the launch loop catches it to try the next zone or
type, so it stays a separate exception (Req 5.1). `run_cloud_cli` mirrors `splatpipe.errors.run_cli`
but also catches `SpendRefused` (from `cloud.spend`) and `CloudError`, prints every message through
`account_redaction`, prints no traceback, and lets `SystemExit` and `KeyboardInterrupt` pass
through (Req 1.9, 5.5, 13.6).

A cloud module may import `splatpipe` to call its public functions; `splatpipe` never imports
`cloud`. This module imports no boto3.
"""

from __future__ import annotations

import sys
from collections.abc import Callable

from splatpipe.errors import PipelineError

from cloud.redact import account_redaction
from cloud.spend import SpendRefused


class CloudError(Exception):
    """A refusal or failure whose message is a Readable_Error, already Account_Redacted."""


class CapacityUnavailable(Exception):
    """AWS has no On-Demand capacity for a type in a zone.

    Raised for `InsufficientInstanceCapacity` or `Unsupported`. Not a `CloudError`: the launch loop
    catches it to try the next zone or type (Req 5.1).
    """

    code: str
    zone: str
    instance_type: str

    def __init__(self, code: str, zone: str, instance_type: str) -> None:
        super().__init__(f"No {instance_type} capacity in {zone} ({code}).")
        self.code = code
        self.zone = zone
        self.instance_type = instance_type


def run_cloud_cli(action: Callable[[], int]) -> int:
    """Run a cloud CLI body and return its exit code.

    PipelineError, SpendRefused or CloudError: print its message, Account_Redacted, on stderr and
    return 1. Any other Exception: print a general bug message with the error's type and its
    Account_Redacted message, and return 1. No traceback is printed either way. SystemExit (argparse's
    usage errors and --help) and KeyboardInterrupt (Ctrl+C) aren't Exceptions, so they pass through.
    """
    try:
        return action()
    except (PipelineError, SpendRefused, CloudError) as error:
        message = str(error)
    except Exception as error:  # anything else is a bug in our code
        message = (f"cloud stopped because of a bug ({type(error).__name__}: {error}). "
                   "Tell the team the command you ran.")
    print(account_redaction(message), file=sys.stderr)
    return 1
