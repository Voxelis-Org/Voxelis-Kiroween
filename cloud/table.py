"""Typed keys, conditions, updates and transaction items for the Website_Table, with no boto3.

`app/` builds every read and write of the table from these values, `cloud.aws.BotoAws` turns them
into DynamoDB expressions with placeholder names and values, and the tests' fake table evaluates the
same values directly. Scalars are `str`, `int` and `bool` only: money is whole cents and times are
epoch seconds, so no float ever reaches the table.

Standard library only, so the tests and `app/` can import it without boto3.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Union

Scalar = Union[str, int, bool]
# The most items one TransactWriteItems or TransactGetItems may hold.
MAX_TRANSACTION_ITEMS = 100


@dataclass(frozen=True)
class Key:
    pk: str
    sk: str


@dataclass(frozen=True)
class Prefix:
    """A sort-key range: every sk that begins with `value`."""

    value: str


@dataclass(frozen=True)
class Between:
    """A sort-key range: low <= sk <= high."""

    low: str
    high: str


KeyRange = Union[Prefix, Between]


# Conditions. A comparison with a missing attribute is false, as in DynamoDB.

@dataclass(frozen=True)
class Exists:
    attr: str


@dataclass(frozen=True)
class NotExists:
    attr: str


@dataclass(frozen=True)
class Equals:
    attr: str
    value: Scalar


@dataclass(frozen=True)
class AtMost:
    """attr <= limit."""

    attr: str
    limit: int


@dataclass(frozen=True)
class AtLeast:
    """attr >= limit."""

    attr: str
    limit: int


@dataclass(frozen=True)
class And:
    parts: tuple["Condition", ...]


@dataclass(frozen=True)
class Or:
    parts: tuple["Condition", ...]


Condition = Union[Exists, NotExists, Equals, AtMost, AtLeast, And, Or]


# Updates. Add on a missing attribute starts from 0, as DynamoDB's ADD does.

@dataclass(frozen=True)
class Set:
    attr: str
    value: Scalar


@dataclass(frozen=True)
class Add:
    attr: str
    amount: int


@dataclass(frozen=True)
class Remove:
    attr: str


Update = Union[Set, Add, Remove]


# Transaction items.

@dataclass(frozen=True)
class Put:
    """Write a whole item, which holds its own pk and sk."""

    item: Mapping[str, Scalar]
    condition: Condition | None = None


@dataclass(frozen=True)
class Change:
    """Update one item's attributes, making the item when it isn't there."""

    key: Key
    updates: tuple[Update, ...]
    condition: Condition | None = None


@dataclass(frozen=True)
class Check:
    key: Key
    condition: Condition


TransactItem = Union[Put, Change, Check]


class ConditionFailed(Exception):
    """A condition failed, so nothing changed. `index` is the failing transaction item, in request
    order, or None for a single put or update."""

    index: int | None

    def __init__(self, index: int | None = None) -> None:
        super().__init__("A condition on the website's table didn't hold, so nothing changed."
                         + ("" if index is None else f" (item {index})"))
        self.index = index


class TransactionConflict(Exception):
    """Another write touched one of the items at the same moment, so nothing changed."""

    def __init__(self) -> None:
        super().__init__("Another write touched the same item at the same moment, so nothing "
                         "changed.")


ITEM_THERE = Exists("pk")       # the item exists
ITEM_ABSENT = NotExists("pk")   # the item doesn't exist yet


def item_key(item: Mapping[str, Scalar]) -> Key:
    """The Key of a whole item."""
    return Key(str(item["pk"]), str(item["sk"]))


def transact_key(entry: TransactItem) -> Key:
    """The Key one transaction item touches."""
    return item_key(entry.item) if isinstance(entry, Put) else entry.key
