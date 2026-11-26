"""The fake Website_Table's evaluator: `cloud.table`'s conditions and updates over plain dicts.

Pure functions, so the fake's semantics stay small and checkable: a comparison with a missing
attribute is false, as in DynamoDB, and `Add` on a missing attribute starts from 0, as DynamoDB's
ADD does. A missing item is an empty dict, so every attribute of it is missing.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from cloud.table import (Add, And, AtLeast, AtMost, Condition, Equals, Exists, NotExists, Or,
                         Remove, Scalar, Set, Update)


def holds(condition: Condition | None, item: Mapping[str, Scalar] | None) -> bool:
    """Whether `condition` holds for `item`, which is None when the item isn't there."""
    if condition is None:
        return True
    values = dict(item or {})
    if isinstance(condition, Exists):
        return condition.attr in values
    if isinstance(condition, NotExists):
        return condition.attr not in values
    if isinstance(condition, Equals):
        return condition.attr in values and _same(values[condition.attr], condition.value)
    if isinstance(condition, AtMost):
        value = values.get(condition.attr)
        return _number(value) and value <= condition.limit
    if isinstance(condition, AtLeast):
        value = values.get(condition.attr)
        return _number(value) and value >= condition.limit
    if isinstance(condition, And):
        return all(holds(part, item) for part in condition.parts)
    if isinstance(condition, Or):
        return any(holds(part, item) for part in condition.parts)
    raise TypeError(f"not a condition: {condition!r}")


def apply(updates: Sequence[Update], item: Mapping[str, Scalar] | None,
          key: Mapping[str, str]) -> dict[str, Scalar]:
    """The item after `updates`, made from `key` when it wasn't there."""
    result: dict[str, Scalar] = dict(item) if item is not None else dict(key)
    for update in updates:
        if isinstance(update, Set):
            result[update.attr] = update.value
        elif isinstance(update, Add):
            current = result.get(update.attr, 0)
            if not _number(current):
                raise TypeError(f"ADD to a non-number attribute {update.attr!r}")
            result[update.attr] = current + update.amount
        elif isinstance(update, Remove):
            result.pop(update.attr, None)
        else:
            raise TypeError(f"not an update: {update!r}")
    return result


def _number(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _same(left: Scalar, right: Scalar) -> bool:
    """DynamoDB equality: the same type and the same value, so True never equals 1."""
    return type(left) is type(right) and left == right
