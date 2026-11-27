"""The fake Website_Table: DynamoDB's condition, update and transaction rules, and its call log."""

from __future__ import annotations

import pytest

from cloud.errors import CloudError
from cloud.table import (ITEM_ABSENT, ITEM_THERE, Add, And, AtLeast, AtMost, Between, Change,
                         ConditionFailed, Equals, Key, NotExists, Or, Prefix, Put, Remove, Set,
                         TransactionConflict)
from tests.fake_aws import FakeAws
from tests.fake_table import apply, holds


def test_a_comparison_with_a_missing_attribute_is_false() -> None:
    assert not holds(AtMost("n", 5), {})
    assert not holds(AtLeast("n", 0), None)
    assert not holds(Equals("s", "x"), {"pk": "a"})
    assert holds(NotExists("n"), {"pk": "a"})
    assert holds(Or((NotExists("n"), AtMost("n", 1))), {})
    assert not holds(And((ITEM_THERE, AtMost("n", 1))), {"pk": "a", "n": 2})


def test_equality_keeps_types_apart() -> None:
    assert not holds(Equals("v", 1), {"v": True})
    assert holds(Equals("v", True), {"v": True})


def test_add_starts_from_zero_and_remove_drops() -> None:
    item = apply((Add("n", 3), Set("s", "x"), Remove("gone")), {"pk": "a", "sk": "b", "gone": 1},
                 {"pk": "a", "sk": "b"})
    assert item == {"pk": "a", "sk": "b", "n": 3, "s": "x"}
    assert apply((Add("n", -1),), None, {"pk": "a", "sk": "b"}) == {"pk": "a", "sk": "b", "n": -1}


def test_put_update_and_get_record_their_calls() -> None:
    fake = FakeAws()
    fake.table_put({"pk": "a", "sk": "b", "v": 1}, ITEM_ABSENT)
    with pytest.raises(ConditionFailed) as caught:
        fake.table_put({"pk": "a", "sk": "b", "v": 2}, ITEM_ABSENT)
    assert caught.value.index is None
    fake.table_update(Key("a", "b"), (Add("v", 1),), ITEM_THERE)
    assert fake.table_get(Key("a", "b")) == {"pk": "a", "sk": "b", "v": 2}
    assert fake.table_get_many([Key("a", "b"), Key("a", "c")]) == [
        {"pk": "a", "sk": "b", "v": 2}, None]
    assert [op for op, _ in fake.calls] == ["table_put", "table_put", "table_update", "table_get",
                                            "table_get_many"]


def test_a_transaction_applies_all_or_nothing_and_names_the_failing_item() -> None:
    fake = FakeAws()
    fake.table_put({"pk": "limits", "sk": "total", "cents": 10})
    before = dict(fake.items)
    with pytest.raises(ConditionFailed) as caught:
        fake.table_transact([Change(Key("limits", "day"), (Add("jobs", 1),)),
                             Change(Key("limits", "total"), (Add("cents", 5),), AtMost("cents", 4)),
                             Put({"pk": "j", "sk": "1"}, ITEM_ABSENT)])
    assert caught.value.index == 1
    assert fake.items == before
    fake.table_transact([Change(Key("limits", "day"), (Add("jobs", 1),)),
                         Put({"pk": "j", "sk": "1"}, ITEM_ABSENT)], token="t1")
    assert fake.items[("limits", "day")]["jobs"] == 1
    fake.table_transact([Change(Key("limits", "day"), (Add("jobs", 1),))], token="t1")
    assert fake.items[("limits", "day")]["jobs"] == 1   # one token applies once


def test_a_transaction_refuses_two_items_on_one_key_and_can_conflict() -> None:
    fake = FakeAws()
    with pytest.raises(CloudError):
        fake.table_transact([Change(Key("a", "b"), (Add("n", 1),)),
                             Change(Key("a", "b"), (Add("n", 1),))])
    fake.conflicts = 1
    with pytest.raises(TransactionConflict):
        fake.table_transact([Change(Key("a", "b"), (Add("n", 1),))])
    fake.table_transact([Change(Key("a", "b"), (Add("n", 1),))])
    assert fake.items[("a", "b")]["n"] == 1


def test_query_reads_a_prefix_or_a_range_in_key_order() -> None:
    fake = FakeAws()
    for sk in ("job#2", "job#1", "upload#1", "failure#5", "lock"):
        fake.table_put({"pk": "p", "sk": sk})
    assert [i["sk"] for i in fake.table_query("p", Prefix("job#"))] == ["job#1", "job#2"]
    assert [i["sk"] for i in fake.table_query("p", Between("failure#", "lock"))] == [
        "failure#5", "job#1", "job#2", "lock"]
