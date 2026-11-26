"""The AWS seam's website additions, offline: botocore's Stubber checks the requests BotoAws sends
and how it reads AWS's errors, and the typed CloudErrors keep today's messages."""

from __future__ import annotations

from typing import Any
from urllib.parse import urlsplit

import boto3
import pytest
from botocore.stub import Stubber

import cloud.aws as aws_module
from cloud.aws import BotoAws, aws_setting, cloudshell_aws, s3_origin, website_aws
from cloud.errors import (CapacityExhausted, CapacityUnavailable, CloudError, NoGpuQuota,
                          QuotaFull, QuotaRefused)
from cloud.launch import launch_with_fallback
from cloud.select import admit
from cloud.table import (ITEM_ABSENT, Add, And, AtMost, Change, ConditionFailed, Equals, Key,
                         NotExists, Or, Put, Set, TransactionConflict)
from splatpipe.config import Config
from tests.fake_aws import FakeAws, cloud_config

TABLE = "voxelis-website-test"


@pytest.fixture
def boto() -> BotoAws:
    session = boto3.Session(aws_access_key_id="AKIAEXAMPLEEXAMPLE", aws_secret_access_key="x" * 40,
                            region_name="us-east-1")
    aws = BotoAws(session, region="us-east-1", bucket="voxelis-jobs-test")
    aws.table = TABLE
    return aws


def test_a_conditional_put_uses_placeholders_only(boto: BotoAws) -> None:
    with Stubber(boto._dynamodb) as stub:
        stub.add_response("put_item", {}, {
            "TableName": TABLE, "Item": {"pk": {"S": "a"}, "sk": {"S": "b"}, "n": {"N": "3"},
                                         "ok": {"BOOL": True}},
            "ConditionExpression": "attribute_not_exists(#n0)",
            "ExpressionAttributeNames": {"#n0": "pk"}})
        boto.table_put({"pk": "a", "sk": "b", "n": 3, "ok": True}, ITEM_ABSENT)


def test_an_update_builds_set_add_and_a_nested_condition(boto: BotoAws) -> None:
    with Stubber(boto._dynamodb) as stub:
        stub.add_response("update_item", {}, {
            "TableName": TABLE, "Key": {"pk": {"S": "limits"}, "sk": {"S": "total"}},
            "UpdateExpression": "SET #n0 = :v0 ADD #n1 :v1",
            "ConditionExpression": "(attribute_not_exists(#n1) OR #n1 <= :v2)",
            "ExpressionAttributeNames": {"#n0": "state", "#n1": "cents"},
            "ExpressionAttributeValues": {":v0": {"S": "held"}, ":v1": {"N": "5"},
                                          ":v2": {"N": "95"}}})
        boto.table_update(Key("limits", "total"), (Set("state", "held"), Add("cents", 5)),
                          Or((NotExists("cents"), AtMost("cents", 95))))


def test_a_cancelled_transaction_names_its_failing_item(boto: BotoAws) -> None:
    items = [Change(Key("a", "1"), (Add("n", 1),)),
             Put({"pk": "a", "sk": "2"}, And((ITEM_ABSENT, NotExists("x")))),
             Change(Key("a", "3"), (Set("s", "v"),), Equals("s", "w"))]
    with Stubber(boto._dynamodb) as stub:
        stub.add_client_error("transact_write_items", "TransactionCanceledException",
                              response_meta={}, modeled_fields={"CancellationReasons": [
                                  {"Code": "None"}, {"Code": "None"},
                                  {"Code": "ConditionalCheckFailed"}]})
        with pytest.raises(ConditionFailed) as caught:
            boto.table_transact(items, token="20261201-120000-abcdef")
    assert caught.value.index == 2
    with Stubber(boto._dynamodb) as stub:
        stub.add_client_error("transact_write_items", "TransactionCanceledException",
                              modeled_fields={"CancellationReasons": [
                                  {"Code": "TransactionConflict"}, {"Code": "None"},
                                  {"Code": "None"}]})
        with pytest.raises(TransactionConflict):
            boto.table_transact(items)
    with Stubber(boto._dynamodb) as stub:
        stub.add_client_error("put_item", "ConditionalCheckFailedException")
        with pytest.raises(ConditionFailed) as caught:
            boto.table_put({"pk": "a", "sk": "b"}, ITEM_ABSENT)
    assert caught.value.index is None


def test_get_many_is_one_transact_get_and_reads_numbers_as_ints(boto: BotoAws) -> None:
    with Stubber(boto._dynamodb) as stub:
        stub.add_response("transact_get_items", {"Responses": [
            {"Item": {"pk": {"S": "a"}, "sk": {"S": "b"}, "n": {"N": "12"}}}, {}]},
            {"TransactItems": [{"Get": {"TableName": TABLE, "Key": {"pk": {"S": "a"}, "sk": {"S": "b"}}}},
                               {"Get": {"TableName": TABLE, "Key": {"pk": {"S": "a"}, "sk": {"S": "c"}}}}]})
        got = boto.table_get_many([Key("a", "b"), Key("a", "c")])
    assert got == [{"pk": "a", "sk": "b", "n": 12}, None]


def test_a_table_call_without_a_table_is_refused(boto: BotoAws) -> None:
    boto.table = None
    with pytest.raises(CloudError, match="No website table"):
        boto.table_get(Key("a", "b"))


def test_vcpu_and_instance_limits_become_quota_refused(boto: BotoAws) -> None:
    for code in ("VcpuLimitExceeded", "InstanceLimitExceeded"):
        with Stubber(boto._ec2) as stub:
            stub.add_client_error("run_instances", code, "limit for 123456789012")
            with pytest.raises(QuotaRefused) as caught:
                boto.run_instances({"ImageId": "ami-1", "InstanceType": "g5.xlarge",
                                    "MinCount": 1, "MaxCount": 1})
        assert "123456789012" not in str(caught.value)


def test_job_machines_filters_by_the_project_and_the_job_id_tag(boto: BotoAws) -> None:
    with Stubber(boto._ec2) as stub:
        stub.add_response("describe_instances", {"Reservations": []}, {"Filters": [
            {"Name": "tag:project", "Values": ["voxelis-kiroween"]},
            {"Name": "tag:job_id", "Values": ["20261201-120000-abcdef"]}]})
        assert boto.job_machines("20261201-120000-abcdef") == []


def test_links_are_built_lazily_on_the_regional_virtual_host(boto: BotoAws) -> None:
    assert boto._links is None
    url = boto.s3_part_link("uploads/" + "a" * 32, "up-1", 3, 900)
    parts = urlsplit(url)
    assert f"{parts.scheme}://{parts.netloc}" == s3_origin("voxelis-jobs-test", "us-east-1")
    assert "partNumber=3" in parts.query and "uploadId=up-1" in parts.query
    assert "X-Amz-Expires=900" in parts.query
    download = boto.s3_download_link("jobs/x/out/clean.sog", "x-clean.sog", 300)
    assert urlsplit(download).path == "/jobs/x/out/clean.sog"
    assert "attachment" in download
    viewer = boto.s3_download_link("jobs/x/out/clean.sog", None, 300)
    assert "attachment" not in viewer and urlsplit(viewer).path.endswith(".sog")


def test_s3_size_is_one_exact_key_list(boto: BotoAws) -> None:
    with Stubber(boto._s3) as stub:
        stub.add_response("list_objects_v2", {"Contents": [{"Key": "uploads/a", "Size": 42}]},
                          {"Bucket": "voxelis-jobs-test", "Prefix": "uploads/a", "MaxKeys": 1})
        stub.add_response("list_objects_v2", {"Contents": [{"Key": "uploads/ab", "Size": 1}]},
                          {"Bucket": "voxelis-jobs-test", "Prefix": "uploads/a", "MaxKeys": 1})
        assert boto.s3_size("uploads/a") == 42
        assert boto.s3_size("uploads/a") is None


def test_change_set_changes_follow_the_next_token(boto: BotoAws) -> None:
    def change(name: str, action: str) -> dict[str, Any]:
        return {"Type": "Resource", "ResourceChange": {
            "Action": action, "LogicalResourceId": name, "Replacement": "False"}}
    with Stubber(boto._cfn) as stub:
        stub.add_response("describe_change_set", {"Changes": [change("A", "Modify")], "NextToken": "t"},
                          {"StackName": "s", "ChangeSetName": "c"})
        stub.add_response("describe_change_set", {"Changes": [change("B", "Add")]},
                          {"StackName": "s", "ChangeSetName": "c", "NextToken": "t"})
        stub.add_response("delete_change_set", {}, {"StackName": "s", "ChangeSetName": "c"})
        assert boto.change_set_changes("s", "c") == [("Modify", "A", "False"), ("Add", "B", "False")]
        boto.delete_change_set("s", "c")


def test_website_and_cloudshell_clients_name_only_the_region(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []

    class FakeSession:
        def __init__(self, **kwargs: Any) -> None:
            calls.append(kwargs)

        def client(self, name: str) -> Any:
            return object()

    monkeypatch.setattr(aws_module.boto3, "Session", FakeSession)
    website_aws("us-east-1")
    cloudshell_aws("us-east-1")
    assert calls == [{"region_name": "us-east-1"}, {"region_name": "us-east-1"}]


def test_aws_setting_reads_aws_and_refuses_a_missing_key() -> None:
    config = Config({"aws": {"stack_name": "voxelis-kiroween"}})
    assert aws_setting(config, "stack_name") == "voxelis-kiroween"
    with pytest.raises(CloudError, match="spend_variable"):
        aws_setting(config, "spend_variable")


def test_the_typed_errors_keep_todays_messages() -> None:
    cc = cloud_config()
    with pytest.raises(NoGpuQuota, match="no GPU quota"):
        admit(["g5.8xlarge"], 0, 0, cc)
    with pytest.raises(QuotaFull, match="no room for the smallest fitting type"):
        admit(["g5.8xlarge"], 8, 32, cc)
    fake = FakeAws(no_capacity=[("g5.8xlarge", "us-east-1a")], subnets={"us-east-1a": "subnet-a"})
    with pytest.raises(CapacityExhausted, match="no On-Demand capacity") as caught:
        launch_with_fallback(fake, [("g5.8xlarge", "subnet-a", "us-east-1a")],
                             lambda attempt, kind, subnet: {"InstanceType": kind, "SubnetId": subnet})
    assert caught.value.tried == (("g5.8xlarge", "us-east-1a"),)
    with pytest.raises(CloudError) as empty:
        launch_with_fallback(fake, [], lambda *args: {})
    assert not isinstance(empty.value, CapacityExhausted)
    assert issubclass(QuotaRefused, CloudError) and not issubclass(QuotaRefused, CapacityUnavailable)
