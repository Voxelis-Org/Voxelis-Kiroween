"""The stack template's website parts, read with cfn-lint's own YAML decoder, and the parts the
website must leave as they were."""

from __future__ import annotations

import json
import re

import pytest
from cfnlint.decode import decode_str

from tests.website_helpers import REPO_ROOT

TEMPLATE_PATH = REPO_ROOT / "cloud" / "aws-setup.yaml"
TEXT = TEMPLATE_PATH.read_text(encoding="utf-8")
TEMPLATE, _ = decode_str(TEXT)
RESOURCES = TEMPLATE["Resources"]
WEBSITE = {"WebsiteTable": "AWS::DynamoDB::Table", "WebsiteLogGroup": "AWS::Logs::LogGroup",
           "WebsiteRole": "AWS::IAM::Role", "WebsitePolicy": "AWS::IAM::ManagedPolicy",
           "WebsiteFunction": "AWS::Lambda::Function", "WebsiteFunctionUrl": "AWS::Lambda::Url",
           "WebsiteUrlPermission": "AWS::Lambda::Permission",
           "WebsiteInvokePermission": "AWS::Lambda::Permission"}
ALLOWED_ACTIONS = {
    "dynamodb:GetItem", "dynamodb:PutItem", "dynamodb:UpdateItem", "dynamodb:Query",
    "logs:CreateLogStream", "logs:PutLogEvents", "cloudformation:DescribeStacks",
    "servicequotas:GetServiceQuota", "ec2:DescribeInstances", "ec2:DescribeImages",
    "ec2:DescribeSubnets", "ec2:DescribeInstanceTypeOfferings", "ec2:RunInstances",
    "ec2:CreateTags", "iam:PassRole", "s3:ListBucket", "s3:GetObject", "s3:PutObject",
    "s3:AbortMultipartUpload"}


def plain(value):
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, list):
        return [plain(v) for v in value]
    return value


def props(name: str) -> dict:
    return plain(RESOURCES[name]["Properties"])


def statements() -> list[dict]:
    return props("WebsitePolicy")["PolicyDocument"]["Statement"]


def test_the_website_resources_and_outputs_are_there() -> None:
    for name, kind in WEBSITE.items():
        assert RESOURCES[name]["Type"] == kind
    outputs = plain(TEMPLATE["Outputs"])
    assert outputs["WebsiteUrl"]["Value"] == {"Fn::GetAtt": ["WebsiteFunctionUrl", "FunctionUrl"]}
    assert outputs["WebsiteTableName"]["Value"] == {"Ref": "WebsiteTable"}
    assert len(TEXT.encode("utf-8")) <= 51200


def test_the_function_url_answers_browsers_only_through_the_url() -> None:
    assert props("WebsiteFunctionUrl")["AuthType"] == "NONE"
    assert props("WebsiteUrlPermission")["FunctionUrlAuthType"] == "NONE"
    assert props("WebsiteInvokePermission")["InvokedViaFunctionUrl"] is True
    function = props("WebsiteFunction")
    assert (function["Runtime"], function["MemorySize"], function["Timeout"]) == ("python3.11", 1024, 300)
    assert function["Environment"] == {"Variables": {"WEBSITE_TABLE": {"Ref": "WebsiteTable"}}}
    assert "APPROVED_SPEND" not in json.dumps(function)
    assert function["LoggingConfig"]["LogGroup"] == {"Ref": "WebsiteLogGroup"}
    assert "Bucket" not in json.dumps({k: v for k, v in function.items() if k != "Code"})


def test_the_function_url_targets_the_function_as_the_live_stack_has_it() -> None:
    # The live stack has this GetAtt, and any other value makes CloudFormation replace the URL,
    # which a function with one URL can't do. A code update shows the URL as Modify with a
    # replacement of Conditional, and app.deploy accepts that next to Modify WebsiteFunction False.
    target = props("WebsiteFunctionUrl")["TargetFunctionArn"]
    assert target == {"Fn::GetAtt": ["WebsiteFunction", "Arn"]}


def test_the_table_the_logs_and_the_bucket_rules() -> None:
    table = props("WebsiteTable")
    assert table["BillingMode"] == "PAY_PER_REQUEST"
    assert table["TimeToLiveSpecification"] == {"AttributeName": "expires_at", "Enabled": True}
    assert table["PointInTimeRecoverySpecification"] == {"PointInTimeRecoveryEnabled": True}
    assert RESOURCES["WebsiteTable"]["DeletionPolicy"] == "RetainExceptOnCreate"
    assert props("WebsiteLogGroup")["RetentionInDays"] == 30
    bucket = props("JobBucket")
    uploads = next(r for r in bucket["LifecycleConfiguration"]["Rules"] if r.get("Prefix") == "uploads/")
    assert uploads["ExpirationInDays"] == 1
    rules = {rule["Id"]: rule for rule in bucket["CorsConfiguration"]["CorsRules"]}
    assert rules["website-part-uploads"]["AllowedMethods"] == ["PUT"]
    assert rules["website-part-uploads"]["ExposedHeaders"] == ["ETag"]
    assert rules["website-viewer-reads"]["AllowedMethods"] == ["GET", "HEAD"]
    for rule in rules.values():
        assert "AllowedHeaders" not in rule
        assert "WebsiteFunctionUrl" in json.dumps(rule["AllowedOrigins"])
    for name in WEBSITE:
        tags = plain(RESOURCES[name].get("Properties", {})).get("Tags")
        if tags is not None:
            assert {"Key": "project", "Value": "voxelis-kiroween"} in tags


def test_the_website_policy_holds_only_the_allowed_actions() -> None:
    actions = set()
    for statement in statements():
        assert statement["Effect"] == "Allow"
        listed = statement["Action"]
        actions.update([listed] if isinstance(listed, str) else listed)
    assert actions <= ALLOWED_ACTIONS
    for forbidden in ("Delete", "Terminate", "Stop", "CreateImage", "Deregister", "StartSession",
                      "Scan", "ConditionCheck", "iam:Get", "iam:List"):
        assert not [a for a in actions if forbidden in a], forbidden
    images = next(s for s in statements() if s["Sid"] == "LaunchFromMachineImage")
    assert images["Condition"]["StringEquals"]["ec2:Owner"] == {"Ref": "AWS::AccountId"}
    assert not [s for s in statements() if s.get("Condition", {}).get("StringEquals", {})
                .get("ec2:Owner") == "amazon"]


def iam_match(pattern: str, key: str) -> bool:
    regex = "".join(".*" if ch == "*" else "." if ch == "?" else re.escape(ch) for ch in pattern)
    return re.fullmatch(regex, key) is not None


def write_patterns() -> list[str]:
    write = next(s for s in statements() if s["Sid"] == "WriteUploadsAndJobInputs")
    return [entry["Fn::Sub"].split("}/", 1)[1] for entry in write["Resource"]]


@pytest.mark.parametrize(("key", "allowed"), [
    ("uploads/" + "a" * 32, True),
    ("jobs/20261201-120000-abcdef/input/video.mp4", True),
    ("jobs/20261201-120000-abcdef/input/job.json", True),
    ("jobs/20261201-120000-abcdef/out/clean.sog", False),
    ("jobs/20261201-120000-abcdef/out/input/x", False),
    ("work/20261201-120000-abcdef/frames/1.jpg", False),
    ("jobs/_smoke/ami-0123456789abcdef0.json", False),
    ("website/" + "a" * 40 + ".zip", False),
])
def test_the_website_writes_never_reach_a_jobs_results(key: str, allowed: bool) -> None:
    assert any(iam_match(pattern, key) for pattern in write_patterns()) is allowed


def test_the_parts_the_website_leaves_alone_are_unchanged() -> None:
    kept = json.loads((REPO_ROOT / "tests" / "snapshots" / "stack_kept.json").read_text())
    for name, snapshot in kept.items():
        assert plain(RESOURCES[name]) == snapshot, name


def test_nothing_on_the_functions_side_refers_to_the_bucket() -> None:
    for name in ("WebsiteTable", "WebsiteLogGroup", "WebsiteRole", "WebsiteFunction",
                 "WebsiteFunctionUrl", "WebsiteUrlPermission", "WebsiteInvokePermission"):
        assert "JobBucket" not in json.dumps(plain(RESOURCES[name])), name
