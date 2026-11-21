# Implementation Plan: aws-setup

## Overview

This plan builds the AWS setup in the design's build order of 4 tasks, within the user's limit of 5: cfn-lint with the Template_Test and the first part of the Stack_Template (the Job_Bucket, its bucket policy and the Job_Security_Group), then the IAM part of the Stack_Template, then the Stack's name in Config with the quota line in the Tech_Steering and the two region lines in the Rules_Steering, and last the Setup_Guide with the README link. Each task ends by running the whole suite and committing its own files. The Stack_Template is plain CloudFormation YAML, the Template_Test is Python 3.11 with pytest and cfn-lint, and the rest is Markdown and TOML. There is no AWS code: the start-command spec adds boto3 and the code that reads the Stack_Outputs. Nothing here calls AWS. Only the Account_Owner deploys and updates the Stack, from CloudShell, by following the Setup_Guide.

Some files grow over several tasks, so each part comes with the task that the design's build order gives it:

- `cloud/aws-setup.yaml`: task 1 writes the header comment, the `Description`, `JobBucket`, `JobBucketPolicy`, `JobSecurityGroup` and the outputs `BucketName`, `SecurityGroupId` and `VpcId`. Task 2 adds the nine IAM resources and the outputs `JobRoleArn`, `JobInstanceProfileArn`, `RunnerRoleArn` and `GroupName`. Each of the two tasks runs the Compliance_Check and checks the template's size before its commit.
- `docs/third-party-licences.md`: task 1 adds the cfn-lint row and the `awslabs.aws-iac-mcp-server` row, and task 4 adds the Session Manager plugin's row.
- `.kiro/steering/structure.md`: task 1 changes the `cloud/` line and task 4 the `docs/` line, so each line goes in with the file it describes.
- `.kiro/steering/rules.md`: task 1 adds CNRI-Python and `regex` under `## Licences`, in the commit that adds cfn-lint, and task 3 adds the two region lines under `## AWS access`.

## Working rules for every task

- Work on `feat/aws-setup`, never on `main`. If the branch doesn't exist yet, make it from `main` before task 1.
- Make no AWS call (Req 17.6), because the read profile doesn't exist until the Account_Owner has deployed the Stack. Run no `aws` command, `aws cloudformation validate-template` included, add or import no boto3, and use none of the power's tools that call AWS, such as `troubleshoot_cloudformation_deployment`. cfn-lint, the Template_Test and the Compliance_Check run on the laptop without credentials. The `aws` commands in the Setup_Guide are for the Account_Owner and the Operators: write them, never run them.
- Kiro never deploys or updates the Stack (Req 17.7). Only the Account_Owner does, from CloudShell in the AWS console, and on a laptop the guard hook blocks the commands that create or update stacks. Nothing here costs money, and no task sets `APPROVED_SPEND`.
- The Template_Test in task 1 is the only new test, and it's required, never optional, so no task or sub-task here carries a `*`. There are no property tests and no other new tests, by the user's choice for this spec (Req 17.5), though the steering asks each task to write its own tests and asks for property tests. The only other test change is the `UNCHANGED_SECTIONS` edit in task 3, which keeps the existing suite passing.
- Tests run on the CPU with no network and no AWS credentials.
- Before each commit that adds or changes `cloud/aws-setup.yaml`, in tasks 1 and 2, run the Compliance_Check, the power's `check_cloudformation_template_compliance` tool, on the whole template (Req 11). Fix what applies to the Stack. Suppress only Unfit_Rules, each in its resource's `Metadata: guard: SuppressedRules` list with a one-line reason as a YAML comment next to the rule id. Ask the user before suppressing any other rule. The template turns off no cfn-lint check (Req 17.3).
- Before each of those commits, also check the template's size by hand, for example with `(Get-Item cloud/aws-setup.yaml).Length` in PowerShell, which prints its size on disk in bytes, line endings included. Keep it at most 51,200 bytes, the most that `--template-body` takes (Req 1.12, 1.13). This is a check, not a test.
- The template holds its own numbers and names, as A5 allows, because CloudFormation can't read `config.toml`. The only Config change is `[aws].stack_name`, in task 3.
- Name, list and describe no Other_Resource, the older project's resources included, in code, docs, commits or the chat (Req 17.8). The template and the guide also name no person, no account id and no region (Req 1.9, 15.1). Before each commit of either file, search it for region names such as `us-east-1`, for 12-digit numbers, which would be account ids, and for the team members' names.
- Build A1 to A7, which the user accepted at review, and Q1 to Q3, which the user decided at review, as the design gives them. Q1 puts the deploy and every update in CloudShell, with `--template-body`, so the template stays within 51,200 bytes. Q2 puts the Work_Files under `work/<Job_Id>/` with no tag, while `input/` and `out/` stay under `jobs/<Job_Id>/`. So the Work_Files carry no tag, the lifecycle rules have no `TagFilters` and no role gets `s3:PutObjectTagging`. Q3 has Step 7 of the guide tell people to answer `n` when `aws configure` offers to set up AWS skills and the AWS MCP server for an AI coding agent.
- The last sub-task of each task is its checkpoint. Run `uv run pytest` and fix what fails until every test passes, `tests/test_guard.py` included. Then stage only that task's files by name with `git add <file> ...`, never with `git add -A`, `git add .` or a folder, and commit only those files with `git commit -m "<message>" -- <file> ...`, so nothing else that's staged goes in. Leave `docs/devlog.md` out: the devlog hook adds and commits its own line after each task.
- If a hook blocks something it shouldn't, say so and fix the hook. Never work around it.

## Tasks

- [ ] 1. cfn-lint, the template test, the job bucket and the security group
  - Files: `cloud/aws-setup.yaml` and `tests/test_aws_setup.py` (new), `pyproject.toml`, `uv.lock`, `.kiro/steering/rules.md`, `.kiro/steering/structure.md` and `docs/third-party-licences.md` (changed).
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.6, 1.7, 1.8, 1.9, 1.10, 1.11, 1.12, 1.13, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 4.1, 4.2, 4.3, 4.4, 10.1, 10.2, 10.3, 11.1, 11.2, 11.3, 11.4, 11.5, 11.6, 11.7, 16.3, 16.4, 16.6, 16.7, 16.8, 16.10, 16.11, 16.12, 17.1, 17.2, 17.3, 17.5, 17.6, 17.7, 17.8_

  - [ ] 1.1 Add cfn-lint to the dev group, and CNRI-Python to the allowed licences
    - In `.kiro/steering/rules.md`, replace the fourth line of the allowed list and the line after the list with the design's text, keeping their two-space indent. CNRI-Python (the licence of Python's early code) joins the list, and `regex` joins the packages named under it. The user OKed CNRI-Python for regex. Leave `## AWS access` as it is, because task 3 adds its two region lines.
    - Run `uv add --dev "cfn-lint>=1.61.0"`. `pyproject.toml` then lists cfn-lint in the `dev` group only, and `dependencies` stays empty.
    - Find the packages that the addition put in `uv.lock` with `git diff uv.lock`. Read each one's licence from its installed metadata with `importlib.metadata`: its `License-Expression`, `License` and licence classifiers. When the metadata only says BSD, read the licence file that comes with the package to tell which BSD it is. The design expects cfn-lint (MIT-0), PyYAML, jsonpatch, jsonpointer, networkx, sympy, mpmath, regex and typing-extensions.
    - If any of them has a licence outside the Allowed_Licences, with CNRI-Python now among them, stop and ask the user, and commit nothing until they answer.
    - _Requirements: 16.4, 16.6, 16.10, 16.11_

  - [ ] 1.2 Add the cfn-lint and MCP server rows to `docs/third-party-licences.md`
    - Add the design's cfn-lint row after the Hypothesis row. Its licence cell gives MIT-0 and each package that 1.1 found in `uv.lock`, with that package's licence, so the row follows the real lock where the design's list differs. It says the template test in `tests/test_aws_setup.py` runs cfn-lint on `cloud/aws-setup.yaml`, in the dev group only, and Installed is "Yes".
    - Add the design's `awslabs.aws-iac-mcp-server` row after the AWS CLI v2 row: Apache-2.0, Kiro runs its cfn-guard compliance check on `cloud/aws-setup.yaml` through the aws-infrastructure-as-code power, our code doesn't import or ship it, and Installed is "Yes, on the laptop, through the Kiro power". It comes now because 1.6 runs the Compliance_Check for the first time.
    - _Requirements: 16.7, 16.8, 16.12_

  - [ ] 1.3 Write `cloud/aws-setup.yaml` with the job bucket, its bucket policy and the security group
    - Start with the design's header comment. It says what the file is, that the Account_Owner deploys it from CloudShell by following `docs/aws-setup.md`, that its own numbers and names (the 60 and 7 days, the ports, the Allowed_Types, the Project_Tag and the prefixes `jobs/` and `work/`) live here and not in `config.toml` because CloudFormation can't read `config.toml` (A5), that changing one of them means editing this file and updating the Stack, and that the file stays within 51,200 bytes, the most that `--template-body` takes.
    - The top-level keys are `AWSTemplateFormatVersion: "2010-09-09"`, the design's `Description`, `Resources` and `Outputs`. There are no `Parameters` (A5), no `Transform`, macro or nested stack, and no `Mappings`, `Conditions` or top-level `Metadata`.
    - Write `JobBucket`, `JobBucketPolicy` and `JobSecurityGroup` as the design's YAML gives them, without the `Metadata` blocks: 1.6 adds the Suppressions that the real Compliance_Check run needs.
    - `JobBucket` has no `BucketName` and no `AccessControl`. It has `DeletionPolicy: RetainExceptOnCreate` and `UpdateReplacePolicy: Retain`, because `UpdateReplacePolicy` has no `RetainExceptOnCreate` value. Deleting the Stack then keeps the bucket and its files, and a failed first create rolls back and deletes the still-empty bucket, so it leaves nothing behind. cfn-lint v1.61.0 accepts the pair (E3035, E3036) and doesn't warn about it (W3011), as the design's "The Job_Bucket" section shows.
    - The 60-day rule, `delete-work-files-after-60-days`, has `Prefix: work/` as its only filter, with no `TagFilters`, and `ExpirationInDays: 60`. The 7-day rule, `abort-unfinished-uploads-after-7-days`, stays a rule of its own with no filter, so it covers the whole bucket, `jobs/` included, and only aborts unfinished uploads. So only files under `work/` are ever deleted.
    - The bucket policy's one `Deny` statement covers the bucket and `${JobBucket.Arn}/*`, and compares `aws:SecureTransport` with the YAML boolean `false`, not the string.
    - `JobSecurityGroup` has no `VpcId`, `GroupName` or `SecurityGroupIngress`, and only the two egress rules, TCP 443 and TCP 80 to `0.0.0.0/0`. Its descriptions use only the characters EC2 allows, so no apostrophes and no `&`.
    - `JobBucket` and `JobSecurityGroup` carry the Project_Tag, `project` with the value `voxelis-kiroween`.
    - Add the outputs `BucketName` (`!Ref JobBucket`), `SecurityGroupId` (`!GetAtt JobSecurityGroup.GroupId`) and `VpcId` (`!GetAtt JobSecurityGroup.VpcId`), each with the design's description and no `Export`. `BucketName`'s description says the bucket holds one `jobs/<job id>/` folder and one `work/<job id>/` folder per job. Never use `!Ref JobSecurityGroup` for the id, because for a group with no `VpcId` it gives the group's name.
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.6, 1.7, 1.8, 1.9, 1.10, 1.11, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 4.1, 4.2, 4.3, 4.4, 10.1, 10.2, 10.3_

  - [ ] 1.4 Change the `cloud/` line in `.kiro/steering/structure.md`
    - Replace it with the design's text, which says `cloud/` also holds `aws-setup.yaml`, the one CloudFormation template that makes our AWS resources. It goes in now, with the file it describes. Leave the other lines as they are.
    - _Requirements: 16.3_

  - [ ] 1.5 Write the Template_Test in `tests/test_aws_setup.py`
    - First check the installed cfn-lint's API by reading its source in `.venv`: `cfnlint.api.lint_file` takes the template's `Path` and a `ManualArgs`, `ManualArgs` takes `regions` and `mandatory_checks`, and each `Match` has `rule.id`, `rule.severity`, `linenumber` and `message`. Also check which strings `rule.severity` gives for E and W rules, because the test compares it with `"error"` and `"warning"`. If the API differs, fit the test to it and keep what the test checks.
    - Check that the installed cfn-lint has its bundled resource schemas, with `cfnlint/data/schemas/providers/us-east-1.json` in `.venv`. cfn-lint downloads schemas from GitHub when it finds none, so without them the test would reach the network. If they're missing, stop and tell the user instead of running the test.
    - Write the file as the design's "The Template_Test" section gives it. It reads `[aws].region` from the real `config.toml` by path with `tomllib`, fails when the template has a `cfn-lint` Metadata key, runs `lint_file` with `regions=[region]` and `mandatory_checks=["E", "W"]`, and fails on every error and warning, listing each as `<rule id> line <n>: <message>`. Informational findings don't fail it (A6).
    - The test is required, never optional. It needs no fixture, no network and no AWS credentials, and it's the only file that imports cfn-lint. Write no other new test (Req 17.5).
    - _Requirements: 17.1, 17.2, 17.3, 17.5_

  - [ ] 1.6 Run the Compliance_Check, make the Template_Test pass, and check the template's size
    - Activate the aws-infrastructure-as-code power, then run its `check_cloudformation_template_compliance` tool, on the `awslabs.aws-iac-mcp-server` server, with the whole of `cloud/aws-setup.yaml` as its `template_content` text.
    - Handle each failing rule with the four steps in the design's "The Compliance_Check" section. Fix it when it applies to the Stack. When the output shows "Unknown" as the resource, find the resource by passing the tool cut-down templates as text, each with `AWSTemplateFormatVersion` and a `Resources` section that holds one candidate, so no file is written.
    - Suppress only Unfit_Rules, in the `Metadata: guard: SuppressedRules` list of the resource that the rule fails on, each with its one-line reason as a YAML comment on the rule id's line. The design expects `S3_BUCKET_LOGGING_ENABLED`, `S3_BUCKET_VERSIONING_ENABLED`, `S3_BUCKET_REPLICATION_ENABLED`, `S3_BUCKET_DEFAULT_LOCK_ENABLED` and `S3_BUCKET_NO_PUBLIC_RW_ACL` on `JobBucket`, and `EC2_SECURITY_GROUP_EGRESS_OPEN_TO_WORLD_RULE` on `JobSecurityGroup`, with the reasons in its "Suppressions" table. Suppress only what the real run fails.
    - Add nothing that an Unfit_Rule asks for: no access logs, versioning, replication, object lock, KMS keys or `AccessControl`. If a rule that isn't an Unfit_Rule fails and can't be fixed within the requirements, stop and ask the user before suppressing it.
    - Run the check on the whole template again until it passes. That run is the check of record.
    - Run `uv run pytest tests/test_aws_setup.py` and fix each finding in the template, never by turning a check off. If a fix changes the template, run the Compliance_Check again.
    - Check the template's size by hand with `(Get-Item cloud/aws-setup.yaml).Length`. It must be at most 51,200 bytes. If it's over, shorten the template, for example its comments, then run the Compliance_Check and the Template_Test again, so the template that's committed has passed all three checks.
    - _Requirements: 1.12, 1.13, 11.1, 11.2, 11.3, 11.4, 11.5, 11.6, 11.7, 17.1, 17.3_

  - [ ] 1.7 Run the tests and commit
    - Search `cloud/aws-setup.yaml` for region names such as `us-east-1`, for 12-digit numbers and for the team members' names. It should hold none.
    - Run `uv run pytest`. Every test passes, the Template_Test and `tests/test_guard.py` included.
    - If anything in this sub-task changes `cloud/aws-setup.yaml`, run 1.6's three checks on it again before committing.
    - Check that the files and the commit message name no Other_Resource, and that nothing in this task called AWS or touched a stack.
    - On `feat/aws-setup`: `git add pyproject.toml uv.lock .kiro/steering/rules.md .kiro/steering/structure.md docs/third-party-licences.md cloud/aws-setup.yaml tests/test_aws_setup.py`, then `git commit -m "AWS setup: cfn-lint, the template test, the job bucket and the security group" -- pyproject.toml uv.lock .kiro/steering/rules.md .kiro/steering/structure.md docs/third-party-licences.md cloud/aws-setup.yaml tests/test_aws_setup.py`. cfn-lint, its licence row and CNRI-Python in `rules.md` go in this one commit.
    - _Requirements: 1.9, 16.12, 17.6, 17.7, 17.8_

- [ ] 2. The IAM part of the template
  - Files: `cloud/aws-setup.yaml` (changed).
  - Every IAM resource here follows the design: no name property (`RoleName`, `ManagedPolicyName`, `InstanceProfileName` or `GroupName`), every policy an `AWS::IAM::ManagedPolicy` with a `Description` and no inline `Policies`, `Version: "2012-10-17"` on each policy document, a `Sid` of letters and digits on each statement, and each ARN written out in full from the pseudo parameters, as the design's "ARNs" table gives it.
  - No statement here grants `s3:PutObjectTagging`, and none uses a wildcard such as `s3:PutObject*`, which would also match `s3:PutObjectTagging` (Q2).
  - After each sub-task, `uv run pytest tests/test_aws_setup.py` passes, so a cfn-lint finding shows up next to the change that caused it.
  - _Requirements: 1.3, 1.5, 1.6, 1.7, 1.8, 1.9, 1.10, 1.11, 1.12, 1.13, 2.6, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 5.1, 5.2, 5.3, 5.4, 5.5, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8, 9.1, 9.2, 9.3, 9.4, 10.1, 10.2, 10.3, 11.1, 11.2, 11.3, 11.4, 11.5, 11.6, 11.7, 17.1, 17.3, 17.6, 17.7, 17.8_

  - [ ] 2.1 Add the job role, its instance profile and its policy
    - Add `JobPolicy`, `JobRole` and `JobInstanceProfile` as the design's "The Job_Role and the Job_Profile" section gives them.
    - `JobPolicy` holds the five statements of the design's table. `ListJobAndWorkFolders` allows `s3:ListBucket` on the bucket only with the `StringLike` condition `s3:prefix`: `["jobs/*", "work/*"]`. `ReadAndUploadJobFiles` allows `s3:GetObject`, `s3:PutObject` and `s3:AbortMultipartUpload` on `${JobBucket.Arn}/jobs/*` and `${JobBucket.Arn}/work/*`. So a Job_Machine lists, reads and uploads under both prefixes.
    - Of the three Session Manager statements, only `SessionManagerAgentChannels` has `"*"` as its resource, with the `aws:RequestedRegion` condition, because its three actions take no resource.
    - A Job_Machine uploads each Work_File with no tag, so `JobPolicy` has no `s3:PutObjectTagging`. There's no `s3:DeleteObject` (A4) and no AWS managed policy, so no `AmazonSSMManagedInstanceCore`.
    - `JobRole` has a `Description`, the trust policy that lets only `ec2.amazonaws.com` assume it, `ManagedPolicyArns: [!Ref JobPolicy]` and the Project_Tag. `JobInstanceProfile` has only `Roles: [!Ref JobRole]`.
    - Add the outputs `JobRoleArn` and `JobInstanceProfileArn`, each with the design's description and no `Export`.
    - _Requirements: 1.3, 1.5, 1.6, 1.7, 1.8, 1.9, 1.10, 1.11, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 7.7, 10.1, 10.2, 10.3_

  - [ ] 2.2 Add the read policy
    - Add `ReadPolicy` with the seven statements of the design's "The Read_Policy" table.
    - Only `DescribeEc2InThisRegion` has `"*"` as its resource, with the `aws:RequestedRegion` condition. `ReadThisStack` takes `!Ref AWS::StackId`, so a later stack with the same name isn't covered.
    - `ListJobBucket` has no prefix condition, and `ReadJobFiles` covers `${JobBucket.Arn}/*`, so the Read_Policy lists the whole bucket and reads every file in it, under `jobs/` and `work/` alike.
    - It has no `s3:ListAllMyBuckets`, no IAM or `sts:` action, and nothing that starts, changes or deletes.
    - _Requirements: 1.3, 1.5, 1.6, 1.8, 1.9, 1.10, 1.11, 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8_

  - [ ] 2.3 Add the runner's launch policy
    - Add `RunnerLaunchPolicy` with the eight statements of the design's launch table, which build A3. Each kind of resource in a launch gets its own statement.
    - `LaunchJobMachine` lists the 8 Allowed_Types in the glossary's order, needs `ec2:MetadataHttpTokens` to be `required` and `aws:RequestTag/project` to be `voxelis-kiroween`, and uses `ArnEquals` on `ec2:InstanceProfile` with `!GetAtt JobInstanceProfile.Arn`.
    - `LaunchInDefaultVpc` uses `ArnEquals` on `ec2:Vpc` with the Default_VPC's ARN, built from `${JobSecurityGroup.VpcId}`. `LaunchWithJobSecurityGroup` names only the Job_Security_Group, built from `${JobSecurityGroup.GroupId}`.
    - `TagOnlyWhenCreating` limits `ec2:CreateTags` with `ec2:CreateAction` to `RunInstances` and `CreateImage`. `PassJobRole` passes only the Job_Role, and only to `ec2.amazonaws.com`.
    - No statement covers `key-pair` or `launch-template` resources, and there's no `ec2:StartInstances`.
    - _Requirements: 1.3, 1.5, 1.6, 1.8, 1.9, 1.10, 1.11, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8_

  - [ ] 2.4 Add the runner's manage policy and the runner role
    - Add `RunnerManagePolicy` with the eight statements of the design's manage table. It has no action that shares, publishes, copies or exports an image or a snapshot, and no `s3:DeleteObject` (A4).
    - `UploadJobFiles` allows only `s3:PutObject` and `s3:AbortMultipartUpload`, only on `${JobBucket.Arn}/jobs/*`, for the job's video and job record in `input/`. So the Runner_Role uploads only under `jobs/` and never under `work/`, and it has no `s3:PutObjectTagging`. It still lists and reads the whole bucket through the Read_Policy.
    - `StartShellOnJobMachine` uses the `ssm:resourceTag/project` condition key, and `StartShellDocument` grants `SSM-SessionManagerRunShell` in a statement of its own. In `OwnShellsOnly`, write `${aws:userid}` as a plain string, never inside a `!Sub`, where it would have to be `${!aws:userid}`.
    - Add `RunnerRole` with a `Description`, the trust policy whose principal is `!Sub "arn:${AWS::Partition}:iam::${AWS::AccountId}:root"` with no MFA condition, `ManagedPolicyArns: [!Ref ReadPolicy, !Ref RunnerLaunchPolicy, !Ref RunnerManagePolicy]`, the Project_Tag and the default `MaxSessionDuration`.
    - Add the output `RunnerRoleArn`, with the design's description and no `Export`.
    - _Requirements: 1.3, 1.5, 1.6, 1.7, 1.8, 1.9, 1.10, 1.11, 5.1, 5.2, 5.3, 5.4, 5.5, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 10.1, 10.2, 10.3_

  - [ ] 2.5 Add the project group and the last output
    - Add `AssumeRunnerPolicy` with its one statement, `AssumeRunnerRole`, which allows `sts:AssumeRole` on `!GetAtt RunnerRole.Arn` and nothing else, with no condition.
    - Add `ProjectGroup` with `ManagedPolicyArns: [!Ref ReadPolicy, !Ref AssumeRunnerPolicy]` and nothing more: no users and no self-service IAM permissions. The template makes no IAM user.
    - Add the output `GroupName` (`!Ref ProjectGroup`). The outputs are then exactly the seven of Req 10.1, each with a description and none with an `Export`.
    - _Requirements: 1.5, 1.6, 1.9, 9.1, 9.2, 9.3, 9.4, 10.1, 10.2, 10.3_

  - [ ] 2.6 Run the Compliance_Check on the whole template, make the Template_Test pass, and check the template's size
    - Run the Compliance_Check on the whole template, task 1's resources included, the same way as in 1.6.
    - The design expects one more Unfit_Rule: `IAM_POLICYDOCUMENT_NO_WILDCARD_RESOURCE` on `JobPolicy` and on `ReadPolicy`, each with the reason in its "Suppressions" table. `IAM_NO_INLINE_POLICY_CHECK` should pass already, because every permission is a managed policy.
    - If the wildcard-resource rule fails on `RunnerLaunchPolicy`, `RunnerManagePolicy` or `AssumeRunnerPolicy`, for example because of ARNs that end in `/*`, it isn't an Unfit_Rule there. Ask the user before suppressing it.
    - Keep `"*"` to the one statement that each reason names, because a Suppression covers its whole resource.
    - Run `uv run pytest tests/test_aws_setup.py` and fix each finding in the template, as in 1.6.
    - Search the whole template for `PutObjectTagging`, `PutObject*` and `TagFilters`. None of them appears (Q2).
    - Check the whole template's size by hand with `(Get-Item cloud/aws-setup.yaml).Length`, as in 1.6. It must be at most 51,200 bytes, and the design expects about 20 KB. If any fix in this sub-task changes the template, run the other checks again, so the template that's committed has passed all of them.
    - _Requirements: 1.12, 1.13, 2.6, 3.6, 7.7, 11.1, 11.2, 11.3, 11.4, 11.5, 11.6, 11.7, 17.1, 17.3_

  - [ ] 2.7 Run the tests and commit
    - Search `cloud/aws-setup.yaml` for region names such as `us-east-1`, for 12-digit numbers and for the team members' names. It should hold none.
    - Run `uv run pytest`. Every test passes, `tests/test_guard.py` included.
    - If anything in this sub-task changes `cloud/aws-setup.yaml`, run 2.6's checks on it again before committing.
    - Check that the template and the commit message name no Other_Resource, and that nothing in this task called AWS or touched a stack.
    - On `feat/aws-setup`: `git add cloud/aws-setup.yaml`, then `git commit -m "AWS setup: the roles, the policies and the project group" -- cloud/aws-setup.yaml`.
    - _Requirements: 1.9, 17.6, 17.7, 17.8_

- [ ] 3. The stack's name in config, the quota line and the region rule
  - Files: `config.toml`, `tests/test_config.py`, `.kiro/steering/tech.md` and `.kiro/steering/rules.md`, all changed.
  - _Requirements: 10.4, 16.1, 16.13, 17.4, 17.6, 17.7, 17.8_

  - [ ] 3.1 Add `stack_name` to `[aws]` in `config.toml`
    - After `spend_variable`, add the design's comment line and `stack_name = "voxelis-kiroween"`. The other `[aws]` keys and every other section stay as they are.
    - _Requirements: 10.4_

  - [ ] 3.2 Add `stack_name` to `UNCHANGED_SECTIONS` in `tests/test_config.py`
    - `UNCHANGED_SECTIONS["aws"]` gets `"stack_name": "voxelis-kiroween"`, so `test_the_real_git_aws_paid_and_devlog_sections_are_unchanged` passes with the new key. Change nothing else in the file. This is the only edit to an existing test.
    - _Requirements: 17.4_

  - [ ] 3.3 Change the quota line in `.kiro/steering/tech.md`
    - Replace the line that starts "The GPU quota is per account and region" with the design's text. It says our quota allows 32 vCPUs of G instances at a time in us-east-1, one job machine runs at a time, fast and default jobs run on a g5.8xlarge with 32 vCPUs on 16 cores and one NVIDIA A10G, the next job waits for it, and a new AWS account starts at 0 and has to ask AWS for more.
    - _Requirements: 16.1_

  - [ ] 3.4 Add the two region lines to `.kiro/steering/rules.md`
    - Under `## AWS access`, right after the line that starts "In Python on our laptops, always name the profile with `boto3.Session(profile_name=...)`", add the two lines in Req 16.13's second note, word for word. The design's "Steering, README and licence edits" section gives the same two lines. The first says aws commands also pass `--region us-east-1`, because an `AWS_REGION` or `AWS_DEFAULT_REGION` environment variable beats the profile's region, even with `--profile`. The second says Python also passes `region_name` from `[aws].region` in `config.toml` to `boto3.Session`.
    - The two lines replace no line. Every other line of `## AWS access` stays as it is, and so do task 1's licence lines.
    - _Requirements: 16.13_

  - [ ] 3.5 Run the tests and commit
    - Run `uv run pytest`. Every test passes, `tests/test_config.py` and `tests/test_guard.py` included.
    - Check that the commit message names no Other_Resource, and that nothing in this task called AWS.
    - On `feat/aws-setup`: `git add config.toml tests/test_config.py .kiro/steering/tech.md .kiro/steering/rules.md`, then `git commit -m "AWS setup: the stack's name in config, the GPU quota line and the region rule" -- config.toml tests/test_config.py .kiro/steering/tech.md .kiro/steering/rules.md`.
    - _Requirements: 17.4, 17.6, 17.7, 17.8_

- [ ] 4. The setup guide
  - Files: `docs/aws-setup.md` (new), `README.md`, `docs/third-party-licences.md` and `.kiro/steering/structure.md` (changed).
  - The whole guide follows the design's "Rules for the whole guide": plain words for someone running their own copy; no person, team member, account id or region, with `<your-region>` and the `config.toml` values instead; no Other_Resource; every aws command on one line; and no account-wide setting.
  - Every aws command in the guide passes `--region <your-region>`, the `aws configure` lines included. The commands run on a laptop also pass `--profile`. The commands run in CloudShell are bash, with `--region` and no `--profile`, because CloudShell runs as the console's login and has no named profiles.
  - No command prints an account id, a key or an ARN. Each lookup prints only what it needs with `--query ... --output text`, and `create-stack` and `create-change-set` end with `> /dev/null`, because their output is an ARN.
  - _Requirements: 10.5, 12.1, 12.2, 12.3, 12.4, 12.5, 12.6, 12.7, 12.8, 12.9, 12.10, 12.11, 12.12, 12.13, 12.14, 13.1, 13.2, 13.3, 13.4, 13.5, 13.6, 13.7, 13.8, 13.9, 13.10, 13.11, 13.12, 13.13, 13.14, 13.15, 14.1, 14.2, 14.3, 14.4, 14.5, 14.6, 14.7, 15.1, 15.2, 15.3, 15.4, 16.2, 16.5, 16.9, 16.12, 17.6, 17.7, 17.8_

  - [ ] 4.1 Write `docs/aws-setup.md` up to Step 6
    - Write `# AWS setup`, `## Who does what`, `## What the stack makes` and Steps 1 to 6, with the headings, the order and the content of the design's "The Setup_Guide" table, and with the design's CloudShell commands for Step 4.
    - `## Who does what`: only the Account_Owner uses the console and CloudShell, with the new admin login and MFA, and with the root user only where Steps 1, 2 and 6 say. Only they deploy or update the Stack. Everyone else uses only the AWS CLI on their own laptop. Kiro never deploys or updates the Stack, and on a laptop the guard hook blocks the commands that create or update stacks when Kiro runs them. Laptop commands name the profile with `--profile`, never `AWS_PROFILE`, because AWS keys in environment variables beat `AWS_PROFILE`. Every command names the region with `--region`, because an `AWS_REGION` or `AWS_DEFAULT_REGION` variable beats the profile's region, even with `--profile`. The CloudShell commands are the one exception to `--profile`. The section ends with the placeholders and the `config.toml` values that the guide uses.
    - `## What the stack makes`: the resources in plain words and the job bucket's layout. The copies of a job's `input/` and `out/` folders go under `jobs/<job id>/input/` and `jobs/<job id>/out/` and stay until someone deletes them. The copy of its `work/` folder goes under `work/<job id>/`, and the bucket deletes each file there 60 days after its upload. The bucket also cancels any upload that's still unfinished 7 days after it started, anywhere in the bucket.
    - Step 1 moves the account to the paid plan, signed in as the root user, because the admin login only comes in Step 2. It says why: an account on AWS's free plan can't use every service, and AWS closes it when its credits run out or after six months. It leaves the setting for IAM access to billing as it is (A2).
    - Step 2 always makes a new admin login with the root user, even when the account already has an IAM user with administrator access, so the setup depends on no other user in the account. The admin login gets a name that starts with the stack's name, like `voxelis-kiroween-admin`, a console password, AWS's managed `AdministratorAccess` policy, the Project_Tag and no access keys, and then an MFA device, once the Account_Owner signs in as it. Its sign-in URL holds the account id, so it stays out of chats. From then on it replaces the root user, except for the budget in Step 6 when the Billing console refuses it.
    - Step 3 checks in the VPC console that `<your-region>` has a default VPC, and makes one with Actions, Create default VPC, when it's missing.
    - Step 4 deploys the Stack once from CloudShell, opened in `<your-region>` with the admin login. The Account_Owner first removes any older copy with `rm -f aws-setup.yaml`, then uploads `cloud/aws-setup.yaml` from their copy of the repo with Actions, Upload file, so it lands in CloudShell's home folder. Then they run `aws cloudformation create-stack --stack-name voxelis-kiroween --template-body file://aws-setup.yaml --capabilities CAPABILITY_IAM --tags Key=project,Value=voxelis-kiroween --region <your-region> > /dev/null`, the wait command and the status check, and look for `CREATE_COMPLETE`.
    - Step 4 also says that if CloudShell isn't available in `<your-region>`, the console opens it in the nearest region that has it, and the commands still work because each passes `--region <your-region>`. It says why the template goes through CloudShell and never through the CloudFormation console's upload of a template file (Q1), in words that mention no other project, and that `--template-body` takes at most 51,200 bytes. It says why the Project_Tag also goes on the Stack as a stack tag. After `ROLLBACK_COMPLETE`, the Account_Owner reads the first failed event, fixes the cause, deletes the failed stack with the design's delete commands and creates it again. Step 4 says that a failed first create leaves no job bucket behind, because CloudFormation deletes the still-empty bucket when the first create rolls back, so there's nothing to clean up by hand. Only the Account_Owner deploys the Stack, never Kiro.
    - Step 5 makes one IAM user per person, the Account_Owner included, with no console access, the Project_Tag and a name that starts with the stack's name. It adds the user to the group that the `GroupName` output names and makes its access key. The keys go to that person privately and only into their own AWS CLI files. Adding someone to the group is the Account_Owner's OK for them to run jobs, and no profile uses an admin's keys.
    - Step 6 makes a monthly cost budget for the whole account (A1), with an email alert and the Project_Tag, with the admin login, or with the root user if the Billing console refuses the admin login (A2). It says the alert only sends an email and stops nothing, and it leaves the setting for IAM access to billing as it is.
    - _Requirements: 12.1, 12.2, 12.3, 12.4, 12.5, 12.6, 12.7, 12.8, 12.9, 12.10, 12.14, 13.1, 13.2, 13.3, 13.4, 13.5, 13.6, 13.11, 13.14, 15.1, 15.2, 15.3, 15.4_

  - [ ] 4.2 Write Steps 7 and 8 and the rest of `docs/aws-setup.md`
    - Write Step 7, Step 8, `## Opening a shell on a job machine`, `## Stack outputs`, `## Updating the stack`, `## Removing the setup` and `## Another region or another name`, as the design's table gives them.
    - Use the design's laptop commands as written, each with `--profile` and `--region <your-region>`: `aws configure --profile kiroween-read --region <your-region>`, the runner profile's `role_arn` line for PowerShell and for bash, the `source_profile` and `region` lines, the two profile checks, the two quota commands and the `start-session` command. Apart from the `role_arn` line, each works the same in PowerShell and in bash.
    - Step 7 is for each person, in their own terminal and not through Kiro: install the AWS CLI v2, make the read profile from their keys, make the runner profile and run the two checks. It has one line telling people to answer `n` if `aws configure` ends by offering to configure AWS skills and the AWS MCP server for their AI coding agent, because Kiro gets AWS access only through the two profiles (Q3). Its notes say that the runner role's ARN holds the account id and goes only into the person's own AWS CLI config file, that AWS error messages can hold the account id, and that `aws sts get-caller-identity` is no check because it prints the account id. They also say the read profile's keys can assume the runner role, so the line between the two profiles is Kiro's Approval_Prompt, the Spend_Guard and our rules, not IAM.
    - Step 8 names the quota and its code, gives each size's vCPUs and the design's Allowed_Types table, says a new account starts at 0, says how to ask for 32 vCPUs in `<your-region>`, and says the runner role refuses every other instance type.
    - `## Opening a shell on a job machine` has people install the Session Manager plugin and open shells with their runner profile, only on machines with the Project_Tag, and says Kiro doesn't open shells.
    - `## Stack outputs` holds the design's Stack_Outputs table and says the ARNs stay out of chats and logs.
    - `## Updating the stack` follows Q1 and A7. In CloudShell, opened with the admin login, the Account_Owner removes the old copy with `rm -f aws-setup.yaml` and uploads the new file, makes a change set, waits for it, reviews it, and only then executes it and waits for the update. They never use `aws cloudformation update-stack`, which skips the review. The review looks for `JobBucket` with a replacement of `True` or `Conditional`. Kiro never runs these commands.
    - The change set is made with `aws cloudformation create-change-set --stack-name voxelis-kiroween --change-set-name <name> --template-body file://aws-setup.yaml --capabilities CAPABILITY_IAM --tags Key=project,Value=voxelis-kiroween --region <your-region> > /dev/null`, as in the design. Its `--tags` keeps the stack's tag whatever CloudFormation does with a change set that leaves `--tags` out. The wait, `describe-change-set`, `execute-change-set`, `wait stack-update-complete` and `delete-change-set` commands are as in the design.
    - `## Removing the setup` first has the Account_Owner terminate every job machine that's still running, then goes in this order. 1: take every person's user out of the project group, because CloudFormation can't delete a group that still has members. 2: note the bucket's name from the `BucketName` output, then delete the Stack with the delete commands, in CloudShell. 3: empty the job bucket in the S3 console and delete it by hand, because deleting the Stack keeps the bucket and its files (`DeletionPolicy: RetainExceptOnCreate`). It also names what was never part of the Stack and stays: the machine images and their snapshots, the IAM users, the admin login and the budget.
    - `## Another region or another name` names `[aws].region` and `[aws].stack_name` as the two values a copy changes.
    - Then read the whole guide against Requirement 15 and Req 13.10, 13.11 and 13.14. Each aws command passes `--region <your-region>`, each laptop command also passes `--profile`, and each CloudShell command passes no `--profile`. No command prints an account id, a key or an ARN. Each command that calls AWS reaches only the Stack, its change sets or the G_Quota, by name, apart from the `start-session` command, which opens a shell on one job machine by its id.
    - _Requirements: 10.5, 12.1, 12.8, 12.11, 12.12, 12.13, 13.7, 13.8, 13.9, 13.10, 13.11, 13.12, 13.13, 13.14, 13.15, 14.1, 14.2, 14.3, 14.4, 14.5, 14.6, 14.7, 15.1, 15.2, 15.3, 15.4_

  - [ ] 4.3 Add the AWS setup section to `README.md`
    - After "Run the tests", add the design's `## AWS setup` section, which links `docs/aws-setup.md`.
    - _Requirements: 16.5_

  - [ ] 4.4 Add the Session Manager plugin's row to `docs/third-party-licences.md`
    - Add the design's "AWS Session Manager plugin" row after the `awslabs.aws-iac-mcp-server` row: Apache-2.0, a separate command line tool that people run with the AWS CLI to open a shell on a job machine, never vendored, linked or shipped, and Installed "Not yet". It comes in this task, with the guide that tells people to install it.
    - _Requirements: 16.9, 16.12_

  - [ ] 4.5 Change the `docs/` line in `.kiro/steering/structure.md`
    - Replace it with the design's text, which adds `aws-setup.md`, the guide to setting up AWS, and goes in with the guide. It stays the only line that starts with ``- `docs/`:`` and still names `pipeline.md`, so `tests/test_docs.py` still passes.
    - _Requirements: 16.2_

  - [ ] 4.6 Run the tests and commit
    - Search `docs/aws-setup.md` for region names such as `us-east-1`, for 12-digit numbers and for the team members' names. It should hold none.
    - Run `uv run pytest`. Every test passes, `tests/test_docs.py` and `tests/test_guard.py` included.
    - Check that the files and the commit message name no Other_Resource, and that nothing in this task ran an `aws` command or touched a stack.
    - On `feat/aws-setup`: `git add docs/aws-setup.md README.md docs/third-party-licences.md .kiro/steering/structure.md`, then `git commit -m "AWS setup: the setup guide" -- docs/aws-setup.md README.md docs/third-party-licences.md .kiro/steering/structure.md`.
    - _Requirements: 15.1, 16.12, 17.6, 17.7, 17.8_

## Notes

- The Template_Test (1.5) is the only new test, and it's required. No task or sub-task is marked optional with `*`. There are no property tests, by the user's choice for this spec (Req 17.5), and the design has no Correctness Properties.
- The last sub-task of each task is its checkpoint: the whole suite, then the commit. So the plan has no separate checkpoint tasks and stays at the design's 4 tasks, within the user's limit of 5.
- Every acceptance criterion in `requirements.md` is named by at least one sub-task. Requirements 17.6 to 17.8 hold for the whole build, so each commit sub-task names them.
- The sub-tasks build what the review accepted and decided as the design gives it: A5 in 1.3, A6 in 1.5, A4 in 2.1 and 2.4, A3 in 2.3, A1 and A2 in 4.1, and A7 in 4.2. Q1 is in 1.3, 1.6 and 2.6 (the header and the size check), 4.1 (Step 4) and 4.2 (updating). Q2 is in 1.3 (the lifecycle rules), 2.1, 2.4 and 2.6 (no tagging) and 4.1 (the bucket's layout). Q3 is in 4.2 (Step 7).
- The graph runs the tasks in order, one at a time, so each task's test run and commit see only finished work. Inside a task, sub-tasks that write different files share a wave. The sub-tasks of task 2 each get their own wave, and so do the two parts of the guide, because they write the same file. `rules.md`, `structure.md` and `docs/third-party-licences.md` each change in two tasks, so their edits are always in different waves.

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "1.3", "1.4"] },
    { "id": 1, "tasks": ["1.2", "1.5"] },
    { "id": 2, "tasks": ["1.6"] },
    { "id": 3, "tasks": ["1.7"] },
    { "id": 4, "tasks": ["2.1"] },
    { "id": 5, "tasks": ["2.2"] },
    { "id": 6, "tasks": ["2.3"] },
    { "id": 7, "tasks": ["2.4"] },
    { "id": 8, "tasks": ["2.5"] },
    { "id": 9, "tasks": ["2.6"] },
    { "id": 10, "tasks": ["2.7"] },
    { "id": 11, "tasks": ["3.1", "3.3", "3.4"] },
    { "id": 12, "tasks": ["3.2"] },
    { "id": 13, "tasks": ["3.5"] },
    { "id": 14, "tasks": ["4.1", "4.3", "4.4", "4.5"] },
    { "id": 15, "tasks": ["4.2"] },
    { "id": 16, "tasks": ["4.6"] }
  ]
}
```
