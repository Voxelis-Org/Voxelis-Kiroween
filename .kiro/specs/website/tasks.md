# Implementation Plan: website

## Overview

This plan builds the Website in `app/` in the 10 tasks of the design's "Build order", in that order. The seam comes first: the `[website]` Config and the App_Project, then the typed table values and the new AWS operations with their fake, then the Start_Core split in `cloud/start.py`. The pure parts follow, each built and tested on its own: the Box_Reader, the Accounts, Sessions and sign-in limit, and the Website_Limits. Then the request handler and the pages, then uploads and starts, which join everything, then the Stack_Template, and last the Deploy_Command and the docs. The web parts are Python 3.11 standard library only, AWS goes through `cloud/aws.py`, the Page_Script is plain JavaScript with no build step, and the template is CloudFormation YAML. The tests use pytest and Hypothesis, which the root environment already holds, together with boto3, botocore and cfn-lint.

Some files grow over several tasks, so each piece comes with the first task that uses it:

- `tests/website_helpers.py`: task 1 writes `website_config(**overrides)`, task 4 adds `mp4(boxes)` and the sparse reader, task 5 adds `make_deps(fake, **overrides)` and `sign_in_as(deps, name, now)`, and task 7 adds `event(method, path, *, cookies, headers, body, source_ip)` and makes `make_deps` load the real `app/static/` files.
- `tests/fake_aws.py` and `tests/fake_table.py`: task 2 adds every new operation to `FakeAws` and writes the whole fake table evaluator. Later tasks only seed and read them.
- `tests/test_website_imports.py`: task 1 writes the import and number scans, with 0, 1 and the HTTP statuses on the allow list. Each later task that adds a protocol constant to `app/` adds it to the allow list with a comment naming its source: task 4 the Box header sizes, task 5 the scrypt memory formula's numbers and the time units, task 7 anything the handler adds, and task 10 the zip's fixed date and mode and CloudFormation's 51,200-byte body limit.
- `app/web.py`: task 1 writes the data classes `Request`, `Response`, `Random`, `SecretsRandom` and `Deps`. Task 7 adds `handle`, `stack_for`, the route table with the open, page and status routes, the cookies, the headers and the Content-Security-Policy. Task 8 adds the upload, precheck, start and download routes to the table.
- `app/static/app.js`: task 7 writes the file with the status refresh. Task 8 adds the Start_Precheck, the part upload with its retries and progress, and the finish.
- `cloud/aws.py`: task 2 adds every new method, the builders, `s3_origin`, `aws_setting`, the 403 fallback and `QuotaRefused`. No later task changes it.
- `cloud/aws-setup.yaml`: task 9 alone changes it.
- `docs/third-party-licences.md`: task 1 adds a row only for a package in `app/uv.lock` that the Licence_List doesn't list yet. Task 10 updates the "How we use it" cells of boto3, botocore, s3transfer, jmespath, python-dateutil, six, urllib3 and Python, and adds a row for pip if it has none.

## Working rules for every task

- Work on `feat/website`, never on `main`.
- Change no file under `splatpipe/`, and leave the root `pyproject.toml` and the root `uv.lock` byte for byte as they are. Task 1's test checks both files' hashes (Req 1.2, 1.5, 18.13). The root `pyproject.toml` already sets `pythonpath = ["."]` for pytest, so `import app` works in the tests with no change to it.
- In the code, boto3 and botocore are imported only in `cloud/aws.py`. No file under `app/` imports either, and `cloud/table.py` imports only the standard library (Req 1.4). Of the new test files, only `tests/test_website_aws.py` imports botocore, for its `Stubber`.
- `app/` imports only the standard library, `app`, `cloud` and the top-level modules of `splatpipe` (`splatpipe.config`, `splatpipe.jobs`, `splatpipe.make`, `splatpipe.run`, `splatpipe.errors`, `splatpipe.jsonio`), never a Step package and never a name that starts with `_` (Req 1.5).
- The Start_Core stays in `cloud/start.py`. Every call to `check_spend` and `make_job` looks the name up in `cloud.start`'s module globals at call time, so the cloud-jobs tests' monkeypatches reach both callers. Change no existing `tests/test_cloud_*.py` and no pipeline-core test. All of them pass unchanged (Req 9.2, 18.13).
- Every number and setting comes from `config.toml`, with a comment on the line directly above each key, except what the design keeps in code on purpose: the template constants in `cloud/aws-setup.yaml` (Req 2.9), the AWS floors in `app/config.py` with their source, the Account_Name_Pattern, `UPLOAD_ID`, the cookie names, `VIDEO_FILES`, `FOOTAGE_LICENCES` and the protocol constants on the allow list of `tests/test_website_imports.py` (Req 2.6).
- Use the design's names for every module, function, data class, route, config key and test file. Where this plan names something the design doesn't, the sub-task says so.

### Tests

- Tests run on the CPU with no network, no GPU and no AWS, against `FakeAws` and the fake table (Req 18.1).
- `tests/conftest.py`'s autouse `temp_config` writes a Config with no `[aws]`, `[cloud]` or `[website]`. So the website code takes its configs as parameters, and a test that needs the real `config.toml` reads it by path as `REPO_ROOT / "config.toml"`, the way `make_deps` does.
- No test writes into the real `data/`. Temporary files go in `tmp_path` or a `tempfile.TemporaryDirectory()`.
- No test and no task sets `APPROVED_SPEND` in the process environment, calls AWS, runs a raw `aws` command, runs a CloudShell_Command for real, deploys or starts anything paid (Req 18.12). The Website builds its own environment dict for `check_spend`, and the tests of the Deploy_Command pass a dict.
- The Deploy_Command, the Account_Command and the Showcase_Command are built and tested against `FakeAws`, a temporary git repo and an environment dict only. Only the Account_Owner runs them, in CloudShell, and Kiro never does.
- Keep the suite fast, as the design's "Keeping the tests fast" says: tiny scrypt settings through `website_config(scrypt_n=2, scrypt_r=1, scrypt_p=1)`, a sparse reader for Boxes that claim gigabytes, `FakeAws.object_sizes` for large sizes, and `handle` called with plain dicts, never a server.
- Property tests use Hypothesis with at least 100 examples (the default) and `@settings(deadline=None)`. Each docstring says `Feature: website, Property N: <title>` and `**Validates: Requirements ...**`.
- Subprocess and CLI tests pass `timeout=60`, set `PYTHONIOENCODING=utf-8` in the child's environment and read its output as UTF-8.
- Every test sub-task is required, unit tests and property tests alike. None is optional (Req 18.11).

### Licences and files

- Whoever adds a dependency, tool, model weight or dataset adds its `docs/third-party-licences.md` row in the same task and commit. If a package's licence is copyleft, non-commercial or unclear, stop and ask the user before committing.
- Nothing over 1 MB goes in git without asking first. Check `app/uv.lock`'s size before committing it.

### The checkpoint, the last sub-task of each task

- Run `uv run pytest` and fix what fails until every test passes, the cloud-jobs, pipeline-core and clean tests included.
- Stage the task's files by name with `git add <file> ...`, never with `git add -A`, `git add .` or a folder.
- Commit only those files with `git commit -m "<title>" -m "<body>" -- <file> ...`, so nothing else that's staged goes in. The title follows the repo's style, as `Website: <what>`, and the body says what the commit adds and why. Keep backticks and `$` out of both, because PowerShell reads them inside double quotes.
- Leave `docs/devlog.md` out: the devlog hook adds and commits its own line after each task.

### Hooks

- If a hook blocks something it shouldn't, say so and fix the hook. Never work around a hook or turn one off without the user's OK.

## Tasks

- [ ] 1. The `[website]` Config and the App_Project
  - Files: `app/__init__.py`, `app/pyproject.toml`, `app/uv.lock`, `app/config.py`, `app/errors.py`, `app/web.py`, `tests/website_helpers.py`, `tests/test_website_config.py`, `tests/test_website_project.py` and `tests/test_website_imports.py` (new), `config.toml` (changed), and `docs/third-party-licences.md` (changed only if `app/uv.lock` holds a package it doesn't list).
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.10, 2.11, 4.11, 18.11, 18.12, 18.13_

  - [ ] 1.1 Record the root files' hashes, then add `[website]` to `config.toml`
    - Before any other change, record the SHA-256 of the committed root `pyproject.toml` and `uv.lock`, each hashed after turning `\r\n` into `\n`, so a Windows checkout with `core.autocrlf` gives the same hash. Task 1.7 writes them into its test as constants.
    - Open the [OWASP Password Storage Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Password_Storage_Cheat_Sheet.html) and check that scrypt with N=2^15, r=8 and p=3 is still one of its listed settings. If it isn't, stop and ask the user before committing any scrypt numbers.
    - Add `[website]` after `[clean]` and its three Preset tables, with exactly the keys, values and comments of the design's "Sections added to `config.toml`", each comment on the line directly above its key. `stop_starting_at` stays text, because the Config_Loader refuses TOML dates in a Pipeline_Section.
    - Leave every other section exactly as it is, `[git]`, `[aws]`, `[paid]` and `[devlog]` included.
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.10_

  - [ ] 1.2 Write `app/config.py`
    - `WebsiteConfig` with the design's fields, `from_mapping(data)`, `load_website_config(path=None)` and `dollars_to_cents(value)`.
    - Define `ScryptSettings` (`n`, `r`, `p`, `dklen`, `salt_bytes`) here, because `WebsiteConfig.scrypt` needs it in this task. Task 5's `app/accounts.py` imports it from here, so `app.accounts.ScryptSettings` is the same class. The design places it in `app/accounts.py`, and this move only avoids an import cycle.
    - `from_mapping` refuses with a `PipelineError` that names the key and says what it must be: a missing key, an unknown key, a `bool` where a number belongs, a number below its floor, a Stop_Time not written as `YYYY-MM-DDTHH:MM:SSZ`, and a `deploy_folder` that isn't an absolute POSIX path strictly inside `/tmp` with no `..` part.
    - The floors: every whole-number key is an `int` of at least 1, `daily_cost_usd` and `total_cost_usd` are an `int` or `float` above 0, and `scrypt_n` is a power of 2 of at least 2. The AWS floors stay in code with their source ([S3 quick facts](https://docs.aws.amazon.com/AmazonS3/latest/userguide/qfacts.html)): `upload_part_bytes` and `copy_part_bytes` of at least 5 MiB, and both link lifetimes of at most 7 days.
    - `load_website_config` validates the whole file with `splatpipe.config.load_config(path)` first, which already refuses NaN, infinity and TOML dates in `[website]`, then reads `[website]` with `from_mapping`. Because `from_mapping` sees only `[website]`, `load_website_config` then refuses an `upload_part_bytes` that would need more than 10,000 parts for `Config.jobs()`'s `max_video_bytes`, naming `upload_part_bytes`.
    - `stop_starting_at` becomes an aware UTC `datetime`. `daily_cost_cents` and `total_cost_cents` come from the dollars through `dollars_to_cents`, which parses the `.2f` text of the value into whole cents, as the Start_Command's quote prints it.
    - _Requirements: 2.1, 2.2, 2.3, 2.6, 2.7, 2.8_

  - [ ] 1.3 Write `app/__init__.py`, `app/errors.py` and the data classes in `app/web.py`
    - `app/__init__.py` is empty.
    - `app/errors.py`: `PublicRefusal(Exception)` with `status`, `kind` and `message`, made as `PublicRefusal(status, kind, message)`, with `str()` giving the message.
    - `app/web.py`: `Request`, `Response`, the `Random` protocol and `Deps`, with the design's fields. Add `SecretsRandom`, a `Random` that uses `secrets.token_bytes` and `secrets.choice`, for Lambda and CloudShell. The design doesn't name it.
    - `Deps.baked` is annotated `BakedCommit` through a `TYPE_CHECKING` import with `from __future__ import annotations`, because `app/baked.py` comes in task 3. `handle` and `stack_for` come in task 7.
    - _Requirements: 1.1, 1.3_

  - [ ] 1.4 Write the App_Project and lock it
    - `app/pyproject.toml`: `name = "voxelis-website"` (the design doesn't name it), `version = "0.1.0"`, `requires-python = ">=3.11,<3.12"` and `dependencies = ["boto3"]`, with no build system, so uv treats it as a virtual project.
    - Run `uv lock --project app`. It writes only `app/uv.lock` and may use the network, as a dev action like `uv add`. The root has no `[tool.uv.workspace]`, so `app/` isn't a workspace member.
    - Afterwards, check that the root `pyproject.toml` and `uv.lock` are byte for byte unchanged with `git status --porcelain -- pyproject.toml uv.lock`, which must print nothing, and that `app/uv.lock` is under 1 MB.
    - For each package in `app/uv.lock` that `docs/third-party-licences.md` doesn't list, add its row in this commit, with its licence and how we use it. Expect boto3, botocore, s3transfer, jmespath, python-dateutil, six and urllib3, which the root project already pulls in. If a new row's licence is copyleft, non-commercial or unclear, stop and ask the user.
    - _Requirements: 1.1, 1.2_

  - [ ] 1.5 Write `website_config(**overrides)` in `tests/website_helpers.py`
    - Read the real `[website]` from `REPO_ROOT / "config.toml"` with `tomllib`, apply the overrides by key name, such as `scrypt_n=2`, and build the `WebsiteConfig` with `WebsiteConfig.from_mapping`, the way the cloud-jobs tests' `cloud_config(**overrides)` builds a `CloudConfig`.
    - _Requirements: 18.1_

  - [ ] 1.6 Write the unit tests for the Config in `tests/test_website_config.py`
    - `load_website_config(REPO_ROOT / "config.toml")` gives the settled values: the Stop_Time as the aware UTC instant 2027-02-04 08:00:00, `daily_jobs` 10, `daily_cost_cents` 5000, `total_cost_cents` 50000, `upload_link_seconds` 900, `download_link_seconds` 300, `signin_max_failures` 5, `signin_window_minutes` 15, `signin_wait_minutes` 15 and `session_days` 7 (Req 2.2 to 2.5, 2.11).
    - `[website]`'s keys share none with `[jobs]`, `[clean]` or `[cloud]` (Req 2.7).
    - `dollars_to_cents`: 50 gives 5000, 4.94 gives 494, and `0.1 + 0.2` gives 30.
    - Refusals, each naming the key: a missing key, an unknown key, `daily_jobs = true`, `daily_jobs = 0`, `scrypt_n = 3`, a part size of 5 MiB minus 1, a part size that needs 10,001 parts for `[jobs].max_video_bytes`, a link lifetime of 7 days plus 1 second, the Stop_Times `"2027-02-04 08:00:00Z"`, `"2027-02-04T08:00:00+00:00"` and `"2027-02-04T08:00:00"`, and the `deploy_folder` values `"/home/cloudshell-user/x"`, `"tmp/x"`, `"/tmp"` and `"/tmp/../etc"` (Req 2.8).
    - The existing comment check and `[git]`, `[aws]`, `[paid]` and `[devlog]` checks in `tests/test_config.py` pass unchanged with `[website]` added (Req 2.1, 2.10).
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.7, 2.8, 2.10, 2.11_

  - [ ] 1.7 Write the unit tests for the App_Project and the imports
    - `tests/test_website_project.py`: `app/pyproject.toml` lists only boto3 and `requires-python = ">=3.11,<3.12"`. Every package in `app/uv.lock` but the virtual project is reachable from boto3 through the lock's dependency edges. The root `pyproject.toml` and `uv.lock` match the hashes recorded in task 1.1, hashed the same way (Req 1.2, 18.13).
    - `tests/test_website_imports.py`, an `ast` scan of every `*.py` under `app/`, which covers later tasks' files with no change: no boto3 or botocore; nothing outside the standard library, `app`, `cloud` and the top-level `.py` modules of `splatpipe`; no imported name that starts with `_`; no string constant `"secretsmanager"` or `"ssm"` and no `get_secret_value` or `get_parameter` name, so nothing uses Secrets Manager or Parameter Store (a plain text search for `ssm` would wrongly match `classmethod`); no `hmac.new` or `hmac.digest` call, so there's no signing secret; and no number outside an allow list of protocol constants, each with a comment naming its source. A failure lists the file and the line (Req 1.3, 1.4, 1.5, 2.6, 4.11).
    - The allow list starts with 0, 1 and the HTTP statuses of the design's status table (200, 303, 400, 401, 403, 404, 409, 413, 422, 429, 500 and 503), citing RFC 9110, because every `PublicRefusal` from task 4 on carries one.
    - The same file scans `splatpipe/` for any import of `app` or `cloud`, and `tests/` for any write of `APPROVED_SPEND` into the process environment, through `os.environ[...] =`, `os.environ.update`, `os.putenv` or `monkeypatch.setenv` (Req 1.5, 18.12).
    - _Requirements: 1.2, 1.3, 1.4, 1.5, 2.6, 4.11, 18.12, 18.13_

  - [ ] 1.8 Write the property test for Property 10 in `tests/test_website_config.py`, with Hypothesis
    - **Property 10: The `[website]` reader refuses a corrupted key by name**
    - **Validates: Requirements 2.1, 2.8**
    - For any key of `[website]` and any corruption of it (removed, the wrong type, a `bool` for a number, a number below its floor, NaN, infinity, a TOML date, or a Stop_Time not written as `YYYY-MM-DDTHH:MM:SSZ`), loading the Config refuses with a `PipelineError` whose message names that key.
    - Draw a key of the real `[website]` and a corruption, write a copy of the real `config.toml` with only that key's line changed into a temporary folder, and load it with `load_website_config(path)`.
    - _Requirements: 2.1, 2.8_

  - [ ] 1.9 Run the tests and commit
    - Run `uv run pytest` until every test passes, and check again that `git status --porcelain -- pyproject.toml uv.lock` prints nothing.
    - On `feat/website`, stage `config.toml`, the new `app/` and `tests/` files and, if it changed, `docs/third-party-licences.md`, each by name, and commit them with the title "Website: the website config and the app project" and a body saying that it adds the website section to config.toml, the app project that locks boto3 alone so the root lock and the machine image never change for the website, and the checks on both.
    - _Requirements: 18.11_

- [ ] 2. The AWS seam and its fake
  - Files: `cloud/table.py`, `tests/fake_table.py`, `tests/test_website_table.py` and `tests/test_website_aws.py` (new), `cloud/aws.py`, `cloud/errors.py`, `cloud/select.py`, `cloud/launch.py` and `tests/fake_aws.py` (changed).
  - _Requirements: 1.4, 1.6, 1.8, 10.4, 11.1, 18.1, 18.2, 18.3_

  - [ ] 2.1 Write `cloud/table.py`
    - The design's typed values, as frozen dataclasses, with the standard library only: `Scalar`, `Key`, `Prefix`, `Between`, `KeyRange`; the conditions `Exists`, `NotExists`, `Equals`, `AtMost`, `AtLeast`, `And`, `Or`; the updates `Set`, `Add`, `Remove`; the transaction items `Put`, `Change`, `Check`; `ConditionFailed` with its `index`; `TransactionConflict`; and `ITEM_THERE` and `ITEM_ABSENT`.
    - Keep DynamoDB's limit of 100 items in one transaction as a named constant with its source.
    - _Requirements: 1.4, 10.4, 18.2_

  - [ ] 2.2 Add the typed errors to `cloud/errors.py` and raise them in `cloud/select.py` and `cloud/launch.py`
    - `NoGpuQuota`, `QuotaFull`, `CapacityExhausted` with `tried: tuple[tuple[str, str], ...]` in attempt order, and `QuotaRefused`, all subclasses of `CloudError`.
    - `admit` raises `NoGpuQuota` when the G_Quota is 0 and `QuotaFull` when no candidate fits, each with today's message word for word.
    - `launch_with_fallback` raises `CapacityExhausted` with today's message when it tried at least one attempt, and a plain `CloudError` with today's message when the plan was empty.
    - `QuotaRefused` isn't a `CapacityUnavailable`, so it stops the launch loop as today's error does. Every `except CloudError` and `pytest.raises(CloudError)` still catches all four.
    - _Requirements: 9.2, 11.1, 11.4_

  - [ ] 2.3 Add the table methods to `Aws` and `BotoAws` in `cloud/aws.py`
    - The `table: str | None` attribute and `table_get`, `table_get_many`, `table_query`, `table_put`, `table_update`, `table_transact(items, token)` and `table_delete`, with the design's signatures. `BotoAws` refuses a table call with a `CloudError` while `table` is unset.
    - `BotoAws` uses the low-level `dynamodb` client and `boto3.dynamodb.types`' serializer and deserializer. Each condition and update becomes an expression with `#n0`-style names and `:v0`-style values, so no attribute name or value is ever pasted into an expression. Numbers come back as `int`.
    - `table_get` reads strongly consistent, `table_get_many` is one `TransactGetItems`, and `table_query` follows `LastEvaluatedKey` to the end. `table_transact` passes `token` as the `ClientRequestToken` and refuses more than 100 items.
    - Errors: `ConditionalCheckFailedException` gives `ConditionFailed(index=None)`. `TransactionCanceledException` is read through its `CancellationReasons`, in request order: the first `ConditionalCheckFailed` gives `ConditionFailed(index=i)`, and otherwise any `TransactionConflict` gives `TransactionConflict`. `TransactionConflictException` on a single write gives `TransactionConflict`. Every other AWS error stays a `CloudError` with `account_redaction(str(error))`.
    - _Requirements: 1.4, 10.4, 18.2_

  - [ ] 2.4 Add the S3, EC2 and CloudFormation additions and the builders to `cloud/aws.py`
    - `s3_start_upload`, `s3_part_link`, `s3_complete_upload`, `s3_abort_upload`, `s3_download_link`, `s3_copy(source_key, key, *, part_bytes)`, `s3_size` and `s3_read_range`, carried out as the design's "How `BotoAws` carries them out" says.
    - Presigned_Links come from a second S3 client with `signature_version="s3v4"`, `addressing_style="virtual"` and `endpoint_url=f"https://s3.{region}.amazonaws.com"`, so every link's host is `s3_origin(bucket, region)`. A part link signs `upload_part` for `PUT`. A download link signs `get_object` with `ResponseContentDisposition` set to `attachment; filename="<filename>"`.
    - `s3_copy` uses boto3's managed `copy` with a `TransferConfig` whose threshold and chunk size are `part_bytes`. `s3_size` is one `list_objects_v2(Prefix=key, MaxKeys=1)`, giving the size only when the first key is exactly `key`. `s3_read_range` sends `Range: bytes=<offset>-<offset+length-1>`.
    - `s3_exists` gains the 403 fallback: on a 403 from `HeadObject` it asks `s3_size`, returns `False` when that finds no key, and raises a `CloudError` when the list is refused too or when it finds the key that `HeadObject` refused. A 404 still gives `False`, as today.
    - `run_instances` turns `VcpuLimitExceeded` and `InstanceLimitExceeded` into `QuotaRefused(account_redaction(str(error)))`, the same text as today's plain `CloudError`.
    - `create_change_set(request)` waits with the `change_set_create_complete` waiter. `change_set_changes(stack_name, change_set_name)` follows `NextToken` and returns each change's `Action`, `LogicalResourceId` and `Replacement`, never a physical id.
    - Module functions: `s3_origin(bucket, region)`; `website_aws(region)` and `cloudshell_aws(region)`, which name the region and no profile; and `aws_setting(config, key)`, which reads a non-empty string from the parsed `[aws]` table or refuses naming the key.
    - _Requirements: 1.6, 1.8, 11.1_

  - [ ] 2.5 Write `tests/fake_table.py` and extend `tests/fake_aws.py`
    - `tests/fake_table.py` is a small pure evaluator over dicts, sharing no code with `cloud/aws.py`: whether a condition holds for an item or for no item, where a comparison with a missing attribute is false; applying updates, where `Add` on a missing attribute starts from 0 and `Remove` drops it; and puts, updates and transactions over `items: dict[tuple[str, str], dict[str, Scalar]]`.
    - A transaction applies all its items or none, raises `ConditionFailed(index=i)` for the first failing item in request order, and refuses two items on one key with a `CloudError`, as DynamoDB does.
    - `FakeAws` gains every new method, each recording `(op, key)` in `calls`, every Presigned_Link included. `table_get_many` is one call. `table_transact` records its token and can be seeded to raise `TransactionConflict` for its next n transactions, changing nothing.
    - Multipart uploads and links: `multipart: dict[str, FakeMultipart]` holds each upload's key and parts. A link is a URL on `s3_origin(bucket, region)` that the fake remembers with its kind, key, part and expiry. `put_part(url, data)` and `follow_link(url)` answer 403 for an expired or unknown link, judged by the fake's settable `now`, and `put_part` gives back an `ETag` like S3's quoted hex digest.
    - Objects: `s3_copy`, `s3_size` and `s3_read_range` work on `objects`, and `object_sizes` lets a test claim a large size without holding the bytes.
    - `quota_refused` holds `(type, zone)` pairs whose launch raises `QuotaRefused`. `create_change_set` records the request, and `change_set_changes` returns seeded lines.
    - Everything `FakeAws` did before behaves as before, so the cloud-jobs tests pass unchanged.
    - _Requirements: 18.1, 18.2, 18.3_

  - [ ] 2.6 Write the unit tests for the fake in `tests/test_website_table.py`
    - Every new `FakeAws` operation records its call, in order, Presigned_Links included, and `table_get_many` records one call (Req 18.3).
    - `table_query` with a `Prefix` and with a `Between` returns the matching items in `sk` order. `table_delete` removes one item.
    - A transaction with two items on one key is refused and changes nothing. A seeded `TransactionConflict` raises once per seeded transaction and changes nothing.
    - Multipart: a part sent with a fresh link is kept, a link past its expiry answers 403, an unknown link answers 403, and completing joins the parts into one object. `s3_size` gives `None` for a missing key and the claimed size from `object_sizes`. `s3_read_range` gives the exact bytes. `follow_link` serves a download link once it's made and answers 403 after it expires.
    - _Requirements: 18.1, 18.2, 18.3_

  - [ ] 2.7 Write the unit tests for `BotoAws` and the typed errors in `tests/test_website_aws.py`
    - Attach botocore's `Stubber` to `BotoAws`'s clients, built from a `boto3.Session` with obviously fake keys like `testing` and `region_name="us-east-1"`, so no request leaves the machine.
    - The expressions for `Put`, `Change` and `Check` use only `#n` names and `:v` values. The `CancellationReasons` mapping gives the index of the first `ConditionalCheckFailed`, or `TransactionConflict`. `ConditionalCheckFailedException` and `TransactionConflictException` on single writes map as the design says. Another error is a `CloudError` with the account id redacted. A table call while `table` is unset is refused. `table_query` follows `LastEvaluatedKey`, and `change_set_changes` follows `NextToken` and holds no physical id.
    - `s3_exists`: a 403 then an empty list gives `False`, a 403 then a refused list raises, and a 403 then the key found raises.
    - `run_instances`: `VcpuLimitExceeded` and `InstanceLimitExceeded` give `QuotaRefused` with the same text as before.
    - A part link's host is `s3_origin(bucket, region)` and it signs a `PUT` of `upload_part`. A download link holds `response-content-disposition`. `s3_origin("b", "us-east-1")` is `https://b.s3.us-east-1.amazonaws.com`.
    - `website_aws` and `cloudshell_aws`, with `boto3.Session` patched to record its arguments, pass `region_name` and no `profile_name` (Req 1.6, 1.8). `aws_setting` refuses a missing or empty key.
    - `admit` with a G_Quota of 0 raises `NoGpuQuota`, and with no room `QuotaFull`. `launch_with_fallback` with only capacity errors raises `CapacityExhausted` with `tried` in order, and with an empty plan a plain `CloudError` that's none of the four. Each message equals today's.
    - _Requirements: 1.6, 1.8, 11.1, 18.1_

  - [ ] 2.8 Write the property test for Property 12 in `tests/test_website_table.py`, with Hypothesis
    - **Property 12: The fake table is faithful**
    - **Validates: Requirements 18.2**
    - For any table state and any put, update or transaction with generated keys, conditions and updates: a write whose conditions all hold leaves the state equal to applying each item's updates in turn, with `Add` on a missing attribute starting from 0; a write with a failing condition raises `ConditionFailed`, naming the first failing item of a transaction, and leaves the state exactly as it was; and a comparison with a missing attribute is false.
    - Draw random items, then a put, an update or a transaction of up to 5 items on distinct keys with random conditions, and compare with an oracle that checks each item's condition against the state before the write and applies each item alone to a copy of the state.
    - _Requirements: 18.2_

  - [ ] 2.9 Run the tests and commit
    - Run `uv run pytest` until every test passes, every `tests/test_cloud_*.py` included, unchanged.
    - Stage the task's files by name and commit them with the title "Website: the table seam, the new AWS operations and their fake" and a body saying that it adds typed table values with no boto3, the new S3, DynamoDB, EC2 and CloudFormation operations on the Aws seam, typed refusals with unchanged messages, and a faithful fake of all of it.
    - _Requirements: 18.11_

- [ ] 3. The Start_Core split
  - Files: `app/baked.py`, `tests/test_start_core.py` and `tests/snapshots/start_job_timeline.json` (new), `cloud/start.py` (changed).
  - _Requirements: 9.1, 9.2, 9.3, 10.2_

  - [ ] 3.1 Record today's `start_job` behaviour, before changing `cloud/start.py`
    - In `tests/test_start_core.py`, write scenario helpers that run today's `start_job` against `FakeAws`, with `cloud.start.check_spend` and `cloud.start.make_job` monkeypatched as the cloud-jobs tests do, for a full run and for each refusal in today's order: an empty Step_List, an uncommitted change, no available Machine_Image, a lock-hash drift, no Smoke_Marker, no type that fits, a G_Quota of 0, no room in the G_Quota, a Spend_Guard refusal, every attempt out of capacity, and an empty launch plan.
    - Run them once on the unchanged code and write each scenario's ordered `FakeAws.calls` and its stdout into `tests/snapshots/start_job_timeline.json`, as small sorted JSON. Task 3.4 compares the split code against this file.
    - _Requirements: 9.2_

  - [ ] 3.2 Split `start_job` into the Start_Core in `cloud/start.py`
    - The frozen dataclasses `Code`, `Gate`, `Choice` and `Launched`, and the public functions `check_step_list()`, `check_image(aws, config, code)`, `choose_machine(aws, gate, entries, preset, config, cc)`, `approve(choice, environ)` and `launch_job(aws, gate, choice, job_id, code, place_inputs, config, cc)`, with the design's signatures and docstrings.
    - `start_job` keeps its signature, its messages, its refusal order and its call order, and composes the five functions in the design's 9 steps. Its `place_inputs` uploads the video, then `code.tar`, then `job.json`.
    - `approve` calls `check_spend` by its bare module-level name, and `start_job` calls `make_job` the same way, so both resolve `cloud.start`'s globals at call time. Never bind either as a default argument, a dataclass field or a local alias. `_smoke_marker_key` stays as it is.
    - _Requirements: 9.1, 9.2_

  - [ ] 3.3 Add `worst_case_cost` to `cloud/start.py` and write `app/baked.py`
    - `worst_case_cost(preset, cc)` takes the dearest type on the Preset's whole Fallback_List by `price_per_hour_usd` and returns its cost in dollars at the Preset's Time_Cap through `cloud.spend_file.rough_cost`, the same cost the quote uses, IPv4 address and root volume included. `start_job` doesn't call it.
    - `app/baked.py`: `BakedCommit` with `code()`, which gives the `cloud.start.Code` the Start_Core takes; `read_baked(root)`, which reads `<root>/baked/commit.json` and checks that `<root>/baked/code.tar` exists; and `write_baked(folder, commit, uv_lock_hash, package_lock_hash, archive)`, which writes `commit.json` as `{"commit": ..., "uv_lock_hash": ..., "package_lock_hash": ...}` and puts the archive at `<folder>/baked/code.tar` unless it's already there.
    - Both refuse a commit that isn't 40 lowercase hex digits or a hash that isn't 64, with a `CloudError` naming the field.
    - _Requirements: 9.3, 10.2_

  - [ ] 3.4 Write the unit tests for the Start_Core in `tests/test_start_core.py`
    - For the full run and for each refusal of task 3.1, the split `start_job`'s `FakeAws.calls` and stdout equal `tests/snapshots/start_job_timeline.json` (Req 9.2).
    - The Website's composition, `check_step_list`, `check_image`, the caller's Job_Record, `choose_machine`, `approve` and `launch_job`, calls `FakeAws` in the order of Requirement 9.1, and `place_inputs` runs once, with the Job's input prefix and the code key, before the first `run_instances` (Req 9.1).
    - `check_image` with a `Code` whose `uv_lock_hash` differs from the Machine_Image's tag refuses with today's drift message (Req 9.3).
    - `approve` with `environ={"APPROVED_SPEND": "1"}` returns, and with `environ={}` raises `SpendRefused`. A monkeypatched `cloud.start.check_spend` is what both `start_job` and `approve` call.
    - `dollars_to_cents(worst_case_cost(preset, cc))` is 494 for fast, 741 for default and 1815 for quality, with the real Config read by path (Req 10.2).
    - `write_baked` then `read_baked` gives back the same `BakedCommit`, `code()` gives a `Code` with the same values, and a malformed `commit.json` is refused.
    - _Requirements: 9.1, 9.2, 9.3, 10.2_

  - [ ] 3.5 Write the property test for Property 5 in `tests/test_start_core.py`, with Hypothesis
    - **Property 5: A Worst_Case_Cost is never below the quote's worst case**
    - **Validates: Requirements 10.2**
    - For any valid `CloudConfig`, with any types, prices, Fallback_Lists, Time_Caps and IPv4 and storage prices, and any hard needs that leave at least one candidate, `dollars_to_cents(worst_case_cost(preset, cc))` is at least the cents of the worst case in the `rough_cost` string that `choose_machine` builds for the same Preset.
    - Draw `cloud_config(**overrides)` with random prices, Fallback_Lists drawn from the real types, Time_Caps and Expected_Times, and random hard needs, and seed `FakeAws` with a G_Quota that admits at least one candidate.
    - _Requirements: 10.2_

  - [ ] 3.6 Run the tests and commit
    - Run `uv run pytest` until every test passes, every `tests/test_cloud_*.py` included, unchanged.
    - Stage the task's files by name and commit them with the title "Website: split start_job into the shared Start_Core" and a body saying that the laptop and the website now start jobs through the same public functions, in the same order and with the same messages, and that the website will run the commit it was deployed from.
    - _Requirements: 18.11_

- [ ] 4. The Box_Reader and the Video_Check
  - Files: `app/video.py` and `tests/test_website_video.py` (new), `tests/website_helpers.py` and `tests/test_website_imports.py` (changed).
  - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8, 9.9, 18.10_

  - [ ] 4.1 Write `find_moov` in `app/video.py`
    - `VIDEO_FILES`, the `Read` alias, `MovieFacts`, `VideoFacts` and `find_moov(read, size, max_boxes, max_moov_bytes)`, as the design's `app/video.py` section gives them.
    - Walk the top-level Boxes from offset 0. At each offset read `min(16, size - offset)` bytes. Fewer than 8 is a broken file. The 32-bit size 1 means the next 8 bytes are the 64-bit `largesize` and the header is 16 bytes. Size 0 means the Box runs to the end of the file. A Box smaller than its own header, or one that runs past the end, is a broken file.
    - A `moov` Box returns its offset and length, after the refusal for one larger than `max_moov_bytes`. Any other Box is skipped by its size, and a size-0 Box that isn't `moov` ends the walk with no `moov`. Walking more than `max_boxes` Boxes is a refusal. Only headers are read, never a payload.
    - Each refusal is a `PublicRefusal` with status 422 and kind `video`. A broken file or no `moov` gets "This file isn't an MP4 or MOV video the pipeline can read. Save it again as MP4 on your phone and upload it again."
    - Add the Box header sizes 8 and 16 to the allow list in `tests/test_website_imports.py`, citing ISO/IEC 14496-12.
    - _Requirements: 8.2, 8.4, 8.5, 8.6_

  - [ ] 4.2 Write `read_movie` and `check_video` in `app/video.py`
    - `read_movie(moov)` walks the `moov` Box's children with the same header rules, bounded by their parent, and reads `mvhd` and each `trak`'s `mdia/hdlr` and `mdia/mdhd`. The handler type is the 4 bytes at offset 8 of the `hdlr` payload, after its version, flags and pre-defined field.
    - In `mvhd` and `mdhd`, version 0 keeps the time scale and duration as 32-bit values at payload offsets 12 and 16, and version 1 keeps the time scale at 20 and a 64-bit duration at 24. A duration of all ones counts as 0, and a time scale of 0 means the length can't be read.
    - The Video_Length is the first `vide` track's `mdhd` duration over its time scale, or the `mvhd` duration over its time scale when that's 0.
    - `check_video(read, size, stated_size, name, wc, jobs)` checks in the design's order: the name ends in `.mp4` or `.mov` in any letter case; the object still exists (`size` isn't `None`), or "That upload isn't there any more. Upload the video again."; the size is within `[jobs].max_video_bytes` and equals `stated_size`; then `find_moov`, one range read of the whole `moov` and `read_movie`; then a video track, a Video_Length above 0 and at most `[jobs].max_video_seconds`, and `splatpipe.make.used_part(video_seconds, None, None, jobs.min_used_seconds)` accepting the whole video.
    - Each refusal is a `PublicRefusal` in the Website's own words for what to do on a phone, meaning the same as the Job_Maker's refusal in that case. It returns `VideoFacts`, with `video_file` from `VIDEO_FILES` by the lower-cased extension.
    - _Requirements: 8.1, 8.3, 8.6, 8.7, 9.9_

  - [ ] 4.3 Add `mp4(boxes)` and the sparse reader to `tests/website_helpers.py`
    - `mp4(boxes)` builds small MP4 and MOV files from Box specs (a type, a payload or a claimed size, and `largesize` or not), with helpers for a `moov` holding an `mvhd` and `trak`s with `mdia/hdlr` and `mdia/mdhd` at version 0 or 1. It packs with `struct` and shares no code with `app/video.py`.
    - The sparse reader serves only the header bytes and the `moov`, records every read, and fails the test when asked for any byte inside another Box's payload, so a Box that claims 4 GiB costs nothing.
    - _Requirements: 18.10_

  - [ ] 4.4 Write the unit tests for the Video_Check in `tests/test_website_video.py`
    - Found: a phone-style MP4 with `ftyp`, `free`, `mdat` and then `moov`; a MOV with `moov` before `mdat`; a 64-bit `largesize` `mdat` that claims 4 GiB through the sparse reader; and a size-0 `moov` last. Version 0 and version 1 headers. An all-ones `mdhd` duration falls back to `mvhd`. The first `vide` track is chosen when a `soun` track comes first.
    - Refused, each with its message: a Box of size 4, a Box past the end, a size-0 `mdat` last with no `moov`, fewer than 8 bytes left, more than `max_boxes` Boxes, a `moov` over `max_moov_bytes`, no video track, a Video_Length of 0, a time scale of 0, over `[jobs].max_video_seconds`, shorter than `[jobs].min_used_seconds`, an `.avi` name, a gone object (`size=None`), a size over `[jobs].max_video_bytes`, and a size that differs from the stated size.
    - `IMG_0042.MOV` is accepted with `video_file` `video.mov`, and `clip.MP4` with `video.mp4`. The sparse reader shows that no payload byte of a non-`moov` Box was read.
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 9.9_

  - [ ] 4.5 Write the property test for Property 3 in `tests/test_website_video.py`, with Hypothesis
    - **Property 3: The Box_Reader finds `moov` anywhere and refuses a broken file**
    - **Validates: Requirements 8.2, 8.3, 8.4, 8.5, 8.6, 8.8, 18.10**
    - For any layout of top-level Boxes with any types and sizes, 32-bit or 64-bit `largesize` headers and an optional size-0 last Box, holding one `moov` Box at any position, with or without a video track and with any time scales and durations, within the bounds in `[website]`: `find_moov` returns that Box's offset and length, `read_movie` returns the layout's video-track flag and its Video_Length (the `mvhd` value when the track's duration is 0), and no read touches a byte inside another Box's payload. For any layout with a Box smaller than its header, a Box that runs past the end, no `moov`, more than `max_boxes` Boxes before `moov`, or a `moov` larger than `max_moov_bytes`, `check_video` refuses with a `PublicRefusal`.
    - Draw a list of Box specs, a `moov` position, a track list, time scales, durations and an optional size-0 last Box, built into a sparse file. Make each broken layout by one mutation of a good one.
    - _Requirements: 8.2, 8.3, 8.4, 8.5, 8.6, 8.8, 18.10_

  - [ ] 4.6 Run the tests and commit
    - Run `uv run pytest` until every test passes.
    - Stage the task's files by name and commit them with the title "Website: the video check reads MP4 and MOV boxes over range reads" and a body saying that it finds the moov box wherever it is by walking box headers, reads the video track and its length, and refuses a bad video for free before any machine starts.
    - _Requirements: 18.11_

- [ ] 5. Accounts, Sessions and the sign-in limit, with the Account_Command
  - Files: `app/accounts.py`, `app/sessions.py`, `app/signin_limit.py`, `app/cloudshell.py`, `app/account_command.py`, `tests/test_website_accounts.py` and `tests/test_website_signin.py` (new), `tests/website_helpers.py` and `tests/test_website_imports.py` (changed).
  - _Requirements: 1.7, 1.8, 1.9, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 4.10, 4.12, 5.3, 5.4, 18.4, 18.15_

  - [ ] 5.1 Write `app/accounts.py`
    - `ACCOUNT_NAME`, `PASSWORD_ALPHABET`, `Account`, `check_account_name`, `new_password(length, random)`, `hash_password(password, salt, settings)`, `password_matches(password, account, configured)`, `profile_key(name)`, `read_account(item)`, `make_account(aws, name, wc, random)`, `replace_password(aws, name, wc, random)` and `remove_account(aws, name)`, with the design's signatures. `ScryptSettings` is imported from `app/config.py`.
    - `hash_password` is `hashlib.scrypt(password.encode("utf-8"), salt=salt, n=..., r=..., p=..., dklen=..., maxmem=256 * r * (n + p + 2))`, worked out from the hash's own settings. That's twice what OpenSSL allocates for those settings, `128 * r * (n + 2)` for its working arrays plus `128 * r * p`. The design's shorter `256 * r * n` is too small whenever `n < p + 2`, as with the tests' `scrypt_n=2, scrypt_p=1`, where OpenSSL refuses it.
    - `password_matches` runs exactly one scrypt either way: with an Account, its salt and its stored settings; with `None`, a fixed dummy salt and the configured settings, returning `False`. It compares with `hmac.compare_digest`.
    - The profile item is `account#<name>` / `profile`, holding `hash` and `salt` in base64, `n`, `r`, `p`, `dklen`, `uid` (32 random hex digits) and `created_at`, and never the password.
    - `make_account` puts the profile on `ITEM_ABSENT`, so a taken name raises `ConditionFailed` and changes nothing. `replace_password` updates the hash, the salt, the settings and a new `uid` on `ITEM_THERE`, which ends every Session of the Account at once. `remove_account` queries the partition `account#<name>`, deletes each item (the profile, its Sessions, its Uploads and its job items) and returns how many it deleted.
    - _Requirements: 3.2, 3.3, 3.5, 3.6, 3.7, 3.8, 4.1, 4.2_

  - [ ] 5.2 Write `app/sessions.py`
    - `SESSION_COOKIE`, `SIGNIN_COOKIE`, `Session`, `parse_session_cookie(value, token_bytes)`, `current(request, deps, now)`, `sign_in(request, deps, now)`, `sign_out(session, deps, now)`, `set_cookie(name, value, max_age, same_site)` and `clear_cookie(name)`.
    - `parse_session_cookie` accepts `<account name>.<id>` only when the name matches `ACCOUNT_NAME` and the id is exactly the unpadded base64url length of `token_bytes` bytes, 43 characters for 32, in the base64url alphabet. Anything else is `None`, with no AWS call.
    - `current` reads the profile item and the session item `account#<name>` / `session#<SHA-256 hex of the id>` in one `table_get_many`. It's a Valid_Session when both exist, the session isn't `ended`, its `expires_at` is after `now` by the handler's clock, and its `uid` equals the profile's. A failed read raises, never "signed out".
    - `sign_in` runs in this order: the sign-in token check, the form field `csrf` against the `__Host-voxelis-signin` cookie with `hmac.compare_digest`, refused with a 403 `PublicRefusal` of kind `csrf`; `signin_limit.locked_until`, refused with a 429 of kind `signin_wait` that gives the minutes to wait, checking no password; one read of the profile item, skipped for a malformed name; `password_matches`; and on a failure, `signin_limit.record_failure` and a 400 `PublicRefusal` of kind `bad_request` saying "That name and password don't match an account.". Task 7's handler shows each refusal on the sign-in page. The Sign_In_Address is `request.source_ip` alone.
    - A new Session draws its id and its CSRF_Token from `deps.random.token_bytes(token_bytes)` each, puts the session item with `csrf`, `uid`, `created_at` and `expires_at` `session_days` later on `ITEM_ABSENT`, and answers a 303 to `/jobs` that sets `__Host-voxelis-session` with `Path=/; Secure; HttpOnly; SameSite=Lax; Max-Age=<session_days in seconds>` and no `Domain`, and clears the sign-in cookie.
    - `sign_out` sets `ended` on the session item, clears the Session_Cookie with `Max-Age=0` and answers a 303 to `/signin`. Nothing is deleted.
    - _Requirements: 4.1, 4.3, 4.6, 4.7, 4.8, 4.9, 4.10, 4.12, 5.3, 5.4_

  - [ ] 5.3 Write `app/signin_limit.py`
    - `locked_until(aws, address, now, wc)` and `record_failure(aws, address, now, wc, random)`, with the design's items: each Sign_In_Failure is `signin#<address>` / `failure#<10-digit epoch seconds>#<8 hex digits>` with `expires_at` one window later, and the lock is `signin#<address>` / `lock` holding `until` and `expires_at`.
    - `locked_until` is one `table_query` of the address's partition, `Between("failure#<now - window + 1 s>", "lock")`, which returns the window's failures and the lock together, and gives the lock's `until` while it's after `now`.
    - `record_failure` puts the new failure, queries again, and when the window holds `signin_max_failures` failures or more, puts the lock with `until = now + signin_wait_minutes`. Times are whole epoch seconds from `int(now.timestamp())`.
    - Nothing is counted per Account, and one address's items never touch another's.
    - Add each number this task's code holds to the allow list in `tests/test_website_imports.py`, with its source: the 2 and 256 of the `maxmem` formula (OpenSSL's scrypt), the 4 random bytes behind a failure's 8 hex digits, 60 seconds a minute and 86,400 seconds a day.
    - _Requirements: 4.4, 4.5_

  - [ ] 5.4 Write `app/cloudshell.py` and `app/account_command.py`
    - `require_cloudshell(environ)` raises a `CloudError` unless `environ.get("AWS_EXECUTION_ENV") == "CloudShell"`, with the design's message: "This command runs only in AWS CloudShell, signed in as the account owner. Open CloudShell in us-east-1 and follow docs/aws-setup.md." `run_cloudshell_cli(action)` is `cloud.errors.run_cloud_cli`.
    - `account_main(argv, *, environ, make_aws, random, out, config_path)` for `python3.11 -m app.account_command <name> [--new-password | --remove]`. The keyword `config_path` isn't in the design: it defaults to the `config.toml` beside `app/`, and the tests pass `REPO_ROOT / "config.toml"`, because the autouse `temp_config` has no `[aws]` or `[website]`.
    - It calls `require_cloudshell(environ)` before it builds any AWS client, builds the client with `make_aws(aws_setting(config, "region"))`, reads `WebsiteTableName` with `aws.stack_outputs(aws_setting(config, "stack_name"))` and sets `aws.table`. With no option it makes the Account, `--new-password` replaces the password, and `--remove` removes the Account.
    - A new password goes to `out` once, alone on its line, and nowhere else. A name that doesn't match `ACCOUNT_NAME` is refused with the pattern. A taken name is refused with a message naming `--new-password`, and nothing changes.
    - The module's `__main__` block runs `run_cloudshell_cli` with `os.environ`, `cloudshell_aws`, `SecretsRandom()` and `sys.stdout`, and never prompts.
    - _Requirements: 1.7, 1.8, 1.9, 3.1, 3.2, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9_

  - [ ] 5.5 Add `make_deps` and `sign_in_as` to `tests/website_helpers.py`
    - `make_deps(fake, **overrides)` loads the real `config.toml` by path into the `Config`, the `CloudConfig` and a `WebsiteConfig` with `scrypt_n=2, scrypt_r=1, scrypt_p=1` by default, sets `fake.table` and `fake.bucket`, and builds `Deps` with a `BakedCommit` over a tiny `code.tar` in a temporary folder, whose lock hashes equal the Machine_Image the test seeds, `static={}` until task 7, a seeded `Random` and a test logger. Any field can be overridden by name.
    - `sign_in_as(deps, name, now)` seeds an Account with `make_account` and a Session item, and returns the Session_Cookie's value and the CSRF_Token.
    - _Requirements: 18.4_

  - [ ] 5.6 Write the unit tests for Accounts and the Account_Command in `tests/test_website_accounts.py`
    - `hash_password` runs once with the real `[website]` settings, which shows `maxmem` is big enough, and with the tiny test settings. `password_matches` is true for the right password, false for a wrong one, and runs one scrypt for `None`, counted through a patched `hashlib.scrypt`. An Account made with other scrypt settings still checks after `[website]`'s settings change (Req 3.3).
    - The Account_Command outside CloudShell is refused before `make_aws` is called (Req 1.9). It builds its client with the region and no profile (Req 1.8).
    - Making an Account prints a password of `password_length` letters and digits once, alone on its line. The password appears nowhere in the table, in stderr or in the captured logs, and the profile item holds only the design's attributes (Req 3.2, 3.4).
    - A bad name is refused with the pattern (Req 3.5). A taken name is refused naming `--new-password`, with the table unchanged (Req 3.6). `--new-password` prints a new password, changes the `uid`, and an old Session is no longer valid (Req 3.7). `--remove` deletes the profile, the Sessions, the Uploads and the job items, and leaves the `jobs/` objects, the Limit_Counters, the Sign_In_Failures and the Showcase_Entries (Req 3.8).
    - No route of the Website makes an Account: the only writer of a profile item is `app/accounts.py`, which the Account_Command alone calls (Req 3.1).
    - _Requirements: 1.8, 1.9, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 18.15_

  - [ ] 5.7 Write the unit tests for signing in in `tests/test_website_signin.py`
    - The right password answers a 303 to `/jobs` with the cookie exactly `__Host-voxelis-session=<name>.<43 characters>; Path=/; Secure; HttpOnly; SameSite=Lax; Max-Age=604800` and no `Domain`. The session item's key holds the SHA-256 hex of the id, never the id, with `csrf`, `uid`, `created_at` and an `expires_at` 7 days later, and the sign-in cookie is cleared (Req 4.1, 4.7, 4.8).
    - A wrong password and an unknown name give the same refusal text, run one scrypt each, count one Sign_In_Failure for the address and make no Session (Req 4.2, 4.3).
    - The fifth failure within 15 minutes starts the wait. The next attempt, even with the right password, is told how many minutes to wait and runs no scrypt. After 15 minutes the right password works (Req 4.4).
    - Failures from one address don't lock another, so failing on the shared Account from many addresses never locks it. A spoofed `X-Forwarded-For` header doesn't change the Sign_In_Address (Req 4.5, 4.6).
    - A Session past its `expires_at` is signed out while its item still exists, and so are an ended Session, a stale `uid` and a removed Account (Req 4.9).
    - Malformed cookies (no dot, a bad name, a wrong id length, padding, characters outside base64url) give `None` with no AWS call, and a well-formed cookie costs exactly one `table_get_many` (Req 5.3, 5.4).
    - A missing or wrong sign-in token is a 403 that writes nothing (Req 4.12). Signing out sets `ended`, clears the cookie with `Max-Age=0` and answers a 303 to `/signin` (Req 4.10). A failed session read raises rather than signing out.
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 4.10, 4.12, 5.3, 5.4, 18.4_

  - [ ] 5.8 Write the property test for Property 7 in `tests/test_website_signin.py`, with Hypothesis
    - **Property 7: The sign-in limit counts per address**
    - **Validates: Requirements 4.4, 4.5**
    - For any sequence of sign-in attempts from several Sign_In_Addresses at non-decreasing times, each with the right password, a wrong one or an unknown name, each attempt's outcome (signed in, refused, or told to wait) equals that of a sliding-window model of its own address alone: an attempt is told to wait, with no password check and no scrypt, while a lock is running, and a lock starts at a counted failure that brings the address's counted failures within the last `signin_window_minutes` to `signin_max_failures` and runs for `signin_wait_minutes`. No address's outcomes change when another address's attempts are added or removed.
    - Draw up to 40 attempts from up to 3 addresses at whole-second times, each right, wrong or an unknown name, and compare with a plain-Python model that uses the same boundaries: a failure at second t counts at `now` when `t >= now - window + 1`, and a lock runs while `now < until`. A patched `hash_password` counts scrypt runs. Running the same draw with one address's attempts removed gives the other addresses the same outcomes.
    - _Requirements: 4.4, 4.5_

  - [ ] 5.9 Run the tests and commit
    - Run `uv run pytest` until every test passes.
    - Stage the task's files by name and commit them with the title "Website: accounts, sessions and the sign-in limit" and a body saying that it adds scrypt password hashes, random sessions stored only as hashes, a per-address limit on failed sign-ins, and the CloudShell-only account command that prints each new password once.
    - _Requirements: 18.11_

- [ ] 6. The Website_Limits and Reservations
  - Files: `app/limits.py` and `tests/test_website_limits.py` (new).
  - _Requirements: 10.2, 10.3, 10.4, 10.5, 10.6, 10.7, 10.8, 10.9, 10.12, 10.13, 11.2, 11.3, 18.5, 18.8_

  - [ ] 6.1 Write the Reservation's pieces in `app/limits.py`
    - `Reservation`, `LimitFull(PublicRefusal)` with its `limit` (`"daily_jobs"`, `"daily_cost"` or `"total_cost"`), `worst_case_cents(preset, cc)` and `utc_day(now)`, as in the design's `app/limits.py` section.
    - `worst_case_cents` is `dollars_to_cents(cloud.start.worst_case_cost(preset, cc))`. `utc_day` gives `YYYY-MM-DD` of `now` in UTC and refuses a naive `datetime`.
    - `LimitFull` has status 409 and kind `limit`. Its message names the limit and says no Job started, and for the Daily_Job_Limit and the Daily_Cost_Limit adds "Jobs can start again after 00:00 UTC."
    - Add `stop_line(wc)`, which the design doesn't name: "The website stopped starting jobs at 08:00 UTC on 4 February 2027.", written from `stop_starting_at` with an English month name and no `%-d`, which Windows lacks. A Stop_Time refusal is a `PublicRefusal` with status 409, kind `stopped` and that message. Task 7's start page and task 8's `begin`, `precheck` and `start` reuse it.
    - _Requirements: 10.2, 11.2, 11.3_

  - [ ] 6.2 Write `check_room`, `take` and `give_back` in `app/limits.py`
    - `check_room(aws, cents, now, wc)` refuses at or after the Stop_Time first, then refuses when one Reservation alone can't fit a limit, then reads `limits` / `day#<utc_day(now)>` and `limits` / `total` in one `table_get_many`, and refuses when the Reservation wouldn't fit, naming the limit: the Daily_Job_Limit first, then the Daily_Cost_Limit, then the Total_Cost_Limit.
    - `take(aws, reservation, steps, now, wc)` calls `check_room` with its own `now`, checks that `reservation.day == utc_day(now)`, then runs one `table_transact` with the Job_Id as its token, of the design's four items in this order: the day's counters, the total counter, the new job item (state `held`, with the Preset, the cents, the day, `started_at`, the Step count, the Upload's id and `launch = "pending"`, on `ITEM_ABSENT`), and the Upload's `used_by` on `And(Equals("state", "complete"), NotExists("used_by"))`.
    - Its failures: `ConditionFailed(index=0)` reads the day item once to tell the Daily_Job_Limit from the Daily_Cost_Limit; `index=1` is the Total_Cost_Limit; `index=3` is a 400 `bad_request` for an Upload already used or unfinished; `index=2` and `TransactionConflict` are a 409 of kind `conflict` saying "Another job started at the same moment, so yours didn't. Press Start again." Nothing loops or retries.
    - `give_back(aws, reservation)` is one `table_transact` of three items: the job item from `held` to `released` on `Equals("state", "held")`; the Reservation's own UTC_Day item, `Add("jobs", -1)` and `Add("cents", -c)` on `AtLeast` conditions; and the total item, `Add("cents", -c)` on `AtLeast("cents", c)`. It returns `True` when that applied. Then a separate `table_update` removes `used_by` from the Upload on `Equals("used_by", job_id)`, ignoring `ConditionFailed`, because the Upload item may have expired.
    - A second give-back fails its first condition, changes nothing and returns `False`. A `TransactionConflict` during a give-back also changes nothing and returns `False`, so the Reservation stays held, which can only overcount. The design doesn't say what a conflict there does, and this is the safe reading.
    - Nothing else in the Website writes a `limits` item, and no Job's real cost is ever written back.
    - _Requirements: 10.3, 10.4, 10.5, 10.6, 10.7, 10.8, 10.9, 10.12_

  - [ ] 6.3 Write the unit tests for the limits in `tests/test_website_limits.py`
    - `worst_case_cents` is 494, 741 and 1815 for fast, default and quality, with the real Config read by path (Req 10.2).
    - Each limit's refusal and its words: with 10 Reservations in the day, the 11th names the Daily_Job_Limit and says "after 00:00 UTC"; near 5000 cents in the day, the Daily_Cost_Limit; near 50000 cents in total, the Total_Cost_Limit, with no "after 00:00 UTC" (Req 10.3, 11.2).
    - A Reservation just inside each limit is taken: the 10th of the day, a day at 5000 - 494 cents, a total at 50000 - 494 cents (Req 10.3). A `website_config` whose daily cost is below one Worst_Case_Cost is refused by `check_room` before any transaction.
    - The UTC_Day boundary: a Reservation at 23:59:59 counts in that day and one at 00:00:00 in the next. One taken at 23:59:59 and given back at 00:00:01 lowers the earlier day's counters (Req 10.5).
    - A kept Reservation stays in the counters. A second give-back returns `False` and changes nothing (Req 10.8, 10.9).
    - The Stop_Time: `check_room` and `take` refuse at exactly `stop_starting_at` and after, with the `stop_line` message, and work one second before (Req 10.12).
    - The transaction: `take` makes exactly one `table_transact`, with the Job_Id as its token and the four items in order. A used Upload gives the `index=3` refusal. A seeded `TransactionConflict` gives "Press Start again" with nothing changed and no second `table_transact` (Req 10.4).
    - Only `take` and `give_back` write `limits` items, which a scan of `FakeAws.calls` across these flows shows (Req 10.6).
    - _Requirements: 10.2, 10.3, 10.4, 10.5, 10.6, 10.8, 10.9, 10.12, 11.2, 18.5_

  - [ ] 6.4 Write the property test for Property 1 in `tests/test_website_limits.py`, with Hypothesis
    - **Property 1: The Website_Limits are never passed**
    - **Validates: Requirements 10.3, 10.4, 10.5, 10.6, 10.7, 10.8, 10.9, 10.12, 10.13, 18.8**
    - For any interleaving of any number of Website starts, each made of its steps (the room check, taking the Reservation, a give-back after a refusal, a launch failure that keeps the Reservation, and a repeated give-back), with UTC_Day changes and the Stop_Time passing between any two steps, after every step: no UTC_Day's Limit_Counters show more than `daily_jobs` Reservations or more than `daily_cost_cents`, the total shows no more than `total_cost_cents`, no counter is below 0, no Reservation was taken at or after the Stop_Time, and each counter equals the cents and count of the Reservations taken in its UTC_Day, or in total, minus those given back.
    - A Hypothesis `RuleBasedStateMachine` over one `FakeAws` table and a clock, with `website_config` overrides that make the limits small, so they're reached often. Rules: begin a start for a Preset (the room check), seeding a complete, unused Upload for it; take its Reservation; give it back; give it back again; keep it after an unknown launch; seed a `TransactionConflict`; move the clock by minutes or days; and move it past the Stop_Time.
    - An `@invariant` checks the limits, the floor of 0, the Stop_Time and the equality with a model sum after every step. The model counts a take or a give-back only when the call says it applied.
    - _Requirements: 10.3, 10.4, 10.5, 10.6, 10.7, 10.8, 10.9, 10.12, 10.13, 18.8_

  - [ ] 6.5 Run the tests and commit
    - Run `uv run pytest` until every test passes.
    - Stage the task's files by name and commit them with the title "Website: daily and total limits with atomic reservations" and a body saying that each start reserves its worst-case cost in one conditional transaction, so two starts at once can't pass the daily or total limit, and a give-back happens at most once.
    - _Requirements: 18.11_

- [ ] 7. The handler, the pages, the statuses and the Showcase
  - Files: `app/lambda_function.py`, `app/pages.py`, `app/statuses.py`, `app/showcase.py`, `app/showcase_command.py`, `app/static/style.css`, `app/static/app.js`, `app/static/favicon.ico`, `tests/test_website_handler.py`, `tests/test_website_pages.py`, `tests/test_website_statuses.py` and `tests/test_website_showcase.py` (new), `app/web.py`, `tests/website_helpers.py` and `tests/test_website_imports.py` (changed).
  - _Requirements: 1.7, 1.9, 1.10, 3.1, 4.10, 4.12, 4.13, 5.1, 5.2, 5.3, 5.4, 5.5, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8, 6.9, 7.1, 11.3, 12.1, 12.2, 12.3, 12.4, 12.5, 12.6, 12.7, 12.8, 12.11, 12.12, 12.13, 13.1, 13.2, 13.3, 13.4, 13.5, 13.6, 13.7, 13.8, 13.9, 16.7, 18.6, 18.9, 18.15_

  - [ ] 7.1 Write `app/statuses.py`
    - `StepTime`, `JobView`, `website_status(out_keys, summary, steps, started_at, cap_minutes, now)`, `result_file(out_keys, clean)`, `job_view(aws, job_id, job_item, config, cc, now, *, timings)` and `may_see(aws, account, job_id)`, as in the design's `app/statuses.py` section, and a table of the design's Website_Status words.
    - `website_status` is pure. With a Job_Summary, the status is the summary's `status`. With none, it's `ended` once `started_at` plus the Time_Cap is at or before `now`, `starting` while `out_keys` is empty, and otherwise `running`, with n the number of `out/reports/*.json` keys plus 1, at most N.
    - `result_file` reads the two names from `config.step_settings("clean", preset)` and gives `clean_sog` when `out/<clean_sog>` is listed, else `train_ply` when `out/<train_ply>` is, else `None`.
    - `job_view` lists `jobs/<Job_Id>/out/` once through the `Aws` protocol's existing list method, reads `out/job.json` when it's listed (a Job_Summary that isn't valid JSON counts as none), and with `timings` reads each listed Step_Report's `duration_seconds`. The start time and the Preset come from the job item, or from the summary's `started_utc` and `preset` for a laptop Job in the Showcase. The warnings are the summary's `steps` whose `required` is false and whose `outcome` is `failed`, with their Readable_Errors, and the run's time is the summary's `duration_seconds`.
    - `may_see` reads `account#<name>` / `job#<Job_Id>` and `showcase` / `job#<Job_Id>` in one `table_get_many`, and gives the held job item and the `ShowcaseEntry`, either or both `None`.
    - Add `list_jobs(aws, account, config, cc, wc, now)`, which the design describes without naming: query `account#<name>` with `Prefix("job#")`, keep the held items, sort by Job_Id newest first, take `max_listed_jobs` and build each `JobView` without timings.
    - `app/statuses.py` imports `app/showcase.py`, so `app/showcase.py` imports `app.statuses` only inside `add_entry`, to avoid an import cycle.
    - _Requirements: 12.1, 12.2, 12.3, 12.5, 12.6, 12.7, 12.8_

  - [ ] 7.2 Write `app/showcase.py` and `app/showcase_command.py`
    - `FOOTAGE_LICENCES`, `NEEDS_CREDIT`, `ShowcaseEntry`, `entry_key(job_id)`, `check_entry(job_id, title, licence, credit, wc, now)`, `add_entry(aws, entry, config)`, `list_entries(aws)`, `remove_entry(aws, job_id)` and `licence_line(entry)`, as in the design's `app/showcase.py` section.
    - `check_entry` runs `check_job_id` and refuses an empty title, a title over `max_title_chars` or holding a control character, an unknown licence, a credit over `max_credit_chars`, and a missing credit for CC BY or CC BY-SA, saying the licence needs the author's credit.
    - `add_entry` reads the Job's `out/job.json` and lists its `out/`, refuses unless the status is `done` or `done_with_warnings` and a Result_File is there, takes any Job in the Job_Bucket, and puts the entry on `ITEM_ABSENT`, refusing an existing one with a pointer to `remove`.
    - `list_entries` gives the entries newest first by `added_at`. `licence_line` gives "Footage: <credit>, CC BY", "Footage: <credit>, CC BY-SA. This splat is shared under CC BY-SA 4.0." or, for another licence's entry with a credit, the credit alone.
    - `showcase_main(argv, *, environ, make_aws, out, config_path, now)` for `add <job_id> --title T --licence L [--credit C]`, `list` and `remove <job_id>`. It takes the same keyword-only arguments as `account_main` except `random`, plus `now` for `added_at`, which the design doesn't list. It calls `require_cloudshell(environ)` first, reads `BucketName` and `WebsiteTableName` from the Stack's outputs, and `list` prints each entry's Job_Id, title, licence and credit. It never prompts.
    - _Requirements: 1.7, 1.9, 13.1, 13.2, 13.3, 13.4, 13.5, 13.8, 13.9_

  - [ ] 7.3 Write `app/pages.py`
    - `esc(value)` is `html.escape(str(value), quote=True)`, and every value enters a page through it, in text and in attributes alike.
    - `layout(title, body, *, csrf, signed_in)`: the viewport meta tag and the stylesheet, and on a signed-in page the CSRF_Token in `<meta name="csrf-token">`, `app.js` loaded with `defer`, links to Jobs, Start and Showcase, and a sign-out form posting its CSRF_Token in a hidden field `csrf`.
    - `signin_page(token, error)`, `job_list_page(rows)`, `start_page(presets, chosen, stopped, upload)`, `result_page(view)`, `showcase_page(entries)`, `not_found_page()`, which links to the job list, and `error_page()`, which says something went wrong and to try again.
    - The start page has one file input with `accept=".mp4,.mov,video/mp4,video/quicktime"`, the three Presets as radio buttons with their Expected_Times, the `[jobs].default_preset` one checked, a Start button, a `<progress>` element and a message area. After the Stop_Time it shows `stop_line(wc)` in place of the form.
    - The Result_Page shows the status words, the Preset, the start time, the run's and each Step's time when present, and each failed Optional_Step's Readable_Error for Done with warnings. For Done and Done with warnings it offers one download link, a GET of `/jobs/<job_id>/download` and never a form, or says the Job has no splat to download when there's no Result_File. A Showcase_Entry's page shows its title and `licence_line`. One `<section id="viewer-slot" aria-labelledby="viewer-label">` holds only its heading "3D viewer", with `id="viewer-label"`.
    - Signed-in pages carry `data-part-retries` and `data-poll-seconds` on `<main>`. `data-job-id` is set only on a Result_Page whose status is Starting or Running, which tells the Page_Script to refresh, so a finished Job's page never reloads itself.
    - No page has an inline script, an inline style, a `style=` attribute or an `on...=` attribute.
    - _Requirements: 1.10, 6.1, 6.2, 6.3, 6.7, 6.8, 7.1, 11.3, 12.5, 12.6, 12.7, 12.8, 12.11, 13.6, 13.7_

  - [ ] 7.4 Write `app/static/`
    - `style.css`: one column with `max-width: 40rem`, no width above 360 px, and inputs and buttons at full width.
    - `favicon.ico`: a 16 by 16 icon under 1 KB, made by a short standard-library script that isn't committed, for example a 16-colour bitmap icon.
    - `app.js`, plain JavaScript with no build step and no inline handlers, holding in this task only the refresh: on a page whose `<main>` has `data-job-id`, `GET /api/jobs/<id>/status` every `data-poll-seconds` seconds, update the status text, and reload the page once the answer says `final`. A 401 opens `/signin`. Set it up from one `DOMContentLoaded` listener, so task 8 adds the start and upload parts beside it.
    - _Requirements: 6.1, 6.2, 12.4_

  - [ ] 7.5 Write `handle`'s parsing, size check, routing and Session check, and `stack_for`, in `app/web.py`
    - `handle(event, deps, now)` parses the payload-2.0 event: `requestContext.http.method`, `rawPath`, `rawQueryString`, `cookies` as a list of `name=value`, the headers with lowercased names, and `body` with `isBase64Encoded`. The Sign_In_Address is `requestContext.http.sourceIp` alone, and the host is `requestContext.domainName`.
    - A body whose length, worked out from the event's text before any decoding (for base64, the decoded length from the text's length and padding), is over `max_body_bytes` gets a 413 at once.
    - The route table is a list of method, compiled regular expression, kind (open, page or script), handler and name. A `<job_id>` segment is any text without `/`. An unknown path, or a known path with the wrong method, is a 404 for a Valid_Session.
    - An Open_Route runs with no Session check. Every other request, unknown paths included, runs `sessions.current` first. With no Valid_Session, a path under `/api/` gets a 401 with `{"error": "signed_out"}` and every other path a 303 to `/signin`, and nothing else happens.
    - `stack_for(deps)` calls `cloud.stack.resolve_stack` once, caches the outputs in `deps.stack` and sets `deps.aws.bucket`. A signed-out request never calls it.
    - _Requirements: 1.7, 5.1, 5.2, 5.3, 5.4, 6.6_

  - [ ] 7.6 Write `handle`'s CSRF check, error mapping and headers in `app/web.py`
    - Every POST is refused with a 403 of kind `csrf`, changing nothing, when its CSRF_Token is missing or wrong (the form field `csrf` on pages, the `X-CSRF-Token` header on Script_Requests, the double-submit cookie on the sign-in form), when an `Origin` header is there and isn't `https://<host>`, or when `Sec-Fetch-Site` is there and isn't `same-origin` or `none`. Tokens compare with `hmac.compare_digest`. Only POST handlers write to the table, start an upload or start a Job.
    - A `PublicRefusal` becomes `{"error": kind, "message": message}` for a Script_Request, the sign-in page with the message and a fresh sign-in token for `POST /signin`, the 404 page for kind `not_found`, and otherwise the error page with the message, each with its status. Every message passes `account_redaction` and `esc`.
    - Any other `Exception` becomes the 500 page, or `{"error": "server"}` for a Script_Request, and one log line holding the error's type and its Account_Redacted message, with no traceback, AWS error text or ARN in the answer.
    - Every response gets the Content-Security-Policy `default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self'; connect-src 'self' <s3_origin>; form-action 'self'; frame-ancestors 'none'; base-uri 'none'`, `X-Content-Type-Options: nosniff` and `Referrer-Policy: no-referrer`. Every signed-in page calls `stack_for` before its headers, so its `connect-src` names `s3_origin(bucket, region)`, and a signed-out answer leaves it out. Every signed-in page and every Script_Request answer also gets `Cache-Control: no-store`.
    - The answer is the payload-2.0 response: `statusCode`, `headers`, `cookies`, `body`, and `isBase64Encoded` for the favicon.
    - Add any further protocol constant the handler needs, such as the base64 sizes 3 and 4 for the body length, to the allow list in `tests/test_website_imports.py`, with its source.
    - _Requirements: 4.12, 4.13, 6.3, 6.4, 6.5, 6.8, 6.9_

  - [ ] 7.7 Add this task's routes to the route table in `app/web.py`
    - Open: `GET /signin` answers the sign-in page and sets `__Host-voxelis-signin` to a fresh token with `Path=/; Secure; HttpOnly; SameSite=Strict; Max-Age=<signin_token_minutes in seconds>`, the same token in the form. `POST /signin` is `sessions.sign_in`. `GET /static/style.css` and `GET /favicon.ico` serve `deps.static` with their content types.
    - Pages: `GET /` answers a 303 to `/jobs`; `GET /jobs` the job list from `list_jobs`; `GET /start` the start page, with the Presets and their Expected_Times from `[jobs]` and `[cloud.presets]`, and `stop_line` after the Stop_Time; `GET /showcase` the entries read live with `list_entries`; `GET /static/app.js` the Page_Script, for signed-in requests only; and `POST /signout` is `sessions.sign_out`.
    - `GET /jobs/<job_id>` passes the segment through `splatpipe.jobs.check_job_id`, then `may_see`, answering the same 404 for a refused Job_Id, a Job that doesn't exist and a Job the Account may not see, then shows `job_view(..., timings=True)`.
    - `GET /api/jobs/<job_id>/status` makes the same checks and answers `{"status": ..., "text": ..., "final": ...}`.
    - No route makes an Account. After the Stop_Time every route keeps working, and only starting a Job refuses.
    - _Requirements: 3.1, 4.10, 6.7, 7.1, 11.3, 12.1, 12.2, 12.4, 12.5, 12.12, 12.13, 13.5, 13.6, 13.7, 16.7_

  - [ ] 7.8 Write `app/lambda_function.py`
    - `handler(event, context)` builds `Deps` on the container's first request and keeps it: the table name from `WEBSITE_TABLE`, the only environment variable it reads, never `APPROVED_SPEND`; `website_aws(aws_setting(config, "region"))` with `table` set; the `Config`, the `CloudConfig` and the `WebsiteConfig` from the bundle's `config.toml`; `read_baked` of the bundle's root; the files of `app/static/` read into memory; `SecretsRandom()`; and the logger. It returns `handle(event, deps, datetime.now(timezone.utc))`.
    - A logging filter applies `account_redaction` to every record, the handler's and boto3's alike. Each request writes one line, with the method, the route's name, the status and the time it took, and never a password, Session id, CSRF_Token, cookie, query string, body or Presigned_Link.
    - _Requirements: 1.6, 6.9_

  - [ ] 7.9 Add `event(...)` to `tests/website_helpers.py` and load the static files in `make_deps`
    - `event(method, path, *, cookies, headers, body, source_ip)` builds a payload-2.0 event with `requestContext.http`, `requestContext.domainName`, `rawPath`, `rawQueryString`, the cookie list, lowercased headers and a plain or base64 body.
    - `make_deps` now reads the real `app/static/` files into `static` by default.
    - _Requirements: 18.6_

  - [ ] 7.10 Write the unit tests for the statuses and the job list in `tests/test_website_statuses.py`
    - Each Website_Status from seeded `out/` files: Starting with an empty `out/`; Running (step 1 of N) with files but no Step_Report, and n capped at N with more reports than Steps; Done, Done with warnings and Failed with its Readable_Error from the Job_Summary; a Job_Summary that isn't valid JSON counted as none; and Ended without a result exactly at `started_at` plus the Time_Cap, with Running one second before (Req 12.3).
    - Done with warnings lists each failed Optional_Step's Readable_Error. The timings come from the Step_Reports and the Job_Summary's `duration_seconds` (Req 12.5, 12.6).
    - The Result_File is `clean.sog` when listed, `train.ply` when only that is, and `None` with neither, read from `[clean]`'s names (Req 12.7, 12.8).
    - `list_jobs` shows only the Account's held items, newest first, at most `max_listed_jobs`, and never another Account's Jobs or a released item (Req 12.1, 12.2).
    - `may_see` makes one `table_get_many` and allows the Account's held job and any Showcase_Entry, and nothing else (Req 12.2).
    - _Requirements: 12.1, 12.2, 12.3, 12.5, 12.6, 12.7, 12.8, 18.6_

  - [ ] 7.11 Write the unit tests for the Showcase in `tests/test_website_showcase.py`
    - `check_entry` refuses a bad Job_Id, an empty title, a title over `max_title_chars`, a title with a control character, an unknown licence, a credit over `max_credit_chars`, and a CC BY or CC BY-SA entry with no credit, the last saying the licence needs the author's credit (Req 13.3, 13.4).
    - `add_entry` refuses a Failed Job, a Job with no Job_Summary and a Done Job with no Result_File, takes a done Job started from a laptop with no job item, and refuses an entry that's already there with a pointer to `remove` (Req 13.1, 13.2).
    - `licence_line` gives the CC BY line, the CC BY-SA line with "This splat is shared under CC BY-SA 4.0.", the credit alone for another licence with a credit, and `None` with no credit. The Showcase page lists entries newest first with their lines, and a Showcase_Entry's Result_Page shows the same title, credit and licence (Req 13.6, 13.7).
    - An entry added or removed after one page load shows on the next load, with no restart of `Deps` (Req 13.5).
    - `showcase_main` outside CloudShell is refused before `make_aws` is called. `add`, `list` and `remove` work against `FakeAws`, and `list` prints each entry's Job_Id, title, licence and credit (Req 13.8, 13.9).
    - _Requirements: 13.1, 13.2, 13.3, 13.4, 13.5, 13.6, 13.7, 13.8, 13.9, 18.15_

  - [ ] 7.12 Write the unit tests for the pages in `tests/test_website_pages.py`
    - Every page has the viewport meta tag and the stylesheet, and no inline script, inline style, `style=` or `on...=` attribute (Req 6.1, 6.2).
    - A signed-in page has the CSRF meta tag, `app.js` with `defer`, the three links and the sign-out form with its token, and holds no password, Password_Hash or AWS key (Req 1.10, 4.10).
    - The start page offers the file input with its `accept` list and the three Presets with their Expected_Times, with fast checked, and after the Stop_Time shows "The website stopped starting jobs at 08:00 UTC on 4 February 2027." in place of the form (Req 7.1, 11.3).
    - The Result_Page has the Viewer_Slot with only its heading, the download link for Done and Done with warnings, the no-splat line with no link when there's no Result_File, and `data-job-id` only for Starting and Running (Req 12.7, 12.8, 12.11).
    - The 404 page links to the job list, and the 500 page says something went wrong and to try again (Req 6.7, 6.8).
    - _Requirements: 1.10, 4.10, 6.1, 6.2, 6.7, 6.8, 7.1, 11.3, 12.7, 12.8, 12.11_

  - [ ] 7.13 Write the unit tests for the handler in `tests/test_website_handler.py`
    - The headers on every kind of answer (an open route, a signed-out 303 and 401, a signed-in page, a Script_Request answer, the 404, the 500 and the 413): the Content-Security-Policy, `nosniff` and `no-referrer`, with `no-store` on every signed-in page and Script_Request answer, and the Job_Bucket's origin in `connect-src` only when signed in (Req 6.4, 6.5).
    - A body at exactly `max_body_bytes` is read and one byte more gets a 413, for plain and for base64 bodies (Req 6.6).
    - A signed-in Account's unknown path gets the 404 page (Req 6.7). A handler that raises a `CloudError` holding an ARN gets the 500 page, and neither the answer nor the log line holds the ARN or a traceback (Req 6.8).
    - A full flow of sign-in page, sign-in, job list, Result_Page, status and sign-out, with the logs captured, holds no password, Session id, CSRF_Token, cookie value or link in any log line (Req 6.9).
    - The sign-in route uses `requestContext.http.sourceIp` and ignores `X-Forwarded-For`. Signing out ends the Session, so the old cookie then gets a 303 (Req 4.10).
    - `GET /jobs/<job_id>` with a refused Job_Id, a missing Job and another Account's Job gives the same 404 body, and no recorded call holds a key built from a refused Job_Id (Req 12.12, 12.13).
    - `handler` reads only `WEBSITE_TABLE` from the environment and builds `Deps` once for two calls, with `website_aws` patched to give `FakeAws` (Req 1.6).
    - _Requirements: 1.6, 4.10, 6.4, 6.5, 6.6, 6.7, 6.8, 6.9, 12.12, 12.13, 18.6_

  - [ ] 7.14 Write the property test for Property 2 in `tests/test_website_handler.py`, with Hypothesis
    - **Property 2: A signed-out request gets nothing**
    - **Validates: Requirements 5.1, 5.2, 5.3, 5.4, 5.5, 4.9, 3.7, 3.8, 18.9**
    - For any request outside the Open_Routes, with any method, any path, any headers and any cookies, including malformed Session_Cookies, Session_Cookies of unknown Accounts, ended, expired or stale-`uid` Sessions and Sessions of removed Accounts, `handle` answers with a 303 to `/signin`, or with a 401 when the path is under `/api/`; it makes no Presigned_Link, no Stack lookup and no AWS call other than at most one `table_get_many` of the cookie's profile and session items; it makes no AWS call at all when the cookie isn't well-formed; and every answer carries the Content-Security-Policy, `nosniff` and `no-referrer` headers.
    - Draw methods from the real ones plus random text, and paths from `app.web`'s route table at test time plus random text, so task 8's routes join the draws with no change to the test. Draw cookies from none, malformed, unknown Account, ended, expired, stale-`uid` and removed-Account Sessions. Check `fake.calls` and the answer's headers.
    - _Requirements: 3.7, 3.8, 4.9, 5.1, 5.2, 5.3, 5.4, 5.5, 18.9_

  - [ ] 7.15 Write the property test for Property 6 in `tests/test_website_pages.py`, with Hypothesis
    - **Property 6: Every value in a page is escaped**
    - **Validates: Requirements 6.3, 1.10**
    - For any strings, including `<`, `>`, `&`, both quote marks, `</script>`, control characters and non-ASCII text, used as the titles, credits, file names, Account names, Readable_Errors and Job fields of every page function, the HTML holds each value only in its `html.escape(value, quote=True)` form, and `html.parser` finds in it exactly the tags and attributes that the same page has with plain values.
    - Draw `st.text()` values with hostile ones mixed in, through every page function, and list the tags and attributes with `html.parser`.
    - _Requirements: 1.10, 6.3_

  - [ ] 7.16 Write the property test for Property 9 in `tests/test_website_handler.py`, with Hypothesis
    - **Property 9: CSRF is checked, and only POSTs change anything**
    - **Validates: Requirements 4.12, 4.13**
    - For any route and any request with a Valid_Session, a POST whose CSRF_Token is missing or differs from the Session's, or from the sign-in cookie for the sign-in form, or whose `Origin` or `Sec-Fetch-Site` header shows another site, gets a 403 and leaves the table, the bucket's objects and the multipart uploads unchanged; and for any GET, whatever its path and query, the table, the objects and the multipart uploads stay unchanged.
    - Draw every POST route and every GET route from `app.web`'s route table at test time, so task 8's routes join the draws. Draw the token missing, wrong or from another Session, a cross-site `Origin` or `Sec-Fetch-Site`, and random queries. Compare the table, the objects and the multipart uploads before and after.
    - _Requirements: 4.12, 4.13_

  - [ ] 7.17 Run the tests and commit
    - Run `uv run pytest` until every test passes.
    - Stage the task's files by name, `app/static/favicon.ico` included, and commit them with the title "Website: the request handler, the pages, the job statuses and the showcase" and a body saying that it adds the standard-library handler behind the function URL, with sign-in on every page but sign-in itself, CSRF checks, strict headers, escaped pages, statuses read from the job files, and the CloudShell-only showcase command.
    - _Requirements: 18.11_

- [ ] 8. Uploads, the Start_Precheck, starting a Job and downloads
  - Files: `app/uploads.py`, `app/website_start.py`, `tests/test_website_uploads.py` and `tests/test_website_start.py` (new), `app/web.py` and `app/static/app.js` (changed).
  - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 7.9, 8.1, 9.4, 9.5, 9.6, 9.7, 9.8, 9.9, 9.10, 10.1, 10.7, 10.8, 10.10, 11.1, 11.2, 11.3, 11.4, 11.5, 11.6, 11.7, 12.9, 12.10, 12.12, 12.13, 16.3, 16.4, 16.7, 18.7, 18.10_

  - [ ] 8.1 Write the Upload record and its pure parts in `app/uploads.py`
    - `UPLOAD_ID`, `Upload`, `clean_source_name(name, max_chars)`, `parts_needed(size, part_bytes)` and `read_upload(aws, account, upload_id)`, as in the design's `app/uploads.py` section.
    - `clean_source_name` keeps the part after the last `/` or `\`, drops every character of Unicode category `Cc`, strips spaces at both ends and cuts the result to `max_chars` characters. An empty result is refused with a 400 `bad_request`. The extension check runs on the cleaned name.
    - `read_upload` reads only `account#<the Session's Account>` / `upload#<id>`, so another Account's Upload is simply not there.
    - _Requirements: 7.6, 9.6, 9.8_

  - [ ] 8.2 Write `begin`, `part_link`, `complete` and `abort` in `app/uploads.py`
    - `begin(deps, session, name, size, now)` refuses after the Stop_Time with the `stop_line` refusal, refuses a name that doesn't end in `.mp4` or `.mov` in any letter case or a stated size over `[jobs].max_video_bytes` with a Readable_Error that says what to change, draws a new 16-byte id with `deps.random.token_bytes`, starts the multipart upload at `uploads/<id>`, never at a key holding the file's name, and puts the Upload item on `ITEM_ABSENT` with its Account, stated size, cleaned name, part count, state `open` and `expires_at` `upload_record_hours` later.
    - `part_link(deps, session, upload_id, part)` makes an Upload_Link only for the Account's own `open` Upload and a part number from 1 to `parts_needed(size, upload_part_bytes)`, valid for `upload_link_seconds`. Anything else is a 404 `not_found`.
    - `complete(deps, session, upload_id, parts)` takes only the part numbers 1 to N in order, each with an `ETag` that looks like S3's quoted hex digest, completes the multipart upload, reads the object's size with `s3_size`, and sets the state to `complete`, or to `refused` with the refusal when the size is over `[jobs].max_video_bytes` or differs from the stated size.
    - `abort(deps, session, upload_id)` aborts an `open` Upload in S3 and marks it `aborted`. Each state change is conditioned on the state it comes from. Nothing here touches a `limits` item.
    - _Requirements: 7.3, 7.4, 7.6, 7.9, 10.10, 11.3_

  - [ ] 8.3 Write the refusal texts, `Started` and `precheck` in `app/website_start.py`
    - The busy, can't-start, may-have-started and copy-failed texts as module constants, word for word from the design's Error Handling table: "All GPU machines are busy right now, so your job didn't start. Try again later.", "The website can't start jobs right now.", "Your job may or may not have started. Its status below will show which." and "The video couldn't be copied for the job, so it didn't start. Press Start again."
    - `Started(job_id, outcome)`, where `outcome` is `"launched"` or `"unknown"`.
    - `precheck(deps, session, preset, now)` takes no Reservation and starts nothing: the Stop_Time; `check_preset`; `limits.check_room` for the Preset's Worst_Case_Cost; then `stack_for`, `check_step_list()`, `check_image(aws, config, deps.baked.code())` and `choose_machine(...)`, refusing an empty launch plan.
    - Its mapping: `QuotaFull` is a 409 `busy` with the busy text. Every other `CloudError` or `PipelineError` from those calls (`NoGpuQuota`, no available Machine_Image, a lock-hash drift, no Smoke_Marker, an empty Step_List, no type that fits, a missing Stack value, an empty plan) is a 503 `cant_start` with the can't-start text, after logging its own message through `account_redaction`.
    - _Requirements: 7.2, 11.1, 11.2, 11.3, 11.4_

  - [ ] 8.4 Write the checks and the Job_Record of `start` in `app/website_start.py`
    - `start(deps, session, upload_id, preset, now)`, the Website_Caller, steps 1 to 5 of the design. First the Stop_Time, `check_preset`, and the Account's own Upload, a 404 when the Account has none with that id and a 400 when it isn't `complete` or is already used.
    - Then `entries = check_step_list()`, `code = deps.baked.code()` and `gate = check_image(aws, config, code)`, so the Baked_Commit's hashes meet the Machine_Image's tags.
    - Then `check_video` with `read = lambda offset, length: aws.s3_read_range(key, offset, length)` over `uploads/<id>` and `aws.s3_size(key)`, so a gone object gives "That upload isn't there any more. Upload the video again."
    - Then the Job_Record from the Pipeline's public pieces: `splatpipe.jobs.new_job_id(now)`, the checked Preset, the whole video as the Used_Part from `used_part` with no Trim, the cleaned `source_name`, the object's size, the Video_Length and `video_file`, written with `splatpipe.jobs.write_record` into a `tempfile.TemporaryDirectory()`, which is under `/tmp` on Lambda, and removed after the upload.
    - Then `choice = choose_machine(aws, gate, entries, preset, config, cc)`. An empty launch plan is the can't-start refusal, before any Reservation. Refusals before the Reservation map as in `precheck`, and a Video_Check refusal is shown as it is.
    - _Requirements: 8.1, 9.4, 9.6, 9.7, 9.8, 9.9, 11.4, 11.5_

  - [ ] 8.5 Write the Reservation, the Spend_Guard and the launch of `start` in `app/website_start.py`
    - Steps 6 to 9 of the design: `limits.take(...)` with `worst_case_cents(preset, cc)` and `len(entries)` as the Step count; then `approve(choice, environ={aws_setting(config, "spend_variable"): "1"})`, an environment built in code for this one call and never read from the process; then `launch_job(...)` with a `place_inputs` that runs `s3_copy("uploads/<id>", "jobs/<Job_Id>/input/<video_file>", part_bytes=copy_part_bytes)`, uploads the Baked_Commit's `code.tar`, uploads `job.json` and only then sets a local `placed` flag; and on a launch, a `table_update` of the job item with `launch = "launched"`, the type and the zone.
    - Decide the Reservation's fate from what was raised and whether `placed` was set, exactly as the design's table says: nothing raised keeps it and gives `Started(job_id, "launched")`; `SpendRefused` gives it back with the can't-start text; a `CloudError` from the copy or an upload gives it back with the copy-failed text; `CapacityExhausted` or `QuotaRefused` gives it back with the busy text; any other `CloudError` keeps it, sets `launch = "unknown"` and gives `Started(job_id, "unknown")`; any other `Exception` gives it back when `placed` isn't set and keeps it with `launch = "unknown"` when it is, then raises for the 500 page.
    - Never retry, relaunch or queue a start. Each start runs the fallback loop at most once.
    - _Requirements: 9.5, 9.10, 10.1, 10.7, 10.8, 11.1, 11.6, 11.7, 16.3, 16.4_

  - [ ] 8.6 Add the upload, precheck, start and download routes to `app/web.py`
    - Script routes, all POST with JSON bodies as the design's "Script_Requests' bodies" table gives them: `/api/precheck`; `/api/uploads`, whose answer gives the Upload's id, its part count and `part_bytes`; `/api/uploads/<upload_id>/parts/<n>`; `/api/uploads/<upload_id>/complete`; `/api/uploads/<upload_id>/abort`; and `/api/jobs`, which answers `{"job": ..., "outcome": "launched"}`, or `"unknown"` with the may-have-started text. A malformed JSON body is a 400 `bad_request`.
    - `<upload_id>` must match `UPLOAD_ID` and `<n>` must be a decimal part number, or the answer is a 404.
    - The page route `GET /jobs/<job_id>/download` runs `check_job_id`, `may_see` and `job_view`, needs Done or Done with warnings and a Result_File, and answers a 303 to `s3_download_link("jobs/<Job_Id>/out/<file>", "<Job_Id>-<file>", download_link_seconds)`. Every other case is the same 404, and it never makes a link for any other key.
    - _Requirements: 7.2, 7.4, 7.6, 7.9, 12.9, 12.10, 12.12, 12.13_

  - [ ] 8.7 Add the start and upload parts to `app/static/app.js`
    - On the start page, pressing Start disables the button, sends `POST /api/precheck`, and shows its refusal or goes on to `POST /api/uploads` with the file's name and size. Every POST sends the CSRF_Token from `<meta name="csrf-token">` in `X-CSRF-Token`.
    - Then for each part in order: `POST .../parts/<n>` for a link, and `PUT` of `file.slice(a, b)` with no content type through `XMLHttpRequest`, setting no header, so the preflight asks for none. Keep each answer's `ETag`, and update the `<progress>` from the request's upload events.
    - A part whose `PUT` answers 403 gets a new link and is sent again, at most `data-part-retries` times. Then the Page_Script aborts the Upload with `POST .../abort` and shows a message.
    - Finish with `POST .../complete` with the parts and their `ETag`s, then `POST /api/jobs`, then open `/jobs/<Job_Id>`. A refusal is shown, and nothing is sent again by itself. A 401 opens `/signin`.
    - _Requirements: 7.2, 7.5, 7.7, 7.8, 11.7_

  - [ ] 8.8 Write the unit tests for uploads in `tests/test_website_uploads.py`
    - A simulated browser, through `handle` and `FakeAws`: it begins an Upload, asks for each part's link, `PUT`s each part with `fake.put_part`, gets a 403 on a link whose time has passed by the fake's `now`, asks for a new link, sends the part again, and completes. The completed object is the joined parts (Req 7.5, 7.7, 7.9).
    - The Upload's key is `uploads/<32 hex digits>` and never holds the file's name, and its item holds the Account, the stated size and the cleaned name (Req 7.4).
    - `begin` refuses an `.avi` name, a stated size over `[jobs].max_video_bytes`, and any Upload after the Stop_Time (Req 7.3, 11.3).
    - `part_link` refuses another Account's Upload, a completed or aborted Upload, part 0 and part N + 1 (Req 7.6).
    - `complete` refuses parts out of order, a missing part and an `ETag` that doesn't look like S3's, and marks the Upload `refused` when the object's size differs from the stated size or is over `[jobs].max_video_bytes`, using `object_sizes` (Req 7.9).
    - `abort` marks an open Upload `aborted` and aborts it in the fake. No upload flow writes a `limits` item (Req 10.10).
    - _Requirements: 7.3, 7.4, 7.5, 7.6, 7.7, 7.9, 10.10, 11.3_

  - [ ] 8.9 Write the unit tests for the precheck and a launched start in `tests/test_website_start.py`
    - The precheck refuses after the Stop_Time, for an unknown Preset, for a full Website_Limit, with the busy text for `QuotaFull`, and with the can't-start text for `NoGpuQuota`, no Smoke_Marker, a lock-hash drift and an empty plan, each logged with the account id redacted. It makes no `table_transact`, `s3_copy`, `s3_upload` or `run_instances` (Req 7.2, 11.1, 11.2, 11.4).
    - A launched start calls `FakeAws` in the Website_Caller's order: the Start_Core's image checks, the range reads of the Upload, the machine choice, the Reservation's `table_transact`, `check_spend`, `s3_copy` from `uploads/<id>` to `jobs/<Job_Id>/input/video.mov`, the upload of the Baked_Commit's `code.tar`, the upload of `job.json` and `run_instances` (Req 9.5, 10.1, 18.7).
    - The video is copied inside S3 and never uploaded from the function. The uploaded `job.json` reads back under `splatpipe.jobs`' rules, with the whole video as the Used_Part, the cleaned `source_name` and `video_file` (Req 9.4, 9.6, 9.7).
    - A patched `check_spend` receives an `environ` that is exactly one entry, the name from `aws_setting(config, "spend_variable")` set to `"1"`, and the process environment holds no `APPROVED_SPEND` before or after (Req 16.3).
    - The job item records the Job_Owner, the Preset, the start time, the Baked_Commit's Step count and `launch = "launched"`, and the job list shows the Job (Req 9.10).
    - A start from another Account's Upload is a 404, and a second start from the same Upload is refused by the Reservation transaction (Req 9.8).
    - _Requirements: 7.2, 9.4, 9.5, 9.6, 9.7, 9.8, 9.10, 10.1, 11.1, 11.2, 11.4, 16.3, 18.7_

  - [ ] 8.10 Write the unit tests for each start refusal in `tests/test_website_start.py`
    - One flow per row of the design's "Why a start can fail" table, each checking its shown text against Requirement 11's words, the Reservation (none taken, given back or kept), the job item's state and `launch`, and the Limit_Counters: `QuotaFull`; `CapacityExhausted` after every attempt; `QuotaRefused` from `quota_refused`; `NoGpuQuota`; no available Machine_Image; a lock-hash drift; no Smoke_Marker; an empty Step_List; no type that fits; a missing Stack value; an empty launch plan; `LimitFull`; the Stop_Time; a Video_Check refusal; a gone Upload; a `TransactionConflict`; `SpendRefused`; a failed copy; and another `CloudError` from the launch loop (Req 10.7, 10.8, 11.1 to 11.6, 9.9).
    - A kept Reservation's Job shows in the job list with the may-have-started text (Req 11.6).
    - No flow retries: each start makes at most one fallback loop of `run_instances` attempts and one Reservation `table_transact` (Req 11.7).
    - _Requirements: 9.9, 10.7, 10.8, 11.1, 11.2, 11.3, 11.4, 11.5, 11.6, 11.7_

  - [ ] 8.11 Write the unit tests for downloads in `tests/test_website_start.py`
    - A Done Job with `out/clean.sog` answers a 303 to a link for `jobs/<Job_Id>/out/clean.sog` named `<Job_Id>-clean.sog`, valid for `download_link_seconds`. A Job with only `out/train.ply` gets that file. Done with warnings works the same (Req 12.9).
    - Another Account's Showcase_Entry can be downloaded. A Running Job, a Done Job with no Result_File, another Account's Job, a missing Job and a refused Job_Id all get the same 404, and no recorded call holds a key built from a refused Job_Id (Req 12.10, 12.12, 12.13).
    - No download writes a `limits` item. After the Stop_Time the job list, the Result_Pages and the downloads still work (Req 10.10, 16.7).
    - _Requirements: 10.10, 12.9, 12.10, 12.12, 12.13, 16.7_

  - [ ] 8.12 Write the property test for Property 4 in `tests/test_website_start.py`, with Hypothesis
    - **Property 4: Nothing paid happens before the Reservation and the Spend_Guard**
    - **Validates: Requirements 16.4, 10.1, 18.10**
    - For any Website start, Preset and Upload, and any outcome at each step (a refusal at any check, a full limit, a transaction conflict, a Spend_Guard refusal, a failed copy, a capacity or quota refusal, or another launch error), the Fake_AWS's ordered call list holds no `s3_copy`, `s3_upload` or `run_instances` unless the Reservation's `table_transact` came first and then the Spend_Guard returned without raising.
    - Draw a Preset and a failure point from every row of the start table. Patch `cloud.start.check_spend` to append `("check_spend", "")` to `fake.calls` and to raise when the draw says so, as `tests/test_cloud_start.py`'s Property 6 does.
    - _Requirements: 10.1, 16.4, 18.10_

  - [ ] 8.13 Write the property test for Property 8 in `tests/test_website_start.py`, with Hypothesis
    - **Property 8: Links and Jobs reach only what the Account may see**
    - **Validates: Requirements 7.6, 12.2, 12.9, 12.10, 12.12, 12.13**
    - For any table and bucket holding several Accounts' Uploads, held and released job items and Showcase_Entries, and any request by a signed-in Account for an Upload_Link, a Result_Page, a status or a download, with any Upload id, part number and Job_Id text, malformed ones included: a link is made only for the Account's own `open` Upload and a part number from 1 to its `parts_needed`, or for the Result_File of a Job that is the Account's held job or a Showcase_Entry; every other request gets a 404 that's the same whether or not the Job exists; and no recorded AWS call holds a key built from text that `check_job_id` refuses.
    - Seed two or three Accounts with Uploads, held and released job items, Showcase_Entries and `out/` files, and draw request ids from real, other-Account, missing and malformed values.
    - _Requirements: 7.6, 12.2, 12.9, 12.10, 12.12, 12.13_

  - [ ] 8.14 Write the property test for Property 11 in `tests/test_website_uploads.py`, with Hypothesis
    - **Property 11: The kept `source_name`**
    - **Validates: Requirements 9.6**
    - For any file name text, `clean_source_name` returns text with no `/`, no `\` and no control character, at most `max_name_chars` characters long, equal to the start of what follows the name's last separator once its control characters are dropped and its outer spaces stripped; and cleaning that result again changes nothing.
    - Draw `st.text()` names with separators and control characters mixed in, and skip the draws whose cleaned result is empty, which `clean_source_name` refuses.
    - _Requirements: 9.6_

  - [ ] 8.15 Run the tests and commit
    - Run `uv run pytest` until every test passes, the properties of task 7 included, which now draw this task's routes too.
    - Stage the task's files by name and commit them with the title "Website: uploads straight to the bucket, checked starts and downloads" and a body saying that the browser sends each part straight to the bucket with short links, a start checks the video and reserves its cost before the spend guard and the shared Start_Core launch a machine, and a download is a five-minute link to one result file.
    - _Requirements: 18.11_

- [ ] 9. The Stack_Template's website parts
  - Files: `tests/snapshots/stack_kept.json` and `tests/test_website_template.py` (new), `cloud/aws-setup.yaml` (changed).
  - _Requirements: 1.7, 2.9, 7.10, 10.11, 14.1, 14.2, 14.3, 14.4, 14.5, 14.6, 14.7, 14.8, 14.9, 14.10, 14.11, 14.12, 14.13, 14.14, 14.15, 14.16, 16.5, 16.6, 18.14_

  - [ ] 9.1 Record the kept parts of today's template, before editing it
    - Read today's `cloud/aws-setup.yaml` with cfn-lint's own YAML decoder, which understands the short-form intrinsic functions, and write the `ReadPolicy`, `RunnerLaunchPolicy`, `RunnerManagePolicy`, `RunnerRole`, `AssumeRunnerPolicy`, `JobPolicy`, `JobRole` and `ProjectGroup` resources into `tests/snapshots/stack_kept.json` as sorted JSON. Task 9.6 compares the edited template against it.
    - _Requirements: 14.12, 18.14_

  - [ ] 9.2 Add the parameters, the table, the log group and the role to `cloud/aws-setup.yaml`
    - The parameters `WebsiteCodeBucket` and `WebsiteCodeKey`, both `""` by default, with the design's `AllowedPattern`s and descriptions, and the condition `HasWebsiteCode`. The design uses two code parameters where Requirement 14.1 names one Website_Code_Parameter.
    - `WebsiteTable`: on-demand, string keys `pk` and `sk`, TTL on `expires_at`, point-in-time recovery on, `DeletionPolicy: RetainExceptOnCreate` and `UpdateReplacePolicy: Retain`, the Project_Tag, and the KMS suppression with its one-line reason.
    - `WebsiteLogGroup`: `/voxelis-kiroween/${AWS::StackName}/website`, kept 30 days, the Project_Tag, and its KMS suppression.
    - `WebsiteRole`: only Lambda may assume it, the Project_Tag, and no `ManagedPolicyArns`, so nothing on the function's side refers to the policy.
    - Keep the file's style: a comment above each resource and a `Description` on each role and policy.
    - _Requirements: 10.11, 14.1, 14.4, 14.5, 14.13, 14.15_

  - [ ] 9.3 Add `WebsitePolicy` to `cloud/aws-setup.yaml`
    - An `AWS::IAM::ManagedPolicy` that attaches itself with `Roles: [!Ref WebsiteRole]`, with every statement of the design's YAML word for word: the table, the log group, `cloudformation:DescribeStacks` on this Stack only, the G_Quota, the four EC2 describes limited to this region, `RunInstances` under the Runner's launch conditions with our own tagged Machine_Image only, `CreateTags` only when launching, `iam:PassRole` of the Job_Role to EC2 only, `s3:ListBucket` with the `s3:prefix` condition on `jobs/*` and `uploads/*`, `s3:GetObject` on `jobs/*` and `uploads/*`, and `s3:PutObject` and `s3:AbortMultipartUpload` on `uploads/*` and `jobs/????????-??????-??????/input/*`.
    - No `Delete*`, `TerminateInstances`, `StopInstances`, `CreateImage`, `DeregisterImage`, `DeleteSnapshot`, `ssm:StartSession`, `iam:Get*`, `iam:List*`, `dynamodb:DeleteItem`, `dynamodb:Scan` or `dynamodb:ConditionCheckItem`. The role's `cloudformation:DescribeStacks` isn't in Requirement 14.9's list, and the design keeps it for `resolve_stack`.
    - Keep its suppression of the EC2 describe wildcard with its one-line reason.
    - _Requirements: 14.8, 14.9, 14.10, 14.11, 14.12_

  - [ ] 9.4 Add the function, its URL, the bucket's new rules, the outputs and the header numbers to `cloud/aws-setup.yaml`
    - `WebsiteFunction`: Python 3.11, x86_64, 1024 MB, 300 seconds, `WebsiteRole`, the `Handler` and `Code` from `HasWebsiteCode` with the design's inline Placeholder_Function, `WEBSITE_TABLE` as its only environment variable, `LoggingConfig` pointing at `WebsiteLogGroup`, the Project_Tag, and the suppressions for a VPC, a dead-letter queue and reserved concurrency with their reasons.
    - `WebsiteFunctionUrl` with `AuthType: NONE` and `InvokeMode: BUFFERED`, `WebsiteUrlPermission` for `lambda:InvokeFunctionUrl` with `FunctionUrlAuthType: NONE`, and `WebsiteInvokePermission` for `lambda:InvokeFunction` with `InvokedViaFunctionUrl: true`, each permission with its public-access suppression and reason.
    - On `JobBucket`, the CORS rule `website-part-uploads` allowing only `PUT` from `https://` plus the host part of `WebsiteFunctionUrl.FunctionUrl`, exposing only `ETag` and listing no `AllowedHeaders`, and after the two existing lifecycle rules, `delete-uploads-after-1-day` on the `uploads/` prefix, which expires objects and aborts unfinished uploads after 1 day.
    - The outputs `WebsiteUrl` and `WebsiteTableName`, after the seven existing ones.
    - No `DependsOn` from the function or the URL to `WebsitePolicy`, and nothing on the function's side that refers to the bucket, so the template has no cycle, as the design's "Why the template has no cycle" table shows.
    - The header comment gains the numbers the template holds: the 1-day `uploads/` rule, the 30-day log retention, the function's 1024 MB and 300 seconds, and the `website/` prefix of the bundles.
    - _Requirements: 2.9, 7.10, 14.1, 14.2, 14.3, 14.6, 14.7, 14.13, 14.14, 14.15, 16.5, 16.6_

  - [ ] 9.5 Run cfn-lint and the Compliance_Check, and fix what they find
    - Run `uv run pytest tests/test_aws_setup.py` until cfn-lint reports no error or warning, with no cfn-lint Metadata key. A cycle shows as E3004. Check that the file stays within 51,200 bytes.
    - Then run the Compliance_Check, the aws-infrastructure-as-code power's `check_cloudformation_template_compliance` tool, on the whole file. Fix each failing rule, or suppress it only when it's on the design's list of expected Unfit_Rules (the KMS rules for the table and the log group, public access to the function on both permissions, a VPC, a dead-letter queue and reserved concurrency for the function, and the EC2 describe wildcard), with a one-line reason, writing each rule id as the tool names it.
    - If the tool isn't available, or a rule fails that isn't on that list and can't be fixed, stop and ask the user before suppressing anything.
    - _Requirements: 14.15, 14.16_

  - [ ] 9.6 Write the unit tests for the template in `tests/test_website_template.py`
    - Read the template with cfn-lint's YAML decoder. The new resource types are exactly the table, the log group, the role, the managed policy, the function, the URL and two permissions, and no API Gateway, CloudFront, App Runner, load balancer or EC2 web server (Req 14.1).
    - The URL's auth type is `NONE`, and both permissions are there with their conditions. The outputs include `WebsiteUrl` and `WebsiteTableName` (Req 14.2, 14.3).
    - The table is on demand with TTL on `expires_at`, point-in-time recovery and the two keep policies. The log group keeps 30 days. The lifecycle rule's prefix is `uploads/`, with 1-day expiry and abort, and the two existing rules are unchanged (Req 7.10, 14.4, 14.5, 14.6).
    - The CORS rule allows only `PUT`, from the URL's host alone, and exposes only `ETag` (Req 14.7).
    - Every `WebsitePolicy` statement's actions are on an allow list, the launch statements carry the Runner's conditions, `PassRole` names the Job_Role for EC2 only, the reads and lists cover only `jobs/` and `uploads/`, and no forbidden action appears (Req 14.8 to 14.11).
    - The function's environment holds only `WEBSITE_TABLE`, its runtime is `python3.11`, and the Placeholder_Function answers 503 when the code parameters are empty (Req 14.13, 14.14, 16.6).
    - Every new taggable resource has the Project_Tag, the file is within 51,200 bytes, and every suppression has a reason (Req 14.15, 14.16).
    - The kept resources equal `tests/snapshots/stack_kept.json` (Req 14.12, 18.14).
    - _Requirements: 7.10, 14.1, 14.2, 14.3, 14.4, 14.5, 14.6, 14.7, 14.8, 14.9, 14.10, 14.11, 14.12, 14.13, 14.14, 14.15, 14.16, 16.6, 18.14_

  - [ ] 9.7 Write the property test for Property 13 in `tests/test_website_template.py`, with Hypothesis
    - **Property 13: The Website_Role's writes can't reach a Job's results**
    - **Validates: Requirements 14.10**
    - For any Job_Id, relative path, image id and commit id, the Website_Policy's write resources, matched the IAM way with `?` as one character and `*` as any run, match `uploads/<id>` and `jobs/<Job_Id>/input/<name>`, and match none of `jobs/<Job_Id>/out/<path>`, `work/<path>`, `jobs/_smoke/<image id>.json` and `website/<commit>.zip`.
    - Draw Job_Ids from the Job_Id pattern, relative paths, image ids and commit ids, and match them with an IAM-style matcher written in the test against the write resources read from the template, after their `${JobBucket.Arn}/` part.
    - _Requirements: 14.10_

  - [ ] 9.8 Run the tests and commit
    - Run `uv run pytest` until every test passes, `tests/test_aws_setup.py` included.
    - Stage the task's files by name and commit them with the title "Website: the website parts of the stack template" and a body saying that it adds the function with its URL, the table, the log group, a least-rights role that can only launch tagged job machines and write job inputs and uploads, one CORS rule and the uploads lifecycle rule, with no cycle and with the existing roles unchanged.
    - _Requirements: 18.11_

- [ ] 10. The Deploy_Command, the bundle and the docs
  - Files: `app/lockfile.py`, `app/deploy.py`, `tests/test_website_deploy.py` and `tests/test_website_docs.py` (new), `docs/aws-setup.md`, `.kiro/steering/rules.md`, `.kiro/steering/tech.md`, `.kiro/steering/structure.md`, `.kiro/steering/product.md`, `.kiro/skills/aws-jobs/SKILL.md`, `docs/third-party-licences.md`, `README.md`, `docs/pipeline.md` and `tests/test_website_imports.py` (changed).
  - _Requirements: 1.7, 1.8, 1.9, 3.1, 10.11, 15.1, 15.2, 15.3, 15.4, 15.5, 15.6, 15.7, 15.8, 15.9, 15.10, 15.11, 16.1, 16.2, 17.1, 17.2, 17.3, 17.4, 17.5, 17.6, 17.7, 17.8, 17.9, 17.10, 17.11, 17.12, 17.13, 17.14, 18.15_

  - [ ] 10.1 Write `app/lockfile.py`
    - `export_requirements(lock_text)` and `main(argv=None)` for `python3.11 -m app.lockfile app/uv.lock`, with the standard library only, so it runs in CloudShell before boto3 is installed.
    - It reads the lock with `tomllib` and writes one `name==version` line per registry package, with a `--hash=sha256:...` for every wheel. It skips the App_Project, whose source is `virtual` or `editable`. It refuses a lock that holds a package twice, and a package with no wheel, because the install takes wheels only. `main` prints the requirements to stdout.
    - _Requirements: 15.3_

  - [ ] 10.2 Write the guards, `spend_strings` and `pip_installer` in `app/deploy.py`
    - `Installer`, `Bundle` and `deploy_main(argv, *, environ, make_aws, repo_root, install, out, now, folder=None)`, with the design's signatures plus `folder`, which the design doesn't have: the real folder that stands for `[website].deploy_folder`, `None` meaning that folder itself. The tests pass a folder under `tmp_path`, because the laptop's temporary folder isn't `/tmp`.
    - Steps 1 to 3 of the design, in order: `require_cloudshell(environ)` before any AWS client; load `repo_root / "config.toml"` and its `[website]`, and refuse unless `repo_root` resolves inside the deploy folder; and refuse uncommitted changes under `app/`, `splatpipe/`, `cloud/` or `config.toml` with `cloud.repo.uncommitted`, naming the paths, before building anything.
    - `spend_strings(wc)` gives "website GPU jobs (up to 10 and $50 a UTC day)", "every day until 2027-02-04 08:00 UTC" and "at most $500 in total", each written from `[website]`, with whole dollars printed without cents. The design splits Requirement 15.7's one string across `check_spend`'s three fields this way.
    - `pip_installer(python)` runs `python -m pip install --target <target> --no-deps --require-hashes --only-binary=:all: --no-cache-dir --disable-pip-version-check -r <requirements>` with a timeout, and with `XDG_CACHE_HOME` and `PIP_CACHE_DIR` pointed under the deploy folder, so no cache goes in the home folder. Nothing runs a `uv sync` of the root project.
    - _Requirements: 1.9, 15.1, 15.2, 15.3, 15.4, 15.7, 15.11_

  - [ ] 10.3 Write `build_bundle` in `app/deploy.py`
    - In a fresh `<deploy folder>/build/`: `head_commit`, `archive_head` to `baked/code.tar` and `lock_hashes`, all from `HEAD`.
    - From that archive, extract `app/` without `app/uv.lock` and `app/pyproject.toml`, the top-level `cloud/*.py`, the top-level `splatpipe/*.py` and `config.toml` into the build folder, and read `cloud/aws-setup.yaml` from it as the template body, so every file comes from the commit and none from the working tree.
    - Write `baked/commit.json` with `app.baked.write_baked`. Turn the archive's `app/uv.lock` into `<deploy folder>/requirements.txt` with `export_requirements`, and `install` it into the build folder.
    - Refuse any `.so`, `.pyd` or `.dylib` file in the tree, and a tree over `max_bundle_bytes`.
    - Zip the tree with sorted entries, the fixed time 1980-01-01 00:00:00 and mode 644, so one commit always gives the same bytes. Give back the `Bundle`.
    - Add the zip's fixed date parts and mode to the allow list in `tests/test_website_imports.py`, citing the zip format's earliest date.
    - _Requirements: 15.2, 15.3, 15.5, 15.6_

  - [ ] 10.4 Write the spend check, the upload and the change set of `deploy_main` in `app/deploy.py`
    - Steps 5 to 9 of the design: `check_spend(machine=..., expected_time=..., rough_cost=..., environ=environ)` with `spend_strings(wc)`, before any AWS client, upload or change set; then `aws = make_aws(aws_setting(config, "region"))`, the Stack's outputs with `aws.stack_outputs(stack_name)` and `aws.bucket` from `BucketName`, never setting `aws.table` and making no table call; then the upload of the zip to `website/<commit>.zip`; then `create_change_set` with the request below and `change_set_changes`.
    - `change_set_request(stack_name, change_set_name, bundle, bucket, key, project_tag)` gives the design's dict exactly: `ChangeSetType` `UPDATE`, `CAPABILITY_IAM`, `DefaultVpcId` with `UsePreviousValue`, the two code parameters, the Project_Tag from `[aws].project_tag` and the description. It refuses a template body over 51,200 bytes first, with that limit and its source on the allow list in `tests/test_website_imports.py`. The change set's name is `website-<commit[:12]>-<now:%Y%m%d%H%M%S>`.
    - Print one line per change, `<Action> <LogicalResourceId> <Replacement>`, then the change set's name and the Setup_Guide's `describe-change-set` and `execute-change-set` commands with that name. Never print the change set's id, an ARN, an account id or a password.
    - The module's `__main__` block runs `run_cloudshell_cli` with `os.environ`, `cloudshell_aws`, the current folder as `repo_root`, `pip_installer(sys.executable)`, `sys.stdout` and the time now. It never prompts.
    - _Requirements: 1.7, 1.8, 10.11, 15.7, 15.8, 15.9, 15.10, 16.1, 16.2_

  - [ ] 10.5 Write the website parts of `docs/aws-setup.md`
    - A new section, "The website", as the design's "What task 10 writes in the docs" says: the CloudShell_Steps, run again in each CloudShell session because CloudShell keeps only the 1 GB home folder; deploying with `APPROVED_SPEND=1 python3.11 -m app.deploy` and reviewing and executing its change set, which also looks for a `Replacement` of `True` on `WebsiteTable`, `WebsiteFunction` or `WebsiteFunctionUrl`; making the judges' one shared Account with the Account_Command; adding and removing Showcase_Entries; the Website_Limits, the Stop_Time and the one yes; what the Website costs; clearing the Website_Jobs' `jobs/<Job_Id>/` and `work/<Job_Id>/` after judging; and taking the Website down after the Stop_Time.
    - "Updating the stack" gives two commands: before the first website deploy, today's; after it, the same with `ParameterKey=WebsiteCodeBucket,UsePreviousValue=true` and `ParameterKey=WebsiteCodeKey,UsePreviousValue=true`, so an update doesn't swap the Website for the Placeholder_Function.
    - "Removing the setup" says the function, its URL, its permissions, the Website_Role, its policy and the log group are deleted with the Stack, the table is kept with its Accounts, Limit_Counters and Showcase_Entries, and the bundles under `website/` stay in the kept bucket.
    - _Requirements: 3.1, 17.1, 17.2, 17.3_

  - [ ] 10.6 Update the steering and the AWS_Jobs_Skill
    - `.kiro/steering/rules.md`: the exception for Jobs started through the app gets its numbers, 10 Jobs and $50 a UTC_Day and $500 in total until the Stop_Time, set in `[website]`. It says the Account_Owner's `APPROVED_SPEND=1` on the Deploy_Command is the one yes for them, that a limit change needs a redeploy and a new yes, and that the Website never holds the Spend_Variable. It says the CloudShell_Commands are the Account_Owner's alone, run in CloudShell with the Admin_Login, and that Kiro never runs them.
    - `.kiro/steering/tech.md` describes the Website: one Lambda function on Python 3.11 with a function URL, one on-demand DynamoDB table, the standard library for the web parts, AWS only through `cloud/`, the App_Project with only boto3, and the Start_Core shared with `cloud.start`.
    - `.kiro/steering/structure.md` lists `app/` and the App_Project, and says the Website's packages never go in the root `pyproject.toml` or `uv.lock`.
    - `.kiro/steering/product.md` replaces "The app around the pipeline comes later. This file gets updated then." with a paragraph on signing in, uploading, picking a Preset, starting a Job, the job list, the Result_Page and its download, the Showcase and the viewer to come.
    - `.kiro/skills/aws-jobs/SKILL.md` gets the same numbers and the same one yes as the Rules_Steering, and the line that Kiro never runs the CloudShell_Commands.
    - _Requirements: 17.4, 17.5, 17.6, 17.7, 17.8, 17.9_

  - [ ] 10.7 Update the Licence_List, the README and the Pipeline_Doc
    - `docs/third-party-licences.md` updates "How we use it" for boto3, botocore, s3transfer, jmespath, python-dateutil, six and urllib3, which the Website_Bundle now ships to Lambda, and for Python, which CloudShell now runs, and adds a row for pip, MIT, if it has none. uv isn't used in CloudShell, so its row stays as it is.
    - `README.md` says the Website exists, that the Setup_Guide documents its deploy, and that the CloudShell_Commands are the Account_Owner's.
    - `docs/pipeline.md`'s "Running a job in the cloud" says the Website starts Jobs through the same Start_Core as `cloud.start`, reads a Job's Website_Status from its Job_Summary and Step_Reports and its Result_File by the names in `[clean]`, and that the Viewer_Spec decides the splat-transform version the viewer is pinned to.
    - _Requirements: 17.10, 17.11, 17.12, 17.13_

  - [ ] 10.8 Write the unit tests for the Deploy_Command in `tests/test_website_deploy.py`
    - Build a temporary git repo under `tmp_path` holding small copies of `app/`, `cloud/`, `splatpipe/`, `config.toml`, `uv.lock`, `package-lock.json` and `app/uv.lock`, committed with `git -c user.name=test -c user.email=test@example.invalid commit`, so no git config changes. Every git call passes `timeout=60` and `PYTHONIOENCODING=utf-8` and reads UTF-8. Pass a fake installer that writes a tiny fake package, `FakeAws` and an environment dict.
    - Outside CloudShell the command is refused before `make_aws` is called. A `repo_root` outside the deploy folder is refused. An uncommitted change under `app/` is refused, naming the path, before the build folder exists (Req 1.9, 15.1, 15.4).
    - The zip holds exactly `config.toml` at its root, `app/` without `pyproject.toml` and `uv.lock`, the top-level `cloud/*.py` and `splatpipe/*.py`, `baked/commit.json` with `HEAD`'s commit and lock hashes, `baked/code.tar`, and the fake package, with no test, doc or Step package. Building twice from one commit gives the same bytes (Req 15.5, 15.6).
    - A `.so` file from the installer is refused, and so is a tree over `max_bundle_bytes`. Every file the command makes is under the deploy folder (Req 15.2).
    - With `app.deploy.check_spend` patched to record its call, it runs before any `FakeAws` call. With an environment that lacks `APPROVED_SPEND`, the command stops with the Spend_Guard's refusal, holding the three strings, and makes no AWS call (Req 15.7, 16.1).
    - The bundle goes to `website/<commit>.zip`. The change set request equals the design's dict. A template body over 51,200 bytes is refused. The printed lines hold the changes, the change set's name and the two commands, and no ARN or account id. No call touches the table (Req 10.11, 15.8, 15.9, 15.10).
    - `pip_installer`, with `subprocess.run` patched, builds the design's command line with both caches under the deploy folder (Req 15.3).
    - _Requirements: 1.9, 10.11, 15.1, 15.2, 15.3, 15.4, 15.5, 15.6, 15.7, 15.8, 15.9, 15.10, 15.11, 16.1, 18.15_

  - [ ] 10.9 Write the unit tests for the docs in `tests/test_website_docs.py`
    - The way `tests/test_cloud_docs.py` checks the cloud-jobs spec's lines, check the lines each file gains: `docs/aws-setup.md`'s website section, its two "Updating the stack" commands and its "Removing the setup" lines; the numbers, the one yes and the CloudShell line in `.kiro/steering/rules.md` and `.kiro/skills/aws-jobs/SKILL.md`; the Website's lines in `tech.md`, `structure.md` and `product.md`, with "comes later" gone from `product.md`; the updated Licence_List rows; and the README and `docs/pipeline.md` lines.
    - Check that the numbers in the docs and the steering match `[website]` in the real `config.toml`, read by path, so a later change of a limit shows up here (Req 16.2).
    - `tests/test_cloud_docs.py` still passes unchanged.
    - _Requirements: 16.2, 17.1, 17.2, 17.3, 17.4, 17.5, 17.6, 17.7, 17.8, 17.9, 17.10, 17.11, 17.12, 17.13, 17.14_

  - [ ] 10.10 Write the property test for Property 14 in `tests/test_website_deploy.py`, with Hypothesis
    - **Property 14: The bundle's zip and the lock export round-trip**
    - **Validates: Requirements 15.3, 15.6**
    - For any small file tree, zipping it with the Deploy_Command's zip writer and unzipping gives back the same paths and bytes, and zipping it again gives the same zip bytes. For any generated `uv.lock` with distinct registry packages and one virtual project, `export_requirements` gives one line per registry package whose name, version and wheel hashes parse back to the lock's, and none for the project.
    - Draw random small trees of names and bytes, written under a temporary folder, and random lock files built as TOML text.
    - _Requirements: 15.3, 15.6_

  - [ ] 10.11 Run the tests and commit
    - Run `uv run pytest` until every test passes, `tests/test_cloud_docs.py` included. Deploy nothing and run no CloudShell_Command.
    - Stage the task's files by name and commit them with the title "Website: the deploy command, the bundle and the docs" and a body saying that the account owner builds the website from a clean commit under /tmp in CloudShell, gives the one yes for the website limits through the spend guard, and reviews the change set it makes, and that the docs, the steering and the skill now say so.
    - _Requirements: 18.11_

## Notes

- Every sub-task is required, the unit tests and the property tests included. No sub-task is optional.
- Each sub-task lists the acceptance criteria it serves, so every criterion of `requirements.md` traces to at least one task.
- Each task ends with its own checkpoint, which runs the whole suite and commits that task's files alone.
- The property tests check the design's 14 Correctness Properties. Properties 1 and 2, which the requirements ask for, sit in tasks 6 and 7. Properties 2 and 9 draw their routes from `app.web`'s route table at test time, so task 8's routes join them with no change to the tests.
- Where the plan reads the design in a particular way, the sub-task says so: `ScryptSettings` in `app/config.py`, the `maxmem` formula, `SecretsRandom`, `list_jobs`, `stop_line`, a conflict during a give-back, the `config_path`, `now` and `folder` keywords of the CloudShell_Commands, and the App_Project's name.

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1"] },
    { "id": 1, "tasks": ["1.2", "1.3", "1.4"] },
    { "id": 2, "tasks": ["1.5"] },
    { "id": 3, "tasks": ["1.6", "1.7"] },
    { "id": 4, "tasks": ["1.8"] },
    { "id": 5, "tasks": ["1.9"] },
    { "id": 6, "tasks": ["2.1", "2.2"] },
    { "id": 7, "tasks": ["2.3"] },
    { "id": 8, "tasks": ["2.4"] },
    { "id": 9, "tasks": ["2.5"] },
    { "id": 10, "tasks": ["2.6", "2.7"] },
    { "id": 11, "tasks": ["2.8"] },
    { "id": 12, "tasks": ["2.9"] },
    { "id": 13, "tasks": ["3.1"] },
    { "id": 14, "tasks": ["3.2"] },
    { "id": 15, "tasks": ["3.3"] },
    { "id": 16, "tasks": ["3.4"] },
    { "id": 17, "tasks": ["3.5"] },
    { "id": 18, "tasks": ["3.6"] },
    { "id": 19, "tasks": ["4.1"] },
    { "id": 20, "tasks": ["4.2", "4.3"] },
    { "id": 21, "tasks": ["4.4"] },
    { "id": 22, "tasks": ["4.5"] },
    { "id": 23, "tasks": ["4.6"] },
    { "id": 24, "tasks": ["5.1", "5.3"] },
    { "id": 25, "tasks": ["5.2", "5.4"] },
    { "id": 26, "tasks": ["5.5"] },
    { "id": 27, "tasks": ["5.6", "5.7"] },
    { "id": 28, "tasks": ["5.8"] },
    { "id": 29, "tasks": ["5.9"] },
    { "id": 30, "tasks": ["6.1"] },
    { "id": 31, "tasks": ["6.2"] },
    { "id": 32, "tasks": ["6.3"] },
    { "id": 33, "tasks": ["6.4"] },
    { "id": 34, "tasks": ["6.5"] },
    { "id": 35, "tasks": ["7.1", "7.2", "7.3", "7.4"] },
    { "id": 36, "tasks": ["7.5"] },
    { "id": 37, "tasks": ["7.6"] },
    { "id": 38, "tasks": ["7.7"] },
    { "id": 39, "tasks": ["7.8", "7.9"] },
    { "id": 40, "tasks": ["7.10", "7.11", "7.12", "7.13"] },
    { "id": 41, "tasks": ["7.14", "7.15"] },
    { "id": 42, "tasks": ["7.16"] },
    { "id": 43, "tasks": ["7.17"] },
    { "id": 44, "tasks": ["8.1", "8.3"] },
    { "id": 45, "tasks": ["8.2", "8.4"] },
    { "id": 46, "tasks": ["8.5", "8.7"] },
    { "id": 47, "tasks": ["8.6"] },
    { "id": 48, "tasks": ["8.8", "8.9"] },
    { "id": 49, "tasks": ["8.10", "8.14"] },
    { "id": 50, "tasks": ["8.11"] },
    { "id": 51, "tasks": ["8.12"] },
    { "id": 52, "tasks": ["8.13"] },
    { "id": 53, "tasks": ["8.15"] },
    { "id": 54, "tasks": ["9.1"] },
    { "id": 55, "tasks": ["9.2"] },
    { "id": 56, "tasks": ["9.3"] },
    { "id": 57, "tasks": ["9.4"] },
    { "id": 58, "tasks": ["9.5"] },
    { "id": 59, "tasks": ["9.6"] },
    { "id": 60, "tasks": ["9.7"] },
    { "id": 61, "tasks": ["9.8"] },
    { "id": 62, "tasks": ["10.1", "10.5", "10.6", "10.7"] },
    { "id": 63, "tasks": ["10.2"] },
    { "id": 64, "tasks": ["10.3"] },
    { "id": 65, "tasks": ["10.4"] },
    { "id": 66, "tasks": ["10.8", "10.9"] },
    { "id": 67, "tasks": ["10.10"] },
    { "id": 68, "tasks": ["10.11"] }
  ]
}
```
