# Requirements Document

## Introduction

Hooks that keep Kiro inside `.kiro/steering/rules.md` when someone forgets a rule. There are eight hooks: guard-paid, guard-git, guard-licence, guard-secrets, branch-check, tests-after-task, devlog and session-status. The spec also adds `scripts/hooks/run.py`, which starts every hook's script, `scripts/check_licences.py`, which checks the packages that come in as dependencies of dependencies, and `config.toml`, which holds every number and setting the scripts use.

This version takes in the user's review of 19 November 2026: the run.py wrapper, guard-secrets (Requirements 20 and 21), the git restore and checkout blocks (Requirements 8.11 to 8.16), npm's licence line, and one final task for all the live checks.

How the hooks work:

- Every hook except tests-after-task runs `scripts/hooks/run.py`, the Runner, which loads that hook's own Python module from `scripts/hooks/`. All of it uses only the standard library. tests-after-task is a prompt, as the user asked, so it has no script.
- Every hook command is `uv run --no-project -q python scripts/hooks/run.py <hook>`, plus the kind of call for a guard's entry. The same command works on Windows, macOS, Linux and Kiro Web. Measured with uv 0.12.5 on the Windows laptop on 19 November 2026: `--no-project` runs the Python 3.11 in the repo's `.venv`, and a `pyproject.toml` that uv can't parse doesn't stop it, while `--no-sync` exits 2 on one. When the script exits 2, uv exits 2. uv's own errors also exit 2, and so does Python when the script file is missing, so a missing Runner or a failing uv blocks a guarded call. An uncaught Python exception exits 1, which Kiro IDE 1.1.14 lets through, so the Runner turns every error, including one while it loads a hook's module, into that hook's exit code (Requirements 1.12 and 1.13).
- The scripts read the event JSON from stdin and look for commands in every string value, so a renamed field doesn't break them.
- Blocking means exit code 2 with a one-line reason on stderr. A live capture on Kiro IDE 1.1.14 for Windows found that exit 1 lets the call run ([hook payload reference](https://awslabs.github.io/aidlc-workflows/reference/kiro-ide-hook-payload/)), so only exit 2 counts.
- Exit 2 can only block on a trigger that comes before what it guards: Pre Tool Use, Prompt Submit, Pre Task Execution and Pre Compaction, which this spec doesn't use. Session Start and Post Task Execution come after the session has started or the task has ended, so they have nothing to block. So the guards and branch-check fail closed: they block when anything goes wrong. session-status runs only on Session Start and devlog only on Post Task Execution, and both fail open: they print a warning instead, so no failure of theirs, the Runner's or uv's can block a message or a task.
- Where the output goes: a guard's or branch-check's reason goes to the agent with the block. On Session Start, Kiro adds stdout to the agent's context on exit 0, and shows stderr to the user as a warning on any other code ([hook triggers](https://kiro.dev/docs/hooks/types/)). After a task, there may be no agent turn left to read a hook's output, so devlog also saves its warnings, and session-status shows them at the next session start.

Facts this spec relies on:

- Kiro Web runs the Prompt Submit, Agent Stop, Session Start, Pre Tool Use and Post Tool Use triggers, but no task triggers ([hook triggers](https://kiro.dev/docs/hooks/types/)).
- A Prompt Submit hook that exits with any code other than 0 blocks the user's prompt ([hook actions](https://kiro.dev/docs/hooks/actions/)). That's why session-status doesn't run on Prompt Submit.
- Kiro IDE 1.1.14 on Windows once ran no Session Start hook in a new chat. The current docs list Session Start for the IDE ([hook triggers](https://kiro.dev/docs/hooks/types/)). session-status runs on Session Start only. The final live check confirms that it fires, and if it doesn't, the user decides what to do next.
- A Pre Tool Use matcher can name a built-in category, such as `read` (all built-in file read tools) or `shell` (all built-in shell tools), a source such as `@mcp` or `@powers`, an exact tool ID or a regular expression ([hook triggers](https://kiro.dev/docs/hooks/types/)).
- The installed Kiro IDE 1.4.154 (agent extension 1.1.873) reads a Hook_Entry's matcher as one string. A matcher with no regular-expression characters other than `.`, `*` and `?` is matched against the tool's ID, its alias and each of its tags, such as `shell`, `read`, `@mcp` or `@powers`. Every matcher is also tested as a regular expression against the tool ID. A matcher with `|` in it is only a regular expression, and no tool ID holds `@`, so `@mcp|@powers` selects no tool, and one matcher can't name both sources. The `read` tag is on the file read tools such as `read_file` and `read_code`, on `list_directory` and `file_search`, and on `grep_search`, the text search. The `shell` tag is on `execute_pwsh` and the other shell tools. The final live check confirms the selections the matchers rely on.
- Kiro IDE 1.1.14 runs the Pre Tool Use hooks one after another, and runs every one even after an earlier one blocks ([hook payload reference](https://awslabs.github.io/aidlc-workflows/reference/kiro-ide-hook-payload/)). One hook run (uv, Python and the hook's imports) took 330 to 390 ms on the Windows laptop, so three shell guards take about 1 s.
- Kiro may load a Hook_File as soon as it's written, so a hook written in one task can already check the rest of the run. Kiro's docs say hooks activate when a session starts, so only a new chat is sure to load every Hook_File as written ([hook files](https://kiro.dev/docs/hooks/)).
- A Hook_Entry's `timeout` is in seconds, 60 by default, and a Hook_Entry with `"enabled": false` is skipped ([hook files](https://kiro.dev/docs/hooks/)). The eye icon next to a hook in the Agent Hooks section of the Kiro panel switches the hook off or on, and the change applies at once ([hook management](https://kiro.dev/docs/hooks/management/)).
- The AWS MCP Server has removed its `call_aws` tool. Its API calls now arrive as Python code in `aws___run_script` ([tool list](https://docs.aws.amazon.com/aws-mcp/latest/userguide/understanding-mcp-server-tools.html), [issue 293](https://github.com/aws/agent-toolkit-for-aws/issues/293)).
- In the AWS CLI, environment variables beat the settings in the config file, and only command line options beat environment variables ([precedence](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-configure.html#configure-precedence)). That applies to the region as well as to the keys.

Choices beyond the clarify answers, decided in the review on 19 November 2026:

1. Every aws command must name `--region us-east-1`. An `AWS_REGION` or `AWS_DEFAULT_REGION` environment variable beats the profile's region, so a command without `--region` could describe the older project's resources in another region.
2. guard-paid blocks Python code that calls AWS through boto3, in `python -c` and in power and MCP calls. The clarify default ignored multi-line text in power calls, but the AWS power's API calls are now multi-line Python, so ignoring them would let paid calls through. That code can't name `--profile kiroween-read`, so the reason sends Kiro to the shell.
3. The bucket prefix is `voxelis-kiroween-`. The AWS setup spec names our bucket with it.
4. devlog knows the HEAD commit from the start of each task, so that it can tell whether the task committed. devlog no longer runs at task start, so branch-check now records that commit (addition B below).
5. `rules.md` gets one line telling Kiro to report a wrong block and fix the hook instead of working around it, so that the instruction outlasts this spec.
6. `git clean` stays blocked in every form, dry runs (`-n`) included, as the user asked.
7. Every hook command runs the Runner with `uv run --no-project`. Only the guards block when their own code is broken (branch-check too, addition A). session-status and devlog never block a message or a task: they print a warning instead.
8. guard-paid's shell entry always applies the shell rules, whatever tool name the Event gives. Two more entries apply the power and MCP rules, one on `@mcp` and one on `@powers`, because a matcher in Kiro IDE 1.4.154 holds one selector. Both use Kiro's documented selectors, not a regular expression on Kiro's internal tool IDs. A tool that carries both tags gets the same check twice, with the same result.
9. About 1 s of guard time per shell command is accepted, and the guards stay separate hooks.
10. The Licence_Table gets lines for git (GPL-2.0) and npm (Artistic-2.0), both as command line tools only, never vendored, linked or shipped. Samarth OK'd both in the chat.
11. All the live checks happen in one final task, which the user runs in a new chat, so that every other task can run without the user.

Choices this update adds beyond the review, for review:

A. branch-check fails closed, like the guards, because its job is to block a task on `main`. The Start_Record it now also writes never blocks a task (Requirement 13.4).
B. branch-check writes the HEAD commit to a Start_Record when it lets a task start, and devlog reads it. branch-check already runs at task start and already blocks the task when uv or the Runner fails, so the record adds no new way to block a task. The other option, taking the commits since devlog's own previous commit, would also count the spec commit before the first task, commits made by hand between tasks, and the previous task's files whenever its devlog line wasn't committed.
C. devlog saves its warnings, and session-status shows them on a fourth line at the next session start. To stay within 4 lines, the branch and the number of uncommitted files now share the first line.
D. guard-secrets is a new, fourth guard rather than part of guard-paid, so that one hook owns the whole secrets rule, and its read entry, which runs on every file read, loads only small code. The cost: each shell call waits for a fourth hook run, about 0.35 s on top of the 1 s in item 9, and every call that the `read` matcher selects, file reads and text searches included, waits for one hook run.
E. Every file anywhere inside a folder named `.aws` counts as a Secret_File, not just `~/.aws/credentials` and `~/.aws/config`. The AWS CLI also keeps temporary credentials and SSO tokens in `~/.aws/cli/cache/` and `~/.aws/sso/cache/`, and the reviewer agent already denies `**/.aws/**`. Matching any `.aws` folder also means no spelling of the home folder can get round the rule. A path to a `.aws` folder itself counts as naming a Secret_File too, because a recursive reader such as `grep -r`, `cp -r` or `tar` reads every file in it (Requirement 20.10).
F. `.env.example` counts as a Secret_File like every other `.env.*` file, matching the reviewer agent's deny rules for `**/.env` and `**/.env.*`. The repo has no `.env.example`, and no spec plans one. `.gitignore` gains `.env.*` to match.
G. guard-secrets' read entry is the `read` matcher alone. In Kiro IDE 1.4.154, `grep_search` carries the `read` tag, so that one entry covers the file read tools and the text search, which prints lines from the files it searches. A second entry on `grep_search` would run guard-secrets twice on every text search. The `read` tag is also on tools that only print names, such as `file_search` and `list_directory`. The guard blocks only a call that names a Secret_File, so those tools add time and no new kind of block. If a later Kiro build stops tagging `grep_search` as `read`, the Live_Check fails, and the fix is an entry on `grep_search`. The guard can't tell from the Event what a tool prints, so a read call with any string value that names a Secret_File is blocked, a `grep_search` whose query is `.env` included (a known limit, see "Out of scope").
H. In a shell command, any argument that names a Secret_File counts as a read, whatever the program, because the guard can't know which programs read their arguments. So `echo .env >> .gitignore` and `cd ~/.aws` are blocked too. Kiro edits `.gitignore` with its file tools.
I. Letter case is ignored and `/` and `\` count alike in Secret_File paths on every platform, not just on Windows, so the tests expect the same results on Windows and Linux.
J. guard-git also blocks `git checkout -f` and `--force`, and `git switch -f`, `--force` and `--discard-changes`, which throw away every uncommitted change at once.
K. `git restore --staged` without `--worktree` stays allowed on folders, because it only changes the index and keeps every change in the working tree.
L. A Live_Check covers each of a hook's entries, not just the hook, so guard-paid's `@mcp` and `@powers` entries and guard-secrets' `read` entry get checked too.
M. guard-secrets also blocks `aws configure get` of names that hold `account`, `arn` or `mfa`, such as `sso_account_id`, `role_arn` and `mfa_serial`, because those settings hold the account ID, which `rules.md` keeps out of chats.

Out of scope:

- The hooks only see Kiro's own tool calls. Commands typed in a normal terminal aren't checked.
- On Kiro Web, branch-check, tests-after-task and devlog don't run. guard-git's block on committing to main covers the branch rule there.
- Kiro Web gets AWS access from an IAM role with no named profiles, so aws commands there are blocked or fail, and session-status skips its AWS line.
- If uv isn't on PATH, the hooks can't start. The shell's "not found" exit code isn't 2, so the guards then let every call through and branch-check lets every task start. On Session Start, Kiro shows the shell's error to the user as a warning. When uv itself fails (it exits 2), the guards block every shell call, guard-paid also blocks every MCP and power call, guard-secrets also blocks every call that the `read` matcher selects, and branch-check blocks every task, until it's fixed. session-status and devlog block nothing either way. Kiro's file-write tools and Kiro's Agent Hooks panel still work in both cases, so the fix needs no chat. The README says how to switch a hook off.
- A syntax error in `run.py` itself can't be caught, because Python exits 1 before it runs any of the file. Until `run.py` is fixed, every guard lets calls through and branch-check lets tasks start. The tests parse every script, so such a file fails the tests before it's committed. The Runner catches a syntax error in any other module.
- While a guard is broken, a broken Config included, it blocks every call it checks. For guard-secrets that means every call that the `read` matcher selects as well as every shell call, so Kiro can't even read the broken file. Kiro's file-write tools still work, so Kiro rewrites the broken file whole from the content it has in the chat, or the user restores the file with git in their own terminal or switches the hook off with the eye icon in the Agent Hooks panel (Requirements 1.12 and 18.3).
- The guards catch mistakes, not deliberate evasion. A command built from variables, a script file that calls AWS, code that opens a Secret_File (such as `python -c "print(open('.env').read())"`), or an encoded command gets through.
- A power or MCP server takes its AWS credentials from its own setup, not from the call. The AWS setup spec points the AWS power at `kiroween-read`.
- guard-secrets checks only the calls that its `shell` and `read` matchers select. A power or MCP tool that reads files isn't checked, unless its tool ID holds `read`, because the `read` matcher is also a regular expression on the tool ID. Kiro's file-write tools aren't checked.
- A search or command that reads a whole folder, such as a `grep_search` that names no file or `grep -r key .`, isn't blocked, because nothing in it names a Secret_File. Kiro IDE's `grep_search` runs ripgrep, which skips git-ignored files by default, and `.gitignore` lists `.env` and `.env.*`. `~/.aws/` is outside the workspace that Kiro's search tools cover.
- guard-secrets doesn't tell a path from a search query or an explanation. It judges every string value of a File_Read_Call without relying on field names, so that a renamed field can't hide a path. That's a known limit: a `grep_search` whose query is `.env` is blocked, and so is one whose query is `\.env`, which is also a Windows path to the `.env` at the root of a drive. So is a read call whose explanation holds a path to a Secret_File. To find out what `.gitignore` lists, Kiro reads `.gitignore`.
- Other aws calls can print secrets too, such as `aws sts get-session-token`, `aws sts assume-role`, `aws secretsmanager get-secret-value`, `aws ssm get-parameter --with-decryption` and `aws ecr get-login-password`. This spec doesn't block them.
- macOS gets no live check of its own. It runs the same code as Linux.
- The cloud spec writes `data/spend.jsonl`. This spec only defines the format and reads it.
- MoGe's git dependency `pipeline` has the same name as our `pipeline/` package, so the two clash on the cloud machine image. The user decided to rename our package in the next spec. This spec only lists MoGe's `pipeline` in the Allowlist (Requirement 11.4), and none of its code depends on the name of our package.

## Glossary

- **Hook_Set**: The eight hooks of this spec, their Hook_Files, the Runner and their Hook_Scripts.
- **Hook_File**: A JSON file in `.kiro/hooks/` in Kiro's v1 hook format (`{"version": "v1", "hooks": [...]}`).
- **Hook_Entry**: One item in a Hook_File's `hooks` list: a trigger, an optional matcher and an action.
- **Hook_Script**: A Python module in `scripts/hooks/` that does one hook's work. The Runner loads it.
- **Runner**: `scripts/hooks/run.py`, the only file a Launcher names. The Runner loads the hook's Hook_Script and turns every error into that hook's exit code.
- **Launcher**: The hook command `uv run --no-project -q python scripts/hooks/run.py <hook> <kind>`. `<hook>` is the hook's name, such as `guard-git`. `<kind>` is given only in a Guard's Hook_Entries, and names the kind of call that the entry checks: `shell` for Shell_Calls, `tools` for Power_Calls and MCP_Calls, and `read` for File_Read_Calls.
- **Fail_Closed_Hook**: A hook that blocks when anything goes wrong in the hook: a Guard or the Branch_Check.
- **Fail_Open_Hook**: A hook that never blocks, and prints a one-line warning when anything goes wrong in the hook: the Devlog or the Session_Status.
- **Event**: The JSON object that Kiro writes to a hook command's stdin.
- **Repo_Root**: The folder that holds `config.toml`, `pyproject.toml` and `.kiro/`.
- **Config**: `config.toml` at the Repo_Root, read with Python's built-in `tomllib`.
- **Guard_Paid**: The guard-paid Hook_Script, `scripts/hooks/guard_paid.py`.
- **Guard_Git**: The guard-git Hook_Script, `scripts/hooks/guard_git.py`.
- **Guard_Licence**: The guard-licence Hook_Script, `scripts/hooks/guard_licence.py`.
- **Guard_Secrets**: The guard-secrets Hook_Script, `scripts/hooks/guard_secrets.py`.
- **Guard**: Guard_Paid, Guard_Git, Guard_Licence or Guard_Secrets.
- **Branch_Check**: The branch-check Hook_Script, `scripts/hooks/branch_check.py`.
- **Start_Record**: A file that git ignores, in which Branch_Check records the HEAD commit when it lets a spec task start, for Devlog to read when the task finishes.
- **Tests_After_Task**: The tests-after-task hook, an agent prompt that runs after a spec task finishes.
- **Devlog**: The devlog Hook_Script, `scripts/hooks/devlog.py`.
- **Devlog_Warning**: A one-line warning about Devlog, printed by Devlog or by the Runner, and also saved in a file that git ignores, so that Session_Status can show it at the next session start.
- **Session_Status**: The session-status Hook_Script, `scripts/hooks/session_status.py`.
- **Licence_Checker**: `scripts/check_licences.py`.
- **Allowlist**: `scripts/hooks/licence_allowlist.txt`, the packages Samarth has OK'd.
- **Licence_Table**: `docs/third-party-licences.md`.
- **Spend_Log**: `data/spend.jsonl`. Each line is one JSON object with `time` (ISO 8601, UTC), `job`, `instance_id`, `instance_type` and `usd` (that line's estimated cost in US dollars).
- **Implementer**: Whoever builds this spec, Kiro or a team member.
- **Block**: Exit code 2 with a Reason on stderr. Kiro then refuses the tool call or the task.
- **Pass**: Exit code 0 with nothing on stdout or stderr. Kiro then runs the tool call or the task.
- **Reason**: The one line a Hook_Script writes to stderr when the Hook_Script blocks.
- **Paid_Reason**: A Reason that contains the sentence "this costs money, give the user the command and the rough cost, they run it".
- **Shell_Call**: A call to one of Kiro's built-in shell tools, which the hook matcher `shell` selects.
- **Power_Call**: A call to a tool of a Kiro power, which the hook matcher `@powers` selects.
- **MCP_Call**: A call to a tool of an MCP server, which the hook matcher `@mcp` selects.
- **File_Read_Call**: A call that guard-secrets' read Hook_Entry must check: a call to one of Kiro's built-in file read tools, to `grep_search`, Kiro IDE's text search, or to any other tool that the hook matcher `read` selects. In Kiro IDE 1.4.154 the `read` matcher selects all of these, because they all carry the `read` tag. The same tag is on `list_directory` and `file_search`.
- **Command**: One command inside a Shell_Call. Every part of a chain (`;`, `&&`, `||`, `|`, `&` or a new line) is a Command. So is every command nested in `$(...)`, backticks or `(...)`, every command after environment-variable prefixes (`NAME=value`, `env`), and every command inside the wrappers `bash -c`, `sh -c`, `cmd /c`, `pwsh -Command`, `powershell -Command`, `Invoke-Expression`, `iex`, the call operator `&`, `npx` and `uv run`.
- **Program**: The first word of a Command, compared without its folder, without a `.exe`, `.cmd` or `.bat` ending, and ignoring letter case.
- **AWS_Command**: A Command whose Program is `aws`.
- **Read_Call**: An AWS_Command whose operation starts with `describe-`, `list-` or `get-`.
- **Stop_Call**: `aws ec2 stop-instances` or `aws ec2 terminate-instances`.
- **S3_High_Level_Command**: `aws s3 ls`, `aws s3 cp`, `aws s3 sync`, `aws s3 mv` or `aws s3 rm`.
- **Account_Field**: A `--query` field named `OwnerId`, `Account` or `AccountId`, or a field whose name ends in `Arn`, ignoring letter case.
- **Our_Bucket**: An S3 bucket whose name starts with the bucket prefix in the Config (`voxelis-kiroween-`).
- **Secret_AWS_Command**: One of these AWS_Commands, whatever its other options: `aws configure get` of a setting whose name holds one of the secret words in the Config (`key`, `secret`, `token`, `account`, `arn` or `mfa`), ignoring letter case; `aws configure export-credentials`; or `aws s3 presign`.
- **Cloud_Command**: A Command that runs one of our cloud modules: `python -m cloud.<x>`, `python cloud/<x>.py`, or Python code passed with `-c` that imports `cloud.<x>`, with or without `uv run`.
- **Free_Cloud_Command**: A Cloud_Command whose module the Config marks free. At the start these are `cloud.list` and `cloud.stop_all`.
- **Paid_Command**: One of these Commands, whatever its other options:
  - `aws ec2 run-instances`, `start-instances`, `request-spot-instances`, `request-spot-fleet`, `create-fleet` and `create-image`
  - `aws cloudformation create-stack`, `update-stack`, `deploy` and `execute-change-set`
  - every `aws batch`, `aws ecs` and `aws sagemaker` call that isn't a Read_Call
  - `cdk deploy`, `cdk watch`, `sam deploy`, `sam sync` and `terraform apply`
  - a Cloud_Command that isn't a Free_Cloud_Command, such as `cloud.start` and `cloud.smoke_test`
- **Spend_Name**: `APPROVED_SPEND` or `kiroween-runner`, ignoring letter case.
- **Mention_Exception**: An argument that only mentions a Spend_Name or a Secret_File and can't set, use or read it: a `git commit -m` message, a search pattern for `git grep`, `git log --grep`, `git log -S`, `git log -G`, `grep`, `rg`, `findstr` or `Select-String -Pattern`, or a `pytest -k` expression, with no `$(...)` and no backticks in that argument.
- **Inline_AWS_Code**: Python code that calls AWS through `boto3`, `botocore` or the AWS MCP Server's `call_boto3`, passed with `python -c` in a Shell_Call or as a string value in a Power_Call or MCP_Call.
- **Neutral_Command**: A Shell_Call whose Commands name none of these: the Programs `aws`, `git`, `cdk`, `sam`, `terraform`, `pip`, `pip3` and `npm`, the words `uv add` and `uv pip`, a Cloud_Command, `boto3`, `botocore`, a Spend_Name or a Secret_File.
- **Staged_File**: A file that a `git commit` would add or change, following git's own rule. A commit that names no paths records the files already staged and the files that an earlier `git add` in the same Shell_Call stages. A commit that names paths records only those paths, plus the staged files when it has `-i` or `--include`. A file the commit only deletes isn't a Staged_File.
- **Folder_Pathspec**: A path argument of a git command that is `.` or `..`, is empty or ends in `/` or `\`, is a path to a folder in the working tree, is a path that doesn't exist in the working tree but that git's index lists files under (a tracked folder that was deleted), holds a wildcard (`*`, `?` or `[`), or starts with `:`, such as `:/`.
- **Checkout_Path**: An argument that `git checkout` reads as a path, by git's own rule: every argument after `--`, and, without `--`, every argument after the first when the first names a commit, branch or tag, or every argument when the first doesn't. Options and their values don't count.
- **Secret**: Text that looks like an AWS access key ID, an AWS secret access key, an AWS session token, an AWS account ID inside an ARN, the header of a private key block, or a GitHub, Hugging Face, npm or PyPI token.
- **Secret_File**: A file anywhere inside a folder named `.aws`, such as `~/.aws/credentials`, `~/.aws/config` and the files in `~/.aws/cli/cache/` and `~/.aws/sso/cache/`, or a file named `.env` or with a name that starts with `.env.`, such as `.env.local` and `.env.example`, in any folder. Names are compared ignoring letter case, and `/` and `\` count alike.
- **Home_Spelling**: A way of writing the home folder at the start of a path: `~`, `$HOME`, `${HOME}`, `$env:HOME`, `$env:USERPROFILE`, `${env:USERPROFILE}` or `%USERPROFILE%`.
- **Package_Add**: A Command that adds packages: `uv add`, `uv pip install`, `pip install`, `pip3 install`, `python -m pip install`, or `npm install`, `npm i` or `npm add` with at least one package name.
- **Allowed_Licences**: The permissive licences in `rules.md` (MIT, MIT-0, MIT-CMU, BSD-2-Clause, BSD-3-Clause, 0BSD, Apache-2.0, Apache-2.0 with the LLVM exception, ISC, PSF-2.0, matplotlib's PSF-style licence, Zlib, BSL-1.0, the Unlicense and CC0), plus their usual spellings and the matching PyPI licence classifiers, such as "MIT License", "BSD License" and "Apache Software License".
- **Place**: Where an Allowlist entry's package may be installed: `any`, `dev` (only in the dev dependency group) or `cloud` (only on the cloud machine image).
- **Blocked_Package**: `opencv-python`, `opencv-contrib-python` or `opencv-contrib-python-headless`.
- **Direct_Dependency**: A package named in `pyproject.toml` (the project dependencies, optional dependencies and dependency groups) or in `package.json` at the Repo_Root (`dependencies` and `devDependencies`).
- **Live_Check**: A check of one hook in a new Kiro chat. For each of the hook's Hook_Entries, the check makes one call or runs one task that the entry must block and one that the entry must let run, or, for a hook that never blocks, one run that shows the hook's output. An entry whose matcher selects no tool in that chat, such as `@mcp` with no MCP server set up, is reported as not checked. A new chat is the only way to be sure Kiro has loaded every Hook_File as written.
- **Live_Check_Task**: The final task of this spec, which holds every Live_Check, and which the user runs in a new chat after every other task is done.

## Requirements

### Requirement 1: Shared hook plumbing

**User Story:** As a team member, I want every hook to start, read its event and report a block the same way on every platform, so that a hook never breaks on one machine and never blocks by accident.

#### Acceptance Criteria

1. THE Hook_Set SHALL start every Hook_Script through the Runner with the Launcher, written with no shell operators, quotes or variables, so that cmd.exe, PowerShell and POSIX shells all read the same command.
2. THE Runner and the Hook_Scripts SHALL use only the standard library of Python 3.11.
3. WHEN a Hook_Script starts, THE Hook_Script SHALL read the Event from stdin as UTF-8 bytes and replace any bytes that aren't valid UTF-8.
4. IF stdin stays open longer than the stdin time limit in the Config, THEN THE Hook_Script SHALL stop reading and use the data received by then.
5. WHEN a Guard judges a Shell_Call, THE Guard SHALL treat every string value in the Event, at any depth and under any field name, as a possible Shell_Call.
6. THE Guards SHALL find every Command in a Shell_Call, as the glossary defines Command.
7. IF a Guard can't split a Shell_Call into Commands (for example because of an unclosed quote) and the Shell_Call names a Program or word that the Guard checks, THEN THE Guard SHALL block the call with a Reason that says the Guard couldn't read the command.
8. WHEN a Guard or Branch_Check blocks, THE blocking Hook_Script SHALL exit with code 2 and write exactly one Reason line to stderr, encoded as UTF-8.
9. THE Reason SHALL start with the hook's name (for example `guard-git:`), say what was blocked, and say what to do instead.
10. WHEN a Guard or Branch_Check lets a call or task run, THE Hook_Script SHALL exit with code 0 and write nothing to stdout or stderr.
11. WHEN an Event gives a hook nothing to check, THE Runner SHALL exit within 500 ms of starting, counting Python's start-up but not uv's.
12. IF a Fail_Closed_Hook hits an unexpected error (an exception, including one while the Runner loads the Hook_Script or a module the Hook_Script uses, a missing or broken Config, or an Event that isn't complete JSON), THEN THE Runner SHALL block with a Reason that names the hook, the problem and the fix. A Guard's Reason SHALL give a way out that needs no tool the Guard blocks: fix the file with a file-write tool, rewriting the file whole when the read tools are blocked too, or ask the user to switch the hook off in Kiro's Agent Hooks panel, because the Guard blocks every call that reaches the Guard until the error is fixed.
13. IF a Fail_Open_Hook hits an unexpected error, including one while the Runner loads the Hook_Script or a module the Hook_Script uses, THEN THE Runner SHALL exit with code 0 and print one line on stdout that names the hook and says what went wrong.
14. WHEN a Guard resolves a relative path, THE Guard SHALL resolve the path against the working directory the Event gives for the call, or against the Repo_Root if the Event gives none.
15. THE Runner, the Hook_Scripts and the Licence_Checker SHALL each have unit tests in `tests/` that run on the CPU with no network, using synthetic Events and temporary folders.
16. THE unit tests SHALL pass on Windows and on Linux.
17. WHEN the Runner starts, THE Runner SHALL load the Hook_Script of the hook that the Launcher names.
18. THE Runner SHALL catch every error raised while the Runner loads or runs a Hook_Script, so that the hook's failure rule in 1.12 or 1.13 decides the exit code.
19. THE Runner SHALL treat the Guards and the Branch_Check as Fail_Closed_Hooks and the Devlog and the Session_Status as Fail_Open_Hooks, and SHALL know which hook is which without reading the Config, so that a broken Config can't change how a hook fails.
20. IF the Launcher names a hook or a kind of call that the Runner doesn't know, THEN THE Runner SHALL exit with code 2 and write one line to stderr that starts with `run.py: ` and names the unknown hook or kind.
21. WHEN a call reaches a Guard through one of the Guard's Hook_Entries, THE Guard SHALL judge the call by the rules for the kind of call that the entry's Launcher names, whatever tool name the Event gives.

### Requirement 2: Settings in config.toml

**User Story:** As a team member, I want every number and setting the hooks use in one file, so that changing a limit never means editing code.

#### Acceptance Criteria

1. THE Config SHALL live at the Repo_Root as `config.toml`, and the Hook_Scripts and the Licence_Checker SHALL read the Config with `tomllib`.
2. THE Config SHALL hold the git settings: the file size limit (1,048,576 bytes), the video extensions, the other blocked extensions (`ply`, `sog`, `npz`, `pt`, `safetensors` and `ckpt`), the image extensions and the folder where images may go (`docs/`).
3. THE Config SHALL hold the AWS settings: the region (`us-east-1`), the project tag (`project=voxelis-kiroween`), the read profile (`kiroween-read`), the runner profile (`kiroween-runner`), the spend variable (`APPROVED_SPEND`), the bucket prefix (`voxelis-kiroween-`) and the Account_Field names.
4. THE Config SHALL hold the cloud settings: the cloud modules `cloud.start`, `cloud.smoke_test`, `cloud.list` and `cloud.stop_all`, and the list of free modules (`cloud.list` and `cloud.stop_all`).
5. THE Config SHALL hold the Paid_Command list, the Allowed_Licences and the Blocked_Package names.
6. THE Config SHALL hold the time limits: the stdin limit, the limit for all licence lookups in one Shell_Call, the limit for the AWS call in Session_Status, the 500 ms limit for an Event with nothing to check, and the 500 ms limit for Guard_Secrets to judge a File_Read_Call.
7. THE Runner, the Hook_Scripts and the Licence_Checker SHALL take every number and setting from the Config, except the Allowlist entries, which live in the Allowlist, the Runner's list of Fail_Closed_Hooks and Fail_Open_Hooks, which lives in the Runner (1.19), and the locations of the Start_Record (`data/hooks/task-start.json`) and the saved Devlog_Warnings (`data/hooks/devlog-warnings.txt`), which live in the Runner because the Runner saves a Devlog_Warning even when the Config is broken (15.11).
8. THE unit tests SHALL check that every licence in the Config's Allowed_Licences is on the permissive list in `rules.md`.
9. THE Config SHALL hold the secrets settings: the folder name `.aws`, the `.env` file names, the secret words that make an `aws configure get` setting secret (`key`, `secret`, `token`, `account`, `arn` and `mfa`), and the Secret_AWS_Commands.

### Requirement 3: guard-paid blocks paid commands

**User Story:** As the AWS account owner, I want Kiro stopped before it starts anything that costs money, so that I approve and run every paid job myself.

#### Acceptance Criteria

1. THE Hook_Set SHALL run Guard_Paid through three Hook_Entries: one on the `shell` matcher before every Shell_Call, one on the `@mcp` matcher before every MCP_Call, and one on the `@powers` matcher before every Power_Call.
2. WHEN a Shell_Call contains a Paid_Command, THE Guard_Paid SHALL block the call with a Paid_Reason.
3. WHEN a Shell_Call contains a Cloud_Command whose module the Config doesn't mark free, THE Guard_Paid SHALL block the call with a Paid_Reason, so that a paid cloud command added later stays blocked until someone marks that module free in the Config.
4. WHEN a Shell_Call contains Inline_AWS_Code that calls a paid AWS operation (the boto3 forms of the Paid_Commands, such as `run_instances` or `RunInstances`), THE Guard_Paid SHALL block the call with a Paid_Reason.
5. WHEN a Shell_Call contains any other Inline_AWS_Code, THE Guard_Paid SHALL block the call with a Reason that says to use the aws CLI with `--profile kiroween-read`, `--region us-east-1` and `--query`, or a Free_Cloud_Command.

### Requirement 4: guard-paid blocks the spend names

**User Story:** As the AWS account owner, I want every command that sets `APPROVED_SPEND` or uses the `kiroween-runner` profile stopped, so that only I can turn the spend guard off or use the runner profile.

#### Acceptance Criteria

1. WHEN a Shell_Call mentions a Spend_Name outside a Mention_Exception, THE Guard_Paid SHALL block the call with a Paid_Reason.
2. WHEN every mention of a Spend_Name in a Shell_Call sits inside a Mention_Exception, THE Guard_Paid SHALL judge the call by the Guard_Paid's other rules only.
3. THE Guard_Paid SHALL treat `bash -c`, `sh -c`, `cmd /c`, `pwsh -Command`, `python -c`, a write to a `.env` file, `$env:`, `set`, `setx`, `export` and `[Environment]::SetEnvironmentVariable` as outside every Mention_Exception.

### Requirement 5: guard-paid's rules for aws commands

**User Story:** As the AWS account owner, I want every aws command to use the read-only profile in us-east-1 and keep the account ID out of the chat, so that Kiro can't reach the wrong account, the wrong region or the older project's resources.

#### Acceptance Criteria

1. WHEN an AWS_Command has no `--profile kiroween-read`, or has a `--profile` that names anything else, THE Guard_Paid SHALL block the call with a Reason that says to add `--profile kiroween-read` and that `AWS_PROFILE` doesn't count.
2. THE Guard_Paid SHALL accept both `--profile kiroween-read` and `--profile=kiroween-read`.
3. WHEN an AWS_Command has no `--region us-east-1`, or has a `--region` that names anything else, THE Guard_Paid SHALL block the call with a Reason that says to add `--region us-east-1`.
4. WHEN a Read_Call has no `--query`, THE Guard_Paid SHALL block the call with a Reason that says to add a `--query` that keeps only the fields needed and no Account_Field.
5. WHEN an AWS_Command's `--query` names an Account_Field, THE Guard_Paid SHALL block the call with a Reason that says the field would put the account ID in the chat.
6. WHEN an AWS_Command lists every bucket (`aws s3 ls` without an `s3://` path, or `aws s3api list-buckets`), THE Guard_Paid SHALL block the call with a Reason that says to name an Our_Bucket.
7. WHEN an AWS_Command names a bucket that isn't an Our_Bucket, in an `s3://` path or a `--bucket` option, THE Guard_Paid SHALL block the call with a Reason that says only buckets that start with the Config's bucket prefix are ours.
8. WHEN an AWS_Command is an IAM list call (`aws iam list-*`), THE Guard_Paid SHALL block the call with a Reason that says IAM list calls show the older project's resources, and to look up our own by name with a get call and `--query`.
9. WHEN an AWS_Command is `aws sts get-caller-identity`, THE Guard_Paid SHALL block the call with a Reason that says the call prints the account ID.

### Requirement 6: guard-paid lets reads and stops run

**User Story:** As a team member, I want read-only and stopping calls to keep working, so that Kiro can check on jobs and shut machines down.

#### Acceptance Criteria

1. WHEN a Read_Call has `--profile kiroween-read`, `--region us-east-1` and a `--query` with no Account_Field, and breaks no rule in Requirement 5, THE Guard_Paid SHALL let the call run.
2. WHEN a Stop_Call has `--profile kiroween-read` and `--region us-east-1`, THE Guard_Paid SHALL let the call run.
3. WHEN an S3_High_Level_Command names at least one Our_Bucket and no other bucket, and has `--profile kiroween-read` and `--region us-east-1`, THE Guard_Paid SHALL let the call run without a `--query`.
4. WHEN a Shell_Call contains a Free_Cloud_Command, THE Guard_Paid SHALL let the call run.

### Requirement 7: guard-paid checks power and MCP calls

**User Story:** As the AWS account owner, I want the AWS power's direct API calls checked as well, so that a paid call can't go around the shell guard, while the infrastructure power's local checks keep working.

#### Acceptance Criteria

1. WHEN a Power_Call or MCP_Call has a string value that is a whole one-line AWS_Command, THE Guard_Paid SHALL judge that command by the rules for Shell_Calls in Requirements 3 to 6.
2. WHEN a Power_Call or MCP_Call has Inline_AWS_Code that calls a paid AWS operation, THE Guard_Paid SHALL block the call with a Paid_Reason.
3. WHEN a Power_Call or MCP_Call has any other Inline_AWS_Code, THE Guard_Paid SHALL block the call with a Reason that says a power can't name `--profile kiroween-read`, and to run the aws CLI in the shell with `--profile kiroween-read`, `--region us-east-1` and `--query` instead.
4. WHEN a one-line string value in a Power_Call or MCP_Call mentions a Spend_Name, THE Guard_Paid SHALL block the call with a Paid_Reason.
5. WHEN a Power_Call or MCP_Call has multi-line text that isn't Inline_AWS_Code, such as a CloudFormation template, an IAM policy, UserData or other file contents, THE Guard_Paid SHALL judge the call without that text.
6. WHEN a Power_Call or MCP_Call names a paid API only as data, such as `ec2:RunInstances` in a policy, a template or a search phrase, THE Guard_Paid SHALL let the call run.
7. WHEN the infrastructure power runs cfn-lint, cfn-guard or a documentation search on a template whose IAM policies name `ec2:RunInstances` and `kiroween-runner`, THE Guard_Paid SHALL let the call run.

### Requirement 8: guard-git blocks risky git commands

**User Story:** As a team member, I want risky git commands stopped before they run, so that nobody stages everything, rewrites shared history, commits to main or loses work by accident.

#### Acceptance Criteria

1. THE Hook_Set SHALL run Guard_Git before every Shell_Call.
2. THE Guard_Git SHALL find the git subcommand after git's own options, such as `-C <path>` and `-c <name>=<value>`.
3. WHEN a Command runs `git add` with `-A`, `--all`, `--no-ignore-removal`, `-u`, `--update` or `.`, alone or inside combined short options such as `-Av`, THE Guard_Git SHALL block the call with a Reason that says to stage files by name.
4. WHEN a Command runs `git add` with a Folder_Pathspec among its paths, such as a path to a folder, a tracked folder that was deleted, a path that holds a wildcard (`*`, `?` or `[`), or a pathspec that starts with `:`, THE Guard_Git SHALL block the call with a Reason that says to stage files by name.
5. WHEN a Command runs `git push` with `--force`, `-f`, `--force-with-lease`, `--force-if-includes`, `--mirror` or a refspec that starts with `+`, THE Guard_Git SHALL block the call with a Reason that says force pushes need the user.
6. WHEN a Command runs a `git push` that would update or delete `main` on a remote (a refspec whose target is `main`, `--all`, or no refspec while the current branch is `main`), THE Guard_Git SHALL block the call with a Reason that says to push the feat/ branch and open a pull request.
7. WHEN a Command runs `git reset --hard`, THE Guard_Git SHALL block the call with a Reason that says the command throws away work and needs the user.
8. WHEN a Command runs `git clean` in any form, dry runs included, THE Guard_Git SHALL block the call with a Reason that says the command deletes untracked files and needs the user.
9. WHEN a Command runs `git commit` while the current branch is `main`, or after an earlier Command in the same Shell_Call switches to `main`, THE Guard_Git SHALL block the call with a Reason that says to switch to a `feat/<topic>` branch.
10. WHEN a Command runs `git commit` with `-a` or `--all`, alone or inside combined short options such as `-am`, THE Guard_Git SHALL block the call with a Reason that says to stage files by name and commit again.
11. WHEN a Command runs a `git restore` that changes the working tree (one without `--staged`, or one with `--worktree`, counting `-S` and `-W` inside combined short options) with a Folder_Pathspec among its paths, THE Guard_Git SHALL block the call with a Reason that says the command throws away uncommitted work in a whole folder, and to restore files by name or ask the user.
12. WHEN a Command runs `git checkout` with a Checkout_Path that is a Folder_Pathspec, such as `git checkout -- .`, THE Guard_Git SHALL block the call with a Reason that says the command throws away uncommitted work in a whole folder, and to check out files by name or ask the user.
13. WHEN a Command runs `git checkout` with `-f` or `--force`, or `git switch` with `-f`, `--force` or `--discard-changes`, alone or inside combined short options, THE Guard_Git SHALL block the call with a Reason that says the option throws away every uncommitted change and needs the user.
14. WHEN a Command runs `git restore` with `--staged` and without `--worktree`, THE Guard_Git SHALL judge the call without the rule in 8.11, because that command only changes the index and keeps every change in the working tree.
15. WHEN a Command runs `git switch`, or a `git checkout` with no Checkout_Path, without `-f`, `--force` or `--discard-changes`, THE Guard_Git SHALL judge the call by the Guard_Git's other rules only, so that switching branches keeps working.
16. WHEN a Command runs `git add`, `git commit`, `git checkout`, or a `git restore` that changes the working tree, with `--pathspec-from-file`, THE Guard_Git SHALL block the call with a Reason that says the Guard_Git can't see the paths in that file, and to name the files in the command.

### Requirement 9: guard-git checks what a commit would record

**User Story:** As a team member, I want each commit checked for big files, media and secrets before it's made, so that nothing heavy or secret lands in git.

#### Acceptance Criteria

1. WHEN a Command runs `git commit`, THE Guard_Git SHALL check every Staged_File of that commit, using the content the commit would record.
2. IF a Staged_File is larger than the size limit in the Config, THEN THE Guard_Git SHALL block the call with a Reason that names the file and says to keep the file in `data/` or ask the user.
3. IF a Staged_File has a video extension or another blocked extension from the Config, ignoring letter case, THEN THE Guard_Git SHALL block the call with a Reason that names the file and says such files go in `data/`.
4. IF a Staged_File has an image extension from the Config and sits outside `docs/`, THEN THE Guard_Git SHALL block the call with a Reason that names the file and says images go in `docs/` or `data/`.
5. IF a line that the commit adds contains a Secret, THEN THE Guard_Git SHALL block the call with a Reason that names the file, the line number and the kind of Secret, without the Secret itself.
6. WHEN an ARN holds CloudFormation's `${AWS::AccountId}` in place of an account ID, THE Guard_Git SHALL treat that ARN as holding no Secret.

### Requirement 10: guard-licence checks packages before they're added

**User Story:** As Samarth, I want every package's licence checked before it's added, so that only permissive licences and the packages I've OK'd get in.

#### Acceptance Criteria

1. THE Hook_Set SHALL run Guard_Licence before every Shell_Call.
2. WHEN a Shell_Call contains a Package_Add, THE Guard_Licence SHALL look up the licence of every package the Package_Add names: on PyPI for Python packages and on the npm registry for npm packages, for the pinned version when the Package_Add pins one, and for the latest version otherwise.
3. WHEN the licence that the registry states for a package is in the Allowed_Licences, THE Guard_Licence SHALL let that package through.
4. WHEN the registry states the licence as an SPDX expression, THE Guard_Licence SHALL treat an `OR` expression as allowed when one of its parts is allowed, and an `AND` or `WITH` expression as allowed only when every part is allowed.
5. IF the licence that the registry states for a package isn't in the Allowed_Licences, THEN THE Guard_Licence SHALL block the call with a Reason that names the package and its licence and says to ask the user.
6. IF a lookup fails, takes longer than the lookup time limit in the Config, or finds no licence, THEN THE Guard_Licence SHALL block the call with a Reason that names the package and says why the lookup failed.
7. WHEN a package matches an Allowlist entry and the Package_Add fits that entry's Place, THE Guard_Licence SHALL let the package through without a lookup.
8. IF a Package_Add names a package whose Allowlist Place is `cloud`, THEN THE Guard_Licence SHALL block the call with a Reason that says the package goes on the cloud machine image only.
9. IF a Package_Add names a package whose Allowlist Place is `dev` and the Package_Add isn't `uv add --dev` or `uv add --group dev`, THEN THE Guard_Licence SHALL block the call with a Reason that says to add the package with `uv add --dev`.
10. IF a Package_Add names a Blocked_Package, THEN THE Guard_Licence SHALL block the call with a Reason that says all the OpenCV packages install the same cv2 module and we only use `opencv-python-headless`.
11. THE Guard_Licence SHALL compare Python package names after PEP 503 normalisation (lowercase, with every run of `-`, `_` and `.` turned into one `-`), and npm names in lowercase.
12. IF a Package_Add installs from a URL, a git repository, a local path or a requirements file, and the package's name matches no Allowlist entry whose Place fits, THEN THE Guard_Licence SHALL block the call with a Reason that says to name the package or ask the user.

### Requirement 11: The licence allowlist

**User Story:** As Samarth, I want the packages I've OK'd written down in one file with their licence and the reason, so that the hook and the licence check agree and every exception is on record.

#### Acceptance Criteria

1. THE Allowlist SHALL hold one entry per line, each with the package name as the package's metadata spells it, an optional version, the licence, the Place and the reason, and SHALL treat lines that start with `#` as comments.
2. WHEN an Allowlist entry's name ends in `*`, THE entry SHALL cover every package whose name starts with the rest of the entry's name.
3. WHEN an Allowlist entry names a version, THE entry SHALL cover only that version of the package.
4. THE Allowlist SHALL start with these entries:

   | Name | Version | Licence | Place | Reason |
   |---|---|---|---|---|
   | hypothesis | | MPL-2.0 | dev | Named exception in rules.md. Tests only. |
   | opencv-python-headless | | Apache-2.0 | any | Its wheels ship FFmpeg under LGPL-2.1, so never use cv2 to read or write video. |
   | torch | | BSD-3-Clause | cloud | Cloud machine image only. Pulls in the `nvidia-*` and `cuda-*` packages. |
   | nvidia-* | | NVIDIA's licence | cloud | torch's CUDA wheels. Named exception in rules.md. Cloud machine image only. |
   | cuda-* | | NVIDIA's licence | cloud | torch's CUDA wheels. Named exception in rules.md. Cloud machine image only. |
   | torchvision | | BSD-3-Clause | cloud | GPU package. Cloud machine image only. |
   | certifi | | MPL-2.0 | any | Named exception in rules.md. Used unmodified. Comes in through requests. |
   | tqdm | | MPL-2.0 AND MIT | any | Named exception in rules.md. Used unmodified. Comes in through huggingface-hub. |
   | gsplat | 1.5.3 | Apache-2.0 | cloud | Apache-2.0 in its repo, but its PyPI page states no licence. |
   | fused-ssim | | MIT | cloud | Installed from GitHub. Its metadata has no licence. |
   | moge | | MIT | cloud | Installed from GitHub at a pinned commit. Its DINOv2 code is Apache-2.0. |
   | utils3d-moge | | MIT | cloud | MoGe's git dependency, installed at a pinned commit. |
   | pipeline | | MIT | cloud | MoGe's git dependency, installed at a pinned commit. |
   | flex-gemm | | MIT | cloud | MoGe's git dependency, installed at a pinned commit. Its metadata spells the name `flex-gemm`. |

5. THE unit tests SHALL check that the "Named exceptions" section of `rules.md` names every Allowlist entry whose licence isn't in the Allowed_Licences.

### Requirement 12: The licence check script

**User Story:** As Samarth, I want one command that lists every installed package with its licence and fails on anything not allowed, so that dependencies of dependencies get checked too.

#### Acceptance Criteria

1. WHEN the Licence_Checker runs, THE Licence_Checker SHALL print one line for every Python package installed in the environment that runs the Licence_Checker and every npm package under `node_modules/` at the Repo_Root, with the package's name, version, licence and verdict.
2. THE Licence_Checker SHALL read each licence from the installed package's own metadata, with no network.
3. THE Licence_Checker SHALL use the same Allowed_Licences, Allowlist and Blocked_Package names as the Guard_Licence.
4. IF an installed package's licence isn't in the Allowed_Licences and no Allowlist entry covers the package, or the package's metadata states no licence, or the package is a Blocked_Package, THEN THE Licence_Checker SHALL exit with code 1 after printing every line.
5. IF a Direct_Dependency has no line in the Licence_Table whose Name column names the dependency, THEN THE Licence_Checker SHALL exit with code 1 and name the dependency.
6. IF a package whose Allowlist Place is `cloud` is installed and the Licence_Checker runs without its cloud option, THEN THE Licence_Checker SHALL exit with code 1 with a verdict that says the package goes on the cloud machine image only.
7. WHERE the Licence_Checker runs with its cloud option on the cloud machine image, THE Licence_Checker SHALL accept packages whose Allowlist Place is `cloud`.
8. IF a package whose Allowlist Place is `dev` is a project dependency in `pyproject.toml` outside the dependency groups, THEN THE Licence_Checker SHALL exit with code 1 and name the package.
9. IF `node_modules/` doesn't exist at the Repo_Root, THEN THE Licence_Checker SHALL print one line that says no npm packages are installed, and check the Python packages only.
10. WHEN every installed package is allowed and every Direct_Dependency has a line in the Licence_Table, THE Licence_Checker SHALL exit with code 0.
11. THE Licence_Checker SHALL use only the standard library of Python 3.11.
12. WHEN the Implementer runs the Licence_Checker on the laptop install at the end of the last task before the Live_Check_Task, THE Licence_Checker SHALL exit with code 0.

### Requirement 13: branch-check

**User Story:** As a team member, I want a spec task to refuse to start on main, and its starting commit recorded, so that task work always lands on a feature branch and the devlog lists only the task's own commits.

#### Acceptance Criteria

1. WHEN a spec task is about to start and the current branch is `main`, THE Branch_Check SHALL block the task with a Reason that says to switch to a `feat/<topic>` branch first.
2. WHEN a spec task is about to start and the current branch is anything other than `main`, a detached HEAD included, THE Branch_Check SHALL let the task start.
3. WHEN the Branch_Check lets a spec task start, THE Branch_Check SHALL write the HEAD commit to the Start_Record, replacing any earlier Start_Record.
4. IF the Branch_Check can't write the Start_Record, THEN THE Branch_Check SHALL still let the task start, because the Start_Record only serves the Devlog, which never holds up a task.

### Requirement 14: tests-after-task

**User Story:** As a team member, I want the tests run after every spec task, so that a failure gets fixed before the next task builds on it.

#### Acceptance Criteria

1. WHEN a spec task finishes, THE Tests_After_Task hook SHALL send the agent a prompt.
2. THE Tests_After_Task prompt SHALL tell the agent to run `uv run pytest -q`, and say that exit code 5 only means there are no tests yet, which is fine.
3. THE Tests_After_Task prompt SHALL tell the agent to fix any failure before the next task, and to tell the user when the fix changes a requirement.
4. THE Tests_After_Task prompt SHALL tell the agent that, when a property test caught a real bug, the agent adds a line to `docs/devlog.md` with the property, the smallest failing input and the fix, and describes an input that looks like a Secret by its shape instead of pasting it, so that Guard_Git doesn't block the devlog's commit.

### Requirement 15: devlog

**User Story:** As the team, I want one devlog line for every finished task, so that the write-up has a real record of how the work went.

#### Acceptance Criteria

1. THE Hook_Set SHALL run Devlog on the Post Task Execution trigger only, where exit 2 blocks nothing, so that no failure of Devlog, the Runner or uv can block a task.
2. WHEN a spec task finishes, THE Devlog SHALL add one line at the end of `docs/devlog.md` with the date and time with its UTC offset, the branch, the spec and the task when the Event carries them, and the paths of the files the task changed.
3. WHEN HEAD has moved since the commit in the Start_Record, THE Devlog SHALL take the changed files from the commits made since that commit.
4. WHEN HEAD hasn't moved since the commit in the Start_Record, or no Start_Record exists, THE Devlog SHALL take the changed files from the uncommitted changes, staged or not, new files included.
5. THE Devlog SHALL keep its line to exactly one line, whatever the branch, spec, task and file names hold.
6. WHEN `docs/devlog.md` had no uncommitted changes before the Devlog added its line and the current branch isn't `main`, THE Devlog SHALL commit only `docs/devlog.md`, with `git commit -m <message> -- docs/devlog.md`, and leave every other staged file staged.
7. IF `docs/devlog.md` already had uncommitted changes before the Devlog added its line, THEN THE Devlog SHALL add the line and leave the commit to the user.
8. WHILE the current branch is `main`, THE Devlog SHALL add its line without committing.
9. IF the Devlog's commit fails, THEN THE Devlog SHALL exit with code 0 and print one line that says the line was added but not committed.
10. WHEN the Devlog leaves its line uncommitted, for any of the reasons in 15.7 to 15.9, THE Devlog SHALL print one line that says why and save that line as a Devlog_Warning.
11. WHEN the Runner prints a warning line about Devlog (1.13), THE Runner SHALL also save that line as a Devlog_Warning.

### Requirement 16: session-status

**User Story:** As a team member, I want a short status when a chat starts, so that I see the branch, the uncommitted work, the spend, any running machine and any devlog warning without asking.

#### Acceptance Criteria

1. THE Hook_Set SHALL run Session_Status on the Session Start trigger only, where exit 2 blocks nothing, so that no failure of Session_Status, the Runner or uv can block a message.
2. WHEN the Session_Status prints the status, THE Session_Status SHALL print at most 4 lines: the current branch with the number of uncommitted files, the estimated spend when the Spend_Log exists, the running EC2 instances with our project tag when AWS is set up, and the warnings when there are any.
3. WHEN the Session Start trigger fires, THE Session_Status SHALL print the status.
4. WHEN Devlog_Warnings are saved, THE Session_Status SHALL put the number of saved Devlog_Warnings and the most recent one on the warnings line, and then delete the saved Devlog_Warnings.
5. THE Session_Status SHALL exit with code 0 in every case that the Runner can catch, because on Session Start Kiro adds stdout to the agent's context only on exit 0.
6. THE Session_Status SHALL count as uncommitted files every path that `git status --porcelain` lists: staged, unstaged and untracked.
7. WHEN the Spend_Log exists, THE Session_Status SHALL print the sum of `usd` over all the Spend_Log's lines and over the lines whose `time` falls on today's date in UTC, each rounded to the cent.
8. THE Session_Status SHALL skip every Spend_Log line that isn't a JSON object with an ISO 8601 `time` and a numeric `usd`.
9. THE Session_Status SHALL find the running instances with the aws CLI (not boto3): `aws ec2 describe-instances --profile kiroween-read --region us-east-1`, filtered on our project tag and on the `pending` and `running` states, with a `--query` that keeps only `InstanceId`, `InstanceType`, `State` and `LaunchTime`.
10. IF the aws CLI is missing, the `kiroween-read` profile is missing, or the AWS call fails or takes longer than the AWS time limit in the Config, THEN THE Session_Status SHALL leave out the instance line and print none of the aws CLI's output.
11. WHEN the AWS call succeeds and finds no instances, THE Session_Status SHALL print the instance line with the word `none`.
12. THE Session_Status SHALL put every running instance on the one instance line.

### Requirement 17: Hook files, build order and live checks

**User Story:** As a team member, I want each hook added only after its script works, and checked live, so that a half-finished hook never blocks us.

#### Acceptance Criteria

1. THE Hook_Set SHALL have these Hook_Files in `.kiro/hooks/`, one per hook:
   - guard-paid: Pre Tool Use, with one Hook_Entry on `shell`, one on `@mcp` and one on `@powers`
   - guard-git: Pre Tool Use on `shell`
   - guard-licence: Pre Tool Use on `shell`
   - guard-secrets: Pre Tool Use, with one Hook_Entry on `shell` and one on `read`
   - branch-check: Pre Task Execution
   - tests-after-task: Post Task Execution, with an agent prompt action
   - devlog: Post Task Execution only
   - session-status: Session Start only
2. THE timeout in each Hook_File SHALL be longer than the longest time limit in the Config for the Hook_Script that the Hook_File runs.
3. THE unit tests SHALL check every Hook_File's triggers, matchers and timeouts, that every command action is the Launcher for a hook the Runner knows, with the kind of call that fits the Hook_Entry's matcher, and that the tests-after-task prompt holds the instructions in Requirement 14.
4. THE Implementer SHALL write each Hook_File only after the commit that adds the Hook_Script the Hook_File runs, with that script's unit tests passing, because Kiro may load a Hook_File as soon as the Hook_File is written and apply the Hook_File to the rest of the run.
5. THE Implementer SHALL put every Live_Check in the Live_Check_Task, so that every other task of this spec runs without the user.
6. WHEN the user runs the Live_Check_Task in a new chat, THE Implementer SHALL run each hook's Live_Check on the Windows laptop in Kiro IDE, run the Live_Checks of guard-paid, guard-git, guard-licence, guard-secrets and session-status in Kiro Web, confirm that uv is on PATH in Kiro Web, that uv finds a Python 3.11 there and that the unit tests pass there, and add every result to `docs/devlog.md`.
7. IF a hook blocks a call or task that the rules allow, THEN THE Implementer SHALL tell the user and fix the hook, instead of working around the block.
8. IF the Session Start trigger doesn't fire in session-status's Live_Check, THEN THE Implementer SHALL tell the user, record the result in `docs/devlog.md` and leave session-status on Session Start only, so that the user decides what to do next.
9. THE unit tests SHALL check that every Hook_Entry that runs a Fail_Open_Hook sits on the Session Start or the Post Task Execution trigger, where exit 2 blocks nothing, and on no other trigger.

### Requirement 18: Docs and steering

**User Story:** As the team, I want the docs and the steering to describe the hooks, so that everyone knows what the hooks need and what they block, and the write-up has a record.

#### Acceptance Criteria

1. THE Licence_Table SHALL gain lines for git (GPL-2.0), npm (Artistic-2.0), the AWS CLI v2 (Apache-2.0) and uv (MIT or Apache-2.0), each a separate command line tool that we never vendor, link or ship. Samarth OK'd the git and npm lines in the chat on 19 November 2026, because GPL-2.0 and Artistic-2.0 are outside the Allowed_Licences and `rules.md` says to ask him before adding any other licence.
2. THE `rules.md` steering file SHALL gain a line that says: if a hook blocks something it shouldn't, say so and fix the hook, and never work around a hook or turn one off without the user's OK.
3. THE README SHALL say that the hooks need uv on PATH, that guard-licence in Kiro Web needs internet access to `pypi.org` and `registry.npmjs.org`, how to run the Licence_Checker, how to turn a hook off, and how to get out when a broken guard blocks every call it checks, every file read and text search included for guard-secrets, without a tool that the guard blocks.
4. THE `docs/kiro-usage.md` file SHALL gain a Hooks section that lists each hook, its triggers and matchers, what it checks, and whether it fails closed or open.

### Requirement 19: Rules that hold for every input

**User Story:** As a team member, I want the most important rules checked against many generated inputs, so that an unusual spelling of a command can't slip past a guard.

#### Acceptance Criteria

1. THE Guard_Paid SHALL block every Shell_Call that sets `APPROVED_SPEND` or uses the `kiroween-runner` profile, in any letter case, in any Command of a chain, wrapper or subshell, and in any form: `NAME=value cmd`, `env`, `export`, `set`, `setx`, `$env:`, `[Environment]::SetEnvironmentVariable`, `--profile kiroween-runner`, `AWS_PROFILE=kiroween-runner`, `profile_name="kiroween-runner"`, `bash -c`, `python -c` or a write to a `.env` file.
2. THE Guard_Paid SHALL let every Read_Call, Stop_Call and S3_High_Level_Command run that has `--profile kiroween-read` and `--region us-east-1`, names no bucket other than Our_Buckets, isn't an IAM list call, `aws s3api list-buckets` or `aws sts get-caller-identity`, has a `--query` with no Account_Field when the command is a Read_Call, and names at least one Our_Bucket when the command is an S3_High_Level_Command, whatever the order of the options, the `--option value` or `--option=value` spelling and the quoting.
3. THE Guard_Paid SHALL block every AWS_Command that has no `--profile kiroween-read`, including AWS_Commands that set `AWS_PROFILE=kiroween-read` instead.
4. THE Guard_Git SHALL let every `git add` run whose arguments are only paths to files (no folders, wildcards or pathspecs that start with `:`), with or without `--`, in any quoting.
5. THE Guard_Paid, the Guard_Git, the Guard_Licence and the Guard_Secrets SHALL let every Neutral_Command run.
6. THE Guard_Paid SHALL block every Paid_Command, whatever its options, the letter case of its Program and its place in a chain, wrapper or subshell.
7. THE Guards and the Branch_Check SHALL make the Reason for every blocked call or task exactly one line that starts with the hook's name.
8. THE Guard_Git SHALL block every commit in which an added line holds a generated AWS access key ID or private key header, wherever the Secret sits in the line.
9. THE Guard_Licence SHALL block every spelling of a Blocked_Package name that PEP 503 normalises to that name.
10. THE Guard_Licence SHALL judge every SPDX expression built from `OR`, `AND` and `WITH` by the rule in Requirement 10.4.
11. THE Session_Status SHALL print, for every Spend_Log that mixes valid and broken lines, the sum of `usd` over the valid lines only.
12. THE Devlog SHALL write exactly one line for every branch, spec, task and file name, including names that hold line breaks.
13. THE unit tests SHALL check each criterion in this requirement with a Hypothesis property test, and the property test tasks SHALL be required, never optional.
14. THE Guard_Secrets SHALL block every Shell_Call with a Command that names a Secret_File in an argument, in the part of an argument after `=` or `:`, or in an input redirect, outside a Mention_Exception, and every File_Read_Call with a string value that names a Secret_File, whatever the path's spelling (a Home_Spelling, an absolute path, a relative path from the Event's working folder or an earlier `cd`, `/` or `\`, `.` and `..` parts, any quoting and any letter case) and the Command's place in a chain, wrapper or subshell.
15. THE Guard_Secrets SHALL block every Secret_AWS_Command, whatever its other options, `--profile kiroween-read` and `--region us-east-1` included, the letter case of the setting's name, and the Command's place in a chain, wrapper or subshell.
16. THE Guard_Secrets SHALL let every File_Read_Call run whose string values name no Secret_File.
17. THE Guard_Git SHALL let every `git restore` and `git checkout` run whose paths are only paths to files (no Folder_Pathspec), with or without `--`, a source commit, `--staged` and `--worktree`, in any quoting, when the Command has no `-f` or `--force`.
18. THE Guard_Git SHALL let every `git checkout <branch>`, `git checkout -b <branch>`, `git switch <branch>` and `git switch -c <branch>` run that has no `-f`, `--force` or `--discard-changes`, for every valid branch name that names no folder in the working tree and no path that git's index lists files under.
19. THE Runner SHALL exit with code 0 whenever the Runner runs Devlog or Session_Status, for every stdin (any bytes, or none), including when the Hook_Script or a module the Hook_Script uses fails to load, the Config is missing or broken, a git call fails, or a file the Hook_Script reads can't be read.

### Requirement 20: guard-secrets blocks reads of secret files

**User Story:** As the AWS account owner, I want Kiro stopped before it reads a file that can hold keys, through the shell or a file tool, so that no key or token reaches the chat.

#### Acceptance Criteria

1. THE Hook_Set SHALL run Guard_Secrets before every Shell_Call, through a Hook_Entry on the `shell` matcher, and before every File_Read_Call, through a second Hook_Entry on the `read` matcher.
2. WHEN a Command in a Shell_Call names a Secret_File in an argument, in the part of an argument after `=` or `:` (such as `--env-file=.env` or `HEAD:.env`), or in an input redirect (`<`), outside a Mention_Exception, THE Guard_Secrets SHALL block the call with a Reason that names the file, says the file can hold keys, and says to ask the user for what's needed from the file.
3. WHEN a File_Read_Call has a string value, at any depth and under any field name, that names a Secret_File, THE Guard_Secrets SHALL block the call with a Reason that names the file, says the file can hold keys, and says to ask the user for what's needed from the file.
4. THE Guard_Secrets SHALL count a path as naming a Secret_File whatever the path's spelling: a Home_Spelling, an absolute or a relative path, `/` or `\` between the parts, `.` and `..` parts, any quoting, and any letter case.
5. WHEN a Command in a Shell_Call names a relative path after an earlier `cd`, `Set-Location`, `sl`, `chdir`, `pushd` or `Push-Location` in the same Shell_Call, THE Guard_Secrets SHALL resolve the path against the folder that the earlier Command moved to, including a folder written with a Home_Spelling.
6. WHEN the last part of a path holds a wildcard (`*`, `?` or `[`) and starts with `.env`, such as `.env*`, THE Guard_Secrets SHALL count the path as naming a Secret_File.
7. WHEN a File_Read_Call reaches the Guard_Secrets, THE Guard_Secrets SHALL decide within the File_Read_Call limit in the Config (500 ms), counting Python's start-up but not uv's, without starting another process or using the network.
8. WHEN a Shell_Call or a File_Read_Call names no Secret_File and holds no Secret_AWS_Command, THE Guard_Secrets SHALL let the call run.
9. THE Implementer SHALL add `.env.*` to `.gitignore` next to `.env`, so that git never stages a Secret_File in the repo.
10. WHEN any part of a path is `.aws`, the last part included, such as `~/.aws` in `cd ~/.aws`, THE Guard_Secrets SHALL count the path as naming a Secret_File, because a recursive reader such as `grep -r`, `cp -r` or `tar` reads every file inside the `.aws` folder.

### Requirement 21: guard-secrets blocks aws commands that print secrets

**User Story:** As the AWS account owner, I want the aws commands that print keys, tokens or the account ID, or that make signed links, stopped even with the read-only profile, so that none of these reach the chat.

#### Acceptance Criteria

1. WHEN a Command runs `aws configure get` of a setting whose name holds one of the Config's secret words, in any letter case, THE Guard_Secrets SHALL block the call with a Reason that says the command prints a key, a token or the account ID from the AWS config files, and to ask the user for what's needed.
2. WHEN a Command runs `aws configure export-credentials`, THE Guard_Secrets SHALL block the call with a Reason that says the command prints the profile's keys.
3. WHEN a Command runs `aws s3 presign`, THE Guard_Secrets SHALL block the call with a Reason that says a presigned link holds the access key ID and opens the object to anyone who has the link, and that the user makes such links.
4. THE Guard_Secrets SHALL block every Secret_AWS_Command whatever its other options, `--profile kiroween-read` and `--region us-east-1` included, and wherever aws's own options such as `--profile` and `--region` sit in the Command.
