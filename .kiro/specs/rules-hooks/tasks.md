# Implementation Plan: rules-hooks

## Overview

This plan builds the eight hooks, the Runner (`scripts/hooks/run.py`), `scripts/check_licences.py` and `config.toml` in the design's build order. `config.toml`, the Event reader and the Runner come first, then the licence-table rows, the `rules.md` hooks line, the `.env.*` line in `.gitignore` and the shell splitter. After that each hook gets its script, the script's unit and property tests, a checkpoint and its hook file, in that order, so a hook file is written only after its script and tests are committed and passing (Requirement 17.4). The docs and a final run of the whole suite and the licence check follow. The last task holds every live check, and the user runs it in a new chat (17.5). The code is Python 3.11 with only the standard library in `scripts/`. The tests use pytest and Hypothesis.

Working rules for every task:

- Every task except the last runs without the user. No task waits for Samarth, Kiro Web, a new chat or a throwaway clone. Samarth OK'd the git and npm licence rows in the chat on 19 November 2026. A checkpoint needs no input either: run `uv run pytest -q` and fix what fails. Stop for the user only when a fix would change a requirement.
- Work on `feat/hooks`, never on `main`. Run `uv run pytest -q` before each commit. Exit code 5 only means there are no tests yet.
- Finish every task with a commit. Stage files by name with `git add <path> ...`, never with `git add -A`, `git add .` or a folder. Commit only the task's own files with `git commit -m "<message>" -- <path> ...`, so a task never picks up files that another task staged.
- Kiro may load a hook file as soon as it's written, so a hook file written during the run may already be active for the rest of it. Every task after a hook file must already comply with it:
  - After guard-paid's file: no spend name in a shell command or tool-call value outside a Mention_Exception, and no aws command. No task needs one.
  - After guard-secrets' file: no Secret_File path in a shell command outside a Mention_Exception, and none in any value of a read call, its path, pattern, query and explanation included. Describe such files in words, and read `.gitignore` to see what it lists. Every file read and text search now waits for one hook run.
  - After guard-git's file: stage and commit as above. guard-git also blocks `git clean`, `git reset --hard`, force pushes, and restoring or checking out a whole folder.
  - After guard-licence's file: add no packages. No task needs one.
  - After branch-check's, tests-after-task's and devlog's files: a task can't start on `main`, the tests run after each task, and devlog commits one line to `docs/devlog.md` after each task. Leave `docs/devlog.md` out of a task's own commit unless the task changed it on purpose.
- The spend names are `APPROVED_SPEND` and `kiroween-runner`, in any letter case. The Secret_Files are `.env`, `.env.*` and anything in a `.aws` folder. Keep both out of file names, test names, fixture paths and shell commands, except inside a Mention_Exception (a `git commit -m` message, a search pattern or a `pytest -k` expression). Give parametrized tests `ids` that hold neither, so no pytest node ID does. The guards read every string in a tool call, a shell call's description included, so describe calls in plain words, without command text, spend names or Secret_File paths. Write files with the file tools, never through the shell.
- Every test task is required. None is optional (rules.md, Requirement 19.13).
- Tests run on the CPU with no network and must pass on Windows and Linux (1.15, 1.16). The registry lookup is an injected fake `Lookup`. git checks use a temporary repo or an injected fake `GitRunner`. The aws CLI is an injected fake `AwsRunner`, and no test runs the real one. Secrets are joined from pieces at run time, so no test file holds a literal key, token, private key header or 12-digit account ARN. Secret_Files exist only as strings or as files in `tmp_path`.
- A test that runs the Runner as a subprocess runs a copy of `scripts/` and `config.toml` in `tmp_path` with `sys.executable`. `AWS_CONFIG_FILE` and `AWS_SHARED_CREDENTIALS_FILE` point at empty files there, and `HOME` and `USERPROFILE` at a temporary folder, so no test writes the real `data/` and none can reach AWS.
- Every property test goes in `tests/test_properties.py`, with `@settings(max_examples=100, deadline=None)` or more and the comment `# Feature: rules-hooks, Property N: <title>`. Shared strategies go in `tests/hook_strategies.py`.
- Scripts import only the standard library. `urllib.request`, `ssl`, `importlib.metadata`, `difflib` and `subprocess` are imported only inside the functions that use them (1.2, 1.11). Every number and setting comes from `config.toml`, except the Allowlist entries and what Requirement 2.7 leaves in the Runner.
- Every hook module has `main(kind, stdin=None, root=None)`, which the Runner calls with the kind of call, the stdin stream and the Repo_Root. It returns `hookio.guard_main(..., stdin, root)` for a Guard or Branch_Check, and `hookio.quiet_main(..., stdin, root)` for Devlog and Session_Status. The Runner is the only file a Launcher names.
- Create hook files with Kiro's `createHook` tool where it's available. It writes one entry per file, so extra entries are added by editing the JSON afterwards. Each command is the Launcher, `uv run --no-project -q python scripts/hooks/run.py <hook>`, followed by the kind for a guard's entry. Before committing a hook file, check it against the design's "Hook files" table, with the timeout on each hook entry.
- If a hook blocks something the rules allow, tell the user and fix the hook with its tests. Never work around a block, and never turn a hook off without the user's OK (17.7).
- If a hook blocks every call or every task, Kiro's file-write tools still work, because they match no hook's matcher. Fix `config.toml` or the script with a file edit, and run the tests once the shell answers again. When guard-secrets or `config.toml` is the broken part, the read tools are blocked too, so rewrite the broken file whole with `fs_write`, from the content you have in the chat, instead of reading it first. Only if that fails, stop and ask the user to restore the file with git in their own terminal, or to switch the hook off with the eye icon in Kiro's Agent Hooks panel, which applies at once.
- Nothing here costs money. No task sets `APPROVED_SPEND`, uses `kiroween-runner` or calls AWS.

## Tasks

- [ ] 1. Set up `config.toml`, the Event reader and the Runner
  - [ ] 1.1 Create `config.toml` and the `scripts` packages
    - Write `config.toml` at the repo root exactly as the design's "config.toml" section gives it: `[hooks]`, `[git]` with its `[[git.secrets]]` patterns as TOML literal strings, `[aws]`, `[secrets]`, `[cloud]`, `[paid]`, `[session_status]`, `[devlog]`, and `[licences]` with `[[licences.allowed]]` and `[[licences.allowed_exceptions]]`.
    - Create empty `scripts/__init__.py` and `scripts/hooks/__init__.py`, so the tests and the Runner can import `scripts.hooks.<module>`. `pyproject.toml` already sets `pythonpath = ["."]`.
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.9_

  - [ ] 1.2 Implement `scripts/hooks/config.py`
    - `repo_root()`, `load(root=None) -> Config` and `ConfigError`, as in the design's "config: loading config.toml" section.
    - Read the file with `tomllib` in binary mode, check that every key exists with the right type, and compile each `[[git.secrets]]` pattern. Return a frozen `Config` dataclass with one attribute per table.
    - A missing file, a TOML error, a missing or wrong-typed key, or a pattern that doesn't compile raises `ConfigError` with the key's dotted name.
    - _Requirements: 1.12, 2.1, 2.7_

  - [ ] 1.3 Add the first shared test helpers
    - In `tests/conftest.py`, add an autouse fixture that replaces `socket.socket.connect` with a function that fails the test, and a fixture that writes a modified copy of `config.toml` into `tmp_path` and returns that folder as the root.
    - Write `tests/hook_events.py` with builders for synthetic Events that use Kiro's field names (`hook_event_name`, `session_id`, `cwd`, `tool_name`, `tool_input`): a shell call, a power or MCP call, a read call, a task start or end, and a session start.
    - _Requirements: 1.15, 1.16_

  - [ ] 1.4 Write unit tests for the Config in `tests/test_config.py`
    - The real `config.toml` loads and holds the exact values of Requirements 2.2 to 2.6 and 2.9.
    - A missing key, a wrong type and a pattern that doesn't compile each raise `ConfigError` naming the key.
    - Every `rules_name` in `[[licences.allowed]]` and `[[licences.allowed_exceptions]]` appears in the permissive list in `.kiro/steering/rules.md`.
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.8, 2.9_

  - [ ] 1.5 Implement `scripts/hooks/hookio.py`
    - `Event`, `EventError`, `read_stdin`, `parse_event`, `dialects`, `Verdict`, `Result`, `allow`, `block`, `one_line`, `Context`, `guard_main` and `quiet_main`, as in the design's "hookio" sections.
    - `read_stdin` reads bytes on a daemon thread until EOF or `[hooks].stdin_seconds`. `parse_event` decodes UTF-8 with `errors="replace"`, collects every string value with an explicit stack, and finds the tool name, `cwd`, session ID and event name by normalised key. The deepest non-empty `cwd` wins, and a relative one resolves against the Repo_Root.
    - `dialects` maps the tool name to the shell readings of design decision 3. It never decides which rules apply, because the Launcher's kind does.
    - `one_line` turns every run of control characters, including every character `str.splitlines()` breaks on, into one space.
    - `guard_main` loads the Config, reads stdin, parses the Event, builds the Context and calls the judge for the kind. A Pass is `Result(0)` and a Block is `Result(2, err=reason)`. A `ConfigError` or `EventError` becomes a Block that names the hook, the problem and the fix, and gives the design's way out: fix the file with a file-write tool, rewriting it whole if the read tools are blocked too, or ask the user to restore it with git or switch the hook off in the Agent Hooks panel. `quiet_main` turns any `Exception` into `Result(0, out=<one line>)`. Both take the stdin, the Repo_Root and a test's Context from their caller, and build the real Context's git, lookup and aws parts only on first use.
    - _Requirements: 1.3, 1.4, 1.5, 1.8, 1.9, 1.10, 1.12, 1.13, 1.14, 1.21_

  - [ ] 1.6 Implement `scripts/hooks/run.py`, the Runner
    - The hook table `HOOKS` with each hook's module, whether it fails closed or open, and the kinds it takes, as in the design's Runner table. The Runner never reads the Config to know which hook is which.
    - The state-file locations `STATE_DIR`, `START_RECORD` and `DEVLOG_WARNINGS`, and `save_devlog_warning(root, line)`, which appends one line, creates the folder when needed and ignores its own errors.
    - `main(argv, stdin=None, root=None, importer=importlib.import_module)` follows steps 1 to 7 of the design's Runner section: check the arguments, check the Python version, put the Repo_Root on `sys.path`, capture stray output, import the module and call its `main` inside one `try` that catches `BaseException`, turn a failure into the hook's exit code, check the Result against the hook's rule, save every devlog stdout line with `save_devlog_warning`, and write the output as UTF-8 to `sys.__stdout__.buffer` or `sys.__stderr__.buffer`.
    - Call the hook's `main(kind, stdin, root)`: `kind` from the Launcher, or `None` for a hook that takes none, the `stdin` the Runner was given (`None` means the real stdin), and the `root` a test passed, or else the Repo_Root two folders above `run.py`. Leave `os._exit` to the `if __name__ == "__main__":` block, so tests can run `main` in-process.
    - A fail-closed hook's Result is exit 0 with no output, or exit 2 with one stderr line. A fail-open hook's Result is exit 0 with nothing on stderr, and devlog's has at most one stdout line. Session-status's status of up to four lines passes through as it is, because the Runner never reads the Config.
    - The failure Reason of a fail-closed hook gives the design's way out: fix the file with a file-write tool, rewriting it whole if the read tools are blocked too, or ask the user to switch the hook off in the Agent Hooks panel.
    - The Runner doesn't import `hookio`. It reads the Result's `code`, `out` and `err`, and keeps its own `_one_line`, so a broken `hookio` is just another failure of the hook. `run.py` must parse and import on Python 3.8, so it can report an old Python.
    - _Requirements: 1.2, 1.8, 1.10, 1.11, 1.12, 1.13, 1.17, 1.18, 1.19, 1.20, 2.7, 15.10, 15.11_

  - [ ] 1.7 Write unit tests for the Event reader in `tests/test_hookio.py`
    - Add to `tests/conftest.py` a `ctx` factory that builds a `Context` with the real Config, a Repo_Root of the test's choice (the real one, or a folder in `tmp_path` for tests that write state), a fixed clock, and `git`, `lookup` and `aws` stubs that fail the test if they're called. Add a helper that copies `scripts/` and `config.toml` into `tmp_path` and runs the copy's `run.py` as the working rules say.
    - `read_stdin` returns within its limit when the pipe never closes.
    - Cover `one_line`, the key normalisation, the choice of `cwd`, and `dialects` for each tool name.
    - `guard_main` and `quiet_main` give the right Results for a Pass, a Block, a `ConfigError` and an `EventError`.
    - In a subprocess run of a copy, a hook module whose `main` returns a Block with a non-ASCII Reason gives that Reason on stderr as UTF-8.
    - _Requirements: 1.3, 1.4, 1.8, 1.9, 1.10, 1.12, 1.13, 1.14_

  - [ ] 1.8 Write unit tests for the Runner in `tests/test_run.py`
    - The hook table holds the seven hooks with their modules and kinds. The guards and branch-check fail closed, and devlog and session-status fail open.
    - An unknown hook, a kind the hook doesn't take, a missing kind for a guard, a kind for a hook that takes none and extra arguments each exit 2 with one stderr line that starts with `run.py: ` and names them.
    - An injected `importer` that raises `SyntaxError`, `ImportError`, `NameError`, `RecursionError`, `MemoryError`, `SystemExit` or `KeyboardInterrupt`, at import or from the module's `main`, gives exit 2 with one Reason line for each fail-closed hook, saying what stays blocked for its kind. It gives exit 0 with one stdout line for each fail-open hook.
    - With a broken `config.toml` in a temporary root and stand-in modules that load it, the fail-closed hooks block and the fail-open hooks exit 0.
    - A Python older than 3.11 is reported as a failure of the hook. Stray prints never reach the output. A Result that breaks the hook's rule (exit 1, two Reason lines, output on a Pass, stderr from a fail-open hook, two stdout lines from devlog) is replaced by the hook's failure output. A session-status status of four lines passes through unchanged. Each fail-closed failure Reason names the hook, says what stays blocked and gives the way out.
    - `_one_line` gives the same result as `hookio.one_line`.
    - The state files are `data/hooks/task-start.json` and `data/hooks/devlog-warnings.txt` under the Repo_Root. Every devlog stdout line is saved exactly once, whether the module returned it or the Runner built it, and a folder in place of the warnings file doesn't change the exit code.
    - An `ast` check that every Python file under `scripts/` parses and imports only modules in `sys.stdlib_module_names` and our own `scripts` package, and that `run.py` parses with `feature_version=(3, 8)`. It covers later scripts as they're added.
    - For each hook in the table, a subprocess run of a copy whose module for that hook has a syntax error, and one whose module raises at import, gives the hook's exit code and one line.
    - _Requirements: 1.2, 1.10, 1.12, 1.13, 1.17, 1.18, 1.19, 1.20, 12.11, 15.10, 15.11_

  - [ ] 1.9 Write property test for reading stdin
    - **Property 13: Reading stdin never fails on bad bytes**
    - **Validates: Requirements 1.3**
    - Use `st.binary()` through an in-memory stream, and `st.text()` encoded as UTF-8 for the round trip.

  - [ ] 1.10 Write property test for unknown hooks and kinds
    - **Property 42: Unknown hooks and kinds exit 2 with one line**
    - **Validates: Requirements 1.20**

  - [ ] 1.11 Write property test for fail-closed hooks, starting with failures to load or run
    - **Property 16: Fail-closed hooks block on every failure**
    - **Validates: Requirements 1.12, 1.18**
    - Cover the import and `main` failures of all five fail-closed hooks and every kind they take, with an injected `importer`. Tasks 5.24, 8.13, 11.24, 14.19 and 18.5 add stdin that isn't complete JSON for each hook as its module is built.

  - [ ] 1.12 Write property test for fail-open hooks, starting with failures to load or run
    - **Property 40: The fail-open hooks always exit 0**
    - **Validates: Requirements 19.19, 1.13, 1.18, 15.11, 16.5**
    - Cover any stdin, or none, with an injected `importer` that fails at import or in `main`, for devlog and session-status, with a temporary Repo_Root for the saved warning. Tasks 22.4 and 25.7 add the other failures.

  - [ ] 1.13 Write property test for one-line Reasons, starting with the Runner
    - **Property 7: Every Reason is one line that starts with the hook's name**
    - **Validates: Requirements 19.7, 1.8, 1.9**
    - Use the Runner's failure Reasons from Property 16, with exception messages that hold control characters and line breaks, and the unknown names of Property 42, whose one line starts with `run.py: ` as the design's Runner step 1 writes it. Tasks 5.23, 8.10, 11.21, 14.16 and 18.4 add the guards and branch-check.

- [ ] 2. Add the licence rows, the hooks line and the `.env.*` ignore line
  - [ ] 2.1 Add the git, npm, AWS CLI v2 and uv rows to `docs/third-party-licences.md`
    - git: GPL-2.0, a separate command line tool that the hooks and people run.
    - npm: Artistic-2.0, a separate command line tool that installs splat-transform's packages.
    - AWS CLI v2: Apache-2.0, called by session-status and by people in their own terminal.
    - uv: MIT or Apache-2.0, runs the hooks and manages the Python environment.
    - Each row says the tool is never vendored, linked or shipped. Its "Installed" cell says whether the tool is on the laptop now. Find out with `Get-Command <tool>`, which runs none of them.
    - Samarth OK'd the git and npm rows in the chat on 19 November 2026, because GPL-2.0 and Artistic-2.0 are outside the Allowed_Licences. Say so in the commit message.
    - _Requirements: 18.1_

  - [ ] 2.2 Add the hooks line to `.kiro/steering/rules.md`
    - Add a "Hooks" section before "Specs" with one line: "If a hook blocks something it shouldn't, say so and fix the hook. Never work around a hook or turn one off without the user's OK."
    - This comes before the first hook file, so every chat that meets a live hook already has the rule.
    - _Requirements: 17.7, 18.2_

  - [ ] 2.3 Add `.env.*` to `.gitignore`
    - Put it on the line after `.env`, so git never stages a Secret_File in the repo, `.env.example` included. Task 8.5 tests it.
    - _Requirements: 20.9_

- [ ] 3. Build the shell splitter and the mention finder
  - [ ] 3.1 Implement `scripts/hooks/shellsplit.py`
    - `Dialect`, `Word`, `Redirect`, `Command`, `SplitError`, `split(text, dialect, cwd, home=None)` and `program_name(word)`, as in the design's "shellsplit" section.
    - Write a tokenizer for each of PowerShell, POSIX sh and cmd.exe. Split chains, and split nested `$(...)`, backticks, `(...)`, `@(...)` and `{...}` again at `depth + 1`.
    - Record input and output redirects apart from the words, and move environment prefixes into `env`. Split the words after each script wrapper and argv wrapper again, skipping the options of `uv run` and `npx` with their values. Read `uv run -m X` as `python -m X` and `uv run path.py` as `python path.py`.
    - Keep heredoc and here-string bodies as the Command's `stdin_text`, and split them again only when they feed a shell.
    - Track `cd`, `Set-Location`, `sl`, `chdir`, `pushd` and `Push-Location` with a literal target, or one that starts with a Home_Spelling, for the later Commands in the chain. Define the seven Home_Spellings once, as `HOME_SPELLINGS` in a new `scripts/hooks/secretpaths.py`, which task 8.1 completes.
    - Raise `SplitError` for an unclosed quote, here-string, heredoc, `$(` or `(`.
    - _Requirements: 1.6, 1.7, 20.5_

  - [ ] 3.2 Write unit tests for the splitter in `tests/test_shellsplit.py`
    - For each dialect, cover every separator, nesting form, wrapper, environment prefix, redirect, heredoc and here-string, PowerShell's `--%` and comments, and the `cd` tracking with literal and Home_Spelling targets.
    - `program_name` drops the folder, a `.exe`, `.cmd` or `.bat` ending and the letter case, so `C:\Program Files\Amazon\AWSCLIV2\aws.EXE` gives `aws`.
    - Each unclosed form raises `SplitError`.
    - _Requirements: 1.6, 1.7, 20.5_

  - [ ] 3.3 Implement `scripts/hooks/mentions.py`
    - `mention_words(cmd, top_level_only) -> set[int]`, for the rows of the design's mentions table. An argument with `$(`, a backtick or a variable reference is never a mention. The values of `-f`, `--file` and `/G:` name files to read, so they aren't mentions either.
    - With `top_level_only=True`, which guard-paid passes, it returns nothing for the forms Requirement 4.3 puts outside every Mention_Exception: a Command inside a script wrapper and a Command with an output redirect to a `.env` or `.env.*` file. guard-secrets passes `top_level_only=False`.
    - _Requirements: 4.2, 4.3, 20.2_

  - [ ] 3.4 Write unit tests for the mention finder in `tests/test_mentions.py`
    - Cover each row of the mentions table, the `-f`, `--file` and `/G:` values, the `$(`, backtick and variable cases, and both values of `top_level_only`.
    - _Requirements: 4.2, 4.3, 20.2_

- [ ] 4. Checkpoint - Ensure all tests pass
  - Run `uv run pytest -q` and fix every failure. Stop for the user only if a fix would change a requirement.

- [ ] 5. Build guard-paid's script and its tests
  - [ ] 5.1 Implement `scripts/hooks/awscli.py`
    - `AwsCall` and `parse(cmd)`, as in the design's "awscli" section.
    - Read global options anywhere, in both the `--name value` and `--name=value` spellings, knowing which global options take a value and which flags don't.
    - Find the service, the operation and the positionals, and collect `buckets` from `s3://` words and `--bucket` values. A word that only holds `s3://` gives an empty bucket name.
    - _Requirements: 5.1, 5.2, 5.3, 5.6, 5.7, 21.4_

  - [ ] 5.2 Write unit tests for `awscli` in `tests/test_awscli.py`
    - Check the service, operation, options in both spellings, positionals and buckets for representative commands, including options before the service, `aws configure get` and an `s3://` word with no bucket name.
    - _Requirements: 5.2, 5.7, 21.4_

  - [ ] 5.3 Implement `scripts/hooks/cloudcmd.py`
    - `cloud_module(cmd)` covers `python -m cloud.X` (any Python Program, with Python's own options before `-m` and `-m` inside a cluster such as `-um`), a path that resolves inside `<Repo_Root>/cloud/`, `python -c` code that imports `cloud.X` or does `from cloud import X, Y`, and the same forms after `uv run`.
    - `inline_aws(code)` finds `boto3`, `botocore` or `call_boto3` as whole identifiers. It reports whether the code names an operation from `[paid].inline_operations` in snake_case or CamelCase, or makes a client or resource for a `[paid].aws_services` service.
    - _Requirements: 3.3, 3.4, 3.5, 6.4, 7.2, 7.3_

  - [ ] 5.4 Implement `judge_aws` in `scripts/hooks/guard_paid.py`
    - Run the nine checks of the design's "judge_aws" table in order, with their Reasons filled in from the Config. The first is the Paid_Command check for aws (`[paid].aws`, and calls to `[paid].aws_services` that aren't Read_Calls), which gives the Paid_Reason.
    - The Account_Field scan of `--query` skips raw string literals, JSON literals and function names, and reads quoted identifiers without their quotes.
    - _Requirements: 3.2, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 6.1, 6.2, 6.3_

  - [ ] 5.5 Implement guard-paid's shell rules in `scripts/hooks/guard_paid.py`
    - `judge_shell_text` runs the design's steps 1 to 7 on each string value: the prefilter, the Spend_Name check with its Mention_Exception spans, the split for each dialect, the Paid_Command check for the `[paid].other` programs, the Cloud_Command check, the Inline_AWS_Code check (a heredoc body that feeds Python included), and `judge_aws` for every aws Command.
    - A `SplitError` blocks with the "couldn't read the command" Reason when the text holds one of guard-paid's trigger words.
    - `judge_shell` runs `judge_shell_text` on every string value of the Event, whatever tool name the Event gives.
    - _Requirements: 1.5, 1.6, 1.7, 1.11, 3.2, 3.3, 3.4, 3.5, 4.1, 4.2, 4.3, 6.4_

  - [ ] 5.6 Implement guard-paid's power and MCP rules and its `main`
    - `judge_tool_strings` follows the design. Inline_AWS_Code in any value blocks. Other multi-line text is skipped. A one-line value that mentions a Spend_Name blocks. A one-line value that reads as exactly one aws Command gets the shell verdict. Any other value is data.
    - `judge_tools` runs it on all the string values of the Event. `main(kind, stdin, root)` returns `guard_main("guard-paid", kind, {"shell": judge_shell, "tools": judge_tools}, stdin, root)`.
    - _Requirements: 1.21, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7_

  - [ ] 5.7 Write unit tests for guard-paid in `tests/test_guard_paid.py`
    - Check every Reason in the design's guard-paid section, and the examples for 5.6, 5.8 and 5.9.
    - Free_Cloud_Commands pass. The Requirement 4.3 forms never count as mentions, so `bash -c "git commit -m APPROVED_SPEND"` blocks. A heredoc body that feeds `python` is checked as code.
    - The 7.7 Events pass: cfn-lint, cfn-guard and a documentation search on a template whose policies name `ec2:RunInstances` and `kiroween-runner`.
    - The live-check calls get the expected verdicts. With the `shell` kind, `uv run python -m cloud.start` blocks and `echo guard-paid-check` passes. With the `tools` kind, the one-line value `aws ec2 run-instances --profile kiroween-read --region us-east-1` blocks and `ec2:RunInstances` passes.
    - In `tests/test_config.py`, a copied Config with another bucket prefix or another list of free modules changes guard-paid's verdicts.
    - Create `tests/test_hook_speed.py`. Run the copy's `run.py` with `sys.executable` on guard-paid's idle Events, a neutral shell command for `shell` and a power call with only data for `tools`: one untimed run, then three timed runs, whose median must be under `[hooks].idle_ms`.
    - _Requirements: 1.11, 2.7, 3.3, 3.5, 4.3, 5.6, 5.8, 5.9, 6.4, 7.5, 7.6, 7.7_

  - [ ] 5.8 Write the shared Hypothesis strategies in `tests/hook_strategies.py`
    - `dialect`, `quoted`, `neutral_part`, `chain`, `nest`, `place`, `shell_event`, `tool_event` and `embed`, as in the design's "Property tests" section. `nest` quotes the inner text correctly for each outer shell.
    - `neutral_part` uses the near misses the design lists and never makes a guarded Program, a guarded word, a Cloud_Command, a spend name or a Secret_File.
    - _Requirements: 19.5, 19.13_

  - [ ] 5.9 Write property test for the spend names
    - **Property 1: Spend names are blocked wherever they appear**
    - **Validates: Requirements 19.1, 4.1, 4.3, 1.6**
    - Add `spend_form()` to `tests/hook_strategies.py`, with the 13 forms of Requirement 19.1 and the name in random letter case.

  - [ ] 5.10 Write property test for good read, stop and S3 calls
    - **Property 2: Good read, stop and S3 calls pass**
    - **Validates: Requirements 19.2, 5.2, 6.1, 6.2, 6.3**
    - Add `good_aws_call()` to `tests/hook_strategies.py`, with shuffled options, both spellings and random quoting.

  - [ ] 5.11 Write property test for paid commands
    - **Property 6: Paid commands are blocked**
    - **Validates: Requirements 19.6, 3.2**
    - Add `paid_command()` to `tests/hook_strategies.py`.

  - [ ] 5.12 Write property test for aws without the read profile
    - **Property 3: aws without the read profile is blocked**
    - **Validates: Requirements 19.3, 5.1**
    - Draw from `good_aws_call()` and the aws calls of `paid_command()`.

  - [ ] 5.13 Write property test for cloud modules
    - **Property 17: Cloud modules are blocked unless marked free**
    - **Validates: Requirements 3.3, 6.4**

  - [ ] 5.14 Write property test for inline AWS code
    - **Property 18: Inline AWS code is always blocked with the right Reason**
    - **Validates: Requirements 3.4, 3.5, 7.2, 7.3**

  - [ ] 5.15 Write property test for mentions of the spend names
    - **Property 19: Mentions alone don't block**
    - **Validates: Requirements 4.2**

  - [ ] 5.16 Write property test for breaking one aws rule
    - **Property 20: Breaking one aws rule gives that rule's Reason**
    - **Validates: Requirements 5.3, 5.4, 5.5, 5.7**

  - [ ] 5.17 Write property test for aws commands in tool calls
    - **Property 21: One-line aws commands in tool calls get the shell verdict**
    - **Validates: Requirements 7.1**

  - [ ] 5.18 Write property test for data in tool calls
    - **Property 22: Data in tool calls passes, one-line spend mentions don't**
    - **Validates: Requirements 7.4, 7.5, 7.6**

  - [ ] 5.19 Write property test for neutral commands, starting with guard-paid
    - **Property 5: Neutral commands pass every guard**
    - **Validates: Requirements 19.5, 20.8**
    - Parametrize the test over a list of guards that holds only guard-paid for now. Tasks 8.9, 11.20 and 14.15 add guard-secrets, guard-git and guard-licence.

  - [ ] 5.20 Write property test for commands placed anywhere in the Event, starting with guard-paid
    - **Property 14: A command is found wherever it sits in the Event**
    - **Validates: Requirements 1.5**
    - Put the commands of Properties 1, 2 and 6 into `embed`. Tasks 8.11, 11.22 and 14.17 add the other guards.

  - [ ] 5.21 Write property test for unreadable commands, starting with guard-paid
    - **Property 15: Unreadable commands that name a guarded word are blocked**
    - **Validates: Requirements 1.7**
    - Tasks 8.12, 11.23 and 14.18 add the other guards.

  - [ ] 5.22 Write property test for the entry's kind, starting with guard-paid
    - **Property 43: The entry's kind decides the rules**
    - **Validates: Requirements 1.21**
    - Use the commands of Properties 1, 2 and 6 with the `shell` and `tools` kinds and the tool names the design lists. Tasks 8.14, 11.25 and 14.20 add the other guards.

  - [ ] 5.23 Extend the one-line Reason property test to guard-paid
    - **Property 7: Every Reason is one line that starts with the hook's name**
    - **Validates: Requirements 19.7, 1.8, 1.9**
    - Add the blocked inputs of Properties 1, 3 and 6, with control characters and line breaks in their commands.

  - [ ] 5.24 Extend the fail-closed property test to guard-paid
    - **Property 16: Fail-closed hooks block on every failure**
    - **Validates: Requirements 1.12, 1.18**
    - Feed stdin that `json.loads` rejects to the Runner's `main` in-process for guard-paid with both its kinds.

- [ ] 6. Checkpoint - Ensure all tests pass before guard-paid goes live
  - Run `uv run pytest -q` and fix every failure. Stop for the user only if a fix would change a requirement.

- [ ] 7. Add the guard-paid hook file
  - [ ] 7.1 Create `.kiro/hooks/guard-paid.json`
    - Use `createHook` for the first entry: trigger `PreToolUse`, matcher `shell`, command `uv run --no-project -q python scripts/hooks/run.py guard-paid shell`, timeout 15.
    - Add the `@mcp` and `@powers` entries by editing the JSON, each with the command `uv run --no-project -q python scripts/hooks/run.py guard-paid tools` and timeout 15, so the file matches the design's example.
    - _Requirements: 1.1, 3.1, 17.1, 17.2, 17.4_

  - [ ] 7.2 Write the hook-file tests in `tests/test_hook_files.py`
    - Checks that loop over every file in `.kiro/hooks/`, so they cover later files as they're added. Each file is in Kiro's v1 format. Every command action is exactly the Launcher for a hook in `run.HOOKS`, with the kind that fits the matcher: `shell` for `shell`, `tools` for `@mcp` and `@powers`, `read` for `read`, and no kind for an entry with no matcher. Each timeout is longer than the sum of the Config's time limits that its hook uses.
    - Every entry that runs a hook `run.HOOKS` marks as failing open sits on `SessionStart` or `PostTaskExec`, where exit 2 blocks nothing, and on no other trigger.
    - guard-paid's file has the three entries of the design's table.
    - _Requirements: 1.1, 3.1, 17.1, 17.2, 17.3, 17.9_

- [ ] 8. Build guard-secrets' script and its tests
  - [ ] 8.1 Complete `scripts/hooks/secretpaths.py`
    - `NormPath`, `pieces`, `normalise`, `is_secret_file` and `names_secret_file`, as in the design's "secretpaths" section, next to the `HOME_SPELLINGS` from task 3.1.
    - Work on text alone, with no file system calls: split pieces at `=`, `:`, `,`, `{`, `}` and line breaks, casefold, turn `\` into `/`, replace a leading Home_Spelling, tell rooted paths from relative ones, resolve `.` and `..`, and drop trailing dots and spaces from parts.
    - A path names a Secret_File when any part is `[secrets].aws_folder`, the last part included, when the last part is `[secrets].env_file` or starts with it plus `.`, or when the last part holds a wildcard and starts with `.env`.
    - _Requirements: 20.3, 20.4, 20.5, 20.6, 20.10_

  - [ ] 8.2 Write unit tests for `secretpaths` in `tests/test_secretpaths.py`
    - Cover `pieces` for `=`, `:`, `,`, braces and line breaks, each Home_Spelling, drives, UNC paths and rooted paths, `..` that climbs out of `.aws` and `..` that stays inside it, and trailing dots and spaces.
    - Every Secret_File and every near miss listed in the design's guard-secrets section gets its expected answer, including a bare `.aws` folder such as `~/.aws` and `%USERPROFILE%\.aws`.
    - _Requirements: 20.4, 20.6, 20.10_

  - [ ] 8.3 Implement guard-secrets' read rules and its `main` in `scripts/hooks/guard_secrets.py`
    - Import only `hookio`, `config` and `secretpaths` at the top of the file.
    - `judge_read` runs `names_secret_file` on every string value of the Event, at any depth, with the Event's `cwd`. The first hit blocks with the design's read Reason, which names the piece.
    - `main(kind, stdin, root)` returns `guard_main("guard-secrets", kind, {"read": judge_read}, stdin, root)`. Task 8.4 adds the `shell` kind.
    - _Requirements: 1.21, 20.3, 20.7, 20.8, 20.10_

  - [ ] 8.4 Implement guard-secrets' shell rules in `scripts/hooks/guard_secrets.py`
    - `judge_shell` imports `shellsplit`, `awscli` and `mentions` inside the function. It runs the prefilter (`aws`, `.env`, or an Event `cwd` inside a `.aws` folder), then the split for each dialect with `ctx.home`. A `SplitError` blocks when the value holds `aws` or `.env`.
    - Block the Secret_AWS_Commands with the three Reasons of the design's table, wherever aws's own options sit and whatever they are.
    - For each Command at any depth, check every word except the ones `mention_words(cmd, top_level_only=False)` returns, every environment prefix value and every input redirect target with `names_secret_file` and the Command's `cwd`. The first hit blocks with the read Reason.
    - Add `"shell": judge_shell` to `main`.
    - _Requirements: 1.5, 1.6, 1.7, 1.11, 20.2, 20.5, 20.8, 20.10, 21.1, 21.2, 21.3, 21.4_

  - [ ] 8.5 Write unit tests for guard-secrets in `tests/test_guard_secrets.py`
    - Every row of the design's "What guard-secrets lets through and what it blocks" table gets its verdict.
    - The three Secret_AWS_Command Reasons. The read and shell Reasons name the path, say the file can hold keys and say to ask the user.
    - `cd`, `Set-Location`, `pushd` and `Push-Location` move where later paths resolve, Home_Spelling targets included. A `cwd` inside `.aws` blocks, and so does `cd ~/.aws`.
    - The read rules run with `subprocess.Popen` and the sockets patched to fail the test.
    - `.gitignore` holds `.env.*`.
    - The live-check calls get the expected verdicts. With the `shell` kind, `Get-Content data/hooks/live-check/.env` blocks and `Get-Content .gitignore -TotalCount 1` passes. With the `read` kind, a `read_file` Event on `data/hooks/live-check/.env` and a `grep_search` Event with that include pattern block, and a `read_file` Event on `.gitignore` passes.
    - In `tests/test_config.py`, a copied Config with another `.aws` folder name or other secret words changes guard-secrets' verdicts.
    - Add guard-secrets' idle shell Event, a neutral shell command, to `tests/test_hook_speed.py`, and a `read_file` Event on an ordinary file, whose median must be under `[hooks].read_ms`.
    - _Requirements: 1.11, 2.7, 20.2, 20.3, 20.5, 20.7, 20.9, 20.10, 21.1, 21.2, 21.3_

  - [ ] 8.6 Write property test for Secret_File reads
    - **Property 35: Secret_File reads are blocked in every spelling**
    - **Validates: Requirements 19.14, 20.2, 20.3, 20.4, 20.5, 20.6, 20.10**
    - Add `secret_path()` and `read_event` to `tests/hook_strategies.py`. `secret_path()` also makes a bare `.aws` folder and folders inside one, such as `~/.aws` and `~/.aws/cli`, as the design's generator table says.

  - [ ] 8.7 Write property test for Secret_AWS_Commands
    - **Property 36: Secret_AWS_Commands are blocked whatever their options**
    - **Validates: Requirements 19.15, 21.1, 21.2, 21.3, 21.4**
    - Add `secret_aws_command()` to `tests/hook_strategies.py`.

  - [ ] 8.8 Write property test for read calls that name no Secret_File
    - **Property 37: Read calls that name no Secret_File pass**
    - **Validates: Requirements 19.16, 20.8**
    - Add `ordinary_path()` to `tests/hook_strategies.py`, and patch `subprocess.Popen` to fail the test.

  - [ ] 8.9 Extend the neutral-command property test to guard-secrets
    - **Property 5: Neutral commands pass every guard**
    - **Validates: Requirements 19.5, 20.8**

  - [ ] 8.10 Extend the one-line Reason property test to guard-secrets
    - **Property 7: Every Reason is one line that starts with the hook's name**
    - **Validates: Requirements 19.7, 1.8, 1.9**
    - Add the blocked inputs of Properties 35 and 36, with control characters and line breaks in their paths and commands.

  - [ ] 8.11 Extend the Event-placement property test to guard-secrets
    - **Property 14: A command is found wherever it sits in the Event**
    - **Validates: Requirements 1.5**
    - Add the commands of Properties 35 and 36.

  - [ ] 8.12 Extend the unreadable-command property test to guard-secrets
    - **Property 15: Unreadable commands that name a guarded word are blocked**
    - **Validates: Requirements 1.7**

  - [ ] 8.13 Extend the fail-closed property test to guard-secrets
    - **Property 16: Fail-closed hooks block on every failure**
    - **Validates: Requirements 1.12, 1.18**
    - Cover both the `shell` and the `read` kind.

  - [ ] 8.14 Extend the entry's-kind property test to guard-secrets
    - **Property 43: The entry's kind decides the rules**
    - **Validates: Requirements 1.21**
    - Add the commands of Properties 35 and 36 with the `shell` kind, and the read values of Properties 35 and 37 with the `read` kind.

- [ ] 9. Checkpoint - Ensure all tests pass before guard-secrets goes live
  - Run `uv run pytest -q` and fix every failure. Stop for the user only if a fix would change a requirement.

- [ ] 10. Add the guard-secrets hook file
  - [ ] 10.1 Create `.kiro/hooks/guard-secrets.json`
    - Use `createHook` for the first entry: trigger `PreToolUse`, matcher `shell`, command `uv run --no-project -q python scripts/hooks/run.py guard-secrets shell`, timeout 10.
    - Add the second entry by editing the JSON: trigger `PreToolUse`, matcher `read`, command `uv run --no-project -q python scripts/hooks/run.py guard-secrets read`, timeout 10. There is no separate `grep_search` entry, because the `read` matcher selects it.
    - _Requirements: 1.1, 17.1, 17.2, 17.4, 20.1_

  - [ ] 10.2 Extend `tests/test_hook_files.py` to guard-secrets
    - Check its two entries' triggers, matchers, Launchers and timeouts.
    - _Requirements: 1.1, 17.1, 17.2, 17.3, 20.1_

- [ ] 11. Build guard-git's script and its tests
  - [ ] 11.1 Implement `scripts/hooks/gitio.py`
    - `GitResult`, `GitError`, the `GitRunner` protocol, `real_git(deadline)`, `current_branch(git, cwd)` and `head_commit(git, cwd)`, as in the design's "gitio" section.
    - `real_git` imports `subprocess` inside itself and runs git with an argument list, `GIT_TERMINAL_PROMPT=0`, `GIT_OPTIONAL_LOCKS=0` and whatever is left of the `[hooks].git_seconds` budget. It raises `GitError` when git is missing, times out or exits with a code the caller didn't expect.
    - `current_branch` returns `None` for a detached HEAD, and `head_commit` returns `None` before the first commit.
    - _Requirements: 1.12, 8.9, 13.1, 13.2, 13.3_

  - [ ] 11.2 Add the git test fixtures to `tests/conftest.py`
    - A temporary repo made with `git init -b feat/test` in `tmp_path`, skipped with `pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")`. Its git calls use `GIT_CONFIG_GLOBAL` pointing at an empty file and `GIT_CONFIG_NOSYSTEM=1`, and commit with `-c user.name=test -c user.email=test@example.invalid`, so nobody's own git settings matter and none change.
    - A fake `GitRunner` that answers from a dict and fails the test when it's asked anything it doesn't expect.
    - _Requirements: 1.15, 1.16_

  - [ ] 11.3 Implement `scripts/hooks/secret_scan.py`
    - `Finding`, `added_lines(new, old)` and `scan(lines, patterns)`, as in the design's "secret_scan" section.
    - A line counts as added when the old version holds no identical line. `scan` uses `re.search`, drops a secret-key match whose `value` group is all hex digits, and never returns the Secret itself.
    - _Requirements: 9.5, 9.6_

  - [ ] 11.4 Write unit tests for `secret_scan` in `tests/test_secret_scan.py`
    - Every Secret kind in the Config is found, wherever it sits in the line. `added_lines` works with and without an old version.
    - Build every Secret from pieces at run time.
    - _Requirements: 9.5, 9.6_

  - [ ] 11.5 Write property test for SHAs and template account IDs
    - **Property 26: SHAs and template account IDs aren't Secrets**
    - **Validates: Requirements 9.6**

  - [ ] 11.6 Implement guard-git's command rules and its `main` in `scripts/hooks/guard_git.py`
    - Read each git Command into a `GitCall`: skip git's own options before the subcommand, resolve each `-C` path, and read `stage` as `add`. Split combined short options using the letters that take a value for each subcommand. Treat any prefix of a blocked long option as that option, and everything after `--` as paths.
    - `folder_pathspecs` returns the five kinds of Folder_Pathspec of the design, with one `git ls-files -z` call for the paths that don't exist.
    - Block the `add`, `push`, `reset`, `clean` and `commit` rows of the design's "Blocked forms" table with their Reasons, `--no-ignore-removal` in `add` included. Block `--pathspec-from-file` in `add` and `commit` with the Reason of the table's `--pathspec-from-file` row, before any git call.
    - `judge_chain` walks the Commands in order. It keeps the branch that an earlier `git switch` moved to (task 11.9 adds `git checkout`), resolving `-` and `@{-N}` with `git rev-parse --abbrev-ref`, and the paths that earlier `git add` Commands name. Without such a switch, the branch comes from `current_branch` in the Command's folder. A git failure blocks.
    - `judge_shell` runs the prefilter (`git`), the split for each dialect and `judge_chain` on every string value. A `SplitError` blocks when the text holds `git`. `main(kind, stdin, root)` returns `guard_main("guard-git", kind, {"shell": judge_shell}, stdin, root)`.
    - _Requirements: 1.7, 1.11, 1.14, 1.21, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8, 8.9, 8.10, 8.16_

  - [ ] 11.7 Write unit tests for guard-git's command rules in `tests/test_guard_git.py`
    - `git reset --hard` blocks. `git clean -n` blocks and `git add -n docs/devlog.md` passes, as the live check expects.
    - A commit on `main` blocks, also after a `git switch main` earlier in the chain, and a commit on a detached HEAD passes. A push with no refspec while on `main` blocks, and so do `--all`, `:main` and `--delete main`.
    - `git add` with `-A`, `-Av`, a prefix such as `--al`, `--no-ignore-removal`, `.`, a folder, a wildcard or `:/` blocks. In a temporary repo, a tracked folder that was deleted counts as a Folder_Pathspec.
    - `git add` and `git commit` with `--pathspec-from-file`, in both spellings, block with the 8.16 Reason, and the fake `GitRunner` gets no call.
    - `git commit -am msg` blocks, and `git commit -ma` passes with the message "a".
    - Relative paths resolve against the Event's `cwd`, an earlier `cd` in the chain, and `-C`. git failing or timing out blocks.
    - Add guard-git's idle Event, a neutral shell command, to `tests/test_hook_speed.py`.
    - _Requirements: 1.11, 1.14, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8, 8.9, 8.10, 8.16_

  - [ ] 11.8 Write property test for git add with file paths
    - **Property 4: git add with file paths passes**
    - **Validates: Requirements 19.4**
    - Add `file_paths()` to `tests/hook_strategies.py`.

  - [ ] 11.9 Implement guard-git's restore, checkout and switch rules in `scripts/hooks/guard_git.py`
    - `checkout_paths` finds the Checkout_Paths by git's own rule, as in the design's "Telling a branch from a folder" section. A first argument that is a Folder_Pathspec of kinds 1 to 3 is a path with no git call. For a folder on disk or in the index, ask `git rev-parse --verify --quiet <arg>^{commit}` within the git budget: exit 0 means a commit, branch or tag, exit 1 means a path, and anything else blocks with the design's Reason.
    - Block the `restore`, `checkout` and `switch` rows of the "Blocked forms" table with their Reasons: a `restore` that changes the working tree with a Folder_Pathspec, a `checkout` with a Checkout_Path that is a Folder_Pathspec, `checkout -f` and `--force`, and `switch -f`, `--force` and `--discard-changes`, counting clusters and long-option prefixes. Block `--pathspec-from-file` in `checkout` and in a `restore` that changes the working tree with the Reason of the table's `--pathspec-from-file` row.
    - A `restore` with `--staged` and without `--worktree` skips the folder rule and the `--pathspec-from-file` rule. A `switch`, or a `checkout` with no Checkout_Path, without the force options, is judged by the other rules only.
    - `judge_chain` also follows an earlier `git checkout`, and a checkout whose first argument is a Checkout_Path leaves the branch as it was.
    - _Requirements: 8.9, 8.11, 8.12, 8.13, 8.14, 8.15, 8.16_

  - [ ] 11.10 Write unit tests for the restore, checkout and switch rules in `tests/test_guard_git.py`
    - In a temporary repo, `git checkout docs` passes when `docs` is a branch and blocks when it's only a folder. git failing during `rev-parse` blocks.
    - `git checkout .` and `git checkout :/` block with no `rev-parse` call. `git checkout -- .` and `git checkout HEAD -- docs` block.
    - `git restore .`, `git restore -W docs` and `git restore -SW docs` block. `git restore --staged .` and `git restore -S docs` pass.
    - `git checkout -f`, `git checkout --force`, `git switch -f` and a prefix of `--discard-changes` block. `git switch -cf` passes and creates a branch named `f`.
    - `--pathspec-from-file` blocks with the 8.16 Reason in `checkout` and in `restore` without `--staged` or with `--worktree`, and `restore --staged` with `--pathspec-from-file` passes.
    - `git checkout feat/hooks` and `git checkout -b feat/x` pass with no `rev-parse` call when no folder has those names.
    - A commit after `git checkout main` blocks, and a commit after a checkout of a file keeps the branch.
    - _Requirements: 8.9, 8.11, 8.12, 8.13, 8.14, 8.15, 8.16_

  - [ ] 11.11 Write property test for the blocked git forms
    - **Property 24: Every blocked git form is blocked wherever it sits**
    - **Validates: Requirements 8.3, 8.4, 8.5, 8.6, 8.8, 8.10, 8.11, 8.12, 8.13, 8.16**

  - [ ] 11.12 Write property test for restoring and checking out files
    - **Property 38: Restoring and checking out files passes**
    - **Validates: Requirements 19.17**

  - [ ] 11.13 Write property test for branch checkouts and switches
    - **Property 39: Branch checkouts and switches pass**
    - **Validates: Requirements 19.18, 8.15**
    - Add `branch_name()` to `tests/hook_strategies.py`.

  - [ ] 11.14 Write property test for unstaging
    - **Property 41: Unstaging never blocks**
    - **Validates: Requirements 8.14**

  - [ ] 11.15 Write property test for git's own options
    - **Property 23: git's own options don't change the verdict**
    - **Validates: Requirements 8.2**
    - Use the git Commands of Properties 4, 24 and 38.

  - [ ] 11.16 Implement guard-git's commit checks in `scripts/hooks/guard_git.py`
    - `staged_files` builds the Staged_File list from the three sources of the design's "What a commit would record" section, following git's rule: a commit with no paths takes the first two, a commit with paths takes only its paths, and `-i` or `--include` takes all three. Working-tree content wins, and deletions are left out. All git calls share the `[hooks].git_seconds` budget.
    - `check_staged` runs the size, extension, image-folder and Secret checks in path order, with the design's Reasons. The size check runs before any content is read.
    - `judge_chain` runs these checks for every `git commit` that passes the other rules.
    - _Requirements: 1.12, 9.1, 9.2, 9.3, 9.4, 9.5, 9.6_

  - [ ] 11.17 Write unit tests for the commit checks in `tests/test_guard_git.py`
    - In a temporary repo, a commit with no paths checks the files already staged and the files that an earlier `git add` names. A commit with paths checks only a file and a folder in its pathspec, so a big file staged by someone else doesn't block it. With `-i` it checks all three. A deleted file isn't checked. A repo with no commit yet works.
    - A file at the size limit passes and one byte over blocks, with a small limit in a copied Config. Add that changed limit to `tests/test_config.py` as a Config-driven verdict.
    - The Secret Reason names the file, the line and the kind, never the Secret. The ARN Reason says to write `${AWS::AccountId}` or `<account-id>`. git failing during the checks blocks.
    - _Requirements: 1.12, 2.7, 9.1, 9.2, 9.5, 9.6_

  - [ ] 11.18 Write property test for Secrets in added lines
    - **Property 8: Secrets in added lines are blocked and never repeated**
    - **Validates: Requirements 19.8, 9.5**

  - [ ] 11.19 Write property test for the file type rules
    - **Property 25: File type rules follow the extension and the folder**
    - **Validates: Requirements 9.3, 9.4**

  - [ ] 11.20 Extend the neutral-command property test to guard-git
    - **Property 5: Neutral commands pass every guard**
    - **Validates: Requirements 19.5, 20.8**
    - Give guard-git a fake `GitRunner` that fails the test if it's called.

  - [ ] 11.21 Extend the one-line Reason property test to guard-git
    - **Property 7: Every Reason is one line that starts with the hook's name**
    - **Validates: Requirements 19.7, 1.8, 1.9**
    - Add the blocked inputs of Properties 8 and 24, with control characters and line breaks in file names and commands.

  - [ ] 11.22 Extend the Event-placement property test to guard-git
    - **Property 14: A command is found wherever it sits in the Event**
    - **Validates: Requirements 1.5**
    - Add the commands of Properties 4 and 24.

  - [ ] 11.23 Extend the unreadable-command property test to guard-git
    - **Property 15: Unreadable commands that name a guarded word are blocked**
    - **Validates: Requirements 1.7**

  - [ ] 11.24 Extend the fail-closed property test to guard-git
    - **Property 16: Fail-closed hooks block on every failure**
    - **Validates: Requirements 1.12, 1.18**

  - [ ] 11.25 Extend the entry's-kind property test to guard-git
    - **Property 43: The entry's kind decides the rules**
    - **Validates: Requirements 1.21**
    - Add the commands of Properties 4 and 24.

- [ ] 12. Checkpoint - Ensure all tests pass before guard-git goes live
  - Run `uv run pytest -q` and fix every failure. Stop for the user only if a fix would change a requirement.

- [ ] 13. Add the guard-git hook file
  - [ ] 13.1 Create `.kiro/hooks/guard-git.json`
    - Use `createHook`: trigger `PreToolUse`, matcher `shell`, command `uv run --no-project -q python scripts/hooks/run.py guard-git shell`, timeout 20.
    - _Requirements: 1.1, 8.1, 17.1, 17.2, 17.4_

  - [ ] 13.2 Extend `tests/test_hook_files.py` to guard-git
    - Check its trigger, matcher, Launcher and timeout.
    - _Requirements: 1.1, 8.1, 17.1, 17.2, 17.3_

- [ ] 14. Build the licence logic and guard-licence's script and its tests
  - [ ] 14.1 Implement `scripts/hooks/licences.py`
    - `pep503`, `npm_name`, `Statement`, `Judgement`, `SpdxNode`, `SpdxError`, `parse_spdx`, `spdx_allowed`, `judge_statements`, `AllowEntry`, `AllowlistError`, `load_allowlist`, `find_entry`, `is_blocked_package` and `PackageSpec`, as in the design's "licences" section. `PackageSpec` lives here because `registry.py` takes one and is built before the guard.
    - `parse_spdx` is a recursive-descent parser in which `WITH` binds tightest and `OR` loosest. It matches operators in any letter case and judges `id+` as `id`.
    - `judge_statements` follows the design's rules for expressions, licence texts and classifiers. A package is allowed only when at least one Statement counts and every Statement that counts is allowed.
    - _Requirements: 10.3, 10.4, 10.6, 10.11, 11.1, 11.2, 11.3, 12.3_

  - [ ] 14.2 Write `scripts/hooks/licence_allowlist.txt`
    - Add the header comments and the 14 starting entries of Requirement 11.4, as the design's "The Allowlist file" section writes them.
    - _Requirements: 11.1, 11.4_

  - [ ] 14.3 Write unit tests for the licence logic in `tests/test_licences.py`
    - SPDX texts that don't parse raise `SpdxError`.
    - Check the Statement mapping for every source in the design's table, including `UNKNOWN`, a pasted licence text and `License :: OSI Approved` on its own.
    - A broken Allowlist line raises `AllowlistError` with its line number. The real Allowlist holds exactly the starting entries.
    - The "Named exceptions" section of `rules.md` names every Allowlist entry whose licence isn't in the Allowed_Licences.
    - _Requirements: 10.3, 10.4, 10.6, 11.1, 11.4, 11.5_

  - [ ] 14.4 Write property test for SPDX expressions
    - **Property 10: SPDX expressions follow the OR, AND and WITH rule**
    - **Validates: Requirements 19.10, 10.4**

  - [ ] 14.5 Write property test for Allowlist matching
    - **Property 29: Allowlist matching follows normalised names, prefixes and versions**
    - **Validates: Requirements 10.11, 11.2, 11.3**

  - [ ] 14.6 Write property test for Allowlist lines
    - **Property 30: Allowlist lines round-trip**
    - **Validates: Requirements 11.1**

  - [ ] 14.7 Implement `scripts/hooks/registry.py`
    - `Lookup`, `LookupFailed` and `http_lookup(cfg)`, as in the design's "registry" section.
    - Build the URL from the Config with the name URL-quoted, so an npm scope's `/` becomes `%2F`, and `{version}` set to the pin or `latest`. Import `urllib.request` inside the function that fetches.
    - HTTP errors, a 404, bad JSON and timeouts raise `LookupFailed` with a short reason.
    - Keep the URL building and the parsing of PyPI and npm JSON into Statements in plain functions, so tests can check them without the network.
    - _Requirements: 10.2, 10.6_

  - [ ] 14.8 Add a fake `Lookup` to `tests/conftest.py`
    - It records its calls, returns the Statements a test gives it, and fails the test when it's asked about a package it doesn't expect.
    - _Requirements: 1.15_

  - [ ] 14.9 Implement Package_Add detection in `scripts/hooks/guard_licence.py`
    - `PackageAdd` and `package_adds(cmd)` for the forms in the design's "Finding Package_Adds" table, at any depth and after `uv run`, building `licences.PackageSpec` values.
    - Parse Python specs as PEP 508 requirements and npm specs as the design describes. Record the exact pin, the group (`--dev` or `--group dev`) and the source (registry, URL, git, path or requirements file).
    - _Requirements: 10.2, 10.9, 10.12_

  - [ ] 14.10 Implement guard-licence's judge and its `main` in `scripts/hooks/guard_licence.py`
    - Judge each spec with the six rules of the design's "Judging each spec" section, in order and with their Reasons.
    - Give all lookups of one Shell_Call one deadline of `[licences].lookup_seconds`. Each lookup gets the time that's left, and specs left when none is left fail with "timed out".
    - `judge_shell` runs the prefilter and the split for each dialect on every string value. A `SplitError` blocks when the text holds a trigger word. `main(kind, stdin, root)` returns `guard_main("guard-licence", kind, {"shell": judge_shell}, stdin, root)`.
    - _Requirements: 1.7, 1.11, 1.21, 10.2, 10.3, 10.5, 10.6, 10.7, 10.8, 10.9, 10.10, 10.12_

  - [ ] 14.11 Write unit tests for guard-licence in `tests/test_guard_licence.py`
    - Cover each Package_Add form and each kind of spec.
    - A failed lookup, a timeout of the shared deadline and a lookup with no licence each block with the 10.6 Reason. An Allowlist package passes with no lookup. Add examples for 10.9 and 10.12.
    - Check the registry URLs and the PyPI and npm JSON parsing with canned JSON.
    - The live-check calls get the expected verdicts: `uv pip install --dry-run opencv-python` blocks without a lookup, and `uv pip install --dry-run iniconfig` passes when the fake lookup returns MIT.
    - Add guard-licence's idle Event, a neutral shell command, to `tests/test_hook_speed.py`.
    - _Requirements: 1.11, 10.2, 10.6, 10.7, 10.9, 10.12_

  - [ ] 14.12 Write property test for the OpenCV package spellings
    - **Property 9: Every spelling of an OpenCV package other than headless is blocked**
    - **Validates: Requirements 19.9, 10.10**

  - [ ] 14.13 Write property test for the lookups
    - **Property 27: Lookups follow the named packages and their licences**
    - **Validates: Requirements 10.2, 10.3, 10.5**

  - [ ] 14.14 Write property test for the Place rules
    - **Property 28: Place rules hold for every Package_Add form**
    - **Validates: Requirements 10.8, 10.9**

  - [ ] 14.15 Extend the neutral-command property test to guard-licence
    - **Property 5: Neutral commands pass every guard**
    - **Validates: Requirements 19.5, 20.8**
    - Give guard-licence a fake `Lookup` that fails the test if it's called.

  - [ ] 14.16 Extend the one-line Reason property test to guard-licence
    - **Property 7: Every Reason is one line that starts with the hook's name**
    - **Validates: Requirements 19.7, 1.8, 1.9**
    - Add Property 9's blocked inputs, with control characters and line breaks in the package names.

  - [ ] 14.17 Extend the Event-placement property test to guard-licence
    - **Property 14: A command is found wherever it sits in the Event**
    - **Validates: Requirements 1.5**
    - Add Property 9's commands.

  - [ ] 14.18 Extend the unreadable-command property test to guard-licence
    - **Property 15: Unreadable commands that name a guarded word are blocked**
    - **Validates: Requirements 1.7**

  - [ ] 14.19 Extend the fail-closed property test to guard-licence
    - **Property 16: Fail-closed hooks block on every failure**
    - **Validates: Requirements 1.12, 1.18**

  - [ ] 14.20 Extend the entry's-kind property test to guard-licence
    - **Property 43: The entry's kind decides the rules**
    - **Validates: Requirements 1.21**
    - Add Property 9's commands.

- [ ] 15. Checkpoint - Ensure all tests pass before guard-licence goes live
  - Run `uv run pytest -q` and fix every failure. Stop for the user only if a fix would change a requirement.

- [ ] 16. Add the guard-licence hook file
  - [ ] 16.1 Create `.kiro/hooks/guard-licence.json`
    - Use `createHook`: trigger `PreToolUse`, matcher `shell`, command `uv run --no-project -q python scripts/hooks/run.py guard-licence shell`, timeout 30.
    - _Requirements: 1.1, 10.1, 17.1, 17.2, 17.4_

  - [ ] 16.2 Extend `tests/test_hook_files.py` to guard-licence
    - Check its trigger, matcher, Launcher and timeout.
    - _Requirements: 1.1, 10.1, 17.1, 17.2, 17.3_

- [ ] 17. Build the licence check script and its tests
  - [ ] 17.1 Implement `scripts/check_licences.py`
    - `Installed`, `Direct`, `python_packages`, `npm_packages`, `direct_dependencies`, `table_names`, `verdict` and `main(argv)` with the `--cloud` option, as in the design's "check_licences.py" section. Start the file with the lines that put the Repo_Root (`parents[1]`) on `sys.path`.
    - Read the Python packages with `importlib.metadata`, deduplicated by `pep503`. Read every `package.json` in a package folder under `node_modules/` at the repo root, at any depth, skipping `.bin`. When `node_modules/` is missing, print the one line that says so.
    - Judge each package in the guard's order, print one sorted line per package, then the Direct_Dependency and dev-only messages, and exit 0 or 1 as the design says.
    - _Requirements: 12.1, 12.2, 12.3, 12.4, 12.5, 12.6, 12.7, 12.8, 12.9, 12.10, 12.11_

  - [ ] 17.2 Write unit tests for the Licence_Checker in `tests/test_check_licences.py`
    - Use fake `.dist-info` folders in `tmp_path` and a temporary `node_modules/` with nested and scoped packages. The sockets stay blocked throughout.
    - Check the exit codes and printed lines for Requirements 12.4 to 12.10, including `--cloud`, a missing table row and a dev-only package in the project dependencies.
    - Matching against the table uses whole tokens, so `pytest` doesn't match `pytest-cov`.
    - A run against the real environment, with no `node_modules/`, exits 0. The `ast` check from task 1.8 now covers this script too.
    - _Requirements: 12.1, 12.2, 12.4, 12.5, 12.6, 12.7, 12.8, 12.9, 12.10, 12.11, 12.12_

  - [ ] 17.3 Write property test for the checker and the guard agreeing
    - **Property 31: The Licence_Checker and Guard_Licence agree**
    - **Validates: Requirements 12.3**

- [ ] 18. Build branch-check's script and its tests
  - [ ] 18.1 Implement `scripts/hooks/branch_check.py`
    - Its judge calls `current_branch` in the Event's folder. On `[git].main_branch` it blocks with the design's Reason. On any other branch or a detached HEAD it writes the Start_Record and passes. A `GitError` blocks with git's first error line and "Run the task from inside the repo".
    - The Start_Record goes to the location in `run.py` (`data/hooks/task-start.json`) and holds `head` from `head_commit`, the time with its UTC offset, the spec and the task. Write it to a temporary file in the same folder and move it into place with `os.replace`. Any error while reading HEAD, making the folder or writing the file is swallowed, and the task still starts with a Pass and no output.
    - Find the spec and the task as the design's devlog section says, with a helper in `scripts/hooks/hookio.py` that devlog reuses.
    - `main(kind, stdin, root)` returns `guard_main("branch-check", None, {None: judge}, stdin, root)`.
    - _Requirements: 1.10, 1.12, 13.1, 13.2, 13.3, 13.4_

  - [ ] 18.2 Write unit tests for branch-check in `tests/test_branch_check.py`
    - Cover `main`, a `feat/` branch, a detached HEAD and git failing.
    - The Start_Record holds HEAD, the time, the spec and the task, and replaces an earlier one. With a file where the `data/hooks` folder should be, the task still passes with no output.
    - `git check-ignore` ignores the Start_Record's path in the real repo.
    - _Requirements: 1.10, 1.12, 13.1, 13.2, 13.3, 13.4_

  - [ ] 18.3 Write property test for branch-check
    - **Property 32: branch-check blocks exactly on main**
    - **Validates: Requirements 13.1, 13.2**

  - [ ] 18.4 Extend the one-line Reason property test to branch-check
    - **Property 7: Every Reason is one line that starts with the hook's name**
    - **Validates: Requirements 19.7, 1.8, 1.9**
    - Add branch-check on `main`, and on git errors whose text holds line breaks.

  - [ ] 18.5 Extend the fail-closed property test to branch-check
    - **Property 16: Fail-closed hooks block on every failure**
    - **Validates: Requirements 1.12, 1.18**

- [ ] 19. Checkpoint - Ensure all tests pass before branch-check goes live
  - Run `uv run pytest -q` and fix every failure. Stop for the user only if a fix would change a requirement.

- [ ] 20. Add the branch-check hook file
  - [ ] 20.1 Create `.kiro/hooks/branch-check.json`
    - Use `createHook`: trigger `PreTaskExec`, no matcher, command `uv run --no-project -q python scripts/hooks/run.py branch-check`, timeout 20.
    - _Requirements: 1.1, 13.1, 17.1, 17.2, 17.4_

  - [ ] 20.2 Extend `tests/test_hook_files.py` to branch-check
    - Check its trigger, Launcher and timeout.
    - _Requirements: 1.1, 17.1, 17.2, 17.3_

- [ ] 21. Add the tests-after-task hook file
  - [ ] 21.1 Create `.kiro/hooks/tests-after-task.json`
    - Use `createHook` with an agent action: trigger `PostTaskExec` and the prompt from the design's "Hook files" section, word for word.
    - _Requirements: 14.1, 14.2, 14.3, 14.4, 17.1_

  - [ ] 21.2 Extend `tests/test_hook_files.py` to tests-after-task
    - Check the trigger and the agent action. The prompt names `uv run pytest -q` and exit code 5, says to fix failures before the next task and to tell the user when a fix changes a requirement, and asks for the devlog line with the property, the smallest failing input and the fix, with an input that looks like a key, a token or an account ARN described by its shape.
    - _Requirements: 14.1, 14.2, 14.3, 14.4, 17.1, 17.3_

- [ ] 22. Build devlog's script and its tests
  - [ ] 22.1 Implement `scripts/hooks/devlog.py`
    - `finish` and `build_line`, as in the design's "devlog" section, with `main(kind, stdin, root)` returning `quiet_main("devlog", finish, stdin, root)`. Find the spec and the task with the helper from task 18.1.
    - Read the Start_Record and delete it. Use it only when it's valid JSON for the same task and its `head` is null or an ancestor of HEAD (`git merge-base --is-ancestor`).
    - Take the changed files from `git log --name-only -z --format=` since the recorded `head` when HEAD has moved, and otherwise from `git status --porcelain=v1 -z`. Check whether `docs/devlog.md` was clean before writing.
    - Append exactly one line with the local time and UTC offset, the branch, the spec, the task and the files, each through `one_line`, adding a line break first if the file doesn't end in one.
    - Commit with `git commit -m <message> -- docs/devlog.md` only when the file was clean and the branch isn't `main`. Otherwise return the matching line from the design, and return one line when the commit fails. The Runner saves every returned line as a Devlog_Warning.
    - _Requirements: 1.13, 15.2, 15.3, 15.4, 15.5, 15.6, 15.7, 15.8, 15.9, 15.10_

  - [ ] 22.2 Write unit tests for devlog in `tests/test_devlog.py`
    - The line format with a fixed clock. In a temporary repo, files come from the commits since the Start_Record when HEAD moved, and from the status otherwise or with no record, new files included.
    - A record for another task, one whose `head` isn't an ancestor and one that isn't JSON each count as no record. The record is deleted after the run.
    - The commit holds only `docs/devlog.md`, and another staged file stays staged. Cover an already-changed `docs/devlog.md`, a run on `main` and a commit that fails.
    - Each warning is returned, and saved exactly once when devlog runs through the Runner's `main`.
    - Add devlog's idle Event, a `PostTaskExec` Event with no task fields in a folder with no git repo, to `tests/test_hook_speed.py`.
    - _Requirements: 1.11, 15.2, 15.3, 15.4, 15.6, 15.7, 15.8, 15.9, 15.10_

  - [ ] 22.3 Write property test for the devlog line
    - **Property 12: The devlog line is always one line**
    - **Validates: Requirements 19.12, 15.5**

  - [ ] 22.4 Extend the fail-open property test to devlog
    - **Property 40: The fail-open hooks always exit 0**
    - **Validates: Requirements 19.19, 1.13, 1.18, 15.11, 16.5**
    - Add a missing or broken `config.toml` in a temporary root, a git call that fails, and a folder in place of `docs/devlog.md` or of the warnings file, through the Runner's `main` in-process.

- [ ] 23. Checkpoint - Ensure all tests pass before devlog goes live
  - Run `uv run pytest -q` and fix every failure. Stop for the user only if a fix would change a requirement.

- [ ] 24. Add the devlog hook file
  - [ ] 24.1 Create `.kiro/hooks/devlog.json`
    - Use `createHook`: trigger `PostTaskExec` only, no matcher, command `uv run --no-project -q python scripts/hooks/run.py devlog`, timeout 20.
    - _Requirements: 1.1, 15.1, 17.1, 17.2, 17.4_

  - [ ] 24.2 Extend `tests/test_hook_files.py` to devlog
    - Check its one trigger, its Launcher and its timeout.
    - _Requirements: 1.1, 15.1, 17.1, 17.2, 17.3, 17.9_

- [ ] 25. Build session-status's script and its tests
  - [ ] 25.1 Implement `scripts/hooks/session_status.py`
    - `status`, `spend_sums`, `instances` and `take_warnings`, as in the design's "session-status" section, with `main(kind, stdin, root)` returning `quiet_main("session-status", status, stdin, root)`.
    - Print at most `[session_status].max_lines` lines in the design's order: the branch with the number of uncommitted files, the spend when the Spend_Log exists, every running instance on one line when the AWS call succeeds, and the saved Devlog_Warnings.
    - `take_warnings` moves the warnings file aside with `os.replace`, reads it and deletes it after the status is built. If it can't, the warnings line says so and the file stays.
    - `spend_sums` skips every line that isn't a JSON object with a readable ISO 8601 `time` and a finite numeric `usd` that isn't a bool, and sums with `math.fsum` for all lines and for today's UTC date.
    - Define the `AwsRunner` protocol here: argument list and timeout in, exit code, stdout and stderr out, and `TimeoutError` when the call runs out of time. The real one uses `subprocess.run` with no shell. `instances` runs the design's `aws ec2 describe-instances` argument list with the profile, region and tag from the Config and `[session_status].aws_seconds`, skips the call when `shutil.which("aws")` finds nothing, and leaves the line out on any failure without printing the CLI's output.
    - _Requirements: 1.13, 16.2, 16.4, 16.5, 16.6, 16.7, 16.8, 16.9, 16.10, 16.11, 16.12_

  - [ ] 25.2 Add a fake `AwsRunner` to `tests/conftest.py`
    - It returns the output, exit code or timeout that a test gives it, and records the argument list it was called with.
    - _Requirements: 1.15_

  - [ ] 25.3 Write unit tests for session-status in `tests/test_session_status.py`
    - In a temporary repo, the count includes staged, unstaged and untracked files.
    - The describe-instances call uses the exact argument list. That list, joined with `shlex.join` and split with `shellsplit` in the POSIX dialect, passes `guard_paid.judge_aws`. This test lives here, not in `tests/test_guard_paid.py`, because it checks session-status's own call and needs both modules, which exist from this task on.
    - A missing aws CLI, a missing profile, an error exit, a timeout and bad JSON each leave out the instance line, and none of the fake CLI's text is printed. An empty result prints `none`.
    - The warnings line holds the count and the latest warning, and a warnings file that can't be read stays for next time.
    - _Requirements: 16.4, 16.6, 16.9, 16.10, 16.11_

  - [ ] 25.4 Write property test for the spend total
    - **Property 11: The spend total counts only valid lines**
    - **Validates: Requirements 19.11, 16.7, 16.8**

  - [ ] 25.5 Write property test for the status lines
    - **Property 33: The status has at most four lines in order**
    - **Validates: Requirements 16.2, 16.12**

  - [ ] 25.6 Write property test for the saved devlog warnings
    - **Property 34: Saved devlog warnings are shown once, then cleared**
    - **Validates: Requirements 16.4**

  - [ ] 25.7 Extend the fail-open property test to session-status
    - **Property 40: The fail-open hooks always exit 0**
    - **Validates: Requirements 19.19, 1.13, 1.18, 15.11, 16.5**
    - Add a missing or broken `config.toml` in a temporary root, a git call that fails, and a folder in place of the warnings file, through the Runner's `main` in-process.

- [ ] 26. Checkpoint - Ensure all tests pass before session-status goes live
  - Run `uv run pytest -q` and fix every failure. Stop for the user only if a fix would change a requirement.

- [ ] 27. Add the session-status hook file
  - [ ] 27.1 Create `.kiro/hooks/session-status.json`
    - Use `createHook`: trigger `SessionStart` only, no matcher, command `uv run --no-project -q python scripts/hooks/run.py session-status`, timeout 30.
    - _Requirements: 1.1, 16.1, 17.1, 17.2, 17.4_

  - [ ] 27.2 Extend `tests/test_hook_files.py` to session-status
    - Check its one trigger, its Launcher and its timeout.
    - _Requirements: 1.1, 16.1, 17.1, 17.2, 17.3, 17.9_

- [ ] 28. Update the docs
  - [ ] 28.1 Add a "Hooks" section to `README.md`
    - Say that the hooks need uv on PATH and a Python 3.11 that uv can find, and that guard-licence in Kiro Web needs internet access to `pypi.org` and `registry.npmjs.org`.
    - Show how to run `uv run python scripts/check_licences.py`, and `--cloud` on the machine image.
    - Say how to turn a hook off: click the eye icon next to it in the Agent Hooks section of the Kiro panel, which applies at once, or set `"enabled": false` on its entries in `.kiro/hooks/<id>.json` and start a new chat.
    - Say what a broken setup does: when uv fails, the guards block every call they check and branch-check blocks every task until it's fixed or turned off, and nothing blocks a message.
    - Say how to get out when a broken guard blocks every call it checks, every file read and text search included for guard-secrets, so that Kiro can't even read the broken file: Kiro rewrites the file whole with a file-write tool, which no hook checks, or you restore the file with git in your own terminal or switch the hook off with the eye icon.
    - _Requirements: 18.3_

  - [ ] 28.2 Add a "Hooks" section to `docs/kiro-usage.md`
    - A table of each hook with its triggers and matchers, what it checks, and whether it fails closed or open.
    - The known limits of guard-paid and guard-secrets from the design.
    - _Requirements: 18.4_

  - [ ] 28.3 Write the docs tests in `tests/test_docs.py`
    - The licence table has the git, npm, AWS CLI v2 and uv rows. `rules.md` has the hooks line. The README covers each point of Requirement 18.3. The Hooks section of `docs/kiro-usage.md` names every hook, its triggers and whether it fails closed or open.
    - _Requirements: 18.1, 18.2, 18.3, 18.4_

- [ ] 29. Final check - Ensure all tests pass and the licence check exits 0
  - [ ] 29.1 Run the whole suite and the licence check
    - Run `uv run pytest -q`. Every test passes.
    - Run `uv run python scripts/check_licences.py` on the laptop install. It exits 0.
    - Check with `git log --oneline` that each hook file's commit comes after the commits of its script and tests, and that the licence rows, the `rules.md` hooks line and the `.gitignore` line came before the first hook file.
    - _Requirements: 1.15, 12.12, 17.4_

- [ ] 30. Live checks (the Live_Check_Task)
  - [ ] 30.1 Run every live check in a new chat
    - This is the only task that needs the user. The user runs it in a new Kiro IDE chat on the Windows laptop, started after every other task is committed, because hooks activate when a session starts, so only a new chat is sure to load every hook file as written (17.5, 17.6). Each hook entry gets one call or task it must block and one it must let run, or, for a hook that never blocks, one run that shows its output. Every must-block call is harmless if the hook lets it run.
    - Step 1, the gate: run `git clean -n`. It must be blocked with guard-git's `git clean` Reason. If it runs instead, stop here and leave this task unticked. Tell the user that either this chat started before the hook files were written or guard-git fails open, and that running this task again in a new chat tells which. If it's blocked with another `guard-git:` Reason, such as "couldn't run its own code", fix guard-git and its tests, and run this task again in a new chat. `git clean -n` is a dry run, so it only lists files, and no other guard judges it.
    - guard-git: step 1 is its must-block call. `git add -n docs/devlog.md` must run. It's a dry run too.
    - guard-paid (`shell`): first confirm with a file tool that `cloud/` holds neither `start.py` nor a `start` folder, and if it does, skip this call and tell the user. `uv run python -m cloud.start` must be blocked with a `guard-paid:` Reason, and `echo guard-paid-check` must run. Only the shell rules block the `cloud.start` call, so its block also shows that the entry applies them. Without `cloud/start.py`, Python only reports a missing module.
    - guard-paid (`@mcp`): through an MCP server's documentation-search tool, the one-line value `aws ec2 run-instances --profile kiroween-read --region us-east-1` must be blocked and `ec2:RunInstances` must run. Use only a tool that searches documentation, never one that runs commands or calls AWS. With no MCP server set up, report the entry as not checked.
    - guard-paid (`@powers`): the same two values through a power's documentation-search tool. With no power installed, report the entry as not checked.
    - guard-licence: `uv pip install --dry-run opencv-python` must be blocked without a lookup, and `uv pip install --dry-run iniconfig` must run after a real PyPI lookup. A dry run installs nothing.
    - guard-secrets (`shell`): `Get-Content data/hooks/live-check/.env` must be blocked, and `Get-Content .gitignore -TotalCount 1` must run. Never create that file, so Get-Content could only report that it's missing.
    - guard-secrets (`read`): `read_file` on `data/hooks/live-check/.env` and `grep_search` with the query `live-check` and the include pattern `data/hooks/live-check/.env` must both be blocked, and `read_file` on `.gitignore` must run. The `grep_search` block also shows that the `read` matcher selects the text search.
    - branch-check, in a throwaway clone: ask the user to run `git clone --branch feat/hooks <this repo> <a folder outside the repo>`, then `git switch -c main` and `uv sync` in the clone, and to open the clone in Kiro IDE. Write a throwaway spec `.kiro/specs/live-check/` there with two tasks, each of which only writes `data/hooks/live-check.txt`. Starting task 1 in a new chat on the clone's `main` must be blocked with a `branch-check:` Reason, and after `git switch feat/hooks` in the clone the same task must run. The task only writes a file that git ignores, and the real `main` is never touched. Ask the user to delete the clone afterwards.
    - tests-after-task and devlog, first run: in the real repo, commit `docs/devlog.md` with the results so far, so it has no uncommitted changes. Create the same two-task throwaway spec `.kiro/specs/live-check/` and never commit it. When the user runs its task 1 on `feat/hooks`, the agent must run `uv run pytest -q` after it, one new line must appear at the end of `docs/devlog.md`, and `git log -1 --stat` must show a commit that holds only `docs/devlog.md`.
    - devlog, second run: add the results so far to `docs/devlog.md` without committing them, and have the user run task 2. The line must be added without a commit, and `data/hooks/devlog-warnings.txt` must hold the warning.
    - session-status: ask the user to open a new chat and ask Kiro there to repeat the status it got at session start. It must have at most 4 lines, and the last one must name the saved Devlog_Warnings with the second run's warning as the latest. In a second new chat, the status must have no warnings line. The instance line shows only when the `kiroween-read` profile exists. If neither chat got a status, Session Start didn't fire: tell the user, record it in `docs/devlog.md`, and leave session-status on Session Start only (17.8). Neither chat is blocked either way, because session-status only reads.
    - Kiro Web: ask the user to push `feat/hooks`, or push it with their OK, and to open Kiro Web on that branch. There, `uv --version` must work, `uv run --no-project -q python --version` must report Python 3.11, and `uv run pytest -q` must pass. Then the user runs the guard-paid, guard-git, guard-licence, guard-secrets and session-status checks above in Kiro Web and tells you the results. Kiro Web runs no task triggers, so branch-check, tests-after-task and devlog aren't checked there, and it has no named AWS profiles, so session-status prints no instance line there.
    - If a must-block call runs or a must-run call is blocked, record it, tell the user, fix the hook with its tests, and repeat that check in a new chat. Never work around a block (17.7).
    - Add every result to `docs/devlog.md`, one line per hook entry, with "not checked" for an entry that selected no tool. Delete the throwaway spec `.kiro/specs/live-check/` and `data/hooks/live-check.txt`, and commit `docs/devlog.md`.
    - _Requirements: 1.16, 3.1, 8.1, 10.1, 13.1, 13.2, 14.1, 15.2, 15.6, 15.7, 15.10, 16.3, 16.4, 16.5, 17.5, 17.6, 17.7, 17.8, 20.1_

## Notes

- Every test task is required, as rules.md and Requirement 19.13 say. No task is marked optional.
- Every property in the design, 1 to 43, has its own test task. Properties 1 to 12 are the Hypothesis tests for Requirements 19.1 to 19.12, and Properties 35 to 40 for 19.14 to 19.19.
- Properties 5, 7, 14, 15, 16, 40 and 43 cover several scripts. Each is first written for the first script it applies to and then extended in each later section, so every script is fully tested before its hook file goes live.
- Only task 30 needs the user. Its first step stops the task, unticked, when the hooks aren't live in the chat, so "run all tasks" ends there instead of reporting false failures.
- Requirement 20.10 (a path to a `.aws` folder counts as naming a Secret_File) is tested by Property 35 in task 8.6 and by tasks 8.2 and 8.5.
- Kiro's file-write tools match no hook's matcher, so a hook that blocks everything can always be fixed with a file edit. When guard-secrets blocks the read tools too, Kiro rewrites the broken file whole instead of reading it first. The user can always switch a hook off with the eye icon in the Agent Hooks panel, which needs no prompt and applies at once.
- MoGe's `pipeline` and our `pipeline/` package share a name. The next spec renames ours, so no task here touches it.

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "2.1", "2.2", "2.3"] },
    { "id": 1, "tasks": ["1.2", "3.1"] },
    { "id": 2, "tasks": ["1.3", "1.5", "3.3"] },
    { "id": 3, "tasks": ["1.4", "1.6", "3.2"] },
    { "id": 4, "tasks": ["1.7", "1.9", "3.4"] },
    { "id": 5, "tasks": ["1.8", "1.10"] },
    { "id": 6, "tasks": ["1.11"] },
    { "id": 7, "tasks": ["1.12"] },
    { "id": 8, "tasks": ["1.13"] },
    { "id": 9, "tasks": ["5.1", "5.3"] },
    { "id": 10, "tasks": ["5.2", "5.4"] },
    { "id": 11, "tasks": ["5.5"] },
    { "id": 12, "tasks": ["5.6"] },
    { "id": 13, "tasks": ["5.7", "5.8"] },
    { "id": 14, "tasks": ["5.9"] },
    { "id": 15, "tasks": ["5.10"] },
    { "id": 16, "tasks": ["5.11"] },
    { "id": 17, "tasks": ["5.12"] },
    { "id": 18, "tasks": ["5.13"] },
    { "id": 19, "tasks": ["5.14"] },
    { "id": 20, "tasks": ["5.15"] },
    { "id": 21, "tasks": ["5.16"] },
    { "id": 22, "tasks": ["5.17"] },
    { "id": 23, "tasks": ["5.18"] },
    { "id": 24, "tasks": ["5.19"] },
    { "id": 25, "tasks": ["5.20"] },
    { "id": 26, "tasks": ["5.21"] },
    { "id": 27, "tasks": ["5.22"] },
    { "id": 28, "tasks": ["5.23"] },
    { "id": 29, "tasks": ["5.24"] },
    { "id": 30, "tasks": ["7.1"] },
    { "id": 31, "tasks": ["7.2"] },
    { "id": 32, "tasks": ["8.1"] },
    { "id": 33, "tasks": ["8.2", "8.3"] },
    { "id": 34, "tasks": ["8.4"] },
    { "id": 35, "tasks": ["8.5", "8.6"] },
    { "id": 36, "tasks": ["8.7"] },
    { "id": 37, "tasks": ["8.8"] },
    { "id": 38, "tasks": ["8.9"] },
    { "id": 39, "tasks": ["8.10"] },
    { "id": 40, "tasks": ["8.11"] },
    { "id": 41, "tasks": ["8.12"] },
    { "id": 42, "tasks": ["8.13"] },
    { "id": 43, "tasks": ["8.14"] },
    { "id": 44, "tasks": ["10.1"] },
    { "id": 45, "tasks": ["10.2"] },
    { "id": 46, "tasks": ["11.1", "11.3"] },
    { "id": 47, "tasks": ["11.2", "11.4", "11.6"] },
    { "id": 48, "tasks": ["11.5", "11.7"] },
    { "id": 49, "tasks": ["11.8", "11.9"] },
    { "id": 50, "tasks": ["11.10", "11.11"] },
    { "id": 51, "tasks": ["11.12", "11.16"] },
    { "id": 52, "tasks": ["11.13", "11.17"] },
    { "id": 53, "tasks": ["11.14"] },
    { "id": 54, "tasks": ["11.15"] },
    { "id": 55, "tasks": ["11.18"] },
    { "id": 56, "tasks": ["11.19"] },
    { "id": 57, "tasks": ["11.20"] },
    { "id": 58, "tasks": ["11.21"] },
    { "id": 59, "tasks": ["11.22"] },
    { "id": 60, "tasks": ["11.23"] },
    { "id": 61, "tasks": ["11.24"] },
    { "id": 62, "tasks": ["11.25"] },
    { "id": 63, "tasks": ["13.1"] },
    { "id": 64, "tasks": ["13.2"] },
    { "id": 65, "tasks": ["14.1", "14.2"] },
    { "id": 66, "tasks": ["14.3", "14.7", "14.9"] },
    { "id": 67, "tasks": ["14.4", "14.8", "14.10"] },
    { "id": 68, "tasks": ["14.5", "14.11"] },
    { "id": 69, "tasks": ["14.6"] },
    { "id": 70, "tasks": ["14.12"] },
    { "id": 71, "tasks": ["14.13"] },
    { "id": 72, "tasks": ["14.14"] },
    { "id": 73, "tasks": ["14.15"] },
    { "id": 74, "tasks": ["14.16"] },
    { "id": 75, "tasks": ["14.17"] },
    { "id": 76, "tasks": ["14.18"] },
    { "id": 77, "tasks": ["14.19"] },
    { "id": 78, "tasks": ["14.20"] },
    { "id": 79, "tasks": ["16.1"] },
    { "id": 80, "tasks": ["16.2"] },
    { "id": 81, "tasks": ["17.1", "18.1"] },
    { "id": 82, "tasks": ["17.2", "17.3", "18.2"] },
    { "id": 83, "tasks": ["18.3"] },
    { "id": 84, "tasks": ["18.4"] },
    { "id": 85, "tasks": ["18.5"] },
    { "id": 86, "tasks": ["20.1"] },
    { "id": 87, "tasks": ["20.2"] },
    { "id": 88, "tasks": ["21.1"] },
    { "id": 89, "tasks": ["21.2"] },
    { "id": 90, "tasks": ["22.1"] },
    { "id": 91, "tasks": ["22.2", "22.3"] },
    { "id": 92, "tasks": ["22.4"] },
    { "id": 93, "tasks": ["24.1"] },
    { "id": 94, "tasks": ["24.2"] },
    { "id": 95, "tasks": ["25.1"] },
    { "id": 96, "tasks": ["25.2", "25.4"] },
    { "id": 97, "tasks": ["25.3", "25.5"] },
    { "id": 98, "tasks": ["25.6"] },
    { "id": 99, "tasks": ["25.7"] },
    { "id": 100, "tasks": ["27.1"] },
    { "id": 101, "tasks": ["27.2"] },
    { "id": 102, "tasks": ["28.1", "28.2"] },
    { "id": 103, "tasks": ["28.3"] },
    { "id": 104, "tasks": ["29.1"] },
    { "id": 105, "tasks": ["30.1"] }
  ]
}
```
