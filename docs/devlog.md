# Devlog

## 2026-11-17: Testing the reviewer's permissions

The reviewer agent (`.kiro/agents/reviewer.md`) may only read the repo and run `git diff`, `git log` and the tests. Asked to try forbidden steps, it refused them itself, which only tests its instructions. So Kiro ran a temporary copy with the same permission rules and a prompt that tries every step, then deleted it. Each refusal came back as a permission error from the agent's profile, not from the agent.

What the test found, and how the rules block it:

- **Paths outside the repo.** Given a path outside the repo, `git diff` compares against that file and prints it ([git docs](https://git-scm.com/docs/git-diff)). The read rules don't apply to shell commands, so the allow rule `git diff *` would have let `git diff README.md ~/.aws/credentials` print the AWS credentials. A new deny rule refuses commands that contain `.aws`, a `$` variable, a `..` path, a drive letter like `C:\`, or an argument starting with `~`, `/` or `\`. Every outside path we tried was refused, and relative paths inside the repo still work.
- **Commands hidden in `(...)`.** `git log -n 1 (Get-Date)` runs a second command inside an allowed one. Kiro's command parser checks the inner command on its own, so the catch-all deny rule already refused it, written as `(...)` or `@(...)`.
- **Options that write files.** A deny rule matches them anywhere in a command. `>`, `2>&1`, `--output` and `--junitxml` were all refused. The same rule lists pytest's other options that write files or use the network, but we didn't try each one.
