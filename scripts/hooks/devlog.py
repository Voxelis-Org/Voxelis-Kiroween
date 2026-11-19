"""Kiro hook, after a spec task: add one line to the devlog and commit just that file.

The line holds the time, the branch, the task and the changed files: the files
in the commits since the last devlog commit, plus anything not committed yet.
Kiro sends the task as JSON on stdin (`spec_name`, `task_name`). The settings
are `[devlog]` and `[git].main_branch` in config.toml. Standard library only.
Run it with:

    uv run --no-project -q python scripts/hooks/devlog.py
"""

from __future__ import annotations

import datetime
import json
import os
import subprocess
import sys
import tomllib

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def git(*args: str) -> str:
    """Run git in the repo root and return its output, or raise with git's first error line."""
    result = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                            encoding="utf-8", errors="replace")
    if result.returncode != 0:
        lines = (result.stderr or result.stdout).strip().splitlines() or ["no output"]
        raise RuntimeError(f"git {args[0]} failed: {lines[0]}")
    return result.stdout


def one_line(text: str) -> str:
    return " ".join(text.split())


def task_text(event: object) -> str:
    """The spec and task from Kiro's hook input, for example `hooks 2. Add the guard`."""
    if not isinstance(event, dict):
        return "task finished"
    parts = [event.get(key) for key in ("spec_name", "task_name")]
    text = one_line(" ".join(part for part in parts if isinstance(part, str)))
    if event.get("task_success") is False:
        text += " (failed)"
    return text or "task finished"


def last_devlog_commit(prefix: str) -> str | None:
    """The newest commit whose subject starts with the devlog's commit message prefix."""
    out = git("log", "--fixed-strings", f"--grep={prefix}", "--format=%H %s", "HEAD")
    for line in out.splitlines():
        sha, _, subject = line.partition(" ")
        if subject.startswith(prefix):
            return sha
    return None


def changed_files(prefix: str) -> list[str]:
    """Files in the commits since the last devlog commit, then the uncommitted ones."""
    since = last_devlog_commit(prefix)
    commits = [f"{since}..HEAD"] if since else ["-1", "HEAD"]
    files = git("log", "--format=", "--name-only", *commits).splitlines()
    for line in git("status", "--porcelain").splitlines():
        files.append(line[3:].split(" -> ")[-1])  # a rename shows as "old -> new"
    return list(dict.fromkeys(name for name in files if name.strip()))


def append_line(path: str, line: str) -> None:
    with open(path, "a+", encoding="utf-8", newline="\n") as file:
        file.seek(0)
        text = file.read()
        file.write(("" if text.endswith("\n") or not text else "\n") + line + "\n")


def run(event: object, config: dict) -> str:
    """Add the line and commit it. Returns one line saying what happened."""
    settings = config["devlog"]
    task = task_text(event)
    prefix = settings["commit_message"].split("{task}")[0]
    branch = git("rev-parse", "--abbrev-ref", "HEAD").strip()
    files = ", ".join(changed_files(prefix)) or "no files changed"
    time = datetime.datetime.now().astimezone().isoformat(sep=" ", timespec="minutes")
    append_line(os.path.join(ROOT, settings["path"]), f"- {time} | {branch} | {task} | {files}")
    if branch == config["git"]["main_branch"]:
        return f"devlog: added the line to {settings['path']} but didn't commit it, because you're on {branch}."
    git("add", "--", settings["path"])
    git("commit", "-q", "-m", settings["commit_message"].format(task=task), "--", settings["path"])
    return f"devlog: added the line for {task} and committed {settings['path']}."


def main() -> int:
    try:
        raw = sys.stdin.buffer.read().decode("utf-8", "replace")
        try:
            event = json.loads(raw)
        except json.JSONDecodeError:
            event = None
        with open(os.path.join(ROOT, "config.toml"), "rb") as file:
            config = tomllib.load(file)
        message = run(event, config)
    except Exception as error:  # a task has already finished, so never fail loudly
        message = (f"devlog: couldn't add the devlog line ({one_line(str(error))}). "
                   "Add it to docs/devlog.md by hand, and tell the user if this keeps happening.")
    print(message)
    return 0


if __name__ == "__main__":
    sys.exit(main())
