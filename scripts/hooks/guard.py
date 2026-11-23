"""Kiro hook, before every shell command: block the commands our rules forbid.

It blocks raw aws commands that start paid things (the list is `[paid].aws` in
config.toml), force pushes, pushes to main, and `git add -A` or `git add .`.

Kiro sends the tool call as JSON on stdin. Exit 2 with one line on stderr
blocks the call and tells Kiro what to do instead. Exit 0 lets it run.
Standard library only. Run it with:

    uv run --no-project -q python scripts/hooks/guard.py
"""

from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Programs that run their arguments as another command line.
SHELLS = {"bash", "sh", "zsh", "dash", "pwsh", "powershell", "cmd", "iex", "invoke-expression"}
# Options that take the next word as their value.
GIT_VALUE_OPTIONS = {"-C", "-c", "--git-dir", "--work-tree", "--namespace", "--config-env"}
PUSH_VALUE_OPTIONS = {"--repo", "--receive-pack", "--exec", "--push-option", "--recurse-submodules"}
AWS_VALUE_OPTIONS = {
    "--profile", "--region", "--output", "--query", "--endpoint-url", "--color",
    "--ca-bundle", "--cli-read-timeout", "--cli-connect-timeout", "--cli-binary-format",
}
# Pathspecs that stage everything.
ADD_EVERYTHING = {".", "./", ".\\", "*", ":/"}

PAID = ("guard: blocked `aws {}`, which starts something paid. Paid runs only go through our "
        "start, image-build or smoke-test command, after the user says yes to the machine, the "
        "time and the cost.")
FORCE = ("guard: blocked a force push. Push without --force, -f or a + refspec, and if the push is "
         "rejected, tell the user instead.")
MAIN = "guard: blocked a push to {}. Push your feature branch instead and open a pull request."
ADD = ("guard: blocked `git add -A` or `git add .`. Stage the files by name instead, for example "
       "`git add path/to/file`.")


def split_commands(text: str, escape: str) -> list[list[str]]:
    """Split shell text into commands, each a list of words with the quotes removed.

    `escape` is the shell's escape character: a backtick for PowerShell, a
    backslash for POSIX shells. Commands end at ; & | ( ) { } < > and line breaks.
    """
    separators = ";&|\n\r(){}<>" + ("`" if escape == "\\" else "")
    commands: list[list[str]] = []
    words: list[str] = []
    word: list[str] = []
    in_word = False
    quote = ""
    i = 0
    while i < len(text):
        ch = text[i]
        i += 1
        if ch == escape and quote != "'" and i < len(text):
            ch = text[i]
            i += 1
            if ch not in "\r\n":  # an escaped line break joins the lines
                word.append(ch)
                in_word = True
        elif quote:
            if ch == quote:
                quote = ""
            else:
                word.append(ch)
        elif ch in "'\"":
            quote = ch
            in_word = True
        elif ch.isspace() or ch in separators:
            if ch in "<>" and "".join(word).isdigit():
                word, in_word = [], False  # the 2 in 2>&1
            if in_word:
                words.append("".join(word))
                word, in_word = [], False
            if ch in separators and words:
                commands.append(words)
                words = []
        else:
            word.append(ch)
            in_word = True
    if in_word:
        words.append("".join(word))
    if words:
        commands.append(words)
    return commands


def program(word: str) -> str:
    """The program a word names: `C:\\Git\\cmd\\git.exe` is `git`."""
    name = word.replace("\\", "/").rsplit("/", 1)[-1].lower()
    for extension in (".exe", ".cmd", ".bat"):
        if name.endswith(extension):
            return name[: -len(extension)]
    return name


def calls(text: str) -> list[tuple[str, list[str]]]:
    """Every aws and git call in shell text, as (program, the words after it).

    The text is read both as PowerShell and as a POSIX shell. A shell such as
    `bash -c "..."` has the rest of its command read again as a command line.
    """
    found: list[tuple[str, list[str]]] = []
    todo, seen = [text], set()
    while todo:
        text = todo.pop()
        if text in seen:
            continue
        seen.add(text)
        for escape in ("`", "\\"):
            for words in split_commands(text, escape):
                names = [program(word) for word in words]
                for target in ("aws", "git"):
                    if target in names:
                        found.append((target, words[names.index(target) + 1:]))
                shell = next((i for i, name in enumerate(names) if name in SHELLS), None)
                if shell is not None:
                    rest = " ".join(words[shell + 1:])
                    if "aws" in rest.lower() or "git" in rest.lower():
                        todo.append(rest)
    return found


def judge_aws(args: list[str], paid: list[str]) -> str | None:
    """Block an aws call whose service and operation are in the paid list."""
    words: list[str] = []
    skip = False
    for arg in args:
        if skip:
            skip = False
        elif arg.startswith("-"):
            skip = arg in AWS_VALUE_OPTIONS
        else:
            words.append(arg.lower())
            if len(words) == 2:
                break
    call = " ".join(words)
    return PAID.format(call) if call in paid else None


def judge_git(args: list[str], main: str, cwd: str) -> str | None:
    """Block git push and git add calls that break our rules."""
    i = 0
    while i < len(args) and args[i].startswith("-"):
        if args[i] == "-C" and i + 1 < len(args):
            cwd = os.path.join(cwd, args[i + 1])
        i += 2 if args[i] in GIT_VALUE_OPTIONS else 1
    if i >= len(args):
        return None
    if args[i] == "push":
        return judge_push(args[i + 1:], main, cwd)
    if args[i] in ("add", "stage"):
        return judge_add(args[i + 1:])
    return None


def judge_push(args: list[str], main: str, cwd: str) -> str | None:
    """Block a force push, and a push whose destination is main."""
    force = every_branch = False
    positional: list[str] = []
    i = 0
    while i < len(args):
        arg = args[i]
        i += 1
        if arg == "--":
            positional += args[i:]
            break
        if not arg.startswith("-") or arg == "-":
            positional.append(arg)
        elif arg.startswith("--"):
            name = arg.split("=", 1)[0]
            force = force or name.startswith("--force")
            every_branch = every_branch or name in ("--all", "--branches", "--mirror")
            if name in PUSH_VALUE_OPTIONS and "=" not in arg:
                i += 1
        else:
            letters, has_o, value = arg[1:].partition("o")  # -o takes a value
            force = force or "f" in letters
            if has_o and not value:
                i += 1
    refspecs = positional[1:]  # the first one is the remote
    if force or any(spec.startswith("+") for spec in refspecs):
        return FORCE
    destinations = [destination(spec) for spec in refspecs]
    if every_branch or any(dest in (main, "refs/heads/" + main) for dest in destinations):
        return MAIN.format(main)
    pushes_current = not refspecs or any(dest in ("HEAD", "@") for dest in destinations)
    if pushes_current and current_branch(cwd) == main:
        return MAIN.format(main)
    return None


def destination(refspec: str) -> str:
    """The branch a refspec updates: `HEAD:main` and `+main` both update main."""
    source, _, dest = refspec.lstrip("+").rpartition(":")
    return dest or source


def current_branch(cwd: str) -> str | None:
    """The branch checked out in the repo that holds `cwd`, read from .git/HEAD."""
    path = os.path.abspath(cwd)
    while True:
        git_dir = os.path.join(path, ".git")
        if os.path.isfile(git_dir):  # a worktree's .git file says "gitdir: <path>"
            with open(git_dir, encoding="utf-8") as file:
                git_dir = os.path.join(path, file.read().strip().removeprefix("gitdir:").strip())
        if os.path.isdir(git_dir):
            with open(os.path.join(git_dir, "HEAD"), encoding="utf-8") as file:
                head = file.read().strip()
            prefix = "ref: refs/heads/"
            return head[len(prefix):] if head.startswith(prefix) else None
        parent = os.path.dirname(path)
        if parent == path:
            return None
        path = parent


def judge_add(args: list[str]) -> str | None:
    """Block `git add -A`, `git add --all` and `git add .`."""
    options = True
    for arg in args:
        if options and arg == "--":
            options = False
        elif options and arg.startswith("--"):
            if arg in ("--all", "--no-ignore-removal"):
                return ADD
        elif options and arg.startswith("-") and len(arg) > 1:
            if "A" in arg[1:]:
                return ADD
        elif arg in ADD_EVERYTHING:
            return ADD
    return None


def judge(name: str, args: list[str], config: dict, cwd: str) -> str | None:
    """Why the guard blocks one aws or git call, or None when it may run."""
    if name == "aws":
        return judge_aws(args, config["paid"]["aws"])
    return judge_git(args, config["git"]["main_branch"], cwd)


def reason(command: str, config: dict, cwd: str) -> str | None:
    """Why the guard blocks a shell command, or None when it may run."""
    for name, args in calls(command):
        why = judge(name, args, config, cwd)
        if why:
            return why
    return None


def load_config() -> dict:
    import tomllib  # only imported when there is an aws or git call to judge

    with open(os.path.join(ROOT, "config.toml"), "rb") as file:
        return tomllib.load(file)


def command_texts(event: object) -> list[str]:
    """Every string under a "command" key, anywhere in Kiro's hook input."""
    found: list[str] = []
    todo = [event]
    while todo:
        value = todo.pop()
        if isinstance(value, dict):
            for key, item in value.items():
                if key == "command" and isinstance(item, str):
                    found.append(item)
                else:
                    todo.append(item)
        elif isinstance(value, list):
            todo.extend(value)
    return found


def working_dir(event: object) -> str:
    """The folder the command runs in: the tool's own cwd, else the session's."""
    if not isinstance(event, dict):
        return os.getcwd()
    base = event.get("cwd") if isinstance(event.get("cwd"), str) and event.get("cwd") else os.getcwd()
    tool_input = event.get("tool_input")
    inner = tool_input.get("cwd") if isinstance(tool_input, dict) else None
    return os.path.join(base, inner) if isinstance(inner, str) and inner else base


def block(line: str) -> int:
    sys.stderr.buffer.write((" ".join(line.split()) + "\n").encode("utf-8", "replace"))
    sys.stderr.flush()
    return 2


def main() -> int:
    try:
        event = json.loads(sys.stdin.buffer.read().decode("utf-8", "replace"))
        found = [call for text in command_texts(event) for call in calls(text)]
        if not found:
            return 0
        config, cwd = load_config(), working_dir(event)
        for name, args in found:
            why = judge(name, args, config, cwd)
            if why:
                return block(why)
        return 0
    except Exception as error:  # an uncaught error exits 1, which would let the command run
        return block(f"guard: couldn't check this command ({type(error).__name__}: {error}). "
                     "Tell the user that scripts/hooks/guard.py needs a fix.")


if __name__ == "__main__":
    sys.exit(main())
