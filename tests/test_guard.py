"""Tests for the guard hook, scripts/hooks/guard.py."""

import importlib.util
import io
import json
import string
import subprocess
import sys
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

ROOT = Path(__file__).resolve().parents[1]
GUARD_PATH = ROOT / "scripts" / "hooks" / "guard.py"
_spec = importlib.util.spec_from_file_location("guard", GUARD_PATH)
guard = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(guard)
CONFIG = guard.load_config()


def reason(command: str, cwd: Path = ROOT) -> str | None:
    return guard.reason(command, CONFIG, str(cwd))


def fake_repo(tmp_path: Path, branch: str) -> Path:
    (tmp_path / ".git").mkdir(parents=True)
    (tmp_path / ".git" / "HEAD").write_text(f"ref: refs/heads/{branch}\n", encoding="utf-8")
    return tmp_path


# Unit tests


@pytest.mark.parametrize("command", [
    "aws ec2 run-instances --image-id ami-123",
    "aws --region us-east-1 ec2 start-instances --instance-ids i-1",
    "aws ec2 create-image --instance-id i-1 --name x",
    "aws cloudformation create-stack --stack-name x",
    "aws cloudformation update-stack --stack-name x",
    'cd infra; & "C:\\Program Files\\Amazon\\AWSCLIV2\\aws.exe" cloudformation deploy --template-file t.yml',
])
def test_blocks_raw_paid_aws_commands(command: str) -> None:
    assert reason(command).startswith("guard: blocked `aws ")


@pytest.mark.parametrize("command", [
    "git push --force",
    "git push -f origin feat/x",
    "git push -uf origin feat/x",
    "git push --force-with-lease origin feat/x",
    "git push origin +feat/x",
])
def test_blocks_force_pushes(command: str) -> None:
    assert reason(command) == guard.FORCE


@pytest.mark.parametrize("command", [
    "git push origin main",
    "git push -u origin HEAD:main 2>&1",
    "git push origin feat/x:refs/heads/main",
    "git push --all origin",
])
def test_blocks_pushes_to_main(command: str) -> None:
    assert reason(command) == guard.MAIN.format("main")


def test_a_plain_push_is_blocked_only_on_main(tmp_path: Path) -> None:
    assert reason("git push", fake_repo(tmp_path / "a", "main")) == guard.MAIN.format("main")
    assert reason("git push origin HEAD", fake_repo(tmp_path / "b", "feat/x")) is None


@pytest.mark.parametrize("command", [
    "git add -A",
    "git add .",
    "git add --all",
    "git -C docs add -v .",
    'bash -c "git add -A"',
])
def test_blocks_staging_everything(command: str) -> None:
    assert reason(command) == guard.ADD


@pytest.mark.parametrize("command", [
    "git status",
    "git add docs/devlog.md scripts/hooks/guard.py",
    "git push -u origin feat/hooks",
    'git commit -m "Block git push --force, git add . and aws ec2 run-instances"',
    "aws ec2 describe-instances --profile kiroween-read --region us-east-1",
    "aws s3 cp data/job.json s3://bucket/job.json --profile kiroween-runner",
    "$env:APPROVED_SPEND = '1'; uv run python -m cloud.start",
    "APPROVED_SPEND=1 uv run python -m cloud.smoke_test",
])
def test_lets_other_aws_and_git_commands_run(command: str) -> None:
    assert reason(command) is None


def run_main(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture, data: bytes) -> tuple[int, str]:
    monkeypatch.setattr(sys, "stdin", io.TextIOWrapper(io.BytesIO(data)))
    code = guard.main()
    return code, capsys.readouterr().err


def test_main_reads_the_command_from_kiros_hook_input(monkeypatch, capsys) -> None:
    event = {"hook_event_name": "PreToolUse", "cwd": str(ROOT), "tool_name": "execute_pwsh",
             "tool_input": {"command": "git status; git add -A", "cwd": None}}
    code, err = run_main(monkeypatch, capsys, json.dumps(event).encode())
    assert (code, err) == (2, guard.ADD + "\n")
    event["tool_input"]["command"] = "git status"
    assert run_main(monkeypatch, capsys, json.dumps(event).encode()) == (0, "")


def test_main_blocks_when_it_cant_read_the_hook_input(monkeypatch, capsys) -> None:
    code, err = run_main(monkeypatch, capsys, b"not json")
    assert code == 2
    assert err.startswith("guard: couldn't check this command") and err.count("\n") == 1


def test_the_hook_command_exits_2_with_one_line() -> None:
    event = {"tool_name": "execute_pwsh", "tool_input": {"command": "git push --force"}}
    result = subprocess.run([sys.executable, str(GUARD_PATH)], input=json.dumps(event).encode(),
                            capture_output=True, cwd=ROOT)
    assert result.returncode == 2
    assert result.stderr.decode().splitlines() == [guard.FORCE]


# config.toml as a merge leaves it when both sides changed the same lines. The three
# marker lines are built in code, so git diff --check and marker searches don't flag this file.
CONFLICTED_CONFIG = "\n".join([
    "[git]",
    'main_branch = "main"',
    "",
    "<" * 7 + " HEAD",
    "[clean]",
    "voxel_size = 0.02",
    "=" * 7,
    "[frames]",
    "count = 200",
    ">" * 7 + " origin/main",
    "",
])


def conflicted_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A repo root on main whose config.toml holds git's merge markers, with the guard pointed at it."""
    root = fake_repo(tmp_path, "main")
    (root / "config.toml").write_text(CONFLICTED_CONFIG, encoding="utf-8")
    monkeypatch.setattr(guard, "ROOT", str(root))
    return root


def hook_input(command: str, cwd: Path) -> bytes:
    event = {"hook_event_name": "PreToolUse", "cwd": str(cwd), "tool_name": "execute_pwsh",
             "tool_input": {"command": command}}
    return json.dumps(event).encode()


def test_main_lets_read_only_git_run_when_config_toml_has_merge_markers(tmp_path, monkeypatch, capsys) -> None:
    root = conflicted_repo(tmp_path, monkeypatch)
    assert run_main(monkeypatch, capsys, hook_input("git status", root)) == (0, "")


def test_main_still_blocks_a_push_on_main_when_config_toml_has_merge_markers(tmp_path, monkeypatch, capsys) -> None:
    root = conflicted_repo(tmp_path, monkeypatch)
    assert run_main(monkeypatch, capsys, hook_input("git push", root)) == (2, guard.MAIN.format("main") + "\n")


@pytest.mark.parametrize("command", [
    "aws sts get-caller-identity --profile kiroween-read --region us-east-1",
    "aws ec2 run-instances --image-id ami-123",
])
def test_main_blocks_every_aws_call_when_config_toml_has_merge_markers(tmp_path, monkeypatch, capsys,
                                                                      command: str) -> None:
    root = conflicted_repo(tmp_path, monkeypatch)
    code, err = run_main(monkeypatch, capsys, hook_input(command, root))
    assert code == 2 and err.count("\n") == 1
    assert err.startswith("guard: blocked an aws call because config.toml can't be read (TOMLDecodeError: ")


# The CloudShell commands' modules (website spec, Req 1.11)

@pytest.mark.parametrize("command", [
    "python -m app.deploy",
    "APPROVED_SPEND=1 python3.11 -m app.deploy",
    "uv run python -m app.account_command judges",
    "python3.11 -m app.showcase_command list",
    "python app/deploy.py",
    "uv run app/account_command.py judges --new-password",
    "python3 -u ./app/showcase_command.py list",
    'bash -c "python3.11 -m app.deploy"',
    "cd /tmp/voxelis-website/repo && python3.11 -m app.deploy",
])
def test_blocks_the_cloudshell_commands(command: str) -> None:
    assert reason(command).startswith("guard: blocked `app.")
    assert "CloudShell" in reason(command)


@pytest.mark.parametrize("command", [
    "git add app/deploy.py app/account_command.py",
    "uv run pytest tests/test_website_deploy.py",
    "grep -m 1 deploy app/deploy.py",
    "python -m app.lockfile app/uv.lock",
    "uv run python -m cloud.status 20261201-120000-abcdef",
])
def test_lets_other_module_runs_and_paths_through(command: str) -> None:
    assert reason(command) is None


def test_the_cloudshell_modules_come_from_config() -> None:
    config = {**CONFIG, "website": {"cloudshell_modules": ["app.other"]}}
    assert guard.reason("python -m app.other", config, str(ROOT)).startswith("guard: blocked `app.other`")
    assert guard.reason("python -m app.deploy", config, str(ROOT)) is None


def test_main_blocks_a_cloudshell_command_when_config_toml_has_merge_markers(tmp_path, monkeypatch,
                                                                           capsys) -> None:
    root = conflicted_repo(tmp_path, monkeypatch)
    code, err = run_main(monkeypatch, capsys, hook_input("python3.11 -m app.deploy", root))
    assert code == 2 and err == guard.CLOUDSHELL.format("app.deploy") + "\n"


# Property tests

SAFE = string.ascii_letters + string.digits + "-._/=:,@%+"


def unrelated(text: str) -> bool:
    """True when the text never spells aws or git, even with quotes, escapes and line breaks removed."""
    plain = "".join(ch for ch in text if ch not in "'\"`\\\r\n").lower()
    return "aws" not in plain and "git" not in plain


word = st.one_of(
    st.text(SAFE, min_size=1, max_size=10),
    st.builds(lambda q, s: q + s + q, st.sampled_from(["'", '"']), st.text(SAFE + " ;&|()", max_size=10)),
).filter(unrelated)
command = st.lists(word, min_size=1, max_size=5).map(" ".join)
separator = st.sampled_from(["; ", " && ", " || ", " | ", "\n", " & "])
aws_option = st.sampled_from(["--region us-east-1", "--region=us-east-1", "--profile kiroween-read",
                              "--output json", "--debug", "--no-cli-pager"])


@st.composite
def paid_aws_command(draw) -> tuple[str, str]:
    """A raw paid aws command from config.toml, spelled and wrapped in different ways."""
    call = draw(st.sampled_from(CONFIG["paid"]["aws"]))
    service, operation = call.split()
    spelling = draw(st.sampled_from(["aws", "AWS", "aws.exe", "aws.cmd", "/usr/local/bin/aws",
                                     '"C:\\Program Files\\Amazon\\AWSCLIV2\\aws.exe"']))
    options = [draw(st.lists(aws_option, max_size=2)) for _ in range(3)]
    args = draw(st.lists(word, max_size=3))
    text = " ".join([spelling, *options[0], service, *options[1], operation, *options[2], *args])
    prefix = draw(st.sampled_from(["", "sudo ", "env AWS_PAGER= ", "& ", "uv run ", "time "]))
    wrapper = draw(st.sampled_from(["{}", "{} 2>&1", "bash -c '{}'", "pwsh -NoProfile -Command '{}'",
                                    "cmd /c {}", "$({})", "({})"]))
    return wrapper.format(prefix + text), call


@given(st.lists(command, max_size=4), paid_aws_command(), st.data())
def test_raw_paid_aws_commands_are_always_blocked(others, paid, data) -> None:
    text, call = paid
    position = data.draw(st.integers(0, len(others)))
    parts = others[:position] + [text] + others[position:]
    line = parts[0] + "".join(data.draw(separator) + part for part in parts[1:])
    assert reason(line) == guard.PAID.format(call)


programs = st.sampled_from(["uv run pytest", "python", "ls", "echo", "Get-ChildItem", "Select-String",
                            "npm", "node", "ffmpeg", "colmap", "Write-Host", "mkdir", "cat"])
unrelated_line = st.lists(st.tuples(programs, command), min_size=1, max_size=4).map(
    lambda pairs: "; ".join(f"{program} {args}" for program, args in pairs))


@given(st.one_of(st.text(), unrelated_line).filter(unrelated))
def test_commands_without_aws_or_git_always_pass(line: str) -> None:
    assert reason(line) is None
