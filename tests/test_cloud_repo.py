"""Unit tests for the Start_Command's git helpers, cloud/repo.py (cloud-jobs task 5.4).

The four helpers shell out to `git`, a dev tool, so each test builds a tiny committed repo with the
shared `git_repo(tmp_path)` helper, which skips the test, with a reason, when `git` isn't installed,
the way pipeline-core skips a test that needs ffprobe. There is no network and no AWS here.

- `uncommitted` lists the given paths that have a change: a dirtied `splatpipe/`, `cloud/` or
  `config.toml` is reported, and a clean repo reports none (Req 7.3).
- `head_commit` returns HEAD's commit sha, and `archive_head` writes a real `git archive HEAD` tar
  holding HEAD's files (Req 7.4).
- `lock_hashes` reads HEAD's committed lockfile blobs, so its digest for a lockfile changes once a
  change to that file is committed, while the other lockfile's digest stays put (Req 7.5, 7.6).
"""

from __future__ import annotations

import re
import subprocess
import tarfile
from pathlib import Path

import pytest

from cloud.repo import archive_head, head_commit, lock_hashes, uncommitted

from fake_aws import git_repo

# The paths the Start_Command passes to `uncommitted`: the pipeline code, the cloud code and the
# one settings file. `git_repo` commits a file under each, so dirtying one leaves the others clean.
PATHS = ("splatpipe", "cloud", "config.toml")

# Each checked path and a tracked file under it that `git_repo` committed, so editing that file
# dirties exactly that path and nothing else.
DIRTY_CASES = [
    ("splatpipe", "splatpipe/run.py"),
    ("cloud", "cloud/start.py"),
    ("config.toml", "config.toml"),
]


def test_clean_repo_lists_no_uncommitted_paths(tmp_path: Path) -> None:
    """A freshly committed repo has nothing uncommitted under any checked path (Req 7.3)."""
    root = git_repo(tmp_path)
    assert uncommitted(PATHS, cwd=root) == []


@pytest.mark.parametrize("dirty_path, tracked_file", DIRTY_CASES, ids=[c[0] for c in DIRTY_CASES])
def test_a_dirty_path_is_listed(tmp_path: Path, dirty_path: str, tracked_file: str) -> None:
    """Editing a tracked file lists its path, and only its path, as uncommitted (Req 7.3)."""
    root = git_repo(tmp_path)
    (root / tracked_file).write_text("changed\n", encoding="utf-8")
    assert uncommitted(PATHS, cwd=root) == [dirty_path]


def test_head_commit_returns_the_head_sha(tmp_path: Path) -> None:
    """`head_commit` returns HEAD's commit sha, a 40-character lowercase hex string (Req 7.4)."""
    root = git_repo(tmp_path)
    sha = head_commit(cwd=root)
    assert re.fullmatch(r"[0-9a-f]{40}", sha), sha


def test_archive_head_writes_a_tar_of_head(tmp_path: Path) -> None:
    """`archive_head` writes a real tar that holds HEAD's committed files (Req 7.4)."""
    root = git_repo(tmp_path)
    dest = tmp_path / "code.tar"
    archive_head(dest, cwd=root)

    assert dest.stat().st_size > 0
    with tarfile.open(dest) as tar:
        names = set(tar.getnames())
    # The committed files are in the archive, so the machine would run HEAD's code.
    assert {"splatpipe/run.py", "cloud/start.py", "config.toml", "uv.lock",
            "package-lock.json"} <= names


def test_lock_hashes_change_when_a_lockfile_changes(tmp_path: Path) -> None:
    """A committed lockfile change moves that file's digest and leaves the other's (Req 7.5, 7.6)."""
    root = git_repo(tmp_path)
    before = lock_hashes(cwd=root)
    # Two sha-256 hex digests, for HEAD's uv.lock and package-lock.json in that order.
    assert len(before) == 2
    for digest in before:
        assert re.fullmatch(r"[0-9a-f]{64}", digest), digest

    # The digests are over HEAD's committed blobs, so a working-tree edit alone changes nothing.
    (root / "uv.lock").write_text("version = 2\n", encoding="utf-8")
    assert lock_hashes(cwd=root) == before

    # Once the uv.lock change is committed, its digest moves; package-lock.json's is untouched.
    _commit(root, "uv.lock")
    after = lock_hashes(cwd=root)
    assert after[0] != before[0]
    assert after[1] == before[1]


def _commit(root: Path, *files: str) -> None:
    """Stage and commit the named files with a throwaway identity, like `git_repo`'s own commits.

    `lock_hashes` reads HEAD's committed blobs, so a lockfile change only shows once it's committed.
    The `-c` flags keep a fresh identity off the host's git config, skip gpg signing and stop any
    line-ending rewrite, so the commit behaves the same on every machine.
    """
    base = ["git", "-c", "core.autocrlf=false", "-c", "user.name=Voxelis Test",
            "-c", "user.email=test@example.invalid", "-c", "commit.gpgsign=false"]
    for command in ([*base, "add", "--", *files], [*base, "commit", "-m", "Change a lockfile"]):
        result = subprocess.run(command, cwd=root, capture_output=True, text=True,
                                encoding="utf-8", errors="replace")
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip() or f"exit code {result.returncode}"
            raise RuntimeError(f"couldn't commit in the test repo: {detail}")
