"""Git helpers for the Start_Command: the uncommitted-change check and the Code_Archive.

Four helpers shell out to `git`, a dev tool. `uncommitted` reports which of the given paths
(splatpipe/, cloud/ and config.toml) have uncommitted changes, so every Job runs committed code
(Req 7.3). `head_commit` gives HEAD's commit sha (Req 7.4). `archive_head` writes a `git archive
HEAD` tar, the Code_Archive the machine runs instead of the image carrying our code (Req 7.4).
`lock_hashes` gives the sha-256 of HEAD's uv.lock and package-lock.json, which the Start_Command
compares with the Machine_Image's recorded tags to refuse a drift before a paid boot (Req 7.5, 7.6).

A missing git, or a git that exits non-zero, is a CloudError whose message a user can read, the way
splatpipe.probe refuses a missing ffprobe. From splatpipe this module imports nothing, and it
imports no boto3.
"""

from __future__ import annotations

import hashlib
import subprocess
from collections.abc import Sequence
from pathlib import Path

from cloud.errors import CloudError

# HEAD's path to each lockfile, in the order lock_hashes returns their hashes.
_LOCK_FILES = ("uv.lock", "package-lock.json")


def uncommitted(paths: Sequence[str], *, cwd: Path) -> list[str]:
    """The given paths that have uncommitted changes, in the order given (Req 7.3).

    A path counts as uncommitted when `git status` reports anything under it: a staged, unstaged or
    untracked change. The Start_Command passes splatpipe/, cloud/ and config.toml.
    """
    return [path for path in paths
            if _git("status", "--porcelain", "--", path, cwd=cwd).strip()]


def head_commit(*, cwd: Path) -> str:
    """HEAD's commit sha (Req 7.4)."""
    return _git("rev-parse", "HEAD", cwd=cwd).decode("utf-8").strip()


def archive_head(dest: Path, *, cwd: Path) -> None:
    """Write a `git archive HEAD` tar to `dest`, so the machine runs HEAD's code (Req 7.4)."""
    _git("archive", "--format=tar", "--output", str(dest), "HEAD", cwd=cwd)


def head_file(name: str, *, cwd: Path) -> bytes:
    """HEAD's committed bytes of one file (`git show HEAD:<name>`), not the working tree's.

    The Image_Build uploads its dependency manifests this way, so the uv.lock the builder installs
    from is byte for byte the one whose hash `lock_hashes` tags on the image.
    """
    return _git("show", f"HEAD:{name}", cwd=cwd)


def lock_hashes(*, cwd: Path) -> tuple[str, str]:
    """The sha-256 of HEAD's uv.lock and package-lock.json (Req 7.5, 7.6).

    The hashes are over the committed blobs (`git show HEAD:<file>`), not the working tree, so they
    match the code the Code_Archive carries and the tags the Image_Build recorded.
    """
    uv_lock, package_lock = (
        hashlib.sha256(_git("show", f"HEAD:{name}", cwd=cwd)).hexdigest()
        for name in _LOCK_FILES)
    return uv_lock, package_lock


def _git(*args: str, cwd: Path) -> bytes:
    """Run `git <args>` in `cwd` and return its stdout as bytes, or raise a CloudError.

    A missing git is a Readable_Error that says to install it; a non-zero exit carries git's first
    stderr line. stdout is captured as bytes, so `archive_head`'s tar and `lock_hashes`' blobs stay
    byte-exact.
    """
    try:
        result = subprocess.run(["git", *args], cwd=cwd, capture_output=True)
    except FileNotFoundError as error:
        raise CloudError("git isn't installed, so the repo can't be checked or archived. Install "
                         "git and make sure `git` runs in a terminal.") from error
    if result.returncode != 0:
        detail = _first_line(result.stderr) or f"exit code {result.returncode}"
        raise CloudError(f"git {args[0]} failed ({detail}). Run the command from inside the repo, "
                         "with HEAD committed, then try again.")
    return result.stdout


def _first_line(stderr: bytes) -> str:
    """git's first stderr line, with bytes that aren't UTF-8 replaced."""
    return stderr.decode("utf-8", errors="replace").strip().partition("\n")[0].strip()
