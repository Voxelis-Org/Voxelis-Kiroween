"""The Baked_Commit: the commit the website bundle was built from, with its Code_Archive.

The Deploy_Command writes `baked/commit.json` and `baked/code.tar` into the bundle. Every Website_Job
runs that commit, and its two lock-file hashes meet the Machine_Image's tags, so after a new image
with a different `uv.lock` every website start refuses until a redeploy.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from cloud.errors import CloudError
from cloud.start import Code

COMMIT = re.compile(r"[0-9a-f]{40}")
LOCK_HASH = re.compile(r"[0-9a-f]{64}")
FOLDER = "baked"


@dataclass(frozen=True)
class BakedCommit:
    commit: str
    uv_lock_hash: str
    package_lock_hash: str
    archive: Path                  # <bundle>/baked/code.tar

    def code(self) -> Code:
        return Code(commit=self.commit, archive=self.archive, uv_lock_hash=self.uv_lock_hash,
                    package_lock_hash=self.package_lock_hash)


def read_baked(root: Path) -> BakedCommit:
    folder = Path(root) / FOLDER
    try:
        data = json.loads((folder / "commit.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise CloudError(f"The website bundle has no readable baked/commit.json ({error}). Deploy "
                         "it again with the deploy command.") from error
    commit, uv, package = (data.get(key) for key in ("commit", "uv_lock_hash", "package_lock_hash"))
    if not (isinstance(commit, str) and COMMIT.fullmatch(commit)
            and all(isinstance(h, str) and LOCK_HASH.fullmatch(h) for h in (uv, package))):
        raise CloudError("The website bundle's baked/commit.json is malformed. Deploy it again.")
    archive = folder / "code.tar"
    if not archive.is_file():
        raise CloudError("The website bundle has no baked/code.tar. Deploy it again.")
    return BakedCommit(commit=commit, uv_lock_hash=uv, package_lock_hash=package, archive=archive)


def write_baked(folder: Path, commit: str, uv_lock_hash: str, package_lock_hash: str) -> None:
    """Write `<folder>/baked/commit.json`. The caller puts code.tar beside it."""
    target = Path(folder) / FOLDER
    target.mkdir(parents=True, exist_ok=True)
    text = json.dumps({"commit": commit, "uv_lock_hash": uv_lock_hash,
                       "package_lock_hash": package_lock_hash}, sort_keys=True)
    (target / "commit.json").write_text(text + "\n", encoding="utf-8")
