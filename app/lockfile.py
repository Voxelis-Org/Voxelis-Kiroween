"""Turn `app/uv.lock` into a hashed requirements file, with the standard library only, so it runs in
CloudShell before boto3 is installed: `python3.11 -m app.lockfile app/uv.lock`.

One `name==version` line per registry package, each with a `--hash=sha256:` for every wheel, so pip
installs exactly the locked wheels with `--require-hashes`. The App_Project itself is skipped.
"""

from __future__ import annotations

import sys
import tomllib
from collections.abc import Sequence
from pathlib import Path

from cloud.errors import CloudError


def export_requirements(lock_text: str) -> str:
    try:
        data = tomllib.loads(lock_text)
    except tomllib.TOMLDecodeError as error:
        raise CloudError(f"app/uv.lock isn't valid TOML ({error}). Get it back from git.") from error
    lines: dict[str, str] = {}
    for package in data.get("package", []):
        name, version = str(package.get("name", "")), str(package.get("version", ""))
        source = package.get("source", {})
        if "virtual" in source or "editable" in source:
            continue
        if "registry" not in source:
            raise CloudError(f"app/uv.lock takes {name} from somewhere other than a registry. Lock "
                             "it from PyPI.")
        if name in lines:
            raise CloudError(f"app/uv.lock holds {name} twice. Lock it again for Python 3.11 only.")
        hashes = [str(wheel.get("hash", "")) for wheel in package.get("wheels", [])]
        if not hashes or not all(h.startswith("sha256:") for h in hashes):
            raise CloudError(f"app/uv.lock has no hashed wheel for {name}, and the install takes "
                             "wheels only.")
        options = "".join(f" \\\n    --hash={h}" for h in hashes)
        lines[name] = f"{name}=={version}{options}"
    return "".join(lines[name] + "\n" for name in sorted(lines))


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        print("usage: python3.11 -m app.lockfile app/uv.lock", file=sys.stderr)
        return 2
    try:
        sys.stdout.write(export_requirements(Path(args[0]).read_text(encoding="utf-8")))
    except (OSError, CloudError) as error:
        print(str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
