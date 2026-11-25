"""Fetch the PlayCanvas engine that the 3D viewer in app/viewer/ runs on.

The engine's build is 2.9 MB, over our 1 MB limit for git, so it never goes in the repo. This
downloads the pinned release from the npm registry, takes its ES module build and its licence out
of the package, checks both against their pinned sha256 and writes them to app/viewer/vendor/,
which git ignores. A file already there with the right sha256 is kept, so running it again
downloads nothing. A wrong sha256 writes nothing and exits with 1.

Engine 2.24.3 reads the bundled .sog that splat-transform 3.8.0 writes (SOG version 2, read
since 2.11.0), with the fixes 2.11.1 to 2.11.7 made for it and the anti-aliasing flag that
2.24.0 reads from each file. The engine is MIT, (c) PlayCanvas Ltd, so its LICENSE is served
beside it.

Standard library only, so it also runs where only python3.11 is installed:

    uv run python scripts/fetch_viewer_engine.py [--into FOLDER]
"""

from __future__ import annotations

import argparse
import hashlib
import io
import ssl
import sys
import tarfile
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

ENGINE_VERSION = "2.24.3"
PACKAGE_URL = f"https://registry.npmjs.org/playcanvas/-/playcanvas-{ENGINE_VERSION}.tgz"

# Each file the viewer needs: its path inside the npm package, the name it's written under and
# its pinned sha256.
FILES = (
    ("package/build/playcanvas.min.mjs", "playcanvas.min.mjs",
     "00138d62a7c7be1b02e1788bfe97000c56374c27a258d6a546f2d1ba195b1015"),
    ("package/LICENSE", "PLAYCANVAS-LICENSE.txt",
     "42fa51ddd556be151a29f381fb97ee975825dcc454e805ea3ea9c079cca04d34"),
)

DEFAULT_FOLDER = REPO_ROOT / "app" / "viewer" / "vendor"
# How long the download may stall before it fails, in seconds.
TIMEOUT_SECONDS = 120
# The system CA bundles of macOS, Debian and Ubuntu, and Amazon Linux (CloudShell).
SYSTEM_CA_BUNDLES = ("/etc/ssl/cert.pem", "/etc/ssl/certs/ca-certificates.crt",
                     "/etc/pki/tls/certs/ca-bundle.crt")


class FetchError(Exception):
    """A download that failed or a file whose sha256 isn't the pinned one."""


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def is_current(folder: Path) -> bool:
    """Whether every file is already in folder with its pinned sha256."""
    return all((folder / name).is_file() and sha256((folder / name).read_bytes()) == digest
               for _, name, digest in FILES)


def tls_context() -> ssl.SSLContext:
    """A verifying TLS context. python.org's macOS Python ships with no CA certificates until its
    Install Certificates step has run, so when the default context holds none, it loads the
    system's own bundle."""
    context = ssl.create_default_context()
    if context.cert_store_stats()["x509_ca"] == 0:
        for bundle in SYSTEM_CA_BUNDLES:
            if Path(bundle).is_file():
                context.load_verify_locations(cafile=bundle)
                break
    return context


def download(url: str) -> bytes:
    try:
        with urllib.request.urlopen(url, timeout=TIMEOUT_SECONDS, context=tls_context()) as answer:
            return answer.read()
    except OSError as error:
        raise FetchError(f"Couldn't download {url}: {error}") from error


def unpack(package: bytes) -> dict[str, bytes]:
    """The pinned files from the package, by the name each is written under, each checked
    against its sha256."""
    wanted = {member: (name, digest) for member, name, digest in FILES}
    found: dict[str, bytes] = {}
    try:
        with tarfile.open(fileobj=io.BytesIO(package), mode="r:gz") as archive:
            for member in archive:
                if member.name in wanted and member.isfile():
                    handle = archive.extractfile(member)
                    if handle is not None:
                        found[member.name] = handle.read()
    except (tarfile.TarError, OSError) as error:
        raise FetchError(f"The package from {PACKAGE_URL} isn't a readable .tgz: {error}") from error
    files: dict[str, bytes] = {}
    for member, (name, digest) in wanted.items():
        if member not in found:
            raise FetchError(f"The package has no {member}.")
        actual = sha256(found[member])
        if actual != digest:
            raise FetchError(f"{member} has sha256 {actual}, not the pinned {digest}. Nothing was written.")
        files[name] = found[member]
    return files


def write(folder: Path, files: dict[str, bytes]) -> None:
    """Write each file whole: a partial one never sits under its real name."""
    folder.mkdir(parents=True, exist_ok=True)
    for name, data in files.items():
        partial = folder / f"{name}.partial"
        partial.write_bytes(data)
        partial.replace(folder / name)


def fetch(folder: Path, url: str | None = None) -> bool:
    """Make sure folder holds the pinned engine files, from url or else the pinned package.
    Returns True when it downloaded them, False when they were already there."""
    if is_current(folder):
        return False
    write(folder, unpack(download(url or PACKAGE_URL)))
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--into", type=Path, default=DEFAULT_FOLDER,
                        help="the folder to write to (default: app/viewer/vendor/)")
    args = parser.parse_args(argv)
    try:
        fetched = fetch(args.into)
    except FetchError as error:
        print(f"fetch_viewer_engine: {error}", file=sys.stderr)
        return 1
    state = "fetched" if fetched else "already there"
    print(f"PlayCanvas engine {ENGINE_VERSION}: {state}, in {args.into}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
