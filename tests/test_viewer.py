"""The 3D viewer in app/viewer/ and its engine fetch step, scripts/fetch_viewer_engine.py.

- The viewer's own tests are JavaScript, in tests/viewer/*.test.mjs: its URL parameter, its
  reading of out/cameras.json and its starting camera. This file runs them with Node's built-in
  test runner, so they need no npm package, and skips them when Node isn't installed.
- The fetch step is tested with a small fake package and a file:// URL, so no test downloads
  anything: the pinned files are written once their sha256 matches, a file already there is kept,
  and a wrong sha256 or a missing file writes nothing.
- The repo side: git ignores the engine's folder, and the viewer imports the file the fetch step
  writes.
"""

from __future__ import annotations

import hashlib
import importlib.util
import io
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
VIEWER = REPO_ROOT / "app" / "viewer"
JS_TESTS = sorted((Path(__file__).parent / "viewer").glob("*.test.mjs"))


def load_fetch() -> ModuleType:
    path = REPO_ROOT / "scripts" / "fetch_viewer_engine.py"
    spec = importlib.util.spec_from_file_location("fetch_viewer_engine", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


fetch_engine = load_fetch()


@pytest.mark.skipif(shutil.which("node") is None, reason="Node is not installed")
def test_the_viewers_javascript_tests_pass() -> None:
    assert JS_TESTS, "no tests/viewer/*.test.mjs found"
    result = subprocess.run(["node", "--test", *map(str, JS_TESTS)], cwd=REPO_ROOT,
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr


def fake_package(tmp_path: Path, contents: dict[str, bytes]) -> str:
    """A .tgz holding the given members, as a file:// URL."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name, data in contents.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    path = tmp_path / "package.tgz"
    path.write_bytes(buffer.getvalue())
    return path.as_uri()


@pytest.fixture
def pinned(monkeypatch: pytest.MonkeyPatch) -> dict[str, bytes]:
    """Pins the fetch step to two tiny files, by member name."""
    files = {"package/build/engine.mjs": b"export const version = '1';\n",
             "package/LICENSE": b"MIT\n"}
    monkeypatch.setattr(fetch_engine, "FILES", (
        ("package/build/engine.mjs", "engine.mjs", hashlib.sha256(files["package/build/engine.mjs"]).hexdigest()),
        ("package/LICENSE", "LICENSE.txt", hashlib.sha256(files["package/LICENSE"]).hexdigest()),
    ))
    return files


def test_the_fetch_writes_the_pinned_files_and_then_keeps_them(tmp_path: Path, pinned: dict[str, bytes]) -> None:
    url = fake_package(tmp_path, {**pinned, "package/README.md": b"not wanted"})
    folder = tmp_path / "vendor"
    assert fetch_engine.fetch(folder, url) is True
    assert (folder / "engine.mjs").read_bytes() == pinned["package/build/engine.mjs"]
    assert (folder / "LICENSE.txt").read_bytes() == pinned["package/LICENSE"]
    assert sorted(path.name for path in folder.iterdir()) == ["LICENSE.txt", "engine.mjs"]
    # A second run finds both files current and downloads nothing, even from a broken URL.
    assert fetch_engine.fetch(folder, (tmp_path / "missing.tgz").as_uri()) is False


def test_a_wrong_sha256_writes_nothing(tmp_path: Path, pinned: dict[str, bytes]) -> None:
    url = fake_package(tmp_path, {**pinned, "package/build/engine.mjs": b"export const evil = 1;\n"})
    folder = tmp_path / "vendor"
    with pytest.raises(fetch_engine.FetchError, match="not the pinned"):
        fetch_engine.fetch(folder, url)
    assert not folder.exists()


def test_a_package_missing_a_file_writes_nothing(tmp_path: Path, pinned: dict[str, bytes]) -> None:
    url = fake_package(tmp_path, {"package/LICENSE": pinned["package/LICENSE"]})
    with pytest.raises(fetch_engine.FetchError, match="has no package/build/engine.mjs"):
        fetch_engine.fetch(tmp_path / "vendor", url)
    assert not (tmp_path / "vendor").exists()


def test_a_failed_download_exits_1_with_a_readable_message(tmp_path: Path, pinned: dict[str, bytes],
                                                           monkeypatch: pytest.MonkeyPatch,
                                                           capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(fetch_engine, "PACKAGE_URL", (tmp_path / "missing.tgz").as_uri())
    assert fetch_engine.main(["--into", str(tmp_path / "vendor")]) == 1
    assert "Couldn't download" in capsys.readouterr().err


def test_the_engine_pin_is_a_release_that_reads_splat_transform_3_8_0s_sog() -> None:
    # 2.24.0 is the first engine release that reads the antialiased flag from each .sog, after
    # 2.11.0 to 2.11.7 added and fixed SOG version 2.
    major, minor, patch = map(int, fetch_engine.ENGINE_VERSION.split("."))
    assert (major, minor, patch) >= (2, 24, 0)
    assert fetch_engine.PACKAGE_URL.endswith(f"/playcanvas-{fetch_engine.ENGINE_VERSION}.tgz")
    assert [name for _, name, _ in fetch_engine.FILES] == ["playcanvas.min.mjs", "PLAYCANVAS-LICENSE.txt"]
    for _, _, digest in fetch_engine.FILES:
        assert len(digest) == 64 and all(c in "0123456789abcdef" for c in digest)


def test_git_ignores_the_engine_folder_and_the_viewer_imports_its_file() -> None:
    ignored = subprocess.run(["git", "check-ignore", "-q", "app/viewer/vendor/playcanvas.min.mjs"],
                             cwd=REPO_ROOT)
    assert ignored.returncode == 0
    assert fetch_engine.DEFAULT_FOLDER == VIEWER / "vendor"
    assert "from './vendor/playcanvas.min.mjs'" in (VIEWER / "viewer.js").read_text(encoding="utf-8")

def test_the_website_serves_every_file_the_viewer_loads() -> None:
    # Every module a viewer module imports, and every stylesheet one links from beside itself,
    # must be in VIEWER_FILES, or the website and share links answer 404 for it.
    from app.handler import VIEWER_FILES
    import re
    wanted: set[str] = set()
    for module in VIEWER.glob("*.js"):
        if module.name in ("page.js",):
            continue
        code = module.read_text(encoding="utf-8")
        wanted.update(re.findall(r"""from\s+'\./([^']+)'""", code))
        wanted.update(re.findall(r"""new URL\('\./([^']+)', import\.meta\.url\)""", code))
    wanted.update(["embed.js", "viewer.css"])
    assert "smash.js" in wanted and "smash.css" in wanted
    assert sorted(wanted - set(VIEWER_FILES)) == []
    for name in VIEWER_FILES:
        assert name.startswith("vendor/") or (VIEWER / name).is_file(), name
