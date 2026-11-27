"""Uploads: a browser that begins an Upload, gets a link per part, sends the parts to the fake
bucket, gets a new link after an expired one, and completes it; and the refusals."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest

from app import uploads
from app.errors import PublicRefusal
from app.sessions import Session
from app.uploads import clean_source_name, parts_needed
from cloud.table import Key
from tests.website_helpers import mp4, website

MIB = 1024 * 1024


def session(name: str = "judges") -> Session:
    return Session(account=name, key=Key(f"account#{name}", "session#x"), csrf="c", uid="u")


@pytest.fixture
def deps(tmp_path: Path):
    deps = website(tmp_path, wc_overrides={"upload_part_bytes": 5 * MIB})
    deps.aws.bucket = "voxelis-kiroween-jobs-fake"
    return deps


def send(deps, upload, data: bytes) -> list[list]:
    parts = []
    size = deps.website.upload_part_bytes
    for n in range(1, upload.parts + 1):
        url = uploads.part_link(deps, session(), upload.id, n)
        status, etag = deps.aws.put_part(url, data[(n - 1) * size:n * size])
        assert status == 200
        parts.append([n, etag])
    return parts


def test_a_three_part_upload_arrives_whole(deps) -> None:
    data = mp4(5) + bytes(11 * MIB)
    upload = uploads.begin(deps, session(), "C:\\DCIM\\IMG_0042.MOV", len(data), deps.clock())
    assert upload.parts == 3 and upload.key == f"uploads/{upload.id}" and upload.name == "IMG_0042.MOV"
    assert "IMG_0042" not in upload.key
    done = uploads.complete(deps, session(), upload.id, send(deps, upload, data))
    assert done.state == "complete" and deps.aws.objects[upload.key] == data
    item = deps.aws.items[("account#judges", f"upload#{upload.id}")]
    assert item["state"] == "complete" and item["expires_at"] - item["created_at"] == 48 * 3600


def test_an_expired_link_gets_a_403_and_a_new_link_works(deps) -> None:
    data = mp4(5)
    upload = uploads.begin(deps, session(), "a.mp4", len(data), deps.clock())
    url = uploads.part_link(deps, session(), upload.id, 1)
    deps.aws.now += timedelta(seconds=deps.website.upload_link_seconds)
    assert deps.aws.put_part(url, data) == (403, None)
    status, etag = deps.aws.put_part(uploads.part_link(deps, session(), upload.id, 1), data)
    assert status == 200
    uploads.complete(deps, session(), upload.id, [[1, etag]])


def test_links_are_only_for_the_accounts_own_open_upload_and_its_parts(deps) -> None:
    upload = uploads.begin(deps, session(), "a.mp4", 100, deps.clock())
    for who, part in (("other", 1), ("judges", 0), ("judges", 2)):
        with pytest.raises(PublicRefusal) as caught:
            uploads.part_link(deps, session(who), upload.id, part)
        assert caught.value.status == 404
    uploads.abort(deps, session(), upload.id)
    with pytest.raises(PublicRefusal) as caught:
        uploads.part_link(deps, session(), upload.id, 1)
    assert caught.value.status == 400
    assert deps.aws.multipart == {}


def test_a_wrong_size_or_bad_part_list_is_refused(deps) -> None:
    data = mp4(5)
    upload = uploads.begin(deps, session(), "a.mp4", len(data) + 1, deps.clock())
    parts = send(deps, upload, data)
    for bad in ([], [[2, parts[0][1]]], [[1, "etag"]], "x"):
        with pytest.raises(PublicRefusal):
            uploads.complete(deps, session(), upload.id, bad)
    with pytest.raises(PublicRefusal, match="didn't arrive complete"):
        uploads.complete(deps, session(), upload.id, parts)
    assert deps.aws.items[("account#judges", f"upload#{upload.id}")]["state"] == "refused"


@pytest.mark.parametrize(("name", "size", "words"), [
    ("clip.avi", 10, "Only MP4 and MOV"), ("a.mp4", 0, "empty"), ("a.mp4", True, "empty"),
    ("a.mp4", 4294967297, "larger"), ("\x00\x01", 10, "no name"), (None, 10, "Pick a video")])
def test_begin_refuses_what_the_job_maker_would(deps, name, size, words) -> None:
    with pytest.raises(PublicRefusal, match=words):
        uploads.begin(deps, session(), name, size, deps.clock())


def test_begin_refuses_after_the_stop_time(deps) -> None:
    with pytest.raises(PublicRefusal) as caught:
        uploads.begin(deps, session(), "a.mp4", 10, deps.website.stop_starting_at)
    assert caught.value.kind == "stopped"


@pytest.mark.parametrize(("raw", "clean"), [
    ("IMG_0042.MOV", "IMG_0042.MOV"), ("/a/b/c.mp4", "c.mp4"), ("a\\b\\c.mov", "c.mov"),
    ("  spaced name.mp4  ", "spaced name.mp4"), ("bad\x00\x1fname\x7f.mp4", "badname.mp4"),
    ("x" * 200 + ".mp4", "x" * 120)])
def test_the_kept_source_name(raw: str, clean: str) -> None:
    assert clean_source_name(raw, 120) == clean
    assert clean_source_name(clean, 120) == clean


def test_parts_needed() -> None:
    assert [parts_needed(n, 10) for n in (1, 10, 11, 20, 21)] == [1, 1, 2, 2, 3]
