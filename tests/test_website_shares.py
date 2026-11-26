"""Share links: the owner makes a secret link for one finished Job, anyone with it sees that Job's
result page and viewer without signing in, read-only, and a wrong or turned-off link shows nothing."""

from __future__ import annotations

import base64
import re
from pathlib import Path

import pytest

from app import pages
from app.handler import handle
from app.sessions import SESSION_COOKIE
from tests.website_helpers import JOB_ID, event, seed_job, sign_in_as, website

HOST = "abc123.lambda-url.us-east-1.on.aws"


@pytest.fixture
def deps(tmp_path: Path):
    deps = website(tmp_path)
    seed_job(deps)
    return deps


def call(deps, method: str, path: str, **kwargs) -> dict:
    return handle(event(method, path, **kwargs), deps, deps.clock())


def body(answer: dict) -> str:
    raw = answer["body"]
    return base64.b64decode(raw).decode() if answer["isBase64Encoded"] else raw


def make_link(deps, cookies, csrf) -> str:
    answer = call(deps, "POST", f"/jobs/{JOB_ID}/share", cookies=cookies, body=f"csrf={csrf}")
    assert answer["statusCode"] == 200
    match = re.search(r'id="share-link" readonly value="https://[^/]+/shared/([A-Za-z0-9_-]{43})"',
                      body(answer))
    assert match, body(answer)
    return match.group(1)


def assert_nothing(deps, answer: dict) -> None:
    assert answer["statusCode"] == 404 and body(answer) == pages.not_found_page()
    assert JOB_ID not in body(answer) and "data-splat-url" not in body(answer)
    assert "amazonaws" not in answer["headers"]["Content-Security-Policy"]


def test_the_owner_makes_a_link_that_shows_the_result_and_viewer_without_signing_in(deps) -> None:
    cookies, csrf = sign_in_as(deps, "judges")
    token = make_link(deps, cookies, csrf)
    stored = str(deps.aws.items)
    assert token not in stored                     # only its hash is kept
    deps.aws.calls.clear()
    shared = call(deps, "GET", f"/shared/{token}")
    html = body(shared)
    assert shared["statusCode"] == 200 and JOB_ID in html
    assert 'data-splat-url="https://voxelis-kiroween-jobs-fake.s3.us-east-1.amazonaws.com/' in html
    assert f'<script type="module" src="/shared/{token}/viewer/embed.js"></script>' in html
    # Read-only: no sign-out, no CSRF token, no Page_Script, no download, no share controls.
    for absent in ("/signout", "csrf-token", "/static/app.js", "/download", 'id="share"'):
        assert absent not in html
    assert shared["headers"]["Cache-Control"] == "no-store"
    assert shared["headers"]["X-Robots-Tag"] == "noindex, nofollow"
    assert "worker-src blob:" in shared["headers"]["Content-Security-Policy"]
    links = [key for op, key in deps.aws.calls if op == "s3_download_link"]
    assert links == [f"jobs/{JOB_ID}/out/clean.sog", f"jobs/{JOB_ID}/out/cameras.json"]
    engine = call(deps, "GET", f"/shared/{token}/viewer/embed.js")
    assert engine["statusCode"] == 200
    assert engine["headers"]["Content-Type"] == "text/javascript; charset=utf-8"
    # The owner's page now says the link is on.
    owner = body(call(deps, "GET", f"/jobs/{JOB_ID}", cookies=cookies))
    assert "A share link is on" in owner and token not in owner


def test_a_wrong_or_malformed_link_shows_nothing(deps) -> None:
    cookies, csrf = sign_in_as(deps, "judges")
    token = make_link(deps, cookies, csrf)
    wrong = token[:-1] + ("A" if token[-1] != "A" else "B")
    for bad in (wrong, "short", token + "x", "%2e%2e"):
        deps.aws.calls.clear()
        assert_nothing(deps, call(deps, "GET", f"/shared/{bad}"))
        assert_nothing(deps, call(deps, "GET", f"/shared/{bad}/viewer/embed.js"))
        assert {op for op, _ in deps.aws.calls} <= {"table_get"}


def test_a_turned_off_link_shows_nothing(deps) -> None:
    cookies, csrf = sign_in_as(deps, "judges")
    token = make_link(deps, cookies, csrf)
    off = call(deps, "POST", f"/jobs/{JOB_ID}/share/off", cookies=cookies, body=f"csrf={csrf}")
    assert off["statusCode"] == 303 and off["headers"]["Location"] == f"/jobs/{JOB_ID}"
    assert_nothing(deps, call(deps, "GET", f"/shared/{token}"))
    assert_nothing(deps, call(deps, "GET", f"/shared/{token}/viewer/embed.js"))
    assert "Make a share link" in body(call(deps, "GET", f"/jobs/{JOB_ID}", cookies=cookies))


def test_a_new_link_turns_the_old_one_off(deps) -> None:
    cookies, csrf = sign_in_as(deps, "judges")
    first = make_link(deps, cookies, csrf)
    second = make_link(deps, cookies, csrf)
    assert_nothing(deps, call(deps, "GET", f"/shared/{first}"))
    assert call(deps, "GET", f"/shared/{second}")["statusCode"] == 200


def test_only_the_owner_of_a_finished_job_makes_a_link(deps) -> None:
    other, other_csrf = sign_in_as(deps, "other")
    deps.aws.items[("showcase", f"job#{JOB_ID}")] = {"pk": "showcase", "sk": f"job#{JOB_ID}",
                                                     "title": "Mill", "licence": "own"}
    answer = call(deps, "POST", f"/jobs/{JOB_ID}/share", cookies=other, body=f"csrf={other_csrf}")
    assert answer["statusCode"] == 404
    assert 'id="share"' not in body(call(deps, "GET", f"/jobs/{JOB_ID}", cookies=other))
    cookies, csrf = sign_in_as(deps, "judges")
    seed_job(deps, "20261202-120000-abcdef", status=None, files=(), reports=("frames",))
    unfinished = call(deps, "POST", "/jobs/20261202-120000-abcdef/share", cookies=cookies,
                      body=f"csrf={csrf}")
    assert unfinished["statusCode"] == 400 and "finished job" in body(unfinished)
    assert call(deps, "POST", f"/jobs/{JOB_ID}/share", cookies=cookies,
                body="csrf=wrong")["statusCode"] == 403
    assert not [k for k in deps.aws.items if k[0] == "share"]


def test_a_link_stops_working_when_the_owners_account_is_removed(deps) -> None:
    from app.accounts import remove_account

    cookies, csrf = sign_in_as(deps, "judges")
    token = make_link(deps, cookies, csrf)
    remove_account(deps.aws, "judges")
    assert_nothing(deps, call(deps, "GET", f"/shared/{token}"))
    assert SESSION_COOKIE in cookies
