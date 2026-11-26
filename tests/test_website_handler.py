"""The request handler end to end over the fake: signing in, the headers, the body limit, CSRF, the
404 and 500 answers, the Result_Page with its viewer, downloads, statuses, a whole upload and start,
and Property 2, that a signed-out request never reaches a page or a file."""

from __future__ import annotations

import base64
import gzip
import logging
import re
import tempfile
from datetime import timedelta
from pathlib import Path
from urllib.parse import urlsplit

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import cloud.start
from app import pages
from app.handler import ROUTES, find_route, handle
from app.sessions import SESSION_COOKIE, SIGNIN_COOKIE
from tests.website_helpers import (JOB_ID, STACK, event, json_body, make_account_item, mp4,
                                   seed_job, sign_in_as, website)

ORIGIN = "https://voxelis-kiroween-jobs-fake.s3.us-east-1.amazonaws.com"
HOST = "abc123.lambda-url.us-east-1.on.aws"


@pytest.fixture
def deps(tmp_path: Path):
    return website(tmp_path)


def call(deps, method: str, path: str, **kwargs) -> dict:
    return handle(event(method, path, **kwargs), deps, deps.clock())


def text(response: dict) -> str:
    body = response["body"]
    return base64.b64decode(body).decode() if response["isBase64Encoded"] else body


def assert_safe_headers(response: dict) -> None:
    headers = response["headers"]
    assert headers["Content-Security-Policy"].startswith("default-src 'none'; script-src 'self'")
    assert headers["X-Content-Type-Options"] == "nosniff"
    assert headers["Referrer-Policy"] == "no-referrer"


def test_signing_in_through_the_handler(deps) -> None:
    make_account_item(deps, "judges", "right-password")
    page = call(deps, "GET", "/signin")
    assert page["statusCode"] == 200 and 'name="viewport"' in text(page)
    cookie = page["cookies"][0]
    assert cookie.startswith(SIGNIN_COOKIE + "=") and "SameSite=Strict" in cookie
    token = cookie.split(";")[0].split("=", 1)[1]
    assert f'value="{token}"' in text(page)
    signed = call(deps, "POST", "/signin", cookies={SIGNIN_COOKIE: token},
                  body=f"csrf={token}&name=judges&password=right-password",
                  headers={"origin": f"https://{HOST}", "sec-fetch-site": "same-origin"})
    assert signed["statusCode"] == 303 and signed["headers"]["Location"] == "/jobs"
    session = next(c for c in signed["cookies"] if c.startswith(SESSION_COOKIE))
    jobs = call(deps, "GET", "/jobs", cookies={SESSION_COOKIE: session.split(";")[0].split("=", 1)[1]})
    assert jobs["statusCode"] == 200 and "Your jobs" in text(jobs)
    assert_safe_headers(jobs)
    assert jobs["headers"]["Cache-Control"] == "no-store"
    assert ORIGIN in jobs["headers"]["Content-Security-Policy"]
    wrong = call(deps, "POST", "/signin", cookies={SIGNIN_COOKIE: token},
                 body=f"csrf={token}&name=judges&password=nope")
    assert wrong["statusCode"] == 400 and "don&#x27;t match" in text(wrong)
    assert wrong["cookies"][0].startswith(SIGNIN_COOKIE)


def test_the_body_limit_applies_only_after_the_session(deps) -> None:
    cookies, csrf = sign_in_as(deps, "judges")
    limit = deps.website.max_body_bytes
    headers = {"x-csrf-token": csrf}
    assert call(deps, "POST", "/api/precheck", cookies=cookies, headers=headers,
                body="x" * (limit + 1))["statusCode"] == 413
    raw = base64.b64encode(b"x" * (limit + 3)).decode()
    big = handle({**event("POST", "/api/precheck", cookies=cookies, headers=headers),
                  "body": raw, "isBase64Encoded": True}, deps, deps.clock())
    assert big["statusCode"] == 413
    assert call(deps, "POST", "/api/precheck", body="x" * (limit * 3))["statusCode"] == 401


def test_csrf_is_checked_on_every_post(deps) -> None:
    cookies, csrf = sign_in_as(deps, "judges")
    before = dict(deps.aws.items)
    for headers in ({}, {"x-csrf-token": "wrong"},
                    {"x-csrf-token": csrf, "origin": "https://evil.example"},
                    {"x-csrf-token": csrf, "sec-fetch-site": "cross-site"}):
        answer = call(deps, "POST", "/api/uploads", cookies=cookies, headers=headers,
                      body='{"name": "a.mp4", "size": 10}')
        assert answer["statusCode"] == 403 and json_body(answer)["error"] == "csrf"
    assert call(deps, "POST", "/signout", cookies=cookies, body="csrf=wrong")["statusCode"] == 403
    assert deps.aws.items == before and deps.aws.multipart == {}
    out = call(deps, "POST", "/signout", cookies=cookies, body=f"csrf={csrf}")
    assert out["statusCode"] == 303 and out["headers"]["Location"] == "/signin"


def test_not_found_is_the_same_for_a_missing_job_another_accounts_and_a_bad_id(deps) -> None:
    cookies, _ = sign_in_as(deps, "judges")
    seed_job(deps, account="other")
    answers = [call(deps, "GET", path, cookies=cookies) for path in
               (f"/jobs/{JOB_ID}", "/jobs/20261201-120000-000000", "/jobs/..%2F..", "/nope",
                f"/jobs/{JOB_ID}/download")]
    assert {a["statusCode"] for a in answers} == {404}
    assert len({text(a) for a in answers}) == 1
    assert not [c for c in deps.aws.calls if c[0] in ("s3_download_link", "s3_list")
                and ".." in c[1]]


def test_an_unforeseen_error_is_a_500_with_no_detail(deps, caplog) -> None:
    cookies, _ = sign_in_as(deps, "judges")
    seed_job(deps)

    def boom(prefix):
        raise RuntimeError("arn:aws:s3:::secret-bucket for 123456789012")

    deps.aws.s3_list = boom
    deps.log.setLevel(logging.INFO)
    with caplog.at_level(logging.INFO, logger=deps.log.name):
        page = call(deps, "GET", "/jobs", cookies=cookies)
        api = call(deps, "GET", f"/api/jobs/{JOB_ID}/status", cookies=cookies)
    assert page["statusCode"] == 500 and "123456789012" not in text(page) and "arn:" not in text(page)
    assert json_body(api) == {"error": "server", "message": "Something went wrong on our side. "
                                                            "Try again in a minute."}
    assert "123456789012" not in caplog.text and "RuntimeError" in caplog.text


def test_the_result_page_holds_the_viewer_and_one_download(deps) -> None:
    cookies, _ = sign_in_as(deps, "judges")
    seed_job(deps)
    page = call(deps, "GET", f"/jobs/{JOB_ID}", cookies=cookies)
    html = text(page)
    assert page["statusCode"] == 200
    assert '<section id="viewer-slot" aria-labelledby="viewer-label">' in html
    splat = re.search(r'data-splat-url="([^"]+)"', html).group(1).replace("&amp;", "&")
    cameras = re.search(r'data-cameras-url="([^"]+)"', html).group(1).replace("&amp;", "&")
    assert urlsplit(splat).path == f"/jobs/{JOB_ID}/out/clean.sog"
    assert urlsplit(cameras).path == f"/jobs/{JOB_ID}/out/cameras.json"
    assert '<script type="module" src="/viewer/embed.js"></script>' in html
    assert '<link rel="stylesheet" href="/viewer/viewer.css">' in html
    policy = page["headers"]["Content-Security-Policy"]
    assert "img-src 'self' blob:" in policy and "worker-src blob:" in policy and ORIGIN in policy
    assert html.count('href="/jobs/' + JOB_ID + '/download"') == 1
    assert "data-job-id" not in html            # a finished Job's page doesn't poll
    download = call(deps, "GET", f"/jobs/{JOB_ID}/download", cookies=cookies)
    assert download["statusCode"] == 303
    status, data = deps.aws.follow_link(download["headers"]["Location"])
    assert (status, data) == (200, b"splat bytes clean.sog")
    link = deps.aws.links[download["headers"]["Location"]]
    assert link.filename == f"{JOB_ID}-clean.sog"
    deps.aws.now += timedelta(seconds=deps.website.download_link_seconds)
    assert deps.aws.follow_link(download["headers"]["Location"])[0] == 403


def test_a_running_jobs_page_polls_and_has_no_viewer(deps) -> None:
    cookies, _ = sign_in_as(deps, "judges")
    seed_job(deps, status=None, files=(), reports=("frames",))
    html = text(call(deps, "GET", f"/jobs/{JOB_ID}", cookies=cookies))
    assert f'data-job-id="{JOB_ID}"' in html and 'data-poll-seconds="20"' in html
    assert "data-splat-url" not in html
    status = call(deps, "GET", f"/api/jobs/{JOB_ID}/status", cookies=cookies)
    assert json_body(status) == {"status": "running", "text": "Running (step 2 of 5)", "final": False}
    assert call(deps, "GET", f"/jobs/{JOB_ID}/download", cookies=cookies)["statusCode"] == 404


def test_the_viewer_files_are_served_as_modules_to_a_signed_in_account(deps) -> None:
    cookies, _ = sign_in_as(deps, "judges")
    for name in ("embed.js", "tour.js", "loading.js", "haunt.js"):
        answer = call(deps, "GET", f"/viewer/{name}", cookies=cookies)
        assert answer["statusCode"] == 200
        assert answer["headers"]["Content-Type"] == "text/javascript; charset=utf-8"
        assert answer["headers"]["Cache-Control"] == "private, max-age=3600"
    engine = call(deps, "GET", "/viewer/vendor/playcanvas.min.mjs", cookies=cookies,
                  headers={"accept-encoding": "gzip, br"})
    assert engine["headers"]["Content-Encoding"] == "gzip" and engine["isBase64Encoded"]
    assert gzip.decompress(base64.b64decode(engine["body"])) == deps.static[
        "/viewer/vendor/playcanvas.min.mjs"]
    assert call(deps, "GET", "/viewer/index.html", cookies=cookies)["statusCode"] == 404
    assert call(deps, "GET", "/viewer/embed.js")["statusCode"] == 303
    assert call(deps, "GET", "/static/style.css")["statusCode"] == 200


def test_a_whole_upload_and_start_through_the_api(deps, monkeypatch, caplog) -> None:
    monkeypatch.setattr(cloud.start, "check_spend", lambda **kwargs: None)
    make_account_item(deps, "judges", "right-password")
    cookies, csrf = sign_in_as(deps, "judges")
    headers = {"x-csrf-token": csrf}
    deps.log.setLevel(logging.INFO)
    data = mp4(15)
    with caplog.at_level(logging.INFO, logger=deps.log.name):
        assert json_body(call(deps, "POST", "/api/precheck", cookies=cookies, headers=headers,
                              body='{"preset": "fast"}')) == {"ok": True}
        begun = json_body(call(deps, "POST", "/api/uploads", cookies=cookies, headers=headers,
                               body=f'{{"name": "IMG_1.MP4", "size": {len(data)}}}'))
        assert begun["parts"] == 1 and begun["part_bytes"] == 16777216
        link = json_body(call(deps, "POST", f"/api/uploads/{begun['upload']}/parts/1",
                              cookies=cookies, headers=headers, body="{}"))["url"]
        status, etag = deps.aws.put_part(link, data)
        body = '{"parts": [[1, ' + '"\\"' + etag.strip('"') + '\\""' + "]]}"
        assert json_body(call(deps, "POST", f"/api/uploads/{begun['upload']}/complete",
                              cookies=cookies, headers=headers, body=body)) == {"ok": True}
        started = json_body(call(deps, "POST", "/api/jobs", cookies=cookies, headers=headers,
                                 body=f'{{"upload": "{begun["upload"]}", "preset": "fast"}}'))
    assert started["outcome"] == "launched"
    assert f"jobs/{started['job']}/input/video.mp4" in deps.aws.objects
    listed = text(call(deps, "GET", "/jobs", cookies=cookies))
    assert started["job"] in listed and "Starting" in listed
    for secret in (csrf, cookies[SESSION_COOKIE].split(".")[1], link, "right-password"):
        assert secret not in caplog.text
    assert "POST api_start 200" in caplog.text


def test_the_start_page_after_the_stop_time(deps) -> None:
    deps.clock.now = deps.website.stop_starting_at
    cookies, _ = sign_in_as(deps, "judges")
    html = text(handle(event("GET", "/start", cookies=cookies), deps, deps.clock()))
    assert "stopped starting jobs at 08:00 UTC on 4 February 2027" in html
    assert 'type="file"' not in html
    assert text(handle(event("GET", "/jobs", cookies=cookies), deps, deps.clock()))


# Feature: website, Property 2: A signed-out request never reaches a page or a file

UPLOAD = "0123456789abcdef0123456789abcdef"
KNOWN_PATHS = ["/", "/jobs", "/start", f"/jobs/{JOB_ID}", f"/jobs/{JOB_ID}/download",
               f"/jobs/{JOB_ID}/share", f"/jobs/{JOB_ID}/share/off", "/showcase", "/static/app.js",
               "/viewer/embed.js", "/viewer/vendor/playcanvas.min.mjs", "/signout",
               "/api/precheck", "/api/uploads", f"/api/uploads/{UPLOAD}/parts/1",
               f"/api/uploads/{UPLOAD}/complete", f"/api/uploads/{UPLOAD}/abort", "/api/jobs",
               f"/api/jobs/{JOB_ID}/status", "/api/nothing", "/shared/", "/shared/x/viewer/embed.js"]
OPEN = {("GET", "/signin"), ("POST", "/signin"), ("GET", "/static/style.css"),
        ("GET", "/favicon.ico")}
COOKIE_KINDS = ("none", "malformed", "unknown", "ended", "expired", "stale", "removed")
SHARE_KINDS = ("random", "revoked", "malformed")


def signed_out_world():
    """A website with a finished, shared Job and Sessions that are no longer valid."""
    tmp = Path(tempfile.mkdtemp(prefix="voxelis-p2-"))
    deps = website(tmp)
    seed_job(deps)
    cookies = {}
    for kind in ("ended", "expired", "stale", "removed"):
        cookie, _ = sign_in_as(deps, "judges" if kind != "removed" else "gone")
        cookies[kind] = cookie[SESSION_COOKIE]
    name, session_id = cookies["ended"].split(".")
    from app.sessions import session_key
    key = session_key(name, session_id)
    deps.aws.items[(key.pk, key.sk)]["ended"] = True
    key = session_key(*cookies["expired"].split("."))
    deps.aws.items[(key.pk, key.sk)]["expires_at"] = int(deps.clock().timestamp())
    key = session_key(*cookies["stale"].split("."))
    deps.aws.items[(key.pk, key.sk)]["uid"] = "x" * 32
    del deps.aws.items[("account#gone", "profile")]
    cookies["unknown"] = "nobody." + "A" * 43
    cookies["malformed"] = "judges.short"
    from app import shares
    from app.sessions import new_token
    revoked = new_token(deps)
    digest = shares.token_hash(revoked)
    deps.aws.items[("share", f"token#{digest}")] = {"pk": "share", "sk": f"token#{digest}",
                                                    "job_id": JOB_ID, "account": "judges",
                                                    "state": "off"}
    deps.aws.items[("account#judges", f"job#{JOB_ID}")]["share_hash"] = "0" * 64
    deps.aws.calls.clear()
    return deps, cookies, revoked


@st.composite
def signed_out_request(draw):
    method = draw(st.sampled_from(["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"]))
    share = draw(st.sampled_from(SHARE_KINDS))
    path = draw(st.one_of(st.sampled_from(KNOWN_PATHS),
                          st.just("/shared/<token>"), st.just("/shared/<token>/viewer/embed.js"),
                          st.text(min_size=0, max_size=30).map(lambda t: "/" + t)))
    cookie = draw(st.sampled_from(COOKIE_KINDS))
    size = draw(st.sampled_from([0, 10, 70_000, 200_000]))
    headers = draw(st.dictionaries(st.sampled_from(["origin", "sec-fetch-site", "x-csrf-token",
                                                    "accept-encoding", "x-forwarded-for"]),
                                   st.text(max_size=20), max_size=3))
    return method, path, share, cookie, size, headers, draw(st.booleans())


@settings(max_examples=150, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(signed_out_request())
def test_property_2_a_signed_out_request_gets_nothing(request_parts) -> None:
    method, path, share, cookie_kind, size, headers, as_base64 = request_parts
    deps, cookies, revoked = signed_out_world()
    token = {"random": "B" * 43, "revoked": revoked, "malformed": "short"}[share]
    path = path.replace("<token>", token)
    if (method, path) in OPEN:
        return
    body = "x" * size if size else None
    sent = event(method, path, cookies={SESSION_COOKIE: cookies[cookie_kind]}
                 if cookie_kind != "none" else None, headers=headers, body=body,
                 base64_body=as_base64 and body is not None)
    answer = handle(sent, deps, deps.clock())
    assert_safe_headers(answer)
    assert "amazonaws" not in answer["headers"]["Content-Security-Policy"]
    ops = [op for op, _ in deps.aws.calls]
    assert not {"s3_download_link", "s3_part_link", "stack_outputs", "s3_list", "s3_get_text",
                "run_instances", "table_put", "table_update", "table_transact"} & set(ops)
    route, _ = find_route(method, path)
    if route is not None and route.kind == "shared":
        assert answer["statusCode"] == 404 and text(answer) == pages.not_found_page()
        assert set(ops) <= {"table_get"} and len(ops) <= 2
        return
    if path.startswith("/api/"):
        assert answer["statusCode"] == 401 and json_body(answer) == {"error": "signed_out"}
    else:
        assert answer["statusCode"] == 303 and answer["headers"]["Location"] == "/signin"
    assert ops in ([], ["table_get_many"])
    if cookie_kind in ("none", "malformed"):
        assert ops == []


def test_every_route_is_open_shared_page_or_script() -> None:
    kinds = {route.kind for route in ROUTES}
    assert kinds == {"open", "shared", "page", "script"}
    assert {(r.method, r.pattern.pattern.replace("\\", "")) for r in ROUTES if r.kind == "open"} == OPEN
    assert all(r.pattern.pattern.startswith("/api/") for r in ROUTES if r.kind == "script")
    assert STACK["BucketName"] in ORIGIN


def test_the_lambda_entry_builds_its_deps_from_the_bundle(tmp_path: Path) -> None:
    import shutil

    from app.baked import write_baked
    from app.lambda_function import build_deps
    from tests.website_helpers import PACKAGE_HASH, REAL_CONFIG, UV_HASH

    shutil.copy(REAL_CONFIG, tmp_path / "config.toml")
    write_baked(tmp_path, "a" * 40, UV_HASH, PACKAGE_HASH)
    (tmp_path / "baked" / "code.tar").write_bytes(b"tar")
    shutil.copytree(REAL_CONFIG.parent / "app" / "static", tmp_path / "app" / "static")
    deps = build_deps({"WEBSITE_TABLE": "table-x"}, tmp_path)
    assert deps.aws.table == "table-x" and deps.aws.region == "us-east-1"
    assert deps.baked.commit == "a" * 40 and deps.baked.code().uv_lock_hash == UV_HASH
    assert set(deps.static) == {"/static/style.css", "/favicon.ico", "/static/app.js"}
    page = handle(event("GET", "/signin"), deps, deps.clock())
    assert page["statusCode"] == 200
