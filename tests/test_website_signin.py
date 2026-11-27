"""Signing in and out: the form token, the password, the per-address limit, the Session item and
its cookie, and the one-read Session check."""

from __future__ import annotations

from datetime import timedelta

import pytest

from app import accounts, sessions
from app.errors import PublicRefusal
from app.sessions import SESSION_COOKIE, SIGNIN_COOKIE, current, sign_in, sign_out
from app.web import Request
from tests.fake_aws import FakeAws
from tests.website_helpers import T0, make_account_item, make_deps, sign_in_as

TOKEN = "t" * 43


def signin_request(name: str, password: str, *, ip: str = "203.0.113.7",
                   token: str = TOKEN, cookie: str | None = TOKEN,
                   headers: dict[str, str] | None = None) -> Request:
    body = f"csrf={token}&name={name}&password={password}".encode()
    cookies = {SIGNIN_COOKIE: cookie} if cookie is not None else {}
    return Request("POST", "/signin", {}, headers or {}, cookies, body, ip, "h.lambda-url.on.aws")


def page_request(cookies: dict[str, str]) -> Request:
    return Request("GET", "/jobs", {}, {}, cookies, b"", "203.0.113.7", "h")


@pytest.fixture
def deps():
    deps = make_deps(FakeAws())
    make_account_item(deps, "judges", "right-password")
    return deps


def test_the_right_password_makes_a_session_and_its_cookie(deps) -> None:
    response = sign_in(signin_request("judges", "right-password"), deps, T0)
    assert response.status == 303 and response.headers["Location"] == "/jobs"
    cookie = next(c for c in response.cookies if c.startswith(SESSION_COOKIE + "="))
    assert cookie.endswith("; Path=/; Secure; HttpOnly; SameSite=Lax; Max-Age=604800")
    assert "Domain" not in cookie
    assert any(c.startswith(SIGNIN_COOKIE + "=;") and "Max-Age=0" in c for c in response.cookies)
    value = cookie.split(";")[0].split("=", 1)[1]
    name, session_id = value.split(".")
    assert name == "judges" and len(session_id) == 43
    key = sessions.session_key(name, session_id)
    item = deps.aws.items[(key.pk, key.sk)]
    assert session_id not in str(item)
    assert item["expires_at"] == int((T0 + timedelta(days=7)).timestamp())
    assert item["uid"] == "u" * 32 and len(item["csrf"]) == 43
    assert current(page_request({SESSION_COOKIE: value}), deps, T0).account == "judges"


@pytest.mark.parametrize(("name", "password"), [("judges", "wrong"), ("nobody", "right-password"),
                                                ("Bad Name!", "x")])
def test_a_wrong_password_or_name_gets_the_same_refusal(deps, name: str, password: str) -> None:
    with pytest.raises(PublicRefusal) as caught:
        sign_in(signin_request(name, password), deps, T0)
    assert (caught.value.status, caught.value.message) == (400, sessions.NO_MATCH)


@pytest.mark.parametrize(("token", "cookie"), [(TOKEN, None), (TOKEN, "u" * 43), ("", "")])
def test_a_missing_or_wrong_form_token_is_refused_before_anything(deps, token, cookie) -> None:
    deps.aws.calls.clear()
    with pytest.raises(PublicRefusal) as caught:
        sign_in(signin_request("judges", "right-password", token=token, cookie=cookie), deps, T0)
    assert caught.value.kind == "csrf" and deps.aws.calls == []


def test_five_failures_from_one_address_make_it_wait_without_a_password_check(
        deps, monkeypatch: pytest.MonkeyPatch) -> None:
    for minute in range(5):
        with pytest.raises(PublicRefusal):
            sign_in(signin_request("judges", "wrong"), deps, T0 + timedelta(minutes=minute))
    runs: list[str] = []
    real = accounts.hash_password
    monkeypatch.setattr(accounts, "hash_password", lambda *a: runs.append("x") or real(*a))
    with pytest.raises(PublicRefusal) as caught:
        sign_in(signin_request("judges", "right-password"), deps, T0 + timedelta(minutes=5))
    assert caught.value.status == 429 and "Wait 14 minutes" in caught.value.message and runs == []
    # A spoofed X-Forwarded-For changes nothing: only the source address counts.
    spoofed = signin_request("judges", "right-password", headers={"x-forwarded-for": "198.51.100.1"})
    with pytest.raises(PublicRefusal):
        sign_in(spoofed, deps, T0 + timedelta(minutes=6))
    # Another address signs in, and after the wait the first does too.
    assert sign_in(signin_request("judges", "right-password", ip="198.51.100.9"), deps,
                   T0 + timedelta(minutes=6)).status == 303
    assert sign_in(signin_request("judges", "right-password"), deps,
                   T0 + timedelta(minutes=20)).status == 303


def test_failures_outside_the_window_dont_count(deps) -> None:
    for minute in (0, 10, 20, 30, 40, 50):
        with pytest.raises(PublicRefusal) as caught:
            sign_in(signin_request("judges", "wrong"), deps, T0 + timedelta(minutes=minute))
        assert caught.value.status == 400


def test_a_session_ends_by_expiry_sign_out_a_new_uid_or_a_removed_account(deps) -> None:
    cookies, _ = sign_in_as(deps, "judges", T0)
    request = page_request(cookies)
    assert current(request, deps, T0) is not None
    assert current(request, deps, T0 + timedelta(days=7)) is None
    deps.aws.items[("account#judges", "profile")]["uid"] = "v" * 32
    assert current(request, deps, T0) is None
    cookies, _ = sign_in_as(deps, "judges", T0)
    session = current(page_request(cookies), deps, T0)
    response = sign_out(session, deps, T0)
    assert response.headers["Location"] == "/signin"
    assert any(c.startswith(SESSION_COOKIE + "=;") for c in response.cookies)
    assert current(page_request(cookies), deps, T0) is None
    cookies, _ = sign_in_as(deps, "judges", T0)
    accounts.remove_account(deps.aws, "judges")
    assert current(page_request(cookies), deps, T0) is None


@pytest.mark.parametrize("value", ["", "judges", "judges." + "a" * 42, "Judges." + "a" * 43,
                                   "judges." + "a" * 42 + "=", "judges." + "a" * 42 + "/"])
def test_a_malformed_cookie_is_signed_out_with_no_aws_call(deps, value: str) -> None:
    deps.aws.calls.clear()
    assert current(page_request({SESSION_COOKIE: value}), deps, T0) is None
    assert deps.aws.calls == []
