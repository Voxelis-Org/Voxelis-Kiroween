"""Accounts and the Account_Command: hashes, make, new password, remove, and the CloudShell guard."""

from __future__ import annotations

import io
from datetime import datetime, timezone

import pytest

from app import accounts
from app.account_command import account_main
from app.accounts import (ACCOUNT_NAME, PASSWORD_ALPHABET, Account, check_account_name,
                          hash_password, make_account, password_matches, profile_key, read_account,
                          remove_account, replace_password)
from app.config import ScryptSettings
from cloud.errors import CloudError
from tests.fake_aws import FakeAws
from tests.website_helpers import REAL_CONFIG, SeededRandom, T0, website_config

CLOUDSHELL = {"AWS_EXECUTION_ENV": "CloudShell"}


def test_a_hash_with_the_real_settings_fits_its_maxmem() -> None:
    wc = website_config(scrypt_n=32768, scrypt_r=8, scrypt_p=3)
    assert len(hash_password("x" * 24, b"s" * 16, wc.scrypt)) == 32


def test_the_tests_tiny_scrypt_settings_also_fit() -> None:
    assert len(hash_password("pw", b"s", ScryptSettings(2, 1, 1, 32, 16))) == 32


def test_password_matches_runs_one_scrypt_either_way(monkeypatch: pytest.MonkeyPatch) -> None:
    wc = website_config()
    runs: list[bytes] = []
    real = accounts.hash_password
    monkeypatch.setattr(accounts, "hash_password", lambda p, s, c: runs.append(s) or real(p, s, c))
    account = Account("judges", "u", real("right", b"salt", wc.scrypt), b"salt", wc.scrypt)
    assert password_matches("right", account, wc.scrypt)
    assert not password_matches("wrong", account, wc.scrypt)
    assert not password_matches("right", None, wc.scrypt)
    assert len(runs) == 3


@pytest.mark.parametrize("name", ["judges", "a", "team-2", "a" * 32])
def test_good_names_pass(name: str) -> None:
    assert check_account_name(name) == name


@pytest.mark.parametrize("name", ["", "Judges", "2team", "a" * 33, "a.b", "a b", "judges\n", "ü"])
def test_bad_names_are_refused_with_the_pattern(name: str) -> None:
    with pytest.raises(CloudError, match="pattern"):
        check_account_name(name)
    assert not ACCOUNT_NAME.fullmatch(name)


def test_make_replace_and_remove_an_account() -> None:
    fake, wc, random = FakeAws(), website_config(), SeededRandom()
    password = make_account(fake, "judges", wc, random, T0)
    assert len(password) == wc.password_length and set(password) <= set(PASSWORD_ALPHABET)
    item = fake.table_get(profile_key("judges"))
    account = read_account("judges", item)
    assert password_matches(password, account, wc.scrypt)
    assert password not in str(item)
    with pytest.raises(CloudError, match="--new-password"):
        make_account(fake, "judges", wc, random, T0)
    new = replace_password(fake, "judges", wc, random)
    item2 = fake.table_get(profile_key("judges"))
    assert new != password and item2["uid"] != item["uid"]
    assert password_matches(new, read_account("judges", item2), wc.scrypt)
    fake.items[("account#judges", "job#20261201-120000-abcdef")] = {"pk": "account#judges",
                                                                   "sk": "job#20261201-120000-abcdef"}
    fake.items[("limits", "total")] = {"pk": "limits", "sk": "total", "cents": 5}
    assert remove_account(fake, "judges") == 2
    assert list(fake.items) == [("limits", "total")]
    with pytest.raises(CloudError, match="no account"):
        replace_password(fake, "judges", wc, random)


def run(argv: list[str], fake: FakeAws, environ: dict[str, str] | None = None,
        seed: int = 1) -> tuple[int, str]:
    out = io.StringIO()
    code = account_main(argv, environ=CLOUDSHELL if environ is None else environ,
                        make_aws=lambda region: fake, random=SeededRandom(seed), out=out,
                        config_path=REAL_CONFIG, now=datetime(2026, 12, 1, tzinfo=timezone.utc))
    return code, out.getvalue()


def stacked() -> FakeAws:
    fake = FakeAws(stack={"WebsiteTableName": "the-table", "BucketName": "b"})
    fake.table = None
    return fake


def test_the_command_prints_a_new_password_once_and_alone(capsys: pytest.CaptureFixture[str]) -> None:
    fake = stacked()
    code, out = run(["judges"], fake)
    assert code == 0 and fake.table == "the-table"
    lines = out.splitlines()
    assert len(lines) == 1 and len(lines[0]) == 24
    assert lines[0] not in capsys.readouterr().err
    code, again = run(["judges"], fake)
    assert code == 1 and again == "" and "--new-password" in capsys.readouterr().err
    code, out2 = run(["judges", "--new-password"], fake, seed=2)
    assert code == 0 and out2.strip() != lines[0]
    code, removed = run(["judges", "--remove"], fake)
    assert code == 0 and "Removed the account judges" in removed


def test_the_command_refuses_outside_cloudshell_before_any_aws(capsys: pytest.CaptureFixture[str]) -> None:
    fake = stacked()
    assert run(["judges"], fake, environ={"AWS_EXECUTION_ENV": "AWS_Lambda_python3.11"})[0] == 1
    assert "only in AWS CloudShell" in capsys.readouterr().err
    assert fake.calls == []


def test_the_command_refuses_a_bad_name_and_a_stack_without_the_website(
        capsys: pytest.CaptureFixture[str]) -> None:
    fake = stacked()
    assert run(["Bad_Name"], fake)[0] == 1 and "pattern" in capsys.readouterr().err
    assert fake.calls == []
    fake.stack = {"BucketName": "b"}
    assert run(["judges"], fake)[0] == 1 and "WebsiteTableName" in capsys.readouterr().err
