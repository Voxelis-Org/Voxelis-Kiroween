"""The `[website]` reader, app/config.py: the settled values and the refusals by key name."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import pytest

from splatpipe.errors import PipelineError

from app.config import KEYS, WebsiteConfig, dollars_to_cents, load_website_config
from tests.website_helpers import REAL_CONFIG, website_mapping


def test_the_real_config_holds_the_settled_values() -> None:
    wc = load_website_config(REAL_CONFIG)
    assert wc.stop_starting_at == datetime(2027, 2, 4, 8, 0, tzinfo=timezone.utc)
    assert (wc.daily_jobs, wc.daily_cost_cents, wc.total_cost_cents) == (10, 5200, 50000)
    assert wc.boot_margin_minutes == 5
    assert wc.ended_check_minutes == 10
    assert (wc.upload_link_seconds, wc.download_link_seconds) == (900, 300)
    assert (wc.scrypt.n, wc.scrypt.r, wc.scrypt.p, wc.scrypt.dklen) == (32768, 8, 3, 32)
    assert wc.cloudshell_modules == ("app.deploy", "app.account_command", "app.showcase_command")
    assert wc.deploy_folder == PurePosixPath("/tmp/voxelis-website")


def test_every_real_key_is_known_and_none_is_shared_with_other_sections() -> None:
    import tomllib

    data = tomllib.loads(REAL_CONFIG.read_text(encoding="utf-8"))
    assert set(data["website"]) == KEYS
    for other in ("jobs", "clean", "cloud"):
        assert not set(data["website"]) & set(data[other]), other


@pytest.mark.parametrize(("value", "cents"), [(52, 5200), (500, 50000), (0.1, 10), (4.94, 494),
                                              (7.42, 742)])
def test_dollars_to_cents_follows_the_two_decimal_text(value: float, cents: int) -> None:
    assert dollars_to_cents(value) == cents == int(f"{value:.2f}".replace(".", ""))


def refused(**changes: object) -> str:
    data = website_mapping()
    for key, value in changes.items():
        if value is MISSING:
            del data[key]
        else:
            data[key] = value
    with pytest.raises(PipelineError) as caught:
        WebsiteConfig.from_mapping(data, max_video_bytes=4294967296)
    return str(caught.value)


MISSING = object()


@pytest.mark.parametrize("key", sorted(KEYS))
def test_a_missing_key_is_refused_by_name(key: str) -> None:
    assert key in refused(**{key: MISSING})


@pytest.mark.parametrize("key", ["daily_jobs", "upload_part_bytes", "scrypt_n", "session_days",
                                 "max_body_bytes"])
def test_a_bool_or_a_low_number_is_refused_by_name(key: str) -> None:
    assert key in refused(**{key: True})
    assert key in refused(**{key: 0})
    assert key in refused(**{key: "10"})


def test_other_refusals_name_their_key() -> None:
    assert "upload_part_bytes" in refused(upload_part_bytes=4 * 1024 * 1024)
    assert "copy_part_bytes" in refused(copy_part_bytes=1024)
    assert "scrypt_n" in refused(scrypt_n=1000)
    assert "daily_cost_usd" in refused(daily_cost_usd=math.nan)
    assert "total_cost_usd" in refused(total_cost_usd=False)
    assert "download_link_seconds" in refused(download_link_seconds=8 * 24 * 3600)
    for bad in ("2027-02-04 08:00:00", "2027-02-04T08:00:00+00:00", "2027-02-30T08:00:00Z", 5):
        assert "stop_starting_at" in refused(stop_starting_at=bad)
    for bad in ([], ["app deploy"], "app.deploy", [3]):
        assert "cloudshell_modules" in refused(cloudshell_modules=bad)
    for bad in ("/home/x", "tmp/x", "/tmp", "/tmp/../etc"):
        assert "deploy_folder" in refused(deploy_folder=bad)
    assert "typo_key" in refused(typo_key=1)


def test_the_config_loader_refuses_a_toml_date_in_website(tmp_path: Path) -> None:
    text = REAL_CONFIG.read_text(encoding="utf-8").replace(
        'stop_starting_at = "2027-02-04T08:00:00Z"', "stop_starting_at = 2027-02-04T08:00:00Z")
    path = tmp_path / "config.toml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(PipelineError, match="stop_starting_at"):
        load_website_config(path)


def test_too_many_parts_for_the_largest_video_is_refused() -> None:
    data = website_mapping()
    with pytest.raises(PipelineError, match="upload_part_bytes"):
        WebsiteConfig.from_mapping(data, max_video_bytes=data["upload_part_bytes"] * 10_001)
