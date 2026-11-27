"""Starting a Job from the website: the Start_Precheck, the Website_Caller, one flow per row of the
design's start table with its message, its Reservation and its job item, and the call order."""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import pytest

import cloud.start
from app import website_start
from app.errors import PublicRefusal
from app.limits import TOTAL, upload_key
from app.sessions import Session
from cloud.aws import InstanceInfo
from cloud.spend import SpendRefused
from cloud.table import Key
from splatpipe.config import load_config
from splatpipe.jobs import open_job
from tests.website_helpers import mp4, website

UPLOAD = "0123456789abcdef0123456789abcdef"
SESSION = Session(account="judges", key=Key("account#judges", "session#x"), csrf="c", uid="u")


@pytest.fixture
def deps(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    deps = website(tmp_path)
    deps.spend_refused = False

    def check_spend(**kwargs):
        deps.aws.calls.append(("check_spend", kwargs["machine"]))
        assert kwargs["environ"] == {"APPROVED_SPEND": "1"}
        if deps.spend_refused:
            raise SpendRefused("Refused to start for the test.")

    monkeypatch.setattr(cloud.start, "check_spend", check_spend)
    return deps


def seed_upload(deps, data: bytes, name: str = "IMG_0042.MP4", upload_id: str = UPLOAD,
                state: str = "complete") -> str:
    key = upload_key("judges", upload_id)
    deps.aws.items[(key.pk, key.sk)] = {"pk": key.pk, "sk": key.sk, "key": f"uploads/{upload_id}",
                                        "s3_upload_id": "x", "name": name, "size": len(data),
                                        "parts": 1, "state": state}
    deps.aws.objects[f"uploads/{upload_id}"] = data
    return upload_id


def total(deps) -> int:
    return deps.aws.items.get((TOTAL.pk, TOTAL.sk), {}).get("cents", 0)


def ops(deps) -> list[str]:
    return [op for op, _ in deps.aws.calls]


def refused(deps, preset: str = "fast", upload: str = UPLOAD) -> PublicRefusal:
    with pytest.raises(PublicRefusal) as caught:
        website_start.start(deps, SESSION, upload, preset, deps.clock())
    return caught.value


def job_item(deps, job_id: str) -> dict:
    return deps.aws.items[("account#judges", f"job#{job_id}")]


def test_a_start_reserves_then_asks_the_spend_guard_then_copies_and_launches(deps, tmp_path) -> None:
    data = mp4(12)
    seed_upload(deps, data)
    started = website_start.start(deps, SESSION, UPLOAD, "fast", deps.clock())
    assert started.outcome == "launched"
    calls = ops(deps)
    order = [calls.index(op) for op in ("table_transact", "check_spend", "s3_copy", "s3_upload",
                                        "run_instances")]
    assert order == sorted(order)
    prefix = f"jobs/{started.job_id}/input"
    assert deps.aws.objects[f"{prefix}/video.mp4"] == data
    assert deps.aws.objects[f"{prefix}/code.tar"] == b"fake code archive"
    item = job_item(deps, started.job_id)
    # steps is the real Step_List's length: frames, cameras, depth, walls, train and clean.
    assert (item["state"], item["launch"], item["instance_type"], item["steps"]) == (
        "held", "launched", "g5.8xlarge", 6)
    assert total(deps) == 515
    assert deps.aws.items[("account#judges", f"upload#{UPLOAD}")]["used_by"] == started.job_id
    # The Job_Record reads back with splatpipe.jobs' own rules.
    folder = tmp_path / "jobs" / started.job_id / "input"
    folder.mkdir(parents=True)
    (folder / "job.json").write_bytes(deps.aws.objects[f"{prefix}/job.json"])
    (folder / "video.mp4").write_bytes(data)
    record = open_job(started.job_id, load_config()).record
    assert (record.preset, record.source_name, record.video_seconds, record.start_seconds,
            record.end_seconds) == ("fast", "IMG_0042.MP4", 12.0, 0.0, 12.0)
    # The same Upload can't start a second Job.
    assert refused(deps).kind == "bad_request"


def test_the_precheck_takes_nothing_and_refuses_like_the_start(deps) -> None:
    website_start.precheck(deps, SESSION, "default", deps.clock())
    assert "table_transact" not in ops(deps) and "run_instances" not in ops(deps)
    with pytest.raises(PublicRefusal) as caught:
        website_start.precheck(deps, SESSION, "best", deps.clock())
    assert caught.value.kind == "bad_request"


def busy_instances() -> list[InstanceInfo]:
    return [InstanceInfo("i-1", "g5.8xlarge", "running", "us-east-1a", "", "")]


@pytest.mark.parametrize(("setup", "kind", "message"), [
    (lambda d: setattr(d.aws, "instances", busy_instances()), "busy", website_start.BUSY),
    (lambda d: setattr(d.aws, "quota", 0.0), "cant_start", website_start.CANT_START),
    (lambda d: setattr(d.aws, "images", []), "cant_start", website_start.CANT_START),
    (lambda d: d.aws.objects.pop("jobs/_smoke/ami-0123456789abcdef0.json"), "cant_start",
     website_start.CANT_START),
    (lambda d: d.aws.stack.pop("VpcId"), "cant_start", website_start.CANT_START),
    (lambda d: setattr(d.aws, "offerings", {}), "cant_start", website_start.CANT_START),
    (lambda d: d.aws.items.__setitem__(("limits", "total"),
                                       {"pk": "limits", "sk": "total", "cents": 49999}),
     "limit", None),
])
def test_refusals_before_the_reservation_take_nothing(deps, setup, kind, message) -> None:
    seed_upload(deps, mp4(12))
    setup(deps)
    refusal = refused(deps)
    assert refusal.kind == kind
    if message:
        assert refusal.message == message
    assert "table_transact" not in ops(deps) and "check_spend" not in ops(deps)
    assert total(deps) in (0, 49999)


def test_a_lock_hash_drift_cant_start(deps) -> None:
    seed_upload(deps, mp4(12))
    deps.aws.images[0].tags["uv_lock_hash"] = "d" * 64
    assert refused(deps).message == website_start.CANT_START


def test_a_video_refusal_and_a_gone_upload_take_nothing(deps) -> None:
    seed_upload(deps, mp4(2))
    assert refused(deps).kind == "video"
    seed_upload(deps, mp4(12))
    del deps.aws.objects[f"uploads/{UPLOAD}"]
    assert "isn't there any more" in refused(deps).message
    assert "table_transact" not in ops(deps)


def test_a_conflict_says_press_start_again(deps) -> None:
    seed_upload(deps, mp4(12))
    deps.aws.conflicts = 1
    refusal = refused(deps)
    assert refusal.kind == "conflict" and "Press Start again" in refusal.message
    assert total(deps) == 0


def reservation_given_back(deps) -> None:
    assert total(deps) == 0
    jobs = [item for (pk, sk), item in deps.aws.items.items() if sk.startswith("job#")]
    assert len(jobs) == 1 and jobs[0]["state"] == "released"
    assert "used_by" not in deps.aws.items[("account#judges", f"upload#{UPLOAD}")]


def test_a_spend_guard_refusal_gives_the_reservation_back(deps) -> None:
    seed_upload(deps, mp4(12))
    deps.spend_refused = True
    assert refused(deps).message == website_start.CANT_START
    assert "s3_copy" not in ops(deps) and "run_instances" not in ops(deps)
    reservation_given_back(deps)


def test_the_stop_time_after_the_copy_launches_nothing(deps) -> None:
    seed_upload(deps, mp4(12))
    real = deps.aws.s3_copy

    def copy_then_late(*args, **kwargs):
        real(*args, **kwargs)
        deps.clock.now = deps.website.stop_starting_at + timedelta(seconds=1)

    deps.aws.s3_copy = copy_then_late
    assert refused(deps).kind == "stopped"
    assert "run_instances" not in ops(deps)
    reservation_given_back(deps)


def test_a_failed_copy_gives_back_and_says_press_start_again(deps) -> None:
    seed_upload(deps, mp4(12))
    deps.aws.copy_failures.add(f"uploads/{UPLOAD}")
    refusal = refused(deps)
    assert refusal.message == website_start.COPY_FAILED and "run_instances" not in ops(deps)
    reservation_given_back(deps)


@pytest.mark.parametrize("field", ["no_capacity", "quota_refused"])
def test_no_capacity_or_a_quota_refusal_is_busy_and_gives_back(deps, field) -> None:
    seed_upload(deps, mp4(12))
    getattr(deps.aws, field).update({(t, "us-east-1a") for t in
                                     ("g5.8xlarge", "g5.4xlarge", "g6.4xlarge", "g5.2xlarge",
                                      "g6.2xlarge")})
    assert refused(deps).message == website_start.BUSY
    reservation_given_back(deps)


def test_another_launch_error_keeps_the_reservation_and_says_unknown(deps) -> None:
    seed_upload(deps, mp4(12))
    deps.aws.launch_failures.add(("g5.8xlarge", "us-east-1a"))
    started = website_start.start(deps, SESSION, UPLOAD, "fast", deps.clock())
    assert started.outcome == "unknown"
    assert job_item(deps, started.job_id)["launch"] == "unknown"
    assert job_item(deps, started.job_id)["state"] == "held" and total(deps) == 515
    assert ops(deps).count("run_instances") == 1


def test_another_accounts_or_an_unfinished_upload_cant_start(deps) -> None:
    seed_upload(deps, mp4(12), state="open")
    assert refused(deps).kind == "bad_request"
    other = Session(account="other", key=Key("account#other", "session#y"), csrf="c", uid="u")
    seed_upload(deps, mp4(12))
    with pytest.raises(PublicRefusal):
        website_start.start(deps, other, UPLOAD, "fast", deps.clock())
    assert json.dumps(deps.aws.calls).count("check_spend") == 0
