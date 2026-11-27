"""The Website_Status from a Job's files, Ended without a result, the Result_File and the job list."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest

from app.statuses import job_view, list_jobs, may_see, result_file, status_text, website_status
from cloud.aws import InstanceInfo
from tests.website_helpers import JOB_ID, T0, seed_job, website

CLEAN = {"clean_sog": "clean.sog", "train_ply": "train.ply"}


@pytest.fixture
def deps(tmp_path: Path):
    return website(tmp_path)


def view(deps, now=T0, job_id=JOB_ID, timings=True):
    item = deps.aws.items.get(("account#judges", f"job#{job_id}"))
    return job_view(deps.aws, job_id, item, deps.config, deps.website, now, timings=timings)


def test_the_pure_status() -> None:
    assert website_status([], None, 5) == ("starting", None)
    assert website_status(["reports/frames.json", "cameras.json"], None, 5) == ("running", 2)
    many = [f"reports/{n}.json" for n in range(9)]
    assert website_status(many, None, 5) == ("running", 5)
    assert website_status(["job.json"], {"status": "failed"}, 5) == ("failed", None)
    assert website_status(["job.json"], {"status": "weird"}, 5) == ("running", 1)


def test_done_done_with_warnings_and_failed_from_the_summary(deps) -> None:
    seed_job(deps)
    done = view(deps)
    assert (done.status, done.result_file, done.has_cameras, done.run_seconds) == (
        "done", "clean.sog", True, 1830.0)
    assert [t.step for t in done.step_times] == ["cameras", "clean", "depth", "frames", "train"]
    deps.aws.objects.clear()
    seed_job(deps, status="done_with_warnings", files=("train.ply",))
    warned = view(deps)
    assert warned.result_file == "train.ply" and status_text(warned) == "Done with warnings"
    assert warned.warnings == (("clean", "The clean step ran out of time."),)
    deps.aws.objects.clear()
    seed_job(deps, status="failed", files=())
    failed = view(deps)
    assert failed.readable_error == "Training ran out of GPU memory." and failed.result_file is None


def test_a_done_job_with_no_splat_has_no_result_file() -> None:
    assert result_file(["job.json", "cameras.json"], CLEAN) is None
    assert result_file(["clean.sog", "train.ply"], CLEAN) == "clean.sog"


def test_starting_and_running_and_a_half_written_summary(deps) -> None:
    seed_job(deps, status=None, files=(), reports=())
    assert view(deps).status == "starting"
    seed_job(deps, status=None, files=(), reports=("frames", "cameras"))
    assert status_text(view(deps)) == "Running (step 3 of 5)"
    deps.aws.objects[f"jobs/{JOB_ID}/out/job.json"] = b'{"status": "do'
    assert view(deps).status == "running"


def test_ended_without_a_result_needs_time_no_machines_and_a_second_empty_read(deps) -> None:
    seed_job(deps, status=None, files=(), reports=("frames",))
    later = T0 + timedelta(minutes=10)
    assert view(deps, now=T0 + timedelta(minutes=9)).status == "running"
    assert view(deps, now=later).status == "ended"
    deps.aws.instances = [InstanceInfo("i-1", "g5.8xlarge", "running", "us-east-1a", "", "")]
    deps.aws.instance_jobs["i-1"] = JOB_ID
    assert view(deps, now=later).status == "running"
    deps.aws.instances = [InstanceInfo("i-1", "g5.8xlarge", "terminated", "us-east-1a", "", "")]
    assert view(deps, now=later).status == "ended"
    deps.aws.job_machines_error = True
    assert view(deps, now=later).status == "running"


def test_the_second_read_can_find_a_summary_that_just_arrived(deps) -> None:
    seed_job(deps, status=None, files=(), reports=("frames",))
    real = deps.aws.job_machines

    def lookup_then_summary(job_id):
        seed_job(deps, status="done")
        return real(job_id)

    deps.aws.job_machines = lookup_then_summary
    assert view(deps, now=T0 + timedelta(hours=1)).status == "done"


def test_the_job_list_is_held_jobs_newest_first_and_capped(deps) -> None:
    ids = [f"2026120{d}-120000-abcdef" for d in range(1, 6)]
    for job_id in ids:
        seed_job(deps, job_id)
    seed_job(deps, "20261209-120000-abcdef", state="released")
    seed_job(deps, "20261208-120000-abcdef", account="other")
    deps.website = deps.website.__class__(**{**deps.website.__dict__, "max_listed_jobs": 3})
    listed = [v.job_id for v in list_jobs(deps.aws, "judges", deps.config, deps.website, T0)]
    assert listed == ids[::-1][:3]


def test_may_see_is_the_accounts_held_job_or_a_showcase_entry(deps) -> None:
    seed_job(deps)
    assert may_see(deps.aws, "judges", JOB_ID)[0] is not None
    assert may_see(deps.aws, "other", JOB_ID) == (None, None)
    deps.aws.items[("showcase", f"job#{JOB_ID}")] = {"pk": "showcase", "sk": f"job#{JOB_ID}"}
    assert may_see(deps.aws, "other", JOB_ID)[1] is not None
