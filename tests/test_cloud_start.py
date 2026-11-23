"""Unit tests for the Start_Command, cloud/start.py (cloud-jobs task 5.5).

The Start_Command does every free, local and read-only check first and spends money only at the very
end, so these tests drive `start_job` against the `FakeAws` seam, a committed `git_repo` and a
monkeypatched `splatpipe.run.STEP_LIST`, all on the CPU with no network and no AWS account. The
"paid" calls are the first `s3_upload` and the `run_instances` call; `_paid` reads them back off the
fake's ordered `calls` list, so every refusal can show it fired before any paid action (Req 6.1).

- An empty Step_List raises `splatpipe.run`'s own "No steps are set up yet…" text, before anything,
  with no upload and no launch (Req 7.2). The shipped default is already `()`; the monkeypatch makes
  the case explicit and pytest restores it, so the real default never changes.
- Each refusal happens before any paid action: an uncommitted checked path (Req 7.3), a lock-hash
  drift against the image's tags (Req 7.5, 7.6), a missing Smoke_Marker (Req 7.7), no Machine_Image
  (Req 7.9), a full G_Quota (Req 4.5) and an unmet hard need (Req 3.4) each raise with no
  `s3_upload`/`run_instances` recorded.
- Happy path: with the Spend_Guard said yes (`APPROVED_SPEND=1`), the video, the Code_Archive and
  the Job_Record upload to `jobs/<id>/input/`, exactly one machine launches, and the chosen type and
  zone are reported (Req 7.1, 7.8, 5.3).
- The money quote's rough and worst-case dollars come from the Config prices, through `rough_cost`
  over each Machine_Spec's `price_per_hour_usd` (Req 6.3).

`make_job` is wrapped so it reads a stand-in Video_Probe and a fixed Job_Id instead of shelling out
to ffprobe, the way pipeline-core's `new_job` fixture makes Jobs without ffprobe. Nothing in
`splatpipe/` or `cloud/` is changed; the Spend_Guard is the real one, fed each test's own `environ`.
Property 6 (nothing paid before the Spend_Guard returns) is the separate task 5.6.
"""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any, NamedTuple
from unittest.mock import patch

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import cloud.start
import splatpipe.make
import splatpipe.run
from splatpipe.config import PRESETS, Config
from splatpipe.errors import PipelineError
from splatpipe.make import CONTAINER_FAMILY
from splatpipe.probe import ProbeResult
from splatpipe.run import StepEntry

from cloud.aws import ImageInfo, InstanceInfo
from cloud.config import CloudConfig
from cloud.errors import CloudError
from cloud.repo import lock_hashes
from cloud.spend import SpendRefused, check_spend as _real_check_spend
from cloud.spend_file import rough_cost
from cloud.start import _smoke_marker_key, start_job

from fake_aws import FakeAws, cloud_config, git_repo

# The fixed Job_Id the wrapped make_job mints, so the upload keys a test asserts on are known.
FIXED_ID = "20261119-142530-7f3a9c"
# A real-shaped AMI id (hex, so render_user_data's AMI pattern accepts it on the happy path).
IMAGE_ID = "ami-0123456789abcdef0"
# The Stack name and Project_Tag read from the test [aws] table; the fake ignores the name.
STACK_NAME = "voxelis-kiroween"
PROJECT_TAG = "project=voxelis-kiroween"

# A tiny Step_List of Fake_Steps. The modules are never imported (start_job runs no Step); only the
# names matter, each naming a Config section step_needs_over reads for the Job's Preset.
STEP_NAMES = ("alpha", "beta")
STEP_ENTRIES = tuple(
    StepEntry(name=name, module=f"splatpipe.{name}", required=True) for name in STEP_NAMES)

# The one type the fake offers a zone and default subnet for, so launch_plan lands on g5.8xlarge,
# the fast preset's primary and first admitted type.
OFFERINGS = {"g5.8xlarge": ["us-east-1a"]}
SUBNETS = {"us-east-1a": "subnet-a"}

# The seven Stack_Outputs resolve_stack maps. No value holds a 12-digit account id (the structural
# redaction check lives in the aws/stack tests, not here).
STACK = {
    "BucketName": "voxelis-kiroween-jobs",
    "JobRoleArn": "arn:aws:iam::role/voxelis-job",
    "JobInstanceProfileArn": "arn:aws:iam::instance-profile/voxelis-job",
    "RunnerRoleArn": "arn:aws:iam::role/voxelis-runner",
    "SecurityGroupId": "sg-fake",
    "VpcId": "vpc-fake",
    "GroupName": "voxelis-kiroween",
}


class Scenario(NamedTuple):
    """The pieces a Start_Command test injects: the repo, the two configs, the fake and the video."""

    root: Path
    config: Config
    cc: CloudConfig
    aws: FakeAws
    video: Path


def _fake_probe(path: Path, timeout_seconds: float) -> ProbeResult:
    """A stand-in Video_Probe: a 10-second MP4, so make_job needs no ffprobe (CPU, no tools)."""
    return ProbeResult(format_name=CONTAINER_FAMILY, has_video=True, video_seconds=10.0)


def _install_make_job(monkeypatch: pytest.MonkeyPatch) -> None:
    """Wrap `cloud.start.make_job` so it uses the stand-in probe and the fixed Job_Id.

    start_job calls `make_job(video, start=, end=, preset=, config=)`; the wrapper forwards those and
    adds the test probe and `new_id`, so the real make_job still makes the Job_Folder and Job_Record,
    without ffprobe. No file under `splatpipe/` is touched.
    """
    def fake_make_job(video: str | Path, *, start: float | None = None, end: float | None = None,
                      preset: str | None = None, config: Config | None = None) -> str:
        return splatpipe.make.make_job(video, start=start, end=end, preset=preset, config=config,
                                       probe=_fake_probe, new_id=lambda: FIXED_ID)

    monkeypatch.setattr(cloud.start, "make_job", fake_make_job)


def _config(tmp_path: Path, *, big_need_step: str | None = None) -> Config:
    """A pipeline Config with [jobs] inside tmp_path, [aws] for the Stack and the Project_Tag, and a
    section per Fake_Step. `big_need_step` gives that Step a memory need no Allowed_Type meets."""
    data: dict[str, Any] = {
        "jobs": {
            "jobs_folder": (tmp_path / "jobs").as_posix(),  # absolute, so it stays inside tmp_path
            "default_preset": "fast",
            "max_video_bytes": 1024,
            "max_video_seconds": 600,
            "min_used_seconds": 3,
            "ffprobe_timeout_seconds": 30,
        },
        "aws": {"stack_name": STACK_NAME, "project_tag": PROJECT_TAG},
    }
    for name in STEP_NAMES:
        need = {"min_memory_gb": 100000} if name == big_need_step else {}
        data[name] = {preset: dict(need) for preset in ("fast", "default", "quality")}
    return Config(data, source="test config")


def _image(uv_hash: str, package_hash: str) -> ImageInfo:
    """An available tagged Machine_Image whose lock-hash tags are `uv_hash` and `package_hash`."""
    return ImageInfo(id=IMAGE_ID, name="voxelis-image", state="available",
                     creation_date="2026-11-20T00:00:00+00:00",
                     tags={"uv_lock_hash": uv_hash, "package_lock_hash": package_hash})


def _scenario(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *,
              entries: tuple[StepEntry, ...] = STEP_ENTRIES, big_need_step: str | None = None,
              images: str = "match", marker: bool = True, quota: float = 32.0,
              running: tuple[InstanceInfo, ...] = ()) -> Scenario:
    """Build the healthy scenario, then bend one knob per refusal test.

    `images` is "match" (tags equal HEAD's lock hashes), "drift" (tags differ) or "none" (no image);
    `marker` seeds the Smoke_Marker for the image; `quota` and `running` set the G_Quota picture.
    Uses `git_repo`, which skips the test with a reason when git isn't installed.
    """
    root = git_repo(tmp_path)
    uv_hash, package_hash = lock_hashes(cwd=root)
    if images == "match":
        imgs = [_image(uv_hash, package_hash)]
    elif images == "drift":
        imgs = [_image("0" * 64, "f" * 64)]  # valid sha-256 shape, but not HEAD's
    else:  # "none"
        imgs = []
    objects = {_smoke_marker_key(IMAGE_ID): b"{}"} if marker else {}
    aws = FakeAws(stack=STACK, images=imgs, objects=objects, quota=quota,
                  offerings=OFFERINGS, subnets=SUBNETS, instances=running)
    monkeypatch.setattr(splatpipe.run, "STEP_LIST", entries)
    _install_make_job(monkeypatch)
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"a few bytes")
    return Scenario(root, _config(tmp_path, big_need_step=big_need_step), cloud_config(), aws, video)


def _paid(aws: FakeAws) -> list[str]:
    """The paid operations recorded on the fake: the S3 uploads and the RunInstances calls."""
    return [op for op, _ in aws.calls if op in ("s3_upload", "run_instances")]


# Req 7.2: an empty Step_List refuses before anything, with no upload or launch.

def test_empty_step_list_refuses_with_no_upload_or_launch(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    aws = FakeAws()
    monkeypatch.setattr(splatpipe.run, "STEP_LIST", ())  # explicit; the shipped default is already ()
    with pytest.raises(PipelineError) as caught:
        start_job(tmp_path / "clip.mp4", aws=aws)
    assert "No steps are set up yet" in str(caught.value)
    assert _paid(aws) == []


# Req 7.3: an uncommitted checked path refuses before any paid action.

def test_uncommitted_changes_refuse_before_any_paid_action(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    scn = _scenario(tmp_path, monkeypatch)
    (scn.root / "config.toml").write_text("changed\n", encoding="utf-8")  # dirty a checked path
    with pytest.raises(CloudError) as caught:
        start_job(scn.video, aws=scn.aws, config=scn.config, cloud_config=scn.cc, repo_root=scn.root)
    assert "config.toml" in str(caught.value)
    assert _paid(scn.aws) == []


# Req 7.9: no available Machine_Image refuses before any paid action.

def test_missing_machine_image_refuses_before_any_paid_action(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    scn = _scenario(tmp_path, monkeypatch, images="none")
    with pytest.raises(CloudError) as caught:
        start_job(scn.video, aws=scn.aws, config=scn.config, cloud_config=scn.cc, repo_root=scn.root)
    assert "cloud.image" in str(caught.value)  # points at the Image_Build
    assert _paid(scn.aws) == []


# Req 7.5, 7.6: a lock-hash drift against the image's tags refuses before any paid action.

def test_lock_hash_drift_refuses_before_any_paid_action(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    scn = _scenario(tmp_path, monkeypatch, images="drift")
    with pytest.raises(CloudError) as caught:
        start_job(scn.video, aws=scn.aws, config=scn.config, cloud_config=scn.cc, repo_root=scn.root)
    assert "cloud.image" in str(caught.value)  # says to rebuild the image
    assert _paid(scn.aws) == []


# Req 7.7: a missing Smoke_Marker refuses before any paid action.

def test_missing_smoke_marker_refuses_before_any_paid_action(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    scn = _scenario(tmp_path, monkeypatch, marker=False)
    with pytest.raises(CloudError) as caught:
        start_job(scn.video, aws=scn.aws, config=scn.config, cloud_config=scn.cc, repo_root=scn.root)
    assert "cloud.smoke" in str(caught.value)  # points at the Smoke_Test
    assert _paid(scn.aws) == []


# Req 4.5: a full G_Quota refuses before any paid action.

def test_full_quota_refuses_before_any_paid_action(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # One running g5.8xlarge already uses all 32 vCPUs of the quota, so no type can be admitted.
    running = (InstanceInfo(id="i-existing", type="g5.8xlarge", state="running", zone="us-east-1a",
                            launch_time="2026-11-20T00:00:00+00:00", state_reason=""),)
    scn = _scenario(tmp_path, monkeypatch, quota=32.0, running=running)
    with pytest.raises(CloudError) as caught:
        start_job(scn.video, aws=scn.aws, config=scn.config, cloud_config=scn.cc, repo_root=scn.root)
    assert "quota" in str(caught.value).lower()
    assert _paid(scn.aws) == []


# Req 3.4: an unmet hard need refuses before any paid action.

def test_unmet_hard_need_refuses_before_any_paid_action(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    scn = _scenario(tmp_path, monkeypatch, big_need_step="alpha")  # needs 100000 GB of memory
    with pytest.raises(CloudError) as caught:
        start_job(scn.video, aws=scn.aws, config=scn.config, cloud_config=scn.cc, repo_root=scn.root)
    assert "100000" in str(caught.value)  # names the unmet figure
    assert _paid(scn.aws) == []


# Req 7.1, 7.8, 5.3: the happy path uploads the inputs and launches exactly one machine.

def test_happy_path_uploads_inputs_and_launches_one_machine(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    scn = _scenario(tmp_path, monkeypatch)
    job_id = start_job(scn.video, aws=scn.aws, config=scn.config, cloud_config=scn.cc,
                       repo_root=scn.root, environ={"APPROVED_SPEND": "1"})

    assert job_id == FIXED_ID
    # The video, the Code_Archive and the Job_Record landed under jobs/<id>/input/ (Req 7.8).
    prefix = f"jobs/{FIXED_ID}/input"
    assert f"{prefix}/video.mp4" in scn.aws.objects
    assert f"{prefix}/code.tar" in scn.aws.objects
    assert f"{prefix}/job.json" in scn.aws.objects
    # Exactly one machine launched, the fast preset's primary (Req 5.3, 3.7).
    assert [key for op, key in scn.aws.calls if op == "run_instances"] == ["g5.8xlarge"]
    assert len(scn.aws.instances) == 1
    # The chosen type and zone are reported.
    out = capsys.readouterr().out
    assert "g5.8xlarge" in out
    assert "us-east-1a" in out


# Req 6.3: the quote's rough and worst-case dollars come from the Config prices.

def test_quote_numbers_come_from_config_prices(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    scn = _scenario(tmp_path, monkeypatch)
    captured: dict[str, Any] = {}

    class _Stop(Exception):
        """Stop start_job right at the Spend_Guard, after the quote and before any paid action."""

    def _spy(**kwargs: Any) -> None:
        captured.update(kwargs)
        raise _Stop

    monkeypatch.setattr(cloud.start, "check_spend", _spy)
    with pytest.raises(_Stop):
        start_job(scn.video, aws=scn.aws, config=scn.config, cloud_config=scn.cc, repo_root=scn.root)

    cc = scn.cc
    # The first admitted type (g5.8xlarge) over the fast Expected_Time, and the dearest fitting type
    # (also g5.8xlarge) over the fast Time_Cap: both figures are rough_cost over a Config price.
    expected = rough_cost(30, cc.spec("g5.8xlarge"), cc)
    worst = rough_cost(120, cc.spec("g5.8xlarge"), cc)
    assert f"${expected:.2f}" in captured["rough_cost"]
    assert f"${worst:.2f}" in captured["rough_cost"]
    assert _paid(scn.aws) == []  # the quote is built before any paid action


# Property 6: Nothing paid happens before the Spend_Guard returns (Req 6.1, 6.6, 16.4)

# The types the property's FakeAws offers a zone and default subnet for, so every preset's primary
# can launch on the approved path and one machine starts: fast and default land on g5.8xlarge,
# quality on g6e.4xlarge. The module-level OFFERINGS above only has g5.8xlarge, which the fast-preset
# unit tests want; the property varies the preset, so it also offers the quality primaries.
PROP_OFFERINGS = {
    "g5.8xlarge": ["us-east-1a"],
    "g6e.4xlarge": ["us-east-1a"],
    "g6e.2xlarge": ["us-east-1a"],
}

# The op the patched Spend_Guard records on the fake's timeline the moment check_spend returns
# without raising, so the test can tell which calls came before the guard. It isn't a real Aws op,
# so it never clashes with a recorded (op, key).
GUARD_MARKER = "check_spend_returned"


class Prop6Env(NamedTuple):
    """The example-independent scaffolding Property 6 builds once: the committed repo and HEAD's lock
    hashes, the pipeline Config and the Cloud_Config, and a counter that mints a unique Job_Id per
    example so make_job never meets a Job_Folder that already exists."""

    root: Path
    config: Config
    cc: CloudConfig
    uv_hash: str
    package_hash: str
    ids: itertools.count


@pytest.fixture(scope="module")
def prop6_env(tmp_path_factory: pytest.TempPathFactory) -> Prop6Env:
    """Build the committed git repo and the two configs once for every Property 6 example.

    Module-scoped on purpose: the repo and its lock hashes don't depend on the drawn example, so
    building them once keeps the property fast and keeps a function-scoped fixture out of the
    `@given` test. `git_repo` skips the whole property, with a reason, when git isn't installed.
    """
    root = git_repo(tmp_path_factory.mktemp("prop6"))
    uv_hash, package_hash = lock_hashes(cwd=root)
    return Prop6Env(root=root, config=_config(root), cc=cloud_config(),
                    uv_hash=uv_hash, package_hash=package_hash, ids=itertools.count())


@settings(deadline=None)
@given(
    preset=st.sampled_from(PRESETS),
    approved=st.booleans(),
    video_bytes=st.binary(min_size=1, max_size=200),
    suffix=st.sampled_from((".mp4", ".mov")),
)
def test_property_6_nothing_paid_before_the_spend_guard_returns(
        prop6_env: Prop6Env, preset: str, approved: bool, video_bytes: bytes,
        suffix: str) -> None:
    """Feature: cloud-jobs, Property 6: Nothing paid happens before the Spend_Guard returns

    **Validates: Requirements 6.1, 6.6, 16.4**

    For any preset, chosen type and inputs, running the Start_Command against a healthy `FakeAws`
    records no `s3_upload` and no `run_instances` before `check_spend` has returned without raising;
    and when the Spend_Guard raises (no `APPROVED_SPEND=1`), the `FakeAws` records neither at all.
    """
    env = prop6_env
    job_id = f"20261119-142530-{next(env.ids):06x}"  # unique per example, so make_job never collides
    video = env.root / f"clip{suffix}"
    video.write_bytes(video_bytes)

    # A healthy FakeAws: the available image's lock-hash tags match HEAD, its Smoke_Marker is
    # present, and the quota admits every preset's primary, so the only gate is the Spend_Guard.
    aws = FakeAws(stack=STACK, images=[_image(env.uv_hash, env.package_hash)],
                  objects={_smoke_marker_key(IMAGE_ID): b"{}"}, quota=32.0,
                  offerings=PROP_OFFERINGS, subnets=SUBNETS)
    environ = {"APPROVED_SPEND": "1"} if approved else {}

    def fake_make_job(video_arg: str | Path, *, start: float | None = None,
                      end: float | None = None, preset: str | None = None,
                      config: Config | None = None) -> str:
        # The stand-in probe and a per-example Job_Id, so make_job needs no ffprobe and no two
        # examples collide on a Job_Folder; everything else is the real make_job (Req 7.1).
        return splatpipe.make.make_job(video_arg, start=start, end=end, preset=preset,
                                       config=config, probe=_fake_probe, new_id=lambda: job_id)

    def guarded_check_spend(**kwargs: Any) -> None:
        # Record a marker the instant the real Spend_Guard returns without raising; when it raises,
        # the SpendRefused propagates and this line is never reached, so no marker is recorded.
        _real_check_spend(**kwargs)
        aws.calls.append((GUARD_MARKER, ""))

    guard_returned = True
    with patch.object(cloud.start, "make_job", fake_make_job), \
            patch.object(splatpipe.run, "STEP_LIST", STEP_ENTRIES), \
            patch.object(cloud.start, "check_spend", guarded_check_spend):
        try:
            start_job(video, preset=preset, aws=aws, config=env.config, cloud_config=env.cc,
                      repo_root=env.root, environ=environ)
        except SpendRefused:
            guard_returned = False

    ops = [op for op, _ in aws.calls]
    if guard_returned:
        # The guard returned, so nothing paid was recorded before its marker (Req 6.1).
        before = ops[:ops.index(GUARD_MARKER)]
        assert "s3_upload" not in before
        assert "run_instances" not in before
    else:
        # The guard raised, so no marker, and nothing paid at all (Req 6.6).
        assert GUARD_MARKER not in ops
        assert "s3_upload" not in ops
        assert "run_instances" not in ops
