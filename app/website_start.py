"""The Start_Precheck and the Website_Caller: starting a Job from an Upload through the same
Start_Core as `cloud.start`.

The order: the Stop_Time, the Preset and the Account's own unused Upload; the Step_List and the
Baked_Commit's gate (image, lock hashes, Smoke_Marker); the Video_Check over range reads; the
Job_Record; the machine choice; the Reservation; then `launch_job`, which calls the Spend_Guard
first, then copies the video inside S3, uploads code.tar and job.json, reads the clock again for the
Stop_Time, and launches. Each outcome decides the Reservation's fate, and nothing retries.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from splatpipe.config import check_preset
from splatpipe.errors import PipelineError
from splatpipe.jobs import JobRecord, new_job_id, write_record

from app.errors import BAD_REQUEST, CONFLICT, UNAVAILABLE, PublicRefusal
from app.limits import (Reservation, check_room, give_back, is_stopped, job_key, stopped, take,
                        worst_case_cents)
from app.uploads import read_upload
from app.video import check_video
from cloud.aws import aws_setting
from cloud.errors import CapacityExhausted, CloudError, NoGpuQuota, QuotaFull, QuotaRefused
from cloud.redact import account_redaction
from cloud.spend import SpendRefused
from cloud.start import check_image, choose_machine, launch_job, require_step_list
from cloud.table import Set

if TYPE_CHECKING:
    from app.sessions import Session
    from app.web import Deps
    from cloud.start import Choice, Gate

BUSY = "All GPU machines are busy right now, so your job didn't start. Try again later."
CANT_START = "The website can't start jobs right now."
COPY_FAILED = "The video couldn't be copied for the job, so it didn't start. Press Start again."
UNKNOWN = "Your job may or may not have started. Its status below will show which."
UNUSABLE_UPLOAD = "That upload can't start a job. Upload the video again."
_JOB_ID_RANDOM_BYTES = 3


@dataclass(frozen=True)
class Started:
    job_id: str
    outcome: str                  # "launched", or "unknown" when a machine may have started


def busy() -> PublicRefusal:
    return PublicRefusal(CONFLICT, "busy", BUSY)


def cant_start(deps: Deps, why: Exception) -> PublicRefusal:
    deps.log.warning("can't start: %s: %s", type(why).__name__, account_redaction(str(why)))
    return PublicRefusal(UNAVAILABLE, "cant_start", CANT_START)


def precheck(deps: Deps, session: Session, preset: object, now: datetime) -> None:
    """Every free check of a start for `preset`, taking no Reservation and starting nothing."""
    chosen = _preset(deps, preset, now)
    check_room(deps.aws, worst_case_cents(chosen, deps.cloud, deps.website), now, deps.website)
    _gate_and_choice(deps, chosen)


def start(deps: Deps, session: Session, upload_id: object, preset: object,
          now: datetime) -> Started:
    chosen = _preset(deps, preset, now)
    upload = read_upload(deps, session.account, upload_id) if isinstance(upload_id, str) else None
    if upload is None or upload.state != "complete" or upload.used_by is not None:
        raise PublicRefusal(BAD_REQUEST, "bad_request", UNUSABLE_UPLOAD)
    entries, gate = _gate(deps)
    aws, wc = deps.aws, deps.website
    facts = check_video(lambda offset, length: aws.s3_read_range(upload.key, offset, length),
                        aws.s3_size(upload.key), upload.size, upload.name, wc, deps.config.jobs())
    job_id = new_job_id(now, deps.random.token_bytes(_JOB_ID_RANDOM_BYTES).hex())
    record = JobRecord(job_id=job_id, preset=chosen, start_seconds=facts.start_seconds,
                       end_seconds=facts.end_seconds, source_name=upload.name,
                       source_bytes=facts.size, video_seconds=facts.video_seconds,
                       video_file=facts.video_file)
    choice = _choice(deps, gate, entries, chosen)
    draft = Reservation(job_id=job_id, account=session.account, upload_id=upload.id,
                        preset=chosen, cents=worst_case_cents(chosen, deps.cloud, wc))
    reservation = take(aws, draft, len(entries), deps.clock, wc)
    return _launch(deps, gate, choice, record, upload.key, reservation)


def _launch(deps: Deps, gate: Gate, choice: Choice, record: JobRecord, upload_key: str,
            reservation: Reservation) -> Started:
    aws, wc = deps.aws, deps.website
    placed = False
    environ = {aws_setting(deps.config, "spend_variable"): "1"}   # built here, for this one call
    with tempfile.TemporaryDirectory(prefix="voxelis-website-") as tmp:
        folder = Path(tmp)
        (folder / "input").mkdir()
        write_record(folder, record)

        def place_inputs(prefix: str, code_key: str) -> None:
            nonlocal placed
            aws.s3_copy(upload_key, f"{prefix}/{record.video_file}", part_bytes=wc.copy_part_bytes)
            aws.s3_upload(deps.baked.archive, code_key)
            aws.s3_upload(folder / "input" / "job.json", f"{prefix}/job.json")
            if is_stopped(deps.clock(), wc):
                raise stopped(wc)
            placed = True

        try:
            launched = launch_job(aws, gate, choice, record.job_id, deps.baked.code(), place_inputs,
                                  deps.config, deps.cloud, environ)
        except SpendRefused as error:
            give_back(aws, reservation)
            raise cant_start(deps, error) from None
        except PublicRefusal:
            give_back(aws, reservation)
            raise
        except (CapacityExhausted, QuotaRefused):
            give_back(aws, reservation)
            raise busy() from None
        except CloudError as error:
            if not placed:
                give_back(aws, reservation)
                deps.log.warning("copy failed: %s", account_redaction(str(error)))
                raise PublicRefusal(CONFLICT, "copy_failed", COPY_FAILED) from None
            _mark(deps, reservation, {"launch": "unknown"})
            deps.log.warning("launch unknown: %s", account_redaction(str(error)))
            return Started(record.job_id, "unknown")
        except Exception:
            if not placed:
                give_back(aws, reservation)
            else:
                _mark(deps, reservation, {"launch": "unknown"})
            raise
    _mark(deps, reservation, {"launch": "launched", "instance_type": launched.instance_type,
                              "zone": launched.zone})
    return Started(record.job_id, "launched")


def _preset(deps: Deps, preset: object, now: datetime) -> str:
    if is_stopped(now, deps.website):
        raise stopped(deps.website)
    try:
        return check_preset(preset)
    except PipelineError as error:
        raise PublicRefusal(BAD_REQUEST, "bad_request", str(error)) from None


def _gate(deps: Deps) -> tuple[tuple, Gate]:
    try:
        entries = require_step_list()
        return entries, check_image(deps.aws, deps.config, deps.baked.code())
    except (PipelineError, CloudError) as error:
        raise cant_start(deps, error) from None


def _choice(deps: Deps, gate: Gate, entries: tuple, preset: str) -> Choice:
    try:
        choice = choose_machine(deps.aws, gate, entries, preset, deps.config, deps.cloud)
    except QuotaFull:
        raise busy() from None
    except (NoGpuQuota, PipelineError, CloudError) as error:
        raise cant_start(deps, error) from None
    if not choice.attempts:
        raise cant_start(deps, CloudError("the launch plan is empty"))
    return choice


def _gate_and_choice(deps: Deps, preset: str) -> None:
    entries, gate = _gate(deps)
    _choice(deps, gate, entries, preset)


def _mark(deps: Deps, reservation: Reservation, values: dict[str, str]) -> None:
    try:
        deps.aws.table_update(job_key(reservation.account, reservation.job_id),
                              tuple(Set(attr, value) for attr, value in values.items()))
    except Exception as error:  # the Reservation stays held either way
        deps.log.warning("couldn't record the launch: %s", type(error).__name__)
