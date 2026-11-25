"""The COLMAP database's one camera, read with Python's built-in sqlite3, and the Calibration_Copy.

The Calibration runs on a copy of the Database, so the Database stays as the Match_Runs left it for
the Incremental_Run (Req 5.1, 5.2). copy_database makes the Calibration_Copy with sqlite3's backup
API, which copies a consistent database whatever journal files lie beside it (C1). The step calls
it after the Far_Match_Run has exited, so no COLMAP process has the Database open (Req 2.11). A
sqlite3.Error or an OSError, like a full disk, is the Req 11.12 failure, naming the
Calibration_Copy and the reason.

read_camera reads the cameras table of COLMAP 4.2.1's database: camera_id, model, width, height,
params, a blob of little-endian doubles, and prior_focal_length, the flag that says COLMAP trusts
the focal length (F12). The step gives every Pick one camera, so a table with any other number of
rows, or a database that can't be read, fails, naming the file and saying to check that COLMAP
4.2.1 is installed and to run the step again.

Both open an existing database with mode=rw. A plain sqlite3.connect would quietly make an empty
database for a missing file, and a read-only connection would leave the -wal and -shm files of
COLMAP's WAL journal behind, which a read-write connection removes as it closes.

calibration_failure is pure. COLMAP 4.2.1's view_graph_calibrator exits with 0 when it finds no
image pairs and when it rejects its focal length estimate (F11). So after an exit of 0, the step
compares the Calibration_Copy's camera with the Database's, which reads what the Calibration did
to the camera rather than its log text (O5, Req 5.4).

From splatpipe, this module imports only splatpipe.errors.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from splatpipe.errors import ExpectedFailure

# The columns of the cameras table, in DatabaseCamera's order (F12).
_SELECT_CAMERAS = ("SELECT camera_id, model, width, height, params, prior_focal_length "
                   "FROM cameras")
_DOUBLES = np.dtype("<f8")  # the params blob holds one little-endian double per parameter


@dataclass(frozen=True)
class DatabaseCamera:
    """A row of the cameras table of a COLMAP database (F12)."""

    camera_id: int
    model_id: int  # COLMAP's camera model id, like 3 for RADIAL (F2)
    width: int  # in pixels
    height: int  # in pixels
    params: tuple[float, ...]  # the params blob, read as little-endian doubles, f first for RADIAL
    prior_focal_length: bool  # true when COLMAP trusts the focal length


def copy_database(source: Path, target: Path) -> None:
    """Copy the COLMAP database `source` to `target`, the Calibration_Copy, with sqlite3's backup
    API (C1).

    Call it only once no COLMAP process has `source` open (Req 2.11). A sqlite3.Error or an
    OSError is the Req 11.12 failure, naming `target` and the reason.
    """
    try:
        with (closing(_open_existing(source)) as original,
              closing(sqlite3.connect(target)) as copy):
            original.backup(copy)
    except (sqlite3.Error, OSError) as error:
        raise ExpectedFailure(
            f"The COLMAP database couldn't be copied to {target} for the focal length "
            f"calibration ({_reason(error)}). Free some space on the disk, or check that you "
            f"can write to {target.parent}, then run the step again.") from error


def read_camera(path: Path) -> DatabaseCamera:
    """The one camera in the cameras table of the COLMAP database `path`, with its params read as
    little-endian doubles.

    A table with any other number of rows, or a database that can't be read, fails, naming `path`
    and saying to check that COLMAP 4.2.1 is installed and to run the step again.
    """
    try:
        with closing(_open_existing(path)) as database:
            rows = database.execute(_SELECT_CAMERAS).fetchall()
    except sqlite3.Error as error:
        raise _unreadable(path, f"can't be read ({error})") from error
    if len(rows) != 1:
        raise _unreadable(path, f"holds {len(rows)} cameras, but it should hold exactly 1, "
                                "because the step gives every pick the same camera")
    camera_id, model_id, width, height, params, prior_focal_length = rows[0]
    return DatabaseCamera(camera_id=camera_id, model_id=model_id, width=width, height=height,
                          params=tuple(np.frombuffer(params, dtype=_DOUBLES).tolist()),
                          prior_focal_length=bool(prior_focal_length))


def calibration_failure(before: DatabaseCamera, after: DatabaseCamera) -> str | None:
    """Why the Calibration is a Failed_Calibration, or None when it worked (O5, Req 5.4).

    `before` is the Database's camera and `after` the Calibration_Copy's, both read after the
    Calibration exited with 0. With no image pairs, view_graph_calibrator changes no camera, so
    the copy's prior_focal_length is still false. Otherwise it marks the focal length as trusted,
    also when it rejects its estimate and keeps the starting focal length, so a copy whose f is
    bit for bit the Database's rejected it (F11). RADIAL has one focal parameter, f, which the
    Calibration reads and writes (F17). The reason completes the Req 5.5 warning,
    "The focal length calibration <reason>, ...".
    """
    if not after.prior_focal_length:
        return "found no image pairs to calibrate"
    if _focal_bytes(after) == _focal_bytes(before):
        return "rejected its focal length estimate and kept the starting one"
    return None


def _focal_bytes(camera: DatabaseCamera) -> bytes:
    """The bytes of the camera's f, its first param, params[0] (F2, F17), to compare bit for
    bit."""
    return np.array(camera.params[:1], dtype=_DOUBLES).tobytes()


def _open_existing(path: Path) -> sqlite3.Connection:
    """A read-write connection to the database `path`, which fails when `path` doesn't exist."""
    return sqlite3.connect(f"{path.absolute().as_uri()}?mode=rw", uri=True)


def _reason(error: sqlite3.Error | OSError) -> str:
    """The reason the system or SQLite gave, like "No space left on device"."""
    if isinstance(error, OSError) and error.strerror:
        return error.strerror
    return str(error)


def _unreadable(path: Path, what: str) -> ExpectedFailure:
    """The failure for the COLMAP database `path`: `what` says what's wrong, then what to do."""
    return ExpectedFailure(f"The COLMAP database {path} {what}. Check that COLMAP 4.2.1 is "
                           "installed, then run the step again.")
