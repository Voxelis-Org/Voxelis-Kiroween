"""Writing the Clean_Splat: the Kept_Splats as a binary little-endian PLY.

`write_splat` writes the chosen rows of the Splat_Data to a new file. The file holds only the
`vertex` element, with the trained splat's properties in the same order and with the same
declared types, and every `comment` and `obj_info` line of its header in their order (Req 4.1 to
4.3). Each row goes out as its raw bytes, so every value comes back bit for bit, a NaN in a colour
included (Req 4.4). `read_splat` has already made the Splat_Data little-endian, whatever the
trained splat's byte order was.

From splatpipe, this module imports only splatpipe.errors and splatpipe.clean.ply_read.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from splatpipe.clean.ply_read import PlyHeader, PlyProperty
from splatpipe.errors import ExpectedFailure


def header_bytes(header: PlyHeader, count: int) -> bytes:
    """The Clean_Splat's PLY_Header for `count` Kept_Splats, as bytes.

    The lines are `ply`, the binary_little_endian format line, every comment and obj_info line
    of `header` in their order, `element vertex <count>`, one property line per property in the
    source's order with its declared type, and `end_header`, each ending in a line feed. They're
    encoded as Latin-1, as `read_header` decoded them, so each comment's bytes come back exactly
    (Req 4.1 to 4.3). A comment that sat between property lines in the source now comes before
    the element line, with the comments' order kept.
    """
    lines = ["ply", "format binary_little_endian 1.0", *header.comments,
             f"element vertex {count}", *map(_property_line, header.properties), "end_header"]
    return "".join(f"{line}\n" for line in lines).encode("latin-1")


def write_splat(path: Path, header: PlyHeader, data: np.ndarray, rows: np.ndarray) -> int:
    """Write the rows `rows` of `data` to the new file `path`, and return its size in bytes.

    `data[rows]` is the one copy of the Kept_Splats. The file is opened with mode "xb", so it's
    always a new one: `path` is in the Work_Folder, which the Step_Runner has just emptied. The
    header goes first, then the rows. The size is for the clean_ply_bytes count (Req 5.14).

    An OSError while opening or writing, like a full disk, is an ExpectedFailure naming the file
    and the OS's reason. When the file was made, the partial file is removed first (Req 4.6, 8.8).
    """
    kept = data[rows]  # the one copy of the Kept_Splats
    head = header_bytes(header, len(rows))
    try:
        handle = open(path, "xb")
    except OSError as error:  # nothing was made, so there's nothing to remove
        raise _not_written(path, error) from error
    try:
        with handle:
            handle.write(head)
            kept.tofile(handle)
    except OSError as error:
        path.unlink(missing_ok=True)  # the partial file
        raise _not_written(path, error) from error
    return path.stat().st_size


def _property_line(prop: PlyProperty) -> str:
    """The header line that declares `prop`, with its type as the source declared it."""
    return f"property {prop.ply_type} {prop.name}"


def _not_written(path: Path, error: OSError) -> ExpectedFailure:
    """The Readable_Error for a Clean_Splat that couldn't be written, with the OS's reason.

    numpy's `tofile` raises an OSError with no strerror when a write comes up short, so the
    error's own message stands in for it then, as the Skeleton's messages do.
    """
    return ExpectedFailure(f"The clean splat couldn't be written to {path} "
                           f"({error.strerror or error}). Check that the disk has free space "
                           "and that you can write to that folder, then run the clean step "
                           "again.")
