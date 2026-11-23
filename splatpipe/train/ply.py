"""The Splat_Layout of the Trained_Splat, with our own PLY_Writer, PLY_Reader and PLY_Check.

The Splat_Layout is the standard 3DGS PLY that the splat-files skill describes, the layout that
gsplat's export_splats(format="ply") writes. The PLY_Writer is our own numpy code and never calls
export_splats() (Req 8.2). The header is ASCII, each line ending in a newline, with nothing else
in it: no comment, no other element and no normals (Req 8.3).

    ply
    format binary_little_endian 1.0
    element vertex <N>
    property float <name>     one line for each field of FIELDS, in its order
    end_header

Then N records of RECORD_BYTES, 236 bytes: the 59 fields of FIELDS as little-endian float32.

    x y z                     the means, in the Levelled_Frame
    f_dc_0 to f_dc_2          sh0, band 0 of R, G and B
    f_rest_0 to f_rest_44     sh_rest, channel-major: f_rest_k is channel k // 15's coefficient
                              k % 15, band by band, so R's 15, then G's, then B's (Req 8.4)
    opacity                   the logit, before the sigmoid
    scale_0 to scale_2        the log of each axis' size
    rot_0 to rot_3            the quaternion as the Trainer holds it, w first (Req 8.5)

splat_records turns Splats into these records. A float32 value is copied bit for bit. A float64
one, like the baked colours, is rounded to float32 inside np.errstate(over="ignore"), so a value
too big for float32 becomes an infinity, which the PLY_Check then reports. In the step,
drop_non_finite has already left out every splat that would get one (Req 8.9). write_ply writes
every splat it's given, 0 included, and turns an OSError into an ExpectedFailure that names the
file and the reason (Req 8.10).

read_ply and check_ply share one header parse. It reads only the first
len(splat_header(max(1, size // RECORD_BYTES))) bytes, the longest header a file of this size can
have, so a broken file never makes it scan the body for end_header. It checks every line: ply, the
format line, element vertex N, a property float line for each field, with the names FIELDS in
order, then end_header. Any other line is named with its number. Then the file's size must be the
header's plus N × RECORD_BYTES. read_ply gives one float32 array per field, for 0 splats too
(Req 8.13). check_ply also refuses 0 splats and any value that isn't finite, and gives back N
(Req 8.6, 8.7). Each refusal is an ExpectedFailure that names the file, with any text from the
file shown with repr().

From splatpipe, this module imports only splatpipe.errors when it runs. The Splats type that
splat_records and write_ply take is imported for type hints only.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from splatpipe.errors import ExpectedFailure

if TYPE_CHECKING:  # the type splat_records and write_ply take, named in type hints only
    from splatpipe.train.splats import Splats

# Fixed_Constants of the Splat_Layout: its 59 field names in order, each a float32, with 45 f_rest
# fields for the 15 coefficients above band 0 of each colour channel at SH_Degree 3.
_REST_FIELDS = 45
FIELDS: tuple[str, ...] = ("x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2",
                           *(f"f_rest_{k}" for k in range(_REST_FIELDS)), "opacity",
                           "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3")
RECORD_BYTES = 4 * len(FIELDS)  # 236: 59 float32 values

_OPENING = ("ply", "format binary_little_endian 1.0")  # the header's lines 1 and 2
_ELEMENT = re.compile(r"element vertex (0|[1-9][0-9]*)")  # line 3, checked with fullmatch
_PROPERTY = re.compile(r"property float (\S+)")  # each field's line, checked with fullmatch
_END = "end_header"
_AGAIN = "Run the step again, and tell the team if it fails the same way."


def splat_header(count: int) -> bytes:
    """The Splat_Layout's header for `count` splats, as ASCII bytes (Req 8.3).

    Its lines are ply, the format line, element vertex `count`, one property float line for each
    field of FIELDS in order, and end_header, each ending in a newline. That's 1,472 bytes for 1
    splat and 1,478 for 3,000,000.
    """
    lines = (*_OPENING, f"element vertex {count}",
             *(f"property float {name}" for name in FIELDS), _END)
    return "".join(f"{line}\n" for line in lines).encode("ascii")


def splat_records(splats: Splats) -> np.ndarray:
    """The splats as the Splat_Layout's records: an (N, 59) array of "<f4", in FIELDS' order.

    One np.empty array is filled by columns. sh_rest, coefficient first and channel last, is
    transposed to channel first, so f_rest_k holds channel k // 15's coefficient k % 15 (Req 8.4).
    Opacities stay logits, scales logs and quaternions w first, as the Trainer holds them
    (Req 8.5). A float32 array is copied bit for bit. A float64 one is rounded to float32 inside
    np.errstate(over="ignore"), so a value too big for float32 becomes an infinity, which the
    PLY_Check reports.
    """
    count = len(splats)
    records = np.empty((count, len(FIELDS)), dtype="<f4")
    with np.errstate(over="ignore"):
        records[:, 0:3] = splats.means  # x, y, z
        records[:, 3:6] = splats.sh0  # f_dc_0 to f_dc_2
        # f_rest_0 to f_rest_44: (N, 15, 3) to (N, 3, 15), then each splat's 45 values in a row
        records[:, 6:51] = splats.sh_rest.transpose(0, 2, 1).reshape(count, _REST_FIELDS)
        records[:, 51] = splats.opacity_logits  # opacity
        records[:, 52:55] = splats.log_scales  # scale_0 to scale_2
        records[:, 55:59] = splats.quats  # rot_0 to rot_3
    return records


def write_ply(path: Path, splats: Splats) -> None:
    """Write the splats to `path` in the Splat_Layout: the header, then the records (Req 8.2–8.5).

    Every splat given is written, 0 included. The records go out through a memoryview of their own
    buffer, with no extra copy of the up to 708 MB. An OSError, like a full disk, becomes an
    ExpectedFailure that names the file and the reason (Req 8.10).
    """
    records = splat_records(splats)
    try:
        with path.open("wb") as file:
            file.write(splat_header(len(records)))
            file.write(memoryview(records))
    except OSError as error:
        raise ExpectedFailure(f"The trained splat {path} couldn't be written "
                              f"({error.strerror or error}). Free some space on the disk, or "
                              f"check that you can write to {path.parent}, then run the step "
                              "again.") from error


def read_ply(path: Path) -> dict[str, np.ndarray]:
    """The values of the Splat_Layout file at `path`: one contiguous float32 array of its N values
    for each field of FIELDS, by name, bit for bit as the file holds them (Req 8.13).

    A file of 0 splats gives 59 arrays of length 0. A header or a size other than the layout's is
    refused as check_ply refuses it, but the values aren't checked.
    """
    count, header_length = _layout(path)
    body = _body(path, count, header_length)
    return {name: np.ascontiguousarray(body[:, column]) for column, name in enumerate(FIELDS)}


def check_ply(path: Path) -> int:
    """The PLY_Check: N, the number of splats in the file at `path`, once the file passes.

    It passes a file whose header is exactly the Splat_Layout's, with an N of 1 or more, whose
    size is the header's plus N × RECORD_BYTES and whose every value is finite (Req 8.6). Any other
    file is refused with an ExpectedFailure that names the file and what failed (Req 8.7): no
    splats, a header line other than the layout's, the fields missing, out of place or not in the
    layout, a size that doesn't match N, or how many splats hold a NaN or an infinity, with each
    field that holds one and its count.
    """
    count, header_length = _layout(path)
    if count == 0:
        raise ExpectedFailure(f"{path} holds no splats. {_AGAIN}")
    non_finite = ~np.isfinite(_body(path, count, header_length))
    if non_finite.any():
        raise _non_finite(path, non_finite)
    return count


def _layout(path: Path) -> tuple[int, int]:
    """The splat count N and the header's length in bytes of the file at `path`, once its header
    is exactly the Splat_Layout's and its size is the header's plus N × RECORD_BYTES.

    Only the first len(splat_header(max(1, size // RECORD_BYTES))) bytes are read, the longest
    header a file of this size can have. The lines are checked in order, so the first one that
    differs from the layout's is refused with its number, and the names are checked once
    end_header is reached.
    """
    size = path.stat().st_size
    with path.open("rb") as file:
        prefix = file.read(len(splat_header(max(1, size // RECORD_BYTES))))
    *lines, rest = prefix.split(b"\n")  # `rest` follows the last newline: a line that doesn't end
    count, length = 0, 0  # N, and the length of the lines so far
    names: list[str] = []
    for number, raw in enumerate(lines, start=1):
        line = _ascii(path, number, raw)
        length += len(raw) + 1
        if number in (1, 2):
            _expect(path, number, line, _OPENING[number - 1])
        elif number == 3:
            count = _vertex_count(path, number, line)
        elif line == _END:
            _check_names(path, names)
            _check_size(path, size, length, count)
            return count, length
        else:
            names.append(_property_name(path, number, line))
    number = len(lines) + 1  # the first line that doesn't end within the bytes read
    raise _not_layout(path, _no_end(size, len(prefix), number, _ascii(path, number, rest)))


def _ascii(path: Path, number: int, raw: bytes) -> str:
    """Header line `number` as text, refused when it isn't ASCII."""
    if not raw.isascii():
        raise _not_layout(path, f"its line {number} isn't ASCII text")
    return raw.decode("ascii")


def _expect(path: Path, number: int, line: str, layout: str) -> None:
    """Refuse header line `number` when it isn't the layout's line `layout`."""
    if line != layout:
        raise _not_layout(path, f"its line {number} is {line!r}, but the layout's line {number} "
                                f"is {layout!r}")


def _vertex_count(path: Path, number: int, line: str) -> int:
    """N, from the element line, refused when it isn't element vertex N in digits."""
    match = _ELEMENT.fullmatch(line)
    if match is None:
        raise _not_layout(path, f"its line {number} is {line!r}, but the layout's line {number} "
                                "is 'element vertex <N>', with N the number of splats")
    return int(match[1])


def _property_name(path: Path, number: int, line: str) -> str:
    """The field name of a property float line. Any other line, like a comment, a second element
    or property double x, is refused with its number."""
    match = _PROPERTY.fullmatch(line)
    if match is None:
        raise _not_layout(path, f"its line {number} is {line!r}, but after its element line the "
                                "layout has only 'property float <name>' lines, then "
                                f"{_END!r}")
    return match[1]


def _check_names(path: Path, names: list[str]) -> None:
    """Refuse field names other than FIELDS in order, listing the fields missing, the fields out
    of place and any name the layout doesn't have.

    A field is out of place when, among the layout's fields that the file has, it isn't at the
    place the layout's order gives it. So two fields swapped are both named, a field left out
    makes no other field out of place, and a field named twice is out of place.
    """
    if tuple(names) == FIELDS:
        return
    found, known = set(names), set(FIELDS)
    in_file = [name for name in names if name in known]  # the layout's fields, in the file's order
    in_layout = [name for name in FIELDS if name in found]  # the same fields, in the layout's order
    missing = [name for name in FIELDS if name not in found]
    out_of_place = [name for place, name in enumerate(in_file)
                    if place >= len(in_layout) or name != in_layout[place]]
    unknown = [repr(name) for name in names if name not in known]
    problems = (("missing", missing), ("out of place", out_of_place),
                ("not in the layout", unknown))
    listed = "; ".join(f"{label}: {', '.join(fields)}" for label, fields in problems if fields)
    raise _not_layout(path, f"its fields aren't the layout's {len(FIELDS)} in order ({listed})")


def _check_size(path: Path, size: int, header_length: int, count: int) -> None:
    """Refuse a file whose size isn't its header's plus `count` records."""
    need = header_length + count * RECORD_BYTES
    if size != need:
        raise ExpectedFailure(f"{path} is {size} bytes, but its header is {header_length} bytes "
                              f"and says it holds {_splats(count)}, so it must be {need} bytes, "
                              f"{RECORD_BYTES} for each splat after the header. {_AGAIN}")


def _no_end(size: int, read: int, number: int, start: str) -> str:
    """Why a header has no end_header line within the `read` bytes read of a file of `size`
    bytes. Line `number`, which starts with `start`, is the first line that doesn't end within
    them."""
    if read == size:
        return f"the file ends after {size} bytes, before its header has an {_END} line"
    starts = f", which starts {start!r}," if start else ""
    return (f"it has no {_END} line in its first {read} bytes, the longest header a file of "
            f"{size} bytes can have, and its line {number}{starts} doesn't end within them")


def _not_layout(path: Path, detail: str) -> ExpectedFailure:
    """The failure for a header other than the Splat_Layout's, saying what `detail` found."""
    return ExpectedFailure(f"{path} isn't in the splat layout the train step writes: {detail}. "
                           f"{_AGAIN}")


def _body(path: Path, count: int, header_length: int) -> np.ndarray:
    """The file's `count` records after its header, as an (N, 59) array of "<f4"."""
    values = np.fromfile(path, dtype="<f4", count=count * len(FIELDS), offset=header_length)
    return values.reshape(count, len(FIELDS))


def _non_finite(path: Path, non_finite: np.ndarray) -> ExpectedFailure:
    """The failure for records holding a NaN or an infinity: how many splats hold one, and each
    field that holds one with its count, in FIELDS' order, like "opacity (2), scale_1 (1)".
    `non_finite` is the (N, 59) mask of the values that aren't finite."""
    holding = int(np.count_nonzero(non_finite.any(axis=1)))  # the splats that hold one
    counts = np.count_nonzero(non_finite, axis=0)  # each field's count
    fields = ", ".join(f"{name} ({int(found)})" for name, found in zip(FIELDS, counts) if found)
    verb = "holds" if holding == 1 else "hold"
    return ExpectedFailure(f"In {path}, {_splats(holding)} {verb} a NaN or an infinity: "
                           f"{fields}. {_AGAIN}")


def _splats(count: int) -> str:
    """A number of splats in words, like "1 splat" or "3 splats"."""
    return f"{count} splat" if count == 1 else f"{count} splats"
