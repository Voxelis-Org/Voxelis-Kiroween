"""Reading the trained splat: a binary 3DGS PLY, in either byte order, by property name.

`read_header` reads the PLY header line by line and checks it. `read_splat` reads the rows of
the vertex element into one packed numpy structured array, with one field per property, in the
file's order and with its declared type. A big-endian file is swapped to little-endian in place,
so the array is always little-endian and every value keeps its bits, NaN payloads included. The
step reads meaning into the eleven Required_Properties only, by name, and carries every other
property through as it is (Req 1.2 to 1.4).

The file is only ever opened to read, so the step can't change the trained splat (Req 1.11).
Each refusal is an ExpectedFailure that names the file and says what's wrong with it (Req 1.5 to
1.10, 8.8).

From splatpipe, this module imports only splatpipe.errors.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO

import numpy as np

from splatpipe.errors import ExpectedFailure

# Format_Constants: the PLY scalar types, the 3DGS names and the f_rest_* counts. They describe
# the file format, not our choices, so they stay in code (Req 1.3).
PLY_TYPES = {"char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1",
             "short": "i2", "int16": "i2", "ushort": "u2", "uint16": "u2",
             "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4",
             "float": "f4", "float32": "f4", "double": "f8", "float64": "f8"}
REQUIRED = ("x", "y", "z", "opacity", "scale_0", "scale_1", "scale_2",
            "rot_0", "rot_1", "rot_2", "rot_3")
# f_rest_* count -> SH_Bands: 3 colour channels of 3, 8 or 15 coefficients.
SH_BANDS = {0: 0, 9: 1, 24: 2, 45: 3}

# The two binary formats a format line can name, each with its byte order for numpy. The third
# format, ascii, is refused.
_BYTE_ORDERS = {"binary_little_endian": "<", "binary_big_endian": ">"}


@dataclass(frozen=True)
class PlyProperty:
    """One property of the vertex element: its name, and its type as the header declares it."""

    name: str
    ply_type: str  # as declared, like "float" or "float32"


@dataclass(frozen=True)
class PlyHeader:
    """What a PLY's header says about its vertex element, as read_header read and checked it."""

    path: Path
    byte_order: str  # "<" or ">"
    comments: tuple[str, ...]  # every comment and obj_info line, verbatim, in file order
    count: int  # element vertex <count>
    properties: tuple[PlyProperty, ...]
    data_offset: int  # bytes before the first vertex row
    sh_bands: int

    def dtype(self, byte_order: str) -> np.dtype:
        """One vertex row in `byte_order`, "<" or ">": packed, with one field per property, in
        file order, each with its declared type."""
        return np.dtype([(prop.name, byte_order + PLY_TYPES[prop.ply_type])
                         for prop in self.properties])


@dataclass(eq=False)
class _Element:
    """One element of the header as it's read: its name, its count and its properties so far.
    Two elements are the same only when they're the same object."""

    name: str
    count: int
    properties: list[PlyProperty] = field(default_factory=list)  # its properties that aren't lists
    names: set[str] = field(default_factory=set)  # every property's name, a list's included

    def row_bytes(self) -> int:
        """The size of one row. Only an element with no list property has a fixed one."""
        return sum(np.dtype(PLY_TYPES[prop.ply_type]).itemsize for prop in self.properties)


def read_header(path: Path) -> PlyHeader:
    """The checked header of the PLY at `path`, which is only ever opened to read (Req 1.11).

    The header is read line by line up to end_header and checked as it's read: line 1 is ply,
    line 2 a binary format line, and each line after it a comment, an obj_info, an element, a
    property or end_header. Then the vertex element must hold every Required_Property, a count
    above 0 and 0, 9, 24 or 45 f_rest_* properties, and last the file must hold every vertex
    row. Each refusal is an ExpectedFailure naming the file (Req 1.5 to 1.10).
    """
    try:
        with open(path, "rb") as handle:
            header = _read(path, handle)
        size = os.stat(path).st_size
    except FileNotFoundError as error:
        raise ExpectedFailure(f"The trained splat {path} doesn't exist. The training step "
                              "writes it, so run the training step on this job first, then "
                              "the clean step.") from error
    except OSError as error:  # like a folder in the file's place, or no permission to read it
        raise ExpectedFailure(f"The trained splat {path} can't be read "
                              f"({error.strerror or error}). Check that it's a file you can "
                              "open, then run the clean step again.") from error
    needed = header.data_offset + header.count * header.dtype("<").itemsize
    if size < needed:  # a longer file is fine, because elements after vertex follow its rows
        raise _cut_short(path, f"is {size} bytes, but its header needs at least {needed}")
    return header


def read_splat(header: PlyHeader) -> np.ndarray:
    """The vertex rows of `header`'s file as one packed structured array, always little-endian.

    The rows are read straight into the array. A big-endian file's array is swapped in place and
    viewed as little-endian, so every value keeps its bits, NaN payloads included, with no second
    copy (Req 1.2, 1.4). A file that's shorter than read_header found it is refused (Req 1.10).
    """
    data = np.fromfile(header.path, dtype=header.dtype(header.byte_order), count=header.count,
                       offset=header.data_offset)
    if len(data) < header.count:
        raise _cut_short(header.path, f"ends after {len(data)} of the {header.count} Gaussians "
                                      "its header lists")
    if header.byte_order == ">":
        data.byteswap(inplace=True)
        data = data.view(header.dtype("<"))
    return data


# Reading the header

def _read(path: Path, handle: BinaryIO) -> PlyHeader:
    """Read the header from `handle` up to end_header, line by line, and check what it holds."""
    lines = enumerate(_lines(handle), start=1)
    _, first = next(lines)
    if first != "ply":
        raise ExpectedFailure(f"The trained splat {path} isn't a PLY file, because its first "
                              "line isn't ply. Check that the training step finished and wrote "
                              "a binary PLY there, then run the clean step again.")
    number, line = next(lines)
    byte_order = _byte_order(path, number, line)
    comments: list[str] = []
    elements: list[_Element] = []
    vertex: _Element | None = None  # the first element named vertex, which is the one read
    for number, line in lines:
        words = line.split()
        keyword = words[0] if words else ""
        if keyword in ("comment", "obj_info"):
            comments.append(line)  # verbatim, wherever it sits in the header (Req 4.3)
        elif keyword == "element":
            elements.append(_element(path, number, line, words))
            if vertex is None and elements[-1].name == "vertex":
                vertex = elements[-1]
        elif keyword == "property":
            _add_property(path, number, line, words, elements[-1] if elements else None,
                          vertex)
        elif words == ["end_header"]:
            break
        else:
            raise _bad_line(path, number, line, "isn't a line a PLY header can hold")
    return _checked(path, byte_order, tuple(comments), elements, vertex, handle.tell())


def _lines(handle: BinaryIO) -> Iterator[str]:
    """Each line of `handle` from where it is, as text, with one \\n and then one \\r stripped.

    Latin-1 maps every byte to one character and back, so a comment's bytes come back exactly.
    At the end of the file every line is "", which no header line may be.
    """
    while True:
        yield handle.readline().decode("latin-1").removesuffix("\n").removesuffix("\r")


def _byte_order(path: Path, number: int, line: str) -> str:
    """The byte order the format line names, "<" or ">". ascii is refused (Req 1.2, 1.7)."""
    words = line.split()
    if len(words) == 3 and words[0] == "format" and words[2] == "1.0":
        if words[1] == "ascii":
            raise ExpectedFailure(f"The trained splat {path} is an ascii PLY. The clean step "
                                  "reads only a binary PLY, and splat-transform can't read an "
                                  "ascii PLY either. Have the training step write it as "
                                  "binary_little_endian, then run the clean step again.")
        if words[1] in _BYTE_ORDERS:
            return _BYTE_ORDERS[words[1]]
    raise _bad_line(path, number, line,
                    "isn't a format line, like format binary_little_endian 1.0")


def _element(path: Path, number: int, line: str, words: list[str]) -> _Element:
    """The element that `line`, element <name> <count>, starts (Req 1.7)."""
    if len(words) != 3:
        raise _bad_line(path, number, line, "isn't a line a PLY header can hold")
    if re.fullmatch(r"[0-9]+", words[2]) is None:
        raise _bad_line(path, number, line,
                        "gives a count that isn't a whole number of 0 or more")
    return _Element(words[1], int(words[2]))


def _add_property(path: Path, number: int, line: str, words: list[str],
                  element: _Element | None, vertex: _Element | None) -> None:
    """Add the property that `line` declares to `element`, the last element above it.

    The line is property <type> <name>, or property list <count type> <item type> <name>. A
    list is refused in the vertex element and in any element before it, because a list has no
    fixed size, and kept out of the properties of an element after it (Req 1.7, 1.8).
    """
    is_list = words[1:2] == ["list"]
    if len(words) != (5 if is_list else 3):
        raise _bad_line(path, number, line, "isn't a line a PLY header can hold")
    if element is None:
        raise _bad_line(path, number, line, "declares a property before any element")
    *types, name = (words[2:] if is_list else words[1:])
    unknown = [kind for kind in types if kind not in PLY_TYPES]
    if unknown:
        raise _bad_line(path, number, line, f"gives the type {unknown[0]!r}, which isn't a "
                                            "PLY type")
    if name in element.names:
        raise _bad_line(path, number, line, f"declares the property {name!r} a second time "
                                            f"in the element {element.name!r}")
    element.names.add(name)
    if not is_list:
        element.properties.append(PlyProperty(name, types[0]))
    elif element is vertex:
        raise ExpectedFailure(f"The trained splat {path} has a list property, {name!r}, in its "
                              "vertex element. A list has no fixed size in a row, so the clean "
                              "step can't carry it through. Have the training step write the "
                              "splat without it, then run the clean step again.")
    elif vertex is None:
        raise ExpectedFailure(f"The trained splat {path} has a list property, {name!r}, in the "
                              f"element {element.name!r}, which comes before any vertex "
                              "element. A list has no fixed size, so the clean step can't tell "
                              "where the Gaussians start. Have the training step write its "
                              "vertex element first, then run the clean step again.")


def _checked(path: Path, byte_order: str, comments: tuple[str, ...], elements: list[_Element],
             vertex: _Element | None, header_bytes: int) -> PlyHeader:
    """The PlyHeader of a header read up to end_header, once its vertex element is checked
    (Req 1.5, 1.7, 1.9). The vertex rows start after the header and every element before them.
    """
    if vertex is None:
        raise ExpectedFailure(f"The trained splat {path} has no vertex element, so it holds no "
                              "Gaussians the clean step can read. Check that the training step "
                              "wrote a splat there, then run the clean step again.")
    names = [prop.name for prop in vertex.properties]
    missing = [name for name in REQUIRED if name not in names]
    if missing:
        kind = "property" if len(missing) == 1 else "properties"
        raise ExpectedFailure(f"The trained splat {path} lacks the {kind} "
                              f"{_words(missing, 'and')}, which the clean step needs. A 3DGS "
                              f"splat's vertex element holds {_words(REQUIRED, 'and')}. Check "
                              "that the training step wrote a 3DGS splat, then run the clean "
                              "step again.")
    if vertex.count == 0:
        raise ExpectedFailure(f"The trained splat {path} holds no Gaussians, because its vertex "
                              "count is 0. Run the training step again, then the clean step.")
    rest = sum(1 for name in names if re.fullmatch(r"f_rest_[0-9]+", name))
    if rest not in SH_BANDS:
        allowed = _words([str(count) for count in SH_BANDS], "or")
        raise ExpectedFailure(f"The trained splat {path} has an f_rest_* count of {rest}, but a "
                              f"splat has {allowed} f_rest_* properties, for "
                              f"{min(SH_BANDS.values())} to {max(SH_BANDS.values())} bands of "
                              "spherical harmonics. Check that the training step wrote a "
                              "standard 3DGS splat, then run the clean step again.")
    before = elements[:elements.index(vertex)]
    offset = header_bytes + sum(element.count * element.row_bytes() for element in before)
    return PlyHeader(path, byte_order, comments, vertex.count, tuple(vertex.properties),
                     offset, SH_BANDS[rest])


# The messages

def _bad_line(path: Path, number: int, line: str, problem: str) -> ExpectedFailure:
    """The refusal of header line `number`, `line`, saying what's wrong with it (Req 1.7)."""
    return ExpectedFailure(f"The trained splat {path} can't be read, because line {number} of "
                           f"its header, {line!r}, {problem}. Check that the training step "
                           "finished and wrote a standard binary PLY, then run the clean step "
                           "again.")


def _cut_short(path: Path, detail: str) -> ExpectedFailure:
    """The refusal of a file too short for the vertex rows its header lists (Req 1.10)."""
    return ExpectedFailure(f"The trained splat {path} {detail}, so it's cut short or still "
                           "being written. Let the training step finish, or run it again if it "
                           "stopped, then run the clean step again.")


def _words(items: Sequence[str], last: str) -> str:
    """`items` written out as a list, "a", "a and b" or "a, b and c", with `last` for and."""
    if len(items) == 1:
        return items[0]
    return f"{', '.join(items[:-1])} {last} {items[-1]}"
