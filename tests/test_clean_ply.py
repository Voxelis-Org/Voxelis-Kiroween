"""Tests for the clean step's PLY reader and writer, in splatpipe/clean/ply_read.py and
splatpipe/clean/ply_write.py (Req 1.2 to 1.11, 4.1 to 4.3, 4.6).

Every splat here is built as bytes by ply_bytes in tests/splats.py, which shares no code with the
reader or the writer, and written into the test's own tmp_path (Req 10.3).

- Reading: a splat with a property of every PLY type reads back right in either byte order, with
  \\r\\n line ends, and with an element before and one after vertex, and reading leaves the
  file's bytes as they were (Req 1.2 to 1.4, 1.11). The f_rest_* counts 0, 9, 24 and 45 give 0
  to 3 SH_Bands, and any other count is refused (Req 1.5).
- Refusing: a missing file, a folder in its place, and each kind of file the reader can't take,
  each with what's wrong with it (Req 1.6 to 1.10).
- Writing: the carry-through example of Req 10.12, and a write that fills the disk partway
  through (Req 4.1 to 4.3, 4.6).
- The round trip, Property 9: any layout and values, written for any rows, read back bit for bit
  (Req 1.2 to 1.4, 4.1 to 4.4). Hypothesis can't take a function-scoped fixture like tmp_path
  with @given, so this test writes into a tempfile.TemporaryDirectory() it makes itself.

Every refusal names the file and holds no traceback (Req 10.4).
"""

import errno
import os
import re
import tempfile
from pathlib import Path
from typing import BinaryIO, NoReturn

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from splatpipe.clean.ply_read import PLY_TYPES, PlyProperty, read_header, read_splat
from splatpipe.clean.ply_write import write_splat
from splatpipe.errors import ExpectedFailure

from splats import (Layout, gaussians, layouts, ply_bytes, row_subsets, standard_fields,
                    vertex_values)


def put(folder: Path, data: bytes) -> Path:
    """Write `data` to train.ply in `folder`, and give the file's path."""
    path = folder / "train.ply"
    path.write_bytes(data)
    return path


def readable(message: str, path: Path, *shown: str) -> str:
    """The Readable_Error `message` with the name of the file at `path` taken out, so a check of
    its words can't be met by a word in the test's folder name.

    The message must name the file and hold no traceback. It can also echo what the test chose,
    like a header line, so the no-traceback check reads only the words the code wrote itself:
    the message with the file's name and each of `shown` taken out. The longest go first, so a
    short one can't split a longer one.
    """
    assert str(path) in message
    text = message.replace(str(path), "")
    own_words = text
    for value in sorted(shown, key=len, reverse=True):
        own_words = own_words.replace(value, "")
    assert "Traceback" not in own_words
    return text


def refusal(path: Path, *shown: str) -> str:
    """The message of the ExpectedFailure that read_header raises for the file at `path`, as
    readable gives it."""
    with pytest.raises(ExpectedFailure) as caught:
        read_header(path)
    return readable(str(caught.value), path, *shown)


def numbers_in(text: str) -> list[str]:
    """Every whole number written in `text`, like ["46", "0", "9"]."""
    return re.findall(r"[0-9]+", text)


# A small good splat: A1's layout with no normals and no f_rest_*, 14 floats, and two Gaussians.
FIELDS = standard_fields(sh_count=0, normals=False)
GAUSSIANS = gaussians(2, sh_count=0, normals=False)
GOOD = ply_bytes(FIELDS, GAUSSIANS)
ROW_BYTES = GAUSSIANS.dtype.itemsize  # 56


# Reading

# Each PLY type with the numpy type its values read back as, written out here rather than taken
# from ply_read.py's PLY_TYPES.
NUMPY_TYPES = {"char": np.int8, "int8": np.int8, "uchar": np.uint8, "uint8": np.uint8,
               "short": np.int16, "int16": np.int16, "ushort": np.uint16, "uint16": np.uint16,
               "int": np.int32, "int32": np.int32, "uint": np.uint32, "uint32": np.uint32,
               "float": np.float32, "float32": np.float32,
               "double": np.float64, "float64": np.float64}
# A1's layout with no normals and no f_rest_*, then a property of each PLY type, like extra_uchar.
TYPED_FIELDS = [*FIELDS, *((f"extra_{ply_type}", ply_type) for ply_type in NUMPY_TYPES)]


def little_endian(ply_type: str) -> np.dtype:
    """The numpy type that values of `ply_type` read back as, little-endian whatever the file's
    byte order."""
    return np.dtype(NUMPY_TYPES[ply_type]).newbyteorder("<")


def column(index: int, ply_type: str) -> list[int | float]:
    """The four values the property at `index` in TYPED_FIELDS holds: the smallest and the
    largest its type can hold, a zero, which is -0.0 for a float so that its sign shows, and its
    own index, so no two properties hold the same four."""
    numpy_type = NUMPY_TYPES[ply_type]
    if np.issubdtype(numpy_type, np.integer):
        whole = np.iinfo(numpy_type)
        return [int(whole.min), int(whole.max), 0, index]
    real = np.finfo(numpy_type)
    return [float(real.min), float(real.max), -0.0, float(index)]


TYPED_COLUMNS = [column(index, ply_type) for index, (_, ply_type) in enumerate(TYPED_FIELDS)]
TYPED_ROWS = [list(row) for row in zip(*TYPED_COLUMNS)]  # the same values, a row per Gaussian
# An element before vertex, whose 5-byte rows the reader skips, and one after it, which it never
# reads.
AROUND = {"before": [("meta", [("a", "uchar"), ("b", "int")], [(1, -2), (3, -4)])],
          "after": [("tail", [("c", "double")], [(1.5,)])]}


@pytest.mark.parametrize("around", [{}, AROUND], ids=["vertex alone", "elements around vertex"])
@pytest.mark.parametrize("eol", [b"\n", b"\r\n"], ids=["LF", "CRLF"])
@pytest.mark.parametrize("byte_order", ["<", ">"], ids=["little-endian", "big-endian"])
def test_every_ply_type_reads_back_right(tmp_path: Path, byte_order: str, eol: bytes,
                                         around: dict[str, object]) -> None:
    assert NUMPY_TYPES.keys() == PLY_TYPES.keys()  # so the layout has every type the reader takes
    written = ply_bytes(TYPED_FIELDS, TYPED_ROWS, byte_order=byte_order,
                        comments=["made by a test"], eol=eol, **around)
    path = put(tmp_path, written)
    header = read_header(path)
    data = read_splat(header)
    assert header.byte_order == byte_order
    assert header.comments == ("comment made by a test",)
    assert header.count == len(TYPED_ROWS)
    assert header.properties == tuple(PlyProperty(name, ply_type)
                                      for name, ply_type in TYPED_FIELDS)
    assert header.sh_bands == 0
    # One packed, little-endian field per property, in the file's order, with every value's bits.
    assert data.dtype == np.dtype([(name, little_endian(ply_type))
                                   for name, ply_type in TYPED_FIELDS])
    for (name, ply_type), values in zip(TYPED_FIELDS, TYPED_COLUMNS):
        assert data[name].tobytes() == np.array(values, little_endian(ply_type)).tobytes(), name
    assert path.read_bytes() == written


@pytest.mark.parametrize(("count", "bands"), [(0, 0), (9, 1), (24, 2), (45, 3)])
def test_an_f_rest_count_of_0_9_24_or_45_gives_0_to_3_sh_bands(tmp_path: Path, count: int,
                                                               bands: int) -> None:
    path = put(tmp_path, ply_bytes(standard_fields(sh_count=count), gaussians(2, sh_count=count)))
    assert read_header(path).sh_bands == bands


@pytest.mark.parametrize("count", [1, 8, 44, 46])
def test_any_other_f_rest_count_is_refused_naming_it(tmp_path: Path, count: int) -> None:
    path = put(tmp_path, ply_bytes(standard_fields(sh_count=count), gaussians(2, sh_count=count)))
    numbers = numbers_in(refusal(path))
    assert str(count) in numbers
    assert {"0", "9", "24", "45"} <= set(numbers)  # and the four counts a splat can have


# Refusing

def test_a_missing_file_is_refused_naming_it_and_the_training_step(tmp_path: Path) -> None:
    text = refusal(tmp_path / "train.ply")
    assert "doesn't exist" in text and "training step writes it" in text


def test_a_folder_in_the_files_place_is_refused_with_the_os_reason(tmp_path: Path) -> None:
    path = tmp_path / "train.ply"
    path.mkdir()
    with pytest.raises(OSError) as opening:  # what the OS says when a folder is opened as a file
        open(path, "rb")
    reason = opening.value.strerror
    assert reason in refusal(path, reason)


def changed(line: str, new: str) -> bytes:
    """GOOD with its header line `line` changed to `new`, which can be more than one line."""
    return GOOD.replace(f"{line}\n".encode(), f"{new}\n".encode(), 1)


# The number of GOOD's header line "property float y", after ply, the format line, the vertex
# element's line and x. Each bad line below takes its place.
Y_LINE = GOOD.split(b"\n").index(b"property float y") + 1

# Files the reader refuses, each with the words its refusal must hold besides the file's name,
# and what in those words the test chose, which the refusal echoes.
BAD_SPLATS = [
    # The start of a zip, like a .sog in the trained splat's place.
    pytest.param(b"PK\x03\x04" + bytes(26), ["isn't a PLY"], [], id="not a PLY"),
    pytest.param(changed("format binary_little_endian 1.0", "format ascii 1.0"),
                 ["an ascii PLY", "splat-transform can't read an ascii PLY either"], [],
                 id="ascii"),
    pytest.param(changed("element vertex 2", "element points 2"), ["no vertex element"], [],
                 id="no vertex"),
    pytest.param(changed("property float y", "proprety float y"),
                 [f"line {Y_LINE}", "proprety float y", "isn't a line a PLY header can hold"],
                 ["proprety float y"], id="a bad header line"),
    pytest.param(changed("property float y", "property float16 y"),
                 [f"line {Y_LINE}", "property float16 y", "isn't a PLY type"],
                 ["property float16 y"], id="an unknown type"),
    pytest.param(changed("property float y", "property float x"),
                 [f"line {Y_LINE}", "property float x", "a second time"],
                 ["property float x"], id="a property listed twice"),
    pytest.param(ply_bytes([field for field in FIELDS if field[0] != "scale_2"], GAUSSIANS),
                 ["lacks the property scale_2"], [], id="no scale_2, as in a 2DGS splat"),
    pytest.param(changed("property float opacity", "property float opacity_linear"),
                 ["lacks the property opacity"], [], id="opacity_linear in place of opacity"),
    pytest.param(changed("property float rot_3",
                         "property float rot_3\nproperty list uchar int track_ids"),
                 ["list property", "track_ids"], ["track_ids"], id="a list property in vertex"),
    pytest.param(ply_bytes(FIELDS, []), ["holds no Gaussians"], [], id="a count of 0"),
]


@pytest.mark.parametrize(("data", "words", "shown"), BAD_SPLATS)
def test_a_file_the_reader_cant_take_is_refused_naming_the_file_and_the_problem(
        tmp_path: Path, data: bytes, words: list[str], shown: list[str]) -> None:
    text = refusal(put(tmp_path, data), *shown)
    for word in words:
        assert word in text


@pytest.mark.parametrize("cut", [1, ROW_BYTES], ids=["one byte short", "one row short"])
def test_a_file_one_byte_or_one_row_short_is_refused_as_cut_short(tmp_path: Path,
                                                                  cut: int) -> None:
    short = GOOD[:-cut]
    path = put(tmp_path, short)
    text = refusal(path)
    assert "cut short or still being written" in text
    assert {str(len(short)), str(len(GOOD))} <= set(numbers_in(text))  # the bytes it has and needs
    # A file that shrinks after read_header has checked its size, as when the training step
    # writes it again, is refused when its rows are read.
    path.write_bytes(GOOD)
    header = read_header(path)
    path.write_bytes(short)
    with pytest.raises(ExpectedFailure) as caught:
        read_splat(header)
    assert "cut short or still being written" in readable(str(caught.value), path)


# Writing

# The comments of the carry-through example. A viewer reads them to tell how the scene has to be
# rendered, so the clean splat keeps them (Req 4.3).
CARRIED_COMMENTS = ("SplatRenderMode: mip", "postshot.anti_aliasing=1")


def test_normals_a_double_property_and_the_comments_are_carried_through(tmp_path: Path) -> None:
    # A1's layout with normals, then a double that a producer added, holding values that a float
    # can't.
    fields = [*standard_fields(), ("confidence", "double")]
    plain = gaussians(6)
    source = np.zeros(len(plain), [*((name, "<f4") for name in plain.dtype.names),
                                   ("confidence", "<f8")])
    for name in plain.dtype.names:
        source[name] = plain[name]
    for k, name in enumerate(("nx", "ny", "nz"), start=1):
        source[name] = 10 * k + np.arange(len(source))  # normals that aren't 0, unlike training's
    source["confidence"] = np.arange(len(source)) / 3
    header = read_header(put(tmp_path, ply_bytes(fields, source, comments=CARRIED_COMMENTS)))
    rows = np.array([0, 2, 3, 5])  # a subset of the six, in order
    clean = tmp_path / "clean.ply"
    size = write_splat(clean, header, read_splat(header), rows)
    back = read_header(clean)
    assert size == clean.stat().st_size
    assert back.byte_order == "<" and back.count == len(rows)
    assert back.properties == tuple(PlyProperty(name, ply_type) for name, ply_type in fields)
    assert back.comments == tuple(f"comment {text}" for text in CARRIED_COMMENTS)
    values = read_splat(back)
    for name, _ in fields:
        assert values[name].tobytes() == source[name][rows].tobytes(), name


class FullDisk(np.ndarray):
    """Splat_Data whose rows fill the disk partway through: tofile writes the first half of
    their bytes to the file, then raises the OSError a full disk gives."""

    def tofile(self, fid: BinaryIO, *args: object, **kwargs: object) -> NoReturn:
        fid.write(self.tobytes()[:self.nbytes // 2])
        raise OSError(errno.ENOSPC, os.strerror(errno.ENOSPC))


def test_a_write_that_fills_the_disk_names_the_file_and_the_reason_and_leaves_no_partial_file(
        tmp_path: Path) -> None:
    header = read_header(put(tmp_path, GOOD))
    work = tmp_path / "work"  # a folder of its own, like the Work_Folder the step writes into
    work.mkdir()
    clean = work / "clean.ply"
    with pytest.raises(ExpectedFailure) as caught:
        write_splat(clean, header, read_splat(header).view(FullDisk), np.arange(header.count))
    reason = os.strerror(errno.ENOSPC)
    assert reason in readable(str(caught.value), clean, reason)
    assert list(work.iterdir()) == []  # no partial clean.ply, and nothing else


# Property tests

def bits(values: np.ndarray) -> np.ndarray:
    """One field's values as little-endian unsigned whole numbers of the same size, so == compares
    their raw bytes: a NaN's payload, an infinity and -0.0 each compare as their own bits."""
    return values.view(f"<u{values.dtype.itemsize}")


@settings(deadline=None)
@given(layouts(), st.data())
def test_a_clean_splat_reads_back_bit_for_bit(layout: Layout, data: st.DataObject) -> None:
    """Feature: clean, Property 9: A Clean_Splat reads back bit for bit

    **Validates: Requirements 1.2, 1.3, 1.4, 4.1, 4.2, 4.3, 4.4**
    """
    source = data.draw(vertex_values(layout.fields), label="values")
    rows = data.draw(row_subsets(len(source)), label="rows")
    written = ply_bytes(layout.fields, source, byte_order=layout.byte_order,
                        comments=layout.comments, before=layout.before, after=layout.after)
    with tempfile.TemporaryDirectory() as folder:
        header = read_header(put(Path(folder), written))
        clean = Path(folder) / "clean.ply"  # a new file, because write_splat opens it with "xb"
        write_splat(clean, header, read_splat(header), rows)
        back = read_header(clean)
        kept = read_splat(back)
    assert back.byte_order == "<"  # binary_little_endian, whatever the source's byte order
    assert back.count == len(rows)
    assert back.properties == tuple(PlyProperty(name, ply_type)
                                    for name, ply_type in layout.fields)
    assert back.comments == tuple(f"comment {text}" for text in layout.comments)
    assert kept.dtype == source.dtype  # the same fields, in the same order, little-endian
    for name in source.dtype.names:
        assert bits(kept[name]).tolist() == bits(source[name][rows]).tolist(), name
