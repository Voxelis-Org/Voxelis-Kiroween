"""Tests for splatpipe/train/ply.py: the PLY_Writer, the PLY_Reader and the PLY_Check.

The Splat_Layout's header lines and its 236-byte record are written here from Req 8.3, not taken
from ply.py, and the header tests write their own files: the layout's lines for 2 splats with one
change, then 2 records of finite float32 values. Every other unit test's file is written by
write_ply from tiny float32 splats, into the test's tmp_path. The full-disk test patches Path.open
to raise ENOSPC, and the overflow test gives write_ply a float64 colour too big for float32, as
the Exposure_Bake can.

Each refusal is an ExpectedFailure whose message names the file and what failed, and holds no
traceback (Req 8.7, 8.10, 14.6).

Property 6 draws 1 to 20 float32 splats with Hypothesis, every value finite, with −0.0, the
smallest and largest subnormals and ±3.4e38 drawn often beside Hypothesis's own finite float32s.
write_ply writes each example's file into its own tempfile.TemporaryDirectory(), since tmp_path is
shared by all of a test's examples, and check_ply must pass it and give back the count.

Property 7 takes the same splats and draws a record r, a field f and one of NaN, +inf and −inf.
The test replaces the four bytes at the header's length + 236 r + 4 f of the written file with
that value, and check_ply must refuse the file, saying 1 splat holds a NaN or an infinity and
naming field f with a count of 1.

Property 8 takes the same splats and draws a field f. The test writes the file itself: the lines
of splat_header without field f's property line, then the records of splat_records with field f's
4 bytes taken out of each by np.delete. check_ply must refuse the file, naming field f as the one
field missing.

Property 9 draws 0 to 20 splats whose every value is a float32 drawn as its 32 bits: any pattern
from st.integers(0, 2**32 − 1), mixed with NaNs of any payload, signalling ones included, and with
the patterns of Hypothesis's own float32s, which draw NaN, ±inf, −0.0 and subnormals often.
read_ply of the file write_ply writes must give 59 float32 arrays, one per field, whose bits equal
the splats', with the test's own map of each field to its array, compared through
.view(np.uint32).
"""

import errno
import math
import struct
import tempfile
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import NoReturn

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from splatpipe.errors import ExpectedFailure
from splatpipe.train.ply import check_ply, read_ply, splat_header, splat_records, write_ply
from splatpipe.train.splats import Splats

# The Splat_Layout's 59 field names in their order (Req 8.3).
NAMES = (["x", "y", "z"] + [f"f_dc_{i}" for i in range(3)] + [f"f_rest_{k}" for k in range(45)]
         + ["opacity"] + [f"scale_{i}" for i in range(3)] + [f"rot_{i}" for i in range(4)])
RECORD = 236  # bytes in a record: the 59 fields as float32

# The shape of one splat's part of each array of Splats.
SHAPES = {
    "means": (3,),
    "log_scales": (3,),
    "quats": (4,),
    "opacity_logits": (),
    "sh0": (3,),
    "sh_rest": (15, 3),
}


def layout_lines(count: int) -> list[str]:
    """The Splat_Layout's header lines for `count` splats, as Req 8.3 gives them."""
    return (["ply", "format binary_little_endian 1.0", f"element vertex {count}"]
            + [f"property float {name}" for name in NAMES] + ["end_header"])


def header_bytes(lines: list[str]) -> bytes:
    """Header `lines` as a file holds them: ASCII, each line ending in a newline."""
    return "".join(f"{line}\n" for line in lines).encode("ascii")


def finite_splats(count: int) -> Splats:
    """`count` float32 splats whose values are 0, 1, 2 and so on, array by array, so every value
    is finite and different."""
    arrays, start = {}, 0
    for name, shape in SHAPES.items():
        size = count * math.prod(shape)
        arrays[name] = np.arange(start, start + size, dtype=np.float32).reshape(count, *shape)
        start += size
    return Splats(**arrays)


def refused(function: Callable[..., object], path: Path, *args: object) -> str:
    """The message of the ExpectedFailure that function(path, *args) raises, once it's checked
    that the message names the file at `path` and holds no traceback (Req 8.7, 8.10, 14.6)."""
    with pytest.raises(ExpectedFailure) as caught:
        function(path, *args)
    message = str(caught.value)
    assert str(path) in message
    assert "Traceback" not in message
    return message


# The header and the size (Req 8.3)

@pytest.mark.parametrize(("count", "size"), [(1, 1_472), (3_000_000, 1_478)])
def test_splat_header_holds_exactly_the_lines_of_the_splat_layout(count: int, size: int) -> None:
    header = splat_header(count)
    assert len(header) == size
    assert header == header_bytes(layout_lines(count))


def test_a_written_file_is_the_header_then_236_bytes_for_each_splat(tmp_path: Path) -> None:
    path = tmp_path / "train.ply"
    write_ply(path, finite_splats(3))
    data = path.read_bytes()
    header = header_bytes(layout_lines(3))
    assert data.startswith(header)
    assert len(data) == len(header) + 3 * RECORD


# The PLY_Check's refusals (Req 8.6, 8.7)

def test_check_ply_refuses_a_file_of_0_splats(tmp_path: Path) -> None:
    path = tmp_path / "train.ply"
    write_ply(path, finite_splats(0))
    assert f"{path} holds no splats." in refused(check_ply, path)


LINES = layout_lines(2)  # the layout's 63 lines for the 2 records each header test writes


def replaced(changes: dict[str, str]) -> list[str]:
    """The layout's lines for 2 splats, with each line in `changes` replaced by its value."""
    return [changes.get(line, line) for line in LINES]


@pytest.mark.parametrize(("lines", "failed"), [
    pytest.param([*LINES[:2], "comment SplatRenderMode: default", *LINES[2:]],
                 "its line 3 is 'comment SplatRenderMode: default', but the layout's line 3 is "
                 "'element vertex <N>'",
                 id="a comment line"),
    # A face element after the vertex properties, at line 63, in a file of 1,526 + 472 bytes. The
    # check reads only the longest header a file of this size can have, 1,472 bytes, which ends
    # 11 bytes into that line.
    pytest.param([*LINES[:-1], "element face 0", "property list uchar int vertex_indices",
                  LINES[-1]],
                 "it has no end_header line in its first 1472 bytes, the longest header a file of "
                 "1998 bytes can have, and its line 63, which starts 'element fac', doesn't end "
                 "within them",
                 id="a second element"),
    pytest.param(replaced({"property float x": "property double x"}),
                 "its line 4 is 'property double x', but after its element line the layout has "
                 "only 'property float <name>' lines",
                 id="property double x"),
    pytest.param(replaced({"property float f_dc_0": "property float f_dc_2",
                           "property float f_dc_2": "property float f_dc_0"}),
                 "its fields aren't the layout's 59 in order (out of place: f_dc_2, f_dc_0)",
                 id="two fields swapped"),
])
def test_check_ply_refuses_a_header_other_than_the_splat_layouts(
        tmp_path: Path, lines: list[str], failed: str) -> None:
    path = tmp_path / "train.ply"
    path.write_bytes(header_bytes(lines) + np.arange(2 * len(NAMES), dtype="<f4").tobytes())
    message = refused(check_ply, path)
    assert f"{path} isn't in the splat layout the train step writes: " in message
    assert failed in message


@pytest.mark.parametrize("change", [-1, 1], ids=["one byte short", "one byte long"])
def test_check_ply_refuses_a_size_other_than_the_headers_plus_236_bytes_per_splat(
        tmp_path: Path, change: int) -> None:
    path = tmp_path / "train.ply"
    write_ply(path, finite_splats(2))
    data = path.read_bytes()
    path.write_bytes(data[:-1] if change < 0 else data + b"\0")
    need = len(header_bytes(LINES)) + 2 * RECORD  # 1,472 + 472 = 1,944
    assert (f"{path} is {need + change} bytes, but its header is 1472 bytes and says it holds 2 "
            f"splats, so it must be {need} bytes" in refused(check_ply, path))


def test_check_ply_gives_how_many_splats_hold_a_nan_and_the_count_in_each_field(
        tmp_path: Path) -> None:
    splats = finite_splats(4)
    splats.opacity_logits[[0, 2]] = np.nan
    splats.log_scales[2, 1] = np.nan  # scale_1, in a splat that already holds a NaN
    path = tmp_path / "train.ply"
    write_ply(path, splats)
    assert ("2 splats hold a NaN or an infinity: opacity (2), scale_1 (1)."
            in refused(check_ply, path))


# The PLY_Writer (Req 8.10, D6)

def test_write_ply_names_the_file_and_the_reason_when_the_disk_is_full(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def full_disk(self: Path, *args: object, **kwargs: object) -> NoReturn:
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(Path, "open", full_disk)
    path = tmp_path / "train.ply"
    assert "(No space left on device)" in refused(write_ply, path, finite_splats(2))


def test_a_baked_float64_value_too_big_for_float32_is_written_as_an_infinity_the_check_reports(
        tmp_path: Path) -> None:
    # The Exposure_Bake gives float64 colours. 1e39 is finite in float64, but too big for float32.
    splats = finite_splats(3)
    baked = replace(splats, sh0=splats.sh0.astype(np.float64),
                    sh_rest=splats.sh_rest.astype(np.float64))
    baked.sh0[1, 2] = 1e39  # splat 1's f_dc_2
    path = tmp_path / "train.ply"
    write_ply(path, baked)
    offset = len(header_bytes(layout_lines(3))) + RECORD * 1 + 4 * NAMES.index("f_dc_2")
    assert struct.unpack_from("<f", path.read_bytes(), offset) == (math.inf,)
    assert ("1 splat holds a NaN or an infinity: f_dc_2 (1)."
            in refused(check_ply, path))


# The PLY_Check passes finite splats: Property 6 (Req 8.6, 8.14)

# −0.0, the smallest and the largest float32 subnormals, and ±3.4e38, the largest float32, each
# drawn often beside Hypothesis's own finite float32s, so a check that took a sign, a tiny value
# or a huge one for a NaN or an infinity fails.
FLOAT32 = np.finfo(np.float32)
EDGE_SIZES = (FLOAT32.smallest_subnormal, np.nextafter(FLOAT32.smallest_normal, np.float32(0)),
              FLOAT32.max)
EDGE_FLOAT32S = (-0.0, *(float(size) for size in EDGE_SIZES),
                 *(-float(size) for size in EDGE_SIZES))
FINITE_FLOAT32S = (st.sampled_from(EDGE_FLOAT32S)
                   | st.floats(width=32, allow_nan=False, allow_infinity=False))


@st.composite
def any_finite_splats(draw: st.DrawFn) -> Splats:
    """The splats of Properties 6, 7 and 8: 1 to 20 float32 splats, each value of each array
    drawn from FINITE_FLOAT32S."""
    count = draw(st.integers(1, 20))
    return Splats(**{name: draw(arrays(np.float32, (count, *shape), elements=FINITE_FLOAT32S))
                     for name, shape in SHAPES.items()})


@settings(max_examples=100, deadline=None)
@given(splats=any_finite_splats())
def test_a_file_of_finite_splats_passes_the_ply_check(splats: Splats) -> None:
    """Feature: splat-training, Property 6: A file of finite splats passes the PLY_Check

    **Validates: Requirements 8.14, 8.6**
    """
    with tempfile.TemporaryDirectory() as folder:  # one folder per example
        path = Path(folder) / "train.ply"
        write_ply(path, splats)
        assert check_ply(path) == len(splats.means)


# One NaN or infinity anywhere fails the PLY_Check: Property 7 (Req 8.15)

NON_FINITE = (math.nan, math.inf, -math.inf)


@st.composite
def splats_and_a_record(draw: st.DrawFn) -> tuple[Splats, int]:
    """The splats of Property 6 and a record r of theirs, from 0 to N − 1."""
    splats = draw(any_finite_splats())
    return splats, draw(st.integers(0, len(splats) - 1))


@settings(max_examples=100, deadline=None)
@given(drawn=splats_and_a_record(), field=st.integers(0, len(NAMES) - 1),
       value=st.sampled_from(NON_FINITE))
def test_one_nan_or_infinity_anywhere_fails_the_ply_check(
        drawn: tuple[Splats, int], field: int, value: float) -> None:
    """Feature: splat-training, Property 7: One NaN or infinity anywhere fails the PLY_Check

    **Validates: Requirements 8.15**
    """
    splats, record = drawn
    with tempfile.TemporaryDirectory() as folder:  # one folder per example
        path = Path(folder) / "train.ply"
        write_ply(path, splats)
        data = bytearray(path.read_bytes())
        offset = len(header_bytes(layout_lines(len(splats)))) + RECORD * record + 4 * field
        data[offset:offset + 4] = struct.pack("<f", value)  # field f of record r
        path.write_bytes(data)
        assert (f"1 splat holds a NaN or an infinity: {NAMES[field]} (1)."
                in refused(check_ply, path))


# A file missing any one field fails the PLY_Check: Property 8 (Req 8.16)

@settings(max_examples=100, deadline=None)
@given(splats=any_finite_splats(), field=st.integers(0, len(NAMES) - 1))
def test_a_file_missing_any_one_field_fails_the_ply_check(splats: Splats, field: int) -> None:
    """Feature: splat-training, Property 8: A file missing any one field fails the PLY_Check

    **Validates: Requirements 8.16**
    """
    lines = splat_header(len(splats)).decode("ascii").splitlines()
    lines.remove(f"property float {NAMES[field]}")  # field f's property line
    records = np.delete(splat_records(splats), field, axis=1)  # field f's 4 bytes in each record
    with tempfile.TemporaryDirectory() as folder:  # one folder per example
        path = Path(folder) / "train.ply"
        path.write_bytes(header_bytes(lines) + records.tobytes())
        # With the brackets, the list must be field f alone, so f_rest_1 can't pass for f_rest_10.
        assert (f"its fields aren't the layout's 59 in order (missing: {NAMES[field]})"
                in refused(check_ply, path))


# The PLY_Reader reads back every value the PLY_Writer wrote, bit for bit: Property 9 (Req 8.17)

def float32_bits(value: float) -> int:
    """The 32 bits of `value`, a float that float32 holds exactly, as struct packs it."""
    return struct.unpack("<I", struct.pack("<f", value))[0]


# A NaN with any payload: either sign, an exponent of all ones and any mantissa but 0. Few of the
# patterns st.integers draws are NaNs, and Hypothesis's float32s hold only quiet NaNs, since a
# float64 NaN cast to float32 is quieted. So this is what makes signalling NaNs, with the
# mantissa's top bit clear, turn up, the NaNs a round trip through float64 would change.
NAN_BITS = st.builds(lambda sign, mantissa: sign << 31 | 0x7F80_0000 | mantissa,
                     st.integers(0, 1), st.integers(1, 2**23 - 1))

# Any 32-bit pattern, mixed with NaNs of any payload and with the patterns of Hypothesis's own
# float32s, which draw NaN, ±inf, −0.0 and subnormals often.
ANY_BITS = (st.integers(0, 2**32 - 1) | NAN_BITS
            | st.floats(width=32).map(float32_bits))


@st.composite
def any_bits_splats(draw: st.DrawFn) -> Splats:
    """The splats of Property 9: 0 to 20 float32 splats, each value of each array the float32
    whose 32 bits ANY_BITS draws."""
    count = draw(st.integers(0, 20))
    bits = {name: draw(arrays(np.uint32, (count, *shape), elements=ANY_BITS))
            for name, shape in SHAPES.items()}
    return Splats(**{name: array.view(np.float32) for name, array in bits.items()})


def field_values(splats: Splats) -> dict[str, np.ndarray]:
    """Each field's values, as the design maps the 59 fields to the splats' arrays (Req 8.3 to
    8.5): x, y and z the means, f_dc_i sh0[:, i], f_rest_k sh_rest[:, k % 15, k // 15], opacity
    the logits, scale_i the log scales and rot_i the quaternions, w first."""
    return {**{name: splats.means[:, i] for i, name in enumerate(["x", "y", "z"])},
            **{f"f_dc_{i}": splats.sh0[:, i] for i in range(3)},
            **{f"f_rest_{k}": splats.sh_rest[:, k % 15, k // 15] for k in range(45)},
            "opacity": splats.opacity_logits,
            **{f"scale_{i}": splats.log_scales[:, i] for i in range(3)},
            **{f"rot_{i}": splats.quats[:, i] for i in range(4)}}


@settings(max_examples=100, deadline=None)
@given(splats=any_bits_splats())
def test_the_ply_reader_reads_back_every_value_the_ply_writer_wrote_bit_for_bit(
        splats: Splats) -> None:
    """Feature: splat-training, Property 9: The PLY_Reader reads back every value the PLY_Writer
    wrote, bit for bit

    **Validates: Requirements 8.17, 8.3, 8.4, 8.5, 8.13**
    """
    expected = field_values(splats)
    with tempfile.TemporaryDirectory() as folder:  # one folder per example
        path = Path(folder) / "train.ply"
        write_ply(path, splats)
        read = read_ply(path)
        assert read.keys() == set(NAMES)  # the 59 fields, one array each
        for name in NAMES:
            assert read[name].dtype == np.float32, name
            assert np.array_equal(read[name].view(np.uint32),
                                  expected[name].view(np.uint32)), name
