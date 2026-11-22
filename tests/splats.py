"""Shared helpers for the clean step's tests.

clean_settings builds a CleanSettings straight from the test config's [clean], the dicts
CLEAN_SHARED and CLEAN_PRESETS in tests/conftest.py, for the tests of pure functions, which need
no config file and no Job.

The PLY builder makes the synthetic splats the tests read, as bytes, so no splat file is
committed (Req 10.3): ply_bytes writes a binary PLY by hand, standard_fields gives the standard
3DGS layout of A1, and gaussians fills that layout with plain values. The builder shares no code
with splatpipe/clean/ply_read.py or ply_write.py, so a reader test isn't checked against the
reader's own logic.

The Names strategies are for Property 10 in tests/test_clean_limits.py: names gives any text and
good file names with one character put in, and three_names gives good names for train_ply,
clean_ply and clean_sog that are often the same name in a different case.

The Layouts and Values strategies are for Property 9 in tests/test_clean_ply.py: layouts gives a
PLY layout, vertex_values fills its vertex element with values from from_dtype, NaNs with
payloads, infinities and -0.0 included, and row_subsets picks the rows to write. layouts draws
the extra properties' types from ply_read.py's PLY_TYPES, so every type the reader takes turns
up, but vertex_values types the values with the builder's own STRUCT_CODES.
"""

import functools
import struct
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from hypothesis import strategies as st
from hypothesis.extra.numpy import from_dtype

from splatpipe.clean.ply_read import PLY_TYPES
from splatpipe.clean.settings import CleanSettings, check_settings
from splatpipe.config import StepSettings

from conftest import CLEAN_PRESETS, CLEAN_SHARED

# The Preset whose table clean_settings reads: the test config's default_preset.
PRESET = "fast"


def clean_settings(**values: object) -> CleanSettings:
    """The test config's [clean] settings for the fast Preset, with the values given changed,
    like clean_settings(neighbours=3).

    They're checked the way the clean step checks its own, so every number setting comes back as
    a float, as the step sees it, and a value the step would refuse raises its ExpectedFailure.
    """
    settings = {**CLEAN_SHARED, **CLEAN_PRESETS[PRESET], **values}
    return check_settings(StepSettings(settings, section="clean", preset=PRESET))


# The PLY builder

# ply_bytes's own table of the PLY types and the struct code each is packed with, written apart
# from ply_read.py's PLY_TYPES. After a byte order, like "<f", each code is also the numpy type
# code of the same size, which the bulk path packs with.
STRUCT_CODES = {"char": "b", "int8": "b", "uchar": "B", "uint8": "B",
                "short": "h", "int16": "h", "ushort": "H", "uint16": "H",
                "int": "i", "int32": "i", "uint": "I", "uint32": "I",
                "float": "f", "float32": "f", "double": "d", "float64": "d"}
# One PLY element's properties, as (name, PLY type) pairs in file order.
Fields = Sequence[tuple[str, str]]
# One PLY element's rows: a sequence with one value per property in each row, in the properties'
# order, or a structured numpy array with a field named after each property.
Rows = Sequence[Sequence[object]] | np.ndarray


def ply_bytes(fields: Fields, rows: Rows, *, byte_order: str = "<",
              comments: Sequence[str] = (), before: Sequence[tuple[str, Fields, Rows]] = (),
              after: Sequence[tuple[str, Fields, Rows]] = (), eol: bytes = b"\n") -> bytes:
    """A binary PLY whose vertex element has the properties `fields` and the rows `rows`, as
    bytes, written by hand.

    The header is ply, the format line, a comment line for each text in `comments`, then the
    elements of `before`, vertex and the elements of `after`, each as an element line with its
    count and a property line for each property, and last end_header. Every line ends in `eol`,
    so eol=b"\\r\\n" writes Windows line ends, and the header is encoded as Latin-1. `byte_order`
    is "<" for binary_little_endian or ">" for binary_big_endian. `before` and `after` hold
    (name, fields, rows) triples, like ("meta", [("a", "uchar"), ("b", "uchar")], [(1, 2)]).

    The body holds each element's rows in the same order, packed in that byte order with no
    padding. Rows given as sequences are packed with struct, one row at a time, by position.
    struct takes a float32 value through a Python float, which keeps the bits of every value
    st.from_dtype draws but makes a signaling NaN quiet. A structured numpy array, like the one
    gaussians gives, takes the bulk path: numpy packs it in one go and keeps every bit. Each
    property's values come from the array's field of the same name, so the array can hold
    fields the header leaves out.
    """
    formats = {"<": "binary_little_endian", ">": "binary_big_endian"}
    lines = ["ply", f"format {formats[byte_order]} 1.0"]
    lines += [f"comment {text}" for text in comments]
    body: list[bytes] = []
    for element, element_fields, element_rows in [*before, ("vertex", fields, rows), *after]:
        lines.append(f"element {element} {len(element_rows)}")
        lines += [f"property {ply_type} {name}" for name, ply_type in element_fields]
        codes = [STRUCT_CODES[ply_type] for _, ply_type in element_fields]
        if isinstance(element_rows, np.ndarray):
            # Named by position, so a property the header lists twice gets a field each time.
            packed = np.empty(len(element_rows),
                              [(f"f{k}", byte_order + code) for k, code in enumerate(codes)])
            for k, (name, _) in enumerate(element_fields):
                packed[f"f{k}"] = element_rows[name]
            body.append(packed.tobytes())
        else:
            row = struct.Struct(byte_order + "".join(codes))
            body += [row.pack(*values) for values in element_rows]
    lines.append("end_header")
    return b"".join(line.encode("latin-1") + eol for line in lines) + b"".join(body)


def standard_fields(sh_count: int = 45, normals: bool = True) -> list[tuple[str, str]]:
    """A1's layout, the standard 3DGS one, as (name, PLY type) pairs in the order 3DGS training
    writes them: x, y, z, then nx, ny, nz when `normals` is true, f_dc_0 to f_dc_2, `sh_count`
    f_rest_* properties, opacity, scale_0 to scale_2 and rot_0 to rot_3, each a float. With the
    defaults that's 62 properties, 248 bytes a row.

    `sh_count` can be any count, so a test can build the f_rest_* counts the reader refuses.
    """
    names = ["x", "y", "z", *(["nx", "ny", "nz"] if normals else []),
             "f_dc_0", "f_dc_1", "f_dc_2", *(f"f_rest_{k}" for k in range(sh_count)),
             "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    return [(name, "float") for name in names]


def gaussians(n: int, *, sh_count: int = 45, normals: bool = True,
              min_opacity: float = clean_settings().min_opacity) -> np.ndarray:
    """`n` Gaussians with plain values, as a structured array with a float32 field for each
    property of standard_fields(sh_count, normals), in that order, for ply_bytes's bulk path. A
    test changes a field in place to make a Gaussian broken, faint or far away.

    Each value comes from a number t from 0 to 1, one for each Gaussian and property, drawn by a
    random generator with a fixed seed, so the values are the same in every run:

    - Centres on a lattice of spacing 1, the smallest cube that holds `n`, filled along x, then
      y, then z. Each coordinate is moved off it by 0.1 * t, which breaks the ties a plain
      lattice has between a Gaussian's Centre_Distances to the others. A jitter of
      (i * 0.618033...) % 1 wouldn't: the step from Gaussian i - 1 to i can be the same as the
      one from i to i + 1, which leaves i with two others at the same distance.
    - Normals of 0, as 3DGS training writes them.
    - f_dc_* from -1 to 1, and f_rest_* from -0.1 to 0.1.
    - Real_Opacities from 10% to 90% of the way from `min_opacity` up to 1, so every one is
      above it. `min_opacity` is the test settings' unless given, and has to be below 1.
    - Stored scales from -8 to 3.
    - Unit quaternions, each a turn about the x axis: rot_0 = cos(h) and rot_1 = sin(h) for an
      h from 0 to pi, and rot_2 = rot_3 = 0.
    """
    names = [name for name, _ in standard_fields(sh_count, normals)]
    t = dict(zip(names, np.random.default_rng(0).random((n, len(names))).T))  # a column each
    data = np.zeros(n, [(name, "<f4") for name in names])
    side = 1
    while side ** 3 < n:
        side += 1
    i = np.arange(n)
    data["x"] = i % side + 0.1 * t["x"]
    data["y"] = i // side % side + 0.1 * t["y"]
    data["z"] = i // side ** 2 + 0.1 * t["z"]
    for name in names:
        if name.startswith("f_dc_"):
            data[name] = 2 * t[name] - 1
        elif name.startswith("f_rest_"):
            data[name] = 0.2 * t[name] - 0.1
    real_opacity = min_opacity + (1 - min_opacity) * (0.1 + 0.8 * t["opacity"])
    data["opacity"] = np.log(real_opacity / (1 - real_opacity))
    for name in ("scale_0", "scale_1", "scale_2"):
        data[name] = -8 + 11 * t[name]
    data["rot_0"] = np.cos(np.pi * t["rot_0"])
    data["rot_1"] = np.sin(np.pi * t["rot_0"])
    return data


# Names, for Property 10

# What a good name is made of: any character from U+0020, the space, up, but /, \ and :.
name_characters = st.characters(min_codepoint=0x20, exclude_characters="/\\:")
# The part of a good name before its ending.
stems = st.text(name_characters, min_size=1)
# Good names: a stem with no ending, which only train_ply takes, or with the ending of clean_ply
# or clean_sog. Never . or .. alone, which name a folder.
good_names = (st.tuples(stems, st.sampled_from(("", ".ply", ".sog"))).map("".join)
              .filter(lambda name: name not in (".", "..")))
# What near_miss_names puts into a good name: /, \ and :, which a name can't hold, a dot, a NUL
# and a line break, and the division slash and the fullwidth slash, which only look like a /.
inserted_characters = st.sampled_from(("/", "\\", ":", ".", "\x00", "\n", "\u2215", "\uff0f"))


@st.composite
def near_miss_names(draw: st.DrawFn) -> str:
    """A good name with one of inserted_characters put in at a random place."""
    name = draw(good_names)
    at = draw(st.integers(min_value=0, max_value=len(name)))
    return name[:at] + draw(inserted_characters) + name[at:]


# The names for the half of Property 10 that checks one name: any text, near misses, and .., a
# drive, an absolute path, a stream and an ending in upper case.
names = st.one_of(st.text(), near_miss_names(),
                  st.sampled_from(("..", "C:clean.ply", "/abs.ply", "clean.ply:x", "clean.PLY")))
# Ways to change a name's case, which turn clean.ply into Clean.ply or CLEAN.PLY. None of them
# makes a character that a good name can't hold.
case_changes = st.sampled_from((str.lower, str.upper, str.capitalize, str.swapcase))


@st.composite
def three_names(draw: st.DrawFn) -> dict[str, str]:
    """Good names for train_ply, clean_ply and clean_sog, for the half of Property 10 that
    checks the clashes.

    clean_ply and clean_sog are stems in a changed case with their own lower-case ending.
    train_ply is one of those two names or another good name, in a changed case, so in many
    draws it's the same as one of them once case is ignored.
    """
    clean_ply = draw(case_changes)(draw(stems)) + ".ply"
    clean_sog = draw(case_changes)(draw(stems)) + ".sog"
    train_ply = draw(case_changes)(draw(st.sampled_from((clean_ply, clean_sog)) | good_names))
    return {"train_ply": train_ply, "clean_ply": clean_ply, "clean_sog": clean_sog}


# Layouts and Values, for Property 9

# The two comments a viewer reads to tell how a scene has to be rendered, which the clean splat
# keeps (Req 4.3). A layout's comments are often one of them.
RENDER_COMMENTS = ("SplatRenderMode: mip", "postshot.anti_aliasing=1")
# The text of a comment line: any Latin-1 text with no line break, because the header is read
# line by line. ply_bytes writes each text as the line "comment <text>".
comment_texts = st.text(st.characters(max_codepoint=0xFF, exclude_characters="\n\r"))
# The element a layout can put before or after vertex: element meta 2, with two uchar
# properties. The reader skips its rows when it comes before vertex, and never reads them after.
META = ("meta", (("a", "uchar"), ("b", "uchar")), ((1, 2), (3, 4)))
# The most Gaussians vertex_values gives. Few, because each value is drawn on its own, a layout
# can have more than 60 fields, and Hypothesis caps how much one example can draw.
MOST_ROWS = 5


@dataclass(frozen=True)
class Layout:
    """A PLY layout for Property 9: the vertex element's properties, the byte order, the comments
    and the elements around vertex, as ply_bytes takes them."""

    fields: tuple[tuple[str, str], ...]  # the vertex element's (name, PLY type) pairs, in order
    byte_order: str  # "<" or ">"
    comments: tuple[str, ...]  # the text of each comment line, in order
    before: tuple[tuple[str, Fields, Rows], ...]  # (META,) or nothing
    after: tuple[tuple[str, Fields, Rows], ...]  # (META,) or nothing, and never META both times


@st.composite
def layouts(draw: st.DrawFn) -> Layout:
    """A PLY layout: A1's layout, with 0, 9, 24 or 45 f_rest_* properties and with or without
    normals, plus extra properties named extra_0, extra_1 and so on, each with a type from
    PLY_TYPES, all shuffled together. Either byte order. Comments of Latin-1 text, often one of
    RENDER_COMMENTS. META before vertex, after it, or nowhere.
    """
    sh_count = draw(st.sampled_from((0, 9, 24, 45)))
    normals = draw(st.booleans())
    extras = draw(st.lists(st.sampled_from(tuple(PLY_TYPES))))
    fields = draw(st.permutations([*standard_fields(sh_count, normals),
                                   *((f"extra_{k}", ply_type)
                                     for k, ply_type in enumerate(extras))]))
    byte_order = draw(st.sampled_from("<>"))
    comments = draw(st.lists(st.sampled_from(RENDER_COMMENTS) | comment_texts))
    meta = draw(st.sampled_from(("nowhere", "before", "after")))
    return Layout(tuple(fields), byte_order, tuple(comments),
                  before=(META,) if meta == "before" else (),
                  after=(META,) if meta == "after" else ())


@st.composite
def vertex_values(draw: st.DrawFn, fields: Fields) -> np.ndarray:
    """1 to MOST_ROWS Gaussians for a vertex element with the properties `fields`, as a
    little-endian structured array with a field of each property's type, from STRUCT_CODES, for
    ply_bytes's bulk path, which keeps every bit.

    Each field's values come from from_dtype, which gives NaNs with payloads, infinities, -0.0
    and subnormals for a float, and the whole range for a whole number.
    """
    row_type = np.dtype([(name, "<" + STRUCT_CODES[ply_type]) for name, ply_type in fields])
    row = st.tuples(*(field_values(row_type[name]) for name in row_type.names))
    # Each value is a numpy scalar of its field's type, which np.array stores bit for bit.
    return np.array(draw(st.lists(row, min_size=1, max_size=MOST_ROWS)), row_type)


@functools.cache
def field_values(field_type: np.dtype) -> st.SearchStrategy[np.generic]:
    """Values of the numpy type `field_type`, from from_dtype, each a numpy scalar of that type.

    The strategy is made once for each type, because Hypothesis checks a new strategy the first
    time it's drawn from, and each example draws from one for every field.
    """
    return from_dtype(field_type)


def row_subsets(count: int) -> st.SearchStrategy[np.ndarray]:
    """Row numbers of a vertex element of `count` rows, as write_splat takes them: increasing,
    none twice, and never none, because the reader refuses a vertex count of 0."""
    return st.sets(st.integers(min_value=0, max_value=count - 1), min_size=1).map(
        lambda chosen: np.array(sorted(chosen), dtype=np.intp))
