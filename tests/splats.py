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

The Splats strategy is for Properties 1, 2 and 7 in tests/test_clean_filters.py and
tests/test_clean_limits.py: splats gives Splat_Data of 1 to 60 Gaussians, with NaNs and
infinities in any property. has_tied_distances is the brute-force tie check, which splats uses
to drop a draw in which a Gaussian has two others at the same Centre_Distance (Req 10.13).

The Centre sets strategy is for Property 3 in tests/test_clean_filters.py: centre_sets gives a k
and a set of more than k centres for nearest_others to search, coincident pairs included.
twin_candidates says where a pair can go without making a tie at a Gaussian's k-th nearest, and
has_tie_at_kth_nearest is the brute-force check that drops a draw with one (Req 10.13).

The Splats for the references strategy is for Property 4 in tests/test_clean_filters.py:
reference_splats gives Splat_Data of 0 to 60 Gaussians, the empty set included, with finite
Required_Properties, stored opacities from -12 to 12 and stored scales from -8 to 3, and drops a
draw with tied Centre_Distances as splats does. The Settings strategy, valid_settings, gives
valid values of the settings the filters and the limits read, as keyword arguments for
clean_settings.

The Splats for the references strategy for Property 5, in tests/test_clean_filters.py too, is
reference_centres: it gives 1 to 60 finite centres for isolated_keep, as float32 values in a
float64 array, and drops a draw with tied Centre_Distances as splats does.

The Splats for the extremes strategies are for Property 8 in tests/test_clean_filters.py.
extreme_splats gives Splat_Data of 0 to 60 Gaussians whose centres, stored opacities and stored
scales are any finite float32 values, 3.4e38, -3.4e38 and the subnormals included. It keeps the
draws with tied Centre_Distances, because faint_keep searches for no neighbours. rescaled_splats
gives Splat_Data and the same Gaussians with c added to every stored scale and every centre
multiplied by 2**m, every moved value exact in float32, and drops a draw with tied
Centre_Distances as splats does.

The Grids strategy is for Property 6 in tests/test_clean_filters.py: grids gives Splat_Data on an
evenly spaced grid of more than neighbours Gaussians, with finite properties, random unit
quaternions and every Real_Opacity above min_opacity. It has no tie check, because a grid is full
of Gaussians with two others at one Centre_Distance, and Property 6 has to hold whichever of them
the tree counts (Req 10.13).
"""

import functools
import itertools
import math
import struct
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeVar

import numpy as np
from hypothesis import assume
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays, from_dtype

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


# Splats, for Properties 1, 2 and 7

T = TypeVar("T")


def mostly(plain: st.SearchStrategy[T], *others: T, one_in: int = 4) -> st.SearchStrategy[T]:
    """Values from `plain`, except one time in `one_in`, when it's one of `others`, which
    Hypothesis shrinks back to a value from `plain`. By default that's three times in four, and
    one of `others` the fourth time.

    st.one_of(plain, plain, plain, ...) wouldn't weight them, because one_of keeps each strategy
    it's given only once.
    """
    sometimes = st.sampled_from(others)
    return st.integers(min_value=0, max_value=one_in - 1).flatmap(
        lambda k: plain if k < one_in - 1 else sometimes)


# The most Gaussians splats gives. Few, so the brute-force tie check and oracles stay quick.
MOST_GAUSSIANS = 60
# A lattice point's coordinates are whole numbers from -LATTICE_REACH to LATTICE_REACH.
LATTICE_REACH = 3
# The most clusters one splat has. A cluster is a lattice point that many Gaussians share, so
# they sit within the jitter of each other, and one Cube can hold enough Faint_Gaussians to keep
# them.
MOST_CLUSTERS = 3
# Where a Gaussian's centre is before its jitter: a lattice point four times in nine, a cluster
# four times in nine, and a far outlier the ninth time. Hypothesis shrinks to the first.
CENTRE_KINDS = ("lattice",) * 4 + ("cluster",) * 4 + ("outlier",)
# The index-based jitter: Gaussian i's centre moves by JITTERS[i], a fixed table of random values
# from 0 to 0.01 on each axis, so Gaussians that share a lattice point or a cluster don't share a
# centre. A far outlier's jitter is scaled by its factor, as FAR_FACTORS says. The design's
# (i * 0.618033...) % 1 often takes the same step from Gaussian i - 1 to i as from i to i + 1,
# which leaves i with two others at the same distance. On these splats without far outliers, it
# put a tie in half to two thirds of the draws, and this table in none of 1,200.
JITTERS = 0.01 * np.random.default_rng(1).random((MOST_GAUSSIANS, 3))
# What pushes a far outlier out from a lattice point other than the origin: a factor from 10 to
# 10**6, and 1e38 one time in 64, so the outliers' coordinates reach 3e38 and -3e38. The factor
# multiplies the lattice point after its jitter is added, so the jitter grows with it. Added
# after the factor, a jitter of at most 0.01 is lost to float32's rounding at coordinates from
# about 10**5 up, which left two far outliers on one centre, or two at one distance from a third,
# in about 13% of the draws. From a factor of about 1e15 on, float64 can't keep apart the distances from
# such an outlier to the Gaussians near the lattice, so a draw with such an outlier and two other
# Gaussians near the lattice has a tie and is dropped. That's why 1e38 is the only factor that
# large, and why it's rare. One time in four, it put a tie in 40 to 50% of the draws, and none of
# 300 draws that passed held a 1e38 outlier.
FAR_FACTORS = mostly(st.floats(min_value=10.0, max_value=1e6), 1e38, one_in=64)
# Stored opacities and scales beyond their plain ranges, near float32's largest value, about
# 3.4028e38.
HUGE = (3.4e38, -3.4e38)
# Unit quaternions that float32 holds exactly: 1 or -1 in one component and 0 in the rest, or 0.5
# or -0.5 in all four, so each component is sometimes the only one that isn't 0.
UNIT_QUATERNIONS = (*(tuple(sign if k == axis else 0.0 for k in range(4))
                      for axis in range(4) for sign in (1.0, -1.0)),
                    *itertools.product((0.5, -0.5), repeat=4))
# The quaternions of zero length: four zeros, each of them 0.0 or -0.0.
ZERO_QUATERNIONS = tuple(itertools.product((0.0, -0.0), repeat=4))
# The eleven Required_Properties, written out here rather than taken from ply_read.py's REQUIRED,
# and the other properties of standard_fields(), the Carried_Properties.
REQUIRED_NAMES = ("x", "y", "z", "opacity", "scale_0", "scale_1", "scale_2",
                  "rot_0", "rot_1", "rot_2", "rot_3")
CARRIED_NAMES = tuple(name for name, _ in standard_fields() if name not in REQUIRED_NAMES)
# The values that splats puts in a property to break it, or to show that it's carried.
NOT_FINITE = (math.nan, math.inf, -math.inf)

lattice_points = st.tuples(*[st.integers(min_value=-LATTICE_REACH, max_value=LATTICE_REACH)] * 3)
# A far outlier, as a lattice point other than the origin and the factor that pushes it out.
far_outliers = st.tuples(lattice_points.filter(any), FAR_FACTORS)
stored_scales = mostly(st.floats(min_value=-8, max_value=3, width=32), *HUGE)
quaternions = mostly(st.sampled_from(UNIT_QUATERNIONS), *ZERO_QUATERNIONS)
# What to put in one Gaussian: a NaN or an infinity in one of its Required_Properties, or in one
# of its Carried_Properties, or None for nothing.
required_faults = st.none() | st.tuples(st.sampled_from(REQUIRED_NAMES),
                                        st.sampled_from(NOT_FINITE))
carried_faults = st.none() | st.tuples(st.sampled_from(CARRIED_NAMES),
                                       st.sampled_from(NOT_FINITE))


@st.composite
def centre_bases(draw: st.DrawFn, clusters: Sequence[tuple[int, int, int]],
                 kinds: Sequence[str] = CENTRE_KINDS) -> tuple[tuple[int, int, int], float]:
    """Where one Gaussian's centre is before its jitter, as a lattice point and the factor that
    multiplies it once the jitter is added: a lattice point or one of `clusters`, with a factor
    of 1, or a far outlier, as `kinds` weights them, CENTRE_KINDS unless given."""
    kind = draw(st.sampled_from(kinds))
    if kind == "lattice":
        return draw(lattice_points), 1.0
    if kind == "cluster":
        return draw(st.sampled_from(clusters)), 1.0
    return draw(far_outliers)


def splats(*, min_opacity: float = clean_settings().min_opacity) -> st.SearchStrategy[np.ndarray]:
    """Splat_Data of 1 to MOST_GAUSSIANS Gaussians, for Properties 1, 2 and 7: a structured array
    with a float32 field for each property of standard_fields(), as gaussians gives.

    - Centres are lattice points, clusters and far outliers up to 3e38 and -3e38, each moved by
      its index-based jitter, which a far outlier's factor scales too.
    - Stored opacities are mostly from -12 to 12, so Faint_Gaussians and the rest both turn up,
      and otherwise 3.4e38, -3.4e38 or the exact logit of `min_opacity`, when that's finite.
      `min_opacity` is the test settings' unless given.
    - Stored scales are mostly from -8 to 3, and otherwise 3.4e38 or -3.4e38.
    - Quaternions are mostly unit ones, and otherwise four zeros.
    - Any Gaussian, up to every one, can have a NaN or an infinity in one Required_Property, and
      any can have one in a Carried_Property. The other Carried_Properties keep the plain values
      gaussians gives them.

    A draw is dropped with assume when a Gaussian has two others at the same Centre_Distance
    (Req 10.13). That's checked on the centres before any NaN or infinity goes in, when they're
    all finite. A Gaussian that then gets a NaN or an infinity in its centre leaves only a subset
    of the rest for a filter to search, and a subset has no ties if the whole set has none.
    """
    logit = (math.log(min_opacity / (1 - min_opacity)),) if 0 < min_opacity < 1 else ()
    return splat_data(mostly(st.floats(min_value=-12, max_value=12, width=32), *logit, *HUGE))


@st.composite
def splat_data(draw: st.DrawFn, opacities: st.SearchStrategy[float]) -> np.ndarray:
    """The draw that splats describes, with its stored opacities from `opacities`."""
    n = draw(st.integers(min_value=1, max_value=MOST_GAUSSIANS))
    clusters = draw(st.lists(lattice_points, min_size=1, max_size=MOST_CLUSTERS))

    def one_each(values: st.SearchStrategy[T]) -> list[T]:
        return draw(st.lists(values, min_size=n, max_size=n))

    data = gaussians(n)
    points, factors = zip(*one_each(centre_bases(clusters)))
    centres = (np.array(factors)[:, None]
               * (np.array(points, dtype=np.float64) + JITTERS[:n]))
    for axis, name in enumerate(("x", "y", "z")):
        data[name] = centres[:, axis]
    # Checked here, before anything else is drawn, so a dropped draw costs little.
    assume(not has_tied_distances(np.column_stack(
        [data[name].astype(np.float64) for name in ("x", "y", "z")])))
    data["opacity"] = one_each(opacities)
    for name in ("scale_0", "scale_1", "scale_2"):
        data[name] = one_each(stored_scales)
    rotations = np.array(one_each(quaternions), dtype=np.float64)
    for axis, name in enumerate(("rot_0", "rot_1", "rot_2", "rot_3")):
        data[name] = rotations[:, axis]
    for faults in (required_faults, carried_faults):
        for row, fault in enumerate(one_each(faults)):
            if fault is not None:
                name, value = fault
                data[name][row] = value
    return data


def has_tied_distances(centres: np.ndarray) -> bool:
    """Whether one of `centres`, an (m, 3) float64 array of finite centres, has two others at the
    same Centre_Distance, found by brute force (Req 10.13).

    Each Centre_Distance is sqrt(dx*dx + dy*dy + dz*dz), summed in float64 in that order, which is
    how the tree works it out, so a tie here is a tie in the tree. The centres are float32 values,
    so no difference between two of them overflows or underflows when it's squared in float64.
    Each centre's distances, sorted, start with its own 0, which is dropped. Two centres in one
    place are at distance 0 from each other, so a third is at the same distance from both.
    """
    d = centres[:, None, :] - centres[None, :, :]
    distances = np.sqrt(d[..., 0] * d[..., 0] + d[..., 1] * d[..., 1] + d[..., 2] * d[..., 2])
    others = np.sort(distances, axis=1)[:, 1:]
    return bool((others[:, 1:] == others[:, :-1]).any())


# Centre sets, for Property 3

# The largest k that centre_sets gives nearest_others: the top of the range the design's
# settings strategy gives neighbours.
MOST_NEIGHBOURS = 10
# The most twins centre_sets puts in one set. A twin is a Gaussian on the centre of one already
# there, so it makes a coincident pair, or a coincident triple when two twins share a centre.
MOST_TWINS = 3


@st.composite
def centre_sets(draw: st.DrawFn) -> tuple[np.ndarray, int]:
    """A k from 1 to MOST_NEIGHBOURS and a set of more than k centres for nearest_others to
    search, coincident pairs included, for Property 3. The centres are an (m, 3) float64 array
    of float32 values, as positions gives them.

    - First come k + 1 to MOST_GAUSSIANS lattice points, clusters and far outliers, each moved
      by its index-based jitter and stored as float32, as splats draws them.
    - Then up to MOST_TWINS twins go on the end, each on a centre that twin_candidates allows,
      so a twin never makes a tie at a Gaussian's k-th nearest.

    A draw is dropped with assume when a Gaussian's k-th and (k + 1)-th nearest others are at
    the same Centre_Distance, the rule of Req 10.13 for this set and its k, because then its k
    nearest others are no single set. Every other tie stays, like the one a pair makes for each
    other Gaussian, which has the two at one distance.

    Put on a centre drawn at random, a twin made a tie at a Gaussian's k-th nearest in 78% of the
    1,498 draws that had one, because the Gaussian it copies is often another's k-th nearest.
    With twin_candidates, about 7 draws in 10 hold a pair and 4 in 100 a triple. In every draw
    with a pair, the tree puts another Gaussian at distance 0 ahead of one in its own row, the
    case nearest_others has to find the Gaussian in. The only draws dropped hold a far outlier
    at 1e38, which is at one distance from every Gaussian near the lattice. That's about 4 draws
    in 100, and up to 15 in one run of 100 examples.
    """
    k = draw(st.integers(min_value=1, max_value=MOST_NEIGHBOURS))
    n = draw(st.integers(min_value=k + 1, max_value=MOST_GAUSSIANS))
    clusters = draw(st.lists(lattice_points, min_size=1, max_size=MOST_CLUSTERS))
    points, factors = zip(*draw(st.lists(centre_bases(clusters), min_size=n, max_size=n)))
    stored = (np.array(factors)[:, None]
              * (np.array(points, dtype=np.float64) + JITTERS[:n])).astype(np.float32)
    centres = stored.astype(np.float64)
    for _ in range(draw(st.integers(min_value=0, max_value=MOST_TWINS))):
        candidates = twin_candidates(centres, k)
        if not candidates:
            break
        centres = np.vstack([centres, centres[draw(st.sampled_from(candidates))]])
    assume(not has_tie_at_kth_nearest(centres, k))
    return centres, k


def twin_candidates(centres: np.ndarray, k: int) -> list[int]:
    """The rows of `centres`, an (m, 3) float64 array of more than k finite centres, whose centre
    can take a twin without making a tie at a Gaussian's k-th nearest that wasn't there.

    A twin of Gaussian a gives every other Gaussian c one more distance to the others, its
    distance to a, and gives a a 0, its distance to the twin. That's the column of a in
    centre_distances, whose own entry is 0. With s_k the distance of c's k-th nearest other, the
    new distance makes a tie at c's k-th nearest exactly when it's s_k, or when it's below s_k
    and c's (k - 1)-th nearest is at s_k too, which moves that tie up to the k-th and (k + 1)-th.
    The twin's distances to the others are a's new ones, so it gets a tie exactly when a does.
    """
    distances = centre_distances(centres)
    others = np.sort(distances, axis=1)[:, 1:]  # each one's distances to the others, nearest first
    kth = others[:, k - 1 : k]  # s_k, as a column
    moves_up = others[:, k - 2 : k - 1] == kth if k > 1 else np.zeros_like(kth, dtype=bool)
    # ties[c, a] says a twin of a would make a tie at c's k-th nearest.
    ties = (distances == kth) | ((distances < kth) & moves_up)
    return np.flatnonzero(~ties.any(axis=0)).tolist()


def has_tie_at_kth_nearest(centres: np.ndarray, k: int) -> bool:
    """Whether one of `centres`, an (m, 3) float64 array of finite centres, has its k-th and
    (k + 1)-th nearest others at the same Centre_Distance, found by brute force (Req 10.13).

    Then its k nearest others are no single set, and the tree's order picks which count (A10). A
    tie nearer in leaves them one set, so it doesn't count here. Nor can a set of k + 1 centres
    have one, because each of them has only k others.
    """
    others = np.sort(centre_distances(centres), axis=1)[:, 1:]
    return others.shape[1] > k and bool((others[:, k - 1] == others[:, k]).any())


def centre_distances(centres: np.ndarray) -> np.ndarray:
    """Every Centre_Distance between two of `centres`, an (m, 3) float64 array of finite centres,
    as an (m, m) array, worked out as has_tied_distances does, which is how the tree does it."""
    d = centres[:, None, :] - centres[None, :, :]
    return np.sqrt(d[..., 0] * d[..., 0] + d[..., 1] * d[..., 1] + d[..., 2] * d[..., 2])


# Splats for the references, for Property 4

# The stored opacities and scales of Property 4's splats, as float32 values: opacities from -12
# to 12, so Faint_Gaussians and the rest both turn up, and scales from -8 to 3. In these ranges
# no exp that Property 4's reference takes overflows.
plain_opacities = st.floats(min_value=-12, max_value=12, width=32)
plain_scales = st.floats(min_value=-8, max_value=3, width=32)
# Where a Gaussian of reference_splats has its centre before its jitter: a cluster six times in
# nine, a lattice point twice and a far outlier once. Clusters come more often than CENTRE_KINDS
# gives them, so that more Cubes hold enough Faint_Gaussians to keep them. Hypothesis shrinks to
# the first.
CLUSTERED_KINDS = ("lattice",) * 2 + ("cluster",) * 6 + ("outlier",)


@st.composite
def reference_splats(draw: st.DrawFn) -> np.ndarray:
    """Splat_Data of 0 to MOST_GAUSSIANS Gaussians for Property 4, the empty set included, as a
    structured array with a float32 field for each property of standard_fields(), as gaussians
    gives. Every Required_Property is finite.

    - Centres are lattice points, clusters and far outliers up to 3e38 and -3e38, each moved by
      its index-based jitter, as splats draws them, but with clusters more often, as
      CLUSTERED_KINDS weights them.
    - Stored opacities are from -12 to 12, and stored scales from -8 to 3.
    - The quaternions and the Carried_Properties keep the plain values gaussians gives them.

    A draw is dropped with assume when a Gaussian has two others at the same Centre_Distance
    (Req 10.13).
    """
    n = draw(st.integers(min_value=0, max_value=MOST_GAUSSIANS))
    clusters = draw(st.lists(lattice_points, min_size=1, max_size=MOST_CLUSTERS))
    bases = draw(st.lists(centre_bases(clusters, CLUSTERED_KINDS), min_size=n, max_size=n))
    # Reshaped, so the empty set gives empty columns too.
    points = np.array([point for point, _ in bases], dtype=np.float64).reshape(n, 3)
    factors = np.array([factor for _, factor in bases], dtype=np.float64).reshape(n, 1)
    centres = factors * (points + JITTERS[:n])
    data = gaussians(n)
    for axis, name in enumerate(("x", "y", "z")):
        data[name] = centres[:, axis]
    # Checked here, before the opacities and scales are drawn, so a dropped draw costs little.
    assume(not has_tied_distances(np.column_stack(
        [data[name].astype(np.float64) for name in ("x", "y", "z")])))
    data["opacity"] = draw(st.lists(plain_opacities, min_size=n, max_size=n))
    for name in ("scale_0", "scale_1", "scale_2"):
        data[name] = draw(st.lists(plain_scales, min_size=n, max_size=n))
    return data


# Splats for the references, for Property 5

# The kinds of centre that one draw of reference_centres takes all its Gaussians from: those of
# CLUSTERED_KINDS, so lattice points and far outliers sit beside clusters, whose Gaussians are
# far nearer each other, and the Isolated_Filter drops some of them, or clusters alone, whose
# Gaussians are spaced alike, so it often drops none. With CLUSTERED_KINDS in every draw, it
# dropped a Gaussian in 91 of the 92 draws of more than neighbours Gaussians in a run of 100
# examples. With this choice it drops one in about three in four, and none in the rest.
# Hypothesis shrinks to the first.
ISOLATION_KINDS = (CLUSTERED_KINDS, ("cluster",))


@st.composite
def reference_centres(draw: st.DrawFn) -> np.ndarray:
    """1 to MOST_GAUSSIANS finite centres for Property 5, as an (n, 3) float64 array of float32
    values, the way positions gives isolated_keep the centres of Splat_Data.

    Centres are lattice points, clusters and far outliers up to 3e38 and -3e38, each moved by
    its index-based jitter and stored as float32, as reference_splats draws them. In half the
    draws they're mixed as CLUSTERED_KINDS weights them, and in the other half every one is in a
    cluster, as ISOLATION_KINDS says.

    A draw is dropped with assume when a Gaussian has two others at the same Centre_Distance
    (Req 10.13).
    """
    n = draw(st.integers(min_value=1, max_value=MOST_GAUSSIANS))
    clusters = draw(st.lists(lattice_points, min_size=1, max_size=MOST_CLUSTERS))
    kinds = draw(st.sampled_from(ISOLATION_KINDS))
    points, factors = zip(*draw(st.lists(centre_bases(clusters, kinds),
                                         min_size=n, max_size=n)))
    stored = (np.array(factors)[:, None]
              * (np.array(points, dtype=np.float64) + JITTERS[:n])).astype(np.float32)
    centres = stored.astype(np.float64)
    assume(not has_tied_distances(centres))
    return centres


# Settings

# faint_min_paint_share is 0 or a normal float, from sys.float_info.min, about 2.2e-308, up to but
# not including 1, and never a subnormal one like 5e-324. faint_keep multiplies the share by a
# Cube_Paint that it has shifted to 1 or more, and a subnormal share makes that product underflow,
# which np.errstate(all="raise") in the filter tests turns into an error. The design doesn't allow
# for that underflow, but no real config uses such a share, so faint.py stays as designed.
paint_shares = (st.sampled_from((0.0, 0.10))
                | st.floats(min_value=sys.float_info.min, max_value=1, exclude_max=True))
# Valid values of the settings the filters and the limits read, as keyword arguments for
# clean_settings, like clean_settings(**values). Each one is in the range Req 6.7 or 6.8 gives
# it, and the config's own values and some ends of the ranges are drawn often.
valid_settings = st.fixed_dictionaries({
    # Hypothesis draws a float from 0 to 1 below 6.2e-6, the Real_Opacity of a stored opacity of
    # -12, so often that 45% of Property 4's draws held no Faint_Gaussian. A float from 0.001 to
    # 0.999, which it spreads more evenly, is drawn as often, and brought that down to about 20%.
    "min_opacity": (st.sampled_from((0.0, 0.005, 1.0)) | st.floats(min_value=0, max_value=1)
                    | st.floats(min_value=0.001, max_value=0.999)),
    "faint_cube_divisions": st.just(128) | st.integers(min_value=1, max_value=300),
    # Small counts, from 2 to 4, come often, so that clusters reach them and a Gaussian alone in
    # its Cube doesn't.
    "faint_min_per_cube": (st.just(8) | st.integers(min_value=2, max_value=4)
                           | st.integers(min_value=1, max_value=10)),
    "faint_min_paint_share": paint_shares,
    "extent_trim_percent": (st.sampled_from((0.0, 1.0))
                            | st.floats(min_value=0, max_value=50, exclude_max=True)),
    "neighbours": st.integers(min_value=1, max_value=10),
    "isolation_ratio": st.floats(min_value=0.5, max_value=20),
    "max_removed_fraction": st.just(0.7) | st.floats(min_value=0, max_value=1),
    "max_workers": st.integers(min_value=0, max_value=2),
})


# Splats for the extremes, for Property 8

# The values extreme_values draws on purpose, beside what st.floats draws: 3.4e38 and -3.4e38,
# from HUGE, and the smallest and the largest subnormal float32, 2**-149 and 2**-126 - 2**-149,
# each with either sign.
SUBNORMALS = (2.0 ** -149, 2.0 ** -126 - 2.0 ** -149)
EXTREMES = (*HUGE, *SUBNORMALS, *(-value for value in SUBNORMALS))
# float32's largest finite value, about 3.4028e38.
FLOAT32_MAX = float(np.finfo(np.float32).max)
# Any finite float32 value from st.floats, and one of EXTREMES one time in four. A float32 field
# stores 3.4e38 as the float32 value nearest it, and every other value exactly. st.floats is
# bounded at FLOAT32_MAX because, unbounded, it draws float64 values and drops each one too big
# for a float32, which made two draws of extreme_splats in three invalid.
extreme_values = mostly(st.floats(min_value=-FLOAT32_MAX, max_value=FLOAT32_MAX, width=32),
                        *EXTREMES)


@st.composite
def extreme_splats(draw: st.DrawFn) -> np.ndarray:
    """Splat_Data of 0 to MOST_GAUSSIANS Gaussians for the first half of Property 8, the empty set
    included, as a structured array with a float32 field for each property of standard_fields(),
    as gaussians gives. Its centres, stored opacities and stored scales are any finite float32
    values from extreme_values, so 3.4e38, -3.4e38 and the subnormals turn up in each. The
    quaternions and the Carried_Properties keep the plain values gaussians gives them.

    Unlike the other Splats strategies, it keeps a draw in which a Gaussian has two others at the
    same Centre_Distance. Req 10.13 keeps those out because the Isolated_Filter's answer can hang
    on the tree's order, but faint_keep searches for no neighbours, so a tie can't change its
    answer. Seen from a centre at 3e38 or 1e30, every centre near the origin is at one float64
    distance, so 911 of 1,000 of these draws had a tie, every one of more than 20 Gaussians
    among them. Dropping them failed Hypothesis's filter_too_much health check in all three runs
    tried.
    """
    n = draw(st.integers(min_value=0, max_value=MOST_GAUSSIANS))
    data = gaussians(n)
    for name in ("x", "y", "z", "opacity", "scale_0", "scale_1", "scale_2"):
        data[name] = draw(st.lists(extreme_values, min_size=n, max_size=n))
    return data


# Stored scales for the second half of Property 8: whole multiples of 1/4 from -8 to 3. And c,
# which goes on every one of them: a whole multiple of 1/4 up to 10**6 in size. Every scale + c is
# then a multiple of 1/4 below 2**22 in size, which float32 holds exactly. Above about 720 in
# size, c puts every Real_Scale outside float64's range. st.integers draws small sizes so often
# that only 1 draw in 7 had a c above 10**5 in size, so c is 10**6 or -10**6 one time in four.
quarter_scales = st.integers(min_value=-32, max_value=12).map(lambda k: k / 4)
scale_shifts = mostly(st.integers(min_value=-4 * 10 ** 6, max_value=4 * 10 ** 6)
                      .map(lambda k: k / 4), 1e6, -1e6)
# m, where every centre is multiplied by 2**m.
centre_powers = st.integers(min_value=-20, max_value=20)
# The smallest size of a centre coordinate that rescaled_splats draws. Each one is a lattice
# coordinate plus its index-based jitter plus this, so it's from 2**-10 to about 3.01 in size, and
# times 2**m it's a normal float32 value, exactly.
SMALLEST_COORDINATE = 2.0 ** -10
# Where a Gaussian of rescaled_splats has its centre before its jitter: a cluster six times in
# eight and a lattice point twice, as CLUSTERED_KINDS gives them but with no far outlier, whose
# coordinates can be far above 2**10.
NEAR_KINDS = ("lattice",) * 2 + ("cluster",) * 6


@dataclass(frozen=True)
class Rescaled:
    """A draw for the second half of Property 8: Splat_Data, and the same Gaussians with c added
    to every stored scale and every centre multiplied by 2**m."""

    data: np.ndarray
    moved: np.ndarray
    c: float
    m: int


@st.composite
def rescaled_splats(draw: st.DrawFn) -> Rescaled:
    """Splat_Data of 0 to MOST_GAUSSIANS Gaussians for the second half of Property 8, the empty
    set included, and the same Gaussians with c added to every stored scale and every centre
    multiplied by 2**m. Both are structured arrays with a float32 field for each property of
    standard_fields(), as gaussians gives.

    - Centres are lattice points and clusters, each moved by its index-based jitter and then by
      SMALLEST_COORDINATE on every axis, so each coordinate is from 2**-10 to 2**10 in size. None
      is 0: at a subnormal extent_trim_percent, a low corner interpolated up from a coordinate of
      0 can underflow to 0 at one scale and not at another, which moves a centre across a Cube
      face.
    - Stored opacities are any finite float32 values from extreme_values, and stored scales are
      from quarter_scales.
    - c is from scale_shifts and m from centre_powers.
    - The quaternions and the Carried_Properties keep the plain values gaussians gives them.

    Every moved value is checked to be the float32 value equal to the exact result. A draw is
    dropped with assume when a Gaussian has two others at the same Centre_Distance (Req 10.13).
    """
    n = draw(st.integers(min_value=0, max_value=MOST_GAUSSIANS))
    clusters = draw(st.lists(lattice_points, min_size=1, max_size=MOST_CLUSTERS))
    bases = draw(st.lists(centre_bases(clusters, NEAR_KINDS), min_size=n, max_size=n))
    # Reshaped, so the empty set gives empty columns too.
    points = np.array([point for point, _ in bases], dtype=np.float64).reshape(n, 3)
    centres = points + JITTERS[:n] + SMALLEST_COORDINATE
    data = gaussians(n)
    for axis, name in enumerate(("x", "y", "z")):
        data[name] = centres[:, axis]
    # Checked here, before the opacities and scales are drawn, so a dropped draw costs little.
    assume(not has_tied_distances(np.column_stack(
        [data[name].astype(np.float64) for name in ("x", "y", "z")])))
    data["opacity"] = draw(st.lists(extreme_values, min_size=n, max_size=n))
    for name in ("scale_0", "scale_1", "scale_2"):
        data[name] = draw(st.lists(quarter_scales, min_size=n, max_size=n))
    c = draw(scale_shifts)
    m = draw(centre_powers)
    moved = data.copy()
    for name in ("scale_0", "scale_1", "scale_2"):
        moved[name] = exactly_float32(data[name].astype(np.float64) + c)
    for name in ("x", "y", "z"):
        moved[name] = exactly_float32(data[name].astype(np.float64) * 2.0 ** m)
    return Rescaled(data, moved, c, m)


def exactly_float32(values: np.ndarray) -> np.ndarray:
    """`values`, a float64 array, as float32, after checking that float32 holds each one exactly,
    so a moved value never differs from the exact result."""
    stored = values.astype(np.float32)
    assert (stored.astype(np.float64) == values).all(), "float32 can't hold a moved value exactly"
    return stored


# Grids, for Property 6

# The most Gaussians a grid has along one axis.
GRID_SIDE = 6
# Every grid's numbers of Gaussians along x, y and z, each from 1 to GRID_SIDE, the smallest grids
# first, so Hypothesis shrinks to them.
GRID_SHAPES = sorted(itertools.product(range(1, GRID_SIDE + 1), repeat=3), key=math.prod)
# e, where a grid's spacing on one axis is 2**e.
spacing_powers = st.integers(min_value=-8, max_value=8)
# m, where a grid's origin on one axis is m times that axis's spacing. Each coordinate is then
# (m + j) * 2**e for a j from 0 to GRID_SIDE - 1: a whole number at most 2**20 + 5 in size, which
# float32 holds exactly, times a power of 2 that it holds exactly too.
origin_multiples = st.integers(min_value=-2 ** 20, max_value=2 ** 20)
# How far above the logit of min_opacity a grid's stored opacities start, so no rounding can make
# one a Faint_Gaussian.
OPACITY_MARGIN = 0.001
# Any finite float32 value, for a grid's stored scales. st.floats often draws its own bounds,
# about 3.4e38 and -3.4e38, and the subnormals.
finite_float32s = st.floats(min_value=-FLOAT32_MAX, max_value=FLOAT32_MAX, width=32)
# A grid's random unit quaternions are made from three numbers from 0 up to 1, each a whole
# multiple of 1 / TURN_STEPS.
TURN_STEPS = 1024


@st.composite
def grids(draw: st.DrawFn, *, min_opacity: float, neighbours: int) -> np.ndarray:
    """Splat_Data on an evenly spaced grid of more than `neighbours` Gaussians, for Property 6, as
    a structured array with a float32 field for each property of standard_fields(), as gaussians
    gives. `min_opacity` has to be above 0 and below 1, as the real config's is.

    - The grid has nx, ny and nz Gaussians along x, y and z, each from 1 to GRID_SIDE, with a
      product above `neighbours`. Its rows run along x first, then y, then z.
    - Each axis has its own spacing, 2**e for an e from spacing_powers, and its own origin, a
      whole multiple of that spacing from origin_multiples. Every coordinate is checked to be the
      float32 value equal to the exact one, so the grid really is even.
    - Stored opacities are float32 values from OPACITY_MARGIN above the logit of `min_opacity` up
      to float32's largest, so every Real_Opacity is above `min_opacity`.
    - Quaternions are random unit ones, by Shoemake's method: for u from 0 to 1 and any v and w,
      (sqrt(1 - u) sin 2πv, sqrt(1 - u) cos 2πv, sqrt(u) sin 2πw, sqrt(u) cos 2πw) has length 1,
      and it's a uniformly random rotation when u, v and w are uniform from 0 up to 1. Each of
      them is a whole multiple of 1 / TURN_STEPS, so every component is 0 or above 1e-18 in size,
      and storing it as float32 can't underflow.
    - Stored scales are any finite float32 values, from finite_float32s.
    - The Carried_Properties keep the plain values gaussians gives them.

    Each per-Gaussian value comes from Hypothesis's arrays, which draws a few of them and fills
    the rest with one more. Drawn one at a time, the values of a grid of 216 Gaussians would take
    more than the 8 KiB that Hypothesis lets one example draw.

    It has no tie check: a grid is full of Gaussians with two others at one Centre_Distance, and
    Property 6 has to hold whichever of them the tree counts (Req 10.13).
    """
    shape = draw(st.sampled_from([sides for sides in GRID_SHAPES if math.prod(sides) > neighbours]))
    lines = []  # each axis's coordinates, exactly, in float64
    for side in shape:
        spacing = 2.0 ** draw(spacing_powers)
        lines.append((draw(origin_multiples) + np.arange(side)) * spacing)
    # Every centre of the grid, x changing fastest, then y, then z.
    z, y, x = np.meshgrid(lines[2], lines[1], lines[0], indexing="ij")
    n = math.prod(shape)
    data = gaussians(n)
    for name, coordinates in (("x", x), ("y", y), ("z", z)):
        data[name] = exactly_float32(coordinates.ravel())
    lowest = float32_at_least(math.log(min_opacity / (1 - min_opacity)) + OPACITY_MARGIN)
    data["opacity"] = draw(arrays(np.float32, n, elements=st.floats(
        min_value=lowest, max_value=FLOAT32_MAX, width=32)))
    scales = draw(arrays(np.float32, (3, n), elements=finite_float32s))
    for axis in range(3):
        data[f"scale_{axis}"] = scales[axis]
    u, v, w = draw(arrays(np.int64, (3, n), elements=st.integers(
        min_value=0, max_value=TURN_STEPS - 1))) / TURN_STEPS
    turns = (np.sqrt(1 - u) * np.sin(2 * np.pi * v), np.sqrt(1 - u) * np.cos(2 * np.pi * v),
             np.sqrt(u) * np.sin(2 * np.pi * w), np.sqrt(u) * np.cos(2 * np.pi * w))
    for name, component in zip(("rot_0", "rot_1", "rot_2", "rot_3"), turns):
        data[name] = component
    return data


def float32_at_least(value: float) -> float:
    """The smallest float32 value at or above `value`, a float inside float32's range.

    st.floats with width=32 takes only bounds that float32 holds exactly. The comparison is made
    between Python floats, because numpy would round `value` to float32 to compare it with one.
    """
    nearest = np.float32(value)
    if float(nearest) < value:
        nearest = np.nextafter(nearest, np.float32(math.inf))
    return float(nearest)
