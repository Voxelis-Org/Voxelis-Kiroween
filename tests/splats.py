"""Shared helpers for the clean step's tests.

clean_settings builds a CleanSettings straight from the test config's [clean], the dicts
CLEAN_SHARED and CLEAN_PRESETS in tests/conftest.py, for the tests of pure functions, which need
no config file and no Job.

The Names strategies are for Property 10 in tests/test_clean_limits.py: names gives any text and
good file names with one character put in, and three_names gives good names for train_ply,
clean_ply and clean_sog that are often the same name in a different case.
"""

from hypothesis import strategies as st

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
