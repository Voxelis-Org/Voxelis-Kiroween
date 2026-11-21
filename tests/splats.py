"""Shared helpers for the clean step's tests.

clean_settings builds a CleanSettings straight from the test config's [clean], the dicts
CLEAN_SHARED and CLEAN_PRESETS in tests/conftest.py, for the tests of pure functions, which need
no config file and no Job.
"""

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
