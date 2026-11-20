"""Tests for the Spend_Guard, cloud/spend.py.

No test here sets APPROVED_SPEND in the environment. A call that leaves `environ` as None, so the
guard reads os.environ, passes a temporary config that names SPLATPIPE_TEST_SPEND instead, and the
test sets or removes that variable through monkeypatch. So an APPROVED_SPEND left in a
developer's shell can't change a result. Every other call passes its own mapping as `environ`.
"""

import os
from pathlib import Path
from types import MappingProxyType

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from cloud.spend import SpendRefused, check_spend

# The Spend_Variable the temporary config names, in place of APPROVED_SPEND.
TEST_VARIABLE = "SPLATPIPE_TEST_SPEND"
# The run the tests describe to the guard, as check_spend's description arguments.
RUN = {"machine": "g5.4xlarge", "expected_time": "about 40 minutes", "rough_cost": "about 2 USD"}


# Unit tests

@pytest.fixture
def spend_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Write a config whose [aws].spend_variable is TEST_VARIABLE, and remove TEST_VARIABLE from
    os.environ, so every test starts with it unset. Gives the config's path."""
    monkeypatch.delenv(TEST_VARIABLE, raising=False)
    path = tmp_path / "spend.toml"
    path.write_text(f'[aws]\nspend_variable = "{TEST_VARIABLE}"\n', encoding="utf-8")
    return path


# The design's refusal for RUN when TEST_VARIABLE isn't exactly "1".
REFUSAL = ("Refused to start g5.4xlarge for about 40 minutes (about 2 USD): this run costs money "
           "and SPLATPIPE_TEST_SPEND isn't set to exactly 1. Get a yes in the chat for that "
           "machine, time and cost, then run this one command again with SPLATPIPE_TEST_SPEND=1.")


@pytest.mark.parametrize("environ", [{}, {TEST_VARIABLE: "yes"}], ids=["unset", "yes"])
def test_the_refusal_gives_the_run_the_variable_and_what_to_do(
        spend_config: Path, environ: dict[str, str]) -> None:
    with pytest.raises(SpendRefused) as refused:
        check_spend(**RUN, environ=environ, config_path=spend_config)
    assert str(refused.value) == REFUSAL


def test_the_variable_the_config_names_is_honoured(
        spend_config: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(TEST_VARIABLE, "1")
    assert check_spend(**RUN, config_path=spend_config) is None
    monkeypatch.delenv(TEST_VARIABLE)
    with pytest.raises(SpendRefused, match=TEST_VARIABLE):
        check_spend(**RUN, config_path=spend_config)


# Each config the guard can't read the Spend_Variable's name from, as the file's text (None for
# no file), and the words its refusal must hold besides the file's path.
BAD_CONFIGS = [
    pytest.param(None, "can't be opened", id="missing file"),
    pytest.param(f"[aws]\nspend_variable = {TEST_VARIABLE}\n", "can't be read as TOML",
                 id="broken TOML"),
    pytest.param('[git]\nmain_branch = "main"\n', "[aws] needs spend_variable",
                 id="no aws section"),
    pytest.param('[aws]\nregion = "us-east-1"\n', "[aws] needs spend_variable",
                 id="no spend_variable"),
    pytest.param('[aws]\nspend_variable = ""\n', "[aws] needs spend_variable", id="empty name"),
]


@pytest.mark.parametrize(("text", "words"), BAD_CONFIGS)
def test_a_config_without_a_variable_name_is_refused(
        tmp_path: Path, text: str | None, words: str) -> None:
    path = tmp_path / "spend.toml"
    if text is not None:
        path.write_text(text, encoding="utf-8")
    with pytest.raises(SpendRefused) as refused:
        check_spend(**RUN, environ={TEST_VARIABLE: "1"}, config_path=path)
    assert str(path) in str(refused.value)
    assert words in str(refused.value)


@pytest.mark.parametrize(("part", "label"), [("machine", "machine"),
                                             ("expected_time", "expected time"),
                                             ("rough_cost", "rough cost")])
def test_a_description_with_an_empty_part_is_refused(
        spend_config: Path, part: str, label: str) -> None:
    with pytest.raises(SpendRefused, match=f"its description has no {label},"):
        check_spend(**{**RUN, part: ""}, environ={TEST_VARIABLE: "1"}, config_path=spend_config)


def test_os_environ_is_the_same_after_allowed_and_refused_calls(
        spend_config: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(TEST_VARIABLE, "1")
    before = dict(os.environ)
    check_spend(**RUN, config_path=spend_config)
    assert dict(os.environ) == before

    monkeypatch.delenv(TEST_VARIABLE)
    before = dict(os.environ)
    with pytest.raises(SpendRefused):
        check_spend(**RUN, config_path=spend_config)
    assert dict(os.environ) == before


def test_a_read_only_environ_is_never_written(spend_config: Path) -> None:
    """Writing to a MappingProxyType raises TypeError, so these calls show that the guard only
    reads the mapping it's given, when it allows the run and when it refuses it."""
    allowed = MappingProxyType({TEST_VARIABLE: "1"})
    assert check_spend(**RUN, environ=allowed, config_path=spend_config) is None
    with pytest.raises(SpendRefused):
        check_spend(**RUN, environ=MappingProxyType({}), config_path=spend_config)


# Property tests

# The real config.toml, read by its path from the repo root. It's only read, and it names
# APPROVED_SPEND, which this test puts only in its own mapping, never in os.environ.
REAL_CONFIG = Path(__file__).resolve().parents[1] / "config.toml"
REAL_VARIABLE = "APPROVED_SPEND"
# st.text() can't give lone surrogates in Hypothesis 6.170 (checked); this one can, like "\ud800".
any_text = st.text() | st.text(st.characters(codec=None, exclude_categories=()))
# None leaves the Spend_Variable unset. The near misses include a fullwidth 1 and an Arabic-Indic
# 1, "\uff11" and "\u0661".
spend_values = st.none() | st.text() | any_text | st.sampled_from(
    ["1", " 1", "1 ", "1\n", "\n1", "01", "1.0", "+1", "\uff11", "\u0661", "true", "yes", "", "0"])


@settings(deadline=None)
@given(spend_values)
def test_the_spend_guard_allows_only_exactly_1(value: str | None) -> None:
    """Feature: pipeline-core, Property 6: The Spend_Guard allows only exactly "1"

    **Validates: Requirements 11.4, 11.5, 11.8**
    """
    environ = {} if value is None else {REAL_VARIABLE: value}
    if value == "1":
        assert check_spend(**RUN, environ=environ, config_path=REAL_CONFIG) is None
        return
    with pytest.raises(SpendRefused) as refused:
        check_spend(**RUN, environ=environ, config_path=REAL_CONFIG)
    for part in (*RUN.values(), REAL_VARIABLE):
        assert part in str(refused.value)
