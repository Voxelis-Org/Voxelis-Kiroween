"""Tests for the Account_Redaction, cloud/redact.py (cloud-jobs task 2.5).

`account_redaction` keeps AWS account ids out of chats, repo logs and data/ by replacing every
maximal run of exactly 12 digits with `<account>`, leaving a run of any other length and every
non-digit character alone (Req 13.6, 13.7, 1.9, 10.10).

Property 1 (the Hypothesis test at the end) checks the rule for any string against an independent
oracle: it finds the maximal digit runs a different way and replaces only the length-12 ones, then
asserts the helper matches, is idempotent, and leaves no maximal 12-digit run behind. The three
example tests pin the shapes the task names: a real-shaped ARN's account field, an 11- and a
13-digit number that must survive, and a 12-digit number that is still replaced. These run on the
CPU with no network and no AWS: redaction is pure string work.
"""

import re

from hypothesis import given
from hypothesis import strategies as st

from cloud.redact import account_redaction

# A maximal run of digits, as the re module sees `\d` (Unicode category Nd, the same class the
# helper's pattern uses). The oracle and the "no 12-run left" check reuse this, so both agree with
# the helper on what a digit is, whatever the alphabet.
_DIGIT_RUN = re.compile(r"\d+")


def expected_redaction(text: str) -> str:
    """An independent restatement of the rule: replace each maximal digit run of length exactly 12
    with `<account>`, keep every other run and every non-digit character verbatim.

    This uses a plain `\\d+` scan and an explicit length test, not the helper's `\\d{12}` lookaround
    substitution, so matching it is real evidence the helper is right.
    """
    def one(match: "re.Match[str]") -> str:
        run = match.group()
        return "<account>" if len(run) == 12 else run

    return _DIGIT_RUN.sub(one, text)


# Example tests

def test_a_real_shaped_arns_account_field_becomes_account() -> None:
    arn = "arn:aws:iam::123456789012:role/kiroween-runner"
    assert account_redaction(arn) == "arn:aws:iam::<account>:role/kiroween-runner"


def test_an_eleven_and_a_thirteen_digit_number_survive_unchanged() -> None:
    assert account_redaction("12345678901") == "12345678901"        # 11 digits
    assert account_redaction("1234567890123") == "1234567890123"    # 13 digits
    both = "id 12345678901 and 1234567890123 end"
    assert account_redaction(both) == both


def test_a_twelve_digit_number_that_is_not_an_account_id_is_still_replaced() -> None:
    assert account_redaction("invoice 987654321098 paid") == "invoice <account> paid"


# Property 1: Account_Redaction clears exactly 12-digit runs (Req 13.6, 13.7, 1.9, 10.10)

# Separators between digit runs: clearly no `\d` characters, so a separator never joins two runs
# into one. A mix of letters, punctuation, whitespace and a couple of non-ASCII letters.
SEP_ALPHABET = "abcdefXYZ .,:/-_=@+#\n\t\u00e7\u03bb"


def sep_text(min_size: int = 0) -> st.SearchStrategy[str]:
    """Non-digit filler of up to a few characters, drawn from SEP_ALPHABET."""
    return st.text(alphabet=SEP_ALPHABET, min_size=min_size, max_size=6)


@st.composite
def digit_run(draw: st.DrawFn) -> str:
    """A run of ASCII digits whose length is over-sampled around 12, so both the replace branch
    (exactly 12) and the keep branch (11, 13 and others) fire often."""
    length = draw(st.one_of(st.sampled_from([11, 12, 13]), st.integers(min_value=1, max_value=24)))
    return draw(st.text(alphabet="0123456789", min_size=length, max_size=length))


@st.composite
def strings_with_digit_runs(draw: st.DrawFn) -> str:
    """A string built from 1-6 digit runs kept apart by non-empty non-digit separators, with
    optional filler at each edge. Consecutive runs stay separate, so a length-12 run really is a
    maximal 12-digit run; the edge filler may be empty, so a run may sit at the string's edge."""
    runs = draw(st.lists(digit_run(), min_size=1, max_size=6))
    pieces = [draw(sep_text())]  # leading filler, may be empty
    for index, run in enumerate(runs):
        pieces.append(run)
        if index < len(runs) - 1:
            pieces.append(draw(sep_text(min_size=1)))  # keep the next run a separate maximal run
    pieces.append(draw(sep_text()))  # trailing filler, may be empty
    return "".join(pieces)


# Both branches: strings engineered to carry 11/12/13-digit runs, and fully arbitrary text.
redaction_inputs = st.one_of(strings_with_digit_runs(), st.text())


@given(redaction_inputs)
def test_property_1_account_redaction_clears_exactly_twelve_digit_runs(text: str) -> None:
    """Feature: cloud-jobs, Property 1: Account_Redaction clears exactly 12-digit runs

    **Validates: Requirements 13.6, 13.7, 1.9, 10.10**
    """
    result = account_redaction(text)
    # Every maximal 12-digit run becomes <account>; every other-length run and every non-digit
    # character is unchanged (matching the independent oracle covers all three at once).
    assert result == expected_redaction(text)
    # Idempotent: redacting again changes nothing.
    assert account_redaction(result) == result
    # So the result holds no maximal 12-digit run.
    assert all(len(match.group()) != 12 for match in _DIGIT_RUN.finditer(result))
