"""Tests for the pipeline errors and the CLI wrapper, splatpipe/errors.py."""

from collections.abc import Callable

import pytest

from splatpipe.errors import ExpectedFailure, PipelineError, run_cli

Capsys = pytest.CaptureFixture[str]
REFUSAL = "Only MP4 and MOV videos are accepted. Save the video as MP4 or MOV and try again."
STEP_FAILURE = "Too few sharp frames were found. Film the place again and move the phone more slowly."
BUG_MESSAGE = "splatpipe stopped because of a bug (ValueError: boom). Tell the team the command you ran."


def raising(error: BaseException) -> Callable[[], int]:
    """A CLI body that raises the error instead of returning an exit code."""
    def action() -> int:
        raise error
    return action


def run(action: Callable[[], int], capsys: Capsys) -> tuple[int, str, str]:
    """Run the action through run_cli. Give back the exit code and what was printed on stdout and stderr."""
    code = run_cli(action)
    captured = capsys.readouterr()
    return code, captured.out, captured.err


@pytest.mark.parametrize("code", [0, 3])
def test_an_action_gives_its_own_exit_code_and_prints_nothing(code: int, capsys: Capsys) -> None:
    assert run(lambda: code, capsys) == (code, "", "")


def test_a_pipeline_error_prints_only_its_message_on_stderr_and_gives_1(capsys: Capsys) -> None:
    code, out, err = run(raising(PipelineError(REFUSAL)), capsys)
    assert (code, out, err) == (1, "", REFUSAL + "\n")
    assert "Traceback" not in err


def test_a_value_error_prints_the_bug_message_and_gives_1(capsys: Capsys) -> None:
    code, out, err = run(raising(ValueError("boom")), capsys)
    assert (code, out, err) == (1, "", BUG_MESSAGE + "\n")
    assert "Traceback" not in err


def test_an_expected_failure_is_a_pipeline_error(capsys: Capsys) -> None:
    assert issubclass(ExpectedFailure, PipelineError)
    assert run(raising(ExpectedFailure(STEP_FAILURE)), capsys) == (1, "", STEP_FAILURE + "\n")


@pytest.mark.parametrize("error", [SystemExit(2), KeyboardInterrupt()],
                         ids=["SystemExit", "KeyboardInterrupt"])
def test_system_exit_and_keyboard_interrupt_are_raised_again(error: BaseException, capsys: Capsys) -> None:
    with pytest.raises(type(error)) as caught:
        run_cli(raising(error))
    assert caught.value is error
    assert capsys.readouterr().err == ""
