"""Tests for the devlog hook, scripts/hooks/devlog.py."""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DEVLOG_PATH = ROOT / "scripts" / "hooks" / "devlog.py"
_spec = importlib.util.spec_from_file_location("devlog", DEVLOG_PATH)
devlog = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(devlog)


@pytest.mark.parametrize("spec_name", [
    "d:\\Voxelis-Kiroween\\.kiro\\specs\\pipeline-core\\tasks.md",
    "/home/sam/Voxelis-Kiroween/.kiro/specs/pipeline-core/tasks.md",
    ".kiro/specs/pipeline-core",
    ".kiro\\specs\\pipeline-core\\",
    "pipeline-core",
])
def test_names_the_spec_by_its_folder(spec_name: str) -> None:
    event = {"spec_name": spec_name, "task_name": "2.3 Write the tests"}
    assert devlog.task_text(event) == "pipeline-core 2.3 Write the tests"


def test_without_a_spec_name_it_gives_the_task_alone() -> None:
    assert devlog.task_text({"task_name": "2.3 Write the tests"}) == "2.3 Write the tests"


def test_a_failed_task_says_so() -> None:
    event = {"spec_name": "pipeline-core", "task_name": "2.3 Write the tests", "task_success": False}
    assert devlog.task_text(event) == "pipeline-core 2.3 Write the tests (failed)"


@pytest.mark.parametrize("event", [None, [], {}, {"spec_name": 3}])
def test_without_a_task_it_says_task_finished(event: object) -> None:
    assert devlog.task_text(event) == "task finished"
