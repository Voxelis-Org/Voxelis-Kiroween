"""The lines the website adds to the docs, the steering and the aws-jobs skill."""

from __future__ import annotations

import pytest

from tests.website_helpers import REPO_ROOT

VIDEOS = ("Kiro never opens or downloads anything under `uploads/` or under any job's `input/` in "
          "the job bucket: those are the judges' and users' videos. Kiro never needs them, because "
          "the status and download commands read only a job's `out/`.")
LIMITS = "at most 10 jobs and $52 a UTC day, and $500 in total, until 08:00 UTC on 4 February 2027"


def read(relative: str) -> str:
    return (REPO_ROOT / relative).read_text(encoding="utf-8")


@pytest.mark.parametrize("relative", [".kiro/steering/rules.md", ".kiro/skills/aws-jobs/SKILL.md"])
def test_the_rules_and_the_skill_hold_the_limits_the_one_yes_and_the_videos_line(relative: str) -> None:
    text = read(relative)
    assert VIDEOS in text
    assert LIMITS in text and "`APPROVED_SPEND=1` on the deploy command" in text
    assert "the team included" in text and "a job started with `cloud.start` included" in text
    assert "Kiro never runs them." in text


def test_the_setup_guide_documents_the_website() -> None:
    text = read("docs/aws-setup.md")
    for words in ("## The website", "echo $AWS_EXECUTION_ENV", "APPROVED_SPEND=1 python3.11 -m app.deploy",
                  "python3.11 -m app.account_command judges", "python3.11 -m app.showcase_command add",
                  "ParameterKey=WebsiteCodeKey,UsePreviousValue=true", "`WebsiteFunctionUrl`",
                  "Its table is kept", "share link", "| `WebsiteUrl` |"):
        assert words in text, words


def test_the_steering_and_the_licences_name_the_website() -> None:
    assert "`app/`" in read(".kiro/steering/structure.md")
    assert "Lambda function on Python 3.11" in read(".kiro/steering/tech.md")
    assert "comes later" not in read(".kiro/steering/product.md")
    assert "reservation" in read(".kiro/steering/cloud.md")
    licences = read("docs/third-party-licences.md")
    assert "| Python | PSF-2.0 |" in licences and "| pip | MIT |" in licences
    assert licences.count("The website bundle ships it unmodified to Lambda") == 7
    assert "app/" in read("README.md") and "Start_Core" in read("docs/pipeline.md")
