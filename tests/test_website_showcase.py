"""The Showcase_Command and the Showcase entries' checks and credit lines."""

from __future__ import annotations

import io
from pathlib import Path

import pytest

from app.showcase import ShowcaseEntry, check_entry, licence_line
from app.showcase_command import showcase_main
from cloud.errors import CloudError
from tests.website_helpers import JOB_ID, REAL_CONFIG, T0, seed_job, website, website_config

CLOUDSHELL = {"AWS_EXECUTION_ENV": "CloudShell"}


@pytest.fixture
def deps(tmp_path: Path):
    deps = website(tmp_path)
    deps.aws.table = None
    seed_job(deps, account=None)
    return deps


def run(deps, *argv: str, environ=CLOUDSHELL) -> tuple[int, str]:
    out = io.StringIO()
    code = showcase_main(list(argv), environ=environ, make_aws=lambda region: deps.aws, out=out,
                         config_path=REAL_CONFIG, now=T0)
    return code, out.getvalue()


def test_add_list_and_remove(deps, capsys) -> None:
    assert run(deps, "add", JOB_ID, "--title", "The old mill", "--licence", "cc-by",
               "--credit", "Ada, example.org")[0] == 0
    assert deps.aws.bucket == "voxelis-kiroween-jobs-fake" and deps.aws.table == "table-fake"
    assert run(deps, "add", JOB_ID, "--title", "Again", "--licence", "own")[0] == 1
    assert "remove" in capsys.readouterr().err
    code, out = run(deps, "list")
    assert code == 0 and out == f"{JOB_ID}\tThe old mill\tcc-by\tAda, example.org\n"
    assert run(deps, "remove", JOB_ID)[0] == 0
    assert run(deps, "remove", JOB_ID)[0] == 1


def test_only_a_finished_job_with_a_splat_goes_in(deps, capsys) -> None:
    seed_job(deps, "20261202-120000-abcdef", account=None, status="failed", files=())
    assert run(deps, "add", "20261202-120000-abcdef", "--title", "x", "--licence", "own")[0] == 1
    assert "hasn't finished" in capsys.readouterr().err
    seed_job(deps, "20261203-120000-abcdef", account=None, files=("cameras.json",))
    assert run(deps, "add", "20261203-120000-abcdef", "--title", "x", "--licence", "own")[0] == 1
    assert "no splat" in capsys.readouterr().err


def test_the_command_refuses_outside_cloudshell(deps, capsys) -> None:
    assert run(deps, "list", environ={})[0] == 1
    assert "only in AWS CloudShell" in capsys.readouterr().err and deps.aws.calls == []


@pytest.mark.parametrize(("args", "words"), [
    (("bad-id", "t", "own", None), "isn't"), ((JOB_ID, "", "own", None), "title"),
    ((JOB_ID, "x" * 81, "own", None), "title"), ((JOB_ID, "a\nb", "own", None), "title"),
    ((JOB_ID, "t", "gpl", None), "licence"), ((JOB_ID, "t", "cc-by-sa", None), "credit"),
    ((JOB_ID, "t", "own", "c" * 201), "credit")])
def test_check_entry_refusals(args, words) -> None:
    with pytest.raises(CloudError, match=words):
        check_entry(*args, website_config(), T0)


def test_the_credit_lines() -> None:
    def entry(licence: str, credit: str | None) -> ShowcaseEntry:
        return ShowcaseEntry(JOB_ID, "t", licence, credit, 0)

    assert licence_line(entry("cc-by", "Ada")) == "Footage: Ada, CC BY"
    assert licence_line(entry("cc-by-sa", "Ada")) == (
        "Footage: Ada, CC BY-SA. This splat is shared under CC BY-SA 4.0.")
    assert licence_line(entry("own", None)) is None
    assert licence_line(entry("pexels", "Bo")) == "Footage: Bo"
