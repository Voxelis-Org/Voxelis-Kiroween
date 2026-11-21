"""The Template_Test: cfn-lint finds no error or warning in cloud/aws-setup.yaml (Req 17.1 to 17.3, A6).

cfn-lint reads the template and its own bundled resource schemas. It makes no AWS call, so the test
needs no network and no credentials (Req 17.2).
"""

import re
import tomllib
from pathlib import Path

from cfnlint.api import ManualArgs, lint_file

ROOT = Path(__file__).resolve().parents[1]  # this file is in tests/, at the repo root
TEMPLATE = ROOT / "cloud" / "aws-setup.yaml"
REAL_CONFIG = ROOT / "config.toml"  # read by path, like the other tests of the real config
# A cfn-lint key in a Metadata section, which would turn checks off. A YAML comment that mentions
# cfn-lint starts with # and doesn't match.
CFN_LINT_KEY = re.compile(r"""^\s*["']?cfn-lint["']?\s*:""", re.MULTILINE)


def test_cfn_lint_finds_no_error_or_warning_in_the_stack_template() -> None:
    region = tomllib.loads(REAL_CONFIG.read_text(encoding="utf-8"))["aws"]["region"]
    assert not CFN_LINT_KEY.search(TEMPLATE.read_text(encoding="utf-8")), (
        "cloud/aws-setup.yaml turns cfn-lint checks off with a cfn-lint Metadata key. "
        "Remove it and fix the findings instead (Req 17.3).")
    matches = lint_file(TEMPLATE, config=ManualArgs(regions=[region], mandatory_checks=["E", "W"]))
    findings = [f"{match.rule.id} line {match.linenumber}: {match.message}"
                for match in matches if match.rule.severity in ("error", "warning")]
    assert not findings, ("cfn-lint found these errors and warnings in cloud/aws-setup.yaml:\n"
                          + "\n".join(findings))
