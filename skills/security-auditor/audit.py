#!/usr/bin/env python3
"""Script for the 'security-auditor' skill: runs the DETERMINISTIC security self-audit and prints the
report (a table of check, result, severity, evidence). It ignores the user's text: the audit is about
the system itself (src/). The model NARRATES this report without changing the verdict. READ-ONLY."""
import sys
from pathlib import Path

# the app code lives in src/ (this script lives in skills/security-auditor/)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

import self_audit  # noqa: E402


def main():
    res = self_audit.audit()          # deterministic checks; the script fixes the verdict
    print(self_audit.format_report(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
