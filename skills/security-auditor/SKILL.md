---
name: security-auditor
when_to_use: security audit, audit yourself, audit your own security, run the self-audit, self-audit, check your own security, security report, review the system security, analyze the code security, check the security invariants, run the SAST, look for vulnerabilities in the system, agent security audit
script: audit.py
---

# Skill: security auditor (self-audit)

The system **audits itself** with DETERMINISTIC checks (SAST plus verification of security
invariants). The `audit.py` script produces the exact report (check, result, severity, evidence); you
**narrate** it as a security analyst. **Absolute rule: do NOT change the verdict and do NOT invent
findings.** Use ONLY what the injected report says.

## What it checks (deterministic)

- **Standard SAST** when the tools are installed (ruff-security, bandit, pip-audit). If one is
  missing, the report says so.
- **Our own invariants:** the guard covers every tool that writes to disk; state-changing endpoints
  have anti-CSWSH protection and an upload size cap; writes are confined to the project directory;
  skill detectors have a timeout (anti-ReDoS); no hardcoded secrets; `.gitignore` protects the data;
  the anti-ghost-execution defense is in place; requirements are pinned.

## How to narrate it

1. Give the **verdict** (OK / WARN / FAIL) exactly as it appears in the report.
2. Explain in 2-4 paragraphs what is healthy (confirmed invariants), what warnings came up and what
   they mean.
3. Be **honest** about scope: this is **SAST plus hand-written invariants, NOT a penetration test**,
   and it does not discover novel vulnerabilities. Its value is catching security **regressions**
   automatically and repeatably.
4. If a check is a **FAIL**, explain it clearly (which invariant broke), but never invent failures
   that are not in the table.
