"""Level 11A: DETERMINISTIC security self-audit.

The system inspects itself. FINDINGS are deterministic (fixed by the SCRIPT, not the 14B model):
  1. Standard SAST if installed (ruff-security, bandit, pip-audit); otherwise HONEST DEGRADATION.
  2. Our own security INVARIANTS, verified over `src/` with `ast`/grep:
     - the guard covers ALL tools that mutate disk (complete `_TOOLS_MUTATE`);
     - every state-changing endpoint (POST/WS) has anti-CSWSH (`_origin_ok`) and uploads
       have a size cap;
     - tool writes are confined to the project (`_safe_path`);
     - skill detectors have a timeout (SIGALRM, anti-ReDoS);
     - no hardcoded secrets; `.gitignore` protects project data and temporaries;
     - the anti-ghost-execution defense exists (external content does not run tools).

The model only NARRATES the report as a security analyst; it does NOT change the verdict. READ-ONLY: it
does not modify any file. External tools are launched via subprocess with a TIMEOUT and go through the
guard (never run anything destructive).

HONEST: this is SAST + invariant verification, NOT a pentest nor discovery of novel vulnerabilities.
Its value: repeatable, automatic, catches security REGRESSIONS.
"""

import ast
import json
import re
import shutil
import subprocess  # nosec B404 - to launch SAST (bandit/ruff/pip-audit) via _run: fixed list, no shell, through the guard
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # src/ -> repo root
SAST_TIMEOUT = 45  # per tool; below run_bash's timeout (60s) for the skill

OK, WARN, FAIL = "OK", "WARN", "FAIL"

# Security INVARIANT: these tools WRITE to disk, so they must ALL be in agent._TOOLS_MUTATE
# (otherwise the anti-loop guard misses them and a fix could spiral/duplicate uncontrolled).
TOOLS_THAT_MUTATE = {
    "write_file",
    "edit_file",
    "create_document",
    "create_office_doc",
    "create_notebook",
    "edit_notebook",
    "convert_notebook",
    "generate_deliverable",
}
# Level 20 (H8): create_office_doc was missing, so it was neither checked to be in the guard nor to confine
# its path. A regression test also requires this set == agent._TOOLS_MUTATE in BOTH directions.

# Placeholders that are NOT real secrets (to avoid false positives in the secret search).
_PLACEHOLDERS = {"", "local", "not-needed", "changeme", "your-key", "xxx", "none"}


def _res(check, result, severity, evidence="", detail="", category="invariant"):
    return {
        "check": check,
        "result": result,
        "severity": severity,
        "evidence": evidence,
        "detail": detail,
        "category": category,
    }


# INVARIANT checks (pure/testable over source or values)
def check_mutating_tools_covered(mutating_tools):
    """The guard/anti-loop must cover ALL tools that mutate disk. FAIL if any is missing."""
    missing = sorted(TOOLS_THAT_MUTATE - set(mutating_tools))
    if missing:
        return _res(
            "_TOOLS_MUTATE covers all tools that write to disk",
            FAIL,
            "HIGH",
            "src/agent.py:_TOOLS_MUTATE",
            f"tools that mutate but are NOT in the guard: {missing}",
        )
    return _res(
        "_TOOLS_MUTATE covers all tools that write to disk",
        OK,
        "HIGH",
        "src/agent.py:_TOOLS_MUTATE",
        f"all {len(TOOLS_THAT_MUTATE)} writing tools are covered",
    )


def _defs_with_decorator(source, patterns):
    """Return [(name, decorator, source)] for functions whose decorator matches any of `patterns`
    (e.g. 'app.post', 'app.websocket'). Robust to syntax errors (returns [] if it does not parse)."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for dec in node.decorator_list:
            txt = ast.unparse(dec) if hasattr(ast, "unparse") else ""
            if any(p in txt for p in patterns):
                out.append((node.name, txt, ast.get_source_segment(source, node) or ""))
                break
    return out


def _calls(src, name):
    """Level 20 (H8): does the code `src` contain a real CALL to `name` (not the word in a comment or a
    string)? Deterministic via AST; if it does not parse, falls back to looking for "name(" outside
    comments."""
    import textwrap

    try:
        tree = ast.parse(textwrap.dedent(src))
    except SyntaxError:
        no_comments = "\n".join(line.split("#", 1)[0] for line in src.splitlines())
        return f"{name}(" in no_comments
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            if (isinstance(f, ast.Name) and f.id == name) or (isinstance(f, ast.Attribute) and f.attr == name):
                return True
    return False


def check_cswsh(server_source):
    """Every state-changing endpoint (POST) and the WebSocket must check the origin (anti-CSWSH).
    FAIL if any does not. Read-only GETs are listed as informational."""
    post = _defs_with_decorator(server_source, ["app.post", "app.put", "app.delete", "app.patch"])
    ws = _defs_with_decorator(server_source, ["app.websocket"])
    without_origin = []
    for name, _dec, src in post:
        if not _calls(src, "_origin_ok"):  # Level 20: a real CALL (not in a comment) = anti-CSWSH
            without_origin.append(name)
    for name, _dec, src in ws:
        if "origin" not in src.lower():  # the WS validates Origin by hand (closes 1008)
            without_origin.append(name)
    n = len(post) + len(ws)
    if without_origin:
        return _res(
            "Anti-CSWSH on all state-changing endpoints (POST/WS)",
            FAIL,
            "HIGH",
            "src/web_server.py",
            f"endpoints WITHOUT origin validation: {without_origin}",
        )
    return _res(
        "Anti-CSWSH on all state-changing endpoints (POST/WS)",
        OK,
        "HIGH",
        "src/web_server.py",
        f"{n} POST/WS endpoints, all validate the origin",
    )


def check_upload_limits(server_source):
    """Upload endpoints (/stt, /upload) must cap the SIZE (memory anti-DoS)."""
    defs = {n: s for n, _d, s in _defs_with_decorator(server_source, ["app.post"])}
    missing = [n for n, s in defs.items() if ("UploadFile" in s or "File(" in s) and "tope" not in s and "413" not in s]
    if missing:
        return _res(
            "Size cap on uploads (anti-DoS)", FAIL, "HIGH", "src/web_server.py", f"uploads without a cap: {missing}"
        )
    return _res(
        "Size cap on uploads (anti-DoS)",
        OK,
        "HIGH",
        "src/web_server.py",
        "uploads (/stt, /upload) cap the size (413/cap)",
    )


def check_path_safety(agent_source):
    """Every tool that writes to disk must CONFINE the path with `_safe_path` (no traversal or symlink
    outside the project). FAIL if any mutating tool does not."""
    try:
        tree = ast.parse(agent_source)
    except SyntaxError:
        return _res(
            "Writes confined to the project (_safe_path)", WARN, "MEDIUM", "src/agent.py", "could not parse agent.py"
        )
    unconfined = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in TOOLS_THAT_MUTATE:
            src = ast.get_source_segment(agent_source, node) or ""
            if not _calls(src, "_safe_path"):  # Level 20: a real CALL, not the word in a comment
                unconfined.append(node.name)
    if unconfined:
        return _res(
            "Writes confined to the project (_safe_path)",
            FAIL,
            "HIGH",
            "src/agent.py",
            f"writing tools without a confined path: {unconfined}",
        )
    return _res(
        "Writes confined to the project (_safe_path)", OK, "HIGH", "src/agent.py", "all writing tools use _safe_path"
    )


def check_regex_timeout(root):
    """Skill detectors (regex over external text) must have a timeout (SIGALRM, anti-ReDoS)."""
    detectors = ["skills/humanizer/detect.py", "skills/apa-citations/review_apa.py", "skills/spelling/review.py"]
    without_timeout = []
    for rel in detectors:
        p = root / rel
        try:
            txt = p.read_text(encoding="utf-8")
        except OSError:
            continue
        if "signal.alarm" not in txt and "SIGALRM" not in txt:
            without_timeout.append(rel)
    if without_timeout:
        return _res(
            "Skill detectors with a timeout (anti-ReDoS)",
            WARN,
            "MEDIUM",
            "skills/*/",
            f"detectors without SIGALRM: {without_timeout}",
        )
    return _res(
        "Skill detectors with a timeout (anti-ReDoS)",
        OK,
        "MEDIUM",
        "skills/*/",
        "detectors use SIGALRM (anti-ReDoS bound)",
    )


def check_secrets(root):
    """No hardcoded secrets in the code or in config.env.example. Grep for assignments to variables
    like *key/token/password/secret with a value that LOOKS real (not a placeholder)."""
    root = Path(root)
    # Level 20 (H8): also catches PREFIXED names (OPENAI_API_KEY, HF_TOKEN, db_password...) and real token
    # formats (OpenAI, Hugging Face, GitHub, AWS); before only the bare word and "sk-..." were caught.
    rx_assign = re.compile(r"""(?ix)(?<![a-z0-9])(?:[a-z0-9]+[_-])*(api[_-]?key|secret|passwd|password|token)
                               (?![a-z0-9])\s*[:=]\s*["']([^"']+)["']""")
    rx_openai = re.compile(
        r"\b(?:sk-[A-Za-z0-9_-]{20,}|hf_[A-Za-z0-9]{30,}|ghp_[A-Za-z0-9]{30,}|"
        r"github_pat_[A-Za-z0-9_]{30,}|AKIA[0-9A-Z]{16})\b"
    )
    findings = []
    for p in list((root / "src").glob("*.py")) + [root / "config.env.example"] + list(root.glob("*.sh")):
        try:
            for i, line in enumerate(p.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                m = rx_assign.search(line)
                if (
                    m
                    and m.group(2).strip().lower() not in _PLACEHOLDERS
                    and len(m.group(2).strip()) >= 8
                    and "cfg.get" not in line
                    and "getenv" not in line
                    and "environ" not in line
                ):
                    findings.append(f"{p.name}:{i}")
                if rx_openai.search(line):
                    findings.append(f"{p.name}:{i} (token in a real format: OpenAI/HF/GitHub/AWS)")
        except OSError:
            continue
    if findings:
        return _res(
            "No hardcoded secrets",
            FAIL,
            "HIGH",
            ", ".join(findings[:10]),
            "possible secrets in the code/config",
            category="secrets",
        )
    return _res(
        "No hardcoded secrets",
        OK,
        "HIGH",
        "src/*.py, config.env.example, *.sh",
        "keys are read from config.env (nothing hardcoded)",
        category="secrets",
    )


def check_gitignore(gitignore_text):
    """`.gitignore` must protect project data and personal temporaries/artifacts."""
    required = ["config.env", ".knowledge", ".snapshots", ".trash", "conversation.jsonl", "__pycache__", ".DS_Store"]
    missing = [r for r in required if r not in gitignore_text]
    if missing:
        return _res(
            ".gitignore protects project data and temporaries",
            FAIL,
            "HIGH",
            ".gitignore",
            f"missing patterns: {missing}",
            category="config",
        )
    return _res(
        ".gitignore protects project data and temporaries",
        OK,
        "HIGH",
        ".gitignore",
        "config.env, .knowledge, .snapshots, .trash, history... protected",
        category="config",
    )


def check_anti_ghost(agent_source):
    """The anti-ghost-execution defense must exist: a tool NARRATED in prose (or external content)
    must not run. We confirm the machinery (_looks_like_narration / fence detection) exists."""
    if "_looks_like_narration" in agent_source or "narrat" in agent_source:
        return _res(
            "Anti-ghost-execution defense (external content does not run tools)",
            OK,
            "MEDIUM",
            "src/agent.py:_looks_like_narration",
            "narrated tool-call detection exists",
        )
    return _res(
        "Anti-ghost-execution defense (external content does not run tools)",
        WARN,
        "MEDIUM",
        "src/agent.py",
        "could not locate the anti-narration machinery (review manually)",
    )


def check_pinned_requirements(requirements_text):
    """Dependencies must be PINNED (==) for reproducibility."""
    unpinned, deps = [], 0
    for line in (requirements_text or "").splitlines():
        s = line.split("#", 1)[0].strip()
        if s and not s.startswith("-") and re.match(r"^[A-Za-z0-9_.-]+", s):
            deps += 1
            if "==" not in s:
                unpinned.append(s)
    if deps == 0:  # Level 20 (H8): empty/unreadable is NOT "all pinned"
        return _res(
            "Dependencies pinned to a version (reproducibility)",
            WARN,
            "LOW",
            "requirements.txt",
            "found no dependencies to check (is requirements.txt missing?)",
            category="config",
        )
    if unpinned:
        return _res(
            "Dependencies pinned to a version (reproducibility)",
            WARN,
            "LOW",
            "requirements.txt",
            f"unpinned (==): {unpinned}",
            category="config",
        )
    return _res(
        "Dependencies pinned to a version (reproducibility)",
        OK,
        "LOW",
        "requirements.txt",
        "all dependencies are pinned with ==",
        category="config",
    )


# External SAST (subprocess with timeout + guard; honest degradation)
def _guard_ok(cmd):
    """The external command goes through the destructive guard (never run anything dangerous)."""
    try:
        import agent

        return not agent._RX_DESTRUCTIVE.search(" ".join(cmd))
    except Exception:  # noqa: BLE001 - if the guard cannot be imported, do not run (fail-closed)
        return False


def _resolve(tool):
    """Executable path of a tool: FIRST the project venv bin (where requirements-dev.txt pins it),
    then the system PATH. None if in neither."""
    venv_tool = ROOT / ".venv" / "bin" / tool
    if venv_tool.is_file():
        return str(venv_tool)
    return shutil.which(tool)


def _run(cmd):
    """Run a READ-ONLY external tool with a timeout, through the guard. Returns
    (returncode, stdout, stderr) or None if not installed / blocked / timed out."""
    exe = _resolve(cmd[0])
    if exe is None or not _guard_ok(cmd):
        return None
    try:
        r = subprocess.run(  # noqa: S603 - fixed argument list, no shell
            [exe] + cmd[1:],
            capture_output=True,
            text=True,  # noqa: S603 - fixed command (list, no shell), built in code and already validated by _guard_ok; exe resolved by _resolve  # nosec B603
            timeout=SAST_TIMEOUT,
            cwd=str(ROOT),
        )
        return r.returncode, r.stdout, r.stderr
    except (OSError, subprocess.TimeoutExpired):
        return None


def sast_ruff_security():
    """ruff security rules (flake8-bandit, `--select S`) over src/."""
    r = _run(["ruff", "check", "src", "--select", "S", "--output-format", "json"])
    if r is None:
        return _res(
            "SAST: ruff (security rules S)", WARN, "MEDIUM", "-", "ruff not available or blocked", category="sast"
        )
    rc, out, err = r
    if rc not in (0, 1) or not (out or "").strip():  # Level 20 (H8): ruff rc>=2 = a ruff ERROR, not "clean"
        return _res(
            "SAST: ruff (security rules S)",
            WARN,
            "MEDIUM",
            "-",
            f"ruff did not complete the analysis (rc={rc}): {(err or out or '').strip()[:120]}",
            category="sast",
        )
    try:
        warnings = json.loads(out)
    except json.JSONDecodeError:
        return _res(
            "SAST: ruff (security rules S)", WARN, "MEDIUM", "-", "could not parse ruff output", category="sast"
        )
    if warnings:
        ev = ", ".join(
            f"{a.get('filename', '?').split('/')[-1]}:{(a.get('location') or {}).get('row', '?')}[{a.get('code')}]"
            for a in warnings[:8]
        )
        return _res(
            "SAST: ruff (security rules S)",
            WARN,
            "MEDIUM",
            ev,
            f"{len(warnings)} ruff security warning(s) (review; many are low risk)",
            category="sast",
        )
    return _res(
        "SAST: ruff (security rules S)",
        OK,
        "MEDIUM",
        "src/",
        "ruff-S clean (intentional subprocess/shell are triaged with # noqa, visible in the code)",
        category="sast",
    )


def sast_bandit():
    r = _run(["bandit", "-r", "src", "-f", "json", "-q"])
    if r is None:
        return _res(
            "SAST: bandit (Python security linter)",
            WARN,
            "MEDIUM",
            "-",
            "bandit NOT installed: install `bandit` for this SAST (honest degradation)",
            category="sast",
        )
    rc, out, err = r
    if rc not in (0, 1) or not (out or "").strip():  # Level 20 (H8): no report = NOT analyzed
        return _res(
            "SAST: bandit (Python security linter)",
            WARN,
            "MEDIUM",
            "-",
            f"bandit did not complete the analysis (rc={rc}): {(err or '').strip()[:120]}",
            category="sast",
        )
    try:
        data = json.loads(out)
        if not isinstance(data, dict) or "results" not in data:
            raise json.JSONDecodeError("no 'results'", out, 0)
        issues = data.get("results", [])
    except json.JSONDecodeError:
        return _res(
            "SAST: bandit (Python security linter)",
            WARN,
            "MEDIUM",
            "-",
            "could not parse bandit output",
            category="sast",
        )
    severe = [x for x in issues if x.get("issue_severity") in ("HIGH", "MEDIUM")]
    if severe:
        ev = ", ".join(f"{x.get('filename', '?').split('/')[-1]}:{x.get('line_number', '?')}" for x in severe[:8])
        return _res(
            "SAST: bandit (Python security linter)",
            WARN,
            "MEDIUM",
            ev,
            f"{len(severe)} MEDIUM/HIGH bandit finding(s)",
            category="sast",
        )
    detail = (
        "bandit clean (intentional subprocess/shell are triaged with # nosec, visible in the code)"
        if not issues
        else f"bandit with no MEDIUM/HIGH findings ({len(issues)} low risk, review)"
    )
    return _res("SAST: bandit (Python security linter)", OK, "MEDIUM", "src/", detail, category="sast")


# Level 20: vulnerabilities ACCEPTED on purpose (with the reason). They are reported as WARN, not OK and not FAIL:
# the risk stays visible without hiding a new one. Review on every dependency update.
ACCEPTED_VULNS = {
    "setuptools": "torch requires setuptools<82; its issues affect installing packages, not the running app",
    "torch": "upgrading torch requires upgrading torchvision and vllm-mlx together; pending a coordinated update",
}


def _venv_site_packages():
    import glob

    paths = sorted(glob.glob(str(ROOT / ".venv" / "lib" / "python3*" / "site-packages")))
    return paths[-1] if paths else None


def sast_pip_audit():
    # Level 20: audit what is INSTALLED in the venv (~150 packages, including the transitive ones pulled by
    # vllm-mlx, transformers…), not only the ~20 pinned in requirements.txt: that is how 16 vulnerable ones showed up.
    sp = _venv_site_packages()
    target = ["--path", sp] if sp else ["-r", "requirements.txt"]
    r = _run(["pip-audit", *target, "--format", "json", "--progress-spinner", "off"])
    if r is None:
        return _res(
            "SAST: pip-audit (vulnerable dependencies)",
            WARN,
            "MEDIUM",
            "-",
            "pip-audit not available or no network (queries an ONLINE database): honest degradation",
            category="sast",
        )
    rc, out, err = r
    if rc not in (0, 1) or not (out or "").strip():  # Level 20 (H8): no network / error != "no vulnerabilities"
        return _res(
            "SAST: pip-audit (vulnerable dependencies)",
            WARN,
            "MEDIUM",
            "-",
            f"pip-audit did not complete the analysis (rc={rc}; no network?): {(err or '').strip()[:120]}",
            category="sast",
        )
    try:
        data = json.loads(out)
        deps = data.get("dependencies") if isinstance(data, dict) else data
        if not isinstance(deps, list) or not deps:
            raise AttributeError("report with no analyzed dependencies")
        vulnerable = [d for d in deps if d.get("vulns")]
    except (json.JSONDecodeError, AttributeError):
        return _res(
            "SAST: pip-audit (vulnerable dependencies)",
            WARN,
            "MEDIUM",
            "-",
            "could not parse pip-audit output (no network?)",
            category="sast",
        )
    new = [d for d in vulnerable if str(d.get("name", "")).lower() not in ACCEPTED_VULNS]
    accepted = [d for d in vulnerable if str(d.get("name", "")).lower() in ACCEPTED_VULNS]
    where = "the installed venv" if sp else "requirements.txt"
    if new:
        ev = ", ".join(f"{d.get('name')}=={d.get('version')}" for d in new[:8])
        return _res(
            "SAST: pip-audit (vulnerable dependencies)",
            FAIL,
            "HIGH",
            ev,
            f"{len(new)} dependency(ies) with known vulnerabilities in {where}",
            category="sast",
        )
    if accepted:
        ev = ", ".join(f"{d.get('name')}=={d.get('version')}" for d in accepted)
        reasons = "; ".join(f"{d.get('name')}: {ACCEPTED_VULNS[str(d.get('name')).lower()]}" for d in accepted)
        return _res(
            "SAST: pip-audit (vulnerable dependencies)",
            WARN,
            "MEDIUM",
            ev,
            f"only ACCEPTED risks remain, each with its reason ({reasons})",
            category="sast",
        )
    return _res(
        "SAST: pip-audit (vulnerable dependencies)",
        OK,
        "MEDIUM",
        where,
        f"no known vulnerabilities in {where}",
        category="sast",
    )


# Report orchestration
def _read_file(p):
    try:
        return (ROOT / p).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def audit(root=None):
    """Run ALL deterministic checks and return the structured report. READ-ONLY."""
    r = Path(root) if root else ROOT
    import agent

    agent_source = _read_file("src/agent.py")
    web_source = _read_file("src/web_server.py")
    checks = [
        # Our own invariants (the most distinctive part)
        check_mutating_tools_covered(agent._TOOLS_MUTATE),
        check_cswsh(web_source),
        check_upload_limits(web_source),
        check_path_safety(agent_source),
        check_regex_timeout(r),
        check_anti_ghost(agent_source),
        check_secrets(r),
        check_gitignore(_read_file(".gitignore")),
        check_pinned_requirements(_read_file("requirements.txt")),
        # Standard SAST (with honest degradation)
        sast_ruff_security(),
        sast_bandit(),
        sast_pip_audit(),
    ]
    summary = {OK: 0, WARN: 0, FAIL: 0}
    for c in checks:
        summary[c["result"]] += 1
    return {"checks": checks, "summary": summary, "verdict": FAIL if summary[FAIL] else (WARN if summary[WARN] else OK)}


_ICON = {OK: "OK", WARN: "WARN", FAIL: "FAIL"}


def format_report(res):
    """Deterministic table (check, result, severity, evidence). The SCRIPT fixes the verdict."""
    lines = [
        "# Security self-audit report (deterministic)\n",
        "> Generated by `src/self_audit.py`. Findings are fixed by the SCRIPT (SAST + invariant",
        "> verification); the model only narrates them. **This is SAST + invariants, NOT a pentest.**\n",
        f"**Verdict: {_ICON[res['verdict']]} {res['verdict']}** - "
        f"{res['summary'][OK]} OK, {res['summary'][WARN]} WARN, {res['summary'][FAIL]} FAIL\n",
        "| Check | Result | Severity | Evidence | Detail |",
        "|-------|--------|----------|----------|--------|",
    ]
    order = {FAIL: 0, WARN: 1, OK: 2}
    for c in sorted(res["checks"], key=lambda x: (order[x["result"]], x["category"])):
        det = (c["detail"] or "").replace("|", "\\|")
        lines.append(
            f"| {c['check']} | {_ICON[c['result']]} {c['result']} | {c['severity']} | `{c['evidence']}` | {det} |"
        )
    lines.append(
        "\n**Honest:** SAST that ran and SAST that was missing are listed above; a `WARN` is not "
        "necessarily a failure (many ruff rules are low risk). The value of this self-audit is "
        "to **catch security regressions** repeatably and automatically."
    )
    return "\n".join(lines)


def write_report(res, path=None):
    """Write the deterministic report to `self_audit_report.md` (repo root, regenerable, git-ignored). Returns the path."""
    path = Path(path) if path else (ROOT / "self_audit_report.md")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(format_report(res) + "\n", encoding="utf-8")
    return path


NARRATION_PROMPT = (
    "You are a senior security analyst. I am giving you the DETERMINISTIC REPORT of a self-audit of a "
    "local AI agent (SAST + verification of security invariants). Narrate it in 2-4 clear paragraphs "
    "for a technical reader: explain what was checked, what is good (confirmed invariants), what "
    "warnings there are and what they mean, and be HONEST about the limits (this is SAST + invariants, "
    "NOT a pentest). ABSOLUTE RULE: do NOT change the verdict or invent findings; use ONLY what the "
    "table says. Do not add recommendations that do not follow from a real finding."
)


def narrate(client, model, res):
    """The model NARRATES the deterministic report as an analyst (without changing the verdict).
    Best-effort: if the model is unavailable, returns '' and the deterministic report still stands."""
    try:
        from agent import chat

        ctx = [{"role": "system", "content": NARRATION_PROMPT + "\n\nREPORT:\n" + format_report(res)}]
        return chat(client, model, "Narrate this self-audit report as a security analyst.", previous_context=ctx)
    except Exception as e:  # noqa: BLE001 - narration is optional; the deterministic verdict rules
        return f"(model narration unavailable: {type(e).__name__})"
