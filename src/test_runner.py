#!/usr/bin/env python3
# test_runner.py: Level 20 (H2/H6/R6). ONE DETERMINISTIC test runner for the whole system.
#
# Before there were two paths and both gave false passes:
#   - worker.verify_python_test trusted pytest's `returncode == 0`, and pytest exits with 0 when EVERY
#     test is @skip or @xfail, so a BROKEN module with its test skipped came out as "the test PASSES".
#   - the orchestrator's "do the tests pass?" check only used `unittest discover`, which does NOT see
#     pytest-style tests (`def test_x(): assert ...`), so "0 tests" and the "don't break what already
#     worked" safeguard never kicked in.
#
# Now: pytest with a JUnit XML report (a REAL per-test count, not free text), and it PASSES only if at least
# 1 test ran and NONE failed, errored, was skipped or was xfail. Anything odd -> it does NOT pass (honest).
# Also, the tests run in their OWN process group and the whole group is killed at the end: a test that
# leaves a background process behind (or hangs) leaves no orphans (R6).
import os
import signal
import subprocess  # noqa: S404 - interpreter + pytest with fixed arguments, no shell  # nosec B404
import sys
import tempfile
import xml.etree.ElementTree as ET  # noqa: S405 - only parses the XML that OUR pytest writes  # nosec B405
from pathlib import Path

# This module's name matches pytest's "test_*.py" pattern: tell pytest it holds no tests.
__test__ = False

_NO_PROXY = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "FTP_PROXY")


def _environment(env=None, path_extra=None):
    base = dict(os.environ if env is None else env)
    base = {k: v for k, v in base.items() if k.upper() not in _NO_PROXY}
    if path_extra:
        base["PATH"] = f"{path_extra}{os.pathsep}{base.get('PATH', '')}"
    return base


def _run_in_group(cmd, cwd, env, timeout):
    """Runs `cmd` in a NEW process group and, whatever happens, kills the group when done.
    Returns (returncode, output, timed_out). returncode=None on timeout."""
    proc = subprocess.Popen(  # noqa: S603 - fixed list built in code, no shell  # nosec B603
        cmd,
        cwd=str(cwd),
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    timed_out = False
    try:
        output, _ = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        output = b""
    finally:
        try:
            os.killpg(proc.pid, signal.SIGKILL)  # the whole group: children and grandchildren the test left
        except (ProcessLookupError, PermissionError):
            pass
        if timed_out:
            try:
                output, _ = proc.communicate(timeout=5)
            except Exception:  # noqa: BLE001 - best effort, it is already dead, just collect what is there
                output = output or b""
    text = (output or b"").decode("utf-8", errors="replace")
    return (None if timed_out else proc.returncode), text, timed_out


def _junit_count(xml_path):
    """{total, failed, errors, skipped} summing every <testsuite>. None if there is no readable XML."""
    try:
        root = ET.parse(xml_path).getroot()  # noqa: S314 - local XML just written by OUR pytest in our own tmp  # nosec B314
    except (ET.ParseError, OSError):
        return None
    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    if not suites:
        return None
    r = {"total": 0, "failed": 0, "errors": 0, "skipped": 0}
    for s in suites:
        r["total"] += int(s.get("tests", 0) or 0)
        r["failed"] += int(s.get("failures", 0) or 0)
        r["errors"] += int(s.get("errors", 0) or 0)
        r["skipped"] += int(s.get("skipped", 0) or 0)  # pytest counts both skip AND xfail here
    return r


def _result(ok, total=0, passed=0, failed=0, errors=0, skipped=0, timeout=False, no_pytest=False, summary=""):
    return {
        "ok": bool(ok),
        "total": total,
        "passed": passed,
        "failed": failed,
        "errors": errors,
        "skipped": skipped,
        "timeout": timeout,
        "no_pytest": no_pytest,
        "summary": summary,
    }


def run_pytest(cwd, files=None, timeout=60, python=None, env=None, path_extra=None):
    """Runs pytest in a BOUNDED way and returns a DETERMINISTIC dict:
        ok, total, passed, failed, errors, skipped, timeout, no_pytest, summary
    `ok` is True ONLY if: it finished in time, >=1 test ran, and 0 failed/errors/skipped(+xfail).
    `no_pytest=True` if the interpreter has no pytest (the caller decides whether to fall back to unittest)."""
    python = python or sys.executable
    with tempfile.TemporaryDirectory(prefix="test_runner_") as tmp:
        xml = Path(tmp) / "junit.xml"
        cmd = [
            python,
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            "-o",
            "addopts=",
            f"--junitxml={xml}",
            *(files or []),
        ]
        try:
            rc, output, timed_out = _run_in_group(cmd, cwd, _environment(env, path_extra), timeout)
        except OSError as e:
            return _result(False, summary=f"could not run pytest ({type(e).__name__})")
        if timed_out:
            return _result(False, timeout=True, summary=f"pytest did not finish in {timeout}s; cancelled")
        if "No module named pytest" in output:
            return _result(False, no_pytest=True, summary="pytest is not installed")
        count = _junit_count(xml)
    if count is None:
        tail = (output.strip().splitlines() or ["(no output)"])[-1][:120]
        return _result(False, summary=f"pytest produced no report (rc={rc}): {tail}")
    passed = count["total"] - count["failed"] - count["errors"] - count["skipped"]
    ok = (
        rc == 0
        and count["total"] >= 1
        and passed >= 1
        and count["failed"] == 0
        and count["errors"] == 0
        and count["skipped"] == 0
    )
    parts = [f"{passed} passed"]
    for key, label in (("failed", "failed"), ("errors", "with errors"), ("skipped", "skipped/xfail")):
        if count[key]:
            parts.append(f"{count[key]} {label}")
    if count["total"] == 0:
        parts = ["0 tests run"]
    return _result(ok, passed=max(passed, 0), summary=", ".join(parts), **count)


def run_unittest(cwd, timeout=60, python=None, env=None, path_extra=None):
    """Plan B when there is no pytest: `unittest discover`. Same contract as run_pytest. unittest gives no
    reliable skipped count as text, so we require "OK" WITHOUT "skipped" in the final line."""
    import re

    python = python or "python3"
    try:
        rc, output, timed_out = _run_in_group(
            [python, "-m", "unittest", "discover"], cwd, _environment(env, path_extra), timeout
        )
    except OSError as e:
        return _result(False, summary=f"could not run unittest ({type(e).__name__})")
    if timed_out:
        return _result(False, timeout=True, summary=f"unittest did not finish in {timeout}s; cancelled")
    m = re.search(r"Ran (\d+) test", output)
    total = int(m.group(1)) if m else 0
    skipped = sum(int(x) for x in re.findall(r"skipped=(\d+)", output))
    expected_failures = sum(int(x) for x in re.findall(r"expected failures=(\d+)", output))
    ok = rc == 0 and total >= 1 and skipped == 0 and expected_failures == 0
    return _result(
        ok,
        total=total,
        passed=total if ok else 0,
        skipped=skipped + expected_failures,
        summary=f"unittest: {total} tests" + ("" if ok else f" (rc={rc})"),
    )


def project_tests_pass(cwd, timeout=60, python=None, path_extra=None):
    """Do the project's tests REALLY pass? pytest (which also collects unittest tests) and, if it is not
    installed, `unittest discover`. Returns the result dict."""
    r = run_pytest(cwd, timeout=timeout, python=python, path_extra=path_extra)
    if r["no_pytest"]:
        r = run_unittest(cwd, timeout=timeout, python=python, path_extra=path_extra)
    return r


def test_exercises_script(test_paths, script_paths):
    """Level 20 (H2): does any test REALLY exercise one of the scripts? Yes if a test IMPORTS it (import x /
    from x import ...) or names it as a file ('x.py', e.g. to launch it with subprocess). A test that
    redefines its own function and never touches the script would give a pass on untested code."""
    import ast

    modules = {Path(s).stem for s in script_paths}
    names = {Path(s).name for s in script_paths}
    for t in test_paths:
        try:
            source = Path(t).read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(source)
        except (OSError, SyntaxError, ValueError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import) and any(a.name.split(".")[0] in modules for a in node.names):
                return True
            if isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[0] in modules:
                return True
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if any(n in node.value for n in names) or node.value in modules:
                    return True  # e.g. subprocess.run([sys.executable, "calc.py"]) or importlib("calc")
    return False


test_exercises_script.__test__ = False  # a helper, not a pytest test (its name starts with "test_")
