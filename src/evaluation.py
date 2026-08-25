#!/usr/bin/env python3
# evaluation.py: evaluation suite for the agent system (Level 4)
#
# Runs several varied TASKS through the full system (orchestrator) N times each
# (to account for the 14B model's variability) and, on each run, compares:
#
#   the SYSTEM's VERDICT   vs   REALITY (a deterministic ORACLE per task)
#
# The oracle does not trust the agent: it inspects the resulting workspace and
# runs the code/tests on its own to decide whether the task is REALLY done. This
# is how we measure which TYPES of task the system's verdict can be trusted on.
#
# Requirement: the Level 1 MLX server must be running (./start_local.sh).
# Usage: ./evaluation.sh                 (2 repetitions per task, default)
#        ./evaluation.sh --repeats 1
import argparse
import json
import re
import subprocess  # nosec B404: launches the agent/tests in a subprocess (fixed list: venv-python + argv, no shell)
import tempfile
import time
from pathlib import Path


def _health_url(cfg):
    """Health-check URL of the MLX server, derived from config.env (do not hardcode host:port:
    the rest of the system uses SERVER_HOST/SERVER_PORT and the suite must stay consistent)."""
    host = cfg.get("SERVER_HOST") or "127.0.0.1"
    port = cfg.get("SERVER_PORT") or "8000"
    return f"http://{host}:{port}/v1/models"

ROOT = Path(__file__).resolve().parent.parent   # code lives in src/; agent.sh and .venv at the root
AGENT_SH = ROOT / "agent.sh"
PY = ROOT / ".venv" / "bin" / "python"
TASK_TIMEOUT = 480   # max seconds per full-system run


# Oracle helpers: run code in an isolated subprocess (venv), in the deliverable's
# directory. Deterministic and independent of the agent.
def _run(workdir, argv, timeout=90):
    """Run `<venv-python> argv...` in workdir. Returns (returncode, stdout+stderr)."""
    try:
        p = subprocess.run([str(PY)] + argv, cwd=str(workdir),  # noqa: S603: fixed list (venv-python + argv defined in the test suite), no shell  # nosec B603
                           capture_output=True, text=True, timeout=timeout)
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except Exception as e:  # noqa: BLE001
        return 124, f"{type(e).__name__}: {e}"


def probe(workdir, code):
    """True if the `code` snippet runs without error (asserts included) in the workspace."""
    rc, _ = _run(workdir, ["-c", code])
    return rc == 0


def tests_pass(workdir):
    """True if there is >=1 test and ALL pass (unittest discover in the workspace)."""
    rc, out = _run(workdir, ["-m", "unittest", "discover"])
    m = re.search(r"Ran (\d+) test", out)
    return bool(m) and int(m.group(1)) > 0 and rc == 0


def has_test_file(workdir):
    return any(p.name.startswith("test") and p.suffix == ".py" for p in workdir.glob("*.py"))


# Per-task ORACLES: return (reality_ok: bool, note: str)
def oracle_median(d):
    fok = probe(d, "import median as m;"
                   "assert m.median([3,1,2])==2;"
                   "assert m.median([1,2,3,4])==2.5;"
                   "assert m.median([]) is None")
    tok = tests_pass(d)
    return fok and tok, f"functional={fok} tests={tok}"


def oracle_fizzbuzz(d):
    fok = probe(d, "import fizzbuzz as m;"
                   "assert m.fizzbuzz(3)=='Fizz';"
                   "assert m.fizzbuzz(5)=='Buzz';"
                   "assert m.fizzbuzz(15)=='FizzBuzz';"
                   "assert str(m.fizzbuzz(7))=='7'")
    tok = tests_pass(d)
    return fok and tok, f"functional={fok} tests={tok}"


def oracle_shapes(d):
    fok = probe(d, "import shapes as m, math;"
                   "assert abs(m.area_circle(1)-math.pi)<1e-6;"
                   "assert m.area_rectangle(2,3)==6")
    tok = tests_pass(d)
    two = (d / "shapes.py").exists() and has_test_file(d)
    return fok and tok and two, f"functional={fok} tests={tok} 2files={two}"


def oracle_temperatures(d):
    fok = probe(d, "import temperatures as m;"
                   "assert m.celsius_to_fahrenheit(100)==212;"
                   "assert m.fahrenheit_to_celsius(32)==0")
    tok = tests_pass(d)  # the task EXPLICITLY asks for "and its tests"
    return fok and tok, f"functional={fok} tests={tok}"


def oracle_words(d):
    # Underspecified task: search any .py for a function that counts words
    # correctly (best-effort functional oracle, no fixed name).
    code = (
        "import importlib, glob\n"
        "ok = False\n"
        "for f in glob.glob('*.py'):\n"
        "    try: m = importlib.import_module(f[:-3])\n"
        "    except Exception: continue\n"
        "    for name in dir(m):\n"
        "        fn = getattr(m, name)\n"
        "        if callable(fn) and not name.startswith('_'):\n"
        "            try:\n"
        "                if fn('hello world') == 2 and fn('one two three') == 3:\n"
        "                    ok = True\n"
        "            except Exception:\n"
        "                pass\n"
        "import sys; sys.exit(0 if ok else 1)\n"
    )
    fok = probe(d, code)
    return fok, f"word_count_function={fok}"


def oracle_calculator(d):
    # Underspecified + runnable: asked for a CLI and "its tests". Reality = tests pass
    # AND there is a CLI (__main__) that does NOT crash (no Traceback) when run without args.
    tok = tests_pass(d)
    cli = next((p for p in d.glob("*.py")
                if not p.name.startswith("test") and "__main__" in p.read_text(errors="ignore")),
               None)
    if cli is None:
        return False, f"no_CLI tests={tok}"
    _, output = _run(d, [cli.name])  # run without arguments
    no_crash = "Traceback (most recent call last)" not in output
    return tok and no_crash, f"cli={cli.name} tests={tok} no_crash_without_args={no_crash}"


def oracle_palindrome(d):
    fok = probe(d, "import palindrome as m;"
                   "assert m.is_palindrome('racecar') is True;"
                   "assert m.is_palindrome('python') is False;"
                   "assert m.is_palindrome('Was it a car or a cat I saw') is True")
    tok = tests_pass(d)
    return fok and tok, f"functional={fok} tests={tok}"


def oracle_document(d):
    # 4D: report.md must exist with the three requested sections (real, non-empty text).
    md = d / "report.md"
    if not md.exists():
        return False, "no report.md"
    txt = md.read_text(errors="ignore").lower()
    sections = all(s in txt for s in ("introduc", "benefit", "conclus"))
    substance = len(txt) > 200
    return sections and substance, f"report.md={md.exists()} sections={sections} substance={substance}"


def oracle_fix(d):
    # Editing an EXISTING file (seeded with a bug): after the fix, factorial must
    # return 120 for 5 and 1 for 0, and the seeded tests must pass.
    fok = probe(d, "import mathutils as m;"
                   "assert m.factorial(5)==120;"
                   "assert m.factorial(0)==1;"
                   "assert m.factorial(1)==1")
    tok = tests_pass(d)
    return fok and tok, f"functional={fok} tests={tok}"


def oracle_question(d, output):
    """CHAT (5D): the correct behavior is that the system ROUTES it as chat and answers DIRECTLY,
    WITHOUT touching disk. Checks: (a) routed as chat, (b) did NOT create any file, (c) the
    answer mentions key concepts from the question (did not answer empty)."""
    routed = "chat" in output.lower()  # the system prints "chat -> direct answer"
    no_files = not any(p.is_file() for p in d.iterdir())
    s = output.lower()
    # substantive answer ON TOPIC (function), not empty and not an error
    content = "func" in s and len(s) > 200 and "error" not in s[:40]
    ok = routed and no_files and content
    return ok, f"chat={routed} no_files={no_files} content_ok={content}"


# --- Seeds: pre-populate the workspace BEFORE the run (edit tasks) ----------
def seed_fix(d):
    """Seed mathutils.py with a BUGGED factorial (wrong base case) plus a test that now fails."""
    (d / "mathutils.py").write_text(
        "def factorial(n):\n"
        "    # BUG: the base case returns 0, so factorial(n) is always 0\n"
        "    if n == 0:\n"
        "        return 0\n"
        "    return n * factorial(n - 1)\n", encoding="utf-8")
    (d / "test_mathutils.py").write_text(
        "import unittest, mathutils\n"
        "class T(unittest.TestCase):\n"
        "    def test_5(self): self.assertEqual(mathutils.factorial(5), 120)\n"
        "    def test_0(self): self.assertEqual(mathutils.factorial(0), 1)\n"
        "if __name__ == '__main__': unittest.main()\n", encoding="utf-8")


# Suite definition
TASKS = [
    {"id": "median", "type": "well-scoped (func+tests)", "oracle": oracle_median,
     "prompt": "Create median.py with a function median(numbers) that returns the median of "
               "a list of numbers, and None if the list is empty. Add tests with unittest."},
    {"id": "fizzbuzz", "type": "well-scoped (func+tests)", "oracle": oracle_fizzbuzz,
     "prompt": "Create fizzbuzz.py with a function fizzbuzz(n) that returns 'Fizz' if n is a "
               "multiple of 3, 'Buzz' if of 5, 'FizzBuzz' if of both, and the number as a "
               "string otherwise. Add tests with unittest."},
    {"id": "shapes", "type": "multi-file", "oracle": oracle_shapes,
     "prompt": "Create shapes.py with functions area_circle(r) and area_rectangle(base, height), "
               "and a separate file test_shapes.py with unittest tests for both. Leave the "
               "tests passing."},
    {"id": "temperatures", "type": "explicit 'and its tests'", "oracle": oracle_temperatures,
     "prompt": "Create temperatures.py with functions celsius_to_fahrenheit(c) and "
               "fahrenheit_to_celsius(f), and its tests."},
    {"id": "words", "type": "underspecified", "oracle": oracle_words,
     "prompt": "Make a small utility to count how many words a text has."},
    {"id": "calculator", "type": "underspecified (CLI)", "oracle": oracle_calculator,
     "prompt": "create a small command-line calculator with basic operations and its tests"},
    {"id": "palindrome", "type": "well-scoped (func+tests)", "oracle": oracle_palindrome,
     "prompt": "Create palindrome.py with a function is_palindrome(text) that returns True if the "
               "text is a palindrome ignoring case and spaces, and False otherwise. Add tests "
               "with unittest."},
    {"id": "document", "type": "document (4D)", "oracle": oracle_document,
     "prompt": "Create a Markdown report named report.md about the benefits of physical "
               "exercise, with the sections Introduction, Benefits and Conclusion."},
    {"id": "fix_existing", "type": "edit existing file", "oracle": oracle_fix,
     "seed": seed_fix,
     "prompt": "In mathutils.py the factorial function is wrong: factorial(0) should be 1, not 0. "
               "Fix it in place so the tests pass."},
    {"id": "question", "type": "question/chat (5D)", "oracle": oracle_question, "chat": True,
     "prompt": "Explain in 3 sentences what a function is in programming and what it is for."},
]


def system_verdict(output):
    """Extract the verdict the SYSTEM gave from its output (no color, comes as text)."""
    if "Verification OK" in output:
        return "OK"
    # 6B: honest stops due to derailment/exhaustion also count as "not completed".
    if ("NOT completed" in output or "Ran out of attempts" in output
            or "Stopped without completing" in output or "Could not complete the task" in output):
        return "NOT_COMPLETED"
    return "?"


def classify(system_ok, reality_ok):
    if system_ok and reality_ok:
        return "OK-correct"           # true positive
    if system_ok and not reality_ok:
        return "FALSE_POSITIVE"        # the worst! said OK but it is not right
    if not system_ok and reality_ok:
        return "FALSE_NEGATIVE"        # said not-OK but it was fine
    return "no-OK-correct"             # true negative (honest: it was not right)


def main():
    ap = argparse.ArgumentParser(description="Agent system evaluation suite (N4)")
    ap.add_argument("--repeats", type=int, default=2, help="runs per task (default 2)")
    ap.add_argument("--tasks", default="", help="ids separated by commas (reduced suite); "
                    "empty = all. E.g.: --tasks median,words,calculator")
    ap.add_argument("--pause", type=float, default=0.0,
                    help="seconds of pause between runs (spares the free-tier quota)")
    ap.add_argument("--local", action="store_true",
                    help="force 100%% local on each run (14B-only baseline; no cloud)")
    ap.add_argument("--no-self-critique", action="store_true",
                    help="pass --no-self-critique to the agent (to measure the effect of self-critique 6C)")
    ap.add_argument("--no-lessons", action="store_true",
                    help="pass --no-lessons to the agent (to measure the effect of learning 6C)")
    ap.add_argument("--no-memory", action="store_true",
                    help="pass --no-memory to the agent (isolates the effect of project memory 4E)")
    ap.add_argument("--save", default="", metavar="FILE",
                    help="save the per-task summary to a JSON (regression baseline)")
    ap.add_argument("--compare", default="", metavar="FILE",
                    help="compare against a previous baseline and warn about REGRESSIONS (exit!=0 if any)")
    args = ap.parse_args()

    sel = [s.strip() for s in args.tasks.split(",") if s.strip()]
    tasks = [t for t in TASKS if not sel or t["id"] in sel]
    if sel:
        missing = [s for s in sel if s not in {t["id"] for t in TASKS}]
        if missing:
            raise SystemExit(f"ERROR: unknown tasks: {missing}. Available: "
                             f"{[t['id'] for t in TASKS]}")

    # Check that the server is alive (the suite uses the local model).
    try:
        import urllib.request
        from agent import load_config
        urllib.request.urlopen(_health_url(load_config()), timeout=4).read()  # noqa: S310: fixed config URL (http://127.0.0.1 to the local MLX server), not user input  # nosec B310
    except Exception:
        raise SystemExit("ERROR: the MLX server is not responding. Start it with ./start_local.sh")

    base = Path(tempfile.mkdtemp(prefix="eval_suite_"))
    print(f"Results/artifacts in: {base}  ({len(tasks)} task(s) x {args.repeats} rep)\n")
    rows = []
    for t in tasks:
        for r in range(1, args.repeats + 1):
            d = base / f"{t['id']}_rep{r}"
            d.mkdir(parents=True, exist_ok=True)
            if t.get("seed"):
                t["seed"](d)   # seed the workspace BEFORE (tasks that edit an existing file)
            cmd = [str(AGENT_SH), "-p", t["prompt"]]
            if args.local:
                cmd.append("--local")   # 14B-only baseline (no cloud)
            if args.no_self_critique:
                cmd.append("--no-self-critique")   # 6C: measure the effect of self-critique
            if args.no_lessons:
                cmd.append("--no-lessons")         # 6C: measure the effect of learning
            if args.no_memory:
                cmd.append("--no-memory")          # 4E: isolate project memory
            try:
                p = subprocess.run(cmd, cwd=str(d),  # noqa: S603: `cmd` is a fixed list (agent.sh + test-suite prompt), no shell  # nosec B603
                                   capture_output=True, text=True, timeout=TASK_TIMEOUT)
                output = (p.stdout or "") + (p.stderr or "")
            except subprocess.TimeoutExpired:
                output = "(timeout)"

            if t.get("chat"):
                # CHAT (5D): the correct behavior is answering directly WITHOUT touching disk. The
                # oracle checks it (routed + no files + content); the FP/FN of tasks does not apply.
                real_ok, note = t["oracle"](d, output)
                vsys = "CHAT" if "chat" in output.lower() else "no-chat"
                sys_ok = real_ok
                cat = "OK-correct" if real_ok else "CHAT_FAIL"
                matches = real_ok
            else:
                vsys = system_verdict(output)
                sys_ok = (vsys == "OK")
                real_ok, note = t["oracle"](d)
                cat = classify(sys_ok, real_ok)
                matches = (sys_ok == real_ok)
            row = {"task": t["id"], "type": t["type"], "rep": r, "verdict": vsys,
                   "reality": "OK" if real_ok else "NO", "category": cat,
                   "matches": matches, "note": note}
            rows.append(row)
            print(f"  [{t['id']:<12} rep{r}] system={vsys:<13} reality="
                  f"{'OK' if real_ok else 'NO':<3} -> {cat:<16} ({note})")
            if args.pause > 0:
                time.sleep(args.pause)  # breathe between runs (spare the free quota)

    # ---------------- Table ----------------
    print("\n" + "=" * 96)
    print(f"{'TASK':<13}{'TYPE':<28}{'REP':<4}{'SYSTEM':<14}{'REAL':<6}{'CATEGORY':<16}{'MATCHES?'}")
    print("-" * 96)
    for f in rows:
        print(f"{f['task']:<13}{f['type']:<28}{f['rep']:<4}{f['verdict']:<14}"
              f"{f['reality']:<6}{f['category']:<16}{'yes' if f['matches'] else 'NO'}")

    # ---------------- Conclusion by TYPE ----------------
    print("\n" + "=" * 96)
    print("CONCLUSION BY TASK TYPE")
    print("-" * 96)
    types = {}
    for f in rows:
        types.setdefault(f["type"], []).append(f)
    for type_, fs in types.items():
        n = len(fs)
        matched = sum(1 for f in fs if f["matches"])
        fp = sum(1 for f in fs if f["category"] == "FALSE_POSITIVE")
        fn = sum(1 for f in fs if f["category"] == "FALSE_NEGATIVE")
        if fp == 0 and matched == n:
            judgment = "TRUSTWORTHY (verdict matched reality in all)"
        elif fp > 0:
            judgment = f"NOT TRUSTWORTHY ({fp} false positive(s): said OK without being so)"
        else:
            judgment = f"PARTIAL ({fn} false negative(s); no false positives)"
        print(f"- {type_:<30} {matched}/{n} match . FP={fp} FN={fn} -> {judgment}")

    # Global summary
    tot = len(rows)
    fp_tot = sum(1 for f in rows if f["category"] == "FALSE_POSITIVE")
    matched_tot = sum(1 for f in rows if f["matches"])
    print("-" * 96)
    print(f"GLOBAL: {matched_tot}/{tot} match . false positives={fp_tot} "
          f"(a false positive = the system said OK with something broken)")

    # ---------------- Regression mode: save / compare ----------------
    # Per-task summary (bounded, JSON-serializable): the safety net to grow without breaking.
    summary = {}
    for f in rows:
        r = summary.setdefault(f["task"], {"n": 0, "matches": 0, "fp": 0})
        r["n"] += 1
        r["matches"] += int(f["matches"])
        r["fp"] += int(f["category"] in ("FALSE_POSITIVE",))

    if args.save:
        Path(args.save).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nRegression baseline saved to: {args.save}")

    if args.compare:
        try:
            base_prev = json.loads(Path(args.compare).read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            raise SystemExit(f"ERROR: could not read the baseline {args.compare}: {e}")
        print("\n" + "=" * 96)
        print(f"REGRESSION vs baseline ({args.compare})")
        print("-" * 96)
        regressions, improvements = [], []
        for tid, now in summary.items():
            before = base_prev.get(tid)
            if not before:
                continue
            rate_before = before["matches"] / before["n"] if before["n"] else 0
            rate_now = now["matches"] / now["n"] if now["n"] else 0
            # Regression = MORE false positives appear, or the hit rate drops.
            if now["fp"] > before["fp"] or rate_now < rate_before - 1e-9:
                regressions.append((tid, before, now, rate_before, rate_now))
            elif now["fp"] < before["fp"] or rate_now > rate_before + 1e-9:
                improvements.append((tid, rate_before, rate_now))
        for tid, a, n, ta, tn in regressions:
            print(f"  REGRESSION {tid:<14} hit rate {ta:.0%}->{tn:.0%}  FP {a['fp']}->{n['fp']}")
        for tid, ta, tn in improvements:
            print(f"  improvement {tid:<14} hit rate {ta:.0%}->{tn:.0%}")
        if not regressions:
            print("  No regressions against the baseline.")
        print("-" * 96)
        print("NOTE: the 14B is stochastic; a one-off difference may be noise. A repeated "
              "REGRESSION (or new FPs) does indicate a real breakage.")
        if regressions:
            raise SystemExit(1)  # exit!=0 -> usable as a safety net in scripts


if __name__ == "__main__":
    main()
