"""Level 20 regressions: worker, test runner, time machine, self-audit, brain/vision.

Each test reproduces a REAL failure found by running code and pins its fix so it does not come back. No
model, no network, no server (like test_invariants.py).

  H2  the python+test verifier requires >=1 test run, no skip/xfail, and that the test USES the script
  H3  the worker classifier uses whole words (English and Spanish) and a named extension wins
  H4  approving a stage without strong verification does NOT turn the run into "done"
  H7  the web and CSV verifiers do not accept half-done artifacts
  H8  the self-audit does not say OK when its own tools did not run
  D3  undo: order, damaged copies, unreadable files, later user edits, honest "failed" list
  R2  vision never stops the brain for nothing (bad input, or no way to start it again)
  R3  the worker releases the agent lock while it waits for the user; an expired pause cancels
  REV review of Level 20's own changes: classifier (test_x.py, page/site, library names, rename/delete),
      the neutral WORKDIR while the worker waits, an unreadable file is not a deletion
"""

import inspect
import json
import os
import stat
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import agent  # noqa: E402
import brain  # noqa: E402
import self_audit  # noqa: E402
import test_runner  # noqa: E402
import time_machine  # noqa: E402
import vision  # noqa: E402
import worker  # noqa: E402


def _tmp():
    return Path(tempfile.mkdtemp()).resolve()


# ─────────────────────────────── H2 · test runner + python/test verifier ───────────────────────────────
def _py_project(d, test_src, mod_src="def add(a, b):\n    return a + b\n"):
    (d / "calc.py").write_text(mod_src, encoding="utf-8")
    (d / "test_calc.py").write_text(test_src, encoding="utf-8")
    return d


_TEST_OK = "from calc import add\n\ndef test_add():\n    assert add(2, 2) == 4\n"
_BROKEN_MOD = "def add(a, b):\n    return a - b\n"


class TestH2PythonTestVerifier(unittest.TestCase):
    def test_skip_with_broken_module_is_NOT_ok(self):
        d = _py_project(
            _tmp(),
            "import pytest\nfrom calc import add\n\n@pytest.mark.skip(reason='later')\n"
            "def test_add():\n    assert add(2, 2) == 4\n",
            _BROKEN_MOD,
        )
        ok, det = worker.verify_python_test(d)
        self.assertFalse(ok, det)

    def test_xfail_with_broken_module_is_NOT_ok(self):
        d = _py_project(
            _tmp(),
            "import pytest\nfrom calc import add\n\n@pytest.mark.xfail\ndef test_add():\n    assert add(2, 2) == 4\n",
            _BROKEN_MOD,
        )
        self.assertFalse(worker.verify_python_test(d)[0])

    def test_a_test_that_does_not_use_the_script_is_NOT_ok(self):
        d = _py_project(
            _tmp(), "def add(a, b):\n    return a + b\n\ndef test_add():\n    assert add(2, 2) == 4\n", _BROKEN_MOD
        )
        ok, det = worker.verify_python_test(d)
        self.assertFalse(ok)
        self.assertTrue(any("does NOT use the script" in t for c, t in det if not c), det)

    def test_launching_the_script_by_subprocess_counts_as_using_it(self):
        d = _tmp()
        (d / "greet.py").write_text("print('hello')\n", encoding="utf-8")
        (d / "test_greet.py").write_text(
            "import subprocess, sys\n\ndef test_greet():\n"
            "    out = subprocess.run([sys.executable, 'greet.py'], capture_output=True, text=True).stdout\n"
            "    assert 'hello' in out\n",
            encoding="utf-8",
        )
        ok, det = worker.verify_python_test(d)
        self.assertTrue(ok, det)

    def test_a_real_passing_test_IS_ok(self):
        ok, det = worker.verify_python_test(_py_project(_tmp(), _TEST_OK))
        self.assertTrue(ok, det)

    def test_a_red_test_says_FAILS(self):
        ok, det = worker.verify_python_test(_py_project(_tmp(), _TEST_OK, _BROKEN_MOD))
        self.assertFalse(ok)
        self.assertTrue(any("FAILS" in t for c, t in det if not c))

    def test_zero_tests_does_not_pass(self):
        d = _py_project(_tmp(), "from calc import add\n# no tests\n")
        r = test_runner.run_pytest(d)
        self.assertFalse(r["ok"])
        self.assertEqual(r["total"], 0)

    def test_result_has_the_shared_keys(self):
        r = test_runner.run_pytest(_py_project(_tmp(), _TEST_OK))
        self.assertEqual(
            set(r), {"ok", "total", "passed", "failed", "errors", "skipped", "timeout", "no_pytest", "summary"}
        )
        self.assertTrue(r["ok"], r)
        self.assertEqual((r["total"], r["passed"]), (1, 1))

    def test_project_tests_pass_sees_pytest_and_unittest_styles(self):
        self.assertTrue(test_runner.project_tests_pass(_py_project(_tmp(), _TEST_OK))["ok"])
        d = _tmp()
        (d / "calc.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
        (d / "test_calc.py").write_text(
            "import unittest\nfrom calc import add\n\n"
            "class T(unittest.TestCase):\n    def test_a(self):\n"
            "        self.assertEqual(add(1, 1), 2)\n",
            encoding="utf-8",
        )
        self.assertTrue(test_runner.project_tests_pass(d)["ok"])

    def test_leaves_no_orphan_processes(self):
        # R6: a test that leaves a background process must not outlive the verifier.
        d = _tmp()
        mark = d / "alive.txt"
        (d / "child.py").write_text(
            f"import time, pathlib\ntime.sleep(2)\npathlib.Path({str(mark)!r}).write_text('x')\n", encoding="utf-8"
        )
        (d / "test_bg.py").write_text(
            "import subprocess, sys\n\ndef test_bg():\n"
            "    subprocess.Popen([sys.executable, 'child.py'])\n"
            "    assert True\n",
            encoding="utf-8",
        )
        r = test_runner.run_pytest(d, timeout=30)
        self.assertTrue(r["ok"], r)
        time.sleep(3)
        self.assertFalse(mark.exists(), "the test's child process outlived the verifier")

    def test_timeout_does_not_pass(self):
        d = _tmp()
        (d / "test_slow.py").write_text("import time\n\ndef test_slow():\n    time.sleep(30)\n", encoding="utf-8")
        r = test_runner.run_pytest(d, timeout=2)
        self.assertFalse(r["ok"])
        self.assertTrue(r["timeout"])

    def test_test_exercises_script(self):
        d = _tmp()
        (d / "calc.py").write_text("x = 1\n", encoding="utf-8")
        (d / "test_a.py").write_text("from calc import x\n", encoding="utf-8")
        (d / "test_b.py").write_text("import os\n\ndef test_b():\n    assert os\n", encoding="utf-8")
        self.assertTrue(test_runner.test_exercises_script([d / "test_a.py"], [d / "calc.py"]))
        self.assertFalse(test_runner.test_exercises_script([d / "test_b.py"], [d / "calc.py"]))


# ─────────────────────────────── H3 · worker classifier ───────────────────────────────
class TestH3Classifier(unittest.TestCase):
    CASES = [
        ("create a single-page static website", "web"),
        ("a Python script that adds two numbers and its test", "python_test"),
        ("write tests in python for utils.py", "python_test"),
        ("create calc.py and its test test_calc.py", "python_test"),
        ("write a short story in story.txt", "open"),
        ("", "open"),
        # false positives that used to credit the WRONG artifact:
        ("a python script that fetches the latest prices", "open"),  # la-TEST
        ("answer the contest emails from yesterday", "open"),  # con-TEST
        ("list the sites I visited yesterday", "open"),  # vi-SITE-d
        ("save the notes on some site in notes.md", "open"),  # site -> web
        ("write a 2-page summary in summary.md", "open"),  # page -> web
        ("Write a 300-word summary in summary.md", "open"),  # 300-word -> docx
        ("Write a 300-word summary of the meeting", "open"),  # 300-word -> docx
        ("a python script that counts each word of a text", "open"),  # word -> docx
        ("create scraper.py that reads a web page", "open"),  # web inside a .py
        ("search the web for AI news and summarize them", "open"),  # searching != building a web
        ("explain what is in notes.md", "open"),  # explicit .md -> open
        # a named extension wins:
        ("create analysis.ipynb in Python that tests the function", "ipynb"),
        ("python script that downloads the latest prices to prices.csv", "csv"),
        ("make index.html and styles.css for a landing", "web"),
        # keywords (no file named) still work:
        ("make a website about excel", "web"),
        ("create a web page with a search bar", "web"),
        ("create a word document with the report", "docx"),
        ("make a presentation with 3 slides", "pptx"),
        ("make me an expense spreadsheet", "xlsx"),
        ("create a jupyter notebook that says hello", "ipynb"),
        ("a csv of the planets with name and diameter", "csv"),
        # the same classifier also takes Spanish tasks:
        ("crea una web estática de una página", "web"),
        ("haz pruebas en python de utils.py", "python_test"),
        ("script Python que comprueba si un número es primo", "open"),  # com-PRUEBA
        ("busca en la web noticias de IA y resúmelas", "open"),
        ("crea un documento word con el informe", "docx"),
        ("haz una presentación de 3 diapositivas", "pptx"),
        ("hazme una hoja de cálculo de gastos", "xlsx"),
    ]

    def test_cases(self):
        for task, expected in self.CASES:
            self.assertEqual(worker.classify_task(task), expected, task)

    def test_extensions_named_is_a_whitelist(self):
        self.assertEqual(worker.extensions_named("see google.com, e.g. calc.py and notes.md"), {"py", "md"})
        self.assertEqual(worker.named_files("write it to ./out/report.md and data.csv"), ["out/report.md", "data.csv"])

    def test_open_task_requires_the_named_files_to_exist(self):
        d = _tmp()
        before = worker._useful_files(d)
        (d / "something_else.md").write_text("x", encoding="utf-8")
        ok, det, _ = worker.generic_checks(d, before, "write the summary in summary.md")
        self.assertFalse(ok)
        self.assertTrue(any("summary.md" in t for c, t in det if not c), det)

    def test_open_task_counts_edits_of_existing_files(self):
        d = _tmp()
        (d / "notes.md").write_text("one\n", encoding="utf-8")
        before = worker._useful_files(d)
        time.sleep(0.01)
        (d / "notes.md").write_text("one\ntwo\n", encoding="utf-8")
        ok, _det, changed = worker.generic_checks(d, before, "extend notes.md")
        self.assertTrue(ok)
        self.assertEqual(changed, ["notes.md"])


# ─────────────────────────────── H4 · approved but unverified stages ───────────────────────────────
class TestH4StagesApprovedUnverified(unittest.TestCase):
    def test_approved_open_stages_do_NOT_end_in_done(self):
        cfg = {"AGENT_PROJECTS_DIR": tempfile.mkdtemp(), "WORKER_MAX_STEPS": "1"}
        proj = Path(cfg["AGENT_PROJECTS_DIR"]) / "p"
        proj.mkdir(parents=True)
        n = [0]

        def resolve(router, cfg, project, t):
            (proj / f"f{n[0]}.txt").write_text("x", encoding="utf-8")
            n[0] += 1

        task = "write one idea → write another idea"
        stages = worker._split_into_stages(task)
        self.assertEqual(len(stages), 2)
        persisted = []

        def approve(cfg_, project, *a, **k):
            persisted.append(worker.read_state(cfg_, project))  # what a reloaded view would see
            return "approve"

        evs = []
        with patch.object(worker, "_wait_for_approval", approve):
            worker.run_by_stages(
                task, stages, "p", evs.append, cfg, router=object(), resolve=resolve, signal_fn=lambda: "continue"
            )
        final = [e for e in evs if e["type"] == "worker_final"][-1]
        self.assertEqual(final["state"], "done_unverified")
        self.assertIn("no strong verification", final["summary"])
        approved = [e["state"] for e in evs if e["type"] == "worker_stage" and e["state"].startswith("approved")]
        self.assertEqual(approved, ["approved_unverified", "approved_unverified"])
        # R3: while waiting, the state is persisted as awaiting_approval WITH the approval event
        self.assertEqual(persisted[0].get("state"), "awaiting_approval")
        self.assertEqual(persisted[0]["approval"]["type"], "worker_awaiting_approval")
        self.assertIn("run_mtime", worker.read_stages(cfg, "p"))


# ─────────────────────────────── H7 · web and CSV verifiers ───────────────────────────────
class TestH7PartialVerifiers(unittest.TestCase):
    def _web(self, d, html, css="body{color:#333}", css_name="styles.css"):
        (d / "index.html").write_text(html, encoding="utf-8")
        if css is not None:
            (d / css_name).write_text(css, encoding="utf-8")
        return d

    _GOOD = (
        '<!doctype html><html><head><title>Hello</title><link rel="stylesheet" href="styles.css">'
        "</head><body><h1>Hello world</h1><p>visible content</p></body></html>"
    )

    def test_valid_web_passes(self):
        ok, det = worker.verify_static_web(self._web(_tmp(), self._GOOD))
        self.assertTrue(ok, det)

    def test_empty_body_with_a_long_title_does_NOT_pass(self):
        html = (
            "<!doctype html><html><head><title>A rather long page title</title>"
            '<link rel="stylesheet" href="styles.css"><style>body{margin:0}</style></head><body></body></html>'
        )
        self.assertFalse(worker.verify_static_web(self._web(_tmp(), html))[0])

    def test_link_to_a_css_that_does_not_exist_does_NOT_pass(self):
        html = self._GOOD.replace('href="styles.css"', 'href="style.css"')
        self.assertFalse(worker.verify_static_web(self._web(_tmp(), html))[0])

    def test_empty_css_does_NOT_pass(self):
        self.assertFalse(worker.verify_static_web(self._web(_tmp(), self._GOOD, css="  /* nothing */ "))[0])

    def test_csv_columns_separators_and_words_that_contain_instructions(self):
        self.assertEqual(
            worker._requested_columns("data.csv with columns: name, age and city"), ["name", "age", "city"]
        )
        self.assertEqual(
            worker._requested_columns("columns: product, data_date, deadline"), ["product", "data_date", "deadline"]
        )
        self.assertEqual(worker._requested_columns("columns: name & email"), ["name", "email"])
        self.assertEqual(worker._requested_columns("columns: name; age / city"), ["name", "age", "city"])
        self.assertEqual(worker._requested_columns("columnas: nombre, edad y ciudad"), ["nombre", "edad", "ciudad"])
        self.assertEqual(worker._requested_columns("columnas: nombre e idioma"), ["nombre", "idioma"])

    def test_csv_verifier_requires_the_column_after_and(self):
        d = _tmp()
        (d / "people.csv").write_text("name,age\nAna,30\n", encoding="utf-8")
        ok, det = worker.verify_csv(d, "people.csv with columns: name, age and city")
        self.assertFalse(ok, det)
        self.assertTrue(any("city" in t for c, t in det if not c), det)


# ─────────────────────────────── H8 · honest self-audit ───────────────────────────────
class TestH8SelfAudit(unittest.TestCase):
    A = self_audit

    def test_mutating_tools_list_matches_the_agent_in_both_directions(self):
        self.assertEqual(self.A.TOOLS_THAT_MUTATE, set(agent._TOOLS_MUTATE))

    def test_origin_ok_only_in_a_comment_is_FAIL(self):
        bad = (
            '@app.post("/x")\nasync def api_x(request: Request):\n'
            '    # _origin_ok(request) would go here\n    return {"ok": True}\n'
        )
        self.assertEqual(self.A.check_cswsh(bad)["result"], "FAIL")
        bad_delete = '@app.delete("/x")\nasync def api_delete(request):\n    return {}\n'
        self.assertEqual(self.A.check_cswsh(bad_delete)["result"], "FAIL")
        good = (
            '@app.post("/x")\nasync def api_x(request):\n'
            "    if not _origin_ok(request):\n        return 403\n    return {}\n"
        )
        self.assertEqual(self.A.check_cswsh(good)["result"], "OK")

    def test_safe_path_only_in_a_comment_is_FAIL(self):
        bad = "def write_file(path, content):\n    # TODO _safe_path(path)\n    open(path, 'w').write(content)\n"
        self.assertEqual(self.A.check_path_safety(bad)["result"], "FAIL")

    def test_secrets_with_a_prefix_and_real_formats(self):
        for line in (
            'OPENAI_API_KEY = "Abc123RealSecretValue999"',
            # fake tokens, built from pieces so the source holds no token-shaped literal
            'HF_TOKEN="' + "hf" + "_" + "x1" * 17 + '"',
            'key = "' + "ghp" + "_" + "y2" * 18 + '"',
        ):
            root = _tmp()
            (root / "src").mkdir()
            (root / "src" / "m.py").write_text(line + "\n", encoding="utf-8")
            self.assertEqual(self.A.check_secrets(root)["result"], "FAIL", line)

    def test_empty_requirements_is_not_ok(self):
        self.assertEqual(self.A.check_pinned_requirements("")["result"], "WARN")
        self.assertEqual(self.A.check_pinned_requirements("# only comments\n")["result"], "WARN")
        self.assertEqual(self.A.check_pinned_requirements("requests==2.32.3\n")["result"], "OK")

    def test_sast_tools_that_fail_do_NOT_give_ok(self):
        for fn in (self.A.sast_ruff_security, self.A.sast_bandit, self.A.sast_pip_audit):
            with patch.object(self.A, "_run", return_value=(2, "", "boom")):
                self.assertEqual(fn()["result"], "WARN", fn.__name__)
            with patch.object(self.A, "_run", return_value=(0, "", "")):
                self.assertEqual(fn()["result"], "WARN", fn.__name__)
        with patch.object(self.A, "_run", return_value=(0, '{"dependencies": []}', "")):
            self.assertEqual(self.A.sast_pip_audit()["result"], "WARN")


# ─────────────────────────────── D3 · undo ───────────────────────────────
class TestD3Undo(unittest.TestCase):
    def setUp(self):
        self.b = _tmp()

    def _task(self, fn):
        sid = time_machine.create_snapshot(self.b, "t")
        fn()
        time_machine.record_changes(self.b, sid)
        return sid

    def test_folder_replaced_by_a_file_is_recovered(self):
        (self.b / "a").mkdir()
        (self.b / "a" / "x.txt").write_text("the user's", encoding="utf-8")
        import shutil as _sh

        def task():
            _sh.rmtree(self.b / "a")
            (self.b / "a").write_text("the task's file", encoding="utf-8")

        sid = self._task(task)
        plan = time_machine.undo(self.b, sid)
        self.assertEqual(plan["failed"], [])
        self.assertEqual((self.b / "a" / "x.txt").read_text(encoding="utf-8"), "the user's")

    def test_damaged_copy_does_not_overwrite_the_live_file(self):
        (self.b / "f.txt").write_text("ORIGINAL", encoding="utf-8")
        sid = self._task(lambda: (self.b / "f.txt").write_text("changed by the task", encoding="utf-8"))
        copy = time_machine._snaps_dir(self.b) / sid / "files" / "f.txt"
        copy.write_text("ORIGI", encoding="utf-8")  # truncated
        plan = time_machine.undo(self.b, sid)
        self.assertIn("f.txt", plan["failed"])
        self.assertEqual((self.b / "f.txt").read_text(encoding="utf-8"), "changed by the task")

    def test_failed_copy_while_snapshotting_does_not_make_the_file_created(self):
        (self.b / "thesis.md").write_text("mine", encoding="utf-8")
        real = time_machine.shutil.copyfile

        def fails(src, dst, *a, **k):
            if str(src).endswith("thesis.md"):
                raise OSError(28, "No space left on device")
            return real(src, dst, *a, **k)

        with patch.object(time_machine.shutil, "copyfile", fails):
            sid = time_machine.create_snapshot(self.b, "t")
        (self.b / "other.txt").write_text("new", encoding="utf-8")
        time_machine.record_changes(self.b, sid)
        plan = time_machine.plan_undo(self.b, sid)
        self.assertNotIn("thesis.md", plan["delete"])
        time_machine.undo(self.b, sid)
        self.assertEqual((self.b / "thesis.md").read_text(encoding="utf-8"), "mine")

    def test_unreadable_file_while_snapshotting_is_still_tracked(self):
        (self.b / "locked.txt").write_text("mine", encoding="utf-8")
        real = time_machine._signature_of

        def unreadable(base, rel):
            if str(rel) == "locked.txt":
                raise PermissionError(13, "Permission denied")
            return real(base, rel)

        with patch.object(time_machine, "_signature_of", unreadable):
            sid = time_machine.create_snapshot(self.b, "t")
        time_machine.record_changes(self.b, sid)
        plan = time_machine.plan_undo(self.b, sid)
        self.assertNotIn("locked.txt", plan["delete"])

    def test_later_user_edits_are_not_lost(self):
        sid = self._task(lambda: (self.b / "new.txt").write_text("the task's", encoding="utf-8"))
        (self.b / "new.txt").write_text("the task's + 3 hours of the user's work", encoding="utf-8")
        plan = time_machine.plan_undo(self.b, sid)
        self.assertEqual(plan["delete"], [])
        self.assertEqual(plan["changed_after"], ["new.txt"])
        time_machine.undo(self.b, sid)
        self.assertTrue((self.b / "new.txt").exists())
        changes = json.loads((time_machine._snaps_dir(self.b) / sid / "changes.json").read_text(encoding="utf-8"))
        self.assertIn("after", changes)

    @unittest.skipUnless(
        "failed" in inspect.getsource(__import__("orchestrator").apply_undo),
        "orchestrator.apply_undo does not honor the 'failed' list yet (group 2)",
    )
    def test_if_something_fails_the_snapshot_is_NOT_consumed(self):
        import orchestrator
        import projects

        cfg = {"AGENT_PROJECTS_DIR": str(_tmp())}
        proj = projects.create_project(cfg, "p")
        store = orchestrator._chat_dir(cfg, "p", None)
        (proj / "f.txt").write_text("ORIGINAL", encoding="utf-8")
        sid = time_machine.create_snapshot(proj, "t", store=store)
        (proj / "f.txt").write_text("changed", encoding="utf-8")
        time_machine.record_changes(proj, sid, store=store)
        (time_machine._snaps_dir(store) / sid / "files" / "f.txt").write_text("X", encoding="utf-8")
        plan = orchestrator.apply_undo(cfg, "p", sid=sid)
        self.assertTrue(plan["failed"])
        self.assertIn(sid, [m["id"] for m in time_machine.list_snapshots(store)])


# ─────────────────────────────── R2 · vision never stops the brain for nothing ───────────────────────────────
_STARTED = object()  # sentinel: a successful start (a Popen-like object)


class _FakeBrain:
    def __init__(self, startable=True, start_result=_STARTED):
        self.calls, self.startable, self.start_result = [], startable, start_result

    def is_alive(self, cfg, timeout=4):
        return True

    def stop(self, cfg):
        self.calls.append("stop")
        return True

    def start(self, cfg):
        self.calls.append("start")
        return self.start_result

    def wait_for_health(self, cfg, timeout=150):
        return True

    def can_start(self, cfg=None):
        return self.startable


class TestR2VisionDoesNotStopTheBrainForNothing(unittest.TestCase):
    def test_missing_image_or_disabled_vision_does_not_stop_the_brain(self):
        fake = _FakeBrain()
        with patch.object(vision, "free_ram_gb", lambda: 1.0):
            _txt, meta = vision.read_image_auto(
                _tmp() / "missing.png", "?", {"VISION_MODEL": "x", "VISION_MIN_FREE_GB": "9"}, brain=fake
            )
            self.assertFalse(meta["ok"])
            vision.read_image_auto("x.png", "?", {"VISION_MODEL": "", "VISION_MIN_FREE_GB": "9"}, brain=fake)
        self.assertEqual(fake.calls, [], "the brain was stopped for nothing")

    def test_no_way_to_restart_means_no_stop(self):
        fake = _FakeBrain(startable=False)
        with (
            patch.object(vision, "free_ram_gb", lambda: 1.0),
            patch.object(vision, "_input_problem", lambda *a, **k: None),
            patch.object(vision, "read_image", lambda *a, **k: ("read", {"ok": True})),
        ):
            vision.read_image_auto("x.png", "?", {"VISION_MIN_FREE_GB": "9"}, brain=fake)
        self.assertNotIn("stop", fake.calls)

    def test_failed_restart_is_reported_clearly(self):
        fake = _FakeBrain(start_result=None)
        logs = []
        with (
            patch.object(vision, "free_ram_gb", lambda: 1.0),
            patch.object(vision, "_input_problem", lambda *a, **k: None),
            patch.object(vision, "read_image", lambda *a, **k: ("read", {"ok": True})),
        ):
            vision.read_image_auto("x.png", "?", {"VISION_MIN_FREE_GB": "9"}, log=logs.append, brain=fake)
        self.assertEqual(fake.calls, ["stop", "start"])
        self.assertTrue(any("Could not restart the brain" in m for m in logs), logs)

    def test_symlinked_image_is_rejected(self):
        d = _tmp()
        (d / "real.png").write_bytes(b"\x89PNG")
        (d / "link.png").symlink_to(d / "real.png")
        with patch.object(vision, "available", lambda cfg: (True, "x")):
            self.assertEqual(vision._input_problem(d / "link.png", {}), "Image path not allowed.")
            self.assertIsNone(vision._input_problem(d / "real.png", {}, project_dir=d))

    def test_brain_can_start_only_with_an_executable_launcher(self):
        # Public repo: no launcher is shipped -> can_start is False -> vision never stops the brain (before
        # it stopped it and could never restart it).
        root = _tmp()
        with patch.object(brain, "ROOT", root):
            self.assertFalse(brain.can_start())
            script = root / "start_local.sh"
            script.write_text("#!/bin/sh\n", encoding="utf-8")
            self.assertFalse(brain.can_start())  # not executable
            script.chmod(script.stat().st_mode | stat.S_IXUSR)
            self.assertTrue(brain.can_start({}))


# ─────────────────────────────── R3 · waiting for the user does not block the chat ───────────────────────────────
class TestR3WaitsDoNotBlockTheChat(unittest.TestCase):
    def test_while_awaiting_approval_the_lock_is_FREE_and_taken_back(self):
        lock = threading.Lock()
        cfg = {"AGENT_PROJECTS_DIR": str(_tmp())}
        (Path(cfg["AGENT_PROJECTS_DIR"]) / "p").mkdir()
        seen = {}

        def signal():
            if lock.acquire(blocking=False):  # could someone else (the chat) get in now? -> yes
                seen["free"] = True
                lock.release()
            return "approve"

        lock.acquire()
        worker._WAIT.lock = lock
        try:
            r = worker._wait_for_approval(cfg, "p", signal, lambda e: None, poll=0.01)
        finally:
            worker._WAIT.lock = None
        self.assertEqual(r, "approve")
        self.assertTrue(seen.get("free"), "the lock was still held while waiting for the user")
        self.assertFalse(lock.acquire(blocking=False), "the worker did not take the lock back")
        lock.release()

    def test_lock_is_also_free_during_a_pause(self):
        lock = threading.Lock()
        seen = {}
        signals = iter(["pause", "pause", "continue"])

        def signal():
            if lock.acquire(blocking=False):
                seen["free"] = True
                lock.release()
            return next(signals, "continue")

        lock.acquire()
        worker._WAIT.lock = lock
        try:
            r = worker._wait_if_paused(signal, lambda e: None, poll=0.01)
        finally:
            worker._WAIT.lock = None
        self.assertEqual(r, "continue")
        self.assertTrue(seen.get("free"))
        self.assertFalse(lock.acquire(blocking=False))
        lock.release()

    def test_an_expired_pause_cancels_instead_of_carrying_on(self):
        evs = []
        r = worker._wait_if_paused(lambda: "pause", evs.append, poll=0.01, pause_limit=0.05)
        self.assertEqual(r, "cancel")

    def test_run_with_lock_sets_and_clears_the_thread_lock(self):
        lock = threading.Lock()
        seen = {}
        with patch.object(worker, "run", lambda *a, **k: seen.setdefault("lock", worker._WAIT.lock)):
            worker.run_with_lock(lock, "task", "p", lambda e: None)
        self.assertIs(seen["lock"], lock)
        self.assertIsNone(getattr(worker._WAIT, "lock", None))


# ─────────────────────────────── stages that depend on the previous one ───────────────────────────────
class TestStagesThatDependOnThePreviousOne(unittest.TestCase):
    def test_test_stage_credits_the_module_from_the_previous_stage(self):
        d = _tmp()
        run_start = time.time() - 100
        (d / "calc.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
        os.utime(d / "calc.py", (time.time() - 50, time.time() - 50))  # stage 1 (before stage 2)
        stage2_start = time.time() - 10
        (d / "test_calc.py").write_text(_TEST_OK, encoding="utf-8")  # stage 2
        ok, det = worker.verify_python_test(d, since=stage2_start, scripts_since=run_start)
        self.assertTrue(ok, det)
        self.assertFalse(worker.verify_python_test(d, since=stage2_start)[0])  # without run start: as before

    def test_a_stage_that_touched_no_py_does_not_pass(self):
        d = _py_project(_tmp(), _TEST_OK)
        old = time.time() - 50
        for f in d.iterdir():
            os.utime(f, (old, old))
        ok, det = worker.verify_python_test(d, since=time.time() - 5, scripts_since=time.time() - 100)
        self.assertFalse(ok)
        self.assertTrue(any("this stage" in t for c, t in det if not c), det)

    def test_verifier_for_passes_the_run_start(self):
        d = _tmp()
        run_start = time.time() - 100
        (d / "calc.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
        os.utime(d / "calc.py", (time.time() - 50, time.time() - 50))
        (d / "test_calc.py").write_text(_TEST_OK, encoding="utf-8")
        verifier = worker._verifier_for(
            "python_test", {}, "write its test", since=time.time() - 10, run_since=run_start
        )
        self.assertTrue(verifier(d)[0])


# ─────────────────────────────── worker event translator ───────────────────────────────
class TestTranslatorReadsTheRealEventKeys(unittest.TestCase):
    def test_verdict_status_and_notice_warning_texts_reach_the_log(self):
        out = []
        tr, _state = worker._translator(out.append)
        tr({"type": "verdict", "status": "failure", "detail": "tests red"})
        tr({"type": "notice", "text": "no progress: honest stop (6B)"})
        tr({"type": "warning", "text": "generation truncated by length (max_tokens)"})
        logs = [e["text"] for e in out if e["type"] == "worker_log"]
        self.assertEqual(
            logs,
            [
                "verification: failure tests red",
                "no progress: honest stop (6B)",
                "generation truncated by length (max_tokens)",
            ],
        )


# ─────────────────────────────── project folder ───────────────────────────────
class TestProjectDirDelegates(unittest.TestCase):
    def test_project_dir_is_projects_project_path(self):
        import projects

        cfg = {"AGENT_PROJECTS_DIR": str(_tmp())}
        manual = Path(cfg["AGENT_PROJECTS_DIR"]) / "My Thesis"
        manual.mkdir()
        self.assertEqual(worker.project_dir(cfg, "My Thesis"), projects.project_path(cfg, "My Thesis"))
        self.assertEqual(worker.project_dir(cfg, "My Thesis").name, "My Thesis")

    def test_default_resolver_works_inside_a_manual_folder(self):
        import orchestrator

        cfg = {"AGENT_PROJECTS_DIR": str(_tmp())}
        (Path(cfg["AGENT_PROJECTS_DIR"]) / "My Thesis").mkdir()
        (Path(cfg["AGENT_PROJECTS_DIR"]) / "app-project").mkdir()
        seen = []
        with (
            patch.object(orchestrator, "resolve_in_project", lambda *a, **k: seen.append("in_project")),
            patch.object(orchestrator, "resolve_task", lambda *a, **k: seen.append("manual")),
            patch.object(orchestrator, "_set_workdir", lambda d: seen.append(Path(d).name)),
        ):
            worker._default_resolver(None, cfg, "My Thesis", "t")
            worker._default_resolver(None, cfg, "app-project", "t")
        self.assertEqual(seen, ["My Thesis", "manual", "in_project"])


# ═══════════════════ REVIEW of Level 20's own changes (independent review) ═══════════════════
class TestLevel20Review(unittest.TestCase):
    """Regressions that Level 20 itself introduced and an independent review found; each test reproduces one."""

    # (4) classifier: test_x.py, 'page/site', library names, rename/delete
    def test_4_classifier_without_regressions(self):
        for q, expected in (
            ("Create calc.py and test_calc.py", "python_test"),
            ("Write utils.py and its test_utils.py", "python_test"),
            ("write calc.py and calc_test.py", "python_test"),
            ("make a python script and get it tested", "python_test"),
            ("Make me a page for my business", "web"),
            ("Create a site for my shop", "web"),
            ("Create a website using Chart.js with my data", "web"),
            ("make a website that loads products.json", "web"),
            ("write a 2 page summary in summary.md", "open"),
            ("write a summary of 2 pages", "open"),
            ("save the notes on some site", "open"),
            ("Crea calc.py y test_calc.py", "python_test"),
            ("Escribe utils.py y su test_utils.py", "python_test"),
            ("haz un script python y testéalo", "python_test"),
            ("Hazme una página de presentación para mi negocio", "web"),
            ("Crea un sitio para mi tienda", "web"),
            ("Crea una web usando Chart.js con mis datos", "web"),
            ("haz una web que cargue productos.json", "web"),
            ("escribe un resumen de 2 páginas en resumen.md", "open"),
            ("guarda las notas en algún sitio de notas.md", "open"),
        ):
            self.assertEqual(worker.classify_task(q), expected, q)

    def test_4b_named_files_skip_libraries_and_files_that_go_away(self):
        self.assertEqual(worker.named_files("Rename main.py to app.py"), ["app.py"])
        self.assertEqual(worker.named_files("Delete temp.txt and create clean.txt"), ["clean.txt"])
        self.assertEqual(worker.named_files("a website using Chart.js in index.html"), ["index.html"])
        self.assertEqual(worker.named_files("Renombra main.py a app.py"), ["app.py"])
        self.assertEqual(worker.named_files("Borra temp.txt y crea limpio.txt"), ["limpio.txt"])
        self.assertEqual(worker.named_files("una web usando Chart.js en index.html"), ["index.html"])
        self.assertEqual(worker.extensions_named("a bot with discord.py and node.js"), set())

    # (6) when the lock is released, WORKDIR goes back to the NEUTRAL one; on resuming, to the worker's own
    def test_6_waiting_does_not_leave_the_worker_workdir_to_others(self):
        import orchestrator

        neutral, own = _tmp(), _tmp()
        wd, cd = orchestrator.WORKDIR, orchestrator.CHATDIR
        lock = threading.Lock()
        lock.acquire()
        worker._WAIT.lock, worker._WAIT.neutral = lock, (neutral, neutral)
        orchestrator._set_workdir(own)
        seen = {}
        try:
            with worker._releasing_agent_lock():
                seen["during"] = orchestrator.WORKDIR
                seen["free"] = lock.acquire(blocking=False)
                if seen["free"]:
                    lock.release()
            seen["after"] = orchestrator.WORKDIR
        finally:
            worker._WAIT.lock = worker._WAIT.neutral = None
            lock.release()
            orchestrator._set_workdir(wd, cd)
        self.assertEqual(seen["during"], neutral.resolve())
        self.assertTrue(seen["free"])
        self.assertEqual(seen["after"], own.resolve())

    def test_6b_run_with_lock_saves_and_clears_the_neutral_workdir(self):
        import orchestrator

        lock = threading.Lock()
        seen = {}
        with patch.object(worker, "run", lambda *a, **k: seen.setdefault("neutral", worker._WAIT.neutral)):
            worker.run_with_lock(lock, "task", "p", lambda e: None)
        self.assertEqual(seen["neutral"], (orchestrator.WORKDIR, orchestrator.CHATDIR))
        self.assertIsNone(getattr(worker._WAIT, "neutral", None))

    # (10a) an unreadable file does not produce "phantom" snapshots
    @unittest.skipIf(hasattr(os, "geteuid") and os.geteuid() == 0, "root can read a chmod 0 file")
    def test_10a_unreadable_file_is_not_a_deletion(self):
        b = _tmp()
        f = b / "secret.txt"
        f.write_text("x", encoding="utf-8")
        os.chmod(f, 0)
        try:
            sid = time_machine.create_snapshot(b, "t")
            self.assertFalse(time_machine.record_changes(b, sid))
        finally:
            os.chmod(f, 0o644)

    def test_10a_a_file_turned_into_a_folder_IS_a_deletion(self):
        b = _tmp()
        (b / "a").write_text("x", encoding="utf-8")
        sid = time_machine.create_snapshot(b, "t")
        (b / "a").unlink()
        (b / "a").mkdir()
        self.assertTrue(time_machine.record_changes(b, sid))
        self.assertIn("a", time_machine.plan_undo(b, sid)["recreate"])


# ─────────────── self-audit of the INSTALLED dependencies (not only requirements.txt) ───────────────
class TestInstalledDependencyAudit(unittest.TestCase):
    """Level 20: pip-audit runs over what is INSTALLED in the venv; the ACCEPTED risks (with their reason)
    are a WARN, and any other vulnerable package is still a FAIL."""

    def _with(self, deps):
        out = json.dumps({"dependencies": deps})
        return patch.object(self_audit, "_run", return_value=(1, out, ""))

    def test_only_accepted_risks_is_a_warn_with_the_reason(self):
        with self._with(
            [
                {"name": "torch", "version": "2.12.1", "vulns": [{"id": "X"}]},
                {"name": "requests", "version": "2.0", "vulns": []},
            ]
        ):
            r = self_audit.sast_pip_audit()
        self.assertEqual(r["result"], "WARN")
        self.assertIn("torch", r["detail"])

    def test_a_new_vulnerable_package_is_a_fail(self):
        with self._with(
            [
                {"name": "torch", "version": "2.12.1", "vulns": [{"id": "X"}]},
                {"name": "pillow", "version": "12.2.0", "vulns": [{"id": "Y"}]},
            ]
        ):
            r = self_audit.sast_pip_audit()
        self.assertEqual(r["result"], "FAIL")
        self.assertIn("pillow", r["evidence"])

    def test_audits_the_venv_when_present(self):
        seen = {}

        def run(cmd):
            seen["cmd"] = cmd
            return (0, json.dumps({"dependencies": [{"name": "x", "version": "1", "vulns": []}]}), "")

        with (
            patch.object(self_audit, "_run", run),
            patch.object(self_audit, "_venv_site_packages", lambda: "/tmp/venv/site-packages"),
        ):
            self_audit.sast_pip_audit()
        self.assertIn("--path", seen["cmd"])


if __name__ == "__main__":
    unittest.main()
