"""Level 20 regressions: orchestrator, knowledge base (RAG) and skills.

Each test reproduces a REAL bug found during the Level 20 audit and pins its fix. No model, no network,
no server (like the rest of the suite). Standard-library `unittest` only.

  H1  the VERDICT parser is a whitelist (markdown bold, 'NOT OK', 'FAIL'... are no longer an OK)
  H5  the chat verifier is READ-ONLY, and if it touches the workspace it is reverted and its OK is void
  H6  'do the tests pass?' understands pytest (not only unittest) and does not count skip/xfail
  D5  a project turn that crashes still records its diff (undoable) and saves the turn
  Q1  work on code/files skips the text-only shortcuts (docs/web/skills)
  Q2  mentioned-document detection: no short/numeric captures, most recent wins, exact source filter
  Q3  the RAG tokenizer folds accents and drops stop words (ES + EN); per-document chunk cap; old
      indexes are rebuilt
  Q5  skill scripts measure the user's TEXT (not the instruction); a failing script injects nothing;
      readability/BibTeX/APA/spelling script fixes; generic verbs do not fire a skill
  Q6  plan steps that name a deliverable are kept
  REV review of Level 20's own changes: writing requests keep their skill, the RAG guard is for orders
      only, outputs of running the code do not void a verdict (uploads are never deleted), tolerant
      VERDICT line, a legitimate skip does not break the project safeguard, no run/open steps in a plan

Run: python -m unittest tests.test_orchestrator_rag_skills -v
"""

import importlib.util
import io
import json
import os
import sys
import tempfile
import time
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import agent  # noqa: E402
import knowledge  # noqa: E402
import orchestrator  # noqa: E402
import projects  # noqa: E402
import skills  # noqa: E402
import time_machine  # noqa: E402
import worker  # noqa: E402
from orchestrator import _parse_verdict  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def _tmp():
    return Path(tempfile.mkdtemp()).resolve()


def _has_test_runner():
    return importlib.util.find_spec("test_runner") is not None


def _python3_with_pytest():
    """The interpreter running these tests has pytest and a `python3` next to it (the orchestrator runs
    'python3' with the venv bin first on PATH)."""
    return importlib.util.find_spec("pytest") is not None and (Path(sys.executable).parent / "python3").exists()


# ─────────────────────────────── H1 · verdict parser ───────────────────────────────
class TestH1Verdict(unittest.TestCase):
    def test_failure_variants_are_NEVER_ok(self):
        for t in (
            "**VERDICT:** FAILURE: the tests fail",
            "**VERDICT**: FAILURE",
            "VERDICT: NOT OK",
            "VERDICT: NO OK",
            "VERDICT: FAIL the division test",
            "VERDICT - FAILURE: README is missing",
            "- VERDICT: failed",
            "VERDICT: ERROR",
            "VERDICT: KO",
            "VERDICT: FAILURE: x",
        ):
            self.assertIs(_parse_verdict(t)[0], False, t)

    def test_ambiguous_or_in_prose_is_inconclusive_not_ok(self):
        for t in (
            "VERDICT: ???",
            "VERDICT: partially correct",
            "I will give VERDICT: OK only if the tests pass.\nThe tests fail.",
            "Remember to end with VERDICT:",
            "VERDICT:",
            "",
        ):
            self.assertIsNone(_parse_verdict(t)[0], t)

    def test_legitimate_oks_are_still_ok(self):
        for t in (
            "VERDICT: OK",
            "**VERDICT:** OK",
            "`VERDICT: OK`",
            "1. VERDICT: OK",
            "All checked.\nVERDICT: OK, no failures anymore",
        ):
            self.assertIs(_parse_verdict(t)[0], True, t)

    def test_the_LAST_verdict_line_counts(self):
        self.assertIs(_parse_verdict("VERDICT: OK\n...but then...\nVERDICT: FAILURE: x")[0], False)
        self.assertIs(_parse_verdict("VERDICT: FAILURE: x\nfixed\nVERDICT: OK")[0], True)

    def test_failure_detail(self):
        self.assertEqual(_parse_verdict("**VERDICT:** FAILURE: test_calc.py is missing")[1], "test_calc.py is missing")
        self.assertEqual(_parse_verdict("VERDICT: NOT OK: the sum subtracts")[1], "the sum subtracts")


# ─────────────────────────────── H5 · read-only verifier ───────────────────────────────
class TestH5VerifierReadOnly(unittest.TestCase):
    def setUp(self):
        self._wd, self._awd, self._ra = orchestrator.WORKDIR, agent.WORKDIR, orchestrator.run_agent
        self.d = _tmp()
        orchestrator.WORKDIR = agent.WORKDIR = self.d

    def tearDown(self):
        orchestrator.WORKDIR, agent.WORKDIR, orchestrator.run_agent = self._wd, self._awd, self._ra

    def test_verify_asks_for_read_only(self):
        seen = {}

        def fake(cli, mod, msgs, cb=False, **kw):
            seen.update(kw)
            return "VERDICT: OK"

        orchestrator.run_agent = fake
        self.assertIs(orchestrator.verify(None, None, "task")[0], True)
        self.assertTrue(seen.get("read_only"))

    def test_a_verifier_that_FIXES_the_code_is_void_and_reverted(self):
        (self.d / "calc.py").write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")

        def fake(cli, mod, msgs, cb=False, **kw):  # 'fixes' it through the shell and says OK
            (self.d / "calc.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
            (self.d / "test_calc.py").write_text("def test_x():\n    pass\n", encoding="utf-8")
            return "VERDICT: OK"

        orchestrator.run_agent = fake
        ok, detail = orchestrator.verify(None, None, "create calc.py with add and its test")
        self.assertIsNone(ok)
        self.assertIn("modified the workspace", detail)
        self.assertIn("a - b", (self.d / "calc.py").read_text(encoding="utf-8"))  # reverted
        self.assertFalse((self.d / "test_calc.py").exists())  # what it created: gone

    def test_a_verifier_that_crashes_after_writing_is_reverted_too(self):
        (self.d / "notes.md").write_text("original\n", encoding="utf-8")

        def fake(cli, mod, msgs, cb=False, **kw):
            (self.d / "notes.md").write_text("changed\n", encoding="utf-8")
            raise ConnectionError("server down")

        orchestrator.run_agent = fake
        ok, _detail = orchestrator.verify(None, None, "task")
        self.assertIsNone(ok)
        self.assertEqual((self.d / "notes.md").read_text(encoding="utf-8"), "original\n")


# ─────────────────────────────── H6 · do the workspace tests pass? ───────────────────────────────
def _py_project(d, test_src, mod_src="def add(a, b):\n    return a + b\n"):
    (d / "calc.py").write_text(mod_src, encoding="utf-8")
    (d / "test_calc.py").write_text(test_src, encoding="utf-8")


_TEST_OK = "from calc import add\n\ndef test_add():\n    assert add(2, 2) == 4\n"
_MOD_BROKEN = "def add(a, b):\n    return a - b\n"


@unittest.skipUnless(_has_test_runner(), "test_runner.py not available yet")
class TestH6TestsPassOrchestrator(unittest.TestCase):
    def setUp(self):
        self._wd, self._vb = orchestrator.WORKDIR, orchestrator.VENV_BIN
        self.d = _tmp()
        orchestrator.WORKDIR = self.d
        orchestrator.VENV_BIN = Path(sys.executable).parent  # 'python3' = the interpreter running the suite

    def tearDown(self):
        orchestrator.WORKDIR, orchestrator.VENV_BIN = self._wd, self._vb

    @unittest.skipUnless(_python3_with_pytest(), "pytest not installed for this interpreter")
    def test_pytest_style_tests_DO_count(self):
        _py_project(self.d, _TEST_OK)
        self.assertTrue(orchestrator._tests_pass())

    @unittest.skipUnless(_python3_with_pytest(), "pytest not installed for this interpreter")
    def test_red_or_skipped_tests_do_not_pass(self):
        _py_project(self.d, _TEST_OK, _MOD_BROKEN)
        self.assertFalse(orchestrator._tests_pass())
        (self.d / "test_calc.py").write_text(
            "import pytest\n\n@pytest.mark.skip\ndef test_x():\n    pass\n", encoding="utf-8"
        )
        self.assertFalse(orchestrator._tests_pass())

    def test_unittest_tests_still_count(self):
        (self.d / "calc.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
        (self.d / "test_calc.py").write_text(
            "import unittest\nfrom calc import add\n\n"
            "class T(unittest.TestCase):\n    def test_a(self):\n"
            "        self.assertEqual(add(1, 1), 2)\n",
            encoding="utf-8",
        )
        self.assertTrue(orchestrator._tests_pass())

    def test_no_tests_is_not_a_pass(self):
        (self.d / "calc.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
        self.assertFalse(orchestrator._tests_pass())


# ─────────────────────────────── D5 · a turn that crashes ───────────────────────────────
class TestD5TurnThatCrashes(unittest.TestCase):
    def test_what_a_crashing_task_changed_can_be_undone(self):
        cfg = {"AGENT_PROJECTS_DIR": str(_tmp()), "PROJECT_MEMORY": "false"}
        wd, awd, rt = orchestrator.WORKDIR, agent.WORKDIR, orchestrator.resolve_task

        def crashes(router, cfg_, task, **kw):
            (orchestrator.WORKDIR / "half_done.py").write_text("x = 1\n", encoding="utf-8")
            raise ConnectionError("the server went down")

        orchestrator.resolve_task = crashes
        try:
            with self.assertRaises(ConnectionError):
                orchestrator.resolve_in_project({"executor": (None, None, "L")}, cfg, "p", "do something")
        finally:
            orchestrator.resolve_task = rt
            orchestrator._set_workdir(wd)
            agent.WORKDIR = awd
        prev = orchestrator.preview_undo(cfg, "p")
        self.assertIsNotNone(prev["plan"], "the snapshot was left without changes.json")
        self.assertEqual(prev["plan"]["delete"], ["half_done.py"])
        conv = projects.load_conversation(orchestrator._chat_dir(cfg, "p"))
        self.assertTrue(any("interrupted" in str(m) for m in conv))

    def test_cli_undo_with_an_unreadable_snapshot_does_not_crash(self):
        cfg = {"AGENT_PROJECTS_DIR": str(_tmp()), "PROJECT_MEMORY": "false"}
        projects.create_project(cfg, "p")
        out = io.StringIO()
        with (
            patch.object(orchestrator, "load_config", return_value=cfg),
            patch.object(orchestrator, "preview_undo", return_value={"id": "s1", "meta": {}, "plan": None}),
            patch.object(orchestrator, "apply_undo") as applied,
            patch.object(sys, "argv", ["orchestrator.py", "--undo", "--project", "p", "--yes"]),
            redirect_stdout(out),
        ):
            orchestrator.main()  # before: TypeError ('NoneType' is not subscriptable)
        self.assertIn("incomplete or damaged", out.getvalue())
        applied.assert_not_called()

    def test_a_partial_undo_keeps_the_snapshot(self):
        cfg = {"AGENT_PROJECTS_DIR": str(_tmp()), "PROJECT_MEMORY": "false"}
        projects.create_project(cfg, "p")
        with (
            patch.object(time_machine, "undo", return_value={"delete": [], "failed": ["a.txt"]}),
            patch.object(time_machine, "delete_snapshot") as deleted,
        ):
            plan = orchestrator.apply_undo(cfg, "p", sid="s1")
        self.assertEqual(plan["failed"], ["a.txt"])
        deleted.assert_not_called()
        with (
            patch.object(time_machine, "undo", return_value={"delete": [], "failed": []}),
            patch.object(time_machine, "delete_snapshot") as deleted,
        ):
            orchestrator.apply_undo(cfg, "p", sid="s1")
        deleted.assert_called_once()


# ═══════════════════════════════ QUALITY ═══════════════════════════════
@unittest.skipUnless(hasattr(worker, "extensions_named"), "worker.extensions_named not available yet")
class TestQ1ShortcutsDoNotHijackWork(unittest.TestCase):
    def test_work_on_files_goes_to_the_triage(self):
        for q in (
            "summarize the README.md",
            "fix the bug in login.py",
            "find and fix the bug on the website",
            "refactor utils.py",
            "fix this bug: the login button does not respond when clicked",
            "fix the error in the login function",
            "resume el README.md",
            "arregla el bug del login.py",
            "busca y arregla el bug de la web",
            "refactoriza utils.py",
            "corrige el error de la función de login",
        ):
            self.assertTrue(orchestrator._works_on_files(q), q)

    def test_questions_and_text_requests_keep_their_shortcuts(self):
        for q in (
            "what does notes.md say about the milestone?",
            "humanize this text: hello",
            "search news about AI",
            "what changed in my CV?",
            "translate to Spanish: hello world",
            "tell me the time",
            "¿qué dice notas.md sobre el hito?",
            "humaniza este texto: hola",
            "busca noticias de IA",
            "¿qué cambió en mi CV?",
            "traduce al inglés: hola mundo",
            "dime la hora",
        ):
            self.assertFalse(orchestrator._works_on_files(q), q)

    def test_edit_verbs_over_prose_keep_the_skill_shortcuts(self):
        # English skill triggers that are themselves edit verbs over text must still reach the skill.
        for q in (
            "fix the spelling: hola q tal estas, espero que bien",
            "fix typos",
            "remove the AI tone: we must leverage our synergies today",
            "rewrite it so it sounds natural: Moreover, it is important to note that",
            "humanize this text: we must fix the onboarding flow and update the docs",
            "update my notes with today's news",  # Level 20 (review): no code object, no file named
            "corrige la ortografía de este texto: hola q tal",
            "reescribe para que suene natural",
            "humaniza esto: tenemos que arreglar el onboarding",
            "actualiza mis notas con lo de hoy",
        ):
            self.assertFalse(orchestrator._works_on_files(q), q)

    def test_resolve_task_skips_docs_web_and_skills_for_work(self):
        called = []
        router = {r: (None, "m", "L") for r in ("executor", "planner", "verifier")}
        with (
            patch.object(orchestrator, "answer_from_documents", side_effect=lambda *a, **k: called.append("docs")),
            patch.object(orchestrator, "answer_with_skill", side_effect=lambda *a, **k: called.append("skill")),
            patch.object(orchestrator, "_is_web_search", side_effect=lambda *a, **k: called.append("web")),
            patch.object(orchestrator, "_triage", return_value="CHAT"),
            patch.object(orchestrator, "chat", return_value="answer"),
        ):
            self.assertEqual(orchestrator.resolve_task(router, {}, "fix the bug in login.py"), "answer")
        self.assertEqual(called, [])


class TestQ2MentionedDocument(unittest.TestCase):
    def _base(self, names):
        b = _tmp()
        for i, n in enumerate(names):
            (b / n).write_text(f"content of the document {n} with data", encoding="utf-8")
            t = time.time() - 1000 + i * 10
            os.utime(b / n, (t, t))
        knowledge.index_docs(b, force=True)
        return b

    def test_a_short_word_of_the_name_does_not_capture_questions(self):
        b = self._base(["how_to_deploy.md", "meeting_2024.md"])
        self.assertIsNone(orchestrator._mentioned_document("how are you today?", b))
        self.assertIsNone(orchestrator._mentioned_document("what happened in 2024", b))
        self.assertEqual(orchestrator._mentioned_document("how do I deploy it", b), "how_to_deploy.md")

    def test_with_several_candidates_the_most_recent_wins(self):
        b = self._base(["CV_2023.md", "CV_2024.md"])
        self.assertEqual(orchestrator._mentioned_document("what does my CV say", b), "CV_2024.md")

    def test_the_source_filter_is_exact(self):
        self.assertTrue(knowledge._same_source("cv.md", "cv.md"))
        self.assertTrue(knowledge._same_source("docs/cv.md", "cv.md"))
        self.assertFalse(knowledge._same_source("old_cv.md", "cv.md"))

    def test_search_restricted_to_a_file_does_not_leak_a_similar_name(self):
        b = _tmp()
        (b / "cv.md").write_text("Skills: Python and statistics.", encoding="utf-8")
        (b / "old_cv.md").write_text("Skills: Python and accounting.", encoding="utf-8")
        knowledge.index_docs(b, force=True)
        res = knowledge.search(b, "python skills", only_source="cv.md")
        self.assertTrue(res)
        self.assertEqual({r["source"] for r in res}, {"cv.md"})

    def test_an_order_to_modify_a_document_is_not_answered_with_text(self):
        wd = orchestrator.WORKDIR
        b = _tmp()
        (b / "notes.md").write_text("Milestone: March.", encoding="utf-8")
        orchestrator.WORKDIR = b
        try:
            with patch.object(orchestrator, "chat", return_value="TEXT ANSWER"):
                self.assertIsNone(orchestrator.answer_from_documents(None, "m", "update my notes with the new date"))
                self.assertEqual(
                    orchestrator.answer_from_documents(None, "m", "what do my notes say about the milestone?"),
                    "TEXT ANSWER",
                )
        finally:
            orchestrator.WORKDIR = wd


class TestQ3Search(unittest.TestCase):
    def test_accents_folded_and_stop_words_dropped(self):
        self.assertEqual(knowledge._tokenize("¿Qué información hay del año?"), ["informacion", "ano"])
        self.assertEqual(knowledge._tokenize("What is the information about the café?"), ["information", "cafe"])

    def test_accented_words_are_not_split(self):
        # Public-repo bug: the tokenizer was [a-z0-9]+ on the raw text, so 'información' became
        # 'informaci' (+ a dropped 'n') and 'año' vanished.
        self.assertEqual(knowledge._tokenize("información"), ["informacion"])
        self.assertEqual(knowledge._tokenize("Año, niño, pingüino"), ["ano", "nino", "pinguino"])

    def test_an_unrelated_question_returns_nothing(self):
        b = _tmp()
        (b / "coffee.md").write_text("Coffee is grown in tropical areas and then roasted.", encoding="utf-8")
        (b / "omelette.md").write_text("The omelette has eggs, potato and onion.", encoding="utf-8")
        knowledge.index_docs(b, force=True)
        self.assertEqual(knowledge.search(b, "What does it say about quantum physics?"), [])
        self.assertTrue(knowledge.search(b, "information about coffee"))

    def test_spanish_document_found_without_accents(self):
        b = _tmp()
        (b / "cafe.md").write_text("El café se cultiva en zonas tropicales y se tuesta.", encoding="utf-8")
        (b / "tortilla.md").write_text("La tortilla lleva huevo, patata y cebolla.", encoding="utf-8")
        knowledge.index_docs(b, force=True)
        self.assertEqual(knowledge.search(b, "¿Qué dice sobre la física cuántica?"), [])
        res = knowledge.search(b, "informacion sobre el cafe")
        self.assertTrue(res)
        self.assertEqual(res[0]["source"], "cafe.md")

    def test_a_big_file_does_not_take_the_whole_index(self):
        b = _tmp()
        (b / "data.csv").write_text("row,value\n" * 400_000, encoding="utf-8")
        (b / "notes").mkdir()
        (b / "notes" / "meeting.md").write_text("Launch date decided: March.", encoding="utf-8")
        knowledge.index_docs(b, force=True)
        self.assertTrue(knowledge.search(b, "launch date decided"))

    def test_an_old_index_is_rebuilt(self):
        b = _tmp()
        (b / "a.md").write_text("emperor penguin", encoding="utf-8")
        knowledge.index_docs(b, force=True)
        path = b / knowledge.KB_DIR / knowledge.INDEX
        data = json.loads(path.read_text(encoding="utf-8"))
        data.pop("version")
        path.write_text(json.dumps(data), encoding="utf-8")
        knowledge.index_docs(b)
        self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["version"], knowledge.INDEX_VERSION)


def _load_skill_script(rel, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "skills" / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class TestQ5Skills(unittest.TestCase):
    def test_the_script_measures_the_text_not_the_instruction(self):
        f = orchestrator._text_to_analyze
        self.assertEqual(
            f("What is the readability of this text? The system processes ten thousand requests."),
            "The system processes ten thousand requests.",
        )
        self.assertEqual(f("check the spelling: helo how r u, hope al is well"), "helo how r u, hope al is well")
        self.assertEqual(
            f("proofread this\nThe sistem procesess ten thousand requests."),
            "The sistem procesess ten thousand requests.",
        )
        self.assertEqual(
            f("¿Qué legibilidad tiene este texto? El sistema procesa diez mil peticiones."),
            "El sistema procesa diez mil peticiones.",
        )
        self.assertEqual(f("humanize this text"), "humanize this text")

    def _run_script(self, src):
        """Run a throwaway skill script through the real path (agent.run_bash). `python` resolves to the
        interpreter running the suite (the repo's own .venv is not part of this checkout)."""
        d = _tmp()
        (d / "script.py").write_text(src, encoding="utf-8")
        wd, vb = agent.WORKDIR, agent.VENV_BIN
        agent.WORKDIR, agent.VENV_BIN = d, Path(sys.executable).parent
        try:
            return orchestrator._run_skill_script(d / "script.py", "text")
        finally:
            agent.WORKDIR, agent.VENV_BIN = wd, vb

    def test_a_failing_script_injects_no_findings(self):
        self.assertEqual(self._run_script("raise SystemExit('ERROR: I broke')\n"), "")
        self.assertEqual(self._run_script("print('half a finding')\nraise RuntimeError('boom')\n"), "")

    @unittest.skipUnless((Path(sys.executable).parent / "python").exists(), "no `python` next to the interpreter")
    def test_a_working_script_injects_its_output(self):
        self.assertEqual(self._run_script("print('finding: one')\n"), "finding: one")

    def test_readability_does_not_split_sentences_on_numbers_or_lists(self):
        m = _load_skill_script("readability/measure.py", "measure_l20")
        self.assertEqual(len(m._sentences("Procesa 10.000 peticiones en 2.5 segundos. Es rápido.")), 2)
        self.assertEqual(len(m._sentences("1. Instala\n2. Configura\n3. Arranca")), 3)

    def test_bibtex_empty_field_is_missing_and_parenthesis_form(self):
        m = _load_skill_script("bibtex-citations/review_bibtex.py", "bib_l20")
        r = m.validate("@article{k, author = {}, title = {T}, journal = {J}, year = {2020}}")
        self.assertIn("author", r["entries"][0]["missing"])
        r2 = m.validate("@article(k2, author = {A}, title = {T}, journal = {J}, year = {2020})")
        self.assertEqual(r2["n_entries"], 1)
        self.assertEqual(r2["entries"][0]["missing"], [])

    def test_apa_does_not_mistake_prose_for_a_reference(self):
        m = _load_skill_script("apa-citations/review_apa.py", "apa_l20")
        self.assertEqual(m.detect("Finalmente, López sostiene que el método funciona."), [])
        self.assertTrue(m.detect("Garcia, Ana. Trabajo sin fecha."))
        self.assertTrue(m.detect("Pérez, J. Trabajo sin fecha."))

    def test_spelling_accepts_the_space_before_percent(self):
        m = _load_skill_script("spelling/review.py", "spelling_l20")
        self.assertFalse(m.RX_SPACE_BEFORE_PUNCT.search("subió un 20 % este año"))
        self.assertTrue(m.RX_SPACE_BEFORE_PUNCT.search("hola , qué tal"))

    def test_a_generic_verb_does_not_fire_a_skill(self):
        cfg = {}
        for q in (
            "write a poem",
            "write a short story",
            "what time is it",
            "escribe un poema",
            "check the login page",
            "fix the bug in the login page",
            "tell me a joke",
        ):
            self.assertIsNone(skills.select(q, cfg), q)
        self.assertIsNotNone(skills.select("write an abstract of this paper", cfg))
        for q, name in (
            ("check my spelling", "spelling"),
            ("fix typos", "spelling"),
            ("check my bibtex", "bibtex-citations"),
            ("review my APA citations", "apa-citations"),
            ("make it sound human", "humanizer"),
            ("translate this to english", "translate"),
            ("how readable is this", "readability"),
        ):
            s = skills.select(q, cfg)
            self.assertEqual(s.name if s else None, name, q)


class TestQ6PlanKeepsSteps(unittest.TestCase):
    def test_steps_with_a_deliverable_stay_and_mechanics_go(self):
        steps = [
            "Create calc.py",
            "Tests: create test_calc.py with pytest",
            "Test the script",
            "Run the tests",
            "Save the data to sales.csv",
            "Check that everything works",
            "Test and add unit tests",
            "Verify division by zero in calc.py",  # Level 20 (review): checking is the verifier's job
        ]
        self.assertEqual(
            orchestrator._filter_steps(steps),
            [
                "Create calc.py",
                "Tests: create test_calc.py with pytest",
                "Save the data to sales.csv",
                "Test and add unit tests",
            ],
        )

    def test_spanish_steps(self):
        self.assertEqual(
            orchestrator._filter_steps(["Pruebas: crear test_calc.py", "Crear calc.py"]),
            ["Pruebas: crear test_calc.py", "Crear calc.py"],
        )

    def test_a_word_that_only_starts_like_a_verb_is_not_dropped(self):
        self.assertEqual(
            orchestrator._filter_steps(["Testimonials page with three quotes", "OpenAPI spec"]),
            ["Testimonials page with three quotes", "OpenAPI spec"],
        )


# ═══════════════════ REVIEW of Level 20's own changes (10 findings of an independent review) ═══════════════════
class TestLevel20Review(unittest.TestCase):
    """Regressions that Level 20 itself introduced and an independent review found; each test reproduces one."""

    # (1) the writing skills keep ordinary requests
    @unittest.skipUnless(hasattr(worker, "extensions_named"), "worker.extensions_named not available yet")
    def test_1_writing_requests_keep_their_skill(self):
        for q in (
            "Fix: we went to the cinema yesterday",
            "Fix this email: hi",
            "Fix the style: x",
            "Rewrite this email more formally: hi",
            "Fix the errors in this text: x",
            "Fix the spelling: the file config.txt has typos",
            "Humanize this text: the README.md is long",
            "Corrige: Ayer fuimos al sine",
            "Corrige este correo: hola",
            "Corrige el estilo: x",
            "Reescribe este email de forma más profesional: hola",
            "Corrige los errores de este texto: x",
            "Corrige la ortografía: el archivo config.txt tiene herrores",
            "Humaniza este texto: el README.md es largo",
        ):
            self.assertFalse(orchestrator._works_on_files(q), q)

    # (2) questions about YOUR documents with a verb inside still go to the RAG; ORDERS do not
    def test_2_rag_guard_only_for_orders(self):
        def is_order(q):
            return bool(orchestrator._RX_WORK_ORDER.search(q) and not orchestrator._RX_QUESTION.search(q))

        for q in (
            "What does my CV say about how to optimize costs?",
            "According to my notes, how is this fixed?",
            "Search my notes for how to update the firmware",
            "What changes in version 2 according to my notes?",
            "¿Qué dice mi CV sobre cómo optimizar costes?",
            "Según mis apuntes, ¿cómo se corrige esto?",
            "Busca en mis apuntes cómo actualizar el firmware",
            "¿Qué cambia en la versión 2 según mis notas?",
        ):
            self.assertFalse(is_order(q), q)
        for q in (
            "update my notes with today's news",
            "please, fix my report.docx",
            "actualiza mis notas con lo de hoy",
            "por favor, corrige mi informe.docx",
        ):
            self.assertTrue(is_order(q), q)

    def test_2b_a_question_with_a_work_verb_is_answered_from_the_documents(self):
        wd = orchestrator.WORKDIR
        orchestrator.WORKDIR = _tmp()
        try:
            with (
                patch.object(knowledge, "has_index", return_value=True),
                patch.object(knowledge, "search", return_value=[]) as searched,
            ):
                r = orchestrator.answer_from_documents(
                    None, None, "According to my notes, how do I update the firmware?"
                )
                self.assertIsNotNone(r)
                searched.assert_called_once()
                self.assertIsNone(orchestrator.answer_from_documents(None, None, "update my notes with today's news"))
        finally:
            orchestrator.WORKDIR = wd

    # (3) running the code while verifying may generate files: that does NOT void the verdict
    def _verify_with(self, fake, task):
        wd, awd, ra = orchestrator.WORKDIR, agent.WORKDIR, orchestrator.run_agent
        orchestrator.run_agent = fake
        try:
            return orchestrator.verify(None, None, task)
        finally:
            orchestrator.WORKDIR, agent.WORKDIR, orchestrator.run_agent = wd, awd, ra

    def test_3_outputs_of_running_do_not_void_and_uploads_are_not_deleted(self):
        wd, awd = orchestrator.WORKDIR, agent.WORKDIR
        d = _tmp()
        orchestrator.WORKDIR = agent.WORKDIR = d
        (d / "app.py").write_text("print(1)\n", encoding="utf-8")
        uploaded = d / "uploaded_meanwhile.pdf"

        def fake(cli, mod, msgs, cb=False, **kw):
            (d / "tasks.db").write_bytes(b"sqlite")  # output of RUNNING the app
            (d / "report.txt").write_text("generated 12:00", encoding="utf-8")
            uploaded.write_bytes(b"%PDF")  # the user uploads something right now
            orchestrator.PROTECTED_FILES.add(str(uploaded))
            return "VERDICT: OK"

        try:
            ok, _ = self._verify_with(fake, "create app.py that saves tasks in sqlite")
        finally:
            orchestrator.WORKDIR, agent.WORKDIR = wd, awd
            orchestrator.PROTECTED_FILES.discard(str(uploaded))
        self.assertIs(ok, True)
        self.assertFalse((d / "tasks.db").exists())
        self.assertFalse((d / "report.txt").exists())
        self.assertTrue(uploaded.exists(), "a file the user uploaded during the verification was deleted")

    def test_3b_creating_the_missing_deliverable_DOES_void(self):
        wd, awd = orchestrator.WORKDIR, agent.WORKDIR
        d = _tmp()
        orchestrator.WORKDIR = agent.WORKDIR = d
        try:
            for name, task in (
                ("notes.txt", "write notes.txt with a summary"),
                ("notas.txt", "escribe notas.txt con un resumen"),
            ):

                def fake(cli, mod, msgs, cb=False, _name=name, **kw):
                    (d / _name).write_text("made by the verifier", encoding="utf-8")
                    return "VERDICT: OK"

                ok, _ = self._verify_with(fake, task)
                self.assertIsNone(ok, task)
                self.assertFalse((d / name).exists(), task)
        finally:
            orchestrator.WORKDIR, agent.WORKDIR = wd, awd

    def test_3c_a_re_upload_during_the_verification_is_not_overwritten(self):
        wd, awd = orchestrator.WORKDIR, agent.WORKDIR
        d = _tmp()
        orchestrator.WORKDIR = agent.WORKDIR = d
        doc = d / "data.md"
        doc.write_text("old version\n", encoding="utf-8")

        def fake(cli, mod, msgs, cb=False, **kw):
            doc.write_text("NEW upload\n", encoding="utf-8")
            orchestrator.PROTECTED_FILES.add(str(doc))
            return "VERDICT: OK"

        try:
            ok, _ = self._verify_with(fake, "summarize data.md")
        finally:
            orchestrator.WORKDIR, agent.WORKDIR = wd, awd
            orchestrator.PROTECTED_FILES.discard(str(doc))
        self.assertIs(ok, True)
        self.assertEqual(doc.read_text(encoding="utf-8"), "NEW upload\n")

    def test_3d_an_old_upload_is_not_protected_forever(self):
        wd, awd = orchestrator.WORKDIR, agent.WORKDIR
        d = _tmp()
        orchestrator.WORKDIR = agent.WORKDIR = d
        doc = d / "notes.md"
        doc.write_text("original\n", encoding="utf-8")
        orchestrator.PROTECTED_FILES.add(str(doc))  # uploaded BEFORE this verification

        def fake(cli, mod, msgs, cb=False, **kw):
            doc.write_text("edited by the verifier\n", encoding="utf-8")
            return "VERDICT: OK"

        try:
            ok, _ = self._verify_with(fake, "check notes.md")
        finally:
            orchestrator.WORKDIR, agent.WORKDIR = wd, awd
            orchestrator.PROTECTED_FILES.discard(str(doc))
        self.assertIsNone(ok)
        self.assertEqual(doc.read_text(encoding="utf-8"), "original\n")

    # (5) verdict with emoji, quotes, brackets, 'FINAL', '='
    def test_5_verdict_tolerates_the_format(self):
        for v in (
            "«VERDICT: OK»",
            "✅ **VERDICT: OK**",
            "VERDICT: ✅ OK",
            "FINAL VERDICT: OK",
            "VERDICT FINAL: OK",
            'VERDICT: "OK"',
            "VERDICT: [OK]",
            "VERDICT = OK",
        ):
            self.assertIs(_parse_verdict(v)[0], True, v)
        self.assertEqual(_parse_verdict("❌ VERDICT: FAILURE: the test is missing"), (False, "the test is missing"))
        self.assertEqual(_parse_verdict("«VERDICT: FAILURE: x is missing»"), (False, "x is missing"))
        self.assertIsNone(_parse_verdict("I will give VERDICT: OK if they pass")[0])

    # (8) a legitimate @skip does not turn the project safeguard "red"
    @unittest.skipUnless(_has_test_runner(), "test_runner.py not available yet")
    @unittest.skipUnless(_python3_with_pytest(), "pytest not installed for this interpreter")
    def test_8_a_legitimate_skip_does_not_break_the_safeguard(self):
        wd, vb = orchestrator.WORKDIR, orchestrator.VENV_BIN
        d = _tmp()
        _py_project(
            d, _TEST_OK + "\nimport pytest\n\n@pytest.mark.skip(reason='on mac')\ndef test_other():\n    pass\n"
        )
        orchestrator.WORKDIR, orchestrator.VENV_BIN = d, Path(sys.executable).parent
        try:
            self.assertTrue(orchestrator._tests_pass())
        finally:
            orchestrator.WORKDIR, orchestrator.VENV_BIN = wd, vb

    @unittest.skipUnless(_has_test_runner(), "test_runner.py not available yet")
    def test_8b_the_rule_is_ok_or_passed_with_no_failures(self):
        base = {"ok": False, "total": 2, "passed": 1, "failed": 0, "errors": 0, "skipped": 1, "timeout": False}
        cases = [
            ({}, True),
            ({"passed": 0, "skipped": 2}, False),
            ({"failed": 1}, False),
            ({"errors": 1}, False),
            ({"timeout": True}, False),
        ]
        for change, expected in cases:
            with patch("test_runner.project_tests_pass", return_value={**base, **change}):
                self.assertIs(orchestrator._tests_pass(), expected, change)

    # (9) a named file does not rescue RUN/OPEN steps (they started servers or the browser)
    def test_9_the_plan_does_not_run_or_open_things(self):
        steps = [
            "Create app.py with Flask",
            "Run app.py to start the server",
            "Open index.html in the browser",
            "Launch server.py",
            "Save the data to sales.csv",
        ]
        self.assertEqual(orchestrator._filter_steps(steps), ["Create app.py with Flask", "Save the data to sales.csv"])
        steps_es = [
            "Crear app.py con Flask",
            "Ejecutar app.py para arrancar el servidor",
            "Abrir index.html en el navegador",
            "Lanzar server.py",
            "Guardar los datos en ventas.csv",
        ]
        self.assertEqual(
            orchestrator._filter_steps(steps_es), ["Crear app.py con Flask", "Guardar los datos en ventas.csv"]
        )


# ─────────────────────── obvious greetings are CHAT without asking the model ───────────────────────
class TestObviousChatWithoutModel(unittest.TestCase):
    """Level 20 model benchmark: with some brains "hi, how are you?" was triaged as TASK and the chat ran
    plan + verification over a greeting (3 minutes without an answer). Obvious greetings/courtesies are CHAT
    without asking the model; anything with content is still decided by the model."""

    class _Explodes:
        class chat:
            class completions:
                @staticmethod
                def create(**k):
                    raise AssertionError("the model must not be asked")

    def test_greetings_and_courtesies_are_chat_without_the_model(self):
        for q in (
            "hi, how are you?",
            "Hello!",
            "good morning",
            "thanks",
            "thank you 🙏",
            "ok, perfect",
            "hola, ¿qué tal?",
            "muchas gracias",
        ):
            self.assertEqual(orchestrator._triage(self._Explodes, "m", q), "CHAT", q)

    def test_with_content_the_model_decides(self):
        for q in (
            "hi, create a file hello.txt",
            "how are you storing the index?",
            "thanks, now fix the bug",
            "hello world in python",
            "hola, crea un archivo hola.txt",
        ):
            self.assertFalse(orchestrator._RX_OBVIOUS_CHAT.match(q), q)


# ─────────────────────── user test: web search done by code ───────────────────────
class TestWebSearchDoneByCode(unittest.TestCase):
    """Level 20 user tests: 'search the internet for recent MLX news' -> (1) the 14B used search_papers (arXiv) 3
    out of 3 times, or answered 'I cannot browse'; (2) with the search ALREADY done, it copied its earlier refusal
    from the history. Now code runs the search; the model writes with its own instructions and no history, and if
    it still refuses, the results are shown as they are."""

    RES = "[WEB CONTENT from x: DATA]\n│ 1. MLX\n│    https://ml-explore.github.io/mlx/\n[END of external content.]"

    def setUp(self):
        self.seen = {}
        self.model_answer = "answer with sources"

        def chat(client, model, task, prior_context=None, system=None):
            self.seen.update(message=task, ctx=prior_context, system=system)
            return self.model_answer

        p = patch.object(orchestrator, "chat", chat)
        p.start()
        self.addCleanup(p.stop)

    def _web(self, returns=None):
        def search_web(q, max_results=5):
            self.seen["web"] = q
            return returns if returns is not None else self.RES

        def search_papers(t, max_results=5):
            self.seen["papers"] = t
            return self.RES

        import web_tools

        return patch.multiple(web_tools, search_web=search_web, search_papers=search_papers)

    def test_general_search_uses_search_web_with_a_clean_query(self):
        with self._web():
            r = orchestrator.answer_from_web(None, "m", "search the internet for recent news about Apple MLX")
        self.assertEqual(self.seen.get("web"), "recent news about Apple MLX")
        self.assertNotIn("papers", self.seen)
        self.assertEqual(r, "answer with sources")

    def test_the_model_gets_the_results_with_its_own_instructions_and_no_history(self):
        with self._web():
            orchestrator.answer_from_web(None, "m", "search the internet for MLX news")
        self.assertIn("https://ml-explore.github.io/mlx/", self.seen["message"])
        self.assertEqual(self.seen["system"], orchestrator.WEB_SYSTEM)
        self.assertIsNone(self.seen["ctx"])

    def test_if_the_model_refuses_the_results_are_shown(self):
        self.model_answer = "I'm sorry, but I cannot search the internet or access real-time information."
        with self._web():
            r = orchestrator.answer_from_web(None, "m", "search the internet for MLX news")
        self.assertIn("Here is what I found on the internet", r)
        self.assertIn("https://ml-explore.github.io/mlx/", r)
        self.assertNotIn("│", r)
        self.assertNotIn("[WEB CONTENT", r)

    def test_papers_use_arxiv(self):
        with self._web():
            orchestrator.answer_from_web(None, "m", "find recent papers about diffusion")
        self.assertIn("papers", self.seen)
        self.assertNotIn("web", self.seen)

    def test_no_results_makes_nothing_up_and_skips_the_model(self):
        with self._web(returns="No results for: xyz"):
            r = orchestrator.answer_from_web(None, "m", "search the internet for xyz")
        self.assertIn("could not find it on the internet", r)
        self.assertNotIn("message", self.seen)

    def test_the_chat_no_longer_leaves_the_search_to_the_model(self):
        import inspect

        self.assertIn("return answer_from_web(ex_cli, ex_mod, task)", inspect.getsource(orchestrator.resolve_task))

    def test_the_chat_prompt_asks_to_admit_what_it_does_not_know(self):
        self.assertIn("say so clearly instead of making it up", agent.CHAT_PROMPT)

    def test_cli_help_does_not_crash(self):
        # an unescaped '%' in an argparse help string made `--help` crash (pre-existing)
        import subprocess

        r = subprocess.run(  # noqa: S603 - fixed argument list, no shell
            [sys.executable, os.path.join(os.path.dirname(__file__), "..", "src", "orchestrator.py"), "--help"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(r.returncode, 0, r.stderr[-300:])


# ─────────────────────── user test: a step that must create a file ───────────────────────
class TestStepThatMustCreateAFile(unittest.TestCase):
    """Level 20 user test: the step 'Create the file test_calc.py...' ended without the file (the 14B answered
    with text) and the worker stopped after 15 minutes. Now a missing file the step asked for is asked again."""

    def setUp(self):
        self.d = Path(tempfile.mkdtemp()).resolve()
        self._wa, self._wo = agent.WORKDIR, orchestrator.WORKDIR
        agent.WORKDIR = orchestrator.WORKDIR = self.d

    def tearDown(self):
        agent.WORKDIR, orchestrator.WORKDIR = self._wa, self._wo

    def test_only_the_file_it_asks_to_create_counts(self):
        rx = orchestrator._RX_CREATE_FILE
        self.assertEqual(
            rx.findall("Create the file test_calc.py with tests for add, importing calc.py"), ["test_calc.py"]
        )
        self.assertEqual(rx.findall("create calc.py with an add function"), ["calc.py"])
        self.assertEqual(rx.findall("Read calc.py with read_file to see its interface"), [])

    def test_only_missing_files_inside_the_project(self):
        (self.d / "calc.py").write_text("x = 1\n", encoding="utf-8")
        self.assertEqual(orchestrator._files_not_created("create calc.py"), [])
        self.assertEqual(orchestrator._files_not_created("create test_calc.py"), ["test_calc.py"])

    def test_a_step_that_did_not_create_the_file_is_asked_again(self):
        calls = []

        def run_agent(cli, model, messages, confirm_bash=False, **kw):
            calls.append(messages[-1]["content"])
            if len(calls) == 2:  # only after the reminder does the "model" write the file
                (self.d / "test_calc.py").write_text("def test_x():\n    assert True\n", encoding="utf-8")
            return "Done."

        router = {"planner": (None, "m", "P"), "executor": (None, "m", "E"), "verifier": (None, "m", "V")}
        cfg = {
            "SELF_CRITIQUE": "false",
            "LEARN_FROM_ERRORS": "false",
            "METACOGNITION": "false",
            "MAX_VERIFY_ATTEMPTS": "1",
        }
        with (
            patch.object(orchestrator, "run_agent", run_agent),
            patch.object(orchestrator, "plan_task", lambda c, m, t: ["Create the file test_calc.py with a test"]),
            patch.object(orchestrator, "verify", lambda *a, **k: (True, "")),
        ):
            orchestrator.resolve_task(router, cfg, "create test_calc.py with a test", use_memory=False, direct=True)
        self.assertTrue((self.d / "test_calc.py").exists())
        self.assertIn("does NOT exist on disk", calls[1])


# ─────────────────────── user test: undo leaves a notice in the project memory ───────────────────────
class TestUndoNoticeInMemory(unittest.TestCase):
    def _undo(self, plan):
        d = Path(tempfile.mkdtemp()).resolve()
        (d / ".memory.md").write_text("## State\n- notes.txt created and verified.\n", encoding="utf-8")
        with (
            patch.object(orchestrator.projects, "project_path", lambda cfg, n: d),
            patch.object(orchestrator, "_chat_dir", lambda cfg, n, c=None: d),
            patch.object(orchestrator.time_machine, "last_snapshot", lambda store: "s1"),
            patch.object(orchestrator.time_machine, "undo", lambda base, sid, store=None: plan),
            patch.object(orchestrator.time_machine, "delete_snapshot", lambda store, sid: None),
        ):
            orchestrator.apply_undo({}, "p")
        return (d / ".memory.md").read_text(encoding="utf-8")

    def test_memory_notes_what_was_undone(self):
        mem = self._undo({"delete": ["notes.txt"], "revert": [], "recreate": [], "failed": []})
        self.assertIn("A task was UNDONE: notes.txt", mem)

    def test_no_changes_leave_the_memory_alone(self):
        self.assertNotIn("UNDONE", self._undo({"delete": [], "revert": [], "recreate": [], "failed": []}))


if __name__ == "__main__":
    unittest.main()
