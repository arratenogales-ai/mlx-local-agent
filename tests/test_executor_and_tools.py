"""Level 20 regressions -- executor loop and tools (agent.py, web_tools.py, code_tools.py, notebooks.py,
deliverables.py).

Each test reproduces a REAL failure found by running code and pins its fix so it does not come back.
No model, no network, no server (standard-library `unittest` only).

  H5  the verifier loop is READ-ONLY: it does not see the writing tools and a writing call is refused
  Q7  tool calls as text: Qwen3.5+ XML format inside <tool_call>, JSON with literal newlines; a run_bash
      that writes counts as progress for the read-loop guard
  S1  bounded gzip/deflate decompression in the web downloader (no decompression bomb)
  S2  the code tools do not read a symlink that points outside the project
  S3  pandoc does not embed files from outside the project (images sanitized, dangerous raw TeX/HTML removed) and
      create_document uses the same single safe pandoc path as the deliverables
  S6  shell guard: more destructive forms blocked; run_bash survives non-UTF-8 output
  R1  central cap on tool results + read_file does not load huge files
  R5  atomic writes keep the file mode (an edited run.sh stays executable)
  Q8  notebooks: cells with ids, replace keeps metadata/attachments/id, honest duplicate message
  REV review of Level 20's own changes: harmless raw TeX/HTML is kept (only the dangerous part goes); the
      python-delete guard only applies inside `python -c`

Run: python -m unittest tests.test_executor_and_tools
"""

import json
import os
import shutil
import stat
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import agent  # noqa: E402


def _tmp():
    return Path(tempfile.mkdtemp()).resolve()


class _WorkdirMixin:
    """Points agent.WORKDIR at a fresh temp folder for the test and restores it afterwards."""

    def setUp(self):
        self._saved_wd = agent.WORKDIR
        self.d = _tmp()
        agent.WORKDIR = self.d

    def tearDown(self):
        agent.WORKDIR = self._saved_wd


# ─────────────────────────────── scripted fake model (streaming) ───────────────────────────────
class _ScriptedStream:
    def __init__(self, text):
        self.text = text

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __iter__(self):
        yield types.SimpleNamespace(
            usage=None,
            choices=[
                types.SimpleNamespace(
                    delta=types.SimpleNamespace(content=self.text, tool_calls=None), finish_reason="stop"
                )
            ],
        )

    def close(self):
        pass


class _ScriptedClient:
    """FAKE OpenAI client: each streaming call returns the next scripted text (the last one repeats)."""

    def __init__(self, script):
        self.script = list(script)
        self.n = 0
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self._create))

    def _create(self, **kw):
        text = self.script[min(self.n, len(self.script) - 1)]
        self.n += 1
        return _ScriptedStream(text)


def _fence(name, args):  # tool call as a ```json``` FENCE (the 14B style)
    return '```json\n{"name":"' + name + '","arguments":' + args + "}\n```"


# ─────────────────────────────── H5 · read-only verifier loop ───────────────────────────────
class TestH5ReadOnlyVerifier(_WorkdirMixin, unittest.TestCase):
    def test_verifier_tools_do_not_write(self):
        names = {t["function"]["name"] for t in agent.READ_ONLY_TOOLS}
        self.assertFalse(names & agent._TOOLS_MUTATE)
        self.assertIn("run_bash", names)
        self.assertIn("read_file", names)

    def test_real_loop_refuses_writes_in_read_only_mode(self):
        cli = _ScriptedClient([_fence("write_file", '{"path":"x.txt","content":"hello"}'), "VERDICT: OK"])
        seen = []
        orig = agent._create_stream

        def spy(client, model, messages, tools=None):
            seen.append({t["function"]["name"] for t in (tools or [])})
            return orig(client, model, messages, tools)

        with patch.object(agent, "_create_stream", spy):
            msgs = [{"role": "system", "content": "verify"}, {"role": "user", "content": "check it"}]
            agent.run_agent(cli, "m", msgs, read_only=True)
        self.assertFalse((self.d / "x.txt").exists())
        self.assertTrue(any(str(m.get("content")) == agent._READ_ONLY_MSG for m in msgs if m.get("role") == "tool"))
        self.assertIn("VERIFICATION mode", agent._READ_ONLY_MSG)
        self.assertTrue(seen and not (seen[0] & agent._TOOLS_MUTATE))

    def test_normal_mode_still_writes(self):
        cli = _ScriptedClient([_fence("write_file", '{"path":"x.txt","content":"hello"}'), "Done."])
        msgs = [{"role": "system", "content": "work"}, {"role": "user", "content": "create x.txt"}]
        agent.run_agent(cli, "m", msgs)
        self.assertEqual((self.d / "x.txt").read_text(encoding="utf-8"), "hello")


# ─────────────────────────────── Q7 · tool calls as text ───────────────────────────────
class TestQ7ToolCallsAsText(unittest.TestCase):
    """XML format of the Qwen3.5+ family (to be able to try new models without losing their calls) and
    JSON with LITERAL newlines (common when writing code), without opening the door to running prose."""

    def test_xml_with_schema_types(self):
        xml = (
            "<tool_call>\n<function=read_file>\n<parameter=path>\na.py\n</parameter>\n"
            "<parameter=start>\n5\n</parameter>\n</function>\n</tool_call>"
        )
        (c,) = agent._toolcalls_from_text(xml)
        self.assertEqual(c["name"], "read_file")
        self.assertEqual(json.loads(c["arguments"]), {"path": "a.py", "start": 5})

    def test_xml_text_content_is_not_converted(self):
        xml = (
            "<tool_call><function=write_file><parameter=path>n.txt</parameter>"
            "<parameter=content>123</parameter></function></tool_call>"
        )
        (c,) = agent._toolcalls_from_text(xml)
        self.assertEqual(json.loads(c["arguments"])["content"], "123")  # string, not int

    def test_xml_outside_tags_or_unknown_tool_is_not_run(self):
        self.assertEqual(agent._toolcalls_from_text("in prose: <function=write_file> ..."), [])
        self.assertEqual(
            agent._toolcalls_from_text(
                "<tool_call><function=delete_everything><parameter=x>1</parameter></function></tool_call>"
            ),
            [],
        )

    def test_json_with_literal_newlines(self):
        js = (
            '<tool_call>{"name":"write_file","arguments":{"path":"a.py",'
            '"content":"def f():\n    return 1"}}</tool_call>'
        )
        (c,) = agent._toolcalls_from_text(js)
        self.assertEqual(json.loads(c["arguments"])["content"], "def f():\n    return 1")


class TestQ7ReadLoop(unittest.TestCase):
    def test_editing_via_shell_counts_as_progress(self):
        def tc(c):
            return {"name": "run_bash", "arguments": json.dumps({"command": c})}

        counter = {}
        seq = [
            tc("pytest -q"),
            tc("sed -i 's/a/b/' calc.py"),
            tc("pytest -q"),
            tc("sed -i 's/c/d/' calc.py"),
            tc("pytest -q"),
            tc("pytest -q"),
        ]
        self.assertFalse(any(agent._is_read_loop(counter, [x]) for x in seq))
        counter = {}
        self.assertTrue(any(agent._is_read_loop(counter, [tc("pytest -q")]) for _ in range(agent.MAX_REP_READ)))

    def test_bash_mutates(self):
        def tc(c):
            return {"name": "run_bash", "arguments": json.dumps({"command": c})}

        for c in ("sed -i 's/a/b/' x.py", "echo hi > out.txt", "cat a | tee b", "mv a b", "git apply p.diff"):
            self.assertTrue(agent._bash_mutates(tc(c)), c)
        for c in ("pytest -q", "python -m pytest -q 2>&1", "grep -rn foo src/", "cat a.txt"):
            self.assertFalse(agent._bash_mutates(tc(c)), c)
        self.assertFalse(agent._bash_mutates({"name": "read_file", "arguments": '{"path":"a"}'}))


# ─────────────────────────────── safe path through a symlinked WORKDIR ───────────────────────────────
class TestSafePathWithSymlink(unittest.TestCase):
    def test_workdir_through_symlink_does_not_reject_legitimate_writes(self):
        real = _tmp()
        link = _tmp() / "link"
        link.symlink_to(real)
        wd = agent.WORKDIR
        agent.WORKDIR = link  # NOT resolved, on purpose
        try:
            self.assertTrue(agent.write_file("hello.txt", "hello").startswith("OK"))
            with self.assertRaises(ValueError):
                agent._safe_path("../outside.txt")  # confinement still intact
        finally:
            agent.WORKDIR = wd
        self.assertTrue((real / "hello.txt").is_file())


# ─────────────────────────────── S6 · shell guard ───────────────────────────────
class TestS6ShellGuard(unittest.TestCase):
    BLOCK = [
        "git restore .",
        "git checkout HEAD -- src/",
        "git reset HEAD~1 --hard",
        "git stash drop",
        r"find . -name '*.pyc' -exec rm {} \;",
        "ls *.log | xargs rm",
        "python -c \"import shutil; shutil.rmtree('build')\"",
        "echo hello\nsudo rm x",
        "echo $(sudo whoami)",
        "rm -rf build",
        "git push origin main",
    ]
    ALLOW = [
        "git status",
        "git diff -- src/",
        "git checkout -b new-branch",
        "pytest -q",
        "python test_shutdown.py",
        "ls > /dev/null",
        "find . -name '*.py'",
        "grep -rn remove src/",
        "git stash list",
        "cat sudoers.txt",
    ]

    def test_blocks_destructive_and_allows_normal(self):
        for c in self.BLOCK:
            self.assertTrue(agent._RX_DESTRUCTIVE.search(c), c)
        for c in self.ALLOW:
            self.assertFalse(agent._RX_DESTRUCTIVE.search(c), c)

    def test_non_utf8_output_does_not_lose_the_result(self):
        wd = agent.WORKDIR
        agent.WORKDIR = _tmp()
        try:
            r = agent.run_bash("echo x >> c.txt; printf 'ok \\xff end'")
        finally:
            agent.WORKDIR = wd
        self.assertIn("[exit 0]", r)
        self.assertIn("end", r)

    def test_command_waiting_for_input_does_not_hang(self):
        wd = agent.WORKDIR
        agent.WORKDIR = _tmp()
        try:
            r = agent.run_bash("read x; echo got:$x")  # stdin is /dev/null: read gets EOF at once
        finally:
            agent.WORKDIR = wd
        self.assertIn("got:", r)


# ─────────────────────────────── S1 · gzip bomb ───────────────────────────────
class TestS1GzipBomb(unittest.TestCase):
    def test_bounded_decompression(self):
        import gzip

        import web_tools as w

        bomb = gzip.compress(b"A" * 50_000_000)
        d = w._bounded_decompressor("gzip")
        out = bytearray()
        for i in range(0, len(bomb), 65536):
            out += d(bomb[i : i + 65536], w.WEB_MAX_BYTES - len(out))
            if len(out) >= w.WEB_MAX_BYTES:
                break
        self.assertLessEqual(len(out), w.WEB_MAX_BYTES)

    def test_unboundable_encodings_are_rejected(self):
        import web_tools as w

        for ce in ("br", "zstd", "gzip, gzip"):
            with self.assertRaises(ValueError):
                w._bounded_decompressor(ce)
        self.assertIsNone(w._bounded_decompressor("identity"))
        self.assertIsNone(w._bounded_decompressor(""))

    def test_deflate_round_trip(self):
        import zlib

        import web_tools as w

        d = w._bounded_decompressor("deflate")
        self.assertEqual(d(zlib.compress(b"hello world"), 1000), b"hello world")


# ─────────────────────────────── S2 · symlink outside in code tools ───────────────────────────────
class TestS2SymlinkOutsideInCodeTools(unittest.TestCase):
    def test_find_symbol_does_not_read_a_py_from_outside(self):
        import code_tools

        outside = _tmp() / "secret.py"
        outside.write_text("API_KEY = 'sk-OUTSIDE'\n", encoding="utf-8")
        d = _tmp()
        (d / "cfg.py").symlink_to(outside)
        (d / "ok.py").write_text("x = 1\n", encoding="utf-8")
        self.assertEqual([f.name for f in code_tools._py_files(d)], ["ok.py"])


# ─────────────────────────────── S3 · pandoc does not embed outside files ───────────────────────────────
class TestS3PandocDoesNotEmbedOutsideFiles(unittest.TestCase):
    def test_outside_images_are_omitted_and_inside_ones_stay(self):
        import deliverables

        root = _tmp()
        (root / "g.png").write_bytes(b"x")
        outside = _tmp() / "s.png"
        outside.write_bytes(b"x")
        md = (
            f"![a](g.png) ![b]({outside}) ![c](http://evil.example/x.png) ![d](~/.ssh/id_rsa)\n\n"
            f"![e][r]\n\n[r]: {outside}\n"
        )
        out = deliverables.sanitize_images(md, root)
        self.assertIn("![a](g.png)", out)
        self.assertNotIn(str(outside), out)
        self.assertNotIn("evil.example", out)
        self.assertNotIn("id_rsa)", out)
        self.assertEqual(out.count("[image omitted"), 3)

    def test_create_document_uses_the_single_safe_pandoc_path(self):
        import deliverables

        wd = agent.WORKDIR
        agent.WORKDIR = _tmp()
        calls = []

        def fake(md, target, extra=None, root=None):
            calls.append((md, target, root))
            return None

        try:
            with patch.object(deliverables, "_pandoc_from_md", fake):
                r = agent.create_document("report.docx", "# Report\n")
        finally:
            target_root, agent.WORKDIR = agent.WORKDIR, wd
        self.assertTrue(r.startswith("OK"), r)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][2], target_root)

    @unittest.skipUnless(shutil.which("pandoc"), "pandoc not installed")
    def test_real_docx_does_not_carry_the_outside_file(self):
        import zipfile

        outside = _tmp() / "secret.txt"
        outside.write_text("PASSWORD-OUTSIDE", encoding="utf-8")
        wd = agent.WORKDIR
        agent.WORKDIR = _tmp()
        try:
            r = agent.create_document("i.docx", f"# I\n\n![x]({outside})\n\n\\input{{{outside}}}\n")
            self.assertTrue(r.startswith("OK"), r)
            z = zipfile.ZipFile(agent.WORKDIR / "i.docx")
            everything = b"".join(z.read(n) for n in z.namelist())
            self.assertNotIn(b"PASSWORD-OUTSIDE", everything)
        finally:
            agent.WORKDIR = wd


# ─────────────────────────────── R1 · central cap ───────────────────────────────
class TestR1CentralCap(unittest.TestCase):
    def test_no_tool_result_goes_past_the_cap(self):
        with patch.dict(agent.DISPATCH, {"read_file": lambda **k: "x" * 200_000}):
            r = agent.run_tool("read_file", {"path": "a"}, False)
        self.assertLessEqual(len(r), agent.TOOL_RESULT_CAP + 300)
        self.assertIn("truncated", r)

    def test_short_results_are_untouched(self):
        self.assertEqual(agent._cap_result("x", "short"), "short")

    def test_read_file_does_not_load_huge_files(self):
        wd = agent.WORKDIR
        agent.WORKDIR = _tmp()
        try:
            with open(agent.WORKDIR / "weights.bin", "wb") as f:
                f.truncate(agent.MAX_READ_BYTES + 1)
            self.assertIn("too large", agent.read_file("weights.bin"))
        finally:
            agent.WORKDIR = wd


# ─────────────────────────────── R5 · permissions in atomic writes ───────────────────────────────
class TestR5PermissionsInAtomicWrites(unittest.TestCase):
    def test_edit_file_keeps_the_execute_bit(self):
        wd = agent.WORKDIR
        agent.WORKDIR = _tmp()
        try:
            (agent.WORKDIR / "run.sh").write_text("#!/bin/sh\necho one\n", encoding="utf-8")
            os.chmod(agent.WORKDIR / "run.sh", 0o755)
            agent.edit_file("run.sh", "echo one", "echo two")
            self.assertEqual(stat.S_IMODE((agent.WORKDIR / "run.sh").stat().st_mode), 0o755)
        finally:
            agent.WORKDIR = wd

    def test_notebook_conversion_keeps_the_mode(self):
        import notebooks

        d = _tmp()
        nb = d / "n.ipynb"
        notebooks.create(nb, [{"type": "code", "source": "print(1)"}])
        out = d / "n.py"
        out.write_text("# old\n", encoding="utf-8")
        os.chmod(out, 0o755)
        self.assertTrue(notebooks.convert(nb, out).startswith("OK"))
        self.assertEqual(stat.S_IMODE(out.stat().st_mode), 0o755)


# ─────────────────────────────── Q8 · notebooks ───────────────────────────────
class TestQ8Notebooks(unittest.TestCase):
    def test_new_cells_have_id_and_replace_keeps_metadata(self):
        import notebooks

        nb = _tmp() / "n.ipynb"
        notebooks.create(nb, [{"type": "text", "source": "![img](attachment:a.png)"}])
        data = json.loads(nb.read_text(encoding="utf-8"))
        self.assertIn("id", data["cells"][0])
        data["cells"][0]["attachments"] = {"a.png": {"image/png": "iVBORw0KGgo="}}
        data["cells"][0]["metadata"] = {"tags": ["parameters"]}
        nb.write_text(json.dumps(data), encoding="utf-8")
        notebooks.edit(nb, 0, "replace", source="![img](attachment:a.png) edited")
        c = json.loads(nb.read_text(encoding="utf-8"))["cells"][0]
        self.assertEqual(c["metadata"], {"tags": ["parameters"]})
        self.assertIn("a.png", c["attachments"])
        self.assertEqual(c["id"], data["cells"][0]["id"])

    def test_insert_duplicate_says_clearly_it_did_not_insert(self):
        import notebooks

        nb = _tmp() / "n.ipynb"
        notebooks.create(nb, [{"type": "code", "source": "df.head()"}])
        r = notebooks.edit(nb, 1, "insert", source="df.head()")
        self.assertIn("NOT inserting it again", r)
        self.assertIn("cell 0", r)


# ═══════════════════ REVIEW of Level 20's own changes (independent review) ═══════════════════
class TestLevel20Review(unittest.TestCase):
    """Regressions that Level 20 itself introduced and an independent review found; each test reproduces one."""

    # (7) pandoc keeps the harmless raw TeX/HTML and strips the dangerous part
    def test_7_sanitizing_removes_only_the_dangerous_part(self):
        import deliverables

        root = _tmp()
        (root / "ok.png").write_bytes(b"x")
        md = (
            "| a<br>b | H<sub>2</sub>O |\n\\newpage\n<!-- note -->\n\\input{/etc/hosts}\n"
            "\\includegraphics[width=3cm]{/etc/x.png}\n\\immediate\\write18{id}\n"
            "<script>alert(1)</script>\n<iframe src='http://x'></iframe>\n<link rel='stylesheet' href='http://x'>\n"
            '<img src=\'/etc/x.png\'>\n<img src="ok.png" alt="ok">\n'
        )
        out = deliverables.sanitize_images(md, root)
        for stays in ("<br>", "<sub>2</sub>", "\\newpage", "<!-- note -->", '<img src="ok.png" alt="ok">'):
            self.assertIn(stays, out)
        for gone in (
            "\\input",
            "\\includegraphics",
            "\\write18",
            "<script",
            "alert(1)",
            "<iframe",
            "<link",
            "/etc/x.png",
        ):
            self.assertNotIn(gone, out)
        self.assertIn("[image omitted]", out)
        self.assertEqual(deliverables._INPUT_FORMAT, "markdown")

    # (10c) the guard does not block searching for "os.remove(" in the code
    def test_10c_python_guard_only_inside_python_c(self):
        for c in ('grep -rn "os.remove(" .', "grep -rn 'shutil.rmtree(' src/", "cat cleanup.py | grep os.unlink("):
            self.assertFalse(agent._RX_DESTRUCTIVE.search(c), c)
        for c in (
            "python3 -c 'import os; os.remove(1)'",
            "python -c \"import shutil; shutil.rmtree('build')\"",
            "python3 -u -c 'import os; os.unlink(\"x\")'",
        ):
            self.assertTrue(agent._RX_DESTRUCTIVE.search(c), c)


if __name__ == "__main__":
    unittest.main()
