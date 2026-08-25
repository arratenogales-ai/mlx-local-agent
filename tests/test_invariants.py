"""Self-contained invariant tests for the local coding agent.

Standard-library `unittest` only, no third-party test deps, and NOTHING is started here: no model,
no server, no network. They import the real functions from `src/` and check three load-bearing
invariants of the codebase:

  1. Path confinement  -- `_safe_path` rejects any path that escapes the project workspace.
  2. Destructive-command guard -- the shell guard flags `rm -rf` (and friends) and `run_bash`
     refuses to run them without explicit confirmation.
  3. Verdict parsing   -- `_parse_verdict` is a deterministic tri-state (OK / FAILURE / inconclusive)
     and never reads an unfinished check as a pass.

Run: python -m unittest discover tests
"""
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import agent          # noqa: E402
import orchestrator   # noqa: E402


class TestPathConfinement(unittest.TestCase):
    """`_safe_path` must keep every file path inside the project workspace (WORKDIR)."""

    def setUp(self):
        self._saved = agent.WORKDIR
        agent.WORKDIR = Path(tempfile.mkdtemp()).resolve()

    def tearDown(self):
        agent.WORKDIR = self._saved

    def test_in_project_path_is_allowed(self):
        result = agent._safe_path("notes/todo.txt")
        self.assertTrue(str(result).startswith(str(agent.WORKDIR)))

    def test_parent_traversal_is_blocked(self):
        with self.assertRaises(ValueError):
            agent._safe_path("../../etc/passwd")

    def test_absolute_path_outside_is_blocked(self):
        with self.assertRaises(ValueError):
            agent._safe_path("/etc/passwd")


class TestDestructiveGuard(unittest.TestCase):
    """Clearly destructive commands are flagged; benign ones are not; run_bash blocks the former."""

    def test_flags_destructive_commands(self):
        for cmd in ("rm -rf /", "rm -rf ~/project", "git push origin main", "shutdown now"):
            self.assertIsNotNone(agent._RX_DESTRUCTIVE.search(cmd), cmd)

    def test_ignores_benign_commands(self):
        for cmd in ("ls -la", "git status", "python -m pytest -q", "cat README.md"):
            self.assertIsNone(agent._RX_DESTRUCTIVE.search(cmd), cmd)

    def test_run_bash_blocks_rm_rf_without_confirm(self):
        # confirm=False + destructive => returns the block message and never spawns a shell.
        out = agent.run_bash("rm -rf /tmp/this_path_is_never_touched_xyz", confirm=False)
        self.assertIn("DESTRUCTIVE", out)
        self.assertIn("blocked", out.lower())


class TestVerdictParser(unittest.TestCase):
    """`_parse_verdict` is a deterministic tri-state and treats an unfinished check as NOT-OK."""

    def test_ok_is_true(self):
        state, _ = orchestrator._parse_verdict("VERDICT: OK")
        self.assertIs(state, True)

    def test_failure_is_false_with_detail(self):
        state, detail = orchestrator._parse_verdict("VERDICT: FAILURE: the test does not pass")
        self.assertIs(state, False)
        self.assertIn("does not pass", detail)

    def test_missing_verdict_is_inconclusive(self):
        state, _ = orchestrator._parse_verdict("I ran out of steps before finishing the check")
        self.assertIsNone(state)

    def test_is_deterministic(self):
        text = "some log line\nVERDICT: OK, no failures anymore"
        self.assertEqual(orchestrator._parse_verdict(text), orchestrator._parse_verdict(text))


if __name__ == "__main__":
    unittest.main()
