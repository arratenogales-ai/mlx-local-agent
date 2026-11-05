"""Level 20 regressions: web server, project/chat storage and the two web pages.

Each test reproduces a REAL bug found in the Level 20 audit and pins its fix. No model, no network, no
running server (FastAPI's in-process TestClient only).

  D1  the "recoverable" trash NEVER deletes what was just deleted (it pruned by folder mtime)
  D2  a project name resolves to its EXACT folder (deleting "My Project" no longer trashes "My-Project")
  D4  undo applies exactly the snapshot shown in the preview, or refuses with 409 if another task ran
  S4/S5  request guard: DNS rebinding (foreign Host), cross-site requests, Origin with pinned port,
         uploads cut by Content-Length before the body is read
  R3  only ONE worker run per project; /worker-status says whether a run is really alive
  REV review of Level 20's own changes: a link from another site may OPEN the pages (never the API), extra
      Origin ports via WEB_ALLOWED_PORTS, uploads are protected from the verifier's cleanup
  UI  backend <-> page contract: every `case "x"` has an emitter and every `ev.X` it reads is a key the
      backend sends (this caught the public-only bug: verdicts read `state` instead of `status`, so every
      verdict showed as "Inconclusive", and `notice`/`warning` events were never rendered)

Run: python -m unittest tests.test_web_and_storage
"""

import ast
import os
import re
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import chats  # noqa: E402
import projects  # noqa: E402
import time_machine  # noqa: E402


def _tmp():
    return Path(tempfile.mkdtemp()).resolve()


def _age(p, days=30):
    t = time.time() - days * 86400
    os.utime(p, (t, t))


def _import_server():
    """(web_server module, TestClient class) or SkipTest if the server can't be imported here."""
    try:
        from fastapi.testclient import TestClient

        import web_server
    except Exception as e:  # noqa: BLE001
        raise unittest.SkipTest(f"could not import web_server ({type(e).__name__}: {e})") from e
    return web_server, TestClient


# ─────────────────────────────── D1 · recoverable trash ───────────────────────────────
class TestD1RecoverableTrash(unittest.TestCase):
    """The trash must NEVER delete what was just moved into it (it used to prune by the folder's mtime,
    which is not the deletion date: an old, untouched project was destroyed the moment it was deleted)."""

    def setUp(self):
        self.cfg = {"AGENT_PROJECTS_DIR": str(_tmp())}

    def test_deleting_an_old_project_with_a_full_trash_does_not_destroy_it(self):
        for i in range(projects.MAX_TRASH):
            projects.create_project(self.cfg, f"filler{i}")
            self.assertTrue(projects.delete_project(self.cfg, f"filler{i}")["ok"])
        important = projects.create_project(self.cfg, "important")
        (important / "thesis.md").write_text("years of work", encoding="utf-8")
        _age(important)  # untouched for 30 days
        self.assertTrue(projects.delete_project(self.cfg, "important")["ok"])
        trash = projects.base_dir(self.cfg) / projects.TRASH_NAME
        self.assertTrue(any((d / "thesis.md").is_file() for d in trash.iterdir()), "the project was lost")
        self.assertLessEqual(len(list(trash.iterdir())), projects.MAX_TRASH)

    def test_pruning_removes_the_oldest_by_deletion_date(self):
        trash = _tmp()
        old = []
        for i in range(3):
            d = projects.unique_destination(trash, f"p{i}")
            d.mkdir()
            old.append(d)
            time.sleep(0.003)
        _age(old[-1], 365)  # an old mtime does NOT matter: the name's stamp does
        new = projects.unique_destination(trash, "new")
        new.mkdir()
        projects.prune_by_age(trash, 2, keep=new)
        self.assertEqual(sorted(d.name for d in trash.iterdir()), sorted([old[-1].name, new.name]))

    def test_unique_destination_never_reuses_a_path(self):
        folder = _tmp()
        a = projects.unique_destination(folder, "x")
        a.mkdir()
        b = projects.unique_destination(folder, "x")
        self.assertNotEqual(a, b)
        self.assertFalse(b.exists())

    def test_old_chat_deleted_with_a_full_trash_stays_in_the_trash(self):
        proj = projects.create_project(self.cfg, "p")
        for i in range(chats.MAX_CHATS_TRASH):
            cid = chats.create_chat(proj, f"c{i}")
            chats.delete_chat(proj, cid)
        cid = chats.create_chat(proj, "valuable")
        _age(chats.chat_path(proj, cid))
        self.assertTrue(chats.delete_chat(proj, cid)["ok"])
        self.assertTrue(any(d.name.startswith("valuable-") for d in (proj / chats.CHATS_TRASH).iterdir()))

    def test_a_migration_backup_just_made_is_not_pruned(self):
        base = _tmp()
        for i in range(chats.MAX_BACKUPS):
            pr = base / f"x{i}"
            pr.mkdir()
            chats._backup(pr, base)
        old = base / "old"
        old.mkdir()
        (old / "a.txt").write_text("a", encoding="utf-8")
        _age(old)
        dest = chats._backup(old, base)
        self.assertTrue((dest / "a.txt").is_file())


# ─────────────────────────────── D2 · exact project folder ───────────────────────────────
class TestD2ExactProject(unittest.TestCase):
    def setUp(self):
        self.cfg = {"AGENT_PROJECTS_DIR": str(_tmp())}
        self.base = projects.base_dir(self.cfg)
        (self.base / "My Project").mkdir()  # made by hand
        (self.base / "My-Project").mkdir()  # made by the app
        (self.base / "My-Project" / "do_not_touch.txt").write_text("x", encoding="utf-8")

    def test_deleting_the_hand_made_one_does_not_delete_the_other(self):
        r = projects.delete_project(self.cfg, "My Project")
        self.assertTrue(r["ok"])
        self.assertEqual(r["name"], "My Project")
        self.assertFalse((self.base / "My Project").exists())
        self.assertTrue((self.base / "My-Project" / "do_not_touch.txt").is_file())

    def test_active_points_to_the_real_folder(self):
        projects.set_active(self.cfg, "My Project")
        self.assertEqual(projects.active_project(self.cfg), "My Project")

    def test_listed_names_all_resolve_to_their_own_folder(self):
        for name in projects.list_projects(self.cfg):
            self.assertEqual(projects.project_path(self.cfg, name), self.base / name)
            self.assertTrue(projects.exists(self.cfg, name))

    def test_path_stays_confined(self):
        for bad in ("../outside", "/etc", ".hidden", "a/../../x"):
            self.assertEqual(projects.project_path(self.cfg, bad).parent, self.base, bad)

    def test_symlink_pointing_outside_is_not_followed(self):
        outside = _tmp()
        (outside / "secret.txt").write_text("s", encoding="utf-8")
        (self.base / "Escape Hatch").symlink_to(outside, target_is_directory=True)
        # the exact-name lookup ignores it (falls back to the slug path, which does not exist)...
        self.assertEqual(projects.project_path(self.cfg, "Escape Hatch"), self.base / "Escape-Hatch")
        # ...and deleting never follows a symlink out of the base
        (self.base / "link").symlink_to(outside, target_is_directory=True)
        self.assertFalse(projects.delete_project(self.cfg, "link")["ok"])
        self.assertTrue((outside / "secret.txt").is_file())


# ─────────────────────────────── D4 · undo the snapshot that was shown ───────────────────────────────
class TestD4UndoTheSnapshotShown(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ws, TestClient = _import_server()
        cls.cli = TestClient(cls.ws.app, base_url="http://127.0.0.1:8080")
        cls.OK = {"origin": "http://127.0.0.1:8080"}

    def setUp(self):
        self._orig = self.ws.cfg.get("AGENT_PROJECTS_DIR")
        self.ws.cfg["AGENT_PROJECTS_DIR"] = str(_tmp())
        self.proj = projects.create_project(self.ws.cfg, "p")

    def tearDown(self):
        self.ws.cfg["AGENT_PROJECTS_DIR"] = self._orig

    def _snapshot(self, name, content):
        sid = time_machine.create_snapshot(self.proj, "t")
        (self.proj / name).write_text(content, encoding="utf-8")
        time_machine.record_changes(self.proj, sid)
        return sid

    def test_if_another_task_ran_after_the_preview_it_is_refused_and_nothing_is_touched(self):
        s1 = self._snapshot("t1.txt", "one")
        prev = self.cli.get("/undo-preview", params={"project": "p"}).json()
        self.assertEqual(prev["id"], s1)
        self._snapshot("worker.txt", "the worker's work")  # another task in between
        r = self.cli.post("/undo", json={"project": "p", "id": s1}, headers=self.OK)
        self.assertEqual(r.status_code, 409)
        self.assertTrue((self.proj / "worker.txt").exists())
        self.assertTrue((self.proj / "t1.txt").exists())

    def test_with_the_snapshot_shown_it_undoes_that_one(self):
        s1 = self._snapshot("t1.txt", "one")
        r = self.cli.post("/undo", json={"project": "p", "id": s1}, headers=self.OK)
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.json()["undone"])
        self.assertFalse((self.proj / "t1.txt").exists())

    def test_a_malformed_snapshot_id_is_rejected(self):
        r = self.cli.post("/undo", json={"project": "p", "id": "../../x"}, headers=self.OK)
        self.assertEqual(r.status_code, 400)


# ─────────────────────────────── S4/S5 · request guard ───────────────────────────────
class TestS4S5RequestGuard(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ws, TestClient = _import_server()
        cls.cli = TestClient(cls.ws.app, base_url="http://127.0.0.1:8080")

    def test_dns_rebinding_foreign_host_is_rejected(self):
        r = self.cli.get("/projects", headers={"host": "evil.example:8080"})
        self.assertEqual(r.status_code, 400)
        self.assertEqual(self.cli.get("/projects").status_code, 200)

    def test_request_from_another_site_sec_fetch_site_is_rejected(self):
        for v in ("cross-site", "same-site"):
            self.assertEqual(self.cli.get("/projects", headers={"sec-fetch-site": v}).status_code, 403, v)
        self.assertEqual(self.cli.get("/projects", headers={"sec-fetch-site": "same-origin"}).status_code, 200)

    def test_origin_from_another_local_port_is_rejected(self):
        self.assertFalse(self.ws._origin_ok("http://localhost:8888"))
        self.assertFalse(self.ws._origin_ok("http://evil.com:8080"))
        self.assertFalse(self.ws._origin_ok("null"))
        self.assertTrue(self.ws._origin_ok("http://127.0.0.1:8080"))
        self.assertTrue(self.ws._origin_ok(None))

    def test_host_header_parsing(self):
        self.assertEqual(self.ws._host_from_header("127.0.0.1:8080"), "127.0.0.1")
        self.assertEqual(self.ws._host_from_header("[::1]:8080"), "::1")
        self.assertEqual(self.ws._host_from_header("LOCALHOST"), "localhost")
        self.assertIn("::1", self.ws._allowed_hosts())

    def test_upload_without_size_or_huge_is_cut_without_reading_the_body(self):
        base = {"type": "http", "method": "POST", "path": "/upload"}
        h = {"host": "127.0.0.1:8080"}
        self.assertEqual(self.ws._reason_to_reject(base, h)[1], 411)
        huge = str(self.ws._upload_limit_bytes() + 50 * 1024 * 1024)
        self.assertEqual(self.ws._reason_to_reject(base, {**h, "content-length": huge})[1], 413)
        self.assertEqual(self.ws._reason_to_reject(base, {**h, "content-length": "1000"}), (None, None))
        bad = {**h, "content-length": "1000", "origin": "https://evil.example"}
        self.assertEqual(self.ws._reason_to_reject(base, bad)[1], 403)

    def test_websocket_from_another_local_port_is_closed(self):
        from starlette.websockets import WebSocketDisconnect

        with self.assertRaises(WebSocketDisconnect):
            with self.cli.websocket_connect(
                "/ws", headers={"host": "127.0.0.1:8080", "origin": "http://localhost:8888"}
            ) as w:
                w.receive_json()


# ─────────────────────────────── R3 · one run per project ───────────────────────────────
class TestR3OneRunPerProject(unittest.TestCase):
    def test_only_one_run_per_project(self):
        ws, TestClient = _import_server()
        orig = ws.cfg.get("AGENT_PROJECTS_DIR")
        ws.cfg["AGENT_PROJECTS_DIR"] = str(_tmp())
        projects.create_project(ws.cfg, "p")
        cli = TestClient(ws.app, base_url="http://127.0.0.1:8080")
        try:
            self.assertFalse(cli.get("/worker-status", params={"project": "p"}).json()["alive"])
            ws._active_runs.add("p")
            with cli.websocket_connect("/ws-agentic", headers={"host": "127.0.0.1:8080"}) as w:
                w.send_json({"task": "do something", "project": "p"})
                ev = w.receive_json()
            self.assertEqual(ev["type"], "worker_final")
            self.assertEqual(ev["state"], "error")
            self.assertIn("already running", ev["summary"])
            self.assertTrue(cli.get("/worker-status", params={"project": "p"}).json()["alive"])
        finally:
            ws._active_runs.discard("p")
            ws.cfg["AGENT_PROJECTS_DIR"] = orig


# ═══════════════════ REVIEW of Level 20's own changes (independent review) ═══════════════════
class TestLevel20Review(unittest.TestCase):
    """Regressions that Level 20 itself introduced and an independent review found; each test reproduces one."""

    @classmethod
    def setUpClass(cls):
        cls.ws, TestClient = _import_server()
        cls.cli = TestClient(cls.ws.app, base_url="http://127.0.0.1:8080")

    # (10b) opening the web from a link on another page is a normal navigation; the API stays protected
    def test_10b_navigation_to_the_pages_yes_api_no(self):
        nav = {
            "host": "127.0.0.1:8080",
            "sec-fetch-site": "cross-site",
            "sec-fetch-mode": "navigate",
            "sec-fetch-dest": "document",
        }
        for path in ("/", "/agentic"):
            self.assertEqual(
                self.ws._reason_to_reject({"type": "http", "method": "GET", "path": path}, nav), (None, None), path
            )
        self.assertEqual(self.ws._reason_to_reject({"type": "http", "method": "GET", "path": "/projects"}, nav)[1], 403)
        self.assertEqual(self.ws._reason_to_reject({"type": "http", "method": "POST", "path": "/"}, nav)[1], 403)
        embedded = {**nav, "sec-fetch-mode": "no-cors", "sec-fetch-dest": "image"}  # <img src=...> on another site
        self.assertEqual(self.ws._reason_to_reject({"type": "http", "method": "GET", "path": "/"}, embedded)[1], 403)
        self.assertEqual(self.cli.get("/", headers=nav).status_code, 200)
        self.assertEqual(self.cli.get("/projects", headers=nav).status_code, 403)

    def test_10b_extra_origin_ports_with_WEB_ALLOWED_PORTS(self):
        with patch.dict(self.ws.cfg, {"WEB_ALLOWED_PORTS": "9090, junk"}):
            self.assertTrue(self.ws._origin_ok("http://127.0.0.1:9090"))
            self.assertTrue(self.ws._origin_ok("http://127.0.0.1:8080"))
            self.assertFalse(self.ws._origin_ok("http://127.0.0.1:9091"))
        self.assertFalse(self.ws._origin_ok("http://127.0.0.1:9090"))

    # (3) a file the user uploads is protected from the cleanup of a verifier running at that moment
    def test_3_an_upload_is_protected_from_the_verifier_cleanup(self):
        import orchestrator

        orig = self.ws.cfg.get("AGENT_PROJECTS_DIR")
        self.ws.cfg["AGENT_PROJECTS_DIR"] = str(_tmp())
        proj = projects.create_project(self.ws.cfg, "p")
        dest = str((proj / "photo.png").resolve())
        try:
            r = self.cli.post(
                "/upload",
                files={"file": ("photo.png", b"x", "image/png")},
                data={"project": "p"},
                headers={"origin": "http://127.0.0.1:8080"},
            )
            self.assertEqual(r.status_code, 200, r.text)
            self.assertIn(dest, orchestrator.PROTECTED_FILES)
        finally:
            orchestrator.PROTECTED_FILES.discard(dest)
            self.ws.cfg["AGENT_PROJECTS_DIR"] = orig


# ─────────────────────────────── backend <-> UI contract ───────────────────────────────
_PAGES = ("index.html", "agentic.html")


def _emitted_events():
    """{type: {keys}} of EVERYTHING the backend emits: emit("type", k=...) and any dict literal with
    "type": "..." (the worker_* events and the ones the server sends with send_json)."""
    emitted = {}
    for f in (ROOT / "src").glob("*.py"):
        for n in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
            if isinstance(n, ast.Call) and n.args:
                fn = n.func.id if isinstance(n.func, ast.Name) else getattr(n.func, "attr", "")
                a0 = n.args[0]
                if fn == "emit" and isinstance(a0, ast.Constant) and isinstance(a0.value, str):
                    emitted.setdefault(a0.value, set()).update(k.arg for k in n.keywords if k.arg)
            if isinstance(n, ast.Dict):
                pairs = {k.value: v for k, v in zip(n.keys, n.values, strict=False) if isinstance(k, ast.Constant)}
                typ = pairs.get("type")
                if isinstance(typ, ast.Constant) and isinstance(typ.value, str):
                    emitted.setdefault(typ.value, set()).update(k for k in pairs if isinstance(k, str))
    return emitted


def _cases(html):
    """[(type, body)] for each `case "type":` of the page's event switch."""
    rx = r'case\s+"(\w+)"\s*:(.*?)(?=case\s+"\w+"\s*:|default\s*:|\n\s{0,2}\}\s*\n)'
    return [(m.group(1), m.group(2)) for m in re.finditer(rx, html, re.S)]


class TestUIEventContract(unittest.TestCase):
    """Level 20: every `case "type"` in the pages must have an emitter in the backend, and every `ev.X` it
    reads must be a key the backend SENDS. It catches the public translation bug: the page read
    `state`/`hint` while the backend sent `status`/`notice`, so every verdict showed as "Inconclusive"."""

    def test_the_pages_only_read_keys_the_backend_sends(self):
        emitted = _emitted_events()
        problems = []
        for page in _PAGES:
            html = (ROOT / "web" / page).read_text(encoding="utf-8")
            for typ, body in _cases(html):
                if typ not in emitted:
                    problems.append(f"{page}: case '{typ}' has no emitter in the backend")
                    continue
                missing = set(re.findall(r"\bev\.(\w+)", body)) - emitted[typ] - {"type"}
                if missing:
                    problems.append(
                        f"{page}: '{typ}' reads {sorted(missing)}; the backend sends {sorted(emitted[typ])}"
                    )
        self.assertEqual(problems, [])

    def test_honest_stop_and_warning_events_are_rendered(self):
        html = (ROOT / "web" / "index.html").read_text(encoding="utf-8")
        handled = {t for t, _ in _cases(html)}
        emitted = _emitted_events()
        for typ in ("verdict", "notice", "warning", "difficulty"):
            if typ in emitted:
                self.assertIn(typ, handled, f"index.html never renders the '{typ}' event")

    def test_search_results_render_the_keys_the_backend_returns(self):
        import conversations

        proj = _tmp()
        projects.save_turn(proj, "hello there", "general kenobi")
        results = conversations.search(proj, "hello")
        self.assertTrue(results)
        html = (ROOT / "web" / "index.html").read_text(encoding="utf-8")
        body = html[html.index("async function searchConversation") : html.index("async function exportConversation")]
        read = set(re.findall(r"\bt\.(\w+)", body))
        self.assertTrue(read)
        self.assertEqual(read - set(results[0]), set(), "the search card reads keys the backend doesn't return")


if __name__ == "__main__":
    unittest.main()
