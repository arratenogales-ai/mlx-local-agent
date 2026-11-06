"""Publishing rules of this repository. They run with the rest of the suite, so a push that breaks one turns CI red,
and running the suite before committing catches it before anything leaves the Mac.

What gets published is what git would commit: the tracked files plus the new ones that .gitignore does not
exclude. None of them may carry personal e-mail addresses, home-folder paths, secrets or local config; images carry
no metadata; the web pages never contain an em dash; and every commit uses the account's anonymous GitHub address.
"""

import re
import shutil
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_GIT = shutil.which("git")

# Built from two halves so this file does not trip its own check: the name of the coding assistant used to
# build the project never appears in what is published.
_ASSISTANT = re.compile("cl" + "aude", re.IGNORECASE)
_NOREPLY = "@users.noreply.github.com"
_EMAIL = re.compile(r"(?<![\\\w.%+-])[\w.%+-]+@([A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,})\b")
_SAFE_EMAIL_DOMAINS = ("users.noreply.github.com", "example.com", "example.org")
_GITHUB_BOT = "noreply@github.com"  # GitHub's own address (its merge commits); it identifies nobody
_HOME_PATH = re.compile(r"/(?:Users|home)/[\w.-]+")
_SECRET = re.compile(
    r"\b(?:sk-[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|hf_[A-Za-z0-9]{30,}"
    r"|AKIA[0-9A-Z]{16}|xox[abprs]-[A-Za-z0-9-]{10,})|-----BEGIN [A-Z ]*PRIVATE KEY-----"
)
_NEVER_NAMES = {"config.env", ".env"}
_NEVER_SUFFIXES = {".pdf", ".key", ".pem", ".p12", ".sqlite", ".db"}
_MAX_BYTES = 2_000_000
_PNG_METADATA = {b"eXIf", b"tEXt", b"zTXt", b"iTXt", b"tIME"}


def _git(*args):
    return subprocess.run(  # noqa: S603 - fixed git arguments, no shell
        [_GIT, "-C", str(ROOT), *args], capture_output=True, text=True, check=True
    ).stdout


def _published_files():
    """What `git add -A && git commit` would publish: tracked + new files not ignored."""
    names = _git("ls-files", "--cached", "--others", "--exclude-standard", "-z").split("\0")
    return [ROOT / n for n in sorted(set(names)) if n and (ROOT / n).is_file()]


def _png_chunks(data):
    pos, kinds = 8, []
    while pos + 8 <= len(data):
        kinds.append(data[pos + 4 : pos + 8])
        pos += 12 + int.from_bytes(data[pos : pos + 4], "big")
    return kinds


@unittest.skipUnless(_GIT and (ROOT / ".git").exists(), "not a git checkout (e.g. a downloaded zip)")
class TestPublishHygiene(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.files = _published_files()
        cls.texts = {}
        for f in cls.files:
            try:
                cls.texts[f] = f.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                pass  # binary (images): checked on their own

    def _where(self, rx, allow=lambda m: False):
        hits = []
        for f, text in self.texts.items():
            for n, line in enumerate(text.splitlines(), 1):
                hits += [f"{f.relative_to(ROOT)}:{n}: {m.group(0)}" for m in rx.finditer(line) if not allow(m)]
        return hits

    def test_there_is_something_to_check(self):
        self.assertGreater(len(self.files), 20)
        self.assertIn(ROOT / "README.md", self.files)

    def test_no_local_config_or_private_documents(self):
        bad = [f.name for f in self.files if f.name in _NEVER_NAMES or f.suffix.lower() in _NEVER_SUFFIXES]
        self.assertEqual(bad, [], "these files must never be published")

    def test_no_big_files(self):
        big = [f"{f.relative_to(ROOT)} ({f.stat().st_size} bytes)" for f in self.files if f.stat().st_size > _MAX_BYTES]
        self.assertEqual(big, [])

    def test_no_personal_email_addresses(self):
        hits = self._where(
            _EMAIL, allow=lambda m: m.group(1).lower().endswith(_SAFE_EMAIL_DOMAINS) or m.group(0) == _GITHUB_BOT
        )
        self.assertEqual(hits, [], "only the anonymous GitHub address (or example.com) may appear")

    def test_no_home_folder_paths(self):
        self.assertEqual(self._where(_HOME_PATH), [], "a path like this reveals the user name of a Mac")

    def test_no_secrets(self):
        self.assertEqual(self._where(_SECRET), [])

    def test_the_assistant_name_never_appears(self):
        in_names = [str(f.relative_to(ROOT)) for f in self.files if _ASSISTANT.search(str(f.relative_to(ROOT)))]
        self.assertEqual(in_names + self._where(_ASSISTANT), [])

    def test_web_pages_have_no_em_dash(self):
        pages = [f for f in self.texts if f.parent.name == "web" and f.suffix == ".html"]
        self.assertEqual(len(pages), 2)
        self.assertEqual([p.name for p in pages if "—" in self.texts[p]], [])

    def test_images_carry_no_metadata(self):
        for f in self.files:
            data = f.read_bytes()
            if f.suffix.lower() == ".png":
                found = sorted(k.decode() for k in set(_png_chunks(data)) & _PNG_METADATA)
                self.assertEqual(found, [], f"{f.relative_to(ROOT)} carries metadata")
            elif f.suffix.lower() in (".jpg", ".jpeg"):
                self.assertNotIn(b"Exif\x00\x00", data[:65536], f"{f.relative_to(ROOT)} carries EXIF")

    def test_commits_use_the_anonymous_address(self):
        log = _git("log", "--format=%an <%ae>%x00%cn <%ce>%x00%B%x01")
        for entry in filter(str.strip, log.split("\x01")):
            author, committer, message = entry.strip("\n").split("\x00", 2)
            for who in (author, committer):
                ok = who.endswith((_NOREPLY + ">", f"<{_GITHUB_BOT}>"))
                self.assertTrue(ok, f"commit signed with a real address: {who}")
            self.assertIsNone(_ASSISTANT.search(message), "commit message names the assistant")


class TestTheRulesCatchWhatTheyShould(unittest.TestCase):
    """The patterns above, on known good and bad samples (so a silent typo cannot disable a rule)."""

    def test_email(self):
        bad = "write to someone@" + "mail.net"  # every bad sample is built from parts, so this file stays clean
        self.assertIsNotNone(_EMAIL.search(bad))
        for ok in ("x@users.noreply.github.com", "a@example.com"):
            self.assertTrue(_EMAIL.search(ok).group(1).endswith(_SAFE_EMAIL_DOMAINS))
        self.assertIsNone(_EMAIL.search('"\\n@pytest.mark.skip"'))  # a decorator inside a string is not an e-mail

    def test_home_path_secret_and_name(self):
        self.assertIsNotNone(_HOME_PATH.search("/Users/" + "someone/project"))
        self.assertIsNotNone(_SECRET.search("token = ghp_" + "a" * 36))
        self.assertIsNotNone(_SECRET.search("-----BEGIN OPENSSH " + "PRIVATE KEY-----"))  # in two halves, like above
        self.assertIsNone(_SECRET.search(r"github_pat_[A-Za-z0-9_]{30,}"))  # the self-audit's own regex
        self.assertIsNotNone(_ASSISTANT.search("Co-Authored-By: " + "Cl" + "aude"))

    def test_png_metadata(self):
        png = b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\x00eXIf" + b"\x00" * 4 + b"\x00\x00\x00\x00IEND" + b"\x00" * 4
        self.assertIn(b"eXIf", _png_chunks(png))


if __name__ == "__main__":
    unittest.main()
