#!/usr/bin/env python3
# orchestrator.py, Level 4: an orchestration layer over the Level 3 agent.
#
# Turns the Level 3 mini-agent into a more CAPABLE system without losing speed,
# using "context engineering": instead of one giant prompt (which sank the
# single-agent approach on the local model), it orchestrates sub-steps, each with
# its own focused, minimal context:
#
#        PLANNER            EXECUTOR (L3 loop)        VERIFIER
#     (decompose)    ->    (runs each step)    ->    (checks and fixes)
#
# It reuses the Level 3 Executor as-is (agent.run_agent); it does not rewrite it.
# Each call keeps a light context: the planner sees no tools and the verifier
# starts from a fresh conversation (both calls are minimal). The executor does
# accumulate context, but only within a single task (bounded by a few high-level
# steps and by clipping outputs to MAX_OUTPUT_CHARS) and it RESETS between tasks:
# what persists across tasks is the workspace on disk, not the chat.
#
# Phases implemented here:
#   4A, Self-verification: before calling something done, the verifier checks the
#       work (runs the code/tests) and, on failure, triggers a fix.
#   4B, Planning: for multi-step tasks, plan first, then execute step by step
#       with focused context.
#   4C, Per-role model router: the executor ALWAYS runs locally; the planner and
#       verifier may use a powerful CLOUD model (FREE only), configured in
#       config.env. Without a key, everything stays 100% local (fallback). See
#       build_router().
#
# Usage:
#   ./agent.sh                    interactive REPL (full L4 system)
#   ./agent.sh -p "task..."       one task and exit
#   ./agent.sh --fast -p "..."    no plan or verification (~ Level 3, faster)
import argparse
import json
import os
import re
import subprocess  # nosec B404: to open the browser / spawn helper processes; fixed list, no shell
import sys
import time
from pathlib import Path

from openai import OpenAI  # per-role client for the STRONG (cloud) brain, Phase 4C

# We reuse the whole of Level 3 as a foundation (nothing in it is rewritten).
from agent import (
    run_agent,         # the Executor: Level 3 tool-calling loop
    chat,              # direct streaming reply, no tools (5D, chat triage)
    load_config,
    color,
    build_client,
    SYSTEM_PROMPT,     # short executor system prompt (light context)
    WORKDIR,           # working directory (for the snapshot/restore safeguard)
    VENV_BIN,          # project toolchain (to run the workspace tests)
    BASH_TIMEOUT,      # subprocess timeout
    MAX_OUTPUT_CHARS,  # L3 clip cap (reused to bound the memory call)
    _stats,            # internal L3 helper reused on purpose: planner tok/s
)
from events import emit  # observability hook (no-op if nobody listens -> CLI untouched)
import agent as _agent   # to point WORKDIR at the active project (Level 7)
import projects          # Level 7A: projects + persistent conversation (light context)
import chats             # Level 14: chats inside projects (migration + per-chat conversation)
import time_machine      # Level 8A: snapshots + undo (time machine)

# Level 14: WORKDIR = the PROJECT folder (documents + .knowledge + .memory.md, SHARED);
# CHATDIR = the active CHAT folder (conversation + chat summary/memory + lessons + snapshots).
# By default CHATDIR = WORKDIR (no project/chat, e.g. cwd/--fast mode -> previous behavior).
CHATDIR = WORKDIR


# --- 1) PER-ROLE MODEL ROUTER (Phase 4C): executor LOCAL; planner/verifier STRONG ---
def _is_free_endpoint(base_url, model):
    """Anti-cost guard: is this (endpoint, model) FREE with no risk of a charge?
    - OpenRouter: ONLY models with the ':free' suffix (its catalog mixes free and paid).
    - Cerebras (api.cerebras.ai): the FREE tier is per ACCOUNT (e.g. 1M tok/day, 30 req/min);
      with no card added there is no cost, so the configured model is accepted (the user is
      responsible for NOT adding a payment method to their account).
    - Other unknown providers: to be safe, require ':free'."""
    u = (base_url or "").lower()
    if "cerebras.ai" in u:
        return True
    return (model or "").endswith(":free")


def build_router(cfg, force_local=False):
    """Return {role: (client, model, label)} for 'executor', 'planner' and 'verifier'.

    - executor: ALWAYS the LOCAL model (vllm-mlx). No cloud is spent on routine work.
    - planner/verifier: the STRONG brain (a CLOUD model) if it is well configured and
      FREE; otherwise the LOCAL one (fallback -> nothing breaks without a key).
    - force_local=True: ignores STRONG and puts all three roles on local (useful to measure
      the local-only-14B baseline with the SAME config as STRONG).

    HARD RESTRICTION (anti-cost): STRONG only activates if the model id ends in ':free'.
    Anything else is REJECTED and falls back to local, so we never incur charges. The key is
    read from config.env or the STRONG_API_KEY environment variable (never from the code).
    The label ('LOCAL:...' / 'STRONG:...') is used in per-call logging (transparency, and to
    watch free-tier usage).
    """
    local_client, local_model, _ = build_client(cfg)   # validates the local server responds
    local = (local_client, local_model, f"LOCAL:{local_model}")

    if force_local:  # local-only-14B baseline: all three roles on local
        print(color("  (forced --local mode: planner/verifier on local too)", "gray"))
        return {"executor": local, "planner": local, "verifier": local}

    strong = None
    s_url = (cfg.get("STRONG_BASE_URL") or "").strip()
    s_model = (cfg.get("STRONG_MODEL") or "").strip()
    s_key = (cfg.get("STRONG_API_KEY") or os.environ.get("STRONG_API_KEY", "")).strip()

    if s_url or s_model or s_key:  # the user IS TRYING to enable the cloud brain
        if not (s_url and s_model and s_key):
            print(color("  ! STRONG_* incomplete (missing base_url/model/api_key) -> 100% local", "yellow"))
        elif not _is_free_endpoint(s_url, s_model):
            # Anti-cost guard: never a paid model/endpoint.
            print(color(f"  x STRONG '{s_model}' @ {s_url} does not guarantee free use "
                        "(OpenRouter requires the ':free' suffix); IGNORED for safety -> 100% local", "red"))
        else:
            try:
                sc = OpenAI(base_url=s_url, api_key=s_key, timeout=180.0)  # don't wait 10 min if it hangs
                sc.models.list()  # METADATA check (does not consume generation quota)
                strong = (sc, s_model, f"STRONG:{s_model}")
                note = ("Cerebras free: 1M tok/day, 30 req/min"
                        if "cerebras.ai" in s_url.lower() else "cloud, FREE")
                print(color(f"  + STRONG brain active ({note}): {s_model}", "green"))
            except Exception as e:  # noqa: BLE001: if it fails, don't break; fall back to local
                print(color(f"  ! could not connect to STRONG ({type(e).__name__}); 100% local", "yellow"))

    return {
        "executor": local,            # day to day: ALWAYS local
        "planner": strong or local,   # hard phase -> STRONG if any; else local
        "verifier": strong or local,
    }


# --- 2) PLANNER (Phase 4B): decomposes the task into steps. Focused context, NO tools. ---
PLANNER_PROMPT = (
    "You are a planner. You split a programming task into VERY FEW high-level steps "
    "(ideally 1; at most 5). Each step is a unit of work with a concrete DELIVERABLE "
    "(e.g. 'create module X with function Y', 'create the tests for X'). NEVER include "
    "micro-actions or mechanics as separate steps: no 'open terminal', 'save the file', "
    "'run', 'test', 'verify' or 'record results'; running and checking are handled "
    "automatically by another agent.\n"
    "COVERAGE: cover EVERYTHING the task literally asks for even if it is underspecified; "
    "do not fall short by leaving out implicit parts. If it asks for something EXECUTABLE "
    "(e.g. a 'command-line calculator'), include creating the EXECUTABLE program itself (the "
    "command-line interface), not just loose functions or tests. Fill gaps reasonably and "
    "minimally; if you assume something to fill a gap, say so in the step text. Even so, do "
    "NOT invent extra functions nobody asked for.\n"
    "BY DEFAULT, A SINGLE STEP: if the task touches a single file or function, return ONE step. "
    "A QUERY (read a URL or file and answer, or search the web and summarize) is also a SINGLE "
    "step. Only split into several steps when there are clearly SEPARATE DELIVERABLES (e.g. the "
    "program and, separately, its tests). When in doubt, FEWER steps.\n"
    "Respond EXCLUSIVELY with valid JSON, no extra text.\n"
    "Examples:\n"
    'Task: "fix the mean function in util.py for the empty list"\n'
    '{"steps": ["Fix util.py so mean() returns 0 for the empty list"]}\n'
    'Task: "read README.md and tell me how to install the project"\n'
    '{"steps": ["Read README.md and answer how the project is installed"]}\n'
    'Task: "look up what the MLX framework is and summarize it"\n'
    '{"steps": ["Search the web for what the MLX framework is and summarize it"]}\n'
    'Task: "a command-line calculator with basic operations and tests"\n'
    '{"steps": ["Create the terminal-EXECUTABLE calculator (CLI) with the basic '
    'operations (add, subtract, multiply, divide)", "Create the tests for '
    'the calculator"]}'
)


def _extract_steps(text):
    """Pull the step list out of the planner text, tolerating ```json```/noise."""
    data = None
    try:
        data = json.loads(text)
    except Exception:  # noqa: BLE001
        m = re.search(r"\{.*\}", text or "", re.S)  # first {...} object
        if m:
            try:
                data = json.loads(m.group(0))
            except Exception:  # noqa: BLE001
                data = None
    if isinstance(data, dict) and isinstance(data.get("steps"), list):
        return [str(p).strip() for p in data["steps"] if str(p).strip()]
    return []


# Verbs that signal pure mechanics or testing. Running and verifying are handled
# AUTOMATICALLY by the verifier, so a plan step starting with one of these is noise
# that inflates context: it is dropped. (Deterministic safety net in case the planner
# ignores the instruction not to include them.)
_DISCARD_VERBS = (
    "test", "run", "execute", "verify", "check", "validate", "launch", "open",
    "save", "log", "record",
)


def _filter_steps(steps):
    """Drop pure mechanics/testing steps. May return an EMPTY list: the caller
    (plan_task) already collapses that case to [task]."""
    return [p for p in steps
            if not p.lower().lstrip("0123456789.)- ").startswith(_DISCARD_VERBS)]


# Hard step cap: PLANNER_PROMPT promises 'at most 5'; we make it true even if the model
# disobeys (keeps the executor context bounded).
MAX_STEPS = 5


def plan_task(client, model, task):
    """Ask the model for a list of steps, discard mechanics/testing ones, cap their count,
    and if anything goes wrong, treat the task as a single step (never crashes)."""
    messages = [
        {"role": "system", "content": PLANNER_PROMPT},
        {"role": "user", "content": f"Task: {task}"},
    ]
    t0 = time.time()
    # max_tokens: a plan is <5 steps of JSON; bound the generation (anti-runaway: the root cause
    # of the infinite loop was generating with NO cap, including on this first call of each task).
    try:
        resp = client.chat.completions.create(model=model, messages=messages, max_tokens=512)
        _stats(resp, time.time() - t0)
        text = (resp.choices[0].message.content or "") if resp.choices else ""
    except Exception:  # noqa: BLE001: if the planner fails (network down, etc.), one task = one step
        print(color("  ! the planner failed; continuing with the task as a single step", "yellow"))
        return [task]
    return (_filter_steps(_extract_steps(text)) or [task])[:MAX_STEPS]


# --- 3) VERIFIER (Phase 4A): checks the work independently. WITH tools. ---
VERIFIER_PROMPT = (
    "You are an independent, skeptical verifier. Your job is NOT to redo the task, but to check "
    "whether it is REALLY finished and correct in the current workspace.\n"
    "ALWAYS START with list_dir to DISCOVER which files ACTUALLY exist and identify the "
    "deliverable by its REAL name (the one in the listing). NEVER assume the name from the task "
    "text: the language of the wording may not match the filename (e.g. the task says "
    "'calculator' but the file is called calculator.py). Test ONLY files that exist in the "
    "listing; if you refer to one that does not appear, the error is YOURS (an assumed name), "
    "NOT the deliverable's, and is NOT grounds for FAILURE.\n"
    "Then do TWO checks, both by ACTUALLY RUNNING with the tools:\n"
    "1) IT WORKS: read the relevant files and run the code/tests with run_bash "
    "to confirm there are no errors and they do what they should.\n"
    "2) IT MEETS THE INTENT of the task: reread the TASK and check that the deliverable "
    "lets you do LITERALLY what it asks. If the task asks for something executable in a "
    "specific way (e.g. a 'command-line calculator'), CHECK IT that way for real: run it "
    "that way (e.g. launch it from the terminal with arguments) and see if it really works. "
    "A module importing or tests passing is NOT enough if the task asked to be able to USE "
    "it another way.\n"
    "3) EVERY REQUESTED PART: enumerate ALL the parts the task explicitly asks for "
    "(each deliverable counts; e.g. 'a command-line calculator' AND 'its tests' are TWO parts) "
    "and check EACH one. If any is missing, e.g. the task asked for tests and there are no "
    "tests, or there are but they don't run or don't pass, it is FAILURE.\n"
    "4) RE-CHECK YOUR PRIOR COMPLAINTS: if problems detected in previous attempts are listed "
    "below, re-check EACH ONE specifically (by running it) and confirm it is resolved; if any "
    "persists, it is FAILURE.\n"
    "IMPORTANT BALANCE: derive the intent ONLY from what the task literally says; do NOT "
    "invent extra requirements (functions, files, flags, formatting or polish the task did not "
    "ask for). It is FAILURE only if a part of what the task asks for LITERALLY is not met or "
    "cannot be used as asked.\n"
    "GOLDEN RULE: give 'VERDICT: OK' ONLY if you have POSITIVELY confirmed that ALL requested "
    "parts exist and work and that your prior complaints are resolved. When in doubt, or if you "
    "cannot confirm something, it is FAILURE, NEVER a false OK.\n"
    "ALWAYS end with a final line EXACTLY in this format:\n"
    "  VERDICT: OK\n"
    "or else:\n"
    "  VERDICT: FAILURE: <which part of the request is not met and how to fix it>"
)


def _parse_verdict(text):
    """Interpret the verdict. Returns (state, detail) TRI-STATE:
      - True  -> 'VERDICT: OK'   (correct)
      - False -> 'VERDICT: FAILURE: ...'  (must fix; detail = what fails)
      - None  -> INCONCLUSIVE: no 'VERDICT' line at all (the verifier did not
                finish: e.g. ran out of iterations or the server did not respond).

    Distinguishing 'inconclusive' from 'OK' is KEY: a verifier that did not finish its
    check must NOT be taken as good (that would nullify 4A self-verification). We decide
    OK/FAILURE by the token RIGHT AFTER 'VERDICT', not by searching for 'FAILURE' in the
    whole line, so 'VERDICT: OK, no failures anymore' reads as OK.
    """
    lines = [l for l in (text or "").splitlines() if "VERDICT" in l.upper()]
    if not lines:
        return None, "the verifier did not emit a clear VERDICT"
    line = lines[-1].strip()
    i = line.upper().rfind("VERDICT")
    rest = line[i + len("VERDICT"):].lstrip(": ").strip()
    if rest.upper().startswith("FAILURE"):
        detail = rest[len("FAILURE"):].lstrip(": ").strip()
        return False, (detail or "(the verifier gave no failure detail)")
    if not rest:  # 'VERDICT:' with NO body (e.g. truncated by length) -> INCONCLUSIVE, not a false OK
        return None, "the verifier did not complete the VERDICT"
    return True, ""


def verify(client, model, task, confirm_bash=False, prior_complaints=None, artifacts=None,
           contents=None):
    """Launch a verifier in a FRESH conversation (focused context): reuse the L3 Executor
    with a verification system prompt. Returns the tri-state verdict (True/False/None) from
    _parse_verdict.

    `prior_complaints`: list of FAILUREs from previous attempts; passed so it RE-CHECKS them
    specifically before giving OK (rigor: no OK with unresolved complaints).
    `artifacts`: paths the task actually CREATED/MODIFIED (workspace diff, deterministic). Given to
    the verifier so it checks THOSE files by their REAL name, and does not re-guess one from the
    wording (e.g. the task says 'a notebook' and the file is called hello_world.ipynb).
    """
    instruction = f"Check whether this task is done and correct in the current workspace:\n{task}"
    if artifacts:
        listing = "\n".join(f"- {a}" for a in artifacts[:20])
        more = "" if len(artifacts) <= 20 else f"\n(...and {len(artifacts) - 20} more)"
        instruction += ("\n\nThe EXECUTOR created or modified EXACTLY these files (they are the REAL "
                        "DELIVERABLE of this task; check them by their REAL NAME and do NOT assume other "
                        f"names from the wording):\n{listing}{more}")
    if contents:
        instruction += ("\n\nREAL CONTENT of what the task ADDED (COMPLETE source, deterministic; "
                        "judge it by THIS, not by a summary or the raw JSON). Cells that already "
                        "existed are NOT a failure, and if the task says 'do not run' do NOT require "
                        f"the code to run:\n{contents}")
    if prior_complaints:
        listing = "\n".join(f"- {q}" for q in prior_complaints[-8:])   # only the last ones (light context)
        instruction += ("\n\nProblems detected in PREVIOUS verifications, re-check EACH ONE "
                        "(by running it) and confirm it is resolved before giving OK:\n"
                        f"{listing}")
    messages = [
        {"role": "system", "content": VERIFIER_PROMPT},
        {"role": "user", "content": instruction},
    ]
    try:
        text = run_agent(client, model, messages, confirm_bash)
    except Exception as e:  # noqa: BLE001: a HOT failure of the verifier (e.g. STRONG/cloud
        # down mid-task) must NOT topple the pipeline: treat it as INCONCLUSIVE (retried within
        # the cap or stopped honestly), never as a false OK.
        print(color(f"  ! the verifier failed ({type(e).__name__}); treating as inconclusive", "yellow"))
        return None, f"the verifier could not run ({type(e).__name__})"
    return _parse_verdict(text)


# --- 4) ORCHESTRATOR: ties PLAN -> EXECUTION -> VERIFICATION/CORRECTION together ---
# Safeguard: a "fix" must never break/rename something that ALREADY works.
MAX_SNAP_BYTES = 5_000_000   # per-file cap in the snapshot: a bigger one is MARKED (size/mtime),
                             # not loaded into memory (avoids blowing up RAM with large models/datasets).


def _snapshot_workspace():
    """Return {relative_path: value} of the workspace files, to revert a harmful fix and for the
    artifact diff. `value` = BYTES if the file is small, or a light MARKER ('__large__', size, mtime)
    if it exceeds MAX_SNAP_BYTES (so the diff detects changes without loading it into memory; those
    are not restored byte for byte). Ignores hidden (.venv/.git/.knowledge/.snapshots), __pycache__,
    node_modules and .pyc."""
    snap = {}
    for p in WORKDIR.rglob("*"):
        parts = p.relative_to(WORKDIR).parts
        if any(part.startswith(".") or part in ("__pycache__", "node_modules")
               for part in parts):
            continue
        if p.is_file() and p.suffix != ".pyc":
            rel = p.relative_to(WORKDIR)
            try:
                st = p.stat()
                snap[rel] = (("__large__", st.st_size, int(st.st_mtime))
                             if st.st_size > MAX_SNAP_BYTES else p.read_bytes())
            except OSError:
                pass
    return snap


def _created_artifacts(snap_pre):
    """Files the task CREATED or MODIFIED (workspace diff against the snapshot taken BEFORE running).
    It is the DETERMINISTIC truth of what the executor did -> the verifier checks THESE, not a name
    re-guessed from the wording. Sorted; excludes what did not change."""
    current = _snapshot_workspace()
    return sorted(str(rel) for rel, data in current.items() if snap_pre.get(rel) != data)


def _artifact_contents(snap_pre):
    """REAL content of what the task changed in each NOTEBOOK (the ADDED cells, with their full
    source), DETERMINISTICALLY. Given to the verifier so it judges by the REAL content (not by the
    read_notebook summary or the raw JSON) and knows what the delta is. Bounded."""
    import notebooks as _nb
    current = _snapshot_workspace()
    parts = []
    for rel, data in sorted(current.items(), key=lambda kv: str(kv[0])):
        if snap_pre.get(rel) == data or rel.suffix.lower() != ".ipynb":   # rel is already a relative Path
            continue
        try:
            after_txt = data.decode("utf-8", errors="replace")
            before = snap_pre.get(rel)
            before_txt = before.decode("utf-8", errors="replace") if before is not None else "{}"
        except (AttributeError, UnicodeError):
            continue
        rendered = _nb.render_cells_diff(before_txt, after_txt)
        if rendered:
            parts.append(f"### {rel}\n{rendered}")
    return "\n\n".join(parts)[:6000]   # bounded context


def _restore_workspace(snap):
    """Rewrite the snapshot files (recreate deleted/renamed ones, revert changed ones). Does NOT
    delete files the fix may have ADDED (we don't assume they are unwanted)."""
    for rel, data in snap.items():
        if not isinstance(data, bytes):             # large file (marker): not restored byte for byte
            continue
        dest = WORKDIR / rel
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
        except OSError:
            pass


def _tests_pass():
    """Do tests exist in the workspace and PASS? Runs `unittest discover` with the project
    toolchain. True only if AT LEAST 1 test ran and the result was OK."""
    env = os.environ.copy()
    if VENV_BIN.is_dir():
        env["PATH"] = f"{VENV_BIN}{os.pathsep}{env.get('PATH', '')}"
    try:
        proc = subprocess.run(["python3", "-m", "unittest", "discover"],  # noqa: S603,S607: fixed command (no shell); python3 via venv PATH on purpose  # nosec B603 B607
                              cwd=str(WORKDIR), env=env,
                              capture_output=True, text=True, timeout=BASH_TIMEOUT)
    except Exception:  # noqa: BLE001: if they can't run, assume there are no passing tests
        return False
    output = (proc.stdout or "") + (proc.stderr or "")
    m = re.search(r"Ran (\d+) test", output)
    ran = int(m.group(1)) if m else 0
    return ran > 0 and proc.returncode == 0


def _final_summary(response, steps, multistep, verified):
    """Coherent final message. If verification was OK, the message is a CLEAR SUCCESS and NEVER
    drags along a partial-failure phrase or an unfilled marker (even if the executor sneaks one in):
    if its reply sounds like 'I couldn't...', it is ignored and a clean success is given. For several
    stages a one-line summary is added (no extra calls)."""
    resp = (response or "").strip()
    if verified:
        # A 'couldn't...' from the executor cannot contradict an OK verdict -> that phrase is dropped.
        sounds_like_failure = any(m in resp.lower() for m in
                                  ("couldn't", "could not", "was unable", "unable to", "but not"))
        body = "" if sounds_like_failure else resp
        base = (f"Task completed and verified ({len(steps)} steps)." if multistep
                else "Done and verified.")
        return f"{base} {body}".strip() if body else base
    # Not verified (delivering an inconclusive one with no real failure): without faking verification,
    # and with an honest NOTE that it was left unclosed (previously the single-step delivery did not say so).
    if multistep:
        return f"Task completed ({len(steps)} steps, verification NOT concluded). Last step: {resp}"
    return f"{resp}\n\n(note: I did the work but could not fully close verification)" if resp else resp


# --- Intent triage (5D): chat vs task ---
TRIAGE_PROMPT = (
    "Classify the user's request in one word:\n"
    "- CHAT: answered DIRECTLY with general knowledge (questions, explanations, definitions, "
    "conversation), WITHOUT tools (no searching, no reading files, no running).\n"
    "- QUERY: needs tools to LOOK SOMETHING UP (search the internet, read a file or page, list, "
    "run to inspect), but the DELIVERABLE is a TEXT ANSWER; it does NOT create or modify any file.\n"
    "- TASK: CREATES or MODIFIES files/code (create, edit, fix, generate a document/PDF, write a "
    "script, leave tests passing...). The deliverable is a FILE/change on disk.\n"
    "RULES:\n"
    "1) Does it CREATE or MODIFY a file? -> TASK. Only look up and answer in text? -> QUERY. "
    "Answered without tools? -> CHAT. When in DOUBT between QUERY and TASK -> TASK.\n"
    "2) 'create/make/generate/write a document/report/script/file' is TASK even if the topic "
    "comes from a search.\n"
    "Examples: 'explain what Python is'->CHAT; 'capital of France?'->CHAT; 'look up the news on "
    "Python 3.13 and summarize it'->QUERY; 'read config.json and tell me the port'->QUERY; 'summarize the "
    "README.md'->QUERY; 'what files are here?'->QUERY; 'create hello.py with a print'->TASK; "
    "'make a PDF report about X'->TASK; 'create a markdown document about coffee'->TASK; "
    "'fix the bug in calc.py'->TASK.\n"
    "Respond with ONE word only: CHAT, QUERY or TASK."
)


# L15-t2 Phase B: triage budget. With 6 tokens, a 'thinking' brain (Gemma/Qwen3) spends them all
# THINKING and never answers -> empty content -> everything fell to TASK (a 'hi' entered the pipeline).
# 256 leaves room for the thinking + the word; with Qwen (no thinking) it answers 1 word as before
# (the prompt requires ONE word and the model stops when done: the real extra cost is ~0).
TRIAGE_MAX_TOKENS = 256


def _triage(client, model, task):
    """Is the request CHAT (answer directly) or TASK (pipeline)? A short, focused call to the local
    model. On ANY doubt or failure -> 'TASK', to avoid losing real work."""
    try:
        r = client.chat.completions.create(model=model, max_tokens=TRIAGE_MAX_TOKENS, messages=[
            {"role": "system", "content": TRIAGE_PROMPT},
            {"role": "user", "content": task}])
        out = (r.choices[0].message.content or "").strip().upper() if r.choices else ""
    except Exception:  # noqa: BLE001: if the classifier fails, treat as TASK (safest)
        return "TASK"
    if "TASK" in out:                # any mention of TASK or doubt -> TASK (don't lose verification)
        return "TASK"
    if "CHAT" in out:
        return "CHAT"
    if "QUERY" in out:
        return "QUERY"
    return "TASK"                    # safe default


# --- Questions about YOUR documents (Level 8B) -> DETERMINISTIC RAG ---
# The 14B fails both at TRIAGING these questions (sometimes CHAT, no tools) and at INVOKING
# search_in_documents on its own -> hallucinates. So if the question is about the user's documents
# and the project has an index, we retrieve the top-k and answer FROM there citing the source,
# without depending on the model. Detection by possessive ('my notes', 'according to my document'...),
# specific so it does not hijack general questions ('the musical notes' does NOT match).
# Vocabulary of COMMON document nouns (Level 10C): besides notes/documents, it recognizes
# CV/resume, report, article, paper, thesis, presentation, slides, spreadsheet. Two branches, both
# anchored so as NOT to hijack general questions:
#   1) POSSESSIVE: "my CV", "my notes", "your report" (my/your + document-noun).
#   2) UPLOADED: "the PDF I uploaded", "the document I attached" (article + noun + upload verb).
_DOC_NOUNS = (r"(?:cv|resumes?|curriculum(?:\s+vitae)?|notes?|documents?|documentation|files?|"
              r"manuals?|pdfs?|reports?|articles?|papers?|memos?|thesis|theses|presentations?|"
              r"slides?|spreadsheets?|excel)")   # Level 12B: +excel/spreadsheet/slides (xlsx/pptx)
_RX_DOCS = re.compile(
    r"\b(?:my|your)\s+" + _DOC_NOUNS + r"\b"
    r"|\b(?:the|that|this|these|those|my|your)\s+" + _DOC_NOUNS +
    r"\s+(?:that\s+)?(?:i\s+|you\s+)?(?:uploaded|attached|loaded|sent|shared|gave|passed)",
    re.IGNORECASE)

# For the INDEXED FILENAME match: common words that must NOT trigger on their own
# (avoids 'the musical notes' matching a notes.md, or 'the data' matching data.csv).
_COMMON_DOC_WORDS = {"notes", "data", "text", "document", "documents", "file", "files",
                     "report", "reports", "code", "project", "readme", "index", "main",
                     "musical", "hello", "test", "temp", "final", "new", "draft"}
_DOC_ABBREVS = {"cv"}   # short but distinctive document abbreviations (matched by filename)


def _is_document_question(task):
    return bool(_RX_DOCS.search(task or ""))


# CREATE/MUTATE orders: a phrase that STARTS with a create/write/export imperative is a TASK,
# not a question about documents -> it must NOT trigger the RAG even if it names an indexed file
# ('create note.txt', 'create an excel', 'make me a spreadsheet', 'export X to csv', 'modify X.xlsx').
_RX_CREATE = re.compile(
    r"^\s*(?:please,?\s+)?"
    r"(?:create|make|write|draft|generate|save|export|add|append|insert|modify|edit|update|"
    r"replace|convert|delete|remove|rename|move|copy|fill|complete|build)\b", re.IGNORECASE)


def _is_create_order(task):
    return bool(_RX_CREATE.search(task or ""))


def _mentioned_document(task, base):
    """Level 10C: does the question name a file ACTUALLY indexed in the project? (e.g. 'according to
    my CV' with CV_Alex.pdf indexed). Returns the filename or None. Used ONLY to RESTRICT the search
    to the named doc (not to TRIGGER the RAG: that is decided by the regex, to avoid false positives
    with tasks like 'create note.txt'). Match on the file STEM (without the extension, so 'txt'/'pdf'
    don't match) with distinctive tokens (>=3 chars or the abbreviation cv; never common ones)."""
    import knowledge
    from pathlib import Path as _P
    # 8B tokenizer: splits on non-alphanumeric (also on '_' and '.', unlike \w) -> 'CV_Alex' -> cv, alex.
    def _distinctive(text):
        return {t for t in knowledge._tokenize(text)
                if (len(t) >= 3 or t in _DOC_ABBREVS) and t not in _COMMON_DOC_WORDS}
    q = _distinctive(task)
    if not q:
        return None
    for name in knowledge.indexed_names(base):
        if q & _distinctive(_P(name).stem):          # ONLY the stem (no extension: 'a.txt' -> 'a')
            return name
    return None


def answer_from_documents(client, model, task, prior_context=None):
    """If the question is ABOUT the user's documents and the project has a knowledge index, retrieve
    the top-k fragments and answer FROM them citing the source, DETERMINISTICALLY (without depending
    on triage or the model's tool-calling). Returns the answer or None (does not apply)."""
    import knowledge
    if not knowledge.has_index(WORKDIR):
        return None
    file = _mentioned_document(task, WORKDIR)   # indexed file named by its own name (or None)
    # TRIGGER if: (a) the document regex matches (possessive/uploaded: 'according to my excel I uploaded'), OR
    # (b) the question NAMES a file ACTUALLY indexed ('according to my budget', with budget.xlsx indexed),
    # unless it is a create/mutate ORDER ('create note.txt', 'create an excel', 'export to excel', 'modify
    # X.xlsx'), which is a TASK and must NOT hijack the RAG. The possessives (a) stay the same; the
    # create/mutate guard only applies to the new by-filename path (b).
    if not (_is_document_question(task)
            or (file is not None and not _is_create_order(task))):
        return None
    try:
        res = knowledge.search(WORKDIR, task, k=4, only_source=file)
    except Exception:  # noqa: BLE001
        return None
    emit("intent", mode="documents")
    if not res:                                       # no match -> DETERMINISTIC cut (never invent)
        print(color("  [docs] question about your documents -> not found in the index", "gray"))
        return "I did not find it in your documents."
    fragments = knowledge.format_result(res)
    print(color("  [docs] question about your documents -> answering from the knowledge index", "gray"))
    system_msg = ("Answer the user's question USING ONLY these fragments from THEIR documents, and "
                  "CITE the source in brackets [file]. If the answer is NOT in the fragments, say so "
                  "clearly ('I did not find it in your documents'); NEVER make it up.\n\n"
                  "FRAGMENTS:\n" + fragments)
    ctx = list(prior_context or []) + [{"role": "system", "content": system_msg}]
    return chat(client, model, task, prior_context=ctx)


# --- Skills (Level 9A): pluggable capability, light context ---
# After the triage-RAG and before the task, we check whether a SKILL matches the message
# (DETERMINISTIC selection by TF-IDF with a threshold, in skills.py: zero false triggers). If it
# matches, we inject ITS instructions ONLY this turn (light context) and, if it declares a
# deterministic script, we run it through the SAME guard/tools and add its findings. The model
# answers guided by both (detection is exact, the script; the rewrite is done by the model). Thin
# layer: it does not touch the core.
MAX_FINDINGS_CHARS = 3000         # cap on the injected script findings (light context)


def _run_skill_script(script_path, text):
    """Run a skill's DECLARED script over `text` through the SAME path as the agent's tools (reuses
    agent.run_bash: destructive guard + timeout + venv PATH). The text goes through a TEMP file (not
    the command line -> no shell injection). Returns the findings (clipped) or '' if there was no
    useful output. Best-effort: a failure never topples the turn."""
    import shlex
    import tempfile
    tmp = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as fh:
            fh.write(text or "")
            tmp = fh.name
        output = _agent.run_bash(f"python {shlex.quote(str(script_path))} {shlex.quote(tmp)}")
    except OSError:
        return ""
    finally:
        if tmp:
            try:
                os.remove(tmp)
            except OSError:
                pass
    # run_bash returns "[exit N]\n<output>"; keep the output, bounded (light context)
    body = output.split("\n", 1)[1] if output.startswith("[exit") and "\n" in output else output
    return body.strip()[:MAX_FINDINGS_CHARS]


# L17 Phase B: a TEXT-ONLY skill must not hijack a CREATE/MUTATE order for a FILE:
# 'create a file summary.txt with...' triggered `academic-summary` by TF-IDF (the token "summary") and
# the model NARRATED the creation without creating anything (skills answer without tools). DETERMINISTIC,
# narrow guard: an ANCHORED create/mutate verb (_RX_CREATE, the same as the 12B RAG) + a file mention
# (a name with an extension starting with a letter, so '3.5' does not match, or the word file). It only
# blocks AUTOMATIC selection; a skill FORCED by hand in the panel (9C) is always respected.
_RX_FILE_MENTION = re.compile(
    r"\b[\w()\[\]-]+\.[a-zA-Z][a-zA-Z0-9]{0,4}\b|\bfiles?\b", re.IGNORECASE)


def _is_file_task(task):
    return _is_create_order(task) and bool(_RX_FILE_MENTION.search(task or ""))


def answer_with_skill(client, model, cfg, task, prior_context=None, force=None):
    """If a SKILL matches `task` (deterministic selection by threshold), or if `force` names a skill
    (picked by hand on the web, 9C), inject ITS instructions (light context, this turn only) and, if
    it declares a script, run it through the guard and add its findings. The model answers guided by
    both, preserving the meaning. Returns the answer, or None if none applies."""
    import skills
    if not skills.enabled(cfg):
        return None
    # 9D: the turn's WORKDIR is the active project folder (or the cwd); if it has its own `skills/`,
    # those skills ARE ADDED to the global catalog only for this project (they don't mix).
    project_dir = str(WORKDIR)
    if force:
        skill = skills.by_name(force, cfg, project_dir=project_dir)
    elif _is_file_task(task):
        return None    # create/mutate order for a FILE -> to the pipeline (a skill would narrate without creating)
    else:
        skill = skills.select(task, cfg, project_dir=project_dir)
    if skill is None:
        return None
    instr = skills.instructions(skill, cfg)
    blocks = [instr] if instr else []
    path = skills.script_path(skill)
    if path is not None:
        findings = _run_skill_script(path, task)
        if findings:
            blocks.append("SCRIPT FINDINGS (deterministic; apply each one):\n" + findings)
    if not blocks:
        return None                                   # skill with no useful body -> don't force the path
    emit("skill", name=skill.name, forced=bool(force))
    print(color(f"  [skill] '{skill.name}' -> instructions injected (light context)", "gray"))
    system_msg = ("Use this SKILL to answer the user. Follow its instructions to the letter and do "
                  "NOT change the meaning of the original text; return only the requested result, "
                  "without meta-comments.\n\n" + "\n\n".join(blocks))
    ctx = list(prior_context or []) + [{"role": "system", "content": system_msg}]
    return chat(client, model, task, prior_context=ctx)


# --- Metacognition (Level 8C): does the task look HARD? warn BEFORE, without blocking ---
_HARD_KEYWORDS = (
    "refactor", "rewrite", "migrat", "architecture", "optimiz", "performance", "concurren",
    "async", "asynchron", "multithread", "paralleliz", "encrypt", "authenticat",
    "database", "from scratch", "whole project", "all the files", "entire application",
    "end-to-end", "start to finish", "deploy", "framework", "compiler",
    "interpreter", "complex algorithm", "complete system",
)
DIFFICULTY_PROMPT = (
    "You are an evaluator. For a programming assistant with a small LOCAL model, is this task "
    "EASY, MODERATE or HARD? Consider the number of steps, ambiguity and scope. Respond with "
    "ONE word only: EASY, MODERATE or HARD."
)


def _difficulty_heuristic(task):
    """FREE signal (no model): broad-scope words + request length."""
    t = (task or "").lower()
    kws = [k for k in _HARD_KEYWORDS if k in t]
    score, reasons = min(2, len(kws)), []
    if kws:
        reasons.append("broad scope (" + ", ".join(kws[:2]) + ")")
    if len(task or "") > 240:
        score += 1
        reasons.append("long/detailed request")
    return score, "; ".join(reasons)


def _self_assess_difficulty(client, model, task):
    try:
        r = client.chat.completions.create(model=model, max_tokens=4, messages=[
            {"role": "system", "content": DIFFICULTY_PROMPT},
            {"role": "user", "content": (task or "")[:400]}])
        out = (r.choices[0].message.content or "").strip().upper() if r.choices else ""
    except Exception:  # noqa: BLE001
        return ""
    if "HARD" in out:
        return "HARD"
    if "MODERA" in out:
        return "MODERATE"
    return "EASY" if "EASY" in out else ""


def estimate_difficulty(cfg, task, client, model):
    """('easy'|'moderate'|'hard', reason). Heuristic (free) + a brief model self-assessment ONLY if
    the heuristic does NOT see it as easy (a simple task spends no call and gives no false warnings).
    Biased against false positives: 'hard' only with a clear signal (strong heuristic, or medium +
    self-assessment)."""
    score, reasons = _difficulty_heuristic(task)
    if score == 0:
        return "easy", ""                         # clearly easy -> no call, no warning
    auto = _self_assess_difficulty(client, model, task)
    if score >= 3 or (score >= 2 and auto == "HARD"):
        return "hard", reasons
    if score >= 1 or auto in ("HARD", "MODERATE"):
        return "moderate", reasons
    return "easy", ""


# --- 6C) SELF-CRITIQUE (review the own plan/output) and LESSONS (learn from errors) ---
# These are SHORT, focused calls (they don't inflate context): a cheap second opinion. They use the
# ROLE brain (planner/verifier): local by default, or STRONG if configured (then they count against
# its free quota). Quality is limited by the 14B; the structure helps regardless (see log). All can
# be disabled (SELF_CRITIQUE / LEARN_FROM_ERRORS, --no-self-critique).
def _clip_short(text, n=1500):
    text = text or ""
    return text if len(text) <= n else text[:n] + "...[clipped]"


PLAN_CRITIC_PROMPT = (
    "You are a plan reviewer, BRIEF and demanding. I give you a TASK and a PLAN (list of steps). "
    "Does the plan fully solve the task, with no extra steps or micro-actions? If it is fine, "
    "respond ONLY {\"ok\": true}. If something CONCRETE is missing or extra, respond ONLY with the "
    "corrected plan {\"steps\": [\"...\"]} (few steps, each with a deliverable). Do NOT invent "
    "requirements the task does not ask for.")


def critique_plan(client, model, task, steps):
    """6C: second opinion on the PLAN before running it. Returns the steps (corrected or the same).
    Cheap; on failure or doubt, keeps the original plan (does not get in the way)."""
    try:
        r = client.chat.completions.create(model=model, max_tokens=300, messages=[
            {"role": "system", "content": PLAN_CRITIC_PROMPT},
            {"role": "user", "content": f"TASK: {task}\nPLAN: {json.dumps(steps, ensure_ascii=False)}"}])
        text = (r.choices[0].message.content or "") if r.choices else ""
    except Exception:  # noqa: BLE001
        return steps
    new_steps = _filter_steps(_extract_steps(text))
    # Biased to KEEP the original plan: we only adopt the critic's version if it does NOT add steps
    # (avoids the 14B "improving" by inflating the plan with micro-steps or invented requirements) and
    # if it was not truncated by MAX_STEPS (a sign of over-splitting).
    if new_steps and new_steps != steps and len(new_steps) <= len(steps) and len(new_steps) <= MAX_STEPS:
        print(color(f"  [critique] plan self-critique -> adjusted to {len(new_steps)} step(s)", "gray"))
        emit("self_critique", phase="plan", steps=new_steps)
        return new_steps
    return steps


OUTPUT_CRITIC_PROMPT = (
    "You are a demanding, BRIEF reviewer. I give you what the user asked for (TASK) and the agent's "
    "RESPONSE/result. Does it REALLY and fully solve what was asked? If YES, respond only 'OK'. If "
    "something CONCRETE and checkable is missing, respond 'MISSING: <what is missing>' in ONE "
    "sentence. Do not invent extra requirements or demand perfection: only what the task really asks for.")


def critique_output(client, model, task, response):
    """6C: second opinion on the OUTPUT before accepting it. Returns '' if it is OK, or a CONCRETE
    gap (string) if something is missing. On failure/doubt -> '' (does not block, to avoid inventing
    complaints)."""
    try:
        r = client.chat.completions.create(model=model, max_tokens=80, messages=[
            {"role": "system", "content": OUTPUT_CRITIC_PROMPT},
            {"role": "user", "content": f"TASK: {task}\nRESPONSE: {_clip_short(response)}"}])
        out = (r.choices[0].message.content or "").strip() if r.choices else ""
    except Exception:  # noqa: BLE001
        return ""
    clean = out.lstrip("*-_\"'`# \t").upper()  # tolerate markdown/quotes/14B preamble
    return out[:300] if clean.startswith("MISSING") else ""


# --- Error lessons (learn so as not to repeat) - 4E style, BOUNDED ---
LESSONS_FILENAME = ".lessons.md"     # hidden, in the working folder
MAX_LESSONS_CHARS = 1200             # HARD cap: a short, curated list, NEVER a log

LESSON_PROMPT = (
    "You are in charge of an agent's LESSONS (errors NOT to repeat), very brief. Given the PREVIOUS "
    "lessons and what just went WRONG in a task, return the UPDATED list in Markdown: MAX 5 "
    "one-line bullets, concrete and actionable (e.g. '- Before editing, locate the real file with "
    "search_in_code; do not invent files'). MERGE and SUMMARIZE; drop what is redundant; NEVER "
    "accumulate history. Respond ONLY with the bullets.")


def _lessons_path():
    return CHATDIR / LESSONS_FILENAME   # Level 14: PER-CHAT lessons (CHATDIR = WORKDIR if no chat)


def _read_lessons():
    p = _lessons_path()
    if not p.exists():
        return ""
    try:
        return p.read_text(encoding="utf-8", errors="replace").strip()[:MAX_LESSONS_CHARS]
    except (OSError, UnicodeError):
        return ""


def record_lesson(client, model, task, result):
    """6C: after a failure/derailment, save a concise LESSON (merged with prior ones, bounded).
    Best-effort: if something fails, nothing is toppled."""
    previous = _read_lessons()
    user = (f"Previous lessons:\n{previous or '(none)'}\n\n"
            f"This task just went WRONG: {_clip_short(task, 400)}\n"
            f"Result: {_clip_short(result, 600)}\n\nReturn the updated lessons.")
    try:
        r = client.chat.completions.create(model=model, max_tokens=200, messages=[
            {"role": "system", "content": LESSON_PROMPT},
            {"role": "user", "content": user}])
        new_lessons = (r.choices[0].message.content or "").strip() if r.choices else ""
    except Exception:  # noqa: BLE001: learning is best-effort, never topples the task
        return
    if not new_lessons:
        return
    if len(new_lessons) > MAX_LESSONS_CHARS:
        new_lessons = new_lessons[:MAX_LESSONS_CHARS].rstrip() + "\n...[clipped]"
    try:  # atomic write
        tmp = _lessons_path().with_suffix(".md.tmp")
        tmp.write_text(new_lessons, encoding="utf-8")
        os.replace(tmp, _lessons_path())
        print(color("  [lesson] lesson recorded (so the error is not repeated)", "gray"))
        emit("lesson", text=new_lessons)
    except OSError:
        pass


# --- 16A: is the task a clear WEB SEARCH? (search verb + web target) ---
# Reason: the chat's text-only shortcuts (docs 8B -> skills 9A) run BEFORE the triage and answer
# WITHOUT tools. A request like 'find me papers on X' matched by TF-IDF the `academic-summary` skill,
# which hijacked it and (unable to search) refused. This DETERMINISTIC, NARROW guard sends it to the
# tool path (QUERY). Deliberately conservative: it requires a SEARCH VERB (search/find/research/gather)
# + a web TARGET (paper/arxiv/news), so 'summarize this paper: ...' (no search verb) is not affected
# and the summarize skill stays the same. Reversible: with WEB_SEARCH=0 the guard is off and everything
# reverts to the previous behavior. The search verb is ANCHORED to the start of the message (like
# _RX_CREATE): 'the study seeks to prove...' mid-sentence (normal speech) is not a search order. It
# allows opening courtesies ('please, ...').
_RX_SEARCH_VERB = re.compile(
    r"^\s*(?:please,?\s+)?"
    r"(search|find|look\s+up|research|investigate|gather)\b", re.IGNORECASE)
# UNAMBIGUOUSLY web/academic target. Deliberately WITHOUT loose polysemous words ('source/link/article'):
# they collide with code and local documents ('the source of the error in payments.py', 'article 5 of
# contract.pdf') and with normal speech -> they gave false positives that hijacked the chat.
_RX_WEB_TARGET = re.compile(
    r"\b(paper|papers|preprint|preprints|arxiv|internet|online|news|url|web)\b", re.IGNORECASE)
# Mention of a LOCAL FILE (something.py/.pdf/.html...): the request is about a local file, not the web.
_RX_LOCAL_FILE = re.compile(
    r"\b[\w./-]+\.(?:py|js|ts|jsx|tsx|html?|css|pdf|txt|md|docx?|xlsx?|pptx?|csv|json|ya?ml|ipynb)\b",
    re.IGNORECASE)
# Intent to CREATE an artifact (NOT anchored, unlike _RX_CREATE): 'search papers ... and GENERATE a
# PDF REPORT' is a TASK that must go through the plan->execute->VERIFY pipeline, not the direct path.
_RX_CREATE_ARTIFACT = re.compile(
    r"\b(generate|create|write|draft|make|export|save|convert|build)\b"
    r"[^.]*\b(report|document|pdf|file|script|notebook|markdown|docx?|xlsx?|pptx?|"
    r"csv|excel|word|presentation|slides?|deliverable)\b", re.IGNORECASE)


def _is_web_search(task):
    """True ONLY if the task is UNAMBIGUOUSLY an internet search AND web search is ENABLED (WEB_SEARCH
    != 0). Narrow and conservative: a search verb AT THE START + a clear web target, and NO local signal
    (own documents, a named file, or creating an artifact). When in doubt -> False and the usual routing
    wins (RAG/skills/triage/pipeline). With WEB_SEARCH=0 -> always False (inert guard -> previous behavior
    intact). Note: it is called AFTER the RAG (8B), so the RAG already had priority."""
    try:
        from web_tools import _search_enabled
        if not _search_enabled():
            return False
    except Exception:  # noqa: BLE001: on any problem, do NOT change the usual routing
        return False
    t = task or ""
    if not (_RX_SEARCH_VERB.search(t) and _RX_WEB_TARGET.search(t)):
        return False
    # LOCAL signals that OVERRIDE the web (let the usual deterministic path win):
    if _is_document_question(t):   # 'my notes', 'the papers I uploaded' -> local RAG (8B)
        return False
    if _RX_LOCAL_FILE.search(t):   # '...in payments.py', '...of contract.pdf' -> local file
        return False
    if _RX_CREATE_ARTIFACT.search(t):   # '...and generate a PDF report' -> TASK (pipeline that verifies)
        return False
    return True


def resolve_task(router, cfg, task, confirm_bash=False, fast=False, use_memory=True,
                 prior_context=None, forced_skill=None, shared_memory=False,
                 direct=False):
    """Resolve a task capably but lightly and return the final answer.

    `router` = build_router(cfg): gives the (client, model, label) of each role.
    The EXECUTOR always runs local; PLANNER and VERIFIER use the STRONG brain if configured
    (Phase 4C), or the local one otherwise.

    LIGHT CONTEXT: each task starts with a NEW Executor conversation (just the system prompt).
    Within the task, the plan steps and the correction loop share that conversation (so one step
    sees what the previous did); BETWEEN tasks nothing accumulates: what persists is the workspace
    on disk, not the chat (the real persistent memory is Phase 4E). The planner and verifier use,
    in addition, their own fresh conversations.

    fast=True skips plan and verification -> behavior ~ Level 3 (faster; useful for trivial tasks or
    to compare speed). In fast mode the executor (local) is used, never the cloud.
    """
    ex_cli, ex_mod, ex_lbl = router["executor"]
    pl_cli, pl_mod, pl_lbl = router["planner"]
    vf_cli, vf_mod, vf_lbl = router["verifier"]

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]  # fresh context per task
    # LIGHT conversation context (Level 7): a summary + the last N project messages (NEVER the whole
    # history). It goes right after the system prompt, as a continuation of the chat.
    if prior_context:
        messages.extend(prior_context)

    # --- Fast path: pure Level 3 Executor (always local) ---
    if fast:
        messages.append({"role": "user", "content": task})
        return run_agent(ex_cli, ex_mod, messages, confirm_bash)

    emit("brains", planner=pl_lbl, executor=ex_lbl, verifier=vf_lbl)  # brain panel (5A)

    # L15-t2: `direct=True` (used by WORKER MODE) SKIPS the chat shortcuts below (docs -> skills ->
    # triage): the worker runs TASKS by definition and those shortcuts answer with TEXT (no tools,
    # no files) -> they hijacked its tasks ('write X to a file' -> chat; long specs -> skill). The
    # CHAT does not change: by default direct=False and everything stays identical.
    if not direct:
        # --- 8B (root): question about YOUR documents -> answer from the knowledge index,
        # citing the source, WITHOUT depending on triage or the model's tool-calling (which fail). ---
        # It goes FIRST (before the 16A web guard): a question about the user's own documents
        # ('search my notes...', 'the papers I uploaded') should be resolved by the deterministic RAG,
        # not the web.
        r_docs = answer_from_documents(ex_cli, ex_mod, task, prior_context)
        if r_docs is not None:
            return r_docs

        # --- 16A: a clear WEB SEARCH (and NOT about your own docs: the RAG already had its turn above)
        # goes STRAIGHT to the tool path, before a text-only SKILL shortcut (9A) steals it; that skill
        # answers WITHOUT tools and cannot search (it hijacked 'find me papers on X' -> refused). ---
        if _is_web_search(task):
            print(color("  [web] web search -> direct tools (so a text skill does not hijack it)", "gray"))
            emit("intent", mode="query")
            messages.append({"role": "user", "content": task})
            return run_agent(ex_cli, ex_mod, messages, confirm_bash)

        # --- 9A (root): does a SKILL match? -> inject its instructions (+ deterministic script),
        # DETERMINISTICALLY (threshold selection: zero false triggers), or the one FORCED by hand (9C).
        # Light context: this turn only. ---
        r_skill = answer_with_skill(ex_cli, ex_mod, cfg, task, prior_context, force=forced_skill)
        if r_skill is not None:
            return r_skill

        # --- Intent TRIAGE (5D): CHAT or TASK? ---
        # A question/chat is NOT a task: it is answered in streaming, WITHOUT tools, no
        # plan/verification and WITHOUT touching disk or project memory. Only the full pipeline
        # (plan->execute->verify->memory) is reserved for what really requires acting/looking up.
        intent = _triage(ex_cli, ex_mod, task)
        if intent == "CHAT":
            print(color("  [chat] chat -> direct reply (no plan/verification, no disk access)", "gray"))
            emit("intent", mode="chat")
            return chat(ex_cli, ex_mod, task, prior_context=prior_context)
        if intent == "QUERY":
            # TEXT deliverable (search/read/look up and answer): uses tools but does NOT create an
            # artifact to verify. It runs and is DELIVERED directly, like chat but with tools -> it
            # does NOT go through file verification (which for text is inconclusive and previously
            # discarded the work). No plan, no verification, no project memory.
            print(color("  [query] query -> run tools and answer directly (text deliverable)", "gray"))
            emit("intent", mode="query")
            messages.append({"role": "user", "content": task})
            return run_agent(ex_cli, ex_mod, messages, confirm_bash)

    # --- Metacognition (8C): does the TASK look HARD? warn BEFORE (does not block) ---
    if cfg.get("METACOGNITION", "true").strip().lower() == "true":
        level, reason = estimate_difficulty(cfg, task, pl_cli, pl_mod)
        if level == "hard":
            suggestion = ("I can escalate it to the powerful brain (STRONG) you have configured"
                          if pl_cli is not ex_cli else "it may be worth splitting it into smaller subtasks")
            notice = f"this looks HARD{(': ' + reason) if reason else ''}. I'll try anyway; {suggestion}."
            print(color("  [meta] " + notice, "yellow"))
            emit("difficulty", level=level, reason=reason, notice=notice)

    # --- Project memory (4E): load the persistent summary AT STARTUP ---
    # A small (bounded) block is injected ONCE into the executor context, to resume the project
    # without re-explaining it. It does not inflate the other calls (planner/verifier start fresh
    # with no memory).
    mem_relevant = True  # does the project memory apply to THIS task? (anti-contamination gate)
    if use_memory:
        memory = _read_memory()
        if memory:
            # Relevance gate: only inject the memory if the task CONTINUES the project. A NEW,
            # unrelated task must NOT inherit the topic/pending items of the previous one.
            mem_relevant = _memory_relevant(ex_cli, ex_mod, task, memory)
            if mem_relevant:
                messages.append({"role": "system", "content":
                    "PROJECT MEMORY (persistent summary; use it to resume the work, do NOT repeat it "
                    "verbatim):\n" + memory})
                print(color(f"  [memory] project memory loaded ({len(memory)} characters)", "gray"))
                emit("memory", text=memory)  # web memory panel (5A)
            else:
                print(color("  [memory] project memory IGNORED (new task unrelated to the project)", "gray"))

    # --- Self-critique and learning (6C): toggles (disableable to measure) ---
    self_critique = cfg.get("SELF_CRITIQUE", "true").strip().lower() == "true"
    learn = cfg.get("LEARN_FROM_ERRORS", "true").strip().lower() == "true"
    # Load LESSONS from past errors (6C): a SMALL, bounded block so as not to repeat them.
    if learn:
        lessons = _read_lessons()
        if lessons:
            messages.append({"role": "system", "content":
                "HINTS from past slip-ups (GUIDANCE, not prohibitions; they may be from ANOTHER "
                "task: ignore them if they don't apply to the current one, and do what the task "
                "asks):\n" + lessons})
            print(color(f"  [lessons] lessons loaded ({len(lessons)} characters)", "gray"))
            emit("lessons", text=lessons)

    # Logging of which brain each role uses (transparency + watching the free tier).
    # (the 'brains' event for the web was already emitted before the triage, above)
    print(color(f"  brains -> planner:{pl_lbl} . executor:{ex_lbl} . verifier:{vf_lbl}", "gray"))

    # --- 1) PLAN (4B) + plan SELF-CRITIQUE (6C) ---
    steps = plan_task(pl_cli, pl_mod, task)
    if self_critique:
        steps = critique_plan(pl_cli, pl_mod, task, steps)  # cheap second opinion on the plan
    multistep = len(steps) > 1
    emit("plan", steps=steps)  # web "live plan" panel (5A)
    if multistep:
        print(color(f"Plan ({len(steps)} steps):", "magenta"))
        for i, p in enumerate(steps, 1):
            print(color(f"   {i}. {p}", "gray"))

    # --- 2) EXECUTE each step (L3 Executor, shared conversation within the task) ---
    # Snapshot BEFORE running: the diff against it gives the real ARTIFACTS (which files the task
    # created/modified) -> the verifier checks THOSE, without re-guessing the name from the wording.
    snap_pre_exec = _snapshot_workspace()
    response = ""
    for i, step in enumerate(steps, 1):
        emit("step", i=i, n=len(steps), desc=step)  # current step on the web (5A)
        if multistep:
            print(color(f"Step {i}/{len(steps)}: {step}", "magenta"))
            instruction = f"Step {i} of {len(steps)} of the task '{task}': {step}"
        else:
            instruction = task
        messages.append({"role": "user", "content": instruction})
        response = run_agent(ex_cli, ex_mod, messages, confirm_bash)

    # --- 3) VERIFY and CORRECT (4A) ---
    # Tri-state verdict: True=OK, False=FAILURE (fix it), None=inconclusive (the verifier did NOT
    # finish -> not taken as good; retried within the cap). min(10, ...): a ceiling in addition to
    # the floor, so a typo in config.env (e.g. "30") does not turn the correction loop into a very
    # long wait (each attempt is a full verification).
    max_attempts = min(10, max(1, _to_int(cfg.get("MAX_VERIFY_ATTEMPTS"), 3)))
    prior_complaints = []  # FAILUREs already detected, so the verifier re-checks them
    verified = False
    derailed = False         # 6B: do we stop for NOT progressing (instead of spiraling and inventing)?
    no_progress = 0          # 6B: consecutive corrections that do NOT improve the existing code
    inconclusive_count = 0   # verifications in a row with NO clear verdict (None) -> cut earlier
    self_critique_done = False  # 6C: the 2nd opinion on the output happens ONCE (no spirals)
    had_real_failure = False    # 6C: did the VERIFIER itself give FAILURE? (only that yields a lesson)
    for attempt in range(1, max_attempts + 1):
        print(color(f"Verifying work (attempt {attempt}/{max_attempts}) [{vf_lbl}]...", "magenta"))
        emit("verifying", attempt=attempt, max=max_attempts)  # verification panel (5A)
        # REAL artifacts (workspace diff): recomputed each attempt so that, after a correction, the
        # verifier keeps seeing the files that ACTUALLY exist, and with the CONTENT (notebook cell
        # delta) so it judges by the real thing, not a summary or the raw JSON.
        artifacts = _created_artifacts(snap_pre_exec)
        contents = _artifact_contents(snap_pre_exec)
        ok, detail = verify(vf_cli, vf_mod, task, confirm_bash, prior_complaints, artifacts, contents)
        if ok is False:
            had_real_failure = True  # a FAILURE from the verifier itself (not from the self-critique)
        # OUTPUT SELF-CRITIQUE (6C): when the verifier gives OK, a cheap 2nd opinion (does it really
        # solve what was asked?), ONCE only. It only downgrades to FAILURE if there are NO green tests
        # already backing the deliverable (if the tests pass, the critique is only INFORMATIONAL: we
        # don't ruin something verified-and-green over a 14B nitpick). Bounded by max_attempts + 6B.
        if ok is True and self_critique and not self_critique_done:
            self_critique_done = True
            gap = critique_output(vf_cli, vf_mod, task, response)
            if gap and not _tests_pass():
                print(color(f"  [critique] output self-critique -> {gap}", "yellow"))
                ok, detail = False, gap
            elif gap:
                print(color(f"  [critique] self-critique (informational, tests already pass): {gap}", "gray"))
        if ok is True:
            print(color("Verification OK", "green"))
            emit("verdict", status="ok")
            verified = True
            break
        if ok is None:
            # Inconclusive: neither success nor a concrete failure -> retry the check, without inventing
            # a bug for the executor. But if it repeats, do NOT insist blindly: stop honestly.
            print(color(f"Verification inconclusive: {detail}", "yellow"))
            emit("verdict", status="inconclusive", detail=detail)
            inconclusive_count += 1
            # Cap on BLIND retries of inconclusive ones (the 14B verifier is often inconclusive on the
            # 1st attempt and right afterwards, so with the default=3 we do NOT cut; this only reins in
            # a very high MAX_VERIFY_ATTEMPTS misconfigured in config.env).
            if inconclusive_count >= 3:
                print(color("  ! the verifier is inconclusive three times in a row; I stop insisting", "yellow"))
                break
            continue
        inconclusive_count = 0  # ok is False -> concrete FAILURE: breaks the inconclusive streak
        # ok is False -> concrete FAILURE: fix WITH the anti-regression safeguard.
        print(color(f"Verification: FAILURE -> {detail}", "red"))
        emit("verdict", status="failure", detail=detail)
        repeated = detail in prior_complaints
        if not repeated:
            prior_complaints.append(detail)  # remember it to re-verify on the next attempt
        snap = _snapshot_workspace()
        was_working = _tests_pass()  # did the deliverable already pass its tests BEFORE touching it?
        # SMART RECOVERY (6B): if the complaint REPEATS or the previous attempt changed nothing, we
        # don't retry blindly: a DIRECTIVE prompt (re-examine reality, don't invent, and be honest if
        # it can't be done).
        if repeated or no_progress >= 1:
            instruction = (
                f"The previous attempt did NOT solve this problem (or changed nothing):\n{detail}\n"
                f"Do NOT repeat the same thing or invent new files. RE-EXAMINE with search_in_code "
                f"and list_dir which files and functions ACTUALLY exist, and fix ONLY on them. "
                f"If you cannot fix it, SAY SO honestly instead of going in circles.")
        else:
            instruction = (
                f"An independent verifier checked the work and found this problem:\n"
                f"{detail}\nFix it IN PLACE using the tools; do NOT rename or delete files that "
                f"already work, and do NOT invent new generic files.")
        messages.append({"role": "user", "content": instruction})
        response = run_agent(ex_cli, ex_mod, messages, confirm_bash)
        snap_post = _snapshot_workspace()  # state AFTER the fix (BEFORE reverting)
        # Safeguard: if it previously passed its tests and the fix broke them or deleted/renamed
        # files, revert (a working deliverable is not ruined by "fixing" it).
        reverted = False
        if was_working:
            lost_files = any(rel not in snap_post for rel in snap)
            if lost_files or not _tests_pass():
                _restore_workspace(snap)
                reverted = True
                reason = "deleted/renamed files" if lost_files else "broke passing tests"
                print(color(f"  x the fix {reason}; reverted (keeping what already worked)", "yellow"))
                emit("notice", text=f"fix reverted: {reason} (previous state kept)")
                # Disk went back to the previous state; 'response' described the undone fix. We
                # reconcile it with what actually remained, so the final summary and the MEMORY (4E)
                # do not claim work that does NOT exist on disk.
                response = (f"(the last fix was reverted because it {reason}; the state on disk is "
                            f"the previous one, which already worked)")
        # DERAILMENT (6B): did the fix IMPROVE the code? It counts as progress only if the agent
        # EDITED a file that ALREADY existed and it was not reverted. (Creating only NEW files while
        # the tests still fail, narrating, or having its change reverted is NOT progress: it's the
        # signature of derailment.) If two corrections in a row make no progress, we stop HONESTLY
        # instead of spiraling and leaving junk.
        edited_existing = any(rel in snap_post and snap_post[rel] != snap[rel] for rel in snap)
        no_progress = 0 if (edited_existing and not reverted) else no_progress + 1
        if no_progress >= 2:
            derailed = True
            print(color("  x no progress (does not improve the code / repeats the failure / tries "
                        "to invent): stopping honestly to avoid derailing.", "yellow"))
            emit("notice", text="no progress: honest stop (6B)")
            break

    # Is the executor's response a real deliverable (not something degenerate)?
    _degenerate = (not response or len(response.strip()) < 15
                   or "repetition loop" in response or "iteration limit" in response
                   or response.strip() == "(empty response)")
    if verified:
        response = _final_summary(response, steps, multistep, True)
    elif not derailed and not had_real_failure and not _degenerate:
        # The verifier NEVER gave FAILURE (only inconclusive): a typical pattern for TEXT deliverables
        # (a summary, an explanation, a search) that leave NO artifact to check with tools, and for
        # which the 14B verifier redoes the task or makes excuses instead of giving a VERDICT. The
        # executor's work is done -> we DELIVER it, without discarding it. We do NOT mark "Verification
        # OK" (not the false OK of H3): this is not turning an inconclusive verdict into a false failure.
        print(color("  [info] delivering the executor's work (verification inconclusive, no real failure)", "gray"))
        response = _final_summary(response, steps, multistep, False)
    else:
        # HONEST STOP (6B): there was a real FAILURE, derailment, or the deliverable is degenerate.
        # Neither fake success nor invent; acknowledge the real work (if the tests pass, say so).
        print(color("Stopped without completing the task; giving an honest result." if derailed
                    else "Ran out of verification attempts; review the result.", "yellow"))
        reason = ("I stopped to avoid derailing and inventing files" if derailed
                  else "the verification attempts ran out")
        pending = (prior_complaints[-1] if prior_complaints else "it was not clear what failed")[:500]
        if _tests_pass():  # tests green (project state; I do NOT claim it's due to my change)
            response = (f"The project tests PASS, but I did not fully close verification of the task "
                        f"({reason}). Check: {pending}")
        else:
            response = (f"I could not fully complete the task ({reason}). "
                        f"What was left pending: {pending}")
        # LEARN FROM THE ERROR (6C): only if there was a REAL FAILURE from the verifier (not a mere
        # self-critique nitpick) -> save a concise lesson, so we don't record FALSE lessons about work
        # that was actually fine.
        if learn and had_real_failure:
            record_lesson(ex_cli, ex_mod, task, response)

    # --- Project memory (4E): update AT THE END (a focused, separate call) ---
    if use_memory:
        # Merge only if the task was for the SAME project; if it was NEW, replace (don't drag along).
        # Level 14: with SHARED memory across chats (shared_memory=True) it is NEVER replaced (always
        # merges), so one chat's task does not erase another's project memory.
        _update_memory(ex_cli, ex_mod, task, response,
                       merge=(mem_relevant or shared_memory))
    return response


def _set_workdir(path, chat_dir=None):
    """Point WORKDIR (of the agent and this module) at the active PROJECT folder (documents +
    .knowledge + .memory.md, shared) and CHATDIR at the active CHAT folder (conversation + chat
    memory + lessons + snapshots). `chat_dir=None` -> CHATDIR = WORKDIR (no chat, as before)."""
    _agent.WORKDIR = path
    globals()["WORKDIR"] = path
    globals()["CHATDIR"] = Path(chat_dir) if chat_dir is not None else path


def resolve_in_project(router, cfg, name, task, confirm_bash=False, fast=False,
                       forced_skill=None, chat=None, direct=False):
    """Level 7A + 14: resolve a task INSIDE a PROJECT (documents + .knowledge + project memory
    `.memory.md`, SHARED -> WORKDIR) and a specific CHAT (its conversation + chat summary/memory +
    lessons + snapshots -> CHATDIR). Migrates the project to the project->chats model lazily and
    safely (with backup) if it was still on the old model. LIGHT per-chat context (summary + last N).
    Saves the turn and re-summarizes on close. Thin layer over resolve_task."""
    proj = projects.create_project(cfg, name)      # idempotent (creates the folder if new)
    projects.set_active(cfg, name)
    base = projects.base_dir(cfg)
    chats.ensure_chat(proj, base=base)             # Level 14: migrate (lazy, with backup) + guarantee a chat
    if (chat or "").strip():                       # chat REQUESTED explicitly...
        chat_id = chats._slug(chat)
        if chat_id not in chats.list_chats(proj):
            chat_id = chats.create_chat(proj, chat_id)   # ...if it doesn't exist, it is CREATED (like --project creates the project)
    else:
        chat_id = chats.active_chat(proj)          # no chat -> the active one (guaranteed by ensure_chat)
    chats.set_active_chat(proj, chat_id)
    chat_dir = chats.chat_path(proj, chat_id)
    _set_workdir(proj, chat_dir)
    emit("project", name=proj.name, chat=chat_id)   # project + chat panel (web)
    # Time machine (8A): SNAPSHOT of the project DOCUMENTS before the task; the snapshot is stored in
    # the CHAT (store=chat_dir). Best-effort: a snapshot failure must never topple the turn.
    sid = None
    try:
        sid = time_machine.create_snapshot(proj, task=task, store=chat_dir)
    except Exception:  # noqa: BLE001
        sid = None
    ctx = projects.conversation_context(chat_dir, projects.last_n(cfg))  # summary + last N of the CHAT
    # shared_memory=True: the PROJECT memory (.memory.md) is SHARED by all chats -> it is never
    # REPLACED by an "unrelated" task from one chat (which would let a chat erase another's context);
    # it is always MERGED. The relevance gate still decides whether it is INJECTED.
    response = resolve_task(router, cfg, task, confirm_bash=confirm_bash, fast=fast,
                            use_memory=True, prior_context=ctx,
                            forced_skill=forced_skill, shared_memory=True,
                            direct=direct)
    if sid:  # RECORD the task diff (scopes the undo to what IT did). If it touched nothing,
        try:  # we discard the snapshot (no-op); if it touched, we keep it and offer undo.
            if time_machine.record_changes(proj, sid, store=chat_dir):
                m = time_machine.read_meta(chat_dir, sid)
                emit("snapshot", id=sid, task=(task or "")[:120], n=m.get("n_files", 0))
            else:
                time_machine.delete_snapshot(chat_dir, sid)
        except Exception:  # noqa: BLE001,S110: the snapshot is a convenience; if it fails, the turn goes on (best-effort)  # nosec B110
            pass
    try:  # persist the WHOLE turn (nothing is lost) + incremental re-summarize (light context) IN THE CHAT
        projects.save_turn(chat_dir, task, response)
        ex_cli, ex_mod, _ = router["executor"]
        projects.update_summary(ex_cli, ex_mod, chat_dir, task, response)
    except OSError:
        pass
    return response


# --- Time machine (8A): undo, with a preview to confirm ---
def _chat_dir(cfg, name, chat=None):
    """Folder of the CHAT (active or the given one) of a project, where its snapshots/conversation
    live. If the project is NOT yet migrated (no `chats/`), returns the project itself (compat:
    legacy snapshots at the project level)."""
    proj = projects.project_path(cfg, name)
    cid = chats._slug(chat) if (chat or "").strip() else chats.active_chat(proj)
    return chats.chat_path(proj, cid) if cid else proj


def preview_undo(cfg, name, chat=None):
    """What the 'undo last task' of the active CHAT would revert (without applying it). None if there
    is no snapshot."""
    proj = projects.project_path(cfg, name)
    store = _chat_dir(cfg, name, chat)
    sid = time_machine.last_snapshot(store)
    if not sid:
        return None
    return {"id": sid, "meta": time_machine.read_meta(store, sid),
            "plan": time_machine.plan_undo(proj, sid, store=store)}


def apply_undo(cfg, name, sid=None, chat=None):
    """Undo the last task (or snapshot `sid`) of the active CHAT and CONSUME that snapshot (so the
    next 'undo' goes to the previous one -> multi-level time machine). Returns the applied plan or
    None."""
    proj = projects.project_path(cfg, name)
    store = _chat_dir(cfg, name, chat)
    if sid is None:
        sid = time_machine.last_snapshot(store)
    if not sid:
        return None
    plan = time_machine.undo(proj, sid, store=store)
    if plan is not None:
        time_machine.delete_snapshot(store, sid)
    return plan


def _to_int(value, default):
    """Safely convert a config string to int."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


# --- 6) PROJECT MEMORY (Phase 4E): a compact summary, NEVER an endless log ---
# The memory is a CURATED, SHORT summary of the project in the working folder. It is read ONCE at
# task startup (a small block is injected into the executor context) and updated at the end with a
# SEPARATE, focused call. It is BOUNDED to MAX_MEMORY_CHARS and RE-SUMMARIZED on each update, so it
# does not grow without end -> it does not sink the system's light-context advantage.
MEMORY_FILENAME = ".memory.md"   # hidden, in the working folder (unobtrusive)
MAX_MEMORY_CHARS = 2000          # HARD cap: memory is a summary, not a history

MEMORY_PROMPT = (
    "You are in charge of a project's MEMORY: a COMPACT, curated summary, NOT a log. Given the "
    "PREVIOUS memory and what was just done, return the UPDATED memory in Markdown, VERY BRIEF "
    "(max ~180 words), with these sections if they apply: '## Project' (what it is), "
    "'## Decisions', '## State', '## Files', '## Pending'. MERGE and SUMMARIZE: drop what is "
    "obsolete or redundant; NEVER accumulate history or repeat. Respond ONLY with the memory."
)


def _memory_path():
    return WORKDIR / MEMORY_FILENAME


def _read_memory():
    """Return the memory summary (bounded to MAX_MEMORY_CHARS), or '' if there is none."""
    path = _memory_path()
    if not path.exists():
        return ""
    try:
        # errors="replace": a corrupt/non-UTF-8 .memory.md degrades to readable text instead of
        # raising UnicodeDecodeError (which is NOT OSError) and toppling the task at startup.
        return path.read_text(encoding="utf-8", errors="replace").strip()[:MAX_MEMORY_CHARS]
    except (OSError, UnicodeError):
        return ""


RELEVANCE_PROMPT = (
    "Decide whether the user's TASK CONTINUES the project described in the MEMORY, or is a NEW task "
    "unrelated to that project (a different topic). Focus on the TOPIC, not minor details. Respond "
    "with ONE word only: RELATED or NEW."
)


def _memory_relevant(client, model, task, memory):
    """Does the task CONTINUE the memory's project, or is it a NEW unrelated task? Prevents the
    context (topic, pending items) of a previous task from contaminating a following, unrelated one,
    without losing the legitimate persistence of a coherent project. On doubt or error, RELATED
    (previous behavior). A SHORT, cheap call, by the local brain."""
    try:
        r = client.chat.completions.create(model=model, max_tokens=4, messages=[
            {"role": "system", "content": RELEVANCE_PROMPT},
            {"role": "user", "content": f"MEMORY (project):\n{(memory or '')[:1000]}\n\nTASK: {(task or '')[:300]}"}])
        out = (r.choices[0].message.content or "").strip().upper() if r.choices else ""
    except Exception:  # noqa: BLE001: best-effort: on failure, behave as before
        return True
    return not out.lstrip("*-_\"'` \t").startswith("NEW")


def _update_memory(client, model, task, result, merge=True):
    """Update .memory.md with a FOCUSED, SEPARATE call (does not inflate the other calls' context):
    merges the previous memory with what was just done and produces a COMPACT summary, which is also
    clipped to MAX_MEMORY_CHARS. If something fails, it does NOT topple the task (memory is
    best-effort). `merge=False` (a NEW unrelated task) -> does NOT drag along the previous memory: it
    replaces it with a summary of this task only (no contamination)."""
    previous = _read_memory() if merge else ""
    # Bound the inputs: the PREVIOUS memory already comes bounded, but 'result' is the executor's final
    # response, which has NO cap (unlike tool outputs). Without clipping, a huge response would inflate
    # THIS call (the bloat 4E wants to avoid). We reuse the L3 cap.
    def _clip(text):
        text = text or ""
        return text if len(text) <= MAX_OUTPUT_CHARS else text[:MAX_OUTPUT_CHARS] + "\n...[truncated]"
    user = (f"Previous memory:\n{previous or '(empty)'}\n\n"
            f"This task was just WORKED ON: {_clip(task)}\nResult (it may be a success or an honest "
            f"stop, go by the text): {_clip(result)}\n\n"
            "Return the UPDATED, compact project memory.")
    try:
        resp = client.chat.completions.create(model=model, max_tokens=400, messages=[
            {"role": "system", "content": MEMORY_PROMPT},
            {"role": "user", "content": user}])  # COMPACT memory: bounded (anti-uncapped-generation)
        new_memory = (resp.choices[0].message.content or "").strip() if resp.choices else ""
    except Exception as e:  # noqa: BLE001: memory must never topple the task
        print(color(f"  [memory] (could not update the memory: {type(e).__name__})", "gray"))
        return
    if not new_memory:
        return
    if len(new_memory) > MAX_MEMORY_CHARS:  # safety net: clip (it is not a log)
        # The total (marker included) must fit within the cap, so _read_memory does not cut it.
        marker = "\n...[memory clipped]"
        new_memory = new_memory[:MAX_MEMORY_CHARS - len(marker)].rstrip() + marker
    try:
        # ATOMIC write: to a temp file + os.replace, so the good memory is not lost if the process is
        # interrupted mid-write (write_text truncates before writing -> corruption).
        dest = _memory_path()
        tmp = dest.with_suffix(".md.tmp")
        tmp.write_text(new_memory, encoding="utf-8")
        os.replace(tmp, dest)
        print(color(f"  [memory] project memory updated ({len(new_memory)} characters)", "gray"))
        emit("memory_updated", text=new_memory)  # refresh the web memory panel (5A)
    except OSError:
        pass


# --- 5) CLI: REPL and non-interactive mode (same look as Level 3) ---
HELP = (
    "Commands: /exit (quit) . /help (this help)\n"
    "Type a task; the system plans, executes, and verifies its own work.\n"
    "Each task starts with a clean context (what persists is the workspace)."
)


def main():
    ap = argparse.ArgumentParser(description="Capable local agent system (Level 4)")
    ap.add_argument("-p", "--prompt", help="run ONE task and exit (non-interactive)")
    ap.add_argument("--confirm-bash", action="store_true",
                    help="ask for confirmation before running each shell command")
    ap.add_argument("--fast", action="store_true",
                    help="no planning or verification (~ Level 3, faster)")
    ap.add_argument("--local", action="store_true",
                    help="force 100% local (ignore STRONG); to measure the local-only-14B baseline")
    ap.add_argument("--no-memory", action="store_true",
                    help="do not read or update the project memory (.memory.md) in this run")
    ap.add_argument("--no-self-critique", action="store_true",
                    help="disable the Level 6C plan/output self-critique (to measure its effect)")
    ap.add_argument("--no-lessons", action="store_true",
                    help="do not read or write lessons (.lessons.md) from Level 6C in this run")
    # --- Projects (Level 7): opt-in; without these flags, the usual behavior (cwd) ---
    ap.add_argument("--project", metavar="NAME",
                    help="work INSIDE the project NAME (its folder, memory and conversation); creates it if new")
    ap.add_argument("--projects", action="store_true", help="list the projects and the active one, and exit")
    ap.add_argument("--new-project", metavar="NAME", help="create a project (leaves it active) and exit")
    # --- Chats inside projects (Level 14) ---
    ap.add_argument("--chat", metavar="ID",
                    help="work in the CHAT ID of the project (its conversation/memory/snapshots); otherwise the active one")
    ap.add_argument("--chats", action="store_true", help="list the project's chats and the active one, and exit")
    ap.add_argument("--migrate", action="store_true",
                    help="migrate ALL projects to the project->chats model (with backup); idempotent, and exit")
    # --- Time machine (Level 8A) ---
    ap.add_argument("--undo", action="store_true",
                    help="undo the LAST task of the project (restore the files to the previous state)")
    ap.add_argument("--snapshots", action="store_true", help="list the project snapshots and exit")
    ap.add_argument("--yes", action="store_true", help="do not ask for confirmation (for --undo)")
    # --- Knowledge index (Level 8B) ---
    ap.add_argument("--index", action="store_true",
                    help="(re)index the project documents for the knowledge index and exit")
    ap.add_argument("--search-conv", metavar="WORDS",
                    help="search the project's saved conversation and show the matching turns, and exit")
    ap.add_argument("--export-conv", metavar="FILE",
                    help="export the project conversation to a file (.md/.pdf/.docx...) and exit")
    ap.add_argument("--self-audit", action="store_true",
                    help="security self-audit (SAST + invariants); generate the report, narrate it, and exit")
    args = ap.parse_args()

    cfg = load_config()

    if args.self_audit:                             # Level 11A: security self-audit (READ-ONLY)
        import self_audit
        res = self_audit.audit()                    # DETERMINISTIC checks (the script sets the verdict)
        print(self_audit.format_report(res))
        path = self_audit.write_report(res)         # writes the self-audit report file (regenerable)
        print(color(f"\n  [report] report written to {path}", "gray"))
        try:                                        # the model NARRATES (best-effort; does not change the verdict)
            router = build_router(cfg)
            ex_cli, ex_mod, _ = router["executor"]
            print(color("\n───────── security analyst narration ─────────", "magenta"))
            print(self_audit.narrate(ex_cli, ex_mod, res))
        except Exception as e:  # noqa: BLE001: without a model, the deterministic report still stands
            print(color(f"  (narration unavailable: {type(e).__name__}; the deterministic verdict rules)", "gray"))
        return

    # --- Project operations that do NOT need the server (pure disk) -> before the router ---
    if args.projects:
        active = projects.active_project(cfg)
        names = projects.list_projects(cfg)
        for n in names:
            print(("* " if n == active else "  ") + n)
        print(color(f"({len(names)} project(s) in {projects.base_dir(cfg)}; active: {active or 'none'})", "gray"))
        return
    if args.new_project:
        p = projects.create_project(cfg, args.new_project)
        projects.set_active(cfg, args.new_project)
        print(color(f"Project created and active: {p.name}  ({p})", "green"))
        return
    if args.migrate:                                # Level 14: explicit migration of ALL projects
        base = projects.base_dir(cfg)
        done_count = 0
        for n in projects.list_projects(cfg):
            r = chats.migrate_project(projects.project_path(cfg, n), base=base)
            status = "migrated" if r.get("migrated") else r.get("reason", "-")
            print(("* " if r.get("migrated") else "  ") + f"{n}: {status}"
                  + (f"  (backup: {Path(r['backup']).name})" if r.get("backup") else ""))
            done_count += 1 if r.get("migrated") else 0
        print(color(f"({done_count} project(s) migrated; backups in {base / chats.BACKUP_DIR})", "gray"))
        return
    if args.chats:                                  # Level 14: list a project's chats
        name = args.project or projects.active_project(cfg)
        if not name:
            print(color("Specify a project (--project NAME) or have one active.", "red"))
            return
        proj = projects.project_path(cfg, name)
        chats.ensure_chat(proj, base=projects.base_dir(cfg))   # migrate if needed + guarantee a chat
        act = chats.active_chat(proj)
        for c in chats.list_chats(proj):
            print(("* " if c == act else "  ") + c)
        print(color(f"({len(chats.list_chats(proj))} chat(s) in '{name}'; active: {act or 'none'})", "gray"))
        return
    if args.index:
        name = args.project or projects.active_project(cfg)
        if not name:
            print(color("Specify a project (--project NAME) or have one active.", "red"))
            return
        import knowledge
        nd, nc = knowledge.index_docs(projects.project_path(cfg, name), force=True)
        print(color(f"Knowledge index of '{name}': {nd} document(s), {nc} fragment(s).", "green"))
        return
    if args.search_conv or args.export_conv:        # Level 10D: conversation searcher / export
        import conversations
        name = args.project or projects.active_project(cfg)
        if not name:
            print(color("Specify a project (--project NAME) or have one active.", "red"))
            return
        proj = projects.project_path(cfg, name)
        chats.ensure_chat(proj, base=projects.base_dir(cfg))   # migrate if needed
        chat_dir = _chat_dir(cfg, name, args.chat)             # conversation of the active CHAT (or the requested one)
        if args.search_conv:
            print(conversations.format_search(conversations.search(chat_dir, args.search_conv), args.search_conv))
        if args.export_conv:
            print(color(conversations.export(chat_dir, args.export_conv), "green"))
        return
    if args.snapshots or args.undo:
        # Time machine (8A): operate on the given project (--project) or the active one.
        name = args.project or projects.active_project(cfg)
        if not name:
            print(color("Specify a project (--project NAME) or have one active.", "red"))
            return
        proj = projects.project_path(cfg, name)
        chats.ensure_chat(proj, base=projects.base_dir(cfg))   # migrate if needed
        store = _chat_dir(cfg, name, args.chat)                # snapshots of the active CHAT (or the requested one)
        if args.snapshots:
            ss = time_machine.list_snapshots(store)
            for m in ss:
                print(f"  {m['id']}  ({m.get('n_files', 0)} files)  {m.get('task', '')[:60]}")
            print(color(f"({len(ss)} snapshot(s) in '{name}')", "gray"))
            return
        prev = preview_undo(cfg, name, chat=args.chat)         # args.undo
        if not prev:
            print(color(f"Nothing to undo in '{name}' (no snapshots).", "yellow"))
            return
        pl = prev["plan"]
        print(color(f"Undoing the last task of '{name}' will revert:", "yellow"))
        if pl["revert"]:   print(f"  . revert (modified): {pl['revert']}")
        if pl["recreate"]: print(f"  . recreate (deleted): {pl['recreate']}")
        if pl["delete"]:   print(f"  . delete (created by the task): {pl['delete']}")
        if pl["not_restorable"]:
            print(color(f"  . (not byte-exact restorable, too large: {pl['not_restorable']})", "gray"))
        if not any((pl["revert"], pl["recreate"], pl["delete"])):
            print(color("  (no file changes to revert)", "gray"))
        if not args.yes and input(color("  Confirm? [y/N] ", "yellow")).strip().lower() not in ("y", "yes"):
            print("Cancelled.")
            return
        apply_undo(cfg, name, prev["id"], chat=args.chat)
        print(color("Undone: the files were restored to the previous state.", "green"))
        return
    if args.no_self_critique:          # 6C: disable the self-critique only in this run (measurement)
        cfg["SELF_CRITIQUE"] = "false"
    if args.no_lessons:                # 6C: disable learning only in this run
        cfg["LEARN_FROM_ERRORS"] = "false"
    # Per-role router (Phase 4C): executor local; planner/verifier STRONG if there is a free key.
    router = build_router(cfg, force_local=args.local)
    # Project memory (Phase 4E): on by config and disableable per run (--no-memory).
    # --fast never uses memory (fast path ~ L3); we reflect it here so the banner does not lie.
    use_memory = (not args.no_memory
                  and not args.fast
                  and cfg.get("PROJECT_MEMORY", "true").strip().lower() == "true")

    # Level 7: with --project, each turn goes INSIDE the project (its folder/memory/persistent
    # conversation, light context). Without --project, the usual behavior (cwd, no conversation).
    def _resolve(t):
        if args.project:
            return resolve_in_project(router, cfg, args.project, t, args.confirm_bash, args.fast,
                                      chat=args.chat)
        return resolve_task(router, cfg, t, args.confirm_bash, args.fast, use_memory)

    # --- Non-interactive mode: one task and out (useful for tests/scripts) ---
    if args.prompt:
        try:
            print(color(_resolve(args.prompt), "green"))
        except Exception as e:  # noqa: BLE001: script mode: clean error and exit != 0
            sys.exit(color(f"Could not complete the task. Technical detail: {type(e).__name__}: {e}\n"
                           "   What to try: retry it; if it repeats, check the brain is alive "
                           "(the web [health] panel, or start it with ./start_local.sh).", "red"))
        return

    # --- Interactive REPL mode ---
    mode = "fast (L3)" if args.fast else "orchestrated (plan+verify)"
    mem = "memory on" if use_memory else "memory off"
    proj_text = f" . project '{projects._slug(args.project)}'" if args.project else ""
    print(color(f"Agent system (Level 4) . executor {router['executor'][2]} . "
                f"planner/verifier {router['planner'][2]} . mode {mode} . {mem}{proj_text}", "green"))
    print(HELP)
    while True:
        try:
            user = input(color("\n> ", "green")).strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        if not user:
            continue
        if user in ("/exit", "/quit"):
            print("Goodbye!")
            break
        if user == "/help":
            print(HELP)
            continue
        try:  # without --project each task is independent; with --project the conversation is chained
            print(color(_resolve(user), "green"))
        except Exception as e:  # noqa: BLE001: the REPL must never die from one turn
            print(color(f"That turn failed. Technical detail: {type(e).__name__}: {e}\n"
                        "   You can keep typing; if it repeats, check the brain is alive "
                        "(./start_local.sh).", "red"))


if __name__ == "__main__":
    main()
