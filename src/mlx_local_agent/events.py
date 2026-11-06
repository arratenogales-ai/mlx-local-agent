#!/usr/bin/env python3
# events.py: lightweight event hook to observe the agent live (Level 5A).
#
# The goal of Phase 5A is to WATCH the agent in real time (plan -> tool ->
# verification -> result) from a local web page, WITHOUT rewriting its logic.
#
# The solution is a single hook: an `emit(type, **data)` function that the agent
# calls at key points (where it already printed). By default it does nothing
# (no subscriber), so the CLI behaves EXACTLY as before. When the web backend
# wants to listen, it installs an emitter with `use_emitter(callback)` only for
# the duration of that task and receives events as JSON-serializable dicts.
#
# It is thread-local on purpose: the backend runs each agent task in its own
# thread, so each task has its own emitter without clobbering others. This keeps
# the agent logic intact: it only emits where told (or nowhere).
import contextlib
import threading

# Per-thread state: each thread (each backend task) has its own emitter.
_local = threading.local()


def emit(event_type, **data):
    """Emit an event to the current thread's subscriber, if any.

    If nobody is listening (the CLI case), this is a silent no-op and the agent
    is unchanged. `event_type` is a short label ("plan", "tool", "verdict", ...)
    and `data` holds JSON-serializable fields. Never raises: a listener failure
    must not affect the agent (observability is best-effort).
    """
    cb = getattr(_local, "emitter", None)
    if cb is None:
        return
    try:
        cb({"type": event_type, **data})
    except Exception:  # noqa: BLE001,S110 - observing must never take down the task (best-effort, nothing to log)  # nosec B110
        pass


@contextlib.contextmanager
def use_emitter(callback):
    """Install `callback` as the current thread's emitter for the duration of the
    `with` block, restoring the previous one on exit. The backend uses this to
    capture the events of ONE agent run and forward them to the web page."""
    previous = getattr(_local, "emitter", None)
    _local.emitter = callback
    try:
        yield
    finally:
        _local.emitter = previous
