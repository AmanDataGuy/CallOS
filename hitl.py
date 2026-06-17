# ============================================================
# hitl.py
# ------------------------------------------------------------
# Human-in-the-Loop coordination layer
#
# What it does:
#   Suspends an ADK agent turn when escalation is triggered and
#   resumes it once a human supervisor responds, using asyncio
#   Futures and Python contextvars to tie the per-call identity
#   across module boundaries without coupling agent code to the
#   API layer.
#
# How it fits in CallOS:
#   agents/agent.py calls transfer_to_human → register() + await;
#   api/main.py exposes POST /escalation/{call_id}/respond which
#   calls resolve() and unblocks the waiting agent turn.
#
# ADK pattern used:
#   Human-in-the-loop interrupt/resume described in
#   ADK/Module 9 – Human in the Loop, implemented with stdlib
#   asyncio.Future — no extra ADK dependency needed.
# ============================================================

import asyncio
from contextvars import ContextVar

# Per-async-task binding: which call_id is currently executing.
# asyncio.create_task() copies the running context, so sub-tasks
# spawned by ADK automatically inherit the value set by handle_turn.
_call_id_var: ContextVar[str] = ContextVar("call_id", default="")

# Live pending escalations: call_id → unsettled Future.
_PENDING: dict[str, "asyncio.Future[str]"] = {}

# Seconds before an unresolved escalation auto-closes with a timeout.
TIMEOUT = 30


def set_call_id(call_id: str) -> None:
    """Bind call_id to the current async-task context.

    Call this inside handle_turn, before running the agent, so that
    transfer_to_human can read it without receiving it as an argument.
    """
    _call_id_var.set(call_id)


def get_call_id() -> str:
    """Return the call_id bound to the current async context."""
    return _call_id_var.get()


def register(call_id: str) -> "asyncio.Future[str]":
    """Create and park a Future for call_id.

    The agent tool awaits this Future. Resolved by resolve() when the
    human supervisor POSTs their response.
    """
    future: asyncio.Future[str] = asyncio.get_running_loop().create_future()
    _PENDING[call_id] = future
    return future


def resolve(call_id: str, human_response: str) -> bool:
    """Unblock the waiting agent turn with the human's response.

    Returns True if the escalation was live, False if it had already
    timed out or the call_id is unknown.
    """
    future = _PENDING.pop(call_id, None)
    if future is None or future.done():
        return False
    future.set_result(human_response)
    return True


def list_pending() -> list[str]:
    """Return all call IDs currently awaiting a human response."""
    return list(_PENDING.keys())
