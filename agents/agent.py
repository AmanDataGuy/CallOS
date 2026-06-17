# ============================================================
# agents/agent.py
# ------------------------------------------------------------
# Live Voice Agent (root ADK agent)
#
# What it does:
#   The primary call handler. Greets the caller, answers questions
#   by delegating to the Knowledge Retrieval agent, watches for
#   compliance and sentiment problems via its sub-agents, and ends
#   or escalates the call.
#
# How it fits in CallOS:
#   This is the `root_agent` that `adk web ./agents/` loads and that
#   api/main.py runs for every turn. Sub-agents (compliance,
#   sentiment, kb) coordinate around it; post-call agents run later
#   in the pipeline.
#
# ADK pattern used:
#   google.adk.agents.Agent as a manager with sub_agents + tools
#   (same pattern as ADK/Module 8 - Multi Agent Systems)
#   Human-in-the-loop escalation via hitl.py
#   (ADK/Module 9 - Human in the Loop)
# ============================================================

import asyncio
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from google.adk.agents import Agent

import config
import hitl

# Import the call-time specialists so the live agent can delegate.
# (Post-call agents — lead scorer, churn, topic — run in the pipeline,
#  not as sub-agents of the live call, so they are not imported here.)
from .compliance_agent import compliance_agent
from .sentiment_agent import sentiment_agent
from .kb_agent import kb_agent


def end_call(reason: str) -> dict:
    """End the current call.

    Args:
        reason (str): why the call is ending (e.g. "caller satisfied").

    Returns:
        dict: {"status": "ended", "reason": str}.
    """
    return {"status": "ended", "reason": reason}


async def transfer_to_human(reason: str) -> dict:
    """Escalate the call to a human agent and wait for their response.

    Registers a pending escalation in hitl.py and suspends the agent
    turn until a supervisor POSTs to /escalation/{call_id}/respond or
    the timeout elapses. If no call_id context is set (e.g. adk web),
    returns immediately without blocking.

    Args:
        reason (str): why a human is needed (e.g. "caller is upset").

    Returns:
        dict: one of —
          {"status": "resumed_after_human", "human_response": str, "reason": str}
          {"status": "escalation_timeout", "reason": str}
          {"status": "transferring", "reason": str}  (no call context)
    """
    call_id = hitl.get_call_id()
    if not call_id:
        # Running outside the API (e.g. adk web dev mode) — skip pause.
        return {"status": "transferring", "reason": reason}

    future = hitl.register(call_id)
    print(f"[HITL] Escalation registered for call {call_id}. "
          f"POST /escalation/{call_id}/respond to resume.")
    try:
        human_response = await asyncio.wait_for(
            asyncio.shield(future), timeout=hitl.TIMEOUT
        )
        return {
            "status": "resumed_after_human",
            "human_response": human_response,
            "reason": reason,
        }
    except asyncio.TimeoutError:
        print(f"[HITL] Escalation timed out for call {call_id}")
        return {"status": "escalation_timeout", "reason": reason}


root_agent = Agent(
    name="live_voice_agent",
    model=config.get_model(),
    description="The primary CallOS voice agent that handles live calls.",
    instruction="""
You are CallOS, a professional AI voice agent handling a live phone call.

How to handle a call:
- Greet the caller and find out what they need.
- For product, pricing, or policy questions, delegate to kb_agent and
  answer from what it returns. Never invent product facts.
- Keep replies short and natural — this is spoken, not written.
- If the caller seems angry or frustrated, consult sentiment_agent; if it
  says to escalate, call transfer_to_human and relay the human's response
  back to the caller once you receive it.
- Stay compliant — avoid promises or pressure tactics (compliance_agent
  monitors you in parallel).
- When the caller's need is met, call end_call.
""",
    sub_agents=[kb_agent, compliance_agent, sentiment_agent],
    tools=[end_call, transfer_to_human],
)
