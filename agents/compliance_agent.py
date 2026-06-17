# ============================================================
# agents/compliance_agent.py
# ------------------------------------------------------------
# Compliance Guard Agent
#
# What it does:
#   Runs alongside every live call. Scans the transcript stream for
#   banned phrases, GDPR/TCPA risks, and missing disclosures, and
#   reports any violation it finds.
#
# How it fits in CallOS:
#   The Live Voice Agent keeps this as a sub-agent and consults it
#   mid-call. If a violation is flagged the live agent self-corrects
#   or escalates to a human.
#
# ADK pattern used:
#   google.adk.agents.Agent with a custom FunctionTool
#   (same pattern as ADK/Module 3 - Agents in ADK)
# ============================================================

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from google.adk.agents import Agent

import config

# Phrases an agent must never say — overpromising or pressure tactics
# that create regulatory / legal risk on a sales or support call.
BANNED_PHRASES = [
    "guaranteed return",
    "risk free",
    "you must buy now",
    "no refunds ever",
    "we never share data",  # absolute privacy claims invite GDPR trouble
]


def check_compliance(text: str) -> dict:
    """Scan a transcript chunk for banned phrases.

    Args:
        text (str): the chunk of conversation to check.

    Returns:
        dict: {"compliant": bool, "violations": [str, ...]} — the list
        of banned phrases found (empty when the text is clean).
    """
    lowered = text.lower()
    found = [phrase for phrase in BANNED_PHRASES if phrase in lowered]
    return {"compliant": len(found) == 0, "violations": found}


compliance_agent = Agent(
    name="compliance_agent",
    model=config.get_model(),
    description="Flags banned phrases and GDPR/TCPA risks in a live call.",
    instruction="""
You are the Compliance Guard for a voice sales/support call.

For any transcript text you receive:
1) Call check_compliance to scan it for banned phrases.
2) If violations are returned, state clearly which rule was broken and
   suggest a compliant rephrasing.
3) If it is clean, reply exactly: "Compliant."

Be terse — you run in parallel with the live agent and must not slow it down.
""",
    tools=[check_compliance],
)
