# ============================================================
# agents/sentiment_agent.py
# ------------------------------------------------------------
# Sentiment Detector Agent
#
# What it does:
#   Watches the caller's words for anger / frustration. When the
#   negative signal crosses a threshold it recommends escalating
#   the call to a human.
#
# How it fits in CallOS:
#   A sub-agent of the Live Voice Agent. If it returns escalate=True,
#   the live agent hands off via transfer_to_human.
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

# Words that signal a frustrated/angry caller. Escalate once the count
# of distinct hits reaches the threshold — one stray word is not enough,
# but a cluster of them means the caller is genuinely upset.
ANGER_WORDS = ["angry", "furious", "ridiculous", "useless", "terrible",
               "scam", "cancel", "lawyer", "worst", "garbage"]
ESCALATION_THRESHOLD = 2


def analyze_sentiment(text: str) -> dict:
    """Score caller sentiment and decide whether to escalate.

    Args:
        text (str): the caller's latest utterance(s).

    Returns:
        dict: {"sentiment": str, "anger_hits": int, "escalate": bool}.
        sentiment is "negative" when any anger word appears, else "neutral".
    """
    lowered = text.lower()
    hits = sum(1 for word in ANGER_WORDS if word in lowered)
    return {
        "sentiment": "negative" if hits else "neutral",
        "anger_hits": hits,
        "escalate": hits >= ESCALATION_THRESHOLD,
    }


sentiment_agent = Agent(
    name="sentiment_agent",
    model=config.get_model(),
    description="Detects caller anger and recommends human escalation.",
    instruction="""
You are the Sentiment Detector on a live call.

For each caller utterance:
1) Call analyze_sentiment on the text.
2) If escalate is True, reply: "ESCALATE — caller is upset, transfer to a human."
3) Otherwise reply with one short line describing the mood.

Be brief — you run in parallel with the live agent.
""",
    tools=[analyze_sentiment],
)
