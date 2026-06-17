# ============================================================
# agents/churn_predictor_agent.py
# ------------------------------------------------------------
# Churn Predictor Agent
#
# What it does:
#   Reads a finished call transcript and flags whether the account
#   is at risk of churning, with a 0-100 risk score and the signals
#   that drove the decision.
#
# How it fits in CallOS:
#   Runs post-call. At-risk accounts surface in the analytics
#   churn-risk query (analytics_server get_churn_risks).
#
# ADK pattern used:
#   google.adk.agents.LlmAgent with a Pydantic output_schema
#   (same pattern as ADK/Module 4 - Structured Output)
# ============================================================

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from google.adk.agents import LlmAgent
from pydantic import BaseModel, Field

import config


class ChurnRisk(BaseModel):
    at_risk: bool = Field(description="True if the account may churn")
    risk_score: float = Field(description="Churn risk 0-100 (higher = riskier)")
    signals: list[str] = Field(description="Short phrases that indicate risk")


churn_predictor_agent = LlmAgent(
    name="churn_predictor_agent",
    model=config.get_model(),
    description="Flags at-risk accounts from post-call conversation signals.",
    instruction="""
You assess churn risk from a call transcript. Look for unresolved
issues, repeated complaints, cancellation talk, and negative sentiment arc.

Return JSON with exactly these fields:
- at_risk: true or false.
- risk_score: 0-100. 70+ means clear churn risk.
- signals: list of short phrases from the call that justify the score
  (empty list if not at risk).

Example: {"at_risk": true, "risk_score": 82, "signals": ["mentioned cancelling", "unresolved billing issue"]}
""",
    output_schema=ChurnRisk,
    output_key="churn_risk",
)
