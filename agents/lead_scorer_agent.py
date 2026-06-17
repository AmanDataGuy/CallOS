# ============================================================
# agents/lead_scorer_agent.py
# ------------------------------------------------------------
# Lead Scorer Agent
#
# What it does:
#   Reads a finished call transcript and classifies the lead as
#   hot / warm / cold with a 0-100 score and a one-line reason.
#
# How it fits in CallOS:
#   Runs post-call. Hot leads are pushed to the CRM (crm_server
#   push_to_crm) and trigger a sales notification.
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


class LeadScore(BaseModel):
    status: str = Field(description="One of: hot, warm, cold")
    score: float = Field(description="Qualification score 0-100")
    reason: str = Field(description="One short sentence justifying the score")


lead_scorer_agent = LlmAgent(
    name="lead_scorer_agent",
    model=config.get_model(),
    description="Classifies a post-call lead as hot/warm/cold with a score.",
    instruction="""
You score sales leads from a call transcript. Read the transcript the
user provides and judge buying intent.

Return JSON with exactly these fields:
- status: "hot" (clear intent / asked to buy), "warm" (interested,
  needs follow-up), or "cold" (no intent).
- score: 0-100. Hot is 80-100, warm 40-79, cold 0-39.
- reason: one short sentence.

Example: {"status": "hot", "score": 88, "reason": "Asked for enterprise pricing and a contract."}
""",
    output_schema=LeadScore,
    output_key="lead_score",
)
