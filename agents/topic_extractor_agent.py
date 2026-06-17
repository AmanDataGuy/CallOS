# ============================================================
# agents/topic_extractor_agent.py
# ------------------------------------------------------------
# Topic Extractor Agent
#
# What it does:
#   Reads one or more call transcripts and clusters what callers
#   talked about into feature complaints, pricing objections, and
#   competitor mentions.
#
# How it fits in CallOS:
#   Runs as a nightly batch over recent calls. Results are stored in
#   the analytics table and surfaced via analytics_server
#   get_topic_clusters / get_product_signals for the BI dashboard.
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


class TopicClusters(BaseModel):
    feature_complaints: list[str] = Field(description="Features callers complained about")
    pricing_objections: list[str] = Field(description="Pricing concerns raised")
    competitor_mentions: list[str] = Field(description="Competitors callers named")


topic_extractor_agent = LlmAgent(
    name="topic_extractor_agent",
    model=config.get_model(),
    description="Clusters call topics into complaints, objections, and competitors.",
    instruction="""
You extract recurring topics from call transcripts. Read the transcript(s)
the user provides and group what callers discussed.

Return JSON with exactly these fields (each a list of short phrases):
- feature_complaints: product features that drew complaints.
- pricing_objections: pricing concerns raised.
- competitor_mentions: names of competitors mentioned.

Use empty lists where nothing applies. Do not invent items not in the text.
""",
    output_schema=TopicClusters,
    output_key="topic_clusters",
)
