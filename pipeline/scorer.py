# ============================================================
# pipeline/scorer.py
# ------------------------------------------------------------
# Post-call LLM scorer (0-100) — instructor structured output
#
# What it does:
#   Sends a finished transcript to an LLM-as-judge via instructor,
#   which validates the response against the CallScore Pydantic schema
#   with automatic retry on parse failure.
#
# How it fits in CallOS:
#   Triggered as a background task from api/main.py after every call.
#   Its score is the gate for fine-tuning — only calls scoring >= 80
#   become training data (see pipeline/dataset_builder.py).
#
# ADK pattern used:
#   direct LiteLLM call with instructor for typed structured output
#   (same routing idea as ADK/Module 5 - Models, used outside an agent)
#
# Tier 3 upgrade:
#   Replaced manual json.loads() + CallScore(**raw) with instructor.
#   instructor wraps the LLM call and handles JSON extraction, Pydantic
#   validation, and up to max_retries automatic correction turns — the
#   same pattern used by production LLM pipelines (Anthropic, OpenAI).
# ============================================================

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import instructor
import litellm
from pydantic import BaseModel, Field

import config
import db

# instructor wraps litellm.acompletion and returns an AsyncInstructor.
# Every .create() call validates the response against response_model
# and retries up to max_retries times if validation fails.
_client = instructor.from_litellm(litellm.acompletion)

SCORER_PROMPT = """
You are a call quality evaluator. Score this sales/support call 0-100.

Evaluate on:
- Objection handling (0-25): Did the agent address concerns clearly?
- Compliance (0-25): Were all required disclosures made?
- Conversion signal (0-25): Did the caller show buying intent?
- Professionalism (0-25): Tone, clarity, staying on script?

Return JSON with these exact fields:
  score (int 0-100), breakdown (object with the four sub-scores),
  outcome (short string), lead_status ("hot" | "warm" | "cold")

Transcript:
{transcript}
"""


class CallScore(BaseModel):
    score: int = Field(description="Overall call quality 0-100")
    breakdown: dict = Field(description="Per-category sub-scores")
    outcome: str = Field(description="Short outcome label for the call")
    lead_status: str = Field(description="hot / warm / cold")


async def score_call(call_id: str, transcript: str) -> CallScore:
    """Score a completed call using LLM-as-judge (0-100).

    Args:
        call_id (str): UUID of the call record in the database.
        transcript (str): full conversation text as a single string.

    Returns:
        CallScore — validated Pydantic model with score, breakdown, outcome.

    Pattern:
        instructor wraps the LiteLLM call and validates the response
        against CallScore with up to 2 automatic correction turns before
        raising. No manual json.loads() needed.
    """
    score: CallScore = await _client.chat.completions.create(
        model=config.get_litellm_model_name(),
        messages=[{
            "role": "user",
            "content": SCORER_PROMPT.format(transcript=transcript),
        }],
        response_model=CallScore,
        max_retries=2,
    )
    await db.update_call_score(call_id, score)
    return score
