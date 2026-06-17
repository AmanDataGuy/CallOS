# ============================================================
# config.py
# ------------------------------------------------------------
# Central configuration + LLM model selection
#
# What it does:
#   Loads .env once and exposes the few settings every layer
#   needs: which LLM key to use, where the local database lives,
#   and a get_model() helper that returns the right ADK model.
#
# How it fits in CallOS:
#   Every agent, MCP server, and pipeline step imports from here
#   so there is exactly one place that decides "which LLM" and
#   "which database". No paid keys are required to run locally.
#
# ADK pattern used:
#   google.adk.models.lite_llm.LiteLlm for non-Gemini providers
#   (same pattern as ADK/Module 5 - Models)
# ============================================================

import os

from dotenv import load_dotenv
from google.adk.models.lite_llm import LiteLlm

load_dotenv()

# Local SQLite file — production swaps this for DATABASE_URL (Postgres).
SQLITE_PATH = os.environ.get("SQLITE_PATH", "callos.db")
DATABASE_URL = os.environ.get("DATABASE_URL", "")

# Quality threshold a call must beat to enter the fine-tune dataset.
# 80 matches the README self-improvement loop ("score >= 80").
MIN_TRAIN_SCORE = 80.0


def get_model():
    """
    Pick the LLM for ADK agents based on which key is in .env.

    Args:
        (none)

    Returns:
        str | LiteLlm — a Gemini model name string when GOOGLE_API_KEY
        is set (ADK talks to Gemini natively), otherwise a LiteLlm
        instance routed to Groq or OpenAI.

    Pattern:
        Checks keys in priority order GOOGLE -> GROQ -> OPENAI, the
        same order the build spec requires. Gemini is returned as a
        plain string because ADK is Gemini-native; the others are
        wrapped in LiteLlm exactly like ADK/Module 5.
    """
    # Gemini is ADK-native — pass the model name straight through.
    if os.environ.get("GOOGLE_API_KEY"):
        return "gemini-2.0-flash"

    # Groq is free and fast — good local fallback.
    if os.environ.get("GROQ_API_KEY"):
        return LiteLlm(model="groq/llama-3.3-70b-versatile")

    # OpenAI last (paid).
    if os.environ.get("OPENAI_API_KEY"):
        return LiteLlm(model="openai/gpt-4o")

    # No key — return Gemini name so imports still succeed. Calls will
    # fail clearly at runtime, which is the desired "set a key" signal.
    return "gemini-2.0-flash"


def get_litellm_model_name() -> str:
    """
    Pick the model STRING for direct litellm.acompletion() calls.

    Args:
        (none)

    Returns:
        str — a litellm-style "provider/model" identifier.

    Pattern:
        The pipeline scorer calls litellm directly (not through ADK),
        so it needs a string, not a LiteLlm object. Same key priority
        as get_model().
    """
    if os.environ.get("GOOGLE_API_KEY"):
        return "gemini/gemini-2.0-flash"
    if os.environ.get("GROQ_API_KEY"):
        return "groq/llama-3.3-70b-versatile"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai/gpt-4o"
    return "gemini/gemini-2.0-flash"
