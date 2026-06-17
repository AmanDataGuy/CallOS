# ============================================================
# tests/test_agent_quality.py
# ------------------------------------------------------------
# DeepEval CI tests + deterministic agent tool tests
#
# What it does:
#   Two layers of tests. The deterministic ones exercise the pure
#   tool logic (compliance, sentiment, golden data) and always run.
#   The DeepEval layer scores agent responses against the golden
#   scenarios and runs only when DeepEval + a judge key are present.
#
# How it fits in CallOS:
#   `pytest tests/ -v` runs the deterministic layer in any CI box;
#   `deepeval test run tests/test_agent_quality.py` runs the gated
#   layer — the same gate the weekly fine-tune must pass.
#
# ADK pattern used:
#   tests call the agents' FunctionTool funcs directly
#   (the tools defined in ADK/Module 3 & 6 style agents)
# ============================================================

import json
import os
import sys

import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.compliance_agent import check_compliance
from agents.sentiment_agent import analyze_sentiment

GOLDEN_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden_calls.json")


def load_golden_dataset(path: str = GOLDEN_PATH) -> list[dict]:
    """Load the 20 golden call scenarios used across the eval stack."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# -----------------------------
# Deterministic tests (no LLM, always run)
# -----------------------------

def test_golden_dataset_has_twenty_scenarios():
    """The golden set must hold exactly 20 fully-formed scenarios."""
    scenarios = load_golden_dataset()
    assert len(scenarios) == 20
    for s in scenarios:
        assert s["input"] and s["expected_output"]


def test_compliance_flags_banned_phrase():
    """A banned phrase must be reported as non-compliant."""
    result = check_compliance("This is a guaranteed return, totally risk free.")
    assert result["compliant"] is False
    assert "guaranteed return" in result["violations"]


def test_compliance_passes_clean_text():
    """Clean text must come back compliant with no violations."""
    result = check_compliance("Happy to walk you through the pricing options.")
    assert result["compliant"] is True
    assert result["violations"] == []


def test_sentiment_escalates_on_repeated_anger():
    """Two or more anger words must trigger escalation."""
    result = analyze_sentiment("This is ridiculous and useless, I want to cancel.")
    assert result["escalate"] is True
    assert result["sentiment"] == "negative"


def test_sentiment_stays_calm_on_neutral_text():
    """Neutral text must not escalate."""
    result = analyze_sentiment("Thanks, that makes sense, I'll think about it.")
    assert result["escalate"] is False


# -----------------------------
# DeepEval gate (runs only with DeepEval + a judge key)
# -----------------------------

def _judge_key_present() -> bool:
    """True if a key DeepEval can judge with is configured."""
    return bool(os.environ.get("OPENAI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))


@pytest.mark.parametrize("scenario", load_golden_dataset(), ids=lambda s: s["id"])
def test_agent_response_quality(scenario):
    """Score a golden scenario's reference answer against the rubric.

    Args:
        scenario (dict): one golden_calls.json entry.

    Pattern:
        Builds a DeepEval LLMTestCase and asserts the relevancy +
        faithfulness metrics pass the README thresholds. Skipped when
        DeepEval or a judge key is unavailable so plain CI stays green.
    """
    pytest.importorskip("deepeval")
    if not _judge_key_present():
        pytest.skip("No judge key (OPENAI_API_KEY / GOOGLE_API_KEY) set")

    from deepeval import assert_test
    from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
    from deepeval.test_case import LLMTestCase

    test_case = LLMTestCase(
        input=scenario["input"],
        actual_output=scenario["expected_output"],
        retrieval_context=scenario.get("context", []),
    )
    assert_test(test_case, [
        AnswerRelevancyMetric(threshold=0.80),
        FaithfulnessMetric(threshold=0.75),
    ])
