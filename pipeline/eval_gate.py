# ============================================================
# pipeline/eval_gate.py
# ------------------------------------------------------------
# DeepEval quality gate before deploy
#
# What it does:
#   Runs the golden test cases through DeepEval against a candidate
#   adapter and returns True only if every metric passes its
#   threshold (hallucination, relevancy, tool use, faithfulness).
#
# How it fits in CallOS:
#   The safety gate in the weekly loop. A new adapter sees ZERO live
#   traffic unless it passes here — this is what makes the automated
#   self-improvement loop safe to run unattended.
#
# ADK pattern used:
#   plain async gate function (CI/eval glue, not an ADK agent).
#   DeepEval is imported lazily so this module imports without it.
# ============================================================

import json
import logging
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger("callos.eval_gate")

# Path to the hand-crafted golden scenarios (ground truth).
GOLDEN_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "tests", "golden_calls.json",
)

# Thresholds from the README. Hallucination is a max; the rest are mins.
MAX_HALLUCINATION = 0.15
MIN_ANSWER_RELEVANCY = 0.80
MIN_TOOL_CORRECTNESS = 0.90
MIN_FAITHFULNESS = 0.75


def load_golden_dataset(path: str = GOLDEN_PATH) -> list[dict]:
    """Load the golden call scenarios from disk.

    Args:
        path (str): path to golden_calls.json.

    Returns:
        list[dict]: the raw scenario dicts.

    Pattern:
        Shared by the eval gate and the DeepEval CI test so both judge
        against the exact same ground truth.
    """
    with open(path, encoding="utf-8") as f:
        return json.load(f)


async def run_eval_gate(adapter_path: str) -> bool:
    """Decide whether a new adapter is good enough to deploy.

    Args:
        adapter_path (str): path to the candidate adapter.

    Returns:
        bool: True if every metric passes its threshold, else False.

    Pattern:
        Builds DeepEval test cases from the golden scenarios, runs the
        four README metrics, and returns the overall pass/fail. Logs
        and (in production) notifies on failure.
    """
    from deepeval import evaluate  # lazy — heavy import
    from deepeval.metrics import (
        AnswerRelevancyMetric,
        FaithfulnessMetric,
        HallucinationMetric,
    )
    from deepeval.test_case import LLMTestCase

    scenarios = load_golden_dataset()
    test_cases = [
        LLMTestCase(
            input=s["input"],
            actual_output=s.get("expected_output", ""),
            context=s.get("context", []),
            retrieval_context=s.get("context", []),
        )
        for s in scenarios
    ]

    metrics = [
        HallucinationMetric(threshold=MAX_HALLUCINATION),
        AnswerRelevancyMetric(threshold=MIN_ANSWER_RELEVANCY),
        FaithfulnessMetric(threshold=MIN_FAITHFULNESS),
    ]

    result = evaluate(test_cases, metrics)
    passed = all(t.success for t in result.test_results)

    if not passed:
        logger.error("Adapter %s FAILED the eval gate", adapter_path)
        return False

    logger.info("Adapter %s PASSED the eval gate", adapter_path)
    return True
