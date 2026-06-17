# ============================================================
# tests/eval_retrieval.py
# ------------------------------------------------------------
# RAGAS knowledge-base retrieval evaluation
#
# What it does:
#   Measures how well the KB retrieval grounds answers using RAGAS:
#   faithfulness and context precision over a set of test questions.
#
# How it fits in CallOS:
#   The development-time eval for the RAG layer (kb_agent / kb_server).
#   The README targets faithfulness > 0.75 and context precision > 0.70.
#
# ADK pattern used:
#   standalone eval over the KB the agents query
#   (evaluates ADK/Module 6-style retrieval tools)
# ============================================================

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Thresholds from the README RAG eval step.
MIN_FAITHFULNESS = 0.75
MIN_CONTEXT_PRECISION = 0.70

# A few KB questions with their ground-truth answers + context.
TEST_QUESTIONS = [
    {
        "question": "How much is the enterprise plan?",
        "answer": "The Enterprise plan is $2,000/month.",
        "contexts": ["CallOS Enterprise plan is $2,000/month and includes unlimited call minutes."],
        "ground_truth": "The Enterprise plan costs $2,000 per month.",
    },
    {
        "question": "What is the refund window?",
        "answer": "Refunds are available within 14 days.",
        "contexts": ["Refunds are available within 14 days."],
        "ground_truth": "Refunds are available within 14 days.",
    },
]


def eval_kb_retrieval(test_questions: list[dict] = TEST_QUESTIONS) -> dict:
    """Run RAGAS faithfulness + context precision over KB answers.

    Args:
        test_questions (list[dict]): question/answer/contexts/ground_truth rows.

    Returns:
        dict: the RAGAS metric scores.

    Pattern:
        Wraps the rows in a HF Dataset and evaluates with RAGAS, then
        asserts the README thresholds. RAGAS is imported lazily so this
        file imports without it installed.
    """
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import context_precision, faithfulness

    dataset = Dataset.from_list(test_questions)
    result = evaluate(dataset, metrics=[faithfulness, context_precision])

    assert result["faithfulness"] > MIN_FAITHFULNESS
    assert result["context_precision"] > MIN_CONTEXT_PRECISION
    return result


if __name__ == "__main__":
    print(eval_kb_retrieval())
