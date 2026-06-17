# ============================================================
# pipeline/dataset_builder.py
# ------------------------------------------------------------
# Training dataset builder (SFT + DPO pairs)
#
# What it does:
#   Pulls high-scoring calls and turns them into supervised
#   fine-tuning pairs, and pairs high- vs low-scoring calls into DPO
#   chosen/rejected preference pairs.
#
# How it fits in CallOS:
#   The bridge between scored calls and the weekly fine-tune. The
#   quality filter here is the critical safety piece — only good
#   calls become SFT targets, only good-vs-bad become DPO signal.
#
# ADK pattern used:
#   plain async pipeline function returning a Pydantic dataset
#   (support code for the fine-tune loop, not an ADK agent)
# ============================================================

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pydantic import BaseModel, Field

import config
import db

# A call must beat this score to become SFT data; calls below the
# low bar are used as DPO "rejected" examples.
HIGH_SCORE = config.MIN_TRAIN_SCORE
LOW_SCORE = 40.0
# Cap DPO pairs so a noisy week can't flood the preference set.
MAX_DPO_PAIRS = 100


class TrainingDataset(BaseModel):
    sft: list[dict] = Field(description="instruction/output pairs")
    dpo: list[dict] = Field(description="prompt/chosen/rejected pairs")


def parse_conversation_turns(transcript: str) -> list[tuple[str, str]]:
    """Split a transcript into (caller, agent) turn pairs.

    Args:
        transcript (str): lines like "Caller: ..." / "Agent: ...".

    Returns:
        list[tuple[str, str]]: one (caller_text, agent_text) per exchange.

    Pattern:
        Walks the lines, pairing each caller line with the agent reply
        that follows it. Lines without a known speaker are ignored.
    """
    caller, pairs = None, []
    for line in transcript.splitlines():
        if line.lower().startswith("caller:"):
            caller = line.split(":", 1)[1].strip()
        elif line.lower().startswith("agent:") and caller is not None:
            pairs.append((caller, line.split(":", 1)[1].strip()))
            caller = None
    return pairs


def _first_agent_reply(transcript: str) -> str:
    """Return the first agent line of a transcript ('' if none)."""
    turns = parse_conversation_turns(transcript)
    return turns[0][1] if turns else ""


async def build_training_dataset(min_score: float = HIGH_SCORE) -> TrainingDataset:
    """Pull high-quality calls and convert them to training format.

    Args:
        min_score (float): minimum quality score for SFT inclusion.

    Returns:
        TrainingDataset: SFT instruction/output + DPO preference pairs.

    Pattern:
        Mirrors the README builder — SFT pairs from each good call's
        turns, DPO pairs by zipping good vs bad calls' first replies.
    """
    top = await db.fetch_all(
        "SELECT transcript FROM calls WHERE quality_score >= ?", (min_score,)
    )
    low = await db.fetch_all(
        "SELECT transcript FROM calls WHERE quality_score < ? LIMIT ?",
        (LOW_SCORE, MAX_DPO_PAIRS),
    )

    sft = []
    for call in top:
        for caller, agent in parse_conversation_turns(call["transcript"] or ""):
            sft.append({"instruction": caller, "output": agent})

    dpo = []
    for good, bad in zip(top[:MAX_DPO_PAIRS], low):
        turns = parse_conversation_turns(good["transcript"] or "")
        if turns:
            dpo.append({
                "prompt": turns[0][0],
                "chosen": turns[0][1],
                "rejected": _first_agent_reply(bad["transcript"] or ""),
            })

    return TrainingDataset(sft=sft, dpo=dpo)
