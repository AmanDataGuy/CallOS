# ============================================================
# scheduler/fine_tune_job.py
# ------------------------------------------------------------
# Weekly fine-tune trigger (APScheduler)
#
# What it does:
#   Every Sunday at 02:00 it builds the training dataset, runs SFT
#   then DPO, gates the new adapter through DeepEval, and (on pass)
#   marks it for A/B deploy — all with no human in the loop.
#
# How it fits in CallOS:
#   The orchestrator of the self-improvement loop. It chains the
#   pipeline modules (dataset_builder -> dpo_trainer -> eval_gate)
#   on a schedule.
#
# ADK pattern used:
#   asyncio scheduler driving plain pipeline functions
#   (support code for the fine-tune loop, not an ADK agent)
# ============================================================

import asyncio
import datetime
import json
import logging
import os
import subprocess
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from apscheduler.schedulers.asyncio import AsyncIOScheduler

import cache
from pipeline.dataset_builder import build_training_dataset
from pipeline.dpo_trainer import run_dpo_alignment
from pipeline.eval_gate import run_eval_gate
from pipeline.grpo_trainer import run_grpo_alignment

logger = logging.getLogger("callos.fine_tune")
logging.basicConfig(level=logging.INFO)

scheduler = AsyncIOScheduler()

# Need at least this many SFT pairs or a week is too thin to train on.
MIN_SFT_PAIRS = 50
# DPO requires paired (chosen, rejected) data. Fall back to GRPO when
# there are fewer pairs than this — GRPO only needs prompts + reward.
MIN_DPO_PAIRS = 20
# Where datasets get staged for LLaMA-Factory.
DATA_DIR = "data"
# Fraction of live traffic the new adapter gets during A/B (README: 5%).
AB_TRAFFIC_PCT = 0.05


def run_sft_training(sft_pairs: list[dict], week_num: int) -> str:
    """Run QLoRA SFT via LLaMA-Factory and return the adapter path.

    Args:
        sft_pairs (list[dict]): instruction/output training pairs.
        week_num (int): ISO week number for the output path.

    Returns:
        str: directory the SFT adapter was written to.

    Pattern:
        Stages the pairs as JSON, then shells out to llamafactory-cli
        with configs/sft_config.yaml — the README's training step.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(os.path.join(DATA_DIR, "callos_sft.json"), "w", encoding="utf-8") as f:
        json.dump(sft_pairs, f)

    output_dir = f"models/adapters/sft/week_{week_num}"
    subprocess.run(["llamafactory-cli", "train", "configs/sft_config.yaml"], check=True)
    return output_dir


def _write_dpo_dataset(dpo_pairs: list[dict]) -> str:
    """Stage DPO preference pairs to disk and return the file path."""
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, "callos_dpo.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dpo_pairs, f)
    return path


def _write_grpo_prompts(sft_pairs: list[dict]) -> str:
    """Extract prompts from SFT pairs and stage them for GRPO."""
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, "callos_grpo_prompts.json")
    prompts = [{"prompt": p["instruction"]} for p in sft_pairs]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(prompts, f)
    return path


@scheduler.scheduled_job("cron", day_of_week="sun", hour=2, minute=0)
async def weekly_fine_tune() -> None:
    """Run the full weekly fine-tune cycle end to end.

    Args:
        (none — fired by the cron trigger)

    Returns:
        None

    Pattern:
        build dataset -> SFT -> DPO -> eval gate -> A/B deploy. Bails
        early if there is too little data or the eval gate fails.
    """
    logger.info("Starting weekly fine-tune cycle")
    week_num = datetime.datetime.now().isocalendar()[1]

    dataset = await build_training_dataset()
    if len(dataset.sft) < MIN_SFT_PAIRS:
        logger.warning("Insufficient data (%d pairs). Skipping.", len(dataset.sft))
        return

    sft_path = run_sft_training(dataset.sft, week_num)

    # Choose alignment method: DPO when paired data is rich, GRPO otherwise.
    if len(dataset.dpo) >= MIN_DPO_PAIRS:
        logger.info("Running DPO alignment (%d pairs).", len(dataset.dpo))
        aligned_path = run_dpo_alignment(
            sft_path, _write_dpo_dataset(dataset.dpo), week_num
        )
    else:
        logger.info(
            "Insufficient DPO pairs (%d < %d). Falling back to GRPO.",
            len(dataset.dpo), MIN_DPO_PAIRS,
        )
        aligned_path = run_grpo_alignment(
            sft_path, _write_grpo_prompts(dataset.sft), week_num
        )

    if not await run_eval_gate(aligned_path):
        return

    # Mark the new adapter for A/B routing. # TODO: swap cache to Redis.
    cache.set("ab:new_adapter", aligned_path)
    cache.set("ab:traffic_split", str(AB_TRAFFIC_PCT))
    logger.info("Week %d fine-tune complete. A/B deploy live at %s.", week_num, aligned_path)


if __name__ == "__main__":
    scheduler.start()
    print("Fine-tune scheduler started — weekly job set for Sun 02:00.", flush=True)
    try:
        asyncio.get_event_loop().run_forever()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped.")
