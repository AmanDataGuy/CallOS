# ============================================================
# pipeline/grpo_trainer.py
# ------------------------------------------------------------
# GRPO alignment pass (TRL) — DeepSeek-style RL from completions
#
# What it does:
#   Applies Group Relative Policy Optimization as an alternative to
#   DPO. GRPO samples G completions per prompt, scores each with a
#   reward function, and updates the policy toward higher-reward
#   completions — all without a reference model or paired data.
#
# How it fits in CallOS:
#   Drop-in alternative to pipeline/dpo_trainer.py for weeks when
#   paired preference data is scarce (GRPO only needs prompts + a
#   reward signal). The same before/after eval harness as DPO is
#   used so improvement is always quantified.
#
# DPO vs GRPO trade-off:
#   DPO  — stable, data-hungry, needs (chosen, rejected) pairs,
#           references to SFT model; great when preference data is rich.
#   GRPO — data-light, reward-driven, no reference model needed;
#           faster to set up and handles new domains better; can
#           be less stable without a carefully tuned reward function.
#
# ADK pattern used:
#   plain training function (runs on the local GPU box, not in ADK).
#   Heavy ML deps are imported lazily so this module imports anywhere.
# ============================================================

import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.dpo_trainer import EVAL_SCENARIOS, CALL_SYSTEM_PROMPT, eval_quality

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
GRPO_LR = 1e-5
NUM_GENERATIONS = 4   # G completions sampled per prompt (trade-off: diversity vs GPU cost)
MAX_NEW_TOKENS = 256

# ---- Reward function ----
# Rule-based scoring — no LLM API call during training for speed.
# Each rubric contributes equally to a [0, 1] score.

_PROHIBITED = [
    "i guarantee", "i promise", "free forever", "100% guaranteed",
    "calm down", "you must", "impossible",
]
_REQUIRED_MARKERS = [
    "thank you", "happy to help", "i understand", "let me",
    "specialist", "follow up", "regarding",
]


def _call_quality_reward(completions: list[str], prompts: list[str] | None = None, **_) -> list[float]:
    """Reward function called by GRPOTrainer after each generation batch.

    Scores on three rubrics:
      - Compliance  (0.4 weight): no prohibited phrases
      - Professionalism (0.3): contains at least one professional marker
      - Conciseness (0.3): response under 80 words

    Args:
        completions: list of generated strings from the model
        prompts: unused here; included to match GRPOTrainer's signature

    Returns:
        list[float] — one reward per completion, in [0.0, 1.0]
    """
    rewards = []
    for text in completions:
        lower = text.lower()
        words = len(text.split())

        compliance = 1.0 if not any(p in lower for p in _PROHIBITED) else 0.0
        professionalism = 1.0 if any(m in lower for m in _REQUIRED_MARKERS) else 0.0
        conciseness = 1.0 if words <= 80 else max(0.0, 1.0 - (words - 80) / 80)

        reward = 0.4 * compliance + 0.3 * professionalism + 0.3 * conciseness
        rewards.append(round(reward, 4))
    return rewards


def load_grpo_dataset(prompts_path: str):
    """Load a list of call prompts as a HF Dataset for GRPO.

    GRPO only needs prompts (no chosen/rejected pairs). The dataset
    JSON should be a list of {"prompt": str} dicts, or the function
    also accepts a list of plain strings.

    Args:
        prompts_path (str): path to JSON file of prompts.

    Returns:
        datasets.Dataset with a "prompt" column.
    """
    from datasets import Dataset

    with open(prompts_path, encoding="utf-8") as f:
        raw = json.load(f)

    if raw and isinstance(raw[0], str):
        rows = [{"prompt": p} for p in raw]
    else:
        rows = raw  # already {"prompt": ...} dicts

    return Dataset.from_list(rows)


def run_grpo_alignment(sft_adapter_path: str, prompts_path: str, week_num: int) -> str:
    """Apply GRPO on top of the SFT adapter using reward-based RL.

    Samples NUM_GENERATIONS completions per prompt, scores each with
    the rule-based call quality reward, and updates the model toward
    the higher-rewarded completions.  Runs the standard before/after
    eval harness (same as DPO) to quantify improvement.

    Args:
        sft_adapter_path (str): path to the SFT LoRA adapter to start from.
        prompts_path (str): JSON file of call prompt strings.
        week_num (int): ISO week number, used in the output path.

    Returns:
        str: directory the GRPO adapter was saved to.
    """
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, load_in_4bit=True, device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, sft_adapter_path)

    print("[GRPO] Running pre-training eval...")
    score_before = eval_quality(model, tokenizer)
    print(f"[GRPO] Quality before: {score_before:.4f}")

    output_dir = f"models/adapters/grpo/week_{week_num}"
    grpo_config = GRPOConfig(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=GRPO_LR,
        num_generations=NUM_GENERATIONS,
        max_new_tokens=MAX_NEW_TOKENS,
        fp16=True,
        logging_steps=10,
    )

    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=load_grpo_dataset(prompts_path),
        reward_funcs=[_call_quality_reward],
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model()

    print("[GRPO] Running post-training eval...")
    score_after = eval_quality(model, tokenizer)
    delta = score_after - score_before
    sign = "+" if delta >= 0 else ""
    print(f"[GRPO] Quality after:  {score_after:.4f}  (Δ {sign}{delta:.4f})")
    print(f"[GRPO] Adapter saved to: {output_dir}")
    return output_dir
