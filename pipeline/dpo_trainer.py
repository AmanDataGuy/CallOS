# ============================================================
# pipeline/dpo_trainer.py
# ------------------------------------------------------------
# DPO alignment pass (TRL) with before/after quality eval
#
# What it does:
#   Applies Direct Preference Optimization on top of the SFT adapter
#   using the week's chosen/rejected pairs, then saves the new adapter.
#   A lightweight eval harness measures call quality before and after
#   training so the improvement is quantified, not just assumed.
#
# How it fits in CallOS:
#   Step 3 of the weekly fine-tune. DPO is what pushes the model
#   toward successful call patterns and away from failed ones — the
#   difference between this loop and plain SFT.
#
# ADK pattern used:
#   plain training function (runs on the local GPU box, not in ADK).
#   Heavy ML deps are imported lazily so this module imports anywhere.
# ============================================================

import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Base model the adapters are trained on (fits RTX 3050 in 4-bit).
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
# KL penalty weight — how far DPO may drift from the SFT reference.
DPO_BETA = 0.1
# Conservative LR for preference tuning so we nudge, not overwrite.
DPO_LR = 5e-6

# Fixed held-out scenarios used for before/after eval.
# These are never part of the training set so the score is unbiased.
EVAL_SCENARIOS = [
    {
        "prompt": "I'm calling about your Enterprise plan pricing.",
        "must_contain": ["2,000", "enterprise", "support"],
        "must_not_contain": ["guaranteed", "promise", "free forever"],
    },
    {
        "prompt": "I'm really frustrated, nobody is helping me.",
        "must_contain": ["understand", "help", "specialist"],
        "must_not_contain": ["calm down", "you need to", "impossible"],
    },
    {
        "prompt": "What's the difference between your plans?",
        "must_contain": ["starter", "enterprise", "minutes"],
        "must_not_contain": ["I don't know", "I'm not sure", "maybe"],
    },
    {
        "prompt": "Does CallOS integrate with Salesforce?",
        "must_contain": ["salesforce", "crm", "composio"],
        "must_not_contain": ["no", "cannot", "not supported"],
    },
    {
        "prompt": "Can I get a refund?",
        "must_contain": ["14", "refund", "days"],
        "must_not_contain": ["never", "impossible", "no refunds"],
    },
]

CALL_SYSTEM_PROMPT = (
    "You are CallOS, a professional AI voice agent. "
    "Answer the caller's question concisely and accurately. "
    "Keep your reply to 1-3 sentences."
)


def _score_response(response: str, scenario: dict) -> float:
    """Rule-based quality score in [0, 1] for one scenario response.

    Checks for required keywords (recall) and prohibited phrases
    (compliance). No LLM API call needed — fast enough to run before
    and after every training run.

    Args:
        response: model's generated text
        scenario: dict with must_contain and must_not_contain lists

    Returns:
        float in [0.0, 1.0]
    """
    resp_lower = response.lower()
    hits = sum(1 for kw in scenario["must_contain"] if kw.lower() in resp_lower)
    misses = sum(1 for kw in scenario["must_not_contain"] if kw.lower() in resp_lower)
    recall = hits / max(len(scenario["must_contain"]), 1)
    penalty = misses * 0.25
    return max(0.0, min(1.0, recall - penalty))


def eval_quality(model, tokenizer) -> float:
    """Run the held-out eval set and return mean quality score [0, 1].

    Generates one response per scenario with greedy decoding (fast,
    deterministic) then scores with _score_response. Used before and
    after training to quantify alignment improvement.

    Args:
        model: the loaded (PEFT-wrapped) causal LM
        tokenizer: corresponding HF tokenizer

    Returns:
        float — mean quality score across EVAL_SCENARIOS
    """
    import torch

    model.eval()
    scores = []
    for scenario in EVAL_SCENARIOS:
        messages = [
            {"role": "system", "content": CALL_SYSTEM_PROMPT},
            {"role": "user", "content": scenario["prompt"]},
        ]
        input_ids = tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True
        ).to(model.device)
        with torch.no_grad():
            out = model.generate(
                input_ids,
                max_new_tokens=128,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        response = tokenizer.decode(
            out[0][input_ids.shape[-1]:], skip_special_tokens=True
        )
        scores.append(_score_response(response, scenario))

    mean_score = sum(scores) / len(scores)
    return round(mean_score, 4)


def load_dpo_dataset(dpo_dataset_path: str):
    """Load DPO prompt/chosen/rejected rows into a HF Dataset.

    Args:
        dpo_dataset_path (str): JSON file of preference pairs.

    Returns:
        datasets.Dataset: the loaded preference dataset.
    """
    from datasets import Dataset

    with open(dpo_dataset_path, encoding="utf-8") as f:
        rows = json.load(f)
    return Dataset.from_list(rows)


def run_dpo_alignment(sft_adapter_path: str, dpo_dataset_path: str, week_num: int) -> str:
    """Apply DPO preference optimization on top of the SFT adapter.

    Args:
        sft_adapter_path (str): path to the SFT LoRA adapter.
        dpo_dataset_path (str): path to the DPO preference JSON.
        week_num (int): ISO week number, used in the output path.

    Returns:
        str: the directory the new DPO adapter was saved to.

    Pattern:
        Loads the 4-bit base, stacks the SFT adapter, runs TRL's
        DPOTrainer for one epoch, then measures the quality delta
        with the held-out eval harness before returning.
    """
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, load_in_4bit=True, device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, sft_adapter_path)

    print("[DPO] Running pre-training eval...")
    score_before = eval_quality(model, tokenizer)
    print(f"[DPO] Quality before: {score_before:.4f}")

    output_dir = f"models/adapters/dpo/week_{week_num}"
    dpo_config = DPOConfig(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=DPO_LR,
        beta=DPO_BETA,
        fp16=True,
    )

    trainer = DPOTrainer(
        model=model,
        args=dpo_config,
        train_dataset=load_dpo_dataset(dpo_dataset_path),
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model()

    print("[DPO] Running post-training eval...")
    score_after = eval_quality(model, tokenizer)
    delta = score_after - score_before
    sign = "+" if delta >= 0 else ""
    print(f"[DPO] Quality after:  {score_after:.4f}  (Δ {sign}{delta:.4f})")
    print(f"[DPO] Adapter saved to: {output_dir}")
    return output_dir
