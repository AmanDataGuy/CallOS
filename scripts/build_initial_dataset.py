"""
scripts/build_initial_dataset.py
---------------------------------
One-time cold-start bootstrap.

Reads call_data/call_recordings.csv, generates an agent response for every
caller transcript via LiteLLM, scores each pair, then writes the files the
fine-tune pipeline expects:

  data/callos_sft.json      — alpaca-format SFT pairs  (score >= 80)
  data/dpo_pairs.json       — chosen/rejected pairs     (high vs low scored)
  data/dataset_info.json    — LlamaFactory dataset registry

Run once before llamafactory-cli train configs/sft_config.yaml.
No database, no API server needed.
"""

import asyncio
import csv
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import instructor
import litellm

import config
from pipeline.scorer import SCORER_PROMPT, CallScore

CSV_PATH = os.path.join("call_data", "call_recordings.csv")
DATA_DIR = "data"
SFT_PATH = os.path.join(DATA_DIR, "callos_sft.json")
DPO_PATH = os.path.join(DATA_DIR, "dpo_pairs.json")
DATASET_INFO_PATH = os.path.join(DATA_DIR, "dataset_info.json")

HIGH_SCORE = config.MIN_TRAIN_SCORE   # 80
LOW_SCORE = 40.0


def _pick_model() -> str:
    # Prefer Groq for bulk inference — generous free quota, no per-day cap issues.
    # Falls back to the standard config priority if Groq isn't configured.
    if os.environ.get("GROQ_API_KEY"):
        return "groq/llama-3.3-70b-versatile"
    return config.get_litellm_model_name()

AGENT_SYSTEM_PROMPT = (
    "You are CallOS, a professional AI voice agent handling a live phone call. "
    "Keep replies short and natural — this is spoken, not written. "
    "Be empathetic and helpful. Never invent product facts. "
    "Stay compliant — no guarantees or pressure tactics. "
    "Respond in 2-4 sentences maximum."
)

_client = instructor.from_litellm(litellm.acompletion)
_model = _pick_model()


async def generate_response(caller_text: str) -> str:
    resp = await litellm.acompletion(
        model=_model,
        messages=[
            {"role": "system", "content": AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": caller_text},
        ],
        max_tokens=150,
    )
    return resp.choices[0].message.content.strip()


async def score_pair(caller: str, agent: str) -> CallScore:
    transcript = f"Caller: {caller}\nAgent: {agent}"
    return await _client.chat.completions.create(
        model=_model,
        messages=[{"role": "user", "content": SCORER_PROMPT.format(transcript=transcript)}],
        response_model=CallScore,
        max_retries=2,
    )


async def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    rows = []
    with open(CSV_PATH, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print(f"[build] {len(rows)} calls found in {CSV_PATH}\n")

    results = []
    for i, row in enumerate(rows, 1):
        caller = row["Transcript"]
        call_id = row["id"]
        label = f"{row['Type']}, {row['Sentiment']}"
        print(f"[{i:02d}/{len(rows)}] {call_id}  ({label})")
        try:
            agent_reply = await generate_response(caller)
            await asyncio.sleep(2)          # stay under Groq 30 RPM free tier
            scored = await score_pair(caller, agent_reply)
            results.append({
                "id": call_id,
                "caller": caller,
                "agent": agent_reply,
                "score": scored.score,
                "outcome": scored.outcome,
                "lead_status": scored.lead_status,
            })
            print(f"         score={scored.score:3d}  lead={scored.lead_status}  {scored.outcome}")
        except Exception as exc:
            print(f"         ERROR: {exc}")
        await asyncio.sleep(2)              # pace between calls

    high = [r for r in results if r["score"] >= HIGH_SCORE]
    low  = [r for r in results if r["score"] < LOW_SCORE]

    print(f"\n[build] high (>={int(HIGH_SCORE)}): {len(high)}   low (<{int(LOW_SCORE)}): {len(low)}")

    # SFT — alpaca format expected by LlamaFactory (instruction / output)
    sft_pairs = [
        {"instruction": r["caller"], "input": "", "output": r["agent"]}
        for r in high
    ]

    # DPO — zip high with low; if counts differ, cycle the shorter list
    import itertools
    low_cycle = itertools.cycle(low) if low else iter([])
    dpo_pairs = [
        {
            "prompt": h["caller"],
            "chosen": h["agent"],
            "rejected": next(low_cycle)["agent"],
        }
        for h in high
    ] if low else []

    # LlamaFactory dataset registry
    dataset_info = {
        "callos_sft": {
            "file_name": "callos_sft.json"
        }
    }

    with open(SFT_PATH, "w", encoding="utf-8") as f:
        json.dump(sft_pairs, f, indent=2, ensure_ascii=False)

    with open(DPO_PATH, "w", encoding="utf-8") as f:
        json.dump(dpo_pairs, f, indent=2, ensure_ascii=False)

    with open(DATASET_INFO_PATH, "w", encoding="utf-8") as f:
        json.dump(dataset_info, f, indent=2)

    print(f"\n[build] {SFT_PATH}          -> {len(sft_pairs)} SFT pairs")
    print(f"[build] {DPO_PATH}          -> {len(dpo_pairs)} DPO pairs")
    print(f"[build] {DATASET_INFO_PATH} -> LlamaFactory registry")
    print("\n[build] Done. Next step:")
    print("        llamafactory-cli train configs/sft_config.yaml")


if __name__ == "__main__":
    asyncio.run(main())
