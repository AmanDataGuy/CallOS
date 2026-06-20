"""
scripts/build_initial_dataset.py
---------------------------------
One-time cold-start bootstrap.

Pipeline:
  1. Load call_data/call_recordings.csv (20 real customer calls)
  2. For each call, generate a 2-round multi-turn conversation (caller + agent)
  3. Score full transcript with LLM-as-judge -> CallScore (0-100)
  4. Classify by threshold and save intermediate results to build_results.json
  5. Generate intentionally bad agent responses for DPO rejected examples
  6. Write final training files

Output files:
  data/callos_sft.json      -- alpaca-format SFT pairs  (score >= 65)
  data/dpo_pairs.json       -- chosen/rejected pairs     (>=80 vs bad generated)
  data/dataset_info.json    -- LlamaFactory dataset registry
  data/build_results.json   -- cached intermediate results (skip re-run)

Threshold logic (multi-turn + fallback floor):
  score >= 80  ->  SFT + DPO chosen
  score 65-79  ->  SFT only
  score <  40  ->  DPO rejected  (natural bad calls if any)
  score 40-64  ->  discarded
  [DPO rejected are also generated synthetically for guaranteed coverage]

Re-run behaviour:
  If data/build_results.json exists, skips generation and loads from cache.
  Delete it to force a full re-run.

Run once before: llamafactory-cli train configs/sft_config.yaml
No database, no API server needed.
"""

import asyncio
import csv
import itertools
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import instructor
import litellm

import config
from pipeline.scorer import SCORER_PROMPT, CallScore

CSV_PATH        = os.path.join("call_data", "call_recordings.csv")
DATA_DIR        = "data"
SFT_PATH        = os.path.join(DATA_DIR, "callos_sft.json")
DPO_PATH        = os.path.join(DATA_DIR, "dpo_pairs.json")
DATASET_INFO_PATH = os.path.join(DATA_DIR, "dataset_info.json")
CACHE_PATH      = os.path.join(DATA_DIR, "build_results.json")

HIGH_SCORE = config.MIN_TRAIN_SCORE   # 80 -- SFT + DPO chosen
MID_SCORE  = 65.0                      # 65 -- SFT only fallback
LOW_SCORE  = 40.0                      # 40 -- DPO rejected (natural)

# Groq free tier: 30 RPM. Each conversation uses ~4 LLM calls.
_INTER_CALL_SLEEP = 2
_INTER_CONV_SLEEP = 3


def _pick_model() -> str:
    if os.environ.get("GROQ_API_KEY"):
        return "groq/llama-3.3-70b-versatile"
    return config.get_litellm_model_name()


AGENT_SYSTEM_PROMPT = (
    "You are CallOS, a professional AI voice agent handling a live phone call. "
    "Keep replies short and natural — this is spoken, not written. "
    "Be empathetic and helpful. Never invent product facts you don't know. "
    "Stay compliant — no guarantees or pressure tactics. "
    "Respond in 2-4 sentences maximum."
)

# Generates bad agent responses for DPO rejected examples.
# Deliberately violates every quality criterion the scorer measures.
BAD_AGENT_SYSTEM_PROMPT = (
    "You are a poorly trained AI call center agent. "
    "Make up plausible-sounding but incorrect product specifications and prices. "
    "Use high-pressure sales tactics: urgency language, fake limited-time offers. "
    "Be dismissive of complaints — tell customers their problem is their own fault. "
    "Ignore what the customer actually asked and push an unrelated product. "
    "Respond in 2-3 sentences."
)

FOLLOWUP_SYSTEM_PROMPT = (
    "You are simulating a real customer on a support/sales phone call. "
    "Given the opening statement you made and the agent's reply, write ONE short "
    "natural follow-up response (1-3 sentences). Stay in character — if you were "
    "angry, stay frustrated but slightly calmer. If you were happy, stay positive. "
    "Ask a specific follow-up question or respond to what the agent said. "
    "Do NOT start with 'Caller:' — just write the words."
)

_client = instructor.from_litellm(litellm.acompletion)
_model  = _pick_model()


async def _llm(system: str, user: str, history: list | None = None) -> str:
    messages = [{"role": "system", "content": system}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user})
    resp = await litellm.acompletion(model=_model, messages=messages, max_tokens=150)
    return resp.choices[0].message.content.strip()


async def generate_conversation(opener: str, call_type: str) -> list[tuple[str, str]]:
    """Build a 2-round (caller, agent) conversation from a single opener."""
    agent1 = await _llm(AGENT_SYSTEM_PROMPT, opener)
    await asyncio.sleep(_INTER_CALL_SLEEP)

    followup_ctx = [
        {"role": "assistant", "content": f"You opened the call by saying: {opener}"},
        {"role": "assistant", "content": f"The agent replied: {agent1}"},
    ]
    caller2 = await _llm(
        FOLLOWUP_SYSTEM_PROMPT,
        f"Call type: {call_type}. Write your follow-up.",
        history=followup_ctx,
    )
    await asyncio.sleep(_INTER_CALL_SLEEP)

    agent2 = await _llm(
        AGENT_SYSTEM_PROMPT,
        caller2,
        history=[
            {"role": "user", "content": opener},
            {"role": "assistant", "content": agent1},
        ],
    )
    await asyncio.sleep(_INTER_CALL_SLEEP)

    return [(opener, agent1), (caller2, agent2)]


async def generate_bad_response(caller_text: str) -> str:
    """Generate a deliberately bad agent response for DPO rejected examples."""
    return await _llm(BAD_AGENT_SYSTEM_PROMPT, caller_text)


async def score_conversation(turns: list[tuple[str, str]]) -> CallScore:
    transcript = "\n".join(f"Caller: {c}\nAgent: {a}" for c, a in turns)
    return await _client.chat.completions.create(
        model=_model,
        messages=[{"role": "user", "content": SCORER_PROMPT.format(transcript=transcript)}],
        response_model=CallScore,
        max_retries=2,
    )


async def score_single(caller: str, agent: str) -> int:
    """Score a single (caller, agent) pair and return the integer score."""
    result = await _client.chat.completions.create(
        model=_model,
        messages=[{
            "role": "user",
            "content": SCORER_PROMPT.format(transcript=f"Caller: {caller}\nAgent: {agent}"),
        }],
        response_model=CallScore,
        max_retries=2,
    )
    return result.score


def _tier(score: int) -> str:
    if score >= HIGH_SCORE:
        return "SFT+DPO"
    if score >= MID_SCORE:
        return "SFT"
    if score < LOW_SCORE:
        return "REJECT"
    return "skip"


async def run_generation(rows: list[dict]) -> list[dict]:
    """Run full 20-call multi-turn generation. Saves cache on completion."""
    results = []
    for i, row in enumerate(rows, 1):
        call_id   = row["id"]
        call_type = row["Type"]
        opener    = row["Transcript"]
        print(f"[{i:02d}/{len(rows)}] {call_id}  ({call_type}, {row['Sentiment']})")
        try:
            turns  = await generate_conversation(opener, call_type)
            scored = await score_conversation(turns)
            tier   = _tier(scored.score)
            results.append({
                "id":          call_id,
                "call_type":   call_type,
                "turns":       turns,
                "score":       scored.score,
                "outcome":     scored.outcome,
                "lead_status": scored.lead_status,
                "tier":        tier,
            })
            print(f"         score={scored.score:3d}  [{tier}]  {scored.outcome}")
        except Exception as exc:
            print(f"         ERROR: {exc}")
        await asyncio.sleep(_INTER_CONV_SLEEP)

    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[build] Intermediate results cached -> {CACHE_PATH}")
    return results


async def build_dpo_rejected(high: list[dict]) -> list[dict]:
    """Generate synthetic bad responses for DPO rejected examples."""
    print(f"\n[dpo]   Generating {len(high)} bad (rejected) responses ...")
    dpo_pairs = []
    for i, h in enumerate(high, 1):
        prompt = h["turns"][0][0]   # first caller turn
        chosen = h["turns"][0][1]   # first agent turn (good)
        print(f"[dpo]   {i}/{len(high)}  {h['id']}")
        try:
            rejected = await generate_bad_response(prompt)
            await asyncio.sleep(_INTER_CALL_SLEEP)
            bad_score = await score_single(prompt, rejected)
            await asyncio.sleep(_INTER_CALL_SLEEP)
            print(f"           bad_score={bad_score}  (target <{int(LOW_SCORE)})")
            dpo_pairs.append({
                "prompt":   prompt,
                "chosen":   chosen,
                "rejected": rejected,
                "bad_score": bad_score,
            })
        except Exception as exc:
            print(f"           ERROR: {exc}")
        await asyncio.sleep(_INTER_CONV_SLEEP)
    return dpo_pairs


async def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    # ---- Step 1: Load or generate conversations ----
    if os.path.exists(CACHE_PATH):
        print(f"[build] Loading cached results from {CACHE_PATH}")
        with open(CACHE_PATH, encoding="utf-8") as f:
            results = json.load(f)
        print(f"[build] {len(results)} calls loaded from cache\n")
    else:
        with open(CSV_PATH, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        print(f"[build] {len(rows)} calls  |  model: {_model}")
        print(f"[build] thresholds: SFT>={int(MID_SCORE)}  DPO-chosen>={int(HIGH_SCORE)}  DPO-rej<{int(LOW_SCORE)}\n")
        results = await run_generation(rows)

    high = [r for r in results if r["tier"] == "SFT+DPO"]
    mid  = [r for r in results if r["tier"] == "SFT"]
    low  = [r for r in results if r["tier"] == "REJECT"]

    print(f"\n[build] SFT+DPO  (>={int(HIGH_SCORE)}):    {len(high)}")
    print(f"[build] SFT-only ({int(MID_SCORE)}-{int(HIGH_SCORE)-1}):  {len(mid)}")
    print(f"[build] DPO-rej  (<{int(LOW_SCORE)}):    {len(low)} (natural)")

    # ---- Step 2: SFT pairs (high + mid, all turns) ----
    sft_pairs = []
    for r in high + mid:
        for caller_text, agent_text in r["turns"]:
            sft_pairs.append({
                "instruction": caller_text,
                "input": "",
                "output": agent_text,
            })

    # ---- Step 3: DPO pairs (synthetic bad responses for guaranteed coverage) ----
    dpo_pairs = await build_dpo_rejected(high)

    # Also include any naturally low-scoring calls as extra rejected examples
    if low:
        low_cycle = itertools.cycle(low)
        for h in high:
            rej = next(low_cycle)
            dpo_pairs.append({
                "prompt":   h["turns"][0][0],
                "chosen":   h["turns"][0][1],
                "rejected": rej["turns"][0][1],
                "bad_score": rej["score"],
            })
        print(f"[dpo]   +{len(low)} natural rejected examples added")

    # Strip scoring metadata before writing (trainer doesn't need it)
    dpo_final = [{"prompt": d["prompt"], "chosen": d["chosen"], "rejected": d["rejected"]}
                 for d in dpo_pairs]

    # ---- Step 4: Write output files ----
    dataset_info = {"callos_sft": {"file_name": "callos_sft.json"}}

    with open(SFT_PATH, "w", encoding="utf-8") as f:
        json.dump(sft_pairs, f, indent=2, ensure_ascii=False)
    with open(DPO_PATH, "w", encoding="utf-8") as f:
        json.dump(dpo_final, f, indent=2, ensure_ascii=False)
    with open(DATASET_INFO_PATH, "w", encoding="utf-8") as f:
        json.dump(dataset_info, f, indent=2)

    print(f"\n[build] {SFT_PATH}           -> {len(sft_pairs)} SFT pairs")
    print(f"[build] {DPO_PATH}           -> {len(dpo_final)} DPO pairs")
    print(f"[build] {DATASET_INFO_PATH}   -> LlamaFactory registry")
    print("\n[build] Done. Next:")
    print("        llamafactory-cli train configs/sft_config.yaml")


if __name__ == "__main__":
    asyncio.run(main())
