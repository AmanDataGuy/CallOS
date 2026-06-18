"""
CallOS Demo
-----------
Streamlit app that walks through the full call processing pipeline
one stage at a time, using live agents.

Run:
    streamlit run demo/app.py
"""

import asyncio
import json
import os
import sys
import time
import uuid

import streamlit as st
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv()

from agents.compliance_agent import check_compliance
from agents.sentiment_agent import analyze_sentiment
import db

# ──────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────
st.set_page_config(page_title="CallOS Demo", page_icon="📞", layout="wide")

# ──────────────────────────────────────────────
# Sidebar — live key status
# ──────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/phone-office.png", width=64)
    st.title("CallOS")
    st.caption("AI Call Center — Live Demo")
    st.divider()

    st.markdown("**Active services**")
    _KEYS = {
        "LLM (Groq)": "GROQ_API_KEY",
        "LLM (Gemini)": "GOOGLE_API_KEY",
        "STT (Deepgram)": "DEEPGRAM_API_KEY",
        "TTS (ElevenLabs)": "ELEVENLABS_API_KEY",
    }
    for label, env in _KEYS.items():
        if os.environ.get(env):
            st.success(f"✓ {label}")
        else:
            st.warning(f"✗ {label} (stub)")

    st.divider()
    st.markdown("**Architecture**")
    st.markdown(
        "- ADK root agent\n"
        "- Compliance sub-agent\n"
        "- Sentiment sub-agent\n"
        "- Lead scorer (post-call)\n"
        "- SQLite call store\n"
        "- ElevenLabs TTS\n"
        "- Deepgram STT\n"
    )

# ──────────────────────────────────────────────
# Async runner — Streamlit is synchronous
# ──────────────────────────────────────────────
def run_sync(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ──────────────────────────────────────────────
# ADK agent runner (cached so imports happen once)
# ──────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading agents...")
def _load_agents():
    from agents.agent import root_agent
    from agents.lead_scorer_agent import lead_scorer_agent
    return root_agent, lead_scorer_agent


async def _agent_turn(agent, text: str) -> str:
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.genai import types

    session_service = InMemorySessionService()
    runner = Runner(agent=agent, app_name="callos-demo", session_service=session_service)
    sid = str(uuid.uuid4())
    await session_service.create_session(
        app_name="callos-demo", user_id="demo", session_id=sid
    )
    msg = types.Content(role="user", parts=[types.Part(text=text)])
    final = ""
    async for event in runner.run_async(user_id="demo", session_id=sid, new_message=msg):
        if event.is_final_response() and event.content and event.content.parts:
            final = event.content.parts[0].text or ""
    return final


def agent_turn(agent, text: str) -> str:
    return run_sync(_agent_turn(agent, text))


# ──────────────────────────────────────────────
# Preset call scenarios
# ──────────────────────────────────────────────
PRESETS = {
    "Pricing inquiry (warm lead)": (
        "Hi, I'm interested in your Enterprise plan. "
        "Can you walk me through the pricing and what features are included?"
    ),
    "Angry caller (escalation)": (
        "This is absolutely ridiculous and useless. Your service is the worst, "
        "I want to cancel everything immediately. I'm furious."
    ),
    "Compliance violation": (
        "I just wanted to say — this is a guaranteed return, totally risk free "
        "investment. You must buy now before the offer expires."
    ),
    "Hot lead (ready to buy)": (
        "I've done my research and I'd like to purchase the Enterprise plan today. "
        "Can you send me a contract? I need this set up by end of week."
    ),
    "Simple support": (
        "Hey, I'm having trouble logging into my account. Can you help me reset my password?"
    ),
}

# ──────────────────────────────────────────────
# Tabs
# ──────────────────────────────────────────────
tab_sim, tab_history, tab_pipeline = st.tabs([
    "📞 Live Call Simulator",
    "📋 Call History",
    "⚙️ Pipeline",
])

# ══════════════════════════════════════════════
# TAB 1 — Live Call Simulator
# ══════════════════════════════════════════════
with tab_sim:
    st.subheader("Live Call Simulator")
    st.caption(
        "Type a caller message and press **Run Call** to watch it flow through "
        "every stage of the CallOS pipeline in real time."
    )

    # Preset buttons above the text area
    st.markdown("**Quick scenarios:**")
    preset_cols = st.columns(len(PRESETS))
    for col, (label, text) in zip(preset_cols, PRESETS.items()):
        if col.button(label, use_container_width=True):
            st.session_state["preset_text"] = text

    # Text area — seeded by preset or default
    default = st.session_state.pop("preset_text", None) or (
        "Hi, I'm interested in your product. Can you tell me more about pricing?"
    )

    caller_input = st.text_area("Caller says:", value=default, height=90, key="caller_text")

    run_btn = st.button("▶ Run Call", type="primary", use_container_width=True)

    if run_btn and caller_input.strip():
        root_agent, lead_scorer_agent = _load_agents()

        st.divider()
        st.markdown("### Pipeline execution")

        # ── Stage 1: STT ──────────────────────────────
        with st.status("**Stage 1 — Speech-to-Text** (Deepgram Nova-3)", expanded=True) as s1:
            time.sleep(0.2)
            has_deepgram = bool(os.environ.get("DEEPGRAM_API_KEY"))
            if has_deepgram:
                st.success("Live Deepgram transcription active")
            else:
                st.info("Stub mode — text passed directly (set DEEPGRAM_API_KEY for live STT)")
            st.code(caller_input, language=None)
            s1.update(label="**Stage 1 — STT** ✓", state="complete")

        # ── Stage 2: LLM Agent ────────────────────────
        agent_response = ""
        with st.status("**Stage 2 — Live Voice Agent** (LLM)", expanded=True) as s2:
            try:
                t0 = time.time()
                agent_response = agent_turn(root_agent, caller_input)
                elapsed = time.time() - t0
                st.success(f"Response in {elapsed:.1f}s")
                st.markdown(f"**Agent:** {agent_response}")
                s2.update(label="**Stage 2 — LLM Agent** ✓", state="complete")
            except Exception as exc:
                st.error(f"Agent error: {exc}")
                s2.update(label="**Stage 2 — LLM Agent** ✗", state="error")

        # ── Stages 3 & 4 side by side ─────────────────
        col_c, col_s = st.columns(2)

        compliance = {"compliant": True, "violations": []}
        with col_c:
            with st.status("**Stage 3 — Compliance**", expanded=True) as s3:
                compliance = check_compliance(caller_input)
                if compliance["compliant"]:
                    st.success("COMPLIANT — no violations")
                else:
                    st.error(f"VIOLATION: {', '.join(compliance['violations'])}")
                    st.caption("Agent must self-correct or escalate.")
                s3.update(
                    label=(
                        "**Stage 3 — Compliance** ✓"
                        if compliance["compliant"]
                        else "**Stage 3 — Compliance** ⚠"
                    ),
                    state="complete" if compliance["compliant"] else "error",
                )

        sentiment = {"sentiment": "neutral", "anger_hits": 0, "escalate": False}
        with col_s:
            with st.status("**Stage 4 — Sentiment**", expanded=True) as s4:
                sentiment = analyze_sentiment(caller_input)
                mood_icon = (
                    "😠" if sentiment["escalate"]
                    else "😟" if sentiment["sentiment"] == "negative"
                    else "😊"
                )
                st.write(f"{mood_icon} **{sentiment['sentiment'].capitalize()}**  "
                         f"· anger signals: {sentiment['anger_hits']}")
                if sentiment["escalate"]:
                    st.error("ESCALATE — transfer to human agent")
                else:
                    st.success("No escalation needed")
                s4.update(label="**Stage 4 — Sentiment** ✓", state="complete")

        # ── Stage 5: Lead Scorer ──────────────────────
        lead = {"status": "cold", "score": 0, "reason": ""}
        with st.status("**Stage 5 — Lead Scorer** (post-call LLM)", expanded=True) as s5:
            try:
                t0 = time.time()
                raw = agent_turn(lead_scorer_agent, caller_input)
                elapsed = time.time() - t0
                try:
                    lead = json.loads(raw)
                except (json.JSONDecodeError, TypeError):
                    lead = {"status": "cold", "score": 0, "reason": raw[:120]}

                icons = {"hot": "🔴", "warm": "🟡", "cold": "🔵"}
                icon = icons.get(lead.get("status", "cold"), "⚪")
                st.write(
                    f"{icon} **{lead.get('status', 'cold').upper()}** — "
                    f"Score: {lead.get('score', 0)}/100  ({elapsed:.1f}s)"
                )
                st.caption(lead.get("reason", ""))
                s5.update(
                    label=f"**Stage 5 — Lead: {lead.get('status','cold').upper()}** ✓",
                    state="complete",
                )
            except Exception as exc:
                st.error(f"Lead scorer error: {exc}")
                s5.update(label="**Stage 5 — Lead Scorer** ✗", state="error")

        # ── Stage 6: TTS ──────────────────────────────
        with st.status("**Stage 6 — Text-to-Speech** (ElevenLabs Flash v2.5)", expanded=True) as s6:
            tts_key = os.environ.get("ELEVENLABS_API_KEY")
            if tts_key and agent_response:
                async def _tts(text: str) -> bytes:
                    from elevenlabs.client import AsyncElevenLabs
                    voice_id = os.environ.get("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")
                    client = AsyncElevenLabs(api_key=tts_key)
                    chunks = []
                    async for chunk in client.text_to_speech.convert(
                        voice_id=voice_id,
                        text=text,
                        model_id="eleven_flash_v2_5",
                    ):
                        chunks.append(chunk)
                    return b"".join(chunks)

                try:
                    audio_bytes = run_sync(_tts(agent_response))
                    st.audio(audio_bytes, format="audio/mp3")
                    st.success("Audio ready — press play to hear the agent")
                    s6.update(label="**Stage 6 — TTS** ✓", state="complete")
                except Exception as exc:
                    st.warning(f"TTS error: {exc}")
                    s6.update(label="**Stage 6 — TTS** ✗", state="error")
            elif agent_response:
                st.info("Stub — set ELEVENLABS_API_KEY to hear audio output")
                st.markdown(f"*Would speak:* \"{agent_response}\"")
                s6.update(label="**Stage 6 — TTS** (stub)", state="complete")
            else:
                s6.update(label="**Stage 6 — TTS** (skipped)", state="complete")

        # ── Save to DB ────────────────────────────────
        st.divider()
        call_id = str(uuid.uuid4())
        transcript = f"Caller: {caller_input}\nAgent: {agent_response}"
        try:
            run_sync(db.execute(
                "INSERT INTO calls "
                "(id, phone_number, direction, transcript, started_at, outcome, lead_status, quality_score) "
                "VALUES (?, ?, 'inbound', ?, CURRENT_TIMESTAMP, ?, ?, ?)",
                (
                    call_id,
                    "+1-555-DEMO",
                    transcript,
                    agent_response[:200],
                    lead.get("status", "cold"),
                    float(lead.get("score", 0)),
                ),
            ))
            st.success(f"Call saved to history — ID: `{call_id[:8]}…`")
        except Exception as exc:
            st.warning(f"Could not save call (run `python scripts/init_db.py` first): {exc}")

        # ── Summary card ──────────────────────────────
        st.divider()
        st.markdown("### Call summary")
        c1, c2, c3 = st.columns(3)
        c1.metric(
            "Compliance",
            "PASS" if compliance["compliant"] else "FAIL",
            delta=None,
            delta_color="normal",
        )
        c2.metric(
            "Sentiment",
            sentiment["sentiment"].capitalize(),
            delta="escalate" if sentiment["escalate"] else None,
            delta_color="inverse",
        )
        icons = {"hot": "🔴 HOT", "warm": "🟡 WARM", "cold": "🔵 COLD"}
        c3.metric("Lead", icons.get(lead.get("status", "cold"), "—"))


# ══════════════════════════════════════════════
# TAB 2 — Call History
# ══════════════════════════════════════════════
with tab_history:
    st.subheader("Call History")

    if st.button("🔄 Refresh", key="refresh_history"):
        st.rerun()

    try:
        calls = run_sync(db.fetch_all(
            "SELECT id, phone_number, direction, lead_status, quality_score, "
            "started_at, outcome FROM calls ORDER BY started_at DESC LIMIT 100"
        ))

        if not calls:
            st.info("No calls yet — run a simulation in the Live Call Simulator tab.")
        else:
            scored = [c["quality_score"] for c in calls if c.get("quality_score")]
            hot = sum(1 for c in calls if c.get("lead_status") == "hot")
            warm = sum(1 for c in calls if c.get("lead_status") == "warm")
            avg = sum(scored) / len(scored) if scored else 0

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Calls", len(calls))
            m2.metric("Hot Leads", hot)
            m3.metric("Warm Leads", warm)
            m4.metric("Avg Quality Score", f"{avg:.0f}/100" if scored else "—")

            st.divider()

            for call in calls:
                icon = {"hot": "🔴", "warm": "🟡", "cold": "🔵"}.get(
                    call.get("lead_status", ""), "⚪"
                )
                score = call.get("quality_score")
                score_str = f"{score:.0f}/100" if score else "unscored"
                ts = (call.get("started_at") or "")[:16]

                with st.expander(
                    f"{icon} {call.get('phone_number','?')}  ·  "
                    f"{(call.get('lead_status') or '?').upper()}  ·  "
                    f"Score: {score_str}  ·  {ts}"
                ):
                    st.caption(f"ID: {call['id']}")
                    st.write(f"**Direction:** {call.get('direction','?')}")
                    if call.get("outcome"):
                        st.write(f"**Agent response:** {call['outcome']}")

    except Exception as exc:
        st.error(f"Could not load calls: {exc}")
        st.info("Run `python scripts/init_db.py` to initialise the database first.")


# ══════════════════════════════════════════════
# TAB 3 — Pipeline
# ══════════════════════════════════════════════
with tab_pipeline:
    st.subheader("Self-Improvement Pipeline")
    st.caption(
        "Calls that score above the quality threshold feed the weekly fine-tune loop. "
        "High-quality calls become SFT training pairs; bad calls become DPO 'rejected' examples."
    )

    if st.button("🔄 Refresh", key="refresh_pipeline"):
        st.rerun()

    try:
        import config as _cfg

        all_calls = run_sync(db.fetch_all("SELECT quality_score, lead_status FROM calls"))

        HIGH = _cfg.MIN_TRAIN_SCORE  # 80
        LOW = 40.0

        high = [c for c in all_calls if c.get("quality_score", 0) >= HIGH]
        low = [c for c in all_calls if 0 < (c.get("quality_score") or 0) < LOW]
        unscored = [c for c in all_calls if not c.get("quality_score")]

        g1, g2, g3, g4 = st.columns(4)
        g1.metric("Total calls", len(all_calls))
        g2.metric(f"SFT candidates (≥{HIGH:.0f})", len(high))
        g3.metric(f"DPO 'rejected' (<{LOW:.0f})", len(low))
        g4.metric("Unscored", len(unscored))

        st.divider()

        # Fine-tune gate
        GATE = 20
        st.markdown(f"**Weekly fine-tune gate:** {len(high)} / {GATE} qualifying calls")
        st.progress(min(len(high) / GATE, 1.0))
        if len(high) >= GATE:
            st.success("Gate passed — fine-tune would run")
        else:
            st.warning(f"{GATE - len(high)} more high-quality calls needed to trigger training")

        st.divider()

        # Dataset estimates
        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown("**SFT dataset**")
            st.metric("Estimated pairs", len(high) * 3)
            st.caption(
                "Each high-quality call contributes ~3 (instruction, output) pairs "
                "from its conversation turns."
            )
        with col_r:
            st.markdown("**DPO dataset**")
            dpo = min(len(high), len(low), 100)
            st.metric("Preference pairs", dpo)
            st.caption(
                "Good calls paired with bad calls teach the model "
                "to prefer compliant, helpful replies."
            )

        st.divider()

        st.markdown("**Pipeline stages**")
        pipeline_stages = [
            ("1  Score calls", "pipeline/scorer.py",
             "LLM-as-judge scores every call 0–100 on helpfulness, compliance, and resolution."),
            ("2  Build dataset", "pipeline/dataset_builder.py",
             "Pulls high/low scored calls, converts to SFT instruction pairs + DPO preference pairs."),
            ("3  Fine-tune (QLoRA)", "pipeline/trainer.py",
             "4-bit quantized LoRA on 7B model — fits in 8 GB VRAM, runs overnight."),
            ("4  Eval gate", "tests/test_agent_quality.py",
             "DeepEval relevancy ≥ 0.80 and faithfulness ≥ 0.75 must pass before deploy."),
            ("5  Deploy", "cloudbuild.yaml",
             "Cloud Build: install → pytest → deepeval gate → docker build → push → Cloud Run."),
        ]
        for name, path, desc in pipeline_stages:
            with st.container(border=True):
                st.markdown(f"**{name}** &nbsp; `{path}`")
                st.caption(desc)

    except Exception as exc:
        st.error(f"Pipeline data unavailable: {exc}")
        st.info("Run `python scripts/init_db.py` to initialise the database.")
