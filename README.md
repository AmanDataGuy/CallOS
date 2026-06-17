# CallOS

**Enterprise AI Voice Agent Platform with a Self-Improving Fine-Tuning Loop** — built on Google ADK 2.2.

Live calls → real-time multi-agent handling → post-call scoring → DPO/GRPO alignment → weekly fine-tune → better calls.

> Full architecture, all six build phases, and the agent/MCP rosters live in
> [CallOS_README.md](CallOS_README.md). This file is the practical run guide.

---

## What's here

A Google ADK multi-agent voice platform that runs **end-to-end locally with no paid APIs**.
Voice services (Deepgram, ElevenLabs) are key-gated — stub fallbacks fire when keys are absent so the full pipeline still runs locally.

```
agents/      8 ADK agents  — live voice root + sub-agents + post-call specialists
mcp/         6 MCP servers — 29 tools across CRM, KB, calls, calendar, scorer, analytics
api/         FastAPI + WebSocket — Twilio ConversationRelay, /test-call, A2A Task API, HITL
pipeline/    scorer (instructor), dataset builder, DPO trainer, GRPO trainer, DeepEval gate
scheduler/   APScheduler weekly fine-tune — DPO when data is rich, GRPO fallback when scarce
scripts/     init_db, index_kb (embeds + stores chunks for two-stage retrieval)
tests/       5 deterministic + 20 DeepEval golden scenarios
configs/     LLaMA-Factory SFT config, LiteLLM routing
terraform/   GCP Cloud Run + Cloud SQL + Redis + Secret Manager
hitl.py      Human-in-the-Loop coordinator (asyncio.Future + contextvars)
config.py    LLM key routing: GOOGLE_API_KEY → GROQ_API_KEY → OPENAI_API_KEY
db.py        async SQLite (aiosqlite); swap to asyncpg for PostgreSQL in production
cache.py     in-process dict pub/sub; swap to Redis in production
```

---

## Tech stack

| Layer | Technology |
|---|---|
| Agent orchestration | Google ADK 2.2, LlmAgent, sub_agents, FunctionTool, Runner |
| Multi-agent | root agent + 3 live sub-agents + 4 post-call specialists |
| Protocol | MCP (stdio, 6 servers, 29 tools), A2A (Agent Card + Task API) |
| RAG | Two-stage: bi-encoder cosine → CrossEncoder reranking (sentence-transformers) |
| LLM routing | LiteLLM (Gemini / Groq / OpenAI) |
| Structured output | Pydantic v2 schemas, instructor (auto-retry on validation failure) |
| Telephony | Twilio ConversationRelay WebSocket, Deepgram Nova-3 STT, ElevenLabs Flash v2.5 TTS |
| API | FastAPI, async WebSocket, key-gated real/stub voice services |
| HITL | asyncio.Future + Python contextvars — real pause/resume on escalation |
| Fine-tuning | QLoRA SFT (LLaMA-Factory) + DPO (TRL) + GRPO (TRL) with before/after eval |
| Evaluation | DeepEval CI gate (golden scenarios), Promptfoo red-teaming, rule-based before/after |
| Scheduling | APScheduler AsyncIOScheduler — weekly fine-tune cron |
| Data layer | SQLite (aiosqlite) locally; PostgreSQL + pgvector in production |
| Cloud infra | Terraform → GCP Cloud Run, Cloud SQL, Memorystore, Secret Manager |
| CI/CD | Cloud Build 6-step pipeline: test → eval-gate → build → push → deploy |

---

## Local quick start (no paid keys needed)

```bash
# 1. Create and activate a venv
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment  (at minimum one LLM key)
cp .env.example .env
# Edit .env: set GOOGLE_API_KEY (free at aistudio.google.com) or GROQ_API_KEY (free at console.groq.com)

# 4. Initialise the database
python scripts/init_db.py
#    → Tables created: calls, leads, kb_chunks, analytics

# 5. Seed the knowledge base (embeds 3 seed docs into kb_chunks)
python scripts/index_kb.py

# 6. Start the API server
uvicorn api.main:app --reload --port 8000
#    → Swagger docs at http://localhost:8000/docs

# 7. Simulate a call
curl -X POST http://localhost:8000/test-call \
  -H "Content-Type: application/json" \
  -d '{"transcript": "Hi, I want to know about your enterprise pricing"}'

# 8. Run the ADK dev playground
adk web ./agents/

# 9. Run tests
pytest tests/ -v
#    → 5 passed, 20 skipped (skipped = DeepEval gate, needs judge key)

# 10. Run the DeepEval gate (needs an LLM key)
deepeval test run tests/test_agent_quality.py
```

---

## Key endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/` | Health check |
| POST | `/test-call` | Simulate an inbound call (no telephony required) |
| POST | `/incoming-call` | Twilio webhook — returns TwiML for ConversationRelay |
| WS | `/ws` | Twilio ConversationRelay WebSocket |
| GET | `/escalation/pending` | List calls paused waiting for human response |
| POST | `/escalation/{call_id}/respond` | Unblock a paused HITL escalation |
| GET | `/.well-known/agent.json` | A2A Agent Card (machine-readable agent description) |
| POST | `/a2a/tasks` | Submit a task (A2A protocol) — returns immediately, runs in background |
| GET | `/a2a/tasks/{task_id}` | Poll task status / result |
| POST | `/a2a/tasks/{task_id}/cancel` | Cancel a running task |
| GET | `/a2a/tasks` | List all tasks |

---

## HITL (Human-in-the-Loop) demo

```bash
# Terminal 1 — start the server
uvicorn api.main:app --port 8000

# Terminal 2 — send a call that will trigger escalation
curl -X POST http://localhost:8000/test-call \
  -H "Content-Type: application/json" \
  -d '{"transcript": "This is ridiculous and useless, I want to cancel and sue you"}'
# → call is PAUSED, waiting for human response

# Terminal 3 — check which calls are pending
curl http://localhost:8000/escalation/pending

# Terminal 3 — respond as the human supervisor
curl -X POST http://localhost:8000/escalation/<call_id>/respond \
  -H "Content-Type: application/json" \
  -d '{"response": "Offer the caller a 30-day extension and a 20% discount"}'
# → Terminal 2 resumes and the agent relays the human response
```

---

## A2A protocol demo

```bash
# Discover the agent
curl http://localhost:8000/.well-known/agent.json

# Submit a task
curl -X POST http://localhost:8000/a2a/tasks \
  -H "Content-Type: application/json" \
  -d '{
    "message": {
      "role": "user",
      "parts": [{"type": "text", "text": "What is included in the Enterprise plan?"}]
    },
    "metadata": {"phone_number": "+15551234567"}
  }'
# → {"id": "...", "status": {"state": "submitted"}, "artifacts": []}

# Poll for result
curl http://localhost:8000/a2a/tasks/<task_id>
# → {"status": {"state": "completed"}, "artifacts": [{"name": "call-result", ...}]}
```

---

## MCP servers (standalone)

Each MCP server speaks stdio and can be run independently for testing:

```bash
python mcp/crm_server.py        # CRM: get_lead, create_lead, update_lead, log_call, get_deal, push_to_crm
python mcp/kb_server.py         # KB:  search_kb (2-stage RAG), get_faq, get_product_info, get_pricing
python mcp/analytics_server.py  # BI:  lead_funnel (%), conversion_trend (7d rolling avg), quality_leaderboard
```

> **Why no `mcp/__init__.py`:** The folder would shadow the installed `mcp` PyPI package.
> Without `__init__.py` the real package always resolves first; servers add project root to `sys.path` directly.

---

## Local substitutions

| Production | Local default | Swap point |
|---|---|---|
| PostgreSQL + pgvector | SQLite (`callos.db`) | `db.py`, `config.py` |
| Redis | in-process dict | `cache.py` |
| Deepgram Nova-3 STT | stub (set `DEEPGRAM_API_KEY` to enable) | `api/main.py: transcribe_audio` |
| ElevenLabs Flash v2.5 TTS | stub (set `ELEVENLABS_API_KEY` to enable) | `api/main.py: synthesize_speech` |
| Twilio call | `POST /test-call` | `api/main.py` |
| Gemini / Groq / OpenAI | whichever key is in `.env` | `config.py: get_model` |

---

## Fine-tuning pipeline (runs on GPU)

```
weekly_fine_tune (Sunday 02:00)
  └─ build_training_dataset()   → SFT pairs + DPO preference pairs
  └─ run_sft_training()         → QLoRA SFT via LLaMA-Factory
  └─ if dpo_pairs >= 20:
  │    run_dpo_alignment()      → DPO with KL-penalised reference model
  └─ else:
  │    run_grpo_alignment()     → GRPO with rule-based reward function
  └─ run_eval_gate()            → DeepEval golden scenario gate
  └─ cache.set("ab:new_adapter", path)  → A/B at 5% traffic
```

Before-and-after quality is measured automatically:
```
[DPO] Quality before: 0.6200
[DPO] Quality after:  0.7800  (Δ +0.1600)
```

---

## CI/CD (Cloud Build)

`cloudbuild.yaml` runs on every push to `main`:

```
Step 1  pip install -r requirements.txt
Step 2  pytest tests/ -v
Step 3  deepeval test run tests/test_agent_quality.py   ← blocks deploy on LLM regression
Step 4  docker build -t gcr.io/$PROJECT_ID/callos-api .
Step 5  docker push gcr.io/$PROJECT_ID/callos-api
Step 6  gcloud run deploy callos-api --region=us-central1
```

---

## Deploy to GCP

```bash
terraform -chdir=terraform apply -var project_id=$GOOGLE_CLOUD_PROJECT
gcloud builds submit --config cloudbuild.yaml
adk deploy cloud_run --project=$GOOGLE_CLOUD_PROJECT --region=us-central1 ./agents/
```

---

## Code conventions

Every file opens with a header naming the ADK module its pattern came from. Functions carry
Args/Returns/Pattern docstrings. No hardcoded keys — all credentials are `os.environ.get()`.
Heavy ML imports (torch, transformers, sentence-transformers) are lazy so every module imports
cheaply outside the training environment.
