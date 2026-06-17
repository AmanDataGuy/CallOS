# CallOS — Complete Study Notes

A from-scratch-to-advanced walkthrough of every concept, framework, and file in this
project. Read top to bottom for a full course; or jump to **Part C** if you just need
"what does file X do."

> **Scope note before you start:** This repo has two layers.
> 1. [CallOS_README.md](CallOS_README.md) is the *full product vision* — a 6-phase plan
>    including Twilio, Deepgram, ElevenLabs, Vertex AI Agent Engine, Langfuse, Composio,
>    A2A protocol, a Streamlit dashboard, multi-tenant isolation. Most of that is **not
>    built** — it's the roadmap.
> 2. What's actually implemented (and what these notes mostly explain) is the **local,
>    no-paid-API slice**: real Google ADK agents, real MCP servers, a real FastAPI
>    backend, a real fine-tuning pipeline design — but running on SQLite instead of
>    Postgres, an in-memory dict instead of Redis, and stub functions instead of
>    Twilio/Deepgram/ElevenLabs. [README.md](README.md) documents this slice and has a
>    table of every substitution.
>
> Wherever a concept below is "aspirational" (planned but not coded), it's marked
> **[ROADMAP]**. Everything else is in the repo right now and runnable.

---

## Table of Contents

- **Part A — Foundational Concepts** (theory, comparisons, how CallOS uses each one)
  1. Agents & the LLM-agent paradigm
  2. Google ADK (Agent Development Kit)
  3. MCP — Model Context Protocol
  4. LiteLLM — multi-provider LLM routing
  5. Pydantic — typed data & structured LLM output
  6. FastAPI + WebSockets
  7. Data layer: SQLite/aiosqlite vs PostgreSQL+pgvector
  8. Caching & pub/sub (Redis concept)
  9. RAG & embeddings
  10. Evaluation frameworks: DeepEval, RAGAS, Promptfoo
  11. Fine-tuning: SFT, LoRA/QLoRA, DPO, PEFT, TRL
  12. APScheduler
  13. Docker & docker-compose
  14. Terraform / IaC & GCP services
  15. CI/CD (Cloud Build)
- **Part B — CallOS System Design**
  1. Data model
  2. The 8 agents and how they relate
  3. The 6 MCP servers and their 28 tools
  4. Request lifecycle — a call from start to finish
  5. The self-improvement loop, step by step
  6. Local substitutions (recap table)
- **Part C — File-by-File Reference** (every file, every function, one line each)
- **Part D — How Everything Connects** (dependency graph, data flow)
- **Part E — Known Limitations / What's Stubbed**
- **Glossary**

---

# Part A — Foundational Concepts

## A.1 Agents & the LLM-agent paradigm

An **LLM agent** is a loop: the model receives a goal + conversation history + a list of
*tools* it's allowed to call, decides whether to respond directly or call a tool, the
tool result is fed back in, and the loop repeats until the model produces a final answer.
This is fundamentally different from a single prompt → completion call — the model is
making **multi-step decisions** about what action to take next.

The three things every agent framework has to solve:
1. **Tool calling** — how does the model express "call function X with these args" and
   how does the host program execute it and feed the result back?
2. **State / memory** — what does the model see across turns (conversation history,
   session variables, long-term memory)?
3. **Orchestration** — when there's more than one agent, who decides which agent handles
   which part of the task (routing, delegation, parallel execution)?

CallOS answers all three with **Google ADK** for orchestration/state, **MCP** for tool
access standardization, and a **manager + sub-agents** topology (see A.2).

---

## A.2 Google ADK (Agent Development Kit)

### What it is
Google ADK (`google-adk` on PyPI) is an open-source Python framework for building,
testing, and deploying LLM agents. It's model-agnostic in principle (via LiteLLM
wrapping) but is "Gemini-native" — passing a plain model name string like
`"gemini-2.0-flash"` works without any wrapper, because ADK talks to Gemini directly.

### Core building blocks (all used in this repo)

| Concept | Class / API | What it does |
|---|---|---|
| **Agent** | `google.adk.agents.Agent` | A conversational agent: name, model, instruction (system prompt), tools, sub_agents |
| **LlmAgent** | `google.adk.agents.LlmAgent` | Same as `Agent` but built for **structured output** — pairs with `output_schema` (a Pydantic model) and `output_key` |
| **Tool** | `google.adk.tools.function_tool.FunctionTool` | Wraps a plain Python function (sync or async) so the model can call it; the function's docstring + type hints become the tool's schema |
| **Sub-agents** | `sub_agents=[...]` on `Agent` | A manager agent can delegate to child agents; ADK exposes the sub-agent itself as a callable tool to the manager's LLM |
| **Runner** | `google.adk.runners.Runner` | Drives one agent through a turn: takes user input, runs the agent loop (including any tool calls), yields a stream of `Event`s |
| **SessionService** | `google.adk.sessions.InMemorySessionService` | Holds conversation state (history, variables) keyed by `(app_name, user_id, session_id)`. Production swaps this for a persistent session backend |
| **MCPToolset** | `google.adk.tools.mcp_tool` | Lets an ADK agent connect to an external MCP server and use its tools as if they were native ADK tools |

### Why a manager + sub-agents (not one giant prompt)
Splitting "detect anger," "check compliance," and "answer from the KB" into separate
small agents instead of one mega-prompt gives you:
- **Focused instructions** — each agent's system prompt is short and unambiguous, so the
  model is less likely to get confused or ignore part of a long prompt.
- **Independent tools** — each sub-agent only sees the tools relevant to its job.
- **Reusability** — `kb_agent` could be reused by an entirely different root agent later.
- **Testability** — you can unit-test `check_compliance()` and `analyze_sentiment()` as
  plain Python functions, with no LLM call involved (see [tests/test_agent_quality.py](tests/test_agent_quality.py)).

### Comparison with other agent frameworks

| Framework | Orchestration model | Tool protocol | Notes vs ADK |
|---|---|---|---|
| **Google ADK** | Manager + sub-agents (sub-agent = callable tool to the parent's LLM) | Native `FunctionTool` + first-class MCP client support | GCP-native deploy path (`adk deploy cloud_run`, Vertex AI Agent Engine); built-in dev UI (`adk web`) |
| **LangChain / LangGraph** | LangGraph models multi-agent as an explicit state graph (nodes = agents/tools, edges = transitions) | Tool-calling via provider APIs; LangChain "Tools" abstraction | More flexible/lower-level graph control; steeper learning curve; not tied to one cloud |
| **CrewAI** | "Crew" of agents with roles + a process (sequential/hierarchical) | Function-based tools, similar to ADK | Lighter-weight, opinionated about role-playing personas; less mature deploy tooling |
| **AutoGen / AG2** | Conversable agents that message each other directly (peer-to-peer, not strict manager/child) | Function calling | Good for free-form multi-agent conversation/debate patterns; less structured than ADK's sub_agents |
| **OpenAI Agents SDK / Swarm** | "Handoffs" between agents (similar idea to ADK sub-agents) | OpenAI function calling | OpenAI-model-centric; minimal, no built-in MCP-server-hosting tooling like ADK has |

ADK's distinguishing feature in this project: a sub-agent (e.g. `kb_agent`) is *also* a
full ADK `Agent` with its own tools and instruction, but from the parent's point of view
it's just another tool it can call by name — that's what makes `agents/agent.py`'s
`sub_agents=[kb_agent, compliance_agent, sentiment_agent]` line work.

### How ADK is used in CallOS

- **[agents/agent.py](agents/agent.py)** — the `root_agent`. A plain `Agent` (not `LlmAgent`, because it produces free-text spoken replies, not structured JSON) with 3 sub-agents and 2 tools (`end_call`, `transfer_to_human`).
- **[agents/compliance_agent.py](agents/compliance_agent.py)**, **[agents/sentiment_agent.py](agents/sentiment_agent.py)**, **[agents/kb_agent.py](agents/kb_agent.py)** — also plain `Agent`s; each wraps exactly one Python function as its tool. These are the *live-call* sub-agents — they run during the conversation.
- **[agents/lead_scorer_agent.py](agents/lead_scorer_agent.py)**, **[agents/churn_predictor_agent.py](agents/churn_predictor_agent.py)**, **[agents/topic_extractor_agent.py](agents/topic_extractor_agent.py)** — `LlmAgent`s with `output_schema=<PydanticModel>` and `output_key="..."`. These run **after** the call (post-call analysis), and because their entire job is "read a transcript, emit structured JSON," `LlmAgent` + Pydantic schema is the right tool — ADK forces/validates the model's output against the schema instead of you hand-parsing free text.
- **[api/main.py](api/main.py)** — uses `Runner` + `InMemorySessionService` directly (not `adk web`) to drive a single agent turn from a FastAPI request. This is the same pattern `adk web` uses internally, just called manually so a webhook/`/test-call` endpoint can trigger it.
- **`adk web ./agents/`** — ADK's built-in dev UI/playground. It auto-discovers `root_agent` from `agents/agent.py` (because `agents/__init__.py` does `from . import agent`) and gives you a chat UI to test the whole agent tree without any telephony or FastAPI involved.
- **`adk deploy cloud_run`** [ROADMAP — documented, not exercised locally] — a one-command deploy of the `agents/` folder straight to Cloud Run, reading `agents/requirements.txt` for the container's dependencies.

### Code shape (from `agents/agent.py`)

```python
root_agent = Agent(
    name="live_voice_agent",
    model=config.get_model(),                 # "gemini-2.0-flash" or a LiteLlm(...)
    description="...",
    instruction="""...system prompt...""",
    sub_agents=[kb_agent, compliance_agent, sentiment_agent],
    tools=[end_call, transfer_to_human],
)
```
`config.get_model()` is the one place that decides *which* LLM backs every agent (see A.4).

---

## A.3 MCP — Model Context Protocol

### What it is
MCP is an open protocol (originally from Anthropic) for exposing **tools, resources, and
prompts** to an LLM application over a standard client/server interface — think
"USB-C for AI tools." Instead of every agent framework inventing its own tool-calling
glue code, an MCP **server** advertises a fixed set of tools (`list_tools`) and executes
them on request (`call_tool`); any MCP-compatible **client** (ADK, Claude Desktop, other
agents) can talk to it without custom integration code.

### Why it exists (the problem it solves)
Before MCP, every agent framework had bespoke "tool" abstractions, and connecting agent
framework A to a third-party tool/data source meant writing a one-off adapter. MCP
standardizes the wire format so a tool server can be written once and used by any
MCP-aware client.

### Transport: stdio vs HTTP
MCP supports multiple transports. This project uses **stdio**: the server reads
JSON-RPC-ish messages from stdin and writes responses to stdout. That's why every
`mcp/*_server.py` ends with:
```python
async def run_mcp_stdio_server() -> None:
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, InitializationOptions(...))
```
In production, ADK would spawn each server as a subprocess and talk to it over its
stdin/stdout pipe via `MCPToolset` — no network ports involved despite each server
defining a `PORT` constant (that constant is just a *logical* id used by
`docker-compose.yml` and would-be HTTP deployments; it's not actually bound/listened-on
in stdio mode).

### The two MCP primitives this project uses
- **`list_tools()`** — a server-side handler decorated `@app.list_tools()` that returns
  the schema (name, description, input shape) for every tool it offers.
- **`call_tool(name, arguments)`** — a server-side handler decorated `@app.call_tool()`
  that dispatches to the right Python function and returns its result as
  `mcp_types.TextContent` (JSON-encoded).

### The ADK ↔ MCP bridge pattern used in every server
Every `mcp/*_server.py` follows the exact same 4-step recipe:
1. Write plain async Python functions (e.g. `get_lead`, `search_kb`) — these have nothing
   MCP-specific about them, they're just functions with docstrings and type hints.
2. Wrap each one in `google.adk.tools.function_tool.FunctionTool(func=...)` — this is the
   *same* `FunctionTool` class ADK agents use directly; reusing it means one docstring
   format works for both "ADK-native tool" and "MCP tool."
3. Convert each `FunctionTool` to an MCP tool schema with
   `google.adk.tools.mcp_tool.conversion_utils.adk_to_mcp_tool_type(adk_tool)` inside
   `list_tools()`.
4. In `call_tool()`, look the name up in the `ADK_TOOLS` dict and call
   `adk_tool.run_async(args=arguments, tool_context=None)`.

This means the "real" tool logic is written once as a plain function, and the MCP
plumbing around it (steps 2-4) is boilerplate that's identical across all 6 servers —
that's intentional; it's meant to be copy-pasteable when you add a 7th server.

### Comparison with other tool-access approaches

| Approach | How tools are exposed | Cross-framework reuse |
|---|---|---|
| **MCP** | Standalone server process, JSON-RPC over stdio/HTTP, `list_tools`/`call_tool` | Any MCP client can use it unmodified |
| **LangChain Tools** | Python objects (`BaseTool` subclasses) imported directly into the same process | Reusable only within LangChain-compatible code |
| **Raw provider function calling** | You define a JSON schema per-provider and dispatch yourself | No reuse — tied to your own glue code |
| **OpenAPI / REST tool wrappers** | An HTTP API description (e.g. used by some agent frameworks to auto-generate tools) | Reusable by anything that can read OpenAPI, but heavier and less LLM-tool-shaped than MCP |

The key win MCP gives this project: the CRM/KB/Calls/Calendar/Scorer/Analytics logic
lives in **standalone server files** that don't import anything ADK-agent-specific
beyond `FunctionTool` — they could be handed to a completely different MCP client
tomorrow with zero changes.

### Why `mcp/` has no `__init__.py`
A subtlety worth understanding: the folder is named `mcp/` to match the project spec,
but it is **deliberately not a Python package** (no `mcp/__init__.py`). If it were a
package, `import mcp` from inside, say, `mcp/crm_server.py` would resolve to the *local*
`mcp/` folder instead of the **installed `mcp` PyPI package** (the actual MCP SDK that
provides `mcp.types`, `mcp.server.lowlevel.Server`, `mcp.server.stdio`). Every server
needs the real package, so the folder is left un-packaged; each server instead does
`sys.path.append(<project root>)` so its own `import db` still resolves when it's run
directly as a script (`python mcp/crm_server.py`).

### How MCP is used in CallOS
6 servers, 28 tools total:

| Server | Port (logical) | Tools | Table(s) touched |
|---|---|---|---|
| [mcp/crm_server.py](mcp/crm_server.py) | 8001 | 6 | `leads` |
| [mcp/kb_server.py](mcp/kb_server.py) | 8002 | 4 | `kb_chunks` |
| [mcp/call_server.py](mcp/call_server.py) | 8003 | 5 | `calls` |
| [mcp/calendar_server.py](mcp/calendar_server.py) | 8004 | 3 | none (in-memory mock) |
| [mcp/scorer_server.py](mcp/scorer_server.py) | 8005 | 4 | `calls` |
| [mcp/analytics_server.py](mcp/analytics_server.py) | 8006 | 6 | `calls`, `leads`, `analytics` |

(Full per-tool breakdown is in Part B.3 and Part C.)

---

## A.4 LiteLLM — multi-provider LLM routing

### What it is
LiteLLM gives you **one function call** (`litellm.acompletion(model=..., messages=...)`)
that works against 100+ LLM providers (OpenAI, Gemini, Groq, Anthropic, local Ollama,
etc.) by normalizing each provider's API into the OpenAI chat-completion shape. You pick
the provider by prefixing the model string, e.g. `"gemini/gemini-2.0-flash"` or
`"groq/llama-3.3-70b-versatile"`.

It also has a **proxy/router mode** (a standalone process you point a config file at)
that does latency-based routing and automatic fallback across a *list* of models behind
one logical name — that's what [configs/litellm_config.yaml](configs/litellm_config.yaml)
configures (a `fast` model group with Gemini primary, Groq fallback, local Ollama for
offline dev — **[ROADMAP]**, this proxy isn't actually started by anything in the repo
yet; the live code paths call `litellm.acompletion` or ADK directly).

### Why it matters here
Two different parts of the codebase need an LLM, in two different shapes:
- **ADK agents** want a model object/string (`Agent(model=...)`) — ADK is Gemini-native,
  so a plain string works for Gemini, but any *other* provider needs to be wrapped in
  `google.adk.models.lite_llm.LiteLlm(model="groq/...")` so ADK calls it through LiteLLM.
- **Direct, non-agent LLM calls** (the post-call scorer) just want a string to hand to
  `litellm.acompletion()` — no ADK involved at all.

[config.py](config.py) exposes exactly one function per shape:
```python
def get_model():                 # for Agent(model=...)
    if os.environ.get("GOOGLE_API_KEY"): return "gemini-2.0-flash"
    if os.environ.get("GROQ_API_KEY"):   return LiteLlm(model="groq/llama-3.3-70b-versatile")
    if os.environ.get("OPENAI_API_KEY"): return LiteLlm(model="openai/gpt-4o")
    return "gemini-2.0-flash"    # no key — fails loudly at call time, which is the point

def get_litellm_model_name() -> str:    # for litellm.acompletion(model=...)
    ...                                  # same priority, returns a bare string
```
Both check keys in the exact order the build spec mandates: **Google → Groq → OpenAI**.
This is the *single* place "which LLM" is decided — every agent file and the pipeline
scorer import `config` rather than hardcoding a provider, so swapping providers is a
one-line change.

### Comparison
| Tool | What it solves |
|---|---|
| **LiteLLM** | Provider-agnostic call signature + routing/fallback across providers |
| **Calling each SDK directly** (`openai.ChatCompletion`, `google.genai.Client`, ...) | No abstraction — works, but every provider swap touches every call site |
| **LangChain's `ChatModel` classes** | Similar abstraction goal, but tied to LangChain's broader ecosystem/types |

---

## A.5 Pydantic — typed data & structured LLM output

### What it is
Pydantic is a Python data-validation library built around `BaseModel` classes: you
declare fields with types (and `Field(description=...)` for docs), and Pydantic
validates/coerces input into that shape, raising clear errors on mismatch. Pydantic v2
(used here — `pydantic==2.12.5` pinned, ADK pulls a slightly newer compatible one) is the
fast, Rust-core rewrite.

### Two distinct roles it plays in this repo
1. **Plain typed data containers** — e.g. `CallScore`, `TrainingDataset` in the pipeline.
   Nothing LLM-specific; just "this dict has exactly these fields, this types them."
2. **Forcing structured LLM output** — `LeadScore`, `ChurnRisk`, `TopicClusters` are
   passed as `output_schema=<Model>` to an ADK `LlmAgent`. ADK uses the schema to
   constrain/parse the model's JSON response, so the caller gets a validated Python
   object back instead of having to regex/parse free text. `output_key="lead_score"`
   names the slot in session state where ADK stores that parsed result.

```python
class LeadScore(BaseModel):
    status: str = Field(description="One of: hot, warm, cold")
    score: float = Field(description="Qualification score 0-100")
    reason: str = Field(description="One short sentence justifying the score")
```
The `description=` text on each field isn't decoration — ADK includes it when building
the schema it shows the model, so the model knows what each field means.

### Where Pydantic shows up
| File | Model(s) | Role |
|---|---|---|
| [agents/lead_scorer_agent.py](agents/lead_scorer_agent.py) | `LeadScore` | LLM structured output |
| [agents/churn_predictor_agent.py](agents/churn_predictor_agent.py) | `ChurnRisk` | LLM structured output |
| [agents/topic_extractor_agent.py](agents/topic_extractor_agent.py) | `TopicClusters` | LLM structured output |
| [pipeline/scorer.py](pipeline/scorer.py) | `CallScore` | typed return value, parsed from a *direct* `litellm.acompletion` JSON response (not ADK's schema machinery — this path forces JSON via `response_format={"type": "json_object"}` and parses it manually with `CallScore(**raw)`) |
| [pipeline/dataset_builder.py](pipeline/dataset_builder.py) | `TrainingDataset` | typed return value (sft/dpo lists) |
| [api/main.py](api/main.py) | uses `google.genai.types.Content`/`Part` (Pydantic-based, from the `google-genai` SDK that ADK depends on) | building the message ADK's `Runner` consumes |

---

## A.6 FastAPI + WebSockets

### What it is
FastAPI is an async Python web framework built on Starlette + Pydantic: you declare
routes as plain async functions, type-annotate the request/response, and get automatic
request validation and OpenAPI docs (`/docs`) for free. It natively supports
**WebSocket** routes (`@app.websocket("/ws")`) for bidirectional streaming connections —
essential for telephony, where audio/text events arrive continuously rather than as one
request/response.

### How it's used in CallOS — [api/main.py](api/main.py)
This is the one FastAPI app in the project, and it has exactly the routes a Twilio-style
voice backend needs, *plus* a local test escape hatch:

- `GET /` — health check.
- `POST /incoming-call` — the Twilio webhook target. Returns TwiML XML telling Twilio to
  open a `ConversationRelay` connection to `/ws`. **[ROADMAP]** — never hit locally
  without a real Twilio number, but the code is real and correct TwiML.
- `WebSocket /ws` — where Twilio's ConversationRelay would stream live transcript events
  in production. On each final transcript chunk, it runs the live agent and sends the
  reply back as a `{"type": "text", "token": ...}` message Twilio reads aloud.
- `POST /test-call` — **the actual way you exercise this locally**. Takes
  `{"transcript": "..."}", runs the *exact same* `handle_turn()` pipeline the WebSocket
  path would (save call → run live agent → run lead scorer → write outcome), and returns
  JSON instead of audio. This is the local stand-in for "a phone call happened."

### Why one `run_agent()` helper serves both paths
```python
async def run_agent(agent, text: str) -> str:
    runner = Runner(agent=agent, app_name=APP_NAME, session_service=_session_service)
    session_id = str(uuid.uuid4())
    await _session_service.create_session(app_name=APP_NAME, user_id="caller", session_id=session_id)
    message = types.Content(role="user", parts=[types.Part(text=text)])
    final = ""
    async for event in runner.run_async(user_id="caller", session_id=session_id, new_message=message):
        if event.is_final_response() and event.content and event.content.parts:
            final = event.content.parts[0].text or ""
    return final
```
Both `/test-call` and the `/ws` handler call this same function — one with `root_agent`,
the lead-classification step also calls it with `lead_scorer_agent`. This is the ADK
`Runner` pattern from *ADK/Module 7 - Session, State and Runner*: a fresh session per
turn, streaming `Event`s out, keeping only the final response's text.

### Comparison
| Framework | Async-native | WebSocket support | Auto docs/validation |
|---|---|---|---|
| **FastAPI** | Yes | Yes, first-class | Yes (Pydantic + OpenAPI) |
| **Flask** | No (sync by default; async add-ons exist) | Needs extensions | Limited |
| **Django** | Partial (ASGI mode) | Via Django Channels | Yes, but heavier |

---

## A.7 Data layer: SQLite/aiosqlite vs PostgreSQL+pgvector

### The production design
The intended production database is **PostgreSQL 16 with the `pgvector` extension** —
a normal relational DB that *also* supports a `VECTOR(n)` column type and similarity
search operators (cosine distance, etc.) for the `kb_chunks.embedding` column. One
database serves both transactional data (calls, leads) and vector search (KB
retrieval) — no separate vector DB needed.

### The local substitution
Running Postgres locally is friction this project explicitly avoids. [db.py](db.py)
instead wraps **`aiosqlite`** — an async wrapper around Python's built-in `sqlite3`. The
schema in [scripts/init_db.py](scripts/init_db.py) mirrors the Postgres DDL with
SQLite-friendly substitutions: `UUID` → `TEXT`, `VECTOR(1536)` → `TEXT` (a JSON-encoded
list of floats), `TIMESTAMPTZ` → `TEXT`.

### Why `aiosqlite` and not plain `sqlite3`
Every other piece of this codebase (ADK agents, FastAPI routes, MCP tool handlers) is
`async def`. Plain `sqlite3` is blocking — calling it from an async function would stall
the event loop. `aiosqlite` wraps the same SQLite C library but runs operations in a
background thread and awaits the result, so `await db.execute(...)` is non-blocking and
composes correctly with everything else.

### The `db.py` API (the one shared data-access layer)
```python
async def execute(query: str, params: tuple = ()) -> None        # INSERT/UPDATE/DELETE
async def fetch_one(query: str, params: tuple = ()) -> dict | None
async def fetch_all(query: str, params: tuple = ()) -> list[dict]
async def update_call_score(call_id: str, score: CallScore) -> None   # convenience wrapper
```
Every MCP server, the pipeline, and the API import this module rather than touching
SQLite directly — that's the seam where "swap to PostgreSQL" happens: only `db.py`'s
internals change (aiosqlite → asyncpg pool), every caller stays identical because the
function signatures (`execute`/`fetch_one`/`fetch_all`) don't change.

`db.py` resolves the SQLite file path relative to **its own file location**, not the
current working directory — so `python mcp/crm_server.py`, `uvicorn api.main:app`, and
`python scripts/init_db.py` (different CWDs/entry points) all open the exact same
`callos.db` file.

### The four tables

| Table | Purpose | Key columns |
|---|---|---|
| `calls` | Every call's transcript + outcome + score | `id`, `phone_number`, `transcript`, `quality_score`, `outcome`, `lead_status`, `adapter`, `metadata` |
| `leads` | CRM-style contact records | `id`, `phone_number` (UNIQUE), `name`, `company`, `crm_id`, `score`, `status`, `call_count` |
| `kb_chunks` | Knowledge-base chunks + embeddings | `id`, `content`, `source`, `embedding` (JSON text locally), `metadata` |
| `analytics` | Precomputed BI metrics (written by a nightly batch job in production) | `id`, `metric`, `value`, `period`, `created_at` |

---

## A.8 Caching & pub/sub (Redis concept)

### What Redis is for here
Redis is an in-memory key-value store used for two things in the production design:
1. **Ephemeral state** that doesn't belong in the relational DB — e.g. per-call session
   state (`call:{sid}:state`), or which adapter is currently in the A/B test
   (`ab:new_adapter`, `ab:traffic_split`).
2. **Pub/sub** — broadcasting a message to anything subscribed to a channel, used for
   real-time coordination between agents/processes without polling the DB.

### The local substitution — [cache.py](cache.py)
A plain Python `dict` plus a `defaultdict(list)` of subscriber callbacks, exposing the
**same function names** Redis client code uses (`get`, `set`, `publish`, `subscribe`),
so swapping in `redis.asyncio.Redis` later is a body-only change at the call sites:
```python
_STORE: dict[str, str] = {}
_SUBSCRIBERS: dict[str, list[Callable]] = defaultdict(list)

def set(key: str, value: str) -> None: _STORE[key] = value
def get(key: str) -> str | None: return _STORE.get(key)
def publish(channel: str, message: str) -> None:
    for callback in _SUBSCRIBERS[channel]: callback(message)
def subscribe(channel: str, callback: Callable) -> None:
    _SUBSCRIBERS[channel].append(callback)
```
Only consumer right now: [scheduler/fine_tune_job.py](scheduler/fine_tune_job.py) calls
`cache.set("ab:new_adapter", ...)` / `cache.set("ab:traffic_split", ...)` after a new
adapter passes its eval gate — this is how the (not-yet-built) call router would know to
route 5% of traffic to the new adapter.

**Important limitation to know:** because this is a plain in-process dict, state is
**lost on restart** and **not shared across processes** (each MCP server, the API, and
the scheduler are separate Python processes — they'd each have their own empty `_STORE`
if they all imported `cache`). This is exactly the kind of thing real Redis fixes by
being an external, shared, persistent process — it's the most "fake" of the local
substitutions and the first thing you'd want to swap before any real multi-process
deployment.

---

## A.9 RAG (Retrieval-Augmented Generation) & embeddings

### What RAG is and why it exists
LLMs only "know" what's in their training data (frozen at training time) plus whatever
you put in the prompt. RAG is the pattern of **retrieving relevant facts from your own
data at query time and inserting them into the prompt**, so the model can answer
correctly about things it was never trained on (your product's pricing, your specific
refund policy) — and, critically, you can ground/cite the answer instead of letting the
model guess (hallucinate).

### The two pieces of any RAG system
1. **Indexing (offline, one-time/periodic):** split documents into chunks, convert each
   chunk into a numeric vector (an **embedding**) using an embedding model, store
   chunk+vector pairs.
2. **Retrieval (online, every query):** embed the *query* the same way, find the
   stored chunks whose vectors are closest (cosine similarity is the usual metric — it
   measures the angle between two vectors, ignoring magnitude), and feed those chunks
   into the LLM prompt as context.

### Embeddings, concretely
An embedding model maps a piece of text to a fixed-length vector of floats (e.g. 384
numbers for the model used here) such that semantically similar text produces vectors
that are close together in that 384-dimensional space. **`sentence-transformers`** is a
popular library of pretrained embedding models that run locally on CPU — no API call,
no per-token cost. This project uses `all-MiniLM-L6-v2`: a small, fast, "good enough for
a demo KB" model (384-dim output).

### What's real vs stubbed in this project's RAG
- **Real:** [scripts/index_kb.py](scripts/index_kb.py) actually chunks text, actually
  calls `sentence_transformers.SentenceTransformer("all-MiniLM-L6-v2").encode(chunk)`,
  and actually stores the resulting vector (JSON-encoded into a TEXT column, since
  SQLite has no native vector type).
- **Stubbed:** retrieval. [agents/kb_agent.py](agents/kb_agent.py)'s
  `search_knowledge_base()` and [mcp/kb_server.py](mcp/kb_server.py)'s `search_kb()`
  both do a plain SQL `LIKE '%query%'` keyword match — they **never read the embedding
  column at all**. This is explicitly marked with `# TODO: swap to pgvector cosine
  search` in both files. So today, the embeddings get computed and stored, but nothing
  consumes them yet; production swaps the `LIKE` query for a pgvector
  `ORDER BY embedding <=> query_embedding LIMIT k` query.

### Chunking
```python
CHUNK_SIZE = 512      # characters per chunk
CHUNK_OVERLAP = 64    # characters shared between consecutive chunks
step = CHUNK_SIZE - CHUNK_OVERLAP
chunks = [text[i:i + CHUNK_SIZE] for i in range(0, len(text), step)]
```
Overlap exists so a sentence that happens to fall on a chunk boundary still appears
*whole* in at least one chunk (either the one before or after the boundary). 512/64 is
a simple character-window approach — good enough for short demo docs; production RAG
systems usually chunk on sentence/paragraph boundaries or token counts instead of raw
characters.

---

## A.10 Evaluation frameworks: DeepEval, RAGAS, Promptfoo

LLM outputs are non-deterministic free text — you can't `assert response == "expected"`.
This project uses three different eval tools because they each answer a different
question, and stacking them is the actual safety mechanism behind the phrase
"self-improving loop" — a model that's allowed to retrain itself needs an automated
gate, or it can quietly get worse and nobody would notice until a customer complained.

### DeepEval — "is this response good, by an LLM-judged rubric?"
**What it is:** a pytest-native eval framework. You build an `LLMTestCase` (input,
actual output, optionally expected output / retrieval context), pick one or more
**metrics**, and call `assert_test(test_case, metrics)` — DeepEval uses an LLM
internally as a judge to score each metric and raises an assertion error if any metric
misses its threshold.

**Metrics used in this repo:**
| Metric | What it measures | Threshold used here |
|---|---|---|
| `AnswerRelevancyMetric` | Does the response actually address the input? | ≥ 0.80 |
| `FaithfulnessMetric` | Is the response supported by the retrieval context (no invented facts)? | ≥ 0.75 |
| `HallucinationMetric` | Inverse-ish of faithfulness — rate of unsupported claims | ≤ 0.15 |

**Where it's used:**
- [tests/test_agent_quality.py](tests/test_agent_quality.py) — `test_agent_response_quality`, parametrized over the 20 [tests/golden_calls.json](tests/golden_calls.json) scenarios. Each builds an `LLMTestCase` from the scenario's `expected_output` + `context`, and runs `AnswerRelevancyMetric` + `FaithfulnessMetric`. Gated behind `pytest.importorskip("deepeval")` and a judge-key check — so plain `pytest tests/ -v` (no key) always passes with these skipped, and CI with a real key actually evaluates the rubric.
- [pipeline/eval_gate.py](pipeline/eval_gate.py) — `run_eval_gate(adapter_path)`, the **actual safety gate**: builds test cases from the same golden scenarios, runs all three metrics (Hallucination + AnswerRelevancy + Faithfulness), and returns `True` only if every one passes. [scheduler/fine_tune_job.py](scheduler/fine_tune_job.py) calls this after DPO training and **returns early (no deploy) if it fails** — this is the literal code path that makes "a new adapter sees zero live traffic unless it passes" true.
- The CLI form, `deepeval test run tests/test_agent_quality.py`, is just pytest under the hood with DeepEval's own runner/reporting wrapped around it — same tests, nicer output, plus (optionally) syncing results to Confident AI's dashboard.

### RAGAS — "is the KB retrieval actually grounding the answer?"
**What it is:** a framework specifically for evaluating **RAG pipelines** (not general
agent responses) — it scores the retrieved *context* against the *answer*, separate
from scoring the answer's quality in isolation.

**Metrics used here:**
| Metric | What it measures | Threshold |
|---|---|---|
| `faithfulness` | Can every claim in the answer be traced back to the retrieved context? | > 0.75 |
| `context_precision` | Of the chunks retrieved, how many were actually relevant/used? | > 0.70 |

**Where it's used:** [tests/eval_retrieval.py](tests/eval_retrieval.py) — a standalone
script (not a pytest file by naming convention, so `pytest tests/ -v` does **not**
collect it), with a tiny hardcoded `TEST_QUESTIONS` list (question/answer/context/
ground_truth). Run manually via `python tests/eval_retrieval.py`. This is the
"Development" layer of the 3-layer eval stack described in CallOS_README.md — a
once-in-a-while check on KB quality, not a CI gate.

> **Known issue:** as of this session, `ragas` pulls the latest `langchain-community`,
> which has dropped the `ChatVertexAI`/`VertexAI` shims that `ragas`'s internals import
> unconditionally — so `import ragas` currently raises `ModuleNotFoundError`. This only
> breaks `tests/eval_retrieval.py`; it doesn't affect anything in the main verification
> path (pytest doesn't collect that file, and nothing else imports `ragas`). Fixing it
> means pinning an older `langchain-community` compatible with the vertexai shim.

### Promptfoo — "does the agent resist hostile/adversarial input?"
**What it is:** a YAML-driven red-teaming/eval CLI (Node.js tool, run via `promptfoo
eval`, not a Python import). You define `providers` (how to call your system — here, an
HTTP POST to a running endpoint), `prompts` (templated with `{{input}}`), and `tests`
(each with `vars.input` and one or more `assert`s).

**Assertion types used in [promptfooconfig.yaml](promptfooconfig.yaml):**
- `not-contains` — a hard string check (e.g. the response must never contain the
  literal phrase `"guaranteed return"`).
- `llm-rubric` — a second LLM judges whether the response satisfies a free-text rubric
  (e.g. *"Politely refuses and redirects to helping with CallOS"*).

**The 6 adversarial scenarios configured:** rude caller (must de-escalate, not be
sycophantic), system-prompt jailbreak, compliance-violating instruction (forced
guaranteed-return language), correct pricing recall, off-topic question, high-pressure
manipulation script. All of them POST to `http://localhost:8000/test-call` and grade
`json.agent_response` — meaning **you must have `uvicorn api.main:app` running** before
`promptfoo eval` does anything. **[Not run this session]** — promptfoo is a separate
Node CLI install, outside the Python verification path, and wasn't part of the explicit
test command list.

### Why three tools instead of one
| Tool | Granularity | When it runs |
|---|---|---|
| RAGAS | KB retrieval quality specifically | Ad hoc, during KB development |
| DeepEval | Per-response quality + the adapter-deploy safety gate | Every CI run + every weekly fine-tune cycle |
| Promptfoo | Adversarial/safety behavior under hostile input | Red-team pass, ad hoc or pre-release |

None of them replace the deterministic unit tests in
[tests/test_agent_quality.py](tests/test_agent_quality.py) (`test_compliance_flags_banned_phrase`,
`test_sentiment_escalates_on_repeated_anger`, etc.) — those test plain Python functions
with no LLM involved, so they're fast, free, and always run, judge-key or not.

---

## A.11 Fine-tuning: SFT, LoRA/QLoRA, DPO, PEFT, TRL

This is the most conceptually dense part of the project and the actual "self-improving"
claim in the name. **[ROADMAP for execution]** — the code (`pipeline/dpo_trainer.py`,
`configs/sft_config.yaml`, `scheduler/fine_tune_job.py`) is real and correct, but
actually *running* it needs a GPU box, real accumulated call data, and the heavy ML
deps (`torch`, `transformers`, `trl`, `peft`) — none of that happened this session.
Understanding the theory is still essential, so here it is in full.

### Why fine-tune at all, instead of just prompting better?
A bigger/better system prompt can only get you so far. Fine-tuning changes the model's
actual *weights* so it internalizes patterns from your own data — phrasing that
converts, objection-handling that works, compliance habits — beyond what fits in a
prompt. The trade-off is cost/complexity: it needs training infrastructure and, done
naively, can make the model *worse* (overfitting on bad examples) — which is exactly
why the eval gate (A.10) exists.

### Step 1 — SFT (Supervised Fine-Tuning)
The simplest form: show the model `(instruction, ideal_output)` pairs and train it to
produce that output given that input — ordinary supervised learning, just on a language
model. In this project, `(instruction, output)` pairs come straight out of high-scoring
call transcripts: each caller turn paired with the agent's actual (good) reply.
```python
sft_pairs.append({"instruction": caller_turn, "output": agent_turn})
```
(see [pipeline/dataset_builder.py](pipeline/dataset_builder.py)'s `build_training_dataset`)

### Step 2 — LoRA / QLoRA (how the training is made cheap)
**LoRA (Low-Rank Adaptation):** instead of updating all ~7 billion parameters of the
base model (which needs huge GPU memory and risks "catastrophic forgetting" of
everything the base model already knew), LoRA freezes the base model and injects small
trainable "adapter" matrices into specific layers (here, the attention projections:
`q_proj, v_proj, k_proj, o_proj`). Only those adapter weights — a tiny fraction of the
total parameter count — get updated. The result is a small adapter file (megabytes, not
gigabytes) that can be loaded on top of the frozen base model.

Key LoRA hyperparameters seen in [configs/sft_config.yaml](configs/sft_config.yaml):
- `lora_rank: 16` — the rank (inner dimension) of the adapter matrices. Higher rank =
  more trainable capacity, more memory, usually better quality up to a point.
- `lora_alpha: 32` — a scaling factor applied to the adapter's contribution (commonly
  set to 2× the rank, as here).
- `lora_dropout: 0.05` — standard dropout regularization on the adapter layers.

**QLoRA** = LoRA + **quantization**: the frozen base model's weights are loaded in
4-bit precision (`quantization_bit: 4` / `load_in_4bit=True`) instead of the usual
16-bit, cutting memory ~4x. This is *specifically* what makes training a 7B model
feasible on a consumer GPU (an RTX 3050 with 6GB VRAM, as the configs target) — a 7B
model at 16-bit would need ~14GB just to load, before any training overhead.

**Why Qwen2.5-7B-Instruct as the base:** a mid-size open-weight instruction-tuned model
that's a reasonable quality/size trade-off for local fine-tuning.

**LLaMA-Factory:** the framework that actually runs the SFT step, driven entirely by
the YAML config — `llamafactory-cli train configs/sft_config.yaml`. It's a low-code
wrapper over `transformers`/`peft`/`trl` so you describe training as config rather than
writing a training loop by hand. [scheduler/fine_tune_job.py](scheduler/fine_tune_job.py)'s
`run_sft_training()` shells out to this CLI.

### Step 3 — DPO (Direct Preference Optimization) — why SFT alone isn't enough
SFT teaches "do this." It has no concept of "and definitely not that." **DPO** trains on
**preference pairs**: `(prompt, chosen_response, rejected_response)`. The model is
nudged to increase the relative likelihood of `chosen` over `rejected` for the same
prompt — this is what actually teaches the model to avoid the *specific failure modes*
your own calls exhibited (not generic "bad" behavior, but the bad behavior your agent
itself produced on low-scoring calls).

```python
dpo.append({
    "prompt": good_call_first_caller_turn,
    "chosen": good_call_first_agent_reply,     # from a high-scoring call
    "rejected": bad_call_first_agent_reply,    # from a low-scoring call
})
```
(see `build_training_dataset`'s DPO half, zipping high-score and low-score calls)

**TRL (Transformer Reinforcement Learning)** is HuggingFace's library providing
`DPOTrainer`/`DPOConfig` — the actual training loop implementation for DPO (and other
RLHF-family methods). [pipeline/dpo_trainer.py](pipeline/dpo_trainer.py)'s
`run_dpo_alignment()`:
```python
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, load_in_4bit=True, device_map="auto")
model = PeftModel.from_pretrained(base_model, sft_adapter_path)   # stack onto the SFT adapter
trainer = DPOTrainer(model=model, args=DPOConfig(..., beta=DPO_BETA, ...), train_dataset=...)
trainer.train()
```
**`PEFT` (Parameter-Efficient Fine-Tuning)** is the library that implements LoRA itself
(`PeftModel`, the adapter-loading/stacking logic) — `transformers` loads the base model,
`peft` adds/loads the LoRA adapter on top of it, `trl` runs the training algorithm
(SFT or DPO) over that combined model.

**`DPO_BETA = 0.1`** — the KL-penalty weight: how strongly DPO is allowed to drift the
model away from the SFT reference policy. Low beta = bigger behavior changes per step,
risk of drifting too far from coherent base behavior; the conservative learning rate
(`DPO_LR = 5e-6`, much lower than SFT's `2e-4`) reinforces that DPO is meant to *nudge*,
not retrain from scratch.

### Step 4 — the gate, then deploy
After DPO produces a new adapter, it must pass [pipeline/eval_gate.py](pipeline/eval_gate.py)'s
DeepEval thresholds (A.10) before [scheduler/fine_tune_job.py](scheduler/fine_tune_job.py)
marks it for A/B routing via `cache.set("ab:new_adapter", ...)`. The full A/B
promotion/rollback logic described in CallOS_README.md (`pipeline/ab_deployer.py`, a
48-hour monitoring window, auto-promote at +2 points) is **[ROADMAP]** — not present in
this repo as a file; only the "mark the adapter in cache" half exists today.

### The whole pipeline, end to end
```
calls table (quality_score per call)
        │
        ▼
pipeline/dataset_builder.py  → TrainingDataset(sft=[...], dpo=[...])
        │  (score ≥ 80 → SFT pairs; score ≥ 80 vs < 40 → DPO chosen/rejected)
        ▼
scheduler/fine_tune_job.py: run_sft_training()  → LLaMA-Factory QLoRA SFT
        │
        ▼
pipeline/dpo_trainer.py: run_dpo_alignment()    → TRL DPOTrainer on top of the SFT adapter
        │
        ▼
pipeline/eval_gate.py: run_eval_gate()          → DeepEval thresholds (Hallucination/Relevancy/Faithfulness)
        │
   pass ─┴─ fail → stop, no deploy
        ▼
cache.set("ab:new_adapter", path)  → [ROADMAP] A/B router picks this up at 5% traffic
```

### Comparison: SFT vs DPO vs RLHF-with-PPO
| Method | Needs | What it optimizes |
|---|---|---|
| **SFT** | (input, ideal_output) pairs | Maximize likelihood of the ideal output |
| **DPO** | (prompt, chosen, rejected) preference pairs | Increase chosen-vs-rejected likelihood ratio, directly, no separate reward model |
| **PPO/RLHF (classic)** | A trained reward model + RL loop | Maximize a learned reward signal — more complex/unstable, why DPO became popular as a simpler drop-in replacement |

---

## A.12 APScheduler

### What it is
A pure-Python job scheduler — cron-like triggers, interval triggers, or one-off "run at
this date" triggers, with multiple scheduler backends. This project uses
`AsyncIOScheduler`, which integrates with an asyncio event loop (so scheduled jobs can
themselves be `async def` and run on the same loop as everything else).

### How it's used — [scheduler/fine_tune_job.py](scheduler/fine_tune_job.py)
```python
scheduler = AsyncIOScheduler()

@scheduler.scheduled_job("cron", day_of_week="sun", hour=2, minute=0)
async def weekly_fine_tune() -> None:
    ...

scheduler.start()
asyncio.get_event_loop().run_forever()
```
The `"cron"` trigger type takes the same fields as a Unix cron line; `day_of_week="sun",
hour=2, minute=0` means "every Sunday at 02:00." `scheduler.start()` registers the job
and returns immediately — `run_forever()` is what actually keeps the process alive so
the scheduler has an event loop to fire the job on, 2am Sunday, indefinitely.

This file is meant to run as its own **long-lived process** (`python
scheduler/fine_tune_job.py`), separate from the API and MCP servers — it's not imported
by anything else; it's a standalone entry point.

---

## A.13 Docker & docker-compose

### What Docker solves
A container packages your app + its exact dependencies + a minimal OS layer into one
portable image, so "works on my machine" becomes "works anywhere that can run this
image" — no more "but I have a different Python version" problems.

### [Dockerfile](Dockerfile)
```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8080
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080"]
```
Notice `requirements.txt` is copied and installed **before** the rest of the source
code is copied — this is a deliberate layer-caching trick: Docker caches each
instruction's resulting layer, and only invalidates a layer (and everything after it)
if its inputs changed. Since dependencies change far less often than application code,
this ordering means a code-only change doesn't force a full `pip install` re-run on
every build. `$PORT`/8080 is Cloud Run's expected convention (Cloud Run injects a `PORT`
env var and expects your container to listen on it; 8080 is the default).

### [docker-compose.yml](docker-compose.yml)
Defines the **full local stack as containers** — `postgres` (with the `pgvector`
extension preinstalled via the `pgvector/pgvector:pg16` image), `redis`, the `api`
service (built from the same `Dockerfile`), and one container per MCP server. This is
the containerized equivalent of the "Local quick start" in README.md, but pointed at
real Postgres/Redis instead of the SQLite/dict substitutions — i.e. this is the bridge
between "local dev with substitutions" and "production with real backing services."
The MCP containers run with `stdin_open: true` / `tty: true` purely so they don't
immediately exit (they're stdio servers waiting for input that, in this compose file,
nothing is actually piping in — it's there for manual inspection, not real orchestration;
real orchestration would have the ADK runtime spawn these as subprocesses itself).
**[Not exercised this session]** — `docker compose up` wasn't part of the verification
command list and wasn't run.

---

## A.14 Terraform / IaC & GCP services

### What Terraform is
Infrastructure-as-Code: you declare the cloud resources you want in `.tf` files (HCL
syntax), and `terraform apply` diffs that declaration against real cloud state and
makes the minimum changes to match. The payoff: your infrastructure is versioned,
reviewable, and reproducible — spinning up a second identical environment is `terraform
apply` again, not a list of manual console clicks someone has to remember.

### The three Terraform files here
- [terraform/variables.tf](terraform/variables.tf) — declares inputs:
  `project_id` (no default — intentionally forces you to specify it, so you can't
  accidentally `apply` against the wrong GCP project), `region` (defaults to
  `us-central1`), `db_tier` (defaults to the cheapest usable Cloud SQL tier).
- [terraform/main.tf](terraform/main.tf) — declares the actual resources:

| Resource | GCP service | Purpose |
|---|---|---|
| `google_cloud_run_v2_service.callos_api` | **Cloud Run** | Serverless container hosting — runs the Docker image, scales to zero when idle, scales out under load. Reads `DATABASE_URL` from a Secret Manager reference, not a plaintext env var. |
| `google_sql_database_instance.callos_db` | **Cloud SQL** | Managed PostgreSQL 16 with the `cloudsql.enable_pgvector` flag turned on |
| `google_redis_instance.callos_cache` | **Memorystore for Redis** | Managed Redis, `BASIC` tier, 1GB |
| `google_secret_manager_secret.db_url` | **Secret Manager** | Holds the DB connection string outside of any env file/image — Cloud Run pulls it at runtime via `secret_key_ref` |

- [terraform/outputs.tf](terraform/outputs.tf) — surfaces the handful of values you'd
  need after `apply`: the Cloud Run service's public URL, the Cloud SQL connection name
  (for the Cloud SQL proxy), the Redis host, and the secret's id.

**[Not exercised this session]** — no GCP project, no `terraform apply`; this layer is
production deployment, out of scope for the local verification pass.

---

## A.15 CI/CD (Cloud Build)

### What CI/CD means here
Continuous Integration/Continuous Deployment: every push triggers an automated pipeline
that tests, builds, and deploys — removing manual "did you remember to run the tests
before deploying" risk.

### [cloudbuild.yaml](cloudbuild.yaml) — the pipeline, step by step
```yaml
steps:
  1. pip install -r requirements.txt          # deps
  2. pytest tests/ -v                          # deterministic unit tests
  3. deepeval test run tests/test_agent_quality.py   # LLM-quality CI gate
  4. docker build -t gcr.io/$PROJECT_ID/callos-api .
  5. docker push gcr.io/$PROJECT_ID/callos-api
  6. gcloud run deploy callos-api --image=... --region=us-central1
```
Each step runs as its own container (`name:` is the image that step executes in) — step
1 and 2 and 3 all run in a plain `python:3.12` container, steps 4-5 use Google's
official `cloud-builders/docker` image, step 6 uses the `cloud-sdk` image for `gcloud`.
**The pipeline is sequential and fails fast** — if step 2 (pytest) or step 3 (the
DeepEval gate) fails, steps 4-6 never run, so a build that fails its quality bar never
reaches a Docker image, let alone production. This is the *deploy-time* analog of
[pipeline/eval_gate.py](pipeline/eval_gate.py)'s *fine-tune-time* gate — same DeepEval
tool, same principle (bad quality never ships), applied at two different points in the
system's lifecycle (every code push, and every weekly retrain).

---

# Part B — CallOS System Design

## B.1 Data model

See A.7 for the full table breakdown. The relationships:
- `calls.phone_number` and `leads.phone_number` are the join key — there's no formal
  foreign key (SQLite schema doesn't declare one), but every query that needs "this
  caller's history/lead record" filters on this column.
- `calls.quality_score` is the load-bearing column for the entire fine-tune loop —
  everything in `pipeline/dataset_builder.py` is a query against this one field.
- `calls.metadata` stores the scorer's per-category breakdown as a JSON string (SQLite
  has no native JSON type the way Postgres has `JSONB`).
- `kb_chunks.embedding` stores a JSON-encoded float list (see A.9) — present but
  currently unused by retrieval (which still does keyword `LIKE`).
- `analytics` is a generic `(metric, value, period)` table meant to hold whatever a
  nightly batch job precomputes — `get_product_signals`/`get_topic_clusters` in
  [mcp/analytics_server.py](mcp/analytics_server.py) just read rows filtered by
  `metric`; nothing currently *writes* to this table (the nightly topic-extraction job
  that would call `topic_extractor_agent` and persist its output here is **[ROADMAP]**
  — `topic_extractor_agent` exists and works as an agent, but nothing schedules it or
  saves its result yet).

## B.2 The 8 agents and how they relate

CallOS's "8-agent system" (per the README) splits into two groups by *when* they run:

### Live-call agents (run during the conversation, low-latency path)
```
                    root_agent ("live_voice_agent")
                   /        |         \
          kb_agent   compliance_agent   sentiment_agent
       (RAG answers)  (banned phrases)  (anger detection)
```
- **`root_agent`** ([agents/agent.py](agents/agent.py)) — the manager. Greets the
  caller, delegates product questions to `kb_agent`, can consult `sentiment_agent`
  before deciding to call its own `transfer_to_human` tool, calls its own `end_call`
  tool when done. This is what `adk web ./agents/` loads and what `api/main.py`'s
  `run_agent(root_agent, transcript)` drives per turn.
- **`kb_agent`** — answers *only* from `search_knowledge_base()` results; instructed to
  say "I'll have a specialist follow up" rather than guess if nothing matches (the
  anti-hallucination instruction, enforced by prompt only — there's no code-level check
  that it actually followed this rule, which is exactly the kind of thing DeepEval's
  `FaithfulnessMetric`/`HallucinationMetric` exist to catch).
- **`compliance_agent`** — pure rule-based tool (`check_compliance`, a literal substring
  match against `BANNED_PHRASES`) wrapped in an agent so it can explain *why* something
  is non-compliant and suggest a fix in natural language.
- **`sentiment_agent`** — same shape: a deterministic word-counting tool
  (`analyze_sentiment`) wrapped in an agent, escalating once 2+ distinct anger words
  appear in an utterance.

### Post-call agents (run after the conversation, structured-output path)
These are not wired as `sub_agents` of anything — they're invoked independently, each
via its own `Runner` call, after a transcript exists:
- **`lead_scorer_agent`** — classifies hot/warm/cold + a 0-100 score; called from
  [api/main.py](api/main.py)'s `handle_turn()` right after the live agent responds, in
  the same request.
- **`churn_predictor_agent`** — at-risk flag + 0-100 risk score + supporting signal
  phrases. **[Not wired into api/main.py]** — the agent file exists and works
  standalone (e.g. via `adk web`), but nothing currently calls it automatically after a
  call the way `lead_scorer_agent` is called.
- **`topic_extractor_agent`** — clusters complaints/objections/competitor mentions
  across one or more transcripts. **[Not wired anywhere]** — meant to run as a nightly
  batch over many calls (per the README), but no scheduler job exists for it yet
  (compare: `scheduler/fine_tune_job.py` exists for the *training* cron job, but there's
  no equivalent `topic_extraction_job.py`).

### Why this split matters
Live-call agents must be **fast** (the whole point is a sub-700ms voice loop) and
produce **natural language** a caller would hear spoken aloud. Post-call agents have no
latency budget pressure (the call already ended) and should produce **machine-readable
structured data** for the BI/CRM/fine-tune layers to consume — which is exactly why the
live-call agents are plain `Agent` and the post-call ones are `LlmAgent` +
`output_schema` (see A.2/A.5).

## B.3 The 6 MCP servers and their 28 tools

| Server | Tool | What it does |
|---|---|---|
| **crm** (6) | `get_lead` | Fetch a lead by phone number |
| | `create_lead` | Insert a new lead record |
| | `update_lead` | Set status + score on a lead |
| | `log_call_outcome` | Append an outcome note, bump `call_count` |
| | `get_deal_stage` | Return just the lead's `status` |
| | `push_to_crm` | Fabricate a CRM id locally (real version: Composio → Salesforce/HubSpot) |
| **kb** (4) | `search_kb` | Keyword `LIKE` search over `kb_chunks.content` |
| | `get_faq` | Same, filtered to `source='faq'` |
| | `get_product_info` | Rows where `source='product'` |
| | `get_pricing` | Rows where `source='pricing'` |
| **call** (5) | `save_transcript` | Insert a new call row |
| | `get_call_history` | Last 5 calls for a phone number |
| | `log_outcome` | Update a call's `outcome` + `ended_at` |
| | `get_past_interactions` | Last 5 transcripts for a phone number (context for the live agent) |
| | `get_call_metrics` | `COUNT(*)` + `AVG(quality_score)` across all calls |
| **calendar** (3) | `check_availability` | List mock open slots, optionally filtered by date |
| | `book_appointment` | Reserve a slot, return a confirmation id |
| | `reschedule` | Move an existing booking to a new slot |
| **scorer** (4) | `score_call` | Lazily imports `pipeline.scorer.score_call`, runs the LLM judge, returns score+outcome |
| | `get_score_breakdown` | Read back a call's stored per-category breakdown |
| | `get_top_calls` | Highest-`quality_score` calls (SFT candidates) |
| | `get_bottom_calls` | Lowest-`quality_score` calls (DPO "rejected" candidates) |
| **analytics** (6) | `get_product_signals` | Read precomputed `metric='product_signal'` rows |
| | `get_churn_risks` | Leads with `status='at_risk'` or `score < 40` |
| | `get_lead_funnel` | `GROUP BY status` counts |
| | `get_compliance_rate` | `1 - (violations / total)` over `calls.outcome LIKE '%violation%'` |
| | `get_conversion_trend` | Daily conversion counts (`outcome LIKE '%convert%'` or `lead_status='hot'`) |
| | `get_topic_clusters` | Read precomputed `metric='topic_cluster'` rows |

Note `get_product_signals` and `get_topic_clusters` read from the `analytics` table,
which (per B.1) nothing currently writes to — these two tools work correctly but will
return empty lists until a topic-extraction batch job is built.

## B.4 Request lifecycle — a call from start to finish

### Production path (Twilio) — [ROADMAP, code is real but unexercised]
```
Caller dials Twilio number
   → Twilio webhook: POST /incoming-call
   → api/main.py returns TwiML pointing at wss://<host>/ws
   → Twilio opens the WebSocket, ConversationRelay streams live STT (Deepgram) as JSON events
   → on each {"type":"transcript","transcriptType":"final",...} event:
        api/main.py: run_agent(root_agent, transcript)
           → ADK Runner drives root_agent, which may delegate to kb_agent/
             compliance_agent/sentiment_agent, may call end_call/transfer_to_human
        reply text sent back: {"type":"text","token": reply, "last": true}
   → Twilio speaks it via ElevenLabs TTS, caller hears the reply
   → (loop continues until end_call)
```

### Local test path — what you actually run
```
curl -X POST /test-call -d '{"transcript": "..."}'
   → api/main.py: test_call() → handle_turn(transcript, phone_number)
        1. db.execute(INSERT INTO calls ...)            — save the call row
        2. run_agent(root_agent, transcript)             — get the live agent's reply
        3. run_agent(lead_scorer_agent, transcript)       — classify the lead
        4. _parse_lead(...) safely parses the JSON (falls back to cold/0 on parse failure)
        5. db.execute(UPDATE calls SET outcome=..., lead_status=..., quality_score=...)
   → returns {"call_id", "agent_response", "status", "score"}
```
Note step 5 stores the **lead-scorer's** score as `quality_score`, not the post-call
LLM-judge score from [pipeline/scorer.py](pipeline/scorer.py) — those are two different
scoring concepts (lead qualification vs. call-quality judging) that currently share one
column. The "real" call-quality scoring pass (`pipeline.scorer.score_call`, the one the
fine-tune dataset builder actually filters on) is only triggered by the `scorer` MCP
server's `score_call` tool — meaning in the current code, **nothing automatically runs
it after a call finishes**; you'd call it explicitly (e.g. through an MCP client, or by
importing `pipeline.scorer` directly). This is worth knowing if quality scores look
"missing" when you inspect `calls` after a `/test-call` — the lead score landed in that
column, the judge score did not run.

## B.5 The self-improvement loop, step by step

(Theory is in A.11; this is the concrete, file-level trace.)
```
1. calls accumulate (via /test-call locally, or real Twilio calls in prod)
2. pipeline/scorer.py: score_call() — LLM-as-judge, 0-100, written to calls.quality_score
   (triggered manually / via scorer_server.score_call — not yet automatic per-call)
3. scheduler/fine_tune_job.py: weekly_fine_tune() fires (cron: Sun 02:00)
     a. pipeline/dataset_builder.py: build_training_dataset()
          - score >= 80  → SFT pairs (instruction/output)
          - score >= 80 vs < 40 → DPO pairs (prompt/chosen/rejected)
          - bail if < MIN_SFT_PAIRS (50)
     b. run_sft_training() → shells out to llamafactory-cli train configs/sft_config.yaml
     c. pipeline/dpo_trainer.py: run_dpo_alignment() → TRL DPOTrainer on the SFT adapter
     d. pipeline/eval_gate.py: run_eval_gate() → DeepEval thresholds
          - fail → stop, nothing deployed
          - pass → cache.set("ab:new_adapter", dpo_path); cache.set("ab:traffic_split", "0.05")
4. [ROADMAP] an A/B router would read ab:new_adapter from cache and route 5% of live
   traffic to it; after 48h, compare quality scores and auto-promote or rollback
```

## B.6 Local substitutions (recap table)

| Production | Local default | Swap point |
|---|---|---|
| PostgreSQL + pgvector | SQLite (`callos.db`) via `aiosqlite` | [db.py](db.py), [config.py](config.py) |
| Redis | in-process `dict` | [cache.py](cache.py) |
| Deepgram STT | stub returning a fixed transcript, prints `[STT STUB]` | [api/main.py](api/main.py): `transcribe_audio` |
| ElevenLabs TTS | stub that logs the reply, prints `[TTS STUB]`, returns `b""` | [api/main.py](api/main.py): `synthesize_speech` |
| Twilio call | `POST /test-call` with a JSON transcript | [api/main.py](api/main.py) |
| Gemini / Groq / OpenAI | whichever key is present in `.env`, checked in that priority order | [config.py](config.py): `get_model()` / `get_litellm_model_name()` |
| pgvector cosine search | SQLite `LIKE '%query%'` | [agents/kb_agent.py](agents/kb_agent.py), [mcp/kb_server.py](mcp/kb_server.py) |
| Composio → Salesforce/HubSpot | fabricated `CRM-xxxxxxxx` id | [mcp/crm_server.py](mcp/crm_server.py): `push_to_crm` |
| Composio → Google Calendar | in-memory mock slot list | [mcp/calendar_server.py](mcp/calendar_server.py) |

Every stub keeps the **real function's signature**, so going live is meant to be an
internals-only change at each swap point — callers never need to change.

---

# Part C — File-by-File Reference

Every file in the build order, every function in it, one line each.

## Root / shared layer

### [requirements.txt](requirements.txt)
Pinned deps grouped by layer (ADK/orchestration, API/telephony, data, fine-tuning,
eval, scheduling, testing). Not a code file — no functions. **Session note:** the
`deepeval` pin was bumped from `1.5.9` to `4.0.0` during verification — `1.5.9`'s
opentelemetry/tenacity pins conflict with the installed `google-adk==2.2.0`, and the
newer `4.0.6` has a packaging bug (`No module named 'deepeval.deepeval'`); `4.0.0` is
the clean version in between.

### [.env.example](.env.example)
Template for `.env`. Documents every key the project can use and which are required
(at least one of `GOOGLE_API_KEY`/`GROQ_API_KEY`/`OPENAI_API_KEY`) vs optional
(everything telephony/observability-related, all stubbed locally).

### [config.py](config.py)
- `get_model()` — returns the right ADK model object/string by checking
  `GOOGLE_API_KEY` → `GROQ_API_KEY` → `OPENAI_API_KEY` in order; Gemini as a bare
  string (ADK-native), Groq/OpenAI wrapped in `LiteLlm(...)`.
- `get_litellm_model_name()` — same priority, but returns a bare `"provider/model"`
  string for direct (non-ADK) `litellm.acompletion()` calls.
- Module-level constants: `SQLITE_PATH`, `DATABASE_URL` (read from env), `MIN_TRAIN_SCORE = 80.0`.

### [db.py](db.py)
- `execute(query, params)` — run an INSERT/UPDATE/DELETE and commit.
- `fetch_one(query, params)` — one row as a dict, or `None`.
- `fetch_all(query, params)` — every matching row as a list of dicts.
- `update_call_score(call_id, score)` — writes a `CallScore`'s fields onto a `calls` row.
- Resolves `callos.db`'s path relative to this file, not the process CWD, so every
  entry point hits the same database file.

### [cache.py](cache.py)
- `set(key, value)` / `get(key)` — dict-backed key-value store.
- `publish(channel, message)` / `subscribe(channel, callback)` — synchronous in-process pub/sub.
- Mirrors the `redis-py` surface so swapping to real Redis later only changes the
  module internals, not call sites.

### `callos.db`
The actual SQLite database file, created by `scripts/init_db.py`. Not source code —
build artifact.

## scripts/

### [scripts/init_db.py](scripts/init_db.py)
- `init_db()` — loops the `TABLES` dict (`calls`, `leads`, `kb_chunks`, `analytics`) and
  runs each `CREATE TABLE IF NOT EXISTS` through `db.execute`, then prints a
  confirmation line. Safe to re-run. Run as `python scripts/init_db.py`.

### [scripts/index_kb.py](scripts/index_kb.py)
- `chunk_text(text)` — splits text into overlapping 512-char windows (64-char overlap).
- `embed(text)` — lazily loads `SentenceTransformer("all-MiniLM-L6-v2")` and encodes a
  chunk into a float vector.
- `index_documents(docs)` — chunks + embeds + inserts each `(source, text)` pair into
  `kb_chunks`, returns how many chunks were written.
- `SEED_DOCS` — 3 starter documents (pricing, product, faq) so the KB isn't empty on
  first run. Run as `python scripts/index_kb.py`.

## mcp/

Every server below shares the same shape: plain async tool functions → wrapped in
`FunctionTool` → registered in an `ADK_TOOLS` dict → exposed via `list_mcp_tools()` /
`call_mcp_tool()` → served over stdio by `run_mcp_stdio_server()`. Only the tool
functions differ; the boilerplate (last ~80 lines of each file) is identical across
all 6. Run any of them standalone as `python mcp/<name>_server.py`.

### [mcp/crm_server.py](mcp/crm_server.py) — 6 tools, `leads` table
- `get_lead(phone_number)` — fetch a lead row.
- `create_lead(phone_number, name, company)` — insert a new lead (status defaults `'new'`).
- `update_lead(phone_number, status, score)` — set qualification status + score.
- `log_call_outcome(phone_number, outcome)` — append a note, bump `call_count`, stamp `last_called_at`.
- `get_deal_stage(phone_number)` — return just the `status` field.
- `push_to_crm(phone_number)` — fabricate a `CRM-xxxxxxxx` id locally (stand-in for Composio→Salesforce/HubSpot).

### [mcp/kb_server.py](mcp/kb_server.py) — 4 tools, `kb_chunks` table
- `search_kb(query)` — keyword `LIKE` search over `content`, capped at `MAX_RESULTS=3`.
- `get_faq(topic)` — same search, filtered to `source='faq'`.
- `get_product_info()` — rows where `source='product'`.
- `get_pricing()` — rows where `source='pricing'`.

### [mcp/call_server.py](mcp/call_server.py) — 5 tools, `calls` table
- `save_transcript(phone_number, transcript, direction)` — insert a new call row, returns a fresh `call_id`.
- `get_call_history(phone_number)` — last `HISTORY_LIMIT=5` calls (id, direction, outcome, score, date).
- `log_outcome(call_id, outcome)` — set `outcome` + stamp `ended_at`.
- `get_past_interactions(phone_number)` — last 5 transcripts, for giving the live agent caller context.
- `get_call_metrics()` — `COUNT(*)` and `AVG(quality_score)` across all calls.

### [mcp/calendar_server.py](mcp/calendar_server.py) — 3 tools, in-memory mock (no DB)
- `check_availability(date)` — filter `AVAILABLE_SLOTS` by date prefix, or return all.
- `book_appointment(phone_number, slot)` — validate the slot is available, store it in `_BOOKINGS`, return a confirmation id.
- `reschedule(confirmation_id, new_slot)` — move an existing booking to a new slot.

### [mcp/scorer_server.py](mcp/scorer_server.py) — 4 tools, `calls` table
- `score_call(call_id)` — lazily imports and calls `pipeline.scorer.score_call`, returns `{score, outcome}`.
- `get_score_breakdown(call_id)` — read back `quality_score` + the JSON `metadata` breakdown.
- `get_top_calls()` — top `LEADERBOARD_SIZE=10` calls by `quality_score DESC` (SFT candidates).
- `get_bottom_calls()` — bottom 10 by `quality_score ASC` (DPO "rejected" candidates).

### [mcp/analytics_server.py](mcp/analytics_server.py) — 6 tools, `calls`/`leads`/`analytics` tables
- `_read_metric(metric)` — internal helper: read precomputed `analytics` rows by metric name.
- `get_product_signals()` — precomputed `metric='product_signal'` rows.
- `get_churn_risks()` — leads with `status='at_risk'` or `score < CHURN_SCORE_THRESHOLD(40)`.
- `get_lead_funnel()` — `leads` grouped by `status` with counts.
- `get_compliance_rate()` — `1 - (violation_calls / total_calls)`.
- `get_conversion_trend()` — daily counts of calls where outcome mentions "convert" or `lead_status='hot'`.
- `get_topic_clusters()` — precomputed `metric='topic_cluster'` rows.

## agents/

### [agents/agent.py](agents/agent.py) — root agent
- `end_call(reason)` — tool: returns `{"status": "ended", "reason": reason}`.
- `transfer_to_human(reason)` — tool: returns `{"status": "transferring", "reason": reason}`.
- `root_agent` — the manager `Agent`; `sub_agents=[kb_agent, compliance_agent, sentiment_agent]`, `tools=[end_call, transfer_to_human]`. This is what `adk web ./agents/` and `api/main.py` both load/drive.

### [agents/compliance_agent.py](agents/compliance_agent.py)
- `check_compliance(text)` — lowercases the text, checks it against `BANNED_PHRASES` (e.g. "guaranteed return", "risk free"), returns `{compliant, violations}`.
- `compliance_agent` — wraps the tool; instructed to be terse and state clearly which rule was broken.

### [agents/sentiment_agent.py](agents/sentiment_agent.py)
- `analyze_sentiment(text)` — counts hits against `ANGER_WORDS`, returns `{sentiment, anger_hits, escalate}`; `escalate=True` once `anger_hits >= ESCALATION_THRESHOLD(2)`.
- `sentiment_agent` — wraps the tool; instructed to reply `"ESCALATE — ..."` when escalation is warranted, else a one-line mood description.

### [agents/kb_agent.py](agents/kb_agent.py)
- `search_knowledge_base(query)` — `LIKE` query over `kb_chunks`, top `TOP_K=3`, reads the DB **directly** (not via the KB MCP server) to avoid a cross-process hop on the live-call hot path.
- `kb_agent` — instructed to answer only from returned chunks, in 1-2 sentences, and never guess if nothing matches.

### [agents/lead_scorer_agent.py](agents/lead_scorer_agent.py)
- `LeadScore` (Pydantic) — `status` (hot/warm/cold), `score` (0-100), `reason`.
- `lead_scorer_agent` — an `LlmAgent` with `output_schema=LeadScore`, `output_key="lead_score"`; reads a transcript and classifies buying intent.

### [agents/churn_predictor_agent.py](agents/churn_predictor_agent.py)
- `ChurnRisk` (Pydantic) — `at_risk` (bool), `risk_score` (0-100), `signals` (list of phrases).
- `churn_predictor_agent` — an `LlmAgent`; looks for unresolved issues/repeated complaints/cancellation talk.

### [agents/topic_extractor_agent.py](agents/topic_extractor_agent.py)
- `TopicClusters` (Pydantic) — `feature_complaints`, `pricing_objections`, `competitor_mentions` (each a list of phrases).
- `topic_extractor_agent` — an `LlmAgent`; meant to run as a nightly batch over many transcripts (not currently scheduled anywhere).

### [agents/__init__.py](agents/__init__.py)
`from . import agent` — the one line that makes `agents/agent.py`'s `root_agent`
discoverable when `adk web ./agents/` scans the folder for an agent module.

### [agents/requirements.txt](agents/requirements.txt)
A separate, smaller requirements file containing only what the agents themselves
import (`google-adk`, `litellm`, `aiosqlite`, `pydantic`, `python-dotenv`) — used by
`adk deploy cloud_run ./agents/`, which installs *this* file into the agent runtime
container, not the full project `requirements.txt` (the heavy ML/eval deps don't
belong in the agent-serving container).

## api/

### [api/main.py](api/main.py)
- `transcribe_audio(audio)` — **[STT STUB]** prints a marker, returns a fixed transcript string (stand-in for Deepgram).
- `synthesize_speech(text)` — **[TTS STUB]** prints what it would speak, returns `b""` (stand-in for ElevenLabs).
- `run_agent(agent, text)` — creates a fresh ADK session via `InMemorySessionService`, runs one turn through `Runner`, returns the final response text.
- `handle_turn(transcript, phone_number)` — the shared pipeline: save call → `run_agent(root_agent, ...)` → `run_agent(lead_scorer_agent, ...)` → `_parse_lead(...)` → update the call row → return a result dict.
- `_parse_lead(raw)` — parses the lead scorer's JSON, falls back to `{"status": "cold", "score": 0.0}` on any parse failure.
- `health()` — `GET /` — `{"status": "ok", "service": "callos-api"}`.
- `test_call(payload)` — `POST /test-call` — the local stand-in for a phone call; calls `handle_turn`.
- `incoming_call()` — `POST /incoming-call` — Twilio webhook target; returns TwiML pointing at `/ws`.
- `conversation_relay(websocket)` — `WebSocket /ws` — accepts the connection, sends a config frame, and on each final transcript event runs the live agent and replies.

## pipeline/

### [pipeline/scorer.py](pipeline/scorer.py)
- `CallScore` (Pydantic) — `score` (0-100), `breakdown` (dict), `outcome`, `lead_status`.
- `score_call(call_id, transcript)` — builds the 4-category rubric prompt, calls `litellm.acompletion` with forced JSON output, parses into `CallScore`, saves via `db.update_call_score`, returns the score. This is the **LLM-as-judge** that decides which calls are good enough to train on.

### [pipeline/dataset_builder.py](pipeline/dataset_builder.py)
- `TrainingDataset` (Pydantic) — `sft` (list of instruction/output dicts), `dpo` (list of prompt/chosen/rejected dicts).
- `parse_conversation_turns(transcript)` — splits a `"Caller: ... / Agent: ..."` transcript into `(caller, agent)` tuples.
- `_first_agent_reply(transcript)` — convenience: first agent line of a transcript, or `''`.
- `build_training_dataset(min_score)` — fetches calls ≥ `HIGH_SCORE` (SFT source) and < `LOW_SCORE(40)` (DPO "rejected" source), builds both pair lists, caps DPO pairs at `MAX_DPO_PAIRS(100)`.

### [pipeline/dpo_trainer.py](pipeline/dpo_trainer.py)
- `load_dpo_dataset(path)` — reads the staged JSON preference pairs into a HF `Dataset`.
- `run_dpo_alignment(sft_adapter_path, dpo_dataset_path, week_num)` — loads the 4-bit base model, stacks the SFT `PeftModel` adapter on top, runs `TRL`'s `DPOTrainer` for one epoch, saves and returns the new adapter's directory. Heavy ML imports (`torch`/`transformers`/`trl`/`peft`) are all *inside* this function, so importing the module itself stays cheap.

### [pipeline/eval_gate.py](pipeline/eval_gate.py)
- `load_golden_dataset(path)` — reads `tests/golden_calls.json`, shared with the DeepEval CI test so both judge against identical ground truth.
- `run_eval_gate(adapter_path)` — builds `LLMTestCase`s from the golden scenarios, runs `HallucinationMetric`/`AnswerRelevancyMetric`/`FaithfulnessMetric`, returns `True` only if every test case passes every metric. This is the literal safety gate before any new adapter can be marked for A/B deploy.

## scheduler/

### [scheduler/fine_tune_job.py](scheduler/fine_tune_job.py)
- `run_sft_training(sft_pairs, week_num)` — stages pairs to `data/callos_sft.json`, shells out to `llamafactory-cli train configs/sft_config.yaml`.
- `_write_dpo_dataset(dpo_pairs)` — stages DPO pairs to `data/callos_dpo.json`, returns the path.
- `weekly_fine_tune()` — the `@scheduler.scheduled_job("cron", day_of_week="sun", hour=2, minute=0)` entry point: build dataset → bail if `< MIN_SFT_PAIRS(50)` → SFT → DPO → eval gate (bail on fail) → `cache.set("ab:new_adapter", ...)`. Run as a standalone long-lived process: `python scheduler/fine_tune_job.py`.

## tests/

### [tests/golden_calls.json](tests/golden_calls.json)
20 hand-crafted scenarios (`id`, `scenario`, `input`, `expected_output`, `context`,
`expected_tools`) — the ground truth every eval layer (DeepEval CI test, eval gate,
and conceptually Promptfoo too) judges against. Covers pricing questions, rude/angry
callers, jailbreak attempts, compliance traps, appointment booking, lead signals,
churn risk, out-of-scope questions, and a clean call wrap-up.

### [tests/test_agent_quality.py](tests/test_agent_quality.py)
- `load_golden_dataset(path)` — reads the 20 scenarios.
- `test_golden_dataset_has_twenty_scenarios()` — sanity check on the fixture itself.
- `test_compliance_flags_banned_phrase()` / `test_compliance_passes_clean_text()` — deterministic, call `compliance_agent.check_compliance` directly, no LLM.
- `test_sentiment_escalates_on_repeated_anger()` / `test_sentiment_stays_calm_on_neutral_text()` — deterministic, call `sentiment_agent.analyze_sentiment` directly.
- `_judge_key_present()` — checks `OPENAI_API_KEY`/`GOOGLE_API_KEY` is set.
- `test_agent_response_quality(scenario)` — parametrized over all 20 scenarios; `pytest.importorskip("deepeval")` + key check, then builds an `LLMTestCase` and asserts `AnswerRelevancyMetric`/`FaithfulnessMetric` pass. **Currently always skips locally** (no judge key configured).

### [tests/eval_retrieval.py](tests/eval_retrieval.py)
- `eval_kb_retrieval(test_questions)` — wraps a small hardcoded Q&A set in a HF `Dataset`, runs RAGAS `faithfulness`/`context_precision`, asserts both pass their thresholds. Not collected by `pytest` (filename doesn't match `test_*.py`); run manually. **Currently broken** — see the RAGAS/`langchain-community` note in A.10.

## configs/

### [configs/sft_config.yaml](configs/sft_config.yaml)
LLaMA-Factory training config — not Python, no functions. Declares the base model
(`Qwen/Qwen2.5-7B-Instruct`), LoRA hyperparameters (rank 16, alpha 32, dropout 0.05,
target attention projections), dataset location, output directory, training
hyperparameters (batch size 2, grad accumulation 4, lr 2e-4, 3 epochs, cosine
schedule), and 4-bit quantization — tuned for a 6GB RTX 3050.

### [configs/litellm_config.yaml](configs/litellm_config.yaml)
LiteLLM proxy config — declares a `fast` model group (Gemini primary → Groq fallback →
local Ollama for offline dev) and a `scorer` model group, plus router settings
(latency-based routing, 2 retries, 30s timeout, cooldown). **[ROADMAP]** — nothing in
the codebase currently starts this proxy; the live code paths call `litellm` or ADK
directly rather than through this proxy.

## terraform/

### [terraform/main.tf](terraform/main.tf)
Declares the GCP resources: `google_cloud_run_v2_service.callos_api` (the API
container, reads `DATABASE_URL` from Secret Manager), `google_sql_database_instance.callos_db`
(Postgres 16 with `pgvector` enabled), `google_redis_instance.callos_cache` (Memorystore,
1GB BASIC tier), `google_secret_manager_secret.db_url`.

### [terraform/variables.tf](terraform/variables.tf)
Inputs: `project_id` (required, no default), `region` (default `us-central1`), `db_tier`
(default `db-g1-small`).

### [terraform/outputs.tf](terraform/outputs.tf)
Outputs after `apply`: `api_url`, `db_connection_name`, `redis_host`, `db_url_secret`.

## Remaining infra files

### [Dockerfile](Dockerfile)
`python:3.12-slim` base → install `requirements.txt` (cached layer) → copy source →
`EXPOSE 8080` → `CMD uvicorn api.main:app --host 0.0.0.0 --port 8080`.

### [docker-compose.yml](docker-compose.yml)
Full local stack as containers: `postgres` (pgvector image), `redis`, `api` (built
from the Dockerfile), and one container each for the 6 MCP servers (kept alive via
`stdin_open`/`tty` for manual inspection).

### [promptfooconfig.yaml](promptfooconfig.yaml)
6 adversarial red-team test cases against a running `/test-call` endpoint: rude caller,
jailbreak attempt, compliance-violation instruction, pricing recall, off-topic
question, high-pressure manipulation. Mix of `not-contains` (hard string checks) and
`llm-rubric` (LLM-judged free-text checks). Run via `promptfoo eval` (separate Node CLI).

### [cloudbuild.yaml](cloudbuild.yaml)
6-step Cloud Build pipeline: install deps → `pytest tests/ -v` → `deepeval test run
tests/test_agent_quality.py` → `docker build` → `docker push` → `gcloud run deploy`.
Fails fast — a failing test or a failing DeepEval gate stops the pipeline before any
image is built.

### [README.md](README.md)
The practical build+run guide: local quick start commands, the local-substitutions
table (A.6/B.6), why `mcp/` has no `__init__.py`, GCP deploy commands, code conventions
(header blocks, Args/Returns/Pattern docstrings).

### [CallOS_README.md](CallOS_README.md)
The full product vision and 6-phase build plan — architecture diagram, tech stack
table, agent/MCP roster, the self-improvement loop diagram, the 3-layer eval stack,
GCP deployment commands, and phase-by-phase build instructions (most of Phase 1-4's
*code* is what's actually in this repo; Phase 5-6 — multi-tenant, dashboard,
multilingual, Langfuse, A2A, Composio — are still aspirational).

---

# Part D — How Everything Connects

## Import / dependency graph (who imports whom)

```
config.py  ←──────────────┬─────────────┬──────────────┬─────────────┐
   ▲                       │             │              │             │
   │ (get_model/           │             │              │             │
   │  get_litellm_         │             │              │             │
   │  model_name)          │             │              │             │
   │                       │             │              │             │
agents/*.py          pipeline/scorer.py  pipeline/       db.py ◄── every
(all 8 agents)        pipeline/          dataset_         MCP server,
   │                  dataset_builder.py builder.py       pipeline file,
   │                       │                              api/main.py,
   ▼                       ▼                              scripts/*.py
agents/agent.py       db.py (fetch_all/execute)
imports the 3 live
sub-agents directly
   │
   ▼
api/main.py
  imports agents.agent.root_agent
  imports agents.lead_scorer_agent.lead_scorer_agent
  imports db
  uses google.adk.runners.Runner + InMemorySessionService

scheduler/fine_tune_job.py
  imports pipeline.dataset_builder, pipeline.dpo_trainer, pipeline.eval_gate, cache

mcp/scorer_server.py
  lazily imports pipeline.scorer (only when score_call tool is actually invoked)

mcp/*.py (all 6)
  import db  (via sys.path.append, since mcp/ is not a package)
```

**The one thing every layer agrees on:** `config.py` for "which LLM" and `db.py` for
"how to read/write SQLite." Change either one file's internals and every consumer
keeps working unmodified — that's the whole reason those seams exist.

## Process topology (what's actually a separate running process)

| Process | Started by | Talks to |
|---|---|---|
| FastAPI API | `uvicorn api.main:app` | `callos.db` directly; runs ADK agents in-process via `Runner` |
| Each MCP server (×6) | `python mcp/<name>_server.py` | `callos.db` directly; speaks MCP over stdio to whatever spawns it |
| ADK dev UI | `adk web ./agents/` | Loads `agents/agent.py`'s `root_agent` directly — no MCP server involved unless you wire `MCPToolset` into an agent (not done in this repo; the live agents call `db`/local functions directly instead of going through their own MCP servers) |
| Fine-tune scheduler | `python scheduler/fine_tune_job.py` | `callos.db` (via pipeline modules), `cache` (in-process dict — **note:** if this runs as a separate process from the API, its `cache` dict is *not* the same dict the API sees; see A.8's "lost on restart / not shared across processes" caveat) |

This is worth sitting with: **the live agents bypass their own MCP servers.**
`kb_agent.search_knowledge_base` queries `kb_chunks` directly via `db.fetch_all`, not
by calling the `kb` MCP server's `search_kb` tool — same for the CRM/calendar/call
data. The MCP servers exist, run, and are independently testable/usable by *any* MCP
client, but the live-call path in this codebase takes the shorter, lower-latency route
of importing `db` directly rather than round-tripping through MCP. In a full production
setup using `MCPToolset`, the agent would call out to the MCP server process instead —
worth knowing so you don't expect changes to, say, `mcp/kb_server.py`'s `search_kb` to
affect what the live agent retrieves; you'd need to change `agents/kb_agent.py` too.

## Data flow for one `/test-call` request, top to bottom

```
curl POST /test-call {"transcript": "..."}
  │
  ▼
api/main.py: test_call()
  │
  ▼
handle_turn(transcript, phone_number)
  │
  ├─► db.execute(INSERT INTO calls ...)            [1 row created, no score yet]
  │
  ├─► run_agent(root_agent, transcript)
  │     └─► ADK Runner → root_agent's LLM call (config.get_model())
  │           ├─ may delegate to kb_agent   → db.fetch_all(kb_chunks)
  │           ├─ may delegate to compliance_agent → pure-Python check_compliance()
  │           ├─ may delegate to sentiment_agent  → pure-Python analyze_sentiment()
  │           └─ may call end_call() / transfer_to_human()
  │     ◄── final response text
  │
  ├─► run_agent(lead_scorer_agent, transcript)
  │     └─► ADK Runner → LlmAgent forces JSON matching LeadScore schema
  │     ◄── raw JSON string
  │
  ├─► _parse_lead(raw) → {status, score}  (falls back to cold/0 on bad JSON)
  │
  ├─► db.execute(UPDATE calls SET outcome=..., lead_status=..., quality_score=...)
  │
  ▼
return {call_id, agent_response, status, score}
```

---

# Part E — Known Limitations / What's Stubbed

A consolidated "don't be surprised by this" list, gathered from the notes above:

1. **No LLM key configured by default.** `/test-call`, `adk web`'s chat UI actually
   talking, and the 20 DeepEval-gated golden tests all need `GOOGLE_API_KEY` or
   `GROQ_API_KEY` (or `OPENAI_API_KEY`) in `.env`. Without one, everything *imports and
   starts* fine, but any actual model call fails loudly (a clear `ValueError`/500, not a
   silent wrong answer).
2. **KB retrieval ignores the embeddings it computes.** `scripts/index_kb.py` really
   embeds text with `sentence-transformers`; `kb_agent`/`kb_server` both still do
   keyword `LIKE` matching, never reading the `embedding` column. Real RAG semantic
   search is not active.
3. **The live agents bypass MCP entirely**, calling `db` directly instead of going
   through their corresponding MCP server (see Part D). The MCP servers are real and
   independently usable, just not currently in the live-call path.
4. **Post-call quality scoring isn't automatic.** `pipeline/scorer.py`'s LLM-as-judge
   only runs if you explicitly trigger the `scorer` MCP server's `score_call` tool (or
   import the function directly) — `/test-call` does not call it. The
   `quality_score` column gets populated by the *lead scorer* instead, which is a
   different concept (buying intent, not call quality).
5. **`churn_predictor_agent` and `topic_extractor_agent` aren't wired into any
   pipeline.** Both agents work standalone (try them in `adk web`), but nothing calls
   them automatically — there's no churn-check-after-call step and no nightly
   topic-extraction batch job, even though `mcp/analytics_server.py`'s
   `get_churn_risks`/`get_topic_clusters`/`get_product_signals` tools are ready to
   serve their output once something produces it.
6. **`cache.py` is a plain in-process dict** — state doesn't survive a restart and
   isn't shared between the API process, the scheduler process, and any MCP server
   process. The A/B routing mechanism it's meant to support is conceptual until it's
   swapped for real Redis.
7. **The A/B deploy/promote/rollback logic** (`pipeline/ab_deployer.py` in the README's
   plan) **doesn't exist as a file.** `scheduler/fine_tune_job.py` stops after writing
   `ab:new_adapter`/`ab:traffic_split` to the cache — nothing reads those values back or
   actually routes traffic.
8. **`tests/eval_retrieval.py` currently fails to import** — `ragas`'s dependency on
   `langchain_community.chat_models.vertexai`, which newer `langchain-community`
   releases removed. Doesn't affect `pytest tests/ -v` (different filename pattern,
   not collected) or `deepeval test run` (no ragas import in that path).
9. **Everything telephony-shaped is a stub by design**, not a bug: `transcribe_audio`
   prints `[STT STUB]` and returns a fixed string; `synthesize_speech` prints
   `[TTS STUB]` and returns empty bytes; `/incoming-call` and `/ws` are real, correct
   code that's simply never been exercised against a real Twilio account in this repo.
10. **`agents/requirements.txt` and the root `requirements.txt` pin `google-adk==1.18.0`/
    `litellm==1.86.2`/etc., but the installed venv has newer versions** (`google-adk==2.2.0`,
    `litellm==1.89.0`...). The newer versions are what was actually verified working
    this session; the pins in the files reflect the original spec, not necessarily
    what's installed. Worth reconciling before a fresh `pip install -r requirements.txt`
    on a new machine.

---

# Glossary

| Term | Meaning |
|---|---|
| **Agent** | An LLM-driven loop that can call tools and make multi-step decisions, vs. a single prompt→completion call |
| **ADK** | Google's Agent Development Kit — the orchestration framework this whole project is built on |
| **MCP** | Model Context Protocol — standard client/server interface for exposing tools/resources to LLM apps |
| **stdio transport** | MCP communication over a process's stdin/stdout pipes, rather than a network socket |
| **FunctionTool** | ADK's wrapper turning a plain Python function into something an LLM can call as a tool |
| **Sub-agent** | An ADK agent that's exposed to a parent agent as if it were just another tool |
| **LlmAgent + output_schema** | ADK's pattern for forcing/validating an agent's response against a Pydantic model |
| **Runner** | The ADK object that drives one agent through a turn, given a session and an input message |
| **SessionService** | Holds conversation state for a (app, user, session) triple; `InMemorySessionService` is the non-persistent local version |
| **LiteLLM** | A library/proxy that normalizes 100+ LLM providers behind one OpenAI-shaped call signature |
| **Pydantic BaseModel** | A typed, validated data class; also used to constrain LLM JSON output |
| **RAG** | Retrieval-Augmented Generation — fetch relevant facts at query time, inject into the prompt |
| **Embedding** | A fixed-length numeric vector representing a piece of text's meaning, used for similarity search |
| **pgvector** | A PostgreSQL extension adding a vector column type + similarity search operators |
| **LoRA** | Low-Rank Adaptation — fine-tune by training small adapter matrices instead of all model weights |
| **QLoRA** | LoRA + 4-bit quantization of the frozen base model, to fit fine-tuning on small GPUs |
| **SFT** | Supervised Fine-Tuning — train on (instruction, ideal_output) pairs |
| **DPO** | Direct Preference Optimization — train on (prompt, chosen, rejected) preference pairs |
| **PEFT** | HuggingFace library implementing parameter-efficient fine-tuning methods like LoRA |
| **TRL** | HuggingFace library providing the training-loop implementations for SFT/DPO/RLHF |
| **DeepEval** | pytest-native LLM eval framework; powers both the CI test suite and the fine-tune adapter safety gate |
| **RAGAS** | Eval framework specifically for RAG pipelines (faithfulness, context precision) |
| **Promptfoo** | YAML-driven adversarial/red-team eval CLI, tests against a live HTTP endpoint |
| **LLM-as-judge** | Using a (usually more capable) LLM to score another LLM's output against a rubric |
| **APScheduler** | Python cron-like job scheduler; `AsyncIOScheduler` integrates with asyncio |
| **Terraform / IaC** | Declarative infrastructure-as-code; `terraform apply` reconciles real cloud state to match `.tf` files |
| **Cloud Run** | GCP's serverless container hosting — scales to zero, pay per request |
| **Cloud Build** | GCP's CI/CD pipeline runner, driven by `cloudbuild.yaml` |
| **Secret Manager** | GCP's secrets store — keeps credentials out of env files/images |
| **A/B deploy** | Routing a small % of traffic to a new model version before fully promoting it |
| **Quantization** | Reducing numeric precision (e.g. 16-bit → 4-bit) of model weights to save memory |
| **KL penalty (DPO beta)** | A term limiting how far a fine-tuned model is allowed to drift from its reference policy |
| **GRPO** | Group Relative Policy Optimization — samples G completions per prompt, scores with a reward fn, updates toward relatively better ones; no reference model or paired data needed |
| **CrossEncoder** | A transformer that jointly encodes (query, document) pairs for highly accurate reranking — more precise than dot product but slower; used after bi-encoder shortlisting |
| **Bi-encoder** | Encodes query and document independently and compares via cosine similarity — fast but less precise than a CrossEncoder |
| **instructor** | Python library wrapping LLM clients to return validated Pydantic models; handles JSON extraction, schema coercion, and retry on validation failure |
| **A2A protocol** | Google's Agent-to-Agent specification: Agent Card for discovery + Task API (send/get/cancel) so different agent frameworks can compose without coupling |
| **HITL** | Human-in-the-Loop — a pattern where the AI pauses execution, waits for human input, and resumes; CallOS implements this with asyncio.Future + contextvars |
| **contextvars** | Python stdlib module for per-async-task ambient state; used in CallOS to propagate call_id into ADK sub-tasks without explicit argument passing |
| **Two-stage RAG** | Retrieval pipeline: fast approximate search (bi-encoder cosine) to build a candidate pool, followed by precise reranking (CrossEncoder) — standard production pattern |

---

## Part F — New Features (Tier 1, 2, 3 Post-Build Upgrades)

This section documents every change made after the initial build, why it was added, how it works, and where in the code it lives.

---

### F1. Human-in-the-Loop (HITL) Escalation

**Files:** `hitl.py` (new), `agents/agent.py`, `api/main.py`

**What:** When the live voice agent decides a caller needs a human supervisor, `transfer_to_human` now genuinely pauses execution instead of returning immediately. The call waits up to 30 seconds for a supervisor to POST a response. The agent then resumes with the human's text and continues the conversation.

**Why it matters:** Every HITL checklist item across all three portfolio projects was `🔥❌` — no project showed it working. This is the most visible gap recruiters check for agentic AI roles.

**How it works — `hitl.py`:**
```
ContextVar  _call_id_var   — per-async-task binding (copied to ADK sub-tasks automatically)
dict        _PENDING        — call_id → asyncio.Future[str]

set_call_id(call_id)       — called at the start of handle_turn
register(call_id)          — creates and parks a Future; called from transfer_to_human
resolve(call_id, response) — sets the Future; called from POST /escalation/{id}/respond
list_pending()             — returns call IDs waiting for input
```

**Key Python mechanic:** `asyncio.create_task()` copies the running context to the new task. So `ContextVar` values set in `handle_turn` are automatically visible inside ADK-spawned sub-tasks — `transfer_to_human` can read `call_id` without receiving it as an argument.

**Flow:**
```
/test-call → handle_turn → hitl.set_call_id(call_id)
           → run_agent(root_agent) → agent calls transfer_to_human(reason)
                                   → hitl.register(call_id) → creates Future
                                   → asyncio.wait_for(future, timeout=30)  ← PAUSED
                                   
GET /escalation/pending   → ["call-id-xyz"]
POST /escalation/call-id-xyz/respond  {"response": "Offer 20% discount"}
                                   → hitl.resolve(call_id, "Offer 20% discount")
                                   → future.set_result(...)   ← RESUMES
                                   → returns {"status": "resumed_after_human", ...}
```

**Endpoints added to `api/main.py`:**
- `GET  /escalation/pending` — list paused calls
- `POST /escalation/{call_id}/respond` — unblock one call

---

### F2. Two-Stage KB Retrieval with CrossEncoder Reranking

**Files:** `agents/kb_agent.py`, `mcp/kb_server.py`

**What:** Replaced the `LIKE '%query%'` keyword match with a proper two-stage semantic retrieval pipeline.

**Stage 1 — Bi-encoder cosine similarity:**
- Query is embedded with `all-MiniLM-L6-v2` (same model used by `scripts/index_kb.py`)
- Embeddings stored in `kb_chunks.embedding` (JSON floats) are read and compared
- Cosine similarity ranks all chunks; top `RERANK_POOL=10` advance

**Stage 2 — CrossEncoder reranking:**
- Model: `cross-encoder/ms-marco-MiniLM-L-6-v2` (~66 MB, CPU-friendly)
- Jointly encodes `(query, chunk)` pair — both texts attend to each other
- Much more precise than dot product; re-scores the 10 candidates
- Returns top `TOP_K=3` by cross-encoder score

**Why CrossEncoder is more accurate than cosine:**
Bi-encoder: `encode(query)` · `encode(chunk)` — vectors computed independently, interaction is lost.
CrossEncoder: `encode([query, chunk])` — full cross-attention over both at once, captures precise relevance signals that cosine misses.

**Trade-off:** CrossEncoder is ~100x slower than bi-encoder. That's why you run bi-encoder first to prune candidates, then CrossEncoder on the small pool — "recall then precision."

**No new packages needed** — `CrossEncoder` is part of `sentence-transformers` which was already installed.

---

### F3. SQL Analytics Depth (CTEs + Window Functions)

**File:** `mcp/analytics_server.py`

**What:** Three SQL upgrades showing production-grade analytical queries. SQLite supports window functions since 3.25 (2018); Python 3.8+ ships with SQLite ≥3.31.

**`get_lead_funnel`** — upgraded with CTE + percentage:
```sql
WITH totals AS (SELECT COUNT(*) AS grand_total FROM leads),
by_status AS (SELECT status, COUNT(*) AS n FROM leads GROUP BY status)
SELECT b.status, b.n, ROUND(100.0 * b.n / NULLIF(t.grand_total, 0), 1) AS pct
FROM by_status b, totals t ORDER BY b.n DESC
```
The CTE computes grand total once; avoids a correlated subquery per row.

**`get_conversion_trend`** — 7-day rolling average via window function:
```sql
WITH daily AS (...)
SELECT day, conversions,
  ROUND(AVG(conversions) OVER (
    ORDER BY day ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
  ), 2) AS rolling_7d_avg
FROM daily ORDER BY day
```
`ROWS BETWEEN 6 PRECEDING AND CURRENT ROW` = "me and the 6 days before me" — a 7-day sliding window.

**`get_quality_leaderboard`** — new tool, RANK() window function:
```sql
WITH daily_quality AS (
  SELECT DATE(created_at) AS day, COUNT(*) AS call_count,
    ROUND(AVG(quality_score), 2) AS avg_quality
  FROM calls WHERE quality_score IS NOT NULL GROUP BY DATE(created_at)
)
SELECT day, call_count, avg_quality,
  RANK() OVER (ORDER BY avg_quality DESC) AS quality_rank
FROM daily_quality ORDER BY quality_rank LIMIT 10
```
`RANK()` assigns equal rank to ties (1, 2, 2, 4...) — correct for a quality leaderboard. `ROW_NUMBER()` would give arbitrary ordering within ties; `DENSE_RANK()` would give 1,2,2,3.

---

### F4. GRPO Trainer — DeepSeek's Alignment Method

**File:** `pipeline/grpo_trainer.py` (new)

**What:** Group Relative Policy Optimization — an alternative to DPO that doesn't need paired preference data. Added as:
1. A standalone training function in `pipeline/grpo_trainer.py`
2. A fallback in `scheduler/fine_tune_job.py` when DPO pairs are scarce (`< MIN_DPO_PAIRS=20`)

**DPO vs GRPO — the core difference:**

| | DPO | GRPO |
|---|---|---|
| Data needed | (prompt, chosen, rejected) pairs | Prompts only |
| Reference model | Yes — KL penalty against SFT reference | No |
| Mechanism | Max log ratio P(chosen)/P(rejected) | Sample G completions, reward relatively better ones |
| Stability | High (bounded by KL term) | Lower — reward fn quality matters more |
| Data cost | High — need human preference labels | Low — only prompts needed |
| Best for | Rich preference data (post-deployment) | New domains, cold-start, data-scarce weeks |

**How GRPO works:**
1. For each prompt, sample `NUM_GENERATIONS=4` completions from the current policy
2. Score each with the reward function `_call_quality_reward()`
3. Compute relative advantages: `r_i - mean(r) / std(r)`  
4. Update policy to increase probability of higher-advantage completions
5. No reference model needed — the "group" provides the baseline

**`_call_quality_reward` — rule-based reward function:**
```
compliance (0.4 weight)      → no prohibited phrases (guaranteed, promise, etc.)
professionalism (0.3 weight) → contains at least one professional marker
conciseness (0.3 weight)     → ≤80 words; penalized linearly above that
```
Rule-based scoring is fast (no LLM API call during training) and transparent.

**Before/after eval (shared with DPO):**
Both trainers import `eval_quality()` from `pipeline/dpo_trainer.py`. It runs 5 held-out scenarios through the model, scores each response with `_score_response()` (keyword recall + compliance penalty), and returns a mean quality score. Output:
```
[GRPO] Quality before: 0.6200
[GRPO] Quality after:  0.7800  (Δ +0.1600)
```

**Scheduler integration (`scheduler/fine_tune_job.py`):**
```python
if len(dataset.dpo) >= MIN_DPO_PAIRS:
    aligned_path = run_dpo_alignment(sft_path, dpo_dataset_path, week_num)
else:
    aligned_path = run_grpo_alignment(sft_path, grpo_prompts_path, week_num)
```

---

### F5. Real Deepgram STT + ElevenLabs TTS

**File:** `api/main.py`

**What:** Both voice functions are now key-gated: if the API key is present in `.env`, the real service is called; without a key, the stub fires and the rest of the pipeline still runs normally.

**`transcribe_audio(audio: bytes) → str`:**
```python
if DEEPGRAM_API_KEY:
    client = DeepgramClient(api_key)
    response = await client.listen.asyncprerecorded.v("1").transcribe_file(
        {"buffer": audio, "mimetype": "audio/wav"},
        PrerecordedOptions(model="nova-3", smart_format=True, language="en"),
    )
    return response.results.channels[0].alternatives[0].transcript
```
- Model: Nova-3 (Deepgram's most accurate English model, 2024)
- `smart_format=True` — auto-capitalizes, adds punctuation
- `asyncprerecorded` — processes complete audio buffers (not streaming)

**`synthesize_speech(text: str) → bytes`:**
```python
if ELEVENLABS_API_KEY:
    client = AsyncElevenLabs(api_key=api_key)
    audio = await client.text_to_speech.convert(
        voice_id=voice_id,
        text=text,
        model_id="eleven_flash_v2_5",
    )
    return audio  # MP3 bytes streamed back to Twilio
```
- Model: Flash v2.5 (ElevenLabs' low-latency model, ~75ms TTFB)
- Voice: Rachel (ID `21m00Tcm4TlvDq8ikWAM`) by default; override via `ELEVENLABS_VOICE_ID`
- Returns raw MP3 bytes; Twilio plays these directly

**New env vars:**
```
DEEPGRAM_API_KEY=          # from console.deepgram.com
ELEVENLABS_API_KEY=        # from elevenlabs.io/app/settings/api-keys
ELEVENLABS_VOICE_ID=       # optional; default is Rachel
```

---

### F6. A2A Protocol (Agent-to-Agent Interoperability)

**File:** `api/main.py`

**What:** Implements Google's Agent-to-Agent protocol so any external orchestrator (another ADK agent, LangGraph, AutoGen, CrewAI) can compose with CallOS as a typed task endpoint.

**The A2A spec components:**

**Agent Card** (`GET /.well-known/agent.json`) — discovery document:
```json
{
  "name": "CallOS Voice Agent",
  "url": "http://localhost:8000",
  "version": "1.0.0",
  "capabilities": {"streaming": false, ...},
  "skills": [{"id": "handle-call", "tags": ["voice", "sales"], ...}]
}
```
External agents fetch this once to understand what CallOS can do.

**Task lifecycle:**
```
POST /a2a/tasks       → {"status": {"state": "submitted"}}  (returns immediately)
     ↓ background coroutine starts
GET  /a2a/tasks/{id}  → {"status": {"state": "working"}}    (poll)
     ↓ handle_turn completes
GET  /a2a/tasks/{id}  → {"status": {"state": "completed"},
                          "artifacts": [{"name": "call-result", "parts": [...]}]}
```

**Task object shape:**
```python
{
  "id": "uuid",
  "status": {
    "state": "submitted|working|completed|failed|canceled",
    "message": {"role": "agent", "parts": [{"type": "text", "text": "..."}]}
  },
  "artifacts": [
    {"name": "call-result", "parts": [{"type": "data", "data": {...}}]}
  ]
}
```

**Why this matters:** A2A is the emerging standard for multi-agent composition across different frameworks. An agent in WealthOS (LangGraph) could call CallOS's `handle-call` skill without knowing anything about ADK — it just follows the A2A spec. The Agent Card is what makes this discoverable.

**HITL integration:** `_execute_a2a_task` calls `hitl.set_call_id(task_id)` before running, so escalation works end-to-end even when a call comes in through the A2A path.

---

### F7. Instructor Structured Output (Tier 3)

**File:** `pipeline/scorer.py`

**What:** Replaced manual `json.loads() + CallScore(**raw)` with `instructor` — a library that wraps LLM clients to return validated Pydantic models directly.

**Before (manual parsing):**
```python
response = await litellm.acompletion(model=..., messages=[...],
    response_format={"type": "json_object"})
raw = json.loads(response.choices[0].message.content)
score = CallScore(**raw)  # raises KeyError/ValidationError on bad output
```

**After (instructor):**
```python
_client = instructor.from_litellm(litellm.acompletion)

score: CallScore = await _client.chat.completions.create(
    model=config.get_litellm_model_name(),
    messages=[{"role": "user", "content": prompt}],
    response_model=CallScore,
    max_retries=2,       # auto-corrects on Pydantic validation failure
)
```

**What instructor does internally:**
1. Sends your messages + a JSON schema derived from `CallScore` to the LLM
2. Parses the response into a `CallScore` instance
3. If Pydantic validation fails, sends a correction prompt automatically (up to `max_retries` times)
4. Returns a fully validated `CallScore` or raises after exhausting retries

**Why this is the production pattern:** Raw `json.loads()` fails silently on partial JSON or missing fields. `instructor` makes structured output reliable enough to use in a real pipeline gate.

---

### F8. Post-Call Pipeline Wiring

**File:** `api/main.py` — `_run_post_call_analysis()` + `handle_turn()`

**What:** Previously `score_call`, `churn_predictor_agent`, and `topic_extractor_agent` all existed but were never triggered after a real call. Now every call automatically fires all three as a background task.

**Implementation:**
```python
async def _run_post_call_analysis(call_id: str, transcript: str) -> None:
    for label, coro in [
        ("scorer",  score_call(call_id, transcript)),       # LLM-as-judge quality score
        ("churn",   run_agent(churn_predictor_agent, ...)), # churn risk classification
        ("topics",  run_agent(topic_extractor_agent, ...)), # topic cluster extraction
    ]:
        try:
            await coro
        except Exception as exc:
            print(f"[POST-CALL] {label} failed (non-fatal): {exc}")

# In handle_turn, after writing the DB:
asyncio.create_task(_run_post_call_analysis(call_id, transcript))
```

**Why background task:** `asyncio.create_task()` fires the coroutine concurrently — `/test-call` returns immediately while scoring runs in parallel. The caller never waits for analysis.

**Why each step is guarded:** If the LLM key is missing, `score_call` will fail. That must not crash the churn or topic steps. Each step is independently try/except wrapped.

**Data flow after wiring:**
```
/test-call
  → handle_turn
    → db.execute (INSERT call)
    → run_agent(root_agent) → agent_response
    → run_agent(lead_scorer_agent) → lead status + score
    → db.execute (UPDATE call)
    → asyncio.create_task(_run_post_call_analysis)  ← non-blocking
  ← returns {call_id, agent_response, lead_status, score}

Background (concurrent):
  → score_call(call_id, transcript)          → writes quality_score (0-100)
  → run_agent(churn_predictor_agent, ...)    → churn risk result
  → run_agent(topic_extractor_agent, ...)    → topic clusters
```

---

### CI/CD — cloudbuild.yaml

**File:** `cloudbuild.yaml`

Yes, CallOS has a full CI/CD pipeline running on GCP Cloud Build. It triggers on every push to `main`:

```yaml
steps:
  1. pip install -r requirements.txt
  2. pytest tests/ -v                          # deterministic unit tests
  3. deepeval test run tests/test_agent_quality.py   # LLM eval gate
  4. docker build -t gcr.io/$PROJECT_ID/callos-api .
  5. docker push gcr.io/$PROJECT_ID/callos-api
  6. gcloud run deploy callos-api --image=...   # deploy to Cloud Run
```

**Why the eval gate is in CI:** Step 3 blocks deployment if any golden scenario fails the AnswerRelevancy or Faithfulness metric thresholds. A model regression from a bad fine-tune will fail CI before it ships — this is the "safe self-improvement" claim.

**Local equivalent:**
```bash
pytest tests/ -v
deepeval test run tests/test_agent_quality.py
```
