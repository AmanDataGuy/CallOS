# ============================================================
# api/main.py
# ------------------------------------------------------------
# FastAPI + WebSocket server for Twilio ConversationRelay
#
# What it does:
#   Hosts the telephony entry points (incoming-call webhook and the
#   ConversationRelay WebSocket) plus a /test-call endpoint that runs
#   the full agent pipeline locally with no phone or paid keys.
#
# How it fits in CallOS:
#   This is the Cloud Run service. The STT/TTS functions are stubs
#   with the same signatures as the real Deepgram/ElevenLabs calls,
#   so swapping in keys later only changes their internals.
#
# ADK pattern used:
#   google.adk.runners.Runner + InMemorySessionService to drive an
#   agent turn (same pattern as ADK/Module 7 - Session, State and Runner)
# ============================================================

import asyncio
import json
import os
import sys
import uuid

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

import db
import hitl
from agents.agent import root_agent
from agents.churn_predictor_agent import churn_predictor_agent
from agents.lead_scorer_agent import lead_scorer_agent
from agents.topic_extractor_agent import topic_extractor_agent
from pipeline.scorer import score_call

APP_NAME = "callos"

app = FastAPI(title="CallOS API")
_session_service = InMemorySessionService()


# -----------------------------
# Voice pipeline stubs (paid services — swap internals later)
# -----------------------------

async def transcribe_audio(audio: bytes) -> str:
    """Speech-to-text via Deepgram Nova-3, or stub when key is absent.

    Args:
        audio (bytes): raw WAV audio chunk from the call.

    Returns:
        str: the recognized transcript.

    Pattern:
        Key-gated: if DEEPGRAM_API_KEY is set, sends audio to
        Deepgram Nova-3 asyncprerecorded API and returns the best
        alternative.  Without a key the stub fires so the rest of
        the pipeline still runs locally.
    """
    api_key = os.environ.get("DEEPGRAM_API_KEY")
    if not api_key:
        print("[STT STUB] No DEEPGRAM_API_KEY — returning fixed transcript")
        return "Hello, I'm calling about your product"

    from deepgram import DeepgramClient, PrerecordedOptions
    client = DeepgramClient(api_key)
    response = await client.listen.asyncprerecorded.v("1").transcribe_file(
        {"buffer": audio, "mimetype": "audio/wav"},
        PrerecordedOptions(model="nova-3", smart_format=True, language="en"),
    )
    return response.results.channels[0].alternatives[0].transcript


async def synthesize_speech(text: str) -> bytes:
    """Text-to-speech via ElevenLabs Flash v2.5, or stub when key is absent.

    Args:
        text (str): the agent's reply to speak.

    Returns:
        bytes: MP3 audio bytes (empty bytes in the stub).

    Pattern:
        Key-gated: if ELEVENLABS_API_KEY is set, calls ElevenLabs
        TTS and returns audio bytes for Twilio to stream to the caller.
        ELEVENLABS_VOICE_ID defaults to the "Rachel" voice if unset.
    """
    api_key = os.environ.get("ELEVENLABS_API_KEY")
    if not api_key:
        print(f"[TTS STUB] No ELEVENLABS_API_KEY — would speak: {text}")
        return b""

    from elevenlabs.client import AsyncElevenLabs
    voice_id = os.environ.get("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")
    client = AsyncElevenLabs(api_key=api_key)
    chunks = []
    async for chunk in client.text_to_speech.convert(
        voice_id=voice_id,
        text=text,
        model_id="eleven_flash_v2_5",
    ):
        chunks.append(chunk)
    return b"".join(chunks)


# -----------------------------
# Agent runner helper
# -----------------------------

async def run_agent(agent, text: str) -> str:
    """Run one agent turn and return its final text response.

    Args:
        agent: the ADK agent to run.
        text (str): the user/caller message.

    Returns:
        str: the agent's final response text ("" if none).

    Pattern:
        Creates a fresh in-memory session and streams events, keeping
        the last final response — exactly the Runner loop from Module 7.
    """
    runner = Runner(agent=agent, app_name=APP_NAME, session_service=_session_service)
    session_id = str(uuid.uuid4())
    await _session_service.create_session(
        app_name=APP_NAME, user_id="caller", session_id=session_id
    )
    message = types.Content(role="user", parts=[types.Part(text=text)])

    final = ""
    async for event in runner.run_async(
        user_id="caller", session_id=session_id, new_message=message
    ):
        if event.is_final_response() and event.content and event.content.parts:
            final = event.content.parts[0].text or ""
    return final


async def _run_post_call_analysis(call_id: str, transcript: str) -> None:
    """Background: LLM scorer + churn + topic analysis after every call.

    Runs concurrently — never blocks the /test-call or WebSocket response.
    Each step is guarded independently so one failure doesn't abort the rest.
    """
    for label, coro in [
        ("scorer",  score_call(call_id, transcript)),
        ("churn",   run_agent(churn_predictor_agent, transcript)),
        ("topics",  run_agent(topic_extractor_agent, transcript)),
    ]:
        try:
            await coro
        except Exception as exc:
            print(f"[POST-CALL] {label} failed (non-fatal): {exc}")


async def handle_turn(transcript: str, phone_number: str) -> dict:
    """Run the full post-call pipeline for one transcript.

    Args:
        transcript (str): the caller's message / full transcript.
        phone_number (str): caller id (used to store the call).

    Returns:
        dict: {call_id, agent_response, lead_status, lead_score}.

    Pattern:
        Saves the call, binds the call_id to the async context so
        transfer_to_human can register an escalation (HITL pattern),
        gets the live agent's reply, runs the lead scorer agent, then
        writes the outcome back to the DB.
    """
    call_id = str(uuid.uuid4())
    hitl.set_call_id(call_id)  # propagates to ADK sub-tasks via contextvars

    await db.execute(
        "INSERT INTO calls (id, phone_number, direction, transcript, started_at) "
        "VALUES (?, ?, 'inbound', ?, CURRENT_TIMESTAMP)",
        (call_id, phone_number, transcript),
    )

    agent_response = await run_agent(root_agent, transcript)
    lead = _parse_lead(await run_agent(lead_scorer_agent, transcript))

    await db.execute(
        "UPDATE calls SET outcome = ?, lead_status = ?, quality_score = ? WHERE id = ?",
        (agent_response[:200], lead["status"], lead["score"], call_id),
    )

    # Fire post-call analysis in background (non-blocking).
    asyncio.create_task(_run_post_call_analysis(call_id, transcript))

    return {"call_id": call_id, "agent_response": agent_response, **lead}


def _parse_lead(raw: str) -> dict:
    """Parse the lead scorer's JSON output, with a safe fallback."""
    try:
        data = json.loads(raw)
        return {"status": data.get("status", "cold"), "score": float(data.get("score", 0))}
    except (json.JSONDecodeError, TypeError, ValueError):
        return {"status": "cold", "score": 0.0}


# -----------------------------
# HTTP + WebSocket endpoints
# -----------------------------

@app.get("/")
async def health() -> dict:
    """Health check used by Cloud Run and `curl /`."""
    return {"status": "ok", "service": "callos-api"}


@app.post("/test-call")
async def test_call(payload: dict) -> dict:
    """Simulate an inbound call with a text transcript (no telephony).

    Args:
        payload (dict): {"transcript": str, "phone_number"?: str}.

    Returns:
        dict: the handle_turn result (response + lead classification).

    Pattern:
        The local stand-in for a Twilio call — drives the same pipeline
        so you can test the agents end-to-end with just curl.
    """
    transcript = payload.get("transcript", "")
    phone_number = payload.get("phone_number", "+10000000000")
    return await handle_turn(transcript, phone_number)


@app.post("/incoming-call")
async def incoming_call() -> Response:
    """Twilio webhook — returns TwiML that opens the ConversationRelay.

    Args:
        (none — Twilio POSTs call metadata we don't need here)

    Returns:
        Response: TwiML XML pointing Twilio at the /ws WebSocket.

    Pattern:
        ConversationRelay streams audio to /ws and reads our text
        replies aloud via the configured TTS voice.
    """
    host = os.environ.get("PUBLIC_HOST", "localhost:8000")
    twiml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<Response><Connect>"
        f'<ConversationRelay url="wss://{host}/ws" transcriptionProvider="deepgram"/>'
        "</Connect></Response>"
    )
    return Response(content=twiml, media_type="application/xml")


@app.get("/escalation/pending")
async def escalation_pending() -> dict:
    """List call IDs currently paused and waiting for human input.

    Returns:
        dict: {"pending": [call_id, ...]}.

    Pattern:
        Human supervisor polls this to discover which calls need
        attention before calling /escalation/{call_id}/respond.
    """
    return {"pending": hitl.list_pending()}


@app.post("/escalation/{call_id}/respond")
async def escalation_respond(call_id: str, payload: dict) -> dict:
    """Send a human supervisor response to a paused call.

    The waiting transfer_to_human tool receives the response and the
    agent turn resumes from where it left off.

    Args:
        call_id (str): the call to unblock (from /escalation/pending).
        payload (dict): {"response": str} — the human's message.

    Returns:
        dict: {"success": bool, "call_id": str}.

    Raises:
        404 if call_id is not in the pending escalation list.
    """
    human_response = payload.get("response", "")
    if not human_response:
        raise HTTPException(status_code=400, detail="'response' field is required")
    resolved = hitl.resolve(call_id, human_response)
    if not resolved:
        raise HTTPException(
            status_code=404,
            detail=f"No pending escalation for call_id '{call_id}'. "
                   "It may have timed out or already been resolved.",
        )
    return {"success": True, "call_id": call_id}


# ============================================================
# A2A Protocol — Agent-to-Agent interoperability
# ============================================================
# Implements the Google A2A spec so external orchestrators (another
# ADK agent, LangGraph, AutoGen, etc.) can treat CallOS as a
# composable task endpoint rather than a proprietary REST service.
#
# Endpoints:
#   GET  /.well-known/agent.json     — Agent Card (discovery)
#   POST /a2a/tasks                  — submit a task and start execution
#   GET  /a2a/tasks/{task_id}        — poll for result
#   POST /a2a/tasks/{task_id}/cancel — abort a running task
#   GET  /a2a/tasks                  — list all tasks (debug)
#
# Task lifecycle: submitted → working → completed | failed | canceled
# ============================================================

_A2A_TASKS: dict[str, dict] = {}


def _a2a_task(task_id: str, state: str = "submitted") -> dict:
    return {
        "id": task_id,
        "status": {"state": state, "message": None},
        "artifacts": [],
    }


async def _execute_a2a_task(task_id: str, transcript: str, phone: str) -> None:
    """Background coroutine: run handle_turn and write result into _A2A_TASKS."""
    task = _A2A_TASKS.get(task_id)
    if task is None:
        return
    task["status"]["state"] = "working"
    hitl.set_call_id(task_id)  # bind call_id so HITL escalation works
    try:
        result = await handle_turn(transcript, phone)
        task["status"]["state"] = "completed"
        task["status"]["message"] = {
            "role": "agent",
            "parts": [{"type": "text", "text": result["agent_response"]}],
        }
        task["artifacts"] = [
            {
                "name": "call-result",
                "parts": [{"type": "data", "data": result}],
            }
        ]
    except Exception as exc:
        task["status"]["state"] = "failed"
        task["status"]["message"] = {
            "role": "agent",
            "parts": [{"type": "text", "text": str(exc)}],
        }


@app.get("/.well-known/agent.json")
async def agent_card() -> dict:
    """A2A Agent Card — advertises this agent's identity and capabilities.

    External orchestrators fetch this once to learn how to send tasks.
    The URL field must match where the agent is actually reachable.

    Returns:
        dict: the Agent Card object per the Google A2A specification.
    """
    host = os.environ.get("PUBLIC_HOST", "localhost:8000")
    scheme = "https" if "." in host else "http"
    return {
        "name": "CallOS Voice Agent",
        "description": (
            "AI voice agent for inbound support and outbound lead qualification. "
            "Processes call transcripts, scores leads, and escalates to humans when needed."
        ),
        "url": f"{scheme}://{host}",
        "version": "1.0.0",
        "capabilities": {
            "streaming": False,
            "pushNotifications": False,
            "stateTransitionHistory": False,
        },
        "skills": [
            {
                "id": "handle-call",
                "name": "Handle Call",
                "description": (
                    "Process a call transcript: respond as the voice agent, "
                    "classify the lead (hot/warm/cold), and persist the outcome."
                ),
                "tags": ["voice", "sales", "support", "lead-scoring"],
                "examples": [
                    "I'm calling about your Enterprise pricing",
                    "I need help with my account",
                    "What's included in the Starter plan?",
                ],
                "inputModes": ["text"],
                "outputModes": ["text", "data"],
            }
        ],
    }


@app.post("/a2a/tasks")
async def a2a_send_task(payload: dict) -> dict:
    """Submit a task to the CallOS agent and start execution.

    Execution runs in the background; poll GET /a2a/tasks/{task_id}
    for the result.  The task_id can be caller-supplied or is generated.

    Args:
        payload (dict): A2A Task object —
            {
              "id"?: str,
              "message": {"role": "user", "parts": [{"type": "text", "text": str}]},
              "metadata"?: {"phone_number": str}
            }

    Returns:
        dict: the Task object with status "submitted".
    """
    task_id = payload.get("id") or str(uuid.uuid4())
    parts = payload.get("message", {}).get("parts", [])
    transcript = next(
        (p.get("text", "") for p in parts if p.get("type") == "text"), ""
    )
    phone = payload.get("metadata", {}).get("phone_number", "+10000000000")

    task = _a2a_task(task_id, "submitted")
    _A2A_TASKS[task_id] = task
    asyncio.create_task(_execute_a2a_task(task_id, transcript, phone))
    return task


@app.get("/a2a/tasks")
async def a2a_list_tasks() -> dict:
    """List all tasks (submitted, working, and terminal).

    Returns:
        dict: {"tasks": [Task, ...]}.
    """
    return {"tasks": list(_A2A_TASKS.values())}


@app.get("/a2a/tasks/{task_id}")
async def a2a_get_task(task_id: str) -> dict:
    """Poll for a task's current status and result.

    Args:
        task_id (str): the task ID returned by /a2a/tasks POST.

    Returns:
        dict: the Task object (state may be working, completed, or failed).

    Raises:
        404 if task_id is unknown.
    """
    task = _A2A_TASKS.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")
    return task


@app.post("/a2a/tasks/{task_id}/cancel")
async def a2a_cancel_task(task_id: str) -> dict:
    """Cancel a task that is still in submitted or working state.

    Args:
        task_id (str): the task to cancel.

    Returns:
        dict: the updated Task object with state "canceled".

    Raises:
        404 if task_id unknown.
        400 if the task has already reached a terminal state.
    """
    task = _A2A_TASKS.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")
    if task["status"]["state"] in ("completed", "failed", "canceled"):
        raise HTTPException(
            status_code=400,
            detail=f"Task is already in terminal state '{task['status']['state']}'",
        )
    task["status"]["state"] = "canceled"
    return task


@app.websocket("/ws")
async def conversation_relay(websocket: WebSocket) -> None:
    """Twilio ConversationRelay WebSocket — STT in, agent reply out.

    Args:
        websocket (WebSocket): the live ConversationRelay connection.

    Returns:
        None

    Pattern:
        On each final transcript event, runs the live agent and sends
        the reply back as a text token Twilio speaks via TTS.
    """
    await websocket.accept()
    await websocket.send_json({"type": "config", "transcriptionProvider": "deepgram"})
    try:
        while True:
            event = await websocket.receive_json()
            if event.get("type") == "transcript" and event.get("transcriptType") == "final":
                reply = await run_agent(root_agent, event.get("transcript", ""))
                await synthesize_speech(reply)  # stub — Twilio TTS does the real audio
                await websocket.send_json({"type": "text", "token": reply, "last": True})
    except WebSocketDisconnect:
        print("[WS] ConversationRelay disconnected")
