# ============================================================
# mcp/call_server.py
# ------------------------------------------------------------
# Call Records MCP Server (5 tools)
#
# What it does:
#   Stores and queries call records over MCP: save_transcript,
#   get_call_history, log_outcome, get_past_interactions,
#   get_call_metrics. Backed by the `calls` table.
#
# How it fits in CallOS:
#   The API saves each finished call here; the Live Voice Agent
#   pulls a caller's history for context; the BI layer reads
#   aggregate metrics from get_call_metrics.
#
# ADK pattern used:
#   stdio MCP server with FunctionTool registry + list_tools/call_tool
#   (same pattern as ADK/Module 11 - MCP in ADK, local_mcp/server.py)
# ============================================================

import asyncio
import json
import logging
import os
import sys
import uuid

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from google.adk.tools.function_tool import FunctionTool
from google.adk.tools.mcp_tool.conversion_utils import adk_to_mcp_tool_type

from mcp import types as mcp_types
from mcp.server.lowlevel import NotificationOptions, Server
from mcp.server.models import InitializationOptions
import mcp.server.stdio

import db

PORT = 8003

# How many past rows context/history tools return — keeps prompts small.
HISTORY_LIMIT = 5

LOG_FILE_PATH = os.path.join(os.path.dirname(__file__), "call_server.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
    handlers=[logging.FileHandler(LOG_FILE_PATH, mode="w")],
)


# -----------------------------
# Call tools (operate on `calls`)
# -----------------------------

async def save_transcript(phone_number: str, transcript: str, direction: str) -> dict:
    """Persist a new call with its transcript.

    Args:
        phone_number (str): the caller / callee number.
        transcript (str): full conversation text.
        direction (str): 'inbound' or 'outbound'.

    Returns:
        dict: {"success": bool, "call_id": str}.
    """
    call_id = str(uuid.uuid4())
    await db.execute(
        "INSERT INTO calls (id, phone_number, direction, transcript, started_at) "
        "VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)",
        (call_id, phone_number, direction, transcript),
    )
    return {"success": True, "call_id": call_id}


async def get_call_history(phone_number: str) -> dict:
    """Return recent calls for a phone number.

    Args:
        phone_number (str): number to look up.

    Returns:
        dict: {"success": bool, "calls": [{...}, ...]}.
    """
    rows = await db.fetch_all(
        "SELECT id, direction, outcome, quality_score, created_at FROM calls "
        "WHERE phone_number = ? ORDER BY created_at DESC LIMIT ?",
        (phone_number, HISTORY_LIMIT),
    )
    return {"success": True, "calls": rows}


async def log_outcome(call_id: str, outcome: str) -> dict:
    """Record the outcome of a finished call.

    Args:
        call_id (str): which call to update.
        outcome (str): short outcome label.

    Returns:
        dict: {"success": bool}.
    """
    await db.execute(
        "UPDATE calls SET outcome = ?, ended_at = CURRENT_TIMESTAMP WHERE id = ?",
        (outcome, call_id),
    )
    return {"success": True, "call_id": call_id, "outcome": outcome}


async def get_past_interactions(phone_number: str) -> dict:
    """Return prior transcripts for a caller so the agent has context.

    Args:
        phone_number (str): number to look up.

    Returns:
        dict: {"success": bool, "transcripts": [str, ...]}.
    """
    rows = await db.fetch_all(
        "SELECT transcript FROM calls WHERE phone_number = ? "
        "ORDER BY created_at DESC LIMIT ?",
        (phone_number, HISTORY_LIMIT),
    )
    return {"success": True, "transcripts": [r["transcript"] for r in rows]}


async def get_call_metrics() -> dict:
    """Return aggregate call metrics for the BI layer.

    Args:
        (none)

    Returns:
        dict: {"success": bool, "total_calls": int, "avg_score": float}.
    """
    row = await db.fetch_one(
        "SELECT COUNT(*) AS total, AVG(quality_score) AS avg_score FROM calls"
    )
    return {
        "success": True,
        "total_calls": row["total"] or 0,
        "avg_score": round(row["avg_score"] or 0.0, 2),
    }


# -----------------------------
# Register ADK tools
# -----------------------------
ADK_TOOLS = {
    "save_transcript": FunctionTool(func=save_transcript),
    "get_call_history": FunctionTool(func=get_call_history),
    "log_outcome": FunctionTool(func=log_outcome),
    "get_past_interactions": FunctionTool(func=get_past_interactions),
    "get_call_metrics": FunctionTool(func=get_call_metrics),
}

# -----------------------------
# MCP server setup
# -----------------------------
app = Server("callos-calls")


@app.list_tools()
async def list_mcp_tools() -> list[mcp_types.Tool]:
    """Advertise every call tool to connected MCP clients."""
    tools = []
    for name, adk_tool in ADK_TOOLS.items():
        if not adk_tool.name:
            adk_tool.name = name
        tools.append(adk_to_mcp_tool_type(adk_tool))
    return tools


@app.call_tool()
async def call_mcp_tool(name: str, arguments: dict) -> list[mcp_types.TextContent]:
    """Dispatch an MCP tool call to the matching ADK FunctionTool."""
    logging.info(f"call_tool: {name} args={arguments}")
    if name not in ADK_TOOLS:
        payload = {"success": False, "message": f"Tool '{name}' not found"}
        return [mcp_types.TextContent(type="text", text=json.dumps(payload))]
    try:
        result = await ADK_TOOLS[name].run_async(args=arguments, tool_context=None)  # type: ignore
        return [mcp_types.TextContent(type="text", text=json.dumps(result))]
    except Exception as e:
        logging.error(f"Error running {name}: {e}", exc_info=True)
        return [mcp_types.TextContent(type="text", text=json.dumps({"success": False, "message": str(e)}))]


async def run_mcp_stdio_server() -> None:
    """Run the MCP server over stdio until the client disconnects."""
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name=app.name,
                server_version="1.0.0",
                capabilities=app.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )


if __name__ == "__main__":
    print(f"Call MCP server running on port {PORT}", flush=True)
    try:
        asyncio.run(run_mcp_stdio_server())
    except KeyboardInterrupt:
        logging.info("Call server stopped manually.")
