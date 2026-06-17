# ============================================================
# mcp/scorer_server.py
# ------------------------------------------------------------
# Call Scorer MCP Server (4 tools)
#
# What it does:
#   Call-quality scoring + leaderboards over MCP: score_call,
#   get_score_breakdown, get_top_calls, get_bottom_calls.
#   Reads/writes the `calls` table.
#
# How it fits in CallOS:
#   score_call runs the LLM-as-judge from pipeline/scorer.py and
#   stores the result; get_top_calls / get_bottom_calls feed the
#   fine-tune dataset builder (high scores = SFT, low = DPO rejected).
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

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from google.adk.tools.function_tool import FunctionTool
from google.adk.tools.mcp_tool.conversion_utils import adk_to_mcp_tool_type

from mcp import types as mcp_types
from mcp.server.lowlevel import NotificationOptions, Server
from mcp.server.models import InitializationOptions
import mcp.server.stdio

import db

PORT = 8005

# How many calls the leaderboard tools return.
LEADERBOARD_SIZE = 10

LOG_FILE_PATH = os.path.join(os.path.dirname(__file__), "scorer_server.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
    handlers=[logging.FileHandler(LOG_FILE_PATH, mode="w")],
)


# -----------------------------
# Scorer tools (operate on `calls`)
# -----------------------------

async def score_call(call_id: str) -> dict:
    """Run the LLM-as-judge scorer on a stored call.

    Args:
        call_id (str): which call to score.

    Returns:
        dict: {"success": bool, "score": int, "outcome": str} or an error.

    Note:
        Delegates to pipeline/scorer.py. Imported lazily so the MCP
        server starts without loading litellm until a score is needed.
    """
    call = await db.fetch_one("SELECT transcript FROM calls WHERE id = ?", (call_id,))
    if not call:
        return {"success": False, "message": f"No call '{call_id}'"}

    from pipeline.scorer import score_call as run_scorer  # lazy import

    result = await run_scorer(call_id, call["transcript"])
    return {"success": True, "score": result.score, "outcome": result.outcome}


async def get_score_breakdown(call_id: str) -> dict:
    """Return the per-category score breakdown for a call.

    Args:
        call_id (str): which call to read.

    Returns:
        dict: {"success": bool, "score": float, "breakdown": {...}}.
    """
    call = await db.fetch_one(
        "SELECT quality_score, metadata FROM calls WHERE id = ?", (call_id,)
    )
    if not call:
        return {"success": False, "message": f"No call '{call_id}'"}
    breakdown = json.loads(call["metadata"]) if call["metadata"] else {}
    return {"success": True, "score": call["quality_score"], "breakdown": breakdown}


async def get_top_calls() -> dict:
    """Return the highest-scoring calls (SFT training candidates).

    Args:
        (none)

    Returns:
        dict: {"success": bool, "calls": [{...}, ...]}.
    """
    rows = await db.fetch_all(
        "SELECT id, quality_score, outcome FROM calls "
        "WHERE quality_score IS NOT NULL ORDER BY quality_score DESC LIMIT ?",
        (LEADERBOARD_SIZE,),
    )
    return {"success": True, "calls": rows}


async def get_bottom_calls() -> dict:
    """Return the lowest-scoring calls (DPO rejected candidates).

    Args:
        (none)

    Returns:
        dict: {"success": bool, "calls": [{...}, ...]}.
    """
    rows = await db.fetch_all(
        "SELECT id, quality_score, outcome FROM calls "
        "WHERE quality_score IS NOT NULL ORDER BY quality_score ASC LIMIT ?",
        (LEADERBOARD_SIZE,),
    )
    return {"success": True, "calls": rows}


# -----------------------------
# Register ADK tools
# -----------------------------
ADK_TOOLS = {
    "score_call": FunctionTool(func=score_call),
    "get_score_breakdown": FunctionTool(func=get_score_breakdown),
    "get_top_calls": FunctionTool(func=get_top_calls),
    "get_bottom_calls": FunctionTool(func=get_bottom_calls),
}

# -----------------------------
# MCP server setup
# -----------------------------
app = Server("callos-scorer")


@app.list_tools()
async def list_mcp_tools() -> list[mcp_types.Tool]:
    """Advertise every scorer tool to connected MCP clients."""
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
    print(f"Scorer MCP server running on port {PORT}", flush=True)
    try:
        asyncio.run(run_mcp_stdio_server())
    except KeyboardInterrupt:
        logging.info("Scorer server stopped manually.")
