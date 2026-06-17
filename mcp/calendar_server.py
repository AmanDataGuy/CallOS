# ============================================================
# mcp/calendar_server.py
# ------------------------------------------------------------
# Calendar MCP Server (3 tools)
#
# What it does:
#   Appointment scheduling over MCP: check_availability,
#   book_appointment, reschedule. Uses an in-module mock calendar
#   so it runs with no external service.
#
# How it fits in CallOS:
#   The Live Voice Agent books demos/callbacks mid-call. Locally
#   the slots are mocked; production swaps in Google Calendar via
#   Composio with the same tool signatures.
#
# ADK pattern used:
#   stdio MCP server with FunctionTool registry + a mock store
#   (same pattern as ADK/Module 11 MCP + Module 8 mock-data tools)
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

PORT = 8004

# Mock free slots offered to callers. # TODO: swap to Composio Google
# Calendar free/busy lookup in production.
AVAILABLE_SLOTS = ["2026-06-15 10:00", "2026-06-15 14:00", "2026-06-16 11:00"]

# In-process booking store keyed by confirmation id (mock only).
_BOOKINGS: dict[str, dict] = {}

LOG_FILE_PATH = os.path.join(os.path.dirname(__file__), "calendar_server.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
    handlers=[logging.FileHandler(LOG_FILE_PATH, mode="w")],
)


# -----------------------------
# Calendar tools (mock store)
# -----------------------------

def check_availability(date: str) -> dict:
    """List open appointment slots, optionally for a given date.

    Args:
        date (str): YYYY-MM-DD to filter by, or '' for all upcoming.

    Returns:
        dict: {"success": bool, "slots": [str, ...]}.
    """
    if date:
        slots = [s for s in AVAILABLE_SLOTS if s.startswith(date)]
    else:
        slots = AVAILABLE_SLOTS
    return {"success": True, "slots": slots}


def book_appointment(phone_number: str, slot: str) -> dict:
    """Book an appointment slot for a caller.

    Args:
        phone_number (str): who the appointment is for.
        slot (str): a slot string from check_availability.

    Returns:
        dict: {"success": bool, "confirmation_id": str} or an error.
    """
    if slot not in AVAILABLE_SLOTS:
        return {"success": False, "message": f"Slot '{slot}' is not available"}
    confirmation_id = f"APPT-{uuid.uuid4().hex[:8]}"
    _BOOKINGS[confirmation_id] = {"phone_number": phone_number, "slot": slot}
    return {"success": True, "confirmation_id": confirmation_id, "slot": slot}


def reschedule(confirmation_id: str, new_slot: str) -> dict:
    """Move an existing booking to a new slot.

    Args:
        confirmation_id (str): id returned by book_appointment.
        new_slot (str): the new slot string.

    Returns:
        dict: {"success": bool, "slot": str} or an error message.
    """
    if confirmation_id not in _BOOKINGS:
        return {"success": False, "message": f"No booking '{confirmation_id}'"}
    _BOOKINGS[confirmation_id]["slot"] = new_slot
    return {"success": True, "confirmation_id": confirmation_id, "slot": new_slot}


# -----------------------------
# Register ADK tools
# -----------------------------
ADK_TOOLS = {
    "check_availability": FunctionTool(func=check_availability),
    "book_appointment": FunctionTool(func=book_appointment),
    "reschedule": FunctionTool(func=reschedule),
}

# -----------------------------
# MCP server setup
# -----------------------------
app = Server("callos-calendar")


@app.list_tools()
async def list_mcp_tools() -> list[mcp_types.Tool]:
    """Advertise every calendar tool to connected MCP clients."""
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
    print(f"Calendar MCP server running on port {PORT}", flush=True)
    try:
        asyncio.run(run_mcp_stdio_server())
    except KeyboardInterrupt:
        logging.info("Calendar server stopped manually.")
