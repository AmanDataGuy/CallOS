# ============================================================
# mcp/crm_server.py
# ------------------------------------------------------------
# CRM MCP Server (6 tools)
#
# What it does:
#   Exposes lead read/write tools over the Model Context Protocol:
#   get_lead, update_lead, create_lead, log_call_outcome,
#   get_deal_stage, push_to_crm. All data lives in the `leads` table.
#
# How it fits in CallOS:
#   The Live Voice Agent (and the post-call Lead Scorer) reach lead
#   records through these tools. Locally the server talks to SQLite;
#   in production push_to_crm hands off to Composio -> Salesforce.
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

# Append (not insert) the project root so `import db` resolves while
# the installed `mcp` package still wins over this local mcp/ folder.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from google.adk.tools.function_tool import FunctionTool
from google.adk.tools.mcp_tool.conversion_utils import adk_to_mcp_tool_type

from mcp import types as mcp_types
from mcp.server.lowlevel import NotificationOptions, Server
from mcp.server.models import InitializationOptions
import mcp.server.stdio

import db

# Logical service id — used by docker-compose and adk_config.yaml.
# Transport here is stdio (ADK connects via MCPToolset/Stdio params);
# the port is how the server is addressed when run over HTTP in prod.
PORT = 8001

LOG_FILE_PATH = os.path.join(os.path.dirname(__file__), "crm_server.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
    handlers=[logging.FileHandler(LOG_FILE_PATH, mode="w")],
)


# -----------------------------
# CRM tools (operate on `leads`)
# -----------------------------

async def get_lead(phone_number: str) -> dict:
    """Fetch a lead record by phone number.

    Args:
        phone_number (str): the lead's phone number (unique key).

    Returns:
        dict: {"success": bool, "lead": {...}} or an error message.
    """
    lead = await db.fetch_one("SELECT * FROM leads WHERE phone_number = ?", (phone_number,))
    if lead:
        return {"success": True, "lead": lead}
    return {"success": False, "message": f"No lead found for {phone_number}"}


async def create_lead(phone_number: str, name: str, company: str) -> dict:
    """Create a new lead record.

    Args:
        phone_number (str): contact phone number.
        name (str): contact name.
        company (str): contact's company.

    Returns:
        dict: {"success": bool, "lead_id": str} or an error message.
    """
    lead_id = str(uuid.uuid4())
    try:
        await db.execute(
            "INSERT INTO leads (id, phone_number, name, company, status, call_count) "
            "VALUES (?, ?, ?, ?, 'new', 0)",
            (lead_id, phone_number, name, company),
        )
        return {"success": True, "lead_id": lead_id}
    except Exception as e:  # most likely UNIQUE conflict on phone_number
        logging.error(f"create_lead failed: {e}")
        return {"success": False, "message": str(e)}


async def update_lead(phone_number: str, status: str, score: float) -> dict:
    """Update a lead's qualification status and score.

    Args:
        phone_number (str): which lead to update.
        status (str): new status (e.g. 'hot', 'warm', 'cold').
        score (float): qualification score 0-100.

    Returns:
        dict: {"success": bool} or an error message.
    """
    await db.execute(
        "UPDATE leads SET status = ?, score = ? WHERE phone_number = ?",
        (status, score, phone_number),
    )
    return {"success": True, "phone_number": phone_number, "status": status}


async def log_call_outcome(phone_number: str, outcome: str) -> dict:
    """Record a call outcome and bump the lead's call counter.

    Args:
        phone_number (str): which lead the call was with.
        outcome (str): short outcome note (e.g. 'booked demo').

    Returns:
        dict: {"success": bool} or an error message.
    """
    await db.execute(
        "UPDATE leads SET notes = ?, call_count = call_count + 1, "
        "last_called_at = CURRENT_TIMESTAMP WHERE phone_number = ?",
        (outcome, phone_number),
    )
    return {"success": True, "outcome": outcome}


async def get_deal_stage(phone_number: str) -> dict:
    """Return the current deal stage (lead status) for a contact.

    Args:
        phone_number (str): which lead to look up.

    Returns:
        dict: {"success": bool, "stage": str} or an error message.
    """
    lead = await db.fetch_one("SELECT status FROM leads WHERE phone_number = ?", (phone_number,))
    if lead:
        return {"success": True, "stage": lead["status"]}
    return {"success": False, "message": f"No lead found for {phone_number}"}


async def push_to_crm(phone_number: str) -> dict:
    """Push a lead to the external CRM and store the returned CRM id.

    Args:
        phone_number (str): which lead to sync.

    Returns:
        dict: {"success": bool, "crm_id": str} or an error message.

    Note:
        Locally this fabricates a CRM id. # TODO: swap to Composio
        Salesforce/HubSpot action in production.
    """
    crm_id = f"CRM-{uuid.uuid4().hex[:8]}"
    await db.execute("UPDATE leads SET crm_id = ? WHERE phone_number = ?", (crm_id, phone_number))
    return {"success": True, "crm_id": crm_id}


# -----------------------------
# Register ADK tools
# -----------------------------
ADK_TOOLS = {
    "get_lead": FunctionTool(func=get_lead),
    "create_lead": FunctionTool(func=create_lead),
    "update_lead": FunctionTool(func=update_lead),
    "log_call_outcome": FunctionTool(func=log_call_outcome),
    "get_deal_stage": FunctionTool(func=get_deal_stage),
    "push_to_crm": FunctionTool(func=push_to_crm),
}

# -----------------------------
# MCP server setup
# -----------------------------
app = Server("callos-crm")


@app.list_tools()
async def list_mcp_tools() -> list[mcp_types.Tool]:
    """Advertise every CRM tool to connected MCP clients."""
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
    print(f"CRM MCP server running on port {PORT}", flush=True)
    try:
        asyncio.run(run_mcp_stdio_server())
    except KeyboardInterrupt:
        logging.info("CRM server stopped manually.")
