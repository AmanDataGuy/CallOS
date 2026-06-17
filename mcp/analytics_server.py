# ============================================================
# mcp/analytics_server.py
# ------------------------------------------------------------
# Business Intelligence MCP Server (7 tools)
#
# What it does:
#   Aggregate insight queries over MCP: get_product_signals,
#   get_churn_risks, get_lead_funnel, get_compliance_rate,
#   get_conversion_trend, get_topic_clusters, get_quality_leaderboard.
#   Reads calls, leads, and the analytics table.
#
# How it fits in CallOS:
#   Powers the BI dashboard described in the README. LLM-derived
#   metrics (product signals, topic clusters) are precomputed by the
#   nightly topic extractor and stored in `analytics`; the counting
#   metrics (funnel, churn, conversion, leaderboard) are computed live
#   here using SQLite CTEs and window functions (requires SQLite ≥3.25,
#   which ships with Python 3.8+).
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

PORT = 8006

# A lead is "at risk" below this score — matches the README churn
# early-warning idea (low engagement / unresolved issues).
CHURN_SCORE_THRESHOLD = 40.0

LOG_FILE_PATH = os.path.join(os.path.dirname(__file__), "analytics_server.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
    handlers=[logging.FileHandler(LOG_FILE_PATH, mode="w")],
)


async def _read_metric(metric: str) -> list[dict]:
    """Helper: read precomputed analytics rows for one metric name."""
    return await db.fetch_all(
        "SELECT value, period FROM analytics WHERE metric = ? ORDER BY created_at DESC",
        (metric,),
    )


# -----------------------------
# Analytics tools
# -----------------------------

async def get_product_signals() -> dict:
    """Return precomputed product complaint/feature signals.

    Args:
        (none)

    Returns:
        dict: {"success": bool, "signals": [{...}, ...]}.
    """
    return {"success": True, "signals": await _read_metric("product_signal")}


async def get_churn_risks() -> dict:
    """Return leads flagged as at-risk of churn.

    Args:
        (none)

    Returns:
        dict: {"success": bool, "at_risk": [{...}, ...]}.
    """
    rows = await db.fetch_all(
        "SELECT phone_number, name, score, status FROM leads "
        "WHERE status = 'at_risk' OR (score IS NOT NULL AND score < ?)",
        (CHURN_SCORE_THRESHOLD,),
    )
    return {"success": True, "at_risk": rows}


async def get_lead_funnel() -> dict:
    """Return lead counts and percentage share grouped by status.

    Uses a CTE to compute the grand total once so the per-status
    percentage can be derived in a single scan without a subquery.

    Args:
        (none)

    Returns:
        dict: {"success": bool, "funnel": [{status, count, pct}, ...]}.
    """
    rows = await db.fetch_all(
        """
        WITH totals AS (
            SELECT COUNT(*) AS grand_total FROM leads
        ),
        by_status AS (
            SELECT status, COUNT(*) AS n FROM leads GROUP BY status
        )
        SELECT
            b.status,
            b.n AS count,
            ROUND(100.0 * b.n / NULLIF(t.grand_total, 0), 1) AS pct
        FROM by_status b, totals t
        ORDER BY b.n DESC
        """
    )
    return {"success": True, "funnel": rows}


async def get_compliance_rate() -> dict:
    """Return the share of calls free of compliance violations.

    Args:
        (none)

    Returns:
        dict: {"success": bool, "compliance_rate": float, "total_calls": int}.

    Note:
        A call counts as a violation if its outcome mentions 'violation'.
        Compliance details are written by the Compliance Guard agent.
    """
    total = await db.fetch_one("SELECT COUNT(*) AS n FROM calls")
    bad = await db.fetch_one("SELECT COUNT(*) AS n FROM calls WHERE outcome LIKE '%violation%'")
    n = total["n"] or 0
    rate = round(1 - (bad["n"] / n), 3) if n else 1.0
    return {"success": True, "compliance_rate": rate, "total_calls": n}


async def get_conversion_trend() -> dict:
    """Return daily conversions with a 7-day rolling average.

    Uses a CTE to aggregate daily counts, then applies an AVG window
    function over a sliding 7-row frame to smooth spikes.  The rolling
    average helps distinguish genuine trends from one-off call bursts.

    Args:
        (none)

    Returns:
        dict: {"success": bool, "trend": [{day, conversions, rolling_7d_avg}, ...]}.
    """
    rows = await db.fetch_all(
        """
        WITH daily AS (
            SELECT
                DATE(created_at) AS day,
                COUNT(*) AS conversions
            FROM calls
            WHERE outcome LIKE '%convert%' OR lead_status = 'hot'
            GROUP BY DATE(created_at)
        )
        SELECT
            day,
            conversions,
            ROUND(AVG(conversions) OVER (
                ORDER BY day
                ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
            ), 2) AS rolling_7d_avg
        FROM daily
        ORDER BY day
        """
    )
    return {"success": True, "trend": rows}


async def get_topic_clusters() -> dict:
    """Return precomputed topic clusters across recent calls.

    Args:
        (none)

    Returns:
        dict: {"success": bool, "clusters": [{...}, ...]}.
    """
    return {"success": True, "clusters": await _read_metric("topic_cluster")}


async def get_quality_leaderboard() -> dict:
    """Return top-10 days ranked by average call quality score.

    Uses a CTE to aggregate daily stats, then RANK() OVER to assign
    each day a quality rank without a self-join.  Days with no quality
    scores (stubs, missing judge key) are excluded automatically via
    the WHERE quality_score IS NOT NULL filter.

    Args:
        (none)

    Returns:
        dict: {"success": bool, "leaderboard": [{day, call_count, avg_quality, rank}, ...]}.
    """
    rows = await db.fetch_all(
        """
        WITH daily_quality AS (
            SELECT
                DATE(created_at) AS day,
                COUNT(*) AS call_count,
                ROUND(AVG(quality_score), 2) AS avg_quality
            FROM calls
            WHERE quality_score IS NOT NULL
            GROUP BY DATE(created_at)
        )
        SELECT
            day,
            call_count,
            avg_quality,
            RANK() OVER (ORDER BY avg_quality DESC) AS quality_rank
        FROM daily_quality
        ORDER BY quality_rank
        LIMIT 10
        """
    )
    return {"success": True, "leaderboard": rows}


# -----------------------------
# Register ADK tools
# -----------------------------
ADK_TOOLS = {
    "get_product_signals": FunctionTool(func=get_product_signals),
    "get_churn_risks": FunctionTool(func=get_churn_risks),
    "get_lead_funnel": FunctionTool(func=get_lead_funnel),
    "get_compliance_rate": FunctionTool(func=get_compliance_rate),
    "get_conversion_trend": FunctionTool(func=get_conversion_trend),
    "get_topic_clusters": FunctionTool(func=get_topic_clusters),
    "get_quality_leaderboard": FunctionTool(func=get_quality_leaderboard),
}

# -----------------------------
# MCP server setup
# -----------------------------
app = Server("callos-analytics")


@app.list_tools()
async def list_mcp_tools() -> list[mcp_types.Tool]:
    """Advertise every analytics tool to connected MCP clients."""
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
    print(f"Analytics MCP server running on port {PORT}", flush=True)
    try:
        asyncio.run(run_mcp_stdio_server())
    except KeyboardInterrupt:
        logging.info("Analytics server stopped manually.")
