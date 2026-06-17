# ============================================================
# mcp/kb_server.py
# ------------------------------------------------------------
# Knowledge Base MCP Server (4 tools)
#
# What it does:
#   Exposes company-knowledge retrieval over MCP: search_kb,
#   get_faq, get_product_info, get_pricing. Reads the kb_chunks
#   table that scripts/index_kb.py populates.
#
# How it fits in CallOS:
#   The Knowledge Retrieval (RAG) Agent calls these during a live
#   call to answer product/pricing questions. search_kb uses a
#   two-stage pipeline: bi-encoder cosine similarity → cross-encoder
#   reranking. Other tools filter by source without reranking.
#
# ADK pattern used:
#   stdio MCP server with FunctionTool registry + list_tools/call_tool
#   (same pattern as ADK/Module 11 - MCP in ADK, local_mcp/server.py)
# ============================================================

import asyncio
import json
import logging
import math
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

PORT = 8002

MAX_RESULTS = 3    # final chunks returned to the caller
RERANK_POOL = 10   # cosine candidates forwarded to cross-encoder

BI_ENCODER_MODEL = "all-MiniLM-L6-v2"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

LOG_FILE_PATH = os.path.join(os.path.dirname(__file__), "kb_server.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
    handlers=[logging.FileHandler(LOG_FILE_PATH, mode="w")],
)

_bi_encoder = None
_cross_encoder = None


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    mag = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(x * x for x in b))
    return dot / (mag + 1e-9)


def _get_bi_encoder():
    global _bi_encoder
    if _bi_encoder is None:
        from sentence_transformers import SentenceTransformer
        _bi_encoder = SentenceTransformer(BI_ENCODER_MODEL)
    return _bi_encoder


def _get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder
        _cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
    return _cross_encoder


# -----------------------------
# KB tools (operate on `kb_chunks`)
# -----------------------------

async def search_kb(query: str) -> dict:
    """Two-stage semantic search over the knowledge base.

    Stage 1: bi-encoder cosine similarity ranks all stored chunks.
    Stage 2: cross-encoder reranks the top RERANK_POOL candidates
    for precision before returning MAX_RESULTS.

    Args:
        query (str): the user's question or keywords.

    Returns:
        dict: {"success": bool, "chunks": [str, ...]}.
    """
    query_vec = _get_bi_encoder().encode(query).tolist()
    rows = await db.fetch_all(
        "SELECT content, embedding FROM kb_chunks WHERE embedding IS NOT NULL"
    )
    if not rows:
        return {"success": True, "chunks": []}

    scored: list[tuple[float, str]] = []
    for row in rows:
        try:
            chunk_vec = json.loads(row["embedding"])
        except (json.JSONDecodeError, TypeError):
            continue
        scored.append((_cosine(query_vec, chunk_vec), row["content"]))

    scored.sort(reverse=True)
    candidates = [c for _, c in scored[:RERANK_POOL]]

    if len(candidates) > 1:
        ce_scores = _get_cross_encoder().predict(
            [(query, c) for c in candidates]
        ).tolist()
        reranked = [c for _, c in sorted(zip(ce_scores, candidates), reverse=True)]
    else:
        reranked = candidates

    return {"success": True, "chunks": reranked[:MAX_RESULTS]}


async def get_faq(topic: str) -> dict:
    """Return FAQ chunks, optionally filtered by a topic keyword.

    Args:
        topic (str): keyword to filter FAQ content (use '' for all).

    Returns:
        dict: {"success": bool, "faq": [str, ...]}.
    """
    rows = await db.fetch_all(
        "SELECT content FROM kb_chunks WHERE source = 'faq' AND content LIKE ? LIMIT ?",
        (f"%{topic}%", MAX_RESULTS),
    )
    return {"success": True, "faq": [r["content"] for r in rows]}


async def get_product_info() -> dict:
    """Return product overview chunks from the knowledge base.

    Args:
        (none)

    Returns:
        dict: {"success": bool, "product": [str, ...]}.
    """
    rows = await db.fetch_all(
        "SELECT content FROM kb_chunks WHERE source = 'product' LIMIT ?",
        (MAX_RESULTS,),
    )
    return {"success": True, "product": [r["content"] for r in rows]}


async def get_pricing() -> dict:
    """Return pricing chunks from the knowledge base.

    Args:
        (none)

    Returns:
        dict: {"success": bool, "pricing": [str, ...]}.
    """
    rows = await db.fetch_all(
        "SELECT content FROM kb_chunks WHERE source = 'pricing' LIMIT ?",
        (MAX_RESULTS,),
    )
    return {"success": True, "pricing": [r["content"] for r in rows]}


# -----------------------------
# Register ADK tools
# -----------------------------
ADK_TOOLS = {
    "search_kb": FunctionTool(func=search_kb),
    "get_faq": FunctionTool(func=get_faq),
    "get_product_info": FunctionTool(func=get_product_info),
    "get_pricing": FunctionTool(func=get_pricing),
}

# -----------------------------
# MCP server setup
# -----------------------------
app = Server("callos-kb")


@app.list_tools()
async def list_mcp_tools() -> list[mcp_types.Tool]:
    """Advertise every KB tool to connected MCP clients."""
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
    print(f"KB MCP server running on port {PORT}", flush=True)
    try:
        asyncio.run(run_mcp_stdio_server())
    except KeyboardInterrupt:
        logging.info("KB server stopped manually.")
