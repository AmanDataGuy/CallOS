# ============================================================
# db.py
# ------------------------------------------------------------
# Async database access layer
#
# What it does:
#   Thin async wrapper around the local SQLite database. Exposes
#   execute / fetch_one / fetch_all plus a couple of CallOS-specific
#   helpers (update_call_score) used by the pipeline and MCP servers.
#
# How it fits in CallOS:
#   MCP servers, the API, and the pipeline all read/write call,
#   lead, kb_chunk, and analytics rows through this one module.
#   In production the same functions point at PostgreSQL — only the
#   driver and connect() change.
#
# ADK pattern used:
#   plain async helpers called from FunctionTool funcs
#   (same pattern as ADK/Module 6 - Tools in ADK, tools stay simple)
# ============================================================

import json
import os

import aiosqlite

import config

# Resolve the DB file next to this module so every entry point
# (python mcp/..., uvicorn api.main, python scripts/...) hits the
# same file regardless of the current working directory.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DB_PATH = config.SQLITE_PATH
if not os.path.isabs(DB_PATH):
    DB_PATH = os.path.join(PROJECT_ROOT, DB_PATH)


async def execute(query: str, params: tuple = ()) -> None:
    """
    Run a write query (INSERT / UPDATE / DELETE) and commit.

    Args:
        query: SQL with `?` placeholders (SQLite style)
        params: values to bind to the placeholders

    Returns:
        None

    Pattern:
        Opens a short-lived aiosqlite connection, runs the statement,
        commits, and closes. Fine for local volumes; production swaps
        in a pooled PostgreSQL connection. # TODO: swap to asyncpg pool
    """
    async with aiosqlite.connect(DB_PATH) as conn:
        await conn.execute(query, params)
        await conn.commit()


async def fetch_one(query: str, params: tuple = ()) -> dict | None:
    """
    Fetch a single row as a dict, or None if there is no match.

    Args:
        query: SQL SELECT with `?` placeholders
        params: values to bind

    Returns:
        dict | None — column name -> value for the first row

    Pattern:
        Uses aiosqlite.Row so rows convert cleanly to dicts, matching
        the dict-return style the ADK course tools use everywhere.
    """
    async with aiosqlite.connect(DB_PATH) as conn:
        conn.row_factory = aiosqlite.Row
        async with conn.execute(query, params) as cursor:
            row = await cursor.fetchone()
            return dict(row) if row else None


async def fetch_all(query: str, params: tuple = ()) -> list[dict]:
    """
    Fetch every matching row as a list of dicts.

    Args:
        query: SQL SELECT with `?` placeholders
        params: values to bind

    Returns:
        list[dict] — one dict per row (empty list if no matches)

    Pattern:
        Same Row factory as fetch_one, returned as a list so callers
        can iterate without touching the driver.
    """
    async with aiosqlite.connect(DB_PATH) as conn:
        conn.row_factory = aiosqlite.Row
        async with conn.execute(query, params) as cursor:
            rows = await cursor.fetchall()
            return [dict(r) for r in rows]


async def update_call_score(call_id: str, score) -> None:
    """
    Persist a post-call score onto its call row.

    Args:
        call_id: UUID of the call record
        score: CallScore Pydantic model (score, breakdown, outcome, lead_status)

    Returns:
        None

    Pattern:
        Called by pipeline/scorer.py after the LLM-as-judge returns.
        The breakdown dict is stored as JSON text so SQLite can hold it.
    """
    await execute(
        "UPDATE calls SET quality_score = ?, outcome = ?, lead_status = ?, "
        "metadata = ? WHERE id = ?",
        (
            score.score,
            score.outcome,
            score.lead_status,
            json.dumps(score.breakdown),
            call_id,
        ),
    )
