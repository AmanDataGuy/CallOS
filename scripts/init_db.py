# ============================================================
# scripts/init_db.py
# ------------------------------------------------------------
# Database bootstrap
#
# What it does:
#   Creates the four CallOS tables (calls, leads, kb_chunks,
#   analytics) in the local SQLite database. Safe to re-run —
#   every statement is CREATE TABLE IF NOT EXISTS.
#
# How it fits in CallOS:
#   Run once before starting the MCP servers or the API. The
#   schema mirrors the PostgreSQL DDL in CallOS_README.md, with
#   SQLite-friendly types (TEXT ids, TEXT embedding) so the same
#   code runs locally with no Postgres.
#
# ADK pattern used:
#   standalone asyncio script (same run style as
#   ADK/Module 7 - Session, State and Runner "Runner Practical.py")
# ============================================================

import asyncio
import os
import sys

# Put the project root on the path so `import db` works when this
# file is run directly as `python scripts/init_db.py`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import db

# Each entry is one CREATE statement. Postgres UUID -> TEXT and
# VECTOR(1536) -> TEXT (JSON list) for SQLite; same columns otherwise.
TABLES = {
    "calls": """
        CREATE TABLE IF NOT EXISTS calls (
            id TEXT PRIMARY KEY,
            phone_number TEXT,
            direction TEXT CHECK (direction IN ('inbound', 'outbound')),
            started_at TEXT,
            ended_at TEXT,
            duration_seconds INTEGER,
            transcript TEXT,
            quality_score REAL,
            outcome TEXT,
            lead_status TEXT,
            adapter TEXT DEFAULT 'base',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            metadata TEXT
        )
    """,
    "leads": """
        CREATE TABLE IF NOT EXISTS leads (
            id TEXT PRIMARY KEY,
            phone_number TEXT UNIQUE,
            name TEXT,
            company TEXT,
            crm_id TEXT,
            score REAL,
            status TEXT,
            call_count INTEGER DEFAULT 0,
            last_called_at TEXT,
            notes TEXT
        )
    """,
    "kb_chunks": """
        CREATE TABLE IF NOT EXISTS kb_chunks (
            id TEXT PRIMARY KEY,
            content TEXT,
            source TEXT,
            embedding TEXT,
            metadata TEXT
        )
    """,
    "analytics": """
        CREATE TABLE IF NOT EXISTS analytics (
            id TEXT PRIMARY KEY,
            metric TEXT,
            value TEXT,
            period TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """,
}


async def init_db() -> None:
    """
    Create every CallOS table if it does not already exist.

    Args:
        (none)

    Returns:
        None

    Pattern:
        Loops over the TABLES dict and runs each CREATE through the
        shared db.execute helper, then prints a confirmation line the
        build checklist greps for.
    """
    for name, ddl in TABLES.items():
        await db.execute(ddl)

    print(f"Database initialized. Tables created: {', '.join(TABLES)}")


if __name__ == "__main__":
    asyncio.run(init_db())
