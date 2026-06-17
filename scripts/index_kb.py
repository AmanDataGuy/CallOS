# ============================================================
# scripts/index_kb.py
# ------------------------------------------------------------
# Knowledge base indexer
#
# What it does:
#   Chunks company documents, embeds each chunk, and stores the
#   chunk + embedding in the kb_chunks table so the Knowledge
#   Retrieval Agent can do semantic search during live calls.
#
# How it fits in CallOS:
#   Run once per company KB update. Locally it uses a small
#   sentence-transformers model so there is no embedding API cost;
#   production swaps the table for pgvector with the same rows.
#
# ADK pattern used:
#   standalone asyncio script (same run style as
#   ADK/Module 7 - Session, State and Runner "Runner Practical.py")
# ============================================================

import asyncio
import json
import os
import sys
import uuid

# Project root on path so `import db` resolves when run directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import db

# Chunking sizes from the README RAG step (size=512, overlap=64).
# Smaller chunks keep retrieval precise; overlap avoids cutting
# sentences across chunk boundaries.
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64

# Small, CPU-friendly local embedder — no API cost. Same model the
# README names as the local default.
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

# A couple of starter docs so the KB is non-empty on first run.
SEED_DOCS = [
    ("pricing", "CallOS Enterprise plan is $2,000/month and includes unlimited "
                "call minutes, priority support, and a dedicated fine-tune job. "
                "The Starter plan is $499/month for up to 5,000 minutes."),
    ("product", "CallOS is an AI voice agent platform. It handles inbound support "
                "and outbound lead qualification, then fine-tunes itself weekly on "
                "its best-scoring calls using a QLoRA + DPO pipeline."),
    ("faq", "Refunds are available within 14 days. Data is stored encrypted at "
            "rest. CallOS supports Salesforce and HubSpot CRM sync via Composio."),
]


def chunk_text(text: str) -> list[str]:
    """
    Split text into overlapping fixed-size character windows.

    Args:
        text: the full document string

    Returns:
        list[str] — chunks of up to CHUNK_SIZE chars with CHUNK_OVERLAP overlap

    Pattern:
        Character windowing keeps this dependency-free and good enough
        for short KB docs. Step is size - overlap so windows overlap.
    """
    step = CHUNK_SIZE - CHUNK_OVERLAP
    return [text[i:i + CHUNK_SIZE] for i in range(0, len(text), step)]


def embed(text: str) -> list[float]:
    """
    Embed a chunk into a vector with the local sentence-transformer.

    Args:
        text: a single chunk of text

    Returns:
        list[float] — the embedding vector

    Pattern:
        Loads the model lazily on first call so importing this module
        is cheap. Production would call pgvector's embedding pipeline.
    """
    from sentence_transformers import SentenceTransformer

    global _model
    try:
        _model
    except NameError:
        _model = SentenceTransformer(EMBED_MODEL_NAME)
    return _model.encode(text).tolist()


async def index_documents(docs: list[tuple[str, str]]) -> int:
    """
    Chunk, embed, and store a list of (source, text) documents.

    Args:
        docs: list of (source_label, document_text) tuples

    Returns:
        int — number of chunks written to kb_chunks

    Pattern:
        Mirrors the README index_documents loop. The embedding is
        stored as a JSON list of floats (TEXT) for SQLite; pgvector
        would store it natively.
    """
    written = 0
    for source, text in docs:
        for chunk in chunk_text(text):
            await db.execute(
                "INSERT INTO kb_chunks (id, content, source, embedding, metadata) "
                "VALUES (?, ?, ?, ?, ?)",
                (str(uuid.uuid4()), chunk, source, json.dumps(embed(chunk)), "{}"),
            )
            written += 1
    return written


if __name__ == "__main__":
    count = asyncio.run(index_documents(SEED_DOCS))
    print(f"KB indexed. Chunks written: {count}")
