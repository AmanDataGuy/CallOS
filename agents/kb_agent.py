# ============================================================
# agents/kb_agent.py
# ------------------------------------------------------------
# Knowledge Retrieval (RAG) Agent — two-stage retrieval
#
# What it does:
#   Answers product, pricing, and FAQ questions by retrieving
#   matching chunks from the company knowledge base using a
#   bi-encoder + cross-encoder reranking pipeline.
#
# How it fits in CallOS:
#   A sub-agent of the Live Voice Agent. When the caller asks a
#   product question, the live agent delegates here so answers are
#   grounded in the KB rather than the model's memory.
#
# ADK pattern used:
#   google.adk.agents.Agent with a custom retrieval FunctionTool
#   (same pattern as ADK/Module 6 - Tools in ADK)
#
# Retrieval pipeline:
#   Stage 1 — bi-encoder cosine similarity (all-MiniLM-L6-v2)
#             scores every stored chunk; top RERANK_POOL candidates
#             are forwarded to stage 2.
#   Stage 2 — cross-encoder reranking (ms-marco-MiniLM-L-6-v2)
#             re-scores each (query, chunk) pair and returns the
#             best TOP_K results. Both models run locally on CPU;
#             no embedding API cost.
# ============================================================

import json
import math
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from google.adk.agents import Agent

import config
import db

TOP_K = 3          # final chunks returned to the agent
RERANK_POOL = 10   # candidates passed from bi-encoder to cross-encoder

BI_ENCODER_MODEL = "all-MiniLM-L6-v2"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

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


async def search_knowledge_base(query: str) -> dict:
    """Retrieve and rerank the most relevant KB chunks for a query.

    Stage 1: embed the query with a bi-encoder and compute cosine
    similarity against every stored embedding (computed by
    scripts/index_kb.py with the same model).  Top RERANK_POOL
    candidates advance to stage 2.

    Stage 2: a cross-encoder scores each (query, chunk) pair with
    full attention over both texts — much more precise than dot
    product alone.  Top TOP_K results are returned.

    Args:
        query (str): the caller's question or keywords.

    Returns:
        dict: {"success": bool, "chunks": [str, ...]} — best-matching text.
    """
    query_vec = _get_bi_encoder().encode(query).tolist()

    rows = await db.fetch_all(
        "SELECT content, embedding FROM kb_chunks WHERE embedding IS NOT NULL"
    )
    if not rows:
        return {"success": True, "chunks": []}

    # Stage 1 — cosine similarity ranking
    scored: list[tuple[float, str]] = []
    for row in rows:
        try:
            chunk_vec = json.loads(row["embedding"])
        except (json.JSONDecodeError, TypeError):
            continue
        sim = _cosine(query_vec, chunk_vec)
        scored.append((sim, row["content"]))

    scored.sort(reverse=True)
    candidates = [content for _, content in scored[:RERANK_POOL]]

    # Stage 2 — cross-encoder reranking (skipped if only one candidate)
    if len(candidates) > 1:
        ce_scores = _get_cross_encoder().predict(
            [(query, c) for c in candidates]
        ).tolist()
        reranked = [c for _, c in sorted(zip(ce_scores, candidates), reverse=True)]
    else:
        reranked = candidates

    return {"success": True, "chunks": reranked[:TOP_K]}


kb_agent = Agent(
    name="kb_agent",
    model=config.get_model(),
    description="Answers product/pricing/FAQ questions from the knowledge base.",
    instruction="""
You are the Knowledge Retrieval agent for CallOS.

When asked a product, pricing, or policy question:
1) Call search_knowledge_base with the key terms from the question.
2) Answer ONLY from the returned chunks. Keep it to 1-2 sentences.
3) If no chunk answers it, say you'll have a specialist follow up — do not guess.
""",
    tools=[search_knowledge_base],
)
