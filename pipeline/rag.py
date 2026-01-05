# pipeline/rag.py
from __future__ import annotations

import os
from typing import List, Dict, Any, Optional

from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from langchain_core.runnables import RunnableLambda

# Import the InstructionSpec schema from your parser module
# (If your InstructionSpec is currently in pipeline/core.py, keep this import for now.)
from pipeline.core import InstructionSpec


# -----------------------------
# Config (set via env vars)
# -----------------------------
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "amazon-api-ops-e5base")
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "intfloat/e5-base-v2")

if not PINECONE_API_KEY:
    raise RuntimeError(
        "Missing PINECONE_API_KEY environment variable. "
        "Set it before running the app."
    )


# -----------------------------
# Initialize clients (module-level singletons)
# -----------------------------
_pc = Pinecone(api_key=PINECONE_API_KEY)
_index = _pc.Index(PINECONE_INDEX_NAME)
_embed_model = SentenceTransformer(EMBED_MODEL_NAME)


# -----------------------------
# Helpers
# -----------------------------
def parse_param_strs(param_strs: List[str]) -> List[Dict[str, Any]]:
    """Turn ['name|in|required|type', ...] into structured dicts."""
    out: List[Dict[str, Any]] = []
    for s in param_strs or []:
        parts = s.split("|")
        if len(parts) != 4:
            continue
        name, pin, req, typ = parts
        out.append(
            {
                "name": name,
                "in": pin,
                "required": (req.strip().lower() == "required"),
                "type": typ,
            }
        )
    return out


def get_top_apis(query_text: str, top_k: int = 3, tag_filter: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """
    Embed the query, search Pinecone, return top_k API operations with metadata.
    Optionally filter by tags.
    """
    # 1) embed the query
    qvec = _embed_model.encode(["query: " + query_text], normalize_embeddings=True)[0].tolist()

    # 2) optional metadata filter
    pine_filter = None
    if tag_filter:
        pine_filter = {"tags": {"$in": tag_filter}}

    # 3) query Pinecone
    res = _index.query(
        vector=qvec,
        top_k=top_k,
        include_values=False,
        include_metadata=True,
        filter=pine_filter,
    )

    # 4) shape results
    results: List[Dict[str, Any]] = []
    for match in res.matches:
        md = match.metadata or {}
        params = parse_param_strs(md.get("parameters"))
        full_url = (md.get("base_url") or "") + (md.get("path") or "")

        results.append(
            {
                "score": match.score,
                "method": md.get("method"),
                "path": md.get("path"),
                "full_url": full_url,
                "operationId": md.get("operationId"),
                "api_name": md.get("api_name"),
                "api_version": md.get("api_version"),
                "tags": md.get("tags"),
                "parameters": params,
                "responses": md.get("responses"),
                "preview": md.get("preview", ""),
            }
        )
    return results


def build_query_from_instruction(spec: InstructionSpec) -> str:
    """InstructionSpec -> search string for Pinecone."""
    ent = ", ".join(spec.entities[:5])
    tasks = ", ".join(spec.subtasks[:5])
    return (
        f"{spec.intent}. "
        f"Key entities: {ent}. "
        f"Needed capabilities: {tasks}. "
        f"Return APIs that directly enable these actions."
    )


def rag_node_fn(spec: InstructionSpec, top_k: int = 3, tag_filter: Optional[List[str]] = None) -> Dict[str, Any]:
    """LangChain runnable target: InstructionSpec -> RAG results dict."""
    query = build_query_from_instruction(spec)
    hits = get_top_apis(query_text=query, top_k=top_k, tag_filter=tag_filter)
    return {
        "query_used": query,
        "top_k": len(hits),
        "results": hits,
    }


# Default RAG node used in the full chain
rag_node = RunnableLambda(lambda spec: rag_node_fn(spec, top_k=3))
