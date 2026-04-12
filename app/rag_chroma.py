"""Persistent local ChromaDB for fraud RAG (learned patterns + indexed flagged rows)."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

COLLECTION_NAME = "fraud_rag"


def chroma_persist_path() -> Path:
    """On-disk location for Chroma (gitignored under ``data/chroma_rag``)."""
    return Path(__file__).resolve().parent.parent / "data" / "chroma_rag"


def _openai_key() -> str | None:
    k = os.environ.get("OPENAI_API_KEY", "").strip()
    return k or None


def get_chroma_collection(persist_directory: Path | None = None):
    """
    Return the shared Chroma collection using OpenAI ``text-embedding-3-small``.

    :raises: ``ValueError`` if ``OPENAI_API_KEY`` is missing.
    """
    import chromadb
    from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

    key = _openai_key()
    if not key:
        raise ValueError(
            "OPENAI_API_KEY is required for Chroma embeddings (text-embedding-3-small). "
            "Set it in the environment or `.streamlit/secrets.toml`."
        )
    persist = persist_directory or chroma_persist_path()
    persist.mkdir(parents=True, exist_ok=True)
    ef = OpenAIEmbeddingFunction(api_key=key, model_name="text-embedding-3-small")
    client = chromadb.PersistentClient(path=str(persist))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"app": "cashflow_fraud_streamlit"},
    )


def _chroma_safe_metadata(meta: dict[str, Any]) -> dict[str, Any]:
    """Chroma metadata values must be str, int, float, or bool."""
    clean: dict[str, Any] = {}
    for key, val in meta.items():
        if val is None:
            continue
        sk = str(key)[:200]
        if isinstance(val, (str, int, float, bool)):
            clean[sk] = val
        elif isinstance(val, list):
            clean[sk] = ",".join(str(x) for x in val)[:8000]
        else:
            clean[sk] = str(val)[:8000]
    return clean


def snapshot_to_query_text(txn: dict[str, Any]) -> str:
    """Build a single string for similarity search from a transaction snapshot."""
    parts = [
        str(txn.get("fraud_reasons", "")),
        str(txn.get("source_account", "")),
        str(txn.get("destination_account", "")),
        str(txn.get("amount", "")),
        str(txn.get("currency", "")),
        str(txn.get("description", ""))[:800],
    ]
    return "\n".join(p for p in parts if p).strip() or "empty"


def upsert_learned_patterns(docs: list[dict[str, Any]]) -> int:
    """
    Upsert **learned pattern** documents (output shape of ``build_vector_kb_document``).

    Each dict should have ``text_for_embedding`` and ``metadata``; optional ``id`` (stable string).
    Metadata gains ``doc_kind="learned_pattern"``.
    """
    col = get_chroma_collection()
    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict[str, Any]] = []
    used: set[str] = set()

    for raw in docs:
        if not isinstance(raw, dict):
            continue
        text = str(raw.get("text_for_embedding", "")).strip()
        if not text:
            continue
        meta_raw = raw.get("metadata") if isinstance(raw.get("metadata"), dict) else {}
        meta = _chroma_safe_metadata({**meta_raw, "doc_kind": "learned_pattern"})
        eid = str(raw.get("id", "")).strip()
        if not eid:
            cid = str(meta_raw.get("case_id", "") or "")
            h = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
            eid = f"pat:{cid}:{h}" if cid else f"pat:{h}"
        while eid in used:
            eid = f"{eid}:x"
        used.add(eid)
        ids.append(eid)
        documents.append(text)
        metadatas.append(meta)

    if not ids:
        return 0
    col.upsert(ids=ids, documents=documents, metadatas=metadatas)
    return len(ids)


def upsert_flagged_snapshots(items: list[dict[str, Any]]) -> int:
    """
    Upsert **flagged transaction** rows for retrieval (same shape as ``build_flagged_rag_items`` elements).

    Each item: ``transaction_id``, ``transaction`` (dict). Stored with ``doc_kind="flagged_txn"``.
    Stable id ``flag:<transaction_id>`` so re-indexing updates in place.
    """
    col = get_chroma_collection()
    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict[str, Any]] = []

    for raw in items:
        if not isinstance(raw, dict):
            continue
        tid = str(raw.get("transaction_id", "")).strip()
        txn = raw.get("transaction")
        if not tid or not isinstance(txn, dict):
            continue
        text = snapshot_to_query_text(txn)
        if not text or text == "empty":
            continue
        meta = _chroma_safe_metadata(
            {
                "doc_kind": "flagged_txn",
                "transaction_id": tid,
                "fraud_score": int(txn.get("fraud_score", 0)),
                "currency": str(txn.get("currency", ""))[:32],
            }
        )
        ids.append(f"flag:{tid}")
        documents.append(text)
        metadatas.append(meta)

    if not ids:
        return 0
    col.upsert(ids=ids, documents=documents, metadatas=metadatas)
    return len(ids)


def search_similar(query: str, k: int = 8) -> list[dict[str, Any]]:
    """Run similarity search; return dicts suitable as ``kb_retrievals`` entries."""
    q = (query or "").strip()
    if not q:
        return []
    col = get_chroma_collection()
    n = max(1, min(int(k), 50))
    r = col.query(
        query_texts=[q],
        n_results=n,
        include=["documents", "metadatas", "distances"],
    )
    hits: list[dict[str, Any]] = []
    row_ids = r.get("ids") or []
    if not row_ids or not row_ids[0]:
        return hits
    for i, eid in enumerate(row_ids[0]):
        md = (r.get("metadatas") or [[{}]])[0][i] or {}
        docs = (r.get("documents") or [[""]])[0]
        doc = docs[i] if i < len(docs) else ""
        dists = r.get("distances") or [[None]]
        dist = dists[0][i] if i < len(dists[0]) else None
        case_id = md.get("case_id") or md.get("transaction_id") or eid
        hits.append(
            {
                "case_id": str(case_id),
                "outcome": str(md.get("outcome", "")),
                "summary": (doc or "")[:4000],
                "pattern_type": str(md.get("pattern_type", "")),
                "doc_kind": str(md.get("doc_kind", "")),
                "chromadb_id": str(eid),
                "vector_distance": dist,
            }
        )
    return hits


def build_kb_map_from_chroma(
    scored_df: pd.DataFrame,
    *,
    k_per_txn: int = 5,
    max_flagged: int = 100,
) -> dict[str, list[dict[str, Any]]]:
    """For each flagged row (up to ``max_flagged``), query Chroma and build ``transaction_id → hits``."""
    from app.fraud_report_payload import build_flagged_rag_items

    items = build_flagged_rag_items(scored_df)[: max(1, int(max_flagged))]
    out: dict[str, list[dict[str, Any]]] = {}
    for it in items:
        tid = str(it.get("transaction_id", ""))
        txn = it.get("transaction") if isinstance(it.get("transaction"), dict) else {}
        q = snapshot_to_query_text(txn)
        out[tid] = search_similar(q, k=k_per_txn)
    return out


def global_kb_from_chroma_for_flagged(
    scored_df: pd.DataFrame,
    *,
    k: int = 12,
    max_chars: int = 6000,
) -> list[dict[str, Any]]:
    """
    One Chroma query built from distinct ``fraud_reasons`` strings on flagged rows;
    returns top hits as a list for ``kb_global_context``.
    """
    if scored_df.empty or "is_flagged" not in scored_df.columns:
        return []
    fl = scored_df.loc[scored_df["is_flagged"]]
    if fl.empty:
        return []
    reasons = fl["fraud_reasons"].astype(str).drop_duplicates().head(25).tolist()
    blob = "\n".join(reasons)[:max_chars]
    return search_similar(blob, k=k)


def chroma_document_count() -> int:
    """Return row count in the collection, or ``-1`` on failure."""
    try:
        return int(get_chroma_collection().count())
    except Exception:
        return -1


def list_stored_documents(*, limit: int = 150, offset: int = 0) -> tuple[list[dict[str, Any]], int]:
    """
    Return a page of stored vectors as flat dicts (for tables / inspection).

    Each row includes ``id``, ``document`` (full stored text), and metadata fields such as
    ``doc_kind``, ``outcome``, ``case_id``, ``transaction_id``. Also returns ``total`` (collection count).

    :raises: ``ValueError`` if ``OPENAI_API_KEY`` is missing (same as ``get_chroma_collection``).
    """
    col = get_chroma_collection()
    total = int(col.count())
    lim = max(1, min(int(limit), 500))
    off = max(0, int(offset))
    r = col.get(include=["documents", "metadatas"], limit=lim, offset=off)
    ids = r.get("ids") or []
    docs = r.get("documents") or []
    metas = r.get("metadatas") or []
    rows: list[dict[str, Any]] = []
    for i, eid in enumerate(ids):
        md = metas[i] if i < len(metas) and isinstance(metas[i], dict) else {}
        doc = docs[i] if i < len(docs) else ""
        rows.append(
            {
                "id": str(eid),
                "doc_kind": md.get("doc_kind", ""),
                "outcome": md.get("outcome", ""),
                "case_id": md.get("case_id", ""),
                "transaction_id": md.get("transaction_id", ""),
                "fraud_score": md.get("fraud_score", ""),
                "pattern_type": md.get("pattern_type", ""),
                "document": doc or "",
            }
        )
    return rows, total


def parse_and_upsert_learned_json(json_text: str) -> tuple[int, str | None]:
    """Parse a JSON array of pattern objects; upsert. Returns ``(count, error_message)``."""
    try:
        payload = json.loads(json_text)
    except json.JSONDecodeError as e:
        return 0, str(e)
    if not isinstance(payload, list):
        return 0, "JSON must be an array of objects."
    n = upsert_learned_patterns([x for x in payload if isinstance(x, dict)])
    return n, None
