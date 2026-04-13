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


def search_similar_batch(query_texts: list[str], k: int = 8) -> list[list[dict[str, Any]]]:
    """Like :func:`search_similar`, but one query list per input string (batched Chroma call)."""
    texts = [(t or "").strip() or "empty" for t in query_texts]
    if not texts:
        return []
    col = get_chroma_collection()
    n = max(1, min(int(k), 50))
    r = col.query(
        query_texts=texts,
        n_results=n,
        include=["documents", "metadatas", "distances"],
    )
    row_ids = r.get("ids") or []
    batch_n = len(texts)
    out: list[list[dict[str, Any]]] = [[] for _ in range(batch_n)]
    for bi in range(batch_n):
        ids_row = row_ids[bi] if bi < len(row_ids) else []
        if not ids_row:
            continue
        mds = (r.get("metadatas") or [[]])[bi] if bi < len(r.get("metadatas") or []) else []
        docs = (r.get("documents") or [[]])[bi] if bi < len(r.get("documents") or []) else []
        dists = (r.get("distances") or [[]])[bi] if bi < len(r.get("distances") or []) else []
        for i, eid in enumerate(ids_row):
            md = mds[i] if i < len(mds) and isinstance(mds[i], dict) else {}
            doc = docs[i] if i < len(docs) else ""
            dist = dists[i] if i < len(dists) else None
            case_id = md.get("case_id") or md.get("transaction_id") or eid
            out[bi].append(
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
    return out


def _row_series_to_txn_dict(row: pd.Series) -> dict[str, Any]:
    """Shape expected by :func:`snapshot_to_query_text` from a scored dataframe row."""
    return {
        "fraud_reasons": str(row.get("fraud_reasons", "")),
        "source_account": str(row.get("source_account", "")),
        "destination_account": str(row.get("destination_account", "")),
        "amount": row.get("amount", ""),
        "currency": str(row.get("currency", "")),
        "description": str(row.get("description", "")),
    }


def _outcome_confirmed_fraud(outcome: str) -> bool:
    o = (outcome or "").strip().lower()
    if "false" in o and "positive" in o:
        return False
    return "fraud" in o and ("confirm" in o or "confirmed" in o)


def _outcome_false_positive(outcome: str) -> bool:
    o = (outcome or "").strip().lower()
    return "false" in o and "positive" in o


def _hit_vector_distance(h: dict[str, Any]) -> float:
    d = h.get("vector_distance")
    if d is None:
        return 9e9
    try:
        return float(d)
    except (TypeError, ValueError):
        return 9e9


def _hit_passes_distance_max(h: dict[str, Any], distance_max: float | None) -> bool:
    if distance_max is None:
        return True
    d = h.get("vector_distance")
    if d is None:
        return True
    try:
        return float(d) <= float(distance_max)
    except (TypeError, ValueError):
        return True


def _learned_precedent_candidates(hits: list[dict[str, Any]], distance_max: float | None) -> list[dict[str, Any]]:
    """Only ``learned_pattern`` docs within ``distance_max`` (when set)."""
    return [
        h
        for h in hits
        if h.get("doc_kind") == "learned_pattern" and _hit_passes_distance_max(h, distance_max)
    ]


def _best_hit_by_distance(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not candidates:
        return None
    return min(candidates, key=_hit_vector_distance)


def enhance_scored_with_chroma_kb(
    scored_df: pd.DataFrame,
    *,
    k: int = 5,
    max_rows: int = 400,
    batch_size: int = 24,
    confirmed_boost: int = 12,
    distance_max: float | None = 0.55,
    false_positive_penalty: int = 0,
    false_positive_penalty_cap: int = 20,
    fp_score_floor: int = 0,
) -> pd.DataFrame:
    """
    Re-query Chroma for each row (rule+graph text + transaction fields) and **adjust** ``fraud_score`` /
    ``fraud_reasons`` / ``is_flagged`` using institutional **learned patterns** only.

    - **Confirmed fraud** precedent: adds up to ``confirmed_boost`` when the closest qualifying learned pattern
      matches (within ``distance_max`` when set).
    - **False positive** precedent: if ``false_positive_penalty`` is **0** (default), appends an informational line
      only. If **> 0**, subtracts points (capped by ``false_positive_penalty_cap`` and by ``base_score - fp_score_floor``)
      only when **no** qualifying confirmed-fraud learned pattern exists for that row — so fraud precedent always wins.

    ``kb_score_boost`` holds the **signed** delta applied to the base rule+graph score (may be negative).

    Empty collection or API issues: returns a copy of the input unchanged (caller may catch exceptions).
    """
    if scored_df.empty or "fraud_score" not in scored_df.columns:
        return scored_df.copy()

    out = scored_df.copy().reset_index(drop=True)
    n = min(len(out), max(1, int(max_rows)))
    bs = max(1, min(int(batch_size), 64))

    fp_pen = max(0, int(false_positive_penalty))
    fp_cap = max(0, int(false_positive_penalty_cap))
    floor = max(0, min(100, int(fp_score_floor)))

    deltas: list[int] = [0] * len(out)
    extra_reasons: list[list[str]] = [[] for _ in range(len(out))]

    for start in range(0, n, bs):
        chunk = out.iloc[start : start + bs]
        queries = [snapshot_to_query_text(_row_series_to_txn_dict(r)) for _, r in chunk.iterrows()]
        try:
            batch_hits = search_similar_batch(queries, k=k)
        except Exception:
            continue
        for j, (_, row) in enumerate(chunk.iterrows()):
            pos = start + j
            if pos >= len(out):
                break
            hits = batch_hits[j] if j < len(batch_hits) else []
            if not hits:
                continue

            learned = _learned_precedent_candidates(hits, distance_max)
            fraud_cands = [h for h in learned if _outcome_confirmed_fraud(str(h.get("outcome", "")))]
            fp_cands = [h for h in learned if _outcome_false_positive(str(h.get("outcome", "")))]

            best_fraud = _best_hit_by_distance(fraud_cands)
            best_fp = _best_hit_by_distance(fp_cands)

            base_sc = int(out.loc[pos, "fraud_score"])

            if best_fraud is not None:
                cid = str(best_fraud.get("case_id", ""))[:80]
                bump = max(0, min(int(confirmed_boost), 100))
                deltas[pos] += bump
                extra_reasons[pos].append(
                    f"KB (Chroma): strong match to confirmed-fraud precedent `{cid}` (learned pattern)."
                )
            elif best_fp is not None:
                cid = str(best_fp.get("case_id", ""))[:80]
                if fp_pen > 0:
                    headroom = max(0, base_sc - floor)
                    applied = min(fp_pen, fp_cap, headroom)
                    if applied > 0:
                        deltas[pos] -= applied
                        extra_reasons[pos].append(
                            f"KB (Chroma): close false-positive precedent `{cid}` — score reduced by {applied} "
                            f"(floor={floor}; human review still recommended)."
                        )
                    else:
                        extra_reasons[pos].append(
                            f"KB (Chroma): false-positive precedent `{cid}` matches but score already at floor ({floor}); "
                            "no reduction applied."
                        )
                else:
                    extra_reasons[pos].append(
                        f"KB (Chroma): nearest learned precedent `{cid}` was a false positive — contextual hint only "
                        "(FP score reduction disabled)."
                    )

    addons: list[str] = []
    new_scores: list[int] = []
    new_flags: list[bool] = []
    new_text: list[str] = []
    for i in range(len(out)):
        base = int(out.loc[i, "fraud_score"])
        adj = max(0, min(base + deltas[i], 100))
        new_scores.append(adj)
        base_msg = str(out.loc[i, "fraud_reasons"])
        add = extra_reasons[i]
        addon_txt = " ".join(add).strip()
        addons.append(addon_txt)
        new_text.append(base_msg if not addon_txt else f"{base_msg} {addon_txt}".strip())
        new_flags.append(adj >= 50)

    out["fraud_score"] = new_scores
    out["fraud_reasons"] = new_text
    out["is_flagged"] = new_flags
    out["kb_score_boost"] = deltas
    out["kb_reason_addon"] = addons
    return out


def kb_score_layer_fingerprint(scored_df: pd.DataFrame) -> str:
    """Fingerprint of **base** rule+graph scores so KB adjustments can be replayed safely across Streamlit reruns."""
    if scored_df.empty:
        return ""
    pairs: list[tuple[str, str, str, int, str]] = []
    for _, r in scored_df.iterrows():
        ts = r.get("timestamp")
        ts_s = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)
        pairs.append(
            (
                str(r.get("transaction_id", "")),
                f"{float(r.get('amount', 0)):.6f}",
                ts_s,
                int(r.get("fraud_score", 0)),
                str(r.get("fraud_reasons", "")),
            )
        )
    pairs.sort()
    return hashlib.sha256(json.dumps(pairs, ensure_ascii=False).encode("utf-8")).hexdigest()


def pack_kb_score_layer(enhanced_df: pd.DataFrame, *, base_fp: str) -> dict[str, Any]:
    """Build a small session-storable dict from :func:`enhance_scored_with_chroma_kb` output."""
    by_tid: dict[str, dict[str, Any]] = {}
    if "transaction_id" not in enhanced_df.columns:
        return {"fp": base_fp, "by_tid": {}}
    for _, r in enhanced_df.iterrows():
        tid = str(r.get("transaction_id", "")).strip()
        if not tid:
            continue
        try:
            b = int(r.get("kb_score_boost", 0))
        except (TypeError, ValueError):
            b = 0
        add = str(r.get("kb_reason_addon", "")).strip()
        if b != 0 or add:
            by_tid[tid] = {"boost": b, "addon": add}
    return {"fp": base_fp, "by_tid": by_tid}


def apply_kb_score_layer(scored_df: pd.DataFrame, layer_pkg: dict[str, Any] | None) -> pd.DataFrame:
    """
    Replay KB score / reason adjustments on a freshly computed dataframe.

    ``layer_pkg`` shape: ``{"fp": str, "by_tid": { transaction_id: {"boost": int, "addon": str} }}``.
    """
    out = scored_df.copy().reset_index(drop=True)
    if not layer_pkg or scored_df.empty:
        out["kb_score_boost"] = 0
        return out
    fp = kb_score_layer_fingerprint(scored_df)
    if layer_pkg.get("fp") != fp:
        out["kb_score_boost"] = 0
        return out
    by_tid = layer_pkg.get("by_tid")
    if not isinstance(by_tid, dict):
        out["kb_score_boost"] = 0
        return out

    boosts_col: list[int] = []
    for i, row in out.iterrows():
        tid = str(row.get("transaction_id", "")).strip()
        spec = by_tid.get(tid) if tid else None
        b = 0
        add = ""
        if isinstance(spec, dict):
            try:
                b = int(spec.get("boost", 0))
            except (TypeError, ValueError):
                b = 0
            add = str(spec.get("addon", "")).strip()
        boosts_col.append(b)
        if b != 0 or add:
            base_score = int(row["fraud_score"])
            base_reason = str(row["fraud_reasons"])
            ns = max(0, min(100, base_score + b))
            out.at[i, "fraud_score"] = ns
            if add:
                out.at[i, "fraud_reasons"] = f"{base_reason} {add}".strip()
            out.at[i, "is_flagged"] = ns >= 50

    out["kb_score_boost"] = boosts_col
    return out


def build_kb_map_from_chroma(
    scored_df: pd.DataFrame,
    *,
    k_per_txn: int = 5,
    max_flagged: int = 100,
    flagged_only: bool = True,
) -> dict[str, list[dict[str, Any]]]:
    """For each selected row (flagged by default), query Chroma and build ``transaction_id → hits``."""
    from app.fraud_report_payload import build_scored_rag_items

    items = build_scored_rag_items(scored_df, flagged_only=flagged_only)[: max(1, int(max_flagged))]
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
