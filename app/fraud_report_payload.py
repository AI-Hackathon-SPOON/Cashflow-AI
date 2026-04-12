"""Build categorized fraud alert dicts for ``generate_fraud_audit_report`` from scored transactions."""

from __future__ import annotations

from typing import Any

import pandas as pd


def build_categorized_fraud_alerts(scored_df: pd.DataFrame) -> dict[str, list[dict[str, Any]]]:
    """
    Map **flagged** rows into French-style category keys expected by the LLM report.

    Categories follow heuristics on ``fraud_reasons`` (English strings from ``fraud_scoring``).
    Each row is placed in the **first** matching category only.
    """
    cats: dict[str, list[dict[str, Any]]] = {
        "Flux_Circulaire": [],
        "Reseau_de_Mules": [],
        "Horodatage_Anormal": [],
        "Montants_Suspects": [],
        "Autres_Signaux": [],
    }
    if scored_df.empty or "is_flagged" not in scored_df.columns:
        return cats

    flagged = scored_df.loc[scored_df["is_flagged"]]
    for _, row in flagged.iterrows():
        r = str(row.get("fraud_reasons", "")).lower()
        item: dict[str, Any] = {
            "transaction_id": str(row.get("transaction_id", "")),
            "entites": [str(row.get("source_account", "")), str(row.get("destination_account", ""))],
            "montant": str(row.get("amount", "")),
            "devise": str(row.get("currency", "")),
            "score": int(row.get("fraud_score", 0)),
            "detail": str(row.get("fraud_reasons", "")),
        }
        if "bidirectional" in r or "round-trip" in r:
            cats["Flux_Circulaire"].append(item)
        elif "hub" in r or "pagerank" in r or "structural influence" in r:
            cats["Reseau_de_Mules"].append(item)
        elif "off-hours" in r:
            cats["Horodatage_Anormal"].append(item)
        elif "high amount" in r or "round amount" in r:
            cats["Montants_Suspects"].append(item)
        else:
            cats["Autres_Signaux"].append(item)

    return cats


def _row_snapshot(row: pd.Series) -> dict[str, Any]:
    """Compact, JSON-friendly dict for RAG / LLM prompts (one scored transaction row)."""
    ts = row.get("timestamp")
    ts_out = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)
    return {
        "transaction_id": str(row.get("transaction_id", "")),
        "timestamp": ts_out,
        "source_account": str(row.get("source_account", "")),
        "destination_account": str(row.get("destination_account", "")),
        "amount": float(row.get("amount", 0.0)),
        "currency": str(row.get("currency", "")),
        "channel": str(row.get("channel", "")),
        "description": str(row.get("description", "")),
        "organization_name": row.get("organization_name"),
        "fraud_score": int(row.get("fraud_score", 0)),
        "fraud_reasons": str(row.get("fraud_reasons", "")),
        "is_flagged": bool(row.get("is_flagged", False)),
    }


def build_flagged_rag_items(scored_df: pd.DataFrame) -> list[dict[str, Any]]:
    """
    One object per **flagged** row for ``generate_flagged_rag_explanations``.

    Each item is ``{"transaction_id", "transaction", "kb_retrievals"}`` with ``kb_retrievals`` initially empty.
    Your pipeline (e.g. n8n + vector DB) should fill ``kb_retrievals`` per ``transaction_id`` via
    :func:`attach_kb_by_transaction_id`.
    """
    if scored_df.empty or "is_flagged" not in scored_df.columns:
        return []
    out: list[dict[str, Any]] = []
    for _, row in scored_df.loc[scored_df["is_flagged"]].iterrows():
        tid = str(row.get("transaction_id", ""))
        out.append(
            {
                "transaction_id": tid,
                "transaction": _row_snapshot(row),
                "kb_retrievals": [],
            }
        )
    return out


def attach_kb_by_transaction_id(
    items: list[dict[str, Any]],
    kb_by_transaction_id: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """
    Copy ``items`` and set ``kb_retrievals`` from ``kb_by_transaction_id`` keys (string ids).

    Unknown ids keep ``kb_retrievals`` as in the original item (typically empty).
    """
    resolved: list[dict[str, Any]] = []
    for it in items:
        tid = str(it.get("transaction_id", ""))
        if tid in kb_by_transaction_id:
            kb = list(kb_by_transaction_id[tid])
        else:
            kb = list(it.get("kb_retrievals") or [])
        resolved.append({**it, "kb_retrievals": kb})
    return resolved


def build_vector_kb_document(
    text_for_embedding: str,
    *,
    outcome: str,
    case_id: str | None = None,
    pattern_type: str | None = None,
    resolution_notes: str | None = None,
    related_transaction_ids: list[str] | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Canonical payload for **indexing a learned pattern** in a vector DB (n8n / LangChain typically embeds
    ``text_for_embedding`` and stores ``metadata`` for filtering).

    After investigations, upsert documents with ``outcome`` in ``{"False Positive", "Confirmed Fraud", ...}`` so
    future RAG retrieval can steer explanations and CFO reporting.
    """
    metadata: dict[str, Any] = {"outcome": outcome}
    if case_id:
        metadata["case_id"] = case_id
    if pattern_type:
        metadata["pattern_type"] = pattern_type
    if resolution_notes:
        metadata["resolution_notes"] = resolution_notes
    if related_transaction_ids:
        metadata["related_transaction_ids"] = list(related_transaction_ids)
    if extra_metadata:
        metadata.update(extra_metadata)
    return {
        "text_for_embedding": str(text_for_embedding).strip(),
        "metadata": metadata,
    }
