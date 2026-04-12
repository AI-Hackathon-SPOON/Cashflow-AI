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
