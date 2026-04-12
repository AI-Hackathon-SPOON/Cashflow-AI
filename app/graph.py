"""Graph aggregation for D3: labels, date filter, node drill-down payload."""

from __future__ import annotations

from collections import defaultdict
from datetime import date
from typing import Any

import pandas as pd


def build_account_graph_labels(scored_df: pd.DataFrame) -> dict[str, str]:
    if scored_df.empty or "organization_name" not in scored_df.columns:
        return {}
    df = scored_df.copy()
    cleaned = df["organization_name"].fillna("").astype(str).str.strip().replace("", pd.NA)
    if cleaned.isna().all():
        return {}
    df = df.assign(_org_clean=cleaned)
    labels: dict[str, str] = {}
    acc_series = pd.concat([df["source_account"].astype(str), df["destination_account"].astype(str)])
    for acc in sorted(pd.unique(acc_series)):
        mask = (df["source_account"].astype(str) == acc) | (df["destination_account"].astype(str) == acc)
        vals = df.loc[mask, "_org_clean"].dropna()
        if vals.empty:
            continue
        mode = vals.astype(str).mode()
        if not mode.empty:
            labels[acc] = str(mode.iloc[0])
    return labels


def filter_scored_df_by_graph_dates(
    scored_df: pd.DataFrame,
    mode: str,
    date_a: date | None,
    date_b: date | None,
) -> pd.DataFrame:
    if scored_df.empty or mode == "all":
        return scored_df
    ts = pd.to_datetime(scored_df["timestamp"], utc=True)
    day = ts.dt.date
    if mode == "on_or_after" and date_a is not None:
        return scored_df.loc[day >= date_a].copy()
    if mode == "on_or_before" and date_a is not None:
        return scored_df.loc[day <= date_a].copy()
    if mode == "between" and date_a is not None and date_b is not None:
        lo, hi = (date_a, date_b) if date_a <= date_b else (date_b, date_a)
        return scored_df.loc[(day >= lo) & (day <= hi)].copy()
    return scored_df


def build_node_transaction_details(
    graph_df: pd.DataFrame,
    *,
    max_per_node: int = 80,
) -> dict[str, list[dict[str, Any]]]:
    """Per account id, recent transactions touching that node (for D3 click panel)."""
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    if graph_df.empty:
        return {}

    rows = graph_df.sort_values("timestamp", ascending=False)
    for _, r in rows.iterrows():
        s = str(r["source_account"])
        d = str(r["destination_account"])
        ts = r["timestamp"]
        ts_s = ts.isoformat() if isinstance(ts, pd.Timestamp) else str(ts)
        base: dict[str, Any] = {
            "transaction_id": str(r["transaction_id"]),
            "amount": float(r["amount"]),
            "abs_amount": float(r["abs_amount"]),
            "currency": str(r["currency"]),
            "timestamp": ts_s,
            "fraud_score": int(r["fraud_score"]),
            "is_flagged": bool(r["is_flagged"]),
            "fraud_reasons": str(r["fraud_reasons"]),
        }
        if len(out[s]) < max_per_node:
            out[s].append({**base, "direction": "out", "counterparty": d})
        if len(out[d]) < max_per_node:
            out[d].append({**base, "direction": "in", "counterparty": s})

    return dict(out)


def build_cashflow_graph_payload(
    scored_df: pd.DataFrame,
    *,
    node_radius: float = 13.0,
    max_transactions_per_node: int = 80,
) -> dict[str, Any]:
    cr = float(max(node_radius * 1.95, node_radius + 10))
    if scored_df.empty:
        return {
            "nodes": [],
            "links": [],
            "node_radius": float(node_radius),
            "collision_radius": cr,
            "node_details": {},
        }

    grouped = (
        scored_df.groupby(["source_account", "destination_account"], as_index=False)
        .agg(
            total_amount=("abs_amount", "sum"),
            tx_count=("transaction_id", "count"),
            flagged_tx=("is_flagged", "sum"),
        )
        .sort_values("total_amount", ascending=False)
    )

    flagged_nodes = set(
        scored_df.loc[scored_df["is_flagged"], ["source_account", "destination_account"]]
        .stack()
        .astype(str)
        .tolist()
    )

    graph_labels = build_account_graph_labels(scored_df)
    node_ids = sorted(set(grouped["source_account"]).union(set(grouped["destination_account"])))
    nodes = [
        {
            "id": str(node),
            "label": graph_labels.get(str(node), str(node)),
            "has_flagged_tx": str(node) in flagged_nodes,
        }
        for node in node_ids
    ]

    links = []
    for _, row in grouped.iterrows():
        links.append(
            {
                "source": str(row["source_account"]),
                "target": str(row["destination_account"]),
                "amount": float(round(row["total_amount"], 2)),
                "tx_count": int(row["tx_count"]),
                "flagged_tx": int(row["flagged_tx"]),
                "is_flagged": int(row["flagged_tx"]) > 0,
            }
        )

    node_details = build_node_transaction_details(
        scored_df,
        max_per_node=max_transactions_per_node,
    )

    return {
        "nodes": nodes,
        "links": links,
        "node_radius": float(node_radius),
        "collision_radius": cr,
        "node_details": node_details,
    }
