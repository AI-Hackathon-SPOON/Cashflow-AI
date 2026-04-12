"""Graph aggregation for D3: labels, date filter, node merge, drill-down payload."""

from __future__ import annotations

import re
from collections import defaultdict
from datetime import date
from typing import Any, Iterable

import pandas as pd


def _slug_org_label(label: str, used_slugs: set[str]) -> str:
    base = re.sub(r"[^a-zA-Z0-9]+", "_", label.strip())[:48].strip("_").lower() or "org"
    cand = base
    n = 0
    while cand in used_slugs:
        n += 1
        cand = f"{base}_{n}"
    used_slugs.add(cand)
    return cand


def build_merge_canonical_mapping(
    account_ids: Iterable[str],
    graph_labels: dict[str, str],
) -> tuple[dict[str, str], dict[str, list[str]], dict[str, str], int]:
    """
    Map raw account ids to canonical graph node ids.

    Only merges when **two or more accounts share the same non-empty organization label**
    (same business entity, different technical ids). Otherwise each account stays its own node.

    Returns:
        account_to_canonical, canonical_to_accounts, canonical_display_label, num_merged_groups
    """
    ids = sorted({str(a) for a in account_ids})
    groups_by_label: dict[str, list[str]] = defaultdict(list)
    for acc in ids:
        lab = (graph_labels.get(acc) or "").strip()
        if lab:
            groups_by_label[lab].append(acc)
        else:
            groups_by_label[f"__id__{acc}"].append(acc)

    account_to_canonical: dict[str, str] = {}
    canonical_to_accounts: dict[str, list[str]] = {}
    canonical_display: dict[str, str] = {}
    used_slugs: set[str] = set()
    merged_groups = 0

    for lab_key, members in groups_by_label.items():
        if lab_key.startswith("__id__"):
            acc = members[0]
            account_to_canonical[acc] = acc
            canonical_to_accounts[acc] = [acc]
            canonical_display[acc] = graph_labels.get(acc) or acc
        elif len(members) == 1:
            acc = members[0]
            account_to_canonical[acc] = acc
            canonical_to_accounts[acc] = [acc]
            canonical_display[acc] = lab_key
        else:
            merged_groups += 1
            slug = _slug_org_label(lab_key, used_slugs)
            canon = f"__org__:{slug}"
            for acc in members:
                account_to_canonical[acc] = canon
            canonical_to_accounts[canon] = sorted(members)
            canonical_display[canon] = lab_key

    return account_to_canonical, canonical_to_accounts, canonical_display, merged_groups


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
        # Use organization only on rows where this account is the **source**, so counterparties
        # do not inherit the payer's org name (which would incorrectly merge suppliers into the org).
        vals = df.loc[df["source_account"].astype(str) == acc, "_org_clean"].dropna()
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


def merge_node_transaction_details(
    raw_by_account: dict[str, list[dict[str, Any]]],
    canonical_to_accounts: dict[str, list[str]],
    *,
    max_per_canonical: int,
) -> dict[str, list[dict[str, Any]]]:
    """Attach merged transaction rows to each canonical node id (for D3 click panel)."""
    out: dict[str, list[dict[str, Any]]] = {}
    for canon, members in canonical_to_accounts.items():
        rows: list[dict[str, Any]] = []
        for m in members:
            rows.extend(raw_by_account.get(m, []))
        rows.sort(key=lambda r: r.get("timestamp") or "", reverse=True)
        out[canon] = rows[:max_per_canonical]
    return out


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
            "merge_applied": False,
            "merged_entity_groups": 0,
        }

    graph_labels = build_account_graph_labels(scored_df)
    raw_accounts = set(scored_df["source_account"].astype(str)) | set(scored_df["destination_account"].astype(str))
    account_to_canon, canon_to_accounts, canon_display, n_merge_groups = build_merge_canonical_mapping(
        raw_accounts, graph_labels
    )

    s_canon = scored_df["source_account"].astype(str).map(account_to_canon)
    d_canon = scored_df["destination_account"].astype(str).map(account_to_canon)
    flow = scored_df.assign(_cs=s_canon, _cd=d_canon).loc[s_canon != d_canon]
    grouped = (
        flow.groupby(["_cs", "_cd"], as_index=False)
        .agg(
            total_amount=("abs_amount", "sum"),
            tx_count=("transaction_id", "count"),
            flagged_tx=("is_flagged", "sum"),
        )
        .sort_values("total_amount", ascending=False)
    )

    links = []
    for _, row in grouped.iterrows():
        flg_ct = int(row["flagged_tx"])
        links.append(
            {
                "source": str(row["_cs"]),
                "target": str(row["_cd"]),
                "amount": float(round(float(row["total_amount"]), 2)),
                "tx_count": int(row["tx_count"]),
                "flagged_tx": flg_ct,
                "is_flagged": flg_ct > 0,
            }
        )

    flagged_raw_accounts = set(
        scored_df.loc[scored_df["is_flagged"], ["source_account", "destination_account"]]
        .stack()
        .astype(str)
        .tolist()
    )
    flagged_canonical = {account_to_canon.get(str(a), str(a)) for a in flagged_raw_accounts}

    canonical_ids = set(account_to_canon.values())
    nodes = []
    for cid in sorted(canonical_ids):
        members = canon_to_accounts.get(cid, [cid])
        merged = len(members) > 1
        node: dict[str, Any] = {
            "id": cid,
            "label": canon_display.get(cid, cid),
            "has_flagged_tx": cid in flagged_canonical,
        }
        if merged:
            node["merged_accounts"] = members
        nodes.append(node)

    raw_details = build_node_transaction_details(
        scored_df,
        max_per_node=max_transactions_per_node,
    )
    merged_cap = max(max_transactions_per_node * 3, 120)
    node_details = merge_node_transaction_details(
        raw_details,
        canon_to_accounts,
        max_per_canonical=merged_cap,
    )

    merge_applied = n_merge_groups > 0
    return {
        "nodes": nodes,
        "links": links,
        "node_radius": float(node_radius),
        "collision_radius": cr,
        "node_details": node_details,
        "merge_applied": merge_applied,
        "merged_entity_groups": n_merge_groups,
    }
