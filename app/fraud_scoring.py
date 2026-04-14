"""Fraud scoring: rule-based heuristics plus NetworkX graph context and explanations."""

from __future__ import annotations

from collections import defaultdict, deque
import networkx as nx
import pandas as pd

# Skip expensive PageRank on very large graphs (keeps UI responsive).
_MAX_NODES_PAGERANK = 2800
_MAX_EDGES_PAGERANK = 45000


def _build_cashflow_multigraph(df: pd.DataFrame) -> nx.MultiDiGraph:
    G: nx.MultiDiGraph = nx.MultiDiGraph()
    for i, row in df.iterrows():
        u = str(row["source_account"])
        v = str(row["destination_account"])
        w = float(row["abs_amount"])
        tid = str(row.get("transaction_id", i))
        G.add_edge(u, v, key=tid, weight=max(w, 1e-9))
    return G


def _combined_degrees(G: nx.MultiDiGraph) -> dict[str, int]:
    deg: dict[str, int] = defaultdict(int)
    for n in G.nodes():
        deg[n] = G.in_degree(n) + G.out_degree(n)
    return dict(deg)


def _reciprocal_undirected_pairs(G: nx.MultiDiGraph) -> set[tuple[str, str]]:
    """Unordered pairs (a,b) with at least one edge in each direction, a != b."""
    pairs: set[tuple[str, str]] = set()
    for u, v, _ in G.edges(keys=True):
        if u == v:
            continue
        if G.has_edge(v, u):
            a, b = (u, v) if u < v else (v, u)
            pairs.add((a, b))
    return pairs


def _networkx_signals_for_rows(scored: pd.DataFrame, G: nx.MultiDiGraph) -> tuple[list[int], list[list[str]]]:
    """Return parallel lists: extra score points (0–100 cap applied later) and graph reason strings per row."""
    extras = [0] * len(scored)
    graph_reasons: list[list[str]] = [[] for _ in range(len(scored))]

    if G.number_of_nodes() == 0:
        return extras, graph_reasons

    combined = _combined_degrees(G)
    deg_values = list(combined.values()) if combined else [0]
    hub_threshold = float(pd.Series(deg_values).quantile(0.92)) if deg_values else 0.0
    hub_threshold = max(hub_threshold, 3.0)

    reciprocal_pairs = _reciprocal_undirected_pairs(G)

    pr: dict[str, float] = {}
    if G.number_of_nodes() <= _MAX_NODES_PAGERANK and G.number_of_edges() <= _MAX_EDGES_PAGERANK:
        try:
            pr = nx.pagerank(G, alpha=0.85, weight="weight")
        except Exception:
            pr = {}

    top_pr_nodes: set[str] = set()
    if pr:
        s = pd.Series(pr).sort_values(ascending=False)
        top_n = max(1, int(len(s) * 0.08))
        top_pr_nodes = set(s.head(top_n).index.astype(str))

    for pos, (_, row) in enumerate(scored.iterrows()):
        s_acc = str(row["source_account"])
        d_acc = str(row["destination_account"])
        pair = (s_acc, d_acc) if s_acc < d_acc else (d_acc, s_acc)

        if pair in reciprocal_pairs:
            extras[pos] += 22
            graph_reasons[pos].append(
                "Graph (NetworkX): bidirectional flow with this counterparty (round-trip pattern)."
            )

        if combined.get(s_acc, 0) >= hub_threshold:
            extras[pos] += 18
            graph_reasons[pos].append(
                f"Graph (NetworkX): source is a high-activity hub (degree {combined[s_acc]} ≥ {hub_threshold:.0f})."
            )
        if combined.get(d_acc, 0) >= hub_threshold:
            extras[pos] += 12
            graph_reasons[pos].append(
                f"Graph (NetworkX): destination is a high-activity hub (degree {combined[d_acc]} ≥ {hub_threshold:.0f})."
            )

        if s_acc in top_pr_nodes:
            extras[pos] += 14
            graph_reasons[pos].append("Graph (NetworkX): source has high structural influence (PageRank top tier).")
        if d_acc in top_pr_nodes:
            extras[pos] += 10
            graph_reasons[pos].append(
                "Graph (NetworkX): destination has high structural influence (PageRank top tier)."
            )

    return extras, graph_reasons


def score_fraud_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Add fraud_score, fraud_reasons, is_flagged, abs_amount; merge rule-based and NetworkX explanations."""
    if df.empty:
        return df

    scored = df.sort_values("timestamp").reset_index(drop=True).copy()
    scored["abs_amount"] = scored["amount"].abs()

    high_amount_threshold = scored["abs_amount"].quantile(0.95)
    if pd.isna(high_amount_threshold) or high_amount_threshold <= 0:
        high_amount_threshold = scored["abs_amount"].max()
    high_amount_threshold = max(float(high_amount_threshold), 1000.0)

    rapid_counts = [1] * len(scored)
    windows: dict[str, deque[pd.Timestamp]] = defaultdict(deque)
    for idx, row in scored.iterrows():
        source = str(row["source_account"])
        ts = row["timestamp"]
        queue = windows[source]
        while queue and (ts - queue[0]).total_seconds() > 10 * 60:
            queue.popleft()
        queue.append(ts)
        rapid_counts[idx] = len(queue)

    scored["rapid_source_tx_10m"] = rapid_counts

    rule_scores: list[int] = []
    rule_reasons: list[list[str]] = []
    for _, row in scored.iterrows():
        score = 0
        parts: list[str] = []
        if row["abs_amount"] >= high_amount_threshold:
            score += 45
            parts.append(f"Rules: high amount (>= {high_amount_threshold:,.2f}).")
        if row["rapid_source_tx_10m"] >= 3:
            score += 25
            parts.append("Rules: multiple transactions from same source within 10 minutes.")
        if str(row["source_account"]) == str(row["destination_account"]):
            score += 35
            parts.append("Rules: source and destination accounts are identical.")
        if row["abs_amount"] >= 1000 and round(row["abs_amount"] % 1000, 2) == 0:
            score += 10
            parts.append("Rules: round amount pattern.")
        if int(row["timestamp"].hour) < 5:
            score += 10
            parts.append("Rules: off-hours transaction.")
        rule_scores.append(min(score, 100))
        rule_reasons.append(parts)

    G = _build_cashflow_multigraph(scored)
    nx_extra, nx_reason_lists = _networkx_signals_for_rows(scored, G)

    final_scores: list[int] = []
    final_text: list[str] = []
    flags: list[bool] = []
    for i in range(len(scored)):
        combined = min(rule_scores[i] + nx_extra[i], 100)
        final_scores.append(combined)
        parts_r = list(rule_reasons[i])
        parts_n = nx_reason_lists[i]
        if not parts_r and not parts_n:
            msg = "No high-risk signals from rules or graph structure."
        else:
            msg = " ".join(parts_r + parts_n)
        final_text.append(msg)
        flags.append(combined >= 50)

    scored["fraud_score"] = final_scores
    scored["fraud_reasons"] = final_text
    scored["is_flagged"] = flags
    return scored
