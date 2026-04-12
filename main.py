"""Streamlit entry: layout, session state, and wiring to ``app`` modules."""

from __future__ import annotations

import os
from datetime import date
from typing import Any

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from app.constants import AUTO_COLUMN_LABEL, MAPPING_CANONICAL_FIELDS
from app.fraud_scoring import score_fraud_signals
from app.fraud_report_openai import generate_fraud_audit_report
from app.fraud_report_payload import build_categorized_fraud_alerts
from app.graph import build_cashflow_graph_payload, filter_scored_df_by_graph_dates
from app.graph_template import render_graph_html
from app.ingestion import (
    normalize_transactions,
    parse_csv_records,
    parse_json_payload,
    parse_json_records,
    peek_csv_fieldnames,
    sample_records,
    suggest_column_mapping,
)

st.set_page_config(page_title="Cashflow Fraud Monitor", layout="wide")


def _bootstrap_openai_api_key() -> None:
    """Copy ``OPENAI_API_KEY`` from Streamlit secrets into ``os.environ`` if not already set."""
    if os.environ.get("OPENAI_API_KEY", "").strip():
        return
    try:
        if "OPENAI_API_KEY" not in st.secrets:
            return
        key = st.secrets["OPENAI_API_KEY"]
    except Exception:
        return
    if key and str(key).strip():
        os.environ["OPENAI_API_KEY"] = str(key).strip()


def append_records(new_records: list[dict[str, Any]], source_name: str) -> None:
    if not new_records:
        st.warning(f"No valid records found from {source_name}.")
        return
    st.session_state.raw_records.extend(new_records)
    st.success(f"Added {len(new_records)} record(s) from {source_name}.")


def explicit_column_mapping_from_session() -> dict[str, str] | None:
    mapping: dict[str, str] = {}
    for canonical, _ in MAPPING_CANONICAL_FIELDS:
        choice = st.session_state.get(f"colmap_select_{canonical}", AUTO_COLUMN_LABEL)
        if choice and choice != AUTO_COLUMN_LABEL:
            mapping[canonical] = choice
    return mapping or None


def render_ingestion_panel() -> None:
    st.subheader("1) Ingest transaction data")
    st.caption("Load data in different formats. Everything gets mapped into one internal transaction schema.")

    left, right = st.columns(2)
    if left.button("Load sample data", use_container_width=True):
        append_records(sample_records(), "sample dataset")
    if right.button("Clear all loaded records", use_container_width=True):
        st.session_state.raw_records = []
        st.success("Cleared all loaded records.")

    st.markdown("### File-based inputs")
    csv_file = st.file_uploader("Upload CSV", type=["csv"], key="csv_upload")

    with st.expander("CSV column mapping", expanded=False):
        st.caption(
            "Map each **internal** field to a column from your CSV (exact header name). "
            'Leave a field on "Auto-detect" to infer it from known aliases (including HKT-style exports). '
            "Leading blank rows before the header row are skipped automatically."
        )
        csv_headers: list[str] = []
        if csv_file is not None:
            try:
                with st.spinner("Reading CSV column headers…"):
                    csv_headers = peek_csv_fieldnames(csv_file.getvalue())
            except Exception as exc:
                st.error(f"Could not read CSV headers: {exc}")

        if csv_headers:
            st.markdown(
                f"**Detected columns** ({len(csv_headers)}): `{', '.join(csv_headers[:12])}`"
                + (" …" if len(csv_headers) > 12 else "")
            )
            if st.button("Fill suggestions from this CSV", key="suggest_csv_mapping"):
                with st.spinner("Applying column suggestions…"):
                    suggestions = suggest_column_mapping(csv_headers)
                    for canonical, _ in MAPPING_CANONICAL_FIELDS:
                        pick = suggestions.get(canonical)
                        st.session_state[f"colmap_select_{canonical}"] = pick if pick else AUTO_COLUMN_LABEL
                st.rerun()
        else:
            st.info("Upload a CSV above to list its columns and fill suggestions.")

        orphans: list[str] = []
        for canonical, _ in MAPPING_CANONICAL_FIELDS:
            v = st.session_state.get(f"colmap_select_{canonical}", AUTO_COLUMN_LABEL)
            if v and v != AUTO_COLUMN_LABEL and v not in csv_headers:
                orphans.append(v)
        option_list = [AUTO_COLUMN_LABEL] + sorted(set(csv_headers) | set(orphans))
        grid = st.columns(2)
        for i, (canonical, label) in enumerate(MAPPING_CANONICAL_FIELDS):
            col = grid[i % 2]
            with col:
                st.selectbox(
                    f"{label} ← file column",
                    options=option_list,
                    key=f"colmap_select_{canonical}",
                )

    if st.button("Add CSV records", key="add_csv"):
        if csv_file is None:
            st.warning("Select a CSV file first.")
        else:
            try:
                with st.spinner("Parsing and loading CSV records…"):
                    records = parse_csv_records(csv_file.getvalue())
                append_records(records, "CSV file")
            except Exception as exc:
                st.error(f"Could not parse CSV: {exc}")

    json_file = st.file_uploader("Upload JSON", type=["json"], key="json_upload")
    if st.button("Add JSON records", key="add_json"):
        if json_file is None:
            st.warning("Select a JSON file first.")
        else:
            try:
                append_records(parse_json_records(json_file.getvalue()), "JSON file")
            except Exception as exc:
                st.error(f"Could not parse JSON: {exc}")

    st.markdown("### Raw JSON input")
    json_text = st.text_area(
        "Paste JSON records",
        value="",
        height=160,
        placeholder='[{"id":"TX-01","date":"2026-04-12T09:15:00Z","from":"A","to":"B","amount":1200}]',
    )
    if st.button("Add pasted JSON records", key="add_json_text"):
        if not json_text.strip():
            st.warning("Paste JSON first.")
        else:
            try:
                append_records(parse_json_payload(json_text), "pasted JSON")
            except Exception as exc:
                st.error(f"Could not parse pasted JSON: {exc}")

    st.markdown("### Manual entry")
    with st.form("manual_entry_form", clear_on_submit=True):
        c1, c2, c3 = st.columns(3)
        tx_id = c1.text_input("Transaction ID", value="")
        ts = c2.text_input("Timestamp (ISO)", value=pd.Timestamp.now(tz="UTC").isoformat())
        amount = c3.number_input("Amount", value=0.0, step=10.0, format="%.2f")

        c4, c5, c6 = st.columns(3)
        source = c4.text_input("Source account", value="")
        destination = c5.text_input("Destination account", value="")
        currency = c6.text_input("Currency", value="USD")

        c7, c8 = st.columns(2)
        channel = c7.text_input("Channel", value="manual")
        description = c8.text_input("Description", value="")

        submitted = st.form_submit_button("Add manual transaction")

    if submitted:
        st.session_state.raw_records.append(
            {
                "transaction_id": tx_id or None,
                "timestamp": ts,
                "source_account": source,
                "destination_account": destination,
                "amount": amount,
                "currency": currency,
                "channel": channel,
                "description": description,
            }
        )
        st.success("Manual transaction added.")


def render_analysis_panel(raw_records: list[dict[str, Any]]) -> None:
    st.subheader("2) Normalize, score, and visualize")
    st.caption(
        "Input records are mapped into a canonical schema. Fraud scoring combines **rule-based** checks "
        "and **NetworkX** graph signals (hubs, bidirectional pairs, PageRank on moderate-sized graphs)."
    )

    column_mapping = explicit_column_mapping_from_session()
    if raw_records:
        with st.spinner("Normalizing transactions (column mapping + validation)…"):
            normalized_records, rejected_records = normalize_transactions(
                raw_records, column_mapping=column_mapping
            )
    else:
        normalized_records, rejected_records = [], []
    if rejected_records:
        st.warning(f"{len(rejected_records)} record(s) were rejected during mapping (usually missing amount).")
        with st.expander("See rejected records"):
            st.json(rejected_records)

    if not normalized_records:
        st.info("No valid transactions available yet.")
        return

    normalized_df = pd.DataFrame(normalized_records)
    scored_df = score_fraud_signals(normalized_df)

    total = len(scored_df)
    flagged = int(scored_df["is_flagged"].sum())
    flow_volume = float(scored_df["abs_amount"].sum())
    m1, m2, m3 = st.columns(3)
    m1.metric("Valid transactions", total)
    m2.metric("Flagged transactions", flagged)
    m3.metric("Total observed flow", f"{flow_volume:,.2f}")

    display_df = scored_df.copy()
    display_df["timestamp"] = display_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S %Z")
    st.markdown("### Canonical internal dataset")
    st.dataframe(
        display_df[
            [
                "transaction_id",
                "timestamp",
                "source_account",
                "destination_account",
                "amount",
                "currency",
                "channel",
                "fraud_score",
                "is_flagged",
                "fraud_reasons",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

    flagged_df = display_df[display_df["is_flagged"]]
    if not flagged_df.empty:
        st.markdown("### Flagged transactions")
        st.dataframe(
            flagged_df[
                [
                    "transaction_id",
                    "timestamp",
                    "source_account",
                    "destination_account",
                    "amount",
                    "fraud_score",
                    "fraud_reasons",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.success("No transactions currently cross the fraud threshold.")

    csv_data = display_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download normalized + scored CSV",
        data=csv_data,
        file_name="normalized_scored_transactions.csv",
        mime="text/csv",
    )

    st.markdown("### CFO audit report (OpenAI)")
    st.caption(
        "Uses **OPENAI_API_KEY** from the environment or `.streamlit/secrets.toml`. "
        "Builds categories from **flagged** transactions, then asks the model for a Markdown report."
    )
    col_rep1, col_rep2 = st.columns([1, 2])
    with col_rep1:
        gen_clicked = st.button("Generate report", type="primary", key="cfo_generate_report")
    with col_rep2:
        if st.button("Clear report", key="cfo_clear_report"):
            st.session_state.pop("cfo_report_md", None)
            st.rerun()

    if gen_clicked:
        alerts = build_categorized_fraud_alerts(scored_df)
        try:
            with st.spinner("Calling OpenAI (gpt-4o)…"):
                report_md = generate_fraud_audit_report(alerts)
            st.session_state["cfo_report_md"] = report_md
        except ValueError as exc:
            st.error(str(exc))
            st.info("Add `OPENAI_API_KEY` to `.streamlit/secrets.toml` or set it in your shell before `streamlit run`.")

    if st.session_state.get("cfo_report_md"):
        st.markdown(st.session_state["cfo_report_md"])

    st.markdown("### 3) D3.js cashflow graph")

    ts_min = scored_df["timestamp"].min()
    ts_max = scored_df["timestamp"].max()
    min_d = ts_min.date() if pd.notna(ts_min) else date.today()
    max_d = ts_max.date() if pd.notna(ts_max) else date.today()

    date_mode_label = st.selectbox(
        "Filter transactions for the graph (UTC calendar day of each transaction)",
        ["All dates", "Between two dates", "On or after", "On or before"],
        key="graph_date_mode",
    )
    mode_key = {
        "All dates": "all",
        "Between two dates": "between",
        "On or after": "on_or_after",
        "On or before": "on_or_before",
    }[date_mode_label]

    dc1, dc2 = st.columns(2)
    with dc1:
        d_from = st.date_input(
            "From date" if date_mode_label != "On or before" else "On or before (UTC day)",
            value=min_d,
            min_value=date(1990, 1, 1),
            max_value=date(2100, 12, 31),
            key="graph_date_from",
        )
    with dc2:
        d_to = st.date_input(
            'To date (inclusive, for "Between" only)',
            value=max_d,
            min_value=date(1990, 1, 1),
            max_value=date(2100, 12, 31),
            key="graph_date_to",
            disabled=date_mode_label != "Between two dates",
        )

    graph_df = filter_scored_df_by_graph_dates(scored_df, mode_key, d_from, d_to)

    graph_node_radius = st.slider(
        "Default graph node radius (pixels)",
        min_value=6,
        max_value=34,
        value=13,
        step=1,
        key="graph_node_radius",
        help="Adjust live in the graph toolbar as well.",
    )

    if graph_df.empty:
        st.warning("No transactions fall in the selected date range. Widen the filter to see the graph.")
    else:
        st.caption(
            f"**{len(graph_df):,}** transactions included in the graph after the date filter "
            "(edges aggregate flows between account pairs in this subset)."
        )
        graph_payload = build_cashflow_graph_payload(graph_df, node_radius=float(graph_node_radius))
        components.html(render_graph_html(graph_payload), height=900, scrolling=False)


def main() -> None:
    _bootstrap_openai_api_key()
    if "raw_records" not in st.session_state:
        st.session_state.raw_records = []
    for canonical, _ in MAPPING_CANONICAL_FIELDS:
        st.session_state.setdefault(f"colmap_select_{canonical}", AUTO_COLUMN_LABEL)

    st.title("Cashflow Fraud Monitor")
    st.caption(
        "Streamlit + D3.js app that ingests transactions from multiple formats, maps them to one schema, "
        "and flags suspicious behavior (rules + NetworkX graph context)."
    )

    with st.expander("Canonical internal format"):
        st.json(
            {
                "transaction_id": "string",
                "timestamp": "datetime (UTC)",
                "source_account": "string",
                "destination_account": "string",
                "amount": "float",
                "currency": "string",
                "channel": "string",
                "description": "string",
                "organization_name": "string | null (used as graph node label when set)",
            }
        )

    render_ingestion_panel()
    st.markdown("---")
    render_analysis_panel(st.session_state.raw_records)


if __name__ == "__main__":
    main()
