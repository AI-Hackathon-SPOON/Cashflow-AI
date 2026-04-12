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

_STYLE = """
<style>
    /* Compact metrics */
    div[data-testid="stMetric"] { padding: 0.3rem 0; }
    div[data-testid="stMetric"] label { font-size: 0.75rem; }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] { font-size: 1.15rem; }
    /* Step badge */
    .step-badge {
        display: inline-block; width: 1.7rem; height: 1.7rem; line-height: 1.7rem;
        border-radius: 50%; text-align: center; font-weight: 700; font-size: 0.85rem;
        margin-right: 0.45rem; vertical-align: middle;
    }
    .step-active  { background: #4a8cff; color: #fff; }
    .step-done    { background: #2ecc71; color: #fff; }
    .step-pending { background: #ddd;    color: #999; }
    .step-title   { font-size: 1.05rem; font-weight: 600; vertical-align: middle; }
</style>
"""

STEP_IMPORT = 1
STEP_MAP = 2
STEP_ANALYZE = 3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bootstrap_openai_api_key() -> None:
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


def _append_records(new_records: list[dict[str, Any]], source_name: str) -> None:
    if not new_records:
        st.toast(f"No valid records from {source_name}.", icon="⚠️")
        return
    st.session_state.raw_records.extend(new_records)
    st.toast(f"Added {len(new_records)} record(s) from {source_name}.", icon="✅")


def _explicit_column_mapping() -> dict[str, str] | None:
    mapping: dict[str, str] = {}
    for canonical, _ in MAPPING_CANONICAL_FIELDS:
        choice = st.session_state.get(f"colmap_select_{canonical}", AUTO_COLUMN_LABEL)
        if choice and choice != AUTO_COLUMN_LABEL:
            mapping[canonical] = choice
    return mapping or None


def _current_step() -> int:
    if not st.session_state.raw_records:
        return STEP_IMPORT
    if not st.session_state.get("mapping_confirmed"):
        return STEP_MAP
    return STEP_ANALYZE


def _step_badge(step_num: int, label: str, current: int) -> str:
    if step_num < current:
        cls = "step-done"
    elif step_num == current:
        cls = "step-active"
    else:
        cls = "step-pending"
    return f'<span class="step-badge {cls}">{step_num}</span><span class="step-title">{label}</span>'


# ---------------------------------------------------------------------------
# Step 1 – Import Data
# ---------------------------------------------------------------------------

def _render_step_import() -> None:
    source = st.segmented_control(
        "Choose a data source",
        options=["📂 Sample Data", "📄 CSV File", "📋 JSON", "✏️ Manual Entry"],
        default="📂 Sample Data",
        key="import_source",
        label_visibility="collapsed",
    )

    if source == "📂 Sample Data":
        st.info("Load the built-in demo dataset (synthetic transactions with some fraud signals).")
        if st.button("Load sample data", type="primary", use_container_width=False):
            _append_records(sample_records(), "sample dataset")
            st.rerun()

    elif source == "📄 CSV File":
        csv_file = st.file_uploader("Upload a CSV file", type=["csv"], key="csv_upload")
        if csv_file is not None:
            try:
                headers = peek_csv_fieldnames(csv_file.getvalue())
                st.caption(f"Detected **{len(headers)}** columns: `{', '.join(headers[:10])}`"
                           + (" …" if len(headers) > 10 else ""))
            except Exception as exc:
                st.error(f"Cannot read CSV headers: {exc}")
                headers = []
            if headers:
                st.session_state["_csv_headers"] = headers
        if st.button("Import CSV", key="add_csv", type="primary"):
            if csv_file is None:
                st.warning("Select a CSV file first.")
            else:
                try:
                    records = parse_csv_records(csv_file.getvalue())
                    _append_records(records, "CSV file")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Could not parse CSV: {exc}")

    elif source == "📋 JSON":
        tab_file, tab_paste = st.tabs(["Upload file", "Paste raw JSON"])
        with tab_file:
            json_file = st.file_uploader("Upload a JSON file", type=["json"], key="json_upload")
            if st.button("Import JSON file", key="add_json", type="primary"):
                if json_file is None:
                    st.warning("Select a JSON file first.")
                else:
                    try:
                        _append_records(parse_json_records(json_file.getvalue()), "JSON file")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Could not parse JSON: {exc}")
        with tab_paste:
            json_text = st.text_area(
                "Paste JSON records",
                height=150,
                key="json_paste",
                placeholder='[{"id":"TX-01","date":"2026-04-12","from":"A","to":"B","amount":1200}]',
            )
            if st.button("Import pasted JSON", key="add_json_text", type="primary"):
                if not json_text.strip():
                    st.warning("Paste JSON first.")
                else:
                    try:
                        _append_records(parse_json_payload(json_text), "pasted JSON")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Could not parse pasted JSON: {exc}")

    elif source == "✏️ Manual Entry":
        c1, c2 = st.columns(2)
        tx_id = c1.text_input("Transaction ID")
        ts = c2.text_input("Timestamp (ISO)", value=pd.Timestamp.now(tz="UTC").isoformat())
        c3, c4 = st.columns(2)
        source_acc = c3.text_input("Source account")
        dest_acc = c4.text_input("Destination account")
        c5, c6, c7 = st.columns(3)
        amount = c5.number_input("Amount", value=0.0, step=10.0, format="%.2f")
        currency = c6.text_input("Currency", value="USD")
        channel = c7.text_input("Channel", value="manual")
        description = st.text_input("Description")
        if st.button("Add transaction", key="man_add", type="primary"):
            st.session_state.raw_records.append({
                "transaction_id": tx_id or None,
                "timestamp": ts,
                "source_account": source_acc,
                "destination_account": dest_acc,
                "amount": amount,
                "currency": currency,
                "channel": channel,
                "description": description,
            })
            st.toast("Manual transaction added.", icon="✅")
            st.rerun()


# ---------------------------------------------------------------------------
# Step 2 – Column Mapping
# ---------------------------------------------------------------------------

def _render_step_map() -> None:
    raw = st.session_state.raw_records
    st.caption(f"**{len(raw)}** raw record(s) loaded. Review the column mapping before processing.")

    csv_headers: list[str] = st.session_state.get("_csv_headers", [])

    if csv_headers:
        col_a, col_b = st.columns([1, 3])
        with col_a:
            if st.button("Auto-suggest from CSV headers", key="suggest_csv"):
                suggestions = suggest_column_mapping(csv_headers)
                for canonical, _ in MAPPING_CANONICAL_FIELDS:
                    pick = suggestions.get(canonical)
                    st.session_state[f"colmap_select_{canonical}"] = pick if pick else AUTO_COLUMN_LABEL
                st.rerun()

        orphans: list[str] = []
        for canonical, _ in MAPPING_CANONICAL_FIELDS:
            v = st.session_state.get(f"colmap_select_{canonical}", AUTO_COLUMN_LABEL)
            if v and v != AUTO_COLUMN_LABEL and v not in csv_headers:
                orphans.append(v)
        option_list = [AUTO_COLUMN_LABEL] + sorted(set(csv_headers) | set(orphans))

        grid = st.columns(3)
        for i, (canonical, label) in enumerate(MAPPING_CANONICAL_FIELDS):
            with grid[i % 3]:
                st.selectbox(f"{label}", options=option_list, key=f"colmap_select_{canonical}")
    else:
        st.info(
            "No CSV column headers detected — auto-detection will be used for all fields. "
            "You can proceed directly."
        )

    st.divider()

    with st.expander("Preview raw records (first 5)"):
        st.json(raw[:5])

    bcol1, bcol2, _ = st.columns([1, 1, 3])
    with bcol1:
        if st.button("✅ Confirm & Process", type="primary", use_container_width=True):
            st.session_state["mapping_confirmed"] = True
            st.rerun()
    with bcol2:
        if st.button("🗑️ Clear data & restart", use_container_width=True):
            st.session_state.raw_records = []
            st.session_state.pop("mapping_confirmed", None)
            st.session_state.pop("_csv_headers", None)
            st.session_state.pop("cfo_report_md", None)
            st.rerun()


# ---------------------------------------------------------------------------
# Step 3 – Analyze & Visualize
# ---------------------------------------------------------------------------

def _render_step_analyze(raw_records: list[dict[str, Any]]) -> None:
    column_mapping = _explicit_column_mapping()
    with st.spinner("Normalizing & scoring…"):
        normalized_records, rejected_records = normalize_transactions(raw_records, column_mapping=column_mapping)

    if rejected_records:
        st.toast(f"{len(rejected_records)} record(s) rejected during mapping.", icon="⚠️")

    if not normalized_records:
        st.error("All records were rejected. Go back and check your mapping or data format.")
        if st.button("← Back to mapping"):
            st.session_state.pop("mapping_confirmed", None)
            st.rerun()
        return

    normalized_df = pd.DataFrame(normalized_records)
    scored_df = score_fraud_signals(normalized_df)

    # --- Top bar: metrics + actions ---
    mc1, mc2, mc3, mc4, mc5 = st.columns([1, 1, 1, 1.2, 1.2], vertical_alignment="bottom")
    mc1.metric("Transactions", len(scored_df))
    mc2.metric("Flagged", int(scored_df["is_flagged"].sum()))
    mc3.metric("Total flow", f"{float(scored_df['abs_amount'].sum()):,.2f}")
    with mc4:
        if st.button("🔄 Re-import data", use_container_width=True):
            st.session_state.pop("mapping_confirmed", None)
            st.rerun()
    with mc5:
        if st.button("🗑️ Clear everything", use_container_width=True):
            for k in ["raw_records", "mapping_confirmed", "_csv_headers", "cfo_report_md"]:
                st.session_state.pop(k, None)
            st.session_state.raw_records = []
            st.rerun()

    # --- Tabs ---
    tab_graph, tab_table, tab_flagged, tab_report = st.tabs(
        ["🔗 Graph", "📋 All Transactions", "🚩 Flagged", "📝 Audit Report"]
    )

    display_df = scored_df.copy()
    display_df["timestamp"] = display_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S %Z")

    # Graph ---
    with tab_graph:
        fc1, fc2, fc3, fc4 = st.columns([1.5, 1, 1, 0.8])
        with fc1:
            date_mode_label = st.selectbox(
                "Date filter",
                ["All dates", "Between two dates", "On or after", "On or before"],
                key="graph_date_mode",
            )
        mode_key = {
            "All dates": "all",
            "Between two dates": "between",
            "On or after": "on_or_after",
            "On or before": "on_or_before",
        }[date_mode_label]

        ts_min, ts_max = scored_df["timestamp"].min(), scored_df["timestamp"].max()
        min_d = ts_min.date() if pd.notna(ts_min) else date.today()
        max_d = ts_max.date() if pd.notna(ts_max) else date.today()

        with fc2:
            d_from = st.date_input("From", value=min_d, key="graph_date_from")
        with fc3:
            d_to = st.date_input(
                "To", value=max_d, key="graph_date_to",
                disabled=date_mode_label != "Between two dates",
            )
        with fc4:
            graph_node_radius = st.slider("Node size", 6, 34, 13, key="graph_node_radius")

        graph_df = filter_scored_df_by_graph_dates(scored_df, mode_key, d_from, d_to)
        if graph_df.empty:
            st.warning("No transactions match this date range.")
        else:
            st.caption(f"**{len(graph_df):,}** transactions in graph")
            payload = build_cashflow_graph_payload(graph_df, node_radius=float(graph_node_radius))
            components.html(render_graph_html(payload), height=650, scrolling=False)

    # All transactions ---
    with tab_table:
        st.dataframe(
            display_df[[
                "transaction_id", "timestamp", "source_account", "destination_account",
                "amount", "currency", "channel", "fraud_score", "is_flagged", "fraud_reasons",
            ]],
            use_container_width=True, hide_index=True, height=520,
        )
        csv_data = display_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇ Download CSV", data=csv_data,
                           file_name="normalized_scored_transactions.csv", mime="text/csv")

    # Flagged ---
    with tab_flagged:
        flagged_df = display_df[display_df["is_flagged"]]
        if not flagged_df.empty:
            st.dataframe(
                flagged_df[[
                    "transaction_id", "timestamp", "source_account", "destination_account",
                    "amount", "fraud_score", "fraud_reasons",
                ]],
                use_container_width=True, hide_index=True, height=520,
            )
        else:
            st.success("No transactions currently exceed the fraud threshold.")

    # Audit report ---
    with tab_report:
        r1, r2, _ = st.columns([1, 1, 4])
        with r1:
            gen_clicked = st.button("Generate report", type="primary", key="cfo_generate_report")
        with r2:
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
                st.info("Add `OPENAI_API_KEY` to `.streamlit/secrets.toml` or set it in your shell.")

        if st.session_state.get("cfo_report_md"):
            st.markdown(st.session_state["cfo_report_md"])
        else:
            st.caption("Click **Generate report** to produce a CFO audit summary via OpenAI.")


# ---------------------------------------------------------------------------
# Welcome placeholder
# ---------------------------------------------------------------------------

def _render_welcome() -> None:
    st.markdown(
        """
        <div style="
            display: flex; flex-direction: column; align-items: center; justify-content: center;
            min-height: 50vh; text-align: center; color: #888; user-select: none;
        ">
            <div style="font-size: 4rem; margin-bottom: 0.5rem;">📊</div>
            <h2 style="margin: 0 0 0.5rem 0; color: #555;">Welcome to Cashflow Fraud Monitor</h2>
            <p style="max-width: 480px; line-height: 1.6; font-size: 0.95rem;">
                Choose a data source below to begin.<br>
                You can load the <b>sample dataset</b>, upload a <b>CSV</b> or <b>JSON</b> file,
                or enter transactions <b>manually</b>.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    _bootstrap_openai_api_key()

    if "raw_records" not in st.session_state:
        st.session_state.raw_records = []
    for canonical, _ in MAPPING_CANONICAL_FIELDS:
        st.session_state.setdefault(f"colmap_select_{canonical}", AUTO_COLUMN_LABEL)

    st.markdown(_STYLE, unsafe_allow_html=True)

    step = _current_step()

    # --- Header row: title + step indicators ---
    hdr1, hdr2 = st.columns([2, 4], vertical_alignment="bottom")
    with hdr1:
        st.markdown("### 📊 Cashflow Fraud Monitor")
    with hdr2:
        badges = "  &nbsp;&nbsp;→&nbsp;&nbsp;  ".join([
            _step_badge(STEP_IMPORT, "Import Data", step),
            _step_badge(STEP_MAP, "Map & Confirm", step),
            _step_badge(STEP_ANALYZE, "Analyze & Visualize", step),
        ])
        st.markdown(badges, unsafe_allow_html=True)

    st.divider()

    # --- Step content ---
    if step == STEP_IMPORT:
        _render_welcome()
        _render_step_import()

    elif step == STEP_MAP:
        _render_step_map()

    elif step == STEP_ANALYZE:
        _render_step_analyze(st.session_state.raw_records)


if __name__ == "__main__":
    main()
