"""Streamlit entry: layout, session state, and wiring to ``app`` modules."""

from __future__ import annotations

import json
import os
from datetime import date
from typing import Any

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from app.constants import AUTO_COLUMN_LABEL, MAPPING_CANONICAL_FIELDS
from app.fraud_scoring import score_fraud_signals
from app.fraud_report_openai import generate_flagged_rag_explanations, generate_fraud_audit_report
from app.fraud_report_payload import (
    attach_kb_by_transaction_id,
    build_categorized_fraud_alerts,
    build_flagged_rag_items,
    build_vector_kb_document,
)
from app.rag_chroma import (
    build_kb_map_from_chroma,
    chroma_document_count,
    chroma_persist_path,
    global_kb_from_chroma_for_flagged,
    list_stored_documents,
    parse_and_upsert_learned_json,
    upsert_flagged_snapshots,
)
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
        "Builds categories from **flagged** transactions. Optional **global KB** (vector-retrieved learned patterns "
        "for this dataset) is merged into the same prompt to strengthen the annex."
    )
    # Chroma → CFO must not assign to widget-bound keys after those widgets run (same script run).
    # Pending values are applied here, before ``cfo_kb_global_json`` / ``cfo_use_kb_global`` widgets mount.
    if "_pending_cfo_kb_global_json" in st.session_state:
        st.session_state["cfo_kb_global_json"] = st.session_state.pop("_pending_cfo_kb_global_json")
    if "_pending_cfo_use_kb_global" in st.session_state:
        st.session_state["cfo_use_kb_global"] = bool(st.session_state.pop("_pending_cfo_use_kb_global"))
    st.session_state.setdefault("cfo_kb_global_json", "[]")
    st.checkbox(
        "Enhance CFO report with global KB snippets (RAG)",
        key="cfo_use_kb_global",
        help="Paste a JSON array of KB objects (e.g. top chunks from your vector DB for this file).",
    )
    if st.session_state.get("cfo_use_kb_global"):
        st.text_area(
            "Global KB — JSON array of objects (max 15 sent to the model)",
            key="cfo_kb_global_json",
            height=140,
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
        kb_global: list[dict[str, Any]] | None = None
        if st.session_state.get("cfo_use_kb_global"):
            try:
                parsed_g = json.loads(st.session_state.get("cfo_kb_global_json") or "[]")
            except json.JSONDecodeError as exc:
                st.warning(f"Global KB JSON invalid ({exc}); report generated without KB.")
            else:
                if isinstance(parsed_g, list) and all(isinstance(x, dict) for x in parsed_g):
                    kb_global = parsed_g
                else:
                    st.warning("Global KB must be a JSON array of objects; report generated without KB.")
        try:
            with st.spinner("Calling OpenAI (gpt-4o)…"):
                report_md = generate_fraud_audit_report(alerts, kb_global_context=kb_global)
            st.session_state["cfo_report_md"] = report_md
        except ValueError as exc:
            st.error(str(exc))
            st.info("Add `OPENAI_API_KEY` to `.streamlit/secrets.toml` or set it in your shell before `streamlit run`.")

    if st.session_state.get("cfo_report_md"):
        st.markdown(st.session_state["cfo_report_md"])

    with st.expander("Local vector KB (Chroma)", expanded=False):
        st.caption(
            f"Embeddings are stored on disk at **{chroma_persist_path()}** (gitignored). "
            "Uses OpenAI **text-embedding-3-small** — same `OPENAI_API_KEY` as the reports. "
            "Upsert **learned patterns** (`build_vector_kb_document` JSON) and/or **index every flagged row** "
            "so similarity search can fill the RAG fields."
        )
        try:
            n_docs = chroma_document_count()
        except ImportError:
            st.error("The `chromadb` package is not installed. Stop Streamlit, run `uv sync`, then restart.")
            n_docs = -1
        except ValueError as exc:
            st.warning(str(exc))
            n_docs = -1
        except Exception as exc:  # noqa: BLE001
            st.warning(f"Chroma could not open: {exc}")
            n_docs = -1
        else:
            st.metric("Documents in Chroma", n_docs if n_docs >= 0 else "—")

        pv1, pv2, pv3 = st.columns([2, 1, 1])
        with pv1:
            chroma_preview_limit = st.number_input(
                "Preview row limit",
                min_value=10,
                max_value=500,
                value=100,
                step=10,
                key="chroma_preview_limit",
            )
        with pv2:
            chroma_preview_offset = st.number_input(
                "Offset",
                min_value=0,
                max_value=100_000,
                value=0,
                step=50,
                key="chroma_preview_offset",
            )
        with pv3:
            preview_btn = st.button("Preview Chroma contents", key="chroma_preview_btn")

        if preview_btn:
            try:
                rows, total = list_stored_documents(
                    limit=int(chroma_preview_limit),
                    offset=int(chroma_preview_offset),
                )
                st.caption(f"Showing **{len(rows)}** row(s); collection total **{total}**.")
                if rows:
                    view = pd.DataFrame(rows)
                    view["document"] = view["document"].astype(str).str.slice(0, 400)
                    st.dataframe(view, use_container_width=True, hide_index=True)
                else:
                    st.info("Collection is empty — upsert patterns or index flagged rows first.")
            except ValueError as exc:
                st.error(str(exc))
            except Exception as exc:  # noqa: BLE001
                st.error(str(exc))

        if "chroma_patterns_json" not in st.session_state:
            st.session_state["chroma_patterns_json"] = json.dumps(
                [
                    build_vector_kb_document(
                        "Payroll hub with many outbound wires; resembled mule fan-out but was consolidated "
                        "treasury run by finance — closed as benign after KYC refresh.",
                        outcome="False Positive",
                        case_id="KB-DEMO-1",
                        pattern_type="velocity_hub",
                    ),
                    build_vector_kb_document(
                        "Bidirectional flows between corporate card and crypto off-ramps within 24h; SAR filed.",
                        outcome="Confirmed Fraud",
                        case_id="KB-DEMO-2",
                        pattern_type="round_trip_crypto",
                    ),
                ],
                ensure_ascii=False,
                indent=2,
            )

        st.text_area(
            "Learned patterns — JSON array (`text_for_embedding` + `metadata`)",
            key="chroma_patterns_json",
            height=160,
        )
        cc1, cc2, cc3, cc4 = st.columns(4)
        with cc1:
            upsert_pat = st.button("Upsert patterns", key="chroma_btn_upsert_pat")
        with cc2:
            idx_flag = st.button("Index all flagged rows", key="chroma_btn_idx_flag")
        with cc3:
            chroma_k = st.number_input("k per flagged txn", min_value=1, max_value=25, value=5, key="chroma_k")
        with cc4:
            chroma_gk = st.number_input("k for global KB", min_value=1, max_value=30, value=12, key="chroma_gk")

        push_map = st.button("Chroma → KB map (fills RAG textarea)", key="chroma_push_map")
        push_glob = st.button("Chroma → global CFO KB", key="chroma_push_global")

        if upsert_pat:
            n_up, err = parse_and_upsert_learned_json(st.session_state.get("chroma_patterns_json") or "[]")
            if err:
                st.error(f"Invalid JSON: {err}")
            elif n_up == 0:
                st.warning("Nothing to upsert (empty `text_for_embedding` or not an array of objects).")
            else:
                st.success(f"Upserted {n_up} pattern document(s) into Chroma.")

        if idx_flag:
            items_all = build_flagged_rag_items(scored_df)
            try:
                n_idx = upsert_flagged_snapshots(items_all)
                st.success(f"Indexed {n_idx} flagged transaction snapshot(s).")
            except ValueError as exc:
                st.error(str(exc))
            except Exception as exc:  # noqa: BLE001
                st.error(str(exc))

        if push_map:
            try:
                kb_m = build_kb_map_from_chroma(
                    scored_df,
                    k_per_txn=int(chroma_k),
                    max_flagged=200,
                )
                st.session_state["rag_kb_map_json"] = json.dumps(kb_m, ensure_ascii=False, indent=2)
                st.success("KB map updated. Scroll to **RAG: explain each flagged transaction** if needed.")
            except ValueError as exc:
                st.error(str(exc))
            except Exception as exc:  # noqa: BLE001
                st.error(str(exc))

        if push_glob:
            try:
                gkb = global_kb_from_chroma_for_flagged(scored_df, k=int(chroma_gk))
                st.session_state["_pending_cfo_kb_global_json"] = json.dumps(gkb, ensure_ascii=False, indent=2)
                st.session_state["_pending_cfo_use_kb_global"] = True
                st.success("Global KB JSON updated; CFO KB checkbox will be enabled after refresh.")
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))
            except Exception as exc:  # noqa: BLE001
                st.error(str(exc))

    st.session_state.setdefault("rag_kb_map_json", "{}")
    with st.expander("RAG: explain each flagged transaction (vector KB)", expanded=False):
        st.caption(
            "Paste a ``transaction_id → [hits]`` map, **or** use **Local vector KB (Chroma)** above to auto-fill "
            "from your on-disk store. You can also manage patterns with ``build_vector_kb_document`` in code / n8n."
        )
        st.text_area(
            "KB by transaction_id — JSON object: each key is a transaction id, value is an array of KB objects",
            key="rag_kb_map_json",
            height=180,
        )
        r1, r2, r3 = st.columns(3)
        with r1:
            max_rag = st.number_input("Max flagged rows", min_value=1, max_value=500, value=35, step=1, key="rag_max_rows")
        with r2:
            rag_chunk = st.number_input("Rows per API batch", min_value=1, max_value=25, value=10, step=1, key="rag_chunk")
        with r3:
            rag_kb_cap = st.number_input("Max KB hits per row", min_value=0, max_value=20, value=5, step=1, key="rag_kb_cap")

        col_rg1, col_rg2 = st.columns([1, 2])
        with col_rg1:
            gen_rag = st.button("Generate RAG explanations", type="secondary", key="rag_gen")
        with col_rg2:
            if st.button("Clear RAG output", key="rag_clear"):
                st.session_state.pop("rag_explain_md", None)
                st.rerun()

        if gen_rag:
            try:
                kb_map_raw = json.loads(st.session_state.get("rag_kb_map_json") or "{}")
            except json.JSONDecodeError as exc:
                st.error(f"KB map JSON is invalid: {exc}")
            else:
                if not isinstance(kb_map_raw, dict):
                    st.error("KB map must be a JSON object (transaction_id → array).")
                else:
                    kb_map: dict[str, list[dict[str, Any]]] = {}
                    bad_val = False
                    for k, v in kb_map_raw.items():
                        kid = str(k)
                        if not isinstance(v, list):
                            bad_val = True
                            break
                        if not all(isinstance(x, dict) for x in v):
                            bad_val = True
                            break
                        kb_map[kid] = list(v)
                    if bad_val:
                        st.error("Each map value must be a JSON array of objects.")
                    else:
                        items = build_flagged_rag_items(scored_df)[: int(max_rag)]
                        if not items:
                            st.info("No flagged transactions in the current data — nothing to explain.")
                        else:
                            merged = attach_kb_by_transaction_id(items, kb_map)
                            try:
                                with st.spinner("Calling OpenAI for per-transaction RAG explanations…"):
                                    rag_md = generate_flagged_rag_explanations(
                                        merged,
                                        chunk_size=int(rag_chunk),
                                        max_kb_per_item=int(rag_kb_cap),
                                    )
                                st.session_state["rag_explain_md"] = rag_md
                            except ValueError as exc:
                                st.error(str(exc))
                                st.info(
                                    "Add `OPENAI_API_KEY` to `.streamlit/secrets.toml` or set it in your environment."
                                )

        if st.session_state.get("rag_explain_md"):
            st.markdown(st.session_state["rag_explain_md"])

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
