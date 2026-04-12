import csv
import io
import json
import re
from collections import defaultdict, deque
from typing import Any

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Cashflow Fraud Monitor", layout="wide")

FIELD_ALIASES = {
    "transaction_id": [
        "transaction_id",
        "txn_id",
        "tx_id",
        "id",
        "reference",
        "ref",
        "transaction_number",
        "transaction_reference",
        "edge_id",
    ],
    "timestamp": ["timestamp", "date", "datetime", "time", "created_at", "txn_date", "payment_date"],
    "source_account": [
        "source",
        "from",
        "from_account",
        "sender",
        "payer",
        "debit_account",
        "source_id",
    ],
    "destination_account": [
        "destination",
        "to",
        "to_account",
        "receiver",
        "beneficiary",
        "credit_account",
        "target_id",
    ],
    "amount": ["amount", "value", "amt", "transaction_amount", "sum", "total_amount", "balance_due"],
    "currency": ["currency", "ccy", "curr", "iso_currency"],
    "channel": [
        "channel",
        "method",
        "payment_method",
        "type",
        "rail",
        "flow_direction",
        "nature_invoice",
        "node_type",
    ],
    "description": [
        "description",
        "memo",
        "note",
        "narration",
        "details",
        "organization_name",
        "supplier_name",
    ],
}

# Order used for mapping UI and session_state widget keys.
MAPPING_CANONICAL_FIELDS: list[tuple[str, str]] = [
    ("transaction_id", "Transaction ID"),
    ("timestamp", "Timestamp / date"),
    ("source_account", "Source account"),
    ("destination_account", "Destination account"),
    ("amount", "Amount"),
    ("currency", "Currency"),
    ("channel", "Channel / type"),
    ("description", "Description"),
]

AUTO_COLUMN_LABEL = "(Auto-detect from column names)"


def normalize_key(key: str) -> str:
    return re.sub(r"[^a-z0-9]", "", key.lower().strip())


def parse_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None

    negative = text.startswith("(") and text.endswith(")")
    if negative:
        text = text[1:-1]

    for token in [",", " ", "$", "€", "£", "¥"]:
        text = text.replace(token, "")

    try:
        parsed = float(text)
    except ValueError:
        return None
    return -parsed if negative else parsed


def parse_timestamp(value: Any) -> pd.Timestamp:
    if value is None or str(value).strip() == "":
        return pd.Timestamp.now(tz="UTC")
    parsed = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(parsed):
        return pd.Timestamp.now(tz="UTC")
    if isinstance(parsed, pd.DatetimeIndex):
        return parsed[0]
    return parsed


def map_record_to_internal(
    record: dict[str, Any],
    index: int,
    column_mapping: dict[str, str] | None = None,
) -> tuple[dict[str, Any] | None, str | None]:
    if not isinstance(record, dict):
        return None, "Record is not an object."

    mapped: dict[str, Any] = {}
    if column_mapping:
        for canonical, source_col in column_mapping.items():
            if not source_col or source_col == AUTO_COLUMN_LABEL:
                continue
            if source_col not in record:
                continue
            value = record.get(source_col)
            if value not in (None, ""):
                mapped[canonical] = value

    col_norm = {str(k): normalize_key(str(k)) for k in record}
    for canonical, aliases in FIELD_ALIASES.items():
        if canonical in mapped:
            continue
        for alias in aliases:
            target = normalize_key(alias)
            if not target:
                continue
            for col_name, col_norm_key in col_norm.items():
                if col_norm_key != target:
                    continue
                value = record.get(col_name)
                if value not in (None, ""):
                    mapped[canonical] = value
                    break
            if canonical in mapped:
                break

    amount = parse_float(mapped.get("amount"))
    if amount is None:
        return None, "Missing or invalid amount."

    source = str(mapped.get("source_account", "")).strip()
    destination = str(mapped.get("destination_account", "")).strip()

    if source and not destination:
        destination = "EXTERNAL"
    elif destination and not source:
        source = "EXTERNAL"
    elif not source and not destination:
        source = "UNKNOWN_SOURCE"
        destination = "UNKNOWN_DESTINATION"

    transaction_id = str(mapped.get("transaction_id") or f"txn-{index:05d}")
    currency = str(mapped.get("currency") or "USD").upper().strip()
    channel = str(mapped.get("channel") or "unknown").strip()
    description = str(mapped.get("description") or "").strip()
    timestamp = parse_timestamp(mapped.get("timestamp"))

    internal = {
        "transaction_id": transaction_id,
        "timestamp": timestamp,
        "source_account": source,
        "destination_account": destination,
        "amount": amount,
        "currency": currency,
        "channel": channel,
        "description": description,
    }
    return internal, None


def normalize_transactions(
    raw_records: list[dict[str, Any]],
    column_mapping: dict[str, str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    normalized: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    for index, record in enumerate(raw_records, start=1):
        internal, error = map_record_to_internal(record, index=index, column_mapping=column_mapping)
        if error:
            rejected.append({"index": index, "error": error, "raw_record": record})
            continue
        normalized.append(internal)

    return normalized, rejected


def detect_csv_header_row_index(lines: list[str], min_nonempty: int = 3) -> int:
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            row = next(csv.reader([line]))
        except csv.Error:
            continue
        nonempty = sum(1 for cell in row if str(cell).strip())
        if nonempty >= min_nonempty:
            return i
    return 0


def csv_text_and_header_slice(csv_bytes: bytes) -> tuple[str, int]:
    text = csv_bytes.decode("utf-8-sig")
    lines = text.splitlines()
    idx = detect_csv_header_row_index(lines)
    remainder = "\n".join(lines[idx:]) if lines else ""
    return remainder, idx


def peek_csv_fieldnames(csv_bytes: bytes) -> list[str]:
    remainder, _ = csv_text_and_header_slice(csv_bytes)
    if not remainder.strip():
        return []
    reader = csv.DictReader(io.StringIO(remainder))
    names = reader.fieldnames or []
    return [str(n) for n in names if n is not None and str(n).strip()]


def suggest_column_mapping(headers: list[str]) -> dict[str, str | None]:
    """Pick a source column per canonical field using normalized alias equality."""
    headers_norm = {h: normalize_key(h) for h in headers}
    suggested: dict[str, str | None] = {}
    for canonical, _ in MAPPING_CANONICAL_FIELDS:
        aliases = FIELD_ALIASES.get(canonical, [])
        chosen: str | None = None
        for alias in aliases:
            target = normalize_key(alias)
            if not target:
                continue
            for header, hn in headers_norm.items():
                if hn == target:
                    chosen = header
                    break
            if chosen:
                break
        suggested[canonical] = chosen
    return suggested


def parse_csv_records(csv_bytes: bytes) -> list[dict[str, Any]]:
    remainder, _ = csv_text_and_header_slice(csv_bytes)
    if not remainder.strip():
        return []
    reader = csv.DictReader(io.StringIO(remainder))
    return [row for row in reader if any(str(v).strip() for v in row.values())]


def parse_json_records(json_bytes: bytes) -> list[dict[str, Any]]:
    return parse_json_payload(json_bytes.decode("utf-8-sig"))


def parse_json_payload(json_text: str) -> list[dict[str, Any]]:
    payload = json.loads(json_text)
    if isinstance(payload, dict):
        payload = payload.get("transactions", [payload])
    if not isinstance(payload, list):
        raise ValueError("JSON must be a list of objects or an object containing a 'transactions' list.")
    return [item for item in payload if isinstance(item, dict)]


def sample_records() -> list[dict[str, Any]]:
    return [
        {
            "txn_id": "TXN-1001",
            "date": "2026-04-10T08:15:00Z",
            "from_account": "Payroll",
            "to_account": "ACC-001",
            "amt": 3500,
            "ccy": "USD",
            "method": "wire",
            "memo": "Monthly salary",
        },
        {
            "id": "TXN-1002",
            "timestamp": "2026-04-10T08:18:00Z",
            "source": "ACC-001",
            "destination": "WALLET-7",
            "amount": 3300,
            "currency": "USD",
            "type": "instant_transfer",
        },
        {
            "reference": "TXN-1003",
            "datetime": "2026-04-10T08:19:00Z",
            "sender": "ACC-001",
            "receiver": "WALLET-8",
            "value": 3200,
            "currency": "USD",
            "details": "split payment",
        },
        {
            "id": "TXN-1004",
            "timestamp": "2026-04-10T02:03:00Z",
            "source": "WALLET-8",
            "destination": "CRYPTO-EXCHANGE",
            "amount": 3000,
            "currency": "USD",
            "channel": "crypto",
        },
        {
            "id": "TXN-1005",
            "timestamp": "2026-04-10T11:20:00Z",
            "source": "ACC-002",
            "destination": "VENDOR-9",
            "amount": 175.25,
            "currency": "USD",
            "description": "Office supplies",
        },
    ]


def score_fraud_signals(df: pd.DataFrame) -> pd.DataFrame:
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

    scores: list[int] = []
    reasons: list[str] = []
    flags: list[bool] = []
    for _, row in scored.iterrows():
        score = 0
        reason_parts: list[str] = []

        if row["abs_amount"] >= high_amount_threshold:
            score += 45
            reason_parts.append(f"High amount (>= {high_amount_threshold:,.2f})")

        if row["rapid_source_tx_10m"] >= 3:
            score += 25
            reason_parts.append("Multiple transactions from same source in 10 minutes")

        if str(row["source_account"]) == str(row["destination_account"]):
            score += 35
            reason_parts.append("Source and destination accounts are identical")

        if row["abs_amount"] >= 1000 and round(row["abs_amount"] % 1000, 2) == 0:
            score += 10
            reason_parts.append("Round amount pattern")

        if int(row["timestamp"].hour) < 5:
            score += 10
            reason_parts.append("Off-hours transaction")

        score = min(score, 100)
        scores.append(score)
        reasons.append("; ".join(reason_parts) if reason_parts else "No high-risk signals.")
        flags.append(score >= 50)

    scored["fraud_score"] = scores
    scored["fraud_reasons"] = reasons
    scored["is_flagged"] = flags
    return scored


def build_cashflow_graph_payload(scored_df: pd.DataFrame) -> dict[str, Any]:
    if scored_df.empty:
        return {"nodes": [], "links": []}

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

    node_ids = sorted(set(grouped["source_account"]).union(set(grouped["destination_account"])))
    nodes = [{"id": str(node), "has_flagged_tx": str(node) in flagged_nodes} for node in node_ids]

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

    return {"nodes": nodes, "links": links}


def build_d3_graph_html(graph_payload: dict[str, Any]) -> str:
    template = """
<div id="cashflow-root"></div>
<style>
#cashflow-root { font-family: Inter, Arial, sans-serif; position: relative; width: 100%; }
#cashflow-root svg { width: 100%; height: 560px; border: 1px solid #E2E8F0; border-radius: 10px; background: #FFFFFF; }
#cashflow-root .tooltip { position: absolute; pointer-events: none; background: rgba(15, 23, 42, 0.95); color: #FFFFFF; padding: 8px 10px; border-radius: 6px; font-size: 12px; opacity: 0; transition: opacity 0.12s ease-in-out; z-index: 99; }
#cashflow-root .legend { margin-bottom: 8px; font-size: 12px; color: #334155; }
#cashflow-root .empty { border: 1px dashed #CBD5E1; border-radius: 8px; padding: 20px; color: #64748B; }
</style>
<div class="legend">Line thickness = total cashflow. Red lines = flows that include flagged transactions.</div>
<script src="https://cdn.jsdelivr.net/npm/d3@7"></script>
<script>
const data = __GRAPH_DATA__;
const root = document.getElementById("cashflow-root");

if (!data.links || data.links.length === 0) {
  root.innerHTML = '<div class="empty">Add transactions that include source and destination accounts to render the graph.</div>';
} else {
  const width = root.clientWidth > 0 ? root.clientWidth : 920;
  const height = 560;

  const svg = d3.select(root)
    .append("svg")
    .attr("viewBox", `0 0 ${width} ${height}`)
    .attr("preserveAspectRatio", "xMidYMid meet");

  const tooltip = d3.select(root).append("div").attr("class", "tooltip");
  const amounts = data.links.map((d) => Number(d.amount) || 0);
  const minAmount = Math.max(1, Math.min(...amounts));
  const maxAmount = Math.max(minAmount + 1, Math.max(...amounts));
  const widthScale = d3.scaleSqrt().domain([minAmount, maxAmount]).range([1.5, 9]);

  const link = svg.append("g")
    .attr("stroke-linecap", "round")
    .selectAll("line")
    .data(data.links)
    .join("line")
    .attr("stroke", (d) => d.is_flagged ? "#DC2626" : "#64748B")
    .attr("stroke-opacity", 0.72)
    .attr("stroke-width", (d) => widthScale(Math.max(1, Number(d.amount) || 0)));

  const node = svg.append("g")
    .selectAll("g")
    .data(data.nodes)
    .join("g")
    .call(
      d3.drag()
        .on("start", dragStarted)
        .on("drag", dragged)
        .on("end", dragEnded)
    );

  node.append("circle")
    .attr("r", 13)
    .attr("fill", (d) => d.has_flagged_tx ? "#F59E0B" : "#2563EB")
    .attr("stroke", "#0F172A")
    .attr("stroke-width", 1.1);

  node.append("text")
    .text((d) => d.id)
    .attr("x", 16)
    .attr("y", 4)
    .attr("font-size", 11)
    .attr("fill", "#0F172A");

  link
    .on("mousemove", (event, d) => {
      const source = typeof d.source === "object" ? d.source.id : d.source;
      const target = typeof d.target === "object" ? d.target.id : d.target;
      tooltip
        .style("opacity", 1)
        .style("left", `${event.offsetX + 14}px`)
        .style("top", `${event.offsetY + 14}px`)
        .html(`<strong>${source} → ${target}</strong><br/>Amount: ${Number(d.amount).toLocaleString()}<br/>Transactions: ${d.tx_count}<br/>Flagged: ${d.flagged_tx}`);
    })
    .on("mouseout", () => tooltip.style("opacity", 0));

  node
    .on("mousemove", (event, d) => {
      tooltip
        .style("opacity", 1)
        .style("left", `${event.offsetX + 14}px`)
        .style("top", `${event.offsetY + 14}px`)
        .html(`<strong>${d.id}</strong><br/>Linked to flagged flow: ${d.has_flagged_tx ? "yes" : "no"}`);
    })
    .on("mouseout", () => tooltip.style("opacity", 0));

  const simulation = d3.forceSimulation(data.nodes)
    .force("link", d3.forceLink(data.links).id((d) => d.id).distance(140).strength(0.18))
    .force("charge", d3.forceManyBody().strength(-520))
    .force("center", d3.forceCenter(width / 2, height / 2))
    .force("collision", d3.forceCollide().radius(26));

  simulation.on("tick", () => {
    link
      .attr("x1", (d) => d.source.x)
      .attr("y1", (d) => d.source.y)
      .attr("x2", (d) => d.target.x)
      .attr("y2", (d) => d.target.y);

    node.attr("transform", (d) => `translate(${d.x}, ${d.y})`);
  });

  function dragStarted(event) {
    if (!event.active) simulation.alphaTarget(0.3).restart();
    event.subject.fx = event.subject.x;
    event.subject.fy = event.subject.y;
  }

  function dragged(event) {
    event.subject.fx = event.x;
    event.subject.fy = event.y;
  }

  function dragEnded(event) {
    if (!event.active) simulation.alphaTarget(0);
    event.subject.fx = null;
    event.subject.fy = null;
  }
}
</script>
"""
    return template.replace("__GRAPH_DATA__", json.dumps(graph_payload))


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
            "Leave a field on “Auto-detect” to infer it from known aliases (including HKT-style exports). "
            "Leading blank rows before the header row are skipped automatically."
        )
        csv_headers: list[str] = []
        if csv_file is not None:
            try:
                csv_headers = peek_csv_fieldnames(csv_file.getvalue())
            except Exception as exc:
                st.error(f"Could not read CSV headers: {exc}")

        if csv_headers:
            st.markdown(f"**Detected columns** ({len(csv_headers)}): `{', '.join(csv_headers[:12])}`" + (" …" if len(csv_headers) > 12 else ""))
            if st.button("Fill suggestions from this CSV", key="suggest_csv_mapping"):
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
                append_records(parse_csv_records(csv_file.getvalue()), "CSV file")
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
    st.caption("Input records are mapped into a canonical schema before fraud scoring.")

    column_mapping = explicit_column_mapping_from_session()
    normalized_records, rejected_records = normalize_transactions(raw_records, column_mapping=column_mapping)
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

    st.markdown("### 3) D3.js cashflow graph")
    graph_payload = build_cashflow_graph_payload(scored_df)
    components.html(build_d3_graph_html(graph_payload), height=640, scrolling=False)


def main() -> None:
    if "raw_records" not in st.session_state:
        st.session_state.raw_records = []
    for canonical, _ in MAPPING_CANONICAL_FIELDS:
        st.session_state.setdefault(f"colmap_select_{canonical}", AUTO_COLUMN_LABEL)

    st.title("Cashflow Fraud Monitor")
    st.caption(
        "Streamlit + D3.js app that ingests transactions from multiple formats, maps them to one schema, and flags suspicious behavior."
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
            }
        )

    render_ingestion_panel()
    st.markdown("---")
    render_analysis_panel(st.session_state.raw_records)


if __name__ == "__main__":
    main()
