"""CSV/JSON parsing, column mapping, and normalization to internal records."""

from __future__ import annotations

import csv
import copy
import io
import json
import re
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd

from app.constants import AUTO_COLUMN_LABEL, FIELD_ALIASES, MAPPING_CANONICAL_FIELDS


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
    organization_name = str(mapped.get("organization_name") or "").strip()
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
        "organization_name": organization_name or None,
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


_SAMPLE_DEMO_CHAIN: list[dict[str, Any]] = [
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

_ID_KEYS = frozenset({"txn_id", "id", "reference"})
_TIME_KEYS = frozenset({"date", "timestamp", "datetime"})


def _parse_iso_utc(s: str) -> datetime:
    s2 = s.strip().replace("Z", "+00:00")
    dt = datetime.fromisoformat(s2)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _format_iso_utc_z(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _clone_demo_instance(templates: list[dict[str, Any]], instance_idx: int) -> list[dict[str, Any]]:
    """One full pass of the 5-row demo; instance 0 keeps original ids/times, later instances get new ids + offsets."""
    out: list[dict[str, Any]] = []
    hours_shift = instance_idx * 30
    for rec in templates:
        row = copy.deepcopy(rec)
        if instance_idx > 0:
            for k in _ID_KEYS:
                if k in row and isinstance(row[k], str) and row[k].strip():
                    row[k] = f"I{instance_idx}-{row[k]}"
            for k in _TIME_KEYS:
                if k in row and isinstance(row[k], str) and "T" in row[k]:
                    try:
                        row[k] = _format_iso_utc_z(_parse_iso_utc(row[k]) + timedelta(hours=hours_shift))
                    except (ValueError, TypeError):
                        pass
        out.append(row)
    return out


def _padding_transactions(start_seq: int, count: int) -> list[dict[str, Any]]:
    """Low-signal rows to reach a target length (mixed keys like the demo)."""
    rows: list[dict[str, Any]] = []
    base = datetime(2026, 4, 12, 9, 0, tzinfo=timezone.utc)
    for j in range(count):
        seq = start_seq + j + 1
        ts = _format_iso_utc_z(base + timedelta(hours=j * 2))
        if j % 3 == 0:
            rows.append(
                {
                    "txn_id": f"TXN-PAD-{seq}",
                    "date": ts,
                    "from_account": f"ACC-BASE-{(j % 4) + 1}",
                    "to_account": f"VENDOR-{(j % 6) + 20}",
                    "amt": float(45 + (j % 15) * 11),
                    "ccy": "USD",
                    "method": "ach",
                    "memo": "Routine vendor payment",
                }
            )
        elif j % 3 == 1:
            rows.append(
                {
                    "id": f"TXN-PAD-{seq}",
                    "timestamp": ts,
                    "source": f"ACC-BASE-{(j % 4) + 1}",
                    "destination": f"VENDOR-{(j % 6) + 20}",
                    "amount": float(120 + (j % 9) * 5),
                    "currency": "USD",
                    "type": "wire",
                }
            )
        else:
            rows.append(
                {
                    "reference": f"TXN-PAD-{seq}",
                    "datetime": ts,
                    "sender": f"ACC-BASE-{(j % 4) + 1}",
                    "receiver": f"VENDOR-{(j % 6) + 20}",
                    "value": float(88.5 + j),
                    "currency": "USD",
                    "details": "Recurring fee",
                }
            )
    return rows


def sample_records(*, total_count: int = 5, pattern_instances: int = 1) -> list[dict[str, Any]]:
    """
    Built-in synthetic transactions for demos.

    :param total_count: Target number of rows in the returned list (capped in the UI).
    :param pattern_instances: How many times the **5-transaction** fraud-demo chain is concatenated
        (distinct ids/times per instance). If ``total_count`` is larger, mundane **padding** rows are appended.
        If smaller, the list is truncated to ``total_count``.
    """
    templates = _SAMPLE_DEMO_CHAIN
    pi = max(1, int(pattern_instances))
    tn = max(1, int(total_count))
    built: list[dict[str, Any]] = []
    for i in range(pi):
        built.extend(_clone_demo_instance(templates, i))
    if len(built) > tn:
        return built[:tn]
    if len(built) < tn:
        built.extend(_padding_transactions(len(built), tn - len(built)))
    return built
