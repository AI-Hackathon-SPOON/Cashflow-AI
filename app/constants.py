"""Shared field definitions for ingestion and column mapping UI."""

FIELD_ALIASES: dict[str, list[str]] = {
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
        "supplier_name",
    ],
    "organization_name": ["organization_name", "org_name", "company_name", "entity_name"],
}

MAPPING_CANONICAL_FIELDS: list[tuple[str, str]] = [
    ("transaction_id", "Transaction ID"),
    ("timestamp", "Timestamp / date"),
    ("source_account", "Source account"),
    ("destination_account", "Destination account"),
    ("amount", "Amount"),
    ("currency", "Currency"),
    ("channel", "Channel / type"),
    ("description", "Description"),
    ("organization_name", "Organization name (graph node label)"),
]

AUTO_COLUMN_LABEL = "(Auto-detect from column names)"
