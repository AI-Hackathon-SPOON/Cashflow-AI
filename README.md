## Cashflow Fraud Monitor (Streamlit + D3.js)
This project ingests transactions from multiple input types, maps them into one canonical internal schema, scores fraud risk with heuristic rules, and renders an interactive D3.js cashflow graph.

## Features
- `uv`-managed Python project
- Streamlit dashboard for ingestion + analysis
- Multiple transaction input paths:
  - CSV upload
  - JSON upload
  - Pasted JSON payload
  - Manual transaction entry
  - Sample dataset loader
- Canonical schema mapping from heterogeneous field names
- Fraud scoring (amount outliers, rapid bursts, off-hours, self-transfers, round amounts)
- Interactive D3 force graph of cashflow between accounts

## Canonical internal schema
```json
{
  "transaction_id": "string",
  "timestamp": "datetime (UTC)",
  "source_account": "string",
  "destination_account": "string",
  "amount": "float",
  "currency": "string",
  "channel": "string",
  "description": "string"
}
```

## Run on a new machine
You only need **[uv](https://docs.astral.sh/uv/getting-started/installation/)** installed (it manages Python for this project).

1. Clone the repository and open the project folder.
2. Install dependencies (creates `.venv` and uses `uv.lock` when present):
   - `uv sync`
3. Start the app:
   - `uv run streamlit run main.py`

Python **3.11+** is required (`requires-python` in `pyproject.toml`).

## Notes
- Different input field names are normalized through alias mapping (for example: `txn_id`, `id`, `reference` all map to `transaction_id`).
- Records missing a valid amount are rejected and shown in the UI for troubleshooting.
- Fraud labels are heuristic and intended as a baseline that you can tune for your own risk policy.