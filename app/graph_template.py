"""Load the D3 cashflow graph markup from ``static/cashflow_graph.html``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_TEMPLATE_PATH = Path(__file__).resolve().parent.parent / "static" / "cashflow_graph.html"


def render_graph_html(graph_payload: dict[str, Any]) -> str:
    template = _TEMPLATE_PATH.read_text(encoding="utf-8")
    return template.replace("__GRAPH_DATA__", json.dumps(graph_payload))
