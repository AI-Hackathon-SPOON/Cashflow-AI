"""
LLM-backed fraud audit report (Markdown) for API use.

Requires ``OPENAI_API_KEY`` in the environment. Uses the OpenAI Python SDK v1.x.
"""

from __future__ import annotations

import json
import os
from typing import Any

# OpenAI SDK v1+
from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI, RateLimitError

# Human-readable headings per category key (extend as needed).
_CATEGORY_HEADINGS: dict[str, str] = {
    "Flux_Circulaire": "## 🔄 Fraude par flux circulaire",
    "Reseau_de_Mules": "## 🕸️ Réseau de mules",
    "Horodatage_Anormal": "## ⏰ Horodatage anormal",
    "Montants_Suspects": "## 💰 Montants suspects",
    "Comptes_Hub": "## 📡 Comptes hub / structuration",
    "Autres_Signaux": "## ⚠️ Autres signaux",
}


def _require_openai_api_key() -> str:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key or not str(api_key).strip():
        raise ValueError(
            "OPENAI_API_KEY is not set or empty. Set it in the environment before calling OpenAI-backed helpers."
        )
    return str(api_key).strip()


def _openai_chat_markdown(
    *,
    system_prompt: str,
    user_prompt: str,
    model: str,
    timeout: float,
) -> str:
    """Run chat completion; return Markdown body or a Markdown error section (never raises on API errors)."""
    api_key = _require_openai_api_key()
    try:
        client = OpenAI(api_key=api_key, timeout=timeout)
        response = client.chat.completions.create(
            model=model,
            temperature=0.25,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        choice = response.choices[0].message
        content = (choice.content or "").strip()
        if not content:
            return (
                "## Rapport indisponible\n\n"
                "Le modèle a renvoyé une réponse vide. Vérifiez les journaux et réessayez."
            )
        return content

    except RateLimitError as e:
        return (
            "## Rapport temporairement indisponible\n\n"
            "**Limite de débit OpenAI** : veuillez réessayer dans quelques instants.\n\n"
            f"Détail technique : `{type(e).__name__}`"
        )
    except APITimeoutError as e:
        return (
            "## Rapport temporairement indisponible\n\n"
            f"**Délai d'attente dépassé** lors de l'appel à l'API OpenAI.\n\n"
            f"Détail : `{type(e).__name__}`"
        )
    except APIConnectionError as e:
        return (
            "## Rapport temporairement indisponible\n\n"
            f"**Erreur de connexion** vers OpenAI. Vérifiez le réseau et le pare-feu.\n\n"
            f"Détail : `{type(e).__name__}: {e}`"
        )
    except APIStatusError as e:
        return (
            "## Rapport temporairement indisponible\n\n"
            f"**Erreur HTTP OpenAI** ({getattr(e, 'status_code', 'unknown')}).\n\n"
            f"Détail : `{type(e).__name__}: {e}`"
        )
    except Exception as e:  # noqa: BLE001 — last-resort for unexpected SDK/runtime errors
        return (
            "## Rapport temporairement indisponible\n\n"
            f"**Erreur inattendue** lors de la génération du rapport.\n\n"
            f"Détail : `{type(e).__name__}: {e}`"
        )


def _openai_chat_json_object(
    *,
    system_prompt: str,
    user_prompt: str,
    model: str,
    timeout: float,
) -> dict[str, Any]:
    """Chat completion with ``response_format`` JSON object; returns parsed dict or ``{"_error": ...}``."""
    api_key = _require_openai_api_key()
    try:
        client = OpenAI(api_key=api_key, timeout=timeout)
        response = client.chat.completions.create(
            model=model,
            temperature=0.25,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        raw = (response.choices[0].message.content or "").strip()
        if not raw:
            return {"_error": "empty_response"}
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {"_error": "not_a_json_object"}
    except json.JSONDecodeError as e:
        return {"_error": f"json_decode: {e}"}
    except RateLimitError as e:
        return {"_error": f"rate_limit: {type(e).__name__}"}
    except APITimeoutError as e:
        return {"_error": f"timeout: {type(e).__name__}"}
    except APIConnectionError as e:
        return {"_error": f"connection: {type(e).__name__}: {e}"}
    except APIStatusError as e:
        return {"_error": f"http_{getattr(e, 'status_code', 'unknown')}: {e}"}
    except Exception as e:  # noqa: BLE001
        return {"_error": f"{type(e).__name__}: {e}"}


def _build_user_prompt(
    payload_json: str,
    category_keys_present: list[str],
    *,
    kb_global_json: str | None = None,
) -> str:
    heading_hints = "\n".join(
        f'- Use exactly this heading for category "{key}": {_CATEGORY_HEADINGS.get(key, f"## {key}")}'
        for key in category_keys_present
    )
    kb_block = ""
    if kb_global_json:
        kb_block = f"""
**Retrieved institutional patterns** (vector DB / past investigations — use only as **supporting precedent**; never invent facts not present in the alert payload or in this JSON):
```json
{kb_global_json}
```
"""
    return f"""You are producing a **Markdown-only** fraud audit annex for the CFO.

**Input data (JSON):**
```json
{payload_json}
```
{kb_block}
**Instructions:**
1. Use **Markdown** only (no HTML). Write in professional French suitable for a compliance annex (or match the language of the JSON content if mixed).
2. For **each non-empty category** in the JSON, include a dedicated section with the heading specified below. **Do not** create sections for categories that are empty arrays or missing.
3. Under each heading, for **every** detected item in that category:
   - Summarize the facts (entities, amounts, patterns) in plain language.
   - Explain the **business and compliance risk**.
   - List **immediate actionable steps** (e.g. freeze account, request KYC refresh, escalate to AML, block pending transfers).
4. If a category has many items, group logically but keep all items addressed.
5. When institutional pattern JSON is provided above, briefly connect alert items to those precedents **only when clearly relevant** (cite ``case_id`` / ``outcome`` when present); otherwise omit.
6. End the document with a short section titled `## Synthèse du risque` with an overall risk posture (élevé / modéré / faible) and 2–4 bullet priorities.

**Required headings for categories present in this payload:**
{heading_hints if heading_hints else "(Aucune alerte fournie : rédigez une courte note de conformité indiquant qu'aucun incident structuré n'a été transmis.)"}
"""


SYSTEM_PROMPT = """You are a Senior Financial Compliance Officer at a regulated financial institution.
You write concise, accurate audit-style reports for the Chief Financial Officer (CFO) and the board risk committee.
You never invent facts that are not supported by the structured data provided; you may infer standard industry interpretations of patterns only when clearly grounded in the alerts.
You prioritize clarity, regulatory tone, and actionable remediation."""


COMPARATIVE_KB_SYSTEM_PROMPT = """You are a Senior Compliance Officer at a regulated financial institution.
You write short, decisive **alert reports** for the CFO when a new graph/network anomaly is detected.
You compare the new case only to the **internal knowledge base excerpts** provided (e.g. top matches from a vector database retrieved via n8n).
You must not invent past cases: if the retrieval payload is thin or ambiguous, say so and recommend human review.
When past outcomes are labeled **False Positive**, treat them as precedent for benign or acceptable patterns only when the facts align strongly.
When past outcomes are **Confirmed Fraud**, treat them as precedent for escalation only when the facts align strongly.
You always state your confidence level and what additional data would reduce uncertainty."""


def _build_kb_comparative_user_prompt(anomaly_json: str, kb_json: str) -> str:
    return f"""My graph algorithm has detected a **new anomaly**. Data (JSON):

```json
{anomaly_json}
```

Here is our **internal knowledge base** — the **top similar past events** (typically the top 3 chunks retrieved from a vector DB by n8n or an equivalent pipeline). Use only these as precedent:

```json
{kb_json}
```

**Task:** Write an **alert report** in Markdown for the CFO.

1. Briefly describe the new anomaly in plain language.
2. For **each** retrieved past event, compare it to the new anomaly (similarities and differences). Cite `case_id` / identifiers if present.
3. If the new case **closely matches** a past **False Positive** precedent, advise the CFO that this is **likely safe** or low priority, with caveats.
4. If it **closely matches** a past **Confirmed Fraud** precedent, **escalate urgently** with concrete next steps (freeze, SAR, AML committee, etc.).
5. If precedent is mixed or weak, recommend **further investigation** and list 3–5 concrete checks.
6. End with a one-paragraph **executive recommendation** (safe / monitor / escalate)."""


def generate_fraud_audit_report(
    categorized_alerts: dict[str, Any],
    *,
    kb_global_context: list[dict[str, Any]] | None = None,
    max_kb_global_items: int = 15,
    model: str = "gpt-4o",
    timeout: float = 120.0,
) -> str:
    """
    Call OpenAI Chat Completions and return a Markdown fraud audit report.

    :param categorized_alerts: Dict mapping category names to lists of alert objects
        (e.g. ``Flux_Circulaire``, ``Reseau_de_Mules``). Empty lists are ignored by the model per instructions.
    :param kb_global_context: Optional top chunks from a **vector DB** (same dataset-wide RAG context for the whole
        annex). Typically produced by n8n after embedding similarity search over **learned patterns**.
    :param max_kb_global_items: Cap on how many KB objects are injected into the prompt.
    :param model: ``gpt-4o`` (default) or ``gpt-4-turbo``.
    :param timeout: HTTP timeout in seconds.
    :return: Markdown string. On recoverable API failures, returns a Markdown document describing the error
        so callers can still return HTTP 200 with a body if desired.
    :raises: ``ValueError`` if ``OPENAI_API_KEY`` is not set.
    """
    non_empty_keys = [
        k
        for k, v in categorized_alerts.items()
        if isinstance(v, list) and len(v) > 0
    ]
    payload_for_model = {
        k: v
        for k, v in categorized_alerts.items()
        if isinstance(v, list) and len(v) > 0
    }
    payload_json = json.dumps(payload_for_model, ensure_ascii=False, indent=2)
    kb_global_json: str | None = None
    if kb_global_context:
        top = [x for x in kb_global_context if isinstance(x, dict)][: max(0, int(max_kb_global_items))]
        if top:
            kb_global_json = json.dumps(top, ensure_ascii=False, indent=2)
    user_prompt = _build_user_prompt(payload_json, non_empty_keys, kb_global_json=kb_global_json)

    return _openai_chat_markdown(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        model=model,
        timeout=timeout,
    )


def generate_kb_comparative_alert(
    anomaly: dict[str, Any] | str,
    kb_retrievals: list[dict[str, Any]],
    *,
    model: str = "gpt-4o",
    timeout: float = 120.0,
    max_kb_items: int = 3,
) -> str:
    """
    Compare a **new** graph anomaly to **vector-retrieved** past events (e.g. n8n → top-k from a vector DB).

    Each ``kb_retrievals`` item should be JSON-serializable and ideally include fields such as:
    ``case_id``, ``outcome`` (e.g. ``"False Positive"``, ``"Confirmed Fraud"``), ``summary``,
    ``similarity_score`` / ``score``, ``date``, ``resolution_notes``.

    Only the first ``max_kb_items`` entries are sent to the model.

    :raises: ``ValueError`` if ``OPENAI_API_KEY`` is not set.
    """
    if isinstance(anomaly, str):
        anomaly_obj: dict[str, Any] = {"description": anomaly}
    else:
        anomaly_obj = dict(anomaly)

    anomaly_json = json.dumps(anomaly_obj, ensure_ascii=False, indent=2)
    top = list(kb_retrievals)[: max(0, int(max_kb_items))]
    kb_json = json.dumps(top, ensure_ascii=False, indent=2)
    user_prompt = _build_kb_comparative_user_prompt(anomaly_json, kb_json)

    return _openai_chat_markdown(
        system_prompt=COMPARATIVE_KB_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        model=model,
        timeout=timeout,
    )


RAG_PER_TXN_SYSTEM_PROMPT = """You are a financial crime compliance analyst.
You explain **flagged transactions** using structured scoring fields and optional **retrieved knowledge chunks**
from a vector database (RAG). Retrieval is **hinting evidence only** — if KB text does not align with the transaction, say so explicitly.
Use Markdown only (no HTML). Prefer concise, auditable wording; French when the transaction content is predominantly French."""


def _normalize_rag_item(item: Any, *, max_kb_per_item: int) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None
    tid = str(item.get("transaction_id", "")).strip()
    txn = item.get("transaction")
    if not isinstance(txn, dict):
        return None
    kb_raw = item.get("kb_retrievals")
    kb_list: list[dict[str, Any]] = []
    if isinstance(kb_raw, list):
        kb_list = [x for x in kb_raw if isinstance(x, dict)][: max(0, int(max_kb_per_item))]
    use_id = tid or str(txn.get("transaction_id", "")).strip() or "?unknown?"
    return {"transaction_id": use_id, "transaction": txn, "kb_retrievals": kb_list}


def _build_rag_chunk_user_prompt(batch: list[dict[str, Any]], batch_index: int, batch_total: int) -> str:
    body = json.dumps(batch, ensure_ascii=False, indent=2)
    return f"""Process **batch {batch_index} of {batch_total}** of flagged transactions. Each object has
``transaction_id``, structured ``transaction`` fields (scores, accounts, etc.), and ``kb_retrievals`` (top similar
**learned patterns** from your vector DB / n8n pipeline — may be empty).

**Batch JSON:**
```json
{body}
```

For **every** object in this array, output Markdown using this template (repeat per row):

### `transaction_id` (literal value from JSON)

- **Contrôle des faits** — 2–4 bullets, grounded only in ``transaction`` and ``kb_retrievals``.
- **Parallèles base de connaissances (RAG)** — if ``kb_retrievals`` is empty, write one bullet: *Aucun extrait KB fourni pour cette transaction.* Otherwise reference ``case_id`` / ``outcome`` / ``summary`` when relevant; if similarity is weak or contradictory, say so.
- **Interprétation risque** — one short paragraph (fraude plausible vs motif opérationnel banal).
- **Actions recommandées** — numbered list, max 4 concrete controls (KYC, freeze, sample review, etc.).

Do not omit any transaction in this batch."""


ROW_RAG_JSON_SYSTEM_PROMPT = """You are a financial crime compliance analyst.
You explain transactions using structured scoring fields and optional retrieved knowledge (RAG). KB chunks are hints only;
if they do not match the transaction, say so. Never invent facts outside the JSON you receive.
You respond with **only** valid JSON (no markdown fences, no commentary)."""


def _build_row_explanation_json_user_prompt(batch: list[dict[str, Any]], batch_index: int, batch_total: int) -> str:
    body = json.dumps(batch, ensure_ascii=False, indent=2)
    ids_list = [str(x.get("transaction_id", "")) for x in batch]
    return f"""Process **batch {batch_index} of {batch_total}**.

**Input JSON** (each item: transaction_id, transaction, kb_retrievals):
```json
{body}
```

Return **one JSON object** with this exact shape:
{{"explanations": {{ "<transaction_id>": "<string>", ... }} }}

Rules:
- Include **every** transaction_id from this batch exactly once as a key. Expected ids: {json.dumps(ids_list, ensure_ascii=False)}.
- Each value is a single plain-text explanation (3–8 sentences): facts from ``transaction``, how ``kb_retrievals`` relates or that none were provided, risk interpretation, and 1–3 concrete next steps when the row is flagged; for non-flagged rows, briefly justify low monitoring priority vs any weak KB parallels.
- Professional French unless the transaction text is predominantly English."""


def generate_per_row_rag_explanation_map(
    items: list[dict[str, Any]],
    *,
    chunk_size: int = 10,
    max_kb_per_item: int = 5,
    model: str = "gpt-4o",
    timeout: float = 180.0,
) -> dict[str, str]:
    """
    For each item (same shape as :func:`generate_flagged_rag_explanations`), call the LLM and collect
    ``transaction_id -> explanation`` strings. Uses JSON response mode for stable parsing.

    On partial API errors, missing ids are omitted from the result; callers may retry.
    """
    max_kb = max(0, int(max_kb_per_item))
    normalized: list[dict[str, Any]] = []
    for raw in items:
        n = _normalize_rag_item(raw, max_kb_per_item=max_kb)
        if n is not None:
            normalized.append(n)

    if not normalized:
        return {}

    cs = max(1, min(int(chunk_size), 25))
    batches: list[list[dict[str, Any]]] = []
    for i in range(0, len(normalized), cs):
        batches.append(normalized[i : i + cs])

    out: dict[str, str] = {}
    for idx, batch in enumerate(batches, start=1):
        user_prompt = _build_row_explanation_json_user_prompt(batch, idx, len(batches))
        payload = _openai_chat_json_object(
            system_prompt=ROW_RAG_JSON_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            model=model,
            timeout=timeout,
        )
        if "_error" in payload:
            for it in batch:
                tid = str(it.get("transaction_id", ""))
                if tid:
                    out[tid] = f"[LLM error: {payload['_error']}]"
            continue
        expl = payload.get("explanations")
        if not isinstance(expl, dict):
            for it in batch:
                tid = str(it.get("transaction_id", ""))
                if tid:
                    out[tid] = "[LLM error: invalid explanations object]"
            continue
        for it in batch:
            tid = str(it.get("transaction_id", ""))
            if not tid:
                continue
            val = expl.get(tid)
            if isinstance(val, str) and val.strip():
                out[tid] = val.strip()
            else:
                out[tid] = "[LLM error: missing explanation for this transaction_id]"

    return out


def generate_flagged_rag_explanations(
    items: list[dict[str, Any]],
    *,
    chunk_size: int = 10,
    max_kb_per_item: int = 5,
    model: str = "gpt-4o",
    timeout: float = 180.0,
) -> str:
    """
    Generate Markdown explanations for **flagged** transactions using **per-row RAG** (``kb_retrievals``).

    Typical flow: (1) embed **learned patterns** into a vector DB; (2) for each new flagged ``transaction_id``,
    your orchestrator (e.g. n8n) runs similarity search and attaches ``kb_retrievals``; (3) call this function with
    the merged list. Large lists are processed in **chunks** (several model calls).

    Each item shape::

        {
          "transaction_id": "...",
          "transaction": { ... },  # structured row / snapshot
          "kb_retrievals": [ {"case_id", "outcome", "summary", ...}, ... ]
        }

    :return: Concatenated Markdown (chunk sections separated by horizontal rules). Empty ``items`` yields a short note.
    """
    max_kb = max(0, int(max_kb_per_item))
    normalized: list[dict[str, Any]] = []
    for raw in items:
        n = _normalize_rag_item(raw, max_kb_per_item=max_kb)
        if n is not None:
            normalized.append(n)

    if not normalized:
        return (
            "## Explications RAG\n\n"
            "Aucune transaction exploitable (liste vide ou objets invalides : attendu "
            "``transaction_id`` + ``transaction`` dict + ``kb_retrievals`` liste)."
        )

    cs = max(1, min(int(chunk_size), 25))
    batches: list[list[dict[str, Any]]] = []
    for i in range(0, len(normalized), cs):
        batches.append(normalized[i : i + cs])

    parts: list[str] = []
    for idx, batch in enumerate(batches, start=1):
        user_prompt = _build_rag_chunk_user_prompt(batch, idx, len(batches))
        parts.append(
            _openai_chat_markdown(
                system_prompt=RAG_PER_TXN_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                model=model,
                timeout=timeout,
            )
        )

    header = f"## Explications transactions (RAG) — {len(normalized)} transaction(s), {len(batches)} lot(s)\n\n"
    return header + "\n\n---\n\n".join(parts)
