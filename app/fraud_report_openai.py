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


def _build_user_prompt(payload_json: str, category_keys_present: list[str]) -> str:
    heading_hints = "\n".join(
        f'- Use exactly this heading for category "{key}": {_CATEGORY_HEADINGS.get(key, f"## {key}")}'
        for key in category_keys_present
    )
    return f"""You are producing a **Markdown-only** fraud audit annex for the CFO.

**Input data (JSON):**
```json
{payload_json}
```

**Instructions:**
1. Use **Markdown** only (no HTML). Write in professional French suitable for a compliance annex (or match the language of the JSON content if mixed).
2. For **each non-empty category** in the JSON, include a dedicated section with the heading specified below. **Do not** create sections for categories that are empty arrays or missing.
3. Under each heading, for **every** detected item in that category:
   - Summarize the facts (entities, amounts, patterns) in plain language.
   - Explain the **business and compliance risk**.
   - List **immediate actionable steps** (e.g. freeze account, request KYC refresh, escalate to AML, block pending transfers).
4. If a category has many items, group logically but keep all items addressed.
5. End the document with a short section titled `## Synthèse du risque` with an overall risk posture (élevé / modéré / faible) and 2–4 bullet priorities.

**Required headings for categories present in this payload:**
{heading_hints if heading_hints else "(Aucune alerte fournie : rédigez une courte note de conformité indiquant qu'aucun incident structuré n'a été transmis.)"}
"""


SYSTEM_PROMPT = """You are a Senior Financial Compliance Officer at a regulated financial institution.
You write concise, accurate audit-style reports for the Chief Financial Officer (CFO) and the board risk committee.
You never invent facts that are not supported by the structured data provided; you may infer standard industry interpretations of patterns only when clearly grounded in the alerts.
You prioritize clarity, regulatory tone, and actionable remediation."""


def generate_fraud_audit_report(
    categorized_alerts: dict[str, Any],
    *,
    model: str = "gpt-4o",
    timeout: float = 120.0,
) -> str:
    """
    Call OpenAI Chat Completions and return a Markdown fraud audit report.

    :param categorized_alerts: Dict mapping category names to lists of alert objects
        (e.g. ``Flux_Circulaire``, ``Reseau_de_Mules``). Empty lists are ignored by the model per instructions.
    :param model: ``gpt-4o`` (default) or ``gpt-4-turbo``.
    :param timeout: HTTP timeout in seconds.
    :return: Markdown string. On recoverable API failures, returns a Markdown document describing the error
        so callers can still return HTTP 200 with a body if desired.
    :raises: ``ValueError`` if ``OPENAI_API_KEY`` is not set.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key or not str(api_key).strip():
        raise ValueError(
            "OPENAI_API_KEY is not set or empty. Set it in the environment before calling generate_fraud_audit_report."
        )

    non_empty_keys = [
        k
        for k, v in categorized_alerts.items()
        if isinstance(v, list) and len(v) > 0
    ]
    # Omit empty list categories from the JSON sent to the model (ignore empty entirely).
    payload_for_model = {
        k: v
        for k, v in categorized_alerts.items()
        if isinstance(v, list) and len(v) > 0
    }
    payload_json = json.dumps(payload_for_model, ensure_ascii=False, indent=2)
    user_prompt = _build_user_prompt(payload_json, non_empty_keys)

    client = OpenAI(api_key=api_key, timeout=timeout)

    try:
        response = client.chat.completions.create(
            model=model,
            temperature=0.25,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
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
            f"**Limite de débit OpenAI** : veuillez réessayer dans quelques instants.\n\n"
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
