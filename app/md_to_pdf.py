"""Convert Markdown to styled PDF bytes (SAR / compliance-style layout)."""

from __future__ import annotations

import html
from datetime import datetime, timezone
from io import BytesIO
from typing import Final

# xhtml2pdf: avoid flex/grid; prefer block layout and ReportLab-friendly fonts.
_SAR_CSS: Final = """
@page {
    size: A4;
    margin-top: 16mm;
    margin-bottom: 18mm;
    margin-left: 18mm;
    margin-right: 18mm;
}

html { margin: 0; padding: 0; }

body {
    margin: 0;
    padding: 0;
    font-family: "Times New Roman", Times, Georgia, serif;
    font-size: 11pt;
    line-height: 1.42;
    color: #111111;
    text-align: justify;
}

/* --- Cover / control sheet (distinct from narrative) --- */
.doc-front {
    margin-bottom: 14pt;
    padding-bottom: 12pt;
    border-bottom: 2pt solid #1a1a1a;
    page-break-after: avoid;
}

.class-banner {
    font-family: Helvetica, Arial, sans-serif;
    font-size: 7.5pt;
    font-weight: bold;
    letter-spacing: 0.6pt;
    text-transform: uppercase;
    text-align: center;
    color: #ffffff;
    background-color: #1a2744;
    padding: 6pt 8pt;
    margin: 0 0 10pt 0;
}

.doc-kind {
    font-family: Helvetica, Arial, sans-serif;
    font-size: 9pt;
    font-weight: bold;
    color: #333333;
    text-align: center;
    margin: 0 0 4pt 0;
}

.doc-main-title {
    font-family: Helvetica, Arial, sans-serif;
    font-size: 16pt;
    font-weight: bold;
    text-align: center;
    color: #0d1b2a;
    margin: 8pt 0 6pt 0;
    line-height: 1.2;
}

.doc-meta-strip {
    font-family: Courier, "Courier New", monospace;
    font-size: 8.5pt;
    color: #444444;
    text-align: center;
    margin: 0 0 2pt 0;
}

.doc-separator {
    margin-top: 10pt;
    border: none;
    border-top: 1pt solid #888888;
}

.narrative-label {
    font-family: Helvetica, Arial, sans-serif;
    font-size: 8.5pt;
    font-weight: bold;
    letter-spacing: 0.4pt;
    text-transform: uppercase;
    color: #555555;
    margin: 14pt 0 6pt 0;
    padding-bottom: 2pt;
    border-bottom: 0.5pt solid #cccccc;
}

/* --- Narrative body (SAR-style: justified, hierarchy) --- */
.doc-body {
    text-align: justify;
}

.doc-body h1 {
    font-family: Helvetica, Arial, sans-serif;
    font-size: 13pt;
    font-weight: bold;
    color: #0d1b2a;
    margin: 16pt 0 8pt 0;
    padding-bottom: 3pt;
    border-bottom: 1pt solid #333333;
    page-break-after: avoid;
    text-align: left;
}

.doc-body h2 {
    font-family: Helvetica, Arial, sans-serif;
    font-size: 11.5pt;
    font-weight: bold;
    color: #1b263b;
    margin: 14pt 0 6pt 0;
    page-break-after: avoid;
    text-align: left;
}

.doc-body h3 {
    font-family: Helvetica, Arial, sans-serif;
    font-size: 10.5pt;
    font-weight: bold;
    color: #2c3e50;
    margin: 10pt 0 4pt 0;
    page-break-after: avoid;
    text-align: left;
}

.doc-body h4, .doc-body h5, .doc-body h6 {
    font-family: Helvetica, Arial, sans-serif;
    font-size: 10pt;
    font-weight: bold;
    margin: 8pt 0 3pt 0;
    text-align: left;
}

.doc-body p {
    margin: 0 0 7pt 0;
    orphans: 2;
    widows: 2;
}

.doc-body strong {
    font-weight: bold;
    color: #000000;
}

.doc-body em {
    font-style: italic;
}

.doc-body ul, .doc-body ol {
    margin: 4pt 0 8pt 0;
    padding-left: 16pt;
}

.doc-body li {
    margin-bottom: 4pt;
}

.doc-body hr {
    border: none;
    border-top: 0.5pt solid #aaaaaa;
    margin: 12pt 0;
}

.doc-body blockquote {
    margin: 8pt 0 8pt 10pt;
    padding: 6pt 10pt;
    border-left: 3pt solid #415a77;
    background-color: #f4f6f8;
    font-size: 10pt;
    color: #333333;
    text-align: left;
}

.doc-body pre {
    font-family: Courier, "Courier New", monospace;
    font-size: 8pt;
    line-height: 1.3;
    background-color: #f0f0f0;
    border: 0.5pt solid #cccccc;
    padding: 6pt 8pt;
    margin: 8pt 0;
    white-space: pre-wrap;
    word-break: break-word;
    text-align: left;
}

.doc-body code {
    font-family: Courier, "Courier New", monospace;
    font-size: 8.5pt;
    background-color: #eeeeee;
    padding: 0 2pt;
}

.doc-body table {
    border-collapse: collapse;
    width: 100%;
    margin: 10pt 0;
    font-size: 9pt;
    text-align: left;
}

.doc-body th {
    font-family: Helvetica, Arial, sans-serif;
    font-size: 8.5pt;
    font-weight: bold;
    background-color: #1b263b;
    color: #ffffff;
    border: 0.5pt solid #1b263b;
    padding: 5pt 6pt;
}

.doc-body td {
    border: 0.5pt solid #bbbbbb;
    padding: 4pt 6pt;
    vertical-align: top;
}

.doc-body tr:nth-child(even) td {
    background-color: #fafafa;
}

/* --- Closing disclaimer --- */
.doc-footer-note {
    margin-top: 18pt;
    padding-top: 8pt;
    border-top: 0.5pt solid #999999;
    font-family: Helvetica, Arial, sans-serif;
    font-size: 7.5pt;
    line-height: 1.35;
    color: #555555;
    text-align: justify;
}
"""


def _escape_xml_text(s: str) -> str:
    """Escape for text nodes inside XHTML (markdown output is HTML; our shell is built manually)."""
    return html.escape(s, quote=True)


def _build_document_shell(
    *,
    markdown_html: str,
    document_title: str,
    document_kind: str,
    classification: str,
    generated_line: str,
) -> str:
    front = f"""
<div class="doc-front">
  <div class="class-banner">{_escape_xml_text(classification)}</div>
  <div class="doc-kind">{_escape_xml_text(document_kind)}</div>
  <div class="doc-main-title">{_escape_xml_text(document_title)}</div>
  <div class="doc-meta-strip">{_escape_xml_text(generated_line)}</div>
  <hr class="doc-separator" />
</div>
<div class="narrative-label">Narrative &amp; findings (structured sections)</div>
<div class="doc-body">
{markdown_html}
</div>
<div class="doc-footer-note">
  This document is an internal compliance worksheet generated from application data and, where applicable,
  model-assisted narrative text. It is not a filed Suspicious Activity Report (SAR) and does not satisfy
  jurisdictional filing requirements. Officers must verify all facts, amounts, and identities against
  source systems and records before any regulatory submission or external disclosure.
</div>
"""
    return front


def try_markdown_to_pdf(
    markdown_text: str,
    *,
    document_title: str = "Financial crime narrative",
    document_kind: str = "Internal compliance worksheet",
    classification: str = "CONFIDENTIAL — Internal use only",
) -> tuple[bytes | None, str | None]:
    """
    Render Markdown to a **styled** PDF (SAR-inspired layout: banner, title block, serif narrative, tables).

    :return: ``(pdf_bytes, None)`` on success, or ``(None, error_message)`` on failure.
    """
    text = (markdown_text or "").strip()
    if not text:
        return None, "empty_markdown"

    try:
        import markdown
        from xhtml2pdf import pisa
    except ImportError as e:
        return None, f"missing_dependency:{e}"

    try:
        raw_html = markdown.markdown(
            text,
            extensions=["extra", "nl2br"],
        )
    except Exception as e:  # noqa: BLE001
        return None, f"markdown_html:{type(e).__name__}:{e}"

    # Demote accidental lone top-level # so cover title stays primary; merge multiple h1s visually.
    markdown_html = raw_html
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    generated_line = f"Generated: {now} · Cashflow Fraud Monitor"

    shell = _build_document_shell(
        markdown_html=markdown_html,
        document_title=document_title,
        document_kind=document_kind,
        classification=classification,
        generated_line=generated_line,
    )

    full_html = (
        "<!DOCTYPE html>\n"
        '<html lang="en"><head><meta charset="utf-8"/>'
        f"<style>{_SAR_CSS}</style>"
        "</head><body>"
        f"{shell}"
        "</body></html>"
    )

    buf = BytesIO()
    try:
        status = pisa.CreatePDF(full_html, dest=buf, encoding="utf-8")
    except Exception as e:  # noqa: BLE001
        return None, f"pisa:{type(e).__name__}:{e}"

    if getattr(status, "err", False):
        return None, "pisa_reported_errors"

    out = buf.getvalue()
    if not out:
        return None, "empty_pdf_output"

    return out, None


def render_pdf_bytes(
    pdf_bytes: bytes,
    *,
    height: int = 760,
    download_filename: str = "report.pdf",
    download_key: str = "pdf_download",
) -> None:
    """
    Inline PDF viewer via ``st.pdf``.

    We **do not** use ``data:`` URLs in iframes: Chrome blocks them (“Cette page a été bloquée par Chrome”).
    A temp file on disk lets Streamlit serve the document like a normal PDF URL.
    """
    import os
    import tempfile

    import streamlit as st

    if not pdf_bytes:
        return

    ss_path_key = f"_pdf_preview_path_{download_key}"
    prev = st.session_state.get(ss_path_key)
    if isinstance(prev, str) and os.path.isfile(prev):
        try:
            os.unlink(prev)
        except OSError:
            pass

    path = ""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    try:
        tmp.write(pdf_bytes)
    except Exception:
        try:
            tmp.close()
        except OSError:
            pass
        try:
            os.unlink(tmp.name)
        except OSError:
            pass
        st.error("Could not write a temporary PDF for preview.")
    else:
        tmp.close()
        path = os.path.abspath(tmp.name)

    if path:
        st.session_state[ss_path_key] = path
        if hasattr(st, "pdf"):
            try:
                st.pdf(path, height=height)  # type: ignore[attr-defined]
            except Exception as e2:
                st.error(f"PDF preview failed ({e2}). Use the download below.")
        else:
            st.error("Upgrade Streamlit (1.44+) for built-in PDF preview, or use the download below.")

    with st.expander("Save PDF copy (download)", expanded=False):
        st.download_button(
            "Download PDF",
            data=pdf_bytes,
            file_name=download_filename,
            mime="application/pdf",
            key=download_key,
        )
