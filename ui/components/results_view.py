"""
Results view component for DocAgent Streamlit UI.

Renders the full PipelineResult in a tabbed interface:
    Tab 1 — Summary   (text + method badge)
    Tab 2 — Questions (numbered list, only for questionnaires)
    Tab 3 — Raw Text  (monospace scrollable)
    Tab 4 — Metadata  (key-value table + skill timings)

Also provides download buttons (Markdown and PDF) themed to the app color scheme.

PDF generation uses reportlab (pure-Python, no system dependencies).
"""

from __future__ import annotations

import io
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

from core.pipeline_result import PipelineResult


# ── Helper: render HTML block ─────────────────────────────────────────────────

def _html(content: str) -> None:
    st.markdown(content, unsafe_allow_html=True)


# ── PDF generator ─────────────────────────────────────────────────────────────

def generate_pdf_bytes(result: PipelineResult, font_size: int = 11, margin: float = 0.9) -> bytes:
    """
    Convert a PipelineResult to a styled PDF document using reportlab.
    Returns raw PDF bytes suitable for st.download_button.
    """
    try:
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_LEFT, TA_CENTER
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.lib.colors import HexColor
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, HRFlowable,
            Table, TableStyle, KeepTogether,
        )
    except ImportError:
        # Return a plaintext PDF fallback if reportlab missing
        return result.to_markdown().encode("utf-8")

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        rightMargin=margin * inch, leftMargin=margin * inch,
        topMargin=margin * inch,  bottomMargin=margin * inch,
    )

    PURPLE = HexColor("#7C3AED")
    CYAN   = HexColor("#0891b2")
    DARK   = HexColor("#1e1e2e")
    GRAY   = HexColor("#64748b")

    sw_s  = getSampleStyleSheet()
    title_s = ParagraphStyle("DA_Title",   parent=sw_s["Title"],
                              fontSize=22, textColor=PURPLE, spaceAfter=6, leading=26)
    h1_s    = ParagraphStyle("DA_H1",      parent=sw_s["Heading1"],
                              fontSize=13, textColor=PURPLE, spaceBefore=16, spaceAfter=4)
    h2_s    = ParagraphStyle("DA_H2",      parent=sw_s["Heading2"],
                              fontSize=11, textColor=CYAN, spaceBefore=10, spaceAfter=3)
    body_s  = ParagraphStyle("DA_Body",    parent=sw_s["Normal"],
                              fontSize=font_size, leading=16, spaceAfter=4, textColor=HexColor("#1a1a2e"))
    meta_k  = ParagraphStyle("DA_MetaKey", parent=sw_s["Normal"],
                              fontSize=9, textColor=PURPLE, fontName="Helvetica-Bold")
    meta_v  = ParagraphStyle("DA_MetaVal", parent=sw_s["Normal"],
                              fontSize=9, textColor=GRAY)
    note_s  = ParagraphStyle("DA_Note",    parent=sw_s["Normal"],
                              fontSize=8, textColor=GRAY, spaceAfter=2)

    def md_to_rl(text: str) -> str:
        """Markdown to ReportLab tag converter (**bold**, *italic*, bullets)."""
        if not text:
            return ""
        
        # 1. Escape XML special characters first
        safe = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        
        # 2. Handle bold (restore <b> tags بعد escaping)
        safe = re.sub(r"(\*\*|__)(.*?)\1", r"<b>\2</b>", safe)
        
        # 3. Handle italics (restore <i> tags)
        safe = re.sub(r"(?<![a-zA-Z0-9])(\*|_)([^\*_]+)\1(?![a-zA-Z0-9])", r"<i>\2</i>", safe)
        
        # 4. Handle inline code
        safe = re.sub(r"`([^`]+)`", r'<font name="Courier" color="#333333" size="9">\1</font>', safe)
        
        # 5. Handle starting bullets
        lines = []
        for line in safe.split("\n"):
            processed = line.strip()
            if processed.startswith(("* ", "- ")):
                processed = f"  &bull;  {processed[2:]}"
            lines.append(processed)
        
        return "\n".join(lines)

    def p(text: str, style=None, is_md: bool = False) -> Paragraph:
        """Create a styled paragraph, optionally converting markdown."""
        if is_md:
            safe = md_to_rl(text)
        else:
            # For non-markdown (headers), we allow explicit <b> but escape other symbols
            # Note: We don't escape < and > here because we use <b> in fixed headers
            safe = (text or "").replace("&", "&amp;")
        
        return Paragraph(safe, style or body_s)

    def hr():
        return HRFlowable(width="100%", thickness=0.5, color=HexColor("#d1d5db"), spaceAfter=8)

    story = []
    # ── Title block ────────────────────────────────────────────────────
    story.append(p("DocAgent — Analysis Report", title_s))
    story.append(hr())

    doc_label = result.doc_type.replace("_", " ").title()
    story.append(p(
        f"**File:** {result.file_name}   |   "
        f"**Format:** {result.file_type.upper()}   |   "
        f"**Document Type:** {doc_label}",
        meta_k, is_md=True
    ))
    story.append(p(
        f"**Words:** {result.word_count:,}   |   "
        f"**Pages/Sheets:** {result.page_count}   |   "
        f"**Classification:** {result.classification_confidence:.0%} ({result.classification_method})   |   "
        f"**Processing:** {result.processing_time_ms / 1000:.2f}s",
        meta_k, is_md=True
    ))
    story.append(Spacer(1, 14))

    # ── Summary ────────────────────────────────────────────────────────
    story.append(p("📋  Summary", h1_s))
    story.append(hr())
    summary_text = result.summary or "No summary generated."
    for para_text in summary_text.split("\n\n"):
        if para_text.strip():
            story.append(p(para_text.strip(), is_md=True))
    story.append(Spacer(1, 10))

    # ── Questions ─────────────────────────────────────────────────────
    if result.questions:
        story.append(p(f"❓  Extracted Questions  ({len(result.questions)} found)", h1_s))
        story.append(hr())
        for i, q in enumerate(result.questions, 1):
            story.append(p(f"{i}.   {q}"))
        story.append(Spacer(1, 10))

    # ── Metadata ───────────────────────────────────────────────────────
    clean_meta = {k: v for k, v in result.metadata.items() if v and k not in ("engine",)}
    if clean_meta:
        story.append(p("ℹ️  Document Metadata", h1_s))
        story.append(hr())
        table_data = [["Key", "Value"]] + [
            [Paragraph(str(k), meta_k), Paragraph(str(v)[:120], meta_v)]
            for k, v in list(clean_meta.items())[:20]
        ]
        tbl = Table(table_data, colWidths=[2.3 * inch, 4.5 * inch])
        tbl.setStyle(TableStyle([
            ("BACKGROUND",  (0, 0), (-1, 0),  HexColor("#7C3AED")),
            ("TEXTCOLOR",   (0, 0), (-1, 0),  colors.white),
            ("FONTNAME",    (0, 0), (-1, 0),  "Helvetica-Bold"),
            ("FONTSIZE",    (0, 0), (-1, 0),  9),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [HexColor("#f8f9fa"), colors.white]),
            ("GRID",        (0, 0), (-1, -1),  0.25, HexColor("#d1d5db")),
            ("VALIGN",      (0, 0), (-1, -1),  "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1),  6),
            ("RIGHTPADDING",(0, 0), (-1, -1),  6),
            ("TOPPADDING",  (0, 0), (-1, -1),  4),
            ("BOTTOMPADDING",(0,0), (-1, -1),  4),
        ]))
        story.append(tbl)
        story.append(Spacer(1, 10))

    # ── Skill timings ──────────────────────────────────────────────────
    if result.skill_timings:
        story.append(p("⏱  Skill Timing Breakdown", h1_s))
        story.append(hr())
        
        timing_data = [["Skill Component", "Duration (ms)"]]
        for skill, ms in result.skill_timings.items():
            timing_data.append([
                Paragraph(skill.replace("_", " ").title(), body_s), 
                Paragraph(f"{ms:,.0f} ms", body_s)
            ])
            
        t_tbl = Table(timing_data, colWidths=[3.5 * inch, 3.3 * inch])
        t_tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), HexColor("#f3f4f6")),
            ("TEXTCOLOR",  (0, 0), (-1, 0), PURPLE),
            ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",   (0, 0), (-1, 0), 10),
            ("GRID",       (0, 0), (-1, -1), 0.5, HexColor("#e5e7eb")),
            ("VALIGN",     (0, 0), (-1, -1), "MIDDLE"),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING",    (0, 0), (-1, -1), 6),
        ]))
        story.append(t_tbl)
        story.append(Spacer(1, 10))

    # ── Errors / warnings ──────────────────────────────────────────────
    if result.errors:
        story.append(p("⚠️  Errors", h2_s))
        for err in result.errors:
            story.append(p(f"  ✘ {err}", note_s))

    # ── Footer ─────────────────────────────────────────────────────────
    story.append(Spacer(1, 30))
    story.append(hr())
    story.append(p(
        f"Generated by DocAgent v1.0.0  ·  {result.processing_time_ms / 1000:.2f}s  ·  "
        f"Summary via {result.summary_method}",
        note_s
    ))

    doc.build(story)
    return buf.getvalue()


# ── Main render function ──────────────────────────────────────────────────────

def render_results(result: PipelineResult, export_cfg: Optional[Any] = None) -> None:
    """Render the full PipelineResult in a tabbed Streamlit UI."""

    # ── Document type banner ───────────────────────────────────────────
    if result.doc_type == "questionnaire":
        icon, label, css_cls = "📋", "Questionnaire / Form", "questionnaire"
    else:
        icon, label, css_cls = "📄", "Normal Document", "normal"

    conf_pct = f"{result.classification_confidence:.0%}"
    _html(f"""
    <div class="doc-type-banner {css_cls} fade-in">
      <span style="font-size:1.5rem">{icon}</span>
      <div>
        <div style="font-size:1rem;font-weight:700">{label}</div>
        <div style="font-size:0.78rem;opacity:.75;font-weight:400;margin-top:2px">
          {conf_pct} confidence · {result.classification_method} · {result.summary_method}
        </div>
      </div>
    </div>
    """)

    # ── Stats row ──────────────────────────────────────────────────────
    _html(f"""
    <div class="stat-grid fade-in">
      <div class="stat-card">
        <div class="stat-value">{result.word_count:,}</div>
        <div class="stat-label">Words</div>
      </div>
      <div class="stat-card">
        <div class="stat-value">{result.page_count}</div>
        <div class="stat-label">Pages / Sheets</div>
      </div>
      <div class="stat-card">
        <div class="stat-value">{len(result.questions)}</div>
        <div class="stat-label">Questions</div>
      </div>
      <div class="stat-card">
        <div class="stat-value">{result.processing_time_ms / 1000:.1f}s</div>
        <div class="stat-label">Processing</div>
      </div>
    </div>
    """)

    # ── Download buttons ───────────────────────────────────────────────
    md_bytes  = result.to_markdown().encode("utf-8")
    fname     = Path(result.file_name).stem

    font_size = getattr(export_cfg, "pdf_font_size", 11) if export_cfg else 11
    margin    = getattr(export_cfg, "pdf_margin_inch", 0.9) if export_cfg else 0.9

    with st.spinner("Preparing downloads…"):
        try:
            pdf_bytes = generate_pdf_bytes(result, font_size=font_size, margin=margin)
        except Exception:
            pdf_bytes = md_bytes  # fallback to md bytes if PDF fails

    dl1, dl2 = st.columns(2)
    with dl1:
        _html('<div class="btn-pdf">')
        st.download_button(
            label="⬇ Download as PDF",
            data=pdf_bytes,
            file_name=f"{fname}_docagent_report.pdf",
            mime="application/pdf",
            use_container_width=True,
            key="dl_pdf",
        )
        _html("</div>")

    with dl2:
        _html('<div class="btn-md">')
        st.download_button(
            label="⬇ Download as Markdown",
            data=md_bytes,
            file_name=f"{fname}_docagent_report.md",
            mime="text/markdown",
            use_container_width=True,
            key="dl_md",
        )
        _html("</div>")

    st.divider()

    # ── Tabs ───────────────────────────────────────────────────────────
    tab_labels = ["📋 Summary", "❓ Questions", "📄 Raw Text", "ℹ️ Metadata"]
    tabs = st.tabs(tab_labels)

    # ── Tab 1: Summary ─────────────────────────────────────────────────
    with tabs[0]:
        if result.summary:
            method_label = result.summary_method.replace("_", " ").title()
            _html(f'<span class="badge badge-purple" style="margin-bottom:.75rem">⚡ {method_label}</span>')
            _html(f'<div class="summary-text fade-in">{result.summary}</div>')
        else:
            _html("""
            <div class="empty-state">
              <span class="icon">🌫</span>
              <h3>No summary generated</h3>
              <p>Try enabling Ollama or check for errors in the Metadata tab.</p>
            </div>""")

    # ── Tab 2: Questions ───────────────────────────────────────────────
    with tabs[1]:
        if result.questions:
            q_method   = result.question_extraction_method.upper()
            q_badge_cls = "badge-cyan" if "llm" in q_method.lower() else "badge-purple"
            _html(f'<span class="badge {q_badge_cls}" style="margin-bottom:.75rem">⚙ {q_method}</span>')
            _html('<div class="fade-in">')
            for i, q in enumerate(result.questions, 1):
                _html(f"""
                <div class="question-item">
                  <div class="question-num">{i}</div>
                  <div class="question-text">{q}</div>
                </div>""")
            _html("</div>")
        elif result.doc_type == "questionnaire":
            _html("""
            <div class="empty-state">
              <span class="icon">🔍</span>
              <h3>No questions found</h3>
              <p>The document was classified as a questionnaire but no questions could be extracted.</p>
            </div>""")
        else:
            _html(f"""
            <div class="empty-state">
              <span class="icon">📄</span>
              <h3>Not a questionnaire</h3>
              <p>This document was classified as a <b>Normal Document</b> 
              ({result.classification_confidence:.0%} confidence).
              Question extraction only runs on questionnaires.</p>
            </div>""")

    # ── Tab 3: Raw Text ────────────────────────────────────────────────
    with tabs[2]:
        if result.raw_text:
            preview = result.raw_text[:8000]
            truncated = len(result.raw_text) > 8000
            _html(f'<div class="raw-text-block fade-in">{st.session_state.get("_rt_safe", preview)}</div>')
            # Use st.text_area for proper rendering
            st.text_area(
                "Raw extracted text",
                value=result.raw_text[:10_000],
                height=400,
                label_visibility="collapsed",
            )
            if truncated:
                _html(f'<p style="color:#64748b;font-size:.8rem">Showing first 10,000 of {len(result.raw_text):,} characters.</p>')
        else:
            _html('<div class="empty-state"><span class="icon">📭</span><h3>No text available</h3></div>')

    # ── Tab 4: Metadata ────────────────────────────────────────────────
    with tabs[3]:
        _render_metadata(result)


def _render_metadata(result: PipelineResult) -> None:
    """Render metadata table + timing breakdown."""
    # Document metadata
    clean_meta = {k: v for k, v in result.metadata.items()
                  if v and str(v).strip() and k not in ("engine",)}

    if clean_meta:
        _html('<div class="glass-card fade-in" style="margin-bottom:1rem">')
        _html("<h4 style='color:#a78bfa;margin:0 0 .75rem'>Document Metadata</h4>")
        _html('<table class="meta-table">')
        for k, v in list(clean_meta.items())[:30]:
            _html(f"<tr><td>{k}</td><td>{str(v)[:200]}</td></tr>")
        _html("</table></div>")

    # Skill timing breakdown
    if result.skill_timings:
        _html('<div class="glass-card fade-in">')
        _html("<h4 style='color:#a78bfa;margin:0 0 .75rem'>⏱ Skill Timing</h4>")
        total_ms = sum(result.skill_timings.values())
        for skill, ms in result.skill_timings.items():
            pct = ms / max(total_ms, 1) * 100
            _html(f"""
            <div class="timing-bar-wrap">
              <div class="timing-label">
                <span>{skill}</span>
                <span>{ms:.0f} ms</span>
              </div>
              <div class="timing-bar-bg">
                <div class="timing-bar-fill" style="width:{pct:.1f}%"></div>
              </div>
            </div>""")
        _html(f'<p style="font-size:.78rem;color:#475569;margin-top:.5rem">Total: {total_ms:.0f} ms</p>')
        _html("</div>")

    # Errors block
    if result.errors or result.warnings:
        _html('<div class="glass-card fade-in" style="margin-top:1rem">')
        if result.errors:
            _html("<h4 style='color:#f87171;margin:0 0 .5rem'>Errors</h4>")
            for e in result.errors:
                _html(f'<p style="color:#f87171;font-size:.85rem">✘ {e}</p>')
        if result.warnings:
            _html("<h4 style='color:#fbbf24;margin:.5rem 0'>Warnings</h4>")
            for w in result.warnings:
                _html(f'<p style="color:#fbbf24;font-size:.85rem">⚠ {w}</p>')
        _html("</div>")
