"""
Main Streamlit application entry point for DocAgent.

Run with:
    streamlit run ui/app.py

Features:
- Premium dark purple theme
- Drag-and-drop multi-file upload (PDF, Excel, CSV)
- Real-time pipeline progress with status messages
- Tabbed results: Summary | Questions | Raw Text | Metadata
- Themed download buttons: PDF (purple) and Markdown (cyan)
- Sidebar: Groq API status + config panel
"""

from __future__ import annotations

import sys
import time
import os
from pathlib import Path

# ── Path fixup so app can be run from repo root ────────────────────────────────
ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

# ── Page config (must be first Streamlit call) ─────────────────────────────────
st.set_page_config(
    page_title="DocAgent — AI Document Analyzer",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/your-org/docagent",
        "About": "DocAgent v1.0.0 — AI Document Understanding via Groq",
    },
)

from utils.config import load_config
from utils.logger import setup_logging, get_logger
from utils.file_handler import save_upload, cleanup_temp_dir, make_temp_dir, validate_file

# Load config & initialise logging once
_cfg = load_config()
setup_logging(level=_cfg.log_level, log_file=_cfg.log_file)
logger = get_logger("ui.app")


# ── CSS injection ──────────────────────────────────────────────────────────────

def _inject_css() -> None:
    css_path = Path(__file__).parent / "styles" / "custom.css"
    if css_path.exists():
        with open(css_path, encoding="utf-8") as fh:
            css = fh.read()
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

_inject_css()


# ── Groq API check ─────────────────────────────────────────────────────────────

def _get_api_key() -> str:
    """Get API key from environment or config."""
    return os.environ.get("GROQ_API_KEY") or _cfg.groq.api_key or ""


# ── Sidebar ────────────────────────────────────────────────────────────────────

def _render_sidebar(cfg) -> dict:
    """Render the sidebar and return any user-overridden settings."""
    overrides: dict = {}

    with st.sidebar:
        st.markdown("## ⚙️ DocAgent")
        st.caption(f"v{cfg.version} · Groq Cloud Powered")
        st.divider()

        # ── Groq status ──────────────────────────────────────────────
        st.markdown("### 🤖 LLM Engine")
        key = _get_api_key()
        has_key = bool(key) and "PASTE" not in key
        
        if has_key:
            st.markdown(
                '<span class="llm-status llm-online">● Groq API Connected</span>',
                unsafe_allow_html=True,
            )
            # Mask the key for display
            masked_key = f"{key[:4]}...{key[-4:]}" if len(key) > 8 else "****"
            st.caption(f"API Key: `{masked_key}`")
        else:
            st.markdown(
                '<span class="llm-status llm-offline">◉ Groq API Key Missing</span>',
                unsafe_allow_html=True,
            )
            st.warning("Please set `GROQ_API_KEY` environment variable.")

        st.caption(f"Model: `{cfg.groq.model}`")

        with st.expander("Change Groq Model", expanded=False):
            model_input = st.text_input(
                "Model name", value=cfg.groq.model,
                help="Recommended: llama-3.3-70b-versatile, mixtral-8x7b-32768",
            )
            if model_input != cfg.groq.model:
                overrides["groq_model"] = model_input

        st.divider()

        # ── PDF settings ───────────────────────────────────────────────
        st.markdown("### 📄 PDF Settings")
        use_ocr = st.toggle(
            "Enable OCR (for scanned PDFs)",
            value=cfg.pdf.use_ocr_fallback,
            help="Requires Tesseract to be installed.",
        )
        overrides["use_ocr_fallback"] = use_ocr

        if use_ocr:
            ocr_lang = st.text_input(
                "Tesseract language", value=cfg.pdf.ocr_language,
                help="e.g. eng, fra, deu",
            )
            overrides["ocr_language"] = ocr_lang

        st.divider()

        # ── Processing settings ────────────────────────────────────────
        st.markdown("### 🔧 Processing")
        q_threshold = st.slider(
            "Questionnaire threshold",
            min_value=0.1, max_value=0.9, value=cfg.classification.questionnaire_threshold,
            step=0.05,
            help="Score ≥ threshold → classified as questionnaire",
        )
        overrides["questionnaire_threshold"] = q_threshold

        chunk_size = st.number_input(
            "Chunk size (chars)", value=cfg.summarization.chunk_size,
            min_value=500, max_value=8000, step=500,
            help="Characters per LLM context window chunk",
        )
        overrides["chunk_size"] = chunk_size

        st.divider()

        # ── Info ───────────────────────────────────────────────────────
        st.markdown("### 📊 Pipeline")
        st.caption(
            "**Parse** → **Clean** → **Classify** → **Summarize** → **Extract**"
        )
        st.divider()
        st.caption("Built with Streamlit · pdfplumber · Groq")

    return overrides


# ── Agent factory ──────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def _get_agent(model: str, q_threshold: float, chunk_size: int,
               use_ocr: bool, ocr_lang: str):
    """Build (and cache) a DocumentAgent with the given settings."""
    from agents.document_agent import DocumentAgent
    from utils.config import load_config

    cfg = load_config()
    agent_cfg = cfg.to_dict()
    agent_cfg["groq"]["model"]                          = model
    agent_cfg["classification"]["questionnaire_threshold"] = q_threshold
    agent_cfg["summarization"]["chunk_size"]            = chunk_size
    agent_cfg["pdf"]["use_ocr_fallback"]                = use_ocr
    agent_cfg["pdf"]["ocr_language"]                    = ocr_lang

    return DocumentAgent(config=agent_cfg)


# ── Hero header ────────────────────────────────────────────────────────────────

def _render_header() -> None:
    st.markdown("""
    <div class="hero-header fade-in">
      <h1 class="hero-title">🤖 DocAgent</h1>
      <p class="hero-subtitle">
        Intelligent Document Analysis powered by Groq Cloud
      </p>
    </div>
    """, unsafe_allow_html=True)


# ── Upload zone ────────────────────────────────────────────────────────────────

def _render_upload() -> list:
    """Render the file uploader and return list of uploaded files."""
    col = st.columns([1, 3, 1])[1]
    with col:
        files = st.file_uploader(
            "Drop files here or click to browse",
            type=["pdf", "xlsx", "xls", "csv"],
            accept_multiple_files=True,
            label_visibility="visible",
            help="PDF, Excel (.xlsx/.xls), or CSV · max 50 MB per file",
        )
    return files or []


# ── Progress display ───────────────────────────────────────────────────────────

STEP_LABELS = {
    "parse":             "📖 Parsing document…",
    "clean":             "🧹 Cleaning and normalising text…",
    "classify":          "🔍 Classifying document type…",
    "summarize":         "📝 Generating summary…",
    "extract_questions": "❓ Extracting questions…",
    "done":              "✅ Analysis complete",
}


def _run_pipeline(uploaded_file, overrides: dict):
    from ui.components.results_view import render_results

    tmp_dir = make_temp_dir()
    try:
        file_path = save_upload(uploaded_file.getvalue(), uploaded_file.name, tmp_dir)
        err = validate_file(file_path, max_size_mb=_cfg.max_file_size_mb)
        if err:
            st.error(f"❌ {err}")
            return

        agent = _get_agent(
            model        = overrides.get("groq_model", _cfg.groq.model),
            q_threshold  = overrides.get("questionnaire_threshold", _cfg.classification.questionnaire_threshold),
            chunk_size   = overrides.get("chunk_size", _cfg.summarization.chunk_size),
            use_ocr      = overrides.get("use_ocr_fallback", _cfg.pdf.use_ocr_fallback),
            ocr_lang     = overrides.get("ocr_language", _cfg.pdf.ocr_language),
        )

        progress_placeholder = st.empty()
        status_placeholder   = st.empty()
        total = len(STEP_LABELS)

        with progress_placeholder.container():
            bar = st.progress(0, text="Starting pipeline…")

        start_ts = time.monotonic()
        _orig_log = agent._log_step
        step_counter = [0]

        def _progress_log_step(skill_name, success, duration_ms, error=None):
            _orig_log(skill_name, success, duration_ms, error)
            step_counter[0] += 1
            pct = min(step_counter[0] / total, 0.95)
            label = STEP_LABELS.get(skill_name, f"⚙ {skill_name}…")
            bar.progress(pct, text=label)

        agent._log_step = _progress_log_step  # type: ignore[method-assign]
        result = agent.run(file_path)

        bar.progress(1.0, text="✅ Complete!")
        elapsed = time.monotonic() - start_ts

        with status_placeholder.container():
            if result.success:
                st.success(
                    f"✅ Analysis complete in **{elapsed:.1f}s** — "
                    f"classified as **{result.doc_type.replace('_', ' ').title()}**"
                )
            else:
                st.error(f"⚠️ Pipeline finished with errors.")

        render_results(result, export_cfg=_cfg.export)

    finally:
        cleanup_temp_dir(tmp_dir)


# ── Main app ───────────────────────────────────────────────────────────────────

def main() -> None:
    overrides = _render_sidebar(_cfg)
    _render_header()
    uploaded_files = _render_upload()

    if not uploaded_files:
        st.markdown("""
        <div class="empty-state fade-in" style="padding: 4rem 1rem">
          <span class="icon">📂</span>
          <h3>Upload a document to get started</h3>
          <p style="color:#475569">Supports PDF, Excel (.xlsx / .xls), and CSV files</p>
        </div>
        """, unsafe_allow_html=True)
        return

    for uf in uploaded_files:
        st.markdown(f"---\n### 🗂 `{uf.name}`")
        _run_pipeline(uf, overrides)


if __name__ == "__main__":
    main()
