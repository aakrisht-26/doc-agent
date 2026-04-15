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

def _get_api_keys() -> list[str]:
    """Get list of API keys from environment or config."""
    raw_keys = (
        os.environ.get("GROQ_API_KEYS")
        or os.environ.get("GROQ_API_KEY")
        or _cfg.groq.api_keys
        or _cfg.groq.api_key
        or ""
    )
    return [k.strip() for k in raw_keys.split(",") if k.strip()]


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
        keys = _get_api_keys()
        has_keys = len(keys) > 0 and "PASTE" not in keys[0]
        
        if has_keys:
            count = len(keys)
            status_text = f"● Groq Connected ({count} key{'' if count==1 else 's'})"
            st.markdown(
                f'<span class="llm-status llm-online">{status_text}</span>',
                unsafe_allow_html=True,
            )
            # Mask the first key for display
            primary = keys[0]
            masked_key = f"{primary[:4]}...{primary[-4:]}" if len(primary) > 8 else "****"
            st.caption(f"Primary Key: `{masked_key}`")
        else:
            st.markdown(
                '<span class="llm-status llm-offline">◉ Groq API Key Missing</span>',
                unsafe_allow_html=True,
            )
            st.warning("Please set `GROQ_API_KEYS` in your `.env` file.")

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

        # ── Summary controls ───────────────────────────────────────────
        st.markdown("### 📝 Summary Controls")
        summary_length = st.select_slider(
            "Summary length",
            options=["short", "medium", "detailed"],
            value="medium",
            help="Controls how long and detailed the generated summary will be.",
        )
        overrides["summary_length"] = summary_length

        technical_level = st.select_slider(
            "Technical level",
            options=["beginner", "intermediate", "expert"],
            value="intermediate",
            help="Adjusts vocabulary and explanation depth.",
        )
        overrides["technical_level"] = technical_level

        st.divider()

        # ── Page Range ─────────────────────────────────────────────────
        st.markdown("### 📄 Page Range")
        st.caption("Leave both at 0 to process all pages.")
        page_col1, page_col2 = st.columns(2)
        with page_col1:
            page_from = st.number_input("From page", min_value=0, value=0, step=1)
        with page_col2:
            page_to = st.number_input("To page", min_value=0, value=0, step=1)
        overrides["page_from"] = int(page_from)
        overrides["page_to"] = int(page_to)

        st.divider()

        # ── Info ───────────────────────────────────────────────────────
        st.markdown("### 📊 Pipeline")
        st.caption(
            "**Parse** → **Clean** → **Classify** → **Summarize** → **Extract** → **Chat**"
        )
        st.divider()
        st.caption("Built with Streamlit · pdfplumber · Groq · sentence-transformers")

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
    agent_cfg["groq"]["model"]                             = model
    agent_cfg["classification"]["questionnaire_threshold"] = q_threshold
    agent_cfg["summarization"]["chunk_size"]               = chunk_size
    agent_cfg["pdf"]["use_ocr_fallback"]                   = use_ocr
    agent_cfg["pdf"]["ocr_language"]                       = ocr_lang

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

def _render_upload() -> dict:
    """Render the file uploader and return uploaded files and youtube url."""
    col = st.columns([1, 3, 1])[1]
    with col:
        files = st.file_uploader(
            "Drop files to analyze",
            type=["pdf", "xlsx", "xls", "csv", "mp3", "mp4", "wav", "m4a"],
            accept_multiple_files=True,
            label_visibility="visible",
            help="PDF, Excel, CSV, or Audio files · max 50 MB per file",
        )
        yt_url = st.text_input("Or paste a YouTube URL and press Enter:")
    return {"files": files or [], "youtube_url": yt_url}


# ── Progress display ───────────────────────────────────────────────────────────

STEP_LABELS = {
    # Core Readers/Cleaners
    "pdf_reader":            "📖 Reading PDF content…",
    "excel_reader":          "📊 Extracting Excel sheets…",
    "text_cleaner":          "🧹 Normalizing and cleaning text…",
    
    # Analysis Tasks
    "document_classifier":   "🔍 Classifying document (Domain detection)…",
    "structure_recognition": "👁️ Analyzing layout and tables…",
    "summarization":         "📝 Generating executive summary…",
    "question_extraction":   "❓ Identifying form questions…",
    
    # Advanced Skills
    "audio_transcription":   "🎙️ Transcribing audio (local Whisper)…",
    "youtube":               "📺 Fetching YouTube transcript…",
    "rag":                   "💬 Initializing local RAG Chat…",
    "tts":                   "🔊 Preparing audio output…",
    "compare_documents":     "⚖️ Comparing documents…",

    "done":                  "✅ Analysis complete",
}


def _run_pipeline(uploaded_file, overrides: dict):
    from ui.components.results_view import render_results

    # Include UX control overrides in the cache key so changing settings re-runs the pipeline
    ux_key = f"{overrides.get('summary_length','medium')}_{overrides.get('technical_level','intermediate')}_{overrides.get('page_from',0)}_{overrides.get('page_to',0)}"
    state_key = f"doc_result_{uploaded_file.file_id}_{ux_key}"
    if state_key in st.session_state:
        render_results(st.session_state[state_key], export_cfg=_cfg.export)
        return

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
        # Pass UX overrides into the agent for summarization
        agent._ux_overrides = overrides

        progress_placeholder = st.empty()
        status_placeholder   = st.empty()
        
        with progress_placeholder.container():
            bar = st.progress(0, text="Initializing pipeline…")

        # Start state
        start_ts = time.monotonic()
        _orig_log = agent._log_step
        step_counter = [0]

        def _progress_log_step(skill_name, success, duration_ms, error=None):
            step_counter[0] += 1
            # Dynamic total estimation
            # We assume at least 4 tasks (reader, cleaner, classifier, summarizer)
            # or we calculate from result if it were available.
            # Best way: we know 'done' is the last one.
            est_total = max(step_counter[0] + 1, 5) 
            pct = min(step_counter[0] / est_total, 0.98)
            label = STEP_LABELS.get(skill_name, f"⚙ {skill_name}…")
            bar.progress(pct, text=label)
            if skill_name == "done":
                bar.progress(1.0, text="✅ Complete!")

        agent._log_step = _progress_log_step  # type: ignore[method-assign]
        
        # We don't know the task count until planning happens inside agent.run()
        # So we update total dynamically when the first log arrives or by inspecting plan
        # For simplicity, we'll wait for the first log to estimate or just use a reasonable default
        # BUT we can actually get the plan from the agent if we wanted. 
        # Better: let _progress_log_step handle it.
        
        result = agent.run(file_path)
        _progress_log_step("done", True, 0)

        elapsed = time.monotonic() - start_ts

        with status_placeholder.container():
            if result.success:
                st.session_state[state_key] = result  # Cache on success
                st.success(
                    f"✅ Analysis complete in **{elapsed:.1f}s** — "
                    f"classified as **{result.doc_type.replace('_', ' ').title()}**"
                )
            else:
                st.error(f"⚠️ Pipeline finished with errors.")

        render_results(result, export_cfg=_cfg.export)
        return result

    finally:
        cleanup_temp_dir(tmp_dir)


# ── Main app ───────────────────────────────────────────────────────────────────

class DummyFile:
    def __init__(self, name, content):
        self.name = name
        self.content = content
        self.file_id = f"dummy_{name}"
    def getvalue(self):
        return self.content

def main() -> None:
    overrides = _render_sidebar(_cfg)
    _render_header()
    inputs = _render_upload()
    uploaded_files = inputs["files"]
    yt_url = inputs["youtube_url"]

    if not uploaded_files and not yt_url:
        st.markdown("""
        <div class="empty-state fade-in" style="padding: 4rem 1rem">
          <span class="icon">📂</span>
          <h3>Upload a document, audio file, or YouTube link to get started</h3>
          <p style="color:#475569">Supports PDF, Excel, CSV, MP3/MP4, and YouTube Transcripts</p>
        </div>
        """, unsafe_allow_html=True)
        return

    results = []

    # Process files
    for uf in uploaded_files:
        st.markdown(f"---\n### 🗂 `{uf.name}`")
        res = _run_pipeline(uf, overrides)
        if res and res.success:
            results.append(res)
            
    # Process YouTube URL
    if yt_url:
        st.markdown(f"---\n### 📺 YouTube Video")
        df = DummyFile("video.youtube", yt_url.encode("utf-8"))
        res = _run_pipeline(df, overrides)
        if res and res.success:
            results.append(res)

    # Multi-Document Compare logic
    if len(results) >= 2:
        st.markdown("---\n## ⚖️ Multi-Document Comparison")
        comp_key = f"compare_{results[0].file_name}_{results[1].file_name}"
        if comp_key not in st.session_state:
            with st.spinner("Generating document comparison..."):
                agent = _get_agent(_cfg.groq.model, _cfg.classification.questionnaire_threshold, _cfg.summarization.chunk_size, False, "")
                from core.models import SkillInput
                comp_out = agent.skills["compare_documents"].safe_execute(SkillInput(data={
                    "result_a": results[0],
                    "result_b": results[1]
                }))
                if comp_out.success:
                    st.session_state[comp_key] = comp_out.data
                else:
                    st.error("Comparison failed: " + str(comp_out.error))
        
        if comp_key in st.session_state:
            cdata = st.session_state[comp_key]
            st.markdown(cdata["llm_analysis"])
            if cdata["overlapping_questions"]:
                st.subheader("Overlapping Questions")
                for q in cdata["overlapping_questions"]:
                    st.info(f"**Doc A:** {q['doc_a_question']}\n\n**Doc B:** {q['doc_b_question']}\n\n*(Similarity: {q['similarity']})*")


if __name__ == "__main__":
    main()
