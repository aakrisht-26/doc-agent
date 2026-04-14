# 🛠️ DocAgent Technology Stack

DocAgent is a production-quality PoC for document understanding, built with a modular Agent-Skill architecture. It uses **Groq Cloud** as its primary intelligence engine.

## 🤖 Core Intelligence
*   **LLM Provider**: [Groq Cloud](https://groq.com/)
*   **API Model**: `llama-3.3-70b-versatile` (Default) / `mixtral-8x7b-32768`
*   **Integration**: OpenAI-compatible Python SDK
*   **Capabilities**:
    *   **Section-aware Summarization**: (Map-Reduce) synthesis that preserves context.
    *   **Smart Dataset Sampling**: Head/Tail sampling for CSV/Excel to ensure data breadth across large files.
    *   **Intelligent Classification**: Hybrid heuristic + LLM logic.

## 🎨 User Interface
*   **Framework**: [Streamlit](https://streamlit.io/)
*   **Styling**: Vanilla CSS3 with Glassmorphism, Backdrop Blur, and custom purple gradients.
*   **Diagnostics**: Real-time pipeline status and skill-level timing breakdown.

## 📄 Document Processing
*   **PDF Extraction**: `pdfplumber` (precision) & `PyMuPDF` (speed)
*   **Excel/CSV**: `openpyxl` & `pandas`
*   **Reporting**: `ReportLab` for generating styled business reports with Markdown support.
*   **OCR**: `pytesseract` (Tesseract)

## 🏗️ Architecture & Framework
*   **Orchestration**: Custom `DocumentAgent`
*   **Capability Registry**: `SkillRegistry` for dynamic skill discovery.
*   **Config**: `PyYAML` & Dataclasses with Env var overrides.
*   **Logging**: `rich` logging for developer-friendly terminal output.
