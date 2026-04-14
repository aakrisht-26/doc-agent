# 🛠️ DocAgent Technology Stack

DocAgent is a production-quality PoC for document understanding, built with a modular Agent-Skill architecture. It uses **Groq Cloud** as its primary intelligence engine.

## 🤖 Core Intelligence
*   **LLM Provider**: [Groq Cloud](https://groq.com/)
*   **API Model**: `llama-3.3-70b-versatile` (Default) / `mixtral-8x7b-32768`
*   **Integration**: OpenAI-compatible Python SDK
*   **Capabilities**:
    *   **Domain-Aware Summarization**: Contextual personas (e.g., Senior Financial Analyst) dynamically injected based on automated domain detection.
    *   **Resilient API Mesh**: Multi-key round-robin rotation for `GROQ_API_KEYS` with automatic 429 backoff and recovery.
    *   **Smart Dataset Sampling**: High-density Head/Tail sampling for Excel/CSV to capture trends and outliers in large datasets.

## 🎨 User Interface
*   **Framework**: [Streamlit](https://streamlit.io/)
*   **Styling**: Vanilla CSS3 with Glassmorphism, Backdrop Blur, and custom purple gradients.
*   **Diagnostics**: Real-time pipeline status and skill-level timing breakdown.

## 📄 Document Processing
*   **Pre-processing Engine**: `OpenCV` + `NumPy` for Gaussian noise diffusion and **Adaptive Gaussian Thresholding** to normalize lighting in real-time.
*   **Geometric Correction**: Custom CV kernels for **Automatic Deskewing** (perspective straightening) on photo-PDFs.
*   **Extraction Core**: `pdfplumber` (v0.10+) with `layout=True` and Bbox-based table masking.
*   **OCR Support**: `pytesseract` optimized with high-DPI (400) rendering and Neural Engine (`--oem 3`).
*   **Validation**: Heuristic-based **Garbage Detection** to automatically trigger OCR on bad hidden text layers.

## 🏗️ Architecture & Framework
*   **Orchestration**: Custom `DocumentAgent`
*   **Capability Registry**: `SkillRegistry` for dynamic skill discovery.
*   **Config**: `PyYAML` & Dataclasses with Env var overrides.
*   **Logging**: `rich` logging for developer-friendly terminal output.
