# 🤖 DocAgent

DocAgent is a high-performance, modular AI agent designed for intelligent document understanding. It goes beyond simple text extraction by utilizing an **Agent-Skill architecture** to classify, summarize, and extract actionable data from complex PDF, Excel, and CSV documents—now powered by **Groq Cloud** for near-instant responses.

![DocAgent Status](https://img.shields.io/badge/Powered%20By-Groq%20Cloud-f59e0b)
![Python 3.10+](https://img.shields.io/badge/Python-3.10+-06B6D4)
![License](https://img.shields.io/badge/License-MIT-green)

## 🌟 Key Features
*   **Domain-Aware Persona Analysis**: Automatically detects document domains (Technical, Financial, Legal, Educational) to adopt specialized analyst personas for high-fidelity summaries.
*   **Multi-Key API Resilience**: Dynamic round-robin rotation for Groq API keys with automatic Rate Limit (429) recovery and retry logic.
*   **Advanced Adaptive OCR**: An OpenCV-driven pipeline that uses **Gaussian Equalization** and **Adaptive Thresholding** to extract text from photos with poor lighting, shadows, or glare.
*   **Structural Table Masking**: Eliminates duplicated or garbled text by physically masking detected tables during raw extraction, ensuring only formatted [TABLE] blocks are processed.
*   **Intelligent Field Extraction**: Context-aware questionnaire detection that dynamically distinguishes between exam questions, form fields, and table headers.
*   **Glassmorphic Web UI**: A stunning, modern interface built with Streamlit, featuring real-time pipeline tracking, "Force OCR" overrides, and detailed diagnostics.

## 🚀 Getting Started

### 1. Prerequisites
*   **Python 3.10+**
*   **Groq API Key**: Obtain one from [Groq Cloud Console](https://console.groq.com/).
*   **Tesseract OCR** (Optional): For processing scanned images/PDFs.

### 2. Installation
```bash
# Clone the repository
git clone https://github.com/your-username/DocAgent.git
cd DocAgent

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration
Set your Groq API keys in a `.env` file in the root directory. You can provide multiple keys for automatic rotation:
```bash
# .env
GROQ_API_KEYS="gsk_key1,gsk_key2,gsk_key3"
```
DocAgent will automatically rotate through these keys if rate limits are hit during large document processing.

### 4. Run the App
```bash
streamlit run ui/app.py
```

## 🏗️ Architecture
DocAgent is built on a decoupled, extensible architecture:
*   **Agents**: Orchestrate the workflow (e.g., `DocumentAgent`).
*   **Skills**: Atomic capabilities (e.g., `PDFReader`, `Summarization`, `QuestionExtraction`).
*   **Unified Client**: A centralized `LLMClient` that handles OpenAI-compatible requests to Groq Cloud with graceful extractive fallbacks.

## 📊 Technical Stack
*   **LLM Engine**: Groq Cloud (Llama 3.3 70B Versatile)
*   **UI**: Streamlit with custom CSS3 Glassmorphism
*   **Computer Vision**: `OpenCV` (for adaptive lighting normalization)
*   **Parsers**: `pdfplumber` (v0.10+), `PyMuPDF`, `openpyxl`, `pandas`
*   **OCR**: `pytesseract` + `numpy` (High-DPI 400 scan mode)
*   **Reporting**: `ReportLab` (Dynamic Markdown reporting)

## 🛡️ License
Distributed under the MIT License. See `LICENSE` for more information.

---
*Built for Speed. Built for Accuracy. Powered by Groq.*
