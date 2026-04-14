# 🤖 DocAgent

DocAgent is a high-performance, modular AI agent designed for intelligent document understanding. It goes beyond simple text extraction by utilizing an **Agent-Skill architecture** to classify, summarize, and extract actionable data from complex PDF, Excel, and CSV documents—now powered by **Groq Cloud** for near-instant responses.

![DocAgent Status](https://img.shields.io/badge/Powered%20By-Groq%20Cloud-f59e0b)
![Python 3.10+](https://img.shields.io/badge/Python-3.10+-06B6D4)
![License](https://img.shields.io/badge/License-MIT-green)

## 🌟 Key Features
*   **Intelligent Summarization**: Section-aware Map-Reduce pipeline that preserves document hierarchy and synthesizes large datasets.
*   **Dataset Mastery**: Optimized for large CSV/Excel files using **Smart Sampling (Head+Tail)** to ensure breadth across all data categories (e.g., summarizing AQI trends across multiple cities).
*   **Premium PDF Exports**: Styled, business-ready PDF reports with embedded markdown support, custom typography, and skill-timing tables.
*   **Question Extraction**: Hybrid Regex + LLM extraction for forms and questionnaires.
*   **Multi-Engine Parsing**: High-precision text extraction with automated Tesseract OCR fallback for scanned documents.
*   **Glassmorphic Web UI**: A stunning, modern interface built with Streamlit, featuring real-time pipeline tracking and detailed diagnostics.

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
Set your Groq API key as an environment variable (recommended) or in the config file:
```powershell
# Windows PowerShell
$env:GROQ_API_KEY="your_api_key_here"
```
Alternatively, edit `configs/default.yaml` and paste your key under the `groq` section.

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
*   **LLM Engine**: Groq Cloud (`llama-3.3-70b-versatile`)
*   **UI**: Streamlit with custom CSS3 Glassmorphism
*   **Parsers**: `pdfplumber`, `PyMuPDF`, `openpyxl`, `pandas`
*   **Reporting**: `ReportLab` (for styled PDF generation)
*   **OCR**: `pytesseract`

## 🛡️ License
Distributed under the MIT License. See `LICENSE` for more information.

---
*Built for Speed. Built for Accuracy. Powered by Groq.*
