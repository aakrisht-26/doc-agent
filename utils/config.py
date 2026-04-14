import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from dotenv import load_dotenv

# Load .env file if it exists
load_dotenv()

_CONFIG_PATH = Path(__file__).parent.parent / "configs" / "default.yaml"
_CONFIG_CACHE: Optional[AppConfig] = None


@dataclass
class GroqConfig:
    """Settings for the Groq Cloud API."""
    enabled: bool = True
    api_key: str = ""
    base_url: str = "https://api.groq.com/openai/v1"
    model: str = "llama-3.3-70b-versatile"
    timeout_seconds: int = 180
    temperature: float = 0.15


@dataclass
class PDFConfig:
    use_ocr_fallback: bool = True
    ocr_language: str = "eng"
    extract_tables: bool = True
    max_pages: int = 500


@dataclass
class ExcelConfig:
    max_sheets: int = 50
    include_formulas: bool = False
    max_rows_per_sheet: int = 10000


@dataclass
class SummarizationConfig:
    chunk_size: int = 4000
    chunk_overlap: int = 300
    max_summary_length: int = 3000
    extractive_sentences: int = 15


@dataclass
class ClassificationConfig:
    questionnaire_threshold: float = 0.35


@dataclass
class QuestionExtractionConfig:
    min_questions: int = 1
    dedup: bool = True
    max_questions: int = 200


@dataclass
class ExportConfig:
    pdf_font_size: int = 11
    pdf_margin_inch: float = 0.9


@dataclass
class AppConfig:
    name: str = "DocAgent"
    version: str = "1.0.0"
    max_file_size_mb: int = 50
    log_level: str = "INFO"
    log_file: str = "logs/docagent.log"

    groq: GroqConfig = field(default_factory=GroqConfig)
    pdf: PDFConfig = field(default_factory=PDFConfig)
    excel: ExcelConfig = field(default_factory=ExcelConfig)
    summarization: SummarizationConfig = field(default_factory=SummarizationConfig)
    classification: ClassificationConfig = field(default_factory=ClassificationConfig)
    question_extraction: QuestionExtractionConfig = field(default_factory=QuestionExtractionConfig)
    export: ExportConfig = field(default_factory=ExportConfig)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "groq": {
                "enabled": self.groq.enabled,
                "api_key": self.groq.api_key,
                "base_url": self.groq.base_url,
                "model": self.groq.model,
                "timeout_seconds": self.groq.timeout_seconds,
                "temperature": self.groq.temperature,
            },
            "pdf": {
                "use_ocr_fallback": self.pdf.use_ocr_fallback,
                "ocr_language": self.pdf.ocr_language,
                "extract_tables": self.pdf.extract_tables,
                "max_pages": self.pdf.max_pages,
            },
            "excel": {
                "max_sheets": self.excel.max_sheets,
                "include_formulas": self.excel.include_formulas,
                "max_rows_per_sheet": self.excel.max_rows_per_sheet,
            },
            "summarization": {
                "chunk_size": self.summarization.chunk_size,
                "chunk_overlap": self.summarization.chunk_overlap,
                "max_summary_length": self.summarization.max_summary_length,
                "extractive_sentences": self.summarization.extractive_sentences,
            },
            "classification": {
                "questionnaire_threshold": self.classification.questionnaire_threshold,
            },
            "question_extraction": {
                "dedup": self.question_extraction.dedup,
                "max_questions": self.question_extraction.max_questions,
            },
            "export": {
                "pdf_font_size": self.export.pdf_font_size,
                "pdf_margin_inch": self.export.pdf_margin_inch,
            },
        }


def load_config(config_path: Optional[Path] = None) -> AppConfig:
    global _CONFIG_CACHE
    if _CONFIG_CACHE is not None:
        return _CONFIG_CACHE

    path = config_path or _CONFIG_PATH
    raw: Dict[str, Any] = {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
    except FileNotFoundError:
        pass

    app_r = raw.get("app", {})
    gro_r = raw.get("groq", {})
    pdf_r = raw.get("pdf", {})
    xls_r = raw.get("excel", {})
    sum_r = raw.get("summarization", {})
    cls_r = raw.get("classification", {})
    q_r   = raw.get("question_extraction", {})
    exp_r = raw.get("export", {})

    cfg = AppConfig(
        name            = app_r.get("name", "DocAgent"),
        version         = app_r.get("version", "1.0.0"),
        max_file_size_mb= int(os.getenv("DOCAGENT_MAX_FILE_MB", app_r.get("max_file_size_mb", 50))),
        log_level       = os.getenv("DOCAGENT_LOG_LEVEL", app_r.get("log_level", "INFO")),
        log_file        = os.getenv("DOCAGENT_LOG_FILE", app_r.get("log_file", "logs/docagent.log")),
        groq=GroqConfig(
            enabled        = _env_bool("DOCAGENT_GROQ_ENABLED", gro_r.get("enabled", True)),
            api_key        = os.getenv("GROQ_API_KEY", gro_r.get("api_key", "")),
            base_url       = os.getenv("DOCAGENT_GROQ_URL", gro_r.get("base_url", "https://api.groq.com/openai/v1")),
            model          = os.getenv("DOCAGENT_GROQ_MODEL", gro_r.get("model", "llama-3.3-70b-versatile")),
            timeout_seconds= int(os.getenv("DOCAGENT_GROQ_TIMEOUT", gro_r.get("timeout_seconds", 180))),
            temperature    = float(gro_r.get("temperature", 0.15)),
        ),
        pdf  = PDFConfig(**{k: v for k, v in pdf_r.items() if k in PDFConfig.__dataclass_fields__}) if pdf_r else PDFConfig(),
        excel= ExcelConfig(**{k: v for k, v in xls_r.items() if k in ExcelConfig.__dataclass_fields__}) if xls_r else ExcelConfig(),
        summarization        = SummarizationConfig(**{k: v for k, v in sum_r.items() if k in SummarizationConfig.__dataclass_fields__}) if sum_r else SummarizationConfig(),
        classification       = ClassificationConfig(**{k: v for k, v in cls_r.items() if k in ClassificationConfig.__dataclass_fields__}) if cls_r else ClassificationConfig(),
        question_extraction  = QuestionExtractionConfig(**{k: v for k, v in q_r.items() if k in QuestionExtractionConfig.__dataclass_fields__}) if q_r else QuestionExtractionConfig(),
        export               = ExportConfig(**{k: v for k, v in exp_r.items() if k in ExportConfig.__dataclass_fields__}) if exp_r else ExportConfig(),
    )

    _CONFIG_CACHE = cfg
    return cfg


def reset_config_cache() -> None:
    global _CONFIG_CACHE
    _CONFIG_CACHE = None


def _env_bool(key: str, default: bool) -> bool:
    val = os.getenv(key)
    if val is None:
        return default
    return val.strip().lower() in ("true", "1", "yes")
