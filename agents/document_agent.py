"""
DocumentAgent — the main orchestrator for the DocAgent system.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from agents.base_agent import BaseAgent
from core.models import ParsedDocument, SkillInput
from core.pipeline_result import PipelineResult
from skills.document_classifier_skill import DocumentClassifierSkill
from skills.excel_reader_skill import ExcelReaderSkill
from skills.pdf_reader_skill import PDFReaderSkill
from skills.structure_recognition_skill import StructureRecognitionSkill
from skills.question_extraction_skill import QuestionExtractionSkill
from skills.summarization_skill import SummarizationSkill
from skills.text_cleaner_skill import TextCleanerSkill
from utils.logger import get_logger

logger = get_logger(__name__)

SUPPORTED_EXTENSIONS: Dict[str, str] = {
    ".pdf":  "pdf",
    ".xlsx": "excel",
    ".xls":  "excel",
    ".csv":  "excel",
}


class DocumentAgent(BaseAgent):
    """
    Main document analysis orchestrator.
    """

    name = "document_agent"
    description = "Orchestrates the full document analysis pipeline."

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config)
        gro  = self.config.get("groq", {})
        pdf  = self.config.get("pdf", {})
        xls  = self.config.get("excel", {})
        summ = self.config.get("summarization", {})
        cls_ = self.config.get("classification", {})
        q_   = self.config.get("question_extraction", {})

        # Groq config block for skills
        groq_skill_cfg: Dict[str, Any] = {
            "groq": {
                "enabled":         gro.get("enabled", True),
                "api_keys":        gro.get("api_keys", ""),
                "api_key":         gro.get("api_key", ""),
                "base_url":        gro.get("base_url", "https://api.groq.com/openai/v1"),
                "model":           gro.get("model", "llama-3.3-70b-versatile"),
                "timeout_seconds": gro.get("timeout_seconds", 180),
                "temperature":     gro.get("temperature", 0.15),
            }
        }

        self._pdf_reader  = PDFReaderSkill(config=pdf)
        self._xls_reader  = ExcelReaderSkill(config=xls)
        self._cleaner     = TextCleanerSkill(config=summ)
        self._classifier  = DocumentClassifierSkill(config={**cls_, **groq_skill_cfg})
        self._struct_rec  = StructureRecognitionSkill(config=pdf)
        self._summarizer  = SummarizationSkill(config={**summ, **groq_skill_cfg})
        self._q_extractor = QuestionExtractionSkill(config={**q_, **groq_skill_cfg})

    # ── Main pipeline ─────────────────────────────────────────────────

    def run(self, file_path: Path) -> PipelineResult:
        pipeline_start = time.monotonic()
        file_path = Path(file_path)

        errors:        List[str] = []
        warnings:      List[str] = []
        skill_timings: Dict[str, float] = {}

        self.logger.info(f"═══ Pipeline START: {file_path.name} ═══")

        # ── Step 0: Validate ──────────────────────────────────────────
        ext = file_path.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            return self._error_result(file_path, f"Unsupported file type")
        file_type = SUPPORTED_EXTENSIONS[ext]

        # ── Step 1: Parse ─────────────────────────────────────────────
        reader = self._pdf_reader if file_type == "pdf" else self._xls_reader
        parse_out = reader.safe_execute(SkillInput(data={"file_path": str(file_path)}))
        skill_timings["parse"] = parse_out.duration_ms
        self._log_step("parse", parse_out.success, parse_out.duration_ms, parse_out.error)

        if not parse_out.success:
            return self._error_result(file_path, f"Parsing failed")

        parsed_doc: ParsedDocument = parse_out.data
        warnings.extend(parse_out.warnings)

        if parsed_doc.is_empty:
            return self._error_result(file_path, "Document empty")

        # ── Step 2: Clean ─────────────────────────────────────────────
        clean_out = self._cleaner.safe_execute(SkillInput(data={"parsed_document": parsed_doc}))
        skill_timings["clean"] = clean_out.duration_ms
        self._log_step("clean", clean_out.success, clean_out.duration_ms, clean_out.error)

        if clean_out.success and clean_out.data:
            parsed_doc = clean_out.data
        full_text = parsed_doc.full_text

        # ── Step 3: Classify ──────────────────────────────────────────
        classify_out = self._classifier.safe_execute(SkillInput(data={"full_text": full_text}))
        skill_timings["classify"] = classify_out.duration_ms
        self._log_step("classify", classify_out.success, classify_out.duration_ms, classify_out.error)

        class_result = classify_out.data if classify_out.success else None
        doc_type     = class_result.doc_type if class_result else "normal_document"
        domain       = class_result.domain if class_result else "General"
        class_conf   = class_result.confidence if class_result else 0.0
        class_method = class_result.method if class_result else "fallback"

        # ── Step 3.5: Specialized Structure Recognition (Tables) ──────
        struct_out = self._struct_rec.safe_execute(SkillInput(data={
            "parsed_document": parsed_doc,
            "file_path": str(file_path),
            "domain": domain
        }))
        skill_timings["structure_recognition"] = struct_out.duration_ms
        self._log_step("structure_recognition", struct_out.success, struct_out.duration_ms, struct_out.error)

        if struct_out.success and struct_out.data:
            parsed_doc = struct_out.data
            full_text = parsed_doc.full_text  # Update full_text with the new high-fidelity tables


        # ── Step 4: Summarize ─────────────────────────────────────────
        summ_out = self._summarizer.safe_execute(SkillInput(data={"full_text": full_text, "doc_type": doc_type, "domain": domain}))
        skill_timings["summarize"] = summ_out.duration_ms
        self._log_step("summarize", summ_out.success, summ_out.duration_ms, summ_out.error)

        summary = summ_out.data.get("summary", "") if (summ_out.success and summ_out.data) else ""
        summary_method = summ_out.data.get("method", "none") if (summ_out.success and summ_out.data) else "none"

        # ── Step 5: Extract Questions ─────────────────────────────────
        q_out = self._q_extractor.safe_execute(SkillInput(data={"full_text": full_text, "doc_type": doc_type, "domain": domain}))
        skill_timings["extract_questions"] = q_out.duration_ms
        self._log_step("extract_questions", q_out.success, q_out.duration_ms, q_out.error)

        questions = q_out.data.get("questions", []) if (q_out.success and q_out.data) else []
        q_method  = q_out.data.get("method", "none") if (q_out.success and q_out.data) else "none"

        # ── Step 6: Assemble ──────────────────────────────────────────
        total_ms = (time.monotonic() - pipeline_start) * 1000
        return PipelineResult(
            file_name=file_path.name,
            file_type=file_type,
            doc_type=doc_type,
            domain=domain,
            classification_confidence=class_conf,
            classification_method=class_method,
            summary=summary,
            summary_method=summary_method,
            questions=questions,
            question_extraction_method=q_method,
            raw_text=full_text,
            word_count=parsed_doc.word_count,
            page_count=parsed_doc.page_count,
            metadata=parsed_doc.metadata,
            errors=errors,
            warnings=warnings,
            processing_time_ms=total_ms,
            skill_timings=skill_timings,
            success=len(errors) == 0,
        )

    @staticmethod
    def _error_result(file_path: Path, error_msg: str) -> PipelineResult:
        return PipelineResult(
            file_name=file_path.name,
            file_type="unknown",
            doc_type="unknown",
            domain="General",
            classification_confidence=0.0,
            classification_method="none",
            summary="",
            summary_method="none",
            questions=[],
            question_extraction_method="none",
            raw_text="",
            word_count=0,
            page_count=0,
            metadata={},
            errors=[error_msg],
            success=False,
        )
