"""
DocumentAgent — the main orchestrator for the DocAgent system.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from agents.base_agent import BaseAgent
from agents.planner_agent import PlannerAgent
from agents.skill_executor import SkillExecutor
from core.models import ParsedDocument, SkillInput
from core.pipeline_result import PipelineResult
from skills.document_classifier_skill import DocumentClassifierSkill
from skills.excel_reader_skill import ExcelReaderSkill
from skills.pdf_reader_skill import PDFReaderSkill
from skills.structure_recognition_skill import StructureRecognitionSkill
from skills.question_extraction_skill import QuestionExtractionSkill
from skills.summarization_skill import SummarizationSkill
from skills.text_cleaner_skill import TextCleanerSkill
from skills.rag_skill import RagSkill
from skills.tts_skill import TTSSkill
from skills.audio_transcription_skill import AudioTranscriptionSkill
from skills.youtube_skill import YouTubeSkill
from skills.compare_documents_skill import CompareDocumentsSkill
from utils.logger import get_logger

logger = get_logger(__name__)

SUPPORTED_EXTENSIONS: Dict[str, str] = {
    ".pdf":  "pdf",
    ".xlsx": "excel",
    ".xls":  "excel",
    ".csv":  "excel",
    ".mp3":  "audio",
    ".mp4":  "audio",
    ".wav":  "audio",
    ".m4a":  "audio",
}


class DocumentAgent(BaseAgent):
    """
    Main document analysis orchestrator.
    Now utilizes dynamic Planner + Executor agent loops instead of hardcoded rules.
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
        
        self.planner = PlannerAgent(config={**cls_, **groq_skill_cfg})

        self.skills = {
            "pdf_reader": PDFReaderSkill(config=pdf),
            "excel_reader": ExcelReaderSkill(config=xls),
            "text_cleaner": TextCleanerSkill(config=summ),
            "document_classifier": DocumentClassifierSkill(config={**cls_, **groq_skill_cfg}),
            "structure_recognition": StructureRecognitionSkill(config=pdf),
            "summarization": SummarizationSkill(config={**summ, **groq_skill_cfg}),
            "question_extraction": QuestionExtractionSkill(config={**q_, **groq_skill_cfg}),
            "rag": RagSkill(config=groq_skill_cfg),
            "tts": TTSSkill(config={}),
            "audio_transcription": AudioTranscriptionSkill(config={}),
            "youtube": YouTubeSkill(config={}),
            "compare_documents": CompareDocumentsSkill(config=groq_skill_cfg)
        }
        
        self.executor = SkillExecutor(self.skills)

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

        # Build dynamic context dict
        ux_overrides = getattr(self, '_ux_overrides', {})
        context = {
            "file_path": str(file_path),
            "file_type": file_type,
            "doc_type": "normal_document",
            "domain": "General",
            "summary_length": ux_overrides.get("summary_length", "medium"),
            "technical_level": ux_overrides.get("technical_level", "intermediate"),
            "page_from": ux_overrides.get("page_from", 0),
            "page_to": ux_overrides.get("page_to", 0),
        }

        # ── Step 1: Dynamic LLM Planning ──────────────────────────────
        tasks = self.planner.plan(context)

        # ── Step 2: Skill Executor ────────────────────────────────────
        self.logger.info(f"Executing planned paths: {tasks}")
        results = self.executor.execute(tasks, context)
        
        # Compile logs directly from the dynamic results matrix
        for task in tasks:
            success = results.get(f"{task}_success", False)
            dur = results.get(f"{task}_ms", 0.0)
            err = results.get(f"{task}_error")
            self._log_step(task, success, dur, err)
            skill_timings[task] = dur
            if err:
                errors.append(err)

        # ── Step 4: Map back to strict UI schema ──────────────────────
        cls_data = results.get("document_classifier_data")
        struc_data = results.get("structure_recognition_data")
        summ_data = results.get("summarization_data")
        q_data = results.get("question_extraction_data")
        
        # Get parsed document from readers or cleaners
        parsed_doc = (
            results.get("text_cleaner_data") or
            results.get("audio_transcription_data") or
            results.get("youtube_data") or
            results.get("pdf_reader_data") or
            results.get("excel_reader_data")
        )
        
        if not parsed_doc:
            return self._error_result(file_path, "Parsing failed — no document extracted.")

        full_text = parsed_doc.full_text

        # Classification mapping
        doc_type     = cls_data.doc_type if cls_data else "normal_document"
        domain       = cls_data.domain if cls_data else "General"
        class_conf   = cls_data.confidence if cls_data else 0.0
        class_method = cls_data.method if cls_data else "fallback"

        # Structural mapping overrides
        if struc_data:
            parsed_doc = struc_data
            full_text = parsed_doc.full_text

        # Summarization mapping
        summary = summ_data.get("summary", "") if summ_data else ""
        summary_method = summ_data.get("method", "none") if summ_data else "none"

        # Questions mapping
        questions = q_data.get("questions", []) if q_data else []
        q_method  = q_data.get("method", "none") if q_data else "none"

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
