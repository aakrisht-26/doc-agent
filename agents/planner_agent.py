"""
PlannerAgent — Dynamically decides which execution tasks to run on the document.
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

from utils.llm_client import LLMClient
from utils.logger import get_logger

logger = get_logger(__name__)


class PlannerAgent:
    """
    Evaluates document text and returns a list of required task names.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        # Support LLM initialization using standard groq_skill_cfg structure passed from DocumentAgent
        self._llm = LLMClient.from_config(self.config)
        self.logger = logger

    def plan(self, context: dict) -> List[str]:
        """
        Determines which skills to run based on the context dictionary.
        """
        available_skills = [
            "document_classifier", "structure_recognition", "summarization", 
            "question_extraction", "rag", "tts", "audio_transcription", "youtube",
            "compare_documents"
        ]

        if not self._llm.available:
            self.logger.warning("PlannerAgent requires a configured LLM. Defaulting to standard schedule.")
            return ["document_classifier", "summarization"]

        start = time.monotonic()
        self.logger.info("PlannerAgent evaluating document context...")
        # Determine form, format context
        document_text = context.get("full_text", "")
        file_type = context.get("file_type", "unknown")
        
        # Determine readers based on file type
        base_tasks = []
        if file_type == "pdf":
            base_tasks = ["pdf_reader", "text_cleaner"]
        elif file_type == "excel":
            base_tasks = ["excel_reader", "text_cleaner"]
        elif file_type == "audio":
            return ["audio_transcription", "document_classifier", "summarization"]
        elif file_type == "youtube":
            return ["youtube", "document_classifier", "summarization"]
        elif file_type == "compare":
            return ["compare_documents"]
            
        # We don't have full_text yet in the new architecture because parsing hasn't happened.
        # We will use heuristics on the file_name or standard LLM generic tasks.
        file_name = context.get("file_path", "").lower()
        
        prompt = f"""
You are the Orchestration Planner for a Document Analysis pipeline.
Your job is to determine which tasks need to be executed for a file named '{file_name}'.

AVAILABLE TASKS:
{', '.join(available_skills)}

RULES:
1. Ignore readers (`pdf_reader`, `excel_reader`), we handle those automatically. 
2. 'document_classifier' should ALWAYS be included.
3. 'summarization' should ALWAYS be included unless the file is explicitly a very short form or metadata-only file.
4. If the filename implies a form, questionnaire, application, or survey, include 'question_extraction'.
5. If the filename implies a highly analytical, scientific, or messy table-like data, include 'structure_recognition'.
6. Return ONLY a valid JSON array of strings corresponding to the exact task names chosen. Absolutely NO other text or markdown fences.

INPUT TYPE: {file_type}
"""
        try:
            response = self._llm.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=200
            )

            if response:
                text = response.replace("```json", "").replace("```", "").strip()
                try:
                    tasks = json.loads(text)
                except json.JSONDecodeError:
                    self.logger.warning(f"Planner returned invalid JSON: {text}")
                    return base_tasks + ["document_classifier", "summarization"]

                if isinstance(tasks, list):
                    # Filter only valid skills
                    available_skills += ["pdf_reader", "excel_reader", "text_cleaner"]
                    valid_tasks = base_tasks + [t for t in tasks if t in available_skills]
                    
                    # Safety net: ensure baseline tasks if LLM returned empty or sparse list
                    if "document_classifier" not in valid_tasks:
                        valid_tasks.append("document_classifier")
                    if "summarization" not in valid_tasks and file_type in ("pdf", "excel"):
                        valid_tasks.append("summarization")

                    self.logger.info(f"[Planner] Tasks decided: {valid_tasks} (took {(time.monotonic() - start):.2f}s)")
                    return valid_tasks

        except Exception as e:
            self.logger.error(f"PlannerAgent failed to generate schedule: {e}")

        # Fallback
        self.logger.warning("[Planner] Using fallback schedule.")
        return base_tasks + ["document_classifier", "summarization", "question_extraction"]
