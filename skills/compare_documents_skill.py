"""
CompareDocumentsSkill — performs LLM-driven comparison of two documents.

Compares summaries, questions, and domains to identify:
  - Common topics
  - Key differences
  - Overlapping questions
  - Recommendation on which document covers a topic better
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from core.models import SkillInput, SkillOutput
from skills.base_skill import BaseSkill
from utils.logger import get_logger

logger = get_logger(__name__)


class CompareDocumentsSkill(BaseSkill):
    """
    Compares two PipelineResult objects and produces a structured comparison.

    Expected inputs:
        result_a (PipelineResult): First document result
        result_b (PipelineResult): Second document result
    """

    name = "compare_documents"
    description = "Compare two documents and identify commonalities, differences, and overlapping questions."
    required_inputs = ["result_a", "result_b"]

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config)
        from utils.llm_client import LLMClient
        self._llm = LLMClient.from_config(self.config)

    def execute(self, inputs: SkillInput) -> SkillOutput:
        start = time.monotonic()
        result_a = inputs.data["result_a"]
        result_b = inputs.data["result_b"]

        name_a = getattr(result_a, "file_name", "Document A")
        name_b = getattr(result_b, "file_name", "Document B")
        sum_a  = getattr(result_a, "summary", "") or ""
        sum_b  = getattr(result_b, "summary", "") or ""
        qs_a   = getattr(result_a, "questions", []) or []
        qs_b   = getattr(result_b, "questions", []) or []
        dom_a  = getattr(result_a, "domain", "General")
        dom_b  = getattr(result_b, "domain", "General")

        # Find overlapping questions via simple text similarity
        overlapping_qs = self._find_overlapping_questions(qs_a, qs_b)

        # LLM comparison
        llm_analysis = self._llm_compare(name_a, name_b, sum_a, sum_b, dom_a, dom_b)

        return SkillOutput(
            success=True,
            data={
                "doc_a_name": name_a,
                "doc_b_name": name_b,
                "doc_a_domain": dom_a,
                "doc_b_domain": dom_b,
                "overlapping_questions": overlapping_qs,
                "questions_a": qs_a,
                "questions_b": qs_b,
                "llm_analysis": llm_analysis,
            },
            duration_ms=(time.monotonic() - start) * 1000,
        )

    def _find_overlapping_questions(self, qs_a: List[str], qs_b: List[str]) -> List[Dict]:
        """Simple keyword-overlap to find near-duplicate questions."""
        overlaps = []
        for qa in qs_a:
            words_a = set(qa.lower().split())
            for qb in qs_b:
                words_b = set(qb.lower().split())
                # Jaccard similarity
                inter = len(words_a & words_b)
                union = len(words_a | words_b)
                sim = inter / union if union > 0 else 0
                if sim > 0.4:  # 40% word overlap threshold
                    overlaps.append({"doc_a_question": qa, "doc_b_question": qb, "similarity": round(sim, 2)})
        return overlaps

    def _llm_compare(
        self, name_a: str, name_b: str,
        sum_a: str, sum_b: str,
        dom_a: str, dom_b: str
    ) -> str:
        if not self._llm.available:
            return "LLM unavailable — cannot generate comparison analysis."

        prompt = (
            f"You are a senior analyst comparing two documents.\n\n"
            f"**Document A**: {name_a} (Domain: {dom_a})\n"
            f"Summary:\n{sum_a[:2000]}\n\n"
            f"**Document B**: {name_b} (Domain: {dom_b})\n"
            f"Summary:\n{sum_b[:2000]}\n\n"
            "Produce a structured comparison report in Markdown with these sections:\n"
            "1. ## Common Topics\n"
            "2. ## Key Differences\n"
            "3. ## Unique Insights in Document A\n"
            "4. ## Unique Insights in Document B\n"
            "5. ## Overall Recommendation\n\n"
            "Be precise, professional, and concise. Use bullet points where appropriate."
        )

        result = self._llm.chat(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1500,
            temperature=0.1,
        )
        return result or "LLM comparison failed — please retry."
