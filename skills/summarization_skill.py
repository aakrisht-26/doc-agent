"""
SummarizationSkill -- generates detailed, structured summaries of document text.

Key improvements:
  - Uses unified LLMClient (Grok API first, Ollama fallback)
  - Section-aware chunking: detects headings and keeps sections intact
  - Structured output: fixed 4-section schema in every LLM prompt
  - Map phase extracts bullet-point facts; Reduce phase synthesises them
  - Extractive fallback: heading-boosted + position-weighted sentence scoring
"""

from __future__ import annotations

import re
import time
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from core.models import SkillInput, SkillOutput
from skills.base_skill import BaseSkill
from utils.logger import get_logger

logger = get_logger(__name__)

# ── Stopwords ─────────────────────────────────────────────────────────────────
_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to",
    "for", "of", "with", "by", "from", "is", "was", "are", "were",
    "be", "been", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "that", "this", "it",
    "he", "she", "they", "we", "you", "i", "not", "no", "can", "as",
    "so", "if", "then", "also", "its", "their", "our", "your", "my",
    "his", "her", "than", "more", "about", "which", "who", "what",
    "when", "where", "how", "all", "any", "both", "each", "few",
    "into", "through", "during", "before", "after", "above", "below",
    "just", "because", "while", "there", "very", "too", "only", "up",
    "out", "over", "such", "being", "between", "here", "them", "these",
    "those", "per", "etc", "eg", "ie", "said", "says", "one", "two",
})

# ── Heading detection patterns ────────────────────────────────────────────────
_HEADING_PATTERNS: List[re.Pattern] = [
    re.compile(r"^#{1,4}\s+(.+)$",              re.MULTILINE),
    re.compile(r"^(\d+(?:\.\d+)*)\s{1,4}([A-Z][^\n]{2,70})$", re.MULTILINE),
    re.compile(r"^([A-Z][A-Z0-9\s\-:]{4,60}[A-Z0-9])$",       re.MULTILINE),
    re.compile(r"^([A-Z][a-zA-Z0-9\s\-]{3,60})\s*\n[=\-]{4,}", re.MULTILINE),
    re.compile(r"^(?:Section|Chapter|Part|Article)\s+\d+[:\.\s]+(.+)$",
               re.IGNORECASE | re.MULTILINE),
]


class SummarizationSkill(BaseSkill):
    """
    Produces detailed, structured summaries of document text.

    Config keys:
        chunk_size           (int)   : Chars per section chunk (default: 4000)
        chunk_overlap        (int)   : Overlap between chunks   (default: 300)
        max_summary_length   (int)   : max_tokens for LLM      (default: 3000)
        extractive_sentences (int)   : Fallback sentence count  (default: 15)
        -- LLM provider settings are read via LLMClient.from_config() --
    """

    name = "summarization"
    description = "Generates detailed, section-aware summaries via LLM or extractive fallback."
    required_inputs = ["full_text"]

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config)
        self._chunk_size      = self.get_config("chunk_size", 4000)
        self._chunk_overlap   = self.get_config("chunk_overlap", 300)
        self._max_tokens      = self.get_config("max_summary_length", 3000)
        self._extractive_n    = self.get_config("extractive_sentences", 15)

        # Build unified LLM client (Grok > Ollama > None)
        from utils.llm_client import LLMClient
        self._llm = LLMClient.from_config(self.config)

    # ── Entry point ───────────────────────────────────────────────────────────

    def execute(self, inputs: SkillInput) -> SkillOutput:
        start    = time.monotonic()
        full_text: str = inputs.data["full_text"]
        doc_type: str  = inputs.data.get("doc_type", "normal_document")
        domain: str    = inputs.data.get("domain", "General")
        
        # UX Controls
        length: str  = inputs.data.get("summary_length", "medium")
        level: str   = inputs.data.get("technical_level", "intermediate")
        page_from: int = int(inputs.data.get("page_from", 0))
        page_to: int   = int(inputs.data.get("page_to", 0))
        
        # Page-range filtering: filter chunks before summarizing
        parsed_doc = inputs.data.get("parsed_document")
        if parsed_doc and (page_from > 0 or page_to > 0):
            selected_chunks = [
                c for c in parsed_doc.chunks
                if (page_from == 0 or int(c.page_or_sheet) >= page_from)
                and (page_to == 0 or int(c.page_or_sheet) <= page_to)
            ]
            if selected_chunks:
                full_text = "\n\n".join(c.text for c in selected_chunks)
                self.logger.info(f"Page range filter: pages {page_from}-{page_to}, {len(selected_chunks)} chunks")

        if not full_text.strip():
            return SkillOutput(
                success=False, data=None,
                error="Cannot summarize an empty document.",
                duration_ms=(time.monotonic() - start) * 1000,
            )

        sections     = self._detect_sections(full_text)
        section_names = [s[0] for s in sections]

        if self._llm.available:
            result = self._summarize_llm(full_text, doc_type, domain, sections, length, level, parsed_doc)
            if result:
                summary, method = result
                self.logger.info(
                    f"Summary via {self._llm.provider_label} ({method}): {len(summary)} chars"
                )
                return SkillOutput(
                    success=True,
                    data={"summary": summary, "method": method,
                          "sections_detected": section_names,
                          "provider": self._llm.provider_label},
                    duration_ms=(time.monotonic() - start) * 1000,
                )
            self.logger.warning(
                f"LLM ({self._llm.provider_label}) failed — falling back to extractive."
            )
            reason = f"LLM provider ({self._llm.provider_label}) failed or timed out"
        else:
            reason = "no LLM provider configured"

        summary = self._extractive_summarize(full_text, sections, reason=reason)

        return SkillOutput(
            success=True,
            data={"summary": summary, "method": "extractive",
                  "sections_detected": section_names, "provider": "none"},
            duration_ms=(time.monotonic() - start) * 1000,
        )

    # ── Section detection ─────────────────────────────────────────────────────

    def _detect_sections(self, text: str) -> List[Tuple[str, str]]:
        heading_positions: List[Tuple[int, str]] = []
        for pattern in _HEADING_PATTERNS:
            for m in pattern.finditer(text):
                heading = (m.group(1) if m.lastindex and m.lastindex >= 1
                           else m.group(0)).strip()
                if len(heading) > 3:
                    heading_positions.append((m.start(), heading))

        if not heading_positions:
            return []

        heading_positions.sort(key=lambda x: x[0])
        deduped: List[Tuple[int, str]] = []
        last_end = -1
        for pos, heading in heading_positions:
            if pos > last_end:
                deduped.append((pos, heading))
                last_end = pos + len(heading)

        sections: List[Tuple[str, str]] = []
        for i, (pos, heading) in enumerate(deduped):
            next_pos = deduped[i + 1][0] if i + 1 < len(deduped) else len(text)
            body = text[pos:next_pos].strip()
            body_lines = body.split("\n")
            body = "\n".join(body_lines[1:]).strip() if len(body_lines) > 1 else ""
            if body:
                sections.append((heading, body))

        self.logger.debug(f"Detected {len(sections)} sections")
        return sections

    # ── LLM summarization ─────────────────────────────────────────────────────

    def _summarize_llm(
        self,
        full_text: str,
        doc_type: str,
        domain: str,
        sections: List[Tuple[str, str]],
        length: str = "medium",
        level: str = "intermediate",
        parsed_doc=None,
    ) -> Optional[Tuple[str, str]]:
        chunks = self._section_aware_chunks(full_text, sections)
        self.logger.info(f"Summarizing {len(chunks)} chunk(s) — {self._llm.provider_label} (Domain: {domain})")
        
        # Build page-tagged chunks for source attribution
        if parsed_doc and hasattr(parsed_doc, 'chunks') and len(parsed_doc.chunks) == len(chunks):
            page_tags = [f"[Page {c.page_or_sheet}]" for c in parsed_doc.chunks]
        else:
            page_tags = [f"[Chunk {i+1}]" for i in range(len(chunks))]

        if len(chunks) == 1:
            result = self._call_single(chunks[0], doc_type, domain, sections, length, level)
            return (result, f"llm_single_{self._llm.provider}") if result else None

        chunk_summaries: List[str] = []
        for idx, (chunk, tag) in enumerate(zip(chunks, page_tags), 1):
            self.logger.debug(f"  Map chunk {idx}/{len(chunks)}")
            s = self._call_map(chunk, domain, tag)
            if s:
                chunk_summaries.append(s)

        if not chunk_summaries:
            return None

        combined = "\n\n---\n\n".join(chunk_summaries)
        final    = self._call_reduce(combined, doc_type, domain, sections, length, level)
        return (final, f"llm_map_reduce_{self._llm.provider}") if final else None

    def _call_single(
        self,
        text: str,
        doc_type: str,
        domain: str,
        sections: List[Tuple[str, str]],
        length: str = "medium",
        level: str = "intermediate",
    ) -> Optional[str]:
        doc_label    = doc_type.replace("_", " ")
        section_hint = ""
        if sections:
            names = ", ".join(f'"{s[0]}"' for s in sections[:8])
            section_hint = f"\nDetected sections: {names}"
        
        length_map = {"short": "2-3 paragraphs", "medium": "4-6 paragraphs", "detailed": "8+ paragraphs with all details"}
        level_map = {
            "beginner": "Use plain English. Avoid jargon. Explain technical terms when used.",
            "intermediate": "Use standard domain terminology. Assume moderate familiarity.",
            "expert": "Use precise domain terminology. Include technical depth and nuanced analysis."
        }
        length_instr = length_map.get(length, length_map["medium"])
        level_instr = level_map.get(level, level_map["intermediate"])

        return self._llm.chat(
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"You are a Senior {domain} Analyst. "
                        "You produce production-grade, highly professional, and domain-specific analysis reports. "
                        "Never invent information — only use what is provided."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Analyze the following {doc_label} from the perspective of an expert {domain} analyst.\n"
                        f"{section_hint}\n\n"
                        f"LENGTH: {length_instr}\n"
                        f"AUDIENCE: {level_instr}\n\n"
                        "IMPORTANT: If this is a dataset (CSV/Excel), ensure you summarize info across ALL categories or groups found "
                        "(e.g., all cities, all departments), identifying overall trends and outliers.\n\n"
                        f"Structure your report dynamically based on industry standards for {domain} analysis. "
                        "Use Markdown headings. At the end of each key claim or data point, add a source reference like [Source: Page N] where N is the approximate page number.\n\n"
                        f"Document:\n\n{text}\n\n"
                        "Analysis Report:"
                    ),
                },
            ],
            max_tokens=self._max_tokens,
        )

    def _call_map(self, chunk: str, domain: str, page_tag: str = "") -> Optional[str]:
        tag_note = f" (from {page_tag})" if page_tag else ""
        return self._llm.chat(
            messages=[{
                "role": "user",
                "content": (
                    f"You are an expert {domain} Analyst extracting data from a document section{tag_note}.\n"
                    "Extract the most critical metrics, data points, trends, and facts relevant to your domain.\n"
                    "Include: specific facts/figures/dates, important entities, outliers, or decisions.\n"
                    f"Be specific — do not paraphrase vaguely. Preserve numbers and proper nouns.\n"
                    f"After each key fact, append a source reference in parentheses like ({page_tag}) so the reader knows the origin.\n\n"
                    f"Section:\n{chunk}\n\n"
                    "Extracted Facts (bullet list with source references):"
                ),
            }],
            temperature=0.1,
            max_tokens=800,
        )

    def _call_reduce(
        self,
        combined_bullets: str,
        doc_type: str,
        domain: str,
        sections: List[Tuple[str, str]],
        length: str = "medium",
        level: str = "intermediate",
    ) -> Optional[str]:
        doc_label = doc_type.replace("_", " ")
        length_map = {"short": "2-3 paragraphs", "medium": "4-6 paragraphs", "detailed": "8+ paragraphs with all details"}
        level_map = {
            "beginner": "Use plain English. Avoid jargon. Explain technical terms when used.",
            "intermediate": "Use standard domain terminology. Assume moderate familiarity.",
            "expert": "Use precise terminology. Include technical depth and nuanced analysis."
        }
        length_instr = length_map.get(length, length_map["medium"])
        level_instr = level_map.get(level, level_map["intermediate"])
        return self._llm.chat(
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"You are a Senior {domain} Analyst. "
                        "Synthesize extracted data into a production-grade, domain-specific analysis report. "
                        "Preserve all source references in the format (Page N) or [Source: Page N] from the extracted facts. "
                        "Never invent information — only use what is provided."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Below are key facts extracted from sections of a {doc_label}.\n"
                        f"LENGTH: {length_instr}\n"
                        f"AUDIENCE: {level_instr}\n\n"
                        f"Synthesize them into a highly professional {domain} analysis report. "
                        "Ensure you cover the BREADTH of information — if different chunks focused on different groups "
                        "(e.g., cities, dates, entities), include all of them to highlight overall trends and comparisons.\n\n"
                        f"Structure the report dynamically based on industry standards for {domain} analysis. "
                        "Use clean Markdown headings. Where you refer to facts from specific pages, include source tags like [Source: Page N].\n\n"
                        f"Extracted facts:\n\n{combined_bullets}\n\n"
                        "Final Analysis Report:"
                    ),
                },
            ],
            max_tokens=self._max_tokens,
        )

    # ── Section-aware chunking ─────────────────────────────────────────────────

    def _section_aware_chunks(
        self,
        full_text: str,
        sections: List[Tuple[str, str]],
    ) -> List[str]:
        if not sections:
            return self._char_chunks(full_text)

        chunks: List[str] = []
        current = ""
        for heading, body in sections:
            section_text = f"## {heading}\n\n{body}"
            if len(current) + len(section_text) > self._chunk_size and current:
                chunks.append(current.strip())
                current = section_text
            else:
                current = (current + "\n\n" + section_text).strip()
        if current:
            chunks.append(current.strip())

        result: List[str] = []
        for chunk in chunks:
            if len(chunk) > self._chunk_size * 1.5:
                result.extend(self._char_chunks(chunk))
            else:
                result.append(chunk)
        return result if result else [full_text]

    def _char_chunks(self, text: str) -> List[str]:
        if len(text) <= self._chunk_size:
            return [text]

        chunks: List[str] = []
        start, text_len = 0, len(text)
        BREAKS = ["\n\n", ".\n", ". ", "? ", "! ", "; "]

        while start < text_len:
            end = min(start + self._chunk_size, text_len)
            if end >= text_len:
                chunks.append(text[start:])
                break
            break_at = end
            for token in BREAKS:
                idx = text.rfind(token, start + self._chunk_overlap, end)
                if idx != -1:
                    break_at = idx + len(token)
                    break
            chunks.append(text[start:break_at])
            start = max(start + 1, break_at - self._chunk_overlap)
        return chunks

    # ── Extractive fallback ────────────────────────────────────────────────────

    def _extractive_summarize(
        self, text: str, sections: List[Tuple[str, str]], reason: str = "unknown"
    ) -> str:
        header = f"[Note: Generated by extractive summarization — {reason}]\n\n"
        if sections:

            return header + self._extractive_with_sections(text, sections)
        return header + self._extractive_flat(text)

    def _extractive_with_sections(
        self, _full_text: str, sections: List[Tuple[str, str]]
    ) -> str:
        per_section = max(2, self._extractive_n // max(len(sections), 1))
        parts: List[str] = []
        for heading, body in sections:
            top = self._top_sentences(body, n=per_section)
            if top:
                parts.append(f"**{heading}**\n" + " ".join(top))
        return "\n\n".join(parts)[: 4000] if parts else self._extractive_flat(_full_text)

    def _extractive_flat(self, text: str) -> str:
        top = self._top_sentences(text, n=self._extractive_n)
        return (" ".join(top) if top else text[:500])[:4000]

    def _top_sentences(self, text: str, n: int) -> List[str]:
        sentences = re.split(r"(?<=[.!?])\s+", text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 25]
        if not sentences:
            return []
        if len(sentences) <= n:
            return sentences

        all_words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
        freq = Counter(w for w in all_words if w not in _STOPWORDS)
        total = len(sentences)

        def score(sent: str, idx: int) -> float:
            ws = [w for w in re.findall(r"\b[a-zA-Z]{3,}\b", sent.lower())
                  if w not in _STOPWORDS]
            tfidf     = sum(freq.get(w, 0) for w in ws) / (len(ws) + 1)
            pos_ratio = idx / max(total, 1)
            pos_bonus = 1.2 if pos_ratio < 0.1 or pos_ratio > 0.9 else 1.0
            len_bonus = 1.1 if 40 < len(sent) < 200 else 0.9
            return tfidf * pos_bonus * len_bonus

        scored = [(score(s, i), i, s) for i, s in enumerate(sentences)]
        scored.sort(reverse=True)
        top = sorted(scored[:n], key=lambda x: x[1])
        return [s for _, _, s in top]
