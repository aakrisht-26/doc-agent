"""
YouTubeSkill — extracts the transcript from a YouTube video and produces a ParsedDocument.

Uses youtube-transcript-api (no API key required, uses YouTube's own CC system).
"""

from __future__ import annotations

import re
import time
from typing import Any, Dict, List, Optional

from core.models import DocumentChunk, ParsedDocument, SkillInput, SkillOutput
from skills.base_skill import BaseSkill
from utils.logger import get_logger

logger = get_logger(__name__)

_YT_PATTERNS = [
    re.compile(r"(?:v=|youtu\.be/|embed/)([A-Za-z0-9_\-]{11})"),
]


def _extract_video_id(url: str) -> Optional[str]:
    for pat in _YT_PATTERNS:
        m = pat.search(url)
        if m:
            return m.group(1)
    return None


class YouTubeSkill(BaseSkill):
    """
    Extracts the transcript from a YouTube video URL and returns a ParsedDocument.

    Config keys:
        yt_languages (list): Preferred caption languages (default: ["en"])
    """

    name = "youtube"
    description = "Extract and process YouTube video transcripts."
    required_inputs = ["youtube_url"]

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config)
        self._languages = self.get_config("yt_languages", ["en"])

    def execute(self, inputs: SkillInput) -> SkillOutput:
        start = time.monotonic()
        url: str = inputs.data.get("youtube_url", "").strip()

        if not url:
            return SkillOutput(success=False, data=None, error="No YouTube URL provided.")

        video_id = _extract_video_id(url)
        if not video_id:
            return SkillOutput(
                success=False, data=None,
                error=f"Could not extract video ID from URL: {url}"
            )

        try:
            from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
        except ImportError:
            return SkillOutput(
                success=False, data=None,
                error="youtube-transcript-api not installed. Run: pip install youtube-transcript-api"
            )

        try:
            self.logger.info(f"Fetching transcript for YouTube video: {video_id}")
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=self._languages)
            full_text = " ".join(entry["text"] for entry in transcript_list)
            full_text = re.sub(r"\s+", " ", full_text).strip()
            self.logger.info(f"YouTube transcript fetched: {len(full_text)} chars")

        except Exception as e:
            return SkillOutput(success=False, data=None, error=f"Failed to fetch transcript: {e}")

        # Split into 500-word chunks, treating each as a "page"
        chunks: List[DocumentChunk] = []
        words = full_text.split()
        chunk_size = 500
        for i in range(0, len(words), chunk_size):
            chunk_words = words[i:i + chunk_size]
            chunk_num = i // chunk_size + 1
            chunks.append(DocumentChunk(
                text=" ".join(chunk_words),
                page_or_sheet=chunk_num,
                chunk_index=chunk_num - 1,
                metadata={"source": "youtube_transcript", "video_id": video_id}
            ))

        parsed_doc = ParsedDocument(
            file_name=f"youtube_{video_id}.txt",
            file_type="youtube",
            chunks=chunks,
            full_text=full_text,
            tables=[],
            metadata={
                "source": "youtube_transcript",
                "video_id": video_id,
                "url": url,
                "word_count": len(words),
            },
            page_count=len(chunks),
        )

        return SkillOutput(
            success=True,
            data=parsed_doc,
            duration_ms=(time.monotonic() - start) * 1000,
        )
