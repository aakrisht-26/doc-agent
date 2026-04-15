"""
AudioTranscriptionSkill — transcribes audio files to text using OpenAI Whisper (local).

Returns a ParsedDocument so the result flows through the same pipeline as PDFs.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.models import DocumentChunk, ParsedDocument, SkillInput, SkillOutput
from skills.base_skill import BaseSkill
from utils.logger import get_logger

logger = get_logger(__name__)


class AudioTranscriptionSkill(BaseSkill):
    """
    Transcribes an audio file using OpenAI Whisper (runs entirely locally).

    Config keys:
        whisper_model (str): Whisper model size — tiny / base / small / medium / large
                             (default: "base" — fast and accurate on most systems)
    """

    name = "audio_transcription"
    description = "Transcribe audio files to text using local Whisper model."
    required_inputs = ["file_path"]

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config)
        self._model_name = self.get_config("whisper_model", "base")
        self._model = None

    def _get_model(self):
        if self._model is None:
            try:
                import whisper
                self.logger.info(f"Loading Whisper model: {self._model_name}...")
                self._model = whisper.load_model(self._model_name)
            except ImportError:
                raise ImportError("openai-whisper not installed. Run: pip install openai-whisper")
        return self._model

    def execute(self, inputs: SkillInput) -> SkillOutput:
        start = time.monotonic()
        file_path = Path(inputs.data["file_path"])

        if not file_path.exists():
            return SkillOutput(success=False, data=None, error=f"Audio file not found: {file_path}")

        supported = {".mp3", ".mp4", ".wav", ".m4a", ".ogg", ".flac", ".webm"}
        if file_path.suffix.lower() not in supported:
            return SkillOutput(
                success=False, data=None,
                error=f"Unsupported audio format: {file_path.suffix}. Supported: {supported}"
            )

        try:
            model = self._get_model()
            self.logger.info(f"Transcribing {file_path.name}...")
            result = model.transcribe(str(file_path), fp16=False)
            transcript = result["text"].strip()
            self.logger.info(f"Transcription complete: {len(transcript)} chars")

            # Build a ParsedDocument so the pipeline can summarize/classify it
            chunks = []
            # Split transcript into 500-word chunks to mimic pages
            words = transcript.split()
            chunk_size = 500
            for i in range(0, len(words), chunk_size):
                chunk_words = words[i:i + chunk_size]
                chunk_text = " ".join(chunk_words)
                chunk_num = i // chunk_size + 1
                chunks.append(DocumentChunk(
                    text=chunk_text,
                    page_or_sheet=chunk_num,
                    chunk_index=chunk_num - 1,
                    metadata={"source": "whisper_transcription"}
                ))

            parsed_doc = ParsedDocument(
                file_name=file_path.name,
                file_type="audio",
                chunks=chunks,
                full_text=transcript,
                tables=[],
                metadata={
                    "source": "whisper_transcription",
                    "whisper_model": self._model_name,
                    "duration_chars": len(transcript),
                    "language": result.get("language", "unknown"),
                },
                page_count=len(chunks),
            )

            return SkillOutput(
                success=True,
                data=parsed_doc,
                duration_ms=(time.monotonic() - start) * 1000,
            )

        except Exception as e:
            self.logger.error(f"Transcription failed: {e}")
            return SkillOutput(success=False, data=None, error=str(e))
