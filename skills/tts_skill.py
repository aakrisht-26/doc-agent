"""
TTSSkill — converts text (summaries) to speech using gTTS.

Returns audio bytes (MP3) suitable for st.audio() playback.
"""

from __future__ import annotations

import io
import time
from typing import Any, Dict, Optional

from core.models import SkillInput, SkillOutput
from skills.base_skill import BaseSkill
from utils.logger import get_logger

logger = get_logger(__name__)


class TTSSkill(BaseSkill):
    """
    Text-to-Speech skill using gTTS (Google TTS, requires internet).

    Config keys:
        tts_lang (str): Language code (default: "en")
        tts_slow (bool): Slow speech mode (default: False)
    """

    name = "tts"
    description = "Convert text to speech audio (MP3 bytes) using gTTS."
    required_inputs = ["text"]

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config)
        self._lang = self.get_config("tts_lang", "en")
        self._slow = self.get_config("tts_slow", False)

    def execute(self, inputs: SkillInput) -> SkillOutput:
        start = time.monotonic()
        text: str = inputs.data.get("text", "").strip()

        if not text:
            return SkillOutput(success=False, data=None, error="No text provided to TTS.")

        # Truncate very long texts to avoid hitting gTTS limits (max ~5000 chars)
        if len(text) > 4500:
            text = text[:4500] + "... (truncated for audio)"

        try:
            from gtts import gTTS
        except ImportError:
            return SkillOutput(
                success=False, data=None,
                error="gTTS not installed. Run: pip install gTTS"
            )

        try:
            self.logger.info(f"Generating TTS audio ({len(text)} chars, lang={self._lang})...")
            tts = gTTS(text=text, lang=self._lang, slow=self._slow)
            buf = io.BytesIO()
            tts.write_to_fp(buf)
            buf.seek(0)
            audio_bytes = buf.read()
            self.logger.info(f"TTS audio generated: {len(audio_bytes)} bytes")

            return SkillOutput(
                success=True,
                data={"audio_bytes": audio_bytes, "format": "mp3"},
                duration_ms=(time.monotonic() - start) * 1000,
            )
        except Exception as e:
            self.logger.error(f"TTS generation failed: {e}")
            return SkillOutput(success=False, data=None, error=str(e))
