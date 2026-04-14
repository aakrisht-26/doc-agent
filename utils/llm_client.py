"""
LLMClient -- unified LLM interface specifically for Groq Cloud.

Provider:
    Groq -- requires GROQ_API_KEY env var or groq.api_key in config.
    Uses the `openai` Python package with Groq's base_url.

Usage:
    from utils.llm_client import LLMClient
    client = LLMClient.from_config(cfg_dict)
    response = client.chat(messages=[...])
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

from utils.logger import get_logger

logger = get_logger(__name__)

# Groq defaults
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_DEFAULT_MODEL = "llama-3.3-70b-versatile"

class LLMClient:
    """
    Client for Groq Cloud (OpenAI-compatible interface).
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str = GROQ_BASE_URL,
        timeout: int = 180,
        temperature: float = 0.15,
    ) -> None:
        self.model       = model
        self.api_key     = api_key
        self.base_url    = base_url
        self.timeout     = timeout
        self.temperature = temperature
        self._provider   = "groq"
        self._client     = None

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "LLMClient":
        """Build from config. Prioritizes GROQ_API_KEY environment variable."""
        groq_cfg = cfg.get("groq", {})
        
        # ── API KEY SOURCE ───────────────────────────────────────────
        api_key = (
            os.environ.get("GROQ_API_KEY")
            or groq_cfg.get("api_key", "")
        )
        # ─────────────────────────────────────────────────────────────
        
        if not api_key:
            logger.warning("No API key found for Groq (GROQ_API_KEY). LLM features disabled.")
            return cls(model="", api_key="")

        model    = groq_cfg.get("model", GROQ_DEFAULT_MODEL)
        base_url = groq_cfg.get("base_url", GROQ_BASE_URL)
        timeout  = int(groq_cfg.get("timeout_seconds", 180))
        temp     = float(groq_cfg.get("temperature", 0.15))
        
        return cls(
            model=model,
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            temperature=temp,
        )

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    @property
    def provider(self) -> str:
        return self._provider

    @property
    def provider_label(self) -> str:
        return f"groq/{self.model}" if self.available else "none"

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: int = 3000,
    ) -> Optional[str]:
        if not self.available:
            return None

        temp = temperature if temperature is not None else self.temperature
        
        try:
            from openai import OpenAI
            
            if self._client is None:
                self._client = OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url,
                    timeout=float(self.timeout),
                )

            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temp,
                max_tokens=max_tokens,
            )
            content = response.choices[0].message.content
            return content.strip() if content else None

        except Exception as exc:
            logger.warning(f"Groq API call failed: {exc}")
            return None

    def __repr__(self) -> str:
        return f"<LLMClient provider=groq model={self.model!r}>"
