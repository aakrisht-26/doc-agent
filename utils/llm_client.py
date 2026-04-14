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
        api_keys: List[str] | str,
        base_url: str = GROQ_BASE_URL,
        timeout: int = 180,
        temperature: float = 0.15,
    ) -> None:
        self.model       = model
        self.api_keys    = [api_keys] if isinstance(api_keys, str) else api_keys
        self.base_url    = base_url
        self.timeout     = timeout
        self.temperature = temperature
        self._provider   = "groq"
        self._client     = None
        self._current_key_idx = 0

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "LLMClient":
        """Build from config. Prioritizes GROQ_API_KEY environment variable."""
        groq_cfg = cfg.get("groq", {})
        
        # ── API KEY SOURCE ───────────────────────────────────────────
        raw_keys = (
            os.environ.get("GROQ_API_KEYS")
            or os.environ.get("GROQ_API_KEY")
            or groq_cfg.get("api_keys", "")
            or groq_cfg.get("api_key", "")
        )
        api_keys = [k.strip() for k in raw_keys.split(",") if k.strip()]
        # ─────────────────────────────────────────────────────────────
        
        model    = groq_cfg.get("model", GROQ_DEFAULT_MODEL)
        
        if not api_keys:
            logger.warning("No API key found for Groq (GROQ_API_KEY/S). LLM features disabled.")
            return cls(model=model, api_keys=[])

        base_url = groq_cfg.get("base_url", GROQ_BASE_URL)
        timeout  = int(groq_cfg.get("timeout_seconds", 180))
        temp     = float(groq_cfg.get("temperature", 0.15))
        
        return cls(
            model=model,
            api_keys=api_keys,
            base_url=base_url,
            timeout=timeout,
            temperature=temp,
        )

    @property
    def available(self) -> bool:
        return bool(self.api_keys)

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
            import openai
        except ImportError:
            logger.error("OpenAI package not installed.")
            return None
        
        attempts = 0
        max_attempts = len(self.api_keys)
        
        while attempts < max_attempts:
            try:
                if self._client is None:
                    self._client = openai.OpenAI(
                        api_key=self.api_keys[self._current_key_idx],
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

            except openai.RateLimitError as exc:
                logger.warning(f"Groq API rate limit hit with key {self._current_key_idx + 1}/{max_attempts}.")
                attempts += 1
                self._current_key_idx = (self._current_key_idx + 1) % max_attempts
                self._client = None # Force client recreation with new key
                time.sleep(1) # Brief pause before trying next
                
            except Exception as exc:
                logger.warning(f"Groq API call failed: {exc}")
                return None
                
        logger.error("All Groq API keys exhausted due to rate limits.")
        return None

    def __repr__(self) -> str:
        return f"<LLMClient provider=groq model={self.model!r}>"
