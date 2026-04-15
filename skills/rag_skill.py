"""
RagSkill — retrieval-augmented question answering over a parsed document.

Pipeline:
    1. Split full_text into overlapping chunks
    2. Embed chunks using sentence-transformers (all-MiniLM-L6-v2) — runs locally
    3. Build an in-memory FAISS flat index
    4. At query time: embed the question, retrieve top-k chunks, call LLM for answer

The index is cached in the skill instance (built once per document upload).
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import numpy as np

from core.models import SkillInput, SkillOutput
from skills.base_skill import BaseSkill
from utils.logger import get_logger

logger = get_logger(__name__)

_EMBED_MODEL_NAME = "all-MiniLM-L6-v2"


class RagSkill(BaseSkill):
    """
    Retrieval-Augmented Generation (RAG) over a document.

    Config keys:
        chunk_size   (int) : Characters per retrieval chunk (default: 800)
        chunk_overlap (int): Overlap between chunks (default: 150)
        top_k        (int) : How many chunks to retrieve per query (default: 4)
    """

    name = "rag"
    description = "Answer questions about a document using local embeddings + LLM."
    required_inputs = ["full_text", "query"]

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config)
        self._chunk_size    = self.get_config("rag_chunk_size", 800)
        self._chunk_overlap = self.get_config("rag_chunk_overlap", 150)
        self._top_k         = self.get_config("rag_top_k", 4)

        # Lazy-loaded embedding model and index (cached per document)
        self._embedder      = None
        self._index         = None
        self._chunks_cache: List[str] = []
        self._indexed_text: str = ""

        from utils.llm_client import LLMClient
        self._llm = LLMClient.from_config(self.config)

    # ── Entry point ────────────────────────────────────────────────────────────

    def execute(self, inputs: SkillInput) -> SkillOutput:
        start      = time.monotonic()
        full_text  = inputs.data["full_text"]
        query      = inputs.data["query"]
        chat_history: List[Dict] = inputs.data.get("chat_history", [])

        if not full_text.strip():
            return SkillOutput(success=False, data=None, error="Document text is empty.")
        if not query.strip():
            return SkillOutput(success=False, data=None, error="Query is empty.")

        # Build or reuse index
        if self._indexed_text != full_text:
            self._build_index(full_text)

        # Retrieve top-k chunks
        context_chunks = self._retrieve(query, self._top_k)
        context = "\n\n---\n\n".join(context_chunks)

        # Build LLM answer
        answer = self._answer(query, context, chat_history)

        return SkillOutput(
            success=True,
            data={"answer": answer, "context_chunks": context_chunks},
            duration_ms=(time.monotonic() - start) * 1000,
        )

    # ── Index builder ──────────────────────────────────────────────────────────

    def _build_index(self, text: str) -> None:
        import faiss

        self.logger.info("Building RAG embedding index...")
        self._chunks_cache = self._split_chunks(text)

        embedder = self._get_embedder()
        vectors = embedder.encode(self._chunks_cache, show_progress_bar=False)
        vectors = np.array(vectors, dtype="float32")

        # Normalize for cosine similarity
        faiss.normalize_L2(vectors)

        dim = vectors.shape[1]
        index = faiss.IndexFlatIP(dim)  # Inner product on normalized vecs = cosine sim
        index.add(vectors)

        self._index = index
        self._indexed_text = text
        self.logger.info(f"RAG index built: {len(self._chunks_cache)} chunks, dim={dim}")

    def _get_embedder(self):
        if self._embedder is None:
            self.logger.info(f"Loading sentence-transformer: {_EMBED_MODEL_NAME}...")
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(_EMBED_MODEL_NAME)
        return self._embedder

    def _split_chunks(self, text: str) -> List[str]:
        """Split text into overlapping chunks for retrieval."""
        chunks: List[str] = []
        start = 0
        length = len(text)
        while start < length:
            end = min(start + self._chunk_size, length)
            chunks.append(text[start:end])
            if end >= length:
                break
            start = end - self._chunk_overlap
        return chunks

    # ── Retrieval ──────────────────────────────────────────────────────────────

    def _retrieve(self, query: str, k: int) -> List[str]:
        import faiss
        import numpy as np

        embedder = self._get_embedder()
        q_vec = embedder.encode([query], show_progress_bar=False)
        q_vec = np.array(q_vec, dtype="float32")
        faiss.normalize_L2(q_vec)

        k = min(k, len(self._chunks_cache))
        _, indices = self._index.search(q_vec, k)
        return [self._chunks_cache[i] for i in indices[0] if i >= 0]

    # ── LLM answer ────────────────────────────────────────────────────────────

    def _answer(self, query: str, context: str, history: List[Dict]) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a precise document assistant. Answer questions based ONLY on the provided document excerpt. "
                    "If the answer is not in the context, say 'I couldn't find that in the document.' "
                    "Be concise and accurate. Do not make things up."
                ),
            }
        ]

        # Add recent chat history (last 6 exchanges = last 3 turns)
        for msg in history[-6:]:
            messages.append(msg)

        messages.append({
            "role": "user",
            "content": (
                f"Document excerpt:\n\n{context}\n\n"
                f"Question: {query}\n\n"
                "Answer:"
            ),
        })

        result = self._llm.chat(messages=messages, max_tokens=600, temperature=0.1)
        return result or "I was unable to generate an answer. Please try again."
