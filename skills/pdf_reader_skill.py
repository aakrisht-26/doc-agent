"""
PDFReaderSkill — extracts text, tables, and metadata from PDF files.

Engine priority:
    1. pdfplumber  — best for text-layer PDFs (preserves layout, extracts tables)
    2. PyMuPDF     — fallback for complex/encrypted layouts
    3. Tesseract   — OCR fallback for scanned/image-only PDFs

All engines produce the same ParsedDocument output, ensuring
downstream skills are engine-agnostic.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.models import DocumentChunk, ParsedDocument, SkillInput, SkillOutput
from skills.base_skill import BaseSkill
from utils.logger import get_logger

logger = get_logger(__name__)


class PDFReaderSkill(BaseSkill):
    """
    Reads and parses PDF documents into a structured ParsedDocument.

    Config keys:
        use_ocr_fallback (bool)  : Enable Tesseract OCR for scanned PDFs (default: True)
        ocr_language     (str)   : Tesseract language code (default: "eng")
        extract_tables   (bool)  : Extract tables separately (default: True)
        max_pages        (int)   : Maximum pages to process (default: 500)
    """

    name = "pdf_reader"
    description = "Extracts text, tables, and metadata from PDFs. Supports OCR for scanned docs."
    required_inputs = ["file_path"]

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config)
        self._use_ocr      = self.get_config("use_ocr_fallback", True)
        self._ocr_lang     = self.get_config("ocr_language", "eng")
        self._extract_tbl  = self.get_config("extract_tables", True)
        self._max_pages    = self.get_config("max_pages", 500)

    # ── Skill entry point ─────────────────────────────────────────────

    def execute(self, inputs: SkillInput) -> SkillOutput:
        start = time.monotonic()
        file_path = Path(inputs.data["file_path"])

        if not file_path.exists():
            return SkillOutput(success=False, data=None,
                               error=f"File not found: {file_path}")

        # Try primary engine
        doc: Optional[ParsedDocument] = None
        try:
            doc = self._parse_pdfplumber(file_path)
            self.logger.info(f"pdfplumber extracted {doc.word_count} words from {doc.page_count} pages")
        except Exception as exc:
            self.logger.warning(f"pdfplumber failed ({exc}); trying PyMuPDF fallback.")
            try:
                doc = self._parse_fitz(file_path)
                self.logger.info(f"PyMuPDF extracted {doc.word_count} words")
            except Exception as exc2:
                return SkillOutput(
                    success=False, data=None,
                    error=f"Both PDF engines failed. pdfplumber: {exc}. PyMuPDF: {exc2}",
                    duration_ms=(time.monotonic() - start) * 1000,
                )

        # OCR escalation for image-only PDFs
        warnings: List[str] = []
        if doc.is_empty and self._use_ocr:
            self.logger.info("No text layer found — escalating to OCR.")
            doc = self._parse_ocr(file_path, fallback_doc=doc)
            if doc.is_empty:
                warnings.append(
                    "OCR produced no text. The document may be encrypted, "
                    "password-protected, or contain unsupported image formats."
                )
        elif doc.is_empty:
            warnings.append(
                "No text extracted (OCR fallback is disabled). "
                "Enable use_ocr_fallback=true in configs/default.yaml for scanned PDFs."
            )

        return SkillOutput(
            success=True,
            data=doc,
            warnings=warnings,
            duration_ms=(time.monotonic() - start) * 1000,
            metadata={"engine": doc.metadata.get("engine", "pdfplumber"),
                      "pages": doc.page_count},
        )

    # ── Engines ───────────────────────────────────────────────────────

    def _parse_pdfplumber(self, file_path: Path) -> ParsedDocument:
        """Primary parser — best text/table extraction for standard PDFs."""
        import pdfplumber

        chunks: List[DocumentChunk] = []
        tables: List[Dict[str, Any]] = []
        text_parts: List[str] = []
        raw_meta: Dict[str, Any] = {}

        with pdfplumber.open(str(file_path)) as pdf:
            raw_meta = {k: str(v) for k, v in (pdf.metadata or {}).items()}
            page_count = len(pdf.pages)

            for i, page in enumerate(pdf.pages[: self._max_pages]):
                page_text = page.extract_text() or ""

                # Table extraction
                if self._extract_tbl:
                    for j, raw_table in enumerate(page.extract_tables() or []):
                        rows = []
                        for row in raw_table:
                            cells = [
                                str(c).strip() if c is not None else ""
                                for c in row
                            ]
                            rows.append(" | ".join(cells))
                        tbl_text = "\n".join(rows)
                        if tbl_text.strip():
                            tables.append({
                                "page": i + 1, "index": j,
                                "text": tbl_text, "data": raw_table,
                            })
                            text_parts.append(f"\n[TABLE — Page {i + 1}, #{j + 1}]\n{tbl_text}")

                text_parts.append(page_text)
                chunks.append(DocumentChunk(
                    text=page_text,
                    page_or_sheet=i + 1,
                    chunk_index=i,
                    metadata={"page": i + 1, "engine": "pdfplumber"},
                ))

        raw_meta["engine"] = "pdfplumber"
        return ParsedDocument(
            file_name=file_path.name,
            file_type="pdf",
            chunks=chunks,
            full_text="\n\n".join(filter(None, text_parts)),
            tables=tables,
            metadata=raw_meta,
            page_count=page_count,
        )

    def _parse_fitz(self, file_path: Path) -> ParsedDocument:
        """Fallback parser — handles complex layouts, encrypted PDFs."""
        import fitz  # PyMuPDF

        chunks: List[DocumentChunk] = []
        text_parts: List[str] = []

        doc = fitz.open(str(file_path))
        page_count = len(doc)

        for i, page in enumerate(doc[: self._max_pages]):
            text = page.get_text("text") or ""  # type: ignore[attr-defined]
            text_parts.append(text)
            chunks.append(DocumentChunk(
                text=text,
                page_or_sheet=i + 1,
                chunk_index=i,
                metadata={"page": i + 1, "engine": "fitz"},
            ))
        doc.close()

        return ParsedDocument(
            file_name=file_path.name,
            file_type="pdf",
            chunks=chunks,
            full_text="\n\n".join(filter(None, text_parts)),
            tables=[],
            metadata={"engine": "fitz"},
            page_count=page_count,
        )

    def _parse_ocr(self, file_path: Path, fallback_doc: ParsedDocument) -> ParsedDocument:
        """
        OCR engine: PyMuPDF renders pages to images, Tesseract reads them.
        Requires: PyMuPDF, pytesseract, Pillow, and Tesseract binary installed.
        """
        try:
            import fitz
            import pytesseract
            from PIL import Image
        except ImportError as exc:
            self.logger.warning(
                f"OCR dependencies not installed ({exc}). "
                "Install PyMuPDF, pytesseract, and Pillow — see README."
            )
            return fallback_doc

        chunks: List[DocumentChunk] = []
        text_parts: List[str] = []

        doc = fitz.open(str(file_path))
        page_count = len(doc)
        self.logger.info(f"OCR processing {page_count} pages (lang={self._ocr_lang})")

        for i, page in enumerate(doc[: self._max_pages]):
            self.logger.debug(f"  OCR page {i + 1}/{page_count}")
            pix = page.get_pixmap(dpi=300)  # type: ignore[attr-defined]
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            text = pytesseract.image_to_string(img, lang=self._ocr_lang)
            text_parts.append(text)
            chunks.append(DocumentChunk(
                text=text,
                page_or_sheet=i + 1,
                chunk_index=i,
                metadata={"page": i + 1, "engine": "ocr_tesseract"},
            ))
        doc.close()

        return ParsedDocument(
            file_name=file_path.name,
            file_type="pdf",
            chunks=chunks,
            full_text="\n\n".join(filter(None, text_parts)),
            tables=[],
            metadata={**fallback_doc.metadata, "engine": "ocr_tesseract", "ocr": True},
            page_count=page_count,
        )
