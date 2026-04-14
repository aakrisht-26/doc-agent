"""
StructureRecognitionSkill — uses PaddleOCR's PP-Structure to parse complex tables layout using GPU.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import json

from core.models import DocumentChunk, ParsedDocument, SkillInput, SkillOutput
from skills.base_skill import BaseSkill
from utils.logger import get_logger

logger = get_logger(__name__)


class StructureRecognitionSkill(BaseSkill):
    """
    Identifies and extracts highly complex tables from given PDFs using PaddleOCR's PPStructure.
    Only intended to be used on domains that heavily feature tables (Technical, Scientific, Financial).
    
    Config keys:
        use_gpu (bool): Whether to use GPU acceleration (default: True).
        show_log (bool): Print paddle logs (default: False).
    """

    name = "structure_recognition"
    description = "Extracts highly structured tables from complex domain PDFs."
    required_inputs = ["parsed_document", "file_path"]

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config)
        self._use_gpu = self.get_config("use_gpu", True)
        self._show_log = self.get_config("show_log", False)
        
        # Load the engine lazily to save VRAM when not in use
        self._engine = None

    def _get_engine(self):
        if self._engine is None:
            self.logger.info(f"Loading PaddleOCR PP-Structure Engine (GPU={self._use_gpu})...")
            try:
                from paddleocr import PPStructure
                # Disable warning logs from paddle
                import logging
                logging.getLogger("ppocr").setLevel(logging.ERROR)
            except ImportError as e:
                self.logger.error("PaddleOCR is not installed. Run `pip install paddleocr paddlepaddle-gpu`")
                raise e
            self._engine = PPStructure(show_log=self._show_log, use_gpu=self._use_gpu, lang="en")
        return self._engine

    def execute(self, inputs: SkillInput) -> SkillOutput:
        start = time.monotonic()
        
        parsed_doc: ParsedDocument = inputs.data["parsed_document"]
        file_path = Path(inputs.data["file_path"])
        domain = inputs.data.get("domain", "General")
        
        # Guard clause: We only do this computationally expensive pass if it's structural
        target_domains = ["Technical", "Financial", "Research", "Scientific"]
        if domain not in target_domains:
            self.logger.info(f"Skipping Structure Recognition (Domain '{domain}' does not require intensive table parsing).")
            return SkillOutput(success=True, data=parsed_doc)
            
        self.logger.info(f"Initiating High-Fidelity Table Search for {domain} document.")

        try:
            import fitz
            import cv2
            import numpy as np
        except ImportError:
            self.logger.error("PyMuPDF (fitz) or cv2 missing for Structure Recognition.")
            return SkillOutput(success=False, data=None, error="Missing dependencies: fitz, cv2")

        with fitz.open(str(file_path)) as doc:
            try:
                engine = self._get_engine()
            except Exception as e:
                self.logger.error(f"Failed to initialize PP-Structure engine: {e}")
                return SkillOutput(success=False, data=parsed_doc, error=str(e))
            
            extracted_tables = []
            new_chunks = []
            
            for i, chunk in enumerate(parsed_doc.chunks):
                # Convert page directly to high-res image
                page_index = min(max(int(chunk.page_or_sheet) - 1, 0), len(doc) - 1)
                page = doc[page_index]
                
                # Use 200 DPI to save VRAM but maintain cell integrity
                pix = page.get_pixmap(dpi=200)
                img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
                
                if pix.n == 4:
                    img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
                elif pix.n == 3:
                    img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                else:
                    img_cv = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)

                self.logger.debug(f"  PP-Structure scanning page {page_index + 1}...")
                results = engine(img_cv)
                
                table_markdown_blocks = []
                
                for res_idx, res in enumerate(results):
                    res_type = res.get('type')
                    if res_type == 'table':
                        # res contains 'res' key with dict bearing 'html', etc.
                        table_html = res.get('res', {}).get('html', '')
                        
                        # Store purely formatted table html/markdown string
                        if table_html:
                            table_str = f"[HIGH-FIDELITY TABLE - PAGE {page_index + 1}]\n{table_html}\n"
                            table_markdown_blocks.append(table_str)
                            
                            extracted_tables.append({
                                "page": page_index + 1,
                                "index": res_idx,
                                "html": table_html
                            })
                
                # If we found high spatial tables, we append them to the existing chunk
                # This allows the summarize skill to read the perfect HTML table rather than garbled strings
                if table_markdown_blocks:
                    new_text = chunk.text + "\n\n" + "\n".join(table_markdown_blocks)
                    self.logger.info(f"  Found {len(table_markdown_blocks)} HD tables on page {page_index + 1}")
                else:
                    new_text = chunk.text
                    
                new_chunks.append(DocumentChunk(
                    text=new_text,
                    page_or_sheet=chunk.page_or_sheet,
                    chunk_index=chunk.chunk_index,
                    metadata={**chunk.metadata, "hd_tables": len(table_markdown_blocks)}
                ))

        # Update the parsed document with improved text and explicit tables list
        new_parsed_doc = ParsedDocument(
            file_name=parsed_doc.file_name,
            file_type=parsed_doc.file_type,
            chunks=new_chunks,
            full_text="\n\n".join(c.text for c in new_chunks),
            tables=parsed_doc.tables + extracted_tables, # append new tables
            metadata={**parsed_doc.metadata, "pp_structure": True},
            page_count=parsed_doc.page_count
        )

        return SkillOutput(
            success=True,
            data=new_parsed_doc,
            duration_ms=(time.monotonic() - start) * 1000
        )
