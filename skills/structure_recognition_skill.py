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
    """

    name = "structure_recognition"
    description = "Extracts highly structured tables from complex domain PDFs."
    required_inputs = ["parsed_document", "file_path"]

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config)
        # Note: GPU mode is disabled — paddlepaddle-gpu conflicts with PyTorch CUDA DLLs.
        # CPU inference is used via a subprocess bridge instead.
        self._use_gpu = False
        self._show_log = self.get_config("show_log", False)

    def execute(self, inputs: SkillInput) -> SkillOutput:
        start = time.monotonic()
        parsed_doc: ParsedDocument = inputs.data["parsed_document"]
        file_path = inputs.data["file_path"]
        domain = inputs.data.get("domain", "General")
        
        target_domains = ["Technical", "Financial", "Research", "Scientific"]
        if domain not in target_domains:
            return SkillOutput(success=True, data=parsed_doc)
            
        self.logger.info(f"Initiating High-Fidelity Table Search for {domain} document.")

        import subprocess
        import sys
        import json

        pages_to_scan = [min(max(int(chunk.page_or_sheet) - 1, 0), parsed_doc.page_count - 1) for chunk in parsed_doc.chunks]
        
        payload = {
            "file_path": file_path,
            "pages_to_scan": list(set(pages_to_scan)), # dedupe
            "use_gpu": self._use_gpu
        }

        bridge_script = str(Path(__file__).parent.parent / "utils" / "paddle_bridge.py")
        
        self.logger.info(f"Enforcing Process Isolation Bridge for Paddle GPU...")
        
        try:
            # First attempt: Try with Configured Settings (usually GPU)
            result = self._run_bridge(payload, bridge_script)
            
            # Critical Fallback: If GPU vision crashes due to CUDA DLL hell, try CPU vision to save the pipeline
            if not result.get("success", False) and self._use_gpu:
                self.logger.warning(f"GPU Table Bridge failed ({result.get('error')}). Retrying on CPU to prevent crash.")
                payload["use_gpu"] = False
                result = self._run_bridge(payload, bridge_script)

        except Exception as e:
            self.logger.error(f"Table Bridge execution failed: {e}")
            return SkillOutput(success=False, data=parsed_doc, error=str(e))

        if not result.get("success", False):
            self.logger.error(f"Table Extraction definitively failed: {result.get('error')}")
            return SkillOutput(success=False, data=parsed_doc, error=result.get("error"))

        blocks = result.get("blocks", {})
        extracted_tables = result.get("tables", [])
        
        new_chunks = []
        for chunk in parsed_doc.chunks:
            page_index = str(min(max(int(chunk.page_or_sheet) - 1, 0), parsed_doc.page_count - 1))
            table_markdown_blocks = blocks.get(page_index, [])
            
            if table_markdown_blocks:
                new_text = chunk.text + "\n\n" + "\n".join(table_markdown_blocks)
                self.logger.info(f"  Found {len(table_markdown_blocks)} HD tables on page {int(page_index) + 1}")
            else:
                new_text = chunk.text
                
            new_chunks.append(DocumentChunk(
                text=new_text,
                page_or_sheet=chunk.page_or_sheet,
                chunk_index=chunk.chunk_index,
                metadata={**chunk.metadata, "hd_tables": len(table_markdown_blocks)}
            ))

        new_parsed_doc = ParsedDocument(
            file_name=parsed_doc.file_name,
            file_type=parsed_doc.file_type,
            chunks=new_chunks,
            full_text="\n\n".join(c.text for c in new_chunks),
            tables=parsed_doc.tables + extracted_tables,
            metadata={**parsed_doc.metadata, "pp_structure": True},
            page_count=parsed_doc.page_count
        )

        return SkillOutput(
            success=True,
            data=new_parsed_doc,
            duration_ms=(time.monotonic() - start) * 1000
        )

    def _run_bridge(self, payload: dict, script_path: str) -> dict:
        import subprocess
        import sys
        import json
        
        try:
            # We use subprocess.run with a fresh environment if possible
            env = os.environ.copy()
            # HIDE common torch env vars from Paddle just in case
            if "TORCH_HOME" in env: del env["TORCH_HOME"]

            # On Windows, we need to hide the console window from Streamlit's process tree
            process = subprocess.run(
                [sys.executable, script_path, json.dumps(payload)],
                capture_output=True,
                text=True,
                encoding='utf-8',
                env=env,
                creationflags=0x08000000 if os.name == 'nt' else 0 # CREATE_NO_WINDOW
            )
            
            if process.returncode != 0:
                return {"success": False, "error": f"Bridge Exit Code {process.returncode}: {process.stderr}"}
            
            # The script prints exactly one JSON line
            for line in process.stdout.splitlines():
                if line.strip().startswith("{"):
                    return json.loads(line)
                    
            return {"success": False, "error": f"No valid JSON output. Stdout: {process.stdout}"}
        except Exception as e:
            return {"success": False, "error": str(e)}
