import os
import sys
import json
import logging
import time

# ABSOLUTE ISOLATION: Hide everything from this process except what is needed
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
os.environ["FLAGS_allocator_strategy"] = "auto"

# Disable logs
logging.getLogger("ppocr").setLevel(logging.ERROR)

def run_extraction(file_path, pages_to_scan, use_gpu):
    try:
        from paddleocr import PPStructureV3
        import fitz
        import cv2
        import numpy as np

        engine = PPStructureV3(device="gpu" if use_gpu else "cpu", lang="en")
        
        extracted_blocks = {}
        extracted_tables = []
        
        with fitz.open(file_path) as doc:
            for page_index in pages_to_scan:
                if page_index >= len(doc):
                    continue
                page = doc[page_index]
                pix = page.get_pixmap(dpi=200)
                img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
                
                if pix.n == 4:
                    img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
                elif pix.n == 3:
                    img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                else:
                    img_cv = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)

                results = engine(img_cv)
                table_htmls = []
                
                for res_idx, res in enumerate(results):
                    if res.get('type') == 'table':
                        html = res.get('res', {}).get('html', '')
                        if html:
                            table_htmls.append(f"[HIGH-FIDELITY TABLE - PAGE {page_index + 1}]\n{html}\n")
                            extracted_tables.append({
                                "page": page_index + 1,
                                "index": res_idx,
                                "html": html
                            })
                
                extracted_blocks[page_index] = table_htmls

        return {
            "success": True,
            "blocks": extracted_blocks,
            "tables": extracted_tables
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"success": False, "error": "No input JSON provided"}))
        sys.exit(1)
        
    try:
        input_data = json.loads(sys.argv[1])
        res = run_extraction(
            input_data["file_path"], 
            input_data["pages_to_scan"], 
            input_data["use_gpu"]
        )
        print(json.dumps(res))
    except Exception as e:
        print(json.dumps({"success": False, "error": f"Internal Script Error: {str(e)}"}))
