"""Single-task code generation contract for the no-decomposition ablation."""


def document_extraction_prompt(template_json: str, layouts: str) -> str:
    return f"""Write ONE reusable Python program, extract.py, that extracts structured
data from unseen PDFs of the same document family following the supplied template.
Solve the entire extraction task yourself in this program. Do not generate a
block separator, per-node extraction scripts, or a pipeline assembly layer.

You have training.pdf (only the pages selected by clustering), training_ocr.csv
(exact word text and PDF coordinates), training_layout.txt, and template.json
in the current folder. These are the complete training inputs. Use the layouts
and OCR to understand the family; do not hard-code their values, coordinates,
page numbers, row counts, or block counts. Derive geometry from each new input.
Only use Python's standard library and pdfplumber. Do not make network/model
calls, install packages, or use other files. Put all runtime logic and the
template in extract.py so it can run from a fresh directory on its own.

CLI CONTRACT
    python extract.py <input.pdf> --out-dir <directory>
Read ALL pages with pdfplumber and write <directory>/extracted.json as UTF-8 JSON.
For OCR use page.extract_words(x_tolerance=3, y_tolerance=3,
keep_blank_chars=False, use_text_flow=False), preserving the text exactly.
No precomputed OCR or repository modules will be available at runtime.

OUTPUT CONTRACT (required for the scorer)
The JSON root is a flat list of physical data occurrences in page order, then
top-to-bottom reading order. For example (the names/IDs here are placeholders):
[
  {{"page": 1, "type": "table", "block_id": "7_0001", "uid": "p1_7_0001",
    "data": [{{"Column A": "exact text", "Column B": ""}}]}},
  {{"page": 1, "type": "kv", "block_id": "9_0001", "uid": "p1_9_0001",
    "data": {{"Field A": "exact text", "Field B": ""}}}}
]
- Use the actual physical template node_id, never bid, as the block_id prefix:
  <node_id>_<page-local-instance>. uid must be unique across the input.
- page is the 1-based LOCAL page number of the input PDF, including when the
  source training pages were noncontiguous. Never use original source numbers.
- type is exactly table or kv. Table data is a list of row objects; KV data is
  one object. Every data object's keys exactly match that node's fields in the
  template. Values are strings; use "" for absent values. No renamed fields.
- Emit every repeated occurrence separately. Keep continuation tables on
  different pages separate. Never merge physical blocks across pages.
- XOR nodes are virtual alternative record slots. Select the matching physical
  alternative by its fields/layout; NEVER emit xor-table/xor-kv nodes as data.
- Do not emit headers, metadata or page numbers as data. Do not discard real
  bottom rows. Page-frame removal requires both edge position and a matching
  document_level metadata regex. Do not invent or normalize visible values.
- Do not group records or write scorer reports/ground truth. The host applies
  the same deterministic XOR-aware grouping to candidate and reference output.
- Write an empty list if no data is found. Malformed output is a contract error;
  missing/extra content is measured by the scorer without repair feedback.

FULL TEMPLATE (including virtual XOR slots and metadata)
{template_json}

SELECTED TRAINING PAGE LAYOUTS
{layouts}

Write extract.py now. You may run it on training.pdf to check the CLI and JSON
contract. Leave no runtime dependencies on training files or auxiliary scripts.
"""
