import json as _json


def PROMPT_BLOCK_EXTRACTION_SCRIPTING(template_id: str, template_node: dict | None) -> str:
    block_type = (template_node or {}).get("type", "unknown")
    template_node_json = _json.dumps(template_node, indent=2) if template_node else "null"

    return f"""You are inside the folder containing word-level OCR samples for block type "{template_id}".

The template node that describes this block type is:
{template_node_json}

The "fields" list above names every field present in this block. Use these names verbatim
as keys in the output (do not rename or invent fields).

The "complexity" object describes the structural difficulty of this block and must guide
how you write the extraction logic:
- For kv blocks:
  - "repeated": true → the block appears more than once inside a single record; your
    extract() must return "data" as a list of dicts, one per repetition.
    CRITICAL: a new repetition begins ONLY when the FIRST field in the template
    "fields" list reappears in a new row. Do NOT start a new repetition when any
    other (non-first) field is seen again — those appearances are continuation lines
    of the current repetition, not a new one. false → return "data" as a single dict.
- For table blocks:
  - "same_row_structure": true → every data row has the same layout (one value per
    column); a simple row-grouping by top-coordinate is sufficient. false → rows may
    differ structurally (spanning cells, blank separators, sub-labels); your logic must
    handle the variation rather than assuming uniform alignment.
  - "nested": true → a sibling/child template node appears spatially within this
    block's page region (the template "child" field points to it). This is a layout
    property only: it does NOT mean cell values are structured. Every cell in THIS
    block must still be extracted as a plain string — do NOT produce dicts, lists, or
    any non-string cell values, regardless of what punctuation (colons, commas, etc.)
    appears inside the cell text. The child node has its own extraction script.
    false → each cell contains a single plain value; extract it directly.

Each .json file in this folder is one sampled instance of this block type and has the structure:
  {{
    "page": <int>,
    "block_id": "<template_id>_<instance>",
    "type": "{block_type}",
    "words": [
      {{"x0": <float>, "top": <float>, "x1": <float>, "bottom": <float>, "text": <str>}},
      ...
    ]
  }}

Word coordinates are relative to the top-left corner of the block (top-left = 0, 0).
The corresponding .png files show the visual rendering of each sample.

Your task:
1. Read and analyse ALL the .json (and optionally .png) sample files in this directory
   to understand the spatial layout of the block.
2. Write a Python script and save it as {template_id}.py in the current directory.

The script must define exactly this function:

  def extract(block: dict) -> dict

  - `block` is a dict with keys "block_id" (str), "type" (str), and "words" (list[dict]).
    Each word dict has "x0", "top", "x1", "bottom", "text" (coordinates relative to the
    block's top-left corner).
  - The function reads block["block_id"] to populate the "block_id" field of the output.

Return format for a TABLE block ("{block_type}" == "table"):
  {{
    "type": "table",
    "block_id": "<block_id>",
    "data": [
      {{"<field_1>": "<value row 1>", "<field_2>": "<value row 1>", ...}},
      {{"<field_1>": "<value row 2>", "<field_2>": "<value row 2>", ...}},
      ...
    ]
  }}
  Each entry in "data" is one data row. Use spatial clustering to identify rows and to
  align each cell with the correct column header from the template fields list.

Return format for a KV block ("{block_type}" == "kv"):
  Single KV group:
  {{
    "type": "kv",
    "block_id": "<block_id>",
    "data": {{"<field_1>": "<value_1>", "<field_2>": "<value_2>", ...}}
  }}
  Multiple KV groups inside the same block:
  {{
    "type": "kv",
    "block_id": "<block_id>",
    "data": [
      {{"<field_1>": "<value_1>", ...}},
      {{"<field_1>": "<value_2>", ...}},
      ...
    ]
  }}
  Determine whether "data" should be a dict or a list by inspecting the samples.

How `extract()` is called at runtime:
  The pipeline (pipeline.py) has already run a separate block-separation step that
  locates every block instance on every page and relativises its word coordinates.
  For each detected instance it calls:

    relative_block = {{**block, "words": _relativize(block["words"])}}
    result = extract(relative_block)

  So `extract()` always receives a single, already-isolated block whose word
  coordinates start at (0, 0). It must NOT attempt to locate the block inside a
  full page, parse an OCR CSV, or do any document-level logic — all of that is
  handled upstream. Focus entirely on parsing the words of the one block passed in
  and returning the structured output.

Additional requirements:
- Use the template "fields" as the canonical key names in all output dicts.
- Use word positions (x0, top, x1, bottom) to group words into rows/columns and to
  match labels to values. Do not hard-code pixel thresholds; derive them from the data.
- COLUMN BOUNDARIES (table blocks): derive the column x-centers from the header
  band of THE BLOCK BEING EXTRACTED, at call time — locate each field label's
  words inside `block["words"]` and use their positions as the column anchors.
  Never bake in x-coordinates measured on the samples: unseen pages shift
  horizontally, and stale anchors silently swap adjacent columns (values land
  under the wrong field, which is worse than a missing value). If the block has
  no header band of its own (a continuation block), fall back to clustering the
  data words by x-position into as many columns as the template has fields, and
  assign columns to fields left-to-right in template order.
- After assigning cells, sanity-check the column mapping: the detected columns'
  left-to-right order must match the template "fields" order (when the labels are
  present). If a value row systematically fills a column whose header words were
  never found, prefer the positional (left-to-right) assignment.
- Row-filter guards: do not reject data rows with strict regexes on token counts
  or exact value formats (e.g. "row must have >= 9 tokens", "first token must be
  a pure integer", "tokens 2-3 must match a date pattern"). OCR on unseen pages
  produces merged/split tokens and o/0 confusions; strict filters silently drop
  every row. Prefer positional evidence (a word under the first column's anchor)
  and treat format checks as soft signals only.
- Be generalizable: the function must work correctly on any instance of block type
  "{template_id}", not only the sampled ones.
- Use only the Python standard library. Do not import any external packages.
- Define ONLY the `extract(block: dict) -> dict` function. Do NOT write a `main()`
  function, a CLI entry point, or any block-detection / document-parsing logic.

Save the finished script as {template_id}.py in the current directory."""
