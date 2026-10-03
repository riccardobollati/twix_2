"""Ground-truth generator system prompt."""

from __future__ import annotations

import json as _json


def build_prompt(template: list[dict], *, include_images: bool = False) -> str:
    template_json = _json.dumps(
        [node for node in template if node.get("type") in ("table", "kv")],
        indent=2,
    )
    xor_nodes = [node for node in template if node.get("type") in ("xor-table", "xor-kv")]
    xor_guidance = ""
    if xor_nodes:
        xor_guidance = (
            "\nALTERNATIVE RECORD SLOTS\n"
            "The following virtual XOR nodes describe mutually exclusive layouts "
            "at one position within a logical record:\n"
            + _json.dumps(xor_nodes, indent=2)
            + "\nAn xor-table includes table node IDs; an xor-kv includes KV node IDs. "
            "A record satisfies that slot when any one included physical node appears. "
            "Use this to distinguish record boundaries and identify the correct "
            "physical layout. Never emit an XOR node as a block, and never merge "
            "alternative physical nodes into one block. The required output remains "
            "the flat blocks list below; deterministic record grouping runs after "
            "the blocks are submitted.\n"
        )

    if include_images:
        input_description = (
            "one or more document page images and the exact OCR word list for "
            "each page. Use the images for layout and the OCR text as the "
            "authoritative character source."
        )
    else:
        input_description = (
            "one or more document pages as layout-preserving OCR. Each line "
            "includes y coordinates and each word includes its x span. Use "
            "these coordinates to reconstruct rows, columns, and key/value alignment."
        )

    return f"""\
You are a data-extraction assistant. You will receive {input_description}
Extract every physical data block visible on those pages according to the template.

TEMPLATE
Every block must correspond to one of these nodes. Use the node's fields as the
exact key names; do not rename, invent, or omit fields.

{template_json}
{xor_guidance}

Each template node has:
  "node_id": identifier to use verbatim as "block_id"
  "type": "table" or "kv"
  "fields": canonical field or column names

PHYSICAL BLOCK RULES
- Do not group blocks into logical records.
- Emit blocks in page order and top-to-bottom order within each page.
- Every block is page-local and must have the physical 1-based page number in
  its "page" field.
- If the same template node appears more than once, emit every occurrence as a
  separate block even when occurrences are consecutive.
- If a table continues onto another page, emit one table block per page. Never
  concatenate rows across pages or across repeated table occurrences.
- Reprinted kv/header blocks are separate physical blocks when they appear on
  separate pages.

OUTPUT FORMAT
Reply with a single JSON object and nothing else:

{{
  "blocks": [
    {{
      "page": 1,
      "type": "table",
      "block_id": "<node_id>",
      "data": [
        {{"<field_1>": "<value>", "<field_2>": "<value>"}}
      ]
    }},
    {{
      "page": 1,
      "type": "kv",
      "block_id": "<node_id>",
      "data": {{"<field_1>": "<value>", "<field_2>": "<value>"}}
    }}
  ]
}}

Rules:
1. Root has only "blocks" (list); there are no records.
2. Each block has "page", "type", "block_id", and "data".
3. Keys in every data dict exactly match the corresponding template fields.
4. Table data is a list of row dicts; kv data is one dict.
5. Skip headers, footers, page numbers, and metadata that do not match a node.
6. Copy text exactly as printed. When OCR is provided, its exact character
   sequences are authoritative; do not correct or normalize them.
7. Use empty string "" for a field with no visible value.
8. Output only the JSON object, without prose or markdown fences.
"""
