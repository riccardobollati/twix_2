def PROMPT_BLOCK_SEPARATION(ocr_csv_path, template_json, metadata_json="[]"):
    return f"""Create a Python script called `block_separation.py` that detects and extracts all blocks defined in the document template from every page of the OCR data.

Context:
- A sample OCR CSV file is at: {ocr_csv_path}
  Columns: id, page, x0, top, x1, bottom, text
  `id` is a globally unique integer assigned to each extracted word across the whole document.
- The document template defines the following blocks:
{template_json}
- The template's metadata patterns (Python `re`-compatible regexes) are:
{metadata_json}

Each template node has:
  "node_id"  : integer block identifier
  "type"     : "table" or "kv"
  "fields"   : list of field/column names that appear as visible labels or column headers

Each metadata entry has:
  "pattern"  : a Python `re` regex matching one metadata phrase
  "type"     : "document_level" (page frame: titles, headers, footers, page numbers)
               or "record_level" (repeats once per record)

The script must define a main function with this exact signature:

  def main(inputpath: str, out_path: str) -> None

  - `inputpath`: path to the OCR CSV file to process.
  - `out_path`: path to the folder where `block_separation.json` must be written.

────────────────────────────────────────────────────────────────
ALGORITHM
────────────────────────────────────────────────────────────────

STEP 1 — Load and group words into rows (page-local)
  Read the CSV. For each page, group words into rows by y-overlap:
  a word joins an existing row if its vertical span overlaps every word already in that
  row (y_i1 <= y_j2 AND y_i2 >= y_j1). Otherwise it starts a new row.
  Sort rows top-to-bottom by their minimum `top` coordinate.

STEP 2 — Strip structural noise (headers, footers, page numbers)
  Before block detection, remove rows that are page-level noise. A row is noise ONLY if
  it satisfies BOTH of the following:
    (a) Its position is either:
        - near the page TOP: within the top 5% of the page's word coordinate range, OR
        - near the page BOTTOM: within the bottom 8% of the page's word coordinate range.
    (b) The row's text (words joined left-to-right with single spaces) matches one of
        the "document_level" metadata patterns above (use `re.search`).
  A row that does not match any document-level metadata pattern must be KEPT, no matter
  how few words it contains — the last data row of a table often sits at the very
  bottom of a page and must never be stripped.
  Do NOT use the page's printed number value for this check — use position plus the
  metadata patterns only. Hard-code the document-level patterns above into the script
  as a list of regex strings.

STEP 3 — Detect and strictly validate block headers
  A block's header row is the row (or contiguous group of rows) where the block's
  field labels appear. Detect it as follows:

  For "table" nodes:
    Require a COMPLETE, UNAMBIGUOUS match of BOTH the fields and the table layout.
    A similar-looking table or a table containing all expected fields plus extra
    columns is NOT a match. Prefer leaving an incompatible table unclassified over
    assigning it to the wrong node_id.

    During generation, inspect the sample OCR to establish each node's structural
    signature: ordered leaf columns, full parent/group-header paths, units, number
    of header tiers, parent spans, and relative column arrangement. The template
    JSON supplies field identities; the sample OCR supplies their physical layout.
    Bind this signature to that node_id. Do not infer a replacement signature from
    an arbitrary runtime candidate, broaden the signature to accept incompatible
    sample tables, invent node IDs, or modify the template. If no sample provides
    an unambiguous complete reference, report that node as unsupported rather than
    creating a permissive fallback.

    At runtime, isolate a candidate's header band using local geometry and the
    transition to body rows. Reconstruct header cells within their own column spans
    and associate each leaf with its enclosing group headers. Require ALL of:
      - Every expected field maps to exactly one distinct leaf column, and every
        observed leaf column maps to exactly one expected field. No missing, extra,
        duplicated, merged, or split columns are allowed.
      - Full field labels, units, column order, header hierarchy, and group spans
        match the reference. A repeated label such as "Axis x-x" must be matched
        under its own parent (e.g. Elastic Modulus versus Plastic Modulus).
      - Relative column positions/widths and header-tier arrangement match the
        reference after translation and uniform scaling, with only small tolerances
        for OCR coordinate jitter and harmless wrapping inside the same header cell.
        Derive tolerances from local text size and spacing; do not permit arbitrary
        column movement, added tiers, or changed parent/child structure.

    Join wrapped words only within the same header cell in reading order. Never
    match a field using an unordered union of tokens from different columns, units,
    body rows, or an arbitrarily enlarged row range. Shared words elsewhere in the
    table cannot supply a missing field. Do not use majority/percentage matching,
    fuzzy similarity, or a nearest-template fallback. If multiple nodes match and
    layout cannot distinguish them, reject the candidate as ambiguous.

    Normalise whitespace and equivalent Unicode typography only when meaning is
    preserved. Preserve meaningful case, symbols, and units (e.g. T versus t,
    N versus n, mm versus cm, cm2 versus cm4); do not lowercase table symbols or
    discard units to force a match. Cosmetic line wrapping is not a new column.

  For "kv" nodes:
    A header row is one that contains at least ONE key label from the node's fields as
    a complete normalised token or contiguous token sequence.

  For kv labels, compare case-insensitively and strip surrounding punctuation
  (`:`, `.`, `,`, `#`). The stricter table rules above govern table headers.

STEP 4 — Assign content rows to blocks
  Process pages in ascending page order (the script can therefore carry state from one
  page to the next). After locating the header row(s) of each block on the page:
    - Include the validated header and its body within that table's column extent.
      Detect boundaries of ALL candidate tables/blocks, including rejected or unknown
      ones. Stop before a new candidate, incompatible header, section boundary, or
      incompatible column structure. A rejected table must never be absorbed into
      the preceding accepted table or kv block. Keep unknown regions unassigned.
    - Use validated column spans and local geometry to distinguish side-by-side
      tables. Vertical proximity alone must not assign words between different tables.
    - Preserve valid wrapped rows, blank cells, and bottom data rows inside the
      accepted table. Sparse rows alone are not evidence of a new table. Noise rows
      stripped in Step 2 are skipped, but do not terminate the current table.
    - CONTINUATION BLOCKS: rows that appear ABOVE the first header band on a page (or
      on a page with no header band at all) may continue the immediately preceding
      page's fully validated table ONLY with affirmative evidence that the entire
      column structure and body-row layout remain compatible. Partial column overlap
      is insufficient. There must be no intervening section boundary, unknown table,
      or rejected header. Any visible repeated header must pass the full Step 3
      validation; a failed header match is NEVER a headerless continuation. Clear
      continuation state on rejection or uncertainty. If continuity cannot be proved,
      leave the region unassigned. Emit validated continuations as separate page-local
      blocks of the same node, preserving the established signature.

────────────────────────────────────────────────────────────────
PROHIBITED HEURISTICS — these overfit the sample pages and break on unseen pages
────────────────────────────────────────────────────────────────

The script will run on OTHER pages of the same document family, not just the sample
CSV. The following shortcuts fit the sample but silently drop data later. Do NOT:

  - Suppress, filter, or skip kv blocks based on page position or page index
    (e.g. "kv blocks only appear on the first page", "drop kv candidates on
    continuation pages"). A kv node can appear on ANY page — once per record —
    and records start mid-document.
  - Filter blocks based on HOW MANY blocks of a type were detected on a page
    (e.g. "if fewer than 2 kv nodes matched, discard all kv candidates").
    Detect each block independently on its own evidence.
  - Assume a fixed number of records or blocks per page, or that all template
    nodes appear on every page / before a new record starts.
  - Let continuation handling discard a page's own header-band detections:
    continuation logic only applies to rows ABOVE the first header band; it must
    never remove or override blocks whose headers were found on the page.
  - Hard-code absolute y-coordinates or page numbers from the sample CSV.
  - Build blocks that span pages. Blocks are PAGE-LOCAL (this is validated after
    the script runs and a violation is a hard failure): a block emitted under
    page P must contain ONLY page-P words. When a table's rows continue onto the
    next page, emit a separate continuation block of the same node on that page
    (Step 4) — never absorb later pages' words into the header page's block, and
    never synthesize a global cross-page y-axis by stacking page heights.

A candidate with missing/extra fields, incompatible layout, ambiguous identity, or
insufficient evidence emits no block for that node. Do not coerce it into a known
template. A page with no accepted blocks must still appear with an empty blocks list.
Print concise rejection diagnostics with page, candidate node_id, and reason
(missing/extra field, unit mismatch, column order, hierarchy, layout, or ambiguity).
Do not add rejected regions to the block JSON under fabricated node IDs.

Before finishing generation, exercise the matching logic on sample-derived cases:
  - Accept the complete reference and translations/uniform scalings of it, with
    small coordinate jitter and harmless within-cell wrapping.
  - Reject removal/addition/duplication of a column, swapped columns, changed units,
    regrouped headers, and the same words scattered into unrelated columns/body rows.
  - Reject a table with an extra imperial mass column even when every expected
    metric field is still present.
  - Ensure a rejected table below a valid block is not swallowed by that block,
    and a rejected repeated header cannot be rescued by continuation handling.
These are offline checks; do not install packages or call external services.

STEP 5 — Output
  Write `<out_path>/block_separation.json`:
  {{
    "pages": [
      {{
        "page": <int>,
        "blocks": [
          {{
            "uid": <int>,
            "block_id": "<node_id>_<instance>",
            "type": "<table|kv>",
            "words": [
              {{"id": <int>, "x0": <float>, "top": <float>, "x1": <float>, "bottom": <float>, "text": <str>}},
              ...
            ]
          }},
          ...
        ]
      }},
      ...
    ]
  }}

  `block_id` is the template node_id followed by an underscore and the 0-based count
  of that node's blocks on the page (e.g. the second block of node 1 on a page is
  "1_1"). The counter resets on every page.

  `uid` is a globally unique integer that increments by 1 for every block across all
  pages, starting from 0.

  Print a summary line per page to stdout: page number, number of blocks found,
  total words assigned.

The script must also contain:

  if __name__ == "__main__":
      import sys
      main(sys.argv[1], sys.argv[2])

Use only the Python standard library (csv, json, os, re, collections). Do not install
any packages. Always open files with encoding="utf-8", errors="replace".
"""
