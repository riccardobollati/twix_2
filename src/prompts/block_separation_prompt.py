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

STEP 3 — Detect block header rows
  A block's header row is the row (or contiguous group of rows) where the block's
  field labels appear. Detect it as follows:

  For "table" nodes:
    A header row is one where at least HALF of the node's field labels appear as
    normalised tokens within the row's words. Multi-word field labels (e.g.
    "Project Description") must appear as a contiguous token sequence within the row.
    The entire contiguous group of rows where these header tokens are spread counts as
    the header band.

  For "kv" nodes:
    A header row is one that contains at least ONE key label from the node's fields as
    a complete normalised token or contiguous token sequence.

  Normalise before comparing: lowercase, strip leading/trailing punctuation (`:`, `.`,
  `,`, `#`). Comparison must be case-insensitive.

STEP 4 — Assign content rows to blocks
  Process pages in ascending page order (the script can therefore carry state from one
  page to the next). After locating the header row(s) of each block on the page:
    - The block's words are all words in the header band PLUS all rows that follow
      below the header band, down to (but not including) the header band of the next
      block on the page (in top-to-bottom order), or the end of the page if no further
      header band exists. Noise rows stripped in Step 2 are simply skipped — they are
      NOT part of any block, but they never terminate one: rows below a stripped noise
      row still belong to the block.
    - If two template nodes have overlapping vertical extents on the page, assign each
      word to the node whose header band it is closest to (by vertical distance).
    - CONTINUATION BLOCKS: rows that appear ABOVE the first header band on a page (or
      on a page with no header band at all) and that column-align with the previous
      page's last block are a continuation of that block's template node. Emit them as
      a block of that node on the current page (a continuation block has no header band
      of its own). Track the node and column layout of the last block emitted on the
      previous page to detect this.

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
