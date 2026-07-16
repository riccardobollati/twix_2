def PROMPT_RECORD_SEPARATION(ocr_csv_path: str, template_json: str) -> str:
    return f"""Create a Python script called `record_separation.py` that groups OCR words from a multi-page document into logical records using the TWIX record separation algorithm (Section 4.1 of the TWIX paper).

Context:
- A sample OCR CSV file is at: {ocr_csv_path}
  Columns: id, page, x0, top, x1, bottom, text
  `id` is a globally unique integer assigned to each extracted word across the whole document.
- The document template defines the following blocks (nodes in the template tree T):
{template_json}

ALGORITHM (TWIX paper, Section 4.1 — Record Separation):

Definitions:
- A "row" is a list of words that are horizontally aligned (their vertical bounding boxes
  overlap on the y-axis: y_i1 <= y_j2 AND y_i2 >= y_j1).
- A "block" B is a sequence of consecutive rows.
- A node v in the template is "visited" by block B, denoted vis(v, B) = True, iff
  ALL fields of v appear in B.phrases, i.e. v.fields ⊆ B.phrases.
- A record is the SMALLEST consecutive block of rows that visits EVERY node in the
  template tree T at least once, considered in pre-order traversal.

The template nodes are processed in pre-order traversal order (the order given in
{template_json}, treating the list as a flat pre-order sequence of non-root nodes).

Record-separation procedure (per Example 6 of the paper):
1. Read all OCR words and sort globally by (page ASC, top ASC, x0 ASC) so words appear
   in natural reading order. The global `id` reflects this order — preserve it.
2. Group words into ROWS using horizontal alignment: scan words in order of id; merge
   word p into an existing row r iff p is horizontally aligned (y-overlap) with EVERY
   word already in r. Otherwise start a new row. A word that aligns with multiple
   open rows joins the FIRST one it aligns with. Rows are page-local (do not merge
   rows across pages).
3. Maintain a "current record" accumulator (a growing list of consecutive rows) and
   two tracking structures:
     - `matched_fields`: a set of (node_index, field_index) pairs — every template
       field token-sequence that has been matched anywhere in the current record so far.
     - `visited`: the set of node indices that are considered "covered" by the current
       record. Use a RELAXED threshold: a node v with F fields is covered when the
       number of its (v, field_index) pairs in `matched_fields` is ≥ max(1, ceil(F×0.5)).
       That is, a node is covered once at least HALF its fields have appeared — not all.

   CRITICAL — do NOT require ALL fields of a node to appear before calling it covered.
   Many real templates have nodes with 10–20+ fields, some of which are optional or
   rare. Requiring 100% coverage means `visited` never reaches `total_nodes` and no
   record boundary ever fires, collapsing the entire document into one record.
   The 50% threshold is the correct, robust implementation.

4. Scan rows in ascending order across the whole document. Before appending each row r:
     a. Compute `row_matched`: the set of (node_index, field_index) pairs whose field
        token-sequence appears in r's word texts (using the same n-gram scan as above).
     b. BOUNDARY CHECK — if ALL of the following are true:
          (i)  `len(visited) == total_nodes`   (every node has been covered, using
               the relaxed ≥ 50% threshold defined in step 3)
          (ii) `row_matched` has any element already in `matched_fields`
               (at least one field seen earlier in the current record reappears in r)
        THEN:
          - Close the current record (the rows accumulated so far form a completed record).
          - Start a NEW current record beginning with r.
          - Reset `matched_fields = row_matched` and recompute `visited` from scratch
            for the new record (using only row_matched).
          - Continue to the next row.
     c. Otherwise: append r to the current record, add `row_matched` into
        `matched_fields`, recompute `visited`.

   IMPORTANT — why field-level re-visit, not node-level:
   A template node may have many fields spread across many rows (e.g. an 11-field KV
   block). No single row will ever contain all 11 fields, so checking
   "v*.fields ⊆ phrases_of(r)" (node-level, single-row) would NEVER fire for such
   nodes and would collapse the entire document into one record.
   Instead, check at the FIELD level: if any individual field that was already matched
   in the current record reappears in the new row, that row is starting a new cycle
   through the template — close the current record and start fresh.

   ANCHOR-VALUE REFINEMENT — verify a boundary before closing a record:
   The field-revisit signal alone over-splits real documents, because field labels
   also reappear WITHIN one record: page headers reprint on every page of a
   multi-page record, and summary/disclaimer sections can echo label words.
   To make boundaries robust, use identifying VALUES, not just labels:
     - Inspect the sample CSV and pick one or two ANCHOR FIELDS: kv fields whose
       value identifies a record instance (an id / number / reference / date —
       e.g. "Contract No", "Invoice #", "Order Number"). Prefer fields whose value
       actually CHANGES between candidate records in the sample.
     - When the boundary check (4.b) fires, extract the anchor field's value near
       row r. If the anchor value EQUALS the current record's anchor value, this is
       the same record re-printing its header (page break) — do NOT close the
       record; keep accumulating (and do not re-arm the boundary on these rows).
     - If the anchor value DIFFERS (or the anchor label appears with a new value),
       close the record.
   If no usable anchor field exists in the template, fall back to the plain
   field-revisit rule.

   ADDITIONAL GUARDS:
     - Rows that match NO template field (legal text, disclaimers, signature lines,
       free-text notes) must never trigger, arm, or influence a boundary — treat
       them as inert content of the current record.
     - If two template nodes have IDENTICAL or near-identical field sets, a match on
       those shared fields is ambiguous between the nodes: never treat cross-matches
       between signature-duplicate nodes as evidence that the template restarted.
       (A node carrying a "signature_duplicate_of" key was already detected as the
       twin of that other node_id — treat their field matches as interchangeable.)
     - Verify on the sample CSV that the number of records produced is plausible
       (e.g. against the count of distinct anchor values) and print both numbers.

5. After scanning all rows, flush the final accumulated record (even if not all nodes
   were visited — append whatever remains as the last record).
6. Assign every word in the document to exactly one record, based on which record's
   rows contain it.

Field-matching notes:
- Compare field strings to word `text` case-insensitively and after stripping
  surrounding whitespace and trailing punctuation (`:`, `.`, `,`).
- A multi-word field (e.g. "Type of Complaint") is considered present in B.phrases iff
  its constituent tokens appear as a contiguous run of words within some row of B,
  in order. (Implement a simple n-gram scan over each row's word texts.)

The script must define a main function with this exact signature:
  def main(inputpath: str, out_path: str) -> None
  - `inputpath`: path to the OCR CSV file to process.
  - `out_path`: path to the folder where `record_separation.json` must be written.

Output: write `<out_path>/record_separation.json` with this exact structure:
   {{
     "records": [
       {{
         "record_id": "<index>",
         "pages": [
           {{
             "page": <int>,
             "y_start": <float>,
             "y_end": <float>,
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
   Each record contains one entry per page it spans. `y_start` / `y_end` are the
   top / bottom coordinates of the record's words on that page. Words within each
   page entry must be sorted by their global `id`.

Print a summary to stdout: one line per record showing its id, the page range it
spans, and which template node indices were visited inside it.

The script must also contain:
  if __name__ == "__main__":
      import sys
      main(sys.argv[1], sys.argv[2])

Use only the Python standard library (csv, json, os, collections). Do not install any packages.
Always open files with encoding="utf-8", errors="replace".
"""
