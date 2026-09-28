# Deterministic structure scan

`scan_structure.py` is a standalone experiment for discovering recurring table
header bands and KV groups **before template inference**. It reads PDF word text
and geometry with `pdfplumber`. It does not load saved templates, separators,
ground truth, credentials, or models. It does not change the TWIX graph.

## Run

From the repository root, using the existing requirements:

```bash
env/bin/python test/scan_structure.py id_05 new_04
env/bin/python test/scan_structure.py data/new_04/new_04.pdf --overlay-pages 1,9,19
env/bin/python test/scan_structure.py id_05 --start-page 1 --end-page 10 --min-pages 3
test/scan_all_structure.sh --list
test/scan_all_structure.sh
```

The default output is `test/scan_results/<PDF basename>/`. Use `--out-dir` to
change its parent directory, `--no-plots` for JSON only, and `--gap-factor` to
adjust how much wider than normal word spacing a phrase boundary must be.
The batch loop scans PDFs directly in `data/` and `data/<document>/`; it skips
deeper generated PDFs such as `agent_space/train/*.pdf`. It saves each document
to `test/scan_results/<document>/`, so same-named PDFs in different document
folders do not overwrite each other. `--list` previews the selected files;
other arguments are passed to each scan, for example
`test/scan_all_structure.sh --min-pages 3`. Set `PYTHON` to use an interpreter
other than `env/bin/python`. The results directory is ignored by Git.
Each PDF scan reads at most 500 pages, starting at page 1 by default or at
`--start-page`; a larger requested range is capped and reported in the terminal.
Remaining thresholds are collected in the `Rules` dataclass and saved in the JSON.
The script rescans the selected PDF pages on each run; there is no hidden cache.

## Decision rules

1. Find unusually large, entirely empty vertical gutters. These split page
   spreads into separate regions, so two side-by-side tables are not joined.
   A gutter must also be larger than the usual projected column gaps.
2. Group words into horizontal rows using their vertical centers. Split each
   row into phrases at gaps measured relative to ordinary spaces, character
   width, and word height. A trailing colon also ends a phrase. This normally
   leaves paragraph lines as a single phrase.
3. Build an index of normalized phrases and the **distinct physical pages** on
   which they occur. Case, Unicode compatibility forms, and whitespace are
   normalized. Repetition several times on one page alone is insufficient.
4. Find KV groups: multiple colon-ended labels across one row, or at least two
   vertically stacked labels whose left edges and value columns align. Preserve
   blank fields. Indented value continuations and a colon inside the value can
   be retained. Confirm a group only when it recurs across pages and at least one
   field has different nonempty values observed across those pages. Constant
   fields can belong to an otherwise changing group.
5. Find table headers: multiple separated text phrases with substantial
   cross-page repetition on the same row, optionally continuing over several
   nearby lines that each have multiple horizontal labels. A short fragment
   aligned to an existing header column may wrap onto a line by itself; a
   single-column paragraph cannot extend the header band.
   Reject rows dominated by numbers, dates, or colon pairs; recognize compact
   letter-plus-exponent unit labels. Require a nearby row with data aligned to
   multiple header anchors. A comment cell under the last column may start
   before the other cells, so the body check can look past such lines. A data
   phrase cannot count for two header anchors.
   Previously confirmed KV groups are excluded from table-header candidacy.
6. Group compatible occurrences using ordered phrase matching plus relative
   geometry of their matched labels. By default, at least 72% of the larger
   phrase list must match, and 80% of matched positions must agree within 7.5%
   of the matched span. Limited missing/extra phrases are allowed; they get
   separate occurrence counts. Representatives stay fixed, preventing a chain of
   weak matches from progressively broadening a family.

Processing order and tie-breaking are deterministic. Family IDs are local to a
run; changing the scanned page range may change them. Each PDF page is read once,
and words are sorted within its regions. Matching uses a label index and compares
candidate groups with compatible family representatives, rather than comparing
all pairs of pages. Cost and memory still grow with document size and structural
diversity; the scanner keeps the selected pages' words/rows in memory.

## Outputs and interpretation

| File | Meaning |
| --- | --- |
| `table_001.png`, etc. | One word-relative spatial heatmap per table-header group |
| `kv_001.png`, etc. | One word-relative spatial heatmap per KV group, showing key words without their values |
| `scan.json` | Rules, page sizes, exact individual word boxes, group membership, occurrence counts, table-body alignment evidence, and unconfirmed candidate groups |
| `page_0001_overlay.png`, etc. | Optional source-page overlays: red header bands, green KV groups |

Each heatmap uses **word-relative PDF coordinates**. For every occurrence of
the group, the leftmost word's x and topmost word's y define `(0, 0)`. The
individual header/key word boxes are shifted by that origin, then occurrences
are overlaid. A group can therefore move on the page without spreading its
heat. Distances and word sizes remain in PDF points; they are not rescaled.
The color at a location is the number of **group occurrences** covering it,
including separate occurrences on the same physical page. Overlapping words
within one occurrence contribute only once. One file is made per confirmed
group, even if its pages have different sizes.

The heatmap is calculated by sweeping the exact edges of word bounding boxes
and drawing their intersections as rectangles. There are no histogram
bins, smoothing kernels, rounding, or coarse spatial cells. Spaces between words
remain empty in the PNG.
White means no matching detection at those relative coordinates, which may be
a detector miss. A change in spacing or layout within the group spreads its
heat across the corresponding relative positions.

The `heatmap_counts` arrays in JSON still retain diagnostic label-by-page
occurrence counts; they do not drive the spatial plots. Wrapped and grouped
headers are retained as spatial phrases containing the original word boxes;
the script does not reconstruct canonical, hierarchical field names. A table
group identifies a recurring header layout, not an individual physical table
spanning many pages. The JSON retains absolute source coordinates so the origin
can be recalculated; version 3 marks the relative-heatmap output.

Rerunning a document replaces
its JSON and plots and removes obsolete plot filenames owned by this script.
Generated results are ignored by Git; source PDFs and existing run artifacts
are read only.

## Initial sample results

Scanned all pages of the two requested PDFs, without their existing templates.

| PDF | Pages | Table groups | KV groups | Observed coverage |
| --- | ---: | ---: | ---: | --- |
| `id_05` | 20 | 3 | 1 | 75 occurrences of each table-header group and of the complainant KV group |
| `new_04` | 23 | 3 | 1 | First left-table layout on pages 1-8; second on 9-23; properties headers and KV group on all 23 pages |

The properties group also records header wording/wrapping changes on pages
19-23. These are inspection results, not measured precision/recall against a
human-labeled corpus. Selected overlays were checked on pages 1, 9, and 19.

Saved plot shortcuts after running the command above:

- `id_05`: [main table](scan_results/id_05/table_001.png),
  [complaints](scan_results/id_05/table_002.png),
  [officers](scan_results/id_05/table_003.png),
  [KV labels](scan_results/id_05/kv_001.png).
- `new_04`: [first dimensions layout](scan_results/new_04/table_001.png),
  [properties](scan_results/new_04/table_002.png),
  [second dimensions layout](scan_results/new_04/table_003.png),
  [KV labels](scan_results/new_04/kv_001.png).

## Optional two-stage structure purge

After the deterministic scan, the first purge stage keeps only confirmed
table-header and KV groups with at least **10 occurrences across the scanned pages**.
This counts occurrences, not distinct pages: two copies on one page count twice.
By default, the second stage reviews each occurrence of an eligible table-header
group with the locally running Laya playground. Eligible KV groups remain in the
purged result without a Laya call because the current Laya question is specific
to table headers. Use `--no-laya` to save the frequency-filtered table and KV
groups without any model calls:

```bash
env/bin/python test/purge_structure.py id_05
env/bin/python test/purge_structure.py id_03 --threshold 0.6
env/bin/python test/purge_structure.py id_07_upscaled --min-occurrences 10
env/bin/python test/purge_structure.py --all
env/bin/python test/purge_structure.py --all --no-laya
```

The default endpoint is `http://127.0.0.1:8770/api/predict`; use `--url` if it
runs elsewhere. `--min-occurrences` changes the frequency gate; its default is
10. `--no-laya` needs only `scan.json`; it does not read the PDF or OCR and does
not require a running Laya server. When it replaces previous Laya output, that
run is moved under `purged/previous_laya/`. With Laya enabled, each eligible
header occurrence has its bounding box expanded
by **10% of the physical page width and height on each side**, clips the crop to
the page, and takes every OCR word intersecting that crop. It reads the saved
`ocr_extraction_full.csv` when available and falls back to the source PDF's
text layer for missing pages. The Laya `noul` request includes the target labels
and all crop words with page coordinates rounded to the nearest PDF point. The
target and nearest rows appear first because the checkpoint can truncate long
requests. Larger crops are split into multiple requests, each repeating the
target and covering a different part of the context; saturated requests are
split again. The retained score is the highest probability from complete
chunks. If even the smallest chunk is truncated, the occurrence is marked
inconclusive and retained. The crop PNG is saved for inspection; Laya's API
accepts text, not the image. The default checkpoint is `typed-decisions`, which
has a larger context than `english`; `--model` can change it.

The default probability threshold is 0.3, chosen conservatively while this
classifier is uncalibrated. An occurrence whose request appears truncated is
marked inconclusive and retained regardless of score. Accepted occurrences remain in a
table group only when they still appear on the scan's minimum number of distinct
pages. All purge outputs are saved in `test/scan_results/<document>/purged/`,
separate from the original `scan.json`:

| File | Meaning |
| --- | --- |
| `purged_scan.json` | Frequency decisions for every table/KV group and retained groups; includes Laya decisions and usage when enabled |
| `purged_table_001.png`, etc. | Word-relative heatmaps of retained header occurrences |
| `purged_kv_001.png`, etc. | Word-relative heatmaps of KV groups that passed the frequency gate |
| `purged_snapshots/*.png` | PDF crops with the candidate header outlined in red when Laya is enabled |
| `purged_requests.jsonl` | Exact text requests sent to Laya, only when Laya is enabled |

`--no-images` skips crop PNG generation. Laya's probability is an experimental
classifier score; no precision or threshold calibration has been measured on
this corpus. `possible_context_truncation` flags responses whose token usage is
near the checkpoint limit, but Laya may silently truncate some requests. A
small live pilot found real headers and a likely page title with very similar
scores, so review saved decisions before using the purged set as training input.
The purge stage is standalone and does not alter template inference or extraction.

## Limits and next integration boundary

- This uses embedded PDF text; it does not perform image OCR. Empty text pages
  are reported rather than treated as evidence that no structure exists.
- Spacing, recurrence, and alignment are heuristic evidence. Sparse two-column
  tables and KV lists can be ambiguous. Constant KV groups stay unconfirmed.
  Repeated titles can still be false positives if their geometry resembles data.
- A structure unique to one page cannot be confirmed as repetitive. Headerless
  continuation pages, predominantly numeric headers, rotated text, and severe
  OCR errors need additional rules. Font-size changes and irregular row heights
  can fragment a band. Gutters crossed by full-width text may not be detected.
- Similarity uses exact normalized phrases, not synonyms. Large changes,
  reordered columns, or internal insertions that move many anchors can split
  related layouts into different groups. Table-body alignment validates local
  structure but is not a complete comparison of data-row schemas.
- Preserve original physical page numbers. When this is later used to select
  training examples, constrain discovery to the designated training pool;
  these full-document scans are exploratory diagnostics, not held-out evaluation.

Offline regression tests use synthetic word boxes and never depend on `data/`:

```bash
env/bin/python -m unittest tests.test_structure_scan -v
```
