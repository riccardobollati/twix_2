# PDF line overlays

Detect PDF vector lines, stroked rectangle edges, and thin filled rectangular
rules. Pages with no such vector lines use a raster fallback for long horizontal
and vertical dark runs. The page image gets a translucent grey veil, then the
detected lines are drawn in red above it.

```bash
env/bin/python test-sorter-lines/detect_lines.py data/id_05/id_05.pdf
env/bin/python test-sorter-lines/detect_lines.py data/id_05/id_05.pdf --pages 1,3-5
env/bin/python test-sorter-lines/detect_lines.py data/id_24/id_24.pdf --all-pages
env/bin/python test-sorter-lines/detect_lines.py data/id_05/id_05.pdf data/new_04/new_04.pdf
test-sorter-lines/run_all.sh --list
test-sorter-lines/run_all.sh
```

By default, only pages 1-30 are processed. Use `--pages` for a custom selection
or `--all-pages` for the complete PDF. Rerunning replaces the previous page
overlays in that PDF's results folder.

`run_all.sh` processes source PDFs directly in `data/` and `data/<document>/`,
skipping deeper generated PDFs. It uses the first-30-pages default and accepts
the detector's options, for example `test-sorter-lines/run_all.sh --dpi 120`.
Set `PYTHON`, `DATA_DIR`, or `OUTPUT_ROOT` to override the default paths.
The batch runner uses the document folder name for results, so PDFs with the
same filename in different document folders do not overwrite one another.

Each PDF writes to `test-sorter-lines/<pdf-stem>/results/`, with
`page_0001_lines.png` and a `lines.json` report containing coordinates in PDF
points. Results are ignored by Git. Use `--min-length`, `--dpi`, and
`--grey-alpha` to adjust detection and display.

## Compare page clusters before template inference

```bash
env/bin/python test-sorter-lines/cluster_pages.py id_05 new_04
env/bin/python test-sorter-lines/cluster_pages.py data/id_05/id_05.pdf --max-pages 20
```

`cluster_pages.py` reads PDF vectors with the same line detector (red), connects
horizontal and vertical alignments of word **centers** inside candidate table
regions (green), and connects the centers of repeated header labels from
`scan_structure.py` in blue (with header text included in scoring). A matching
`test/scan_results/<name>/scan.json` is reused; otherwise
the scanner runs in memory. It does not use a template or make model calls.
The older `clustering.py` idea compares exact recurring text rows; the scanner
supplies its repeated-header signal here while allowing added or missing
header fragments.

The script compares complete, average, and single hierarchical linkage with
two fixed channel weightings. `tight_*` gives headers 40% weight to distinguish
changed fields. `structure_*` gives red/green layout 95% weight so small
header changes can stay together. These are exploratory thresholds; inspect
the clusters and representative overlays before using them for template-page
selection. All cluster assignments and metrics are saved to
`test-sorter-lines/clustering_results/<name>/clusters.json`. The comparison
table is `test-sorter-lines/clustering_results/report.md`, and representative
`page_####_features.png` images show red, green, and blue features on the PDF.
Outputs are ignored by Git.

The medoid page of each cluster is a candidate for early template inference.
The report includes how many clusters the first five pages would cover.
Pages with no words or table-structure evidence are listed separately and do
not become template sampling candidates.
Complete linkage avoids joining groups through one intermediate page; single
linkage shows how permissive chaining behaves. A page can contain multiple
tables, so these are page groups, not individual extracted table instances.
Decorative rules and diagrams can contaminate red features; missed scanner
headers can weaken blue features. The comparison is a diagnostic, not a
ground-truth accuracy score.
