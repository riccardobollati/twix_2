# Browse deterministic table clustering

Run the pre-template page clustering over source PDFs directly in `data/` and
`data/<document>/`, excluding nested generated PDFs:

```bash
env/bin/python test-clustering/run_all.py --list
env/bin/python test-clustering/run_all.py
env/bin/python test-clustering/verify_results.py
```

The batch is resumable. `--force` regenerates outputs; `--only new_04` runs one
document. Each PDF is scanned through page 500 by default, following the
existing structure-scan limit. `--max-pages` can lower this limit. Scan coverage
is shown in the HTML; PDFs longer than the limit are explicitly marked.

Open `test-clustering/index.html` locally. Choose a document and clustering
method, inspect each cluster's representative red/green/blue overlay, click a
page number to open its PDF page, or search for a page number. `report.md` gives
the numerical comparison. Each document's `clusters.json` stores assignments,
metrics, and per-page feature counts. Generated HTML, JSON, PNG, and logs are
ignored by Git; the scripts and this README are source files.
