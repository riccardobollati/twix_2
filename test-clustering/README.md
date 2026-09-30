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

## Clustering viewing 2

The second comparison uses detected table headers and PDF border geometry.
It tests header-heavy, balanced, and structure-heavy methods with complete or
average linkage. A page without usable red borders can derive a body structure
signature from aligned words **below a detected header**. Pages with usable
borders do not use word alignment. Pages with neither a header nor usable
borders are listed as skipped, rather than silently assigned to a cluster.
For bordered tables, each detected table has its own normalized column-rule
sequence. A strong change in one table can no longer be hidden by an unchanged
neighboring table. An extra column can still match when most rules align. This
strong rule is used when the scan has a small, stable set of header families;
the detected rule counts must also be consistent across pages. Documents with
many competing families or unstable rule counts use the ordinary page-wide score.
When the initial scanner misses a recurring header or mistakes many body rows
for headers, the script finds a repeated, horizontally aligned header band in
the upper part of the pages. This still uses only page text and coordinates;
it makes no model calls.

```bash
env/bin/python test-clustering/run_viewing_2.py --list
env/bin/python test-clustering/run_viewing_2.py
env/bin/python test-clustering/verify_results.py --view 2
```

Open `test-clustering/viewing-2/index.html` and its linked report
`test-clustering/viewing-2/report.md`. Use `--only id_05 new_04` for a focused
run and `--force` to regenerate. The viewer opens on `balanced_complete`; its
menu also shows stricter and broader strategies. The first comparison stays available at
`test-clustering/index.html`.

### Independent KV clusters

The same viewing-2 run also writes `kv_clusters.json` and `kv.html` for each PDF.
The document viewer links between **Table clusters** and **KV clusters**; their
eligible pages and assignments are computed separately. The corpus index shows
both counts, and `kv_report.md` summarizes the KV pass. Use `--force-kv` to
rebuild KV results while reusing valid table results.

KV discovery starts with the deterministic structure scan. A family needs
stable key phrases on distinct pages and different nonempty values in at least
one stable slot. Key positions and key-to-value offsets are compared relative
to the group, so the group may move on the page. Value *text* does not affect
similarity. Missing or extra KV groups get a limited penalty. Table headers,
numeric table rows, and short page-edge metadata groups are excluded. Pages
without confirmed KV evidence are listed in the KV viewer rather than forced
into a cluster. Blue boxes mark keys and green boxes mark value slots in the
representative PNGs. The scan cap remains 500 pages.
