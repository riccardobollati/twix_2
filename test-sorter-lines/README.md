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
