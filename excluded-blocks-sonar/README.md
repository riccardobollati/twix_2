# Excluded blocks sonar

The pipeline now runs this measurement after block separation on up to the
first 100 structured pages. It saves per-page counts, the mean coverage, and
the pages flagged by a relative drop threshold under
`<output>/artifacts/excluded_blocks_sonar/`. The dashboard's Missed sonar tab
lets you change the threshold without rerunning the separator, or replay a
chosen source-page range.

This utility counts words assigned to blocks versus words outside blocks on
each PDF page. It creates:

- stacked bars for words inside and outside blocks;
- a secondary-axis line for `words inside blocks / total page words`.

Words appearing in more than one block are deduplicated by their globally
unique OCR `id`. If an `id` is absent, the complete word object is used as the
fallback identity. Pages containing no words have no ratio value, so the line
has a gap there.

The PDF and `block_separation.json` must come from the same word-level OCR run.
The block JSON supplies the inside-block words; `pdfplumber.extract_words()` on
the PDF supplies the total page-word count.

Run it with the project environment:

```bash
env/bin/python excluded-blocks-sonar/excluded_blocks_sonar.py document.pdf block_separation.json
```

Choose the chart path with:

```bash
env/bin/python excluded-blocks-sonar/excluded_blocks_sonar.py document.pdf block_separation.json \
  --output excluded-blocks-sonar/result.png
```

The reusable entry point is `analyze_pdf_words`. It returns the per-page
measurements and the Matplotlib `Figure`:

```python
import sys

sys.path.insert(0, "excluded-blocks-sonar")
from excluded_blocks_sonar import analyze_pdf_words

measurements, figure = analyze_pdf_words(
    "document.pdf",
    "block_separation.json",
    output_path="word_coverage.png",
)
```
