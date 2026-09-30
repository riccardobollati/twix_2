# TWIX 2 local dashboard

Start the interface from the repository root:

```bash
python3 UI/app.py
```

Open `http://127.0.0.1:8765`. Upload a PDF, choose the run settings, and the
dashboard will run the existing `run.py` pipeline locally. Its six-stage view
follows template inference, block separation, missed sonar, deterministic record separation,
data extraction with train evaluation, and independent test evaluation. The
template and final JSON are shown beside their PDFs; block and record overlays
come from the unseen test split when evaluation is enabled. Train holdout and
test evaluation are separate controls:
holdout defaults to zero, while evaluation defaults to twenty pages and enabled.
All generated files remain available in the artifact drawer.

Missed sonar runs the generated separator on up to 100 source pages starting
at the detected structured-data page. Its tab plots words inside and outside
blocks with the coverage ratio, flags pages whose coverage is a configured
relative percentage below the scanned-page average, and renders an overlay
when a flagged page is selected. Adjusting the threshold updates the list
immediately. Use Choose page range and Replay scan to analyze another range;
the range controls show the PDF's maximum source page.
Older runs with a saved source PDF and `pipeline/block_separation.py` show
**Run sonar on this run** in the same tab. This uses the saved separator and
the metadata source start page, so the extraction pipeline does not rerun.

After a successful UI run, the dashboard calls the reusable single-document
renderer from `viz_batch_results.py`. The final-results Stats tab shows summary
metrics and the generated overview, phase-cost, and token-breakdown plots. These
files are stored under `UI/runs/<run-id>/artifacts/stats/`; plot-generation
errors are reported in Logs without changing a successful pipeline into a
failed run.

The Model routing panel can override the `.env` easy and hard models for one
run. Enable “Use one model for every non-reference phase” to pass
`--no-routing` and use the selected pipeline model for front-matter detection,
block separation, every extractor, and repair. Template inference uses its
own selected model; train and test ground truth use the reference model.
Both default to GPT-5.4. The template selector also offers DeepSeek V4/V4.1,
GLM 5.3 Flash, and Qwen3.8 Flash through OpenRouter. The template panel shows
the source pages selected by deterministic table and KV clustering, including
the first three ordering pages and later structural variants.

With routing enabled, the selected hard model always generates the block
separator and performs difficulty classification; it also handles hard block
extractors and repair. The easy model is reserved for blocks classified easy.

The dashboard always enables `--save-log`, so detailed OpenCode session logs
are retained under each run's `debug/` folder. OpenCode work is limited to two
sessions globally across simultaneous dashboard, CLI, and batch runs, with a
fresh temporary OpenCode database for every attempt. Completed log entries use
a green check for success and a red cross for failure; failed extractor attempts
are retried up to three times and retain their diagnostics in the debug logs.

When the repository virtual environment exists at `env/bin/python`, the
dashboard uses it automatically to execute the pipeline. This lets the UI be
started with another local Python interpreter without losing TWIX dependencies.

Runs are stored under `UI/runs/`. This directory can contain PDFs, OCR, model
outputs, and debug material; treat it with the same care as other generated
artifacts. The server only binds to `127.0.0.1` by default.
