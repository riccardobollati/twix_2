# TWIX 2 local dashboard

Start the interface from the repository root:

```bash
python3 UI/app.py
```

Open `http://127.0.0.1:8765`. Upload a PDF, choose the run settings, and the
dashboard will run the existing `run.py` pipeline locally. Its seven-stage view
follows page clustering, template inference, block separation, missed sonar, deterministic record separation,
data extraction with train evaluation, and independent test evaluation. The
template and final JSON are shown beside their PDFs. Final results default to
logical records from `pipeline/grouped.json`, with a Blocks view for the flat
`pipeline/extracted.json`. Block and record overlays come from the unseen test
split when evaluation is enabled and are refreshed after the finished pipeline
runs on that split. Train holdout and
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
GLM 5.3 Flash, Qwen3.8 Flash, and the paid and free Qwen3.8 27B routes through
OpenRouter. The clustering stage lists table and KV clusters separately, their
member pages, representative overlays, and the page selected for each cluster.
It also shows the first ordering pages, additional cluster samples, training
pages, and independent evaluation pages. Template inference is the following
stage.

Failed runs can resume from block separation using their saved template and
page plan. If a run stops during training evaluation (for example, because
reference-model credits run out), **Resume** reuses its complete generated
pipeline and retries training evaluation before continuing to test extraction.
Existing model selections and the manual test-ground-truth setting are retained.

For a UI run with evaluation enabled, select **Paste final test ground truth**
to skip the OpenRouter call for the held-out test ground truth. After test
extraction, the run pauses at **Awaiting ground truth**. Open its Ground truth
tab, copy the `$twix-ground-truth` prompt if you want Codex to prepare the
held-out page images and exact OCR, then paste the resulting `{"blocks": [...]}`
JSON and identify the model or source used. The UI validates page numbers,
node IDs, field names, and value types before local scoring. An invalid paste
can be corrected and retried. When the skill finds a genuinely missing structured
table/KV region, it can include an optional `template_extension` with the complete
expanded template and page evidence for every added node. Original physical nodes
remain unchanged. The UI saves `pipeline/evaluation_template.json` and groups both
ground truth and raw extraction with it, so missed nodes affect evaluation and
subsequent rescoring. The learned templates and scripts remain unchanged; held-out
evidence is used only for evaluation. Train ground truth still calls the selected
reference model; choose a non-GPT reference model if you also want to avoid
GPT-5.4 for that phase. The template model remains independent.

For an explicitly grouped reference, paste `{"records": [...]}` instead of
`{"blocks": [...]}`. Each record has a unique non-empty string `record_id` and a
non-empty `data` list of physical blocks in document order. The same page, node,
field, and value validation applies. Include `template_extension` alongside
`records` when the reference uses additional layouts. This format preserves the
submitted reference boundaries and the saved TWIX candidate grouping; it records
`manual_ground_truth.grouping = "explicit_records"` in metadata. Flat submissions
continue to group both sides automatically with the evaluation template.

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
