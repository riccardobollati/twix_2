# CLAUDE.md

This file provides a compact Claude Code entry point. `AGENTS.md` is the
authoritative contributor guide; read it before changing contracts or pipeline
behavior and keep this summary consistent with it.

## What this is

Thesis research project: an agentic pipeline that learns a reusable structured
extractor from a small sample of template-related PDFs, then scores held-out
extraction against reference-model-generated ground truth. LangGraph
orchestrates the phases; configurable models are called through OpenRouter; the
`opencode` CLI writes the actual Python separation/extraction scripts.

## Commands

```bash
source env/bin/activate                  # local virtualenv (python 3.14); deps in requirements.txt
python run.py path/to/document.pdf --no-eval  # direct input without test scoring
python run.py <name> --eval              # train/test split + ground truth + scoring (the default)
python run.py <name> --eval --pages 5 --train-threshold 0.9   # with OpenCode repair loop on low train accuracy
python api.py --host 127.0.0.1 --port 8000  # async PDF upload/status/result API
python src/run_batch.py --jobs 2 --eval  # run all data/ folders with a terminal dashboard (or pass folder names)
python single_eval.py ...                # re-run GT + eval standalone on an existing run
python fix.py <doc_id>                   # re-run test GT + eval, patch metadata.json in place
python rescore.py ...                    # re-score saved artifacts without model calls
python -m unittest discover -s tests -v  # offline regression suite (no API calls)
python -m compileall -q api.py run.py fix.py rescore.py single_eval.py src
```

There is no linter. The `tests/` directory contains offline `unittest` regressions. Visualization helpers: `viz_batch_results.py` (batch accuracy tables), `viz_blocks.py` / `viz_bb.py` (draw block/word bboxes on pages), `viz_agent_log.py` (OpenCode session logs).

Requires `OPENROUTER_API_KEY` in `.env` (never print, replace, or commit it) and
OpenCode on `PATH` (DeepSeek recommends 1.14.24 or newer).
`LLM_PROVIDER=openrouter` uses `src/models/openrouter.py`. Direct calls,
easy/hard OpenCode generation, and the reference/ground-truth model are
independently configurable; see `AGENTS.md` and `.env.example` for the current
variables.

## Architecture

The LangGraph state machine is defined in `src/agents/graph.py`:

```
detect_beginning → environment_creation → template_inference
→ block_separation → run_block_separation → block_artifacts → block_samples → block_difficulty_router
→ block_extraction_scripting → collect_extraction_scripts → train_eval
→ [train_repair loop] → extraction
→ [eval_mode: ground_truth → eval] → save_metadata
```

Key ideas that span multiple files:

- **Front-matter detection.** `detect_beginning.py` scans physical PDF pages in
  chronological OCR-layout batches to find the first actual structured-record page.
  `environment_creation.py` starts both PDF and OCR splits there and renumbers
  split-local pages from 1, so covers and tables of contents never reach template
  inference or the generated pipeline.
- **Two kinds of model output.** Direct OpenRouter calls return JSON (beginning
  detection, template inference, ground truth, and difficulty routing).
  Separation and extraction are instead **generated Python scripts**: prompts
  in `src/prompts/` instruct OpenCode (wrapped by `src/models/opencode.py`) to
  write `block_separation.py` and per-block
  `extraction_scripts/<node_id>.py`. Generation is retried at most three times.
  Separation scripts run with a 300-second timeout and JSON validation.
  Extractor targets are deleted before every attempt and must parse with exactly
  one synchronous top-level `extract` function accepting one positional
  argument, so stale or invalid scripts cannot count as success. All OpenCode
  calls share a cross-process two-slot limit and use a fresh temporary database
  per attempt; database-lock output is a transient failure, and timeout logs
  preserve partial diagnostics and usage.
- **Template contract.** `template.json` nodes have `node_id`, `type` (table/kv), `fields`, `bid`, `complexity`. Field names within a node MUST be unique — grouped column headers are prefixed (`"Balance - Total"`); `template_inference.py` validates and dedups this because duplicate dict keys silently collapse everywhere downstream. Template `metadata` regex patterns are consumed by the block-separation prompt to strip page headers/footers (never strip unmatched bottom rows — they are usually data).
- **ID scheme.** `block_id = "<node_id>_<instance>"`; consumers recover the node with `block_id.rsplit("_", 1)[0]`. Template lookups key on `node_id`, not `bid`.
- **Train/test.** `environment_creation` splits the PDF under
  `<output>/agent_space/{train,test}/`. Separation and extractor samples exclude
  the configured train holdout; `train_eval` still runs the finished pipeline
  on every train page. Final extraction applies the portable pipeline to test
  pages (or the cropped train PDF with `--no-eval`). `src/pipeline.py` emits flat
  page-local blocks, and `src/record_grouping.py` writes deterministic logical
  records to `grouped.json`.
- **Ground truth** is itself LLM-generated: `src/agents/gt_common.py` is the single shared generator (parallel windowed calls returning flat page-local blocks) used by both `train_eval.py` (window 5) and `ground_truth.py` (window 20). Train GT uses OCR layouts; final test GT uses page images plus exact OCR.
- **Eval** (`src/eval/`): converts supported shapes into records with flat nodes
  (`convert.py`), uses Hungarian record and node alignment (`align.py`), scores
  tables with order-preserving row alignment (`content.py`), and renders
  actionable `eval.txt` diagnostics (`render.py`). For the current flat nodes,
  headline accuracy is 50% typed NodeMatchF1 and 50% size-weighted ContentScore;
  structure is not applicable and RecordMatchF1 is diagnostic only. Load every
  scored JSON through `convert.load_json_dedup` so repeated keys survive as
  `key`, `key#2`, and so on.
- **Repair loop.** Train evaluation always runs. A score below
  `--train-threshold` or degenerate `pipeline_health.json` triggers
  `train_repair.py`, then train evaluation re-runs. Setting the threshold to
  zero disables only this repair loop. Retries are bounded by
  `--train-max-retries`, and the graph recursion limit is derived from it.
- **Costs.** Every node returns `cost` entries (phase, duration, tokens via `src/agents/_usage.py`); `save_metadata.py` writes them to `<output>/metadata.json`.

## Per-document output layout

`<output>/`: `template.json`, `metadata.json` and `metadata/cost.json`, train and
optional held-out overlays under `artifacts/`, the split workspace under
`agent_space/`, the portable generated bundle under `pipeline/` (including
`extracted.json`, `grouped.json`, `pipeline_health.json`, train evaluation, and
optional held-out ground truth/evaluation), and attempt-specific OpenCode logs
under `debug/` with `--save-log`. See `AGENTS.md` for the complete layout.

## Gotchas

- Scorer behavior has changed repeatedly. Never compare historical accuracies
  without running `python rescore.py --dry-run ...` first and recording the
  resulting `eval_version`.
- The `data/` directory is gitignored and large; don't glob or grep it blindly.
- Generated block IDs are `<node_id>_<page-local-instance>` and must be split
  with `rsplit("_", 1)`; `uid` is the cross-document physical-block identity.
- Blocks are page-local. Do not merge cross-page table continuations or discard
  unmatched bottom rows.
- `.agents/skills/` contains the `/caveman` prompt material referenced by `--caveman`.
