# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Thesis research project: an agentic pipeline that extracts structured data (tables / key-value blocks) from PDF documents generated from a common template, then scores the extraction against LLM-generated ground truth. LangGraph orchestrates the phases; DeepSeek reasons over layout-preserving OCR; the `opencode` CLI writes the actual Python separation/extraction scripts.

## Commands

```bash
source env/bin/activate                  # local virtualenv (python 3.14); deps in requirements.txt
python run.py path/to/document.pdf       # direct input; writes outputs/<document>/
python run.py <name>                     # run pipeline on data/<name>/<name>.pdf (template + extraction)
python run.py <name> --eval              # train/test split + ground truth + scoring
python run.py <name> --eval --pages 5 --train-threshold 0.9   # with OpenCode repair loop on low train accuracy
python src/run_batch.py --jobs 2 --eval  # run all data/ folders with a terminal dashboard (or pass folder names)
python single_eval.py ...                # re-run GT + eval standalone on an existing run
python fix.py <doc_id>                   # re-run test GT + eval, patch metadata.json in place
python -m unittest discover -s tests -v  # offline regression suite (no API calls)
```

There is no linter. The `tests/` directory contains offline `unittest` regressions. Visualization helpers: `viz_batch_results.py` (batch accuracy tables), `viz_blocks.py` / `viz_bb.py` (draw block/word bboxes on pages), `viz_agent_log.py` (OpenCode session logs).

Requires `OPENROUTER_API_KEY` in `.env` (never commit it; it is gitignored) and OpenCode on `PATH` (DeepSeek recommends 1.14.24 or newer). `LLM_PROVIDER=openrouter` uses `src/models/openrouter.py`; direct OpenRouter and OpenCode model IDs remain independently configurable for experiments.

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
- **Two kinds of model output.** DeepSeek calls return JSON directly (beginning detection, template inference, ground truth, difficulty routing). Separation and extraction are instead **generated Python scripts**: prompts in `src/prompts/` instruct OpenCode (wrapped by `src/models/opencode.py`) to write `block_separation.py` and per-block `extraction_scripts/<node_id>.py`; the agents then run those scripts via subprocess (300s timeout) and validate their JSON output, feeding failures back into a retry loop (max 3 attempts).
- **Template contract.** `template.json` nodes have `node_id`, `type` (table/kv), `fields`, `bid`, `complexity`. Field names within a node MUST be unique — grouped column headers are prefixed (`"Balance - Total"`); `template_inference.py` validates and dedups this because duplicate dict keys silently collapse everywhere downstream. Template `metadata` regex patterns are consumed by the block-separation prompt to strip page headers/footers (never strip unmatched bottom rows — they are usually data).
- **ID scheme.** `block_id = "<node_id>_<instance>"`; consumers recover the node with `block_id.rsplit("_", 1)[0]`. Template lookups key on `node_id`, not `bid`.
- **Train/test.** `environment_creation` splits the PDF under `<output>/agent_space/{train,test}/`. The pipeline scripts are built on train pages only, then applied to test pages by `src/pipeline.py`, which emits a flat page-local `extracted.json` list without record grouping.
- **Ground truth** is itself LLM-generated: `src/agents/gt_common.py` is the single shared generator (parallel windowed calls returning flat page-local blocks) used by both `train_eval.py` (window 5) and `ground_truth.py` (window 20). Train GT uses OCR layouts; final test GT uses page images plus exact OCR.
- **Eval** (`src/eval/`): converts records to a node tree (`convert.py`), aligns records/nodes (`align.py`), scores content with order-preserving row alignment (`content.py`, banded Needleman–Wunsch; rows below `MIN_ROW_SIM` become missing/extra unless they share an exact anchor cell), renders `eval.txt` (`render.py`). Accuracy = weighted record-F1 + node-F1 + content; structure weight is redistributed when trees are flat (always, currently). All gold/candidate JSON must be loaded through `convert.load_json_dedup` so duplicate keys survive as `key`, `key#2`, ….
- **Repair loop.** With `--train-threshold`, low train accuracy triggers `train_repair.py`: an OpenCode session edits the generated scripts, then train eval re-runs (bounded by `--train-max-retries`; graph recursion limit is derived from it in `build_graph`).
- **Costs.** Every node returns `cost` entries (phase, duration, tokens via `src/agents/_usage.py`); `save_metadata.py` writes them to `<output>/metadata.json`.

## Per-document output layout

`<output>/`: `template.json`, `metadata.json` (source hash, configuration, accuracy, and per-phase costs), `artifacts/blocks/` (annotated train-page block overlays), `agent_space/{train,test}/` (splits + generated scripts run there), `pipeline/` (`pipeline.py`, generated scripts, `block_separation.json`, `extracted.json`, `ground_truth.json`, `eval.json`, `eval.txt`, `train_eval/`), `debug/` (OpenCode session logs with `--save-log`).

## Gotchas

- Eval scoring semantics changed on 2026-07-08 (structure floor removed, missing rows penalized): accuracy numbers produced before that date are inflated and not comparable — re-score old artifacts with the current `src/eval` before putting them in the same table.
- The `data/` directory is gitignored and large; don't glob or grep it blindly.
- `.agents/skills/` contains the `/caveman` prompt material referenced by `--caveman`.
