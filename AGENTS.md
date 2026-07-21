# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## What this is

Thesis research project: an agentic pipeline that extracts structured data (tables / key-value blocks) from PDF documents generated from a common template, then scores the extraction against LLM-generated ground truth. LangGraph orchestrates the phases; OpenAI vision models do inference; the `codex` CLI writes the actual Python separation/extraction scripts.

## Commands

```bash
source env/bin/activate                  # local virtualenv (python 3.14); deps in requirements.txt
python run.py <name>                     # run pipeline on data/<name>/<name>.pdf (template + extraction)
python run.py <name> --eval              # train/test split + ground truth + scoring
python run.py <name> --eval --pages 5 --train-threshold 0.9   # with Codex repair loop on low train accuracy
python src/run_batch.py --jobs 2 --eval  # run all data/ folders with a terminal dashboard (or pass folder names)
python single_eval.py ...                # re-run GT + eval standalone on an existing run
python fix.py <doc_id>                   # re-run test GT + eval, patch metadata.json in place
```

There is no test suite or linter. The root `test_*.py` files are ad-hoc experiment scripts, not pytest tests. Visualization helpers: `viz_batch_results.py` (batch accuracy tables), `viz_blocks.py` / `viz_bb.py` (draw block/word bboxes on pages), `viz_agent_log.py` (Codex session logs).

Requires `OPENAI_API_KEY` in `.env` (never commit it; it is gitignored) and the `codex` CLI on PATH. `LLM_PROVIDER` selects `src/models/{openai,azure}.py`; default model constant is `DEFAULT_MODEL` in those modules.

## Architecture

The LangGraph state machine is defined in `src/agents/graph.py`:

```
environment_creation → template_inference → record_separation → run_record_separation
→ block_separation → run_block_separation → block_samples → block_difficulty_router
→ block_extraction_scripting → collect_extraction_scripts → train_grouping
→ train_ground_truth → train_eval → [train_repair loop] → grouping
→ [eval_mode: ground_truth → eval] → save_metadata
```

Key ideas that span multiple files:

- **Two kinds of LLM output.** Vision calls return JSON directly (template inference, ground truth, difficulty routing). Separation and extraction are instead **generated Python scripts**: prompts in `src/prompts/` instruct `codex` (wrapped by `src/models/codex.py`) to write `record_separation.py`, `block_separation.py`, and per-block `extraction_scripts/<node_id>.py`; the agents then run those scripts via subprocess (300s timeout) and validate their JSON output, feeding failures back into a retry loop (max 3 attempts).
- **Template contract.** `template.json` nodes have `node_id`, `type` (table/kv), `fields`, `bid`, `complexity`. Field names within a node MUST be unique — grouped column headers are prefixed (`"Balance - Total"`); `template_inference.py` validates and dedups this because duplicate dict keys silently collapse everywhere downstream. Template `metadata` regex patterns are consumed by the block-separation prompt to strip page headers/footers (never strip unmatched bottom rows — they are usually data).
- **ID scheme.** `block_id = "<node_id>_<instance>"`; consumers recover the node with `block_id.rsplit("_", 1)[0]`. Template lookups key on `node_id`, not `bid`.
- **Train/test.** `environment_creation` splits the PDF under `data/<name>/agent_space/{train,test}/`. The pipeline scripts are built on train pages only, then applied to test pages by `src/pipeline.py` (the standalone runtime that runs separation + extractors + grouping without the graph).
- **Ground truth** is itself LLM-generated: `src/agents/gt_common.py` is the single shared generator (windowed vision calls, parallel via ThreadPoolExecutor, boundary-merge pass) used by both `train_eval.py` (window 5) and `ground_truth.py` (window 20). Train and test GT must keep identical semantics — they diverged once and it broke the repair loop.
- **Eval** (`src/eval/`): converts records to a node tree (`convert.py`), aligns records/nodes (`align.py`), scores content with order-preserving row alignment (`content.py`, banded Needleman–Wunsch; rows below `MIN_ROW_SIM` become missing/extra unless they share an exact anchor cell), renders `eval.txt` (`render.py`). Accuracy = weighted record-F1 + node-F1 + content; structure weight is redistributed when trees are flat (always, currently). All gold/candidate JSON must be loaded through `convert.load_json_dedup` so duplicate keys survive as `key`, `key#2`, ….
- **Repair loop.** With `--train-threshold`, low train accuracy triggers `train_repair.py`: a Codex session edits the generated scripts, then train eval re-runs (bounded by `--train-max-retries`; graph recursion limit is derived from it in `build_graph`).
- **Costs.** Every node returns `cost` entries (phase, duration, tokens via `src/agents/_usage.py`); `save_metadata.py` writes them to `data/<name>/metadata.json`.

## Per-document output layout

`data/<name>/`: `template.json`, `metadata.json` (accuracy + per-phase costs), `agent_space/{train,test}/` (splits + generated scripts run there), `pipeline/` (`block_separation.json`, `extracted.json`, `grouped.json`, `ground_truth.json`, `eval.json`, `eval.txt`, `train_eval/`), `debug/` (Codex session logs with `--save-log`).

## Gotchas

- Eval scoring semantics changed on 2026-07-08 (structure floor removed, missing rows penalized): accuracy numbers produced before that date are inflated and not comparable — re-score old artifacts with the current `src/eval` before putting them in the same table.
- The `data/` directory is gitignored and large; don't glob or grep it blindly.
- `.agents/skills/` contains the `/caveman` Codex skills referenced by `--caveman` (token-reduction prompt prefix).
