# AGENTS.md

Guidance for coding agents working in this repository.

## Project purpose

TWIX 2 is a thesis research system for learning a reusable structured-data
extractor from a small sample of a template-based PDF family. It combines:

- `pdfplumber` OCR and geometry extraction;
- configurable OpenRouter calls over layout-preserving OCR for front-matter
  detection, template inference, block difficulty routing, and train ground truth;
- configurable multimodal reference-model calls through OpenRouter over page
  images plus exact OCR for final held-out ground truth (GPT-5.4 by default);
- the OpenCode CLI for writing document-specific Python separation and extraction
  scripts; and
- LangGraph for orchestration, train evaluation, optional repair, test
  extraction, scoring, and metadata collection. Both direct model calls and
  OpenCode model calls are routed through OpenRouter.

The output is both the extracted JSON and a portable `pipeline/` directory that
can process unseen PDFs with the same layout without running LangGraph or making
model calls.

## Setup and common commands

```bash
python3.14 -m venv env
source env/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env

python run.py path/to/document.pdf --no-eval
python run.py path/to/document.pdf --eval
python run.py <dataset-name> --eval --pages 5 --train-threshold 0.9
python api.py --host 127.0.0.1 --port 8000
python src/run_batch.py --jobs 2 --eval

python -m unittest discover -s tests -v
python -m compileall -q api.py run.py fix.py rescore.py single_eval.py src
```

There is no configured linter. The tests are offline and must not make API
calls. Full pipeline runs require `OPENROUTER_API_KEY` and OpenCode available as
`opencode` on `PATH`; DeepSeek recommends OpenCode 1.14.24 or newer.

Environment selection:

- `LLM_PROVIDER=openrouter` uses `src/models/openrouter.py` and requires
  `OPENROUTER_API_KEY`; `OPENROUTER_MODEL` overrides the direct-call model.
- `OPENROUTER_REASONING_EFFORT` controls direct-call reasoning.
- `OPENROUTER_GROUND_TRUTH_MODEL` configures the reference model used for train
  and final held-out/test ground truth (`openai/gpt-5.4` by default); it uses
  the same `OPENROUTER_API_KEY` as other direct calls.
- `OPENROUTER_TEMPLATE_INFERENCE_MODEL` configures template inference separately
  (`openai/gpt-5.4` by default); `--template-model` overrides it for one run.
- `OPENROUTER_GROUND_TRUTH_MAX_TOKENS` caps held-out/test ground-truth output
  (`128000` by default, GPT-5.4's full output allowance). OpenRouter requires
  the API key's credit limit to cover this maximum during its credit check.
- `OPENCODE_MODEL`, `OPENCODE_EASY_MODEL`, and `OPENCODE_HARD_MODEL` select
  OpenCode's `provider/model` code-generation and routing targets. OpenRouter
  targets use `openrouter/<publisher>/<model>`.
- `--easy-model`, `--hard-model`, and `--reference-model` override those roles
  for one run. The hard model always generates block separation and performs
  direct difficulty classification; it also handles hard extractors and repair.
  With `--no-routing`, the easy-model selection is used for every non-reference
  phase; template inference and ground truth keep their separate selected models.
- Never print, commit, or replace `.env`. The CLI currently prints only the
  first ten key characters for batch-job identification.

## Repository map

| Path | Responsibility |
| --- | --- |
| `api.py` | Asynchronous PDF upload, pipeline status, and extraction-result HTTP API |
| `run.py` | Main CLI, input resolution, full-document OCR, graph invocation |
| `src/agents/graph.py` | LangGraph topology and train-repair routing |
| `src/agents/cluster_sampling.py` | Deterministic table/KV page sampling and source-page split plan |
| `src/agents/` | Individual graph phases and shared graph state |
| `test-clustering/` | Shared deterministic table/KV cluster engines and research viewers |
| `src/prompts/` | Contracts given to DeepSeek and OpenCode |
| `src/models/` | DeepSeek provider and OpenCode CLI wrapper |
| `src/pipeline.py` | Portable, model-free runtime copied into every generated pipeline |
| `src/eval/` | Format conversion, alignment, content/structure scoring, report rendering |
| `src/ground_truth_gen/` | Ground-truth prompt plus OCR/render/JSON helpers |
| `src/run_batch.py` | Parallel benchmark runner with terminal dashboard |
| `single_eval.py` | Evaluate a generated pipeline on another PDF or page sample |
| `fix.py` | Regenerate test ground truth/eval for one legacy `data/<id>` run |
| `rescore.py` | Re-score saved artifacts with current `src/eval` code, without LLM calls |
| `viz_*.py` | Bounding-box, block, OpenCode-log, and batch-result visualizations |
| `tests/` | Offline `unittest` regressions |

`data/`, `outputs/`, `api_runs/`, `env/`, `.env`, PDFs, OCR dumps, and generated
artifacts are runtime data, not source. `data/` can be very large: do not
recursively glob, grep, copy, or format it unless a specific document is in
scope.

## End-to-end graph

`src/agents/graph.py` is authoritative. The current order is:

```text
detect_beginning (unless --start-page)
  -> cluster_sampling
  -> environment_creation
  -> template_inference
  -> block_separation -> run_block_separation -> excluded_blocks_sonar
  -> block_artifacts
  -> block_samples
  -> block_difficulty_router
  -> block_extraction_scripting
  -> collect_extraction_scripts
  -> train_eval
  -> [train_repair -> train_eval] repeated while the gate fails
  -> extraction
  -> [ground_truth -> eval] when evaluation is enabled (the default)
  -> save_metadata
```

Important implications:

- Train evaluation always runs, even with `--no-eval`. Setting
  `--train-threshold 0` disables only the repair loop.
- Test evaluation is enabled by default. The generated pipeline is built on the
  selected train pages and the final `extraction` phase runs it on up to 20
  disjoint, spread-out test pages by default.
  With `--no-eval`, it runs
  on the cropped training PDF.
- `--no-routing` skips difficulty-model calls and uses the selected easy model
  for every non-reference direct and OpenCode phase.
- The graph recursion limit is derived from `--train-max-retries`; keep the
  compile-time and state values synchronized.

## Core contracts and invariants

### Page and OCR coordinates

`run.py` assigns globally unique OCR word IDs and physical 1-based page numbers.
After front-matter detection, `cluster_sampling.py` scans up to 500 structured
source pages using independent table and KV clustering. It selects the first
three structured pages for block order plus pages covering every detected
cluster. `environment_creation.py` builds train and test PDFs from explicit,
disjoint physical-page lists and remaps each split to local pages beginning at
1. `artifacts/clustering/selection.json` and `metadata.json` preserve the
local-to-source mapping. Generated scripts operate only in split-local coordinates.

In eval mode, at least two usable structured pages are required. A train-page
selection must leave at least one source page for independent evaluation;
`--pages` is a minimum and cannot remove cluster-representative pages.

The holdout is an anti-overfitting aid: template inference still sees every
train-page OCR layout, but generated separation code and its downstream block samples
use `ocr_extraction_build.csv`, which excludes only optional held-out train
pages. First pages and cluster samples always remain available for script
generation and block samples.
`train_eval` then runs the finished pipeline on all train pages.

### Template contract

`template.json` has this root shape:

```json
{
  "template": [
    {
      "type": "table",
      "fields": ["Field A", "Field B"],
      "bid": [0],
      "child": -1,
      "node_id": 0,
      "complexity": {
        "same_row_structure": true,
        "nested": false
      }
    }
  ],
  "metadata": [
    {"pattern": "Page \\d+ of \\d+", "type": "document_level"}
  ]
}
```

Every node requires `type`, non-empty `fields`, `bid`, `node_id`, and
`complexity`. Field names inside one node must be unique. Two-tier headers use
`"Group - Column"`; persistent duplicates receive deterministic ` (2)`, ` (3)`
suffixes. Nodes with identical normalized field signatures are tagged with
`signature_duplicate_of` because field matching cannot otherwise distinguish
them reliably.

Use `node_id` for all template lookups. `bid` describes visual block positions
within a record; it is not the lookup key.

### Generated-script contracts

OpenCode writes two script layers:

1. `block_separation.py <ocr.csv> <out_dir>` writes
   `block_separation.json` with page-local blocks.
2. `extraction_scripts/<node_id>.py` exposes
   `extract(block: dict) -> dict` for one relatively positioned block.

Separation and extractor generation are each retried at most three times.
Separation scripts are executed with the active Python interpreter, have a
300-second runtime timeout, and must pass basic JSON validation. Before every
extractor attempt, the target script is removed so a stale file cannot count as
success. A generated extractor must parse and expose exactly one synchronous,
top-level `extract` function that accepts exactly one positional argument;
generation failures and contract errors are fed into the next attempt. Usage is
accumulated across all attempts, including failed ones.

OpenCode sessions themselves have a 3600-second timeout. All OpenCode calls use
an isolated temporary database per attempt and share a cross-process two-slot
limiter, preventing concurrent TWIX UI, CLI, and batch runs from contending for
OpenCode's global SQLite database. Database-lock output is treated as a
transient failure even if OpenCode exits zero. Timeout debug logs preserve any
partial output and usage that OpenCode emitted.

Never weaken these invariants:

- `block_id = "<node_id>_<page-local-instance>"`; recover the node with
  `block_id.rsplit("_", 1)[0]`.
- `uid` uniquely identifies a physical block across the full runtime input and
  links block separation to extraction.
- Blocks are page-local. A block emitted under page P may contain only page-P
  words. Multi-page tables use separate continuation blocks that remain separate.
- Do not discard unmatched bottom rows; they are commonly real table data.
  Page-frame removal requires both an edge position and a matching
  `document_level` metadata regex.
- Generated logic must derive geometry from the current input. Do not hard-code
  sample values, absolute coordinates, page numbers, or block
  counts.
- Extractors receive word coordinates translated so the block's top-left is
  `(0, 0)`.
- Generated scripts are expected to use the standard library only. The portable
  runner requires only `pdfplumber`.

### Standalone runtime

`src/pipeline.py` performs OCR, executes block separation, imports every
extractor, relativizes block coordinates, and writes a flat, page-local
`extracted.json` list. `src/record_grouping.py` then deterministically traverses
the template node IDs and writes logical records to `grouped.json`. A completed
template restart or a duplicate non-repeatable node starts a record;
`complexity.repeated`, multi-`bid` nodes, and adjacent cross-page table
continuations remain in the current record. Physical blocks remain separate.

`pipeline_health.json` marks output degenerate when extraction is empty, blocks
cover fewer than half the pages, or blocks cross page boundaries. A degenerate train result fails the
repair gate even if its numerical accuracy reaches the threshold.

Immediately after the generated block separator passes validation,
`excluded_blocks_sonar` applies it to up to 100 source pages from the detected
structured-data start, measures OCR-word coverage against its own OCR CSV, and
writes a coverage plot and per-page `sonar.json` under
`<output>/artifacts/excluded_blocks_sonar/`. Pages are flagged when coverage
falls by the configured relative percentage below the scanned-page average.
The UI can replay the separator on another source-page range and renders block
overlays for selected pages on demand.

After that,
`block_artifacts` reapplies it to the complete train PDF, then calls the reusable
renderers in `viz_blocks.py` and `viz_record_separation.py`. It writes block
overlays under `<output>/artifacts/blocks/` and deterministic record overlays
under `<output>/artifacts/record_separation/`. In eval mode it also applies the
learned separator to the independent test PDF and writes UI-only overlays under
`<output>/artifacts/test_blocks/` and
`<output>/artifacts/test_record_separation/`; these do not feed back into
training.

### Ground truth

`src/agents/gt_common.py` is the shared train/test implementation. Do not fork
its semantics between `train_eval.py` and `ground_truth.py`.

- Train ground truth uses five-page windows; test defaults to twenty pages.
- Independent windows are called concurrently, but results are consumed in page
  order.
- Train ground truth sends layout-preserving OCR to the selected reference model
  through OpenRouter. Final held-out/test ground truth sends images and exact
  OCR text to that same model, keeping evaluation independent of the
  operational pipeline model.
- A window with suspiciously low value-character coverage is retried once.
- The model emits a flat `blocks` list with an explicit page per physical block.
- Blocks from independent windows are concatenated in page order without record
  grouping or same-node table merging. The shared deterministic grouper writes
  `ground_truth_grouped.json` afterward for evaluation.
- An entirely empty result from a window size greater than five retries the
  whole document with five-page windows; a second empty result is a hard error.

All JSON used for scoring must be loaded through
`src.eval.convert.load_json_dedup`, which preserves repeated object keys as
`key`, `key#2`, and so on.

### Evaluation semantics

Evaluation converts supported artifact shapes into records containing flat
nodes, then performs Hungarian record alignment and two-pass Hungarian node
alignment. Table rows are aligned in order using exact anchors plus banded
Needleman-Wunsch; weak pairs become missing/extra rows unless they share an
exact non-empty anchor cell.

Headline weights in `src/eval/score.py` are:

- typed NodeMatchF1: 0.40;
- size-weighted ContentScore: 0.40; and
- parent-edge StructureScore: 0.20.

For the flat nodes produced by `convert.py`, structure is not applicable, so its
weight is renormalized away: headline accuracy is 50% NodeMatchF1 and 50%
ContentScore. `RecordMatchF1`, untyped node F1, path accuracy, and note agreement
are diagnostics and do not directly enter accuracy. Missing and extra nodes
contribute zero content, preventing survivorship bias.

The scorer normalizes Unicode compatibility characters, whitespace, case,
dash/quote variants, surrounding formatting punctuation, numeric formatting,
and spurious intra-token spaces. `eval.txt` is the actionable repair report;
reason codes there are consumed by the OpenCode repair prompt.

Scorer behavior has changed over the project. Never compare old and new saved
accuracies blindly: run `python rescore.py ...` and record the resulting
`eval_version` before aggregating experiments.

## Output layout

For a document artifact directory `<output>/`:

```text
<output>/
├── ocr_extraction_full.csv
├── template.json
├── metadata.json
├── metadata/cost.json
├── artifacts/blocks/               # block-separation overlay PNGs
├── artifacts/excluded_blocks_sonar/ # per-page coverage, plot, and on-demand overlays
├── artifacts/record_separation/    # deterministic train record overlays
├── artifacts/stats/                # UI runs only: plots plus stats.json
├── artifacts/test_blocks/          # held-out overlays in eval mode
├── artifacts/test_record_separation/ # held-out record overlays in eval mode
├── debug/                         # only with --save-log
├── agent_space/
│   ├── train/ and test/           # eval mode
│   └── ...                        # unsplit workspace otherwise
└── pipeline/
    ├── pipeline.py
    ├── record_grouping.py
    ├── requirements.txt
    ├── template.json
    ├── block_separation.py
    ├── extraction_scripts/<node_id>.py
    ├── ocr.csv
    ├── block_separation.json
    ├── extracted.json
    ├── grouped.json
    ├── pipeline_health.json
    ├── train_eval/
    ├── ground_truth.json          # eval mode
    ├── ground_truth_grouped.json  # eval mode
    ├── eval.json                  # eval mode
    └── eval.txt                   # eval mode
```

Do not assume every optional path exists. The exact `agent_space` shape differs
between eval and non-eval runs.

## Safe change workflow

Before editing:

1. Read the graph node, its prompt, its artifact consumers, and relevant tests.
   Many contracts are duplicated intentionally across generation prompts,
   runtime validation, ground truth, repair, and evaluation.
2. Check `git status`; the worktree commonly contains research changes and
   generated artifacts. Preserve unrelated edits and never reset the tree.
3. Avoid external model calls unless the task explicitly requires an online
   pipeline run. Prefer fixtures and unit tests.

When changing a contract:

- Template changes: inspect `template_inference.py`, its prompt, block/extractor
  prompts, the ground-truth prompt, runtime extraction, and conversion.
- Block ID changes: update every `rsplit("_", 1)` consumer and visualization.
- Ground-truth changes: keep train/test behavior shared in `gt_common.py` and
  check `single_eval.py` plus `rescore.py` for mirrored legacy logic.
- Eval changes: run `rescore.py --dry-run` on representative saved artifacts
  when available, and update reports/plot assumptions if output shape changes.
- CLI changes: forward them through `src/run_batch.py` and add parser/command
  tests.
- Pipeline bundle changes: update `src/pipeline.py`,
  `src/pipeline_requirements.txt`, collection logic, and the fresh-directory
  standalone test.

Minimum verification for source changes:

```bash
python -m unittest discover -s tests -v
python -m compileall -q api.py run.py fix.py rescore.py single_eval.py src
```

Add focused offline regressions for bug fixes. Do not make tests depend on the
ignored `data/` corpus, credentials, live provider output, or the installed
OpenCode service.

## Known sharp edges

- The pipeline learns one document family at a time. A generated bundle is not
  a generic PDF extractor.
- Ground truth is model-generated, not human truth; low-coverage and empty-output
  defenses reduce but do not eliminate measurement error.
- `single_eval.py` contains its own sequential ground-truth driver for sampled
  pages. Shared boundary helpers come from `gt_common.py`, but generation
  behavior is not identical to the graph's parallel coverage-checked path.
- `fix.py` assumes the legacy `data/<doc_id>` layout and mutates existing ground
  truth, eval, and metadata artifacts in place.
- `rescore.py` can rewrite reports and metadata. Use `--dry-run` first when
  examining historical experiments.
- `--group-rows` changes the OCR granularity even though generated prompts and
  most runtime contracts are word-oriented; treat it as an experimental mode.
- Debug logs can contain prompts, OCR text, and agent command output. Treat them
  as potentially sensitive artifacts.
