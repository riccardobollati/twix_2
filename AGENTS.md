# AGENTS.md

Guidance for coding agents working in this repository.

## Project purpose

TWIX 2 is a thesis research system for learning a reusable structured-data
extractor from a small sample of a template-based PDF family. It combines:

- `pdfplumber` OCR and geometry extraction;
- DeepSeek calls over layout-preserving OCR for front-matter detection, template
  inference, block difficulty routing, and train ground truth;
- GPT-5.4 vision calls through OpenRouter over page images plus exact OCR for final held-out ground truth;
- the OpenCode CLI for writing document-specific Python separation and extraction
  scripts; and
- LangGraph for orchestration, train evaluation, optional repair, test
  extraction, scoring, and metadata collection. Both direct DeepSeek calls and
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

python run.py path/to/document.pdf
python run.py path/to/document.pdf --eval
python run.py <dataset-name> --eval --pages 5 --train-threshold 0.9
python src/run_batch.py --jobs 2 --eval

python -m unittest discover -s tests -v
python -m compileall -q run.py fix.py rescore.py single_eval.py src
```

There is no configured linter. The tests are offline and must not make API
calls. Full pipeline runs require `OPENROUTER_API_KEY` and OpenCode available as
`opencode` on `PATH`; DeepSeek recommends OpenCode 1.14.24 or newer.

Environment selection:

- `LLM_PROVIDER=openrouter` uses `src/models/openrouter.py` and requires
  `OPENROUTER_API_KEY`; `OPENROUTER_MODEL` overrides the direct-call model.
- `OPENROUTER_REASONING_EFFORT` controls direct-call reasoning.
- `OPENROUTER_GROUND_TRUTH_MODEL` configures the vision model used only for
  final held-out/test ground truth (`openai/gpt-5.4` by default); it uses the
  same `OPENROUTER_API_KEY` as other direct calls.
- `OPENROUTER_GROUND_TRUTH_MAX_TOKENS` caps held-out/test ground-truth output
  (`128000` by default, GPT-5.4's full output allowance). OpenRouter requires
  the API key's credit limit to cover this maximum during its credit check.
- `OPENCODE_MODEL`, `OPENCODE_EASY_MODEL`, and `OPENCODE_HARD_MODEL` select
  OpenCode's `provider/model` code-generation and routing targets. OpenRouter
  targets use `openrouter/<publisher>/<model>`.
- Never print, commit, or replace `.env`. The CLI currently prints only the
  first ten key characters for batch-job identification.

## Repository map

| Path | Responsibility |
| --- | --- |
| `run.py` | Main CLI, input resolution, full-document OCR, graph invocation |
| `src/agents/graph.py` | LangGraph topology and train-repair routing |
| `src/agents/` | Individual graph phases and shared graph state |
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

`data/`, `outputs/`, `env/`, `.env`, PDFs, OCR dumps, and generated artifacts
are runtime data, not source. `data/` can be very large: do not recursively glob,
grep, copy, or format it unless a specific document is in scope.

## End-to-end graph

`src/agents/graph.py` is authoritative. The current order is:

```text
detect_beginning (unless --start-page)
  -> environment_creation
  -> template_inference
  -> record_separation -> run_record_separation
  -> block_separation  -> run_block_separation
  -> block_samples
  -> block_difficulty_router
  -> block_extraction_scripting
  -> collect_extraction_scripts
  -> train_eval
  -> [train_repair -> train_eval] repeated while the gate fails
  -> grouping
  -> merge_records
  -> [ground_truth -> eval] only with --eval
  -> save_metadata
```

Important implications:

- Train evaluation always runs, even without `--eval`. Setting
  `--train-threshold 0` disables only the repair loop.
- With `--eval`, the generated pipeline is built on the train split and the
  final `grouping` phase runs it on the test split. Without `--eval`, grouping
  runs on the cropped training PDF.
- `--no-routing` skips difficulty-model calls and assigns
  `OPENCODE_EASY_MODEL` to every extraction block.
- The graph recursion limit is derived from `--train-max-retries`; keep the
  compile-time and state values synchronized.

## Core contracts and invariants

### Page and OCR coordinates

`run.py` assigns globally unique OCR word IDs and physical 1-based page numbers.
After front-matter detection, `environment_creation.py` crops PDF and OCR splits
from the detected source page and remaps each split to local pages beginning at
1. Generated scripts operate only in split-local coordinates.

In eval mode, at least two usable structured pages are required. The train size
is capped at half of the usable pages so at least one test page remains.

The holdout is an anti-overfitting aid: template inference still sees every
train-page OCR layout, but generated separation code and its downstream block samples
use `ocr_extraction_build.csv`, which excludes the last held-out train pages.
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

OpenCode writes three script layers:

1. `record_separation.py <ocr.csv> <out_dir>` writes
   `record_separation.json` with record IDs and per-page word/y ranges.
2. `block_separation.py <ocr.csv> <out_dir>` writes
   `block_separation.json` with page-local blocks.
3. `extraction_scripts/<node_id>.py` exposes
   `extract(block: dict) -> dict` for one relatively positioned block.

Separation generation is retried at most three times. Each attempt is executed
with the active Python interpreter, has a 300-second runtime timeout, and must
pass basic JSON validation. OpenCode sessions themselves have an 1800-second
timeout.

Never weaken these invariants:

- `block_id = "<node_id>_<page-local-instance>"`; recover the node with
  `block_id.rsplit("_", 1)[0]`.
- `uid` uniquely identifies a physical block across the full runtime input and
  links separation, extraction, and record grouping.
- Blocks are page-local. A block emitted under page P may contain only page-P
  words. Multi-page tables use continuation blocks, later merged logically.
- Do not discard unmatched bottom rows; they are commonly real table data.
  Page-frame removal requires both an edge position and a matching
  `document_level` metadata regex.
- Generated logic must derive geometry from the current input. Do not hard-code
  sample values, absolute coordinates, page numbers, record counts, or block
  counts.
- Extractors receive word coordinates translated so the block's top-left is
  `(0, 0)`.
- Generated scripts are expected to use the standard library only. The portable
  runner requires only `pdfplumber`.

### Standalone runtime and grouping

`src/pipeline.py` performs OCR, executes both separation scripts, imports every
extractor, relativizes block coordinates, writes `extracted.json`, and spatially
assigns block `uid`s to records. Strict record containment is preferred; the
greatest y-overlap is a fallback for boundary-touching blocks.

After grouping, `_maybe_merge_single_block`:

- collapses a one-node table template to one record and one concatenated table;
- collapses records for a one-node non-table template while retaining its
  blocks; and
- concatenates all same-node table continuation blocks within each record.

`pipeline_health.json` marks output degenerate when extraction is empty, blocks
cover fewer than half the pages, blocks cross page boundaries, or more than 20%
of detected records receive no blocks. A degenerate train result fails the
repair gate even if its numerical accuracy reaches the threshold.

### Ground truth

`src/agents/gt_common.py` is the shared train/test implementation. Do not fork
its semantics between `train_eval.py` and `ground_truth.py`.

- Train ground truth uses five-page windows; test defaults to twenty pages.
- Independent windows are called concurrently, but results are consumed in page
  order.
- Train ground truth sends layout-preserving OCR to DeepSeek through OpenRouter.
  Final held-out/test ground truth sends images and exact OCR text together to
  GPT-5.4 through OpenRouter, keeping the evaluation model independent of the
  training model.
- A window with suspiciously low value-character coverage is retried once.
- Records crossing a window boundary are merged using template order and
  repeated identity values.
- An entirely empty result from a window size greater than five retries the
  whole document with five-page windows; a second empty result is a hard error.
- Raw pre-merge records are saved as `not_merged_gt.json` for later auditing or
  `rescore.py --remerge`.

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
├── debug/                         # only with --save-log
├── agent_space/
│   ├── train/ and test/           # eval mode
│   └── ...                        # unsplit workspace otherwise
└── pipeline/
    ├── pipeline.py
    ├── requirements.txt
    ├── record_separation.py
    ├── block_separation.py
    ├── extraction_scripts/<node_id>.py
    ├── ocr.csv
    ├── record_separation.json
    ├── block_separation.json
    ├── extracted.json
    ├── grouped.json
    ├── pipeline_health.json
    ├── train_eval/
    ├── ground_truth.json          # eval mode
    ├── not_merged_gt.json         # eval mode
    ├── eval.json                  # eval mode
    └── eval.txt                   # eval mode
```

Do not assume every optional path exists. The exact `agent_space` shape differs
between eval and non-eval runs.

## Safe change workflow

Before editing:

1. Read the graph node, its prompt, its artifact consumers, and relevant tests.
   Many contracts are duplicated intentionally across generation prompts,
   runtime validation, grouping, ground truth, repair, and evaluation.
2. Check `git status`; the worktree commonly contains research changes and
   generated artifacts. Preserve unrelated edits and never reset the tree.
3. Avoid external model calls unless the task explicitly requires an online
   pipeline run. Prefer fixtures and unit tests.

When changing a contract:

- Template changes: inspect `template_inference.py`, its prompt, record/block
  prompts, extractor prompt, ground-truth prompt, grouping, and conversion.
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
python -m compileall -q run.py fix.py rescore.py single_eval.py src
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
