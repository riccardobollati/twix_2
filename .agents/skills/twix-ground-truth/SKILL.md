---
name: twix-ground-truth
description: Generate held-out test ground truth for a TWIX UI run using its inferred template, including XOR record slots, page images, and authoritative OCR. Use when a run awaits a manual ground-truth paste or the user asks Codex to prepare that JSON without an OpenRouter call.
---

# TWIX Ground Truth

Generate the final held-out ground truth that the user will paste into the TWIX UI. Do not submit it to the UI yourself.

## Prepare the exact inputs

Run:

```bash
env/bin/python .agents/skills/twix-ground-truth/scripts/prepare_ground_truth.py <run-directory>
```

Use the run directory supplied by the user. If none is supplied, the helper selects the only `UI/runs/run-*` whose `ui_run.json` status is `awaiting_ground_truth`; if there is not exactly one, ask for the run directory.

The helper writes `manual_ground_truth_input/manifest.json`, `system_prompt.txt`, and one PNG/OCR-text pair per held-out page. It imports `src.ground_truth_gen.prompt.build_prompt` and injects the run's inferred `template.json`, so `system_prompt.txt` is the exact current multimodal ground-truth generator prompt. It renders the same 150-DPI page images and uses the same exact OCR character stream as the built-in generator.

## Extract

Read `system_prompt.txt` completely and treat it as the authoritative system prompt without paraphrasing or adding rules. Read `manifest.json`, inspect every listed page image, and read its matching OCR text file. Use images for layout and OCR as the authoritative character source.

The prompt lists physical `table` and `kv` nodes separately from any virtual `xor-table` and `xor-kv` record slots. For each XOR slot, identify which included physical layout appears in each record. The slot is satisfied by any one included node. Extract that physical node with its own fields and `node_id`; never create an XOR block or combine alternatives. Use the shared slot when checking record boundaries, especially when the layout changes between pages. The UI deterministically groups the submitted physical blocks into `ground_truth_grouped.json` using these XOR slots.

For more than 20 pages, work in consecutive windows of at most 20 while applying the same system prompt to every window, then concatenate the physical blocks in page order. Do not merge repeated blocks or cross-page tables.

Before responding, verify every block against the inferred template embedded in `system_prompt.txt`:

- `page` is the held-out PDF's local 1-based page number;
- `block_id` is a template `node_id` verbatim, with no instance suffix;
- `type` matches that node;
- `block_id` never names a virtual XOR node;
- every data object has exactly that node's fields and only string values;
- blocks remain in page and top-to-bottom order.

Return only the single JSON object required by `system_prompt.txt`, with no Markdown fence or commentary. The user will paste it into the Ground truth tab and separately enter the Codex model name shown for this session.
