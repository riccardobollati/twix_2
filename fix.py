"""Re-run the test-phase ground truth + eval and patch metadata.json.

Assumes a prior full run has already produced:
    data/<doc_id>/pipeline/grouped.json
    data/<doc_id>/pipeline/ground_truth.json  (will be overwritten)
    data/<doc_id>/metadata.json               (will be patched in-place)

Re-runs:
    ground_truth_node  → pipeline/ground_truth.json  (+ not_merged_gt.json)
    eval_node          → pipeline/eval.json + eval.txt

Then patches metadata.json:
    test_eval.ground_truth_cost  ← new cost entry (with tokens)
    test_eval.accuracy           ← re-computed accuracy
    test_eval.exact_match        ← re-computed exact_match
"""

import json
import os

from src.agents.eval import eval_node
from src.agents.ground_truth import ground_truth_node
from src.models import get_llm

# ── config ────────────────────────────────────────────────────────────────────

doc_id = "id_04"
window = 20

# ── paths ─────────────────────────────────────────────────────────────────────

data_dir      = os.path.join("data", doc_id)
agent_space   = os.path.join(data_dir, "agent_space")
pipeline_dir  = os.path.join(data_dir, "pipeline")
metadata_path = os.path.join(data_dir, "metadata.json")
template_path = os.path.join(data_dir, "template.json")

for p in [pipeline_dir, metadata_path, template_path]:
    if not os.path.exists(p):
        raise SystemExit(f"ERROR: not found: {p}")

test_pdf = os.path.join(agent_space, "test", f"{doc_id}.pdf")
test_ocr = os.path.join(agent_space, "test", "ocr_extraction.csv")
train_pdf = os.path.join(agent_space, "train", f"{doc_id}.pdf")
train_ocr = os.path.join(agent_space, "train", "ocr_extraction.csv")

if not os.path.isfile(test_pdf):
    raise SystemExit(f"ERROR: test PDF not found: {test_pdf}. Run test.py first to create splits.")

with open(template_path, encoding="utf-8") as f:
    template = json.load(f)["template"]
print(f"Template: {len(template)} node(s)")

# ── build state ───────────────────────────────────────────────────────────────

state = {
    "agent_space":   agent_space,
    "pdf_path":      train_pdf,
    "ocr_csv_path":  train_ocr,
    "test_pdf_path": test_pdf,
    "test_ocr_path": test_ocr if os.path.isfile(test_ocr) else None,
    "template":      template,
    "messages":      [],
    "cost":          [],
}

# ── ground truth ──────────────────────────────────────────────────────────────

print("\n── ground_truth_node ────────────────────────────────────────")
llm = get_llm()
result = ground_truth_node(state, llm, window=window)
gt_cost = result["cost"][0]
print(f"duration: {gt_cost['duration_s']}s  tokens: {gt_cost.get('tokens', 'n/a')}")

# ── eval ──────────────────────────────────────────────────────────────────────

print("\n── eval_node ────────────────────────────────────────────────")
result = eval_node(state)
print(f"duration: {result['cost'][0]['duration_s']}s")

# ── patch metadata.json ───────────────────────────────────────────────────────

with open(metadata_path, encoding="utf-8") as f:
    metadata = json.load(f)

eval_json_path = os.path.join(pipeline_dir, "eval.json")
with open(eval_json_path, encoding="utf-8") as f:
    ev = json.load(f)

if "test_eval" not in metadata:
    metadata["test_eval"] = {}

metadata["test_eval"]["ground_truth_cost"] = gt_cost
metadata["test_eval"]["accuracy"]    = ev.get("accuracy")
metadata["test_eval"]["exact_match"] = ev.get("exact_match")

with open(metadata_path, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

print(f"\nmetadata.json patched → {metadata_path}")
print(f"  accuracy={ev['accuracy']:.6f}  exact_match={ev['exact_match']}")
print(f"  ground_truth_cost={gt_cost}")
