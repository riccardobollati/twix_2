"""Re-score existing eval artifacts with the current src/eval code (no LLM calls).

For every data/<doc>/ that has pipeline/{ground_truth,grouped}.json and
metadata.json, recompute the score, rewrite eval.json / eval.txt, and patch
metadata.json in place:

    test_eval.accuracy / test_eval.exact_match   ← re-computed
    train_eval.accuracy / train_eval.exact_match ← re-computed (from
                                                   pipeline/train_eval/, if present)
    eval_version                                 ← content hash of src/eval/*.py

Use after any change to src/eval so metadata.json and the charts in
data/results/ never mix scorer versions. Regenerate charts afterwards with:

    python viz_batch_results.py data

Usage:
    python rescore.py [data_folder] [--doc name ...] [--dry-run]
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os

from src.agents.grouping import _maybe_merge_single_block
from src.agents.gt_common import _merge_records, _node_order, _should_merge
from src.eval.convert import load_json_dedup
from src.eval.render import render_to_file
from src.eval.score import score as eval_score


def eval_version() -> str:
    """Short content hash of src/eval/*.py — identifies the scorer version."""
    h = hashlib.sha256()
    eval_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src", "eval")
    for path in sorted(glob.glob(os.path.join(eval_dir, "*.py"))):
        with open(path, "rb") as f:
            h.update(f.read())
    return h.hexdigest()[:12]


def remerge_gt(doc_dir: str, gt_path: str, write: bool) -> bool:
    """Rebuild ground_truth.json from not_merged_gt.json with the current
    boundary-merge logic (no LLM calls). Returns True when the merged record
    structure changed."""
    raw_path = os.path.join(os.path.dirname(gt_path), "not_merged_gt.json")
    template_path = os.path.join(doc_dir, "template.json")
    if not (os.path.isfile(raw_path) and os.path.isfile(template_path)):
        return False
    if os.path.getmtime(template_path) > os.path.getmtime(raw_path):
        # template regenerated after the GT was produced (a re-run is in
        # flight) — re-merging old artifacts with a new template corrupts them
        print(f"  WARN: {os.path.basename(doc_dir.rstrip(os.sep))}: template.json "
              f"newer than not_merged_gt.json — skipping remerge")
        return False
    with open(template_path, encoding="utf-8") as f:
        template = json.load(f)["template"]
    order = _node_order(template)
    with open(raw_path, encoding="utf-8") as f:
        raw = json.load(f)

    def _window_of(rec: dict):
        blocks = rec.get("data", [])
        return tuple(blocks[0].get("window_pages") or []) if blocks else ()

    # Mirror generate_ground_truth's merge pass exactly: only the FIRST record
    # of each window is a continuation candidate; later records in the same
    # window are always appended.
    records: list[dict] = []
    prev_window = None
    for rec in raw:
        window = _window_of(rec)
        first_of_window = window != prev_window
        prev_window = window
        if records and first_of_window and _should_merge(records[-1], rec, order):
            records[-1] = _merge_records(records[-1], rec)
        else:
            records.append({"record_id": str(len(records)), "data": rec.get("data", [])})

    # apply the same single-block collapse pass as ground_truth_node, on a
    # temp file so --dry-run never touches ground_truth.json
    tmp_path = gt_path + ".remerge.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    try:
        _maybe_merge_single_block(tmp_path, template)
        with open(tmp_path, encoding="utf-8") as f:
            rebuilt = json.load(f)
        with open(gt_path, encoding="utf-8") as f:
            current = json.load(f)
        changed = rebuilt != current
        if changed and write:
            os.replace(tmp_path, gt_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    return changed


def rescore_pair(gt_path: str, cand_path: str, write_reports: bool) -> dict | None:
    if not (os.path.isfile(gt_path) and os.path.isfile(cand_path)):
        return None
    result = eval_score(
        load_json_dedup(gt_path), load_json_dedup(cand_path),
        reference="model_reference", threshold=0.9,
    )
    if write_reports:
        out_dir = os.path.dirname(cand_path)
        with open(os.path.join(out_dir, "eval.json"), "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        render_to_file(result, os.path.join(out_dir, "eval.txt"))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("folder", nargs="?", default="data", help="root data folder")
    parser.add_argument("--doc", action="append", help="only these doc names (repeatable)")
    parser.add_argument("--dry-run", action="store_true", help="print new scores, write nothing")
    parser.add_argument(
        "--remerge", action="store_true",
        help="rebuild test ground_truth.json from not_merged_gt.json with the "
             "current window-merge logic before scoring (no LLM calls)",
    )
    args = parser.parse_args()

    version = eval_version()
    print(f"scorer version: {version}\n")

    changed = 0
    for doc_dir in sorted(glob.glob(os.path.join(args.folder, "*", ""))):
        name = os.path.basename(doc_dir.rstrip(os.sep))
        if args.doc and name not in args.doc:
            continue
        metadata_path = os.path.join(doc_dir, "metadata.json")
        pipeline_dir = os.path.join(doc_dir, "pipeline")
        if not os.path.isfile(metadata_path):
            continue

        with open(metadata_path, encoding="utf-8") as f:
            metadata = json.load(f)

        updates: list[str] = []
        for split, sub in [("test_eval", ""), ("train_eval", "train_eval")]:
            base = os.path.join(pipeline_dir, sub)
            if split == "test_eval" and args.remerge:
                gt_path = os.path.join(base, "ground_truth.json")
                if os.path.isfile(gt_path) and remerge_gt(doc_dir, gt_path, write=not args.dry_run):
                    updates.append("  test_eval: ground_truth.json re-merged (record structure changed)")
            result = rescore_pair(
                os.path.join(base, "ground_truth.json"),
                os.path.join(base, "grouped.json"),
                write_reports=not args.dry_run,
            )
            if result is None:
                continue
            entry = metadata.setdefault(split, {})
            old = entry.get("accuracy")
            new = result["accuracy"]
            entry["accuracy"] = new
            entry["exact_match"] = result["exact_match"]
            delta = "" if old is None else f" (was {old:.4f}, Δ{new - old:+.4f})"
            marker = "  <-- changed" if old is None or abs(new - old) > 1e-4 else ""
            updates.append(f"  {split}: {new:.4f}{delta}{marker}")

        if not updates:
            continue
        metadata["eval_version"] = version
        if not args.dry_run:
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
        changed += 1
        print(name)
        for line in updates:
            print(line)

    action = "would patch" if args.dry_run else "patched"
    print(f"\n{action} {changed} metadata.json file(s).")
    if not args.dry_run:
        print("Regenerate charts with: python viz_batch_results.py " + args.folder)


if __name__ == "__main__":
    main()
