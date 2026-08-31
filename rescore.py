"""Re-score existing eval artifacts with the current src/eval code (no LLM calls).

For every data/<doc>/ that has pipeline ground-truth/candidate artifacts and
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
            grouped_gt = os.path.join(base, "ground_truth_grouped.json")
            grouped_candidate = os.path.join(base, "grouped.json")
            result = rescore_pair(
                grouped_gt
                if os.path.isfile(grouped_gt)
                else os.path.join(base, "ground_truth.json"),
                grouped_candidate
                if os.path.isfile(grouped_candidate)
                else os.path.join(base, "extracted.json"),
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
