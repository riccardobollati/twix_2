#!/usr/bin/env python3
"""Run the repository's unchanged TWIX scorer on frozen evaluation inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from src.eval.render import render_to_file
from src.eval.score import score_paths


SCORER_FILES = [
    Path("src/eval/score.py"), Path("src/eval/convert.py"), Path("src/eval/align.py"),
    Path("src/eval/content.py"), Path("src/eval/render.py"),
]


def combined_hash(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path).encode("utf-8") + b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("eval_json", type=Path)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()

    result = score_paths(args.reference, args.candidate, reference="model_reference")
    result.update({
        "doc_name": "id_07_upscaled pages 6-25",
        "candidate": "frozen baseline extractor via evaluation adapter",
        "candidate_model": "rule-based pdfplumber extractor",
        "reference_model": "Codex local PDF image/text review",
        "eval_version": "twix-eval-source-sha256:" + combined_hash(SCORER_FILES),
    })
    args.eval_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    render_to_file(result, args.report)
    print(json.dumps({"accuracy": result["accuracy"], "exact_match": result["exact_match"],
                      "eval_version": result["eval_version"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
