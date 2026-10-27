"""Resume a failed UI run using its saved page plan and generated artifacts.

Usage: python -m src.resume_run UI/runs/<run-id>
Separator resumes rebuild scripts; train-eval resumes retain the assembled
pipeline and its original generation costs. Earlier logs are retained.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
from datetime import datetime, timezone


def train_eval_resume_available(run_dir: Path) -> bool:
    """Require a complete bundle matching the learned template before reuse."""
    root = Path(run_dir)
    pipeline = root / "pipeline"
    try:
        template = json.loads((root / "template.json").read_text())
        if json.loads((pipeline / "template.json").read_text()) != template:
            return False
        nodes = template["template"]
        physical = [n for n in nodes if n.get("type") in {"table", "kv"}]
        required = [pipeline / name for name in (
            "pipeline.py", "record_grouping.py", "block_separation.py",
        )] + [pipeline / "extraction_scripts" / f"{n['node_id']}.py" for n in physical]
        return bool(physical) and all(path.is_file() for path in required)
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return False


def load_resume_state(run_dir: Path) -> dict:
    saved = json.loads((run_dir / "ui_run.json").read_text())
    config = saved["config"]
    source = run_dir / Path(saved["filename"]).name
    template = json.loads((run_dir / "template.json").read_text())
    plan = json.loads((run_dir / "artifacts/clustering/selection.json").read_text())
    if not template.get("template"):
        raise ValueError("The saved template is empty")
    train, test = plan["train_source_pages"], plan["test_source_pages"]
    if (not train or train != sorted(set(train)) or test != sorted(set(test))
            or any(type(p) is not int or not 1 <= p <= plan["source_total_pages"] for p in train + test)
            or set(train) & set(test)):
        raise ValueError("The saved page split is invalid")
    if config.get("eval", True) and not test:
        raise ValueError("The saved split has no independent test pages")
    full_ocr = run_dir / "ocr_extraction_full.csv"
    folder = run_dir / "agent_space" / ("train" if config.get("eval", True) else "")
    for path in (source, full_ocr, folder / source.name, folder / "ocr_extraction.csv"):
        if not path.is_file():
            raise ValueError(f"Missing saved artifact: {path.name}")
    with source.open("rb") as handle:
        source_hash = hashlib.file_digest(handle, "sha256").hexdigest()
    return {
        "full_pdf_path": str(source), "full_ocr_csv_path": str(full_ocr),
        "source_sha256": source_hash,
        "output_dir": str(run_dir), "doc_name": source.stem,
        "train_pages": config["pages"], "eval_mode": config.get("eval", True),
        "max_train_pages": plan.get("max_train_pages", config.get("maxTrainPages")),
        "first_train_pages": plan.get("first_train_pages", config.get("firstTrainPages", 3)),
        "uncovered_clusters": plan.get("uncovered_clusters", []),
        "eval_pages": config.get("evalPages", 20),
        "holdout_pages": config.get("holdoutPages", 0),
        "train_accuracy_threshold": config.get("trainThreshold", 0),
        "train_repair_max_retries": config.get("trainMaxRetries", 3),
        "sonar_drop_percent": config.get("sonarDropPercent", 15),
        "no_routing": config.get("noRouting", False),
        "manual_ground_truth": config.get("manualGroundTruth", False),
        "save_log": True,
        "detected_start_page": plan["source_start_page"],
        **{key: plan[key] for key in (
            "train_source_pages", "test_source_pages", "first_source_pages",
            "cluster_sample_pages", "cluster_coverage", "table_cluster_count", "kv_cluster_count",
        )},
        "template": template["template"], "metadata": template.get("metadata", []),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--from-phase", choices=("block_separation", "train_eval"), default="block_separation")
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    state = load_resume_state(run_dir)
    resume_train = args.from_phase == "train_eval"
    if resume_train and not train_eval_resume_available(run_dir):
        raise ValueError("Train-eval resume requires a complete saved pipeline matching template.json")
    config = json.loads((run_dir / "ui_run.json").read_text())["config"]
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    from src.agents.environment_creation import environment_creation_node
    from src.agents.graph import build_graph
    from src.models import get_llm, get_ground_truth_llm
    from src.models.catalog import (
        resolve_easy_model, resolve_hard_model, resolve_reference_model,
        resolve_template_model, direct_model_id,
    )
    easy = resolve_easy_model(config.get("singleModel") if state["no_routing"] else config.get("easyModel"))
    hard = easy if state["no_routing"] else resolve_hard_model(config.get("hardModel"))
    reference = resolve_reference_model(config.get("referenceModel"))
    primary = get_llm(model=direct_model_id(easy)) if state["no_routing"] else get_llm()
    difficulty = primary if state["no_routing"] else get_llm(model=direct_model_id(hard))
    truth = get_ground_truth_llm(model=reference)
    state.update(
        easy_model=easy, hard_model=hard, opencode_model=hard,
        reference_model=reference, template_model=resolve_template_model(config.get("templateModel")),
        llm_model=direct_model_id(easy) if state["no_routing"] else getattr(primary, "model_name", "unknown"),
        difficulty_model=direct_model_id(hard), ground_truth_model=reference,
    )
    # Recreate only the inexpensive local splits from the exact saved plan.
    state.update(environment_creation_node(state))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    archive = run_dir / "debug" / "previous_attempts" / stamp
    archive.mkdir(parents=True)
    if resume_train:
        cost_path = run_dir / "metadata/cost.json"
        if cost_path.is_file():
            shutil.copy2(cost_path, archive / "cost.json")
            state["cost"] = json.loads(cost_path.read_text())["phases"]
    else:
        for path in (run_dir / "debug").glob("block_separation_*"):
            if path.is_file():
                shutil.move(str(path), archive / path.name)
    (run_dir / "resume.json").write_text(json.dumps({
        "resumedAt": stamp, "phase": args.from_phase,
        "costScope": ("Original generation phases plus continuation." if resume_train
                      else "Continuation only; previous attempt logs are preserved."),
    }, indent=2))
    print(f"{args.from_phase}: resuming saved template and page selection"
          + (" with the existing generated pipeline." if resume_train else "; costs cover this continuation."), flush=True)
    graph = build_graph(primary, train_repair_max_retries=state["train_repair_max_retries"],
                        ground_truth_llm=truth, difficulty_routing_llm=difficulty,
                        resume_from_block_separation=not resume_train,
                        resume_from_train_eval=resume_train)
    graph.invoke(state)


if __name__ == "__main__":
    main()
