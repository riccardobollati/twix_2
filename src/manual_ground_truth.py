"""Validate and finalize a manually supplied held-out ground truth."""

from __future__ import annotations

import datetime
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import pdfplumber

from src.eval.convert import load_json_dedup, loads_json_dedup
from src.eval.render import render_to_file
from src.eval.score import score as eval_score
from src.record_grouping import group_blocks_into_records


class ManualGroundTruthError(ValueError):
    """The pasted ground truth does not satisfy the generator contract."""


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as target:
        json.dump(value, target, indent=2, ensure_ascii=False)
        target.write("\n")
        temporary = Path(target.name)
    os.replace(temporary, path)


def _template_specs(template: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(node.get("node_id")): node for node in template}


def _validate_data(
    data: Any,
    *,
    block_type: str,
    fields: list[str],
    label: str,
) -> None:
    rows: list[Any]
    if block_type == "table":
        if not isinstance(data, list):
            raise ManualGroundTruthError(f"{label}.data must be a list for a table")
        rows = data
    else:
        if not isinstance(data, dict):
            raise ManualGroundTruthError(f"{label}.data must be an object for kv")
        rows = [data]

    expected = set(fields)
    for row_index, row in enumerate(rows):
        row_label = f"{label}.data[{row_index}]" if block_type == "table" else f"{label}.data"
        if not isinstance(row, dict):
            raise ManualGroundTruthError(f"{row_label} must be an object")
        actual = set(row)
        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            details = []
            if missing:
                details.append(f"missing {missing}")
            if extra:
                details.append(f"unexpected {extra}")
            raise ManualGroundTruthError(
                f"{row_label} keys must exactly match the template ({'; '.join(details)})"
            )
        non_strings = [key for key, value in row.items() if not isinstance(value, str)]
        if non_strings:
            raise ManualGroundTruthError(
                f"{row_label} values must be strings; invalid fields: {non_strings}"
            )


def parse_manual_ground_truth(
    raw: str,
    template: list[dict[str, Any]],
    page_count: int,
) -> list[dict[str, Any]]:
    """Parse the exact generator response shape and validate its template contract."""
    try:
        payload = loads_json_dedup(raw)
    except json.JSONDecodeError as exc:
        raise ManualGroundTruthError(f"Ground truth is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict) or set(payload) != {"blocks"}:
        raise ManualGroundTruthError('Root must be an object containing only "blocks"')
    blocks = payload["blocks"]
    if not isinstance(blocks, list) or not blocks:
        raise ManualGroundTruthError('"blocks" must be a non-empty list')

    specs = _template_specs(template)
    if not specs:
        raise ManualGroundTruthError("The inferred template has no nodes")
    previous_page = 0
    normalized: list[dict[str, Any]] = []
    for index, block in enumerate(blocks):
        label = f"blocks[{index}]"
        if not isinstance(block, dict):
            raise ManualGroundTruthError(f"{label} must be an object")
        missing = [key for key in ("page", "type", "block_id", "data") if key not in block]
        if missing:
            raise ManualGroundTruthError(f"{label} is missing {missing}")
        try:
            page = int(block["page"])
        except (TypeError, ValueError) as exc:
            raise ManualGroundTruthError(f"{label}.page must be an integer") from exc
        if not 1 <= page <= page_count:
            raise ManualGroundTruthError(
                f"{label}.page {page} is outside the test PDF range 1-{page_count}"
            )
        if page < previous_page:
            raise ManualGroundTruthError("Blocks must be emitted in page order")
        previous_page = page

        block_id = str(block["block_id"])
        spec = specs.get(block_id)
        if spec is None:
            raise ManualGroundTruthError(
                f"{label}.block_id {block_id!r} is not in the inferred template"
            )
        expected_type = spec.get("type")
        if block["type"] != expected_type:
            raise ManualGroundTruthError(
                f"{label}.type must be {expected_type!r} for node {block_id}"
            )
        fields = spec.get("fields") or []
        _validate_data(
            block["data"],
            block_type=expected_type,
            fields=fields,
            label=label,
        )
        normalized.append({**block, "page": page, "block_id": block_id})
    return normalized


def load_manual_ground_truth_inputs(
    run_dir: str | Path,
    raw: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load the inferred template and validate pasted GT against the test PDF."""
    root = Path(run_dir).resolve()
    template_payload = load_json_dedup(root / "template.json")
    template = template_payload.get("template") if isinstance(template_payload, dict) else None
    if not isinstance(template, list):
        raise ManualGroundTruthError("Run template.json does not contain a template list")
    test_pdfs = sorted((root / "agent_space" / "test").glob("*.pdf"))
    if len(test_pdfs) != 1:
        raise ManualGroundTruthError("Expected exactly one held-out test PDF for this run")
    with pdfplumber.open(test_pdfs[0]) as pdf:
        page_count = len(pdf.pages)
    return parse_manual_ground_truth(raw, template, page_count), template


def finalize_manual_ground_truth(
    run_dir: str | Path,
    raw: str,
    model: str,
) -> dict[str, Any]:
    """Persist pasted GT, score it locally, and log its external model."""
    model = model.strip()
    if not model:
        raise ManualGroundTruthError("Model is required")
    if len(model) > 200:
        raise ManualGroundTruthError("Model must be 200 characters or fewer")
    if any(character in model for character in "\r\n"):
        raise ManualGroundTruthError("Model must be a single line")

    started = time.time()
    root = Path(run_dir).resolve()
    blocks, template = load_manual_ground_truth_inputs(root, raw)
    pipeline = root / "pipeline"
    candidate_path = pipeline / "grouped.json"
    if not candidate_path.is_file():
        raise ManualGroundTruthError("The held-out extraction is not ready yet")

    grouped_ground_truth = group_blocks_into_records(blocks, template)
    candidate = load_json_dedup(candidate_path)
    evaluation = eval_score(
        grouped_ground_truth,
        candidate,
        reference="model_reference",
        threshold=0.9,
    )

    _atomic_json(pipeline / "ground_truth.json", blocks)
    _atomic_json(pipeline / "ground_truth_grouped.json", grouped_ground_truth)
    _atomic_json(pipeline / "eval.json", evaluation)
    eval_text_temp = pipeline / ".eval.manual.tmp"
    render_to_file(evaluation, str(eval_text_temp))
    os.replace(eval_text_temp, pipeline / "eval.txt")

    metadata_path = root / "metadata.json"
    metadata = load_json_dedup(metadata_path)
    if not isinstance(metadata, dict):
        raise ManualGroundTruthError("Run metadata.json is missing or invalid")
    test_eval = metadata.get("test_eval") or {}
    test_eval.update({
        "status": "completed",
        "accuracy": evaluation.get("accuracy"),
        "exact_match": evaluation.get("exact_match"),
        "ground_truth_source": "manual_ui_submission",
        "ground_truth_model": model,
    })
    # Codex usage is external to TWIX, so there is no measured token/cost
    # entry to aggregate or price as an OpenRouter call.
    test_eval.pop("ground_truth_cost", None)
    health_path = pipeline / "pipeline_health.json"
    if health_path.is_file():
        health = load_json_dedup(health_path)
        if isinstance(health, dict) and health.get("degenerate"):
            test_eval["degenerate_pipeline"] = True
            test_eval["pipeline_problems"] = health.get("problems", [])
    metadata["test_eval"] = test_eval
    metadata["manual_ground_truth"] = {
        "status": "submitted",
        "model": model,
        "submitted_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    totals = metadata.get("totals") or {}
    totals["duration_s"] = round(
        float(totals.get("duration_s") or 0) + (time.time() - started), 3
    )
    metadata["totals"] = totals
    _atomic_json(metadata_path, metadata)
    return evaluation
