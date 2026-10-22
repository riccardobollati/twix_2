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


def _manual_payload(raw: str) -> dict:
    """Accept flat generator blocks or explicitly grouped manual records."""
    try:
        payload = loads_json_dedup(raw)
    except json.JSONDecodeError as exc:
        raise ManualGroundTruthError(f"Ground truth is not valid JSON: {exc}") from exc
    if (not isinstance(payload, dict)
            or ("blocks" in payload) == ("records" in payload)
            or set(payload) - {"blocks", "records", "template_extension"}):
        raise ManualGroundTruthError(
            'Root requires either "blocks" or "records", with optional "template_extension"'
        )
    if "records" not in payload:
        return payload
    records = payload["records"]
    if not isinstance(records, list) or not records:
        raise ManualGroundTruthError('"records" must be a non-empty list')
    blocks = []
    seen = set()
    for index, record in enumerate(records):
        label = f"records[{index}]"
        if not isinstance(record, dict) or set(record) != {"record_id", "data"}:
            raise ManualGroundTruthError(f"{label} requires record_id and data only")
        record_id = record["record_id"]
        if not isinstance(record_id, str) or not record_id.strip():
            raise ManualGroundTruthError(f"{label}.record_id must be a non-empty string")
        if record_id in seen:
            raise ManualGroundTruthError("Each record_id must be unique")
        seen.add(record_id)
        if not isinstance(record["data"], list) or not record["data"]:
            raise ManualGroundTruthError(f"{label}.data must be a non-empty list of physical blocks")
        for block in record["data"]:
            if not isinstance(block, dict):
                raise ManualGroundTruthError(f"{label}.data must contain physical block objects")
            if "record_id" in block and block["record_id"] != record_id:
                raise ManualGroundTruthError("Block record_id must match its containing record")
            blocks.append({**block, "record_id": record_id})
    return {**payload, "blocks": blocks}


def _explicit_records(blocks: list[dict]) -> list[dict]:
    records: list[dict] = []
    for block in blocks:
        if not records or records[-1]["record_id"] != block["record_id"]:
            records.append({"record_id": block["record_id"], "data": []})
        records[-1]["data"].append(block)
    return records


def _template_specs(template: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        str(node.get("node_id")): node
        for node in template if node.get("type") in ("table", "kv")
    }


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


def _evaluation_template(payload: dict, original: list[dict], page_count: int) -> list[dict]:
    """Validate an additive, evaluation-only template and its page evidence."""
    extension = payload.get("template_extension")
    if extension is None:
        return original
    if not isinstance(extension, dict) or set(extension) != {"template", "evidence"}:
        raise ManualGroundTruthError("template_extension requires template and evidence")
    template = extension["template"]
    if not isinstance(template, list) or not all(isinstance(n, dict) for n in template):
        raise ManualGroundTruthError("Extended template must be a list of nodes")
    # Reuse the inference contract, including XOR membership/type checks.
    from src.agents.template_inference import _validate_template
    problems = _validate_template(template)
    if problems:
        raise ManualGroundTruthError("Invalid extended template: " + "; ".join(problems))
    ids = [str(n["node_id"]) for n in template]
    original_ids = [str(n["node_id"]) for n in original]
    if [i for i in ids if i in original_ids] != original_ids:
        raise ManualGroundTruthError("Keep all original nodes in their original relative order")
    by_id = dict(zip(ids, template))
    for node in original:
        updated = by_id[str(node["node_id"])]
        if node.get("type") in ("xor-table", "xor-kv"):
            if ({k: v for k, v in updated.items() if k != "includes"}
                    != {k: v for k, v in node.items() if k != "includes"}
                    or not set(map(str, node["includes"])).issubset(map(str, updated["includes"]))):
                raise ManualGroundTruthError("Existing XOR slots may only gain alternatives")
        elif updated != node:
            raise ManualGroundTruthError("Existing physical nodes must remain unchanged")
    additions = [n for n in template if str(n["node_id"]) not in original_ids]
    if not additions:
        raise ManualGroundTruthError("An extension must add a node")
    try:
        maximum = max(int(n["node_id"]) for n in original)
    except (ValueError, TypeError, KeyError) as exc:
        raise ManualGroundTruthError("Original template must have integer node IDs") from exc
    signatures = {
        (n["type"], frozenset(str(f).strip().casefold() for f in n.get("fields", [])))
        for n in original if n.get("type") in ("table", "kv")
    }
    for node in additions:
        if type(node["node_id"]) is not int or node["node_id"] <= maximum:
            raise ManualGroundTruthError("New node IDs must be fresh integers above existing IDs")
        if node["type"] in ("table", "kv"):
            fields, bids, complexity = node["fields"], node["bid"], node["complexity"]
            if (not isinstance(fields, list) or not fields
                    or not all(isinstance(f, str) and f.strip() for f in fields)
                    or len(set(fields)) != len(fields)):
                raise ManualGroundTruthError("New nodes require unique non-empty string fields")
            signature = (node["type"], frozenset(f.strip().casefold() for f in fields))
            if signature in signatures:
                raise ManualGroundTruthError("Reuse existing nodes for the same type and field signature")
            signatures.add(signature)
            if not isinstance(bids, list) or not bids or any(type(b) is not int or b < 0 for b in bids):
                raise ManualGroundTruthError("New nodes require non-negative bid positions")
            keys = ("same_row_structure", "nested") if node["type"] == "table" else ("repeated",)
            if not isinstance(complexity, dict) or any(type(complexity.get(k)) is not bool for k in keys):
                raise ManualGroundTruthError("New nodes require TWIX complexity booleans")
            if type(node.get("child")) is not int or (node["child"] != -1 and str(node["child"]) not in by_id):
                raise ManualGroundTruthError("New nodes require a valid child reference or -1")
    evidence = extension["evidence"]
    if not isinstance(evidence, list):
        raise ManualGroundTruthError("Template evidence must be a list")
    new_ids = {str(n["node_id"]) for n in additions}
    supported = set()
    for item in evidence:
        if not isinstance(item, dict) or set(item) != {"node_id", "pages", "reason"}:
            raise ManualGroundTruthError("Evidence requires node_id, pages, and reason")
        key = str(item["node_id"])
        pages = item["pages"]
        if key not in new_ids or key in supported:
            raise ManualGroundTruthError("Evidence must identify each new node exactly once")
        if not isinstance(pages, list) or not pages or any(type(p) is not int or not 1 <= p <= page_count for p in pages):
            raise ManualGroundTruthError("Evidence pages must be local held-out page numbers")
        if not isinstance(item["reason"], str) or not item["reason"].strip():
            raise ManualGroundTruthError("Evidence must explain the structured layout and record position")
        if by_id[key]["type"] in ("table", "kv"):
            for page in pages:
                if not any(str(b.get("block_id")) == key and b.get("page") == page
                           and b.get("data") for b in payload.get("blocks", []) if isinstance(b, dict)):
                    raise ManualGroundTruthError("Physical node evidence needs a non-empty block on each cited page")
        supported.add(key)
    if supported != new_ids:
        raise ManualGroundTruthError("Every new node requires evidence")
    return template


def parse_manual_ground_truth(
    raw: str,
    template: list[dict[str, Any]],
    page_count: int,
) -> list[dict[str, Any]]:
    """Validate physical blocks in a flat or explicitly grouped submission."""
    payload = _manual_payload(raw)
    blocks = payload["blocks"]
    if not isinstance(blocks, list) or not blocks:
        raise ManualGroundTruthError('"blocks" must be a non-empty list')

    template = _evaluation_template(payload, template, page_count)
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
    blocks = parse_manual_ground_truth(raw, template, page_count)
    return blocks, _evaluation_template(_manual_payload(raw), template, page_count)


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

    payload = _manual_payload(raw)
    explicit_grouping = "records" in payload
    grouped_ground_truth = (_explicit_records(blocks) if explicit_grouping
                            else group_blocks_into_records(blocks, template))
    extension = payload.get("template_extension")
    if explicit_grouping:
        candidate = load_json_dedup(candidate_path)
    elif extension is not None:
        extracted_path = pipeline / "extracted.json"
        if not extracted_path.is_file():
            raise ManualGroundTruthError("Template extension requires flat extracted.json for symmetric grouping")
        candidate = group_blocks_into_records(load_json_dedup(extracted_path), template)
    else:
        extracted_path = pipeline / "extracted.json"
        candidate = (group_blocks_into_records(load_json_dedup(extracted_path), template)
                     if extracted_path.is_file() else load_json_dedup(candidate_path))
    evaluation = eval_score(
        grouped_ground_truth,
        candidate,
        reference="model_reference",
        threshold=0.9,
    )

    if extension is not None:
        _atomic_json(pipeline / "evaluation_template.json", extension)
        # rescore.py uses these grouped artifacts; raw extraction is untouched.
    else:
        (pipeline / "evaluation_template.json").unlink(missing_ok=True)
    if not explicit_grouping:
        _atomic_json(candidate_path, candidate)
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
        "template_extended": extension is not None,
        "evaluation_template": "pipeline/evaluation_template.json" if extension is not None else None,
        "grouping": "explicit_records" if explicit_grouping else "template",
    }
    totals = metadata.get("totals") or {}
    totals["duration_s"] = round(
        float(totals.get("duration_s") or 0) + (time.time() - started), 3
    )
    metadata["totals"] = totals
    _atomic_json(metadata_path, metadata)
    return evaluation
