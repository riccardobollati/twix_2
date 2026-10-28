"""Generate and execute one document extractor, without task decomposition."""
from __future__ import annotations

import ast
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import time

import pdfplumber

from src.agents._ui import spinner
from src.agents._usage import apply_usage
from src.eval.convert import load_json_dedup
from src.ground_truth_gen.helpers import load_ocr_pages
from src.models.opencode import DEFAULT_OPENCODE_MODEL, opencode_failure_reason, opencode_silent
from src.prompts.document_extraction_prompt import document_extraction_prompt
from src.record_grouping import group_blocks_into_records

SCRIPT_TIMEOUT_S = 300
MAX_ATTEMPTS = 3


def _save_cost(state: dict, root: Path, entry: dict) -> None:
    """Keep completed generation usage available if final reference calls fail."""
    phases = list(state.get("cost") or []) + [entry]
    directory = root / "metadata"
    directory.mkdir(exist_ok=True)
    (directory / "cost.json").write_text(json.dumps({
        "phases": phases,
        "total_duration_s": round(sum(p.get("duration_s", 0) for p in phases), 3),
    }, indent=2), encoding="utf-8")


def _validate_script(script: Path) -> None:
    tree = ast.parse(script.read_text(encoding="utf-8"))
    allowed = sys.stdlib_module_names | {"pdfplumber"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules = [alias.name.split(".")[0] for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                raise ValueError("extract.py must not depend on relative imports")
            modules = [(node.module or "").split(".")[0]]
        else:
            continue
        if set(modules) - allowed:
            raise ValueError("extract.py may import only the standard library and pdfplumber")


def finalize_output(out_dir: Path, template: list[dict], page_count: int) -> None:
    """Validate the scorer contract and apply shared grouping, without repairing data."""
    blocks = load_json_dedup(out_dir / "extracted.json")
    if not isinstance(blocks, list):
        raise ValueError("extracted.json must be a flat list of physical blocks")
    nodes = {str(n["node_id"]): n for n in template if n.get("type") in {"table", "kv"}}
    uids: set[str] = set()
    instances: set[tuple[int, str]] = set()
    previous_page = 0
    for block in blocks:
        if not isinstance(block, dict):
            raise ValueError("Every block must be an object")
        page = block.get("page")
        if type(page) is not int or not 1 <= page <= page_count or page < previous_page:
            raise ValueError("Blocks need valid split-local page numbers in page order")
        previous_page = page
        block_id = block.get("block_id")
        if not isinstance(block_id, str) or "_" not in block_id:
            raise ValueError("block_id must be <node_id>_<page-local-instance>")
        node_id, instance = block_id.rsplit("_", 1)
        node = nodes.get(node_id)
        if not node or not instance or block.get("type") != node["type"]:
            raise ValueError(f"Unknown physical template node or incorrect type: {block_id}")
        key = (page, block_id)
        uid = block.get("uid")
        if key in instances or not isinstance(uid, str) or not uid or uid in uids:
            raise ValueError("Block instances must be page-local unique and uid globally unique")
        instances.add(key)
        uids.add(uid)
        data = block.get("data")
        rows = data if node["type"] == "table" else [data]
        if not isinstance(rows, list) or any(
            not isinstance(row, dict) or set(row) != set(node["fields"])
            or any(not isinstance(value, str) for value in row.values()) for row in rows
        ):
            raise ValueError(f"{block_id}: data must use exact template fields and string values")
    records = group_blocks_into_records(blocks, template)
    (out_dir / "grouped.json").write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    covered = {block["page"] for block in blocks}
    problems = []
    if not blocks:
        problems.append("Extraction is empty")
    if len(covered) < page_count / 2:
        problems.append("Blocks cover fewer than half the input pages")
    (out_dir / "pipeline_health.json").write_text(json.dumps({
        "degenerate": bool(problems), "problems": problems,
        "total_pages": page_count, "pages_with_blocks": len(covered),
        "blocks_extracted": len(blocks),
    }, indent=2), encoding="utf-8")


def run_extractor(script: Path, pdf: Path, output: Path, template: list[dict]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    # Never accept stale results after a successful process that wrote nothing.
    for name in ("extracted.json", "grouped.json", "pipeline_health.json"):
        (output / name).unlink(missing_ok=True)
    try:
        result = subprocess.run(
            [sys.executable, str(script.resolve()), str(pdf.resolve()), "--out-dir", str(output.resolve())],
            cwd=output, capture_output=True, text=True, timeout=SCRIPT_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"extract.py timed out after {SCRIPT_TIMEOUT_S}s") from exc
    if result.returncode:
        raise RuntimeError(f"extract.py failed (exit {result.returncode}):\n{result.stderr}")
    with pdfplumber.open(pdf) as document:
        page_count = len(document.pages)
    finalize_output(output, template, page_count)
    if result.stdout:
        print(result.stdout, end="", flush=True)


def document_extraction_scripting_node(state: dict) -> dict:
    """One code-generation task, scoped to the cluster-selected training inputs."""
    started = time.monotonic()
    root = Path(state["agent_space"]).parent
    # Do not use agent_space/train as CWD: in no-eval mode it may expose source
    # artifacts, and in eval mode test inputs must never reach the coding model.
    workspace = Path(state["agent_space"]) / "document_codegen"
    workspace.mkdir(parents=True, exist_ok=True)
    shutil.copy2(state["pdf_path"], workspace / "training.pdf")
    shutil.copy2(state["ocr_csv_path"], workspace / "training_ocr.csv")
    template = {"template": state["template"], "metadata": state.get("metadata", [])}
    template_json = json.dumps(template, indent=2, ensure_ascii=False)
    (workspace / "template.json").write_text(template_json, encoding="utf-8")
    pages = load_ocr_pages(state["ocr_csv_path"], range(1, state["actual_train_pages"] + 1))
    layouts = "\n\n".join(f"Training page {page} (split-local):\n{data['layout']}" for page, data in pages.items())
    (workspace / "training_layout.txt").write_text(layouts, encoding="utf-8")
    base_prompt = document_extraction_prompt(template_json, layouts)
    prompt = base_prompt
    script = workspace / "extract.py"
    usage = [0, 0, 0]
    model = state.get("opencode_model") or DEFAULT_OPENCODE_MODEL
    print(f"document_extraction_scripting_node: generating one extractor with {model}", flush=True)
    for attempt in range(1, MAX_ATTEMPTS + 1):
        script.unlink(missing_ok=True)
        stop = threading.Event()
        status: dict = {}
        thread = threading.Thread(target=spinner, args=("document_extraction_scripting", stop, status), daemon=True)
        thread.start()
        error = None
        try:
            code, inp, out, total = opencode_silent(
                prompt, str(workspace), task="document_extraction_scripting", model=model,
                caveman=state.get("use_caveman", False),
                log_path=str(root / "debug" / f"document_extraction_{attempt}.log") if state.get("save_log") else None,
            )
            for index, value in enumerate((inp, out, total)):
                usage[index] += value or 0
            if code:
                raise RuntimeError(opencode_failure_reason(code))
            _validate_script(script)
            # Contract-only validation on training inputs. No reference truth,
            # accuracy report, or test data is returned to the coding model.
            with tempfile.TemporaryDirectory(prefix="twix-document-validation-") as directory:
                validation = Path(directory)
                shutil.copy2(script, validation / "extract.py")
                shutil.copy2(workspace / "training.pdf", validation / "input.pdf")
                run_extractor(validation / "extract.py", validation / "input.pdf", validation, state["template"])
        except (OSError, ValueError, SyntaxError, RuntimeError) as exc:
            error = str(exc)
        finally:
            status["tokens"] = sum(usage) or None
            status["ok"] = error is None
            stop.set()
            thread.join(timeout=0.5)
        if error is None:
            break
        print(f"document_extraction_scripting_node: attempt {attempt} failed: {error}", flush=True)
        prompt = base_prompt + f"\nPrevious attempt failed the execution/output contract:\n{error}\nWrite a complete corrected extract.py."
    entry = {"phase": "document_extraction_scripting", "model": model,
             "attempts": attempt, "duration_s": round(time.monotonic() - started, 3)}
    apply_usage(entry, *usage)
    _save_cost(state, root, entry)
    if error is not None:
        raise RuntimeError(f"document extractor generation failed after {MAX_ATTEMPTS} attempts: {error}")
    pipeline = root / "pipeline"
    pipeline.mkdir(exist_ok=True)
    shutil.copy2(script, pipeline / "extract.py")
    (pipeline / "template.json").write_text(template_json, encoding="utf-8")
    (pipeline / "requirements.txt").write_text("pdfplumber\n", encoding="utf-8")
    return {"cost": [entry]}


def document_extraction_node(state: dict) -> dict:
    started = time.monotonic()
    pipeline = Path(state["agent_space"]).parent / "pipeline"
    pdf = Path(state.get("test_pdf_path") or state["pdf_path"])
    print("extraction_node: running the generated document extractor", flush=True)
    run_extractor(pipeline / "extract.py", pdf, pipeline, state["template"])
    entry = {"phase": "extraction", "duration_s": round(time.monotonic() - started, 3)}
    _save_cost(state, pipeline.parent, entry)
    return {"cost": [entry]}
