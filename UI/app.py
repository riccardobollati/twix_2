#!/usr/bin/env python3
"""Local browser dashboard for running and inspecting TWIX 2.

The server deliberately binds to localhost by default: a run may use the local
OpenRouter/OpenCode configuration and can expose PDFs, OCR, and debug logs.
No data is uploaded anywhere by this dashboard itself.
"""

from __future__ import annotations

import argparse
import json
import math
import mimetypes
import os
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

UI_DIR = Path(__file__).resolve().parent
PROJECT_DIR = UI_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.models.catalog import (
    DEFAULT_DIRECT_MODEL,
    MODEL_OPTIONS,
    REFERENCE_MODEL_OPTIONS,
    direct_model_id,
    resolve_easy_model,
    resolve_hard_model,
    resolve_reference_model,
)


RUNS_DIR = UI_DIR / "runs"
MAX_UPLOAD_BYTES = 500 * 1024 * 1024
PREVIEW_BYTES = 600_000
_ANSI_ESCAPE = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|[@-_][0-?]*[ -/]*[@-~])")
_SPINNER_FRAMES = frozenset("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏")
_STATS_RENDER_LOCK = threading.Lock()


def _pipeline_python() -> str:
    """Use the project's virtual environment when it is available.

    The dashboard can be started with any local ``python3``.  The actual TWIX
    run, however, needs the repository environment where pdfplumber, LangGraph,
    and the model clients are installed.
    """
    venv_python = PROJECT_DIR / "env" / "bin" / "python"
    return str(venv_python) if venv_python.is_file() and os.access(venv_python, os.X_OK) else sys.executable

STAGES = [
    ("template", "Template inference"),
    ("blocks", "Block separation"),
    ("sonar", "Missed sonar"),
    ("record_separation", "Record separation"),
    ("extraction", "Data extraction"),
    ("results", "Test evaluation & results"),
]


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _safe_filename(name: str) -> str:
    name = Path(name or "input.pdf").name
    cleaned = re.sub(r"[^A-Za-z0-9._ -]", "_", name).strip(" .")
    return cleaned or "input.pdf"


def _normalise_log_line(raw: str) -> str | None:
    """Make a terminal line readable in a browser log.

    Agent spinners redraw one terminal line with carriage returns and ANSI erase
    sequences. A pipe receives every redraw, but a browser cannot reproduce the
    terminal's in-place update. Keep only the final spinner result (a ✓ or ✗
    line) and discard interim frames.
    """
    clean = _ANSI_ESCAPE.sub("", raw).replace("\b", "")
    if "\r" in clean:
        clean = clean.rsplit("\r", 1)[-1]
    clean = clean.strip()
    if not clean:
        return None
    if any(frame in clean for frame in _SPINNER_FRAMES) and not any(
        marker in clean for marker in ("✓", "✗")
    ):
        return None
    return clean


def _dedupe_log_lines(lines: list[str]) -> list[str]:
    """Collapse terminal redraw artifacts while preserving real repeated output."""
    result: list[str] = []
    seen_completions: set[str] = set()
    for line in lines:
        if any(marker in line for marker in ("✓", "✗")):
            if line in seen_completions:
                continue
            seen_completions.add(line)
        if result and result[-1] == line:
            continue
        result.append(line)
    return result


def _safe_artifact_path(run_dir: Path, requested: str) -> Path | None:
    try:
        target = (run_dir / requested).resolve()
        target.relative_to(run_dir.resolve())
    except (OSError, ValueError):
        return None
    return target


def _stage_from_logs(lines: list[str], eval_enabled: bool) -> str:
    text = "\n".join(lines).lower()
    if re.search(r"(?:^|\n)(?:extraction|ground_truth|eval)_node:", text):
        return "results"
    markers = [
        ("results", ("generating test ground truth", "pipeline/eval.json")),
        (
            "extraction",
            (
                "block_samples", "block_difficulty_router",
                "block_extraction_scripting", "collect_extraction_scripts",
                "train_eval_node:", "train evaluation", "train_repair",
                "starting repair attempt",
            ),
        ),
        ("record_separation", ("record image(s)",)),
        ("sonar", ("excluded_blocks_sonar_node:",)),
        ("blocks", ("block_separation", "block_artifacts_node:")),
        (
            "template",
            (
                "template_inference", "cropped source pages", "train: source pages",
                "structured data starts", "extracting full ocr", "ocr saved",
            ),
        ),
    ]
    for stage, phrases in markers:
        if any(phrase in text for phrase in phrases):
            return stage
    return "template"


def _load_json(path: Path) -> Any | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _relative_file(run_dir: Path, path: Path | None) -> str | None:
    if path is None or not path.is_file():
        return None
    try:
        return path.relative_to(run_dir).as_posix()
    except ValueError:
        return None


def _first_pdf(folder: Path) -> Path | None:
    if not folder.is_dir():
        return None
    return next(iter(sorted(folder.glob("*.pdf"))), None)


def _sonar_script(run_dir: Path) -> Path | None:
    """Locate the saved separator in current and older UI run layouts."""
    for path in (
        run_dir / "pipeline" / "block_separation.py",
        run_dir / "agent_space" / "train" / "block_separation.py",
        run_dir / "agent_space" / "block_separation.py",
    ):
        if path.is_file():
            return path
    return None


def _page_sort_key(path: Path) -> tuple[int, str]:
    match = re.search(r"page_(\d+)_", path.name)
    return (int(match.group(1)) if match else 10**9, path.name)


def _effective_model_assignments(run: "Run") -> list[dict[str, Any]]:
    """Describe which effective model is responsible for each pipeline role."""
    metadata = _load_json(run.run_dir / "metadata.json") or {}
    config = run.config
    single_model_mode = bool(config.get("noRouting", False))

    if single_model_mode:
        easy_model = resolve_easy_model(str(config.get("singleModel") or ""))
        hard_model = easy_model
        primary_model = direct_model_id(easy_model)
    else:
        easy_model = resolve_easy_model(str(config.get("easyModel") or ""))
        hard_model = resolve_hard_model(str(config.get("hardModel") or ""))
        primary_model = os.getenv("OPENROUTER_MODEL", DEFAULT_DIRECT_MODEL).strip()

    opencode_model = hard_model
    difficulty_model = direct_model_id(hard_model)

    primary_model = metadata.get("llm_model") or primary_model
    reference_model = (
        metadata.get("reference_model")
        or metadata.get("ground_truth_model")
        or resolve_reference_model(str(config.get("referenceModel") or ""))
    )
    opencode_model = metadata.get("opencode_model") or opencode_model
    easy_model = metadata.get("easy_model") or easy_model
    hard_model = metadata.get("hard_model") or hard_model
    if metadata.get("difficulty_model"):
        difficulty_model = metadata["difficulty_model"]
    elif metadata:
        # Metadata from runs created before the dedicated routing-model field
        # reflects the former primary-model behavior.
        difficulty_model = primary_model

    assignments: list[dict[str, Any]] = [
        {
            "id": "beginning_detection",
            "stage": "template",
            "task": "Beginning detection",
            "model": direct_model_id(primary_model),
            "transport": "OpenRouter direct",
        },
        {
            "id": "template_inference",
            "stage": "template",
            "task": "Template inference",
            "model": direct_model_id(reference_model),
            "transport": "Reference · OpenRouter direct",
        },
        {
            "id": "block_separation",
            "stage": "blocks",
            "task": "Block-separation script",
            "model": direct_model_id(opencode_model),
            "transport": "OpenCode via OpenRouter",
        },
        {
            "id": "excluded_blocks_sonar",
            "stage": "sonar",
            "task": "Excluded blocks sonar",
            "model": "Generated Python separator",
            "transport": "No model call",
        },
        {
            "id": "record_grouping",
            "stage": "record_separation",
            "task": "Deterministic record grouping",
            "model": "Template-node traversal",
            "transport": "No model call",
        },
    ]

    if not single_model_mode:
        assignments.append({
            "id": "difficulty_routing",
            "stage": "extraction",
            "task": "Block difficulty routing",
            "model": direct_model_id(difficulty_model),
            "transport": "OpenRouter direct",
        })

    if single_model_mode:
        assignments.append({
            "id": "all_extractors",
            "stage": "extraction",
            "task": "All block extractors and repairs",
            "model": direct_model_id(easy_model),
            "transport": "OpenCode via OpenRouter · single model",
        })
    else:
        assignments.extend([
            {
                "id": "easy_extractors",
                "stage": "extraction",
                "task": "Easy block extractors",
                "model": direct_model_id(easy_model),
                "transport": "OpenCode via OpenRouter",
            },
            {
                "id": "hard_extractors",
                "stage": "extraction",
                "task": "Hard block extractors and repairs",
                "model": direct_model_id(hard_model),
                "transport": "OpenCode via OpenRouter",
            },
        ])

    assignments.extend([
        {
            "id": "ground_truth",
            "stage": "results",
            "stages": ["extraction", "results"],
            "task": "Train/test ground truth",
            "model": direct_model_id(reference_model),
            "transport": "Reference · OpenRouter direct",
        },
        {
            "id": "portable_extraction",
            "stage": "results",
            "task": "Test PDF extraction",
            "model": "Generated Python pipeline",
            "transport": "No model call",
        },
        {
            "id": "evaluation",
            "stage": "results",
            "task": "Accuracy evaluation",
            "model": "Local evaluator",
            "transport": "No model call",
        },
    ])

    block_models = metadata.get("block_models") or {}
    if block_models:
        mapping = " · ".join(
            f"{block_id} → {direct_model_id(str(model))}"
            for block_id, model in sorted(block_models.items(), key=lambda item: str(item[0]))
        )
        assignments.append({
            "id": "saved_block_assignments",
            "stage": "extraction",
            "task": "Saved per-block assignments",
            "model": mapping,
            "transport": "Actual routing used by the completed run",
            "detail": True,
        })
    return assignments


def _active_model_assignment_ids(
    lines: list[str], stage: str, single_model_mode: bool
) -> list[str]:
    """Infer the currently relevant model role from the latest console marker."""
    text = "\n".join(lines[-120:]).lower()
    extraction_ids = ["all_extractors"] if single_model_mode else [
        "easy_extractors", "hard_extractors"
    ]
    candidates: dict[str, list[tuple[tuple[str, ...], list[str]]]] = {
        "template": [
            (("detect_beginning", "structured data starts"), ["beginning_detection"]),
            (("template_inference",), ["template_inference"]),
        ],
        "blocks": [
            (("block_separation", "block_artifacts_node"), ["block_separation"]),
        ],
        "sonar": [
            (("excluded_blocks_sonar_node:",), ["excluded_blocks_sonar"]),
        ],
        "record_separation": [
            (("record image(s)",), ["record_grouping"]),
        ],
        "extraction": [
            (("block_difficulty_router",), [] if single_model_mode else ["difficulty_routing"]),
            (("block_extraction_scripting",), extraction_ids),
            (("train_repair", "starting repair attempt"), ["all_extractors"] if single_model_mode else ["hard_extractors"]),
            (("train_eval_node", "train ground truth"), ["ground_truth"]),
        ],
        "results": [
            (("extraction_node:",), ["portable_extraction"]),
            (("ground_truth", "generating test ground truth"), ["ground_truth"]),
            (("eval_node:", "pipeline/eval.json"), ["evaluation"]),
        ],
    }
    ranked: list[tuple[int, list[str]]] = []
    for phrases, assignment_ids in candidates.get(stage, []):
        position = max((text.rfind(phrase) for phrase in phrases), default=-1)
        if position >= 0:
            ranked.append((position, assignment_ids))
    if ranked:
        return max(ranked, key=lambda item: item[0])[1]
    defaults = {
        "template": ["beginning_detection"],
        "blocks": ["block_separation"],
        "sonar": ["excluded_blocks_sonar"],
        "record_separation": ["record_grouping"],
        "extraction": extraction_ids,
        "results": ["portable_extraction"],
    }
    return defaults.get(stage, [])


def _workflow_payload(run: "Run") -> dict[str, Any]:
    """Collect the user-facing artifacts for the workflow stages."""
    with run.lock:
        logs = list(run.logs)
    root = run.run_dir
    pipeline = root / "pipeline"
    train_pdf = _first_pdf(root / "agent_space" / "train")
    test_pdf = _first_pdf(root / "agent_space" / "test")
    source_pdf = root / run.filename

    test_blocks_dir = root / "artifacts" / "test_blocks"
    train_blocks_dir = root / "artifacts" / "blocks"
    block_dir = test_blocks_dir if any(test_blocks_dir.glob("page_*_blocks.png")) else train_blocks_dir
    block_images = [
        _relative_file(root, path)
        for path in sorted(block_dir.glob("page_*_blocks.png"), key=_page_sort_key)
    ]
    test_records_dir = root / "artifacts" / "test_record_separation"
    train_records_dir = root / "artifacts" / "record_separation"
    record_dir = (
        test_records_dir
        if any(test_records_dir.glob("page_*_records.png"))
        else train_records_dir
    )
    record_images = [
        _relative_file(root, path)
        for path in sorted(record_dir.glob("page_*_records.png"), key=_page_sort_key)
    ]
    stats_dir = root / "artifacts" / "stats"
    sonar_dir = root / "artifacts" / "excluded_blocks_sonar"
    stats_images = [
        _relative_file(root, stats_dir / filename)
        for filename in ("overview.png", "phase_cost.png", "token_breakdown.png")
    ]

    stage = _stage_from_logs(logs, run.eval_enabled)
    metadata = _load_json(root / "metadata.json") or {}
    source_start = metadata.get("source_start_page", 1)
    if not isinstance(source_start, int) or source_start < 1:
        source_start = 1
    return {
        "stage": stage,
        "template": _load_json(root / "template.json"),
        "templatePdf": _relative_file(root, train_pdf or source_pdf),
        "testPdf": _relative_file(root, test_pdf or source_pdf),
        "blockImages": [path for path in block_images if path],
        "blockSource": "test" if block_dir == test_blocks_dir else "train",
        "recordImages": [path for path in record_images if path],
        "recordSource": "test" if record_dir == test_records_dir else "train",
        "extracted": _load_json(pipeline / "extracted.json"),
        "groundTruth": _load_json(pipeline / "ground_truth.json"),
        "evaluation": _load_json(pipeline / "eval.json"),
        "trainEvaluation": _load_json(pipeline / "train_eval" / "eval.json"),
        "statistics": _load_json(stats_dir / "stats.json"),
        "statsImages": [path for path in stats_images if path],
        "sonar": _load_json(sonar_dir / "sonar.json"),
        "sonarError": _load_json(sonar_dir / "error.json"),
        "sonarAvailable": (
            source_pdf.is_file() and _sonar_script(root) is not None
            and run.status not in {"queued", "running"}
        ),
        "sonarStartPage": source_start,
        "modelAssignments": _effective_model_assignments(run),
        "activeModelAssignments": _active_model_assignment_ids(
            logs, stage, bool(run.config.get("noRouting", False))
        ),
    }


def _file_tree(root: Path, maximum_entries: int = 1_500) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    try:
        paths = sorted(root.rglob("*"), key=lambda p: (not p.is_dir(), str(p).lower()))
    except OSError:
        return entries
    for path in paths[:maximum_entries]:
        try:
            relative = path.relative_to(root).as_posix()
            info: dict[str, Any] = {"path": relative, "name": path.name, "directory": path.is_dir()}
            if path.is_file():
                info["size"] = path.stat().st_size
            entries.append(info)
        except OSError:
            continue
    return entries


@dataclass
class Run:
    run_id: str
    run_dir: Path
    filename: str
    config: dict[str, Any]
    status: str = "queued"
    started_at: str | None = None
    finished_at: str | None = None
    return_code: int | None = None
    command: str | None = None
    logs: list[str] = field(default_factory=list)
    process: subprocess.Popen[str] | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)
    sonar_lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def eval_enabled(self) -> bool:
        return bool(self.config.get("eval", True))

    def snapshot(self, include_logs: bool = False) -> dict[str, Any]:
        with self.lock:
            result: dict[str, Any] = {
                "id": self.run_id,
                "filename": self.filename,
                "status": self.status,
                "startedAt": self.started_at,
                "finishedAt": self.finished_at,
                "returnCode": self.return_code,
                "command": self.command,
                "config": self.config,
                "stage": _stage_from_logs(self.logs, self.eval_enabled),
                "artifactCount": len(_file_tree(self.run_dir)),
                "logCursor": len(self.logs),
            }
            if include_logs:
                result["logs"] = self.logs[-1_000:]
            return result

    def persist(self) -> None:
        payload = self.snapshot(include_logs=False)
        payload["filename"] = self.filename
        try:
            (self.run_dir / "ui_run.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except OSError:
            pass


def _build_run_command(run: Run, input_path: Path) -> list[str]:
    """Translate dashboard settings into the equivalent ``run.py`` command."""
    command = [
        _pipeline_python(), str(PROJECT_DIR / "run.py"), str(input_path),
        "--out-dir", str(run.run_dir), "--pages", str(run.config["pages"]),
        "--holdout-pages", str(run.config.get("holdoutPages", 0)),
        "--sonar-drop-percent", str(run.config.get("sonarDropPercent", 15)),
        "--save-log",
    ]
    if run.config["eval"]:
        command.extend([
            "--eval", "--eval-pages", str(run.config.get("evalPages", 20)),
        ])
    else:
        command.append("--no-eval")

    reference_model = str(run.config.get("referenceModel", "")).strip()
    if reference_model:
        command.extend(["--reference-model", reference_model])

    if run.config.get("noRouting", False):
        command.append("--no-routing")
        single_model = str(run.config.get("singleModel", "")).strip()
        if single_model:
            command.extend(["--easy-model", single_model])
    else:
        easy_model = str(run.config.get("easyModel", "")).strip()
        hard_model = str(run.config.get("hardModel", "")).strip()
        if easy_model:
            command.extend(["--easy-model", easy_model])
        if hard_model:
            command.extend(["--hard-model", hard_model])

    if run.config["trainThreshold"] > 0:
        command.extend(["--train-threshold", str(run.config["trainThreshold"])])
        command.extend(["--train-max-retries", str(run.config["trainMaxRetries"])])
    return command


def _generate_run_statistics(run_dir: Path) -> dict:
    """Render the reusable per-document report for one completed UI run."""
    from viz_batch_results import generate_document_results

    with _STATS_RENDER_LOCK:
        return generate_document_results(
            str(run_dir), str(run_dir / "artifacts" / "stats")
        )


class RunManager:
    def __init__(self) -> None:
        self.runs: dict[str, Run] = {}
        self.lock = threading.Lock()
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        self._restore_completed_runs()

    def _restore_completed_runs(self) -> None:
        for directory in sorted(RUNS_DIR.glob("run-*"), reverse=True):
            state_path = directory / "ui_run.json"
            if not state_path.is_file():
                continue
            try:
                saved = json.loads(state_path.read_text(encoding="utf-8"))
                run = Run(
                    run_id=saved["id"], run_dir=directory, filename=saved.get("filename", "input.pdf"),
                    config=saved.get("config", {}), status=saved.get("status", "interrupted"),
                    started_at=saved.get("startedAt"), finished_at=saved.get("finishedAt"),
                    return_code=saved.get("returnCode"),
                    command=saved.get("command"),
                )
                if run.status in {"queued", "running"}:
                    run.status = "interrupted"
                log_path = directory / "ui_run.log"
                if log_path.is_file():
                    run.logs = _dedupe_log_lines([
                        clean for raw in log_path.read_text(encoding="utf-8", errors="replace").splitlines()
                        if (clean := _normalise_log_line(raw))
                    ])[-1_000:]
                self.runs[run.run_id] = run
            except (OSError, KeyError, json.JSONDecodeError):
                continue

    def create(self, filename: str, data: bytes, config: dict[str, Any]) -> Run:
        run_id = f"run-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        run_dir = RUNS_DIR / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        filename = _safe_filename(filename)
        input_path = run_dir / filename
        input_path.write_bytes(data)
        run = Run(run_id=run_id, run_dir=run_dir, filename=filename, config=config)
        with self.lock:
            self.runs[run_id] = run
        run.persist()
        self._start(run, input_path)
        return run

    def get(self, run_id: str) -> Run | None:
        with self.lock:
            return self.runs.get(run_id)

    def list(self) -> list[dict[str, Any]]:
        with self.lock:
            runs = list(self.runs.values())
        return [run.snapshot() for run in sorted(runs, key=lambda r: r.run_id, reverse=True)]

    def cancel(self, run: Run) -> bool:
        with run.lock:
            if run.status != "running" or run.process is None:
                return False
            run.process.terminate()
            run.logs.append("[UI] Cancellation requested.")
            return True

    def _start(self, run: Run, input_path: Path) -> None:
        command = _build_run_command(run, input_path)
        run.command = shlex.join(command)

        def execute() -> None:
            with run.lock:
                run.status = "running"
                run.started_at = _utc_now()
                run.logs.append("[UI] Starting TWIX pipeline.")
            run.persist()
            log_path = run.run_dir / "ui_run.log"
            seen_completions: set[str] = set()
            try:
                process = subprocess.Popen(
                    command, cwd=PROJECT_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, bufsize=1,
                )
                with run.lock:
                    run.process = process
                with log_path.open("a", encoding="utf-8") as log_file:
                    for line in process.stdout or []:
                        line = _normalise_log_line(line)
                        if line is None:
                            continue
                        with run.lock:
                            is_completion = any(
                                marker in line for marker in ("✓", "✗")
                            )
                            if is_completion and line in seen_completions:
                                continue
                            if is_completion:
                                seen_completions.add(line)
                            if run.logs and run.logs[-1] == line:
                                continue
                            run.logs.append(line)
                        log_file.write(line + "\n")
                        log_file.flush()
                code = process.wait()
                with run.lock:
                    run.return_code = code
                    run.process = None
                if code == 0:
                    with run.lock:
                        run.logs.append("[UI] Generating plots and run statistics.")
                    try:
                        result = _generate_run_statistics(run.run_dir)
                        with run.lock:
                            run.logs.append(
                                f"[UI] Statistics ready ({len(result['plots'])} plots)."
                            )
                    except Exception as exc:
                        with run.lock:
                            run.logs.append(
                                f"[UI] Warning: statistics could not be generated: {exc}"
                            )
                with run.lock:
                    run.status = "completed" if code == 0 else "failed"
                    run.finished_at = _utc_now()
                    run.logs.append(
                        f"[UI] Pipeline {'completed' if code == 0 else 'failed'} "
                        f"(exit {code})."
                    )
            except Exception as exc:  # surfaced in the dashboard, never swallowed
                with run.lock:
                    run.status = "failed"
                    run.finished_at = _utc_now()
                    run.logs.append(f"[UI] Could not start pipeline: {exc}")
            run.persist()

        threading.Thread(target=execute, name=run.run_id, daemon=True).start()


MANAGER = RunManager()


class Handler(BaseHTTPRequestHandler):
    server_version = "TwixUI/0.1"

    def log_message(self, format: str, *args: object) -> None:
        # Browser requests are already visible in the dashboard; keep the terminal useful.
        return

    def _send_json(self, payload: Any, status: int = HTTPStatus.OK) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _send_file(self, path: Path) -> None:
        if not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        mime, _ = mimetypes.guess_type(str(path))
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mime or "application/octet-stream")
        self.send_header("Content-Length", str(path.stat().st_size))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        with path.open("rb") as source:
            shutil.copyfileobj(source, self.wfile)

    def _run_from_path(self, parts: list[str]) -> Run | None:
        return MANAGER.get(parts[3]) if len(parts) >= 4 else None

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        if path == "/":
            return self._send_file(UI_DIR / "index.html")
        if path in {"/app.js", "/styles.css", "/sonar.css"}:
            return self._send_file(UI_DIR / path.lstrip("/"))
        if path == "/api/runs":
            return self._send_json({"runs": MANAGER.list()})
        if path == "/api/models":
            return self._send_json({
                "models": [
                    {"label": label, "value": value}
                    for label, value in MODEL_OPTIONS
                ],
                "referenceModels": [
                    {"label": label, "value": value}
                    for label, value in REFERENCE_MODEL_OPTIONS
                ],
            })

        parts = path.split("/")
        run = self._run_from_path(parts)
        if run is None:
            self.send_error(HTTPStatus.NOT_FOUND, "Run not found")
            return
        if len(parts) == 4:
            return self._send_json(run.snapshot(include_logs=True))
        if len(parts) == 5 and parts[4] == "tree":
            return self._send_json({"entries": _file_tree(run.run_dir)})
        if len(parts) == 5 and parts[4] == "events":
            query = parse_qs(parsed.query)
            try:
                cursor = max(0, int(query.get("cursor", ["0"])[0]))
            except ValueError:
                cursor = 0
            return self._stream_events(run, cursor)
        if len(parts) == 5 and parts[4] == "comparison":
            return self._send_comparison(run)
        if len(parts) == 5 and parts[4] == "workflow":
            return self._send_json(_workflow_payload(run))
        if len(parts) == 7 and parts[4:6] == ["sonar", "page"]:
            try:
                page = int(parts[6])
                from src.excluded_blocks_sonar import render_page
                with run.sonar_lock:
                    preview = render_page(run.run_dir / "artifacts" / "excluded_blocks_sonar", page)
                return self._send_file(preview)
            except (ValueError, OSError, KeyError, FileNotFoundError):
                return self._send_json({"error": "Page is outside the sonar scan."}, HTTPStatus.BAD_REQUEST)
        if len(parts) >= 6 and parts[4] in {"preview", "artifact"}:
            requested = "/".join(parts[5:])
            target = _safe_artifact_path(run.run_dir, requested)
            if target is None:
                self.send_error(HTTPStatus.BAD_REQUEST, "Invalid artifact path")
                return
            if parts[4] == "artifact":
                return self._send_file(target)
            return self._send_preview(target, requested)
        self.send_error(HTTPStatus.NOT_FOUND)

    def _stream_events(self, run: Run, cursor: int = 0) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        try:
            while True:
                with run.lock:
                    cursor = min(cursor, len(run.logs))
                    logs = run.logs[cursor:]
                    cursor = len(run.logs)
                snapshot = run.snapshot()
                payload = json.dumps({"run": snapshot, "logs": logs}, ensure_ascii=False)
                self.wfile.write(f"data: {payload}\n\n".encode("utf-8"))
                self.wfile.flush()
                if snapshot["status"] in {"completed", "failed", "interrupted"}:
                    return
                time.sleep(0.8)
        except (BrokenPipeError, ConnectionResetError):
            return

    def _send_preview(self, target: Path, requested: str) -> None:
        if not target.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        if target.suffix.lower() not in {".json", ".py", ".txt", ".csv", ".log", ".md", ".jsonl"}:
            return self._send_json({"path": requested, "kind": "binary", "size": target.stat().st_size})
        raw = target.read_bytes()
        truncated = len(raw) > PREVIEW_BYTES
        content = raw[:PREVIEW_BYTES].decode("utf-8", errors="replace")
        if target.suffix.lower() == ".json":
            try:
                content = json.dumps(json.loads(content), ensure_ascii=False, indent=2)
            except json.JSONDecodeError:
                pass
        return self._send_json({"path": requested, "kind": "text", "content": content, "truncated": truncated})

    def _send_comparison(self, run: Run) -> None:
        pipeline = run.run_dir / "pipeline"
        payload: dict[str, Any] = {}
        for key, filename in (("extracted", "extracted.json"), ("groundTruth", "ground_truth.json"), ("evaluation", "eval.json")):
            path = pipeline / filename
            if path.is_file():
                try:
                    payload[key] = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    payload[key] = None
        return self._send_json(payload)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        if path == "/api/runs":
            return self._create_run(parsed)
        parts = path.split("/")
        run = self._run_from_path(parts)
        if run and len(parts) == 5 and parts[4] == "cancel":
            return self._send_json({"cancelled": MANAGER.cancel(run)})
        if run and len(parts) == 5 and parts[4] == "ground-truth":
            return self._submit_ground_truth(run)
        if run and len(parts) == 6 and parts[4:6] == ["sonar", "scan"]:
            return self._scan_sonar(run)
        self.send_error(HTTPStatus.NOT_FOUND)

    def _scan_sonar(self, run: Run) -> None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length < 4096:
                raise ValueError("Invalid scan request")
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("Scan request must be an object")
            current = _load_json(run.run_dir / "artifacts" / "excluded_blocks_sonar" / "sonar.json")
            metadata = _load_json(run.run_dir / "metadata.json") or {}
            default_start = metadata.get("source_start_page", 1)
            if not isinstance(default_start, int) or default_start < 1:
                default_start = 1
            start = int(payload.get("startPage", current["startPage"] if current else default_start))
            raw_end = payload.get("endPage", current["endPage"] if current else None)
            end = int(raw_end) if raw_end is not None else None
            threshold = float(payload.get("thresholdPercent", run.config.get("sonarDropPercent", 15)))
            if start < 1 or (end is not None and end < start):
                raise ValueError("Invalid sonar page range")
            if current and end is not None and end > int(current["totalPages"]):
                raise ValueError(f"Page range must be within 1–{current['totalPages']}")
            if not math.isfinite(threshold) or not 0 <= threshold <= 100:
                raise ValueError("Drop threshold must be between 0 and 100 percent")
            source_pdf = run.run_dir / run.filename
            script = _sonar_script(run.run_dir)
            if not source_pdf.is_file() or script is None:
                raise ValueError("This run needs its source PDF and saved block separator")
        except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            return self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        if not run.sonar_lock.acquire(blocking=False):
            return self._send_json({"error": "A sonar scan or page render is already running"}, HTTPStatus.CONFLICT)
        try:
            command = [
                _pipeline_python(), "-m", "src.excluded_blocks_sonar",
                str(source_pdf), str(script),
                str(run.run_dir / "artifacts" / "excluded_blocks_sonar"),
                "--start", str(start), "--threshold", str(threshold),
            ]
            if end is not None:
                command.extend(["--end", str(end)])
            result = subprocess.run(command, cwd=PROJECT_DIR, capture_output=True, text=True, timeout=3600)
            if result.returncode:
                return self._send_json({"error": result.stderr[-3000:] or "Sonar scan failed"}, HTTPStatus.INTERNAL_SERVER_ERROR)
            return self._send_json(_load_json(run.run_dir / "artifacts" / "excluded_blocks_sonar" / "sonar.json"))
        except subprocess.TimeoutExpired:
            return self._send_json({"error": "Sonar scan timed out"}, HTTPStatus.INTERNAL_SERVER_ERROR)
        except OSError as exc:
            return self._send_json({"error": f"Could not run sonar: {exc}"}, HTTPStatus.INTERNAL_SERVER_ERROR)
        finally:
            run.sonar_lock.release()

    def _create_run(self, parsed: Any) -> None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if length <= 0 or length > MAX_UPLOAD_BYTES:
            return self._send_json({"error": "Upload must be a PDF smaller than 500 MB."}, HTTPStatus.BAD_REQUEST)
        data = self.rfile.read(length)
        if not data.startswith(b"%PDF-"):
            return self._send_json({"error": "The uploaded file is not a PDF."}, HTTPStatus.BAD_REQUEST)
        query = parse_qs(parsed.query)
        try:
            pages = max(1, min(int(query.get("pages", ["5"])[0]), 50))
            threshold = max(0.0, min(float(query.get("trainThreshold", ["0"])[0]), 1.0))
            sonar_drop = float(query.get("sonarDropPercent", ["15"])[0])
            if not math.isfinite(sonar_drop) or not 0 <= sonar_drop <= 100:
                raise ValueError("Invalid sonar threshold")
            retries = max(0, min(int(query.get("trainMaxRetries", ["3"])[0]), 10))
            eval_pages = max(1, min(int(query.get("evalPages", ["20"])[0]), 1_000))
            holdout_pages = max(0, min(int(query.get("holdoutPages", ["0"])[0]), 48))
        except ValueError:
            return self._send_json({"error": "Invalid run configuration."}, HTTPStatus.BAD_REQUEST)
        config = {
            "pages": pages,
            "eval": query.get("eval", ["true"])[0].lower() != "false",
            "trainThreshold": threshold,
            "sonarDropPercent": sonar_drop,
            "trainMaxRetries": retries,
            "evalPages": eval_pages,
            "holdoutPages": holdout_pages,
            "easyModel": query.get("easyModel", [""])[0].strip()[:200],
            "hardModel": query.get("hardModel", [""])[0].strip()[:200],
            "singleModel": query.get("singleModel", [""])[0].strip()[:200],
            "referenceModel": query.get(
                "referenceModel", ["openai/gpt-5.4"]
            )[0].strip()[:200],
            "noRouting": query.get("noRouting", ["false"])[0].lower() == "true",
        }
        run = MANAGER.create(self.headers.get("X-Filename", "input.pdf"), data, config)
        return self._send_json(run.snapshot(), HTTPStatus.CREATED)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local TWIX 2 dashboard.")
    parser.add_argument("--host", default="127.0.0.1", help="Default: loopback only")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"TWIX UI available at http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nTWIX UI stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
