#!/usr/bin/env python3
"""Local browser dashboard for running and inspecting TWIX 2.

The server deliberately binds to localhost by default: a run may use the local
OpenRouter/OpenCode configuration and can expose PDFs, OCR, and debug logs.
No data is uploaded anywhere by this dashboard itself.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import re
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
RUNS_DIR = UI_DIR / "runs"
MAX_UPLOAD_BYTES = 500 * 1024 * 1024
PREVIEW_BYTES = 600_000
_ANSI_ESCAPE = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|[@-_][0-?]*[ -/]*[@-~])")
_SPINNER_FRAMES = frozenset("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏")


def _pipeline_python() -> str:
    """Use the project's virtual environment when it is available.

    The dashboard can be started with any local ``python3``.  The actual TWIX
    run, however, needs the repository environment where pdfplumber, LangGraph,
    and the model clients are installed.
    """
    venv_python = PROJECT_DIR / "env" / "bin" / "python"
    return str(venv_python) if venv_python.is_file() and os.access(venv_python, os.X_OK) else sys.executable

STAGES = [
    ("ocr", "OCR & layout"),
    ("beginning", "Find data start"),
    ("environment", "Create workspace"),
    ("template", "Infer template"),
    ("records", "Separate records"),
    ("blocks", "Separate blocks"),
    ("samples", "Sample blocks"),
    ("routing", "Route difficulty"),
    ("extractors", "Generate extractors"),
    ("package", "Package pipeline"),
    ("train_eval", "Train evaluation"),
    ("repair", "Repair, if needed"),
    ("group", "Group records"),
    ("merge", "Merge continuations"),
    ("ground_truth", "Ground truth"),
    ("evaluate", "Evaluate"),
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
    terminal's in-place update. Keep only the final spinner result (the ✓ line)
    and discard interim frames.
    """
    clean = _ANSI_ESCAPE.sub("", raw).replace("\b", "")
    if "\r" in clean:
        clean = clean.rsplit("\r", 1)[-1]
    clean = clean.strip()
    if not clean:
        return None
    if any(frame in clean for frame in _SPINNER_FRAMES) and "✓" not in clean:
        return None
    return clean


def _safe_artifact_path(run_dir: Path, requested: str) -> Path | None:
    try:
        target = (run_dir / requested).resolve()
        target.relative_to(run_dir.resolve())
    except (OSError, ValueError):
        return None
    return target


def _stage_from_logs(lines: list[str], eval_enabled: bool) -> str:
    text = "\n".join(lines).lower()
    markers = [
        ("evaluate", ("eval_node:", "eval.json →")),
        ("ground_truth", ("ground_truth_node:", "generating test ground truth")),
        ("merge", ("merge_records",)),
        ("group", ("grouping_node:",)),
        ("repair", ("train_repair", "starting repair attempt")),
        ("train_eval", ("train_eval_node:", "train evaluation")),
        ("package", ("collect_extraction_scripts",)),
        ("extractors", ("block_extraction_scripting",)),
        ("routing", ("block_difficulty_router",)),
        ("samples", ("block_samples",)),
        ("blocks", ("block_separation",)),
        ("records", ("record_separation",)),
        ("template", ("template_inference",)),
        ("environment", ("cropped source pages", "train: source pages", "build: pages")),
        ("beginning", ("detect_beginning", "structured data starts")),
        ("ocr", ("extracting full ocr", "ocr saved")),
    ]
    for stage, phrases in markers:
        if any(phrase in text for phrase in phrases):
            return stage
    return "ocr"


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
    logs: list[str] = field(default_factory=list)
    process: subprocess.Popen[str] | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)

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
                "config": self.config,
                "stage": _stage_from_logs(self.logs, self.eval_enabled),
                "artifactCount": len(_file_tree(self.run_dir)),
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
                )
                if run.status in {"queued", "running"}:
                    run.status = "interrupted"
                log_path = directory / "ui_run.log"
                if log_path.is_file():
                    run.logs = [
                        clean for raw in log_path.read_text(encoding="utf-8", errors="replace").splitlines()
                        if (clean := _normalise_log_line(raw))
                    ][-1_000:]
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
        command = [
            _pipeline_python(), str(PROJECT_DIR / "run.py"), str(input_path),
            "--out-dir", str(run.run_dir), "--pages", str(run.config["pages"]),
            "--save-log",
        ]
        if run.config["eval"]:
            command.append("--eval")
        if run.config["trainThreshold"] > 0:
            command.extend(["--train-threshold", str(run.config["trainThreshold"])])
            command.extend(["--train-max-retries", str(run.config["trainMaxRetries"])])

        def execute() -> None:
            with run.lock:
                run.status = "running"
                run.started_at = _utc_now()
                run.logs.append("[UI] Starting TWIX pipeline.")
            run.persist()
            log_path = run.run_dir / "ui_run.log"
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
                            run.logs.append(line)
                            run.logs = run.logs[-3_000:]
                        log_file.write(line + "\n")
                        log_file.flush()
                code = process.wait()
                with run.lock:
                    run.return_code = code
                    run.status = "completed" if code == 0 else "failed"
                    run.finished_at = _utc_now()
                    run.process = None
                    run.logs.append(f"[UI] Pipeline {'completed' if code == 0 else 'failed'} (exit {code}).")
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
        if path in {"/app.js", "/styles.css"}:
            return self._send_file(UI_DIR / path.lstrip("/"))
        if path == "/api/runs":
            return self._send_json({"runs": MANAGER.list()})

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
            return self._stream_events(run)
        if len(parts) == 5 and parts[4] == "comparison":
            return self._send_comparison(run)
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

    def _stream_events(self, run: Run) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        cursor = 0
        try:
            while True:
                with run.lock:
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
        for key, filename in (("grouped", "grouped.json"), ("groundTruth", "ground_truth.json"), ("evaluation", "eval.json")):
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
        self.send_error(HTTPStatus.NOT_FOUND)

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
            retries = max(0, min(int(query.get("trainMaxRetries", ["3"])[0]), 10))
        except ValueError:
            return self._send_json({"error": "Invalid run configuration."}, HTTPStatus.BAD_REQUEST)
        config = {
            "pages": pages,
            "eval": query.get("eval", ["true"])[0].lower() != "false",
            "trainThreshold": threshold,
            "trainMaxRetries": retries,
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
