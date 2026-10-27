#!/usr/bin/env python3
"""HTTP API for asynchronous TWIX PDF extraction jobs.

The service accepts a PDF as the raw request body, starts ``run.py`` in a
bounded worker pool, exposes pipeline status, and returns the final grouped and
flat extraction artifacts. It binds to loopback by default because jobs can
contain sensitive PDFs, OCR, model prompts, and generated code.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

import pypdfium2 as pdfium


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_RUNS_DIR = PROJECT_DIR / "api_runs"
MAX_UPLOAD_BYTES = 500 * 1024 * 1024
_ANSI_ESCAPE = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|[@-_][0-?]*[ -/]*[@-~])")
_TERMINAL_STATUSES = frozenset({"completed", "failed", "interrupted"})


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _pipeline_python() -> str:
    venv_python = PROJECT_DIR / "env" / "bin" / "python"
    if venv_python.is_file() and os.access(venv_python, os.X_OK):
        return str(venv_python)
    return sys.executable


def _safe_filename(name: str) -> str:
    name = Path(name or "input.pdf").name
    cleaned = re.sub(r"[^A-Za-z0-9._ -]", "_", name).strip(" .")
    if not cleaned.lower().endswith(".pdf"):
        cleaned = f"{cleaned or 'input'}.pdf"
    return cleaned


def _normalise_log_line(raw: str) -> str | None:
    clean = _ANSI_ESCAPE.sub("", raw).replace("\b", "")
    if "\r" in clean:
        clean = clean.rsplit("\r", 1)[-1]
    clean = clean.strip()
    return clean or None


def _stage_from_logs(lines: list[str]) -> str:
    text = "\n".join(lines[-300:]).lower()
    stages = [
        ("saving_metadata", ("save_metadata", "graph complete")),
        ("extracting_document", ("[api] applying generated pipeline",)),
        ("extracting", ("extraction_node:", "running the generated pipeline")),
        ("train_evaluation", ("train_eval_node:", "train ground truth")),
        ("repairing", ("train_repair", "starting repair attempt")),
        (
            "building_extractors",
            ("block_extraction_scripting", "collect_extraction_scripts"),
        ),
        ("routing_blocks", ("block_difficulty_router", "block_samples")),
        ("separating_blocks", ("block_separation", "block_artifacts_node:")),
        ("inferring_template", ("template_inference", "cropped source pages")),
        ("clustering_pages", ("cluster_sampling",)),
        ("detecting_beginning", ("structured data starts", "extracting full ocr")),
    ]
    ranked = [
        (max(text.rfind(marker) for marker in markers), stage)
        for stage, markers in stages
    ]
    position, stage = max(ranked)
    return stage if position >= 0 else "queued"


def _load_json(path: Path) -> Any | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _parse_bool(value: str, field_name: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{field_name} must be true or false")


def _query_value(query: dict[str, list[str]], name: str, default: str) -> str:
    values = query.get(name, [default])
    return values[0] if values else default


def _parse_config(query: dict[str, list[str]]) -> dict[str, Any]:
    try:
        pages = int(_query_value(query, "pages", "3"))
        max_train_raw = _query_value(query, "maxTrainPages", "").strip()
        max_train_pages = int(max_train_raw) if max_train_raw else None
        first_train_pages = int(_query_value(query, "firstTrainPages", "3"))
        start_page_raw = _query_value(query, "startPage", "").strip()
        start_page = int(start_page_raw) if start_page_raw else None
        threshold = float(_query_value(query, "trainThreshold", "0"))
        retries = int(_query_value(query, "trainMaxRetries", "3"))
        no_routing = _parse_bool(
            _query_value(query, "noRouting", "false"), "noRouting"
        )
        group_rows = _parse_bool(
            _query_value(query, "groupRows", "false"), "groupRows"
        )
    except ValueError as exc:
        raise ValueError(f"invalid extraction configuration: {exc}") from exc

    if not 1 <= pages <= 50:
        raise ValueError("pages must be between 1 and 50")
    if max_train_pages is not None and max_train_pages < 1:
        raise ValueError("maxTrainPages must be greater than zero")
    if first_train_pages < 0:
        raise ValueError("firstTrainPages must be zero or greater")
    if start_page is not None and start_page < 1:
        raise ValueError("startPage must be greater than zero")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("trainThreshold must be between 0 and 1")
    if not 0 <= retries <= 10:
        raise ValueError("trainMaxRetries must be between 0 and 10")

    config: dict[str, Any] = {
        "pages": pages,
        "maxTrainPages": max_train_pages,
        "firstTrainPages": first_train_pages,
        "startPage": start_page,
        "trainThreshold": threshold,
        "trainMaxRetries": retries,
        "noRouting": no_routing,
        "groupRows": group_rows,
    }
    for query_name, config_name in (
        ("easyModel", "easyModel"),
        ("hardModel", "hardModel"),
        ("singleModel", "singleModel"),
        ("referenceModel", "referenceModel"),
        ("templateModel", "templateModel"),
    ):
        value = _query_value(query, query_name, "").strip()
        if len(value) > 200:
            raise ValueError(f"{query_name} must be 200 characters or fewer")
        config[config_name] = value
    return config


@dataclass
class ExtractionJob:
    job_id: str
    run_dir: Path
    filename: str
    config: dict[str, Any]
    status: str = "queued"
    created_at: str = field(default_factory=_utc_now)
    started_at: str | None = None
    finished_at: str | None = None
    return_code: int | None = None
    command: str | None = None
    error: str | None = None
    logs: list[str] = field(default_factory=list)
    process: subprocess.Popen[str] | None = None
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def state_path(self) -> Path:
        return self.run_dir / "api_job.json"

    @property
    def log_path(self) -> Path:
        return self.run_dir / "api_job.log"

    def snapshot(self, *, include_logs: bool = False) -> dict[str, Any]:
        with self.lock:
            stage = _stage_from_logs(self.logs)
            if self.status == "running" and stage == "queued":
                stage = "starting"
            payload: dict[str, Any] = {
                "id": self.job_id,
                "status": self.status,
                "stage": (
                    "completed" if self.status == "completed"
                    else self.status if self.status in {"failed", "interrupted"}
                    else stage
                ),
                "filename": self.filename,
                "createdAt": self.created_at,
                "startedAt": self.started_at,
                "finishedAt": self.finished_at,
                "returnCode": self.return_code,
                "error": self.error,
                "config": self.config,
            }
            if include_logs:
                payload["logs"] = self.logs[-100:]
            return payload

    def persist(self) -> None:
        payload = self.snapshot()
        payload["command"] = self.command
        temporary = self.state_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary.replace(self.state_path)


class ExtractionManager:
    def __init__(
        self,
        runs_dir: Path,
        *,
        workers: int = 2,
        pipeline_python: str | None = None,
    ) -> None:
        self.runs_dir = runs_dir.resolve()
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.pipeline_python = pipeline_python or _pipeline_python()
        self.jobs: dict[str, ExtractionJob] = {}
        self.lock = threading.Lock()
        self.stopping = threading.Event()
        self.executor = ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="twix-api"
        )
        self._restore_jobs()

    def _restore_jobs(self) -> None:
        for directory in sorted(self.runs_dir.glob("job-*")):
            state_path = directory / "api_job.json"
            if not state_path.is_file():
                continue
            try:
                saved = json.loads(state_path.read_text(encoding="utf-8"))
                job = ExtractionJob(
                    job_id=saved["id"],
                    run_dir=directory,
                    filename=saved.get("filename", "input.pdf"),
                    config=saved.get("config", {}),
                    status=saved.get("status", "interrupted"),
                    created_at=saved.get("createdAt") or _utc_now(),
                    started_at=saved.get("startedAt"),
                    finished_at=saved.get("finishedAt"),
                    return_code=saved.get("returnCode"),
                    command=saved.get("command"),
                    error=saved.get("error"),
                )
                if job.log_path.is_file():
                    job.logs = [
                        line
                        for raw in job.log_path.read_text(
                            encoding="utf-8", errors="replace"
                        ).splitlines()
                        if (line := _normalise_log_line(raw))
                    ][-1_000:]
                if job.status not in _TERMINAL_STATUSES:
                    job.status = "interrupted"
                    job.finished_at = _utc_now()
                    job.error = "API process restarted before the job completed"
                    job.persist()
                self.jobs[job.job_id] = job
            except (OSError, KeyError, json.JSONDecodeError, TypeError):
                continue

    def create(self, filename: str, data: bytes, config: dict[str, Any]) -> ExtractionJob:
        job_id = f"job-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
        run_dir = self.runs_dir / job_id
        run_dir.mkdir(parents=True, exist_ok=False)
        filename = _safe_filename(filename)
        input_path = run_dir / filename
        input_path.write_bytes(data)
        job = ExtractionJob(job_id, run_dir, filename, config)
        job.command = shlex.join(self._build_command(job, input_path))
        with self.lock:
            self.jobs[job_id] = job
        job.persist()
        self.executor.submit(self._execute, job, input_path)
        return job

    def get(self, job_id: str) -> ExtractionJob | None:
        with self.lock:
            return self.jobs.get(job_id)

    def list(self) -> list[dict[str, Any]]:
        with self.lock:
            jobs = list(self.jobs.values())
        return [job.snapshot() for job in sorted(jobs, key=lambda item: item.job_id)]

    def _build_command(self, job: ExtractionJob, input_path: Path) -> list[str]:
        config = job.config
        command = [
            self.pipeline_python,
            str(PROJECT_DIR / "run.py"),
            str(input_path),
            "--out-dir",
            str(job.run_dir),
            "--pages",
            str(config["pages"]),
            "--no-eval",
            "--save-log",
        ]
        if config.get("startPage") is not None:
            command.extend(["--start-page", str(config["startPage"])])
        if config.get("maxTrainPages") is not None:
            command.extend(["--max-train-pages", str(config["maxTrainPages"])])
        command.extend(["--first-train-pages", str(config.get("firstTrainPages", 3))])
        if config.get("groupRows"):
            command.append("--group-rows")
        if config.get("trainThreshold", 0) > 0:
            command.extend([
                "--train-threshold",
                str(config["trainThreshold"]),
                "--train-max-retries",
                str(config["trainMaxRetries"]),
            ])
        reference_model = str(config.get("referenceModel") or "").strip()
        if reference_model:
            command.extend(["--reference-model", reference_model])
        template_model = str(config.get("templateModel") or "").strip()
        if template_model:
            command.extend(["--template-model", template_model])
        if config.get("noRouting"):
            command.append("--no-routing")
            single_model = str(config.get("singleModel") or "").strip()
            if single_model:
                command.extend(["--easy-model", single_model])
        else:
            easy_model = str(config.get("easyModel") or "").strip()
            hard_model = str(config.get("hardModel") or "").strip()
            if easy_model:
                command.extend(["--easy-model", easy_model])
            if hard_model:
                command.extend(["--hard-model", hard_model])
        return command

    def _execute(self, job: ExtractionJob, input_path: Path) -> None:
        if self.stopping.is_set():
            with job.lock:
                job.status = "interrupted"
                job.finished_at = _utc_now()
                job.error = "API service stopped before the job started"
            job.persist()
            return
        command = self._build_command(job, input_path)
        with job.lock:
            job.status = "running"
            job.started_at = _utc_now()
            job.logs.append("[API] Starting TWIX extraction pipeline.")
        job.persist()
        try:
            return_code = self._run_process(job, command)
            failure = (
                f"pipeline generation exited with status {return_code}"
                if return_code != 0 else None
            )
            if return_code == 0 and not self.stopping.is_set():
                result_dir = job.run_dir / "api_result"
                result_dir.mkdir(exist_ok=True)
                structured_pdf = self._structured_pdf(job, input_path, result_dir)
                final_command = [
                    self.pipeline_python,
                    str(job.run_dir / "pipeline" / "pipeline.py"),
                    str(structured_pdf),
                    "--out-dir",
                    str(result_dir),
                ]
                with job.lock:
                    job.logs.append(
                        "[API] Applying generated pipeline to the uploaded document."
                    )
                return_code = self._run_process(job, final_command)
                if return_code != 0:
                    failure = (
                        "full-document extraction exited with status "
                        f"{return_code}"
                    )

            grouped_path = job.run_dir / "api_result" / "grouped.json"
            extracted_path = job.run_dir / "api_result" / "extracted.json"
            missing_result = not grouped_path.is_file() or not extracted_path.is_file()
            with job.lock:
                job.return_code = return_code
                job.process = None
                job.finished_at = _utc_now()
                if self.stopping.is_set():
                    job.status = "interrupted"
                    job.error = "API service stopped before the job completed"
                    job.logs.append(f"[API] Extraction interrupted (exit {return_code}).")
                elif return_code == 0 and not missing_result:
                    job.status = "completed"
                    job.logs.append("[API] Extraction completed.")
                else:
                    job.status = "failed"
                    job.error = failure or "pipeline completed without extraction artifacts"
                    job.logs.append(f"[API] Extraction failed: {job.error}.")
        except Exception as exc:
            with job.lock:
                job.status = "failed"
                job.finished_at = _utc_now()
                job.process = None
                job.error = str(exc)
                job.logs.append(f"[API] Could not run pipeline: {exc}")
        job.persist()

    def _run_process(self, job: ExtractionJob, command: list[str]) -> int:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        with job.lock:
            job.process = process
        with job.log_path.open("a", encoding="utf-8") as log_file:
            for raw_line in process.stdout or []:
                line = _normalise_log_line(raw_line)
                if line is None:
                    continue
                with job.lock:
                    job.logs.append(line)
                    job.logs = job.logs[-1_000:]
                log_file.write(line + "\n")
                log_file.flush()
        return process.wait()

    @staticmethod
    def _structured_pdf(
        job: ExtractionJob, input_path: Path, result_dir: Path
    ) -> Path:
        metadata = _load_json(job.run_dir / "metadata.json") or {}
        try:
            source_start = max(1, int(metadata.get("source_start_page", 1)))
        except (TypeError, ValueError):
            source_start = 1
        if source_start == 1:
            return input_path

        source = pdfium.PdfDocument(str(input_path))
        cropped = pdfium.PdfDocument.new()
        try:
            cropped.import_pages(source, list(range(source_start - 1, len(source))))
            cropped_path = result_dir / "structured_input.pdf"
            cropped.save(str(cropped_path))
        finally:
            cropped.close()
            source.close()
        return cropped_path

    def result(self, job: ExtractionJob) -> dict[str, Any]:
        result_dir = job.run_dir / "api_result"
        grouped = _load_json(result_dir / "grouped.json")
        extracted = _load_json(result_dir / "extracted.json")
        if grouped is None or extracted is None:
            raise FileNotFoundError("extraction artifacts are unavailable")
        metadata = _load_json(job.run_dir / "metadata.json") or {}
        return {
            "id": job.job_id,
            "status": "completed",
            "data": grouped,
            "blocks": extracted,
            "template": _load_json(job.run_dir / "template.json"),
            "health": _load_json(result_dir / "pipeline_health.json"),
            "sourceStartPage": metadata.get("source_start_page", 1),
        }

    def shutdown(self) -> None:
        self.stopping.set()
        with self.lock:
            jobs = list(self.jobs.values())
        queued: list[ExtractionJob] = []
        for job in jobs:
            with job.lock:
                if job.status == "queued":
                    job.status = "interrupted"
                    job.finished_at = _utc_now()
                    job.error = "API service stopped before the job started"
                    queued.append(job)
                elif job.status == "running" and job.process is not None:
                    job.process.terminate()
        for job in queued:
            job.persist()
        self.executor.shutdown(wait=True, cancel_futures=True)


class TwixApiServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        server_address: tuple[str, int],
        manager: ExtractionManager,
    ) -> None:
        self.manager = manager
        super().__init__(server_address, ApiHandler)


class ApiHandler(BaseHTTPRequestHandler):
    server_version = "TwixAPI/1.0"

    @property
    def manager(self) -> ExtractionManager:
        return self.server.manager  # type: ignore[attr-defined, no-any-return]

    def log_message(self, format: str, *args: object) -> None:
        return

    def _send_json(
        self,
        payload: Any,
        status: int = HTTPStatus.OK,
        *,
        retry_after: int | None = None,
    ) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        if retry_after is not None:
            self.send_header("Retry-After", str(retry_after))
        self.end_headers()
        self.wfile.write(data)

    def _error(self, status: int, message: str) -> None:
        self._send_json({"error": message}, status)

    @staticmethod
    def _links(job_id: str) -> dict[str, str]:
        base = f"/v1/extractions/{job_id}"
        return {"status": f"{base}/status", "result": f"{base}/result"}

    def _job_from_path(self, parts: list[str]) -> ExtractionJob | None:
        if len(parts) < 4 or parts[1:3] != ["v1", "extractions"]:
            return None
        return self.manager.get(parts[3])

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = unquote(parsed.path).rstrip("/") or "/"
        if path == "/health":
            return self._send_json({"status": "ok"})
        if path == "/v1/extractions":
            return self._send_json({"jobs": self.manager.list()})

        parts = path.split("/")
        job = self._job_from_path(parts)
        if job is None:
            return self._error(HTTPStatus.NOT_FOUND, "extraction job not found")
        if len(parts) == 5 and parts[4] == "status":
            payload = job.snapshot()
            payload["links"] = self._links(job.job_id)
            return self._send_json(payload)
        if len(parts) == 5 and parts[4] == "result":
            snapshot = job.snapshot()
            if snapshot["status"] not in _TERMINAL_STATUSES:
                snapshot["links"] = self._links(job.job_id)
                return self._send_json(
                    snapshot, HTTPStatus.ACCEPTED, retry_after=2
                )
            if snapshot["status"] != "completed":
                snapshot["links"] = self._links(job.job_id)
                return self._send_json(snapshot, HTTPStatus.CONFLICT)
            try:
                return self._send_json(self.manager.result(job))
            except FileNotFoundError as exc:
                return self._error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))
        return self._error(HTTPStatus.NOT_FOUND, "endpoint not found")

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = unquote(parsed.path).rstrip("/")
        if path != "/v1/extractions":
            return self._error(HTTPStatus.NOT_FOUND, "endpoint not found")
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if length <= 0:
            return self._error(HTTPStatus.BAD_REQUEST, "request body must contain a PDF")
        if length > MAX_UPLOAD_BYTES:
            return self._error(
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                "PDF exceeds the 500 MB upload limit",
            )
        data = self.rfile.read(length)
        if b"%PDF-" not in data[:1024]:
            return self._error(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, "request body is not a PDF")
        try:
            config = _parse_config(parse_qs(parsed.query, keep_blank_values=True))
        except ValueError as exc:
            return self._error(HTTPStatus.BAD_REQUEST, str(exc))

        try:
            job = self.manager.create(
                self.headers.get("X-Filename", "input.pdf"), data, config
            )
        except OSError as exc:
            return self._error(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                f"could not create extraction job: {exc}",
            )
        payload = job.snapshot()
        payload["links"] = self._links(job.job_id)
        return self._send_json(payload, HTTPStatus.ACCEPTED, retry_after=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the TWIX extraction HTTP API.")
    parser.add_argument("--host", default="127.0.0.1", help="Default: loopback only")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        choices=range(1, 9),
        metavar="N",
        help="Maximum concurrent extraction jobs (default: 2)",
    )
    args = parser.parse_args()

    manager = ExtractionManager(args.runs_dir, workers=args.workers)
    server = TwixApiServer((args.host, args.port), manager)
    print(f"TWIX API available at http://{args.host}:{args.port}")
    print("POST /v1/extractions · GET /v1/extractions/<id>/status · GET /v1/extractions/<id>/result")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nTWIX API stopped.")
    finally:
        server.server_close()
        manager.shutdown()


if __name__ == "__main__":
    main()
