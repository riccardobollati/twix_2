from __future__ import annotations

import http.client
import json
import socket
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pypdfium2 as pdfium
from PIL import Image

from api import (
    ApiHandler,
    ExtractionJob,
    ExtractionManager,
    _parse_config,
    _stage_from_logs,
)


class _FakeManager:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.jobs: dict[str, ExtractionJob] = {}
        self.created: tuple[str, bytes, dict] | None = None

    def create(self, filename: str, data: bytes, config: dict) -> ExtractionJob:
        self.created = (filename, data, config)
        run_dir = self.root / "job-test"
        run_dir.mkdir(exist_ok=True)
        job = ExtractionJob("job-test", run_dir, filename, config)
        self.jobs[job.job_id] = job
        return job

    def get(self, job_id: str) -> ExtractionJob | None:
        return self.jobs.get(job_id)

    def list(self) -> list[dict]:
        return [job.snapshot() for job in self.jobs.values()]

    def result(self, job: ExtractionJob) -> dict:
        return ExtractionManager.result(self, job)


class _FakeServer:
    def __init__(self, manager: _FakeManager) -> None:
        self.manager = manager


class ApiEndpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.manager = _FakeManager(self.root)
        self.server = _FakeServer(self.manager)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _request(
        self,
        method: str,
        path: str,
        body: bytes | None = None,
        headers: dict[str, str] | None = None,
    ) -> tuple[http.client.HTTPResponse, dict]:
        body = body or b""
        request_headers = {
            "Host": "localhost",
            "Connection": "close",
            "Content-Length": str(len(body)),
            **(headers or {}),
        }
        raw_request = (
            f"{method} {path} HTTP/1.1\r\n"
            + "".join(f"{name}: {value}\r\n" for name, value in request_headers.items())
            + "\r\n"
        ).encode("ascii") + body
        client_socket, handler_socket = socket.socketpair()
        try:
            client_socket.sendall(raw_request)
            client_socket.shutdown(socket.SHUT_WR)
            ApiHandler(
                handler_socket,
                ("127.0.0.1", 12345),
                self.server,  # type: ignore[arg-type]
            )
            response = http.client.HTTPResponse(client_socket)
            response.begin()
            payload = json.loads(response.read().decode("utf-8"))
            return response, payload
        finally:
            handler_socket.close()
            client_socket.close()

    def test_pdf_upload_status_and_result_flow(self) -> None:
        response, created = self._request(
            "POST",
            (
                "/v1/extractions?pages=5&noRouting=true&"
                "singleModel=deepseek/deepseek-v4.1-flash"
            ),
            b"%PDF-1.4\n% test",
            {"Content-Type": "application/pdf", "X-Filename": "invoice.pdf"},
        )

        self.assertEqual(response.status, 202)
        self.assertEqual(created["id"], "job-test")
        self.assertEqual(
            created["links"]["status"], "/v1/extractions/job-test/status"
        )
        self.assertEqual(self.manager.created[0], "invoice.pdf")  # type: ignore[index]
        self.assertEqual(self.manager.created[2]["pages"], 5)  # type: ignore[index]
        self.assertTrue(self.manager.created[2]["noRouting"])  # type: ignore[index]

        response, status = self._request(
            "GET", "/v1/extractions/job-test/status"
        )
        self.assertEqual(response.status, 200)
        self.assertEqual(status["status"], "queued")

        response, pending = self._request(
            "GET", "/v1/extractions/job-test/result"
        )
        self.assertEqual(response.status, 202)
        self.assertEqual(response.getheader("Retry-After"), "2")
        self.assertEqual(pending["status"], "queued")

        job = self.manager.jobs["job-test"]
        result_dir = job.run_dir / "api_result"
        result_dir.mkdir()
        (job.run_dir / "template.json").write_text(
            json.dumps({"template": []}), encoding="utf-8"
        )
        (job.run_dir / "metadata.json").write_text(
            json.dumps({"source_start_page": 2}), encoding="utf-8"
        )
        (result_dir / "grouped.json").write_text(
            json.dumps([{"record": 1}]), encoding="utf-8"
        )
        (result_dir / "extracted.json").write_text(
            json.dumps([{"block": 1}]), encoding="utf-8"
        )
        (result_dir / "pipeline_health.json").write_text(
            json.dumps({"degenerate": False}), encoding="utf-8"
        )
        job.status = "completed"

        response, result = self._request(
            "GET", "/v1/extractions/job-test/result"
        )
        self.assertEqual(response.status, 200)
        self.assertEqual(result["data"], [{"record": 1}])
        self.assertEqual(result["blocks"], [{"block": 1}])
        self.assertEqual(result["health"], {"degenerate": False})
        self.assertEqual(result["sourceStartPage"], 2)

    def test_upload_rejects_non_pdf_and_invalid_configuration(self) -> None:
        response, payload = self._request(
            "POST", "/v1/extractions", b"not a pdf"
        )
        self.assertEqual(response.status, 415)
        self.assertIn("not a PDF", payload["error"])

        response, payload = self._request(
            "POST", "/v1/extractions?pages=0", b"%PDF-1.4\n"
        )
        self.assertEqual(response.status, 400)
        self.assertIn("pages", payload["error"])

    def test_unknown_job_returns_json_404(self) -> None:
        response, payload = self._request(
            "GET", "/v1/extractions/missing/status"
        )
        self.assertEqual(response.status, 404)
        self.assertEqual(payload["error"], "extraction job not found")


class ApiConfigurationTests(unittest.TestCase):
    def test_default_configuration_is_extraction_only(self) -> None:
        config = _parse_config({})
        self.assertEqual(config["pages"], 3)
        self.assertIsNone(config["maxTrainPages"])
        self.assertEqual(config["firstTrainPages"], 3)
        self.assertFalse(config["noRouting"])
        self.assertEqual(config["trainThreshold"], 0.0)

    def test_pipeline_command_uses_no_eval_and_model_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manager = ExtractionManager(
                root / "runs", workers=1, pipeline_python="python-test"
            )
            config = _parse_config({
                "pages": ["4"],
                "maxTrainPages": ["2"],
                "firstTrainPages": ["0"],
                "startPage": ["2"],
                "easyModel": ["deepseek/deepseek-v4.1-flash"],
                "hardModel": ["deepseek/deepseek-v4-pro"],
                "templateModel": ["z-ai/glm-5.3-flash"],
            })
            job = ExtractionJob("job-test", root / "job-test", "input.pdf", config)
            command = manager._build_command(job, root / "input.pdf")
            manager.shutdown()

        self.assertIn("--no-eval", command)
        self.assertNotIn("--eval", command)
        self.assertEqual(command[command.index("--pages") + 1], "4")
        self.assertEqual(command[command.index("--max-train-pages") + 1], "2")
        self.assertEqual(command[command.index("--first-train-pages") + 1], "0")
        self.assertEqual(command[command.index("--start-page") + 1], "2")
        self.assertEqual(
            command[command.index("--easy-model") + 1],
            "deepseek/deepseek-v4.1-flash",
        )
        self.assertEqual(command[command.index("--template-model") + 1],
                         "z-ai/glm-5.3-flash")

    def test_invalid_training_page_limits_are_rejected(self) -> None:
        for query in ({"maxTrainPages": ["0"]}, {"maxTrainPages": ["-2"]},
                      {"maxTrainPages": ["many"]}, {"firstTrainPages": ["-1"]}):
            with self.subTest(query=query), self.assertRaises(ValueError):
                _parse_config(query)
        self.assertIsNone(_parse_config({"maxTrainPages": [""]})["maxTrainPages"])

    def test_stage_uses_latest_pipeline_marker(self) -> None:
        self.assertEqual(_stage_from_logs(["cluster_sampling: scanning 20 source pages"]),
                         "clustering_pages")
        self.assertEqual(
            _stage_from_logs([
                "train_eval_node: accuracy=0.4",
                "train_repair (attempt 1)",
            ]),
            "repairing",
        )
        self.assertEqual(
            _stage_from_logs([
                "save_metadata_node: complete",
                "[API] Applying generated pipeline to the uploaded document.",
            ]),
            "extracting_document",
        )

    def test_final_input_excludes_detected_front_matter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "input.pdf"
            pages = [
                Image.new("RGB", (20, 20), color)
                for color in ("red", "green", "blue")
            ]
            pages[0].save(input_path, "PDF", save_all=True, append_images=pages[1:])
            for page in pages:
                page.close()
            (root / "metadata.json").write_text(
                json.dumps({"source_start_page": 2}), encoding="utf-8"
            )
            result_dir = root / "api_result"
            result_dir.mkdir()
            job = ExtractionJob("job-test", root, "input.pdf", {})

            cropped_path = ExtractionManager._structured_pdf(
                job, input_path, result_dir
            )
            cropped = pdfium.PdfDocument(str(cropped_path))

            self.assertEqual(len(cropped), 2)
            cropped.close()

    def test_successful_job_applies_bundle_to_complete_document(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "job-test"
            run_dir.mkdir()
            input_path = run_dir / "input.pdf"
            input_path.write_bytes(b"%PDF-1.4\n")
            manager = ExtractionManager(
                root / "runs", workers=1, pipeline_python="python-test"
            )
            job = ExtractionJob(
                "job-test", run_dir, "input.pdf", _parse_config({})
            )
            calls: list[list[str]] = []

            def fake_process(_job: ExtractionJob, command: list[str]) -> int:
                calls.append(command)
                if len(calls) == 1:
                    pipeline = run_dir / "pipeline"
                    pipeline.mkdir()
                    (pipeline / "pipeline.py").write_text("# generated")
                    (run_dir / "metadata.json").write_text(
                        json.dumps({"source_start_page": 1}), encoding="utf-8"
                    )
                else:
                    result_dir = run_dir / "api_result"
                    (result_dir / "grouped.json").write_text("[]")
                    (result_dir / "extracted.json").write_text("[]")
                return 0

            with patch.object(manager, "_run_process", side_effect=fake_process):
                manager._execute(job, input_path)
            manager.shutdown()

        self.assertEqual(job.status, "completed")
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[1][2], str(input_path))
        self.assertEqual(
            calls[1][calls[1].index("--out-dir") + 1],
            str(run_dir / "api_result"),
        )


if __name__ == "__main__":
    unittest.main()
