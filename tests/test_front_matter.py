from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import pdfplumber
from PIL import Image

from src.agents.detect_beginning import _parse_response
from src.agents.environment_creation import environment_creation_node


class FrontMatterTests(unittest.TestCase):
    def test_detection_response_is_range_checked(self) -> None:
        parsed = _parse_response(
            '{"start_page": 3, "confidence": "high", "reason": "records begin"}',
            1,
            6,
        )
        self.assertEqual(parsed["start_page"], 3)

        with self.assertRaisesRegex(ValueError, "between 1 and 6"):
            _parse_response(
                '{"start_page": 7, "confidence": "high", "reason": "bad"}',
                1,
                6,
            )

    def test_environment_skips_front_matter_and_remaps_pages(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_dir = root / "source"
            output_dir = root / "artifacts"
            source_dir.mkdir()
            pdf_path = source_dir / "document.pdf"
            pages = [Image.new("RGB", (80, 80), color) for color in ("white", "red", "blue")]
            pages[0].save(pdf_path, "PDF", save_all=True, append_images=pages[1:])

            ocr_path = source_dir / "ocr.csv"
            with ocr_path.open("w", newline="", encoding="utf-8") as target:
                writer = csv.DictWriter(
                    target,
                    fieldnames=["id", "page", "x0", "top", "x1", "bottom", "text"],
                )
                writer.writeheader()
                for page in range(1, 4):
                    writer.writerow({
                        "id": page,
                        "page": page,
                        "x0": 0,
                        "top": 0,
                        "x1": 10,
                        "bottom": 10,
                        "text": f"page-{page}",
                    })

            result = environment_creation_node({
                "full_pdf_path": str(pdf_path),
                "full_ocr_csv_path": str(ocr_path),
                "output_dir": str(output_dir),
                "doc_name": "document",
                "train_pages": 2,
                "eval_mode": False,
                "detected_start_page": 2,
            })

            self.assertEqual(Path(result["agent_space"]), output_dir / "agent_space")
            with pdfplumber.open(result["pdf_path"]) as cropped:
                self.assertEqual(len(cropped.pages), 2)
            with Path(result["ocr_csv_path"]).open(encoding="utf-8") as source:
                rows = list(csv.DictReader(source))
            self.assertEqual([row["page"] for row in rows], ["1", "2"])
            self.assertEqual([row["text"] for row in rows], ["page-2", "page-3"])


if __name__ == "__main__":
    unittest.main()
