from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from src.excluded_blocks_sonar import flag_pages, measure_coverage


class ExcludedBlocksSonarTests(unittest.TestCase):
    def test_measure_uses_original_ocr_ids_and_deduplicates_overlapping_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ocr = root / "ocr.csv"
            with ocr.open("w", newline="", encoding="utf-8") as target:
                writer = csv.DictWriter(target, fieldnames=["id", "page", "text"])
                writer.writeheader()
                writer.writerows([
                    {"id": 0, "page": 1, "text": "A"},
                    {"id": 1, "page": 1, "text": "B"},
                    {"id": 2, "page": 2, "text": "C"},
                ])
            blocks = root / "block_separation.json"
            blocks.write_text(json.dumps({"pages": [
                {"page": 1, "blocks": [
                    {"words": [{"id": 0}]}, {"words": [{"id": 0}]},
                ]},
                {"page": 2, "blocks": []},
            ]}), encoding="utf-8")

            pages = measure_coverage(ocr, blocks)

        self.assertEqual(pages[0]["insideBlockWords"], 1)
        self.assertEqual(pages[0]["outsideBlockWords"], 1)
        self.assertEqual(pages[0]["coverage"], 0.5)
        self.assertEqual(pages[1]["coverage"], 0)

    def test_flags_relative_drop_from_mean_of_scanned_pages(self) -> None:
        pages = [
            {"page": 4, "coverage": 1.0},
            {"page": 5, "coverage": 0.8},
            {"page": 6, "coverage": 0.2},
            {"page": 7, "coverage": None},
        ]
        average, flagged = flag_pages(pages, 15)
        self.assertAlmostEqual(average, 2 / 3)
        self.assertEqual(flagged, [6])
        self.assertEqual(flag_pages(pages, 80)[1], [])
        with self.assertRaises(ValueError):
            flag_pages(pages, float("nan"))
        self.assertEqual(flag_pages([
            {"page": 8, "coverage": 0.0, "totalWords": 40},
            {"page": 9, "coverage": None, "totalWords": 0},
        ], 15)[1], [8])

    def test_rejects_block_word_from_a_different_page(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "ocr.csv").write_text("id,page,text\n0,1,A\n", encoding="utf-8")
            (root / "blocks.json").write_text(json.dumps({"pages": [
                {"page": 2, "blocks": [{"words": [{"id": 0}]}]},
            ]}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "foreign word"):
                measure_coverage(root / "ocr.csv", root / "blocks.json")
