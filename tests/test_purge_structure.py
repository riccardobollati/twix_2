from __future__ import annotations

import csv
from io import BytesIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from test.purge_structure import (QUESTION_ID, ask_laya, build_state, expanded_bbox,
                                  frequency_filter, purge_report, run, score_occurrence,
                                  word_in_region)


def occurrence(page: int, fields: list[int]) -> dict:
    return dict(page=page, bbox=[40, 100, 190, 110], field_indices=fields,
                labels=[dict(text="Date", x0=40, top=100, x1=60, bottom=110),
                        dict(text="Amount", x0=150, top=100, x1=190, bottom=110)])


class PurgeStructureTests(unittest.TestCase):
    def test_frequency_gate_counts_occurrences_for_both_kinds(self):
        families = [dict(id=identifier, kind=kind, occurrences=[{}] * count, pages=pages)
                    for identifier, kind, count, pages in (
                        ("table_001", "table", 9, [1, 2]),
                        ("table_002", "table", 10, [1, 2]),
                        ("kv_001", "kv", 10, [1]),
                        ("kv_002", "kv", 8, [1, 2]))]
        scan = dict(families=families)
        eligible, decisions = frequency_filter(scan, 10)
        self.assertEqual([family["id"] for family in eligible["families"]], ["table_002", "kv_001"])
        self.assertEqual([decision["passed"] for decision in decisions], [False, True, True, False])
        self.assertEqual(len(scan["families"]), 4)

    def test_crop_expands_by_ten_percent_of_page_and_keeps_word_boxes(self):
        region = expanded_bbox([40, 100, 190, 110], width=600, height=800)
        self.assertEqual(region, [0, 20, 250, 190])
        self.assertTrue(word_in_region(dict(x0=248, top=150, x1=255, bottom=160), region))
        self.assertFalse(word_in_region(dict(x0=250, top=150, x1=255, bottom=160), region))

    def test_laya_state_keeps_target_and_all_context_coordinates(self):
        candidate = occurrence(2, [0, 1])
        words = [dict(text="Date", x0=40.125, top=100, x1=60, bottom=110),
                 dict(text="2026", x0=40, top=125, x1=64, bottom=135),
                 dict(text="Amount", x0=150, top=100, x1=190, bottom=110)]
        state = build_state(candidate, words, [0, 20, 250, 190], 600, 800)
        self.assertLess(state.index("TARGET LABELS"), state.index("CROP WORDS IN THIS REQUEST"))
        for box in ("Y100-110 40-60:Date",
                    "Y125-135 40-64:2026",
                    "150-190:Amount"):
            self.assertIn(box, state)

    def test_purge_removes_rejected_occurrences_and_their_optional_fields(self):
        family = dict(id="table_001", kind="table",
                      fields=[dict(text="Date"), dict(text="Amount"), dict(text="Extra")],
                      occurrences=[occurrence(1, [0, 1]), occurrence(2, [0, 1, 2]), occurrence(3, [0, 1])],
                      representative=occurrence(1, [0, 1]), pages=[1, 2, 3])
        scan = dict(rules={"min_pages": 2}, pages=[1, 2, 3], families=[family])
        scores = {"table_001_p0001_o0001": 0.9,
                  "table_001_p0002_o0002": 0.1,
                  "table_001_p0003_o0003": 0.8}
        purged, decisions = purge_report(scan, scores, 0.5)
        self.assertEqual([field["text"] for field in purged["families"][0]["fields"]], ["Date", "Amount"])
        self.assertEqual(purged["families"][0]["pages"], [1, 3])
        self.assertEqual(purged["families"][0]["heatmap_counts"], [[1, 0, 1], [1, 0, 1]])
        self.assertEqual([d["included_in_purged_group"] for d in decisions], [True, False, True])
        purged, decisions = purge_report(scan, {**scores, "table_001_p0003_o0003": 0.2}, 0.5)
        self.assertFalse(purged["families"])  # one accepted page is not a recurring group
        purged, decisions = purge_report(scan, {**scores, "table_001_p0003_o0003": 0.2}, 0.5,
                                         {"table_001_p0003_o0003"})
        self.assertEqual(purged["families"][0]["pages"], [1, 3])
        self.assertEqual(decisions[2]["status"], "inconclusive_truncated")

    def test_laya_noul_response_is_parsed(self):
        response = {"answers": {QUESTION_ID: {"type": "noul", "noul": 0.82}}}
        with patch("test.purge_structure.urllib.request.urlopen") as open_url:
            open_url.return_value.__enter__.return_value = BytesIO(json.dumps(response).encode())
            probability, result = ask_laya("http://127.0.0.1:8770/api/predict",
                                           {"state": "test", "questions": {}}, 5)
        self.assertEqual(probability, 0.82)
        self.assertEqual(result, response)

    def test_large_crop_is_split_until_every_word_is_seen_without_truncation(self):
        candidate = occurrence(1, [0, 1])
        target = [dict(text="Date", x0=40, top=100, x1=60, bottom=110),
                  dict(text="Amount", x0=150, top=100, x1=190, bottom=110)]
        context = [dict(text=f"w{i}", x0=40 + i % 10 * 20, top=150 + i // 10 * 12,
                        x1=50 + i % 10 * 20, bottom=160 + i // 10 * 12)
                   for i in range(100)]

        def respond(url, payload, timeout):
            count = payload["state"].count(":w")
            tokens = 1024 if count > 30 else 500
            return 0.7, {"answers": {QUESTION_ID: {"noul": 0.7}},
                         "usage": {"input_tokens": tokens}}

        with patch("test.purge_structure.ask_laya", side_effect=respond):
            probability, calls, requests, incomplete = score_occurrence(
                candidate, target + context, [0, 20, 600, 500], 600, 800,
                "http://127.0.0.1:8770/api/predict", "typed-decisions", 5)
        self.assertEqual(probability, 0.7)
        self.assertFalse(incomplete)
        self.assertGreater(len(calls), 2)
        self.assertEqual(len(calls), len(requests))
        covered = "\n".join(request["payload"]["state"] for request in requests if not request["saturated"])
        for i in range(100):
            self.assertRegex(covered, rf":w{i}(?:\s|$)")

    def test_run_reads_saved_ocr_and_writes_separate_purged_result(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pdf = root / "sample.pdf"
            pdf.touch()  # CSV supplies the OCR; images are disabled in this offline test.
            with (root / "ocr_extraction_full.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["id", "page", "x0", "top", "x1", "bottom", "text"])
                writer.writeheader()
                for page in (1, 2):
                    for word_id, text, x, y in ((0, "Date", 40, 100), (1, "Amount", 150, 100),
                                                (2, "2026", 40, 125), (3, "23.50", 150, 125)):
                        writer.writerow(dict(id=word_id, page=page, x0=x, top=y,
                                             x1=x + len(text) * 4, bottom=y + 10, text=text))
            family = dict(id="table_001", kind="table", fields=[dict(text="Date"), dict(text="Amount")],
                          occurrences=[occurrence(1, [0, 1]), occurrence(2, [0, 1])],
                          representative=occurrence(1, [0, 1]), pages=[1, 2])
            scan = dict(source_pdf=str(pdf), rules={"min_pages": 2}, pages=[1, 2],
                        page_summary=[dict(page=p, width=600, height=800) for p in (1, 2)],
                        families=[family])
            (root / "scan.json").write_text(json.dumps(scan))
            answer = {"answers": {QUESTION_ID: {"type": "noul", "noul": 0.8}},
                      "usage": {"input_tokens": 100}}
            with patch("test.purge_structure.ask_laya", return_value=(0.8, answer)) as ask, \
                 patch("test.purge_structure.plot_results", return_value=[]) as plot:
                result = run(root, "http://127.0.0.1:8770/api/predict", "typed-decisions",
                             threshold=0.5, margin=0.1, timeout=5, save_images=False,
                             minimum_occurrences=2)
            self.assertEqual(ask.call_count, 2)
            self.assertIn("150-170:23.50", ask.call_args.args[1]["state"])
            self.assertEqual(result["purge"]["retained_occurrences"], 2)
            self.assertTrue(result["purge"]["laya_enabled"])
            self.assertEqual(plot.call_args.args[1], root / "purged")
            self.assertTrue((root / "purged" / "purged_scan.json").is_file())
            self.assertEqual(len((root / "purged" / "purged_requests.jsonl").read_text().splitlines()), 2)
            self.assertEqual(result["purge"]["source_scan"], "../scan.json")
            self.assertFalse((root / "purged_scan.json").exists())

    def test_run_skips_small_groups_before_laya_and_keeps_eligible_kv(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pdf = root / "sample.pdf"
            pdf.touch()
            families = []
            for identifier, kind, count in (("table_001", "table", 9),
                                            ("table_002", "table", 10),
                                            ("kv_001", "kv", 10),
                                            ("kv_002", "kv", 9)):
                occurrences = [occurrence(index % 2 + 1, [0, 1]) for index in range(count)]
                families.append(dict(id=identifier, kind=kind,
                                     fields=[dict(text="Date"), dict(text="Amount")],
                                     occurrences=occurrences, representative=occurrences[0],
                                     pages=[1, 2]))
            scan = dict(source_pdf=str(pdf), rules={"min_pages": 2}, pages=[1, 2],
                        page_summary=[dict(page=p, width=600, height=800) for p in (1, 2)],
                        families=families)
            (root / "scan.json").write_text(json.dumps(scan))
            document_words = {page: [] for page in (1, 2)}
            ocr_sources = {page: str(pdf) for page in (1, 2)}
            with patch("test.purge_structure.load_document_words",
                       return_value=(document_words, ocr_sources)), \
                 patch("test.purge_structure.score_occurrence",
                       return_value=(0.8, [], [], False)) as score, \
                 patch("test.purge_structure.plot_results", return_value=[]):
                result = run(root, "http://127.0.0.1:8770/api/predict", "typed-decisions",
                             threshold=0.5, margin=0.1, timeout=5, save_images=False)
            self.assertEqual(score.call_count, 10)
            self.assertEqual([family["id"] for family in result["families"]],
                             ["table_002", "kv_001"])
            self.assertEqual(result["purge"]["examined_occurrences"], 10)
            self.assertEqual(result["purge"]["retained_occurrences"], 10)
            self.assertEqual(result["candidate_count"], 20)
            self.assertEqual([item["passed"] for item in result["purge"]["frequency_decisions"]],
                             [False, True, True, False])

    def test_no_laya_uses_scan_only_and_archives_previous_laya_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            families = []
            for identifier, kind, count in (("table_001", "table", 10),
                                            ("kv_001", "kv", 10),
                                            ("table_002", "table", 9)):
                instances = []
                for index in range(count):
                    instance = occurrence(index % 2 + 1, [0, 1])
                    for label in instance["labels"]:
                        label["words"] = [dict(text=label["text"], x0=label["x0"], x1=label["x1"],
                                               top=label["top"], bottom=label["bottom"])]
                    instances.append(instance)
                families.append(dict(id=identifier, kind=kind,
                                     fields=[dict(text="Date"), dict(text="Amount")],
                                     occurrences=instances, representative=instances[0], pages=[1, 2]))
            (root / "scan.json").write_text(json.dumps(dict(source_pdf=str(root / "missing.pdf"),
                                                             rules={"min_pages": 2}, pages=[1, 2],
                                                             families=families)))
            output = root / "purged"
            output.mkdir()
            (output / "purged_scan.json").write_text(json.dumps({"purge": {"laya_enabled": True}}))
            (output / "purged_requests.jsonl").write_text("old request\n")
            (output / "purged_table_999.png").write_bytes(b"old plot")
            snapshots = output / "purged_snapshots"
            snapshots.mkdir()
            (snapshots / "table_001_p0001_o0001.png").write_bytes(b"old snapshot")
            with patch("test.purge_structure.load_document_words") as load, \
                 patch("test.purge_structure.score_occurrence") as score:
                result = run(root, "http://127.0.0.1:8770/api/predict", "typed-decisions",
                             threshold=0.5, margin=0.1, timeout=5, use_laya=False)
            load.assert_not_called()
            score.assert_not_called()
            self.assertEqual([family["id"] for family in result["families"]], ["table_001", "kv_001"])
            self.assertFalse(result["purge"]["laya_enabled"])
            self.assertEqual(result["purge"]["request_count"], 0)
            self.assertEqual(result["purge"]["decisions"], [])
            self.assertEqual(set(result["artifacts"]), {"purged_table_001.png", "purged_kv_001.png"})
            self.assertTrue(all((output / name).is_file() for name in result["artifacts"]))
            self.assertTrue((output / "purged_scan.json").is_file())
            self.assertFalse((output / "purged_requests.jsonl").exists())
            self.assertFalse((output / "purged_table_999.png").exists())
            self.assertFalse(snapshots.exists())
            archive = output / "previous_laya"
            self.assertEqual((archive / "purged_requests.jsonl").read_text(), "old request\n")
            self.assertTrue((archive / "purged_scan.json").is_file())
            archived_report = json.loads((archive / "purged_scan.json").read_text())
            self.assertEqual(archived_report["purge"]["source_scan"], "../../scan.json")
            self.assertTrue((archive / "purged_table_999.png").is_file())
            self.assertTrue((archive / "purged_snapshots" / "table_001_p0001_o0001.png").is_file())


if __name__ == "__main__":
    unittest.main()
