from __future__ import annotations

import unittest

from src.pipeline import _group_records, _health_check


class PipelineTests(unittest.TestCase):
    def test_blocks_are_grouped_into_the_containing_record(self) -> None:
        records = {
            "records": [
                {"record_id": "a", "pages": [{"page": 1, "y_start": 0, "y_end": 50}]},
                {"record_id": "b", "pages": [{"page": 1, "y_start": 50, "y_end": 100}]},
            ]
        }
        blocks = {
            "pages": [{
                "page": 1,
                "blocks": [
                    {"uid": "u1", "words": [{"top": 10, "bottom": 20}]},
                    {"uid": "u2", "words": [{"top": 70, "bottom": 80}]},
                ],
            }]
        }
        extracted = [{"uid": "u1", "value": "one"}, {"uid": "u2", "value": "two"}]

        grouped = _group_records(records, blocks, extracted)

        self.assertEqual([record["record_id"] for record in grouped], ["a", "b"])
        self.assertEqual(grouped[0]["data"][0]["value"], "one")
        self.assertEqual(grouped[1]["data"][0]["value"], "two")

    def test_health_check_detects_cross_page_blocks(self) -> None:
        words = [{"id": 1, "page": 1}, {"id": 2, "page": 2}]
        records = {"records": [{"record_id": "r1"}]}
        blocks = {
            "pages": [{
                "page": 1,
                "blocks": [{
                    "uid": "u1",
                    "block_id": "0_0",
                    "words": [{"id": 1}, {"id": 2}],
                }],
            }]
        }
        extracted = [{"uid": "u1"}]
        grouped = [{"record_id": "r1", "data": extracted}]

        health = _health_check(words, records, blocks, extracted, grouped)

        self.assertTrue(health["degenerate"])
        self.assertEqual(len(health["cross_page_blocks"]), 1)


if __name__ == "__main__":
    unittest.main()
