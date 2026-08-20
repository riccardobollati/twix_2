from __future__ import annotations

import unittest
from unittest.mock import patch

from src.agents.gt_common import generate_ground_truth


class GroundTruthFallbackTests(unittest.TestCase):
    @patch("src.agents.gt_common._generate_ground_truth_once")
    def test_empty_large_window_retries_with_window_five(self, generate_once) -> None:
        fallback_records = [{"record_id": "0", "data": [{"block_id": "0"}]}]
        generate_once.side_effect = [
            ([], [], {"input_tokens": 100, "output_tokens": 20, "total_tokens": 0}),
            (
                fallback_records,
                fallback_records,
                {"input_tokens": 40, "output_tokens": 10, "total_tokens": 0},
            ),
        ]

        records, raw_records, usage = generate_ground_truth(
            "document.pdf", "ocr.csv", [], object(), 20, include_images=True
        )

        self.assertEqual(records, fallback_records)
        self.assertEqual(raw_records, fallback_records)
        self.assertEqual([call.args[-1] for call in generate_once.call_args_list], [20, 5])
        self.assertTrue(all(call.kwargs["include_images"] for call in generate_once.call_args_list))
        self.assertEqual(usage["input_tokens"], 140)
        self.assertEqual(usage["output_tokens"], 30)
        self.assertTrue(usage["empty_result_fallback"])
        self.assertEqual(usage["effective_window"], 5)

    @patch("src.agents.gt_common._generate_ground_truth_once")
    def test_empty_fallback_fails_explicitly(self, generate_once) -> None:
        generate_once.return_value = (
            [],
            [],
            {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        )

        with self.assertRaisesRegex(RuntimeError, "after retrying"):
            generate_ground_truth("document.pdf", "ocr.csv", [], object(), 20)

        self.assertEqual([call.args[-1] for call in generate_once.call_args_list], [20, 5])

    @patch("src.agents.gt_common._generate_ground_truth_once")
    def test_empty_five_page_window_fails_without_duplicate_retry(self, generate_once) -> None:
        generate_once.return_value = (
            [],
            [],
            {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        )

        with self.assertRaisesRegex(RuntimeError, "window=5"):
            generate_ground_truth("document.pdf", "ocr.csv", [], object(), 5)

        generate_once.assert_called_once()


if __name__ == "__main__":
    unittest.main()
