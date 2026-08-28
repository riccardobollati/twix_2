from __future__ import annotations

import unittest
from unittest.mock import patch

from src.agents.gt_common import _normalize_window_blocks, generate_ground_truth


class GroundTruthFallbackTests(unittest.TestCase):
    def test_flat_blocks_keep_repeated_node_ids_separate(self) -> None:
        raw = [
            {"page": 2, "block_id": "4", "type": "table", "data": [{"A": "x"}]},
            {"page": 2, "block_id": "4", "type": "table", "data": [{"A": "y"}]},
        ]

        blocks = _normalize_window_blocks(raw, [2, 3])

        self.assertEqual(blocks, raw)
        self.assertEqual(len(blocks), 2)

    def test_missing_page_is_only_inferred_for_single_page_window(self) -> None:
        block = {"block_id": "0", "type": "kv", "data": {"A": "x"}}

        self.assertEqual(_normalize_window_blocks([block], [7])[0]["page"], 7)
        self.assertEqual(_normalize_window_blocks([block], [7, 8]), [])

    @patch("src.agents.gt_common._generate_ground_truth_once")
    def test_empty_large_window_retries_with_window_five(self, generate_once) -> None:
        fallback_blocks = [{"page": 1, "block_id": "0"}]
        generate_once.side_effect = [
            ([], {"input_tokens": 100, "output_tokens": 20, "total_tokens": 0}),
            (
                fallback_blocks,
                {"input_tokens": 40, "output_tokens": 10, "total_tokens": 0},
            ),
        ]

        blocks, usage = generate_ground_truth(
            "document.pdf", "ocr.csv", [], object(), 20, include_images=True
        )

        self.assertEqual(blocks, fallback_blocks)
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
            {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        )

        with self.assertRaisesRegex(RuntimeError, "after retrying"):
            generate_ground_truth("document.pdf", "ocr.csv", [], object(), 20)

        self.assertEqual([call.args[-1] for call in generate_once.call_args_list], [20, 5])

    @patch("src.agents.gt_common._generate_ground_truth_once")
    def test_empty_five_page_window_fails_without_duplicate_retry(self, generate_once) -> None:
        generate_once.return_value = (
            [],
            {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        )

        with self.assertRaisesRegex(RuntimeError, "window=5"):
            generate_ground_truth("document.pdf", "ocr.csv", [], object(), 5)

        generate_once.assert_called_once()


if __name__ == "__main__":
    unittest.main()
