from __future__ import annotations

import unittest

from src.pipeline import _health_check


class PipelineTests(unittest.TestCase):
    def test_health_check_detects_cross_page_blocks(self) -> None:
        words = [{"id": 1, "page": 1}, {"id": 2, "page": 2}]
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
        health = _health_check(words, blocks, extracted)

        self.assertTrue(health["degenerate"])
        self.assertEqual(len(health["cross_page_blocks"]), 1)
        self.assertNotIn("records_detected", health)


if __name__ == "__main__":
    unittest.main()
