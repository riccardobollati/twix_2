from __future__ import annotations

import unittest

from src.ground_truth_gen.helpers import parse_json_response


class GroundTruthParsingTests(unittest.TestCase):
    def test_duplicate_json_keys_are_preserved(self) -> None:
        parsed = parse_json_response('prefix {"value": "a", "value": "b"} suffix')

        self.assertEqual(parsed, {"value": "a", "value#2": "b"})


if __name__ == "__main__":
    unittest.main()
