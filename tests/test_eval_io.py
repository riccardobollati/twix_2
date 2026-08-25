from __future__ import annotations

import unittest

from src.eval.score import score
from src.ground_truth_gen.helpers import parse_json_response


class GroundTruthParsingTests(unittest.TestCase):
    def test_duplicate_json_keys_are_preserved(self) -> None:
        parsed = parse_json_response('prefix {"value": "a", "value": "b"} suffix')

        self.assertEqual(parsed, {"value": "a", "value#2": "b"})

    def test_model_generated_ground_truth_is_not_labeled_human(self) -> None:
        result = score({}, {}, reference="model_reference")

        self.assertEqual(result["reference"], "model_reference")

    def test_legacy_human_reference_is_renamed(self) -> None:
        result = score({}, {}, reference="human")

        self.assertEqual(result["reference"], "model_reference")


if __name__ == "__main__":
    unittest.main()
