from __future__ import annotations

import unittest

from src.agents.template_inference import (
    _dedup_fields,
    _duplicate_fields,
    _flag_signature_duplicates,
    _validate_template,
)


class TemplateValidationTests(unittest.TestCase):
    def test_duplicate_fields_are_deduplicated_deterministically(self) -> None:
        template = [{"node_id": 0, "fields": ["Total", "Total", "Total"]}]

        self.assertTrue(_duplicate_fields(template))
        _dedup_fields(template)

        self.assertEqual(template[0]["fields"], ["Total", "Total (2)", "Total (3)"])
        self.assertEqual(_duplicate_fields(template), [])

    def test_structural_validation_rejects_empty_template(self) -> None:
        self.assertTrue(_validate_template([]))

    def test_identical_field_signatures_are_flagged(self) -> None:
        template = [
            {"node_id": 0, "fields": ["Name", "Date"]},
            {"node_id": 1, "fields": ["date", "name"]},
        ]

        warnings = _flag_signature_duplicates(template)

        self.assertEqual(template[1]["signature_duplicate_of"], 0)
        self.assertEqual(len(warnings), 1)

    def test_xor_nodes_reference_distinct_matching_physical_nodes(self) -> None:
        template = [
            {"type": "xor-table", "node_id": 0, "includes": [1, 2]},
            {"type": "table", "node_id": 1, "fields": ["Width"], "bid": [0], "complexity": {}},
            {"type": "table", "node_id": 2, "fields": ["Diameter"], "bid": [0], "complexity": {}},
        ]
        self.assertEqual(_validate_template(template), [])
        template[2]["type"] = "kv"
        self.assertTrue(_validate_template(template))
        template[2]["type"] = "table"
        template[0]["includes"] = [1, 1]
        self.assertTrue(_validate_template(template))


if __name__ == "__main__":
    unittest.main()
