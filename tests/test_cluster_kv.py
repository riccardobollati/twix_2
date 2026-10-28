"""Offline checks for independent key/value page clustering."""
from __future__ import annotations

import unittest


from src.clustering import cluster_kv


def label(key: str, value: str, x: float, y: float, value_x: float | None = None) -> dict:
    value_x = x + 80 if value_x is None else value_x
    return {"key": key, "value": value, "x0": x, "x1": x + 55,
            "top": y, "bottom": y + 10,
            "value_bbox": [value_x, y, value_x + 30, y + 10]}


def group(page: int, *labels: dict) -> dict:
    return {"page": page, "labels": list(labels), "bbox": [min(x["x0"] for x in labels),
            min(x["top"] for x in labels), max(x["value_bbox"][2] for x in labels),
            max(x["bottom"] for x in labels)]}


class ClusterKvTests(unittest.TestCase):
    def test_stable_keys_need_distinct_pages_and_varying_values(self):
        occurrences = [group(page, label("Name", f"Person {page}", 20, 100),
                             label("Date", "2026", 20, 120)) for page in range(1, 11)]
        family = {"kind": "kv", "id": "kv_001", "occurrences": occurrences}
        self.assertEqual(cluster_kv.stable_keys(family, 10), {"name", "date"})
        family["occurrences"] = [group(1, label("Name", f"Person {n}", 20, 100),
                                        label("Date", "2026", 20, 120)) for n in range(10)]
        self.assertFalse(cluster_kv.stable_keys(family, 10))

    def test_table_rows_and_page_frame_are_rejected(self):
        family = {"kind": "kv", "id": "kv_001", "occurrences": [
            group(page, *(label("Total", str(n), 20, 100 + 12 * n) for n in range(9)))
            for page in range(1, 11)]}
        self.assertFalse(cluster_kv.stable_keys(family, 10))
        top = group(1, label("Date", "2026", 20, 2), label("Page", "1", 200, 2))
        self.assertIsNone(cluster_kv.group_feature("kv_001", top, {"date", "page"}, [], 600))

    def test_value_text_does_not_affect_similarity_and_optional_group_is_allowed(self):
        a = cluster_kv.group_feature("kv_001", group(1, label("Name", "Alice", 20, 100),
                                                    label("Date", "Monday", 20, 120)),
                                     {"name", "date"}, [], 600)
        b = cluster_kv.group_feature("kv_001", group(2, label("Name", "Bob", 120, 250),
                                                    label("Date", "Tuesday", 120, 270)),
                                     {"name", "date"}, [], 600)
        self.assertIsNotNone(a)
        self.assertIsNotNone(b)
        self.assertAlmostEqual(cluster_kv.group_similarity(a, b), 1.0)
        extra = cluster_kv.group_feature("kv_002", group(2, label("City", "Paris", 120, 300)),
                                         {"city"}, [], 600)
        self.assertGreater(cluster_kv.page_similarity([a], [b, extra]), 0.77)


if __name__ == "__main__":
    unittest.main()
