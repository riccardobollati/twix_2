"""Offline checks for independent key/value page clustering."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "test-clustering"))
import build_kv_html  # noqa: E402
import cluster_kv  # noqa: E402


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

    def test_kv_view_has_own_cluster_assignments_and_navigation(self):
        report = {"page_count": 3, "source_total_pages": 3, "skipped_pages": [2],
                  "families": [{"id": "kv_001"}],
                  "pages": [{"page": n, "groups": [{"family": "kv_001", "keys": ["name"],
                            "pairs": [{"key": "Name", "value": "Alice"}]}] if n != 2 else []}
                            for n in range(1, 4)],
                  "overlays": ["kv_page_0001.png"],
                  "algorithms": {method: {"cluster_count": 1, "clusters": {"1": [1, 3]},
                                          "representatives": {"1": 1}, "mean_within_similarity": 0.9,
                                          "first_five_cluster_coverage": 1.0}
                                 for method in cluster_kv.METHODS}}
        html = build_kv_html.document_html("sample", report, Path("/tmp/sample.pdf"), Path("/tmp"))
        self.assertIn("Table clusters", html)
        self.assertIn("KV cluster", html)
        self.assertIn("kv_page_0001.png", html)
        self.assertIn("Pages without confirmed KV groups (1)", html)


if __name__ == "__main__":
    unittest.main()
