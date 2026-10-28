"""Focused offline checks for the header/structure clustering rules."""
from __future__ import annotations

import unittest

import numpy as np


from src.clustering import cluster_pages_v2 as v2


class ClusterPagesV2Tests(unittest.TestCase):
    def test_consensus_header_requires_a_wide_repeated_row(self):
        class Page:
            width, height = 200, 100

            def __init__(self, labels):
                self.labels = labels

            def extract_words(self):
                return [dict(text=text, x0=10 + 28 * i, x1=25 + 28 * i,
                             top=10, bottom=14) for i, text in enumerate(self.labels)]

        class Pdf:
            def __init__(self, labels):
                self.pages = [Page(labels) for _ in range(3)]

        good = v2.consensus_headers(Pdf(["Chemical", "Form", "Material", "Pressure", "Size", "Rating"]), [1, 2, 3])
        self.assertEqual(set(good), {1, 2, 3})
        self.assertEqual(len(good[1][0]["labels"]), 6)
        self.assertEqual(v2.consensus_headers(Pdf(["Material", "Quantity", "Price"]), [1, 2, 3]), {})

    def test_recover_known_header_requires_text_at_expected_position(self):
        class Page:
            width, height, page_number = 200, 100, 3
            bbox = (0, 0, 200, 100)

            def __init__(self, text):
                self.text = text

            def crop(self, bbox):
                self.bbox = bbox
                return self

            def extract_text(self):
                return self.text

        prototype = {"bbox": [10, 20, 180, 30],
                     "labels": [{"text": text} for text in ("Date", "Invoice Number", "Amount")]}
        families = [{"kind": "table", "id": "table_001", "representative": prototype}]
        recovered = v2.recover_headers(Page("Date Invoice Number Amount"), [], families)
        self.assertEqual(len(recovered), 1)
        self.assertTrue(recovered[0]["recovered"])
        self.assertEqual(v2.recover_headers(Page("Date and unrelated prose"), [], families), [])
        shifted = Page("Date Invoice Number Amount")
        shifted.bbox = (-10, -10, 175, 90)
        self.assertEqual(len(v2.recover_headers(shifted, [], families)), 1)
        self.assertLessEqual(shifted.bbox[2], 175)

    def feature(self, words=(), red=True, geometry=None, table_columns=None):
        vec = np.zeros(80)
        vec[0] = 1
        return {"header_words": {word: 1 for word in words}, "headers": [1] if words else [],
                "header_geometry": np.ones(40) if geometry is None else geometry,
                "structure": vec, "has_red": red, "table_columns": table_columns or []}

    def test_one_changed_table_cannot_hide_behind_an_unchanged_table(self):
        old_left = {"x_span": [0.05, 0.45], "positions": [0.10, 0.20, 0.30, 0.40, 0.50]}
        new_left = {"x_span": [0.05, 0.45], "positions": [0.10, 0.25, 0.35, 0.45, 0.55]}
        right = {"x_span": [0.55, 0.95], "positions": [0.10, 0.20, 0.30, 0.40, 0.50]}
        first = self.feature(("size", "width", "depth"), table_columns=[old_left, right])
        changed = self.feature(("size", "width", "depth"), table_columns=[new_left, right])
        score, _, _ = v2.pair_similarity(first, changed, 0.60)
        self.assertLess(score, v2.METHODS["balanced_complete"][2])
        self.assertAlmostEqual(v2.scores([first, changed], 0.60)[0][0, 1], score)
        first["column_layout_reliable"] = changed["column_layout_reliable"] = False
        self.assertEqual(v2.pair_similarity(first, changed, 0.60)[0], 1.0)
        first["column_layout_reliable"] = changed["column_layout_reliable"] = True
        optional = self.feature(("size", "width", "depth"), table_columns=[
            {"x_span": [0.05, 0.45], "positions": [0.10, 0.20, 0.30, 0.40, 0.50, 0.60]}, right])
        self.assertGreaterEqual(v2.pair_similarity(first, optional, 0.60)[0],
                                v2.METHODS["balanced_complete"][2])

    def test_stacked_tables_match_their_own_families(self):
        first = {"family": "table_001", "x_span": [0.1, 0.9],
                 "positions": [0.10, 0.20, 0.30, 0.40, 0.50]}
        second = {"family": "table_002", "x_span": [0.1, 0.9],
                  "positions": [0.12, 0.24, 0.36, 0.48, 0.60]}
        a = self.feature(("size", "width"), table_columns=[first, second])
        b = self.feature(("size", "width"), table_columns=[first, second])
        self.assertFalse(v2.local_layout_mismatch(a, b))
        self.assertEqual(v2.pair_similarity(a, b, 0.60)[0], 1.0)

    def test_unstable_column_detection_disables_strong_local_gate(self):
        def observations(counts):
            return [self.feature(table_columns=[{"family": "table_001", "x_span": [0.1, 0.9],
                                                 "positions": [index / 100 for index in range(count)]}])
                    for count in counts]

        self.assertTrue(v2.column_counts_reliable(observations([14] * 8 + [15] * 5)))
        self.assertFalse(v2.column_counts_reliable(observations([4, 5, 6, 80, 82, 85])))

    def test_repeated_horizontal_table_rules_count_without_vertical_detection(self):
        lines = [dict(x0=x0, x1=x1, top=y, bottom=y)
                 for y in (10, 20, 30, 40, 50, 60)
                 for x0, x1 in ((10, 70), (80, 140))]
        self.assertTrue(v2.has_structure(lines, 150, 100))
        self.assertFalse(v2.has_structure(lines[:2], 150, 100))

    def test_one_column_header_does_not_crop_the_table_grid(self):
        lines = [dict(x0=50, x1=190, top=y, bottom=y) for y in (10, 30, 50)]
        headers = [{"bbox": [10, 5, 30, 15]}]
        self.assertEqual(len(v2.relevant_lines(lines, headers, 200, 100)), 3)

    def test_red_only_pages_can_reach_strict_threshold(self):
        score, header, structure = v2.pair_similarity(self.feature(), self.feature(), 0.8)
        self.assertEqual((score, header, structure), (1.0, 0.0, 1.0))

    def test_optional_header_column_can_stay_with_same_structure(self):
        a = self.feature(("date", "value", "amount"))
        b = self.feature(("date", "value", "amount", "note"))
        score, _, _ = v2.pair_similarity(a, b, 0.8)
        self.assertGreaterEqual(score, v2.METHODS["header_strict"][2])
        different = self.feature(("code", "quantity", "unit"))
        score, _, _ = v2.pair_similarity(a, different, 0.8)
        self.assertLess(score, v2.METHODS["header_strict"][2])

    def test_word_structure_only_used_on_headered_borderless_page(self):
        red = self.feature(("date", "amount"), red=True)
        inferred = self.feature(("date", "amount"), red=False)
        score, _, _ = v2.pair_similarity(red, inferred, 0.6)
        self.assertGreaterEqual(score, v2.METHODS["balanced_complete"][2])
        missing = self.feature(red=False)
        score, _, _ = v2.pair_similarity(red, missing, 0.6)
        self.assertEqual(score, 0)

    def test_empty_header_signatures_are_not_perfect_matches(self):
        a = self.feature(red=False)
        b = self.feature(red=False)
        self.assertEqual(v2.header_similarity(a, b), 0.0)
        self.assertEqual(v2.pair_similarity(a, b, 0.6)[0], 0.0)

    def test_cached_matrix_matches_pair_rules(self):
        features = [self.feature(("date", "amount")), self.feature(("date", "amount"), red=False),
                    self.feature(), self.feature(red=False)]
        for weight in (0.8, 0.6, 0.35):
            matrix, _, _ = v2.scores(features, weight)
            for i, first in enumerate(features):
                for j, second in enumerate(features):
                    if i != j:
                        self.assertAlmostEqual(matrix[i, j], v2.pair_similarity(first, second, weight)[0])


if __name__ == "__main__":
    unittest.main()
