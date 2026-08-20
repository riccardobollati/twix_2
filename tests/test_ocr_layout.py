from __future__ import annotations

import unittest

from src.ground_truth_gen.helpers import format_words_layout


class OcrLayoutTests(unittest.TestCase):
    def test_layout_preserves_lines_x_spans_and_exact_text(self) -> None:
        rendered = format_words_layout([
            {"text": "Alice", "x0": 80, "x1": 110, "top": 10, "bottom": 20},
            {"text": "Name:", "x0": 10, "x1": 50, "top": 10.5, "bottom": 20.5},
            {"text": "$1,200", "x0": 80, "x1": 120, "top": 35, "bottom": 45},
        ])

        lines = rendered.splitlines()
        self.assertEqual(len(lines), 2)
        self.assertLess(lines[0].index('"Name:"'), lines[0].index('"Alice"'))
        self.assertIn("x=80.0-120.0 \"$1,200\"", lines[1])


if __name__ == "__main__":
    unittest.main()
