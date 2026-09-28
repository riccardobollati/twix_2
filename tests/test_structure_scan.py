from __future__ import annotations

import json
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import MagicMock, patch

from test.scan_structure import (Rules, cluster_candidates, compatibility, make_rows,
                                load_pdf, plot_results, scan_pages, spatial_coverage,
                                spatial_word_boxes)


def words(text: str, x: float, y: float, scale: float = 1) -> list[dict]:
    result = []
    for part in text.split():
        width = len(part) * 4 * scale
        result.append(dict(text=part, x0=x, x1=x + width, top=y, bottom=y + 8 * scale))
        x += width + 2 * scale
    return result


def page(number: int, content: list[dict], width: float = 600) -> dict:
    return dict(page=number, width=width, height=800, words=content)


def table(number: int, y: float = 100, shift: float = 0, scale: float = 1) -> list[dict]:
    content = []
    for text, x in (("Description", 40), ("Quantity", 160), ("Price", 240), ("Total", 320)):
        content += words(text, x * scale + shift, y, scale)
    for row in range(2):
        for text, x in ((f"Item{number}{row}", 40), (str(number + row), 160), ("1.50", 240), (str(number * 3), 320)):
            content += words(text, x * scale + shift, y + (row + 1) * 14 * scale, scale)
    return content


class StructureScanTests(unittest.TestCase):
    def test_heatmaps_are_png_only_for_scan_and_purge(self):
        family = dict(id="table_001", kind="table", fields=[dict(text="Header")], pages=[1],
                      occurrences=[dict(page=1, labels=[dict(words=words("Header", 40, 100))])])
        with TemporaryDirectory() as directory:
            output = Path(directory)
            report = dict(families=[family])
            self.assertEqual(plot_results(report, output, "sample"), ["table_001.png"])
            self.assertEqual(plot_results(report, output, "sample", prefix="purged_"),
                             ["purged_table_001.png"])
            self.assertEqual({path.name for path in output.iterdir()},
                             {"table_001.png", "purged_table_001.png"})

    def test_paragraph_spacing_does_not_make_a_header(self):
        content = words("This ordinary repeated paragraph contains words separated by normal spaces.", 40, 100)
        result = scan_pages([page(1, content), page(2, content)])
        self.assertFalse(result["families"])
        self.assertEqual(len(make_rows(content, Rules())[0]["chunks"]), 1)

    def test_header_requires_aligned_body(self):
        content = words("Annual report", 40, 100) + words("Company name", 320, 100)
        result = scan_pages([page(1, content), page(2, content)])
        self.assertFalse(result["families"])

    def test_repetition_is_counted_by_distinct_pages(self):
        result = scan_pages([page(1, table(1) + table(1, 250))])
        self.assertFalse(result["families"])

    def test_tables_count_multiple_occurrences_and_missing_pages(self):
        result = scan_pages([page(1, table(1) + table(1, 250)), page(2, []), page(3, table(3))])
        tables = [f for f in result["families"] if f["kind"] == "table"]
        self.assertEqual(len(tables), 1)
        self.assertEqual(tables[0]["pages"], [1, 3])
        self.assertEqual(tables[0]["heatmap_counts"], [[2, 0, 1]] * 4)
        self.assertEqual(result["empty_text_pages"], [2])

    def test_translation_and_scale_preserve_family(self):
        result = scan_pages([page(1, table(1)), page(2, table(2, 300, 30, 1.2))])
        tables = [f for f in result["families"] if f["kind"] == "table"]
        self.assertEqual(len(tables), 1)
        self.assertEqual(tables[0]["pages"], [1, 2])

    def test_repeated_categorical_data_is_not_part_of_header(self):
        pages = []
        for number in (1, 2, 3):
            content = []
            for text, x in (("Date", 40), ("Officer", 160), ("Result", 300)):
                content += words(text, x, 100)
            for text, x in ((f"{number}/01/2026", 40), ("MULTIPLE OFFICERS", 160), ("NOT SUSTAINED", 300)):
                content += words(text, x, 114)
            pages.append(page(number, content))
        tables = [f for f in scan_pages(pages)["families"] if f["kind"] == "table"]
        self.assertEqual(len(tables), 1)
        self.assertEqual([f["text"] for f in tables[0]["fields"]], ["Date", "Officer", "Result"])

    def test_horizontal_kv_keeps_blank_keys_and_checks_values(self):
        pages = []
        for number, name in ((1, "Alice"), (2, "Bob")):
            content = words("Name:", 40, 100) + words(name, 70, 100)
            content += words("DOB:", 160, 100) + words("City:", 240, 100) + words("Zurich", 270, 100)
            pages.append(page(number, content))
        kvs = [f for f in scan_pages(pages)["families"] if f["kind"] == "kv"]
        self.assertEqual(len(kvs), 1)
        self.assertEqual([f["text"] for f in kvs[0]["fields"]], ["Name", "DOB", "City"])
        self.assertEqual(kvs[0]["varying_value_fields"], [0])

    def test_vertical_kv_values_can_be_constant_in_some_fields(self):
        pages = []
        for number, value in ((1, "Small"), (2, "Large")):
            content = words("Specification", 40, 100) + words("ISO123", 160, 100)
            content += words("Size range", 40, 130) + words(value, 160, 130)
            pages.append(page(number, content))
        result = scan_pages(pages)
        self.assertEqual(len(result["families"]), 1)
        self.assertEqual(result["families"][0]["kind"], "kv")
        self.assertEqual(result["families"][0]["varying_value_fields"], [1])

    def test_kv_rows_with_missing_value_positions_do_not_crash(self):
        def blank(y):
            return words("Name:", 40, y) + words("4:", 160, y)

        def filled(y):
            return words("City", 40, y) + words("Zurich", 160, y)

        for blank_first in (True, False):
            with self.subTest(blank_first=blank_first):
                content = blank(100) + filled(115) if blank_first else filled(100) + blank(115)
                pages = [page(1, content), page(2, content)]
                report = scan_pages(pages)
                self.assertEqual(report["pages"], [1, 2])

    def test_pdf_scan_caps_selected_pages_at_500(self):
        mock_page = MagicMock(width=600, height=800)
        mock_page.dedupe_chars.return_value.extract_words.return_value = []
        with patch("pdfplumber.open") as open_pdf, redirect_stdout(StringIO()) as output:
            open_pdf.return_value.__enter__.return_value.pages = [mock_page] * 700
            pages, total = load_pdf(Path("unused.pdf"), start=101)
        self.assertEqual(total, 700)
        self.assertEqual(len(pages), 500)
        self.assertEqual((pages[0]["page"], pages[-1]["page"]), (101, 600))
        self.assertIn("Limiting scan to 500 pages", output.getvalue())

    def test_constant_key_value_groups_stay_unconfirmed(self):
        content = words("Name:", 40, 100) + words("Alice", 70, 100)
        content += words("City:", 240, 100) + words("Zurich", 270, 100)
        result = scan_pages([page(1, content), page(2, content)])
        self.assertFalse(result["families"])
        self.assertTrue(any("no observed cross-page value variation" in f["unconfirmed_reasons"] for f in result["unconfirmed"]))

    def test_wrapped_headers_remain_one_band(self):
        pages = []
        for number in (1, 2):
            content = []
            for text, x in (("Mass", 40), ("Section", 160), ("Surface", 280)):
                content += words(text, x, 100)
            for text, x in (("per metre", 40), ("depth", 160), ("area", 280)):
                content += words(text, x, 110)
            for text, x in ((str(number), 40), ("25.0", 160), ("16.5", 280)):
                content += words(text, x, 135)
            pages.append(page(number, content))
        tables = [f for f in scan_pages(pages)["families"] if f["kind"] == "table"]
        self.assertEqual(len(tables), 1)
        self.assertEqual(len(tables[0]["fields"]), 6)

    def test_single_column_paragraph_is_not_a_header_line(self):
        pages = []
        for number in (1, 2):
            content = []
            for text, x in (("Date claim", 100), ("Docket", 230), ("Other", 300)):
                content += words(text, x, 100)
            for text, x in (("Claim no", 40), ("Was filed", 100),
                            ("Filer", 180), ("No.", 230), ("Comments", 300)):
                content += words(text, x, 112)
            for line in range(5):
                content += words("This repeated paragraph is part of the comments cell", 300, 130 + 12 * line)
            for text, x in ((str(number), 40), ("2/23/2016", 100),
                            ("Smith", 180), ("3963", 230), ("decision", 300)):
                content += words(text, x, 202)
            pages.append(page(number, content))
        tables = [f for f in scan_pages(pages)["families"] if f["kind"] == "table"]
        self.assertEqual(len(tables), 1)
        self.assertEqual({field["line"] for field in tables[0]["fields"]}, {1, 2})
        self.assertEqual(tables[0]["fields"][-1]["text"], "Comments")
        self.assertEqual(len(tables[0]["occurrences"][0]["body_evidence"]), 1)

    def test_short_single_column_fragment_can_wrap_header(self):
        pages = []
        for number in (1, 2):
            content = words("Section", 40, 100) + words("Dimensions", 160, 100)
            content += words("designation", 40, 112)
            content += words("Mass", 40, 124) + words("Depth", 160, 124) + words("Width", 280, 124)
            content += words(f"UB{number}", 40, 138) + words("100", 160, 138) + words("200", 280, 138)
            pages.append(page(number, content))
        tables = [f for f in scan_pages(pages)["families"] if f["kind"] == "table"]
        self.assertEqual(len(tables), 1)
        self.assertIn("designation", [field["text"] for field in tables[0]["fields"]])

    def test_side_by_side_tables_are_separate(self):
        pages = []
        for number in (1, 2):
            left = table(number)
            right = []
            for text, x in (("Surface", 650), ("Area", 770), ("Mass", 850), ("Width", 930)):
                right += words(text, x, 100)
                right += words(str(number * x), x, 114)
            pages.append(page(number, left + right, width=1100))
        tables = [f for f in scan_pages(pages)["families"] if f["kind"] == "table"]
        self.assertEqual(len(tables), 2)

    def test_case_and_unicode_whitespace_normalize(self):
        a, b = table(1), table(2)
        for word in b:
            if word["top"] == 100:
                word["text"] = word["text"].upper()
        self.assertEqual(len(scan_pages([page(1, a), page(2, b)])["families"]), 1)

    def test_results_are_repeatable_and_do_not_mutate_input(self):
        pages = [page(1, table(1)), page(2, table(2))]
        before = json.dumps(pages)
        self.assertEqual(scan_pages(pages), scan_pages(pages))
        self.assertEqual(json.dumps(pages), before)

    def test_optional_column_has_its_own_presence_counts(self):
        pages = [page(1, table(1))]
        for number in (2, 3):
            content = table(number) + words("Discount", 400, 100)
            content += words(str(number), 400, 114) + words(str(number + 1), 400, 128)
            pages.append(page(number, content))
        tables = [f for f in scan_pages(pages)["families"] if f["kind"] == "table"]
        self.assertEqual(len(tables), 1)
        self.assertEqual(tables[0]["fields"][-1]["text"], "Discount")
        self.assertEqual(tables[0]["heatmap_counts"][-1], [0, 1, 1])

    def test_geometry_must_agree_even_when_text_matches(self):
        def candidate(xs):
            labels = [{**words(text, x, 100)[0], "key": text, "line": 1}
                      for text, x in zip(("alpha", "beta", "gamma", "delta"), xs)]
            return dict(kind="table", labels=labels)
        score, _ = compatibility(candidate([40, 140, 240, 340]), candidate([40, 60, 80, 340]), Rules())
        self.assertEqual(score, 0)

    def test_clusters_cannot_expand_through_a_chain_of_variants(self):
        candidates = []
        for number, fields in enumerate((4, 5, 6), 1):
            for physical_page in (number * 2 - 1, number * 2):
                labels = [{**words(f"label{i}", 40 + 80 * i, 100)[0], "key": f"label{i}", "line": 1}
                          for i in range(fields)]
                candidates.append(dict(kind="table", labels=labels, page=physical_page, bbox=[40, 100, 550, 108]))
        families, _ = cluster_candidates(candidates, Rules())
        self.assertEqual([f["pages"] for f in families], [[1, 2, 3, 4], [5, 6]])

    def test_unit_symbols_stay_in_header_and_do_not_count_as_body(self):
        pages = []
        for number in (1, 2):
            content = []
            for text, x in (("Area", 40), ("Inertia", 160), ("Volume", 280)):
                content += words(text, x, 100)
            for text, x in (("m2", 40), ("cm4", 160), ("cm3", 280)):
                content += words(text, x, 115)
            pages.append(page(number, content))
        self.assertFalse(scan_pages(pages)["families"])
        for p in pages:
            for x in (40, 160, 280):
                p["words"] += words(str(p["page"] * x), x, 130)
        families = scan_pages(pages)["families"]
        self.assertEqual(len(families), 1)
        self.assertEqual(len(families[0]["fields"]), 6)

    def test_colon_inside_a_value_does_not_hide_vertical_kv(self):
        pages = []
        for number in (1, 2):
            content = words("Dimensions", 40, 100) + words("EN10365", 160, 100)
            content += words("Range", 40, 130) + words(f"UB: {number} to 100", 160, 130)
            content += words(f"UC: {number} to 200", 160, 143)
            pages.append(page(number, content))
        kvs = [f for f in scan_pages(pages)["families"] if f["kind"] == "kv"]
        self.assertEqual(len(kvs), 1)
        self.assertEqual([f["text"] for f in kvs[0]["fields"]], ["Dimensions", "Range"])
        self.assertIn("UC:", kvs[0]["occurrences"][0]["labels"][1]["value"])

    def test_spatial_coverage_keeps_fractional_edges_and_empty_spaces(self):
        coverage = spatial_coverage([(1, 10.125, 20.25, 12.5, 22.75),
                                     (2, 11.125, 20.25, 13.5, 22.75),
                                     (1, 16.1, 20.25, 17.9, 22.75)])
        self.assertEqual(coverage, [(10.125, 20.25, 11.125, 22.75, 1),
                                    (11.125, 20.25, 12.5, 22.75, 2),
                                    (12.5, 20.25, 13.5, 22.75, 1),
                                    (16.1, 20.25, 17.9, 22.75, 1)])

    def test_overlapping_words_on_one_page_count_once(self):
        coverage = spatial_coverage([(1, 0, 0, 10, 10), (1, 5, 0, 15, 10),
                                     (1, 0, 0, 10, 10), (2, 5, 5, 10, 15)])
        def level(x, y):
            return sum(c for x0, y0, x1, y1, c in coverage if x0 <= x < x1 and y0 <= y < y1)
        self.assertEqual(level(7, 2), 1)
        self.assertEqual(level(7, 7), 2)
        self.assertEqual(level(7, 12), 1)
        self.assertEqual(level(12, 12), 0)

    def test_spatial_maps_preserve_word_gaps_and_exclude_kv_values(self):
        pages = []
        for number, name in ((1, "Alice"), (2, "Bob")):
            content = words("Full name:", 40, 100) + words(name, 100, 100)
            content += words("City:", 240, 100) + words("Zurich", 270, 100)
            pages.append(page(number, content))
        report = scan_pages(pages)
        boxes = spatial_word_boxes(report["families"][0])
        self.assertEqual(len(boxes), 6)  # Full, name:, City: on each page
        self.assertEqual(boxes[0], (0, 0, 0, 16, 8))
        self.assertEqual(boxes[1], (0, 18, 0, 38, 8))
        self.assertFalse(any(x0 <= 65 < x1 for _, x0, _, x1, _ in boxes))

    def test_spatial_maps_align_translated_occurrences_even_on_one_page(self):
        family = dict(occurrences=[
            dict(page=1, labels=[dict(words=words("Header", 40, 100) + words("Other", 140, 100))]),
            dict(page=1, labels=[dict(words=words("Header", 240, 300) + words("Other", 340, 300))]),
            dict(page=2, labels=[dict(words=words("Header", 70, 150) + words("Other", 170, 150))]),
        ])
        boxes = spatial_word_boxes(family)
        self.assertEqual(boxes, [(i, x0, 0, x1, 8)
                                 for i in range(3) for x0, x1 in ((0, 24), (100, 120))])
        self.assertEqual(spatial_coverage(boxes), [(0, 0, 24, 8, 3), (100, 0, 120, 8, 3)])
        self.assertEqual(spatial_coverage([]), [])


if __name__ == "__main__":
    unittest.main()
