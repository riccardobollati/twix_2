"""Offline checks for deterministic table-structure page clustering."""
from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np


from src.clustering import cluster_pages


class ClusterPagesTests(unittest.TestCase):
    def test_multiline_words_create_separate_rows_and_center_columns(self):
        words = []
        for y in (20, 40, 60):
            for x in (20, 60, 100):
                words.append(dict(text="value", x0=x - 2, x1=x + 2, top=y - 1, bottom=y + 1))
        words.append(dict(text="wrapped", x0=18, x1=22, top=24, bottom=26))
        lines = cluster_pages.green_lines(words, [(0, 0, 120, 100)], [], 120, 100)
        rows = [line for line in lines if line["source"] == "word_row"]
        columns = [line for line in lines if line["source"] == "word_column"]
        self.assertEqual(len(rows), 3)
        self.assertEqual(len(columns), 3)
        self.assertTrue(any(line["x0"] == 20 and line["bottom"] >= 60 for line in columns))

    def test_repeated_header_regions_stop_at_next_overlapping_header(self):
        headers = [dict(bbox=[10, 10, 110, 20]), dict(bbox=[10, 60, 110, 70])]
        regions = cluster_pages.table_regions(headers, [], 120, 100)
        self.assertEqual(len(regions), 2)
        self.assertEqual(regions[0][3], 60)
        self.assertEqual(regions[1][3], 100)

    def test_final_table_stops_at_last_border_before_page_footer(self):
        headers = [dict(bbox=[10, 10, 110, 20])]
        borders = [dict(x0=10, top=y, x1=110, bottom=y) for y in (30, 40, 50)]
        region = cluster_pages.table_regions(headers, borders, 120, 100)[0]
        self.assertEqual(region[3], 52)

    def test_blue_uses_header_text_without_repeated_body_counts(self):
        header = dict(labels=[dict(text="Date"), dict(text=" Value "), dict(text="Date")])
        signature = cluster_pages.header_signature([header, header])
        self.assertEqual(signature, {"date": 1, "value": 1})

    def test_blue_connects_header_word_centers_on_each_line(self):
        header = dict(labels=[dict(text="A", x0=10, x1=20, top=10, bottom=20, line=1),
                              dict(text="B", x0=50, x1=70, top=10, bottom=20, line=1),
                              dict(text="C", x0=10, x1=20, top=30, bottom=40, line=2)])
        lines = cluster_pages.blue_lines([header])
        self.assertEqual(len(lines), 1)
        self.assertEqual((lines[0]["x0"], lines[0]["x1"], lines[0]["top"]), (15, 60, 15))

    def test_optional_column_is_allowed_but_larger_header_change_is_split(self):
        base = np.zeros(cluster_pages.GRID_X * cluster_pages.GRID_Y * 2)
        base[4] = 1
        features = [dict(red=base, green=base, blue_geometry=base,
                         blue=cluster_pages.Counter({"a": 1, "b": 1})),
                    dict(red=base, green=base, blue_geometry=base,
                         blue=cluster_pages.Counter({"a": 1, "b": 1})),
                    dict(red=base, green=base, blue_geometry=base,
                         blue=cluster_pages.Counter({"a": 1, "b": 1, "extra": 1}))]
        similarities = cluster_pages.similarity_matrices(features)
        tight = cluster_pages.CONFIGS["tight_complete"]
        structural = cluster_pages.CONFIGS["structure_complete"]
        tight_labels, _ = cluster_pages.cluster(similarities, tight[1], tight[0], tight[2])
        structural_labels, _ = cluster_pages.cluster(similarities, structural[1], structural[0], structural[2])
        self.assertEqual(len(set(tight_labels)), 1)  # one extra field is allowed by this threshold
        self.assertEqual(len(set(structural_labels)), 1)

        # Several different header labels should split only the strict view.
        features[2]["blue"] = cluster_pages.Counter({"a": 1, "c": 1, "d": 1})
        similarities = cluster_pages.similarity_matrices(features)
        tight_labels, _ = cluster_pages.cluster(similarities, tight[1], tight[0], tight[2])
        structural_labels, _ = cluster_pages.cluster(similarities, structural[1], structural[0], structural[2])
        self.assertEqual(len(set(tight_labels)), 2)
        self.assertEqual(len(set(structural_labels)), 1)

    def test_single_page_is_supported(self):
        similarities = {key: np.eye(1) for key in ("red", "green", "blue")}
        labels, distance = cluster_pages.cluster(similarities, {"red": 1}, "complete", 0.8)
        self.assertEqual(labels, [1])
        self.assertEqual(distance.shape, (1, 1))

    def test_blank_page_is_excluded_from_template_candidates(self):
        with tempfile.TemporaryDirectory() as temporary:
            pdf_path = Path(temporary) / "sample.pdf"
            stream = b"10 50 m 190 50 l S\nBT /F1 10 Tf 20 65 Td (Column     Value     Amount) Tj ET\n"
            objects = [
                b"<< /Type /Catalog /Pages 2 0 R >>",
                b"<< /Type /Pages /Kids [3 0 R 4 0 R] /Count 2 >>",
                b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 100] /Resources << /Font << /F1 5 0 R >> >> /Contents 6 0 R >>",
                b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 100] /Resources << /Font << /F1 5 0 R >> >> /Contents 7 0 R >>",
                b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
                b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"endstream",
                b"<< /Length 0 >>\nstream\n\nendstream",
            ]
            content = bytearray(b"%PDF-1.4\n")
            offsets = [0]
            for number, obj in enumerate(objects, 1):
                offsets.append(len(content))
                content.extend(f"{number} 0 obj\n".encode() + obj + b"\nendobj\n")
            xref = len(content)
            content.extend(f"xref\n0 {len(offsets)}\n0000000000 65535 f \n".encode())
            for offset in offsets[1:]:
                content.extend(f"{offset:010d} 00000 n \n".encode())
            content.extend(f"trailer\n<< /Size {len(offsets)} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode())
            pdf_path.write_bytes(content)
            result = cluster_pages.process(pdf_path, 2, Path(temporary) / "out", None, 0)
            self.assertEqual(result["skipped_pages"], [2])
            self.assertEqual(result["algorithms"]["tight_complete"]["clusters"], {"1": [1]})


if __name__ == "__main__":
    unittest.main()
