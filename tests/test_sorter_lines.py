from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest

from PIL import Image, ImageDraw


SCRIPT = Path(__file__).resolve().parents[1] / "test-sorter-lines" / "detect_lines.py"
SPEC = importlib.util.spec_from_file_location("detect_lines", SCRIPT)
assert SPEC and SPEC.loader
detect_lines = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(detect_lines)


class SorterLineTests(unittest.TestCase):
    def test_vector_lines_include_pdf_line_and_thin_fill_but_not_large_fill(self):
        page = SimpleNamespace(width=100, height=100,
                               lines=[dict(x0=10, y0=80, x1=90, y1=80)],
                               rects=[dict(x0=10, top=40, x1=90, bottom=41,
                                           stroke=False, fill=True, non_stroking_color=(0, 0, 0)),
                                      dict(x0=10, top=50, x1=90, bottom=70,
                                           stroke=False, fill=True, non_stroking_color=(0, 0, 0)),
                                      dict(x0=10, top=75, x1=90, bottom=76,
                                           stroke=False, fill=True, non_stroking_color=(1, 1, 1))])
        lines = detect_lines.vector_lines(page, min_length=12)
        self.assertEqual([(line["source"], round(line["top"])) for line in lines],
                         [("pdf_line", 20), ("thin_rect", 40)])

    def test_raster_fallback_finds_long_rule_and_veil_is_under_red(self):
        original = Image.new("RGB", (200, 100), "white")
        draw = ImageDraw.Draw(original)
        draw.line((10, 50, 190, 50), fill="black", width=2)
        lines = detect_lines.raster_lines(original, 200, 100, min_length=12)
        self.assertTrue(any(line["source"] == "raster" and abs(line["top"] - 50) <= 2
                            and line["x1"] - line["x0"] >= 170 for line in lines))
        overlay = detect_lines.overlay_lines(original, lines, 200, 100, grey_alpha=100, line_width=3)
        self.assertEqual(overlay.getpixel((100, 50)), (235, 32, 32))
        self.assertLess(overlay.getpixel((100, 10))[0], 255)
        overlay.close()
        original.close()

    def test_page_selection_rejects_out_of_range_pages(self):
        self.assertEqual(detect_lines.page_numbers(None, 50), list(range(1, 31)))
        self.assertEqual(detect_lines.page_numbers(None, 50, limit=None), list(range(1, 51)))
        self.assertEqual(detect_lines.page_numbers(None, 8), list(range(1, 9)))
        self.assertEqual(detect_lines.page_numbers("1,3-4,3", 5), [1, 3, 4])
        self.assertEqual(detect_lines.page_numbers("31", 50), [31])
        with self.assertRaises(ValueError):
            detect_lines.page_numbers("2-6", 5)

    def test_batch_runner_lists_source_pdfs_without_nested_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            data = Path(temporary) / "data"
            data.mkdir()
            (data / "direct.pdf").touch()
            document = data / "variant"
            document.mkdir()
            (document / "direct.pdf").touch()
            nested = document / "agent_space" / "train"
            nested.mkdir(parents=True)
            (nested / "generated.pdf").touch()
            env = {**os.environ, "DATA_DIR": str(data)}
            runner = SCRIPT.parent / "run_all.sh"
            result = subprocess.run(["bash", str(runner), "--list"], env=env,
                                    capture_output=True, text=True, check=True)
            lines = result.stdout.splitlines()
            self.assertEqual(len(lines), 2)
            self.assertTrue(any("direct.pdf ->" in line and "/direct/results/" in line for line in lines))
            self.assertTrue(any("variant/direct.pdf ->" in line and "/variant/results/" in line for line in lines))
            self.assertNotIn("generated.pdf", result.stdout)

    def test_batch_runner_processes_pdf_into_document_results(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data" / "sample"
            data.mkdir(parents=True)
            Image.new("RGB", (120, 120), "white").save(data / "source.pdf", "PDF")
            output = root / "overlays"
            env = {**os.environ, "DATA_DIR": str(data.parent), "OUTPUT_ROOT": str(output)}
            runner = SCRIPT.parent / "run_all.sh"
            subprocess.run(["bash", str(runner), "--dpi", "72"], env=env,
                           capture_output=True, text=True, check=True)
            folder = output / "sample" / "results"
            self.assertTrue((folder / "page_0001_lines.png").is_file())
            self.assertTrue((folder / "lines.json").is_file())


if __name__ == "__main__":
    unittest.main()
