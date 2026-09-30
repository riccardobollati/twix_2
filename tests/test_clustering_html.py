"""Offline checks for the batch results browser and resume validation."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "test-clustering"))
import build_html  # noqa: E402
import run_all  # noqa: E402
import verify_results  # noqa: E402
import cluster_pages_v2  # noqa: E402
import run_viewing_2  # noqa: E402


def sample_report(pdf_path: Path) -> dict:
    algorithms = {}
    for method in build_html.METHODS:
        algorithms[method] = {
            "cluster_count": 2, "clusters": {"1": [1], "2": [2]},
            "representatives": {"1": 1, "2": 2},
            "mean_silhouette": 0.7,
            "first_five_cluster_coverage": 1.0,
            "header_family_agreement": 0.8,
        }
    return {
        "version": 1, "result_name": "sample", "source_pdf": str(pdf_path.resolve()),
        "source_total_pages": 3, "source_mtime_ns": pdf_path.stat().st_mtime_ns,
        "page_count": 2, "skipped_pages": [],
        "pages": [dict(page=n, red_line_count=1, green_line_count=2,
                       blue_line_count=1, word_count=10, header_families=["table_001"], eligible=True)
                  for n in (1, 2)],
        "algorithms": algorithms, "overlays": ["page_0001_features.png", "page_0002_features.png"],
    }


class ClusteringHtmlTests(unittest.TestCase):
    def test_table_view_rebuilds_results_from_an_older_clustering_revision(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pdf_path = root / "sample.pdf"
            pdf_path.touch()
            result_path = root / "sample" / "clusters.json"
            result_path.parent.mkdir()
            report = sample_report(pdf_path)
            report["version"] = 2
            report["algorithms"] = {name: next(iter(report["algorithms"].values()))
                                    for name in cluster_pages_v2.METHODS}
            for filename in report["overlays"]:
                (result_path.parent / filename).touch()
            result_path.write_text(json.dumps(report), encoding="utf-8")
            self.assertFalse(run_viewing_2.valid_result(result_path, pdf_path.resolve(), 3, 2))
            report["algorithm_revision"] = cluster_pages_v2.TABLE_ALGORITHM_REVISION
            result_path.write_text(json.dumps(report), encoding="utf-8")
            self.assertTrue(run_viewing_2.valid_result(result_path, pdf_path.resolve(), 3, 2))

    def test_second_viewer_uses_its_own_methods_and_title(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            pdf_path = base / "sample.pdf"
            pdf_path.touch()
            folder = base / "viewing-2" / "sample"
            folder.mkdir(parents=True)
            report = sample_report(pdf_path)
            report["version"] = 2
            report["algorithms"] = {name: next(iter(report["algorithms"].values()))
                                    for name in cluster_pages_v2.METHODS}
            (folder / "clusters.json").write_text(json.dumps(report), encoding="utf-8")
            methods = list(cluster_pages_v2.METHODS)
            build_html.build(folder.parent, [("sample", pdf_path, 3)], methods, "Clustering viewing 2")
            page = (folder / "index.html").read_text(encoding="utf-8")
            index = (folder.parent / "index.html").read_text(encoding="utf-8")
            self.assertIn("Clustering viewing 2", page)
            self.assertIn("header_strict", page)
            self.assertIn('<option value="balanced_complete" selected>', page)
            self.assertIn("Balanced complete clusters", index)
            self.assertIn("Structure average clusters", index)

    def test_browser_shows_all_methods_page_links_and_scan_limit(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            pdf_path = base / "data" / "sample" / "sample.pdf"
            pdf_path.parent.mkdir(parents=True)
            pdf_path.touch()
            output = base / "test-clustering"
            folder = output / "sample"
            folder.mkdir(parents=True)
            report = sample_report(pdf_path)
            (folder / "clusters.json").write_text(json.dumps(report), encoding="utf-8")
            rows = build_html.build(output, [("sample", pdf_path.resolve(), 3), ("pending", base / "missing.pdf", 10)])
            document = (folder / "index.html").read_text(encoding="utf-8")
            index = (output / "index.html").read_text(encoding="utf-8")
            self.assertEqual([row["status"] for row in rows], ["complete", "pending"])
            self.assertIn("Scanned pages 1–2 of 3", document)
            self.assertIn("../../data/sample/sample.pdf", document)
            self.assertIn("structure_single", document)
            self.assertIn("page_0001_features.png", document)
            self.assertIn("sample/index.html", index)
            self.assertIn("1 / 2", index)

    def test_resume_requires_matching_source_and_present_overlays(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            pdf_path = base / "sample.pdf"
            pdf_path.touch()
            result_path = base / "sample" / "clusters.json"
            result_path.parent.mkdir()
            report = sample_report(pdf_path)
            result_path.write_text(json.dumps(report), encoding="utf-8")
            self.assertFalse(run_all.valid_result(result_path, pdf_path.resolve(), 3, 2))
            for filename in report["overlays"]:
                (result_path.parent / filename).touch()
            self.assertTrue(run_all.valid_result(result_path, pdf_path.resolve(), 3, 2))
            self.assertFalse(run_all.valid_result(result_path, pdf_path.resolve(), 3, 3))

    def test_audit_detects_missing_cluster_page(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            pdf_path = folder / "sample.pdf"
            pdf_path.touch()
            report = sample_report(pdf_path)
            for filename in report["overlays"]:
                (folder / filename).touch()
            (folder / "index.html").touch()
            self.assertEqual(verify_results.validate_report(report, folder, 3, 2), [])
            report["algorithms"]["tight_complete"]["clusters"]["2"] = []
            self.assertTrue(any("partition" in issue for issue in
                                verify_results.validate_report(report, folder, 3, 2)))


if __name__ == "__main__":
    unittest.main()
