from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from langchain_core.messages import AIMessage
from PIL import Image

from src.agents.cluster_sampling import _clusterers, plan_pages
from src.agents.block_samples import block_samples_node
from src.agents.environment_creation import environment_creation_node
from src.agents.graph import build_graph
from src.agents.template_inference import template_inference_node


def report(method: str, clusters: dict[str, list[int]], reps: dict[str, int],
           page_count: int = 50) -> dict:
    return {"page_count": page_count, "algorithms": {method: {
        "cluster_count": len(clusters), "clusters": clusters, "representatives": reps,
    }}}


class ClusterSamplingPipelineTests(unittest.TestCase):
    def test_graph_clusters_before_environment_and_template_inference(self) -> None:
        edges = {(edge.source, edge.target) for edge in build_graph(Mock()).get_graph().edges}
        self.assertIn(("detect_beginning", "cluster_sampling"), edges)
        self.assertIn(("cluster_sampling", "environment_creation"), edges)
        self.assertIn(("environment_creation", "template_inference"), edges)

    def test_joint_table_kv_coverage_reuses_a_far_page_and_test_is_disjoint(self) -> None:
        tables = report("balanced_complete", {"1": list(range(1, 11)), "2": list(range(30, 36))},
                        {"1": 2, "2": 32})
        kv = report("kv_balanced", {"1": list(range(1, 6)), "2": list(range(30, 41))},
                    {"1": 1, "2": 33})

        selection = plan_pages(tables, kv, source_start=1, total_pages=50,
                               requested_train_pages=3, eval_mode=True, eval_pages=8)

        self.assertEqual(selection["first_source_pages"], [1, 2, 3])
        self.assertEqual(selection["train_source_pages"], [1, 2, 3, 32])
        self.assertEqual(selection["cluster_sample_pages"], [32])
        self.assertEqual(set(selection["cluster_coverage"]),
                         {"table:1", "table:2", "kv:1", "kv:2"})
        self.assertEqual(selection["cluster_coverage"]["table:2"], 32)
        self.assertEqual(selection["cluster_coverage"]["kv:2"], 32)
        self.assertEqual(len(selection["test_source_pages"]), 8)
        self.assertFalse(set(selection["train_source_pages"]) & set(selection["test_source_pages"]))
        self.assertGreater(max(selection["test_source_pages"]), 30)

    def test_source_start_and_eval_reservation_are_respected(self) -> None:
        empty_table = report("balanced_complete", {}, {}, page_count=4)
        empty_kv = report("kv_balanced", {}, {}, page_count=4)
        selection = plan_pages(empty_table, empty_kv, source_start=5, total_pages=8,
                               requested_train_pages=3, eval_mode=True, eval_pages=20)
        self.assertEqual(selection["first_source_pages"], [5, 6, 7])
        self.assertEqual(selection["test_source_pages"], [8])

    def test_cap_keeps_initial_pages_then_maximizes_joint_cluster_coverage(self) -> None:
        tables = report("balanced_complete", {"1": [5], "2": [9], "3": [11]},
                        {"1": 5, "2": 9, "3": 11})
        kv = report("kv_balanced", {"1": [9], "2": [12]}, {"1": 9, "2": 12})
        selection = plan_pages(tables, kv, source_start=5, total_pages=15,
                               requested_train_pages=10, eval_mode=True, eval_pages=20,
                               max_train_pages=2, first_train_pages=1)
        self.assertEqual(selection["first_source_pages"], [5])
        self.assertEqual(selection["train_source_pages"], [5, 9])
        self.assertEqual(selection["cluster_sample_pages"], [9])
        self.assertEqual(selection["cluster_coverage"], {"table:1": 5, "table:2": 9, "kv:1": 9})
        self.assertEqual(selection["uncovered_clusters"], ["kv:2", "table:3"])
        self.assertEqual(selection["test_source_pages"], [6, 7, 8, 10, 11, 12, 13, 14, 15])

    def test_initial_page_count_and_cap_respect_document_and_eval_capacity(self) -> None:
        tables = report("balanced_complete", {}, {})
        kv = report("kv_balanced", {}, {})
        cases = [
            # initial count, cap, eval mode, expected train, expected test
            (1, None, True, [5], [6, 7, 8]),
            (6, 2, True, [5, 6], [7, 8]),
            (20, 20, True, [5, 6, 7], [8]),
            (20, 20, False, [5, 6, 7, 8], []),
            (3, 1, False, [5], []),
        ]
        for initial, cap, eval_mode, train, test in cases:
            with self.subTest(initial=initial, cap=cap, eval_mode=eval_mode):
                selection = plan_pages(tables, kv, source_start=5, total_pages=8,
                                       requested_train_pages=1, eval_mode=eval_mode,
                                       eval_pages=20, max_train_pages=cap, first_train_pages=initial)
                self.assertEqual(selection["first_source_pages"], train)
                self.assertEqual(selection["train_source_pages"], train)
                self.assertEqual(selection["test_source_pages"], test)

    def test_zero_initial_pages_starts_with_cluster_representative(self) -> None:
        tables = report("balanced_complete", {"1": [7, 8]}, {"1": 8})
        kv = report("kv_balanced", {}, {})
        selection = plan_pages(tables, kv, source_start=1, total_pages=10,
                               requested_train_pages=1, eval_mode=True, eval_pages=3,
                               max_train_pages=1, first_train_pages=0)
        self.assertEqual(selection["first_source_pages"], [])
        self.assertEqual(selection["train_source_pages"], [8])
        self.assertEqual(selection["cluster_sample_pages"], [8])
        self.assertEqual(selection["uncovered_clusters"], [])

    def test_cap_is_a_maximum_and_does_not_fill_unused_budget(self) -> None:
        tables = report("balanced_complete", {"1": [9]}, {"1": 9})
        kv = report("kv_balanced", {}, {})
        selection = plan_pages(tables, kv, source_start=1, total_pages=20,
                               requested_train_pages=2, eval_mode=True, eval_pages=3,
                               max_train_pages=10, first_train_pages=1)
        self.assertEqual(selection["train_source_pages"], [1, 9])

    def test_uncapped_selection_still_requires_complete_cluster_coverage(self) -> None:
        tables = report("balanced_complete", {str(p): [p] for p in range(1, 5)},
                        {str(p): p for p in range(1, 5)})
        kv = report("kv_balanced", {}, {})
        with self.assertRaisesRegex(RuntimeError, "Cannot represent every"):
            plan_pages(tables, kv, source_start=1, total_pages=4,
                       requested_train_pages=3, eval_mode=True, eval_pages=20)

    def test_invalid_train_limits_are_rejected(self) -> None:
        tables = report("balanced_complete", {}, {})
        kv = report("kv_balanced", {}, {})
        for options in ({"max_train_pages": 0}, {"max_train_pages": -1},
                        {"first_train_pages": -1}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                plan_pages(tables, kv, source_start=1, total_pages=5,
                           requested_train_pages=3, eval_mode=True, eval_pages=20, **options)

    def test_table_cluster_overlay_uses_physical_source_page_after_front_matter(self) -> None:
        engine, _ = _clusterers()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pdf = root / "source.pdf"
            pages = [Image.new("RGB", (40, 40), "white") for _ in range(5)]
            pages[0].save(pdf, "PDF", save_all=True, append_images=pages[1:])
            labels = [dict(text=f"Key {i}", x0=2 + 7 * i, x1=7 + 7 * i,
                           top=2, bottom=4, line=1) for i in range(3)]
            header = {"bbox": [2, 2, 27, 5], "labels": labels}
            scan = {"pages": [3, 4, 5], "families": [{
                "kind": "table", "id": "table_001", "representative": header,
                "occurrences": [{"page": page, **header} for page in (3, 4, 5)],
            }]}
            lines = ([dict(x0=1, x1=29, top=y, bottom=y) for y in (6, 14, 22)]
                     + [dict(x0=x, x1=x, top=6, bottom=24) for x in (3, 12, 21)])
            with (patch.object(engine.base, "vector_lines", return_value=lines),
                  patch.object(engine.base, "overlay") as overlay):
                result = engine.process(pdf, 3, root, None, 500, "clusters",
                                        start_page=3, scan_report=scan)
            self.assertEqual(result["source_start_page"], 3)
            self.assertEqual(result["page_count"], 3)
            self.assertTrue(overlay.called)
            self.assertEqual(overlay.call_args.args[1]["page"], 3)

    def test_split_remaps_noncontiguous_source_pages_and_keeps_samples_in_build_ocr(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pdf = root / "source.pdf"
            pages = [Image.new("RGB", (40, 40), "white") for _ in range(25)]
            pages[0].save(pdf, "PDF", save_all=True, append_images=pages[1:])
            ocr = root / "ocr.csv"
            with ocr.open("w", newline="", encoding="utf-8") as target:
                writer = csv.DictWriter(target, fieldnames=["id", "page", "x0", "top", "x1", "bottom", "text"])
                writer.writeheader()
                for page in range(1, 26):
                    writer.writerow(dict(id=page, page=page, x0=0, top=0, x1=5,
                                         bottom=5, text=f"source-{page}"))

            state = environment_creation_node({
                "full_pdf_path": str(pdf), "full_ocr_csv_path": str(ocr),
                "output_dir": str(root / "output"), "doc_name": "sample",
                "train_pages": 3, "eval_mode": True, "eval_pages": 3,
                "detected_start_page": 1,
                "train_source_pages": [1, 2, 3, 24],
                "test_source_pages": [4, 20, 25],
                "first_source_pages": [1, 2, 3], "cluster_sample_pages": [24],
                "holdout_pages": 1,
            })

            def entries(path: str) -> list[dict]:
                with open(path, newline="", encoding="utf-8") as source:
                    return list(csv.DictReader(source))

            self.assertEqual([(row["page"], row["text"]) for row in entries(state["ocr_csv_path"])],
                             [("1", "source-1"), ("2", "source-2"), ("3", "source-3"), ("4", "source-24")])
            self.assertEqual(entries(state["build_ocr_csv_path"]), entries(state["ocr_csv_path"]))
            self.assertEqual([(row["page"], row["text"]) for row in entries(state["test_ocr_path"])],
                             [("1", "source-4"), ("2", "source-20"), ("3", "source-25")])
            self.assertEqual(state["build_source_pages"], [1, 2, 3, 24])

    def test_template_prompt_marks_ordering_pages_and_cluster_variants(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "agent_space").mkdir()
            response = AIMessage(content=json.dumps({"template": [{
                "type": "table", "fields": ["Name"], "bid": [0], "node_id": 0,
                "child": -1, "complexity": {"same_row_structure": True, "nested": False},
            }], "metadata": []}))
            llm = Mock()
            llm.invoke.return_value = response
            with patch("src.agents.template_inference.load_ocr_pages", return_value={
                index: {"layout": f"layout-{index}"} for index in range(1, 5)
            }):
                template_inference_node({
                    "agent_space": str(root / "agent_space"), "ocr_csv_path": "unused",
                    "actual_train_pages": 4, "train_source_pages": [1, 2, 3, 24],
                    "first_source_pages": [1, 2, 3], "cluster_sample_pages": [24],
                    "cluster_coverage": {"table:2": 24},
                    "table_cluster_count": 2, "kv_cluster_count": 0,
                }, llm)
            human = llm.invoke.call_args.args[0][1].content
            self.assertIn("source PDF page 24; cluster sample", human)
            self.assertIn("represents clusters: table:2", human)
            self.assertIn("source PDF page 1; ordering anchor", human)
            llm.reset_mock()
            with patch("src.agents.template_inference.load_ocr_pages", return_value={
                1: {"layout": "layout-1"}
            }):
                template_inference_node({
                    "agent_space": str(root / "agent_space"), "ocr_csv_path": "unused",
                    "actual_train_pages": 1, "train_source_pages": [24],
                    "first_source_pages": [], "cluster_sample_pages": [24],
                }, llm)
            human = llm.invoke.call_args.args[0][1].content
            self.assertIn("source PDF page 24; cluster sample", human)
            self.assertNotIn("ordering anchor", human)

    def test_block_samples_include_a_far_cluster_page_even_beyond_random_cap(self) -> None:
        class FakeImage:
            def save(self, path, **_kwargs):
                Path(path).write_bytes(b"png")

        class FakePage:
            bbox = (0, 0, 100, 100)

            def crop(self, _bbox):
                return self

            def to_image(self, **_kwargs):
                return FakeImage()

        class FakePdf:
            pages = [FakePage() for _ in range(4)]

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return None

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train = root / "agent_space" / "train"
            train.mkdir(parents=True)
            word = {"x0": 10, "top": 10, "x1": 20, "bottom": 20, "text": "value"}
            separation = {"pages": [
                {"page": local, "blocks": [{"block_id": f"0_{local}", "words": [word]}]}
                for local in (1, 2, 3, 4)
            ]}
            (train / "block_separation.json").write_text(json.dumps(separation), encoding="utf-8")
            with (patch("src.agents.block_samples.pdfplumber.open", return_value=FakePdf()),
                  patch("src.agents.block_samples._SAMPLE_SIZE", 1)):
                block_samples_node({
                    "pdf_path": str(train / "sample.pdf"), "doc_name": "sample",
                    "train_source_pages": [1, 2, 3, 24],
                    "first_source_pages": [1, 2, 3],
                    "cluster_sample_pages": [24],
                })
            samples = sorted((train / "block_samples" / "0").glob("*.json"))
            self.assertEqual({json.loads(path.read_text())["source_page"] for path in samples},
                             {1, 2, 3, 24})


if __name__ == "__main__":
    unittest.main()
