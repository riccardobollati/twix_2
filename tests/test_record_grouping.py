from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.agents.eval import eval_node
from src.record_grouping import group_blocks_into_records


def _block(node_id: int, *, page: int = 1, instance: int = 0, kind: str = "kv") -> dict:
    return {
        "block_id": f"{node_id}_{instance}",
        "type": kind,
        "page": page,
        "data": {},
    }


def _template(*, repeated: set[int] | None = None) -> list[dict]:
    repeated = repeated or set()
    return [
        {
            "node_id": node_id,
            "type": "kv",
            "bid": [node_id],
            "complexity": {"repeated": node_id in repeated},
        }
        for node_id in range(4)
    ]


def _ids(records: list[dict]) -> list[list[str]]:
    return [
        [block["block_id"].split("_", 1)[0] for block in record["data"]]
        for record in records
    ]


class RecordGroupingTests(unittest.TestCase):
    def test_complete_template_restart_starts_a_record(self) -> None:
        blocks = [
            _block(i, instance=j // 4)
            for j, i in enumerate([0, 1, 2, 3, 0, 1, 2, 3])
        ]
        self.assertEqual(
            _ids(group_blocks_into_records(blocks, _template())),
            [["0", "1", "2", "3"], ["0", "1", "2", "3"]],
        )

    def test_duplicate_non_repeated_node_closes_an_incomplete_record(self) -> None:
        blocks = [_block(i, instance=j) for j, i in enumerate([0, 1, 0, 1, 2, 3])]
        self.assertEqual(
            _ids(group_blocks_into_records(blocks, _template())),
            [["0", "1"], ["0", "1", "2", "3"]],
        )

    def test_repeated_complexity_keeps_duplicate_in_current_record(self) -> None:
        blocks = [_block(i, instance=j) for j, i in enumerate([0, 1, 1, 2, 3])]
        self.assertEqual(
            _ids(group_blocks_into_records(blocks, _template(repeated={1}))),
            [["0", "1", "1", "2", "3"]],
        )

    def test_multiple_bid_positions_make_node_repeatable(self) -> None:
        template = _template()
        template[1]["bid"] = [1, 2]
        blocks = [_block(i, instance=j) for j, i in enumerate([0, 1, 1, 2, 3])]
        self.assertEqual(
            _ids(group_blocks_into_records(blocks, template)),
            [["0", "1", "1", "2", "3"]],
        )

    def test_repeated_node_after_completed_record_starts_next_record(self) -> None:
        blocks = [
            _block(i, instance=j)
            for j, i in enumerate([0, 1, 2, 3, 1, 1, 2, 3])
        ]
        self.assertEqual(
            _ids(group_blocks_into_records(blocks, _template(repeated={1}))),
            [["0", "1", "2", "3"], ["1", "1", "2", "3"]],
        )

    def test_adjacent_cross_page_table_is_a_continuation(self) -> None:
        template = _template()
        template[3]["type"] = "table"
        blocks = [
            _block(0), _block(1), _block(2),
            _block(3, page=1, kind="table"),
            _block(3, page=2, kind="table"),
            _block(0, page=2), _block(1, page=2), _block(2, page=2),
            _block(3, page=2, instance=1, kind="table"),
        ]
        self.assertEqual(
            _ids(group_blocks_into_records(blocks, template)),
            [["0", "1", "2", "3", "3"], ["0", "1", "2", "3"]],
        )

    def test_page_gap_forces_boundary_for_sampled_pages(self) -> None:
        blocks = [
            _block(0, page=1),
            _block(1, page=1),
            _block(2, page=3),
            _block(3, page=3),
        ]
        self.assertEqual(
            _ids(group_blocks_into_records(blocks, _template())),
            [["0", "1"], ["2", "3"]],
        )

    @patch("src.agents.eval.render_to_file")
    @patch(
        "src.agents.eval.eval_score",
        return_value={"accuracy": 1.0, "exact_match": True},
    )
    def test_eval_rebuilds_grouped_files_for_legacy_flat_artifacts(
        self, score, _render
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            document = Path(tmp) / "document"
            pipeline = document / "pipeline"
            pipeline.mkdir(parents=True)
            blocks = [_block(i) for i in range(4)]
            for name in ("extracted.json", "ground_truth.json"):
                (pipeline / name).write_text(json.dumps(blocks), encoding="utf-8")

            eval_node({
                "agent_space": str(document / "agent_space"),
                "template": _template(),
            })

            self.assertTrue((pipeline / "grouped.json").is_file())
            self.assertTrue((pipeline / "ground_truth_grouped.json").is_file())
            gold, candidate = score.call_args.args[:2]
            self.assertEqual(_ids(gold), [["0", "1", "2", "3"]])
            self.assertEqual(_ids(candidate), [["0", "1", "2", "3"]])


if __name__ == "__main__":
    unittest.main()
