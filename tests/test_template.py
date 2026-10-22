from __future__ import annotations

import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from langchain_core.messages import AIMessage
from openai import LengthFinishReasonError
from openai.types.chat import ChatCompletion

from src.agents import template_inference

from src.agents.template_inference import (
    _dedup_fields,
    _duplicate_fields,
    _flag_signature_duplicates,
    _validate_template,
)


class TemplateValidationTests(unittest.TestCase):
    def test_duplicate_fields_are_deduplicated_deterministically(self) -> None:
        template = [{"node_id": 0, "fields": ["Total", "Total", "Total"]}]

        self.assertTrue(_duplicate_fields(template))
        _dedup_fields(template)

        self.assertEqual(template[0]["fields"], ["Total", "Total (2)", "Total (3)"])
        self.assertEqual(_duplicate_fields(template), [])

    def test_structural_validation_rejects_empty_template(self) -> None:
        self.assertTrue(_validate_template([]))

    def test_identical_field_signatures_are_flagged(self) -> None:
        template = [
            {"node_id": 0, "fields": ["Name", "Date"]},
            {"node_id": 1, "fields": ["date", "name"]},
        ]

        warnings = _flag_signature_duplicates(template)

        self.assertEqual(template[1]["signature_duplicate_of"], 0)
        self.assertEqual(len(warnings), 1)

    def test_xor_nodes_reference_distinct_matching_physical_nodes(self) -> None:
        template = [
            {"type": "xor-table", "node_id": 0, "includes": [1, 2]},
            {"type": "table", "node_id": 1, "fields": ["Width"], "bid": [0], "complexity": {}},
            {"type": "table", "node_id": 2, "fields": ["Diameter"], "bid": [0], "complexity": {}},
        ]
        self.assertEqual(_validate_template(template), [])
        template[2]["type"] = "kv"
        self.assertTrue(_validate_template(template))
        template[2]["type"] = "table"
        template[0]["includes"] = [1, 1]
        self.assertTrue(_validate_template(template))


class TemplateLengthRecoveryTests(unittest.TestCase):
    def length_error(self):
        return LengthFinishReasonError(completion=ChatCompletion(
            id="offline", created=0, model="deepseek/deepseek-v4.1-flash",
            object="chat.completion",
            choices=[{"index": 0, "finish_reason": "length", "message": {
                "role": "assistant", "content": '{"template": ['}}],
            usage={"prompt_tokens": 100, "completion_tokens": 500,
                   "total_tokens": 600, "completion_tokens_details": {"reasoning_tokens": 490}},
        ))

    def success(self):
        return AIMessage(content=json.dumps({"template": [{
            "type": "table", "node_id": 0, "fields": ["Name"], "bid": [0],
            "child": -1, "complexity": {"same_row_structure": True, "nested": False},
        }], "metadata": []}), usage_metadata={
            "input_tokens": 80, "output_tokens": 20, "total_tokens": 100,
        })

    def run_node(self, root, llm):
        workspace = Path(root) / "agent_space"
        workspace.mkdir()
        with patch.object(template_inference, "load_ocr_pages", return_value={
            1: {"layout": "page-one-layout"}, 2: {"layout": "cluster-page-layout"},
        }), patch.object(template_inference, "_spinner") as spinner:
            try:
                return template_inference.template_inference_node({
                    "agent_space": str(workspace), "ocr_csv_path": "unused.csv",
                    "actual_train_pages": 2, "train_source_pages": [1, 136],
                }, llm)
            finally:
                self.spinner_result = spinner.call_args.args[2]

    def model(self, name="deepseek/deepseek-v4.1-flash"):
        return Mock(model_name=name, extra_body={
            "reasoning": {"effort": "low", "exclude": True}, "provider": {"order": ["example"]},
        })

    def test_length_retry_disables_deepseek_reasoning_and_counts_failed_usage(self):
        llm = self.model()
        llm.invoke.side_effect = [self.length_error(), self.success()]
        with tempfile.TemporaryDirectory() as root:
            result = self.run_node(root, llm)
            self.assertTrue((Path(root) / "template.json").is_file())
        self.assertEqual(llm.invoke.call_count, 2)
        first, retry = llm.invoke.call_args_list
        self.assertEqual(first.kwargs, {})
        self.assertEqual(retry.kwargs["extra_body"], {
            "reasoning": {"enabled": False, "exclude": True},
            "provider": {"order": ["example"]},
        })
        self.assertEqual(retry.args[0][:2], first.args[0])
        self.assertIn("cluster-page-layout", retry.args[0][1].content)
        self.assertEqual(len(result["messages"]), 1)
        cost = result["cost"][0]
        self.assertEqual(cost["input_tokens"], 180)
        self.assertEqual(cost["output_tokens"], 520)
        self.assertEqual(cost["length_failures"], 1)
        self.assertEqual(cost["recovery_reasoning"], "none")
        self.assertTrue(self.spinner_result["ok"])
        self.assertEqual(llm.extra_body["reasoning"]["effort"], "low")

    def test_other_models_retry_with_minimal_reasoning(self):
        llm = self.model("openai/gpt-5.4")
        llm.invoke.side_effect = [self.length_error(), self.success()]
        with tempfile.TemporaryDirectory() as root:
            self.run_node(root, llm)
        self.assertEqual(llm.invoke.call_args.kwargs["extra_body"]["reasoning"], {
            "effort": "minimal", "exclude": True,
        })

    def test_repeated_length_failures_stop_after_three_attempts(self):
        llm = self.model()
        llm.invoke.side_effect = [self.length_error() for _ in range(3)]
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(RuntimeError, "failed after 3.*output token limit"):
                self.run_node(root, llm)
            self.assertFalse((Path(root) / "template.json").exists())
        self.assertEqual(llm.invoke.call_count, 3)
        self.assertFalse(self.spinner_result["ok"])
        self.assertEqual(self.spinner_result["tokens"], 1800)

    def test_unrelated_errors_are_not_retried(self):
        llm = self.model()
        llm.invoke.side_effect = ValueError("unrelated failure")
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(ValueError, "unrelated failure"):
                self.run_node(root, llm)
        llm.invoke.assert_called_once()
        self.assertFalse(self.spinner_result["ok"])


if __name__ == "__main__":
    unittest.main()
