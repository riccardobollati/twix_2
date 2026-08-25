import os
import tempfile
import unittest

from viz_batch_results import (
    _break_even_pages,
    _ground_truth_cost_usd,
    _mean_token_type_percentages,
    _phase_token_breakdowns,
    _split_token_types,
    plot_average_token_type_spend,
    plot_document_token_breakdown,
)


class TokenBreakdownTests(unittest.TestCase):
    def test_ground_truth_is_priced_as_gpt_5_4(self):
        # GPT-5.4 list pricing is $2.50/M input and $15/M output tokens.
        self.assertAlmostEqual(
            _ground_truth_cost_usd({"input_tokens": 1_000_000, "output_tokens": 1_000_000}),
            17.5,
        )

    def test_break_even_prices_train_ground_truth_as_gpt_5_4(self):
        result = _break_even_pages(
            {
                "train_pages": 1,
                "test_pages": 1,
                "pipeline_phases": [],
                "train_eval": {
                    "ground_truth_cost": {"input_tokens": 1_000_000},
                    "agentic_pipeline_cost": {},
                },
                "test_eval": {
                    "ground_truth_cost": {"input_tokens": 1_000_000},
                    "agentic_pipeline_cost": {},
                },
            }
        )
        self.assertAlmostEqual(result, 1.0)

    def test_input_and_output_are_split(self):
        self.assertEqual(
            _split_token_types(
                {
                    "input_tokens": 100,
                    "output_tokens": 50,
                }
            ),
            {"input": 100.0, "output": 50.0},
        )

    def test_camel_case_usage_is_supported(self):
        self.assertEqual(
            _split_token_types(
                {
                    "inputTokens": 80,
                    "outputTokens": 35,
                }
            ),
            {"input": 80.0, "output": 35.0},
        )

    def test_phase_buckets_include_extraction_and_all_repairs(self):
        doc = {
            "pipeline_phases": [
                {
                    "phase": "template_inference",
                    "input_tokens": 100,
                    "output_tokens": 50,
                },
                {
                    "phase": "block_difficulty_router",
                    "input_tokens": 10,
                    "output_tokens": 5,
                },
                {
                    "phase": "block_extraction_scripting",
                    "input_tokens": 20,
                    "output_tokens": 12,
                },
            ],
            "train_eval": {
                "repairs": [
                    {"input_tokens": 7, "output_tokens": 4},
                    {"input_tokens": 3, "output_tokens": 6},
                ]
            },
        }

        phases = _phase_token_breakdowns(doc)

        self.assertEqual(
            phases["template_inference"],
            {"input": 100.0, "output": 50.0},
        )
        self.assertEqual(
            phases["data_extraction"],
            {"input": 30.0, "output": 17.0},
        )
        self.assertEqual(
            phases["fixing_loops"],
            {"input": 10.0, "output": 10.0},
        )

    def test_percentages_are_computed_per_document_before_averaging(self):
        means, count = _mean_token_type_percentages(
            [
                {"input": 90.0, "output": 10.0},
                {"input": 10.0, "output": 90.0},
                {"input": 0.0, "output": 0.0},
            ]
        )

        self.assertEqual(count, 2)
        self.assertAlmostEqual(means["input"], 50.0)
        self.assertAlmostEqual(means["output"], 50.0)

    def test_breakdown_plots_are_written(self):
        doc = {
            "_name": "sample",
            "pipeline_phases": [
                {
                    "phase": "template_inference",
                    "input_tokens": 100,
                    "output_tokens": 50,
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            document_path = os.path.join(tmp, "document.png")
            average_path = os.path.join(tmp, "average.png")

            plot_document_token_breakdown(doc, document_path)
            plot_average_token_type_spend([doc], average_path)

            self.assertTrue(os.path.isfile(document_path))
            self.assertTrue(os.path.isfile(average_path))


if __name__ == "__main__":
    unittest.main()
