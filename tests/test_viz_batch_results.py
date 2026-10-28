import os
import json
import io
import tempfile
import unittest
from unittest.mock import patch

from inspect_results.viz_batch_results import (
    _break_even_pages,
    _ground_truth_cost_usd,
    _mean_token_type_percentages,
    _phase_costs_usd,
    _phase_model,
    _phase_token_breakdowns,
    _split_token_types,
    fetch_openrouter_pricing,
    generate_document_results,
    plot_average_token_type_spend,
    plot_document,
    plot_document_token_breakdown,
)


class TokenBreakdownTests(unittest.TestCase):
    def test_template_cost_uses_its_separate_model(self):
        document = {"template_model": "z-ai/glm-5.3-flash",
                    "reference_model": "openai/gpt-5.4"}
        self.assertEqual(_phase_model(document, "template_inference"),
                         "z-ai/glm-5.3-flash")

    def test_separator_and_router_costs_are_attributed_to_hard_model(self):
        document = {
            "llm_model": "vendor/primary",
            "opencode_model": "openrouter/vendor/hard",
            "hard_model": "openrouter/vendor/hard",
            "difficulty_model": "vendor/hard",
        }

        self.assertEqual(_phase_model(document, "block_separation"), "vendor/hard")
        self.assertEqual(
            _phase_model(document, "block_difficulty_router"), "vendor/hard"
        )

    def test_legacy_router_cost_keeps_primary_model_attribution(self):
        document = {
            "llm_model": "vendor/primary",
            "hard_model": "openrouter/vendor/hard",
        }

        self.assertEqual(
            _phase_model(document, "block_difficulty_router"), "vendor/primary"
        )

    @staticmethod
    def _fake_pricing(model_ids):
        return {
            "source": "openrouter_api",
            "endpoint": "https://openrouter.ai/api/v1/models",
            "retrieved_at": "2026-08-28T12:00:00+00:00",
            "unit": "USD per 1M tokens",
            "models": {
                model: {
                    "input_per_million": 1.0,
                    "output_per_million": 2.0,
                }
                for model in model_ids
            },
            "missing_models": [],
        }

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

    def test_ground_truth_uses_saved_reference_model_price(self):
        document = {
            "reference_model": "qwen/qwen3.8-flash",
            "model_pricing": {
                "models": {
                    "qwen/qwen3.8-flash": {
                        "input_per_million": 0.15,
                        "output_per_million": 0.47,
                    }
                }
            },
        }
        self.assertAlmostEqual(
            _ground_truth_cost_usd(
                {"input_tokens": 1_000_000, "output_tokens": 1_000_000},
                document,
            ),
            0.62,
        )

    def test_openrouter_catalog_prices_are_converted_to_per_million(self):
        response = io.BytesIO(json.dumps({
            "data": [{
                "id": "example/model",
                "pricing": {
                    "prompt": "0.0000002",
                    "completion": "0.0000007",
                    "input_cache_read": "0.00000002",
                },
            }]
        }).encode())
        with patch("inspect_results.viz_batch_results.urllib.request.urlopen", return_value=response):
            snapshot = fetch_openrouter_pricing(["openrouter/example/model"])

        self.assertEqual(snapshot["source"], "openrouter_api")
        self.assertEqual(
            snapshot["models"]["example/model"],
            {
                "input_per_million": 0.2,
                "output_per_million": 0.7,
                "cache_read_per_million": 0.02,
            },
        )

    def test_phase_costs_use_exact_saved_block_model_prices(self):
        document = {
            "block_models": {
                "0": "openrouter/vendor/easy",
                "1": "openrouter/vendor/hard",
            },
            "model_pricing": {
                "models": {
                    "vendor/easy": {
                        "input_per_million": 1.0,
                        "output_per_million": 2.0,
                    },
                    "vendor/hard": {
                        "input_per_million": 10.0,
                        "output_per_million": 20.0,
                    },
                }
            },
            "pipeline_phases": [{
                "phase": "block_extraction_scripting",
                "blocks": [
                    {"id": "0", "input_tokens": 1_000_000},
                    {"id": "1", "output_tokens": 1_000_000},
                ],
            }],
        }

        self.assertAlmostEqual(_phase_costs_usd(document)["data_extraction"], 21.0)

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

    def test_single_document_api_writes_plots_and_summary_to_exact_directory(self):
        metadata = {
            "doc_name": "invoice_family",
            "train_pages": 5,
            "test_pages": 20,
            "block_models": {"0": "openrouter/example/model"},
            "pipeline_phases": [{
                "phase": "template_inference",
                "input_tokens": 100,
                "output_tokens": 50,
            }],
            "train_eval": {"accuracy": 0.8},
            "test_eval": {"accuracy": 0.75, "exact_match": False},
            "totals": {"duration_s": 12.5, "input_tokens": 100, "output_tokens": 50},
            "reference_model": "openai/gpt-5.4",
        }
        with tempfile.TemporaryDirectory() as tmp:
            document_dir = os.path.join(tmp, "run")
            output_dir = os.path.join(document_dir, "artifacts", "stats")
            os.makedirs(document_dir)
            with open(os.path.join(document_dir, "metadata.json"), "w") as target:
                json.dump(metadata, target)

            result = generate_document_results(
                document_dir,
                output_dir,
                pricing_fetcher=self._fake_pricing,
            )

            self.assertEqual(
                {os.path.basename(path) for path in result["plots"]},
                {"overview.png", "phase_cost.png", "token_breakdown.png"},
            )
            self.assertTrue(all(os.path.isfile(path) for path in result["plots"]))
            self.assertEqual(result["statistics"]["test_accuracy"], 0.75)
            self.assertEqual(result["statistics"]["block_types"], 1)
            self.assertEqual(result["statistics"]["generation_cost_usd"], 0.0002)
            with open(os.path.join(output_dir, "stats.json")) as source:
                self.assertEqual(json.load(source)["document"], "invoice_family")
            with open(os.path.join(document_dir, "metadata.json")) as source:
                saved_metadata = json.load(source)
            self.assertEqual(saved_metadata["model_pricing"]["source"], "openrouter_api")
            self.assertIn("example/model", saved_metadata["model_pricing"]["models"])
            self.assertIn("openai/gpt-5.4", saved_metadata["model_pricing"]["models"])

    def test_batch_document_wrapper_keeps_nested_results_layout(self):
        document = {
            "_name": "sample",
            "pipeline_phases": [{
                "phase": "template_inference",
                "input_tokens": 10,
                "output_tokens": 5,
            }],
        }
        with tempfile.TemporaryDirectory() as tmp:
            plot_document(document, tmp)

            self.assertTrue(os.path.isfile(os.path.join(tmp, "sample", "overview.png")))
            self.assertTrue(os.path.isfile(os.path.join(tmp, "sample", "phase_cost.png")))
            self.assertTrue(os.path.isfile(os.path.join(tmp, "sample", "token_breakdown.png")))
            self.assertFalse(os.path.exists(os.path.join(tmp, "sample", "stats.json")))


if __name__ == "__main__":
    unittest.main()
