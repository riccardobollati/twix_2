from __future__ import annotations

import json
import os

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import END, START, StateGraph

from src.agents.block_extraction_scripting import block_extraction_scripting_node
from src.agents.block_samples import block_samples_node
from src.agents.block_separation import block_separation_node, run_block_separation_node
from src.agents.collect_extraction_scripts import collect_extraction_scripts_node
from src.agents.environment_creation import environment_creation_node
from src.agents.eval import eval_node
from src.agents.ground_truth import ground_truth_node
from src.agents.grouping import grouping_node, merge_records_node
from src.agents.record_separation import record_separation_node, run_record_separation_node
from src.agents.save_metadata import save_metadata_node
from src.agents.template_inference import ExtractionState, template_inference_node
from src.agents.train_eval import train_eval_node
from src.agents.train_repair import train_repair_node


def _route_after_train_eval(state: dict) -> str:
    """Return 'codex_fix' if repair should run, otherwise 'grouping'."""
    threshold = state.get("train_accuracy_threshold", 0.0)
    if not threshold or threshold <= 0.0:
        return "grouping"

    document_dir = os.path.dirname(state["agent_space"])
    eval_json    = os.path.join(document_dir, "pipeline", "train_eval", "eval.json")
    try:
        with open(eval_json, encoding="utf-8") as f:
            ev = json.load(f)
        accuracy = ev.get("accuracy", 1.0)
    except (OSError, json.JSONDecodeError):
        return "grouping"

    attempts    = state.get("train_repair_attempts", 0)
    max_retries = state.get("train_repair_max_retries", 3)

    if accuracy >= threshold:
        print(f"  train accuracy {accuracy:.3f} >= {threshold:.3f} — proceeding to test phase")
        return "grouping"
    if attempts >= max_retries:
        print(
            f"  train accuracy {accuracy:.3f} < {threshold:.3f} "
            f"but max retries ({max_retries}) exhausted — proceeding anyway"
        )
        return "grouping"

    print(
        f"  train accuracy {accuracy:.3f} < {threshold:.3f} "
        f"— starting repair attempt {attempts + 1}/{max_retries}"
    )
    return "codex_fix"


def build_graph(llm: BaseChatModel):
    """Build and compile the PDF template inference graph.

    Args:
        llm: Any LangChain-compatible vision chat model (BaseChatModel).

    Returns:
        A compiled LangGraph ready to be invoked with an ExtractionState.

    Example::

        graph = build_graph(llm)
        result = graph.invoke({
            "full_pdf_path": "...", "full_ocr_csv_path": "...",
            "doc_name": "...", "train_pages": 3, "eval_mode": False,
        })
        print(result["template"])
    """
    graph = StateGraph(ExtractionState)

    graph.add_node("environment_creation", environment_creation_node)
    graph.add_node(
        "template_inference",
        lambda state: template_inference_node(state, llm),
    )
    graph.add_node("record_separation", record_separation_node)
    graph.add_node("run_record_separation", run_record_separation_node)
    graph.add_node("block_separation", block_separation_node)
    graph.add_node("run_block_separation", run_block_separation_node)
    graph.add_node("block_samples", block_samples_node)
    graph.add_node("block_extraction_scripting", block_extraction_scripting_node)
    graph.add_node("collect_extraction_scripts", collect_extraction_scripts_node)
    graph.add_node("train_eval", lambda state: train_eval_node(state, llm))
    graph.add_node("codex_fix", train_repair_node)
    graph.add_node("grouping", grouping_node)
    graph.add_node("merge_records", merge_records_node)
    graph.add_node("ground_truth", lambda state: ground_truth_node(state, llm, window=state.get("gt_test_window") or 20))
    graph.add_node("eval", eval_node)
    graph.add_node("save_metadata", save_metadata_node)

    graph.add_edge(START, "environment_creation")
    graph.add_edge("environment_creation", "template_inference")
    graph.add_edge("template_inference", "record_separation")
    graph.add_edge("record_separation", "run_record_separation")
    graph.add_edge("run_record_separation", "block_separation")
    graph.add_edge("block_separation", "run_block_separation")
    graph.add_edge("run_block_separation", "block_samples")
    graph.add_edge("block_samples", "block_extraction_scripting")
    graph.add_edge("block_extraction_scripting", "collect_extraction_scripts")

    graph.add_edge("collect_extraction_scripts", "train_eval")
    graph.add_conditional_edges(
        "train_eval",
        _route_after_train_eval,
        {"grouping": "grouping", "codex_fix": "codex_fix"},
    )
    graph.add_edge("codex_fix", "train_eval")
    graph.add_edge("grouping", "merge_records")
    graph.add_conditional_edges(
        "merge_records",
        lambda state: "ground_truth" if state.get("eval_mode") else "save_metadata",
    )
    graph.add_edge("ground_truth", "eval")
    graph.add_edge("eval", "save_metadata")
    graph.add_edge("save_metadata", END)

    return graph.compile()
