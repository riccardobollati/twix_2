from __future__ import annotations

import json
import os

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import END, START, StateGraph

from src.agents.block_difficulty_router import block_difficulty_router_node
from src.agents.block_artifacts import block_artifacts_node
from src.agents.block_extraction_scripting import block_extraction_scripting_node
from src.agents.block_samples import block_samples_node
from src.agents.block_separation import block_separation_node, run_block_separation_node
from src.agents.collect_extraction_scripts import collect_extraction_scripts_node
from src.agents.detect_beginning import detect_beginning_node
from src.agents.environment_creation import environment_creation_node
from src.agents.eval import eval_node
from src.agents.ground_truth import ground_truth_node
from src.agents.extraction import extraction_node
from src.agents.save_metadata import save_metadata_node
from src.agents.template_inference import ExtractionState, template_inference_node
from src.agents.train_eval import train_eval_node
from src.agents.train_repair import train_repair_node
from src.excluded_blocks_sonar import sonar_node


def _route_after_train_eval(state: dict) -> str:
    """Return 'opencode_fix' if repair should run, otherwise 'extraction'."""
    threshold = state.get("train_accuracy_threshold", 0.0)
    if not threshold or threshold <= 0.0:
        return "extraction"

    document_dir = os.path.dirname(state["agent_space"])
    eval_json    = os.path.join(document_dir, "pipeline", "train_eval", "eval.json")
    try:
        with open(eval_json, encoding="utf-8") as f:
            ev = json.load(f)
        accuracy = ev.get("accuracy", 1.0)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"  warning: could not read {eval_json} ({exc}) — skipping repair, proceeding to extraction")
        return "extraction"

    # A repair round can "improve" train accuracy by violating pipeline
    # invariants (e.g. cross-page mega-blocks that game a small train set), so
    # a degenerate health report fails the gate even above the accuracy
    # threshold — the violations are handed to the next repair round.
    health_problems: list[str] = []
    health_json = os.path.join(document_dir, "pipeline", "train_eval", "pipeline_health.json")
    try:
        with open(health_json, encoding="utf-8") as f:
            health = json.load(f)
        if health.get("degenerate"):
            health_problems = health.get("problems", [])
    except (OSError, json.JSONDecodeError):
        pass

    attempts    = state.get("train_repair_attempts", 0)
    max_retries = state.get("train_repair_max_retries", 3)

    if accuracy >= threshold and not health_problems:
        print(f"  train accuracy {accuracy:.3f} >= {threshold:.3f} — proceeding to test phase")
        return "extraction"

    if accuracy >= threshold:
        reason = f"train pipeline output is degenerate: {'; '.join(health_problems)}"
    else:
        reason = f"train accuracy {accuracy:.3f} < {threshold:.3f}"

    if attempts >= max_retries:
        print(f"  {reason} but max retries ({max_retries}) exhausted — proceeding anyway")
        return "extraction"

    print(f"  {reason} — starting repair attempt {attempts + 1}/{max_retries}")
    return "opencode_fix"


def recursion_limit_for(max_repair_retries: int) -> int:
    """Recursion budget for one graph run: the base node chain plus the extra
    opencode_fix → train_eval → routing steps added by each repair round."""
    return 30 + 4 * max_repair_retries


def build_graph(
    llm: BaseChatModel,
    train_repair_max_retries: int = 3,
    *,
    ground_truth_llm: BaseChatModel | None = None,
    template_inference_llm: BaseChatModel | None = None,
    difficulty_routing_llm: BaseChatModel | None = None,
):
    """Build and compile the PDF template inference graph.

    Args:
        llm: The LangChain-compatible primary OpenRouter chat model.
        template_inference_llm: Optional model reserved for template inference.
            When omitted, the primary model is used for backward compatibility.
        difficulty_routing_llm: Optional hard model reserved for block
            difficulty classification. When omitted, the primary model is used
            for backward compatibility with direct ``build_graph`` callers.
        ground_truth_llm: OpenRouter reference model for train ground truth and
            image-backed final test ground truth. When omitted, the primary
            model is used without page images for backward compatibility with
            direct ``build_graph`` callers.
        train_repair_max_retries: maximum repair rounds the run may use — the
            compiled graph's recursion_limit is derived from it, so pass the
            same value given as "train_repair_max_retries" in the invoke state.

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

    _llm_model = getattr(llm, "model_name", None) or getattr(llm, "model", None) or "unknown"
    _template_inference_llm = template_inference_llm or llm
    _difficulty_routing_llm = difficulty_routing_llm or llm
    _ground_truth_llm = ground_truth_llm or llm
    _difficulty_model = (
        getattr(_difficulty_routing_llm, "model_name", None)
        or getattr(_difficulty_routing_llm, "model", None)
        or "unknown"
    )
    _ground_truth_model = (
        getattr(_ground_truth_llm, "model_name", None)
        or getattr(_ground_truth_llm, "model", None)
        or "unknown"
    )
    graph.add_node(
        "detect_beginning",
        lambda state: detect_beginning_node(state, llm),
    )
    graph.add_node(
        "environment_creation",
        lambda state: {
            **environment_creation_node(state),
            "llm_model": _llm_model,
            "difficulty_model": _difficulty_model,
            "ground_truth_model": _ground_truth_model,
        },
    )
    graph.add_node(
        "template_inference",
        lambda state: template_inference_node(state, _template_inference_llm),
    )
    graph.add_node("block_separation", block_separation_node)
    graph.add_node("run_block_separation", run_block_separation_node)
    graph.add_node("excluded_blocks_sonar", sonar_node)
    graph.add_node("block_artifacts", block_artifacts_node)
    graph.add_node("block_samples", block_samples_node)
    graph.add_node(
        "block_difficulty_router",
        lambda state: block_difficulty_router_node(state, _difficulty_routing_llm),
    )
    graph.add_node("block_extraction_scripting", block_extraction_scripting_node)
    graph.add_node("collect_extraction_scripts", collect_extraction_scripts_node)
    graph.add_node(
        "train_eval",
        lambda state: train_eval_node(state, _ground_truth_llm),
    )
    graph.add_node("opencode_fix", train_repair_node)
    graph.add_node("extraction", extraction_node)
    graph.add_node(
        "ground_truth",
        lambda state: ground_truth_node(
            state,
            _ground_truth_llm,
            window=state.get("gt_test_window") or 20,
            include_images=ground_truth_llm is not None,
        ),
    )
    graph.add_node("eval", eval_node)
    graph.add_node("save_metadata", save_metadata_node)

    graph.add_conditional_edges(
        START,
        lambda state: (
            "environment_creation"
            if state.get("detected_start_page") is not None
            else "detect_beginning"
        ),
        {
            "detect_beginning": "detect_beginning",
            "environment_creation": "environment_creation",
        },
    )
    graph.add_edge("detect_beginning", "environment_creation")
    graph.add_edge("environment_creation", "template_inference")
    graph.add_edge("template_inference", "block_separation")
    graph.add_edge("block_separation", "run_block_separation")
    graph.add_edge("run_block_separation", "excluded_blocks_sonar")
    graph.add_edge("excluded_blocks_sonar", "block_artifacts")
    graph.add_edge("block_artifacts", "block_samples")
    graph.add_edge("block_samples", "block_difficulty_router")
    graph.add_edge("block_difficulty_router", "block_extraction_scripting")
    graph.add_edge("block_extraction_scripting", "collect_extraction_scripts")

    graph.add_edge("collect_extraction_scripts", "train_eval")
    graph.add_conditional_edges(
        "train_eval",
        _route_after_train_eval,
        {"extraction": "extraction", "opencode_fix": "opencode_fix"},
    )
    graph.add_edge("opencode_fix", "train_eval")
    graph.add_conditional_edges(
        "extraction",
        lambda state: "ground_truth" if state.get("eval_mode") else "save_metadata",
    )
    graph.add_edge("ground_truth", "eval")
    graph.add_edge("eval", "save_metadata")
    graph.add_edge("save_metadata", END)

    return graph.compile().with_config(
        recursion_limit=recursion_limit_for(train_repair_max_retries)
    )
