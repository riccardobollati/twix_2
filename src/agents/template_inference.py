from __future__ import annotations

import json
import os
import re
import threading
import time
from collections import Counter
from typing import Annotated, Any, NotRequired

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from src.agents._ui import spinner as _spinner
from src.agents._usage import apply_usage, split_tokens
from src.ground_truth_gen.helpers import load_ocr_pages
from src.prompts.template_inference_prompt import PROMPT_TEMPLATE_INFERENCE


# ---------------------------------------------------------------------------
# Cost reducer
# ---------------------------------------------------------------------------

def _merge_cost(a: list[dict] | None, b: list[dict] | None) -> list[dict]:
    return (a or []) + (b or [])


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class ExtractionState(TypedDict):
    """Shared state passed between every node in the extraction graph."""

    # ── inputs to graph.invoke ────────────────────────────────────────────────
    full_pdf_path:     str   # path to the original full PDF
    full_ocr_csv_path: str   # path to the full OCR CSV (all pages)
    output_dir:        NotRequired[str]  # artifact directory (defaults beside input PDF)
    source_sha256:     NotRequired[str]  # fingerprint of the original input PDF
    opencode_model:    NotRequired[str]  # block-separation/fallback OpenCode model
    easy_model:        NotRequired[str]  # OpenCode target for easy blocks
    hard_model:        NotRequired[str]  # OpenCode target for hard blocks/repair
    reference_model:   NotRequired[str]  # direct OpenRouter template/GT model
    doc_name:          str   # document folder name, used to name split PDFs
    train_pages:       int   # number of pages requested for the train set
    eval_mode:         bool  # whether to create a test split
    sonar_drop_percent: NotRequired[float]  # relative drop below mean coverage

    eval_pages:      NotRequired[int]        # max test pages (None = all remaining)
    holdout_pages:   NotRequired[int]        # train pages hidden from script generation
                                             # but included in train eval (default 0)
    gt_test_window:  NotRequired[int]        # pages per LLM call for test GT (default 20)
    block_opencode_models: NotRequired[dict] # template_id → OpenCode model string
    llm_model:          NotRequired[str]     # primary direct model used for this run
    difficulty_model:   NotRequired[str]     # direct hard model used for routing
    ground_truth_model: NotRequired[str]     # OpenRouter model used for held-out test GT
    no_routing:         NotRequired[bool]    # use the easy model for every block type
    use_caveman:        NotRequired[bool]    # prepend /caveman to every OpenCode prompt
    save_log:           NotRequired[bool]    # save raw OpenCode session output to debug/

    # ── set by detect_beginning_node ─────────────────────────────────────────
    detected_start_page: NotRequired[int]    # first physical page with structured data
    beginning_confidence: NotRequired[str]
    beginning_reason:     NotRequired[str]

    # ── train-repair loop ────────────────────────────────────────────────────────
    train_accuracy_threshold: NotRequired[float]  # accuracy target; 0.0 = disabled
    train_repair_max_retries: NotRequired[int]    # max repair attempts (default 3)
    train_repair_attempts:    NotRequired[int]    # how many repair rounds have run

    # ── set by environment_creation_node ─────────────────────────────────────
    agent_space:        NotRequired[str]       # path to agent_space working directory
    pdf_path:           NotRequired[str]       # path to the train PDF
    ocr_csv_path:       NotRequired[str]       # path to the train OCR CSV (all train pages)
    build_ocr_csv_path: NotRequired[str]       # train OCR minus held-out pages — what
                                               # the generated scripts are built against
    actual_train_pages: NotRequired[int]       # actual number of train pages used

    # eval-only — set by environment_creation_node when eval_mode=True
    test_pdf_path: NotRequired[str]
    test_ocr_path: NotRequired[str]
    test_start:    NotRequired[int]
    test_end:      NotRequired[int]

    # ── accumulated across all nodes ─────────────────────────────────────────
    messages: Annotated[list[BaseMessage], add_messages]
    cost:     Annotated[list[dict], _merge_cost]

    # ── set by template_inference_node ────────────────────────────────────────
    template: NotRequired[list[dict[str, Any]]]
    metadata: NotRequired[list[str]]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MAX_ATTEMPTS = 3
_REQUIRED_NODE_KEYS = ("type", "fields", "bid", "node_id", "complexity")


def _parse_json_response(text: str) -> dict[str, Any]:
    """Parse the LLM response as JSON, stripping markdown fences if present."""
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if fenced:
        text = fenced.group(1)
    return json.loads(text)


def _validate_template(template: Any) -> list[str]:
    """Return a list of structural problems in the parsed template ([] = valid)."""
    if not isinstance(template, list) or not template:
        return ['"template" must be a non-empty list of node objects']
    problems: list[str] = []
    for i, node in enumerate(template):
        if not isinstance(node, dict):
            problems.append(f"template node {i} is not an object")
            continue
        missing = [k for k in _REQUIRED_NODE_KEYS if k not in node]
        if missing:
            problems.append(f"template node {i} is missing key(s): {', '.join(missing)}")
        if "fields" in node and not node.get("fields"):
            problems.append(f"template node {i} has an empty \"fields\" array")
    return problems


def _duplicate_fields(template: list[dict]) -> list[str]:
    """Return one message per node that contains duplicate field names ([] = none)."""
    problems: list[str] = []
    for node in template:
        counts = Counter(node.get("fields") or [])
        dups = sorted(name for name, n in counts.items() if n > 1)
        if dups:
            problems.append(
                f"node_id {node.get('node_id')} has duplicate field name(s): {', '.join(repr(d) for d in dups)}"
            )
    return problems


def _dedup_fields(template: list[dict]) -> None:
    """Deterministically dedup duplicate field names by suffixing \" (2)\", \" (3)\", ..."""
    for node in template:
        fields = node.get("fields") or []
        seen: Counter = Counter()
        deduped: list[str] = []
        for name in fields:
            seen[name] += 1
            deduped.append(name if seen[name] == 1 else f"{name} ({seen[name]})")
        node["fields"] = deduped


def _flag_signature_duplicates(template: list[dict]) -> list[str]:
    """Mark nodes whose field sets are identical to another node's.

    Separation scripts identify nodes by their field labels; two nodes with the
    same signature are indistinguishable that way, which corrupts record
    boundaries (a re-match on the twin looks like the template restarting).
    Stamps ``signature_duplicate_of`` on the later node — the flag rides along
    in the template JSON embedded into every downstream OpenCode prompt.
    Returns one warning message per duplicate pair.
    """
    warnings: list[str] = []
    seen: dict[frozenset, Any] = {}
    for node in template:
        signature = frozenset(str(f).strip().lower() for f in node.get("fields") or [])
        if not signature:
            continue
        if signature in seen:
            node["signature_duplicate_of"] = seen[signature]
            warnings.append(
                f"node_id {node.get('node_id')} has the same field signature as "
                f"node_id {seen[signature]} — separation scripts cannot tell them "
                "apart by field labels alone"
            )
        else:
            seen[signature] = node.get("node_id")
    return warnings


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def template_inference_node(state: ExtractionState, llm: BaseChatModel) -> dict[str, Any]:
    """Send page OCR layouts to the reference model to infer the template.

    The LLM call is retried up to 3 times: JSON parse errors and structural
    validation errors are fed back into the conversation. Duplicate field
    names within a node trigger one explicit re-ask; if duplicates persist
    they are deduped deterministically with " (2)", " (3)" suffixes.

    Reads:
        state["ocr_csv_path"]       – train OCR with split-local page numbers.
        state["actual_train_pages"] – number of training pages to inspect.

    Writes:
        messages            – appends the assistant response(s).
        template            – list of template node dicts (type, fields, bid, child, node_id).
        metadata            – list of fixed metadata phrase strings.
    """
    content = [
        "Here are the training pages as OCR layouts. Coordinates are PDF points; "
        "use y positions for lines and x spans for columns and key/value alignment."
    ]
    page_numbers = range(1, state["actual_train_pages"] + 1)
    ocr_pages = load_ocr_pages(state["ocr_csv_path"], page_numbers)
    for page_number in page_numbers:
        layout = ocr_pages[page_number]["layout"]
        content.append(
            f"Page {page_number} OCR layout:\n{layout or '[no OCR words]'}"
        )

    stop_event = threading.Event()
    spinner_thread = threading.Thread(
        target=_spinner, args=("template_inference", stop_event, {}), daemon=True
    )
    spinner_thread.start()

    messages: list[BaseMessage] = [
        SystemMessage(content=PROMPT_TEMPLATE_INFERENCE),
        HumanMessage(content="\n\n".join(content)),
    ]
    responses: list[BaseMessage] = []
    input_tokens = output_tokens = total_tokens = 0
    dup_reasked = False
    parsed: dict[str, Any] | None = None
    last_error = ""

    t0 = time.time()
    try:
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            response = llm.invoke(messages)
            responses.append(response)
            inp, out, tot = split_tokens(response.usage_metadata or {})
            input_tokens  += inp
            output_tokens += out
            total_tokens  += tot

            error: str | None = None
            candidate: dict[str, Any] | None = None
            try:
                candidate = _parse_json_response(response.content)
            except (json.JSONDecodeError, AttributeError, TypeError) as exc:
                error = (
                    f"Your previous response could not be parsed as JSON: {exc}\n"
                    "Return ONLY the single JSON object described in the instructions."
                )

            if candidate is not None:
                problems = _validate_template(candidate.get("template"))
                if problems:
                    error = (
                        "The template JSON you returned is invalid:\n- "
                        + "\n- ".join(problems)
                        + "\nEvery node must have type, fields (non-empty), bid, node_id "
                        "and complexity. Return the corrected JSON object only."
                    )

            if error is None:
                dups = _duplicate_fields(candidate.get("template", []))
                if dups and not dup_reasked and attempt < _MAX_ATTEMPTS:
                    dup_reasked = True
                    error = (
                        "Field names within a node MUST be unique, but your template has duplicates:\n- "
                        + "\n- ".join(dups)
                        + "\nIf the duplicates come from a group-header row spanning sub-columns, "
                        "prefix each sub-column with its group label joined by \" - \" "
                        "(e.g. \"Balance - Total\"). Return the corrected JSON object only."
                    )
                else:
                    if dups:
                        _dedup_fields(candidate["template"])
                        print(f"  warning: template_inference deduped duplicate field names: {'; '.join(dups)}")
                    parsed = candidate
                    break

            last_error = error
            if attempt < _MAX_ATTEMPTS:
                messages = messages + [response, HumanMessage(content=error)]
    finally:
        duration = time.time() - t0
        stop_event.set()
        spinner_thread.join(timeout=0.5)

    if parsed is None:
        raise RuntimeError(
            f"template_inference failed after {_MAX_ATTEMPTS} attempt(s): {last_error}"
        )

    sig_warnings = _flag_signature_duplicates(parsed.get("template", []))
    for warning in sig_warnings:
        print(f"  warning: template_inference: {warning}")

    template_path = os.path.join(os.path.dirname(state["agent_space"]), "template.json")
    with open(template_path, "w", encoding="utf-8") as f:
        json.dump({"template": parsed.get("template", []), "metadata": parsed.get("metadata", [])}, f, indent=2)

    cost_entry: dict = {"phase": "template_inference", "duration_s": round(duration, 3)}
    apply_usage(cost_entry, input_tokens, output_tokens, total_tokens)

    return {
        "messages": responses,
        "template": parsed.get("template", []),
        "metadata": parsed.get("metadata", []),
        "cost": [cost_entry],
    }
