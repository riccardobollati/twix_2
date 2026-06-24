"""eval.md §7.2 — render eval.json into human-readable eval.txt."""

from __future__ import annotations

import json
from pathlib import Path
from collections import defaultdict


REASON_EXPLANATIONS = {
    "R-RECORD-MISSING":       "gold record has no candidate counterpart above tau_record=0.3",
    "R-RECORD-EXTRA":         "candidate record has no gold counterpart above tau_record=0.3",
    "R-NODE-UNALIGNED":       "no candidate node aligned to this gold node above tau=0.4",
    "R-NODE-EXTRA":           "candidate node has no gold counterpart above tau=0.4",
    "R-TYPE-MISMATCH":        "aligned pair has different `type` (eval.md §3)",
    "R-HEADER-DIFFERS":       "table header strings differ after eval.md §3.4 normalization",
    "R-HEADER-ORDER":         "table headers are the same set but in different order",
    "R-ROW-MISSING":          "gold row has no aligned candidate row (eval.md §5.1)",
    "R-ROW-EXTRA":            "candidate row has no aligned gold row (eval.md §5.1)",
    "R-CELL-DIFFERS":         "aligned cell value differs after eval.md §3.4",
    "R-KV-KEY-MISSING":       "key_value: key present in gold, absent in candidate",
    "R-KV-KEY-EXTRA":         "key_value: key present in candidate, absent in gold",
    "R-KV-VALUE-DIFFERS":     "key_value: shared key but values differ after eval.md §3.4",
    "R-METADATA-STRING-MISSING": "metadata: string present in gold, absent in candidate",
    "R-METADATA-STRING-EXTRA":   "metadata: string present in candidate, absent in gold",
    "R-PARENT-DIFFERS":       "aligned node has a different parent under the alignment",
    "R-NORMALIZATION-RESIDUAL": (
        "strings differ only because eval.md §3.4 didn't fold a particular character class "
        "(en-dash, smart quotes, NBSP, ...) - usually a bug in the extractor, not in the gold"
    ),
}

SUGGESTED_FIXES = {
    "R-RECORD-MISSING":       "the candidate failed to emit this record; check record-boundary detection.",
    "R-RECORD-EXTRA":         "the candidate invented a record that is not in the gold; tighten boundary logic.",
    "R-NODE-UNALIGNED":       "emit a node for this content; check the parser is not collapsing this block.",
    "R-NODE-EXTRA":           "the extractor is producing a spurious node; gate the rule that fires here.",
    "R-TYPE-MISMATCH":        "re-classify this block (table vs key_value vs metadata).",
    "R-HEADER-DIFFERS":       "fix header detection: the column header text drifted from gold.",
    "R-HEADER-ORDER":         "preserve column order when emitting headers.",
    "R-ROW-MISSING":          "row is not being parsed; check row-detection heuristics for this table.",
    "R-ROW-EXTRA":            "candidate is splitting a single row into multiple, or hallucinating a row.",
    "R-CELL-DIFFERS":         "the cell value is wrong; check column-boundary detection or text grouping.",
    "R-KV-KEY-MISSING":       "add this field to the key_value block; the parser missed it.",
    "R-KV-KEY-EXTRA":         "remove this spurious field from the key_value block.",
    "R-KV-VALUE-DIFFERS":     "the value text is being captured incorrectly; check OCR / line wrapping.",
    "R-METADATA-STRING-MISSING": "include this metadata string in the metadata block.",
    "R-METADATA-STRING-EXTRA":   "drop this string from the metadata block; it is not in the gold.",
    "R-PARENT-DIFFERS":       "fix the parent linkage when emitting this node.",
    "R-NORMALIZATION-RESIDUAL": "normalize hyphen/dash, smart quotes, or NBSP before emitting cells.",
}


def _truncate(s, limit: int = 80) -> str:
    if s is None:
        return "null"
    if not isinstance(s, str):
        try:
            s = json.dumps(s, ensure_ascii=False)
        except Exception:
            s = str(s)
    if len(s) > limit:
        return s[:limit] + " ... (truncated)"
    return s


def _q(v) -> str:
    if v is None:
        return "null"
    if isinstance(v, (dict, list)):
        try:
            return json.dumps(v, ensure_ascii=False)
        except Exception:
            return str(v)
    return f'"{v}"'


def _fmt_num(x) -> str:
    if x is None:
        return "n/a"
    try:
        return f"{float(x):.3f}"
    except Exception:
        return str(x)


def _record_pair_key(entry: dict) -> tuple:
    return (
        entry.get("gold_record_id") or "",
        entry.get("candidate_record_id") or "",
    )


def _group_by_record(entries: list[dict]) -> dict:
    groups: dict = defaultdict(list)
    for e in entries:
        groups[_record_pair_key(e)].append(e)
    return dict(groups)


def _subheading(gold_rid, cand_rid) -> str:
    if gold_rid and cand_rid:
        return f"--- record {gold_rid} <-> {cand_rid} ---"
    if gold_rid:
        return f"--- gold record {gold_rid} (no candidate) ---"
    if cand_rid:
        return f"--- candidate record {cand_rid} (no gold) ---"
    return "--- (unknown record) ---"


def render(eval_json: dict, threshold: float | None = None) -> str:
    """Return the eval.txt body for the given eval.json dict."""
    threshold = threshold if threshold is not None else eval_json.get("threshold", 0.9)
    accuracy = float(eval_json.get("accuracy", 0.0))
    if accuracy >= float(threshold):
        verdict = "PASS"
    elif accuracy >= float(threshold) - 0.05:
        verdict = "BORDERLINE"
    else:
        verdict = "FAIL"

    lines: list[str] = []
    lines.append("=== EVAL REPORT =========================================================")
    lines.append(f"doc:        {eval_json.get('doc_name', '')}")
    ref = eval_json.get("reference", "agent")
    cand = eval_json.get("candidate", "code")
    ref_model = eval_json.get("reference_model") or "?"
    cand_model = eval_json.get("candidate_model") or "?"
    lines.append(f"reference:  {ref} ({ref_model})")
    lines.append(f"candidate:  {cand} ({cand_model})")
    lines.append(f"threshold:  {float(threshold):.2f}")
    em = eval_json.get("exact_match", False)
    lines.append(f"accuracy:   {accuracy:.3f}      exact_match: {em}     verdict: {verdict}")
    lines.append("")

    comp = eval_json.get("components", {})
    rm = comp.get("RecordMatchF1", {})
    nm = comp.get("NodeMatchF1", {})
    cs = comp.get("ContentScore", {})
    ss = comp.get("StructureScore", {})
    by_type = cs.get("by_type", {}) or {}
    lines.append("--- Score breakdown ----------------------------------------------------")
    if rm:
        lines.append(
            f"RecordMatchF1   typed={_fmt_num(rm.get('typed_f1'))}  "
            f"P={_fmt_num(rm.get('precision'))}  R={_fmt_num(rm.get('recall'))}"
        )
    lines.append(
        f"NodeMatchF1     typed={_fmt_num(nm.get('typed_f1'))}  "
        f"untyped={_fmt_num(nm.get('untyped_f1'))}  "
        f"P={_fmt_num(nm.get('precision'))}  R={_fmt_num(nm.get('recall'))}"
    )
    lines.append(
        f"ContentScore    overall={_fmt_num(cs.get('overall'))}   "
        f"table={_fmt_num(by_type.get('table'))}  "
        f"key_value={_fmt_num(by_type.get('key_value'))}  "
        f"metadata={_fmt_num(by_type.get('metadata'))}"
    )
    lines.append(
        f"StructureScore  edge={_fmt_num(ss.get('edge_accuracy'))}  "
        f"path={_fmt_num(ss.get('path_accuracy'))}  "
        f"note_agreement={_fmt_num(ss.get('note_agreement_rate'))}"
    )
    lines.append("")

    # --- Records section ---
    rec_alignment = eval_json.get("record_alignment") or []
    missing_gold_recs = eval_json.get("missing_gold_records") or []
    extra_cand_recs = eval_json.get("extra_candidate_records") or []
    lines.append("--- Records ------------------------------------------------------------")
    if rec_alignment:
        for ra in rec_alignment:
            lines.append(
                f"matched: {ra.get('gold_record_id')} <-> {ra.get('candidate_record_id')} "
                f"(sim={_fmt_num(ra.get('record_sim'))})"
            )
    else:
        lines.append("matched: (none)")
    if missing_gold_recs:
        lines.append("missing gold records:")
        for r in missing_gold_recs:
            lines.append(f"  {r.get('gold_record_id')}  [{r.get('reason')}]  "
                         f"{REASON_EXPLANATIONS.get(r.get('reason', ''), '')}")
    else:
        lines.append("missing gold records:    (none)")
    if extra_cand_recs:
        lines.append("extra candidate records:")
        for r in extra_cand_recs:
            lines.append(f"  {r.get('candidate_record_id')}  [{r.get('reason')}]  "
                         f"{REASON_EXPLANATIONS.get(r.get('reason', ''), '')}")
    else:
        lines.append("extra candidate records: (none)")
    lines.append("")

    missing = eval_json.get("missing_gold_nodes", []) or []
    extra = eval_json.get("extra_candidate_nodes", []) or []
    type_mm = eval_json.get("type_mismatches", []) or []
    content_diffs = eval_json.get("content_diffs", []) or []
    structure_diffs = eval_json.get("structure_diffs", []) or []

    def section(title: str, count: int, body_lines: list[str]):
        lines.append(f"--- {title} ({count}) " + "-" * max(0, 60 - len(title)))
        if not body_lines:
            lines.append("(none)")
        else:
            lines.extend(body_lines)
        lines.append("")

    # --- Missing nodes (grouped by record) ---
    body: list[str] = []
    for (gid, cid), group in _group_by_record(missing).items():
        body.append(_subheading(gid, cid))
        for m in group:
            reason = m.get("reason", "R-NODE-UNALIGNED")
            body.append(f"[gold {m.get('gold_id')} / {m.get('type')}]")
            body.append(f"  what:  gold node id={m.get('gold_id')!r}, type={m.get('type')!r}")
            body.append(
                f"  where: closest candidate={m.get('closest_candidate_id')!r} "
                f"(sim={_fmt_num(m.get('closest_similarity'))})"
            )
            body.append(f"  why:   [{reason}] {REASON_EXPLANATIONS.get(reason, '')}")
            body.append(f"  fix:   {SUGGESTED_FIXES.get(reason, '')}")
    section("Missing nodes", len(missing), body)

    # --- Extra nodes (grouped by record) ---
    body = []
    for (gid, cid), group in _group_by_record(extra).items():
        body.append(_subheading(gid, cid))
        for e in group:
            reason = e.get("reason", "R-NODE-EXTRA")
            body.append(f"[candidate {e.get('candidate_id')} / {e.get('type')}]")
            body.append(f"  what:  candidate node id={e.get('candidate_id')!r}, type={e.get('type')!r}")
            body.append(f"  why:   [{reason}] {REASON_EXPLANATIONS.get(reason, '')}")
            body.append(f"  fix:   {SUGGESTED_FIXES.get(reason, '')}")
    section("Extra (hallucinated) nodes", len(extra), body)

    # --- Type mismatches (grouped by record) ---
    body = []
    for (gid, cid), group in _group_by_record(type_mm).items():
        body.append(_subheading(gid, cid))
        for t in group:
            reason = t.get("reason", "R-TYPE-MISMATCH")
            body.append(f"[gold {t.get('gold_id')} / candidate {t.get('candidate_id')}]")
            body.append(f"  gold_type:      {_q(t.get('gold'))}")
            body.append(f"  candidate_type: {_q(t.get('candidate'))}")
            body.append(f"  why:            [{reason}] {REASON_EXPLANATIONS.get(reason, '')}")
            body.append(f"  fix:            {SUGGESTED_FIXES.get(reason, '')}")
    section("Type mismatches", len(type_mm), body)

    # --- Content mismatches (grouped by record) ---
    body = []
    for (gid, cid), group in _group_by_record(content_diffs).items():
        body.append(_subheading(gid, cid))
        for d in group:
            reason = d.get("reason", "R-CELL-DIFFERS")
            body.append(f"[gold {d.get('gold_id')} / {d.get('type')} / {d.get('field')}]")
            body.append(f"  gold:        {_q(_truncate(d.get('gold')))}")
            body.append(f"  candidate:   {_q(_truncate(d.get('candidate')))}")
            body.append(f"  why:         [{reason}] {REASON_EXPLANATIONS.get(reason, '')}")
            body.append(f"  fix:         {SUGGESTED_FIXES.get(reason, '')}")
    section("Content mismatches", len(content_diffs), body)

    # --- Structure mismatches (grouped by record) ---
    body = []
    for (gid, cid), group in _group_by_record(structure_diffs).items():
        body.append(_subheading(gid, cid))
        for sd in group:
            reason = sd.get("reason", "R-PARENT-DIFFERS")
            body.append(f"[gold {sd.get('gold_id')} / candidate {sd.get('candidate_id')}]")
            body.append(f"  gold parent:      {_q(sd.get('gold_parent'))}")
            body.append(f"  candidate parent: {_q(sd.get('candidate_parent'))}")
            body.append(f"  why:              [{reason}] {REASON_EXPLANATIONS.get(reason, '')}")
            body.append(f"  fix:              {SUGGESTED_FIXES.get(reason, '')}")
    section("Structure mismatches", len(structure_diffs), body)

    # --- Summary of root causes ---
    lines.append("--- Summary of root causes --------------------------------------------")
    bucket: dict = {}
    for d in content_diffs + structure_diffs + type_mm:
        r = d.get("reason", "?")
        bucket[r] = bucket.get(r, 0) + 1
    for _ in missing:
        r = _.get("reason", "R-NODE-UNALIGNED")
        bucket[r] = bucket.get(r, 0) + 1
    for _ in extra:
        r = _.get("reason", "R-NODE-EXTRA")
        bucket[r] = bucket.get(r, 0) + 1
    for _ in missing_gold_recs:
        bucket["R-RECORD-MISSING"] = bucket.get("R-RECORD-MISSING", 0) + 1
    for _ in extra_cand_recs:
        bucket["R-RECORD-EXTRA"] = bucket.get("R-RECORD-EXTRA", 0) + 1
    if not bucket:
        lines.append("- No mismatches; the candidate is exactly equal to the gold.")
    else:
        for reason, count in sorted(bucket.items(), key=lambda kv: -kv[1]):
            lines.append(
                f"- {count} of total mismatches: [{reason}] "
                f"{REASON_EXPLANATIONS.get(reason, '')}"
            )
    lines.append("========================================================================")
    lines.append("")
    return "\n".join(lines)


def render_to_file(eval_json: dict, out_path: str | Path, threshold: float | None = None):
    txt = render(eval_json, threshold=threshold)
    Path(out_path).write_text(txt, encoding="utf-8")
    return out_path
