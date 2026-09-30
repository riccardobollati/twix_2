"""Select source pages for template learning and independent evaluation.

The exploratory table and KV clusterers are deliberately reused here: the
pipeline must make the same deterministic grouping shown in viewing 2.
"""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import sys
import time
from typing import Any

import pdfplumber


MAX_SCAN_PAGES = 500
FIRST_TEMPLATE_PAGES = 3


def _clusterers():
    source_dir = Path(__file__).resolve().parents[2] / "test-clustering"
    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))
    import cluster_pages_v2  # type: ignore[import-not-found]
    import cluster_kv  # type: ignore[import-not-found]
    return cluster_pages_v2, cluster_kv


def _memberships(table_report: dict, kv_report: dict) -> tuple[dict[int, set[str]], dict[str, int]]:
    by_page: dict[int, set[str]] = defaultdict(set)
    representatives: dict[str, int] = {}
    for kind, report, method in (
        ("table", table_report, "balanced_complete"),
        ("kv", kv_report, "kv_balanced"),
    ):
        outcome = report["algorithms"][method]
        for cluster_id, pages in outcome["clusters"].items():
            key = f"{kind}:{cluster_id}"
            representatives[key] = int(outcome["representatives"][cluster_id])
            for page in pages:
                by_page[int(page)].add(key)
    return dict(by_page), representatives


def _cover_clusters(candidates: list[int], memberships: dict[int, set[str]],
                    representatives: dict[str, int], selected: set[int],
                    limit: int) -> tuple[set[int], dict[str, int]]:
    """Cover both independent partitions, reusing pages that show both kinds."""
    universe = set(representatives)
    covered = set().union(*(memberships.get(page, set()) for page in selected)) if selected else set()
    assigned = {key: min(page for page in selected if key in memberships.get(page, set()))
                for key in covered}
    while uncovered := universe - covered:
        available = [page for page in candidates if page not in selected and memberships.get(page, set()) & uncovered]
        if not available or len(selected) >= limit:
            raise RuntimeError(
                "Cannot represent every table and KV cluster while keeping an independent "
                "evaluation page; use more structured pages or disable evaluation. "
                f"Uncovered: {', '.join(sorted(uncovered))}"
            )
        page = min(available, key=lambda p: (
            -len(memberships[p] & uncovered),
            -sum(representatives[key] == p for key in memberships[p] & uncovered),
            p,
        ))
        selected.add(page)
        for key in memberships[page] & uncovered:
            assigned[key] = page
        covered.update(memberships[page])
    return selected, assigned


def _spread_pages(candidates: list[int], count: int) -> list[int]:
    """Pick deterministic quantiles instead of the next contiguous run."""
    if count >= len(candidates):
        return candidates
    if count <= 0:
        return []
    return [candidates[min(len(candidates) - 1, int((i + 0.5) * len(candidates) / count))]
            for i in range(count)]


def plan_pages(table_report: dict, kv_report: dict, *, source_start: int,
               total_pages: int, requested_train_pages: int, eval_mode: bool,
               eval_pages: int | None) -> dict[str, Any]:
    """Make disjoint physical-page lists; every detected cluster reaches train."""
    if not 1 <= source_start <= total_pages:
        raise ValueError("structured-data start is outside the source PDF")
    usable = list(range(source_start, total_pages + 1))
    if eval_mode and len(usable) < 2:
        raise RuntimeError("eval_mode requires at least two structured-data pages")
    memberships, representatives = _memberships(table_report, kv_report)
    capacity = len(usable) - int(eval_mode)
    first = usable[:min(FIRST_TEMPLATE_PAGES, capacity)]
    selected = set(first)
    selected, assigned = _cover_clusters(usable, memberships, representatives, selected, capacity)
    cluster_samples = sorted(selected - set(first))

    target = min(capacity, max(requested_train_pages, len(selected)))
    for page in usable:
        if len(selected) >= target:
            break
        selected.add(page)
    train = sorted(selected)

    test: list[int] = []
    if eval_mode:
        limit = min(eval_pages or 20, len(usable) - len(train))
        available = [page for page in usable if page not in selected]
        # Prefer distinct table/KV structures in the evaluation set as well.
        remaining_keys = set(representatives)
        while len(test) < limit:
            candidates = [page for page in available if page not in test
                          and memberships.get(page, set()) & remaining_keys]
            if not candidates:
                break
            page = min(candidates, key=lambda p: (
                -len(memberships[p] & remaining_keys),
                -sum(representatives[key] == p for key in memberships[p] & remaining_keys), p,
            ))
            test.append(page)
            remaining_keys.difference_update(memberships[page])
        extra = [page for page in available if page not in test]
        test.extend(_spread_pages(extra, limit - len(test)))
        test.sort()

    assert not set(train) & set(test)
    assert set(assigned) == set(representatives)
    return {
        "source_start_page": source_start,
        "source_total_pages": total_pages,
        "scanned_pages": table_report["page_count"],
        "first_source_pages": first,
        "cluster_sample_pages": cluster_samples,
        "train_source_pages": train,
        "test_source_pages": test,
        "cluster_coverage": assigned,
        "cluster_representatives": representatives,
        "table_cluster_count": table_report["algorithms"]["balanced_complete"]["cluster_count"],
        "kv_cluster_count": kv_report["algorithms"]["kv_balanced"]["cluster_count"],
    }


def cluster_sampling_node(state: dict[str, Any]) -> dict[str, Any]:
    """Scan structured pages, cluster tables/KV independently, and plan splits."""
    started = time.time()
    table_engine, kv_engine = _clusterers()
    pdf_path = Path(state["full_pdf_path"]).resolve()
    output_dir = Path(state.get("output_dir") or pdf_path.parent)
    start = int(state.get("detected_start_page", 1))
    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
    if not 1 <= start <= total_pages:
        raise ValueError(f"start page {start} is outside the PDF's 1–{total_pages} pages")
    numbers = list(range(start, min(total_pages, start + MAX_SCAN_PAGES - 1) + 1))
    print(f"cluster_sampling: scanning {len(numbers)} source pages from page {start}", flush=True)
    scan = table_engine.base.scan_report(pdf_path, numbers, None)

    artifact_root = output_dir / "artifacts"
    table = table_engine.process(pdf_path, MAX_SCAN_PAGES, artifact_root, None, 500,
                                 "clustering", start_page=start, scan_report=scan)
    kv = kv_engine.process(pdf_path, MAX_SCAN_PAGES, artifact_root, None, 500,
                           "clustering", start_page=start, scan_report=scan)
    selection = plan_pages(
        table, kv, source_start=start, total_pages=total_pages,
        requested_train_pages=int(state["train_pages"]),
        eval_mode=bool(state["eval_mode"]), eval_pages=state.get("eval_pages"),
    )
    destination = artifact_root / "clustering" / "selection.json"
    destination.write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")
    print(
        f"cluster_sampling: {selection['table_cluster_count']} table / "
        f"{selection['kv_cluster_count']} KV clusters; train source pages "
        f"{selection['train_source_pages']}; evaluation pages {selection['test_source_pages']}",
        flush=True,
    )
    return {
        "train_source_pages": selection["train_source_pages"],
        "test_source_pages": selection["test_source_pages"],
        "first_source_pages": selection["first_source_pages"],
        "cluster_sample_pages": selection["cluster_sample_pages"],
        "cluster_coverage": selection["cluster_coverage"],
        "table_cluster_count": selection["table_cluster_count"],
        "kv_cluster_count": selection["kv_cluster_count"],
        "cost": [{"phase": "cluster_sampling", "duration_s": round(time.time() - started, 3)}],
    }
