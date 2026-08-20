import csv
import json
import os
import re
from collections import defaultdict


def _tokenize(text: str) -> list[str]:
    return [t.strip().rstrip(":.,;") for t in text.strip().lower().split()]


def _field_tokens(field: str) -> list[str]:
    return _tokenize(field)


def _match_field_in_row(field: str, row_texts: list[str]) -> bool:
    tokens = _field_tokens(field)
    if not tokens:
        return False
    row_lower = [w.strip().rstrip(":.,;").lower() for w in row_texts]
    for i in range(len(row_lower) - len(tokens) + 1):
        if row_lower[i:i + len(tokens)] == tokens:
            return True
    return False


def _match_field_anywhere(field: str, row_texts: list[str]) -> bool:
    tokens = _field_tokens(field)
    if not tokens:
        return False
    row_lower = [w.strip().rstrip(":.,;").lower() for w in row_texts]
    ti = 0
    for rw in row_lower:
        if ti < len(tokens) and rw == tokens[ti]:
            ti += 1
    return ti == len(tokens)


def _rows_overlap(a: dict, b: dict) -> bool:
    return a["bottom"] >= b["top"] and b["bottom"] >= a["top"]


def _extract_designation(row_texts: list[str]) -> str | None:
    for i, t in enumerate(row_texts):
        if t.upper() == "LI" and i + 1 < len(row_texts):
            nxt = row_texts[i + 1]
            if re.match(r"^P\d+$", nxt):
                return f"LI-{nxt}"
    return None


def _find_next_designation(rows: list[list[dict]], start_idx: int) -> str | None:
    for i in range(start_idx, min(start_idx + 10, len(rows))):
        texts = [w["text"] for w in rows[i]]
        d = _extract_designation(texts)
        if d is not None:
            return d
    return None


def _clean_text(text: str) -> str:
    return text.strip().rstrip(":.,")


def _field_matches_for_row(row_texts: list[str], template: list[dict]) -> set:
    result = set()
    for node in template:
        nid = node["node_id"]
        for fi, field in enumerate(node["fields"]):
            if _match_field_in_row(field, row_texts) or _match_field_anywhere(field, row_texts):
                result.add((nid, fi))
    return result


def _visited_nodes(rec_rows: list[list[dict]], template: list[dict]) -> set:
    all_matched = set()
    for row in rec_rows:
        row_texts = [w["text"] for w in row]
        all_matched |= _field_matches_for_row(row_texts, template)
    visited = set()
    for node in template:
        nid = node["node_id"]
        f_count = len(node["fields"])
        threshold = max(1, int(f_count * 0.5 + 0.999))
        count = sum(1 for fi in range(f_count) if (nid, fi) in all_matched)
        if count >= threshold:
            visited.add(nid)
    return visited


def main(inputpath: str, out_path: str) -> None:
    words = []
    with open(inputpath, encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            words.append({
                "id": int(row["id"]),
                "page": int(row["page"]),
                "x0": float(row["x0"]),
                "top": float(row["top"]),
                "x1": float(row["x1"]),
                "bottom": float(row["bottom"]),
                "text": _clean_text(row["text"]),
            })

    words.sort(key=lambda w: (w["id"], w["page"], w["top"], w["x0"]))

    template = [
        {
            "node_id": 0,
            "fields": [
                "Identificazione - Nome Limiti laterali Identification \u2013 Name Lateral limits",
                "Limite sup. Limite inf. Upper limit Lower limit",
                "Tipo di attivit\u00e0 Type of activity",
                "Note Remarks",
                "Validit\u00e0 Validity",
            ],
        }
    ]

    total_nodes = len(template)

    rows = []
    for w in words:
        placed = False
        for r in reversed(rows):
            if r[0]["page"] != w["page"]:
                continue
            if any(_rows_overlap(w, ew) for ew in r):
                r.append(w)
                placed = True
                break
        if not placed:
            rows.append([w])

    records = []
    current_record_rows = []
    current_matched = set()
    current_anchor = None

    for row_idx, row in enumerate(rows):
        row_texts = [w["text"] for w in row]
        row_matched = _field_matches_for_row(row_texts, template)
        designation = _extract_designation(row_texts)
        page = row[0]["page"]

        if not current_record_rows:
            current_record_rows.append(row)
            current_matched = row_matched.copy()
            if designation is not None:
                current_anchor = designation
            continue

        if designation is not None:
            if designation != current_anchor and current_anchor is not None:
                records.append({
                    "rows": current_record_rows,
                    "anchor": current_anchor,
                })
                current_record_rows = [row]
                current_anchor = designation
                continue
            elif current_anchor is None:
                current_anchor = designation

        first_page = current_record_rows[0][0]["page"]

        if page != first_page and designation is None and not row_matched:
            records.append({
                "rows": current_record_rows,
                "anchor": current_anchor,
            })
            current_record_rows = [row]
            current_anchor = _find_next_designation(rows, row_idx + 1)
            current_matched = set()
            continue

        if row_matched and current_matched:
            overlap = row_matched & current_matched
            if overlap:
                next_desig = designation if designation is not None else _find_next_designation(rows, row_idx + 1)
                if next_desig is not None and next_desig != current_anchor:
                    records.append({
                        "rows": current_record_rows,
                        "anchor": current_anchor,
                    })
                    current_record_rows = [row]
                    current_anchor = next_desig
                    current_matched = row_matched.copy()
                    continue

        current_record_rows.append(row)
        if row_matched:
            current_matched |= row_matched

    if current_record_rows:
        records.append({
            "rows": current_record_rows,
            "anchor": current_anchor,
        })

    result = {"records": []}
    for rec_idx, rec_data in enumerate(records):
        rec_rows = rec_data["rows"]
        pages_dict = defaultdict(list)
        for row in rec_rows:
            for w in row:
                pages_dict[w["page"]].append(w)
        pages_list = []
        for p in sorted(pages_dict):
            page_words = sorted(pages_dict[p], key=lambda w: w["id"])
            y_start = min(w["top"] for w in page_words)
            y_end = max(w["bottom"] for w in page_words)
            pages_list.append({
                "page": p,
                "y_start": y_start,
                "y_end": y_end,
                "words": page_words,
            })
        pages_list.sort(key=lambda x: x["page"])
        result["records"].append({
            "record_id": str(rec_idx),
            "pages": pages_list,
        })

    os.makedirs(out_path, exist_ok=True)
    out_file = os.path.join(out_path, "record_separation.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    total_records = len(result["records"])
    total_li_entries = sum(
        1 for rec_data in records
        for row in rec_data["rows"]
        if _extract_designation([w["text"] for w in row]) is not None
    )
    print(f"Total records: {total_records}, distinct anchor values (approx): {total_li_entries}")
    for rec_idx, rec_data in enumerate(records):
        rec_rows = rec_data["rows"]
        pages_set = set()
        for row in rec_rows:
            for w in row:
                pages_set.add(w["page"])
        page_range = f"p{min(pages_set)}" if len(pages_set) == 1 else f"p{min(pages_set)}-p{max(pages_set)}"
        visited = _visited_nodes(rec_rows, template)
        visited_str = ",".join(str(v) for v in sorted(visited)) if visited else "none"
        print(f"  record {rec_idx}: {page_range}, {len(rec_rows)} rows, visited nodes: [{visited_str}]")


if __name__ == "__main__":
    import sys
    main(sys.argv[1], sys.argv[2])