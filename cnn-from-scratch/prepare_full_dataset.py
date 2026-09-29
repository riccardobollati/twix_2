"""Sample and render the expanded, document-disjoint CNN dataset."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent.parent / "full_dataset"
DATASET = ROOT / "dataset_full"
TRAIN_VAL = [
    "id_01", "id_04", "id_05", "id_07", "id_08", "id_18_upscaled",
    "id_27", "id_35", "id_37_upscaled", "id_39_upscaled",
    "id_40_upscaled", "id_47_upscaled", "id_57", "new_01", "new_02",
    "new_05", "new_06", "new_08", "new_09", "new_10",
]
TEST = [
    "id_03", "id_13", "id_24", "id_25", "id_36", "id_60",
    "new_03", "new_04", "new_07", "new_14",
]
# These are the only available source files for the requested IDs.
SOURCE_ALIASES = {
    "id_07": "id_07_upscaled", "id_08": "id_08_upscaled",
    "id_27": "id_27_upscaled", "id_25": "id_25_upscaled", "id_36": "id_36_upscaled",
}


def page_count(pdf: Path) -> int:
    result = subprocess.run(["pdfinfo", str(pdf)], check=True, capture_output=True, text=True)
    match = re.search(r"^Pages:\s*(\d+)", result.stdout, flags=re.MULTILINE)
    if not match:
        raise ValueError(f"Could not read page count: {pdf}")
    return int(match.group(1))


def evenly_spaced(count: int, n: int) -> list[int]:
    if count < n:
        raise ValueError(f"PDF has {count} pages, need {n}")
    # Includes early, middle, and late layouts; exactly n distinct physical pages.
    return [1 + (i * (count - 1)) // (n - 1) for i in range(n)]


def render(task: tuple[Path, int, Path, int, bool]) -> None:
    pdf, page, image, long_side, force = task
    if image.exists() and not force:
        return
    image.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        "pdftoppm", "-f", str(page), "-l", str(page),
        "-scale-to", str(long_side), "-singlefile", "-png",
        str(pdf), str(image.with_suffix("")),
    ], check=True, capture_output=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--long-side", type=int, default=1280)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.long_side < 512 or args.jobs < 1:
        parser.error("long-side must be >= 512 and jobs >= 1")

    entries = []
    tasks = []
    document_manifest = {}
    for document in TRAIN_VAL + TEST:
        source_name = SOURCE_ALIASES.get(document, document)
        pdf = SOURCE / f"{source_name}.pdf"
        if not pdf.is_file():
            raise FileNotFoundError(pdf)
        count = page_count(pdf)
        selected = evenly_spaced(count, 20 if document in TRAIN_VAL else 5)
        document_manifest[document] = {"source": str(pdf), "total_pages": count,
                                       "selected_pages": selected}
        for i, page in enumerate(selected):
            # Every training document contributes 15 train and 5 validation pages.
            # The validation positions are distributed across the PDF.
            split = "test" if document in TEST else ("val" if i in {2, 6, 10, 14, 18} else "train")
            stem = f"{document}_p{page:04d}"
            image = DATASET / split / "images" / f"{stem}.png"
            entries.append({
                "id": stem, "split": split, "document": document,
                "source_pdf": str(pdf), "page": page,
                "image": str(image.relative_to(ROOT)),
            })
            tasks.append((pdf, page, image, args.long_side, args.force))
            (DATASET / split / "annotations").mkdir(parents=True, exist_ok=True)
            (DATASET / split / "masks").mkdir(parents=True, exist_ok=True)

    DATASET.mkdir(exist_ok=True)
    (DATASET / "manifest.json").write_text(json.dumps({
        "label_map": {"background": 0, "table": 1, "key_value": 2},
        "sampling": "20 evenly spaced pages per train/validation document (15 train, 5 validation); 5 evenly spaced pages per test document",
        "omitted_requested_test_documents": ["id_44"],
        "documents": document_manifest, "pages": entries,
    }, indent=2) + "\n")
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for _ in pool.map(render, tasks):
            pass
    print(f"Rendered {len(entries)} pages from {len(document_manifest)} PDFs into {DATASET}")


if __name__ == "__main__":
    main()
