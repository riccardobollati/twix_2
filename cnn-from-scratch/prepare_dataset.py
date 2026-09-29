"""Render the curated source PDF pages without touching existing annotations."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
SOURCES = {
    "train": ["id_05", "id_12", "id_49", "new_03", "new_07", "new_08"],
    "val": ["new_04", "new_11"],
    "test": ["id_46", "id_18_upscaled"],
}
PAGES = (3, 4)  # One-based PDF pages, deliberately skipping most cover pages.


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--long-side", type=int, default=1280)
    parser.add_argument("--force", action="store_true", help="Re-render existing PNGs")
    args = parser.parse_args()
    if args.long_side < 512:
        parser.error("--long-side must be at least 512")
    all_documents = [document for documents in SOURCES.values() for document in documents]
    if len(all_documents) != len(set(all_documents)):
        raise ValueError("A source PDF appears in more than one split")

    entries = []
    for split, documents in SOURCES.items():
        image_dir = ROOT / "dataset" / split / "images"
        image_dir.mkdir(parents=True, exist_ok=True)
        (ROOT / "dataset" / split / "annotations").mkdir(parents=True, exist_ok=True)
        (ROOT / "dataset" / split / "masks").mkdir(parents=True, exist_ok=True)
        for document in documents:
            pdfs = sorted((REPO / "data" / document).glob("*.pdf"))
            if len(pdfs) != 1:
                raise RuntimeError(f"Expected one top-level PDF for {document}, found {len(pdfs)}")
            pdf = pdfs[0]
            for page in PAGES:
                stem = f"{document}_p{page:04d}"
                png = image_dir / f"{stem}.png"
                if args.force or not png.exists():
                    command = [
                        "pdftoppm", "-f", str(page), "-l", str(page),
                        "-scale-to", str(args.long_side), "-singlefile",
                        "-png", str(pdf), str(png.with_suffix("")),
                    ]
                    subprocess.run(command, check=True, capture_output=True)
                entries.append({
                    "id": stem,
                    "split": split,
                    "document": document,
                    "source_pdf": str(pdf.relative_to(REPO)),
                    "page": page,
                    "image": str(png.relative_to(ROOT)),
                })
    manifest = ROOT / "dataset" / "manifest.json"
    manifest.write_text(json.dumps({"label_map": {"background": 0, "table": 1, "key_value": 2}, "pages": entries}, indent=2) + "\n")
    print(f"Rendered {len(entries)} pages; manifest: {manifest}")


if __name__ == "__main__":
    main()
