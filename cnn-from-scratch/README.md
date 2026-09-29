# Page region segmentation from scratch

This experiment predicts one class for every page pixel:

Annotations follow the repository's [table and KV definitions](LABELING_POLICY.md).

| ID | Class | Review color |
| --- | --- | --- |
| 0 | Background | Unchanged |
| 1 | Table | Green |
| 2 | Key–value block | Blue |

The model is a small, randomly initialized U-Net implemented in `model.py`
(1,964,131 parameters at the default width, about 8 MB of FP32 weights).
Its output colors entire regions, including the whitespace inside a region.
It does **not** read text, identify individual table cells, or link a particular
key to its value.

## Dataset

### Expanded dataset from `../full_dataset`

`prepare_full_dataset.py` samples 20 evenly spaced physical pages from each of
the 20 requested training documents: 15 train and 5 validation pages per PDF.
It samples 5 pages from each of the 10 available, distinct test PDFs. This is
300 train, 100 validation, and 50 test pages. `ID_44` is omitted because it is
absent from `../full_dataset`, as requested. `ID_07`, `ID_08`, `ID_25`,
`ID_27`, and `ID_36` use their available upscaled PDF files. The manifest
records the exact source file and physical page for every render.

The new data lives in `dataset_full/`; the original `dataset/` and checkpoint
remain intact. Annotations are agent-drafted and need visual review before
training. The split follows the requested document list, so train and
validation contain different pages of the *same* 20 PDFs; test PDFs are
document-disjoint. Validation thus measures within-document performance,
while test probes different documents.

```bash
cd cnn-from-scratch
python prepare_full_dataset.py             # rerender only if images are missing
python build_full_masks.py                 # validate JSON and generate masks/review
python train.py --dataset-root dataset_full --output-dir runs/full --epochs 60 --image-size 768 --batch-size 1
python review_test.py --dataset-root dataset_full --label-review-dir review_full --checkpoint runs/full/best.pt
```

For a 16×16 kernel in the first convolution, use:

```bash
python train.py --dataset-root dataset_full --output-dir runs/kernel16 --first-kernel 16 --image-size 768 --batch-size 1 --epochs 40 --patience 10
python review_test.py --dataset-root dataset_full --label-review-dir review_full --checkpoint runs/kernel16/best.pt
```

This first convolution also has stride 16. The model receives the full page;
later convolutional layers combine information across windows, and pixel shuffle
restores a full-resolution three-class output mask. The checkpoint stores the
kernel choice, so `review_test.py` and `predict.py` load the correct model.

Open [review_full/index.html](review_full/index.html) and select a document to
inspect original pages beside the class overlays. Edit any annotation JSON in
`dataset_full/<split>/annotations/` and rerun `build_full_masks.py`; the script
rejects missing, overlapping, out-of-bounds, or mismatched labels. Use
`--allow-missing` only to preview work while labeling is in progress.

### Original small dataset

`prepare_dataset.py` renders pages 3 and 4 from ten distinct top-level PDFs in
`../data/` at a 1280-pixel long side. The fixed, document-level split is:

| Split | Source documents | Pages |
| --- | --- | ---: |
| Train | `id_05`, `id_12`, `id_49`, `new_03`, `new_07`, `new_08` | 12 |
| Validation | `new_04`, `new_11` | 4 |
| Test | `id_46`, `id_18_upscaled` | 4 |

Each page has an image, an agent-drafted JSON annotation, and a generated
single-channel PNG mask in `dataset/<split>/`. The JSON boxes are in the original
rendered image's pixel coordinates, with exclusive right and bottom edges. The
mask is the training target and contains class IDs 0, 1, or 2, not RGB colors.

The labels are a **draft**. Inspect [review/index.html](review/index.html) before
using them as ground truth. It pairs each page with a green/blue overlay and
links to the corresponding JSON. In particular, form-like blocks with borders
must be classified by their row structure, not by the borders. Edit the JSON boxes or
class names as needed, then regenerate masks and previews:

```bash
python build_masks.py
```

`build_masks.py` rejects missing labels, mismatched image dimensions, boxes
outside the page, and overlapping boxes. It writes `review/summary.json`.
The original split contains only 20 pages, so validation/test scores will be
noisy and should not be read as a generalization claim.

## Setup and training

Python 3.10+ and PyTorch are required. `pdftoppm` is needed only to re-render
the source PDFs.

```bash
cd cnn-from-scratch
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python build_masks.py
python train.py --epochs 60 --image-size 768 --batch-size 1
```

The training script selects CUDA, Apple Metal (MPS), or CPU automatically.
It uses class-weighted cross entropy plus foreground Dice loss, chooses the
checkpoint with the best validation foreground IoU, and evaluates that
checkpoint on the test split once training is complete. Outputs go to
`runs/first/best.pt` and `runs/first/metrics.json`. For a quick smoke run, use
`--epochs 1 --image-size 256`.

To color an unseen page image after training:

```bash
python predict.py page.png --checkpoint runs/first/best.pt --output page.overlay.png
```

To review the held-out test pages beside their labels and predictions:

```bash
python review_test.py
```

Open `runs/first/test_review/index.html` for the full-size comparison or
`runs/first/test_review/overview.jpg` for a contact sheet. The script also
writes per-page IoU to `runs/first/test_review/metrics.json`.

For a visual generalization check on eight PDFs that do not appear in any
dataset split, run `python review_unseen.py` and open
`runs/first/unseen_review/index.html`. It renders two pages per PDF and checks
source paths against `dataset/manifest.json` before inference. These extra
pages are unlabeled, so no IoU is available for them.

The 768-pixel training input preserves page aspect ratio and pads it to a
square. A larger `--image-size` can preserve more small text and narrow gaps
but requires more memory. The model size stays the same when the image size
changes; activation memory and computation increase.
