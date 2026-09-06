from __future__ import annotations

import json
import math
from pathlib import Path

from PIL import Image
from reportlab.lib.colors import HexColor, Color
from reportlab.lib.pagesizes import A4, landscape
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas


ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data" / "id_05"
OUT = ROOT / "project-overview.pdf"

PAGE_W, PAGE_H = landscape(A4)
M = 42

BG = HexColor("#F6F7F2")
INK = HexColor("#17213A")
MUTED = HexColor("#667085")
BLUE = HexColor("#3559E0")
PURPLE = HexColor("#7C4DFF")
TEAL = HexColor("#169C8C")
ORANGE = HexColor("#E97B38")
GREEN = HexColor("#2D8A59")
RED = HexColor("#CC4B5E")
LINE = HexColor("#D8DDE7")
WHITE = HexColor("#FFFFFF")
PALE_BLUE = HexColor("#E8EEFF")
PALE_GREEN = HexColor("#E7F5EE")
PALE_ORANGE = HexColor("#FFF0E3")
PALE_PURPLE = HexColor("#F0EBFF")


def load_json(rel: str):
    return json.loads((DATA / rel).read_text(encoding="utf-8"))


META = load_json("metadata.json")
TEMPLATE = load_json("template.json")
TEST_EVAL = load_json("pipeline/eval.json")
TRAIN_EVAL = load_json("pipeline/train_eval/eval.json")
TEST_HEALTH = load_json("pipeline/pipeline_health.json")


def hex_alpha(color: HexColor, alpha: float) -> Color:
    return Color(color.red, color.green, color.blue, alpha=alpha)


def rounded(c, x, y, w, h, fill=WHITE, stroke=LINE, radius=10, sw=0.7):
    c.setLineWidth(sw)
    c.setStrokeColor(stroke)
    c.setFillColor(fill)
    c.roundRect(x, y, w, h, radius, fill=1, stroke=1)


def wrap_text(text: str, font: str, size: float, max_width: float) -> list[str]:
    paragraphs = text.split("\n")
    result: list[str] = []
    for paragraph in paragraphs:
        if not paragraph:
            result.append("")
            continue
        words = paragraph.split()
        line = words[0]
        for word in words[1:]:
            trial = f"{line} {word}"
            if stringWidth(trial, font, size) <= max_width:
                line = trial
            else:
                result.append(line)
                line = word
        result.append(line)
    return result


def paragraph(c, text, x, y, width, size=10, leading=14, color=INK,
              font="Helvetica", max_lines=None):
    c.setFillColor(color)
    c.setFont(font, size)
    lines = wrap_text(text, font, size, width)
    if max_lines:
        lines = lines[:max_lines]
    for line in lines:
        c.drawString(x, y, line)
        y -= leading
    return y


def bullet_list(c, items, x, y, width, size=9.5, leading=13.5, color=INK,
                bullet_color=BLUE, gap=6):
    for item in items:
        lines = wrap_text(item, "Helvetica", size, width - 18)
        c.setFillColor(bullet_color)
        c.circle(x + 3, y + 3, 2.2, fill=1, stroke=0)
        c.setFillColor(color)
        c.setFont("Helvetica", size)
        for i, line in enumerate(lines):
            c.drawString(x + 14, y, line)
            y -= leading
        y -= gap
    return y


def label(c, text, x, y, color=BLUE):
    c.setFillColor(color)
    c.setFont("Helvetica-Bold", 8)
    c.drawString(x, y, text.upper())


def title(c, text, x=M, y=PAGE_H - 76, size=25, width=None):
    c.setFillColor(INK)
    c.setFont("Helvetica-Bold", size)
    if width is None:
        c.drawString(x, y, text)
        return y - size * 1.25
    lines = wrap_text(text, "Helvetica-Bold", size, width)
    for line in lines:
        c.drawString(x, y, line)
        y -= size * 1.18
    return y


def page_base(c, page_no, section):
    c.setFillColor(BG)
    c.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)
    c.setStrokeColor(LINE)
    c.setLineWidth(0.6)
    c.line(M, 28, PAGE_W - M, 28)
    c.setFont("Helvetica", 7.5)
    c.setFillColor(MUTED)
    c.drawString(M, 16, "TWIX 2  /  PROJECT OVERVIEW")
    c.drawCentredString(PAGE_W / 2, 16, section.upper())
    c.drawRightString(PAGE_W - M, 16, f"{page_no:02d}")


def section_header(c, page_no, kicker, heading, sub=None):
    page_base(c, page_no, kicker)
    label(c, kicker, M, PAGE_H - 45)
    y = title(c, heading, y=PAGE_H - 76, size=24)
    if sub:
        paragraph(c, sub, M, y - 2, PAGE_W - 2 * M, size=10.5, leading=14, color=MUTED)


def metric_card(c, x, y, w, h, value, name, detail="", color=BLUE):
    rounded(c, x, y, w, h, fill=WHITE, stroke=LINE)
    c.setFillColor(color)
    c.setFont("Helvetica-Bold", 25)
    c.drawString(x + 15, y + h - 32, value)
    c.setFillColor(INK)
    c.setFont("Helvetica-Bold", 8.5)
    c.drawString(x + 15, y + h - 49, name.upper())
    if detail:
        paragraph(c, detail, x + 15, y + h - 65, w - 30, size=7.5,
                  leading=10, color=MUTED, max_lines=2)


def draw_image_contain(c, path: Path, x, y, w, h, pad=0, bg=WHITE, stroke=LINE):
    rounded(c, x, y, w, h, fill=bg, stroke=stroke, radius=8)
    with Image.open(path) as im:
        iw, ih = im.size
    scale = min((w - 2 * pad) / iw, (h - 2 * pad) / ih)
    dw, dh = iw * scale, ih * scale
    c.drawImage(str(path), x + (w - dw) / 2, y + (h - dh) / 2,
                width=dw, height=dh, preserveAspectRatio=True, mask="auto")


def draw_arrow(c, x1, y1, x2, y2, color=BLUE, width=2):
    c.setStrokeColor(color)
    c.setFillColor(color)
    c.setLineWidth(width)
    c.line(x1, y1, x2, y2)
    angle = math.atan2(y2 - y1, x2 - x1)
    ah = 7
    for offset in (2.55, -2.55):
        c.line(x2, y2, x2 + ah * math.cos(angle + offset), y2 + ah * math.sin(angle + offset))


def process_box(c, x, y, w, h, index, heading, detail, fill=WHITE, color=BLUE):
    rounded(c, x, y, w, h, fill=fill, stroke=hex_alpha(color, 0.55), radius=9)
    c.setFillColor(color)
    c.circle(x + 18, y + h - 18, 10, fill=1, stroke=0)
    c.setFillColor(WHITE)
    c.setFont("Helvetica-Bold", 8)
    c.drawCentredString(x + 18, y + h - 21, str(index))
    c.setFillColor(INK)
    c.setFont("Helvetica-Bold", 9.5)
    c.drawString(x + 34, y + h - 21, heading)
    paragraph(c, detail, x + 12, y + h - 40, w - 24, size=7.7, leading=10,
              color=MUTED, max_lines=4)


def code_block(c, text, x, y, w, h, font_size=7.4):
    rounded(c, x, y, w, h, fill=HexColor("#111827"), stroke=HexColor("#111827"), radius=7)
    c.setFont("Courier", font_size)
    c.setFillColor(HexColor("#DDE6F3"))
    yy = y + h - 16
    for line in text.splitlines():
        if yy < y + 8:
            break
        c.drawString(x + 12, yy, line)
        yy -= font_size + 3


def page_1(c):
    page_base(c, 1, "Preprint-style project brief")
    c.setFillColor(BLUE)
    c.rect(0, 0, 14, PAGE_H, fill=1, stroke=0)
    label(c, "Thesis research system", M, PAGE_H - 50)
    c.setFillColor(INK)
    c.setFont("Helvetica-Bold", 38)
    c.drawString(M, PAGE_H - 104, "Compile once.")
    c.setFillColor(BLUE)
    c.drawString(M, PAGE_H - 148, "Extract indefinitely.")
    paragraph(c,
        "TWIX 2 learns a reusable structured-data extractor from a small sample of a templatized PDF family, then ships ordinary Python code that processes unseen pages without further model calls.",
        M, PAGE_H - 184, 455, size=13, leading=18, color=INK)

    rounded(c, 545, 250, 245, 252, fill=HexColor("#17213A"), stroke=HexColor("#17213A"), radius=14)
    c.setFillColor(HexColor("#A9B9FF"))
    c.setFont("Helvetica-Bold", 8)
    c.drawString(565, 476, "ID_05 CASE STUDY")
    c.setFillColor(WHITE)
    c.setFont("Helvetica-Bold", 52)
    c.drawString(565, 406, "95.1%")
    c.setFont("Helvetica-Bold", 10)
    c.drawString(568, 384, "HELD-OUT ACCURACY")
    c.setFillColor(HexColor("#B9C1D5"))
    paragraph(c, "15 test pages  /  56 records\n227 physical blocks  /  0 model calls at runtime",
              568, 355, 192, size=9.5, leading=16, color=HexColor("#D3D8E5"))
    c.setStrokeColor(HexColor("#3A4663"))
    c.line(568, 315, 767, 315)
    c.setFont("Helvetica", 8.5)
    c.setFillColor(HexColor("#AEB7CA"))
    c.drawString(568, 294, "Train accuracy")
    c.drawRightString(767, 294, "97.5%")
    c.drawString(568, 274, "Record / node F1")
    c.drawRightString(767, 274, "1.000 / 1.000")

    metric_card(c, M, 77, 142, 83, "20", "source pages", "5 train + 15 independent test", BLUE)
    metric_card(c, M + 154, 77, 142, 83, "4", "learned node types", "3 tables + 1 key-value block", PURPLE)
    metric_card(c, M + 308, 77, 142, 83, "0", "runtime model calls", "marginal LLM/API cost after compile", TEAL)
    c.setFont("Helvetica", 8)
    c.setFillColor(MUTED)
    c.drawString(M, 50, "Project overview  /  1 September 2026  /  Evidence drawn from data/id_05")


def page_2(c):
    section_header(c, 2, "01 / Research premise", "Why decompose document extraction?",
                   "A single frontier model can read a page. TWIX 2 asks whether a learned program can repeat that work more economically across an entire document family.")
    x0, y0 = M, 300
    w, h = 355, 175
    rounded(c, x0, y0, w, h, fill=WHITE, stroke=LINE)
    label(c, "Conventional model-per-page approach", x0 + 18, y0 + h - 26, RED)
    c.setStrokeColor(LINE)
    c.line(x0 + 42, y0 + 34, x0 + 42, y0 + 133)
    c.line(x0 + 42, y0 + 34, x0 + w - 24, y0 + 34)
    c.setStrokeColor(RED)
    c.setLineWidth(3)
    c.line(x0 + 48, y0 + 43, x0 + w - 38, y0 + 125)
    c.setFillColor(RED)
    c.circle(x0 + w - 38, y0 + 125, 4, fill=1, stroke=0)
    c.setFont("Helvetica", 7.5)
    c.setFillColor(MUTED)
    c.drawString(x0 + 48, y0 + 18, "pages processed")
    c.saveState(); c.translate(x0 + 13, y0 + 58); c.rotate(90); c.drawString(0, 0, "LLM/API cost"); c.restoreState()
    c.setFont("Helvetica-Bold", 8)
    c.setFillColor(RED)
    c.drawString(x0 + 155, y0 + 99, "cost grows with every page")

    x1 = 434
    rounded(c, x1, y0, w, h, fill=WHITE, stroke=LINE)
    label(c, "TWIX 2 compile-once approach", x1 + 18, y0 + h - 26, GREEN)
    c.setStrokeColor(LINE)
    c.line(x1 + 42, y0 + 34, x1 + 42, y0 + 133)
    c.line(x1 + 42, y0 + 34, x1 + w - 24, y0 + 34)
    c.setStrokeColor(GREEN)
    c.setLineWidth(3)
    c.line(x1 + 48, y0 + 43, x1 + 83, y0 + 122)
    c.line(x1 + 83, y0 + 122, x1 + w - 38, y0 + 122)
    c.setFillColor(GREEN)
    c.circle(x1 + 83, y0 + 122, 4, fill=1, stroke=0)
    c.setFont("Helvetica", 7.5)
    c.setFillColor(MUTED)
    c.drawString(x1 + 48, y0 + 18, "pages processed")
    c.saveState(); c.translate(x1 + 13, y0 + 58); c.rotate(90); c.drawString(0, 0, "LLM/API cost"); c.restoreState()
    c.setFont("Helvetica-Bold", 8)
    c.setFillColor(GREEN)
    c.drawString(x1 + 118, y0 + 113, "no further model calls")

    rounded(c, M, 66, PAGE_W - 2 * M, 216, fill=PALE_BLUE, stroke=HexColor("#BAC8FF"), radius=12)
    label(c, "Core hypothesis", M + 20, 255)
    c.setFillColor(INK)
    c.setFont("Helvetica-Bold", 18)
    c.drawString(M + 20, 222, "Structure converts a broad reasoning problem into small, repeatable programs.")
    bullet_list(c, [
        "Infer the recurring schema, page frame, block boundaries and record order once from a small representative sample.",
        "Route difficult code-generation tasks to stronger models and simpler tasks to cheaper or potentially locally hosted models.",
        "Package the learned logic as standard-library Python plus pdfplumber, so operational extraction is deterministic and model-free.",
        "Use a frontier reference model such as GPT-5.4 during evaluation, while keeping it outside the deployed extraction path."
    ], M + 20, 190, PAGE_W - 2 * M - 40, size=9.4, leading=13, bullet_color=BLUE)
    c.setFont("Helvetica-Oblique", 8)
    c.setFillColor(MUTED)
    c.drawString(M + 20, 82, "“Free” here means zero marginal LLM/API spend after compilation; CPU time, storage and hosting are still real resources.")


def page_3(c):
    section_header(c, 3, "02 / Architecture", "The pipeline at a glance",
                   "The upper path is an agentic compilation process. The lower path is the portable runtime used on unseen PDFs.")
    rounded(c, M, 295, PAGE_W - 2 * M, 178, fill=WHITE, stroke=LINE, radius=12)
    label(c, "One-time learning / compilation", M + 18, 452, PURPLE)
    boxes = [
        ("Detect", "Find first structured page"),
        ("Split + OCR", "Train, holdout and test"),
        ("Infer template", "Nodes, fields, metadata"),
        ("Write separator", "Page-local physical blocks"),
        ("Route + write", "One extractor per node"),
        ("Evaluate / repair", "Ground truth and gate")
    ]
    bw, gap = 111, 12
    start = M + 18
    for i, (head, detail) in enumerate(boxes):
        xx = start + i * (bw + gap)
        process_box(c, xx, 345, bw, 94, i + 1, head, detail,
                    fill=PALE_PURPLE if i in (2, 3, 4) else WHITE,
                    color=PURPLE if i in (2, 3, 4) else BLUE)
        if i < len(boxes) - 1:
            draw_arrow(c, xx + bw + 2, 392, xx + bw + gap - 2, 392, color=MUTED, width=1)

    rounded(c, M, 71, PAGE_W - 2 * M, 197, fill=HexColor("#17213A"), stroke=HexColor("#17213A"), radius=12)
    label(c, "Portable runtime / repeated use", M + 18, 242, HexColor("#74E2D2"))
    run_boxes = [
        ("New PDF", "same document family"),
        ("pdfplumber OCR", "words + geometry"),
        ("block_separation.py", "uids + block ids"),
        ("extraction_scripts/*.py", "relative blocks to JSON"),
        ("record_grouping.py", "logical records"),
        ("grouped.json", "portable output")
    ]
    for i, (head, detail) in enumerate(run_boxes):
        xx = start + i * (bw + gap)
        c.setStrokeColor(HexColor("#46516B"))
        c.setFillColor(HexColor("#202C48"))
        c.roundRect(xx, 111, bw, 93, 9, fill=1, stroke=1)
        c.setFillColor(HexColor("#74E2D2") if i not in (0, 5) else HexColor("#A9B9FF"))
        c.setFont("Helvetica-Bold", 9)
        c.drawCentredString(xx + bw / 2, 169, head)
        c.setFillColor(HexColor("#BBC4D6"))
        lines = wrap_text(detail, "Helvetica", 7.5, bw - 20)
        yy = 149
        for line in lines[:3]:
            c.drawCentredString(xx + bw / 2, yy, line)
            yy -= 11
        if i < len(run_boxes) - 1:
            draw_arrow(c, xx + bw + 2, 157, xx + bw + gap - 2, 157, color=HexColor("#7E8AA5"), width=1)
    c.setFillColor(HexColor("#74E2D2"))
    c.setFont("Helvetica-Bold", 9)
    c.drawCentredString(PAGE_W / 2, 88, "NO LANGGRAPH  •  NO OPENCODE  •  NO MODEL CALLS")


def page_4(c):
    section_header(c, 4, "03 / Case study", "id_05: from a 20-page sample to a reusable extractor",
                   "The source is a landscape, template-based complaint report. Repeated records contain four visually distinct blocks.")
    img = ROOT / "tmp/pdfs/project-overview/source_pages/page_1.png"
    draw_image_contain(c, img, M, 95, 478, 373, pad=6)
    c.setFont("Helvetica", 7.5)
    c.setFillColor(MUTED)
    c.drawString(M + 8, 81, "Figure 1. Original training page; source redactions are preserved.")

    x = 546
    metric_card(c, x, 385, 116, 80, "7,270", "OCR words", "global ids across 20 pages", BLUE)
    metric_card(c, x + 128, 385, 116, 80, "5 / 15", "train / test", "independent split", PURPLE)
    metric_card(c, x, 291, 116, 80, "3 + 2", "build + holdout", "code sees 3; gate sees all 5", TEAL)
    metric_card(c, x + 128, 291, 116, 80, "19 / 56", "train / test records", "after deterministic grouping", ORANGE)
    rounded(c, x, 95, 244, 177, fill=WHITE, stroke=LINE)
    label(c, "Coordinate discipline", x + 16, 247)
    bullet_list(c, [
        "OCR words carry text, bounding boxes, a physical page and a globally unique id.",
        "After splitting, generated code operates in local page coordinates beginning at page 1.",
        "The build subset excludes two held-out train pages, limiting direct overfitting before train evaluation.",
        "The test split remains independent until the final extraction and scoring phase."
    ], x + 16, 224, 212, size=8.1, leading=11.2, gap=4)


def page_5(c):
    section_header(c, 5, "04 / Template inference", "A schema becomes the contract for every later stage",
                   "Template inference sees OCR layout and describes node types, fields, visual positions and recurring metadata.")
    nodes = TEMPLATE["template"]
    left_x, top = M, 444
    widths = [72, 76, 300]
    headers = ["node_id", "type", "fields"]
    row_h = 62
    table_w = sum(widths)
    rounded(c, left_x, 162, table_w, 310, fill=WHITE, stroke=LINE, radius=8)
    c.setFillColor(HexColor("#17213A"))
    c.roundRect(left_x, top, table_w, 28, 8, fill=1, stroke=0)
    xx = left_x
    c.setFillColor(WHITE); c.setFont("Helvetica-Bold", 7.5)
    for w, head in zip(widths, headers):
        c.drawString(xx + 8, top + 10, head.upper())
        xx += w
    y = top - row_h
    for idx, node in enumerate(nodes):
        if idx % 2 == 0:
            c.setFillColor(HexColor("#F8F9FC")); c.rect(left_x, y, table_w, row_h, fill=1, stroke=0)
        values = [str(node["node_id"]), node["type"], ", ".join(node["fields"])]
        xx = left_x
        for col, (w, val) in enumerate(zip(widths, values)):
            font = "Helvetica-Bold" if col < 2 else "Helvetica"
            size = 8 if col < 2 else 7.2
            c.setFont(font, size); c.setFillColor(INK)
            lines = wrap_text(val, font, size, w - 14)[:4]
            yy = y + row_h - 16
            for line in lines:
                c.drawString(xx + 8, yy, line); yy -= 10
            xx += w
        c.setStrokeColor(LINE); c.line(left_x, y, left_x + table_w, y)
        y -= row_h
    c.setFont("Helvetica", 7.5); c.setFillColor(MUTED)
    c.drawString(left_x + 8, 143, "Four nodes: three tables and one key-value section; fields remain unique within each node.")

    code = '''{
  "node_id": 2,
  "type": "table",
  "fields": [
    "Type Of Complaint",
    "Description",
    "Complaint Disposition"
  ],
  "bid": [2],
  "complexity": {
    "same_row_structure": true,
    "nested": false
  }
}'''
    code_block(c, code, 527, 234, 266, 238, font_size=7.2)
    rounded(c, 527, 96, 266, 120, fill=PALE_ORANGE, stroke=HexColor("#FFD2AE"))
    label(c, "Metadata filters", 543, 190, ORANGE)
    paragraph(c, "Eight document-level regexes identify the report title, agency, timestamp and page frame; two record-level patterns identify complaint and officer labels.",
              543, 166, 234, size=8.5, leading=12, color=INK)
    paragraph(c, "These patterns help remove repeated page furniture without discarding real rows near page edges.",
              543, 122, 234, size=8, leading=11, color=MUTED)


def page_6(c):
    section_header(c, 6, "05 / Block separation", "Generated geometry code turns each page into physical blocks",
                   "Every block stays on one page. Its id encodes the template node and page-local instance; uid remains unique across the runtime input.")
    img = ROOT / "tmp/pdfs/project-overview/block_overlays/page_1_blocks.png"
    draw_image_contain(c, img, M, 76, 525, 393, pad=5)
    c.setFont("Helvetica", 7.5); c.setFillColor(MUTED)
    c.drawString(M + 8, 62, "Figure 2. Regenerated from the saved train block_separation.json; page 1 contains 12 blocks.")
    x = 590
    rounded(c, x, 319, 203, 150, fill=WHITE, stroke=LINE)
    label(c, "Block identity", x + 15, 444)
    code_block(c, 'block_id = "<node_id>_<instance>"\n\n0_1  -> complaint summary table\n1_1  -> complainant key/value\n2_1  -> complaint rows\n3_1  -> officer rows\n\nuid = physical-block join key', x + 15, 337, 173, 89, font_size=6.8)
    rounded(c, x, 163, 203, 139, fill=PALE_BLUE, stroke=HexColor("#BAC8FF"))
    label(c, "id_05 evidence", x + 15, 277)
    bullet_list(c, [
        "Build subset: 44 blocks over 3 pages.",
        "Full train evaluation: 77 extracted blocks over 5 pages.",
        "Held-out test: 227 blocks over 15 pages.",
        "Health check: non-degenerate; no cross-page blocks."
    ], x + 15, 253, 173, size=8.1, leading=11, gap=4)
    rounded(c, x, 76, 203, 70, fill=PALE_GREEN, stroke=HexColor("#B7E3CC"))
    c.setFillColor(GREEN); c.setFont("Helvetica-Bold", 9)
    c.drawString(x + 15, 121, "Reusable boundary logic")
    paragraph(c, "Coordinates are inferred from the current page - not copied from sample values or page numbers.",
              x + 15, 104, 173, size=7.6, leading=10, color=INK)


def page_7(c):
    section_header(c, 7, "06 / Record separation", "Physical blocks are grouped into logical records",
                   "The current runtime traverses template node ids deterministically: a template restart or duplicate non-repeatable node opens a new record.")
    img = DATA / "artifacts/record_separation/page_1_records.png"
    draw_image_contain(c, img, M, 78, 525, 389, pad=5)
    c.setFont("Helvetica", 7.5); c.setFillColor(MUTED)
    c.drawString(M + 8, 64, "Figure 3. Saved id_05 record overlay. Color spans show complete multi-block records.")
    x = 590
    rounded(c, x, 316, 203, 151, fill=WHITE, stroke=LINE)
    label(c, "One record on page 1", x + 15, 442, GREEN)
    steps = [("0_0", "summary"), ("1_0", "complainant"), ("2_0", "complaints"), ("3_0", "officers")]
    yy = 411
    for i, (bid, desc) in enumerate(steps):
        c.setFillColor([RED, PURPLE, TEAL, ORANGE][i]); c.circle(x + 21, yy + 3, 5, fill=1, stroke=0)
        c.setFillColor(INK); c.setFont("Courier-Bold", 8.5); c.drawString(x + 34, yy, bid)
        c.setFont("Helvetica", 8.2); c.setFillColor(MUTED); c.drawString(x + 75, yy, desc)
        if i < 3:
            c.setStrokeColor(LINE); c.line(x + 21, yy - 7, x + 21, yy - 20)
        yy -= 27
    rounded(c, x, 166, 203, 133, fill=PALE_GREEN, stroke=HexColor("#B7E3CC"))
    label(c, "Why separation matters", x + 15, 274, GREEN)
    bullet_list(c, [
        "Tables may continue across pages without becoming a cross-page block.",
        "Repeated or multi-position nodes stay in the current record when the template permits.",
        "The test set produced exactly 56 candidate records and 56 reference records."
    ], x + 15, 250, 173, size=7.9, leading=10.8, gap=5, bullet_color=GREEN)
    rounded(c, x, 78, 203, 70, fill=PALE_ORANGE, stroke=HexColor("#FFD2AE"))
    paragraph(c, "Snapshot note: this archived run also contains a generated record_separation.py. The current project uses shared deterministic record_grouping.py instead.",
              x + 14, 127, 175, size=7.3, leading=9.5, color=INK)


def page_8(c):
    section_header(c, 8, "07 / Code synthesis", "Route by difficulty, then package the learned extractor",
                   "The hard model handles separation and difficult nodes; easier nodes can use a cheaper target. Each extractor receives a relative block whose top-left is (0, 0).")
    x = M
    cards = [
        ("Node 0", "table", "169 lines", "DeepSeek v4 Pro", PURPLE),
        ("Node 1", "key-value", "58 lines", "DeepSeek v4 Flash", TEAL),
        ("Node 2", "table", "140 lines", "DeepSeek v4 Pro", PURPLE),
        ("Node 3", "table", "165 lines", "DeepSeek v4 Pro", PURPLE),
    ]
    for i, (node, kind, lines, model, color) in enumerate(cards):
        xx = x + i * 188
        rounded(c, xx, 356, 176, 112, fill=WHITE, stroke=hex_alpha(color, .5))
        label(c, node, xx + 14, 444, color)
        c.setFillColor(INK); c.setFont("Helvetica-Bold", 14); c.drawString(xx + 14, 416, kind)
        c.setFillColor(MUTED); c.setFont("Helvetica", 8); c.drawString(xx + 14, 396, lines)
        c.setFont("Helvetica-Bold", 7.5); c.setFillColor(color); c.drawString(xx + 14, 376, model)

    rounded(c, M, 84, 356, 240, fill=HexColor("#17213A"), stroke=HexColor("#17213A"), radius=12)
    label(c, "Generated interface", M + 18, 298, HexColor("#74E2D2"))
    code_block(c, '''# extraction_scripts/<node_id>.py
def extract(block: dict) -> dict:
    """Convert one relative physical block to data."""
    ...

# Runtime dispatch
node_id = block["block_id"].rsplit("_", 1)[0]
result = extractor[node_id].extract(relative_block)
result["page"] = page
result["uid"] = uid''', M + 18, 109, 320, 166, font_size=7.2)

    rounded(c, 418, 84, 375, 240, fill=WHITE, stroke=LINE, radius=12)
    label(c, "Portable bundle", 436, 298, BLUE)
    items = [
        ("pipeline.py", "OCR, separation, dispatch, outputs, health"),
        ("block_separation.py", "374-line learned geometry program"),
        ("extraction_scripts/0.py ... 3.py", "one node-specific parser each"),
        ("record_grouping.py", "shared deterministic logical grouping"),
        ("requirements.txt", "pdfplumber-only runtime dependency"),
        ("template.json", "schema contract copied into current bundles")
    ]
    yy = 269
    for name, desc in items:
        c.setFont("Courier-Bold", 8); c.setFillColor(INK); c.drawString(436, yy, name)
        c.setFont("Helvetica", 7.7); c.setFillColor(MUTED); c.drawString(436, yy - 12, desc)
        yy -= 32


def page_9(c):
    section_header(c, 9, "08 / Runtime outputs", "From flat physical blocks to a usable JSON record",
                   "The runtime first emits page-local blocks, then groups them without merging away their physical provenance.")
    extracted = load_json("pipeline/extracted.json")[0]
    grouped = load_json("pipeline/grouped.json")[0]
    raw_excerpt = {
        "type": extracted["type"], "block_id": extracted["block_id"],
        "page": extracted["page"], "uid": extracted["uid"],
        "data": [dict(list(extracted["data"][0].items())[:4])]
    }
    group_excerpt = {
        "record_id": grouped["record_id"],
        "blocks": [
            {"block_id": b["block_id"], "type": b["type"], "page": b["page"]}
            for b in grouped["data"]
        ]
    }
    code_block(c, json.dumps(raw_excerpt, indent=2), M, 203, 350, 245, font_size=6.7)
    code_block(c, json.dumps(group_excerpt, indent=2), 449, 203, 344, 245, font_size=6.7)
    label(c, "extracted.json / first block (abridged)", M + 12, 460, BLUE)
    label(c, "grouped.json / first record (abridged)", 461, 460, TEAL)
    draw_arrow(c, 398, 328, 439, 328, color=TEAL, width=2)

    rounded(c, M, 79, PAGE_W - 2 * M, 108, fill=PALE_BLUE, stroke=HexColor("#BAC8FF"), radius=10)
    label(c, "Concrete extracted values", M + 18, 161)
    values = [
        ("Date", "7/19/2012"), ("Number", "08-14"),
        ("Investigator", "Kim, Joon"), ("Disposition", "UNFOUNDED"),
        ("Gender", "MALE"), ("Officer", "Carter, Felix."),
        ("Complaint", "R-3B.3 Courtesy: Conduct"), ("Action", "NONE")
    ]
    for i, (k, v) in enumerate(values):
        col, row = i % 4, i // 4
        xx = M + 18 + col * 187; yy = 133 - row * 34
        c.setFont("Helvetica-Bold", 7.4); c.setFillColor(MUTED); c.drawString(xx, yy + 10, k.upper())
        c.setFont("Helvetica-Bold", 8.4); c.setFillColor(INK); c.drawString(xx, yy - 2, v[:29])


def page_10(c):
    section_header(c, 10, "09 / Evaluation", "Independent test accuracy: 95.1%",
                   "The scorer aligns records and nodes, then measures normalized content. Missing and extra nodes receive zero content credit.")
    metric_card(c, M, 366, 164, 104, "97.5%", "train accuracy", "5 pages / 19 records", PURPLE)
    metric_card(c, M + 178, 366, 164, 104, "95.1%", "test accuracy", "15 pages / 56 records", BLUE)
    metric_card(c, M + 356, 366, 164, 104, "1.000", "record match F1", "perfect precision and recall", TEAL)
    metric_card(c, M + 534, 366, 164, 104, "1.000", "node match F1", "typed precision and recall", GREEN)

    rounded(c, M, 95, 366, 240, fill=WHITE, stroke=LINE)
    label(c, "Held-out score components", M + 18, 310)
    components = [
        ("RecordMatchF1", 1.0, TEAL),
        ("NodeMatchF1", 1.0, GREEN),
        ("Content overall", 0.902038, BLUE),
        ("Key-value content", 0.946429, PURPLE),
        ("Table content", 0.850106, ORANGE),
    ]
    yy = 276
    for name, val, color in components:
        c.setFont("Helvetica", 8.2); c.setFillColor(INK); c.drawString(M + 18, yy + 3, name)
        bx = M + 132; bw = 178
        c.setFillColor(HexColor("#E9ECF2")); c.roundRect(bx, yy, bw, 10, 5, fill=1, stroke=0)
        c.setFillColor(color); c.roundRect(bx, yy, bw * val, 10, 5, fill=1, stroke=0)
        c.setFont("Helvetica-Bold", 8); c.setFillColor(color); c.drawRightString(M + 344, yy + 2, f"{val:.3f}")
        yy -= 34
    c.setFont("Helvetica", 7.4); c.setFillColor(MUTED)
    paragraph(c, "For flat nodes, structure is not applicable; the headline score renormalizes to 50% typed NodeMatchF1 and 50% ContentScore.",
              M + 18, 122, 330, size=7.4, leading=10, color=MUTED)

    rounded(c, 433, 95, 360, 240, fill=PALE_ORANGE, stroke=HexColor("#FFD2AE"))
    label(c, "What remains imperfect", 451, 310, ORANGE)
    bullet_list(c, [
        "Complaint labels such as “Complaint #:1” are sometimes retained in the extracted complaint type.",
        "Wrapped text can spill across a table boundary: “SEIZURE” occasionally moves into Description.",
        "A few officer rows split incorrectly, creating extra fragments; long action values may spill into Body Cam.",
        "These are content-boundary errors, not missed records or missed node types."
    ], 451, 282, 322, size=8.3, leading=11.6, gap=7, bullet_color=ORANGE)
    c.setFont("Helvetica-Bold", 8); c.setFillColor(INK)
    c.drawString(451, 125, "Reference: manual held-out ground truth in this archived run.")
    c.setFont("Helvetica", 7.3); c.setFillColor(MUTED)
    c.drawString(451, 108, "Current default supports image + exact-OCR evaluation with GPT-5.4 via OpenRouter.")


def page_11(c):
    section_header(c, 11, "10 / Artifact trail", "What the compilation run creates",
                   "The artifact directory is both an audit trail and a deployable extractor. The entries below are observed in data/id_05.")
    sections = [
        ("ROOT / INPUT + AUDIT", [
            ("id_05.pdf", "20-page case-study source"),
            ("ocr_extraction_full.csv", "7,270 OCR words with geometry"),
            ("template.json", "4-node schema + 10 metadata patterns"),
            ("metadata.json", "configuration, models, timing, scores"),
            ("metadata/cost.json", "phase durations and token totals")
        ], BLUE),
        ("AGENT_SPACE / SPLITS", [
            ("train/id_05.pdf", "5-page train PDF"),
            ("train/ocr_extraction.csv", "full train OCR: 1,798 rows"),
            ("train/ocr_extraction_build.csv", "3-page build OCR: 1,021 rows"),
            ("train/block_separation.json", "44 learned build blocks"),
            ("train/record_separation.json", "11 build records"),
            ("train/block_samples/", "examples routed to extractor writers"),
            ("test/id_05.pdf + OCR", "15-page independent test: 5,472 rows")
        ], PURPLE),
        ("ARTIFACTS / VISUAL QA", [
            ("record_separation/page_*.png", "5 train record overlays"),
            ("block overlays", "regenerable from saved separator + JSON"),
            ("test overlays", "current runs also write held-out overlays")
        ], TEAL),
    ]
    x_positions = [M, 300, 558]
    widths = [238, 238, 235]
    for (head, items, color), x, w in zip(sections, x_positions, widths):
        rounded(c, x, 85, w, 378, fill=WHITE, stroke=hex_alpha(color, .45), radius=10)
        label(c, head, x + 16, 438, color)
        yy = 407
        for name, desc in items:
            c.setFont("Courier-Bold", 7.6); c.setFillColor(INK); c.drawString(x + 16, yy, name)
            lines = wrap_text(desc, "Helvetica", 7.3, w - 32)
            c.setFont("Helvetica", 7.3); c.setFillColor(MUTED)
            for line in lines[:2]:
                yy -= 11; c.drawString(x + 16, yy, line)
            yy -= 17
    c.setFont("Helvetica-Oblique", 7.4); c.setFillColor(MUTED)
    c.drawString(M, 63, "The id_05 snapshot predates some current artifact naming; the report distinguishes observed files from current-run behavior.")


def page_12(c):
    section_header(c, 12, "11 / Deployable bundle", "Files that perform and verify extraction",
                   "Everything under pipeline/ is either executable runtime code, an intermediate result, or evaluation evidence.")
    cols = [
        ("RUNTIME CODE", [
            ("pipeline.py", "381-line standalone runner"),
            ("block_separation.py", "learned page-to-block program"),
            ("record_separation.py", "legacy generated record separator"),
            ("extraction_scripts/0.py...3.py", "four learned node parsers"),
            ("requirements.txt", "portable dependency list")
        ], BLUE),
        ("TEST OUTPUT", [
            ("ocr.csv", "5,472 held-out OCR rows"),
            ("block_separation.json", "227 physical blocks"),
            ("record_separation.json", "56 record spans"),
            ("extracted.json", "227 extracted block results"),
            ("grouped.json", "56 logical records"),
            ("pipeline_health.json", "non-degenerate runtime health")
        ], TEAL),
        ("EVALUATION", [
            ("ground_truth.json", "56 manually referenced records"),
            ("eval.json", "machine-readable 95.1% score"),
            ("eval.txt", "actionable mismatch report"),
            ("train_eval/*", "train OCR, outputs, health, truth, score"),
            ("train_eval/eval_history.json", "iteration history")
        ], ORANGE),
    ]
    x_positions = [M, 300, 558]
    widths = [238, 238, 235]
    for (head, items, color), x, w in zip(cols, x_positions, widths):
        rounded(c, x, 141, w, 322, fill=WHITE, stroke=hex_alpha(color, .45), radius=10)
        label(c, head, x + 16, 438, color)
        yy = 405
        for name, desc in items:
            c.setFont("Courier-Bold", 7.2); c.setFillColor(INK); c.drawString(x + 16, yy, name)
            c.setFont("Helvetica", 7.2); c.setFillColor(MUTED)
            for line in wrap_text(desc, "Helvetica", 7.2, w - 32)[:2]:
                yy -= 10.5; c.drawString(x + 16, yy, line)
            yy -= 17
    rounded(c, M, 73, PAGE_W - 2 * M, 48, fill=HexColor("#17213A"), stroke=HexColor("#17213A"), radius=8)
    c.setFillColor(WHITE); c.setFont("Helvetica-Bold", 10)
    c.drawString(M + 18, 94, "Deployment contract:")
    c.setFont("Helvetica", 9); c.setFillColor(HexColor("#D6DCE9"))
    c.drawString(M + 128, 94, "copy the portable bundle beside a same-family PDF; run locally; receive grouped JSON without LangGraph or an LLM.")


def page_13(c):
    section_header(c, 13, "12 / Economics and scope", "What “run once” changes - and what it does not",
                   "TWIX 2 moves model expense from every production page into a one-time, auditable compilation step for one document family.")
    rounded(c, M, 309, 355, 166, fill=PALE_GREEN, stroke=HexColor("#B7E3CC"), radius=12)
    label(c, "After successful compilation", M + 18, 449, GREEN)
    c.setFillColor(GREEN); c.setFont("Helvetica-Bold", 28); c.drawString(M + 18, 402, "$0")
    c.setFillColor(INK); c.setFont("Helvetica-Bold", 11); c.drawString(M + 78, 409, "marginal LLM/API extraction cost")
    bullet_list(c, [
        "No model context window grows with the production document.",
        "No remote model call is required for each new page.",
        "The same code can process 10, 1,000 or more same-family pages."
    ], M + 18, 374, 320, size=8.5, leading=11.5, gap=4, bullet_color=GREEN)

    rounded(c, 438, 309, 355, 166, fill=PALE_PURPLE, stroke=HexColor("#D3C4FF"), radius=12)
    label(c, "id_05 one-time compilation", 456, 449, PURPLE)
    c.setFillColor(PURPLE); c.setFont("Helvetica-Bold", 28); c.drawString(456, 402, "38.2 min")
    c.setFillColor(INK); c.setFont("Helvetica-Bold", 11); c.drawString(580, 409, "recorded wall-clock duration")
    bullet_list(c, [
        "5.11M recorded input + output tokens across agentic phases.",
        "Flash handled the easy key-value extractor; Pro handled three tables.",
        "This compile cost is amortized over every later same-family page."
    ], 456, 374, 320, size=8.5, leading=11.5, gap=4, bullet_color=PURPLE)

    rounded(c, M, 91, PAGE_W - 2 * M, 184, fill=WHITE, stroke=LINE, radius=12)
    label(c, "Boundaries of the claim", M + 18, 248, ORANGE)
    bullet_list(c, [
        "Family-specific, not universal: a learned bundle targets documents that share the same layout and semantics.",
        "Model-generated ground truth can be imperfect. The id_05 held-out reference is manual; current workflows can use GPT-5.4 with page images and exact OCR.",
        "Layout drift, OCR errors, unseen nested structures and multi-page continuation patterns remain failure modes requiring evaluation and possible recompilation.",
        "Zero marginal model cost does not mean zero compute: local OCR and Python execution still consume CPU, memory, storage and operational effort.",
        "The local-model opportunity is a research direction: routing narrows each generation task, which may let smaller hosted or local models approach frontier-model extraction quality."
    ], M + 18, 218, PAGE_W - 2 * M - 36, size=8.6, leading=11.7, gap=5, bullet_color=ORANGE)


def page_14(c):
    page_base(c, 14, "Conclusion")
    label(c, "The project in one sentence", M, PAGE_H - 59)
    c.setFillColor(INK); c.setFont("Helvetica-Bold", 31)
    c.drawString(M, PAGE_H - 111, "Use models to write the extractor,")
    c.setFillColor(BLUE); c.drawString(M, PAGE_H - 151, "not to perform every extraction.")
    paragraph(c,
        "TWIX 2 treats frontier-model reasoning as a compilation resource. It observes a small sample, infers the recurring structure, writes and evaluates document-specific code, and exports a transparent runtime that can process new pages at zero marginal model cost.",
        M, PAGE_H - 191, 690, size=12.5, leading=18, color=INK)
    rounded(c, M, 188, PAGE_W - 2 * M, 170, fill=HexColor("#17213A"), stroke=HexColor("#17213A"), radius=14)
    c.setFillColor(HexColor("#A9B9FF")); c.setFont("Helvetica-Bold", 8)
    c.drawString(M + 22, 331, "ID_05 RESULT")
    statements = [
        ("5", "training pages"),
        ("15", "held-out test pages"),
        ("56", "correctly matched records"),
        ("95.1%", "overall accuracy"),
        ("0", "runtime model calls")
    ]
    for i, (value, desc) in enumerate(statements):
        xx = M + 22 + i * 143
        c.setFillColor(WHITE); c.setFont("Helvetica-Bold", 22); c.drawString(xx, 280, value)
        paragraph(c, desc, xx, 258, 113, size=8.2, leading=11, color=HexColor("#C5CCDA"), max_lines=2)
    c.setStrokeColor(HexColor("#3A4663")); c.line(M + 22, 232, PAGE_W - M - 22, 232)
    c.setFillColor(HexColor("#74E2D2")); c.setFont("Helvetica-Bold", 9)
    c.drawCentredString(PAGE_W / 2, 207, "A REUSABLE PROGRAM REPLACES A REPEATED MODEL CALL")
    paragraph(c,
        "This overview is a technical project brief rather than the full thesis manuscript. All quantitative claims and examples are taken from the archived id_05 run and its saved evaluation artifacts.",
        M, 143, 700, size=9.2, leading=13, color=MUTED)
    c.setFillColor(BLUE); c.setFont("Helvetica-Bold", 9)
    c.drawString(M, 90, "TWIX 2  /  structured extraction for templatized PDF families")


def build():
    c = canvas.Canvas(str(OUT), pagesize=(PAGE_W, PAGE_H), pageCompression=1)
    c.setTitle("TWIX 2 - Project Overview")
    c.setSubject("Compile-once structured data extraction for templatized PDF families")
    c.setAuthor("TWIX 2 thesis project")
    for fn in [page_1, page_2, page_3, page_4, page_5, page_6, page_7,
               page_8, page_9, page_10, page_11, page_12, page_13, page_14]:
        fn(c)
        c.showPage()
    c.save()
    print(OUT)


if __name__ == "__main__":
    build()
