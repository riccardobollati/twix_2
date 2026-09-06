import fs from "node:fs/promises";
import path from "node:path";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const ROOT = "/Users/riccardo/tesi/code/twix_2";
const RUN_REL = "UI/runs/run-20260901-184959-d47392";
const RUN_DIR = path.join(ROOT, RUN_REL);
const TMP = path.join(ROOT, "tmp/presentations/project-overview");
const OUT = path.join(ROOT, "project-overview.pptx");
const W = 1280;
const H = 720;

const C = {
  paper: "#f4f0e7",
  paper2: "#ebe5d8",
  card: "#fbf8f1",
  line: "#d9d0bf",
  ink: "#252b28",
  soft: "#5f6862",
  muted: "#8b928c",
  forest: "#284e40",
  forest2: "#3f705f",
  mint: "#dce8df",
  red: "#a94f43",
  amber: "#b57b32",
  dark: "#202824",
  code: "#e8eee9",
  white: "#ffffff",
};

const FONT_SERIF = "Georgia";
const FONT_SANS = "Arial";
const FONT_MONO = "Courier New";

async function writeBlob(filePath, blob) {
  await fs.writeFile(filePath, new Uint8Array(await blob.arrayBuffer()));
}

async function readBytes(filePath) {
  const bytes = await fs.readFile(filePath);
  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
}

function addRect(slide, x, y, w, h, fill = C.card, stroke = C.line, sw = 1, name = undefined) {
  return slide.shapes.add({
    geometry: "rect",
    name,
    position: { left: x, top: y, width: w, height: h },
    fill,
    line: { style: "solid", fill: stroke, width: sw },
  });
}

function addText(slide, text, x, y, w, h, opts = {}) {
  const shape = slide.shapes.add({
    geometry: "textbox",
    name: opts.name,
    position: { left: x, top: y, width: w, height: h },
    fill: "none",
    line: { style: "solid", fill: "none", width: 0 },
  });
  shape.text = text;
  shape.text.style = {
    fontFamily: opts.fontFamily || FONT_SANS,
    fontSize: opts.fontSize || 18,
    bold: opts.bold || false,
    italic: opts.italic || false,
    color: opts.color || C.ink,
    alignment: opts.alignment || "left",
    verticalAlignment: opts.verticalAlignment || "top",
  };
  return shape;
}

function addRule(slide, x, y, w, color = C.line, thickness = 1) {
  return addRect(slide, x, y, w, thickness, color, color, 0);
}

function addLine(slide, x1, y1, x2, y2, color = C.ink, width = 2) {
  return slide.shapes.add({
    geometry: "line",
    position: {
      left: Math.min(x1, x2),
      top: Math.min(y1, y2),
      width: Math.abs(x2 - x1),
      height: Math.abs(y2 - y1),
      horizontalFlip: x2 < x1,
      verticalFlip: y2 < y1,
    },
    fill: "none",
    line: { style: "solid", fill: color, width },
  });
}

function addArrow(slide, x, y, w, h, color = C.forest2) {
  return slide.shapes.add({
    geometry: "rightArrow",
    position: { left: x, top: y, width: w, height: h },
    fill: color,
    line: { style: "solid", fill: color, width: 0 },
  });
}

function addDot(slide, x, y, d, color = C.forest2) {
  return slide.shapes.add({
    geometry: "ellipse",
    position: { left: x, top: y, width: d, height: d },
    fill: color,
    line: { style: "solid", fill: color, width: 0 },
  });
}

function baseSlide(presentation, section, page) {
  const slide = presentation.slides.add();
  slide.background.fill = C.paper;
  addRect(slide, 0, 0, 18, H, C.forest, C.forest, 0, "left-rail");
  addText(slide, "TWIX 2", 54, 24, 150, 28, { fontFamily: FONT_SERIF, fontSize: 24, bold: true, color: C.ink });
  addText(slide, "EXTRACTION WORKBENCH", 990, 31, 220, 18, { fontFamily: FONT_MONO, fontSize: 11, color: C.muted, alignment: "right" });
  addRule(slide, 54, 674, 1156, C.line, 1);
  addText(slide, section.toUpperCase(), 54, 684, 420, 16, { fontFamily: FONT_MONO, fontSize: 10, color: C.muted });
  addText(slide, String(page).padStart(2, "0"), 1150, 684, 60, 16, { fontFamily: FONT_MONO, fontSize: 10, color: C.muted, alignment: "right" });
  return slide;
}

function slideTitle(slide, eyebrow, title, subtitle = "") {
  addText(slide, eyebrow.toUpperCase(), 54, 74, 450, 22, { fontFamily: FONT_MONO, fontSize: 12, color: C.muted });
  addText(slide, title, 54, 104, 1136, 58, { fontFamily: FONT_SERIF, fontSize: 38, bold: true, color: C.ink });
  if (subtitle) addText(slide, subtitle, 54, 165, 1110, 48, { fontFamily: FONT_SANS, fontSize: 17, color: C.soft });
}

function addMetric(slide, value, label, x, y, w, accent = C.forest2, detail = "") {
  addRule(slide, x, y, w, accent, 5);
  addText(slide, value, x, y + 18, w, 50, { fontFamily: FONT_SERIF, fontSize: 34, bold: true, color: accent });
  addText(slide, label.toUpperCase(), x, y + 68, w, 22, { fontFamily: FONT_MONO, fontSize: 11, color: C.muted });
  if (detail) addText(slide, detail, x, y + 94, w, 40, { fontFamily: FONT_SANS, fontSize: 15, color: C.soft });
}

function addBullet(slide, text, x, y, w, color = C.forest2, size = 17) {
  addRect(slide, x, y + 8, 7, 7, color, color, 0);
  addText(slide, text, x + 20, y, w - 20, 50, { fontFamily: FONT_SANS, fontSize: size, color: C.ink });
}

function setSources(slide, sources) {
  slide.speakerNotes.textFrame.setText(`[Sources]\n${sources.map((s) => `- ${s}`).join("\n")}\n[/Sources]`);
}

function addImageFrame(slide, bytes, contentType, x, y, w, h, alt) {
  addRect(slide, x, y, w, h, C.card, C.line, 1);
  slide.images.add({
    blob: bytes,
    contentType,
    alt,
    fit: "contain",
    position: { left: x + 8, top: y + 8, width: w - 16, height: h - 16 },
    geometry: "rect",
  });
}

function addCode(slide, text, x, y, w, h, fontSize = 15) {
  addRect(slide, x, y, w, h, C.dark, C.dark, 0);
  addText(slide, text, x + 18, y + 16, w - 36, h - 28, { fontFamily: FONT_MONO, fontSize, color: C.code });
}

function addDiagramNode(slide, number, title, detail, x, y, w, h, active = false) {
  addRect(slide, x, y, w, h, active ? C.mint : C.card, active ? C.forest2 : C.line, active ? 2 : 1);
  addRect(slide, x, y, 38, h, active ? C.forest : C.paper2, active ? C.forest : C.line, 0);
  addText(slide, number, x + 7, y + 15, 24, 26, { fontFamily: FONT_MONO, fontSize: 18, bold: true, color: active ? C.white : C.soft, alignment: "center" });
  addText(slide, title, x + 52, y + 12, w - 64, 28, { fontFamily: FONT_SANS, fontSize: 17, bold: true, color: C.ink });
  addText(slide, detail, x + 52, y + 45, w - 64, h - 52, { fontFamily: FONT_SANS, fontSize: 16, color: C.soft });
}

async function build() {
  const originalPage = await readBytes(path.join(ROOT, "tmp/pdfs/project-overview/source_pages/page_1.png"));
  const blockPage = await readBytes(path.join(RUN_DIR, "artifacts/blocks/page_1_blocks.png"));
  const recordPage = await readBytes(path.join(RUN_DIR, "artifacts/record_separation/page_1_records.png"));

  const presentation = Presentation.create({ slideSize: { width: W, height: H } });

  // 1. Minimal title slide.
  {
    const s = baseSlide(presentation, "Project overview", 1);
    addText(s, "THESIS RESEARCH SYSTEM", 64, 100, 360, 26, { fontFamily: FONT_MONO, fontSize: 13, color: C.forest2 });
    addText(s, "Compile once.\nExtract indefinitely.", 64, 150, 730, 155, { fontFamily: FONT_SERIF, fontSize: 58, bold: true, color: C.ink });
    addRule(s, 64, 329, 620, C.forest2, 5);
    addText(s, "TWIX 2 learns a reusable structured-data extractor from a small sample of a template-based PDF family, then ships ordinary Python code for every later page.", 64, 365, 700, 115, { fontFamily: FONT_SANS, fontSize: 23, color: C.soft });
    addRect(s, 860, 132, 300, 350, C.dark, C.dark, 0);
    addText(s, "ID_05 CASE STUDY", 890, 164, 230, 22, { fontFamily: FONT_MONO, fontSize: 12, color: "#9db6aa" });
    addText(s, "95.5%", 890, 205, 230, 76, { fontFamily: FONT_SERIF, fontSize: 58, bold: true, color: C.white });
    addText(s, "HELD-OUT ACCURACY", 890, 286, 230, 24, { fontFamily: FONT_MONO, fontSize: 12, color: C.code });
    addRule(s, 890, 337, 230, "#46534d", 1);
    addText(s, "15 test pages\n56 matched records\n224 physical blocks\n0 runtime model calls", 890, 360, 230, 100, { fontFamily: FONT_SANS, fontSize: 17, color: C.code });
    addText(s, "1 September 2026", 64, 615, 300, 24, { fontFamily: FONT_MONO, fontSize: 12, color: C.muted });
    setSources(s, [`${RUN_REL}/metadata.json`, `${RUN_REL}/pipeline/eval.json`, "UI/styles.css"]);
  }

  // 2. Intro economics diagram.
  {
    const s = baseSlide(presentation, "Research premise", 2);
    slideTitle(s, "01 / Why this architecture", "The problem is not one page - it is the repeated model call", "A frontier model can parse a page well. The operational cost still grows every time another page is sent.");
    addRect(s, 80, 260, 500, 274, C.card, C.line, 1);
    addRect(s, 620, 260, 520, 274, C.card, C.line, 1);

    addText(s, "CONVENTIONAL MODEL-PER-PAGE", 106, 282, 340, 22, { fontFamily: FONT_MONO, fontSize: 12, color: C.red });
    addText(s, "LLM / API COST", 140, 321, 150, 18, { fontFamily: FONT_MONO, fontSize: 10, color: C.muted });
    addRect(s, 140, 345, 2, 125, C.line, C.line, 0);
    addRect(s, 140, 468, 390, 2, C.line, C.line, 0);
    addLine(s, 158, 455, 512, 353, C.red, 4);
    addDot(s, 503, 344, 18, C.red);
    addRect(s, 296, 370, 230, 49, C.card, C.card, 0);
    addText(s, "cost grows with every page", 305, 374, 215, 22, { fontFamily: FONT_SANS, fontSize: 15, bold: true, color: C.red });
    addText(s, "pages processed", 140, 483, 175, 18, { fontFamily: FONT_MONO, fontSize: 10, color: C.muted });

    addText(s, "TWIX 2 COMPILE-ONCE", 646, 282, 300, 22, { fontFamily: FONT_MONO, fontSize: 12, color: C.forest2 });
    addText(s, "LLM / API COST", 680, 321, 150, 18, { fontFamily: FONT_MONO, fontSize: 10, color: C.muted });
    addRect(s, 680, 345, 2, 125, C.line, C.line, 0);
    addRect(s, 680, 468, 410, 2, C.line, C.line, 0);
    addLine(s, 698, 455, 752, 350, C.forest2, 4);
    addLine(s, 752, 350, 1068, 350, C.forest2, 4);
    addDot(s, 743, 341, 18, C.forest2);
    addText(s, "no further model calls", 792, 360, 220, 22, { fontFamily: FONT_SANS, fontSize: 15, bold: true, color: C.forest2 });
    addText(s, "pages processed", 680, 483, 175, 18, { fontFamily: FONT_MONO, fontSize: 10, color: C.muted });

    addRect(s, 80, 556, 1060, 72, C.mint, C.forest2, 1);
    addText(s, "The hypothesis: structure turns a broad reasoning task into smaller code-generation tasks that cheaper - and potentially local - models can solve.", 104, 556, 1012, 72, { fontFamily: FONT_SANS, fontSize: 19, bold: true, color: C.forest, alignment: "center", verticalAlignment: "middle" });
    setSources(s, ["AGENTS.md", "UI/styles.css"]);
  }

  // 3. End-to-end architecture diagram.
  {
    const s = baseSlide(presentation, "Architecture", 3);
    slideTitle(s, "02 / End-to-end pipeline", "Models compile the extractor; ordinary Python runs it", "Learning happens once; extraction becomes a repeatable, model-free runtime.");
    addText(s, "ONE-TIME COMPILATION", 54, 238, 320, 22, { fontFamily: FONT_MONO, fontSize: 12, color: C.forest2 });
    const xs = [54, 285, 516, 747, 978];
    for (let i = 0; i < 4; i++) addArrow(s, xs[i] + 202, 324, 29, 22, C.muted);
    const nodes = [
      ["1", "Detect + split", "Find content start; create train and test OCR."],
      ["2", "Infer template", "Define node ids, fields and metadata patterns."],
      ["3", "Write code", "Generate separator and one extractor per node."],
      ["4", "Train gate", "Evaluate all train pages; repair if needed."],
      ["5", "Package", "Copy model-free code into pipeline/."],
    ];
    nodes.forEach((n, i) => addDiagramNode(s, n[0], n[1], n[2], xs[i], 274, 202, 144, i === 2));
    addRule(s, 54, 448, 1126, C.line, 2);
    addText(s, "REPEATED RUNTIME", 54, 474, 320, 22, { fontFamily: FONT_MONO, fontSize: 12, color: C.forest2 });
    const rxs = [120, 370, 620, 870];
    for (let i = 0; i < 3; i++) addArrow(s, rxs[i] + 190, 558, 60, 22, C.forest2);
    const runtime = [
      ["PDF", "same family"],
      ["OCR", "words + geometry"],
      ["Code", "generated scripts"],
      ["Extracted data", "logical records"],
    ];
    runtime.forEach((n, i) => {
      addRect(s, rxs[i], 520, 190, 92, i === 2 ? C.mint : C.card, i === 2 ? C.forest2 : C.line, i === 2 ? 2 : 1);
      addText(s, n[0], rxs[i] + 14, 538, 162, 26, { fontFamily: FONT_SANS, fontSize: 19, bold: true, alignment: "center" });
      addText(s, n[1], rxs[i] + 14, 575, 162, 20, { fontFamily: FONT_MONO, fontSize: 11, color: C.muted, alignment: "center" });
    });
    setSources(s, ["src/agents/graph.py", "src/pipeline.py", "src/record_grouping.py"]);
  }

  // 4. Case study source.
  {
    const s = baseSlide(presentation, "Case study", 4);
    slideTitle(s, "03 / id_05", "A 20-page sample contains a stable four-block record pattern", "The report repeats complaint summaries, complainant details, complaint rows and officer rows.");
    addImageFrame(s, originalPage, "image/png", 54, 228, 720, 398, "Original id_05 complaint report page with source redactions preserved");
    addMetric(s, "20", "source pages", 830, 242, 150, C.forest2, "5 train + 15 test");
    addMetric(s, "7,270", "OCR words", 1010, 242, 170, C.forest, "global ids + geometry");
    addMetric(s, "4", "template nodes", 830, 420, 150, C.amber, "3 tables + 1 KV");
    addMetric(s, "56", "test records", 1010, 420, 170, C.red, "after deterministic grouping");
    addText(s, "Source redactions are preserved in every visual.", 830, 590, 350, 26, { fontFamily: FONT_MONO, fontSize: 11, color: C.muted });
    setSources(s, [`${RUN_REL}/id_05.pdf`, `${RUN_REL}/metadata.json`, `${RUN_REL}/ocr_extraction_full.csv`, "tmp/pdfs/project-overview/source_pages/page_1.png"]);
  }

  // 5. Split and holdout diagram.
  {
    const s = baseSlide(presentation, "Case study", 5);
    slideTitle(s, "04 / Independent evaluation split", "Five pages compile the extractor; fifteen pages test it", "This run uses all five training pages for code generation, then evaluates the frozen bundle on fifteen independent pages.");
    addArrow(s, 218, 326, 94, 24, C.muted);
    addArrow(s, 608, 326, 94, 24, C.muted);
    addArrow(s, 956, 326, 34, 24, C.muted);
    addRect(s, 64, 260, 154, 168, C.card, C.line, 1);
    addText(s, "20", 94, 290, 94, 56, { fontFamily: FONT_SERIF, fontSize: 46, bold: true, color: C.ink, alignment: "center" });
    addText(s, "SOURCE PAGES", 84, 360, 114, 24, { fontFamily: FONT_MONO, fontSize: 11, color: C.muted, alignment: "center" });
    addRect(s, 312, 240, 296, 208, C.mint, C.forest2, 2);
    addText(s, "5 TRAIN", 344, 268, 232, 30, { fontFamily: FONT_MONO, fontSize: 13, color: C.forest, alignment: "center" });
    addRect(s, 344, 318, 232, 92, C.card, C.line, 1);
    addText(s, "5", 388, 330, 144, 40, { fontFamily: FONT_SERIF, fontSize: 34, bold: true, alignment: "center" });
    addText(s, "BUILD PAGES", 360, 378, 200, 18, { fontFamily: FONT_MONO, fontSize: 10, color: C.muted, alignment: "center" });
    addRect(s, 702, 260, 254, 168, C.card, C.line, 1);
    addText(s, "15", 764, 290, 130, 56, { fontFamily: FONT_SERIF, fontSize: 46, bold: true, color: C.ink, alignment: "center" });
    addText(s, "INDEPENDENT TEST", 738, 360, 182, 24, { fontFamily: FONT_MONO, fontSize: 11, color: C.muted, alignment: "center" });
    addRect(s, 990, 260, 210, 168, C.dark, C.dark, 0);
    addText(s, "FROZEN BUNDLE", 1015, 287, 160, 20, { fontFamily: FONT_MONO, fontSize: 12, color: "#9db6aa", alignment: "center" });
    addText(s, "95.5%", 1010, 332, 170, 38, { fontFamily: FONT_SERIF, fontSize: 30, bold: true, color: C.white, alignment: "center" });
    addText(s, "held-out accuracy", 1010, 382, 170, 24, { fontFamily: FONT_SANS, fontSize: 16, color: C.code, alignment: "center" });
    addRect(s, 64, 500, 1136, 96, C.card, C.line, 1);
    addText(s, "Result", 88, 527, 100, 22, { fontFamily: FONT_MONO, fontSize: 12, color: C.forest2 });
    addText(s, "The bundle is generated once from five training pages, then evaluated without changing its code on fifteen independent pages.", 200, 520, 960, 50, { fontFamily: FONT_SANS, fontSize: 22, bold: true, color: C.ink });
    setSources(s, [`${RUN_REL}/metadata.json`, `${RUN_REL}/agent_space/train/ocr_extraction.csv`, `${RUN_REL}/agent_space/test/ocr_extraction.csv`, `${RUN_REL}/pipeline/train_eval/eval.json`]);
  }

  // 6. Template contract.
  {
    const s = baseSlide(presentation, "Template", 6);
    slideTitle(s, "05 / Template contract", "Four node ids define every lookup, block and extractor", "Field names describe semantics; bid describes visual position; metadata regexes distinguish repeated page furniture from record content.");
    addCode(s, `{
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
}`, 54, 238, 420, 378, 16);
    const rows = [
      ["0", "table", "10 fields", "Complaint summary"],
      ["1", "key-value", "5 fields", "Complainant details"],
      ["2", "table", "3 fields", "Complaint rows"],
      ["3", "table", "7 fields", "Officer rows"],
    ];
    addRect(s, 526, 238, 654, 54, C.dark, C.dark, 0);
    ["NODE", "TYPE", "SIZE", "ROLE"].forEach((t, i) => addText(s, t, 544 + [0, 100, 260, 360][i], 256, [80, 140, 80, 240][i], 18, { fontFamily: FONT_MONO, fontSize: 11, color: C.code }));
    rows.forEach((r, i) => {
      const y = 292 + i * 66;
      addRect(s, 526, y, 654, 66, i % 2 ? C.paper2 : C.card, C.line, 1);
      addText(s, r[0], 544, y + 20, 80, 24, { fontFamily: FONT_MONO, fontSize: 16, bold: true });
      addText(s, r[1], 644, y + 20, 140, 24, { fontFamily: FONT_SANS, fontSize: 16, bold: true });
      addText(s, r[2], 804, y + 20, 80, 24, { fontFamily: FONT_MONO, fontSize: 14, color: C.muted });
      addText(s, r[3], 904, y + 20, 250, 24, { fontFamily: FONT_SANS, fontSize: 16, color: C.soft });
    });
    addRect(s, 526, 573, 654, 43, C.mint, C.forest2, 1);
    addText(s, "9 metadata regexes: 7 document-level + 2 record-level", 548, 585, 610, 20, { fontFamily: FONT_MONO, fontSize: 13, color: C.forest });
    setSources(s, [`${RUN_REL}/template.json`, "AGENTS.md"]);
  }

  // 7. Block separation visual.
  {
    const s = baseSlide(presentation, "Block separation", 7);
    slideTitle(s, "06 / Physical block separation", "Generated geometry code finds page-local blocks", "On page 1, twelve visual regions map to four node types; multi-page tables remain separate continuation blocks.");
    addImageFrame(s, blockPage, "image/png", 54, 228, 820, 406, "id_05 page 1 with color-coded physical block overlays");
    addRect(s, 910, 228, 270, 406, C.card, C.line, 1);
    addText(s, "IDENTITY CONTRACT", 936, 252, 220, 22, { fontFamily: FONT_MONO, fontSize: 12, color: C.forest2 });
    addCode(s, `block_id =
<node_id>_<instance>

0_1  summary
1_1  complainant
2_1  complaints
3_1  officers

uid = physical join key`, 936, 292, 218, 226, 15);
    addText(s, "Page-local only", 936, 530, 210, 26, { fontFamily: FONT_SERIF, fontSize: 24, bold: true, color: C.red });
    addText(s, "A block emitted under page P may contain only page-P words.", 936, 564, 210, 50, { fontFamily: FONT_SANS, fontSize: 16, color: C.soft });
    setSources(s, [`${RUN_REL}/agent_space/train/block_separation.json`, `${RUN_REL}/artifacts/blocks/page_1_blocks.png`, `${RUN_REL}/metadata.json`]);
  }

  // 8. Record grouping visual.
  {
    const s = baseSlide(presentation, "Record grouping", 8);
    slideTitle(s, "07 / Logical record grouping", "Blocks 0 → 1 → 2 → 3 form one complaint record", "Deterministic grouping preserves every physical block while producing a record-level JSON structure.");
    addImageFrame(s, recordPage, "image/png", 54, 228, 820, 406, "id_05 page 1 with complete logical record overlays");
    addRect(s, 910, 228, 270, 406, C.mint, C.forest2, 1);
    addText(s, "ONE RECORD", 936, 252, 220, 22, { fontFamily: FONT_MONO, fontSize: 12, color: C.forest2 });
    const seq = [["0_0", "summary"], ["1_0", "KV details"], ["2_0", "complaint rows"], ["3_0", "officer rows"]];
    seq.forEach((r, i) => {
      if (i < 3) addRect(s, 961, 333 + i * 64, 3, 34, C.forest2, C.forest2, 0);
      addDot(s, 948, 294 + i * 64, 28, i === 0 ? C.red : C.forest2);
      addText(s, r[0], 994, 298 + i * 64, 60, 24, { fontFamily: FONT_MONO, fontSize: 15, bold: true });
      addText(s, r[1], 1064, 298 + i * 64, 92, 42, { fontFamily: FONT_SANS, fontSize: 15, color: C.soft });
    });
    addRule(s, 936, 560, 218, C.forest2, 2);
    addText(s, "56 / 56", 936, 571, 218, 32, { fontFamily: FONT_SERIF, fontSize: 26, bold: true, color: C.forest, alignment: "center" });
    addText(s, "matched test records", 936, 606, 218, 20, { fontFamily: FONT_MONO, fontSize: 11, color: C.soft, alignment: "center" });
    setSources(s, [`${RUN_REL}/artifacts/record_separation/page_1_records.png`, "src/record_grouping.py", `${RUN_REL}/pipeline/eval.json`]);
  }

  // 9. Routing and generated scripts.
  {
    const s = baseSlide(presentation, "Code synthesis", 9);
    slideTitle(s, "08 / Difficulty routing", "Routing reserves the stronger model for difficult nodes", "Nodes 0 and 1 use Qwen3.8 Flash; the harder table extractors use GLM-5.3 Flash.");
    const rows = [
      ["NODE 0", "TABLE", "198 lines", "Qwen3.8 Flash", C.paper2],
      ["NODE 1", "KEY-VALUE", "139 lines", "Qwen3.8 Flash", C.paper2],
      ["NODE 2", "TABLE", "224 lines", "GLM-5.3 Flash", C.mint],
      ["NODE 3", "TABLE", "190 lines", "GLM-5.3 Flash", C.mint],
    ];
    rows.forEach((r, i) => {
      const y = 238 + i * 82;
      addRect(s, 54, y, 720, 66, r[4], i >= 2 ? C.forest2 : C.line, i >= 2 ? 2 : 1);
      addText(s, r[0], 76, y + 20, 110, 24, { fontFamily: FONT_MONO, fontSize: 14, bold: true, color: C.forest });
      addText(s, r[1], 196, y + 20, 150, 24, { fontFamily: FONT_SANS, fontSize: 17, bold: true });
      addText(s, r[2], 376, y + 20, 100, 24, { fontFamily: FONT_MONO, fontSize: 13, color: C.muted });
      addText(s, r[3], 496, y + 20, 250, 24, { fontFamily: FONT_SANS, fontSize: 16, color: C.soft });
    });
    addCode(s, `# extraction_scripts/<node_id>.py
def extract(block: dict) -> dict:
    ...

# runtime dispatch
node_id = block["block_id"].rsplit("_", 1)[0]
result = extractor[node_id].extract(relative_block)
result["page"] = page
result["uid"] = uid`, 820, 238, 360, 394, 15);
    setSources(s, [`${RUN_REL}/metadata.json`, `${RUN_REL}/pipeline/extraction_scripts/0.py`, `${RUN_REL}/pipeline/extraction_scripts/1.py`, `${RUN_REL}/pipeline/extraction_scripts/2.py`, `${RUN_REL}/pipeline/extraction_scripts/3.py`]);
  }

  // 10. Runtime diagram.
  {
    const s = baseSlide(presentation, "Portable runtime", 10);
    slideTitle(s, "09 / Runtime contract", "The deployed path is a six-step local program", "LangGraph, OpenCode and model providers disappear from the production extraction path.");
    const xs = [54, 248, 442, 636, 830, 1024];
    for (let i = 0; i < 5; i++) addArrow(s, xs[i] + 160, 349, 34, 22, C.forest2);
    const nodes = [
      ["1", "New PDF", "same family"],
      ["2", "OCR", "words + geometry"],
      ["3", "Separate", "blocks + uids"],
      ["4", "Extract", "relative blocks"],
      ["5", "Group", "logical records"],
      ["6", "JSON", "portable output"],
    ];
    nodes.forEach((n, i) => addDiagramNode(s, n[0], n[1], n[2], xs[i], 286, 160, 150, i === 3));
    addRect(s, 190, 488, 900, 94, C.dark, C.dark, 0);
    addText(s, "NO LANGGRAPH", 232, 520, 190, 24, { fontFamily: FONT_MONO, fontSize: 16, bold: true, color: C.code, alignment: "center" });
    addText(s, "NO OPENCODE", 528, 520, 190, 24, { fontFamily: FONT_MONO, fontSize: 16, bold: true, color: C.code, alignment: "center" });
    addText(s, "NO MODEL CALLS", 824, 520, 210, 24, { fontFamily: FONT_MONO, fontSize: 16, bold: true, color: "#9db6aa", alignment: "center" });
    addText(s, "The marginal LLM/API cost of extraction is zero after compilation; local compute remains.", 190, 606, 900, 30, { fontFamily: FONT_SANS, fontSize: 17, color: C.soft, alignment: "center" });
    setSources(s, ["src/pipeline.py", "src/record_grouping.py", `${RUN_REL}/pipeline/requirements.txt`]);
  }

  // 11. Runtime output example.
  {
    const s = baseSlide(presentation, "Runtime output", 11);
    slideTitle(s, "10 / Extracted record", "Physical blocks stay visible inside each logical record", "Grouping preserves page, uid and block identity while making the output usable downstream.");
    addCode(s, `{
  "record_id": "0",
  "blocks": [
    {"block_id": "0_0", "type": "table", "page": 1},
    {"block_id": "1_0", "type": "kv",    "page": 1},
    {"block_id": "2_0", "type": "table", "page": 1},
    {"block_id": "3_0", "type": "table", "page": 1}
  ]
}`, 54, 238, 540, 368, 17);
    addRect(s, 638, 238, 542, 368, C.card, C.line, 1);
    addText(s, "RECORD 0 / SELECTED VALUES", 666, 264, 470, 22, { fontFamily: FONT_MONO, fontSize: 12, color: C.forest2 });
    const values = [
      ["DATE", "7/19/2012"], ["NUMBER", "08-14"],
      ["INVESTIGATOR", "Kim, Joon"], ["DISPOSITION", "UNFOUNDED"],
      ["GENDER", "MALE"], ["OFFICER", "Carter, Felix."],
      ["COMPLAINT", "R-3B.3 Courtesy: Conduct"], ["ACTION", "NONE"],
    ];
    values.forEach((v, i) => {
      const col = i % 2;
      const row = Math.floor(i / 2);
      const x = 666 + col * 250;
      const y = 318 + row * 68;
      addText(s, v[0], x, y, 220, 18, { fontFamily: FONT_MONO, fontSize: 10, color: C.muted });
      addText(s, v[1], x, y + 22, 220, 26, { fontFamily: FONT_SANS, fontSize: 17, bold: true, color: C.ink });
    });
    setSources(s, [`${RUN_REL}/pipeline/extracted.json`, `${RUN_REL}/pipeline/grouped.json`]);
  }

  // 12. Evaluation.
  {
    const s = baseSlide(presentation, "Evaluation", 12);
    slideTitle(s, "11 / Held-out result", "Held-out extraction reaches 95.5% accuracy", "Across 15 independent pages, every record and node type is found; remaining errors are localized value-boundary and column-assignment issues.");
    addMetric(s, "95.1%", "train accuracy", 54, 242, 240, C.forest2, "5 pages / 19 records");
    addMetric(s, "95.5%", "test accuracy", 344, 242, 240, C.forest, "15 pages / 56 records");
    addMetric(s, "1.000", "record match F1", 634, 242, 240, C.amber, "perfect precision + recall");
    addMetric(s, "1.000", "node match F1", 924, 242, 256, C.red, "typed precision + recall");
    const bars = [
      ["Content overall", 0.909636, C.forest2],
      ["Key-value content", 0.839286, C.forest],
      ["Table content", 0.939048, C.amber],
    ];
    bars.forEach((b, i) => {
      const y = 462 + i * 58;
      addText(s, b[0], 54, y, 230, 22, { fontFamily: FONT_SANS, fontSize: 17, color: C.ink });
      addRect(s, 300, y + 4, 700, 18, C.paper2, C.line, 0);
      addRect(s, 300, y + 4, 700 * b[1], 18, b[2], b[2], 0);
      addText(s, b[1].toFixed(3), 1030, y, 120, 22, { fontFamily: FONT_MONO, fontSize: 14, bold: true, color: b[2], alignment: "right" });
    });
    setSources(s, [`${RUN_REL}/pipeline/eval.json`, `${RUN_REL}/pipeline/train_eval/eval.json`, "src/eval/score.py"]);
  }

  // 13. Error analysis.
  {
    const s = baseSlide(presentation, "Evaluation", 13);
    slideTitle(s, "12 / Error analysis", "The residual gap is narrow and actionable", "The scorer reports field-level reasons that can be handed directly to the optional repair loop.");
    addRect(s, 54, 238, 540, 368, C.card, C.line, 1);
    addText(s, "TYPICAL MISMATCH", 80, 264, 480, 22, { fontFamily: FONT_MONO, fontSize: 12, color: C.red });
    addText(s, "Gold", 80, 318, 90, 24, { fontFamily: FONT_MONO, fontSize: 12, color: C.muted });
    addText(s, "FORMAL", 170, 312, 380, 48, { fontFamily: FONT_SANS, fontSize: 19, bold: true });
    addRule(s, 80, 376, 480, C.line, 1);
    addText(s, "Candidate", 80, 404, 90, 24, { fontFamily: FONT_MONO, fontSize: 12, color: C.muted });
    addText(s, "FORMAL Citizen", 170, 398, 380, 58, { fontFamily: FONT_SANS, fontSize: 19, bold: true, color: C.red });
    addRule(s, 80, 474, 480, C.line, 1);
    addText(s, "Reason", 80, 503, 90, 24, { fontFamily: FONT_MONO, fontSize: 12, color: C.muted });
    addText(s, "A neighboring column value crosses the Category / Type boundary.", 170, 496, 380, 60, { fontFamily: FONT_SANS, fontSize: 17, color: C.soft });
    addRect(s, 638, 238, 542, 368, C.mint, C.forest2, 1);
    addText(s, "WHAT IS NOT FAILING", 666, 264, 470, 22, { fontFamily: FONT_MONO, fontSize: 12, color: C.forest2 });
    addBullet(s, "No missing gold records", 666, 318, 450, C.forest2, 19);
    addBullet(s, "No extra candidate records", 666, 382, 450, C.forest2, 19);
    addBullet(s, "No missing or extra nodes", 666, 446, 450, C.forest2, 19);
    addBullet(s, "No node type mismatches", 666, 510, 450, C.forest2, 19);
    setSources(s, [`${RUN_REL}/pipeline/eval.txt`, `${RUN_REL}/pipeline/eval.json`, "src/agents/train_repair.py"]);
  }

  // 14. Artifact map.
  {
    const s = baseSlide(presentation, "Artifacts", 14);
    slideTitle(s, "13 / Audit trail", "Every phase leaves a file that can be inspected or deployed", "The run directory is simultaneously a provenance record, a visual QA workspace and a portable extractor.");
    addRect(s, 54, 238, 1126, 382, C.card, C.line, 1);
    addText(s, `${RUN_REL}/`, 82, 258, 700, 26, { fontFamily: FONT_MONO, fontSize: 15, bold: true, color: C.forest });
    const tree = [
      ["├── ocr_extraction_full.csv", "full-document OCR"],
      ["├── template.json", "schema + metadata contract"],
      ["├── metadata.json", "models, configuration, scores"],
      ["├── artifacts/", "block + record overlays"],
      ["├── agent_space/", "train/test PDFs and OCR"],
      ["└── pipeline/", "deployable bundle + outputs"],
    ];
    tree.forEach((r, i) => {
      const y = 304 + i * 46;
      addText(s, r[0], 82, y, 430, 22, { fontFamily: FONT_MONO, fontSize: 15, color: i === 5 ? C.forest : C.ink, bold: i === 5 });
      addText(s, r[1], 520, y, 300, 22, { fontFamily: FONT_SANS, fontSize: 16, color: C.soft });
    });
    addRule(s, 840, 272, 1, C.line, 1);
    addText(s, "pipeline/", 878, 286, 230, 26, { fontFamily: FONT_MONO, fontSize: 17, bold: true, color: C.forest });
    const bundle = ["pipeline.py", "block_separation.py", "extraction_scripts/", "record_grouping.py", "requirements.txt", "template.json", "grouped.json", "pipeline_health.json", "eval.json + eval.txt"];
    bundle.forEach((t, i) => addText(s, t, 878, 326 + i * 31, 255, 20, { fontFamily: FONT_MONO, fontSize: 13, color: i < 6 ? C.ink : C.muted }));
    setSources(s, [`${RUN_REL}/`, "AGENTS.md"]);
  }

  // 15. Economics.
  {
    const s = baseSlide(presentation, "Economics", 15);
    slideTitle(s, "14 / Cost model", "One compilation cost is amortized over every later page", "The architectural claim is about marginal model cost, not about eliminating local compute.");
    addRect(s, 54, 238, 540, 330, C.mint, C.forest2, 2);
    addText(s, "AFTER COMPILATION", 82, 266, 470, 22, { fontFamily: FONT_MONO, fontSize: 12, color: C.forest2 });
    addText(s, "$0", 82, 316, 170, 80, { fontFamily: FONT_SERIF, fontSize: 64, bold: true, color: C.forest });
    addText(s, "marginal LLM/API cost", 240, 342, 300, 28, { fontFamily: FONT_SANS, fontSize: 22, bold: true });
    addBullet(s, "No model context window grows with production pages.", 82, 424, 470, C.forest2, 17);
    addBullet(s, "The same code handles each same-family page.", 82, 482, 470, C.forest2, 17);
    addRect(s, 638, 238, 542, 330, C.card, C.line, 1);
    addText(s, "ID_05 COMPILATION SCALE", 666, 266, 470, 22, { fontFamily: FONT_MONO, fontSize: 12, color: C.red });
    addText(s, "3.40M", 666, 316, 220, 70, { fontFamily: FONT_SERIF, fontSize: 48, bold: true, color: C.red });
    addText(s, "input + output tokens", 874, 341, 266, 26, { fontFamily: FONT_SANS, fontSize: 19, bold: true });
    addText(s, "One-time code-generation workload", 666, 420, 450, 30, { fontFamily: FONT_SANS, fontSize: 18, color: C.soft });
    addText(s, "This expense is paid once for the document family, then amortized across every future extraction.", 666, 470, 450, 70, { fontFamily: FONT_SANS, fontSize: 18, color: C.ink });
    addText(s, "Local OCR, CPU, storage and hosting still consume resources.", 54, 600, 1126, 24, { fontFamily: FONT_MONO, fontSize: 12, color: C.muted, alignment: "center" });
    setSources(s, [`${RUN_REL}/metadata.json`, `${RUN_REL}/metadata/cost.json`]);
  }

  // 16. Limitations and close.
  {
    const s = baseSlide(presentation, "Conclusion", 16);
    slideTitle(s, "15 / What the result supports", "Use models to write the extractor - not every extraction", "id_05 preserves record structure and approaches frontier-level content quality on an independent split.");
    addRect(s, 54, 236, 710, 360, C.dark, C.dark, 0);
    addText(s, "ID_05 RESULT", 82, 262, 300, 22, { fontFamily: FONT_MONO, fontSize: 12, color: "#9db6aa" });
    addText(s, "5", 82, 316, 100, 48, { fontFamily: FONT_SERIF, fontSize: 38, bold: true, color: C.white });
    addText(s, "training pages", 82, 370, 140, 22, { fontFamily: FONT_SANS, fontSize: 15, color: C.code });
    addText(s, "15", 236, 316, 100, 48, { fontFamily: FONT_SERIF, fontSize: 38, bold: true, color: C.white });
    addText(s, "test pages", 236, 370, 120, 22, { fontFamily: FONT_SANS, fontSize: 15, color: C.code });
    addText(s, "56", 382, 316, 100, 48, { fontFamily: FONT_SERIF, fontSize: 38, bold: true, color: C.white });
    addText(s, "matched records", 382, 370, 140, 22, { fontFamily: FONT_SANS, fontSize: 15, color: C.code });
    addText(s, "95.5%", 538, 316, 150, 48, { fontFamily: FONT_SERIF, fontSize: 38, bold: true, color: C.white });
    addText(s, "overall accuracy", 538, 370, 150, 22, { fontFamily: FONT_SANS, fontSize: 15, color: C.code });
    addRule(s, 82, 430, 654, "#46534d", 1);
    addText(s, "0 runtime model calls", 82, 468, 654, 42, { fontFamily: FONT_SERIF, fontSize: 30, bold: true, color: "#9db6aa", alignment: "center" });
    addText(s, "A reusable program replaces a repeated model call.", 82, 528, 654, 30, { fontFamily: FONT_MONO, fontSize: 13, color: C.code, alignment: "center" });
    addText(s, "BOUNDARIES", 816, 250, 340, 22, { fontFamily: FONT_MONO, fontSize: 12, color: C.red });
    addBullet(s, "Family-specific, not a generic PDF extractor.", 816, 302, 350, C.red, 17);
    addBullet(s, "Layout drift and OCR errors may require recompilation.", 816, 372, 350, C.red, 17);
    addBullet(s, "Ground truth is only as reliable as its reference process.", 816, 454, 350, C.red, 17);
    addBullet(s, "Zero model cost still requires local compute.", 816, 536, 350, C.red, 17);
    setSources(s, [`${RUN_REL}/metadata.json`, `${RUN_REL}/pipeline/eval.json`, "AGENTS.md"]);
  }

  await fs.mkdir(path.join(TMP, "rendered"), { recursive: true });
  await fs.mkdir(path.join(TMP, "layouts"), { recursive: true });

  for (const [index, slide] of presentation.slides.items.entries()) {
    const stem = `slide-${String(index + 1).padStart(2, "0")}`;
    const png = await presentation.export({ slide, format: "png", scale: 1 });
    await writeBlob(path.join(TMP, "rendered", `${stem}.png`), png);
    const layout = await slide.export({ format: "layout" });
    await fs.writeFile(path.join(TMP, "layouts", `${stem}.layout.json`), await layout.text());
  }
  const montage = await presentation.export({ format: "webp", montage: true, scale: 1 });
  await writeBlob(path.join(TMP, "deck-montage.webp"), montage);
  const pptx = await PresentationFile.exportPptx(presentation);
  await pptx.save(OUT);
  console.log(OUT);
}

build().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
