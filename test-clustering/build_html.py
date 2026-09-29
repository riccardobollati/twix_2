"""Build a self-contained, local HTML browser for page clustering results."""
from __future__ import annotations

from html import escape
import json
import os
from pathlib import Path
from urllib.parse import quote


METHODS = ["tight_complete", "tight_average", "structure_complete",
           "structure_average", "structure_single"]
CSS = """
:root{font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;color:#1c2430;background:#f3f5f8}
*{box-sizing:border-box}body{margin:0}a{color:#184e91}header{background:#102b46;color:white;padding:24px 32px}header a{color:#d9ebff}
h1{font-size:25px;margin:0 0 7px}h2{font-size:19px}h3{font-size:15px}p{line-height:1.5}main{max-width:1600px;margin:auto;padding:24px}
.sub{color:#c6d8e8;margin:0}.bar{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin:18px 0}
.card,.panel{background:white;border:1px solid #dce3ec;border-radius:10px;box-shadow:0 2px 7px #14233909}
.card{padding:16px}.summary-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin:20px 0}
.label{font-size:12px;text-transform:uppercase;letter-spacing:.06em;color:#617184}.value{font-size:22px;font-weight:700;margin-top:4px}
input,select{border:1px solid #b8c5d5;border-radius:7px;padding:9px 11px;font:inherit;background:white;color:#1c2430}
button{font:inherit;border:1px solid #afbed0;border-radius:6px;background:white;color:#183d64;cursor:pointer;padding:5px 9px}
button:hover,.page-button.active{background:#d9eafb;border-color:#5389c2}table{border-collapse:collapse;width:100%;background:white}
th,td{text-align:left;padding:11px 12px;border-bottom:1px solid #e5eaf0;vertical-align:top}th{background:#e9eef4;font-size:12px;text-transform:uppercase;letter-spacing:.04em;position:sticky;top:0}
tr:hover td{background:#f8fbff}.table-wrap{overflow:auto;border:1px solid #dce3ec;border-radius:9px}
.pill{display:inline-block;border-radius:99px;padding:3px 8px;font-size:12px;background:#e6f0fb;color:#184e91}
.pill.warn{background:#fff0d6;color:#7d4a00}.pill.bad{background:#fde8e8;color:#9d2424}
.layout{display:grid;grid-template-columns:minmax(360px,44%) minmax(420px,56%);gap:16px;align-items:start}
.clusters{display:grid;gap:13px;max-height:calc(100vh - 160px);overflow:auto;padding-right:4px}
.cluster{padding:14px}.cluster-top{display:flex;align-items:baseline;justify-content:space-between;gap:10px}
.cluster img{width:100%;max-height:260px;object-fit:contain;background:#eef1f4;border:1px solid #e4e8ed;border-radius:5px;cursor:pointer}
.pages{display:flex;flex-wrap:wrap;gap:5px;margin-top:9px}.page-button{min-width:35px;font-size:12px}
.muted{color:#607083}.preview{position:sticky;top:16px;padding:16px;min-height:540px}
.preview img{width:100%;max-height:70vh;object-fit:contain;background:#eef1f4;border:1px solid #e4e8ed;border-radius:6px}
.preview iframe{width:100%;height:70vh;border:1px solid #dce3ec;border-radius:6px;background:#edf0f3}
.metrics{display:flex;gap:16px;flex-wrap:wrap;margin:8px 0 14px;color:#506379;font-size:13px}
.legend{display:flex;gap:14px;flex-wrap:wrap;font-size:13px;margin:12px 0}.swatch{display:inline-block;width:13px;height:4px;vertical-align:middle;margin-right:5px}
.note{background:#fff8e8;border-left:3px solid #d59724;padding:9px 12px;border-radius:4px}
@media(max-width:980px){.layout{grid-template-columns:1fr}.clusters{max-height:none}.preview{position:static}main{padding:14px}header{padding:20px}}
"""


def _json_script(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":")).replace("<", "\\u003c")


def _percent(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.0f}%"


def document_html(name: str, report: dict, pdf_path: Path, output: Path) -> str:
    pdf_url = quote(os.path.relpath(pdf_path.resolve(), output.resolve()), safe="/.")
    payload = {
        "methods": {method: report["algorithms"][method] for method in METHODS},
        "pages": {str(item["page"]): {key: item[key] for key in ("red_line_count", "green_line_count",
                                                       "blue_line_count", "word_count", "header_families", "eligible")}
                  for item in report["pages"]},
        "overlays": report["overlays"],
        "pdf": pdf_url,
    }
    total = report.get("source_total_pages", report["page_count"])
    skipped = len(report["skipped_pages"])
    limited = report["page_count"] < total
    note = (f"Scanned pages 1–{report['page_count']} of {total}; later pages are outside the configured scan limit."
            if limited else f"Scanned all {total} pages.")
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{escape(name)} · table clustering</title><style>{CSS}</style></head>
<body><header><a href="../index.html">← All documents</a><h1>{escape(name)}</h1><p class="sub">Deterministic table-structure page clusters</p></header>
<main>
<div class="summary-grid"><div class="card"><div class="label">Pages scanned</div><div class="value">{report['page_count']} / {total}</div></div>
<div class="card"><div class="label">Tight clusters</div><div class="value">{report['algorithms']['tight_complete']['cluster_count']}</div></div>
<div class="card"><div class="label">Structure clusters</div><div class="value">{report['algorithms']['structure_single']['cluster_count']}</div></div>
<div class="card"><div class="label">Pages without table evidence</div><div class="value">{skipped}</div></div></div>
<p class="note">{escape(note)} Each page number links to the source PDF. Cluster images show the representative page with detected features.</p>
<div class="bar"><label for="method">Clustering method</label><select id="method">{''.join(f'<option value="{m}">{m.replace('_', ' ')}</option>' for m in METHODS)}</select>
<label for="jump">Find page</label><input id="jump" type="number" min="1" max="{report['page_count']}" placeholder="Page number"><button id="find">Find</button>
<a href="{escape(pdf_url, quote=True)}" target="_blank">Open source PDF ↗</a> <a href="clusters.json">Download JSON</a></div>
<div id="method-metrics" class="metrics"></div><div class="legend"><span><i class="swatch" style="background:#e62323"></i>PDF borders</span><span><i class="swatch" style="background:#009b4b"></i>Word-center alignment</span><span><i class="swatch" style="background:#0f50eb"></i>Repeated headers</span></div>
<div class="layout"><section id="clusters" class="clusters" aria-label="Clusters"></section><section class="panel preview" id="preview"><h2>Choose a page</h2><p class="muted">Select a page number in a cluster to inspect its overlay or source PDF.</p></section></div>
</main><script>const DATA={_json_script(payload)};
const methods=DATA.methods, pages=DATA.pages, overlaySet=new Set(DATA.overlays), methodControl=document.getElementById('method'), clusterRoot=document.getElementById('clusters'), preview=document.getElementById('preview'), metricRoot=document.getElementById('method-metrics');
let currentMethod='tight_complete', currentPage=null;
function el(tag,cls,text){{const node=document.createElement(tag);if(cls)node.className=cls;if(text!==undefined)node.textContent=String(text);return node}}
function pdfPage(number){{return DATA.pdf+'#page='+number}}
function showPage(number){{const info=pages[String(number)];if(!info)return;currentPage=number;preview.replaceChildren();preview.appendChild(el('h2','',`Page ${{number}}`));const meta=el('p','muted',`${{info.red_line_count}} red · ${{info.green_line_count}} green · ${{info.blue_line_count}} blue lines · ${{info.word_count}} words`);preview.appendChild(meta);const link=el('a','','Open page in PDF ↗');link.href=pdfPage(number);link.target='_blank';preview.appendChild(link);const filename=`page_${{String(number).padStart(4,'0')}}_features.png`;if(overlaySet.has(filename)){{const image=el('img');image.src=filename;image.alt=`Feature overlay for page ${{number}}`;preview.appendChild(image)}}else{{const frame=el('iframe');frame.src=pdfPage(number);frame.title=`PDF page ${{number}}`;preview.appendChild(frame)}}document.querySelectorAll('.page-button').forEach(button=>button.classList.toggle('active',Number(button.dataset.page)===number))}}
function render(){{currentMethod=methodControl.value;const outcome=methods[currentMethod];metricRoot.replaceChildren();for(const [label,value] of [['Clusters',outcome.cluster_count],['Silhouette',outcome.mean_silhouette===null?'—':outcome.mean_silhouette],['First 5 coverage',outcome.first_five_cluster_coverage===null?'—':Math.round(outcome.first_five_cluster_coverage*100)+'%'],['Header-family agreement',outcome.header_family_agreement===null?'—':Math.round(outcome.header_family_agreement*100)+'%']]){{metricRoot.appendChild(el('span','',`${{label}}: ${{value}}`))}}clusterRoot.replaceChildren();for(const [clusterId,numbers] of Object.entries(outcome.clusters)){{const card=el('article','card cluster');card.id='cluster-'+clusterId;const top=el('div','cluster-top');top.appendChild(el('h3','',`Cluster ${{clusterId}}`));top.appendChild(el('span','pill',`${{numbers.length}} page${{numbers.length===1?'':'s'}}`));card.appendChild(top);const rep=outcome.representatives[clusterId];card.appendChild(el('p','muted',`Representative: page ${{rep}}`));const filename=`page_${{String(rep).padStart(4,'0')}}_features.png`;if(overlaySet.has(filename)){{const image=el('img');image.src=filename;image.alt=`Representative page ${{rep}} feature overlay`;image.onclick=()=>showPage(rep);card.appendChild(image)}}const box=el('div','pages');for(const number of numbers){{const button=el('button','page-button',number);button.dataset.page=number;button.onclick=()=>showPage(number);box.appendChild(button)}}card.appendChild(box);clusterRoot.appendChild(card)}}if(currentPage!==null&&pages[String(currentPage)])showPage(currentPage)}}
methodControl.onchange=render;document.getElementById('find').onclick=()=>{{const number=Number(document.getElementById('jump').value);if(!pages[String(number)])return;let containing=null;for(const [key,numbers] of Object.entries(methods[currentMethod].clusters))if(numbers.includes(number)){{containing=key;break}}showPage(number);if(containing)document.getElementById('cluster-'+containing)?.scrollIntoView({{behavior:'smooth',block:'start'}})}};document.getElementById('jump').onkeydown=event=>{{if(event.key==='Enter')document.getElementById('find').click()}};render();
</script></body></html>"""


def index_html(rows: list[dict]) -> str:
    table_rows = []
    for row in rows:
        name = escape(row["name"])
        if row["status"] == "complete":
            link = f'<a href="{quote(row["name"], safe="")}/index.html">{name}</a>'
            badge = '<span class="pill">Complete</span>'
            scanned = f"{row['scanned']} / {row['total']}"
            tight = row["tight_clusters"]
            structure = row["structure_clusters"]
            coverage = _percent(row["first_five_coverage"])
        else:
            link = name
            badge = '<span class="pill bad">Pending</span>'
            scanned = f"— / {row['total']}"
            tight = structure = coverage = "—"
        table_rows.append(f"<tr data-name=\"{name.casefold()}\"><td>{link}</td><td>{badge}</td><td>{scanned}</td><td>{tight}</td><td>{structure}</td><td>{coverage}</td></tr>")
    done = sum(row["status"] == "complete" for row in rows)
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Table clustering · all documents</title><style>{CSS}</style></head><body><header><h1>Table clustering</h1><p class="sub">Browse page groups from all source PDFs</p></header>
<main><div class="summary-grid"><div class="card"><div class="label">Documents processed</div><div class="value">{done} / {len(rows)}</div></div>
<div class="card"><div class="label">Pages scanned</div><div class="value">{sum(row.get('scanned',0) for row in rows if row['status']=='complete'):,}</div></div></div>
<p>Clusters combine red PDF borders, green word-center alignments, and blue repeated headers. Open a document to compare five deterministic methods and inspect its representative overlays or any source PDF page.</p>
<p class="note">Documents longer than 500 pages use the first 500, following the scan limit. The scanned/total column shows the coverage for each PDF. These groups are candidates for page sampling; they are not verified table identities.</p>
<div class="bar"><label for="filter">Filter documents</label><input id="filter" type="search" placeholder="Type a document name"><a href="report.md">Comparison report</a><a href="summary.json">Summary JSON</a></div>
<div class="table-wrap"><table><thead><tr><th>Document</th><th>Status</th><th>Pages scanned / total</th><th>Tight clusters</th><th>Structure clusters</th><th>First 5 coverage</th></tr></thead><tbody>{''.join(table_rows)}</tbody></table></div>
</main><script>document.getElementById('filter').oninput=e=>{{const q=e.target.value.toLowerCase();document.querySelectorAll('tbody tr').forEach(row=>row.hidden=!row.dataset.name.includes(q))}}</script></body></html>"""


def build(output_root: Path, sources: list[tuple[str, Path, int]]) -> list[dict]:
    output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, pdf_path, total in sources:
        folder = output_root / name
        result_path = folder / "clusters.json"
        row = {"name": name, "source_pdf": str(pdf_path), "total": total, "status": "pending"}
        if result_path.is_file():
            report = json.loads(result_path.read_text(encoding="utf-8"))
            if report.get("source_pdf") == str(pdf_path.resolve()) and report.get("source_mtime_ns") == pdf_path.stat().st_mtime_ns:
                row.update(status="complete", scanned=report["page_count"],
                           tight_clusters=report["algorithms"]["tight_complete"]["cluster_count"],
                           structure_clusters=report["algorithms"]["structure_single"]["cluster_count"],
                           first_five_coverage=report["algorithms"]["tight_complete"]["first_five_cluster_coverage"])
                (folder / "index.html").write_text(document_html(name, report, pdf_path, folder), encoding="utf-8")
        rows.append(row)
    (output_root / "summary.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    (output_root / "index.html").write_text(index_html(rows), encoding="utf-8")
    return rows


if __name__ == "__main__":
    import argparse
    import pdfplumber

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parents[1] / "data")
    parser.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    sources = []
    for path in sorted([*args.data_dir.glob("*.pdf"), *args.data_dir.glob("*/*.pdf")]):
        name = path.stem if path.parent == args.data_dir else path.parent.name
        with pdfplumber.open(path) as pdf:
            sources.append((name, path.resolve(), len(pdf.pages)))
    rows = build(args.output_root, sources)
    print(f"Built {args.output_root / 'index.html'} ({sum(row['status']=='complete' for row in rows)}/{len(rows)} complete)")
