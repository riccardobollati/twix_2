"""Render an independent KV clustering view beside each table view."""
from __future__ import annotations

from html import escape
import json
import os
from pathlib import Path
from urllib.parse import quote

from build_html import CSS, _json_script
from cluster_kv import METHODS


def document_html(name: str, report: dict, pdf_path: Path, folder: Path) -> str:
    payload = {
        "methods": report["algorithms"],
        "pages": {str(entry["page"]): entry for entry in report["pages"]},
        "overlays": report["overlays"],
        "pdf": quote(os.path.relpath(pdf_path.resolve(), folder.resolve()), safe="/."),
    }
    scanned = report["page_count"]
    total = report["source_total_pages"]
    eligible = scanned - len(report["skipped_pages"])
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{escape(name)} · KV clustering</title><style>{CSS}</style></head>
<body><header><a href="../index.html">← All documents</a><h1>{escape(name)} · KV clustering</h1><p class="sub">Independent key/value page groups</p></header>
<main><nav class="bar"><a href="index.html">Table clusters</a><strong>KV clusters</strong></nav>
<div class="summary-grid"><div class="card"><div class="label">Pages scanned</div><div class="value">{scanned} / {total}</div></div>
<div class="card"><div class="label">Pages with repeated KV groups</div><div class="value">{eligible}</div></div>
<div class="card"><div class="label">Confirmed KV groups</div><div class="value">{len(report['families'])}</div></div>
<div class="card"><div class="label">Balanced clusters</div><div class="value">{report['algorithms']['kv_balanced']['cluster_count']}</div></div></div>
<p class="note">Keys recur across distinct pages at similar relative positions. Blue boxes mark stable keys; green boxes mark value slots. Value text is shown for inspection but does not determine cluster membership. Pages without confirmed KV evidence are listed below and do not enter these clusters.</p>
<div class="bar"><label for="method">KV method</label><select id="method">{''.join(f'<option value="{escape(method)}"{" selected" if method == "kv_balanced" else ""}>{method.replace("_", " ")}</option>' for method in METHODS)}</select>
<label for="jump">Find page</label><input id="jump" type="number" min="1" max="{scanned}" placeholder="Page number"><button id="find">Find</button>
<a href="kv_clusters.json">Download KV JSON</a></div>
<div id="metrics" class="metrics"></div><div class="layout"><section id="clusters" class="clusters" aria-label="KV clusters"></section>
<section class="panel preview" id="preview"><h2>Choose a page</h2><p class="muted">Select a cluster page or enter a page number.</p></section></div>
<details><summary>Pages without confirmed KV groups ({len(report['skipped_pages'])})</summary><p>{', '.join(map(str, report['skipped_pages'])) or 'None'}</p></details>
</main><script>const DATA={_json_script(payload)};
const methods=DATA.methods,pages=DATA.pages,overlaySet=new Set(DATA.overlays),select=document.getElementById('method'),clusters=document.getElementById('clusters'),preview=document.getElementById('preview'),metrics=document.getElementById('metrics');
let currentPage=null;
function el(tag,cls,value){{const node=document.createElement(tag);if(cls)node.className=cls;if(value!==undefined)node.textContent=String(value);return node}}
function pdfPage(n){{return DATA.pdf+'#page='+n}}
function showPage(n){{const info=pages[String(n)];if(!info)return;currentPage=n;preview.replaceChildren();preview.appendChild(el('h2','',`Page ${{n}}`));
const pdf=el('a','','Open page in PDF ↗');pdf.href=pdfPage(n);pdf.target='_blank';preview.appendChild(pdf);
const filename=`kv_page_${{String(n).padStart(4,'0')}}.png`;
if(overlaySet.has(filename)){{const img=el('img');img.src=filename;img.alt=`KV overlay for page ${{n}}`;preview.appendChild(img)}}
else{{const frame=el('iframe');frame.src=pdfPage(n);frame.title=`PDF page ${{n}}`;preview.appendChild(frame)}}
if(!info.groups.length)preview.appendChild(el('p','muted','No confirmed KV group on this page.'));
for(const group of info.groups){{const card=el('div','card');card.appendChild(el('h3','',group.family));
for(const pair of group.pairs)card.appendChild(el('p','',`${{pair.key}} → ${{pair.value||'∅'}}`));preview.appendChild(card)}}
document.querySelectorAll('.page-button').forEach(button=>button.classList.toggle('active',Number(button.dataset.page)===n))}}
function render(){{const outcome=methods[select.value];clusters.replaceChildren();metrics.replaceChildren();
for(const [label,value] of [['Clusters',outcome.cluster_count],['Within similarity',outcome.mean_within_similarity??'—'],['First 5 coverage',outcome.first_five_cluster_coverage===null?'—':Math.round(outcome.first_five_cluster_coverage*100)+'%']])metrics.appendChild(el('span','',`${{label}}: ${{value}}`));
for(const [id,numbers] of Object.entries(outcome.clusters)){{const card=el('article','card cluster');card.id='cluster-'+id;
const top=el('div','cluster-top');top.appendChild(el('h3','',`KV cluster ${{id}}`));top.appendChild(el('span','pill',`${{numbers.length}} page${{numbers.length===1?'':'s'}}`));card.appendChild(top);
const rep=outcome.representatives[id];card.appendChild(el('p','muted',`Representative: page ${{rep}}`));
const filename=`kv_page_${{String(rep).padStart(4,'0')}}.png`;if(overlaySet.has(filename)){{const img=el('img');img.src=filename;img.alt=`Representative KV overlay for page ${{rep}}`;img.onclick=()=>showPage(rep);card.appendChild(img)}}
const keys=[...new Set(numbers.flatMap(n=>pages[String(n)].groups.flatMap(g=>g.keys)))];card.appendChild(el('p','muted',keys.join(' · ')));
const buttons=el('div','pages');for(const n of numbers){{const button=el('button','page-button',n);button.dataset.page=n;button.onclick=()=>showPage(n);buttons.appendChild(button)}}card.appendChild(buttons);clusters.appendChild(card)}}
if(!outcome.cluster_count)clusters.appendChild(el('p','note','No recurring KV groups passed the deterministic support and value-variation gates.'));
if(currentPage!==null)showPage(currentPage)}}
select.onchange=render;document.getElementById('find').onclick=()=>{{const n=Number(document.getElementById('jump').value);if(!pages[String(n)])return;showPage(n);for(const [id,numbers] of Object.entries(methods[select.value].clusters))if(numbers.includes(n)){{document.getElementById('cluster-'+id)?.scrollIntoView({{behavior:'smooth',block:'start'}});break}}}};
document.getElementById('jump').onkeydown=event=>{{if(event.key==='Enter')document.getElementById('find').click()}};render();
</script></body></html>"""


def build(folder: Path, name: str, pdf_path: Path, report: dict) -> Path:
    path = folder / "kv.html"
    path.write_text(document_html(name, report, pdf_path, folder), encoding="utf-8")
    return path


def write_comparison(reports: list[dict], output_root: Path) -> Path:
    lines = ["# KV clustering", "", "Independent deterministic page clusters from recurring key phrases, their relative positions, and value-slot geometry. Value contents do not enter similarity.", "",
             f"{len(reports)} PDFs; {sum(r['page_count'] for r in reports):,} scanned pages; {sum(len(r['families']) for r in reports)} confirmed KV groups.", "",
             "| PDF | KV groups | Eligible pages | Balanced clusters | Strict clusters |", "| --- | ---: | ---: | ---: | ---: |"]
    for report in reports:
        lines.append(f"| {report['result_name']} | {len(report['families'])} | "
                     f"{report['page_count'] - len(report['skipped_pages'])} | "
                     f"{report['algorithms']['kv_balanced']['cluster_count']} | "
                     f"{report['algorithms']['kv_strict']['cluster_count']} |")
    path = output_root / "kv_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
