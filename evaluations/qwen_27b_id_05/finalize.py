import json,hashlib,os,collections
from pathlib import Path
from src.eval.convert import load_json_dedup
from src.eval.score import score_paths
from src.eval.render import render_to_file
W=Path(__file__).resolve().parent
def h(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(r):
 tmp=W/'result.tmp';tmp.write_text(json.dumps(r,indent=2));os.replace(tmp,W/'evaluation_result.json')
r=load_json_dedup(W/'evaluation_result.json');inventory=load_json_dedup(W/'reference/page_inventory.json');gold=load_json_dedup(W/'reference/canonical.json');cand=load_json_dedup(W/'runs/canonical.json')
assert len(gold['records'])==sum(p['records'] for p in inventory)==56
assert all(len(x['nodes'])==4 for x in gold['records'])
assert r['hashes']['adapter']==h(W/'adapter.py');assert r['hashes']['mapping_manifest']==h(W/'mapping_manifest.json')
r['pages']['evaluated']=len(inventory);r['pages']['shortfall']=20-len(inventory);r['pages']['source_pdf']=str(Path('/Users/riccardo/tesi/code/full_dataset/id_05.pdf'));r['pages']['printed_footer_total']=74;r['pages']['note']='Available source PDF has 20 physical pages; printed footer refers to original 74-page report. Evaluate all 15 available pages after development pages.'
r['reference'].update(records=len(gold['records']),nodes=sum(len(x['nodes']) for x in gold['records']),complaint_rows=sum(p['complaint_rows'] for p in inventory),officer_rows=sum(p['officer_rows'] for p in inventory),rows_or_items=sum(p['records']*2+p['complaint_rows']+p['officer_rows'] for p in inventory),template_additions=[])
qa={'source':'Codex-generated, model-assisted local PDF image and text reference; not human-verified truth','reviewed_pages':list(range(6,21)),'independence':'Reference transcription reads PDF geometry/text only; no candidate, adapted output, scores or saved reference used as transcription input.','structures':'Every record: yellow complaint header, complainant KV row, complaints table, officers table. All fit existing benchmark nodes 0-3.','coverage':inventory,'format_notes':['Category / Type includes formal/inquiry/admin review and citizen/internal in reading order.','Recorded On Camera represented as N/A No to match deterministic frozen adapter concatenation.','Wrapped officer ID suffixes below border retained and concatenated without spaces.','Partial surviving phone punctuation remains literal; fully redacted text is blank.','Print anomalies and abbreviated/clipped values retained literally; no inferred names or completions.','Complaint/Officer sequence labels excluded as benchmark record-level metadata; all array rows remain ordered.'],'qa_before_scoring':'Verified printed complaint/officer counts independently against all images. Corrected reference helper before scoring to cap wrapped ID cells at following headers and move completed date fragments to their printed field. No adapter revisions.'}
(W/'reference/qa_notes.json').write_text(json.dumps(qa,indent=2));r['hashes'].update(reference=h(W/'reference/canonical.json'),evaluation_template=h(W/'evaluation_template.json'));r['phase']='reference_completed';save(r)
try:
 report=score_paths(W/'reference/canonical.json',W/'runs/canonical.json',reference='model_reference');files=sorted(Path('src/eval').glob('*.py'));digest=hashlib.sha256()
 for f in files:digest.update(str(f).encode()+b'\0'+f.read_bytes())
 report.update(doc_name='id_05 Qwen 27B baseline pages 6-20',candidate='Frozen baseline extractor with evaluation-only adapter',reference_model='Codex local PDF image/text review',eval_version='twix-eval-source-sha256:'+digest.hexdigest());(W/'scorer_result.json').write_text(json.dumps(report,indent=2));render_to_file(report,W/'scorer_report.txt')
 r['status']='completed';r['accuracy']=report['accuracy'];r['score_components']={k:v for k,v in report.items() if k not in ['records','alignment','content_diffs','notes','doc_name','candidate','reference_model']};r['eval_version']=report['eval_version'];r['hashes']['scorer_source']=digest.hexdigest();r['phase']='scored'
except Exception as e:
 r['errors'].append({'phase':'scoring','page':None,'what_happened':str(e),'cause':type(e).__name__,'effect':'No valid score','evidence':repr(e)});save(r);raise
r['artifacts']={k:str(W/p) for k,p in {'full_raw_output':'runs/full_raw.json','selected_raw_output':'runs/selected_raw.json','candidate':'runs/canonical.json','adapter':'adapter.py','mapping_manifest':'mapping_manifest.json','reference':'reference/canonical.json','raw_reference':'reference/raw_blocks.json','page_inventory':'reference/page_inventory.json','qa_notes':'reference/qa_notes.json','transcription':'reference/transcription.json','evaluation_template':'evaluation_template.json','scorer_report':'scorer_report.txt','scorer_json':'scorer_result.json','page_log':'logs/pages.json','stdout':'logs/full_stdout.txt','stderr':'logs/full_stderr.txt'}.items()};r['artifacts']['baseline_summary']='/Users/riccardo/tesi/code/baseline-qwen-27-b/id_05/evaluation_result.json';r['artifacts']['baseline_report']='/Users/riccardo/tesi/code/baseline-qwen-27-b/id_05/scorer_report.txt';save(r)
print(json.dumps({'accuracy':r['accuracy'],'report_keys':list(report),'reference':r['reference']},indent=2))
