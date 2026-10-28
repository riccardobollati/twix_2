import sys,json,hashlib,os,collections
from pathlib import Path
W=Path(__file__).resolve().parent;R=W.parents[1];sys.path.insert(0,str(R))
from src.eval.convert import load_json_dedup
from src.eval.score import score
from src.eval.render import render
from adapter import adapt

def h(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(r):
 t=W/'evaluation_result.tmp';t.write_text(json.dumps(r,indent=2));os.replace(t,W/'evaluation_result.json')
r=load_json_dedup(W/'evaluation_result.json')
assert h(W/'adapter.py')==r['hashes']['adapter'];assert h(W/'mapping_manifest.json')==r['hashes']['mapping_manifest']
assert h(W/'evaluation_template.json')==r['hashes']['evaluation_template_at_freeze']
c=load_json_dedup(W/'runs/candidate_canonical.json');g=load_json_dedup(W/'reference/ground_truth_canonical.json')
assert c==adapt(load_json_dedup(W/'runs/selected_with_provenance.json'))
assert [x['page'] for x in g['records']]==r['pages']['evaluation']
assert set(x['page'] for x in c['records'])<=set(r['pages']['evaluation'])
scorer_files=sorted((R/'src/eval').glob('*.py'));source_hashes={p.name:h(p) for p in scorer_files};r['scorer']={'name':'TWIX existing scorer unchanged','source_hashes':source_hashes,'version':hashlib.sha256(json.dumps(source_hashes,sort_keys=True).encode()).hexdigest(),'representation':'one page-local physical table per page record in both candidate and reference','reference':'model_reference','headline_weights_effective':{'NodeMatchF1':0.5,'ContentScore':0.5},'structure_applicable':False}
s=score(g,c,reference='model_reference');(W/'scorer_result.json').write_text(json.dumps(s,indent=2));(W/'scorer_report.txt').write_text(render(s))
r.update(status='partial' if r['errors'] else 'completed',accuracy=s['accuracy'],score_components=s['components']);r['pages']['evaluated']=20
r['hashes'].update(reference=h(W/'reference/ground_truth_canonical.json'),reference_raw=h(W/'reference/ground_truth_raw.json'),candidate=h(W/'runs/candidate_canonical.json'),evaluation_template=h(W/'evaluation_template.json'),scorer=r['scorer']['version'])
r['artifacts'].update(scorer_report=str(W/'scorer_report.txt'),scorer_result=str(W/'scorer_result.json'),qa_notes=str(W/'reference/qa_notes.md'),coverage_checks=str(W/'reference/coverage_checks.json'),transcription=str(W/'reference/transcription.txt'),full_stdout=str(W/'logs/full_stdout.txt'),full_stderr=str(W/'logs/full_stderr.txt'),selected_pdf=str(W/'runs/selected.pdf'),development_mapping=str(W/'logs/development_mapping.json'),adapter_validation=str(W/'logs/adapter_validation.json'))
# Evidence-based diagnostics only after reference has been frozen and scored.
raw=load_json_dedup(W/'runs/selected_with_provenance.json');candidate_nos={x['claim_no'] for x in raw['claims']};missing=[]
for rec in g['records']:
 for row in rec['nodes'][0]['content']['rows']:
  values={x['key']:x['value'] for x in row}
  if int(values['Claim sequence number']) not in candidate_nos:missing.append({'source_page':rec['page'],'claim_sequence':values['Claim sequence number'],'claim_id':values['CLAIM NO.']})
r['diagnostics']={'missing_claim_rows':missing,'reason_counts':dict(collections.Counter(d.get('reason','unknown') for d in s['content_diffs'])),'full_document_rows':len(load_json_dedup(W/'runs/full_raw.json')['claims']),'page_execution_failures':0}
r['artifacts']['baseline_summary']='/Users/riccardo/tesi/code/baseline-qwen-27-b/id_03/evaluation_result.json';r['artifacts']['baseline_scorer_report']='/Users/riccardo/tesi/code/baseline-qwen-27-b/id_03/scorer_report.txt'
assert r['cost'] is None
save(r)
print(json.dumps({'status':r['status'],'accuracy':r['accuracy'],'components':r['score_components'],'diagnostics':r['diagnostics'],'content_diffs':len(s['content_diffs'])},indent=2))
