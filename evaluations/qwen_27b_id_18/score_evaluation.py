import sys,json,hashlib,collections
from pathlib import Path
W=Path(__file__).resolve().parent;ROOT=W.parents[1];sys.path.insert(0,str(ROOT))
from src.eval.convert import load_json_dedup
from src.eval.score import score
from src.eval.render import render
r=load_json_dedup(W/'evaluation_result.json')
def save():
 t=W/'evaluation_result.json.tmp';t.write_text(json.dumps(r,indent=2));t.replace(W/'evaluation_result.json')
try:
 candidate=load_json_dedup(W/'candidate_canonical.json');reference=load_json_dedup(W/'reference_canonical.json')
 reference['doc_name']='id_18_upscaled pages 6-25';reference['model']='Codex-generated local PDF reference';candidate['model']='baseline-qwen-27-b frozen extract.py'
 scored=score(reference,candidate,reference='model_reference')
 (W/'scorer_report.json').write_text(json.dumps(scored,indent=2))
 (W/'scorer_report.txt').write_text(render(scored))
 r.update(status='partial' if r['errors'] else 'completed',accuracy=scored['accuracy'],score_components=scored['components'])
 r['artifacts'].update(scorer_report=str(W/'scorer_report.txt'),scorer_json=str(W/'scorer_report.json'))
 r['hashes']['scorer_sources']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'src/eval').glob('*.py'))}
 r['hashes']['scorer_source_hash']=hashlib.sha256(''.join(k+v for k,v in r['hashes']['scorer_sources'].items()).encode()).hexdigest()
 counts=collections.Counter(d['reason'] for d in scored['content_diffs'])
 r['diagnostics']={'reason_counts':dict(counts),'missing_rows':sum(d['reason']=='R-ROW-MISSING' for d in scored['content_diffs']),'candidate_rows':260,'reference_rows':273,'execution_failures':0,'scoring_representation':'one record per physical source page, page-local header/table/total nodes; no cross-page logical merging','reference_method':'Codex local image and text inspection with independent transcription helper; no external API calls'}
 # Assert frozen assets and baseline untouched before completing.
 for name,file in [('adapter','adapter.py'),('mapping_manifest','mapping_manifest.json')]:assert hashlib.sha256((W/file).read_bytes()).hexdigest()==r['hashes'][name]
 base=Path('/Users/riccardo/tesi/code/baseline-qwen-27-b/id_18_upscaled')
 for name,file in [('extractor','extract.py'),('baseline_template','template.json'),('development_pdf','id_18_upscaled.pdf')]:assert hashlib.sha256((base/file).read_bytes()).hexdigest()==r['hashes'][name]
except Exception as e:
 r['status']='failed';r['accuracy']=None;r['errors'].append({'phase':'scoring','page':None,'what_happened':'Evaluation scoring failed','cause':type(e).__name__,'effect':'No valid accuracy','evidence':str(e)});save();raise
save()
print(json.dumps({'accuracy':r['accuracy'],'components':r['score_components'],'diagnostics':r['diagnostics']}))
