import json,hashlib,os,subprocess,time,collections
from pathlib import Path
import pypdfium2 as pdfium
from src.eval.convert import load_json_dedup
from src.eval.score import score_paths
from src.eval.render import render_to_file
W=Path(__file__).resolve().parent;B=Path('/Users/riccardo/tesi/code/baseline-qwen-27-b/new_04');P=Path('/Users/riccardo/tesi/code/full_dataset/new_04.pdf')
def h(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(x,p):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2,ensure_ascii=False));os.replace(t,p)
r=load_json_dedup(W/'evaluation_result.json')
assert h(W/'adapter.py')==r['hashes']['adapter'];assert h(W/'mapping_manifest.json')==r['hashes']['mapping_manifest']
# The successful subset still merges rows and cannot attribute them to physical pages.
full=pdfium.PdfDocument(str(P));part=pdfium.PdfDocument.new();part.import_pages(full,[n-1 for n in r['pages']['evaluation']]);part.save(str(W/'runs/selected.pdf'))
start=time.monotonic();proc=subprocess.run([str(Path('env/bin/python').resolve()),str(B/'extract.py'),str(W/'runs/selected.pdf'),str(W/'runs/selected_merged_raw.json')],capture_output=True,text=True,timeout=300)
(W/'logs/selected_stdout.txt').write_text(proc.stdout);(W/'logs/selected_stderr.txt').write_text(proc.stderr)
r['selected_document_run']={'exit_code':proc.returncode,'duration_seconds':time.monotonic()-start,'page_isolation_reason':'Full and selected output merge section rows across pages and have only a section-level page list. Isolated frozen-script runs supply unambiguous page provenance.','order_note':'Page-isolated runs were completed before the merged selected-PDF confirmation run; both use identical frozen script.'}
inv=load_json_dedup(W/'reference/page_inventory.json');logs=load_json_dedup(W/'logs/pages.json');r['candidate']['successful_execution_pages']=18;r['candidate']['pages_with_content']=sum(x['rows']>0 for x in logs);r['candidate']['empty_prediction_pages']=[x['source_page'] for x in logs if x['rows']==0]
for x in logs:
 if x['rows']==0:
  r['errors'].append({'phase':'extraction_coverage','page':x['source_page'],'what_happened':'Successful process produced no sections or rows for a substantive table page','cause':'parse_page requires a Section/Dimensions/Properties banner and designation-led numeric rows in the five-page HE/IPE layout; imperial/metric tables use different headers and row-spanning designations','effect':'Empty prediction contributes missing table and metadata content to the held-out score','evidence':f"exit_code=0, sections=0, rows=0; local reference has {next(v['rows'] for v in inv if v['source_page']==x['source_page'])} table rows"})
r['phase']='reference_verified';r['reference']['table_rows']=sum(x['rows'] for x in inv);r['reference']['rows_or_items']=r['reference']['table_rows']+54
qa=load_json_dedup(W/'reference/qa_notes.json');qa['qa_checks']={'numeric_cell_coverage':'Every dense table row contains exactly 26 (metric/HE) or 27 (imperial) printed numeric cells; assertions cover all selected rows.','counts_per_page':{str(x['source_page']):x['rows'] for x in inv},'designation_wraps':'HE split mass suffixes and imperial parenthesized metric-size continuation retained. Three-line W24/W30/W33 designations verified and transcription helper corrected before scoring.','reference_addition_review':'All pages 9-18 reviewed for added lb/ft field; every selected page remains in reference.','development_format_validation':'Development image confirms semantic field order and decimal rendering including warping constants. Adapter frozen before independent transcription and unchanged.'};save(qa,W/'reference/qa_notes.json')
r['hashes'].update(reference=h(W/'reference/canonical.json'),evaluation_template=h(W/'evaluation_template.json'));save(r,W/'evaluation_result.json')
try:
 report=score_paths(W/'reference/canonical.json',W/'runs/canonical.json',reference='model_reference')
 digest=hashlib.sha256()
 for p in sorted(Path('src/eval').glob('*.py')):digest.update(str(p).encode()+b'\0'+p.read_bytes())
 report.update(doc_name='new_04 Qwen 27B baseline physical pages 6-23',candidate='Frozen Qwen 27B baseline with evaluation adapter',reference_model='Codex local PDF image/text reference',eval_version='twix-eval-source-sha256:'+digest.hexdigest());save(report,W/'scorer_result.json');render_to_file(report,W/'scorer_report.txt')
 r.update(status='completed' if r['full_document_run']['exit_code']==0 and all(x['exit_code']==0 for x in logs) else 'partial',accuracy=report['accuracy'],phase='scored',eval_version=report['eval_version'])
 r['score_components']={k:v for k,v in report.items() if k not in ['records','alignment','content_diffs','notes','doc_name','candidate','reference_model']};r['hashes']['scorer_source']=digest.hexdigest()
except Exception as e:
 r['errors'].append({'phase':'scoring','page':None,'what_happened':str(e),'cause':type(e).__name__,'effect':'No valid score','evidence':repr(e)})
r['artifacts']={k:str(W/p) for k,p in {'full_raw_output':'runs/full_raw.json','selected_merged_raw_output':'runs/selected_merged_raw.json','selected_raw_output':'runs/selected_raw.json','candidate':'runs/canonical.json','adapter':'adapter.py','mapping_manifest':'mapping_manifest.json','reference':'reference/canonical.json','raw_reference':'reference/raw_blocks.json','page_inventory':'reference/page_inventory.json','qa_notes':'reference/qa_notes.json','transcription':'reference/transcription.json','evaluation_template':'evaluation_template.json','scorer_report':'scorer_report.txt','scorer_json':'scorer_result.json','page_log':'logs/pages.json','full_stdout':'logs/full_stdout.txt','full_stderr':'logs/full_stderr.txt'}.items()}
r['artifacts'].update(baseline_summary=str(B/'evaluation_result.json'),baseline_report=str(B/'scorer_report.txt'));r['error_counts']=dict(collections.Counter(x['phase'] for x in r['errors']));r['notes']=['18 of requested 20 pages available; physical pages 6-23 scored.','Completed denotes successful execution and valid scoring; silent empty predictions are retained as accuracy failures.','Cost intentionally null for manual entry.','Reference is Codex-generated and model-assisted, not human-verified.','No baseline artifacts modified other than the required summary/report copies.']
assert r['cost'] is None;assert r['hashes']['extractor']==h(B/'extract.py');assert r['hashes']['baseline_template']==h(B/'template.json');save(r,W/'evaluation_result.json');print(json.dumps({'status':r['status'],'accuracy':r['accuracy'],'reference_rows':r['reference']['table_rows'],'candidate_rows':r['candidate']['rows_or_items'],'components':r['score_components']},indent=2))
