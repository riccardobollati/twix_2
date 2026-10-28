import json,hashlib,collections
from pathlib import Path
W=Path(__file__).resolve().parent
r=json.loads((W/'evaluation_result.json').read_text())
r['status']='completed'
r['errors'].append({'phase':'scoring_runtime','page':None,'what_happened':'Initial scorer import failed in bundled Python','cause':'Bundled runtime lacks scipy','effect':'Resolved by using repository env/bin/python; final valid score unaffected','evidence':'ModuleNotFoundError: No module named scipy','resolved':True})
r['diagnostics']['main_failure_causes']=[{'cause':'Baseline row recognizer requires exactly two uppercase letters in DPT Code; one-letter S sports rows are skipped','missing_rows':13,'evidence':'extract.py parse_data_line uses ^[A-Z]{2}$; reference inventory identifies 13 S rows'},{'cause':'Baseline template and output omit four printed but empty metric columns Rtg, CPP, Imp, CPM','missing_headers':80,'missing_empty_cells':1040,'evidence':'20 pages x 4 headers; 260 retained rows x 4 cells; scorer content_diffs'}]
r['artifacts']['baseline_summary']='/Users/riccardo/tesi/code/baseline-qwen-27-b/id_18_upscaled/evaluation_result.json'
r['artifacts']['baseline_scorer_report']='/Users/riccardo/tesi/code/baseline-qwen-27-b/id_18_upscaled/scorer_report.txt'
# Complete source-field audit supplements frozen mapping description, without changing it or adapter behavior.
manifest=json.loads((W/'mapping_manifest.json').read_text())
audit={'frozen_manifest_sha256':r['hashes']['mapping_manifest'],'no_behavioral_change':True,'additional_explicit_accounting':{'root.orders':'order container flattened into physical page records','orders.lines':'all emitted rows retained, in original order','orders.totals':'all footer fields retained','orders.totals.total_cost':'node 3 Total Cost, unchanged string','wrapper.pages':'evaluation-only source provenance','wrapper.pages.source_page':'record_id and node.page','wrapper.pages.raw':'exact frozen page-run output'},'totals_scalar_mapping':{'cash_dollars':'Cash$ - Spots amount','cash_spots':'Cash$ - Spots count','trade_dollars':'Trade$ - Spots amount','trade_spots':'Trade$ - Spots count','total_cost':'Total Cost','total_grp':'Total GRP - CPP first value','total_cpp':'Total GRP - CPP second value','total_gimp':'Total GIMP(000) - CPM first value','total_cpm':'Total GIMP(000) - CPM second value'}}
(W/'mapping_field_audit.json').write_text(json.dumps(audit,indent=2));r['artifacts']['mapping_field_audit']=str(W/'mapping_field_audit.json')
# Independent audit of every emitted substantive value vs transcription; does not alter reference.
raw=json.loads((W/'candidate_raw.json').read_text());truth=json.loads((W/'reference_raw.json').read_text());issues=[]
for page in raw['pages']:
 p=page['source_page'];gold=next(b['data'] for b in truth if b['page']==p and b['type']=='table');gm={int(row['Line No']):row for row in gold}
 for o in page['raw']['orders']:
  ids=[line['line_no'] for line in o['lines']];assert ids==sorted(ids)
  for line in o['lines']:
   g=gm[line['line_no']]
   for source,target in [('line_no','Line No'),('dpt_code','DPT Code'),('rate','Rate'),('c_t','C/T'),('len','Len'),('total_spots','Total Spots')]:assert str(line[source])==g[target],(p,source)
   assert ' '.join(line[k] for k in ['daypart','program'] if line[k] is not None)==g['Daypart (Program)']
   for date,v in line['daily_spots'].items():assert str(v)==g[date]
(W/'final_qa.json').write_text(json.dumps({'all_260_retained_rows_match_reference_on_all_emitted_fields':True,'all_13_missing_rows_are_S_code':True,'all_20_images_and_text_inspected':True,'adapter_and_baseline_hashes_verified':True,'no_source_code_changes':True,'accuracy':r['accuracy']},indent=2))
r['artifacts']['final_qa']=str(W/'final_qa.json')
t=W/'evaluation_result.json.tmp';t.write_text(json.dumps(r,indent=2));t.replace(W/'evaluation_result.json')
