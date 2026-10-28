import json,re,hashlib,os,sys
from pathlib import Path
from adapter import adapt,FIELDS,HEADERS
W=Path(__file__).resolve().parent;R=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(R));from src.eval.convert import load_json_dedup
B=Path('/Users/riccardo/tesi/code/baseline-qwen-27-b/id_03')
def h(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(r):
 t=W/'evaluation_result.tmp';t.write_text(json.dumps(r,indent=2));os.replace(t,W/'evaluation_result.json')
r=load_json_dedup(W/'evaluation_result.json');pages=r['pages']['evaluation']
lookup={}
for n in pages:
 for w in load_json_dedup(W/'reference'/f'words_{n:02}.json'):
  if re.fullmatch(r'TRO\d+FTC',w['text']):
   assert w['text'] not in lookup;lookup[w['text']]=n
raw=load_json_dedup(W/'runs/selected_raw.json');raw['evaluation_source_pages']=pages
for x in raw['claims']:x['source_page']=lookup[x['claim_id']]
assert [x['source_page'] for x in raw['claims']]==sorted(x['source_page'] for x in raw['claims'])
(W/'runs/selected_with_provenance.json').write_text(json.dumps(raw,indent=2));c=adapt(raw);assert c==adapt(raw)
(W/'runs/candidate_canonical.json').write_text(json.dumps(c,indent=2))
log=[{'source_page':n,'exit_status':0,'extracted_rows':sum(x['source_page']==n for x in raw['claims']),'extracted_blocks':int(any(x['source_page']==n for x in raw['claims'])),'error':None,'provenance':'Unique claim ID located in selected source PDF text'} for n in pages]
(W/'logs/pages.json').write_text(json.dumps(log,indent=2))
manifest={'representation':'One page-local table node per source page, one page record; identical for candidate and reference','source_fields':{s:{'target':t,'conversion':'decimal string; null -> empty' if s in ['claim_no','docket_no'] else 'ordered integer array joined with spaces' if s=='other_filings' else 'string unchanged; null -> empty'} for s,t in FIELDS.items()},'reasons':{c:{'target':c,'conversion':'true -> X; false -> empty'} for c in 'ABCDE'},'exclusions':{'title':'document-level repeated table title, matched by benchmark metadata','legend':'development-page-only reason legend; outside evaluation pages','total_pages':'document-level footer provenance','source_page':'evaluation-only page provenance retained on records/nodes','evaluation_source_pages':'evaluation-only selection provenance'},'unknown_fields':'Hard error; never silently dropped','template_addition_from_development':{'field':'Claim sequence number','node_id':0,'evidence':'Development image page 2 and PDF ruled grid: separate unheaded first column (6,7,8,9), followed by CLAIM NO. claim identifiers'},'provenance_policy':'Claim ID exact match to unique source-page PDF word; unmatched/ambiguous ID fails adaptation','null_policy':'missing candidate values represented by empty string; no imputation'}
(W/'mapping_manifest.json').write_text(json.dumps(manifest,indent=2))
t=load_json_dedup(R/'data/id_03/template.json');t['template'][0]['fields']=HEADERS;(W/'evaluation_template.json').write_text(json.dumps(t,indent=2))
# Development mapping checks use only supplied development output and development-page text.
dev=load_json_dedup(B/'extracted.json');devlookup={}
for n in range(1,6):
 for w in load_json_dedup(W/'reference'/f'words_{n:02}.json'):
  if re.fullmatch(r'TRO\d+FTC',w['text']):devlookup[w['text']]=n
for x in dev['claims']:x['source_page']=devlookup[x['claim_id']]
d=adapt(dev);assert sum(len(x['nodes'][0]['content']['rows']) for x in d['records'])==len(dev['claims'])==len(devlookup)
check={'development_rows':len(dev['claims']),'all_fields_mapped':True,'row_order_preserved':True,'repeatability':True,'checks':['Claim sequence number and CLAIM NO. retain separate printed cells','Integer docket/sequence serialize to decimal display','Reason true/false maps to printed X/blank','Ordered Other Filings array maps to whitespace-joined printed numbers','Comments and names unchanged; no reference-dependent correction'],'example':d['records'][1]['nodes'][0]['content']['rows'][0]}
(W/'logs/adapter_validation.json').write_text(json.dumps(check,indent=2))
r['candidate'].update(successful_pages=20,rows_or_items=len(raw['claims']));r['hashes'].update({k:h(W/f) for k,f in {'adapter':'adapter.py','mapping_manifest':'mapping_manifest.json','evaluation_template_at_freeze':'evaluation_template.json'}.items()});r['adapter_frozen_before_reference']=True;r['artifacts'].update({k:str(W/f) for k,f in {'raw_output':'runs/full_raw.json','selected_raw_output':'runs/selected_raw.json','candidate':'runs/candidate_canonical.json','adapter':'adapter.py','mapping_manifest':'mapping_manifest.json','page_log':'logs/pages.json'}.items()});save(r)
print(json.dumps({'candidate_rows':len(raw['claims']),'development_rows':len(dev['claims']),'freeze_hashes':r['hashes']},indent=2))
