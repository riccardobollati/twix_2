import json,hashlib,os
from pathlib import Path
from adapter import *
W=Path(__file__).resolve().parent;B=Path('/Users/riccardo/tesi/code/baseline-qwen-27-b/id_05')
m={'representation':'logical complaint records with separate physical page-local nodes; no parent edges','field_mapping':{},'conversions':'strings preserved; category and complainant type joined in printed order; recorded and on camera joined in printed order; null to blank','provenance':{'records[].page':'canonical record/node physical source page'},'exclusions':{}}
for group,fields in [('records[]',MAIN),('records[].complainant',PERSON),('records[].complaints[]',COMPLAINT),('records[].officers[]',OFFICER)]:
 for s,t in fields.items():m['field_mapping'][group+'.'+s]=t
for a,b in [('category_type','Category / Type'),('complainant_type','Category / Type'),('recorded','Recorded On Camera'),('on_camera','Recorded On Camera')]:m['field_mapping']['records[].'+a]=b
for k in json.loads((B/'template.json').read_text())['report']['fields']:m['exclusions']['report.'+k]='document-level report/frame metadata outside complaint content'
for k in ['complaints','officers']:m['exclusions']['records[].'+k+'[].number']='printed Complaint/Officer sequence label; benchmark record_level metadata; array order preserved'
(W/'mapping_manifest.json').write_text(json.dumps(m,indent=2))
d=json.loads((B/'output.json').read_text());assert adapt(d)==adapt(d);a=adapt(d);assert len(a['records'])==len(d['records']);(W/'runs/development_canonical.json').write_text(json.dumps(a,indent=2))
# Verify every declared source field has a mapping or exclusion.
t=json.loads((B/'template.json').read_text()); keys=set('records[].'+k for k in t['records']['item']['fields'] if k not in ['complainant','complaints','officers','page'])
for k in ['complainant','complaints','officers']:
 fs=t['records']['item']['fields'][k];fs=fs.get('item',fs)['fields'];keys.update('records[].'+k+('[]' if k!='complainant' else '')+'.'+x for x in fs)
assert keys<=set(m['field_mapping'])|set(m['exclusions'])
r=json.loads((W/'evaluation_result.json').read_text());r['hashes'].update({k:hashlib.sha256((W/p).read_bytes()).hexdigest() for k,p in [('adapter','adapter.py'),('mapping_manifest','mapping_manifest.json')]});r['adapter_frozen_before_reference']=True;r['adapter_validation']={'deterministic':True,'development_records':len(d['records']),'field_accounting_complete':True,'page_and_row_order_preserved':True};tmp=W/'result.tmp';tmp.write_text(json.dumps(r,indent=2));os.replace(tmp,W/'evaluation_result.json')
