import json,hashlib,os,importlib.util
from pathlib import Path
W=Path(__file__).resolve().parent;B=Path('/Users/riccardo/tesi/code/baseline-qwen-27-b/new_04');T=Path('/Users/riccardo/tesi/code/final_eval DSV4.1F/new_04/template.json')
bench=json.loads(T.read_text());fields=next(n['fields'] for n in bench['template'] if n['node_id']==2)
spec=importlib.util.spec_from_file_location('baseline',B/'extract.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
keys=['designation']+[c[0] for c in m.COLUMNS];mapping=dict(zip(keys,fields));meta={'standard':'Dimensions','material_specification':'Specification','size_range':'Size Range'}
code='''import json,sys
from decimal import Decimal
MAPPING = '''+repr(mapping)+'''
META = '''+repr(meta)+'''
def value(v):
 if v is None: return ""
 if isinstance(v,(int,float)): return format(Decimal(str(v)),"f")
 return str(v)
def adapt(raw):
 records=[]
 for entry in raw:
  page=entry['source_page'];nodes=[]
  for i,s in enumerate(entry['raw']['sections']):
   kv=[{'key':target,'value':value(s.get(key))} for key,target in META.items() if s.get(key) is not None]
   if kv:nodes.append({'id':f'0_{page}_{i}','type':'key_value','content':kv,'relationship':{'parent_id':None}})
   headers=[]
   for row in s['rows']:
    for key in row:
     target=MAPPING.get(key,key)
     if target not in headers:headers.append(target)
   rows=[]
   for row in s['rows']:
    converted={MAPPING.get(k,k):value(v) for k,v in row.items()}
    rows.append([{'key':h,'value':converted.get(h,'')} for h in headers])
   if rows:nodes.append({'id':f'2_{page}_{i}','type':'table','content':{'headers':headers,'rows':rows},'relationship':{'parent_id':None}})
  records.append({'record_id':f'page_{page}','source_page':page,'nodes':nodes})
 return {'records':records}
if __name__=='__main__':
 from src.eval.convert import load_json_dedup
 result=adapt(load_json_dedup(sys.argv[1]));open(sys.argv[2],'w').write(json.dumps(result,indent=2))
'''
(W/'adapter.py').write_text(code)
manifest={'representation':'One record per physical page, separate metadata KV and physical table nodes; same representation for reference. No cross-page merging.','fields':{**{f'sections.rows.{k}':{'target':v,'conversion':'None to empty printed cell; numbers to fixed decimal strings; strings verbatim'} for k,v in mapping.items()},**{f'sections.{k}':{'target':v,'conversion':'verbatim; omit absent values'} for k,v in meta.items()},**{k:{'disposition':'excluded provenance/schema metadata','reason':v} for k,v in {'source.file':'Input filename','source.page_count':'Input PDF count','sections.pages':'Page provenance, supplied by isolated page wrapper','sections.section_number':'Running section heading number, document-level metadata','sections.title':'Running chapter heading, document-level metadata','sections.columns.key':'Schema description already represented by mapped fields','sections.columns.symbol':'Schema description','sections.columns.label':'Schema description','sections.columns.unit':'Schema description, units represented in headers'}.items()}},'unknown_row_fields':'Preserved verbatim as extra prediction columns','development_validation':'27 columns in reading order; null blank mass cell; decimals including warping constants preserve numeric values. Source field names mapped by semantics despite erroneous symbols D/h/b in baseline schema; printed development header and benchmark node 2 authoritative.'}
(W/'mapping_manifest.json').write_text(json.dumps(manifest,indent=2))
spec=importlib.util.spec_from_file_location('adapter',W/'adapter.py');a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
raw=json.loads((W/'runs/selected_raw.json').read_text());c=a.adapt(raw);assert c==a.adapt(raw)
assert sum(len(n['content']['rows']) for r in c['records'] for n in r['nodes'] if n['type']=='table')==sum(len(s['rows']) for e in raw for s in e['raw']['sections'])
(W/'runs/canonical.json').write_text(json.dumps(c,indent=2));sample=json.loads((B/'extracted.json').read_text());(W/'runs/development_canonical.json').write_text(json.dumps(a.adapt([{'source_page':1,'raw':sample}]),indent=2))
r=json.loads((W/'evaluation_result.json').read_text());r['hashes'].update({k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in {'adapter':W/'adapter.py','mapping_manifest':W/'mapping_manifest.json','benchmark_template':T}.items()});r['phase']='adapter_frozen';tmp=W/'result.tmp';tmp.write_text(json.dumps(r,indent=2));os.replace(tmp,W/'evaluation_result.json');(W/'evaluation_template.json').write_text(json.dumps(bench,indent=2));print('Frozen adapter and mapping manifest')
