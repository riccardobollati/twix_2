import json,hashlib,os,time,subprocess
from pathlib import Path
import pdfplumber
W=Path(__file__).resolve().parent
B=Path('/Users/riccardo/tesi/code/baseline-qwen-27-b/id_05')
F=Path('/Users/riccardo/tesi/code/full_dataset/id_05.pdf')
def h(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save(r):
 p=W/'evaluation_result.json'; t=p.with_suffix('.tmp');t.write_text(json.dumps(r,indent=2));os.replace(t,p)
r={'status':'failed','accuracy':None,'accuracy_scale':'0_to_1','cost':None,'pages':{'development':[],'evaluation':[],'requested':20,'evaluated':0},'full_document_run':{'status':'not_run','exit_code':None,'reason':None},'candidate':{'successful_pages':0,'empty_failed_pages':[],'rows_or_items':0},'reference':{'source':'Codex-generated local PDF reference','rows_or_items':0,'template_additions':[]},'score_components':{},'errors':[],'artifacts':{},'hashes':{}}
save(r)
with pdfplumber.open(B/'id_05.pdf') as dev,pdfplumber.open(F) as full:
 mapping=[]
 for d in dev.pages:
  text=d.extract_text(); matches=[i+1 for i,p in enumerate(full.pages) if p.extract_text()==text]; assert len(matches)==1,matches;mapping.append(matches[0])
 pages=list(range(max(mapping)+1,min(max(mapping)+20,len(full.pages))+1))
 r['pages'].update(development=mapping,evaluation=pages,full_document_count=len(full.pages));r['hashes']={k:h(p) for k,p in {'full_pdf':F,'extractor':B/'extract.py','baseline_template':B/'template.json','development_pdf':B/'id_05.pdf'}.items()};save(r)
 for n in [1]+pages:
  p=full.pages[n-1];(W/'reference'/f'page_{n:02}.txt').write_text(p.extract_text(layout=True));(W/'reference'/f'words_{n:02}.json').write_text(json.dumps(p.extract_words(),indent=2));p.to_image(resolution=110).save(W/'renders'/f'page_{n:02}.png')
start=time.monotonic(); run=subprocess.run([str(Path('env/bin/python').resolve()),str(B/'extract.py'),str(F),str(W/'runs/full_raw.json')],capture_output=True,text=True)
(W/'logs/full_stdout.txt').write_text(run.stdout);(W/'logs/full_stderr.txt').write_text(run.stderr)
r['full_document_run']={'status':'completed' if run.returncode==0 else 'failed','exit_code':run.returncode,'duration_seconds':time.monotonic()-start,'reason':run.stderr or None};save(r)
if run.returncode==0:
 raw=json.loads((W/'runs/full_raw.json').read_text());raw['records']=[x for x in raw['records'] if x['page'] in pages];(W/'runs/selected_raw.json').write_text(json.dumps(raw,indent=2));log=[{'source_page':n,'exit_status':0,'records':sum(x['page']==n for x in raw['records']),'rows_or_items':sum(1+len(x['complaints'])+len(x['officers']) for x in raw['records'] if x['page']==n),'error':None} for n in pages];(W/'logs/pages.json').write_text(json.dumps(log,indent=2));r['candidate'].update(successful_pages=len(pages),rows_or_items=sum(x['rows_or_items'] for x in log));save(r)
print(json.dumps(r,indent=2))
