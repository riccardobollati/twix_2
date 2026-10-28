import json, hashlib, os, re, subprocess, time, sys
from pathlib import Path
import pdfplumber
from pypdf import PdfReader, PdfWriter
W=Path(__file__).resolve().parent
B=Path('/Users/riccardo/tesi/code/baseline-qwen-27-b/id_03')
F=Path('/Users/riccardo/tesi/code/full_dataset/id_03.pdf')
def h(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(r):
 t=W/'evaluation_result.tmp';t.write_text(json.dumps(r,indent=2));os.replace(t,W/'evaluation_result.json')
r={'status':'failed','accuracy':None,'accuracy_scale':'0_to_1','cost':None,'pages':{'development':[],'evaluation':[],'requested':20,'evaluated':0},'full_document_run':{'status':'not_run','exit_code':None,'reason':None},'candidate':{'successful_pages':0,'empty_failed_pages':[],'rows_or_items':0},'reference':{'source':'Codex-generated local PDF reference','rows_or_items':0,'template_additions':[]},'score_components':{},'errors':[],'artifacts':{},'hashes':{}}
save(r)
with pdfplumber.open(B/'id_03.pdf') as dev,pdfplumber.open(F) as full:
 mapping=[]; proof=[]
 for i,p in enumerate(dev.pages):
  text=p.extract_text();footer=re.search(r'Page (\d+) of (\d+)',text);n=int(footer[1]);assert text==full.pages[n-1].extract_text();mapping.append(n);proof.append({'development_local_page':i+1,'source_page':n,'text_sha256':hashlib.sha256(text.encode()).hexdigest(),'exact_text_match':True})
 pages=list(range(max(mapping)+1,min(max(mapping)+20,len(full.pages))+1))
 r['pages'].update(development=mapping,evaluation=pages,full_document_count=len(full.pages),shortfall=20-len(pages));r['hashes']={k:h(p) for k,p in {'full_pdf':F,'extractor':B/'extract.py','baseline_template':B/'template.json','development_pdf':B/'id_03.pdf'}.items()};save(r)
 (W/'logs/development_mapping.json').write_text(json.dumps(proof,indent=2))
 for n in [1,2,3,4,5]+pages:
  p=full.pages[n-1];(W/'reference'/f'page_{n:02}.txt').write_text(p.extract_text(layout=True));(W/'reference'/f'words_{n:02}.json').write_text(json.dumps(p.extract_words(),indent=2));p.to_image(resolution=100).save(W/'renders'/f'page_{n:02}.png')
reader=PdfReader(F);writer=PdfWriter()
for n in pages:writer.add_page(reader.pages[n-1])
with (W/'runs/selected.pdf').open('wb') as f:writer.write(f)
def run(label,pdf):
 start=time.monotonic();p=subprocess.run([sys.executable,str(B/'extract.py'),str(pdf),str(W/'runs'/f'{label}_raw.json')],capture_output=True,text=True,timeout=3600)
 (W/'logs'/f'{label}_stdout.txt').write_text(p.stdout);(W/'logs'/f'{label}_stderr.txt').write_text(p.stderr)
 return {'status':'completed' if p.returncode==0 and not re.search(r'warning: page \d+ failed:',p.stderr) else 'failed','exit_code':p.returncode,'duration_seconds':time.monotonic()-start,'reason':p.stderr or None}
r['full_document_run']=run('full',F)
if r['full_document_run']['status']=='failed':r['errors'].append({'phase':'full_document_run','page':None,'what_happened':'Frozen full-document extraction failed or skipped pages','cause':r['full_document_run']['reason'] or 'unknown','effect':'Full-document failure retained; held-out evaluation proceeds independently','evidence':str(W/'logs/full_stderr.txt')})
save(r)
r['selected_document_run']=run('selected',W/'runs/selected.pdf');save(r)
print(json.dumps(r,indent=2))
