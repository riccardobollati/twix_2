import json,hashlib,os,subprocess,time
from pathlib import Path
import pdfplumber,pypdfium2 as pdfium
W=Path(__file__).resolve().parent
B=Path('/Users/riccardo/tesi/code/baseline-qwen-27-b/new_04')
P=Path('/Users/riccardo/tesi/code/full_dataset/new_04.pdf')
def save(o,p):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(o,indent=2));os.replace(t,p)
def h(p):return hashlib.sha256(p.read_bytes()).hexdigest()
r={'status':'failed','phase':'initialized','accuracy':None,'accuracy_scale':'0_to_1','cost':None,'pages':{'development':[],'evaluation':[],'requested':20,'evaluated':0},'full_document_run':{'status':'not_run','exit_code':None,'reason':None},'candidate':{'successful_pages':0,'empty_failed_pages':[],'rows_or_items':0},'reference':{'source':'Codex-generated local PDF reference','rows_or_items':0,'template_additions':[]},'score_components':{},'errors':[],'artifacts':{},'hashes':{}}
save(r,W/'evaluation_result.json')
with pdfplumber.open(P) as full,pdfplumber.open(B/'new_04.pdf') as dev:
 mapping=[]
 for pg in dev.pages:
  matches=[i+1 for i,q in enumerate(full.pages) if q.extract_text()==pg.extract_text()]
  assert len(matches)==1,matches
  mapping.append(matches[0])
 assert len(mapping)==5
 selected=list(range(max(mapping)+1,min(len(full.pages),max(mapping)+20)+1))
 r['pages'].update(development=mapping,evaluation=selected,total=len(full.pages),shortfall=20-len(selected),source_pdf=str(P),mapping_verification='Exact extracted text match per development page, unique in full PDF')
 r['hashes']={k:h(p) for k,p in {'source_pdf':P,'development_pdf':B/'new_04.pdf','extractor':B/'extract.py','baseline_template':B/'template.json'}.items()}
 save(r,W/'evaluation_result.json')
 start=time.monotonic();proc=subprocess.run([str(Path('env/bin/python').resolve()),str(B/'extract.py'),str(P),str(W/'runs/full_raw.json')],capture_output=True,text=True,timeout=300)
 for k,s in [('stdout',proc.stdout),('stderr',proc.stderr)]: (W/f'logs/full_{k}.txt').write_text(s)
 r['full_document_run']={'status':'completed' if proc.returncode==0 else 'failed','exit_code':proc.returncode,'reason':proc.stderr or None,'duration_seconds':time.monotonic()-start}
 save(r,W/'evaluation_result.json')
 doc=pdfium.PdfDocument(str(P));logs=[];raw=[]
 for n in selected:
  one=pdfium.PdfDocument.new();one.import_pages(doc,[n-1]);one.save(str(W/f'runs/source_{n:02}.pdf'));one.close()
  start=time.monotonic();p=subprocess.run([str(Path('env/bin/python').resolve()),str(B/'extract.py'),str(W/f'runs/source_{n:02}.pdf'),str(W/f'runs/page_{n:02}_raw.json')],capture_output=True,text=True,timeout=60)
  data=json.loads((W/f'runs/page_{n:02}_raw.json').read_text()) if p.returncode==0 else {'sections':[]}
  count=sum(len(s['rows']) for s in data['sections']);raw.append({'source_page':n,'raw':data})
  logs.append({'source_page':n,'exit_code':p.returncode,'rows':count,'blocks':len(data['sections']),'stderr':p.stderr,'stdout':p.stdout,'duration_seconds':time.monotonic()-start})
  pg=full.pages[n-1];(W/f'reference/page_{n:02}.txt').write_text(pg.extract_text(layout=True));save(pg.extract_words(),W/f'reference/words_{n:02}.json')
  doc[n-1].render(scale=1.6).to_pil().save(W/f'renders/page_{n:02}.png')
 # development image and text
 doc[0].render(scale=1.6).to_pil().save(W/'renders/development_01.png')
 (W/'reference/development_01.txt').write_text(full.pages[0].extract_text(layout=True))
 save(raw,W/'runs/selected_raw.json');save(logs,W/'logs/pages.json')
 r['candidate'].update(successful_pages=sum(p['exit_code']==0 for p in logs),rows_or_items=sum(p['rows'] for p in logs),empty_failed_pages=[p['source_page'] for p in logs if p['exit_code']])
 r['phase']='extracted';save(r,W/'evaluation_result.json');print(json.dumps(r['pages']));print(logs)
