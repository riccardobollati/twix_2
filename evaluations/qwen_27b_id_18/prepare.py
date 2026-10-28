import json,hashlib,time,subprocess,os
from pathlib import Path
import pdfplumber,pypdf,pypdfium2 as pdfium
W=Path(__file__).resolve().parent
BASE=Path('/Users/riccardo/tesi/code/baseline-qwen-27-b/id_18_upscaled')
PDF=Path('/Users/riccardo/tesi/code/full_dataset/id_18_upscaled.pdf')
def save(name,obj):
 p=W/name;t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(obj,indent=2));t.replace(p)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
r={'status':'failed','accuracy':None,'accuracy_scale':'0_to_1','cost':None,'pages':{'development':[],'evaluation':[],'requested':20,'evaluated':0},'full_document_run':{'status':'not_run','exit_code':None,'reason':None},'candidate':{'successful_pages':0,'empty_failed_pages':[],'rows_or_items':0},'reference':{'source':'Codex-generated local PDF reference','rows_or_items':0,'template_additions':[]},'score_components':{},'errors':[],'artifacts':{},'hashes':{}}
save('evaluation_result.json',r)
with pdfplumber.open(PDF) as full,pdfplumber.open(BASE/'id_18_upscaled.pdf') as dev:
 texts=[p.extract_text() for p in full.pages];dtexts=[p.extract_text() for p in dev.pages]
 mapping=[]
 for t in dtexts:
  matches=[i+1 for i,s in enumerate(texts) if s==t];assert len(matches)==1,matches;mapping.append(matches[0])
 assert len(mapping)==5
 selected=[p for p in range(max(mapping)+1,len(texts)+1) if p not in mapping][:20]
 r['pages'].update(development=mapping,evaluation=selected,full_document_count=len(texts),shortfall=20-len(selected))
 r['hashes'].update(full_pdf=sha(PDF),extractor=sha(BASE/'extract.py'),baseline_template=sha(BASE/'template.json'),development_pdf=sha(BASE/'id_18_upscaled.pdf'))
 save('page_mapping.json',{'development':mapping,'verified_by':'unique exact PDF text equality','evaluation':selected})
 (W/'text').mkdir(exist_ok=True);(W/'images').mkdir(exist_ok=True)
 for n in mapping+selected:
  (W/'text'/f'page_{n:03}.txt').write_text(texts[n-1])
  save(f'text/page_{n:03}_words.json',full.pages[n-1].extract_words())
 save('evaluation_result.json',r)
def run(inp,out,stem):
 start=time.monotonic()
 try:
  p=subprocess.run([__import__('sys').executable,str(BASE/'extract.py'),str(inp),str(out)],capture_output=True,text=True,timeout=300)
  code=p.returncode;stdout=p.stdout;stderr=p.stderr
 except subprocess.TimeoutExpired as e:code=-1;stdout=str(e.stdout or '');stderr='Timeout after 300 seconds '+str(e.stderr or '')
 (W/f'{stem}.stdout.txt').write_text(stdout);(W/f'{stem}.stderr.txt').write_text(stderr)
 return {'exit_code':code,'duration_seconds':time.monotonic()-start,'stdout':str(W/f'{stem}.stdout.txt'),'stderr':str(W/f'{stem}.stderr.txt'),'error':stderr}
fullrun=run(PDF,W/'full_raw.json','full_run');r['full_document_run']={**fullrun,'status':'completed' if fullrun['exit_code']==0 else 'failed','reason':fullrun['error'] or None}
if fullrun['exit_code']!=0:r['errors'].append({'phase':'full_document_run','page':None,'what_happened':'Frozen extractor failed','cause':'unknown','effect':'Full document failed; held-out scoring continues','evidence':fullrun['error']})
save('evaluation_result.json',r)
reader=pypdf.PdfReader(PDF); writer=pypdf.PdfWriter()
for n in selected:writer.add_page(reader.pages[n-1])
with (W/'selected.pdf').open('wb') as f:writer.write(f)
selrun=run(W/'selected.pdf',W/'selected_raw.json','selected_run');save('selected_run.json',selrun)
# Page-local execution required because merged orders lack row-level page provenance.
logs=[];rawpages=[];(W/'page_runs').mkdir(exist_ok=True)
for n in selected:
 writer=pypdf.PdfWriter();writer.add_page(reader.pages[n-1]);inp=W/'page_runs'/f'page_{n:03}.pdf'
 with inp.open('wb') as f:writer.write(f)
 out=W/'page_runs'/f'page_{n:03}_raw.json';log=run(inp,out,f'page_runs/page_{n:03}');log['source_page']=n
 raw=json.loads(out.read_text()) if log['exit_code']==0 else {'orders':[]}
 log['rows_or_items']=sum(len(o['lines']) for o in raw['orders']);logs.append(log);rawpages.append({'source_page':n,'raw':raw})
 if log['exit_code']!=0:r['errors'].append({'phase':'page_extraction','page':n,'what_happened':'Frozen extractor failed on isolated page','cause':'unknown','effect':'Empty prediction for this page','evidence':log['error']})
save('page_log.json',logs);save('candidate_raw.json',{'pages':rawpages})
r['candidate'].update(successful_pages=sum(x['exit_code']==0 for x in logs),empty_failed_pages=[x['source_page'] for x in logs if x['exit_code']!=0],rows_or_items=sum(x['rows_or_items'] for x in logs),representation='physical page-local nodes',selected_run=selrun)
r['artifacts'].update(raw_output=str(W/'candidate_raw.json'),full_raw_output=str(W/'full_raw.json'),page_log=str(W/'page_log.json'))
save('evaluation_result.json',r)
doc=pdfium.PdfDocument(str(PDF))
for n in mapping+selected:doc[n-1].render(scale=1.4).to_pil().save(W/'images'/f'page_{n:03}.png')
print(json.dumps({'pages':r['pages'],'full_run':r['full_document_run'],'candidate':r['candidate']}))
