"""Independent reference transcription from PDF geometry, text and visual QA.
Never reads baseline predictions or scoring artifacts.
"""
import json,re
from pathlib import Path
import pdfplumber
W=Path(__file__).resolve().parent
HEAD=['Date','Number','Investigator','Date Assigned','Racial','Category / Type','Location Of Occurrence','Disposition','Completed','Recorded On Camera']
PERSON=['Complainant:','DOB:','Gender:','Address:','H Phone:']
COMPLAINT=['Type Of Complaint','Description','Complaint Disposition']
OFFICER=['Name','ID No.','Rank','Division','Officer Disposition','Action Taken','Body Cam']
# Boundaries established from printed cell borders and labels, independent of extractor.
MAIN_X=[43,97,155,218,280,307,400,580,635,679,763]
PERSON_X=[108,240,264,316,348,386,424,655,697,763]
def join(words,concat=False):
 lines=[]
 for w in sorted(words,key=lambda w:(w['top'],w['x0'])):
  if not lines or w['top']-lines[-1][0]['top']>3:lines.append([])
  lines[-1].append(w)
 parts=[' '.join(w['text'] for w in sorted(l,key=lambda w:w['x0'])) for l in lines]
 return ('' if concat else ' ').join(parts).strip()
def visible(w,rects):
 # Text under opaque black redaction has no legible reference value.
 return not any(r['x0']<= (w['x0']+w['x1'])/2 <=r['x1'] and r['top']<= (w['top']+w['bottom'])/2 <=r['bottom'] for r in rects)
def cell(words,x0,x1,y0,y1,concat=False):return join([w for w in words if x0<=w['x0']<x1 and y0<=w['top']<y1],concat)
def table_rows(t,words,fields,extend=0):
 rows=[]
 for ri,row in enumerate(t.rows):
  cells=[c for c in row.cells if c]
  assert len(cells)==len(fields),(t.bbox,cells)
  data={}
  for name,c in zip(fields,cells):
   x0,y0,x1,y1=c
   # Extend officer cells past bottom border for wrapped IDs, capped by next row.
   cap=min([w['top'] for w in words if w['text']=='Investigator' and w['top']>y0]+[9999]); limit=min(y1+(extend if name=='ID No.' else 0),cap); limit=min(limit,t.rows[ri+1].bbox[1]) if ri+1<len(t.rows) else limit; data[name]=cell(words,x0-0.4,x1-0.4,y0,limit,concat=name=='ID No.')
  rows.append(data)
 return rows
blocks=[];inventory=[];transcription=[]
expected_records=[3,3,4,3,4,4,3,4,4,4,4,4,4,4,4]
expected_complaints=[9,4,6,7,5,6,6,8,5,7,7,7,5,6,5]
expected_officers=[3,3,5,3,5,4,4,4,5,5,5,4,4,5,5]
with pdfplumber.open('/Users/riccardo/tesi/code/full_dataset/id_05.pdf') as doc:
 for n in range(6,21):
  p=doc.pages[n-1]; words=p.extract_words();black=[r for r in p.rects if r.get('non_stroking_color') in (0,(0,0,0),[0,0,0])]; words=[w for w in words if visible(w,black)]
  tables=p.find_tables(); mains=[t for t in tables if t.bbox[0]<100]; log={'source_page':n,'records':len(mains),'complaint_rows':0,'officer_rows':0,'image_reviewed':True,'text_reviewed':True,'new_structures':[],'record_numbers':[]}
  for i,m in enumerate(mains):
   y=m.bbox[1]; cy=next(w['top'] for w in words if w['text']=='Complainant:' and y<w['top']<m.bbox[3]); main={k:cell(words,MAIN_X[j],MAIN_X[j+1],y,cy-3) for j,k in enumerate(HEAD)}
   date_match=re.search(r'\s+(\d{1,2}/\d{1,2}/\d{4})$',main['Disposition']);
   if date_match: main['Completed']=date_match[1]; main['Disposition']=main['Disposition'][:date_match.start()]
   # Shared printed Recorded On Camera heading: N/A and No shown in separate vertical positions.
   tokens=main['Recorded On Camera'].split();main['Recorded On Camera']=' '.join(tokens[1:]+tokens[:1]) if tokens and tokens[0]=='No' else main['Recorded On Camera']
   person={k:cell(words,PERSON_X[2*j],PERSON_X[2*j+1],cy-2,cy+7) for j,k in enumerate(PERSON)}
   end=mains[i+1].bbox[1]-8 if i+1<len(mains) else p.height-45
   subs=[t for t in tables if t.bbox[0]>200 and m.bbox[3]<t.bbox[1]<end]
   complaints=[];officers=[]
   for t in subs:
    cells=[c for c in t.rows[0].cells if c]; count=len(cells)
    if count==3:complaints+=table_rows(t,words,COMPLAINT)
    elif count==7:officers+=table_rows(t,words,OFFICER,extend=8)
    else:raise ValueError((n,t.bbox,count))
   data=[main,person,complaints,officers]; transcription.append({'page':n,'record_index':i,'data':data});log['record_numbers'].append(main['Number']);log['complaint_rows']+=len(complaints);log['officer_rows']+=len(officers)
   for node,content in enumerate(data):blocks.append({'page':n,'type':'kv' if node==1 else 'table','block_id':f'{node}_{n:02}{i:02}','data':content if node==1 else ([content] if node==0 else content)})
  assert log['records']==expected_records[n-6],log
  assert log['complaint_rows']==expected_complaints[n-6],log
  assert log['officer_rows']==expected_officers[n-6],log
  inventory.append(log)
(W/'reference/raw_blocks.json').write_text(json.dumps(blocks,indent=2));(W/'reference/transcription.json').write_text(json.dumps(transcription,indent=2));(W/'reference/page_inventory.json').write_text(json.dumps(inventory,indent=2))
# Preserve benchmark IDs and field lists: all selected pages use its four structures.
t=json.loads(Path('data/id_05/template.json').read_text());(W/'evaluation_template.json').write_text(json.dumps(t,indent=2))
from src.record_grouping import group_blocks_into_records
from src.eval.convert import to_eval_format
canonical=to_eval_format(group_blocks_into_records(blocks,t['template']));(W/'reference/canonical.json').write_text(json.dumps(canonical,indent=2))
print(json.dumps(inventory,indent=2))
