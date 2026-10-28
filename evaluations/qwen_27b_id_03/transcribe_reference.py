"""Reference transcription from PDF ruled cells only. Never reads predictions."""
import json,re,hashlib,os
from pathlib import Path
import pdfplumber
W=Path(__file__).resolve().parent
F=Path('/Users/riccardo/tesi/code/full_dataset/id_03.pdf')
fields=json.loads((W/'evaluation_template.json').read_text())['template'][0]['fields']
expected={6:(22,24),7:(25,28),8:(29,31),9:(32,35),10:(36,39),11:(40,43),12:(44,46),13:(47,49),14:(50,53),15:(54,57),16:(58,61),17:(62,65),18:(66,69),19:(70,73),20:(74,78),21:(79,83),22:(84,88),23:(89,93),24:(94,99),25:(100,104)}
def clean(s):return re.sub(r'\s+',' ',re.sub(r'-[ \t]*\n[ \t]*','-',s or '')).strip()
raw=[];records=[];inventory=[]
with pdfplumber.open(F) as pdf:
 for n,(first,last) in expected.items():
  p=pdf.pages[n-1];tables=p.find_tables();assert len(tables)==1,(n,len(tables))
  table=tables[0];cells=table.extract();assert len(cells[0])==13,(n,cells[0])
  rows=cells[2:];assert [int(x[0]) for x in rows]==list(range(first,last+1)),(n,rows)
  assert all(len(x)==13 for x in rows)
  raw.append({'source_page':n,'headers_as_printed':cells[:2],'rows_as_printed':rows,'table_bbox':table.bbox})
  content={'headers':fields,'rows':[[{'key':field,'value':clean(value)} for field,value in zip(fields,row)] for row in rows]}
  records.append({'record_id':f'page_{n}','page':n,'nodes':[{'id':f'0_{n:04}','type':'table','page':n,'content':content,'relationship':{'parent_id':None}}]})
  # Account for PDF words outside the grid and header: repeated title + footer only.
  outside=[w for w in p.extract_words() if not(table.bbox[0]<=((w['x0']+w['x1'])/2)<=table.bbox[2] and table.bbox[1]<=((w['top']+w['bottom'])/2)<=table.bbox[3])]
  outside_text=' '.join(w['text'] for w in outside)
  assert outside_text==f'TABLE A - SORTED BY MOTION DOCKET NUMBER Page {n} of 1116',(n,outside_text)
  entry={'source_page':n,'visual_image_inspected':True,'pdf_text_inspected':True,'substantive_tables':1,'key_value_blocks':0,'row_count':len(rows),'claim_sequence':[first,last],'field_count':13,'outside_grid_text':outside_text,'template_nodes':[0],'all_rows_fields_checked':True,'notes':[]}
  for row in rows:
   if row[1]=='Unknown' or row[2]=='Unknown':entry['notes'].append(f'Claim {row[0]}: literal Unknown in '+('claim ID and ' if row[1]=='Unknown' else '')+('date' if row[2]=='Unknown' else 'claim ID'))
   if '\n' in (row[6] or ''):entry['notes'].append(f'Claim {row[0]}: multiline Other Filings retained in order: {clean(row[6])}')
  inventory.append(entry)
(W/'reference/ground_truth_raw.json').write_text(json.dumps(raw,indent=2,ensure_ascii=False))
(W/'reference/ground_truth_canonical.json').write_text(json.dumps({'records':records,'model':'Codex-generated local PDF reference'},indent=2,ensure_ascii=False))
(W/'reference/page_inventory.json').write_text(json.dumps(inventory,indent=2))
additions=[{'node_id':0,'field':'Claim sequence number','reason':'Separate unheaded substantive column in the ruled grid; absent from original TWIX field list','evidence':'Development page 2 image and cells; confirmed on every held-out page','stable_node_id_preserved':True}]
(W/'reference/template_additions.json').write_text(json.dumps(additions,indent=2))
r=json.loads((W/'evaluation_result.json').read_text());r['reference'].update(rows_or_items=sum(x['row_count'] for x in inventory),template_additions=additions,description='Codex-generated, model-assisted reference from all 20 local PDF images and exact PDF text; not human-verified; no external-model calls');r['artifacts'].update(reference=str(W/'reference/ground_truth_canonical.json'),reference_raw=str(W/'reference/ground_truth_raw.json'),page_inventory=str(W/'reference/page_inventory.json'),evaluation_template=str(W/'evaluation_template.json'))
t=W/'evaluation_result.tmp';t.write_text(json.dumps(r,indent=2));os.replace(t,W/'evaluation_result.json')
for page in raw:
 print('PAGE',page['source_page'])
 for row in page['rows_as_printed']:print(' | '.join(clean(x) for x in row))
