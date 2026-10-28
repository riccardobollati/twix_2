"""Local PDF text transcription, independent of baseline predictions and adapter."""
import re,json,copy,hashlib
from pathlib import Path
W=Path(__file__).resolve().parent
benchmark=Path('/Users/riccardo/tesi/code/twix_2/data/id_18_upscaled/template.json')
template=json.loads(benchmark.read_text())
fields=list(dict.fromkeys(k for n in template['template'] for k in n['fields'] if n['type']=='kv' and k!='Separation'))
label=re.compile('('+ '|'.join(re.escape(k) for k in sorted(fields,key=len,reverse=True))+'):')
dates=['10/30','10/31','11/1','11/2','11/3','11/4','11/5']
rowpattern=re.compile(r'^(\d+) ([A-Z]+) (\$[\d,.]+) ([A-Z]) (\d+) ((?:\d+ ?)+)$')
# Manual inventory from inspection of every rendered source page, independent of candidate.
ranges={6:(12,26),7:(1,11),8:(12,28),9:(1,11),10:(12,29),11:(1,11),12:(12,26),13:(1,11),14:(12,29),15:(1,11),16:(12,26),17:(1,11),18:(12,28),19:(1,11),20:(12,28),21:(1,11),22:(12,25),23:(1,11),24:(12,28),25:(1,11)}
raw=[];inventory=[]
for p in range(6,26):
 lines=(W/'text'/f'page_{p:03}.txt').read_text().splitlines()
 header_end=next(i for i,l in enumerate(lines) if l=='Line Daypart DPT Total')
 short='(Page' in lines[0];header={}
 for text in lines[1:header_end]:
  if text=='Daily Airtime Lines':continue
  matches=list(label.finditer(text))
  assert matches,text
  for i,m in enumerate(matches):header[m.group(1)]=text[m.end():matches[i+1].start() if i+1<len(matches) else None].strip()
 assert len(header)==(9 if short else 33),(p,len(header))
 rows=[];i=header_end+2
 while i<len(lines) and not lines[i].startswith('Total Spots:'):
  daypart=lines[i];m=rowpattern.fullmatch(lines[i+1]);assert m,(p,lines[i:i+3])
  no,code,rate,ct,length,nums=m.groups();numbers=nums.split();assert len(numbers)==8
  program=lines[i+2]
  row={'Line No':no,'Daypart (Program)':daypart+' '+program,'DPT Code':code,'Rate':rate,'C/T':ct,'Len':length,**dict(zip(dates,numbers[:7])),'Total Spots':numbers[7],'Rtg':'','CPP':'','Imp':'','CPM':''}
  assert sum(map(int,numbers[:7]))==int(numbers[7]),(p,no)
  rows.append(row);i+=3
 assert [int(r['Line No']) for r in rows]==list(range(ranges[p][0],ranges[p][1]+1))
 blocks=[{'type':'kv','block_id':f'{2 if short else 0}_{p}','page':p,'data':header},{'type':'table','block_id':f'1_{p}','page':p,'data':rows}]
 daily={};footer={}
 if i<len(lines):
  daily=dict(zip(dates,lines[i].removeprefix('Total Spots:').split()));assert len(daily)==7
  footer_text=lines[i+1];matches=list(label.finditer(footer_text))
  for j,m in enumerate(matches):footer[m.group(1)]=footer_text[m.end():matches[j+1].start() if j+1<len(matches) else None].strip()
  assert len(footer)==5
  blocks += [{'type':'kv','block_id':f'4_{p}','page':p,'data':daily},{'type':'kv','block_id':f'3_{p}','page':p,'data':footer}]
  assert i+2==len(lines)
 raw.extend(blocks)
 inventory.append({'source_page':p,'image':str(W/'images'/f'page_{p:03}.png'),'text':str(W/'text'/f'page_{p:03}.txt'),'image_and_text_inspected':True,'header_kind':'continuation' if short else 'full','header_fields':len(header),'table_rows':len(rows),'line_number_range':ranges[p],'table_fields_per_row':18,'blank_columns':['Rtg','CPP','Imp','CPM'],'single_letter_S_rows':[r['Line No'] for r in rows if r['DPT Code']=='S'],'daily_totals':daily,'footer_fields':len(footer),'other_substantive_blocks':[],'qa':'All rows visually accounted for, three text lines per row; per-row daily spots sum checked; blank metric columns verified in image.'})
# Existing benchmark omitted the separate printed per-date Total Spots row.
addition={'type':'kv','fields':dates,'bid':[4],'child':-1,'node_id':4,'complexity':{'repeated':False}}
template['template'].append(addition)
# Separation is part of the printed Comments value, not a separate label-value field.
template['template'][0]['fields'].remove('Separation')
(W/'evaluation_template.json').write_text(json.dumps(template,indent=2));(W/'reference_raw.json').write_text(json.dumps(raw,indent=2));(W/'page_inventory.json').write_text(json.dumps(inventory,indent=2))
# Canonical physical page records, same representation as frozen candidate; no baseline imports.
records=[]
for p in range(6,26):
 nodes=[]
 for b in raw:
  if b['page']!=p:continue
  d=b['data'];content=[{'key':k,'value':v} for k,v in d.items()] if b['type']=='kv' else {'headers':list(d[0]),'rows':[[{'key':k,'value':v} for k,v in row.items()] for row in d]}
  nodes.append({'id':b['block_id'],'page':p,'type':'key_value' if b['type']=='kv' else 'table','content':content,'relationship':{'parent_id':None}})
 records.append({'record_id':f'page_{p}','nodes':nodes})
(W/'reference_canonical.json').write_text(json.dumps({'records':records},indent=2))
notes={'source':'Codex-generated, model-assisted local PDF reference; not human-verified truth','independence':'Generated only from exact local PDF text and inspected page images, benchmark template descriptions; no candidate or saved reference access. No external model calls.','inspected_pages':list(range(6,26)),'table_row_count':sum(x['table_rows'] for x in inventory),'template_additions':[{'node_id':4,'reason':'Separate printed Total Spots row has seven per-date values; preserve independently from footer amounts','verified_pages':[p for p in range(6,26) if p%2==0]}],'template_clarifications':[{'node_id':0,'field':'Separation','reason':'Appears within the Comments value (Separation: 15/30/45), not an independent printed header label. Whole value retained as Comments.'}],'template_review':'All twenty page structures revisited for added daily-total node; no new date columns or nonblank rating/impression metrics.','coverage':'All printed label/value header fields, all airtime rows, all seven daily spots cells, Total Spots, all four blank metric columns, daily footer totals and all five footer fields included. Document titles and Daily Airtime Lines heading excluded as structural labels.'}
(W/'reference_qa.json').write_text(json.dumps(notes,indent=2))
r=json.loads((W/'evaluation_result.json').read_text());r['reference'].update(rows_or_items=notes['table_row_count'],template_additions=notes['template_additions'],description=notes['source'],physical_nodes=len(raw));r['pages']['evaluated']=20;r['artifacts'].update(reference=str(W/'reference_canonical.json'),raw_reference=str(W/'reference_raw.json'),evaluation_template=str(W/'evaluation_template.json'),page_inventory=str(W/'page_inventory.json'),reference_qa=str(W/'reference_qa.json'));r['hashes'].update(reference=hashlib.sha256((W/'reference_canonical.json').read_bytes()).hexdigest(),evaluation_template=hashlib.sha256((W/'evaluation_template.json').read_bytes()).hexdigest(),benchmark_template=hashlib.sha256(benchmark.read_bytes()).hexdigest())
t=W/'evaluation_result.json.tmp';t.write_text(json.dumps(r,indent=2));t.replace(W/'evaluation_result.json')
print(json.dumps({'reference_rows':notes['table_row_count'],'nodes':len(raw),'inventory':[(x['source_page'],x['table_rows'],x['single_letter_S_rows']) for x in inventory]}))
