"""Independent PDF-word transcription; no candidate or adapter imports."""
import json,re,os
from pathlib import Path
W=Path(__file__).resolve().parent
def save(x,p):p.write_text(json.dumps(x,indent=2,ensure_ascii=False))
T=json.loads((W/'evaluation_template.json').read_text());fields=next(n['fields'] for n in T['template'] if n['node_id']==2)
base=[f for f in fields if f!='Dimensions - Mass per Metre lb/ft'];extra='Dimensions - Mass per Metre lb/ft';
if extra not in fields:fields.insert(1,extra)
additions=[{'node_id':2,'field':extra,'pages':list(range(9,19)),'evidence':'Imperial table explicitly prints separate lb/ft and kg/m columns; preserve both instead of combining values.'}]
save(T,W/'evaluation_template.json');records=[];blocks=[];inventory=[];transcription=[];previous_des=''
num=re.compile(r'^\d[\d,]*(?:\.\d+)?$')
for page in range(6,24):
 words=json.loads((W/f'reference/words_{page:02}.json').read_text());lines=[]
 for w in sorted(words,key=lambda w:(w['top'],w['x0'])):
  if not lines or abs(lines[-1][0]['top']-w['top'])>2:lines.append([w])
  else:lines[-1].append(w)
 for line in lines:line.sort(key=lambda w:w['x0'])
 imperial=9<=page<=18
 # rows identified independently by dense numeric properties on both halves.
 data_lines=[ln for ln in lines if ln[0]['top']>275 and sum(num.fullmatch(w['text']) is not None for w in ln if w['x0']>90)>=25]
 expected=27 if imperial else 26;rows=[];audit=[]
 for idx,ln in enumerate(data_lines):
  vals=[w for w in ln if w['x0']>90 and num.fullmatch(w['text'])]
  assert len(vals)==expected,(page,ln[0]['top'],len(vals),[w['text'] for w in ln])
  des=' '.join(w['text'] for w in ln if w['x0']<90)
  y=ln[0]['top'];next_y=data_lines[idx+1][0]['top'] if idx+1<len(data_lines) else 840
  tails=[' '.join(w['text'] for w in z if w['x0']<90) for z in lines if y+2<z[0]['top']<(y+26 if imperial else min(next_y,y+18))]
  tails=[s for s in tails if s]
  if des:
   des=' '.join([des]+tails);previous_des=des
  else:des=previous_des
  assert des,(page,y)
  hdrs=[base[0]]+([extra]+base[1:] if imperial else base[1:])
  row={hdrs[0]:des};row.update(zip(hdrs[1:],[v['text'] for v in vals]));rows.append(row);audit.append({'source_page':page,'y':y,'designation':des,'numeric_cells':len(vals),'values':[v['text'] for v in vals]})
 # metadata from right top heading, including continuation of size range.
 kv={}
 for ln in lines:
  right=[w for w in ln if w['x0']>=630 and w['top']<180];s=' '.join(w['text'] for w in right)
  for label in ['Dimensions','Specification','Size Range']:
   if s.startswith(label+' '):kv[label]=s[len(label):].strip()
  if s.startswith(('UC:','MC:')):kv['Size Range']+=' '+s
 assert len(kv)==3,(page,kv)
 nodes=[{'id':f'0_{page}_0','type':'key_value','content':[{'key':k,'value':v} for k,v in kv.items()],'relationship':{'parent_id':None}},{'id':f'2_{page}_0','type':'table','content':{'headers':hdrs,'rows':[[{'key':h,'value':r[h]} for h in hdrs] for r in rows]},'relationship':{'parent_id':None}}]
 records.append({'record_id':f'page_{page}','source_page':page,'nodes':nodes});blocks.extend([{'page':page,'type':'kv','block_id':f'0_{page}','data':kv},{'page':page,'type':'table','block_id':f'2_{page}','data':rows}]);inventory.append({'source_page':page,'tables':1,'kv_blocks':1,'rows':len(rows),'numeric_columns':expected,'fields':hdrs,'metadata':kv,'first_designation':rows[0][hdrs[0]],'last_designation':rows[-1][hdrs[0]],'review':'Page image structure reviewed; every dense numeric row retained including bottom rows and continuation designations.'});transcription.extend(audit)
save({'records':records},W/'reference/canonical.json');save(blocks,W/'reference/raw_blocks.json');save(inventory,W/'reference/page_inventory.json');save(transcription,W/'reference/transcription.json');save({'source':'Codex-generated, model-assisted reference, not human-verified truth','independence':'Only local PDF words and rendered images used; no baseline predictions, scores or existing reference consumed.','images_reviewed':list(range(6,24)),'template_additions':additions,'notes':['All 18 pages have one property table and one Dimensions/Specification/Size Range KV block.','Pages 6-8 HE; pages 9-18 imperial UB/UC; pages 19-23 metric MB/MC.','Separate imperial lb/ft column preserved.','Row-spanning designations propagated to their numerical rows; parenthesized size and split HE suffix included; page continuations carry the open designation.','Running chapter titles and printed page numbers are document provenance; diagram letters are schema illustrations, not extra data records.']},W/'reference/qa_notes.json')
r=json.loads((W/'evaluation_result.json').read_text());r['phase']='reference_completed';r['pages']['evaluated']=18;r['reference'].update(rows_or_items=sum(x['rows']+3 for x in inventory),table_rows=sum(x['rows'] for x in inventory),template_additions=additions);t=W/'result.tmp';save(r,t);os.replace(t,W/'evaluation_result.json');print([(x['source_page'],x['rows'],x['first_designation'],x['last_designation']) for x in inventory])
