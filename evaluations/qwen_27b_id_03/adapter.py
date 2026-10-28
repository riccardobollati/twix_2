"""Frozen evaluation-only adapter. Deterministic; no PDF/reference/network access."""
import json,sys
from pathlib import Path
FIELDS={'claim_no':'Claim sequence number','claim_id':'CLAIM NO.','date_claim_was_filed':'DATE WAS FILED','injured_party':'INJURED PARTY','motion_filer':'MOTION FILER','docket_no':'Docket No.','other_filings':'Filings','comment':'Comments and explanations for rulings'}
HEADERS=list(FIELDS.values())[:-1]+list('ABCDE')+[FIELDS['comment']]
def scalar(v):
 if v is None:return ''
 if isinstance(v,(str,int,float)):return str(v)
 raise TypeError(f'Unexpected scalar {v!r}')
def adapt(raw):
 if set(raw)-{'title','legend','total_pages','claims','evaluation_source_pages'}:raise ValueError('Unmapped root fields')
 by_page={}
 for r in raw['claims']:
  if set(r)!=set(FIELDS)|{'reasons','source_page'}:raise ValueError(f'Unmapped/missing claim fields: {set(r)}')
  page=r['source_page'];assert isinstance(page,int)
  if set(r['reasons'])!=set('ABCDE'):raise ValueError('Unmapped reason fields')
  row={t:scalar(r[s]) for s,t in FIELDS.items() if s!='other_filings'}
  if not isinstance(r['other_filings'],list):raise TypeError('other_filings must be an array')
  row['Filings']=' '.join(scalar(x) for x in r['other_filings'])
  for code,v in r['reasons'].items():
   if not isinstance(v,bool):raise TypeError('Reason flags must be boolean')
   row[code]='X' if v else ''
  by_page.setdefault(page,[]).append(row)
 records=[]
 for page,rows in sorted(by_page.items()):
  content={'headers':HEADERS,'rows':[[{'key':h,'value':row[h]} for h in HEADERS] for row in rows]}
  records.append({'record_id':f'page_{page}','page':page,'nodes':[{'id':f'0_{page:04}','type':'table','content':content,'page':page,'relationship':{'parent_id':None}}]})
 return {'records':records}
if __name__=='__main__':
 sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
 from src.eval.convert import load_json_dedup
 Path(sys.argv[2]).write_text(json.dumps(adapt(load_json_dedup(sys.argv[1])),indent=2,ensure_ascii=False))
