"""Frozen, lossless substantive-field adapter; no reference/PDF access."""
import json,sys
MAIN={'date':'Date','number':'Number','investigator':'Investigator','date_assigned':'Date Assigned','racial':'Racial','location_of_occurrence':'Location Of Occurrence','disposition':'Disposition','date_completed':'Completed'}
PERSON={'name':'Complainant:','dob':'DOB:','gender':'Gender:','address':'Address:','phone':'H Phone:'}
COMPLAINT={'type_of_complaint':'Type Of Complaint','description':'Description','disposition':'Complaint Disposition'}
OFFICER={'name':'Name','id_no':'ID No.','rank':'Rank','division':'Division','disposition':'Officer Disposition','action_taken':'Action Taken','body_cam':'Body Cam'}
HEAD=['Date','Number','Investigator','Date Assigned','Racial','Category / Type','Location Of Occurrence','Disposition','Completed','Recorded On Camera']
def value(v): return '' if v is None else str(v)
def table(rows,headers): return {'headers':headers,'rows':[[{'key':k,'value':value(r.get(k,''))} for k in headers] for r in rows]}
def adapt(raw):
 assert set(raw)<= {'report','records'}
 out=[]
 for i,r in enumerate(raw['records']):
  assert set(r)<=set(MAIN)|{'page','category_type','complainant_type','recorded','on_camera','complainant','complaints','officers'}
  main={t:value(r[s]) for s,t in MAIN.items()};main['Category / Type']=' '.join(value(r[k]) for k in ['category_type','complainant_type']).strip();main['Recorded On Camera']=' '.join(value(r[k]) for k in ['recorded','on_camera']).strip()
  nodes=[]
  contents=[table([main],HEAD),[{'key':t,'value':value(r['complainant'][s])} for s,t in PERSON.items()],table([{t:value(row[s]) for s,t in COMPLAINT.items()} for row in r['complaints']],list(COMPLAINT.values())),table([{t:value(row[s]) for s,t in OFFICER.items()} for row in r['officers']],list(OFFICER.values()))]
  for n,c in enumerate(contents):
   if n in (2,3) and not c['rows']: continue
   nodes.append({'id':f'{n}_{i:04}','type':'key_value' if n==1 else 'table','content':c,'page':r['page'],'relationship':{'parent_id':None}})
  out.append({'record_id':f'page_{r["page"]}_record_{i}','page':r['page'],'nodes':nodes})
 return {'records':out}
if __name__=='__main__':
 from pathlib import Path
 sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
 from src.eval.convert import load_json_dedup
 Path(sys.argv[2]).write_text(json.dumps(adapt(load_json_dedup(sys.argv[1])),indent=2))
