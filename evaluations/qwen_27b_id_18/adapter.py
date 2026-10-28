"""Frozen deterministic raw baseline -> physical page-local scorer records."""
import json,sys
HEADER={'order_id':'OrderID','status':'Status','agency':'Agency','office':'Office','client':'Client','product':'Product','piggyback_product':'Piggyback Product','estimate':'Estimate','primary_demo':'Primary Demo','demo_2':'Demo 2','demo_3':'Demo 3','grp':'GRP','cpp':'CPP','gimp':'GIMP','cpm':'CPM','comments':'Comments','station':'Station','order_type':'Order Type','flight_start':'Flight Start','flight_end':'Flight End','cancel_date':'Cancel Date','hiatus':'Hiatus','received_in_eport':'Received in ePort','local_national':'Local/National','rep_office':'Rep Office','ae':'AE','phone':'Phone','market_share':'MarketShare','airtime_dollars':'Airtime Dollars','interactive_dollars':'Interactive Dollars','non_airtime_dollars':'Non-Airtime Dollars','total_dollars':'Total Dollars','total_spots':'Total Spots'}
SHORT={'order_id','agency','station','client','product','estimate','flight_start','flight_end','received_in_eport'}
LINE={'line_no':'Line No','dpt_code':'DPT Code','rate':'Rate','c_t':'C/T','len':'Len','total_spots':'Total Spots'}
FOOT={'cash_dollars':('Cash$ - Spots','cash_spots'),'trade_dollars':('Trade$ - Spots','trade_spots'),'total_cost':('Total Cost',None),'total_grp':('Total GRP - CPP','total_cpp'),'total_gimp':('Total GIMP(000) - CPM','total_cpm')}
def val(v):return '' if v is None else str(v)
def kv(n,p,d):return {'id':f'{n}_{p}','type':'key_value','page':p,'content':[{'key':k,'value':val(v)} for k,v in d.items()],'relationship':{'parent_id':None}}
def table(n,p,rows):
 headers=list(dict.fromkeys(k for r in rows for k in r))
 return {'id':f'{n}_{p}','type':'table','page':p,'content':{'headers':headers,'rows':[[{'key':k,'value':val(r.get(k))} for k in headers] for r in rows]},'relationship':{'parent_id':None}}
def adapt(raw):
 records=[]
 for entry in raw['pages']:
  p=entry['source_page'];nodes=[]
  for o in entry['raw'].get('orders',[]):
   short=o.get('office') is None and o.get('status') is None
   d={('Order ID' if short and k=='order_id' else target):val(o.get(k)) for k,target in HEADER.items() if not short or k in SHORT or o.get(k) is not None}
   for k,v in o.items():
    if k not in HEADER and k not in {'flight_dates','lines','totals','pages'}:d['extra:'+k]=json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else val(v)
   nodes.append(kv(2 if short else 0,p,d))
   rows=[]
   for line in o.get('lines',[]):
    row={'Line No':val(line.get('line_no')),'Daypart (Program)':' '.join(val(line.get(k)) for k in ['daypart','program'] if line.get(k) is not None),'DPT Code':val(line.get('dpt_code')),'Rate':val(line.get('rate')),'C/T':val(line.get('c_t')),'Len':val(line.get('len'))}
    row.update({k:val(v) for k,v in line.get('daily_spots',{}).items()});row['Total Spots']=val(line.get('total_spots'))
    for k,v in line.items():
     if k not in set(LINE)|{'daypart','program','daily_spots'}:row['extra:'+k]=json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else val(v)
    rows.append(row)
   if rows:nodes.append(table(1,p,rows))
   t=o.get('totals')
   if t is not None:
    if t.get('daily_spots'):nodes.append(kv(4,p,t['daily_spots']))
    footer={}
    for k,(target,other) in FOOT.items():
     if k in t or (other and other in t):footer[target]=val(t.get(k))+(' - '+val(t.get(other)) if other else '')
    known={'daily_spots'}|set(FOOT)|{x[1] for x in FOOT.values()}
    footer.update({'extra:'+k:val(v) for k,v in t.items() if k not in known})
    if footer:nodes.append(kv(3,p,footer))
  records.append({'record_id':f'page_{p}','nodes':nodes})
 return {'records':records}
if __name__=='__main__':
 with open(sys.argv[1]) as f:raw=json.load(f)
 with open(sys.argv[2],'w') as f:json.dump(adapt(raw),f,indent=2,ensure_ascii=False)
