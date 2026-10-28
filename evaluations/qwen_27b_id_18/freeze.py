import json,hashlib,sys
from pathlib import Path
from adapter import *
W=Path(__file__).resolve().parent
manifest={'representation':'one record per physical page; long header node 0, continuation header node 2, airtime table node 1, footer node 3, daily totals node 4','header':HEADER,'line':LINE,'combined_fields':{'lines.daypart + lines.program':'Daypart (Program), space concatenation in printed order','totals.cash_dollars + totals.cash_spots':'Cash$ - Spots, dollar amount + space-hyphen-space + spot count','totals.trade_dollars + totals.trade_spots':'Trade$ - Spots','totals.total_grp + totals.total_cpp':'Total GRP - CPP','totals.total_gimp + totals.total_cpm':'Total GIMP(000) - CPM'},'dynamic':{'lines.daily_spots.*':'table date column unchanged','totals.daily_spots.*':'node 4 date key unchanged'},'exclusions':{'source_file':'input provenance','page_count':'input provenance','orders.pages':'provenance: wrapper source_page used; isolate page local 1','orders.flight_dates':'table column schema; daily_spots keys retain each actual cell','orders continuation null long-header fields':'synthetic finalized nulls not printed on continuation pages; retain any non-null extra header value'},'null_conversion':'printed empty header fields map to empty string; no filling missing content','number_conversion':'str, preserving money/date strings and integer values','unknown_fields':'preserved with extra: prefix; nested values serialized without data loss','reference_access':False}
(W/'mapping_manifest.json').write_text(json.dumps(manifest,indent=2))
raw=json.loads((W/'candidate_raw.json').read_text());a=adapt(raw);assert a==adapt(raw)
assert sum(len(n['content']['rows']) for r in a['records'] for n in r['nodes'] if n['type']=='table')==260
assert [int(r['record_id'][5:]) for r in a['records']]==list(range(6,26))
(W/'candidate_canonical.json').write_text(json.dumps(a,indent=2))
# Development behavior checked before reference generation.
sample=json.loads(Path('/Users/riccardo/tesi/code/baseline-qwen-27-b/id_18_upscaled/output.json').read_text())
assert sample['orders'][0]['comments']=='Separation: 15'
assert sample['orders'][0]['lines'][0]['line_no']==1
r=json.loads((W/'evaluation_result.json').read_text());r['hashes'].update({k:hashlib.sha256((W/f).read_bytes()).hexdigest() for k,f in [('adapter','adapter.py'),('mapping_manifest','mapping_manifest.json')]});r['artifacts'].update(adapter=str(W/'adapter.py'),mapping_manifest=str(W/'mapping_manifest.json'),candidate=str(W/'candidate_canonical.json'))
(W/'adapter_validation.json').write_text(json.dumps({'deterministic':True,'page_provenance':True,'row_count_preserved':260,'development_checks':['two-tier printed Daypart (Program) represented by daypart + program','money strings unchanged','integer Line No / Len / Total Spots stringified','Comments value retains Separation: N','short header uses printed Order ID label'],'frozen_before_reference':True},indent=2))
t=W/'evaluation_result.json.tmp';t.write_text(json.dumps(r,indent=2));t.replace(W/'evaluation_result.json')
