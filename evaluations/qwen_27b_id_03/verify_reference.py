import json,re,collections
from pathlib import Path
import pdfplumber
W=Path(__file__).resolve().parent
raw=json.loads((W/'reference/ground_truth_raw.json').read_text())
checks=[]
with pdfplumber.open('/Users/riccardo/tesi/code/full_dataset/id_03.pdf') as pdf:
 for entry in raw:
  n=entry['source_page'];p=pdf.pages[n-1];t=p.find_tables()[0];body_top=t.rows[2].bbox[1]
  chars=[c for c in p.chars if t.bbox[0]<=((c['x0']+c['x1'])/2)<=t.bbox[2] and body_top<=((c['top']+c['bottom'])/2)<=t.bbox[3]]
  source=collections.Counter(c['text'] for c in chars if not c['text'].isspace())
  cells=collections.Counter(c for row in entry['rows_as_printed'] for cell in row for c in (cell or '') if not c.isspace())
  assert source==cells,(n,source-cells,cells-source)
  checks.append({'source_page':n,'body_characters':sum(source.values()),'all_body_characters_transcribed':True,'row_count':len(entry['rows_as_printed']),'field_count':13})
(W/'reference/coverage_checks.json').write_text(json.dumps(checks,indent=2))
print('All 20 pages: every non-whitespace PDF body character accounted for in reference cells.')
