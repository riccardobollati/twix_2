import json,sys
from decimal import Decimal
MAPPING = {'designation': 'Section designation', 'mass_per_metre': 'Dimensions - Mass per Metre kg/m', 'depth': 'Dimensions - Depth of Section D mm', 'width': 'Dimensions - Width of Section B mm', 'flange_thickness': 'Dimensions - Thickness Flange T mm', 'web_thickness': 'Dimensions - Thickness Web t mm', 'root_radius': 'Dimensions - Root Radius r mm', 'depth_between_fillet_tangencies': 'Dimensions - Depth between Fillets d mm', 'cross_section_area': 'Dimensions - Area of Section A cm2', 'ratio_b_over_t': 'Dimensions - Ratios for Local Buckling Flange b/T', 'ratio_d_over_t': 'Dimensions - Ratios for Local Buckling Web d/t', 'end_notch_clearance': 'Dimensions - Dimensions for Detailing End clearance C mm', 'buckling_detailing_width': 'Dimensions - Dimensions for Detailing Notch N mm', 'buckling_detailing_depth': 'Dimensions - Dimensions for Detailing Notch n mm', 'surface_area_per_metre': 'Dimensions - Surface Area Per metre m2', 'second_moment_of_area_x': 'Properties - Second Moment of Area Axis x-x cm4', 'second_moment_of_area_y': 'Properties - Second Moment of Area Axis y-y cm4', 'radius_of_gyration_x': 'Properties - Radius of Gyration Axis x-x cm', 'radius_of_gyration_y': 'Properties - Radius of Gyration Axis y-y cm', 'elastic_section_modulus_x': 'Properties - Elastic Modulus Axis x-x cm3', 'elastic_section_modulus_y': 'Properties - Elastic Modulus Axis y-y cm3', 'plastic_section_modulus_x': 'Properties - Plastic Modulus Axis x-x cm3', 'plastic_section_modulus_y': 'Properties - Plastic Modulus Axis y-y cm3', 'buckling_parameter': 'Properties - Buckling Parameter u', 'torsional_index': 'Properties - Torsional Index x', 'warping_constant': 'Properties - Warping Constant H dm6', 'torsional_constant': 'Properties - Torsional Constant J cm4'}
META = {'standard': 'Dimensions', 'material_specification': 'Specification', 'size_range': 'Size Range'}
def value(v):
 if v is None: return ""
 if isinstance(v,(int,float)): return format(Decimal(str(v)),"f")
 return str(v)
def adapt(raw):
 records=[]
 for entry in raw:
  page=entry['source_page'];nodes=[]
  for i,s in enumerate(entry['raw']['sections']):
   kv=[{'key':target,'value':value(s.get(key))} for key,target in META.items() if s.get(key) is not None]
   if kv:nodes.append({'id':f'0_{page}_{i}','type':'key_value','content':kv,'relationship':{'parent_id':None}})
   headers=[]
   for row in s['rows']:
    for key in row:
     target=MAPPING.get(key,key)
     if target not in headers:headers.append(target)
   rows=[]
   for row in s['rows']:
    converted={MAPPING.get(k,k):value(v) for k,v in row.items()}
    rows.append([{'key':h,'value':converted.get(h,'')} for h in headers])
   if rows:nodes.append({'id':f'2_{page}_{i}','type':'table','content':{'headers':headers,'rows':rows},'relationship':{'parent_id':None}})
  records.append({'record_id':f'page_{page}','source_page':page,'nodes':nodes})
 return {'records':records}
if __name__=='__main__':
 from src.eval.convert import load_json_dedup
 result=adapt(load_json_dedup(sys.argv[1]));open(sys.argv[2],'w').write(json.dumps(result,indent=2))
