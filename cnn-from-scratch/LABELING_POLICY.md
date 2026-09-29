# Labeling policy: follow TWIX block types

The source of truth is [`template_inference_prompt.py`](../src/prompts/template_inference_prompt.py),
especially “Core Concepts,” “Row Labels,” “Identify Block Types and Label Rows,”
and the police and invoice examples. The [`ground-truth prompt`](../src/ground_truth_gen/prompt.py)
requires a separate physical block per occurrence and per page. This dataset
uses `key_value` as the mask/annotation spelling of TWIX's `kv` type.

## Class decision

- **Table (`1`, green):** one shared K/header band of field or column names,
  followed by one or more V/data rows aligned under those headers. Multi-line
  headers, wrapped cell values, sparse columns, and absent grid lines do not
  change the type. A page-local continuation of a table remains a table even
  if its header appeared on the previous page.
- **Key–value (`2`, blue):** each row contains its own field label followed by
  its value on the same row. Consecutive KV rows in one coherent section form
  one block. There is no shared column-header row above them. A form can have
  visible borders and still be KV. Repeated instances are separate blocks.
- **Background (`0`):** titles, headings, logos, page numbers, footers,
  disclaimers, report filters, standalone text or images, and content with no
  stable field/value structure. Grid lines alone do not make a table; product
  prose alone does not make a KV block. A heading that only introduces a block
  stays background unless it is itself a field.

## Region extent

Each rectangle covers one physical structured block on one page, including
its field labels, data values, and whitespace internal to that block. Do not
merge adjacent blocks merely because they touch or belong to one record. Do
not merge repeated records or blocks across pages. For mixed layouts, split
table and KV sections into separate non-overlapping rectangles. Mask pixels
can have only one class.

For the two-page spread rendered as `new_04_p0003`, the side-by-side halves
are one table and use one enclosing region, as confirmed during label review.

For `id_05`, `data/id_05/template.json` specifies a table, a KV section,
then two more tables per complaint record. For `new_11`, the saved template
specifies a KV block followed by a charges table. Use those source artifacts
to resolve the corresponding page layouts; do not copy their labels blindly
where the rendered page differs.

The annotation JSON format remains
`{"image": "...png", "width": W, "height": H, "regions": [{"class": "table"|"key_value", "box": [x0,y0,x1,y1]}]}`.
Coordinates are pixels of the 1280-pixel render, with exclusive right and
bottom edges. `build_masks.py` checks dimensions, bounds, and overlaps.
