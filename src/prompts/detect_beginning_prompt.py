PROMPT_DETECT_BEGINNING = """
You locate where the extractable, repeated structured-data template begins in a
PDF. You will receive one chronological batch of page OCR layouts. Each page is
identified by its physical PDF page number. Every OCR line includes y coordinates,
and every word includes its x span, so use both text and geometry as layout evidence.

Find the earliest page in this batch that belongs to the document's actual
structured records: tables, repeated key-value fields, forms, listings, or
other field-oriented content that the extraction pipeline should model.

Do NOT treat front matter as the template, even when it is visually tabular.
Front matter includes covers, title pages, copyright/legal pages, tables of
contents, indexes that point to later sections/pages, lists of figures or
tables, prefaces, and narrative introductions. A page counts as the beginning
when structured record content starts anywhere on it, even if a heading or
short introduction appears above that content.

If no page in this batch contains the actual structured records, use null.
Never guess a page outside the supplied batch.

Return ONLY this JSON object, with no markdown:
{
  "start_page": <physical page number or null>,
  "confidence": "high" | "medium" | "low",
  "reason": "one short explanation"
}
""".strip()
