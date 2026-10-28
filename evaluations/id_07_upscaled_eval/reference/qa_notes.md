# Reference QA notes

This is a Codex-generated, model-assisted local PDF reference. It is not
human-verified truth and was not produced by an external model service.

- Reviewed rendered source pages 6 through 25 individually at full page
  resolution, alongside local PDF text and direct table-cell extraction.
- Every page contains one `ZONE VIETATE / PROHIBITED AREAS` table with the same
  five physical columns. The evaluation representation expands the first two
  physical columns into the eight printed/scored fields in
  `evaluation_template.json`.
- Printed page labels match physical source pages 6 through 25.
- Visually confirmed row boundaries and identifiers on every selected page.
  Counts by page are: 6=13, 7=12, 8=12, 9=14, 10=10, 11=10, 12=11,
  13=10, 14=11, 15=12, 16=12, 17=14, 18=12, 19=11, 20=12, 21=11,
  22=11, 23=14, 24=13, 25=13 (238 total).
- Identifiers are contiguous from `LI P332` through `LI P569`, with no missing
  or duplicate identifiers in the selected pages.
- No row crosses a selected-page boundary, so physical-page provenance and row
  ownership are unambiguous.
- Long remarks, polygon coordinate lists, paired upper/lower limits, NIL cells,
  and PERM validity cells were checked for inclusion. No additional substantive
  table, key-value block, node, or field was found.
- The activity and remarks values include variants not present on every
  development page, but they instantiate existing fields and require no
  evaluation-template addition.
- Direct transcription reported zero parse errors and no empty scored fields.

The raw reference preserves source cells and their canonical transcription in
`ground_truth_raw.json`; the scorer-ready representation is
`ground_truth_canonical.json`.
