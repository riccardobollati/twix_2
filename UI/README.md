# TWIX 2 local dashboard

Start the interface from the repository root:

```bash
python3 UI/app.py
```

Open `http://127.0.0.1:8765`. Upload a PDF, choose the run settings, and the
dashboard will run the existing `run.py` pipeline locally. It streams the log,
shows progress and artifacts as they are created, and exposes the final grouped
JSON, model reference, and evaluation report.

The dashboard always enables `--save-log`, so detailed OpenCode session logs
are retained under each run's `debug/` folder.

When the repository virtual environment exists at `env/bin/python`, the
dashboard uses it automatically to execute the pipeline. This lets the UI be
started with another local Python interpreter without losing TWIX dependencies.

Runs are stored under `UI/runs/`. This directory can contain PDFs, OCR, model
outputs, and debug material; treat it with the same care as other generated
artifacts. The server only binds to `127.0.0.1` by default.
