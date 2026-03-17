# Intelli-Credit

Intelli-Credit is a hackathon prototype for AI-assisted corporate credit appraisal in Indian lending workflows. It combines structured financial inputs, unstructured documents, analyst notes, explainable scoring, and an ML-assisted risk view to generate a credit recommendation and a downloadable Credit Appraisal Memo (CAM).

The goal of the prototype is to reduce the manual effort involved in stitching together GST data, bank statements, annual reports, legal signals, and qualitative due diligence into a single decision-support workflow.

## What Problem It Solves

Corporate credit appraisal is usually fragmented across multiple data sources:

- Structured financial data such as bank statements, GST summaries, ITRs, and shareholding files
- Unstructured documents such as annual reports, legal notices, sanction letters, and rating reports
- Qualitative inputs such as site visit observations, management notes, and external intelligence

In practice, analysts must manually reconcile these sources, identify red flags, and prepare a credit memo. Intelli-Credit turns that into a single explainable workflow.

## Key Features

- Multi-source ingestion for CSV, XLSX, TXT, and PDF inputs
- PDF parsing with native extraction and OCR fallback
- GST vs bank reconciliation with mismatch and circular-pattern checks
- Local RAG pipeline over unstructured documents
- Research and qualitative signal aggregation
- Explainable Five Cs scorecard
- ML-assisted risk view using engineered features
- Hybrid recommendation combining scorecard and ML outputs
- CAM preview with TXT, DOCX, and PDF export
- Demo mode with prebuilt borrower scenarios for reliable walkthroughs

## End-to-End Workflow

1. Ingest borrower inputs from structured files, PDFs, and analyst notes
2. Extract text and signals from uploaded documents
3. Reconcile GST turnover against bank inflows
4. Retrieve supporting evidence from unstructured text using local RAG
5. Incorporate qualitative and Indian-context risk indicators
6. Score the borrower using the Five Cs framework
7. Generate an ML-assisted risk view from engineered features
8. Produce a final hybrid decision, suggested limit, pricing band, and CAM

## Repository Structure

```text
app.py
modules/
sample_data/
requirements.txt
README.md
```

- `app.py`: Streamlit entrypoint and UI
- `modules/`: ingestion, document intelligence, scoring, ML, recommendation, RAG, and export logic
- `sample_data/`: bundled demo inputs and manual upload packs for evaluation

## How To Run

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

If your virtual environment was created from the Microsoft Store Python shim and stops launching, recreate it from a working Python installation before running the app.

## How To Evaluate The Prototype

There are two simple ways to test the app.

### Option 1: Demo Mode

Use the built-in demo scenarios inside the app:

- Good Borrower
- Borderline / Manual Review
- Risky Borrower

This is the fastest way to see the complete workflow end to end.

### Option 2: Manual File Upload

Use the verification files under `sample_data/upload_packs`:

- `sample_data/upload_packs/good_case`
- `sample_data/upload_packs/risky_case`

Suggested upload flow:

1. Turn `Demo Mode` off
2. Upload `bank_statement.csv`
3. Upload `gst_summary.csv`
4. Upload `external_notes.txt`
5. Paste `primary_notes.txt` into the primary due diligence box
6. Optionally upload supporting PDFs such as annual report or legal notice if available locally
7. Click `Run Intelli-Credit Analysis`

## Main Modules

- `modules/ingestion.py`: file loading, PDF parsing, OCR fallback, and source summaries
- `modules/document_intel.py`: signal extraction from narrative documents
- `modules/reconciliation.py`: GST-bank reconciliation and transaction pattern checks
- `modules/rag_engine.py`: local retrieval over uploaded unstructured text
- `modules/research_agent.py`: consolidated research and qualitative insight cards
- `modules/scoring.py`: explainable Five Cs scoring logic
- `modules/ml_hybrid.py`: synthetic-data-backed prototype ML view and feature engineering
- `modules/hybrid_recommendation.py`: final decision fusion logic
- `modules/cam_generator.py`: CAM text construction
- `modules/cam_export.py`: DOCX and PDF export

## Why The Prototype Is Explainable

This project is intentionally not built as a black-box decision engine.

- The scorecard logic is rule-based and visible
- Feature engineering for the ML layer is transparent
- The hybrid decision shows how scorecard and ML outputs align or disagree
- CAM output includes rationale, findings, and supporting observations

This makes the prototype easier to present, audit, and discuss in a lending context.

## Tech Stack

- Python
- Streamlit
- pandas
- pdfplumber
- PyPDF2
- openpyxl
- scikit-learn
- python-docx
- reportlab
- pytesseract
- feedparser

## Current Scope And Limitations

- This is a hackathon prototype, not a production underwriting system
- The ML layer uses synthetic training data for demonstration
- Some India-specific signals are implemented as placeholders or proxy rules
- Live research is optional and designed to fail gracefully
- Final credit decisions should be treated as prototype recommendations, not production approvals

## Future Improvements

- Replace synthetic ML data with real historical credit outcomes
- Add live integrations for MCA, court, bureau, and regulatory data
- Introduce stronger document classification and extraction pipelines
- Add optional LLM-based narrative summarization while keeping core decisions explainable
- Expand from single-borrower review to portfolio monitoring

## Submission Note

This repository is intentionally kept focused on the working prototype, the supporting modules, and the sample verification inputs needed to run and evaluate the app.
