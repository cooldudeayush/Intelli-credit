# Intelli-Credit - AI-Powered Corporate Credit Appraisal Prototype

An end-to-end credit decision support prototype for Indian lending workflows. Intelli-Credit ingests structured financial data, unstructured borrower documents, and analyst notes, then combines reconciliation checks, document intelligence, local RAG retrieval, explainable Five Cs scoring, and an ML-assisted risk view to generate a final hybrid recommendation and a Credit Appraisal Memo (CAM).

---

## Features

- **Multi-source ingestion** - Bank statements, GST summaries, ITR/shareholding files, PDFs, and analyst notes
- **Structured + unstructured analysis** - Handles CSV, XLSX, TXT, and PDF inputs in one workflow
- **PDF parsing with fallback** - Native extraction via `pdfplumber` and `PyPDF2`, with OCR fallback support
- **GST-Bank reconciliation** - Detects turnover mismatch, circular transaction patterns, and possible revenue inflation
- **Document intelligence** - Extracts revenue, debt, liabilities, cash flow, litigation, and governance signals from narrative text
- **Local RAG engine** - Builds a lightweight local retrieval layer over uploaded unstructured documents
- **Research Agent panel** - Aggregates sector signals, promoter risk, litigation context, regulatory notes, and analyst observations
- **Five Cs scorecard** - Transparent scoring across Character, Capacity, Capital, Collateral, and Conditions
- **ML-assisted risk view** - Uses engineered features and a local prototype classifier to produce `Lend`, `Review`, or `Reject`
- **Hybrid recommendation engine** - Combines scorecard and ML views using conservative conflict-resolution logic
- **CAM generation + export** - Produces a structured Credit Appraisal Memo with TXT, DOCX, and PDF export options
- **Demo-ready scenarios** - Built-in Good, Borderline, and Risky borrower cases for reliable walkthroughs
- **Manual upload packs** - Included sample verification files for recording and evaluation

---

## Tech Stack

| Layer | Technology |
|---|---|
| App Framework | Streamlit |
| Language | Python |
| Data Processing | pandas, openpyxl |
| PDF Parsing | pdfplumber, PyPDF2 |
| OCR Fallback | pytesseract, pypdfium2 |
| ML | scikit-learn |
| Export | python-docx, reportlab |
| Research | feedparser |
| Retrieval | Custom local TF-IDF / bag-of-words RAG |

---

## Why Intelli-Credit?

Corporate credit appraisal is usually fragmented across:

- GST data
- bank statements
- annual reports
- legal notices
- analyst notes
- external intelligence

Credit officers often spend days manually combining these inputs before arriving at a decision and preparing a memo. Intelli-Credit turns that workflow into a single explainable pipeline that produces:

- a risk score
- a recommendation
- a suggested limit
- a pricing band
- a CAM-ready narrative output

---

## Core Workflow

1. **Data ingestion** - Load structured files, PDFs, and qualitative notes
2. **Document parsing** - Extract text using native PDF methods with OCR fallback
3. **Reconciliation** - Compare GST turnover against bank inflows
4. **Document intelligence** - Detect financial, litigation, and governance signals
5. **Local RAG retrieval** - Index and retrieve evidence from uploaded unstructured documents
6. **Research aggregation** - Combine sector, promoter, litigation, and regulatory observations
7. **Five Cs scoring** - Generate an explainable credit score out of 100
8. **ML-assisted risk view** - Produce a structured ML risk classification
9. **Hybrid decisioning** - Combine scorecard and ML into a final recommendation
10. **CAM generation** - Build a structured Credit Appraisal Memo with downloads

---

## Project Structure

```text
Intelli-credit/
|-- app.py
|-- requirements.txt
|-- README.md
|-- modules/
|   |-- __init__.py
|   |-- cam_export.py
|   |-- cam_generator.py
|   |-- data_loader.py
|   |-- demo_scenarios.py
|   |-- document_intel.py
|   |-- hybrid_recommendation.py
|   |-- indian_context.py
|   |-- ingestion.py
|   |-- live_research.py
|   |-- ml_hybrid.py
|   |-- qualitative.py
|   |-- rag_engine.py
|   |-- recommendation.py
|   |-- reconciliation.py
|   |-- research_agent.py
|   `-- research_insights.py
`-- sample_data/
    |-- annual_report_excerpt.txt
    |-- bank_statement.csv
    |-- gst_summary.csv
    |-- qualitative_notes.txt
    `-- upload_packs/
        |-- README.md
        |-- good_case/
        `-- risky_case/
```

---

## Setup & Installation

### Prerequisites

- Python 3.10+
- Recommended: a normal local Python install, not the Microsoft Store shim

### 1. Create a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 3. Run the app

```powershell
python -m streamlit run app.py
```

Open the local Streamlit URL shown in the terminal.

---

## How to Use

## Option 1: Demo Mode

Use the built-in demo scenarios for the fastest walkthrough:

- **Good Borrower**
- **Borderline / Manual Review**
- **Risky Borrower**

This is the best option for quick testing and live presentations.

## Option 2: Manual File Upload

Use the bundled verification packs:

- `sample_data/upload_packs/good_case`
- `sample_data/upload_packs/risky_case`

Recommended manual demo flow:

1. Turn **Demo Mode** off
2. Upload `bank_statement.csv`
3. Upload `gst_summary.csv`
4. Upload `external_notes.txt`
5. Paste `primary_notes.txt` into the primary due diligence box
6. Run the analysis

For a stronger video walkthrough, you can additionally upload your own PDFs for annual report or legal notice testing.

---

## Main Screens

The Streamlit prototype contains 14 major screens/tabs:

1. **Inputs & Demo** - Upload controls, scenario selection, and source coverage
2. **Document Insights** - Extracted financial metrics and document processing details
3. **GST-Bank Reconciliation** - Mismatch checks and circular transaction indicators
4. **Research & Qualitative** - Sector and analyst intelligence summary
5. **Scoring & Explainability** - Five Cs table, penalties, boosts, and drivers
6. **CAM & Recommendation** - Scorecard, ML, hybrid decision, and CAM preview
7. **Unstructured Intelligence** - RAG retrieval evidence by theme
8. **Research Agent** - Intelligence cards and generated summary
9. **Indian Context Signals** - India-specific red flags and placeholders
10. **Hybrid Recommendation** - Side-by-side rule vs ML comparison
11. **ML Diagnostics** - Feature values and importance view
12. **Export CAM** - TXT, DOCX, and PDF export
13. **Live Research** - Optional headline retrieval mode
14. **How It Works** - High-level visual pipeline

---

## Decision Logic

### Five Cs Scorecard

The explainable rule-based engine scores:

- **Character**
- **Capacity**
- **Capital**
- **Collateral**
- **Conditions**

Each dimension contributes to a transparent total score out of 100.

### ML-Assisted Risk View

The ML layer uses engineered features derived from:

- Five Cs scores
- GST-bank mismatch
- circular trading flags
- litigation and governance indicators
- stress indicators
- qualitative adjustments
- RAG and source usage signals

### Hybrid Recommendation

The final recommendation blends:

- scorecard output
- ML risk view
- confidence-aware conflict handling

This keeps the prototype explainable while still demonstrating an ML-supported workflow.

---

## Key Modules

| Module | Responsibility |
|---|---|
| `ingestion.py` | File loading, parsing, and fallback handling |
| `document_intel.py` | Extracts financial and risk cues from text |
| `reconciliation.py` | GST-bank comparison and circular pattern checks |
| `rag_engine.py` | Local chunking, indexing, and retrieval |
| `research_agent.py` | Aggregates multi-source research context |
| `scoring.py` | Five Cs credit scoring engine |
| `recommendation.py` | Rule-based baseline recommendation |
| `ml_hybrid.py` | Feature engineering and ML-assisted risk view |
| `hybrid_recommendation.py` | Final scorecard + ML fusion logic |
| `cam_generator.py` | Builds CAM text |
| `cam_export.py` | Generates DOCX and PDF outputs |

---

## Demo Walkthrough

Recommended evaluation flow:

1. Launch the app and first show **Demo Mode**
2. Run the **Good Borrower** case to show the full happy-path workflow
3. Walk through reconciliation, document insights, scoring, ML diagnostics, and CAM preview
4. Turn **Demo Mode** off
5. Use `sample_data/upload_packs/good_case` to show manual ingestion
6. Repeat with `sample_data/upload_packs/risky_case` to demonstrate how the outcome changes for a stressed borrower

This shows both:

- stable end-to-end prototype behavior
- real file ingestion capability

---

## Notes for Judges

- This repository is intentionally focused on the working prototype and the sample files needed to verify it
- The app is primarily offline-capable
- The ML layer is prototype-oriented and uses synthetic data rather than proprietary bank data
- Some Indian ecosystem checks are implemented as rule-based placeholders to show the intended product direction
- The strongest value of the prototype is the explainable end-to-end workflow, not black-box automation

---

## Limitations

- This is a hackathon prototype, not a production underwriting system
- The ML classifier is not trained on real historical loan-book outcomes
- OCR behavior depends on local system availability of Tesseract
- Optional live research depends on RSS availability
- Some India-specific indicators are placeholders for future live integrations

---

## Future Scope

- Real MCA, bureau, and court integrations
- GSTR-2A vs 3B live reconciliation
- Production-grade borrower portfolio dashboard
- Real historical credit outcome training data
- Optional LLM-based summary generation while keeping core decisions explainable
- Enterprise deployment with API and role-based access control

---

## Team

> Built for the **National AI/ML Hackathon by Vivriti Capital**

| | |
|---|---|
| Developer | Ayush Ranjan |
| Institution | Goa Institute of Management |
| GitHub | [cooldudeayush](https://github.com/cooldudeayush) |
| Hackathon | National AI/ML Hackathon |

---
