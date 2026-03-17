# Intelli-Credit (Hackathon Prototype)

Intelli-Credit is a presentation-ready AI-assisted corporate credit appraisal prototype for Indian lending use cases.
The prototype is centered on `app.py`, with the analytical workflow split into focused modules under `modules/`.

## 1. Quick Run

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

If your local `.venv` was created from the Microsoft Store Python shim and stops launching, recreate it from a working interpreter before submission.

## 2. Repository Layout

```text
app.py
modules/
sample_data/
requirements.txt
README.md
```

- `app.py`: Streamlit entrypoint for the prototype demo
- `modules/`: ingestion, scoring, RAG, recommendation, and export logic
- `sample_data/`: demo inputs used by the built-in scenarios

Generated files, caches, and virtual environments should stay out of Git.

## 3. Core Architecture

1. Multi-source ingestion (structured + unstructured + notes)
2. OCR-aware PDF parsing fallback
3. Source coverage tracking
4. Local RAG over unstructured documents
5. Research Agent and Indian-context signal layers
6. Five Cs scorecard (transparent rules)
7. ML-assisted recommendation layer (hybrid)
8. CAM generation + DOCX/PDF export

## 4. Round 3: Scorecard + ML Hybrid Recommendation

### Scorecard layer (existing, preserved)
- Five Cs: Character, Capacity, Capital, Collateral, Conditions
- Explainable penalties/boosts
- Rule-based recommendation baseline

### ML layer (new)
- Module: `modules/ml_hybrid.py`
- Local model: `RandomForestClassifier` (scikit-learn)
- Trained on synthetic prototype data (documented and deterministic)
- Produces:
  - ML decision (`Lend/Review/Reject`)
  - confidence score
  - class probabilities
  - top feature importance list

### Hybrid layer (new)
- Module: `modules/hybrid_recommendation.py`
- Combines scorecard decision + ML decision
- Conflict policy is transparent and conservative with confidence gating
- Outputs:
  - final hybrid decision
  - hybrid limit
  - hybrid pricing band
  - alignment/override reason

## 5. ML Features Used

Engineered in `engineer_features(...)`:
- `character_score`
- `capacity_score`
- `capital_score`
- `collateral_score`
- `conditions_score`
- `total_score`
- `gst_bank_mismatch_ratio`
- `circular_trading_flag`
- `litigation_count`
- `governance_flag`
- `debt_stress_flag`
- `operational_stress_flag`
- `sector_risk_flag`
- `promoter_risk_flag`
- `shareholding_concentration_flag`
- `mca_compliance_flag`
- `qualitative_positive_adjustment`
- `qualitative_negative_adjustment`
- `rag_signal_count`
- `structured_signal_count`
- `external_signal_count`
- `primary_due_diligence_signal_count`

Defaults are applied safely when inputs are missing.

## 6. Local RAG (Round 2, retained)

Module: `modules/rag_engine.py`

Indexed unstructured sources:
- Annual Report
- Legal Notice
- Sanction Letter
- Rating Report
- Board Minutes
- ITR/Shareholding PDFs when provided as PDF text

Flow:
- clean text -> chunk -> local vectorization -> theme retrieval
- no remote API required
- metadata per chunk includes source and chunk id

## 7. Research Agent Panel (Round 2, retained)

Module: `modules/research_agent.py`

Provides structured cards for:
- promoter/management risk
- sector headwinds
- litigation summary
- regulatory alerts
- MCA notes
- external notes
- primary due diligence notes

Includes a `Generate Research Summary` action for presentation use.

## 8. Indian Context Signals (Round 2, retained)

Module: `modules/indian_context.py`

Cards include:
- GST vs bank mismatch
- circular trading / revenue inflation
- GSTR-2A vs 3B placeholder
- CIBIL commercial placeholder
- MCA compliance flag
- promoter/related-party indicator
- shareholding concentration
- legal/e-courts proxy signal
- sector regulatory alert

Each card is marked as real logic, rules+notes, structured rule, or placeholder.

## 9. CAM Export (Round 3)

- CAM preview remains in app.
- Export options:
  - TXT
  - DOCX (`modules/cam_export.py`)
  - PDF (`modules/cam_export.py`, ReportLab-based)

CAM includes scorecard vs ML summary and final hybrid recommendation sections.

## 10. Optional Live Research (Round 3)

Module: `modules/live_research.py`

- Uses optional RSS fetch (`feedparser`) for lightweight live headlines.
- Toggle: `Live Research Mode`.
- Graceful fallback to manual/mock mode when unavailable.
- Core app remains fully functional offline.

## 11. Dependencies Added Across Rounds

- `python-docx`
- `pytesseract` (OCR path)
- `scikit-learn` (ML layer)
- `reportlab` (PDF CAM export)
- `feedparser` (optional live news)

## 12. Notes

- This is a hackathon prototype, not a production underwriting system.
- Synthetic ML data is used for demonstration only.
- Recommendation remains explainable: scorecard + ML + hybrid conflict rationale are all visible.
