# Intelli-Credit — Rebuild Plan

**Status:** Draft v1 · **Created:** 2026-09-12
**Baseline audited:** `app.py` (911 lines) + 18 modules in `modules/` (3,342 lines total)

---

## 1. Why a rebuild

The prototype covers every pillar of the problem statement at least partially, and three
things in it are genuinely good: the GST-vs-bank reconciliation logic, the scoring audit
trail, and the confidence-gated hybrid override policy.

The problem is not missing features. It is that **several features are scaffolding presented
as substance** — placeholders, mock data, and silent fallbacks that a judge or interviewer
will find in under ten minutes. The rebuild converts scaffolding into working logic, and
where that is not feasible, labels it honestly and cheaply.

### Audit summary

| # | Requirement | Now | Target |
|---|---|---|---|
| P1a | Unstructured parsing (annual report / legal / sanction) | Partial | Full |
| P1b | GST × bank → circular trading / revenue inflation | **Met** | Keep + extend |
| P1c | Databricks ingestion | **Absent** | Adapter layer |
| P2a | Auto web research (promoter / sector / litigation) | ✅ **Met** (Phase 2) | — |
| P2b | Officer qualitative notes → score | **Met** | Keep + extend |
| P3a | CAM generator (Word / PDF, Five Cs) | **Met** | Keep + polish |
| P3b | Loan amount + rate, transparent model | **Met** | Keep |
| P3c | Explains rejection / limit | **Met** | Strengthen |
| E1 | Extraction accuracy on messy scanned PDFs | ✅ **Met** (Phase 1) | — |
| E2 | Research depth (finds what is *not* in the files) | ✅ **Met** (Phase 2) | — |
| E3 | Explainability | ✅ **Met, per-instance** (Phase 4) | — |
| E4 | Indian context (GSTR-2A/3B, CIBIL) | Partial | 2A/3B real |

---

## 2. Guiding principles

1. **Never fabricate a number.** If extraction fails, say so. A visible "could not extract"
   beats a confident wrong figure — and it *is* the trustworthiness story.
2. **Every claim in the README/report/deck must be true of the code.** If we say TF-IDF,
   there must be an IDF term. Fix the code or fix the wording; never leave the gap.
3. **Label prototype surfaces explicitly.** A card that says "Placeholder — needs GSTN API"
   costs nothing and reads as engineering judgment. A card that silently mocks reads as a bug.
4. **Scoring changes flow through one ledger.** All adjustments — rules, qualitative, RAG,
   research — keep using the `{factor, type, points, reason}` shape so explainability stays
   uniform and the CAM needs no special-casing.
5. **Refactor behind tests.** There are currently zero tests. Nothing structural moves until
   Phase 0 lands.

---

## 3. Phases

### Phase 0 — Safety net ✅ COMPLETE (2026-09-13)

No behaviour change. Makes the rest of the rebuild safe.

- [x] Add `tests/` with pytest — 90 tests across 6 files, plus `requirements-dev.txt`.
- [x] **Golden-output test:** all three demo scenarios snapshotted in
      `tests/test_golden_scenarios.py` (`five_cs_scores`, `total_score`, `risk_category`,
      scorecard/ML/hybrid decisions, limits, pricing band, recon flags, adjustment counts).
      Verified against the live UI — the rendered dashboard matches the golden values exactly.
- [x] **Unit tests for pure logic:** `reconcile_gst_bank`, `compute_five_cs_score`,
      `extract_qualitative_adjustments`, `rag_engine`, `combine_scorecard_ml`,
      `extract_document_signals`.
- [x] Pin `requirements.txt` to exact versions, verified on Python 3.13.7.
- [x] Add `pypdfium2` (was undeclared — OCR was dead on any clean install).
- [x] `tests/pipeline_harness.py` — originally a headless replica of `app.py`'s demo
      pipeline, since a top-level Streamlit script cannot be imported. **Phase 1.3 retired
      the duplication:** it is now a thin wrapper that builds `AnalysisInputs` and calls
      the real `modules.pipeline.run_analysis`, so tests and the app share one code path.

**Status:** `85 passed, 5 xfailed` in 0.9s. App verified rendering in a browser.

#### What Phase 0 found

The environment was more broken than the audit suggested, and three real defects surfaced:

1. **The app could not start on this machine's default Python.** `pdfplumber`, `PyPDF2`,
   `pytesseract`, `feedparser` and `pypdfium2` were all absent from Python 3.13;
   `import modules` failed outright. Fixed by the pinned venv.
2. **pandas 3.0 removed `fillna(method=)`.** `data_loader.py:57` would have crashed at
   runtime. Fixed to `.ffill()` — behaviour-identical, forward-compatible.
3. **Revenue mis-extraction (NEW — not in the original audit).** On the *Good Borrower*
   demo, `"Revenue for FY2025 was INR 1,240 crore"` extracts as **INR 2,025** — the regex
   grabs the year out of `FY2025`. Leverage then computes as 1,234,568x, costing that
   borrower 10 Capital points and collapsing its loan limit to the INR 50 lakh floor.
4. **Sign loss on negative amounts (NEW).** The capture group `(\d[\d,]*\.?\d*)` cannot
   match a leading minus. On the *Risky Borrower* demo, `-45 crore` operating cash flow
   parses as **+450,000,000**, earning a **+2 Capacity boost instead of a -8 penalty** —
   a 10-point swing in the wrong direction on the scenario specifically designed to fail.
5. **Qualitative rules miss their own demo data.** The Risky Borrower produces **zero**
   qualitative adjustments: its notes say "resignation concerns" and "concentration in two
   counterparties", neither of which matches the literal keyword lists.
6. **The Borderline scenario is not borderline.** It scores 77.1 = "Low" risk and the
   scorecard says *Lend*; only the ML disagreement drags it to Review. The deck claims
   this case scores 50-74.

Defects 3, 4, 5 and the dead RAG weighting are captured as `strict=True` xfail tests, so
the moment a Phase 1 fix lands the test flips to XPASS and forces the marker's removal.

**Running the tests:**
```
.venv/Scripts/python.exe -m pytest
```

---

### Phase 1 — Trust and correctness ✅ COMPLETE (2026-09-13)

**Status:** `98 passed, 1 xfailed`. All four sub-items landed; verified in a browser.

**Score movement (intended, re-baselined in `tests/test_golden_scenarios.py`):**

| Scenario | Before | After | Cause |
|---|---|---|---|
| Good Borrower | Capital 10, total 78.3, limit INR 50L | Capital 20, total 88.3, limit INR 5.66Cr | Revenue reads 1,240 crore, not the year 2025 |
| Risky Borrower | Capacity 12, total 40, limit INR 61L | Capacity 2, total 30, limit INR 50L | -45 crore OCF now penalised -8, was boosted +2 |
| Borderline | unchanged | unchanged | Its text contains neither defect |

Risk separation between Good and Risky widened from 38.3 points to **58.3**.

**Also fixed along the way:** `modules/cam_generator.py` printed `INR 0` for a missing
figure; it now prints `[NOT EXTRACTED]` and lists data gaps. Two dead helpers removed from
`app.py` (`_build_docx_bytes`, `_source_findings_line`).

**Deviation from plan:** 1.1 specified a labelled sector-median assumption as a fallback.
Implemented `extracted` / `missing` only — no assumed values. A labelled assumption still
flows into the score and the limit, which contradicts principle 1. Where revenue is
missing, the limit falls back to the bank-throughput cap alone, which is both more
conservative and harder to game than a declared revenue figure.

---

### Phase 1 — original plan (for reference)

#### 1.1 ✅ Kill silent fabricated defaults — `modules/document_intel.py`

**Problem.** `_extract_amount` returns hardcoded values (revenue ₹95 Cr, debt ₹28 Cr,
liabilities ₹51 Cr, OCF ₹11 Cr) when regex finds nothing. These flow into the score, the
limit, and the CAM indistinguishable from extracted figures. On the exact scenario the
judges test — a messy scanned PDF — this is what fires.

**Target.**
- Return `{"value": float|None, "confidence": "extracted"|"assumed"|"missing", "evidence": str}`
  per field instead of a bare float.
- Keep a sector-median assumption as a *labelled* fallback, never a silent one.
- Surface an "Extraction Confidence" panel: field, value, source sentence, status chip.
- CAM section 3 prints `[ASSUMED]` next to any non-extracted figure.
- Scoring treats `None` as "insufficient data" rather than 0 — a missing OCF must not
  read as "negative cash flow, −8 points".

#### 1.2 ✅ Fix the regex — sign loss and first-match-wins

Phase 0 proved both of these fire on the project's own demo data. Failing tests already
exist (`tests/test_document_intel.py`, xfail strict).

**(a) Negative amounts lose their sign.** The capture group `(\d[\d,]*\.?\d*)` cannot match
a leading `-`. Risky Borrower's `-45 crore` OCF becomes `+450,000,000`, which trips the
`> 120M` branch in `scoring.py` and awards a **+2 boost where a -8 penalty belongs**.
- Allow an optional leading `-` (and a parenthesised `(45)` accounting negative).
- Re-baseline: Risky Borrower's Capacity should drop by ~10 points.

**(b) First match wins, even when it is a year.** `"Revenue for FY2025 was INR 1,240 crore"`
returns `2025`. The `{0,60}?` lazy window stops at the first digit run it sees.
- Collect all candidate matches; score by unit presence, keyword proximity, and magnitude
  plausibility; take the best rather than the first.
- Reject candidates that look like years (1900-2100 with no unit suffix).
- Prefer the largest of the top candidates for revenue-like fields — prior-year
  comparatives are usually smaller and later in the sentence.
- Record which candidate won and why, for the evidence string in 1.1.

**Expected blast radius:** every golden value for Good Borrower (Capital 10 -> ~20, limit
INR 50 lakh -> revenue-capped) and Risky Borrower (Capacity 12 -> ~2). Re-baseline both in
the same commit as the fix.

#### 1.3 ✅ Make the Run button real — `app.py`

**Problem.** The pipeline runs unconditionally on every rerun; `run_clicked` only hides an
info message (`app.py:194-200`). Every keystroke re-runs scoring, RAG indexing, and live
RSS fetching.

- Wrap the pipeline in a function, call it only on `run_clicked`, cache results in
  `st.session_state`.
- Render tabs from session state so widget interaction does not recompute.
- `st.stop()` before the tabs until the first run completes.

#### 1.4 ✅ Fix the dead RAG weighting — `modules/rag_engine.py:148`

`derive_rag_adjustments` looks up per-theme weights on line 145 and then overwrites them
with `points = -1` on line 148. The weighting you designed never executes.

- Delete the override; use the mapped value.
- Re-tune the map now that it is live (litigation/governance likely −2, others −1).
- Re-baseline the Phase 0 golden test — this *will* move scores, deliberately.

**Exit criteria:** upload a scanned PDF with no extractable figures; dashboard shows
"could not extract" rather than ₹95 Cr; score reflects missing data, not fake data.

---

### Phase 2 — A research agent that actually researches ✅ COMPLETE (2026-09-13)

**Status:** `135 passed, 0 xfailed`. Verified live in a browser — real Google News
headlines fetched across 5 queries.

**Criterion E2 moves from red to green.** News signals now reach the score, the model and
the CAM with full attribution, instead of being displayed and discarded.

#### What landed

- **`modules/live_research.py` rewritten.** Five query templates covering the India-specific
  registers where distress surfaces first — NCLT/insolvency and RBI/SEBI enforcement —
  not just generic promoter and sector news. Real timeout control via `urllib` (feedparser
  has none of its own), cross-query deduplication, and a three-mode contract:
  `live` / `offline_sample` / `disabled`.
- **Headlines are scored.** Four risk buckets (financial distress, promoter/governance,
  litigation, regulatory/sector) map to Five Cs adjustments in the standard ledger shape,
  capped at 3 and one per bucket, each quoting the headline that caused it.
- **ML feature #23, `live_news_signal_count`**, added to the synthetic generator and
  feature engineering. Model retrained; CV accuracy 0.876 with 23 features (was 0.879).
- **Research on by default**, cached 15 minutes per borrower via `st.cache_data`.
- **Offline fixtures** in `sample_data/offline_headlines.json` so the feature still
  demonstrates without a network. Every fixture title is prefixed `[OFFLINE SAMPLE]` at
  load time, so the label survives into the UI, the adjustment reason and the exported CAM.
  A test enforces it.
- **Sector profiles externalised** to `data/sector_profiles.json` — 8 sectors (was 5
  hardcoded), each with `watch_items`, `last_reviewed` and a source note. The UI sector
  list is driven from the file.
- **Qualitative rules widened** (closes the last open xfail) and two new rules added:
  delayed statutory dues, and stock-statement discrepancy.
- **Unified explainability table rebuilt from the single ledger** rather than stitched from
  each producer, so any future signal source appears automatically. Now shows the net
  point effect of all signals.

#### Bug found and fixed during the phase

**Substring matching raised false flags.** `"rbi"` matched inside **"a*rbi*tration"**, so an
arbitration headline produced a regulatory-pressure penalty. Keyword matching is now
whole-word (`\brbi\b`), which also stops `"ban"` firing on `"urban"`. Parametrised tests
cover both.

#### Score movement

| Scenario | Before | After | Cause |
|---|---|---|---|
| Risky | Capacity 2, total 30, qual 0 | Capacity **0**, total **28**, qual **2** | Widened rules detect "heavy concentration in two counterparties" and "resignation concerns" |
| Good | 88.3 | 88.3 | unchanged |
| Borderline | 71.1 Review | 71.1 Review | unchanged |

Demo goldens run with research **disabled** so they stay deterministic; news scoring is
covered by its own tests against fixed headlines.

#### Deviation from plan

2.3 proposed adding NCLT/IBBI queries *alongside* the three existing ones. Replaced the
query set entirely instead — the old "promoter news / sector headwinds / litigation
regulatory" trio overlapped heavily, and five labelled templates cover more ground with
fewer wasted fetches.

---

### Phase 2 — original plan (for reference)

This is criterion E2, currently the weakest, and it is worth the most.

#### 2.1 ✅ Promote live research to first-class — `modules/live_research.py`

**Problem today.** Off by default; fetches titles only; **the headlines have zero effect on
the score** — they are concatenated into a display string (`app.py:344-348`) and shown in a tab.

- Default the toggle **on**, with a clean offline fallback (bundled fixture headlines,
  clearly labelled "offline sample") so a demo without wifi still shows the feature.
- Cache fetches per `(company, sector)` in session state; never re-fetch on rerun.
- Add a timeout and a visible "live / cached / offline" status chip.

#### 2.2 ✅ Score the headlines

- Risk-keyword scan over headline text across four buckets: promoter/governance, litigation,
  regulatory/sector, financial distress (default, insolvency, NCLT, IBC, downgrade).
- Emit adjustments through the **same ledger shape** as RAG, capped (max 3) and attributed:
  `reason = "Live news: '<headline>' (<source>, <date>)"`.
- Add `live_news_signal_count` as ML feature #23; retrain and re-baseline.
- Show them in the existing unified explainability table — no new surface needed.

#### 2.3 ✅ Widen retrieval beyond one RSS query set

- Add NCLT / IBBI and RBI-defaulter style query templates alongside the three current queries.
- Deduplicate headlines across queries before scoring.
- Keep every fetch attributable: query → headline → link → adjustment.

#### 2.4 ✅ Sector intelligence that is not a 5-row dict

`research_insights.py` says "mock external intelligence" in its own docstring, and
`SECTOR_RISK_MAP` is 5 hardcoded sectors.

- Externalise to `data/sector_profiles.json`, expand coverage, add a `last_reviewed` date
  and a source note per sector.
- Blend the static baseline with live-news signal rather than replacing it.

**Exit criteria:** with live mode on, a borrower with adverse news scores measurably lower
than the same borrower without it, and the CAM cites the headline that caused it.

---

### Phase 3 — Indian context with real logic

#### 3.1 GSTR-2A vs 3B reconciliation (real, not placeholder)

The problem statement names this explicitly. We already have the reconciliation pattern in
`reconciliation.py`; this is the same shape on different inputs.

- New uploaders: GSTR-2A (purchase register) and GSTR-3B (summary return).
- Compute ITC claimed in 3B vs ITC available in 2A; flag excess claim, which is the single
  most common Indian GST fraud pattern.
- Bands mirroring the existing style: >20% excess = High, >8% = Moderate.
- Wire into Capacity + Conditions through the standard ledger.
- Replace the "Placeholder" card with live status.

#### 3.2 CIBIL Commercial — structured manual input

A real bureau feed is not obtainable for a prototype. Make the ingestion path real so only
the data source is stubbed.

- Structured form: CMR rank (1–10), overdue amount, DPD bucket, suit-filed flag, enquiry count.
- Map to Character/Capacity through the ledger. CMR 1–3 = boost, 7–10 = material penalty.
- Card reads "Manual input — bureau API ready" instead of "Placeholder".

#### 3.3 Sanction-letter extraction (closes P1a)

Currently a sanction letter is only keyword-counted. The problem statement asks for
"key financial commitments ... from sanction letters from other banks".

- Extract: existing facility limits, lender name, interest rate, security offered, key covenants.
- Compute **total existing exposure** and net it against our recommended limit — a borrower
  already carrying ₹40 Cr elsewhere should not get a naive fresh ₹30 Cr.
- New CAM subsection: "Existing Banking Arrangements".

**Exit criteria:** zero cards in the Indian Context tab read "Placeholder" without a
stated reason and a working input path.

---

### Phase 4 — Explainability, per-decision ✅ COMPLETE (2026-09-13)

**Status:** `113 passed, 1 xfailed`. Verified in a browser.

#### 4.2 decision: the anchor was removed

Measured before deciding. The evidence was one-sided:

- RandomForest 5-fold CV accuracy on the synthetic set: **0.879**, classes balanced
  (252/238/210). The model did not need propping up.
- Model and anchor **agree** on Good and Risky at every blend weight from 0.35 to 1.0.
- The anchor's *only* practical effect was forcing the Borderline scenario to "Review" —
  a hardcoded formula existing to make one demo produce its intended label.

So `predict_ml_view` now returns the model's probabilities directly, and
`tests/test_ml_explainability.py` fails if `anchor_probs` or the `0.35 *` blend is ever
reintroduced. The guard against a bad model is the hybrid layer, which already refuses to
let ML override the scorecard below 90% confidence — that is the right place for it.

#### Root cause found while removing the anchor

Borderline scored 77.1 ("Low") because **cash-flow thresholds were absolute**: INR 42 crore
of operating cash flow tripped the `> INR 12 crore` "strong cash flow" boost even though it
is only a **5.1% margin** on INR 820 crore of revenue. Fixed properly rather than by editing
demo data — cash flow is now judged as a margin on revenue (`<6%` thin, `>=12%` strong, with
the old absolute bands kept as a fallback when revenue could not be extracted).

**Consequence, and the best outcome of this phase:** Borderline now scores 71.1
("Moderate"), the scorecard says Review, ML says Lend at 0.56 confidence, and because that
is below the 0.90 override bar the cautious option wins. **The hybrid conflict path — the
project's central governance claim — is now exercised by a demo, through real disagreement
rather than a hardcoded anchor.** No scenario data was changed.

#### Third bug found in retrieval

`"related party"`, `"legal notice"` and `"under capacity"` were stored as single dictionary
keys in `THEME_QUERIES` while chunks are tokenised into single words — so they could never
match anything. Three of six themes had dead query terms, including the single most
important governance signal in Indian corporate credit. Queries are now tokenised the same
way chunks are.

#### Score movement

| Scenario | Before | After | Cause |
|---|---|---|---|
| Good | 88.3, conf 0.84 | 88.3, conf **0.94** | Pure model confidence, no anchor dilution |
| Risky | 30, rag_adj 1 | 30, rag_adj **2** | Tokenised queries now match "related party" |
| Borderline | 77.1 Low, Lend | **71.1 Moderate, Review** | Cash flow judged as a margin |

#### Deviation from plan

4.3 offered "change the docs to say term-frequency" as the cheap alternative. Implemented
real IDF instead, as the plan preferred. The relevance threshold moved 0.12 -> 0.10 and is
now the named constant `RELEVANCE_THRESHOLD`, chosen from the measured gap between
incidental term overlap (~0.02-0.05) and deliberate risk language (~0.11-0.17).

**Known limitation:** the demo scenario texts are ~70 words, so they produce a single chunk
each, and IDF is degenerate on a one-chunk corpus. IDF does real work on uploaded
multi-page PDFs; it does nothing in demo mode. Worth lengthening the demo narratives.

---

### Phase 4 — original plan (for reference)

E3 is our strongest criterion; small work makes it unarguable.

#### 4.1 ✅ Per-instance feature attribution — `modules/ml_hybrid.py`

`top_features` currently uses `model.feature_importances_`, which is global — every borrower
shows the same ranking. It explains the model, not the decision.

- Replace with per-prediction contributions (SHAP if we accept the dependency; otherwise
  a leave-one-out delta on `predict_proba`, which is ~22 cheap calls and needs no new library).
- Show direction and magnitude for *this* borrower.

#### 4.2 ✅ Be honest about the anchor blend — *chose (a): removed it*

`predict_ml_view` returns `0.35 × model_probs + 0.65 × anchor_probs`, where `anchor_probs`
comes from a hardcoded linear formula. The RandomForest is a minority shareholder in its
own output, and nothing in the UI or docs says so.

Pick one, deliberately:
- **(a)** Shift the weighting toward the model (e.g. 0.7/0.3) once Phase 0 tests prove the
  demo scenarios still behave; or
- **(b)** Keep the blend and **label it** — "ML view = 35% RandomForest + 65% rule anchor
  (prototype stability)" in the ML Diagnostics tab and the report.

Either is defensible. Silence is not.

#### 4.3 ✅ Real IDF — `modules/rag_engine.py`

`Counter(tokens)` is raw term frequency; there is no IDF anywhere, but the README, report
and deck all say TF-IDF.

- Add document-frequency computation at index build; weight query and chunk vectors by IDF.
- Re-tune the 0.12 relevance threshold — IDF changes the score distribution.
- Alternative if we keep it simple: change every occurrence of "TF-IDF" in the docs to
  "term-frequency cosine similarity". Cheaper, equally honest. **Prefer implementing IDF** —
  it is ~15 lines and genuinely improves retrieval on boilerplate-heavy annual reports.

---

### Phase 5 — Ingestion at scale (Databricks)

The problem statement says "ingest multi-source data (Databricks)". The word does not appear
anywhere in the repo. Full Databricks is out of scope for a prototype; a credible adapter is not.

- [ ] `modules/data_sources.py` with a `DataSource` protocol: `LocalFileSource` (today's
      behaviour) and `DatabricksSource` (reads via `databricks-sql-connector`).
- [ ] Config-driven selection; Databricks path degrades to local with a visible banner when
      credentials are absent.
- [ ] Document the intended Delta table schemas in the README even if unused at runtime.

This converts "we didn't do it" into "here is the seam where it plugs in", which is the
honest and defensible position.

---

### Phase 6 — Documentation realignment ✅ COMPLETE (2026-09-13)

Every claim in the documentation now matches the code, and the claims that changed are
recorded rather than quietly corrected.

| Artifact | Action |
|---|---|
| `README.md` | Rewritten. Honest capability table (real logic / reference data / placeholder / not implemented), mermaid architecture diagram, accurate module list, Known Limitations section. Every number verified against the codebase. |
| `IntelliCredit_Technical_Report_v2.docx` | **New file**, 12 pages, 19 sections, 14 tables. Generated by a script that imports the live modules, so quantities cannot drift. |
| `IntelliCredit_Technical_Report.docx` | **Left untouched** — it is the submitted artifact. |
| `DECK_CORRECTIONS.md` | **New.** Precise find/replace list for the `.ppt`, plus a screenshot capture checklist. |

#### Why the report is a new file, not an edit

The original is a submitted deliverable; overwriting it would destroy the record of what was
submitted. v2 sits alongside it. If you would rather replace the original, that is a one-line
change to the generator's output path.

#### What the report now says that v1 did not

- **Section 2, "What Is Real and What Is Not"** — a capability table drawing the line between
  working logic and scaffolding explicitly, matching the labelling inside the app.
- **Section 15, "Verification"** — the test suite, which did not exist for v1.
- **Section 16, "Defects Found and Corrected Since v1"** — all ten defects with cause,
  effect and resolution. This is the most useful section for a technical conversation: it
  shows the reasoning, not just the corrected state.
- **Section 18, "Known Limitations"** — synthetic training data stated first and plainly.

#### Deck: what could not be done here

The deck is a legacy binary `.ppt`. python-pptx reads `.pptx` only and no LibreOffice is
installed on this machine, so it cannot be edited programmatically. `DECK_CORRECTIONS.md`
carries the exact change list instead — **all 13 target strings were verified present in the
file** before being listed, so none of them send you hunting for text that is not there.

Screenshots could not be written to disk either: the browser tooling returns images into the
conversation but cannot save PNG files. The six placeholders on slide 6 therefore remain the
user's to capture, with a checklist provided. Each of the six screens was confirmed to render
correctly at 1400x880.

#### Corrections the deck needs

Ten numeric fixes (chiefly `22 features` → `23`, `0.12` → `0.10`, `8 rules` → `10`,
`3 queries` → `5`), three substantive claim changes (secondary research is now on by default
and scored; cash flow is margin-relative; the Live Research tab is renamed), and two demo
pricing bands. One claim that was **false in v1 is now true**: the Borderline scenario really
does score 50-74.

---

### Phase 6 — original plan (for reference)

Do last, once behaviour is settled.

- [ ] README: architecture diagram, accurate module list, honest capability table with a
      "real logic / manual input / placeholder" column.
- [ ] Update `IntelliCredit_Technical_Report.docx`: TF-IDF claim, ML anchor blend, the
      dead-weighting fix, new Phase 2/3 capabilities.
- [ ] Update the deck: same corrections, **and fill the six `[INSERT SCREENSHOT]`
      placeholders on slide 6** — they are still empty in the submitted file.
- [ ] Add a "Known Limitations" section stating synthetic training data plainly. Saying it
      first is far stronger than being caught on it.

---

## 4. Sequencing

```
Phase 0  ──►  Phase 1  ──►  Phase 2  ──►  Phase 6
                  │              │
                  └──► Phase 3 ──┤
                       Phase 4 ──┤
                       Phase 5 ──┘
```

Phase 0 gates everything. Phase 1 before Phase 2 (no point scoring news signals on top of
fabricated financials). Phases 3, 4, 5 are independent of each other and can be reordered
or dropped by available time. Phase 6 last, always.

**If time is short, do Phase 0 + 1 + 2 and stop.** That fixes both weak judged criteria and
every credibility landmine. Phases 3–5 add breadth, not defensibility.

---

## 4b. Follow-ups surfaced by Phase 4

- **Demo narratives are too short for retrieval to matter.** ~70 words each produces one
  chunk, so IDF is degenerate and the RAG layer barely engages in demo mode. Lengthening
  the three scenario texts to a few hundred words would make retrieval visible on stage.
- **Absolute thresholds elsewhere.** Cash flow is now a margin, but the leverage and
  liability rules are already ratios, and the INR 50 lakh limit floor remains absolute by
  design. Worth a pass to confirm nothing else is size-blind.
- **PyPDF2 is deprecated** and warns on every run; migrating to `pypdf` is a small change.

---

## 5. Explicitly out of scope

- Real CIBIL / bureau feed (no prototype access)
- Live GSTN API (requires registered GSP credentials)
- e-Courts scraping (terms of use; manual + news proxy instead)
- Training on real lending data (none available — synthetic stays, but gets labelled)
- Auth, multi-tenancy, persistence — single-analyst local tool remains the target

---

## 6. Risks

| Risk | Mitigation |
|---|---|
| Phase 1.1 + 1.4 will move demo scores; a demo could start failing mid-rebuild | Phase 0 golden tests make the movement visible and intentional; re-baseline in the same commit |
| Live research fails on demo day (no wifi / RSS blocked) | Bundled offline fixture headlines, labelled as such; never a hard dependency |
| Scope creep across 6 phases | Phase 0+1+2 is the committed core; 3–5 are opt-in |
| OCR still unavailable on judge machines (no Tesseract binary) | Declare `pypdfium2`, detect the missing Tesseract binary explicitly, show an actionable message rather than a silent fallback |

---

## 7. Decision (settled 2026-09-12)

**Target: interview and portfolio defensibility** — code that survives technical scrutiny.
The hackathon submission has shipped, so the rubric is no longer the constraint; the
constraint is whether a sharp reviewer can find something embarrassing.

**Consequences for ordering:**

- **Phase 4 moves up, ahead of Phase 2.** The anchor blend, the global-vs-per-instance
  importances, and the TF-IDF claim are precisely the things a technical interviewer
  will find. They outrank breadth.
- **Phase 0 is non-negotiable.** "There are no tests" is itself an interview finding,
  independent of what the tests protect.
- **Phase 3 narrows to 3.3 (sanction-letter exposure netting).** It is the item with real
  credit-domain reasoning behind it. 3.1 and 3.2 were rubric plays; they become optional.
- **Phase 5 becomes optional.** A Databricks seam matters to judges reading a spec. It
  matters far less to someone reading the code.

**Revised critical path:**

```
Phase 0  ──►  Phase 1  ──►  Phase 4  ──►  Phase 2  ──►  Phase 6
                                              │
                              (optional) 3.3 ─┤
                              (optional) 3.1  │
                              (optional) 5   ─┘
```

**Definition of done for the committed core:** Phases 0, 1, 4, 2, 6. At that point every
claim in the docs is true of the code, no number on screen is fabricated, every decision
explains itself per-borrower, and the research agent does something real.
