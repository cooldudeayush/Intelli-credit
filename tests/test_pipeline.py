"""End-to-end tests for the extracted analysis pipeline.

Phase 1.1 made missing financial figures explicit rather than silently defaulted.
These tests exercise the path where extraction genuinely fails, which is what a
messy scanned PDF produces and what the old code masked with hardcoded values.
"""

from __future__ import annotations

import pandas as pd
import pytest

from modules import AnalysisInputs, empty_source_result, run_analysis


def _bank_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range("2025-04-01", periods=12, freq="ME"),
            "inflow": [20_000_000] * 12,
            "outflow": [8_000_000] * 12,
            "balance": [50_000_000] * 12,
        }
    )


def _gst_df() -> pd.DataFrame:
    return pd.DataFrame({"month": [f"M{i}" for i in range(12)], "taxable_turnover": [20_000_000] * 12})


def _inputs(report_text: str) -> AnalysisInputs:
    return AnalysisInputs(
        company_name="Testco Pvt Ltd",
        sector="Manufacturing",
        bank_df=_bank_df(),
        gst_df=_gst_df(),
        primary_notes="Plant visit completed.",
        external_notes="",
        annual_result={
            "text": report_text,
            "uploaded": True,
            "parsed": bool(report_text),
            "method": "native",
            "message": "test",
            "chars": len(report_text),
        },
    )


@pytest.fixture(scope="module")
def unparseable(model_bundle):
    """A document that parsed, but contains no financial figures at all."""
    return run_analysis(
        _inputs(
            "The board convened and approved the minutes of the previous meeting. "
            "Directors discussed the agenda and resolved to reconvene next quarter."
        ),
        model_bundle=model_bundle,
    )


def test_pipeline_completes_without_financial_figures(unparseable):
    """The whole appraisal must survive a document it could not read."""
    assert unparseable["score_bundle"]["total_score"] >= 0
    assert unparseable["hybrid_view"]["final_decision"] in {"Lend", "Review", "Reject"}
    assert unparseable["cam_text"]


def test_missing_figures_are_declared_not_defaulted(unparseable):
    signals = unparseable["doc_signals"]
    assert signals["revenue"] is None
    assert signals["operating_cash_flow"] is None
    assert signals["cash_flow_indicator"] == "Unknown"
    assert len(signals["data_gaps"]) == 4


def test_data_gaps_reach_the_score_bundle(unparseable):
    """The score must carry forward what it had to do without."""
    gaps = unparseable["score_bundle"]["data_gaps"]
    assert gaps
    assert any("cash flow" in gap.lower() for gap in gaps)
    assert any("leverage" in gap.lower() for gap in gaps)


def test_cam_states_the_gap_rather_than_printing_zero(unparseable):
    cam = unparseable["cam_text"]
    assert "[NOT EXTRACTED]" in cam
    assert "INR 0" not in cam.split("4. GST-Bank Reconciliation")[0], (
        "Financial summary printed a false zero for a missing figure"
    )


def test_missing_cash_flow_does_not_penalise_capacity(unparseable, model_bundle):
    """An unreadable document must not look like negative cash flow.

    Under the old code a missing OCF defaulted to a value; had it defaulted to
    zero it would have read as 'tight' and cost the borrower 4 points for a
    parsing failure.
    """
    healthy = run_analysis(
        _inputs("Cash flow from operations was INR 200 crore for the year."),
        model_bundle=model_bundle,
    )
    assert (
        unparseable["score_bundle"]["five_cs_scores"]["Capacity"]
        >= healthy["score_bundle"]["five_cs_scores"]["Capacity"] - 2
    )


def test_limit_falls_back_to_bank_throughput_when_revenue_is_missing(unparseable):
    """Bank throughput is the conservative anchor; it must still produce a limit."""
    assert unparseable["recommendation"]["recommended_limit"] >= 5_000_000
    assert "bank-throughput cap alone" in unparseable["recommendation"]["limit_rationale"]
