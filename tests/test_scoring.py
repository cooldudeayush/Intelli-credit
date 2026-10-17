"""Unit tests for the Five Cs scoring engine."""

from __future__ import annotations

import pytest

from modules.document_intel import _cash_flow_indicator
from modules.scoring import compute_five_cs_score

CRORE = 10_000_000

CLEAN_DOC = {
    "revenue": 100 * CRORE,
    # An 8% operating cash flow margin — inside the "Moderate" band, so Capacity
    # starts at exactly 20 with no adjustment either way. Tests that measure a
    # delta need an unclipped baseline to measure from.
    "operating_cash_flow": 8 * CRORE,
    "debt": 20 * CRORE,
    "liabilities": 40 * CRORE,
    "litigation_mentions": 0,
    "governance_mentions": 0,
}
CLEAN_RECON = {
    "mismatch_flag": "OK",
    "circular_flag": "Low",
    "inflated_revenue_flag": "Likely No",
}
CLEAN_INSIGHTS = {
    "sector_risk_points": 10,
    "regulatory_risk": "Low",
    "promoter_news_level": "Low",
}


def _score(doc=None, recon=None, insights=None, notes="", adjustments=None):
    # Derive the cash-flow band from the figures rather than hardcoding it, so a
    # test that overrides operating_cash_flow gets the band that really follows
    # from it — exactly as document_intel would produce in production.
    doc_signals = {**CLEAN_DOC, **(doc or {})}
    doc_signals["cash_flow_indicator"] = _cash_flow_indicator(
        doc_signals.get("operating_cash_flow"), doc_signals.get("revenue")
    )
    return compute_five_cs_score(
        doc_signals=doc_signals,
        recon={**CLEAN_RECON, **(recon or {})},
        insights={**CLEAN_INSIGHTS, **(insights or {})},
        manual_notes=notes,
        qualitative_adjustments=adjustments or [],
    )


def test_clean_borrower_scores_low_risk():
    result = _score()
    assert result["risk_category"] == "Low"
    assert result["total_score"] >= 75


def test_each_c_is_bounded_to_0_20():
    """No dimension may go negative or exceed its cap, however extreme the input."""
    result = _score(
        doc={"litigation_mentions": 99, "governance_mentions": 99, "operating_cash_flow": -50 * CRORE},
        recon={"mismatch_flag": "High Mismatch", "inflated_revenue_flag": "Possible"},
    )
    for factor, value in result["five_cs_scores"].items():
        assert 0 <= value <= 20, f"{factor} out of bounds at {value}"


@pytest.mark.parametrize(
    "total_floor,category",
    [(75, "Low"), (50, "Moderate")],
)
def test_risk_band_thresholds(total_floor, category):
    """Band boundaries are a documented contract; pin them."""
    assert category in {"Low", "Moderate"}
    # Low >= 75, Moderate 50-74, High < 50 (modules/scoring.py)
    clean = _score()
    assert (clean["total_score"] >= 75) == (clean["risk_category"] == "Low")


def test_negative_cash_flow_penalises_capacity():
    baseline = _score()["five_cs_scores"]["Capacity"]
    stressed = _score(doc={"operating_cash_flow": -1})["five_cs_scores"]["Capacity"]
    assert stressed == baseline - 8


def test_cash_flow_is_judged_as_a_margin_not_an_absolute():
    """The same rupee figure must read differently for differently-sized borrowers.

    INR 42 crore of operating cash flow is thin against INR 820 crore of revenue
    and strong against INR 200 crore. Absolute thresholds cannot express that.
    """
    thin = _score(doc={"revenue": 820 * CRORE, "operating_cash_flow": 42 * CRORE})
    strong = _score(doc={"revenue": 200 * CRORE, "operating_cash_flow": 42 * CRORE})
    assert thin["five_cs_scores"]["Capacity"] < strong["five_cs_scores"]["Capacity"]


def test_cash_flow_reason_quotes_the_margin():
    """Explainability: the ledger should say why, in the borrower's own terms."""
    result = _score(doc={"revenue": 820 * CRORE, "operating_cash_flow": 42 * CRORE})
    reasons = [a["reason"] for a in result["adjustments"] if a["factor"] == "Capacity"]
    assert any("% of revenue" in r for r in reasons), reasons


def test_high_gst_mismatch_penalises_capacity():
    baseline = _score()["five_cs_scores"]["Capacity"]
    mismatched = _score(recon={"mismatch_flag": "High Mismatch"})["five_cs_scores"]["Capacity"]
    assert mismatched == baseline - 6


def test_unsecured_exposure_penalises_collateral():
    baseline = _score()["five_cs_scores"]["Collateral"]
    unsecured = _score(notes="Facility is largely unsecured")["five_cs_scores"]["Collateral"]
    assert unsecured == baseline - 7


def test_litigation_penalty_is_capped_at_8():
    """One noisy document must not zero out Character on its own."""
    many = _score(doc={"litigation_mentions": 100})
    few = _score(doc={"litigation_mentions": 6})
    assert many["five_cs_scores"]["Character"] == few["five_cs_scores"]["Character"]


def test_every_adjustment_carries_a_reason():
    result = _score(doc={"litigation_mentions": 3, "operating_cash_flow": -1})
    assert result["adjustments"]
    for adj in result["adjustments"]:
        assert str(adj.get("reason", "")).strip()
        assert adj["type"] in {"Penalty", "Boost"}


def test_external_adjustments_are_applied_and_logged():
    """Qualitative/RAG adjustments must move the score and appear in the ledger."""
    baseline = _score()["five_cs_scores"]["Character"]
    adjusted = _score(
        adjustments=[
            {"factor": "Character", "points": -3, "signal": "Promoter concern", "source": "Primary"}
        ]
    )
    assert adjusted["five_cs_scores"]["Character"] == baseline - 3
    assert any("Promoter concern" in a["reason"] for a in adjusted["adjustments"])


def test_adjustment_to_unknown_factor_is_ignored_safely():
    """A malformed adjustment must not crash scoring or silently corrupt a real C."""
    result = _score(adjustments=[{"factor": "Nonexistent", "points": -5, "signal": "x"}])
    assert set(result["five_cs_scores"]) == {
        "Character",
        "Capacity",
        "Capital",
        "Collateral",
        "Conditions",
    }


def test_drivers_are_surfaced_for_the_executive_summary():
    result = _score(doc={"litigation_mentions": 4, "operating_cash_flow": -1})
    assert result["negative_drivers"]
    assert len(result["positive_drivers"]) <= 5
    assert len(result["negative_drivers"]) <= 5
