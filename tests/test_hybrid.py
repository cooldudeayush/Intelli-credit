"""Unit tests for the scorecard/ML decision fusion engine.

The override policy is the project's central governance claim — that ML cannot
unilaterally overrule the rulebook — so it gets explicit coverage.
"""

from __future__ import annotations

import pytest

from modules.hybrid_recommendation import combine_scorecard_ml

DOC = {
    "revenue": 1_000_000_000,
    "debt": 200_000_000,
    "liabilities": 400_000_000,
    "operating_cash_flow": 80_000_000,
    "litigation_mentions": 0,
    "governance_mentions": 0,
}


def _combine(scorecard: str, ml: str, confidence: float, doc=None, qual=None, rag=None):
    return combine_scorecard_ml(
        scorecard_reco={"decision": scorecard, "recommended_limit": 50_000_000},
        ml_view={"ml_decision": ml, "ml_confidence": confidence},
        doc_signals={**DOC, **(doc or {})},
        qualitative_adjustments=qual or [],
        rag_adjustments=rag or [],
    )


@pytest.mark.parametrize("decision", ["Lend", "Review", "Reject"])
def test_agreement_is_adopted_directly(decision):
    result = _combine(decision, decision, 0.75)
    assert result["final_decision"] == decision
    assert "aligned" in result["alignment_note"].lower()


def test_high_confidence_disagreement_takes_the_cautious_side():
    """The core governance claim: ML at >=90% still resolves conservatively."""
    result = _combine("Review", "Lend", 0.95)
    assert result["final_decision"] == "Review", "ML should not be able to upgrade to Lend"


def test_ml_may_downgrade_when_highly_confident():
    result = _combine("Lend", "Review", 0.95)
    assert result["final_decision"] == "Review"


def test_moderate_confidence_disagreement_resolves_cautiously():
    result = _combine("Lend", "Review", 0.60)
    assert result["final_decision"] == "Review"


def test_extreme_disagreement_below_threshold_trusts_the_scorecard():
    """Lend vs Reject with ML under 82% keeps the scorecard's Lend."""
    result = _combine("Lend", "Reject", 0.70)
    assert result["final_decision"] == "Lend"


def test_extreme_disagreement_above_threshold_steps_down_to_review():
    result = _combine("Lend", "Reject", 0.85)
    assert result["final_decision"] == "Review"


def test_limit_multiplier_is_bounded():
    """However many penalties stack, the limit stays within 0.45x-1.30x."""
    worst = _combine(
        "Reject",
        "Reject",
        0.99,
        doc={"operating_cash_flow": 1, "governance_mentions": 9, "litigation_mentions": 9,
             "liabilities": 900_000_000},
        rag=[{"points": -1}, {"points": -1}, {"points": -1}],
    )
    best = _combine(
        "Lend", "Lend", 0.99,
        doc={"operating_cash_flow": 200_000_000},
        qual=[{"points": 4}],
    )
    assert worst["hybrid_limit"] >= 5_000_000, "Floor of INR 50 lakh must hold"
    assert best["hybrid_limit"] <= 50_000_000 * 1.30 + 100_000


def test_limit_never_falls_below_the_floor():
    result = _combine("Reject", "Reject", 0.99, doc={"operating_cash_flow": 1})
    assert result["hybrid_limit"] >= 5_000_000


def test_pricing_escalates_with_legal_and_governance_load():
    clean = _combine("Review", "Review", 0.8)
    loaded = _combine(
        "Review", "Review", 0.8,
        doc={"litigation_mentions": 2, "governance_mentions": 2},
    )
    assert clean["hybrid_pricing_band"] != loaded["hybrid_pricing_band"]


def test_rationale_and_provenance_flags_are_populated():
    result = _combine("Lend", "Review", 0.7, qual=[{"points": -2}], rag=[{"points": -1}])
    assert result["rationale"].strip()
    assert result["qualitative_changed"] is True
    assert result["rag_changed"] is True
    assert result["scorecard_decision"] == "Lend"
    assert result["ml_decision"] == "Review"
