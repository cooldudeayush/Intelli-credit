"""Golden-output snapshot of the three demo scenarios.

These values are the Phase 0 baseline: what the pipeline produces *today*, bugs
included. The point is not that they are correct — several are provably wrong
(see ``test_document_intel.py``) — but that any change to them becomes visible
and deliberate.

WHEN A PHASE 1+ CHANGE MOVES THESE NUMBERS:
  1. Confirm the movement is the one you intended.
  2. Update the constants below in the *same commit* as the code change.
  3. Note the before/after in the commit message.

Never update a golden value to make a red test green without doing step 1.
"""

from __future__ import annotations

import pytest

# Baseline re-captured 2026-09-13 after Phase 1 (extraction, scoring, RAG weights)
# against: pandas 3.0.5 · scikit-learn 1.9.1 · numpy 2.5.3 · Python 3.13.7
#
# Movement from the Phase 0 baseline, all intended:
#   Good Borrower   Capital 10 -> 20, total 78.3 -> 88.3, limit 50L -> 5.66Cr
#                   (revenue now reads 1,240 crore instead of the year 2025)
#   Risky Borrower  Capacity 12 -> 2, total 40 -> 30, limit 61L -> 50L floor
#                   (-45 crore OCF now penalised -8 instead of boosted +2)
GOLDEN = {
    "Good Borrower": {
        "five_cs": {
            "Character": 16.3,
            "Capacity": 20,
            "Capital": 20,
            "Collateral": 16.0,
            "Conditions": 16.0,
        },
        "total_score": 88.3,
        "risk_category": "Low",
        "scorecard_decision": "Lend",
        "scorecard_limit": 56_600_000,
        "ml_decision": "Lend",
        "final_decision": "Lend",
        "hybrid_limit": 62_800_000,
        "hybrid_pricing_band": "9.25% - 10.75%",
        "recon": ("OK", "Low", "Likely No"),
        "adjustment_counts": {"qual": 3, "rag": 0, "all": 4},
    },
    "Risky Borrower": {
        "five_cs": {
            "Character": 0,
            # Phase 2: 2 -> 0. The widened qualitative rules now detect
            # "heavy concentration in two counterparties" in the site-visit notes.
            "Capacity": 0,
            "Capital": 10,
            "Collateral": 11,
            "Conditions": 7,
        },
        "total_score": 28,
        "risk_category": "High",
        "scorecard_decision": "Reject",
        "scorecard_limit": 9_800_000,
        "ml_decision": "Reject",
        "final_decision": "Reject",
        "hybrid_limit": 5_000_000,
        "hybrid_pricing_band": "14.0% - 16.0%",
        "recon": ("High Mismatch", "High", "Possible"),
        # qual went 0 -> 2 in Phase 2 (promoter concern + receivable
        # concentration are now recognised in natural phrasing); rag went
        # 1 -> 2 in Phase 4 (tokenised queries made "related party" matchable).
        # "all" is qual(2) + rag(2) + one research-agent governance signal row.
        # news is 0 because the demo harness runs with research disabled, which
        # keeps the golden values deterministic.
        "adjustment_counts": {"qual": 2, "rag": 2, "all": 5},
    },
    "Borderline / Manual Review": {
        "five_cs": {
            "Character": 15.1,
            # Phase 4: 14 -> 8. INR 42cr of cash flow on INR 820cr of revenue is a
            # 5.1% margin, which is thin. The old absolute threshold read it as
            # "strong" and awarded +2; it now takes the -4 thin-margin penalty.
            "Capacity": 8.0,
            "Capital": 17,
            "Collateral": 14.0,
            "Conditions": 17.0,
        },
        # This scenario is now genuinely borderline, matching its name and the
        # 50-74 band the deck claims. It reaches Review through real
        # scorecard-vs-ML disagreement rather than the removed anchor formula:
        # the scorecard says Review, ML says Lend at 0.56 confidence, and since
        # that is below the 0.90 override bar the cautious option wins. This is
        # the only demo that exercises the hybrid conflict path.
        "total_score": 71.1,
        "risk_category": "Moderate",
        "scorecard_decision": "Review",
        "scorecard_limit": 29_900_000,
        "ml_decision": "Lend",
        "final_decision": "Review",
        "hybrid_limit": 30_200_000,
        "hybrid_pricing_band": "11.5% - 14.5%",
        "recon": ("High Mismatch", "Low", "Possible"),
        "adjustment_counts": {"qual": 3, "rag": 0, "all": 4},
    },
}

SCENARIOS = list(GOLDEN)


@pytest.mark.parametrize("name", SCENARIOS)
def test_five_cs_scores(pipeline_results, name):
    actual = pipeline_results[name]["score_bundle"]["five_cs_scores"]
    for factor, expected in GOLDEN[name]["five_cs"].items():
        assert actual[factor] == pytest.approx(expected, abs=0.05), (
            f"{name} / {factor}: expected {expected}, got {actual[factor]}"
        )


@pytest.mark.parametrize("name", SCENARIOS)
def test_total_score_and_risk_band(pipeline_results, name):
    bundle = pipeline_results[name]["score_bundle"]
    expected = GOLDEN[name]
    assert bundle["total_score"] == pytest.approx(expected["total_score"], abs=0.05)
    assert bundle["risk_category"] == expected["risk_category"]


@pytest.mark.parametrize("name", SCENARIOS)
def test_scorecard_recommendation(pipeline_results, name):
    reco = pipeline_results[name]["recommendation"]
    expected = GOLDEN[name]
    assert reco["decision"] == expected["scorecard_decision"]
    assert reco["recommended_limit"] == expected["scorecard_limit"]


@pytest.mark.parametrize("name", SCENARIOS)
def test_ml_and_hybrid_decision(pipeline_results, name):
    result = pipeline_results[name]
    expected = GOLDEN[name]
    assert result["ml_view"]["ml_decision"] == expected["ml_decision"]
    assert result["hybrid_view"]["final_decision"] == expected["final_decision"]
    assert result["hybrid_view"]["hybrid_limit"] == expected["hybrid_limit"]
    assert result["hybrid_view"]["hybrid_pricing_band"] == expected["hybrid_pricing_band"]


@pytest.mark.parametrize("name", SCENARIOS)
def test_reconciliation_flags(pipeline_results, name):
    recon = pipeline_results[name]["recon"]
    mismatch, circular, inflated = GOLDEN[name]["recon"]
    assert recon["mismatch_flag"] == mismatch
    assert recon["circular_flag"] == circular
    assert recon["inflated_revenue_flag"] == inflated


@pytest.mark.parametrize("name", SCENARIOS)
def test_adjustment_counts(pipeline_results, name):
    result = pipeline_results[name]
    expected = GOLDEN[name]["adjustment_counts"]
    assert len(result["qual_adjustments"]) == expected["qual"]
    assert len(result["rag_adjustments"]) == expected["rag"]
    assert len(result["all_adjustments"]) == expected["all"]


@pytest.mark.parametrize("name", SCENARIOS)
def test_cam_generates_all_29_sections(pipeline_results, name):
    """The CAM must contain every numbered section, not just render without error."""
    cam = pipeline_results[name]["cam_text"]
    missing = [n for n in range(1, 30) if f"{n}. " not in cam]
    assert not missing, f"{name}: CAM missing sections {missing}"


@pytest.mark.parametrize("name", SCENARIOS)
def test_every_adjustment_is_explainable(pipeline_results, name):
    """Explainability is the project's strongest claim — enforce it.

    Every scoring adjustment must carry a factor, a signed point value, and a
    non-empty human-readable reason. A silent adjustment is a regression.
    """
    valid_factors = {"Character", "Capacity", "Capital", "Collateral", "Conditions"}
    for adj in pipeline_results[name]["score_bundle"]["adjustments"]:
        assert adj["factor"] in valid_factors, f"{name}: bad factor {adj['factor']!r}"
        assert adj["type"] in {"Penalty", "Boost"}, f"{name}: bad type {adj['type']!r}"
        assert isinstance(adj["points"], (int, float))
        assert str(adj.get("reason", "")).strip(), f"{name}: adjustment without a reason: {adj}"


def test_determinism():
    """Two runs of the same scenario must agree exactly.

    The pipeline seeds NumPy and sklearn at 42; if this ever fails, something
    has introduced nondeterminism and the golden values become meaningless.
    """
    from tests.pipeline_harness import run_demo_pipeline

    first = run_demo_pipeline("Risky Borrower")
    second = run_demo_pipeline("Risky Borrower")
    assert first["score_bundle"]["total_score"] == second["score_bundle"]["total_score"]
    assert first["ml_view"]["class_probabilities"] == second["ml_view"]["class_probabilities"]
    assert first["hybrid_view"]["hybrid_limit"] == second["hybrid_view"]["hybrid_limit"]


def test_risk_ordering_holds():
    """Whatever the absolute numbers, the risky borrower must score worst.

    This is the one assertion that should survive every future refactor — it
    encodes intent rather than current behaviour.
    """
    from tests.pipeline_harness import run_demo_pipeline

    good = run_demo_pipeline("Good Borrower")["score_bundle"]["total_score"]
    risky = run_demo_pipeline("Risky Borrower")["score_bundle"]["total_score"]
    assert risky < good, f"Risky ({risky}) should score below Good ({good})"
