from __future__ import annotations

from typing import Dict


RANK = {"Reject": 0, "Review": 1, "Lend": 2}


def _risk_price(decision: str) -> str:
    if decision == "Lend":
        return "9.25% - 10.75%"
    if decision == "Review":
        return "10.75% - 13.25%"
    return "13.5% - 15.5%"


def combine_scorecard_ml(
    scorecard_reco: Dict,
    ml_view: Dict,
    doc_signals: Dict,
    qualitative_adjustments: list,
    rag_adjustments: list,
) -> Dict:
    """Combine rule-scorecard and ML outputs into final hybrid decision."""
    scorecard_decision = scorecard_reco.get("decision", "Review")
    ml_decision = ml_view.get("ml_decision", "Review")
    ml_conf = float(ml_view.get("ml_confidence", 0.5))

    if scorecard_decision == ml_decision:
        final_decision = scorecard_decision
        decision_note = "Scorecard and ML are aligned."
    else:
        # Practical conflict policy: use ML override only at strong confidence.
        if ml_conf >= 0.90:
            final_decision = scorecard_decision if RANK[scorecard_decision] < RANK[ml_decision] else ml_decision
            decision_note = (
                f"Scorecard ({scorecard_decision}) and ML ({ml_decision}) disagreed; "
                f"high ML confidence ({ml_conf:.2f}) triggered conservative override: {final_decision}."
            )
        else:
            # Blend to middle ground to avoid noisy hard overrides in prototype setting.
            if {scorecard_decision, ml_decision} == {"Lend", "Reject"}:
                final_decision = "Lend" if scorecard_decision == "Lend" and ml_conf < 0.82 else "Review"
            else:
                final_decision = scorecard_decision if RANK[scorecard_decision] < RANK[ml_decision] else ml_decision
            decision_note = (
                f"Scorecard ({scorecard_decision}) and ML ({ml_decision}) disagreed; "
                f"moderate ML confidence ({ml_conf:.2f}) used blended decision: {final_decision}."
            )

    base_limit = float(scorecard_reco.get("recommended_limit", 5_000_000))

    collateral_score = float(doc_signals.get("liabilities", 0))
    cashflow = float(doc_signals.get("operating_cash_flow", 0))
    governance = float(doc_signals.get("governance_mentions", 0))
    litigation = float(doc_signals.get("litigation_mentions", 0))

    multiplier = 1.0
    if final_decision == "Lend":
        multiplier += 0.10
    if final_decision == "Reject":
        multiplier -= 0.35
    if cashflow > 120_000_000:
        multiplier += 0.08
    if cashflow < 40_000_000:
        multiplier -= 0.12
    if governance > 0 or litigation > 0:
        multiplier -= 0.08
    if len(rag_adjustments) >= 2:
        multiplier -= 0.06
    if sum(float(x.get("points", 0)) for x in qualitative_adjustments) > 2:
        multiplier += 0.04
    if collateral_score > 700_000_000:
        multiplier -= 0.03

    multiplier = max(0.45, min(1.30, multiplier))
    hybrid_limit = round((base_limit * multiplier) / 100_000) * 100_000
    hybrid_limit = max(5_000_000, int(hybrid_limit))

    pricing_band = _risk_price(final_decision)
    if litigation + governance >= 3:
        pricing_band = "11.5% - 14.5%" if final_decision != "Reject" else "14.0% - 16.0%"

    qualitative_changed = len(qualitative_adjustments) > 0
    rag_changed = len(rag_adjustments) > 0

    rationale = (
        f"Final hybrid decision: {final_decision}. {decision_note} "
        f"Limit calibrated from scorecard baseline INR {base_limit:,.0f} using cashflow/collateral/"
        f"risk modifiers (x{multiplier:.2f}). Pricing aligned to hybrid risk class."
    )

    return {
        "scorecard_decision": scorecard_decision,
        "ml_decision": ml_decision,
        "final_decision": final_decision,
        "hybrid_limit": hybrid_limit,
        "hybrid_pricing_band": pricing_band,
        "decision_confidence": ml_view.get("ml_confidence", 0.5),
        "alignment_note": decision_note,
        "rationale": rationale,
        "qualitative_changed": qualitative_changed,
        "rag_changed": rag_changed,
    }
