from __future__ import annotations

from typing import Dict


def generate_credit_recommendation(
    score_bundle: Dict,
    doc_signals: Dict,
    recon: Dict,
) -> Dict:
    """Translate risk score into decision, pricing, and limit guidance."""
    total_score = score_bundle.get("total_score", 0)
    risk_category = score_bundle.get("risk_category", "High")

    revenue = doc_signals.get("revenue", 0)
    bank_inflows = recon.get("bank_inflows", 0)

    decision = "Reject"
    interest_band = "15.0%+"

    if risk_category == "Low":
        decision = "Lend"
        interest_band = "9.5% - 11.0%"
    elif risk_category == "Moderate":
        decision = "Review"
        interest_band = "11.0% - 13.5%"

    throughput_cap = bank_inflows * 0.2
    revenue_cap = revenue * 0.12
    base_limit = min(revenue_cap, throughput_cap)
    if decision == "Review":
        base_limit *= 0.75
    if decision == "Reject":
        base_limit *= 0.35

    recommended_limit = max(5_000_000, round(base_limit / 100_000) * 100_000)

    plain_english = (
        f"Decision: {decision}. The borrower scored {total_score}/100 ({risk_category} risk). "
        f"Observed business throughput supports an indicative limit near INR {recommended_limit:,.0f}. "
        f"Pricing guidance: {interest_band}, subject to final collateral and policy checks."
    )

    if decision == "Reject":
        plain_english = (
            f"Decision: {decision}. The score is {total_score}/100 with elevated risk signals. "
            "Recommend declining fresh exposure unless major mitigants are provided."
        )

    decision_rationale = {
        "Lend": "Low risk profile with stable cash flow and acceptable reconciliation quality.",
        "Review": "Mixed risk profile; exposure can be considered with tighter monitoring and covenants.",
        "Reject": "High risk profile due to weak score and/or critical risk flags.",
    }[decision]

    limit_rationale = (
        f"Limit derived from lower of revenue-based cap ({revenue_cap:,.0f}) and "
        f"bank-throughput cap ({throughput_cap:,.0f}), then adjusted for decision band."
    )
    if decision == "Reject":
        limit_rationale += " Conservative haircut applied for elevated risk."
    elif decision == "Review":
        limit_rationale += " Prudential moderation applied for review category."
    else:
        limit_rationale += " No risk haircut beyond standard policy cap."

    pricing_rationale = (
        "Pricing band selected from risk category and decision outcome; "
        "higher risk bands carry additional risk premium."
    )

    return {
        "decision": decision,
        "recommended_limit": recommended_limit,
        "interest_band": interest_band,
        "explanation": plain_english,
        "decision_rationale": decision_rationale,
        "limit_rationale": limit_rationale,
        "pricing_rationale": pricing_rationale,
    }
