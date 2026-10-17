from __future__ import annotations

from typing import Dict, List


def _bounded(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def compute_five_cs_score(
    doc_signals: Dict,
    recon: Dict,
    insights: Dict,
    manual_notes: str,
    qualitative_adjustments: List[Dict] | None = None,
) -> Dict:
    """Explainable rules-based scoring for the Five Cs of Credit (out of 100)."""
    revenue = doc_signals.get("revenue")
    debt = doc_signals.get("debt")
    liabilities = doc_signals.get("liabilities")
    ocf = doc_signals.get("operating_cash_flow")

    # A figure that could not be extracted is unknown, not zero. Treating it as
    # zero would read "no revenue" or "negative cash flow" and penalise a
    # borrower for a parsing failure, so every ratio below is skipped instead
    # and the omission recorded in data_gaps.
    has_revenue = revenue is not None and revenue > 0
    leverage_ratio = debt / revenue if has_revenue and debt is not None else None
    liability_ratio = liabilities / revenue if has_revenue and liabilities is not None else None

    adjustments: List[Dict] = []
    data_gaps: List[str] = list(doc_signals.get("data_gaps", []))

    character = 20
    litigation_penalty = min(8, doc_signals.get("litigation_mentions", 0) * 1.5)
    governance_penalty = min(6, doc_signals.get("governance_mentions", 0) * 1.2)
    character -= litigation_penalty
    character -= governance_penalty
    if litigation_penalty > 0:
        adjustments.append({"factor": "Character", "type": "Penalty", "points": -litigation_penalty, "reason": "Litigation mentions in report"})
    if governance_penalty > 0:
        adjustments.append({"factor": "Character", "type": "Penalty", "points": -governance_penalty, "reason": "Governance/promoter risk cues"})
    if insights.get("promoter_news_level") == "Moderate":
        character -= 3
        adjustments.append({"factor": "Character", "type": "Penalty", "points": -3, "reason": "Moderate promoter negative news signal"})
    if doc_signals.get("litigation_mentions", 0) >= 4:
        character -= 3
        adjustments.append({"factor": "Character", "type": "Penalty", "points": -3, "reason": "Elevated unresolved legal intensity"})
    if doc_signals.get("governance_mentions", 0) >= 3:
        character -= 2
        adjustments.append({"factor": "Character", "type": "Penalty", "points": -2, "reason": "Multiple governance warning mentions"})

    capacity = 20
    # Cash-flow strength is judged as a margin on revenue rather than an absolute
    # rupee amount, so the same rule reads correctly for a INR 200 crore borrower
    # and an INR 2,000 crore one. See document_intel for the band definitions.
    cash_flow_band = doc_signals.get("cash_flow_indicator")
    if ocf is None:
        data_gaps.append("Operating cash flow unavailable - no Capacity cash-flow adjustment applied.")
    elif cash_flow_band == "Stress":
        capacity -= 8
        adjustments.append({"factor": "Capacity", "type": "Penalty", "points": -8, "reason": "Negative operating cash flow"})
    elif cash_flow_band == "Tight":
        capacity -= 4
        reason = "Tight operating cash flow"
        if has_revenue:
            reason = f"Operating cash flow is only {ocf / revenue:.1%} of revenue"
        adjustments.append({"factor": "Capacity", "type": "Penalty", "points": -4, "reason": reason})
    elif cash_flow_band == "Healthy":
        capacity += 2
        reason = "Strong operating cash flow"
        if has_revenue:
            reason = f"Operating cash flow is {ocf / revenue:.1%} of revenue"
        adjustments.append({"factor": "Capacity", "type": "Boost", "points": 2, "reason": reason})
    if recon.get("mismatch_flag") == "High Mismatch":
        capacity -= 6
        adjustments.append({"factor": "Capacity", "type": "Penalty", "points": -6, "reason": "High GST-bank mismatch"})
    elif recon.get("mismatch_flag") == "Moderate Mismatch":
        capacity -= 3
        adjustments.append({"factor": "Capacity", "type": "Penalty", "points": -3, "reason": "Moderate GST-bank mismatch"})
    if recon.get("inflated_revenue_flag") == "Possible":
        capacity -= 4
        adjustments.append({"factor": "Capacity", "type": "Penalty", "points": -4, "reason": "Possible inflated revenue signal"})

    capital = 20
    if leverage_ratio is None:
        data_gaps.append("Debt or revenue unavailable - no leverage adjustment applied to Capital.")
    elif leverage_ratio > 0.6:
        capital -= 6
        adjustments.append({"factor": "Capital", "type": "Penalty", "points": -6, "reason": "Elevated debt to revenue ratio"})
    elif leverage_ratio > 0.4:
        capital -= 3
        adjustments.append({"factor": "Capital", "type": "Penalty", "points": -3, "reason": "Moderate debt to revenue ratio"})
    elif leverage_ratio < 0.25:
        capital += 2
        adjustments.append({"factor": "Capital", "type": "Boost", "points": 2, "reason": "Low leverage profile"})
    if liability_ratio is None:
        data_gaps.append("Liabilities or revenue unavailable - no liability-ratio adjustment applied to Capital.")
    elif liability_ratio > 0.8:
        capital -= 4
        adjustments.append({"factor": "Capital", "type": "Penalty", "points": -4, "reason": "High liabilities to revenue"})

    collateral = 20
    lower_notes = (manual_notes or "").lower()
    if "unsecured" in lower_notes:
        collateral -= 7
        adjustments.append({"factor": "Collateral", "type": "Penalty", "points": -7, "reason": "Exposure appears unsecured"})
    if "inventory" in lower_notes:
        collateral -= 2
        adjustments.append({"factor": "Collateral", "type": "Penalty", "points": -2, "reason": "Inventory-heavy collateral quality"})
    if "property" in lower_notes or "plant" in lower_notes:
        collateral += 1
        adjustments.append({"factor": "Collateral", "type": "Boost", "points": 1, "reason": "Tangible property/plant collateral available"})

    conditions = 20
    sector_penalty = int(insights.get("sector_risk_points", 10) / 2)
    conditions -= sector_penalty
    adjustments.append({"factor": "Conditions", "type": "Penalty", "points": -sector_penalty, "reason": "Sector headwind baseline risk"})
    if insights.get("regulatory_risk") == "Moderate":
        conditions -= 2
        adjustments.append({"factor": "Conditions", "type": "Penalty", "points": -2, "reason": "Moderate regulatory risk"})
    if recon.get("circular_flag") == "High":
        conditions -= 4
        adjustments.append({"factor": "Conditions", "type": "Penalty", "points": -4, "reason": "High circular trading pattern"})
    elif recon.get("circular_flag") == "Watch":
        conditions -= 2
        adjustments.append({"factor": "Conditions", "type": "Penalty", "points": -2, "reason": "Watchlist circular trading behavior"})
    elif recon.get("circular_flag") == "Low":
        conditions += 1
        adjustments.append({"factor": "Conditions", "type": "Boost", "points": 1, "reason": "Low circular trading indication"})

    scores = {
        "Character": _bounded(character, 0, 20),
        "Capacity": _bounded(capacity, 0, 20),
        "Capital": _bounded(capital, 0, 20),
        "Collateral": _bounded(collateral, 0, 20),
        "Conditions": _bounded(conditions, 0, 20),
    }

    # Apply explicit note-derived adjustments after baseline rule scoring.
    for item in qualitative_adjustments or []:
        factor = item.get("factor")
        points = float(item.get("points", 0))
        if factor in scores:
            scores[factor] = _bounded(scores[factor] + points, 0, 20)
            adjustments.append(
                {
                    "factor": factor,
                    "type": "Boost" if points > 0 else "Penalty",
                    "points": points,
                    "reason": f"{item.get('signal', 'Qualitative note')} ({item.get('source', 'Notes')})",
                }
            )

    total = round(sum(scores.values()), 1)

    risk_category = "High"
    if total >= 75:
        risk_category = "Low"
    elif total >= 50:
        risk_category = "Moderate"

    reasons: List[str] = []
    positive_drivers: List[str] = []
    negative_drivers: List[str] = []
    if recon.get("mismatch_flag") != "OK":
        negative_drivers.append(f"GST-bank mismatch flagged: {recon.get('mismatch_flag')}")
    if doc_signals.get("litigation_mentions", 0) > 0:
        negative_drivers.append(f"Litigation mentions in documents: {doc_signals.get('litigation_mentions')}")
    if doc_signals.get("governance_mentions", 0) > 0:
        negative_drivers.append(f"Governance risk cues observed: {doc_signals.get('governance_mentions')}")
    if cash_flow_band in {"Stress", "Tight"}:
        negative_drivers.append("Operating cash flow appears tight relative to revenue")
    elif cash_flow_band == "Healthy":
        positive_drivers.append("Operating cash flow trend is strong")
    if insights.get("regulatory_risk") == "Moderate":
        negative_drivers.append("Sector carries moderate regulatory exposure")
    else:
        positive_drivers.append("Regulatory exposure remains manageable")
    if recon.get("mismatch_flag") == "OK":
        positive_drivers.append("GST and banking flows are broadly aligned")
    if recon.get("circular_flag") == "Low":
        positive_drivers.append("No material circular trading pattern observed")
    if leverage_ratio is not None and leverage_ratio < 0.35:
        positive_drivers.append("Leverage profile is relatively controlled")
    for item in qualitative_adjustments or []:
        if float(item.get("points", 0)) > 0:
            positive_drivers.append(f"{item.get('signal')} ({item.get('source')})")
        elif float(item.get("points", 0)) < 0:
            negative_drivers.append(f"{item.get('signal')} ({item.get('source')})")

    reasons.extend(negative_drivers[:4])
    if not reasons:
        reasons.append("Financial and qualitative checks are broadly stable")

    if data_gaps:
        negative_drivers.append(
            f"Score computed with {len(data_gaps)} missing input(s) - see data gaps"
        )

    return {
        "five_cs_scores": scores,
        "total_score": total,
        "risk_category": risk_category,
        "top_reasons": reasons[:5],
        "positive_drivers": positive_drivers[:5],
        "negative_drivers": negative_drivers[:5],
        "adjustments": adjustments,
        "data_gaps": data_gaps,
    }
