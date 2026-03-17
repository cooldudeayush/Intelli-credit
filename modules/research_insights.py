from __future__ import annotations

from typing import Dict


SECTOR_RISK_MAP = {
    "Manufacturing": {"headwind": "Input cost volatility and export demand softness.", "risk": 12},
    "Infrastructure": {"headwind": "Execution delays and working capital stretch.", "risk": 14},
    "Retail": {"headwind": "Demand concentration and margin pressure.", "risk": 11},
    "Pharma": {"headwind": "Regulatory observations and price caps.", "risk": 10},
    "IT Services": {"headwind": "Client budget cuts in key geographies.", "risk": 9},
}


def generate_research_insights(
    sector: str,
    document_signals: Dict,
    manual_notes: str,
) -> Dict:
    """Create mock external intelligence summary for a hackathon demo."""
    sector_data = SECTOR_RISK_MAP.get(sector, {"headwind": "Mixed sector outlook.", "risk": 10})

    negative_news_flag = "Low"
    promoter_news_summary = "No major unresolved promoter adverse media observed in mock feed."

    lower_notes = (manual_notes or "").lower()
    if "pledge" in lower_notes or "resignation" in lower_notes:
        negative_news_flag = "Moderate"
        promoter_news_summary = "Signals of governance stress found in analyst notes (pledge/resignation cues)."

    litigation_level = "Low"
    if document_signals.get("litigation_mentions", 0) >= 3:
        litigation_level = "Moderate"
    if document_signals.get("litigation_mentions", 0) >= 6:
        litigation_level = "High"

    regulatory_risk = "Moderate" if sector in {"Pharma", "Infrastructure"} else "Low"

    summary_lines = [
        f"Sector headwind: {sector_data['headwind']}",
        f"Promoter negative news risk: {negative_news_flag}",
        f"Litigation signal level: {litigation_level}",
        f"Regulatory risk: {regulatory_risk}",
    ]

    return {
        "sector_headwind": sector_data["headwind"],
        "promoter_news": promoter_news_summary,
        "promoter_news_level": negative_news_flag,
        "litigation_signal": litigation_level,
        "regulatory_risk": regulatory_risk,
        "sector_risk_points": sector_data["risk"],
        "ai_summary": "\n".join(summary_lines),
    }
