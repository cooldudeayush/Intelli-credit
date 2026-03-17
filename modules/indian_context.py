from __future__ import annotations

from typing import Dict, List


def build_indian_context_cards(
    recon: Dict,
    insights: Dict,
    doc_signals: Dict,
    shareholding_summary: Dict,
    combined_notes: str,
) -> List[Dict]:
    """Generate India-specific credit context indicator cards."""
    notes_lower = (combined_notes or "").lower()

    share_conc = shareholding_summary.get("concentration_flag", "Unknown")
    mca_flag = "Watch" if any(k in notes_lower for k in ["mca", "non-compliance", "non compliance", "roc filing delay"]) else "Low"
    related_party = "Elevated" if doc_signals.get("governance_mentions", 0) >= 2 else "Low"

    return [
        {
            "Indicator": "GST Turnover vs Bank Inflows",
            "Status": recon.get("mismatch_flag", "NA"),
            "Type": "Real Logic",
            "Detail": f"Variance {recon.get('variance_pct', 0):.2f}%",
            "Five C Relevance": "Capacity",
        },
        {
            "Indicator": "Circular Trading / Revenue Inflation",
            "Status": f"{recon.get('circular_flag', 'NA')} / {recon.get('inflated_revenue_flag', 'NA')}",
            "Type": "Real Logic",
            "Detail": "Derived from bank mirror-pattern and GST-bank variance rules",
            "Five C Relevance": "Capacity, Conditions",
        },
        {
            "Indicator": "GSTR-2A vs 3B Interpretation",
            "Status": "Placeholder",
            "Type": "Prototype Placeholder",
            "Detail": "Future-ready module for invoice-level reconciliation support",
            "Five C Relevance": "Capacity",
        },
        {
            "Indicator": "CIBIL Commercial Input",
            "Status": "Placeholder",
            "Type": "Prototype Placeholder",
            "Detail": "Manual ingestible risk input for future bureau integration",
            "Five C Relevance": "Character",
        },
        {
            "Indicator": "MCA Compliance Flag",
            "Status": mca_flag,
            "Type": "Rules + Notes",
            "Detail": "Detected from manual/external notes",
            "Five C Relevance": "Character, Conditions",
        },
        {
            "Indicator": "Related-Party / Promoter Risk",
            "Status": related_party,
            "Type": "Real Logic",
            "Detail": "Based on governance/promoter mentions from documents",
            "Five C Relevance": "Character",
        },
        {
            "Indicator": "Shareholding Concentration",
            "Status": str(share_conc),
            "Type": "Structured Rule",
            "Detail": "Derived from shareholding data when percentage column available",
            "Five C Relevance": "Character",
        },
        {
            "Indicator": "Legal / e-Courts Risk",
            "Status": insights.get("litigation_signal", "Low"),
            "Type": "Rules + RAG Support",
            "Detail": "Litigation signals from document parsing and retrieval evidence",
            "Five C Relevance": "Character",
        },
        {
            "Indicator": "Sector Regulatory Alert",
            "Status": insights.get("regulatory_risk", "Low"),
            "Type": "Real Logic",
            "Detail": insights.get("sector_headwind", "No headwind summary"),
            "Five C Relevance": "Conditions",
        },
    ]
