from __future__ import annotations

from typing import Dict, List


def build_research_agent_findings(
    manual_inputs: Dict,
    base_insights: Dict,
    rag_theme_results: Dict,
    primary_notes: str,
    external_notes: str,
) -> Dict:
    """Create a structured research panel output (rule-based, non-autonomous)."""
    promoter_obs = (manual_inputs.get("promoter_obs") or "").strip()
    sector_obs = (manual_inputs.get("sector_obs") or "").strip()
    regulatory_obs = (manual_inputs.get("regulatory_obs") or "").strip()
    mca_notes = (manual_inputs.get("mca_notes") or "").strip()
    litigation_notes = (manual_inputs.get("litigation_notes") or "").strip()

    def _theme_hits(theme: str) -> int:
        return len((rag_theme_results.get(theme) or {}).get("snippets", []))

    cards = {
        "Promoter / Management Risk": promoter_obs or base_insights.get("promoter_news", "No explicit promoter note."),
        "Sector Headwinds": sector_obs or base_insights.get("sector_headwind", "No sector headwind provided."),
        "Litigation Summary": litigation_notes or base_insights.get("litigation_signal", "No litigation note provided."),
        "Regulatory Alerts": regulatory_obs or f"Base signal: {base_insights.get('regulatory_risk', 'Low')}",
        "MCA Notes": mca_notes or "No MCA note provided (manual placeholder).",
        "External Intelligence Notes": external_notes or "No external intelligence notes provided.",
        "Primary Due Diligence Notes": primary_notes or "No primary due diligence note provided.",
    }

    signal_rows: List[Dict] = []
    if _theme_hits("Governance & Promoter Concerns") > 0:
        signal_rows.append(
            {
                "source": "Research Agent",
                "origin": "unstructured_rag",
                "cue": "Promoter/governance risk evidence retrieved",
                "five_c": "Character",
                "score_effect": -1,
                "reason": "RAG found governance/promoter cues in uploaded documents.",
            }
        )
    if _theme_hits("Regulatory / Sector Risk Signals") > 0:
        signal_rows.append(
            {
                "source": "Research Agent",
                "origin": "unstructured_rag",
                "cue": "Regulatory/sector pressure evidence retrieved",
                "five_c": "Conditions",
                "score_effect": -1,
                "reason": "RAG surfaced regulatory/sector stress commentary.",
            }
        )

    summary = "\n".join(
        [
            f"Promoter risk: {cards['Promoter / Management Risk']}",
            f"Sector signal: {cards['Sector Headwinds']}",
            f"Litigation signal: {cards['Litigation Summary']}",
            f"Regulatory signal: {cards['Regulatory Alerts']}",
            f"MCA note: {cards['MCA Notes']}",
        ]
    )

    return {
        "cards": cards,
        "signal_rows": signal_rows,
        "summary": summary,
    }
