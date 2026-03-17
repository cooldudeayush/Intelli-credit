from __future__ import annotations

from typing import Dict


def _fmt_inr(amount: float) -> str:
    return f"INR {amount:,.0f}"


def build_cam_text(payload: Dict) -> str:
    """Generate a structured Credit Appraisal Memo for preview/download."""
    company = payload.get("company_name", "Applicant Company")
    sector = payload.get("sector", "NA")
    doc = payload.get("doc_signals", {})
    recon = payload.get("recon", {})
    insights = payload.get("insights", {})
    score = payload.get("score", {})
    reco = payload.get("recommendation", {})
    notes = payload.get("manual_notes", "")
    completeness_note = payload.get("completeness_note", "Data quality checks not available.")
    data_source_note = payload.get("data_source_note", "Mixed uploaded and fallback data.")
    source_coverage = payload.get("source_coverage", [])
    source_findings = payload.get("source_findings", {})
    primary_notes = payload.get("primary_notes", notes)
    external_notes = payload.get("external_notes", "")
    qualitative_adjustments = payload.get("qualitative_adjustments", [])
    rag_theme_results = payload.get("rag_theme_results", {})
    research_agent = payload.get("research_agent", {})
    indian_context_cards = payload.get("indian_context_cards", [])
    ml_view = payload.get("ml_view", {})
    hybrid_view = payload.get("hybrid_view", {})
    live_research = payload.get("live_research", {})

    lines = [
        "INTELLI-CREDIT | CREDIT APPRAISAL MEMO",
        "=" * 52,
        "",
        "1. Company Overview",
        f"- Applicant: {company}",
        f"- Sector: {sector}",
        "- Purpose: Working capital / term loan assessment (prototype run)",
        "",
        "2. Business and Sector Snapshot",
        f"- Sector Headwinds: {insights.get('sector_headwind', 'NA')}",
        f"- Regulatory Environment: {insights.get('regulatory_risk', 'NA')}",
        "",
        "3. Financial Summary",
        f"- Revenue: {_fmt_inr(doc.get('revenue', 0))}",
        f"- Debt: {_fmt_inr(doc.get('debt', 0))}",
        f"- Liabilities: {_fmt_inr(doc.get('liabilities', 0))}",
        f"- Operating Cash Flow: {_fmt_inr(doc.get('operating_cash_flow', 0))}",
        f"- Cash Flow Indicator: {doc.get('cash_flow_indicator', 'NA')}",
        "",
        "4. GST-Bank Reconciliation Findings",
        f"- GST Turnover: {_fmt_inr(recon.get('gst_turnover', 0))}",
        f"- Bank Inflows: {_fmt_inr(recon.get('bank_inflows', 0))}",
        f"- Variance: {recon.get('variance_pct', 0):.2f}%",
        f"- Mismatch Flag: {recon.get('mismatch_flag', 'NA')}",
        f"- Circular Trading Indicator: {recon.get('circular_flag', 'NA')}",
        f"- Revenue Inflation Indicator: {recon.get('inflated_revenue_flag', 'NA')}",
        "",
        "5. Document Intelligence Findings",
        f"- Litigation Mentions: {doc.get('litigation_mentions', 0)}",
        f"- Governance/Promoter Risk Mentions: {doc.get('governance_mentions', 0)}",
        "",
        "6. External Intelligence Summary",
        f"- Promoter Negative News Signal: {insights.get('promoter_news_level', 'NA')}",
        f"- Litigation Signal: {insights.get('litigation_signal', 'NA')}",
        f"- Regulatory Risk: {insights.get('regulatory_risk', 'NA')}",
        "",
        "7. Qualitative Due Diligence Notes",
        f"- Analyst/Officer Note: {notes[:350] + ('...' if len(notes) > 350 else '')}",
        "",
        "8. Five Cs Analysis",
        f"- Character: {score.get('five_cs_scores', {}).get('Character', 0):.1f}/20",
        f"- Capacity: {score.get('five_cs_scores', {}).get('Capacity', 0):.1f}/20",
        f"- Capital: {score.get('five_cs_scores', {}).get('Capital', 0):.1f}/20",
        f"- Collateral: {score.get('five_cs_scores', {}).get('Collateral', 0):.1f}/20",
        f"- Conditions: {score.get('five_cs_scores', {}).get('Conditions', 0):.1f}/20",
        "",
        "9. Risk Score Summary",
        f"- Total Risk Score: {score.get('total_score', 0):.1f}/100",
        f"- Risk Category: {score.get('risk_category', 'NA')}",
        "",
        "10. Final Recommendation",
        f"- Credit Decision: {reco.get('decision', 'NA')}",
        "",
        "11. Suggested Loan Limit",
        f"- Indicative Limit: {_fmt_inr(reco.get('recommended_limit', 0))}",
        "",
        "12. Suggested Interest Rate / Pricing Band",
        f"- Pricing Guidance: {reco.get('interest_band', 'NA')}",
        "",
        "13. Key Reasons",
        f"- Decision Rationale: {reco.get('decision_rationale', 'NA')}",
        f"- Limit Rationale: {reco.get('limit_rationale', 'NA')}",
        f"- Pricing Rationale: {reco.get('pricing_rationale', 'NA')}",
        "",
        "14. Analyst Notes / Caveats",
        f"- Data Completeness: {completeness_note}",
        f"- Source Mode: {data_source_note}",
        "- Output is a prototype recommendation and not a final sanction note.",
    ]

    lines.extend(["", "15. Source Coverage Summary"])
    if source_coverage:
        for row in source_coverage:
            lines.append(
                f"- {row.get('Source')}: uploaded={row.get('Uploaded')}, parsed={row.get('Parsed')}, used={row.get('Used in Analysis')}"
            )
    else:
        lines.append("- Source coverage details not available")

    lines.extend(["", "16. Additional Document Findings"])
    if source_findings:
        for name, summary in source_findings.items():
            lines.append(f"- {name}: {summary}")
    else:
        lines.append("- Additional document findings not available")

    lines.extend(
        [
            "",
            "17. External Intelligence Notes",
            f"- {external_notes[:300] + ('...' if len(external_notes) > 300 else '') if external_notes else 'Not provided'}",
            "",
            "18. Primary Due Diligence Notes",
            f"- {primary_notes[:300] + ('...' if len(primary_notes) > 300 else '') if primary_notes else 'Not provided'}",
            "",
            "19. Qualitative Score Adjustments",
        ]
    )
    if qualitative_adjustments:
        for item in qualitative_adjustments:
            lines.append(f"- {item.get('signal')} -> {item.get('factor')} ({item.get('points')}): {item.get('reason')}")
    else:
        lines.append("- No explicit qualitative adjustments detected")

    lines.extend(["", "20. Unstructured Document Intelligence Findings", "- Retrieval themes scanned across indexed document chunks"])
    rag_hits = 0
    for theme, result in rag_theme_results.items():
        if result.get("snippets"):
            rag_hits += 1
            top_src = result["snippets"][0].get("source", "NA")
            lines.append(f"- {theme}: evidence found; top source={top_src}; implication=adds risk context to scoring")
    if rag_hits == 0:
        lines.append("- No RAG evidence themes were triggered")

    lines.extend(["", "21. Retrieved Evidence Summary"])
    for theme, result in rag_theme_results.items():
        if result.get("snippets"):
            for snip in result["snippets"][:2]:
                lines.append(f"- [{theme}] ({snip.get('source')}:{snip.get('chunk_id')}): {snip.get('snippet')[:120]}...")

    lines.extend(["", "22. Research Agent Findings"])
    ra_summary = research_agent.get("summary", "")
    lines.append(f"- {ra_summary if ra_summary else 'Research summary not generated'}")
    lines.append(
        f"- Live Research Mode: {live_research.get('mode', 'fallback')} ({live_research.get('status', 'not_run')})"
    )

    lines.extend(["", "23. Indian Context Risk Indicators"])
    if indian_context_cards:
        for card in indian_context_cards:
            lines.append(
                f"- {card.get('Indicator')}: {card.get('Status')} | {card.get('Type')} | Five Cs: {card.get('Five C Relevance')}"
            )
    else:
        lines.append("- Indian context indicators not available")

    lines.extend(
        [
            "",
            "24. Scorecard vs ML Recommendation Summary",
            f"- Scorecard Decision: {hybrid_view.get('scorecard_decision', reco.get('decision', 'NA'))}",
            f"- ML Decision: {hybrid_view.get('ml_decision', ml_view.get('ml_decision', 'NA'))}",
            f"- ML Confidence: {ml_view.get('ml_confidence', hybrid_view.get('decision_confidence', 'NA'))}",
            "",
            "25. Final Hybrid Recommendation",
            f"- Final Decision: {hybrid_view.get('final_decision', reco.get('decision', 'NA'))}",
            f"- Alignment / Override Note: {hybrid_view.get('alignment_note', 'NA')}",
            "",
            "26. Suggested Loan Limit",
            f"- Hybrid Suggested Limit: {_fmt_inr(hybrid_view.get('hybrid_limit', reco.get('recommended_limit', 0)))}",
            "",
            "27. Suggested Interest Rate / Pricing Band",
            f"- Hybrid Pricing Band: {hybrid_view.get('hybrid_pricing_band', reco.get('interest_band', 'NA'))}",
            "",
            "28. Key Reasons",
            f"- Hybrid Rationale: {hybrid_view.get('rationale', reco.get('explanation', 'NA'))}",
            "",
            "29. Caveats / Analyst Notes",
            f"- Qualitative notes changed outcome: {hybrid_view.get('qualitative_changed', False)}",
            f"- RAG/research evidence changed outcome: {hybrid_view.get('rag_changed', False)}",
        ]
    )

    lines.extend(
        [
            "",
            "Appendix A: Quick Qualitative Risk Indicators",
            f"- Litigation Mentions: {doc.get('litigation_mentions', 0)}",
            f"- Governance Risk Mentions: {doc.get('governance_mentions', 0)}",
            f"- Reconciliation Status: {recon.get('overall_status', 'NA')}",
        ]
    )

    reasons = score.get("top_reasons", [])
    if reasons:
        lines.append("- Top Drivers:")
        for item in reasons:
            lines.append(f"  * {item}")

    lines.append("")
    lines.append("Disclaimer: Prototype output for hackathon demonstration only.")

    return "\n".join(lines)
