"""End-to-end credit analysis pipeline.

This is the whole appraisal, independent of Streamlit: given parsed inputs, it
runs extraction, reconciliation, retrieval, scoring, ML and fusion, and returns
every intermediate result for the UI to render and the CAM to quote.

Keeping it here rather than inline in ``app.py`` means the analysis can be run
and tested headlessly, and — critically — that it runs once per click instead of
once per Streamlit rerun.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

from .cam_generator import build_cam_text
from .document_intel import extract_document_signals
from .hybrid_recommendation import combine_scorecard_ml
from .indian_context import build_indian_context_cards
from .ingestion import summarize_source_text, summarize_structured_source
from .live_research import score_headlines
from .ml_hybrid import HybridModelBundle, engineer_features, predict_ml_view, train_hybrid_model
from .qualitative import extract_qualitative_adjustments
from .rag_engine import build_local_rag_index, derive_rag_adjustments, retrieve_evidence_by_theme
from .recommendation import generate_credit_recommendation
from .reconciliation import reconcile_gst_bank
from .research_agent import build_research_agent_findings
from .research_insights import generate_research_insights
from .scoring import compute_five_cs_score

OPTIONAL_UNSTRUCTURED = ("Legal Notice", "Sanction Letter", "Rating Report", "Board Minutes")

LIVE_RESEARCH_DISABLED = {
    "mode": "disabled",
    "status": "Live research is switched off.",
    "headlines": [],
    "queries": [],
    "summary": "Secondary research disabled; scoring uses uploaded documents only.",
}


def empty_source_result(message: str = "Not provided") -> Dict:
    """The shape ``extract_pdf_text_with_fallback`` returns for an absent file."""
    return {
        "text": "",
        "uploaded": False,
        "parsed": False,
        "method": "not_provided",
        "message": message,
        "chars": 0,
    }


@dataclass
class AnalysisInputs:
    """Everything the pipeline needs, already parsed out of files or widgets."""

    company_name: str
    sector: str
    bank_df: pd.DataFrame
    gst_df: pd.DataFrame
    primary_notes: str = ""
    external_notes: str = ""
    annual_result: Dict = field(default_factory=empty_source_result)
    additional_unstructured: Dict[str, Dict] = field(
        default_factory=lambda: {src: empty_source_result() for src in OPTIONAL_UNSTRUCTURED}
    )
    itr_result: Dict = field(
        default_factory=lambda: {
            "kind": "missing",
            "dataframe": pd.DataFrame(),
            "text": "",
            "parsed": False,
            "method": "not_provided",
        }
    )
    share_result: Dict = field(
        default_factory=lambda: {
            "kind": "missing",
            "dataframe": pd.DataFrame(),
            "text": "",
            "parsed": False,
            "method": "not_provided",
        }
    )
    research_agent_inputs: Dict = field(default_factory=dict)
    live_research: Dict = field(default_factory=lambda: dict(LIVE_RESEARCH_DISABLED))
    completeness_note: str = ""
    data_source_note: str = ""
    bank_uploaded: bool = False
    gst_uploaded: bool = False


def _yes_no(flag) -> str:
    return "Yes" if flag else "No"


def _source_findings_line(summary: Dict) -> str:
    pairs = [f"{k}={v}" for k, v in summary.items() if k != "source"]
    return "; ".join(pairs[:6]) if pairs else "No finding"


def _build_source_rows(inputs: AnalysisInputs) -> List[Dict]:
    """Per-source coverage table. ``Type`` and ``Used in Analysis`` feed the model."""
    rows = [
        {
            "Source": "Annual Report",
            "Type": "Unstructured",
            "Uploaded": _yes_no(inputs.annual_result.get("uploaded")),
            "Parsed": _yes_no(inputs.annual_result.get("parsed")),
            "Used in Analysis": _yes_no(inputs.annual_result.get("text")),
            "Parse Method": inputs.annual_result.get("method", "NA"),
        }
    ]
    for src, result in inputs.additional_unstructured.items():
        rows.append(
            {
                "Source": src,
                "Type": "Unstructured",
                "Uploaded": _yes_no(result.get("uploaded")),
                "Parsed": _yes_no(result.get("parsed")),
                "Used in Analysis": _yes_no(result.get("text")),
                "Parse Method": result.get("method", "NA"),
            }
        )

    rows.extend(
        [
            {
                "Source": "Bank Statement",
                "Type": "Structured",
                "Uploaded": _yes_no(inputs.bank_uploaded),
                "Parsed": "Yes",
                "Used in Analysis": "Yes",
                "Parse Method": "csv",
            },
            {
                "Source": "GST Summary",
                "Type": "Structured",
                "Uploaded": _yes_no(inputs.gst_uploaded),
                "Parsed": "Yes",
                "Used in Analysis": "Yes",
                "Parse Method": "csv/excel",
            },
            {
                "Source": "ITR",
                "Type": "Structured/Semi-Structured",
                "Uploaded": _yes_no(inputs.itr_result.get("kind") not in {"missing", None}),
                "Parsed": _yes_no(inputs.itr_result.get("parsed")),
                "Used in Analysis": _yes_no(inputs.itr_result.get("parsed")),
                "Parse Method": inputs.itr_result.get("method", "NA"),
            },
            {
                "Source": "Shareholding Pattern",
                "Type": "Structured/Semi-Structured",
                "Uploaded": _yes_no(inputs.share_result.get("kind") not in {"missing", None}),
                "Parsed": _yes_no(inputs.share_result.get("parsed")),
                "Used in Analysis": _yes_no(inputs.share_result.get("parsed")),
                "Parse Method": inputs.share_result.get("method", "NA"),
            },
            {
                "Source": "Primary Due Diligence Notes",
                "Type": "Primary Insights",
                "Uploaded": _yes_no(inputs.primary_notes),
                "Parsed": _yes_no(inputs.primary_notes),
                "Used in Analysis": _yes_no(inputs.primary_notes),
                "Parse Method": "text",
            },
            {
                "Source": "External Intelligence Notes",
                "Type": "External Intelligence",
                "Uploaded": _yes_no(inputs.external_notes),
                "Parsed": _yes_no(inputs.external_notes),
                "Used in Analysis": _yes_no(inputs.external_notes),
                "Parse Method": "text/txt",
            },
        ]
    )
    return rows


def _summarize_structured(label: str, result: Dict) -> Dict:
    kind = result.get("kind")
    if kind == "dataframe":
        return summarize_structured_source(label, result.get("dataframe"))
    if kind == "pdf_text":
        return summarize_source_text(label, result.get("text", ""))
    return {"source": label, "insight": f"No {label} source provided"}


def _collect_unstructured_text(inputs: AnalysisInputs) -> str:
    parts = []
    if inputs.annual_result.get("text"):
        parts.append(f"[Annual Report] {inputs.annual_result['text']}")
    for src, result in inputs.additional_unstructured.items():
        if result.get("text"):
            parts.append(f"[{src}] {result['text']}")
    if inputs.itr_result.get("kind") == "pdf_text" and inputs.itr_result.get("text"):
        parts.append(f"[ITR PDF] {inputs.itr_result['text']}")
    if inputs.share_result.get("kind") == "pdf_text" and inputs.share_result.get("text"):
        parts.append(f"[Shareholding PDF] {inputs.share_result['text']}")
    return "\n".join(parts)


def _build_rag_documents(inputs: AnalysisInputs) -> List[Dict]:
    documents = [
        {
            "source": "Annual Report",
            "source_type": "PDF",
            "source_category": "Unstructured",
            "text": inputs.annual_result.get("text", ""),
        }
    ]
    documents += [
        {
            "source": src,
            "source_type": "PDF",
            "source_category": "Unstructured",
            "text": result.get("text", ""),
        }
        for src, result in inputs.additional_unstructured.items()
    ]
    if inputs.itr_result.get("kind") == "pdf_text":
        documents.append(
            {
                "source": "ITR PDF",
                "source_type": "PDF",
                "source_category": "Structured/Semi",
                "text": inputs.itr_result.get("text", ""),
            }
        )
    if inputs.share_result.get("kind") == "pdf_text":
        documents.append(
            {
                "source": "Shareholding PDF",
                "source_type": "PDF",
                "source_category": "Structured/Semi",
                "text": inputs.share_result.get("text", ""),
            }
        )
    return documents


def run_analysis(
    inputs: AnalysisInputs,
    model_bundle: Optional[HybridModelBundle] = None,
) -> Dict:
    """Run the full appraisal and return every intermediate result.

    ``model_bundle`` is accepted so the caller can reuse a cached model; when
    omitted a fresh one is trained (deterministic, seeded at 42).
    """
    combined_unstructured_text = _collect_unstructured_text(inputs)
    combined_notes = f"{inputs.primary_notes}\n{inputs.external_notes}".strip()

    # --- Extraction and cross-verification ---------------------------------
    doc_signals = extract_document_signals(combined_unstructured_text)
    recon = reconcile_gst_bank(inputs.gst_df, inputs.bank_df)

    # --- Secondary research ------------------------------------------------
    # Headlines are fetched by the caller (so the network call sits behind the
    # run gate) and scored here, alongside every other signal.
    live_research = inputs.live_research or dict(LIVE_RESEARCH_DISABLED)
    news_adjustments = score_headlines(live_research)

    insights = generate_research_insights(
        sector=inputs.sector,
        document_signals=doc_signals,
        manual_notes=combined_notes,
        news_adjustments=news_adjustments,
    )

    # --- Retrieval ---------------------------------------------------------
    rag_index = build_local_rag_index(_build_rag_documents(inputs))
    rag_theme_results = retrieve_evidence_by_theme(rag_index, top_k=3)
    rag_adjustments = derive_rag_adjustments(rag_theme_results)

    # --- Qualitative and research signals ----------------------------------
    qual_adjustments = extract_qualitative_adjustments(
        primary_notes=inputs.primary_notes,
        external_notes=inputs.external_notes,
    )
    research_agent_findings = build_research_agent_findings(
        manual_inputs=inputs.research_agent_inputs or {},
        base_insights=insights,
        rag_theme_results=rag_theme_results,
        primary_notes=inputs.primary_notes,
        external_notes=inputs.external_notes,
    )

    if live_research.get("headlines"):
        top_titles = "; ".join(h.get("title", "")[:90] for h in live_research["headlines"][:3])
        research_agent_findings["cards"]["Live Research Headlines"] = top_titles
        research_agent_findings["summary"] = (
            f"{research_agent_findings.get('summary', '')}\n"
            f"Secondary research ({live_research.get('mode')}): "
            f"{live_research.get('summary', '')}"
        ).strip()

    research_agent_adjustments = [
        {
            "source": row.get("source", "Research Agent"),
            "source_type": row.get("origin", "external_intelligence"),
            "signal": row.get("cue", "Research cue"),
            "factor": row.get("five_c", "Conditions"),
            "points": row.get("score_effect", 0),
            "reason": row.get("reason", "Research panel-derived signal"),
        }
        for row in research_agent_findings.get("signal_rows", [])
    ]
    all_adjustments = (
        qual_adjustments + rag_adjustments + research_agent_adjustments + news_adjustments
    )

    # --- Scoring and baseline recommendation -------------------------------
    score_bundle = compute_five_cs_score(
        doc_signals=doc_signals,
        recon=recon,
        insights=insights,
        manual_notes=combined_notes,
        qualitative_adjustments=all_adjustments,
    )
    recommendation = generate_credit_recommendation(
        score_bundle=score_bundle,
        doc_signals=doc_signals,
        recon=recon,
    )

    # --- Source coverage bookkeeping ---------------------------------------
    source_rows = _build_source_rows(inputs)
    itr_summary = _summarize_structured("ITR", inputs.itr_result)
    share_summary = _summarize_structured("Shareholding Pattern", inputs.share_result)

    source_findings = {
        "Annual Report Findings": _source_findings_line(
            summarize_source_text("Annual Report", inputs.annual_result.get("text", ""))
        )
    }
    for src, result in inputs.additional_unstructured.items():
        source_findings[f"{src} Findings"] = _source_findings_line(
            summarize_source_text(src, result.get("text", ""))
        )
    source_findings["ITR Findings"] = _source_findings_line(itr_summary)
    source_findings["Shareholding Pattern Findings"] = _source_findings_line(share_summary)
    source_findings["Primary Due Diligence Findings"] = _source_findings_line(
        summarize_source_text("Primary Notes", inputs.primary_notes)
    )
    source_findings["External Intelligence Findings"] = _source_findings_line(
        summarize_source_text("External Notes", inputs.external_notes)
    )

    unstructured_processing = [
        {
            "Source": "Annual Report",
            "Uploaded": inputs.annual_result.get("uploaded"),
            "Parsed": inputs.annual_result.get("parsed"),
            "Method": inputs.annual_result.get("method"),
            "Chars": inputs.annual_result.get("chars", 0),
            "Message": inputs.annual_result.get("message", ""),
        }
    ]
    unstructured_processing += [
        {
            "Source": src,
            "Uploaded": result.get("uploaded"),
            "Parsed": result.get("parsed"),
            "Method": result.get("method"),
            "Chars": result.get("chars", 0),
            "Message": result.get("message", ""),
        }
        for src, result in inputs.additional_unstructured.items()
    ]

    # --- Indian context, ML, fusion ----------------------------------------
    indian_context_cards = build_indian_context_cards(
        recon=recon,
        insights=insights,
        doc_signals=doc_signals,
        shareholding_summary=share_summary if isinstance(share_summary, dict) else {},
        combined_notes=combined_notes,
    )

    bundle = model_bundle or train_hybrid_model()
    ml_features = engineer_features(
        score_bundle=score_bundle,
        recon=recon,
        doc_signals=doc_signals,
        insights=insights,
        qualitative_adjustments=qual_adjustments,
        rag_adjustments=rag_adjustments,
        source_rows=source_rows,
        indian_context_cards=indian_context_cards,
        news_adjustments=news_adjustments,
    )
    ml_view = predict_ml_view(bundle, ml_features)
    hybrid_view = combine_scorecard_ml(
        scorecard_reco=recommendation,
        ml_view=ml_view,
        doc_signals=doc_signals,
        qualitative_adjustments=qual_adjustments,
        rag_adjustments=rag_adjustments,
    )

    cam_payload = {
        "company_name": inputs.company_name,
        "sector": inputs.sector,
        "doc_signals": doc_signals,
        "recon": recon,
        "insights": insights,
        "score": score_bundle,
        "recommendation": recommendation,
        "manual_notes": inputs.primary_notes,
        "primary_notes": inputs.primary_notes,
        "external_notes": inputs.external_notes,
        "completeness_note": inputs.completeness_note,
        "data_source_note": inputs.data_source_note,
        "source_coverage": source_rows,
        "source_findings": source_findings,
        "qualitative_adjustments": all_adjustments,
        "rag_theme_results": rag_theme_results,
        "research_agent": research_agent_findings,
        "indian_context_cards": indian_context_cards,
        "ml_view": ml_view,
        "hybrid_view": hybrid_view,
        "live_research": live_research,
    }

    return {
        "company_name": inputs.company_name,
        "sector": inputs.sector,
        "bank_df": inputs.bank_df,
        "gst_df": inputs.gst_df,
        "primary_notes": inputs.primary_notes,
        "external_notes": inputs.external_notes,
        "completeness_note": inputs.completeness_note,
        "data_source_note": inputs.data_source_note,
        "doc_signals": doc_signals,
        "recon": recon,
        "insights": insights,
        "rag_index": rag_index,
        "rag_theme_results": rag_theme_results,
        "rag_adjustments": rag_adjustments,
        "qual_adjustments": qual_adjustments,
        "news_adjustments": news_adjustments,
        "research_agent_findings": research_agent_findings,
        "all_adjustments": all_adjustments,
        "score_bundle": score_bundle,
        "recommendation": recommendation,
        "source_rows": source_rows,
        "source_findings": source_findings,
        "unstructured_processing": unstructured_processing,
        "indian_context_cards": indian_context_cards,
        "ml_features": ml_features,
        "ml_view": ml_view,
        "hybrid_view": hybrid_view,
        "live_research": live_research,
        "cam_payload": cam_payload,
        "cam_text": build_cam_text(cam_payload),
    }
