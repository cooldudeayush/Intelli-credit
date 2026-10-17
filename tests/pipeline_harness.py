"""Run a demo scenario through the real pipeline.

Phase 1.3 extracted the analysis out of ``app.py`` into ``modules/pipeline.py``,
so this no longer re-implements the orchestration — it only builds the
``AnalysisInputs`` that demo mode would build and calls ``run_analysis``. Tests
and the app now execute the same code path.
"""

from __future__ import annotations

from typing import Dict, Optional

from modules import (
    AnalysisInputs,
    OPTIONAL_UNSTRUCTURED,
    empty_source_result,
    get_demo_scenarios,
    run_analysis,
    train_hybrid_model,
)

_model_bundle = None


def _get_model():
    """Train once per session, mirroring app.py's ``@st.cache_resource``."""
    global _model_bundle
    if _model_bundle is None:
        _model_bundle = train_hybrid_model()
    return _model_bundle


def build_demo_inputs(scenario_name: str) -> AnalysisInputs:
    """Assemble the inputs demo mode feeds the pipeline for one scenario."""
    scenarios = get_demo_scenarios()
    if scenario_name not in scenarios:
        raise KeyError(f"Unknown scenario {scenario_name!r}. Have: {list(scenarios)}")

    scenario = scenarios[scenario_name]
    report_text = scenario["report_text"]

    return AnalysisInputs(
        company_name=scenario["company_name"],
        sector=scenario["sector"],
        bank_df=scenario["bank_df"].copy(),
        gst_df=scenario["gst_df"].copy(),
        primary_notes=scenario["manual_notes"],
        external_notes=scenario.get("external_notes", ""),
        annual_result={
            "text": report_text,
            "uploaded": False,
            "parsed": True,
            "method": "demo_text",
            "message": "Demo scenario narrative",
            "chars": len(report_text),
        },
        additional_unstructured={src: empty_source_result() for src in OPTIONAL_UNSTRUCTURED},
        # Demo mode leaves the five research-agent text areas untouched.
        research_agent_inputs={
            "promoter_obs": "",
            "sector_obs": "",
            "regulatory_obs": "",
            "mca_notes": "",
            "litigation_notes": "",
        },
        completeness_note="High confidence: full scenario dataset available (demo mode).",
        data_source_note=f"Demo Mode: {scenario_name}",
    )


def run_demo_pipeline(scenario_name: str, model_bundle: Optional[object] = None) -> Dict:
    """Run one demo scenario end to end and return every intermediate result."""
    result = run_analysis(
        build_demo_inputs(scenario_name),
        model_bundle=model_bundle or _get_model(),
    )
    result["scenario_name"] = scenario_name
    return result
