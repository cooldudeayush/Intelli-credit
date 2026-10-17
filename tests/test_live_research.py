"""Tests for secondary research (Phase 2).

Network access is never required: fetches are forced to fail with a sub-millisecond
timeout so the offline fixture path is what gets exercised.
"""

from __future__ import annotations

import pandas as pd
import pytest

from modules import AnalysisInputs, run_analysis
from modules.live_research import (
    MAX_NEWS_ADJUSTMENTS,
    QUERY_TEMPLATES,
    RISK_BUCKETS,
    _mentions,
    fetch_live_research,
    score_headlines,
)

NO_NETWORK = 0.001  # forces every fetch to fail, exercising the fixture path


def _headline(title: str) -> dict:
    return {"title": title, "link": "", "source": "Test Feed", "published": "", "label": "test"}


def _research(*titles: str, mode: str = "live") -> dict:
    return {"mode": mode, "status": "", "headlines": [_headline(t) for t in titles], "queries": []}


# --- fetching --------------------------------------------------------------


def test_disabled_mode_returns_nothing():
    result = fetch_live_research("Acme Ltd", "Retail", enabled=False)
    assert result["mode"] == "disabled"
    assert result["headlines"] == []
    assert score_headlines(result) == []


def test_offline_fallback_when_network_unavailable():
    result = fetch_live_research("Acme Ltd", "Infrastructure", timeout=NO_NETWORK)
    assert result["mode"] == "offline_sample"
    assert result["headlines"], "Offline fixture should still demonstrate the feature"


def test_offline_headlines_are_labelled_as_samples():
    """A synthetic headline must never be mistakable for real reporting."""
    result = fetch_live_research("Acme Ltd", "Infrastructure", timeout=NO_NETWORK)
    for headline in result["headlines"]:
        assert headline["title"].startswith("[OFFLINE SAMPLE]")


def test_unknown_sector_still_gets_fixture_headlines():
    result = fetch_live_research("Acme Ltd", "Aquaculture", timeout=NO_NETWORK)
    assert result["headlines"], "Unknown sectors should fall back to the default fixture"


def test_queries_cover_the_india_specific_registers():
    """Distress surfaces in NCLT and RBI/SEBI actions, not just general news."""
    labels = {label for label, _ in QUERY_TEMPLATES}
    assert {"promoter", "litigation", "insolvency", "regulatory", "sector"} <= labels


# --- keyword matching ------------------------------------------------------


@pytest.mark.parametrize(
    "text,keyword,expected",
    [
        ("arbitration tribunal hears the claim", "rbi", False),  # inside "arbitration"
        ("rbi tightens norms for lenders", "rbi", True),
        ("urban housing demand recovers", "ban", False),  # inside "urban"
        ("regulator issues ban on the unit", "ban", True),
        ("company admitted to nclt", "nclt", True),
    ],
)
def test_keywords_match_whole_words_only(text, keyword, expected):
    assert _mentions(text, keyword) is expected


# --- scoring ---------------------------------------------------------------


def test_distress_headline_produces_a_capacity_penalty():
    adjustments = score_headlines(_research("Acme Ltd admitted to NCLT over payment default"))
    assert adjustments
    distress = [a for a in adjustments if a["signal"] == "Financial distress"]
    assert distress and distress[0]["factor"] == "Capacity"
    assert distress[0]["points"] == -2


def test_promoter_headline_hits_character():
    adjustments = score_headlines(_research("Acme Ltd promoter pledge rises to 80 percent"))
    promoter = [a for a in adjustments if a["signal"] == "Promoter / governance concern"]
    assert promoter and promoter[0]["factor"] == "Character"


def test_benign_headlines_produce_no_adjustments():
    adjustments = score_headlines(_research("Acme Ltd opens new plant and hires 200 staff"))
    assert adjustments == []


def test_adjustments_are_capped():
    """Repeated bad news must not stack without limit."""
    noisy = _research(
        *[" ".join(config["keywords"][:3]) for config in RISK_BUCKETS.values()],
    )
    assert len(score_headlines(noisy)) <= MAX_NEWS_ADJUSTMENTS


def test_one_adjustment_per_risk_category():
    """Ten articles about the same tribunal hearing should count once."""
    repeated = _research(*[f"Acme Ltd faces tribunal hearing number {i}" for i in range(10)])
    signals = [a["signal"] for a in score_headlines(repeated)]
    assert len(signals) == len(set(signals))


def test_every_adjustment_quotes_its_headline():
    """A credit officer must be able to check the claim that moved the score."""
    for adjustment in score_headlines(_research("Acme Ltd admitted to NCLT over default")):
        assert adjustment["evidence_snippet"]
        assert "News mentions" in adjustment["reason"]
        assert adjustment["source_type"] == "secondary_research"


def test_sample_provenance_survives_into_the_reason():
    """Score impact from fixture data must stay identifiable in the CAM."""
    research = fetch_live_research("Acme Ltd", "Infrastructure", timeout=NO_NETWORK)
    adjustments = score_headlines(research)
    assert adjustments, "Infrastructure fixture should trip at least one bucket"
    for adjustment in adjustments:
        assert adjustment["is_sample"] is True
        assert "[OFFLINE SAMPLE]" in adjustment["reason"]


# --- end to end ------------------------------------------------------------


def _inputs(live_research: dict) -> AnalysisInputs:
    return AnalysisInputs(
        company_name="Testco Pvt Ltd",
        sector="Infrastructure",
        bank_df=pd.DataFrame(
            {"inflow": [20_000_000] * 12, "outflow": [8_000_000] * 12, "balance": [5e7] * 12}
        ),
        gst_df=pd.DataFrame({"taxable_turnover": [20_000_000] * 12}),
        annual_result={
            "text": (
                "Revenue for the year was INR 300 crore with total debt of INR 60 crore. "
                "Total liabilities stood at INR 120 crore and cash flow from operations "
                "was INR 45 crore."
            ),
            "uploaded": True,
            "parsed": True,
            "method": "native",
            "message": "",
            "chars": 180,
        },
        live_research=live_research,
    )


def test_adverse_news_lowers_the_score(model_bundle):
    """The point of the whole phase: news that is not in the files must count."""
    quiet = run_analysis(_inputs(fetch_live_research("X", "Y", enabled=False)), model_bundle)
    adverse = run_analysis(
        _inputs(_research("Testco Pvt Ltd admitted to NCLT after loan default")),
        model_bundle,
    )
    assert adverse["score_bundle"]["total_score"] < quiet["score_bundle"]["total_score"]


def test_news_reaches_the_cam_with_attribution(model_bundle):
    result = run_analysis(
        _inputs(_research("Testco Pvt Ltd promoter pledge rises sharply")), model_bundle
    )
    assert result["news_adjustments"]
    assert "promoter pledge" in result["cam_text"].lower()


def test_news_feeds_the_model(model_bundle):
    result = run_analysis(
        _inputs(_research("Testco Pvt Ltd admitted to NCLT after default")), model_bundle
    )
    assert result["ml_features"]["live_news_signal_count"] >= 1


def test_news_adjustments_join_the_single_ledger(model_bundle):
    result = run_analysis(
        _inputs(_research("Testco Pvt Ltd admitted to NCLT after default")), model_bundle
    )
    ledger_signals = {a.get("signal") for a in result["all_adjustments"]}
    assert "Financial distress" in ledger_signals


# ---------------------------------------------------------------------------
# Startup robustness: the sector selector renders on first paint, before any
# upload, so anything it depends on must never fail open.
# ---------------------------------------------------------------------------


def test_sector_list_is_never_empty(monkeypatch):
    """An empty options list with index=0 raises and takes the whole app down.

    This is what a deployed instance looks like when the profile file cannot be
    read: a bare "Error running app" with no further detail.
    """
    from pathlib import Path

    from modules import research_insights

    research_insights._load_profiles.cache_clear()
    monkeypatch.setattr(
        research_insights, "SECTOR_PROFILE_PATH", Path("does/not/exist.json")
    )
    try:
        sectors = research_insights.available_sectors()
        assert sectors, "Sector list fell back to empty - the selectbox would raise"
        assert "Manufacturing" in sectors
    finally:
        research_insights._load_profiles.cache_clear()


def test_sector_profile_lookup_survives_a_missing_file(monkeypatch):
    from pathlib import Path

    from modules import research_insights

    research_insights._load_profiles.cache_clear()
    monkeypatch.setattr(
        research_insights, "SECTOR_PROFILE_PATH", Path("does/not/exist.json")
    )
    try:
        profile = research_insights.get_sector_profile("Manufacturing")
        assert profile.get("risk_points") is not None
    finally:
        research_insights._load_profiles.cache_clear()
