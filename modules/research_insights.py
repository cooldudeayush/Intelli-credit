from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, List

SECTOR_PROFILE_PATH = Path(__file__).resolve().parents[1] / "data" / "sector_profiles.json"

_HARDCODED_FALLBACK = {
    "headwind": "Mixed sector outlook; no dedicated profile on file.",
    "risk_points": 10,
    "regulatory_risk": "Low",
    "watch_items": [],
    "last_reviewed": "unknown",
    "source": "Built-in fallback - sector profile file unavailable",
}


@lru_cache(maxsize=1)
def _load_profiles() -> Dict:
    """Load analyst-maintained sector profiles from disk.

    Kept as data rather than code so the credit view can be revised without a
    release, and so each sector carries a visible ``last_reviewed`` date instead
    of an undated constant buried in a module.
    """
    try:
        payload = json.loads(SECTOR_PROFILE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"sectors": {}, "_fallback": dict(_HARDCODED_FALLBACK)}
    return payload


def get_sector_profile(sector: str) -> Dict:
    payload = _load_profiles()
    sectors = payload.get("sectors", {})
    fallback = payload.get("_fallback", _HARDCODED_FALLBACK)
    return sectors.get(sector, fallback)


def available_sectors() -> List[str]:
    """Sectors with a dedicated profile, for the UI selector."""
    return sorted(_load_profiles().get("sectors", {}).keys())


def generate_research_insights(
    sector: str,
    document_signals: Dict,
    manual_notes: str,
    news_adjustments: List[Dict] | None = None,
) -> Dict:
    """Combine the sector baseline with document and news-derived signals."""
    profile = get_sector_profile(sector)
    lower_notes = (manual_notes or "").lower()

    # Promoter risk: analyst notes first, then anything secondary research found.
    negative_news_flag = "Low"
    promoter_news_summary = "No promoter adverse-media signal identified."
    if "pledge" in lower_notes or "resignation" in lower_notes:
        negative_news_flag = "Moderate"
        promoter_news_summary = (
            "Signals of governance stress found in analyst notes (pledge/resignation cues)."
        )

    news_signals = news_adjustments or []
    promoter_news = [
        item for item in news_signals if item.get("signal") == "Promoter / governance concern"
    ]
    if promoter_news:
        negative_news_flag = "Moderate"
        promoter_news_summary = promoter_news[0].get("reason", promoter_news_summary)

    litigation_level = "Low"
    litigation_mentions = document_signals.get("litigation_mentions", 0)
    if litigation_mentions >= 3:
        litigation_level = "Moderate"
    if litigation_mentions >= 6:
        litigation_level = "High"
    if any(item.get("signal") == "Litigation exposure" for item in news_signals):
        litigation_level = "High" if litigation_level == "Moderate" else "Moderate"

    regulatory_risk = profile.get("regulatory_risk", "Low")
    if any(item.get("signal") == "Regulatory / sector pressure" for item in news_signals):
        regulatory_risk = "Moderate" if regulatory_risk == "Low" else "High"

    distress_signals = [
        item for item in news_signals if item.get("signal") == "Financial distress"
    ]

    summary_lines = [
        f"Sector headwind: {profile.get('headwind')}",
        f"Promoter negative news risk: {negative_news_flag}",
        f"Litigation signal level: {litigation_level}",
        f"Regulatory risk: {regulatory_risk}",
    ]
    if distress_signals:
        summary_lines.append(f"Distress signal: {distress_signals[0].get('reason')}")

    return {
        "sector_headwind": profile.get("headwind"),
        "sector_watch_items": profile.get("watch_items", []),
        "sector_profile_reviewed": profile.get("last_reviewed", "unknown"),
        "sector_profile_source": profile.get("source", "unknown"),
        "promoter_news": promoter_news_summary,
        "promoter_news_level": negative_news_flag,
        "litigation_signal": litigation_level,
        "regulatory_risk": regulatory_risk,
        "distress_signal": distress_signals[0].get("reason") if distress_signals else "",
        "sector_risk_points": profile.get("risk_points", 10),
        "ai_summary": "\n".join(summary_lines),
    }
