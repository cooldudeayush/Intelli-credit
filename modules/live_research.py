from __future__ import annotations

from typing import Dict, List


def fetch_live_research(company_name: str, sector: str) -> Dict:
    """Optional lightweight live news retrieval via public RSS endpoints."""
    try:
        import feedparser
    except Exception:
        return {
            "mode": "fallback",
            "status": "feedparser_not_installed",
            "headlines": [],
            "summary": "Live mode unavailable; using manual/mock research.",
        }

    queries = [
        f"{company_name} promoter news",
        f"{sector} India sector headwinds",
        f"{company_name} litigation regulatory",
    ]

    headlines: List[Dict] = []
    for q in queries:
        url = f"https://news.google.com/rss/search?q={q.replace(' ', '+')}"
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:3]:
                headlines.append(
                    {
                        "query": q,
                        "title": getattr(entry, "title", ""),
                        "link": getattr(entry, "link", ""),
                    }
                )
        except Exception:
            continue

    if not headlines:
        return {
            "mode": "fallback",
            "status": "no_live_results",
            "headlines": [],
            "summary": "No live headlines fetched; using manual/mock research.",
        }

    return {
        "mode": "live",
        "status": "ok",
        "headlines": headlines[:8],
        "summary": f"Fetched {len(headlines[:8])} live headlines (prototype RSS mode).",
    }
