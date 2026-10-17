"""Secondary research: news and filings signals from outside the uploaded files.

The problem statement asks the engine to find things that are *not* in the
documents the borrower supplied. Retrieval alone cannot do that — it only ever
sees what was uploaded. This module queries public news feeds, scans what comes
back for credit-risk language, and emits score adjustments in the same ledger
shape as every other signal, so news reaches the CAM with full attribution.

Headlines are evidence about a borrower, not instructions: they are only ever
keyword-matched and quoted, never executed or followed.
"""

from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

GOOGLE_NEWS_RSS = "https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"
REQUEST_TIMEOUT_SECONDS = 6.0
MAX_HEADLINES_PER_QUERY = 4
MAX_HEADLINES_TOTAL = 12

# At most this many news-derived adjustments may reach the score, matching the
# cap on retrieval signals so one noisy news cycle cannot dominate an appraisal.
MAX_NEWS_ADJUSTMENTS = 3

OFFLINE_FIXTURE = Path(__file__).resolve().parents[1] / "sample_data" / "offline_headlines.json"

# Search templates. Beyond generic promoter/sector queries these target the
# India-specific registers where distress surfaces first: NCLT admissions under
# the IBC, and RBI/SEBI enforcement.
QUERY_TEMPLATES = [
    ("promoter", '"{company}" promoter OR management'),
    ("litigation", '"{company}" litigation OR court OR tribunal'),
    ("insolvency", '"{company}" NCLT OR insolvency OR default'),
    ("regulatory", '"{company}" RBI OR SEBI OR regulatory'),
    ("sector", "{sector} India sector outlook OR headwinds"),
]

# Risk vocabulary, grouped by what the signal actually evidences. Weights follow
# the same logic as the retrieval mapping: statements about conduct or solvency
# hit harder than statements about context.
RISK_BUCKETS = {
    "Financial distress": {
        "keywords": [
            "default", "defaults", "defaulted", "nclt", "ibc", "insolvency", "insolvent",
            "bankruptcy", "liquidation", "downgrade", "downgraded", "npa", "wilful defaulter",
            "restructuring", "moratorium", "one-time settlement",
        ],
        "factor": "Capacity",
        "points": -2,
    },
    "Promoter / governance concern": {
        "keywords": [
            "promoter pledge", "pledged shares", "resigns", "resignation", "steps down",
            "auditor quits", "related party", "siphon", "fraud", "governance lapse",
            "whistleblower", "stake sale",
        ],
        "factor": "Character",
        "points": -2,
    },
    "Litigation exposure": {
        "keywords": [
            "lawsuit", "litigation", "court", "tribunal", "arbitration", "summons",
            "raid", "probe", "investigation", "chargesheet", "plea",
        ],
        "factor": "Character",
        "points": -1,
    },
    "Regulatory / sector pressure": {
        "keywords": [
            "rbi", "sebi", "penalty", "fined", "ban", "banned", "show cause",
            "non-compliance", "licence cancelled", "curbs", "headwind", "slowdown",
        ],
        "factor": "Conditions",
        "points": -1,
    },
}


def _normalise_title(title: str) -> str:
    return re.sub(r"\s+", " ", (title or "").strip().lower())


def _mentions(text: str, keyword: str) -> bool:
    """Whole-word keyword match.

    Plain substring matching is unsafe for short acronyms: "rbi" occurs inside
    "arbitration", so an arbitration headline would raise a regulatory flag.
    Word boundaries also keep "ban" from firing on "urban".
    """
    return re.search(rf"\b{re.escape(keyword)}\b", text) is not None


def _fetch_feed(query: str, timeout: float) -> List[Dict]:
    """Fetch and parse one RSS query. Returns [] on any failure."""
    try:
        import feedparser
    except Exception:
        return []

    url = GOOGLE_NEWS_RSS.format(query=urllib.parse.quote(query))
    try:
        # feedparser.parse has no timeout of its own, so the fetch is done here
        # to guarantee the UI cannot hang on a slow or unreachable feed.
        request = urllib.request.Request(url, headers={"User-Agent": "Intelli-Credit/1.0"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
    except Exception:
        return []

    try:
        feed = feedparser.parse(raw)
    except Exception:
        return []

    entries = []
    for entry in getattr(feed, "entries", [])[:MAX_HEADLINES_PER_QUERY]:
        entries.append(
            {
                "title": getattr(entry, "title", "") or "",
                "link": getattr(entry, "link", "") or "",
                "source": getattr(getattr(entry, "source", None), "title", "") or "Google News",
                "published": getattr(entry, "published", "") or "",
            }
        )
    return entries


def _load_offline_fixture(company_name: str, sector: str) -> List[Dict]:
    """Bundled sample headlines so the feature still demonstrates without a network.

    Every fixture title is prefixed so it stays identifiable as synthetic even in
    a screenshot or an exported CAM.
    """
    try:
        payload = json.loads(OFFLINE_FIXTURE.read_text(encoding="utf-8"))
    except Exception:
        return []

    entries = payload.get(sector) or payload.get("_default") or []
    return [
        {
            "title": f"[OFFLINE SAMPLE] {item['title'].format(company=company_name, sector=sector)}",
            "link": "",
            "source": "Bundled sample fixture",
            "published": "",
            "label": item.get("label", "sector"),
        }
        for item in entries
    ]


def fetch_live_research(
    company_name: str,
    sector: str,
    enabled: bool = True,
    timeout: float = REQUEST_TIMEOUT_SECONDS,
) -> Dict:
    """Retrieve news headlines for a borrower, falling back to bundled samples.

    Modes:
      ``live``           - headlines came from the network
      ``offline_sample`` - synthetic fixtures; usable for a demo, clearly labelled
      ``disabled``       - research was switched off, no headlines
    """
    if not enabled:
        return {
            "mode": "disabled",
            "status": "Live research is switched off.",
            "headlines": [],
            "queries": [],
            "summary": "Secondary research disabled; scoring uses uploaded documents only.",
        }

    queries = [
        {"label": label, "query": template.format(company=company_name, sector=sector)}
        for label, template in QUERY_TEMPLATES
    ]

    headlines: List[Dict] = []
    seen: set[str] = set()
    for item in queries:
        for entry in _fetch_feed(item["query"], timeout):
            key = _normalise_title(entry["title"])
            if not key or key in seen:
                continue
            seen.add(key)
            entry["label"] = item["label"]
            entry["query"] = item["query"]
            headlines.append(entry)
            if len(headlines) >= MAX_HEADLINES_TOTAL:
                break
        if len(headlines) >= MAX_HEADLINES_TOTAL:
            break

    if headlines:
        return {
            "mode": "live",
            "status": f"Fetched {len(headlines)} headlines across {len(queries)} queries.",
            "headlines": headlines,
            "queries": queries,
            "summary": f"Live secondary research returned {len(headlines)} headlines.",
        }

    fixture = _load_offline_fixture(company_name, sector)
    return {
        "mode": "offline_sample",
        "status": "No network results; using bundled sample headlines.",
        "headlines": fixture,
        "queries": queries,
        "summary": (
            "Offline sample headlines are synthetic fixtures for demonstration, "
            "not real reporting about this borrower."
        ),
    }


def score_headlines(research: Dict) -> List[Dict]:
    """Turn headlines into explainable Five Cs adjustments.

    One adjustment per risk bucket at most, so ten articles about the same
    tribunal hearing count once rather than ten times. Each carries the headline
    that caused it so a credit officer can check the claim.
    """
    headlines = (research or {}).get("headlines") or []
    if not headlines:
        return []

    is_sample = research.get("mode") == "offline_sample"
    adjustments: List[Dict] = []

    for bucket, config in RISK_BUCKETS.items():
        match: Optional[Dict] = None
        matched_term = ""
        for headline in headlines:
            text = _normalise_title(headline.get("title", ""))
            for keyword in config["keywords"]:
                if _mentions(text, keyword):
                    match, matched_term = headline, keyword
                    break
            if match:
                break

        if not match:
            continue

        source = match.get("source") or "news"
        adjustments.append(
            {
                "source": "Live News" if not is_sample else "Sample News",
                "source_type": "secondary_research",
                "signal": bucket,
                "factor": config["factor"],
                "points": config["points"],
                "reason": (
                    f"News mentions '{matched_term}': \"{match.get('title', '')[:130]}\" "
                    f"({source})"
                ),
                "evidence_source": source,
                "evidence_link": match.get("link", ""),
                "evidence_snippet": match.get("title", ""),
                "is_sample": is_sample,
            }
        )
        if len(adjustments) >= MAX_NEWS_ADJUSTMENTS:
            break

    return adjustments
