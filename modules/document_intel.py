from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional

# Figures in Indian financial documents look like:
#   "INR 1,240 crore"  "Rs. 45 lakh"  "-45 crore"  "(45) crore"  "₹2.5 bn"
# The sign group also accepts an opening bracket, the accounting convention for
# a negative. Only the digits are mandatory, so the pattern never matches empty.
_AMOUNT_RE = re.compile(
    r"(?P<sign>-|−|\()?\s*"
    r"(?:INR|Rs\.?|₹)?\s*"
    r"(?P<num>\d[\d,]*(?:\.\d+)?)\s*"
    r"\)?\s*"
    r"(?P<unit>crores?|cr|lakhs?|lacs?|million|mn|billion|bn)?",
    re.IGNORECASE,
)

_UNIT_MULTIPLIERS = {
    "crore": 10_000_000,
    "crores": 10_000_000,
    "cr": 10_000_000,
    "lakh": 100_000,
    "lakhs": 100_000,
    "lac": 100_000,
    "lacs": 100_000,
    "million": 1_000_000,
    "mn": 1_000_000,
    "billion": 1_000_000_000,
    "bn": 1_000_000_000,
}

# How far past a keyword to look for its figure.
_SEARCH_WINDOW = 90

LITIGATION_KEYWORDS = ["litigation", "legal notice", "arbitration", "dispute", "court", "penalty"]
GOVERNANCE_KEYWORDS = [
    "related party",
    "resignation",
    "audit qualification",
    "fraud",
    "governance",
    "promoter pledge",
]

_FIELD_KEYWORDS = {
    "revenue": ["revenue", "turnover", "income from operations"],
    "debt": ["total debt", "borrowings", "long term debt"],
    "liabilities": ["total liabilities", "current liabilities", "liabilities"],
    "operating_cash_flow": [
        "operating cash flow",
        "cash flow from operations",
        "net cash from operating activities",
    ],
}

_FIELD_LABELS = {
    "revenue": "Revenue",
    "debt": "Total debt",
    "liabilities": "Total liabilities",
    "operating_cash_flow": "Operating cash flow",
}


@dataclass
class _Candidate:
    value: float
    unit: Optional[str]
    score: float
    evidence: str


def _looks_like_a_year(value: float, unit: Optional[str]) -> bool:
    """Reject bare four-digit years.

    "Revenue for FY2025 was INR 1,240 crore" otherwise resolves to 2025, because
    the year is simply the first digit run after the keyword.
    """
    return unit is None and 1900 <= value <= 2100 and float(value).is_integer()


def _evidence_for(text: str, position: int) -> str:
    """Return the readable span around a matched figure, for the audit trail."""
    start = max(0, position - 45)
    end = min(len(text), position + 55)
    snippet = text[start:end].strip()
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(text) else ""
    return f"{prefix}{snippet}{suffix}"


def _collect_candidates(text: str, keywords: List[str]) -> List[_Candidate]:
    """Find every plausible figure near any of the keywords.

    Returns candidates scored so the caller can pick the best one rather than
    blindly taking the first regex hit.
    """
    lower_text = text.lower()
    candidates: List[_Candidate] = []

    for rank, key in enumerate(keywords):
        key_lower = key.lower()
        cursor = 0
        while True:
            idx = lower_text.find(key_lower, cursor)
            if idx == -1:
                break
            window_start = idx + len(key_lower)
            cursor = window_start
            window = text[window_start : window_start + _SEARCH_WINDOW]

            for match in _AMOUNT_RE.finditer(window):
                raw_number = match.group("num")
                try:
                    magnitude = float(raw_number.replace(",", ""))
                except ValueError:
                    continue

                unit = (match.group("unit") or "").lower() or None
                if _looks_like_a_year(magnitude, unit):
                    continue

                value = magnitude * _UNIT_MULTIPLIERS.get(unit, 1)
                if match.group("sign"):
                    value = -value

                # A figure carrying an explicit unit is far more likely to be the
                # real one; after that, prefer figures sitting closest to the
                # keyword, and earlier (more specific) keywords over later ones.
                score = 0.0
                if unit:
                    score += 100.0
                score -= match.start() * 0.5
                score -= rank * 5.0

                candidates.append(
                    _Candidate(
                        value=value,
                        unit=unit,
                        score=score,
                        evidence=_evidence_for(text, window_start + match.start()),
                    )
                )

    return candidates


def _extract_amount(text: str, keywords: List[str]) -> Dict:
    """Extract the most plausible figure for a field.

    Returns a record rather than a bare number, so a failure to extract is
    visible downstream instead of being silently replaced by a default.
    """
    candidates = _collect_candidates(text, keywords)
    if not candidates:
        return {"value": None, "confidence": "missing", "evidence": "", "candidates": 0}

    # Highest score wins; ties break toward the larger absolute figure, since
    # prior-year comparatives are usually the smaller of two nearby numbers.
    best = max(candidates, key=lambda c: (c.score, abs(c.value)))
    return {
        "value": best.value,
        "confidence": "extracted",
        "evidence": best.evidence,
        "candidates": len(candidates),
    }


def _count_mentions(text: str, keywords: List[str]) -> int:
    lower_text = text.lower()
    return sum(lower_text.count(word.lower()) for word in keywords)


def _find_snippets(text: str, keywords: List[str], max_snippets: int = 3) -> List[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    found = []

    for sentence in sentences:
        low_sentence = sentence.lower()
        if any(k.lower() in low_sentence for k in keywords):
            cleaned = sentence.strip()
            if cleaned:
                found.append(cleaned)
        if len(found) >= max_snippets:
            break

    return found


# Operating cash flow as a share of revenue. Absolute rupee thresholds cannot
# work across borrower sizes: INR 42 crore is thin for an INR 820 crore business
# and comfortable for an INR 200 crore one. Margins below ~6% leave little room
# to service debt; above ~12% is genuinely strong for mid-market lending.
THIN_OCF_MARGIN = 0.06
STRONG_OCF_MARGIN = 0.12

# Used only when revenue could not be extracted and no margin is computable.
THIN_OCF_ABSOLUTE = 50_000_000
STRONG_OCF_ABSOLUTE = 120_000_000


def _cash_flow_indicator(
    operating_cash_flow: Optional[float],
    revenue: Optional[float] = None,
) -> str:
    if operating_cash_flow is None:
        return "Unknown"
    if operating_cash_flow < 0:
        return "Stress"

    if revenue is not None and revenue > 0:
        margin = operating_cash_flow / revenue
        if margin < THIN_OCF_MARGIN:
            return "Tight"
        return "Healthy" if margin >= STRONG_OCF_MARGIN else "Moderate"

    if operating_cash_flow < THIN_OCF_ABSOLUTE:
        return "Tight"
    return "Healthy" if operating_cash_flow >= STRONG_OCF_ABSOLUTE else "Moderate"


def extract_document_signals(report_text: str) -> Dict:
    """Extract financial and qualitative risk signals from unstructured text.

    Financial fields are ``None`` when they could not be found. Nothing is
    invented: a missing figure is reported as missing, and ``data_gaps`` lists
    what the downstream score had to do without. ``extraction`` carries the
    supporting evidence for every figure that was found.
    """
    text = report_text or ""

    extraction: Dict[str, Dict] = {
        field: _extract_amount(text, keywords) for field, keywords in _FIELD_KEYWORDS.items()
    }
    values = {field: record["value"] for field, record in extraction.items()}

    data_gaps = [
        f"{_FIELD_LABELS[field]} could not be extracted from the supplied documents."
        for field, record in extraction.items()
        if record["confidence"] == "missing"
    ]

    return {
        "revenue": values["revenue"],
        "debt": values["debt"],
        "liabilities": values["liabilities"],
        "operating_cash_flow": values["operating_cash_flow"],
        "cash_flow_indicator": _cash_flow_indicator(
            values["operating_cash_flow"], values["revenue"]
        ),
        "litigation_mentions": _count_mentions(text, LITIGATION_KEYWORDS),
        "litigation_snippets": _find_snippets(text, LITIGATION_KEYWORDS),
        "governance_mentions": _count_mentions(text, GOVERNANCE_KEYWORDS),
        "governance_snippets": _find_snippets(text, GOVERNANCE_KEYWORDS),
        "extraction": extraction,
        "data_gaps": data_gaps,
    }
