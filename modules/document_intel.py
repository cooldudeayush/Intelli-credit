from __future__ import annotations

import re
from typing import Dict, List


def _extract_amount(text: str, keywords: List[str], default_value: float) -> float:
    """Extract approximate amount near keywords, fallback to realistic mock values."""
    lower_text = text.lower()
    for key in keywords:
        pattern = rf"{key}[^\n\r]{{0,60}}?(\d[\d,]*\.?\d*)\s*(crore|cr|lakh|million|bn|billion)?"
        match = re.search(pattern, lower_text, flags=re.IGNORECASE)
        if not match:
            continue

        raw_num = match.group(1).replace(",", "")
        unit = (match.group(2) or "").lower()
        value = float(raw_num)

        if unit in {"crore", "cr"}:
            return value * 10_000_000
        if unit in {"lakh"}:
            return value * 100_000
        if unit in {"million"}:
            return value * 1_000_000
        if unit in {"bn", "billion"}:
            return value * 1_000_000_000
        return value

    return default_value


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


def extract_document_signals(report_text: str) -> Dict:
    """Simulate document intelligence by extracting financial and qualitative risk signals."""
    text = report_text or ""

    revenue = _extract_amount(text, ["revenue", "turnover", "income from operations"], 950_000_000)
    debt = _extract_amount(text, ["total debt", "borrowings", "long term debt"], 280_000_000)
    liabilities = _extract_amount(text, ["total liabilities", "current liabilities", "liabilities"], 510_000_000)
    operating_cash_flow = _extract_amount(
        text,
        ["operating cash flow", "cash flow from operations", "net cash from operating activities"],
        110_000_000,
    )

    litigation_keywords = ["litigation", "legal notice", "arbitration", "dispute", "court", "penalty"]
    governance_keywords = ["related party", "resignation", "audit qualification", "fraud", "governance", "promoter pledge"]

    litigation_mentions = _count_mentions(text, litigation_keywords)
    governance_mentions = _count_mentions(text, governance_keywords)

    cash_flow_flag = "Healthy"
    if operating_cash_flow < 0:
        cash_flow_flag = "Stress"
    elif operating_cash_flow < 50_000_000:
        cash_flow_flag = "Tight"

    return {
        "revenue": revenue,
        "debt": debt,
        "liabilities": liabilities,
        "operating_cash_flow": operating_cash_flow,
        "cash_flow_indicator": cash_flow_flag,
        "litigation_mentions": litigation_mentions,
        "litigation_snippets": _find_snippets(text, litigation_keywords),
        "governance_mentions": governance_mentions,
        "governance_snippets": _find_snippets(text, governance_keywords),
    }
