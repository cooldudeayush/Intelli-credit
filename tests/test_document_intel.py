"""Unit tests for financial signal extraction from unstructured text."""

from __future__ import annotations

import pytest

from modules.document_intel import extract_document_signals

CRORE = 10_000_000
LAKH = 100_000


def test_indian_units_are_converted():
    signals = extract_document_signals(
        "Total debt stood at INR 250 crore during the year."
    )
    assert signals["debt"] == 250 * CRORE


def test_lakh_unit_is_converted():
    signals = extract_document_signals("Total debt of INR 45 lakh remains outstanding.")
    assert signals["debt"] == 45 * LAKH


def test_comma_separated_numbers_parse():
    signals = extract_document_signals("Total debt stood at INR 1,240 crore.")
    assert signals["debt"] == 1240 * CRORE


@pytest.mark.parametrize(
    "text,flag",
    [
        ("Cash flow from operations was INR 2 crore.", "Tight"),
        ("Cash flow from operations improved to INR 210 crore.", "Healthy"),
    ],
)
def test_cash_flow_banding(text, flag):
    assert extract_document_signals(text)["cash_flow_indicator"] == flag


def test_litigation_and_governance_counting():
    signals = extract_document_signals(
        "There is ongoing litigation and an arbitration matter. "
        "A legal notice was served. Related party transactions were disclosed "
        "alongside a promoter pledge."
    )
    assert signals["litigation_mentions"] >= 3
    assert signals["governance_mentions"] >= 2
    assert signals["litigation_snippets"], "Snippets must back the count"


def test_empty_input_does_not_crash():
    signals = extract_document_signals("")
    assert signals["litigation_mentions"] == 0
    assert signals["governance_mentions"] == 0


# ---------------------------------------------------------------------------
# Regression guards for the defects fixed in Phase 1.
# Each of these failed against the Phase 0 baseline.
# ---------------------------------------------------------------------------


def test_fiscal_year_is_not_mistaken_for_revenue():
    signals = extract_document_signals(
        "Revenue for FY2025 was INR 1,240 crore with stable margin expansion."
    )
    assert signals["revenue"] == 1240 * CRORE, (
        f"Extracted {signals['revenue']:,.0f} — the year, not the revenue"
    )


def test_negative_amounts_keep_their_sign():
    """Regression: the old capture group could not match a leading minus, so
    the Risky Borrower's -45 crore OCF read as +450,000,000 and earned a +2
    Capacity boost instead of a -8 penalty."""
    signals = extract_document_signals(
        "Cash flow from operations was INR -45 crore for the year."
    )
    assert signals["operating_cash_flow"] < 0, (
        f"Parsed as {signals['operating_cash_flow']:,.0f} — sign was dropped"
    )
    assert signals["cash_flow_indicator"] == "Stress"


def test_missing_figures_are_reported_not_invented():
    """Text with no financials must not yield confident numbers."""
    signals = extract_document_signals(
        "The board met on Tuesday and approved the minutes of the previous meeting."
    )
    assert signals["revenue"] is None
    assert signals["extraction"]["revenue"]["confidence"] == "missing"
    assert signals["data_gaps"], "A missing figure must be reported as a data gap"


def test_extracted_figures_carry_their_evidence():
    """Every figure must be traceable back to the text it came from."""
    signals = extract_document_signals(
        "Revenue for FY2025 was INR 1,240 crore with stable margin expansion."
    )
    record = signals["extraction"]["revenue"]
    assert record["confidence"] == "extracted"
    assert "1,240" in record["evidence"]


def test_accounting_parentheses_are_read_as_negative():
    signals = extract_document_signals("Cash flow from operations was INR (45) crore.")
    assert signals["operating_cash_flow"] < 0


def test_unknown_cash_flow_is_not_reported_as_healthy():
    """A missing OCF must not silently present as a healthy one."""
    signals = extract_document_signals("The board approved the minutes.")
    assert signals["cash_flow_indicator"] == "Unknown"
