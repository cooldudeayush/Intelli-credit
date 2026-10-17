"""Unit tests for GST-vs-bank reconciliation.

This is the project's strongest domain logic, so it gets the most direct coverage.
"""

from __future__ import annotations

import pandas as pd
import pytest

from modules.reconciliation import reconcile_gst_bank


def _gst(total: float, months: int = 12) -> pd.DataFrame:
    return pd.DataFrame({"taxable_turnover": [total / months] * months})


def _bank(inflow_total: float, outflow_total: float = 0.0, rows: int = 12) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "inflow": [inflow_total / rows] * rows,
            "outflow": [outflow_total / rows] * rows,
        }
    )


def test_matched_books_are_clean():
    result = reconcile_gst_bank(_gst(100_000_000), _bank(100_000_000))
    assert result["mismatch_flag"] == "OK"
    assert result["inflated_revenue_flag"] == "Likely No"
    assert result["overall_status"] in {"Healthy", "Review"}


@pytest.mark.parametrize(
    "gst_total,bank_total,expected",
    [
        (100_000_000, 100_000_000, "OK"),          # 0% variance
        (100_000_000, 85_000_000, "Moderate Mismatch"),  # 15%
        (100_000_000, 60_000_000, "High Mismatch"),      # 40%
    ],
)
def test_mismatch_bands(gst_total, bank_total, expected):
    result = reconcile_gst_bank(_gst(gst_total), _bank(bank_total))
    assert result["mismatch_flag"] == expected


def test_revenue_inflation_detected_when_gst_far_exceeds_bank():
    """Declaring far more turnover than ever reaches the bank is the core signal."""
    result = reconcile_gst_bank(_gst(200_000_000), _bank(100_000_000))
    assert result["inflated_revenue_flag"] == "Possible"
    assert result["overall_status"] == "Alert"


def test_circular_trading_detected_on_mirrored_transactions():
    """Money in and straight back out, near-identical amounts, is round-tripping."""
    mirrored = pd.DataFrame(
        {
            "inflow": [10_000_000] * 12,
            "outflow": [9_995_000] * 12,  # within the INR 25,000 tolerance
        }
    )
    result = reconcile_gst_bank(_gst(120_000_000), mirrored)
    assert result["circular_flag"] == "High"


def test_normal_trading_is_not_flagged_as_circular():
    normal = pd.DataFrame({"inflow": [10_000_000] * 12, "outflow": [3_000_000] * 12})
    result = reconcile_gst_bank(_gst(120_000_000), normal)
    assert result["circular_flag"] == "Low"


def test_zero_turnover_does_not_divide_by_zero():
    result = reconcile_gst_bank(_gst(0), _bank(50_000_000))
    assert result["variance_pct"] == 0.0


def test_missing_columns_are_tolerated():
    """Reconciliation must degrade, not explode, on a malformed upload."""
    result = reconcile_gst_bank(pd.DataFrame({"month": ["Apr"]}), pd.DataFrame({"date": ["x"]}))
    assert result["gst_turnover"] == 0.0
    assert result["bank_inflows"] == 0.0


def test_findings_table_is_populated():
    result = reconcile_gst_bank(_gst(100_000_000), _bank(100_000_000))
    assert len(result["table"]) == 3
    assert set(result["table"].columns) == {"Check", "Value", "Status"}
