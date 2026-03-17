from __future__ import annotations

import pandas as pd


def reconcile_gst_bank(gst_df: pd.DataFrame, bank_df: pd.DataFrame) -> dict:
    """Compare GST turnover with bank inflows and produce mismatch risk flags."""
    gst_turnover = float(gst_df["taxable_turnover"].sum()) if "taxable_turnover" in gst_df else 0.0
    bank_inflows = float(bank_df["inflow"].sum()) if "inflow" in bank_df else 0.0

    variance = gst_turnover - bank_inflows
    variance_pct = (variance / gst_turnover * 100) if gst_turnover > 0 else 0.0

    mismatch_flag = "OK"
    if abs(variance_pct) > 25:
        mismatch_flag = "High Mismatch"
    elif abs(variance_pct) > 12:
        mismatch_flag = "Moderate Mismatch"

    circular_count = 0
    if {"inflow", "outflow"}.issubset(bank_df.columns):
        circular_count = int(((bank_df["inflow"] > 0) & (bank_df["outflow"] > 0) & ((bank_df["inflow"] - bank_df["outflow"]).abs() < 25000)).sum())

    circular_flag = "Watch"
    if circular_count >= 8:
        circular_flag = "High"
    elif circular_count <= 2:
        circular_flag = "Low"

    inflated_revenue_flag = "Likely No"
    if gst_turnover > bank_inflows * 1.30 and bank_inflows > 0:
        inflated_revenue_flag = "Possible"

    findings_df = pd.DataFrame(
        [
            {
                "Check": "GST Turnover vs Bank Inflows",
                "Value": f"GST INR {gst_turnover:,.0f} | Bank INR {bank_inflows:,.0f}",
                "Status": mismatch_flag,
            },
            {
                "Check": "Possible Circular Trading Pattern",
                "Value": f"{circular_count} near mirror in/out transactions",
                "Status": circular_flag,
            },
            {
                "Check": "Inflated Revenue Indicator",
                "Value": f"Variance {variance_pct:.1f}%",
                "Status": inflated_revenue_flag,
            },
        ]
    )

    overall = "Healthy"
    if mismatch_flag == "High Mismatch" or circular_flag == "High" or inflated_revenue_flag == "Possible":
        overall = "Alert"
    elif mismatch_flag == "Moderate Mismatch" or circular_flag == "Watch":
        overall = "Review"

    return {
        "gst_turnover": gst_turnover,
        "bank_inflows": bank_inflows,
        "variance": variance,
        "variance_pct": variance_pct,
        "mismatch_flag": mismatch_flag,
        "circular_flag": circular_flag,
        "inflated_revenue_flag": inflated_revenue_flag,
        "overall_status": overall,
        "table": findings_df,
    }
