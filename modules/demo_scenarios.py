from __future__ import annotations

from typing import Dict

import pandas as pd


def _bank_df(records: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    return df


def _gst_df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def get_demo_scenarios() -> Dict[str, Dict]:
    """Return prebuilt borrower cases for hackathon demos."""
    return {
        "Good Borrower": {
            "company_name": "Shivtara Auto Components Pvt Ltd",
            "sector": "Manufacturing",
            "report_text": (
                "Revenue for FY2025 was INR 1,240 crore with stable margin expansion. "
                "Total debt stood at INR 250 crore and total liabilities at INR 560 crore. "
                "Cash flow from operations improved to INR 210 crore. "
                "No material litigation was reported apart from one routine tax matter. "
                "Governance practices remain stable with no adverse audit remarks."
            ),
            "manual_notes": (
                "Plant visit indicates strong utilization and disciplined collections. "
                "Collateral includes factory land, building, and first charge on plant and machinery. "
                "No unsecured exposure requested in initial tranche."
            ),
            "bank_df": _bank_df(
                [
                    {"date": "2025-04-05", "description": "Customer receipts", "inflow": 19500000, "outflow": 6200000, "balance": 84200000},
                    {"date": "2025-05-05", "description": "Customer receipts", "inflow": 20200000, "outflow": 6400000, "balance": 98000000},
                    {"date": "2025-06-05", "description": "Customer receipts", "inflow": 21100000, "outflow": 6700000, "balance": 112000000},
                    {"date": "2025-07-05", "description": "Customer receipts", "inflow": 21800000, "outflow": 7100000, "balance": 126700000},
                    {"date": "2025-08-05", "description": "Customer receipts", "inflow": 22700000, "outflow": 7300000, "balance": 141100000},
                    {"date": "2025-09-05", "description": "Customer receipts", "inflow": 23600000, "outflow": 7600000, "balance": 157100000},
                    {"date": "2025-10-05", "description": "Customer receipts", "inflow": 24300000, "outflow": 7900000, "balance": 173500000},
                    {"date": "2025-11-05", "description": "Customer receipts", "inflow": 24900000, "outflow": 8300000, "balance": 189100000},
                    {"date": "2025-12-05", "description": "Customer receipts", "inflow": 25300000, "outflow": 8600000, "balance": 204800000},
                    {"date": "2026-01-05", "description": "Customer receipts", "inflow": 26000000, "outflow": 9000000, "balance": 221800000},
                    {"date": "2026-02-05", "description": "Customer receipts", "inflow": 26600000, "outflow": 9300000, "balance": 238600000},
                    {"date": "2026-03-05", "description": "Customer receipts", "inflow": 27200000, "outflow": 9800000, "balance": 255900000},
                ]
            ),
            "gst_df": _gst_df(
                [
                    {"month": "Apr-2025", "taxable_turnover": 20500000, "gst_paid": 3690000},
                    {"month": "May-2025", "taxable_turnover": 21200000, "gst_paid": 3816000},
                    {"month": "Jun-2025", "taxable_turnover": 21900000, "gst_paid": 3942000},
                    {"month": "Jul-2025", "taxable_turnover": 22500000, "gst_paid": 4050000},
                    {"month": "Aug-2025", "taxable_turnover": 23100000, "gst_paid": 4158000},
                    {"month": "Sep-2025", "taxable_turnover": 23800000, "gst_paid": 4284000},
                    {"month": "Oct-2025", "taxable_turnover": 24400000, "gst_paid": 4392000},
                    {"month": "Nov-2025", "taxable_turnover": 25100000, "gst_paid": 4518000},
                    {"month": "Dec-2025", "taxable_turnover": 25700000, "gst_paid": 4626000},
                    {"month": "Jan-2026", "taxable_turnover": 26300000, "gst_paid": 4734000},
                    {"month": "Feb-2026", "taxable_turnover": 26900000, "gst_paid": 4842000},
                    {"month": "Mar-2026", "taxable_turnover": 27600000, "gst_paid": 4968000},
                ]
            ),
            "external_notes": "Sector demand steady; no significant promoter adverse media.",
        },
        "Risky Borrower": {
            "company_name": "Metroline Commodity Traders LLP",
            "sector": "Infrastructure",
            "report_text": (
                "The company reported revenue of INR 1,050 crore and total debt of INR 690 crore. "
                "Total liabilities rose to INR 980 crore with cash flow from operations at INR -45 crore. "
                "Multiple litigation matters are ongoing including arbitration, legal notices, court disputes, and penalties. "
                "Disclosures mention related party transactions, promoter pledge concerns, and governance lapses. "
                "Additional legal disputes and litigation exposure remain unresolved."
            ),
            "manual_notes": (
                "Site visit found uneven inventory quality and stretched supplier payments. "
                "Promoter cooperation was limited with resignation concerns; proposed structure is largely unsecured with pledge stress. "
                "Collections show heavy concentration in two counterparties with same-day in-out movements."
            ),
            "bank_df": _bank_df(
                [
                    {"date": "2025-04-04", "description": "Layered receipt", "inflow": 9800000, "outflow": 9705000, "balance": 22000000},
                    {"date": "2025-05-04", "description": "Layered receipt", "inflow": 10200000, "outflow": 10186000, "balance": 22500000},
                    {"date": "2025-06-04", "description": "Layered receipt", "inflow": 10800000, "outflow": 10787000, "balance": 23100000},
                    {"date": "2025-07-04", "description": "Layered receipt", "inflow": 10900000, "outflow": 10886000, "balance": 23700000},
                    {"date": "2025-08-04", "description": "Layered receipt", "inflow": 11300000, "outflow": 11285000, "balance": 24300000},
                    {"date": "2025-09-04", "description": "Layered receipt", "inflow": 11700000, "outflow": 11686000, "balance": 25100000},
                    {"date": "2025-10-04", "description": "Layered receipt", "inflow": 11800000, "outflow": 11786000, "balance": 25800000},
                    {"date": "2025-11-04", "description": "Layered receipt", "inflow": 12200000, "outflow": 12185000, "balance": 26500000},
                    {"date": "2025-12-04", "description": "Layered receipt", "inflow": 12500000, "outflow": 12487000, "balance": 27200000},
                    {"date": "2026-01-04", "description": "Layered receipt", "inflow": 12700000, "outflow": 12686000, "balance": 28000000},
                    {"date": "2026-02-04", "description": "Layered receipt", "inflow": 12900000, "outflow": 12887000, "balance": 28700000},
                    {"date": "2026-03-04", "description": "Layered receipt", "inflow": 13200000, "outflow": 13185000, "balance": 29500000},
                ]
            ),
            "gst_df": _gst_df(
                [
                    {"month": "Apr-2025", "taxable_turnover": 28800000, "gst_paid": 5184000},
                    {"month": "May-2025", "taxable_turnover": 30100000, "gst_paid": 5418000},
                    {"month": "Jun-2025", "taxable_turnover": 31600000, "gst_paid": 5688000},
                    {"month": "Jul-2025", "taxable_turnover": 32500000, "gst_paid": 5850000},
                    {"month": "Aug-2025", "taxable_turnover": 33100000, "gst_paid": 5958000},
                    {"month": "Sep-2025", "taxable_turnover": 34600000, "gst_paid": 6228000},
                    {"month": "Oct-2025", "taxable_turnover": 35700000, "gst_paid": 6426000},
                    {"month": "Nov-2025", "taxable_turnover": 36500000, "gst_paid": 6570000},
                    {"month": "Dec-2025", "taxable_turnover": 37900000, "gst_paid": 6822000},
                    {"month": "Jan-2026", "taxable_turnover": 38800000, "gst_paid": 6984000},
                    {"month": "Feb-2026", "taxable_turnover": 39500000, "gst_paid": 7110000},
                    {"month": "Mar-2026", "taxable_turnover": 40200000, "gst_paid": 7236000},
                ]
            ),
            "external_notes": "Sector slowdown and delayed government receivables continue to pressure liquidity.",
        },
        "Borderline / Manual Review": {
            "company_name": "Orbin Retail Networks Ltd",
            "sector": "Retail",
            "report_text": (
                "Revenue was INR 820 crore with total debt of INR 360 crore and liabilities of INR 640 crore. "
                "Cash flow from operations remained positive at INR 42 crore but below peers. "
                "One consumer litigation and one tax notice are under resolution. "
                "Governance framework is adequate, though one related party disclosure and elongated receivable cycles were noted."
            ),
            "manual_notes": (
                "Operations are stable but store-level demand is uneven across regions. "
                "Collateral mix includes inventory and limited property support with partial unsecured portion. "
                "Recommend tighter drawing power monitoring and monthly stock statements."
            ),
            "bank_df": _bank_df(
                [
                    {"date": "2025-04-06", "description": "Store settlements", "inflow": 14900000, "outflow": 9100000, "balance": 52500000},
                    {"date": "2025-05-06", "description": "Store settlements", "inflow": 15300000, "outflow": 9500000, "balance": 58300000},
                    {"date": "2025-06-06", "description": "Store settlements", "inflow": 15700000, "outflow": 9800000, "balance": 64200000},
                    {"date": "2025-07-06", "description": "Store settlements", "inflow": 16000000, "outflow": 10300000, "balance": 69900000},
                    {"date": "2025-08-06", "description": "Store settlements", "inflow": 16200000, "outflow": 10700000, "balance": 75400000},
                    {"date": "2025-09-06", "description": "Store settlements", "inflow": 16500000, "outflow": 11100000, "balance": 80800000},
                    {"date": "2025-10-06", "description": "Store settlements", "inflow": 16800000, "outflow": 11400000, "balance": 86200000},
                    {"date": "2025-11-06", "description": "Store settlements", "inflow": 17100000, "outflow": 11700000, "balance": 91600000},
                    {"date": "2025-12-06", "description": "Store settlements", "inflow": 17400000, "outflow": 12000000, "balance": 97000000},
                    {"date": "2026-01-06", "description": "Store settlements", "inflow": 17600000, "outflow": 12400000, "balance": 102200000},
                    {"date": "2026-02-06", "description": "Store settlements", "inflow": 17900000, "outflow": 12800000, "balance": 107300000},
                    {"date": "2026-03-06", "description": "Store settlements", "inflow": 18200000, "outflow": 13300000, "balance": 112200000},
                ]
            ),
            "gst_df": _gst_df(
                [
                    {"month": "Apr-2025", "taxable_turnover": 19500000, "gst_paid": 3510000},
                    {"month": "May-2025", "taxable_turnover": 20100000, "gst_paid": 3618000},
                    {"month": "Jun-2025", "taxable_turnover": 20800000, "gst_paid": 3744000},
                    {"month": "Jul-2025", "taxable_turnover": 21400000, "gst_paid": 3852000},
                    {"month": "Aug-2025", "taxable_turnover": 22000000, "gst_paid": 3960000},
                    {"month": "Sep-2025", "taxable_turnover": 22600000, "gst_paid": 4068000},
                    {"month": "Oct-2025", "taxable_turnover": 23300000, "gst_paid": 4194000},
                    {"month": "Nov-2025", "taxable_turnover": 23900000, "gst_paid": 4302000},
                    {"month": "Dec-2025", "taxable_turnover": 24500000, "gst_paid": 4410000},
                    {"month": "Jan-2026", "taxable_turnover": 25200000, "gst_paid": 4536000},
                    {"month": "Feb-2026", "taxable_turnover": 25800000, "gst_paid": 4644000},
                    {"month": "Mar-2026", "taxable_turnover": 26500000, "gst_paid": 4770000},
                ]
            ),
            "external_notes": "Demand momentum is mixed with margin pressure from discounts.",
        },
    }
