from __future__ import annotations

from typing import Dict, List


RULES = [
    {
        "signal": "Factory operating below capacity",
        "keywords": ["below capacity", "40% capacity", "under-utilized", "underutilized"],
        "adjustments": [("Capacity", -3), ("Conditions", -2)],
        "reason": "Lower asset utilization can weaken repayment resilience.",
    },
    {
        "signal": "Collateral available",
        "keywords": ["collateral", "industrial land", "factory land", "property", "plant"],
        "adjustments": [("Collateral", 2)],
        "reason": "Tangible collateral improves recovery comfort.",
    },
    {
        "signal": "Promoter concern",
        # Credit officers write these observations in prose, not in the tool's
        # vocabulary. The list covers how the concern is actually phrased on a
        # site visit report: limited cooperation, a resignation, a share pledge.
        "keywords": [
            "promoter concern",
            "promoter issue",
            "reputation concern",
            "integrity concern",
            "resignation",
            "resigned",
            "promoter pledge",
            "pledge stress",
            "pledged shares",
            "cooperation was limited",
            "limited cooperation",
            "evasive",
            "promoter dispute",
            "diversion of funds",
        ],
        "adjustments": [("Character", -3)],
        "reason": "Promoter behavior risk affects Character assessment.",
    },
    {
        "signal": "Statutory dues delayed",
        "keywords": [
            "statutory dues",
            "pf dues",
            "esi dues",
            "gst dues",
            "tds default",
            "delayed statutory",
        ],
        "adjustments": [("Character", -2), ("Capacity", -2)],
        "reason": "Delayed statutory payments are an early liquidity and conduct warning.",
    },
    {
        "signal": "Stock statement discrepancy",
        "keywords": [
            "stock statement discrepancy",
            "stock discrepancy",
            "inventory discrepancy",
            "drawing power shortfall",
        ],
        "adjustments": [("Collateral", -3), ("Character", -1)],
        "reason": "Security cover reported does not reconcile with what was inspected.",
    },
    {
        "signal": "Promoter reputation satisfactory",
        "keywords": ["promoter reputation satisfactory", "management quality strong", "strong management quality"],
        "adjustments": [("Character", 2)],
        "reason": "Positive management quality supports Character.",
    },
    {
        "signal": "Receivable concentration high",
        "keywords": [
            "receivable concentration",
            "receivables concentration",
            "customer concentration",
            "concentration in two",
            "concentration in a few",
            "concentrated in two",
            "heavy concentration",
            "single buyer",
            "top two customers",
        ],
        "adjustments": [("Capacity", -2)],
        "reason": "Concentrated receivables can increase cash flow volatility.",
    },
    {
        "signal": "Labour unrest",
        "keywords": ["labour unrest", "labor unrest", "strike", "industrial action"],
        "adjustments": [("Conditions", -3)],
        "reason": "Operational disruption risk from labour issues.",
    },
    {
        "signal": "Regulatory issue under review",
        "keywords": ["regulatory issue", "regulatory notice", "under review", "compliance issue"],
        "adjustments": [("Conditions", -3)],
        "reason": "Regulatory uncertainty increases operating risk.",
    },
    {
        "signal": "Stable operations",
        "keywords": ["stable operations", "operations are stable", "order book", "disciplined collections"],
        "adjustments": [("Capacity", 2), ("Conditions", 1)],
        "reason": "Operational stability supports repayment confidence.",
    },
]


def extract_qualitative_adjustments(primary_notes: str, external_notes: str) -> List[Dict]:
    """Map qualitative note signals to explainable Five Cs score adjustments."""
    adjustments: List[Dict] = []

    note_sources = [
        ("Primary Due Diligence", primary_notes or ""),
        ("External Intelligence", external_notes or ""),
    ]

    for source_name, text in note_sources:
        lower_text = text.lower()
        if not lower_text.strip():
            continue

        for rule in RULES:
            if any(keyword in lower_text for keyword in rule["keywords"]):
                for factor, points in rule["adjustments"]:
                    adjustments.append(
                        {
                            "source": source_name,
                            "signal": rule["signal"],
                            "factor": factor,
                            "points": points,
                            "reason": rule["reason"],
                        }
                    )

    # Deduplicate repeated hits from multiple keywords in same rule/source/factor.
    unique = {}
    for item in adjustments:
        key = (item["source"], item["signal"], item["factor"])
        unique[key] = item

    return list(unique.values())
