from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


FEATURE_COLUMNS = [
    "character_score",
    "capacity_score",
    "capital_score",
    "collateral_score",
    "conditions_score",
    "total_score",
    "gst_bank_mismatch_ratio",
    "circular_trading_flag",
    "litigation_count",
    "governance_flag",
    "debt_stress_flag",
    "operational_stress_flag",
    "sector_risk_flag",
    "promoter_risk_flag",
    "shareholding_concentration_flag",
    "mca_compliance_flag",
    "qualitative_positive_adjustment",
    "qualitative_negative_adjustment",
    "rag_signal_count",
    "structured_signal_count",
    "external_signal_count",
    "primary_due_diligence_signal_count",
]

CLASS_MAP = {0: "Reject", 1: "Review", 2: "Lend"}
RISK_MAP = {"Reject": "High", "Review": "Moderate", "Lend": "Low"}
MODEL_VERSION = "ml_v2_anchor"


@dataclass
class HybridModelBundle:
    model: RandomForestClassifier
    feature_columns: List[str]


def _build_synthetic_training_data(n: int = 700, seed: int = 42) -> Tuple[pd.DataFrame, np.ndarray]:
    """Synthetic prototype training data for stable local demo behavior.

    The samples are anchored on a latent borrower quality score so the model
    learns a sensible separation between lend, review, and reject cases.
    """
    rng = np.random.default_rng(seed)
    borrower_quality = rng.uniform(0, 1, n)

    df = pd.DataFrame(
        {
            "character_score": np.clip(6 + borrower_quality * 13 + rng.normal(0, 1.4, n), 3, 20),
            "capacity_score": np.clip(5 + borrower_quality * 13 + rng.normal(0, 1.6, n), 3, 20),
            "capital_score": np.clip(5 + borrower_quality * 12 + rng.normal(0, 1.5, n), 3, 20),
            "collateral_score": np.clip(5 + borrower_quality * 12 + rng.normal(0, 1.7, n), 3, 20),
            "conditions_score": np.clip(4 + borrower_quality * 11 + rng.normal(0, 1.8, n), 2, 20),
            "gst_bank_mismatch_ratio": np.clip((1 - borrower_quality) * 32 + rng.normal(0, 4.5, n), 0, 45),
            "circular_trading_flag": rng.binomial(1, np.clip(0.58 - 0.65 * borrower_quality, 0.03, 0.75)),
            "litigation_count": np.clip(np.rint((1 - borrower_quality) * 6 + rng.normal(0, 1.1, n)), 0, 9),
            "governance_flag": rng.binomial(1, np.clip(0.50 - 0.55 * borrower_quality, 0.02, 0.65)),
            "debt_stress_flag": rng.binomial(1, np.clip(0.54 - 0.60 * borrower_quality, 0.03, 0.70)),
            "operational_stress_flag": rng.binomial(1, np.clip(0.48 - 0.52 * borrower_quality, 0.04, 0.65)),
            "sector_risk_flag": rng.binomial(1, np.clip(0.42 - 0.30 * borrower_quality, 0.08, 0.50)),
            "promoter_risk_flag": rng.binomial(1, np.clip(0.46 - 0.40 * borrower_quality, 0.04, 0.60)),
            "shareholding_concentration_flag": rng.binomial(1, np.clip(0.36 - 0.24 * borrower_quality, 0.05, 0.45)),
            "mca_compliance_flag": rng.binomial(1, np.clip(0.34 - 0.24 * borrower_quality, 0.05, 0.42)),
            "qualitative_positive_adjustment": np.clip(np.rint(borrower_quality * 4 + rng.normal(0, 1.0, n)), 0, 6),
            "qualitative_negative_adjustment": np.clip(np.rint((1 - borrower_quality) * 5 + rng.normal(0, 1.1, n)), 0, 7),
            "rag_signal_count": np.clip(np.rint((1 - borrower_quality) * 4 + rng.normal(0, 1.0, n)), 0, 6),
            "structured_signal_count": np.clip(np.rint(3 + borrower_quality * 3 + rng.normal(0, 1.0, n)), 1, 8),
            "external_signal_count": np.clip(np.rint((1 - borrower_quality) * 3 + rng.normal(0, 1.0, n)), 0, 6),
            "primary_due_diligence_signal_count": np.clip(np.rint((1 - borrower_quality) * 3 + rng.normal(0, 0.9, n)), 0, 5),
        }
    )

    df["total_score"] = (
        df[["character_score", "capacity_score", "capital_score", "collateral_score", "conditions_score"]].sum(axis=1)
        + df["qualitative_positive_adjustment"] * 0.4
        - df["qualitative_negative_adjustment"] * 0.7
        - df["rag_signal_count"] * 0.4
    ).clip(0, 100)

    raw = (
        0.085 * df["total_score"]
        - 0.045 * df["gst_bank_mismatch_ratio"]
        - 1.25 * df["circular_trading_flag"]
        - 0.28 * df["litigation_count"]
        - 0.90 * df["governance_flag"]
        - 0.95 * df["debt_stress_flag"]
        - 0.80 * df["operational_stress_flag"]
        - 0.45 * df["sector_risk_flag"]
        - 0.55 * df["promoter_risk_flag"]
        - 0.25 * df["shareholding_concentration_flag"]
        - 0.25 * df["mca_compliance_flag"]
        + 0.18 * df["qualitative_positive_adjustment"]
        - 0.28 * df["qualitative_negative_adjustment"]
        - 0.18 * df["rag_signal_count"]
    )

    review_cut, lend_cut = np.quantile(raw, [0.36, 0.70])
    y = np.where(raw >= lend_cut, 2, np.where(raw >= review_cut, 1, 0))
    return df[FEATURE_COLUMNS], y


def train_hybrid_model() -> HybridModelBundle:
    X, y = _build_synthetic_training_data()
    model = RandomForestClassifier(
        n_estimators=260,
        max_depth=8,
        min_samples_leaf=4,
        class_weight="balanced_subsample",
        random_state=42,
    )
    model.fit(X, y)
    return HybridModelBundle(model=model, feature_columns=FEATURE_COLUMNS)


def engineer_features(
    score_bundle: Dict,
    recon: Dict,
    doc_signals: Dict,
    insights: Dict,
    qualitative_adjustments: List[Dict],
    rag_adjustments: List[Dict],
    source_rows: List[Dict],
    indian_context_cards: List[Dict],
) -> Dict:
    scores = score_bundle.get("five_cs_scores", {})

    pos_adj = sum(max(0, float(x.get("points", 0))) for x in qualitative_adjustments)
    neg_adj = sum(abs(min(0, float(x.get("points", 0)))) for x in qualitative_adjustments)

    source_count = len(source_rows)
    used_count = sum(1 for s in source_rows if str(s.get("Used in Analysis", "No")).lower() == "yes")
    structured_count = sum(1 for s in source_rows if "structured" in str(s.get("Type", "")).lower() and str(s.get("Used in Analysis", "No")).lower() == "yes")

    primary_signal_count = sum(1 for q in qualitative_adjustments if q.get("source") == "Primary Due Diligence")
    external_signal_count = sum(1 for q in qualitative_adjustments if q.get("source") == "External Intelligence")

    share_flag = 0
    mca_flag = 0
    for card in indian_context_cards:
        ind = str(card.get("Indicator", "")).lower()
        status = str(card.get("Status", "")).lower()
        if "shareholding concentration" in ind and status in {"high", "elevated"}:
            share_flag = 1
        if "mca compliance" in ind and status in {"watch", "high", "elevated"}:
            mca_flag = 1

    features = {
        "character_score": float(scores.get("Character", 0)),
        "capacity_score": float(scores.get("Capacity", 0)),
        "capital_score": float(scores.get("Capital", 0)),
        "collateral_score": float(scores.get("Collateral", 0)),
        "conditions_score": float(scores.get("Conditions", 0)),
        "total_score": float(score_bundle.get("total_score", 0)),
        "gst_bank_mismatch_ratio": abs(float(recon.get("variance_pct", 0))),
        "circular_trading_flag": 1 if recon.get("circular_flag") in {"High", "Watch"} else 0,
        "litigation_count": float(doc_signals.get("litigation_mentions", 0)),
        "governance_flag": 1 if float(doc_signals.get("governance_mentions", 0)) > 0 else 0,
        "debt_stress_flag": 1 if float(doc_signals.get("debt", 0)) > float(doc_signals.get("revenue", 1)) * 0.55 else 0,
        "operational_stress_flag": 1 if doc_signals.get("cash_flow_indicator") in {"Stress", "Tight"} else 0,
        "sector_risk_flag": 1 if insights.get("regulatory_risk") == "Moderate" else 0,
        "promoter_risk_flag": 1 if insights.get("promoter_news_level") == "Moderate" else 0,
        "shareholding_concentration_flag": share_flag,
        "mca_compliance_flag": mca_flag,
        "qualitative_positive_adjustment": float(pos_adj),
        "qualitative_negative_adjustment": float(neg_adj),
        "rag_signal_count": float(len(rag_adjustments)),
        "structured_signal_count": float(structured_count if structured_count > 0 else min(used_count, max(1, source_count // 3))),
        "external_signal_count": float(external_signal_count),
        "primary_due_diligence_signal_count": float(primary_signal_count),
    }
    return features


def predict_ml_view(bundle: HybridModelBundle, features: Dict) -> Dict:
    X = pd.DataFrame([features], columns=bundle.feature_columns)
    model_probs = bundle.model.predict_proba(X)[0]

    # Anchor the prototype ML view to an interpretable surrogate score so demo
    # scenarios remain directionally sensible even without real training data.
    anchored_score = (
        0.10 * float(features.get("total_score", 0))
        - 0.05 * float(features.get("gst_bank_mismatch_ratio", 0))
        - 1.20 * float(features.get("circular_trading_flag", 0))
        - 0.28 * float(features.get("litigation_count", 0))
        - 0.90 * float(features.get("governance_flag", 0))
        - 0.90 * float(features.get("debt_stress_flag", 0))
        - 0.75 * float(features.get("operational_stress_flag", 0))
        - 0.35 * float(features.get("sector_risk_flag", 0))
        - 0.45 * float(features.get("promoter_risk_flag", 0))
        - 0.22 * float(features.get("shareholding_concentration_flag", 0))
        - 0.22 * float(features.get("mca_compliance_flag", 0))
        + 0.18 * float(features.get("qualitative_positive_adjustment", 0))
        - 0.25 * float(features.get("qualitative_negative_adjustment", 0))
        - 0.15 * float(features.get("rag_signal_count", 0))
        + 0.05 * float(features.get("structured_signal_count", 0))
    )

    if anchored_score >= 6.6:
        anchor_probs = np.array([0.04, 0.18, 0.78])
    elif anchored_score >= 4.8:
        anchor_probs = np.array([0.12, 0.70, 0.18])
    else:
        anchor_probs = np.array([0.82, 0.14, 0.04])

    probs = 0.35 * model_probs + 0.65 * anchor_probs
    probs = probs / probs.sum()
    pred_class = int(np.argmax(probs))
    decision = CLASS_MAP[pred_class]
    confidence = float(np.max(probs))

    importances = bundle.model.feature_importances_
    importance_rows = []
    for idx, feat in enumerate(bundle.feature_columns):
        value = float(features.get(feat, 0))
        direction = 1.0 if feat in {
            "character_score",
            "capacity_score",
            "capital_score",
            "collateral_score",
            "conditions_score",
            "total_score",
            "qualitative_positive_adjustment",
            "structured_signal_count",
        } else -1.0
        importance_rows.append(
            {
                "feature": feat,
                "importance": round(float(importances[idx]), 4),
                "value": round(value, 4),
                "impact_hint": round(direction * float(importances[idx]) * abs(value), 4),
            }
        )
    importance_rows = sorted(importance_rows, key=lambda x: x["importance"], reverse=True)

    return {
        "ml_decision": decision,
        "ml_risk": RISK_MAP[decision],
        "ml_confidence": round(confidence, 3),
        "class_probabilities": {
            "Reject": round(float(probs[0]), 3),
            "Review": round(float(probs[1]), 3),
            "Lend": round(float(probs[2]), 3),
        },
        "top_features": importance_rows[:8],
    }
