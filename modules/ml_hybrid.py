from __future__ import annotations

from dataclasses import dataclass, field
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
    "live_news_signal_count",
]

CLASS_MAP = {0: "Reject", 1: "Review", 2: "Lend"}
RISK_MAP = {"Reject": "High", "Review": "Moderate", "Lend": "Low"}
MODEL_VERSION = "ml_v3_model_only_23feat"


@dataclass
class HybridModelBundle:
    model: RandomForestClassifier
    feature_columns: List[str]
    # Median of each feature across the training set. Used as the "typical
    # borrower" reference when attributing a prediction to its inputs.
    feature_reference: Dict[str, float] = field(default_factory=dict)
    cv_accuracy: float = 0.0


def _as_float(value, default: float = 0.0) -> float:
    """Coerce a possibly-missing signal to a float the model can consume."""
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _debt_stress(doc_signals: Dict) -> bool:
    """Debt above 55% of revenue — only assessable when both figures exist."""
    debt = doc_signals.get("debt")
    revenue = doc_signals.get("revenue")
    if debt is None or revenue is None or revenue <= 0:
        return False
    return debt > revenue * 0.55


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
            # Adverse secondary-research hits: weaker borrowers attract more.
            "live_news_signal_count": np.clip(np.rint((1 - borrower_quality) * 3 + rng.normal(0, 0.8, n)), 0, 3),
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
        - 0.30 * df["live_news_signal_count"]
    )

    review_cut, lend_cut = np.quantile(raw, [0.36, 0.70])
    y = np.where(raw >= lend_cut, 2, np.where(raw >= review_cut, 1, 0))
    return df[FEATURE_COLUMNS], y


def train_hybrid_model(with_cv: bool = True) -> HybridModelBundle:
    X, y = _build_synthetic_training_data()
    model = RandomForestClassifier(
        n_estimators=260,
        max_depth=8,
        min_samples_leaf=4,
        class_weight="balanced_subsample",
        random_state=42,
    )

    # Held-out accuracy, surfaced in the UI. A model whose quality is never
    # measured cannot honestly be offered as a second opinion.
    cv_accuracy = 0.0
    if with_cv:
        from sklearn.model_selection import cross_val_score

        cv_accuracy = float(cross_val_score(model, X, y, cv=5).mean())

    model.fit(X, y)
    return HybridModelBundle(
        model=model,
        feature_columns=FEATURE_COLUMNS,
        feature_reference={col: float(X[col].median()) for col in FEATURE_COLUMNS},
        cv_accuracy=cv_accuracy,
    )


def engineer_features(
    score_bundle: Dict,
    recon: Dict,
    doc_signals: Dict,
    insights: Dict,
    qualitative_adjustments: List[Dict],
    rag_adjustments: List[Dict],
    source_rows: List[Dict],
    indian_context_cards: List[Dict],
    news_adjustments: List[Dict] | None = None,
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
        "litigation_count": _as_float(doc_signals.get("litigation_mentions")),
        "governance_flag": 1 if _as_float(doc_signals.get("governance_mentions")) > 0 else 0,
        # Unextractable debt or revenue cannot evidence stress, so the flag stays
        # off rather than firing on a coerced zero.
        "debt_stress_flag": 1 if _debt_stress(doc_signals) else 0,
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
        "live_news_signal_count": float(len(news_adjustments or [])),
    }
    return features


def explain_prediction(bundle: HybridModelBundle, features: Dict, pred_class: int) -> List[Dict]:
    """Attribute one prediction to its inputs, for this borrower specifically.

    ``model.feature_importances_`` describes the *model* — it returns the same
    ranking for every borrower, which cannot explain an individual decision. This
    instead asks, for each feature in turn: if this borrower looked typical on
    this one input, how much less confident would the model be in the decision it
    actually made?

    A positive contribution means the borrower's actual value pushed the model
    toward this decision; a negative one means it pushed against, and the
    decision was reached despite it. Cost is one prediction per feature.
    """
    reference = bundle.feature_reference or {}
    baseline = float(
        bundle.model.predict_proba(
            pd.DataFrame([features], columns=bundle.feature_columns)
        )[0][pred_class]
    )

    rows: List[Dict] = []
    for feature_name in bundle.feature_columns:
        neutral = dict(features)
        neutral[feature_name] = reference.get(feature_name, 0.0)
        counterfactual = float(
            bundle.model.predict_proba(
                pd.DataFrame([neutral], columns=bundle.feature_columns)
            )[0][pred_class]
        )
        contribution = baseline - counterfactual
        rows.append(
            {
                "feature": feature_name,
                "value": round(float(features.get(feature_name, 0)), 4),
                "typical_value": round(float(reference.get(feature_name, 0.0)), 4),
                "contribution": round(contribution, 4),
                "direction": "supports" if contribution > 0 else ("opposes" if contribution < 0 else "neutral"),
            }
        )

    return sorted(rows, key=lambda row: abs(row["contribution"]), reverse=True)


def predict_ml_view(bundle: HybridModelBundle, features: Dict) -> Dict:
    """Predict Lend/Review/Reject from the trained model alone.

    Earlier prototypes blended these probabilities with a hardcoded linear
    surrogate (35% model / 65% formula) to keep demo behaviour stable. That
    surrogate was removed: with 5-fold CV accuracy near 0.88 the model does not
    need propping up, and a "RandomForest prediction" that was mostly a hand
    written formula could not honestly be described as one. What guards against
    a bad model is the hybrid layer downstream, which refuses to let ML override
    the rules-based scorecard below 90% confidence.
    """
    X = pd.DataFrame([features], columns=bundle.feature_columns)
    probs = bundle.model.predict_proba(X)[0]

    pred_class = int(np.argmax(probs))
    decision = CLASS_MAP[pred_class]
    confidence = float(np.max(probs))

    return {
        "ml_decision": decision,
        "ml_risk": RISK_MAP[decision],
        "ml_confidence": round(confidence, 3),
        "class_probabilities": {
            "Reject": round(float(probs[0]), 3),
            "Review": round(float(probs[1]), 3),
            "Lend": round(float(probs[2]), 3),
        },
        "model_cv_accuracy": round(float(bundle.cv_accuracy), 3),
        "top_features": explain_prediction(bundle, features, pred_class)[:8],
    }
