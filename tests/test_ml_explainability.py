"""Tests for per-instance ML attribution (Phase 4.1) and the anchor removal (4.2)."""

from __future__ import annotations

import inspect

import pytest

from modules.ml_hybrid import FEATURE_COLUMNS, explain_prediction, predict_ml_view


def _features(**overrides) -> dict:
    base = {col: 0.0 for col in FEATURE_COLUMNS}
    base.update(
        {
            "character_score": 16.0,
            "capacity_score": 16.0,
            "capital_score": 16.0,
            "collateral_score": 16.0,
            "conditions_score": 16.0,
            "total_score": 80.0,
            "structured_signal_count": 4.0,
        }
    )
    base.update(overrides)
    return base


def test_attribution_covers_every_feature(model_bundle):
    rows = explain_prediction(model_bundle, _features(), pred_class=2)
    assert {row["feature"] for row in rows} == set(FEATURE_COLUMNS)


def test_attribution_is_ranked_by_magnitude(model_bundle):
    rows = explain_prediction(model_bundle, _features(), pred_class=2)
    magnitudes = [abs(row["contribution"]) for row in rows]
    assert magnitudes == sorted(magnitudes, reverse=True)


def test_attribution_differs_between_borrowers(model_bundle):
    """The whole point of per-instance attribution: two borrowers, two stories.

    Global ``feature_importances_`` returns an identical ranking for everyone,
    which cannot explain an individual decision.
    """
    strong = predict_ml_view(model_bundle, _features())
    weak = predict_ml_view(
        model_bundle,
        _features(
            character_score=2.0,
            capacity_score=3.0,
            total_score=25.0,
            litigation_count=7.0,
            governance_flag=1.0,
            circular_trading_flag=1.0,
            gst_bank_mismatch_ratio=40.0,
        ),
    )
    assert strong["ml_decision"] != weak["ml_decision"]

    strong_top = [row["feature"] for row in strong["top_features"][:5]]
    weak_top = [row["feature"] for row in weak["top_features"][:5]]
    assert strong_top != weak_top, "Attribution did not vary between borrowers"


def test_a_feature_can_oppose_the_decision(model_bundle):
    """An otherwise-strong borrower carrying one bad signal should show it.

    The model may still say Lend, but the attribution must admit which input
    argued against that conclusion.
    """
    view = predict_ml_view(
        model_bundle,
        _features(litigation_count=6.0, governance_flag=1.0),
    )
    directions = {row["direction"] for row in view["top_features"]}
    assert "opposes" in directions or view["ml_decision"] != "Lend"


def test_attribution_records_the_comparison_baseline(model_bundle):
    for row in explain_prediction(model_bundle, _features(), pred_class=2):
        assert "typical_value" in row
        assert row["direction"] in {"supports", "opposes", "neutral"}


# ---------------------------------------------------------------------------
# Phase 4.2 — the hardcoded surrogate that used to dominate the prediction
# ---------------------------------------------------------------------------


def test_prediction_comes_from_the_model_alone(model_bundle):
    """The ML view must be the model's, not a hand-written formula's.

    Earlier code returned ``0.35 * model_probs + 0.65 * anchor_probs``, where the
    anchor was a hardcoded linear score. Calling the output a RandomForest
    prediction was then not accurate.
    """
    features = _features()
    view = predict_ml_view(model_bundle, features)

    import pandas as pd

    raw = model_bundle.model.predict_proba(
        pd.DataFrame([features], columns=model_bundle.feature_columns)
    )[0]

    assert view["class_probabilities"]["Lend"] == pytest.approx(raw[2], abs=1e-3)
    assert view["class_probabilities"]["Review"] == pytest.approx(raw[1], abs=1e-3)
    assert view["class_probabilities"]["Reject"] == pytest.approx(raw[0], abs=1e-3)


def test_no_anchor_formula_remains_in_the_source():
    """Guard against the surrogate being quietly reintroduced."""
    source = inspect.getsource(predict_ml_view)
    assert "anchor_probs" not in source
    assert "0.35 *" not in source


def test_model_reports_its_own_accuracy(model_bundle):
    """A model offered as a second opinion must state how good it is."""
    view = predict_ml_view(model_bundle, _features())
    assert 0.0 < view["model_cv_accuracy"] <= 1.0
    assert view["model_cv_accuracy"] > 0.7, "Model is too weak to be a useful second opinion"
