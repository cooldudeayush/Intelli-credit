"""Unit tests for the qualitative note -> Five Cs rules engine."""

from __future__ import annotations

import pytest

from modules.qualitative import extract_qualitative_adjustments


def _factors(adjustments):
    return {a["factor"] for a in adjustments}


def test_factory_under_capacity_hits_capacity_and_conditions():
    adj = extract_qualitative_adjustments(
        primary_notes="Factory found operating at 40% capacity during the site visit.",
        external_notes="",
    )
    assert _factors(adj) == {"Capacity", "Conditions"}


def test_collateral_note_boosts_collateral():
    adj = extract_qualitative_adjustments(
        primary_notes="Collateral includes factory land and plant.",
        external_notes="",
    )
    collateral = [a for a in adj if a["factor"] == "Collateral"]
    assert collateral and collateral[0]["points"] > 0


def test_labour_unrest_penalises_conditions():
    adj = extract_qualitative_adjustments("Labour unrest reported at the unit.", "")
    assert any(a["factor"] == "Conditions" and a["points"] < 0 for a in adj)


def test_matching_is_case_insensitive():
    # Avoid words like "plant"/"property" here — they trigger the Collateral rule
    # as well and would make this assertion about two rules instead of one.
    lower = extract_qualitative_adjustments("labour unrest at the unit", "")
    upper = extract_qualitative_adjustments("LABOUR UNREST AT THE UNIT", "")
    assert len(lower) == len(upper) == 1


def test_source_is_attributed():
    primary = extract_qualitative_adjustments("Labour unrest observed.", "")
    external = extract_qualitative_adjustments("", "Labour unrest observed.")
    assert primary[0]["source"] == "Primary Due Diligence"
    assert external[0]["source"] == "External Intelligence"


def test_same_signal_from_both_sources_is_not_double_counted_per_source():
    """Dedup is keyed on (source, signal, factor) — one hit per source, not per keyword."""
    adj = extract_qualitative_adjustments(
        primary_notes="Labour unrest and a strike and industrial action all occurred.",
        external_notes="",
    )
    assert len(adj) == 1, f"Three synonyms produced {len(adj)} adjustments"


def test_empty_notes_produce_nothing():
    assert extract_qualitative_adjustments("", "") == []
    assert extract_qualitative_adjustments(None, None) == []


def test_every_adjustment_is_well_formed():
    adj = extract_qualitative_adjustments(
        "Factory at 40% capacity with labour unrest and a regulatory issue.", ""
    )
    for item in adj:
        assert item["factor"] in {
            "Character",
            "Capacity",
            "Capital",
            "Collateral",
            "Conditions",
        }
        assert isinstance(item["points"], (int, float))
        assert str(item.get("reason", "")).strip()


# ---------------------------------------------------------------------------
# Regression guard for Phase 2: the rules used to match only a narrow set of
# literal phrases, so the Risky Borrower demo produced zero qualitative
# adjustments despite notes describing promoter and concentration risk.
# ---------------------------------------------------------------------------


def test_natural_phrasing_of_real_risk_is_detected():
    adj = extract_qualitative_adjustments(
        primary_notes=(
            "Promoter cooperation was limited with resignation concerns; "
            "collections show heavy concentration in two counterparties."
        ),
        external_notes="",
    )
    assert _factors(adj) & {"Character", "Capacity"}, (
        "Promoter resignation and customer concentration both went undetected"
    )
